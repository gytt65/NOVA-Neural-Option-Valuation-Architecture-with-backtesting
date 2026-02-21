"""
pricer_router.py — Tiered CPU-budgeted pricer routing.
"""

from __future__ import annotations

import time
from typing import Dict, Optional

import numpy as np
from scipy.stats import norm


def _bsm_price(spot, strike, T, r, q, sigma, option_type):
    s = max(float(spot), 1e-8)
    k = max(float(strike), 1e-8)
    t = max(float(T), 1e-6)   # floor at 1e-6 to prevent sqrt overflow
    v = max(float(sigma), 1e-8)
    if t <= 0:
        if option_type.upper() in ("CE", "CALL"):
            return max(s - k, 0.0)
        return max(k - s, 0.0)
    sqrt_t = np.sqrt(t)
    d1 = np.clip(
        (np.log(s / k) + (r - q + 0.5 * v * v) * t) / (v * sqrt_t),
        -38.0, 38.0,   # clamp before norm.cdf to prevent exp overflow
    )
    d2 = d1 - v * sqrt_t
    if option_type.upper() in ("CE", "CALL"):
        return float(s * np.exp(-q * t) * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2))
    return float(k * np.exp(-r * t) * norm.cdf(-d2) - s * np.exp(-q * t) * norm.cdf(-d1))


class RBergomiPricer:
    """
    Rough Bergomi pricer with Cholesky-based fractional Brownian motion.

    Implements a proper rough-volatility model where the log-variance
    follows a fractional Brownian motion with Hurst exponent H ∈ (0, 0.5):

        log V(t) = log V(0) + η · W_H(t) - 0.5 · η² · t^{2H}

    where W_H is an fBm with Hurst parameter H, and η is vol-of-vol.

    The spot price is driven by:
        dS/S = (r - q) dt + √V(t) (ρ dW_H + √(1-ρ²) dW⊥)

    Uses Cholesky decomposition of the fBm covariance matrix for
    exact (not approximate) path generation.  Keeps path count small
    for CPU friendliness in the tiered pricer router context.

    References:
        Bayer, Friz, Gatheral (2016) — "Pricing under rough volatility"
        Bennedsen, Lunde, Pakkanen (2017) — Hybrid scheme for rBergomi
    """

    # Typical Hurst values by regime
    REGIME_HURST = {
        'crisis': 0.05,
        'volatile': 0.08,
        'normal': 0.12,
        'trending': 0.25,
        'calm': 0.35,
    }

    def __init__(
        self,
        hurst: float = 0.12,
        eta: float = 1.2,
        rho: float = -0.7,
        n_paths: int = 2048,
        n_steps: int = 32,
        regime: Optional[str] = None,
    ):
        self.hurst = float(np.clip(hurst, 0.01, 0.49))
        self.eta = float(max(eta, 0.01))
        self.rho = float(np.clip(rho, -0.99, 0.99))
        self.n_paths = int(max(n_paths, 256))
        self.n_steps = int(max(n_steps, 8))
        # Variable Hurst: override with regime-coupled value if provided
        if regime is not None and regime in self.REGIME_HURST:
            self.hurst = self.REGIME_HURST[regime]

    def _fbm_covariance(self, n: int, T: float) -> np.ndarray:
        """
        Build n×n covariance matrix for fBm increments.

        Cov(W_H(t_i) - W_H(t_{i-1}), W_H(t_j) - W_H(t_{j-1}))
        = 0.5 (|t_i - t_{j-1}|^{2H} + |t_{i-1} - t_j|^{2H}
               - 2|t_i - t_j|^{2H})   for i ≠ j

        Var(increment) = (T/n)^{2H}
        """
        H = self.hurst
        dt = T / n
        H2 = 2.0 * H

        cov = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                ti, tj = (i + 1) * dt, (j + 1) * dt
                ti1, tj1 = i * dt, j * dt
                cov[i, j] = 0.5 * (
                    abs(ti - tj1) ** H2
                    + abs(ti1 - tj) ** H2
                    - 2.0 * abs(ti - tj) ** H2
                )
        # Ensure positive-definite (numerical safety)
        cov += np.eye(n) * 1e-10
        return cov

    def _simulate_fbm_increments(self, n: int, T: float, n_paths: int) -> np.ndarray:
        """
        Generate fBm increments via Cholesky decomposition.

        Returns shape (n_paths, n) array of fBm increments.
        """
        cov = self._fbm_covariance(n, T)
        try:
            L = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            # Fallback: add more regularization
            cov += np.eye(n) * 1e-6
            L = np.linalg.cholesky(cov)

        rng = np.random.default_rng()
        Z = rng.standard_normal((n_paths, n))
        return Z @ L.T  # (n_paths, n)

    def _simulate_paths(
        self, spot: float, T: float, r: float, q: float, sigma: float
    ) -> np.ndarray:
        """
        Simulate rough Bergomi price paths.

        Returns terminal spot prices S(T), shape (n_paths,).
        """
        n = self.n_steps
        dt = T / n
        H = self.hurst
        eta = self.eta

        # 1. Generate fBm increments for the variance process
        dW_H = self._simulate_fbm_increments(n, T, self.n_paths)  # (n_paths, n)

        # 2. Generate correlated standard BM increments for the spot
        # Build correlation BEFORE normalizing, to preserve fBm covariance.
        # dW_spot = ρ·dW_H + √(1-ρ²)·dZ  (in fBm-increment space)
        # Then normalize dW_spot for the spot Euler step.
        rng = np.random.default_rng()
        dZ = rng.standard_normal((self.n_paths, n))
        dW_spot_raw = self.rho * dW_H + np.sqrt(1.0 - self.rho ** 2) * dZ
        # Normalize spot increments to unit std per timestep for Euler step
        dW_spot = dW_spot_raw / np.std(dW_spot_raw, axis=0, keepdims=True).clip(1e-8)

        # 3. Build variance paths
        #    log V(t) = log V(0) + η·W_H(t) - 0.5·η²·t^{2H}
        V0 = sigma ** 2
        W_H_cumsum = np.cumsum(dW_H, axis=1)  # (n_paths, n)
        t_grid = np.arange(1, n + 1) * dt       # (n,)

        log_V = np.log(V0) + eta * W_H_cumsum - 0.5 * eta ** 2 * t_grid ** (2 * H)
        # Adaptive variance clipping: ceiling depends on initial variance
        _log_V_ceil = max(np.log(V0) + 6.0, 5.0)  # ~e^6 ≈ 400× initial variance
        V = np.exp(np.clip(log_V, -20, _log_V_ceil))

        # 4. Simulate spot price using Euler scheme
        log_S = np.full(self.n_paths, np.log(spot))
        sqrt_dt = np.sqrt(dt)

        for i in range(n):
            vol_i = np.sqrt(V[:, i].clip(1e-10))
            log_S += (r - q - 0.5 * V[:, i]) * dt + vol_i * sqrt_dt * dW_spot[:, i]

        return np.exp(log_S)

    def price(
        self,
        spot: float,
        strike: float,
        T: float,
        r: float,
        q: float,
        sigma: float,
        option_type: str,
    ) -> tuple:
        """
        Price European option via rough Bergomi Monte Carlo.

        Returns (price, effective_sigma).
        """
        T = max(float(T), 1e-8)
        sigma = max(float(sigma), 1e-8)

        if T < 1e-6:
            intrinsic = _bsm_price(spot, strike, T, r, q, sigma, option_type)
            return intrinsic, sigma

        S_T = self._simulate_paths(spot, T, r, q, sigma)
        discount = np.exp(-r * T)

        if option_type.upper() in ('CE', 'CALL'):
            payoffs = np.maximum(S_T - strike, 0.0)
        else:
            payoffs = np.maximum(strike - S_T, 0.0)

        price = float(discount * np.mean(payoffs))
        price = max(price, 0.0)

        # Effective sigma (implied from the MC price via BSM inversion, approximate)
        eff_sigma = sigma  # keep input sigma as reference
        return price, eff_sigma


class TieredPricerRouter:
    """
    Route pricing across:
      Tier 0: BS from surface IV (fastest)
      Tier 1: Heston COS (fast refinement)
      Tier 2: Heston+Jump QMC MC (selective, budget-aware)
    """

    def __init__(self, default_cpu_budget_ms: float = 20.0):
        self.default_cpu_budget_ms = float(default_cpu_budget_ms)

    def route_price(
        self,
        *,
        spot: float,
        strike: float,
        T: float,
        r: float,
        q: float,
        option_type: str,
        surface_iv: float,
        regime_params: Optional[Dict] = None,
        india_features: Optional[Dict] = None,
        quant_engine=None,
        mc_pricer=None,
        full_chain_mode: bool = False,
        cpu_budget_ms: Optional[float] = None,
        liquidity_score: float = 1.0,
        anomaly_score: float = 0.0,
        mispricing_hint: float = 0.0,
        config: Optional[Dict] = None,
    ) -> Dict[str, float]:
        budget_ms = float(cpu_budget_ms if cpu_budget_ms is not None else self.default_cpu_budget_ms)
        t0 = time.perf_counter()

        out = {
            "price": np.nan,
            "tier_used": "tier0_surface_bs",
            "surface_iv": float(surface_iv),
            "latency_ms": 0.0,
            "bs_price": np.nan,
            "cos_price": np.nan,
            "mc_price": np.nan,
            "mc_std_error": np.nan,
        }

        bs_price = _bsm_price(spot, strike, T, r, q, max(surface_iv, 1e-4), option_type)
        out["bs_price"] = float(bs_price)
        out["price"] = float(bs_price)

        # Optional plug-in model path (kept fully opt-in).
        cfg = config or {}
        model_name = str(cfg.get("model", "")).strip().lower()
        if model_name == "rbergomi":
            try:
                rb = RBergomiPricer(
                    hurst=float(cfg.get("hurst", 0.12)),
                    eta=float(cfg.get("eta", 1.2)),
                    rho=float(cfg.get("rho", -0.7)),
                )
                rb_price, rb_sigma = rb.price(
                    spot, strike, T, r, q, max(surface_iv, 1e-4), option_type
                )
                if np.isfinite(rb_price) and rb_price > 0:
                    out["price"] = float(rb_price)
                    out["tier_used"] = "tier_rbergomi"
                    out["rbergomi_sigma_eff"] = float(rb_sigma)
                    out["latency_ms"] = float((time.perf_counter() - t0) * 1000.0)
                    return out
            except Exception:
                # Fall through to legacy routing.
                pass

        def _elapsed_ms():
            return (time.perf_counter() - t0) * 1000.0

        def _within_budget(extra_ms: float = 0.0):
            return (_elapsed_ms() + extra_ms) <= budget_ms

        # Tier 1: COS refinement
        cos_price = None
        if quant_engine is not None and hasattr(quant_engine, "heston_cos") and _within_budget(1.0):
            try:
                V0 = max(surface_iv ** 2, 1e-8)
                rp = regime_params or {}
                cos_price = quant_engine.heston_cos.price(
                    spot, strike, T, r, q, V0,
                    rp.get("kappa", 2.0),
                    rp.get("theta_v", V0),
                    rp.get("sigma_v", 0.3),
                    rp.get("rho_sv", -0.5),
                    option_type,
                )
                if np.isfinite(cos_price) and cos_price > 0:
                    out["cos_price"] = float(cos_price)
                    out["price"] = float(cos_price)
                    out["tier_used"] = "tier1_heston_cos"
            except Exception:
                pass

        # Tier 2 eligibility: selective and CPU-budget controlled
        run_tier2 = False
        if mc_pricer is not None and _within_budget(3.0):
            if full_chain_mode:
                # Chain scans: Tier 2 only for exceptional candidates
                run_tier2 = (
                    liquidity_score >= 0.5
                    and abs(mispricing_hint) >= 0.08
                    and anomaly_score <= 0.8
                )
            else:
                run_tier2 = (
                    liquidity_score >= 0.4
                    and anomaly_score <= 0.9
                    and abs(mispricing_hint) >= 0.03
                )

        if run_tier2 and _within_budget(5.0):
            rp = regime_params or {}
            old_n_paths = getattr(mc_pricer, "n_paths", None)
            old_use_sobol = getattr(mc_pricer, "use_sobol", None)
            try:
                # Keep Tier 2 path counts small for CPU-friendliness
                if old_n_paths is not None:
                    mc_pricer.n_paths = int(max(1500, min(6000, old_n_paths // 8)))
                if old_use_sobol is not None:
                    mc_pricer.use_sobol = True

                mc_price, mc_se, _ = mc_pricer.price(
                    spot, strike, T, r, q, max(surface_iv, 1e-4),
                    rp, option_type, india_features or {},
                )
                if np.isfinite(mc_price) and mc_price > 0:
                    out["mc_price"] = float(mc_price)
                    out["mc_std_error"] = float(max(mc_se, 0.0))

                    # Dual-control-variate style blend:
                    # base BS + COS + MC (MC gets largest weight).
                    c = float(cos_price) if (cos_price is not None and np.isfinite(cos_price) and cos_price > 0) else float(bs_price)
                    blended = 0.70 * float(mc_price) + 0.20 * c + 0.10 * float(bs_price)
                    out["price"] = float(max(blended, 0.0))
                    out["tier_used"] = "tier2_qmc_mc"
            except Exception:
                pass
            finally:
                if old_n_paths is not None:
                    mc_pricer.n_paths = old_n_paths
                if old_use_sobol is not None:
                    mc_pricer.use_sobol = old_use_sobol

        out["latency_ms"] = float(_elapsed_ms())
        return out
