#!/usr/bin/env python3
"""
pinn_vol_surface.py — Arbitrage-Free RBF Volatility Surface (PINN-style)
==========================================================================

A numpy/scipy-only constrained RBF model that learns a total-variance surface
w(k,T) = σ²·T satisfying FIVE simultaneous physics constraints:

  1. Data fidelity       — w(k,T) matches observed market implied variances
  2. No-butterfly arb    — full Gatheral (2006) density g(k,T) ≥ 0
                           (corrected from the common truncated approximation)
  3. No-calendar arb     — ∂w/∂T ≥ 0  (total variance non-decreasing in T)
  4. Roger-Lee (2004) wing bound — w(k,T) ≤ 2|k| for |k| ≥ 0.3
                           This is NOT captured by butterfly or calendar alone:
                           a surface with g≥0 and ∂w/∂T≥0 can still violate the
                           Lee bound, implying negative probability mass at wings.
  5. Martingale constraint — risk-neutral density integrates to 1 at each
                           expiry T: ∫ p(k,T) dk = 1  [put-call parity]

Note on "PINN" label:
    The Dupire forward PDE ∂C/∂T = ½σ²_loc·(∂²C/∂k²−∂C/∂k) is an algebraic
    identity of the Gatheral formula — it is satisfied automatically for any
    smooth surface with g>0 and ∂w/∂T>0.  The genuinely new constraints here
    are the Roger-Lee wing bound (4) and the martingale normalization (5), which
    are independent of the butterfly+calendar conditions.

Architecture:
    Input: (log_moneyness k = log(K/F),  time_to_expiry T)
    → RBF layer: Σ_j c_j · exp(−‖x − μ_j‖² / 2σ_j²)
    → Linear output: total_variance  w(k,T) = σ²_impl · T

References:
    Dupire (1994)          — "Pricing with a Smile"
    Gatheral (2006)        — "The Volatility Surface" (g(k) density formula)
    Raissi et al. (2019)   — "Physics-informed neural networks"
    Breeden & Litzenberger (1978) — "Prices of state-contingent claims..."
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from typing import Dict, Optional


# ============================================================================
# RBF NETWORK
# ============================================================================

class RBFNetwork:
    """
    Radial Basis Function network.

    f(x) = Σ_j  w_j · exp(-||x - c_j||² / (2·σ_j²))  + bias

    Parameters:
    - centers c_j: shape (n_centers, input_dim)
    - widths σ_j: shape (n_centers,)
    - weights w_j: shape (n_centers,)
    - bias: scalar
    """

    def __init__(self, n_centers: int = 25, input_dim: int = 2):
        self.n_centers = n_centers
        self.input_dim = input_dim
        # Total params: centers + widths + weights + bias
        self.n_params = n_centers * input_dim + n_centers + n_centers + 1

    def forward(self, X: np.ndarray, params: np.ndarray) -> np.ndarray:
        """
        Evaluate RBF network.

        Parameters
        ----------
        X : shape (n, input_dim) — inputs
        params : shape (n_params,) — all network parameters

        Returns
        -------
        shape (n,) — network output (total variance w)
        """
        nc = self.n_centers
        nd = self.input_dim

        # Parse parameters
        centers = params[:nc * nd].reshape(nc, nd)          # (nc, nd)
        widths = np.abs(params[nc * nd:nc * nd + nc]) + 0.1  # (nc,), ensure positive
        weights = params[nc * nd + nc:nc * nd + 2 * nc]      # (nc,)
        bias = params[-1]

        # Compute RBF activations: exp(-||x - c||² / (2σ²))
        # X: (n, nd), centers: (nc, nd) → distances: (n, nc)
        diff = X[:, np.newaxis, :] - centers[np.newaxis, :, :]  # (n, nc, nd)
        sq_dist = np.sum(diff ** 2, axis=2)                      # (n, nc)
        activations = np.exp(-sq_dist / (2.0 * widths ** 2))     # (n, nc)

        # Output
        output = activations @ weights + bias  # (n,)
        return output

    def gradient_k(self, X: np.ndarray, params: np.ndarray) -> np.ndarray:
        """∂w/∂k — derivative of total variance w.r.t. log-moneyness."""
        eps = 1e-5
        X_plus = X.copy()
        X_plus[:, 0] += eps
        X_minus = X.copy()
        X_minus[:, 0] -= eps
        return (self.forward(X_plus, params) - self.forward(X_minus, params)) / (2 * eps)

    def gradient_T(self, X: np.ndarray, params: np.ndarray) -> np.ndarray:
        """∂w/∂T — derivative of total variance w.r.t. expiry."""
        eps = 1e-5
        X_plus = X.copy()
        X_plus[:, 1] += eps
        X_minus = X.copy()
        X_minus[:, 1] -= eps
        return (self.forward(X_plus, params) - self.forward(X_minus, params)) / (2 * eps)

    def hessian_kk(self, X: np.ndarray, params: np.ndarray) -> np.ndarray:
        """∂²w/∂k² — second derivative w.r.t. log-moneyness."""
        eps = 1e-4
        X_plus = X.copy()
        X_plus[:, 0] += eps
        X_minus = X.copy()
        X_minus[:, 0] -= eps
        f_plus = self.forward(X_plus, params)
        f_center = self.forward(X, params)
        f_minus = self.forward(X_minus, params)
        return (f_plus - 2 * f_center + f_minus) / (eps ** 2)


# ============================================================================
# PINN VOLATILITY SURFACE
# ============================================================================

class PINNVolSurface:
    """
    Arbitrage-free RBF volatility surface with five physics constraints.

    Fits w(k,T) = σ²·T via an RBF network with five simultaneous loss terms:
        1. Data fidelity    — match market implied variances
        2. Butterfly no-arb — full Gatheral (2006) g(k,T) ≥ 0
        3. Calendar no-arb  — ∂w/∂T ≥ 0
        4. Roger-Lee bound  — w(k,T) ≤ 2|k| for |k|≥0.3  (Lee 2004)
        5. Martingale       — ∫p(k,T)dk = 1 at each expiry

    Constraints 4-5 are genuinely new beyond the standard butterfly+calendar:
    a surface satisfying 2-3 can still violate both, yielding implied densities
    with negative mass at the wings or density not integrating to 1.

    Usage:
        pinn = PINNVolSurface()
        pinn.fit(log_moneyness, expiries, market_ivs)
        iv = pinn.get_iv(k=0.0, T=0.05)
        surface = pinn.get_surface(k_grid, T_grid)

    API compatible with ArbFreeSurfaceState for drop-in use.
    """

    def __init__(
        self,
        n_centers: int = 25,
        lambda_butterfly: float = 1.0,
        lambda_calendar: float = 0.5,
        lambda_smooth: float = 0.01,
        lambda_dupire: float = 0.3,
        lambda_martingale: float = 0.5,
        max_iter: int = 200,
    ):
        """
        Parameters
        ----------
        n_centers        : int   — number of RBF centers
        lambda_butterfly : float — weight on Gatheral butterfly loss
        lambda_calendar  : float — weight on calendar no-arb loss
        lambda_smooth    : float — weight on L2 smoothness regularization
        lambda_dupire    : float — weight on Roger-Lee wing upper-bound loss
        lambda_martingale: float — weight on martingale normalization loss
        max_iter         : int   — maximum L-BFGS-B iterations
        """
        self.rbf = RBFNetwork(n_centers=n_centers, input_dim=2)
        self.lambda_butterfly = lambda_butterfly
        self.lambda_calendar = lambda_calendar
        self.lambda_smooth = lambda_smooth
        self.lambda_dupire = lambda_dupire
        self.lambda_martingale = lambda_martingale
        self.max_iter = max_iter
        self.params: Optional[np.ndarray] = None
        self.is_fitted = False
        self.last_diagnostics: Dict = {}

    def fit(
        self,
        log_moneyness: np.ndarray,
        expiries: np.ndarray,
        market_ivs: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Fit PINN volatility surface to market data.

        Parameters
        ----------
        log_moneyness : array of log(K/F) values
        expiries : array of corresponding expiry times (years)
        market_ivs : array of corresponding implied volatilities
        weights : optional importance weights (e.g. by liquidity)

        Returns
        -------
        dict with fitting diagnostics
        """
        k = np.asarray(log_moneyness, dtype=float)
        T = np.asarray(expiries, dtype=float)
        iv = np.asarray(market_ivs, dtype=float)

        # Filter valid data
        valid = np.isfinite(k) & np.isfinite(T) & np.isfinite(iv) & (T > 0) & (iv > 0)
        k, T, iv = k[valid], T[valid], iv[valid]

        if len(k) < 5:
            self.last_diagnostics = {'error': 'insufficient_data', 'n_points': len(k)}
            return self.last_diagnostics

        # Target: total variance w = σ² · T
        w_target = iv ** 2 * T
        X = np.column_stack([k, T])  # (n, 2)

        if weights is None:
            W = np.ones(len(k))
        else:
            W = np.asarray(weights[valid], dtype=float)
        W = W / np.mean(W)  # normalize

        # ── Collocation grids for physics constraints ─────────────────────
        n_colloc = min(200, len(k) * 3)
        rng = np.random.default_rng(42)
        k_c = rng.uniform(k.min() - 0.05, k.max() + 0.05, n_colloc)
        T_c = rng.uniform(max(T.min(), 0.003), T.max() + 0.01, n_colloc)
        X_colloc = np.column_stack([k_c, T_c])

        # Smaller random subset for the Dupire PDE residual
        # (each evaluation does 5 RBF passes, so keep this compact)
        n_pde = min(60, n_colloc // 3)
        pde_idx = rng.choice(n_colloc, n_pde, replace=False)
        X_pde = X_colloc[pde_idx]

        # Structured k×T grid for martingale normalization
        # (need multiple k-points per T level for trapezoidal integration)
        n_T_mart = min(12, max(2, len(np.unique(np.round(T, 2)))))
        n_k_mart = 25
        T_mart_vals = np.linspace(max(T.min(), 0.003), T.max(), n_T_mart)
        k_mart_vals = np.linspace(k.min() - 0.3, k.max() + 0.3, n_k_mart)
        K_mg, T_mg = np.meshgrid(k_mart_vals, T_mart_vals)
        X_mart = np.column_stack([K_mg.ravel(), T_mg.ravel()])

        # Initialize parameters
        self._init_params(k, T, w_target)

        def total_loss(params):
            # 1. Data fidelity loss
            w_pred = self.rbf.forward(X, params)
            data_loss = float(np.mean(W * (w_pred - w_target) ** 2))

            # 2. Butterfly arbitrage loss (g(k) ≥ 0)
            w_c = np.maximum(self.rbf.forward(X_colloc, params), 1e-10)
            dw_dk = self.rbf.gradient_k(X_colloc, params)
            d2w_dk2 = self.rbf.hessian_kk(X_colloc, params)

            # Full Gatheral (2006) no-butterfly density condition:
            #   g(k) = (1 - k·w'/(2w))² - (w')²/4·(1/w + 1/4) + w''/2 ≥ 0
            # The simplified form "1 - k·w'/(2w) + w''/(2w)" was WRONG —
            # it omitted the quadratic -(w')²/4·(1/w+1/4) term, which is the
            # dominant driver of butterfly violations at the wings.
            term1 = (1.0 - k_c * dw_dk / (2.0 * w_c)) ** 2
            term2 = -(dw_dk ** 2) / 4.0 * (1.0 / w_c + 0.25)
            term3 = d2w_dk2 / 2.0
            g = term1 + term2 + term3
            butterfly_loss = float(np.mean(np.maximum(-g, 0.0) ** 2))

            # 3. Calendar arbitrage loss (∂w/∂T ≥ 0)
            dw_dT = self.rbf.gradient_T(X_colloc, params)
            calendar_loss = float(np.mean(np.maximum(-dw_dT, 0.0) ** 2))

            # 4. Positivity constraint (w > 0)
            positivity_loss = float(np.mean(np.maximum(-w_c + 1e-6, 0.0) ** 2))

            # 5. Smoothness regularization (L2 on weights)
            smooth_loss = float(np.mean(params ** 2))

            # 6. Roger-Lee (2004) wing upper-bound: w(k,T) ≤ 2|k| for |k|≥0.3
            #    NOT captured by butterfly or calendar — independent constraint
            roger_lee_loss = self._roger_lee_wing_loss(X_pde, params)

            # 7. Martingale normalization: ∫p(k,T)dk = 1 at each expiry
            martingale_loss = self._martingale_loss(X_mart, params)

            total = (
                data_loss
                + self.lambda_butterfly * butterfly_loss
                + self.lambda_calendar * calendar_loss
                + 10.0 * positivity_loss
                + self.lambda_smooth * smooth_loss
                + self.lambda_dupire * roger_lee_loss
                + self.lambda_martingale * martingale_loss
            )

            return total

        # Optimize
        try:
            result = minimize(
                total_loss,
                self.params,
                method='L-BFGS-B',
                options={'maxiter': self.max_iter, 'ftol': 1e-8},
            )
            self.params = result.x
            opt_loss = result.fun
            n_iter = result.nit
        except Exception as e:
            # Fallback to Nelder-Mead
            try:
                result = minimize(
                    total_loss,
                    self.params,
                    method='Nelder-Mead',
                    options={'maxiter': self.max_iter * 2},
                )
                self.params = result.x
                opt_loss = result.fun
                n_iter = result.nit
            except Exception:
                opt_loss = float('inf')
                n_iter = 0

        self.is_fitted = True

        # Diagnostics
        w_pred = self.rbf.forward(X, self.params)
        iv_pred = np.sqrt(np.maximum(w_pred / T, 1e-10))
        rmse = float(np.sqrt(np.mean((iv_pred - iv) ** 2)))
        max_err = float(np.max(np.abs(iv_pred - iv)))

        # Check no-arb conditions
        w_c = np.maximum(self.rbf.forward(X_colloc, self.params), 1e-10)
        dw_dk = self.rbf.gradient_k(X_colloc, self.params)
        d2w_dk2 = self.rbf.hessian_kk(X_colloc, self.params)
        dw_dT = self.rbf.gradient_T(X_colloc, self.params)

        # Same full Gatheral density as in total_loss — must be consistent
        g = (1.0 - k_c * dw_dk / (2.0 * w_c)) ** 2 \
            - (dw_dk ** 2) / 4.0 * (1.0 / w_c + 0.25) \
            + d2w_dk2 / 2.0

        # Local vol range from Dupire formula
        sigma2_loc_diag = np.where(g > 1e-6, dw_dT / np.maximum(g, 1e-6), np.nan)
        sigma_loc_valid = np.sqrt(np.maximum(
            sigma2_loc_diag[np.isfinite(sigma2_loc_diag)], 0.0
        ))

        # Roger-Lee wing violation and martingale error at converged params
        roger_lee_diag = self._roger_lee_wing_loss(X_pde, self.params)
        martingale_diag = self._martingale_loss(X_mart, self.params)

        self.last_diagnostics = {
            'loss': float(opt_loss),
            'rmse_iv': rmse,
            'max_error_iv': max_err,
            'n_iterations': int(n_iter),
            'n_data_points': len(k),
            'butterfly_violations': int(np.sum(g < -1e-4)),
            'calendar_violations': int(np.sum(dw_dT < -1e-4)),
            'butterfly_ok': bool(np.sum(g < -1e-4) == 0),
            'calendar_ok': bool(np.sum(dw_dT < -1e-4) == 0),
            # New physics-loss diagnostics
            'roger_lee_violation': round(float(roger_lee_diag), 6),
            'martingale_error': round(float(martingale_diag), 6),
            'local_vol_range': [
                round(float(sigma_loc_valid.min()), 4) if len(sigma_loc_valid) else None,
                round(float(sigma_loc_valid.max()), 4) if len(sigma_loc_valid) else None,
            ],
        }
        return self.last_diagnostics

    def get_iv(self, k: float, T: float) -> float:
        """Get implied volatility at (log_moneyness, expiry)."""
        if not self.is_fitted or self.params is None:
            return 0.15  # fallback

        X = np.array([[float(k), max(float(T), 1e-6)]])
        w = self.rbf.forward(X, self.params)[0]
        w = max(float(w), 1e-10)
        iv = np.sqrt(w / max(T, 1e-6))
        return float(np.clip(iv, 0.01, 5.0))

    def get_total_variance(self, k: float, T: float) -> float:
        """Get total variance w(k, T) = σ²·T."""
        if not self.is_fitted or self.params is None:
            return 0.15 ** 2 * T

        X = np.array([[float(k), max(float(T), 1e-6)]])
        w = self.rbf.forward(X, self.params)[0]
        return float(max(w, 1e-10))

    def get_surface(
        self,
        k_grid: np.ndarray,
        T_grid: np.ndarray,
    ) -> np.ndarray:
        """
        Evaluate IV surface on a grid.

        Parameters
        ----------
        k_grid : shape (nk,) — log-moneyness values
        T_grid : shape (nt,) — expiry values

        Returns
        -------
        shape (nt, nk) — IV surface
        """
        if not self.is_fitted or self.params is None:
            return np.full((len(T_grid), len(k_grid)), 0.15)

        K, TT = np.meshgrid(k_grid, T_grid)
        X = np.column_stack([K.ravel(), TT.ravel()])
        w = self.rbf.forward(X, self.params)
        w = np.maximum(w, 1e-10)
        iv = np.sqrt(w / np.maximum(TT.ravel(), 1e-6))
        return np.clip(iv.reshape(len(T_grid), len(k_grid)), 0.01, 5.0)

    def _init_params(self, k: np.ndarray, T: np.ndarray, w: np.ndarray):
        """Smart initialization of RBF parameters from data."""
        nc = self.rbf.n_centers
        rng = np.random.default_rng(42)

        # Place centers on a grid spanning the data
        k_centers = np.linspace(k.min() - 0.05, k.max() + 0.05,
                                int(np.sqrt(nc)) + 1)
        T_centers = np.linspace(max(T.min(), 0.003), T.max() + 0.01,
                                int(np.sqrt(nc)) + 1)
        K_c, T_c = np.meshgrid(k_centers, T_centers)
        centers_grid = np.column_stack([K_c.ravel(), T_c.ravel()])

        # Subsample to nc centers
        if len(centers_grid) > nc:
            idx = rng.choice(len(centers_grid), nc, replace=False)
            centers = centers_grid[idx]
        else:
            # Pad with random centers
            n_extra = nc - len(centers_grid)
            extra = np.column_stack([
                rng.uniform(k.min(), k.max(), n_extra),
                rng.uniform(T.min(), T.max(), n_extra),
            ])
            centers = np.vstack([centers_grid, extra])[:nc]

        # Widths: proportional to spacing
        widths = np.full(nc, 0.3)

        # Weights: small random initialization
        weights = rng.normal(0, np.std(w) / nc, nc)

        # Bias: mean of target
        bias = np.mean(w)

        self.params = np.concatenate([
            centers.ravel(),    # nc * 2
            widths,             # nc
            weights,            # nc
            [bias],             # 1
        ])

    # ------------------------------------------------------------------
    # NEW PINN PHYSICS METHODS
    # ------------------------------------------------------------------

    def _roger_lee_wing_loss(self, X_pde: np.ndarray, params: np.ndarray) -> float:
        """
        Roger-Lee (2004) wing upper-bound loss.

        The Lee moment formula proves that for ANY valid implied vol surface:

            lim_{k→+∞}  w(k,T) / k  ≤  2   (right wing: OTM calls)
            lim_{k→−∞}  w(k,T) / |k| ≤  2   (left wing:  OTM puts)

        Equivalently: w(k,T) ≤ 2|k| for large |k|.

        This constraint is NOT captured by the butterfly or calendar conditions:
        a surface can have g≥0 and ∂w/∂T≥0 everywhere while still violating the
        Lee bound at the wings (implying negative risk-neutral probability mass
        at extreme strikes, which is a model-free arbitrage).

        For typical NSE/Nifty data, the relevant bound is:
            w(k,T)  ≤  2 · |k|  for |k| ≥ 0.5

        We use a soft version: penalise w(k,T) − 2|k| when positive.

        Returns
        -------
        float — mean squared wing violation (zero when all wings are valid)
        """
        k_pts = X_pde[:, 0]
        w_pts = np.maximum(self.rbf.forward(X_pde, params), 1e-10)

        # Roger-Lee upper bound: w ≤ 2|k|   (active only for |k| ≥ 0.3)
        abs_k = np.abs(k_pts)
        wing_mask = abs_k >= 0.3
        if not np.any(wing_mask):
            return 0.0

        violation = np.maximum(w_pts[wing_mask] - 2.0 * abs_k[wing_mask], 0.0)
        return float(np.mean(violation ** 2))

    def _martingale_loss(self, X_mart: np.ndarray, params: np.ndarray) -> float:
        """
        Martingale normalization loss.

        The risk-neutral density in log-moneyness space is (Breeden-Litzenberger):

            p(k, T) = n(d₁) / √w · g(k, T)

        For the pricing measure to be a valid probability measure, the density
        must integrate to 1 at every expiry T:

            ∫ p(k, T) dk = 1

        Violations mean the surface implies a risk-neutral density that does
        not integrate to 1 — a model-free form of put-call parity failure.

        X_mart should be a structured k×T grid (multiple k-points per unique T)
        so that the trapezoidal integral is accurate.

        Returns
        -------
        float — mean squared martingale error across expiry slices
        """
        from scipy.stats import norm as _norm

        k_pts  = X_mart[:, 0]
        T_pts  = X_mart[:, 1]
        w_pts  = np.maximum(self.rbf.forward(X_mart, params), 1e-10)
        dw_dk  = self.rbf.gradient_k(X_mart, params)
        d2w_dk = self.rbf.hessian_kk(X_mart, params)

        sqrt_w = np.sqrt(w_pts)
        d1     = (-k_pts + 0.5 * w_pts) / sqrt_w
        n_d1   = _norm.pdf(d1)

        t1 = (1.0 - k_pts * dw_dk / (2.0 * w_pts)) ** 2
        t2 = -(dw_dk ** 2) / 4.0 * (1.0 / w_pts + 0.25)
        t3 = d2w_dk / 2.0
        g  = t1 + t2 + t3

        # Risk-neutral density (clip g to non-negative)
        p = n_d1 / sqrt_w * np.maximum(g, 0.0)

        # Integrate at each unique T via trapezoidal rule
        unique_T = np.unique(T_pts)
        total = 0.0
        count = 0
        for Ti in unique_T:
            mask = T_pts == Ti
            if np.sum(mask) < 4:
                continue
            k_s = k_pts[mask]
            p_s = p[mask]
            idx = np.argsort(k_s)
            integral = float(np.trapezoid(p_s[idx], k_s[idx]))
            total += (integral - 1.0) ** 2
            count += 1

        return total / max(count, 1)
