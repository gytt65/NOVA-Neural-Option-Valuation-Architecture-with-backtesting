#!/usr/bin/env python3
"""
neural_jsde.py — Neural Jump Stochastic Differential Equation Pricer
=====================================================================

The most advanced pricing model: drift, diffusion, AND jump parameters
are learned as functions of market state via neural networks.

Architecture:
    Traditional:  dS = μ·S·dt + σ·S·dW + J·dN(λ)     [fixed μ, σ, λ, J]
    Neural J-SDE: dS = μ_θ(X)·S·dt + σ_θ(X)·S·dW + J_θ(X)·dN(λ_θ(X))

    Where X = (S, t, VIX, regime_prob, FII_flow, PCR, ...) are market features
    and θ are neural network weights learned from historical option chains.

Training objective:
    Minimize Σ_i |Model_Price(K_i, T_i; θ) - Market_Price(K_i, T_i)|²
    across all strikes K and expiries T simultaneously.

This numpy-only implementation uses a 3-layer RBF network as the
"neural" component, trained via L-BFGS-B.

Key insight: The jump parameters (λ, μ_j, σ_j) become OUTPUT HEADS
of the neural network, allowing the model to learn state-dependent
dynamics that change with VIX, regime, and flows.

References:
    "Neural Jump Stochastic Differential Equation Model" — arXiv 2025
    "Neural SDEs for Option Pricing" — Gierjatowicz et al. 2022
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm
from typing import Dict, List, Optional, Tuple


class NeuralParameterNetwork:
    """
    A small RBF network that maps market state features to SDE parameters.

    Input:  X = [VIX/100, regime_crisis, regime_normal, regime_trending, log_moneyness, T]
    Output: [drift_adj, vol_mult, jump_intensity, jump_mean, jump_std]

    Compact architecture: 6 → 15 RBF centers → 5 outputs
    Total params: 15*6 (centers) + 15 (widths) + 15*5 (weights) + 5 (biases) = 185
    """

    def __init__(self, n_features: int = 6, n_centers: int = 15, n_outputs: int = 5):
        self.n_features = n_features
        self.n_centers = n_centers
        self.n_outputs = n_outputs
        self.n_params = (
            n_centers * n_features   # centers
            + n_centers              # widths
            + n_centers * n_outputs  # weights
            + n_outputs              # biases
        )

    def forward(self, X: np.ndarray, params: np.ndarray) -> np.ndarray:
        """
        Evaluate network.

        Parameters
        ----------
        X : shape (n, n_features)
        params : shape (n_params,)

        Returns
        -------
        shape (n, n_outputs) — raw SDE parameter outputs
        """
        nc, nf, no = self.n_centers, self.n_features, self.n_outputs

        # Parse params
        idx = 0
        centers = params[idx:idx + nc * nf].reshape(nc, nf)
        idx += nc * nf
        widths = np.abs(params[idx:idx + nc]) + 0.1
        idx += nc
        weights = params[idx:idx + nc * no].reshape(nc, no)
        idx += nc * no
        biases = params[idx:idx + no]

        # RBF activations
        diff = X[:, np.newaxis, :] - centers[np.newaxis, :, :]
        sq_dist = np.sum(diff ** 2, axis=2)
        phi = np.exp(-sq_dist / (2.0 * widths ** 2))

        # Output = weighted sum + bias
        return phi @ weights + biases


class NeuralJSDE:
    """
    Neural Jump-SDE pricer.

    The SDE dynamics are parameterized by a neural network:
        μ(X), σ(X), λ(X), μ_j(X), σ_j(X) = NetworkOutput(features)

    where features include VIX, regime probabilities, moneyness, and expiry.

    Usage:
        njsde = NeuralJSDE()

        # Option 1: Price with pre-set parameters (no training needed)
        price = njsde.price(spot, strike, T, r, q, sigma,
                            features={'vix': 15.0, 'regime_crisis': 0.1, ...})

        # Option 2: Calibrate to market data, then price
        njsde.calibrate(spot, strikes, market_prices, T, r, q, features)
        price = njsde.price(spot, strike, T, r, q, sigma, features)

    API compatible with HestonJumpDiffusionPricer.
    """

    # Default SDE parameter bounds (after sigmoid/softplus transform)
    PARAM_DEFAULTS = {
        'drift_adj': 0.0,       # drift adjustment (centered at 0)
        'vol_mult': 1.0,        # vol multiplier (centered at 1)
        'jump_intensity': 3.0,  # λ in annualized terms
        'jump_mean': -0.02,     # mean jump size (slightly negative)
        'jump_std': 0.03,       # jump size std
    }

    def __init__(
        self,
        n_paths: int = 10000,
        n_steps: int = 50,
        n_centers: int = 35,
        seed: int = 42,
    ):
        self.n_paths = n_paths if n_paths % 2 == 0 else n_paths + 1
        self.n_steps = n_steps
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.network = NeuralParameterNetwork(
            n_features=6, n_centers=n_centers, n_outputs=5
        )
        self.params: Optional[np.ndarray] = None
        self.is_calibrated = False

        # Feature names for consistent ordering
        self.feature_names = [
            'vix_norm',        # VIX / 100
            'regime_crisis',   # P(crisis)
            'regime_normal',   # P(normal)
            'regime_trending', # P(trending)
            'log_moneyness',   # log(K/S)
            'time_to_expiry',  # T in years
        ]

    # Feature scale factors for normalization (approximate std per feature)
    _FEATURE_SCALES = np.array([0.10, 0.3, 0.3, 0.3, 0.20, 0.15], dtype=float)
    _FEATURE_MEANS  = np.array([0.15, 0.1, 0.6, 0.2, 0.00, 0.08], dtype=float)

    def _extract_features(self, features: dict, log_m: float = 0.0,
                           T: float = 0.05) -> np.ndarray:
        """Convert feature dict to normalized array (unit-variance per feature)."""
        raw = np.array([
            features.get('vix_norm', features.get('vix', 15.0) / 100.0),
            features.get('regime_crisis', 0.1),
            features.get('regime_normal', 0.6),
            features.get('regime_trending', 0.2),
            log_m,
            T,
        ], dtype=float)
        # Normalize to roughly unit variance for RBF distance computation
        return (raw - self._FEATURE_MEANS) / np.maximum(self._FEATURE_SCALES, 1e-6)

    def _decode_sde_params(self, raw: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Transform raw network outputs to valid SDE parameter ranges.

        Uses sigmoid/softplus to enforce constraints:
        - drift_adj ∈ [-0.5, 0.5]  (tanh)
        - vol_mult ∈ [0.3, 3.0]    (sigmoid scaled)
        - λ ∈ [0.1, 50.0]          (softplus)
        - μ_j ∈ [-0.15, 0.05]      (sigmoid scaled)
        - σ_j ∈ [0.005, 0.10]      (sigmoid scaled)
        """
        drift_adj = 0.5 * np.tanh(raw[:, 0])
        vol_mult = 0.3 + 2.7 / (1.0 + np.exp(-raw[:, 1]))
        lam = 0.1 + 49.9 / (1.0 + np.exp(-raw[:, 2]))
        mu_j = -0.15 + 0.20 / (1.0 + np.exp(-raw[:, 3]))
        sig_j = 0.005 + 0.095 / (1.0 + np.exp(-raw[:, 4]))

        return {
            'drift_adj': drift_adj,
            'vol_mult': vol_mult,
            'lambda_j': lam,
            'mu_j': mu_j,
            'sigma_j': sig_j,
        }

    def _simulate(
        self,
        spot: float,
        T: float,
        r: float,
        q: float,
        sigma: float,
        sde_params: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        Simulate SDE paths with neural-parameterized dynamics.

        Returns terminal spot values S(T).
        """
        n = self.n_paths
        n_steps = max(int(T * 252), 5) if self.n_steps is None else self.n_steps
        dt = T / n_steps
        half_n = n // 2

        # Extract scalar SDE params (use first element if array)
        drift_adj = float(sde_params['drift_adj'][0]) if len(sde_params['drift_adj']) > 0 else 0.0
        vol_mult = float(sde_params['vol_mult'][0]) if len(sde_params['vol_mult']) > 0 else 1.0
        lam = float(sde_params['lambda_j'][0]) if len(sde_params['lambda_j']) > 0 else 3.0
        mu_j = float(sde_params['mu_j'][0]) if len(sde_params['mu_j']) > 0 else -0.02
        sig_j = float(sde_params['sigma_j'][0]) if len(sde_params['sigma_j']) > 0 else 0.03

        # Effective vol
        sigma_eff = sigma * vol_mult

        # Jump compensator
        k_comp = np.exp(mu_j + 0.5 * sig_j ** 2) - 1.0

        # Drift with neural adjustment
        drift = (r - q - lam * k_comp + drift_adj) * dt

        log_S = np.full(n, np.log(max(spot, 1e-8)))

        for step in range(n_steps):
            # Antithetic normals
            z_half = self.rng.standard_normal(half_n)
            z = np.concatenate([z_half, -z_half])

            # Jumps: Poisson counts must be independent (can't negate a count).
            # Use independent draws for both halves; antithetic only applies to
            # the Gaussian jump-size noise for variance reduction.
            n_jumps_half = self.rng.poisson(lam * dt, half_n)
            n_jumps_anti = self.rng.poisson(lam * dt, half_n)
            n_jumps = np.concatenate([n_jumps_half, n_jumps_anti])
            z_jump_half = self.rng.standard_normal(half_n)
            z_jump = np.concatenate([z_jump_half, -z_jump_half])
            jump_sizes = np.where(
                n_jumps > 0,
                n_jumps * mu_j + np.sqrt(np.maximum(n_jumps, 1e-8)) * sig_j * z_jump,
                0.0,
            )

            log_S += drift - 0.5 * sigma_eff ** 2 * dt + sigma_eff * np.sqrt(dt) * z + jump_sizes

        return np.exp(log_S)

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def price(
        self,
        spot: float,
        strike: float,
        T: float,
        r: float = 0.065,
        q: float = 0.012,
        sigma: float = 0.15,
        option_type: str = 'CE',
        features: Optional[dict] = None,
        **kwargs,
    ) -> Tuple[float, float]:
        """
        Price an option using Neural J-SDE.

        Returns (price, std_error).
        """
        if features is None:
            features = {}

        # Get SDE parameters from neural network
        log_m = np.log(max(strike, 1e-8) / max(spot, 1e-8))
        X = self._extract_features(features, log_m, T).reshape(1, -1)

        if self.params is not None:
            raw = self.network.forward(X, self.params)
            sde_params = self._decode_sde_params(raw)
        else:
            # Uncalibrated: use BSM-neutral defaults (no vol scaling,
            # no jumps, no drift adj) so the price converges to BSM.
            sde_params = {
                'drift_adj': np.zeros(1),
                'vol_mult': np.ones(1),       # 1.0 = no scaling
                'lambda_j': np.zeros(1),       # 0 jumps = pure diffusion
                'mu_j': np.zeros(1),
                'sigma_j': np.full(1, 0.01),
            }

        # Reset RNG for reproducibility
        self.rng = np.random.default_rng(self.seed)

        # Simulate
        S_T = self._simulate(spot, T, r, q, sigma, sde_params)

        # Payoffs
        disc = np.exp(-r * T)
        if option_type.upper() in ('CE', 'CALL'):
            payoffs = np.maximum(S_T - strike, 0)
        else:
            payoffs = np.maximum(strike - S_T, 0)

        price_val = float(disc * np.mean(payoffs))
        std_err = float(disc * np.std(payoffs) / np.sqrt(len(payoffs)))

        return max(price_val, 0.0), std_err

    def _mc_loss(
        self,
        params: np.ndarray,
        spot: float,
        strikes: np.ndarray,
        market_prices: np.ndarray,
        T: float,
        r: float,
        q: float,
        sigma: float,
        option_type: str,
        features: dict,
        crn_seed: int,
    ) -> float:
        """
        MC pricing loss evaluated with a fixed random seed (Common Random Numbers).

        Using the same seed for both the θ+ and θ- evaluations in SPSA means
        the MC noise is shared across perturbations and largely cancels out in
        the difference (f+ − f−), yielding a low-variance gradient estimate
        without needing to increase n_paths.
        """
        total = 0.0
        for K, mkt_p in zip(strikes, market_prices):
            log_m = np.log(max(K, 1e-8) / max(spot, 1e-8))
            X = self._extract_features(features, log_m, T).reshape(1, -1)
            raw = self.network.forward(X, params)
            sde_params = self._decode_sde_params(raw)
            # CRN: reset to same seed so noise is identical for θ± evaluations
            self.rng = np.random.default_rng(crn_seed)
            S_T = self._simulate(spot, T, r, q, sigma, sde_params)
            disc = np.exp(-r * T)
            if option_type.upper() in ('CE', 'CALL'):
                payoffs = np.maximum(S_T - K, 0.0)
            else:
                payoffs = np.maximum(K - S_T, 0.0)
            model_p = disc * float(np.mean(payoffs))
            # Relative squared error: prevents ATM options dominating OTM ones
            rel_err = (model_p - mkt_p) / max(abs(mkt_p), 1.0)
            total += rel_err ** 2
        total += 0.001 * float(np.sum(params ** 2))  # L2 regularisation
        return total

    def calibrate(
        self,
        spot: float,
        strikes: np.ndarray,
        market_prices: np.ndarray,
        T: float,
        r: float = 0.065,
        q: float = 0.012,
        sigma: float = 0.15,
        option_type: str = 'CE',
        features: Optional[dict] = None,
        max_iter: int = 50,
    ) -> Dict:
        """
        Calibrate Neural J-SDE to market prices via Adam + SPSA.

        Previous implementation used Nelder-Mead on 185 parameters.
        Nelder-Mead scales as O(n²) per iteration and diverges in high
        dimensions — with 185 parameters and only 50 iterations it never
        reaches a useful minimum.

        Replacement: Adam optimizer with SPSA gradient estimates.
        Key insight: the MC loss uses a fixed random seed (Common Random
        Numbers), so the noise in f(θ+cΔ) and f(θ−cΔ) is identical and
        cancels in the difference, giving a low-variance gradient estimate
        even with only 500 MC paths.

        Cost per step: 2 × n_strikes MC simulations (vs Nelder-Mead's
        O(n_params) ≈ 185 simulations per function evaluation).
        """
        features = features or {}
        strikes = np.asarray(strikes, dtype=float)
        market_prices = np.asarray(market_prices, dtype=float)

        # Filter invalid observations
        valid = np.isfinite(market_prices) & (market_prices > 0) & np.isfinite(strikes)
        strikes = strikes[valid]
        market_prices = market_prices[valid]
        if len(strikes) == 0:
            return {'loss': float('inf'), 'n_strikes': 0, 'is_calibrated': False}

        # Initialise network parameters
        if self.params is None:
            rng_init = np.random.default_rng(42)
            self.params = rng_init.normal(0, 0.1, self.network.n_params)

        # Reduce n_paths for calibration speed (CRN keeps variance low)
        n_paths_orig = self.n_paths
        self.n_paths = min(self.n_paths, 500)

        # ── Adam + SPSA hyper-parameters ────────────────────────────────
        lr      = 0.01
        beta1, beta2, eps_adam = 0.9, 0.999, 1e-8
        c_spsa  = 0.05    # SPSA perturbation size (larger: MC loss is noisy)
        n_param = len(self.params)

        m = np.zeros(n_param)
        v = np.zeros(n_param)
        rng_spsa = np.random.default_rng(self.seed + 999)

        best_loss   = float('inf')
        best_params = self.params.copy()

        for t in range(1, max_iter + 1):
            # Decaying perturbation (Spall 1998 schedule)
            c_t = c_spsa / (t ** 0.161)

            # Rademacher perturbation vector
            delta = rng_spsa.choice([-1.0, 1.0], size=n_param).astype(float)

            # Fixed CRN seed for this step — shared by θ± evaluations
            crn_seed_t = self.seed + t

            f_plus = self._mc_loss(
                self.params + c_t * delta,
                spot, strikes, market_prices, T, r, q, sigma, option_type,
                features, crn_seed_t,
            )
            f_minus = self._mc_loss(
                self.params - c_t * delta,
                spot, strikes, market_prices, T, r, q, sigma, option_type,
                features, crn_seed_t,
            )

            # SPSA gradient estimate: ĝ = (f+ − f−)/(2c) · Δ
            grad = ((f_plus - f_minus) / (2.0 * c_t)) * delta

            # Adam moment updates
            m = beta1 * m + (1.0 - beta1) * grad
            v = beta2 * v + (1.0 - beta2) * grad ** 2
            m_hat = m / (1.0 - beta1 ** t)
            v_hat = v / (1.0 - beta2 ** t)
            self.params -= lr * m_hat / (np.sqrt(v_hat) + eps_adam)

            # Track best (evaluated with same CRN seed for consistency)
            if f_plus < best_loss:
                best_loss   = f_plus
                best_params = self.params.copy()

        self.params     = best_params
        self.n_paths    = n_paths_orig
        self.is_calibrated = True

        return {
            'loss': float(best_loss),
            'n_strikes': int(len(strikes)),
            'is_calibrated': True,
        }

    def get_learned_dynamics(self, features: dict, T: float = 0.05) -> Dict[str, float]:
        """
        Inspect the learned SDE parameters for given market state.

        Returns the human-readable drift, vol, jump parameters
        that the network would use for pricing.
        """
        X = self._extract_features(features, 0.0, T).reshape(1, -1)
        if self.params is not None:
            raw = self.network.forward(X, self.params)
        else:
            raw = np.zeros((1, 5))

        decoded = self._decode_sde_params(raw)
        return {
            'drift_adjustment': float(decoded['drift_adj'][0]),
            'vol_multiplier': float(decoded['vol_mult'][0]),
            'jump_intensity': float(decoded['lambda_j'][0]),
            'jump_mean': float(decoded['mu_j'][0]),
            'jump_std': float(decoded['sigma_j'][0]),
        }
