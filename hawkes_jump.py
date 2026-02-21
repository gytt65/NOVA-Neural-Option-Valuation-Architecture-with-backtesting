#!/usr/bin/env python3
"""
hawkes_jump.py — Hawkes Self-Exciting Jump Process for Option Pricing
=====================================================================

Implements a self-exciting jump model where jumps CLUSTER — each jump
temporarily increases the probability of subsequent jumps, capturing
cascading events (budget → RBI → FII outflow) common in Indian markets.

Key upgrade over Merton (1976) Poisson jumps:
    Merton:  λ(t) = constant
    Hawkes:  λ(t) = λ_base + Σ α·exp(-β·(t - t_i))   for past jump times t_i

Parameters:
    λ_base : baseline jump intensity (jumps/year when no recent activity)
    α      : excitation magnitude (how much each jump boosts intensity)
    β      : decay rate (how fast the excitation fades)

References:
    - Hawkes (1971): "Spectra of some self-exciting and mutually exciting point processes"
    - Aït-Sahalia et al. (2015): "Modeling Financial Contagion Using Mutually Exciting Jump Processes"
    - Compound CARMA-Hawkes (arXiv 2024, University of Milan)
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from typing import Dict, List, Optional, Tuple


# ============================================================================
# HAWKES PROCESS CORE
# ============================================================================

class HawkesProcess:
    """
    Univariate Hawkes process with exponential kernel.

    Intensity:
        λ(t) = μ + Σ_{t_i < t} α · exp(-β · (t - t_i))

    where μ is the base rate, α is the excitation magnitude, and
    β is the decay speed.  Stationarity requires α < β (branching ratio < 1).
    """

    def __init__(self, mu: float = 2.0, alpha: float = 0.5, beta: float = 1.0):
        self.mu = max(float(mu), 1e-6)
        self.alpha = max(float(alpha), 1e-8)
        self.beta = max(float(beta), self.alpha + 1e-6)  # enforce stationarity

    @property
    def branching_ratio(self) -> float:
        """α/β — must be < 1 for stationarity."""
        return self.alpha / max(self.beta, 1e-8)

    @property
    def stationary_mean_intensity(self) -> float:
        """E[λ] = μ / (1 - α/β)."""
        br = self.branching_ratio
        if br >= 1.0:
            return self.mu * 10.0  # non-stationary fallback
        return self.mu / (1.0 - br)

    def intensity(self, t: float, event_times: np.ndarray) -> float:
        """
        Compute Hawkes intensity λ(t) given past event times.

        Parameters
        ----------
        t : float — current time
        event_times : array of past event times (< t)

        Returns
        -------
        float — instantaneous jump intensity λ(t)
        """
        past = event_times[event_times < t]
        if len(past) == 0:
            return self.mu
        excitation = self.alpha * np.sum(np.exp(-self.beta * (t - past)))
        return self.mu + excitation

    def intensity_path(self, times: np.ndarray, event_times: np.ndarray) -> np.ndarray:
        """Compute λ(t) at each time in `times`."""
        result = np.full(len(times), self.mu, dtype=float)
        if len(event_times) == 0:
            return result
        for i, t in enumerate(times):
            past = event_times[event_times < t]
            if len(past) > 0:
                result[i] += self.alpha * np.sum(np.exp(-self.beta * (t - past)))
        return result

    def simulate(self, T: float, seed: Optional[int] = None) -> np.ndarray:
        """
        Simulate Hawkes process on [0, T] via Ogata's thinning algorithm.

        Returns array of event times.
        """
        rng = np.random.default_rng(seed)
        events = []
        t = 0.0
        lambda_star = self.mu  # upper bound on intensity

        while t < T:
            # Generate inter-arrival from exponential(λ*)
            u1 = rng.random()
            dt = -np.log(max(u1, 1e-15)) / lambda_star
            t += dt

            if t >= T:
                break

            # Compute actual intensity
            lam_t = self.intensity(t, np.array(events))

            # Accept/reject
            u2 = rng.random()
            if u2 <= lam_t / lambda_star:
                events.append(t)
                # Update upper bound — must be >= actual intensity after accept
                lambda_star = max(lambda_star, lam_t + self.alpha)
            else:
                lambda_star = max(lam_t, self.mu)

            # Safety: prevent λ* from going below baseline
            lambda_star = max(lambda_star, self.mu)

        return np.array(events)

    def log_likelihood(self, event_times: np.ndarray, T: float) -> float:
        """
        Log-likelihood of the Hawkes process for observed event times on [0, T].

        ℓ(μ, α, β) = Σ log λ(t_i) - ∫₀ᵀ λ(t) dt

        The integral has a closed-form for exponential kernel:
            ∫₀ᵀ λ(t) dt = μ·T + (α/β) Σᵢ [1 - exp(-β(T - tᵢ))]
        """
        if len(event_times) == 0:
            return -self.mu * T

        times = np.sort(event_times)
        n = len(times)

        # Σ log λ(tᵢ)
        sum_log_lam = 0.0
        for i in range(n):
            lam_i = self.mu
            if i > 0:
                past = times[:i]
                lam_i += self.alpha * np.sum(np.exp(-self.beta * (times[i] - past)))
            sum_log_lam += np.log(max(lam_i, 1e-15))

        # Compensator: ∫₀ᵀ λ(t) dt
        integral = self.mu * T
        for i in range(n):
            integral += (self.alpha / self.beta) * (1.0 - np.exp(-self.beta * (T - times[i])))

        return sum_log_lam - integral


# ============================================================================
# HAWKES JUMP ESTIMATOR — Drop-in alongside EMJumpEstimator
# ============================================================================

class HawkesJumpEstimator:
    """
    Estimates self-exciting jump parameters from return data.

    Replaces the simple Merton Poisson assumption with Hawkes clustering:
    - Step 1: Detect jump events from returns (threshold-based)
    - Step 2: Fit Hawkes process parameters (μ, α, β) via MLE
    - Step 3: Estimate jump size distribution (μ_j, σ_j) from detected jumps

    Also provides:
    - `clustering_score`: How much jump clustering is happening RIGHT NOW
    - `conditional_intensity`: Current λ(t) given recent jump history
    - `time_varying_intensity`: λ(t) as a function of VIX (Hawkes + VIX regression)

    Usage:
        hawkes = HawkesJumpEstimator()
        result = hawkes.fit(returns)
        # result['lambda_base'], result['alpha'], result['beta']
        # result['clustering_score'], result['current_intensity']
    """

    def __init__(self, jump_threshold_sigma: float = 2.5, max_iter: int = 200):
        """
        Parameters
        ----------
        jump_threshold_sigma : float
            Returns exceeding mean ± threshold × σ are classified as jumps.
        max_iter : int
            Maximum iterations for MLE optimization.
        """
        self.jump_threshold_sigma = jump_threshold_sigma
        self.max_iter = max_iter
        self.last_fit = {}

    def fit(self, returns: np.ndarray, dt: float = 1.0 / 252) -> Dict:
        """
        Fit Hawkes jump model to return data.

        Parameters
        ----------
        returns : np.ndarray — Daily log returns
        dt      : float — Time step (1/252 for daily)

        Returns
        -------
        dict with keys:
            lambda_base, alpha, beta, branching_ratio,
            mu_j, sigma_j, mu_d, sigma_d,
            clustering_score, current_intensity,
            n_jumps, jump_times_idx
        """
        returns = np.asarray(returns, dtype=float)
        returns = returns[np.isfinite(returns)]
        n = len(returns)

        if n < 30:
            return self._default_params()

        # ── Step 1: Detect jump events ──
        mu_r = np.mean(returns)
        sigma_r = np.std(returns, ddof=1)
        threshold = self.jump_threshold_sigma * sigma_r

        is_jump = np.abs(returns - mu_r) > threshold
        jump_indices = np.where(is_jump)[0]
        n_jumps = len(jump_indices)

        if n_jumps < 3:
            # Too few jumps to fit Hawkes — fall back to Poisson
            return self._poisson_fallback(returns, dt, is_jump, jump_indices)

        # Convert jump indices to "times" (in years)
        jump_times = jump_indices.astype(float) * dt
        T_total = n * dt

        # ── Step 2: Fit Hawkes process via MLE ──
        mu_init = n_jumps / T_total * 0.5  # base rate guess
        alpha_init = n_jumps / T_total * 0.3
        beta_init = alpha_init * 3.0  # ensure stationarity

        def neg_ll(params):
            mu_h, alpha_h, beta_h = params
            if mu_h <= 0 or alpha_h <= 0 or beta_h <= 0:
                return 1e10
            if alpha_h >= beta_h:  # non-stationary
                return 1e10 + (alpha_h - beta_h) * 100

            hp = HawkesProcess(mu=mu_h, alpha=alpha_h, beta=beta_h)
            ll = hp.log_likelihood(jump_times, T_total)
            if not np.isfinite(ll):
                return 1e10
            return -ll

        _opt_result = None
        try:
            _opt_result = minimize(
                neg_ll,
                x0=[mu_init, alpha_init, beta_init],
                method='Nelder-Mead',
                options={'maxiter': self.max_iter, 'xatol': 1e-6, 'fatol': 1e-6}
            )
            mu_h, alpha_h, beta_h = _opt_result.x
            mu_h = max(mu_h, 1e-4)
            alpha_h = max(alpha_h, 1e-6)
            beta_h = max(beta_h, alpha_h + 1e-4)
        except Exception:
            mu_h = mu_init
            alpha_h = alpha_init
            beta_h = beta_init

        # ── Step 3: Jump size distribution ──
        jump_returns = returns[is_jump]
        non_jump_returns = returns[~is_jump]

        mu_j = float(np.mean(jump_returns)) if len(jump_returns) > 0 else -0.01
        sigma_j = float(np.std(jump_returns, ddof=1)) if len(jump_returns) > 1 else 0.03
        mu_d = float(np.mean(non_jump_returns)) if len(non_jump_returns) > 0 else 0.0003
        sigma_d = float(np.std(non_jump_returns, ddof=1)) if len(non_jump_returns) > 1 else 0.01

        # Estimate excess kurtosis of jumps for heavy-tail detection
        _jump_kurtosis = 0.0
        if len(jump_returns) >= 4:
            from scipy.stats import kurtosis as _kurt_fn
            _jump_kurtosis = float(_kurt_fn(jump_returns, fisher=True))
            _jump_kurtosis = max(_jump_kurtosis, 0.0)  # excess kurtosis ≥ 0 for fat tails

        # ── Step 4: Current state ──
        hp = HawkesProcess(mu=mu_h, alpha=alpha_h, beta=beta_h)

        # Current intensity (at the end of the observation window)
        current_intensity = hp.intensity(T_total, jump_times)

        # Clustering score: how much above baseline we are
        baseline = hp.stationary_mean_intensity
        clustering_score = float(np.clip(
            (current_intensity - baseline) / max(baseline, 1e-4), 0.0, 5.0
        ))

        # Branching ratio
        branching_ratio = float(alpha_h / max(beta_h, 1e-8))

        self.last_fit = {
            'lambda_base': float(mu_h),
            'alpha': float(alpha_h),
            'beta': float(beta_h),
            'branching_ratio': branching_ratio,
            'mu_j': mu_j,
            'sigma_j': sigma_j,
            'mu_d': mu_d,
            'sigma_d': sigma_d,
            'lambda_j': float(current_intensity),  # backward compat with EMJumpEstimator
            'jump_prob_daily': float(min(current_intensity * dt, 0.5)),
            'current_intensity': float(current_intensity),
            'stationary_intensity': float(baseline),
            'clustering_score': clustering_score,
            'n_jumps': int(n_jumps),
            'jump_kurtosis': float(_jump_kurtosis),  # excess kurtosis; >0 = fat tails
            'jump_times_idx': jump_indices.tolist(),
            'iterations': int(getattr(_opt_result, 'nit', 0)) if _opt_result is not None else 0,
            'model': 'hawkes',
        }
        return self.last_fit

    def time_varying_intensity(
        self,
        returns: np.ndarray,
        vix_series: np.ndarray,
        dt: float = 1.0 / 252,
    ) -> Dict:
        """
        Combine Hawkes base fit with VIX-dependent modulation.

        λ(t) = [μ + Σ α·exp(-β·(t - t_i))] · (1 + γ · (VIX(t)/VIX_mean - 1))

        Returns
        -------
        dict — base fit + gamma (VIX sensitivity) + lambda_func
        """
        base = self.fit(returns, dt)

        if vix_series is None or len(vix_series) < len(returns):
            return base

        vix = np.asarray(vix_series[-len(returns):], dtype=float)
        vix_mean = float(np.mean(vix[np.isfinite(vix)]))
        if vix_mean <= 0:
            return base

        # Simple regression: jump probability ~ VIX level
        is_jump_array = np.zeros(len(returns))
        if base.get('jump_times_idx'):
            for idx in base['jump_times_idx']:
                if 0 <= idx < len(is_jump_array):
                    is_jump_array[idx] = 1.0

        # Estimate γ via logistic regression on jump indicators
        try:
            vix_norm = vix / vix_mean - 1.0
            # Simple correlation-based gamma
            if np.std(vix_norm) > 1e-6:
                gamma = float(np.clip(
                    np.corrcoef(vix_norm, is_jump_array)[0, 1] * 2.0,
                    -1.0, 3.0
                ))
            else:
                gamma = 0.0
        except Exception:
            gamma = 0.0

        base['vix_gamma'] = gamma
        base['vix_mean'] = vix_mean
        base['lambda_func'] = lambda vix_val: (
            base['current_intensity'] * (1.0 + gamma * (vix_val / vix_mean - 1.0))
        )
        return base

    def clustering_active(self) -> bool:
        """Returns True if jump clustering is currently elevated."""
        return self.last_fit.get('clustering_score', 0.0) > 0.5

    def regime_adjustments(self) -> Dict[str, float]:
        """
        Return parameter multipliers for downstream models based on Hawkes state.

        When clustering is active:
        - Increase jump intensity (lambda_mult)
        - Increase vol-of-vol (eta_mult)
        - Widen confidence intervals (conf_mult)
        """
        cs = self.last_fit.get('clustering_score', 0.0)
        br = self.last_fit.get('branching_ratio', 0.0)

        # Sigmoid-like response to clustering score
        lambda_mult = 1.0 + 0.8 * np.tanh(cs)       # [1.0, 1.8]
        eta_mult = 1.0 + 0.4 * np.tanh(cs)           # [1.0, 1.4]
        conf_mult = 1.0 + 0.3 * min(br, 0.9)         # [1.0, 1.27]
        kappa_mult = 1.0 - 0.2 * np.tanh(cs)         # [0.8, 1.0] — slower mean reversion during clusters

        return {
            'lambda_mult': float(lambda_mult),
            'eta_mult': float(eta_mult),
            'conf_mult': float(conf_mult),
            'kappa_mult': float(kappa_mult),
            'clustering_active': bool(cs > 0.5),
        }

    def _poisson_fallback(
        self,
        returns: np.ndarray,
        dt: float,
        is_jump: np.ndarray,
        jump_indices: np.ndarray,
    ) -> Dict:
        """Fallback to simple Poisson when too few jumps for Hawkes."""
        n = len(returns)
        n_jumps = len(jump_indices)
        T_total = n * dt

        jump_returns = returns[is_jump] if np.any(is_jump) else np.array([-0.01])
        non_jump = returns[~is_jump] if np.any(~is_jump) else returns

        lam = n_jumps / T_total if T_total > 0 else 5.0

        return {
            'lambda_base': float(lam),
            'alpha': 0.0,
            'beta': 1.0,
            'branching_ratio': 0.0,
            'mu_j': float(np.mean(jump_returns)),
            'sigma_j': float(np.std(jump_returns, ddof=1)) if len(jump_returns) > 1 else 0.03,
            'mu_d': float(np.mean(non_jump)),
            'sigma_d': float(np.std(non_jump, ddof=1)) if len(non_jump) > 1 else 0.01,
            'lambda_j': float(lam),
            'jump_prob_daily': float(min(lam * dt, 0.5)),
            'current_intensity': float(lam),
            'stationary_intensity': float(lam),
            'clustering_score': 0.0,
            'n_jumps': int(n_jumps),
            'jump_times_idx': jump_indices.tolist(),
            'iterations': 0,
            'model': 'poisson_fallback',
        }

    @staticmethod
    def _default_params() -> Dict:
        return {
            'lambda_base': 5.0,
            'alpha': 0.5,
            'beta': 1.5,
            'branching_ratio': 0.33,
            'mu_j': -0.01,
            'sigma_j': 0.02,
            'mu_d': 0.0003,
            'sigma_d': 0.01,
            'lambda_j': 5.0,
            'jump_prob_daily': 0.02,
            'current_intensity': 5.0,
            'stationary_intensity': 5.0,
            'clustering_score': 0.0,
            'n_jumps': 0,
            'jump_times_idx': [],
            'iterations': 0,
            'model': 'default',
        }
