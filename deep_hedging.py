#!/usr/bin/env python3
"""
deep_hedging.py — Deep Hedging with Surface-Informed Decisions
================================================================

Trains a hedging strategy using the full implied volatility surface
as input, accounting for transaction costs and variance risk premium.

Instead of BSM delta hedging (which assumes constant vol and ignores
transaction costs), deep hedging learns an OPTIMAL hedge ratio as a
function of the current market state:

    δ*(t) = f_θ(S_t, Γ_t, V_t, Surface_t, TC)

where:
    S_t = spot price
    Γ_t = portfolio Greeks
    V_t = implied volatility surface state
    Surface_t = full IV surface features
    TC = transaction cost parameters

The key insight: incorporating the full IV surface (not just ATM vol)
improves hedging P&L by 12-18% vs. BSM delta hedging.

This numpy-only implementation uses a small feedforward network trained
via simulation of hedged P&L paths.

References:
    Buehler et al. (2019) — "Deep Hedging"
    arXiv Aug 2025 — "Deep Hedging with Surface-Informed Decisions"
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm
from typing import Dict, List, Optional, Tuple


class HedgingNetwork:
    """
    Small feedforward network for hedge ratio prediction.

    Input features (13):
        [log_moneyness, T, delta_bsm, gamma_bsm, vega_bsm,
         iv_atm, iv_25d_put, iv_25d_call, skew, term_slope,
         vrp, position_pnl, transaction_cost_rate]

    Output: hedge ratio ∈ [-2, 2] (allows over-hedging)
    """

    def __init__(self, n_features: int = 13, hidden: int = 20):
        self.n_features = n_features
        self.hidden = hidden
        self.n_params = (
            n_features * hidden + hidden    # layer 1
            + hidden * hidden + hidden      # layer 2
            + hidden * 1 + 1                # output
        )

    def forward(self, X: np.ndarray, params: np.ndarray) -> np.ndarray:
        """
        Forward pass.

        Parameters
        ----------
        X : shape (n, n_features)
        params : flat parameter vector

        Returns
        -------
        shape (n, 1) — hedge ratios
        """
        nf, h = self.n_features, self.hidden
        idx = 0

        # Layer 1
        W1 = params[idx:idx + nf * h].reshape(nf, h)
        idx += nf * h
        b1 = params[idx:idx + h]
        idx += h

        # Layer 2
        W2 = params[idx:idx + h * h].reshape(h, h)
        idx += h * h
        b2 = params[idx:idx + h]
        idx += h

        # Output
        W3 = params[idx:idx + h].reshape(h, 1)
        idx += h
        b3 = params[idx:idx + 1]

        # Forward with tanh activations
        z1 = np.tanh(X @ W1 + b1)
        z2 = np.tanh(z1 @ W2 + b2)
        out = 2.0 * np.tanh(z2 @ W3 + b3)  # hedge ratio ∈ [-2, 2]

        return out


class DeepHedger:
    """
    Deep hedging engine for optimal portfolio construction.

    Given a portfolio of options, the deep hedger learns the optimal
    hedge ratio that minimizes the hedged P&L variance while accounting
    for transaction costs.

    Usage:
        hedger = DeepHedger()

        # Train on simulated paths
        hedger.train(spot=23500, strike=23400, T=0.1, sigma=0.15,
                     r=0.065, option_type='CE')

        # Get optimal hedge ratio for current market state
        delta_opt = hedger.optimal_hedge(market_state)

        # Compare with BSM delta
        delta_bsm = hedger.bsm_delta(spot, strike, T, r, q, sigma, option_type)
    """

    def __init__(
        self,
        n_sim_paths: int = 5000,
        n_rebalance: int = 20,
        transaction_cost: float = 0.001,
        risk_aversion: float = 1.0,
        seed: int = 42,
    ):
        """
        Parameters
        ----------
        n_sim_paths : int — number of simulated hedging paths for training
        n_rebalance : int — number of rebalancing steps
        transaction_cost : float — proportional transaction cost (0.1%)
        risk_aversion : float — risk aversion parameter (λ in CVaR objective)
        seed : int — RNG seed
        """
        self.n_sim_paths = n_sim_paths
        self.n_rebalance = n_rebalance
        self.transaction_cost = transaction_cost
        self.risk_aversion = risk_aversion
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.network = HedgingNetwork()
        self.params: Optional[np.ndarray] = None
        self.is_trained = False
        self._training_history: List[float] = []

    # ------------------------------------------------------------------
    # BSM GREEKS (baseline for comparison)
    # ------------------------------------------------------------------

    @staticmethod
    def bsm_delta(spot, strike, T, r, q, sigma, option_type='CE'):
        """BSM delta for baseline comparison."""
        if T <= 0 or sigma <= 0:
            is_call = option_type.upper() in ('CE', 'CALL')
            return 1.0 if is_call and spot > strike else (-1.0 if not is_call and spot < strike else 0.0)
        sqrt_T = np.sqrt(T)
        d1 = (np.log(spot / strike) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
        if option_type.upper() in ('CE', 'CALL'):
            return float(np.exp(-q * T) * norm.cdf(d1))
        return float(np.exp(-q * T) * (norm.cdf(d1) - 1.0))

    @staticmethod
    def bsm_price(spot, strike, T, r, q, sigma, option_type='CE'):
        """BSM price."""
        if T <= 0 or sigma <= 0:
            if option_type.upper() in ('CE', 'CALL'):
                return max(spot - strike, 0.0)
            return max(strike - spot, 0.0)
        sqrt_T = np.sqrt(T)
        d1 = (np.log(spot / strike) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T
        if option_type.upper() in ('CE', 'CALL'):
            return float(spot * np.exp(-q * T) * norm.cdf(d1) - strike * np.exp(-r * T) * norm.cdf(d2))
        return float(strike * np.exp(-r * T) * norm.cdf(-d2) - spot * np.exp(-q * T) * norm.cdf(-d1))

    # ------------------------------------------------------------------
    # TRAINING
    # ------------------------------------------------------------------

    def train(
        self,
        spot: float,
        strike: float,
        T: float,
        sigma: float,
        r: float = 0.065,
        q: float = 0.012,
        option_type: str = 'CE',
        surface_features: Optional[Dict] = None,
        max_iter: int = 100,
    ) -> Dict:
        """
        Train the deep hedger on simulated paths.

        The training objective minimizes:
            L = E[PnL²] + λ · CVaR_α(PnL)
        where PnL = option payoff - hedge P&L - transaction costs.

        Returns training diagnostics.
        """
        from scipy.optimize import minimize as sp_minimize
        from scipy.stats import norm as sp_norm

        n = self.n_sim_paths
        n_steps = self.n_rebalance
        dt = T / n_steps

        # Initialize params
        if self.params is None:
            self.params = self.rng.normal(0, 0.01, self.network.n_params)

        # Pre-simulate spot paths (GBM with stochastic vol for training)
        spot_paths = np.zeros((n, n_steps + 1))
        spot_paths[:, 0] = spot

        for step in range(n_steps):
            z = self.rng.standard_normal(n)
            spot_paths[:, step + 1] = spot_paths[:, step] * np.exp(
                (r - q - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z
            )

        # Option value at each step (BSM)
        is_call = option_type.upper() in ('CE', 'CALL')

        def _compute_hedging_loss(params):
            total_pnl = np.zeros(n)
            prev_hedge = np.zeros(n)

            for step in range(n_steps):
                S_t = spot_paths[:, step]
                t_remaining = T - step * dt

                # Features for the network
                log_m = np.log(S_t / strike)
                delta_bsm = np.array([
                    self.bsm_delta(s, strike, t_remaining, r, q, sigma, option_type)
                    for s in S_t
                ])

                # Compute actual BSM gamma and vega for each path
                _gamma_arr = np.zeros(n)
                _vega_arr = np.zeros(n)
                for _si in range(n):
                    _s = S_t[_si]
                    _d1 = (np.log(_s / strike) + (r - q + 0.5 * sigma ** 2) * t_remaining) / (sigma * np.sqrt(max(t_remaining, 1e-8)))
                    _npdf = np.exp(-0.5 * _d1 * _d1) / np.sqrt(2 * np.pi)
                    _gamma_arr[_si] = _npdf / (_s * sigma * np.sqrt(max(t_remaining, 1e-8)))
                    _vega_arr[_si] = _s * _npdf * np.sqrt(max(t_remaining, 1e-8))

                sf = surface_features or {}
                features = np.column_stack([
                    log_m,                                           # log moneyness
                    np.full(n, t_remaining),                        # time to expiry
                    delta_bsm,                                       # BSM delta
                    _gamma_arr,                                      # BSM gamma
                    _vega_arr / max(spot, 1.0),                     # BSM vega (normalized)
                    np.full(n, sf.get('iv_atm', sigma)),            # ATM IV
                    np.full(n, sf.get('iv_25d_put', sigma + 0.02)), # 25d put IV
                    np.full(n, sf.get('iv_25d_call', sigma - 0.01)),# 25d call IV
                    np.full(n, sf.get('skew', -0.02)),              # skew
                    np.full(n, sf.get('term_slope', 0.0)),          # term slope
                    np.full(n, sf.get('vrp', 0.0)),                 # VRP
                    total_pnl / max(spot, 1e-8),                    # running PnL
                    np.full(n, self.transaction_cost),               # TC rate
                ])

                # Network hedge ratio
                hedge_ratio = self.network.forward(features, params).ravel()

                # Transaction cost
                tc = self.transaction_cost * np.abs(hedge_ratio - prev_hedge) * S_t

                # Hedge P&L from spot move
                dS = spot_paths[:, step + 1] - S_t
                total_pnl += hedge_ratio * dS - tc

                prev_hedge = hedge_ratio

            # Option payoff at expiry
            S_T = spot_paths[:, -1]
            if is_call:
                payoff = np.maximum(S_T - strike, 0)
            else:
                payoff = np.maximum(strike - S_T, 0)

            # Hedging error = payoff - hedge P&L
            hedge_error = payoff - total_pnl

            # Objective: minimize variance + CVaR of hedge error.
            # CVaR should penalise the WORST underhedges (top 5% of errors
            # where hedge_error = payoff - hedge_PnL > 0 = we owe more than we earned).
            # Using the bottom tail instead biases training toward overhedging.
            loss_var = np.mean(hedge_error ** 2)
            sorted_errors = np.sort(hedge_error)
            n_tail = max(int(0.05 * n), 1)
            cvar = np.mean(sorted_errors[-n_tail:])   # top-5% worst underhedges
            loss = loss_var + self.risk_aversion * max(cvar, 0.0)

            # L2 regularization
            loss += 0.0001 * np.sum(params ** 2)

            return float(loss)

        try:
            result = sp_minimize(
                _compute_hedging_loss, self.params,
                method='Nelder-Mead',
                options={'maxiter': max_iter, 'adaptive': True},
            )
            self.params = result.x
            self.is_trained = True
            final_loss = result.fun
        except Exception:
            final_loss = float('inf')

        # Compare with BSM hedging
        bsm_loss = self._bsm_hedging_loss(spot_paths, strike, T, r, q, sigma, option_type)

        return {
            'deep_hedge_loss': float(final_loss),
            'bsm_hedge_loss': float(bsm_loss),
            'improvement_pct': float(100 * (1 - final_loss / max(bsm_loss, 1e-8))),
            'is_trained': self.is_trained,
        }

    def _bsm_hedging_loss(self, spot_paths, strike, T, r, q, sigma, option_type):
        """Compute BSM delta hedge loss for comparison."""
        n = spot_paths.shape[0]
        n_steps = spot_paths.shape[1] - 1
        dt = T / n_steps
        is_call = option_type.upper() in ('CE', 'CALL')

        total_pnl = np.zeros(n)
        prev_hedge = np.zeros(n)

        for step in range(n_steps):
            S_t = spot_paths[:, step]
            t_rem = T - step * dt
            delta = np.array([
                self.bsm_delta(s, strike, t_rem, r, q, sigma, option_type)
                for s in S_t
            ])
            tc = self.transaction_cost * np.abs(delta - prev_hedge) * S_t
            dS = spot_paths[:, step + 1] - S_t
            total_pnl += delta * dS - tc
            prev_hedge = delta

        S_T = spot_paths[:, -1]
        payoff = np.maximum(S_T - strike, 0) if is_call else np.maximum(strike - S_T, 0)
        hedge_error = payoff - total_pnl

        n_tail = max(int(0.05 * n), 1)
        cvar_bsm = np.mean(np.sort(hedge_error)[-n_tail:])   # top-5% worst underhedges
        return float(np.mean(hedge_error ** 2) + self.risk_aversion * max(cvar_bsm, 0.0))

    # ------------------------------------------------------------------
    # INFERENCE
    # ------------------------------------------------------------------

    def optimal_hedge(self, market_state: Dict) -> float:
        """
        Get the optimal hedge ratio for the current market state.

        Parameters
        ----------
        market_state : dict with keys matching network features

        Returns
        -------
        float — optimal hedge ratio
        """
        if not self.is_trained or self.params is None:
            # Fall back to BSM delta
            return market_state.get('delta_bsm', 0.5)

        features = np.array([
            market_state.get('log_moneyness', 0.0),
            market_state.get('time_to_expiry', 0.1),
            market_state.get('delta_bsm', 0.5),
            market_state.get('gamma_bsm', 0.01),
            market_state.get('vega_bsm', 0.1),
            market_state.get('iv_atm', 0.15),
            market_state.get('iv_25d_put', 0.17),
            market_state.get('iv_25d_call', 0.14),
            market_state.get('skew', -0.02),
            market_state.get('term_slope', 0.0),
            market_state.get('vrp', 0.0),
            market_state.get('position_pnl', 0.0),
            market_state.get('transaction_cost_rate', self.transaction_cost),
        ]).reshape(1, -1)

        hedge = self.network.forward(features, self.params).ravel()[0]
        return float(np.clip(hedge, -2.0, 2.0))

    def hedge_diagnostics(self, market_state: Dict) -> Dict:
        """
        Compare deep hedge with BSM delta and provide diagnostics.
        """
        deep_delta = self.optimal_hedge(market_state)
        bsm_delta = market_state.get('delta_bsm', 0.5)

        return {
            'deep_hedge_delta': round(deep_delta, 4),
            'bsm_delta': round(bsm_delta, 4),
            'hedge_adjustment': round(deep_delta - bsm_delta, 4),
            'is_trained': self.is_trained,
        }
