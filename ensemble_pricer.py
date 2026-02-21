#!/usr/bin/env python3
"""
ensemble_pricer.py — Online Adaptive Ensemble Pricer
=====================================================

Combines multiple pricing models (NIRV Heston-MC, rBergomi, KAN corrector,
PINN surface) into a single adaptive ensemble with online weight learning.

Each model contributes a price estimate. Weights are updated online based
on recent prediction accuracy (exponentially decayed tracking of each
model's residuals vs. market prices).

Architecture:
    Model_i → Price_i  ──┐
                          ├──→  w_1·P_1 + w_2·P_2 + ... + w_n·P_n = Ensemble Price
    Weights (online)  ──┘

Weight update rule (exponential hedge / multiplicative weights):
    w_i(t+1) = w_i(t) · exp(-η · |residual_i(t)|) / Z(t)
    where Z(t) normalizes weights to sum to 1.

This follows the "regret-free" online learning framework:
- No single model dominates across all market conditions
- In bull markets, BSM-based models may win
- In crisis, jump-heavy models dominate
- The ensemble automatically adapts

References:
    Cesa-Bianchi & Lugosi (2006) — "Prediction, Learning, and Games"
    Bergmeir et al. (2024) — "Ensemble Methods for Financial Forecasting"
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple, Any


class EnsemblePricer:
    """
    Adaptive ensemble that combines multiple pricing models.

    Usage:
        ensemble = EnsemblePricer()
        ensemble.register_model('heston_mc', heston_pricer)
        ensemble.register_model('rbergomi', rbergomi_pricer)
        ensemble.register_model('kan', kan_pricer)

        # Get ensemble price
        price, diagnostics = ensemble.price(spot, strike, T, ...)

        # Update weights based on realized market price
        ensemble.update(market_price=132.50)
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        decay: float = 0.95,
        min_weight: float = 0.01,
    ):
        """
        Parameters
        ----------
        learning_rate : float — multiplicative weight update rate (η)
        decay : float — exponential decay for old observations
        min_weight : float — minimum weight floor (prevents starvation)
        """
        self.learning_rate = learning_rate
        self.decay = decay
        self.min_weight = min_weight

        self.models: Dict[str, Any] = {}
        self.weights: Dict[str, float] = {}
        self.cumulative_loss: Dict[str, float] = {}
        self.n_updates: int = 0
        self._last_predictions: Dict[str, float] = {}
        self._last_ensemble_price: Optional[float] = None
        self._history: List[Dict] = []

    def register_model(self, name: str, model: Any, initial_weight: float = 1.0):
        """
        Register a pricing model.

        The model must have a callable interface:
            price = model.price(spot, strike, T, r, q, sigma, ...) → float or (float, ...)
        OR be a callable: price = model(spot, strike, T, r, q, sigma)
        """
        self.models[name] = model
        self.weights[name] = initial_weight
        self.cumulative_loss[name] = 0.0

    def price(
        self,
        spot: float,
        strike: float,
        T: float,
        r: float = 0.065,
        q: float = 0.012,
        sigma: float = 0.15,
        option_type: str = 'CE',
        **kwargs,
    ) -> Tuple[float, Dict]:
        """
        Compute weighted ensemble price.

        Parameters
        ----------
        spot, strike, T, r, q, sigma : standard option parameters
        option_type : 'CE' or 'PE'
        **kwargs : additional args passed to each model

        Returns
        -------
        (ensemble_price, diagnostics)
            diagnostics includes per-model prices, weights, and confidence
        """
        if not self.models:
            return 0.0, {'error': 'no models registered'}

        predictions = {}
        errors = {}

        for name, model in self.models.items():
            try:
                result = self._call_model(
                    model, spot, strike, T, r, q, sigma, option_type, **kwargs
                )
                if isinstance(result, tuple):
                    price_val = result[0]
                else:
                    price_val = float(result)

                if np.isfinite(price_val) and price_val >= 0:
                    predictions[name] = price_val
            except Exception as e:
                errors[name] = str(e)

        if not predictions:
            return 0.0, {'error': 'all models failed', 'errors': errors}

        # Normalize weights for active models
        active_weights = {k: self.weights.get(k, 1.0) for k in predictions}
        total_w = sum(active_weights.values())
        if total_w <= 0:
            total_w = len(active_weights)
            active_weights = {k: 1.0 for k in predictions}

        norm_weights = {k: v / total_w for k, v in active_weights.items()}

        # Weighted ensemble
        ensemble_price = sum(
            norm_weights[k] * predictions[k] for k in predictions
        )

        # Store for later update
        self._last_predictions = predictions
        self._last_ensemble_price = ensemble_price

        diagnostics = {
            'ensemble_price': float(ensemble_price),
            'model_prices': {k: round(float(v), 4) for k, v in predictions.items()},
            'model_weights': {k: round(float(v), 4) for k, v in norm_weights.items()},
            'n_active_models': len(predictions),
            'n_updates': self.n_updates,
        }

        if errors:
            diagnostics['model_errors'] = errors

        # Spread-based confidence: tighter spread → higher confidence
        if len(predictions) >= 2:
            prices = list(predictions.values())
            spread = max(prices) - min(prices)
            mean_price = np.mean(prices)
            rel_spread = spread / max(mean_price, 1e-8)
            diagnostics['model_agreement'] = round(float(1.0 - min(rel_spread, 1.0)), 4)

        return float(ensemble_price), diagnostics

    def update(self, market_price: float):
        """
        Update model weights based on realized market price.

        Uses multiplicative weight update (exponential hedge):
            w_i → w_i · exp(-η · |predicted_i - market|)
        Then normalize.
        """
        if not self._last_predictions:
            return

        self.n_updates += 1

        for name, pred in self._last_predictions.items():
            abs_residual = abs(pred - market_price)

            # Robust relative error: scale by max of price and recent std to
            # avoid OTM options (small prices) causing weight ping-pong.
            _price_scale = max(abs(market_price), 1.0)
            if len(self._history) >= 5:
                _recent_prices = [h['market_price'] for h in self._history[-20:]]
                _price_scale = max(_price_scale, float(np.std(_recent_prices) + 1e-6))
            rel_residual = abs_residual / _price_scale

            # Decay old losses
            self.cumulative_loss[name] = (
                self.decay * self.cumulative_loss.get(name, 0.0) + rel_residual
            )

            # Multiplicative weight update with tanh clipping to prevent
            # a single large residual from permanently killing a good model.
            _clipped_residual = float(np.tanh(rel_residual))
            self.weights[name] = max(
                self.weights.get(name, 1.0) * np.exp(-self.learning_rate * _clipped_residual),
                self.min_weight,
            )

        # Normalize
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}

        # Track history
        self._history.append({
            'market_price': market_price,
            'predictions': dict(self._last_predictions),
            'weights_after': dict(self.weights),
        })

        # Trim history
        if len(self._history) > 500:
            self._history = self._history[-200:]

    def get_model_rankings(self) -> List[Tuple[str, float, float]]:
        """
        Get models ranked by current weight.

        Returns list of (name, weight, cumulative_loss) sorted by weight descending.
        """
        rankings = []
        for name in self.models:
            w = self.weights.get(name, 0.0)
            loss = self.cumulative_loss.get(name, 0.0)
            rankings.append((name, float(w), float(loss)))
        return sorted(rankings, key=lambda x: -x[1])

    def get_recent_performance(self, n: int = 20) -> Dict[str, float]:
        """Get mean absolute error of each model over last n updates."""
        if not self._history:
            return {}

        recent = self._history[-n:]
        mae = {}
        for name in self.models:
            errors = []
            for h in recent:
                if name in h.get('predictions', {}):
                    errors.append(abs(h['predictions'][name] - h['market_price']))
            if errors:
                mae[name] = float(np.mean(errors))
        return mae

    # ------------------------------------------------------------------
    def _call_model(self, model, spot, strike, T, r, q, sigma, option_type, **kwargs):
        """Flexibly call a model — handles various API signatures."""
        if callable(model) and not hasattr(model, 'price'):
            return model(spot, strike, T, r, q, sigma, option_type)

        if hasattr(model, 'price'):
            try:
                return model.price(
                    spot=spot, strike=strike, T=T, r=r, q=q,
                    sigma=sigma, option_type=option_type, **kwargs,
                )
            except TypeError:
                # Simpler signature
                return model.price(spot, strike, T, r, q, sigma, option_type)

        raise ValueError(f"Model {type(model)} has no callable interface")
