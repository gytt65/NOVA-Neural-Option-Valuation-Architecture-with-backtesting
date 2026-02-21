#!/usr/bin/env python3
"""
kan_corrector.py — Kolmogorov-Arnold Network Pricing Corrector
===============================================================

A numpy-only implementation of KAN (Kolmogorov-Arnold Networks) for
option pricing correction. Drop-in replacement for the GradientBoosting-based
MLPricingCorrector with the same API surface.

Key differences from MLP/GBM:
    - Learnable activation functions on EDGES (B-splines), not fixed activations on nodes
    - More interpretable: can extract symbolic pricing rules from learned splines
    - Better extrapolation: splines generalize more naturally than piecewise-constant trees
    - Smaller networks needed: [n_features, 8, 4, 1] suffices vs 200-tree ensembles

Architecture:
    Input → [B-spline edges] → Hidden_1(8) → [B-spline edges] → Hidden_2(4) → [B-spline edges] → Output(1)

Each edge has its own learnable B-spline with k knots, parameterized by
control point coefficients optimized via L-BFGS-B.

References:
    Liu et al. (2024) — "KAN: Kolmogorov-Arnold Networks"
    Bozorgasl & Chen (2024) — "Wav-KAN: Wavelet Kolmogorov-Arnold Networks"
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from typing import Dict, List, Optional, Tuple


# ============================================================================
# B-SPLINE BASIS
# ============================================================================

def _bspline_basis(x: np.ndarray, knots: np.ndarray, degree: int = 3) -> np.ndarray:
    """
    Evaluate B-spline basis functions at points x.

    Parameters
    ----------
    x : shape (n,) — evaluation points (should be in [knots[degree], knots[-degree-1]])
    knots : shape (n_knots,) — augmented knot vector
    degree : int — spline degree (3 = cubic)

    Returns
    -------
    shape (n, n_basis) — basis function values, where n_basis = len(knots) - degree - 1
    """
    n_basis = len(knots) - degree - 1
    n_pts = len(x)
    B = np.zeros((n_pts, n_basis))

    # Degree 0: indicator functions
    for i in range(n_basis + degree):
        if i < len(knots) - 1:
            mask = (x >= knots[i]) & (x < knots[i + 1])
            if i < n_basis:
                B[mask, i] = 1.0

    # Cox-de Boor recursion
    for d in range(1, degree + 1):
        B_new = np.zeros((n_pts, n_basis))
        for i in range(n_basis + degree - d):
            if i < n_basis:
                # Left term
                denom1 = knots[i + d] - knots[i]
                if denom1 > 1e-10:
                    if i < B.shape[1]:
                        B_new[:, i] += B[:, i] * (x - knots[i]) / denom1

                # Right term
                denom2 = knots[i + d + 1] - knots[i + 1]
                if denom2 > 1e-10 and (i + 1) < B.shape[1]:
                    B_new[:, i] += B[:, i + 1] * (knots[i + d + 1] - x) / denom2

        B = B_new

    # Handle right boundary: B-splines use half-open intervals [t_i, t_{i+1}).
    # Points at or beyond the last internal knot must map cleanly to the final
    # basis function.  The original line "-1 == np.argmax(x)" was always False
    # (argmax returns a non-negative index), so the boundary was never enforced.
    if n_basis > 0 and len(x) > 0:
        last_knot = knots[-degree - 1]
        at_boundary = x >= last_knot - 1e-10
        if np.any(at_boundary):
            B[at_boundary, :] = 0.0
            B[at_boundary, -1] = 1.0

    return B


def _make_knots(n_internal: int, degree: int = 3,
                x_min: float = -3.0, x_max: float = 3.0) -> np.ndarray:
    """Create augmented knot vector for B-spline."""
    internal = np.linspace(x_min, x_max, n_internal + 2)
    knots = np.concatenate([
        np.full(degree, x_min),
        internal,
        np.full(degree, x_max),
    ])
    return knots


# ============================================================================
# KAN LAYER
# ============================================================================

class KANLayer:
    """
    One layer of Kolmogorov-Arnold Network.

    Each edge (i, j) from input_dim to output_dim has its own B-spline
    parameterized by `n_spline_params` control points.

    Forward pass:
        output_j = Σ_i  spline_{i,j}(input_i)  +  bias_j
    """

    def __init__(self, input_dim: int, output_dim: int,
                 n_spline_knots: int = 5, spline_degree: int = 3):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_spline_knots = n_spline_knots
        self.spline_degree = spline_degree

        # Knots (shared across all edges in the layer)
        self.knots = _make_knots(n_spline_knots, spline_degree)
        self.n_basis = len(self.knots) - spline_degree - 1

        # Parameters: control points for each edge spline + biases
        # Shape: (input_dim, output_dim, n_basis) for spline coefficients
        # Shape: (output_dim,) for biases
        self.n_params = input_dim * output_dim * self.n_basis + output_dim

    def forward(self, x: np.ndarray, params: np.ndarray) -> np.ndarray:
        """
        Forward pass through this KAN layer.

        Parameters
        ----------
        x : shape (batch, input_dim) — input activations
        params : shape (n_params,) — flattened parameters for this layer

        Returns
        -------
        shape (batch, output_dim) — output activations
        """
        batch = x.shape[0]

        # Extract spline coefficients and biases
        n_coeff = self.input_dim * self.output_dim * self.n_basis
        coeffs = params[:n_coeff].reshape(self.input_dim, self.output_dim, self.n_basis)
        biases = params[n_coeff:]

        output = np.zeros((batch, self.output_dim))

        for i in range(self.input_dim):
            # Clip inputs to knot range for numerical stability
            x_i = np.clip(x[:, i], self.knots[self.spline_degree],
                          self.knots[-self.spline_degree - 1] - 1e-6)
            # Evaluate B-spline basis
            B = _bspline_basis(x_i, self.knots, self.spline_degree)  # (batch, n_basis)

            for j in range(self.output_dim):
                # Apply spline: sum of basis * coefficients
                output[:, j] += B @ coeffs[i, j, :]  # (batch,)

        output += biases  # broadcast (output_dim,)
        return output


# ============================================================================
# KAN NETWORK
# ============================================================================

class KANCorrector:
    """
    Kolmogorov-Arnold Network for option pricing correction.

    Architecture: Input(n_features) → KAN_Layer(8) → KAN_Layer(4) → KAN_Layer(1)

    API compatible with MLPricingCorrector:
        - predict_correction(features) → (correction, confidence)
        - add_sample(features, residual)
        - get_feature_importance()

    Advantages over GBM:
        1. Interpretable: learned splines can be visualized per feature
        2. Better extrapolation: splines vs piecewise-constant
        3. Compact: ~500 params vs 200 trees × many leaves

    Usage:
        kan = KANCorrector()
        kan.add_sample(features, residual)
        # ... add more samples ...
        correction, confidence = kan.predict_correction(features)
    """

    MIN_SAMPLES = 30
    RETRAIN_EVERY = 20

    def __init__(self, hidden_dims: Tuple[int, ...] = (8, 4),
                 n_spline_knots: int = 5, spline_degree: int = 3,
                 feature_names: Optional[List[str]] = None):
        self.hidden_dims = hidden_dims
        self.n_spline_knots = n_spline_knots
        self.spline_degree = spline_degree
        self.feature_names = feature_names

        self.layers: List[KANLayer] = []
        self.params: Optional[np.ndarray] = None
        self.is_trained = False
        self.training_X: List[list] = []
        self.training_y: List[float] = []
        self._input_mean: Optional[np.ndarray] = None
        self._input_std: Optional[np.ndarray] = None
        self._n_features: Optional[int] = None

    def _build_network(self, n_features: int):
        """Construct KAN layers."""
        self._n_features = n_features
        self.layers = []
        dims = [n_features] + list(self.hidden_dims) + [1]

        for i in range(len(dims) - 1):
            layer = KANLayer(dims[i], dims[i + 1],
                             self.n_spline_knots, self.spline_degree)
            self.layers.append(layer)

        # Total parameter count
        total_params = sum(layer.n_params for layer in self.layers)
        return total_params

    def _forward(self, X: np.ndarray, params: np.ndarray) -> np.ndarray:
        """
        Full forward pass through the KAN.

        Parameters
        ----------
        X : shape (batch, n_features)
        params : shape (total_params,) — all network parameters

        Returns
        -------
        shape (batch, 1) — predictions
        """
        # Standardize inputs
        if self._input_mean is not None:
            X = (X - self._input_mean) / np.maximum(self._input_std, 1e-8)

        h = X
        offset = 0
        for layer in self.layers:
            p = params[offset:offset + layer.n_params]
            h = layer.forward(h, p)
            # Activation between hidden layers (not on output)
            if layer != self.layers[-1]:
                h = np.tanh(h)  # secondary activation for numerical stability
            offset += layer.n_params

        return h

    def _loss(self, params: np.ndarray, X: np.ndarray, y: np.ndarray,
              l2_reg: float = 0.01) -> float:
        """MSE loss with L2 regularization on spline coefficients."""
        pred = self._forward(X, params).ravel()
        mse = np.mean((pred - y) ** 2)
        reg = l2_reg * np.mean(params ** 2)
        return float(mse + reg)

    def _loss_grad_spsa(
        self,
        params: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        l2_reg: float = 0.01,
        n_spsa: int = 8,
    ) -> np.ndarray:
        """
        SPSA (Simultaneous Perturbation Stochastic Approximation) gradient.

        Produces an unbiased gradient estimate with exactly 2·n_spsa loss
        evaluations, regardless of the number of parameters.  This is
        O(1) in n_params, replacing the previous O(n_params) forward-
        difference loop that required one evaluation PER parameter.

        Algorithm — Spall (1992):
            Δ  = random ±1 vector (Rademacher)
            ĝ  = [L(θ + c·Δ) − L(θ − c·Δ)] / (2c) · Δ   (component-wise)

        With Δᵢ = ±1 the division 1/Δᵢ = Δᵢ, so the estimator simplifies to
        multiplying the scalar difference by Δ.  Averaging over n_spsa
        draws reduces variance without increasing per-evaluation cost.

        Parameters
        ----------
        n_spsa : number of simultaneous perturbation pairs (default 8)
        """
        rng = np.random.default_rng()
        grad = np.zeros_like(params)
        # Adaptive perturbation: proportional to parameter scale
        c = max(0.01 * (np.std(params) + 1e-4), 1e-4)

        for _ in range(n_spsa):
            delta = rng.choice([-1.0, 1.0], size=len(params)).astype(float)
            f_plus  = self._loss(params + c * delta, X, y, l2_reg)
            f_minus = self._loss(params - c * delta, X, y, l2_reg)
            # ĝ = (f+ - f-) / (2c) * Δ   [since 1/Δᵢ = Δᵢ for ±1]
            grad += ((f_plus - f_minus) / (2.0 * c)) * delta

        return grad / n_spsa

    def _train(self):
        """
        Train KAN via Adam + SPSA.

        Previous implementation used L-BFGS-B with a finite-difference
        Jacobian that required O(n_params) ≈ 500 forward passes per
        gradient step — completely impractical for online learning.

        Replacement: Adam optimizer with SPSA gradient estimates.
        Cost: 2·n_spsa forward passes per step, independent of n_params.
        Convergence: typically 100–300 steps, ~1600 forward passes total
        vs ~50 × 500 = 25,000 for the old L-BFGS-B approach.
        """
        if len(self.training_X) < self.MIN_SAMPLES:
            return

        X = np.array(self.training_X, dtype=float)
        y = np.array(self.training_y, dtype=float)
        n_features = X.shape[1]

        # Standardize inputs
        self._input_mean = np.mean(X, axis=0)
        self._input_std  = np.std(X, axis=0) + 1e-8

        # Build network architecture if first call or feature count changed
        if not self.layers or self._n_features != n_features:
            total_params = self._build_network(n_features)
        else:
            total_params = sum(layer.n_params for layer in self.layers)

        # Initialise or retain existing parameters (warm-start on retrain)
        rng = np.random.default_rng(42)
        if self.params is None or len(self.params) != total_params:
            self.params = rng.normal(0, 0.1, total_params)

        # ── Adam hyper-parameters ───────────────────────────────────────
        lr    = 0.005
        beta1, beta2, eps_adam = 0.9, 0.999, 1e-8
        l2_reg = 0.01
        n = len(X)
        batch_size = min(64, n)
        max_steps  = max(150, n)          # more data → more steps

        m = np.zeros(total_params)
        v = np.zeros(total_params)

        best_loss   = float('inf')
        best_params = self.params.copy()

        for t in range(1, max_steps + 1):
            # Mini-batch sampling
            idx     = rng.choice(n, batch_size, replace=(batch_size > n))
            X_batch = X[idx]
            y_batch = y[idx]

            # SPSA gradient estimate — O(16) evaluations per step
            grad = self._loss_grad_spsa(self.params, X_batch, y_batch,
                                        l2_reg=l2_reg, n_spsa=8)

            # Adam moment updates
            m = beta1 * m + (1.0 - beta1) * grad
            v = beta2 * v + (1.0 - beta2) * grad ** 2
            m_hat = m / (1.0 - beta1 ** t)
            v_hat = v / (1.0 - beta2 ** t)
            # Cosine annealing LR decay: reduces oscillation late in training
            lr_t = lr * (0.5 * (1.0 + np.cos(np.pi * t / max_steps)))
            lr_t = max(lr_t, lr * 0.01)  # floor at 1% of initial LR
            self.params -= lr_t * m_hat / (np.sqrt(v_hat) + eps_adam)

            # Track global best (on full dataset, every 25 steps)
            if t % 25 == 0 or t == max_steps:
                current_loss = self._loss(self.params, X, y, l2_reg)
                if current_loss < best_loss:
                    best_loss   = current_loss
                    best_params = self.params.copy()

        self.params    = best_params
        self.is_trained = True

    # ------------------------------------------------------------------
    # PUBLIC API (same as MLPricingCorrector)
    # ------------------------------------------------------------------

    def predict_correction(self, features: dict) -> Tuple[float, float]:
        """
        Returns (correction_factor, confidence).

        correction is a fraction (e.g. 0.03 ⇒ +3% adjustment to NIRV price).
        """
        if not self.is_trained or self.params is None:
            return 0.0, 0.0

        try:
            X = self._features_to_array(features).reshape(1, -1)
            pred = self._forward(X, self.params).ravel()[0]
            correction = float(np.clip(pred, -0.20, 0.20))
            confidence = min(0.85, len(self.training_X) / 500.0)
            return correction, confidence
        except Exception:
            return 0.0, 0.0

    def add_sample(self, features: dict, residual: float):
        """Add training sample; auto-retrain with adaptive interval."""
        X = self._features_to_array(features)
        self.training_X.append(X.tolist())
        self.training_y.append(float(residual))

        n = len(self.training_X)
        retrain_interval = max(self.RETRAIN_EVERY, n // 15)
        if n >= self.MIN_SAMPLES and n % retrain_interval == 0:
            self._train()

    def get_feature_importance(self) -> Dict[str, float]:
        """
        KAN interpretability: compute feature importance by ablation.

        For each feature, zero it out and measure increase in loss.
        Higher increase → more important feature.
        """
        if not self.is_trained or self.params is None or len(self.training_X) < 10:
            return {}

        X = np.array(self.training_X[-200:], dtype=float)
        y = np.array(self.training_y[-200:], dtype=float)
        base_loss = self._loss(self.params, X, y, 0.0)

        importance = {}
        names = self.feature_names or [f'f{i}' for i in range(X.shape[1])]

        for i in range(min(X.shape[1], len(names))):
            X_ablated = X.copy()
            X_ablated[:, i] = 0.0  # zero out feature
            ablated_loss = self._loss(self.params, X_ablated, y, 0.0)
            imp = max(0.0, ablated_loss - base_loss) / max(base_loss, 1e-8)
            if imp > 0.005:
                importance[names[i]] = round(float(imp), 4)

        return dict(sorted(importance.items(), key=lambda x: -x[1]))

    def get_edge_splines(self, layer_idx: int = 0) -> Dict[str, np.ndarray]:
        """
        Extract learned spline functions for visualization/interpretation.

        Returns dict mapping (input_name, output_idx) to arrays of
        (x_values, y_values) for plotting.
        """
        if not self.is_trained or self.params is None or layer_idx >= len(self.layers):
            return {}

        layer = self.layers[layer_idx]
        offset = sum(self.layers[i].n_params for i in range(layer_idx))
        n_coeff = layer.input_dim * layer.output_dim * layer.n_basis
        coeffs = self.params[offset:offset + n_coeff].reshape(
            layer.input_dim, layer.output_dim, layer.n_basis
        )

        x_eval = np.linspace(-3, 3, 100)
        B = _bspline_basis(x_eval, layer.knots, layer.spline_degree)

        splines = {}
        names = self.feature_names or [f'f{i}' for i in range(layer.input_dim)]
        for i in range(layer.input_dim):
            for j in range(layer.output_dim):
                y_eval = B @ coeffs[i, j, :]
                key = f'{names[i] if i < len(names) else f"f{i}"} → h{j}'
                splines[key] = np.column_stack([x_eval, y_eval])

        return splines

    # ------------------------------------------------------------------
    def _features_to_array(self, features: dict) -> np.ndarray:
        """Convert feature dict to numpy array."""
        if self.feature_names:
            return np.array([features.get(n, 0.0) for n in self.feature_names],
                            dtype=np.float64)
        # If no feature names set, use all dict values in sorted key order
        return np.array([features[k] for k in sorted(features.keys())],
                        dtype=np.float64)
