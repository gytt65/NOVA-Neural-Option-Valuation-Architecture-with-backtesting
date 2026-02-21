"""
essvi_surface.py — Lightweight eSSVI volatility surface
======================================================

Robust, bounded eSSVI implementation for no-static-arbitrage-friendly
surface generation across strikes and maturities.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from surface_checks import check_butterfly_arbitrage_slice, check_calendar_arbitrage
try:
    from scipy.optimize import minimize
except Exception:
    minimize = None

try:
    from omega_features import get_features
except Exception:
    get_features = lambda: type(
        "Features",
        (),
        {
            "USE_LIQUIDITY_WEIGHTING": False,
            "USE_INTERVAL_LOSS": False,
        },
    )()


class ESSVISurface:
    """
    Minimal eSSVI surface:
        w(k, t) = 0.5 * theta(t) * [1 + rho(t) * phi(t) * k
                                    + sqrt((phi(t) * k + rho(t))^2 + 1 - rho(t)^2)]
    """

    def __init__(self):
        self.t_nodes = np.array([], dtype=float)
        self.theta_nodes = np.array([], dtype=float)
        self.rho_nodes = np.array([], dtype=float)
        self.phi_nodes = np.array([], dtype=float)
        self._is_fitted = False
        self.last_diagnostics: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def fit(self, slices: Sequence[Dict], spot: float) -> bool:
        """
        Fit from slices:
          slices = [{"T": float, "strikes": np.ndarray, "ivs": np.ndarray}, ...]
        """
        rows = []
        for sl in slices:
            T = float(sl.get("T", 0.0))
            strikes = np.asarray(sl.get("strikes", []), dtype=float)
            ivs = np.asarray(sl.get("ivs", []), dtype=float)
            if T <= 0 or len(strikes) < 3 or len(ivs) < 3:
                continue
            mask = np.isfinite(strikes) & np.isfinite(ivs) & (strikes > 0) & (ivs > 0)
            if np.sum(mask) < 3:
                continue
            k = np.log(strikes[mask] / max(float(spot), 1e-8))
            w = (ivs[mask] ** 2) * T
            idx = np.argsort(k)
            k = k[idx]
            w = w[idx]
            strikes_sorted = strikes[mask][idx]
            sl_row = {
                "T": T,
                "k": k,
                "w": w,
                "strikes": strikes_sorted,
                "raw_slice": sl,
            }
            rows.append(sl_row)

        if not rows:
            self._is_fitted = False
            return False

        rows.sort(key=lambda x: x["T"])
        t_nodes = []
        theta_nodes = []
        rho_nodes = []
        phi_nodes = []

        for row in rows:
            T = row["T"]
            k = row["k"]
            w = row["w"]
            strikes_sorted = row["strikes"]
            sl = row["raw_slice"]

            weights = self._compute_liquidity_weights(
                k=k,
                w=w,
                T=T,
                strikes=strikes_sorted,
                sl=sl,
            )
            theta = self._estimate_theta(k, w, weights=weights)
            rho = self._estimate_rho(theta, k, w)
            phi = self._estimate_phi(theta, rho, k, w)
            theta, rho, phi = self._refine_slice_params(
                theta=theta,
                rho=rho,
                phi=phi,
                k=k,
                w_obs=w,
                T=T,
                weights=weights,
                sl=sl,
            )
            t_nodes.append(T)
            theta_nodes.append(theta)
            rho_nodes.append(rho)
            phi_nodes.append(phi)

        t = np.asarray(t_nodes, dtype=float)
        theta = np.maximum(np.asarray(theta_nodes, dtype=float), 1e-10)
        rho = np.clip(np.asarray(rho_nodes, dtype=float), -0.99, 0.99)
        phi = np.maximum(np.asarray(phi_nodes, dtype=float), 1e-8)

        # Calendar no-arbitrage friendly theta smoothing: enforce non-decreasing total variance at ATM
        theta = np.maximum.accumulate(theta)

        # Re-apply Gatheral/Jacquier-like bound for each node: theta*phi*(1+|rho|) < 4
        bound = 4.0 / np.maximum(theta * (1.0 + np.abs(rho)), 1e-8)
        phi = np.minimum(phi, 0.95 * bound)
        phi = np.maximum(phi, 1e-8)

        self.t_nodes = t
        self.theta_nodes = theta
        self.rho_nodes = rho
        self.phi_nodes = phi
        self._is_fitted = True
        self.last_diagnostics = self._compute_diagnostics()
        return True

    @staticmethod
    def _estimate_theta(k: np.ndarray, w: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
        # ATM total variance estimate from nearest-to-ATM points
        idx = np.argsort(np.abs(k))[: max(1, min(5, len(k)))]
        if weights is not None and len(weights) == len(w):
            ww = np.maximum(np.asarray(weights, dtype=float)[idx], 1e-12)
            return float(np.sum(w[idx] * ww) / np.sum(ww))
        return float(np.mean(w[idx]))

    @staticmethod
    def _estimate_rho(theta: float, k: np.ndarray, w: np.ndarray) -> float:
        if len(k) < 4:
            return -0.3
        # Local slope around ATM
        mask = np.abs(k) <= 0.1
        if np.sum(mask) < 3:
            mask = np.abs(k) <= 0.2
        if np.sum(mask) < 3:
            return -0.3
        ks = k[mask]
        ws = w[mask]
        try:
            slope = np.polyfit(ks, ws, 1)[0]
        except Exception:
            slope = (ws[-1] - ws[0]) / max(ks[-1] - ks[0], 1e-8)
        # Map slope -> rho with bounded nonlinearity
        scale = max(theta, 1e-8)
        rho = -np.tanh(2.5 * slope / scale)
        return float(np.clip(rho, -0.95, 0.95))

    @staticmethod
    def _estimate_phi(theta: float, rho: float, k: np.ndarray, w: np.ndarray) -> float:
        if len(k) < 4:
            return 1.0
        # Wing slope proxy from absolute moneyness
        mask = np.abs(k) >= 0.1
        if np.sum(mask) < 2:
            mask = np.abs(k) >= 0.05
        if np.sum(mask) < 2:
            wing = 0.5
        else:
            kw = np.abs(k[mask])
            ww = np.maximum(w[mask] - theta, 1e-8)
            wing = float(np.median(ww / np.maximum(kw, 1e-8)))
        phi = wing / max(theta, 1e-8)
        phi = float(np.clip(phi, 1e-3, 20.0))

        # No-arbitrage bound
        phi_max = 0.95 * (4.0 / max(theta * (1.0 + abs(rho)), 1e-8))
        return float(min(phi, phi_max))

    @staticmethod
    def _slice_total_variance(k: np.ndarray, theta: float, rho: float, phi: float) -> np.ndarray:
        x = phi * np.asarray(k, dtype=float)
        root = np.sqrt((x + rho) ** 2 + np.maximum(1.0 - rho ** 2, 1e-12))
        return 0.5 * theta * (1.0 + rho * x + root)

    def _compute_liquidity_weights(
        self,
        k: np.ndarray,
        w: np.ndarray,
        T: float,
        strikes: np.ndarray,
        sl: Dict,
    ) -> Optional[np.ndarray]:
        if not bool(getattr(get_features(), "USE_LIQUIDITY_WEIGHTING", False)):
            return None

        raw_spreads = sl.get("spreads")
        raw_vegas = sl.get("vegas")
        spreads = None
        vegas = None
        if isinstance(raw_spreads, dict):
            arr = []
            for kk in strikes:
                try:
                    arr.append(float(raw_spreads.get(float(kk), np.nan)))
                except Exception:
                    arr.append(np.nan)
            spreads = np.asarray(arr, dtype=float)
        elif raw_spreads is not None:
            spreads = np.asarray(raw_spreads, dtype=float)
        if isinstance(raw_vegas, dict):
            arr = []
            for kk in strikes:
                try:
                    arr.append(float(raw_vegas.get(float(kk), np.nan)))
                except Exception:
                    arr.append(np.nan)
            vegas = np.asarray(arr, dtype=float)
        elif raw_vegas is not None:
            vegas = np.asarray(raw_vegas, dtype=float)
        if spreads is None or len(spreads) != len(k):
            return None
        if vegas is None or len(vegas) != len(k):
            vegas = np.ones(len(k), dtype=float)
        denom = np.maximum(spreads ** 2, 1e-12)
        weights = np.maximum(vegas, 1e-12) / denom
        weights = np.where(np.isfinite(weights), weights, 1.0)
        weights = np.maximum(weights, 1e-12)
        return weights / np.mean(weights)

    def _extract_interval_bounds(self, sl: Dict, T: float, n: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if not bool(getattr(get_features(), "USE_INTERVAL_LOSS", False)):
            return None, None

        iv_bid = sl.get("iv_bid")
        iv_ask = sl.get("iv_ask")
        if iv_bid is None or iv_ask is None:
            return None, None
        iv_bid = np.asarray(iv_bid, dtype=float)
        iv_ask = np.asarray(iv_ask, dtype=float)
        if len(iv_bid) != n or len(iv_ask) != n:
            return None, None
        valid = np.isfinite(iv_bid) & np.isfinite(iv_ask) & (iv_bid > 0) & (iv_ask >= iv_bid)
        if np.sum(valid) == 0:
            return None, None
        w_bid = np.full(n, np.nan, dtype=float)
        w_ask = np.full(n, np.nan, dtype=float)
        w_bid[valid] = (iv_bid[valid] ** 2) * T
        w_ask[valid] = (iv_ask[valid] ** 2) * T
        return w_bid, w_ask

    def _refine_slice_params(
        self,
        theta: float,
        rho: float,
        phi: float,
        k: np.ndarray,
        w_obs: np.ndarray,
        T: float,
        weights: Optional[np.ndarray],
        sl: Dict,
    ) -> Tuple[float, float, float]:
        if minimize is None:
            return theta, rho, phi
        if len(k) < 4:
            return theta, rho, phi
        use_liq = bool(getattr(get_features(), "USE_LIQUIDITY_WEIGHTING", False))
        use_interval = bool(getattr(get_features(), "USE_INTERVAL_LOSS", False))
        if not (use_liq or use_interval):
            return theta, rho, phi

        wts = np.ones(len(k), dtype=float) if weights is None else np.asarray(weights, dtype=float)
        wts = np.maximum(np.where(np.isfinite(wts), wts, 1.0), 1e-12)
        wts = wts / np.mean(wts)

        w_bid, w_ask = self._extract_interval_bounds(sl, T=T, n=len(k))

        def _objective(x):
            th, rh, ph = float(x[0]), float(x[1]), float(x[2])
            th = max(th, 1e-10)
            rh = float(np.clip(rh, -0.99, 0.99))
            ph = max(ph, 1e-8)
            model_w = self._slice_total_variance(k, th, rh, ph)
            if use_interval and w_bid is not None and w_ask is not None:
                resid = np.zeros_like(model_w)
                for i in range(len(model_w)):
                    if np.isfinite(w_bid[i]) and np.isfinite(w_ask[i]):
                        if model_w[i] < w_bid[i]:
                            resid[i] = model_w[i] - w_bid[i]
                        elif model_w[i] > w_ask[i]:
                            resid[i] = model_w[i] - w_ask[i]
                        else:
                            resid[i] = 0.0
                    else:
                        resid[i] = model_w[i] - w_obs[i]
            else:
                resid = model_w - w_obs
            return float(np.sum(wts * (resid ** 2)))

        x0 = np.array([theta, rho, phi], dtype=float)
        res = minimize(
            _objective,
            x0=x0,
            method="L-BFGS-B",
            bounds=[(1e-8, 5.0), (-0.99, 0.99), (1e-8, 50.0)],
            options={"maxiter": 120},
        )
        if not bool(getattr(res, "success", False)):
            return theta, rho, phi
        th, rh, ph = res.x
        th = max(float(th), 1e-10)
        rh = float(np.clip(rh, -0.99, 0.99))
        ph = max(float(ph), 1e-8)
        # Keep no-arb style bound.
        phi_max = 0.95 * (4.0 / max(th * (1.0 + abs(rh)), 1e-8))
        ph = min(ph, phi_max)
        return th, rh, ph

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
    def _interp_param(self, t: float, nodes: np.ndarray) -> float:
        if len(self.t_nodes) == 0:
            return float(nodes[0]) if len(nodes) else 0.0
        if len(self.t_nodes) == 1:
            return float(nodes[0])
        return float(np.interp(t, self.t_nodes, nodes, left=nodes[0], right=nodes[-1]))

    def total_variance(self, k: float, t: float) -> float:
        if not self._is_fitted:
            return 1e-8
        tt = max(float(t), 1e-8)
        kk = float(k)

        theta = max(self._interp_param(tt, self.theta_nodes), 1e-10)
        rho = float(np.clip(self._interp_param(tt, self.rho_nodes), -0.99, 0.99))
        phi = max(self._interp_param(tt, self.phi_nodes), 1e-8)
        x = phi * kk
        root = np.sqrt((x + rho) ** 2 + max(1.0 - rho ** 2, 1e-12))
        w = 0.5 * theta * (1.0 + rho * x + root)
        return float(max(w, 1e-10))

    def implied_vol(self, k: float, t: float) -> float:
        tt = max(float(t), 1e-8)
        return float(np.sqrt(self.total_variance(k, tt) / tt))

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def _compute_diagnostics(self) -> Dict[str, float]:
        if not self._is_fitted:
            return {}

        k_grid = np.linspace(-0.35, 0.35, 81)
        t_grid = np.unique(np.concatenate([self.t_nodes, np.linspace(self.t_nodes[0], self.t_nodes[-1], 8)]))
        t_ref = float(self.t_nodes[0]) if len(self.t_nodes) else 0.1

        b_ok, b_metric = check_butterfly_arbitrage_slice(
            lambda kk: np.array([self.total_variance(float(k), t_ref) for k in kk], dtype=float),
            k_grid=k_grid,
        )
        c_ok, c_metric = check_calendar_arbitrage(
            lambda tt, kk: self.total_variance(float(kk), float(tt)),
            t_grid=t_grid,
            k_grid=k_grid[::8],
        )

        return {
            "butterfly_ok": float(bool(b_ok)),
            "calendar_ok": float(bool(c_ok)),
            "butterfly_min_d2": float(b_metric.get("min_second_derivative", np.nan)),
            "calendar_min_forward_diff": float(c_metric.get("min_forward_diff", np.nan)),
            "calendar_violations": float(c_metric.get("violations", np.nan)),
        }
