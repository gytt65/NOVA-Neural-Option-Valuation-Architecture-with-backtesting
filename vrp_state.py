"""
vrp_state.py — Model-free variance risk premium term-structure state.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np


class ModelFreeVRPState:
    """
    Snapshot estimator for VRP term-structure state:

      VRP(T) = RN_var(T) - E[RV(T)]
      VRP_slope = VRP(short) - VRP(long)
    """

    HORIZONS = (7, 30, 60)

    def __init__(self):
        self.last_state: Dict[str, float] = {}

    @staticmethod
    def _safe_var(returns: np.ndarray) -> float:
        if returns is None or len(returns) < 2:
            return 0.0
        return float(np.var(np.asarray(returns, dtype=float), ddof=1))

    def expected_realized_var(self, returns_history: np.ndarray, horizon_days: int) -> float:
        """
        Lightweight HAR-RV style expectation (daily data).
        """
        arr = np.asarray(returns_history if returns_history is not None else [], dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) < 5:
            return 0.0001

        rv1 = self._safe_var(arr[-1:])
        rv5 = self._safe_var(arr[-5:]) if len(arr) >= 5 else rv1
        rv22 = self._safe_var(arr[-22:]) if len(arr) >= 22 else rv5

        # HAR-inspired convex combination, then annualize
        daily_var = 0.55 * rv1 + 0.30 * rv5 + 0.15 * rv22
        annual_var = float(max(daily_var * 252.0, 1e-8))

        # Horizon-adjusted expectation (mild mean reversion)
        h = max(int(horizon_days), 1)
        blend = min(h / 60.0, 1.0)
        long_run = float(max(rv22 * 252.0, 1e-8))
        exp_var = (1.0 - 0.35 * blend) * annual_var + (0.35 * blend) * long_run
        return float(max(exp_var, 1e-8))

    def _bootstrap_realized_var(
        self,
        returns_history: np.ndarray,
        horizon_days: int,
        n_bootstrap: int = 200,
    ) -> tuple:
        """
        Bootstrap posterior for expected realized variance.

        Returns (mean, std, ci_lower, ci_upper) of annualized variance.
        """
        arr = np.asarray(returns_history if returns_history is not None else [], dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) < 10:
            point = self.expected_realized_var(returns_history, horizon_days)
            return point, 0.0, point, point

        rng = np.random.default_rng()  # non-deterministic for proper bootstrap variance
        estimates = np.empty(n_bootstrap)
        n = len(arr)

        for b in range(n_bootstrap):
            # Block bootstrap (block size = max(5, horizon//2)) for serial correlation
            block_size = max(5, horizon_days // 2)
            n_blocks = max(n // block_size, 2)
            indices = []
            for _ in range(n_blocks):
                start = rng.integers(0, max(n - block_size, 1))
                indices.extend(range(start, min(start + block_size, n)))
            sample = arr[np.array(indices[:n])]
            estimates[b] = self.expected_realized_var(sample, horizon_days)

        mean_est = float(np.mean(estimates))
        std_est = float(np.std(estimates, ddof=1))
        ci_lo = float(np.percentile(estimates, 5.0))
        ci_hi = float(np.percentile(estimates, 95.0))
        return mean_est, std_est, ci_lo, ci_hi

    def compute_state(
        self,
        rn_var_term: Optional[Dict[int, float]],
        returns_history: Optional[np.ndarray],
    ) -> Dict[str, float]:
        """
        Parameters
        ----------
        rn_var_term : dict {7: RN_var_7d, 30: RN_var_30d, 60: RN_var_60d}
                      Values are annualized variances in decimal^2.
        returns_history : recent log returns
        """
        rn_var_term = dict(rn_var_term or {})
        rv_exp = {}
        vrp = {}
        vrp_ci = {}  # bootstrap uncertainty

        for h in self.HORIZONS:
            rn = rn_var_term.get(h, np.nan)
            erv = self.expected_realized_var(returns_history, h)
            rv_exp[h] = float(erv)

            # Bootstrap posterior for uncertainty estimation
            erv_mean, erv_std, erv_lo, erv_hi = self._bootstrap_realized_var(
                returns_history, h, n_bootstrap=150
            )
            if np.isfinite(rn) and rn > 0:
                vrp[h] = float(rn - erv)
                vrp_ci[h] = {
                    'std': float(erv_std),
                    'ci_lower': float(rn - erv_hi),  # VRP lower = RN - RV_high
                    'ci_upper': float(rn - erv_lo),  # VRP upper = RN - RV_low
                }
            else:
                vrp[h] = np.nan
                vrp_ci[h] = {'std': 0.0, 'ci_lower': 0.0, 'ci_upper': 0.0}

        # Missing RN variance fallback: use nearest available
        for h in self.HORIZONS:
            if not np.isfinite(vrp[h]):
                nearest = None
                best_dist = 10**9
                for hh in self.HORIZONS:
                    if np.isfinite(vrp.get(hh, np.nan)):
                        d = abs(h - hh)
                        if d < best_dist:
                            best_dist = d
                            nearest = vrp[hh]
                vrp[h] = float(nearest) if nearest is not None else 0.0

        vrp_level = float(vrp[30])
        vrp_slope = float(vrp[7] - vrp[60])

        # Posterior certainty: coefficient of variation (lower = more certain)
        vrp_30_std = vrp_ci.get(30, {}).get('std', 0.0)
        certainty = 1.0 / (1.0 + 10.0 * vrp_30_std)  # [0, 1], higher = more certain

        # State logic
        if vrp_level > 0.01 and vrp_slope > 0.002:
            regime = "FEAR_STEEP"
        elif vrp_level > 0.0:
            regime = "FEAR"
        elif vrp_level < -0.005 and vrp_slope < -0.002:
            regime = "COMPLACENCY_STEEP"
        elif vrp_level < 0.0:
            regime = "COMPLACENCY"
        else:
            regime = "NEUTRAL"

        out = {
            "vrp_7d": float(vrp[7]),
            "vrp_30d": float(vrp[30]),
            "vrp_60d": float(vrp[60]),
            "vrp_level": vrp_level,
            "vrp_slope": vrp_slope,
            "vrp_std": float(vrp_30_std),
            "vrp_ci_lower": float(vrp_ci.get(30, {}).get('ci_lower', vrp_level)),
            "vrp_ci_upper": float(vrp_ci.get(30, {}).get('ci_upper', vrp_level)),
            "vrp_certainty": float(certainty),
            "rn_var_7d": float(rn_var_term.get(7, np.nan)) if 7 in rn_var_term else np.nan,
            "rn_var_30d": float(rn_var_term.get(30, np.nan)) if 30 in rn_var_term else np.nan,
            "rn_var_60d": float(rn_var_term.get(60, np.nan)) if 60 in rn_var_term else np.nan,
            "exp_rv_7d": float(rv_exp[7]),
            "exp_rv_30d": float(rv_exp[30]),
            "exp_rv_60d": float(rv_exp[60]),
            "state_label": regime,
        }
        self.last_state = out
        return out

    @staticmethod
    def parameter_adjustments(state: Dict[str, float]) -> Dict[str, float]:
        """
        Conservative bounded multipliers for Heston params.

        Now incorporates posterior certainty: when the VRP estimate is
        uncertain, adjustments are dampened toward 1.0 (neutral).
        """
        lvl = float(state.get("vrp_level", 0.0))
        slope = float(state.get("vrp_slope", 0.0))
        certainty = float(state.get("vrp_certainty", 1.0))

        # Raw multipliers
        kappa_raw = float(np.clip(1.0 - 2.0 * slope, 0.85, 1.15))
        theta_raw = float(np.clip(1.0 + 2.5 * lvl, 0.85, 1.20))
        sigma_v_raw = float(np.clip(
            1.0 + 1.8 * max(lvl, 0.0) + 0.8 * max(slope, 0.0), 0.85, 1.25
        ))

        # Dampen toward neutral (1.0) when uncertainty is high
        kappa_mult = 1.0 + certainty * (kappa_raw - 1.0)
        theta_mult = 1.0 + certainty * (theta_raw - 1.0)
        sigma_v_mult = 1.0 + certainty * (sigma_v_raw - 1.0)

        return {
            "kappa_mult": float(kappa_mult),
            "theta_mult": float(theta_mult),
            "sigma_v_mult": float(sigma_v_mult),
            "certainty": float(certainty),
        }

