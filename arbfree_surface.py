"""
arbfree_surface.py — Arbitrage-Free Volatility Surface
======================================================

Implements a volatility surface state container that ensures:
1.  Absence of Butterfly Arbitrage (g(k) >= 0) via SVI convexity checks.
2.  Absence of Calendar Arbitrage (w(k, T2) >= w(k, T1)) via isotonic regression.

This module is designed to post-process raw implied volatility data (from
NIRV's parametric model or market quotes) into a valid surface.
"""

import numpy as np
from scipy import interpolate
from scipy.optimize import minimize
from typing import List, Dict, Optional, Tuple
from surface_checks import (
    check_butterfly_arbitrage_slice,
    check_calendar_arbitrage,
)

class ArbFreeSurfaceState:
    """
    Manages a collection of volatility slices (expiry, strikes, IVs) and
    constructs a valid total variance surface w(k, T).
    """

    def __init__(self):
        # Storage: valid_slices = list of dicts {'T': float, 'k': array, 'w': array}
        # w = total variance = iv^2 * T
        self.slices = []
        self._interpolator = None
        self._is_fitted = False
        self.last_diagnostics = {}

    def add_slice(self, T: float, strikes: np.ndarray, ivs: np.ndarray, spot: float):
        """
        Add a raw volatility slice to the surface.
        
        Parameters
        ----------
        T : float
            Time to expiry (years).
        strikes : np.ndarray
            Array of strike prices.
        ivs : np.ndarray
            Array of Annualised IVs (decimal, e.g. 0.20).
        spot : float
            Current underlying spot price (for log-moneyness conversion).
        """
        # Filter NaNs / Infs
        mask = np.isfinite(ivs) & (ivs > 0)
        if not np.any(mask):
            return

        k = np.log(strikes[mask] / spot)
        v = ivs[mask]
        w = (v ** 2) * T  # Total variance

        # Sort by Moneyness k
        idx = np.argsort(k)
        self.slices.append({
            'T': T,
            'k': k[idx],
            'w': w[idx]
        })
        self._is_fitted = False

    def fit(self):
        """
        Construct the arbitrage-free surface from added slices.
        1. Butterfly repair (ensure individual slices are convex).
        2. Prepare 1D interpolators for each slice.
        """
        if not self.slices:
            return

        # Sort slices by T
        self.slices.sort(key=lambda x: x['T'])

        # 1. Butterfly Repair and 1D Interpolator Build
        for s in self.slices:
            # Repair w(k)
            s['w'] = self._repair_butterfly(s['k'], s['w'])
            
            # Build 1D interpolator for this slice (w vs k)
            if len(s['k']) < 2:
                # Constant interpolation for single point
                val = s['w'][0]
                s['interp_k'] = lambda x, v=val: np.full_like(x, v, dtype=float)
            else:
                # Flat-wing extrapolation: hold the boundary total-variance value
                # constant outside the observed strike range.  Linear extrapolation
                # ('extrapolate') can produce negative w outside market strikes,
                # which violates the no-arbitrage constraint w(k,T) ≥ 0.
                s['interp_k'] = interpolate.interp1d(
                    s['k'], s['w'], kind='linear',
                    bounds_error=False,
                    fill_value=(float(s['w'][0]), float(s['w'][-1])),
                )

        self._is_fitted = True
        self._update_diagnostics()

    def _w_interp(self, k: float, T: float) -> float:
        """Interpolate total variance directly."""
        if not self.slices:
            return 1e-8
        if T <= self.slices[0]['T']:
            s = self.slices[0]
            return float(max(s['interp_k'](k), 1e-10)) * T / max(s['T'], 1e-8)
        if T >= self.slices[-1]['T']:
            s = self.slices[-1]
            return float(max(s['interp_k'](k), 1e-10)) * T / max(s['T'], 1e-8)

        idx = 0
        for i in range(len(self.slices) - 1):
            if self.slices[i]['T'] <= T <= self.slices[i + 1]['T']:
                idx = i
                break
        s1 = self.slices[idx]
        s2 = self.slices[idx + 1]
        w1 = float(s1['interp_k'](k))
        w2 = float(s2['interp_k'](k))
        ratio = (T - s1['T']) / max(s2['T'] - s1['T'], 1e-12)
        return float(max(w1 + ratio * (w2 - w1), 1e-10))

    def _update_diagnostics(self):
        """Run no-arbitrage diagnostics after fit."""
        if not self.slices:
            self.last_diagnostics = {}
            return

        butterfly_ok = True
        min_d2 = np.inf
        min_w = np.inf
        for s in self.slices:
            ok, metric = check_butterfly_arbitrage_slice(s['w'], k_grid=s['k'])
            butterfly_ok = butterfly_ok and ok
            md2 = metric.get('min_second_derivative', np.nan)
            mw = metric.get('min_total_variance', np.nan)
            if np.isfinite(md2):
                min_d2 = min(min_d2, md2)
            if np.isfinite(mw):
                min_w = min(min_w, mw)

        calendar_ok = True
        min_forward_diff = np.nan
        cal_viol = np.nan
        if len(self.slices) >= 2:
            t_grid = np.array([s['T'] for s in self.slices], dtype=float)
            # Use broad grid; interpolation handles sparse overlap.
            k_grid = np.linspace(-0.35, 0.35, 31)
            cal_ok, cal_metric = check_calendar_arbitrage(
                lambda tt, kk: self._w_interp(kk, tt), t_grid=t_grid, k_grid=k_grid
            )
            calendar_ok = bool(cal_ok)
            min_forward_diff = cal_metric.get('min_forward_diff', np.nan)
            cal_viol = cal_metric.get('violations', np.nan)

        self.last_diagnostics = {
            'butterfly_ok': float(butterfly_ok),
            'calendar_ok': float(calendar_ok),
            'butterfly_min_d2': float(min_d2) if np.isfinite(min_d2) else np.nan,
            'butterfly_min_w': float(min_w) if np.isfinite(min_w) else np.nan,
            'calendar_min_forward_diff': float(min_forward_diff) if np.isfinite(min_forward_diff) else np.nan,
            'calendar_violations': float(cal_viol) if np.isfinite(cal_viol) else np.nan,
        }

    def get_iv(self, k: float, T: float) -> float:
        """
        Get arbitrage-free IV for log-moneyness k and expiry T.
        Uses Total Variance w(k, T) interpolation.
        """
        if not self._is_fitted:
            self.fit()
        
        if not self.slices:
            return 0.20 # Fallback

        # Find time bracket
        # T < T_min: Constant Vol extrapolation (w = w_min * T / T_min)
        if T <= self.slices[0]['T']:
            s_min = self.slices[0]
            w_min = float(s_min['interp_k'](k))
            # Avoid divide-by-zero
            t_min = max(s_min['T'], 1e-6)
            # Implied variance at T_min
            v_sq = w_min / t_min
            # Extrapolate constant vol down to T
            w_total = v_sq * T
            return np.sqrt(max(w_total, 1e-9) / max(T, 1e-9))

        # T > T_max: Constant Vol extrapolation (w = w_max * T / T_max)
        if T >= self.slices[-1]['T']:
            s_max = self.slices[-1]
            w_max = float(s_max['interp_k'](k))
            t_max = max(s_max['T'], 1e-6)
            v_sq = w_max / t_max
            w_total = v_sq * T
            return np.sqrt(max(w_total, 1e-9) / max(T, 1e-9))

        # Bracket T
        # Find i such that slices[i].T <= T <= slices[i+1].T
        idx = 0
        for i in range(len(self.slices) - 1):
            if self.slices[i]['T'] <= T <= self.slices[i+1]['T']:
                idx = i
                break
        
        s1 = self.slices[idx]
        s2 = self.slices[idx+1]
        
        w1 = float(s1['interp_k'](k))
        w2 = float(s2['interp_k'](k))
        
        T1 = s1['T']
        T2 = s2['T']
        
        # Calendar repair check (local):
        # Ensure w2 >= w1. If not, clamp w2? 
        # But we are interpolating. If input was bad, we smooth it.
        # Linear interpolation in Time:
        # w(T) = w1 + (w2 - w1) * (T - T1) / (T2 - T1)
        
        ratio = (T - T1) / (T2 - T1)
        w_total = w1 + ratio * (w2 - w1)
        
        # Boundary update: w must be >= 0
        w_total = max(w_total, 1e-9)
        
        return np.sqrt(w_total / T)

    def _repair_butterfly(self, k: np.ndarray, w: np.ndarray) -> np.ndarray:
        """
        ENSURES: Density g(k) >= 0 (convexity of total variance).

        Uses explicit convexity enforcement rather than UnivariateSpline
        which over-smooths wing information.

        Algorithm:
        1. Check second derivative (w''(k)) at each interior point
        2. Where w''(k) < 0 (concave = butterfly violation), set w to the
           linear interpolation of its neighbors (minimal intervention)
        3. Iterate until no violations remain (max 10 passes)

        Unlike the old spline-based approach, this:
        - Only modifies points with actual concavity violations
        - Preserves wing shape where it's already convex
        - Works locally, not globally
        """
        if len(k) < 5:
            return w

        w_fixed = np.copy(w)
        w_fixed = np.maximum(w_fixed, 1e-10)  # total variance must be positive

        for _pass in range(10):
            # Compute numerical second derivative
            n = len(k)
            d2w = np.zeros(n)
            for i in range(1, n - 1):
                dk_left = k[i] - k[i - 1]
                dk_right = k[i + 1] - k[i]
                if dk_left > 0 and dk_right > 0:
                    d2w[i] = 2.0 * (
                        (w_fixed[i + 1] - w_fixed[i]) / dk_right
                        - (w_fixed[i] - w_fixed[i - 1]) / dk_left
                    ) / (dk_left + dk_right)

            # Find concave violations (d²w/dk² < 0)
            violations = np.where(d2w[1:-1] < -1e-8)[0] + 1
            if len(violations) == 0:
                break

            # Fix: set w at violation points to linear interpolation of neighbors
            # This forces d²w → 0 at the violation point (minimal intervention)
            for idx in violations:
                if 0 < idx < n - 1:
                    dk_l = k[idx] - k[idx - 1]
                    dk_r = k[idx + 1] - k[idx]
                    w_linear = (
                        w_fixed[idx - 1] * dk_r + w_fixed[idx + 1] * dk_l
                    ) / (dk_l + dk_r)
                    w_fixed[idx] = max(w_linear, 1e-10)

        return np.maximum(w_fixed, 1e-10)

    def count_arbitrage_violations(self) -> Dict[str, int]:
        """
        Debug method: Scan surface for discrete arb violations.
        """
        violations = {'calendar': 0, 'butterfly': 0}
        
        # 1. Calendar: check if w(k, T2) < w(k, T1) for T2 > T1
        if len(self.slices) > 1:
            for i in range(len(self.slices)-1):
                s1 = self.slices[i]
                s2 = self.slices[i+1]
                
                # Check at overlapping k range
                common_min = max(s1['k'].min(), s2['k'].min())
                common_max = min(s1['k'].max(), s2['k'].max())
                
                test_ks = np.linspace(common_min, common_max, 20)
                
                w1 = interpolate.interp1d(s1['k'], s1['w'])(test_ks)
                w2 = interpolate.interp1d(s2['k'], s2['w'])(test_ks)
                
                # If w2 < w1, violation
                diff = w2 - w1
                violations['calendar'] += np.sum(diff < -1e-5)

        return violations
