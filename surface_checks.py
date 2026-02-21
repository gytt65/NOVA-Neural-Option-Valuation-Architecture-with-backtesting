"""
surface_checks.py — No-arbitrage diagnostics for volatility surfaces.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, Optional, Tuple, Union

import numpy as np


ArrayLike = Union[np.ndarray, Iterable[float]]


def _as_array(x: ArrayLike) -> np.ndarray:
    return np.asarray(list(x) if not isinstance(x, np.ndarray) else x, dtype=float)


def check_butterfly_arbitrage_slice(
    w_or_fn: Union[Callable[[np.ndarray], np.ndarray], ArrayLike],
    k_grid: Optional[ArrayLike] = None,
    tol: float = 1e-8,
) -> Tuple[bool, Dict[str, float]]:
    """
    Approximate butterfly no-arbitrage check from convexity of total variance w(k).
    """
    if k_grid is None:
        k = np.linspace(-0.4, 0.4, 121)
    else:
        k = _as_array(k_grid)
    if len(k) < 5:
        return False, {"min_second_derivative": np.nan, "min_total_variance": np.nan}

    if callable(w_or_fn):
        w = np.asarray(w_or_fn(k), dtype=float)
    else:
        w = _as_array(w_or_fn)
        if len(w) != len(k):
            return False, {"min_second_derivative": np.nan, "min_total_variance": np.nan}

    if not np.all(np.isfinite(w)):
        return False, {"min_second_derivative": np.nan, "min_total_variance": np.nan}

    d1 = np.gradient(w, k)
    d2 = np.gradient(d1, k)
    min_d2 = float(np.min(d2))
    min_w = float(np.min(w))

    ok = bool(min_d2 >= -tol and min_w >= -tol)
    return ok, {
        "min_second_derivative": min_d2,
        "min_total_variance": min_w,
    }


def check_calendar_arbitrage(
    w_surface: Union[Callable[[float, float], float], np.ndarray],
    t_grid: ArrayLike,
    k_grid: ArrayLike,
    tol: float = 1e-8,
) -> Tuple[bool, Dict[str, float]]:
    """
    Calendar no-arbitrage check: w(k, T) should be non-decreasing in T for each k.
    """
    t = _as_array(t_grid)
    k = _as_array(k_grid)
    if len(t) < 2 or len(k) < 1:
        return False, {"min_forward_diff": np.nan, "violations": np.nan}

    if callable(w_surface):
        w = np.array([[float(w_surface(float(tt), float(kk))) for kk in k] for tt in t], dtype=float)
    else:
        w = np.asarray(w_surface, dtype=float)
        if w.shape != (len(t), len(k)):
            return False, {"min_forward_diff": np.nan, "violations": np.nan}

    if not np.all(np.isfinite(w)):
        return False, {"min_forward_diff": np.nan, "violations": np.nan}

    diffs = w[1:, :] - w[:-1, :]
    min_diff = float(np.min(diffs))
    violations = int(np.sum(diffs < -tol))
    return bool(violations == 0), {"min_forward_diff": min_diff, "violations": float(violations)}


def check_monotonicity_and_convexity_of_call_prices(
    strikes: ArrayLike,
    call_prices: ArrayLike,
    tol: float = 1e-8,
) -> Tuple[bool, Dict[str, float]]:
    """
    Call-price static arbitrage checks:
      - C(K) should be non-increasing in strike
      - C(K) should be convex in strike
    """
    k = _as_array(strikes)
    c = _as_array(call_prices)
    if len(k) != len(c) or len(k) < 3:
        return False, {"max_monotonicity_breach": np.nan, "min_convexity": np.nan}

    order = np.argsort(k)
    k = k[order]
    c = c[order]

    dc = np.diff(c)
    max_mono_breach = float(np.max(dc))  # >0 is violation
    d2 = c[2:] - 2.0 * c[1:-1] + c[:-2]
    min_convexity = float(np.min(d2))

    ok_mono = np.all(dc <= tol)
    ok_convex = np.all(d2 >= -tol)
    return bool(ok_mono and ok_convex), {
        "max_monotonicity_breach": max_mono_breach,
        "min_convexity": min_convexity,
    }
