"""
model_free_variance.py — NSE-consistent model-free variance / VIX engine
=========================================================================

Implements VIX-style model-free implied variance with microstructure filters:
  - midpoint from bid/ask only
  - spread quality filter
  - strike interpolation for missing quotes (inside reliable range only)
  - minute-accurate 30-day interpolation across expiries
"""

from __future__ import annotations

import datetime as _dt
from typing import Dict, Iterable, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

from nse_specs import time_to_expiry_minutes


MINUTES_PER_YEAR = 365.0 * 24.0 * 60.0
SPREAD_FILTER_RATIO = 0.30


def _calendar_minutes_to_expiry(now_ts, expiry_ts) -> int:
    now_dt = pd.Timestamp(now_ts).to_pydatetime()
    exp_dt = pd.Timestamp(expiry_ts).to_pydatetime()
    if exp_dt <= now_dt:
        return 0
    return int((exp_dt - now_dt).total_seconds() // 60)


def _to_df(chain_slice) -> pd.DataFrame:
    if isinstance(chain_slice, pd.DataFrame):
        return chain_slice.copy()
    if isinstance(chain_slice, list):
        return pd.DataFrame(chain_slice)
    if isinstance(chain_slice, dict):
        # Common format: {"options": [...]} or {"data": [...]}
        if "options" in chain_slice:
            return pd.DataFrame(chain_slice["options"])
        if "data" in chain_slice:
            return pd.DataFrame(chain_slice["data"])
    return pd.DataFrame(chain_slice)


def _normalize_option_type(s: str) -> str:
    t = str(s or "").upper().strip()
    if t in ("CALL", "C"):
        return "CE"
    if t in ("PUT", "P"):
        return "PE"
    return t


def _extract_mid_quotes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "option_type" not in out.columns:
        if "type" in out.columns:
            out["option_type"] = out["type"]
        else:
            out["option_type"] = ""
    out["option_type"] = out["option_type"].map(_normalize_option_type)

    if "strike" not in out.columns and "strike_price" in out.columns:
        out["strike"] = out["strike_price"]
    out["strike"] = pd.to_numeric(out.get("strike"), errors="coerce")

    # Bid/ask-only midpoint rule
    bid = pd.to_numeric(out.get("bid"), errors="coerce")
    ask = pd.to_numeric(out.get("ask"), errors="coerce")

    mid = (bid + ask) / 2.0
    spread = ask - bid

    has_ba = np.isfinite(bid) & np.isfinite(ask) & (bid > 0) & (ask > 0)
    spread_ok = has_ba & (spread <= SPREAD_FILTER_RATIO * np.maximum(mid, 1e-12))
    mid_valid = np.where(spread_ok, mid, np.nan)

    out["mid"] = mid_valid
    out = out[np.isfinite(out["strike"]) & out["option_type"].isin(["CE", "PE"])]
    out = out[["strike", "option_type", "mid"]]
    return out


def _fill_missing_inside_range(strikes: np.ndarray, values: np.ndarray) -> np.ndarray:
    vals = values.astype(float).copy()
    ok = np.isfinite(vals) & (vals > 0)
    if ok.sum() < 2:
        return vals

    k_ok = strikes[ok]
    v_ok = vals[ok]
    k_min, k_max = float(k_ok.min()), float(k_ok.max())

    missing_inside = (~ok) & (strikes >= k_min) & (strikes <= k_max)
    if not np.any(missing_inside):
        return vals

    try:
        if ok.sum() >= 4:
            spline = CubicSpline(k_ok, v_ok, bc_type="natural")
            filled = spline(strikes[missing_inside])
        else:
            filled = np.interp(strikes[missing_inside], k_ok, v_ok)
        filled = np.maximum(filled, 0.0)
        vals[missing_inside] = filled
    except Exception:
        vals[missing_inside] = np.interp(strikes[missing_inside], k_ok, v_ok)
    return vals


def _build_mid_grid(df_mid: pd.DataFrame) -> pd.DataFrame:
    if df_mid.empty:
        return pd.DataFrame(columns=["strike", "call_mid", "put_mid"])

    pivot = (
        df_mid.pivot_table(index="strike", columns="option_type", values="mid", aggfunc="first")
        .reset_index()
        .sort_values("strike")
        .rename(columns={"CE": "call_mid", "PE": "put_mid"})
    )

    k = pivot["strike"].to_numpy(dtype=float)
    call = _fill_missing_inside_range(k, pivot.get("call_mid", np.nan).to_numpy(dtype=float))
    put = _fill_missing_inside_range(k, pivot.get("put_mid", np.nan).to_numpy(dtype=float))

    pivot["call_mid"] = call
    pivot["put_mid"] = put
    return pivot


def estimate_forward_from_chain(
    chain_slice,
    r: float,
    now_ts: Union[str, _dt.datetime, pd.Timestamp],
    expiry_ts: Union[str, _dt.datetime, pd.Timestamp],
    exchange_calendar=None,
    method: str = "legacy",
) -> float:
    """
    Estimate forward from call-put parity across strikes.

    Legacy method:
        choose strike with smallest |C-P| and set F = K + e^{rT}(C-P)

    Improved method:
        robust weighted average over near-ATM parity points.
    """
    if exchange_calendar is None:
        minutes = _calendar_minutes_to_expiry(now_ts, expiry_ts)
    else:
        minutes = time_to_expiry_minutes(now_ts, expiry_ts, exchange_calendar=exchange_calendar)
    if minutes <= 0:
        return np.nan
    T = minutes / MINUTES_PER_YEAR
    if T <= 0:
        return np.nan

    df = _to_df(chain_slice)
    if df is None or len(df) == 0:
        return np.nan
    mids = _extract_mid_quotes(df)
    mid_grid = _build_mid_grid(mids)
    if mid_grid.empty:
        return np.nan

    both = mid_grid[np.isfinite(mid_grid["call_mid"]) & np.isfinite(mid_grid["put_mid"])].copy()
    if both.empty:
        return np.nan
    both["parity_diff"] = both["call_mid"] - both["put_mid"]
    both["F_i"] = both["strike"] + np.exp(r * T) * both["parity_diff"]
    both = both[np.isfinite(both["F_i"]) & (both["F_i"] > 0)]
    if both.empty:
        return np.nan

    m = str(method or "legacy").strip().lower()
    if m != "improved":
        idx = int(np.argmin(np.abs(both["parity_diff"].to_numpy(dtype=float))))
        return float(both.iloc[idx]["F_i"])

    # Improved: use a robust weighted average from near-ATM candidates.
    abs_diff = np.abs(both["parity_diff"].to_numpy(dtype=float))
    order = np.argsort(abs_diff)
    n_keep = max(3, min(8, len(order)))
    sel = both.iloc[order[:n_keep]].copy()
    w = 1.0 / np.maximum(np.abs(sel["parity_diff"].to_numpy(dtype=float)), 1e-6)
    w = w / np.sum(w)
    f = float(np.sum(w * sel["F_i"].to_numpy(dtype=float)))
    return f if np.isfinite(f) and f > 0 else np.nan


def _compute_strip(mid_grid: pd.DataFrame, forward: float) -> Tuple[pd.DataFrame, float]:
    if mid_grid.empty:
        return pd.DataFrame(columns=["K", "Q"]), np.nan

    strikes = np.sort(mid_grid["strike"].unique())
    below = strikes[strikes <= forward]
    if len(below) == 0:
        return pd.DataFrame(columns=["K", "Q"]), np.nan

    k0 = float(below[-1])
    rows = []
    for _, row in mid_grid.iterrows():
        k = float(row["strike"])
        c = float(row["call_mid"]) if np.isfinite(row["call_mid"]) else np.nan
        p = float(row["put_mid"]) if np.isfinite(row["put_mid"]) else np.nan
        q = np.nan
        if abs(k - k0) < 1e-12:
            if np.isfinite(c) and np.isfinite(p):
                q = 0.5 * (c + p)
            elif np.isfinite(c):
                q = c
            elif np.isfinite(p):
                q = p
        elif k < k0:
            q = p
        else:
            q = c
        if np.isfinite(q) and q > 0:
            rows.append((k, q))

    strip = pd.DataFrame(rows, columns=["K", "Q"]).sort_values("K")
    return strip, k0


def _delta_k(strikes: np.ndarray) -> np.ndarray:
    n = len(strikes)
    if n < 2:
        return np.array([], dtype=float)
    dk = np.zeros(n, dtype=float)
    dk[0] = strikes[1] - strikes[0]
    dk[-1] = strikes[-1] - strikes[-2]
    if n > 2:
        dk[1:-1] = 0.5 * (strikes[2:] - strikes[:-2])
    return dk


def _tail_correction(strip: pd.DataFrame, forward: float, k0: float, r: float, T: float, base_var: float) -> float:
    if strip.empty or len(strip) < 4 or T <= 0:
        return 0.0

    k_arr = strip["K"].to_numpy(dtype=float)
    q_arr = strip["Q"].to_numpy(dtype=float)
    if len(k_arr) < 3:
        return 0.0

    left_ratio = float(k_arr.min() / max(forward, 1e-8))
    right_ratio = float(k_arr.max() / max(forward, 1e-8))
    sparse_or_truncated = (len(k_arr) < 12) or (left_ratio > 0.85) or (right_ratio < 1.15)
    if not sparse_or_truncated:
        return 0.0

    ert = np.exp(r * T)
    corr = 0.0

    # Left wing (puts): log-linear in strike vs price
    if len(k_arr) >= 3 and q_arr[0] > 0 and q_arr[1] > 0:
        step = max(k_arr[1] - k_arr[0], 1e-8)
        k_ext = max(k_arr[0] - step, 1e-8)
        slope = (np.log(q_arr[1]) - np.log(q_arr[0])) / max(k_arr[1] - k_arr[0], 1e-8)
        q_ext = q_arr[0] * np.exp(slope * (k_ext - k_arr[0]))
        q_ext = max(float(q_ext), 0.0)
        dk = step
        corr += (2.0 * ert / T) * (dk / (k_ext ** 2)) * q_ext

    # Right wing (calls): log-linear decay
    if len(k_arr) >= 3 and q_arr[-1] > 0 and q_arr[-2] > 0:
        step = max(k_arr[-1] - k_arr[-2], 1e-8)
        k_ext = k_arr[-1] + step
        slope = (np.log(q_arr[-1]) - np.log(q_arr[-2])) / max(k_arr[-1] - k_arr[-2], 1e-8)
        q_ext = q_arr[-1] * np.exp(slope * (k_ext - k_arr[-1]))
        q_ext = max(float(q_ext), 0.0)
        dk = step
        corr += (2.0 * ert / T) * (dk / (k_ext ** 2)) * q_ext

    # Conservative bound: max 10% of base variance
    bound = 0.10 * max(base_var, 1e-8)
    return float(np.clip(corr, 0.0, bound))


def compute_variance_for_expiry(
    chain_slice,
    forward: Optional[float],
    r: float,
    now_ts: Union[str, _dt.datetime, pd.Timestamp],
    expiry_ts: Union[str, _dt.datetime, pd.Timestamp],
    exchange_calendar=None,
    tail_corrected: bool = False,
    forward_method: str = "legacy",
) -> float:
    """
    Compute model-free variance for a single expiry using VIX-style strip integration.
    """
    if exchange_calendar is None:
        minutes = _calendar_minutes_to_expiry(now_ts, expiry_ts)
    else:
        minutes = time_to_expiry_minutes(now_ts, expiry_ts, exchange_calendar=exchange_calendar)
    if minutes <= 0:
        return np.nan

    T = minutes / MINUTES_PER_YEAR
    if T <= 0:
        return np.nan

    df = _to_df(chain_slice)
    if df is None or len(df) == 0:
        return np.nan
    fwd = float(forward) if forward is not None and np.isfinite(forward) else np.nan
    if (not np.isfinite(fwd)) or fwd <= 0:
        fwd = estimate_forward_from_chain(
            chain_slice=chain_slice,
            r=float(r),
            now_ts=now_ts,
            expiry_ts=expiry_ts,
            exchange_calendar=exchange_calendar,
            method=forward_method,
        )
    if not np.isfinite(fwd) or fwd <= 0:
        return np.nan

    mids = _extract_mid_quotes(df)
    mid_grid = _build_mid_grid(mids)
    strip, k0 = _compute_strip(mid_grid, forward=fwd)
    if strip.empty or not np.isfinite(k0):
        return np.nan

    k = strip["K"].to_numpy(dtype=float)
    q = strip["Q"].to_numpy(dtype=float)
    if len(k) < 3:
        return np.nan

    dk = _delta_k(k)
    ert = np.exp(r * T)
    sum_term = np.sum((dk / np.maximum(k, 1e-12) ** 2) * q)
    var = (2.0 * ert / T) * sum_term - (1.0 / T) * ((fwd / k0) - 1.0) ** 2
    var = max(float(var), 0.0)

    if tail_corrected:
        var += _tail_correction(strip, forward=fwd, k0=k0, r=r, T=T, base_var=var)

    return float(max(var, 0.0))


def _compute_30d_variance(
    var_near: float,
    var_next: float,
    minutes_near: int,
    minutes_next: int,
    target_minutes: int = 30 * 24 * 60,
) -> float:
    if not np.isfinite(var_near) or not np.isfinite(var_next):
        return np.nan
    if minutes_near <= 0 or minutes_next <= 0:
        return np.nan
    if minutes_next <= minutes_near:
        return np.nan

    T1 = minutes_near / MINUTES_PER_YEAR
    T2 = minutes_next / MINUTES_PER_YEAR
    w1 = (minutes_next - target_minutes) / max(minutes_next - minutes_near, 1e-12)
    w2 = (target_minutes - minutes_near) / max(minutes_next - minutes_near, 1e-12)

    base = T1 * var_near * w1 + T2 * var_next * w2
    scale = MINUTES_PER_YEAR / float(target_minutes)
    return float(max(base * scale, 0.0))


def compute_vix_30d(
    chain_near,
    chain_next,
    forward_near: float,
    forward_next: float,
    r: float,
    now_ts: Union[str, _dt.datetime, pd.Timestamp],
    expiry_near_ts: Union[str, _dt.datetime, pd.Timestamp],
    expiry_next_ts: Union[str, _dt.datetime, pd.Timestamp],
    exchange_calendar=None,
    tail_corrected: bool = False,
    target_minutes: int = 30 * 24 * 60,
) -> float:
    """
    Compute 30-day VIX-style index value (percent volatility).
    """
    var_near = compute_variance_for_expiry(
        chain_near, forward_near, r, now_ts, expiry_near_ts,
        exchange_calendar=exchange_calendar, tail_corrected=tail_corrected,
    )
    var_next = compute_variance_for_expiry(
        chain_next, forward_next, r, now_ts, expiry_next_ts,
        exchange_calendar=exchange_calendar, tail_corrected=tail_corrected,
    )

    if exchange_calendar is None:
        n1 = _calendar_minutes_to_expiry(now_ts, expiry_near_ts)
        n2 = _calendar_minutes_to_expiry(now_ts, expiry_next_ts)
    else:
        n1 = time_to_expiry_minutes(now_ts, expiry_near_ts, exchange_calendar=exchange_calendar)
        n2 = time_to_expiry_minutes(now_ts, expiry_next_ts, exchange_calendar=exchange_calendar)
    var30 = _compute_30d_variance(var_near, var_next, n1, n2, target_minutes=target_minutes)
    if not np.isfinite(var30):
        return np.nan
    return float(100.0 * np.sqrt(max(var30, 0.0)))


def compute_vix_30d_with_details(*args, **kwargs) -> Tuple[float, Dict[str, float]]:
    """
    Same as compute_vix_30d, but returns diagnostics for evaluation/reporting.
    """
    chain_near = args[0]
    chain_next = args[1]
    forward_near = kwargs.get("forward_near", args[2] if len(args) > 2 else np.nan)
    forward_next = kwargs.get("forward_next", args[3] if len(args) > 3 else np.nan)
    r = kwargs.get("r", args[4] if len(args) > 4 else 0.0)
    now_ts = kwargs.get("now_ts", args[5] if len(args) > 5 else _dt.datetime.now())
    expiry_near_ts = kwargs.get("expiry_near_ts", args[6] if len(args) > 6 else now_ts)
    expiry_next_ts = kwargs.get("expiry_next_ts", args[7] if len(args) > 7 else now_ts)
    exchange_calendar = kwargs.get("exchange_calendar")
    tail_corrected = kwargs.get("tail_corrected", False)
    target_minutes = kwargs.get("target_minutes", 30 * 24 * 60)

    var_near = compute_variance_for_expiry(
        chain_near, forward_near, r, now_ts, expiry_near_ts,
        exchange_calendar=exchange_calendar, tail_corrected=tail_corrected,
    )
    var_next = compute_variance_for_expiry(
        chain_next, forward_next, r, now_ts, expiry_next_ts,
        exchange_calendar=exchange_calendar, tail_corrected=tail_corrected,
    )
    if exchange_calendar is None:
        n1 = _calendar_minutes_to_expiry(now_ts, expiry_near_ts)
        n2 = _calendar_minutes_to_expiry(now_ts, expiry_next_ts)
    else:
        n1 = time_to_expiry_minutes(now_ts, expiry_near_ts, exchange_calendar=exchange_calendar)
        n2 = time_to_expiry_minutes(now_ts, expiry_next_ts, exchange_calendar=exchange_calendar)
    var30 = _compute_30d_variance(var_near, var_next, n1, n2, target_minutes=target_minutes)
    vix30 = float(100.0 * np.sqrt(max(var30, 0.0))) if np.isfinite(var30) else np.nan
    return vix30, {
        "var_near": float(var_near) if np.isfinite(var_near) else np.nan,
        "var_next": float(var_next) if np.isfinite(var_next) else np.nan,
        "var_30d": float(var30) if np.isfinite(var30) else np.nan,
        "minutes_near": int(max(n1, 0)),
        "minutes_next": int(max(n2, 0)),
        "target_minutes": int(target_minutes),
    }
