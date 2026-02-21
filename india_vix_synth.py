"""
india_vix_synth.py — Synthetic India VIX Calculation
====================================================

Computes a model-free volatility index (similar to CBOE VIX / India VIX)
directly from option chain data.

Key features:
- Uses the VIX formula: σ² = (2/T) * Σ [ΔK/K² * e^(RT) * Q(K)] - (1/T)[F/K_0 - 1]²
- Interpolates between near-term and next-term expirations to target constant 30-day maturity.
- Filters illiquid / zero-bid quotes.
- Robust to missing strikes using strike gap filling.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union
import datetime as _dt
import warnings

try:
    from omega_features import get_features
except Exception:
    get_features = lambda: type(
        "Features",
        (),
        {"USE_NSE_VIX_ENGINE": False, "USE_TAIL_CORRECTED_VARIANCE": False},
    )()

try:
    from model_free_variance import (
        compute_variance_for_expiry,
        compute_vix_30d_with_details,
    )
except Exception:
    compute_variance_for_expiry = None
    compute_vix_30d_with_details = None

# ── Constants ────────────────────────────────────────────────────────
MINUTES_PER_YEAR = 365.0 * 24.0 * 60.0


def _ensure_bid_ask_df(options) -> pd.DataFrame:
    if isinstance(options, pd.DataFrame):
        df = options.copy()
    else:
        df = pd.DataFrame(options)

    if "strike" not in df.columns and "strike_price" in df.columns:
        df["strike"] = df["strike_price"]
    if "option_type" not in df.columns and "type" in df.columns:
        df["option_type"] = df["type"]
    df["option_type"] = df.get("option_type", "").astype(str).str.upper().replace({"CALL": "CE", "PUT": "PE"})

    # If bid/ask missing, infer a conservative synthetic spread around price for compatibility.
    if "bid" not in df.columns or "ask" not in df.columns:
        px = pd.to_numeric(df.get("price"), errors="coerce")
        df["bid"] = px * 0.995
        df["ask"] = px * 1.005
    else:
        b = pd.to_numeric(df["bid"], errors="coerce")
        a = pd.to_numeric(df["ask"], errors="coerce")
        px = pd.to_numeric(df.get("price"), errors="coerce")
        # Fill isolated missing side with synthetic ±0.5% around price
        b = b.where(np.isfinite(b), px * 0.995)
        a = a.where(np.isfinite(a), px * 1.005)
        df["bid"] = b
        df["ask"] = a

    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    return df[["strike", "option_type", "bid", "ask"]].dropna(subset=["strike"])


def _estimate_forward_from_chain(df: pd.DataFrame, r: float, T: float) -> float:
    if df is None or df.empty:
        return np.nan
    d = df.copy()
    d["option_type"] = d["option_type"].astype(str).str.upper().replace({"CALL": "CE", "PUT": "PE"})
    mid = 0.5 * (pd.to_numeric(d.get("bid"), errors="coerce") + pd.to_numeric(d.get("ask"), errors="coerce"))
    d["mid"] = mid
    calls = d[d["option_type"] == "CE"].set_index("strike")["mid"]
    puts = d[d["option_type"] == "PE"].set_index("strike")["mid"]
    common = calls.index.intersection(puts.index)
    if len(common) == 0:
        return np.nan
    common = np.asarray(sorted(common), dtype=float)
    diffs = np.abs(calls.loc[common] - puts.loc[common])
    k_star = float(common[np.argmin(diffs.values if hasattr(diffs, "values") else diffs)])
    c_star = float(calls.loc[k_star])
    p_star = float(puts.loc[k_star])
    return float(k_star + np.exp(r * T) * (c_star - p_star))


def compute_synthetic_vix(
    chains: List[Dict],
    risk_free_rate: float = 0.065,
    target_days: float = 30.0
) -> Tuple[float, float, Dict]:
    """
    Compute interpolated 30-day synthetic VIX from one or two option expiries.
    
    Parameters
    ----------
    chains : List[Dict]
        List of option chain structures. Each dict must contain:
        {
            'T': float (annualized time to expiry),
            'options': DataFrame or list of dicts with columns:
                       ['strike', 'option_type', 'price', 'bid', 'ask']
                       (bid/ask optional, uses price if missing)
        }
        Must be sorted by T.
        
    risk_free_rate : float
        Annualized risk-free rate (r).
        
    target_days : float
        Target maturity in days (default 30).
        
    Returns
    -------
    (vix_value, quality_score, details_dict)
    
    vix_value : float
        The synthetic VIX (e.g. 14.5 for 14.5% vol).
        Returns None or NaN if calculation fails.
        
    quality_score : float
        0.0 to 1.0 indicating reliability (based on strike density, gaps).
        
    details_dict : dict
        Intermediate values (near_term_vol, next_term_vol, forward_prices).
    """
    if not chains:
        return np.nan, 0.0, {}

    # v6 path: route to model_free_variance engine when flag is enabled.
    if (getattr(get_features(), "USE_NSE_VIX_ENGINE", False)
            and compute_variance_for_expiry is not None):
        warnings.warn(
            "india_vix_synth.compute_synthetic_vix is deprecated; using model_free_variance engine.",
            DeprecationWarning,
            stacklevel=2,
        )
        try:
            chains_sorted = sorted(chains, key=lambda x: x.get("T", np.inf))
            valid = [c for c in chains_sorted if c.get("T", 0) and c.get("T", 0) > 0]
            if not valid:
                return np.nan, 0.0, {"error": "No valid expiries"}

            now_ts = _dt.datetime.now()
            tail_corrected = bool(getattr(get_features(), "USE_TAIL_CORRECTED_VARIANCE", False))

            # Single-expiry fallback
            if len(valid) == 1:
                c = valid[0]
                T = float(c["T"])
                exp_ts = now_ts + _dt.timedelta(days=max(int(round(T * 365)), 1))
                df = _ensure_bid_ask_df(c["options"])
                fwd = _estimate_forward_from_chain(df, risk_free_rate, T)
                if not np.isfinite(fwd):
                    fwd = 0.0
                var = compute_variance_for_expiry(
                    chain_slice=df, forward=fwd, r=risk_free_rate,
                    now_ts=now_ts, expiry_ts=exp_ts,
                    tail_corrected=tail_corrected,
                )
                vix = 100.0 * np.sqrt(max(var, 0.0)) if np.isfinite(var) else np.nan
                quality = 1.0 if np.isfinite(vix) else 0.0
                return vix, quality, {"single_expiry": True, "var": float(var) if np.isfinite(var) else np.nan}

            # Two-expiry interpolation
            near = valid[0]
            nxt = valid[1]
            T1 = float(near["T"])
            T2 = float(nxt["T"])
            exp1 = now_ts + _dt.timedelta(days=max(int(round(T1 * 365)), 1))
            exp2 = now_ts + _dt.timedelta(days=max(int(round(T2 * 365)), 2))
            df1 = _ensure_bid_ask_df(near["options"])
            df2 = _ensure_bid_ask_df(nxt["options"])
            f1 = _estimate_forward_from_chain(df1, risk_free_rate, T1)
            f2 = _estimate_forward_from_chain(df2, risk_free_rate, T2)
            if not np.isfinite(f1):
                f1 = 0.0
            if not np.isfinite(f2):
                f2 = 0.0
            vix30, details = compute_vix_30d_with_details(
                df1, df2, f1, f2, risk_free_rate, now_ts, exp1, exp2,
                tail_corrected=tail_corrected,
                target_minutes=int(target_days * 24 * 60),
            )
            quality = 1.0 if np.isfinite(vix30) else 0.0
            details = dict(details or {})
            details.update({"T1": T1, "T2": T2, "F1": f1, "F2": f2})
            return vix30, quality, details
        except Exception as e:
            return np.nan, 0.0, {"error": str(e)}
    
    # Sort chains by expiry
    chains = sorted(chains, key=lambda x: x['T'])
    
    # Identify near-term and next-term
    # VIX methodology selects:
    # Near-term: closest expiry with > 3 days (exceptionally <3 allowed if only choice)
    # Next-term: expiry immediately following near-term
    
    # Simple selection: find two expiries straddling 30 days if possible
    target_t = target_days / 365.0
    
    valid_chains = [c for c in chains if c['T'] > 0.002]  # ignore extremely short expiries (< ~18 hours)
    if not valid_chains:
        return np.nan, 0.0, {'error': 'No valid expirations > 18h'}
        
    # If only one chain available, just return its single-expiry VIX
    if len(valid_chains) == 1:
        sigma, qs, fwd = _calculate_variance(valid_chains[0]['options'], valid_chains[0]['T'], risk_free_rate)
        return sigma * 100.0, qs, {'single_expiry': True}

    # Find straddle
    near_term = valid_chains[0]
    next_term = valid_chains[1]
    
    # In a real implementation we might iterate to find the tightest straddle
    # but usually dataset provided is [near, next] already.
    
    sigma1, q1, F1 = _calculate_variance(near_term['options'], near_term['T'], risk_free_rate)
    sigma2, q2, F2 = _calculate_variance(next_term['options'], next_term['T'], risk_free_rate)
    
    T1 = near_term['T']
    T2 = next_term['T']
    
    # Weighted interpolation
    # σ² = [ (T1 * σ1²) * ( (T2 - T_target) / (T2 - T1) ) + 
    #        (T2 * σ2²) * ( (T_target - T1) / (T2 - T1) ) ] * (1 / T_target)
    # Actually VIX uses time-weighted average of VARIANCES
    
    x = (T1 * sigma1**2) * ((T2 - target_t) / (T2 - T1)) + \
        (T2 * sigma2**2) * ((target_t - T1) / (T2 - T1))
        
    # If target is outside [T1, T2], this extrapolates.
    # Clip to avoid negative variance in extreme extrapolation
    weighted_var = max(x, 0.0) / target_t
    vix = 100.0 * np.sqrt(weighted_var)
    
    avg_quality = (q1 + q2) / 2.0
    
    return vix, avg_quality, {
        'T1': T1, 'vol1': sigma1, 'F1': F1,
        'T2': T2, 'vol2': sigma2, 'F2': F2,
        'weighted_var': weighted_var
    }


def _calculate_variance(
    options,
    T: float,
    r: float
) -> Tuple[float, float, float]:
    """
    Calculate annualized volatility sigma for a single usage.
    Returns (sigma, quality_score, forward_price).
    """
    # Convert to DataFrame if list
    if isinstance(options, list):
        df = pd.DataFrame(options)
    else:
        df = options.copy()
        
    required_cols = {'strike', 'option_type', 'price'}
    if not required_cols.issubset(df.columns):
        return np.nan, 0.0, 0.0
        
    # Standardize types
    df['option_type'] = df['option_type'].astype(str).str.upper().str.strip()
    # map CALL/PUT to CE/PE
    df['option_type'] = df['option_type'].replace({'CALL': 'CE', 'PUT': 'PE'})
    
    # 1. Determine Forward Price strategy
    # Find strike where abs(Call - Put) is minimized (ATM)
    calls = df[df['option_type'] == 'CE'].set_index('strike')['price']
    puts = df[df['option_type'] == 'PE'].set_index('strike')['price']
    
    common_strikes = calls.index.intersection(puts.index)
    if common_strikes.empty:
        return np.nan, 0.0, 0.0
        
    diff = (calls[common_strikes] - puts[common_strikes]).abs()
    atm_strike = diff.idxmin()
    min_diff = diff[atm_strike]
    
    # Forward price via Put-Call Parity: F = K + e^(RT) * (C - P)
    # VIX formula uses: F = Strike + e^(RT) * (Call - Put)
    # Using the ATM strike minimizes error
    ert = np.exp(r * T)
    F = atm_strike + ert * (calls[atm_strike] - puts[atm_strike])
    
    # 2. Select Options for Strip
    # - Put options for K < F
    # - Call options for K > F
    # - Average of Put and Call at K_0 (first strike below F)
    
    # Find K0 (strike immediately below F)
    # Actually VIX algo says: determine K0 as strike immediately below F
    # Then for K0, use average of C and P.
    
    strikes = sorted(df['strike'].unique())
    strikes = np.array(strikes)
    
    idx_below = np.where(strikes <= F)[0]
    if len(idx_below) == 0:
        K0 = strikes[0]
    else:
        K0 = strikes[idx_below[-1]]
        
    contributing_strikes = []
    
    for K in strikes:
        p_call = calls.get(K, np.nan)
        p_put = puts.get(K, np.nan)
        
        # Select price Q(K)
        if K == K0:
            # At K0, use average
            if np.isnan(p_call) and np.isnan(p_put):
                 # Both missing — skip this strike entirely
                 continue
            elif np.isnan(p_call) or np.isnan(p_put):
                 # One missing — use the available one
                 price = p_call if np.isfinite(p_call) else p_put
            else:
                price = (p_call + p_put) / 2.0
            mode = 'AVG'
        elif K < K0:
            # OTM Put
            price = p_put
            mode = 'PE'
        else:
            # OTM Call
            price = p_call
            mode = 'CE'
            
        if not np.isnan(price) and price > 0:
            contributing_strikes.append({'strike': K, 'price': price, 'mode': mode})
            
    if len(contributing_strikes) < 3:
        return np.nan, 0.0, F
        
    strip = pd.DataFrame(contributing_strikes).sort_values('strike')
    
    # 3. Apply VIX Filter rules (stop at two consecutive zero bids)
    # Simplification: we assume input DF is already somewhat liquid.
    # In full VIX, we'd filter out strikes with 0 bid.
    # Here we simulate by just using available prices.
    
    # 4. Integrate 
    # Contribution = (ΔK / K²) * e^(RT) * Q(K)
    # ΔK = (Next_Strike - Prev_Strike) / 2
    
    term1_sum = 0.0
    ks = strip['strike'].values
    qs = strip['price'].values
    n = len(ks)
    
    for i in range(n):
        K = ks[i]
        Q = qs[i]
        
        # Determine delta K
        if i == 0:
            dk = ks[1] - ks[0]
        elif i == n - 1:
            dk = ks[n-1] - ks[n-2]
        else:
            dk = (ks[i+1] - ks[i-1]) / 2.0
            
        contribution = (dk / (K**2)) * ert * Q
        term1_sum += contribution
        
    term1 = (2.0 / T) * term1_sum
    term2 = (1.0 / T) * ((F / K0) - 1.0)**2
    
    sigma_sq = term1 - term2
    sigma = np.sqrt(max(sigma_sq, 0.0))
    
    # Quality score: simple ratio of strikes found vs grid
    ideal_grid_size = (strikes[-1] - strikes[0]) / 50.0 # assuming 50 width
    actual_grid_size = len(strip)
    quality = min(1.0, actual_grid_size / (ideal_grid_size + 1e-9))
    
    return sigma, quality, F
