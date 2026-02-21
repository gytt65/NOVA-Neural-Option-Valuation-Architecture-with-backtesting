import datetime as dt

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from model_free_variance import (
    compute_variance_for_expiry,
    compute_vix_30d,
    compute_vix_30d_with_details,
)


def _bs_price(S, K, T, r, sigma, opt_type):
    d1 = (np.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if opt_type == "CE":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def _make_chain(S, T, r, sigma, strikes):
    rows = []
    for K in strikes:
        c = _bs_price(S, K, T, r, sigma, "CE")
        p = _bs_price(S, K, T, r, sigma, "PE")
        rows.append({"strike": K, "option_type": "CE", "bid": c * 0.995, "ask": c * 1.005})
        rows.append({"strike": K, "option_type": "PE", "bid": p * 0.995, "ask": p * 1.005})
    return pd.DataFrame(rows)


def test_nse_vix_engine_toy_recovery():
    S = 23500.0
    r = 0.065
    sigma = 0.20
    strikes = np.arange(20000, 27050, 50)

    now_ts = dt.datetime(2026, 2, 20, 10, 0)
    exp_near = now_ts + dt.timedelta(days=20)
    exp_next = now_ts + dt.timedelta(days=40)
    T1 = 20.0 / 365.0
    T2 = 40.0 / 365.0

    c1 = _make_chain(S, T1, r, sigma, strikes)
    c2 = _make_chain(S, T2, r, sigma, strikes)

    var1 = compute_variance_for_expiry(c1, forward=S * np.exp(r * T1), r=r, now_ts=now_ts, expiry_ts=exp_near)
    assert np.sqrt(var1) == pytest.approx(sigma, abs=0.02)

    vix30 = compute_vix_30d(
        c1, c2,
        forward_near=S * np.exp(r * T1),
        forward_next=S * np.exp(r * T2),
        r=r,
        now_ts=now_ts,
        expiry_near_ts=exp_near,
        expiry_next_ts=exp_next,
    )
    assert vix30 == pytest.approx(20.0, abs=1.0)


def test_spread_filter_and_spline_fill_keeps_engine_stable():
    S = 23500.0
    r = 0.06
    sigma = 0.18
    T = 30.0 / 365.0
    strikes = np.arange(21000, 26050, 50)
    chain = _make_chain(S, T, r, sigma, strikes)

    # Create invalid spreads and missing interiors.
    for bad_k in [22500, 23500, 24500]:
        idx = (chain["strike"] == bad_k) & (chain["option_type"] == "CE")
        chain.loc[idx, "ask"] = chain.loc[idx, "bid"] * 2.0  # invalid spread >30%
    for miss_k in [22800, 24000]:
        idx = (chain["strike"] == miss_k) & (chain["option_type"] == "PE")
        chain.loc[idx, ["bid", "ask"]] = np.nan

    now_ts = dt.datetime(2026, 2, 20, 11, 0)
    exp_ts = now_ts + dt.timedelta(days=30)
    var = compute_variance_for_expiry(
        chain,
        forward=S * np.exp(r * T),
        r=r,
        now_ts=now_ts,
        expiry_ts=exp_ts,
    )
    assert np.isfinite(var)
    assert var > 0


def test_30d_interpolation_uses_minutes():
    S = 23500.0
    r = 0.0
    strikes = np.arange(21000, 26050, 50)
    now_ts = dt.datetime(2026, 2, 20, 9, 20)
    exp_near = now_ts + dt.timedelta(days=25, hours=2)
    exp_next = now_ts + dt.timedelta(days=35, hours=4)

    T1 = (exp_near - now_ts).total_seconds() / (365.0 * 24 * 3600.0)
    T2 = (exp_next - now_ts).total_seconds() / (365.0 * 24 * 3600.0)

    c1 = _make_chain(S, T1, r, 0.17, strikes)
    c2 = _make_chain(S, T2, r, 0.23, strikes)

    vix30, details = compute_vix_30d_with_details(
        c1, c2, S, S, r, now_ts, exp_near, exp_next
    )
    assert np.isfinite(vix30)
    assert details["minutes_near"] < details["target_minutes"] < details["minutes_next"]
    # Interpolated 30d vol should lie between near and next vols.
    near_vol = np.sqrt(details["var_near"]) * 100.0
    next_vol = np.sqrt(details["var_next"]) * 100.0
    lo, hi = sorted([near_vol, next_vol])
    assert lo <= vix30 <= hi

