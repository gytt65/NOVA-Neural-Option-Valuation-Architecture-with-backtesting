import datetime as dt

import numpy as np

from model_free_variance import estimate_forward_from_chain, compute_variance_for_expiry


def _build_parity_chain(forward, strikes, t_years=20.0 / 365.0, r=0.06):
    """
    Build a synthetic parity-consistent chain:
      C - P = exp(-rT) * (F - K)
    """
    disc = np.exp(-r * t_years)
    rows = []
    for k in strikes:
        intrinsic = max(forward - k, 0.0)
        time_val = 8.0 + 0.05 * abs(k - forward) / 50.0
        call_mid = max(disc * intrinsic + time_val, 0.5)
        put_mid = max(call_mid - disc * (forward - k), 0.5)
        rows.append({"strike": float(k), "option_type": "CE", "bid": call_mid * 0.995, "ask": call_mid * 1.005})
        rows.append({"strike": float(k), "option_type": "PE", "bid": put_mid * 0.995, "ask": put_mid * 1.005})
    return rows


def test_forward_estimator_legacy_and_improved():
    f_true = 23620.0
    strikes = np.arange(23200, 24050, 50)
    chain = _build_parity_chain(f_true, strikes)
    now_ts = dt.datetime(2026, 2, 19, 9, 30)
    exp_ts = now_ts + dt.timedelta(days=20)

    f_legacy = estimate_forward_from_chain(chain, r=0.06, now_ts=now_ts, expiry_ts=exp_ts, method="legacy")
    f_improved = estimate_forward_from_chain(chain, r=0.06, now_ts=now_ts, expiry_ts=exp_ts, method="improved")

    assert np.isfinite(f_legacy) and np.isfinite(f_improved)
    assert abs(f_legacy - f_true) < 80.0
    assert abs(f_improved - f_true) < 60.0


def test_variance_engine_accepts_missing_forward_with_estimator():
    f_true = 23600.0
    strikes = np.arange(23200, 24050, 50)
    chain = _build_parity_chain(f_true, strikes)
    now_ts = dt.datetime(2026, 2, 19, 9, 30)
    exp_ts = now_ts + dt.timedelta(days=20)

    var_legacy = compute_variance_for_expiry(
        chain_slice=chain,
        forward=None,
        r=0.06,
        now_ts=now_ts,
        expiry_ts=exp_ts,
        forward_method="legacy",
    )
    var_improved = compute_variance_for_expiry(
        chain_slice=chain,
        forward=None,
        r=0.06,
        now_ts=now_ts,
        expiry_ts=exp_ts,
        forward_method="improved",
    )
    assert np.isfinite(var_legacy) and var_legacy >= 0
    assert np.isfinite(var_improved) and var_improved >= 0

