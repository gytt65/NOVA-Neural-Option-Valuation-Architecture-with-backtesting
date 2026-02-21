import numpy as np

from arbfree_surface import ArbFreeSurfaceState
from surface_checks import (
    check_butterfly_arbitrage_slice,
    check_calendar_arbitrage,
    check_monotonicity_and_convexity_of_call_prices,
)


def _bs_call(S, K, T, r, sigma):
    from scipy.stats import norm

    d1 = (np.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def test_surface_no_arbitrage_properties():
    k = np.linspace(-0.3, 0.3, 81)
    w = 0.025 + 0.04 * k * k - 0.002 * k
    ok_bfly, metric_bfly = check_butterfly_arbitrage_slice(w, k_grid=k)
    assert ok_bfly
    assert metric_bfly["min_second_derivative"] >= -1e-8

    t_grid = np.array([7, 30, 60], dtype=float) / 365.0
    k_grid = np.linspace(-0.2, 0.2, 15)
    ok_cal, metric_cal = check_calendar_arbitrage(
        lambda tt, kk: (0.018 + 0.12 * tt) * (1.0 + 0.2 * kk * kk),
        t_grid=t_grid,
        k_grid=k_grid,
    )
    assert ok_cal
    assert metric_cal["min_forward_diff"] >= -1e-8

    S = 23500.0
    T = 30.0 / 365.0
    r = 0.06
    strikes = np.arange(21000, 26050, 100)
    calls = np.array([_bs_call(S, K, T, r, 0.20) for K in strikes])
    ok_calls, metric_calls = check_monotonicity_and_convexity_of_call_prices(strikes, calls)
    assert ok_calls
    assert metric_calls["max_monotonicity_breach"] <= 1e-8


def test_bad_smile_fails_and_repair_triggers():
    spot = 100.0
    T = 0.1
    k = np.linspace(-0.2, 0.2, 5)
    strikes = spot * np.exp(k)
    iv_bad = np.array([0.18, 0.28, 0.40, 0.28, 0.18])  # sharp concave smile
    w_bad = (iv_bad ** 2) * T

    ok_before, _ = check_butterfly_arbitrage_slice(w_bad, k_grid=k)
    assert not ok_before

    surf = ArbFreeSurfaceState()
    surf.add_slice(T, strikes, iv_bad, spot)
    w_before = surf.slices[0]["w"].copy()
    surf.fit()
    w_after = surf.slices[0]["w"]

    # Repair path should alter raw slice values and provide fitted outputs.
    assert not np.allclose(w_before, w_after)
    assert "butterfly_ok" in surf.last_diagnostics
    assert np.isfinite(surf.get_iv(0.0, T))

