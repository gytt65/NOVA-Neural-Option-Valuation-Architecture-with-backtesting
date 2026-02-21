import numpy as np

from quant_engine import GEXCalculator


def test_gex_enhanced_vector_preserves_legacy_and_adds_metrics():
    strikes = [23400, 23500, 23600]
    call_oi = {23400: 1000, 23500: 1500, 23600: 1200}
    put_oi = {23400: 900, 23500: 1300, 23600: 1400}
    call_gamma = {23400: 0.02, 23500: 0.03, 23600: 0.025}
    put_gamma = {23400: 0.018, 23500: 0.028, 23600: 0.03}
    call_charm = {23400: -0.001, 23500: -0.0015, 23600: -0.0012}
    put_charm = {23400: -0.0008, 23500: -0.0012, 23600: -0.0011}
    call_vanna = {23400: 0.002, 23500: 0.0025, 23600: 0.0022}
    put_vanna = {23400: 0.0018, 23500: 0.0021, 23600: 0.0024}

    out = GEXCalculator.compute_gex(
        spot=23500,
        strikes=strikes,
        call_oi=call_oi,
        put_oi=put_oi,
        call_gamma=call_gamma,
        put_gamma=put_gamma,
        lot_size=65,
        call_charm=call_charm,
        put_charm=put_charm,
        call_vanna=call_vanna,
        put_vanna=put_vanna,
        bucket_width=100.0,
    )

    # Legacy compatibility
    assert "total_gex" in out
    assert "gex_by_strike" in out
    assert "regime" in out
    assert "zero_gamma_strike" in out

    # Additive enhancements
    assert "gex_sign" in out
    assert out["gex_sign"] in (-1.0, 0.0, 1.0)
    assert "bucketed_gex" in out and isinstance(out["bucketed_gex"], dict)
    assert "gamma_flip" in out and np.isfinite(out["gamma_flip"])
    assert "charm" in out and np.isfinite(out["charm"])
    assert "vanna" in out and np.isfinite(out["vanna"])

