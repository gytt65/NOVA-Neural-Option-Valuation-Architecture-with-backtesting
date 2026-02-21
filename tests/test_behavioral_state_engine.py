import numpy as np

from behavioral_state_engine import BehavioralStateEngine


def test_behavioral_state_engine_output_shape_and_bounds():
    eng = BehavioralStateEngine()
    out = eng.build_market_state(
        regime={"name": "Sideways"},
        vrp={"vrp_level": 0.01},
        srp={"srp_level": 0.0},
        liquidity={"spread": 0.02},
        behavioral_inputs={
            "news_sentiment": 0.6,
            "social_sentiment": 0.2,
            "otm_call_skew": 0.5,
            "put_call_ratio": 0.9,
            "bid_ask_spread": 0.03,
            "volume_oi_ratio": 0.08,
            "total_gex": -1.5e6,
            "gamma_flip": 23500.0,
            "charm": -0.2,
            "vanna": 0.1,
            "dealer_flow_imbalance": 0.4,
        },
    )

    assert "behavioral" in out
    beh = out["behavioral"]
    assert -1.0 <= beh["sentiment"] <= 1.0
    assert -1.0 <= beh["lottery_demand"] <= 1.0
    assert 0.0 <= beh["limits_to_arb"] <= 1.0

    dflow = beh["dealer_flow"]
    assert "total_gex" in dflow
    assert "gex_sign" in dflow
    assert np.isfinite(dflow["gex_sign"])

