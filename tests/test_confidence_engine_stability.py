import numpy as np

from nirv_model import BayesianConfidenceEngine


def test_confidence_engine_handles_nonfinite_paths():
    eng = BayesianConfidenceEngine(n_bootstrap=256)

    s_terminal = np.array(
        [23500.0, 23650.0, np.nan, np.inf, 0.0, 21000.0, 26000.0, -np.inf],
        dtype=float,
    )

    out = eng.compute_profit_probability(
        S_terminal=s_terminal,
        strike=23500.0,
        market_price=120.0,
        r=0.065,
        T=7.0 / 365.0,
        option_type="CE",
        lot_size=65,
        spot=23500.0,
        returns_30d=np.random.normal(0.0, 0.01, 30),
        regime="Sideways",
        iv=0.16,
    )

    assert len(out) == 6
    fair_values = out[-1]
    assert np.all(np.isfinite(fair_values))
    assert np.isfinite(out[0])
    assert np.isfinite(out[1])
    assert np.isfinite(out[2])
    assert np.isfinite(out[3])
    assert np.isfinite(out[4])

