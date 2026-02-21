import numpy as np

from backtester import SyntheticNiftyGenerator


def test_synthetic_generator_returns_history_is_past_only():
    # Determinism check: adding one future day must not mutate prior snapshots.
    g1 = SyntheticNiftyGenerator(seed=123, initial_spot=23500)
    s40 = g1.generate(n_days=40)

    g2 = SyntheticNiftyGenerator(seed=123, initial_spot=23500)
    s41 = g2.generate(n_days=41)

    assert len(s40) == 40
    assert len(s41) == 41

    for i in range(40):
        r40 = np.asarray(s40[i].returns_30d, dtype=float)
        r41 = np.asarray(s41[i].returns_30d, dtype=float)
        assert len(r40) == min(i + 1, 30)
        assert len(r41) == min(i + 1, 30)
        assert np.allclose(r40, r41, atol=1e-12)

