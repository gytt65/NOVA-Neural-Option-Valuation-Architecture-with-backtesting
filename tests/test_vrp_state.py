import numpy as np

from vrp_state import ModelFreeVRPState


def test_vrp_state_deterministic_output():
    est = ModelFreeVRPState()
    returns = np.array([0.001, -0.0008, 0.0005, -0.0004] * 20, dtype=float)
    rn_term = {7: 0.030, 30: 0.028, 60: 0.026}

    s1 = est.compute_state(rn_term, returns)
    s2 = est.compute_state(rn_term, returns)

    assert s1["vrp_level"] == s2["vrp_level"]
    assert s1["vrp_slope"] == s2["vrp_slope"]
    assert s1["state_label"] == s2["state_label"]


def test_vrp_state_stable_with_missing_rn_quotes():
    est = ModelFreeVRPState()
    returns = np.array([0.002, -0.001, 0.0012, -0.0007] * 15, dtype=float)
    rn_term_sparse = {30: 0.03}  # missing 7d/60d

    state = est.compute_state(rn_term_sparse, returns)
    assert np.isfinite(state["vrp_7d"])
    assert np.isfinite(state["vrp_30d"])
    assert np.isfinite(state["vrp_60d"])
    adj = est.parameter_adjustments(state)
    assert 0.85 <= adj["kappa_mult"] <= 1.15
    assert 0.85 <= adj["theta_mult"] <= 1.20
    assert 0.85 <= adj["sigma_v_mult"] <= 1.25

