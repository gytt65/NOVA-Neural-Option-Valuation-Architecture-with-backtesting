from omega_features import (
    OmegaFeatures,
    set_best_mode_macbook,
    set_best_mode_max_accuracy,
    set_features,
)


def test_best_mode_macbook_profile():
    feat = OmegaFeatures.best_mode_macbook()
    d = feat.to_dict()

    assert d["USE_NSE_CONTRACT_SPECS"] is True
    assert d["USE_NSE_VIX_ENGINE"] is True
    assert d["USE_ESSVI_SURFACE"] is True
    assert d["USE_SVI_FIXED_POINT_WARMSTART"] is True
    assert d["USE_MODEL_FREE_VRP"] is True
    assert d["USE_TIERED_PRICER"] is True
    assert d["USE_CONFORMAL_INTERVALS"] is True
    assert d["USE_RESEARCH_HIGH_CONVICTION"] is False
    assert d["USE_OOS_RELIABILITY_GATE"] is False

    # CPU-friendly default keeps tail correction off unless explicitly needed.
    assert d["USE_TAIL_CORRECTED_VARIANCE"] is False
    assert d["USE_LIQUIDITY_WEIGHTING"] is False
    assert d["USE_INTERVAL_LOSS"] is False
    assert d["USE_STALENESS_FEATURES"] is False
    assert d["USE_ENHANCED_RANKING"] is False
    assert d["USE_IMPROVED_VIX_ESTIMATOR"] is False
    assert d["ENFORCE_STATIC_NO_ARB"] is False


def test_set_best_mode_updates_singleton():
    set_features()  # reset baseline
    feat = set_best_mode_macbook()
    d = feat.to_dict()
    assert d["USE_TIERED_PRICER"] is True
    assert d["USE_NSE_VIX_ENGINE"] is True


def test_best_mode_max_accuracy_profile():
    feat = OmegaFeatures.best_mode_max_accuracy()
    d = feat.to_dict()
    assert d["USE_NSE_CONTRACT_SPECS"] is True
    assert d["USE_NSE_VIX_ENGINE"] is True
    assert d["USE_ESSVI_SURFACE"] is True
    assert d["USE_SVI_FIXED_POINT_WARMSTART"] is True
    assert d["USE_MODEL_FREE_VRP"] is True
    assert d["USE_TIERED_PRICER"] is True
    assert d["USE_CONFORMAL_INTERVALS"] is True
    assert d["USE_TAIL_CORRECTED_VARIANCE"] is True
    assert d["USE_RESEARCH_HIGH_CONVICTION"] is True
    assert d["USE_OOS_RELIABILITY_GATE"] is True


def test_set_best_mode_max_accuracy_updates_singleton():
    set_features()  # reset baseline
    feat = set_best_mode_max_accuracy()
    d = feat.to_dict()
    assert d["USE_TAIL_CORRECTED_VARIANCE"] is True
    assert d["USE_NSE_VIX_ENGINE"] is True
    assert d["USE_RESEARCH_HIGH_CONVICTION"] is True
    assert d["USE_OOS_RELIABILITY_GATE"] is True


def test_new_flags_default_off_for_regression_safety():
    d = OmegaFeatures.all_off().to_dict()
    assert d["USE_LIQUIDITY_WEIGHTING"] is False
    assert d["USE_INTERVAL_LOSS"] is False
    assert d["USE_STALENESS_FEATURES"] is False
    assert d["USE_ENHANCED_RANKING"] is False
    assert d["USE_IMPROVED_VIX_ESTIMATOR"] is False
    assert d["ENFORCE_STATIC_NO_ARB"] is False
