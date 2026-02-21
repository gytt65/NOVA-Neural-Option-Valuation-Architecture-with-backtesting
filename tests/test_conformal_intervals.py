import numpy as np

from omega_features import set_features
from omega_model import MLPricingCorrector


class _IdentityScaler:
    def transform(self, x):
        return x


class _ZeroModel:
    def predict(self, x):
        return np.zeros(len(x), dtype=float)


def test_conformal_interval_coverage_toy_distribution(tmp_path):
    set_features(USE_CONFORMAL_INTERVALS=True)
    try:
        ml = MLPricingCorrector(model_path=str(tmp_path / "dummy.joblib"))
        ml.is_trained = True
        ml.scaler = _IdentityScaler()
        ml.model = _ZeroModel()

        rng = np.random.default_rng(7)
        residuals = rng.normal(0.0, 0.05, 50000)
        ml._conformal_q_global = float(np.quantile(np.abs(residuals), 0.90))
        ml._conformal_q_by_regime = {}

        feats = {"regime_sideways": 1.0}
        corr, conf, lo, hi = ml.predict_correction_with_interval(feats)
        assert corr == 0.0
        assert lo < hi

        coverage = np.mean((residuals >= lo) & (residuals <= hi))
        assert 0.885 <= coverage <= 0.925
    finally:
        set_features()


def test_conformal_flag_off_preserves_baseline_interval_behavior(tmp_path):
    set_features(USE_CONFORMAL_INTERVALS=False)
    try:
        ml = MLPricingCorrector(model_path=str(tmp_path / "dummy.joblib"))
        ml.is_trained = True
        ml.scaler = _IdentityScaler()
        ml.model = _ZeroModel()
        feats = {"regime_sideways": 1.0}
        c0, conf0 = ml.predict_correction(feats)
        c1, conf1, lo, hi = ml.predict_correction_with_interval(feats)
        assert c0 == c1
        assert conf0 == conf1
        assert lo == c1 and hi == c1
    finally:
        set_features()

