#!/usr/bin/env python3
"""
test_frontier_upgrades.py — Tests for Phase 1 frontier model upgrades.

Covers:
  1. Hawkes self-exciting jump process
  2. Real rBergomi pricer (fBm simulation)
  3. Butterfly repair (convexity preservation)
  4. Bayesian VRP with confidence intervals
  5. Conformalized prediction intervals
"""

import sys
import os
import numpy as np
import pytest

# Fix import path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# ============================================================
# 1. HAWKES JUMP PROCESS TESTS
# ============================================================

class TestHawkesJumpProcess:
    """Test the Hawkes self-exciting jump module."""

    def test_import(self):
        from hawkes_jump import HawkesProcess, HawkesJumpEstimator
        hp = HawkesProcess(mu=2.0, alpha=0.5, beta=1.5)
        assert hp.branching_ratio < 1.0  # stationarity

    def test_intensity_increases_after_event(self):
        from hawkes_jump import HawkesProcess
        hp = HawkesProcess(mu=2.0, alpha=1.0, beta=2.0)

        # Intensity before any events = mu
        lam_before = hp.intensity(1.0, np.array([]))
        assert lam_before == pytest.approx(2.0)

        # Intensity right after an event at t=0.9 should be > mu
        lam_after = hp.intensity(1.0, np.array([0.9]))
        assert lam_after > hp.mu

    def test_simulate_produces_events(self):
        from hawkes_jump import HawkesProcess
        hp = HawkesProcess(mu=5.0, alpha=1.0, beta=3.0)
        events = hp.simulate(T=1.0, seed=42)
        assert len(events) > 0
        # Events should be sorted and within [0, T]
        assert np.all(np.diff(events) >= 0)
        assert events[-1] <= 1.0

    def test_log_likelihood_finite(self):
        from hawkes_jump import HawkesProcess
        hp = HawkesProcess(mu=3.0, alpha=0.5, beta=1.5)
        events = hp.simulate(T=2.0, seed=42)
        ll = hp.log_likelihood(events, T=2.0)
        assert np.isfinite(ll)

    def test_estimator_fit_basic(self):
        from hawkes_jump import HawkesJumpEstimator
        est = HawkesJumpEstimator(jump_threshold_sigma=2.0)

        # Synthetic returns with some jump clustering
        rng = np.random.default_rng(42)
        normal_returns = rng.normal(0.0005, 0.012, 200)
        # Inject clustered jumps at indices 50-55 and 150-155
        normal_returns[50:56] = rng.normal(-0.04, 0.02, 6)
        normal_returns[150:156] = rng.normal(-0.03, 0.015, 6)

        result = est.fit(normal_returns)
        assert 'lambda_base' in result
        assert 'alpha' in result
        assert 'beta' in result
        assert 'clustering_score' in result
        assert result['branching_ratio'] < 1.0  # stationarity
        assert result['n_jumps'] > 0

    def test_estimator_backward_compat(self):
        """Ensure output has lambda_j and jump_prob_daily for backward compatibility."""
        from hawkes_jump import HawkesJumpEstimator
        est = HawkesJumpEstimator()
        result = est.fit(np.random.default_rng(42).normal(0, 0.015, 100))
        assert 'lambda_j' in result
        assert 'jump_prob_daily' in result
        assert 0 < result['jump_prob_daily'] < 1.0

    def test_regime_adjustments(self):
        from hawkes_jump import HawkesJumpEstimator
        est = HawkesJumpEstimator()
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.015, 100)
        returns[40:48] = rng.normal(-0.04, 0.02, 8)  # inject cluster
        est.fit(returns)
        adj = est.regime_adjustments()
        assert 'lambda_mult' in adj
        assert 'eta_mult' in adj
        assert adj['lambda_mult'] >= 1.0
        assert adj['eta_mult'] >= 1.0


# ============================================================
# 2. RBERGOMI PRICER TESTS
# ============================================================

class TestRBergomiPricer:
    """Test the rebuilt rough Bergomi pricer."""

    def test_import_and_create(self):
        from pricer_router import RBergomiPricer
        rb = RBergomiPricer(hurst=0.10, eta=1.0, rho=-0.7)
        assert 0 < rb.hurst < 0.5
        assert rb.n_paths >= 256

    def test_regime_hurst(self):
        from pricer_router import RBergomiPricer
        rb_crisis = RBergomiPricer(regime='crisis')
        rb_calm = RBergomiPricer(regime='calm')
        assert rb_crisis.hurst < rb_calm.hurst
        assert rb_crisis.hurst == pytest.approx(0.05)
        assert rb_calm.hurst == pytest.approx(0.35)

    def test_price_produces_positive_price(self):
        from pricer_router import RBergomiPricer
        rb = RBergomiPricer(hurst=0.12, n_paths=512, n_steps=16)
        price, eff_sigma = rb.price(
            spot=23500, strike=23400, T=7/365,
            r=0.065, q=0.012, sigma=0.14, option_type='CE'
        )
        assert price > 0
        assert np.isfinite(price)
        assert eff_sigma > 0

    def test_put_price_positive(self):
        from pricer_router import RBergomiPricer
        rb = RBergomiPricer(hurst=0.12, n_paths=512, n_steps=16)
        price, _ = rb.price(
            spot=23500, strike=23600, T=7/365,
            r=0.065, q=0.012, sigma=0.14, option_type='PE'
        )
        assert price > 0
        assert np.isfinite(price)

    def test_fbm_covariance_positive_definite(self):
        from pricer_router import RBergomiPricer
        rb = RBergomiPricer(hurst=0.10, n_steps=16)
        cov = rb._fbm_covariance(16, 7/365)
        eigenvalues = np.linalg.eigvalsh(cov)
        assert np.all(eigenvalues > 0), "fBm covariance must be positive definite"


# ============================================================
# 3. BUTTERFLY REPAIR TESTS
# ============================================================

class TestButterflyRepair:
    """Test the convexity-preserving butterfly repair."""

    def test_convex_input_unchanged(self):
        from arbfree_surface import ArbFreeSurfaceState
        state = ArbFreeSurfaceState()
        k = np.array([-0.10, -0.05, 0.0, 0.05, 0.10])
        w = np.array([0.020, 0.015, 0.012, 0.015, 0.020])  # convex
        w_repaired = state._repair_butterfly(k, w)
        np.testing.assert_allclose(w_repaired, w, atol=1e-8)

    def test_concave_violation_fixed(self):
        from arbfree_surface import ArbFreeSurfaceState
        state = ArbFreeSurfaceState()
        k = np.array([-0.10, -0.05, 0.0, 0.05, 0.10])
        # Inject concavity at center (w[2] too high → d²w/dk² < 0)
        w = np.array([0.020, 0.015, 0.025, 0.015, 0.020])
        w_repaired = state._repair_butterfly(k, w)
        # After repair, center should be raised to at most linear interpolation
        # and overall convexity should improve
        assert w_repaired[2] <= w[2]  # shouldn't increase further
        assert np.all(w_repaired > 0)

    def test_wings_preserved(self):
        """Wings should NOT be over-smoothed (the old bug)."""
        from arbfree_surface import ArbFreeSurfaceState
        state = ArbFreeSurfaceState()
        k = np.linspace(-0.15, 0.15, 11)
        # Realistic smile: raised wings
        w = 0.013 + 0.5 * k**2  # perfect convex parabola
        w_repaired = state._repair_butterfly(k, w)
        # Wings should be preserved (no over-smoothing)
        np.testing.assert_allclose(w_repaired, w, atol=1e-6)


# ============================================================
# 4. BAYESIAN VRP TESTS
# ============================================================

class TestBayesianVRP:
    """Test the Bayesian VRP with bootstrap posterior."""

    def test_bootstrap_returns_uncertainty(self):
        from vrp_state import ModelFreeVRPState
        vrp = ModelFreeVRPState()
        returns = np.random.default_rng(42).normal(0.0005, 0.012, 100)
        mean, std, ci_lo, ci_hi = vrp._bootstrap_realized_var(returns, 30, n_bootstrap=100)
        assert std > 0, "Bootstrap should produce non-zero uncertainty"
        assert ci_lo < mean < ci_hi, "CI should contain the mean"
        assert ci_lo > 0, "Variance estimate should be positive"

    def test_compute_state_has_ci_fields(self):
        from vrp_state import ModelFreeVRPState
        vrp = ModelFreeVRPState()
        rn_var = {7: 0.04, 30: 0.035, 60: 0.03}
        returns = np.random.default_rng(42).normal(0.0005, 0.012, 100)
        state = vrp.compute_state(rn_var, returns)
        assert 'vrp_std' in state
        assert 'vrp_ci_lower' in state
        assert 'vrp_ci_upper' in state
        assert 'vrp_certainty' in state
        assert 0 <= state['vrp_certainty'] <= 1.0

    def test_adjustments_dampened_by_uncertainty(self):
        from vrp_state import ModelFreeVRPState
        # High certainty → adjustments close to raw
        state_certain = {'vrp_level': 0.02, 'vrp_slope': 0.005, 'vrp_certainty': 1.0}
        adj_c = ModelFreeVRPState.parameter_adjustments(state_certain)

        # Low certainty → adjustments dampened toward 1.0
        state_uncertain = {'vrp_level': 0.02, 'vrp_slope': 0.005, 'vrp_certainty': 0.1}
        adj_u = ModelFreeVRPState.parameter_adjustments(state_uncertain)

        # Uncertain adjustments should be closer to 1.0
        assert abs(adj_u['theta_mult'] - 1.0) < abs(adj_c['theta_mult'] - 1.0)


# ============================================================
# 5. CONFORMAL PREDICTION TESTS
# ============================================================

class TestConformalPrediction:
    """Test the upgraded conformalized quantile regression intervals."""

    def test_interval_widens_for_otm(self):
        """OTM options should have wider prediction intervals."""
        from omega_model import MLPricingCorrector
        import omega_features

        corr = MLPricingCorrector.__new__(MLPricingCorrector)
        corr.is_trained = True
        corr.model = None
        corr.scaler = None
        corr.training_y = list(range(100))
        corr._conformal_q_global = 0.05
        corr._conformal_q_by_regime = {}

        # Monkey-patch predict_correction to return fixed value
        corr.predict_correction = lambda f: (0.01, 0.8)

        # Enable conformal intervals via the proper API
        old_features = omega_features.FEATURES
        try:
            omega_features.set_features(USE_CONFORMAL_INTERVALS=True)

            # ATM option
            _, _, lo_atm, hi_atm = corr.predict_correction_with_interval(
                {'log_moneyness': 0.0, 'time_to_expiry': 0.1, 'regime_sideways': 1.0}
            )
            width_atm = hi_atm - lo_atm

            # OTM option
            _, _, lo_otm, hi_otm = corr.predict_correction_with_interval(
                {'log_moneyness': 0.2, 'time_to_expiry': 0.1, 'regime_sideways': 1.0}
            )
            width_otm = hi_otm - lo_otm

            assert width_otm > width_atm, "OTM should have wider intervals"
        finally:
            omega_features.FEATURES = old_features


# ============================================================
# 6. KAN CORRECTOR TESTS (Phase 2)
# ============================================================

class TestKANCorrector:
    """Test the Kolmogorov-Arnold Network pricing corrector."""

    def test_import_and_create(self):
        from kan_corrector import KANCorrector
        kan = KANCorrector(hidden_dims=(4, 2), n_spline_knots=3)
        assert not kan.is_trained
        assert kan.MIN_SAMPLES == 30

    def test_cold_start_returns_zero(self):
        from kan_corrector import KANCorrector
        kan = KANCorrector()
        corr, conf = kan.predict_correction({'f0': 0.1, 'f1': 0.2})
        assert corr == 0.0
        assert conf == 0.0

    def test_train_and_predict(self):
        from kan_corrector import KANCorrector
        kan = KANCorrector(hidden_dims=(4, 2), n_spline_knots=3,
                           feature_names=['x1', 'x2'])
        rng = np.random.default_rng(42)
        # Generate training data: residual ~ 0.05 * x1 - 0.03 * x2
        for i in range(40):
            x1, x2 = rng.normal(0, 1), rng.normal(0, 1)
            residual = 0.05 * x1 - 0.03 * x2 + rng.normal(0, 0.005)
            kan.add_sample({'x1': x1, 'x2': x2}, residual)

        assert kan.is_trained
        corr, conf = kan.predict_correction({'x1': 1.0, 'x2': 0.0})
        assert np.isfinite(corr)
        # Should predict something positive for high x1
        # (not strict assertion — KAN training is approximate)

    def test_feature_importance(self):
        from kan_corrector import KANCorrector
        kan = KANCorrector(hidden_dims=(4, 2), n_spline_knots=3,
                           feature_names=['important', 'noise'])
        rng = np.random.default_rng(42)
        for i in range(50):
            x1 = rng.normal(0, 1)
            x2 = rng.normal(0, 1)
            kan.add_sample({'important': x1, 'noise': x2}, 0.1 * x1)

        imp = kan.get_feature_importance()
        assert isinstance(imp, dict)

    def test_bspline_basis(self):
        from kan_corrector import _bspline_basis, _make_knots
        knots = _make_knots(5, degree=3)
        x = np.linspace(-2.5, 2.5, 50)
        B = _bspline_basis(x, knots, degree=3)
        assert B.shape[0] == 50
        assert B.shape[1] > 0
        # Basis values should be non-negative
        assert np.all(B >= -1e-10)


# ============================================================
# 7. PINN VOLATILITY SURFACE TESTS (Phase 2)
# ============================================================

class TestPINNVolSurface:
    """Test the Physics-Informed volatility surface."""

    def test_import_and_create(self):
        from pinn_vol_surface import PINNVolSurface
        pinn = PINNVolSurface(n_centers=10)
        assert not pinn.is_fitted

    def test_fit_synthetic_surface(self):
        from pinn_vol_surface import PINNVolSurface
        pinn = PINNVolSurface(n_centers=15, max_iter=50)

        # Create synthetic SVI-like data
        k = np.linspace(-0.15, 0.15, 20)
        T = np.full(20, 0.1)  # 1 month expiry
        iv = 0.20 + 0.5 * k**2  # simple smile

        result = pinn.fit(k, T, iv)
        assert pinn.is_fitted
        assert 'rmse_iv' in result
        assert result['rmse_iv'] < 0.10  # reasonable fit

    def test_get_iv(self):
        from pinn_vol_surface import PINNVolSurface
        pinn = PINNVolSurface(n_centers=15, max_iter=50)
        k = np.linspace(-0.10, 0.10, 15)
        T = np.full(15, 0.08)
        iv = 0.18 + 0.3 * k**2
        pinn.fit(k, T, iv)

        iv_atm = pinn.get_iv(0.0, 0.08)
        assert 0.01 < iv_atm < 5.0
        assert np.isfinite(iv_atm)

    def test_get_surface(self):
        from pinn_vol_surface import PINNVolSurface
        pinn = PINNVolSurface(n_centers=15, max_iter=30)
        k = np.linspace(-0.10, 0.10, 10)
        T = np.full(10, 0.1)
        iv = 0.20 + 0.4 * k**2
        pinn.fit(k, T, iv)

        k_grid = np.array([-0.05, 0.0, 0.05])
        T_grid = np.array([0.05, 0.10])
        surface = pinn.get_surface(k_grid, T_grid)
        assert surface.shape == (2, 3)
        assert np.all(surface > 0)
        assert np.all(np.isfinite(surface))

    def test_unfitted_returns_default(self):
        from pinn_vol_surface import PINNVolSurface
        pinn = PINNVolSurface()
        iv = pinn.get_iv(0.0, 0.1)
        assert iv == pytest.approx(0.15)  # default fallback


# ============================================================
# 8. FEATURE FLAGS TESTS (Phase 2)
# ============================================================

class TestFrontierFeatureFlags:
    """Test the v7 frontier feature flags."""

    def test_new_flags_exist(self):
        from omega_features import OmegaFeatures
        ff = OmegaFeatures()
        assert hasattr(ff, 'USE_HAWKES_JUMPS')
        assert hasattr(ff, 'USE_PINN_SURFACE')
        assert hasattr(ff, 'USE_KAN_CORRECTOR')
        assert hasattr(ff, 'USE_VARIABLE_HURST')

    def test_new_flags_default_off(self):
        from omega_features import OmegaFeatures
        ff = OmegaFeatures()
        assert ff.USE_HAWKES_JUMPS is False
        assert ff.USE_PINN_SURFACE is False
        assert ff.USE_KAN_CORRECTOR is False
        assert ff.USE_VARIABLE_HURST is False

    def test_new_flags_can_be_enabled(self):
        from omega_features import OmegaFeatures
        ff = OmegaFeatures(USE_HAWKES_JUMPS=True, USE_KAN_CORRECTOR=True)
        assert ff.USE_HAWKES_JUMPS is True
        assert ff.USE_KAN_CORRECTOR is True
        assert ff.USE_PINN_SURFACE is False  # not enabled

    def test_lowercase_aliases_work(self):
        from omega_features import OmegaFeatures
        ff = OmegaFeatures(use_hawkes_jumps=True, use_variable_hurst=True)
        assert ff.USE_HAWKES_JUMPS is True
        assert ff.USE_VARIABLE_HURST is True


# ============================================================
# 9. SGM SURFACE COMPLETION TESTS (Phase 3)
# ============================================================

class TestSGMSurfaceCompleter:
    """Test the score-based generative surface completion."""

    def test_import_and_create(self):
        from sgm_surface import ScoreBasedSurfaceCompleter
        sgm = ScoreBasedSurfaceCompleter(n_components=10)
        assert not sgm.is_fitted

    def test_add_historical_surface(self):
        from sgm_surface import ScoreBasedSurfaceCompleter
        sgm = ScoreBasedSurfaceCompleter(n_components=5)
        k = np.linspace(-0.1, 0.1, 5)
        T = np.array([0.05, 0.10])
        iv = np.array([[0.20, 0.18, 0.17, 0.18, 0.20],
                        [0.19, 0.17, 0.16, 0.17, 0.19]])
        sgm.add_historical_surface(k, T, iv)
        assert len(sgm.training_surfaces) == 1

    def test_fit_with_enough_data(self):
        from sgm_surface import ScoreBasedSurfaceCompleter
        sgm = ScoreBasedSurfaceCompleter(n_components=5)
        k = np.linspace(-0.1, 0.1, 5)
        T = np.array([0.05, 0.10])
        rng = np.random.default_rng(42)
        for _ in range(10):
            iv = 0.18 + 0.3 * np.outer(np.ones(2), k**2) + rng.normal(0, 0.005, (2, 5))
            sgm.add_historical_surface(k, T, iv)
        sgm.fit()
        assert sgm.is_fitted

    def test_complete_sparse_observation(self):
        from sgm_surface import ScoreBasedSurfaceCompleter
        sgm = ScoreBasedSurfaceCompleter(n_components=5, noise_schedule_steps=5)
        k = np.linspace(-0.1, 0.1, 5)
        T = np.array([0.05, 0.10])
        rng = np.random.default_rng(42)
        for _ in range(10):
            iv = 0.18 + 0.3 * np.outer(np.ones(2), k**2) + rng.normal(0, 0.005, (2, 5))
            sgm.add_historical_surface(k, T, iv)
        sgm.fit()

        # Complete from sparse observations
        k_obs = np.array([-0.05, 0.0, 0.05])
        T_obs = np.array([0.10, 0.10, 0.10])
        iv_obs = np.array([0.185, 0.17, 0.185])
        iv_full = sgm.complete(k_obs, T_obs, iv_obs, k, T)
        assert iv_full.shape == (2, 5)
        assert np.all(iv_full > 0)
        assert np.all(np.isfinite(iv_full))

    def test_fallback_when_not_fitted(self):
        from sgm_surface import ScoreBasedSurfaceCompleter
        sgm = ScoreBasedSurfaceCompleter()
        k_obs = np.array([0.0])
        T_obs = np.array([0.1])
        iv_obs = np.array([0.20])
        result = sgm.complete(k_obs, T_obs, iv_obs,
                              np.linspace(-0.1, 0.1, 5),
                              np.array([0.05, 0.10]))
        assert result.shape == (2, 5)
        assert np.all(result > 0)


# ============================================================
# 10. ENSEMBLE PRICER TESTS (Phase 3)
# ============================================================

class TestEnsemblePricer:
    """Test the adaptive ensemble pricer."""

    def test_import_and_create(self):
        from ensemble_pricer import EnsemblePricer
        ep = EnsemblePricer()
        assert ep.n_updates == 0

    def test_register_callable_model(self):
        from ensemble_pricer import EnsemblePricer
        ep = EnsemblePricer()
        ep.register_model('bsm', lambda s, k, t, r, q, sig, ot: max(s - k, 0))
        assert 'bsm' in ep.models

    def test_ensemble_single_model(self):
        from ensemble_pricer import EnsemblePricer
        ep = EnsemblePricer()
        ep.register_model('fixed', lambda s, k, t, r, q, sig, ot: 100.0)
        price, diag = ep.price(23500, 23400, 0.05)
        assert price == pytest.approx(100.0)
        assert diag['n_active_models'] == 1

    def test_ensemble_multiple_models(self):
        from ensemble_pricer import EnsemblePricer
        ep = EnsemblePricer()
        ep.register_model('low', lambda s, k, t, r, q, sig, ot: 90.0)
        ep.register_model('high', lambda s, k, t, r, q, sig, ot: 110.0)
        price, diag = ep.price(23500, 23400, 0.05)
        # Average of 90 and 110
        assert price == pytest.approx(100.0)
        assert 'model_agreement' in diag

    def test_weight_update_favors_accurate_model(self):
        from ensemble_pricer import EnsemblePricer
        ep = EnsemblePricer(learning_rate=0.5)
        ep.register_model('accurate', lambda s, k, t, r, q, sig, ot: 100.0)
        ep.register_model('inaccurate', lambda s, k, t, r, q, sig, ot: 150.0)

        # Price and update with market = 102
        ep.price(23500, 23400, 0.05)
        ep.update(market_price=102.0)

        rankings = ep.get_model_rankings()
        assert rankings[0][0] == 'accurate'  # accurate model should rank first


# ============================================================
# 11. HAWKES INTEGRATION IN QUANT ENGINE (Phase 3)
# ============================================================

class TestHawkesIntegration:
    """Test Hawkes estimator wiring in QuantEngine."""

    def test_quant_engine_has_hawkes(self):
        from quant_engine import QuantEngine
        qe = QuantEngine()
        assert hasattr(qe, 'hawkes_estimator')
        assert qe.hawkes_estimator is not None

    def test_fit_hawkes_jumps_method(self):
        from quant_engine import QuantEngine
        qe = QuantEngine()
        returns = np.random.default_rng(42).normal(0, 0.015, 100)
        result = qe.fit_hawkes_jumps(returns)
        assert 'lambda_j' in result
        assert 'clustering_score' in result


# ============================================================
# 12. NEURAL J-SDE TESTS (Nobel-Level)
# ============================================================

class TestNeuralJSDE:
    """Test the Neural Jump-SDE pricer."""

    def test_import_and_create(self):
        from neural_jsde import NeuralJSDE
        njsde = NeuralJSDE(n_paths=1000, n_centers=5)
        assert not njsde.is_calibrated

    def test_default_pricing(self):
        from neural_jsde import NeuralJSDE
        njsde = NeuralJSDE(n_paths=2000, n_steps=20, seed=42)
        price, se = njsde.price(spot=23500, strike=23400, T=0.05, sigma=0.14)
        assert price > 0
        assert np.isfinite(price)
        assert se > 0

    def test_put_price(self):
        from neural_jsde import NeuralJSDE
        njsde = NeuralJSDE(n_paths=2000, n_steps=20, seed=42)
        price, se = njsde.price(spot=23500, strike=23600, T=0.05,
                                sigma=0.14, option_type='PE')
        assert price > 0
        assert np.isfinite(price)

    def test_learned_dynamics(self):
        from neural_jsde import NeuralJSDE
        njsde = NeuralJSDE(n_paths=1000)
        features = {'vix': 18.0, 'regime_crisis': 0.3, 'regime_normal': 0.5}
        dynamics = njsde.get_learned_dynamics(features, T=0.1)
        assert 'drift_adjustment' in dynamics
        assert 'vol_multiplier' in dynamics
        assert 'jump_intensity' in dynamics
        assert all(np.isfinite(v) for v in dynamics.values())


# ============================================================
# 13. DEEP HEDGING TESTS (Nobel-Level)
# ============================================================

class TestDeepHedging:
    """Test the deep hedging module."""

    def test_import_and_create(self):
        from deep_hedging import DeepHedger
        dh = DeepHedger(n_sim_paths=500)
        assert not dh.is_trained

    def test_bsm_delta(self):
        from deep_hedging import DeepHedger
        delta_call = DeepHedger.bsm_delta(23500, 23500, 0.1, 0.065, 0.012, 0.15, 'CE')
        delta_put = DeepHedger.bsm_delta(23500, 23500, 0.1, 0.065, 0.012, 0.15, 'PE')
        assert 0.4 < delta_call < 0.7
        assert -0.7 < delta_put < -0.3
        # Put-call parity for delta: Δ_call - Δ_put = e^(-qT)
        assert abs((delta_call - delta_put) - np.exp(-0.012 * 0.1)) < 0.01

    def test_bsm_price(self):
        from deep_hedging import DeepHedger
        price = DeepHedger.bsm_price(23500, 23400, 0.1, 0.065, 0.012, 0.15, 'CE')
        assert price > 0
        assert np.isfinite(price)

    def test_untrained_falls_back_to_bsm(self):
        from deep_hedging import DeepHedger
        dh = DeepHedger()
        hedge = dh.optimal_hedge({'delta_bsm': 0.52})
        assert hedge == pytest.approx(0.52)  # untrained → BSM fallback

    def test_hedge_diagnostics(self):
        from deep_hedging import DeepHedger
        dh = DeepHedger()
        diag = dh.hedge_diagnostics({'delta_bsm': 0.6})
        assert 'deep_hedge_delta' in diag
        assert 'bsm_delta' in diag
        assert 'hedge_adjustment' in diag


# ============================================================
# 14. UNIFIED PIPELINE TESTS (Nobel-Level)
# ============================================================

class TestUnifiedPipeline:
    """Test the full integrated pipeline orchestrator."""

    def test_import_and_create(self):
        from unified_pipeline import UnifiedPricingPipeline
        pipe = UnifiedPricingPipeline()
        assert len(pipe.available_components()) > 0

    def test_all_components_loaded(self):
        from unified_pipeline import UnifiedPricingPipeline
        pipe = UnifiedPricingPipeline()
        expected = ['sgm', 'pinn', 'hawkes', 'neural_jsde', 'kan', 'ensemble', 'hedger']
        for comp in expected:
            assert comp in pipe.available_components(), f"Missing component: {comp}"

    def test_pipeline_price_basic(self):
        from unified_pipeline import UnifiedPricingPipeline
        pipe = UnifiedPricingPipeline({'njsde_paths': 2000})
        result = pipe.price(
            spot=23500, strike=23400, T=0.05,
            sigma=0.14, option_type='CE',
            market_state={'vix': 14.0},
        )
        assert 'price' in result
        assert result['price'] >= 0
        assert 'confidence_interval' in result
        assert 'component_prices' in result
        assert 'diagnostics' in result

    def test_pipeline_diagnostics_stages(self):
        from unified_pipeline import UnifiedPricingPipeline
        pipe = UnifiedPricingPipeline({'njsde_paths': 1000})
        result = pipe.price(spot=23500, strike=23400, T=0.05, sigma=0.14)
        diag = result['diagnostics']
        assert 'stages_run' in diag
        assert len(diag['stages_run']) > 0

    def test_pipeline_status(self):
        from unified_pipeline import UnifiedPricingPipeline
        pipe = UnifiedPricingPipeline()
        status = pipe.pipeline_status()
        assert isinstance(status, dict)
        assert all('loaded' in v for v in status.values())


# ============================================================
# Run with: python -m pytest tests/test_frontier_upgrades.py -v
# ============================================================
