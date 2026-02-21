#!/usr/bin/env python3
"""
test_golden_nirv_outputs.py — Golden Master Regression Tests
============================================================

Verifies that NIRV pricing outputs match frozen golden-master snapshots
exactly (within floating-point tolerance) when all v5 features are OFF.

Run:
    python3 -m pytest tests/golden/test_golden_nirv_outputs.py -v
"""

import sys
import os
import json
import glob
import importlib.util
import numpy as np
import pytest

# ── Setup paths ──────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

# Import nirv_model
# Import nirv_model
try:
    import nirv_model
    nirv = nirv_model
except ImportError:
    spec = importlib.util.spec_from_file_location(
        "nirv_model", os.path.join(ROOT, "nirv_model.py")
    )
    nirv = importlib.util.module_from_spec(spec)
    sys.modules['nirv_model'] = nirv
    spec.loader.exec_module(nirv)

NIRVModel = nirv.NIRVModel
HestonJumpDiffusionPricer = nirv.HestonJumpDiffusionPricer

NIRVModel = nirv.NIRVModel
HestonJumpDiffusionPricer = nirv.HestonJumpDiffusionPricer

# Ensure v5 features are OFF
from omega_features import OmegaFeatures, set_features
set_features()  # all OFF


# ── Tolerances ───────────────────────────────────────────────────────
# Floating-point comparison tolerances
PRICE_RTOL = 0.005       # 0.5% relative tolerance for prices
PRICE_ATOL = 0.50        # ₹0.50 absolute tolerance for prices
PCT_ATOL   = 1.0         # 1 percentage point for probabilities/confidence
GREEK_RTOL = 0.05        # 5% relative tolerance for Greeks
GREEK_ATOL = 0.01        # Absolute tolerance for small Greeks


# ── Load snapshots ───────────────────────────────────────────────────
SNAPSHOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "snapshots")


def load_snapshots():
    """Load all golden master snapshots."""
    pattern = os.path.join(SNAPSHOT_DIR, "*.json")
    files = sorted(glob.glob(pattern))
    snapshots = []
    for f in files:
        with open(f, 'r') as fh:
            data = json.load(fh)
            data['_file'] = os.path.basename(f)
            snapshots.append(data)
    return snapshots


SNAPSHOTS = load_snapshots()


def _run_pricing(snapshot):
    """Run NIRV pricing for a snapshot with deterministic RNG."""
    inp = snapshot["inputs"]
    seed = snapshot["seed"]
    n_paths = snapshot["n_paths"]
    n_bootstrap = snapshot["n_bootstrap"]

    returns_30d = np.array(inp["returns_30d"])

    # Reset global RNG (used by Poisson calls in the pricer)
    np.random.seed(seed)

    # Create model with matching parameters
    model = NIRVModel(n_paths=n_paths, n_bootstrap=n_bootstrap)
    model.pricer = HestonJumpDiffusionPricer(
        n_paths=n_paths, use_sobol=False, seed=seed
    )

    result = model.price_option(
        spot=inp["spot"],
        strike=inp["strike"],
        T=inp["T"],
        r=inp["r"],
        q=inp["q"],
        option_type=inp["option_type"],
        market_price=inp["market_price"],
        india_vix=inp["india_vix"],
        fii_net_flow=inp["fii_net_flow"],
        dii_net_flow=inp["dii_net_flow"],
        days_to_rbi=inp["days_to_rbi"],
        pcr_oi=inp["pcr_oi"],
        returns_30d=returns_30d,
    )

    return result


# ── Parametrized tests ───────────────────────────────────────────────

@pytest.fixture(params=SNAPSHOTS, ids=[s['id'] for s in SNAPSHOTS])
def snapshot(request):
    """Parametrize over all golden snapshots."""
    return request.param


class TestGoldenMasterBaseline:
    """
    Verify that NIRV outputs match golden master snapshots exactly
    when all v5 features are OFF.
    """

    def test_fair_value_matches(self, snapshot):
        """Fair value must match within price tolerance."""
        result = _run_pricing(snapshot)
        expected = snapshot["expected_outputs"]["fair_value"]
        assert result.fair_value == pytest.approx(
            expected, rel=PRICE_RTOL, abs=PRICE_ATOL
        ), (
            f"[{snapshot['id']}] fair_value: got {result.fair_value}, "
            f"expected {expected}"
        )

    def test_signal_matches(self, snapshot):
        """Trading signal must match exactly."""
        result = _run_pricing(snapshot)
        expected = snapshot["expected_outputs"]["signal"]
        assert result.signal == expected, (
            f"[{snapshot['id']}] signal: got {result.signal}, "
            f"expected {expected}"
        )

    def test_regime_matches(self, snapshot):
        """Detected regime must match exactly."""
        result = _run_pricing(snapshot)
        expected = snapshot["expected_outputs"]["regime"]
        assert result.regime == expected, (
            f"[{snapshot['id']}] regime: got {result.regime}, "
            f"expected {expected}"
        )

    def test_confidence_matches(self, snapshot):
        """Confidence level must match within percentage tolerance."""
        result = _run_pricing(snapshot)
        expected = snapshot["expected_outputs"]["confidence_level"]
        assert result.confidence_level == pytest.approx(
            expected, abs=PCT_ATOL
        ), (
            f"[{snapshot['id']}] confidence: got {result.confidence_level}, "
            f"expected {expected}"
        )

    def test_profit_probability_matches(self, snapshot):
        """Profit probability must match within tolerance."""
        result = _run_pricing(snapshot)
        expected_rn = snapshot["expected_outputs"]["profit_probability"]
        expected_phys = snapshot["expected_outputs"]["physical_profit_prob"]
        assert result.profit_probability == pytest.approx(
            expected_rn, abs=PCT_ATOL
        ), (
            f"[{snapshot['id']}] profit_prob_rn: got {result.profit_probability}, "
            f"expected {expected_rn}"
        )
        assert result.physical_profit_prob == pytest.approx(
            expected_phys, abs=PCT_ATOL
        ), (
            f"[{snapshot['id']}] physical_profit_prob: got {result.physical_profit_prob}, "
            f"expected {expected_phys}"
        )

    def test_mispricing_pct_matches(self, snapshot):
        """Mispricing percentage must match."""
        result = _run_pricing(snapshot)
        expected = snapshot["expected_outputs"]["mispricing_pct"]
        assert result.mispricing_pct == pytest.approx(
            expected, abs=1.0
        ), (
            f"[{snapshot['id']}] mispricing_pct: got {result.mispricing_pct}, "
            f"expected {expected}"
        )

    def test_greeks_match(self, snapshot):
        """Greeks must match within tolerance."""
        result = _run_pricing(snapshot)
        expected_greeks = snapshot["expected_outputs"]["greeks"]

        for greek_name, expected_val in expected_greeks.items():
            actual_val = result.greeks.get(greek_name, 0.0)
            if abs(expected_val) < GREEK_ATOL:
                # For near-zero Greeks, use absolute tolerance
                assert actual_val == pytest.approx(
                    expected_val, abs=GREEK_ATOL
                ), (
                    f"[{snapshot['id']}] greek {greek_name}: "
                    f"got {actual_val}, expected {expected_val}"
                )
            else:
                assert actual_val == pytest.approx(
                    expected_val, rel=GREEK_RTOL, abs=GREEK_ATOL
                ), (
                    f"[{snapshot['id']}] greek {greek_name}: "
                    f"got {actual_val}, expected {expected_val}"
                )


class TestFeatureFlagsOff:
    """Verify feature flag system works and all flags are OFF by default."""

    def test_default_all_off(self):
        features = OmegaFeatures()
        assert features.india_vix_synth is False
        assert features.arb_free_surface is False
        assert features.vrr_state is False
        assert features.surface_shock is False
        assert features.USE_NSE_CONTRACT_SPECS is False
        assert features.USE_NSE_VIX_ENGINE is False
        assert features.USE_TAIL_CORRECTED_VARIANCE is False
        assert features.USE_ESSVI_SURFACE is False
        assert features.USE_SVI_FIXED_POINT_WARMSTART is False
        assert features.USE_MODEL_FREE_VRP is False
        assert features.USE_TIERED_PRICER is False
        assert features.USE_CONFORMAL_INTERVALS is False
        assert features.USE_RESEARCH_HIGH_CONVICTION is False
        assert features.USE_OOS_RELIABILITY_GATE is False

    def test_immutable(self):
        features = OmegaFeatures()
        with pytest.raises(AttributeError):
            features.india_vix_synth = True

    def test_env_override(self):
        os.environ['OMEGA_FEATURES_JSON'] = '{"india_vix_synth": true}'
        try:
            features = OmegaFeatures()
            assert features.india_vix_synth is True
            assert features.arb_free_surface is False
        finally:
            del os.environ['OMEGA_FEATURES_JSON']

    def test_programmatic_override(self):
        features = OmegaFeatures(arb_free_surface=True)
        assert features.arb_free_surface is True
        assert features.india_vix_synth is False

    def test_all_on(self):
        features = OmegaFeatures.all_on()
        assert features.india_vix_synth is True
        assert features.arb_free_surface is True
        assert features.vrr_state is True
        assert features.surface_shock is True
        assert features.USE_NSE_CONTRACT_SPECS is True
        assert features.USE_NSE_VIX_ENGINE is True
        assert features.USE_TAIL_CORRECTED_VARIANCE is True
        assert features.USE_ESSVI_SURFACE is True
        assert features.USE_SVI_FIXED_POINT_WARMSTART is True
        assert features.USE_MODEL_FREE_VRP is True
        assert features.USE_TIERED_PRICER is True
        assert features.USE_CONFORMAL_INTERVALS is True

    def test_to_dict(self):
        features = OmegaFeatures(india_vix_synth=True)
        d = features.to_dict()
        assert d['india_vix_synth'] is True
        assert d['arb_free_surface'] is False
        assert d['vrr_state'] is False
        assert d['surface_shock'] is False
        assert d['USE_NSE_VIX_ENGINE'] is False
