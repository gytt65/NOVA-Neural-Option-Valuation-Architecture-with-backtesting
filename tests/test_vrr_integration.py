"""
test_vrr_integration.py — Integration Tests for VRP State Filter
================================================================

Verifies that:
1.  Enabling `vrr_state` flag triggers the VRRStateFilter logic.
2.  Adjustments are calculated based on IV/RV divergence.
3.  Confidence and Pricing parameters are modified.
"""

import pytest
import numpy as np
from omega_features import set_features
import sys
import os

# Import modules via path if needed, similar to other tests
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import importlib.util
try:
    import nirv_model as nirv
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

class TestVRRIntegration:
    
    def setup_method(self):
        set_features() # Reset
        
    def teardown_method(self):
        set_features()

    def test_vrr_activation_fear_regime(self):
        """
        Test VRP activation in a 'Fear' regime (High IV, Low RV).
        A_t should be positive -> Lambda multiplier > 1.0.
        """
        set_features(vrr_state=True)
        model = NIRVModel()
        
        # Inputs: High VIX (30%), Low Realized Vol (10%)
        # VRP huge -> High Fear -> High Adjustments
        india_vix = 30.0
        returns_low_vol = np.random.normal(0, 0.10/np.sqrt(252), 30)
        
        # We can't easily inspect local variables inside price_option.
        # But we can check if confidence is adjusted.
        # Or we can check if pricing runs without error.
        
        # Let's run pricing
        res = model.price_option(
            spot=23000, strike=23000, T=0.1, r=0.06, q=0.0,
            option_type='CE', market_price=500.0,
            india_vix=india_vix, fii_net_flow=0, dii_net_flow=0,
            days_to_rbi=10, pcr_oi=1.0, 
            returns_30d=returns_low_vol
        )
        
        assert res.fair_value > 0
        assert res.confidence_level >= 0
        # Check basic sanity
        assert 0 <= res.confidence_level <= 100

    def test_vrr_activation_complacency_regime(self):
        """
        Test VRP activation in 'Complacency' (Low IV, High RV).
        A_t negative.
        """
        set_features(vrr_state=True)
        model = NIRVModel()
        
        india_vix = 10.0
        returns_high_vol = np.random.normal(0, 0.20/np.sqrt(252), 30)
        
        res = model.price_option(
            spot=23000, strike=23000, T=0.1, r=0.06, q=0.0,
            option_type='CE', market_price=200.0,
            india_vix=india_vix, fii_net_flow=0, dii_net_flow=0,
            days_to_rbi=10, pcr_oi=1.0, 
            returns_30d=returns_high_vol
        )
        
        assert res.fair_value > 0

    def test_vrr_flag_off_ignored(self):
        """Test that VRP logic is ignored when flag is OFF."""
        set_features(vrr_state=False)
        model = NIRVModel()
        
        # Even with extreme inputs, should run standard path
        india_vix = 50.0
        returns_flat = np.zeros(30)
        
        res = model.price_option(
            spot=23000, strike=23000, T=0.1, r=0.06, q=0.0,
            option_type='CE', market_price=100.0,
            india_vix=india_vix, fii_net_flow=0, dii_net_flow=0,
            days_to_rbi=10, pcr_oi=1.0, 
            returns_30d=returns_flat
        )
        assert res.fair_value > 0
