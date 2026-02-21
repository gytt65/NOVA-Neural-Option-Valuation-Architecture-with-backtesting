"""
test_arbfree_integration.py — Integration Tests for Arbitrage-Free Surface
==========================================================================

Verifies that:
1.  Enabling `arb_free_surface` flag activates the logic in `VolatilitySurface`.
2.  `ArbFreeSurfaceState` is built and cached.
3.  Pricing continues to work with reasonable values.
"""

import pytest
import numpy as np
from omega_features import set_features, OmegaFeatures
try:
    from nirv_model import NIRVModel
except ImportError:
    # Handle import if recursive or path issues, though in tests root dir is usually in path
    import sys
    import os
    sys.path.append(os.getcwd())
    # Try importing with the exact filename if module name is tricky
    try:
        import nirv_model
        NIRVModel = nirv_model.NIRVModel
    except ImportError:
        import importlib.util
        # Assuming ROOT is defined or needs to be defined, for example:
        ROOT = os.path.dirname(os.path.abspath(__file__)) # Or os.getcwd() depending on project structure
        spec = importlib.util.spec_from_file_location(
            "nirv_model", os.path.join(ROOT, "nirv_model.py")
        )
        nirv = importlib.util.module_from_spec(spec)
        sys.modules['nirv_model'] = nirv
        spec.loader.exec_module(nirv)
        NIRVModel = nirv.NIRVModel

class TestArbFreeIntegration:
    
    def setup_method(self):
        # Reset features to default (OFF)
        set_features()
        
    def teardown_method(self):
        set_features()

    def test_surface_activation(self):
        """Test that flag enables surface building."""
        set_features(arb_free_surface=True)
        
        model = NIRVModel()
        
        # Mock features
        india_features = {
            'india_vix': 14.0,
            'flow_ratio': 1.0,
            'gamma_amp': 0.0,
            'sentiment_score': 0.0
        }
        
        # Spot, Strike, T
        spot = 23000
        strike = 23000
        T = 30.0 / 365.0
        regime = 'Sideways'
        
        # 1. Check cache is empty initially
        assert len(model.vol_surface.arb_surfaces) == 0
        
        # 2. Call get_implied_vol
        iv = model.vol_surface.get_implied_vol(spot, strike, T, regime, india_features)
        
        # 3. Check cache is populated
        assert regime in model.vol_surface.arb_surfaces
        surf = model.vol_surface.arb_surfaces[regime]
        assert surf._is_fitted
        
        # 4. Check IV is reasonable (approx 14% +- regime adj)
        assert 0.10 < iv < 0.25
        
    def test_price_option_with_surface(self):
        """Test full pricing pipeline with flag ON."""
        set_features(arb_free_surface=True)
        model = NIRVModel()
        
        # Pricing Inputs
        snapshot_inputs = {
            "spot": 23500.0,
            "strike": 23500.0,
            "T": 7.0/365.0,
            "r": 0.065,
            "q": 0.0,
            "option_type": "CE",
            "market_price": 200.0,
            "india_vix": 14.0,
            "fii_net_flow": 1200.0,
            "dii_net_flow": 400.0,
            "days_to_rbi": 25,
            "pcr_oi": 0.9,
            "returns_30d_seed": 100,
            "returns_30d": np.random.normal(0, 0.01, 30)
        }
        
        # Run pricing
        result = model.price_option(**snapshot_inputs)
        
        # Should contain valid output
        assert result.fair_value > 0
        assert 100 < result.fair_value < 300
        assert result.confidence_level > 0

    def test_arb_surface_fallback_behavior(self):
        """Test fallback if arb surface fails (simulated)."""
        set_features(arb_free_surface=True)
        model = NIRVModel()
        
        # Mock _ensure_arb_surface to fail
        original_ensure = model.vol_surface._ensure_arb_surface
        
        def mock_ensure(*args, **kwargs):
            raise ValueError("Simulated Failure")
            
        model.vol_surface._ensure_arb_surface = mock_ensure
        
        india_features = {'india_vix': 14.0}
        iv = model.vol_surface.get_implied_vol(23000, 23000, 0.1, 'Sideways', india_features)
        
        # Should succeed (using fallback parametric logic)
        assert iv > 0
        
        # Restore (though teardown handles feature reset, object is local)
