"""
test_arbfree_surface.py — Unit Tests for Arbitrage-Free Surface
==============================================================

Verifies that the surface state:
1.  Ingests slices correctly.
2.  Interpolates IVs.
3.  Detects arbitrage (calendar/butterfly) in bad data.
"""

import numpy as np
import pytest
from arbfree_surface import ArbFreeSurfaceState

class TestArbFreeSurface:
    
    def test_basic_interpolation(self):
        """Test adding a slice and retrieving IV (Single Slice case)."""
        surf = ArbFreeSurfaceState()
        T = 0.1
        strikes = np.array([22000, 23000, 24000])
        ivs = np.array([0.22, 0.20, 0.18]) # Skew
        spot = 23000
        
        surf.add_slice(T, strikes, ivs, spot)
        surf.fit()
        
        # Query at known point (ATM)
        k_atm = 0.0
        iv = surf.get_iv(k_atm, T)
        
        # Single slice extrapolation logic: W = W_min * T / T_min
        # iv = sqrt(W/T) = sqrt(W_min/T_min) * sqrt(T/T) ??? 
        # No, W = (IV_min^2 * T_min) * (T / T_min) = IV_min^2 * T
        # IV = sqrt(IV_min^2 * T / T) = IV_min.
        # So constant volatility extrapolation.
        assert iv == pytest.approx(0.20, abs=0.01)
        
    def test_calendar_interpolation(self):
        """Test interpolation between two slices."""
        surf = ArbFreeSurfaceState()
        spot = 23000
        strikes = np.array([23000])
        
        # T1=0.1, IV=0.20
        surf.add_slice(0.1, strikes, np.array([0.20]), spot)
        # T2=0.2, IV=0.20
        surf.add_slice(0.2, strikes, np.array([0.20]), spot)
        
        surf.fit()
        
        # T=0.15, should be 0.20
        iv = surf.get_iv(0.0, 0.15)
        assert iv == pytest.approx(0.20, abs=0.001)

    def test_calendar_arbitrage_detection(self):
        """Test detection of decreasing total variance."""
        surf = ArbFreeSurfaceState()
        strikes = np.array([23000])
        spot = 23000
        
        # T1: 0.1 yr, IV=0.20 -> w = 0.004
        surf.add_slice(0.1, strikes, np.array([0.20]), spot)
        
        # T2: 0.2 yr, IV=0.10 -> w = 0.002
        # T2 total variance < T1 total variance
        surf.add_slice(0.2, strikes, np.array([0.10]), spot)
        
        surf.fit()
        
        viol = surf.count_arbitrage_violations()
        assert viol['calendar'] > 0
        
    def test_butterfly_repair_sanity(self):
        """Test that fit runs on concave data without error."""
        surf = ArbFreeSurfaceState()
        T = 0.1
        strikes = np.array([22000, 23000, 24000])
        # Concave smile
        ivs = np.array([0.18, 0.25, 0.18]) 
        spot = 23000
        
        surf.add_slice(T, strikes, ivs, spot)
        surf.fit()
        
        # Should return valid positive IVs
        iv = surf.get_iv(0.0, T)
        assert iv > 0.0
