"""
test_india_vix_synth.py — Unit Tests for Synthetic VIX
======================================================

Verifies model-free implied volatility calculation using:
1. Synthetic option chains generated from Black-Scholes (flat vol).
2. Known variance inputs.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm
from india_vix_synth import compute_synthetic_vix


def black_scholes_price(S, K, T, r, sigma, option_type='CE'):
    """Generate theoretical price for testing."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'CE':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def generate_chain(S, T, r, sigma, strikes):
    """Create a clean option chain dataframe."""
    data = []
    for K in strikes:
        call_price = black_scholes_price(S, K, T, r, sigma, 'CE')
        put_price = black_scholes_price(S, K, T, r, sigma, 'PE')
        
        # Add both call and put for every strike
        data.append({'strike': K, 'option_type': 'CE', 'price': call_price})
        data.append({'strike': K, 'option_type': 'PE', 'price': put_price})
        
    return pd.DataFrame(data)


class TestSyntheticVIX:
    
    def test_flat_vol_recovery(self):
        """
        If we input a chain priced with flat volatility (e.g. 20%),
        the VIX formula (variance swap) should recover ~20%.
        It won't be exact due to discrete strike spacing and discretization error,
        but should be close (within 0.5 vol points).
        """
        S = 23500
        r = 0.065
        sigma = 0.20  # 20%
        T = 30.0 / 365.0
        
        # Dense grid of strikes to minimize discretization error
        strikes = np.arange(20000, 27000, 50)
        
        df = generate_chain(S, T, r, sigma, strikes)
        
        chain_structure = {'T': T, 'options': df}
        
        vix, quality, debug = compute_synthetic_vix([chain_structure], r, target_days=30)
        
        # Expect ~20.0
        assert vix == pytest.approx(20.0, abs=0.5)
        
    def test_interpolation(self):
        """Test interpolation between two expiries."""
        S = 23500
        r = 0.0
        sigma = 0.20
        
        # T1 = 20 days, T2 = 40 days
        # Both priced at 20% vol
        T1 = 20.0 / 365.0
        T2 = 40.0 / 365.0
        
        strikes = np.arange(20000, 27000, 50)
        
        df1 = generate_chain(S, T1, r, sigma, strikes)
        df2 = generate_chain(S, T2, r, sigma, strikes)
        
        chains = [
            {'T': T1, 'options': df1},
            {'T': T2, 'options': df2}
        ]
        
        vix, quality, debug = compute_synthetic_vix(chains, r, target_days=30)
        
        # Should still be ~20.0
        assert vix == pytest.approx(20.0, abs=0.5)
        
        # Check details
        assert debug['T1'] == T1
        assert debug['T2'] == T2
        
    def test_insufficient_strikes(self):
        """Test with too few strikes (should return NaN or handle gracefully)."""
        chain = {'T': 0.1, 'options': pd.DataFrame([
            {'strike': 23500, 'option_type': 'CE', 'price': 100},
            {'strike': 23500, 'option_type': 'PE', 'price': 100}
        ])}
        
        vix, quality, _ = compute_synthetic_vix([chain])
        assert np.isnan(vix) or quality == 0.0

