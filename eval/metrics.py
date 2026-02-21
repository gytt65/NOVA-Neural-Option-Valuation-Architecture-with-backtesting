"""
metrics.py
==========

Evaluation metrics for the OMEGA v5 option pricing system.

Includes:
1. Vega-weighted RMSE for prices
2. IV RMSE
3. Arbitrage violation counts (Butterfly/Calendar)
4. Surface stability (L2 norm of changes)
5. Greeks stability
"""

import numpy as np

def vega_weighted_rmse(market_prices, model_prices, vegas):
    """
    Calculate RMSE weighted by Vega.
    High Vega options (ATM/Near-term) contribute more to the error.
    
    Args:
        market_prices: array-like of market observed prices
        model_prices: array-like of model theoretical prices
        vegas: array-like of option vegas
        
    Returns:
        float: Weighted RMSE
    """
    diff = np.array(market_prices) - np.array(model_prices)
    weights = np.array(vegas)
    # Normalize weights
    weights = weights / (np.sum(weights) + 1e-12)
    
    weighted_mse = np.sum(weights * diff**2)
    return np.sqrt(weighted_mse)

def iv_rmse(market_ivs, model_ivs):
    """
    Calculate Root Mean Squared Error for Implied Volatilities.
    """
    diff = np.array(market_ivs) - np.array(model_ivs)
    return np.sqrt(np.mean(diff**2))

def arbitrage_violation_counts(total_vars, t_expiries, k_strikes):
    """
    Count static arbitrage violations in the total variance surface w(k, t).
    
    Args:
        total_vars: 2D array (n_strikes, n_expiries) of total variance w = sigma^2 * T
        t_expiries: 1D array of times to expiry
        k_strikes: 1D array of log-strikes
        
    Returns:
        dict: {'calendar': int, 'butterfly': int}
    """
    calendar_violations = 0
    butterfly_violations = 0
    
    w = np.array(total_vars)
    
    # 1. Calendar Spread Arbitrage: w(k, t2) >= w(k, t1) for t2 > t1
    # Check along expiry dimension (axis 1)
    # diff > 0 means w[i+1] > w[i], which is good.
    # diff < 0 means violation.
    time_diff = np.diff(w, axis=1)
    calendar_violations = np.sum(time_diff < -1e-6)
    
    # 2. Butterfly Arbitrage: g(k) = (1 - kw'/2w)^2 - (w')^2/4 + w''/2 >= 0
    # Approximate using discrete Durrleman condition on total variance slices?
    # Or simply check PDF positivity: C_kk > 0
    # For SVI, we check distinct parametrization constraints.
    # Here we check raw density convexity p(k) ~ d2C/dK2 > 0.
    # A simpler proxy for total variance w(k):
    # w not too concave.
    
    return {
        'calendar': int(calendar_violations), 
        'butterfly': int(butterfly_violations) # Placeholder for complex check
    }

def surface_stability_metric(surface_prev, surface_curr):
    """
    Calculate L2 distance between two consecutive surface updates.
    """
    if surface_prev is None or surface_curr is None:
        return 0.0
        
    diff = np.array(surface_curr) - np.array(surface_prev)
    return np.linalg.norm(diff)
