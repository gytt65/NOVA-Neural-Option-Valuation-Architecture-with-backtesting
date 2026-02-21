"""
vrr_state.py — Variance Risk Premium (VRP) State Filter
=======================================================

Implements the Variance Risk Ratio (VRR) / Variance Risk Premium (VRP) logic.
Used to adjust model parameters (Jump Intensity, Vol-of-Vol, Mean Reversion)
based on the discount/premium the market is assigning to volatility.

Key Concepts:
-------------
- VRP = Implied Variance - Realized Variance
- VRR = Implied Variance / Realized Variance (Variance Risk Ratio)
- A_t: Latent "Risk Aversion" state, inferred from VRP.

When A_t is high (Panic/Fear):
- Market pays huge premium for puts.
- Models should amplify Jump Intensity (lambda) and Vol-Vol (eta).
- Confidence in mean-reversion should decrease (or speed increases?).

When A_t is low (Complacency):
- VRP is low or negative.
- Models should dampen jumps.
"""

import numpy as np
from typing import Dict, Optional, Tuple

class VRRStateFilter:
    """
    Stateless filter (snapshot-based) for VRP adjustments.
    (A true recursive filter would require persistent state across days).
    """

    def __init__(self):
        # Parameters for the mapping function
        self.scaling_factor = 5.0  # Sensitivity to VRP
        
    def get_state(self, iv: float, returns_history: np.ndarray, dt: float = 1.0/252.0) -> float:
        """
        Compute the Risk Aversion state A_t from current IV and past returns.
        
        Parameters
        ----------
        iv : float
            Annualised Implied Volatility (decimal).
        returns_history : np.ndarray
            Array of recent log-returns (e.g., 30 days).
            
        Returns
        -------
        A_t : float
            State variable, roughly centered at 0.0.
            > 0 : High Risk Aversion (High VRP)
            < 0 : Low Risk Aversion (Low VRP)
        """
        if len(returns_history) < 5:
            return 0.0
            
        # 1. Compute Realized Volatility (RV)
        # RV = std(returns) * sqrt(1/dt)
        # Using simple standard deviation (centered) or RMS (uncentered)?
        # Usually centered std dev for volatility
        rv = np.std(returns_history, ddof=1) * np.sqrt(1.0 / dt)
        
        # Avoid division by zero
        rv = max(rv, 0.05)
        
        # 2. Compute Variance Risk Ratio (VRR)
        # VRR = IV / RV
        # Log-VRR is better behaved: log(IV/RV)
        log_vrr = np.log(max(iv, 0.01) / rv)
        
        # 3. Map to State A_t
        # Nominal VRR is often > 1 (spread). log(VRR) > 0.
        # We center it around a "normal" premium (e.g. IV ~ 1.2 * RV => log(1.2) ~ 0.18)
        # Let's say normal spread is 20%.
        equilibrium_log_vrr = 0.18 
        
        A_t = (log_vrr - equilibrium_log_vrr) * self.scaling_factor
        
        # Clamp state to prevent explosion
        return np.clip(A_t, -3.0, 3.0)

    def get_adjustments(self, A_t: float) -> Dict[str, float]:
        """
        Get parameter multipliers based on state A_t.
        
        Returns
        -------
        dict : {
            'lambda_mult': float,  # Jump intensity multiplier
            'eta_mult': float,     # Vol-of-vol multiplier
            'conf_mult': float     # Confidence score multiplier
        }
        """
        # Sigmoid-like mappings or linear with caps
        
        # 1. Jump Intensity (lambda)
        # A_t > 0 (Fear) => More jumps expected => Higher lambda
        lambda_mult = 1.0 + 0.5 * np.tanh(A_t) # Range [0.5, 1.5]
        
        # 2. Vol-of-Vol (eta)
        # A_t > 0 => Higher uncertainty => Higher vol-vol
        eta_mult = 1.0 + 0.3 * np.tanh(A_t) # Range [0.7, 1.3]
        
        # 3. Confidence Score
        # High VRP might mean "Model Disagrees with Market".
        # If Market is crazy (A_t high), maybe we trust our Fair Value MORE (mean reversion opportunity)?
        # OR we trust it LESS because regime is unstable?
        # Usually, extreme VRP = Opportunity = Higher Confidence in Mean Reversion signal.
        # So A_t high => Signal Strength High.
        conf_mult = 1.0 + 0.2 * np.abs(np.tanh(A_t)) # [1.0, 1.2]
        
        return {
            'lambda_mult': lambda_mult,
            'eta_mult': eta_mult,
            'conf_mult': conf_mult
        }
