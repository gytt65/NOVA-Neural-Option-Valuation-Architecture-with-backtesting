"""
svi_fixed_point.py — Lightweight fixed-point warm-start for SVI calibration.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def fixed_point_svi_warmstart(
    spot: float,
    strikes,
    market_ivs,
    T: float,
    prev_params: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Build a robust initial guess for raw-SVI-like parameters.

    Returns keys compatible with common SVI optimizers:
      {a, b, rho, m, sig}
    """
    s = max(float(spot), 1e-8)
    tt = max(float(T), 1e-8)
    K = np.asarray(strikes, dtype=float)
    iv = np.asarray(market_ivs, dtype=float)
    mask = np.isfinite(K) & np.isfinite(iv) & (K > 0) & (iv > 0)
    if np.sum(mask) < 3:
        if prev_params:
            return {
                "a": float(prev_params.get("a", 0.01)),
                "b": float(prev_params.get("b", 1.0)),
                "rho": float(prev_params.get("rho", -0.3)),
                "m": float(prev_params.get("m", 0.0)),
                "sig": float(prev_params.get("sig", 0.15)),
            }
        return {"a": 0.01, "b": 1.0, "rho": -0.3, "m": 0.0, "sig": 0.15}

    K = K[mask]
    iv = iv[mask]
    k = np.log(K / s)
    w = (iv ** 2) * tt

    idx = np.argsort(k)
    k = k[idx]
    w = w[idx]

    # ATM statistics
    atm_idx = np.argsort(np.abs(k))[: max(1, min(5, len(k)))]
    k_atm = float(np.mean(k[atm_idx]))
    w_atm = float(np.mean(w[atm_idx]))

    # Local slopes for skew asymmetry
    left_mask = k < k_atm
    right_mask = k > k_atm
    if np.sum(left_mask) >= 2:
        sl = np.polyfit(k[left_mask], w[left_mask], 1)[0]
    else:
        sl = -0.05
    if np.sum(right_mask) >= 2:
        sr = np.polyfit(k[right_mask], w[right_mask], 1)[0]
    else:
        sr = 0.05

    # Heuristic mapping to raw-SVI-like params
    m = float(np.clip(k_atm, -0.15, 0.15))
    b = float(np.clip(0.5 * (abs(sl) + abs(sr)), 0.05, 8.0))
    if b > 1e-8:
        rho = float(np.clip((sr + sl) / (2.0 * b), -0.99, 0.99))
    else:
        rho = -0.3

    # Curvature proxy from dispersion of k around ATM
    sig = float(np.clip(np.std(k - m) * 0.8 + 0.05, 0.01, 0.8))

    # Raw-SVI ATM relation: w(0)=a+b*(rho*(0-m)+sqrt((0-m)^2+sig^2))
    atm_term = rho * (0.0 - m) + np.sqrt((0.0 - m) ** 2 + sig ** 2)
    a = float(max(w_atm - b * atm_term, 1e-8))

    guess = {"a": a, "b": b, "rho": rho, "m": m, "sig": sig}

    # Gentle blending with previous calibrated params for stability
    if prev_params:
        blend = 0.35
        out = {}
        for kname, val in guess.items():
            pv = float(prev_params.get(kname, val))
            out[kname] = float((1.0 - blend) * val + blend * pv)
        return out
    return guess

