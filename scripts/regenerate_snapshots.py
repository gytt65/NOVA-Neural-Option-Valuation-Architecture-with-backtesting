#!/usr/bin/env python3
"""
regenerate_snapshots.py
=======================

Regenerates Golden Master snapshots using the current NIRV model codebase.
This is necessary after changing the Random Number Generator (Sobol -> Pseudo)
or improving the pricing physics.

Run:
    python3 scripts/regenerate_snapshots.py
"""

import sys
import os
import json
import glob
import importlib.util
import numpy as np

# ── Setup paths ──────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Import nirv_model
# Import nirv_model
try:
    import nirv_model
except ImportError:
    # Fallback if not in path (though ROOT insert should handle it)
    spec = importlib.util.spec_from_file_location(
        "nirv_model", os.path.join(ROOT, "nirv_model.py")
    )
    nirv_model = importlib.util.module_from_spec(spec)
    sys.modules['nirv_model'] = nirv_model
    spec.loader.exec_module(nirv_model)

NIRVModel = nirv_model.NIRVModel
HestonJumpDiffusionPricer = nirv_model.HestonJumpDiffusionPricer

# Ensure v5 features are OFF
from omega_features import set_features
set_features()  # all OFF

SNAPSHOT_DIR = os.path.join(ROOT, "tests", "golden", "snapshots")

def regenerate():
    pattern = os.path.join(SNAPSHOT_DIR, "*.json")
    files = sorted(glob.glob(pattern))
    
    print(f"Found {len(files)} snapshots in {SNAPSHOT_DIR}")
    
    for f in files:
        print(f"Processing {os.path.basename(f)}...")
        
        with open(f, 'r') as fh:
            data = json.load(fh)
            
        inp = data["inputs"]
        seed = data["seed"]
        n_paths = data["n_paths"]
        n_bootstrap = data["n_bootstrap"]
        
        # Reset RNG
        np.random.seed(seed)
        
        # Instatiate model
        model = NIRVModel(n_paths=n_paths, n_bootstrap=n_bootstrap)
        # Use defaults which now include use_sobol=False, but explicitly set seed
        model.pricer = HestonJumpDiffusionPricer(
            n_paths=n_paths, use_sobol=False, seed=seed
        )
        
        returns_30d = np.array(inp["returns_30d"])
        
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
        
        # Update expected outputs
        # Rounding to match original precision (mostly 1 or 2 decimals)
        # Filter out non-Greek metadata (like quant_extras dict)
        greeks_clean = {k: v for k, v in result.greeks.items() if isinstance(v, (int, float))}
        
        new_outputs = {
            "fair_value": round(result.fair_value, 2),
            "market_price": float(result.market_price),
            "mispricing_pct": round(result.mispricing_pct, 2),
            "signal": result.signal,
            "profit_probability": round(result.profit_probability, 1),
            "physical_profit_prob": round(result.physical_profit_prob, 1),
            "confidence_level": round(result.confidence_level, 1),
            "expected_pnl": round(result.expected_pnl, 2),
            "physical_expected_pnl": round(result.physical_expected_pnl, 2),
            "regime": result.regime,
            "greeks": greeks_clean
        }
        
        data["expected_outputs"] = new_outputs
        
        # Write back
        with open(f, 'w') as fh:
            json.dump(data, fh, indent=2)
            
    print("Done.")

if __name__ == "__main__":
    regenerate()
