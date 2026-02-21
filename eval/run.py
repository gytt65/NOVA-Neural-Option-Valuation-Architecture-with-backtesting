"""
run.py
======

CLI runner for OMEGA v5 evaluation harness.

Usage:
    python -m eval.run --snapshots tests/golden/snapshots --features '{"arb_free_surface":true}'

This script loads the Golden Master snapshots and re-runs pricing with specific
v5 features enabled, comparing the new outputs against the baseline (v4) outputs
stored in the snapshot JSONs.
"""

import argparse
import sys
import os
import json
import glob
import importlib.util
import numpy as np

# ── Setup NIRV import ────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from omega_features import OmegaFeatures, set_features
import eval.metrics as metrics

try:
    import nirv_model
except ImportError:
    spec = importlib.util.spec_from_file_location(
        "nirv_model", os.path.join(ROOT, "nirv_model.py")
    )
    nirv_model = importlib.util.module_from_spec(spec)
    sys.modules['nirv_model'] = nirv_model
    spec.loader.exec_module(nirv_model)

NIRVModel = nirv_model.NIRVModel
HestonJumpDiffusionPricer = nirv_model.HestonJumpDiffusionPricer


def run_evaluation(snapshot_dir, features_dict, diagnostics=False):
    """
    Run evaluation on all snapshots in the directory.
    """
    # 1. Configure features
    print(f"Setting features: {features_dict}")
    # Convert dict keys to match OmegaFeatures init args if needed
    # (assuming exact match for now)
    set_features(**features_dict) # Global set
    
    # 2. Load snapshots
    pattern = os.path.join(snapshot_dir, "*.json")
    files = sorted(glob.glob(pattern))
    print(f"Found {len(files)} snapshots.")
    
    results = []
    
    for f in files:
        with open(f, 'r') as fh:
            data = json.load(fh)
            
        inp = data["inputs"]
        seed = data["seed"]
        n_paths = data["n_paths"]
        
        # Baseline (v4) values
        base_fair_value = data["expected_outputs"]["fair_value"]
        
        # 3. Run Model (v5)
        # Reset RNG
        np.random.seed(seed)
        
        model = NIRVModel(n_paths=n_paths)
        model.pricer = HestonJumpDiffusionPricer(
            n_paths=n_paths, use_sobol=False, seed=seed
        )
        
        returns_30d = np.array(inp["returns_30d"])
        
        res = model.price_option(
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
        
        # 4. Compare
        diff = res.fair_value - base_fair_value
        pct_diff = (diff / base_fair_value) * 100 if base_fair_value != 0 else 0
        
        print(f"[{os.path.basename(f)}] Base={base_fair_value:.2f} New={res.fair_value:.2f} Diff={diff:.2f} ({pct_diff:.2f}%)")
        if diagnostics:
            qx = {}
            if hasattr(res, "greeks") and isinstance(res.greeks, dict):
                qx = res.greeks.get("quant_extras", {}) or {}
            state = getattr(model, "state", {}) if hasattr(model, "state") else {}
            diag = {
                "model_free_var_30d": state.get("model_free_var_30d") if isinstance(state, dict) else None,
                "model_free_var_term_structure": state.get("model_free_var_term_structure") if isinstance(state, dict) else None,
                "vrp_state": state.get("vrp_state") if isinstance(state, dict) else None,
                "surface_diagnostics": state.get("surface_diagnostics") if isinstance(state, dict) else None,
                "quant_extras": qx,
            }
            print("  diagnostics:", json.dumps(diag, default=str))
        
        results.append({
            "file": os.path.basename(f),
            "base_price": base_fair_value,
            "new_price": res.fair_value,
            "diff": diff,
            "pct_diff": pct_diff
        })
        
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OMEGA v5 Evaluation Runner")
    parser.add_argument("--snapshots", required=True, help="Path to snapshot directory")
    parser.add_argument("--features", default="{}", help="JSON string of features to enable")
    parser.add_argument("--diagnostics", action="store_true",
                        help="Print v6 diagnostics (VIX engine, surface checks, VRP state)")
    
    args = parser.parse_args()
    
    features_dict = json.loads(args.features)
    run_evaluation(args.snapshots, features_dict, diagnostics=args.diagnostics)
