#!/usr/bin/env python3
"""
generate_golden_snapshots.py — Create Golden Master Snapshots
=============================================================

Generates reproducible market snapshots and runs them through the
NIRV pricing pipeline with fixed RNG seeds.  The expected outputs
are stored alongside the inputs so that regression tests can verify
the baseline is preserved.

Strategy for reproducibility:
  - Use use_sobol=False + explicit seed to pin the RNG
  - Use small n_paths for speed + determinism
  - Reset np.random.seed AND pricer._rng before EACH scenario
  - Avoid T > 7/365 in most scenarios (regime switching has a known
    shape bug with Poisson calls that we don't want golden tests to
    depend on)

Run:
    python3 tests/golden/generate_golden_snapshots.py
"""

import sys
import os
import json
import importlib.util
import numpy as np

# ── Setup paths ──────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

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

SNAPSHOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "snapshots")
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# Fixed seed for all scenarios
MASTER_SEED = 42
N_PATHS = 5000
N_BOOTSTRAP = 1000


def _make_returns(seed, n=30, mean=0.0, std=0.01):
    """Reproducible returns array."""
    rng = np.random.RandomState(seed)
    return rng.normal(mean, std, n).tolist()


# ============================================================================
# Define 5 representative market scenarios
# ============================================================================
# NOTE: All T ≤ 7/365 to avoid regime-switching path which has a known
# Poisson shape bug when lambda_j becomes vectorized.

SCENARIOS = [
    {
        "id": "01_atm_call_normal",
        "description": "ATM call in normal regime (VIX=14, T=7d)",
        "inputs": {
            "spot": 23500.0,
            "strike": 23500.0,
            "T": 7.0 / 365.0,
            "r": 0.065,
            "q": 0.012,
            "option_type": "CE",
            "market_price": 200.0,
            "india_vix": 14.0,
            "fii_net_flow": -500.0,
            "dii_net_flow": 800.0,
            "days_to_rbi": 15,
            "pcr_oi": 0.85,
            "returns_30d_seed": 100,
        },
    },
    {
        "id": "02_otm_put_high_vix",
        "description": "OTM put with high VIX (VIX=24, T=7d)",
        "inputs": {
            "spot": 23500.0,
            "strike": 22800.0,
            "T": 7.0 / 365.0,
            "r": 0.065,
            "q": 0.012,
            "option_type": "PE",
            "market_price": 80.0,
            "india_vix": 24.0,
            "fii_net_flow": -2000.0,
            "dii_net_flow": 1500.0,
            "days_to_rbi": 5,
            "pcr_oi": 1.3,
            "returns_30d_seed": 200,
        },
    },
    {
        "id": "03_itm_call_low_vix",
        "description": "ITM call with low VIX (VIX=12, T=5d)",
        "inputs": {
            "spot": 23500.0,
            "strike": 23000.0,
            "T": 5.0 / 365.0,
            "r": 0.065,
            "q": 0.012,
            "option_type": "CE",
            "market_price": 530.0,
            "india_vix": 12.0,
            "fii_net_flow": 1200.0,
            "dii_net_flow": 400.0,
            "days_to_rbi": 25,
            "pcr_oi": 0.6,
            "returns_30d_seed": 300,
        },
    },
    {
        "id": "04_near_expiry_atm",
        "description": "Near-expiry ATM call (T=1d, VIX=16)",
        "inputs": {
            "spot": 23500.0,
            "strike": 23500.0,
            "T": 1.0 / 365.0,
            "r": 0.065,
            "q": 0.012,
            "option_type": "CE",
            "market_price": 50.0,
            "india_vix": 16.0,
            "fii_net_flow": 0.0,
            "dii_net_flow": 0.0,
            "days_to_rbi": 20,
            "pcr_oi": 1.0,
            "returns_30d_seed": 400,
        },
    },
    {
        "id": "05_otm_call_low_vol",
        "description": "OTM call in low-vol (VIX=10, T=7d)",
        "inputs": {
            "spot": 23500.0,
            "strike": 24000.0,
            "T": 7.0 / 365.0,
            "r": 0.065,
            "q": 0.012,
            "option_type": "CE",
            "market_price": 25.0,
            "india_vix": 10.0,
            "fii_net_flow": 300.0,
            "dii_net_flow": 200.0,
            "days_to_rbi": 30,
            "pcr_oi": 0.7,
            "returns_30d_seed": 500,
        },
    },
]


def run_scenario(scenario):
    """Run a single scenario through NIRV and capture outputs."""
    inp = scenario["inputs"]

    # Generate reproducible returns
    returns_30d = np.array(_make_returns(inp["returns_30d_seed"]))

    # Reset ALL random state before each scenario
    np.random.seed(MASTER_SEED)

    # Create model with pseudo-random paths (no Sobol → fully deterministic)
    model = NIRVModel(n_paths=N_PATHS, n_bootstrap=N_BOOTSTRAP)
    # Replace pricer with non-Sobol, seeded pricer
    model.pricer = HestonJumpDiffusionPricer(
        n_paths=N_PATHS, use_sobol=False, seed=MASTER_SEED
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

    # Extract serializable outputs
    greeks_clean = {}
    for k, v in result.greeks.items():
        if k != 'quant_extras' and isinstance(v, (int, float)):
            greeks_clean[k] = v

    expected = {
        "fair_value": result.fair_value,
        "market_price": result.market_price,
        "mispricing_pct": result.mispricing_pct,
        "signal": result.signal,
        "profit_probability": result.profit_probability,
        "physical_profit_prob": result.physical_profit_prob,
        "confidence_level": result.confidence_level,
        "expected_pnl": result.expected_pnl,
        "physical_expected_pnl": result.physical_expected_pnl,
        "regime": result.regime,
        "greeks": greeks_clean,
    }

    return expected


def main():
    print("=" * 60)
    print("Generating Golden Master Snapshots")
    print(f"  n_paths={N_PATHS}, n_bootstrap={N_BOOTSTRAP}, seed={MASTER_SEED}")
    print(f"  use_sobol=False (for reproducibility)")
    print("=" * 60)

    all_ok = True
    for scenario in SCENARIOS:
        sid = scenario["id"]
        print(f"\n  Running: {sid}")
        print(f"    {scenario['description']}")

        try:
            expected = run_scenario(scenario)

            # Build full snapshot
            snapshot = {
                "id": sid,
                "description": scenario["description"],
                "seed": MASTER_SEED,
                "n_paths": N_PATHS,
                "n_bootstrap": N_BOOTSTRAP,
                "use_sobol": False,
                "inputs": dict(scenario["inputs"]),
                "expected_outputs": expected,
            }
            # Store returns as list
            snapshot["inputs"]["returns_30d"] = _make_returns(
                scenario["inputs"]["returns_30d_seed"]
            )

            path = os.path.join(SNAPSHOT_DIR, f"{sid}.json")
            with open(path, 'w') as f:
                json.dump(snapshot, f, indent=2)

            print(f"    fair_value = {expected['fair_value']}")
            print(f"    signal     = {expected['signal']}")
            print(f"    regime     = {expected['regime']}")
            print(f"    confidence = {expected['confidence_level']}")
            print(f"    → Saved to {os.path.basename(path)}")

        except Exception as e:
            print(f"    ❌ FAILED: {e}")
            import traceback; traceback.print_exc()
            all_ok = False

    print(f"\n{'=' * 60}")
    if all_ok:
        print(f"✅ Generated {len(SCENARIOS)} golden snapshots in {SNAPSHOT_DIR}")
    else:
        print("⚠️  Some snapshots failed — check errors above")
    print(f"{'=' * 60}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
