#!/usr/bin/env python3
"""
bench_pricing.py — CPU benchmark for OMEGA/NIRV pricing tiers.

Benchmarks 100 strikes x 2 expiries and prints latency summaries.
"""

import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from omega_features import set_features
from nirv_model import NIRVModel


def run_bench(
    use_tiered: bool,
    cpu_budget_ms: float,
    n_strikes: int = 100,
    n_paths: int = 12000,
    n_bootstrap: int = 300,
):
    set_features(USE_TIERED_PRICER=bool(use_tiered))

    model = NIRVModel(n_paths=int(n_paths), n_bootstrap=int(n_bootstrap))
    spot = 23500.0
    r = 0.065
    q = 0.012
    vix = 16.0
    strikes = np.linspace(spot * 0.9, spot * 1.1, int(n_strikes))
    expiries = [7.0 / 365.0, 30.0 / 365.0]
    rets = np.random.normal(0.0003, 0.012, 30)

    latencies = []
    t0 = time.perf_counter()
    n = 0
    for T in expiries:
        for K in strikes:
            p0 = max(15.0, abs(spot - K) * 0.02 + 100.0)
            st = time.perf_counter()
            _ = model.price_option(
                spot=spot,
                strike=float(K),
                T=float(T),
                r=r,
                q=q,
                option_type="CE",
                market_price=float(p0),
                india_vix=vix,
                fii_net_flow=-300.0,
                dii_net_flow=250.0,
                days_to_rbi=18,
                pcr_oi=1.02,
                returns_30d=rets,
                full_chain_mode=True,
                cpu_budget_ms=cpu_budget_ms,
                mispricing_hint=0.0,
            )
            latencies.append((time.perf_counter() - st) * 1000.0)
            n += 1
    total_ms = (time.perf_counter() - t0) * 1000.0

    arr = np.array(latencies)
    print(f"TieredPricer: {use_tiered}")
    print(f"CPU budget per option (ms): {cpu_budget_ms:.1f}")
    print(f"Priced options: {n}")
    print(f"Total wall time (ms): {total_ms:.2f}")
    print(f"Mean latency per option (ms): {arr.mean():.3f}")
    print(f"P50/P95/P99 latency (ms): {np.percentile(arr, 50):.3f} / {np.percentile(arr, 95):.3f} / {np.percentile(arr, 99):.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark pricing latency")
    parser.add_argument("--tiered", action="store_true", help="Enable USE_TIERED_PRICER")
    parser.add_argument("--cpu-budget-ms", type=float, default=8.0, help="CPU budget per option in ms")
    parser.add_argument("--strikes", type=int, default=100, help="Number of strikes")
    parser.add_argument("--paths", type=int, default=12000, help="Monte Carlo paths")
    parser.add_argument("--bootstrap", type=int, default=300, help="Bootstrap samples")
    args = parser.parse_args()
    run_bench(
        use_tiered=args.tiered,
        cpu_budget_ms=args.cpu_budget_ms,
        n_strikes=args.strikes,
        n_paths=args.paths,
        n_bootstrap=args.bootstrap,
    )
