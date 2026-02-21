# OMEGA v6.0 Upgrade Notes

This document summarizes the v6.0 upgrade focused on correctness, regression safety, and CPU-friendly performance.

## What Changed

1. NSE contract-spec layer (`nse_specs.py`)
- Added `ContractSpecResolver` as a single source for expiries, lot sizes, tick sizes.
- Supports local contract-master CSV/CSV.GZ with broker fallback interfaces.
- Added minute-accurate `time_to_expiry_minutes(...)`.
- Added Tuesday-centric fallback expiry generation with holiday rollback.

2. NSE-consistent model-free variance engine (`model_free_variance.py`)
- Added VIX-style model-free variance implementation:
  - midpoint from bid/ask
  - spread quality filtering
  - spline interpolation for missing interior quotes
  - minute-based 30-day interpolation
- Optional conservative tail correction (flag-gated).
- `india_vix_synth.py` is retained and now routes to this engine when enabled.

3. Surface robustness upgrades
- Added `essvi_surface.py` (lightweight bounded eSSVI).
- Added `svi_fixed_point.py` warm-start for SVI calibration.
- Added `surface_checks.py` for butterfly/calendar/call-price no-arbitrage checks.
- `arbfree_surface.py` now runs and exposes diagnostics after fit.

4. VRP term-structure state (`vrp_state.py`)
- Added model-free VRP level/slope estimation:
  - `VRP(T)=RN_var(T)-E[RV(T)]`
  - horizons: 7d/30d/60d
- Added conservative parameter multipliers for `kappa/theta/sigma_v`.

5. Tiered pricer routing (`pricer_router.py`)
- Added CPU-budgeted routing:
  - Tier 0: surface-IV BS
  - Tier 1: Heston COS
  - Tier 2: selective QMC MC (candidate-gated)
- Integrated in `nirv_model.py` behind flag.

6. ML uncertainty intervals (conformal)
- Added conformal-style prediction intervals in `omega_model.py` (global + regime-binned quantiles).
- Mispricing actionability can be interval-gated when enabled.

7. Streamlit integration
- Added sidebar v6 feature toggles (default OFF).
- Added one-click **Apply Best Mode (MacBook CPU)** profile in the sidebar.
- Added one-click **Max Accuracy (CPU Intensive)** profile in the sidebar.
- Added one-click **Max Results (CPU Uncapped)** profile for research-only deep scans.
- Added runtime safety controls:
  - **Strict Live Data** (blocks runs when synthetic fallback data is used)
  - **Emergency Kill Switch** (blocks new pricing/scans immediately)
  - session fallback telemetry + reset button
- Added cached model factories (`st.cache_resource`) and explicit model refresh control.

## Feature Flags

All new v6 flags default `OFF`:

- `USE_NSE_CONTRACT_SPECS`
- `USE_NSE_VIX_ENGINE`
- `USE_TAIL_CORRECTED_VARIANCE`
- `USE_ESSVI_SURFACE`
- `USE_SVI_FIXED_POINT_WARMSTART`
- `USE_MODEL_FREE_VRP`
- `USE_TIERED_PRICER`
- `USE_CONFORMAL_INTERVALS`
- `USE_LIQUIDITY_WEIGHTING`
- `USE_INTERVAL_LOSS`
- `USE_STALENESS_FEATURES`
- `USE_ENHANCED_RANKING`
- `USE_IMPROVED_VIX_ESTIMATOR`
- `ENFORCE_STATIC_NO_ARB`
- `USE_RESEARCH_HIGH_CONVICTION`
- `USE_OOS_RELIABILITY_GATE`

Legacy v5 flags remain available:

- `india_vix_synth`
- `arb_free_surface`
- `vrr_state`
- `surface_shock`

## Updated Assumptions

- Weekly/monthly fallback expiry logic is Tuesday-centric when contract masters are not available.
- Lot-size resolution is dynamic via contract specs with explicit fallback warnings.
- VIX interpolation is minute-based in the new model-free variance engine.

## Performance Mode (MacBook CPU)

Recommended settings for fast chain scans:

1. Keep these flags OFF for fastest path:
- `USE_ESSVI_SURFACE`
- `USE_TIERED_PRICER`
- `USE_NSE_VIX_ENGINE`

2. If pricing fidelity is needed:
- Enable `USE_TIERED_PRICER` with low `cpu_budget_ms` (for scans, e.g. 8–15ms).
- Keep Tier 2 selective by liquidity/anomaly/mispricing prefilters.

3. Use refresh controls:
- Rebuild cached models only when underlying/expiry/feature flags change or when pressing **Refresh Models**.

## Best Mode Profile (MacBook CPU)

Programmatic preset:

- `OmegaFeatures.best_mode_macbook()`
- `set_best_mode_macbook()`
- `OmegaFeatures.best_mode_max_accuracy()`
- `set_best_mode_max_accuracy()`

Profile enables:

- `USE_NSE_CONTRACT_SPECS`
- `USE_NSE_VIX_ENGINE`
- `USE_ESSVI_SURFACE`
- `USE_SVI_FIXED_POINT_WARMSTART`
- `USE_MODEL_FREE_VRP`
- `USE_TIERED_PRICER`
- `USE_CONFORMAL_INTERVALS`

Profile keeps disabled by default:

- `USE_TAIL_CORRECTED_VARIANCE` (can be enabled explicitly for sparse/truncated wings)

Max Accuracy profile:

- Enables all MacBook profile flags plus `USE_TAIL_CORRECTED_VARIANCE`.
- Sets a high default tiered CPU budget in the app (`120 ms/option`), adjustable in sidebar.

Max Results profile (research-only):

- Enables max-accuracy stack plus `USE_RESEARCH_HIGH_CONVICTION` and `USE_OOS_RELIABILITY_GATE`.
- Raises default tiered budget to `2000 ms/option` (user adjustable up to `10000`).
- Focuses scanner output to 9/10 and 10/10 conviction candidates.

Out-of-sample reliability gate:

- Uses tracked realized outcomes to compute directional signal reliability.
- Blocks new BUY/SELL calls when historical tracked samples are insufficient or below reliability thresholds.
- Keeps baseline behavior unchanged when disabled.
