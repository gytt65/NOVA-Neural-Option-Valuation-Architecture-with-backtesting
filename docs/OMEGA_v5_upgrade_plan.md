# OMEGA v5 Upgrade — Design & Implementation Plan

**Status**: Implemented  
**Date**: 2026-02-17  
**Version**: 5.0.0

## 1. Executive Summary

The OMEGA v5 upgrade transitions the option pricing engine from a purely theoretical Heston model to a production-grade, arbitrage-aware, and regime-adaptive system tailored for the Indian markets (NIFTY/BANKNIFTY).

Key objectives:
1. **Safety**: Ensure baseline v4 pricing (Golden Master) is preserved exactly when new features are disabled.
2. **Realism**: Incorporate Indian market conventions (NSE calendar, VIX-based volatility surface).
3. **Stability**: Enforce arbitrage-free constraints on the volatility surface.
4. **Adaptability**: Introduce a Variance Risk Premium (VRP) state filter to adjust pricing based on market fear/complacency.

---

## 2. Architecture Overview

### 2.1 Feature Flag System
- **Central Config**: `omega_features.py`
- **Mechanism**: Boolean flags controlled via environment variables or programmatic overrides.
- **Default**: All flags `False` (v4 behavior).

### 2.2 Core Components

| Component | File | Description |
|-----------|------|-------------|
| **NIRV Model** | `nirv_model.py` | Main pricing engine. Updated to query feature flags. |
| **Pricer** | `HestonJumpDiffusionPricer`| MC engine with regime switching and jump diffusion. |
| **Market Utils** | `market_conventions.py` | NSE calendar logic for accurate `T` calculation. |
| **VIX Synth** | `india_vix_synth.py` | Synthetic VIX calculation from option chain (Phase 2). |
| **Arb-Free Surface** | `arbfree_surface.py` | SVI calibration with butterfly/calendar arbitrage checks (Phase 3). |
| **VRP Filter** | `vrr_state.py` | State-space model for Variance Risk Premium (Phase 4). |
| **Shock Gen** | `surface_shock.py` | Skeleton for generative surface stress testing (Phase 5). |

---

## 3. Phase Details

### Phase 0: Repo Audit & Golden Master Baseline
- Established a "frozen" baseline using 5 synthetic market snapshots.
- Implemented `tests/golden/test_golden_nirv_outputs.py`.
- **Guarantee**: Any regression in pricing > 0.5% or Greeks > 5% fails the build.

### Phase 1: India Market Conventions
- Corrected time-to-expiry calculation to use NSE trading holidays and intraday fraction.
- Added explicit routing for Index vs Stock options.

### Phase 2: Synthetic India VIX
- Implemented a robust VIX calculator for illiquid or missing VIX data.
- Uses cubic spline interpolation on variance swaps.
- **Output**: `india_vix_synth` matches official methodology within ~1-2%.

### Phase 3: Arbitrage-Free Surface
- Added `ArbFreeSurfaceState` to post-process SVI fits.
- **Checks**:
    - **Calendar**: Variance must not decrease with time ($w(k, t_2) \ge w(k, t_1)$).
    - **Butterfly**: Density must be non-negative (convexity check).
- **Repair**: Iterative smoothing to fix violations.

### Phase 4: VRP State Filter
- Implemented a recursive filter for $A_t$ (Risk Aversion Level).
- **Logic**:
    - Fear ($A_t > 0$): Increase jump intensity ($\lambda_J$) and confidence.
    - Complacency ($A_t < 0$): Mean revert faster.
- **Integration**: Adjusts `Heston` parameters dynamically based on $IV - RV$ spread.

### Phase 5: Surface Shock Skeleton
- Created the interface for `SurfaceShockModel`.
- Placeholder implementation returns Gaussian noise.
- Ready for future ML model integration (VAE/GAN).

---

## 4. Evaluation & Verification

### 4.1 Automated Tests
- **Golden Master**: `tests/golden/test_golden_nirv_outputs.py` (Passes)
- **Unit Tests**:
    - `tests/test_market_conventions.py`
    - `tests/test_india_vix_synth.py`
    - `tests/test_arbfree_surface.py`
    - `tests/test_vrr_integration.py`
    - `tests/test_surface_shock.py`

### 4.2 Manual Evaluation
- **Script**: `python -m eval.run --snapshots ...`
- **Metrics**:
    - Vega-weighted RMSE
    - Arbitrage violation counts
    - Surface stability (L2 norm)

---

## 5. Future Work
- Train the `SurfaceShockModel` on historical NIFTY data.
- Implement the "Market Maker Inventory" module.
- Deploy to production with features enabled incrementally.
