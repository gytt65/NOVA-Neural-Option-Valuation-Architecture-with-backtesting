# Changelog

## Unreleased (Stability-First Safe Extension)

### Added
- New optional behavioral state module:
  - `/Users/caaniketh/Optionpricingmodel Ai/OPM-with-TVR-NIRV-OMEGA-Models-/behavioral_state_engine.py`
  - Exposes additive `market_state` payload with `regime`, `vrp`, `srp`, `behavioral`, and `liquidity`.
- New optional feature flags (all default `False`):
  - `USE_LIQUIDITY_WEIGHTING`
  - `USE_INTERVAL_LOSS`
  - `USE_STALENESS_FEATURES`
  - `USE_ENHANCED_RANKING`
  - `USE_IMPROVED_VIX_ESTIMATOR`
  - `ENFORCE_STATIC_NO_ARB`
- Enhanced dealer-flow vector fields in `GEXCalculator` while preserving legacy outputs:
  - `gex_sign`, `bucketed_gex`, `gamma_flip`, `charm`, `vanna`.
- rBergomi plug-in route in tiered pricer router (`config.model == "rbergomi"`).
- Additional tests for new modules and regression safety.

### Changed (Flag-Gated, Default Behavior Preserved)
- `nirv_model.py` SVI calibration:
  - Optional liquidity-aware weighting: `vega / (spread^2 + eps)`.
  - Optional interval loss if IV bid/ask bounds are provided.
- `essvi_surface.py`:
  - Optional weighted and interval-aware slice refinement.
- `model_free_variance.py`:
  - Added parity-based forward estimator with `legacy` and `improved` methods.
  - Variance engine can infer forward when missing (existing call paths remain unchanged unless used).
- `omega_model.py`:
  - Optional staleness feature extraction.
  - Optional enhanced chain ranking mode.
  - Optional behavioral-state pass-through.
- `quant_engine.py`:
  - Optional static no-arbitrage projection for neural calibrator outputs.
- `opmAI_app.py`:
  - Added sidebar toggles for new feature flags.

### Compatibility Notes
- No existing public function names were removed/renamed.
- Legacy behavior remains the default execution path with flags disabled.
- Existing ranking, calibration, and pricing paths are preserved unless feature flags are enabled.
