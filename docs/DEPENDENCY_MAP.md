# Dependency Map (Insertion-Oriented)

This map documents where upgrades were inserted without refactoring unrelated architecture.

```mermaid
flowchart TD
  APP[opmAI_app.py] --> FLAGS[omega_features.py]
  APP --> OMEGA[omega_model.py]
  APP --> NIRV[nirv_model.py]
  APP --> QE[quant_engine.py]

  OMEGA --> FF[FeatureFactory]
  OMEGA --> NIRV
  OMEGA --> BSE[behavioral_state_engine.py]

  NIRV --> MVAR[model_free_variance.py]
  NIRV --> ESSVI[essvi_surface.py]
  NIRV --> ROUTER[pricer_router.py]
  NIRV --> SURFCHECK[surface_checks.py]

  QE --> GEX[GEXCalculator]
  QE --> NSDE[NeuralSDECalibrator]

  ROUTER --> RB[RBergomiPricer]

  TESTS[tests/*] --> OMEGA
  TESTS --> NIRV
  TESTS --> QE
  TESTS --> ROUTER
  TESTS --> MVAR
  TESTS --> BSE
```

## Integration Points

- Surface calibration:
  - `nirv_model.py` SVI objective (optional liquidity/interval modes)
  - `essvi_surface.py` slice refinement (optional liquidity/interval modes)
- Pricing router:
  - `pricer_router.py` (`config.model == "rbergomi"` path)
- Behavioral state:
  - `behavioral_state_engine.py` additive payload passed via optional kwargs
- Dealer flow vector:
  - `quant_engine.py::GEXCalculator.compute_gex` (legacy fields preserved)
- VIX variance forward estimator:
  - `model_free_variance.py::estimate_forward_from_chain`
  - `nirv_model.py` uses improved estimator only when feature flag is enabled
