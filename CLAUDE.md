# CLAUDE.md — OPM / OMEGA Indian Options Trading System

## Project Overview

**OMEGA** (Options Market Efficiency & Generative Analysis) is an institutional-grade option pricing and trading system for **NSE Nifty 50 index options** (India). It stacks a 6-layer ML/AI engine on top of a rigorous mathematical pricer and exposes everything via a Streamlit web dashboard.

Pricing formula:
```
OMEGA_price = NIRV_base × (1 + ML_correction) + sentiment_adjustment
```

---

## File Map

| File | Purpose | Size |
|------|---------|------|
| `opmAI_app.py` | Streamlit dashboard — main UI, live data, paper trading | ~25,000 lines |
| `omega_model.py` | OMEGA orchestrator — 6-layer ML/AI pricing engine | ~1,800 lines |
| `nirv_model (1).py` | NIRV core — mathematical option pricing (Heston SV + Jump MC + SVI + HMM) | ~1,700 lines |
| `quant_engine.py` | 15 institutional quant methods (SABR, GARCH, COS, GEX, Kelly, etc.) | ~2,000 lines |
| `iv_solver.py` | Jaeckel "Let's Be Rational" machine-precision IV solver | ~350 lines |
| `requirements.txt` | Python dependencies |  |
| `config.env` | API keys template — **never commit with real keys** |  |
| `trading_data/` | SQLite trade journal databases |  |

---

## Architecture

```
opmAI_app.py  (Streamlit Dashboard)
    └── omega_model.py  (OMEGA Engine)
            Layer 0: NIRV mathematical base
            Layer 1: ML correction (GBM learns NIRV residuals)
            Layer 2: Anomaly detection (Isolation Forest)
            Layer 3: Sentiment intelligence (Gemini / Perplexity)
            Layer 4: Behavioral engine (actors: FII, RBI, Fed, Trump)
            Layer 5: Adaptive learning (prediction → outcome → retrain)
        └── nirv_model.py  (NIRV Core)
                RegimeDetector (4-state HMM)
                VolatilitySurface (SVI)
                JumpDiffusionPricer (Heston MC + Sobol QMC)
                BayesianConfidenceEngine
                GreeksCalculator (CRN bump-and-reprice)
            ├── quant_engine.py  (15 quant methods)
            └── iv_solver.py     (Jaeckel rational IV)
```

---

## Running the App

```bash
# Activate venv (pre-configured at .venv/)
source .venv/bin/activate

# Install core dependencies
pip install numpy scipy streamlit plotly pandas

# Optional ML stack
pip install arch hmmlearn scikit-learn xgboost lightgbm joblib

# Run the dashboard
streamlit run opmAI_app.py
```

---

## API Keys (config.env)

The file `config.env` holds all credentials. **It contains real keys — do not commit it.**

| Key | Service | Used for |
|-----|---------|---------|
| `UPSTOX_API_KEY` / `UPSTOX_API_SECRET` | Upstox v3 | Live NSE market data, option chains |
| `GEMINI_API_KEY` | Google Gemini AI | Sentiment analysis (Layer 3) |
| `PERPLEXITY_API_KEY` | Perplexity AI | Sentiment analysis (Layer 3) |
| `ANGEL_*` | Angel One SmartAPI | Alternative broker data feed |
| `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` | Telegram | Trading signal alerts (optional) |

All API keys are loaded at runtime from `config.env` (not hardcoded in source).

---

## Key Classes & Entry Points

### `omega_model.py`
- `OMEGAModel` — master orchestrator; call `price_option(...)` or `scan_chain(...)`
- `FeatureFactory.extract(market_data)` — builds 50+ normalized ML features
- `FactorRegistry` — catalog of 60+ market factors with auto-fetch metadata
- `MLPricingCorrector` — gradient boosting, cold-start safe (needs ≥30 samples)
- `EfficiencyHunter` — Isolation Forest + LOF anomaly scoring
- `SentimentIntelligence.analyse(gemini_resp, perplexity_resp, headlines)` — sentiment score ∈ [−1, +1]
- `BehavioralEngine.predict(actor, context)` — actor pattern prediction
- `PredictionTracker` — records predictions + outcomes for adaptive learning
- `TradePlanGenerator.generate(omega_out, spot)` — entry/exit/stop/Kelly sizing

### `nirv_model (1).py`
- `NIRVModel` — main model; call `price_option(...)`
- Returns `NirvOutput` with `.fair_value`, `.signal`, `.profit_probability`, `.greeks`, `.regime`

### `iv_solver.py`
- `bs_implied_vol(market_price, S, K, T, r, q, option_type)` — Jaeckel IV

### `quant_engine.py`
- `QuantEngine` — unified wrapper for all 15 quant tools

---

## Important Conventions

- **Option types:** Use `'CE'` / `'PE'` (Indian convention), not `'call'`/`'put'`
- **Underlying:** Nifty 50 index options on NSE; lot size defaults to 65
- **VIX:** India VIX (not CBOE), passed as a percentage (e.g. `14.0` = 14%)
- **Flows:** FII/DII net flows in ₹ crores (positive = net buy, negative = net sell)
- **Time:** `T` is in years (e.g. `7/365` for 1-week expiry)
- **Rates:** `r` = risk-free rate (~0.065–0.07), `q` = dividend yield (~0.012–0.013)
- **Data persistence:** ML models saved to `omega_data/` (joblib); behavioral/prediction logs in `omega_data/*.json`

---

## Optional Dependencies (graceful degradation)

All heavy ML libraries are optional — the system degrades gracefully:

| Library | If missing |
|---------|-----------|
| `sklearn` | ML correction (Layer 1) and anomaly detection (Layer 2) disabled |
| `arch` | GARCH forecasting in quant_engine disabled |
| `hmmlearn` | HMM falls back to rule-based regime detection |
| `xgboost` / `lightgbm` | ML falls back to sklearn GradientBoosting |
| `joblib` | Model persistence disabled (models retrain each session) |

---

## Common Tasks

**Add a new market factor:**
1. Register it in `FactorRegistry.FACTORS` in `omega_model.py`
2. Add fetching logic in `opmAI_app.py` (the `_fetch_*` methods)
3. Include it in `FeatureFactory.extract()` if it should be an ML feature

**Retrain the ML corrector:**
Call `omega.learn_from_outcome(prediction_id, actual_return)` after each trade resolves. The model auto-retrains at adaptive intervals (every `max(15, n//20)` samples).

**Check model status:**
```python
omega.get_status()  # returns training sample counts, accuracy, feature importance
```

---

## Warnings

- `opmAI_app.py` is ~25,000 lines; read it in chunks using `offset` + `limit` parameters
- The `config.env` file contains real API credentials — never stage or commit it
- The `trading_data/*.db` files are live SQLite databases — handle carefully
- All ML layers are cold-start safe and return zero correction until ≥30 training samples exist
- This software is for educational/research use only — not financial advice
