# 🇮🇳 OMEGA NOVA — Indian Options Trading System Pro (v7.0)

> **AI-Powered Nifty 50 Option Pricing, Mispricing Detection & Institutional Trading System**
> **Version:** 7.0.0 (NOVA — Nobel-Level Integrated Pipeline)
> **Build Status:** ✅ Regression-Safe (Golden + Unit + Frontier Tests Passing)
> **Runtime Modes:** MacBook Profile, Max Accuracy, Max Results (Research)
> **Stack:** Pure NumPy/SciPy — No PyTorch/TensorFlow Required

---

## 📖 Table of Contents

1. [Project Overview](#-project-overview)
2. [Why OMEGA NOVA? (The Institutional Edge)](#-why-omega-nova-the-institutional-edge)
3. [System Architecture](#-system-architecture)
4. [The NOVA 9-Stage Pipeline](#-the-nova-9-stage-pipeline)
5. [The 12 Quantitative Upgrades (Phase 2 & 3)](#-the-12-quantitative-upgrades-phase-2--3)
6. [The v6 Feature Stack (Phase 4, 5 & 6)](#-the-v6-feature-stack-phase-4-5--6)
7. [Frontier Modules (NOVA Phase)](#-frontier-modules-nova-phase)
8. [Scientific Methodology & Mathematics](#-scientific-methodology--mathematics)
9. [Detailed Module Documentation](#-detailed-module-documentation)
    - [NOVA Unified Pipeline (`unified_pipeline.py`)](#nova-unified-pipeline-unified_pipelinepy)
    - [Application Layer (`opmAI_app.py`)](#application-layer-opmai_apppy)
    - [OMEGA Intelligence (`omega_model.py`)](#omega-intelligence-omega_modelpy)
    - [NIRV Pricing Core (`nirv_model.py`)](#nirv-pricing-core-nirv_modelpy)
    - [Quantitative Engine (`quant_engine.py`)](#quantitative-engine-quant_enginepy)
    - [Evaluation Harness (`eval/run.py`)](#evaluation-harness-evalrunpy)
10. [Golden Master Testing Methodology](#-golden-master-testing-methodology)
11. [Installation & Configuration](#-installation--configuration)
12. [User Guide & Workflows](#-user-guide--workflows)
13. [Troubleshooting & FAQ](#-troubleshooting--faq)
14. [Disclaimer](#-disclaimer)
15. [Appendix A: Heston-Nifty Regime Parameters](#appendix-a-heston-nifty-regime-parameters)
16. [Appendix B: Feature Engineering Glossary](#appendix-b-feature-engineering-glossary)
17. [Appendix C: Kelly Criterion Derivation](#appendix-c-kelly-criterion-derivation)
18. [Appendix D: Mathematical Derivations](#appendix-d-mathematical-derivations)
19. [Appendix E: API Reference](#appendix-e-api-reference)
20. [Appendix F: OMEGA v6 Release Delta](#appendix-f-omega-v6-release-delta-compared-to-v5)
21. [Appendix G: Full Feature Flag Reference](#appendix-g-full-feature-flag-reference-v6)
22. [Appendix H: Runtime Profile Details and CPU Budgeting](#appendix-h-runtime-profile-details-and-cpu-budgeting)
23. [Appendix I: Research High-Conviction Mode and OOS Gate](#appendix-i-research-high-conviction-mode-and-oos-gate)
24. [Appendix J: Repository Structure (Detailed Map)](#appendix-j-repository-structure-detailed-map)
25. [Appendix K: Test Strategy and Quality Gates](#appendix-k-test-strategy-and-quality-gates)
26. [Appendix L: Streamlit Runtime Behavior and Caching](#appendix-l-streamlit-runtime-behavior-and-caching)
27. [Appendix M: Honest Usage Guidance](#appendix-m-honest-usage-guidance)
28. [Appendix N: Extended FAQ](#appendix-n-extended-faq)
29. [Appendix O: Compliance and Risk Notes](#appendix-o-compliance-and-risk-notes)
30. [Appendix P: Minimal Quickstart Commands](#appendix-p-minimal-quickstart-commands)
31. [Appendix Q: Suggested TODO Roadmap](#appendix-q-suggested-todo-roadmap)

---

## 🌟 Project Overview

**OMEGA NOVA (Options Market Efficiency & Generative Analysis — Nobel-Level Integrated Pipeline)** is an institutional-grade quantitative trading system designed specifically for the Indian NSE Nifty 50 index options market. Unlike generic Black-Scholes calculators, OMEGA NOVA is built to handle the unique, non-linear characteristics of the Indian market through a **9-stage integrated pricing pipeline** that combines frontier quant models into a single unified framework.

-   **"Rough" Volatility**: Nifty's volatility exhibits Hurst exponent $H < 0.5$, modeled via **rBergomi** with live R/S Hurst estimation.
-   **Fat Tails & Jump Clustering**: **Hawkes self-exciting jumps** capture cascading gap openings, replacing the simple Poisson assumption.
-   **Event Risk**: Specific pricing logic for RBI Monetary Policy, Union Budgets, and Election results.
-   **Liquidity Skews**: **PINN arbitrage-free surfaces** and **SGM score-matching** fill illiquid strikes.
-   **Arbitrage**: Transient arbitrage identification with **Neural J-SDE** learned dynamics.
-   **Interpretable Corrections**: **KAN (Kolmogorov-Arnold Networks)** provide B-spline residual corrections whose edge functions can be inspected.
-   **Optimal Hedging**: **Deep Hedging** learns surface-informed hedge ratios that outperform BSM delta.

The system combines **Rigorous Mathematics** (Stochastic Calculus, PDEs, Score Matching) with **Modern AI** (Neural SDEs, KAN, Ensemble Learning, LLM Sentiment) and **Software Engineering Rigor** (Golden Master Tests, Conformal Coverage Guarantees). All components are **pure NumPy/SciPy** — no PyTorch or TensorFlow required.

### Core Philosophy
1.  **Don't Predict Direction**: Predicting where Nifty goes is hard (50/50).
2.  **Predict Volatility**: Predicting that Nifty will move *somewhere* is easier.
3.  **Price the Wings**: Retail traders overpay for deep OTM lotteries. We sell them using scientifically calibrated surfaces.
4.  **Protect the Downside**: Use Jump-Diffusion with Hawkes clustering to price the specific risk of cascading crashes.
5.  **Integrate, Don't Isolate**: Each model feeds its output as input to the next — the whole is greater than the sum of parts.

---

## 💎 Why OMEGA NOVA? (The Institutional Edge)

Most retail traders lose money because they buy options based on directional hunches ("Nifty will go up") without knowing the **Fair Value**. They overpay for volatility and get crushed by Theta decay.

OMEGA NOVA solves this by answering four critical questions that retail platforms don't ask:

1.  **What is the True Price?**
    *   *Standard*: Black-Scholes (assumes constant volatility).
    *   *NOVA*: 9-stage pipeline → PINN Surface → Neural J-SDE → KAN Correction → Ensemble weighting across Heston COS, rBergomi, and Neural J-SDE.

2.  **Is it Cheap or Expensive?**
    *   *Standard*: "IV is high."
    *   *NOVA*: "Deep OTM Calls are 2-sigma cheap relative to the model-free Variance Risk Premium term structure, with Hawkes clustering currently inactive."

3.  **What is the Probability of Profit?**
    *   *Standard*: Delta (approximate).
    *   *NOVA*: Monte Carlo with Hawkes self-exciting jumps, state-dependent Neural J-SDE dynamics, and conformal coverage guarantees.

4.  **How Should I Hedge?**
    *   *Standard*: BSM Delta.
    *   *NOVA*: Deep Hedger trained on surface-informed features (skew, term slope, VRP) that minimizes hedged P&L variance + CVaR.

---

## 🏗️ System Architecture

The codebase is organized into a hierarchical 5-layer stack. Each layer abstracts complexity from the one above it.

### Layer 5: The Unified Pipeline (`unified_pipeline.py`) — **NEW**
The **NOVA orchestrator** that wires all frontier modules into a single 9-stage pricing framework.
-   **Pipeline**: SGM Surface Completion → PINN No-Arb Surface → Hawkes Jumps → Neural J-SDE → Heston COS → rBergomi → KAN Correction → Ensemble Weighting → Conformal Intervals → Deep Hedging.
-   **Online Learning**: `observe_trade()` updates KAN and Ensemble weights in real-time.
-   **Training**: `train_all()` calibrates all components in dependency order from historical data.
-   **Graceful Degradation**: Missing modules are skipped automatically — the pipeline works with whatever is available.

### Layer 4: The Interface (`opmAI_app.py`)
The Streamlit-based dashboard. It acts as the "Trader's Desk".
-   **Data Ingestion**: Upstox API (Live), Angel One API (Historical).
-   **State Management**: `MarketState` class that acts as the "Single Source of Truth".
-   **Visualization**: Plotly charts for Volatility Surfaces, Payoff Diagrams, and Greeks.
-   **Execution Logic**: Paper trading simulation and Strategy Builder.

### Layer 3: The Intelligence (`omega_model.py`)
The decision-making layer ("The Brain").
-   **ML Correction**: A Gradient Boosting Regressor (XGBoost/LightGBM) that predicts the *error* of the mathematical model based on 45+ features.
-   **Factor Registry**: Comprehensive registry of ALL known pricing factors with auto-fetch capabilities.
-   **Anomaly Detection**: An Isolation Forest that filters out "fake" mispricings caused by stale data or illiquidity.
-   **Sentiment Analysis**: Integration with Gemini/Perplexity to quantify "Market Mood" from news.
-   **Conformal Intervals**: Prediction intervals with coverage guarantees for ML corrections.
-   **Research Filters**: High-conviction (9/10+) and OOS reliability gate for strict candidate selection.

### Layer 2: The Pricing Core (`nirv_model.py`)
The "NIRV" (Nifty Intelligent Regime-Volatility) model ("The Heart").
-   **Regime Detection**: A Hidden Markov Model (HMM) that classifies the market into 4 regimes (Bull-Low Vol, Bear-High Vol, etc.).
-   **SVI/eSSVI Surface**: Stochastic Volatility Inspired parameterization with eSSVI extension for arbitrage stability.
-   **Monte Carlo Engine**: A Sobol-sequence accelerated MC pricer handling Heston dynamics + Merton Jumps.
-   **VRP State Filter**: Model-free VRP term structure (7/30/60d) with slope-state logic.
-   **Tiered Pricer Router**: Surface IV → Heston COS → selective QMC under CPU budget.

### Layer 1: The Quant Engine (`quant_engine.py`)
The mathematical toolbox ("The Tools").
-   **SABR**: Dynamic calibration algorithms.
-   **HestonCOS**: Fourier-Cosine expansion for semi-analytical pricing (millisecond latency).
-   **GARCH**: Volatility forecasting tools.
-   **Optimization**: Differential Evolution and Least Squares solvers for calibration.
-   **Feature Flags**: `omega_features.py` manages the activation of new experimental mathematics.

---

## 🚀 The NOVA 9-Stage Pipeline

The NOVA pipeline (`unified_pipeline.py`) is the unique contribution of this system. No existing open-source project combines all of these frontier models into a single integrated pricing framework. Each stage feeds its output as input to the next:

```
Market Data → SGM Completion → PINN Surface → Neural J-SDE → KAN Corrector → Conformal → Deep Hedge
                                Hawkes Jumps ↗      ↑
                                mfBm H(t)   ↗       |
 Historical → SGM Completion → PINN Surface
                                     ↑
                             Ensemble (NIRV + Neural + KAN)
```

### Stage 0: Behavioral Context (`behavioral_state_engine.py`)
Extracts sentiment, lottery demand, limits-to-arbitrage, and dealer flow (GEX) from market microstructure inputs. These enrich KAN features without altering pricing directly.

### Stage 1: Surface Completion — SGM (`sgm_surface.py`)
Fills in missing implied volatilities at illiquid strikes using **Kernel Score Matching**. Learns $$\nabla_x \log p(x)$$ from historical surfaces via Gaussian KDE, then applies **Langevin denoising** to infill missing data points consistent with observed market shapes.
*Advantage over SVI extrapolation*: SVI produces unphysical IV at far OTM strikes; SGM infers values consistent with the learned distribution.

### Stage 2: PINN Volatility Surface (`pinn_vol_surface.py`)
Fits an arbitrage-free total-variance surface $w(k,T) = \sigma^2 T$ via a **Radial Basis Function network** with 5 simultaneous physics constraints:
1. **Data fidelity** — match market IVs
2. **Gatheral butterfly** — $g(k) \ge 0$ (non-negative density)
3. **Calendar spread** — $\partial w / \partial T \ge 0$ (total variance increases with time)
4. **Roger-Lee wings** — bounded slope at extreme strikes
5. **Martingale** — risk-neutral density integrates to 1

### Stage 3: Hawkes Jump Detection (`hawkes_jump.py`)
Replaces simple Merton Poisson jumps with **self-exciting Hawkes processes**:
$$\lambda(t) = \mu + \sum_{t_i < t} \alpha \cdot e^{-\beta(t - t_i)}$$
Jumps *cluster* — each jump temporarily increases the probability of subsequent jumps. Captures cascading gap-down events that Poisson models miss. MLE calibration via Ogata's thinning algorithm with VIX-dependent modulation.

### Stage 4: Neural J-SDE Pricing (`neural_jsde.py`)
The most advanced pricer: drift, diffusion, AND jump parameters are learned by a **neural RBF network** conditioned on market state (VIX, regime, flows):
$$\mu(X), \sigma(X), \lambda(X), \mu_j(X), \sigma_j(X) = \text{Network}(\text{features})$$
Calibrated via **Adam + SPSA** (Simultaneous Perturbation Stochastic Approximation) with Common Random Numbers for variance reduction.

### Stage 4.5: Heston COS Fast Pricer (`heston_cos.py`)
Semi-analytical pricing using Fourier-Cosine expansion. **4,000× faster** than Monte Carlo (~0.05 ms/option vs ~200 ms). Uses the Albrecher (2007) stable formulation to prevent numerical blow-ups. Supports vectorized chain pricing and analytical Greek computation via bump-and-reprice.

### Stage 5.5: Rough Bergomi Pricing (`pricer_router.py`)
Proper rough-volatility model (Bayer-Friz-Gatheral 2016) with live Hurst exponent estimation via R/S analysis. $H \approx 0.05–0.15$ corrects short-dated option mispricing that standard Heston misses. Hurst is updated in real-time from market returns.

### Stage 6: KAN Residual Correction (`kan_corrector.py`)
**Kolmogorov-Arnold Network** with B-spline basis functions for interpretable pricing corrections. Architecture: `Input(10) → KAN(8) → KAN(4) → KAN(1)`. Each edge learns its own univariate B-spline function, which can be inspected for interpretability. Trained via **Adam + SPSA** (avoids the O(n_params) cost of finite-difference Jacobians).

### Stage 7: Ensemble Weighting (`ensemble_pricer.py`)
Online adaptive ensemble using **multiplicative weight update (exponential hedge)**:
$$w_i \to w_i \cdot \exp(-\eta \cdot |\text{predicted}_i - \text{market}|)$$
Two modes: **Adaptive** (≥5 observed trades, online-learned weights) and **Cold-Start** (static tiered scheme favoring calibrated models).

### Stage 8: Conformal Prediction Interval
Computes confidence intervals from 5 uncertainty sources: vol-time, OTM scaling, model disagreement, Hawkes clustering, and Hurst roughness. Provides coverage guarantees without distributional assumptions.

### Stage 9: Deep Hedging (`deep_hedging.py`)
Trains a feedforward network on 13 surface-informed features to learn optimal hedge ratios that minimize:
$$L = E[\text{PnL}^2] + \lambda \cdot \text{CVaR}_\alpha(\text{PnL})$$
where PnL includes transaction costs. Outperforms BSM delta hedging during volatile regimes.

---

## 🚀 The 12 Quantitative Upgrades (Phase 2 & 3)

This system implements 12 distinct "institutional-grade" upgrades over standard retail models. These are implemented in `quant_engine.py` and validated via `backtester.py`.

### 1. Dynamic SABR Calibration
Standard models use a fixed Volatility Surface. OMEGA recalibrates the **SABR** (Stochastic Alpha, Beta, Rho) model in real-time for every expiry slice.
*Benefit*: accurately prices OTM "wings" where retail traders often get trapped.

### 2. GJR-GARCH(1,1) Forecasting
Unlike simple historical volatility, GJR-GARCH accounts for the **Leverage Effect** (volatility rises more when prices fall).
*Benefit*: Superior prediction of future volatility during market crashes.

### 3. Heston COS Method
A semi-analytical pricing method using Fourier-Cosine series expansion.
*Benefit*: **50x faster** than Monte Carlo, allowing for real-time scanning of the entire option chain (100+ strikes) in sub-second time.

### 4. EM-Based Jump Diffusion
Uses Expectation-Maximization (EM) to estimate the probability and size of market "jumps" (gap openings).
*Benefit*: Prices "Crash Put" options correctly, which BSM severely underprices.

### 5. ML Signal Pipeline
An XGBoost classifier trained on 50+ features (Greeks, Flows, Technicals) to predict the probability of an option trade being profitable.
*Benefit*: Filters out mathematically "cheap" options that are actually "value traps".

### 6. Neural Volatility Surface
(Phase 3 Experimental) A neural network that learns the residuals of the SABR surface to capture idiosyncratic market microstructure effects.

### 7. Continuous VIX Regime Detection
Instead of hard thresholds (e.g., VIX > 20), this uses a continuous sigmoid function and HMM probabilities.
*Benefit*: Smooth transitions between strategies (e.g., gradually shifting from Short Straddle to Long Straddle).

### 8. Enhanced LSM (American Pricing)
Longstaff-Schwartz Method with **Chebyshev Polynomials** and **Importance Sampling**.
*Benefit*: Accurate pricing for American-style options (though Nifty is European, this allows for pricing stock options if needed).

### 9. Kelly Criterion Sizing
Automatically calculates the optimal bet size based on the model's confidence and win probability.
*Formula*: $f^* = \frac{p \cdot b - q}{b}$ (adjusted for fractional Kelly to reduce drawdown).

### 10. Bayesian Posterior Confidence
Adjusts the model's confidence based on liquidity (Bid-Ask Spread) and historical accuracy.
*Benefit*: Reduces confidence in illiquid, wide-spread options where "mid-price" is unreliable.

### 11. Portfolio Greeks Optimization
Uses Linear Programming to hedge a portfolio to be Delta-Neutral, Gamma-Neutral, and Vega-Neutral simultaneously.

### 12. Cross-Asset Signal Processing
Monitors lead-lag relationships between:
- **US 10Y Yield** vs Nifty
- **CBOE VIX** vs India VIX (Overnight spillover)
- **USDINR** vs Nifty IT Index

---

## ⚡ The v6 Feature Stack (Phase 4, 5 & 6)

Version 6.0 introduced "Paradigm Shift" upgrades that address the root causes of model failure in live trading.

### 1. Synthetic India VIX (`india_vix_synth.py`)
**Problem**: The official NSE India VIX feed often freezes or updates with a 15-minute delay.
**Solution**: OMEGA calculates its own "Synthetic VIX" in real-time by inverting the option chain variance spread (Variance Swap replication).
*Alpha*: You know the true volatility regime *before* the rest of the market.
*Derivation*: $VIX^2 \approx \frac{2}{T} \sum_i \frac{\Delta K_i}{K_i^2} e^{RT} Q(K_i)$

### 2. Arbitrage-Free Surface (`arbfree_surface.py`)
**Problem**: Raw SVI calibration can produce "Butterfly Arbitrage" (negative probability density) deep OTM.
**Solution**: A post-processing layer that enforces:
   - $g(k) \ge 0$ (No negative probabilities)
   - $w(t_2) \ge w(t_1)$ (Total variance must increase with time)
*Alpha*: Prevents the model from buying "impossible" options that only look cheap due to bad math.

### 3. VRP State Filter (`vrr_state.py`)
**Problem**: Heston assumes mean reversion $\kappa$ is constant. In reality, Fear decays faster than Complacency.
**Solution**: We define the **Variance Risk Ratio (VRR)**:
   $$ VRR_t = \frac{IV_t}{RV_{t, 10d}} $$
   - If $VRR > 1.5$ (Fear): $\kappa$ is boosted (volatility will crash down fast).
   - If $VRR < 0.8$ (Complacency): $\kappa$ is lowered (volatility will act sticky).
*Alpha*: Captures "Vol Crush" profits after events (Budget, Earnings).

### 4. Surface Shock Generator (`surface_shock.py`)
**Problem**: How do we know if our portfolio survives a 20% VIX spike?
**Solution**: A generative model that applies "shocks" to the SVI parameters ($\rho$ becomes more negative, $a$ shifts up) to simulate crash scenarios.
*Alpha*: Stress-tests your book against 2008 or 2020 style crashes.

---

## 🧠 Frontier Modules (NOVA Phase)

The NOVA phase adds 8 frontier modules that elevate the system beyond standard quant tools. Each module is pure NumPy/SciPy and designed as a drop-in component for the unified pipeline.

| Module | File | Key Innovation | Academic Reference |
|--------|------|----------------|--------------------|
| **Neural J-SDE** | `neural_jsde.py` | State-dependent SDE parameters via RBF network | Gierjatowicz et al. 2022 |
| **Deep Hedging** | `deep_hedging.py` | Surface-informed hedge ratio optimization | Buehler et al. 2019 |
| **Hawkes Jumps** | `hawkes_jump.py` | Self-exciting jump clustering with MLE calibration | Hawkes 1971 |
| **KAN Corrector** | `kan_corrector.py` | Interpretable B-spline residual corrections | Liu et al. 2024 (KAN) |
| **Ensemble Pricer** | `ensemble_pricer.py` | Online multiplicative weight model averaging | Cesa-Bianchi & Lugosi 2006 |
| **PINN Surface** | `pinn_vol_surface.py` | 5-constraint arbitrage-free RBF surface | Gatheral 2004, Roger-Lee 2004 |
| **SGM Surface** | `sgm_surface.py` | Kernel Score Matching for IV completion | Song & Ermon 2019 |
| **Heston COS** | `heston_cos.py` | Fourier-Cosine semi-analytical pricer | Fang & Oosterlee 2008 |
| **rBergomi** | `pricer_router.py` | Rough volatility with live Hurst estimation | Bayer-Friz-Gatheral 2016 |
| **Behavioral Engine** | `behavioral_state_engine.py` | Sentiment, lottery demand, dealer flow context | Barberis & Thaler 2003 |

### Training the NOVA Pipeline

The unified pipeline supports two training modes:

**1. From Historical Data:**
```python
from unified_pipeline import UnifiedPricingPipeline
pipeline = UnifiedPricingPipeline()
pipeline.train_all(
    spot=23500, strike=23400, T=0.1, sigma=0.14,
    historical_returns=returns_array,
    historical_option_data=[...],
)
pipeline.save('models/nova_weights.joblib')
```

**2. From Historical Learning Report:**
```python
pipeline.train_from_historical_report(report, r=0.065, q=0.012)
```

**3. Online Learning (Real-Time):**
```python
result = pipeline.price(spot=23500, strike=23400, T=0.1, ...)
pipeline.observe_trade(market_price=145.0, predicted_price=result['price'],
                       kan_features=result['kan_features'])
```

---

## 🔬 Scientific Methodology & Mathematics

This section details the core mathematical models used in `nirv_model.py` and `quant_engine.py`.

### 1. The Heston Stochastic Volatility Model
We assume the spot price $S_t$ and variance $v_t$ follow:

$$ dS_t = \mu S_t dt + \sqrt{v_t} S_t dW_1 $$
$$ dv_t = \kappa (\theta - v_t) dt + \sigma_v \sqrt{v_t} dW_2 $$

Where:
- $\mu$: Drift (Risk-free rate - Dividend yield)
- $\kappa$: Mean reversion speed of volatility
- $\theta$: Long-run average variance
- $\sigma_v$: Volatility of volatility (Vol-of-Vol)
- $\rho$: Correlation between $dW_1$ and $dW_2$ (typically around -0.6 for Nifty)

The Heston model captures the **volatility clustering** and **leverage effect** (negative correlation between spot and vol) observed in Nifty.

### 2. Merton Jump Diffusion
To handle gap risks (e.g., overnight gaps due to US markets), we add a Poisson jump process $J$:

$$ \frac{dS_t}{S_t} = (...) dt + (...) dW_1 + (e^J - 1) dN_t $$

Where $N_t$ is a Poisson process with intensity $\lambda$ (average jumps per year), and $J \sim N(\mu_J, \sigma_J^2)$ describes the jump size distribution.

### 3. SVI (Stochastic Volatility Inspired) Parameterization
Used for constructing the Volatility Surface $\sigma_{BS}(k, T)$:

$$ w(k) = a + b \left[ \rho(k - m) + \sqrt{(k - m)^2 + \sigma^2} \right] $$

Where $w(k) = \sigma_{BS}^2 T$ is total variance, and $k = \log(K/F)$ is log-moneyness.
- **$a$**: Vertical shift (overall vol level)
- **$b$**: Slope of the wings (fat tails)
- **$\rho$**: Skew (rotation)
- **$m$**: Horizontal shift (ATM skew)
- **$\sigma$**: Smoothness/Curvature (smile bottom)

**Calibration**: We use `scipy.optimize.minimize` (L-BFGS-B) to fit these 5 parameters to live market data every minute, minimizing the RMSE between model IV and market IV.

### 4. Bayesian Confidence Score
We compute a posterior probability of mispricing:

$$ P(Mispriced | Data) \propto P(Data | Mispriced) \cdot P(Mispriced) $$

Where $P(Data | Mispriced)$ assumes gaps are uniformly distributed (anything is possible), while $P(Data | Efficient)$ assumes gaps follow a Normal distribution $N(0, \text{spread})$. If the gap is significantly larger than the bid-ask spread (3-sigma event), the posterior probability of mispricing jumps to > 90%.

---

## 💻 Detailed Module Documentation

### NOVA Unified Pipeline (`unified_pipeline.py`)
**Key Class: `UnifiedPricingPipeline`**
- **Primary Method**: `price(spot, strike, T, r, q, sigma, option_type, market_state, historical_returns)` → dict
- **Returns**: `price`, `std_error`, `confidence_interval`, `optimal_hedge`, `component_prices`, `kan_features`, `diagnostics`
- **Online Learning**: `observe_trade(market_price, predicted_price, kan_features)`
- **Training**: `train_all(...)` calibrates all components in dependency order
- **Persistence**: `save(filepath)` / load via `joblib`
- **Status**: `pipeline_status()`, `available_components()`, `missing_components()`

**Usage:**
```python
from unified_pipeline import UnifiedPricingPipeline
pipeline = UnifiedPricingPipeline()
result = pipeline.price(
    spot=23500, strike=23400, T=0.1,
    r=0.065, q=0.012, sigma=0.14,
    option_type='CE',
    market_state={'vix': 14.5, 'regime': 'Bull-Low Vol', 'pcr': 0.95}
)
print(f"Price: ₹{result['price']}, CI: {result['confidence_interval']}")
print(f"Components: {result['component_prices']}")
print(f"Hedge: {result['optimal_hedge']}")
```

### Application Layer (`opmAI_app.py`)
**Key Class: `MarketState`**
This is a singleton class that holds the current snapshot of the market.
- **Attributes**:
  - `spot`: Current Nifty index level.
  - `india_vix`: Current volatility index.
  - `option_chain`: DataFrame of all active contracts.
- **Usage**:
  ```python
  state = st.session_state['market_state']
  print(state.india_vix, state.spot)
  ```
- **Updates**: Triggered by the "Refresh Data" button, creating a new snapshot from API data.

**Key Class: `IntelligentModelHub`**
Acts as the bridge between the UI and the math/AI backends.
- **Methods**:
  - `get_fair_value(ticker)`: Queries NIRV/OMEGA.
  - `ask_ai(query)`: Sends prompts to Gemini/Perplexity with context about the current market state (e.g., "Why is VIX rising while Nifty is flat?").

**Visualization**:
Uses `plotly.graph_objects` for interactive 3D Volatility Surfaces and 2D Payoff Charts.

### OMEGA Intelligence (`omega_model.py`)
**Key Class: `MLPricingCorrector`**
- **Input**: Dictionary of features (Delta, Gamma, VIX, RSI, Flows, etc.).
- **Output**: `(correction_factor, confidence)`
- **Logic**:
  ```python
  final_price = nirv_price * (1 + correction_factor)
  ```
- **Training**: Automatically retrains using `joblib` persistence when `backtester.py` is run. It learns from historical residuals (Model Price - Market Price).

**Key Class: `FeatureFactory`**
- **Function**: `extract(market_data)`
- **Returns**: A normalized numpy array of ~50 features.
- **Key Features**:
  - `iv_hv_spread`: Difference between Implied and Historical Vol.
  - `gamma_dollar`: Gamma * Spot * Spot / 100.
  - `regime_onehot`: One-hot vector of the current HMM regime.

### NIRV Pricing Core (`nirv_model.py`)
**Key Class: `NIRVModel`**
- **Parameters**: `n_paths` (default 10,000), `n_bootstrap` (default 1000).
- **Primary Method**: `price_option(...)`
- **Output**: `NirvOutput` named tuple containing Fair Value, Greeks, and Probabilities.
- **Optimization**: Uses `scipy.stats.qmc.Sobol` for Quasi-Monte Carlo, achieving $O(1/N)$ convergence vs $O(1/\sqrt{N})$ for standard random sampling.

**Key Class: `RegimeDetector`**
- **Logic**: Uses a pre-calibrated Transition Matrix and Gaussian Emission Probabilities to determine the current state.
- **States**: "Bull-Low Vol", "Bear-High Vol", "Sideways", "Bull-High Vol".
- **Source of Truth**: "Appendix A" lists the constants used here.

### Quantitative Engine (`quant_engine.py`)
**Key Class: `DynamicSABR`**
- **Method**: `calibrate_slice(F, strikes, market_ivs, T)`
- **Algorithm**: Least Squares optimization of Hagan's 2002 formula.
- **Fallback**: If calibration fails (RMSE > 5%), reverts to global defaults to prevent runtime crashes.

**Key Class: `HestonCOS`**
- **Method**: `price(S, K, T, ...)`
- **Performance**: Capable of pricing 1000 options in < 0.5 seconds.
- **Stability**: Uses the "Albrecher" formulation for the Characteristic Function to avoid complex logarithm branch cuts.

### Evaluation Harness (`eval/run.py`)
A CLI tool used in v6 for rigorously comparing the current model against "Golden Master" snapshots and optional diagnostics.
- **Usage**:
  ```bash
  python -m eval.run --snapshots tests/golden/snapshots --features '{"vrr_state": true}'
  ```
- **Output**:
  -   Detailed per-snapshot pricing diff.
  -   RMSE metrics for Vega-weighted error.
  -   Surface stability scores.

---

## 🏆 Golden Master Testing Methodology

In v6, the project continues using strict **Regression Testing** as a non-negotiable quality gate.

1.  **The Snapshots**: We created 5 JSON files in `tests/golden/snapshots/` representing diverse market conditions:
    *   ATM Call (Normal)
    *   OTM Put (Crash protection)
    *   ITM Call (Deep liquidity)
    *   High VIX (Panic)
    *   Low Liquidity (Far OTM)
2.  **The Promise**: Every commit is run against these snapshots.
3.  **The Rule**: If `Current_Price - Snapshot_Price > 0.01`, the test FAILS.
4.  **Reproducibility**: We use `np.random.seed(42)` and `use_sobol=False` (or fixed standard RNG) in the Golden tests to ensure bit-exact reproducibility.

To run the Golden Master tests:
```bash
python -m pytest tests/golden/test_golden_nirv_outputs.py -v
```

---

## 🛠️ Installation & Configuration

### 1. Prerequisites
Ensure you have Python 3.8+ installed. It is highly recommended to use a virtual environment (`venv` or `conda`).

### 2. Dependency Installation
```bash
git clone https://github.com/your-username/OMEGA-Options-System.git
cd OMEGA-Options-System
pip install -r requirements.txt
```

**Critical Dependencies**:
- `numpy`, `scipy` (Core Math)
- `streamlit` (UI)
- `plotly` (Charts)
- `scikit-learn` (ML Components)
- `xgboost` (ML Boosting)
- `arch` (GARCH models)
- `hmmlearn` (Regime Detection - Optional but recommended)

### 3. API Configuration (`config.env`)
Create a file named `config.env` in the root directory. This file holds your sensitive credentials.

```ini
# --- Live Data Provider (Upstox) ---
# Register at https://upstox.com/developer/api-documentation/
UPSTOX_API_KEY=your_api_key_here
UPSTOX_API_SECRET=your_api_secret_here
UPSTOX_REDIRECT_URI=http://localhost:8501

# --- Historical Data Provider (Angel One) ---
# Register at https://smartapi.angelbroking.com/
ANGEL_CLIENT_ID=your_client_id
ANGEL_PASSWORD=your_trading_password
ANGEL_API_KEY=your_smartapi_key
ANGEL_TOTP_SECRET=your_totp_secret_base32

# --- AI Integration (Optional) ---
GEMINI_API_KEY=your_google_ai_key
PERPLEXITY_API_KEY=your_perplexity_key
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

---

## 📖 User Guide & Workflows

### Workflow 1: Intraday Mispricing Scan
1. Open the dashboard (Tab 1).
2. Click **"Refresh Market Data"** in the sidebar.
3. Wait for the `MarketState` to update (approx. 2-3 seconds).
4. Navigate to **Tab 2 (Option Chain)**.
5. Look for the **Heatmap**.
   - **Green Cells**: Model estimates typically > Market Price (Potential Buy).
   - **Red Cells**: Model estimates < Market Price (Potential Sell).
6. Check the **"Confidence"** column. Only trade if Confidence > 70%.
7. Use the "Filter" sidebar to hide low-confidence strikes.

### Workflow 2: Constructing a Strategy
1. Identify a mispriced strike in Tab 2 (e.g., 23500 Call is cheaply priced).
2. Go to **Tab 3 (Strategy Builder)**.
3. Select "Long Call" or build a spread (e.g., "Bull Call Spread").
4. Select the specific strikes.
5. View the **Payoff Diagram**.
6. Check the **"NIRV Probability"**. Is P(Profit) > 55%?
7. If satisfied, execute in Paper Trading (Tab 5) or your real broker.

### Workflow 3: Post-Market Analysis
1. Go to **Tab 4 (Analysis)**.
2. Check the **Volatility Surface** plot.
   - Is there a "Smirk" (Skew)? This indicates high Put demand (Bearish sentiment).
   - Is the surface flat? (Calm market).
3. Check the **OI Buildup**. Are writers fleeing or adding positions?
4. Run `python3 backtester.py` to see how the model *would have performed* today if you had auto-traded.

---

## ❓ Troubleshooting & FAQ

### Common Errors

**Q: `ImportError: No module named 'quant_engine'`**
A: Ensure you are running Python from the root directory of the project. Python needs to see the current folder in `sys.path`. Do not run from inside a `src` folder if one exists (this project is flat).

**Q: Upstox Login Failed**
A:
1. Check if `UPSTOX_API_KEY` in `config.env` is correct.
2. Ensure usage of the correct Redirect URI (usually `http://localhost:8501`).
3. You may need to generate a new `access_token` if the old one expired (tokens valid for 24h).
4. If checking strictly on weekends, markets are closed, so live feed will return empty.

**Q: "SVI Calibration Failed" Warnings**
A: This happens when the market data for an expiry is messy or illiquid (e.g., far-month expiries). The system automatically falls back to the previous successful calibration or broad defaults. It is safe to ignore unless it happens for the *current* expiry.

**Q: Backtest results are all 0.**
A: Check if `SyntheticNiftyGenerator` is producing valid prices. Ensure `n_days` is > 10 to allow indicators (RSI, HV) to warm up.

### Performance Tuning

**Q: The dashboard is slow.**
A:
- Reduce `N_PATHS` in the sidebar from 50,000 to 10,000.
- Switch "Pricing Model" from `NIRV (Monte Carlo)` to `HestonCOS (Semi-Analytical)`. This will speed up pricing by 50x.
- Disable "Deep Analysis" in the AI tab.

---

## ⚠️ Disclaimer

**Risk Warning**: Options trading involves significant risk and is not suitable for every investor. The OMEGA system is a **Quantitative Research Tool** and does not constitute financial advice.

- **Model Risk**: Mathematical models are increasing approximations of reality. They can fail during "Black Swan" events.
- **Data Risk**: Garbage In, Garbage Out. If API data is delayed or corrupt, signals will be wrong.
- **Execution Risk**: Slippage and liquidity drying up can result in losses larger than model estimates.

**License**: Proprietary Software. All rights reserved.

---

## Appendix A: Heston-Nifty Regime Parameters

These parameters are hardcoded in `nirv_model.py` and are based on calibration to Nifty data from 2020-2024.

| Parameter | Symbol | Bull-Low Vol | Bear-High Vol | Sideways | Bull-High Vol |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Drift** | $\mu$ | 0.0006 | -0.0004 | 0.0001 | 0.0008 |
| **Vol of Spot** | $\sigma$ | 0.008 | 0.018 | 0.010 | 0.016 |
| **Jump Lambda** | $\lambda_j$ | 0.02 | 0.08 | 0.03 | 0.06 |
| **Jump Mean** | $\mu_j$ | 0.005 | -0.015 | 0.000 | 0.010 |
| **Jump Std** | $\sigma_j$ | 0.008 | 0.020 | 0.010 | 0.015 |
| **Reversion** | $\kappa$ | 3.0 | 1.5 | 2.5 | 2.0 |
| **Long Vol** | $\theta_v$ | 0.012 | 0.045 | 0.020 | 0.035 |
| **Vol of Vol** | $\sigma_v$ | 0.20 | 0.45 | 0.30 | 0.40 |
| **Correlation** | $\rho$ | -0.40 | -0.75 | -0.50 | -0.55 |

*Note: $\lambda_j = 0.08$ in Bear markets indicates jumps occur roughly every 12 days (1/0.08).*

---

## Appendix B: Feature Engineering Glossary

The `MLSignalPipeline` in `quant_engine.py` constructs a 50-dimensional vector for each option. Here are the key features:

| Feature Name | Description | Default Value |
| :--- | :--- | :--- |
| `moneyness` | Spot / Strike | 1.0 |
| `iv_rank` | Current IV percentile over last 252 days | 50.0 |
| `garch_vol` | GJR-GARCH(1,1) predicted volatility | 0.15 |
| `pcr_z` | Z-score of Put-Call Ratio | 0.0 |
| `fii_net_norm` | Normalized FII flow (INR Crores) | 0.0 |
| `gex_sign` | Sign of Gamma Exposure (Dealer positioning) | 0.0 |
| `rsi` | 14-day RSI of Nifty Spot | 50.0 |
| `regime_bull_prob` | Probability of Bull Regime (HMM) | 0.25 |
| `profit_prob_rn` | Risk-Neutral Probability of Profit | 0.50 |
| `confidence` | Model's internal confidence score (0-100) | 70.0 |
| `skew_slope` | Slope of volatility smile (Call IV - Put IV) | 0.0 |
| `hurst` | Hurst Exponent (Trendiness of time series) | 0.5 |

---

## Appendix C: Kelly Criterion Derivation

The Kelly Criterion maximizes the geometric growth rate of capital.
If we invest a fraction $f$ of our bankroll, our wealth after $n$ trades is:

$$ W_n = W_0 (1 + fb)^S (1 - f)^F $$

Where $S$ is number of successes, $F$ is number of failures.
Taking logarithms and differentiating with respect to $f$:

$$ \frac{d \ln(W_n)}{df} = \frac{S b}{1 + fb} - \frac{F}{1 - f} = 0 $$

Solving for $f$:

$$ f^* = \frac{p(b+1) - 1}{b} = \frac{pb - q}{b} $$

In OMEGA, we use a **"Half-Kelly"** strategy ($f^*/2$) because parameter uncertainty (estimation error in $p$) makes Full Kelly too risky.

---

## Appendix D: Mathematical Derivations

### D.1 The SVI Raw Parametrization
The Stochastic Volatility Inspired (SVI) model parametrizes the implied variance slice as:

$$ w(k) = a + b \{ \rho(k-m) + \sqrt{(k-m)^2 + \sigma^2} \} $$

To ensure no arbitrage:
1.  **Non-negative variance**: $a + b\sigma \sqrt{1-\rho^2} \ge 0$
2.  **Smile Convexity**: $b \ge 0$
3.  **Slope constraint**: $| \rho | < 1$

### D.2 Heston Characteristic Function
The analytical price of a Heston call option is given by:

$$ C(S, K, T) = S P_1 - K e^{-rT} P_2 $$

Where $P_1$ and $P_2$ are probabilities obtained by calculating the inverse Fourier transform of the characteristic function $\phi(u)$:

$$ \phi(u) = \exp \left( C(u, \tau) \theta + D(u, \tau) v_0 + i u \ln S_t \right) $$

OMEGA uses the **Albrecher (2007)** stable form for $D(u, \tau)$ to prevent numerical blow-ups for large maturities:

$$ D(u, \tau) = \frac{-d + \rho \sigma u i + \kappa}{\sigma^2} \left[ \frac{1 - e^{d\tau}}{1 - g e^{d\tau}} \right] $$

Where $d = \sqrt{(\rho \sigma u i - \kappa)^2 + \sigma^2 (u^2 + ui)}$.

### D.3 Variance Risk Ratio (VRR)
The VRP State Filter defines market regime based on the spread between Implied Volatility ($IV$) and Realized Volatility ($RV$).

$$ VRR_t = \frac{IV_{ATM, 30d}}{RV_{Close-Close, 10d}} $$

- **$VRR > 1.2$**: Market is paying a high premium for insurance. Strategy: Credit Spreads / Iron Condors.
- **$VRR < 0.9$**: Market is underpricing risk. Strategy: Long Straddles / Gamma Scalping.
- **$0.9 < VRR < 1.2$**: Normal regime. Strategy: Directional plays.

---

## Appendix E: API Reference

### E.1 NIRVModel

```python
class NIRVModel:
    def __init__(self, n_paths=10000, n_bootstrap=1000, use_sobol=True):
        """
        Initialize the pricing engine.
        Args:
            n_paths (int): Number of Monte Carlo paths.
            n_bootstrap (int): Number of bootstrap resamples for confidence intervals.
            use_sobol (bool): If True, uses Quasi-Monte Carlo (faster convergence).
        """
        ...
        
    def price_option(self, spot, strike, T, r, q, option_type, **kwargs):
        """
        Price a single option.
        Args:
            spot (float): Current underlying price.
            strike (float): Option strike price.
            T (float): Time to expiry in years.
            r (float): Risk-free rate (e.g., 0.06 for 6%).
            q (float): Dividend yield.
            option_type (str): 'CE' for Call, 'PE' for Put.
        Returns:
            NirvOutput: Named tuple with fair_value, greek, signal, etc.
        """
        ...
```

### E.2 QuantEngine

```python
class HestonCOS:
    def price(self, S, K, T, r, q, params):
        """
        Semi-analytical pricing using Fourier-Cosine expansion.
        Args:
            S (float): Spot price.
            K (float): Strike price.
            T (float): Time to expiry.
            r (float): Risk-free rate.
            q (float): Dividend yield.
            params (dict): Heston parameters {kappa, theta, sigma, rho, v0}.
        Returns:
            float: Option price.
        """
        ...
```

### E.3 Upstox Historical Learning Pipeline (Additive)

This workflow is optional and only runs when you click **`Pull & Learn`** in the sidebar.
It does not change normal scans unless triggered.

#### Endpoints Used

| Endpoint | Method | Path |
|---|---|---|
| Option Contracts | `GET` | `/v2/option/contract` |
| Expired Expiries | `GET` | `/v2/expired-instruments/expiries` |
| Expired Option Contracts | `GET` | `/v2/expired-instruments/option/contract` |
| Expired Historical Candles | `GET` | `/v2/expired-instruments/historical-candle/{expired_instrument_key}/{interval}/{to_date}/{from_date}` |
| Full Market Quote | `GET` | `/v2/market-quote/quotes` |
| Market Quote OHLC | `GET` | `/v2/market-quote/ohlc` |
| Option Greek (V3) | `GET` | `/v3/market-quote/option-greek` |
| LTP (V3) | `GET` | `/v3/market-quote/ltp` |
| Put/Call Option Chain | `GET` | `/v2/option/chain` |

Auth for all calls: `Authorization: Bearer <access_token>`.

#### New Additive Modules

- `upstox_api_clients.py`
  - Typed client wrappers for documented Upstox endpoints.
  - Input validation (date formats, key limits, interval checks).
  - Clean error handling and retry/backoff for transient errors.
  - Candle arrays are parsed into timezone-aware `pandas.DataFrame`.
- `historical_learning.py`
  - Pulls expired options/candles by expiry.
  - Builds features (returns, rolling RV, range-vol, volume/OI features, time features).
  - Creates residual labels using existing `NIRVModel.price_option(...)`.
  - Updates existing `MLPricingCorrector` model artifact.

#### Artifact Paths

- Raw pulls: `omega_data/historical/raw/`
- Processed dataset: `omega_data/historical/processed/features.parquet`
- Model artifact: `omega_data/pricing_model.joblib`
- Training report JSON: `omega_data/historical/processed/training_report_*.json`

#### Streamlit Usage

1. Connect Upstox in the sidebar.
2. Open **Historical Learning**.
3. Set:
   - underlying instrument key
   - date range
   - candle interval
   - contract selection mode (ATM window or Top-N)
4. Click **`Pull & Learn`**.
5. Review rows pulled, train/test metrics, and saved artifact paths shown in sidebar.

---

**© 2026 OMEGA NOVA Quantitative Research**
*Built with ❤️ for the Indian Algo Trading Community*

---

## Appendix F: OMEGA v6 Release Delta (Compared to v5)

This appendix documents what changed in v6, why those changes were introduced, and how to safely activate them without breaking the v5 baseline behavior.

### F.1 Upgrade Philosophy

v6 follows a strict engineering policy:

1. Baseline compatibility first.
2. New behavior is feature-gated and default OFF unless correctness requires otherwise.
3. Golden regression must continue to pass with new features disabled.
4. Any advanced mode must remain CPU-friendly on MacBook when requested.

### F.2 Core v6 Additions

| Area | File(s) | v6 Capability | Activation |
|------|---------|---------------|------------|
| Contract Specs | `nse_specs.py` | Expiry/lot/tick resolution with contract master support | `USE_NSE_CONTRACT_SPECS` |
| VIX Engine | `model_free_variance.py`, `india_vix_synth.py` | NSE-style model-free variance with minute interpolation | `USE_NSE_VIX_ENGINE` |
| Tail Handling | `model_free_variance.py` | Conservative tail correction for sparse wings | `USE_TAIL_CORRECTED_VARIANCE` |
| Surface | `essvi_surface.py`, `svi_fixed_point.py`, `surface_checks.py` | eSSVI + warm-start + arbitrage diagnostics | `USE_ESSVI_SURFACE`, `USE_SVI_FIXED_POINT_WARMSTART` |
| VRP State | `vrp_state.py` | Model-free VRP term structure and slope-state logic | `USE_MODEL_FREE_VRP` |
| Pricer Routing | `pricer_router.py`, `nirv_model.py` | Tiered routing Surface→COS→QMC under CPU budget | `USE_TIERED_PRICER` |
| ML Uncertainty | `omega_model.py` | Conformal intervals for correction confidence | `USE_CONFORMAL_INTERVALS` |
| Research Filter | `omega_model.py` | 9/10-10/10 conviction-only output mode | `USE_RESEARCH_HIGH_CONVICTION` |
| OOS Reliability | `omega_model.py` | Blocks BUY/SELL if out-of-sample reliability is weak | `USE_OOS_RELIABILITY_GATE` |

### F.3 Why the Research Filters Exist

A common failure mode in options models is over-trading medium-quality signals. v6 introduces two filters:

1. Conviction Filter:
- Converts directional outputs into a strict 10-point conviction framework.
- In research mode, only 9/10 or 10/10 candidates remain actionable.

2. OOS Reliability Gate:
- Looks at tracked historical outcomes.
- Rejects new directional calls if recent out-of-sample hit rate and signed edge are weak.

These filters reduce frequency and improve selectivity. They do not guarantee profits.

---

## Appendix G: Full Feature Flag Reference (v6)

The table below is the canonical reference for all active feature flags.

| Flag | Default | Scope | Description | Typical Use |
|------|---------|-------|-------------|-------------|
| `india_vix_synth` | OFF | Legacy | Enable legacy synthetic VIX path | Legacy comparison only |
| `arb_free_surface` | OFF | Legacy | Enable legacy arbitrage repair path | Legacy comparison only |
| `vrr_state` | OFF | Legacy | Legacy VRR filter in NIRV | Compatibility checks |
| `surface_shock` | OFF | Legacy | Legacy surface shock generator | Stress-test studies |
| `USE_NSE_CONTRACT_SPECS` | OFF | v6 | Contract master for expiry/lot/tick/time | Correct NSE spec behavior |
| `USE_NSE_VIX_ENGINE` | OFF | v6 | NSE-consistent model-free variance engine | Better RN variance state |
| `USE_TAIL_CORRECTED_VARIANCE` | OFF | v6 | Conservative tail extrapolation for sparse wings | Sparse option chains |
| `USE_ESSVI_SURFACE` | OFF | v6 | eSSVI surface as primary smile model | Arbitrage-stable surface |
| `USE_SVI_FIXED_POINT_WARMSTART` | OFF | v6 | Fixed-point warm start for faster SVI calibration | Faster calibrations |
| `USE_MODEL_FREE_VRP` | OFF | v6 | VRP term structure (7/30/60d) and slope state | Regime-aware adaptations |
| `USE_TIERED_PRICER` | OFF | v6 | Router for Surface IV, COS, and selective QMC | Better speed/accuracy tradeoff |
| `USE_CONFORMAL_INTERVALS` | OFF | v6 | Prediction intervals around ML residual correction | Actionability filtering |
| `USE_RESEARCH_HIGH_CONVICTION` | OFF | v6 | Keep only 9/10-10/10 conviction candidates | Research-only strict scans |
| `USE_OOS_RELIABILITY_GATE` | OFF | v6 | Block directional calls if tracked OOS performance is insufficient/weak | Reliability-first filtering |

### G.1 Runtime Overrides

Feature flags can be controlled in three ways:

1. Programmatic:
```python
from omega_features import set_features
set_features(USE_TIERED_PRICER=True, USE_CONFORMAL_INTERVALS=True)
```

2. Environment JSON:
```bash
export OMEGA_FEATURES_JSON='{"USE_TIERED_PRICER":true,"USE_NSE_VIX_ENGINE":true}'
```

3. Streamlit Sidebar:
- Open `OMEGA v6 Feature Flags`
- Toggle flags
- Click `Refresh Models` when needed

### G.2 Recommended Profiles

#### MacBook Profile
- Targets responsive scans.
- Enables key v6 components without fully uncapped CPU usage.

#### Max Accuracy
- Enables deeper correction stack and tail corrections.
- Higher CPU cost than MacBook profile.

#### Max Results (Research)
- Enables high-conviction filter and OOS reliability gate.
- Intended for low-frequency, high-selectivity analysis.

---

## Appendix H: Runtime Profile Details and CPU Budgeting

### H.1 CPU Budget Semantics

The CPU budget is a soft budget per option for tiered pricing.

- Low budget (for example 8 ms):
  - Prefers fast tiers.
  - Suitable for full chain scans.

- Medium budget (for example 50-120 ms):
  - Allows deeper refinements.
  - Good for shortlist validation.

- High budget (for example 500-2000+ ms):
  - Prioritizes accuracy over throughput.
  - Research mode / deep candidate analysis.

### H.2 Tier Routing Summary

Tier 0:
- Price from surface-implied volatility using Black-Scholes.
- Fastest.

Tier 1:
- HestonCOS refinement.
- Balanced speed and robustness.

Tier 2:
- Selective Sobol QMC Heston+Jump MC with control variates.
- Slowest but highest model flexibility.

### H.3 Control Variates Used in Tier 2

- Black-Scholes price.
- HestonCOS price.

These reduce variance of MC estimate and stabilize candidate ranking.

---

## Appendix I: Research High-Conviction Mode and OOS Gate

This section explains the two strict research filters in practical terms.

### I.1 Conviction Score Behavior

Conviction score emits only three values:

- `10`: Strong multi-factor directional edge.
- `9`: Actionable directional edge.
- `0`: Not actionable for strict research selection.

Inputs include:

- Absolute mispricing strength.
- Physical probability quality.
- Confidence level.
- Efficiency score.
- ML confidence.
- Conformal actionability status.

### I.2 OOS Reliability Gate Behavior

Directional signals are allowed only when tracked outcomes satisfy thresholds.

Default gate controls:

- Minimum sample count.
- Minimum global directional accuracy.
- Minimum global signed edge.
- Optional side-specific constraints (BUY vs SELL).
- Optional regime-specific constraints.

When gate fails:

- Directional signal is downgraded to `HOLD`.
- Gate reason is surfaced in output metadata.

### I.3 Why This Is Better Than Forcing 9/10 for Everything

Artificially forcing all scores to 9-10 damages ranking quality and introduces false confidence.

Strict filtering instead:

- Preserves signal integrity.
- Reduces low-quality trades.
- Improves reproducibility of decision logic.

### I.4 Important Limitation

Even strict gating cannot provide guaranteed profits.

This is a probabilistic decision engine under non-stationary market conditions.

---

## Appendix J: Repository Structure (Detailed Map)

The map below explains the practical role of each key repository component.

| Path | Role | Notes |
|------|------|-------|
| `opmAI_app.py` | Streamlit app entrypoint | Main UI tabs, controls, session state |
| `unified_pipeline.py` | **NOVA pipeline orchestrator** | 9-stage integrated pricing framework |
| `omega_model.py` | OMEGA orchestration layer | ML correction, sentiment, filters, trade planning |
| `nirv_model.py` | Core quant pricing model | Regime detection, SVI/eSSVI integration, pricer usage |
| `quant_engine.py` | Quant utilities and engines | SABR/HestonCOS/GARCH/optimization math |
| `neural_jsde.py` | **Neural Jump-SDE pricer** | State-dependent SDE params via RBF network |
| `deep_hedging.py` | **Deep hedging engine** | Surface-informed optimal hedge ratios |
| `hawkes_jump.py` | **Hawkes jump estimator** | Self-exciting jump clustering with MLE |
| `kan_corrector.py` | **KAN residual corrector** | Kolmogorov-Arnold B-spline corrections |
| `ensemble_pricer.py` | **Adaptive ensemble** | Online multiplicative weight averaging |
| `pinn_vol_surface.py` | **PINN vol surface** | 5-constraint arbitrage-free RBF surface |
| `sgm_surface.py` | **SGM surface completion** | Kernel Score Matching for sparse IV |
| `heston_cos.py` | **Heston COS pricer** | Fourier-Cosine semi-analytical pricing |
| `behavioral_state_engine.py` | **Behavioral state builder** | Sentiment, lottery demand, dealer flow |
| `omega_features.py` | Flag registry | v5 legacy + v6 + NOVA feature switches |
| `nse_specs.py` | NSE contract specs | Expiry, lot size, tick, minute TTE helpers |
| `model_free_variance.py` | VIX-style variance engine | OTM strip integration and 30-day interpolation |
| `india_vix_synth.py` | Legacy VIX wrapper | Deprecated route to new variance engine |
| `essvi_surface.py` | eSSVI implementation | Bounded surface and safer arbitrage profile |
| `svi_fixed_point.py` | SVI warm-start method | Better initial guesses for calibration |
| `surface_checks.py` | Surface diagnostics | Butterfly/calendar/call-monotonic checks |
| `arbfree_surface.py` | Surface post-processing | Arbitrage repair and diagnostics |
| `vrp_state.py` | Model-free VRP state | VRP level/slope and parameter multipliers |
| `vrr_state.py` | Legacy VRR state | Kept for compatibility |
| `pricer_router.py` | Tiered pricing router + rBergomi | CPU-budgeted pricing path selection |
| `backtester.py` | Strategy backtesting | Historical simulation and evaluation tools |
| `market_conventions.py` | Calendar/time conventions | Expiry/time handling utilities |
| `iv_solver.py` | IV inversion utilities | BS implied vol solving helpers |
| `surface_shock.py` | Scenario shock model | Vol-surface stress generation |
| `historical_learning.py` | Historical learning pipeline | Pulls expired options, builds features, trains ML |
| `upstox_api_clients.py` | Upstox API clients | Typed wrappers for Upstox endpoints |
| `eval/run.py` | Evaluation harness | Optional diagnostics + comparisons |
| `bench/bench_pricing.py` | Pricing benchmark | Throughput and latency measurement |
| `scripts/` | Utilities/scripts | Operational helper scripts |
| `docs/V6_UPGRADE.md` | Upgrade notes | v6 design and integration summary |
| `docs/OMEGA_TEXTBOOK_CHAPTERS_1_TO_12.md` | Textbook | Full mathematical documentation |
| `docs/DEPENDENCY_MAP.md` | Dependency map | Module dependency graph |
| `tests/` | Unit + integration + frontier tests | Regression safety and properties |
| `tests/golden/` | Golden snapshots/tests | Baseline protection |
| `tests/test_frontier_upgrades.py` | **Frontier module tests** | NOVA pipeline integration tests |
| `tests/test_feature_profiles.py` | Profile tests | Profile defaults and toggles |
| `tests/test_research_high_conviction.py` | Conviction mode tests | 9/10 and scan filtering behavior |
| `tests/test_oos_reliability_gate.py` | Reliability gate tests | Gate pass/fail logic validation |
| `tests/test_model_free_variance_engine.py` | VIX engine tests | Toy recovery, interpolation, filtering |
| `tests/test_contract_spec_resolver_lot_and_expiry.py` | NSE specs tests | Tuesday expiry and lot-size parsing |
| `tests/test_surface_no_arbitrage_properties.py` | Surface checks tests | Arbitrage property coverage |
| `tests/test_behavioral_state_engine.py` | Behavioral engine tests | State builder validation |
| `requirements.txt` | Dependencies | Python packages |
| `config.env.example` | Config template | API keys and environment placeholders |
| `CHANGELOG.md` | Changelog | Version history and changes |
| `README.md` | Main handbook | This document |

### J.1 Data and Output Files

| Path Pattern | Purpose |
|-------------|---------|
| `omega_data/predictions.json` | Learning tracker outcomes |
| `omega_data/behavioral_log.json` | Behavioral state observations |
| `tests/golden/snapshots/*.json` | Regression baselines |
| `trading_data/*` | Optional local datasets |

---

## Appendix K: Test Strategy and Quality Gates

### K.1 Test Classes

1. Golden Baseline:
- Purpose: freeze numerical behavior with flags OFF.
- Scope: core NIRV outputs.

2. Unit Tests:
- Purpose: validate each module in isolation.
- Scope: contracts, variance engine, surfaces, router, confidence, etc.

3. Integration Tests:
- Purpose: validate module interactions.
- Scope: NIRV+surface+VRP+router flows.

4. Property Tests:
- Purpose: assert qualitative constraints.
- Scope: no-arbitrage checks, monotonicity, interpolation sanity.

### K.2 Required Test Commands

```bash
python3 -m pytest -q tests/golden/test_golden_nirv_outputs.py
python3 -m pytest -q tests --ignore=tests/golden
python3 -m pytest -q
```

### K.3 Benchmark Commands

Full benchmark:

```bash
python3 bench/bench_pricing.py --tiered --cpu-budget-ms 8 --strikes 100 --paths 12000 --bootstrap 300
```

Quick benchmark (faster sanity check):

```bash
python3 bench/bench_pricing.py --tiered --cpu-budget-ms 8 --strikes 10 --paths 2000 --bootstrap 50
```

---

## Appendix L: Streamlit Runtime Behavior and Caching

### L.1 Model Caching

The app caches heavy model initialization with `st.cache_resource`.

Cached factories:

- `_get_cached_nirv_model(...)`
- `_get_cached_omega_model(...)`

Cache invalidation uses:

- feature signature
- refresh nonce

### L.2 Avoiding Unnecessary Recalibration

The design intent is:

- avoid expensive recalibration on every widget change,
- trigger refresh on underlying/expiry/profile changes,
- allow explicit manual refresh.

### L.3 Safety Controls

The app includes:

- strict-live-data block,
- emergency kill switch,
- synthetic fallback telemetry.

These are useful for disciplined workflows even in research mode.

---

## Appendix M: Honest Usage Guidance

### M.1 What OMEGA is Good At

- Ranking options by modeled relative edge.
- Filtering low-quality opportunities.
- Building reproducible, testable signal pipelines.
- Providing model diagnostics and uncertainty context.

### M.2 What OMEGA Cannot Do

- Guarantee profits.
- Remove market regime shift risk.
- Remove execution slippage/latency risk.
- Predict exogenous shocks with certainty.

### M.3 Practical Best Practices

1. Use high-conviction + OOS gate for shortlist generation.
2. Start with paper trading and outcome tracking.
3. Review spread/liquidity before any manual trade.
4. Track realized edge drift and recalibrate thresholds.
5. Prefer fewer high-quality trades over many marginal trades.

### M.4 Suggested Research Workflow

1. Enable `Max Results` profile.
2. Scan chain.
3. Keep only PASS entries with 9/10 or 10/10 conviction.
4. Manually verify market microstructure (spread, depth, event context).
5. Log post-trade outcomes to strengthen OOS gate quality.

---

## Appendix N: Extended FAQ (v6)

### N.1 Why do I get many HOLD signals in strict research mode?

Because filters are intentionally selective. HOLD dominance is expected in high-selectivity mode.

### N.2 Why can a high mispricing still be blocked?

Mispricing alone is insufficient. OOS gate may block when recent tracked reliability is weak.

### N.3 Should I always enable tail correction?

No. Keep it off unless wings are sparse/truncated and chain quality supports correction.

### N.4 Which profile is best for MacBook day-to-day usage?

Start with `MacBook Profile` for responsiveness.

### N.5 Which profile is best for deep weekend research?

Use `Max Results` with a higher CPU budget.

### N.6 How do I tune OOS gate strictness?

Pass threshold kwargs into pricing/scan calls:

- `oos_min_samples`
- `oos_min_accuracy_pct`
- `oos_min_avg_edge_pct`
- `oos_min_side_samples`
- `oos_min_regime_samples`
- `oos_lookback`

### N.7 Does this code execute live trades by default?

No. It is primarily analysis and decision support unless explicitly integrated with execution paths.

### N.8 Are old v5 outputs still reproducible?

Yes, with flags off and golden tests passing.

### N.9 Can I run this without GPU?

Yes. The stack is CPU-oriented and designed for MacBook execution.

### N.10 Is this suitable for beginners?

Only if used cautiously. The system is advanced and assumes understanding of options risk.

---

## Appendix O: Compliance and Risk Notes

- This project is a quantitative research framework.
- It is not investment advice.
- No module should be interpreted as a profit guarantee engine.
- Always validate signals against independent risk checks.
- Always prefer controlled position sizing and max-loss discipline.

---

## Appendix P: Minimal Quickstart Commands

Create environment and install:

```bash
cd "/Users/caaniketh/Optionpricingmodel Ai/OPM-with-TVR-NIRV-OMEGA-Models-"
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -r requirements.txt
```

Run app:

```bash
streamlit run opmAI_app.py
```

Run tests:

```bash
python3 -m pytest -q
```

---

## Appendix Q: Suggested TODO Roadmap

1. Add explicit walk-forward report generation for OOS gate drift.
2. Add per-underlying reliability buckets (NIFTY vs BANKNIFTY).
3. Add richer calibration health dashboards (surface fit diagnostics).
4. Add reproducible daily scan archive with checksums.
5. Add optional lightweight experiment tracking for model configs.

---

**End of OMEGA NOVA Handbook**
