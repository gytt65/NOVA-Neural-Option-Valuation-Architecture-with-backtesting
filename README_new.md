# 🇮🇳 OMEGA — Indian Options Trading System Pro (v5.0)

> **AI-Powered Nifty 50 Option Pricing, Mispricing Detection & Institutional Trading System**
> **Version:** 5.0.0 "Golden Master"
> **Build Status:** ✅ Passing (5/5 Golden Snapshots)

---

## 📖 Table of Contents

1. [Project Overview](#-project-overview)
2. [Why OMEGA v5? (The Institutional Edge)](#-why-omega-v5-the-institutional-edge)
3. [System Architecture](#-system-architecture)
4. [The 12 Quantitative Upgrades (Phase 2 & 3)](#-the-12-quantitative-upgrades-phase-2--3)
5. [The v5 "Alpha" Features (Phase 4 & 5)](#-the-v5-alpha-features-phase-4--5)
6. [Scientific Methodology & Mathematics](#-scientific-methodology--mathematics)
7. [Detailed Module Documentation](#-detailed-module-documentation)
    - [Application Layer (`opmAI_app.py`)](#application-layer-opmai_apppy)
    - [OMEGA Intelligence (`omega_model.py`)](#omega-intelligence-omega_modelpy)
    - [NIRV Pricing Core (`nirv_model.py`)](#nirv-pricing-core-nirv_modelpy)
    - [Quantitative Engine (`quant_engine.py`)](#quantitative-engine-quant_enginepy)
    - [Evaluation Harness (`eval/run.py`)](#evaluation-harness-evalrunpy)
8.  [Golden Master Testing Methodology](#-golden-master-testing-methodology)
9.  [Installation & Configuration](#-installation--configuration)
10. [User Guide & Workflows](#-user-guide--workflows)
11. [Troubleshooting & FAQ](#-troubleshooting--faq)
12. [Disclaimer](#-disclaimer)
13. [Appendix A: Heston-Nifty Regime Parameters](#appendix-a-heston-nifty-regime-parameters)
14. [Appendix B: Feature Engineering Glossary](#appendix-b-feature-engineering-glossary)
15. [Appendix C: Kelly Criterion Derivation](#appendix-c-kelly-criterion-derivation)
16. [Appendix D: Mathematical Derivations](#appendix-d-mathematical-derivations)
17. [Appendix E: API Reference](#appendix-e-api-reference)

---

## 🌟 Project Overview

**OMEGA (Options Market Efficiency & Generative Analysis)** is an institutional-grade quantitative trading system designed specifically for the Indian NSE Nifty 50 index options market. Unlike generic Black-Scholes calculators found on GitHub, OMEGA is built to handle the unique, non-linear characteristics of the Indian market:

-   **"Rough" Volatility**: Nifty's volatility often exhibits a Hurst exponent $H < 0.5$, meaning price paths are rougher/noisier than standard Brownian motion.
-   **Fat Tails**: Frequent "gap" openings due to global cues (SGX Nifty, Fed rates) which standard Gaussian models underestimate by 40-50%.
-   **Event Risk**: Specific pricing logic for RBI Monetary Policy, Union Budgets, and Election results.
-   **Liquidity Skews**: Realistic modeling of bid-ask spreads that widen for OTM options (the "Smile").
-   **Arbitrage**: The Indian market is *not* perfectly efficient. OMEGA identifies transient arbitrage opportunities (Box spreads, Butterflies) that exist for < 5 minutes.

The system combines **Rigorous Mathematics** (Stochastic Calculus, PDEs) with **Modern AI** (Gradient Boosting, LLM Sentiment Analysis) and **Software Engineering Rigor** (Golden Master Tests) to identify mispriced options with high probability.

### Core Philosophy
1.  **Don't Predict Direction**: Predicting where Nifty goes is hard (50/50).
2.  **Predict Volatility**: Predicting that Nifty will move *somewhere* is easier.
3.  **Price the Wings**: Retail traders overpay for deep OTM Lotteries. We sell them using scientifically calibrated surfaces.
4.  **Protect the Downside**: Use Jump-Diffusion to price the specific risk of a 10% lower circuit crash.

---

## 💎 Why OMEGA v5? (The Institutional Edge)

Most retail traders lose money because they buy options based on directional hunches ("Nifty will go up") without knowing the **Fair Value**. They overpay for volatility and get crushed by Theta decay.

OMEGA solves this by answering three critical questions that retail platforms don't ask:

1.  **What is the True Price?**
    *   *Standard*: Black-Scholes (assumes constant volatility).
    *   *OMEGA*: Heston Stochastic Volatility + Merton Jump Diffusion + SVI Surface + VRP Filter.

2.  **Is it Cheap or Expensive?**
    *   *Standard*: "IV is high."
    *   *OMEGA*: "Deep OTM Calls are 2-sigma cheap relative to the realized Variance Risk Premium of the last 10 days."

3.  **What is the Probability of Profit?**
    *   *Standard*: Delta (approximate).
    *   *OMEGA*: 50,000 Monte Carlo simulations using Sobolev Quasi-Random sequences to map the exact payoff distribution.

---

## 🏗️ System Architecture

The codebase is organized into a hierarchical 4-layer stack. Each layer abstracts complexity from the one above it.

### Layer 4: The Interface (`opmAI_app.py`)
The Streamlit-based dashboard. It acts as the "Trader's Desk".
-   **Data Ingestion**: Upstox API (Live), Angel One API (Historical).
-   **State Management**: `MarketState` class that acts as the "Single Source of Truth".
-   **Visualization**: Plotly charts for Volatility Surfaces, Payoff Diagrams, and Greeks.
-   **Execution Logic**: Paper trading simulation and Strategy Builder.

### Layer 3: The Intelligence (`omega_model.py`)
The decision-making layer ("The Brain").
-   **ML Correction**: A Gradient Boosting Regressor (XGBoost/LightGBM) that predicts the *error* of the mathematical model based on 50+ features.
-   **Anomaly Detection**: An Isolation Forest that filters out "fake" mispricings caused by stale data or illiquidity.
-   **Sentiment Analysis**: Integration with Gemini/Perplexity to quantify "Market Mood" from news.

### Layer 2: The Pricing Core (`nirv_model.py`)
The "NIRV" (Nifty Intelligent Regime-Volatility) model ("The Heart").
-   **Regime Detection**: A Hidden Markov Model (HMM) that classifies the market into 4 regimes (Bull-Low Vol, Bear-High Vol, etc.).
-   **SVI Surface**: Stochastic Volatility Inspired parameterization to model the smile/skew accurately.
-   **Monte Carlo Engine**: A Sobol-sequence accelerated-MC pricer handling Heston dynamics + Merton Jumps.
-   **VRP State Filter**: (New in v5) Separates "Fear" from "Movement".

### Layer 1: The Quant Engine (`quant_engine.py`)
The mathematical toolbox ("The Tools").
-   **SABR**: Dynamic calibration algorithms.
-   **HestonCOS**: Fourier-Cosine expansion for semi-analytical pricing (millisecond latency).
-   **GARCH**: Volatility forecasting tools.
-   **Optimization**: Differential Evolution and Least Squares solvers for calibration.
-   **Feature Flags**: `omega_features.py` manages the activation of new experimental mathematics.

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

## ⚡ The v5 "Alpha" Features (Phase 4 & 5)

Version 5.0 introduces "Paradigm Shift" upgrades that address the root causes of model failure in live trading.

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
A new CLI tool in v5 for rigorously comparing the current model against "Golden Master" snapshots.
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

In v5, we moved away from "looks good to me" testing to **Regression Testing**.

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

---

**© 2026 OMEGA Quantitative Research**
*Built with ❤️ for the Indian Algo Trading Community*
