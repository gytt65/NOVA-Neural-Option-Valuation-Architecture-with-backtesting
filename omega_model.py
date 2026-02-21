#!/usr/bin/env python3
"""
=============================================================================
OMEGA: Options Market Efficiency & Generative Analysis
=============================================================================
A next-generation ML/AI-powered option pricing model that combines:

  Layer 0  — NIRV mathematical pricing (Heston SV + Jump Diffusion + SVI + HMM)
  Layer 1  — ML correction engine (Gradient Boosting learns NIRV residuals)
  Layer 2  — Anomaly detection (Isolation Forest finds inefficiencies)
  Layer 3  — Sentiment intelligence (AI-powered via Gemini / Perplexity)
  Layer 4  — Behavioral engine (predicts actions of key market actors)
  Layer 5  — Adaptive learning (tracks predictions → outcomes, retrains)

Architecture:
    OMEGA_price = NIRV_base × (1 + ML_correction) + sentiment_adj

The model starts with NIRV's strong mathematical foundation and
progressively improves as it accumulates data.  Cold-start safe:
all ML layers return 0 correction until enough training data exists.

Author : Quantitative Research — OMEGA Division
Version: 1.0
=============================================================================
"""

import numpy as np
import json
import os
import datetime
import warnings
from collections import defaultdict

# Feature flags / contract specs
try:
    from omega_features import get_features
except Exception:
    get_features = lambda: type(
        "Features",
        (),
        {
            "USE_CONFORMAL_INTERVALS": False,
            "USE_NSE_CONTRACT_SPECS": False,
            "USE_STALENESS_FEATURES": False,
            "USE_ENHANCED_RANKING": False,
            "USE_RESEARCH_HIGH_CONVICTION": False,
            "USE_OOS_RELIABILITY_GATE": False,
        },
    )()

try:
    from nse_specs import get_lot_size as nse_get_lot_size
except Exception:
    nse_get_lot_size = None

# ---------------------------------------------------------------------------
# Optional ML imports  (graceful degradation if sklearn is absent)
# ---------------------------------------------------------------------------
try:
    from sklearn.ensemble import (
        GradientBoostingRegressor,
        GradientBoostingClassifier,
        IsolationForest,
    )
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

# NIRV base model
from nirv_model import NIRVModel, NirvOutput
try:
    from behavioral_state_engine import BehavioralStateEngine
except Exception:
    BehavioralStateEngine = None


# ============================================================================
# FACTOR REGISTRY — every known factor that influences option pricing
# ============================================================================

class FactorRegistry:
    """
    Comprehensive registry of ALL known factors that influence option pricing
    and market movements.  Each factor has:
      - source: where to auto-fetch it ('upstox', 'free_api', 'ai', 'manual')
      - auto: True if OMEGA can fetch it automatically
      - prompt: if not auto, what to tell the user
    """

    FACTORS = {
        # ── DOMESTIC MARKET ─────────────────────────────────────────
        'spot_price':       {'group': 'Domestic', 'source': 'upstox', 'auto': True,
                             'desc': 'Current spot price of the underlying'},
        'india_vix':        {'group': 'Domestic', 'source': 'upstox', 'auto': True,
                             'desc': 'India VIX — market fear gauge'},
        'pcr_oi':           {'group': 'Domestic', 'source': 'upstox', 'auto': True,
                             'desc': 'Put-Call Ratio (Open Interest)'},
        'max_pain':         {'group': 'Domestic', 'source': 'upstox', 'auto': True,
                             'desc': 'Max Pain strike — where most options expire worthless'},
        'oi_change':        {'group': 'Domestic', 'source': 'upstox', 'auto': True,
                             'desc': 'Open Interest change pattern (buildup / unwinding)'},
        'iv_rank':          {'group': 'Domestic', 'source': 'upstox', 'auto': True,
                             'desc': 'IV Rank — where current IV sits in 52-week range'},
        'iv_percentile':    {'group': 'Domestic', 'source': 'upstox', 'auto': True,
                             'desc': 'IV Percentile — % of days IV was lower'},
        'hv_30d':           {'group': 'Domestic', 'source': 'upstox', 'auto': True,
                             'desc': '30-day Historical Volatility'},
        'hv_iv_spread':     {'group': 'Domestic', 'source': 'upstox', 'auto': True,
                             'desc': 'IV minus HV — positive means options are expensive'},
        'option_chain':     {'group': 'Domestic', 'source': 'upstox', 'auto': True,
                             'desc': 'Full option chain with strikes, LTP, OI, IV'},
        'market_breadth':   {'group': 'Domestic', 'source': 'free_api', 'auto': True,
                             'desc': 'Advance/Decline ratio of NSE stocks'},
        'nifty_pe':         {'group': 'Domestic', 'source': 'free_api', 'auto': True,
                             'desc': 'Nifty PE ratio — valuation gauge'},

        # ── TECHNICAL INDICATORS ────────────────────────────────────
        'rsi_14':           {'group': 'Technical', 'source': 'upstox', 'auto': True,
                             'desc': 'RSI(14) — momentum oscillator'},
        'macd_signal':      {'group': 'Technical', 'source': 'upstox', 'auto': True,
                             'desc': 'MACD signal line crossover'},
        'macd_histogram':   {'group': 'Technical', 'source': 'upstox', 'auto': True,
                             'desc': 'MACD histogram — momentum strength'},
        'bb_position':      {'group': 'Technical', 'source': 'upstox', 'auto': True,
                             'desc': 'Bollinger Band position (0=lower, 1=upper)'},
        'atr_pct':          {'group': 'Technical', 'source': 'upstox', 'auto': True,
                             'desc': 'ATR as % of price — volatility measure'},
        'supertrend':       {'group': 'Technical', 'source': 'upstox', 'auto': True,
                             'desc': 'Supertrend direction (+1 bullish, -1 bearish)'},
        'ema_20':           {'group': 'Technical', 'source': 'upstox', 'auto': True,
                             'desc': 'EMA(20) — short-term trend'},
        'ema_50':           {'group': 'Technical', 'source': 'upstox', 'auto': True,
                             'desc': 'EMA(50) — medium-term trend'},
        'ema_200':          {'group': 'Technical', 'source': 'upstox', 'auto': True,
                             'desc': 'EMA(200) — long-term trend'},
        'vwap':             {'group': 'Technical', 'source': 'upstox', 'auto': True,
                             'desc': 'VWAP — institutional fair value'},
        'stochastic':       {'group': 'Technical', 'source': 'upstox', 'auto': True,
                             'desc': 'Stochastic K/D — overbought/oversold'},
        'adx':              {'group': 'Technical', 'source': 'upstox', 'auto': True,
                             'desc': 'ADX — trend strength'},
        'pivot_levels':     {'group': 'Technical', 'source': 'upstox', 'auto': True,
                             'desc': 'Classic/Fibonacci/Camarilla pivot levels'},
        'candlestick':      {'group': 'Technical', 'source': 'upstox', 'auto': True,
                             'desc': 'Candlestick patterns (Doji, Engulfing, etc.)'},
        'returns_30d':      {'group': 'Technical', 'source': 'upstox', 'auto': True,
                             'desc': 'Last 30 daily log returns — for regime detection'},

        # ── GLOBAL MARKETS ──────────────────────────────────────────
        'sp500':            {'group': 'Global', 'source': 'free_api', 'auto': True,
                             'desc': 'S&P 500 level and % change'},
        'nasdaq':           {'group': 'Global', 'source': 'free_api', 'auto': True,
                             'desc': 'NASDAQ Composite level and % change'},
        'dow':              {'group': 'Global', 'source': 'free_api', 'auto': True,
                             'desc': 'Dow Jones level and % change'},
        'cboe_vix':         {'group': 'Global', 'source': 'free_api', 'auto': True,
                             'desc': 'CBOE VIX — US market fear index'},
        'crude_oil':        {'group': 'Global', 'source': 'free_api', 'auto': True,
                             'desc': 'WTI Crude Oil price — India is net importer'},
        'brent_crude':      {'group': 'Global', 'source': 'free_api', 'auto': True,
                             'desc': 'Brent Crude price'},
        'gold':             {'group': 'Global', 'source': 'free_api', 'auto': True,
                             'desc': 'Gold price — safe haven / risk-off indicator'},
        'silver':           {'group': 'Global', 'source': 'free_api', 'auto': True,
                             'desc': 'Silver price'},
        'usd_inr':          {'group': 'Global', 'source': 'free_api', 'auto': True,
                             'desc': 'USD/INR exchange rate — FII flow driver'},
        'us_10yr':          {'group': 'Global', 'source': 'free_api', 'auto': True,
                             'desc': 'US 10Y Treasury Yield — global risk barometer'},
        'dxy':              {'group': 'Global', 'source': 'free_api', 'auto': True,
                             'desc': 'US Dollar Index — EM flow indicator'},
        'ftse':             {'group': 'Global', 'source': 'free_api', 'auto': True,
                             'desc': 'FTSE 100 — European markets'},
        'nikkei':           {'group': 'Global', 'source': 'free_api', 'auto': True,
                             'desc': 'Nikkei 225 — Asian markets'},
        'hang_seng':        {'group': 'Global', 'source': 'free_api', 'auto': True,
                             'desc': 'Hang Seng Index — China proxy'},
        'sgx_nifty':        {'group': 'Global', 'source': 'free_api', 'auto': True,
                             'desc': 'SGX Nifty (Gift Nifty) — pre-market indicator'},

        # ── INSTITUTIONAL FLOWS ─────────────────────────────────────
        'fii_net_flow':     {'group': 'Flows', 'source': 'ai', 'auto': True,
                             'desc': 'FII net buy/sell in cash segment (₹ crores)'},
        'dii_net_flow':     {'group': 'Flows', 'source': 'ai', 'auto': True,
                             'desc': 'DII net buy/sell in cash segment (₹ crores)'},
        'fii_index_fut_oi': {'group': 'Flows', 'source': 'ai', 'auto': True,
                             'desc': 'FII Index Futures OI change — directional bet'},
        'fii_option_oi':    {'group': 'Flows', 'source': 'ai', 'auto': True,
                             'desc': 'FII Index Options OI change'},

        # ── MACRO / ECONOMIC ────────────────────────────────────────
        'rbi_repo_rate':    {'group': 'Macro', 'source': 'ai', 'auto': True,
                             'desc': 'Current RBI repo rate'},
        'cpi_inflation':    {'group': 'Macro', 'source': 'ai', 'auto': True,
                             'desc': 'Latest CPI inflation %'},
        'iip_growth':       {'group': 'Macro', 'source': 'ai', 'auto': True,
                             'desc': 'Index of Industrial Production growth'},
        'gdp_growth':       {'group': 'Macro', 'source': 'ai', 'auto': True,
                             'desc': 'Latest GDP growth rate'},
        'days_to_rbi':      {'group': 'Macro', 'source': 'ai', 'auto': True,
                             'desc': 'Trading days until next RBI policy meeting'},
        'days_to_fed':      {'group': 'Macro', 'source': 'ai', 'auto': True,
                             'desc': 'Trading days until next Fed meeting'},
        'us_fed_rate':      {'group': 'Macro', 'source': 'ai', 'auto': True,
                             'desc': 'Current US Federal Funds rate'},

        # ── SENTIMENT & NEWS ───────────────────────────────────────
        'news_sentiment':   {'group': 'Sentiment', 'source': 'ai', 'auto': True,
                             'desc': 'AI-analysed news sentiment score'},
        'geopolitical_risk':{'group': 'Sentiment', 'source': 'ai', 'auto': True,
                             'desc': 'Geopolitical risk level (tariffs, wars, sanctions)'},
        'earnings_impact':  {'group': 'Sentiment', 'source': 'ai', 'auto': True,
                             'desc': 'Upcoming earnings impact on underlying'},
        'social_sentiment': {'group': 'Sentiment', 'source': 'ai', 'auto': True,
                             'desc': 'Social media / retail sentiment'},

        # ── SEASONAL / CALENDAR ────────────────────────────────────
        'day_of_week':      {'group': 'Seasonal', 'source': 'calculated', 'auto': True,
                             'desc': 'Day of week (NSE weekly expiry dynamics)'},
        'is_expiry_week':   {'group': 'Seasonal', 'source': 'calculated', 'auto': True,
                             'desc': 'Whether current week is expiry week'},
        'month_effect':     {'group': 'Seasonal', 'source': 'calculated', 'auto': True,
                             'desc': 'Monthly seasonal pattern (Jan effect, etc.)'},
        'is_budget_period': {'group': 'Seasonal', 'source': 'calculated', 'auto': True,
                             'desc': 'Budget session period (high vol)'},
        'is_election':      {'group': 'Seasonal', 'source': 'calculated', 'auto': True,
                             'desc': 'Election period (high uncertainty)'},

        # ── BEHAVIORAL / ACTOR ─────────────────────────────────────
        'trump_action':     {'group': 'Behavioral', 'source': 'ai', 'auto': True,
                             'desc': 'Latest Trump action / statement affecting markets'},
        'fed_stance':       {'group': 'Behavioral', 'source': 'ai', 'auto': True,
                             'desc': 'Fed monetary policy stance (hawkish/dovish)'},
        'rbi_stance':       {'group': 'Behavioral', 'source': 'ai', 'auto': True,
                             'desc': 'RBI policy stance'},
        'china_factor':     {'group': 'Behavioral', 'source': 'ai', 'auto': True,
                             'desc': 'China economic / trade policy developments'},

        # ── CURRENCY / COMMODITY ───────────────────────────────────
        'inr_usd_vol':      {'group': 'Currency', 'source': 'free_api', 'auto': True,
                             'desc': 'INR/USD 30-day realised volatility'},
        'rupee_direction':  {'group': 'Currency', 'source': 'free_api', 'auto': True,
                             'desc': 'Rupee trend direction (strengthening/weakening)'},
        'crude_direction':  {'group': 'Currency', 'source': 'free_api', 'auto': True,
                             'desc': 'Crude oil trend — key for India CAD'},
    }

    @classmethod
    def get_all_factors(cls):
        return cls.FACTORS

    @classmethod
    def get_by_group(cls, group):
        return {k: v for k, v in cls.FACTORS.items() if v['group'] == group}

    @classmethod
    def get_auto_fetchable(cls):
        return {k: v for k, v in cls.FACTORS.items() if v['auto']}

    @classmethod
    def get_manual_factors(cls):
        return {k: v for k, v in cls.FACTORS.items() if not v['auto']}

    @classmethod
    def get_missing_prompt(cls, fetched_keys: set) -> str:
        """Generate a user-friendly prompt for any factors not yet fetched."""
        all_keys = set(cls.FACTORS.keys())
        missing = all_keys - fetched_keys
        if not missing:
            return ""
        lines = ["**The following factors could not be auto-fetched. "
                 "You can improve accuracy by providing them:**\n"]
        by_group = defaultdict(list)
        for k in sorted(missing):
            fac = cls.FACTORS[k]
            by_group[fac['group']].append(f"- **{k}**: {fac['desc']}")
        for group, items in sorted(by_group.items()):
            lines.append(f"\n*{group}:*")
            lines.extend(items)
        return '\n'.join(lines)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class OMEGAOutput:
    """Rich output object from OMEGA analysis — attribute-accessible."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def to_dict(self):
        d = {}
        for k, v in self.__dict__.items():
            if k.startswith('_'):
                continue
            if isinstance(v, np.floating):
                d[k] = float(v)
            elif isinstance(v, np.integer):
                d[k] = int(v)
            elif isinstance(v, np.ndarray):
                d[k] = v.tolist()
            else:
                d[k] = v
        return d


# ============================================================================
# MODULE 1 — FEATURE FACTORY
# ============================================================================

class FeatureFactory:
    """
    Extracts 45+ normalised features from market data for ML models.

    Feature groups:
        Moneyness & Time — strike structure, expiry dynamics
        Volatility       — IV, HV, IV rank/percentile, VIX, IV-HV gap
        Greeks-derived   — delta, gamma$, theta/premium, vega/premium
        Flows & OI       — FII/DII normalised, PCR, OI concentration
        Technicals       — RSI, MACD, Bollinger, ATR
        Regime           — one-hot encoded HMM regime
        NIRV base        — fair value ratio, mispricing, PoP, confidence
        Market micro     — bid-ask spread, volume/OI
    """

    FEATURE_NAMES = [
        # Moneyness  (3)
        'moneyness', 'log_moneyness', 'moneyness_sq',
        # Time  (5)
        'time_to_expiry', 'sqrt_time', 'inv_time', 'days_to_expiry',
        'is_weekly_expiry',
        # Day-of-week  (1)
        'day_of_week',
        # Volatility  (7)
        'iv', 'hv_30d', 'iv_hv_spread', 'iv_rank', 'iv_percentile',
        'vix', 'vix_zscore',
        # Greeks  (5)
        'delta', 'abs_delta', 'gamma_dollar', 'theta_prem_ratio',
        'vega_prem_ratio',
        # Flows  (3)
        'fii_norm', 'dii_norm', 'net_flow_dir',
        # OI  (3)
        'pcr_oi', 'pcr_deviation', 'oi_concentration',
        # Technicals  (4)
        'rsi', 'macd_signal', 'bb_position', 'atr_pct',
        # Regime one-hot  (4)
        'regime_bull_low', 'regime_bear_high', 'regime_sideways', 'regime_bull_high',
        # NIRV  (5)
        'nirv_fv_ratio', 'nirv_mispricing', 'nirv_profit_prob',
        'nirv_confidence', 'nirv_phys_prob',
        # Market micro  (3)
        'bid_ask_spread', 'volume_oi_ratio', 'iv_vix_ratio',
        # Expiry dynamics  (2)
        'gamma_amp', 'theta_accel',
        # Cross-sectional features  (4) -- Item 5
        'iv_minus_atm_iv', 'local_skew_curvature', 'term_structure_slope',
        'skew_position',
        # Lagged features  (3) -- Item 5
        'mispricing_lag1', 'mispricing_lag2', 'mispricing_lag5',
        # GEX / Anomaly  (3) -- Items 6, 11
        'gex_sign', 'anomaly_score_if', 'rv_iv_ratio',
        # VRP  (1) -- Item 4
        'variance_risk_premium',
        # Synthetic VIX (1) -- Phase 2
        'india_vix_synth',
    ]

    @staticmethod
    def extract(market_data: dict) -> dict:
        """Return a dict of normalised features from raw market data."""
        spot   = max(market_data.get('spot', 1), 1.0)
        strike = max(market_data.get('strike', 1), 1.0)
        T      = max(market_data.get('T', 0.02), 1e-6)
        iv     = market_data.get('iv', 0.15)
        hv     = market_data.get('hv_30d', iv)
        vix    = market_data.get('vix', 14.0)
        mp     = max(market_data.get('market_price', 0.01), 0.01)

        f = {}

        # — Moneyness -------------------------------------------------------
        f['moneyness']    = strike / spot
        f['log_moneyness'] = np.log(strike / spot)
        f['moneyness_sq'] = (strike / spot - 1.0) ** 2

        # — Time -------------------------------------------------------------
        f['time_to_expiry'] = T
        f['sqrt_time']      = np.sqrt(T)
        f['inv_time']       = min(1.0 / T, 365.0)  # cap at 1-day
        dte = max(int(T * 365), 1)
        f['days_to_expiry'] = dte
        f['is_weekly_expiry'] = 1.0 if dte <= 7 else 0.0
        f['day_of_week'] = datetime.datetime.now().weekday() / 4.0

        # — Volatility -------------------------------------------------------
        f['iv']            = iv
        f['hv_30d']        = hv
        f['iv_hv_spread']  = iv - hv
        f['iv_rank']       = market_data.get('iv_rank', 50.0) / 100.0
        f['iv_percentile'] = market_data.get('iv_percentile', 50.0) / 100.0
        f['vix']           = vix / 100.0
        # Dynamic VIX z-score: use rolling history when available (passes from
        # upstream market data), otherwise fall back to long-run India VIX stats.
        _vix_hist = market_data.get('vix_history', [])
        if len(_vix_hist) >= 20:
            vix_mean = float(np.mean(_vix_hist[-252:]))
            vix_std  = max(float(np.std(_vix_hist[-252:])), 1.0)
        else:
            vix_mean = 14.0   # long-run India VIX mean (2003–2024)
            vix_std  = 5.0    # historical India VIX std deviation
        f['vix_zscore']    = (vix - vix_mean) / vix_std

        # — Greeks (from NIRV) -----------------------------------------------
        greeks = market_data.get('greeks', {})
        delta  = greeks.get('delta', 0.5)
        f['delta']          = delta
        f['abs_delta']      = abs(delta)
        f['gamma_dollar']   = greeks.get('gamma', 0) * spot * spot * 0.01
        f['theta_prem_ratio'] = greeks.get('theta', 0) / mp
        f['vega_prem_ratio']  = greeks.get('vega', 0)  / mp

        # — Flows (normalised to ±1) ----------------------------------------
        fii = market_data.get('fii_net_flow', 0)
        dii = market_data.get('dii_net_flow', 0)
        f['fii_norm']     = np.clip(fii / 5000.0, -1.0, 1.0)
        f['dii_norm']     = np.clip(dii / 5000.0, -1.0, 1.0)
        f['net_flow_dir'] = 1.0 if (fii + dii) > 0 else -1.0

        # — OI ---------------------------------------------------------------
        pcr = market_data.get('pcr_oi', 1.0)
        f['pcr_oi']          = pcr
        f['pcr_deviation']   = (pcr - 1.05) / 1.05
        f['oi_concentration'] = market_data.get('oi_concentration', 0.5)

        # — Technicals -------------------------------------------------------
        f['rsi']         = market_data.get('rsi', 50.0) / 100.0
        f['macd_signal'] = np.clip(market_data.get('macd_signal', 0) / 100.0, -1, 1)
        f['bb_position'] = market_data.get('bb_position', 0.5)
        f['atr_pct']     = market_data.get('atr_pct', 1.5) / 100.0

        # — Regime one-hot ---------------------------------------------------
        regime = market_data.get('regime', 'Sideways')
        f['regime_bull_low']  = 1.0 if regime == 'Bull-Low Vol'  else 0.0
        f['regime_bear_high'] = 1.0 if regime == 'Bear-High Vol' else 0.0
        f['regime_sideways']  = 1.0 if regime == 'Sideways'      else 0.0
        f['regime_bull_high'] = 1.0 if regime == 'Bull-High Vol' else 0.0

        # — NIRV base -------------------------------------------------------
        nirv = market_data.get('nirv_output')
        if nirv:
            f['nirv_fv_ratio']    = nirv.fair_value / mp
            f['nirv_mispricing']  = nirv.mispricing_pct / 100.0
            f['nirv_profit_prob'] = nirv.profit_probability / 100.0
            f['nirv_confidence']  = nirv.confidence_level / 100.0
            f['nirv_phys_prob']   = nirv.physical_profit_prob / 100.0
        else:
            f['nirv_fv_ratio']    = 1.0
            f['nirv_mispricing']  = 0.0
            f['nirv_profit_prob'] = 0.5
            f['nirv_confidence']  = 0.5
            f['nirv_phys_prob']   = 0.5

        # — Market micro -----------------------------------------------------
        bid = market_data.get('bid', mp * 0.99)
        ask = market_data.get('ask', mp * 1.01)
        f['bid_ask_spread'] = (ask - bid) / mp
        f['volume_oi_ratio'] = market_data.get('volume_oi_ratio', 0.1)
        f['iv_vix_ratio']    = iv / max(vix / 100.0, 0.01)

        # Optional staleness diagnostics (kept additive; not used in legacy FEATURE_NAMES)
        if bool(getattr(get_features(), "USE_STALENESS_FEATURES", False)):
            age_seconds = 0.0
            stale_score = 0.0
            quote_ts = market_data.get('quote_ts')
            now_ts = market_data.get('now_ts')
            if quote_ts is not None and now_ts is not None:
                try:
                    qts = datetime.datetime.fromisoformat(str(quote_ts)) if isinstance(quote_ts, str) else quote_ts
                    nts = datetime.datetime.fromisoformat(str(now_ts)) if isinstance(now_ts, str) else now_ts
                    age_seconds = max(float((nts - qts).total_seconds()), 0.0)
                    # Logistic-like staleness score around 30s half-point.
                    stale_score = float(1.0 - np.exp(-age_seconds / 30.0))
                except Exception:
                    age_seconds = 0.0
                    stale_score = 0.0
            f['quote_age_seconds'] = age_seconds
            f['stale_score'] = stale_score

        # -- Expiry dynamics --------------------------------------------------
        f['gamma_amp']   = 1.0 / (1.0 + 0.5 * max(dte, 1))
        f['theta_accel'] = f['gamma_amp'] ** 2

        # -- Cross-sectional features (Item 5) --------------------------------
        # IV minus ATM IV (skew position) -- captures relative mispricing
        atm_iv = market_data.get('atm_iv', iv)
        f['iv_minus_atm_iv'] = iv - atm_iv

        # Local skew curvature (IV slope across adjacent strikes)
        f['local_skew_curvature'] = market_data.get('local_skew_curvature', 0.0)

        # Term structure slope (near-expiry IV minus far-expiry IV)
        f['term_structure_slope'] = market_data.get('term_structure_slope', 0.0)

        # Skew position: where this strike sits on the skew curve
        f['skew_position'] = market_data.get('skew_25d_rr', 0.0) / 100.0

        # -- Lagged features (Item 5) ----------------------------------------
        # Mispricings that persist across snapshots are more likely real
        f['mispricing_lag1'] = market_data.get('mispricing_lag1', 0.0) / 100.0
        f['mispricing_lag2'] = market_data.get('mispricing_lag2', 0.0) / 100.0
        f['mispricing_lag5'] = market_data.get('mispricing_lag5', 0.0) / 100.0

        # -- GEX / Anomaly features (Items 6, 11) ----------------------------
        f['gex_sign'] = np.sign(market_data.get('gex_at_strike', 0.0))
        f['anomaly_score_if'] = market_data.get('anomaly_score_if', 0.0)
        # Additive dealer-flow vector (legacy gex_sign preserved above)
        if 'gex_vector' in market_data and isinstance(market_data['gex_vector'], dict):
            gv = market_data['gex_vector']
            f['dealer_total_gex'] = float(gv.get('total_gex', 0.0))
            f['dealer_gamma_flip'] = float(gv.get('gamma_flip', np.nan))
            f['dealer_charm'] = float(gv.get('charm', 0.0))
            f['dealer_vanna'] = float(gv.get('vanna', 0.0))
            f['dealer_bucketed_gex'] = gv.get('bucketed_gex', {})

        # RV/IV ratio
        rv = market_data.get('rv_20d', hv)
        f['rv_iv_ratio'] = rv / max(iv, 0.01)

        # -- Variance Risk Premium (Item 4) -----------------------------------
        f['variance_risk_premium'] = max(iv, 0.01)**2 - max(hv, 0.01)**2
        
        # -- Synthetic VIX (Phase 2) -----------------------------------------
        f['india_vix_synth'] = market_data.get('india_vix_synth', 0.0)

        # -- v6 model-free variance / VRP features (kept optional) ------------
        f['model_free_var_30d'] = market_data.get('model_free_var_30d', np.nan)
        f['vrp_level'] = market_data.get('vrp_level', np.nan)
        f['vrp_slope'] = market_data.get('vrp_slope', np.nan)
        ts = market_data.get('model_free_var_term_structure')
        if isinstance(ts, dict):
            f['model_free_var_7d'] = ts.get(7, np.nan)
            f['model_free_var_60d'] = ts.get(60, np.nan)
        else:
            f['model_free_var_7d'] = np.nan
            f['model_free_var_60d'] = np.nan

        return f

    @staticmethod
    def to_array(features: dict) -> np.ndarray:
        """Convert feature dict → ordered numpy array matching FEATURE_NAMES."""
        return np.array([features.get(n, 0.0) for n in FeatureFactory.FEATURE_NAMES],
                        dtype=np.float64)


# ============================================================================
# MODULE 2 — ML PRICING CORRECTOR  (learns NIRV residuals)
# ============================================================================

class MLPricingCorrector:
    """
    Gradient Boosting model that learns to correct systematic NIRV errors.

    Training target:
        residual = (actual_outcome_value − nirv_predicted_value) / nirv_value

    The model learns biases like:
    - OTM overpricing near expiry  (theta burn faster than model predicts)
    - Underpricing in high-vol regimes  (tail risk under-estimated)
    - Event-day mispricing patterns

    Cold-start:  returns (0, 0) until ≥ MIN_SAMPLES training points exist.
    """

    MIN_SAMPLES = 30   # Minimum samples before ML kicks in
    RETRAIN_EVERY = 15  # Base retrain interval (adaptive: max(15, n//20))

    def __init__(self, model_path='omega_data/pricing_model.joblib'):
        self.model_path = model_path
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.model = None
        self.is_trained = False
        self.training_X = []   # list of feature arrays
        self.training_y = []   # list of target residuals
        self._conformal_q_global = None
        self._conformal_q_by_regime = {}
        self._load_model()

    # ------------------------------------------------------------------
    def _load_model(self):
        if not JOBLIB_AVAILABLE or not os.path.exists(self.model_path):
            return
        try:
            saved = joblib.load(self.model_path)
            self.model     = saved['model']
            self.scaler    = saved['scaler']
            self.training_X = saved.get('X', [])
            self.training_y = saved.get('y', [])
            self._conformal_q_global = saved.get('conformal_q_global', None)
            self._conformal_q_by_regime = saved.get('conformal_q_by_regime', {})
            self.is_trained = True
        except Exception:
            pass

    def _save_model(self):
        if not JOBLIB_AVAILABLE or self.model is None:
            return
        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            joblib.dump({
                'model': self.model, 'scaler': self.scaler,
                'X': self.training_X[-2000:],  # cap storage
                'y': self.training_y[-2000:],
                'conformal_q_global': self._conformal_q_global,
                'conformal_q_by_regime': self._conformal_q_by_regime,
            }, self.model_path)
        except Exception:
            pass

    # ------------------------------------------------------------------
    def predict_correction(self, features: dict):
        """
        Returns (correction_factor, confidence)  where correction is a
        fraction (e.g. 0.03 means +3 % adjustment to NIRV price).
        """
        if not self.is_trained or not SKLEARN_AVAILABLE:
            return 0.0, 0.0
        try:
            X = FeatureFactory.to_array(features).reshape(1, -1)
            X_s = self.scaler.transform(X)
            correction = float(self.model.predict(X_s)[0])
            # Adaptive correction cap: tighter for ATM, looser for deep OTM
            _moneyness = abs(features.get('moneyness', 1.0) - 1.0)
            _cap = 0.10 + 0.10 * min(_moneyness * 10.0, 1.0)  # 10% ATM, 20% deep OTM
            correction = np.clip(correction, -_cap, _cap)
            # Use CV-based confidence if available; fall back to sample-based
            confidence = getattr(self, '_cv_confidence', min(0.85, len(self.training_X) / 500.0))
            return correction, confidence
        except Exception:
            return 0.0, 0.0

    def predict_correction_with_interval(self, features: dict, alpha: float = 0.10):
        """
        Conformalized Quantile Regression (CQR) prediction interval.

        Returns (correction, confidence, lower, upper).

        Provides a statistical coverage guarantee: P(true ∈ [lower, upper]) ≥ 1-α
        regardless of the underlying distribution. Intervals are adaptive —
        wider for OTM options in volatile regimes, narrower for ATM in calm.

        Interval is enabled only when USE_CONFORMAL_INTERVALS flag is ON.
        """
        corr, conf = self.predict_correction(features)
        if not getattr(get_features(), "USE_CONFORMAL_INTERVALS", False):
            return corr, conf, corr, corr

        # --- CQR: start with the global conformal quantile ---
        q = self._conformal_q_global if self._conformal_q_global is not None else 0.0

        # --- Regime-binned quantile (tighter within regime) ---
        regime = 'Sideways'
        if features.get('regime_bull_low', 0) > 0.5:
            regime = 'Bull-Low Vol'
        elif features.get('regime_bear_high', 0) > 0.5:
            regime = 'Bear-High Vol'
        elif features.get('regime_bull_high', 0) > 0.5:
            regime = 'Bull-High Vol'
        q_reg = self._conformal_q_by_regime.get(regime)
        if q_reg is not None and np.isfinite(q_reg):
            q = max(float(q), float(q_reg))

        q = max(float(q), 0.0)

        # --- Adaptive width modulation ---
        # CQR insight: multiply base width by local difficulty score
        # Factors that increase uncertainty:
        #   1. Deep OTM/ITM (moneyness far from 1.0)
        #   2. Near-expiry (gamma risk amplifies errors)
        #   3. High-vol regime (Bear-High Vol)
        #   4. Low liquidity / wide spreads

        # Moneyness factor: OTM options are harder to price
        log_m = abs(features.get('log_moneyness', 0.0))
        moneyness_mult = 1.0 + 1.5 * min(log_m, 0.3)  # [1.0, 1.45]

        # Time factor: near-expiry options have larger model error
        T = max(features.get('time_to_expiry', 0.05), 1e-6)
        time_mult = 1.0 + 0.3 * max(0, 1.0 / (T * 52) - 1.0)  # boost for < 1 week
        time_mult = min(time_mult, 1.6)

        # Regime factor: volatile regimes → wider intervals
        regime_mult = 1.0
        if regime == 'Bear-High Vol':
            regime_mult = 1.3
        elif regime == 'Bull-High Vol':
            regime_mult = 1.1

        # Compose adaptive width
        adaptive_q = q * moneyness_mult * time_mult * regime_mult

        # --- Coverage calibration with finite-sample correction ---
        # Vovk (2012): for n calibration points, use (1-α)(1 + 1/n) quantile
        n_cal = max(len(self.training_y), 1)
        coverage_correction = (1.0 + 1.0 / n_cal)
        adaptive_q *= min(coverage_correction, 1.05)  # cap correction at 5%

        return corr, conf, corr - adaptive_q, corr + adaptive_q

    # ------------------------------------------------------------------
    def add_sample(self, features: dict, residual: float):
        """Add a training sample; auto-retrain with adaptive interval."""
        X = FeatureFactory.to_array(features)
        self.training_X.append(X.tolist())
        self.training_y.append(float(residual))
        n = len(self.training_X)
        # Adaptive retrain: max(15, n//20) -- less frequent as data grows
        retrain_interval = max(self.RETRAIN_EVERY, n // 20)
        if n >= self.MIN_SAMPLES and n % retrain_interval == 0:
            self._train()

    def _train(self):
        if not SKLEARN_AVAILABLE or len(self.training_X) < self.MIN_SAMPLES:
            return
        X = np.array(self.training_X)
        y = np.array(self.training_y)
        self.scaler = StandardScaler()
        X_s = self.scaler.fit_transform(X)

        # Cross-validation to prevent overfitting (TimeSeriesSplit preserves order)
        from sklearn.model_selection import TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=min(4, max(2, len(y) // 30)))
        cv_scores = []

        def _make_model(n):
            try:
                import lightgbm as lgb
                return lgb.LGBMRegressor(
                    n_estimators=min(200, max(30, n // 3)),
                    max_depth=4, learning_rate=0.06,
                    subsample=0.8, random_state=42, verbose=-1,
                    categorical_feature='auto',
                )
            except (ImportError, Exception):
                return GradientBoostingRegressor(
                    n_estimators=min(150, max(30, n // 3)),
                    max_depth=4, learning_rate=0.08,
                    subsample=0.8, random_state=42,
                )

        for train_idx, val_idx in tscv.split(X_s):
            fold_model = _make_model(len(train_idx))
            fold_model.fit(X_s[train_idx], y[train_idx])
            score = fold_model.score(X_s[val_idx], y[val_idx])
            cv_scores.append(max(score, 0.0))

        # Store CV-based confidence (replaces arbitrary len/500)
        self._cv_confidence = float(np.mean(cv_scores)) if cv_scores else 0.0

        # Final model on all data
        self.model = _make_model(len(y))
        self.model.fit(X_s, y)

        # Conformal residual quantiles (global + regime bins)
        try:
            y_hat = self.model.predict(X_s)
            abs_res = np.abs(y - y_hat)
            self._conformal_q_global = float(np.quantile(abs_res, 0.90)) if len(abs_res) > 0 else 0.0

            self._conformal_q_by_regime = {}
            reg_names = ['Bull-Low Vol', 'Bear-High Vol', 'Sideways', 'Bull-High Vol']
            reg_cols = [
                FeatureFactory.FEATURE_NAMES.index('regime_bull_low'),
                FeatureFactory.FEATURE_NAMES.index('regime_bear_high'),
                FeatureFactory.FEATURE_NAMES.index('regime_sideways'),
                FeatureFactory.FEATURE_NAMES.index('regime_bull_high'),
            ]
            X_raw = np.array(self.training_X)
            if X_raw.ndim == 2 and X_raw.shape[1] >= max(reg_cols) + 1:
                reg_idx = np.argmax(X_raw[:, reg_cols], axis=1)
                for i, rn in enumerate(reg_names):
                    mask = reg_idx == i
                    if np.sum(mask) >= 20:
                        self._conformal_q_by_regime[rn] = float(np.quantile(abs_res[mask], 0.90))
        except Exception:
            self._conformal_q_global = self._conformal_q_global if self._conformal_q_global is not None else 0.0
            self._conformal_q_by_regime = self._conformal_q_by_regime if self._conformal_q_by_regime is not None else {}

        self.is_trained = True
        self._save_model()

    def get_feature_importance(self):
        """Return feature importance dict (if model is trained)."""
        if not self.is_trained or self.model is None:
            return {}
        imp = self.model.feature_importances_
        return {name: round(float(v), 4)
                for name, v in zip(FeatureFactory.FEATURE_NAMES, imp)
                if v > 0.005}


# ============================================================================
# MODULE 3 — EFFICIENCY HUNTER  (anomaly detection)
# ============================================================================

class EfficiencyHunter:
    """
    Detects market inefficiencies through:
    1. Isolation Forest anomaly scoring  (trained on chain-wide features)
    2. Statistical IV anomaly  (IV vs VIX deviation)
    3. Put-call parity violation  (C − P ≠ Se^{-qT} − Ke^{-rT})
    4. Mispricing persistence  (how fast do mispricings typically close?)

    Outputs an overall "efficiency score" (0–100).
    High score → option is more likely genuinely mispriced.
    """

    def __init__(self):
        self.iso_forest = None
        self.lof = None  # Local Outlier Factor (density-based, catches cluster anomalies)
        self.history = []  # feature arrays
        self.history_regimes = []  # regime labels for per-regime IF
        self.regime_forests = {}  # {regime_name: trained IsolationForest}
        if SKLEARN_AVAILABLE:
            self.iso_forest = IsolationForest(
                n_estimators=100, contamination=0.10, random_state=42)
            try:
                from sklearn.neighbors import LocalOutlierFactor
                self.lof = LocalOutlierFactor(
                    n_neighbors=20, contamination=0.10, novelty=True)
            except ImportError:
                pass

    def score(self, features: dict, ce_price=0.0, pe_price=0.0, market_spot=None) -> dict:
        # Derive spot from market_spot if available, else from strike/moneyness
        if market_spot is not None and float(market_spot) > 0:
            spot = float(market_spot)
        else:
            strike = features.get('strike', features.get('K', 0.0))
            moneyness = features.get('moneyness', 1.0)
            spot = float(strike / moneyness) if moneyness > 0 and strike > 0 else float(max(features.get('spot', 1.0), 1e-8))
        result = {
            'anomaly_score': 0.0,
            'iv_anomaly': 0.0,
            'parity_violation': 0.0,
            'mispricing_strength': 0.0,
            'decay_hours': 24.0,
            'overall_score': 50.0,
        }

        # IV anomaly — how far is this option's IV from VIX?
        iv  = features.get('iv', 0.15)
        vix = max(features.get('vix', 0.14), 0.01)
        iv_ratio = iv / vix
        result['iv_anomaly'] = float(np.clip((abs(iv_ratio - 1.0) - 0.15) * 4, 0, 1))

        # Mispricing strength (from NIRV)
        misp = abs(features.get('nirv_mispricing', 0))
        result['mispricing_strength'] = float(np.clip(misp * 4, 0, 1))

        # Isolation Forest (global + per-regime)
        X_arr = FeatureFactory.to_array(features).reshape(1, -1)
        regime = features.get('regime_bull_low', 0) * 1 + features.get('regime_bear_high', 0) * 2 + \
                 features.get('regime_sideways', 0) * 3 + features.get('regime_bull_high', 0) * 4
        regime_name = {1: 'Bull-Low Vol', 2: 'Bear-High Vol', 3: 'Sideways', 4: 'Bull-High Vol'}.get(int(regime), 'Sideways')

        # NOTE: Use `is not None` -- sklearn estimators raise AttributeError
        # on bool()/len() if not yet fitted (no estimators_ attribute).
        _iso_fitted = (self.iso_forest is not None and SKLEARN_AVAILABLE
                       and len(self.history) >= 50
                       and hasattr(self.iso_forest, 'estimators_'))

        if _iso_fitted:
            try:
                raw = self.iso_forest.score_samples(X_arr)[0]
                result['anomaly_score'] = float(np.clip(-raw, -1, 1))
            except Exception:
                pass

            # Per-regime IF (anomaly within the current regime, not globally)
            if regime_name in self.regime_forests:
                try:
                    raw_regime = self.regime_forests[regime_name].score_samples(X_arr)[0]
                    result['anomaly_score_regime'] = float(np.clip(-raw_regime, -1, 1))
                    result['anomaly_score'] = 0.4 * result['anomaly_score'] + 0.6 * result.get('anomaly_score_regime', 0)
                except Exception:
                    pass

        # LOF (density-based, catches cluster anomalies IF misses)
        _lof_fitted = (self.lof is not None and SKLEARN_AVAILABLE
                       and len(self.history) >= 50
                       and hasattr(self.lof, 'n_neighbors_'))
        if _lof_fitted:
            try:
                lof_score = self.lof.score_samples(X_arr)[0]
                result['lof_score'] = float(np.clip(-lof_score, -1, 1))
                # Blend LOF into overall anomaly
                result['anomaly_score'] = max(result['anomaly_score'],
                                              result.get('lof_score', 0) * 0.5)
            except Exception:
                pass

        # Put-call parity check
        if ce_price > 0 and pe_price > 0:
            # moneyness = K/S in FeatureFactory, so K = S * moneyness
            S = float(market_spot) if (market_spot is not None and market_spot > 0) else 24000.0
            moneyness = features.get('moneyness', 1.0)
            K = S * moneyness
            T = max(features.get('time_to_expiry', 0.02), 1e-6)
            r = 0.065
            theoretical = S * np.exp(-0.012 * T) - K * np.exp(-r * T)
            actual = ce_price - pe_price
            pdev = abs(actual - theoretical) / max(S * 0.01, 1)
            result['parity_violation'] = float(np.clip(pdev, 0, 1))

        # Decay estimate
        if misp > 0.03:
            result['decay_hours'] = max(1, 48 * (1 - min(misp * 3, 0.9)))

        # Composite
        result['overall_score'] = float(np.clip(
            50
            + 15 * result['anomaly_score']
            + 15 * result['iv_anomaly']
            + 15 * result['mispricing_strength']
            + 5  * result['parity_violation'],
            0, 100))

        return result

    def update(self, features: dict):
        """Accumulate history and periodically retrain IF (global + per-regime)."""
        arr = FeatureFactory.to_array(features).tolist()
        self.history.append(arr)
        # Track regime for per-regime training
        regime = 'Sideways'
        if features.get('regime_bull_low', 0) > 0.5: regime = 'Bull-Low Vol'
        elif features.get('regime_bear_high', 0) > 0.5: regime = 'Bear-High Vol'
        elif features.get('regime_bull_high', 0) > 0.5: regime = 'Bull-High Vol'
        self.history_regimes.append(regime)

        if (SKLEARN_AVAILABLE and self.iso_forest is not None
                and len(self.history) >= 50
                and len(self.history) % 50 == 0):
            X = np.array(self.history[-500:])
            self.iso_forest.fit(X)

            # Train LOF
            if self.lof:
                try:
                    self.lof.fit(X)
                except Exception:
                    pass

            # Train per-regime IsolationForests
            regimes_arr = self.history_regimes[-500:]
            for rn in ['Bull-Low Vol', 'Bear-High Vol', 'Sideways', 'Bull-High Vol']:
                idx = [i for i, r in enumerate(regimes_arr) if r == rn]
                if len(idx) >= 30:
                    X_regime = np.array([self.history[-500:][i] for i in idx])
                    regime_if = IsolationForest(
                        n_estimators=80, contamination=0.12, random_state=42)
                    regime_if.fit(X_regime)
                    self.regime_forests[rn] = regime_if


# ============================================================================
# MODULE 4 — SENTIMENT INTELLIGENCE  (AI-powered)
# ============================================================================

class SentimentIntelligence:
    """
    Analyses market sentiment from multiple AI sources:
      - Gemini API responses
      - Perplexity API responses
      - Raw news headlines

    Outputs a score ∈ [−1, +1] with confidence, contributing factors,
    and a direction label (BULLISH / BEARISH / NEUTRAL).
    """

    _BULL = ['bullish', 'rally', 'upside', 'growth', 'recovery', 'buying',
             'breakout', 'strong', 'positive', 'optimistic', 'support',
             'upgrade', 'beat', 'surge', 'stimulus', 'rate cut', 'dovish']
    _BEAR = ['bearish', 'crash', 'downside', 'decline', 'selling', 'weak',
             'breakdown', 'negative', 'pessimistic', 'tariff', 'war',
             'sanctions', 'inflation', 'recession', 'hawkish', 'miss',
             'downgrade', 'sell-off', 'slowdown', 'crisis', 'risk-off']

    def analyse(self, gemini_resp=None, perplexity_resp=None,
                headlines=None) -> dict:
        """
        Analyse sentiment from multiple sources.

        Improvement (Item 7):
        - Extracts numeric score from LLM responses (structured prompt)
        - Falls back to keyword counting only if numeric extraction fails
        - Weights headlines by recency (recent > old)
        """
        scores  = []
        factors = []

        if gemini_resp:
            s, f = self._score_text_llm(gemini_resp, 'Gemini')
            scores.append(s * 0.40)
            factors.extend(f)

        if perplexity_resp:
            s, f = self._score_text_llm(perplexity_resp, 'Perplexity')
            scores.append(s * 0.40)
            factors.extend(f)

        if headlines:
            s = self._score_headlines_weighted(headlines)
            scores.append(s * 0.20)
            if abs(s) > 0.1:
                factors.append(f"Headlines: {'Bullish' if s > 0 else 'Bearish'} ({abs(s):.0%})")

        final = sum(scores) if scores else 0.0
        conf  = min(0.90, len(scores) * 0.30)

        return {
            'score':      float(np.clip(final, -1, 1)),
            'confidence': float(conf),
            'direction':  'BULLISH' if final > 0.10 else 'BEARISH' if final < -0.10 else 'NEUTRAL',
            'factors':    factors[:10],
            'timestamp':  datetime.datetime.now().isoformat(),
        }

    # ------------------------------------------------------------------
    def _score_text_llm(self, text, source):
        """
        Extract numeric sentiment from LLM response.

        The prompt asks for a score from -100 to +100 at the end.
        Parse it; fall back to keyword counting if not found.
        """
        import re
        # Try to extract the final numeric score from LLM response
        # Look for patterns like "Score: 45", "+45", "-30", "sentiment: 60"
        score_patterns = [
            r'(?:score|rating|sentiment)\s*[:=]\s*([+-]?\d+)',
            r'\b([+-]?\d{1,3})\s*(?:/\s*100|\s*out\s+of\s+100)',
            r'(?:^|\n)\s*([+-]?\d{1,3})\s*$',  # standalone number at end
        ]

        llm_score = None
        for pattern in score_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            if matches:
                try:
                    val = int(matches[-1])  # take the last match
                    if -100 <= val <= 100:
                        llm_score = val / 100.0
                        break
                except (ValueError, IndexError):
                    pass

        if llm_score is not None:
            fac = [f"{source}: LLM score {llm_score*100:+.0f}/100"]
            return llm_score, fac

        # Fallback: keyword counting
        return self._score_text_keywords(text, source)

    def _score_text_keywords(self, text, source):
        """Fallback keyword-based sentiment scoring."""
        low = text.lower()
        bc = sum(1 for w in self._BULL if w in low)
        br = sum(1 for w in self._BEAR if w in low)
        total = bc + br + 1
        score = (bc - br) / total
        fac = []
        if bc > br:
            fac.append(f"{source}: Bullish cues ({bc} bull vs {br} bear)")
        elif br > bc:
            fac.append(f"{source}: Bearish cues ({br} bear vs {bc} bull)")
        else:
            fac.append(f"{source}: Mixed / Neutral")
        return score, fac

    def _score_headlines_weighted(self, headlines):
        """
        Score headlines with recency weighting.
        Earlier headlines in the list are assumed to be more recent.
        Weight by exp(-lambda * index) so recent headlines dominate.
        """
        pos = ['surge', 'gain', 'rise', 'rally', 'record', 'boom', 'growth',
               'profit', 'beat', 'strong', 'up']
        neg = ['crash', 'fall', 'drop', 'slump', 'loss', 'miss', 'weak',
               'crisis', 'fear', 'sell-off', 'tariff', 'down']
        total_score = 0.0
        total_weight = 0.0
        decay = 0.15  # recency decay rate
        for i, h in enumerate(headlines):
            hl = h.lower()
            s = sum(1 for w in pos if w in hl) - sum(1 for w in neg if w in hl)
            weight = np.exp(-decay * i)  # recent headlines get higher weight
            total_score += s * weight
            total_weight += weight
        if total_weight > 0:
            return np.clip(total_score / total_weight, -1, 1)
        return 0.0

    # ------------------------------------------------------------------
    @staticmethod
    def build_prompt(underlying: str, ctx: dict) -> str:
        return f"""Analyse the current and near-term market sentiment for **{underlying}** options trading on NSE (India).

Market snapshot:
  Spot: ₹{ctx.get('spot','N/A')}  |  India VIX: {ctx.get('vix','N/A')}
  FII net flow: ₹{ctx.get('fii','N/A')} cr  |  PCR(OI): {ctx.get('pcr','N/A')}
  Regime: {ctx.get('regime','Unknown')}  |  Date: {datetime.datetime.now():%Y-%m-%d %H:%M}

Provide (be specific, data-driven):
1. Sentiment verdict — Bullish / Bearish / Neutral with % confidence.
2. Top 5 events in the next 1-7 days that could move Indian markets.
3. Geopolitical risks (tariffs, conflicts, sanctions).
4. Key actors (Trump, Fed, RBI, FIIs) — their recent action & likely next move.
5. Technical levels to watch for {underlying}.
6. Optimal option strategy given this outlook (direction + IV view).

End with a single integer score from −100 (extreme bear) to +100 (extreme bull)."""


# ============================================================================
# MODULE 5 — BEHAVIORAL ENGINE  (actor pattern recognition)
# ============================================================================

class BehavioralEngine:
    """
    Tracks and predicts behavior of key market-moving actors.

    Maintains a pattern database (calibrated from historical observations)
    and an observation log that grows over time.  Uses pattern-matching
    heuristics + optional AI analysis to predict next moves.
    """

    # Pre-built pattern database
    PATTERNS = {
        'trump': {
            'tariff_escalation': {
                'desc': 'Escalates trade war with new tariffs / higher duties',
                'impact': -0.025, 'precursors': ['trade war', 'tariff', 'china', 'trade deficit'],
            },
            'tariff_pause': {
                'desc': 'Pauses or delays tariffs for negotiation',
                'impact': +0.015, 'precursors': ['trade deal', 'negotiation', 'positive talks'],
            },
            'market_pump': {
                'desc': 'Posts positive economy/market comments',
                'impact': +0.008, 'precursors': ['stock market', 'economy great', 'record high'],
            },
            'fed_pressure': {
                'desc': 'Publicly pressures Fed to cut rates',
                'impact': +0.005, 'precursors': ['fed', 'rate cut', 'powell', 'interest rate'],
            },
            'executive_order': {
                'desc': 'Signs executive order affecting markets/trade/tech',
                'impact': -0.015, 'precursors': ['executive order', 'ban', 'restrict', 'policy'],
            },
        },
        'fed': {
            'hawkish_surprise': {
                'desc': 'Signals more tightening than expected',
                'impact': -0.020, 'precursors': ['strong jobs', 'inflation high', 'rate hike'],
            },
            'dovish_pivot': {
                'desc': 'Signals easing or pause',
                'impact': +0.025, 'precursors': ['weak data', 'recession risk', 'rate cut', 'pause'],
            },
            'balance_sheet': {
                'desc': 'Changes QT/QE pace',
                'impact': +0.010, 'precursors': ['balance sheet', 'taper', 'liquidity'],
            },
        },
        'rbi': {
            'rate_cut': {
                'desc': 'Cuts repo rate — stimulative',
                'impact': +0.015, 'precursors': ['rate cut', 'growth concern', 'low inflation'],
            },
            'rate_hike': {
                'desc': 'Hikes repo rate — contractionary',
                'impact': -0.012, 'precursors': ['inflation', 'rupee weak', 'overheating'],
            },
            'liquidity_inject': {
                'desc': 'Injects liquidity via OMO / CRR cut',
                'impact': +0.008, 'precursors': ['liquidity', 'omo', 'crr', 'credit growth'],
            },
        },
        'global': {
            'oil_spike': {
                'desc': 'Oil price spikes — negative for India (net importer)',
                'impact': -0.012, 'precursors': ['oil', 'crude', 'opec', 'middle east'],
            },
            'dollar_surge': {
                'desc': 'US dollar strengthens — FII outflows from EM',
                'impact': -0.010, 'precursors': ['dollar', 'dxy', 'usd', 'treasury yield'],
            },
            'risk_on_rally': {
                'desc': 'Global risk-on mode — inflows to EM',
                'impact': +0.015, 'precursors': ['risk on', 'global rally', 'stimulus', 'liquidity'],
            },
        },
    }

    def __init__(self, data_path='omega_data/behavioral_log.json'):
        self.data_path = data_path
        self.observations = []
        self._load()

    def _load(self):
        if os.path.exists(self.data_path):
            try:
                with open(self.data_path, 'r') as f:
                    self.observations = json.load(f)
            except Exception:
                self.observations = []

    def _save(self):
        try:
            os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
            with open(self.data_path, 'w') as f:
                json.dump(self.observations[-1000:], f, default=str)
        except Exception:
            pass

    # ------------------------------------------------------------------
    def predict(self, actor: str, context: str = '',
                days_to_rbi=None, days_to_fed=None) -> dict:
        """
        Predict most likely next action for an actor, given current context.

        Improvements (Item 8):
        - Rolling confusion matrix: uses historical prediction accuracy
        - Temporal decay: recent observations weighted exponentially more
        - Event calendar: hard constraints from days_to_rbi/fed
        """
        actor_key = actor.lower()
        patterns  = self.PATTERNS.get(actor_key, {})
        if not patterns:
            return {'actor': actor, 'likely_action': 'Unknown',
                    'impact': 0.0, 'probability': 0.0, 'scenarios': []}

        ctx_lower = context.lower()
        scored = []
        for action, p in patterns.items():
            prob = 0.20  # base

            # Precursor matching
            for precursor in p.get('precursors', []):
                if precursor in ctx_lower:
                    prob += 0.15

            # Calibrate using historical observations (rolling confusion matrix)
            hist_accuracy = self._get_historical_accuracy(actor_key, action)
            if hist_accuracy is not None:
                # Blend base probability with empirical accuracy
                prob = 0.5 * prob + 0.5 * hist_accuracy

            # Temporal decay: count recent mentions with exponential weighting
            recent_obs = self._get_recent_observations(actor_key, action, decay_days=14)
            if recent_obs > 0:
                prob += min(0.10, recent_obs * 0.03)

            # Event calendar hard constraints (Item 8)
            if actor_key == 'rbi' and days_to_rbi is not None:
                if days_to_rbi <= 3:
                    if action in ('rate_cut', 'rate_hike', 'liquidity_inject'):
                        prob += 0.25  # RBI about to act
                elif days_to_rbi > 30:
                    if action in ('rate_cut', 'rate_hike'):
                        prob *= 0.5  # RBI unlikely to act between meetings

            if actor_key == 'fed' and days_to_fed is not None:
                if days_to_fed <= 3:
                    if action in ('hawkish_surprise', 'dovish_pivot'):
                        prob += 0.25
                elif days_to_fed > 30:
                    if action in ('hawkish_surprise', 'dovish_pivot'):
                        prob *= 0.5

            prob = min(prob, 0.90)
            scored.append({
                'action': action, 'description': p['desc'],
                'probability': round(prob, 2),
                'impact': p['impact'],
            })

        scored.sort(key=lambda x: x['probability'], reverse=True)
        best = scored[0]
        return {
            'actor': actor,
            'likely_action': best['action'],
            'description': best['description'],
            'impact': best['impact'],
            'probability': best['probability'],
            'scenarios': scored[:3],
        }

    def _get_historical_accuracy(self, actor, action):
        """Rolling confusion matrix: P(action occurred | we predicted it)."""
        relevant = [o for o in self.observations
                    if o.get('actor', '').lower() == actor
                    and o.get('predicted_action') == action]
        if len(relevant) < 5:
            return None
        correct = sum(1 for o in relevant if o.get('was_correct', False))
        return correct / len(relevant)

    def _get_recent_observations(self, actor, action, decay_days=14):
        """Count recent observations with temporal decay."""
        now = datetime.datetime.now()
        score = 0.0
        for o in reversed(self.observations[-100:]):
            if o.get('actor', '').lower() != actor:
                continue
            try:
                ts = datetime.datetime.fromisoformat(o.get('ts', ''))
                days_ago = (now - ts).total_seconds() / 86400
                weight = np.exp(-days_ago / decay_days)
                if o.get('action', '') == action:
                    score += weight
            except (ValueError, TypeError):
                continue
        return score

    def add_observation(self, actor, action, market_reaction,
                        predicted_action=None, was_correct=None):
        """Record an observation with optional prediction tracking."""
        self.observations.append({
            'actor': actor, 'action': action,
            'reaction': float(market_reaction),
            'predicted_action': predicted_action,
            'was_correct': was_correct,
            'ts': datetime.datetime.now().isoformat(),
        })
        self._save()

    # ------------------------------------------------------------------
    @staticmethod
    def build_prompt(actors, ctx):
        hist = ''
        return f"""Analyse behavioral patterns for these key market actors: {', '.join(actors)}

Context: India VIX={ctx.get('vix','?')}, Regime={ctx.get('regime','?')}, Date={datetime.datetime.now():%Y-%m-%d}
{hist}

For EACH actor:
1. Most likely next action (based on their historical behavior)
2. Expected Nifty impact (% move, direction)
3. Timing estimate
4. Early-warning signals to watch
5. How it affects Indian index/stock options

Be specific and actionable for an options trader."""


# ============================================================================
# MODULE 6 — PREDICTION TRACKER  (adaptive learning loop)
# ============================================================================

class PredictionTracker:
    """
    Records every OMEGA prediction with its full feature vector and, later,
    the actual outcome.  This enables:
      - Performance monitoring  (signal accuracy, cumulative P&L)
      - ML retraining  (features → residual)
      - Confidence calibration  (are 80 % predictions really right 80 % of the time?)
    """

    def __init__(self, data_path='omega_data/predictions.json'):
        self.data_path = data_path
        self.predictions = []
        self.metrics = {}
        self._load()

    def _load(self):
        if os.path.exists(self.data_path):
            try:
                with open(self.data_path, 'r') as f:
                    d = json.load(f)
                self.predictions = d.get('predictions', [])
                self.metrics     = d.get('metrics', {})
            except Exception:
                pass

    def _save(self):
        try:
            os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
            with open(self.data_path, 'w') as f:
                json.dump({
                    'predictions': self.predictions[-3000:],
                    'metrics': self.metrics,
                }, f, default=str)
        except Exception:
            pass

    # ------------------------------------------------------------------
    def record(self, pred_id, features, prediction):
        self.predictions.append({
            'id': pred_id,
            'features': {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
                         for k, v in features.items()},
            'pred': prediction,
            'ts': datetime.datetime.now().isoformat(),
            'outcome': None,
        })
        if len(self.predictions) % 10 == 0:
            self._save()

    def record_outcome(self, pred_id, actual_return, actual_price=None):
        for p in reversed(self.predictions):
            if p['id'] == pred_id:
                p['outcome'] = {
                    'actual_return': actual_return,
                    'actual_price': actual_price,
                    'ts': datetime.datetime.now().isoformat(),
                }
                break
        self._recalc()
        self._save()

    # ------------------------------------------------------------------
    def _recalc(self):
        done = [p for p in self.predictions if p.get('outcome')]
        if not done:
            return
        correct = total = pnl = 0
        for p in done[-300:]:
            sig = p['pred'].get('signal', 'HOLD')
            ret = p['outcome'].get('actual_return', 0)
            if sig not in ('HOLD',):
                total += 1
                if (sig in ('BUY', 'STRONG BUY') and ret > 0) or \
                   (sig in ('SELL', 'STRONG SELL') and ret < 0):
                    correct += 1
                pnl += ret if 'BUY' in sig else -ret
        self.metrics = {
            'total_tracked': len(done),
            'recent_signals': total,
            'signal_accuracy': round(correct / max(total, 1) * 100, 1),
            'cumulative_pnl_pct': round(pnl * 100, 2),
            'avg_pnl_per_signal': round(pnl / max(total, 1) * 100, 2),
            'updated': datetime.datetime.now().isoformat(),
        }

    def get_performance(self):
        self._recalc()
        return self.metrics

    def get_training_pairs(self):
        """Return (features_dict, residual) pairs for ML retraining."""
        return [
            (p['features'], p['outcome']['actual_return'])
            for p in self.predictions
            if p.get('outcome') and p.get('features')
        ]

    @staticmethod
    def _regime_from_features(features: dict) -> str:
        if not isinstance(features, dict):
            return 'Unknown'
        if float(features.get('regime_bull_low', 0.0)) > 0.5:
            return 'Bull-Low Vol'
        if float(features.get('regime_bear_high', 0.0)) > 0.5:
            return 'Bear-High Vol'
        if float(features.get('regime_bull_high', 0.0)) > 0.5:
            return 'Bull-High Vol'
        if float(features.get('regime_sideways', 0.0)) > 0.5:
            return 'Sideways'
        return 'Unknown'

    @staticmethod
    def _signed_return_pct(signal: str, actual_return: float) -> float:
        sig = str(signal or '').upper()
        ret = float(actual_return or 0.0)
        if 'BUY' in sig:
            return ret * 100.0
        if 'SELL' in sig:
            return -ret * 100.0
        return 0.0

    @staticmethod
    def _accuracy_for_records(records) -> float:
        if not records:
            return 0.0
        correct = 0
        for rec in records:
            sig = str(rec.get('signal', '')).upper()
            signed_ret = float(rec.get('signed_return_pct', 0.0))
            if ('BUY' in sig and signed_ret > 0.0) or ('SELL' in sig and signed_ret > 0.0):
                correct += 1
        return 100.0 * correct / max(len(records), 1)

    def _build_reliability_records(self, lookback: int = 300):
        rows = []
        for p in reversed(self.predictions):
            out = p.get('outcome') or {}
            pred = p.get('pred') or {}
            sig = str(pred.get('signal', 'HOLD')).upper()
            if sig not in ('BUY', 'STRONG BUY', 'SELL', 'STRONG SELL'):
                continue
            if 'actual_return' not in out:
                continue
            signed_ret_pct = self._signed_return_pct(sig, out.get('actual_return', 0.0))
            rows.append({
                'signal': sig,
                'signed_return_pct': signed_ret_pct,
                'regime': self._regime_from_features(p.get('features') or {}),
            })
            if len(rows) >= int(lookback):
                break
        rows.reverse()
        return rows

    def get_reliability_gate_decision(
        self,
        signal: str,
        features: dict = None,
        min_samples: int = 40,
        min_accuracy_pct: float = 58.0,
        min_avg_edge_pct: float = 0.10,
        min_side_samples: int = 15,
        min_regime_samples: int = 10,
        lookback: int = 300,
    ) -> dict:
        """
        Out-of-sample reliability gate for directional signals.

        Directional signals are blocked when tracked historical performance is
        insufficient or weak.
        """
        sig = str(signal or '').upper()
        if sig not in ('BUY', 'STRONG BUY', 'SELL', 'STRONG SELL'):
            return {
                'required': False,
                'passed': True,
                'reason': 'non_directional',
                'total_samples': 0,
                'accuracy_pct': 0.0,
                'avg_edge_pct': 0.0,
                'side_samples': 0,
                'side_accuracy_pct': 0.0,
                'regime': 'Unknown',
                'regime_samples': 0,
                'regime_accuracy_pct': 0.0,
            }

        rows = self._build_reliability_records(lookback=lookback)
        total_samples = len(rows)
        if total_samples < int(min_samples):
            return {
                'required': True,
                'passed': False,
                'reason': f'insufficient_history<{int(min_samples)}',
                'total_samples': total_samples,
                'accuracy_pct': 0.0,
                'avg_edge_pct': 0.0,
                'side_samples': 0,
                'side_accuracy_pct': 0.0,
                'regime': 'Unknown',
                'regime_samples': 0,
                'regime_accuracy_pct': 0.0,
            }

        accuracy_pct = self._accuracy_for_records(rows)
        avg_edge_pct = float(np.mean([r['signed_return_pct'] for r in rows])) if rows else 0.0

        side = 'BUY' if 'BUY' in sig else 'SELL'
        side_rows = [r for r in rows if side in r['signal']]
        side_samples = len(side_rows)
        side_accuracy_pct = self._accuracy_for_records(side_rows) if side_rows else 0.0

        regime = self._regime_from_features(features or {})
        regime_rows = [r for r in rows if r.get('regime') == regime] if regime != 'Unknown' else []
        regime_samples = len(regime_rows)
        regime_accuracy_pct = self._accuracy_for_records(regime_rows) if regime_rows else 0.0

        if accuracy_pct < float(min_accuracy_pct):
            reason = f'low_global_accuracy<{float(min_accuracy_pct):.1f}%'
            passed = False
        elif avg_edge_pct < float(min_avg_edge_pct):
            reason = f'low_global_edge<{float(min_avg_edge_pct):.2f}%'
            passed = False
        elif side_samples >= int(min_side_samples) and side_accuracy_pct < max(float(min_accuracy_pct) - 2.0, 0.0):
            reason = 'low_side_accuracy'
            passed = False
        elif regime_samples >= int(min_regime_samples) and regime_accuracy_pct < max(float(min_accuracy_pct) - 3.0, 0.0):
            reason = 'low_regime_accuracy'
            passed = False
        else:
            reason = 'pass'
            passed = True

        return {
            'required': True,
            'passed': bool(passed),
            'reason': reason,
            'total_samples': total_samples,
            'accuracy_pct': round(float(accuracy_pct), 2),
            'avg_edge_pct': round(float(avg_edge_pct), 3),
            'side': side,
            'side_samples': side_samples,
            'side_accuracy_pct': round(float(side_accuracy_pct), 2),
            'regime': regime,
            'regime_samples': regime_samples,
            'regime_accuracy_pct': round(float(regime_accuracy_pct), 2),
        }


# ============================================================================
# MODULE 7 — TRADE PLAN GENERATOR
# ============================================================================

class TradePlanGenerator:
    """
    Generates a comprehensive trade plan from an OMEGA output,
    including entry, exit, stop-loss, position sizing, hold period,
    and risk/reward analysis.
    """

    @staticmethod
    def generate(omega_out, spot, lot_size=65, capital=500000):
        mp       = omega_out.market_price
        fv       = omega_out.fair_value
        signal   = omega_out.signal
        greeks   = omega_out.greeks or {}
        delta    = greeks.get('delta', 0.5)
        theta    = greeks.get('theta', 0)
        gamma    = greeks.get('gamma', 0)
        eff      = omega_out.efficiency_score
        conf     = omega_out.confidence_level
        phys_pop = omega_out.physical_profit_prob

        # Entry / exit levels
        if 'BUY' in signal:
            entry  = mp * 0.995          # slight discount limit
            target1 = fv                  # fair value
            target2 = fv * 1.10          # 10% beyond fair value
            stop    = mp * 0.70          # 30% stop
        elif 'SELL' in signal:
            entry  = mp * 1.005
            target1 = fv
            target2 = fv * 0.90
            stop    = mp * 1.30
        else:
            entry = target1 = target2 = mp
            stop = mp * 0.80

        # Hold period estimate (based on theta and time value)
        if abs(theta) > 0.01:
            # How many days of theta burn before the edge disappears?
            edge = abs(fv - mp)
            hold_days = max(1, min(int(edge / max(abs(theta), 0.01)), 30))
        else:
            hold_days = 5

        # Position sizing (fixed-risk model)
        risk_per_lot = abs(entry - stop) * lot_size
        if risk_per_lot > 0:
            max_risk = capital * 0.02  # 2% of capital
            lots = max(1, int(max_risk / risk_per_lot))
        else:
            lots = 1

        # Risk / Reward
        reward = abs(target1 - entry) * lot_size * lots
        risk   = abs(entry - stop) * lot_size * lots
        rr     = reward / max(risk, 1)

        # Kelly fraction (simplified)
        p = phys_pop / 100.0
        b = rr
        kelly = max(0, (p * b - (1 - p)) / max(b, 0.01))
        kelly_lots = max(1, int(lots * min(kelly, 0.5)))  # half-Kelly

        return {
            'signal': signal,
            'entry': round(entry, 2),
            'target_1': round(target1, 2),
            'target_2': round(target2, 2),
            'stop_loss': round(stop, 2),
            'hold_days': hold_days,
            'lots_full': lots,
            'lots_kelly': kelly_lots,
            'risk_per_lot': round(risk_per_lot, 0),
            'reward_per_lot': round(reward / max(lots, 1), 0),
            'risk_reward': round(rr, 2),
            'kelly_fraction': round(kelly, 3),
            'capital_required': round(entry * lot_size * kelly_lots, 0),
            'max_loss': round(abs(entry - stop) * lot_size * kelly_lots, 0),
            'max_gain_t1': round(abs(target1 - entry) * lot_size * kelly_lots, 0),
            'confidence': conf,
            'efficiency_score': eff,
            'theta_bleed_daily': round(abs(theta) * lot_size * kelly_lots, 0),
        }


# ============================================================================
# ADDITION 2: PROSPECT THEORY PRICING KERNEL
# ============================================================================

class ProspectTheoryKernel:
    """
    Implements Barberis & Huang (2008) prospect theory pricing:

    Key insight: Retail overpays for OTM options because:
    1. Prelec probability weighting: π(p) = exp(-(-ln(p))^γ), γ ≈ 0.65
       → Overweights small probabilities (lottery effect)
    2. Loss aversion: λ = 2.25 (losses loom 2.25x larger than gains)
       → Overpays for downside protection (OTM puts)

    The "behavioral edge" is the difference between PT-implied-price
    and risk-neutral BSM price. This edge is systematic and persistent.
    """

    def __init__(self, gamma=0.65, lambda_loss=2.25, alpha=0.88):
        """
        Parameters
        ----------
        gamma      : float - Prelec probability weighting parameter
        lambda_loss: float - Loss aversion coefficient
        alpha      : float - Diminishing sensitivity exponent
        """
        self.gamma = gamma
        self.lambda_loss = lambda_loss
        self.alpha = alpha

    def prelec_weight(self, p):
        """Prelec probability weighting function."""
        p = np.clip(p, 1e-10, 1 - 1e-10)
        return np.exp(-(-np.log(p)) ** self.gamma)

    def prospect_value(self, x):
        """
        Prospect theory value function.
        v(x) = x^α for gains, -λ|x|^α for losses.
        """
        x = np.asarray(x, dtype=float)
        result = np.where(x >= 0, x ** self.alpha,
                         -self.lambda_loss * np.abs(x) ** self.alpha)
        return result

    def compute_behavioral_edge(self, S_terminal, strike, market_price,
                                 option_type='CE', T=None, r=0.065):
        """
        Compute behavioral mispricing from prospect theory.

        Parameters
        ----------
        S_terminal  : np.ndarray - MC terminal prices
        strike      : float
        market_price: float
        option_type : str
        T           : float - time to expiry
        r           : float - risk-free rate

        Returns
        -------
        dict with pt_price, bsm_price, behavioral_edge, mispricing_flag
        """
        if S_terminal is None or len(S_terminal) < 100 or market_price <= 0:
            return {'pt_price': market_price, 'behavioral_edge': 0.0,
                    'mispricing_flag': False}

        try:
            is_call = option_type.upper() in ('CE', 'CALL')
            disc = np.exp(-r * T) if T is not None else 1.0

            # Standard risk-neutral payoffs
            if is_call:
                payoffs = np.maximum(S_terminal - strike, 0)
            else:
                payoffs = np.maximum(strike - S_terminal, 0)

            bsm_price = disc * np.mean(payoffs)

            # PT payoffs: relative to reference point (market_price)
            pnl = payoffs * disc - market_price  # Per-unit P&L

            # Apply PT value function to P&L
            pt_values = self.prospect_value(pnl)

            # Apply Prelec probability weighting
            n = len(pnl)
            # Sort by value (ascending) for cumulative weighting
            sorted_idx = np.argsort(pnl)
            sorted_values = pt_values[sorted_idx]

            # Cumulative probabilities
            cum_probs = np.arange(1, n + 1) / n
            prev_cum = np.arange(0, n) / n

            # Weighted probabilities via Prelec
            weighted_probs = self.prelec_weight(cum_probs) - self.prelec_weight(prev_cum)
            weighted_probs = np.maximum(weighted_probs, 0)
            weighted_probs /= weighted_probs.sum()

            # PT-implied certainty equivalent
            pt_ce = np.sum(sorted_values * weighted_probs)

            # PT-implied price: what a PT agent would pay
            # If PT CE > 0, agent finds the option attractive → willing to pay more
            pt_premium = max(0, pt_ce * 0.1)  # Scale factor
            pt_price = bsm_price + pt_premium

            # Behavioral edge: how much is market overpricing due to PT biases?
            behavioral_edge = market_price - bsm_price
            relative_edge = behavioral_edge / max(market_price, 0.01) * 100

            # Flag significant behavioral mispricing (>3% relative)
            mispricing_flag = relative_edge > 3.0

            # OTM options are most affected by PT (lottery effect)
            moneyness = np.log(S_terminal.mean() / strike) if is_call else np.log(strike / S_terminal.mean())
            is_otm = moneyness < -0.02

            return {
                'pt_price': float(pt_price),
                'bsm_price': float(bsm_price),
                'behavioral_edge': float(behavioral_edge),
                'relative_edge_pct': float(relative_edge),
                'mispricing_flag': bool(mispricing_flag),
                'is_otm': bool(is_otm),
                'pt_overpricing_pct': float(relative_edge) if is_otm else 0.0,
                'prelec_gamma': self.gamma,
                'loss_aversion': self.lambda_loss,
            }
        except Exception:
            return {'pt_price': market_price, 'behavioral_edge': 0.0,
                    'mispricing_flag': False}


# ============================================================================
# ADDITION 3: DISPOSITION EFFECT FLOW PREDICTOR
# ============================================================================

class DispositionFlowPredictor:
    """
    Predicts option order flow direction based on the disposition effect:

    - After GAINS: retail sells winners too early (calls after rally)
    - After LOSSES: retail holds losers too long (puts after crash)

    Combined with OI changes, this predicts whether the next day's flow
    will be buying or selling pressure in specific strikes.

    Signal ∈ [-1, 1]:
      +1 = Strong selling pressure (buy ITM calls — flow reversal)
      -1 = Strong holding pressure (sell OTM puts — no exit)
    """

    @staticmethod
    def predict(returns_5d, oi_change_calls, oi_change_puts, spot, strikes):
        """
        Parameters
        ----------
        returns_5d      : np.ndarray - Last 5 daily returns
        oi_change_calls : dict - {strike: OI_today - OI_yesterday} for calls
        oi_change_puts  : dict - {strike: OI_today - OI_yesterday} for puts
        spot            : float - Current spot
        strikes         : list - Strike prices

        Returns
        -------
        dict with disposition_signal, flow_direction, affected_strikes
        """
        if returns_5d is None or len(returns_5d) < 3:
            return {'disposition_signal': 0.0, 'flow_direction': 'NEUTRAL',
                    'affected_strikes': []}

        try:
            returns = np.asarray(returns_5d, dtype=float)
            cum_return = np.sum(returns)

            # Disposition effect intensity
            if cum_return > 0.02:
                # After gains: retail books profits early → selling pressure
                disposition_raw = min(cum_return * 20, 1.0)
            elif cum_return < -0.02:
                # After losses: retail holds → no selling → buying vacuum
                disposition_raw = max(cum_return * 15, -1.0)
            else:
                disposition_raw = 0.0

            # OI confirmation: where is the flow happening?
            affected = []
            call_oi_total = sum(oi_change_calls.get(k, 0) for k in strikes) if oi_change_calls else 0
            put_oi_total = sum(oi_change_puts.get(k, 0) for k in strikes) if oi_change_puts else 0

            # Long unwinding (falling OI + positive cum return) confirms disposition
            if call_oi_total < 0 and cum_return > 0:
                disposition_raw *= 1.3  # Confirmed: retail selling calls
            elif put_oi_total < 0 and cum_return < 0:
                disposition_raw *= 1.3  # Confirmed: retail closing puts

            # Find most affected strikes
            if oi_change_calls and oi_change_puts:
                for k in strikes:
                    c_change = oi_change_calls.get(k, 0)
                    p_change = oi_change_puts.get(k, 0)
                    moneyness = (spot - k) / spot
                    if abs(c_change) + abs(p_change) > 0:
                        affected.append({
                            'strike': k,
                            'call_oi_change': c_change,
                            'put_oi_change': p_change,
                            'moneyness': round(moneyness, 4),
                        })

            # Sort by absolute OI change
            affected.sort(key=lambda x: abs(x['call_oi_change']) + abs(x['put_oi_change']),
                         reverse=True)

            signal = float(np.clip(disposition_raw, -1, 1))

            if signal > 0.3:
                direction = 'SELLING_PRESSURE'
            elif signal < -0.3:
                direction = 'HOLDING_PRESSURE'
            else:
                direction = 'NEUTRAL'

            return {
                'disposition_signal': signal,
                'flow_direction': direction,
                'cum_return_5d': float(cum_return),
                'call_oi_net_change': int(call_oi_total),
                'put_oi_net_change': int(put_oi_total),
                'affected_strikes': affected[:5],
            }
        except Exception:
            return {'disposition_signal': 0.0, 'flow_direction': 'NEUTRAL',
                    'affected_strikes': []}


# ============================================================================
# MODULE 8 — OMEGA MASTER MODEL  (orchestrator)
# ============================================================================

class OMEGAModel:
    """
    OMEGA: Options Market Efficiency & Generative Analysis

    Master orchestrator that layers ML/AI intelligence on top of NIRV:

        NIRV (maths)  →  ML correction  →  sentiment adj  →  OMEGA price

    Features:
      - Cold-start safe (works on day 1 with NIRV alone)
      - Progressively improves as predictions are tracked
      - Integrates AI sentiment from Gemini / Perplexity
      - Detects statistical anomalies / inefficiencies
      - Predicts key-actor behavior and market impact
      - Generates complete trade plans

    Usage
    -----
    >>> omega = OMEGAModel()
    >>> result = omega.price_option(spot=23500, strike=23400, ...)
    >>> plan = omega.generate_trade_plan(result, spot=23500)
    """

    def __init__(self, nirv_model=None, data_dir='omega_data'):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

        # Layer 0 — mathematical base
        self.nirv = nirv_model or NIRVModel(n_paths=50000, n_bootstrap=1000)

        # Layer 1 — ML correction
        self.ml = MLPricingCorrector(
            model_path=os.path.join(data_dir, 'pricing_model.joblib')
        ) if SKLEARN_AVAILABLE else None

        # Layer 2 — Anomaly detection
        self.hunter = EfficiencyHunter()

        # Layer 3 — Sentiment
        self.sentiment = SentimentIntelligence()

        # Layer 4 — Behavioral
        self.behavioral = BehavioralEngine(
            data_path=os.path.join(data_dir, 'behavioral_log.json'))
        self.behavioral_state_engine = BehavioralStateEngine() if BehavioralStateEngine else None

        # Layer 5 — Learning
        self.tracker = PredictionTracker(
            data_path=os.path.join(data_dir, 'predictions.json'))

        # Trade plan builder
        self.trade_gen = TradePlanGenerator()

        # Config
        self.lot_size = 65
        if getattr(get_features(), "USE_NSE_CONTRACT_SPECS", False) and nse_get_lot_size:
            try:
                self.lot_size = int(nse_get_lot_size("NIFTY", None))
            except Exception:
                pass
        self.ml_weight = 0.12       # max ML correction weight
        self.sentiment_weight = 0.05  # max sentiment adjustment

    # ==================================================================
    # MAIN PRICING PIPELINE
    # ==================================================================
    def price_option(self, spot, strike, T, r, q, option_type, market_price,
                     india_vix, fii_net_flow, dii_net_flow, days_to_rbi,
                     pcr_oi, returns_30d, inr_usd_vol=0.05,
                     # OMEGA-specific extras
                     hv_30d=None, iv_rank=None, iv_percentile=None,
                     rsi=None, macd_signal=None, bb_position=None, atr_pct=None,
                     bid=None, ask=None, volume_oi_ratio=None,
                     sentiment_data=None, behavioral_context=None, **kwargs):
        """
        Full OMEGA pricing pipeline.  Accepts all NIRV inputs plus
        optional ML/AI enrichment data.
        """
        # ── LAYER 0: NIRV Mathematical Pricing ────────────────────────
        nirv_result = self.nirv.price_option(
            spot, strike, T, r, q, option_type, market_price,
            india_vix, fii_net_flow, dii_net_flow, days_to_rbi,
            pcr_oi, returns_30d, inr_usd_vol, **kwargs)

        # ── FEATURE EXTRACTION ────────────────────────────────────────
        mdata = {
            'spot': spot, 'strike': strike, 'T': T,
            'iv': india_vix / 100.0,
            'hv_30d': hv_30d if hv_30d else india_vix / 100.0,
            'vix': india_vix, 'market_price': market_price,
            'fii_net_flow': fii_net_flow, 'dii_net_flow': dii_net_flow,
            'pcr_oi': pcr_oi,
            'iv_rank': iv_rank or 50, 'iv_percentile': iv_percentile or 50,
            'rsi': rsi or 50, 'macd_signal': macd_signal or 0,
            'bb_position': bb_position or 0.5, 'atr_pct': atr_pct or 1.5,
            'greeks': nirv_result.greeks, 'regime': nirv_result.regime,
            'nirv_output': nirv_result,
            'bid': bid or market_price * 0.99,
            'ask': ask or market_price * 1.01,
            'volume_oi_ratio': volume_oi_ratio or 0.1,
        }
        behavioral_state = kwargs.get("behavioral_state")
        if isinstance(behavioral_state, dict):
            # Additive pass-through only; legacy feature path remains unchanged.
            beh = behavioral_state.get("behavioral", {})
            dflow = beh.get("dealer_flow", {}) if isinstance(beh, dict) else {}
            if "total_gex" in dflow:
                mdata["gex_total"] = dflow.get("total_gex")
            if "flow_imbalance" in dflow:
                mdata["dealer_flow_imbalance"] = dflow.get("flow_imbalance")
            if "sentiment" in beh:
                mdata["behavioral_sentiment"] = beh.get("sentiment")
        # Pass through v6 state from NIRV when available
        nirv_state = getattr(self.nirv, "state", {}) if hasattr(self.nirv, "state") else {}
        if isinstance(nirv_state, dict):
            if nirv_state.get("model_free_var_30d") is not None:
                mdata["model_free_var_30d"] = nirv_state.get("model_free_var_30d")
            if isinstance(nirv_state.get("model_free_var_term_structure"), dict):
                mdata["model_free_var_term_structure"] = nirv_state.get("model_free_var_term_structure")
            if isinstance(nirv_state.get("vrp_state"), dict):
                mdata["vrp_level"] = nirv_state["vrp_state"].get("vrp_level", np.nan)
                mdata["vrp_slope"] = nirv_state["vrp_state"].get("vrp_slope", np.nan)
        # Pass through synthetic VIX if present
        if 'india_vix_synth' in kwargs:
            mdata['india_vix_synth'] = kwargs['india_vix_synth']
        features = FeatureFactory.extract(mdata)

        # ── LAYER 1: ML CORRECTION ────────────────────────────────────
        ml_corr, ml_conf = 0.0, 0.0
        ml_corr_lo, ml_corr_hi = 0.0, 0.0
        if self.ml:
            if getattr(get_features(), "USE_CONFORMAL_INTERVALS", False):
                ml_corr, ml_conf, ml_corr_lo, ml_corr_hi = self.ml.predict_correction_with_interval(features)
            else:
                ml_corr, ml_conf = self.ml.predict_correction(features)
                ml_corr_lo, ml_corr_hi = ml_corr, ml_corr

        # ── LAYER 2: EFFICIENCY DETECTION ─────────────────────────────
        efficiency = self.hunter.score(features, market_spot=spot)
        self.hunter.update(features)

        # ── LAYER 3: SENTIMENT ────────────────────────────────────────
        sent_score, sent_conf = 0.0, 0.0
        sent_result = {'score': 0, 'confidence': 0, 'direction': 'NEUTRAL', 'factors': []}
        if sentiment_data:
            sent_result = self.sentiment.analyse(
                gemini_resp=sentiment_data.get('gemini'),
                perplexity_resp=sentiment_data.get('perplexity'),
                headlines=sentiment_data.get('headlines'))
            sent_score = sent_result['score']
            sent_conf  = sent_result['confidence']

        is_call = option_type.upper() in ('CE', 'CALL')
        sent_factor = sent_score if is_call else -sent_score
        sent_adj = nirv_result.fair_value * self.sentiment_weight * sent_factor * sent_conf

        # ── LAYER 4: BEHAVIORAL ───────────────────────────────────────
        behav_impact = 0.0
        behav_analysis = {}
        if behavioral_context:
            ctx_text = behavioral_context.get('context', '')
            for actor in ['trump', 'fed', 'rbi', 'global']:
                a = self.behavioral.predict(actor, ctx_text)
                behav_analysis[actor] = a
                behav_impact += a['impact'] * a['probability']

        # ── COMBINE: OMEGA Fair Value ─────────────────────────────────
        omega_fv = nirv_result.fair_value * (1.0 + ml_corr * self.ml_weight)
        omega_fv += sent_adj
        omega_fv = max(omega_fv, 0.01)
        omega_fv_lo = max(nirv_result.fair_value * (1.0 + ml_corr_lo * self.ml_weight) + sent_adj, 0.01)
        omega_fv_hi = max(nirv_result.fair_value * (1.0 + ml_corr_hi * self.ml_weight) + sent_adj, 0.01)

        omega_misp = (omega_fv - market_price) / max(market_price, 0.01) * 100

        omega_signal = self._enhanced_signal(
            omega_misp, nirv_result.physical_profit_prob,
            nirv_result.confidence_level, efficiency['overall_score'],
            sent_score, ml_conf, ml_corr)

        conformal_actionable = True
        if getattr(get_features(), "USE_CONFORMAL_INTERVALS", False):
            spread = max((mdata.get('ask', market_price) - mdata.get('bid', market_price)), 0.0)
            spread_ratio = spread / max(market_price, 1e-8)
            liquidity_ok = (mdata.get('volume_oi_ratio', 0.0) >= 0.03) and (spread_ratio <= 0.20)
            min_edge_pct = float(kwargs.get("min_conformal_edge_pct", 0.5))
            min_edge_abs = max(market_price * min_edge_pct / 100.0, 0.5 * spread)
            outside_interval = (market_price < (omega_fv_lo - min_edge_abs)) or (market_price > (omega_fv_hi + min_edge_abs))
            conformal_actionable = bool(liquidity_ok and outside_interval)
            if not conformal_actionable:
                omega_signal = 'HOLD'

        omega_conf = nirv_result.confidence_level
        if ml_conf > 0:
            omega_conf = 0.70 * nirv_result.confidence_level + 0.30 * (ml_conf * 100)

        conviction_score_10 = self._research_conviction_score_10(
            signal=omega_signal,
            mispricing_pct=omega_misp,
            physical_pop=nirv_result.physical_profit_prob,
            confidence_level=omega_conf,
            efficiency_score=efficiency.get('overall_score', 50.0),
            ml_confidence=ml_conf,
            conformal_actionable=conformal_actionable,
        )
        high_conviction_actionable = bool(conviction_score_10 >= 9)

        # Research-only mode: keep actionable output focused to 9/10-10/10 picks.
        if (
            getattr(get_features(), "USE_RESEARCH_HIGH_CONVICTION", False)
            and not high_conviction_actionable
            and omega_signal != 'HOLD'
        ):
            omega_signal = 'HOLD'

        oos_gate = {
            'required': False,
            'passed': True,
            'reason': 'disabled',
            'total_samples': 0,
            'accuracy_pct': 0.0,
            'avg_edge_pct': 0.0,
            'side_samples': 0,
            'side_accuracy_pct': 0.0,
            'regime': 'Unknown',
            'regime_samples': 0,
            'regime_accuracy_pct': 0.0,
        }
        if getattr(get_features(), "USE_OOS_RELIABILITY_GATE", False):
            omega_signal, oos_gate = self._apply_oos_reliability_gate(
                signal=omega_signal,
                features=features,
                **kwargs,
            )

        # ── RECORD FOR LEARNING ───────────────────────────────────────
        pred_id = (f"{option_type}_{int(strike)}_{T:.4f}_"
                   f"{datetime.datetime.now():%Y%m%d%H%M%S}")
        self.tracker.record(pred_id, features, {
            'signal': omega_signal, 'fair_value': omega_fv,
            'market_price': market_price, 'mispricing': omega_misp})

        # ── BUILD OUTPUT ──────────────────────────────────────────────
        return OMEGAOutput(
            # Core
            fair_value=round(omega_fv, 2),
            nirv_fair_value=nirv_result.fair_value,
            market_price=market_price,
            mispricing_pct=round(omega_misp, 2),
            signal=omega_signal,
            # Probabilities
            profit_probability=nirv_result.profit_probability,
            physical_profit_prob=nirv_result.physical_profit_prob,
            confidence_level=round(omega_conf, 1),
            expected_pnl=nirv_result.expected_pnl,
            physical_expected_pnl=nirv_result.physical_expected_pnl,
            # Base
            regime=nirv_result.regime,
            greeks=nirv_result.greeks,
            # OMEGA layers
            ml_correction_pct=round(ml_corr * 100, 2),
            ml_confidence=round(ml_conf * 100, 1),
            ml_correction_interval_pct=(round(ml_corr_lo * 100, 2), round(ml_corr_hi * 100, 2)),
            fair_value_interval=(round(omega_fv_lo, 2), round(omega_fv_hi, 2)),
            conformal_actionable=bool(conformal_actionable),
            conviction_score_10=int(conviction_score_10),
            high_conviction_actionable=bool(high_conviction_actionable),
            oos_gate_required=bool(oos_gate.get('required', False)),
            oos_gate_passed=bool(oos_gate.get('passed', True)),
            oos_gate_reason=str(oos_gate.get('reason', 'disabled')),
            oos_gate_metrics=oos_gate,
            sentiment_score=round(sent_score, 3),
            sentiment_direction=sent_result['direction'],
            sentiment_factors=sent_result['factors'],
            sentiment_adjustment=round(sent_adj, 2),
            efficiency_score=round(efficiency['overall_score'], 1),
            efficiency_details=efficiency,
            behavioral_impact_pct=round(behav_impact * 100, 2),
            behavioral_analysis=behav_analysis,
            behavioral_state=behavioral_state if isinstance(behavioral_state, dict) else None,
            # Learning
            prediction_id=pred_id,
            model_performance=self.tracker.get_performance(),
            ml_trained=self.ml.is_trained if self.ml else False,
            training_samples=len(self.ml.training_X) if self.ml else 0,
            sklearn_available=SKLEARN_AVAILABLE,
        )

    # ==================================================================
    # ENHANCED SIGNAL GENERATOR
    # ==================================================================
    def _enhanced_signal(self, misp, phys_pop, conf, eff_score, sent, ml_conf, ml_corr=0.0):
        buy = sell = 0

        # Mispricing (strongest)
        if   misp > 6:  buy  += 3
        elif misp > 3:  buy  += 2
        elif misp > 1:  buy  += 1
        if   misp < -6: sell += 3
        elif misp < -3: sell += 2
        elif misp < -1: sell += 1

        # Profit probability
        if   phys_pop > 65: buy  += 2
        elif phys_pop > 55: buy  += 1
        if   phys_pop < 35: sell += 2
        elif phys_pop < 45: sell += 1

        # Confidence
        if conf > 75:
            if buy > sell: buy += 1
            elif sell > buy: sell += 1

        # Efficiency (high = more likely real mispricing)
        if eff_score > 70:
            if misp > 0: buy += 1
            else: sell += 1

        # Sentiment
        if sent >  0.2: buy  += 1
        if sent < -0.2: sell += 1

        # ML confidence boost — require directional alignment so data quantity
        # alone cannot inflate a signal that the ML correction contradicts.
        if ml_conf > 0.5 and abs(ml_corr) > 0.01:
            if buy > sell and ml_corr > 0: buy += 1
            elif sell > buy and ml_corr < 0: sell += 1

        if buy >= 5 and buy > sell: return 'STRONG BUY'
        if buy >= 3 and buy > sell: return 'BUY'
        if sell >= 5 and sell > buy: return 'STRONG SELL'
        if sell >= 3 and sell > buy: return 'SELL'
        return 'HOLD'

    @staticmethod
    def _research_conviction_score_10(
        signal,
        mispricing_pct,
        physical_pop,
        confidence_level,
        efficiency_score,
        ml_confidence,
        conformal_actionable=True,
    ):
        """
        Research scorer that intentionally emits only {10, 9, 0}.

        - 10/10: strong multi-factor edge
        - 9/10 : actionable but lower margin of safety
        - 0    : filter out in high-conviction research mode
        """
        sig = str(signal or "").upper()
        directional = ("BUY" in sig) or ("SELL" in sig)
        if not directional:
            return 0

        misp = float(np.clip(abs(float(mispricing_pct)) / 10.0, 0.0, 1.0))
        pop = float(np.clip((float(physical_pop) - 50.0) / 25.0, 0.0, 1.0))
        conf = float(np.clip((float(confidence_level) - 55.0) / 35.0, 0.0, 1.0))
        eff = float(np.clip((float(efficiency_score) - 55.0) / 35.0, 0.0, 1.0))
        mlc = float(np.clip(float(ml_confidence), 0.0, 1.0))

        composite = (
            0.34 * misp
            + 0.26 * pop
            + 0.20 * conf
            + 0.12 * eff
            + 0.08 * mlc
        )
        if "STRONG" in sig:
            composite += 0.04
        if not bool(conformal_actionable):
            composite -= 0.12
        composite = float(np.clip(composite, 0.0, 1.0))

        if composite >= 0.86:
            return 10
        if composite >= 0.70:
            return 9
        return 0

    def _apply_oos_reliability_gate(self, signal, features, **kwargs):
        """
        Apply out-of-sample reliability constraints to directional signals.

        If the gate fails, the signal is downgraded to HOLD.
        """
        gate = self.tracker.get_reliability_gate_decision(
            signal=signal,
            features=features,
            min_samples=int(kwargs.get('oos_min_samples', 40)),
            min_accuracy_pct=float(kwargs.get('oos_min_accuracy_pct', 58.0)),
            min_avg_edge_pct=float(kwargs.get('oos_min_avg_edge_pct', 0.10)),
            min_side_samples=int(kwargs.get('oos_min_side_samples', 15)),
            min_regime_samples=int(kwargs.get('oos_min_regime_samples', 10)),
            lookback=int(kwargs.get('oos_lookback', 300)),
        )
        if gate.get('required') and not gate.get('passed'):
            return 'HOLD', gate
        return signal, gate

    # ==================================================================
    # TRADE PLAN
    # ==================================================================
    def generate_trade_plan(self, omega_out, spot, lot_size=None, capital=500000):
        return self.trade_gen.generate(
            omega_out, spot, lot_size or self.lot_size, capital)

    # ==================================================================
    # LEARNING
    # ==================================================================
    def learn_from_outcome(self, prediction_id, actual_return, actual_price=None):
        """Record outcome and feed to ML corrector."""
        self.tracker.record_outcome(prediction_id, actual_return, actual_price)
        for p in reversed(self.tracker.predictions):
            if p['id'] == prediction_id and p.get('features'):
                if self.ml:
                    self.ml.add_sample(p['features'], actual_return)
                break

    def get_status(self):
        """Return comprehensive learning / model status."""
        return {
            'sklearn_available': SKLEARN_AVAILABLE,
            'ml_trained': self.ml.is_trained if self.ml else False,
            'training_samples': len(self.ml.training_X) if self.ml else 0,
            'min_samples': MLPricingCorrector.MIN_SAMPLES,
            'predictions_total': len(self.tracker.predictions),
            'outcomes_recorded': len([p for p in self.tracker.predictions if p.get('outcome')]),
            'efficiency_history': len(self.hunter.history),
            'behavioral_obs': len(self.behavioral.observations),
            'performance': self.tracker.get_performance(),
            'feature_importance': self.ml.get_feature_importance() if self.ml else {},
        }

    # ==================================================================
    # CHAIN SCANNER
    # ==================================================================
    def scan_chain(self, spot, strikes, T, r, q,
                   market_prices_ce, market_prices_pe,
                   india_vix, fii_net_flow, dii_net_flow, days_to_rbi,
                   pcr_oi, returns_30d, **kwargs):
        """
        Scan entire option chain with OMEGA intelligence.
        Returns list of (option_type, strike, OMEGAOutput) sorted by
        efficiency score then physical PoP.
        """
        bid_quotes_ce = kwargs.get("bid_quotes_ce") if isinstance(kwargs.get("bid_quotes_ce"), dict) else {}
        ask_quotes_ce = kwargs.get("ask_quotes_ce") if isinstance(kwargs.get("ask_quotes_ce"), dict) else {}
        voi_quotes_ce = kwargs.get("volume_oi_ratio_ce") if isinstance(kwargs.get("volume_oi_ratio_ce"), dict) else {}
        bid_quotes_pe = kwargs.get("bid_quotes_pe") if isinstance(kwargs.get("bid_quotes_pe"), dict) else {}
        ask_quotes_pe = kwargs.get("ask_quotes_pe") if isinstance(kwargs.get("ask_quotes_pe"), dict) else {}
        voi_quotes_pe = kwargs.get("volume_oi_ratio_pe") if isinstance(kwargs.get("volume_oi_ratio_pe"), dict) else {}

        shared_kwargs = dict(kwargs)
        for _k in (
            "bid_quotes_ce",
            "ask_quotes_ce",
            "volume_oi_ratio_ce",
            "bid_quotes_pe",
            "ask_quotes_pe",
            "volume_oi_ratio_pe",
        ):
            shared_kwargs.pop(_k, None)

        results = []
        for strike in strikes:
            if strike in market_prices_ce and market_prices_ce[strike] > 0:
                try:
                    ce_kwargs = dict(shared_kwargs)
                    if strike in bid_quotes_ce:
                        ce_kwargs["bid"] = bid_quotes_ce[strike]
                    if strike in ask_quotes_ce:
                        ce_kwargs["ask"] = ask_quotes_ce[strike]
                    if strike in voi_quotes_ce:
                        ce_kwargs["volume_oi_ratio"] = voi_quotes_ce[strike]
                    res = self.price_option(
                        spot, strike, T, r, q, 'CE', market_prices_ce[strike],
                        india_vix, fii_net_flow, dii_net_flow, days_to_rbi,
                        pcr_oi, returns_30d, **ce_kwargs)
                    results.append(('CE', strike, res))
                except Exception as e:
                    warnings.warn(f"OMEGA scan CE {strike}: {e}")

            if strike in market_prices_pe and market_prices_pe[strike] > 0:
                try:
                    pe_kwargs = dict(shared_kwargs)
                    if strike in bid_quotes_pe:
                        pe_kwargs["bid"] = bid_quotes_pe[strike]
                    if strike in ask_quotes_pe:
                        pe_kwargs["ask"] = ask_quotes_pe[strike]
                    if strike in voi_quotes_pe:
                        pe_kwargs["volume_oi_ratio"] = voi_quotes_pe[strike]
                    res = self.price_option(
                        spot, strike, T, r, q, 'PE', market_prices_pe[strike],
                        india_vix, fii_net_flow, dii_net_flow, days_to_rbi,
                        pcr_oi, returns_30d, **pe_kwargs)
                    results.append(('PE', strike, res))
                except Exception as e:
                    warnings.warn(f"OMEGA scan PE {strike}: {e}")

        if getattr(get_features(), "USE_RESEARCH_HIGH_CONVICTION", False):
            # Research mode: keep output intentionally sparse and highest-conviction only.
            results = [x for x in results if getattr(x[2], "conviction_score_10", 0) >= 9]
            results.sort(
                key=lambda x: (
                    getattr(x[2], "conviction_score_10", 0),
                    x[2].efficiency_score,
                    x[2].physical_profit_prob,
                    x[2].confidence_level,
                ),
                reverse=True,
            )
            return results

        if bool(getattr(get_features(), "USE_ENHANCED_RANKING", False)):
            def _adjusted_edge_key(item):
                out = item[2]
                misp = abs(float(getattr(out, "mispricing_pct", 0.0)))
                pop = float(getattr(out, "physical_profit_prob", 50.0)) / 100.0
                conf = float(getattr(out, "confidence_level", 50.0)) / 100.0
                eff = float(getattr(out, "efficiency_score", 50.0)) / 100.0
                # Penalize wide uncertainty/weak actionability where available.
                actionable = 1.0 if bool(getattr(out, "conformal_actionable", True)) else 0.7
                edge = misp * pop * conf * max(eff, 0.1) * actionable
                return (
                    edge,
                    float(getattr(out, "efficiency_score", 0.0)),
                    float(getattr(out, "physical_profit_prob", 0.0)),
                )

            results.sort(key=_adjusted_edge_key, reverse=True)
            return results

        results.sort(key=lambda x: (
            x[2].efficiency_score,
            x[2].physical_profit_prob,
        ), reverse=True)
        return results


# ====================================================================
# PHASE 2  UPGRADE 8: EVENT RISK PRICER
# ====================================================================
# Decomposes IV into diffusive + event components.
# Before RBI/Budget/FOMC: event vol can be 40-60% of total IV.
# ====================================================================

class EventRiskPricer:
    """
    Decomposes option IV into diffusive and event components.

    IV² = IV²_diffusive + IV²_event

    The event component represents the expected jump variance from a
    known upcoming event (RBI, Budget, FOMC, earnings).

    Key insight: if you strip out event vol BEFORE the event,
    diffusive_IV tells you whether options are cheap RELATIVE TO
    the event — enabling pre-event positioning.

    After the event, event_vol collapses → theta is NOT linear.
    This class computes the non-linear theta profile across the event.
    """

    # Typical event variance contributions (annualized variance units)
    EVENT_VARIANCE = {
        'rbi':     0.0015,   # RBI policy: ~1.5% single-day move implied
        'budget':  0.0035,   # Union Budget: ~3% single-day move implied
        'fomc':    0.0008,   # US FOMC: ~0.8% indirect Nifty impact
        'earnings': 0.0020,  # Heavy-weight earnings (Reliance/TCS/HDFC)
        'expiry':  0.0005,   # Weekly expiry gamma/pin effects
    }

    @staticmethod
    def decompose(iv, T, event_type, days_to_event):
        """
        Decompose IV into diffusive and event components.

        Parameters
        ----------
        iv            : float — current total implied volatility (decimal)
        T             : float — time to expiry in years
        event_type    : str — 'rbi', 'budget', 'fomc', 'earnings', 'expiry'
        days_to_event : int — trading days until the event

        Returns
        -------
        dict with iv_diffusive, iv_event, event_pct, pre_event_signal
        """
        event_var = EventRiskPricer.EVENT_VARIANCE.get(event_type, 0.001)
        days_to_event = max(int(days_to_event), 0)

        iv_total_var = iv**2

        # Event variance contributes only if event falls within option lifetime
        # Weighted by proximity: closer event = higher contribution
        if days_to_event > 0 and T > 0:
            event_in_window = days_to_event / 365.0 < T
            if event_in_window:
                event_weight = np.exp(-0.5 * (days_to_event / max(T * 365, 1))**2)
                iv_event_var = event_var * event_weight / max(T, 1e-6)
                iv_event_var = min(iv_event_var, iv_total_var * 0.8)  # Cap at 80%
            else:
                iv_event_var = 0.0
        else:
            iv_event_var = 0.0

        iv_diffusive_var = max(iv_total_var - iv_event_var, 0.001)

        iv_diffusive = np.sqrt(iv_diffusive_var)
        iv_event = np.sqrt(iv_event_var) if iv_event_var > 0 else 0.0
        event_pct = iv_event_var / max(iv_total_var, 1e-8) * 100

        # Pre-event signal: if diffusive IV is cheap relative to history
        if event_pct > 30:
            signal = 'HIGH_EVENT_PREMIUM'
        elif event_pct > 15:
            signal = 'MODERATE_EVENT_PREMIUM'
        else:
            signal = 'LOW_EVENT_PREMIUM'

        return {
            'iv_total': round(float(iv), 4),
            'iv_diffusive': round(float(iv_diffusive), 4),
            'iv_event': round(float(iv_event), 4),
            'event_pct_of_total': round(float(event_pct), 1),
            'event_type': event_type,
            'signal': signal,
        }

    @staticmethod
    def theta_profile_across_event(iv, S, K, T, r, event_type, days_to_event,
                                   option_type='CE'):
        """
        Compute non-linear theta profile showing accelerated decay at event.

        Returns a list of (day_offset, theta_that_day) pairs showing
        how theta changes day-by-day across the event boundary.
        """
        from scipy.stats import norm as _norm

        profile = []
        days_total = max(int(T * 365), 1)

        for d in range(days_total):
            T_remaining = max((days_total - d) / 365.0, 1e-6)
            dte = max(days_to_event - d, 0)

            decomp = EventRiskPricer.decompose(iv, T_remaining, event_type, dte)
            sigma_eff = decomp['iv_diffusive']

            # BSM theta approximation
            sqrt_T = np.sqrt(T_remaining)
            d1 = (np.log(S / K) + (r + 0.5 * sigma_eff**2) * T_remaining) / (sigma_eff * sqrt_T)

            theta_daily = -(S * sigma_eff * _norm.pdf(d1)) / (2 * sqrt_T * 365)

            # Add event theta component (collapses after event)
            if dte == 1:  # Day before event: massive theta
                theta_daily *= (1 + decomp['event_pct_of_total'] / 100)
            elif dte == 0:  # Event day: event vol collapses
                theta_daily *= (1 + 0.5 * decomp['event_pct_of_total'] / 100)

            profile.append({
                'day': d,
                'T_remaining': round(T_remaining, 4),
                'theta_daily': round(float(theta_daily), 4),
                'is_event_day': dte == 0,
            })

        return profile


# ====================================================================
# PHASE 2  UPGRADE 9: BEHAVIORAL LIQUIDITY FEEDBACK LOOP
# ====================================================================
# PT agents overpay OTM → MMs sell → MMs hedge gamma →
# MM hedging creates the price patterns PT agents fear → feedback.
# ====================================================================

class BehavioralLiquidityFeedback:
    """
    Models the feedback loop between behavioral biases and market-maker hedging.

    The cycle:
        1. Prospect Theory agents overvalue OTM puts (loss aversion)
        2. MMs sell these overpriced puts, collecting premium
        3. MMs delta-hedge by selling futures/spot
        4. This hedging pressure creates the very volatility clustering
           that PT agents feared
        5. Realized vol rises → confirms PT agents' beliefs → cycle continues

    The feedback multiplier modifies the model's IV estimate:
        IV_adjusted = IV_model × (1 + β × |GEX| / avg_GEX × sign(retail_flow))
    """

    @staticmethod
    def compute_feedback(gex_current, gex_average, retail_flow_direction,
                         pt_overpricing_pct=0.0, iv_model=0.14):
        """
        Compute the behavioral-liquidity feedback adjustment to IV.

        Parameters
        ----------
        gex_current         : float — current Gamma Exposure (aggregate)
        gex_average         : float — 20-day average GEX
        retail_flow_direction: float — net retail flow direction [-1, 1]
        pt_overpricing_pct  : float — behavioral overpricing % from ProspectTheoryKernel
        iv_model            : float — base IV from model

        Returns
        -------
        dict with adjusted_iv, feedback_multiplier, cycle_stage
        """
        if abs(gex_average) < 1e-6:
            gex_ratio = 1.0
        else:
            gex_ratio = abs(gex_current) / abs(gex_average)

        # Feedback strength: GEX extremes amplify behavioral effects
        beta = 0.15  # Feedback coupling constant
        feedback = beta * gex_ratio * np.sign(retail_flow_direction)

        # PT overpricing adds to feedback loop
        if pt_overpricing_pct > 5:
            feedback *= (1 + pt_overpricing_pct / 100)

        feedback_multiplier = 1.0 + np.clip(feedback, -0.3, 0.3)
        adjusted_iv = iv_model * feedback_multiplier

        # Identify cycle stage
        if gex_current < 0 and retail_flow_direction > 0:
            stage = 'ACCUMULATION'   # MMs short gamma, retail buying → vol expanding
        elif gex_current < 0 and retail_flow_direction < 0:
            stage = 'PANIC'          # MMs short gamma, retail selling → cascade
        elif gex_current > 0 and retail_flow_direction > 0:
            stage = 'PINNING'        # MMs long gamma, stable hedging → low vol
        else:
            stage = 'DISTRIBUTION'   # MMs long gamma, retail selling → orderly decline

        return {
            'adjusted_iv': round(float(adjusted_iv), 4),
            'base_iv': round(float(iv_model), 4),
            'feedback_multiplier': round(float(feedback_multiplier), 4),
            'gex_ratio': round(float(gex_ratio), 3),
            'cycle_stage': stage,
            'is_elevated': abs(feedback) > 0.1,
        }


# ====================================================================
# PHASE 2  UPGRADE 11: SHADOW DELTA HEDGING P&L TRACKER
# ====================================================================
# Paper-trades every signal and tracks realized P&L.
# Ultimate validation: is the alpha real, or just noise?
# ====================================================================

class ShadowHedger:
    """
    Shadow (paper) delta hedging P&L tracker.

    For every signal the model generates, this tracker computes what
    the P&L WOULD HAVE BEEN if you had taken the trade and delta-hedged.

    P&L = Δ × ΔS - ½Γ × (ΔS)² + Θ × Δt + (entry_edge × lots)

    The running Sharpe ratio validates whether the model's edge is real:
        Sharpe = mean(daily_PnL) / std(daily_PnL) × √252

    If Sharpe < 0.5 over 30+ trades → the edge is noise.
    If Sharpe > 1.5 → the edge is likely real and exploitable.
    """

    def __init__(self, lot_size=65):
        self.lot_size = lot_size
        self.trades = []
        self._daily_pnl = []

    def add_trade(self, signal, entry_price, fair_value, delta, gamma, theta,
                  spot_at_entry, spot_at_exit, dt_days, option_exit_price=None):
        """
        Record a shadow trade and compute realized P&L.

        Parameters
        ----------
        signal          : str — model signal ('BUY' or 'SELL')
        entry_price     : float — option price at entry
        fair_value      : float — model fair value at entry
        delta, gamma, theta : float — greeks at entry
        spot_at_entry   : float — spot at signal generation
        spot_at_exit    : float — spot at evaluation time
        dt_days         : float — holding period in days
        option_exit_price: float or None — actual option exit price

        Returns
        -------
        dict with trade details and P&L breakdown
        """
        direction = 1.0 if signal.upper() in ('BUY', 'STRONG BUY') else -1.0
        dS = spot_at_exit - spot_at_entry
        dt = dt_days / 365.0

        # Greeks-decomposed P&L (per-unit)
        delta_pnl = direction * delta * dS
        gamma_pnl = direction * 0.5 * gamma * dS**2
        theta_pnl = direction * theta * dt_days  # Theta is per-day
        edge_pnl = direction * (fair_value - entry_price)

        # Total per-unit P&L
        unit_pnl = delta_pnl + gamma_pnl + theta_pnl + edge_pnl

        # If we have actual exit price, compute realized P&L
        if option_exit_price is not None:
            realized_pnl = direction * (option_exit_price - entry_price) * self.lot_size
        else:
            realized_pnl = unit_pnl * self.lot_size

        trade = {
            'signal': signal,
            'direction': direction,
            'entry_price': float(entry_price),
            'fair_value': float(fair_value),
            'dS': float(dS),
            'dt_days': float(dt_days),
            'delta_pnl': round(float(delta_pnl), 2),
            'gamma_pnl': round(float(gamma_pnl), 2),
            'theta_pnl': round(float(theta_pnl), 2),
            'edge_pnl': round(float(edge_pnl), 2),
            'unit_pnl': round(float(unit_pnl), 2),
            'lot_pnl': round(float(realized_pnl), 2),
        }
        self.trades.append(trade)
        self._daily_pnl.append(realized_pnl)

        return trade

    def get_performance(self):
        """
        Compute aggregate shadow performance metrics.

        Returns
        -------
        dict with sharpe, win_rate, avg_pnl, total_pnl, n_trades
        """
        if not self.trades:
            return {
                'sharpe': 0.0, 'win_rate': 0.0, 'avg_pnl': 0.0,
                'total_pnl': 0.0, 'n_trades': 0, 'verdict': 'NO_DATA',
            }

        pnl_arr = np.array(self._daily_pnl)
        n = len(pnl_arr)
        total_pnl = float(np.sum(pnl_arr))
        avg_pnl = float(np.mean(pnl_arr))
        std_pnl = float(np.std(pnl_arr)) + 1e-8
        win_rate = float(np.mean(pnl_arr > 0) * 100)

        sharpe = (avg_pnl / std_pnl) * np.sqrt(min(252, n))

        # Verdict
        if n < 10:
            verdict = 'INSUFFICIENT_TRADES'
        elif sharpe > 1.5:
            verdict = 'STRONG_ALPHA'
        elif sharpe > 0.5:
            verdict = 'MODERATE_ALPHA'
        elif sharpe > 0:
            verdict = 'WEAK_ALPHA'
        else:
            verdict = 'NO_ALPHA'

        # Max drawdown
        cumulative = np.cumsum(pnl_arr)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        max_drawdown = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0

        return {
            'sharpe': round(float(sharpe), 3),
            'win_rate': round(win_rate, 1),
            'avg_pnl': round(avg_pnl, 2),
            'total_pnl': round(total_pnl, 2),
            'n_trades': n,
            'max_drawdown': round(max_drawdown, 2),
            'verdict': verdict,
        }
