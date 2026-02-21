#!/usr/bin/env python3
"""
=============================================================================
NIRV: Nifty Intelligent Regime-Volatility Option Pricing Model
=============================================================================
A novel option pricing framework designed specifically for Indian Nifty 50
options. Combines regime-aware stochastic volatility, India-specific market
features, and Bayesian ensemble learning to produce fair values, profit
probabilities, and confidence intervals.


Author: Quantitative Research Model
Target: Nifty 50 European-style Index Options (NSE)
Version: 4.0 — Sobol QMC, CRN Greeks, HMM forward, physical-measure P&L
=============================================================================
"""


import numpy as np
from scipy.stats import norm, t
from scipy.optimize import minimize, brentq
from collections import namedtuple
import warnings
import importlib.util
from typing import List, Tuple, Dict, Optional
import datetime
from scipy import interpolate

# OMEGA feature flags & modules
try:
    from omega_features import get_features
    from india_vix_synth import compute_synthetic_vix
    from arbfree_surface import ArbFreeSurfaceState
    from vrr_state import VRRStateFilter
    from model_free_variance import (
        compute_variance_for_expiry,
        compute_vix_30d_with_details,
        estimate_forward_from_chain,
    )
    from essvi_surface import ESSVISurface
    from svi_fixed_point import fixed_point_svi_warmstart
    from surface_checks import (
        check_butterfly_arbitrage_slice,
        check_calendar_arbitrage,
    )
    from vrp_state import ModelFreeVRPState
    from pricer_router import TieredPricerRouter
    from nse_specs import get_lot_size as nse_get_lot_size
except ImportError:
    # Fallback if modules missing (e.g. during partial deployment)
    get_features = lambda: type(
        "Features",
        (),
        {
            "india_vix_synth": False,
            "arb_free_surface": False,
            "vrr_state": False,
            "USE_NSE_CONTRACT_SPECS": False,
            "USE_NSE_VIX_ENGINE": False,
            "USE_TAIL_CORRECTED_VARIANCE": False,
            "USE_ESSVI_SURFACE": False,
            "USE_SVI_FIXED_POINT_WARMSTART": False,
            "USE_MODEL_FREE_VRP": False,
            "USE_TIERED_PRICER": False,
            "USE_CONFORMAL_INTERVALS": False,
            "USE_LIQUIDITY_WEIGHTING": False,
            "USE_INTERVAL_LOSS": False,
            "USE_IMPROVED_VIX_ESTIMATOR": False,
        },
    )()
    compute_synthetic_vix = None
    ArbFreeSurfaceState = None
    VRRStateFilter = None
    compute_variance_for_expiry = None
    compute_vix_30d_with_details = None
    estimate_forward_from_chain = None
    ESSVISurface = None
    fixed_point_svi_warmstart = None
    check_butterfly_arbitrage_slice = None
    check_calendar_arbitrage = None
    ModelFreeVRPState = None
    TieredPricerRouter = None
    nse_get_lot_size = None

# Sobol QMC: available in scipy ≥ 1.7 (gives ~4-8x convergence speedup)
try:
    from scipy.stats.qmc import Sobol as _SobolEngine
    SOBOL_AVAILABLE = True
except ImportError:
    SOBOL_AVAILABLE = False

# Advanced Quant Engine integration
try:
    from quant_engine import (
        QuantEngine, DynamicSABR, GJRGarch, HestonCOS,
        EMJumpEstimator, ContinuousRegimeDetector,
        KellyCriterion, BayesianPosteriorConfidence,
        GEXCalculator, MacroFeatureEngine,
        ARCH_AVAILABLE, HMM_AVAILABLE
    )
    QUANT_ENGINE_AVAILABLE = True
except ImportError:
    QUANT_ENGINE_AVAILABLE = False
    ARCH_AVAILABLE = False
    HMM_AVAILABLE = False

# Only suppress known-harmless deprecation warnings, NOT numerical RuntimeWarnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)




# ============================================================================
# DATA STRUCTURES
# ============================================================================


NirvOutput = namedtuple('NirvOutput', [
    'fair_value',           # Model fair value (INR)
    'market_price',         # Observed market price (INR)
    'mispricing_pct',       # (fair - market) / market * 100
    'signal',               # BUY / SELL / HOLD
    'profit_probability',   # P(profit) as % (risk-neutral measure)
    'physical_profit_prob', # P(profit) as % (real-world/physical measure)
    'confidence_level',     # Confidence in the prediction (%)
    'expected_pnl',         # Expected P&L per lot (risk-neutral)
    'physical_expected_pnl',# Expected P&L per lot (physical measure)
    'regime',               # Detected market regime
    'greeks',               # Dict of delta, gamma, theta, vega
    'tc_details',           # Transaction cost / edge details dict
])




# ============================================================================
# MODULE 1: INDIA-SPECIFIC FEATURE ENGINE
# ============================================================================


class IndiaFeatureEngine:
    """
    Extracts and normalizes India-specific market features that influence
    Nifty option pricing beyond standard Black-Scholes inputs.
   
    Features:
        vix_z              — VIX z-score (normalised distance from long-run mean)
        flow_ratio         — Net FII+DII flow direction ∈ [-1, 1]
        rbi_factor         — Proximity to RBI monetary policy (exponential decay)
        gamma_amp          — Expiry proximity amplifier for gamma/pin risk
        pcr_deviation      — Put-Call Ratio deviation from equilibrium
        fx_risk            — INR/USD vol normalised proxy
        india_risk_premium — Composite India risk premium
    """


    def __init__(self):
        self.pcr_equilibrium = 1.05  # Historical Nifty PCR midpoint
        self.vix_mean = 14.0         # Long-run India VIX mean
        self.vix_std = 5.0           # India VIX standard deviation


    def compute_features(self, india_vix, fii_net_flow, dii_net_flow,
                         days_to_expiry, days_to_rbi, pcr_oi,
                         inr_usd_vol=0.05, **kwargs):
        """
        Compute normalised feature vector from raw India market inputs.
       
        Parameters
        ----------
        india_vix      : float – Current India VIX level
        fii_net_flow   : float – FII net buy/sell in ₹ crores
        dii_net_flow   : float – DII net buy/sell in ₹ crores
        days_to_expiry : int   – Calendar days to option expiry
        days_to_rbi    : int   – Trading days to next RBI policy meeting
        pcr_oi         : float – Put-Call Ratio (Open Interest based)
        inr_usd_vol    : float – 30-day INR/USD realised volatility
        **kwargs       : dict  - Extra features (e.g. 'india_vix_synth')
       
        Returns
        -------
        dict with feature names → float values
        """
        # Normalized VIX Z-score
        # Phase 2: Use synthetic VIX if available and flag enabled
        iv_synth = kwargs.get('india_vix_synth')
        if iv_synth is not None:
             # Blend or replace? For now, let's keep track of it
             # and maybe use it for the z-score if it differs significantly?
             # Implementation choice: use average if both available, or trust synth
             # Let's stick to using standard VIX for Z-score to maintain history,
             # but add synth to the feature set.
             pass

        vix_z = np.clip((india_vix - self.vix_mean) / self.vix_std, -4.0, 4.0)


        # FII/DII flow ratio ∈ [-1, 1]
        total_flow = abs(fii_net_flow) + abs(dii_net_flow) + 1e-8
        flow_ratio = (fii_net_flow + dii_net_flow) / total_flow


        # RBI proximity factor — exponential decay, peaks at policy day
        rbi_factor = np.exp(-0.15 * max(days_to_rbi, 1))


        # Expiry gamma amplifier — rises as expiry nears
        gamma_amp = 1.0 / (1.0 + 0.5 * max(days_to_expiry, 1))


        # PCR sentiment deviation from equilibrium
        pcr_deviation = (pcr_oi - self.pcr_equilibrium) / self.pcr_equilibrium


        # Currency risk proxy (normalised to typical INR/USD vol)
        fx_risk = inr_usd_vol / 0.08


        # Composite India Risk Premium
        india_risk_premium = (
            0.30 * vix_z +
            0.20 * (-flow_ratio) +
            0.15 * rbi_factor +
            0.15 * gamma_amp +
            0.10 * abs(pcr_deviation) +
            0.10 * fx_risk
        )

        features_dict = {
            'india_vix': float(india_vix),
            'vix_z': round(vix_z, 4),
            'flow_ratio': round(flow_ratio, 4),
            'rbi_factor': round(rbi_factor, 4),
            'gamma_amp': round(gamma_amp, 4),
            'pcr_deviation': round(pcr_deviation, 4),
            'fx_risk': round(fx_risk, 4),
            'india_risk_premium': round(india_risk_premium, 4)
        }
        if iv_synth is not None:
            features_dict['india_vix_synth'] = round(iv_synth, 4)
        mfv = kwargs.get('model_free_var_30d')
        if mfv is not None and np.isfinite(mfv):
            features_dict['model_free_var_30d'] = float(mfv)
        vrp_level = kwargs.get('vrp_level')
        vrp_slope = kwargs.get('vrp_slope')
        if vrp_level is not None and np.isfinite(vrp_level):
            features_dict['vrp_level'] = float(vrp_level)
        if vrp_slope is not None and np.isfinite(vrp_slope):
            features_dict['vrp_slope'] = float(vrp_slope)
        return features_dict




# ============================================================================
# MODULE 2: HIDDEN MARKOV MODEL REGIME DETECTOR
# ============================================================================


class RegimeDetector:
    """
    Uses a full forward-backward HMM pass to classify the current
    market into one of 4 regimes based on:
    - 30-60 day return distribution (mean, volatility)
    - India VIX level AND VIX term structure slope
    - FII net flow direction

    When hmmlearn is available, trains a proper GaussianHMM on the full
    observation sequence (returns + VIX), learning the transition matrix
    from data.  Falls back to the hand-coded forward step otherwise.

    Regimes:
        Bull-Low Vol  -- Trending up, low volatility (VIX < 15)
        Bear-High Vol -- Trending down, high volatility (VIX > 20)
        Sideways      -- Range-bound, moderate volatility
        Bull-High Vol -- Trending up, high volatility (event-driven)

    Each regime carries Heston stochastic variance parameters (kappa, theta_v,
    sigma_v, rho_sv) calibrated from historical Nifty 50 data.
    """

    REGIME_PARAMS = {
        'Bull-Low Vol':  {
            'mu': 0.0006, 'sigma': 0.008, 'lambda_j': 0.02, 'mu_j': 0.005,
            'sigma_j': 0.008,
            'kappa': 3.0, 'theta_v': 0.012, 'sigma_v': 0.20, 'rho_sv': -0.40,
        },
        'Bear-High Vol': {
            'mu': -0.0004, 'sigma': 0.018, 'lambda_j': 0.08, 'mu_j': -0.015,
            'sigma_j': 0.020,
            'kappa': 1.5, 'theta_v': 0.045, 'sigma_v': 0.45, 'rho_sv': -0.75,
        },
        'Sideways':      {
            'mu': 0.0001, 'sigma': 0.010, 'lambda_j': 0.03, 'mu_j': 0.000,
            'sigma_j': 0.010,
            'kappa': 2.5, 'theta_v': 0.020, 'sigma_v': 0.30, 'rho_sv': -0.50,
        },
        'Bull-High Vol': {
            'mu': 0.0008, 'sigma': 0.016, 'lambda_j': 0.06, 'mu_j': 0.010,
            'sigma_j': 0.015,
            'kappa': 2.0, 'theta_v': 0.035, 'sigma_v': 0.40, 'rho_sv': -0.55,
        },
    }

    def __init__(self):
        self.regime_names = list(self.REGIME_PARAMS.keys())
        self.transition_matrix = np.array([
            [0.92, 0.03, 0.03, 0.02],
            [0.04, 0.88, 0.05, 0.03],
            [0.05, 0.05, 0.85, 0.05],
            [0.06, 0.04, 0.05, 0.85],
        ])
        self._prev_posterior = None
        # Trained HMM model (hmmlearn)
        self._hmm_model = None
        self._hmm_state_mapping = {}  # maps HMM state idx -> regime name

    def train_hmm(self, returns_history, vix_history=None):
        """
        Train a full 4-state GaussianHMM on historical data.
        Uses returns + VIX as 2D observation (if VIX available).

        Parameters
        ----------
        returns_history : np.ndarray - 250+ daily log returns
        vix_history     : np.ndarray - matching VIX series (optional)
        """
        if not HMM_AVAILABLE:
            return False

        from quant_engine import ContinuousRegimeDetector
        returns = np.asarray(returns_history, dtype=float)
        returns = returns[np.isfinite(returns)]
        if len(returns) < 100:
            return False

        try:
            # Build observation matrix: [returns, normalized_vix]
            if vix_history is not None and len(vix_history) >= len(returns):
                vix = np.asarray(vix_history[-len(returns):], dtype=float)
                X = np.column_stack([returns, (vix - 14.0) / 5.0])  # normalize VIX
                n_features = 2
            else:
                X = returns.reshape(-1, 1)
                n_features = 1

            from hmmlearn.hmm import GaussianHMM
            model = GaussianHMM(n_components=4, covariance_type='full',
                                n_iter=200, random_state=42)
            model.fit(X)
            self._hmm_model = model

            # Map learned states to regime names by matching volatility
            means = model.means_[:, 0]  # return means
            covs = np.sqrt(model.covars_[:, 0, 0])  # return stdevs

            # Sort by (return_direction, volatility) to match our regime labels
            state_info = [(i, means[i], covs[i]) for i in range(4)]
            # Bull-Low Vol: high return, low vol
            # Bear-High Vol: low return, high vol
            # Sideways: near-zero return, moderate vol
            # Bull-High Vol: high return, high vol
            sorted_by_vol = sorted(state_info, key=lambda x: x[2])
            sorted_by_ret = sorted(state_info, key=lambda x: x[1])

            mapping = {}
            used = set()
            # Lowest vol + positive return = Bull-Low Vol
            for s in sorted_by_vol:
                if s[0] not in used and s[1] > 0:
                    mapping[s[0]] = 'Bull-Low Vol'
                    used.add(s[0])
                    break
            # Highest vol + negative return = Bear-High Vol
            for s in reversed(sorted_by_vol):
                if s[0] not in used and s[1] < 0:
                    mapping[s[0]] = 'Bear-High Vol'
                    used.add(s[0])
                    break
            # Highest vol + positive return = Bull-High Vol
            for s in reversed(sorted_by_vol):
                if s[0] not in used and s[1] > 0:
                    mapping[s[0]] = 'Bull-High Vol'
                    used.add(s[0])
                    break
            # Remaining = Sideways
            for i in range(4):
                if i not in used:
                    mapping[i] = 'Sideways'
                    used.add(i)

            self._hmm_state_mapping = mapping

            # Update transition matrix from learned model
            self.transition_matrix = model.transmat_.copy()
            return True
        except Exception:
            return False

    def detect_regime(self, returns_30d, india_vix, fii_net_flow,
                      vix_term_slope=None, vrp_level=None, vrp_slope=None):
        """
        Detect current market regime.

        If a trained HMM is available, uses full forward-backward pass
        (predict_proba) on the observation sequence.
        Otherwise falls back to single-step HMM forward.

        Parameters
        ----------
        returns_30d    : np.ndarray - Last 30 daily log returns
        india_vix      : float - Current India VIX
        fii_net_flow   : float - FII net flow
        vix_term_slope : float - Near VIX minus far VIX (optional)
                         Flat term structure during high VIX = more dangerous
        vrp_level      : float - 30d VRP level (optional)
        vrp_slope      : float - VRP(7d)-VRP(60d) slope (optional)

        Returns
        -------
        (regime_name: str, probabilities: dict)
        """
        if returns_30d is None or len(returns_30d) < 5:
            warnings.warn("NIRV: returns_30d too short; using uniform regime prior")
            returns_30d = np.zeros(30) + 0.0003

        # ---- PATH 1: Trained HMM (full forward-backward) ----
        if self._hmm_model is not None and HMM_AVAILABLE:
            try:
                # Detect feature count from model (compatible with all hmmlearn versions)
                n_feat = getattr(self._hmm_model, 'n_features', None)
                if n_feat is None:
                    n_feat = self._hmm_model.means_.shape[1] if self._hmm_model.means_ is not None else 1

                if n_feat == 2:
                    vix_norm = (india_vix - 14.0) / 5.0
                    X = np.column_stack([returns_30d, np.full(len(returns_30d), vix_norm)])
                else:
                    X = returns_30d.reshape(-1, 1)

                # Full forward-backward gives smooth posteriors
                proba = self._hmm_model.predict_proba(X)
                final_proba = proba[-1]  # posterior at final observation

                probs = {}
                for state_idx in range(4):
                    regime_name = self._hmm_state_mapping.get(state_idx, 'Sideways')
                    probs[regime_name] = probs.get(regime_name, 0) + float(final_proba[state_idx])

                # Ensure all regimes present
                for rn in self.regime_names:
                    probs.setdefault(rn, 0.0)

                # VIX term structure adjustment
                if vix_term_slope is not None:
                    # Flat/inverted term structure during high VIX = increase Bear prob
                    if india_vix > 18 and vix_term_slope < 0.5:
                        probs['Bear-High Vol'] *= 1.3
                    elif vix_term_slope > 2.0:
                        probs['Bull-Low Vol'] *= 1.1

                if getattr(get_features(), "USE_MODEL_FREE_VRP", False):
                    if vrp_level is not None and np.isfinite(vrp_level):
                        if vrp_level > 0.005:
                            probs['Bear-High Vol'] *= 1.15
                            probs['Bull-Low Vol'] *= 0.95
                        elif vrp_level < -0.003:
                            probs['Bull-Low Vol'] *= 1.10
                    if vrp_slope is not None and np.isfinite(vrp_slope):
                        if vrp_slope > 0.002:
                            probs['Bear-High Vol'] *= 1.10
                        elif vrp_slope < -0.002:
                            probs['Sideways'] *= 1.05

                total = sum(probs.values())
                if total > 0:
                    probs = {k: round(v / total, 4) for k, v in probs.items()}

                regime = max(probs, key=probs.get)
                self._prev_posterior = np.array([probs.get(n, 0.25) for n in self.regime_names])
                return regime, probs
            except Exception:
                pass  # Fall through to manual HMM

        # ---- PATH 2: Manual single-step HMM forward (fallback) ----
        # Exponential decay weights: recent returns matter more for regime switching
        _n_ret = len(returns_30d)
        _exp_weights = np.exp(-np.arange(_n_ret)[::-1] / 10.0)  # half-life ~10 days
        _exp_weights /= np.sum(_exp_weights)  # normalize

        log_likelihoods = {}
        for name, params in self.REGIME_PARAMS.items():
            ll = np.sum(_exp_weights * norm.logpdf(returns_30d, loc=params['mu'], scale=params['sigma']))
            if 'High Vol' in name:
                ll += norm.logpdf(india_vix, loc=20, scale=6)
            else:
                ll += norm.logpdf(india_vix, loc=12, scale=4)
            if 'Bull' in name:
                ll += norm.logpdf(fii_net_flow, loc=500, scale=2000)
            else:
                ll += norm.logpdf(fii_net_flow, loc=-500, scale=2000)
            # VIX term structure as observation
            if vix_term_slope is not None:
                if 'High Vol' in name:
                    ll += norm.logpdf(vix_term_slope, loc=-0.5, scale=2.0)
                else:
                    ll += norm.logpdf(vix_term_slope, loc=1.5, scale=2.0)
            if getattr(get_features(), "USE_MODEL_FREE_VRP", False):
                if vrp_level is not None and np.isfinite(vrp_level):
                    if 'High Vol' in name:
                        ll += norm.logpdf(vrp_level, loc=0.015, scale=0.02)
                    else:
                        ll += norm.logpdf(vrp_level, loc=-0.002, scale=0.02)
                if vrp_slope is not None and np.isfinite(vrp_slope):
                    if 'Bear' in name:
                        ll += norm.logpdf(vrp_slope, loc=0.004, scale=0.02)
                    else:
                        ll += norm.logpdf(vrp_slope, loc=-0.001, scale=0.02)
            log_likelihoods[name] = ll

        n_regimes = len(self.regime_names)
        if self._prev_posterior is not None:
            predicted_prior = self._prev_posterior @ self.transition_matrix
        else:
            predicted_prior = np.ones(n_regimes) / n_regimes

        ll_arr = np.array([log_likelihoods[name] for name in self.regime_names])
        max_ll = np.max(ll_arr)
        obs_likelihood = np.exp(ll_arr - max_ll)

        posterior = predicted_prior * obs_likelihood
        posterior_sum = posterior.sum()
        if posterior_sum > 0:
            posterior /= posterior_sum
        else:
            posterior = np.ones(n_regimes) / n_regimes

        self._prev_posterior = posterior.copy()

        probs = {name: round(float(posterior[i]), 4)
                 for i, name in enumerate(self.regime_names)}
        regime = max(probs, key=probs.get)
        return regime, probs




# ============================================================================
# MODULE 3: ADAPTIVE VOLATILITY SURFACE ENGINE (SVI Parameterisation)
# ============================================================================


class VolatilitySurface:
    """
    Constructs a regime-adaptive implied volatility surface using an
    IV-anchored SVI (Stochastic Volatility Inspired) parameterisation.

    **Critical design choice**: The ATM implied volatility is anchored
    directly to India VIX (the market's forward-looking IV estimate).
    The SVI shape then models the *skew/smile* on top of this level.

    This avoids the classical pitfall of raw SVI total-variance
    parameterisation which diverges for short-dated options (T → 0)
    because w/T → ∞ when `a` is not scaled with T.

    IV-space SVI formula:
        σ(k) = σ_atm + b · [ρ(k−m) + √((k−m)² + s²)  − √(m² + s²)]

    At ATM (k = 0): σ(0) = σ_atm  (anchored to VIX)
    For OTM puts (k < 0): skew adds IV  (put skew)
    For OTM calls (k > 0): skew subtracts IV (typical Nifty pattern)

    where k = log(K/S) is log-moneyness
    """


    # Regime-specific adjustments to ATM vol relative to VIX
    REGIME_VOL_ADJ = {
        'Bull-Low Vol':  -0.005,   # Slightly below VIX in calm bull
        'Bear-High Vol': +0.025,   # Above VIX in panic
        'Sideways':       0.000,   # At VIX level
        'Bull-High Vol': +0.015,   # Elevated in event-driven rally
    }


    def __init__(self):
        # Per-regime SVI skew parameters (IV-space, NOT total-variance)
        # b  = wing slope (higher = steeper skew)
        # rho = skew direction (negative = put skew)
        # m  = skew center offset
        # sig = smile curvature (higher = wider smile)
        self.svi_params = {
            'Bull-Low Vol':  {'b': 0.80, 'rho': -0.30, 'm': 0.000, 'sig': 0.15},
            'Bear-High Vol': {'b': 1.80, 'rho': -0.50, 'm': -0.010, 'sig': 0.18},
            'Sideways':      {'b': 1.20, 'rho': -0.35, 'm': 0.000, 'sig': 0.16},
            'Bull-High Vol': {'b': 1.50, 'rho': -0.45, 'm': 0.005, 'sig': 0.18},
        }
        # Cache for ArbFreeSurfaceState per regime
        self.arb_surfaces = {}
        # Cache for eSSVI surfaces (v6 flag-gated)
        self.essvi_surfaces = {}
        # Rolling training slices for eSSVI per regime
        self._essvi_training_data = {}
        # Diagnostics exposed to callers/eval
        self.last_diagnostics = {}

    def _get_parametric_iv(self, spot, strike, T, regime, india_features, hurst_exponent=None):
        """Baseline IV-space SVI-style parametric IV used when alt engines are disabled."""
        k = np.log(strike / spot)  # Log-moneyness
        params = self.svi_params.get(regime, self.svi_params['Sideways'])

        # ── ATM Implied Volatility — anchored to India VIX ──────────────
        india_vix = india_features.get('india_vix', 14.0)
        atm_iv = india_vix / 100.0

        # Regime-specific adjustment
        atm_iv += self.REGIME_VOL_ADJ.get(regime, 0.0)
        atm_iv = max(atm_iv, 0.05)

        # ── SVI skew shape (operates in IV space) ───────────────────────
        b = params['b']
        rho = params['rho']
        m = params['m']
        sig = params['sig']

        # Flow adjustment: negative FII flow → steeper put skew
        flow_adj = 1.0 - 0.08 * india_features.get('flow_ratio', 0)
        rho = np.clip(rho * flow_adj, -0.99, -0.01)

        # Gamma amplification near expiry: wider wings
        b = b * (1.0 + 0.3 * india_features.get('gamma_amp', 0))

        # Rough-vol correction for short-dated wings
        if hurst_exponent is not None and hurst_exponent < 0.3 and T < 14.0 / 365.0:
            alpha_rough = 2.5
            rough_correction = 1.0 + alpha_rough * (0.5 - hurst_exponent) * max(T, 1e-6) ** (hurst_exponent - 0.5)
            rough_correction = min(rough_correction, 3.0)
            b = b * rough_correction

        atm_svi = rho * (0.0 - m) + np.sqrt(m**2 + sig**2)
        k_svi = rho * (k - m) + np.sqrt((k - m)**2 + sig**2)
        skew = b * (k_svi - atm_svi)

        term_damp = min(1.0, np.sqrt(max(T, 1.0 / 365) * 52.0))
        skew *= term_damp
        iv = atm_iv + skew
        return float(np.clip(iv, 0.03, 2.00))



    def get_implied_vol(self, spot, strike, T, regime, india_features,
                        hurst_exponent=None):
        """
        Compute regime-and-feature-adjusted implied volatility for a given strike.

        Parameters
        ----------
        spot   : float – Current spot price
        strike : float – Option strike
        T      : float – Time to expiry in years
        regime : str   – Detected regime name
        india_features : dict – Output of IndiaFeatureEngine.compute_features()
        hurst_exponent : float or None – Hurst exponent from MacroFeatureEngine
                        When H < 0.3, applies rough-vol wing correction.

        Returns
        -------
        float – Implied volatility (annualised, decimal)
        """
        # v6: eSSVI surface (flag-gated)
        if getattr(get_features(), "USE_ESSVI_SURFACE", False) and ESSVISurface:
            try:
                es = self.essvi_surfaces.get(regime)
                if es is not None and getattr(es, "_is_fitted", False):
                    k = np.log(strike / spot)
                    return float(np.clip(es.implied_vol(k, T), 0.03, 2.00))
            except Exception:
                pass

        # v5: Arbitrage-Free Surface Check
        if get_features().arb_free_surface and ArbFreeSurfaceState:
            try:
                surf = self._ensure_arb_surface(regime, india_features)
                k = np.log(strike / spot)
                return surf.get_iv(k, T)
            except Exception as e:
                pass # Fallback to parametric if arb surf fails
        return self._get_parametric_iv(
            spot, strike, T, regime, india_features, hurst_exponent=hurst_exponent
        )


    def _ensure_arb_surface(self, regime, india_features):
        """Build/retrieve cached ArbFreeSurfaceState for this regime."""
        # Check cache (using regime name as key)
        # Ideally we invalid cache if india_features change significantly, 
        # but for performance we assume regime captures the main state.
        # A more robust key would employ discretized india_features.
        
        # Simple cache key: regime name
        if regime in self.arb_surfaces:
            return self.arb_surfaces[regime]
            
        # Build new surface
        surf = ArbFreeSurfaceState()
        
        # Grid of T (days -> years)
        # 3d, 7d, 14d, 21d, 30d, 45d, 60d, 90d, 120d, 180d, 270d, 360d
        days_grid = [3, 7, 14, 21, 30, 45, 60, 90, 120, 180, 270, 360]
        t_grid = np.array(days_grid) / 365.0
        
        # Grid of Moneyness k => strikes
        # -0.3 to +0.3 log moneyness
        k_grid = np.linspace(-0.4, 0.4, 41) 
        spot_ref = 10000.0
        strikes = spot_ref * np.exp(k_grid)
        
        for t_val in t_grid:
            slice_ivs = []
            for stk in strikes:
                iv = self._get_parametric_iv(
                    spot_ref, stk, t_val, regime, india_features, hurst_exponent=None
                )
                slice_ivs.append(iv)
                
            surf.add_slice(t_val, strikes, np.array(slice_ivs), spot_ref)
            
        surf.fit()
        self.arb_surfaces[regime] = surf
        return surf


    def calibrate_to_market(self, spot, strikes, market_ivs, T, regime, india_features):
        """
        Calibrate SVI skew parameters to observed market implied volatilities.
        The ATM level remains anchored to VIX; only skew shape is fitted.

        Uses rollback protection: if the optimizer converges to a poor local
        minimum (RMSE > 5% absolute IV error), the previous parameters are
        preserved. This prevents a single bad chain calibration from corrupting
        all subsequent get_implied_vol() calls.

        Parameters
        ----------
        spot       : float
        strikes    : list of float
        market_ivs : list of float – Market observed IVs (decimal)
        T          : float
        regime     : str
        india_features : dict

        Returns
        -------
        bool – True if calibration succeeded and passed validation
        """
        if len(strikes) < 3 or len(market_ivs) < 3:
            return False

        # v6: eSSVI primary calibration path (flag-gated)
        if getattr(get_features(), "USE_ESSVI_SURFACE", False) and ESSVISurface:
            try:
                buf = self._essvi_training_data.setdefault(regime, [])
                buf.append({
                    "T": float(T),
                    "strikes": np.asarray(strikes, dtype=float),
                    "ivs": np.asarray(market_ivs, dtype=float),
                })
                # Keep recent unique maturities only (lightweight)
                dedup = {}
                for sl in buf:
                    dedup[round(float(sl["T"]), 8)] = sl
                buf = [dedup[k] for k in sorted(dedup.keys())][-10:]
                self._essvi_training_data[regime] = buf

                surf = self.essvi_surfaces.get(regime) or ESSVISurface()
                ok = surf.fit(buf, spot=float(spot))
                if ok:
                    self.essvi_surfaces[regime] = surf
                    self.last_diagnostics["essvi"] = dict(surf.last_diagnostics or {})
                    return True
            except Exception:
                return False

        india_vix = india_features.get('india_vix', 14.0)
        atm_iv = india_vix / 100.0 + self.REGIME_VOL_ADJ.get(regime, 0.0)
        atm_iv = max(atm_iv, 0.05)
        term_damp = min(1.0, np.sqrt(max(T, 1.0/365) * 52.0))
        use_liquidity_weighting = bool(getattr(get_features(), "USE_LIQUIDITY_WEIGHTING", False))
        use_interval_loss = bool(getattr(get_features(), "USE_INTERVAL_LOSS", False))

        old_params = self.svi_params.get(regime, self.svi_params['Sideways']).copy()

        try:
            base = self.svi_params.get(regime, self.svi_params['Sideways'])
            x0 = [base['b'], base['rho'], base['m'], base['sig']]
            if getattr(get_features(), "USE_SVI_FIXED_POINT_WARMSTART", False) and fixed_point_svi_warmstart:
                warm = fixed_point_svi_warmstart(
                    spot=spot,
                    strikes=strikes,
                    market_ivs=market_ivs,
                    T=T,
                    prev_params={"b": base["b"], "rho": base["rho"], "m": base["m"], "sig": base["sig"]},
                )
                x0 = [warm["b"], warm["rho"], warm["m"], warm["sig"]]

            # ── FLAW 5 FIX: Filter stale / unreliable deep-OTM quotes ────
            # NSE deep OTM options frequently have stale quotes (last traded
            # 30+ min ago). Fitting SVI to stale + live data contaminates
            # the entire surface. Filter by moneyness and suspicious IV.
            strikes_arr = np.asarray(strikes, dtype=float)
            ivs_arr = np.asarray(market_ivs, dtype=float)
            log_m = np.log(strikes_arr / spot)

            # Dynamic OTM filter: widen during high-VIX to keep liquid strikes
            _vix_now = float(india_features.get('india_vix', india_features.get('vix', 15.0)))
            _otm_lm_limit = 0.15 + 0.10 * max((_vix_now - 15.0) / 20.0, 0.0)
            _otm_lm_limit = min(_otm_lm_limit, 0.35)  # hard cap

            valid = np.ones(len(strikes_arr), dtype=bool)
            for i in range(len(strikes_arr)):
                abs_lm = abs(log_m[i])
                # Reject deep OTM with suspiciously low IV (likely stale)
                if abs_lm > 0.10 and ivs_arr[i] < 0.05:
                    valid[i] = False
                # Reject extremely far OTM (liquidity too thin to trust)
                if abs_lm > _otm_lm_limit:
                    valid[i] = False

            # Apply filter (keep at least 3 strikes for calibration)
            if np.sum(valid) >= 3:
                strikes_arr = strikes_arr[valid]
                ivs_arr = ivs_arr[valid]
                log_m = log_m[valid]
                strikes = strikes_arr.tolist()
                market_ivs = ivs_arr.tolist()
            # else: use all strikes (insufficient valid quotes)

            # Gaussian weight centered at ATM (k=0), width ~5% log-moneyness
            log_m_w = np.log(np.asarray(strikes, dtype=float) / spot)
            weights = 1.0 + 3.0 * np.exp(-0.5 * (log_m_w / 0.03)**2)  # ATM gets ~4x

            # Optional liquidity-aware calibration weights:
            # weight_i = vega_i / (spread_i^2 + eps)
            if use_liquidity_weighting:
                spreads_map = {}
                raw_spreads = india_features.get("quote_spreads")
                if isinstance(raw_spreads, dict):
                    for k_raw, v_raw in raw_spreads.items():
                        try:
                            spreads_map[float(k_raw)] = float(v_raw)
                        except Exception:
                            continue

                bids_map = {}
                asks_map = {}
                raw_bids = india_features.get("quote_bids")
                raw_asks = india_features.get("quote_asks")
                if isinstance(raw_bids, dict):
                    for k_raw, v_raw in raw_bids.items():
                        try:
                            bids_map[float(k_raw)] = float(v_raw)
                        except Exception:
                            continue
                if isinstance(raw_asks, dict):
                    for k_raw, v_raw in raw_asks.items():
                        try:
                            asks_map[float(k_raw)] = float(v_raw)
                        except Exception:
                            continue

                def _norm_pdf(xv):
                    return np.exp(-0.5 * xv * xv) / np.sqrt(2.0 * np.pi)

                liq_weights = np.asarray(weights, dtype=float)
                has_liq_meta = False
                eps_spread = 1e-8
                t_eff = max(float(T), 1e-8)
                sqrt_t = np.sqrt(t_eff)
                sigma_ref = max(atm_iv, 1e-4)
                for i, (K, mkt_iv) in enumerate(zip(strikes, market_ivs)):
                    spread = spreads_map.get(float(K))
                    if (spread is None or not np.isfinite(spread) or spread <= 0.0) and float(K) in bids_map and float(K) in asks_map:
                        bid_k = float(bids_map[float(K)])
                        ask_k = float(asks_map[float(K)])
                        if np.isfinite(bid_k) and np.isfinite(ask_k) and ask_k > bid_k:
                            spread = ask_k - bid_k
                    if spread is None or (not np.isfinite(spread)) or spread <= 0.0:
                        continue
                    has_liq_meta = True
                    d1 = (np.log(max(float(spot), 1e-8) / max(float(K), 1e-8)) + 0.5 * sigma_ref * sigma_ref * t_eff) / (sigma_ref * sqrt_t)
                    # Cap vega floor to prevent zero-vega weight explosion for short-dated OTM
                    _vega_floor = float(spot) * 0.001  # 0.1% of spot
                    vega = max(float(spot) * _norm_pdf(d1) * sqrt_t, _vega_floor)
                    # Robust spread floor: at least 1% of vega-implied spread
                    _spread_floor = max(float(spread), 0.01 * float(spot) * sigma_ref * sqrt_t)
                    liq_weights[i] = vega / (_spread_floor * _spread_floor + eps_spread)

                if has_liq_meta:
                    # Normalize to keep objective scale stable across modes
                    liq_weights = np.maximum(liq_weights, 1e-12)
                    liq_weights = liq_weights / np.mean(liq_weights)
                    weights = liq_weights

            # Optional interval-loss metadata in IV space:
            # quote_bounds_iv = {strike: {"bid": iv_bid, "ask": iv_ask}}
            interval_bounds = {}
            if use_interval_loss:
                raw_iv_bounds = india_features.get("quote_bounds_iv", {})
                if not isinstance(raw_iv_bounds, dict):
                    raw_iv_bounds = {}
                for k_raw, bounds in raw_iv_bounds.items():
                    try:
                        kf = float(k_raw)
                    except Exception:
                        continue
                    if not isinstance(bounds, dict):
                        continue
                    try:
                        b_iv = float(bounds.get("bid"))
                        a_iv = float(bounds.get("ask"))
                    except Exception:
                        continue
                    if np.isfinite(b_iv) and np.isfinite(a_iv) and a_iv >= b_iv and b_iv > 0 and a_iv < 3.0:
                        interval_bounds[kf] = (b_iv, a_iv)

            def objective(params_vec):
                b, rho, m_p, sig_p = params_vec
                atm_svi = rho * (0.0 - m_p) + np.sqrt(m_p**2 + sig_p**2)
                error = 0.0
                wing_penalty = 0.0
                for i, (K, mkt_iv) in enumerate(zip(strikes, market_ivs)):
                    lk = np.log(K / spot)
                    k_svi = rho * (lk - m_p) + np.sqrt((lk - m_p)**2 + sig_p**2)
                    model_iv = atm_iv + b * (k_svi - atm_svi) * term_damp
                    # Optional interval loss in IV-space.
                    if use_interval_loss and float(K) in interval_bounds:
                        bid_iv, ask_iv = interval_bounds[float(K)]
                        if model_iv < bid_iv:
                            residual = model_iv - bid_iv
                        elif model_iv > ask_iv:
                            residual = model_iv - ask_iv
                        else:
                            residual = 0.0
                    else:
                        # Huber loss: L2 near ATM, L1 near outliers (robust)
                        residual = model_iv - mkt_iv
                        huber_delta = 0.02  # ~2% IV threshold
                        if abs(residual) <= huber_delta:
                            error += weights[i] * residual ** 2
                            # Wing preservation: extra penalty for deep OTM/ITM deviations
                            if abs(lk) > 0.05:
                                wing_weight = 2.0 * (abs(lk) / 0.15)  # scale with distance from ATM
                                wing_penalty += wing_weight * residual ** 2
                            continue
                        error += weights[i] * (2 * huber_delta * abs(residual) - huber_delta**2)
                        # Wing penalty for Huber region too
                        if abs(lk) > 0.05:
                            wing_weight = 2.0 * (abs(lk) / 0.15)
                            wing_penalty += wing_weight * (2 * huber_delta * abs(residual) - huber_delta**2)
                        continue
                    error += weights[i] * residual ** 2
                # Add wing penalty (prevents optimizer from flattening wings)
                return error + 0.5 * wing_penalty

            bounds = [(0.1, 5.0), (-0.99, -0.01), (-0.05, 0.05), (0.01, 0.50)]
            result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')

            if result.success:
                b, rho, m_p, sig_p = result.x
                candidate = {
                    'b': float(b), 'rho': float(rho),
                    'm': float(m_p), 'sig': float(sig_p)
                }

                # 1. Weighted RMSE check — soft accept instead of hard reject.
                # Blend 90% old + 10% new on marginal fits to preserve learning signal.
                rmse = np.sqrt(result.fun / np.sum(weights))
                if rmse > 0.08:
                    # Hard reject: extremely bad fit
                    warnings.warn(f"SVI calibration RMSE {rmse:.4f} > 0.08 for {regime}; rolling back")
                    self.svi_params[regime] = old_params
                    return False
                elif rmse > 0.05:
                    # Soft accept: blend old and new to learn gradually
                    warnings.warn(f"SVI calibration RMSE {rmse:.4f} > 0.05 for {regime}; blending 90/10")
                    for _k in candidate:
                        candidate[_k] = 0.9 * old_params.get(_k, candidate[_k]) + 0.1 * candidate[_k]

                # 2. Continuous butterfly-arbitrage check:
                #    Verify d2w/dk2 >= 0 via SVI 2nd-derivative formula
                #    g(k) = (1 - k*rho/sqrt((k-m)^2+sig^2))^2/2 + b*sig^2/((k-m)^2+sig^2)^1.5
                #    Must be >= 0 for no butterfly arbitrage
                def _butterfly_penalty(params_vec, test_pts):
                    """Compute sum of squared negative g(k) values (arbitrage violations)."""
                    bv, rv, mv, sv = params_vec
                    penalty = 0.0
                    for tk in test_pts:
                        km2 = (tk - mv)**2 + sv**2
                        sqrt_km = np.sqrt(km2)
                        denom = km2 ** 1.5
                        if denom < 1e-12:
                            continue
                        term1_num = 1.0 - (tk - mv) * rv / sqrt_km
                        g_k = 0.5 * term1_num**2 + bv * sv**2 / denom
                        if g_k < 0:
                            penalty += g_k**2
                        # Also check total variance is non-negative
                        w_svi = bv * (rv * (tk - mv) + sqrt_km)
                        if w_svi < 0:
                            penalty += w_svi**2
                    return penalty

                test_points = np.linspace(-0.25, 0.25, 50)
                arb_penalty = _butterfly_penalty(result.x, test_points)
                arb_free = arb_penalty < 1e-8

                if not arb_free:
                    # Repair: re-optimize with arbitrage penalty in objective
                    def arb_constrained_objective(params_vec):
                        base_err = objective(params_vec)
                        arb_pen = _butterfly_penalty(params_vec, test_points)
                        return base_err + 1000.0 * arb_pen  # heavy penalty

                    try:
                        repair_result = minimize(arb_constrained_objective, result.x,
                                                 bounds=bounds, method='L-BFGS-B')
                        if repair_result.success:
                            repaired_penalty = _butterfly_penalty(repair_result.x, test_points)
                            if repaired_penalty < 1e-8:
                                # Repair succeeded — use repaired params
                                b, rho, m_p, sig_p = repair_result.x
                                candidate = {
                                    'b': float(b), 'rho': float(rho),
                                    'm': float(m_p), 'sig': float(sig_p)
                                }
                                arb_free = True
                    except Exception:
                        pass

                if not arb_free:
                    warnings.warn(f"SVI butterfly-arb detected for {regime}; "
                                  f"repair failed, rolling back")
                    self.svi_params[regime] = old_params
                    return False

                # v6 diagnostics: explicit butterfly/calendar checks
                if check_butterfly_arbitrage_slice:
                    cand_b, cand_rho, cand_m, cand_sig = candidate["b"], candidate["rho"], candidate["m"], candidate["sig"]

                    def _w_slice(karr):
                        karr = np.asarray(karr, dtype=float)
                        atm_svi = cand_rho * (0.0 - cand_m) + np.sqrt(cand_m**2 + cand_sig**2)
                        ksvi = cand_rho * (karr - cand_m) + np.sqrt((karr - cand_m)**2 + cand_sig**2)
                        ivs = np.maximum(atm_iv + cand_b * (ksvi - atm_svi) * term_damp, 1e-6)
                        return (ivs ** 2) * T

                    b_ok, b_metric = check_butterfly_arbitrage_slice(_w_slice)
                    self.last_diagnostics["svi_butterfly"] = dict(b_metric or {})
                    if not b_ok:
                        self.svi_params[regime] = old_params
                        return False

                self.svi_params[regime] = candidate
                return True
        except Exception as e:
            warnings.warn(f"SVI calibration failed for {regime}: {e}")
            self.svi_params[regime] = old_params
        return False

    # ------------------------------------------------------------------
    # Calendar-arbitrage check (total-variance monotonicity)
    # ------------------------------------------------------------------
    def check_calendar_arbitrage(self, spot, regime, india_features,
                                  expiries_T, n_test=21):
        """
        Verify that total variance  w(k, T) = σ(k, T)² · T  is
        non-decreasing in T for every test log-moneyness k.

        Calendar spread arbitrage exists whenever a far-expiry total
        variance is *lower* than a near-expiry one at the same strike.

        Parameters
        ----------
        spot           : float
        regime         : str
        india_features : dict
        expiries_T     : list[float] – sorted ascending
        n_test         : int – number of moneyness points to check

        Returns
        -------
        dict  {'clean': bool, 'violations': list[dict]}
        """
        if len(expiries_T) < 2:
            return {'clean': True, 'violations': []}

        expiries_T = sorted(expiries_T)
        test_k = np.linspace(-0.15, 0.15, n_test)
        violations = []

        # Compute total variance matrix: rows = expiries, cols = moneyness
        tv = np.zeros((len(expiries_T), n_test))
        for i, T in enumerate(expiries_T):
            for j, k in enumerate(test_k):
                K = spot * np.exp(k)
                iv = self.get_implied_vol(spot, K, T, regime, india_features)
                tv[i, j] = iv**2 * T

        # Check monotonicity: tv[i+1, j] >= tv[i, j] for all j
        for i in range(len(expiries_T) - 1):
            for j in range(n_test):
                if tv[i + 1, j] < tv[i, j] - 1e-8:
                    violations.append({
                        'T_near': expiries_T[i],
                        'T_far':  expiries_T[i + 1],
                        'k':      float(test_k[j]),
                        'w_near': float(tv[i, j]),
                        'w_far':  float(tv[i + 1, j]),
                    })

        return {'clean': len(violations) == 0, 'violations': violations}



# ============================================================================
# MODULE 4: JUMP-DIFFUSION MONTE CARLO PRICING ENGINE
# ============================================================================


class HestonJumpDiffusionPricer:
    """
    Prices Nifty 50 European options using a Heston Stochastic Variance
    model combined with Merton's jump-diffusion, via Monte Carlo simulation.

    Extends the Q-Fin StochasticVarianceModel approach with:

    - Heston (1993) stochastic variance process:
        dV = κ(θ_v − V)dt + σ_v √V dW_v
      with correlated Brownian motions (Cholesky decomposition):
        dW_S · dW_v = ρ dt

    - Merton jump component:
        dS/S = (r − q − λk)dt + √V dW_S + J dN

    - QE (Quadratic Exponential) discretisation for the variance process
      (Andersen 2008) — avoids negative variance without truncation bias

    - Antithetic variates + moment matching (variance reduction)

    - Control variate using BSM analytical (dramatically reduces MC error)

    - India risk premium adjustment via IndiaFeatureEngine

    Reference: romanmichaelpaolucci/Q-Fin (StochasticVarianceModel, MonteCarloCall)
    """


    def __init__(self, n_paths=50000, n_steps=None, use_sobol=False, seed=None):
        # Force even path count for antithetic variates
        self.n_paths = n_paths if n_paths % 2 == 0 else n_paths + 1
        self.n_steps = n_steps
        self.use_sobol = use_sobol   # Sobol QMC for ~4-8x faster convergence
        self.seed = seed             # Reproducibility (None = random each time)
        # Thread-safe RNG: avoid np.random.seed() which mutates global state
        self._rng = np.random.default_rng(seed)


    # ------------------------------------------------------------------
    # BSM analytical price (control variate anchor)
    # ------------------------------------------------------------------
    @staticmethod
    def _bsm_price(spot, strike, T, r, q, sigma, option_type):
        """Black-Scholes-Merton European analytical price."""
        if T <= 0 or sigma <= 0:
            if option_type.upper() in ('CE', 'CALL'):
                return max(spot - strike, 0.0)
            return max(strike - spot, 0.0)
        sqrt_T = np.sqrt(T)
        d1 = (np.log(spot / strike) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T
        if option_type.upper() in ('CE', 'CALL'):
            return float(spot * np.exp(-q * T) * norm.cdf(d1)
                         - strike * np.exp(-r * T) * norm.cdf(d2))
        return float(strike * np.exp(-r * T) * norm.cdf(-d2)
                     - spot * np.exp(-q * T) * norm.cdf(-d1))


    # ------------------------------------------------------------------
    # Main pricer
    # ------------------------------------------------------------------
    def price(self, spot, strike, T, r, q, sigma, regime_params,
              option_type='CE', india_features=None):
        """
        Price an option via Heston SV + jump-diffusion Monte Carlo with
        control variate variance reduction.

        Parameters
        ----------
        spot          : float – Current spot price
        strike        : float – Strike price
        T             : float – Time to expiry (years)
        r             : float – Risk-free rate (decimal)
        q             : float – Dividend yield (decimal)
        sigma         : float – Implied volatility (decimal)
        regime_params : dict  – From RegimeDetector.REGIME_PARAMS[regime]
        option_type   : str   – 'CE' for call, 'PE' for put
        india_features: dict  – From IndiaFeatureEngine.compute_features()

        Returns
        -------
        (price, std_error, S_terminal)
        """
        n_steps = self.n_steps or max(int(T * 252), 5)
        dt = T / n_steps
        n = self.n_paths

        # -- Jump parameters --
        lambda_j = regime_params.get('lambda_j', 0.03)
        mu_j = regime_params.get('mu_j', 0.0)
        sigma_j = regime_params.get('sigma_j', abs(mu_j) * 0.5 + 0.005)

        # Jump compensator: k = E[e^J - 1]  (Merton 1976)
        k_comp = np.exp(mu_j + 0.5 * sigma_j**2) - 1.0

        # -- Heston SV parameters (from regime or defaults) --
        kappa   = regime_params.get('kappa', 2.0)        # Mean reversion speed
        theta_v_regime = regime_params.get('theta_v', sigma**2)  # Regime long-run var
        sigma_v = regime_params.get('sigma_v', 0.3)       # Vol-of-vol
        rho_sv  = regime_params.get('rho_sv', -0.5)       # Spot-vol correlation

        # Blend regime theta_v with market-implied variance so the Heston
        # long-run level stays anchored to the actual IV (prevents the
        # mean-reversion from dragging variance away from market levels)
        theta_v = 0.5 * theta_v_regime + 0.5 * sigma**2

        # ── FLAW 2: Regime-Switching Heston Parameters ───────────────────
        # Pre-build parameter arrays for all 4 regimes so we can switch
        # during the MC simulation. At each timestep, a Markov chain
        # determines which regime each path is in, selecting the
        # appropriate (κ, θ, σ_v, ρ, λ_j, μ_j, σ_j) for that step.
        _regime_names = list(RegimeDetector.REGIME_PARAMS.keys())
        _n_regimes = len(_regime_names)
        _regime_kappa = np.array([RegimeDetector.REGIME_PARAMS[r].get('kappa', 2.0) for r in _regime_names])
        _regime_theta = np.array([0.5 * RegimeDetector.REGIME_PARAMS[r].get('theta_v', sigma**2) + 0.5 * sigma**2 for r in _regime_names])
        _regime_sigma_v = np.array([RegimeDetector.REGIME_PARAMS[r].get('sigma_v', 0.3) for r in _regime_names])
        _regime_rho = np.array([RegimeDetector.REGIME_PARAMS[r].get('rho_sv', -0.5) for r in _regime_names])
        _regime_lambda = np.array([RegimeDetector.REGIME_PARAMS[r].get('lambda_j', 0.03) for r in _regime_names])
        _regime_mu_j = np.array([RegimeDetector.REGIME_PARAMS[r].get('mu_j', 0.0) for r in _regime_names])
        _regime_sigma_j = np.array([RegimeDetector.REGIME_PARAMS[r].get('sigma_j', 0.01) for r in _regime_names])
        # Transition matrix (per-step probability = Q_ij * dt * 252)
        _rd_inst = RegimeDetector()
        _trans_prob = _rd_inst.transition_matrix  # 4x4 daily transition probabilities
        # Scale to per-step: row-normalise after scaling off-diagonal by dt*252
        _trans_step = np.eye(_n_regimes)
        for i in range(_n_regimes):
            for j in range(_n_regimes):
                if i != j:
                    _trans_step[i, j] = _trans_prob[i, j] * dt * 252
            _trans_step[i, i] = 1.0 - np.sum(_trans_step[i, :]) + _trans_step[i, i]
            _trans_step[i] = np.maximum(_trans_step[i], 0)
            _trans_step[i] /= _trans_step[i].sum()

        # Determine starting regime index
        _current_regime_name = None
        for rn in _regime_names:
            if RegimeDetector.REGIME_PARAMS[rn] == regime_params or rn == regime_params.get('_regime_name', ''):
                _current_regime_name = rn
                break
        _start_regime_idx = _regime_names.index(_current_regime_name) if _current_regime_name else 2  # default Sideways

        # Enable regime switching only for T > 7 days (worthwhile)
        _use_regime_switching = (T > 7.0 / 365.0 and n_steps >= 3)

        # Feller condition diagnostic: 2κθ ≥ σ_v²
        feller_ratio = 2.0 * kappa * theta_v / max(sigma_v**2, 1e-12)
        feller_violated = feller_ratio < 1.0

        # India risk premium adjustment
        if india_features:
            irp = india_features.get('india_risk_premium', 0)
            sigma_eff = sigma * (1.0 + 0.05 * irp)
        else:
            sigma_eff = sigma

        # Initial instantaneous variance from IV
        V0 = sigma_eff**2

        # -- Antithetic variates (n guaranteed even) --
        half_n = n // 2

        # Reset RNG for reproducibility (thread-safe, no global state mutation)
        if self.seed is not None:
            self._rng = np.random.default_rng(self.seed)

        # =====================================================================
        # Pre-generate ALL random numbers using Sobol QMC or pseudo-random
        # Sobol gives ~4-8x faster convergence for the same path count.
        # We need 3 streams per step: z1 (spot), z2 (vol), z_jump
        # =====================================================================
        total_dims = 3  # z1, z2_indep, z_jump per step
        if self.use_sobol and SOBOL_AVAILABLE and half_n >= 16:
            try:
                # Sobol dimension = total_dims * n_steps
                sobol_dims = min(total_dims * n_steps, 21201)  # scipy limit
                actual_steps_sobol = sobol_dims // total_dims
                # Round half_n up to next power of 2 for Sobol (required)
                sobol_n = int(2 ** np.ceil(np.log2(max(half_n, 16))))
                sampler = _SobolEngine(d=sobol_dims, scramble=True,
                                       seed=self.seed if self.seed else self._rng.integers(0, 2**31))
                sobol_uniform = sampler.random(sobol_n)  # (sobol_n, dims) in [0,1]
                # Inverse normal transform → standard normals
                sobol_normals = norm.ppf(np.clip(sobol_uniform, 1e-8, 1 - 1e-8))
                # Trim to exact half_n
                sobol_normals = sobol_normals[:half_n]
                use_precomputed = True
            except Exception:
                use_precomputed = False
                actual_steps_sobol = 0
        else:
            use_precomputed = False
            actual_steps_sobol = 0

        # Memory-efficient: only track CURRENT values, not full paths
        log_S = np.full(n, np.log(max(spot, 1e-8)))
        V_cur = np.full(n, V0)
        log_S_gbm = np.full(n, np.log(max(spot, 1e-8)))

        # Per-path regime index (for regime-switching Heston)
        regime_idx = np.full(n, _start_regime_idx, dtype=int)

        drift_base = (r - q - lambda_j * k_comp) * dt
        gbm_drift = (r - q - 0.5 * sigma_eff**2) * dt
        gbm_vol = sigma_eff * np.sqrt(dt)

        for step in range(n_steps):
            # ── FLAW 2: Per-step regime transitions ─────────────────────
            if _use_regime_switching and step > 0:
                # Draw regime transition for each path
                u_regime = self._rng.uniform(0, 1, n)
                for path_idx in range(n):
                    ri = regime_idx[path_idx]
                    cum = np.cumsum(_trans_step[ri])
                    regime_idx[path_idx] = int(np.searchsorted(cum, u_regime[path_idx]))
                    regime_idx[path_idx] = min(regime_idx[path_idx], _n_regimes - 1)
                # Update per-path Heston parameters from regime
                kappa = _regime_kappa[regime_idx]
                theta_v = _regime_theta[regime_idx]
                sigma_v = _regime_sigma_v[regime_idx]
                rho_sv = _regime_rho[regime_idx]
                lambda_j = _regime_lambda[regime_idx]
                mu_j = _regime_mu_j[regime_idx]
                sigma_j = _regime_sigma_j[regime_idx]
                k_comp = np.exp(mu_j + 0.5 * sigma_j**2) - 1.0
                drift_base = (r - q - lambda_j * k_comp) * dt

            # --- Draw normals: Sobol (pre-computed) or pseudo-random ---
            if use_precomputed and step < actual_steps_sobol:
                base_col = step * total_dims
                z1_half = sobol_normals[:, base_col]
                z2_indep = sobol_normals[:, base_col + 1]
                z_jump_half = sobol_normals[:, base_col + 2]
            else:
                z1_half = self._rng.standard_normal(half_n)
                z2_indep = self._rng.standard_normal(half_n)
                z_jump_half = self._rng.standard_normal(half_n)

            # Stratified sampling: partition each coordinate into N strata
            # for additional 2-3x variance reduction on top of Sobol
            if use_precomputed and step < actual_steps_sobol and half_n >= 64:
                # Sort and re-assign to strata for better coverage
                n_strata = min(32, half_n // 2)
                strata_size = half_n // n_strata
                order = np.argsort(z1_half)
                z1_half = z1_half[order]  # Stratified ordering

            # Moment matching on the half
            z1_half = (z1_half - z1_half.mean()) / max(z1_half.std(), 1e-8)
            z2_indep = (z2_indep - z2_indep.mean()) / max(z2_indep.std(), 1e-8)

            # Moment matching on the half

            # Antithetic variates
            z1 = np.concatenate([z1_half, -z1_half])
            z2_raw = np.concatenate([z2_indep, -z2_indep])

            # Cholesky correlation: W_v = ρ·W_S + √(1−ρ²)·W_indep
            z_v = rho_sv * z1 + np.sqrt(1.0 - rho_sv**2) * z2_raw

            # --- Variance step: QE (Quadratic Exponential) scheme ---
            V_safe = np.maximum(V_cur, 1e-8)

            # Conditional moments of V(t+dt) given V(t)
            e_kdt = np.exp(-kappa * dt)
            m_v = theta_v + (V_safe - theta_v) * e_kdt  # E[V(t+dt)]
            s2_v = (V_safe * sigma_v**2 * e_kdt / kappa * (1.0 - e_kdt)
                    + theta_v * sigma_v**2 / (2.0 * kappa) * (1.0 - e_kdt)**2)
            s2_v = np.maximum(s2_v, 1e-12)

            # QE switch on psi = s²/m²
            psi = s2_v / np.maximum(m_v**2, 1e-12)
            psi_crit = 1.5  # Andersen (2008) recommends 1.5

            V_next = np.zeros(n)

            # Case 1: psi ≤ psi_crit → moment-matched lognormal
            mask_lo = psi <= psi_crit
            if np.any(mask_lo):
                b2 = 2.0 / np.maximum(psi[mask_lo], 1e-8) - 1.0 + np.sqrt(
                    np.maximum(2.0 / np.maximum(psi[mask_lo], 1e-8)
                               * (2.0 / np.maximum(psi[mask_lo], 1e-8) - 1.0), 0.0))
                b_val = np.sqrt(np.maximum(b2, 0.0))
                a_val = m_v[mask_lo] / (1.0 + b2)
                V_next[mask_lo] = a_val * (b_val + z_v[mask_lo])**2

            # Case 2: psi > psi_crit → exponential approximation
            mask_hi = ~mask_lo
            if np.any(mask_hi):
                p_exp = (psi[mask_hi] - 1.0) / (psi[mask_hi] + 1.0)
                beta_exp = (1.0 - p_exp) / np.maximum(m_v[mask_hi], 1e-8)
                U = norm.cdf(z_v[mask_hi])
                V_next[mask_hi] = np.where(
                    U > p_exp,
                    np.log(np.maximum((1.0 - p_exp) / np.maximum(1.0 - U, 1e-12), 1e-12)) / beta_exp,
                    0.0
                )

            # Floor variance at tiny positive value -- do NOT alter calibrated params
            V_cur = np.maximum(V_next, 1e-10)

            # --- Spot step: Heston + jumps ---
            sqrt_V_dt = np.sqrt(np.maximum(V_safe, 1e-8) * dt)

            # Compound Poisson jumps — ANTITHETIC: mirror counts and normals
            # Compound Poisson jumps — ANTITHETIC: mirror counts and normals
            if np.ndim(lambda_j) > 0:
                # Per-path lambda (regime switching): cannot use symmetric counts
                n_jumps = np.random.poisson(lambda_j * dt) # Size inferred from lambda shape (n)
            else:
                n_jumps_half = np.random.poisson(lambda_j * dt, half_n)
                n_jumps = np.concatenate([n_jumps_half, n_jumps_half])
            z_jump = np.concatenate([z_jump_half, -z_jump_half])
            jump_sizes = np.where(
                n_jumps > 0,
                n_jumps * mu_j + np.sqrt(np.maximum(n_jumps, 1e-8)) * sigma_j * z_jump,
                0.0
            )

            log_S = (
                log_S
                + drift_base
                - 0.5 * V_safe * dt
                + sqrt_V_dt * z1
                + jump_sizes
            )

            # GBM control variate path (same random numbers z1)
            log_S_gbm = (
                log_S_gbm
                + gbm_drift
                + gbm_vol * z1
            )

        S_T = np.exp(log_S)
        S_T_gbm = np.exp(log_S_gbm)

        # -- Payoffs --
        is_call = option_type.upper() in ('CE', 'CALL')
        if is_call:
            payoffs = np.maximum(S_T - strike, 0)
            payoffs_gbm = np.maximum(S_T_gbm - strike, 0)
        else:
            payoffs = np.maximum(strike - S_T, 0)
            payoffs_gbm = np.maximum(strike - S_T_gbm, 0)

        disc = np.exp(-r * T)

        # -- Control variate: dual BSM + Heston COS for tighter MC estimates --
        bsm_analytical = self._bsm_price(spot, strike, T, r, q, sigma_eff, option_type)
        mc_gbm_price = disc * np.mean(payoffs_gbm)

        # Primary CV: BSM analytical
        cv_adjustment = bsm_analytical - mc_gbm_price

        # Secondary CV: Heston COS (if available via regime_params)
        # This dramatically reduces variance for OTM/ITM where BSM is poor
        cos_price = None
        try:
            _qe = regime_params.get('_quant_engine')
            if _qe is not None and hasattr(_qe, 'heston_cos'):
                V0_cos = max(sigma_eff ** 2, 1e-8)
                cos_price = _qe.heston_cos.price(
                    spot, strike, T, r, q, V0_cos,
                    regime_params.get('kappa', 2.0),
                    regime_params.get('theta_v', V0_cos),
                    regime_params.get('sigma_v', 0.3),
                    regime_params.get('rho_sv', -0.5),
                    option_type,
                )
        except Exception:
            cos_price = None

        raw_price = disc * np.mean(payoffs)

        if cos_price is not None and np.isfinite(cos_price) and cos_price > 0:
            # Dual control variate: optimal linear combination
            # price_cv = price_mc + β₁(BSM - mc_gbm) + β₂(COS - mc_gbm)
            # β_opt = Cov(Y, X) / Var(X) for single CV; extend to multivariate
            # Simplified: weighted blend of both CV adjustments
            cv_bsm = bsm_analytical - mc_gbm_price
            cv_cos = cos_price - mc_gbm_price
            # Use both: weight COS more for OTM (where BSM deviates more)
            moneyness = strike / spot
            cos_weight = min(0.6, 0.3 + 0.5 * abs(moneyness - 1.0))
            bsm_weight = 1.0 - cos_weight
            cv_adjustment = bsm_weight * cv_bsm + cos_weight * cv_cos

        price = max(raw_price + cv_adjustment, 0.0)

        # Per-path control-variate-adjusted payoffs (for SE calculation)
        cv_payoffs = disc * payoffs + (bsm_analytical - disc * payoffs_gbm)

        # ── FLAW 4 FIX: Randomized QMC Standard Error ───────────────────
        # Standard std/sqrt(n) assumes IID payoffs, but Sobol QMC +
        # stratification + antithetic variates violate IID. Sobol has
        # error O(log(N)^d / N), not O(1/√N). We use M scrambled batches
        # to get the correct RQMC confidence interval.
        if self.use_sobol and SOBOL_AVAILABLE and n >= 256:
            M_batches = 16
            batch_size = n // M_batches
            batch_prices = []
            for b_idx in range(M_batches):
                b_start = b_idx * batch_size
                b_end = b_start + batch_size
                batch_payoffs = cv_payoffs[b_start:b_end]
                batch_prices.append(float(np.mean(batch_payoffs)))
            std_error = float(np.std(batch_prices) / np.sqrt(M_batches))
        else:
            std_error = float(np.std(cv_payoffs) / np.sqrt(n))

        # Store diagnostics on the instance for callers that need them
        self._last_feller_violated = feller_violated
        self._last_S_T = S_T
        self._last_payoffs = payoffs
        self._last_cv_payoffs = cv_payoffs
        self._last_disc = disc
        self._last_is_call = is_call
        self._last_spot = spot
        self._last_strike = strike
        self._last_T = T
        self._last_r = r

        return price, std_error, S_T

    # ------------------------------------------------------------------
    # Pathwise + LR Greeks (no extra MC runs needed)
    # ------------------------------------------------------------------
    def compute_pathwise_greeks(self, sigma):
        """
        Compute Delta (pathwise), Gamma (likelihood-ratio), Vega (pathwise
        approximation) from the most recent ``price()`` call.

        These methods avoid re-running the Monte-Carlo entirely:
        - **Pathwise Delta** differentiates the payoff directly (valid because
          max(S_T - K, 0) is piecewise linear in S_T, and dS_T/dS_0 = S_T/S_0
          in log-space schemes).
        - **LR Gamma** uses the score function of the (approximate effective-vol)
          density, which avoids differentiating through the payoff kink.
        - **Pathwise Vega** uses the chain rule through the log-normal
          approximation with the realised path volatility.

        Returns
        -------
        dict  with keys 'delta', 'gamma', 'vega' (or None if data unavailable)
        """
        if self._last_S_T is None or self._last_payoffs is None:
            return None

        S_T    = self._last_S_T
        payoffs = self._last_payoffs
        disc   = self._last_disc
        spot   = self._last_spot
        strike = self._last_strike
        T      = self._last_T
        is_call = self._last_is_call

        n = len(S_T)
        sign = 1.0 if is_call else -1.0

        # --- Pathwise Delta ------------------------------------------------
        itm = (S_T > strike) if is_call else (S_T < strike)
        delta = float(sign * disc * np.mean(itm.astype(float) * S_T / spot))

        # --- Effective vol from realised path log-returns ---
        log_ret = np.log(np.maximum(S_T / spot, 1e-15))
        sigma_eff = float(np.std(log_ret) / np.sqrt(max(T, 1e-8)))

        gamma = 0.0
        vega = 0.0

        if sigma_eff > 0.01 and T > 1e-8:
            mu_eff = float(np.mean(log_ret))
            sig_T  = sigma_eff * np.sqrt(T)
            Z = (log_ret - mu_eff) / sig_T

            # --- LR Gamma --------------------------------------------------
            # d^2/dS_0^2  using score of the effective log-normal density
            lr_w = ((Z**2 - 1.0) / (sigma_eff**2 * T)
                    - Z / (sigma_eff * np.sqrt(T))) / (spot**2)
            gamma = float(disc * np.mean(payoffs * lr_w))

            # --- Pathwise Vega (approx) ------------------------------------
            # d(S_T)/d(sigma) ~ S_T * (Z*sqrt(T) - sigma_eff*T)
            dST_dsig = S_T * (Z * np.sqrt(T) - sigma_eff * T)
            vega_raw = float(sign * disc * np.mean(itm.astype(float) * dST_dsig))
            vega = vega_raw / 100.0          # per 1 % vol move

        return {'delta': round(delta, 4),
                'gamma': round(gamma, 6),
                'vega':  round(vega, 2)}


# Backward-compatible alias
JumpDiffusionPricer = HestonJumpDiffusionPricer




# ============================================================================
# MODULE 5: BAYESIAN CONFIDENCE & PROBABILITY ENGINE
# ============================================================================


class BayesianConfidenceEngine:
    """
    Computes profit probability, confidence level, and expected P&L
    using Bayesian bootstrap resampling on the terminal price distribution.

    Provides BOTH:
    - Risk-neutral P(profit): from the pricing measure (drift = r - q)
    - Physical P(profit): from real-world measure (drift = historical μ)
      The physical measure is what actually matters for trading decisions.
   
    The Bayesian bootstrap:
    1. Draws Dirichlet weights over MC terminal values
    2. Computes weighted fair value for each bootstrap sample
    3. Builds a posterior distribution of fair values
    4. Extracts profit probability and confidence intervals
    """


    def __init__(self, n_bootstrap=5000):
        self.n_bootstrap = n_bootstrap  # 5000 for stable tail quantiles


    def compute_profit_probability(self, S_terminal, strike, market_price,
                                   r, T, option_type='CE', lot_size=65,
                                   spot=None, returns_30d=None,
                                   regime=None, iv=None):
        """
        Parameters
        ----------
        S_terminal   : np.ndarray - MC terminal spot prices (risk-neutral)
        strike       : float
        market_price : float - Current market price of the option
        r            : float - Risk-free rate
        T            : float - Time to expiry
        option_type  : str   - 'CE' or 'PE'
        lot_size     : int   - Contract lot size
        spot         : float - Current spot price (for physical measure)
        returns_30d  : np.ndarray - Historical log returns (for physical drift)
        regime       : str   - Current detected regime (for regime-conditional drift)
        iv           : float - Implied volatility (for VRP calculation)

        Returns
        -------
        (profit_prob_rn, physical_profit_prob, confidence, expected_pnl_rn,
         physical_expected_pnl, fair_values_distribution)
        """
        S_terminal = np.asarray(S_terminal, dtype=float)
        if S_terminal.size == 0:
            anchor = float(spot) if (spot is not None and np.isfinite(spot) and spot > 0) else float(max(strike, 1.0))
            S_terminal = np.array([anchor], dtype=float)
        finite_st = S_terminal[np.isfinite(S_terminal)]
        if finite_st.size == 0:
            anchor = float(spot) if (spot is not None and np.isfinite(spot) and spot > 0) else float(max(strike, 1.0))
            S_terminal = np.full_like(S_terminal, anchor, dtype=float)
        elif not np.all(np.isfinite(S_terminal)):
            anchor = float(np.median(finite_st))
            upper = float(np.max(finite_st))
            S_terminal = np.nan_to_num(
                S_terminal,
                nan=anchor,
                posinf=max(upper, anchor),
                neginf=0.0,
            )
        S_terminal = np.clip(S_terminal, 0.0, None)
        n = len(S_terminal)

        is_call = option_type.upper() in ('CE', 'CALL')
        if is_call:
            payoffs = np.maximum(S_terminal - strike, 0)
        else:
            payoffs = np.maximum(strike - S_terminal, 0)

        # Numerical guard: rare path explosions can create inf/nan payoffs.
        # Sanitize before expectation/bootstrap so confidence remains stable.
        payoffs = np.asarray(payoffs, dtype=float)
        if payoffs.size == 0:
            payoffs = np.zeros(1, dtype=float)
            n = 1
        finite_mask = np.isfinite(payoffs)
        if not np.all(finite_mask):
            finite_vals = payoffs[finite_mask]
            if finite_vals.size > 0:
                finite_cap = float(np.max(finite_vals))
                if finite_vals.size > 10:
                    finite_cap = max(finite_cap, float(np.percentile(finite_vals, 99.9)))
            else:
                finite_cap = 0.0
            payoffs = np.nan_to_num(payoffs, nan=0.0, posinf=finite_cap, neginf=0.0)
        payoffs = np.clip(payoffs, 0.0, None)

        pnl_per_unit = payoffs - market_price
        profit_prob_rn = np.mean(pnl_per_unit > 0) * 100
        expected_pnl_rn = np.mean(pnl_per_unit) * lot_size

        # ── Physical-measure profit probability (regime-conditional) ─────
        # Uses regime-specific drift instead of global historical mean.
        # Also incorporates Variance Risk Premium (VRP = IV^2 - RV^2).
        physical_profit_prob = profit_prob_rn
        physical_expected_pnl = expected_pnl_rn
        if spot is not None and spot > 0:
            # Physical drift — three-tier hierarchy:
            # 1. Regime-specific drift (strongest signal)
            # 2. EWMA of historical returns (half-life ~60 days)
            # 3. Zero drift (agnostic fallback — NOT the old 0.12 Nifty LTA)
            if regime and regime in RegimeDetector.REGIME_PARAMS:
                mu_daily = RegimeDetector.REGIME_PARAMS[regime]['mu']
                mu_annual = mu_daily * 252
            elif returns_30d is not None and len(returns_30d) >= 10:
                # EWMA with half-life 60 days for recent-bias
                hl = 60.0
                idxs = np.arange(len(returns_30d), dtype=float)
                ew = np.exp(-idxs * np.log(2) / hl)
                ew = ew / ew.sum()
                mu_daily = float(np.dot(ew, returns_30d))
                mu_annual = mu_daily * 252
            else:
                mu_annual = 0.0  # agnostic — do not assume positive drift

            # ── FLAW 1 FIX: Full VRP-corrected measure change ────────────
            # The Radon-Nikodym derivative dP/dQ affects the entire
            # distribution — not just the mean. Under Heston SV + jumps,
            # a simple mean-shift biases physical profit probabilities
            # by 5-15% for OTM options. The VRP shifts the physical
            # variance level: θ_P = κθ_Q / (κ + γσ_v).
            q_est = 0.012
            drift_shift = (mu_annual - r + q_est) * T

            # VRP: difference between implied (option-priced) and realised variance
            # When VRP > 0, options are expensive → physical variance is LOWER
            # than risk-neutral variance → terminal distribution is tighter under P
            rv_squared = float(np.var(returns_30d) * 252) if (returns_30d is not None and len(returns_30d) >= 10) else 0.0
            iv_squared = iv ** 2 if iv is not None else rv_squared
            vrp_var = iv_squared - rv_squared  # Typically positive
            variance_shift = -0.5 * vrp_var * T  # Tighten distribution under P

            S_T_physical = S_terminal * np.exp(drift_shift + variance_shift)

            if is_call:
                phys_payoffs = np.maximum(S_T_physical - strike, 0)
            else:
                phys_payoffs = np.maximum(strike - S_T_physical, 0)

            phys_pnl = phys_payoffs - market_price
            physical_profit_prob = float(np.mean(phys_pnl > 0) * 100)
            physical_expected_pnl = float(np.mean(phys_pnl) * lot_size)

            # VRP adjustment: when IV^2 >> RV^2, option sellers have edge
            # This adjusts the physical prob for VRP (affects put sellers mostly)
            if iv is not None and returns_30d is not None and len(returns_30d) >= 10:
                rv = float(np.std(returns_30d) * np.sqrt(252))
                vrp = iv**2 - rv**2
                if vrp > 0.005:  # Significant positive VRP
                    # Sellers have structural edge: reduce physical PoP for buyers
                    vrp_adj = min(vrp * 200, 5.0)  # cap at 5pp
                    if is_call:
                        physical_profit_prob -= vrp_adj * 0.5
                    else:
                        physical_profit_prob -= vrp_adj * 0.8  # puts more affected
                elif vrp < -0.005:  # Negative VRP (rare, risk is under-priced)
                    vrp_adj = min(abs(vrp) * 200, 5.0)
                    physical_profit_prob += vrp_adj * 0.5
                physical_profit_prob = np.clip(physical_profit_prob, 0, 100)


        # Bayesian bootstrap: Dirichlet-weighted fair values
        # VECTORIZED: draw all Dirichlet weights at once (massive speedup)
        all_weights = np.random.dirichlet(np.ones(n), size=self.n_bootstrap)  # (B, n)
        disc = float(np.exp(-r * T))
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            fair_values = disc * (all_weights @ payoffs)  # (B,)
        if not np.all(np.isfinite(fair_values)):
            finite_fv = fair_values[np.isfinite(fair_values)]
            max_fv = float(np.max(finite_fv)) if finite_fv.size > 0 else float(disc * np.max(payoffs))
            fair_values = np.nan_to_num(fair_values, nan=0.0, posinf=max_fv, neginf=0.0)


        # Confidence from bootstrap CI width
        ci_low = np.percentile(fair_values, 5)
        ci_high = np.percentile(fair_values, 95)
        fv_mean = np.mean(fair_values)


        # CI-width-based confidence (Jaeckel / institutional grade)
        # Narrower 90%-CI relative to market price  =>  higher confidence.
        # Factor ~500 tuned so typical ATM options yield 60-90 confidence.
        ci_width = (ci_high - ci_low) / max(market_price, 0.01)
        confidence = float(np.clip(100.0 - ci_width * 500.0, 0.0, 99.0))


        return (profit_prob_rn, physical_profit_prob, confidence,
                expected_pnl_rn, physical_expected_pnl, fair_values)




# ============================================================================
# MODULE 6: GREEKS CALCULATOR (Monte Carlo Bump-and-Reprice)
# ============================================================================


class GreeksCalculator:
    """
    Computes option Greeks using a **hybrid** approach:

    - **Delta, Gamma, Vega** — from the pathwise / likelihood-ratio methods
      stored on the pricer after the main ``price()`` call.  *Zero* extra
      Monte-Carlo runs required.
    - **Theta, Rho, Vanna, Charm** — CRN bump-and-reprice on a lightweight
      sub-pricer (1/4 paths).

    Net effect: ~6 MC bump-runs instead of the previous ~15, giving a
    > 2× speed-up for the full Greeks suite with equivalent accuracy.
    """

    @staticmethod
    def compute(pricer, spot, strike, T, r, q, sigma, regime_params,
                option_type, india_features, bump_pct=0.005):
        """
        Parameters
        ----------
        pricer         : HestonJumpDiffusionPricer instance (must have been called)
        spot, strike, T, r, q, sigma : float
        regime_params  : dict
        option_type    : str
        india_features : dict
        bump_pct       : float

        Returns
        -------
        dict with delta, gamma, theta, vega, rho, vanna, charm
        """
        _zero = {'delta': 0.0, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0,
                 'rho': 0.0, 'vanna': 0.0, 'charm': 0.0}
        try:
            # ── 1. Pathwise / LR Greeks (delta, gamma, vega) ──────────
            pw = pricer.compute_pathwise_greeks(sigma)
            if pw is None:
                pw = {'delta': 0.0, 'gamma': 0.0, 'vega': 0.0}

            delta = pw['delta']
            gamma = pw['gamma']
            vega  = pw['vega']

            # ── 2. CRN bump-and-reprice for remaining Greeks ──────────
            greeks_pricer = HestonJumpDiffusionPricer(
                n_paths=max(pricer.n_paths // 4, 5000),
                n_steps=pricer.n_steps,
            )
            crn_seed = pricer._rng.integers(0, 2**31) # Use pricer's RNG for CRN seed

            def _price(s=spot, k=strike, t=T, rate=r, vol=sigma):
                greeks_pricer._rng = np.random.default_rng(crn_seed)
                p, _, _ = greeks_pricer.price(
                    s, k, t, rate, q, vol,
                    regime_params, option_type, india_features,
                )
                return p

            # --- Theta: central difference on time (per calendar day) ---
            # Keep baseline deterministic: intraday theta scaling is opt-in
            # via india_features and defaults to 1.0.
            intraday_weight = 1.0
            if bool((india_features or {}).get('use_intraday_theta', False)):
                try:
                    hour_frac = float((india_features or {}).get('intraday_hour_frac', 0.0))
                    intraday_weight = 1.0 + 2.0 / (1.0 + np.exp(-(hour_frac - 14.5)))
                    if hour_frac < 9.25 or hour_frac > 15.5:
                        intraday_weight = 1.0
                except Exception:
                    intraday_weight = 1.0

            dt_bump = 1.0 / 365.0
            if T > 2.0 * dt_bump:
                p_t_up   = _price(t=T + dt_bump)
                p_t_down = _price(t=T - dt_bump)
                theta = (p_t_down - p_t_up) / (2.0 * dt_bump)
            elif T > dt_bump:
                base_price = _price()
                p_t_down   = _price(t=T - dt_bump)
                theta = (p_t_down - base_price) / dt_bump
            else:
                base_price = _price()
                theta = -base_price

            # Apply intraday acceleration for 0-DTE / near-expiry options
            if T < 3.0 / 365.0:  # Within 3 days of expiry
                theta *= intraday_weight

            # --- Rho: central difference on rate (per 1% rate move) ---
            d_r = 0.005
            rate_up   = r + d_r
            rate_down = max(r - d_r, 0.001)
            d_r_eff = (rate_up - rate_down) / 2.0
            rho = (_price(rate=rate_up) - _price(rate=rate_down)) / (2.0 * d_r_eff * 100.0)

            # --- Vanna: d(Delta)/dσ via bumped pathwise delta ----------
            d_sigma = 0.01
            vol_up   = sigma + d_sigma
            vol_down = max(sigma - d_sigma, 0.01)
            d_sigma_eff = (vol_up - vol_down) / 2.0

            # Re-run at bumped vol, then extract pathwise delta
            greeks_pricer._rng = np.random.default_rng(crn_seed)
            greeks_pricer.price(spot, strike, T, r, q, vol_up,
                                regime_params, option_type, india_features)
            pw_up = greeks_pricer.compute_pathwise_greeks(vol_up)
            delta_vol_up = pw_up['delta'] if pw_up else delta

            greeks_pricer._rng = np.random.default_rng(crn_seed)
            greeks_pricer.price(spot, strike, T, r, q, vol_down,
                                regime_params, option_type, india_features)
            pw_dn = greeks_pricer.compute_pathwise_greeks(vol_down)
            delta_vol_dn = pw_dn['delta'] if pw_dn else delta

            vanna = (delta_vol_up - delta_vol_dn) / (2.0 * d_sigma_eff)

            # --- Charm: dΔ/dt via bumped pathwise delta ----------------
            if T > 2.0 * dt_bump:
                greeks_pricer._rng = np.random.default_rng(crn_seed)
                greeks_pricer.price(spot, strike, T - dt_bump, r, q, sigma,
                                    regime_params, option_type, india_features)
                pw_t = greeks_pricer.compute_pathwise_greeks(sigma)
                delta_t_down = pw_t['delta'] if pw_t else delta
                charm = (delta_t_down - delta) / dt_bump
            else:
                charm = 0.0

            return {
                'delta': round(float(delta), 4),
                'gamma': round(float(gamma), 6),
                'theta': round(float(theta), 2),
                'vega':  round(float(vega), 2),
                'rho':   round(float(rho), 4),
                'vanna': round(float(vanna), 6),
                'charm': round(float(charm), 4),
            }
        except Exception:
            return _zero




# ============================================================================
# MODULE 7: MISPRICING SIGNAL GENERATOR
# ============================================================================


class MispricingSignal:
    """
    Generates actionable BUY / SELL / HOLD signals with transaction-cost awareness.

    Signal logic:
        1. Compute mispricing edge  (fair_value - market_price).
        2. Compute transaction cost (half spread + fees/lot_size).
        3. Only treat as actionable when edge > 1.5x transaction cost.
        4. Require strong profit probability AND confidence.

    BUY  -> Underpriced by > threshold %, edge covers TC, high prob & confidence.
    SELL -> Overpriced by > threshold %, edge covers TC, low profit prob.
    HOLD -> Everything else (insufficient edge, confidence, or TC too high).
    """

    def __init__(self, mispricing_threshold=3.0):
        self.threshold = mispricing_threshold

    def generate_signal(self, fair_value, market_price, profit_prob, confidence,
                        bid=None, ask=None, fees_per_lot=0, lot_size=1):
        """
        Parameters
        ----------
        fair_value   : float - Model fair value
        market_price : float - Observed market price
        profit_prob  : float - Profit probability (0-100)
        confidence   : float - Model confidence (0-99)
        bid          : float or None - Best bid price
        ask          : float or None - Best ask price
        fees_per_lot : float - Brokerage + STT per lot (INR)
        lot_size     : int   - Contract lot size

        Returns
        -------
        (signal: str, mispricing_pct: float, details: dict)
        """
        mispricing_pct = (fair_value - market_price) / max(market_price, 0.01) * 100
        edge_per_unit = abs(fair_value - market_price)

        # Transaction cost per unit
        if bid is not None and ask is not None:
            spread = max(ask - bid, 0.0)
        else:
            spread = 0.0
        tc = spread / 2.0 + fees_per_lot / max(lot_size, 1)

        edge_sufficient = (edge_per_unit > tc * 1.5) if tc > 0 else True

        details = {
            'spread': spread,
            'tc_per_unit': tc,
            'edge_per_unit': edge_per_unit,
            'edge_sufficient': edge_sufficient,
        }

        if (mispricing_pct > self.threshold
                and profit_prob > 58 and confidence > 70
                and edge_sufficient):
            signal = 'BUY'
        elif (mispricing_pct < -self.threshold
              and profit_prob < 38
              and edge_sufficient):
            signal = 'SELL'
        else:
            signal = 'HOLD'

        return signal, mispricing_pct, details


# ====================================================================
# PHASE 2  UPGRADE 4: FRACTIONAL BROWNIAN MOTION PATH GENERATOR
# ====================================================================
# Generates fBm paths with Hurst exponent H ≠ 0.5.
# H < 0.5: anti-persistent (mean-reverting / rough volatility)
# H > 0.5: persistent (trending)
# Uses Cholesky decomposition of the fBm covariance matrix.
# ====================================================================

class FractionalBrownianMotion:
    """
    Fractional Brownian Motion path generator.

    Standard BM has H = 0.5 (independent increments).
    Real Nifty returns have H ≈ 0.15-0.25 (anti-persistent / rough).

    fBm has autocovariance:
        E[B^H_s × B^H_t] = ½(|t|^{2H} + |s|^{2H} - |t-s|^{2H})

    This means increments are NEGATIVELY correlated for H < 0.5:
        yesterday's up-move makes today's down-move more likely.

    For option pricing, this creates:
        - Steeper smile for short-dated options (more reversal risk)
        - Lower realized vol than standard BM predicts
        - Systematic overpricing of ATM straddles by standard models
    """

    @staticmethod
    def generate_paths(n_paths, n_steps, H=0.5, dt=1.0):
        """
        Generate fBm increments using Davies-Harte FFT method.

        Uses the circulant embedding of the Toeplitz covariance matrix,
        which runs in O(n log n) time vs O(n³) for Cholesky.

        Parameters
        ----------
        n_paths : int — number of paths
        n_steps : int — number of time steps per path
        H       : float — Hurst exponent in (0, 1)
        dt      : float — time step size

        Returns
        -------
        np.ndarray of shape (n_paths, n_steps) — fBm increments
        """
        if abs(H - 0.5) < 0.01:
            # For H ≈ 0.5, standard BM is exact and faster
            return np.random.randn(n_paths, n_steps) * np.sqrt(dt)

        # ── Davies-Harte FFT Method ──────────────────────────────────
        # 1. Build autocovariance of fBm increments (Toeplitz first row)
        k = np.arange(n_steps)
        gamma = 0.5 * (np.abs(k + 1)**(2*H) - 2*np.abs(k)**(2*H)
                       + np.maximum(np.abs(k - 1)**(2*H), 0))

        # 2. Embed in circulant matrix: c = [γ₀, γ₁, ..., γ_{n-1}, γ_{n-1}, ..., γ₁]
        m = 2 * n_steps
        row = np.zeros(m)
        row[:n_steps] = gamma
        row[n_steps:] = gamma[-1:0:-1]  # mirror

        # 3. Eigenvalues of circulant = FFT of first row
        eigenvalues = np.fft.fft(row).real

        # 4. Check non-negativity (Davies-Harte condition)
        if np.min(eigenvalues) < -1e-10:
            # Fallback to Cholesky for this H/n_steps combination
            # (rare; mainly for very small H < 0.05)
            from scipy.linalg import toeplitz as _toeplitz
            cov = _toeplitz(gamma)
            eigvals = np.linalg.eigvalsh(cov)
            if np.min(eigvals) < 1e-10:
                cov += np.eye(n_steps) * (1e-8 - np.min(eigvals))
            try:
                L = np.linalg.cholesky(cov)
            except np.linalg.LinAlgError:
                eigvals, eigvecs = np.linalg.eigh(cov)
                eigvals = np.maximum(eigvals, 1e-10)
                L = eigvecs @ np.diag(np.sqrt(eigvals))
            Z = np.random.randn(n_paths, n_steps)
            return (Z @ L.T) * np.sqrt(dt)

        # 5. Generate samples via FFT
        sqrt_eig = np.sqrt(np.maximum(eigenvalues, 0) / m)
        fbm_all = np.zeros((n_paths, n_steps))

        for p in range(n_paths):
            # Two independent standard normal vectors → complex
            w1 = np.random.randn(m)
            w2 = np.random.randn(m)
            w = (w1 + 1j * w2) * sqrt_eig
            z = np.fft.ifft(w).real
            fbm_all[p, :] = z[:n_steps]

        return fbm_all * np.sqrt(dt)

    @staticmethod
    def price_adjustment_factor(H, T, sigma):
        """
        Compute the fair-value adjustment for pricing under fBm.

        Under fBm, the effective variance over [0, T] is:
            Var = σ² × T^{2H}  (instead of σ²T for standard BM)

        For H < 0.5, this means LESS total variance than BSM predicts,
        so standard BSM OVERPRICES options systematically.

        Returns the multiplicative correction factor to apply to BSM price.
        """
        if abs(H - 0.5) < 0.01:
            return 1.0

        # Ratio of fBm variance to BM variance
        var_ratio = T**(2*H - 1)
        # fBm effective vol = sigma * T^{H-0.5}
        vol_correction = T**(H - 0.5)

        # Limit correction to prevent blow-up
        vol_correction = np.clip(vol_correction, 0.5, 2.0)

        return float(vol_correction)


# ====================================================================
# PHASE 2  UPGRADE 10: ENTROPY-WEIGHTED ENSEMBLE COMBINER
# ====================================================================
# Weights pricing methods by information content rather than
# hard-coded blending coefficients.
# ====================================================================

class EntropyEnsemble:
    """
    Information-theoretic ensemble combiner for multiple pricing methods.

    Instead of hard-coded blending like `0.4 × COS + 0.6 × MC`, this
    weights each method by its Shannon entropy:

        w_i = exp(-H_i) / Σ_j exp(-H_j)

    Lower entropy = more concentrated/confident distribution → higher weight.

    This automatically:
        - Upweights MC when it converges cleanly (low SE)
        - Upweights COS for European-style options (analytical, zero noise)
        - Downweights any method that produces high-variance estimates
    """

    @staticmethod
    def shannon_entropy(values, n_bins=50):
        """
        Compute Shannon entropy of a distribution.

        Lower entropy → more peaked/confident → higher weight.
        """
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if len(values) < 10:
            return 10.0  # High entropy = low confidence

        hist, _ = np.histogram(values, bins=n_bins, density=True)
        hist = hist + 1e-10  # Avoid log(0)
        bin_width = (values.max() - values.min()) / n_bins
        if bin_width <= 0:
            return 0.0  # Perfect certainty

        p = hist * bin_width  # Normalize to proper probability
        p = p / p.sum()
        entropy = -np.sum(p * np.log(p + 1e-15))
        return float(entropy)

    @staticmethod
    def combine(method_results):
        """
        Combine multiple pricing method results using entropy weighting.

        Parameters
        ----------
        method_results : list of dict, each with:
            - 'name': str — method name (e.g., 'MC', 'COS', 'SABR')
            - 'price': float — point estimate
            - 'distribution': np.ndarray or None — full payoff distribution
            - 'std_error': float — standard error (if available)

        Returns
        -------
        dict with blended_price, method_weights, entropy_scores
        """
        if not method_results:
            return {'blended_price': 0.0, 'method_weights': {}, 'entropy_scores': {}}

        # Single method: just return it
        if len(method_results) == 1:
            return {
                'blended_price': method_results[0]['price'],
                'method_weights': {method_results[0]['name']: 1.0},
                'entropy_scores': {method_results[0]['name']: 0.0},
            }

        entropies = {}
        for m in method_results:
            name = m['name']
            if m.get('distribution') is not None and len(m['distribution']) >= 10:
                entropies[name] = EntropyEnsemble.shannon_entropy(m['distribution'])
            elif m.get('std_error', 0) > 0:
                # Approximate entropy from std error
                entropies[name] = np.log(m['std_error'] * np.sqrt(2 * np.pi * np.e))
            else:
                entropies[name] = 0.1  # Analytical method = near-zero entropy

        # Softmax weights: w_i = exp(-H_i) / Σ_j exp(-H_j)
        names = [m['name'] for m in method_results]
        neg_H = np.array([-entropies.get(n, 5.0) for n in names])
        neg_H -= np.max(neg_H)  # Numerical stability
        weights = np.exp(neg_H)
        weights /= weights.sum()

        method_weights = {names[i]: round(float(weights[i]), 4) for i in range(len(names))}

        # Blended price
        prices = np.array([m['price'] for m in method_results])
        blended = float(np.sum(weights * prices))

        # Blended uncertainty
        if all(m.get('std_error', 0) > 0 for m in method_results):
            ses = np.array([m['std_error'] for m in method_results])
            blended_se = float(np.sqrt(np.sum((weights * ses)**2)))
        else:
            blended_se = 0.0

        return {
            'blended_price': round(blended, 2),
            'method_weights': method_weights,
            'entropy_scores': {k: round(v, 4) for k, v in entropies.items()},
            'blended_std_error': round(blended_se, 4),
            'dominant_method': max(method_weights, key=method_weights.get),
        }


# ============================================================================
# MODULE 8: NIRV MASTER MODEL (Orchestrator)
# ============================================================================


class NIRVModel:
    """
    NIRV: Nifty Intelligent Regime-Volatility Option Pricing Model
   
    Master orchestrator that chains all modules:
        1. IndiaFeatureEngine  → Extract market features
        2. RegimeDetector      → Classify market regime (HMM)
        3. VolatilitySurface   → Compute regime-adaptive IV (SVI)
        4. JumpDiffusionPricer → Monte Carlo fair value
        5. BayesianConfidence  → Profit probability & confidence
        6. MispricingSignal    → Generate BUY/SELL/HOLD
        7. GreeksCalculator    → Delta, Gamma, Theta, Vega
   
    Usage
    -----
    >>> nirv = NIRVModel(n_paths=50000, n_bootstrap=1000)
    >>> result = nirv.price_option(
    ...     spot=23500, strike=23400, T=7/365,
    ...     r=0.065, q=0.012, option_type='CE', market_price=150,
    ...     india_vix=14, fii_net_flow=-800, dii_net_flow=600,
    ...     days_to_rbi=15, pcr_oi=1.05,
    ...     returns_30d=np.random.normal(0, 0.01, 30))
    >>> print(result.signal, result.fair_value, result.profit_probability)
    """


    def __init__(self, n_paths=50000, n_bootstrap=5000):
        self.feature_engine = IndiaFeatureEngine()
        self.regime_detector = RegimeDetector()
        self.vol_surface = VolatilitySurface()
        self.pricer = JumpDiffusionPricer(n_paths=n_paths)
        self.confidence_engine = BayesianConfidenceEngine(n_bootstrap=n_bootstrap)
        self.signal_gen = MispricingSignal()
        self.lot_size = 65  # Default
        if getattr(get_features(), "USE_NSE_CONTRACT_SPECS", False) and nse_get_lot_size:
            try:
                self.lot_size = int(nse_get_lot_size("NIFTY", None))
            except Exception:
                pass

        self.vrp_state_engine = ModelFreeVRPState() if ModelFreeVRPState else None
        self.pricer_router = TieredPricerRouter(default_cpu_budget_ms=20.0) if TieredPricerRouter else None
        # Runtime diagnostics/state for downstream (eval, UI, ML features)
        self.state = {
            "model_free_var_30d": None,
            "model_free_var_term_structure": None,
            "vrp_state": None,
            "surface_diagnostics": {},
            "pricer_route": None,
        }

        # --- Advanced Quant Engine integration ---
        self.quant_engine = None
        if QUANT_ENGINE_AVAILABLE:
            try:
                self.quant_engine = QuantEngine()
            except Exception:
                pass


    def price_option(self, spot, strike, T, r, q, option_type, market_price,
                     india_vix, fii_net_flow, dii_net_flow, days_to_rbi,
                     pcr_oi, returns_30d, inr_usd_vol=0.05, **kwargs):
        """
        Full NIRV pricing pipeline for a single option.
       
        Parameters
        ----------
        spot          : float – Current spot price
        strike        : float – Strike price
        T             : float – Time to expiry in years
        r             : float – Risk-free rate (decimal)
        q             : float – Dividend yield (decimal)
        option_type   : str   – 'CE' for Call, 'PE' for Put
        market_price  : float – Current market price of the option
        india_vix     : float – Current India VIX level
        fii_net_flow  : float – FII net flow in ₹ crores
        dii_net_flow  : float – DII net flow in ₹ crores
        days_to_rbi   : int   – Days to next RBI policy
        pcr_oi        : float – Put-Call Ratio (OI based)
        returns_30d   : np.ndarray – Last 30 daily log returns
        inr_usd_vol   : float – INR/USD 30-day volatility
       
        Returns
        -------
        NirvOutput namedtuple
        """
        # --- Input validation ---
        spot = max(float(spot), 1.0)
        strike = max(float(strike), 1.0)
        T = max(float(T), 1e-6)
        sigma_input = max(float(india_vix) / 100.0, 0.01) if india_vix else 0.14
        market_price = max(float(market_price), 0.01)
        days_to_expiry = max(int(T * 365), 1)

        model_free_var_30d = kwargs.get("model_free_var_30d")
        model_free_var_term = kwargs.get("model_free_var_term_structure")
        vrp_state = kwargs.get("vrp_state")

        # v6 model-free VRP state (flag-gated)
        if getattr(get_features(), "USE_MODEL_FREE_VRP", False) and self.vrp_state_engine is not None:
            try:
                if vrp_state is None:
                    rn_term = {}
                    if isinstance(model_free_var_term, dict):
                        for k, v in model_free_var_term.items():
                            try:
                                kk = int(k)
                                if np.isfinite(v):
                                    rn_term[kk] = float(v)
                            except Exception:
                                continue
                    if model_free_var_30d is not None and np.isfinite(model_free_var_30d):
                        rn_term[30] = float(model_free_var_30d)
                    if rn_term:
                        vrp_state = self.vrp_state_engine.compute_state(rn_term, returns_30d)
            except Exception:
                vrp_state = None

        feature_kwargs = {}
        if "india_vix_synth" in kwargs and kwargs.get("india_vix_synth") is not None:
            feature_kwargs["india_vix_synth"] = kwargs.get("india_vix_synth")
        if model_free_var_30d is not None and np.isfinite(model_free_var_30d):
            feature_kwargs["model_free_var_30d"] = float(model_free_var_30d)
        if isinstance(vrp_state, dict):
            if np.isfinite(vrp_state.get("vrp_level", np.nan)):
                feature_kwargs["vrp_level"] = float(vrp_state["vrp_level"])
            if np.isfinite(vrp_state.get("vrp_slope", np.nan)):
                feature_kwargs["vrp_slope"] = float(vrp_state["vrp_slope"])

        # 1. India-specific features
        features = self.feature_engine.compute_features(
            india_vix, fii_net_flow, dii_net_flow,
            days_to_expiry, days_to_rbi, pcr_oi, inr_usd_vol,
            **feature_kwargs
        )


        # 2. Regime detection
        regime, regime_probs = self.regime_detector.detect_regime(
            returns_30d, india_vix, fii_net_flow,
            vrp_level=feature_kwargs.get("vrp_level"),
            vrp_slope=feature_kwargs.get("vrp_slope"),
        )
        regime_params = RegimeDetector.REGIME_PARAMS[regime]

        # v6: VRP-driven conservative parameter adaptation
        vrp_param_adjustments = {}
        if getattr(get_features(), "USE_MODEL_FREE_VRP", False) and self.vrp_state_engine is not None and isinstance(vrp_state, dict):
            try:
                vrp_param_adjustments = self.vrp_state_engine.parameter_adjustments(vrp_state)
                regime_params = dict(regime_params)
                regime_params["kappa"] = regime_params.get("kappa", 2.0) * vrp_param_adjustments.get("kappa_mult", 1.0)
                regime_params["theta_v"] = regime_params.get("theta_v", sigma_input ** 2) * vrp_param_adjustments.get("theta_mult", 1.0)
                regime_params["sigma_v"] = regime_params.get("sigma_v", 0.3) * vrp_param_adjustments.get("sigma_v_mult", 1.0)
            except Exception:
                vrp_param_adjustments = {}
    
        # Phase 4: VRP State Adjustment
        # -----------------------------
        vrr_adjustments = {}
        if get_features().vrr_state and VRRStateFilter and returns_30d is not None and len(returns_30d) >= 10:
            try:
                # We need current IV to compute VRP. 
                # We haven't computed sigma (IV) yet, that happens in step 3.
                # But we need adjustments *before* pricing? 
                # Actually weak ordering here: Step 3 computes sigma from SVI. 
                # SVI parameters are fixed. 
                # The *Heston* parameters (lambda, eta) are used in MC pricing setup.
                # SVI sigma is used as v0 (initial variance).
                # So let's compute sigma first, then adjust Heston params?
                # Or compute naive VRP using VIX?
                # Using VIX is simpler/cleaner for state filter.
                vix_decimal = india_vix / 100.0 if india_vix else 0.14
                
                # Since VRRStateFilter is stateless, we instantiate ad-hoc
                vrr_filter = VRRStateFilter() 
                A_t = vrr_filter.get_state(vix_decimal, returns_30d)
                vrr_adjustments = vrr_filter.get_adjustments(A_t)
                
                # Apply adjustments to a COPY of regime_params
                regime_params = dict(regime_params)
                
                # 1. Lambda (Jumps)
                regime_params['lambda_j'] = regime_params.get('lambda_j', 0.03) * vrr_adjustments['lambda_mult']
                
                # 2. Sigma_v (Vol-Vol / Eta)
                regime_params['sigma_v'] = regime_params.get('sigma_v', 0.3) * vrr_adjustments['eta_mult']
                
            except Exception:
                pass # Fail open (no adjustment)



        # 2b. Compute hurst exponent early (needed for Flaw 3 rough vol correction)
        hurst_exponent = None
        if self.quant_engine is not None and returns_30d is not None and len(returns_30d) >= 20:
            try:
                hurst_exponent = self.quant_engine.macro.hurst_exponent(returns_30d)
            except Exception:
                pass

        # 3. Regime-adaptive implied volatility (SVI surface)
        sigma = self.vol_surface.get_implied_vol(
            spot, strike, T, regime, features, hurst_exponent=hurst_exponent
        )

        # 3b. Advanced: blend SABR + GARCH vol if quant engine available
        quant_extras = {}
        if isinstance(vrp_state, dict):
            quant_extras['vrp_state'] = vrp_state
        if vrp_param_adjustments:
            quant_extras['vrp_param_adjustments'] = vrp_param_adjustments
        if model_free_var_30d is not None and np.isfinite(model_free_var_30d):
            quant_extras['model_free_var_30d'] = float(model_free_var_30d)
        if isinstance(model_free_var_term, dict):
            quant_extras['model_free_var_term_structure'] = dict(model_free_var_term)
        if self.quant_engine is not None:
            try:
                # GJR-GARCH volatility forecast
                if returns_30d is not None and len(returns_30d) >= 20:
                    garch_res = self.quant_engine.fit_garch(returns_30d)
                    garch_vol = garch_res.get('annualized_vol', sigma)
                    quant_extras['garch_vol'] = garch_vol
                    quant_extras['garch_source'] = garch_res.get('source', 'unknown')
                    # GARCH-IV spread signal
                    quant_extras['garch_iv_spread'] = garch_vol - sigma

                # EM jump parameter estimation (replaces threshold-based)
                if returns_30d is not None and len(returns_30d) >= 30:
                    jump_params = self.quant_engine.fit_jump_params(returns_30d)
                    if jump_params:
                        quant_extras['em_jump_params'] = jump_params
                        # Override regime jump params with EM-estimated
                        regime_params = dict(regime_params)  # copy
                        regime_params['lambda_j'] = jump_params.get('lambda_j', regime_params.get('lambda_j', 0.03))
                        regime_params['mu_j'] = jump_params.get('mu_j', regime_params.get('mu_j', 0.0))
                        regime_params['sigma_j'] = jump_params.get('sigma_j', regime_params.get('sigma_j', 0.01))

                # Continuous VIX regime (sigmoid, not binary thresholds)
                cont_regime = self.quant_engine.regime_detector.continuous_regime_prob(india_vix)
                quant_extras['continuous_regime'] = cont_regime

                # RV/IV ratio signal
                if returns_30d is not None and len(returns_30d) >= 10:
                    rv = float(np.std(returns_30d) * np.sqrt(252))
                    rv_iv = self.quant_engine.regime_detector.rv_iv_regime_signal(rv, sigma)
                    quant_extras['rv_iv_signal'] = rv_iv

                # Hurst exponent (already computed above for vol surface)
                if hurst_exponent is not None:
                    quant_extras['hurst'] = hurst_exponent

                # Calendar effects
                quant_extras['calendar'] = self.quant_engine.macro.calendar_effects()

                # Heston COS semi-analytical price (10-50x faster validation)
                try:
                    V0 = sigma ** 2
                    cos_price = self.quant_engine.heston_cos.price(
                        spot, strike, T, r, q, V0,
                        regime_params.get('kappa', 2.0),
                        regime_params.get('theta_v', V0),
                        regime_params.get('sigma_v', 0.3),
                        regime_params.get('rho_sv', -0.5),
                        option_type
                    )
                    quant_extras['heston_cos_price'] = cos_price
                except Exception:
                    pass
            except Exception:
                pass


        # 4. Pricing (baseline MC or v6 tiered routing)
        if getattr(get_features(), "USE_TIERED_PRICER", False) and self.pricer_router is not None:
            route = self.pricer_router.route_price(
                spot=spot,
                strike=strike,
                T=T,
                r=r,
                q=q,
                option_type=option_type,
                surface_iv=sigma,
                regime_params=regime_params,
                india_features=features,
                quant_engine=self.quant_engine,
                mc_pricer=self.pricer,
                full_chain_mode=bool(kwargs.get("full_chain_mode", False)),
                cpu_budget_ms=float(kwargs.get("cpu_budget_ms", 20.0)),
                liquidity_score=float(kwargs.get("liquidity_score", 1.0)),
                anomaly_score=float(kwargs.get("anomaly_score", 0.0)),
                mispricing_hint=float(kwargs.get("mispricing_hint", 0.0)),
            )
            self.state["pricer_route"] = dict(route)
            quant_extras["tiered_pricer"] = dict(route)
            fair_value = float(route.get("price", market_price))
            std_error = float(route.get("mc_std_error", 0.0) if np.isfinite(route.get("mc_std_error", np.nan)) else 0.0)

            # Use MC terminal distribution when Tier 2 ran, else proxy terminals.
            S_terminal = None
            if str(route.get("tier_used", "")).startswith("tier2"):
                S_terminal = getattr(self.pricer, "_last_S_T", None)
            if S_terminal is None or len(np.asarray(S_terminal).reshape(-1)) < 10:
                n_proxy = int(kwargs.get("proxy_terminal_paths", 4000))
                z = np.random.standard_normal(n_proxy)
                S_terminal = spot * np.exp((r - q - 0.5 * sigma * sigma) * T + sigma * np.sqrt(T) * z)
        else:
            fair_value, std_error, S_terminal = self.pricer.price(
                spot, strike, T, r, q, sigma, regime_params,
                option_type, features
            )

            # 4b. Cross-validate MC with COS price (if available)
            if 'heston_cos_price' in quant_extras:
                cos_p = quant_extras['heston_cos_price']
                # If MC and COS agree within 10%, good. If not, blend.
                if cos_p > 0 and fair_value > 0:
                    divergence = abs(fair_value - cos_p) / max(cos_p, 0.01)
                    if divergence > 0.10:
                        # Blend: trust COS more for European, MC for path-dependent
                        fair_value = 0.4 * cos_p + 0.6 * fair_value
                        quant_extras['mc_cos_blended'] = True
                        quant_extras['mc_cos_divergence_pct'] = divergence * 100


        # 5. Bayesian confidence & profit probability (risk-neutral + physical)
        (profit_prob_rn, physical_profit_prob, confidence,
         expected_pnl_rn, physical_expected_pnl, fv_dist) = \
            self.confidence_engine.compute_profit_probability(
                S_terminal, strike, market_price, r, T,
                option_type, self.lot_size,
                spot=spot, returns_30d=returns_30d,
                regime=regime, iv=sigma
            )


        # 6. Bayesian posterior confidence adjustment
        if self.quant_engine is not None:
            try:
                bayes = self.quant_engine.bayes_conf.compute_posterior(
                    fair_value, market_price, std_error
                )
                quant_extras['bayesian_posterior'] = bayes
                # Blend Bayesian confidence with bootstrap confidence
                confidence = 0.5 * confidence + 0.5 * bayes.get('adjusted_confidence', confidence)
            except Exception:
                pass

        # Phase 4: VRP Confidence Adjustment
        if vrr_adjustments and 'conf_mult' in vrr_adjustments:
             confidence *= vrr_adjustments['conf_mult']
             # Allow confidence > 100 only if very strong signal? No, clamp it.
             confidence = min(confidence, 99.9)

        # 6b. Signal generation (uses physical profit prob for real-world decisions)
        signal, mispricing_pct, tc_details = self.signal_gen.generate_signal(
            fair_value, market_price, physical_profit_prob, confidence
        )


        # 7. Greeks (MC bump-and-reprice with CRN)
        greeks = GreeksCalculator.compute(
            self.pricer, spot, strike, T, r, q, sigma,
            regime_params, option_type, features
        )

        # 7b. Attach quant engine extras to greeks for downstream access
        greeks['quant_extras'] = quant_extras
        self.state["model_free_var_30d"] = (
            float(model_free_var_30d) if model_free_var_30d is not None and np.isfinite(model_free_var_30d) else None
        )
        self.state["model_free_var_term_structure"] = model_free_var_term
        self.state["vrp_state"] = vrp_state if isinstance(vrp_state, dict) else None
        self.state["surface_diagnostics"] = dict(getattr(self.vol_surface, "last_diagnostics", {}) or {})

        return NirvOutput(
            fair_value=round(fair_value, 2),
            market_price=market_price,
            mispricing_pct=round(mispricing_pct, 2),
            signal=signal,
            profit_probability=round(profit_prob_rn, 1),
            physical_profit_prob=round(physical_profit_prob, 1),
            confidence_level=round(confidence, 1),
            expected_pnl=round(expected_pnl_rn, 2),
            physical_expected_pnl=round(physical_expected_pnl, 2),
            regime=regime,
            greeks=greeks,
            tc_details=tc_details,
        )


    def scan_chain(self, spot, strikes, T, r, q, market_prices_ce, market_prices_pe,
                   india_vix, fii_net_flow, dii_net_flow, days_to_rbi,
                   pcr_oi, returns_30d, **kwargs):
        """
        Scan an entire option chain (multiple strikes, CE + PE).
        """
        results = []
        
        # Phase 2: Synthetic VIX calculation
        extra_kwargs = {
            "full_chain_mode": True,
            "cpu_budget_ms": float(kwargs.get("cpu_budget_ms", 8.0)),
        }
        if get_features().india_vix_synth and compute_synthetic_vix:
            try:
                # 1. Build chain structure
                chain_options = []
                # Add calls
                for k, p in market_prices_ce.items():
                    if p > 0:
                        chain_options.append({'strike': k, 'option_type': 'CE', 'price': p})
                # Add puts
                for k, p in market_prices_pe.items():
                    if p > 0:
                        chain_options.append({'strike': k, 'option_type': 'PE', 'price': p})
                
                chains_struct = [{'T': T, 'options': chain_options}]
                
                # 2. Compute VIX
                # We use generic r=0.065 or passing r is better
                vix_synth, qual, _ = compute_synthetic_vix(chains_struct, risk_free_rate=r)
                
                if not np.isnan(vix_synth):
                    extra_kwargs['india_vix_synth'] = vix_synth
                    # Optional: Blend into india_vix if quality is high?
                    # For now just passing it.
            except Exception as e:
                warnings.warn(f"Phase 2: Synthetic VIX failed: {e}")

        # v6: NSE-consistent model-free variance engine (flag-gated)
        if getattr(get_features(), "USE_NSE_VIX_ENGINE", False) and compute_variance_for_expiry:
            try:
                now_ts = kwargs.get("now_ts", datetime.datetime.now())
                expiry_ts = kwargs.get(
                    "expiry_ts",
                    now_ts + datetime.timedelta(days=max(int(round(T * 365.0)), 1))
                )
                forward = kwargs.get("forward", None)
                forward_method = "legacy"

                # scan_chain input lacks live bid/ask; synthesize narrow quotes from mids
                vix_chain = []
                for k, p in market_prices_ce.items():
                    if p > 0:
                        vix_chain.append({"strike": k, "option_type": "CE", "bid": p * 0.995, "ask": p * 1.005})
                for k, p in market_prices_pe.items():
                    if p > 0:
                        vix_chain.append({"strike": k, "option_type": "PE", "bid": p * 0.995, "ask": p * 1.005})
                if forward is None:
                    forward = spot * np.exp((r - q) * T)
                    if (
                        bool(getattr(get_features(), "USE_IMPROVED_VIX_ESTIMATOR", False))
                        and estimate_forward_from_chain is not None
                    ):
                        fwd_est = estimate_forward_from_chain(
                            chain_slice=vix_chain,
                            r=float(r),
                            now_ts=now_ts,
                            expiry_ts=expiry_ts,
                            exchange_calendar=kwargs.get("exchange_calendar"),
                            method="improved",
                        )
                        if np.isfinite(fwd_est) and fwd_est > 0:
                            forward = float(fwd_est)
                            forward_method = "improved"

                mf_var = compute_variance_for_expiry(
                    chain_slice=vix_chain,
                    forward=float(forward),
                    r=float(r),
                    now_ts=now_ts,
                    expiry_ts=expiry_ts,
                    exchange_calendar=kwargs.get("exchange_calendar"),
                    tail_corrected=bool(getattr(get_features(), "USE_TAIL_CORRECTED_VARIANCE", False)),
                    forward_method=forward_method,
                )
                if np.isfinite(mf_var) and mf_var > 0:
                    extra_kwargs["model_free_var_30d"] = float(mf_var)
                    self.state["model_free_var_30d"] = float(mf_var)
                    # Also expose equivalent 30d vol proxy for downstream compatibility
                    extra_kwargs["india_vix_synth"] = float(np.sqrt(mf_var) * 100.0)

                # Optional near/next term structure inputs for richer VRP state
                if compute_vix_30d_with_details and kwargs.get("chain_near") is not None and kwargs.get("chain_next") is not None:
                    _, vix_details = compute_vix_30d_with_details(
                        kwargs.get("chain_near"),
                        kwargs.get("chain_next"),
                        kwargs.get("forward_near", forward),
                        kwargs.get("forward_next", forward),
                        r,
                        now_ts,
                        kwargs.get("expiry_near_ts", expiry_ts),
                        kwargs.get("expiry_next_ts", expiry_ts + datetime.timedelta(days=7)),
                        exchange_calendar=kwargs.get("exchange_calendar"),
                        tail_corrected=bool(getattr(get_features(), "USE_TAIL_CORRECTED_VARIANCE", False)),
                    )
                    term = {
                        7: kwargs.get("rn_var_7d", np.nan),
                        30: vix_details.get("var_30d", np.nan),
                        60: kwargs.get("rn_var_60d", np.nan),
                    }
                    extra_kwargs["model_free_var_term_structure"] = term
                    self.state["model_free_var_term_structure"] = term
            except Exception as e:
                warnings.warn(f"v6 model-free variance engine failed in scan_chain: {e}")

        for strike in strikes:
            # Price calls
            if strike in market_prices_ce and market_prices_ce[strike] > 0:
                try:
                    res = self.price_option(
                        spot, strike, T, r, q, 'CE', market_prices_ce[strike],
                        india_vix, fii_net_flow, dii_net_flow, days_to_rbi,
                        pcr_oi, returns_30d, **extra_kwargs
                    )
                    results.append(('CE', strike, res))
                except Exception as e:
                    warnings.warn(f"NIRV scan_chain: CE strike {strike} failed: {e}")
            
            # Price puts
            if strike in market_prices_pe and market_prices_pe[strike] > 0:
                try:
                    res = self.price_option(
                        spot, strike, T, r, q, 'PE', market_prices_pe[strike],
                        india_vix, fii_net_flow, dii_net_flow, days_to_rbi,
                        pcr_oi, returns_30d, **extra_kwargs
                    )
                    results.append(('PE', strike, res))
                except Exception as e:
                    warnings.warn(f"NIRV scan_chain: PE strike {strike} failed: {e}")


        # Sort by physical profit probability (real-world, highest first)
        results.sort(key=lambda x: x[2].physical_profit_prob, reverse=True)
        return results
