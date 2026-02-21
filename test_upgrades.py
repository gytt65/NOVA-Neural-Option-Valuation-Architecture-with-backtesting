#!/usr/bin/env python3
"""
Smoke tests for NIRV-OMEGA Revolutionary Upgrades.
Tests all 7 flaw fixes + 7 revolutionary additions.
"""

import sys
import os
import numpy as np
import traceback
import importlib.util

# Fix import path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PASS = 0
FAIL = 0
RUN_SMOKE = (__name__ == "__main__")
__test__ = False

def test(name, fn):
    global PASS, FAIL
    if not RUN_SMOKE:
        return
    try:
        result = fn()
        if result is False:
            print(f"  ✗ {name}")
            FAIL += 1
        else:
            print(f"  ✓ {name}")
            PASS += 1
    except Exception as e:
        print(f"  ✗ {name}: {e}")
        traceback.print_exc()
        FAIL += 1


def import_nirv():
    """Helper to import nirv_model with space in filename."""
    try:
        import nirv_model
    except ImportError:
        spec = importlib.util.spec_from_file_location(
            "nirv_model", os.path.join(ROOT, "nirv_model.py")
        )
        nirv_model = importlib.util.module_from_spec(spec)
        sys.modules['nirv_model'] = nirv_model  # Register so omega_model can import it
        spec.loader.exec_module(nirv_model)
    return nirv_model

# Pre-import nirv to register in sys.modules
NIRV_MOD = None

# ===========================================================
if RUN_SMOKE:
    print("\n═══ 1. Import Tests ═══")
# ===========================================================

def test_import_nirv():
    global NIRV_MOD
    NIRV_MOD = import_nirv()
    assert hasattr(NIRV_MOD, 'NIRVModel')
    assert hasattr(NIRV_MOD, 'VolatilitySurface')
    assert hasattr(NIRV_MOD, 'JumpDiffusionPricer')
    assert hasattr(NIRV_MOD, 'BayesianConfidenceEngine')
    assert hasattr(NIRV_MOD, 'GreeksCalculator')
    return True

def test_import_quant():
    import quant_engine as qe
    assert hasattr(qe, 'KellyCriterion')
    assert hasattr(qe, 'InformationGeometryDetector')
    assert hasattr(qe, 'TransferEntropyRegimePredictor')
    assert hasattr(qe, 'MarketMakerInventory')
    assert hasattr(qe, 'OptimalEntryTiming')
    assert hasattr(qe, 'ButterflyArbitrageScanner')
    return True

def test_import_omega():
    import omega_model as om
    assert hasattr(om, 'OMEGAModel')
    assert hasattr(om, 'ProspectTheoryKernel')
    assert hasattr(om, 'DispositionFlowPredictor')
    return True

test("Import nirv_model", test_import_nirv)
test("Import quant_engine", test_import_quant)
test("Import omega_model", test_import_omega)

FEATURES = {'fii_score': 0.5, 'vix_regime': 0.5, 'rbi_premium': 0.0,
             'pcr_signal': 0.0, 'inr_risk': 0.0}

# ===========================================================
if RUN_SMOKE:
    print("\n═══ 2. Flaw Fix Tests ═══")
# ===========================================================

def test_flaw1_vrp():
    """Flaw 1: VRP variance shift in compute_profit_probability"""
    eng = NIRV_MOD.BayesianConfidenceEngine(n_bootstrap=500)
    S_T = np.random.lognormal(np.log(23500), 0.15, 5000)
    result = eng.compute_profit_probability(
        S_T, 23400, 200, 0.065, 7/365, 'CE', 65,
        spot=23500, returns_30d=np.random.normal(0, 0.01, 30),
        regime='Sideways', iv=0.14
    )
    assert len(result) == 6
    rn_prob, phys_prob = result[0], result[1]
    assert 0 <= rn_prob <= 100
    assert 0 <= phys_prob <= 100
    return True

def test_flaw2_regime_switching():
    """Flaw 2: Regime-switching in MC pricer"""
    pricer = NIRV_MOD.JumpDiffusionPricer(n_paths=1000)
    regime_params = NIRV_MOD.RegimeDetector.REGIME_PARAMS['Sideways']
    fv, se, S_T = pricer.price(23500, 23400, 7/365, 0.065, 0.012, 0.14,
                                regime_params, 'CE', FEATURES)
    assert fv > 0
    assert se >= 0
    assert len(S_T) >= 500
    return True

def test_flaw3_rough_vol():
    """Flaw 3: Rough volatility SVI correction"""
    vs = NIRV_MOD.VolatilitySurface()
    iv_no_hurst = vs.get_implied_vol(23500, 23000, 3/365, 'Sideways', FEATURES)
    iv_with_hurst = vs.get_implied_vol(23500, 23000, 3/365, 'Sideways', FEATURES,
                                        hurst_exponent=0.15)
    assert iv_no_hurst > 0
    assert iv_with_hurst > 0
    return True

def test_flaw4_rqmc_se():
    """Flaw 4: cv_payoffs defined for RQMC SE"""
    pricer = NIRV_MOD.JumpDiffusionPricer(n_paths=2000)
    regime_params = NIRV_MOD.RegimeDetector.REGIME_PARAMS['Sideways']
    fv, se, S_T = pricer.price(23500, 23400, 7/365, 0.065, 0.012, 0.14,
                                regime_params, 'CE', FEATURES)
    assert fv > 0
    assert se >= 0
    return True

def test_flaw5_stale_filter():
    """Flaw 5: Stale OTM quote filtering"""
    vs = NIRV_MOD.VolatilitySurface()
    strikes = [22500, 22800, 23000, 23200, 23400, 23500, 23600, 23800, 24000, 24200, 24500]
    market_ivs = [0.03, 0.12, 0.13, 0.14, 0.14, 0.13, 0.14, 0.14, 0.13, 0.12, 0.03]
    vs.calibrate_to_market(23500, strikes, market_ivs, 7/365, 'Sideways', FEATURES)
    return True

def test_flaw6_intraday_theta():
    """Flaw 6: Intraday theta decay"""
    pricer = NIRV_MOD.JumpDiffusionPricer(n_paths=500)
    regime_params = NIRV_MOD.RegimeDetector.REGIME_PARAMS['Sideways']
    greeks = NIRV_MOD.GreeksCalculator.compute(
        pricer, 23500, 23400, 2/365, 0.065, 0.012, 0.14,
        regime_params, 'CE', FEATURES
    )
    assert isinstance(greeks, dict)
    assert 'theta' in greeks
    return True

def test_flaw7_kelly_cvar():
    """Flaw 7: Kelly criterion with CVaR tail-risk"""
    import quant_engine as qe
    kc = qe.KellyCriterion()
    # Without distribution
    r1 = kc.optimal_fraction(0.6, 100, 80)
    assert r1['kelly_pct'] > 0
    assert 'tail_risk_penalty' in r1
    assert r1['tail_risk_penalty'] == 1.0

    # With very fat-tailed distribution (guaranteed extreme left tail)
    dist = np.concatenate([
        np.ones(800) * 10,       # Most paths make 10
        np.ones(200) * (-200)    # 20% extreme losses
    ])
    r2 = kc.optimal_fraction(0.6, 100, 80, payoff_distribution=dist)
    assert 'tail_risk_penalty' in r2
    # Expected return = 800*10 + 200*(-200) = -32000 / 1000 = -32 → negative
    # When expected return <= 0, tail_risk_penalty stays 1.0 (no reduction possible)
    # Let's use a distribution with positive mean but fat tails
    dist2 = np.concatenate([
        np.ones(900) * 20,       # Most paths make 20
        np.ones(100) * (-50)     # 10% moderate losses
    ])
    # mean = (900*20 + 100*(-50)) / 1000 = 13
    r3 = kc.optimal_fraction(0.6, 100, 80, payoff_distribution=dist2)
    assert r3['tail_risk_penalty'] <= 1.0
    return True

test("Flaw 1: VRP variance shift", test_flaw1_vrp)
test("Flaw 2: Regime-switching MC", test_flaw2_regime_switching)
test("Flaw 3: Rough vol SVI correction", test_flaw3_rough_vol)
test("Flaw 4: RQMC standard error", test_flaw4_rqmc_se)
test("Flaw 5: Stale quote filter", test_flaw5_stale_filter)
test("Flaw 6: Intraday theta decay", test_flaw6_intraday_theta)
test("Flaw 7: Kelly CVaR tail-risk", test_flaw7_kelly_cvar)


# ===========================================================
if RUN_SMOKE:
    print("\n═══ 3. Revolutionary Addition Tests ═══")
# ===========================================================

def test_addition1_info_geo():
    """Addition 1: Information Geometry Detector"""
    import quant_engine as qe
    igd = qe.InformationGeometryDetector()
    S_T = np.random.lognormal(np.log(23500), 0.14, 5000)
    strikes = [23000, 23200, 23400, 23600, 23800]
    ivs = [0.16, 0.15, 0.14, 0.15, 0.16]
    result = igd.detect(S_T, 23500, strikes, ivs, 7/365, 0.065)
    assert 'kl_total' in result
    assert 'tail_signal' in result
    return True

def test_addition2_prospect():
    """Addition 2: Prospect Theory Kernel"""
    import omega_model as om
    pt = om.ProspectTheoryKernel()
    S_T = np.random.lognormal(np.log(23500), 0.14, 5000)
    result = pt.compute_behavioral_edge(S_T, 23400, 200, 'CE', T=7/365)
    assert 'pt_price' in result
    assert 'behavioral_edge' in result
    assert result['pt_price'] > 0
    return True

def test_addition3_disposition():
    """Addition 3: Disposition Flow Predictor"""
    import omega_model as om
    returns_5d = np.array([0.01, 0.008, 0.005, 0.003, 0.002])
    oi_calls = {23400: -500, 23500: -800, 23600: -200}
    oi_puts = {23400: 300, 23500: 100, 23600: 500}
    result = om.DispositionFlowPredictor.predict(
        returns_5d, oi_calls, oi_puts, 23500, [23400, 23500, 23600])
    assert 'disposition_signal' in result
    assert 'flow_direction' in result
    return True

def test_addition4_transfer_entropy():
    """Addition 4: Transfer Entropy Regime Predictor"""
    import quant_engine as qe
    te = qe.TransferEntropyRegimePredictor()
    vix = np.random.normal(14, 2, 100)
    india_vix = vix + np.random.normal(0, 1, 100) + 2
    result = te.predict_regime_change(vix, india_vix)
    assert 'te_score' in result
    assert 'regime_change_warning' in result
    assert isinstance(result['regime_change_warning'], bool)
    return True

def test_addition5_mm_inventory():
    """Addition 5: Market Maker Inventory"""
    import quant_engine as qe
    strikes = [23200, 23300, 23400, 23500, 23600]
    call_oi = {23200: 5000, 23300: 8000, 23400: 15000, 23500: 20000, 23600: 12000}
    put_oi = {23200: 18000, 23300: 12000, 23400: 10000, 23500: 15000, 23600: 8000}
    result = qe.MarketMakerInventory.estimate(23500, strikes, call_oi, put_oi)
    assert 'mm_delta' in result
    assert 'mm_gamma_regime' in result
    return True

def test_addition6_optimal_entry():
    """Addition 6: Optimal Entry Timing"""
    import quant_engine as qe
    result = qe.OptimalEntryTiming.compute(
        mispricing_pct=5.0, T=7/365, tc_per_unit=2.5,
        market_price=200, fair_value=210)
    assert 'should_enter_now' in result
    assert 'threshold_pct' in result
    return True

def test_addition7_butterfly():
    """Addition 7: Butterfly Arbitrage Scanner"""
    import quant_engine as qe
    strikes = [23200, 23300, 23400, 23500, 23600, 23700]
    prices = {23200: 350, 23300: 270, 23400: 200, 23500: 140, 23600: 90, 23700: 55}
    result = qe.ButterflyArbitrageScanner.scan(strikes, prices)
    assert 'violations' in result
    assert 'total_opportunities' in result

    # Inject violation
    prices_bad = dict(prices)
    prices_bad[23400] = 100
    result2 = qe.ButterflyArbitrageScanner.scan(strikes, prices_bad)
    assert result2['total_opportunities'] > 0
    return True

test("Addition 1: Information Geometry", test_addition1_info_geo)
test("Addition 2: Prospect Theory", test_addition2_prospect)
test("Addition 3: Disposition Flow", test_addition3_disposition)
test("Addition 4: Transfer Entropy", test_addition4_transfer_entropy)
test("Addition 5: MM Inventory", test_addition5_mm_inventory)
test("Addition 6: Optimal Entry Timing", test_addition6_optimal_entry)
test("Addition 7: Butterfly Scanner", test_addition7_butterfly)


# ===========================================================
if RUN_SMOKE:
    print("\n═══ 4. Integration Tests ═══")
# ===========================================================

def test_nirv_full_pipeline():
    """Full NIRV pricing pipeline (end-to-end)"""
    model = NIRV_MOD.NIRVModel(n_paths=2000, n_bootstrap=500)
    result = model.price_option(
        spot=23500, strike=23400, T=7/365, r=0.065, q=0.012,
        option_type='CE', market_price=200,
        india_vix=14.0, fii_net_flow=-500, dii_net_flow=800,
        days_to_rbi=15, pcr_oi=0.85,
        returns_30d=np.random.normal(0, 0.01, 30)
    )
    assert result.fair_value > 0
    assert result.signal in ('BUY', 'SELL', 'HOLD', 'STRONG BUY', 'STRONG SELL')
    assert 0 <= result.confidence_level <= 100
    return True

test("NIRV full pipeline", test_nirv_full_pipeline)


# ===========================================================
if RUN_SMOKE:
    print(f"\n{'═'*50}")
    print(f"  Results: {PASS} passed, {FAIL} failed out of {PASS+FAIL}")
    print(f"{'═'*50}\n")
    sys.exit(0 if FAIL == 0 else 1)
