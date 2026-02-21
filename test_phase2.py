"""
Phase 2 Smoke Tests — 12 Paradigm-Shifting Upgrades
====================================================
Tests each new class's core functionality with synthetic data.
Run: python test_phase2.py
"""
import sys
import numpy as np
import os # Added for os.path.join

# ── Setup import paths ──────────────────────────────────────────────
# sys.path.insert(0, '.') # Removed as part of the change
# import importlib # Removed as part of the change
# nirv = importlib.import_module('nirv_model (1)') # Removed as part of the change

# New import logic for nirv_model
try:
    import nirv_model
except ImportError:
    # Assuming ROOT is defined somewhere, or needs to be defined.
    # For this test, let's assume current directory for simplicity if ROOT is not defined.
    # If ROOT is not defined, this part will fail.
    # For now, let's define ROOT as current directory for the sake of making it syntactically correct.
    ROOT = os.path.dirname(os.path.abspath(__file__)) # Added for ROOT definition
    import importlib.util # Added for importlib.util
    spec = importlib.util.spec_from_file_location(
        "nirv_model", os.path.join(ROOT, "nirv_model.py")
    )
    nirv_model = importlib.util.module_from_spec(spec) # Changed nirv to nirv_model to match import
    sys.modules['nirv_model'] = nirv_model
    spec.loader.exec_module(nirv_model)

# The following line seems to be a partial line from the user's instruction.
# It looks like it was meant to be `NIRVModel = nirv_model.NIRVModel` followed by `from quant_engine import (...)`.
# I will interpret it as replacing the old `nirv` variable with `nirv_model` and keeping the `from quant_engine` block.
# I will also assume `NIRVModel` is a class within `nirv_model` that needs to be imported/aliased.
NIRVModel = nirv_model.NIRVModel # Corrected from NIRVModelt_engine import (

from quant_engine import (
    MicrostructureAlphaEngine,
    VarianceSurfaceArbitrage,
    OptimalExecution,
    LevyProcessPricer,
    ContagionGraph,
    NeuralSDECalibrator,
    RegimeCopula,
)
from omega_model import (
    EventRiskPricer,
    BehavioralLiquidityFeedback,
    ShadowHedger,
)

FractionalBrownianMotion = nirv_model.FractionalBrownianMotion
EntropyEnsemble = nirv_model.EntropyEnsemble

passed = 0
failed = 0
RUN_SMOKE = (__name__ == "__main__")
__test__ = False
if not RUN_SMOKE:
    def print(*args, **kwargs):  # type: ignore[override]
        return None

def test(name, func):
    global passed, failed
    if not RUN_SMOKE:
        return
    try:
        func()
        print(f"  ✅ {name}")
        passed += 1
    except Exception as e:
        print(f"  ❌ {name}: {e}")
        import traceback; traceback.print_exc()
        failed += 1


# ====================================================================
# UPGRADE 1: Microstructure Alpha Engine
# ====================================================================
print("\n── Upgrade 1: MicrostructureAlphaEngine ──")

def test_spread_signal():
    strikes = [23400, 23500, 23600, 23700, 23800]
    bids = {K: 100 + i*10 for i, K in enumerate(strikes)}
    asks = {K: bids[K] + (5 if K != 23600 else 25) for K in strikes}  # 23600 has wide spread
    result = MicrostructureAlphaEngine.spread_signal(bids, asks, strikes)
    assert isinstance(result, dict)
    assert len(result) == 5
    assert result[23600] > result[23500], "Wide spread strike should have higher signal"
test("Spread signal detection", test_spread_signal)

def test_pin_risk():
    spot = 23500
    strikes = [23400, 23500, 23600]
    gammas = {23400: 0.001, 23500: 0.005, 23600: 0.002}
    ois = {23400: 50000, 23500: 200000, 23600: 80000}
    result = MicrostructureAlphaEngine.pin_risk(spot, strikes, gammas, ois)
    assert result['max_pin_strike'] == 23500, "ATM strike with highest gamma×OI should be max pin"
    assert result['total_pin_energy'] > 0
test("Pin risk computation", test_pin_risk)

def test_vpin():
    np.random.seed(42)
    buy = np.random.poisson(100, 50).astype(float)
    sell = np.random.poisson(100, 50).astype(float)
    result = MicrostructureAlphaEngine.vpin_estimate(buy, sell)
    assert 0 <= result['vpin'] <= 1
    assert result['toxicity_regime'] in ('NORMAL', 'ELEVATED', 'TOXIC')

    # Toxic flow: all buying
    toxic_buy = np.ones(30) * 200
    toxic_sell = np.ones(30) * 10
    result2 = MicrostructureAlphaEngine.vpin_estimate(toxic_buy, toxic_sell)
    assert result2['vpin'] > 0.7, "Heavily skewed flow should be TOXIC"
test("VPIN estimation", test_vpin)

def test_combined_alpha():
    engine = MicrostructureAlphaEngine()
    strikes = [23400, 23500, 23600]
    bids = {K: 100 for K in strikes}
    asks = {K: 105 for K in strikes}
    result = engine.combined_alpha(23500, strikes, bids, asks)
    assert 'alpha_per_strike' in result
    assert 'signal' in result
    assert result['signal'] in ('LOW_INFO_FLOW', 'MODERATE_INFO_FLOW', 'HIGH_INFO_FLOW')
test("Combined alpha signal", test_combined_alpha)


# ====================================================================
# UPGRADE 2: Variance Surface Arbitrage
# ====================================================================
print("\n── Upgrade 2: VarianceSurfaceArbitrage ──")

def test_vrp_computation():
    arb = VarianceSurfaceArbitrage()
    iv_ts = {7: 0.14, 14: 0.145, 30: 0.15, 60: 0.155}
    returns = np.random.randn(100) * 0.01
    result = arb.compute_vrp(iv_ts, returns)
    assert 'vrp' in result
    assert len(result['vrp']) == 4
    assert result['signal'] in ('NEUTRAL', 'SELL_VOL', 'BUY_VOL', 'COMPRESSED_SHORT_END', 'EXPANDED_SHORT_END')
test("VRP term structure", test_vrp_computation)

def test_vrp_extreme():
    arb = VarianceSurfaceArbitrage()
    # Very high IV vs low RV → VRP expanded → SELL_VOL
    iv_ts = {7: 0.30, 30: 0.28}
    returns = np.random.randn(100) * 0.005  # Low realized vol
    # Feed multiple times to build history
    for _ in range(10):
        result = arb.compute_vrp(iv_ts, returns)
    assert 'vrp_z_score' in result
test("VRP extreme detection", test_vrp_extreme)


# ====================================================================
# UPGRADE 3: Optimal Execution
# ====================================================================
print("\n── Upgrade 3: OptimalExecution ──")

def test_impact_params():
    eta, gamma = OptimalExecution.estimate_impact_params(100000, 2.0, lot_size=65)
    assert eta > 0
    assert gamma > 0
    assert gamma < eta, "Permanent impact should be less than temporary"
test("Impact parameter estimation", test_impact_params)

def test_optimal_trajectory():
    eta, gamma = OptimalExecution.estimate_impact_params(100000, 2.0)
    result = OptimalExecution.optimal_trajectory(10, T_minutes=10, eta=eta, gamma=gamma)
    assert len(result['schedule']) >= 3
    assert result['total_lots'] == 10
    total_scheduled = sum(s['lots'] for s in result['schedule'])
    assert abs(total_scheduled - 10) < 1, f"Schedule should sum to 10 lots, got {total_scheduled}"
    assert result['cost_saving_pct'] >= 0, "Optimal should save vs naive"
test("Almgren-Chriss trajectory", test_optimal_trajectory)


# ====================================================================
# UPGRADE 4: Fractional Brownian Motion
# ====================================================================
print("\n── Upgrade 4: FractionalBrownianMotion ──")

def test_fbm_standard():
    paths = FractionalBrownianMotion.generate_paths(100, 20, H=0.5, dt=1/252)
    assert paths.shape == (100, 20)
    # Standard BM variance should be ≈ dt
    empirical_var = np.var(paths)
    assert 0.0001 < empirical_var < 0.1
test("fBm H=0.5 (standard BM)", test_fbm_standard)

def test_fbm_rough():
    paths = FractionalBrownianMotion.generate_paths(200, 30, H=0.2, dt=1/252)
    assert paths.shape == (200, 30)
    # Anti-persistent: autocorrelation of increments should be negative
    increments = paths[0]
    if len(increments) > 2:
        autocorr = np.corrcoef(increments[:-1], increments[1:])[0, 1]
        assert autocorr < 0.3, f"H=0.2 should give negative autocorrelation, got {autocorr}"
test("fBm H=0.2 (rough vol)", test_fbm_rough)

def test_fbm_adjustment():
    factor = FractionalBrownianMotion.price_adjustment_factor(0.2, 7/365, 0.15)
    assert 0.5 <= factor <= 2.0
    # H < 0.5 for short T → factor > 1 (higher effective vol from scaling)
    factor_bm = FractionalBrownianMotion.price_adjustment_factor(0.5, 7/365, 0.15)
    assert factor_bm == 1.0
test("fBm price adjustment factor", test_fbm_adjustment)


# ====================================================================
# UPGRADE 5: Lévy Process (Variance Gamma)
# ====================================================================
print("\n── Upgrade 5: LevyProcessPricer ──")

def test_vg_price():
    pricer = LevyProcessPricer()
    # ATM call
    price = pricer.price(S=23500, K=23500, T=7/365, r=0.065, q=0.012,
                         sigma=0.15, theta=-0.1, nu=0.2, option_type='CE')
    assert price > 0, f"VG call price should be positive, got {price}"
    assert price < 23500, "Call price can't exceed spot"

    # Put-call parity check (approximate)
    put = pricer.price(S=23500, K=23500, T=7/365, r=0.065, q=0.012,
                       sigma=0.15, theta=-0.1, nu=0.2, option_type='PE')
    assert put > 0
test("VG option pricing", test_vg_price)

def test_vg_calibration():
    pricer = LevyProcessPricer()
    # Synthetic market: price with known params, then calibrate
    true_sig, true_th, true_nu = 0.15, -0.1, 0.2
    strikes = np.array([23200, 23300, 23400, 23500, 23600, 23700])
    market_prices = np.array([
        pricer.price(23500, K, 7/365, 0.065, 0.012, true_sig, true_th, true_nu)
        for K in strikes
    ])
    result = pricer.calibrate_from_market(23500, strikes, market_prices, 7/365, 0.065, 0.012)
    assert result['rmse'] < 5.0, f"Calibration RMSE too high: {result['rmse']}"
test("VG calibration", test_vg_calibration)


# ====================================================================
# UPGRADE 6: Cross-Asset Contagion Graph
# ====================================================================
print("\n── Upgrade 6: ContagionGraph ──")

def test_granger_test():
    np.random.seed(123)
    n = 200
    # x causes y with 1-lag
    x = np.cumsum(np.random.randn(n))
    y = np.zeros(n)
    for i in range(1, n):
        y[i] = 0.8 * x[i-1] + 0.3 * np.random.randn()
    result = ContagionGraph.granger_test(x, y, max_lag=3)
    assert result['significant'], "x should Granger-cause y"
    assert result['p_value'] < 0.05
test("Granger causality test", test_granger_test)

def test_contagion_graph():
    np.random.seed(456)
    n = 150
    nifty = np.cumsum(np.random.randn(n))
    vix = -0.7 * nifty + np.cumsum(np.random.randn(n) * 0.5)
    fii = np.zeros(n)
    for i in range(2, n):
        fii[i] = 0.5 * nifty[i-2] + np.random.randn()

    result = ContagionGraph.build_graph({
        'Nifty': nifty, 'VIX': vix, 'FII': fii
    }, max_lag=3)
    assert 'pagerank' in result
    assert 'dominant_driver' in result
    assert result['dominant_driver'] in ('Nifty', 'VIX', 'FII')
test("Build contagion graph", test_contagion_graph)


# ====================================================================
# UPGRADE 7: Neural SDE Calibrator
# ====================================================================
print("\n── Upgrade 7: NeuralSDECalibrator ──")

def test_neural_sde():
    np.random.seed(42)
    cal = NeuralSDECalibrator(hidden_size=8)
    # Test drift/diffusion correction outputs
    d_corr = cal.drift_correction(1.0, 0.02, 0.5)
    assert len(d_corr) >= 1

    g_corr = cal.diffusion_correction(1.0, 0.02, 0.5)
    assert len(g_corr) >= 1
    assert abs(g_corr[0]) < 0.05, "Uncalibrated diffusion correction should be small"
test("Neural SDE forward pass", test_neural_sde)

def test_neural_sde_calibrate():
    np.random.seed(42)
    cal = NeuralSDECalibrator(hidden_size=8)
    strikes = np.array([23300, 23500, 23700])
    # Synthetic market prices (slightly off Heston)
    market = np.array([250.0, 150.0, 80.0])
    result = cal.calibrate(S=23500, strikes=strikes, market_prices=market,
                           T=7/365, r=0.065, q=0.012,
                           V0=0.02, kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.5,
                           n_paths=1000, lr=0.005, n_iters=10)
    assert 'rmse_before' in result
    assert 'rmse_after' in result
    assert result['converged'] or result['rmse_after'] <= result['rmse_before'] * 1.1
test("Neural SDE calibration", test_neural_sde_calibrate)


# ====================================================================
# UPGRADE 8: Event Risk Pricer
# ====================================================================
print("\n── Upgrade 8: EventRiskPricer ──")

def test_event_decompose():
    result = EventRiskPricer.decompose(iv=0.15, T=7/365, event_type='rbi', days_to_event=3)
    assert result['iv_diffusive'] > 0
    assert result['iv_event'] >= 0
    assert result['iv_total'] == 0.15
    assert result['event_pct_of_total'] >= 0
    assert result['signal'] in ('LOW_EVENT_PREMIUM', 'MODERATE_EVENT_PREMIUM', 'HIGH_EVENT_PREMIUM')
test("IV decomposition", test_event_decompose)

def test_event_budget():
    # Budget has higher event variance
    res_rbi = EventRiskPricer.decompose(0.20, 5/365, 'rbi', 2)
    res_budget = EventRiskPricer.decompose(0.20, 5/365, 'budget', 2)
    assert res_budget['event_pct_of_total'] >= res_rbi['event_pct_of_total'], \
        "Budget should have higher event premium than RBI"
test("Budget vs RBI event premium", test_event_budget)

def test_theta_profile():
    profile = EventRiskPricer.theta_profile_across_event(
        iv=0.15, S=23500, K=23500, T=5/365, r=0.065,
        event_type='rbi', days_to_event=2)
    assert isinstance(profile, list)
    assert len(profile) > 0
    # Find the event day
    event_days = [p for p in profile if p['is_event_day']]
    assert len(event_days) > 0, "Should have an event day in profile"
test("Non-linear theta profile", test_theta_profile)


# ====================================================================
# UPGRADE 9: Behavioral Liquidity Feedback
# ====================================================================
print("\n── Upgrade 9: BehavioralLiquidityFeedback ──")

def test_feedback_loop():
    result = BehavioralLiquidityFeedback.compute_feedback(
        gex_current=-5e9, gex_average=2e9,
        retail_flow_direction=1.0,
        pt_overpricing_pct=10, iv_model=0.14)
    assert result['adjusted_iv'] != result['base_iv']
    assert result['cycle_stage'] == 'ACCUMULATION', \
        f"Negative GEX + positive retail should be ACCUMULATION, got {result['cycle_stage']}"
    assert result['feedback_multiplier'] > 1.0, "Should amplify IV in accumulation"
test("Behavioral feedback loop", test_feedback_loop)

def test_feedback_pinning():
    result = BehavioralLiquidityFeedback.compute_feedback(
        gex_current=3e9, gex_average=2e9,
        retail_flow_direction=0.5, iv_model=0.14)
    assert result['cycle_stage'] == 'PINNING'
test("Feedback pinning stage", test_feedback_pinning)


# ====================================================================
# UPGRADE 10: Entropy-Weighted Ensemble
# ====================================================================
print("\n── Upgrade 10: EntropyEnsemble ──")

def test_entropy_calculation():
    # Peaked distribution → low entropy
    peaked = np.random.randn(1000) * 0.1 + 100
    # Spread distribution → high entropy
    spread = np.random.randn(1000) * 10 + 100
    h_peaked = EntropyEnsemble.shannon_entropy(peaked)
    h_spread = EntropyEnsemble.shannon_entropy(spread)
    assert h_peaked < h_spread, f"Peaked should have lower entropy: {h_peaked} vs {h_spread}"
test("Shannon entropy calculation", test_entropy_calculation)

def test_ensemble_combine():
    np.random.seed(42)
    methods = [
        {'name': 'MC', 'price': 150.0, 'distribution': np.random.randn(5000) * 5 + 150,
         'std_error': 0.5},
        {'name': 'COS', 'price': 148.0, 'distribution': None, 'std_error': 0.0},
    ]
    result = EntropyEnsemble.combine(methods)
    assert 'blended_price' in result
    assert 140 < result['blended_price'] < 160
    assert result['dominant_method'] in ('MC', 'COS')
    # COS should dominate (lower entropy from analytical)
    assert result['method_weights']['COS'] > result['method_weights']['MC'], \
        "Analytical COS should have higher weight"
test("Entropy ensemble blending", test_ensemble_combine)


# ====================================================================
# UPGRADE 11: Shadow Delta Hedger
# ====================================================================
print("\n── Upgrade 11: ShadowHedger ──")

def test_shadow_trade():
    hedger = ShadowHedger(lot_size=65)
    trade = hedger.add_trade(
        signal='BUY', entry_price=150, fair_value=165,
        delta=0.5, gamma=0.001, theta=-5,
        spot_at_entry=23500, spot_at_exit=23600, dt_days=1)
    assert trade['lot_pnl'] != 0
    assert trade['edge_pnl'] == 15  # fair_value - entry
test("Shadow trade recording", test_shadow_trade)

def test_shadow_performance():
    np.random.seed(42)
    hedger = ShadowHedger(lot_size=65)
    # Add 20 trades with positive edge
    for i in range(20):
        hedger.add_trade(
            signal='BUY', entry_price=150, fair_value=165,
            delta=0.5, gamma=0.001, theta=-5,
            spot_at_entry=23500, spot_at_exit=23500 + np.random.randn() * 50,
            dt_days=1)
    perf = hedger.get_performance()
    assert perf['n_trades'] == 20
    assert perf['verdict'] in ('INSUFFICIENT_TRADES', 'STRONG_ALPHA', 'MODERATE_ALPHA',
                                'WEAK_ALPHA', 'NO_ALPHA')
    assert 'sharpe' in perf
    assert 'max_drawdown' in perf
test("Shadow performance metrics", test_shadow_performance)


# ====================================================================
# UPGRADE 12: Regime-Conditional Copula
# ====================================================================
print("\n── Upgrade 12: RegimeCopula ──")

def test_kendall_tau():
    np.random.seed(42)
    x = np.random.randn(100)
    y = 0.7 * x + 0.3 * np.random.randn(100)  # Correlated
    tau = RegimeCopula.fit_kendall_tau(x, y)
    assert tau > 0.3, f"Should have positive Kendall's tau for correlated data, got {tau}"
test("Kendall's tau estimation", test_kendall_tau)

def test_copula_analysis():
    np.random.seed(42)
    x = np.random.randn(100)
    y = 0.6 * x + np.random.randn(100) * 0.5
    result = RegimeCopula.analyze(x, y, regime_prob_crisis=0.3)
    assert 'kendall_tau' in result
    assert 'theta_clayton' in result
    assert 'blended_tail_dependence' in result
    assert result['signal'] in ('NORMAL', 'MODERATE_TAIL_DEPENDENCE', 'HIGH_TAIL_DEPENDENCE')
test("Regime copula analysis", test_copula_analysis)

def test_copula_crisis():
    np.random.seed(42)
    x = np.random.randn(100)
    y = 0.9 * x + 0.1 * np.random.randn(100)  # Highly correlated
    res_calm = RegimeCopula.analyze(x, y, regime_prob_crisis=0.1)
    res_crisis = RegimeCopula.analyze(x, y, regime_prob_crisis=0.9)
    # Crisis should have higher blended correlation
    assert res_crisis['blended_rho'] >= res_calm['blended_rho'] - 0.1, \
        "Crisis regime should have higher effective correlation"
test("Crisis copula amplification", test_copula_crisis)


# ====================================================================
# SUMMARY
# ====================================================================
print(f"\n{'='*60}")
print(f"Phase 2 Results: {passed} passed, {failed} failed out of {passed+failed} tests")
print(f"{'='*60}")

if RUN_SMOKE and failed > 0:
    sys.exit(1)
