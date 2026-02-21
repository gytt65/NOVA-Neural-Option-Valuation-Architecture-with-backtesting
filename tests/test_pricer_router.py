import numpy as np

from pricer_router import TieredPricerRouter


class _DummyCOS:
    def price(self, S, K, T, r, q, V0, kappa, theta, sigma_v, rho, option_type="CE"):
        # Deterministic refinement above BS baseline
        return max(0.5, 0.04 * S * np.sqrt(max(T, 1e-8)))


class _DummyQuantEngine:
    def __init__(self):
        self.heston_cos = _DummyCOS()


class _DummyMC:
    def __init__(self):
        self.n_paths = 4000
        self.use_sobol = False
        self._last_S_T = None

    def price(self, spot, strike, T, r, q, sigma, regime_params, option_type, india_features):
        z = np.linspace(-2, 2, 1024)
        self._last_S_T = spot * np.exp((r - q - 0.5 * sigma * sigma) * T + sigma * np.sqrt(T) * z)
        pay = np.maximum(self._last_S_T - strike, 0.0) if option_type == "CE" else np.maximum(strike - self._last_S_T, 0.0)
        px = float(np.exp(-r * T) * np.mean(pay))
        return px, 0.15, self._last_S_T


def test_pricer_router_tier_selection_does_not_break():
    router = TieredPricerRouter(default_cpu_budget_ms=50.0)
    qe = _DummyQuantEngine()
    mc = _DummyMC()
    base_kwargs = dict(
        spot=23500.0,
        strike=23500.0,
        T=7.0 / 365.0,
        r=0.065,
        q=0.012,
        option_type="CE",
        surface_iv=0.18,
        regime_params={"kappa": 2.0, "theta_v": 0.03, "sigma_v": 0.3, "rho_sv": -0.5},
        india_features={},
    )

    # Tier 0 only
    out0 = router.route_price(**base_kwargs, quant_engine=None, mc_pricer=None)
    assert out0["tier_used"] == "tier0_surface_bs"
    assert np.isfinite(out0["price"]) and out0["price"] > 0

    # Tier 1 refinement
    out1 = router.route_price(**base_kwargs, quant_engine=qe, mc_pricer=None)
    assert out1["tier_used"] in ("tier1_heston_cos", "tier0_surface_bs")
    assert np.isfinite(out1["price"]) and out1["price"] > 0

    # Tier 2 selective escalation
    out2 = router.route_price(
        **base_kwargs,
        quant_engine=qe,
        mc_pricer=mc,
        full_chain_mode=False,
        liquidity_score=1.0,
        anomaly_score=0.1,
        mispricing_hint=0.12,
    )
    assert out2["tier_used"] in ("tier2_qmc_mc", "tier1_heston_cos", "tier0_surface_bs")
    assert np.isfinite(out2["price"]) and out2["price"] > 0

    # Full-chain mode should usually stay at Tier 0/1 unless candidate is strong
    out_chain = router.route_price(
        **base_kwargs,
        quant_engine=qe,
        mc_pricer=mc,
        full_chain_mode=True,
        liquidity_score=0.9,
        anomaly_score=0.1,
        mispricing_hint=0.01,
    )
    assert out_chain["tier_used"] != "tier2_qmc_mc"
    assert np.isfinite(out_chain["price"]) and out_chain["price"] > 0

