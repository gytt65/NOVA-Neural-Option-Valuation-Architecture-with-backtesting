import numpy as np

from pricer_router import TieredPricerRouter


def test_pricer_router_rbergomi_model_switch():
    router = TieredPricerRouter(default_cpu_budget_ms=20.0)

    out_rb = router.route_price(
        spot=23500.0,
        strike=23500.0,
        T=10.0 / 365.0,
        r=0.065,
        q=0.012,
        option_type="CE",
        surface_iv=0.18,
        config={"model": "rbergomi", "hurst": 0.12, "eta": 1.1, "rho": -0.7},
    )
    assert out_rb["tier_used"] == "tier_rbergomi"
    assert np.isfinite(out_rb["price"]) and out_rb["price"] > 0
    assert np.isfinite(out_rb.get("rbergomi_sigma_eff", np.nan))

    out_default = router.route_price(
        spot=23500.0,
        strike=23500.0,
        T=10.0 / 365.0,
        r=0.065,
        q=0.012,
        option_type="CE",
        surface_iv=0.18,
    )
    assert out_default["tier_used"] != "tier_rbergomi"

