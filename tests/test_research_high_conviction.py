import types
import numpy as np

from omega_features import set_features
from omega_model import OMEGAModel


def test_research_conviction_score_buckets():
    strong = OMEGAModel._research_conviction_score_10(
        signal="STRONG BUY",
        mispricing_pct=9.0,
        physical_pop=78.0,
        confidence_level=88.0,
        efficiency_score=84.0,
        ml_confidence=0.85,
        conformal_actionable=True,
    )
    assert strong == 10

    moderate = OMEGAModel._research_conviction_score_10(
        signal="BUY",
        mispricing_pct=7.5,
        physical_pop=72.0,
        confidence_level=78.0,
        efficiency_score=72.0,
        ml_confidence=0.60,
        conformal_actionable=True,
    )
    assert moderate == 9

    weak = OMEGAModel._research_conviction_score_10(
        signal="BUY",
        mispricing_pct=1.8,
        physical_pop=54.0,
        confidence_level=60.0,
        efficiency_score=58.0,
        ml_confidence=0.20,
        conformal_actionable=True,
    )
    assert weak == 0

    blocked = OMEGAModel._research_conviction_score_10(
        signal="BUY",
        mispricing_pct=7.5,
        physical_pop=72.0,
        confidence_level=78.0,
        efficiency_score=72.0,
        ml_confidence=0.60,
        conformal_actionable=False,
    )
    assert blocked == 0


def test_scan_chain_filters_to_high_conviction_when_flag_on():
    try:
        set_features(USE_RESEARCH_HIGH_CONVICTION=True)

        model = OMEGAModel.__new__(OMEGAModel)

        def fake_price_option(
            spot,
            strike,
            T,
            r,
            q,
            option_type,
            market_price,
            india_vix,
            fii_net_flow,
            dii_net_flow,
            days_to_rbi,
            pcr_oi,
            returns_30d,
            **kwargs,
        ):
            mapping = {
                23000: (10, 82.0, 67.0, 85.0, "STRONG BUY"),
                23100: (9, 76.0, 64.0, 79.0, "BUY"),
                23200: (0, 88.0, 70.0, 90.0, "HOLD"),
            }
            conv, eff, pop, conf, sig = mapping[int(strike)]
            return types.SimpleNamespace(
                conviction_score_10=conv,
                efficiency_score=eff,
                physical_profit_prob=pop,
                confidence_level=conf,
                signal=sig,
            )

        model.price_option = fake_price_option

        strikes = [23000, 23100, 23200]
        ce = {s: 100.0 for s in strikes}
        pe = {}
        out = model.scan_chain(
            spot=23100,
            strikes=strikes,
            T=7 / 365.0,
            r=0.065,
            q=0.012,
            market_prices_ce=ce,
            market_prices_pe=pe,
            india_vix=14.0,
            fii_net_flow=0.0,
            dii_net_flow=0.0,
            days_to_rbi=30,
            pcr_oi=1.0,
            returns_30d=np.zeros(30),
        )

        assert len(out) == 2
        assert [int(x[1]) for x in out] == [23000, 23100]
        assert [x[2].conviction_score_10 for x in out] == [10, 9]
    finally:
        set_features()


def test_scan_chain_baseline_sorting_unchanged_when_flag_off():
    try:
        set_features()

        model = OMEGAModel.__new__(OMEGAModel)

        def fake_price_option(
            spot,
            strike,
            T,
            r,
            q,
            option_type,
            market_price,
            india_vix,
            fii_net_flow,
            dii_net_flow,
            days_to_rbi,
            pcr_oi,
            returns_30d,
            **kwargs,
        ):
            # Conviction intentionally opposite to efficiency to validate baseline sort key.
            mapping = {
                23000: (9, 60.0, 55.0, 70.0, "BUY"),
                23100: (10, 75.0, 58.0, 72.0, "BUY"),
                23200: (0, 85.0, 61.0, 74.0, "HOLD"),
            }
            conv, eff, pop, conf, sig = mapping[int(strike)]
            return types.SimpleNamespace(
                conviction_score_10=conv,
                efficiency_score=eff,
                physical_profit_prob=pop,
                confidence_level=conf,
                signal=sig,
            )

        model.price_option = fake_price_option

        strikes = [23000, 23100, 23200]
        ce = {s: 100.0 for s in strikes}
        pe = {}
        out = model.scan_chain(
            spot=23100,
            strikes=strikes,
            T=7 / 365.0,
            r=0.065,
            q=0.012,
            market_prices_ce=ce,
            market_prices_pe=pe,
            india_vix=14.0,
            fii_net_flow=0.0,
            dii_net_flow=0.0,
            days_to_rbi=30,
            pcr_oi=1.0,
            returns_30d=np.zeros(30),
        )

        assert [int(x[1]) for x in out] == [23200, 23100, 23000]
    finally:
        set_features()


def test_scan_chain_passes_per_leg_liquidity_fields_when_provided():
    try:
        set_features()

        model = OMEGAModel.__new__(OMEGAModel)
        captured = []

        def fake_price_option(
            spot,
            strike,
            T,
            r,
            q,
            option_type,
            market_price,
            india_vix,
            fii_net_flow,
            dii_net_flow,
            days_to_rbi,
            pcr_oi,
            returns_30d,
            **kwargs,
        ):
            captured.append(
                (
                    option_type,
                    int(strike),
                    kwargs.get("bid"),
                    kwargs.get("ask"),
                    kwargs.get("volume_oi_ratio"),
                )
            )
            return types.SimpleNamespace(
                conviction_score_10=0,
                efficiency_score=50.0,
                physical_profit_prob=50.0,
                confidence_level=50.0,
                signal="HOLD",
            )

        model.price_option = fake_price_option

        strikes = [23000]
        out = model.scan_chain(
            spot=23100,
            strikes=strikes,
            T=7 / 365.0,
            r=0.065,
            q=0.012,
            market_prices_ce={23000: 100.0},
            market_prices_pe={23000: 90.0},
            india_vix=14.0,
            fii_net_flow=0.0,
            dii_net_flow=0.0,
            days_to_rbi=30,
            pcr_oi=1.0,
            returns_30d=np.zeros(30),
            bid_quotes_ce={23000: 99.5},
            ask_quotes_ce={23000: 100.5},
            volume_oi_ratio_ce={23000: 0.18},
            bid_quotes_pe={23000: 89.0},
            ask_quotes_pe={23000: 91.0},
            volume_oi_ratio_pe={23000: 0.14},
        )

        assert len(out) == 2
        assert captured == [
            ("CE", 23000, 99.5, 100.5, 0.18),
            ("PE", 23000, 89.0, 91.0, 0.14),
        ]
    finally:
        set_features()
