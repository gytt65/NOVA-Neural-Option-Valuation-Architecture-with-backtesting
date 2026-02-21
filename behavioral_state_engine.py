"""
behavioral_state_engine.py
==========================

Additive behavioral state builder for market microstructure context.
Designed to be optional and backward-compatible with existing pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class BehavioralInputs:
    news_sentiment: float = 0.0
    social_sentiment: float = 0.0
    otm_call_skew: float = 0.0
    put_call_ratio: float = 1.0
    bid_ask_spread: float = 0.01
    volume_oi_ratio: float = 0.1
    total_gex: float = 0.0
    gamma_flip: float = np.nan
    charm: float = 0.0
    vanna: float = 0.0
    dealer_flow_imbalance: float = 0.0


class BehavioralStateEngine:
    """
    Computes a compact behavioral state object from optional market inputs.
    All outputs are bounded and deterministic.
    """

    @staticmethod
    def _clip(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
        return float(np.clip(float(x), lo, hi))

    def compute_sentiment(self, inp: BehavioralInputs) -> float:
        # Blend news + social with mild saturation.
        raw = 0.65 * float(inp.news_sentiment) + 0.35 * float(inp.social_sentiment)
        return self._clip(np.tanh(raw))

    def compute_lottery_demand(self, inp: BehavioralInputs) -> float:
        # OTM call demand tends to increase with call-skew and low PCR.
        skew_term = self._clip(inp.otm_call_skew, -2.0, 2.0) / 2.0
        pcr_term = self._clip((1.0 - float(inp.put_call_ratio)) / 0.5, -2.0, 2.0) / 2.0
        return self._clip(0.6 * skew_term + 0.4 * pcr_term)

    def compute_limits_to_arb(self, inp: BehavioralInputs) -> float:
        # Wider spreads and weaker depth imply stronger limits-to-arbitrage.
        spread_z = self._clip((float(inp.bid_ask_spread) - 0.01) / 0.05, -2.0, 2.0)
        depth_z = self._clip((0.2 - float(inp.volume_oi_ratio)) / 0.2, -2.0, 2.0)
        return self._clip(0.5 * spread_z + 0.5 * depth_z, 0.0, 1.0)

    def compute_dealer_flow(self, inp: BehavioralInputs) -> Dict[str, float]:
        total_gex = float(inp.total_gex)
        gex_sign = float(np.sign(total_gex))
        return {
            "total_gex": total_gex,
            "gex_sign": gex_sign,
            "gamma_flip": float(inp.gamma_flip) if np.isfinite(inp.gamma_flip) else np.nan,
            "charm": float(inp.charm),
            "vanna": float(inp.vanna),
            "flow_imbalance": self._clip(inp.dealer_flow_imbalance),
        }

    def build_market_state(
        self,
        regime: Optional[Dict] = None,
        vrp: Optional[Dict] = None,
        srp: Optional[Dict] = None,
        liquidity: Optional[Dict] = None,
        behavioral_inputs: Optional[Dict] = None,
    ) -> Dict:
        inp = BehavioralInputs(**(behavioral_inputs or {}))
        sentiment = self.compute_sentiment(inp)
        lottery = self.compute_lottery_demand(inp)
        limits_to_arb = self.compute_limits_to_arb(inp)
        dealer_flow = self.compute_dealer_flow(inp)

        return {
            "regime": dict(regime or {}),
            "vrp": dict(vrp or {}),
            "srp": dict(srp or {}),
            "behavioral": {
                "sentiment": float(sentiment),
                "lottery_demand": float(lottery),
                "limits_to_arb": float(limits_to_arb),
                "dealer_flow": dealer_flow,
            },
            "liquidity": dict(liquidity or {}),
        }
