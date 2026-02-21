from pathlib import Path
import sys
import types

import numpy as np
import pandas as pd

import historical_learning as hl


class _MockClient:
    def get_expiries(self, instrument_key):
        return ["2024-01-25"]

    def get_full_market_quote(self, instrument_keys):
        return [{"instrument_key": instrument_keys[0], "last_price": 22100.0}]

    def get_expired_option_contracts(self, instrument_key, expiry_date, page=None):
        if page and page > 1:
            return []
        return [
            {
                "instrument_key": "NSE_FO|ABC24012522000CE",
                "trading_symbol": "ABC24012522000CE",
                "strike_price": 22000,
                "option_type": "CE",
                "expiry": "2024-01-25",
                "volume": 1000,
            },
            {
                "instrument_key": "NSE_FO|ABC24012522000PE",
                "trading_symbol": "ABC24012522000PE",
                "strike_price": 22000,
                "option_type": "PE",
                "expiry": "2024-01-25",
                "volume": 900,
            },
        ]

    def get_expired_historical_candle_data(self, *, expired_instrument_key, interval, to_date, from_date):
        ts = pd.date_range("2024-01-10", periods=5, freq="D", tz="UTC")
        return pd.DataFrame(
            {
                "timestamp": ts,
                "open": [100, 102, 101, 104, 103],
                "high": [103, 104, 105, 106, 108],
                "low": [99, 100, 100, 101, 102],
                "close": [102, 101, 104, 103, 107],
                "volume": [1200, 1100, 1500, 1300, 1600],
                "open_interest": [3000, 3100, 3150, 3200, 3300],
            }
        )


def test_engineer_features_from_candles_has_expected_columns():
    ts = pd.date_range("2024-01-01", periods=12, freq="D", tz="UTC")
    raw = pd.DataFrame(
        {
            "timestamp": ts,
            "open": np.linspace(100, 111, 12),
            "high": np.linspace(101, 113, 12),
            "low": np.linspace(99, 109, 12),
            "close": np.linspace(100, 112, 12),
            "volume": np.linspace(1000, 2200, 12),
            "open_interest": np.linspace(3000, 3500, 12),
            "instrument_key": ["NSE_FO|X"] * 12,
            "strike_price": [22000] * 12,
            "expiry_date": ["2024-01-25"] * 12,
            "option_type": ["CE"] * 12,
        }
    )
    out = hl.engineer_features_from_candles(raw)
    expected = {"ret_1", "log_ret_1", "rv_10", "hl_range_pct", "volume_ratio", "oi_change", "dow", "hour"}
    assert expected.issubset(set(out.columns))
    assert len(out) == 12


def test_compute_model_residual_labels_with_mocked_nirv():
    fake_mod = types.ModuleType("nirv_model")

    class _FakeNIRVModel:
        def __init__(self, *args, **kwargs):
            pass

        def price_option(self, **kwargs):
            return types.SimpleNamespace(fair_value=float(kwargs["market_price"]) * 0.95)

    fake_mod.NIRVModel = _FakeNIRVModel
    sys.modules["nirv_model"] = fake_mod

    try:
        ts = pd.date_range("2024-01-01", periods=35, freq="D", tz="UTC")
        df = pd.DataFrame(
            {
                "timestamp": ts,
                "instrument_key": ["NSE_FO|X"] * len(ts),
                "strike_price": [22000] * len(ts),
                "close": np.linspace(90, 110, len(ts)),
                "expiry_date": ["2024-02-29"] * len(ts),
                "option_type": ["CE"] * len(ts),
                "log_ret_1": np.random.normal(0, 0.01, len(ts)),
                "rv_20": [0.2] * len(ts),
            }
        )
        cfg = hl.HistoricalLearningConfig()
        out = hl._compute_model_residual_labels(df, cfg)
        assert out["model_fair_value"].notna().sum() > 0
        assert out["residual_label"].notna().sum() > 0
    finally:
        sys.modules.pop("nirv_model", None)


def test_pull_and_train_writes_artifacts(tmp_path, monkeypatch):
    mock_client = _MockClient()

    def _mock_labels(df, config, progress_cb=None):
        out = df.copy()
        out["model_fair_value"] = out["close"] * 0.96
        out["residual_label"] = (out["close"] - out["model_fair_value"]) / out["model_fair_value"]
        return out

    def _mock_train(df, config, progress_cb=None):
        return {
            "trained": True,
            "rows_train": 10,
            "rows_test": 2,
            "mae_test": 0.01,
            "rmse_test": 0.02,
            "model_path": config.model_path,
        }

    monkeypatch.setattr(hl, "_compute_model_residual_labels", _mock_labels)
    monkeypatch.setattr(hl, "train_or_update_ml_corrector", _mock_train)

    cfg = hl.HistoricalLearningConfig(
        underlying_instrument_key="NSE_INDEX|Nifty 50",
        from_date="2024-01-01",
        to_date="2024-01-31",
        interval="day",
        output_root=str(tmp_path / "omega_data"),
        model_path=str(tmp_path / "omega_data" / "pricing_model.joblib"),
        max_expiries=1,
        max_contracts_per_expiry=4,
    )
    report = hl.pull_and_train(access_token="dummy-token", config=cfg, client=mock_client)
    assert report["rows_raw_candles"] > 0
    assert report["rows_processed"] > 0

    artifacts = report["artifacts"]
    assert Path(artifacts["raw_contracts_path"]).exists()
    assert Path(artifacts["processed_features_path"]).exists()
    assert Path(artifacts["training_report_path"]).exists()

