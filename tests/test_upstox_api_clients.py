import pandas as pd
import pytest

from upstox_api_clients import UpstoxAPIClients, UpstoxAPIError


class _DummyResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload


class _DummySession:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append(
            {"url": url, "params": params or {}, "headers": headers or {}, "timeout": timeout}
        )
        if not self._responses:
            raise RuntimeError("No dummy responses left")
        return self._responses.pop(0)


def test_get_option_contracts_returns_list():
    payload = {"status": "success", "data": [{"instrument_key": "NSE_FO|ABC", "expiry": "2026-03-31"}]}
    session = _DummySession([_DummyResponse(200, payload)])
    client = UpstoxAPIClients(access_token="test-token", session=session)

    out = client.get_option_contracts("NSE_INDEX|Nifty 50")
    assert isinstance(out, list)
    assert out[0]["instrument_key"] == "NSE_FO|ABC"
    assert session.calls[0]["url"].endswith("/v2/option/contract")


def test_expired_historical_candles_parse_to_dataframe():
    payload = {
        "status": "success",
        "data": {
            "candles": [
                ["2024-01-01T09:15:00+05:30", 100, 110, 95, 102, 1200, 3000],
                ["2024-01-02T09:15:00+05:30", 102, 111, 100, 109, 1400, 3200],
            ]
        },
    }
    session = _DummySession([_DummyResponse(200, payload)])
    client = UpstoxAPIClients(access_token="test-token", session=session)

    df = client.get_expired_historical_candle_data(
        expired_instrument_key="NSE_FO|ABC24012520000CE",
        interval="day",
        to_date="2024-01-10",
        from_date="2024-01-01",
    )
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "open_interest",
    ]
    assert df["timestamp"].dt.tz is not None
    assert df["close"].iloc[-1] == 109


def test_quote_limits_and_interval_validation():
    session = _DummySession([])
    client = UpstoxAPIClients(access_token="test-token", session=session)

    with pytest.raises(ValueError):
        client.get_full_market_quote(["K"] * 501)
    with pytest.raises(ValueError):
        client.get_option_greek(["K"] * 51)
    with pytest.raises(ValueError):
        client.get_market_quote_ohlc(["NSE_INDEX|Nifty 50"], interval="5m")


def test_expired_option_contracts_converts_date_format_and_raises_on_error():
    ok_payload = {"status": "success", "data": [{"instrument_key": "NSE_FO|X"}]}
    err_payload = {"status": "error", "message": "failed"}
    session = _DummySession([_DummyResponse(200, ok_payload), _DummyResponse(200, err_payload)])
    client = UpstoxAPIClients(access_token="test-token", session=session)

    out = client.get_expired_option_contracts(
        instrument_key="NSE_INDEX|Nifty 50",
        expiry_date="2024-01-25",
        page=1,
    )
    assert len(out) == 1
    assert session.calls[0]["params"]["expiry_date"] == "2024-01-25"

    with pytest.raises(UpstoxAPIError):
        client.get_option_contracts("NSE_INDEX|Nifty 50")


def test_expired_option_contracts_retries_alternate_date_on_udapi1088():
    invalid_payload = {
        "status": "error",
        "errors": [{"errorCode": "UDAPI1088", "message": "Invalid date"}],
    }
    ok_payload = {"status": "success", "data": [{"instrument_key": "NSE_FO|X"}]}
    session = _DummySession([
        _DummyResponse(400, invalid_payload),
        _DummyResponse(200, ok_payload),
    ])
    client = UpstoxAPIClients(access_token="test-token", session=session)

    out = client.get_expired_option_contracts(
        instrument_key="NSE_INDEX|Nifty 50",
        expiry_date="25-01-2024",
        page=1,
    )
    assert len(out) == 1
    assert session.calls[0]["params"]["expiry_date"] == "25-01-2024"
    assert session.calls[1]["params"]["expiry_date"] == "2024-01-25"
