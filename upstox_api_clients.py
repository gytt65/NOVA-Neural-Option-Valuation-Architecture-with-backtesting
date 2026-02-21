#!/usr/bin/env python3
"""
Upstox API clients (additive module for historical learning and data ingestion).

Design goals:
- Keep integration minimal and backward compatible.
- Use documented endpoints and Bearer-token auth.
- Validate parameters and return clean Python structures.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
from urllib.parse import quote

import pandas as pd
import requests


DEFAULT_BASE_URL = "https://api.upstox.com"


class UpstoxAPIError(RuntimeError):
    """Raised when an Upstox API request fails."""


def _best_effort_load_env(config_env_path: Optional[Union[str, Path]] = None) -> None:
    """
    Load environment values from `config.env`.

    Priority is still controlled by callers (`os.environ` first).
    """
    env_path = Path(config_env_path) if config_env_path else Path(__file__).resolve().parent / "config.env"
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(dotenv_path=env_path, override=False)
        return
    except Exception:
        pass

    if not env_path.exists():
        return
    try:
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip()
            if key and key not in os.environ:
                os.environ[key] = val
    except Exception:
        # Fail-open: env loading should never break the app.
        return


def _validate_non_empty(value: str, name: str) -> str:
    out = str(value or "").strip()
    if not out:
        raise ValueError(f"{name} is required.")
    return out


def _validate_date_yyyy_mm_dd(value: str, name: str) -> str:
    value = _validate_non_empty(value, name)
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(f"{name} must be in YYYY-MM-DD format.") from exc
    return value


def _expiry_date_candidates(value: str, name: str) -> List[str]:
    """
    Build robust date-format candidates for Upstox expired-contract endpoint.

    In practice, environments may differ between YYYY-MM-DD and DD-MM-YYYY.
    We preserve caller intent first, then try the alternate format if needed.
    """
    value = _validate_non_empty(value, name)
    # If caller already passed one supported format, keep it first.
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        dt = datetime.strptime(value, "%Y-%m-%d")
        return [value, dt.strftime("%d-%m-%Y")]
    if re.fullmatch(r"\d{2}-\d{2}-\d{4}", value):
        dt = datetime.strptime(value, "%d-%m-%Y")
        return [value, dt.strftime("%Y-%m-%d")]
    raise ValueError(f"{name} must be in YYYY-MM-DD or DD-MM-YYYY format.")


def _normalize_instrument_keys(
    instrument_keys: Union[str, Sequence[str]],
    *,
    max_count: Optional[int] = None,
    name: str = "instrument_key",
) -> List[str]:
    if isinstance(instrument_keys, str):
        keys = [k.strip() for k in instrument_keys.split(",") if k.strip()]
    else:
        keys = [str(k).strip() for k in instrument_keys if str(k).strip()]

    if not keys:
        raise ValueError(f"At least one {name} is required.")
    if max_count is not None and len(keys) > max_count:
        raise ValueError(f"{name} supports at most {max_count} values per request.")
    return keys


def _json_success_or_raise(payload: Dict[str, Any], *, endpoint: str) -> Dict[str, Any]:
    status = payload.get("status")
    if status is not None and str(status).lower() != "success":
        err = payload.get("errors") or payload.get("message") or payload
        raise UpstoxAPIError(f"{endpoint} returned non-success status: {err}")
    return payload


def _extract_payload_items(payload: Dict[str, Any]) -> Union[List[Any], Dict[str, Any]]:
    data = payload.get("data", {})
    if isinstance(data, dict):
        if "data" in data and isinstance(data["data"], list):
            return data["data"]
        if "candles" in data and isinstance(data["candles"], list):
            return data["candles"]
        return data
    if isinstance(data, list):
        return data
    return []


def _candles_to_dataframe(candles: Sequence[Sequence[Any]]) -> pd.DataFrame:
    if not candles:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "open_interest",
            ]
        )

    raw_df = pd.DataFrame(list(candles))
    if raw_df.empty:
        return pd.DataFrame(
            columns=["timestamp", "open", "high", "low", "close", "volume", "open_interest"]
        )

    column_order = ["timestamp", "open", "high", "low", "close", "volume", "open_interest"]
    col_count = min(raw_df.shape[1], len(column_order))
    raw_df = raw_df.iloc[:, :col_count].copy()
    raw_df.columns = column_order[:col_count]
    for missing in column_order[col_count:]:
        raw_df[missing] = 0.0

    raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"], utc=True, errors="coerce")
    for c in ["open", "high", "low", "close", "volume", "open_interest"]:
        raw_df[c] = pd.to_numeric(raw_df[c], errors="coerce")

    raw_df = raw_df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return raw_df[column_order]


@dataclass
class RetryConfig:
    retries: int = 3
    backoff_seconds: float = 0.5
    timeout_seconds: float = 20.0


class UpstoxBaseClient:
    """
    Base client with retries, validation and clean errors.
    """

    def __init__(
        self,
        access_token: Optional[str] = None,
        *,
        base_url: str = DEFAULT_BASE_URL,
        retry: Optional[RetryConfig] = None,
        config_env_path: Optional[Union[str, Path]] = None,
        session: Optional[requests.Session] = None,
    ) -> None:
        _best_effort_load_env(config_env_path)

        self.base_url = base_url.rstrip("/")
        self.retry = retry or RetryConfig()
        self.session = session or requests.Session()
        self.access_token = (
            access_token
            or os.environ.get("UPSTOX_ACCESS_TOKEN")
            or os.environ.get("UPSTOX_API_ACCESS_TOKEN")
            or ""
        )
        if not self.access_token:
            raise ValueError(
                "Upstox access token is required. Provide it explicitly or set UPSTOX_ACCESS_TOKEN."
            )

    def _headers(self, api_version: Optional[str] = None) -> Dict[str, str]:
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.access_token}",
        }
        if api_version:
            headers["Api-Version"] = api_version
        return headers

    def _get(self, path: str, *, params: Optional[Dict[str, Any]] = None, api_version: Optional[str] = None) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        last_error: Optional[Exception] = None
        retries = max(int(self.retry.retries), 0)
        timeout = max(float(self.retry.timeout_seconds), 1.0)
        backoff = max(float(self.retry.backoff_seconds), 0.0)

        for attempt in range(retries + 1):
            try:
                response = self.session.get(
                    url,
                    params=params or None,
                    headers=self._headers(api_version=api_version),
                    timeout=timeout,
                )
                if response.status_code in (429, 500, 502, 503, 504) and attempt < retries:
                    _sleep = min(backoff * (2 ** attempt), 30.0)  # cap at 30 s
                    if response.status_code == 429:
                        try:
                            _sleep = min(float(response.headers.get("Retry-After", _sleep)), 60.0)
                        except (ValueError, TypeError):
                            pass
                    time.sleep(_sleep)
                    continue

                if response.status_code >= 400:
                    body = (response.text or "").strip()
                    body = body[:400] + ("..." if len(body) > 400 else "")
                    raise UpstoxAPIError(f"HTTP {response.status_code} for {path}: {body}")

                payload = response.json()
                if not isinstance(payload, dict):
                    raise UpstoxAPIError(f"Unexpected JSON shape for {path}.")
                return payload
            except (requests.Timeout, requests.ConnectionError, requests.RequestException) as exc:
                last_error = exc
                if attempt >= retries:
                    break
                time.sleep(min(backoff * (2 ** attempt), 30.0))  # cap at 30 s

        raise UpstoxAPIError(f"Request failed for {path}: {last_error}")


class UpstoxAPIClients(UpstoxBaseClient):
    """
    Clients for Upstox endpoints used by the OMEGA historical-learning flow.
    """

    EXPIRED_CANDLE_INTERVALS = {"1minute", "30minute", "day", "week", "month"}
    OHLC_INTERVALS = {"I1", "I30", "1d"}

    def get_option_contracts(
        self,
        instrument_key: str,
        *,
        expiry_date: Optional[str] = None,
        page: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        instrument_key = _validate_non_empty(instrument_key, "instrument_key")
        params: Dict[str, Any] = {"instrument_key": instrument_key}
        if expiry_date:
            params["expiry_date"] = _validate_date_yyyy_mm_dd(expiry_date, "expiry_date")
        if page is not None:
            params["page"] = int(page)

        payload = self._get("/v2/option/contract", params=params, api_version="2.0")
        _json_success_or_raise(payload, endpoint="get_option_contracts")
        data = _extract_payload_items(payload)
        return list(data) if isinstance(data, list) else []

    def get_expired_historical_candle_data(
        self,
        *,
        expired_instrument_key: str,
        interval: str,
        to_date: str,
        from_date: str,
    ) -> pd.DataFrame:
        expired_instrument_key = _validate_non_empty(expired_instrument_key, "expired_instrument_key")
        interval = _validate_non_empty(interval, "interval")
        if interval not in self.EXPIRED_CANDLE_INTERVALS:
            raise ValueError(
                f"interval must be one of {sorted(self.EXPIRED_CANDLE_INTERVALS)}."
            )

        to_date = _validate_date_yyyy_mm_dd(to_date, "to_date")
        from_date = _validate_date_yyyy_mm_dd(from_date, "from_date")
        if from_date > to_date:
            raise ValueError("from_date must be <= to_date.")

        key_enc = quote(expired_instrument_key, safe="")
        path = (
            f"/v2/expired-instruments/historical-candle/"
            f"{key_enc}/{interval}/{to_date}/{from_date}"
        )
        payload = self._get(path, api_version="2.0")
        _json_success_or_raise(payload, endpoint="get_expired_historical_candle_data")
        data = _extract_payload_items(payload)
        candles = data if isinstance(data, list) else []
        return _candles_to_dataframe(candles)

    def get_full_market_quote(self, instrument_keys: Union[str, Sequence[str]]) -> List[Dict[str, Any]]:
        keys = _normalize_instrument_keys(instrument_keys, max_count=500)
        payload = self._get(
            "/v2/market-quote/quotes",
            params={"instrument_key": ",".join(keys)},
            api_version="2.0",
        )
        _json_success_or_raise(payload, endpoint="get_full_market_quote")
        data = _extract_payload_items(payload)
        if isinstance(data, dict):
            return [{"instrument_key": k, **(v if isinstance(v, dict) else {"value": v})} for k, v in data.items()]
        return list(data) if isinstance(data, list) else []

    def get_market_quote_ohlc(
        self,
        instrument_keys: Union[str, Sequence[str]],
        *,
        interval: str,
    ) -> List[Dict[str, Any]]:
        keys = _normalize_instrument_keys(instrument_keys, max_count=500)
        interval = _validate_non_empty(interval, "interval")
        if interval not in self.OHLC_INTERVALS:
            raise ValueError(f"interval must be one of {sorted(self.OHLC_INTERVALS)}.")

        payload = self._get(
            "/v2/market-quote/ohlc",
            params={"instrument_key": ",".join(keys), "interval": interval},
            api_version="2.0",
        )
        _json_success_or_raise(payload, endpoint="get_market_quote_ohlc")
        data = _extract_payload_items(payload)
        if isinstance(data, dict):
            return [{"instrument_key": k, **(v if isinstance(v, dict) else {"value": v})} for k, v in data.items()]
        return list(data) if isinstance(data, list) else []

    def get_option_greek(self, instrument_keys: Union[str, Sequence[str]]) -> List[Dict[str, Any]]:
        keys = _normalize_instrument_keys(instrument_keys, max_count=50)
        payload = self._get(
            "/v3/market-quote/option-greek",
            params={"instrument_key": ",".join(keys)},
        )
        _json_success_or_raise(payload, endpoint="get_option_greek")
        data = _extract_payload_items(payload)
        if isinstance(data, dict):
            return [{"instrument_key": k, **(v if isinstance(v, dict) else {"value": v})} for k, v in data.items()]
        return list(data) if isinstance(data, list) else []

    def get_ltp_v3(self, instrument_keys: Union[str, Sequence[str]]) -> List[Dict[str, Any]]:
        keys = _normalize_instrument_keys(instrument_keys, name="instrument_key")
        payload = self._get(
            "/v3/market-quote/ltp",
            params={"instrument_key": ",".join(keys)},
        )
        _json_success_or_raise(payload, endpoint="get_ltp_v3")
        data = _extract_payload_items(payload)
        if isinstance(data, dict):
            return [{"instrument_key": k, **(v if isinstance(v, dict) else {"value": v})} for k, v in data.items()]
        return list(data) if isinstance(data, list) else []

    def get_pc_option_chain(self, *, instrument_key: str, expiry_date: str) -> List[Dict[str, Any]]:
        instrument_key = _validate_non_empty(instrument_key, "instrument_key")
        expiry_date = _validate_date_yyyy_mm_dd(expiry_date, "expiry_date")
        payload = self._get(
            "/v2/option/chain",
            params={"instrument_key": instrument_key, "expiry_date": expiry_date},
            api_version="2.0",
        )
        _json_success_or_raise(payload, endpoint="get_pc_option_chain")
        data = _extract_payload_items(payload)
        return list(data) if isinstance(data, list) else []

    def get_expiries(self, *, instrument_key: str) -> List[str]:
        instrument_key = _validate_non_empty(instrument_key, "instrument_key")
        payload = self._get(
            "/v2/expired-instruments/expiries",
            params={"instrument_key": instrument_key},
            api_version="2.0",
        )
        _json_success_or_raise(payload, endpoint="get_expiries")
        data = _extract_payload_items(payload)
        if isinstance(data, dict):
            expiries = data.get("expiry_dates") or data.get("expiries") or []
        else:
            expiries = data
        return [str(x) for x in expiries if str(x).strip()]

    def get_expired_option_contracts(
        self,
        *,
        instrument_key: str,
        expiry_date: str,
        page: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        instrument_key = _validate_non_empty(instrument_key, "instrument_key")
        date_candidates = _expiry_date_candidates(expiry_date, "expiry_date")
        last_error: Optional[Exception] = None

        for candidate_date in date_candidates:
            params: Dict[str, Any] = {"instrument_key": instrument_key, "expiry_date": candidate_date}
            if page is not None:
                params["page"] = int(page)
            try:
                payload = self._get(
                    "/v2/expired-instruments/option/contract",
                    params=params,
                    api_version="2.0",
                )
                _json_success_or_raise(payload, endpoint="get_expired_option_contracts")
                data = _extract_payload_items(payload)
                return list(data) if isinstance(data, list) else []
            except UpstoxAPIError as exc:
                last_error = exc
                msg = str(exc)
                # Retry alternate format only on explicit invalid-date errors.
                invalid_date = ("UDAPI1088" in msg) or ("Invalid date" in msg)
                if invalid_date:
                    continue
                raise

        if last_error is not None:
            raise last_error
        return []


def build_upstox_client(
    access_token: Optional[str] = None,
    *,
    config_env_path: Optional[Union[str, Path]] = None,
    session: Optional[requests.Session] = None,
) -> UpstoxAPIClients:
    return UpstoxAPIClients(
        access_token=access_token,
        config_env_path=config_env_path,
        session=session,
    )


def get_session_access_token(session_state: Optional[Dict[str, Any]] = None) -> str:
    """
    Fetch access token from Streamlit session-state first, then environment.
    """
    if session_state:
        token = str(session_state.get("upstox_access_token", "") or "").strip()
        if token:
            return token
    _best_effort_load_env()
    token = (
        os.environ.get("UPSTOX_ACCESS_TOKEN")
        or os.environ.get("UPSTOX_API_ACCESS_TOKEN")
        or ""
    )
    token = str(token).strip()
    if not token:
        raise ValueError(
            "No Upstox access token found. Connect in Streamlit or set UPSTOX_ACCESS_TOKEN."
        )
    return token
