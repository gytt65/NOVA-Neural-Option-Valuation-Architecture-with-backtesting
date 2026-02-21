#!/usr/bin/env python3
"""
Historical Learning Pipeline (additive module).

Pulls expired option history from Upstox, builds features, computes residual labels
using the existing NIRV pricing stack, and updates OMEGA's ML corrector artifact.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from upstox_api_clients import UpstoxAPIClients, build_upstox_client


ProgressCallback = Callable[[str, float], None]


SUPPORTED_INTERVALS = ("1minute", "30minute", "day", "week", "month")


@dataclass
class HistoricalLearningConfig:
    underlying_instrument_key: str = "NSE_INDEX|Nifty 50"
    from_date: str = "2024-01-01"
    to_date: str = "2024-12-31"
    interval: str = "day"
    strike_window: int = 6
    top_n_contracts: int = 0
    max_contracts_per_expiry: int = 8
    max_expiries: int = 2
    max_pages_per_expiry: int = 2
    train_model: bool = True
    risk_free_rate: float = 0.065
    dividend_yield: float = 0.012
    india_vix_proxy: float = 15.0
    output_root: str = "omega_data"
    model_path: str = "omega_data/pricing_model.joblib"
    n_paths: int = 1200
    n_bootstrap: int = 120
    max_rows_for_pricing: int = 120
    max_label_runtime_sec: float = 45.0
    max_pull_runtime_sec: float = 90.0
    walk_forward_splits: int = 3
    rollback_on_degradation: bool = True
    degradation_tolerance_pct: float = 2.0
    backup_existing_model: bool = True


def _emit(progress_cb: Optional[ProgressCallback], message: str, pct: float) -> None:
    if progress_cb:
        try:
            progress_cb(message, float(np.clip(pct, 0.0, 1.0)))
        except Exception:
            return


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if np.isfinite(v):
            return v
    except Exception:
        pass
    return float(default)


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(round(float(x)))
    except Exception:
        return int(default)


def _parse_date_flexible(raw: Any) -> Optional[datetime]:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    try:
        return pd.to_datetime(s, utc=True, errors="raise").to_pydatetime()
    except Exception:
        return None


def _option_type(contract: Dict[str, Any]) -> str:
    raw = str(contract.get("option_type") or contract.get("instrument_type") or "").upper().strip()
    if raw in {"CE", "CALL"}:
        return "CE"
    if raw in {"PE", "PUT"}:
        return "PE"
    tsym = str(contract.get("trading_symbol") or contract.get("symbol") or "").upper()
    if tsym.endswith("CE"):
        return "CE"
    if tsym.endswith("PE"):
        return "PE"
    return ""


def _strike(contract: Dict[str, Any]) -> Optional[float]:
    for k in ("strike_price", "strike", "strikePrice"):
        if k in contract:
            val = _safe_float(contract.get(k), default=np.nan)
            if np.isfinite(val) and val > 0:
                return float(val)
    tsym = str(contract.get("trading_symbol") or contract.get("symbol") or "")
    m = re.search(r"(\d{4,7})(CE|PE)$", tsym.upper())
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    return None


def _contract_key(contract: Dict[str, Any]) -> str:
    for k in ("instrument_key", "expired_instrument_key", "symbol", "trading_symbol"):
        val = str(contract.get(k) or "").strip()
        if val:
            return val
    return ""


def _contract_expiry(contract: Dict[str, Any], fallback: str) -> str:
    val = str(contract.get("expiry") or contract.get("expiry_date") or "").strip()
    return val or str(fallback)


def _contract_volume(contract: Dict[str, Any]) -> float:
    for k in ("volume", "traded_volume", "total_volume"):
        if k in contract:
            return max(_safe_float(contract.get(k), 0.0), 0.0)
    return 0.0


def _dedup_union(existing: Optional[pd.DataFrame], incoming: pd.DataFrame, subset: Sequence[str]) -> pd.DataFrame:
    if existing is None or existing.empty:
        out = incoming.copy()
    else:
        out = pd.concat([existing, incoming], ignore_index=True)
    return out.drop_duplicates(subset=list(subset), keep="last").reset_index(drop=True)


def _read_parquet_if_exists(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)


def _split_time_train_test(df: pd.DataFrame, train_frac: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return df.copy(), df.copy()
    n = len(df)
    cut = max(1, min(n - 1, int(round(n * train_frac))))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def _residual_metrics(actual: Sequence[float], pred: Sequence[float]) -> Dict[str, Optional[float]]:
    a = np.asarray(actual, dtype=float)
    p = np.asarray(pred, dtype=float)
    if a.size == 0 or p.size == 0 or a.size != p.size:
        return {
            "mae": None,
            "rmse": None,
            "mape_pct": None,
            "direction_hit_rate": None,
            "signed_corr": None,
            "bias": None,
        }

    err = a - p
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    mape = float(np.mean(np.abs(err) / np.maximum(np.abs(a), 1e-6)) * 100.0)
    direction_hit = float(np.mean(np.sign(a) == np.sign(p)))
    signed_corr = float(np.corrcoef(a, p)[0, 1]) if a.size >= 2 and np.std(a) > 0 and np.std(p) > 0 else None
    bias = float(np.mean(p - a))
    return {
        "mae": mae,
        "rmse": rmse,
        "mape_pct": mape,
        "direction_hit_rate": direction_hit,
        "signed_corr": signed_corr,
        "bias": bias,
    }


def _estimate_spot_hint(client: UpstoxAPIClients, instrument_key: str) -> float:
    try:
        quotes = client.get_full_market_quote([instrument_key])
    except Exception:
        return float("nan")
    if not quotes:
        return float("nan")
    q = quotes[0] if isinstance(quotes[0], dict) else {}
    for k in ("last_price", "ltp", "close"):
        if k in q and np.isfinite(_safe_float(q.get(k), np.nan)):
            return float(q.get(k))
    nested = q.get("ohlc") or {}
    for k in ("close", "open"):
        if k in nested and np.isfinite(_safe_float(nested.get(k), np.nan)):
            return float(nested.get(k))
    return float("nan")


def _filter_expiries(expiries: Sequence[str], from_date: str, to_date: str) -> List[str]:
    from_dt = _parse_date_flexible(from_date)
    to_dt = _parse_date_flexible(to_date)
    if from_dt is None or to_dt is None:
        return list(expiries)

    out: List[Tuple[datetime, str]] = []
    for ex in expiries:
        ex_dt = _parse_date_flexible(ex)
        if ex_dt is None:
            continue
        if from_dt <= ex_dt <= to_dt:
            out.append((ex_dt, str(ex)))
    out.sort(key=lambda x: x[0])
    return [x[1] for x in out]


def _select_contracts_for_expiry(
    contracts: Sequence[Dict[str, Any]],
    *,
    spot_hint: float,
    strike_window: int,
    top_n_contracts: int,
    max_contracts_per_expiry: int,
) -> List[Dict[str, Any]]:
    cleaned: List[Dict[str, Any]] = []
    for c in contracts:
        k = _contract_key(c)
        t = _option_type(c)
        st = _strike(c)
        if not k or not t or st is None:
            continue
        row = dict(c)
        row["_contract_key"] = k
        row["_option_type"] = t
        row["_strike"] = st
        row["_volume"] = _contract_volume(c)
        cleaned.append(row)

    if not cleaned:
        return []

    if top_n_contracts > 0 and any(r.get("_volume", 0.0) > 0 for r in cleaned):
        cleaned = sorted(cleaned, key=lambda r: r.get("_volume", 0.0), reverse=True)[:top_n_contracts]
    else:
        strikes = sorted({float(r["_strike"]) for r in cleaned if np.isfinite(_safe_float(r["_strike"], np.nan))})
        if not strikes:
            return cleaned[:max_contracts_per_expiry]
        if not np.isfinite(spot_hint):
            spot_hint = float(np.median(strikes))
        atm_idx = min(range(len(strikes)), key=lambda i: abs(strikes[i] - spot_hint))
        lo = max(0, atm_idx - max(strike_window, 0))
        hi = min(len(strikes), atm_idx + max(strike_window, 0) + 1)
        strike_bucket = set(strikes[lo:hi])
        cleaned = [r for r in cleaned if float(r["_strike"]) in strike_bucket]

    # Keep both CE and PE where possible, then cap.
    ce = [r for r in cleaned if r.get("_option_type") == "CE"]
    pe = [r for r in cleaned if r.get("_option_type") == "PE"]
    merged: List[Dict[str, Any]] = []
    for group in (ce, pe):
        merged.extend(group[: max_contracts_per_expiry // 2 + 1])
    if not merged:
        merged = cleaned
    unique: Dict[str, Dict[str, Any]] = {}
    for row in merged:
        unique[row["_contract_key"]] = row
    return list(unique.values())[:max_contracts_per_expiry]


def pull_historical_option_data(
    client: UpstoxAPIClients,
    config: HistoricalLearningConfig,
    *,
    progress_cb: Optional[ProgressCallback] = None,
) -> Dict[str, Any]:
    started = time.perf_counter()
    _emit(progress_cb, "Fetching expiry list...", 0.05)
    expiries = client.get_expiries(instrument_key=config.underlying_instrument_key)
    expiries = _filter_expiries(expiries, config.from_date, config.to_date)
    if config.max_expiries > 0:
        expiries = expiries[-config.max_expiries :]

    spot_hint = _estimate_spot_hint(client, config.underlying_instrument_key)
    all_contracts: List[Dict[str, Any]] = []
    all_candles: List[pd.DataFrame] = []
    total_exp = max(len(expiries), 1)

    for i, expiry in enumerate(expiries):
        if (time.perf_counter() - started) > float(max(config.max_pull_runtime_sec, 5.0)):
            _emit(progress_cb, "Data pull time budget reached; using partial data.", 0.78)
            break
        pct = 0.10 + 0.65 * (i / total_exp)
        _emit(progress_cb, f"Loading expired contracts for {expiry}...", pct)
        contracts: List[Dict[str, Any]] = []
        for page in range(1, max(config.max_pages_per_expiry, 1) + 1):
            if (time.perf_counter() - started) > float(max(config.max_pull_runtime_sec, 5.0)):
                break
            batch = client.get_expired_option_contracts(
                instrument_key=config.underlying_instrument_key,
                expiry_date=expiry,
                page=page,
            )
            if not batch:
                break
            contracts.extend(batch)
            # Defensive stop: if fewer rows than a page, pagination likely complete.
            if len(batch) < 25:
                break

        selected = _select_contracts_for_expiry(
            contracts,
            spot_hint=spot_hint,
            strike_window=config.strike_window,
            top_n_contracts=config.top_n_contracts,
            max_contracts_per_expiry=config.max_contracts_per_expiry,
        )
        all_contracts.extend(selected)
        if not selected:
            continue

        for j, c in enumerate(selected):
            if (time.perf_counter() - started) > float(max(config.max_pull_runtime_sec, 5.0)):
                _emit(progress_cb, "Data pull time budget reached; stopping contract loop.", 0.79)
                break
            k = c.get("_contract_key") or _contract_key(c)
            if not k:
                continue
            msg = f"Candles {expiry}: {j + 1}/{len(selected)}"
            _emit(progress_cb, msg, pct + (0.60 / total_exp) * ((j + 1) / max(len(selected), 1)))
            candles = client.get_expired_historical_candle_data(
                expired_instrument_key=k,
                interval=config.interval,
                to_date=config.to_date,
                from_date=config.from_date,
            )
            if candles is None or candles.empty:
                continue
            df = candles.copy()
            df["instrument_key"] = str(k)
            df["underlying_instrument_key"] = str(config.underlying_instrument_key)
            df["expiry_date"] = _contract_expiry(c, fallback=str(expiry))
            df["strike_price"] = _safe_float(c.get("_strike"), np.nan)
            df["option_type"] = c.get("_option_type") or _option_type(c)
            df["trading_symbol"] = str(c.get("trading_symbol") or c.get("symbol") or "")
            all_candles.append(df)

    if all_candles:
        candles_df = pd.concat(all_candles, ignore_index=True)
        candles_df = candles_df.drop_duplicates(
            subset=["instrument_key", "timestamp"], keep="last"
        ).sort_values(["instrument_key", "timestamp"]).reset_index(drop=True)
    else:
        candles_df = pd.DataFrame(
            columns=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "open_interest",
                "instrument_key",
                "underlying_instrument_key",
                "expiry_date",
                "strike_price",
                "option_type",
                "trading_symbol",
            ]
        )

    return {
        "expiries_used": expiries,
        "contracts": all_contracts,
        "candles": candles_df,
        "spot_hint": spot_hint,
    }


def engineer_features_from_candles(candles_df: pd.DataFrame) -> pd.DataFrame:
    if candles_df is None or candles_df.empty:
        return pd.DataFrame()

    df = candles_df.copy()
    if "timestamp" not in df.columns:
        raise ValueError("candles_df must include a 'timestamp' column.")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values(["instrument_key", "timestamp"]).reset_index(drop=True)

    for col in ("open", "high", "low", "close", "volume", "open_interest", "strike_price"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    g = df.groupby("instrument_key", dropna=False, sort=False)
    df["ret_1"] = g["close"].pct_change()
    df["log_ret_1"] = np.log(df["close"] / g["close"].shift(1))
    df["rv_10"] = g["log_ret_1"].transform(lambda s: s.rolling(10, min_periods=5).std() * np.sqrt(252))
    df["rv_20"] = g["log_ret_1"].transform(lambda s: s.rolling(20, min_periods=10).std() * np.sqrt(252))
    df["hl_range_pct"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
    df["volume_ma_10"] = g["volume"].transform(lambda s: s.rolling(10, min_periods=3).mean())
    df["volume_ratio"] = df["volume"] / df["volume_ma_10"].replace(0, np.nan)
    df["oi_change"] = g["open_interest"].diff()
    df["oi_ratio"] = df["open_interest"] / g["open_interest"].transform(lambda s: s.rolling(10, min_periods=3).mean()).replace(0, np.nan)
    df["dow"] = df["timestamp"].dt.dayofweek
    df["hour"] = df["timestamp"].dt.hour

    df["ret_1"] = df["ret_1"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["log_ret_1"] = df["log_ret_1"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    for col in ("rv_10", "rv_20", "hl_range_pct", "volume_ratio", "oi_change", "oi_ratio"):
        df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return df


def _compute_model_residual_labels(
    features_df: pd.DataFrame,
    config: HistoricalLearningConfig,
    *,
    progress_cb: Optional[ProgressCallback] = None,
) -> pd.DataFrame:
    if features_df.empty:
        out = features_df.copy()
        out["model_fair_value"] = np.nan
        out["residual_label"] = np.nan
        return out

    # Lazy imports to keep this module lightweight when training is not used.
    from nirv_model import NIRVModel  # noqa: WPS433

    out = features_df.copy()
    out["model_fair_value"] = np.nan
    out["residual_label"] = np.nan

    nirv = NIRVModel(n_paths=max(config.n_paths, 1000), n_bootstrap=max(config.n_bootstrap, 100))

    # Keep pricing bounded for responsiveness.
    max_rows = max(config.max_rows_for_pricing, 100)
    if len(out) > max_rows:
        out = out.iloc[-max_rows:].copy()

    grouped_returns = out.groupby("instrument_key", dropna=False)["log_ret_1"]
    total = max(len(out), 1)
    started = time.perf_counter()
    for idx, row in enumerate(out.itertuples(index=True), start=1):
        if (time.perf_counter() - started) > float(max(config.max_label_runtime_sec, 5.0)):
            _emit(progress_cb, "Labeling time budget reached; finishing with partial labels.", 0.94)
            break
        if idx % 25 == 0 or idx == total:
            _emit(progress_cb, f"Running NIRV pricing labels... ({idx}/{total})", 0.86 + 0.08 * (idx / total))

        strike = _safe_float(getattr(row, "strike_price", np.nan), np.nan)
        close_price = _safe_float(getattr(row, "close", np.nan), np.nan)
        if not np.isfinite(strike) or strike <= 0 or not np.isfinite(close_price) or close_price <= 0:
            continue

        ts = getattr(row, "timestamp", None)
        expiry_raw = getattr(row, "expiry_date", None)
        expiry_dt = _parse_date_flexible(expiry_raw)
        ts_dt = pd.to_datetime(ts, utc=True, errors="coerce")
        if expiry_dt is None or pd.isna(ts_dt):
            T = 7.0 / 365.0
        else:
            T = max((expiry_dt - ts_dt.to_pydatetime()).total_seconds() / (365.0 * 24 * 3600), 1.0 / 365.0)

        option_type = str(getattr(row, "option_type", "CE") or "CE").upper().strip()
        option_type = "CE" if option_type in {"CE", "CALL"} else "PE"

        # Spot proxy:
        # expired options do not always provide synchronized underlying history in this endpoint.
        # Use strike as stable fallback proxy when spot is unavailable.
        spot_proxy = _safe_float(getattr(row, "underlying_spot_price", np.nan), np.nan)
        if not np.isfinite(spot_proxy) or spot_proxy <= 0:
            spot_proxy = strike

        rv_proxy = _safe_float(getattr(row, "rv_20", np.nan), np.nan)
        if not np.isfinite(rv_proxy) or rv_proxy <= 0:
            rv_proxy = 0.18
        returns_series = grouped_returns.get_group(getattr(row, "instrument_key")) if getattr(row, "instrument_key") in grouped_returns.groups else pd.Series(dtype=float)
        returns_np = returns_series.tail(30).to_numpy(dtype=float, copy=False)
        if returns_np.size < 30:
            returns_np = np.pad(returns_np, (30 - returns_np.size, 0), mode="constant")

        try:
            priced = nirv.price_option(
                spot=spot_proxy,
                strike=strike,
                T=T,
                r=float(config.risk_free_rate),
                q=float(config.dividend_yield),
                option_type=option_type,
                market_price=close_price,
                india_vix=float(config.india_vix_proxy),
                fii_net_flow=0.0,
                dii_net_flow=0.0,
                days_to_rbi=30,
                pcr_oi=1.0,
                returns_30d=returns_np,
            )
            fair = _safe_float(getattr(priced, "fair_value", np.nan), np.nan)
        except Exception:
            fair = np.nan

        if np.isfinite(fair) and fair > 0:
            residual = (close_price - fair) / max(fair, 1e-6)
            out.at[row.Index, "model_fair_value"] = fair
            out.at[row.Index, "residual_label"] = residual

    return out


def _build_omega_features(row: pd.Series, config: HistoricalLearningConfig) -> Dict[str, float]:
    from omega_model import FeatureFactory  # noqa: WPS433

    spot = _safe_float(row.get("underlying_spot_price", np.nan), np.nan)
    strike = _safe_float(row.get("strike_price", np.nan), np.nan)
    if not np.isfinite(spot) or spot <= 0:
        spot = strike if np.isfinite(strike) and strike > 0 else 1.0
    if not np.isfinite(strike) or strike <= 0:
        strike = max(spot, 1.0)

    ts = pd.to_datetime(row.get("timestamp"), utc=True, errors="coerce")
    expiry_dt = _parse_date_flexible(row.get("expiry_date"))
    if pd.isna(ts) or expiry_dt is None:
        T = 7.0 / 365.0
    else:
        T = max((expiry_dt - ts.to_pydatetime()).total_seconds() / (365.0 * 24 * 3600), 1.0 / 365.0)

    iv_proxy = _safe_float(row.get("rv_20", np.nan), np.nan)
    if not np.isfinite(iv_proxy) or iv_proxy <= 0:
        iv_proxy = max(_safe_float(config.india_vix_proxy, 15.0) / 100.0, 0.1)

    market_price = max(_safe_float(row.get("close", 0.0), 0.0), 0.01)
    bid_proxy = max(market_price * 0.995, 0.01)
    ask_proxy = max(market_price * 1.005, bid_proxy + 0.01)
    oi = max(_safe_float(row.get("open_interest", 0.0), 0.0), 0.0)
    vol = max(_safe_float(row.get("volume", 0.0), 0.0), 0.0)

    mkt = {
        "spot": spot,
        "strike": strike,
        "T": T,
        "iv": iv_proxy,
        "hv_30d": max(_safe_float(row.get("rv_20", iv_proxy), iv_proxy), 0.01),
        "vix": max(_safe_float(config.india_vix_proxy, 15.0), 1.0),
        "market_price": market_price,
        "bid": bid_proxy,
        "ask": ask_proxy,
        "volume_oi_ratio": vol / max(oi, 1.0),
        "pcr_oi": 1.0,
        "iv_rank": 50.0,
        "iv_percentile": 50.0,
        "regime": "Sideways",
    }
    return FeatureFactory.extract(mkt)


def _append_training_sample_without_retrain(
    ml: Any,
    features: Dict[str, float],
    target: float,
) -> None:
    """
    Add samples directly to buffers when available, then train once.
    This avoids repeated expensive retraining in long loops.
    """
    if hasattr(ml, "training_X") and hasattr(ml, "training_y"):
        try:
            from omega_model import FeatureFactory  # noqa: WPS433

            arr = FeatureFactory.to_array(features)
            ml.training_X.append(arr.tolist())
            ml.training_y.append(float(target))
            return
        except Exception:
            pass
    ml.add_sample(features, float(target))


def _evaluate_ml_on_df(ml: Any, df: pd.DataFrame, config: HistoricalLearningConfig) -> Dict[str, Optional[float]]:
    if df.empty:
        out = _residual_metrics([], [])
        out["rows"] = 0
        return out
    preds: List[float] = []
    actual: List[float] = []
    for row in df.itertuples(index=False):
        row_s = pd.Series(row._asdict())
        target = _safe_float(row_s.get("residual_label", np.nan), np.nan)
        if not np.isfinite(target):
            continue
        feats = _build_omega_features(row_s, config)
        pred, _ = ml.predict_correction(feats)
        preds.append(float(pred))
        actual.append(float(target))
    out = _residual_metrics(actual, preds)
    out["rows"] = int(len(actual))
    return out


def _aggregate_fold_metrics(fold_metrics: List[Dict[str, Optional[float]]]) -> Dict[str, Optional[float]]:
    if not fold_metrics:
        return {
            "folds_used": 0,
            "rows_total": 0,
            "mae": None,
            "rmse": None,
            "mape_pct": None,
            "direction_hit_rate": None,
            "signed_corr": None,
            "bias": None,
        }

    weights = np.array([max(int(m.get("rows", 0) or 0), 0) for m in fold_metrics], dtype=float)
    if np.sum(weights) <= 0:
        weights = np.ones(len(fold_metrics), dtype=float)
    weights = weights / np.sum(weights)

    out: Dict[str, Optional[float]] = {
        "folds_used": int(len(fold_metrics)),
        "rows_total": int(sum(int(m.get("rows", 0) or 0) for m in fold_metrics)),
    }
    keys = ["mae", "rmse", "mape_pct", "direction_hit_rate", "signed_corr", "bias"]
    for k in keys:
        vals = np.array([_safe_float(m.get(k), np.nan) for m in fold_metrics], dtype=float)
        mask = np.isfinite(vals)
        if not np.any(mask):
            out[k] = None
            continue
        w = weights.copy()
        w[~mask] = 0.0
        if np.sum(w) <= 0:
            out[k] = None
            continue
        w = w / np.sum(w)
        out[k] = float(np.sum(vals[mask] * w[mask]))
    return out


def _walk_forward_evaluate(valid_df: pd.DataFrame, config: HistoricalLearningConfig) -> Dict[str, Any]:
    from omega_model import MLPricingCorrector  # noqa: WPS433

    df = valid_df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    splits = max(1, min(int(config.walk_forward_splits), 8))
    fold_size = max(10, n // (splits + 1))

    folds: List[Dict[str, Any]] = []
    for i in range(1, splits + 1):
        train_end = fold_size * i
        test_end = min(n, fold_size * (i + 1))
        if train_end < 30 or (test_end - train_end) < 5:
            continue

        train_df = df.iloc[:train_end].copy()
        test_df = df.iloc[train_end:test_end].copy()

        tmp_model_path = Path(tempfile.gettempdir()) / f"omega_walkforward_{os.getpid()}_{i}.joblib"
        if tmp_model_path.exists():
            try:
                tmp_model_path.unlink()
            except Exception:
                pass

        ml = MLPricingCorrector(model_path=str(tmp_model_path))
        for row in train_df.itertuples(index=False):
            row_s = pd.Series(row._asdict())
            feats = _build_omega_features(row_s, config)
            target = float(row_s.get("residual_label", 0.0))
            _append_training_sample_without_retrain(ml, feats, target)
        if hasattr(ml, "_train"):
            try:
                ml._train()
            except Exception:
                pass

        fold_metric = _evaluate_ml_on_df(ml, test_df, config)
        fold_metric["fold"] = i
        fold_metric["train_rows"] = int(len(train_df))
        fold_metric["test_rows"] = int(len(test_df))
        folds.append(fold_metric)

        try:
            if tmp_model_path.exists():
                tmp_model_path.unlink()
        except Exception:
            pass

    return {
        "aggregate": _aggregate_fold_metrics(folds),
        "folds": folds,
    }


def train_or_update_ml_corrector(
    labeled_df: pd.DataFrame,
    config: HistoricalLearningConfig,
    *,
    progress_cb: Optional[ProgressCallback] = None,
) -> Dict[str, Any]:
    from omega_model import MLPricingCorrector  # noqa: WPS433

    valid = labeled_df[np.isfinite(labeled_df.get("residual_label", np.nan))].copy()
    if valid.empty:
        return {
            "trained": False,
            "message": "No valid residual labels to train on.",
            "rows_train": 0,
            "rows_test": 0,
            "mae_test": None,
            "rmse_test": None,
            "mape_test_pct": None,
            "direction_hit_rate_test": None,
            "baseline_metrics": None,
            "candidate_metrics": None,
            "walk_forward": None,
            "rolled_back": False,
            "backup_model_path": None,
        }

    valid = valid.sort_values("timestamp").reset_index(drop=True)
    train_df, test_df = _split_time_train_test(valid, train_frac=0.8)
    _emit(progress_cb, "Updating ML corrector...", 0.95)

    # Evaluate existing artifact on the holdout before any update.
    baseline_metrics = None
    baseline_ml = MLPricingCorrector(model_path=config.model_path)
    if bool(getattr(baseline_ml, "is_trained", False)):
        baseline_metrics = _evaluate_ml_on_df(baseline_ml, test_df, config)

    # Backup existing model so rollback is deterministic.
    backup_model_path = None
    model_path_obj = Path(config.model_path)
    if config.backup_existing_model and model_path_obj.exists():
        backup_dir = Path(config.output_root) / "models" / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup_model_path = backup_dir / f"pricing_model_{datetime.now(tz=timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.joblib"
        try:
            shutil.copy2(model_path_obj, backup_model_path)
        except Exception:
            backup_model_path = None

    ml = MLPricingCorrector(model_path=config.model_path)
    for row in train_df.itertuples(index=False):
        row_s = pd.Series(row._asdict())
        feats = _build_omega_features(row_s, config)
        target = float(row_s.get("residual_label", 0.0))
        _append_training_sample_without_retrain(ml, feats, target)

    # Ensure one final fit after bulk ingestion.
    if hasattr(ml, "_train"):
        try:
            ml._train()
        except Exception:
            pass

    candidate_metrics = _evaluate_ml_on_df(ml, test_df, config)
    walk_forward = _walk_forward_evaluate(valid, config)

    rolled_back = False
    degradation_detected = False
    if (
        config.rollback_on_degradation
        and baseline_metrics is not None
        and candidate_metrics is not None
    ):
        b_mae = _safe_float(baseline_metrics.get("mae"), np.nan)
        c_mae = _safe_float(candidate_metrics.get("mae"), np.nan)
        if np.isfinite(b_mae) and np.isfinite(c_mae):
            threshold = b_mae * (1.0 + max(config.degradation_tolerance_pct, 0.0) / 100.0)
            degradation_detected = bool(c_mae > threshold)
            if degradation_detected and backup_model_path and Path(backup_model_path).exists():
                try:
                    shutil.copy2(backup_model_path, model_path_obj)
                    rolled_back = True
                except Exception:
                    rolled_back = False

    active_metrics = baseline_metrics if rolled_back and baseline_metrics is not None else candidate_metrics

    return {
        "trained": bool(getattr(ml, "is_trained", False)),
        "rows_train": int(len(train_df)),
        "rows_test": int(len(test_df)),
        "mae_test": active_metrics.get("mae") if isinstance(active_metrics, dict) else None,
        "rmse_test": active_metrics.get("rmse") if isinstance(active_metrics, dict) else None,
        "mape_test_pct": active_metrics.get("mape_pct") if isinstance(active_metrics, dict) else None,
        "direction_hit_rate_test": active_metrics.get("direction_hit_rate") if isinstance(active_metrics, dict) else None,
        "signed_corr_test": active_metrics.get("signed_corr") if isinstance(active_metrics, dict) else None,
        "bias_test": active_metrics.get("bias") if isinstance(active_metrics, dict) else None,
        "baseline_metrics": baseline_metrics,
        "candidate_metrics": candidate_metrics,
        "walk_forward": walk_forward,
        "rolled_back": bool(rolled_back),
        "degradation_detected": bool(degradation_detected),
        "backup_model_path": str(backup_model_path) if backup_model_path else None,
        "model_path": config.model_path,
    }


def pull_and_train(
    *,
    access_token: str,
    config: HistoricalLearningConfig,
    progress_cb: Optional[ProgressCallback] = None,
    client: Optional[UpstoxAPIClients] = None,
) -> Dict[str, Any]:
    if config.interval not in SUPPORTED_INTERVALS:
        raise ValueError(f"interval must be one of {SUPPORTED_INTERVALS}.")

    client = client or build_upstox_client(access_token=access_token)

    output_root = Path(config.output_root)
    raw_dir = output_root / "historical" / "raw"
    proc_dir = output_root / "historical" / "processed"
    models_dir = output_root / "models"
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    _emit(progress_cb, "Starting historical pull...", 0.02)
    pulled = pull_historical_option_data(client, config, progress_cb=progress_cb)
    candles = pulled.get("candles", pd.DataFrame())
    contracts = pulled.get("contracts", [])
    _emit(progress_cb, "Engineering features...", 0.82)
    features_df = engineer_features_from_candles(candles)
    labeled = _compute_model_residual_labels(features_df, config, progress_cb=progress_cb)

    # Persist raw artifacts with dedup.
    run_ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_contracts_path = raw_dir / f"contracts_{run_ts}.json"
    raw_candles_run_path = raw_dir / f"candles_{run_ts}.parquet"
    raw_candles_master_path = raw_dir / "candles_master.parquet"
    proc_features_path = proc_dir / "features.parquet"
    report_path = proc_dir / f"training_report_{run_ts}.json"

    _write_json(
        raw_contracts_path,
        {
            "generated_at": run_ts,
            "underlying_instrument_key": config.underlying_instrument_key,
            "config": asdict(config),
            "contracts": contracts,
        },
    )

    if not candles.empty:
        candles.to_parquet(raw_candles_run_path, index=False)
        prev_raw = _read_parquet_if_exists(raw_candles_master_path)
        merged_raw = _dedup_union(prev_raw, candles, subset=["instrument_key", "timestamp"])
        merged_raw.to_parquet(raw_candles_master_path, index=False)

    if not labeled.empty:
        prev_proc = _read_parquet_if_exists(proc_features_path)
        merged_proc = _dedup_union(prev_proc, labeled, subset=["instrument_key", "timestamp"])
        merged_proc.to_parquet(proc_features_path, index=False)
    else:
        merged_proc = labeled

    train_summary = {
        "trained": False,
        "rows_train": 0,
        "rows_test": 0,
        "mae_test": None,
        "rmse_test": None,
        "mape_test_pct": None,
        "direction_hit_rate_test": None,
        "signed_corr_test": None,
        "bias_test": None,
        "baseline_metrics": None,
        "candidate_metrics": None,
        "walk_forward": None,
        "rolled_back": False,
        "degradation_detected": False,
        "backup_model_path": None,
        "model_path": config.model_path,
    }
    if config.train_model and not merged_proc.empty:
        train_summary = train_or_update_ml_corrector(merged_proc, config, progress_cb=progress_cb)

    report = {
        "generated_at": run_ts,
        "config": asdict(config),
        "rows_raw_candles": int(len(candles)),
        "rows_processed": int(len(merged_proc)),
        "rows_labeled": int(np.isfinite(merged_proc.get("residual_label", np.nan)).sum() if not merged_proc.empty else 0),
        "expiries_used": pulled.get("expiries_used", []),
        "spot_hint": pulled.get("spot_hint"),
        "training": train_summary,
        "artifacts": {
            "raw_contracts_path": str(raw_contracts_path),
            "raw_candles_run_path": str(raw_candles_run_path),
            "raw_candles_master_path": str(raw_candles_master_path),
            "processed_features_path": str(proc_features_path),
            "training_report_path": str(report_path),
        },
    }
    _write_json(report_path, report)
    _emit(progress_cb, "Historical learning finished.", 1.0)
    return report
