"""
nse_specs.py — NSE Contract Specification Resolver
==================================================

Single source of truth for expiry dates, lot sizes, tick sizes,
and minute-accurate time-to-expiry calculations.
"""

from __future__ import annotations

import datetime as _dt
import os
import warnings
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd


DateLike = Union[str, _dt.date, _dt.datetime, pd.Timestamp]


DEFAULT_LOT_SIZES: Dict[str, int] = {
    "NIFTY": 25,
    "BANKNIFTY": 15,
    "FINNIFTY": 25,
    "MIDCPNIFTY": 50,
}

DEFAULT_TICK_SIZES: Dict[str, float] = {
    "NIFTY": 0.05,
    "BANKNIFTY": 0.05,
    "FINNIFTY": 0.05,
    "MIDCPNIFTY": 0.05,
}

NSE_OPEN_TIME = _dt.time(9, 15)
NSE_CLOSE_TIME = _dt.time(15, 30)


def _to_date(x: DateLike) -> _dt.date:
    if isinstance(x, _dt.datetime):
        return x.date()
    if isinstance(x, _dt.date):
        return x
    ts = pd.Timestamp(x)
    return ts.to_pydatetime().date()


def _to_datetime(x: DateLike) -> _dt.datetime:
    if isinstance(x, _dt.datetime):
        return x
    if isinstance(x, _dt.date):
        return _dt.datetime.combine(x, NSE_CLOSE_TIME)
    return pd.Timestamp(x).to_pydatetime()


def _normalize_symbol(symbol: str) -> str:
    s = str(symbol or "").upper().strip()
    for tok in ("-I", "IDX", "INDEX", "FUT", "OPT"):
        s = s.replace(tok, "")
    return "".join(ch for ch in s if ch.isalnum())


def _is_trading_day(date_obj: _dt.date, exchange_calendar=None) -> bool:
    if exchange_calendar is not None and hasattr(exchange_calendar, "is_trading_day"):
        try:
            return bool(exchange_calendar.is_trading_day(date_obj))
        except Exception:
            pass
    return date_obj.weekday() < 5


def _previous_trading_day(date_obj: _dt.date, exchange_calendar=None) -> _dt.date:
    d = date_obj
    while not _is_trading_day(d, exchange_calendar):
        d -= _dt.timedelta(days=1)
    return d


def _next_weekday(date_obj: _dt.date, weekday: int) -> _dt.date:
    # weekday: Monday=0 ... Sunday=6
    delta = (weekday - date_obj.weekday()) % 7
    return date_obj + _dt.timedelta(days=delta)


def _last_weekday_of_month(year: int, month: int, weekday: int) -> _dt.date:
    if month == 12:
        next_month = _dt.date(year + 1, 1, 1)
    else:
        next_month = _dt.date(year, month + 1, 1)
    d = next_month - _dt.timedelta(days=1)
    while d.weekday() != weekday:
        d -= _dt.timedelta(days=1)
    return d


@dataclass
class _ContractsStore:
    frame: pd.DataFrame
    source: str


class ContractSpecResolver:
    """
    Resolves NSE derivatives contract specs from local contract-master files
    with optional broker fallback.

    The broker fallback is intentionally generic:
      - get_contract_master() -> DataFrame/list[dict]
      - get_expiry_dates(underlying) -> iterable of date-like
      - get_lot_size(symbol, expiry_date) -> int
    """

    def __init__(
        self,
        contract_master_path: Optional[str] = None,
        broker_source=None,
        fallback_lot_sizes: Optional[Dict[str, int]] = None,
        fallback_tick_sizes: Optional[Dict[str, float]] = None,
        weekly_expiry_weekday: int = 1,  # Tuesday
    ):
        self.contract_master_path = (
            contract_master_path
            or os.environ.get("NSE_CONTRACT_MASTER_PATH")
            or ""
        )
        self.broker_source = broker_source
        self.weekly_expiry_weekday = int(weekly_expiry_weekday)
        self._fallback_lot_sizes = dict(DEFAULT_LOT_SIZES)
        if fallback_lot_sizes:
            self._fallback_lot_sizes.update(
                {str(k).upper(): int(v) for k, v in fallback_lot_sizes.items()}
            )
        self._fallback_tick_sizes = dict(DEFAULT_TICK_SIZES)
        if fallback_tick_sizes:
            self._fallback_tick_sizes.update(
                {str(k).upper(): float(v) for k, v in fallback_tick_sizes.items()}
            )
        self._store: Optional[_ContractsStore] = None

    # ------------------------------------------------------------------
    # Loading and parsing
    # ------------------------------------------------------------------
    def _candidate_master_files(self) -> List[str]:
        path = self.contract_master_path
        if not path:
            return []
        if os.path.isfile(path):
            return [path]
        if not os.path.isdir(path):
            return []
        out: List[str] = []
        for fn in sorted(os.listdir(path)):
            lfn = fn.lower()
            if lfn.endswith(".csv") or lfn.endswith(".csv.gz"):
                out.append(os.path.join(path, fn))
        return out

    @staticmethod
    def _find_col(cols: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
        lc = {c.lower().strip(): c for c in cols}
        for can in candidates:
            if can in lc:
                return lc[can]
        for c in cols:
            c2 = c.lower().strip()
            for can in candidates:
                if can in c2:
                    return c
        return None

    def _normalize_contract_df(self, raw: pd.DataFrame, source_name: str) -> pd.DataFrame:
        if raw is None or raw.empty:
            return pd.DataFrame(
                columns=["underlying", "symbol", "expiry_date", "lot_size", "tick_size", "source"]
            )

        cols = list(raw.columns)
        col_symbol = self._find_col(cols, ["underlying", "symbol", "name", "tradingsymbol", "instrument"])
        col_expiry = self._find_col(cols, ["expiry", "expiry_date", "expiration", "maturity"])
        col_lot = self._find_col(cols, ["lot_size", "lotsize", "market_lot", "qty", "contract_size"])
        col_tick = self._find_col(cols, ["tick_size", "tick", "min_tick"])

        if col_symbol is None or col_expiry is None or col_lot is None:
            return pd.DataFrame(
                columns=["underlying", "symbol", "expiry_date", "lot_size", "tick_size", "source"]
            )

        parsed = pd.DataFrame()
        parsed["symbol"] = raw[col_symbol].astype(str).str.upper().str.strip()
        parsed["underlying"] = parsed["symbol"].map(_normalize_symbol)
        parsed["expiry_date"] = pd.to_datetime(raw[col_expiry], errors="coerce").dt.date
        parsed["lot_size"] = pd.to_numeric(raw[col_lot], errors="coerce")
        if col_tick is not None:
            parsed["tick_size"] = pd.to_numeric(raw[col_tick], errors="coerce")
        else:
            parsed["tick_size"] = np.nan
        parsed["source"] = source_name

        parsed = parsed.dropna(subset=["underlying", "expiry_date", "lot_size"])
        if parsed.empty:
            return parsed

        parsed["lot_size"] = parsed["lot_size"].astype(int)
        parsed = parsed[parsed["lot_size"] > 0]
        return parsed

    def load_contracts(self, force_reload: bool = False) -> pd.DataFrame:
        if self._store is not None and not force_reload:
            return self._store.frame

        chunks: List[pd.DataFrame] = []

        # Local contract masters (CSV / CSV.GZ)
        for fp in self._candidate_master_files():
            try:
                df = pd.read_csv(fp, compression="infer", low_memory=False)
                norm_df = self._normalize_contract_df(df, source_name=os.path.basename(fp))
                if not norm_df.empty:
                    chunks.append(norm_df)
            except Exception:
                continue

        # Generic broker fallback: get_contract_master
        if self.broker_source is not None and hasattr(self.broker_source, "get_contract_master"):
            try:
                broker_raw = self.broker_source.get_contract_master()
                broker_df = pd.DataFrame(broker_raw)
                norm_df = self._normalize_contract_df(broker_df, source_name="broker_source")
                if not norm_df.empty:
                    chunks.append(norm_df)
            except Exception:
                pass

        if chunks:
            full = pd.concat(chunks, ignore_index=True)
            full = full.drop_duplicates(subset=["underlying", "expiry_date", "lot_size", "source"])
            full = full.sort_values(["underlying", "expiry_date"], ascending=[True, True]).reset_index(drop=True)
        else:
            full = pd.DataFrame(
                columns=["underlying", "symbol", "expiry_date", "lot_size", "tick_size", "source"]
            )

        self._store = _ContractsStore(frame=full, source="local+broker")
        return self._store.frame

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_expiry_dates(
        self,
        underlying: str = "NIFTY",
        as_of: Optional[DateLike] = None,
        n: int = 12,
        exchange_calendar=None,
    ) -> List[_dt.date]:
        as_of_date = _to_date(as_of or _dt.date.today())
        und = _normalize_symbol(underlying)

        df = self.load_contracts()
        out: List[_dt.date] = []

        if not df.empty:
            und_col = df["underlying"].astype(str)
            mask = und_col == und
            if not np.any(mask):
                mask = und_col.str.contains(und, na=False)
            expiries = sorted(set(d for d in df.loc[mask, "expiry_date"].tolist() if d and d >= as_of_date))
            if expiries:
                return expiries[:n]

        if self.broker_source is not None and hasattr(self.broker_source, "get_expiry_dates"):
            try:
                broker_expiries = self.broker_source.get_expiry_dates(underlying)
                parsed = sorted(
                    set(_to_date(x) for x in broker_expiries if _to_date(x) >= as_of_date)
                )
                if parsed:
                    return parsed[:n]
            except Exception:
                pass

        # Rule-based fallback: Tuesday weekly + last Tuesday monthly
        weekly = []
        start = as_of_date
        first = _next_weekday(start, self.weekly_expiry_weekday)
        if first < as_of_date:
            first += _dt.timedelta(days=7)
        d = first
        for _ in range(max(6, n * 2)):
            wd = _previous_trading_day(d, exchange_calendar=exchange_calendar)
            if wd >= as_of_date:
                weekly.append(wd)
            d += _dt.timedelta(days=7)

        monthly = []
        y, m = as_of_date.year, as_of_date.month
        for i in range(max(6, n)):
            mm = ((m - 1 + i) % 12) + 1
            yy = y + ((m - 1 + i) // 12)
            lm = _last_weekday_of_month(yy, mm, self.weekly_expiry_weekday)
            lm = _previous_trading_day(lm, exchange_calendar=exchange_calendar)
            if lm >= as_of_date:
                monthly.append(lm)

        out = sorted(set(weekly + monthly))
        return out[:n]

    def get_lot_size(self, symbol: str, expiry_date: Optional[DateLike] = None) -> int:
        sym = _normalize_symbol(symbol)
        exp_d = _to_date(expiry_date) if expiry_date is not None else None

        df = self.load_contracts()
        if not df.empty:
            und_col = df["underlying"].astype(str)
            mask = und_col == sym
            if not np.any(mask):
                mask = und_col.str.contains(sym, na=False)
            sub = df.loc[mask].copy()
            if not sub.empty:
                sub = sub.sort_values("expiry_date")
                if exp_d is not None:
                    exact = sub.loc[sub["expiry_date"] == exp_d]
                    if not exact.empty:
                        return int(exact.iloc[0]["lot_size"])
                    # nearest previous known contract
                    prev = sub.loc[sub["expiry_date"] <= exp_d]
                    if not prev.empty:
                        return int(prev.iloc[-1]["lot_size"])
                    # nearest future fallback
                    nxt = sub.loc[sub["expiry_date"] > exp_d]
                    if not nxt.empty:
                        return int(nxt.iloc[0]["lot_size"])
                else:
                    return int(sub.iloc[-1]["lot_size"])

        if self.broker_source is not None and hasattr(self.broker_source, "get_lot_size"):
            try:
                lot = self.broker_source.get_lot_size(symbol, expiry_date)
                if lot:
                    return int(lot)
            except Exception:
                pass

        for key, lot in self._fallback_lot_sizes.items():
            if sym.startswith(_normalize_symbol(key)):
                warnings.warn(
                    f"ContractSpecResolver: using fallback lot size {lot} for {symbol}",
                    RuntimeWarning,
                )
                return int(lot)

        warnings.warn(
            f"ContractSpecResolver: unknown lot size for {symbol}; using 1",
            RuntimeWarning,
        )
        return 1

    def get_tick_size(self, symbol: str) -> float:
        sym = _normalize_symbol(symbol)
        df = self.load_contracts()
        if not df.empty:
            mask = df["underlying"].astype(str).str.contains(sym, na=False)
            sub = df.loc[mask]
            if not sub.empty:
                ts = sub["tick_size"].dropna()
                if not ts.empty and np.isfinite(ts.iloc[-1]):
                    return float(ts.iloc[-1])
        for key, tick in self._fallback_tick_sizes.items():
            if sym.startswith(_normalize_symbol(key)):
                return float(tick)
        return 0.05

    @staticmethod
    def time_to_expiry_minutes(now_ts: DateLike, expiry_ts: DateLike, exchange_calendar=None) -> int:
        now_dt = _to_datetime(now_ts)
        exp_dt = _to_datetime(expiry_ts)
        if now_dt >= exp_dt:
            return 0

        if exchange_calendar is not None and hasattr(exchange_calendar, "trading_minutes_between"):
            try:
                val = exchange_calendar.trading_minutes_between(now_dt, exp_dt)
                return int(max(round(float(val)), 0))
            except Exception:
                pass

        open_time = getattr(exchange_calendar, "open_time", NSE_OPEN_TIME) if exchange_calendar else NSE_OPEN_TIME
        close_time = getattr(exchange_calendar, "close_time", NSE_CLOSE_TIME) if exchange_calendar else NSE_CLOSE_TIME

        minutes = 0
        cur_date = now_dt.date()
        end_date = exp_dt.date()

        while cur_date <= end_date:
            if _is_trading_day(cur_date, exchange_calendar=exchange_calendar):
                s0 = _dt.datetime.combine(cur_date, open_time)
                s1 = _dt.datetime.combine(cur_date, close_time)
                start = max(s0, now_dt)
                end = min(s1, exp_dt)
                if end > start:
                    minutes += int((end - start).total_seconds() // 60)
            cur_date += _dt.timedelta(days=1)
        return max(minutes, 0)


# Module-level default resolver and convenience wrappers
_DEFAULT_RESOLVER: Optional[ContractSpecResolver] = None


def get_default_resolver() -> ContractSpecResolver:
    global _DEFAULT_RESOLVER
    if _DEFAULT_RESOLVER is None:
        _DEFAULT_RESOLVER = ContractSpecResolver()
    return _DEFAULT_RESOLVER


def get_expiry_dates(underlying: str = "NIFTY", **kwargs) -> List[_dt.date]:
    resolver = kwargs.pop("resolver", None) or get_default_resolver()
    return resolver.get_expiry_dates(underlying=underlying, **kwargs)


def get_lot_size(symbol: str, expiry_date: Optional[DateLike] = None, **kwargs) -> int:
    resolver = kwargs.pop("resolver", None) or get_default_resolver()
    return resolver.get_lot_size(symbol=symbol, expiry_date=expiry_date)


def time_to_expiry_minutes(now_ts: DateLike, expiry_ts: DateLike, exchange_calendar=None, **kwargs) -> int:
    resolver = kwargs.pop("resolver", None) or get_default_resolver()
    return resolver.time_to_expiry_minutes(now_ts, expiry_ts, exchange_calendar=exchange_calendar)
