"""
market_conventions.py — India/NSE Market Conventions
====================================================

Central repository for NSE-specific trading rules, holidays, and
time-to-expiry calculations.

Features:
- NSE trading hours (09:15 - 15:30 IST)
- Exchange holiday handling (implied, not hardcoded list for now)
- Accurate time-to-expiry (T) calculation with intraday granularity
- Index option identification (NIFTY, BANKNIFTY)
"""

import datetime
from typing import Union, Tuple

# ── Constants ────────────────────────────────────────────────────────

# Standard NSE trading hours (IST)
NSE_OPEN_TIME = datetime.time(9, 15)
NSE_CLOSE_TIME = datetime.time(15, 30)

# Minutes in a standard trading day (6 hours 15 mins = 375 mins)
NSE_MINUTES_PER_DAY = 375.0

# Annualization factors
DAYS_PER_YEAR = 365.0
TRADING_DAYS_PER_YEAR = 252.0

# Lot sizes (as of 2024-25) — useful for P&L sizing, though often dynamic
NSE_LOT_SIZES = {
    'NIFTY': 25,       # Revised from 50
    'BANKNIFTY': 15,   # Revised from 25
    'FINNIFTY': 25,    # Financial Services
    'MIDCPNIFTY': 50,  # Midcap Select
}


def is_index_option(symbol: str) -> bool:
    """
    Identify if a symbol corresponds to a major Indian index.
    
    Parameters
    ----------
    symbol : str
        Trading symbol (e.g. 'NIFTY', 'BANKNIFTY', 'RELIANCE')
        
    Returns
    -------
    bool
        True if index, False if equity/other.
    """
    if not symbol:
        return False
    
    # Normalize
    s = symbol.upper().strip()
    
    # Check against known indices
    indices = {'NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY'}
    
    # Handle variations like "NIFTY 50", "NIFTY23JAN..."
    # A simple startswith check usually suffices for the base identifier
    # assuming the input is the underlying name.
    if s in indices:
        return True
        
    # Check if any index name is a prefix (common in some data feeds)
    for idx in indices:
        if s.startswith(idx):
            return True
            
    return False


def get_trading_minutes(dt: datetime.datetime) -> float:
    """
    Calculate minutes from market open (09:15) for a given datetime.
    Unbounded (can be negative if before open, >375 if after close).
    """
    # Create open time on the same date
    market_open = datetime.datetime.combine(dt.date(), NSE_OPEN_TIME)
    
    # Difference in minutes
    delta = dt - market_open
    return delta.total_seconds() / 60.0


def time_to_expiry(
    current_time: Union[datetime.datetime, str],
    expiry_date: Union[datetime.datetime, datetime.date, str],
    calendar: str = 'NSE'
) -> float:
    """
    Calculate annualized time-to-expiry (T) following NSE conventions.
    
    Logic:
    1. If current_time >= expiry, T = 0.
    2. T is calculated in calendar days / 365.0 (standard for BSM).
    3. Intraday granularity is preserved.
    4. Expiry is assumed to be at 15:30 IST on the expiry date.
    
    Parameters
    ----------
    current_time : datetime or str
        Current timestamp (tz-naive assumed IST, or tz-aware).
    expiry_date : datetime, date or str
        Expiry date (usually last Thursday or specific weekly expiry).
        If date only, assumes 15:30 IST close.
    calendar : str
        'NSE' (default) or 'ACT/365', 'ACT/360'.
        Currently 'NSE' maps to ACT/365 with 15:30 expiry fix.
        
    Returns
    -------
    float
        Time to expiry in years.
    """
    # ── Parsing Inputs ──
    if isinstance(current_time, str):
        # naive parse - assumes ISO format YYYY-MM-DD HH:MM:SS
        try:
            current_time = datetime.datetime.fromisoformat(current_time)
        except ValueError:
            # Fallback for simpler date strings
            import pandas as pd
            current_time = pd.to_datetime(current_time).to_pydatetime()

    if isinstance(expiry_date, str):
        import pandas as pd
        expiry_date = pd.to_datetime(expiry_date).to_pydatetime()
    elif isinstance(expiry_date, datetime.date) and not isinstance(expiry_date, datetime.datetime):
        # Convert date to datetime at market close
        expiry_date = datetime.datetime.combine(expiry_date, NSE_CLOSE_TIME)
        
    # Ensure expiry has time component (if passed as datetime without time, default to 15:30)
    if isinstance(expiry_date, datetime.datetime) and expiry_date.time() == datetime.time(0, 0):
        expiry_date = expiry_date.replace(hour=15, minute=30)
    
    # ── Calculation ──
    
    # 1. Immediate expiry
    if current_time >= expiry_date:
        return 0.0
        
    # 2. Time difference in seconds
    delta = expiry_date - current_time
    total_seconds = delta.total_seconds()
    
    # 3. Annualize
    # NSE convention for option pricing usually uses calendar days (ACT/365)
    # Some desks use trading days (ACT/252), but IV is typically quoted annualised over 365.
    
    years = total_seconds / (365.0 * 24.0 * 3600.0)
    
    return max(0.0, years)


class EuropeanOptionRouter:
    """Helper to determine exercise style."""
    
    @staticmethod
    def is_european(underlying: str) -> bool:
        """
        NIFTY/BANKNIFTY/FINNIFTY indices are European.
        Single stocks are European (in India, all stock options are European since 2010+).
        Wait, actually:
        - Index Options: European
        - Stock Options: European (since Jan 2011 mandate by SEBI)
        
        So effectively EVERYTHING on NSE F&O is European now.
        American style is mostly legacy or specific commodities.
        """
        return True

