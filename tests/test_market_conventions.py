"""
test_market_conventions.py — Unit Tests for NSE Market Rules
============================================================

Verifies:
- Accurate time-to-expiry calculation (intraday, annualization)
- Index identification logic
- Trading hours constants
"""

import math
import datetime
import pytest
from market_conventions import (
    time_to_expiry,
    is_index_option,
    EuropeanOptionRouter,
    NSE_OPEN_TIME,
    NSE_CLOSE_TIME
)

# ── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def expiry_date():
    """A standard expiry date: Thursday, 2025-10-30"""
    return datetime.date(2025, 10, 30)

@pytest.fixture
def expiry_dt(expiry_date):
    """Expiry datetime: 2025-10-30 15:30:00"""
    return datetime.datetime.combine(expiry_date, NSE_CLOSE_TIME)


# ── Time to Expiry Tests ────────────────────────────────────────────

def test_time_to_expiry_zero(expiry_dt):
    """At or after expiry, T should be 0."""
    # Exact expiry time
    assert time_to_expiry(expiry_dt, expiry_dt) == 0.0
    
    # 1 minute after
    after = expiry_dt + datetime.timedelta(minutes=1)
    assert time_to_expiry(after, expiry_dt) == 0.0


def test_time_to_expiry_one_year():
    """Exactly 365 days before expiry should be T=1.0."""
    expiry = datetime.datetime(2025, 12, 31, 15, 30)
    now = datetime.datetime(2024, 12, 31, 15, 30)
    
    t = time_to_expiry(now, expiry)
    assert t == pytest.approx(1.0, abs=1e-5)


def test_time_to_expiry_intraday():
    """
    Test intraday decay.
    09:15 to 15:30 is 6h 15m = 6.25 hours.
    T at 09:15 on expiry day should be 6.25 / (24*365).
    """
    expiry = datetime.datetime(2025, 10, 30, 15, 30)
    now = datetime.datetime(2025, 10, 30, 9, 15)
    
    # Expected T in years
    hours_remaining = 6.25
    expected_t = hours_remaining / (24.0 * 365.0)
    
    t = time_to_expiry(now, expiry)
    assert t == pytest.approx(expected_t, abs=1e-7)


def test_time_to_expiry_string_input():
    """Should handle string inputs robustly."""
    now_str = "2025-10-30 09:15:00"
    expiry_str = "2025-10-30 15:30:00"
    
    t = time_to_expiry(now_str, expiry_str)
    assert t > 0
    assert t < 1.0/365.0  # Less than a day


# ── Index Identification Tests ──────────────────────────────────────

@pytest.mark.parametrize("symbol, expected", [
    ("NIFTY", True),
    ("BANKNIFTY", True),
    ("FINNIFTY", True),
    ("MIDCPNIFTY", True),
    ("NIFTY 50", True),
    ("RELIANCE", False),
    ("TCS", False),
    ("INFY", False),
    ("", False),
    (None, False),
])
def test_is_index_option(symbol, expected):
    assert is_index_option(symbol) is expected


# ── Option Style Tests ──────────────────────────────────────────────

def test_european_option_router():
    """All NSE F&O options are European style."""
    assert EuropeanOptionRouter.is_european("NIFTY") is True
    assert EuropeanOptionRouter.is_european("RELIANCE") is True
