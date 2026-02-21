import datetime as dt

import pandas as pd

from nse_specs import ContractSpecResolver, time_to_expiry_minutes


class ToyCalendar:
    open_time = dt.time(9, 15)
    close_time = dt.time(15, 30)

    def __init__(self, holidays=None):
        self.holidays = set(holidays or [])

    def is_trading_day(self, date_obj):
        return date_obj.weekday() < 5 and date_obj not in self.holidays


def test_contract_spec_resolver_lot_and_expiry(tmp_path):
    csv_path = tmp_path / "contract_master.csv"
    pd.DataFrame(
        [
            {"underlying": "NIFTY", "expiry_date": "2026-02-24", "lot_size": 25},
            {"underlying": "NIFTY", "expiry_date": "2026-03-03", "lot_size": 25},
            {"underlying": "BANKNIFTY", "expiry_date": "2026-02-24", "lot_size": 15},
        ]
    ).to_csv(csv_path, index=False)

    resolver = ContractSpecResolver(contract_master_path=str(csv_path))
    expiries = resolver.get_expiry_dates("NIFTY", as_of=dt.date(2026, 2, 20), n=5)
    assert expiries[0] == dt.date(2026, 2, 24)
    assert resolver.get_lot_size("NIFTY", dt.date(2026, 2, 24)) == 25
    assert resolver.get_lot_size("BANKNIFTY", dt.date(2026, 2, 24)) == 15


def test_tuesday_expiry_rule_with_holiday_adjustment():
    cal = ToyCalendar(holidays={dt.date(2026, 2, 24)})  # Tuesday holiday
    resolver = ContractSpecResolver(contract_master_path="")
    expiries = resolver.get_expiry_dates("NIFTY", as_of=dt.date(2026, 2, 23), n=3, exchange_calendar=cal)
    # Holiday-adjusted weekly expiry should roll back to previous trading day (Monday).
    assert expiries[0] == dt.date(2026, 2, 23)


def test_time_to_expiry_minutes_holiday_weekend():
    cal = ToyCalendar(holidays={dt.date(2026, 2, 24)})  # Tuesday holiday

    now_ts = dt.datetime(2026, 2, 20, 14, 30)      # Friday
    expiry_ts = dt.datetime(2026, 2, 25, 15, 30)   # Wednesday

    mins = time_to_expiry_minutes(now_ts, expiry_ts, exchange_calendar=cal)
    # Friday: 60m (14:30 -> 15:30)
    # Monday: 375m
    # Tuesday holiday: 0m
    # Wednesday: 375m
    assert mins == 810

