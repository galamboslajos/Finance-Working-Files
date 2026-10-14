"""Calendar cutoffs from fabricated metadata, without market-value inspection."""
from __future__ import annotations

import unittest

import pyarrow as pa

from dmv2.data import month_end_sessions, session_dates
from dmv2.universe import DataContractError


CALENDAR_SCHEMA = pa.schema([
    ("trading_date", pa.string()),
    ("calendar_name", pa.string()),
    ("is_trading_session", pa.bool_()),
])


def calendar(*dates, overrides=None):
    rows = [{"trading_date": day, "calendar_name": "TEST_US",
             "is_trading_session": True} for day in dates]
    if overrides is not None:
        rows.extend(overrides)
    return pa.Table.from_pylist(rows, schema=CALENDAR_SCHEMA)


def complete_test_calendar():
    # Sparse artificial metadata checks month selection, not exchange-day counts.
    return calendar("2019-12-31", "2020-01-02", "2020-01-30", "2020-01-31",
                    "2020-02-03", "2020-02-27", "2020-02-28", "2020-03-02",
                    "2020-03-31", "2020-04-01")


class SessionCalendarTests(unittest.TestCase):
    def test_sorted_source_sessions_only_without_fabricating_missing_dates(self):
        value = calendar("2020-01-31", "2020-01-02", overrides=[{
            "trading_date": "2020-01-03", "calendar_name": "TEST_US",
            "is_trading_session": False}])
        self.assertEqual(session_dates(value), ("2020-01-02", "2020-01-31"))

    def test_duplicate_source_sessions_fail(self):
        with self.assertRaises(DataContractError):
            session_dates(calendar("2020-01-31", "2020-01-31"))

    def test_malformed_calendar_dates_fail(self):
        for bad_date in ["2020-02-30", "2020-1-31", "2020-01-31T00:00:00", None]:
            with self.subTest(bad_date=bad_date):
                with self.assertRaises(DataContractError):
                    session_dates(calendar(bad_date))

    def test_unknown_session_state_fails(self):
        with self.assertRaises(DataContractError):
            session_dates(calendar(overrides=[{
                "trading_date": "2020-01-31", "calendar_name": "TEST_US",
                "is_trading_session": None}]))

    def test_null_blank_or_mixed_calendar_names_fail(self):
        for name in [None, "", " "]:
            with self.subTest(name=name):
                with self.assertRaises(DataContractError):
                    session_dates(calendar(overrides=[{
                        "trading_date": "2020-01-31", "calendar_name": name,
                        "is_trading_session": True}]))
        with self.assertRaises(DataContractError):
            session_dates(calendar("2020-01-30", overrides=[{
                "trading_date": "2020-01-31", "calendar_name": "OTHER_CALENDAR",
                "is_trading_session": True}]))

    def test_empty_or_missing_calendar_columns_fail(self):
        with self.assertRaises(DataContractError):
            session_dates(calendar())
        with self.assertRaises(DataContractError):
            session_dates(calendar("2020-01-31").drop(["is_trading_session"]))


class MonthEndCalendarTests(unittest.TestCase):
    def test_month_ends_use_last_source_session_and_not_calendar_day(self):
        self.assertEqual(month_end_sessions(complete_test_calendar(),
                                            start="2020-01-01", end="2020-03-31"),
                         ("2020-01-31", "2020-02-28", "2020-03-31"))

    def test_end_bound_inside_month_does_not_create_a_fake_month_end(self):
        self.assertEqual(month_end_sessions(complete_test_calendar(),
                                            start="2020-01-01", end="2020-02-15"),
                         ("2020-01-31",))
        self.assertEqual(month_end_sessions(complete_test_calendar(),
                                            start="2020-01-01", end="2020-01-15"),
                         ())

    def test_start_bound_filters_real_month_end_after_months_are_resolved(self):
        self.assertEqual(month_end_sessions(complete_test_calendar(),
                                            start="2020-01-15", end="2020-02-28"),
                         ("2020-01-31", "2020-02-28"))

    def test_missing_whole_requested_calendar_month_fails(self):
        value = calendar("2020-01-31", "2020-03-31", "2020-04-01")
        with self.assertRaises(DataContractError):
            month_end_sessions(value, start="2020-01-01", end="2020-03-31")

    def test_missing_requested_first_month_fails(self):
        value = calendar("2020-02-28", "2020-03-31", "2020-04-01")
        with self.assertRaises(DataContractError):
            month_end_sessions(value, start="2020-01-01", end="2020-03-31")

    def test_final_partial_calendar_month_is_not_certified(self):
        value = calendar("2020-01-31", "2020-02-28", "2020-03-16")
        with self.assertRaises(DataContractError):
            month_end_sessions(value, start="2020-01-01", end="2020-03-31")

    def test_final_calendar_month_requires_later_source_month_evidence(self):
        # A last row falling on the apparent month end still lacks next-month
        # evidence under this conservative finite-calendar contract.
        value = calendar("2020-01-31", "2020-02-28")
        with self.assertRaises(DataContractError):
            month_end_sessions(value, start="2020-01-01", end="2020-02-28")

    def test_invalid_and_reversed_bounds_fail(self):
        for start, end in [("2020-02-30", "2020-03-31"),
                           ("2020-1-01", "2020-03-31"),
                           ("2020-03-01", "2020-02-28")]:
            with self.subTest(start=start, end=end):
                with self.assertRaises(DataContractError):
                    month_end_sessions(complete_test_calendar(), start=start, end=end)


if __name__ == "__main__":
    unittest.main()
