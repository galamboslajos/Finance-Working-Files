"""Independent calendar checks, using date-only structural source calendars."""
from datetime import date, timedelta
from unittest.mock import patch
import unittest

import pyarrow as pa

from dmv2.exchange_sessions import (
    CALENDAR_POLICY, EXCHANGE_CALENDARS_VERSION, reconcile_exchange_calendar,
)
from dmv2.monthly import build_month_windows
from dmv2.universe import DataContractError


def weekdays(start, end):
    rows = []
    day, last = date.fromisoformat(start), date.fromisoformat(end)
    while day <= last:
        if day.weekday() < 5:
            rows.append({"trading_date": day.isoformat(), "calendar_name": "XNYS",
                         "is_trading_session": True})
        day += timedelta(days=1)
    return pa.Table.from_pylist(rows)


class IndependentCalendarTests(unittest.TestCase):
    def test_2000_exchange_holidays_are_not_trading_session_price_gaps(self):
        source = weekdays("1999-12-01", "2000-03-31")
        result = reconcile_exchange_calendar(source, start_month="2000-01", end_month="2000-02")
        self.assertIs(result.source_calendar, source)
        days = result.table["trading_date"].to_pylist()
        self.assertNotIn("2000-01-17", days)
        self.assertNotIn("2000-02-21", days)
        self.assertEqual(sum(day.startswith("2000-01") for day in days), 20)
        self.assertEqual(sum(day.startswith("2000-02") for day in days), 20)
        self.assertTrue({"2000-01-17", "2000-02-21"}
                        <= set(result.provenance["source_only_dates"]))
        self.assertEqual(result.provenance["scope"]["calendar_start"], "1999-12-01")
        self.assertEqual(result.provenance["scope"]["calendar_end"], "2000-03-31")
        self.assertEqual(result.provenance["policy"], CALENDAR_POLICY)
        self.assertEqual(result.provenance["package_version"], EXCHANGE_CALENDARS_VERSION)
        self.assertTrue(result.provenance["selected_sessions_sha256"].startswith("sha256:"))
        self.assertTrue(result.provenance["calendar_code_sha256"].startswith("sha256:"))
        self.assertIn("exchange_calendar_xnys.py", result.provenance["calendar_source_file_sha256"])

    def test_good_friday_changes_true_march2018_month_end(self):
        source = weekdays("2018-02-01", "2018-04-30")
        result = reconcile_exchange_calendar(source, start_month="2018-03", end_month="2018-03")
        window, = build_month_windows(result.table, start_month="2018-03", end_month="2018-03")
        self.assertEqual(window.signal_date, "2018-02-28")
        self.assertEqual(window.outcome_end, "2018-03-29")
        self.assertNotIn("2018-03-30", window.expected_outcome_sessions)
        self.assertIn("2018-03-30", result.provenance["source_only_dates"])
        self.assertIn("2018-03-30", source["trading_date"].to_pylist())

    def test_september2001_extraordinary_closures(self):
        result = reconcile_exchange_calendar(
            weekdays("2001-08-01", "2001-10-31"), start_month="2001-09", end_month="2001-09")
        days = result.table["trading_date"].to_pylist()
        closed = {"2001-09-11", "2001-09-12", "2001-09-13", "2001-09-14"}
        self.assertFalse(closed & set(days))
        self.assertTrue(closed <= set(result.provenance["source_only_dates"]))
        self.assertIn("2001-09-17", days)

    def test_early_close_metadata_is_preserved(self):
        result = reconcile_exchange_calendar(
            weekdays("1999-12-01", "2000-03-31"), start_month="2000-01", end_month="2000-02")
        row = next(r for r in result.table.to_pylist() if r["trading_date"] == "1999-12-31")
        self.assertIs(row["is_early_close"], True)
        self.assertEqual(row["session_open_utc"], "1999-12-31T14:30:00+00:00")
        self.assertEqual(row["session_close_utc"], "1999-12-31T18:00:00+00:00")

    def test_independent_only_sessions_are_visible_and_not_removed_by_source(self):
        source = weekdays("1999-12-01", "2000-03-31")
        rows = [r for r in source.to_pylist() if r["trading_date"] != "2000-01-31"]
        result = reconcile_exchange_calendar(pa.Table.from_pylist(rows),
                                             start_month="2000-01", end_month="2000-02")
        self.assertIn("2000-01-31", result.table["trading_date"].to_pylist())
        self.assertIn("2000-01-31", result.provenance["independent_only_dates"])

    def test_identical_source_and_code_produce_identical_metadata_hashes(self):
        source = weekdays("1999-12-01", "2000-03-31")
        a = reconcile_exchange_calendar(source, start_month="2000-01", end_month="2000-02")
        b = reconcile_exchange_calendar(source, start_month="2000-01", end_month="2000-02")
        self.assertTrue(a.table.equals(b.table))
        self.assertEqual(a.provenance, b.provenance)

    def test_scope_and_source_validation_precede_dependency_use(self):
        source = weekdays("1999-12-01", "2000-03-31")
        bad_source = pa.Table.from_pylist([
            {"trading_date": "2000-01-31", "calendar_name": "XNAS", "is_trading_session": True}])
        cases = [(source, "2000-1", "2000-02"),
                 (source, "2000-01", "2000-04"),
                 (source, "2000-03", "2000-02"),
                 (source, "0001-01", "0001-01"),
                 (source, "9999-12", "9999-12"),
                 (bad_source, "2000-01", "2000-02")]
        for table, first, last in cases:
            with self.subTest(first=first, last=last, name=table["calendar_name"].to_pylist()[0]):
                with patch("dmv2.exchange_sessions.version") as package:
                    with self.assertRaises(DataContractError):
                        reconcile_exchange_calendar(table, start_month=first, end_month=last)
                    package.assert_not_called()

    def test_wrong_package_version_fails_instead_of_silently_changing_rules(self):
        with patch("dmv2.exchange_sessions.version", return_value="4.13.3"):
            with self.assertRaisesRegex(DataContractError, "differs from the reviewed pin"):
                reconcile_exchange_calendar(weekdays("1999-12-01", "2000-03-31"),
                                             start_month="2000-01", end_month="2000-02")


if __name__ == "__main__":
    unittest.main()
