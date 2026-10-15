"""Adversarial monthly contracts; tiny structural fixtures, never research data."""
import json
import unittest

import pyarrow as pa

from dmv2.monthly import build_month_windows, build_monthly_returns
from dmv2.identity_policy import PROVISIONAL_IDENTITY_POLICY, STRICT_IDENTITY_POLICY
from dmv2.universe import EXCLUDED_SCHEMA, UNIVERSE_SCHEMA, UniverseSnapshot


PRICE_SCHEMA = pa.schema([
    ("intrinio_security_id", pa.string()), ("trading_date", pa.string()),
    ("adj_close", pa.float64()), ("frequency", pa.string()),
    ("intraperiod", pa.bool_()), ("price_quality_status", pa.string()),
    ("volume", pa.float64()), ("adj_volume", pa.float64()),
])


def calendar():
    # Declared source sessions only; holidays must not become invented gaps.
    sessions = ["1999-12-30", "1999-12-31", "2000-01-03", "2000-01-04",
                "2000-01-31", "2000-02-01", "2000-02-29", "2000-03-01"]
    rows = [{"trading_date": day, "calendar_name": "TEST_ONLY",
             "is_trading_session": True} for day in sessions]
    rows.append({"trading_date": "2000-01-01", "calendar_name": "TEST_ONLY",
                 "is_trading_session": False})
    return pa.Table.from_pylist(rows)


def member(source="member-a", target="id-a", status="resolved", **changes):
    row = {
        "source_security_id": source, "unified_security_id": "unified-" + source,
        "intrinio_security_id": target, "identity_status": status,
        "identity_review_required": status == "needs_review", "resolution_status": "resolved",
        "indices": ["sp500"], "membership_tickers": ["STRUCTURAL_ONLY"],
        "membership_records_json": "[]",
    }
    row.update(changes)
    return row


def snapshot(day="1999-12-31", rows=None):
    rows = [member()] if rows is None else rows
    rows = [{**row, "as_of": day} for row in rows]
    counts = {}
    for row in rows:
        counts[row["identity_status"]] = counts.get(row["identity_status"], 0) + 1
    return UniverseSnapshot(
        pa.Table.from_pylist(rows, schema=UNIVERSE_SCHEMA),
        pa.Table.from_pylist([], schema=EXCLUDED_SCHEMA), day, "combined",
        {"knowledge_time_verified": False, "mode": "historical_reconstruction"},
        {"constituents": len(rows), "identity_status_counts": counts},
    )


def quote(day, value=10.0, target="id-a", **changes):
    row = {"intrinio_security_id": target, "trading_date": day, "adj_close": value,
           "frequency": "daily", "intraperiod": False, "price_quality_status": "accepted",
           "volume": 2.0, "adj_volume": 3.0}
    row.update(changes)
    return row


def prices(rows=None):
    rows = [quote("1999-12-31", 10), quote("2000-01-03", 12),
            quote("2000-01-04", 13), quote("2000-01-31", 20)] if rows is None else rows
    return pa.Table.from_pylist(rows, schema=PRICE_SCHEMA)


def run(rows=None, *, daily=None, snapshots=None, requested=("id-a",),
        end_month="2000-01", mode="historical_reconstruction",
        identity_policy=STRICT_IDENTITY_POLICY):
    return build_monthly_returns(
        prices(rows) if daily is None else daily, calendar(),
        [snapshot()] if snapshots is None else snapshots,
        start_month="2000-01", end_month=end_month,
        requested_security_ids=requested, mode=mode, identity_policy=identity_policy,
    )


class FinancialConversionTrap:
    """Forward Arrow metadata, fail if financial values enter Python first."""
    def __init__(self, table):
        self.table = table

    @property
    def column_names(self):
        return self.table.column_names

    @property
    def schema(self):
        return self.table.schema

    @property
    def num_rows(self):
        return self.table.num_rows

    def select(self, columns):
        selected = self.table.select(columns)
        return FinancialConversionTrap(selected) if "adj_close" in columns else selected

    def to_pylist(self):
        raise AssertionError("Financial Python conversion happened before metadata rejection")

    def to_pydict(self):
        raise AssertionError("Financial Python conversion happened before metadata rejection")

    def __getitem__(self, field):
        if field == "adj_close":
            raise AssertionError("Financial column accessed before metadata rejection")
        return self.table[field]


class MonthWindowTests(unittest.TestCase):
    def test_window_uses_previous_month_end_and_real_outcome_sessions(self):
        windows = build_month_windows(calendar(), start_month="2000-01", end_month="2000-02")
        self.assertEqual(len(windows), 2)
        self.assertEqual(windows[0].signal_date, "1999-12-31")
        self.assertEqual(windows[0].outcome_month, "2000-01")
        self.assertEqual(windows[0].outcome_end, "2000-01-31")
        self.assertEqual(windows[0].expected_outcome_sessions,
                         ("2000-01-03", "2000-01-04", "2000-01-31"))
        self.assertEqual(windows[1].signal_date, "2000-01-31")
        self.assertEqual(windows[1].outcome_end, "2000-02-29")

    def test_missing_anchor_missing_month_and_partial_final_calendar_fail(self):
        for excluded in (("1999-12-30", "1999-12-31"),
                         ("2000-02-01", "2000-02-29"), ("2000-03-01",)):
            rows = [row for row in calendar().to_pylist() if row["trading_date"] not in excluded]
            with self.subTest(excluded=excluded), self.assertRaises(ValueError):
                build_month_windows(pa.Table.from_pylist(rows), start_month="2000-01", end_month="2000-02")

    def test_malformed_reversed_and_protected_month_requests_fail(self):
        for start, end in (("200001", "2000-01"), ("2000-13", "2000-13"),
                           ("2000-02", "2000-01"), ("2024-01", "2024-01")):
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                build_month_windows(calendar(), start_month=start, end_month=end)

    def test_2024_calendar_metadata_can_certify_end_of_2023_without_financial_holdout_reads(self):
        dates = ["2023-11-30", "2023-12-01", "2023-12-29", "2024-01-02"]
        source_calendar = pa.Table.from_pylist([
            {"trading_date": day, "calendar_name": "TEST_ONLY", "is_trading_session": True}
            for day in dates])
        daily = prices([quote("2023-11-30", 10), quote("2023-12-01", 12),
                        quote("2023-12-29", 20)])
        result = build_monthly_returns(daily, source_calendar, [snapshot("2023-11-30")],
            start_month="2023-12", end_month="2023-12", requested_security_ids=["id-a"])
        row = result.table.to_pylist()[0]
        self.assertEqual(row["outcome_month"], "2023-12")
        self.assertEqual(row["return_status"], "ok")


class MonthlyReturnTests(unittest.TestCase):
    def test_explicit_provisional_admission_preserves_flag_and_source_status(self):
        cohorts = [snapshot(rows=[member(status="needs_review")])]
        strict = run(snapshots=cohorts).table.to_pylist()[0]
        self.assertEqual(strict["return_status"], "identity_unresolved")
        self.assertFalse(strict["identity_eligible"])
        panel = run(snapshots=cohorts, identity_policy=PROVISIONAL_IDENTITY_POLICY)
        row = panel.table.to_pylist()[0]
        self.assertEqual(row["return_status"], "ok")
        self.assertEqual(row["provider_adjusted_return"], 1.0)
        self.assertEqual(row["identity_status"], "needs_review")
        self.assertEqual(row["source_identity_status"], "needs_review")
        self.assertTrue(row["identity_review_required"])
        self.assertTrue(row["identity_eligible"])
        self.assertEqual(row["identity_eligibility_reason"], "provisional_review_flag_accepted")
        self.assertEqual(row["identity_eligibility_policy"], PROVISIONAL_IDENTITY_POLICY)
        self.assertEqual(panel.provenance["identity_policy"], PROVISIONAL_IDENTITY_POLICY)
        self.assertEqual(panel.summary["by_outcome_month"]["2000-01"]["provisional_review_flags_accepted"], 1)
        self.assertFalse(row["knowledge_time_verified"])
        self.assertFalse(row["accounting_certified"])
        self.assertFalse(row["daily_backtest_ready"])
        self.assertEqual(cohorts[0].table.to_pylist()[0]["identity_status"], "needs_review")

    def test_provisional_admission_still_blocks_full_cohort_provider_collision(self):
        cohorts = [snapshot(rows=[member(), member("member-b", "id-a", "needs_review")])]
        panel = run(snapshots=cohorts, identity_policy=PROVISIONAL_IDENTITY_POLICY)
        rows = panel.table.to_pylist()
        self.assertEqual(len(rows), 2)
        self.assertEqual({row["return_status"] for row in rows}, {"identity_unresolved"})
        self.assertTrue(all(row["identity_status"] == "target_collision" for row in rows))
        self.assertTrue(all(not row["identity_eligible"] for row in rows))
        self.assertTrue(all(row["provider_adjusted_return"] is None for row in rows))
        self.assertEqual({row["source_identity_status"] for row in rows}, {"resolved", "needs_review"})

    def test_provisional_admission_does_not_accept_unknown_or_failed_route_metadata(self):
        for changes in ({"identity_review_required": None}, {"resolution_status": None},
                        {"resolution_status": "pending"}, {"unified_security_id": None},
                        {"intrinio_security_id": None}):
            with self.subTest(changes=changes):
                panel = run(snapshots=[snapshot(rows=[member(status="needs_review", **changes)])],
                            identity_policy=PROVISIONAL_IDENTITY_POLICY)
                row = panel.table.to_pylist()[0]
                self.assertFalse(row["identity_eligible"])
                self.assertEqual(row["return_status"], "identity_unresolved")
                self.assertIsNone(row["provider_adjusted_return"])

    def test_unknown_identity_policy_fails_before_financial_conversion(self):
        with self.assertRaises(ValueError):
            run(daily=FinancialConversionTrap(prices()), identity_policy="accept_everything")

    def test_exact_endpoint_ratio_is_not_accounting_certification(self):
        daily = prices()
        original = daily.combine_chunks()
        panel = run(daily=daily)
        row = panel.table.to_pylist()[0]
        self.assertEqual(row["return_status"], "ok")
        self.assertEqual(row["provider_adjusted_return"], 1.0)
        self.assertEqual(row["signal_date"], "1999-12-31")
        self.assertEqual(row["outcome_month"], "2000-01")
        self.assertEqual(row["expected_daily_session_count"], 3)
        self.assertEqual(row["observed_daily_session_count"], 3)
        self.assertTrue(row["daily_quote_path_complete"])
        self.assertFalse(row["daily_backtest_ready"])
        self.assertFalse(row["knowledge_time_verified"])
        self.assertFalse(row["accounting_certified"])
        self.assertFalse(panel.provenance["knowledge_time_verified"])
        self.assertFalse(panel.provenance["accounting_certified"])
        self.assertTrue(daily.equals(original))

    def test_monthly_volume_sums_outcome_sessions_without_signal_anchor(self):
        row = run().table.to_pylist()[0]
        self.assertEqual(row["monthly_volume"], 6.0)
        self.assertEqual(row["monthly_adj_volume"], 9.0)

    def test_interior_gap_preserves_endpoint_ratio_but_blocks_complete_daily_path(self):
        rows = [row for row in prices().to_pylist() if row["trading_date"] != "2000-01-04"]
        row = run(rows).table.to_pylist()[0]
        self.assertEqual(row["return_status"], "ok")
        self.assertEqual(row["provider_adjusted_return"], 1.0)
        self.assertEqual(row["interior_gap_dates"], ["2000-01-04"])
        self.assertFalse(row["daily_quote_path_complete"])
        self.assertIsNone(row.get("monthly_volume"))

    def test_missing_first_outcome_session_is_an_interior_gap_between_price_endpoints(self):
        rows = [row for row in prices().to_pylist() if row["trading_date"] != "2000-01-03"]
        row = run(rows).table.to_pylist()[0]
        self.assertEqual(row["return_status"], "ok")
        self.assertEqual(row["provider_adjusted_return"], 1.0)
        self.assertIn("2000-01-03", row["missing_daily_dates"])
        self.assertIn("2000-01-03", row["interior_gap_dates"])

    def test_missing_exact_endpoints_never_use_last_or_first_observation(self):
        for day, expected in (("1999-12-31", "missing_start"), ("2000-01-31", "missing_end")):
            rows = [row for row in prices().to_pylist() if row["trading_date"] != day]
            with self.subTest(day=day):
                row = run(rows).table.to_pylist()[0]
                self.assertEqual(row["return_status"], expected)
                self.assertIsNone(row["provider_adjusted_return"])
                self.assertFalse(row["daily_quote_path_complete"])

    def test_removed_names_stay_and_new_entrants_begin_only_next_outcome_month(self):
        removed = member(membership_records_json=json.dumps([
            {"membership_end_date": "2000-01-14"}]))
        entrant = member("member-b", "id-b")
        start = snapshot(rows=[removed])
        next_month = snapshot("2000-01-31", [entrant])
        rows = prices().to_pylist() + [quote("2000-01-31", 30, "id-b"),
            quote("2000-02-01", 40, "id-b"), quote("2000-02-29", 60, "id-b")]
        # Quotes for the removed stock in February are deliberately absent.
        # January still uses its starting cohort; February uses the new cohort.
        panel = run(rows, snapshots=[start, next_month], requested=("id-a", "id-b"),
                    end_month="2000-02")
        result = {(row["outcome_month"], row["source_security_id"]): row
                  for row in panel.table.to_pylist()}
        self.assertEqual(set(result), {("2000-01", "member-a"), ("2000-02", "member-b")})
        self.assertEqual(result[("2000-01", "member-a")]["provider_adjusted_return"], 1.0)
        self.assertEqual(result[("2000-02", "member-b")]["provider_adjusted_return"], 1.0)

    def test_unrequested_review_and_missing_price_members_keep_original_denominator(self):
        members = [member(), member("member-b", "id-b"),
                   member("member-c", "id-c", "needs_review"),
                   member("member-d", "id-d")]
        panel = run(snapshots=[snapshot(rows=members)], requested=("id-a", "id-d"))
        rows = {row["source_security_id"]: row for row in panel.table.to_pylist()}
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows["member-a"]["return_status"], "ok")
        self.assertEqual(rows["member-b"]["return_status"], "not_requested")
        self.assertEqual(rows["member-c"]["return_status"], "identity_unresolved")
        for source, status in (("member-b", "not_requested"), ("member-c", "identity_unresolved")):
            self.assertEqual(rows[source]["daily_coverage_status"], status)
            self.assertIsNone(rows[source]["observed_daily_session_count"])
            self.assertIsNone(rows[source]["missing_daily_dates"])
            self.assertIsNone(rows[source]["interior_gap_dates"])
        self.assertIsNone(rows["member-d"]["provider_adjusted_return"])
        self.assertNotEqual(rows["member-d"]["return_status"], "ok")
        self.assertEqual(rows["member-d"]["observed_daily_session_count"], 0)
        self.assertEqual(rows["member-d"]["missing_daily_dates"],
                         ["2000-01-03", "2000-01-04", "2000-01-31"])

    def test_same_target_collision_is_quarantined_even_if_input_status_claims_resolved(self):
        members = [member(), member("member-b", "id-a")]
        panel = run(snapshots=[snapshot(rows=members)])
        rows = panel.table.to_pylist()
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["identity_status"] == "target_collision" for row in rows))
        self.assertTrue(all(row["return_status"] == "identity_unresolved" for row in rows))
        self.assertTrue(all(row["provider_adjusted_return"] is None for row in rows))

    def test_claimed_resolved_identity_cannot_override_supplied_review_or_resolution_flags(self):
        for changes in ({"identity_review_required": True}, {"resolution_status": "unresolved"}):
            cohort = snapshot(rows=[member(**changes)])
            with self.subTest(changes=changes):
                row = run(snapshots=[cohort]).table.to_pylist()[0]
                self.assertEqual(row["source_identity_status"], "resolved")
                self.assertEqual(row["identity_status"], "needs_review")
                self.assertEqual(row["return_status"], "identity_unresolved")
                self.assertIsNone(row["provider_adjusted_return"])
                self.assertIsNone(row["missing_daily_dates"])

    def test_invalid_endpoint_prices_are_not_zero_returns(self):
        for value in (None, float("nan"), float("inf"), 0.0, -1.0):
            for day in ("1999-12-31", "2000-01-31"):
                rows = prices().to_pylist()
                for row in rows:
                    if row["trading_date"] == day:
                        row["adj_close"] = value
                with self.subTest(value=value, day=day):
                    result = run(rows).table.to_pylist()[0]
                    self.assertEqual(result["return_status"], "invalid_endpoint")
                    self.assertIsNone(result["provider_adjusted_return"])

    def test_finite_endpoints_cannot_emit_nonfinite_ratio_or_claim_complete_path(self):
        rows = prices().to_pylist()
        rows[0]["adj_close"] = 1e-308
        rows[-1]["adj_close"] = 1e308
        row = run(rows).table.to_pylist()[0]
        self.assertEqual(row["return_status"], "invalid_endpoint")
        self.assertIsNone(row["provider_adjusted_return"])
        self.assertFalse(row["daily_quote_path_complete"])

    def test_frequency_intraperiod_and_quality_defects_never_become_accepted_endpoints(self):
        for field, value in (("frequency", "monthly"), ("intraperiod", True),
                             ("price_quality_status", "review")):
            rows = prices().to_pylist()
            rows[-1][field] = value
            with self.subTest(field=field):
                row = run(rows).table.to_pylist()[0]
                self.assertEqual(row["return_status"], "source_quality_flag")
                self.assertIsNone(row["provider_adjusted_return"])

    def test_explicit_positive_quality_states_accept_completed_daily_bars(self):
        for status in ("accepted", "ok"):
            rows = prices().to_pylist()
            for row in rows:
                row["price_quality_status"] = status
            with self.subTest(status=status):
                panel = run(rows)
                result = panel.table.to_pylist()[0]
                self.assertEqual(result["return_status"], "ok")
                self.assertEqual(result["provider_adjusted_return"], 1.0)
                self.assertTrue(result["daily_quote_path_complete"])
                self.assertEqual(result["start_price_quality_status"], status)
                self.assertEqual(result["end_price_quality_status"], status)
                self.assertEqual(panel.provenance["accepted_price_quality_states"], ["accepted", "ok"])

    def test_unknown_null_or_rejected_quality_cannot_be_accepted_by_default(self):
        for status in ("unknown", None, "rejected", ""):
            rows = prices().to_pylist()
            rows[-1]["price_quality_status"] = status
            with self.subTest(status=status):
                result = run(rows).table.to_pylist()[0]
                self.assertEqual(result["return_status"], "source_quality_flag")
                self.assertIsNone(result["provider_adjusted_return"])
                self.assertFalse(result["daily_quote_path_complete"])

    def test_invalid_interior_quote_preserves_endpoints_and_audits_daily_defect(self):
        rows = prices().to_pylist()
        rows[2]["adj_close"] = float("nan")
        row = run(rows).table.to_pylist()[0]
        self.assertEqual(row["return_status"], "ok")
        self.assertEqual(row["provider_adjusted_return"], 1.0)
        self.assertIn("2000-01-04", row["invalid_daily_dates"])
        self.assertFalse(row["daily_quote_path_complete"])

    def test_bad_volume_cannot_create_partial_sum_or_invalidate_independent_price_ratio(self):
        for value in (None, float("nan"), -1.0):
            rows = prices().to_pylist()
            rows[2]["volume"] = value
            with self.subTest(value=value):
                row = run(rows).table.to_pylist()[0]
                self.assertEqual(row["return_status"], "ok")
                self.assertEqual(row["provider_adjusted_return"], 1.0)
                self.assertIsNone(row["monthly_volume"])
                self.assertEqual(row["monthly_adj_volume"], 9.0)

    def test_overflowing_volume_aggregate_is_null_and_recorded_as_issue(self):
        rows = prices().to_pylist()
        for row in rows:
            row["volume"] = 1e308
        row = run(rows).table.to_pylist()[0]
        self.assertEqual(row["return_status"], "ok")
        self.assertIsNone(row["monthly_volume"])
        self.assertIn("monthly:volume:nonfinite_aggregate", row["optional_value_issues"])

    def test_duplicate_source_or_snapshot_dates_are_not_silently_deduplicated(self):
        for snapshots in ([snapshot(rows=[member(), member()])], [snapshot(), snapshot()]):
            with self.subTest(snapshots=len(snapshots)), self.assertRaises(ValueError):
                run(snapshots=snapshots)

    def test_snapshot_at_outcome_end_cannot_replace_signal_time_snapshot(self):
        with self.assertRaises(ValueError):
            run(snapshots=[snapshot("2000-01-31")])

    def test_empty_requested_sample_and_missing_volume_leave_cohort_visible(self):
        row = run([], requested=()).table.to_pylist()[0]
        self.assertEqual(row["return_status"], "not_requested")
        self.assertIsNone(row["provider_adjusted_return"])
        daily = prices().drop(["volume", "adj_volume"])
        row = run(daily=daily).table.to_pylist()[0]
        self.assertEqual(row["return_status"], "ok")
        self.assertIsNone(row.get("monthly_volume"))

    def test_knowledge_time_is_not_claimed_from_reconstructed_sources(self):
        with self.assertRaises(ValueError):
            run(mode="knowledge_time")


class MetadataFirstGuardTests(unittest.TestCase):
    def test_protected_out_of_request_and_non_session_quotes_reject_before_financial_conversion(self):
        cases = [
            (quote("2024-01-02"), "(?i)protected|2024"),
            (quote("2000-01-03", target="id-z"), "(?i)request|outside"),
            (quote("2000-01-01"), "(?i)calendar|session|request|outside"),
            (quote("2000-02-01"), "(?i)window|range|request|outside"),
            (quote("2000-02-30"), "(?i)date"),
            (quote("2000-01-03", target=""), "(?i)id|security"),
        ]
        for value, message in cases:
            daily = FinancialConversionTrap(prices([value]))
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, message):
                run(daily=daily)

    def test_duplicate_quotes_reject_before_financial_conversion(self):
        daily = FinancialConversionTrap(prices([quote("1999-12-31"), quote("1999-12-31")]))
        with self.assertRaisesRegex(ValueError, "(?i)duplicate"):
            run(daily=daily)


if __name__ == "__main__":
    unittest.main()
