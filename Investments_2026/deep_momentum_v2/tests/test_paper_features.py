"""Synthetic structural tests for causal monthly inputs; no research values."""
import calendar
import math
import unittest

import pandas as pd
import pyarrow as pa

from dmv2.paper_features import compute_paper_features, paper_feature_specs
from dmv2.identity_policy import PROVISIONAL_IDENTITY_POLICY, STRICT_IDENTITY_POLICY
from dmv2.universe import DataContractError, EXCLUDED_SCHEMA, UNIVERSE_SCHEMA, UniverseSnapshot


MONTHS = [(1998 + offset // 12, offset % 12 + 1) for offset in range(39)]
SESSIONS = tuple(f"{year:04d}-{month:02d}-{calendar.monthrange(year, month)[1]:02d}"
                 for year, month in MONTHS)
SIGNAL = "2000-01-31"
REQUIRED = ("intrinio_security_id", "trading_date", "frequency", "intraperiod",
            "price_quality_status", "adj_close")


def member(source, target, *, status="resolved", review=False, resolution="resolved"):
    return {"source_security_id": source, "intrinio_security_id": target,
            "identity_status": status, "identity_review_required": review,
            "resolution_status": resolution, "unified_security_id": "u-" + source,
            "indices": ["sp500"], "membership_tickers": [source],
            "membership_records_json": "[]"}


def cohort(signal=SIGNAL, members=None):
    rows = [member("a", "id-a"), member("b", "id-b")] if members is None else members
    return UniverseSnapshot(
        pa.Table.from_pylist([{**row, "as_of": signal} for row in rows], schema=UNIVERSE_SCHEMA),
        pa.Table.from_pylist([], schema=EXCLUDED_SCHEMA), signal, "combined",
        {"mode": "historical_reconstruction", "knowledge_time_verified": False}, {},
    )


def quote(day, value, target="id-a", **changes):
    return {"intrinio_security_id": target, "trading_date": day, "frequency": "daily",
            "intraperiod": False, "price_quality_status": "accepted", "adj_close": value,
            **changes}


def prices(max_day=SIGNAL, targets=("id-a", "id-b")):
    rows = []
    for target in targets:
        growth = {"id-a": 1.01, "id-b": 1.02, "id-nonmember": 1.50}[target]
        for offset, day in enumerate(SESSIONS):
            if day <= max_day:
                rows.append(quote(day, 10.0 * growth ** offset, target))
    return pa.Table.from_pylist(rows)


def run(daily=None, *, snapshots=None, requested=("id-a", "id-b"), sessions=SESSIONS,
        policy="complete_cohort", identity_policy=STRICT_IDENTITY_POLICY):
    return compute_paper_features(prices() if daily is None else daily, sessions,
                                  signal_cohorts=[cohort()] if snapshots is None else snapshots,
                                  requested_security_ids=requested, normalization_policy=policy,
                                  identity_policy=identity_policy)


class FinancialConversionTrap:
    def __init__(self, table):
        self.table = table

    @property
    def column_names(self):
        return self.table.column_names

    def select(self, columns):
        result = self.table.select(columns)
        return FinancialConversionTrap(result) if "adj_close" in columns else result

    def to_pylist(self):
        raise AssertionError("Financial conversion occurred before metadata rejection")


class PaperFeatureTests(unittest.TestCase):
    def test_provisional_flag_admission_preserves_original_full_cohort_and_no_subset_normalization(self):
        members = [member("a", "id-a"), member("b", "id-b", status="needs_review", review=True),
                   member("missing", None, status="unmapped_intrinio", review=None)]
        snapshots = [cohort(members=members)]
        strict = run(snapshots=snapshots).table.set_index("source_security_id")
        self.assertEqual(strict.at["b", "MOM1_raw_status"], "identity_unresolved")
        result = run(snapshots=snapshots, identity_policy=PROVISIONAL_IDENTITY_POLICY)
        rows = result.table.set_index("source_security_id")
        self.assertEqual(set(rows.index), {"a", "b", "missing"})
        self.assertEqual(rows.at["b", "identity_status"], "needs_review")
        self.assertEqual(rows.at["b", "source_identity_status"], "needs_review")
        self.assertTrue(rows.at["b", "identity_review_required"])
        self.assertTrue(rows.at["b", "identity_eligible"])
        self.assertEqual(rows.at["b", "identity_eligibility_reason"], "provisional_review_flag_accepted")
        for horizon in (1, 3, 6, 9, 12):
            self.assertEqual(rows.at["b", f"MOM{horizon}_raw_status"], "ok")
            self.assertTrue(pd.notna(rows.at["b", f"MOM{horizon}_raw"]))
        self.assertFalse(rows.at["missing", "identity_eligible"])
        self.assertTrue(result.table[[spec["name"] for spec in result.specs]].isna().all().all())
        self.assertEqual(result.coverage.original_cohort_count.tolist(), [3] * 5)
        self.assertEqual(result.coverage.raw_momentum_available_count.tolist(), [2] * 5)
        self.assertEqual(result.coverage.normalization_status.tolist(), ["incomplete_cohort"] * 5)
        self.assertEqual(result.provenance["identity_policy"], PROVISIONAL_IDENTITY_POLICY)
        self.assertFalse(result.provenance["knowledge_time_verified"])
        self.assertFalse(result.table.knowledge_time_verified.any())

    def test_provisional_flag_admission_still_quarantines_all_same_date_colliding_members(self):
        members = [member("a", "id-a"), member("b", "id-a", status="needs_review", review=True)]
        result = run(snapshots=[cohort(members=members)], identity_policy=PROVISIONAL_IDENTITY_POLICY)
        self.assertEqual(len(result.table), 2)
        self.assertTrue((result.table.identity_status == "target_collision").all())
        self.assertFalse(result.table.identity_eligible.any())
        self.assertTrue((result.table.MOM1_raw_status == "identity_unresolved").all())
        self.assertTrue(result.table.MOM1_raw.isna().all())
        self.assertEqual(set(result.table.source_identity_status), {"resolved", "needs_review"})

    def test_provisional_flag_admission_blocks_unknown_flag_and_failed_resolution(self):
        for review, resolution in ((None, "resolved"), (True, "pending"), (True, None)):
            with self.subTest(review=review, resolution=resolution):
                members = [member("a", "id-a"), member("b", "id-b", status="needs_review",
                            review=review, resolution=resolution)]
                rows = run(snapshots=[cohort(members=members)],
                           identity_policy=PROVISIONAL_IDENTITY_POLICY).table.set_index("source_security_id")
                self.assertFalse(rows.at["b", "identity_eligible"])
                self.assertEqual(rows.at["b", "MOM1_raw_status"], "identity_unresolved")

    def test_unknown_identity_policy_fails_before_financial_conversion(self):
        with self.assertRaises(ValueError):
            run(FinancialConversionTrap(prices()), identity_policy="accept_everything")

    def test_registers_exact_fifteen_and_records_adaptation(self):
        specs = paper_feature_specs()
        self.assertEqual(len(specs), 15)
        self.assertEqual({spec["name"] for spec in specs},
                         {f"{prefix}{h}" for prefix in ("zMOM", "meanMOM", "stdMOM") for h in (1, 3, 6, 9, 12)})
        self.assertTrue(all(spec["role"] == "paper" for spec in specs))
        self.assertFalse(any("size" in spec["name"] for spec in specs))
        result = run()
        self.assertEqual(result.provenance["standard_deviation_ddof"], 1)
        self.assertFalse(result.provenance["knowledge_time_verified"])
        self.assertFalse(result.table["knowledge_time_verified"].any())
        self.assertFalse(result.table["accounting_certified"].any())

    def test_exact_monthly_windows_include_one_and_skip_current_for_long(self):
        rows = run().table.set_index("source_security_id")
        for source, growth in (("a", 1.01), ("b", 1.02)):
            self.assertAlmostEqual(rows.at[source, "MOM1_raw"], growth - 1)
            for horizon in (3, 6, 9, 12):
                self.assertAlmostEqual(rows.at[source, f"MOM{horizon}_raw"], growth ** (horizon - 1) - 1)
        # Change only t: MOM1 changes; all longer features exclude t entirely.
        changed = prices().to_pylist()
        for row in changed:
            if row["trading_date"] == SIGNAL:
                row["adj_close"] *= 2
        changed_result = run(pa.Table.from_pylist(changed)).table.set_index("source_security_id")
        self.assertNotEqual(changed_result.at["a", "MOM1_raw"], rows.at["a", "MOM1_raw"])
        for horizon in (3, 6, 9, 12):
            for source in ("a", "b"):
                self.assertEqual(changed_result.at[source, f"MOM{horizon}_raw"], rows.at[source, f"MOM{horizon}_raw"])

    def test_cohort_means_and_sample_sd_and_no_target_overlap(self):
        result = run()
        rows = result.table.set_index("source_security_id")
        for horizon in (1, 3, 6, 9, 12):
            a, b = rows.at["a", f"MOM{horizon}_raw"], rows.at["b", f"MOM{horizon}_raw"]
            mean, sd = (a + b) / 2, abs(b - a) / math.sqrt(2)
            self.assertAlmostEqual(rows.at["a", f"meanMOM{horizon}"], mean)
            self.assertAlmostEqual(rows.at["b", f"stdMOM{horizon}"], sd)
            self.assertAlmostEqual(rows.at["a", f"zMOM{horizon}"], -1 / math.sqrt(2))
            self.assertAlmostEqual(rows.at["b", f"zMOM{horizon}"], 1 / math.sqrt(2))
        self.assertTrue(all("outcome" not in column and "target" not in column
                            for column in result.table.columns))

    def test_future_price_changes_leave_earlier_signal_features_unchanged(self):
        last = "2000-02-29"
        snapshots = [cohort("1999-12-31"), cohort(SIGNAL), cohort(last)]
        before = run(prices(last), snapshots=snapshots)
        changed = prices(last).to_pylist()
        for row in changed:
            if row["trading_date"] > SIGNAL:
                row["adj_close"] *= 7
        after = run(pa.Table.from_pylist(changed), snapshots=snapshots)
        pd.testing.assert_frame_equal(before.table[before.table.signal_date <= SIGNAL].reset_index(drop=True),
                                      after.table[after.table.signal_date <= SIGNAL].reset_index(drop=True))

    def test_target_columns_cannot_enter_the_price_interface(self):
        changed = prices().append_column("next_month_target", pa.array([987.0] * prices().num_rows))
        changed = changed.append_column("provider_adjusted_return", pa.array([-0.9] * prices().num_rows))
        with self.assertRaisesRegex(DataContractError, 'prohibited'):
            run(changed)

    def test_nonmember_prices_do_not_enter_normalization(self):
        result = run(prices(targets=("id-a", "id-b", "id-nonmember")),
                     requested=("id-a", "id-b", "id-nonmember"))
        pd.testing.assert_frame_equal(result.table, run().table)
        self.assertEqual(result.coverage.original_cohort_count.tolist(), [2] * 5)

    def test_requested_sample_preserves_missing_original_cohort_and_withholds_normalization(self):
        result = run(prices(targets=("id-a",)), requested=("id-a",))
        rows = result.table.set_index("source_security_id")
        self.assertEqual(set(rows.index), {"a", "b"})
        self.assertEqual(rows.at["b", "MOM1_raw_status"], "not_requested")
        self.assertTrue(pd.notna(rows.at["a", "MOM1_raw"]))
        for spec in result.specs:
            self.assertTrue(result.table[spec["name"]].isna().all())
        self.assertEqual(result.coverage.normalization_status.tolist(), ["incomplete_cohort"] * 5)
        self.assertEqual(result.coverage.raw_momentum_available_count.tolist(), [1] * 5)

    def test_unresolved_identity_kept_even_when_price_exists(self):
        unresolved = [member("a", "id-a"), member("b", "id-b", status="needs_review", review=True)]
        result = run(snapshots=[cohort(members=unresolved)])
        row = result.table.set_index("source_security_id").loc["b"]
        self.assertEqual(row.MOM1_raw_status, "identity_unresolved")
        self.assertTrue(pd.isna(row.MOM1_raw))
        self.assertEqual(result.coverage.identity_unresolved_count.tolist(), [1] * 5)
        self.assertTrue(result.table.zMOM1.isna().all())

    def test_malformed_resolved_flags_and_collision_are_defensively_quarantined(self):
        cases = [
            [member("a", "id-a"), member("b", "id-b", review=True)],
            [member("a", "id-a"), member("b", "id-b", resolution="pending")],
            [member("a", "id-a"), member("b", "id-a")],
        ]
        for members in cases:
            with self.subTest(members=members):
                result = run(snapshots=[cohort(members=members)])
                self.assertTrue(result.table.zMOM1.isna().all())
                self.assertIn(result.table.identity_status.iloc[-1], ("needs_review", "target_collision"))

    def test_removed_and_entering_members_follow_each_signal_cohort(self):
        first = cohort("1999-12-31", [member("a", "id-a"), member("b", "id-b")])
        second = cohort(SIGNAL, [member("a", "id-a"), member("new", "id-nonmember")])
        daily = prices(targets=("id-a", "id-b", "id-nonmember"))
        result = run(daily, snapshots=[first, second], requested=("id-a", "id-b", "id-nonmember"))
        first_rows = result.table[result.table.signal_date == "1999-12-31"]
        second_rows = result.table[result.table.signal_date == SIGNAL]
        self.assertEqual(set(first_rows.source_security_id), {"a", "b"})
        self.assertEqual(set(second_rows.source_security_id), {"a", "new"})
        # The entrant may use pre-entry historical prices, never future quotes.
        self.assertTrue(second_rows.set_index("source_security_id").at["new", "MOM12_raw"] > 0)

    def test_missing_intermediate_month_never_telescopes_or_compresses(self):
        rows = [row for row in prices().to_pylist() if not
                (row["intrinio_security_id"] == "id-a" and row["trading_date"] == "1999-06-30")]
        result = run(pa.Table.from_pylist(rows)).table.set_index("source_security_id")
        self.assertTrue(pd.notna(result.at["a", "MOM1_raw"]))
        self.assertTrue(pd.isna(result.at["a", "MOM12_raw"]))
        self.assertEqual(result.at["a", "MOM12_raw_status"], "missing_exact_endpoint")
        self.assertTrue(pd.isna(result.at["b", "zMOM12"]))

    def test_stale_endpoint_is_not_substituted(self):
        rows = [row for row in prices().to_pylist() if not
                (row["intrinio_security_id"] == "id-a" and row["trading_date"] == SIGNAL)]
        # An extra declared interior session with a valid stale quote cannot rescue month end.
        sessions = tuple(sorted([*SESSIONS, "2000-01-30"]))
        rows.append(quote("2000-01-30", 22))
        result = run(pa.Table.from_pylist(rows), sessions=sessions).table.set_index("source_security_id")
        self.assertTrue(pd.isna(result.at["a", "MOM1_raw"]))
        self.assertEqual(result.at["a", "MOM1_raw_status"], "missing_exact_endpoint")

    def test_quality_and_invalid_prices_do_not_create_values(self):
        for changes in ({"frequency": "weekly"}, {"intraperiod": True},
                        {"price_quality_status": "rejected"}, {"adj_close": None},
                        {"adj_close": 0.0}, {"adj_close": -1.0}, {"adj_close": float("inf")},
                        {"adj_close": float("nan")}):
            rows = prices().to_pylist()
            for row in rows:
                if row["intrinio_security_id"] == "id-a" and row["trading_date"] == SIGNAL:
                    row.update(changes)
            with self.subTest(changes=changes):
                result = run(pa.Table.from_pylist(rows)).table.set_index("source_security_id")
                self.assertTrue(pd.isna(result.at["a", "MOM1_raw"]))
                self.assertTrue(pd.isna(result.at["b", "zMOM1"]))

    def test_zero_dispersion_and_singleton_cross_sections_are_withheld(self):
        same = prices().to_pylist()
        a_prices = {row["trading_date"]: row["adj_close"] for row in same if row["intrinio_security_id"] == "id-a"}
        for row in same:
            row["adj_close"] = a_prices[row["trading_date"]]
        result = run(pa.Table.from_pylist(same))
        self.assertEqual(result.coverage.normalization_status.tolist(), ["degenerate_dispersion"] * 5)
        self.assertTrue(result.table.meanMOM1.isna().all())
        singleton = run(snapshots=[cohort(members=[member("a", "id-a")])])
        self.assertEqual(singleton.coverage.normalization_status.tolist(), ["insufficient_cohort_size"] * 5)
        self.assertTrue(singleton.table.stdMOM1.isna().all())

    def test_input_mutation_never_occurs_and_dataframe_input_works(self):
        daily = prices().to_pandas()
        original = daily.copy(deep=True)
        result = run(daily)
        pd.testing.assert_frame_equal(daily, original)
        pd.testing.assert_frame_equal(result.table, run().table)

    def test_missing_calendar_month_and_insufficient_warmup_are_visible(self):
        sessions = tuple(day for day in SESSIONS if day[:7] != "1999-06")
        rows = [row for row in prices().to_pylist() if row["trading_date"] in sessions]
        result = run(pa.Table.from_pylist(rows), sessions=sessions)
        self.assertEqual(result.table.MOM12_raw_status.tolist(), ["insufficient_calendar_warmup"] * 2)
        self.assertTrue(result.table.zMOM12.isna().all())

    def test_empty_cohort_keeps_schema_and_coverage_denominator(self):
        result = run(snapshots=[cohort(members=[])])
        self.assertEqual(len(result.table), 0)
        self.assertTrue(all(spec["name"] in result.table for spec in result.specs))
        self.assertEqual(result.coverage.original_cohort_count.tolist(), [0] * 5)
        self.assertEqual(result.coverage.normalization_status.tolist(), ["empty_cohort"] * 5)


class PaperFeatureGuardTests(unittest.TestCase):
    def test_future_protected_duplicates_and_unknown_requested_fail_before_conversion(self):
        for row in (quote("2000-02-29", 1), quote("2024-01-02", 1),
                    quote("1999-12-31", 1), quote("1999-12-31", 1, "unknown"),
                    quote("2000-01-30", 1)):
            rows = [*prices().to_pylist(), row]
            with self.subTest(row=row), self.assertRaises(DataContractError):
                run(FinancialConversionTrap(pa.Table.from_pylist(rows)))

    def test_calendar_and_signal_contracts(self):
        cases = [
            {"sessions": (*SESSIONS, SESSIONS[-1])},
            {"sessions": tuple(reversed(SESSIONS))},
            {"sessions": tuple(day for day in SESSIONS if day <= SIGNAL)},
            {"snapshots": [cohort("2000-01-30")]},
            {"snapshots": [cohort(), cohort()]},
            {"requested": ("id-a", "id-a")},
            {"policy": "requested_subset"},
            {"snapshots": []},
        ]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(DataContractError):
                run(**kwargs)

    def test_cohort_source_keys_and_knowledge_time_mode_fail_closed(self):
        duplicate = [member("a", "id-a"), member("a", "id-b")]
        with self.assertRaises(DataContractError):
            run(snapshots=[cohort(members=duplicate)])
        snapshot = cohort()
        snapshot.provenance["mode"] = "knowledge_time"
        with self.assertRaises(DataContractError):
            run(snapshots=[snapshot])

    def test_2024_calendar_metadata_is_allowed_but_financial_rows_are_not(self):
        sessions = ("2022-12-30", "2023-01-31", "2023-02-28", "2023-03-31", "2023-04-28",
                    "2023-05-31", "2023-06-30", "2023-07-31", "2023-08-31", "2023-09-29",
                    "2023-10-31", "2023-11-30", "2023-12-29", "2024-01-02")
        rows = [quote(day, 10 * growth ** offset, target)
                for target, growth in (("id-a", 1.01), ("id-b", 1.02))
                for offset, day in enumerate(sessions[:-1])]
        result = run(pa.Table.from_pylist(rows), snapshots=[cohort("2023-12-29")], sessions=sessions)
        self.assertTrue(result.table.zMOM12.notna().all())
        self.assertEqual(result.provenance["daily_financial_last_date"], "2023-12-29")


if __name__ == "__main__":
    unittest.main()
