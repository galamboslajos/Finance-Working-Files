"""Metadata-only identity coverage is distinct from observed price/features."""
import unittest
from unittest.mock import Mock, patch

import pyarrow as pa

from dmv2.data import BRIDGE_COLUMNS, CALENDAR_COLUMNS, LINK_COLUMNS, MEMBERSHIP_COLUMNS
from dmv2.loading import BudgetExceeded, LoadedTable, ReadBudget
from dmv2.universe import DataContractError, EXCLUDED_SCHEMA, UNIVERSE_SCHEMA, UniverseSnapshot


def identity_snapshot(rows=None, *, as_of="2000-01-31"):
    if rows is None:
        rows = [
            dict(source_security_id="a", intrinio_security_id="p-a", indices=["sp500"], identity_status="resolved", identity_review_required=False, resolution_status="resolved"),
            dict(source_security_id="b", intrinio_security_id="p-b", indices=["sp500"], identity_status="resolved", identity_review_required=True, resolution_status="resolved"),
            dict(source_security_id="c", intrinio_security_id="p-c", indices=["sp400"], identity_status="resolved", identity_review_required=False, resolution_status="resolved"),
            dict(source_security_id="d", intrinio_security_id="p-c", indices=["sp400"], identity_status="needs_review", identity_review_required=True, resolution_status="resolved"),
            dict(source_security_id="e", intrinio_security_id=None, indices=["sp400"], identity_status="unmapped_bridge", identity_review_required=None, resolution_status=None),
            dict(source_security_id="f", intrinio_security_id="p-f", indices=["sp500"], identity_status="resolved", identity_review_required=False, resolution_status="ambiguous"),
        ]
    full = [dict(as_of=as_of, **row) for row in rows]
    return UniverseSnapshot(pa.Table.from_pylist(full, schema=UNIVERSE_SCHEMA),
        pa.Table.from_pylist([], schema=EXCLUDED_SCHEMA), as_of, "combined",
        {"mode": "historical_reconstruction", "knowledge_time_verified": False},
        {"constituents": 999, "identity_status_counts": {"resolved": 999}})


def loaded(table, name, *, scanned=None):
    count = table.num_rows if scanned is None else scanned
    return LoadedTable(table, {"source": name, "financial": False},
        {"metadata_files": 1, "data_files": 1, "scanned_rows": count,
         "returned_rows": table.num_rows, "projected_compressed_bytes": count*8})


class CoverageAuditPreflightTests(unittest.TestCase):
    def test_bad_dates_and_budget_fail_before_any_source_calls(self):
        from dmv2.coverage_audit import build_coverage_audit
        changes = [dict(dates=()), dict(dates="2000-01-31"), dict(dates=("2000-1-31",)),
                   dict(dates=("2000-01-31", "2000-01-31")), dict(dates=("2005-01-31", "2000-01-31")),
                   dict(dates=("2024-01-31",)), dict(dates=("2000-02-30",)), dict(budget={})]
        for kwargs in changes:
            with self.subTest(kwargs=kwargs):
                catalog = Mock()
                with patch("dmv2.coverage_audit.BoundedTableReader") as reader:
                    with self.assertRaises(DataContractError):
                        build_coverage_audit(catalog, **kwargs)
                    reader.assert_not_called()
                self.assertEqual(catalog.mock_calls, [])

    def test_initial_training_scope_count_and_budget_ceiling_preflight(self):
        from dmv2.coverage_audit import AUDIT_BUDGET, build_coverage_audit
        from dataclasses import replace
        cases = [dict(dates=("1999-12-31",)), dict(dates=("2010-01-29",)),
                 dict(dates=("2000-01-31", "2001-01-31", "2005-01-31", "2009-01-30")),
                 dict(universe="sp600"), dict(budget=replace(AUDIT_BUDGET,
                      max_scanned_rows=AUDIT_BUDGET.max_scanned_rows+1))]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                catalog = Mock()
                with patch("dmv2.coverage_audit.BoundedTableReader") as reader:
                    with self.assertRaises(DataContractError):
                        build_coverage_audit(catalog, **kwargs)
                    reader.assert_not_called()
                self.assertEqual(catalog.mock_calls, [])


class CoverageSummaryTests(unittest.TestCase):
    def summarize(self, snapshot=None, **kwargs):
        from dmv2.coverage_audit import summarize_coverage_snapshot
        return summarize_coverage_snapshot(snapshot or identity_snapshot(), **kwargs)

    def test_original_denominator_and_defensive_identity_review(self):
        result = self.summarize()
        self.assertEqual(result["original_members"], 6)
        universe = result["groups"]["universe"]
        self.assertEqual(universe["original_members"], 6)
        self.assertEqual(universe["qualified_identity_members"], 1)
        self.assertEqual(universe["qualified_identity_fraction"], 1/6)
        self.assertEqual(universe["identity_status_counts"], dict(
            resolved=1, needs_review=2, target_collision=2, unmapped_bridge=1))
        self.assertEqual(sum(universe["identity_status_counts"].values()), 6)
        self.assertEqual(len(result["constituent_ledger"]), 6)
        self.assertFalse(result["knowledge_time_verified"])
        self.assertFalse(result["financial_values_read"])
        self.assertFalse(result["price_history_coverage_verified"])
        self.assertFalse(result["normalization_adequacy_verified"])
        for group in result["groups"].values():
            self.assertFalse(group["five_momentum_population_verified"])
            self.assertIsNone(group["inventory_intersection_members"])
        # Metadata-only safe IDs do not establish a qualified complete cohort.
        self.assertEqual(universe["candidate_signal_qualified_population_upper_bound"], 1)
        self.assertFalse(universe["active_complete_cohort_possible_from_identity"])
        self.assertEqual(universe["active_complete_cohort_normalized_rows_upper_bound"], 0)

    def test_sleeve_denominators_are_retained_separately(self):
        result = self.summarize()
        large, middle = result["groups"]["sp500"], result["groups"]["sp400"]
        self.assertEqual((large["original_members"], large["qualified_identity_members"]), (3, 1))
        self.assertEqual((middle["original_members"], middle["qualified_identity_members"]), (3, 0))
        self.assertEqual(large["qualified_identity_fraction"], 1/3)
        self.assertEqual(middle["qualified_identity_fraction"], 0)

    def test_optional_inventory_never_changes_identity_or_claims_price_coverage(self):
        baseline = self.summarize()
        inventory = self.summarize(historical_price_ids=("p-a", "p-b", "p-c", "p-f", "future-only-id"))
        missing = self.summarize(historical_price_ids=())
        for result in (inventory, missing):
            group = result["groups"]["universe"]
            self.assertEqual(group["qualified_identity_members"], 1)
            self.assertEqual(group["original_members"], 6)
            self.assertEqual(group["identity_status_counts"], baseline["groups"]["universe"]["identity_status_counts"])
            self.assertFalse(result["price_history_coverage_verified"])
            self.assertFalse(group["five_momentum_population_verified"])
            self.assertEqual([r["source_security_id"] for r in result["constituent_ledger"]], list("abcdef"))
        self.assertEqual(inventory["groups"]["universe"]["inventory_intersection_members"], 1)
        self.assertEqual(missing["groups"]["universe"]["inventory_intersection_members"], 0)

    def test_complete_identity_population_remains_unverified_numerically(self):
        source = identity_snapshot().table.to_pylist()
        selected = [row for row in source if row["source_security_id"] in ("a", "c")]
        rows = [{key: value for key, value in row.items() if key != "as_of"} for row in selected]
        result = self.summarize(identity_snapshot(rows))
        group = result["groups"]["universe"]
        self.assertEqual((group["original_members"], group["qualified_identity_members"]), (2, 2))
        self.assertEqual(group["candidate_signal_qualified_population_upper_bound"], 2)
        self.assertTrue(group["active_complete_cohort_possible_from_identity"])
        self.assertEqual(group["active_complete_cohort_normalized_rows_upper_bound"], 2)
        self.assertFalse(group["five_momentum_population_verified"])
        self.assertFalse(result["normalization_adequacy_verified"])

    def test_overlap_is_one_combined_source_but_present_in_both_sleeves(self):
        rows = [dict(source_security_id=sid, intrinio_security_id="p-"+sid,
                     indices=indices, identity_status="resolved", identity_review_required=False,
                     resolution_status="resolved") for sid, indices in
                (("a", ["sp500", "sp400"]), ("b", ["sp500"]), ("c", ["sp400"]))]
        result = self.summarize(identity_snapshot(rows))
        self.assertEqual(result["original_members"], 3)
        self.assertEqual(result["groups"]["sp500"]["original_members"], 2)
        self.assertEqual(result["groups"]["sp400"]["original_members"], 2)

    def test_invalid_summary_routes_dates_and_inventory_fail_closed(self):
        malformed = []
        rows = identity_snapshot().table.to_pylist()
        rows = [{k: v for k, v in row.items() if k != "as_of"} for row in rows]
        malformed.append(identity_snapshot(rows+[rows[0]]))
        malformed.append(identity_snapshot([{**rows[0], "intrinio_security_id": None}]))
        malformed.append(identity_snapshot([{**rows[0], "indices": ["sp600"]}]))
        malformed.append(identity_snapshot([{**rows[0], "source_security_id": ""}]))
        malformed.append(identity_snapshot(as_of="2024-01-31"))
        for snapshot in malformed:
            with self.subTest(snapshot=snapshot.as_of), self.assertRaises(DataContractError):
                self.summarize(snapshot)
        for inventory in ("p-a", ("p-a", "p-a"), ("",), (None,)):
            with self.subTest(inventory=inventory), self.assertRaises(DataContractError):
                self.summarize(historical_price_ids=inventory)


class CoverageAuditOrchestrationTests(unittest.TestCase):
    def setUp(self):
        self.dates = ("2000-01-31", "2005-01-31", "2009-01-30")
        calendar = pa.Table.from_pylist([
            dict(trading_date=day, calendar_name="XNYS", is_trading_session=True)
            for day in ("1999-12-31", *self.dates, "2009-02-02")])
        members = pa.Table.from_pylist([dict(index_id="sp500", trading_date=day, security_id=sid,
            membership_ticker="TEST_ONLY", is_member=True, membership_effective_date="1999-01-01",
            membership_end_date=None, membership_availability_date="1999-01-01",
            identity_confidence="test", pit_policy="test") for day in self.dates for sid in ("a", "unmapped")])
        bridge = pa.Table.from_pylist([dict(source_product_id="sp_composite1500", source_security_id="a",
            unified_security_id="u-a", bridge_method="test", bridge_confidence="test")])
        links = pa.Table.from_pylist([dict(unified_security_id="u-a", intrinio_security_id="p-a",
            identity_review_required=False, resolution_status="resolved")])
        self.calendar, self.bridge, self.links = loaded(calendar, "calendar"), loaded(bridge, "bridge"), loaded(links, "links")
        self.members = [loaded(members.filter(pa.compute.equal(members["trading_date"], day)), "membership_"+day) for day in self.dates]
        self.catalog = Mock()
        self.catalog.provenance.return_value = [{"source_key": "structural-test", "manifest_sha256": "test"}]
        self.reader = Mock()
        self.reader.read_metadata.side_effect = [self.calendar, self.bridge, self.links]
        self.reader.read.side_effect = self.members
        from dmv2.exchange_sessions import ReconciledExchangeCalendar
        self.reader_patch = patch("dmv2.coverage_audit.BoundedTableReader", return_value=self.reader)
        self.reader_class = self.reader_patch.start()
        self.calendar_patch = patch("dmv2.coverage_audit.reconcile_exchange_calendar",
            side_effect=lambda source, **kwargs: ReconciledExchangeCalendar(source, source, {"test_only": True}))
        self.calendar_builder = self.calendar_patch.start()
        self.addCleanup(self.reader_patch.stop)
        self.addCleanup(self.calendar_patch.stop)

    def test_only_approved_metadata_tables_and_projections_are_read(self):
        from dmv2.coverage_audit import build_coverage_audit
        result = build_coverage_audit(self.catalog)
        tables = {("siblis", "trading_calendar_sessions"): CALENDAR_COLUMNS,
                  ("identity", "security_identity_bridge"): BRIDGE_COLUMNS,
                  ("intrinio", "intrinio_identity_links"): LINK_COLUMNS,
                  ("siblis", "membership_by_index_by_date"): MEMBERSHIP_COLUMNS}
        for call in [*self.reader.read_metadata.call_args_list, *self.reader.read.call_args_list]:
            self.assertIn(call.args, tables)
            self.assertEqual(tuple(call.kwargs["columns"]), tables[call.args])
            self.assertEqual(tuple(call.kwargs["allowed_columns"]), tables[call.args])
            self.assertFalse(set(call.kwargs["columns"]) & {"close", "adj_close", "volume", "target", "label"})
        self.assertEqual(self.reader.read.call_count, 3)
        for call, day in zip(self.reader.read.call_args_list, self.dates, strict=True):
            self.assertEqual((call.kwargs["start"], call.kwargs["end"]), (day, day))
            self.assertFalse(call.kwargs["financial"])
            self.assertEqual(call.kwargs["filters"], {"index_id": ("sp500", "sp400")})
        self.assertEqual(self.calendar_builder.call_count, 3)
        for call, day in zip(self.calendar_builder.call_args_list, self.dates, strict=True):
            self.assertEqual(call.kwargs, dict(start_month=day[:7], end_month=day[:7]))
        self.assertEqual([s["original_members"] for s in result["snapshots"]], [2, 2, 2])
        self.assertTrue(all(s["groups"]["universe"]["qualified_identity_members"] == 1 for s in result["snapshots"]))
        self.assertFalse(result["financial_values_read"])
        self.assertFalse(result["targets_read"])
        self.assertEqual(result["fits_performed"], 0)
        self.assertEqual(result["loading_stats"]["total"]["scanned_rows"], 13)

    def test_independent_non_month_end_fails_before_id_and_membership_reads(self):
        from dmv2.coverage_audit import build_coverage_audit
        with self.assertRaisesRegex(DataContractError, "month-ends"):
            build_coverage_audit(self.catalog, dates=("2000-01-28",))
        self.assertEqual(self.reader.read_metadata.call_count, 1)
        self.reader.read.assert_not_called()
        self.assertEqual(self.catalog.mock_calls, [])

    def test_cumulative_budget_rejects_overrun_without_partial_audit(self):
        from dmv2.coverage_audit import build_coverage_audit
        self.reader.read_metadata.side_effect = [loaded(self.calendar.table, "calendar", scanned=6),
                                                 loaded(self.bridge.table, "bridge", scanned=6), self.links]
        with self.assertRaises(BudgetExceeded):
            build_coverage_audit(self.catalog, budget=ReadBudget(max_scanned_rows=10))
        self.assertEqual(self.reader.read_metadata.call_count, 2)
        self.reader.read.assert_not_called()

    def test_remaining_budgets_are_reduced_between_source_stages(self):
        from dmv2.coverage_audit import AUDIT_BUDGET, build_coverage_audit
        observed = []
        records = iter([self.calendar, self.bridge, self.links])
        def metadata(*args, **kwargs):
            observed.append(self.reader.budget.max_scanned_rows)
            return next(records)
        self.reader.budget = AUDIT_BUDGET
        self.reader.read_metadata.side_effect = metadata
        build_coverage_audit(self.catalog)
        self.assertEqual(observed, [AUDIT_BUDGET.max_scanned_rows,
                                   AUDIT_BUDGET.max_scanned_rows-self.calendar.stats["scanned_rows"],
                                   AUDIT_BUDGET.max_scanned_rows-self.calendar.stats["scanned_rows"]-self.bridge.stats["scanned_rows"]])


if __name__ == "__main__":
    unittest.main()
