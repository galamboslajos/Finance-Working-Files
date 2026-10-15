"""Structural orchestration tests; these fixtures are never research observations."""
import unittest
from unittest.mock import Mock, patch

import pyarrow as pa

from dmv2.data import BRIDGE_COLUMNS, CALENDAR_COLUMNS, LINK_COLUMNS, MEMBERSHIP_COLUMNS
from dmv2.loading import BudgetExceeded, LoadedTable, ProtectedDataError, ReadBudget
from dmv2.exchange_sessions import CALENDAR_POLICY, ReconciledExchangeCalendar
from dmv2.monthly_loading import DAILY_SAMPLE_COLUMNS, load_monthly_sample
from dmv2.universe import DataContractError, KnowledgeTimeUnavailable


def loaded(table, name, *, rows=None, files=1):
    count = table.num_rows if rows is None else rows
    return LoadedTable(table, {"name": name, "financial": name == "prices"},
                       {"metadata_files": files, "data_files": files,
                        "scanned_rows": count, "returned_rows": table.num_rows,
                        "projected_compressed_bytes": count * 8})


def fixtures():
    calendar = pa.Table.from_pylist([
        {"trading_date": day, "calendar_name": "XNYS", "is_trading_session": True}
        for day in ["1999-12-30", "1999-12-31", "2000-01-03", "2000-01-31",
                    "2000-02-01", "2000-02-29", "2000-03-01"]])
    # z disappears from the later signal cohort, while a enters then. The
    # first-date ID selection must retain z and must not prefer future a.
    entries = [("1999-12-31", "z"), ("1999-12-31", "b"),
               ("1999-12-31", "unresolved"),
               ("2000-01-31", "a"), ("2000-01-31", "b"),
               ("2000-01-31", "unresolved")]
    members = pa.Table.from_pylist([
        {"index_id": "sp500", "trading_date": day, "security_id": sid,
         "membership_ticker": sid, "is_member": True,
         "membership_effective_date": "1999-12-01",
         "membership_end_date": "2000-03-01", "membership_availability_date": day,
         "identity_confidence": "high_isin", "pit_policy": "test_contract"}
        for day, sid in entries])
    bridge = pa.Table.from_pylist([
        {"source_product_id": "sp_composite1500", "source_security_id": sid,
         "unified_security_id": "u-" + sid, "bridge_method": "isin",
         "bridge_confidence": "high"} for sid in ["z", "b", "a"]])
    links = pa.Table.from_pylist([
        {"unified_security_id": "u-" + sid, "intrinio_security_id": "p-" + sid,
         "identity_review_required": False, "resolution_status": "resolved"}
        for sid in ["z", "b", "a"]])
    # No prices exist for z or a. Price absence must not alter ID selection.
    prices = pa.Table.from_pylist([
        {"intrinio_security_id": "p-b", "trading_date": "1999-12-31",
         "adj_close": 10.0, "frequency": "daily", "intraperiod": False,
         "price_quality_status": "accepted"}])
    return {"calendar": loaded(calendar, "calendar"),
            "membership": loaded(members, "membership"),
            "bridge": loaded(bridge, "bridge"),
            "links": loaded(links, "links"), "prices": loaded(prices, "prices")}


class MonthlyGuardTests(unittest.TestCase):
    def test_invalid_requests_fail_before_catalog_or_reader_calls(self):
        cases = [
            ({"start_month": "2000-1"}, DataContractError),
            ({"start_month": "2000-01-01"}, DataContractError),
            ({"start_month": "2000-00"}, DataContractError),
            ({"start_month": "2000-13"}, DataContractError),
            ({"start_month": None}, DataContractError),
            ({"start_month": "0001-01"}, DataContractError),
            ({"start_month": "2000-03"}, DataContractError),
            ({"end_month": "2000-04"}, DataContractError),
            ({"start_month": "2023-12", "end_month": "2024-01"}, ProtectedDataError),
            ({"sample_size": 0}, DataContractError),
            ({"sample_size": 11}, DataContractError),
            ({"sample_size": True}, DataContractError),
            ({"sample_size": 2.0}, DataContractError),
            ({"universe": "sp600"}, DataContractError),
            ({"mode": "knowledge_time"}, KnowledgeTimeUnavailable),
            ({"mode": "unknown"}, DataContractError),
            ({"budget": {}}, DataContractError),
        ]
        for changes, error in cases:
            with self.subTest(changes=changes):
                catalog = Mock()
                params = dict(start_month="2000-01", end_month="2000-02", universe="combined")
                params.update(changes)
                with patch("dmv2.monthly_loading.BoundedTableReader") as reader, \
                        patch("dmv2.monthly_loading.PriceLoader") as price:
                    with self.assertRaises(error):
                        load_monthly_sample(catalog, **params)
                self.assertEqual(catalog.mock_calls, [])
                reader.assert_not_called()
                price.assert_not_called()


class MonthlyOrchestrationTests(unittest.TestCase):
    def setUp(self):
        self.source = fixtures()
        self.catalog = Mock()
        self.catalog.provenance.return_value = [{"source_key": "test", "manifest_sha256": "test-pin"}]
        self.reader = Mock()
        self.reader.read_metadata.side_effect = [self.source["calendar"],
                                                 self.source["bridge"], self.source["links"]]
        self.reader.read.return_value = self.source["membership"]
        self.price = Mock()
        self.price.load.return_value = self.source["prices"]
        self.panel = Mock()
        self.reader_patch = patch("dmv2.monthly_loading.BoundedTableReader", return_value=self.reader)
        self.price_patch = patch("dmv2.monthly_loading.PriceLoader", return_value=self.price)
        self.panel_patch = patch("dmv2.monthly_loading.build_monthly_returns", return_value=self.panel)
        self.calendar_patch = patch(
            "dmv2.monthly_loading.reconcile_exchange_calendar",
            side_effect=lambda source, **kwargs: ReconciledExchangeCalendar(
                source, source, {"policy": CALENDAR_POLICY, "test_fixture": True}))
        self.reader_class = self.reader_patch.start()
        self.price_class = self.price_patch.start()
        self.panel_builder = self.panel_patch.start()
        self.calendar_builder = self.calendar_patch.start()
        self.addCleanup(self.reader_patch.stop)
        self.addCleanup(self.price_patch.stop)
        self.addCleanup(self.panel_patch.stop)
        self.addCleanup(self.calendar_patch.stop)

    def run_sample(self, **overrides):
        params = dict(start_month="2000-01", end_month="2000-02", universe="combined")
        params.update(overrides)
        return load_monthly_sample(self.catalog, **params)

    def test_batched_exact_signal_membership_and_metadata_read_once(self):
        result = self.run_sample()
        self.assertEqual(self.reader.read.call_count, 1)
        call = self.reader.read.call_args
        self.assertEqual(call.args, ("siblis", "membership_by_index_by_date"))
        self.assertEqual(call.kwargs["start"], "1999-12-31")
        self.assertEqual(call.kwargs["end"], "2000-01-31")
        self.assertEqual(call.kwargs["filters"], {
            "index_id": ("sp500", "sp400"),
            "trading_date": ("1999-12-31", "2000-01-31")})
        self.assertIs(call.kwargs["financial"], False)
        self.assertEqual(call.kwargs["columns"], MEMBERSHIP_COLUMNS)
        self.assertEqual(self.reader.read_metadata.call_count, 3)
        expected = [(("siblis", "trading_calendar_sessions"), CALENDAR_COLUMNS),
                    (("identity", "security_identity_bridge"), BRIDGE_COLUMNS),
                    (("intrinio", "intrinio_identity_links"), LINK_COLUMNS)]
        for call, (args, columns) in zip(self.reader.read_metadata.call_args_list, expected):
            self.assertEqual(call.args, args)
            self.assertEqual(call.kwargs["columns"], columns)
        self.assertEqual(tuple(s.as_of for s in result.snapshots),
                         ("1999-12-31", "2000-01-31"))
        self.calendar_builder.assert_called_once_with(
            self.source["calendar"].table, start_month="2000-01", end_month="2000-02")

    def test_first_signal_selection_ignores_future_membership_and_missing_prices(self):
        result = self.run_sample()
        self.assertEqual(result.requested_ids, ("p-b", "p-z"))
        self.assertNotIn("p-a", result.requested_ids)
        self.assertEqual([s.table.num_rows for s in result.snapshots], [3, 3])
        self.assertEqual(result.snapshots[0].table.to_pylist()[-1]["source_security_id"], "z")
        self.assertEqual(self.price.load.call_count, 1)
        self.price.load.assert_called_once_with(
            ("p-b", "p-z"), "1999-12-31", "2000-02-29", DAILY_SAMPLE_COLUMNS)
        self.assertIn("frequency", DAILY_SAMPLE_COLUMNS)
        self.assertIn("intraperiod", DAILY_SAMPLE_COLUMNS)
        self.assertTrue(set(["open", "high", "low", "close", "volume", "adj_open",
                            "adj_high", "adj_low", "adj_close", "adj_volume", "factor",
                            "split_ratio", "dividend", "raw_ohlc_bounds_valid",
                            "adjusted_ohlc_bounds_valid", "price_quality_status"])
                        <= set(DAILY_SAMPLE_COLUMNS))

    def test_unchanged_daily_source_and_full_snapshots_pass_to_monthly_core(self):
        result = self.run_sample()
        self.assertIs(result.daily_prices, self.source["prices"])
        self.assertIs(result.daily_prices.table, self.source["prices"].table)
        self.assertIs(result.source_calendar, self.source["calendar"].table)
        self.assertIs(result.panel, self.panel)
        call = self.panel_builder.call_args
        self.assertIs(call.args[0], self.source["prices"].table)
        self.assertIs(call.args[1], self.source["calendar"].table)
        self.assertEqual(call.args[2], result.snapshots)
        self.assertEqual(call.kwargs["requested_security_ids"], result.requested_ids)
        self.assertEqual(call.kwargs["mode"], "historical_reconstruction")
        self.assertEqual(call.kwargs["start_month"], "2000-01")
        self.assertEqual(call.kwargs["end_month"], "2000-02")

    def test_missing_required_signal_fails_before_identity_or_price_reads(self):
        members = self.source["membership"].table.slice(0, 3)
        self.reader.read.return_value = loaded(members, "membership")
        with self.assertRaisesRegex(DataContractError, "required signal: 2000-01-31"):
            self.run_sample()
        self.assertEqual(self.reader.read_metadata.call_count, 1)
        self.price_class.assert_not_called()
        self.panel_builder.assert_not_called()

    def test_all_failed_signal_availability_fails_before_price_reads(self):
        rows = self.source["membership"].table.to_pylist()
        for row in rows:
            if row["trading_date"] == "2000-01-31":
                row["membership_availability_date"] = "2000-02-01"
        self.reader.read.return_value = loaded(pa.Table.from_pylist(rows), "membership")
        with self.assertRaisesRegex(DataContractError, "failed date/availability gates: 2000-01-31"):
            self.run_sample()
        self.price_class.assert_not_called()

    def test_no_resolved_first_signal_ids_does_not_borrow_future_ids(self):
        rows = self.source["links"].table.to_pylist()
        for row in rows:
            if row["unified_security_id"] in ("u-z", "u-b"):
                row["identity_review_required"] = True
        self.reader.read_metadata.side_effect = [self.source["calendar"], self.source["bridge"],
                                                loaded(pa.Table.from_pylist(rows), "links")]
        with self.assertRaisesRegex(DataContractError, "First signal cohort has no resolved"):
            self.run_sample()
        self.price_class.assert_not_called()

    def test_cumulative_budget_reaches_price_reader_and_totals_are_preserved(self):
        budget = ReadBudget(max_scanned_rows=1000)
        result = self.run_sample(budget=budget)
        used = sum(self.source[name].stats["scanned_rows"]
                   for name in ("calendar", "membership", "bridge", "links"))
        price_budget = self.price_class.call_args.args[1]
        self.assertEqual(price_budget.max_scanned_rows, 1000 - used)
        self.assertEqual(result.loading_stats["total"]["scanned_rows"], used + 1)
        self.assertEqual(set(result.loading_provenance["stages"]),
                         {"calendar", "membership", "bridge", "links", "daily_prices"})
        self.assertEqual(result.loading_provenance["stage_budget"]["max_scanned_rows"], 1000)
        self.assertTrue(result.loading_provenance["request_sha256"].startswith("sha256:"))
        self.assertEqual(result.loading_provenance["request"]["signal_dates"],
                         ["1999-12-31", "2000-01-31"])
        self.assertEqual(result.loading_provenance["request"]["calendar_basis"], CALENDAR_POLICY)
        self.assertEqual(result.loading_provenance["calendar_reconciliation"]["policy"], CALENDAR_POLICY)

    def test_independent_calendar_drives_windows_while_source_is_preserved(self):
        # The vendor includes the Good Friday2018-03-30 holiday as month-end.
        # Independent metadata chooses2018-03-29 without looking at prices.
        source = pa.Table.from_pylist([
            {"trading_date": day, "calendar_name": "XNYS", "is_trading_session": True}
            for day in ["2018-02-28", "2018-03-29", "2018-03-30", "2018-04-02"]])
        independent = source.slice(0, 2)
        independent = pa.concat_tables([independent, source.slice(3, 1)])
        self.reader.read_metadata.side_effect = [loaded(source, "calendar"),
                                                self.source["bridge"], self.source["links"]]
        members = self.source["membership"].table.slice(0, 3).to_pylist()
        for row in members:
            row.update(trading_date="2018-02-28", membership_effective_date="2018-01-01",
                       membership_end_date="2018-04-01", membership_availability_date="2018-02-28")
        self.reader.read.return_value = loaded(pa.Table.from_pylist(members), "membership")
        self.calendar_builder.side_effect = None
        self.calendar_builder.return_value = ReconciledExchangeCalendar(
            independent, source, {"policy": CALENDAR_POLICY, "source_only_dates": ["2018-03-30"]})
        price_rows=self.source['prices'].table.to_pylist()
        for row in price_rows:row['trading_date']='2018-02-28'
        self.price.load.return_value=loaded(pa.Table.from_pylist(price_rows),'prices')
        result = self.run_sample(start_month="2018-03", end_month="2018-03")
        self.assertIs(result.source_calendar, source)
        self.assertIs(result.calendar, independent)
        self.assertEqual(self.price.load.call_args.args[2], "2018-03-29")
        self.assertIs(self.panel_builder.call_args.args[1], independent)
        self.assertEqual(result.loading_provenance["calendar_reconciliation"]["source_only_dates"],
                         ["2018-03-30"])

    def test_non_session_quotes_are_audited_without_mutating_raw_daily_input(self):
        source_rows=self.source['prices'].table.to_pylist()
        source_rows.append({**source_rows[0],'trading_date':'2000-01-15','adj_close':999.})
        raw=loaded(pa.Table.from_pylist(source_rows),'prices')
        self.price.load.return_value=raw
        result=self.run_sample()
        self.assertIs(result.daily_prices,raw)
        self.assertEqual(raw.table.num_rows,2)
        self.assertEqual(self.panel_builder.call_args.args[0].num_rows,1)
        audit=result.loading_provenance['quote_session_audit']
        self.assertEqual(audit['excluded_non_session_rows'],1)
        self.assertEqual(audit['excluded_quote_keys'][0]['trading_date'],'2000-01-15')

    def test_exhausted_cumulative_budget_rejects_next_read_without_autoscaling(self):
        # Calendar alone consumes the whole allowed stage row budget.
        with self.assertRaisesRegex(BudgetExceeded, "Monthly stage budget exhausted"):
            self.run_sample(budget=ReadBudget(max_scanned_rows=7))
        self.reader.read.assert_not_called()
        self.price_class.assert_not_called()

    def test_insufficient_source_calendar_fails_before_membership_reads(self):
        calendar = self.source["calendar"].table.slice(0, 2)
        self.reader.read_metadata.side_effect = [loaded(calendar, "calendar")]
        with self.assertRaises(DataContractError):
            self.run_sample()
        self.reader.read.assert_not_called()
        self.price_class.assert_not_called()


if __name__ == "__main__":
    unittest.main()
