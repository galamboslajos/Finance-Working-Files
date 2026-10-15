"""Feature/target boundary regressions using artificial structural fixtures."""
from collections import Counter
from dataclasses import replace
import unittest
from unittest.mock import Mock, patch

import pandas as pd
import pyarrow as pa

from dmv2.features import (FeaturePanel, build_feature_panel, predictor_frame,
                           registered_feature_specs, training_slice)
from dmv2.identity_policy import PROVISIONAL_IDENTITY_POLICY, STRICT_IDENTITY_POLICY
from dmv2.universe import DataContractError, KnowledgeTimeUnavailable


WINDOWS = {
    "1999-12-31": {"outcome_month": "2000-01", "outcome_start": "2000-01-03", "outcome_end": "2000-01-31"},
    "2000-01-31": {"outcome_month": "2000-02", "outcome_start": "2000-02-01", "outcome_end": "2000-02-29"},
}
PREDICTORS = ["px_simple_return_1s", "zMOM1"]


def panel_fixture():
    specs = registered_feature_specs()
    rows = []
    for offset, (source, day) in enumerate((
        ("a", "1999-12-31"), ("b", "1999-12-31"),
        ("a", "2000-01-31"), ("b", "2000-01-31"),
    )):
        rows.append({"source_security_id": source, "signal_date": day,
                     **{spec["name"]: float(offset + 1) for spec in specs}})
    rows[1]["zMOM1"] = None
    return FeaturePanel(pa.Table.from_pylist(rows), pa.Table.from_pylist([]), specs, {}, {},
                        {"target_windows": WINDOWS, "knowledge_time_verified": False})


def outcomes_fixture():
    rows = []
    for day, window in WINDOWS.items():
        for source, value in (("a", 0.10), ("b", None)):
            rows.append({"source_security_id": source, "signal_date": day, **window,
                         "provider_adjusted_return": value,
                         "return_status": "ok" if value is not None else "missing_end"})
    return pa.Table.from_pylist(rows)


def training(panel=None, outcomes=None, **overrides):
    params = dict(train_start="2000-01-01", train_end="2000-02-29",
                  label_observed_through="2000-01-31", columns=PREDICTORS)
    params.update(overrides)
    return training_slice(panel_fixture() if panel is None else panel,
                          outcomes_fixture() if outcomes is None else outcomes, **params)


class RegistrationAndPredictorTests(unittest.TestCase):
    def test_exact_registered_bank_includes_paper_fifteen(self):
        specs = registered_feature_specs()
        self.assertEqual(len(specs), 728)
        self.assertEqual(len({spec["name"] for spec in specs}), 728)
        self.assertEqual(Counter(spec["role"] for spec in specs),
                         {"technical": 298, "paper": 15, "cross_sectional": 320,
                          "market_state": 65, "relative_context": 30})
        self.assertEqual({spec["name"] for spec in specs if spec["role"] == "paper"},
                         {f"{prefix}{h}" for prefix in ("zMOM", "meanMOM", "stdMOM")
                          for h in (1, 3, 6, 9, 12)})

    def test_metadata_targets_and_diagnostics_cannot_be_selected(self):
        panel = panel_fixture()
        forbidden = ("provider_adjusted_return", "return_status", "outcome_end", "outcome_start",
                     "outcome_month", "identity_status", "identity_review_required",
                     "identity_eligible", "identity_eligibility_policy", "identity_eligibility_reason",
                     "source_identity_status",
                     "knowledge_time_verified", "accounting_certified", "model_ready",
                     "MOM1_raw", "target", "label", "signal_date")
        for name in forbidden:
            with self.subTest(name=name), self.assertRaises(DataContractError):
                predictor_frame(panel, columns=[name])
        for columns in ([], [PREDICTORS[0], PREDICTORS[0]]):
            with self.assertRaises(DataContractError):
                predictor_frame(panel, columns=columns)

    def test_unregistered_table_field_and_spec_injection_are_rejected(self):
        panel = panel_fixture()
        for name in ("provider_adjusted_return", "identity_review_required"):
            with self.subTest(name=name):
                mutated = replace(panel, table=panel.table.append_column(name, pa.array([1.0] * 4)))
                with self.assertRaises(DataContractError):
                    predictor_frame(mutated)
                replacement = {**panel.specs[0], "name": name}
                registered_injection = replace(panel, specs=(replacement, *panel.specs[1:]))
                with self.assertRaises(DataContractError):
                    predictor_frame(registered_injection)

    def test_constants_and_exact_rank_aliases_cannot_be_reenabled_by_panel_flags(self):
        panel = panel_fixture()
        excluded = [spec["name"] for spec in panel.specs if not spec.get("model_eligible", True)]
        self.assertGreaterEqual(len(excluded), 3)
        X = predictor_frame(panel)
        self.assertFalse(set(excluded) & set(X))
        mutated = replace(panel, specs=tuple({**spec, "model_eligible": True} for spec in panel.specs))
        for name in excluded:
            with self.subTest(name=name), self.assertRaises(DataContractError):
                predictor_frame(mutated, columns=[name])

    def test_predictor_key_uniqueness_and_missing_source_ids_fail_closed(self):
        panel = panel_fixture()
        rows = panel.table.to_pylist()
        for malformed in ([rows[0], rows[0], *rows[2:]],
                          [{**rows[0], "source_security_id": None}, *rows[1:]],
                          [{**rows[0], "source_security_id": ""}, *rows[1:]]):
            with self.subTest(key=malformed[0]["source_security_id"]), self.assertRaises(DataContractError):
                predictor_frame(replace(panel, table=pa.Table.from_pylist(malformed)), columns=PREDICTORS)


class TrainingBoundaryTests(unittest.TestCase):
    def test_only_completed_labels_enter_and_missing_survivor_outcomes_are_retained(self):
        sliced = training()
        self.assertEqual(list(sliced.X.index), [("a", "1999-12-31"), ("b", "1999-12-31")])
        self.assertEqual(len(sliced.y), 2)
        self.assertAlmostEqual(sliced.y.iloc[0], 0.10)
        self.assertTrue(pd.isna(sliced.y.iloc[1]))
        self.assertEqual(sliced.ledger.return_status.tolist(), ["ok", "missing_end"])
        self.assertEqual(sliced.ledger.target_available.tolist(), [True, False])
        self.assertEqual(sliced.ledger.feature_complete.tolist(), [True, False])
        self.assertFalse(sliced.ledger.model_ready.any())
        self.assertFalse(sliced.provenance["knowledge_time_verified"])

    def test_future_uncompleted_targets_never_select_or_modify_training_X(self):
        baseline = training()
        rows = outcomes_fixture().to_pylist()
        for row in rows:
            if row["outcome_end"] > "2000-01-31":
                row["provider_adjusted_return"] = 1e200
                row["return_status"] = "ok"
        result = training(outcomes=pa.Table.from_pylist(rows))
        pd.testing.assert_frame_equal(baseline.X, result.X)
        pd.testing.assert_series_equal(baseline.y, result.y)
        pd.testing.assert_frame_equal(baseline.ledger, result.ledger)

    def test_target_value_and_availability_perturbation_does_not_select_X(self):
        baseline = training()
        rows = outcomes_fixture().to_pylist()
        rows[0].update(provider_adjusted_return=None, return_status="missing_end")
        rows[1].update(provider_adjusted_return=-0.8, return_status="ok")
        result = training(outcomes=pa.Table.from_pylist(rows))
        pd.testing.assert_frame_equal(baseline.X, result.X)
        self.assertEqual(len(result.X), 2)
        self.assertTrue(pd.isna(result.y.iloc[0]))
        self.assertAlmostEqual(result.y.iloc[1], -0.8)

    def test_all_invalid_nonfinite_or_missing_targets_remain_original_rows(self):
        for value in (None, float("inf"), float("nan"), "not-a-return"):
            rows = outcomes_fixture().to_pylist()
            for row in rows:
                row.update(provider_adjusted_return=value, return_status="ok")
            with self.subTest(value=value):
                result = training(outcomes=pd.DataFrame(rows))
                self.assertEqual(len(result.X), 2)
                self.assertTrue(result.y.isna().all())
                self.assertFalse(result.ledger.target_available.any())

    def test_completely_missing_member_ledger_row_rejected_not_survivor_filtered(self):
        rows = outcomes_fixture().to_pylist()
        with self.assertRaises(DataContractError):
            training(outcomes=pa.Table.from_pylist([rows[0], *rows[2:]]))

    def test_extra_nonmember_target_and_duplicate_keys_are_rejected(self):
        rows = outcomes_fixture().to_pylist()
        for changed in ([*rows, rows[0]],
                        [{**rows[0], "source_security_id": "future_nonmember"}, *rows[1:]]):
            with self.subTest(rows=len(changed)), self.assertRaises(DataContractError):
                training(outcomes=pa.Table.from_pylist(changed))

    def test_target_interval_is_strictly_next_month_and_complete_exchange_end(self):
        rows = outcomes_fixture().to_pylist()
        malformed = [
            {"outcome_start": "1999-12-31"},
            {"outcome_start": "2000-01-31", "outcome_end": "2000-01-03"},
            {"outcome_start": "2000-02-01", "outcome_end": "2000-02-29", "outcome_month": "2000-02"},
            {"outcome_end": "2000-01-15"},
            {"outcome_start": "2000-01-04"},
            {"outcome_end": "2000-02-29"},
            {"outcome_month": "1999-12"},
        ]
        for change in malformed:
            changed = [{**rows[0], **change}, *rows[1:]]
            with self.subTest(change=change), self.assertRaises(DataContractError):
                training(outcomes=pa.Table.from_pylist(changed))

    def test_missing_independent_calendar_window_cannot_certify_completed_label(self):
        panel = replace(panel_fixture(), provenance={})
        with self.assertRaises(DataContractError):
            training(panel=panel)

    def test_training_cutoff_order_and_protected_target_values_fail_closed(self):
        for changes in ({"train_start": "2000-03-01"},
                        {"label_observed_through": "2000-03-01"},
                        {"train_end": "2024-01-31"},
                        {"train_start": "2000-1-01"}):
            with self.subTest(changes=changes), self.assertRaises(DataContractError):
                training(**changes)
        rows = outcomes_fixture().to_pylist()
        rows.append({**rows[0], "signal_date": "2023-12-29", "outcome_month": "2024-01",
                     "outcome_start": "2024-01-02", "outcome_end": "2024-01-31"})
        with self.assertRaises(DataContractError):
            training(outcomes=pa.Table.from_pylist(rows))

    def test_duplicate_dataframe_index_does_not_multiply_members(self):
        frame = outcomes_fixture().to_pandas()
        frame.index = [7, 7, 9, 9]
        result = training(outcomes=frame)
        baseline = training()
        self.assertEqual(len(result.X), 2)
        pd.testing.assert_frame_equal(result.X, baseline.X)
        pd.testing.assert_series_equal(result.y, baseline.y)

    def test_signal_boundary_shared_price_is_allowed_without_target_return_overlap(self):
        # MOM1 may end at signal close, and the target starts after that close.
        result = training()
        self.assertEqual(result.provenance["feature_interval"], "ends_at_signal_close")
        self.assertEqual(result.provenance["target_interval"], "(signal_close,next_month_end_close]")
        self.assertTrue(all(day < start for (_, day), start in zip(result.ledger.index, result.ledger.outcome_start)))


class AssemblyIntegrationTests(unittest.TestCase):
    def setUp(self):
        # Use full real worker dictionaries, but bounded metadata/structural data.
        from test_paper_features import SESSIONS, cohort, prices
        self.sessions = SESSIONS
        self.cohort = cohort
        self.price_rows = prices().to_pylist()
        self.calendar = pa.Table.from_pylist([
            {"trading_date": day, "calendar_name": "STRUCTURAL_ONLY", "is_trading_session": True}
            for day in self.sessions])
        for row in self.price_rows:
            value = row["adj_close"]
            row.update(adj_open=value * 0.99, adj_high=value * 1.02, adj_low=value * 0.98,
                       close=value, high=value * 1.02, low=value * 0.98,
                       volume=1000.0, factor=1.0, dividend=0.0, split_ratio=1.0)
        self.prices = pa.Table.from_pylist(self.price_rows)

    def build(self, snapshots=None, daily=None, **kwargs):
        return build_feature_panel(self.prices if daily is None else daily, self.calendar,
                                   [self.cohort()] if snapshots is None else snapshots,
                                   requested_security_ids=("id-a", "id-b"), **kwargs)

    def test_actual_worker_assembly_preserves_keys_and_only_registered_inputs(self):
        panel = self.build()
        self.assertEqual(panel.table.num_rows, 2)
        self.assertEqual(len(panel.table.column_names), 730)
        self.assertEqual(panel.summary["registered_features"], 728)
        self.assertEqual(panel.summary["identity_eligibility_policy"], STRICT_IDENTITY_POLICY)
        self.assertFalse(panel.provenance["outcomes_consumed"])
        self.assertTrue(panel.provenance["cohort_denominator_preserved"])
        self.assertEqual(panel.provenance["target_windows"]["2000-01-31"]["outcome_end"], "2000-02-29")
        self.assertEqual(set(panel.table.column_names), {"source_security_id", "signal_date", *[s["name"] for s in panel.specs]})
        self.assertTrue(predictor_frame(panel, columns=PREDICTORS).notna().all().all())

    def test_paper_indexing_uses_exact_months_with_nonconstant_returns(self):
        from dmv2.paper_features import compute_paper_features
        rows = []
        quote_by_day = {}
        for offset, row in enumerate(self.price_rows):
            modified = {**row, "adj_close": float((offset + 2) ** 2)}
            rows.append(modified)
            quote_by_day[(modified["intrinio_security_id"], modified["trading_date"])] = modified["adj_close"]
        result = compute_paper_features(pa.Table.from_pylist(rows), self.sessions,
            signal_cohorts=[self.cohort()], requested_security_ids=("id-a", "id-b"))
        a = result.table.set_index("source_security_id").loc["a"]
        self.assertAlmostEqual(a.MOM1_raw,
            quote_by_day[("id-a", "2000-01-31")] / quote_by_day[("id-a", "1999-12-31")] - 1)
        for horizon, anchor in ((3, "1999-10-31"), (6, "1999-07-31"),
                                (9, "1999-04-30"), (12, "1999-01-31")):
            self.assertAlmostEqual(a[f"MOM{horizon}_raw"],
                quote_by_day[("id-a", "1999-12-31")] / quote_by_day[("id-a", anchor)] - 1)

    def test_review_flags_and_provider_collisions_block_daily_quote_attachment(self):
        from test_paper_features import member
        for members in (
            [member("a", "id-a"), member("b", "id-b", review=True)],
            [member("a", "id-a"), member("b", "id-b", resolution="pending")],
            [member("a", "id-a"), member("b", "id-a")],
        ):
            with self.subTest(members=members):
                panel = self.build(snapshots=[self.cohort(members=members)])
                coverage = panel.coverage.to_pandas().set_index("source_security_id")
                row = panel.table.to_pandas().set_index("source_security_id").loc["b"]
                technical = [s["name"] for s in panel.specs if s["role"] == "technical"]
                self.assertTrue(row[technical].isna().all())
                self.assertEqual(coverage.at["b", "feature_status"], "identity_unresolved")
                self.assertFalse(coverage.at["b", "model_ready"])

    def test_explicit_provisional_panel_attaches_reviewed_quote_without_erasing_metadata(self):
        from test_paper_features import member
        snapshots = [self.cohort(members=[member("a", "id-a"),
                                        member("b", "id-b", status="needs_review", review=True)])]
        strict = self.build(snapshots=snapshots)
        result = self.build(snapshots=snapshots, identity_policy=PROVISIONAL_IDENTITY_POLICY)
        technical = [s["name"] for s in result.specs if s["role"] == "technical"]
        strict_b = strict.table.to_pandas().set_index("source_security_id").loc["b"]
        row = result.table.to_pandas().set_index("source_security_id").loc["b"]
        self.assertTrue(strict_b[technical].isna().all())
        self.assertTrue(pd.notna(row["px_simple_return_1s"]))
        coverage = result.coverage.to_pandas().set_index("source_security_id")
        self.assertTrue(coverage.at["b", "identity_eligible"])
        self.assertEqual(coverage.at["b", "identity_status"], "needs_review")
        self.assertEqual(coverage.at["b", "source_identity_status"], "needs_review")
        self.assertEqual(coverage.at["b", "identity_eligibility_policy"], PROVISIONAL_IDENTITY_POLICY)
        self.assertTrue(coverage.at["b", "identity_review_required"])
        self.assertEqual(coverage.at["b", "resolution_status"], "resolved")
        self.assertEqual(coverage.at["b", "feature_status"], "partial_features")
        self.assertEqual(result.summary["identity_eligible_rows"], 2)
        self.assertEqual(result.summary["provisional_identity_eligible_rows"], 1)
        self.assertEqual(result.provenance["identity_eligibility_policy"], PROVISIONAL_IDENTITY_POLICY)
        self.assertFalse(result.provenance["knowledge_time_verified"])
        self.assertFalse(coverage.model_ready.any())
        self.assertEqual(result.table.num_rows, strict.table.num_rows)
        audit_fields = set(coverage.columns) - {"source_security_id", "signal_date"}
        self.assertFalse(audit_fields & set(predictor_frame(result).columns))

    def test_provisional_panel_keeps_colliding_or_incomplete_routes_quarantined(self):
        from test_paper_features import member
        variants = [member("b", "id-a", status="needs_review", review=True),
                    member("b", "id-b", review=True, resolution="pending"),
                    {**member("b", "id-b", status="needs_review", review=True),
                     "unified_security_id": None}]
        for reviewed in variants:
            with self.subTest(member=reviewed):
                panel = self.build(snapshots=[self.cohort(members=[member("a", "id-a"), reviewed])],
                                   identity_policy=PROVISIONAL_IDENTITY_POLICY)
                row = panel.table.to_pandas().set_index("source_security_id").loc["b"]
                technical = [s["name"] for s in panel.specs if s["role"] == "technical"]
                self.assertTrue(row[technical].isna().all())
                ledger = panel.coverage.to_pandas().set_index("source_security_id")
                self.assertFalse(ledger.at["b", "identity_eligible"])
                self.assertEqual(ledger.at["b", "feature_status"], "identity_unresolved")
                self.assertEqual(panel.table.num_rows, 2)

    def test_unknown_identity_policy_and_future_financial_rows_still_fail_closed(self):
        with self.assertRaises(DataContractError):
            self.build(identity_policy="waive_everything")
        rows = [*self.price_rows, {**self.price_rows[-1], "trading_date": "2000-02-29"}]
        with self.assertRaises(DataContractError):
            self.build(daily=pa.Table.from_pylist(rows), identity_policy=PROVISIONAL_IDENTITY_POLICY)

    def test_worker_cannot_drop_original_signal_member(self):
        from dmv2.paper_features import compute_paper_features
        original = compute_paper_features(self.prices, self.sessions,
            signal_cohorts=[self.cohort()], requested_security_ids=("id-a", "id-b"))
        dropped = replace(original, table=original.table.iloc[:1].copy())
        with patch("dmv2.paper_features.compute_paper_features", return_value=dropped), self.assertRaises(DataContractError):
            self.build()

    def test_unknown_or_knowledge_time_mode_and_future_financial_rows_are_rejected(self):
        for mode, error in (("knowledge_time", KnowledgeTimeUnavailable), ("future", DataContractError)):
            with self.subTest(mode=mode), self.assertRaises(error):
                self.build(mode=mode)
        rows = [*self.price_rows, {**self.price_rows[-1], "trading_date": "2000-02-29"}]
        with self.assertRaises(DataContractError):
            self.build(daily=pa.Table.from_pylist(rows))


class FeatureLoadingBoundaryTests(unittest.TestCase):
    def setUp(self):
        from test_monthly_loading import fixtures, loaded
        self.loaded = loaded
        self.source = fixtures()
        dates = pd.bdate_range("1997-01-01", "2000-03-31").strftime("%Y-%m-%d").tolist()
        calendar = pa.Table.from_pylist([
            {"trading_date": day, "calendar_name": "XNYS", "is_trading_session": True}
            for day in dates])
        self.source["calendar"] = loaded(calendar, "calendar")
        self.dates = dates
        self.catalog = Mock()
        self.catalog.provenance.return_value = [{"source_key": "structural-test", "manifest_sha256": "test"}]
        self.reader = Mock()
        self.reader.read_metadata.side_effect = [self.source["calendar"], self.source["bridge"], self.source["links"]]
        self.reader.read.return_value = self.source["membership"]
        self.price = Mock()
        self.price.load.return_value = self.source["prices"]
        from dmv2.exchange_sessions import ReconciledExchangeCalendar
        patches = [
            patch("dmv2.feature_loading.BoundedTableReader", return_value=self.reader),
            patch("dmv2.feature_loading.PriceLoader", return_value=self.price),
            patch("dmv2.feature_loading.reconcile_exchange_calendar",
                  side_effect=lambda source, **kwargs: ReconciledExchangeCalendar(source, source, {"test_only": True})),
        ]
        self.reader_class, self.price_class, self.calendar_reconciliation = [item.start() for item in patches]
        for item in patches:
            self.addCleanup(item.stop)

    def load(self, **overrides):
        from dmv2.feature_loading import load_feature_sample
        params = dict(start_month="2000-01", end_month="2000-02", universe="combined")
        params.update(overrides)
        return load_feature_sample(self.catalog, **params)

    def test_exact_warmup_ends_at_signal_never_future_outcome_and_samples_first_cohort(self):
        from dmv2.monthly_loading import DAILY_SAMPLE_COLUMNS
        result = self.load()
        self.assertEqual(result.requested_ids, ("p-b", "p-z"))
        self.assertNotIn("p-a", result.requested_ids)
        first_signal_position = self.dates.index("1999-12-31")
        raw_start = self.dates[first_signal_position - 502]
        self.price.load.assert_called_once_with(("p-b", "p-z"), raw_start, "2000-01-31", DAILY_SAMPLE_COLUMNS)
        self.assertEqual(result.loading_provenance["financial_cutoff"], "2000-01-31")
        self.assertFalse(result.loading_provenance["outcome_tables_consumed"])
        self.assertFalse(result.loading_provenance["daily_context_loaded"])
        self.assertFalse(result.loading_provenance["knowledge_time_verified"])
        self.assertEqual([s.table.num_rows for s in result.snapshots], [3, 3])

    def test_membership_queries_only_declared_signal_dates_and_preserves_future_entrants(self):
        result = self.load()
        call = self.reader.read.call_args
        self.assertEqual(call.kwargs["filters"], {"index_id": ("sp500", "sp400"),
                                               "trading_date": ("1999-12-31", "2000-01-31")})
        self.assertIs(call.kwargs["financial"], False)
        self.assertEqual(call.kwargs["start"], "1999-12-31")
        self.assertEqual(call.kwargs["end"], "2000-01-31")
        before = {row["source_security_id"] for row in result.snapshots[0].table.to_pylist()}
        after = {row["source_security_id"] for row in result.snapshots[1].table.to_pylist()}
        self.assertEqual(before, {"z", "b", "unresolved"})
        self.assertEqual(after, {"a", "b", "unresolved"})
        self.calendar_reconciliation.assert_called_once_with(self.source["calendar"].table,
            start_month="2000-01", end_month="2000-02", warmup_sessions=503)

    def test_invalid_warmup_fails_before_any_reader_or_catalog_access(self):
        for history in (502, 504, True, 503.0):
            with self.subTest(history=history), self.assertRaises(DataContractError):
                self.load(history_sessions=history)
        self.reader_class.assert_not_called()
        self.price_class.assert_not_called()
        self.assertEqual(self.catalog.mock_calls, [])

    def test_insufficient_calendar_history_fails_before_membership_or_prices(self):
        original = self.source["calendar"].table
        calendar = original.slice(original.num_rows - 10)
        self.reader.read_metadata.side_effect = [self.loaded(calendar, "calendar")]
        with self.assertRaises((DataContractError, ValueError)):
            self.load()
        self.reader.read.assert_not_called()
        self.price_class.assert_not_called()

    def test_unresolved_first_cohort_cannot_borrow_later_future_identity(self):
        rows = self.source["links"].table.to_pylist()
        for row in rows:
            if row["unified_security_id"] in ("u-z", "u-b"):
                row["identity_review_required"] = True
        self.reader.read_metadata.side_effect = [self.source["calendar"], self.source["bridge"],
                                                self.loaded(pa.Table.from_pylist(rows), "links")]
        with self.assertRaises(DataContractError):
            self.load()
        self.price_class.assert_not_called()


if __name__ == "__main__":
    unittest.main()
