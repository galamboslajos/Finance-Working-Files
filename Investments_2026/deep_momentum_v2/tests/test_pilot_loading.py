"""Synthetic pilot projections, temporal guards and shared classification slices."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from dmv2.catalog import canonical_sha256
from dmv2.full_panel import _artifact, _hash, _json, _specs, _write_table
from dmv2.full_target_panel import _save_target_state
from dmv2.monthly import MonthWindow
import dmv2.pilot_loading as loading
from dmv2.targets import COHORT_FIELDS, build_monthly_targets
from dmv2.universe import DataContractError
from test_ret import target_fixture


def fixture(root):
    feature_root, target_root = root / "outputs/primary80", root / "outputs/full_targets"
    feature_root.mkdir(parents=True)
    target_root.mkdir(parents=True)
    primary = {"directory": feature_root, "audit": {"build_identity_sha256": "primary-pin", "years": {}}}
    targets = {"directory": target_root, "audit": {"build_identity_sha256": "target-pin", "years": {}}}
    specs = _specs()
    for year in loading.YEARS:
        panel = target_fixture(first=f"{year}-01", count=12,
            missing_month=f"{year}-02" if year in (2011, 2018) else None)
        # A real annual cache ends at its final exchange session, which can
        # precede the calendar-year boundary. Preserve that stricter evidence.
        if year in (2011, 2017):
            observed = "2011-12-30" if year == 2011 else "2017-12-29"
            windows = []
            for row in panel.provenance["month_windows"]:
                end = observed if row["outcome_month"] == f"{year}-12" else row["outcome_end"]
                windows.append(MonthWindow(row["outcome_month"], row["signal_date"],
                    row["outcome_start"], end, (row["outcome_start"], end)))
            outcomes = []
            for row in panel.table.to_pylist():
                outcomes.append({**{name: row[name] for name in
                    (*loading.KEYS, "outcome_month", "outcome_start", "outcome_end")},
                    "outcome_end": observed if row["outcome_month"] == f"{year}-12" else row["outcome_end"],
                    "requested": row["outcome_requested"],
                    "provider_adjusted_return": row["label_return"],
                    "return_status": row["source_return_status"]})
            panel = build_monthly_targets(panel.table.select(list(COHORT_FIELDS)),
                pa.Table.from_pylist(outcomes), label_observed_through=observed, month_windows=windows)
        keys = panel.table.select(list(loading.KEYS)).to_pandas()
        numbers = [float(index % 10 + year - 2010) for index in range(len(keys))]
        features = pa.table({**{name: keys[name].tolist() for name in loading.KEYS},
            **{spec["name"]: numbers for spec in specs}})
        records = []
        for sid, signal in keys.itertuples(index=False, name=None):
            records.append({"source_security_id": sid, "signal_date": signal,
                "intrinio_security_id": "native-" + sid, "identity_status": "resolved",
                "source_identity_status": "resolved", "identity_eligible": True,
                "identity_eligibility_policy": "provisional_unique_resolved_link",
                "identity_eligibility_reason": "synthetic_unique_route",
                "identity_review_required": True, "resolution_status": "resolved",
                "paper_prediction_eligible": True, "requested": True,
                "knowledge_time_verified": False, "accounting_certified": False,
                "available_feature_count": 728, "registered_feature_count": 728,
                "feature_status": "synthetic", "model_ready": False})
        coverage = pa.Table.from_pylist(records)
        fdir, tdir = feature_root / "years" / str(year), target_root / "years" / str(year)
        fdir.mkdir(parents=True)
        tdir.mkdir(parents=True)
        fa, ta = {}, {}
        for name, table in (("features", features), ("feature_coverage", coverage),
                            ("original_cohort", panel.table.select(list(loading.KEYS)))):
            _write_table(fdir, name, table, fa)
        windows = {row["signal_date"]: {name: row[name] for name in
            ("outcome_month", "outcome_start", "outcome_end")}
            for row in panel.provenance["month_windows"]}
        _json(fdir / "audit.json", {"year": year, "build_identity_sha256": "primary-pin",
            "artifacts": fa, "feature_provenance": {"target_windows": windows,
                "signal_dates": sorted(windows), "feature_dictionary_sha256": canonical_sha256(specs)}})
        _save_target_state(tdir, panel, ta)
        _json(tdir / "audit.json", {"year": year, "build_identity_sha256": "target-pin",
            "status": "COMPLETE", "target_artifact_sha256": panel.artifact_sha256, "artifacts": ta})
        primary["audit"]["years"][str(year)] = _artifact(feature_root, fdir / "audit.json")
        targets["audit"]["years"][str(year)] = _artifact(target_root, tdir / "audit.json")
    for run in (primary, targets):
        _json(run["directory"] / "audit.json", run["audit"])
    roles = {"primary": primary, "roles": {"primary_policy": loading.PRIMARY_POLICY}}
    return roles, targets


class PilotLoadingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.roles, cls.targets = fixture(cls.root)
        cls.catalog = SimpleNamespace(root=cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def plan(self):
        return loading.plan_pilot_data(self.catalog, roles=self.roles, target_run=self.targets)

    def rewrite(self, run, year, name, table):
        directory = run["directory"] / "years" / str(year)
        audit_path = directory / "audit.json"
        original = {path: path.read_bytes() for path in
            (directory / (name + ".parquet"), audit_path, run["directory"] / "audit.json")}
        old_year = dict(run["audit"]["years"][str(year)])
        def restore():
            for path, content in original.items():
                path.write_bytes(content)
            run["audit"]["years"][str(year)] = old_year
        self.addCleanup(restore)
        pq.write_table(table, directory / (name + ".parquet"))
        audit = json.loads(audit_path.read_text())
        audit["artifacts"][name] = _artifact(directory, directory / (name + ".parquet"))
        _json(audit_path, audit)
        run["audit"]["years"][str(year)] = _artifact(run["directory"], audit_path)
        _json(run["directory"] / "audit.json", run["audit"])

    def test_metadata_plan_decodes_no_feature_class_or_return_values(self):
        original = pq.ParquetFile.read
        decoded = []
        def guarded(parquet, columns=None, *args, **kwargs):
            self.assertIsNotNone(columns)
            allowed = set(loading.COVERAGE_METADATA) | set(loading.TARGET_METADATA) | {"outcome_month"}
            self.assertLessEqual(set(columns), allowed)
            self.assertNotIn("target_class", columns)
            self.assertNotIn("label_return", columns)
            decoded.append(tuple(columns))
            return original(parquet, columns=columns, *args, **kwargs)
        with patch.object(pq.ParquetFile, "read", guarded), \
             patch.object(loading, "_load_target_state", side_effect=AssertionError("Read numerical targets")), \
             patch.object(loading, "classification_training_slice", side_effect=AssertionError("Built X during planning")):
            plan = self.plan()
        self.assertEqual(len(decoded), 50)
        self.assertEqual(plan.read_plan["rows_original_train"], 960)
        self.assertEqual(plan.read_plan["rows_original_validation"], 240)
        self.assertEqual(plan.read_plan["rows_eligible_train"], 960)
        self.assertEqual(plan.read_plan["rows_labelled_train"], 950)
        self.assertEqual(plan.read_plan["rows_labelled_validation"], 230)
        self.assertEqual(plan.read_plan["signals"], 120)
        self.assertEqual(len(plan.read_plan["excluded_registered_columns"]), 13)
        self.assertGreater(plan.read_plan["projected_compressed_bytes"], 0)
        self.assertFalse(plan.read_plan["financial_values_read"])
        self.assertFalse(plan.read_plan["financial_2020_plus_read"])

    def test_full_load_uses_shared_adapter_for_every_year_and_retains_missing_labels(self):
        plan = self.plan()
        calls = []
        original = loading.classification_training_slice
        def shared(panel, targets, **kwargs):
            calls.append(kwargs)
            self.assertEqual(len(panel.specs), 728)
            self.assertLess(max(targets.table["outcome_end"].to_pylist()), "2020-01-01")
            return original(panel, targets, **kwargs)
        with patch.object(loading, "classification_training_slice", side_effect=shared):
            data = loading.load_pilot_data(self.catalog, plan=plan)
        self.assertEqual(len(calls), 10)
        self.assertEqual(data.train.X.shape, (960, 715))
        self.assertEqual(data.validation.X.shape, (240, 715))
        self.assertEqual(data.train.y.notna().sum(), 950)
        self.assertEqual(data.validation.y.notna().sum(), 230)
        self.assertEqual(data.train.provenance["train_start"], "2010-01-01")
        self.assertEqual(data.train.provenance["train_end"], "2017-12-31")
        self.assertEqual(data.validation.provenance["train_start"], "2018-01-01")
        self.assertEqual(data.validation.provenance["train_end"], "2019-12-31")
        self.assertEqual(len(data.train.provenance["source_month_target_fingerprints"]), 96)
        self.assertEqual(len(data.validation.provenance["source_month_target_fingerprints"]), 24)
        self.assertTrue(data.train.ledger.identity_review_required.all())
        self.assertFalse(set(plan.read_plan["excluded_registered_columns"]) & set(data.train.X))
        self.assertTrue(data.train.X.index.equals(data.train.ledger.index))
        self.assertFalse(data.provenance["financial_2020_plus_read"])
        self.assertFalse(data.provenance["preprocessing_fitted"])

    def test_calendar_year_boundary_never_advances_saved_observation_cutoff(self):
        plan = self.plan()
        calls = []
        original = loading.classification_training_slice
        def observe(panel, targets, **kwargs):
            calls.append(kwargs)
            self.assertLessEqual(kwargs["label_observed_through"], targets.provenance["label_observed_through"])
            return original(panel, targets, **kwargs)
        with patch.object(loading, "classification_training_slice", side_effect=observe):
            data = loading.load_pilot_data(self.catalog, plan=plan)
        year2011 = next(call for call in calls if call["train_start"] == "2011-01-01")
        self.assertEqual(year2011["train_end"], "2011-12-31")
        self.assertEqual(year2011["label_observed_through"], "2011-12-30")
        self.assertEqual(data.train.provenance["train_end"], "2017-12-31")
        self.assertEqual(data.train.provenance["label_observed_through"], "2017-12-29")
        self.assertEqual(data.train.ledger.outcome_end.max(), "2017-12-29")
        self.assertEqual(data.validation.provenance["label_observed_through"], "2019-12-31")

    def test_future_feature_rows_are_rejected_before_numerical_decode(self):
        run = self.roles["primary"]
        path = run["directory"] / "years/2010/features.parquet"
        table = pq.read_table(path)
        dates = table["signal_date"].to_pylist()
        dates[0] = "2020-01-31"
        table = table.set_column(table.schema.get_field_index("signal_date"), "signal_date", pa.array(dates))
        self.rewrite(run, 2010, "features", table)
        with patch.object(loading, "_load_target_state", side_effect=AssertionError("Read numerical targets")):
            with self.assertRaisesRegex(DataContractError, "approved 2010"):
                self.plan()

    def test_future_target_rows_are_rejected_before_numerical_decode(self):
        path = self.targets["directory"] / "years/2010/targets.parquet"
        table = pq.read_table(path)
        table = table.set_column(table.schema.get_field_index("outcome_end"), "outcome_end",
            pa.array(["2020-01-31"] * table.num_rows))
        self.rewrite(self.targets, 2010, "targets", table)
        with patch.object(loading, "_load_target_state", side_effect=AssertionError("Read numerical targets")):
            with self.assertRaisesRegex(DataContractError, "independent windows"):
                self.plan()

    def test_missing_original_member_is_rejected_before_numerical_decode(self):
        run = self.roles["primary"]
        table = pq.read_table(run["directory"] / "years/2010/feature_coverage.parquet")
        self.rewrite(run, 2010, "feature_coverage", table.slice(1))
        with self.assertRaisesRegex(DataContractError, "denominator"):
            self.plan()

    def test_admission_cannot_depend_on_future_target_availability(self):
        table = pq.read_table(self.targets["directory"] / "years/2010/targets.parquet")
        table = table.set_column(table.schema.get_field_index("paper_prediction_eligible"),
            "paper_prediction_eligible", pa.array([False] * table.num_rows))
        self.rewrite(self.targets, 2010, "targets", table)
        with self.assertRaisesRegex(DataContractError, "admission flags"):
            self.plan()

    def test_changed_cached_bytes_reject_before_any_value_read(self):
        plan = self.plan()
        path = self.roles["primary"]["directory"] / "years/2010/features.parquet"
        original = path.read_bytes()
        self.addCleanup(lambda: path.write_bytes(original))
        path.write_bytes(b"changed financial cache")
        with patch.object(pq.ParquetFile, "read", side_effect=AssertionError("Decoded changed values")):
            with self.assertRaises(DataContractError):
                loading.load_pilot_data(self.catalog, plan=plan)

    def test_changed_plan_paths_and_metadata_reject_before_any_value_read(self):
        for field, value in (("feature_directory", self.roles["primary"]["directory"] / "years/2020"),
                             ("target_directory", self.targets["directory"] / "years/2020")):
            plan = self.plan()
            plan.years[2010] = replace(plan.years[2010], **{field: value})
            with self.subTest(field=field), \
                 patch.object(pq.ParquetFile, "read", side_effect=AssertionError("Decoded changed values")), \
                 self.assertRaisesRegex(DataContractError, "metadata or paths"):
                loading.load_pilot_data(self.catalog, plan=plan)
        plan = self.plan()
        plan.provenance["read_plan"]["rows_original_train"] -= 1
        with patch.object(pq.ParquetFile, "read", side_effect=AssertionError("Decoded changed values")):
            with self.assertRaisesRegex(DataContractError, "unchanged approved metadata plan"):
                loading.load_pilot_data(self.catalog, plan=plan)

    def test_nonprimary_policy_is_rejected(self):
        roles = {**self.roles, "roles": {"primary_policy": "available_cohort_90pct"}}
        with self.assertRaisesRegex(DataContractError, "80% primary"):
            loading.plan_pilot_data(self.catalog, roles=roles, target_run=self.targets)

    def test_changed_manifest_after_verified_read_is_rejected(self):
        roles = {**self.roles, "primary": {**self.roles["primary"], "audit": {"changed": True}}}
        with self.assertRaisesRegex(DataContractError, "manifest changed"):
            loading.plan_pilot_data(self.catalog, roles=roles, target_run=self.targets)

    def test_default_plan_never_calls_financial_report_readers(self):
        with patch.object(loading, "read_pilot_context_metadata", return_value=self.roles) as contexts, \
             patch.object(loading, "read_pilot_target_metadata", return_value=self.targets) as targets, \
             patch.object(pd, "read_csv", side_effect=AssertionError("Read financial CSV report")), \
             patch.object(pd, "read_parquet", side_effect=AssertionError("Read financial Parquet report")):
            plan = loading.plan_pilot_data(self.catalog)
        contexts.assert_called_once_with(self.catalog)
        targets.assert_called_once_with(self.catalog)
        self.assertFalse(plan.provenance["saved_financial_report_values_read"])

    def test_role_metadata_checks_full_pins_without_decoding_financial_csv(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            directory = root / "outputs/role"
            directory.mkdir(parents=True)
            dictionary = [{"name": "synthetic"}]
            dictionary_pin = canonical_sha256(dictionary)
            identity = {"configuration": {"context_policy": loading.PRIMARY_POLICY},
                        "feature_dictionary_sha256": dictionary_pin}
            identity_pin = canonical_sha256(identity)
            _json(directory / "build_identity.json", identity)
            _json(directory / "feature_dictionary.json", dictionary)
            (directory / "financial_summary_2023.csv").write_text("year,feature_mean\n2023,123.45\n")
            years = {}
            for year in loading.FULL_TARGET_YEARS:
                annual = directory / "years" / str(year)
                annual.mkdir(parents=True)
                _json(annual / "audit.json", {"year": year, "build_identity_sha256": identity_pin,
                                             "artifacts": {}})
                years[str(year)] = _artifact(directory, annual / "audit.json")
            report = {"status": "COMPLETE", "build_identity_sha256": identity_pin,
                "configuration": identity["configuration"], "years": years,
                "models_fitted": False, "imputation_fitted": False, "model_ready": False,
                "daily_backtest_ready": False,
                "reports": {name: _artifact(directory, directory / name) for name in
                    ("build_identity.json", "feature_dictionary.json", "financial_summary_2023.csv")},
                "summary": {"feature_dictionary_sha256": dictionary_pin,
                    "original_stock_months": 152307, "paper_qualified_stock_months": 140777,
                    "completed_signals": 168}}
            _json(directory / "audit.json", report)
            pin = {"directory": "outputs/role", "audit_sha256": _hash(directory / "audit.json"),
                   "build_identity_sha256": identity_pin, "context_policy": loading.PRIMARY_POLICY}
            with patch.object(pd, "read_csv", side_effect=AssertionError("Decoded CSV summary")), \
                 patch.object(pd, "read_parquet", side_effect=AssertionError("Decoded Parquet summary")), \
                 patch.object(pq.ParquetFile, "read", side_effect=AssertionError("Decoded numerical source")):
                run, loaded_identity = loading._context_run_metadata(SimpleNamespace(root=root), pin)
            self.assertEqual(loaded_identity, identity)
            self.assertNotIn("tables", run)
            (directory / "financial_summary_2023.csv").write_text("changed")
            with self.assertRaises(DataContractError):
                loading._context_run_metadata(SimpleNamespace(root=root), pin)

    def test_target_metadata_verifies_saved_run_without_parquet_reports(self):
        from test_full_target_panel import SavedRunReaderTests
        saved = SavedRunReaderTests(methodName="test_default_reader_loads_only_small_reports_and_leaves_labels_and_daily_lazy")
        saved.setUp()
        self.addCleanup(saved.doCleanups)
        with patch.object(loading, "target_build_identity", return_value=saved.identity), \
             patch.object(pd, "read_csv", side_effect=AssertionError("Read CSV report")), \
             patch.object(pd, "read_parquet", side_effect=AssertionError("Read Parquet report")), \
             patch.object(pq.ParquetFile, "read", side_effect=AssertionError("Read numerical source")):
            run = loading.read_pilot_target_metadata(saved.catalog)
        self.assertEqual(run["audit"], saved.audit)
        self.assertNotIn("tables", run)


if __name__ == "__main__":
    unittest.main()
