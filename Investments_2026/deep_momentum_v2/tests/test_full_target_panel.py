"""Synthetic configuration and immutable saved-state reader contracts."""
from copy import deepcopy
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
import dmv2.full_target_panel as full
from dmv2.full_panel import _artifact, _hash, _json, _write_table
from dmv2.targets import verify_target_panel
from dmv2.universe import DataContractError
from test_ret import target_fixture


PROJECT = Path(__file__).resolve().parents[1]
CONFIGURATION = json.loads((PROJECT / "configs/full_target_build.json").read_text())
REPORT_NAMES = (
    "target_coverage_month", "target_coverage_year", "target_coverage_period", "target_status_month",
    "return_observation_status_month", "target_class_month", "target_class_year", "target_class_period",
    "rolling_training_coverage", "rolling_training_classes", "ret_coverage_signal", "ret_coverage_year",
    "ret_month_class_grid", "ret_class_payoffs",
)


def write_configuration(root, value=None):
    (root / "configs").mkdir(exist_ok=True)
    _json(root / "configs/full_target_build.json", CONFIGURATION if value is None else value)


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=PROJECT)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.catalog = SimpleNamespace(root=self.root)

    def check_configuration(self, value):
        write_configuration(self.root, value)
        return full.configuration(self.catalog)

    def test_approved_configuration_is_cache_only_and_does_not_fit(self):
        actual = self.check_configuration(CONFIGURATION)
        self.assertEqual(actual["execution"]["backend"], "local_verified_cache")
        self.assertIs(actual["execution"]["raw_source_fallback"], False)
        self.assertIs(actual["execution"]["model_fitting"], False)

    def test_unsafe_execution_modes_are_rejected(self):
        for key, value in (("workers", 2), ("workers", True), ("backend", "cloud"),
                           ("raw_source_fallback", True), ("model_fitting", True),
                           ("imputation_fitting", True), ("backtesting", True),
                           ("cloud_mutations", True), ("model_fitting", 0)):
            cfg = deepcopy(CONFIGURATION)
            cfg["execution"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(DataContractError):
                self.check_configuration(cfg)

    def test_scope_and_year_types_cannot_change(self):
        for key, value in (("start_year", 2000), ("end_year", 2024), ("start_year", 2010.0),
                           ("end_year", 2023.0), ("schema_version", True),
                           ("universe", "sp500"), ("source", "raw_source"),
                           ("identity_policy", "accept_everything")):
            cfg = deepcopy(CONFIGURATION)
            cfg[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(DataContractError):
                self.check_configuration(cfg)

    def test_resource_ceilings_are_explicit_positive_integers(self):
        for key in ("maximum_wall_seconds", "maximum_peak_rss_bytes", "maximum_output_bytes",
                    "maximum_projected_read_bytes"):
            for value in (None, 0, -1, True, 1.0):
                cfg = deepcopy(CONFIGURATION)
                cfg["execution"][key] = value
                with self.subTest(key=key, value=value), self.assertRaises(DataContractError):
                    self.check_configuration(cfg)

    def test_ret_calendar_and_class_grid_are_fixed(self):
        for key, value in (("calendar_months", 119), ("all_ten_classes_every_month", False),
                           ("missing_grid_policy", "average_available"),
                           ("canonical_outcome_years", [2020]), ("all_feature_signals", 167)):
            cfg = deepcopy(CONFIGURATION)
            cfg["ret_audit"][key] = value
            with self.subTest(key=key), self.assertRaises(DataContractError):
                self.check_configuration(cfg)


class MetadataPlanningTests(unittest.TestCase):
    def plans(self):
        plans = {}
        for index, year in enumerate(full.YEARS):
            windows = tuple(SimpleNamespace(signal_date=f"synthetic-signal-{index:02d}-{month:02d}")
                            for month in range(12))
            plans[year] = SimpleNamespace(windows=windows, read_plan={
                "original_rows": 152307 if index == 0 else 0,
                "paper_qualified_rows": 140777 if index == 0 else 0,
                "projected_compressed_bytes": 100, "estimated_peak_memory_bytes": 1000,
                "daily_cache_rows": 200})
        return plans

    def execute(self, plans, *, limits=None):
        identity = {"configuration": deepcopy(CONFIGURATION)}
        if limits:
            identity["configuration"]["execution"].update(limits)
        roles = {"primary": {"synthetic": True}, "sensitivity": {"synthetic": True}}
        with patch.object(full, "build_identity", return_value=identity), \
             patch.object(full, "read_context_panel_roles", return_value=roles), \
             patch.object(full, "plan_cached_target_year", side_effect=lambda run, year: plans[year]), \
             patch.object(full, "build_cached_target_year", side_effect=AssertionError("Materialized a planning input")), \
             patch.object(full.pq, "read_table", side_effect=AssertionError("Read financial values during planning")):
            return full.plan_full_targets(SimpleNamespace(root=PROJECT))

    def test_planning_only_combines_metadata_and_records_original_denominators(self):
        _, _, _, totals = self.execute(self.plans())
        self.assertEqual(totals["signals"], 168)
        self.assertEqual(totals["original_rows"], 152307)
        self.assertEqual(totals["paper_qualified_rows"], 140777)
        self.assertEqual(totals["projected_read_bytes"], 1400)
        self.assertIs(totals["financial_values_read"], False)
        self.assertIs(totals["raw_source_reads"], False)

    def test_changed_original_scope_and_duplicate_signals_are_rejected(self):
        for field in ("original_rows", "paper_qualified_rows"):
            plans = self.plans()
            plans[2010].read_plan[field] += 1
            with self.subTest(field=field), self.assertRaises(DataContractError):
                self.execute(plans)
        plans = self.plans()
        plans[2011].windows = plans[2010].windows
        with self.assertRaises(DataContractError):
            self.execute(plans)

    def test_projected_reads_and_memory_cannot_exceed_resource_ceilings(self):
        for limits in ({"maximum_projected_read_bytes": 1399}, {"maximum_peak_rss_bytes": 999}):
            with self.subTest(limits=limits), self.assertRaises(DataContractError):
                self.execute(self.plans(), limits=limits)


class SavedTargetStateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=PROJECT)
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def test_saved_state_roundtrip_preserves_policy_labels_and_fingerprints(self):
        for tied_month in (None, "2010-01"):
            panel = target_fixture(count=2, tied_month=tied_month)
            artifacts = {}
            full._save_target_state(self.directory, panel, artifacts)
            self.assertEqual(set(artifacts), {"targets", "target_diagnostics", "target_state"})
            loaded = full._load_target_state(self.directory)
            verify_target_panel(loaded)
            self.assertTrue(panel.table.equals(loaded.table))
            self.assertTrue(panel.diagnostics.equals(loaded.diagnostics))
            self.assertEqual(loaded.artifact_sha256, panel.artifact_sha256)

    def test_saved_provenance_and_target_values_cannot_be_modified(self):
        for mutate in ("provenance", "target_values"):
            panel = target_fixture(count=1)
            full._save_target_state(self.directory, panel, {})
            if mutate == "provenance":
                path = self.directory / "target_state.json"
                state = json.loads(path.read_text())
                state["provenance"]["rows_retained"] += 1
                _json(path, state)
            else:
                path = self.directory / "targets.parquet"
                changed = panel.table.set_column(panel.table.schema.get_field_index("label_return"),
                                                 "label_return", pa.array([0.1] * panel.table.num_rows))
                pq.write_table(changed, path)
            with self.subTest(mutate=mutate), self.assertRaises(DataContractError):
                full._load_target_state(self.directory)

    def test_preserved_manifest_detects_changed_bytes_and_path_escape(self):
        path = self.directory / "immutable.json"
        path.write_text("original")
        manifest = {"artifacts": [_artifact(self.directory, path)], "scope": "synthetic immutable inputs"}
        self.assertEqual(full.verify_preserved_inputs(self.directory, manifest), 1)
        path.write_text("modified")
        with self.assertRaises(DataContractError):
            full.verify_preserved_inputs(self.directory, manifest)
        manifest["artifacts"][0]["path"] = "../outside.json"
        with self.assertRaises(DataContractError):
            full.verify_preserved_inputs(self.directory, manifest)

    def test_preserved_manifest_cannot_be_an_empty_noop(self):
        with self.assertRaises(DataContractError):
            full.verify_preserved_inputs(self.directory, {"artifacts": [], "scope": "nothing"})
        path = self.directory / "immutable.json"
        path.write_text("original")
        entry = _artifact(self.directory, path)
        with self.assertRaises(DataContractError):
            full.verify_preserved_inputs(self.directory, {"artifacts": [entry, entry]})


class SavedRunReaderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=PROJECT)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.catalog = SimpleNamespace(root=self.root)
        self.identity = {"schema_version": 1, "configuration": deepcopy(CONFIGURATION), "synthetic_identity": True}
        self.pin = canonical_sha256(self.identity)
        self.directory = self.root / "outputs/full_target_panel/2010-2023-synthetic"
        self.directory.mkdir(parents=True)
        self.artifacts, self.years = {}, {}
        panel = target_fixture(count=1)
        full._save_target_state(self.directory, panel, self.artifacts)
        _write_table(self.directory, "monthly_returns", pa.table({"synthetic": [True]}), self.artifacts)
        for name in REPORT_NAMES:
            report = self.directory / "reports" / f"{name}.parquet"
            report.parent.mkdir(exist_ok=True)
            pq.write_table(pa.table({"synthetic": [True]}), report)
            self.artifacts[name] = _artifact(self.directory, report)
        (self.root / "immutable_input.json").write_text("unchanged source")
        self.preserved = {"artifacts": [_artifact(self.root, self.root / "immutable_input.json")],
                          "scope": "synthetic immutable inputs"}
        for name, data in (("build_identity.json", self.identity), ("source_plan.json", {"synthetic": True}),
                           ("preserved_source_manifest.json", self.preserved)):
            _json(self.directory / name, data)
            self.artifacts[name] = _artifact(self.directory, self.directory / name)
        for year in full.YEARS:
            annual = self.directory / "years" / str(year)
            annual.mkdir(parents=True)
            annual_artifacts = {}
            full._save_target_state(annual, panel, annual_artifacts)
            _write_table(annual, "monthly_returns", pa.table({"synthetic": [True]}), annual_artifacts)
            _json(annual / "monthly_provenance.json", {"synthetic": True})
            annual_artifacts["monthly_provenance"] = _artifact(annual, annual / "monthly_provenance.json")
            _json(annual / "audit.json", {"year": year, "status": "COMPLETE", "build_identity_sha256": self.pin,
                                         "artifacts": annual_artifacts})
            self.years[str(year)] = _artifact(self.directory, annual / "audit.json")
        self.audit = {"status": "COMPLETE", "build_identity_sha256": self.pin,
                      "years": self.years, "artifacts": self.artifacts, "summary": {"synthetic": True}}
        self.write_audit()

    def write_audit(self):
        _json(self.directory / "audit.json", self.audit)
        _json(self.directory.parent / "LATEST.json", {"path": self.directory.name,
            "build_identity_sha256": self.pin, "audit_sha256": _hash(self.directory / "audit.json")})

    def read(self, **kwargs):
        with patch.object(full, "build_identity", return_value=self.identity):
            return full.read_full_target_run(self.catalog, **kwargs)

    def test_default_reader_loads_only_small_reports_and_leaves_labels_and_daily_lazy(self):
        paths = []
        def small_report(path):
            path = Path(path)
            paths.append(path)
            self.assertIn(path.stem, full.SMALL_REPORTS)
            return pd.DataFrame({"synthetic": [True]})
        with patch.object(full, "_load_target_state", side_effect=AssertionError("Read labels implicitly")), \
             patch.object(full.pq, "read_table", side_effect=AssertionError("Read financial Arrow values implicitly")), \
             patch.object(full.pd, "read_parquet", side_effect=small_report):
            run = self.read()
        self.assertEqual(set(run["tables"]), set(full.SMALL_REPORTS))
        self.assertEqual(len(paths), len(full.SMALL_REPORTS))

    def test_pointer_audit_hash_and_identity_changes_are_rejected(self):
        pointer_path = self.directory.parent / "LATEST.json"
        pointer = json.loads(pointer_path.read_text())
        for field, replacement in (("audit_sha256", "0" * 64), ("build_identity_sha256", "sha256:" + "0" * 64)):
            changed = {**pointer, field: replacement}
            _json(pointer_path, changed)
            with self.subTest(field=field), self.assertRaises(DataContractError):
                self.read()

    def test_explicit_reader_cannot_escape_saved_run_parent(self):
        with self.assertRaises(DataContractError):
            self.read(directory=self.root)

    def test_saved_build_identity_json_must_match_current_identity_even_if_manifest_rehashed(self):
        path = self.directory / "build_identity.json"
        _json(path, {**self.identity, "synthetic_identity": False})
        self.artifacts["build_identity.json"] = _artifact(self.directory, path)
        self.write_audit()
        with self.assertRaises(DataContractError):
            self.read()

    def test_incomplete_scope_and_missing_global_artifacts_are_rejected(self):
        original = deepcopy(self.audit)
        for mutate in ("incomplete_status", "missing_year", "missing_targets", "missing_grid", "missing_identity"):
            self.audit = deepcopy(original)
            if mutate == "incomplete_status":
                self.audit["status"] = "PARTIAL"
            elif mutate == "missing_year":
                del self.audit["years"]["2023"]
            else:
                key = {"missing_targets": "targets", "missing_grid": "ret_month_class_grid",
                       "missing_identity": "build_identity.json"}[mutate]
                del self.audit["artifacts"][key]
            self.write_audit()
            with self.subTest(mutate=mutate), self.assertRaises(DataContractError):
                self.read()

    def test_annual_checkpoint_year_and_recipe_must_match_its_directory(self):
        path = self.directory / "years/2010/audit.json"
        original = json.loads(path.read_text())
        for field, value in (("year", 2011), ("year", 2010.0), ("build_identity_sha256", "stale"), ("status", "PARTIAL")):
            _json(path, {**original, field: value})
            self.years["2010"] = _artifact(self.directory, path)
            self.write_audit()
            with self.subTest(field=field), self.assertRaises(DataContractError):
                self.read()

    def test_target_file_hash_and_preserved_input_bytes_are_verified_before_reporting(self):
        for change in ("target_file", "preserved_source"):
            path = self.directory / "targets.parquet" if change == "target_file" else self.root / "immutable_input.json"
            original = path.read_bytes()
            path.write_bytes(b"changed")
            with self.subTest(change=change), self.assertRaises(DataContractError):
                self.read()
            path.write_bytes(original)

    def test_explicit_label_read_uses_frozen_target_state_after_artifact_verification(self):
        run = self.read()
        panel = full.read_full_targets(run)
        self.assertEqual(panel.table.num_rows, 10)
        self.assertEqual(panel.provenance["rows_retained"], 10)
        self.assertEqual(panel.table["target_available"].to_pylist(), [True] * 10)


if __name__ == "__main__":
    unittest.main()
