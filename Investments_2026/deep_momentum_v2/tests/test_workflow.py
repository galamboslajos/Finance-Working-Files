"""Notebook safety contracts using metadata fixtures and mocked source reads.

No source file, market observation, model or full panel is read by these tests.
The quote fixture is empty, so it tests missing/unrequested accounting without
fabricating financial observations.
"""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import pyarrow as pa

from dmv2.data import LoadedUniverse
from dmv2.loading import LoadedTable
from dmv2.universe import EXCLUDED_SCHEMA, UNIVERSE_SCHEMA, UniverseSnapshot
from dmv2.workflow import (
    InspectionRun, load_workflow, run_inspection, save_inspection,
    validate_workflow,
)


def configuration():
    path = Path(__file__).resolve().parents[1] / "configs/research_workflow.json"
    return json.loads(path.read_text())


def source_fixture():
    rows = [
        {"source_security_id": "member-a", "intrinio_security_id": "id-a",
         "identity_status": "resolved"},
        {"source_security_id": "member-b", "intrinio_security_id": "id-b",
         "identity_status": "resolved"},
        {"source_security_id": "member-c", "intrinio_security_id": "id-c",
         "identity_status": "needs_review"},
    ]
    for row in rows:
        row.update(as_of=configuration()['inspection']['as_of'],unified_security_id='u-'+row['source_security_id'],
                   identity_review_required=row['identity_status']=='needs_review',resolution_status='resolved')
    snapshot = UniverseSnapshot(
        pa.Table.from_pylist(rows, schema=UNIVERSE_SCHEMA),
        pa.Table.from_pylist([], schema=EXCLUDED_SCHEMA),
        configuration()["inspection"]["as_of"], "combined", {"knowledge_time_verified": False},
        {"constituents": 3, "identity_status_counts": {"resolved": 2, "needs_review": 1}},
    )
    return LoadedUniverse(snapshot, {"fixture": "metadata only"}, {"fixture_rows": 3})


class WorkflowValidationTests(unittest.TestCase):
    def test_selected_configuration_is_valid_and_keeps_price_stage_opt_in(self):
        settings = configuration()
        validate_workflow(settings)
        self.assertEqual(settings["research_start"], "2010-01-01")
        self.assertEqual(settings["training"]["mode"], "rolling")
        self.assertEqual(settings["training"]["train_years"], 10)
        self.assertFalse(settings["inspection"]["run_price_sample"])
        self.assertIsNone(settings["universe"]["primary"])

    def test_policy_is_explicit_and_legacy_configuration_remains_strict(self):
        from dmv2.workflow import workflow_identity_policy
        from dmv2.identity_policy import STRICT_IDENTITY_POLICY,PROVISIONAL_IDENTITY_POLICY
        settings=configuration();validate_workflow(settings)
        self.assertEqual(workflow_identity_policy(settings),PROVISIONAL_IDENTITY_POLICY)
        legacy=deepcopy(settings);legacy['schema_version']=3;legacy.pop('identity')
        validate_workflow(legacy)
        self.assertEqual(workflow_identity_policy(legacy),STRICT_IDENTITY_POLICY)

    def test_identity_policy_cannot_claim_verified_or_missing_waiver(self):
        for field,value in [('accept_identity_uncertainty',False),('accept_identity_uncertainty',1),
            ('research_status','VERIFIED'),('policy','ignore_all'),('policy_version',True)]:
            settings=configuration();settings['identity'][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):validate_workflow(settings)

    def test_forbidden_execution_and_invalid_dates_fail_before_source_reads(self):
        cases = []
        for flag in ("full_panel_build_enabled", "model_execution_enabled"):
            settings = configuration()
            settings["execution"][flag] = True
            cases.append(settings)
        for day in ("2024-01-31", "1999-12-31", "2000-02-30", "20000131"):
            settings = configuration()
            settings["inspection"]["as_of"] = day
            cases.append(settings)
        settings = configuration()
        settings["training"]["last_test_year"] = 2024
        cases.append(settings)
        with patch("dmv2.workflow.load_universe") as universe_loader, \
                patch("dmv2.workflow.PriceLoader") as price_loader:
            for settings in cases:
                with self.subTest(settings=settings), self.assertRaises(ValueError):
                    run_inspection(Mock(), settings)
            universe_loader.assert_not_called()
            price_loader.assert_not_called()

    def test_price_stage_requires_membership_audit_before_any_source_read(self):
        settings = configuration()
        settings["inspection"].update(run_universe_audit=False, run_price_sample=True)
        with patch("dmv2.workflow.load_universe") as loader:
            with self.assertRaisesRegex(ValueError, "requires"):
                run_inspection(Mock(), settings)
            loader.assert_not_called()

    def test_nonboolean_controls_and_invalid_read_budget_are_rejected(self):
        for field, value in (("run_universe_audit", "true"),
                             ("run_price_sample", 1), ("persist_audit", None)):
            settings = configuration()
            settings["inspection"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_workflow(settings)
        settings = configuration()
        settings["inspection"]["read_budget"]["max_scanned_rows"] = 0
        with self.assertRaises(ValueError):
            validate_workflow(settings)

    def test_missing_and_nonobject_nested_settings_fail_with_value_error(self):
        for field in ("training", "warmup", "universe", "inspection", "execution"):
            for value in (None, []):
                settings = configuration()
                settings[field] = value
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    validate_workflow(settings)
        settings = configuration()
        del settings["training"]["last_test_year"]
        with self.assertRaises(ValueError):
            validate_workflow(settings)

    def test_json_duplicate_keys_and_nonfinite_constants_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "configs").mkdir()
            path = root / "configs/research_workflow.json"
            for payload in ('{"schema_version":1,"schema_version":1}',
                            '{"schema_version":NaN}'):
                path.write_text(payload)
                with self.subTest(payload=payload), self.assertRaises(ValueError):
                    load_workflow(root)


class InspectionOrchestrationTests(unittest.TestCase):
    def setUp(self):
        self.catalog = Mock()
        self.catalog.provenance.return_value = {"sources": [{"source_key": "fixture"}]}

    def test_disabled_audit_does_not_touch_sources(self):
        settings = configuration()
        settings["inspection"]["run_universe_audit"] = False
        with patch("dmv2.workflow.load_universe") as loader:
            self.assertIsNone(run_inspection(self.catalog, settings))
            loader.assert_not_called()
            self.catalog.provenance.assert_not_called()

    def test_universe_only_preserves_whole_ledger_and_avoids_prices(self):
        with patch("dmv2.workflow.load_universe", return_value=source_fixture()) as loader, \
                patch("dmv2.workflow.PriceLoader") as price_loader:
            result = run_inspection(self.catalog, configuration())
        self.assertEqual(result.snapshot.table.num_rows, 3)
        self.assertEqual(len(result.report["constituent_ledger"]), 3)
        self.assertFalse(result.report["pit"]["knowledge_time_verified"])
        self.assertIsNone(result.prices)
        price_loader.assert_not_called()
        self.assertEqual(loader.call_args.kwargs["as_of"], configuration()["inspection"]["as_of"])
        self.assertEqual(loader.call_args.kwargs["mode"], "historical_reconstruction")

    def test_partial_price_request_keeps_missing_unrequested_and_review_states(self):
        settings = configuration()
        settings['identity'].update(policy='strict_unique_unflagged',accept_identity_uncertainty=False)
        settings["inspection"].update(run_price_sample=True, price_sample_size=1)
        empty_quotes = pa.Table.from_pylist([], schema=pa.schema([
            ("intrinio_security_id", pa.string()), ("trading_date", pa.string()),
            ("close", pa.float64()),
        ]))
        price_result = LoadedTable(empty_quotes, {"fixture": True}, {"output_rows": 0})
        with patch("dmv2.workflow.load_universe", return_value=source_fixture()), \
                patch("dmv2.workflow.PriceLoader") as price_loader:
            price_loader.return_value.load.return_value = price_result
            result = run_inspection(self.catalog, settings)
        self.assertEqual(price_loader.return_value.load.call_args.args[:3],
                         (["id-a"], settings["inspection"]["as_of"], settings["inspection"]["as_of"]))
        self.assertEqual(result.quote_coverage.table.num_rows, 3)
        self.assertEqual(result.quote_coverage.summary["quote_status_counts"],
                         {"missing_price": 1, "not_requested": 1, "identity_unresolved": 1})
        self.assertEqual(len(result.report["constituent_ledger"]), 3)
        self.assertEqual(result.report["price_sampling"]["rows"], [])

    def test_no_resolved_price_sample_fails_without_reading_any_quotes(self):
        settings = configuration()
        settings["inspection"]["run_price_sample"] = True
        loaded = source_fixture()
        rows = loaded.snapshot.table.to_pylist()
        for row in rows:
            row["identity_status"] = "needs_review"
            row['resolution_status']='not_resolved'
        snapshot = UniverseSnapshot(pa.Table.from_pylist(rows, schema=UNIVERSE_SCHEMA),
                                    loaded.snapshot.excluded, loaded.snapshot.as_of,
                                    loaded.snapshot.universe, {}, {"constituents": 3})
        loaded = LoadedUniverse(snapshot, {}, {})
        with patch("dmv2.workflow.load_universe", return_value=loaded), \
                patch("dmv2.workflow.PriceLoader") as price_loader:
            with self.assertRaisesRegex(ValueError, "No uniquely mapped"):
                run_inspection(self.catalog, settings)
            price_loader.assert_not_called()


class PrivatePersistenceTests(unittest.TestCase):
    def inspection(self):
        return InspectionRun(SimpleNamespace(as_of="2000-01-31"), None, None, {
            "workflow_sha256": "sha256:" + "0" * 64,
            "source_pins": {"sources": [{"source_key": "fixture"}]},
            "code_provenance": {"worker_sha256": "sha256:" + "1" * 64,
                                "runtime": {"python": "fixture", "pyarrow": "fixture"}},
            "summary": {"constituents": 3},
        })

    def test_repeat_save_is_idempotent_without_overwriting_original_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inspection = self.inspection()
            path = save_inspection(root, inspection)
            before = (path.stat().st_mtime_ns, path.read_bytes())
            self.assertEqual(save_inspection(root, inspection), path)
            self.assertEqual((path.stat().st_mtime_ns, path.read_bytes()), before)
            self.assertTrue(path.is_relative_to(root.resolve() / "outputs/notebook_audits"))
            self.assertEqual(list(path.parent.glob("*.tmp")), [])

    def test_changed_result_under_same_pins_fails_and_preserves_old_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inspection = self.inspection()
            path = save_inspection(root, inspection)
            before = path.read_bytes()
            changed = deepcopy(inspection.report)
            changed["summary"]["constituents"] = 2
            with self.assertRaisesRegex(ValueError, "changed"):
                save_inspection(root, InspectionRun(inspection.snapshot, None, None, changed))
            self.assertEqual(path.read_bytes(), before)

    def test_new_source_pin_gets_a_separate_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inspection = self.inspection()
            first = save_inspection(root, inspection)
            changed = deepcopy(inspection.report)
            changed["source_pins"]["sources"][0]["source_key"] = "another-fixture"
            second = save_inspection(root, InspectionRun(inspection.snapshot, None, None, changed))
            self.assertNotEqual(first, second)
            self.assertTrue(first.exists() and second.exists())

    def test_changed_worker_fingerprint_gets_separate_artifact_for_code_development(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inspection = self.inspection()
            first = save_inspection(root, inspection)
            changed = deepcopy(inspection.report)
            changed["code_provenance"]["worker_sha256"] = "sha256:" + "2" * 64
            second = save_inspection(root, InspectionRun(inspection.snapshot, None, None, changed))
            self.assertNotEqual(first, second)
            self.assertTrue(first.exists() and second.exists())

    def test_output_directory_symlink_cannot_escape_root(self):
        with tempfile.TemporaryDirectory() as temporary, tempfile.TemporaryDirectory() as external:
            root = Path(temporary)
            (root / "outputs").symlink_to(Path(external), target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "leaves"):
                save_inspection(root, self.inspection())
            self.assertEqual(list(Path(external).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
