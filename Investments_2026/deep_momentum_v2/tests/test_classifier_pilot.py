"""Synthetic orchestration/immutability checks; no research-data estimator fits."""
from copy import deepcopy
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from dmv2.catalog import canonical_sha256
from dmv2.full_panel import _artifact, _hash, _json
from dmv2.universe import DataContractError


PROJECT = Path(__file__).resolve().parents[1]
CONFIGURATION = json.loads((PROJECT / "configs/classifier_pilot.json").read_text())


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.full = importlib.import_module("dmv2.classifier_pilot")
        self.temporary = tempfile.TemporaryDirectory(dir=PROJECT)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "configs").mkdir()
        self.catalog = SimpleNamespace(root=self.root)

    def check(self, value):
        (self.root / "configs/classifier_pilot.json").write_text(json.dumps(value))
        return self.full.configuration(self.catalog)

    def test_frozen_scope_has_exact_four_arms_and_no_automatic_fitting(self):
        actual = self.check(CONFIGURATION)
        self.assertEqual(len(actual["arms"]), 4)
        self.assertEqual({(item["estimator"], item["feature_set"]) for item in actual["arms"]},
                         {("rf", "paper15"), ("xgb", "paper15"), ("rf", "expanded"), ("xgb", "expanded")})
        self.assertEqual(actual["train_outcome_end"], "2017-12-31")
        self.assertEqual(actual["validation_outcome_end"], "2019-12-31")
        self.assertEqual(actual["execution"]["maximum_total_fits"], 4)
        self.assertIs(actual["execution"]["automatic_notebook_fitting"], False)

    def test_chronology_universe_primary80_and_missingness_scope_cannot_change(self):
        changes = (("schema_version", True), ("universe", "sp500"),
                   ("identity_policy", "all_identities_certified"),
                   ("feature_panel_role", "sensitivity"), ("context_policy", "available_cohort_90pct"),
                   ("train_outcome_start", "2000-01-01"), ("train_outcome_end", "2018-12-31"),
                   ("validation_outcome_start", "2017-01-01"), ("validation_outcome_end", "2020-12-31"),
                   ("train_label_observed_through", "2019-12-31"),
                   ("validation_label_observed_through", "2020-12-31"),
                   ("preprocessing_policy", "full_history_median"))
        for key, value in changes:
            changed = deepcopy(CONFIGURATION)
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(DataContractError):
                self.check(changed)

    def test_missing_duplicate_or_additional_arms_cannot_expand_trial_budget(self):
        for arms in (CONFIGURATION["arms"][:3], CONFIGURATION["arms"] + [CONFIGURATION["arms"][0]],
                     [CONFIGURATION["arms"][0]] * 4):
            changed = deepcopy(CONFIGURATION)
            changed["arms"] = arms
            with self.subTest(arms=arms), self.assertRaises(DataContractError):
                self.check(changed)

    def test_execution_flags_cannot_authorize_extra_fits_or_data(self):
        changes = (("workers", 2), ("workers", True), ("estimator_threads", 2),
                   ("fits_per_arm", 2), ("maximum_total_fits", 5),
                   ("parameter_search", True), ("early_stopping", True), ("outer_refit", True),
                   ("outer_test_evaluation", True), ("RET_evaluation", True), ("backtesting", True),
                   ("cloud_mutations", True), ("source_fallback", True),
                   ("automatic_notebook_fitting", True), ("backend", "raw_cloud_data"))
        for key, value in changes:
            changed = deepcopy(CONFIGURATION)
            changed["execution"][key] = value
            with self.subTest(key=key), self.assertRaises(DataContractError):
                self.check(changed)

    def test_resource_limits_are_explicit_positive_integers(self):
        for key in ("maximum_wall_seconds", "maximum_peak_rss_bytes", "maximum_output_bytes",
                    "maximum_projected_read_bytes"):
            for value in (None, 0, -1, True, 1.0):
                changed = deepcopy(CONFIGURATION)
                changed["execution"][key] = value
                with self.subTest(key=key, value=value), self.assertRaises(DataContractError):
                    self.check(changed)

    def test_probability_and_metric_rules_cannot_be_changed_silently(self):
        for section, key, value in (("probability_construction", "canonical_classes", list(range(10))),
                                    ("probability_construction", "repair_clipping_or_renormalization", True),
                                    ("metrics", "log_loss_true_class_probability_floor", 1e-5),
                                    ("metrics", "calibration_fitted", True)):
            changed = deepcopy(CONFIGURATION)
            changed[section][key] = value
            with self.subTest(section=section, key=key), self.assertRaises(DataContractError):
                self.check(changed)

    def test_software_package_pins_must_be_complete_strings(self):
        for missing in CONFIGURATION["packages"]:
            changed = deepcopy(CONFIGURATION)
            del changed["packages"][missing]
            with self.subTest(missing=missing), self.assertRaises(DataContractError):
                self.check(changed)
        for value in (None, "", True, 1.9):
            changed = deepcopy(CONFIGURATION)
            changed["packages"]["scikit-learn"] = value
            with self.subTest(version=value), self.assertRaises(DataContractError):
                self.check(changed)


class LazyImportTests(unittest.TestCase):
    def test_read_only_module_import_does_not_import_estimators(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = str(PROJECT / "src")
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        result = subprocess.run([sys.executable, "-c",
            "import sys; import dmv2.classifier_pilot; "
            "assert 'sklearn' not in sys.modules; assert 'xgboost' not in sys.modules"],
            cwd=PROJECT, env=env, text=True, capture_output=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)


class ExecutionGateTests(unittest.TestCase):
    def setUp(self):
        self.full = importlib.import_module("dmv2.classifier_pilot")
        self.temporary = tempfile.TemporaryDirectory(dir=PROJECT)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.catalog = SimpleNamespace(root=self.root, provenance=lambda: {"synthetic": True})
        self.plan = SimpleNamespace(fingerprint="synthetic-plan", provenance={"synthetic": True},
            read_plan={"projected_compressed_bytes": 100, "estimated_peak_memory_bytes": 1000})
        self.identity = {"configuration": deepcopy(CONFIGURATION), "synthetic": True}

    def test_runtime_package_mismatch_fails_before_any_fit_or_data_load(self):
        with patch.object(self.full, "configuration", return_value=deepcopy(CONFIGURATION)), \
             patch.object(self.full, "version", return_value="different"), \
             patch.object(self.full, "load_pilot_data", side_effect=AssertionError("Read financial data")), \
             patch.object(self.full, "fit_predict_pilot_arm", side_effect=AssertionError("Fitted model")):
            with self.assertRaisesRegex(DataContractError, "dependency"):
                self.full.build_identity(self.catalog, self.plan)

    def execute(self):
        with patch.object(self.full, "configuration", return_value=deepcopy(CONFIGURATION)), \
             patch.object(self.full, "plan_pilot_data", return_value=self.plan), \
             patch.object(self.full, "build_identity", return_value=self.identity), \
             patch.object(self.full, "load_pilot_data", side_effect=AssertionError("Read financial data")), \
             patch.object(self.full, "fit_predict_pilot_arm", side_effect=AssertionError("Fitted model")):
            return self.full.run_classifier_pilot(self.catalog)

    def test_measured_reads_and_ram_are_checked_before_materializing_inputs(self):
        for field, budget in (("projected_compressed_bytes", "maximum_projected_read_bytes"),
                              ("estimated_peak_memory_bytes", "maximum_peak_rss_bytes")):
            original = deepcopy(self.plan.read_plan)
            self.plan.read_plan[field] = CONFIGURATION["execution"][budget] + 1
            with self.subTest(field=field), self.assertRaisesRegex(DataContractError, "budget"):
                self.execute()
            self.plan.read_plan = original
        self.assertFalse((self.root / "outputs").exists())

    def test_incomplete_run_is_preserved_without_automatic_duplicate_fit(self):
        pin = canonical_sha256(self.identity)
        destination = self.root / "outputs/model_pilot" / ("2010-2017__2018-2019-" + pin[7:23])
        destination.mkdir(parents=True)
        sentinel = destination / "ATTEMPT.json"
        sentinel.write_text("preserve attempted fit evidence")
        with self.assertRaisesRegex(DataContractError, "incomplete pilot"):
            self.execute()
        self.assertEqual(sentinel.read_text(), "preserve attempted fit evidence")


def validation_fixture():
    rows, keys, labels = [], [], []
    for offset, month in enumerate(pd.period_range("2018-01", "2019-12", freq="M")):
        signal = (month - 1).end_time.date().isoformat()
        for member in range(4):
            eligible = member < 3
            labelled = eligible and (member == 0 or (offset < 2 and member == 1))
            label = offset % 10 + 1 if labelled else None
            keys.append((f"s{member}", signal))
            labels.append(label)
            rows.append({"prediction_eligible": eligible, "paper_prediction_eligible": eligible,
                         "identity_eligible": True, "identity_review_required": True,
                         "target_available": labelled, "target_class": label,
                         "outcome_month": str(month), "outcome_start": str(month) + "-01",
                         "outcome_end": month.end_time.date().isoformat()})
    index = pd.MultiIndex.from_tuples(keys, names=["source_security_id", "signal_date"])
    ledger = pd.DataFrame(rows, index=index)
    ledger["target_class"] = pd.array(ledger["target_class"], dtype="Int64")
    y = pd.Series(pd.array(labels, dtype="Float64"), index=index)
    prediction_index = ledger.index[ledger.prediction_eligible]
    return ledger, y, prediction_index, np.full((len(prediction_index), 10), .1)


class PredictionAndEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.full = importlib.import_module("dmv2.classifier_pilot")
        self.ledger, self.y, self.prediction_index, self.probabilities = validation_fixture()

    def test_all_original_rows_and_46_missing_labels_are_retained(self):
        result = self.full._predictions(self.ledger, self.probabilities, self.prediction_index)
        self.assertEqual(len(result), 96)
        predicted = result.loc[result.prediction_eligible]
        self.assertEqual(len(predicted), 72)
        self.assertEqual(int((~predicted.target_available).sum()), 46)
        probability_columns = [f"probability_{i}" for i in range(1, 11)]
        self.assertTrue(np.isfinite(predicted[probability_columns].to_numpy()).all())
        self.assertTrue(result.loc[~result.prediction_eligible, probability_columns].isna().all().all())
        self.assertTrue(result.identity_review_required.all())
        self.assertEqual(list(result.prediction_status.value_counts().items()),
                         [("predicted", 72), ("withheld_signal_time_baseline", 24)])
        self.assertTrue(predicted.loc[~predicted.target_available, "target_class"].isna().all())

    def test_prediction_index_cannot_select_only_observed_labels_or_reorder_cohort(self):
        for index in (self.prediction_index[::-1], self.ledger.index[self.ledger.target_available]):
            with self.subTest(length=len(index)), self.assertRaises(DataContractError):
                self.full._predictions(self.ledger, np.full((len(index), 10), .1), index)

    def test_saved_evaluation_aligns_multiindex_labels_after_metadata_reset(self):
        with tempfile.TemporaryDirectory(dir=PROJECT) as temporary:
            directory, artifacts = Path(temporary), {}
            summary = self.full._save_evaluation(directory, self.ledger, self.probabilities,
                                                  self.y, self.prediction_index, artifacts)
            self.assertEqual(summary["prediction_rows"], 72)
            self.assertEqual(summary["missing_label_rows"], 46)
            self.assertEqual(summary["outcome_months"], 24)
            self.assertEqual(summary["observed_label_months"], 24)
            predictions = pd.read_parquet(directory / "predictions.parquet")
            scores = pd.read_parquet(directory / "reports/row_scores.parquet")
            self.assertEqual(len(predictions), 96)
            self.assertEqual(len(scores), 72)
            self.assertEqual(int(scores.true_class.isna().sum()), 46)
            self.assertTrue({"predictions", "monthly", "row_scores", "summary_report"} <= set(artifacts))


class SourcePreservationTests(unittest.TestCase):
    def setUp(self):
        self.full = importlib.import_module("dmv2.classifier_pilot")
        self.temporary = tempfile.TemporaryDirectory(dir=PROJECT)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.path = self.root / "unchanged_source.json"
        self.path.write_text("immutable bytes")
        self.item = _artifact(self.root, self.path)

    def test_source_hash_size_and_approved_path_are_verified(self):
        self.assertEqual(self.full._verify_sources(self.root, {"artifacts": [self.item]}), 1)
        for change in ("size", "hash", "path"):
            item = deepcopy(self.item)
            if change == "size":
                item["size_bytes"] += 1
            elif change == "hash":
                item["sha256"] = "0" * 64
            else:
                item["path"] = "../outside.json"
            with self.subTest(change=change), self.assertRaises(DataContractError):
                self.full._verify_sources(self.root, {"artifacts": [item]})

    def test_empty_or_duplicate_source_manifest_is_rejected(self):
        for items in ([], [self.item, self.item]):
            with self.subTest(items=items), self.assertRaises(DataContractError):
                self.full._verify_sources(self.root, {"artifacts": items})


class SavedReaderTests(unittest.TestCase):
    def setUp(self):
        self.full = importlib.import_module("dmv2.classifier_pilot")
        self.temporary = tempfile.TemporaryDirectory(dir=PROJECT)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.catalog = SimpleNamespace(root=self.root)
        self.identity = {"configuration": deepcopy(CONFIGURATION), "synthetic": True}
        self.pin = canonical_sha256(self.identity)
        self.directory = self.root / "outputs/model_pilot/synthetic"
        self.directory.mkdir(parents=True)
        self.artifacts, self.arms = {}, {}
        source = self.root / "source.json"
        source.write_text("immutable")
        manifest = {"artifacts": [_artifact(self.root, source)]}
        for name, value in (("build_identity.json", self.identity), ("read_plan.json", {}),
                            ("loading_provenance.json", {}), ("source_preservation_manifest.json", manifest)):
            _json(self.directory / name, value)
            self.artifacts[name] = _artifact(self.directory, self.directory / name)
        for relative in ("data/train_ledger", "data/validation_ledger"):
            self.write_artifact(relative, relative + ".parquet")
        for feature_set in ("paper15", "expanded"):
            for name in ("manifest.json", "train_X.npy", "validation_X.npy"):
                relative = f"preprocessors/{feature_set}/{name}"
                self.write_artifact(relative, relative)
        report = self.directory / "reports/comparison.parquet"
        report.parent.mkdir()
        pq.write_table(pa.table({"arm_id": ["synthetic"]}), report)
        self.artifacts["comparison"] = _artifact(self.directory, report)
        for arm in [*self.full.ARMS, {"arm_id": "uniform"}, {"arm_id": "train_prior"}]:
            arm_id = arm["arm_id"]
            directory = self.directory / ("arms" if "estimator" in arm else "refs") / arm_id
            directory.mkdir(parents=True)
            artifacts = {}
            for name in ("predictions", "summary_report", *self.full.REPORT_NAMES):
                path = directory / (name + ".parquet")
                path.write_bytes(b"synthetic not decoded")
                artifacts[name] = _artifact(directory, path)
            if "estimator" in arm:
                for name in ("model", "attempt", *(("booster_configuration",) if arm["estimator"] == "xgb" else ())):
                    path = directory / (name + ".json")
                    path.write_text("synthetic not deserialized")
                    artifacts[name] = _artifact(directory, path)
            audit = {"status": "COMPLETE", "artifacts": artifacts, "build_identity_sha256": self.pin}
            if "estimator" in arm:
                audit.update(arm=arm, build_identity_sha256=self.pin)
            else:
                audit["reference"] = arm_id
            _json(directory / "audit.json", audit)
            self.arms[arm_id] = _artifact(self.directory, directory / "audit.json")
        self.audit = {"status": "COMPLETE", "build_identity_sha256": self.pin, "summary": {},
                      "artifacts": self.artifacts, "arms": self.arms}
        self.save_audit()

    def write_artifact(self, key, relative):
        path = self.directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic not decoded")
        self.artifacts[key] = _artifact(self.directory, path)

    def save_audit(self):
        _json(self.directory / "audit.json", self.audit)
        _json(self.directory.parent / "LATEST.json", {"path": self.directory.name,
            "build_identity_sha256": self.pin, "audit_sha256": _hash(self.directory / "audit.json")})

    def read(self, **kwargs):
        with patch.object(self.full, "plan_pilot_data", return_value=SimpleNamespace(fingerprint="synthetic")), \
             patch.object(self.full, "build_identity", return_value=self.identity):
            return self.full.read_classifier_pilot_run(self.catalog, **kwargs)

    def test_default_reader_only_reads_flat_comparison_and_never_loads_or_fits_model(self):
        def comparison_only(path):
            self.assertEqual(Path(path), self.directory / "reports/comparison.parquet")
            return pd.DataFrame({"arm_id": ["synthetic"]})
        with patch.object(self.full.np, "load", side_effect=AssertionError("Decoded fitted matrices")), \
             patch.object(self.full, "fit_predict_pilot_arm", side_effect=AssertionError("Refitted model")), \
             patch.object(self.full, "load_pilot_data", side_effect=AssertionError("Loaded research values")), \
             patch.object(self.full.pd, "read_parquet", side_effect=comparison_only):
            result = self.read()
        self.assertEqual(set(result["tables"]), {"comparison"})

    def test_stale_pointer_recipe_and_changed_bytes_are_rejected(self):
        pointer_path = self.directory.parent / "LATEST.json"
        original_pointer = json.loads(pointer_path.read_text())
        for key, value in (("audit_sha256", "stale"), ("build_identity_sha256", "stale")):
            _json(pointer_path, {**original_pointer, key: value})
            with self.subTest(key=key), self.assertRaises(DataContractError):
                self.read()
        _json(pointer_path, original_pointer)
        path = self.directory / "data/train_ledger.parquet"
        path.write_bytes(b"changed")
        with self.assertRaises(DataContractError):
            self.read()

    def test_source_preservation_is_verified_before_report_read(self):
        (self.root / "source.json").write_text("changed")
        with patch.object(self.full.pd, "read_parquet", side_effect=AssertionError("Read reports before source check")):
            with self.assertRaises(DataContractError):
                self.read()

    def test_saved_identity_json_must_match_even_if_artifact_pin_is_rewritten(self):
        path = self.directory / "build_identity.json"
        _json(path, {**self.identity, "synthetic": False})
        self.artifacts["build_identity.json"] = _artifact(self.directory, path)
        self.save_audit()
        with self.assertRaises(DataContractError):
            self.read()

    def test_missing_arm_and_incomplete_status_are_rejected(self):
        original = deepcopy(self.audit)
        for change in ("arm", "status"):
            self.audit = deepcopy(original)
            if change == "arm":
                del self.audit["arms"]["rf_extended"]
            else:
                self.audit["status"] = "PARTIAL"
            self.save_audit()
            with self.subTest(change=change), self.assertRaises(DataContractError):
                self.read()

    def test_explicit_reader_cannot_escape_saved_run_parent(self):
        with self.assertRaises(DataContractError):
            self.read(directory=self.root)

    def test_arm_recipe_and_required_model_artifacts_are_verified(self):
        path = self.directory / self.arms["rf_paper15"]["path"]
        original = json.loads(path.read_text())
        for change in ("recipe", "arm", "model", "predictions"):
            altered = deepcopy(original)
            if change == "recipe":
                altered["build_identity_sha256"] = "stale"
            elif change == "arm":
                altered["arm"]["arm_id"] = "rf_extended"
            else:
                del altered["artifacts"][change]
            _json(path, altered)
            self.arms["rf_paper15"] = _artifact(self.directory, path)
            self.save_audit()
            with self.subTest(change=change), self.assertRaises(DataContractError):
                self.read()


if __name__ == "__main__":
    unittest.main()
