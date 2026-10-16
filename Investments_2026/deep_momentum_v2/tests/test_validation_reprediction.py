"""Synthetic inference-only checks; no research samples or estimator fits."""
import ast
import copy
import inspect
import json
from pathlib import Path
import shutil
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
from dmv2.full_panel import _artifact, _json
from dmv2.paper_features import PAPER_FEATURE_NAMES
from dmv2.pilot_loading import COVERAGE_METADATA, TARGET_METADATA
from dmv2.universe import DataContractError
import dmv2.validation_reprediction as worker
from test_fallback_models import fallback_fixture, PARENT

NEW = "sha256:" + "b" * 64


class FrozenEstimator:
    def __init__(self, *, kind="rf", order=None, width=2, margins=None):
        self.classes_ = np.array(order if order is not None else
            list(range(1, 11)) if kind == "rf" else list(range(10)))
        self.n_features_in_ = width
        self.calls = []
        self.margins = margins

    def fit(self, *args, **kwargs):
        raise AssertionError("No fit is allowed")

    def predict_proba(self, matrix):
        self.calls.append("predict_proba")
        vector = np.arange(1, 11, dtype=float)
        return np.broadcast_to(vector / vector.sum(), (len(matrix), 10)).copy()

    def predict(self, matrix, *, output_margin):
        assert output_margin is True
        self.calls.append("output_margin")
        return np.broadcast_to(self.margins, (len(matrix), 10)).copy()

    def load_model(self, path):
        self.calls.append("load_model")


def preprocessing_state():
    names = list(PAPER_FEATURE_NAMES) + ["px_simple_return_1s"]
    manifest = {"schema_version": 1,
        "preprocessing_policy": "train_partition_median_missing_indicator_v1",
        "input_columns": names, "retained_columns": names,
        "output_columns": [column for name in names for column in (name, name + "__missing")],
        "medians": {name: float(i) for i, name in enumerate(names)},
        "dropped_columns": {}, "fit_rows": 3,
        "fit_keys_sha256": "sha256:" + "c" * 64,
        "train_start": "2010-01-01", "train_end": "2017-12-31",
        "label_observed_through": "2017-12-29", "training_only_schema_selection": True,
        "scaling": None, "clipping": None, "winsorization": None}
    return {"manifest": manifest, "fingerprint": canonical_sha256(manifest)}


class SavedInferenceTests(unittest.TestCase):
    def test_rf_class_order_is_canonically_aligned_and_fit_is_not_called(self):
        model = FrozenEstimator(order=list(range(10, 0, -1)))
        result = worker.predict_saved_estimator(model, "rf", np.ones((3, 2), dtype=np.float32))
        self.assertEqual(model.calls, ["predict_proba"])
        np.testing.assert_array_equal(result[0], np.arange(10, 0, -1) / 55)

    def test_xgb_uses_native_margins_and_original_float64_softmax(self):
        margins = np.arange(10, dtype=np.float32) + 1000
        model = FrozenEstimator(kind="xgb", margins=margins)
        result = worker.predict_saved_estimator(model, "xgb", np.ones((2, 2), dtype=np.float32),
            expected_native_ids=list(range(10)))
        expected = np.exp(margins.astype(np.float64) - float(margins.max()))
        np.testing.assert_array_equal(result[0], expected / expected.sum())
        self.assertEqual(result.dtype, np.float64)
        self.assertEqual(model.calls, ["output_margin"])

    def test_empty_matrix_never_calls_estimator(self):
        model = FrozenEstimator()
        result = worker.predict_saved_estimator(model, "rf", np.ones((0, 2), dtype=np.float32))
        self.assertEqual(result.shape, (0, 10))
        self.assertFalse(model.calls)

    def test_nonfinite_dtype_width_and_mapping_fail_before_inference(self):
        for change in ("nonfinite", "float64", "bool", "width", "mapping", "kind"):
            model = FrozenEstimator()
            values, kind = np.ones((2, 2), dtype=np.float32), "rf"
            if change == "nonfinite": values[0, 0] = np.nan
            elif change == "float64": values = values.astype(np.float64)
            elif change == "bool": values = values.astype(bool)
            elif change == "width": model.n_features_in_ = 3
            elif change == "mapping": model.classes_ = np.arange(10)
            else: kind = "unknown"
            with self.subTest(change=change), self.assertRaises(DataContractError):
                worker.predict_saved_estimator(model, kind, values)
            self.assertFalse(model.calls)

    def test_xgb_bad_margins_are_rejected(self):
        for margins in (np.full(10, np.nan), np.arange(9), np.full(10, np.inf)):
            model = FrozenEstimator(kind="xgb", margins=margins)
            with self.assertRaises((DataContractError, ValueError)):
                worker.predict_saved_estimator(model, "xgb", np.ones((2, 2), dtype=np.float32))

    def test_restore_keeps_frozen_medians_even_when_validation_distribution_changes(self):
        saved = preprocessing_state()
        before = copy.deepcopy(saved)
        fitted = worker.restore_fitted_preprocessor(saved)
        index = pd.MultiIndex.from_tuples([("a", "2017-12-29"), ("b", "2017-12-29")],
            names=["source_security_id", "signal_date"])
        frame = pd.DataFrame(1., index=index, columns=saved["manifest"]["input_columns"])
        extra = "px_simple_return_1s"
        frame.loc[index[0], extra] = np.nan
        frame.loc[index[1], extra] = 1e20
        prepared = fitted.transform(frame, prediction_mask=pd.Series(True, index=index))
        self.assertEqual(prepared.X.loc[index[0], extra], saved["manifest"]["medians"][extra])
        self.assertEqual(prepared.X.loc[index[0], extra + "__missing"], 1)
        self.assertEqual(saved, before)
        self.assertFalse(prepared.provenance["fit_performed"])

    def test_restore_rejects_tampering_future_training_and_unexpected_scaling(self):
        for change in ("tamper", "future", "cutoff", "scaling", "policy"):
            saved = preprocessing_state()
            if change == "tamper": saved["manifest"]["medians"]["zMOM1"] += 1
            else:
                if change == "future": saved["manifest"]["train_end"] = "2018-12-31"
                elif change == "cutoff": saved["manifest"]["label_observed_through"] = "2018-01-01"
                elif change == "scaling": saved["manifest"]["scaling"] = "zscore"
                else: saved["manifest"]["training_only_schema_selection"] = False
                saved["fingerprint"] = canonical_sha256(saved["manifest"])
            with self.subTest(change=change), self.assertRaises(DataContractError):
                worker.restore_fitted_preprocessor(saved)

    def test_worker_has_no_fit_or_fitted_state_learning_entrypoint(self):
        tree = ast.parse(inspect.getsource(worker))
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        self.assertFalse(any(isinstance(call.func, ast.Attribute) and call.func.attr in
            {"fit", "fit_transform", "partial_fit"} for call in calls))
        self.assertFalse(any(isinstance(call.func, ast.Name) and call.func.id in
            {"fit_preprocessor", "fit_predict_pilot_arm"} for call in calls))


class ValidationPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name).resolve()
        cls.parent, cls.configuration = fallback_fixture(cls.root)
        cls.run_directory = cls.root / "outputs/independent"
        for year in (2018, 2019):
            directory = cls.run_directory / "years" / str(year)
            shutil.copytree(cls.parent / "years" / str(year), directory)
            audit = json.loads((directory / "audit.json").read_text())
            audit["build_identity_sha256"] = NEW
            _json(directory / "audit.json", audit)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def plan(self):
        with patch.object(worker, "_parent", return_value=(self.parent,
                {"build_identity_sha256": PARENT, "model_identity_sha256": "sha256:" + "d" * 64},
                {}, {})):
            return worker.plan_validation_reprediction(self.root, self.run_directory, {}, NEW,
                fitted_run_directory=self.parent)

    def test_plan_only_projects_metadata_and_exact_new_two_years(self):
        original, reads = pq.ParquetFile.read, []
        def guarded(parquet, columns=None, *args, **kwargs):
            self.assertIsNotNone(columns)
            self.assertLessEqual(set(columns), set(COVERAGE_METADATA) | set(TARGET_METADATA))
            self.assertNotIn("target_class", columns)
            self.assertNotIn("label_return", columns)
            reads.append(tuple(columns))
            return original(parquet, columns=columns, *args, **kwargs)
        with patch.object(pq.ParquetFile, "read", guarded), \
                patch.object(worker, "_load_validation", side_effect=AssertionError("Financial decode")):
            plan = self.plan()
        self.assertTrue(reads)
        self.assertEqual(tuple(plan["years"]), (2018, 2019))
        self.assertEqual(plan["provenance"]["original_rows"], 360)
        self.assertFalse(plan["provenance"]["training_financial_values_decoded"])
        self.assertEqual(plan["provenance"]["new_estimator_fits"], 0)

    def test_financial_materialization_retains_original_keys_missing_labels_and_all_features(self):
        plan = self.plan()
        part = worker._load_validation(plan)
        self.assertEqual(len(part.X), 360)
        self.assertEqual(len(part.X.columns), 715)
        self.assertEqual(part.y.isna().sum(), 1)
        self.assertTrue(part.ledger.outcome_start.ge("2018-01-01").all())
        self.assertTrue(part.ledger.outcome_end.le("2019-12-31").all())

    def test_out_of_window_metadata_fails_before_financial_decoding(self):
        directory = self.run_directory / "years/2019"
        path, audit_path = directory / "features.parquet", directory / "audit.json"
        originals = {path: path.read_bytes(), audit_path: audit_path.read_bytes()}
        self.addCleanup(lambda: [item.write_bytes(content) for item, content in originals.items()])
        table = pq.read_table(path)
        dates = table["signal_date"].to_pylist()
        dates[0] = "2020-01-31"
        table = table.set_column(table.schema.get_field_index("signal_date"), "signal_date", pa.array(dates))
        pq.write_table(table, path)
        audit = json.loads(audit_path.read_text())
        audit["artifacts"]["features"] = _artifact(directory, path)
        _json(audit_path, audit)
        with patch.object(worker, "_load_validation", side_effect=AssertionError("Financial decode")), \
                self.assertRaises(DataContractError):
            self.plan()

    def test_original_validation_denominator_cannot_change(self):
        directory = self.run_directory / "years/2018"
        path, audit_path = directory / "original_cohort.parquet", directory / "audit.json"
        originals = {path: path.read_bytes(), audit_path: audit_path.read_bytes()}
        self.addCleanup(lambda: [item.write_bytes(content) for item, content in originals.items()])
        table = pq.read_table(path).slice(1)
        pq.write_table(table, path)
        audit = json.loads(audit_path.read_text())
        audit["artifacts"]["original_cohort"] = _artifact(directory, path)
        _json(audit_path, audit)
        with self.assertRaises(DataContractError):
            self.plan()

    def test_zero_new_fit_authority_is_mandatory_before_parent_loading(self):
        with patch.object(worker, "read_fallback_models", side_effect=AssertionError("Unexpected parent load")), \
                self.assertRaises(DataContractError):
            worker._parent(self.root, self.parent, {"execution": {"maximum_new_model_fits": 4}})

    def test_full_four_arm_saved_inference_has_zero_fits_and_verifiable_reused_identity(self):
        plan = self.plan()
        part = worker._load_validation(plan)
        fingerprints = {}
        created = []
        for feature_set in ("paper15", "expanded"):
            saved = preprocessing_state()
            names = list(PAPER_FEATURE_NAMES) if feature_set == "paper15" else list(part.X)
            saved["manifest"].update(input_columns=names, retained_columns=names,
                output_columns=[column for name in names for column in (name, name + "__missing")],
                medians={name: 0. for name in names})
            saved["fingerprint"] = canonical_sha256(saved["manifest"])
            fingerprints[feature_set] = saved["fingerprint"]
            path = self.parent / "models/preprocessors" / feature_set / "manifest.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            _json(path, saved)
            created.append(path)
        for arm in worker.ARMS:
            directory = self.parent / "models/arms" / arm["arm_id"]
            directory.mkdir(parents=True, exist_ok=True)
            model = directory / ("model.joblib" if arm["estimator"] == "rf" else "model.ubj")
            model.write_bytes(b"synthetic frozen model; no fit")
            native = list(range(1, 11)) if arm["estimator"] == "rf" else list(range(10))
            _json(directory / "audit.json", {"artifacts": {"model": _artifact(directory, model)},
                "metadata": {"native_class_ids": native,
                    "preprocessor_fingerprint": fingerprints[arm["feature_set"]],
                    "fit_keys_sha256": "sha256:" + "e" * 64,
                    "prepared_train_matrix_sha256": "sha256:" + "f" * 64,
                    "train_rows": 1440, "probability_extraction": "synthetic canonical interface"}})
        parent_audit = {"build_identity_sha256": PARENT,
            "model_identity_sha256": "sha256:" + "d" * 64,
            "summary": {"training_prior_vector": [.1] * 10, "train_fit_rows": 1440}}
        recipe = json.loads((Path(__file__).resolve().parents[1] / "configs/classifier_pilot.json").read_text())
        def fake_rf_load(path):
            width = 30 if "paper15" in str(path) else 1430
            return FrozenEstimator(width=width)
        class FakeXgb(FrozenEstimator):
            def __init__(self):
                super().__init__(kind="xgb", width=30, margins=np.arange(10, dtype=np.float32))
            def load_model(self, path):
                self.n_features_in_ = 30 if "paper15" in str(path) else 1430
        with patch.object(worker, "_parent", return_value=(self.parent, parent_audit, recipe, {})), \
                patch("joblib.load", side_effect=fake_rf_load), \
                patch.dict(sys.modules, {"xgboost": SimpleNamespace(XGBClassifier=FakeXgb)}), \
                patch("dmv2.preprocessing.fit_preprocessor", side_effect=AssertionError("Preprocessing fit forbidden")), \
                patch("dmv2.pilot_models.fit_predict_pilot_arm", side_effect=AssertionError("Model fit forbidden")):
            result = worker.run_validation_reprediction(self.root, self.run_directory,
                {}, NEW, fitted_run_directory=self.parent)
        self.addCleanup(lambda: shutil.rmtree(result))
        saved = worker.read_validation_reprediction(result, root=self.root, identity_sha256=NEW)
        self.assertEqual(saved["audit"]["summary"]["newly_fitted_arms"], 0)
        self.assertEqual(saved["audit"]["summary"]["reused_fitted_arms"], 4)
        self.assertEqual(saved["audit"]["summary"]["validation_original_rows"], 360)
        self.assertEqual(saved["audit"]["summary"]["validation_months"], 24)
        for arm in worker.ARMS:
            audit = json.loads((result / "arms" / arm["arm_id"] / "audit.json").read_text())
            self.assertEqual(audit["metadata"]["new_fit_count"], 0)
            self.assertEqual(audit["metadata"]["parent_model_identity_sha256"], parent_audit["model_identity_sha256"])
            self.assertEqual(pq.ParquetFile(result / "arms" / arm["arm_id"] / "predictions.parquet").metadata.num_rows, 360)


if __name__ == "__main__":
    unittest.main()
