"""Only synthetic parameter/inference adapters; never fit any estimator."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

SOURCE = Path(__file__).resolve().parents[1] / "scripts/verify_source_corrected_models.py"
spec = importlib.util.spec_from_file_location("_corrected_verifier_tests", SOURCE)
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)


def checkpoint(directory, kind, *, effective_change=None):
    parameters = {"n_jobs": 1, "n_estimators": 500, "max_depth": 12} if kind == "rf" else {
        "n_jobs": 1, "n_estimators": 300, "objective": "multi:softprob", "num_class": 10}
    effective = {**parameters, **(effective_change or {})}
    (directory / "FIT_COMPLETE.json").write_text(json.dumps({"metadata": {
        "explicit_parameters": parameters, "effective_parameters": effective}}))
    return parameters


class RF:
    classes_ = np.arange(1, 11)
    n_features_in_ = 2
    def __init__(self, parameters):
        self.parameters = parameters
        self.estimators_ = [SimpleNamespace(tree_=SimpleNamespace(max_depth=7))] * 500
    def get_params(self, deep=False): return self.parameters
    def predict_proba(self, X): return np.full((len(X), 10), .1)


class Booster:
    rounds = 300
    def save_config(self):
        return json.dumps({"learner": {"objective": {"name": "multi:softprob"},
                                      "learner_model_param": {"num_class": "10", "num_feature": "2"}}})
    def num_boosted_rounds(self): return self.rounds
    def num_features(self): return 2
    def attr(self, name): return None


class XGB:
    classes_ = np.arange(10)
    def __init__(self, **kwargs): self.booster = Booster()
    def load_model(self, path): pass
    def set_params(self, **kwargs): pass
    def get_booster(self): return self.booster
    def predict(self, X, output_margin=False):
        assert output_margin
        return np.zeros((len(X), 10))


class AdapterTests(unittest.TestCase):
    def test_saved_rf_parameters_structure_and_probabilities(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp); parameters = checkpoint(path, "rf")
            with patch("joblib.load", return_value=RF(parameters)):
                difference = v.verify_actual_model(SimpleNamespace(json_file=v.js), path, "rf",
                                                  np.zeros((3, 2)), np.full((3, 10), .1))
            self.assertEqual(difference, 0)

    def test_serialized_rf_parameter_tamper_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp); parameters = checkpoint(path, "rf")
            with patch("joblib.load", return_value=RF({**parameters, "n_jobs": 4})):
                with self.assertRaisesRegex(AssertionError, "Serialized RF parameters"):
                    v.verify_actual_model(SimpleNamespace(json_file=v.js), path, "rf",
                                          np.zeros((3, 2)), np.full((3, 10), .1))

    def test_presave_effective_parameter_tamper_refused_before_model_load(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp); checkpoint(path, "rf", effective_change={"n_jobs": 2})
            with patch("joblib.load", side_effect=AssertionError("Unexpected model load")):
                with self.assertRaisesRegex(AssertionError, "Pre-save effective"):
                    v.verify_actual_model(SimpleNamespace(json_file=v.js), path, "rf",
                                          np.zeros((3, 2)), np.full((3, 10), .1))

    def test_xgb_stored_objective_width_rounds_and_softmax(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp); checkpoint(path, "xgb")
            with patch("xgboost.XGBClassifier", XGB):
                difference = v.verify_actual_model(SimpleNamespace(json_file=v.js), path, "xgb",
                                                  np.zeros((3, 2)), np.full((3, 10), .1))
            self.assertEqual(difference, 0)

    def test_xgb_wrong_round_count_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp); checkpoint(path, "xgb")
            model = XGB(); model.booster.rounds = 299
            with patch("xgboost.XGBClassifier", return_value=model):
                with self.assertRaisesRegex(AssertionError, "boosting rounds"):
                    v.verify_actual_model(SimpleNamespace(json_file=v.js), path, "xgb",
                                          np.zeros((3, 2)), np.full((3, 10), .1))

    def test_artifact_pin_is_exact_and_unambiguous(self):
        rows = [{"path": "one", "sha256": "1"}, {"path": "two", "sha256": "2"}]
        self.assertEqual(v._artifact_by_path(rows, "one"), rows[0])
        for records, key in ((rows, "missing"), (rows + [rows[0]], "one")):
            with self.assertRaisesRegex(AssertionError, "missing or duplicated"):
                v._artifact_by_path(records, key)

    def test_receipt_cannot_replace_prior_result(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "receipt.json"
            v.save_once(path, {"status": "PASS"})
            with self.assertRaisesRegex(AssertionError, "Preserve"):
                v.save_once(path, {"status": "DIFFERENT"})

    def test_metadata_seal_refuses_outside_root_and_binds_exact_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "approved"; root.mkdir()
            approved = root / "audit.json"; approved.write_text('{"status":"COMPLETE"}')
            seal = v.metadata_pin(root, approved)
            self.assertEqual(seal["path"], "audit.json")
            self.assertEqual(seal["sha256"], v.sha(approved))
            outside = Path(temp) / "outside.json"; outside.write_text("{}")
            with self.assertRaisesRegex(AssertionError, "approved root"):
                v.metadata_pin(root, outside)


if __name__ == "__main__": unittest.main()
