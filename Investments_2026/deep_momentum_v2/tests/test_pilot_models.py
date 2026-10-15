"""Synthetic/mocked fits only: no research data or validation-label fitting."""
import importlib.metadata
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from dmv2.pilot_models import fit_predict_pilot_arm
from dmv2.universe import DataContractError


RF = dict(n_estimators=2, n_jobs=1, random_state=1729, max_features="sqrt",
          max_depth=3, min_samples_leaf=1, bootstrap=True, class_weight=None)
XGB = dict(n_estimators=2, n_jobs=1, random_state=1729, objective="multi:softprob",
           num_class=10, tree_method="hist", max_depth=2, learning_rate=.05,
           subsample=.8, colsample_bytree=.8, eval_metric="mlogloss", device="cpu")


class FakeEstimator:
    def __init__(self, **parameters):
        self.parameters = parameters
        self.fit_calls = []
        self.predict_calls = []

    def fit(self, X, y):
        self.fit_calls.append((X.copy(), y.copy()))
        self.classes_ = np.unique(y)[::-1]
        return self

    def predict_proba(self, X):
        self.predict_calls.append((X.copy(), {}))
        return np.tile(np.arange(1., 11.) / 55, (len(X), 1))

    def predict(self, X, **kwargs):
        self.predict_calls.append((X.copy(), kwargs))
        return np.tile(np.arange(10, dtype=np.float32), (len(X), 1))

    def get_params(self, deep=False):
        return self.parameters.copy()


class PilotModelTests(unittest.TestCase):
    def setUp(self):
        self.X = np.arange(120, dtype=float).reshape(40, 3)
        self.y = np.tile(np.arange(1, 11), 4)
        self.future = np.arange(9, dtype=float).reshape(3, 3)
        self.versions = {"scikit-learn": "frozen", "xgboost": "frozen"}

    def run_fake(self, kind="rf", **changes):
        args = dict(parameters=RF if kind == "rf" else XGB, package_versions=self.versions)
        args.update(changes)
        with patch("dmv2.pilot_models.version", return_value="frozen"), patch("dmv2.pilot_models._constructor", return_value=FakeEstimator):
            return fit_predict_pilot_arm(kind, self.X, self.y, self.future, **args)

    def test_rf_fits_once_without_evalset_and_aligns_shuffled_classes(self):
        result = self.run_fake()
        self.assertEqual(len(result.estimator.fit_calls), 1)
        np.testing.assert_array_equal(result.estimator.fit_calls[0][1], self.y)
        np.testing.assert_allclose(result.probabilities[0], np.arange(10., 0., -1.) / 55)
        self.assertEqual(result.metadata["probability_class_ids"], list(range(1, 11)))
        self.assertFalse(result.metadata["validation_labels_supplied_to_estimator"])
        self.assertEqual(result.estimator.fit_calls[0][0].dtype, np.float32)

    def test_xgb_encodes_labels_and_constructs_softmax_from_raw_margins(self):
        result = self.run_fake("xgb")
        np.testing.assert_array_equal(result.estimator.fit_calls[0][1], self.y - 1)
        self.assertEqual(result.estimator.predict_calls[0][1], {"output_margin": True})
        expected = np.exp(np.arange(10.) - 9)
        expected /= expected.sum()
        np.testing.assert_array_equal(result.probabilities[0], expected[::-1])
        self.assertEqual(result.metadata["native_to_canonical_mapping"][0], {"native_class_id": 9, "canonical_class_id": 10})
        self.assertFalse(result.metadata["probability_repair_renormalization_performed"])

    def test_no_probability_repair(self):
        with patch.object(FakeEstimator, "predict_proba", return_value=np.full((3, 10), .2)):
            with self.assertRaises(DataContractError):
                self.run_fake()

    def test_missing_classes_rejected_before_fit(self):
        self.y[self.y == 10] = 9
        with self.assertRaisesRegex(DataContractError, "missing canonical"):
            self.run_fake()

    def test_bad_training_labels_are_not_coerced(self):
        for bad in (None, "1", True, np.nan, np.inf, 1.5, 0, 11):
            with self.subTest(bad=bad):
                self.y = np.tile(np.arange(1, 11), 4).astype(object)
                self.y[0] = bad
                with self.assertRaises(DataContractError):
                    self.run_fake()

    def test_nonfinite_boolean_text_complex_and_overflow_matrices_rejected(self):
        for value in (np.full((40, 3), np.nan), np.ones((40, 3), dtype=bool),
                      np.full((40, 3), "1"), np.ones((40, 3), dtype=complex),
                      np.full((40, 3), 1e40)):
            with self.subTest(dtype=value.dtype):
                self.X = value
                with self.assertRaises(DataContractError):
                    self.run_fake()

    def test_frozen_package_version_required(self):
        for versions in ({}, {"scikit-learn": "wrong"}):
            with self.assertRaises(DataContractError):
                self.run_fake(package_versions=versions)

    def test_explicit_parameters_and_no_tuning_controls(self):
        for kind, params in (("rf", {}), ("rf", {**RF, "n_jobs": 2}),
                             ("rf", {**RF, "class_weight": "balanced"}),
                             ("rf", {**RF, "warm_start": True}),
                             ("xgb", {**XGB, "objective": "multi:softmax"}),
                             ("xgb", {**XGB, "early_stopping_rounds": 10}),
                             ("xgb", {**XGB, "callbacks": []})):
            with self.subTest(kind=kind, params=params), self.assertRaises(DataContractError):
                self.run_fake(kind, parameters=params)

    def test_dataframe_column_or_y_index_mismatch_rejected(self):
        self.X = pd.DataFrame(self.X, columns=["a", "b", "c"])
        self.future = pd.DataFrame(self.future, columns=["b", "a", "c"])
        with self.assertRaises(DataContractError):
            self.run_fake()
        self.future.columns = self.X.columns
        self.y = pd.Series(self.y, index=np.arange(40)[::-1])
        with self.assertRaises(DataContractError):
            self.run_fake()

    def test_input_matrices_are_unchanged(self):
        before, after = self.X.copy(), self.future.copy()
        self.run_fake()
        np.testing.assert_array_equal(self.X, before)
        np.testing.assert_array_equal(self.future, after)

    def test_empty_prediction_rows_keep_ten_columns(self):
        self.future = np.empty((0, 3))
        result = self.run_fake()
        self.assertEqual(result.probabilities.shape, (0, 10))
        self.assertEqual(result.estimator.predict_calls, [])

    def test_actual_tiny_estimators_on_synthetic_matrices(self):
        for kind, package, params in (("rf", "scikit-learn", RF), ("xgb", "xgboost", XGB)):
            try:
                installed = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                continue
            with self.subTest(kind=kind):
                result = fit_predict_pilot_arm(kind, self.X, self.y, self.future,
                                              parameters=params, package_versions={package: installed})
                self.assertEqual(result.probabilities.shape, (3, 10))
                np.testing.assert_allclose(result.probabilities.sum(axis=1), 1, atol=1e-12)
                self.assertEqual(result.metadata["fit_count"], 1)


if __name__ == "__main__":
    unittest.main()
