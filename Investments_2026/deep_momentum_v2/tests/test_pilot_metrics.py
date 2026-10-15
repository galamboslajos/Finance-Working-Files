"""Hand-checkable synthetic probability metrics; missing labels stay missing."""
import math
import unittest

import numpy as np
import pandas as pd

from dmv2.pilot_metrics import evaluate_probabilities
from dmv2.universe import DataContractError


def metadata(months=("2018-01", "2018-01", "2018-02")):
    return pd.DataFrame({"source_security_id": [f"s{i}" for i in range(len(months))],
                         "signal_date": ["2017-12-29" if month == "2018-01" else "2018-01-31" for month in months],
                         "outcome_month": list(months),
                         "outcome_end": [month + "-28" for month in months]})


class PoisonNumericalInput:
    def __array__(self, *args, **kwargs):
        raise AssertionError("Protected values must not be converted")


class PilotMetricTests(unittest.TestCase):
    def setUp(self):
        self.probabilities = np.full((3, 10), .1)
        self.labels = [1, 2, 10]
        self.meta = metadata()

    def evaluate(self, **changes):
        arguments = dict(probabilities=self.probabilities, labels=self.labels,
                         class_ids=list(range(1, 11)), row_metadata=self.meta)
        arguments.update(changes)
        return evaluate_probabilities(**arguments)

    def test_uniform_reference_exact_metrics(self):
        result = self.evaluate()
        self.assertAlmostEqual(result.summary["mean_monthly_log_loss"], math.log(10))
        self.assertAlmostEqual(result.summary["mean_monthly_brier_sum10"], .9)
        self.assertAlmostEqual(result.summary["mean_monthly_accuracy"], .25)
        self.assertAlmostEqual(result.summary["pooled_accuracy"], 1 / 3)
        self.assertEqual(result.summary["hard_mode_probability_tie_rule"], "lowest_canonical_class_id")

    def test_equal_month_means_do_not_pool_unequal_cohorts(self):
        self.probabilities[0] = 0
        self.probabilities[0, 0] = 1
        self.probabilities[1] = 0
        self.probabilities[1, 1] = 1
        result = self.evaluate()
        self.assertAlmostEqual(result.summary["mean_monthly_log_loss"], math.log(10) / 2)
        self.assertAlmostEqual(result.summary["pooled_log_loss"], math.log(10) / 3)

    def test_missing_labels_keep_rows_and_never_zero_fill(self):
        result = self.evaluate(labels=[1, None, np.nan])
        self.assertEqual(len(result.tables["row_scores"]), 3)
        self.assertEqual(result.summary["labelled_rows"], 1)
        self.assertEqual(result.summary["missing_label_rows"], 2)
        self.assertTrue(result.tables["row_scores"].loc[1:, "true_class"].isna().all())
        self.assertTrue(result.tables["row_scores"].loc[1:, "log_loss"].isna().all())
        self.assertEqual(result.tables["monthly"].iloc[1]["status"], "no_observed_labels")

    def test_logloss_only_evaluation_floor_does_not_change_probabilities(self):
        self.probabilities[0] = 0
        self.probabilities[0, 5] = 1
        self.probabilities[1] = (1 - 1e-20) / 9
        self.probabilities[1, 1] = 1e-20
        before = self.probabilities.copy()
        result = self.evaluate()
        self.assertEqual(result.summary["zero_true_class_probability_rows"], 1)
        self.assertEqual(result.summary["floored_true_class_probability_rows"], 2)
        self.assertAlmostEqual(result.tables["row_scores"].iloc[0]["log_loss"], -math.log(1e-15))
        self.assertEqual(result.tables["row_scores"].iloc[0]["true_class_probability"], 0)
        np.testing.assert_array_equal(self.probabilities, before)

    def test_class_column_alignment(self):
        self.probabilities[:] = 0
        self.probabilities[np.arange(3), np.array(self.labels) - 1] = 1
        canonical = self.evaluate()
        ids = list(range(10, 0, -1))
        shuffled = self.evaluate(probabilities=self.probabilities[:, ::-1], class_ids=ids)
        self.assertEqual(canonical.summary, shuffled.summary)
        self.assertEqual(shuffled.summary["pooled_log_loss"], 0)
        self.assertEqual(shuffled.summary["pooled_brier_sum10"], 0)

    def test_extreme_precision_reports_support_and_abstention(self):
        result = self.evaluate(labels=[None, 2, 10])
        extremes = result.tables["extreme_precision"].query("period == 'all'").set_index("class_id")
        self.assertEqual(extremes.loc[1, "hard_mode_prediction_count"], 3)
        self.assertEqual(extremes.loc[1, "labelled_hard_mode_count"], 2)
        self.assertEqual(extremes.loc[1, "precision"], 0)
        self.assertEqual(extremes.loc[10, "status"], "no_hard_mode_predictions")
        self.assertTrue(pd.isna(extremes.loc[10, "precision"]))

    def test_extreme_all_missing_labels_abstains(self):
        result = self.evaluate(labels=[None, None, None])
        extremes = result.tables["extreme_precision"].query("period == 'all'").set_index("class_id")
        self.assertEqual(extremes.loc[1, "status"], "no_observed_labels_in_hard_mode")
        self.assertIsNone(result.summary["mean_monthly_log_loss"])
        self.assertIsNone(result.summary["pooled_top_label_ece"])

    def test_calibration_uses_fixed_ten_bins_with_explicit_empty_cells(self):
        self.probabilities[0] = 0
        self.probabilities[0, 0] = 1
        result = self.evaluate()
        top = result.tables["toplabel_calibration"]
        classes = result.tables["classwise_calibration"]
        self.assertEqual(len(top), 10)
        self.assertEqual(len(classes), 100)
        self.assertEqual(top.iloc[9]["labelled_count"], 1)
        self.assertEqual(top.iloc[1]["labelled_count"], 2)
        self.assertEqual(int(top.labelled_count.sum()), 3)
        self.assertEqual(int(classes.labelled_count.sum()), 30)
        self.assertTrue(top.loc[top.labelled_count == 0, "mean_probability"].isna().all())

    def test_invalid_labels_rejected_without_coercion(self):
        for bad in ("1", True, np.inf, -1, 0, 11, 1.5):
            with self.subTest(bad=bad), self.assertRaises(DataContractError):
                self.evaluate(labels=[bad, 2, 10])

    def test_safe_metadata_guard_precedes_financial_conversion(self):
        self.meta.loc[0, "outcome_month"] = "2020-01"
        with self.assertRaisesRegex(DataContractError, "2018–2019"):
            self.evaluate(probabilities=PoisonNumericalInput(), labels=PoisonNumericalInput())

    def test_wrong_signal_month_duplicate_keys_or_bad_dates_rejected(self):
        for alteration in ("signal", "duplicate", "date"):
            frame = metadata()
            if alteration == "signal":
                frame.loc[0, "signal_date"] = "2018-01-31"
            elif alteration == "duplicate":
                frame.loc[1, ["source_security_id", "signal_date"]] = frame.loc[0, ["source_security_id", "signal_date"]]
            else:
                frame.loc[0, "outcome_end"] = "2018-02-28"
            with self.subTest(alteration=alteration), self.assertRaises(DataContractError):
                self.evaluate(row_metadata=frame)

    def test_index_and_row_count_mismatch_rejected(self):
        with self.assertRaises(DataContractError):
            self.evaluate(labels=pd.Series(self.labels, index=[2, 1, 0]))
        with self.assertRaises(DataContractError):
            self.evaluate(probabilities=self.probabilities[:2])

    def test_empty_cohort_returns_null_metrics_and_fixed_calibration_grid(self):
        result = self.evaluate(probabilities=np.empty((0, 10)), labels=[], row_metadata=self.meta.iloc[:0])
        self.assertEqual(result.summary["prediction_rows"], 0)
        self.assertIsNone(result.summary["pooled_log_loss"])
        self.assertEqual(len(result.tables["classwise_calibration"]), 100)


if __name__ == "__main__":
    unittest.main()
