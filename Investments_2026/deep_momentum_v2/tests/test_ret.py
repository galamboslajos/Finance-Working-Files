"""Synthetic contracts for shared monthly realized payoffs; no research prices."""
from calendar import monthrange
from dataclasses import replace
from copy import deepcopy
import unittest

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from dmv2.monthly import MonthWindow, _step_month
from dmv2.ret import (CLASS_IDS, RET_POLICY, align_class_probabilities, build_ret_payoffs,
                      score_ret, verify_ret_payoffs)
from dmv2.universe import DataContractError


def month_end(month):
    year, number = map(int, month.split("-"))
    return f"{month}-{monthrange(year, number)[1]:02d}"


def target_fixture(*, first="2010-01", count=120, variable_members=False,
                   tied_month=None, missing_month=None, absent_month=None):
    """Explicit synthetic calendar evidence and original paper-qualified names."""
    from dmv2.targets import build_monthly_targets
    cohort, outcomes, windows = [], [], []
    for offset in range(count):
        month = _step_month(first, offset)
        signal, start, end = month_end(_step_month(month, -1)), month + "-01", month_end(month)
        windows.append(MonthWindow(month, signal, start, end, (start, end)))
        if month == absent_month:
            continue
        number = 20 if variable_members and offset % 2 else 10
        for rank in range(number):
            source = f"synthetic-{rank:02d}"
            cohort.append({"source_security_id": source, "signal_date": signal,
                           "paper_prediction_eligible": True, "identity_eligible": True,
                           "requested": True})
            value = (0.2 if offset % 2 else 0.0) + (number - rank) / 1000
            if month == tied_month:
                value = 0.1
            missing = month == missing_month and rank == 0
            outcomes.append({"source_security_id": source, "signal_date": signal,
                             "outcome_month": month, "outcome_start": start, "outcome_end": end,
                             "requested": True,
                             "provider_adjusted_return": None if missing else value,
                             "return_status": "missing_end" if missing else "ok"})
    return build_monthly_targets(pa.Table.from_pylist(cohort), pa.Table.from_pylist(outcomes),
                                 label_observed_through=month_end(_step_month(first, count - 1)),
                                 month_windows=windows)


def run(panel, *, signal="2019-12-31", cutoff=None):
    return build_ret_payoffs(panel, signal_date=signal,
                             label_observed_through=signal if cutoff is None else cutoff)


class ProbabilityAlignmentTests(unittest.TestCase):
    def test_explicit_shuffled_class_ids_reorder_without_changing_values(self):
        values = np.arange(1, 11, dtype=float)[None, :] / 55
        ids = [10, 1, 9, 2, 8, 3, 7, 4, 6, 5]
        supplied = values[:, [class_id - 1 for class_id in ids]]
        np.testing.assert_array_equal(align_class_probabilities(supplied, class_ids=ids), values)

    def test_missing_duplicate_boolean_float_or_unknown_class_ids_rejected(self):
        probabilities = np.full((1, 10), 0.1)
        for ids in ([1, 2], list(range(10)), [1] * 10,
                    [True, *range(2, 11)], [1.0, *range(2, 11)]):
            with self.subTest(ids=ids), self.assertRaises(DataContractError):
                align_class_probabilities(probabilities, class_ids=ids)

    def test_values_are_finite_nonnegative_real_and_sum_to_one(self):
        invalid = [np.full((2, 10), 0.2), np.full((1, 10), np.nan),
                   np.full((1, 10), np.inf), [["0.1"] * 10], [[True] * 10],
                   np.full((1, 10), 0.1 + 1j), np.ones(10), np.ones((2, 9))]
        negative = np.full((1, 10), 0.1)
        negative[0, :2] = [-0.1, 0.3]
        invalid.append(negative)
        for values in invalid:
            with self.subTest(values=str(values)), self.assertRaises(DataContractError):
                align_class_probabilities(values, class_ids=CLASS_IDS)

    def test_sum_tolerance_does_not_renormalize(self):
        values = np.full((1, 10), 0.1)
        values[0, 0] += 5e-9
        np.testing.assert_array_equal(align_class_probabilities(values, class_ids=CLASS_IDS), values)
        values[0, 0] += 2e-8
        with self.assertRaises(DataContractError):
            align_class_probabilities(values, class_ids=CLASS_IDS)

    def test_empty_forecast_matrix_retains_ten_explicit_columns(self):
        self.assertEqual(align_class_probabilities(np.empty((0, 10)), class_ids=CLASS_IDS).shape, (0, 10))


class RetPayoffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.targets = target_fixture(variable_members=True, count=121)
        cls.payoffs = run(cls.targets)

    def test_first_window_is_exactly_january2010_through_december2019(self):
        payoffs = self.payoffs
        self.assertEqual(payoffs.provenance["ret_policy"], RET_POLICY)
        self.assertEqual(payoffs.provenance["first_outcome_month"], "2010-01")
        self.assertEqual(payoffs.provenance["last_outcome_month"], "2019-12")
        self.assertEqual(payoffs.monthly.num_rows, 1200)
        self.assertTrue(payoffs.provenance["vector_available"])
        self.assertEqual(payoffs.provenance["available_month_class_cells"], 1200)
        self.assertTrue(verify_ret_payoffs(payoffs))
        self.assertFalse(payoffs.provenance["model_ready"])
        self.assertFalse(payoffs.provenance["knowledge_time_certified"])

    def test_each_month_has_equal_weight_rather_than_each_stock_row(self):
        selected = self.targets.table.filter(pc.less_equal(self.targets.table["outcome_month"], "2019-12"))
        rows = selected.to_pylist()
        for class_id, payoff in zip(CLASS_IDS, self.payoffs.table["payoff"].to_pylist()):
            monthly = []
            for month in sorted({row["outcome_month"] for row in rows}):
                values = [row["label_return"] for row in rows
                          if row["outcome_month"] == month and row["target_class"] == class_id]
                monthly.append(np.mean(values))
            self.assertAlmostEqual(payoff, np.mean(monthly), places=14)
            pooled = np.mean([row["label_return"] for row in rows if row["target_class"] == class_id])
            self.assertGreater(abs(payoff - pooled), 0.02)

    def test_monthly_update_rolls_one_completed_calendar_month(self):
        result = run(self.targets, signal="2020-01-31")
        self.assertEqual(result.provenance["first_outcome_month"], "2010-02")
        self.assertEqual(result.provenance["last_outcome_month"], "2020-01")
        self.assertTrue(result.provenance["vector_available"])

    def test_missing_calendar_month_cannot_compress_window(self):
        panel = target_fixture(absent_month="2015-06")
        result = run(panel)
        self.assertFalse(result.provenance["vector_available"])
        self.assertEqual(result.monthly.num_rows, 1200)
        self.assertEqual(result.table["payoff"].null_count, 10)
        statuses = {row["status"] for row in result.monthly.to_pylist() if row["outcome_month"] == "2015-06"}
        self.assertTrue(statuses <= {"missing_month", "missing_month_completion_evidence"})

    def test_ties_may_leave_empty_classes_and_withhold_whole_vector(self):
        result = run(target_fixture(tied_month="2012-03"))
        tied = [row for row in result.monthly.to_pylist() if row["outcome_month"] == "2012-03"]
        self.assertEqual(sum(row["status"] == "available" for row in tied), 1)
        self.assertEqual(sum(row["status"] == "empty_realized_class" for row in tied), 9)
        self.assertEqual(result.table["payoff"].null_count, 10)

    def test_missing_return_is_never_filled_or_reallocated_by_payoff_worker(self):
        result = run(target_fixture(missing_month="2011-04"))
        self.assertFalse(result.provenance["vector_available"])
        self.assertEqual(result.table["payoff"].null_count, 10)
        failed = result.monthly.filter(pc.equal(result.monthly["outcome_month"], "2011-04"))
        self.assertEqual(failed["class_mean_return"].null_count, 10)

    def test_midmonth_or_early_observation_cutoff_withholds_latest_month(self):
        for signal, cutoff in (("2019-12-15", "2019-12-15"), ("2019-12-31", "2019-12-30")):
            with self.subTest(signal=signal, cutoff=cutoff):
                result = run(self.targets, signal=signal, cutoff=cutoff)
                self.assertEqual(result.provenance["first_outcome_month"], "2010-01")
                self.assertEqual(result.provenance["last_outcome_month"], "2019-12")
                self.assertEqual(result.table["payoff"].null_count, 10)
                late = [row for row in result.monthly.to_pylist() if row["outcome_month"] == "2019-12"]
                self.assertTrue(all(row["status"] == "month_not_observed_by_cutoff" for row in late))

    def test_target_source_ending_early_withholds_without_claiming_new_observations(self):
        result = run(target_fixture(count=119))
        self.assertEqual(result.provenance["last_outcome_month"], "2019-12")
        self.assertEqual(result.table["payoff"].null_count, 10)
        self.assertEqual(result.monthly.num_rows, 1200)
        self.assertNotIn("2019-12", result.provenance["source_month_target_fingerprints"])

    def test_future_or_protected_cutoff_rejected(self):
        for signal, cutoff in (("2019-12-31", "2020-01-01"), ("2024-01-31", "2023-12-31")):
            with self.subTest(signal=signal), self.assertRaises(DataContractError):
                run(self.targets, signal=signal, cutoff=cutoff)

    def test_unused_future_financial_mutations_are_not_decoded_or_fingerprinted(self):
        table = self.targets.table
        future = pc.greater(table["outcome_month"], "2019-12")
        changed = table.set_column(table.schema.get_field_index("label_return"), "label_return",
                                   pc.if_else(future, float("nan"), table["label_return"]))
        changed = changed.set_column(changed.schema.get_field_index("target_class"), "target_class",
                                     pc.if_else(future, pa.scalar(0, pa.int8()), changed["target_class"]))
        result = run(replace(self.targets, table=changed))
        self.assertTrue(result.table.equals(self.payoffs.table))
        self.assertTrue(result.monthly.equals(self.payoffs.monthly))
        self.assertEqual(result.fingerprint, self.payoffs.fingerprint)

    def test_mutated_selected_month_or_provenance_is_rejected(self):
        changed = self.targets.table.set_column(
            self.targets.table.schema.get_field_index("label_return"), "label_return",
            pa.array([1.0] * self.targets.table.num_rows, type=pa.float64()))
        with self.assertRaises(DataContractError):
            run(replace(self.targets, table=changed))
        provenance = deepcopy(self.payoffs.provenance)
        provenance["signal_date"] = "2020-01-31"
        with self.assertRaisesRegex(DataContractError, "fingerprint"):
            verify_ret_payoffs(replace(self.payoffs, provenance=provenance))

    def test_scoring_aligns_classes_and_preserves_all_forecast_rows(self):
        index = pd.MultiIndex.from_tuples([("a", "2019-12-31"), ("b", "2019-12-31")],
                                         names=["source_security_id", "signal_date"])
        probabilities = pd.DataFrame(np.eye(10)[[0, 9]], index=index)
        result = score_ret(probabilities, self.payoffs, class_ids=CLASS_IDS,
                           signal_date="2019-12-31", label_observed_through="2019-12-31")
        self.assertTrue(result.index.equals(index))
        np.testing.assert_array_equal(result["ret_score"],
                                      np.asarray(self.payoffs.table["payoff"].to_pylist())[[0, 9]])
        shuffled = [10, 4, 2, 8, 6, 3, 1, 9, 5, 7]
        permuted = probabilities.iloc[:, [class_id - 1 for class_id in shuffled]]
        again = score_ret(permuted, self.payoffs, class_ids=shuffled,
                          signal_date="2019-12-31", label_observed_through="2019-12-31")
        pd.testing.assert_frame_equal(result, again)

    def test_withheld_vector_retains_every_forecast_with_null_score(self):
        payoffs = run(target_fixture(tied_month="2019-12"))
        probabilities = pd.DataFrame(np.full((3, 10), 0.1), index=["a", "b", "c"])
        result = score_ret(probabilities, payoffs, class_ids=CLASS_IDS,
                           signal_date="2019-12-31", label_observed_through="2019-12-31")
        self.assertEqual(list(result.index), ["a", "b", "c"])
        self.assertTrue(result["ret_score"].isna().all())
        self.assertTrue((result["ret_status"] == "withheld_incomplete_120_month_class_grid").all())

    def test_empty_forecast_stays_empty_without_a_label_membership_lookup(self):
        result = score_ret(pd.DataFrame(np.empty((0, 10)), index=pd.Index([], name="forecast_id")),
                           self.payoffs, class_ids=CLASS_IDS,
                           signal_date="2019-12-31", label_observed_through="2019-12-31")
        self.assertEqual(result.shape, (0, 2))
        self.assertEqual(result.index.name, "forecast_id")

    def test_score_rejects_date_mismatch_mutated_payoffs_and_invalid_forecast_keys(self):
        values = pd.DataFrame(np.full((2, 10), 0.1), index=["a", "b"])
        cases = [(values, self.payoffs, "2020-01-31", "2020-01-31"),
                 (values, self.payoffs, "2019-12-31", "2019-12-30"),
                 (values.set_axis(["a", "a"]), self.payoffs, "2019-12-31", "2019-12-31"),
                 (values.set_axis(["a", None]), self.payoffs, "2019-12-31", "2019-12-31")]
        mixed = pd.MultiIndex.from_tuples([("a", "2019-12-31"), ("b", "2020-01-31")],
                                         names=["source_security_id", "signal_date"])
        cases.append((values.set_axis(mixed), self.payoffs, "2019-12-31", "2019-12-31"))
        for probabilities, payoffs, signal, cutoff in cases:
            with self.subTest(index=str(probabilities.index), signal=signal), self.assertRaises(DataContractError):
                score_ret(probabilities, payoffs, class_ids=CLASS_IDS,
                          signal_date=signal, label_observed_through=cutoff)
        corrupted = self.payoffs.table.set_column(1, "payoff", pa.array([0.0] * 10, type=pa.float64()))
        with self.assertRaisesRegex(DataContractError, "fingerprint"):
            score_ret(values, replace(self.payoffs, table=corrupted), class_ids=CLASS_IDS,
                      signal_date="2019-12-31", label_observed_through="2019-12-31")


if __name__ == "__main__":
    unittest.main()
