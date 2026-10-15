"""Synthetic full-history coverage, original-denominator and RET audit checks."""
from dataclasses import replace
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from dmv2.target_coverage import (aggregate_target_coverage, audit_ret_coverage,
                                  combine_target_panels)
from dmv2.targets import verify_target_panel
from dmv2.universe import DataContractError
from test_ret import target_fixture


class TargetCoverageTests(unittest.TestCase):
    def test_original_and_prediction_denominators_not_conditioned_on_labels(self):
        panel = target_fixture(count=2, missing_month="2010-01")
        report = aggregate_target_coverage(panel)
        summary = report["summary"]
        self.assertEqual(summary["original_members"], 20)
        self.assertEqual(summary["prediction_eligible_members"], 20)
        self.assertEqual(summary["observed_return_members"], 19)
        # A missing outcome leaves nine valid returns: that month's entire rank
        # is withheld, but the twenty original prediction rows remain.
        self.assertEqual(summary["labelled_members"], 10)
        self.assertEqual(summary["eligible_unlabelled_members"], 10)
        self.assertEqual(summary["labelled_prediction_coverage"], 0.5)
        self.assertFalse(summary["future_label_filter_applied_to_prediction_cohort"])
        reasons = report["tables"]["target_status_month"].to_pylist()
        self.assertEqual(sum(row["members"] for row in reasons), 20)

    def test_ties_keep_ten_explicit_bins_and_can_leave_empty_classes(self):
        report = aggregate_target_coverage(target_fixture(count=2, tied_month="2010-01"))
        summary = report["summary"]
        self.assertEqual(summary["months_with_tied_returns"], 1)
        self.assertEqual(summary["months_with_empty_classes"], 1)
        self.assertEqual(summary["empty_month_class_cells"], 9)
        january = [row for row in report["tables"]["target_class_month"].to_pylist()
                   if row["outcome_month"] == "2010-01"]
        self.assertEqual(len(january), 10)
        self.assertEqual(sum(row["empty_class"] for row in january), 9)
        self.assertEqual(sum(row["labelled_members"] for row in january), 10)

    def test_annual_and_period_counts_are_pooled_rows_not_average_percentages(self):
        panel = target_fixture(count=2, variable_members=True, missing_month="2010-01")
        report = aggregate_target_coverage(panel)
        year = report["tables"]["target_coverage_year"].to_pylist()[0]
        self.assertEqual(year["original_members"], 30)
        self.assertEqual(year["observed_return_members"], 29)
        self.assertEqual(year["observed_original_coverage"], 29 / 30)
        self.assertEqual(year["labelled_members"], 20)
        self.assertEqual(sum(row["labelled_members"] for row in report["tables"]["target_class_year"].to_pylist()), 20)

    def test_rolling_training_windows_are_annual_and_have_class_composition(self):
        report = aggregate_target_coverage(target_fixture(count=120))
        windows = report["tables"]["rolling_training_coverage"].to_pylist()
        first = windows[0]
        self.assertEqual(first["test_year"], 2020)
        self.assertEqual(first["first_outcome_month"], "2010-01")
        self.assertEqual(first["last_outcome_month"], "2019-12")
        self.assertTrue(first["all_120_calendar_months_present"])
        self.assertTrue(first["all_ten_classes_present"])
        self.assertEqual(first["absent_classes"], [])
        self.assertFalse(windows[1]["all_120_calendar_months_present"])
        self.assertEqual(report["tables"]["rolling_training_classes"].num_rows, 40)


class CombinedTargetTests(unittest.TestCase):
    def test_verified_annual_concatenation_retains_values_and_pins_denominators(self):
        first, second = target_fixture(count=12), target_fixture(first="2011-01", count=12)
        pins = {"2010": {"targets_sha256": first.artifact_sha256},
                "2011": {"targets_sha256": second.artifact_sha256}}
        panel = combine_target_panels([second, first], source_year_pins=pins)
        verify_target_panel(panel)
        self.assertEqual(panel.table.num_rows, 240)
        self.assertEqual(panel.diagnostics.num_rows, 24)
        self.assertEqual(panel.provenance["source_year_pins"], pins)
        self.assertEqual(panel.provenance["label_observed_through"], "2011-12-31")
        expected = sorted(first.table.to_pylist() + second.table.to_pylist(),
                          key=lambda row: (row["signal_date"], row["source_security_id"]))
        self.assertEqual(panel.table.to_pylist(), expected)

    def test_overlaps_missing_months_and_missing_year_pins_fail(self):
        panel = target_fixture(count=12)
        for panels, pins in (([panel, panel], {"2010": "a"}),
                             ([panel], {"2011": "a"}),
                             ([target_fixture(count=1), target_fixture(first="2010-03", count=1)], {"2010": "a"})):
            with self.subTest(pins=pins), self.assertRaises(DataContractError):
                combine_target_panels(panels, source_year_pins=pins)

    def test_mutated_annual_values_rejected_before_combination(self):
        panel = target_fixture(count=1)
        changed = panel.table.set_column(panel.table.schema.get_field_index("label_return"), "label_return",
                                         pa.array([0.1] * 10))
        with self.assertRaises(DataContractError):
            combine_target_panels([replace(panel, table=changed)], source_year_pins={"2010": "a"})


class RetCoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel = target_fixture(count=121, variable_members=True)

    def test_exact_120_months_and_canonical_monthly_rolls(self):
        report = audit_ret_coverage(self.panel, ["2019-11-30", "2019-12-31", "2020-01-31"])
        rows = report["tables"]["ret_coverage_signal"].to_pylist()
        self.assertEqual([row["vector_available"] for row in rows], [False, True, True])
        self.assertEqual([row["first_outcome_month"] for row in rows], ["2009-12", "2010-01", "2010-02"])
        self.assertEqual(rows[0]["unavailable_month_class_cells"], 10)
        self.assertEqual(report["tables"]["ret_month_class_grid"].num_rows, 3600)
        self.assertEqual(report["tables"]["ret_class_payoffs"].num_rows, 30)
        self.assertEqual(report["summary"]["canonical_worker_verified_signals"], 2)
        self.assertEqual(len(report["canonical_payoffs"]), 2)

    def test_full_168_signal_history_has_120_warmup_and_48_nominal_test_vectors(self):
        panel = target_fixture(count=168)
        signals = [row["signal_date"] for row in panel.provenance["month_windows"]]
        report = audit_ret_coverage(panel, signals)
        self.assertEqual(report["summary"]["signals"], 168)
        self.assertEqual(report["summary"]["withheld_vectors"], 120)
        self.assertEqual(report["summary"]["available_vectors"], 48)
        self.assertEqual(report["summary"]["first_available_signal"], "2019-12-31")
        self.assertEqual(report["summary"]["canonical_worker_verified_signals"], 48)
        self.assertEqual(report["summary"]["nominal_test_available_vectors"], 48)
        self.assertEqual(report["tables"]["ret_month_class_grid"].num_rows, 201600)
        self.assertEqual(report["tables"]["ret_class_payoffs"].num_rows, 1680)
        self.assertEqual(report["summary"]["maximum_financial_outcome_end_consumed"], "2023-11-30")

    def test_equal_month_means_match_canonical_worker_not_pooled_stock_means(self):
        report = audit_ret_coverage(self.panel, ["2019-12-31"])
        means = report["tables"]["ret_month_class_grid"].to_pylist()
        for row in report["tables"]["ret_class_payoffs"].to_pylist():
            expected = np.mean([month["class_mean_return"] for month in means if month["class_id"] == row["class_id"]])
            self.assertAlmostEqual(row["payoff"], expected, places=14)
        self.assertTrue(report["summary"]["equal_month_weighting"])

    def test_missing_calendar_month_and_empty_class_propagate_to_full_vector(self):
        for panel, status in ((target_fixture(absent_month="2015-06"), "missing_month_completion_evidence"),
                              (target_fixture(tied_month="2015-06"), "empty_realized_class")):
            with self.subTest(status=status):
                report = audit_ret_coverage(panel, ["2019-12-31"])
                self.assertEqual(report["tables"]["ret_class_payoffs"]["payoff"].null_count, 10)
                self.assertFalse(report["summary"]["available_vectors"])
                self.assertIn(status, report["tables"]["ret_month_class_grid"]["status"].to_pylist())

    def test_future_target_mutations_cannot_change_prior_signal_audit(self):
        initial = audit_ret_coverage(self.panel, ["2019-12-31"])
        future = pc.greater(self.panel.table["outcome_month"], "2019-12")
        changed = self.panel.table.set_column(self.panel.table.schema.get_field_index("label_return"), "label_return",
                                             pc.if_else(future, float("nan"), self.panel.table["label_return"]))
        changed = changed.set_column(changed.schema.get_field_index("target_class"), "target_class",
                                     pc.if_else(future, pa.scalar(0, pa.int8()), changed["target_class"]))
        again = audit_ret_coverage(replace(self.panel, table=changed), ["2019-12-31"])
        for name in initial["tables"]:
            self.assertTrue(initial["tables"][name].equals(again["tables"][name]))
        self.assertEqual(initial["summary"], again["summary"])
        self.assertLessEqual(initial["summary"]["maximum_financial_outcome_end_consumed"], "2019-12-31")

    def test_early_signal_and_midmonth_cutoff_remain_explicitly_incomplete(self):
        report = audit_ret_coverage(self.panel, ["2009-12-31", "2019-12-15"])
        rows = report["tables"]["ret_coverage_signal"].to_pylist()
        self.assertFalse(any(row["vector_available"] for row in rows))
        self.assertEqual(rows[0]["calendar_months_completed_by_signal"], 0)
        self.assertEqual(rows[0]["unavailable_month_class_cells"], 1200)
        self.assertIn("month_not_observed_by_cutoff", report["tables"]["ret_month_class_grid"]["status"].to_pylist())

    def test_duplicate_protected_or_external_canonical_signals_fail(self):
        for dates, canonical in ((["2019-12-31"] * 2, None), (["2024-01-31"], None),
                                 (["2019-12-31"], ["2020-01-31"])):
            with self.subTest(dates=dates), self.assertRaises(DataContractError):
                audit_ret_coverage(self.panel, dates, canonical_signal_dates=canonical)


if __name__ == "__main__":
    unittest.main()
