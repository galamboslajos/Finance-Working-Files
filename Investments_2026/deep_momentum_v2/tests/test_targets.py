"""Structural monthly-label cases; no source reads or financial experiment."""
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import unittest

import pandas as pd
import pyarrow as pa

from dmv2.monthly import MonthlyReturnPanel, MonthWindow
from dmv2.targets import (CLASS_IDS, TARGET_POLICY, build_monthly_targets,
                          target_panel_fingerprint, verify_target_panel)
from dmv2.universe import DataContractError


JAN = MonthWindow("2010-01", "2009-12-31", "2010-01-04", "2010-01-29", ())
FEB = MonthWindow("2010-02", "2010-01-29", "2010-02-01", "2010-02-26", ())


def fixture(values=None, *, windows=(JAN,), count=20):
    values = list(range(count)) if values is None else list(values)
    cohort, outcomes = [], []
    for window in windows:
        for i, value in enumerate(values):
            cohort.append({"source_security_id": f"source-{i:03d}", "signal_date": window.signal_date,
                           "identity_eligible": True, "requested": True,
                           "paper_prediction_eligible": True})
            outcomes.append({"source_security_id": f"source-{i:03d}", "signal_date": window.signal_date,
                "outcome_month": window.outcome_month, "outcome_start": window.outcome_start,
                "outcome_end": window.outcome_end, "requested": True, "return_status": "ok",
                "provider_adjusted_return": float(value) / 100})
    return pd.DataFrame(cohort), pd.DataFrame(outcomes)


def run(values=None, *, windows=(JAN,), observed="2010-02-26", count=20):
    cohort, outcomes = fixture(values, windows=windows, count=count)
    return build_monthly_targets(cohort, outcomes, label_observed_through=observed,
                                 month_windows=windows)


class FinancialTrap(float):
    calls = 0

    def __float__(self):
        type(self).calls += 1
        raise AssertionError("An unapproved financial value was converted")


class MonthlyTargetTests(unittest.TestCase):
    def test_twenty_names_descending_two_per_class(self):
        panel = run()
        rows = panel.table.to_pylist()
        self.assertEqual([row["target_class"] for row in rows], [10 - i // 2 for i in range(20)])
        self.assertEqual(len(rows), 20)
        self.assertTrue(all(row["prediction_eligible"] and row["target_available"] for row in rows))
        self.assertEqual(panel.diagnostics.to_pylist()[0]["empty_classes"], [])
        self.assertEqual(panel.provenance["policy_id"], TARGET_POLICY)
        self.assertFalse(panel.provenance["model_ready"])

    def test_exact_boundary_midrank_not_float_floor(self):
        # This covers all ten boundaries repeatedly, using rational arithmetic.
        panel = run(count=1000)
        classes = panel.table.column("target_class").to_pylist()
        self.assertEqual(classes, [10 - i // 100 for i in range(1000)])

    def test_equal_values_share_one_class_without_id_jitter(self):
        panel = run([0] * 20)
        self.assertEqual(set(panel.table.column("target_class").to_pylist()), {6})
        self.assertEqual(panel.diagnostics.to_pylist()[0]["empty_classes"], [1, 2, 3, 4, 5, 7, 8, 9, 10])

    def test_ties_cross_boundaries_without_splitting(self):
        panel = run([0] * 10 + [1] * 10)
        self.assertEqual(panel.table.column("target_class").to_pylist(), [8] * 10 + [3] * 10)

    def test_tiny_unequal_values_are_not_rounded_into_ties(self):
        panel = run([i * 1e-14 for i in range(20)])
        self.assertEqual(set(panel.table.column("target_class").to_pylist()), set(CLASS_IDS))

    def test_input_reordering_preserves_class_per_original_key(self):
        cohort, outcomes = fixture([0] * 5 + list(range(15)))
        first = build_monthly_targets(cohort, outcomes, label_observed_through=JAN.outcome_end, month_windows=(JAN,))
        second = build_monthly_targets(cohort.sample(frac=1, random_state=7), outcomes.sample(frac=1, random_state=9),
                                       label_observed_through=JAN.outcome_end, month_windows=(JAN,))
        def classes(panel):
            return {r["source_security_id"]: r["target_class"] for r in panel.table.to_pylist()}
        self.assertEqual(classes(first), classes(second))

    def test_months_rank_independently(self):
        cohort, outcomes = fixture(windows=(JAN, FEB))
        outcomes.loc[outcomes.outcome_month == FEB.outcome_month, "provider_adjusted_return"] *= -1
        panel = build_monthly_targets(cohort, outcomes, label_observed_through=FEB.outcome_end, month_windows=(JAN, FEB))
        rows = panel.table.to_pylist()
        self.assertEqual(rows[0]["target_class"], 10)
        self.assertEqual(rows[20]["target_class"], 1)

    def test_minimum_ten_observed_withholds_small_month(self):
        panel = run(count=9)
        self.assertTrue(all(row["prediction_eligible"] and row["observed_return_available"] for row in panel.table.to_pylist()))
        self.assertTrue(all(not row["target_available"] and row["target_class"] is None for row in panel.table.to_pylist()))
        self.assertEqual(panel.diagnostics.to_pylist()[0]["rank_status"], "insufficient_observed_population")

    def test_exhaustive_request_gate_is_not_eighty_percent_target_rule(self):
        cohort, outcomes = fixture()
        cohort.loc[19, "requested"] = False
        panel = build_monthly_targets(cohort, outcomes, label_observed_through=JAN.outcome_end, month_windows=(JAN,))
        self.assertTrue(all(row["prediction_eligible"] for row in panel.table.to_pylist()))
        self.assertFalse(any(row["target_available"] for row in panel.table.to_pylist()))
        self.assertEqual(panel.diagnostics.to_pylist()[0]["rank_status"], "non_exhaustive_original_requests")

    def test_forward_requests_are_independent_from_past_feature_requests(self):
        cohort, outcomes = fixture()
        outcomes.loc[11:, "requested"] = False
        outcomes.loc[11:, "return_status"] = "not_requested"
        panel = build_monthly_targets(cohort, outcomes, label_observed_through=JAN.outcome_end, month_windows=(JAN,))
        rows = panel.table.to_pylist()
        self.assertTrue(all(row["requested"] and row["prediction_eligible"] for row in rows))
        self.assertFalse(any(row["target_available"] for row in rows))
        self.assertEqual(sum(row["observed_return_available"] for row in rows), 11)
        self.assertEqual(panel.diagnostics.to_pylist()[0]["outcome_unrequested_identity_members"], 9)
        self.assertFalse(panel.diagnostics.to_pylist()[0]["exhaustive_requests"])

    def test_not_requested_status_blocks_even_if_forward_flag_claims_true(self):
        cohort, outcomes = fixture()
        outcomes.loc[19, "return_status"] = "not_requested"
        panel = build_monthly_targets(cohort, outcomes, label_observed_through=JAN.outcome_end, month_windows=(JAN,))
        self.assertFalse(any(panel.table.column("target_available").to_pylist()))
        self.assertEqual(panel.diagnostics.to_pylist()[0]["source_not_requested_identity_members"], 1)

    def test_forward_requests_include_paper_ineligible_original_member(self):
        cohort, outcomes = fixture()
        cohort.loc[19, "paper_prediction_eligible"] = False
        outcomes.loc[19, "return_status"] = "not_requested"
        panel = build_monthly_targets(cohort, outcomes, label_observed_through=JAN.outcome_end, month_windows=(JAN,))
        self.assertFalse(any(panel.table.column("target_available").to_pylist()))
        self.assertEqual(panel.diagnostics.to_pylist()[0]["source_not_requested_identity_members"], 1)

    def test_missing_outcome_record_withholds_unknown_request_coverage(self):
        cohort, outcomes = fixture()
        panel = build_monthly_targets(cohort, outcomes.iloc[:-1], label_observed_through=JAN.outcome_end, month_windows=(JAN,))
        self.assertFalse(any(panel.table.column("target_available").to_pylist()))
        self.assertIsNone(panel.table.to_pylist()[-1]["outcome_requested"])
        self.assertEqual(panel.diagnostics.to_pylist()[0]["outcome_request_unknown_identity_members"], 1)

    def test_explicit_forward_request_metadata_required_for_bare_table(self):
        cohort, outcomes = fixture()
        with self.assertRaises(DataContractError):
            build_monthly_targets(cohort, outcomes.drop(columns="requested"), label_observed_through=JAN.outcome_end, month_windows=(JAN,))
        outcomes["requested"] = outcomes.requested.astype(object)
        outcomes.loc[0, "requested"] = "True"
        with self.assertRaises(DataContractError):
            build_monthly_targets(cohort, outcomes, label_observed_through=JAN.outcome_end, month_windows=(JAN,))

    def test_forward_unrequested_numeric_values_not_decoded(self):
        cohort, outcomes = fixture()
        outcomes["provider_adjusted_return"] = outcomes.provider_adjusted_return.astype(object)
        outcomes.at[0, "requested"] = False
        outcomes.at[0, "provider_adjusted_return"] = FinancialTrap(7)
        panel = build_monthly_targets(cohort, outcomes, label_observed_through=JAN.outcome_end, month_windows=(JAN,))
        self.assertEqual(panel.table.to_pylist()[0]["target_status"], "outcome_not_requested")
        self.assertFalse(any(panel.table.column("target_available").to_pylist()))

    def test_wrapped_request_manifest_derives_forward_flags_without_quote_presence(self):
        cohort, outcomes = fixture()
        outcomes["intrinio_security_id"] = [f"native-{i}" for i in range(20)]
        outcomes = outcomes.drop(columns="requested")
        source = MonthlyReturnPanel(pa.Table.from_pandas(outcomes, preserve_index=False), {},
            {"requested_security_ids": [f"native-{i}" for i in range(11)]}, (JAN,))
        panel = build_monthly_targets(cohort, source, label_observed_through=JAN.outcome_end)
        self.assertEqual(sum(panel.table.column("outcome_requested").to_pylist()), 11)
        self.assertFalse(any(panel.table.column("target_available").to_pylist()))
        self.assertEqual(source.table.column_names.count("requested"), 0)

    def test_wrapped_explicit_requests_must_match_declared_manifest(self):
        cohort, outcomes = fixture()
        outcomes["intrinio_security_id"] = [f"native-{i}" for i in range(20)]
        source = MonthlyReturnPanel(pa.Table.from_pandas(outcomes, preserve_index=False), {},
            {"requested_security_ids": ["native-0"]}, (JAN,))
        with self.assertRaisesRegex(DataContractError, "disagree"):
            build_monthly_targets(cohort, source, label_observed_through=JAN.outcome_end)

    def test_blocked_unrequested_identity_does_not_block_request_gate(self):
        cohort, outcomes = fixture()
        cohort.loc[19, ["identity_eligible", "requested", "paper_prediction_eligible"]] = False
        panel = build_monthly_targets(cohort, outcomes, label_observed_through=JAN.outcome_end, month_windows=(JAN,))
        self.assertEqual(sum(panel.table.column("target_available").to_pylist()), 19)
        self.assertEqual(panel.table.to_pylist()[19]["target_status"], "identity_ineligible")

    def test_paper_ineligible_outcome_does_not_change_ranks(self):
        cohort, outcomes = fixture()
        cohort.loc[19, "paper_prediction_eligible"] = False
        outcomes.loc[19, "provider_adjusted_return"] = 9999
        panel = build_monthly_targets(cohort, outcomes, label_observed_through=JAN.outcome_end, month_windows=(JAN,))
        last = panel.table.to_pylist()[19]
        self.assertFalse(last["prediction_eligible"])
        self.assertFalse(last["target_available"])
        self.assertIsNone(last["label_return"])
        self.assertEqual(panel.diagnostics.to_pylist()[0]["prediction_eligible_members"], 19)

    def test_missing_outcome_record_retains_original_key_and_no_imputation(self):
        cohort, outcomes = fixture()
        panel = build_monthly_targets(cohort, outcomes.iloc[:-1], label_observed_through=JAN.outcome_end, month_windows=(JAN,))
        self.assertEqual(panel.table.num_rows, len(cohort))
        self.assertEqual(panel.table.to_pylist()[-1]["target_status"], "missing_outcome_record")
        self.assertTrue(panel.table.to_pylist()[-1]["prediction_eligible"])
        self.assertEqual(panel.diagnostics.to_pylist()[0]["observed_original_coverage"], 19 / 20)

    def test_null_nonfinite_bad_source_are_distinct_and_never_imputed(self):
        cohort, outcomes = fixture()
        outcomes.loc[0, "provider_adjusted_return"] = None
        outcomes.loc[1, "provider_adjusted_return"] = float("inf")
        outcomes.loc[2, "return_status"] = "missing_end"
        outcomes.loc[2, "provider_adjusted_return"] = 5
        outcomes.loc[3, "provider_adjusted_return"] = -1.01
        panel = build_monthly_targets(cohort, outcomes, label_observed_through=JAN.outcome_end, month_windows=(JAN,))
        rows = panel.table.to_pylist()
        self.assertEqual(rows[0]["target_status"], "nonfinite_return")
        self.assertEqual(rows[1]["target_status"], "nonfinite_return")
        self.assertEqual(rows[2]["target_status"], "source_return_status:missing_end")
        self.assertEqual(rows[3]["target_status"], "invalid_return_lower_bound")
        self.assertTrue(all(row["label_return"] is None and not row["target_available"] for row in rows[:4]))
        self.assertTrue(all(row["prediction_eligible"] for row in rows))

    def test_minus_one_and_unbounded_finite_gain_not_winsorized(self):
        cohort, outcomes = fixture()
        outcomes.loc[0, "provider_adjusted_return"] = -1
        outcomes.loc[19, "provider_adjusted_return"] = 12345
        panel = build_monthly_targets(cohort, outcomes, label_observed_through=JAN.outcome_end, month_windows=(JAN,))
        rows = panel.table.to_pylist()
        self.assertEqual(rows[0]["label_return"], -1)
        self.assertEqual(rows[19]["label_return"], 12345)
        self.assertEqual(rows[0]["target_class"], 10)
        self.assertEqual(rows[19]["target_class"], 1)

    def test_return_strings_and_booleans_are_not_coerced(self):
        for bad in ("0.10", True):
            cohort, outcomes = fixture()
            outcomes["provider_adjusted_return"] = outcomes.provider_adjusted_return.astype(object)
            outcomes.loc[0, "provider_adjusted_return"] = bad
            with self.subTest(bad=bad), self.assertRaises(DataContractError):
                build_monthly_targets(cohort, outcomes, label_observed_through=JAN.outcome_end, month_windows=(JAN,))

    def test_boolean_metadata_requires_true_booleans(self):
        for field in ("identity_eligible", "requested", "paper_prediction_eligible"):
            for bad in (1, "True", None):
                cohort, outcomes = fixture()
                cohort[field] = cohort[field].astype(object)
                cohort.loc[0, field] = bad
                with self.subTest(field=field, bad=bad), self.assertRaises(DataContractError):
                    build_monthly_targets(cohort, outcomes, label_observed_through=JAN.outcome_end, month_windows=(JAN,))

    def test_future_outcomes_are_withheld_before_financial_conversion(self):
        cohort, outcomes = fixture(windows=(JAN, FEB))
        outcomes["provider_adjusted_return"] = outcomes.provider_adjusted_return.astype(object)
        outcomes["return_status"] = outcomes.return_status.astype(object)
        for i in range(20, 40):
            outcomes.at[i, "provider_adjusted_return"] = FinancialTrap(7)
            outcomes.at[i, "return_status"] = object()
        FinancialTrap.calls = 0
        panel = build_monthly_targets(cohort, outcomes, label_observed_through=JAN.outcome_end, month_windows=(JAN, FEB))
        self.assertEqual(FinancialTrap.calls, 0)
        self.assertTrue(all(row["target_status"] == "outcome_not_completed" for row in panel.table.to_pylist()[20:]))

    def test_protected_extra_metadata_rejected_before_any_value_conversion(self):
        cohort, outcomes = fixture()
        outcomes["provider_adjusted_return"] = outcomes.provider_adjusted_return.astype(object)
        outcomes.at[0, "provider_adjusted_return"] = FinancialTrap(7)
        protected = outcomes.iloc[-1].copy()
        protected["source_security_id"] = "extra-protected"
        protected["signal_date"] = "2023-12-29"
        protected["outcome_month"] = "2024-01"
        protected["outcome_start"] = "2024-01-02"
        protected["outcome_end"] = "2024-01-31"
        outcomes = pd.concat([outcomes, protected.to_frame().T], ignore_index=True)
        FinancialTrap.calls = 0
        with self.assertRaisesRegex(DataContractError, "protected 2024"):
            build_monthly_targets(cohort, outcomes, label_observed_through=JAN.outcome_end, month_windows=(JAN,))
        self.assertEqual(FinancialTrap.calls, 0)

    def test_bad_source_status_does_not_decode_flagged_value(self):
        cohort, outcomes = fixture()
        outcomes["provider_adjusted_return"] = outcomes.provider_adjusted_return.astype(object)
        outcomes.at[0, "provider_adjusted_return"] = FinancialTrap(7)
        outcomes.at[0, "return_status"] = "source_quality_flag"
        panel = build_monthly_targets(cohort, outcomes, label_observed_through=JAN.outcome_end, month_windows=(JAN,))
        self.assertEqual(panel.table.to_pylist()[0]["target_status"], "source_return_status:source_quality_flag")

    def test_duplicate_or_extra_outcome_keys_fail(self):
        cohort, outcomes = fixture()
        for bad in (pd.concat([outcomes, outcomes.iloc[:1]], ignore_index=True),
                    pd.concat([outcomes, outcomes.iloc[:1].assign(source_security_id="outsider")], ignore_index=True)):
            with self.assertRaises(DataContractError):
                build_monthly_targets(cohort, bad, label_observed_through=JAN.outcome_end, month_windows=(JAN,))

    def test_original_duplicate_keys_fail(self):
        cohort, outcomes = fixture()
        with self.assertRaises(DataContractError):
            build_monthly_targets(pd.concat([cohort, cohort.iloc[:1]]), outcomes,
                                  label_observed_through=JAN.outcome_end, month_windows=(JAN,))

    def test_independent_calendar_mandatory_and_must_match(self):
        cohort, outcomes = fixture()
        with self.assertRaises(DataContractError):
            build_monthly_targets(cohort, outcomes, label_observed_through=JAN.outcome_end)
        outcomes.loc[0, "outcome_end"] = "2010-01-28"
        with self.assertRaises(DataContractError):
            build_monthly_targets(cohort, outcomes, label_observed_through=JAN.outcome_end, month_windows=(JAN,))

    def test_same_month_target_invalid(self):
        cohort, outcomes = fixture()
        bad = {"signal_date": JAN.signal_date, "outcome_month": "2009-12", "outcome_start": "2009-12-30", "outcome_end": JAN.signal_date}
        with self.assertRaises(DataContractError):
            build_monthly_targets(cohort, outcomes, label_observed_through=JAN.outcome_end, month_windows=(bad,))

    def test_monthly_return_panel_windows_and_arrow_inputs(self):
        cohort, outcomes = fixture()
        source = MonthlyReturnPanel(pa.Table.from_pandas(outcomes, preserve_index=False), {}, {"source": "test"}, (JAN,))
        panel = build_monthly_targets(pa.Table.from_pandas(cohort, preserve_index=False), source,
                                     label_observed_through=JAN.outcome_end)
        self.assertEqual(panel.table.num_rows, 20)
        self.assertIsNotNone(panel.provenance["source_outcome_provenance_sha256"])

    def test_verifier_binds_tables_diagnostics_and_mutable_provenance(self):
        panel = run()
        changed_values = panel.table.set_column(panel.table.schema.get_field_index("label_return"), "label_return", pa.array([7.] * 20))
        with self.assertRaisesRegex(DataContractError, "monthly value fingerprint"):
            verify_target_panel(replace(panel, table=changed_values))
        with self.assertRaises(DataContractError):
            verify_target_panel(replace(panel, diagnostics=panel.diagnostics.slice(0, 0)))
        changed_provenance = deepcopy(panel.provenance)
        changed_provenance["limitations"].append("unreviewed")
        with self.assertRaisesRegex(DataContractError, "provenance fingerprint"):
            verify_target_panel(replace(panel, provenance=changed_provenance))
        with self.assertRaises(FrozenInstanceError):
            panel.artifact_sha256 = "fake"

    def test_historical_subset_verification_does_not_consult_later_values(self):
        panel = run(windows=(JAN, FEB))
        values = panel.table.column("label_return").to_pylist()
        values[-1] = 999
        changed = replace(panel, table=panel.table.set_column(panel.table.schema.get_field_index("label_return"), "label_return", pa.array(values)))
        verify_target_panel(changed, through=JAN.outcome_end, start_month="2010-01")
        with self.assertRaises(DataContractError):
            verify_target_panel(changed)
        self.assertEqual(target_panel_fingerprint(panel), panel.artifact_sha256)

    def test_subset_still_rejects_later_date_metadata_mutation(self):
        panel = run(windows=(JAN, FEB))
        dates = panel.table.column("outcome_end").to_pylist()
        dates[-1] = "2024-01-31"
        changed = replace(panel, table=panel.table.set_column(panel.table.schema.get_field_index("outcome_end"), "outcome_end", pa.array(dates)))
        with self.assertRaises(DataContractError):
            verify_target_panel(changed, through=JAN.outcome_end)

    def test_unknown_policy_and_protected_observation_cutoff_fail(self):
        cohort, outcomes = fixture()
        for kwargs in ({"policy": "paper_unknown_ties"}, {"label_observed_through": "2024-01-01"}):
            arguments = {"label_observed_through": JAN.outcome_end, "month_windows": (JAN,), **kwargs}
            with self.assertRaises(DataContractError):
                build_monthly_targets(cohort, outcomes, **arguments)


if __name__ == "__main__":
    unittest.main()
