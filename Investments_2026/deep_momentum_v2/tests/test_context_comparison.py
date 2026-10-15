"""Policy sensitivity checks preserve causal inputs and original denominators."""
import unittest

import numpy as np
import pandas as pd

from dmv2.context_comparison import (
    PERIODS, cohort_composition, compare_diagnostics, compare_feature_frames,
    value_periods, _composition_periods,
)
from dmv2.context_features import balanced_cross_sectional_bases
from dmv2.universe import DataContractError


SPECS = (
    {"name": "paper", "role": "paper", "family": "paper"},
    {"name": "technical", "role": "technical", "family": "price"},
    {"name": "context", "role": "cross_sectional", "family": "cross_sectional"},
    {"name": "excluded", "role": "market_state", "family": "market", "model_eligible": False},
)


def fixture(signal="2009-12-31"):
    frame = pd.DataFrame({"source_security_id": ["a", "b", "c"], "signal_date": signal,
                          "paper": [1., 2., np.nan], "technical": [3., 4., np.nan],
                          "context": [1., np.nan, np.nan], "excluded": [np.nan] * 3})
    sensitivity = frame.copy()
    sensitivity["context"] = [1., 2., np.nan]
    ledger = frame[["source_security_id", "signal_date"]].assign(paper_prediction_eligible=[True, True, False])
    return frame, sensitivity, ledger


def diagnostics(signal="2009-12-31", observed=8, cohort=10, status="below_coverage_threshold"):
    return pd.DataFrame([{
        "signal_date": signal, "feature_name": "xs_universe_rank_pct__px_simple_return_1s",
        "context_group": "universe", "cohort_size": cohort, "observed_members": observed,
        "coverage_fraction": observed / cohort, "latest_input_date": signal,
        "identity_eligible_members": 10., "requested_identity_eligible_members": 10.,
        "request_exhaustive": True, "status": status,
    }])


class ContextComparisonTests(unittest.TestCase):
    def test_gain_counts_preserve_original_and_paper_denominators(self):
        a, b, ledger = fixture()
        result = compare_feature_frames(a, b, ledger, SPECS, outcome_year=2010)
        original = result[result.population == "original"].iloc[0]
        qualified = result[result.population == "paper_qualified"].iloc[0]
        self.assertEqual((original.rows, original.reference_finite_rows, original.sensitivity_finite_rows), (3, 1, 2))
        self.assertEqual((qualified.rows, qualified.gained_finite_rows), (2, 1))
        self.assertEqual(original.shared_abs_difference_max, 0.)
        self.assertEqual(original.changed_shared_values, 0)
        self.assertEqual(original.gained_value_mean, 2.)
        self.assertEqual(original.gained_value_std_population, 0.)
        self.assertEqual(set(result.feature_name), {"context"})

    def test_lost_and_changed_shared_context_are_rejected(self):
        for replacement, message in ((np.nan, "lost finite"), (1.0000000001, "Shared finite")):
            a, b, ledger = fixture()
            b.loc[0, "context"] = replacement
            with self.subTest(replacement=replacement), self.assertRaisesRegex(DataContractError, message):
                compare_feature_frames(a, b, ledger, SPECS, outcome_year=2010)

    def test_paper_technical_alias_and_original_keys_cannot_change(self):
        for name in ("paper", "technical", "excluded"):
            a, b, ledger = fixture()
            b.loc[0, name] = 5.
            if name == "excluded":
                # Even excluded context definitions must respect shared values.
                a.loc[0, name] = 1.
            with self.subTest(name=name), self.assertRaises(DataContractError):
                compare_feature_frames(a, b, ledger, SPECS, outcome_year=2010)
        a, b, ledger = fixture()
        with self.assertRaisesRegex(DataContractError, "historical keys"):
            compare_feature_frames(a, b.iloc[:2], ledger, SPECS, outcome_year=2010)

    def test_outcome_year_uses_next_month_and_rejects_protected_dates(self):
        a, b, ledger = fixture("2015-12-31")
        compare_feature_frames(a, b, ledger, SPECS, outcome_year=2016)
        with self.assertRaisesRegex(DataContractError, "next-month"):
            compare_feature_frames(a, b, ledger, SPECS, outcome_year=2015)
        a, b, ledger = fixture("2024-01-31")
        with self.assertRaisesRegex(DataContractError, "protected"):
            compare_feature_frames(a, b, ledger, SPECS, outcome_year=2024)

    def test_period_counts_pool_denominators_and_pre2016_boundary(self):
        parts = []
        for year in (2010, 2015, 2016, 2020):
            a, b, ledger = fixture(f"{year-1}-12-31")
            parts.append(compare_feature_frames(a, b, ledger, SPECS, outcome_year=year))
        result = value_periods(pd.concat(parts, ignore_index=True))
        self.assertEqual(set(result.period), set(PERIODS))
        original = result[result.population == "original"].set_index("period")
        self.assertEqual(original.loc["before_2016_2010_2015", "rows"], 6)
        self.assertEqual(original.loc["initial_train_2010_2019", "rows"], 9)
        self.assertEqual(original.loc["nominal_test_2020_2023", "rows"], 3)
        self.assertEqual(original.loc["full_history_2010_2023", "gained_finite_rows"], 4)
        self.assertAlmostEqual(original.loc["full_history_2010_2023", "sensitivity_finite_fraction"], 2/3)

    def test_gate_comparison_checks_same_observed_population(self):
        a, b = diagnostics(), diagnostics(status="ok")
        result = compare_diagnostics(a, b, outcome_year=2010)
        self.assertTrue(result.newly_status_ok.iloc[0])
        b.loc[0, "observed_members"] = 9
        with self.assertRaisesRegex(DataContractError, "contributor populations"):
            compare_diagnostics(a, b, outcome_year=2010)
        b = diagnostics(status="ok")
        b.loc[0, "cohort_size"] = 9
        with self.assertRaisesRegex(DataContractError, "membership and requests"):
            compare_diagnostics(a, b, outcome_year=2010)

    def test_daily_overlap_roles_assign_december_to_following_outcome_year(self):
        a = pd.concat([diagnostics(signal=day) for day in ("2008-12-31", "2009-12-31", "2010-11-30")], ignore_index=True)
        b = a.assign(status="ok")
        result = compare_diagnostics(a, b, outcome_year=2010, daily=True).set_index("signal_date")
        self.assertEqual(result.loc["2008-12-31", "date_role"], "overlapping_warmup")
        self.assertEqual(result.loc["2009-12-31", "date_role"], "primary_signal_year")
        self.assertEqual(result.loc["2010-11-30", "date_role"], "primary_signal_year")

    def test_composition_keeps_identity_blocked_and_missing_inputs_separate(self):
        bases = balanced_cross_sectional_bases()
        signal = "2009-12-31"
        feature = pd.DataFrame({"source_security_id": [f"s{i}" for i in range(10)], "signal_date": signal,
                                **{name: np.r_[np.arange(8), np.nan, np.nan] for name in bases}})
        members = feature[["source_security_id", "signal_date"]].assign(
            indices=[["sp500"]] * 5 + [["sp400"]] * 5)
        ledger = feature[["source_security_id", "signal_date"]].assign(
            identity_eligible=[True] * 9 + [False], requested=[True] * 9 + [False],
            identity_review_required=[True] * 10, paper_prediction_eligible=[True] * 8 + [False] * 2)
        records = []
        for group, count, observed in (("universe", 10, 8), ("sp500", 5, 5), ("sp400", 5, 3)):
            scope = "universe" if group == "universe" else "index_sleeve"
            for base in bases:
                records.append({"signal_date": signal, "feature_name": f"xs_{scope}_rank_pct__{base}",
                    "context_group": group, "cohort_size": count, "reference_observed_members": observed,
                    "sensitivity_observed_members": observed, "reference_status_ok": observed/count >= .9,
                    "sensitivity_status_ok": observed/count >= .8})
        result = cohort_composition(feature, members, ledger, pd.DataFrame(records), outcome_year=2010)
        universe = result[(result.context_group == "universe") & (result.base_feature == bases[0])].iloc[0]
        self.assertEqual(universe.passing_category, "80_only")
        self.assertEqual((universe.observed_members, universe.omitted_members), (8, 2))
        self.assertEqual(universe.omitted_identity_blocked_members, 1)
        self.assertEqual(universe.omitted_base_input_missing_members, 1)
        self.assertEqual(universe.observed_sp400_members, 3)
        self.assertEqual(universe.omitted_sp400_members, 2)
        self.assertEqual(universe.observed_provisional_review_members, 8)
        period = _composition_periods(result)
        row = period[(period.period == "before_2016_2010_2015") & (period.context_group == "universe")].iloc[0]
        self.assertEqual(row.feature_date_cohort_groups, 80)
        self.assertEqual(row.cohort_size, 800)
        self.assertEqual(row.observed_fraction, .8)
        self.assertEqual(row.observed_sp400_fraction, 3/8)
        self.assertEqual(row.omitted_identity_blocked_members, 80)


if __name__ == "__main__":
    unittest.main()
