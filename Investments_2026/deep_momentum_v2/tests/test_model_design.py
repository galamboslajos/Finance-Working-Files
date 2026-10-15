"""Causal calendar boundaries for the unexecuted classifier design; no market data."""
from copy import deepcopy
from datetime import date, timedelta
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

from dmv2.model_design import (
    ModelDesignError, experiment_matrix, load_model_plan, outcome_schedule,
    readiness_report, validate_model_plan, validate_outcome_schedule,
)


ROOT = Path(__file__).resolve().parents[1]
DRAFT = ROOT / "configs" / "model_comparison.proposed.json"


def month_count(first, last):
    fy, fm = map(int, first[:7].split("-"))
    ly, lm = map(int, last[:7].split("-"))
    return (ly - fy) * 12 + lm - fm + 1


class ModelDesignTests(unittest.TestCase):
    def setUp(self):
        self.plan = load_model_plan(DRAFT)

    def test_primary_universe_is_user_selected_historical_combined(self):
        report = readiness_report(self.plan)
        self.assertEqual(report["primary_universe"], "combined")
        self.assertEqual(report["universe_selection"], "USER_SELECTED")
        self.assertEqual(self.plan["universe"]["membership_basis"], "historical_signal_date")
        self.assertEqual(self.plan["panels"]["primary"], "available_cohort_80pct")
        self.assertEqual(self.plan["panels"]["sensitivity"], "available_cohort_90pct")

    def test_four_arms_isolate_model_choice_from_added_inputs(self):
        matrix = experiment_matrix(self.plan)
        pairs = {(row["estimator"], row["feature_set"]) for row in matrix}
        self.assertEqual(len(matrix), 4)
        self.assertEqual(pairs, {(estimator, feature_set)
                               for estimator in ("xgboost_multiclass", "random_forest_multiclass")
                               for feature_set in ("paper15", "extended715_candidates")})
        self.assertTrue(all(row["retained_predictors"] == "determined_in_training_partition"
                            for row in matrix))
        self.assertTrue(all("missing_indicators" in row["transformed_columns"] for row in matrix))
        self.assertFalse(self.plan["preprocessing"]["extended_predictor_count_is_guaranteed"])

    def test_first_fit_separates_december_features_from_january_label(self):
        january = outcome_schedule(self.plan)[0]
        self.assertEqual(january["prediction_outcome_month"], "2020-01")
        self.assertEqual(january["feature_month"], "2019-12")
        self.assertEqual(january["train_outcome_start"], "2010-01-01")
        self.assertEqual(january["train_outcome_end"], "2019-12-31")
        self.assertEqual(january["train_first_feature_month"], "2009-12")
        self.assertEqual(january["train_last_feature_month"], "2019-11")
        self.assertLess(january["classifier_label_cutoff"], "2020-01-01")
        # December features are past inputs to the January test; their January
        # target is still outside this fit even though the features exist.
        self.assertGreater(january["feature_month"], january["train_last_feature_month"])

    def test_last_fold_rolls_to_2013_2022_without_protected_outcomes(self):
        rows = outcome_schedule(self.plan)
        last = rows[-1]
        self.assertEqual(last["prediction_outcome_month"], "2023-12")
        self.assertEqual(last["feature_month"], "2023-11")
        self.assertEqual(last["train_outcome_start"], "2013-01-01")
        self.assertEqual(last["train_outcome_end"], "2022-12-31")
        self.assertEqual(last["train_last_feature_month"], "2022-11")
        self.assertTrue(all(row["prediction_outcome_month"] < "2024-01" for row in rows))

    def test_inner_splits_are_96_24_completed_months_with_no_overlap(self):
        for row in outcome_schedule(self.plan):
            with self.subTest(month=row["prediction_outcome_month"]):
                self.assertEqual(month_count(row["development_outcome_start"], row["development_outcome_end"]), 96)
                self.assertEqual(month_count(row["validation_outcome_start"], row["validation_outcome_end"]), 24)
                self.assertEqual(month_count(row["train_outcome_start"], row["train_outcome_end"]), 120)
                self.assertEqual(date.fromisoformat(row["development_outcome_end"]) + timedelta(days=1),
                                 date.fromisoformat(row["validation_outcome_start"]))
                self.assertEqual(row["development_label_cutoff"], row["development_outcome_end"])
                self.assertEqual(row["validation_outcome_end"], row["classifier_label_cutoff"])
                self.assertLess(row["classifier_label_cutoff"], row["prediction_outcome_month"] + "-01")

    def test_classifier_is_annual_while_ret_rolls_only_completed_months(self):
        rows = outcome_schedule(self.plan)
        for row in rows:
            self.assertEqual(month_count(row["RET_outcome_start_month"], row["RET_outcome_end_month"]), 120)
            self.assertEqual(row["RET_outcome_end_month"], row["feature_month"])
            self.assertLess(row["RET_label_cutoff"], row["prediction_outcome_month"] + "-01")
        january, february, march = rows[:3]
        self.assertEqual(january["classifier_label_cutoff"], february["classifier_label_cutoff"])
        self.assertEqual(february["RET_outcome_end_month"], "2020-01")
        self.assertEqual(march["RET_label_cutoff"], "2020-02-29")
        self.assertEqual(january["RET_outcome_start_month"], "2010-01")
        self.assertEqual(february["RET_outcome_start_month"], "2010-02")
        self.assertGreater(february["RET_label_cutoff"], february["classifier_label_cutoff"])

    def test_months_are_unique_ordered_and_every_year_is_complete(self):
        rows = outcome_schedule(self.plan)
        months = [row["prediction_outcome_month"] for row in rows]
        self.assertEqual(len(months), 48)
        self.assertEqual(months, sorted(set(months)))
        self.assertEqual({month[:4] for month in months}, {"2020", "2021", "2022", "2023"})
        for year in range(2020, 2024):
            self.assertEqual(sum(month.startswith(str(year)) for month in months), 12)
        self.assertTrue(validate_outcome_schedule(rows, self.plan))

    def test_changed_training_cutoff_cannot_admit_first_forecast_label(self):
        rows = outcome_schedule(self.plan)
        rows[0]["classifier_label_cutoff"] = "2020-01-31"
        with self.assertRaisesRegex(ModelDesignError, "cutoff|partition"):
            validate_outcome_schedule(rows, self.plan)

    def test_ret_cannot_use_its_own_forecast_outcome(self):
        rows = outcome_schedule(self.plan)
        rows[1]["RET_outcome_end_month"] = "2020-02"
        rows[1]["RET_label_cutoff"] = "2020-02-29"
        with self.assertRaisesRegex(ModelDesignError, "cutoff|partition"):
            validate_outcome_schedule(rows, self.plan)

    def test_validation_cannot_overlap_development(self):
        rows = outcome_schedule(self.plan)
        rows[0]["validation_outcome_start"] = rows[0]["development_outcome_end"]
        with self.assertRaisesRegex(ModelDesignError, "cutoff|partition"):
            validate_outcome_schedule(rows, self.plan)

    def test_missing_duplicate_or_protected_forecast_month_is_rejected(self):
        rows = outcome_schedule(self.plan)
        for changed in (rows[:-1], rows + [deepcopy(rows[-1])], rows[::-1]):
            with self.subTest(count=len(changed)):
                with self.assertRaises(ModelDesignError):
                    validate_outcome_schedule(changed, self.plan)
        rows[-1]["prediction_outcome_month"] = "2024-01"
        with self.assertRaises(ModelDesignError):
            validate_outcome_schedule(rows, self.plan)

    def test_all_execution_enabling_flags_are_rejected(self):
        names = ("label_build_enabled", "preprocessing_fit_enabled", "model_fit_enabled",
                 "parameter_search_enabled", "backtest_enabled", "cloud_launch_enabled",
                 "dependency_install_enabled", "automatic_execution")
        for name in names:
            with self.subTest(flag=name):
                changed = deepcopy(self.plan)
                changed["execution"][name] = True
                with self.assertRaises(ModelDesignError):
                    validate_model_plan(changed)

    def test_development_pilot_cannot_open_outer_test_for_retuning(self):
        pilot = self.plan["execution"]["pilot_proposal"]
        self.assertEqual(pilot["scope"], "inner_development_validation_only")
        self.assertFalse(pilot["outer_test_metrics_inspected"])
        self.assertEqual(pilot["inner_training_outcome_end"], "2017-12-31")
        self.assertEqual(pilot["inner_validation_outcome_start"], "2018-01-01")
        self.assertEqual(pilot["inner_validation_outcome_end"], "2019-12-31")
        self.assertEqual(pilot["metrics_scope"], "classification_probabilities_only")
        for field, value in (("scope", "outer_2020_test"), ("outer_test_metrics_inspected", True),
                             ("outer_refit_enabled", True), ("RET_evaluation_enabled", True),
                             ("additional_pre2010_outcome_reads_enabled", True)):
            changed = deepcopy(self.plan)
            changed["execution"]["pilot_proposal"][field] = value
            with self.assertRaises(ModelDesignError):
                validate_model_plan(changed)

    def test_pilot_validation_has_insufficient_120month_ret_history_without_reading_more(self):
        pilot = self.plan["execution"]["pilot_proposal"]
        # The first validation outcome would need completed 2008-2017 payoffs;
        # the agreed 2010 history cannot supply that. Do not shorten RET or
        # silently read earlier labels just to give this development pilot a score.
        first_validation = pilot["inner_validation_outcome_start"]
        self.assertLess(month_count("2010-01", first_validation[:7]) - 1, 120)
        self.assertFalse(pilot["RET_evaluation_enabled"])
        self.assertFalse(pilot["additional_pre2010_outcome_reads_enabled"])

    def test_global_selection_future_membership_and_random_split_are_rejected(self):
        mutations = (("comparison", "full_history_coverage_selects_features", True),
                     ("comparison", "sensitivity_parameters_selected_on_test", True),
                     ("target_proposal", "prediction_membership_uses_future_targets", True),
                     ("target_proposal", "missing_labels_imputed", True),
                     ("validation_proposal", "random_row_split", True),
                     ("validation_proposal", "preprocessing_fit_on_inner_training_only", False),
                     ("comparison", "same_prediction_cohort", False))
        for section, field, value in mutations:
            with self.subTest(field=field):
                changed = deepcopy(self.plan)
                changed[section][field] = value
                with self.assertRaises(ModelDesignError):
                    validate_model_plan(changed)

    def test_universe_chronology_and_context_cannot_silently_change(self):
        mutations = (("universe", "primary", "sp500"),
                     ("universe", "membership_basis", "latest_survivors"),
                     ("chronology", "mode", "expanding"),
                     ("chronology", "last_test_year", 2024),
                     ("chronology", "train_years", True),
                     ("panels", "primary", "available_cohort_90pct"),
                     ("panels", "paper15_changed", True))
        for section, field, value in mutations:
            with self.subTest(field=field):
                changed = deepcopy(self.plan)
                changed[section][field] = value
                with self.assertRaises(ModelDesignError):
                    outcome_schedule(changed)

    def test_incomplete_mismatched_or_duplicate_arms_are_rejected(self):
        for variant in (self.plan["experiment_matrix"][:-1],
                        self.plan["experiment_matrix"] + [self.plan["experiment_matrix"][0]]):
            changed = deepcopy(self.plan)
            changed["experiment_matrix"] = variant
            with self.assertRaises(ModelDesignError):
                experiment_matrix(changed)
        changed = deepcopy(self.plan)
        changed["experiment_matrix"][2]["candidate_base_predictors"] = 7150
        with self.assertRaises(ModelDesignError):
            experiment_matrix(changed)

    def test_readiness_reports_open_labels_hyperparameters_and_no_authority(self):
        report = readiness_report(self.plan)
        for name in ("execution_authorized", "execution_ready", "labels_built", "model_fitted",
                     "preprocessing_fitted", "market_data_read", "paper_exact_replication"):
            self.assertIs(report[name], False)
        self.assertNotIn("label_rank_and_ties", report["unresolved_fields"])
        self.assertIn("XGBoost_boosting_round_cap", report["unresolved_fields"])
        self.assertIn("XGBoost_seed_and_variation_scheme", report["unresolved_fields"])
        self.assertIn("random_forest_package_version", report["unresolved_fields"])
        self.assertEqual(report["pilot"]["status"], "PROPOSED_NOT_RUN")
        self.assertFalse(report["pilot"]["paper100fit_ensemble_equivalent"])
        self.assertIn("missing indicators", report["input_note"])

    def test_paper_facts_and_v2_adaptations_remain_distinct(self):
        reference = self.plan["paper_reference"]
        self.assertEqual(reference["paper_predictors"], 16)
        self.assertEqual(reference["paper_annual_probability_average_fits"], 100)
        self.assertIn("expanding", reference["paper_classifier_history"])
        self.assertEqual(self.plan["chronology"]["mode"], "rolling")
        self.assertFalse(reference["v2_primary_is_exact_replication"])
        self.assertEqual(self.plan["execution"]["pilot_proposal"]["fits_per_arm"], 1)
        self.assertEqual(self.plan["estimator_proposals"]["random_forest"]["annual_models"],
                         "one_forest_per_annual_cutoff")

    def test_draft_validation_returns_detached_state(self):
        first = validate_model_plan(self.plan)
        first["panels"]["primary"] = "changed"
        self.assertEqual(self.plan["panels"]["primary"], "available_cohort_80pct")
        matrix = experiment_matrix(self.plan)
        matrix[0]["arm_id"] = "changed"
        self.assertEqual(experiment_matrix(self.plan)[0]["arm_id"], "xgb_paper15")

    def test_load_from_root_does_not_modify_config_or_create_outputs(self):
        before = sha256(DRAFT.read_bytes()).hexdigest()
        before_paths = set(ROOT.iterdir())
        self.assertEqual(load_model_plan(ROOT), self.plan)
        self.assertEqual(sha256(DRAFT.read_bytes()).hexdigest(), before)
        self.assertEqual(set(ROOT.iterdir()), before_paths)

    def test_malformed_missing_or_nonfinite_draft_fails_with_design_error(self):
        for value in ([], None, {}, {"schema_version": 2}):
            with self.subTest(value=value):
                with self.assertRaises(ModelDesignError):
                    validate_model_plan(value)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "draft.json"
            path.write_text("{malformed", encoding="utf-8")
            with self.assertRaises(ModelDesignError):
                load_model_plan(path)
            with self.assertRaises(ModelDesignError):
                load_model_plan(Path(temporary) / "absent.json")
        changed = deepcopy(self.plan)
        changed["extra_nonfinite"] = float("nan")
        with self.assertRaises(ModelDesignError):
            validate_model_plan(changed)


if __name__ == "__main__":
    unittest.main()
