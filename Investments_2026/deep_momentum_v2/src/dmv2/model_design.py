"""Calendar and experiment metadata for an unexecuted model comparison.

This module reads only the explicitly supplied draft JSON. It constructs no
labels, reads no market data and imports no estimator. Calendar outcome bounds
are necessary timing constraints, not proof of source availability or a trading
execution timestamp. Existing pinned panel configurations remain unchanged.
"""
from __future__ import annotations

from calendar import monthrange
from copy import deepcopy
from datetime import date
import json
from pathlib import Path

from .chronology import build_annual_folds


class ModelDesignError(ValueError):
    """The draft violates a declared comparison or no-execution boundary."""


_PLAN_NAME = "model_comparison.proposed.json"
_ARMS = (
    ("xgb_paper15", "xgboost_multiclass", "paper15", 15),
    ("rf_paper15", "random_forest_multiclass", "paper15", 15),
    ("xgb_extended", "xgboost_multiclass", "extended715_candidates", 715),
    ("rf_extended", "random_forest_multiclass", "extended715_candidates", 715),
)


def _get(plan, dotted):
    value = plan
    for name in dotted.split("."):
        if not isinstance(value, dict) or name not in value:
            raise ModelDesignError(f"Missing draft field: {dotted}")
        value = value[name]
    return value


def _require(plan, dotted, expected):
    value = _get(plan, dotted)
    # bool is an int subclass: do not accept 1 as authorization or an integer bound.
    try:
        exact = (type(value) is type(expected)
                 and json.dumps(value, allow_nan=False) == json.dumps(expected, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ModelDesignError(f"{dotted} must contain finite JSON metadata") from error
    if not exact:
        raise ModelDesignError(f"{dotted} must remain {expected!r} in this draft")


def validate_model_plan(plan):
    """Validate the locked comparison boundaries; return a detached JSON copy.

    The user-selected universe is binding. Estimator settings, validation and
    target details are proposals. Changing execution flags cannot turn this
    metadata file into execution authority, even after open decisions are filled.
    """
    if not isinstance(plan, dict):
        raise ModelDesignError("Model comparison draft must be a JSON object")
    requirements = {
        "schema_version": 1,
        "status": "PROPOSED_NOT_EXECUTION_AUTHORITY",
        "universe.primary": "combined",
        "universe.selection_status": "USER_SELECTED",
        "universe.membership_basis": "historical_signal_date",
        "universe.identity_policy": "provisional_unique_resolved_link",
        "universe.identity_status": "PROVISIONAL_IDENTITY_UNVERIFIED",
        "panels.primary": "available_cohort_80pct",
        "panels.sensitivity": "available_cohort_90pct",
        "panels.paper15_policy": "signal_qualified_common_momentum_v1",
        "panels.paper15_changed": False,
        "panels.daily_inputs_and_backtest_tables_separate": True,
        "chronology.history_start": "2010-01-01",
        "chronology.mode": "rolling",
        "chronology.train_years": 10,
        "chronology.test_years": 1,
        "chronology.last_test_year": 2023,
        "chronology.prediction_frequency": "monthly",
        "chronology.refit_frequency": "annual",
        "chronology.label_cutoff": "preceding_completed_outcome_year_end",
        "chronology.protected_financial_start": "2024-01-01",
        "chronology.knowledge_time_certified": False,
        "chronology.calendar_dates_define_execution": False,
        "target_proposal.kind": "next_month_cross_sectional_return_decile",
        "target_proposal.number_of_classes": 10,
        "target_proposal.class_order": "1_highest_return_10_lowest_return",
        "target_proposal.cohort_origin": "historical_signal_membership",
        "target_proposal.missing_labels_imputed": False,
        "target_proposal.prediction_membership_uses_future_targets": False,
        "target_proposal.retain_original_membership_ledger": True,
        "probabilities_and_score.aligned_class_ids": list(range(1, 11)),
        "probabilities_and_score.persist_estimator_class_mapping": True,
        "probabilities_and_score.primary_score": "RET",
        "probabilities_and_score.payoff_window_completed_months": 120,
        "probabilities_and_score.payoff_update_frequency": "monthly",
        "probabilities_and_score.monthly_class_mean": "equal_weight_observed_realized_class_returns",
        "probabilities_and_score.class_payoff_time_weighting": "equal_month_weights",
        "probabilities_and_score.payoffs_use_true_realized_classes": True,
        "probabilities_and_score.payoff_cutoff": "preceding_completed_outcome_month",
        "probabilities_and_score.calendar_proposal_is_paper_exact_claim": False,
        "preprocessing.policy": "train_partition_median_missing_indicator_v1",
        "preprocessing.baseline_all15_required_before_extra_inputs": True,
        "preprocessing.fit_scope": "inner_training_partition_then_frozen_outer_training_refit",
        "preprocessing.impute_targets": False,
        "preprocessing.drop_all_missing_in_fit_partition": True,
        "preprocessing.drop_constant_in_fit_partition": True,
        "preprocessing.missing_indicators": "one_per_retained_predictor_frozen_schema",
        "preprocessing.extended_predictor_count_is_guaranteed": False,
        "validation_proposal.mode": "last_completed_outcome_months_chronological",
        "validation_proposal.validation_months": 24,
        "validation_proposal.development_months": 96,
        "validation_proposal.all_stocks_of_same_outcome_month_stay_together": True,
        "validation_proposal.random_row_split": False,
        "validation_proposal.oob_is_outer_forecast_test": False,
        "validation_proposal.preprocessing_fit_on_inner_training_only": True,
        "validation_proposal.parameters_frozen_before_outer_test": True,
        "validation_proposal.final_refit": "full_preceding_120_completed_outcome_months_after_parameters_frozen",
        "paper_reference.paper_predictors": 16,
        "paper_reference.paper_size_decile_omitted_in_v2": True,
        "paper_reference.paper_classifier_history": "expanding_after_initial_ten_years",
        "paper_reference.paper_inner_validation": "random_historical_row_80_20",
        "paper_reference.paper_annual_probability_average_fits": 100,
        "paper_reference.paper_fit_variation_and_exact_defaults_unspecified": True,
        "paper_reference.v2_primary_is_exact_replication": False,
        "estimator_proposals.random_forest.status": "UNTUNED_PROPOSAL_NOT_AUTHOR_SETTING",
        "estimator_proposals.random_forest.annual_models": "one_forest_per_annual_cutoff",
        "estimator_proposals.random_forest.trees_are_not_500_independent_complete_forests": True,
        "estimator_proposals.xgboost.status": "OPEN_NO_INVENTED_AUTHOR_PARAMETERS",
        "estimator_proposals.xgboost.objective": "multi:softprob",
        "estimator_proposals.xgboost.num_class": 10,
        "estimator_proposals.xgboost.confirmatory_fits_per_annual_cutoff_proposal": 100,
        "estimator_proposals.xgboost.ensemble_variation_requires_specification": True,
        "execution.metadata_only": True,
    }
    for dotted, expected in requirements.items():
        _require(plan, dotted, expected)
    for name in ("same_outer_windows", "same_prediction_cohort", "same_label_definition",
                 "same_return_basis", "same_class_payoff_estimator", "same_preprocessing_policy"):
        _require(plan, f"comparison.{name}", True)
    for name in ("full_history_coverage_selects_features", "sensitivity_parameters_selected_on_test"):
        _require(plan, f"comparison.{name}", False)
    for name in ("label_build_enabled", "preprocessing_fit_enabled", "model_fit_enabled",
                 "parameter_search_enabled", "backtest_enabled", "cloud_launch_enabled",
                 "dependency_install_enabled", "automatic_execution"):
        _require(plan, f"execution.{name}", False)
    for name in ("scaling", "clipping", "winsorization"):
        _require(plan, f"preprocessing.{name}", None)
    arms = _get(plan, "experiment_matrix")
    expected_arms = [dict(zip(("arm_id", "estimator", "feature_set", "candidate_base_predictors"), arm))
                     for arm in _ARMS]
    if arms != expected_arms:
        raise ModelDesignError("experiment_matrix must retain the paired 2x2 estimator/input comparison")
    # Fields listed as open are surfaced, not silently filled with library defaults.
    remaining = _get(plan, "execution.remaining_decisions")
    if (not isinstance(remaining, list) or not remaining
            or any(not isinstance(item, str) or not item.strip() for item in remaining)
            or len(remaining) != len(set(remaining))):
        raise ModelDesignError("execution.remaining_decisions must explicitly list unresolved design gates")
    _require(plan, "execution.pilot_proposal.status", "PROPOSED_NOT_RUN")
    _require(plan, "execution.pilot_proposal.first_outer_test_year", 2020)
    _require(plan, "execution.pilot_proposal.scope", "inner_development_validation_only")
    _require(plan, "execution.pilot_proposal.outer_test_metrics_inspected", False)
    _require(plan, "execution.pilot_proposal.inner_training_outcome_start", "2010-01-01")
    _require(plan, "execution.pilot_proposal.inner_training_outcome_end", "2017-12-31")
    _require(plan, "execution.pilot_proposal.inner_validation_outcome_start", "2018-01-01")
    _require(plan, "execution.pilot_proposal.inner_validation_outcome_end", "2019-12-31")
    _require(plan, "execution.pilot_proposal.outer_refit_enabled", False)
    _require(plan, "execution.pilot_proposal.RET_evaluation_enabled", False)
    _require(plan, "execution.pilot_proposal.additional_pre2010_outcome_reads_enabled", False)
    _require(plan, "execution.pilot_proposal.metrics_scope", "classification_probabilities_only")
    _require(plan, "execution.pilot_proposal.fits_per_arm", 1)
    _require(plan, "execution.pilot_proposal.paper100fit_ensemble_equivalent", False)
    for dotted in (
        "target_proposal.rank_tie_rule", "target_proposal.rank_population_with_missing_targets",
        "target_proposal.economic_return_and_terminal_event_policy",
        "probabilities_and_score.absent_training_class_policy",
        "probabilities_and_score.payoff_missing_month_rule",
        "validation_proposal.selection_metric", "validation_proposal.candidate_search_budget",
        "estimator_proposals.random_forest.package_version",
        "estimator_proposals.xgboost.package_version", "estimator_proposals.xgboost.parameter_snapshot",
        "estimator_proposals.xgboost.boosting_round_cap", "estimator_proposals.xgboost.early_stopping_patience",
        "estimator_proposals.xgboost.seed_and_variation_scheme",
        "execution.pilot_proposal.resource_and_trial_budget",
    ):
        _get(plan, dotted)
    try:
        return json.loads(json.dumps(plan, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ModelDesignError("Draft must contain finite JSON metadata only") from error


def load_model_plan(path_or_root):
    """Read an explicit draft file, or ``root/configs/model_comparison.proposed.json``."""
    path = Path(path_or_root)
    if path.is_dir():
        path = path / "configs" / _PLAN_NAME
    try:
        plan = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ModelDesignError(f"Cannot read model comparison draft: {path}") from error
    return validate_model_plan(plan)


def experiment_matrix(plan):
    """Describe candidate inputs; retained predictors and indicators are fit-specific."""
    plan = validate_model_plan(plan)
    rows = []
    for arm in plan["experiment_matrix"]:
        rows.append({**deepcopy(arm), "universe": "combined",
                     "primary_context": plan["panels"]["primary"],
                     "sensitivity_context": plan["panels"]["sensitivity"],
                     "output": "aligned_10_class_probabilities_then_RET",
                     "retained_predictors": "determined_in_training_partition",
                     "transformed_columns": "retained_predictors_plus_frozen_missing_indicators",
                     "status": "PROPOSED_NOT_RUN"})
    return rows


def _month_number(year, month):
    return year * 12 + month - 1


def _month(number):
    year, zero_month = divmod(number, 12)
    return f"{year:04d}-{zero_month + 1:02d}"


def _month_end(number):
    year, zero_month = divmod(number, 12)
    return date(year, zero_month + 1, monthrange(year, zero_month + 1)[1]).isoformat()


def _schedule(plan):
    chronology = plan["chronology"]
    folds = build_annual_folds(chronology["history_start"], chronology["last_test_year"],
                               chronology["train_years"], mode=chronology["mode"])
    rows = []
    validation_months = plan["validation_proposal"]["validation_months"]
    for fold in folds:
        year = int(fold.test_start[:4])
        training_end = _month_number(year - 1, 12)
        training_start = training_end - 119
        validation_start = training_end - validation_months + 1
        for month in range(1, 13):
            outcome = _month_number(year, month)
            # Model parameters remain annual; RET updates from newly COMPLETED
            # outcomes using a fixed ex-ante estimator, never current outcome data.
            rows.append({
                "fold_id": fold.fold_id,
                "prediction_outcome_month": _month(outcome),
                "feature_month": _month(outcome - 1),
                "train_outcome_start": fold.train_start,
                "train_outcome_end": fold.train_end,
                "train_first_feature_month": _month(training_start - 1),
                "train_last_feature_month": _month(training_end - 1),
                "classifier_label_cutoff": fold.label_cutoff,
                "development_outcome_start": fold.train_start,
                "development_outcome_end": _month_end(validation_start - 1),
                "development_label_cutoff": _month_end(validation_start - 1),
                "validation_outcome_start": _month(validation_start) + "-01",
                "validation_outcome_end": fold.train_end,
                "RET_outcome_start_month": _month(outcome - 120),
                "RET_outcome_end_month": _month(outcome - 1),
                "RET_label_cutoff": _month_end(outcome - 1),
                "calendar_timing_only": True,
                "status": "PROPOSED_NOT_RUN",
            })
    return rows


def outcome_schedule(plan):
    """Return 48 monthly calendar specifications, without reading or selecting rows.

    For January 2020 the annual classifier admits completed outcomes 2010-2019;
    its last training feature month is November 2019. December 2019 features
    predict the still-unknown January 2020 outcome and cannot train that fit.
    RET then rolls monthly, so February can use a completed January outcome if
    its economic label and source availability satisfy the unresolved rules.
    """
    plan = validate_model_plan(plan)
    return _schedule(plan)


def validate_outcome_schedule(rows, plan):
    """Reject edited schedules that move labels across a declared timing boundary.

    Useful when a notebook exports the design table. This verifies metadata
    integrity only; future model workers must independently check actual row
    outcome and availability timestamps before any fitted operation.
    """
    plan = validate_model_plan(plan)
    expected = _schedule(plan)
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise ModelDesignError("Monthly schedule must retain all 48 pre-2024 forecast months")
    for row, required in zip(rows, expected):
        if not isinstance(row, dict) or row != required:
            raise ModelDesignError("Monthly schedule changed a completed-outcome cutoff or calendar partition")
        outcome_start = date.fromisoformat(row["prediction_outcome_month"] + "-01")
        if not (row["classifier_label_cutoff"] < outcome_start.isoformat()
                and row["RET_label_cutoff"] < outcome_start.isoformat()
                and row["development_outcome_end"] < row["validation_outcome_start"]
                <= row["validation_outcome_end"] == row["classifier_label_cutoff"]
                and row["prediction_outcome_month"] < "2024-01"):
            raise ModelDesignError("Training, validation and forecast outcome boundaries overlap")
    return True


def readiness_report(plan):
    """Surface open decisions. A validated sketch always has no execution authority."""
    plan = validate_model_plan(plan)
    score = plan["probabilities_and_score"]
    target = plan["target_proposal"]
    xgb = plan["estimator_proposals"]["xgboost"]
    validation = plan["validation_proposal"]
    open_fields = {
        "label_rank_and_ties": target["rank_tie_rule"],
        "missing_target_rank_population": target["rank_population_with_missing_targets"],
        "economic_return_terminal_events_and_label_coverage": target["economic_return_and_terminal_event_policy"],
        "absent_training_class_handling": score["absent_training_class_policy"],
        "RET_missing_month_minimum_history_rule": score["payoff_missing_month_rule"],
        "validation_metric": validation["selection_metric"],
        "candidate_search_budget": validation["candidate_search_budget"],
        "random_forest_package_version": plan["estimator_proposals"]["random_forest"]["package_version"],
        "XGBoost_package_version": xgb["package_version"],
        "XGBoost_parameter_snapshot": xgb["parameter_snapshot"],
        "XGBoost_boosting_round_cap": xgb["boosting_round_cap"],
        "XGBoost_early_stopping_patience": xgb["early_stopping_patience"],
        "XGBoost_seed_and_variation_scheme": xgb["seed_and_variation_scheme"],
        "pilot_resource_and_trial_budget": plan["execution"]["pilot_proposal"]["resource_and_trial_budget"],
    }
    return {
        "status": plan["status"],
        "primary_universe": plan["universe"]["primary"],
        "universe_selection": plan["universe"]["selection_status"],
        "proposed_arms": len(plan["experiment_matrix"]),
        "nominal_annual_folds": 4,
        "nominal_monthly_predictions_per_arm": 48,
        "execution_authorized": False,
        "execution_ready": False,
        "labels_built": False,
        "model_fitted": False,
        "preprocessing_fitted": False,
        "market_data_read": False,
        "paper_exact_replication": False,
        "identity_status": plan["universe"]["identity_status"],
        "unresolved_fields": [name for name, value in open_fields.items() if value is None],
        "remaining_decisions": deepcopy(plan["execution"]["remaining_decisions"]),
        "pilot": deepcopy(plan["execution"]["pilot_proposal"]),
        "timing_note": "Completed-outcome calendars do not certify availability or define market execution.",
        "input_note": "15 and 715 are candidate base definitions; partition-only drop rules and missing indicators determine the fitted schema.",
    }
