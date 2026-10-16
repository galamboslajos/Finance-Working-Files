"""Frozen-formula probability diagnostics for2020--2023 outer predictions; no fitting or portfolio logic.

Predictions for missing labels remain present. Log loss alone uses the declared
1e-15 evaluation floor; stored probabilities and all other metrics are unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

import numpy as np
import pandas as pd

from .monthly import _step_month
from .ret import CLASS_IDS, align_class_probabilities
from .universe import DataContractError, iso_date


LOG_LOSS_FLOOR = 1e-15
CALIBRATION_BINS = 10
VALIDATION_FIRST_MONTH = "2020-01"
VALIDATION_LAST_MONTH = "2023-12"
KEY_COLUMNS = ("source_security_id", "signal_date")
METADATA_COLUMNS = (*KEY_COLUMNS, "outcome_month", "outcome_end")


@dataclass(frozen=True)
class ProbabilityEvaluation:
    summary: dict
    tables: dict[str, pd.DataFrame]


def _metadata(frame, first_month, last_month):
    if not (isinstance(first_month,str) and isinstance(last_month,str)
            and VALIDATION_FIRST_MONTH <= first_month <= last_month <= VALIDATION_LAST_MONTH):
        raise DataContractError("Walk-forward metrics require explicit2020--2023 outcome bounds")
    iso_date(first_month+"-01","first outcome month")
    iso_date(last_month+"-01","last outcome month")
    if not isinstance(frame, pd.DataFrame) or not set(METADATA_COLUMNS) <= set(frame):
        raise DataContractError(f"Probability evaluation requires row metadata {METADATA_COLUMNS}")
    if not frame.index.is_unique or frame.columns.duplicated().any():
        raise DataContractError("Probability metadata requires unique row indices and columns")
    selected = frame.loc[:, list(METADATA_COLUMNS)].copy()
    if selected.duplicated(list(KEY_COLUMNS)).any():
        raise DataContractError("Duplicate prediction source-security/signal keys")
    # Inspect safe dates before touching any supplied class/probability values.
    for sid, signal, month, end in selected.itertuples(index=False, name=None):
        if not isinstance(sid, str) or not sid.strip():
            raise DataContractError("Prediction source_security_id must be a nonempty string")
        iso_date(signal, "prediction signal_date")
        iso_date(end, "prediction outcome_end")
        if (not isinstance(month, str) or len(month) != 7
                or not first_month <= month <= last_month):
            raise DataContractError("Walk-forward probability evaluation exceeds its explicit2020--2023 outcome window")
        iso_date(month + "-01", "prediction outcome_month")
        if signal[:7] != _step_month(month, -1) or signal >= month + "-01" or end[:7] != month:
            raise DataContractError("Validation outcome must be exactly the month after its signal")
    return selected


def _labels(labels, n):
    try:
        values = np.asarray(labels, dtype=object)
    except (TypeError, ValueError) as error:
        raise DataContractError("Validation labels must be an aligned nullable class vector") from error
    if values.ndim != 1 or len(values) != n:
        raise DataContractError("Validation labels must have one entry per prediction row")
    result = np.full(n, np.nan)
    for i, value in enumerate(values):
        if value is None or value is pd.NA or value is pd.NaT:
            continue
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise DataContractError("Validation labels must be canonical integer classes 1 through 10 or missing")
        if isinstance(value, (float, np.floating)) and math.isnan(value):
            continue
        if not math.isfinite(value) or int(value) != value or int(value) not in CLASS_IDS:
            raise DataContractError("Validation labels must be canonical integer classes 1 through 10 or missing")
        result[i] = int(value)
    return result


def _calibration(probabilities, labels, predicted, *, top_label):
    records = []
    classes = (None,) if top_label else CLASS_IDS
    for class_id in classes:
        confidence = probabilities.max(axis=1) if top_label else probabilities[:, class_id - 1]
        observed = labels == predicted if top_label else labels == class_id
        bins = np.minimum((confidence * CALIBRATION_BINS).astype(int), CALIBRATION_BINS - 1)
        for bin_id in range(CALIBRATION_BINS):
            chosen = bins == bin_id
            count = int(chosen.sum())
            mean = float(confidence[chosen].mean()) if count else None
            frequency = float(observed[chosen].mean()) if count else None
            records.append({
                **({} if top_label else {"class_id": class_id}),
                "bin_id": bin_id, "lower_bound": bin_id / CALIBRATION_BINS,
                "upper_bound": (bin_id + 1) / CALIBRATION_BINS,
                "upper_inclusive": bin_id == CALIBRATION_BINS - 1,
                "labelled_count": count, "mean_probability": mean,
                "observed_frequency": frequency,
                "absolute_gap": abs(mean - frequency) if count else None,
                "status": "available" if count else "empty_bin",
            })
    return pd.DataFrame.from_records(records)


def _extremes(rows):
    records = []
    periods = [("all", rows)] + [(str(month), group) for month, group in rows.groupby("outcome_month", sort=True)]
    for period, group in periods:
        for class_id in (1, 10):
            chosen = group.loc[group["predicted_class"] == class_id]
            labelled = chosen.loc[chosen["label_available"]]
            support, observed = len(chosen), len(labelled)
            hits = int((labelled["true_class"] == class_id).sum())
            status = ("no_hard_mode_predictions" if not support else
                      "no_observed_labels_in_hard_mode" if not observed else "available")
            records.append({"period": period, "class_id": class_id,
                            "prediction_rows": len(group), "hard_mode_prediction_count": support,
                            "labelled_hard_mode_count": observed,
                            "missing_label_hard_mode_count": support - observed,
                            "true_positive_count": hits, "precision": hits / observed if observed else None,
                            "prediction_share": support / len(group) if len(group) else None,
                            "status": status})
    return pd.DataFrame.from_records(records)


def _mode_classes(rows):
    """Separate class precision/recall, with missing-outcome support explicit."""
    records = []
    periods = [("all", rows)] + [(str(month), group) for month, group in rows.groupby("outcome_month", sort=True)]
    for period, group in periods:
        for class_id in CLASS_IDS:
            predicted = group.predicted_class.eq(class_id)
            observed_predicted = predicted & group.label_available
            actual = group.label_available & group.true_class.eq(class_id).fillna(False)
            hits = int((observed_predicted & actual).sum())
            predicted_n, observed_n, actual_n = int(predicted.sum()), int(observed_predicted.sum()), int(actual.sum())
            records.append({"period": period, "class_id": class_id,
                "prediction_rows": len(group), "predicted_support": predicted_n,
                "labelled_predicted_support": observed_n,
                "missing_label_predicted_support": predicted_n - observed_n,
                "actual_support": actual_n, "true_positive_count": hits,
                "precision": hits / observed_n if observed_n else None,
                "recall": hits / actual_n if actual_n else None,
                "status": "available" if observed_n and actual_n else "missing_predicted_or_actual_support"})
    return pd.DataFrame(records)


def _confusion(rows):
    records = []
    periods = [("all", rows)] + [(str(month), group) for month, group in rows.groupby("outcome_month", sort=True)]
    for period, group in periods:
        observed = group.loc[group.label_available]
        counts = observed.groupby(["true_class", "predicted_class"], observed=True).size().to_dict()
        for actual in CLASS_IDS:
            for predicted in CLASS_IDS:
                records.append({"period": period, "actual_class": actual,
                    "predicted_class": predicted, "count": int(counts.get((actual, predicted), 0))})
    return pd.DataFrame(records)


def evaluate_probabilities(probabilities, labels, *, class_ids, row_metadata, first_month, last_month):
    """Score one arm/reference within explicit completed outer outcome months.

    Metadata and optional pandas indices are checked before financial values.
    Canonical class 1 is best, class 10 worst. Hard-mode ties deterministically
    select the lowest canonical class ID; no jitter is added. Missing outcomes
    are retained but cannot contribute to supervised scores or calibration.
    Primary metrics average monthly means equally, with pooled metrics reported
    separately. Brier is the sum of ten squared probability errors per row.
    """
    metadata = _metadata(row_metadata, first_month, last_month)
    for name, value in (("probabilities", probabilities), ("labels", labels)):
        if isinstance(value, (pd.DataFrame, pd.Series)) and not value.index.equals(metadata.index):
            raise DataContractError(f"{name} index differs from prediction metadata")
    aligned = align_class_probabilities(probabilities, class_ids=class_ids)
    if len(aligned) != len(metadata):
        raise DataContractError("Probability rows differ from prediction metadata")
    targets = _labels(labels, len(metadata))
    available = np.isfinite(targets)
    predicted = np.argmax(aligned, axis=1) + 1
    true_prob = np.full(len(metadata), np.nan)
    losses = np.full(len(metadata), np.nan)
    briers = np.full(len(metadata), np.nan)
    correct = np.full(len(metadata), np.nan)
    selected_targets = targets[available].astype(int)
    selected_probabilities = aligned[available]
    if available.any():
        true_prob[available] = selected_probabilities[np.arange(available.sum()), selected_targets - 1]
        losses[available] = -np.log(np.maximum(true_prob[available], LOG_LOSS_FLOOR))
        errors = selected_probabilities.copy()
        errors[np.arange(available.sum()), selected_targets - 1] -= 1
        briers[available] = np.square(errors).sum(axis=1)
        correct[available] = predicted[available] == selected_targets
    rows = metadata.copy()
    rows["true_class"] = pd.array(targets, dtype="Int64")
    rows["label_available"] = available
    rows["predicted_class"] = predicted.astype(np.int8)
    rows["top_probability"] = aligned.max(axis=1)
    rows["true_class_probability"] = true_prob
    rows["log_loss"] = losses
    rows["brier_sum10"] = briers
    rows["correct"] = correct
    rows["log_loss_probability_floored"] = available & (true_prob < LOG_LOSS_FLOOR)
    monthly_records = []
    for month, group in rows.groupby("outcome_month", sort=True):
        observed = int(group["label_available"].sum())
        monthly_records.append({
            "outcome_month": month, "prediction_rows": len(group), "labelled_rows": observed,
            "missing_label_rows": len(group) - observed, "label_coverage": observed / len(group),
            "log_loss": float(group["log_loss"].mean()) if observed else None,
            "brier_sum10": float(group["brier_sum10"].mean()) if observed else None,
            "accuracy": float(group["correct"].mean()) if observed else None,
            "zero_true_class_probability_rows": int((group["true_class_probability"] == 0).sum()),
            "floored_true_class_probability_rows": int(group["log_loss_probability_floored"].sum()),
            "status": "available" if observed else "no_observed_labels",
        })
    monthly = pd.DataFrame.from_records(monthly_records, columns=[
        "outcome_month", "prediction_rows", "labelled_rows", "missing_label_rows", "label_coverage",
        "log_loss", "brier_sum10", "accuracy", "zero_true_class_probability_rows",
        "floored_true_class_probability_rows", "status"])
    top = _calibration(selected_probabilities, selected_targets, predicted[available], top_label=True)
    classwise = _calibration(selected_probabilities, selected_targets, predicted[available], top_label=False)
    def scalar_mean(frame, name):
        return float(frame[name].mean()) if len(frame) and frame[name].notna().any() else None
    labelled_n = int(available.sum())
    summary = {
        "prediction_rows": len(metadata), "labelled_rows": labelled_n,
        "missing_label_rows": len(metadata) - labelled_n,
        "label_coverage": labelled_n / len(metadata) if len(metadata) else None,
        "outcome_months": len(monthly), "observed_label_months": int((monthly["labelled_rows"] > 0).sum()),
        "first_outcome_month": monthly.iloc[0]["outcome_month"] if len(monthly) else None,
        "last_outcome_month": monthly.iloc[-1]["outcome_month"] if len(monthly) else None,
        "mean_monthly_log_loss": scalar_mean(monthly, "log_loss"),
        "mean_monthly_brier_sum10": scalar_mean(monthly, "brier_sum10"),
        "mean_monthly_accuracy": scalar_mean(monthly, "accuracy"),
        "pooled_log_loss": scalar_mean(rows, "log_loss"),
        "pooled_brier_sum10": scalar_mean(rows, "brier_sum10"),
        "pooled_accuracy": scalar_mean(rows, "correct"),
        "pooled_top_label_ece": float((top["absolute_gap"].fillna(0) * top["labelled_count"]).sum() / labelled_n) if labelled_n else None,
        "pooled_classwise_ece_mean": float((classwise["absolute_gap"].fillna(0) * classwise["labelled_count"]).sum() / (10 * labelled_n)) if labelled_n else None,
        "zero_true_class_probability_rows": int(np.sum(true_prob == 0)),
        "floored_true_class_probability_rows": int(np.sum(available & (true_prob < LOG_LOSS_FLOOR))),
        "log_loss_evaluation_floor": LOG_LOSS_FLOOR,
        "log_loss_floor_changes_stored_probabilities": False,
        "probability_clipping_or_renormalization": False,
        "brier_definition": "sum_of_ten_squared_probability_errors",
        "primary_aggregation": "equal_weight_mean_of_observed_monthly_means",
        "hard_mode_probability_tie_rule": "lowest_canonical_class_id",
        "calibration_fixed_equal_width_bins": CALIBRATION_BINS,
        "label_imputation_performed": False, "RET_or_portfolio_evaluation_performed": False,
    }
    return ProbabilityEvaluation(summary, {"row_scores": rows, "monthly": monthly,
                                           "classwise_calibration": classwise,
                                           "toplabel_calibration": top,
                                           "extreme_precision": _extremes(rows),
                                           "mode_class_performance": _mode_classes(rows),
                                           "confusion": _confusion(rows)})
