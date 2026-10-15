"""Causal, shared realized-class payoffs and probability-to-RET arithmetic.

Only supplied target tables are consumed. Each class payoff is the equal-weight
mean of 120 *monthly* realized-class means, never a pooled stock-row mean or a
mean of predicted buckets. Every class must be observed in every calendar month;
otherwise the whole payoff vector is withheld. Nothing here fits an estimator,
changes the prediction cohort, certifies economic returns, or executes trades.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from .catalog import canonical_sha256
from .monthly import _month, _step_month
from .universe import DataContractError, iso_date


RET_POLICY = "realized_class_equal_month_120_v1"
CLASS_IDS = tuple(range(1, 11))
PAYOFF_MONTHS = 120
PROBABILITY_SUM_TOLERANCE = 1e-8
RET_POLICY_DEFINITION = {
    "policy_id": RET_POLICY, "class_ids": list(CLASS_IDS),
    "completed_calendar_months": PAYOFF_MONTHS,
    "window_end": "signal_calendar_month",
    "monthly_class_mean": "equal_weight_observed_realized_class_returns",
    "time_weighting": "equal_calendar_month_weights",
    "incomplete_month_class_grid": "withhold_entire_payoff_vector",
    "probability_sum_tolerance": PROBABILITY_SUM_TOLERANCE,
    "probability_renormalization": False, "absent_class_zero_padding": False,
}

PAYOFF_SCHEMA = pa.schema([
    ("class_id", pa.int8()), ("payoff", pa.float64()), ("status", pa.string()),
])
MONTHLY_PAYOFF_SCHEMA = pa.schema([
    ("outcome_month", pa.string()), ("outcome_end", pa.string()),
    ("class_id", pa.int8()), ("original_member_count", pa.int64()),
    ("prediction_eligible_count", pa.int64()), ("member_count", pa.int64()),
    ("class_mean_return", pa.float64()), ("status", pa.string()),
])


@dataclass(frozen=True)
class RetPayoffPanel:
    """Frozen tables and a checked provenance fingerprint; no model state."""

    table: pa.Table
    monthly: pa.Table
    provenance: dict
    fingerprint: str


def _cutoffs(signal_date, label_observed_through):
    signal = iso_date(signal_date, "signal_date")
    cutoff = iso_date(label_observed_through, "label_observed_through")
    if signal >= "2024-01-01" or cutoff >= "2024-01-01":
        raise DataContractError("Protected 2024+ RET dates cannot be requested")
    if cutoff > signal:
        raise DataContractError("RET label_observed_through cannot exceed signal_date")
    return signal, cutoff


def _fingerprint(table, monthly, provenance):
    return canonical_sha256({"table": table.to_pylist(), "monthly": monthly.to_pylist(),
                             "provenance": provenance})


def verify_ret_payoffs(payoffs):
    """Reject modified state before any probability score is produced."""
    if not isinstance(payoffs, RetPayoffPanel):
        raise DataContractError("payoffs must be a RetPayoffPanel")
    if payoffs.table.schema != PAYOFF_SCHEMA or payoffs.monthly.schema != MONTHLY_PAYOFF_SCHEMA:
        raise DataContractError("RET payoff schema changed")
    if payoffs.table.num_rows != 10 or payoffs.monthly.num_rows != 1200:
        raise DataContractError("RET requires ten classes and a complete 120 by 10 calendar grid")
    try:
        fingerprint = _fingerprint(payoffs.table, payoffs.monthly, payoffs.provenance)
    except (TypeError, ValueError, OverflowError) as error:
        raise DataContractError("RET payoff fingerprint contains invalid metadata or values") from error
    if fingerprint != payoffs.fingerprint:
        raise DataContractError("RET payoff fingerprint changed")
    if payoffs.provenance.get("ret_policy") != RET_POLICY:
        raise DataContractError("Unknown RET payoff policy")
    if payoffs.provenance.get("policy_sha256") != canonical_sha256(RET_POLICY_DEFINITION):
        raise DataContractError("RET payoff policy fingerprint differs from the fixed shared policy")
    signal, cutoff = _cutoffs(payoffs.provenance.get("signal_date"), payoffs.provenance.get("label_observed_through"))
    rows = payoffs.table.to_pylist()
    if [row["class_id"] for row in rows] != list(CLASS_IDS):
        raise DataContractError("RET payoff class order changed")
    available = [row["status"] == "available" for row in rows]
    if any(available) != all(available):
        raise DataContractError("RET cannot release a partially observed payoff vector")
    for row in rows:
        value = row["payoff"]
        if row["status"] == "available":
            if value is None or not math.isfinite(value):
                raise DataContractError("Available RET payoffs must be finite")
        elif row["status"] != "withheld_incomplete_120_month_class_grid" or value is not None:
            raise DataContractError("Withheld RET payoffs must be explicitly null")
    last = signal[:7]
    first = _step_month(last, -(PAYOFF_MONTHS - 1))
    monthly = payoffs.monthly.to_pylist()
    expected_grid = [(_step_month(first, i), class_id)
                     for i in range(PAYOFF_MONTHS) for class_id in CLASS_IDS]
    if [(row["outcome_month"], row["class_id"]) for row in monthly] != expected_grid:
        raise DataContractError("RET monthly grid changed its consecutive calendar months or class order")
    monthly_available = [row["status"] == "available" for row in monthly]
    if (payoffs.provenance.get("first_outcome_month") != first
            or payoffs.provenance.get("last_outcome_month") != last
            or type(payoffs.provenance.get("vector_available")) is not bool
            or payoffs.provenance["vector_available"] != all(monthly_available)
            or all(available) != all(monthly_available)):
        raise DataContractError("RET availability or trailing calendar window is inconsistent")
    for row in monthly:
        if row["status"] == "available":
            if (row["member_count"] < 1 or row["class_mean_return"] is None
                    or not math.isfinite(row["class_mean_return"])
                    or row["outcome_end"] is None or row["outcome_end"] > min(signal, cutoff)):
                raise DataContractError("Observed RET class means require completed finite nonempty source groups")
        elif row["class_mean_return"] is not None:
            raise DataContractError("Unavailable RET monthly class means must be null")
    if all(available):
        for row in rows:
            expected = math.fsum(month["class_mean_return"] / PAYOFF_MONTHS
                                 for month in monthly if month["class_id"] == row["class_id"])
            if row["payoff"] != expected:
                raise DataContractError("RET payoff differs from its equal-calendar-month class means")
    return True


def build_ret_payoffs(target_panel, *, signal_date, label_observed_through):
    """Build strictly trailing realized-class payoffs as of one monthly signal.

    The expected last month is always the signal's calendar month. Source month
    windows must explicitly establish completion by both requested dates. An
    early cutoff withholds that month; it never silently shifts to an older,
    shorter, or compressed window. Calendar/session assertions are source
    evidence, not certification of same-close operational availability.

    Metadata is validated before a completed, trailing month is selected. Only
    selected completed months have their financial values decoded and checked;
    later outcome returns/classes are never used to construct past payoffs.
    """
    from .targets import MonthlyTargetPanel, verify_target_panel

    if not isinstance(target_panel, MonthlyTargetPanel):
        raise DataContractError("target_panel must be a MonthlyTargetPanel")
    signal, cutoff = _cutoffs(signal_date, label_observed_through)
    last = _month(signal[:7], "signal month")
    first = _step_month(last, -(PAYOFF_MONTHS - 1))
    months = [_step_month(first, i) for i in range(PAYOFF_MONTHS)]
    # This verifier checks all timing metadata first, then fingerprints only
    # selected completed months. Its default whole-panel mode is not causal
    # when the supplied table also contains labels after this signal cutoff.
    source_cutoff = iso_date(target_panel.provenance["label_observed_through"],
                             "target panel label_observed_through")
    verify_target_panel(target_panel, through=min(cutoff, source_cutoff), start_month=first)
    windows = {row["outcome_month"]: row
               for row in target_panel.provenance["month_windows"]}
    safe_columns = ["outcome_month", "outcome_end", "prediction_eligible"]
    safe_rows = target_panel.table.select(safe_columns).to_pylist()
    counts = {}
    for row in safe_rows:
        month = row["outcome_month"]
        if first <= month <= last:
            original, eligible = counts.get(month, (0, 0))
            counts[month] = (original + 1, eligible + int(row["prediction_eligible"]))
    effective_cutoff = min(signal, cutoff, source_cutoff)
    monthly_rows, selected_pins = [], {}
    for month in months:
        window = windows.get(month)
        original, eligible = counts.get(month, (0, 0))
        end = None if window is None else window["outcome_end"]
        if window is None:
            status = "missing_month_completion_evidence"
        elif end > effective_cutoff:
            status = "month_not_observed_by_cutoff"
        elif original == 0:
            status = "missing_month"
        else:
            status = None
        groups = {class_id: [] for class_id in CLASS_IDS}
        if status is None:
            # Financial fields are accessed only after the month is selected by
            # safe metadata and explicitly established to be completed.
            selected = target_panel.table.filter(pc.equal(
                target_panel.table["outcome_month"], month))
            selected_pins[month] = target_panel.provenance["month_target_fingerprints"][month]
            for row in selected.select(["target_available", "target_class", "label_return"]).to_pylist():
                if row["target_available"]:
                    class_id, value = row["target_class"], row["label_return"]
                    if (isinstance(class_id, bool) or class_id not in CLASS_IDS
                            or not isinstance(value, Real) or isinstance(value, bool)
                            or not math.isfinite(value) or value < -1):
                        raise DataContractError("Invalid observed realized target class or return")
                    groups[class_id].append(float(value))
        for class_id in CLASS_IDS:
            values = groups[class_id]
            available = status is None and bool(values)
            monthly_rows.append({
                "outcome_month": month, "outcome_end": end, "class_id": class_id,
                "original_member_count": original, "prediction_eligible_count": eligible,
                "member_count": len(values),
                "class_mean_return": math.fsum(value / len(values) for value in values) if available else None,
                "status": "available" if available else (status or "empty_realized_class"),
            })
    all_available = all(row["status"] == "available" for row in monthly_rows)
    payoff_rows = []
    for class_id in CLASS_IDS:
        values = [row["class_mean_return"] for row in monthly_rows if row["class_id"] == class_id]
        payoff_rows.append({
            "class_id": class_id,
            "payoff": math.fsum(value / PAYOFF_MONTHS for value in values) if all_available else None,
            "status": "available" if all_available else "withheld_incomplete_120_month_class_grid",
        })
    table = pa.Table.from_pylist(payoff_rows, schema=PAYOFF_SCHEMA)
    monthly = pa.Table.from_pylist(monthly_rows, schema=MONTHLY_PAYOFF_SCHEMA)
    provenance = {
        "ret_policy": RET_POLICY, "policy_sha256": canonical_sha256(RET_POLICY_DEFINITION),
        "source_target_policy_sha256": target_panel.provenance["policy_sha256"],
        "signal_date": signal,
        "label_observed_through": cutoff, "first_outcome_month": first,
        "last_outcome_month": last, "expected_calendar_months": PAYOFF_MONTHS,
        "expected_classes": list(CLASS_IDS), "expected_month_class_cells": 1200,
        "available_month_class_cells": sum(row["status"] == "available" for row in monthly_rows),
        "vector_available": all_available,
        "monthly_class_mean": "equal_weight_observed_realized_class_returns",
        "payoff_time_weighting": "equal_calendar_month_weights",
        "missing_month_or_class_policy": "withhold_entire_payoff_vector",
        "uses_predicted_classes": False, "financial_rows_after_cutoff_consumed": False,
        "source_month_target_fingerprints": selected_pins,
        "source_completion_windows": [windows[month] for month in months if month in windows],
        "return_basis": "provisional_provider_adjusted_endpoint_ratio",
        "economic_return_certified": False, "knowledge_time_certified": False,
        "same_close_operational_availability_certified": False,
        "model_ready": False, "estimator_fit_performed": False,
    }
    result = RetPayoffPanel(table, monthly, provenance, _fingerprint(table, monthly, provenance))
    verify_ret_payoffs(result)
    return result


def align_class_probabilities(probabilities, *, class_ids):
    """Validate explicit estimator classes and return columns in order 1..10.

    No absent classes are zero-padded and no invalid probabilities are clipped
    or renormalized. Explicit class IDs describe the supplied column order.
    """
    try:
        ids = list(class_ids)
    except TypeError as error:
        raise DataContractError("class_ids must explicitly enumerate ten integer class IDs") from error
    if (len(ids) != 10 or any(isinstance(value, (bool, np.bool_))
                             or not isinstance(value, Integral) for value in ids)
            or len(set(ids)) != 10 or set(ids) != set(CLASS_IDS)):
        raise DataContractError("class_ids must contain each integer class ID 1 through 10 exactly once")
    try:
        objects = np.asarray(probabilities, dtype=object)
    except (TypeError, ValueError) as error:
        raise DataContractError("Probabilities must be a two-dimensional numerical matrix") from error
    if objects.ndim != 2 or objects.shape[1] != 10:
        raise DataContractError("Probabilities need exactly ten columns and two dimensions")
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
           for value in objects.flat):
        raise DataContractError("Probabilities must contain real numerical values without boolean or text values")
    values = np.asarray(objects, dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise DataContractError("Probabilities must be finite and nonnegative")
    if not np.all(np.abs(values.sum(axis=1) - 1.0) <= PROBABILITY_SUM_TOLERANCE):
        raise DataContractError("Each probability row must sum to one within 1e-8; no renormalization is performed")
    return values[:, [ids.index(class_id) for class_id in CLASS_IDS]].copy()


def score_ret(probabilities, payoffs, *, class_ids, signal_date, label_observed_through):
    """Apply the same frozen payoff vector to every supplied prediction row.

    Targets are not an argument, so future label availability cannot select the
    scoring cohort. DataFrame indices must be unique and nonmissing; they are preserved.
    A withheld payoff vector retains every row with a null score and status.
    """
    signal, cutoff = _cutoffs(signal_date, label_observed_through)
    verify_ret_payoffs(payoffs)
    if (signal != payoffs.provenance["signal_date"]
            or cutoff != payoffs.provenance["label_observed_through"]):
        raise DataContractError("Score signal_date and label cutoff must equal the frozen payoff dates")
    if isinstance(probabilities, pd.DataFrame):
        index = probabilities.index.copy()
        if not index.is_unique or index.to_frame(index=False).isna().any().any():
            raise DataContractError("Prediction row indices must be unique and nonmissing")
        if isinstance(index, pd.MultiIndex) and "signal_date" in index.names:
            if any(iso_date(value, "prediction row signal_date") != signal
                   for value in index.get_level_values("signal_date")):
                raise DataContractError("Prediction row signal_date differs from the frozen payoff signal")
    else:
        index = None
    aligned = align_class_probabilities(probabilities, class_ids=class_ids)
    if index is None:
        index = pd.RangeIndex(len(aligned))
    available = payoffs.provenance["vector_available"]
    scores = (aligned @ np.asarray(payoffs.table["payoff"].to_pylist(), dtype=float)
              if available else np.full(len(aligned), np.nan))
    result = pd.DataFrame({"ret_score": scores,
                           "ret_status": "available" if available else "withheld_incomplete_120_month_class_grid"},
                          index=index)
    result.attrs.update({"ret_policy": RET_POLICY, "payoff_fingerprint": payoffs.fingerprint,
                         "signal_date": signal, "label_observed_through": cutoff,
                         "canonical_class_ids": list(CLASS_IDS), "prediction_rows_retained": len(result),
                         "future_target_membership_filter_applied": False,
                         "model_ready": False})
    return result
