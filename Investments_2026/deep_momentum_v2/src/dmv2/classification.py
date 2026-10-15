"""Shared monthly class-label slices for probability estimators.

This adapter returns the existing :class:`TrainingSlice` interface. It does not
fit preprocessing or an estimator. Original historical rows remain present even
when their next-month return or class is unavailable. Financial values outside
the explicit completed-outcome bounds are not converted to Python/pandas.
"""
from dataclasses import replace

import numpy as np
import pandas as pd
import pyarrow as pa

from .catalog import canonical_sha256
from .features import FeaturePanel, KEYS, TrainingSlice, predictor_frame
from .paper_features import PAPER_FEATURE_NAMES
from .universe import DataContractError, iso_date


_ADMISSION = ("identity_eligible", "requested", "paper_prediction_eligible")
_WINDOW_FIELDS = ("outcome_month", "outcome_start", "outcome_end")


def _metadata(table, names, label):
    if not isinstance(table, pa.Table) or len(set(table.column_names)) != len(table.column_names):
        raise DataContractError(f"{label} must be an Arrow table with unique column names")
    if not set(names) <= set(table.column_names):
        raise DataContractError(f"{label} lacks required metadata columns")
    frame = table.select(names).to_pandas().reset_index(drop=True)
    if frame[list(KEYS)].isna().any().any() or frame.duplicated(list(KEYS)).any():
        raise DataContractError(f"{label} needs every unique original source-ID/signal key")
    for sid, signal in frame[list(KEYS)].itertuples(index=False, name=None):
        if not isinstance(sid, str) or not sid.strip():
            raise DataContractError(f"{label} has an invalid source security ID")
        if iso_date(signal, "signal_date") >= "2024-01-01":
            raise DataContractError("Protected 2024+ feature or target rows cannot enter classification")
    return frame


def _window(signal, value):
    iso_date(signal, "target window signal")
    if not isinstance(value, dict) or not set(_WINDOW_FIELDS) <= set(value):
        raise DataContractError("Features need independent complete next-month calendar windows")
    start = iso_date(value["outcome_start"], "outcome_start")
    end = iso_date(value["outcome_end"], "outcome_end")
    following = str(pd.Period(signal[:7], "M") + 1)
    if (not signal < start <= end or start[:7] != end[:7]
            or value["outcome_month"] != following or end[:7] != following
            or end >= "2024-01-01"):
        raise DataContractError("Target windows must end in the complete next pre-holdout calendar month")
    return {name: value[name] for name in _WINDOW_FIELDS}


def classification_training_slice(feature_panel, target_panel, *, train_start,
                                  train_end, label_observed_through, columns=None):
    """Align ten-class targets and predictors without filtering missing labels.

    Bounds name completed OUTCOME dates. Annual rolling ten-year folds and
    shorter chronological inner partitions use this same function for both
    estimators. The caller later selects an explicit supervised ``fit_mask``;
    target availability never determines which prediction rows exist.
    """
    from .targets import MonthlyTargetPanel, verify_target_panel

    if not isinstance(feature_panel, FeaturePanel) or not isinstance(target_panel, MonthlyTargetPanel):
        raise DataContractError("Classification requires a FeaturePanel and verified MonthlyTargetPanel")
    first, last, observed = (iso_date(v, name) for v, name in (
        (train_start, "train_start"), (train_end, "train_end"),
        (label_observed_through, "label_observed_through")))
    if first > last or observed > last or last >= "2024-01-01":
        raise DataContractError("Classification needs ordered completed pre-holdout outcome bounds")
    cutoff = min(last, observed)
    windows = feature_panel.provenance.get("target_windows")
    if not isinstance(windows, dict) or not windows:
        raise DataContractError("Features lack independently pinned next-month target windows")
    validated = {signal: _window(signal, window) for signal, window in windows.items()}
    selected_windows = {signal: window for signal, window in validated.items()
                        if first <= window["outcome_start"] and window["outcome_end"] <= cutoff}
    if not selected_windows:
        raise DataContractError("No complete target month exists within the classification bounds")

    feature_meta = _metadata(feature_panel.table, list(KEYS), "Feature panel")
    coverage = _metadata(feature_panel.coverage, [*KEYS, *_ADMISSION], "Feature admission ledger")
    if set(map(tuple, feature_meta[list(KEYS)].to_numpy())) != set(map(tuple, coverage[list(KEYS)].to_numpy())):
        raise DataContractError("Feature admission ledger changed the original historical cohort")
    for name in _ADMISSION:
        if not coverage[name].map(lambda v: isinstance(v, (bool, np.bool_))).all():
            raise DataContractError("Feature admission states must be explicit booleans")
    target_meta = _metadata(target_panel.table, [*KEYS, *_WINDOW_FIELDS, *_ADMISSION], "Target panel")
    # Validate timing metadata for the entire supplied ledger before touching
    # any target values, including rows later than this particular fit cutoff.
    for row in target_meta.to_dict("records"):
        actual = _window(row["signal_date"], row)
        if row["signal_date"] in validated and actual != validated[row["signal_date"]]:
            raise DataContractError("Feature and target calendar endpoints disagree")
    positions = feature_meta.index[feature_meta.signal_date.isin(selected_windows)]
    if set(feature_meta.loc[positions, "signal_date"]) != set(selected_windows):
        raise DataContractError("A declared completed classification month has no original feature cohort")
    expected_index = pd.MultiIndex.from_frame(feature_meta.loc[positions, list(KEYS)])
    target_positions = target_meta.index[(target_meta.outcome_start >= first)
                                         & (target_meta.outcome_end <= cutoff)]
    selected_target_meta = target_meta.loc[target_positions]
    actual_index = pd.MultiIndex.from_frame(selected_target_meta[list(KEYS)])
    if not len(expected_index) or set(expected_index) != set(actual_index):
        raise DataContractError("Class targets changed the original signal-cohort denominator; retain missing members")

    # The target verifier uses metadata cutoffs before converting financial
    # columns and checks immutable per-month contents and diagnostic digests.
    first_selected_month = min(window["outcome_month"] for window in selected_windows.values())
    verify_target_panel(target_panel, through=cutoff, start_month=first_selected_month)
    chosen = None if columns is None else list(columns)
    if chosen is not None and not set(PAPER_FEATURE_NAMES) <= set(chosen):
        raise DataContractError("Classification inputs must retain all fifteen baseline predictors")
    selected_features = feature_panel.table.take(pa.array(positions.tolist(), type=pa.int64()))
    X = predictor_frame(replace(feature_panel, table=selected_features), columns=chosen)
    target = target_panel.table.take(pa.array(target_positions.tolist(), type=pa.int64())).to_pandas()
    target = target.set_index(list(KEYS)).reindex(X.index)
    admission = coverage.set_index(list(KEYS)).reindex(X.index)
    for name in _ADMISSION:
        if not target[name].equals(admission[name]):
            raise DataContractError("Feature and target signal-time admission cohorts disagree")
    if any(X[name].map(lambda value: isinstance(value, (str, bool, np.bool_))).any() for name in X):
        raise DataContractError("Class predictors must be numerical values, never text or boolean flags")
    try:
        numeric = X.astype(float)
    except (ValueError, TypeError) as error:
        raise DataContractError("Class predictors must contain numerical values or missing data") from error
    baseline = np.isfinite(numeric[list(PAPER_FEATURE_NAMES)]).all(axis=1)
    if not baseline.equals(admission.paper_prediction_eligible.astype(bool)):
        raise DataContractError("Paper15 finite inputs disagree with signal-time prediction eligibility")
    if (admission.paper_prediction_eligible & ~(admission.identity_eligible & admission.requested)).any():
        raise DataContractError("Paper15 eligibility cannot override identity or request admission")

    y = target.target_class.astype(float).rename("next_month_return_class")
    available = target.target_available.astype(bool)
    if (not y.notna().equals(available)
            or not y[available].isin(range(1, 11)).all()
            or (available & ~admission.paper_prediction_eligible).any()):
        raise DataContractError("Class labels must be available canonical IDs on the shared prediction cohort")
    ledger = target.drop(columns=[name for name in ("target_class", "label_return") if name in target]).copy()
    ledger["target_available"] = available
    ledger["feature_complete"] = np.isfinite(numeric).all(axis=1)
    ledger["model_ready"] = False
    selected_keys = [list(key) for key in X.index]
    provenance = {
        "train_start": first, "train_end": last, "label_observed_through": observed,
        "target_kind": "next_month_cross_sectional_return_class",
        "class_ids": list(range(1, 11)), "class_order": "1_highest_return_10_lowest_return",
        "target_policy_id": target_panel.provenance["policy_id"],
        "source_month_target_fingerprints": {
            window["outcome_month"]: target_panel.provenance["month_target_fingerprints"][window["outcome_month"]]
            for window in selected_windows.values()},
        "target_interval": "(signal_close,next_month_end_close]",
        "feature_interval": "ends_at_signal_close", "rows_retained": len(X),
        "original_keys_sha256": canonical_sha256(selected_keys),
        "target_statuses_do_not_select_features": True,
        "target_availability_means": "canonical_class_available_not_merely_return_observed",
        "same_interface_for_estimators": True, "rolling_window_selected_by_caller": True,
        "future_financial_rows_converted": False,
        "no_preprocessing_fit_performed": True, "estimator_fitted": False,
        "knowledge_time_verified": False, "accounting_certified": False, "model_ready": False,
    }
    return TrainingSlice(X, y, ledger, provenance)
