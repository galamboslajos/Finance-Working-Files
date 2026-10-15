"""Shared next-month classes, retaining every original signal-cohort member.

The midrank/tie and missing-outcome conventions are V2 engineering decisions,
not undocumented author settings. This module never reads source files or fits
a model. Provider-adjusted endpoint ratios remain provisional research returns.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from copy import deepcopy
import hashlib
import json
import math
from numbers import Real
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa

from .catalog import canonical_sha256
from .monthly import MonthWindow, MonthlyReturnPanel, _month, _step_month
from .universe import DataContractError, iso_date


TARGET_POLICY = "monthly_paper_qualified_midrank_deciles_v1"
CLASS_IDS = tuple(range(1, 11))
KEYS = ("source_security_id", "signal_date")
COHORT_FIELDS = (*KEYS, "identity_eligible", "requested", "paper_prediction_eligible")
WINDOW_FIELDS = ("outcome_month", "outcome_start", "outcome_end")
OUTCOME_METADATA = (*KEYS, *WINDOW_FIELDS, "requested")
SAFE_METADATA = (*KEYS, *WINDOW_FIELDS, "identity_eligible", "requested",
                 "paper_prediction_eligible", "prediction_eligible", "outcome_requested")
TARGET_POLICY_DEFINITION = {
    "policy_id": TARGET_POLICY,
    "universe": "historical_sp500_sp400_union",
    "classes": list(CLASS_IDS), "class_1": "highest_return",
    "ranking_population": "signal_time_identity_and_paper15_prediction_eligible",
    "ranking": "descending_average_rank_exact_equal_values",
    "class_formula": "clip(1+floor(10*(average_rank-0.5)/observed_n),1,10)",
    "minimum_observed_population": 10,
    "exhaustive_requests": "past_feature_and_forward_outcome_all_original_identity_eligible_members",
    "missing_return": "retain_original_row_never_impute",
    "return_basis": "provider_adjusted_close_ratio",
    "accepted_return_status": "ok", "minimum_return": -1,
    "outcome_years": [2010, 2023],
    "target_availability_determines_prediction_eligibility": False,
}
DEFAULT_POLICY = deepcopy(TARGET_POLICY_DEFINITION)
_TARGET_POLICY_SHA256 = canonical_sha256(TARGET_POLICY_DEFINITION)


@dataclass(frozen=True)
class MonthlyTargetPanel:
    table: pa.Table
    diagnostics: pa.Table
    provenance: dict
    artifact_sha256: str


def target_month_fingerprint(table: pa.Table) -> str:
    """Hash exact Arrow values/schema independently of chunking/schema metadata."""
    if not isinstance(table, pa.Table):
        raise DataContractError("Target fingerprint requires an Arrow table")
    # Slicing/take can preserve different offsets/null bitmaps even when the
    # logical values match. Rebuild only this already-authorized projection.
    canonical = pa.Table.from_pylist(table.to_pylist(), schema=table.schema.remove_metadata())
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, canonical.schema) as stream:
        stream.write_table(canonical)
    return "sha256:" + hashlib.sha256(sink.getvalue()).hexdigest()


def target_panel_fingerprint(panel: MonthlyTargetPanel) -> str:
    """Bind detached provenance, including per-month table and diagnostic pins.

    This metadata-only operation deliberately does not decode financial rows.
    ``verify_target_panel`` checks the requested completed-month values as well.
    """
    if not isinstance(panel, MonthlyTargetPanel):
        raise DataContractError("Expected a MonthlyTargetPanel")
    try:
        return canonical_sha256(panel.provenance)
    except (TypeError, ValueError) as error:
        raise DataContractError("Target provenance is not canonical JSON") from error


def _table(value, name):
    if not isinstance(value, (pa.Table, pd.DataFrame)):
        raise DataContractError(f"{name} must be an Arrow table or DataFrame")
    columns = value.column_names if isinstance(value, pa.Table) else list(value.columns)
    if len(columns) != len(set(columns)):
        raise DataContractError(f"Duplicate {name} column names")
    return value, columns


def _metadata(table, columns):
    # Project before Python conversion; never convert the full outcome table.
    if isinstance(table, pa.Table):
        return table.select(list(columns)).to_pylist()
    return table.loc[:, list(columns)].to_dict("records")


def _identifier(value):
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise DataContractError("source_security_id must be a nonblank canonical string")
    return value


def _boolean(value, name):
    if not isinstance(value, (bool, np.bool_)):
        raise DataContractError(f"{name} must contain explicit nonnull boolean values")
    return bool(value)


def _window(raw):
    if isinstance(raw, MonthWindow):
        raw = {"signal_date": raw.signal_date, "outcome_month": raw.outcome_month,
               "outcome_start": raw.outcome_start, "outcome_end": raw.outcome_end}
    if not isinstance(raw, Mapping) or not {"signal_date", *WINDOW_FIELDS} <= set(raw):
        raise DataContractError("Explicit complete next-month calendar windows are required")
    signal = iso_date(raw["signal_date"], "signal_date")
    start = iso_date(raw["outcome_start"], "outcome_start")
    end = iso_date(raw["outcome_end"], "outcome_end")
    month = _month(raw["outcome_month"], "outcome_month")
    if month < "2010-01" or month >= "2024-01":
        raise DataContractError("Target outcome months must be 2010–2023; protected 2024+ values cannot enter")
    if not signal < start <= end or start[:7] != month or end[:7] != month or _step_month(signal[:7], 1) != month:
        raise DataContractError("Targets must occupy the next calendar month strictly after the signal")
    return {"signal_date": signal, "outcome_month": month,
            "outcome_start": start, "outcome_end": end}


def _windows(raw):
    if isinstance(raw, Mapping):
        values = []
        for signal, value in raw.items():
            if not isinstance(value, Mapping):
                raise DataContractError("Calendar windows must contain explicit date bounds")
            if "signal_date" in value and value["signal_date"] != signal:
                raise DataContractError("Calendar window signal key differs from its explicit signal date")
            values.append({**value, "signal_date": signal})
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        values = list(raw)
    else:
        raise DataContractError("Supply independent month_windows or MonthlyReturnPanel.windows")
    result = [_window(row) for row in values]
    if not result or len({r["signal_date"] for r in result}) != len(result) or len({r["outcome_month"] for r in result}) != len(result):
        raise DataContractError("Calendar windows must be nonempty and unique by signal and outcome month")
    return {row["signal_date"]: row for row in result}


def _safe_rows(table, required, name):
    table, columns = _table(table, name)
    if not set(required) <= set(columns):
        raise DataContractError(f"{name} is missing required columns")
    return _metadata(table, required)


def _keys(rows):
    result = []
    for row in rows:
        result.append((_identifier(row["source_security_id"]), iso_date(row["signal_date"], "signal_date")))
    if len(result) != len(set(result)):
        raise DataContractError("Duplicate original cohort/outcome key")
    return result


def _finite_return(value):
    if value is None or value is pd.NA:
        return None, "missing_return"
    if not isinstance(value, Real) or isinstance(value, (bool, np.bool_)):
        raise DataContractError("Return values must be numeric, never booleans or coerced strings")
    value = float(value)
    if not math.isfinite(value):
        return None, "nonfinite_return"
    if value < -1:
        return None, "invalid_return_lower_bound"
    return value, "ok"


def build_monthly_targets(original_cohort, outcomes, *, label_observed_through,
                          month_windows=None, policy=None) -> MonthlyTargetPanel:
    """Assign shared completed-outcome classes without selecting future survivors.

    ``original_cohort`` explicitly supplies signal-time identity/request/Paper15
    flags. Independent calendar bounds are mandatory. Extra/duplicate outcome
    keys fail; missing records remain visible. Metadata for *all* rows/windows
    is checked before reading any return/status value. Future uncompleted rows
    remain unlabelled without converting their financial values.
    """
    if policy not in (None, TARGET_POLICY):
        raise DataContractError("Unknown monthly target policy")
    observed = iso_date(label_observed_through, "label_observed_through")
    if observed >= "2024-01-01":
        raise DataContractError("Target observation cutoff must remain before protected 2024+")
    wrapped = isinstance(outcomes, MonthlyReturnPanel)
    source_provenance = outcomes.provenance if wrapped else None
    if month_windows is None and wrapped:
        month_windows = outcomes.windows
    windows = _windows(month_windows)
    outcome_table = outcomes.table if wrapped else outcomes
    _, outcome_columns = _table(outcome_table, "Outcomes")
    if wrapped and ("requested" not in outcome_columns or
                    "requested_security_ids" in source_provenance and "intrinio_security_id" in outcome_columns):
        # The existing monthly worker records requests in its provenance rather
        # than repeating a boolean column. Reconstruct that explicit manifest;
        # never infer a request from the presence of a quote or return.
        request_ids = source_provenance.get("requested_security_ids")
        if not isinstance(request_ids, (list, tuple)):
            raise DataContractError("Monthly return provenance requires an explicit unique outcome request manifest")
        for request_id in request_ids:
            _identifier(request_id)
        if len(request_ids) != len(set(request_ids)):
            raise DataContractError("Monthly return outcome request manifest contains duplicate IDs")
        if "intrinio_security_id" not in outcome_columns:
            raise DataContractError("Monthly return request manifest requires native ID metadata")
        native_rows = _metadata(outcome_table, ("intrinio_security_id",))
        requested_ids = set(request_ids)
        request_flags = [isinstance(row["intrinio_security_id"], str) and row["intrinio_security_id"] in requested_ids for row in native_rows]
        if "requested" in outcome_columns:
            explicit_flags = [_boolean(row["requested"], "outcome requested") for row in _metadata(outcome_table, ("requested",))]
            if explicit_flags != request_flags:
                raise DataContractError("Explicit outcome requests disagree with the monthly request manifest")
        else:
            if isinstance(outcome_table, pa.Table):
                outcome_table = outcome_table.append_column("requested", pa.array(request_flags, type=pa.bool_()))
            else:
                outcome_table = outcome_table.assign(requested=request_flags)
            outcome_columns = [*outcome_columns, "requested"]
    if not {"return_status", "provider_adjusted_return"} <= set(outcome_columns):
        raise DataContractError("Outcomes require return_status and provider_adjusted_return")
    cohort = _safe_rows(original_cohort, COHORT_FIELDS, "Original cohort")
    outcome_metadata = _safe_rows(outcome_table, OUTCOME_METADATA, "Outcomes")
    cohort_keys, outcome_keys = _keys(cohort), _keys(outcome_metadata)
    if not cohort:
        raise DataContractError("Original cohort must be nonempty")
    for row in cohort:
        signal = row["signal_date"]
        if signal not in windows:
            raise DataContractError("Original cohort signal lacks independent next-month calendar bounds")
        for field in COHORT_FIELDS[2:]:
            row[field] = _boolean(row[field], field)
        row["prediction_eligible"] = row["identity_eligible"] and row["paper_prediction_eligible"]
    # Check every outcome's timing, including extra and future rows, first.
    for row in outcome_metadata:
        candidate = _window(row)
        if row["signal_date"] not in windows or candidate != windows[row["signal_date"]]:
            raise DataContractError("Outcome metadata differs from independent complete calendar bounds")
        row["requested"] = _boolean(row["requested"], "outcome requested")
    if not set(outcome_keys) <= set(cohort_keys):
        raise DataContractError("Outcome ledger contains keys outside the original signal cohort")
    used_windows = {s: windows[s] for _, s in cohort_keys}
    outcome_lookup = dict(zip(outcome_keys, range(len(outcome_keys))))
    cohort_lookup = dict(zip(cohort_keys, cohort))
    completed_indices = [i for i, row in enumerate(outcome_metadata)
                         if row["outcome_end"] <= observed and
                         cohort_lookup[outcome_keys[i]]["identity_eligible"]]
    # Consult statuses first. Rejected source rows do not require converting
    # their numeric field at all, and future rows are absent from both reads.
    if isinstance(outcome_table, pa.Table):
        statuses = outcome_table.select(["return_status"]).take(pa.array(completed_indices, type=pa.int64())).column(0).to_pylist()
    else:
        statuses = outcome_table.loc[:, "return_status"].iloc[completed_indices].tolist()
    if any(not isinstance(status, str) or not status for status in statuses):
        raise DataContractError("Source return_status must be an explicit nonblank string")
    accepted_indices = [index for index, status in zip(completed_indices, statuses)
                        if status == "ok" and outcome_metadata[index]["requested"]
                        and cohort_lookup[outcome_keys[index]]["prediction_eligible"]]
    if isinstance(outcome_table, pa.Table):
        values = outcome_table.select(["provider_adjusted_return"]).take(pa.array(accepted_indices, type=pa.int64())).column(0).to_pylist()
    else:
        values = outcome_table.loc[:, "provider_adjusted_return"].iloc[accepted_indices].tolist()
    accepted_values = dict(zip(accepted_indices, values))
    financial = {index: {"return_status": status, "provider_adjusted_return": accepted_values.get(index)}
                 for index, status in zip(completed_indices, statuses)}
    rows = []
    for member, key in zip(cohort, cohort_keys):
        row = {**member, **used_windows[key[1]], "label_return": None, "return_available": False,
               "observed_return_available": False,
               "target_class": None, "target_available": False, "source_return_status": None}
        idx = outcome_lookup.get(key)
        row["outcome_requested"] = outcome_metadata[idx]["requested"] if idx is not None else None
        if not member["identity_eligible"]:
            status = "identity_ineligible"
        elif not member["paper_prediction_eligible"]:
            status = "paper_prediction_ineligible"
        elif row["outcome_end"] > observed:
            status = "outcome_not_completed"
        elif idx is None:
            status = "missing_outcome_record"
        else:
            source = financial[idx]
            source_status = source["return_status"]
            if not isinstance(source_status, str) or not source_status:
                raise DataContractError("Source return_status must be an explicit nonblank string")
            row["source_return_status"] = source_status
            if not row["outcome_requested"]:
                status = "outcome_not_requested"
            elif source_status != "ok":
                status = "source_return_status:" + source_status
            else:
                value, status = _finite_return(source["provider_adjusted_return"])
                row["label_return"] = value
                row["return_available"] = status == "ok"
                row["observed_return_available"] = status == "ok"
        row["return_observation_status"] = status
        row["target_status"] = status
        rows.append(row)
    diagnostics = []
    for signal in sorted(used_windows):
        month_rows = [row for row in rows if row["signal_date"] == signal]
        eligible = [row for row in month_rows if row["prediction_eligible"]]
        valid = [row for row in eligible if row["return_available"]]
        past_unrequested = sum(row["identity_eligible"] and not row["requested"] for row in month_rows)
        forward_unrequested = sum(row["identity_eligible"] and row["outcome_requested"] is False for row in month_rows)
        forward_unknown = sum(row["identity_eligible"] and row["outcome_requested"] is None for row in month_rows)
        source_unrequested = {outcome_keys[index] for index, source in financial.items()
                              if source["return_status"] == "not_requested"}
        unrequested = sum(row["identity_eligible"] and
                         (not row["requested"] or row["outcome_requested"] is not True or
                          (row["source_security_id"], signal) in source_unrequested)
                         for row in month_rows)
        complete = used_windows[signal]["outcome_end"] <= observed
        if not complete:
            rank_status = "outcome_not_completed"
        elif unrequested:
            rank_status = "non_exhaustive_original_requests"
        elif len(valid) < 10:
            rank_status = "insufficient_observed_population"
        else:
            rank_status = "ok"
            descending = sorted(valid, key=lambda row: row["label_return"], reverse=True)
            lower = 0
            while lower < len(descending):
                upper = lower + 1
                while upper < len(descending) and descending[upper]["label_return"] == descending[lower]["label_return"]:
                    upper += 1
                # Positions L=lower+1,U=upper: exact midrank integer arithmetic.
                class_id = min(10, max(1, 1 + (10 * (lower + upper)) // (2 * len(valid))))
                for row in descending[lower:upper]:
                    row["target_class"] = class_id
                    row["target_available"] = True
                    row["target_status"] = "ok"
                lower = upper
        if rank_status != "ok":
            for row in valid:
                row["target_status"] = rank_status
        populated = {row["target_class"] for row in valid if row["target_available"]}
        tie_counts = Counter(row["label_return"] for row in valid)
        class_counts = Counter(row["target_class"] for row in valid if row["target_available"])
        diagnostics.append({**used_windows[signal], "original_members": len(month_rows),
            "identity_eligible_members": sum(row["identity_eligible"] for row in month_rows),
            "prediction_eligible_members": len(eligible), "observed_return_members": len(valid),
            "labelled_members": sum(row["target_available"] for row in month_rows),
            "unrequested_identity_eligible_members": unrequested,
            "past_feature_unrequested_identity_members": past_unrequested,
            "outcome_unrequested_identity_members": forward_unrequested,
            "outcome_request_unknown_identity_members": forward_unknown,
            "source_not_requested_identity_members": sum((row["source_security_id"], signal) in source_unrequested for row in month_rows),
            "missing_outcome_records": sum((row["source_security_id"], signal) not in outcome_lookup for row in month_rows),
            "observed_original_coverage": len(valid) / len(month_rows),
            "observed_prediction_coverage": len(valid) / len(eligible) if eligible else None,
            "exhaustive_requests": unrequested == 0, "rank_status": rank_status,
            "empty_classes": [k for k in CLASS_IDS if k not in populated],
            "class_counts_json": json.dumps({str(k): class_counts.get(k, 0) for k in CLASS_IDS}, sort_keys=True),
            "tied_return_groups": sum(n > 1 for n in tie_counts.values()),
            "maximum_tie_size": max(tie_counts.values(), default=0),
            "return_observation_status_counts_json": json.dumps(dict(Counter(row["return_observation_status"] for row in month_rows)), sort_keys=True),
            "model_ready": False})
    schema = pa.schema([(k, pa.string()) for k in (*KEYS, *WINDOW_FIELDS)] +
        [(k, pa.bool_()) for k in ("identity_eligible", "requested", "paper_prediction_eligible", "prediction_eligible", "outcome_requested")] +
        [("target_class", pa.int8()), ("label_return", pa.float64()), ("return_available", pa.bool_()),
         ("observed_return_available", pa.bool_()),
         ("target_available", pa.bool_()), ("target_status", pa.string()),
         ("return_observation_status", pa.string()), ("source_return_status", pa.string())])
    table = pa.Table.from_pylist(rows, schema=schema)
    diagnostic_table = pa.Table.from_pylist(diagnostics)
    months = [used_windows[s]["outcome_month"] for s in sorted(used_windows)]
    month_indices = {month: [i for i, row in enumerate(rows) if row["outcome_month"] == month] for month in months}
    provenance = {"policy_id": TARGET_POLICY, "policy": deepcopy(TARGET_POLICY_DEFINITION),
        "policy_sha256": _TARGET_POLICY_SHA256,
        "label_observed_through": observed, "month_windows": [used_windows[s] for s in sorted(used_windows)],
        "input_key_sha256": canonical_sha256([list(k) for k in cohort_keys]),
        "metadata_sha256": target_month_fingerprint(table.select(list(SAFE_METADATA))),
        "month_target_fingerprints": {m: target_month_fingerprint(table.take(pa.array(month_indices[m], type=pa.int64()))) for m in months},
        "month_diagnostic_fingerprints": {m: target_month_fingerprint(diagnostic_table.slice(i, 1)) for i, m in enumerate(months)},
        "source_outcome_provenance_sha256": canonical_sha256(source_provenance) if source_provenance is not None else None,
        "rows_retained": len(rows), "model_ready": False,
        "knowledge_time_verified": False, "accounting_certified": False,
        "identity_verified": False, "survivorship_bias_free_certified": False,
        "return_basis": "provider_adjusted_close_ratio",
        "limitations": ["identity_admission_is_provisional", "date_only_observation_cutoff",
            "retrieval_time_adjusted_price_history", "missing_terminal_outcomes_not_imputed",
            "observed_return_ranks_may_be_selected", "no_executable_economic_accounting"]}
    result = MonthlyTargetPanel(table, diagnostic_table, provenance, canonical_sha256(provenance))
    verify_target_panel(result)
    return result


def verify_target_panel(panel: MonthlyTargetPanel, *, through=None, start_month=None) -> None:
    """Verify provenance/all date metadata, then only the requested month values.

    RET can verify a historical 120-month slice without decoding later returns.
    A whole-panel call validates every table/diagnostic month against its pin.
    """
    if target_panel_fingerprint(panel) != panel.artifact_sha256:
        raise DataContractError("Target provenance fingerprint mismatch")
    provenance = panel.provenance
    if provenance.get("policy_id") != TARGET_POLICY or provenance.get("policy") != TARGET_POLICY_DEFINITION or provenance.get("policy_sha256") != _TARGET_POLICY_SHA256:
        raise DataContractError("Target policy differs from the fixed shared target contract")
    observed = iso_date(provenance.get("label_observed_through"), "label_observed_through")
    cutoff = observed if through is None else iso_date(through, "through")
    if observed >= "2024-01-01" or cutoff >= "2024-01-01" or cutoff > observed:
        raise DataContractError("Target verification cutoff exceeds observed pre-holdout evidence")
    first = None if start_month is None else _month(start_month, "start_month")
    windows = _windows(provenance.get("month_windows"))
    metadata = _safe_rows(panel.table, SAFE_METADATA, "Target panel")
    keys = _keys(metadata)
    if len(metadata) != provenance.get("rows_retained") or canonical_sha256([list(k) for k in keys]) != provenance.get("input_key_sha256"):
        raise DataContractError("Target original-key denominator fingerprint mismatch")
    for row in metadata:
        if _window(row) != windows.get(row["signal_date"]):
            raise DataContractError("Target table changed its independent calendar bounds")
        for field in SAFE_METADATA[5:]:
            if field != "outcome_requested" or row[field] is not None:
                _boolean(row[field], field)
        if row["prediction_eligible"] != (row["identity_eligible"] and row["paper_prediction_eligible"]):
            raise DataContractError("Target availability changed signal-time prediction eligibility")
    if target_month_fingerprint(panel.table.select(list(SAFE_METADATA))) != provenance.get("metadata_sha256"):
        raise DataContractError("Target signal metadata fingerprint mismatch")
    diagnostic_metadata = _safe_rows(panel.diagnostics, ("signal_date", *WINDOW_FIELDS), "Target diagnostics")
    if len(diagnostic_metadata) != len(windows) or any(_window(row) != windows.get(row["signal_date"]) for row in diagnostic_metadata):
        raise DataContractError("Target diagnostic calendar windows changed")
    months = {row["outcome_month"] for row in metadata}
    if months != set(provenance.get("month_target_fingerprints", {})) or months != set(provenance.get("month_diagnostic_fingerprints", {})):
        raise DataContractError("Target per-month integrity manifest is incomplete")
    for month in sorted(months):
        bounds = next(w for w in windows.values() if w["outcome_month"] == month)
        if through is not None and bounds["outcome_end"] > cutoff or first is not None and month < first:
            continue
        indices = [i for i, row in enumerate(metadata) if row["outcome_month"] == month]
        selected = panel.table.take(pa.array(indices, type=pa.int64()))
        if target_month_fingerprint(selected) != provenance["month_target_fingerprints"][month]:
            raise DataContractError("Target monthly value fingerprint mismatch")
        diagnostic_indices = [i for i, row in enumerate(diagnostic_metadata) if row["outcome_month"] == month]
        selected_diagnostic = panel.diagnostics.take(pa.array(diagnostic_indices, type=pa.int64()))
        if target_month_fingerprint(selected_diagnostic) != provenance["month_diagnostic_fingerprints"][month]:
            raise DataContractError("Target monthly diagnostic fingerprint mismatch")
