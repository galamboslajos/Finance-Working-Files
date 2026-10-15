"""Audits of the fixed shared targets and strictly trailing RET calendar grids.

The reports retain original denominators. Availability is conditional on the
observed provider-adjusted outcomes; it is not evidence that missing terminal
returns, identities, knowledge time, or executable returns are certified.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import json
import math
from typing import Mapping

import pyarrow as pa
import pyarrow.compute as pc

from .catalog import canonical_sha256
from .monthly import _step_month
from .ret import CLASS_IDS, PAYOFF_MONTHS, build_ret_payoffs
from .targets import (MonthlyTargetPanel, SAFE_METADATA, TARGET_POLICY,
                      target_month_fingerprint, verify_target_panel)
from .universe import DataContractError, iso_date


PERIODS = (
    ("full_2010_2023", "2010-01", "2023-12"),
    ("initial_training_2010_2019", "2010-01", "2019-12"),
    ("nominal_test_2020_2023", "2020-01", "2023-12"),
    ("before_2016", "2010-01", "2015-12"),
    ("2016_2023", "2016-01", "2023-12"),
)


def _table(rows):
    return pa.Table.from_pylist(rows)


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def combine_target_panels(panels, *, source_year_pins: Mapping) -> MonthlyTargetPanel:
    """Combine verified annual outputs without reconstructing ranks or labels.

    ``source_year_pins`` is the calling persistence layer's immutable annual
    artifact manifest. It is bound into provenance; the caller verifies those
    file hashes before constructing the annual input panels.
    """
    panels = list(panels)
    if not panels or not isinstance(source_year_pins, Mapping) or not source_year_pins:
        raise DataContractError("Annual target panels and immutable source-year pins are required")
    seen_months, windows, records = set(), [], []
    source_hashes = []
    for panel in panels:
        verify_target_panel(panel)
        months = set(panel.provenance["month_target_fingerprints"])
        if months & seen_months:
            raise DataContractError("Annual target panels overlap outcome months")
        seen_months |= months
        windows.extend(deepcopy(panel.provenance["month_windows"]))
        source_hashes.append(panel.artifact_sha256)
        records.extend(panel.table.to_pylist())
    windows.sort(key=lambda row: row["signal_date"])
    expected = [_step_month(min(seen_months), i) for i in range(len(seen_months))]
    if sorted(seen_months) != expected:
        raise DataContractError("Combined target history must retain consecutive calendar months")
    years = {month[:4] for month in seen_months}
    if {str(year) for year in source_year_pins} != years:
        raise DataContractError("Source-year pins must exactly cover combined target outcome years")
    records.sort(key=lambda row: (row["signal_date"], row["source_security_id"]))
    table = pa.Table.from_pylist(records, schema=panels[0].table.schema)
    diagnostics = pa.concat_tables([panel.diagnostics for panel in panels]).sort_by("signal_date")
    if any(panel.table.schema != table.schema for panel in panels):
        raise DataContractError("Annual target schemas differ")
    index = defaultdict(list)
    for position, row in enumerate(records):
        index[row["outcome_month"]].append(position)
    diagnostic_index = {row["outcome_month"]: i
                        for i, row in enumerate(diagnostics.select(["outcome_month"]).to_pylist())}
    # Preserve policy/limitations while replacing all annual scope-dependent pins.
    provenance = deepcopy(panels[0].provenance)
    provenance.update({
        "label_observed_through": max(panel.provenance["label_observed_through"] for panel in panels),
        "month_windows": windows,
        "input_key_sha256": canonical_sha256([[row["source_security_id"], row["signal_date"]] for row in records]),
        "metadata_sha256": target_month_fingerprint(table.select(list(SAFE_METADATA))),
        "month_target_fingerprints": {month: target_month_fingerprint(
            table.take(pa.array(index[month], type=pa.int64()))) for month in sorted(index)},
        "month_diagnostic_fingerprints": {month: target_month_fingerprint(
            diagnostics.slice(diagnostic_index[month], 1)) for month in sorted(index)},
        "rows_retained": table.num_rows,
        "source_annual_target_fingerprints": source_hashes,
        "source_year_pins": deepcopy(dict(source_year_pins)),
        "source_outcome_provenance_sha256": canonical_sha256([
            panel.provenance["source_outcome_provenance_sha256"] for panel in panels]),
        "combination_policy": "verified_annual_targets_concatenate_no_reranking_v1",
    })
    combined = MonthlyTargetPanel(table, diagnostics, provenance, canonical_sha256(provenance))
    verify_target_panel(combined)
    return combined


def _summarize(diagnostics):
    fields = ("original_members", "identity_eligible_members", "prediction_eligible_members",
              "observed_return_members", "labelled_members", "unrequested_identity_eligible_members",
              "past_feature_unrequested_identity_members", "outcome_unrequested_identity_members",
              "outcome_request_unknown_identity_members", "source_not_requested_identity_members",
              "missing_outcome_records", "tied_return_groups")
    result = {field: sum(row[field] for row in diagnostics) for field in fields}
    result.update({
        "calendar_months": len(diagnostics),
        "observed_original_coverage": _ratio(result["observed_return_members"], result["original_members"]),
        "observed_prediction_coverage": _ratio(result["observed_return_members"], result["prediction_eligible_members"]),
        "labelled_original_coverage": _ratio(result["labelled_members"], result["original_members"]),
        "labelled_prediction_coverage": _ratio(result["labelled_members"], result["prediction_eligible_members"]),
        "identity_ineligible_members": result["original_members"] - result["identity_eligible_members"],
        "prediction_ineligible_members": result["original_members"] - result["prediction_eligible_members"],
        "eligible_unlabelled_members": result["prediction_eligible_members"] - result["labelled_members"],
        "all_requests_exhaustive": all(row["exhaustive_requests"] for row in diagnostics),
        "rank_ok_months": sum(row["rank_status"] == "ok" for row in diagnostics),
        "months_with_tied_returns": sum(row["tied_return_groups"] > 0 for row in diagnostics),
        "maximum_tie_size": max((row["maximum_tie_size"] for row in diagnostics), default=0),
        "months_with_empty_classes": sum(bool(row["empty_classes"]) for row in diagnostics),
        "empty_month_class_cells": sum(len(row["empty_classes"]) for row in diagnostics),
    })
    return result


def aggregate_target_coverage(target_panel):
    """Return auditable class/missingness counts with all ten class bins present."""
    verify_target_panel(target_panel)
    rows = target_panel.table.to_pylist()
    diagnostics = target_panel.diagnostics.to_pylist()
    by_month = {row["outcome_month"]: row for row in diagnostics}
    monthly, class_month = [], []
    for month in sorted(by_month):
        diagnostic = by_month[month]
        monthly.append({**diagnostic, **_summarize([diagnostic]), "outcome_year": int(month[:4])})
        counts = json.loads(diagnostic["class_counts_json"])
        for class_id in CLASS_IDS:
            count = counts[str(class_id)]
            class_month.append({"signal_date": diagnostic["signal_date"], "outcome_month": month,
                "outcome_year": int(month[:4]), "class_id": class_id, "labelled_members": count,
                "share_of_labelled": _ratio(count, diagnostic["labelled_members"]),
                "empty_class": count == 0})
    def status_report(field):
        reasons = []
        counts = Counter((row["outcome_month"], row[field], row["prediction_eligible"]) for row in rows)
        for (month, status, eligible), count in sorted(counts.items()):
            diagnostic = by_month[month]
            reasons.append({"outcome_month": month, "outcome_year": int(month[:4]),
                field: status, "prediction_eligible": eligible, "members": count,
                "share_of_original": _ratio(count, diagnostic["original_members"]),
                "share_of_prediction_eligible": _ratio(count, diagnostic["prediction_eligible_members"]) if eligible else None})
        return reasons
    annual, periods, class_year, class_period, rolling, rolling_classes = [], [], [], [], [], []
    years = sorted({month[:4] for month in by_month})
    def summarize_classes(selected, label):
        total = sum(row["labelled_members"] for row in selected)
        return [{**label, "class_id": class_id,
                 "labelled_members": sum(row["labelled_members"] for row in selected if row["class_id"] == class_id),
                 "share_of_labelled": _ratio(sum(row["labelled_members"] for row in selected if row["class_id"] == class_id), total),
                 "calendar_months_with_class": sum(not row["empty_class"] for row in selected if row["class_id"] == class_id),
                 "absent_in_entire_window": not any(row["labelled_members"] for row in selected if row["class_id"] == class_id)}
                for class_id in CLASS_IDS]
    for year in years:
        selected = [row for row in diagnostics if row["outcome_month"].startswith(year)]
        annual.append({"outcome_year": int(year), **_summarize(selected)})
        class_year.extend(summarize_classes([row for row in class_month if row["outcome_year"] == int(year)], {"outcome_year": int(year)}))
    for label, first, last in PERIODS:
        selected = [row for row in diagnostics if first <= row["outcome_month"] <= last]
        if selected:
            periods.append({"period": label, "first_outcome_month": first, "last_outcome_month": last, **_summarize(selected)})
            class_period.extend(summarize_classes([row for row in class_month if first <= row["outcome_month"] <= last], {"period": label}))
    # The classifier is refit annually on ten years. RET below refreshes monthly.
    for test_year in range(2020, 2024):
        first, last = f"{test_year - 10}-01", f"{test_year - 1}-12"
        selected = [row for row in diagnostics if first <= row["outcome_month"] <= last]
        class_rows = summarize_classes([row for row in class_month if first <= row["outcome_month"] <= last],
                                      {"test_year": test_year, "first_outcome_month": first, "last_outcome_month": last})
        complete = {row["outcome_month"] for row in selected} == {_step_month(first, i) for i in range(120)}
        absent = [row["class_id"] for row in class_rows if row["absent_in_entire_window"]]
        rolling.append({"test_year": test_year, "first_outcome_month": first, "last_outcome_month": last,
                        "all_120_calendar_months_present": complete, "absent_classes": absent,
                        "all_ten_classes_present": not absent, **_summarize(selected)})
        rolling_classes.extend(class_rows)
    summary = {"target_policy": TARGET_POLICY, **_summarize(diagnostics),
        "target_panel_sha256": target_panel.artifact_sha256,
        "target_status_counts": dict(Counter(row["target_status"] for row in rows)),
        "eligible_target_status_counts": dict(Counter(row["target_status"] for row in rows if row["prediction_eligible"])),
        "return_observation_status_counts": dict(Counter(row["return_observation_status"] for row in rows)),
        "eligible_return_observation_status_counts": dict(Counter(row["return_observation_status"] for row in rows if row["prediction_eligible"])),
        "class_counts": {str(class_id): sum(row["target_class"] == class_id for row in rows) for class_id in CLASS_IDS},
        "original_rows_retained": len(rows), "future_label_filter_applied_to_prediction_cohort": False,
        "return_basis": "provider_adjusted_close_ratio", "economic_return_certified": False,
        "knowledge_time_certified": False, "survivorship_bias_free_certified": False,
        "estimator_fit_performed": False,
        "interpretation": "Ranks and class means are conditional on observed returns; missing outcomes remain unlabelled."}
    return {"tables": {"target_coverage_month": _table(monthly), "target_coverage_year": _table(annual),
        "target_coverage_period": _table(periods), "target_status_month": _table(status_report("target_status")),
        "return_observation_status_month": _table(status_report("return_observation_status")),
        "target_class_month": _table(class_month), "target_class_year": _table(class_year),
        "target_class_period": _table(class_period), "rolling_training_coverage": _table(rolling),
        "rolling_training_classes": _table(rolling_classes)}, "summary": summary}


def audit_ret_coverage(target_panel, signals, *, canonical_signal_dates=None):
    """Audit all requested 120×10 grids and cross-check nominal tests canonically.

    Financial values beyond the latest requested signal are never decoded.
    Means are computed within completed outcome-month/classes, then selected by
    each signal's own strict 120-calendar-month window. Every nominal 2020–2023
    test signal is independently passed through ``build_ret_payoffs`` by default.
    """
    signal_dates = [iso_date(value, "signal_date") for value in signals]
    if not signal_dates or len(signal_dates) != len(set(signal_dates)):
        raise DataContractError("RET audit requires nonempty unique signal dates")
    if any(signal >= "2024-01-01" for signal in signal_dates):
        raise DataContractError("RET audit cannot request protected 2024+ signals")
    signal_dates.sort()
    source_cutoff = iso_date(target_panel.provenance["label_observed_through"], "target observation cutoff")
    maximum_cutoff = min(signal_dates[-1], source_cutoff)
    first_requested_month = _step_month(signal_dates[0][:7], -119)
    verify_target_panel(target_panel, through=maximum_cutoff, start_month=first_requested_month)
    windows = {row["outcome_month"]: row for row in target_panel.provenance["month_windows"]}
    safe = target_panel.table.select(["outcome_month", "prediction_eligible"]).to_pylist()
    original_counts, eligible_counts = Counter(), Counter()
    for row in safe:
        original_counts[row["outcome_month"]] += 1
        eligible_counts[row["outcome_month"]] += int(row["prediction_eligible"])
    month_groups = {}
    for month, window in windows.items():
        if not first_requested_month <= month <= signal_dates[-1][:7] or window["outcome_end"] > maximum_cutoff:
            continue
        selected = target_panel.table.filter(pc.equal(target_panel.table["outcome_month"], month))
        groups = {class_id: [] for class_id in CLASS_IDS}
        for row in selected.select(["target_available", "target_class", "label_return"]).to_pylist():
            if row["target_available"]:
                value, class_id = row["label_return"], row["target_class"]
                if (class_id not in CLASS_IDS or isinstance(class_id, bool) or value is None
                        or isinstance(value, bool) or not math.isfinite(value) or value < -1):
                    raise DataContractError("Invalid observed realized target class or return")
                groups[class_id].append(value)
        month_groups[month] = {class_id: (len(values), math.fsum(value / len(values) for value in values) if values else None)
                               for class_id, values in groups.items()}
    if canonical_signal_dates is None:
        canonical_dates = {signal for signal in signal_dates
                           if "2020-01" <= _step_month(signal[:7], 1) <= "2023-12"}
    else:
        canonical_dates = {iso_date(value, "canonical signal date") for value in canonical_signal_dates}
        if not canonical_dates <= set(signal_dates):
            raise DataContractError("Canonical RET signals must belong to the requested signal audit")
    signal_report, grid, payoffs, canonical = [], [], [], []
    for signal in signal_dates:
        first, last = _step_month(signal[:7], -119), signal[:7]
        effective_cutoff = min(signal, source_cutoff)
        monthly = []
        for offset in range(PAYOFF_MONTHS):
            month = _step_month(first, offset)
            window = windows.get(month)
            end = None if window is None else window["outcome_end"]
            if window is None:
                status = "missing_month_completion_evidence"
            elif end > effective_cutoff:
                status = "month_not_observed_by_cutoff"
            elif not original_counts[month]:
                status = "missing_month"
            else:
                status = None
            for class_id in CLASS_IDS:
                count, mean = (0, None) if status is not None else month_groups[month][class_id]
                monthly.append({"outcome_month": month, "outcome_end": end, "class_id": class_id,
                    "original_member_count": original_counts[month],
                    "prediction_eligible_count": eligible_counts[month], "member_count": count,
                    "class_mean_return": mean,
                    "status": status or ("available" if count else "empty_realized_class")})
        available = all(row["status"] == "available" for row in monthly)
        class_rows = [{"class_id": class_id,
            "payoff": math.fsum(row["class_mean_return"] / PAYOFF_MONTHS for row in monthly if row["class_id"] == class_id) if available else None,
            "status": "available" if available else "withheld_incomplete_120_month_class_grid"}
            for class_id in CLASS_IDS]
        fingerprint = None
        if signal in canonical_dates:
            actual = build_ret_payoffs(target_panel, signal_date=signal, label_observed_through=signal)
            if actual.monthly.to_pylist() != monthly or actual.table.to_pylist() != class_rows:
                raise DataContractError("RET calendar audit differs from the canonical shared payoff worker")
            fingerprint = actual.fingerprint
            canonical.append(actual)
        statuses = Counter(row["status"] for row in monthly)
        prediction_month = _step_month(last, 1)
        signal_report.append({"signal_date": signal, "prediction_outcome_month": prediction_month,
            "prediction_outcome_year": int(prediction_month[:4]), "first_outcome_month": first,
            "last_outcome_month": last, "label_observed_through": signal,
            "expected_calendar_months": 120, "expected_month_class_cells": 1200,
            "available_month_class_cells": statuses["available"],
            "unavailable_month_class_cells": 1200 - statuses["available"],
            "calendar_months_with_completion_evidence": sum(_step_month(first, i) in windows for i in range(120)),
            "calendar_months_completed_by_signal": sum(
                _step_month(first, i) in windows and windows[_step_month(first, i)]["outcome_end"] <= effective_cutoff for i in range(120)),
            "missing_month_class_status_counts_json": json.dumps(dict(sorted(statuses.items())), sort_keys=True),
            "vector_available": available, "canonical_worker_verified": signal in canonical_dates,
            "canonical_payoff_fingerprint": fingerprint,
            "nominal_test_signal": "2020-01" <= prediction_month <= "2023-12"})
        grid.extend({"signal_date": signal, **row} for row in monthly)
        payoffs.extend({"signal_date": signal, **row} for row in class_rows)
    yearly = []
    for year in sorted({row["prediction_outcome_year"] for row in signal_report}):
        selected = [row for row in signal_report if row["prediction_outcome_year"] == year]
        yearly.append({"prediction_outcome_year": year, "signals": len(selected),
            "available_vectors": sum(row["vector_available"] for row in selected),
            "withheld_vectors": sum(not row["vector_available"] for row in selected),
            "canonical_worker_verified_signals": sum(row["canonical_worker_verified"] for row in selected),
            "available_month_class_cells": sum(row["available_month_class_cells"] for row in selected),
            "expected_month_class_cells": 1200 * len(selected)})
    nominal = [row for row in signal_report if row["nominal_test_signal"]]
    summary = {"signals": len(signal_report), "available_vectors": sum(row["vector_available"] for row in signal_report),
        "withheld_vectors": sum(not row["vector_available"] for row in signal_report),
        "nominal_test_signals": len(nominal), "nominal_test_available_vectors": sum(row["vector_available"] for row in nominal),
        "canonical_worker_verified_signals": len(canonical),
        "first_available_signal": next((row["signal_date"] for row in signal_report if row["vector_available"]), None),
        "strict_calendar_months": 120, "classes": list(CLASS_IDS),
        "calendar_grid_rows": len(grid), "class_payoff_rows": len(payoffs),
        "maximum_financial_outcome_end_consumed": max((window["outcome_end"] for month, window in windows.items() if month in month_groups), default=None),
        "source_target_panel_sha256": target_panel.artifact_sha256,
        "equal_month_weighting": True, "incomplete_vector_withheld": True,
        "prediction_cohort_changed": False, "estimator_fit_performed": False,
        "economic_return_certified": False, "knowledge_time_certified": False,
        "same_close_operational_availability_certified": False}
    return {"tables": {"ret_coverage_signal": _table(signal_report), "ret_coverage_year": _table(yearly),
        "ret_month_class_grid": _table(grid), "ret_class_payoffs": _table(payoffs)},
        "summary": summary, "canonical_payoffs": canonical}
