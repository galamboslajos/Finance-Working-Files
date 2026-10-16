"""Metadata-only dependency bounds for a versioned identity amendment.

This worker accepts identifiers, dates and cohort metadata only. It never reads
files, prices, feature values, labels or model states. Counts are potential
rebuild scopes, not observations that numerical cells changed.
"""
from __future__ import annotations

from collections import Counter
from bisect import bisect_left
from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from .daily_features import daily_feature_specs
from .context_features import context_feature_specs, RELATIVE_WINDOWS
from .paper_features import PAPER_FEATURE_NAMES
from .universe import DataContractError

META_COLUMNS = frozenset({
    "source_security_id", "signal_date", "as_of", "indices", "index_id",
    "unified_security_id", "intrinio_security_id", "identity_status",
    "source_identity_status", "identity_review_required", "resolution_status",
    "identity_eligible", "identity_eligibility_policy", "identity_eligibility_reason",
    "membership_effective_date", "membership_availability_date", "membership_end_date",
    "membership_record_count", "membership_records_json", "membership_tickers",
    "bridge_method", "bridge_confidence", "knowledge_time_verified", "accounting_certified",
    "year_artifact", "universe", "pit_mode", "source_product_id",
    "requested", "paper_prediction_eligible", "paper_common_population",
})
DEPENDENCY_COLUMNS = ["source_security_id", "signal_date", "artifact_type", "family", "scope",
    "potential_columns_per_row", "dependency_reason", "coverage_basis", "financial_values_decoded"]


@dataclass(frozen=True)
class IdentityImpactPlan:
    direct_keys: pd.DataFrame
    dependent_keys: pd.DataFrame
    scope_counts: pd.DataFrame
    payoff_signal_windows: pd.DataFrame
    summary: dict

    @property
    def tables(self):
        return {name: getattr(self, name) for name in (
            "direct_keys", "dependent_keys", "scope_counts", "payoff_signal_windows")}


def _day(value, name):
    if not isinstance(value, (str, pd.Timestamp)):
        raise DataContractError(f"{name} requires an explicit metadata date")
    try:
        day = pd.Timestamp(value)
    except (ValueError, TypeError, OverflowError) as error:
        raise DataContractError(f"Invalid {name}") from error
    if pd.isna(day) or day.tzinfo is not None or day != day.normalize():
        raise DataContractError(f"{name} requires a complete timezone-naive date")
    return day.strftime("%Y-%m-%d")


def _sleeves(value):
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple, set)) or hasattr(value, "tolist"):
        values = list(value)
    elif value is None or value is pd.NA or pd.isna(value):
        values = []
    else:
        raise DataContractError("Cohort index labels must be explicit strings or sequences")
    if any(not isinstance(item, str) or item not in ("sp500", "sp400") for item in values):
        raise DataContractError("Impact cohorts support only original S&P500/400 index labels")
    return tuple(sorted(set(values)))


def _cohorts(frame, name):
    if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique:
        raise DataContractError(f"{name} requires a metadata DataFrame with unique columns")
    unexpected = set(frame) - META_COLUMNS
    if unexpected:
        raise DataContractError(f"{name} contains non-cohort/financial columns: {sorted(unexpected)}")
    if "source_security_id" not in frame or not {"signal_date", "as_of"} & set(frame):
        raise DataContractError(f"{name} requires original source/date keys")
    date_column = "signal_date" if "signal_date" in frame else "as_of"
    rows, date_cache = [], {}
    projection = [column for column in ("source_security_id", "signal_date", "as_of", "indices", "index_id",
                  "unified_security_id", "intrinio_security_id") if column in frame]
    for row in frame.loc[:, projection].to_dict("records"):
        sid = row["source_security_id"]
        if not isinstance(sid, str) or not sid.strip() or sid != sid.strip():
            raise DataContractError("Cohort source IDs must be canonical nonblank strings")
        raw_day = row[date_column]
        if not isinstance(raw_day, (str, pd.Timestamp)):
            raise DataContractError("Cohort dates must be explicit date metadata")
        if raw_day not in date_cache:
            date_cache[raw_day] = _day(raw_day, name + " dates")
        day = date_cache[raw_day]
        if day >= "2024-01-01":
            raise DataContractError("Protected 2024+ cohort metadata is outside this impact plan")
        if "signal_date" in row and "as_of" in row and _day(row["as_of"], "as_of") != day:
            raise DataContractError("Cohort signal_date/as_of disagree")
        indices = _sleeves(row.get("indices"))
        index_id = _sleeves(row.get("index_id"))
        if indices and index_id and indices != index_id:
            raise DataContractError("Cohort index_id/indices disagree")
        sleeve = indices or index_id
        rows.append(dict(source_security_id=sid, signal_date=day, sleeves=sleeve,
            unified_security_id=row.get("unified_security_id"),
            intrinio_security_id=row.get("intrinio_security_id")))
    result = pd.DataFrame(rows, columns=["source_security_id", "signal_date", "sleeves", "unified_security_id", "intrinio_security_id"])
    # Annual warmup artifacts overlap. Identical source/date/index metadata is
    # one dependency; disagreeing historical routes are an evidence conflict.
    unique = result.drop_duplicates(ignore_index=True)
    if unique.duplicated(["source_security_id", "signal_date"]).any():
        raise DataContractError("Overlapping annual cohort metadata disagrees")
    return unique


def _counts(dependencies, financial_read_through):
    rows = []
    columns = ["artifact_type", "family", "scope", "potential_rows", "potential_source_ids", "potential_signals",
        "potential_cells_upper_bound", "metadata_rows_after_financial_boundary", "row_unit", "actual_values_compared"]
    for (artifact, family, scope), frame in dependencies.groupby(["artifact_type", "family", "scope"], sort=True):
        rows.append(dict(artifact_type=artifact, family=family, scope=scope, potential_rows=len(frame),
            potential_source_ids=frame.source_security_id.nunique(), potential_signals=frame.signal_date.nunique(),
            potential_cells_upper_bound=int(frame.potential_columns_per_row.sum()),
            metadata_rows_after_financial_boundary=int(frame.signal_date.gt(financial_read_through).sum()),
            row_unit="original_source_signal", actual_values_compared=False))
    return pd.DataFrame(rows, columns=columns)


def plan_identity_impact(signal_cohorts, daily_cohorts, sessions: Iterable, *, affected_source_ids,
        amendment_start, amendment_end=None, analysis_start=None, analysis_end=None,
        financial_read_through="2019-12-31", daily_metadata_provenance=None,
        materialize_dependent_keys=True) -> IdentityImpactPlan:
    """Return conservative feature/label/payoff dependency scopes from metadata.

    ``daily_cohorts`` may be None, in which case nonlocal daily benchmark tails
    are explicitly UNKNOWN; monthly membership is never expanded into daily
    membership. A supplied daily table must be an authoritative extraction of
    actual original membership records; annual identical warmup overlap is
    deduplicated. No price availability, labels or feature values select rows.
    Amendment dates express the logical audit scope, not legal identifier
    validity. Run separately for an active pre-2020 amendment and a broader
    historical metadata impact audit.
    """
    if type(materialize_dependent_keys) is not bool or daily_metadata_provenance is not None and not isinstance(daily_metadata_provenance, dict):
        raise DataContractError("Explicit materialization flag and metadata-provenance mapping required")
    monthly = _cohorts(signal_cohorts, "signal_cohorts")
    daily = None if daily_cohorts is None else _cohorts(daily_cohorts, "daily_cohorts")
    if monthly.empty:
        raise DataContractError("An impact plan requires the original monthly cohort inventory")
    source_ids = tuple(affected_source_ids)
    if (not source_ids or len(set(source_ids)) != len(source_ids)
            or any(not isinstance(sid, str) or not sid.strip() for sid in source_ids)):
        raise DataContractError("Affected source IDs must be explicit, nonempty and unique")
    start = _day(amendment_start, "amendment_start")
    end = _day(amendment_end, "amendment_end") if amendment_end is not None else monthly.signal_date.max()
    lower = _day(analysis_start, "analysis_start") if analysis_start is not None else monthly.signal_date.min()
    upper = _day(analysis_end, "analysis_end") if analysis_end is not None else monthly.signal_date.max()
    boundary = _day(financial_read_through, "financial_read_through")
    if start > end or lower > upper or upper >= "2024-01-01" or boundary >= "2024-01-01":
        raise DataContractError("Impact scope dates must be ordered and analysis/preexisting boundary pre-2024")
    calendar = tuple(_day(value, "sessions") for value in sessions)
    if not calendar or calendar != tuple(sorted(set(calendar))):
        raise DataContractError("Sessions must be complete explicit ordered unique metadata dates")
    positions = {day: index for index, day in enumerate(calendar)}
    all_days = set(monthly.signal_date) | (set() if daily is None else set(daily.signal_date))
    if not all_days <= set(positions):
        raise DataContractError("Cohort dates are absent from the supplied session metadata")
    monthly = monthly.loc[monthly.signal_date.between(lower, upper)].copy()
    if monthly.empty:
        raise DataContractError("Analysis scope contains no original monthly cohort rows")
    direct = monthly.loc[monthly.source_security_id.isin(source_ids) & monthly.signal_date.between(start, end)].copy()
    changed_signal_days = set(direct.signal_date)
    direct_keys = set(zip(direct.source_security_id, direct.signal_date))
    direct_sleeves = {day: set(sleeve for value in group.sleeves for sleeve in value)
                      for day, group in direct.groupby("signal_date", sort=False)}
    affected_daily = None if daily is None else daily.loc[
        daily.source_security_id.isin(source_ids) & daily.signal_date.between(start, end)].copy()
    # Changed source memberships must identify their historical sleeve. An
    # empty extraction is allowed when there are no such keys, but missing
    # evidence on an actual changed key must not silently erase propagation.
    if direct.sleeves.map(len).eq(0).any():
        raise DataContractError("Affected monthly identity keys require nonempty valid sleeve metadata")
    if affected_daily is not None and affected_daily.sleeves.map(len).eq(0).any():
        raise DataContractError("Affected daily identity keys require nonempty valid sleeve metadata")
    changed_daily_days = set() if affected_daily is None else set(affected_daily.signal_date)
    changed_sleeve_days = {label: set() for label in ("sp500", "sp400")}
    if affected_daily is not None:
        for row in affected_daily.itertuples(index=False):
            for label in row.sleeves:
                changed_sleeve_days[label].add(row.signal_date)
    dependencies, scope_records = [], []

    def add(frame, artifact, family, scope, columns, reason, basis="original_observed_cohort_metadata"):
        if frame.empty:
            return
        scope_records.append(dict(artifact_type=artifact, family=family, scope=scope, potential_rows=len(frame),
            potential_source_ids=frame.source_security_id.nunique(), potential_signals=frame.signal_date.nunique(),
            potential_cells_upper_bound=int(columns*len(frame)),
            metadata_rows_after_financial_boundary=int(frame.signal_date.gt(boundary).sum()),
            row_unit="original_source_signal", actual_values_compared=False))
        if not materialize_dependent_keys:
            return
        for row in frame.itertuples(index=False):
            dependencies.append(dict(source_security_id=row.source_security_id, signal_date=row.signal_date,
                artifact_type=artifact, family=family, scope=scope, potential_columns_per_row=columns,
                dependency_reason=reason, coverage_basis=basis, financial_values_decoded=False))

    family_counts = Counter(spec["family"] for spec in daily_feature_specs())
    for family, count in family_counts.items():
        add(direct, "predictor", family, "direct_source", count, "native_route_or_admission_changes_local_feature_attachment")
    same_signal = monthly.loc[monthly.signal_date.isin(changed_signal_days)]
    add(same_signal, "predictor", "paper_momentum_state", "common_five_population", len(PAPER_FEATURE_NAMES),
        "changed_common_population_or_momenta_can_change_shared_mean_SD_and_all_standardized_rows")
    context = context_feature_specs()
    for scope, name in (("universe", "universe_cross_section"), ("index_sleeve", "same_historical_sleeve")):
        count = sum(spec["family"] == "cross_sectional" and spec.get("model_eligible", True)
                    and spec["name"].startswith("xs_" + scope + "_") for spec in context)
        affected = same_signal if scope == "universe" or same_signal.empty else same_signal.loc[same_signal.apply(
            lambda row: not row.sleeves or bool(set(row.sleeves) & direct_sleeves[row.signal_date]), axis=1)]
        add(affected, "predictor", "cross_sectional", name, count,
            "same_date_rank_MAD_or_original_denominator_80_90pct_coverage_gate_can_change")
    # Monthly market states use current-day actual membership, not a frozen
    # backward cohort. In an incomplete daily inventory, same-month exposure is
    # only a conservative potential scope and explicitly labelled unverified.
    market_days = changed_signal_days | (changed_daily_days & set(monthly.signal_date))
    market_rows = monthly.loc[monthly.signal_date.isin(market_days)]
    market_count = sum(spec["family"] == "market_state" and spec.get("model_eligible", True) for spec in context)
    add(market_rows, "predictor", "market_state", "shared_current_signal_state", market_count,
        "amended_signal_day_daily_cohort_can_change_shared_market_aggregates",
        "unknown_daily_membership_no_fill" if daily is None else "observed_daily_and_monthly_metadata_potential_union")
    changed_positions = sorted(positions[day] for day in changed_daily_days)
    changed_sleeve_positions = {label: sorted(positions[day] for day in days)
                                for label, days in changed_sleeve_days.items()}
    direct_mask = pd.Series(list(zip(monthly.source_security_id, monthly.signal_date)), index=monthly.index).isin(direct_keys)

    def touches(day, window, changed):
        slot = positions[day]
        index = bisect_left(changed, slot - window + 1)
        return index < len(changed) and changed[index] <= slot

    for window in RELATIVE_WINDOWS:
        market_affected_signal = {day for day in monthly.signal_date.unique() if touches(day, window, changed_positions)}
        market_rows = monthly.loc[monthly.signal_date.isin(market_affected_signal) | direct_mask]
        add(market_rows, "predictor", "relative_context", f"market_{window}s", 4,
            "direct_source_or_changed_daily_market_benchmark_in_exact_trailing_return_pairs",
            "direct_only_nonlocal_daily_tail_unknown" if daily is None else "observed_daily_benchmark_dependency")
        sleeve_affected = {label: {day for day in monthly.signal_date.unique()
                                  if touches(day, window, changed)}
                           for label, changed in changed_sleeve_positions.items()}
        sleeve_mask = monthly.apply(lambda row: (row.source_security_id, row.signal_date) in direct_keys or any(
            row.signal_date in sleeve_affected[label] for label in (row.sleeves or ("sp500", "sp400"))), axis=1)
        add(monthly.loc[sleeve_mask], "predictor", "relative_context", f"sleeve_{window}s", 2,
            "direct_source_or_changed_same_sleeve_daily_benchmark_in_trailing_pairs",
            "direct_only_nonlocal_daily_tail_unknown" if daily is None else "observed_daily_benchmark_dependency")
    add(direct, "monthly_return", "monthly_return", "direct_source_outcome", 1,
        "same_signal_source_route_determines_signal_and_next_month_adjusted_endpoints")
    add(same_signal, "target", "target_class", "complete_same_month_rank_population", 1,
        "changed_return_or_prediction_eligibility_can_change_rank_N_ties_and_other_stock_classes")
    result = pd.DataFrame(dependencies, columns=DEPENDENCY_COLUMNS)
    if not result.empty:
        result = result.drop_duplicates(["source_security_id", "signal_date", "artifact_type", "family", "scope"], ignore_index=True)
    scope_counts = pd.DataFrame(scope_records, columns=["artifact_type", "family", "scope", "potential_rows",
        "potential_source_ids", "potential_signals", "potential_cells_upper_bound", "metadata_rows_after_financial_boundary",
        "row_unit", "actual_values_compared"])
    scope_counts = scope_counts.sort_values(["artifact_type","family","scope"],ignore_index=True)
    signal_months = sorted({pd.Period(day[:7], "M") for day in monthly.signal_date})
    payoff_records = []
    affected_ret_months = set()
    for outcome in sorted({pd.Period(day[:7], "M") + 1 for day in changed_signal_days}):
        last = outcome + 119
        selected = [month for month in signal_months if outcome <= month <= last]
        affected_ret_months.update(selected)
        payoff_records.append(dict(changed_outcome_month=str(outcome), first_payoff_signal_month=str(outcome),
            last_possible_payoff_signal_month=str(last), bounded_signal_month_count=len(selected),
            potential_class_cells_per_signal=10, completed_history_gate_unchanged=True, values_compared=False))
    payoff = pd.DataFrame(payoff_records, columns=["changed_outcome_month", "first_payoff_signal_month",
        "last_possible_payoff_signal_month", "bounded_signal_month_count", "potential_class_cells_per_signal",
        "completed_history_gate_unchanged", "values_compared"])
    if affected_ret_months:
        scope_counts = pd.concat([scope_counts, pd.DataFrame([dict(artifact_type="RET_payoff",family="realized_class_payoff",
            scope="120_calendar_month_dependency_union",potential_rows=len(affected_ret_months),potential_source_ids=0,
            potential_signals=len(affected_ret_months),potential_cells_upper_bound=10*len(affected_ret_months),
            metadata_rows_after_financial_boundary=sum(month > pd.Period(boundary[:7], "M") for month in affected_ret_months),
            row_unit="ten_class_payoff_signal_vector",actual_values_compared=False)])],ignore_index=True)
    summary = dict(method="metadata_potential_dependency_upper_bounds_v1", amendment_start=start,amendment_end=end,
        logical_scope_is_legal_identifier_validity=False,analysis_start=lower,analysis_end=upper,
        financial_read_through=boundary,research_financial_values_read=False,financial_values_compared=False,
        estimators_fitted=False,backtests_run=False,affected_source_ids=list(source_ids),original_monthly_rows=len(monthly),
        direct_source_signal_rows=len(direct),direct_signal_count=len(changed_signal_days),
        distinct_affected_daily_membership_days=None if affected_daily is None else len(changed_daily_days),
        daily_inventory_supplied=daily is not None,nonlocal_relative_closure_unknown=daily is None,
        daily_metadata_provenance=daily_metadata_provenance,
        daily_inventory_semantics="supplied_observed_membership_keys_only_no_absence_or_daily_fill_inference",
        affected_source_sleeve_evidence_required=True,
        other_monthly_missing_sleeve_policy="conservative_union_of_both_sleeves_for_dependency_bounds",
        other_monthly_missing_sleeve_rows=int(monthly.sleeves.map(len).eq(0).sum()),
        dependent_keys_materialized=materialize_dependent_keys,
        maximum_native_lookback_sessions=503,maximum_relative_return_pairs=252,maximum_benchmark_tail_later_sessions=251,
        forward_24_month_market_memory=False,original_denominators_preserved=True,
        daily_membership_filled_from_monthly=False,potential_not_actual_changed_cells=True,
        native_technical_other_source_rows_changed_by_join_only=False,
        predictor_potential_cells_upper_bound=int(scope_counts.loc[scope_counts.artifact_type.eq("predictor"),"potential_cells_upper_bound"].sum()) if not scope_counts.empty else 0,
        saved_target_or_RET_financial_values_consumed=False,old_runs_mutated=False)
    return IdentityImpactPlan(direct.reset_index(drop=True),result,scope_counts,payoff,summary)
