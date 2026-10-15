"""Exact-session monthly arithmetic with historical cohort denominators intact.

This worker reconstructs provider-adjusted endpoint ratios. It is not a return
accounting engine, CRSP-equivalent total-return certification, or executable
backtest. The source cohort is fixed before consulting any outcome quotes.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import date
import math
import re
from typing import Sequence

import pyarrow as pa

from .catalog import canonical_sha256
from .data import session_dates
from .identity_policy import (STRICT_IDENTITY_POLICY, identity_policy_provenance,
                              qualify_identity_records, validate_identity_policy)
from .universe import DataContractError, KnowledgeTimeUnavailable, UniverseSnapshot, iso_date


REQUIRED_PRICE_COLUMNS = (
    "intrinio_security_id", "trading_date", "adj_close", "frequency", "intraperiod",
    "price_quality_status",
)
OPTIONAL_AUDIT_COLUMNS = ("close", "volume", "adj_volume", "dividend", "split_ratio", "factor")
RETURN_STATUSES = (
    "identity_unresolved", "not_requested", "missing_start", "missing_end",
    "invalid_endpoint", "source_quality_flag", "ok",
)
HOLDOUT_START = "2024-01-01"
ACCEPTED_PRICE_QUALITY_STATES = ("accepted", "ok")


@dataclass(frozen=True, slots=True)
class MonthWindow:
    outcome_month: str
    signal_date: str
    outcome_start: str
    outcome_end: str
    expected_outcome_sessions: tuple[str, ...]


@dataclass(frozen=True)
class MonthlyReturnPanel:
    table: pa.Table
    summary: dict
    provenance: dict
    windows: tuple[MonthWindow, ...]


def _month(value, name):
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}", value):
        raise DataContractError(f"{name} must use YYYY-MM")
    try:
        parsed = date.fromisoformat(value + "-01")
    except ValueError as error:
        raise DataContractError(f"Invalid {name}") from error
    if parsed.isoformat()[:7] != value:
        raise DataContractError(f"{name} must use YYYY-MM")
    return value


def _step_month(value, delta):
    year, month = map(int, value.split("-"))
    year, month_zero = divmod(year * 12 + month - 1 + delta, 12)
    if not 1 <= year <= 9999:
        raise DataContractError("Requested monthly window has no representable adjacent month")
    return f"{year:04d}-{month_zero + 1:02d}"


def build_month_windows(calendar, *, start_month, end_month) -> tuple[MonthWindow, ...]:
    """Use only supplied calendar sessions, requiring adjacent-month boundary evidence.

    Start/end identify completed OUTCOME months. The signal anchor is the last
    listed session in the preceding month. A source session in the following
    month is required before claiming the outcome month has ended. This does
    not independently certify the calendar's internal exchange-day completeness.
    """
    first, last = _month(start_month, "start_month"), _month(end_month, "end_month")
    if first > last:
        raise DataContractError("start_month is later than end_month")
    if last >= "2024-01":
        raise DataContractError("Protected 2024+ financial outcome months cannot be built")
    sessions = session_dates(calendar)
    # Reject conflicting true/false observations of the same source date too.
    all_dates = calendar.select(["trading_date"]).column(0).to_pylist()
    if len(all_dates) != len(set(all_dates)):
        raise DataContractError("Duplicate calendar dates")
    grouped = {}
    for day in sessions:
        grouped.setdefault(day[:7], []).append(day)
    windows = []
    month = first
    while month <= last:
        previous, following = _step_month(month, -1), _step_month(month, 1)
        for required in (previous, month, following):
            if required not in grouped:
                raise DataContractError(f"Calendar missing required adjacent month: {required}")
        days = tuple(grouped[month])
        windows.append(MonthWindow(month, grouped[previous][-1], days[0], days[-1], days))
        month = following
    return tuple(windows)


def _required(table, names, label):
    if len(set(table.column_names)) != len(table.column_names):
        raise DataContractError(f"{label} has duplicate column names")
    missing = set(names) - set(table.column_names)
    if missing:
        raise DataContractError(f"{label} missing columns: {', '.join(sorted(missing))}")


def _identifier(value, label):
    if not isinstance(value, str) or not value.strip():
        raise DataContractError(f"Missing or invalid {label}")
    return value


def _number(value, *, positive=False):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and (not positive or value > 0))


def _quality(row):
    return (row["frequency"] == "daily" and row["intraperiod"] is False
            and row["price_quality_status"] in ACCEPTED_PRICE_QUALITY_STATES)


def _valid_quote(row):
    return row is not None and _quality(row) and _number(row["adj_close"], positive=True)


def _snapshot_cohorts(snapshots, windows, identity_policy):
    if not isinstance(snapshots, Sequence) or isinstance(snapshots, (str, bytes)):
        raise DataContractError("snapshots must be an explicit sequence of UniverseSnapshot records")
    required_dates = {window.signal_date for window in windows}
    cohorts, universes, provenance = {}, set(), {}
    for snapshot in snapshots:
        if not isinstance(snapshot, UniverseSnapshot):
            raise DataContractError("snapshots must contain UniverseSnapshot records")
        cutoff = iso_date(snapshot.as_of, "snapshot as_of")
        if cutoff not in required_dates:
            raise DataContractError("Snapshot date falls outside the required signal cutoffs")
        if cutoff in cohorts:
            raise DataContractError("Duplicate snapshot signal date")
        if snapshot.universe not in ("sp500", "sp400", "combined"):
            raise DataContractError("Unknown snapshot universe")
        universes.add(snapshot.universe)
        _required(snapshot.table, ["source_security_id", "as_of", "intrinio_security_id",
                                   "identity_status"], "snapshot")
        columns = [name for name in ("source_security_id", "as_of", "unified_security_id",
                                     "intrinio_security_id", "identity_status", "indices",
                                     "membership_tickers", "identity_review_required",
                                     "resolution_status", "source_identity_status")
                   if name in snapshot.table.column_names]
        records = snapshot.table.select(columns).to_pylist()
        seen = set()
        for record in records:
            sid = _identifier(record["source_security_id"], "source_security_id")
            if sid in seen:
                raise DataContractError("Duplicate snapshot source_security_id")
            seen.add(sid)
            if iso_date(record["as_of"], "snapshot row as_of") != cutoff:
                raise DataContractError("Snapshot row cutoff differs from snapshot as_of")
            status = _identifier(record["identity_status"], "identity_status")
            target = record["intrinio_security_id"]
            if target not in (None, ""):
                _identifier(target, "intrinio_security_id")
            elif status == "resolved":
                raise DataContractError("A resolved identity has no provider security ID")
        # Admission is recomputed across every original member, independent of
        # requested prices. Producer flags/statuses are never approval evidence.
        records = qualify_identity_records(records, identity_policy=identity_policy)
        cohorts[cutoff] = sorted(records, key=lambda row: row["source_security_id"])
        provenance[cutoff] = snapshot.provenance
    if set(cohorts) != required_dates:
        raise DataContractError("A required signal-date UniverseSnapshot is missing")
    if len(universes) != 1:
        raise DataContractError("All monthly snapshots must use the same explicit universe")
    return cohorts, next(iter(universes)), provenance


def build_monthly_returns(daily_prices, calendar, snapshots: Sequence[UniverseSnapshot], *,
                          start_month, end_month, requested_security_ids,
                          mode="historical_reconstruction",
                          identity_policy=STRICT_IDENTITY_POLICY) -> MonthlyReturnPanel:
    """Preserve every signal-cohort name while computing exact endpoint ratios.

    Removed names remain in their signal cohort; outcome-month entrants never
    appear unless present at that month's preceding signal cutoff. No price,
    current delisting flag, terminal quote, or end-of-month membership determines
    admission. Missing quote/status fields remain visible, never forward filled.
    """
    validate_identity_policy(identity_policy)
    if mode == "knowledge_time":
        raise KnowledgeTimeUnavailable("Monthly reconstruction cannot certify historical knowledge-time identity or revisions")
    if mode != "historical_reconstruction":
        raise DataContractError("Unknown monthly PIT mode")
    windows = build_month_windows(calendar, start_month=start_month, end_month=end_month)
    cohorts, universe, snapshot_provenance = _snapshot_cohorts(snapshots, windows, identity_policy)
    if (not isinstance(requested_security_ids, Sequence)
            or isinstance(requested_security_ids, (str, bytes))):
        raise DataContractError("requested_security_ids must be an explicit sequence")
    requested = tuple(_identifier(value, "requested_security_id") for value in requested_security_ids)
    if len(requested) != len(set(requested)):
        raise DataContractError("Duplicate requested provider security IDs")
    requested_set = set(requested)
    _required(daily_prices, REQUIRED_PRICE_COLUMNS, "daily_prices")
    permitted_dates = {day for window in windows
                       for day in (window.signal_date, *window.expected_outcome_sessions)}
    source_sessions = set(session_dates(calendar))
    metadata_columns = ["intrinio_security_id", "trading_date", "frequency", "intraperiod",
                        "price_quality_status"]
    metadata = daily_prices.select(metadata_columns).to_pylist()
    seen = set()
    # Validate the ENTIRE request's metadata before converting any financial field.
    for row in metadata:
        sid = _identifier(row["intrinio_security_id"], "price intrinio_security_id")
        day = iso_date(row["trading_date"], "price trading_date")
        if day >= HOLDOUT_START:
            raise DataContractError("Protected 2024+ financial rows cannot enter monthly arithmetic")
        if sid not in requested_set:
            raise DataContractError("Price ID falls outside the declared request")
        if day not in source_sessions:
            raise DataContractError("Price date is not a pinned source-calendar session")
        if day not in permitted_dates:
            raise DataContractError("Price date falls outside the monthly request")
        key = (sid, day)
        if key in seen:
            raise DataContractError("Duplicate daily price security/date key")
        seen.add(key)
    optional = [name for name in OPTIONAL_AUDIT_COLUMNS if name in daily_prices.column_names]
    financial_columns = ["adj_close", *optional]
    financial = daily_prices.select(financial_columns).to_pylist()
    quotes = {(row["intrinio_security_id"], row["trading_date"]): {**row, **values}
              for row, values in zip(metadata, financial, strict=True)}
    output = []
    for window in windows:
        expected = set(window.expected_outcome_sessions)
        for constituent in cohorts[window.signal_date]:
            sid = constituent.get("intrinio_security_id")
            eligible = constituent["identity_eligible"]
            admitted = eligible and sid in requested_set
            start_quote = quotes.get((sid, window.signal_date)) if admitted else None
            end_quote = quotes.get((sid, window.outcome_end)) if admitted else None
            path = {day: quotes[(sid, day)] for day in window.expected_outcome_sessions
                    if admitted and (sid, day) in quotes}
            missing = sorted(expected - set(path))
            invalid = sorted(day for day, row in path.items() if not _valid_quote(row))
            # The return's endpoints are signal_date and outcome_end; outcome_start
            # is an interior session, not the arithmetic start endpoint.
            interior = sorted(day for day in missing if day != window.outcome_end)
            failures = []
            if not eligible:
                status = "identity_unresolved"
            elif not admitted:
                status = "not_requested"
            elif start_quote is None:
                status = "missing_start"
            elif end_quote is None:
                status = "missing_end"
            elif not (_number(start_quote["adj_close"], positive=True)
                      and _number(end_quote["adj_close"], positive=True)):
                status = "invalid_endpoint"
            elif not (_quality(start_quote) and _quality(end_quote)):
                status = "source_quality_flag"
            else:
                status = "ok"
            if admitted:
                if start_quote is None:
                    failures.append("missing_start")
                if end_quote is None:
                    failures.append("missing_end")
                for label, quote in (("start", start_quote), ("end", end_quote)):
                    if quote is not None:
                        if not _number(quote["adj_close"], positive=True):
                            failures.append(f"{label}_invalid_endpoint")
                        if not _quality(quote):
                            failures.append(f"{label}_source_quality_flag")
            ratio = None
            if status == "ok":
                ratio = end_quote["adj_close"] / start_quote["adj_close"] - 1
                if not _number(ratio):
                    ratio, status = None, "invalid_endpoint"
                    failures.append("nonfinite_endpoint_ratio")
            def endpoint(quote):
                value = None if quote is None else quote["adj_close"]
                return float(value) if _number(value, positive=True) else None
            complete_path = bool(admitted and status == "ok" and not missing and not invalid
                                 and _valid_quote(start_quote))
            coverage_status = ("identity_unresolved" if not eligible else "not_requested" if not admitted
                               else "complete" if complete_path else "incomplete")
            row = {
                "source_security_id": constituent["source_security_id"],
                "unified_security_id": constituent.get("unified_security_id"),
                "intrinio_security_id": sid or None,
                "identity_status": constituent["identity_status"],
                "source_identity_status": constituent["source_identity_status"],
                "identity_review_required": constituent.get("identity_review_required"),
                "resolution_status": constituent.get("resolution_status"),
                "identity_eligible": eligible,
                "identity_eligibility_policy": constituent["identity_eligibility_policy"],
                "identity_eligibility_reason": constituent["identity_eligibility_reason"],
                "universe": universe, "signal_date": window.signal_date,
                "outcome_month": window.outcome_month, "outcome_start": window.outcome_start,
                "outcome_end": window.outcome_end, "return_status": status,
                "provider_adjusted_return": float(ratio) if ratio is not None else None,
                "start_adj_close": endpoint(start_quote), "end_adj_close": endpoint(end_quote),
                "start_frequency": None if start_quote is None else start_quote["frequency"],
                "end_frequency": None if end_quote is None else end_quote["frequency"],
                "start_intraperiod": None if start_quote is None else start_quote["intraperiod"],
                "end_intraperiod": None if end_quote is None else end_quote["intraperiod"],
                "start_price_quality_status": None if start_quote is None else start_quote["price_quality_status"],
                "end_price_quality_status": None if end_quote is None else end_quote["price_quality_status"],
                "observed_daily_session_count": len(path) if admitted else None,
                "expected_daily_session_count": len(expected), "daily_coverage_status": coverage_status,
                "missing_daily_dates": missing if admitted else None,
                "invalid_daily_dates": invalid if admitted else None,
                "interior_gap_dates": interior if admitted else None,
                "endpoint_failure_reasons": failures,
                "daily_quote_path_complete": complete_path,
                "daily_backtest_ready": False, "knowledge_time_verified": False,
                "accounting_certified": False,
                "monthly_volume": None, "monthly_adj_volume": None,
            }
            issues = []
            for name in optional:
                for label, quote in (("start", start_quote), ("end", end_quote)):
                    value = None if quote is None else quote[name]
                    row[f"{label}_{name}"] = float(value) if _number(value) else None
                    if quote is not None and not _number(value):
                        issues.append(f"{label}:{name}:null_or_nonfinite_or_nonnumeric")
                if name in ("volume", "adj_volume"):
                    valid_volumes = (admitted and not missing and not invalid and
                                     all(_number(quote[name]) and quote[name] >= 0 for quote in path.values()))
                    try:
                        total = math.fsum(quote[name] for quote in path.values()) if valid_volumes else None
                    except OverflowError:
                        total = None
                    row[f"monthly_{name}"] = float(total) if total is not None and _number(total) else None
                    if valid_volumes and row[f"monthly_{name}"] is None:
                        issues.append(f"monthly:{name}:nonfinite_aggregate")
            row["optional_value_issues"] = issues
            output.append(row)
    schema = pa.schema([
        ("source_security_id", pa.string()), ("unified_security_id", pa.string()),
        ("intrinio_security_id", pa.string()), ("identity_status", pa.string()),
        ("source_identity_status", pa.string()), ("universe", pa.string()),
        ("identity_review_required", pa.bool_()), ("resolution_status", pa.string()),
        ("identity_eligible", pa.bool_()), ("identity_eligibility_policy", pa.string()),
        ("identity_eligibility_reason", pa.string()),
        ("signal_date", pa.string()), ("outcome_month", pa.string()),
        ("outcome_start", pa.string()), ("outcome_end", pa.string()),
        ("return_status", pa.string()), ("provider_adjusted_return", pa.float64()),
        ("start_adj_close", pa.float64()), ("end_adj_close", pa.float64()),
        ("start_frequency", pa.string()), ("end_frequency", pa.string()),
        ("start_intraperiod", pa.bool_()), ("end_intraperiod", pa.bool_()),
        ("start_price_quality_status", pa.string()), ("end_price_quality_status", pa.string()),
        ("observed_daily_session_count", pa.int64()), ("expected_daily_session_count", pa.int64()),
        ("daily_coverage_status", pa.string()),
        ("missing_daily_dates", pa.list_(pa.string())), ("invalid_daily_dates", pa.list_(pa.string())),
        ("interior_gap_dates", pa.list_(pa.string())), ("endpoint_failure_reasons", pa.list_(pa.string())),
        ("daily_quote_path_complete", pa.bool_()), ("daily_backtest_ready", pa.bool_()),
        ("knowledge_time_verified", pa.bool_()), ("accounting_certified", pa.bool_()),
        *[(f"{label}_{name}", pa.float64()) for name in optional for label in ("start", "end")],
        ("monthly_volume", pa.float64()), ("monthly_adj_volume", pa.float64()),
        ("optional_value_issues", pa.list_(pa.string())),
    ])
    table = pa.Table.from_pylist(output, schema=schema)
    month_counts = {}
    for window in windows:
        records = [row for row in output if row["outcome_month"] == window.outcome_month]
        month_counts[window.outcome_month] = {
            "signal_date": window.signal_date, "constituents": len(records),
            "return_status_counts": dict(sorted(Counter(row["return_status"] for row in records).items())),
            "identity_eligible": sum(row["identity_eligible"] for row in records),
            "provisional_review_flags_accepted": sum(
                row["identity_eligibility_reason"] == "provisional_review_flag_accepted" for row in records),
            "daily_quote_path_complete": sum(row["daily_quote_path_complete"] for row in records),
        }
    summary = {
        "rows": table.num_rows, "months": len(windows), "universe": universe,
        "return_status_counts": dict(sorted(Counter(row["return_status"] for row in output).items())),
        "daily_quote_path_complete": sum(row["daily_quote_path_complete"] for row in output),
        "daily_backtest_ready": 0, "by_outcome_month": month_counts,
        "identity_policy": identity_policy,
        "identity_eligible_rows": sum(row["identity_eligible"] for row in output),
        "knowledge_time_verified": False, "accounting_certified": False,
    }
    provenance = {
        "mode": mode, "start_month": start_month, "end_month": end_month,
        **identity_policy_provenance(identity_policy),
        "identity_admission": "Declared research eligibility; original producer flags/statuses retained; no historical identity certification",
        "requested_security_ids": sorted(requested), "universe": universe,
        "cohort_rule": "historical source-ID membership at preceding source month-end; never outcome/end membership or quote availability",
        "return_column": "provider_adjusted_return",
        "return_formula": "adj_close[exact_outcome_end] / adj_close[exact_signal_date] - 1",
        "accepted_price_quality_states": list(ACCEPTED_PRICE_QUALITY_STATES),
        "adjustment_semantics": "provider-adjusted prices; no additional dividend or split application",
        "return_certification": "not certified CRSP total return or executable backtest return",
        "daily_counts": "outcome-month supplied calendar sessions only; preceding signal anchor excluded",
        "unobserved_coverage": "unresolved/unrequested rows have null observed counts and gap lists; expected source-session count remains known",
        "missing_values": "no forward fill, last-quote substitution, zero-return fabrication or cohort removal",
        "calendar_evidence": "supplied sessions with adjacent-month evidence; caller records the calendar origin and reconciliation",
        "optional_audit_values": "finite numeric endpoint copies; null/nonfinite/nonnumeric copied as null with issues; original daily table unchanged",
        "terminal_events": "missing end remains in denominator; delisting/merger cash flows and historical identity continuity unverified",
        "daily_backtest_ready": False, "knowledge_time_verified": False, "accounting_certified": False,
        "snapshot_provenance": snapshot_provenance,
        "calendar_sessions_sha256": canonical_sha256(list(session_dates(calendar))),
        "cohort_sha256": canonical_sha256(cohorts),
        "optional_source_columns": optional,
    }
    return MonthlyReturnPanel(table, summary, provenance, windows)
