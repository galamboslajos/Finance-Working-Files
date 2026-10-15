"""Independently generated, pinned XNYS metadata without using quote availability.

The vendor calendar remains unchanged evidence. This reconciliation covers only
the requested months plus the full adjacent boundary months; it is not an
exchange-issued certificate of the entire research history.
"""
from __future__ import annotations

import calendar as calendar_module
from dataclasses import dataclass
from datetime import date
from importlib.metadata import PackageNotFoundError, version
import hashlib
from pathlib import Path
import re

import pyarrow as pa

from .catalog import canonical_sha256
from .data import session_dates
from .universe import DataContractError


EXCHANGE_CALENDARS_VERSION = "4.13.2"
CALENDAR_POLICY = "exchange_calendars_XNYS"
INDEPENDENT_CALENDAR_SCHEMA = pa.schema([
    ("trading_date", pa.string()), ("calendar_name", pa.string()),
    ("is_trading_session", pa.bool_()), ("session_open_utc", pa.string()),
    ("session_close_utc", pa.string()), ("is_early_close", pa.bool_()),
])


@dataclass(frozen=True)
class ReconciledExchangeCalendar:
    table: pa.Table
    source_calendar: pa.Table
    provenance: dict


def _parse_month(value, name):
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}", value):
        raise DataContractError(f"{name} must use YYYY-MM")
    try:
        return date.fromisoformat(value + "-01")
    except ValueError as error:
        raise DataContractError(f"Invalid {name}: {value!r}") from error


def _shift_month(first, increment):
    number = first.year * 12 + first.month - 1 + increment
    year, zero_month = divmod(number, 12)
    try:
        return date(year, zero_month + 1, 1)
    except ValueError as error:
        raise DataContractError("Calendar scope requires representable adjacent months") from error


def _scope(start_month, end_month):
    first, last = _parse_month(start_month, "start_month"), _parse_month(end_month, "end_month")
    count = (last.year - first.year) * 12 + last.month - first.month + 1
    if count <= 0:
        raise DataContractError("start_month is later than end_month")
    if count > 3:
        raise DataContractError("Calendar reconciliation permits at most three outcome months")
    preceding, following = _shift_month(first, -1), _shift_month(last, 1)
    end = date(following.year, following.month,
               calendar_module.monthrange(following.year, following.month)[1])
    return preceding.isoformat(), end.isoformat()


def _package_code_hash(package):
    """Fingerprint actual installed Python calendar code, not only a version label."""
    root = Path(package.__file__).resolve().parent
    records = {}
    for path in sorted(root.rglob("*.py")):
        if not path.is_file() or not path.resolve().is_relative_to(root):
            raise DataContractError("Calendar source code leaves its installed package")
        records[path.relative_to(root).as_posix()] = (
            "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest())
    if "exchange_calendar_xnys.py" not in records:
        raise DataContractError("Installed package lacks auditable XNYS calendar source")
    return records, canonical_sha256(records)


def reconcile_exchange_calendar(source_calendar, *, start_month, end_month, warmup_sessions=None):
    """Generate bounded independent sessions and expose vendor date differences.

No price IDs, values or quote dates participate. Source-only dates are recorded
instead of becoming missing-price diagnoses; independent-only dates remain
visible and required signal-date membership must be checked by the adapter.
"""
    scope_start, scope_end = _scope(start_month, end_month)
    if warmup_sessions is not None:
        if type(warmup_sessions) is not int or not 1 <= warmup_sessions <= 600:
            raise DataContractError("Bounded feature warmup_sessions must be an integer in 1..600")
        # Metadata envelope only: the feature adapter selects an exact count of
        # sessions before requesting prices. No average-days-per-month estimate
        # establishes numerical warmup or replaces the independent calendar.
        preceding = date.fromisoformat(scope_start)
        scope_start = _shift_month(preceding, -36).isoformat()
    source_dates = session_dates(source_calendar)
    if set(source_calendar["calendar_name"].to_pylist()) != {"XNYS"}:
        raise DataContractError("Calendar reconciliation requires an explicit XNYS source calendar")
    bounded_source = tuple(day for day in source_dates if scope_start <= day <= scope_end)
    try:
        installed = version("exchange-calendars")
    except PackageNotFoundError as error:
        raise DataContractError(
            f"Calendar reconciliation requires exchange-calendars=={EXCHANGE_CALENDARS_VERSION}") from error
    if installed != EXCHANGE_CALENDARS_VERSION:
        raise DataContractError(
            f"Calendar version differs from the reviewed pin: {installed}; "
            f"required {EXCHANGE_CALENDARS_VERSION}")
    import exchange_calendars as xcals

    try:
        calendar = xcals.get_calendar("XNYS", start=scope_start, end=scope_end)
        # The constructor's civil-date bounds can fall on weekends/holidays.
        # sessions_in_range rejects a bound before its first listed session;
        # select the explicitly constructed session index by civil bounds.
        labels = calendar.sessions
        days = labels.strftime("%Y-%m-%d")
        labels = labels[(days >= scope_start) & (days <= scope_end)]
    except (ValueError, KeyError) as error:
        raise DataContractError("Independent XNYS calendar cannot cover the declared scope") from error
    if calendar.name != "XNYS" or type(calendar).__module__ != "exchange_calendars.exchange_calendar_xnys":
        raise DataContractError("Calendar registry did not return the reviewed XNYS implementation")
    early = set(calendar.early_closes.strftime("%Y-%m-%d"))
    rows = []
    for label in labels:
        day = label.strftime("%Y-%m-%d")
        schedule = calendar.schedule.loc[label]
        rows.append({"trading_date": day, "calendar_name": "XNYS", "is_trading_session": True,
                     "session_open_utc": schedule["open"].isoformat(),
                     "session_close_utc": schedule["close"].isoformat(),
                     "is_early_close": day in early})
    independent = pa.Table.from_pylist(rows, schema=INDEPENDENT_CALENDAR_SCHEMA)
    independent_dates = session_dates(independent)
    if not independent_dates:
        raise DataContractError("Independent XNYS calendar returned no sessions")
    code_hashes, code_hash = _package_code_hash(xcals)
    provenance = {
        "policy": CALENDAR_POLICY, "calendar_name": "XNYS",
        "package": "exchange-calendars", "package_version": installed,
        "runtime_dependencies": {name: version(name) for name in ("pandas", "numpy", "tzdata")},
        "scope": {"outcome_start_month": start_month, "outcome_end_month": end_month,
                  "calendar_start": scope_start, "calendar_end": scope_end,
                  "boundary_policy": "Full preceding signal month and full following evidence month."},
        "source_calendar_preserved": True,
        "bounded_source_session_count": len(bounded_source),
        "independent_session_count": len(independent_dates),
        "source_only_dates": sorted(set(bounded_source) - set(independent_dates)),
        "independent_only_dates": sorted(set(independent_dates) - set(bounded_source)),
        "source_session_dates_sha256": canonical_sha256(list(bounded_source)),
        "session_dates_sha256": canonical_sha256(list(independent_dates)),
        "selected_sessions_sha256": canonical_sha256(rows),
        "calendar_code_sha256": code_hash,
        "calendar_source_file_sha256": code_hashes,
        "source_reference": "https://github.com/gerrymanoim/exchange_calendars/tree/v4.13.2",
        "evidence_status": (
            "Pinned independent rule-based XNYS metadata for this bounded scope; "
            "not an exchange-issued or all-history certification. Holidays and early "
            "closes do not establish instrument-specific availability or execution fills."),
        "selection_independence": "No price observations, available-price intersection or survivor screening.",
    }
    if warmup_sessions is not None:
        provenance["scope"]["feature_warmup_sessions_requested"] = warmup_sessions
        provenance["scope"]["boundary_policy"] = (
            "Three-year additional metadata envelope before first signal month; "
            "prices separately bounded to exact warmup sessions and latest signal.")
    provenance["request_sha256"] = canonical_sha256(provenance)
    return ReconciledExchangeCalendar(independent, source_calendar, provenance)
