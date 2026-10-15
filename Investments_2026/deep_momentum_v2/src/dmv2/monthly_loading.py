"""A bounded monthly engineering sample selected before any price observations.

The full source-ID signal cohorts remain in the returned monthly ledger. The
few price IDs are chosen from the first signal cohort only, so later membership
or quote availability cannot turn this diagnostic into a survivor intersection.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import date
import re

import pyarrow as pa
import pyarrow.compute as pc

from .catalog import SourceCatalog, canonical_sha256
from .data import BRIDGE_COLUMNS, CALENDAR_COLUMNS, LINK_COLUMNS, MEMBERSHIP_COLUMNS,session_dates
from .exchange_sessions import CALENDAR_POLICY, reconcile_exchange_calendar
from .quote_sessions import exact_session_quotes
from .loading import (
    BoundedTableReader, BudgetExceeded, HOLDOUT_START, LoadedTable, PriceLoader,
    ProtectedDataError, ReadBudget,
)
from .monthly import MonthlyReturnPanel, build_month_windows, build_monthly_returns
from .universe import DataContractError, KnowledgeTimeUnavailable, UniverseSnapshot, build_snapshot
from .identity_policy import (STRICT_IDENTITY_POLICY,validate_identity_policy,
                             select_engineering_ids,identity_policy_provenance)


DAILY_SAMPLE_COLUMNS = (
    "intrinio_security_id", "trading_date", "frequency", "intraperiod",
    "open", "high", "low", "close", "volume",
    "adj_open", "adj_high", "adj_low", "adj_close", "adj_volume",
    "factor", "split_ratio", "dividend", "raw_ohlc_bounds_valid",
    "adjusted_ohlc_bounds_valid", "price_quality_status",
)


@dataclass(frozen=True)
class LoadedMonthlySample:
    daily_prices: LoadedTable
    panel: MonthlyReturnPanel
    snapshots: tuple[UniverseSnapshot, ...]
    calendar: pa.Table
    source_calendar: pa.Table
    requested_ids: tuple[str, ...]
    loading_stats: dict
    loading_provenance: dict


def _month_number(value, name):
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}", value):
        raise DataContractError(f"{name} must use YYYY-MM")
    try:
        first = date.fromisoformat(value + "-01")
    except ValueError as error:
        raise DataContractError(f"Invalid {name}: {value!r}") from error
    return first.year * 12 + first.month - 1


def _preflight(start_month, end_month, universe, sample_size, mode, budget):
    # These checks must happen before constructing a source reader or asking
    # the catalog for any file, pin, footer or value.
    if mode == "knowledge_time":
        raise KnowledgeTimeUnavailable(
            "Pinned sources cannot certify knowledge-time membership, identities "
            "or price revisions; the monthly sample requires historical_reconstruction.")
    if mode != "historical_reconstruction":
        raise DataContractError("Unknown monthly reconstruction mode")
    if universe not in ("sp500", "sp400", "combined"):
        raise DataContractError("universe must be sp500, sp400 or combined")
    if type(sample_size) is not int or not 1 <= sample_size <= 10:
        raise DataContractError("sample_size must be an integer from 1 through 10")
    first = _month_number(start_month, "start_month")
    last = _month_number(end_month, "end_month")
    if first > last:
        raise DataContractError("start_month is later than end_month")
    if first == 12:  # January year 1 has no representable previous month.
        raise DataContractError("start_month requires a representable preceding signal month")
    if last - first + 1 > 3:
        raise DataContractError("Engineering samples permit at most three outcome months")
    if date.fromisoformat(end_month + "-01") >= HOLDOUT_START:
        raise ProtectedDataError("Monthly outcomes must end before protected 2024-01-01")
    if budget is not None and not isinstance(budget, ReadBudget):
        raise DataContractError("budget must be a ReadBudget")
    return budget or ReadBudget(max_scanned_rows=12_000_000)


_BUDGET_STATS = {
    "max_metadata_files": "metadata_files",
    "max_data_files": "data_files",
    "max_scanned_rows": "scanned_rows",
    "max_output_rows": "returned_rows",
    "max_read_bytes": "projected_compressed_bytes",
}


def _remaining_budget(ceiling, stats):
    remaining = {}
    for limit, measured in _BUDGET_STATS.items():
        used = sum(stage[measured] for stage in stats.values())
        left = getattr(ceiling, limit) - used
        if left <= 0:
            raise BudgetExceeded(f"Monthly stage budget exhausted: {limit}, "
                                 f"used={used}, limit={getattr(ceiling, limit)}")
        remaining[limit] = left
    return ReadBudget(**remaining)


def load_monthly_sample(catalog: SourceCatalog, *, start_month, end_month, universe,
                        sample_size=2, budget=None, mode="historical_reconstruction",
                        identity_policy=STRICT_IDENTITY_POLICY):
    """Load at most three outcome months and retain unchanged daily input rows.

Budgets cover this complete stage cumulatively, including small shared metadata
tables. The sample is an engineering diagnostic; it is not the intended complete
research universe. No current-master or future-price filter chooses its IDs.
"""
    validate_identity_policy(identity_policy)
    ceiling = _preflight(start_month, end_month, universe, sample_size, mode, budget)
    reader = BoundedTableReader(catalog, ceiling)
    stats, provenance = {}, {}

    def capture(name, loaded):
        stats[name] = loaded.stats
        provenance[name] = loaded.provenance
        return loaded.table

    source_calendar = capture("calendar", reader.read_metadata(
        "siblis", "trading_calendar_sessions", columns=CALENDAR_COLUMNS,
        allowed_columns=CALENDAR_COLUMNS, max_rows=20_000,
        key_columns=("trading_date",)))
    reconciliation = reconcile_exchange_calendar(
        source_calendar, start_month=start_month, end_month=end_month)
    calendar = reconciliation.table
    windows = build_month_windows(calendar, start_month=start_month, end_month=end_month)
    signal_dates = tuple(window.signal_date for window in windows)
    if not signal_dates:
        raise DataContractError("Calendar produced no monthly windows")

    # A single filtered membership request avoids rescanning the history once
    # per month; only the exact preceding-month signal sessions reach Python.
    indices = ("sp500", "sp400") if universe == "combined" else (universe,)
    reader.budget = _remaining_budget(ceiling, stats)
    membership = capture("membership", reader.read(
        "siblis", "membership_by_index_by_date",
        start=min(signal_dates), end=max(signal_dates),
        columns=MEMBERSHIP_COLUMNS, allowed_columns=MEMBERSHIP_COLUMNS,
        key_columns=("index_id", "security_id", "trading_date"),
        filters={"index_id": indices, "trading_date": signal_dates}, financial=False))
    for day in signal_dates:
        if membership.filter(pc.equal(membership["trading_date"], day)).num_rows == 0:
            raise DataContractError(f"No source membership observations for required signal: {day}")

    reader.budget = _remaining_budget(ceiling, stats)
    bridge = capture("bridge", reader.read_metadata(
        "identity", "security_identity_bridge", columns=BRIDGE_COLUMNS,
        allowed_columns=BRIDGE_COLUMNS, max_rows=30_000,
        key_columns=("source_product_id", "source_security_id")))
    reader.budget = _remaining_budget(ceiling, stats)
    links = capture("links", reader.read_metadata(
        "intrinio", "intrinio_identity_links", columns=LINK_COLUMNS,
        allowed_columns=LINK_COLUMNS, max_rows=20_000,
        key_columns=("unified_security_id",)))

    pins = catalog.provenance()
    snapshots = []
    for day in signal_dates:
        dated = membership.filter(pc.equal(membership["trading_date"], day))
        snapshot = build_snapshot(dated, bridge, links, as_of=day,
                                  universe=universe, mode=mode)
        if not snapshot.table.num_rows:
            raise DataContractError(
                f"All membership observations failed date/availability gates: {day}")
        snapshots.append(replace(snapshot, provenance={**snapshot.provenance, "source_pins": pins}))
    snapshots = tuple(snapshots)

    requested_ids=select_engineering_ids(snapshots[0],sample_size,identity_policy=identity_policy)
    if not requested_ids:
        raise DataContractError("First signal cohort has no resolved engineering price IDs")
    daily_prices = PriceLoader(catalog, _remaining_budget(ceiling, stats)).load(
        requested_ids, signal_dates[0], windows[-1].outcome_end, DAILY_SAMPLE_COLUMNS)
    capture("daily_prices", daily_prices)
    analysis_prices,session_audit=exact_session_quotes(daily_prices.table,
        session_dates(calendar),
        start=signal_dates[0],end=windows[-1].outcome_end)
    panel = build_monthly_returns(
        analysis_prices, calendar, snapshots,
        start_month=start_month, end_month=end_month,
        requested_security_ids=requested_ids, mode=mode,identity_policy=identity_policy)

    totals = {field: sum(stage[field] for stage in stats.values())
              for field in _BUDGET_STATS.values()}
    # Real readers enforce their remaining budget before returning. This check
    # also prevents a future adapter from bypassing the cumulative contract.
    for limit, measured in _BUDGET_STATS.items():
        if totals[measured] > getattr(ceiling, limit):
            raise BudgetExceeded(f"Monthly stage exceeded {limit}; no partial sample returned")
    loading_provenance = {
        "stages": provenance, "source_pins": pins,
        "request": {"start_month": start_month, "end_month": end_month,
                    "universe": universe, "mode": mode, "sample_size": sample_size,
                    "calendar_basis": CALENDAR_POLICY,
                    "signal_dates": list(signal_dates), "requested_ids": list(requested_ids)},
        "calendar_reconciliation": reconciliation.provenance,
        "quote_session_audit":session_audit,
        "sample_selection": (
            "Deterministic first-signal eligible IDs; provisional smoke includes a clean control and flagged route when available. "
            "Engineering sample chosen before prices; no replacement after observing history/outcomes."),
        'identity_policy':identity_policy_provenance(identity_policy),
        "daily_preservation": "Unchanged LoadedTable from the bounded daily OHLCV/action/quality read.",
        "stage_budget": asdict(ceiling),
        "budget_semantics": (
            "Cumulative counts across calendar, exact-date membership, bridge, links and prices; "
            "compressed projected bytes do not certify RAM or total I/O."),
        "holdout_policy": (
            "Metadata stages contain no financial values. The daily reader rejects protected "
            "outcome dates and masks excluded rows in Arrow before output/analysis; selected "
            "Parquet blocks may transiently decode excluded values."),
    }
    loading_provenance["request_sha256"] = canonical_sha256(loading_provenance)
    return LoadedMonthlySample(daily_prices, panel, snapshots, calendar, source_calendar,
                               requested_ids, {"stages": stats, "total": totals}, loading_provenance)
