"""Independent signal-only engineering read; never consumes monthly outcomes."""
from dataclasses import asdict, dataclass, replace

import pyarrow as pa
import pyarrow.compute as pc

from .catalog import canonical_sha256
from .data import BRIDGE_COLUMNS, CALENDAR_COLUMNS, LINK_COLUMNS, MEMBERSHIP_COLUMNS, session_dates
from .exchange_sessions import reconcile_exchange_calendar
from .loading import BoundedTableReader, BudgetExceeded, PriceLoader, ReadBudget
from .monthly import build_month_windows
from .quote_sessions import exact_session_quotes
from .monthly_loading import DAILY_SAMPLE_COLUMNS, _BUDGET_STATS, _preflight, _remaining_budget
from .universe import DataContractError, build_snapshot
from .identity_policy import (STRICT_IDENTITY_POLICY,validate_identity_policy,
                             select_engineering_ids,identity_policy_provenance)


@dataclass(frozen=True)
class LoadedFeatureSample:
    daily_prices: object
    snapshots: tuple
    calendar: pa.Table
    source_calendar: pa.Table
    requested_ids: tuple[str, ...]
    signal_dates: tuple[str, ...]
    loading_stats: dict
    loading_provenance: dict


def load_feature_sample(catalog, *, start_month, end_month, universe, sample_size=2,
                        history_sessions=503, budget=None, mode="historical_reconstruction",
                        identity_policy=STRICT_IDENTITY_POLICY):
    """Load <=3 preceding signal month-ends and <=10 declared sample IDs.

    Month arguments align with outcome month roles, but no outcomes or their
    endpoints are loaded. Selection uses only the first signal cohort. Numeric
    history ends at the last signal; later calendar rows are metadata evidence.
    Full context/feature panels require a separately measured execution plan.
    """
    validate_identity_policy(identity_policy)
    ceiling = _preflight(start_month, end_month, universe, sample_size, mode, budget)
    if type(history_sessions) is not int or history_sessions != 503:
        raise DataContractError("The registered feature bank requires an explicit 503-session engineering warmup")
    reader = BoundedTableReader(catalog, ceiling)
    stats, provenance = {}, {}

    def capture(name, loaded):
        stats[name], provenance[name] = loaded.stats, loaded.provenance
        return loaded.table

    source_calendar = capture("calendar", reader.read_metadata(
        "siblis", "trading_calendar_sessions", columns=CALENDAR_COLUMNS,
        allowed_columns=CALENDAR_COLUMNS, max_rows=20_000, key_columns=("trading_date",)))
    reconciliation = reconcile_exchange_calendar(
        source_calendar, start_month=start_month, end_month=end_month,
        warmup_sessions=history_sessions)
    calendar = reconciliation.table
    windows = build_month_windows(calendar, start_month=start_month, end_month=end_month)
    signals = tuple(window.signal_date for window in windows)
    if not signals:
        raise DataContractError("Calendar produced no feature signals")
    sessions = session_dates(calendar)
    first_position = sessions.index(signals[0])
    if first_position + 1 < history_sessions:
        raise DataContractError("Independent calendar cannot supply the complete feature warmup")
    raw_start, raw_end = sessions[first_position - history_sessions + 1], signals[-1]
    indices = ("sp500", "sp400") if universe == "combined" else (universe,)
    reader.budget = _remaining_budget(ceiling, stats)
    membership = capture("membership", reader.read(
        "siblis", "membership_by_index_by_date", start=signals[0], end=signals[-1],
        columns=MEMBERSHIP_COLUMNS, allowed_columns=MEMBERSHIP_COLUMNS,
        key_columns=("index_id", "security_id", "trading_date"),
        filters={"index_id": indices, "trading_date": signals}, financial=False))
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
    pins, snapshots = catalog.provenance(), []
    for day in signals:
        dated = membership.filter(pc.equal(membership["trading_date"], day))
        if not dated.num_rows:
            raise DataContractError(f"No source membership observations for feature signal {day}")
        snapshot = build_snapshot(dated, bridge, links, as_of=day, universe=universe, mode=mode)
        if not snapshot.table.num_rows:
            raise DataContractError(f"No qualifying historical cohort at {day}")
        snapshots.append(replace(snapshot, provenance={**snapshot.provenance, "source_pins": pins}))
    ids=select_engineering_ids(snapshots[0],sample_size,identity_policy=identity_policy)
    if not ids:
        raise DataContractError("First signal cohort has no resolved engineering price IDs")
    prices = PriceLoader(catalog, _remaining_budget(ceiling, stats)).load(
        ids, raw_start, raw_end, DAILY_SAMPLE_COLUMNS)
    capture("feature_daily_history", prices)
    analysis_table,session_audit=exact_session_quotes(prices.table,sessions,start=raw_start,end=raw_end)
    prices=replace(prices,table=analysis_table,provenance={**prices.provenance,'exchange_session_partition':session_audit})
    totals = {field: sum(s[field] for s in stats.values()) for field in _BUDGET_STATS.values()}
    for limit, measured in _BUDGET_STATS.items():
        if totals[measured] > getattr(ceiling, limit):
            raise BudgetExceeded(f"Feature stage exceeded {limit}; no partial sample returned")
    loading_provenance = {
        "stages": provenance, "source_pins": pins, "stage_budget": asdict(ceiling),
        "request": {"start_month": start_month, "end_month": end_month, "universe": universe,
                    "mode": mode, "sample_size": sample_size, "signal_dates": list(signals),
                    "requested_ids": list(ids), "raw_start": raw_start, "raw_end": raw_end,
                    "history_sessions": history_sessions},
        "calendar_reconciliation": reconciliation.provenance,
        "quote_session_audit":session_audit,
        "sample_selection": "Deterministic first-signal eligible IDs; provisional smoke includes clean/flagged routes when available before prices, without availability-based replacements.",
        'identity_policy':identity_policy_provenance(identity_policy),
        "financial_cutoff": raw_end, "outcome_tables_consumed": False,
        "daily_context_loaded": False,
        "context_policy": "Withhold cohort-derived statistics until every original required member/input is available; no sample ranks.",
        "knowledge_time_verified": False,
        "holdout_policy": "Financial reads end before 2024; Arrow masks exclusions before values reach Python/analysis. Parquet chunks may transiently decode excluded values.",
    }
    loading_provenance["request_sha256"] = canonical_sha256(loading_provenance)
    return LoadedFeatureSample(prices, tuple(snapshots), calendar, source_calendar, ids,
                               signals, {"stages": stats, "total": totals}, loading_provenance)
