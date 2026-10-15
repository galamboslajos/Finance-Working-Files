"""Explicit, bounded annual sources for the authorized 2010--2023 panel.

Planning reads metadata and Parquet footers only. Annual materialization keeps
the unchanged outcome/daily quote read separate from a past-only feature view.
Neither source-ID membership nor provisional identity admission consult prices.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import pyarrow as pa
import pyarrow.compute as pc

from .catalog import canonical_sha256
from .data import BRIDGE_COLUMNS, CALENDAR_COLUMNS, LINK_COLUMNS, MEMBERSHIP_COLUMNS, session_dates
from .exchange_sessions import reconcile_exchange_calendar
from .identity_policy import (STRICT_IDENTITY_POLICY, identity_policy_provenance,
                              qualify_identity_records, validate_identity_policy)
from .loading import (BoundedTableReader, BudgetExceeded, LoadedTable, PriceLoader,
                      ProtectedDataError, ReadBudget, FINANCIAL_COLUMNS, HOLDOUT_START, _stats_bounds,
                      parse_date)
from .monthly import build_month_windows
from .monthly_loading import DAILY_SAMPLE_COLUMNS, _BUDGET_STATS, _remaining_budget
from .quote_sessions import exact_session_quotes
from .universe import DataContractError, KnowledgeTimeUnavailable, build_snapshot


METADATA_BUDGET = ReadBudget(max_metadata_files=128, max_data_files=8,
    max_scanned_rows=16_000_000, max_output_rows=250_000, max_read_bytes=256*1024*1024)
MEMBERSHIP_BUDGET = ReadBudget(max_metadata_files=8, max_data_files=4,
    max_scanned_rows=16_000_000, max_output_rows=1_000_000, max_read_bytes=128*1024*1024)
PRICE_BUDGET = ReadBudget(max_metadata_files=128, max_data_files=105,
    max_scanned_rows=27_000_000, max_output_rows=2_000_000, max_read_bytes=2*1024*1024*1024)


@dataclass(frozen=True)
class FullFeatureSourcePlan:
    catalog: object
    years: tuple[int, ...]
    windows: tuple
    snapshots: tuple
    calendar: pa.Table
    source_calendar: pa.Table
    bridge: pa.Table
    links: pa.Table
    year_requests: dict
    loading_stats: dict
    provenance: dict
    identity_policy: str
    universe: str


@dataclass(frozen=True)
class LoadedFeatureYear:
    raw_daily_prices: LoadedTable
    feature_daily_prices: LoadedTable
    snapshots: tuple
    daily_cohorts: pa.Table
    calendar: pa.Table
    source_calendar: pa.Table
    requested_ids: tuple[str, ...]
    signal_dates: tuple[str, ...]
    loading_stats: dict
    loading_provenance: dict
    daily_membership: LoadedTable
    year: int


@dataclass(frozen=True)
class PreparedFeatureYear:
    source_plan_sha256: str
    year: int
    daily_membership: LoadedTable
    daily_cohorts: pa.Table
    snapshots: tuple
    signal_dates: tuple[str, ...]
    requested_ids: tuple[str, ...]
    context_audit: dict
    exact_price_estimate: dict


def _validate_scope(start_year, end_year, universe, history_sessions, mode, identity_policy):
    validate_identity_policy(identity_policy)
    if mode == "knowledge_time":
        raise KnowledgeTimeUnavailable("The pinned inputs cannot certify knowledge-time PIT")
    if mode != "historical_reconstruction":
        raise DataContractError("Unknown full-panel reconstruction mode")
    if (type(start_year) is not int or type(end_year) is not int or
            start_year < 2010 or start_year > end_year):
        raise DataContractError("Full-panel outcome years must start in 2010 or later and be ordered")
    if end_year >= 2024:
        raise ProtectedDataError("Full-panel financial outcomes must end before protected 2024")
    if universe not in ("sp500", "sp400", "combined"):
        raise DataContractError("Full-panel universe must be sp500, sp400 or combined")
    if type(history_sessions) is not int or history_sessions != 503:
        raise DataContractError("Full feature history requires exactly 503 warmup sessions")


def _total(stats):
    return {field: sum(stage[field] for stage in stats.values())
            for field in _BUDGET_STATS.values()}


def estimate_bounded_read(catalog, source, table, *, start, end, columns,
                          allowed_columns, filters=None, budget=None):
    """Footer-only conservative projection estimate, never column values.

    Candidate rows and bytes describe complete selected Parquet chunks. The
    returned-row upper bound is not a prediction of actual matching rows.
    A footer lacks within-chunk date/ID intersections; output limits remain
    enforced during the actual Arrow mask/read.
    """
    lower, upper = parse_date(start), parse_date(end)
    if lower > upper:
        raise DataContractError("Estimate start is later than end")
    if source == "intrinio" and table == "intrinio_daily_prices" and upper >= HOLDOUT_START:
        raise ProtectedDataError("Financial estimate enters protected 2024")
    ceiling = PRICE_BUDGET if budget is None else budget
    if not isinstance(ceiling, ReadBudget):
        raise DataContractError("Estimate budget must be a ReadBudget")
    filtered = filters or {}
    if set(filtered) & FINANCIAL_COLUMNS:
        raise DataContractError("Financial values cannot serve as estimate metadata filters")
    for values in filtered.values():
        if isinstance(values, (str, bytes)) or not values or any(v is None for v in values):
            raise DataContractError("Estimate filters require explicit nonempty sequences")
    reader = BoundedTableReader(catalog, ceiling)
    schema, footers = reader._footers(source, table)
    keys = ("intrinio_security_id", "trading_date") if source == "intrinio" else ("index_id", "security_id", "trading_date")
    extras = tuple(dict.fromkeys(("trading_date", *keys, *sorted(filtered))))
    projected = reader._request(schema, columns, allowed_columns, extras)
    groups = []
    for obj, metadata in footers:
        for index in range(metadata.num_row_groups):
            group = metadata.row_group(index)
            bounds = _stats_bounds(group, "trading_date")
            if bounds is not None and bounds.null_count:
                raise DataContractError("Null candidate dates in footer estimate")
            if bounds is not None and bounds.has_min_max:
                if parse_date(bounds.max) < lower or parse_date(bounds.min) > upper:
                    continue
            skip = False
            for name, values in filtered.items():
                if isinstance(values, (str, bytes)) or not values:
                    raise DataContractError("Estimate filters require explicit nonempty sequences")
                statistics = _stats_bounds(group, name)
                if statistics is not None and statistics.has_min_max:
                    skip = all(value < statistics.min or value > statistics.max for value in values)
                    if skip:
                        break
            if not skip:
                groups.append((obj, index, group))
    files, rows, size = reader._budget(groups, projected)
    return {"metadata_files": len(footers), "data_files": files,
        "scanned_rows": rows, "returned_rows_upper_bound": rows,
        "projected_compressed_bytes": size,
        "selected_row_groups": len(groups), "financial_values_read": False,
        "request": {"source": source, "table": table, "start": start, "end": end,
                    "columns": list(projected), "filters": {k: list(v) for k,v in filtered.items()}},
        "budget": asdict(ceiling),
        "semantics": "Whole candidate column chunks; not RAM, disk I/O or actual masked output rows."}


def _calendar(source_calendar, start_year, end_year):
    # Retain the reviewed <=3-month reconciler rather than relaxing its sample
    # guard. Overlapping independent metadata must be byte-identical by date.
    rows, audits, schema = {}, [], None
    for year in range(start_year, end_year + 1):
        for first in (1, 4, 7, 10):
            reconciliation = reconcile_exchange_calendar(source_calendar,
                start_month=f"{year:04d}-{first:02d}", end_month=f"{year:04d}-{first+2:02d}",
                **({"warmup_sessions": 503} if year == start_year and first == 1 else {}))
            schema = reconciliation.table.schema
            for row in reconciliation.table.to_pylist():
                day = row["trading_date"]
                if day in rows and rows[day] != row:
                    raise DataContractError("Independent calendar partitions disagree on a session")
                rows[day] = row
            audits.append(reconciliation.provenance)
    calendar = pa.Table.from_pylist([rows[d] for d in sorted(rows)], schema=schema)
    return calendar, {"partitions": audits, "partition_count": len(audits),
        "session_dates_sha256": canonical_sha256(list(session_dates(calendar))),
        "selection_independence": "No quote values, IDs or availability used to generate sessions."}


def plan_full_feature_sources(catalog, *, start_year=2010, end_year=2023,
        universe="combined", history_sessions=503,
        identity_policy=STRICT_IDENTITY_POLICY, metadata_budget=None,
        mode="historical_reconstruction"):
    """Read original monthly metadata and plan annual bounded financial reads."""
    _validate_scope(start_year, end_year, universe, history_sessions, mode, identity_policy)
    ceiling = METADATA_BUDGET if metadata_budget is None else metadata_budget
    if not isinstance(ceiling, ReadBudget):
        raise DataContractError("metadata_budget must be a ReadBudget")
    reader = BoundedTableReader(catalog, ceiling)
    stats, provenance = {}, {}
    def capture(name, loaded):
        stats[name], provenance[name] = loaded.stats, loaded.provenance
        return loaded.table
    source_calendar = capture("calendar", reader.read_metadata("siblis", "trading_calendar_sessions",
        columns=CALENDAR_COLUMNS, allowed_columns=CALENDAR_COLUMNS, max_rows=20_000,
        key_columns=("trading_date",)))
    calendar, reconciliation = _calendar(source_calendar, start_year, end_year)
    windows = build_month_windows(calendar, start_month=f"{start_year}-01", end_month=f"{end_year}-12")
    signals = tuple(w.signal_date for w in windows)
    indices = ("sp500", "sp400") if universe == "combined" else (universe,)
    reader.budget = _remaining_budget(ceiling, stats)
    membership = capture("signal_membership", reader.read("siblis", "membership_by_index_by_date",
        start=signals[0], end=signals[-1], columns=MEMBERSHIP_COLUMNS,
        allowed_columns=MEMBERSHIP_COLUMNS, key_columns=("index_id", "security_id", "trading_date"),
        filters={"index_id": indices, "trading_date": signals}, financial=False))
    reader.budget = _remaining_budget(ceiling, stats)
    bridge = capture("bridge", reader.read_metadata("identity", "security_identity_bridge",
        columns=BRIDGE_COLUMNS, allowed_columns=BRIDGE_COLUMNS, max_rows=30_000,
        key_columns=("source_product_id", "source_security_id")))
    reader.budget = _remaining_budget(ceiling, stats)
    links = capture("links", reader.read_metadata("intrinio", "intrinio_identity_links",
        columns=LINK_COLUMNS, allowed_columns=LINK_COLUMNS, max_rows=20_000,
        key_columns=("unified_security_id",)))
    pins, snapshots = catalog.provenance(), []
    for day in signals:
        dated = membership.filter(pc.equal(membership["trading_date"], day))
        snapshot = build_snapshot(dated, bridge, links, as_of=day, universe=universe, mode=mode)
        if not snapshot.table.num_rows:
            raise DataContractError(f"Missing original monthly cohort after membership clocks: {day}")
        snapshots.append(replace(snapshot, provenance={**snapshot.provenance, "source_pins": pins}))
    sessions, requests = session_dates(calendar), {}
    broad_ids = tuple(sorted({r["intrinio_security_id"] for r in links.to_pylist()
                             if isinstance(r["intrinio_security_id"], str) and r["intrinio_security_id"].strip()}))
    for year in range(start_year, end_year+1):
        selected = tuple(w for w in windows if w.outcome_month.startswith(str(year)))
        first = sessions.index(selected[0].signal_date)
        if first < history_sessions-1:
            raise DataContractError("Independent calendar lacks the exact full feature warmup")
        request = {"year": year, "start_month": f"{year}-01", "end_month": f"{year}-12",
            "signal_dates": [w.signal_date for w in selected],
            "raw_start": sessions[first-history_sessions+1],
            "feature_end": selected[-1].signal_date, "raw_end": selected[-1].outcome_end,
            "daily_context_start": sessions[first-251],
            "history_sessions": history_sessions,
            "daily_context_policy": "Original daily membership, 252 return observations before first signal inclusive; no fill.",
            "price_estimate_population": "Conservative every nonblank native link ID; exact original admitted IDs set before annual prices."}
        # This is an upper envelope, not a selection/admission population.
        request["price_estimate"] = estimate_bounded_read(catalog, "intrinio", "intrinio_daily_prices",
            start=request["raw_start"], end=request["raw_end"], columns=DAILY_SAMPLE_COLUMNS,
            allowed_columns=DAILY_SAMPLE_COLUMNS, filters={"intrinio_security_id": broad_ids})
        request["membership_estimate"] = estimate_bounded_read(catalog, "siblis", "membership_by_index_by_date",
            start=request["daily_context_start"], end=request["feature_end"], columns=MEMBERSHIP_COLUMNS,
            allowed_columns=MEMBERSHIP_COLUMNS, filters={"index_id": indices}, budget=MEMBERSHIP_BUDGET)
        requests[year] = request
    plan_provenance = {"stages": provenance, "source_pins": pins,
        "calendar_reconciliation": reconciliation, "start_year": start_year, "end_year": end_year,
        "universe": universe, "history_sessions": history_sessions, "year_requests": requests,
        "monthly_original_rows": sum(s.table.num_rows for s in snapshots),
        "signal_dates": list(signals), "financial_values_read": False,
        "mode": mode, "stage_budget": asdict(ceiling), **identity_policy_provenance(identity_policy)}
    plan_provenance["request_sha256"] = canonical_sha256(plan_provenance)
    return FullFeatureSourcePlan(catalog, tuple(range(start_year,end_year+1)), windows,
        tuple(snapshots), calendar, source_calendar, bridge, links, requests,
        {"stages": stats, "total": _total(stats)}, plan_provenance, identity_policy, universe)


_DAILY_SCHEMA = pa.schema([
    ("source_security_id", pa.string()), ("signal_date", pa.string()),
    ("indices", pa.list_(pa.string())), ("unified_security_id", pa.string()),
    ("intrinio_security_id", pa.string()), ("identity_status", pa.string()),
    ("identity_review_required", pa.bool_()), ("resolution_status", pa.string()),
    ("membership_effective_date", pa.string()), ("membership_availability_date", pa.string()),
    ("membership_end_date", pa.string()), ("membership_record_count", pa.int64()),
])


def _daily_cohorts(membership, bridge, links, dates, universe, mode):
    """Use the original snapshot rule on each observed date; no carry forward."""
    # Pruning is metadata-only and retains every candidate route/link for the
    # observed source IDs, including conflicting rows. It changes no rule.
    source_ids = pc.unique(membership["security_id"])
    bridge = bridge.filter(pc.is_in(bridge["source_security_id"], value_set=source_ids))
    uids = pc.unique(bridge["unified_security_id"])
    links = links.filter(pc.is_in(links["unified_security_id"], value_set=uids))
    rows, excluded, missing_dates = [], [], []
    for day in dates:
        dated = membership.filter(pc.equal(membership["trading_date"], day))
        if not dated.num_rows:
            missing_dates.append(day)
            continue
        snapshot = build_snapshot(dated, bridge, links, as_of=day, universe=universe, mode=mode)
        excluded.extend(snapshot.excluded.to_pylist())
        source_records = {}
        for row in dated.to_pylist():
            source_records.setdefault(row["security_id"], []).append(row)
        for row in snapshot.table.to_pylist():
            # For overlap rows all clocks have already passed individually.
            # Keep conservative clock aggregates and the original index list;
            # the context worker blocks ambiguous index sleeves explicitly.
            records = [r for r in source_records[row["source_security_id"]]
                       if r["index_id"] in row["indices"]]
            ends = [r["membership_end_date"] for r in records if r["membership_end_date"] not in (None, "")]
            rows.append({"source_security_id": row["source_security_id"], "signal_date": day,
                "indices": row["indices"], "unified_security_id": row["unified_security_id"],
                "intrinio_security_id": row["intrinio_security_id"], "identity_status": row["identity_status"],
                "identity_review_required": row["identity_review_required"], "resolution_status": row["resolution_status"],
                "membership_effective_date": max(r["membership_effective_date"] for r in records),
                "membership_availability_date": max(r["membership_availability_date"] for r in records),
                "membership_end_date": min(ends) if ends else None,
                "membership_record_count": len(records)})
    return pa.Table.from_pylist(rows, schema=_DAILY_SCHEMA), {
        "expected_dates": len(dates), "observed_dates": len(dates)-len(missing_dates),
        "missing_dates": missing_dates, "membership_clock_exclusions": excluded,
        "original_daily_rows": len(rows), "membership_filled": False,
        "clock_aggregation": "Single-record clocks unchanged. Overlap: maximum effective/availability, minimum nonnull inclusive end; all original index memberships retained.",
        "underlying_evidence": "Unchanged bounded daily_membership LoadedTable retained separately."}


def annual_feature_calendar(plan, year):
    """Independent exact warmup through next-month boundary evidence only."""
    if not isinstance(plan, FullFeatureSourcePlan) or type(year) is not int or year not in plan.years:
        raise DataContractError("Calendar year must belong to the explicit source plan")
    request = plan.year_requests[year]
    return plan.calendar.filter(pc.and_(
        pc.greater_equal(plan.calendar["trading_date"], request["raw_start"]),
        pc.less_equal(plan.calendar["trading_date"], f"{year+1}-01-31")))


def prepare_feature_year(plan, year, *, membership_budget=None):
    """Actual annual original metadata and exact price footer estimate; no prices."""
    if not isinstance(plan, FullFeatureSourcePlan) or type(year) is not int or year not in plan.years:
        raise DataContractError("Feature year must belong to the explicit source plan")
    mb = MEMBERSHIP_BUDGET if membership_budget is None else membership_budget
    if not isinstance(mb, ReadBudget):
        raise DataContractError("Annual membership budget must be a ReadBudget")
    request = plan.year_requests[year]
    signal_dates = tuple(request["signal_dates"])
    snapshots = tuple(s for s in plan.snapshots if s.as_of in signal_dates)
    annual_calendar = annual_feature_calendar(plan, year)
    sessions = session_dates(annual_calendar)
    context_dates = tuple(d for d in sessions if request["daily_context_start"] <= d <= request["feature_end"])
    indices = ("sp500", "sp400") if plan.universe == "combined" else (plan.universe,)
    membership = BoundedTableReader(plan.catalog, mb).read("siblis", "membership_by_index_by_date",
        start=request["daily_context_start"], end=request["feature_end"], columns=MEMBERSHIP_COLUMNS,
        allowed_columns=MEMBERSHIP_COLUMNS, key_columns=("index_id", "security_id", "trading_date"),
        filters={"index_id": indices, "trading_date": context_dates}, financial=False)
    daily_cohorts, context_audit = _daily_cohorts(membership.table, plan.bridge, plan.links,
        context_dates, plan.universe, "historical_reconstruction")
    # Per-date qualification preserves collisions against ALL original members;
    # rows are not pooled across dates before collision checks.
    ids = set()
    for snapshot in snapshots:
        ids.update(r["intrinio_security_id"] for r in qualify_identity_records(snapshot.table.to_pylist(),
            identity_policy=plan.identity_policy) if r["identity_eligible"])
    for day in context_dates:
        dated = daily_cohorts.filter(pc.equal(daily_cohorts["signal_date"], day))
        ids.update(r["intrinio_security_id"] for r in qualify_identity_records(dated.to_pylist(),
            identity_policy=plan.identity_policy) if r["identity_eligible"])
    ids = tuple(sorted(ids))
    if not ids:
        raise DataContractError("Annual original cohorts contain no admitted native price IDs")
    estimate = estimate_bounded_read(plan.catalog, "intrinio", "intrinio_daily_prices",
        start=request["raw_start"], end=request["raw_end"], columns=DAILY_SAMPLE_COLUMNS,
        allowed_columns=DAILY_SAMPLE_COLUMNS, filters={"intrinio_security_id": ids})
    return PreparedFeatureYear(plan.provenance["request_sha256"], year, membership,
        daily_cohorts, snapshots, signal_dates, ids, context_audit, estimate)


def load_feature_year(plan, year, *, membership_budget=None, price_budget=None,
                      include_outcome_daily=True, prepared=None):
    """Materialize one declared annual block; prices never choose its members.

    Raw quotes may include completed outcome months for later daily backtesting.
    The feature view is independently masked in Arrow at the final signal before
    any financial conversion and the session filter is metadata-only.
    """
    if not isinstance(plan, FullFeatureSourcePlan) or type(year) is not int or year not in plan.years:
        raise DataContractError("Feature year must belong to the explicit source plan")
    if type(include_outcome_daily) is not bool:
        raise DataContractError("include_outcome_daily must be boolean")
    mb = MEMBERSHIP_BUDGET if membership_budget is None else membership_budget
    pb = PRICE_BUDGET if price_budget is None else price_budget
    if not isinstance(mb, ReadBudget) or not isinstance(pb, ReadBudget):
        raise DataContractError("Annual source budgets must be ReadBudget objects")
    if prepared is None:
        prepared = prepare_feature_year(plan, year, membership_budget=mb)
    elif (not isinstance(prepared, PreparedFeatureYear) or prepared.year != year or
            prepared.source_plan_sha256 != plan.provenance["request_sha256"]):
        raise DataContractError("Prepared annual metadata differs from the requested pinned source plan")
    request = plan.year_requests[year]
    membership, daily_cohorts = prepared.daily_membership, prepared.daily_cohorts
    snapshots, signal_dates = prepared.snapshots, prepared.signal_dates
    ids, context_audit = prepared.requested_ids, prepared.context_audit
    annual_calendar = annual_feature_calendar(plan, year)
    sessions = session_dates(annual_calendar)
    raw_end = request["raw_end"] if include_outcome_daily else request["feature_end"]
    exact_estimate = estimate_bounded_read(plan.catalog, "intrinio", "intrinio_daily_prices",
        start=request["raw_start"], end=raw_end, columns=DAILY_SAMPLE_COLUMNS,
        allowed_columns=DAILY_SAMPLE_COLUMNS, filters={"intrinio_security_id": ids}, budget=pb)
    raw = PriceLoader(plan.catalog, pb).load(ids, request["raw_start"], raw_end, DAILY_SAMPLE_COLUMNS)
    # String ISO keys only; protected/future financial scalars are never examined.
    feature_table = raw.table.filter(pc.less_equal(raw.table["trading_date"], request["feature_end"]))
    feature_table, quote_audit = exact_session_quotes(feature_table, sessions,
        start=request["raw_start"], end=request["feature_end"])
    features = replace(raw, table=feature_table, provenance={**raw.provenance,
        "feature_cutoff": request["feature_end"], "outcome_rows_consumed": False,
        "exchange_session_partition": quote_audit,
        "raw_rows": raw.table.num_rows, "feature_rows": feature_table.num_rows})
    stats = {"daily_membership": membership.stats, "raw_daily_prices": raw.stats}
    provenance = {"year": year, "source_plan_sha256": plan.provenance["request_sha256"],
        "source_pins": plan.catalog.provenance(), "request": {**request,
            "raw_end": raw_end, "requested_ids": list(ids)},
        "stages": {"daily_membership": membership.provenance, "raw_daily_prices": raw.provenance},
        "exact_price_estimate": exact_estimate, "daily_context_audit": context_audit,
        "quote_session_audit": quote_audit, "feature_financial_cutoff": request["feature_end"],
        "raw_daily_financial_cutoff": raw_end, "future_quote_eligibility_used": False,
        "current_security_master_filter_used": False, "outcome_tables_consumed_by_features": False,
        "daily_preservation": "Unchanged LoadedTable; financial feature view is a separate past-only Arrow partition.",
        "mode": "historical_reconstruction", **identity_policy_provenance(plan.identity_policy)}
    provenance["request_sha256"] = canonical_sha256(provenance)
    return LoadedFeatureYear(raw, features, snapshots, daily_cohorts, annual_calendar,
        plan.source_calendar, ids, signal_dates, {"stages": stats, "total": _total(stats)},
        provenance, membership, year)
