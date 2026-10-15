"""Bounded full-cohort feature assembly using the frozen numerical workers.

The optimized paths change storage and dependency counting, not formulas,
timing, identity admission, feature selection or the explicitly selected cohort policy. They accept an
explicit annual source partition and never read source files or outcomes.
"""
from __future__ import annotations

from collections import Counter
import math

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from .catalog import canonical_sha256
from .context_features import (
    ContextFeaturesResult, MARKET_HORIZONS, RELATIVE_WINDOWS, SLEEVES,
    _cohorts, _coverage_row, _dates, _frame, _market, _validate_signal_context_inventory,
    _validated_price_arrays, _xs, balanced_cross_sectional_bases,
    _coverage_gate, _request_diagnostics, validate_context_policy, context_coverage_fraction,
    context_feature_specs,
)
from .daily_features import (
    DailyFeaturesResult, HOLDOUT_START, PRICE_FIELDS, SOURCE_COMMIT, _definitions, _numeric,
    _sequence, _technical_series, _validate_input, daily_feature_specs,
)
from .data import session_dates as calendar_session_dates
from .features import FeaturePanel, KEYS
from .identity_policy import (STRICT_IDENTITY_POLICY, identity_policy_provenance,
                              validate_identity_policy)
from .paper_features import COMMON_MOMENTUM_POLICY, compute_paper_features
from .universe import DataContractError, KnowledgeTimeUnavailable, iso_date


DAILY_STATUS_LEGEND = {
    0: "ok", 1: "insufficient_calendar_warmup", 2: "missing_session",
    3: "invalid_quote_metadata", 4: "missing_or_invalid_input",
    5: "undefined_formula", 6: "identity_unresolved", 7: "not_requested",
}


def _segments(spec):
    """Disjoint lag intervals with identical required fields, exactly once."""
    boundaries = sorted({v for _, lo, hi in spec["_spans"] for v in (lo, hi + 1)})
    return tuple((lo, end - 1, tuple(sorted({field for field, a, b in spec["_spans"]
                                           if a <= lo <= b})))
                 for lo, end in zip(boundaries, boundaries[1:])
                 if any(a <= lo <= b for _, a, b in spec["_spans"]))


def _prefix(mask):
    return np.r_[0, np.cumsum(mask, dtype=np.int64)]


def _count(prefix, indexes, lo, hi):
    start = np.maximum(indexes - hi, 0)
    end = np.maximum(indexes - lo + 1, 0)
    return prefix[end] - prefix[start]


def _native_inputs(stock, quality):
    # Identical native-value and OHLC masks to the security-local worker.
    native = {}
    for name in PRICE_FIELDS:
        values = stock[name] if name in stock else pd.Series(np.nan, index=stock.index)
        native[name] = _numeric(values, positive=name not in ("volume", "dividend", "split_ratio"),
                                nonnegative=name in ("volume", "split_ratio")).where(quality)
    for prefix in ("adj_", ""):
        high_name, low_name = prefix + "high", prefix + "low"
        high, low = native[high_name], native[low_name]
        bounds = high.notna() & low.notna() & (high < low)
        if prefix:
            bad_open = pd.Series(False, index=stock.index)
            for field in ("adj_open", "adj_close"):
                value = native[field]
                bounds |= high.notna() & value.notna() & (high < value)
                bounds |= low.notna() & value.notna() & (low > value)
                if field == "adj_open":
                    bad_open |= high.notna() & value.notna() & (high < value)
                    bad_open |= low.notna() & value.notna() & (low > value)
            if "adjusted_ohlc_bounds_valid" in stock:
                flagged = stock["adjusted_ohlc_bounds_valid"].map(
                    lambda x: isinstance(x, (bool, np.bool_)) and not x)
                bounds |= flagged
                bad_open |= flagged
            native["adj_open"] = native["adj_open"].mask(bad_open)
        native[high_name] = high.mask(bounds)
        native[low_name] = low.mask(bounds)
    return native


def compute_compact_daily_features(daily_prices, calendar_sessions, *, signal_dates,
                                  requested_security_ids):
    """Frozen daily arithmetic with prefix-counted exact dependencies.

    ``coverage`` is a compact uint8 status matrix keyed by native ID/signal,
    rather than one Python dictionary for every ID/signal/feature. Statuses
    have the same precedence as ``compute_daily_features``. No missing input
    is converted into a successful formula or forward-filled.
    """
    sessions = tuple(iso_date(day, "session date") for day in
                     _sequence(calendar_sessions, "session_dates"))
    if sessions != tuple(sorted(sessions)):
        raise DataContractError("session_dates must be in increasing exact-session order")
    signals = tuple(iso_date(day, "signal date") for day in _sequence(signal_dates, "signal_dates"))
    if any(day >= HOLDOUT_START or day not in set(sessions) for day in signals):
        raise DataContractError("Signals must be supplied pre-holdout exchange sessions")
    requested = _sequence(requested_security_ids, "requested_security_ids")
    if any(not isinstance(sid, str) or not sid.strip() for sid in requested):
        raise DataContractError("Invalid requested security ID")
    # All date/ID/prohibited-column checks run before financial conversion.
    frame = _validate_input(daily_prices, sessions, signals, requested)
    grid = tuple(day for day in sessions if day <= max(signals))
    position = {day: i for i, day in enumerate(grid)}
    indexes = np.asarray([position[s] for s in signals], dtype=np.int64)
    definitions = _definitions()
    names = [s["name"] for s in definitions]
    spans = [_segments(s) for s in definitions]
    # One grouped index avoids rescanning all annual prices for every native ID.
    groups = {sid: rows for sid, rows in frame.groupby("intrinio_security_id", sort=False)}
    empty = frame.iloc[:0].copy()
    outputs, code_frames = [], []
    for sid in requested:
        stock = groups.get(sid, empty).set_index("trading_date").reindex(grid)
        present = stock["intrinio_security_id"].notna().to_numpy()
        quality = ((stock["frequency"] == "daily") & stock["intraperiod"].map(
            lambda v: isinstance(v, (bool, np.bool_)) and not v)
            & stock["price_quality_status"].isin(("accepted", "ok"))).to_numpy(dtype=bool)
        native = _native_inputs(stock, quality)
        series = _technical_series(native)  # Shared frozen numerical definitions.
        values = np.column_stack([series[name].iloc[indexes].to_numpy() for name in names])
        codes = np.zeros(values.shape, dtype=np.uint8)
        absent, bad_quality = _prefix(~present), _prefix(~quality)
        field_valid = {name: value.notna().to_numpy() for name, value in native.items()}
        invalid_by_fields = {}
        for j, intervals in enumerate(spans):
            missing = np.zeros(len(signals), dtype=np.int64)
            invalid_metadata = np.zeros(len(signals), dtype=np.int64)
            bad_input = np.zeros(len(signals), dtype=np.int64)
            for lo, hi, fields in intervals:
                missing += _count(absent, indexes, lo, hi)
                invalid_metadata += _count(bad_quality, indexes, lo, hi)
                if fields not in invalid_by_fields:
                    invalid_by_fields[fields] = _prefix(~np.logical_and.reduce(
                        [field_valid[field] for field in fields]))
                bad_input += _count(invalid_by_fields[fields], indexes, lo, hi)
            codes[~np.isfinite(values[:, j]), j] = 5
            codes[bad_input > 0, j] = 4
            codes[invalid_metadata > 0, j] = 3
            codes[missing > 0, j] = 2
            codes[indexes < max(hi for _, hi, _ in intervals), j] = 1
        values[codes != 0] = np.nan
        keys = pd.DataFrame({"intrinio_security_id": sid, "signal_date": signals})
        outputs.append(pd.concat([keys, pd.DataFrame(values, columns=names)], axis=1))
        code_frames.append(pd.concat([keys, pd.DataFrame(codes, columns=names)], axis=1))
    specs = daily_feature_specs()
    return DailyFeaturesResult(pd.concat(outputs, ignore_index=True),
        pd.concat(code_frames, ignore_index=True), specs,
        {"source_commit": SOURCE_COMMIT, "feature_count": len(specs),
         "window_policy": "STRICT_COMPLETE_EXACT_SESSION_DEPENDENCIES_V1",
         "signal_dates": list(signals), "input_through": max(signals),
         "requested_security_ids": list(requested), "calendar_sessions": len(grid),
         "future_rows_rejected": True, "target_columns_rejected": True,
         "historical_reconstruction": True, "knowledge_time_verified": False,
         "adjustment_vintage_verified": False, "model_ready": False,
         "optimization": "frozen technical series; disjoint exact dependency prefix counts",
         "coverage_representation": "native_id_signal_uint8_status_matrix",
         "status_legend": DAILY_STATUS_LEGEND})


def _compact_daily_context(historical, arrays, sessions, requested, *,
                           context_policy="complete_cohort", coverage=None, through=None):
    """Daily original-cohort benchmarks with explicit request/coverage gates."""
    positions = {day: i for i, day in enumerate(sessions)}
    historical = historical.copy()
    historical["stock_return"] = np.nan
    for sid, indexes in historical.groupby("intrinio_security_id", sort=False).groups.items():
        if sid not in arrays or sid not in requested:
            continue
        rows = historical.loc[indexes]
        slots = np.asarray([positions[day] for day in rows.signal_date], dtype=np.int64)
        closes = arrays[sid]["adj_close"]
        usable = slots > 0
        returns = np.full(len(slots), np.nan)
        # Positive, finite native closes and accepted metadata are already masked.
        returns[usable] = closes[slots[usable]] / closes[slots[usable] - 1] - 1
        returns[~rows.identity_eligible.to_numpy(dtype=bool)] = np.nan
        historical.loc[indexes, "stock_return"] = returns
    market, sleeve = {}, {}
    groups = dict(tuple(historical.groupby("signal_date", sort=True)))
    through = historical.signal_date.max() if through is None else through
    for day in (day for day in sessions if day <= through):
        original = groups.get(day)
        rows = historical.iloc[:0] if original is None else original
        for label in ("universe", *SLEEVES):
            members = rows if label == "universe" else rows.loc[rows.index_id == label]
            returns = members.stock_return.to_numpy(dtype=float)
            valid = np.isfinite(returns)
            request_info = _request_diagnostics(members, requested)
            status = ("context_missing_date" if original is None else
                      _coverage_gate(len(members), int(valid.sum()),
                                     request_info["request_exhaustive"], context_policy))
            value = returns[valid].mean() if status == "ok" else np.nan
            if label == "universe":
                market[day] = value
            else:
                sleeve[(day, label)] = value
            if coverage is not None:
                coverage.append(_coverage_row(day, "benchmark_daily_ew_return_1s", label,
                    len(members), int(valid.sum()), status,
                    context_policy=context_policy, request_info=None if original is None else request_info))
    return market, sleeve, historical


def _compact_relative(table, cohorts, market, sleeve, historical, sessions, coverage, *,
                      context_policy="complete_cohort"):
    # One compact calendar array per original source preserves every missing
    # membership, route change and sleeve switch. No snapshot is frozen backward.
    members = {}
    for sid, group in historical.groupby("source_security_id", sort=False):
        grid = group.set_index("signal_date").reindex(sessions)
        members[sid] = {"route": grid.intrinio_security_id.to_numpy(),
            "sleeve": grid.index_id.to_numpy(),
            "eligible": grid.identity_eligible.eq(True).to_numpy(dtype=bool),
            "return": grid.stock_return.to_numpy(dtype=float)}
    positions = {day: i for i, day in enumerate(sessions)}
    market_array = np.asarray([market.get(day, np.nan) for day in sessions])
    sleeve_arrays = {index: np.asarray([sleeve.get((day, index), np.nan) for day in sessions])
                     for index in SLEEVES}
    stats = {}
    row_positions = dict(zip(map(tuple, table[list(KEYS)].to_numpy()), table.index))
    for row in cohorts.to_dict("records"):
        day, sid = row["signal_date"], row["source_security_id"]
        i = positions[day]
        member = members.get(sid)
        for n in RELATIVE_WINDOWS:
            start = max(0, i - n + 1)
            span = slice(start, i + 1)
            for benchmark in ("market", "sleeve"):
                names = [f"relative_{benchmark}_beta_{n}s", f"relative_relative_return_{benchmark}_{n}s"]
                if benchmark == "market":
                    names += [f"relative_idio_vol_market_{n}s", f"relative_residual_return_market_{n}s"]
                complete = i - start + 1 == n and row["identity_eligible"] and member is not None
                stock = np.full(i - start + 1, np.nan)
                if member is not None:
                    same_route = member["route"][span] == row["intrinio_security_id"]
                    valid = member["eligible"][span] & same_route
                    if benchmark == "sleeve":
                        valid &= member["sleeve"][span] == row["index_id"]
                    # The signal itself must also agree on sleeve for either benchmark.
                    complete = complete and (member["route"][i] == row["intrinio_security_id"]
                                              and member["sleeve"][i] == row["index_id"])
                    stock = np.where(valid, member["return"][span], np.nan)
                reference = (market_array[span] if benchmark == "market" else
                             sleeve_arrays.get(row["index_id"], np.full(len(sessions), np.nan))[span])
                complete = complete and bool((np.isfinite(stock) & np.isfinite(reference)).all())
                values = {name: np.nan for name in names}
                status = {name: "incomplete_calendar_pairs" for name in names}
                if complete:
                    simple_name = f"relative_relative_return_{benchmark}_{n}s"
                    values[simple_name], status[simple_name] = np.sum(stock - reference), "ok"
                    variance = np.var(reference, ddof=1)
                    if variance > 0:
                        covariance = np.cov(stock, reference, ddof=1)[0, 1]
                        beta = covariance / variance
                        beta_name = f"relative_{benchmark}_beta_{n}s"
                        values[beta_name], status[beta_name] = beta, "ok"
                        if benchmark == "market":
                            stock_variance = np.var(stock, ddof=1)
                            residual = stock_variance - covariance ** 2 / variance
                            tolerance = 1e-12 * max(abs(stock_variance), 1e-30)
                            vol_name = f"relative_idio_vol_market_{n}s"
                            if residual >= -tolerance:
                                values[vol_name], status[vol_name] = np.sqrt(252 * max(residual, 0)), "ok"
                            else:
                                status[vol_name] = "invalid_residual_variance"
                            return_name = f"relative_residual_return_market_{n}s"
                            values[return_name], status[return_name] = stock.sum() - beta * reference.sum(), "ok"
                    else:
                        for name in names:
                            if name != simple_name:
                                status[name] = "undefined_zero_benchmark_variance"
                for name in names:
                    value = values[name]
                    if np.isfinite(value):
                        table.at[row_positions[(sid, day)], name] = value
                    record = stats.setdefault((day, name), {"total": 0, "valid": 0, "states": set()})
                    record["total"] += 1
                    record["valid"] += int(np.isfinite(value))
                    record["states"].add(status[name])
    for (day, name), summary in sorted(stats.items()):
        status = "ok" if summary["states"] == {"ok"} else ";".join(sorted(summary["states"]))
        coverage.append(_coverage_row(day, name, "source_members", summary["total"], summary["valid"], status,
                                      context_policy=context_policy))


def compute_compact_context_features(daily_result, daily_prices, session_dates, *,
        signal_cohorts, requested_security_ids, daily_cohorts=None,
        identity_policy=STRICT_IDENTITY_POLICY, context_policy="complete_cohort"):
    """Same declared context policy using compact daily membership storage."""
    validate_identity_policy(identity_policy)
    validate_context_policy(context_policy)
    sessions = tuple(iso_date(day, "session date") for day in session_dates)
    if not sessions or tuple(sorted(set(sessions))) != sessions:
        raise DataContractError("Session calendar must be sorted and unique")
    requested = _sequence(requested_security_ids, "requested_security_ids")
    if any(not isinstance(s, str) or not s.strip() for s in requested):
        raise DataContractError("Invalid requested security ID")
    requested = set(requested)
    cohorts = _cohorts(signal_cohorts, "signal_cohorts", identity_policy=identity_policy)
    if cohorts.empty or not cohorts.signal_date.isin(sessions).all():
        raise DataContractError("Signal cohorts must use supplied calendar dates")
    indicators = _frame(daily_result.table, "daily_result")
    required = {"intrinio_security_id", "signal_date", *balanced_cross_sectional_bases()}
    if required - set(indicators):
        raise DataContractError("Daily-result feature bases missing")
    indicators["signal_date"] = _dates(indicators["signal_date"], "indicator date")
    if (indicators.duplicated(["intrinio_security_id", "signal_date"]).any()
            or not indicators.intrinio_security_id.isin(requested).all()
            or not indicators.signal_date.isin(sessions).all()):
        raise DataContractError("Daily-result keys outside the calendar/request or duplicated")
    specs = context_feature_specs()
    table = pd.concat([cohorts[list(KEYS)].reset_index(drop=True),
        pd.DataFrame(np.nan, index=range(len(cohorts)), columns=[r["name"] for r in specs])], axis=1)
    coverage = []
    _xs(table, cohorts, indicators, requested, coverage, context_policy=context_policy)
    arrays = _validated_price_arrays(daily_prices, sessions, requested, cohorts.signal_date.max())
    if daily_cohorts is None:
        for day, rows in cohorts.groupby("signal_date", sort=True):
            for spec in specs:
                if spec["role"] != "cross_sectional":
                    coverage.append(_coverage_row(day, spec["name"], "not_loaded", len(rows), 0, "context_not_loaded",
                                                  context_policy=context_policy))
    else:
        historical = _cohorts(daily_cohorts, "daily_cohorts", identity_policy=identity_policy)
        if historical.empty or not historical.signal_date.isin(sessions).all():
            raise DataContractError("Daily cohorts must use supplied calendar dates")
        if (historical.signal_date > cohorts.signal_date.max()).any():
            raise DataContractError("Daily cohort rows exceed the final declared context signal")
        _validate_signal_context_inventory(cohorts, historical)
        _market(table, cohorts, historical, arrays, sessions, requested, coverage,
                context_policy=context_policy)
        market, sleeve, historical = _compact_daily_context(historical, arrays, sessions, requested,
            context_policy=context_policy, coverage=coverage, through=cohorts.signal_date.max())
        _compact_relative(table, cohorts, market, sleeve, historical, sessions, coverage,
                          context_policy=context_policy)
    provenance = {"source_commit": "e9a7a01881a46374694f60fc34eb7d7e4a26c2b8",
        "registered_columns": 415, "cross_sectional_bases": 80,
        "context_policy": context_policy, "context_scope": "historical_sp500_sp400_union",
        "identity_eligibility_policy": identity_policy,
        "minimum_original_cohort_coverage": context_coverage_fraction(context_policy),
        "exhaustive_identity_admitted_requests_required": True,
        "daily_membership_loaded": daily_cohorts is not None,
        "complete_calendar_windows_required": True,
        "knowledge_time_verified": False, "identity_continuity_certified": False,
        "current_security_master_filter_used": False, "future_quote_eligibility_used": False,
        "target_fields_used": False, "no_fill_no_zero_imputation": True,
        "sleeve_switch_policy": "complete same-index history required within relative window",
        "optimization": "compact daily member arrays; unchanged cross-sectional/market aggregates"}
    provenance.update(identity_policy_provenance(identity_policy))
    return ContextFeaturesResult(table, pd.DataFrame(coverage), specs, provenance)


def _daily_original_coverage(codes, cohort, names):
    attached = cohort[[*KEYS, "intrinio_security_id", "identity_eligible", "requested"]].merge(
        codes, on=["intrinio_security_id", "signal_date"], how="left", validate="many_to_one")
    rows = []
    for signal, group in attached.groupby("signal_date", sort=True):
        for name in names:
            values = group[name].fillna(7).to_numpy(dtype=np.uint8, copy=True)
            values[~group.identity_eligible.to_numpy(dtype=bool)] = 6
            for code, count in zip(*np.unique(values, return_counts=True)):
                rows.append({"signal_date": signal, "feature_name": name,
                    "status": DAILY_STATUS_LEGEND[int(code)], "original_members": int(count)})
    return pd.DataFrame(rows)


def build_full_feature_panel(daily_prices, calendar, signal_cohorts, *,
        requested_security_ids, daily_cohorts=None, mode="historical_reconstruction",
        identity_policy=STRICT_IDENTITY_POLICY,
        normalization_policy=COMMON_MOMENTUM_POLICY, context_policy="complete_cohort"):
    """Assemble one explicit annual partition while retaining original rows.

    Raw daily data is an independent caller-owned object, not compressed into
    monthly candles or mutated here. The feature table contains only original
    source/signal keys and the frozen 728 definitions. Audit/status values and
    raw momentum stay in diagnostics. Targets are never accepted or inspected.
    """
    if mode == "knowledge_time":
        raise KnowledgeTimeUnavailable("Current sources cannot certify knowledge time")
    if mode != "historical_reconstruction":
        raise DataContractError("Unknown feature reconstruction mode")
    validate_identity_policy(identity_policy)
    if isinstance(requested_security_ids, (str, bytes)):
        raise DataContractError("Requested IDs must be an explicit sequence")
    snapshots = tuple(signal_cohorts)
    if not snapshots:
        raise DataContractError("Explicit original signal cohorts are required")
    signals = [iso_date(snapshot.as_of, "signal date") for snapshot in snapshots]
    if signals != sorted(set(signals)) or signals[-1] >= HOLDOUT_START:
        raise DataContractError("Signal cohorts must be unique, chronological and pre-holdout")
    ids = tuple(requested_security_ids)
    sessions = calendar_session_dates(calendar)
    daily = compute_compact_daily_features(daily_prices, sessions, signal_dates=signals,
                                          requested_security_ids=ids)
    # Metadata validation already covered ALL daily rows above. Project only
    # past exchange month-end quotes for the unchanged paper worker afterwards.
    months = {}
    for day in sessions:
        months.setdefault(day[:7], []).append(day)
    ends = [days[-1] for month, days in months.items()
            if str(pd.Period(month, "M") + 1) in months and days[-1] <= signals[-1]]
    if isinstance(daily_prices, pa.Table):
        endpoints = daily_prices.filter(pc.is_in(daily_prices["trading_date"], value_set=pa.array(ends)))
    else:
        endpoints = daily_prices.loc[daily_prices.trading_date.isin(ends)]
    paper = compute_paper_features(endpoints, sessions, signal_cohorts=snapshots,
        requested_security_ids=ids, identity_policy=identity_policy,
        normalization_policy=normalization_policy)
    context = compute_compact_context_features(daily, daily_prices, sessions,
        signal_cohorts=snapshots, requested_security_ids=ids, daily_cohorts=daily_cohorts,
        identity_policy=identity_policy, context_policy=context_policy)
    specs = (*daily.specs, *paper.specs, *context.specs)
    names = [spec["name"] for spec in specs]
    if len(names) != len(set(names)) or len(names) != 728:
        raise DataContractError("Full panel changed the frozen 728 definitions")
    fields = ["intrinio_security_id", "identity_status", "source_identity_status",
        "identity_eligible", "identity_eligibility_policy", "identity_eligibility_reason",
        "identity_review_required", "resolution_status", "paper_common_population", "paper_prediction_eligible"]
    fields = [name for name in fields if name in paper.table]
    cohort = paper.table[[*KEYS, *fields]].copy()
    cohort["requested"] = cohort.intrinio_security_id.isin(ids)
    cohort["knowledge_time_verified"] = False
    cohort["accounting_certified"] = False
    original = [(r["source_security_id"], snapshot.as_of) for snapshot in snapshots
                for r in snapshot.table.select(["source_security_id"]).to_pylist()]
    if (len(original) != len(set(original)) or set(original) != set(map(tuple, cohort[list(KEYS)].to_numpy()))):
        raise DataContractError("Full panel changed original historical cohort denominators")
    attached = cohort[[*KEYS, "intrinio_security_id", "identity_eligible"]].copy()
    attached.loc[~attached.identity_eligible, "intrinio_security_id"] = None
    technical_names = [spec["name"] for spec in daily.specs]
    part = attached.merge(daily.table, on=["intrinio_security_id", "signal_date"],
                          how="left", validate="many_to_one")
    frame = cohort[list(KEYS)].merge(part[[*KEYS, *technical_names]],
                                    on=list(KEYS), how="left", validate="one_to_one")
    for result in (paper, context):
        cols = [spec["name"] for spec in result.specs]
        if set(map(tuple, result.table[list(KEYS)].to_numpy())) != set(original):
            raise DataContractError("Full feature worker changed original denominators")
        frame = frame.merge(result.table[[*KEYS, *cols]], on=list(KEYS), how="left", validate="one_to_one")
    frame = frame[[*KEYS, *names]].copy()
    frame[names] = frame[names].astype(float).replace([np.inf, -np.inf], np.nan)
    finite = frame[names].notna().sum(axis=1)
    ledger = cohort.copy()
    ledger["available_feature_count"] = finite.to_numpy()
    ledger["registered_feature_count"] = len(names)
    ledger["feature_status"] = np.where(~cohort.identity_eligible, "identity_unresolved",
        np.where(~cohort.requested, "not_requested", np.where(finite > 0, "partial_features", "no_usable_features")))
    ledger["model_ready"] = False
    targets = {}
    for signal in signals:
        following = str(pd.Period(signal[:7], "M") + 1)
        evidence = str(pd.Period(signal[:7], "M") + 2)
        if following in months and evidence in months:
            targets[signal] = {"outcome_month": following,
                "outcome_start": months[following][0], "outcome_end": months[following][-1]}
    summary = {"rows": len(frame), "signals": len(signals), "registered_features": len(names),
        "roles": dict(Counter(spec["role"] for spec in specs)),
        "candidate_model_columns": sum(spec.get("model_eligible", True) for spec in specs),
        "feature_status_counts": dict(Counter(ledger.feature_status)),
        "finite_feature_values": int(finite.sum()), "model_ready": False,
        "knowledge_time_verified": False, "accounting_certified": False,
        "identity_eligibility_policy": identity_policy,
        "identity_eligible_rows": int(cohort.identity_eligible.sum()),
        "provisional_identity_eligible_rows": int((cohort.identity_eligible & cohort.identity_status.ne("resolved")).sum()),
        "original_identity_status_counts": dict(Counter(cohort.source_identity_status)),
        "full_daily_context_loaded": daily_cohorts is not None}
    provenance = {"mode": mode, "signal_dates": signals, "requested_ids": list(ids),
        "identity_eligibility_policy": identity_policy,
        "feature_dictionary_sha256": canonical_sha256(specs), "technical": daily.provenance,
        "paper": paper.provenance, "context": context.provenance, "outcomes_consumed": False,
        "feature_clock": "Through closed signal session, inclusive",
        "target_clock": "Strictly after signal close through next month-end; joined separately",
        "target_windows": targets, "calendar_sessions_sha256": canonical_sha256(sessions),
        "cohort_denominator_preserved": True, "knowledge_time_verified": False,
        "source_limitations": "Retrieval-time adjusted history and identity routes lack historical revision/validity certification."}
    provenance.update(identity_policy_provenance(identity_policy))
    diagnostics = {"daily_status_codes": daily.coverage,
        "daily_status_legend": DAILY_STATUS_LEGEND,
        "daily": _daily_original_coverage(daily.coverage, cohort, technical_names),
        "paper": paper.coverage,
        "context": context.coverage.loc[context.coverage.feature_name.ne("benchmark_daily_ew_return_1s")].reset_index(drop=True),
        "daily_benchmark": context.coverage.loc[context.coverage.feature_name.eq("benchmark_daily_ew_return_1s")].reset_index(drop=True),
        "paper_raw_momentum": paper.table[[*KEYS, *(name for name in paper.table
                                    if name.startswith("MOM") and "_raw" in name)]].copy()}
    return FeaturePanel(pa.Table.from_pandas(frame, preserve_index=False),
        pa.Table.from_pandas(ledger, preserve_index=False), specs, diagnostics, summary, provenance)
