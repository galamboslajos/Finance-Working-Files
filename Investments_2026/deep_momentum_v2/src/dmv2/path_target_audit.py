"""Bounded one-stock path targets from immutable corrected caches.

The accounting kernel remains frozen. An isolated flat cash reference makes its
balanced two-leg interface usable as a one-asset calculator, never as a sampled
portfolio. At zero costs/borrow, unit-notional wealth is 1+2*(kernel NAV-1).
No synthetic cash instrument is a research security or a source-price repair.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .catalog import canonical_sha256
from .exchange_sessions import EXCHANGE_CALENDARS_VERSION, _package_code_hash
from .universe import DataContractError
from .walkforward_accounting_v2 import run_walkforward_accounting_v2, DailyAccountingPolicy

SIGNALS = ("2019-12-31", "2020-09-30")
KEYS = ("source_security_id", "signal_date")
VARIANTS = ("unstopped", "asset_stop_10pct")
SIDES = ("long", "short")
CASH_LONG = "__dmv2_path_cash_reference_long__"
CASH_SHORT = "__dmv2_path_cash_reference_short__"
PRICE_COLUMNS = ("trading_date", "intrinio_security_id", "adj_open", "adj_high", "adj_low", "adj_close",
    "open", "close", "frequency", "intraperiod", "price_quality_status",
    "raw_ohlc_bounds_valid", "adjusted_ohlc_bounds_valid")
PRICE_METADATA = ("trading_date", "intrinio_security_id", "frequency", "intraperiod",
    "price_quality_status", "raw_ohlc_bounds_valid", "adjusted_ohlc_bounds_valid")
TARGET_METADATA = (*KEYS, "outcome_month", "outcome_start", "outcome_end", "identity_eligible", "requested",
    "paper_prediction_eligible", "prediction_eligible", "outcome_requested", "target_available", "target_status")
TARGET_VALUES = ("label_return", "target_class")
PROTOCOL = "bounded_corrected_unit_path_audit_JanOct2020_v1"
MAX_COMPRESSED_BYTES = 10 * 1024 * 1024
MAX_SOURCE_ROWS = 300_000


def _need(condition, message):
    if not bool(condition):
        raise DataContractError(message)


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path):
    return json.loads(Path(path).read_text())


def _pin(path):
    path = Path(path).resolve()
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": _hash(path)}


def _verify(item, archive=None):
    path = Path(item["path"])
    if not path.exists() and archive is not None:
        relocation = _json(Path(archive) / "relocation_index.json")
        path = Path(relocation.get("files", {}).get(str(path), str(path)))
    _need(path.is_file() and path.stat().st_size == item["size_bytes"]
        and _hash(path) == item["sha256"].removeprefix("sha256:"), "Pinned path-audit input changed or is absent: " + str(path))
    return path.resolve()


def _projected_bytes(path, columns):
    file = pq.ParquetFile(path)
    _need(set(columns) <= set(file.schema_arrow.names), "Path-audit projection lacks required source columns")
    return sum(file.metadata.row_group(group).column(column).total_compressed_size
        for group in range(file.metadata.num_row_groups) for column in range(file.metadata.num_columns)
        if file.metadata.schema.column(column).path.split(".")[0] in columns)


def _metadata(path, columns):
    _projected_bytes(path, columns)
    return pq.ParquetFile(path).read(columns=list(columns))


def _clock():
    from importlib.metadata import version
    _need(version("exchange-calendars") == EXCHANGE_CALENDARS_VERSION, "Path audit calendar version changed")
    import exchange_calendars as xcals
    calendar = xcals.get_calendar("XNYS", start="2019-12-01", end="2020-11-02")
    _need(type(calendar).__module__ == "exchange_calendars.exchange_calendar_xnys", "Unexpected XNYS implementation")
    sessions = tuple(d for d in calendar.sessions.strftime("%Y-%m-%d") if "2019-12-01" <= d <= "2020-11-02")
    windows = []
    for signal in SIGNALS:
        month = str(pd.Period(signal[:7], "M") + 1)
        following = str(pd.Period(month, "M") + 1)
        prior = [d for d in sessions if d[:7] == signal[:7]]
        current = [d for d in sessions if d[:7] == month]
        after = [d for d in sessions if d[:7] == following]
        _need(prior[-1] == signal and current and after, "Incomplete independently evidenced path window")
        required_marks = [{"date": d, "mark": mark} for d in current for mark in ("open", "close")]
        required_marks.append({"date": after[0], "mark": "open"})
        windows.append({"signal_date": signal, "outcome_month": month, "outcome_start": current[0],
            "outcome_end": current[-1], "entry_date": current[0], "exit_date": after[0], "exit_mark": "open",
            "required_marks": required_marks})
    _, code_hash = _package_code_hash(xcals)
    return sessions, windows, {"exchange": "XNYS", "package_version": EXCHANGE_CALENDARS_VERSION,
        "package_code_sha256": code_hash, "session_dates_sha256": canonical_sha256(sessions),
        "quote_inventory_used": False}


def _guard_keys(table, date_column, *, lower, upper, keys):
    rows = table.select(list(keys)).to_pylist()
    values = [tuple(r[k] for k in keys) for r in rows]
    _need(len(values) == len(set(values)), "Duplicate original path-audit source keys")
    _need(all(isinstance(row[k], str) and row[k].strip() for row in rows for k in keys), "Null/blank source key")
    for day in set(table[date_column].to_pylist()):
        _need(pd.Timestamp(day).strftime("%Y-%m-%d") == day and lower <= day <= upper,
              "Path-audit metadata leaves its approved financial dates")


def _guard_target_clock(table):
    clock_fields = ("signal_date", "outcome_month", "outcome_start", "outcome_end")
    clocks = {tuple(row[name] for name in clock_fields) for row in table.select(list(clock_fields)).to_pylist()}
    for signal, month, start, end in clocks:
        _need(all(isinstance(value, str) for value in (signal, month, start, end))
            and "2020-01-01" <= start <= end <= "2020-12-31"
            and start[:7] == end[:7] == month and str(pd.Period(signal[:7], "M") + 1) == month,
            "Legacy target values are not physically scoped to completed2020outcomes")


def _table_records(table):
    return table.to_pylist()


def plan_path_target_audit(project_root, *, archive_directory=None):
    """Read JSON, footer and projected metadata only; no source value columns."""
    root = Path(project_root).resolve()
    archive = Path(archive_directory).resolve() if archive_directory else root / "outputs/corrected_refit/2020-2023-20261021-v1"
    source_plan_path = archive / "backtest_source_plan.json"
    source = _json(source_plan_path)
    catalog_path = _verify(source["catalog_pin"], archive)
    catalog = _json(catalog_path)
    annual = next(record for record in catalog["annuals"] if record["year"] == 2020)
    sessions, windows, calendar = _clock()
    artifacts = annual["artifacts"]
    paths = {}
    pins = [_pin(source_plan_path), _pin(catalog_path)]
    for name in ("original_cohort", "feature_coverage", "targets", "daily_prices_raw"):
        path = _verify(artifacts[name], archive)
        paths[name] = str(path)
        pins.append(_pin(path))
    original_columns = pq.ParquetFile(paths["original_cohort"]).schema_arrow.names
    coverage_columns = pq.ParquetFile(paths["feature_coverage"]).schema_arrow.names
    _need(not set(original_columns + coverage_columns) & {"label_return", "target_class", "open", "close", "volume", "adj_close"},
          "Metadata projection cannot include predictors or returns")
    original = _metadata(paths["original_cohort"], original_columns)
    coverage = _metadata(paths["feature_coverage"], coverage_columns)
    targets = _metadata(paths["targets"], TARGET_METADATA)
    quotes = _metadata(paths["daily_prices_raw"], PRICE_METADATA)
    _guard_target_clock(targets)
    for table in (original, coverage, targets):
        _guard_keys(table, "signal_date", lower="2019-12-01", upper="2020-11-30", keys=KEYS)
    _guard_keys(quotes, "trading_date", lower="2020-01-01", upper="2020-12-31", keys=("intrinio_security_id", "trading_date"))
    target_windows = {row["signal_date"]: row for row in windows}
    metadata_mask = pc.is_in(original["signal_date"], value_set=pa.array(SIGNALS))
    selected_original = original.filter(metadata_mask)
    selected_coverage = coverage.filter(pc.is_in(coverage["signal_date"], value_set=pa.array(SIGNALS)))
    selected_targets = targets.filter(pc.is_in(targets["signal_date"], value_set=pa.array(SIGNALS)))
    frames = [table.to_pandas().set_index(list(KEYS)) for table in (selected_original, selected_coverage, selected_targets)]
    frames[2] = frames[2].rename(columns={"target_available": "legacy_target_available", "target_status": "legacy_target_status"})
    _need(frames[0].index.equals(frames[1].index) and frames[0].index.equals(frames[2].index),
          "Corrected source, eligibility and target ledgers changed original order/denominators")
    cohort = frames[0].copy()
    for extra in frames[1:]:
        for name in extra:
            if name in cohort:
                same = (cohort[name] == extra[name]) | (cohort[name].isna() & extra[name].isna())
                _need(same.all(), "Corrected cohort metadata conflict: " + name)
            else:
                cohort[name] = extra[name]
    cohort = cohort.reset_index()
    _need(set(cohort.signal_date) == set(SIGNALS), "Bounded audit requires both prespecified complete original months")
    for row in cohort.to_dict("records"):
        clock = target_windows[row["signal_date"]]
        _need(all(row[field] == clock[field] for field in ("outcome_month", "outcome_start", "outcome_end")),
              "Target metadata differs from independent XNYS next-month bounds")
        _need(type(row["prediction_eligible"]) is bool and row["prediction_eligible"] == row["paper_prediction_eligible"],
              "Prediction cohort must be signal-time Paper15 eligibility")
    # All metadata and date checks have passed before any numerical values.
    execution_path = _verify(source["execution_phase_pin"], archive)
    execution = _json(execution_path)
    pins.append(_pin(execution_path))
    events = execution["event_registry"]["events"]
    _need(len({e["event_id"] for e in events}) == len(events), "Duplicate reviewed legal event")
    relevant_events = [e for e in events if any(w["entry_date"] <= e["effective_date"] <= w["exit_date"] for w in windows)]
    repair_path = _verify(source["execution_repairs"]["values"]["pin"], archive)
    repair_metadata = _metadata(repair_path, PRICE_METADATA)
    _guard_keys(repair_metadata, "trading_date", lower="2020-01-01", upper="2023-12-31", keys=("intrinio_security_id", "trading_date"))
    paths["execution_repairs"] = str(repair_path)
    pins.append(_pin(repair_path))
    projections = [(paths["original_cohort"], original_columns), (paths["feature_coverage"], coverage_columns),
        (paths["targets"], TARGET_METADATA), (paths["targets"], TARGET_VALUES),
        (paths["daily_prices_raw"], PRICE_METADATA),
        (paths["daily_prices_raw"], [n for n in PRICE_COLUMNS if n not in PRICE_METADATA]),
        (paths["execution_repairs"], PRICE_METADATA),
        (paths["execution_repairs"], [n for n in PRICE_COLUMNS if n not in PRICE_METADATA])]
    planning_metadata_projections = [(paths["original_cohort"], original_columns),
        (paths["feature_coverage"], coverage_columns), (paths["targets"], TARGET_METADATA),
        (paths["daily_prices_raw"], PRICE_METADATA), (paths["execution_repairs"], PRICE_METADATA)]
    runtime_projections = [(paths["daily_prices_raw"], PRICE_METADATA),
        (paths["daily_prices_raw"], [n for n in PRICE_COLUMNS if n not in PRICE_METADATA]),
        (paths["execution_repairs"], PRICE_METADATA),
        (paths["execution_repairs"], [n for n in PRICE_COLUMNS if n not in PRICE_METADATA]),
        (paths["targets"], TARGET_METADATA), (paths["targets"], TARGET_VALUES)]
    planning_bytes = sum(_projected_bytes(path, columns) for path, columns in planning_metadata_projections)
    materialization_bytes = sum(_projected_bytes(path, columns) for path, columns in runtime_projections)
    compressed = planning_bytes + materialization_bytes
    candidate_rows = sum(pq.ParquetFile(path).metadata.num_rows for path in set(paths.values()))
    _need(compressed <= MAX_COMPRESSED_BYTES and candidate_rows <= MAX_SOURCE_ROWS,
          "Bounded path-audit cumulative read budget exceeded before financial materialization")
    ids = sorted({row.intrinio_security_id for row in cohort.itertuples() if row.prediction_eligible})
    _need(CASH_LONG not in ids and CASH_SHORT not in ids, "Synthetic cash-control name collides with a real route")
    selected_quote_rows = 0
    requested_dates = sorted({r["date"] for w in windows for r in w["required_marks"]})
    for row in quotes.select(["intrinio_security_id", "trading_date"]).to_pylist():
        selected_quote_rows += row["intrinio_security_id"] in ids and row["trading_date"] in requested_dates
    records = pa.Table.from_pandas(cohort, preserve_index=False).to_pylist()
    plan = {"schema": "dmv2-bounded-path-audit-plan/v1", "protocol": PROTOCOL,
        "project_root": str(root), "archive_directory": str(archive), "source_paths": paths,
        "source_pins": pins, "calendar": calendar, "sessions": list(sessions), "windows": windows,
        "original_cohort": records, "requested_ids": ids, "events": relevant_events,
        "historical_retirements": execution["historical_retirements"],
        "source_key_metadata_sha256": canonical_sha256(quotes.select(["intrinio_security_id", "trading_date"]).to_pylist()),
        "target_key_metadata_sha256": canonical_sha256(targets.select(list(KEYS)).to_pylist()),
        "read_plan": {"original_rows": len(cohort), "prediction_eligible_rows": int(cohort.prediction_eligible.sum()),
            "monthly_original_rows": cohort.groupby("outcome_month").size().to_dict(),
            "monthly_prediction_eligible_rows": cohort.groupby("outcome_month").prediction_eligible.sum().astype(int).to_dict(),
            "source_candidate_rows": candidate_rows, "selected_daily_rows_before_overlay": int(selected_quote_rows),
            "projected_compressed_bytes": compressed, "planning_metadata_projected_compressed_bytes": planning_bytes,
            "materialization_projected_compressed_bytes": materialization_bytes,
            "integrity_hashing_bytes_separate_no_value_decode": sum(pin["size_bytes"] for pin in pins),
            "maximum_projected_compressed_bytes": MAX_COMPRESSED_BYTES,
            "maximum_source_rows": MAX_SOURCE_ROWS, "maximum_episode_calls": len(cohort) * 4,
            "financial_values_read": False, "financial_2024_plus_read": False,
            "existing_execution_overlay_rows": repair_metadata.num_rows,
            "price_projection": list(PRICE_COLUMNS), "legacy_value_projection": list(TARGET_VALUES)},
        "limits": {"wall_seconds": 600, "peak_rss_bytes": 2 * 1024 ** 3, "output_bytes": 64 * 1024 ** 2},
        "semantics": {"accounting_kernel": "walkforward_accounting_v2", "unit_notional_not_portfolio_weight": True,
            "cost_bps": 0, "borrow_annual_bps": 0, "synthetic_flat_cash_reference_is_research_security": False,
            "cash_reference_market_PnL": 0, "wealth_transform": "1+2*(kernel_NAV-1)",
            "scheduled_exit_availability_even_if_stopped_early": True,
            "engine_capital_failure_is_unknown_not_minus100pct_censor": True,
            "full_training_history_path_target_coverage_ready": False,
            "model_fitting": False, "full_portfolio_backtest": False,
            "mixed_row_group_Arrow_decode_possible_excluded_rows_not_converted": True}}
    from importlib.metadata import version
    workers = ("path_target_audit.py", "path_targets.py", "walkforward_accounting_v2.py",
               "walkforward_accounting.py", "backtest_accounting.py")
    plan["worker_sha256"] = {name: _hash(Path(__file__).parent / name) for name in workers}
    plan["cli_sha256"] = _hash(Path(__file__).resolve().parents[2] / "scripts/run_path_target_audit.py")
    plan["runtime"] = {name: version(name) for name in ("numpy", "pandas", "pyarrow", "exchange-calendars")}
    plan["plan_sha256"] = canonical_sha256(plan)
    return plan


def verify_path_audit_plan(plan):
    _need(isinstance(plan, dict) and plan.get("schema") == "dmv2-bounded-path-audit-plan/v1", "Path-audit plan schema mismatch")
    clone = {k: v for k, v in plan.items() if k != "plan_sha256"}
    _need(canonical_sha256(clone) == plan.get("plan_sha256") and plan.get("protocol") == PROTOCOL,
          "Path-audit plan changed after metadata review")
    _need(tuple(w["signal_date"] for w in plan["windows"]) == SIGNALS, "Path audit cannot expand beyond prespecified months")
    _need(plan["read_plan"]["projected_compressed_bytes"] <= MAX_COMPRESSED_BYTES
          and plan["read_plan"]["source_candidate_rows"] <= MAX_SOURCE_ROWS,
          "Reviewed path plan exceeds frozen resource scope")
    from importlib.metadata import version
    _need(all(version(name) == expected for name, expected in plan["runtime"].items()), "Path-audit runtime changed after plan")
    _need(_hash(Path(__file__).resolve().parents[2] / "scripts/run_path_target_audit.py") == plan["cli_sha256"],
          "Path-audit CLI changed after plan")
    _need(all(_hash(Path(__file__).parent / name) == expected for name, expected in plan["worker_sha256"].items()),
          "Path target/accounting worker changed after plan")
    for pin in plan["source_pins"]:
        _verify(pin)
    return True


def load_path_audit_values(plan):
    """Load only the immutable measured two-month projection after approval."""
    verify_path_audit_plan(plan)
    frames = []
    dates = sorted({row["date"] for window in plan["windows"] for row in window["required_marks"]})
    for key in ("daily_prices_raw", "execution_repairs"):
        file = pq.ParquetFile(plan["source_paths"][key])
        meta = file.read(columns=list(PRICE_METADATA))
        _guard_keys(meta, "trading_date", lower="2020-01-01", upper="2020-12-31" if key == "daily_prices_raw" else "2023-12-31",
            keys=("intrinio_security_id", "trading_date"))
        if key == "daily_prices_raw":
            _need(canonical_sha256(meta.select(["intrinio_security_id", "trading_date"]).to_pylist()) == plan["source_key_metadata_sha256"],
                  "Daily key projection changed after plan")
        requested = set(plan["requested_ids"]) | {e["successor_id"] for e in plan["events"] if e.get("successor_id")}
        mask = pc.and_(pc.is_in(meta["trading_date"], value_set=pa.array(dates)),
            pc.is_in(meta["intrinio_security_id"], value_set=pa.array(sorted(requested))))
        selected = meta.filter(mask)
        extra_columns = [name for name in PRICE_COLUMNS if name not in PRICE_METADATA]
        # This source is physically <=2023. The date/ID mask is applied in Arrow
        # before excluded financial rows enter Python or any accounting state.
        extra = file.read(columns=extra_columns).filter(mask)
        for name in extra.column_names:
            selected = selected.append_column(name, extra[name])
        frame = selected.select(list(PRICE_COLUMNS)).to_pandas()
        if key == "execution_repairs":
            frame["path_execution_overlay"] = True
        else:
            frame["path_execution_overlay"] = False
        frames.append(frame)
    prices = frames[0]
    overlay = frames[1]
    if len(overlay):
        # Existing reviewed overlay replaces only its exact dated native keys;
        # it never selects stocks or supplies a different route.
        replacement = set(zip(overlay.intrinio_security_id, overlay.trading_date))
        prices = prices.loc[[key not in replacement for key in zip(prices.intrinio_security_id, prices.trading_date)]]
        prices = pd.concat([prices, overlay], ignore_index=True)
    _need(not prices.duplicated(["intrinio_security_id", "trading_date"]).any(), "Conflicting path quote keys")
    file = pq.ParquetFile(plan["source_paths"]["targets"])
    metadata = file.read(columns=list(TARGET_METADATA))
    _guard_keys(metadata, "signal_date", lower="2019-12-01", upper="2020-11-30", keys=KEYS)
    _guard_target_clock(metadata)
    _need(canonical_sha256(metadata.select(list(KEYS)).to_pylist()) == plan["target_key_metadata_sha256"], "Legacy target keys changed")
    mask = pc.is_in(metadata["signal_date"], value_set=pa.array(SIGNALS))
    selected = metadata.filter(mask)
    values = file.read(columns=list(TARGET_VALUES)).filter(mask)
    for name in values.column_names:
        selected = selected.append_column(name, values[name])
    legacy = selected.to_pandas()
    verify_path_audit_plan(plan)
    return prices, legacy


def replay_asset_episode(prices, *, source_security_id, intrinio_security_id, signal_date,
        window, side, path_variant, events=(), historical_retirements=()):
    """Replay a single asset with two isolated constant-unit cash controls."""
    _need(side in SIDES and path_variant in VARIANTS, "Unknown separate directional path variant")
    _need(intrinio_security_id not in (CASH_LONG, CASH_SHORT) and source_security_id not in (CASH_LONG, CASH_SHORT),
          "Real asset cannot use cash-control identity")
    _need("trading_date" in prices and all(isinstance(day, str) and "2020-01-01" <= day <= "2023-12-31"
          for day in prices.trading_date.unique()), "Path prices contain unauthorized date metadata")
    _need(all(e.get("predecessor_id") not in (CASH_LONG, CASH_SHORT)
          and e.get("successor_id") not in (CASH_LONG, CASH_SHORT) for e in events),
          "Research actions cannot apply to synthetic cash controls")
    entry, exit_date = window["entry_date"], window["exit_date"]
    _need("2020-01-01" <= entry < exit_date <= "2023-12-31", "Reusable path adapter is restricted to 2020–2023")
    required = window["required_marks"]
    sessions = sorted({m["date"] for m in required})
    _need(required[0] == {"date": entry, "mark": "open"} and required[-1] == {"date": exit_date, "mark": "open"},
          "Path schedule must include exact entry and scheduled next-open exit")
    relevant_events = [e for e in events if entry <= str(e["effective_date"])[:10] <= exit_date]
    needed_ids = {intrinio_security_id} | {e.get("successor_id") for e in relevant_events if e.get("successor_id")}
    selected_prices = prices.loc[prices.intrinio_security_id.isin(needed_ids) & prices.trading_date.isin(sessions)].copy()
    controls = [{"trading_date": day, "intrinio_security_id": sid,
        "open": 1., "close": 1., "adj_open": 1., "adj_high": 1., "adj_low": 1., "adj_close": 1.,
        "frequency": "daily", "intraperiod": False, "price_quality_status": "accepted",
        "raw_ohlc_bounds_valid": True, "adjusted_ohlc_bounds_valid": True}
        for day in sessions for sid in (CASH_LONG, CASH_SHORT)]
    selected_prices = pd.concat([selected_prices, pd.DataFrame(controls)], ignore_index=True)
    opposite = CASH_SHORT if side == "long" else CASH_LONG
    selections = pd.DataFrame([
        {"source_security_id": source_security_id, "intrinio_security_id": intrinio_security_id,
            "signal_date": signal_date, "execution_date": entry, "side": side},
        {"source_security_id": opposite, "intrinio_security_id": opposite,
            "signal_date": signal_date, "execution_date": entry, "side": "short" if side == "long" else "long"},
        *[{"source_security_id": sid, "intrinio_security_id": sid,
            "signal_date": window["outcome_end"], "execution_date": exit_date, "side": label}
            for sid, label in ((CASH_LONG, "long"), (CASH_SHORT, "short"))]])
    rebalances = pd.DataFrame([{"signal_date": signal_date, "execution_date": entry},
        {"signal_date": window["outcome_end"], "execution_date": exit_date}])
    result = run_walkforward_accounting_v2(selections, selected_prices, sessions,
        rebalances=rebalances, terminal_date=exit_date,
        policy=DailyAccountingPolicy(cost_bps=0, borrow_annual_bps=0, required_names_per_side=1,
            asset_stop_fraction=.1 if path_variant == "asset_stop_10pct" else None),
        events=relevant_events, historical_retirements=historical_retirements)
    normalized = []
    permitted = {(pd.Timestamp(m["date"]), m["mark"]) for m in required}
    for row in result.daily.itertuples():
        if (row.date, row.mark) not in permitted or pd.isna(row.nav):
            continue
        normalized.append({"date": row.date.strftime("%Y-%m-%d"), "mark": row.mark,
            "wealth": 1. + 2. * (float(row.nav) - 1.)})
    reason = "" if result.status == "complete_proxy" else ";".join(sorted(set(result.failures.reason)))
    orders = result.orders
    retired = (not orders.empty and ((orders.source_security_id == source_security_id)
        & orders.status.eq("unfilled_retired_security")).any())
    if retired:
        reason = "retired_before_entry_unfilled_not_zero_payoff"
    available = result.status == "complete_proxy" and not retired and len(normalized) == len(required)
    if not available and not reason:
        reason = "incomplete_required_mark_grid"
    _need(float(result.summary["executed_trade_cost"]) == 0 and float(result.summary["valid_prefix_borrow_cost"]) == 0,
          "Cash-reference unit normalization requires zero costs/borrow")
    cash_rows = result.trades[result.trades.intrinio_security_id.isin([CASH_LONG, CASH_SHORT])]
    _need(cash_rows.empty or (cash_rows.price.eq(1).all() and cash_rows.cost.eq(0).all()),
          "Synthetic cash control acquired a nonflat price or cost")
    return {"requested": True, "status": "complete" if available else "unavailable", "reason": reason,
        "marks": normalized, "stop_count": len(result.stops), "action_count": len(result.action_events),
        "unfilled_entry": bool(retired), "engine_status": result.status,
        "failures": result.failures.to_dict("records"),
        "engine_capital_gate_triggered": bool(not result.failures.empty and result.failures.reason.str.contains("nav|capital", case=False).any())}


def build_audit_paths(plan, prices, *, resource_check=None):
    """Request every eligible original key independently; retain all others."""
    windows = {w["signal_date"]: w for w in plan["windows"]}
    episodes, marks, details = [], [], []
    for ordinal, original in enumerate(plan["original_cohort"]):
        if resource_check is not None and ordinal % 10 == 0:
            resource_check()
        if ordinal % 100 == 0:
            print(json.dumps({"stage": "bounded_single_asset_paths", "original_members_processed": ordinal,
                "original_members_total": len(plan["original_cohort"])}), flush=True)
        identity = {k: original[k] for k in KEYS}
        eligible = original["prediction_eligible"]
        for side in SIDES:
            for variant in VARIANTS:
                key = {**identity, "side": side, "path_variant": variant}
                if not eligible:
                    episodes.append({**key, "requested": False, "status": "not_requested", "reason": "signal_time_prediction_ineligible"})
                    continue
                result = replay_asset_episode(prices, source_security_id=original["source_security_id"],
                    intrinio_security_id=original["intrinio_security_id"], signal_date=original["signal_date"],
                    window=windows[original["signal_date"]], side=side, path_variant=variant,
                    events=plan["events"], historical_retirements=plan["historical_retirements"])
                episodes.append({**key, **{n: result[n] for n in ("requested", "status", "reason")}})
                marks.extend({**key, **row} for row in result["marks"])
                details.append({**key, **{n: result[n] for n in ("stop_count", "action_count", "unfilled_entry", "engine_status", "engine_capital_gate_triggered")},
                    "failure_records_json": json.dumps(result["failures"], default=str)})
    return pd.DataFrame(episodes), pd.DataFrame(marks), pd.DataFrame(details)


def read_path_target_audit(output_directory):
    """Read <=1MiB JSON and integrity-hash private artifacts; decode no tables."""
    directory = Path(output_directory).resolve()
    path = directory / "audit.json"
    if not path.is_file():
        raise FileNotFoundError("No saved bounded path-target audit")
    _need(path.stat().st_size <= 1024 * 1024, "Saved audit JSON exceeds passive byte ceiling")
    audit = _json(path)
    _need(audit.get("status") == "COMPLETE_BOUNDED_AUDIT" and audit.get("protocol") == PROTOCOL,
          "Saved bounded path audit is incomplete or belongs to another protocol")
    for pin in audit["artifacts"].values():
        candidate = (directory / pin["path"]).resolve()
        _need(candidate.is_relative_to(directory), "Saved path artifact escapes its output directory")
        _verify({**pin, "path": str(candidate)})
    return audit
