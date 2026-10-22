"""Explicit corrected 2010–2023 path-label audit; no models or portfolios.

Planning decodes metadata only. Financial conversion is annual, with independently
dated next-year first-open requests, and never includes a 2024 quote. Completed
annual checkpoints are immutable and individually bound to the reviewed plan.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .catalog import canonical_sha256
from .exchange_sessions import EXCHANGE_CALENDARS_VERSION, _package_code_hash
from .universe import DataContractError

YEARS = tuple(range(2010, 2024))
KEYS = ("source_security_id", "signal_date")
SIDES = ("long", "short")
VARIANTS = ("unstopped", "asset_stop_10pct")
PROTOCOL = "corrected_full_2010_2023_directional_paths_v1"
SCHEMA = "dmv2-full-path-audit-plan/v1"
CATALOG_SHA256 = "7ebc434ae394410ff5f5377a538b0e43eb290d6685f7f9659ef098d8378c7394"
PRICE_METADATA = ("trading_date", "intrinio_security_id", "frequency", "intraperiod",
    "price_quality_status", "raw_ohlc_bounds_valid", "adjusted_ohlc_bounds_valid")
PRICE_VALUES = ("adj_open", "adj_high", "adj_low", "adj_close", "open", "close")
TARGET_METADATA = (*KEYS, "outcome_month", "outcome_start", "outcome_end", "identity_eligible", "requested",
    "paper_prediction_eligible", "prediction_eligible", "outcome_requested", "target_available", "target_status")
TARGET_VALUES = ("label_return", "target_class")
MAX_COMPRESSED_BYTES = 512 * 1024 ** 2
MAX_SOURCE_ROWS = 8_000_000
MAX_JSON_BYTES = 4 * 1024 ** 2
WORKERS = ("full_path_audit.py", "full_path_accounting.py", "path_targets.py", "catalog.py", "exchange_sessions.py", "universe.py")


def need(condition, message):
    if not bool(condition):
        raise DataContractError(message)


def hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b""):
            digest.update(block)
    return digest.hexdigest()


def pin(path):
    path = Path(path).resolve()
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": hash_file(path)}


def verify_pin(item):
    path = Path(item["path"])
    need(path.is_file() and path.stat().st_size == item["size_bytes"]
        and hash_file(path) == item["sha256"].removeprefix("sha256:"), "Immutable path-audit input changed: " + str(path))
    return path


def read_json(path, cap=MAX_JSON_BYTES):
    path = Path(path)
    need(path.stat().st_size <= cap, "Path-audit JSON exceeds explicit byte cap")
    return json.loads(path.read_text())


def projection(path, columns):
    file = pq.ParquetFile(path)
    need(set(columns) <= set(file.schema_arrow.names), "Missing projected source columns")
    sizes = [sum(file.metadata.row_group(group).column(column).total_compressed_size
        for column in range(file.metadata.num_columns)
        if file.metadata.schema.column(column).path.split(".")[0] in columns)
        for group in range(file.metadata.num_row_groups)]
    return {"columns": list(columns), "rows": file.metadata.num_rows,
        "row_groups": len(sizes), "row_group_compressed_bytes": sizes,
        "projected_compressed_bytes": sum(sizes)}


def metadata(path, columns):
    projection(path, columns)
    return pq.ParquetFile(path).read(columns=list(columns), use_threads=False)


def guard_keys(table, fields, date_field, lower, upper):
    rows = table.select(list(fields)).to_pylist()
    tuples = [tuple(row[name] for name in fields) for row in rows]
    need(len(tuples) == len(set(tuples)), "Duplicate annual source keys")
    need(all(type(row[name]) is str and row[name].strip() for row in rows for name in fields), "Missing original metadata key")
    need(all(type(day) is str and pd.Timestamp(day).strftime("%Y-%m-%d") == day and lower <= day <= upper
        for day in set(table[date_field].to_pylist())), "Financial metadata date outside reviewed scope")


def guard_target_clock(table, year):
    clocks = {tuple(row[k] for k in ("signal_date", "outcome_month", "outcome_start", "outcome_end"))
        for row in table.select(["signal_date", "outcome_month", "outcome_start", "outcome_end"]).to_pylist()}
    need(len(clocks) == 12, "Annual targets must retain all12 original monthly clocks")
    for signal, month, start, end in clocks:
        need(all(type(value) is str for value in (signal, month, start, end))
            and f"{year}-01-01" <= start <= end <= f"{year}-12-31" and start[:7] == end[:7] == month
            and str(pd.Period(signal[:7], "M") + 1) == month,
            "Target values leave the annual pre-2024 outcome clock")


def full_clock():
    from importlib.metadata import version
    need(version("exchange-calendars") == EXCHANGE_CALENDARS_VERSION, "Independent calendar version changed")
    import exchange_calendars as xcals
    calendar = xcals.get_calendar("XNYS", start="2009-12-01", end="2024-01-05")
    need(type(calendar).__module__ == "exchange_calendars.exchange_calendar_xnys", "Unexpected XNYS implementation")
    sessions = tuple(calendar.sessions.strftime("%Y-%m-%d"))
    by_month = {}
    for day in sessions:
        by_month.setdefault(day[:7], []).append(day)
    windows = []
    for month in pd.period_range("2010-01", "2023-12", freq="M"):
        prior, current, following = (by_month[str(month + offset)] for offset in (-1, 0, 1))
        grid = [{"date": day, "mark": mark} for day in current for mark in ("open", "close")]
        grid.append({"date": following[0], "mark": "open"})
        windows.append({"signal_date": prior[-1], "outcome_month": str(month), "outcome_start": current[0],
            "outcome_end": current[-1], "entry_date": current[0], "exit_date": following[0], "exit_mark": "open",
            "required_marks": grid})
    _, package_hash = _package_code_hash(xcals)
    return sessions, windows, {"exchange": "XNYS", "package_version": EXCHANGE_CALENDARS_VERSION,
        "package_code_sha256": package_hash, "session_dates_sha256": canonical_sha256(sessions),
        "quote_inventory_used": False, "2024_dates_are_calendar_metadata_only": True}


def _cohort_from_metadata(paths, year, windows):
    original_columns = pq.ParquetFile(paths["original_cohort"]).schema_arrow.names
    coverage_columns = pq.ParquetFile(paths["feature_coverage"]).schema_arrow.names
    forbidden = {"label_return", "target_class", "open", "close", "volume", "adj_close"}
    need(not set(original_columns + coverage_columns) & forbidden, "Unexpected financial field in cohort metadata")
    tables = [metadata(paths["original_cohort"], original_columns), metadata(paths["feature_coverage"], coverage_columns),
        metadata(paths["targets"], TARGET_METADATA)]
    guard_target_clock(tables[2], year)
    for table in tables:
        guard_keys(table, KEYS, "signal_date", f"{year-1}-12-01", f"{year}-11-30")
    frames = [table.to_pandas().set_index(list(KEYS)) for table in tables]
    need(frames[0].index.equals(frames[1].index) and frames[0].index.equals(frames[2].index),
        "Original, feature eligibility and legacy target keys/order differ")
    frames[2] = frames[2].rename(columns={"target_available": "legacy_target_available", "target_status": "legacy_target_status"})
    cohort = frames[0].copy()
    for frame in frames[1:]:
        for name in frame:
            if name in cohort:
                need(((cohort[name] == frame[name]) | (cohort[name].isna() & frame[name].isna())).all(),
                    "Conflicting corrected signal metadata: " + name)
            else:
                cohort[name] = frame[name]
    cohort = cohort.reset_index()
    clock = {row["signal_date"]: row for row in windows if row["outcome_month"].startswith(str(year))}
    need(set(cohort.signal_date) == set(clock), "Annual original ledger lacks a monthly cross-section")
    for row in cohort.to_dict("records"):
        need(all(row[field] == clock[row["signal_date"]][field] for field in ("outcome_month", "outcome_start", "outcome_end")),
            "Legacy monthly endpoints differ from independent calendar")
        need(type(row["prediction_eligible"]) is bool and row["prediction_eligible"] == row["paper_prediction_eligible"],
            "Eligibility must remain the corrected signal-time Paper15 mask")
    return cohort, original_columns, coverage_columns


def _metadata_hash(frame):
    return canonical_sha256(pa.Table.from_pandas(frame, preserve_index=False).to_pylist())


def _workers():
    directory = Path(__file__).parent
    return {name: hash_file(directory / name) for name in WORKERS if (directory / name).exists()}


def plan_full_path_audit(project_root, *, archive_directory=None):
    """Measure all14 annual scopes using only metadata projections and footers."""
    from importlib.metadata import version
    root = Path(project_root).resolve()
    archive = Path(archive_directory).resolve() if archive_directory else root / "outputs/corrected_refit/2020-2023-20261021-v1"
    source_path = archive / "backtest_source_plan.json"
    source = read_json(source_path)
    need(source["catalog_pin"]["sha256"].removeprefix("sha256:") == CATALOG_SHA256,
        "Full target audit must use the exact reviewed corrected catalog")
    catalog_path = verify_pin(source["catalog_pin"])
    catalog = read_json(catalog_path)
    need(catalog["status"] == "VERIFIED_PROVISIONAL_FROZEN" and tuple(r["year"] for r in catalog["annuals"]) == YEARS,
        "Full audit requires exact latest corrected2010–2023 annual catalog")
    sessions, windows, calendar = full_clock()
    pins = [pin(source_path), pin(catalog_path)]
    protocol_path = Path(__file__).resolve().parents[2] / "configs/full_path_target_audit_20261022.json"
    protocol = read_json(protocol_path)
    need(protocol["source_policy"]["catalog_sha256"] == CATALOG_SHA256
        and protocol["execution"]["model_fits"] == protocol["execution"]["preprocessing_fits"] == protocol["execution"]["portfolio_backtests"] == 0
        and protocol["execution"]["worker_processes"] == 1 and protocol["execution"]["peak_process_rss_bytes"] == 4 * 1024 ** 3
        and protocol["execution"]["active_wall_seconds"] == 2700
        and protocol["targets"]["risk_penalty"] == 1.0 and protocol["targets"]["cost_bps"] == 0,
        "Frozen full path protocol differs from implemented audit scope")
    pins.append(pin(protocol_path))
    execution_path = verify_pin(source["execution_phase_pin"])
    execution = read_json(execution_path)
    pins.append(pin(execution_path))
    events = execution["event_registry"]["events"] + execution["historical_retirements"]
    need(len({e["event_id"] for e in events}) == len(events) and all(e["verified"] is True for e in events),
        "Duplicated or unverified signed event evidence")
    overlay_path = verify_pin(source["execution_repairs"]["values"]["pin"])
    pins.append(pin(overlay_path))
    overlay_meta = metadata(overlay_path, PRICE_METADATA)
    guard_keys(overlay_meta, ("intrinio_security_id", "trading_date"), "trading_date", "2010-01-01", "2023-12-31")
    annuals = []
    planning_bytes = projection(overlay_path, PRICE_METADATA)["projected_compressed_bytes"]
    source_rows = overlay_meta.num_rows
    for annual in catalog["annuals"]:
        year = annual["year"]
        paths = {name: str(verify_pin(annual["artifacts"][name])) for name in
            ("original_cohort", "feature_coverage", "targets", "daily_prices_raw")}
        pins.extend(pin(path) for path in paths.values())
        cohort, original_columns, coverage_columns = _cohort_from_metadata(paths, year, windows)
        provenance_path = verify_pin(annual["artifacts"]["source_provenance"])
        provenance = read_json(provenance_path)
        candidate_path = verify_pin(annual["artifacts"]["candidate_quote_provenance"])
        candidate_columns = ("intrinio_security_id", "trading_date")
        candidate_metadata = metadata(candidate_path, candidate_columns)
        pins.extend((pin(provenance_path), pin(candidate_path)))
        declared_ids = set(provenance["requested_ids"]) | set(candidate_metadata["intrinio_security_id"].to_pylist())
        candidate_projection = projection(candidate_path, candidate_columns)
        prices_meta = metadata(paths["daily_prices_raw"], PRICE_METADATA)
        guard_keys(prices_meta, ("intrinio_security_id", "trading_date"), "trading_date", f"{year}-01-01", f"{year}-12-31")
        projections = {"original": projection(paths["original_cohort"], original_columns),
            "coverage": projection(paths["feature_coverage"], coverage_columns),
            "target_metadata": projection(paths["targets"], TARGET_METADATA),
            "target_values": projection(paths["targets"], TARGET_VALUES),
            "price_metadata": projection(paths["daily_prices_raw"], PRICE_METADATA),
            "price_values": projection(paths["daily_prices_raw"], PRICE_VALUES)}
        planning = sum(projections[key]["projected_compressed_bytes"] for key in ("original", "coverage", "target_metadata", "price_metadata"))
        materialization = planning + projections["target_values"]["projected_compressed_bytes"] + projections["price_values"]["projected_compressed_bytes"]
        planning += candidate_projection["projected_compressed_bytes"]
        planning_bytes += planning
        source_rows += sum(pq.ParquetFile(path).metadata.num_rows for path in paths.values()) + candidate_metadata.num_rows
        # Adjacent annual cache is a separately measured physical projection;
        # only exact next-year first-open rows may cross the Arrow/Python boundary.
        monthly = cohort.groupby("outcome_month", sort=True).agg(original_rows=("source_security_id", "size"),
            prediction_eligible_rows=("prediction_eligible", "sum"), legacy_target_available_rows=("legacy_target_available", "sum"))
        monthly_records = monthly.reset_index().to_dict("records")
        for record in monthly_records:
            members = cohort.loc[cohort.outcome_month == record["outcome_month"]]
            record["original_key_sha256"] = canonical_sha256(sorted([list(key) for key in members.loc[:, list(KEYS)].itertuples(index=False, name=None)]))
            record["eligible_key_sha256"] = canonical_sha256(sorted([list(key) for key in members.loc[members.prediction_eligible, list(KEYS)].itertuples(index=False, name=None)]))
        annuals.append({"year": year, "paths": paths, "original_rows": len(cohort),
            "prediction_eligible_rows": int(cohort.prediction_eligible.sum()),
            "legacy_target_available_rows": int(cohort.legacy_target_available.sum()),
            "legacy_target_status_counts": dict(Counter(cohort.legacy_target_status)),
            "monthly": monthly_records,
            "cohort_metadata_sha256": _metadata_hash(cohort),
            "price_key_metadata_sha256": canonical_sha256(prices_meta.select(["intrinio_security_id", "trading_date"]).to_pylist()),
            "projections": projections, "planning_compressed_bytes": planning,
            "materialization_compressed_bytes": materialization,
            "physical_price_rows": prices_meta.num_rows,
            "declared_cache_requested_ids": sorted(declared_ids),
            "declared_request_evidence": {"source_provenance": str(provenance_path), "candidate_quote_provenance": str(candidate_path),
                "candidate_metadata_projection": candidate_projection},
            "first_session_observed_ids": sorted(set(prices_meta.filter(pc.equal(prices_meta["trading_date"],
                next(w["entry_date"] for w in windows if w["outcome_month"] == f"{year}-01")))["intrinio_security_id"].to_pylist())),
            "december_original_requests": pa.Table.from_pandas(cohort.loc[cohort.prediction_eligible & cohort.outcome_month.eq(f"{year}-12"),
                ["source_security_id", "intrinio_security_id"]], preserve_index=False).to_pylist(),
            "requested_ids": sorted(set(cohort.loc[cohort.prediction_eligible, "intrinio_security_id"])),
            "action_successor_request_evidence": [{"event_id": e["event_id"], "effective_date": e["effective_date"],
                "predecessor_id": e["predecessor_id"], "successor_id": e["successor_id"],
                "successor_declared_requested": e["successor_id"] in declared_ids}
                for e in events if e["effective_date"].startswith(str(year)) and e.get("successor_id")
                and e["predecessor_id"] in set(cohort.loc[cohort.prediction_eligible, "intrinio_security_id"])],
            "windows": [w for w in windows if w["outcome_month"].startswith(str(year))]})
        print(json.dumps({"stage": "metadata_plan", "year": year, "original_rows": len(cohort),
            "eligible_rows": int(cohort.prediction_eligible.sum())}), flush=True)
    overlay_projection = projection(overlay_path, (*PRICE_METADATA, *PRICE_VALUES))
    for index, annual in enumerate(annuals):
        annual["next_year_endpoint"] = None
        if index + 1 < len(annuals):
            following = annuals[index + 1]
            endpoint = annual["windows"][-1]["exit_date"]
            byte_count = following["projections"]["price_metadata"]["projected_compressed_bytes"] + following["projections"]["price_values"]["projected_compressed_bytes"]
            annual["next_year_endpoint"] = {"path": following["paths"]["daily_prices_raw"], "date": endpoint,
                "physical_year": following["year"], "projected_compressed_bytes": byte_count,
                "row_groups": following["projections"]["price_metadata"]["row_groups"]}
            requested_ids = set(following["declared_cache_requested_ids"])
            observed_ids = set(following["first_session_observed_ids"])
            states, missing = Counter(), []
            for request in annual["december_original_requests"]:
                native = request["intrinio_security_id"]
                for event in sorted(events, key=lambda e: (e["effective_date"], e["event_id"])):
                    if annual["windows"][-1]["entry_date"] <= event["effective_date"] <= endpoint and event["predecessor_id"] == native:
                        native = event.get("successor_id")
                status = ("cash_action_no_terminal_quote_required" if native is None else "observed_cache_key" if native in observed_ids
                    else "requested_boundary_cache_quote_missing" if native in requested_ids else "not_requested_boundary")
                states[status] += 1
                if status in {"requested_boundary_cache_quote_missing", "not_requested_boundary"}:
                    missing.append({**request, "endpoint_native": native, "date": endpoint, "status": status})
            annual["boundary_request_audit"] = {"counts": dict(states), "unobserved_requests": missing,
                "absence_is_not_evidence_source_price_does_not_exist": True}
            annual["materialization_compressed_bytes"] += byte_count
        annual["materialization_compressed_bytes"] += overlay_projection["projected_compressed_bytes"]
    materialization_bytes = sum(a["materialization_compressed_bytes"] for a in annuals)
    total = planning_bytes + materialization_bytes
    need(total <= MAX_COMPRESSED_BYTES and source_rows <= MAX_SOURCE_ROWS, "Full path plan exceeds measured read cap")
    workers = _workers()
    cli_path = Path(__file__).resolve().parents[2] / "scripts/run_full_path_audit.py"
    plan = {"schema": SCHEMA, "protocol": PROTOCOL, "project_root": str(root), "archive_directory": str(archive),
        "source_pins": pins, "annuals": annuals, "sessions": list(sessions), "calendar": calendar,
        "protocol_pin": pin(protocol_path), "protocol_id": protocol["protocol_id"],
        "events": events, "historical_retirements": execution["historical_retirements"],
        "execution_overlay": {"path": str(overlay_path), "metadata_sha256": canonical_sha256(overlay_meta.to_pylist()),
            "projection": overlay_projection}, "worker_sha256": workers,
        "cli_sha256": hash_file(cli_path) if cli_path.exists() else None,
        "runtime": {name: version(name) for name in ("numpy", "pandas", "pyarrow", "exchange-calendars")},
        "ready_for_materialization": set(workers) == set(WORKERS) and cli_path.exists(),
        "read_plan": {"original_rows": sum(a["original_rows"] for a in annuals),
            "prediction_eligible_rows": sum(a["prediction_eligible_rows"] for a in annuals),
            "legacy_target_available_rows": sum(a["legacy_target_available_rows"] for a in annuals),
            "original_months": 168, "original_years": 14, "source_candidate_rows": source_rows,
            "planning_metadata_projected_compressed_bytes": planning_bytes,
            "materialization_projected_compressed_bytes": materialization_bytes,
            "cumulative_projected_compressed_bytes": total,
            "maximum_projected_compressed_bytes": MAX_COMPRESSED_BYTES,
            "integrity_hash_bytes_per_complete_verification_pass": sum(p["size_bytes"] for p in pins),
            "financial_values_read": False, "financial_2024_plus_read": False,
            "next_year_endpoint_projections": 13, "protected_final_endpoint": "2024-01-02",
            "price_value_projection": list(PRICE_VALUES), "legacy_value_projection": list(TARGET_VALUES)},
        "limits": {"active_wall_seconds": 2700, "peak_rss_bytes": 4 * 1024 ** 3,
            "output_bytes_per_year": 512 * 1024 ** 2, "output_bytes_total": 8 * 1024 ** 3},
        "lineage": {"corrected_catalog_schema": catalog["schema"], "new_learning_quote_rows_in_parent": catalog["new_learning_quote_rows"],
            "canonical_identity_promoted": False, "original_endpoint_targets_rebuilt_here": False,
            "new_path_target_families_built_separately": True,
            "primary_prices_are_corrected_learning_catalog_raw_caches": True,
            "execution_overlay_is_supplemental_diagnostic_not_learning_admitted": True,
            "reviewed_action_count": len(events), "pre2020_reviewed_action_count": len(execution["historical_retirements"]),
            "events_are_retrieval_time_review_not_certified_knowledge_time": True,
            "earlier_unreviewed_corporate_actions_not_certified": True,
            "future_quote_availability_does_not_change_signal_eligibility": True},
        "semantics": {"unit_notional": 1.0, "cost_bps": 0, "borrow_bps": 0, "risk_penalty": 1.0,
            "endpoint": "next_month_first_session_open", "early_stop_does_not_advance_class_availability": True,
            "full_original_cohort_retained": True, "financial_2024_plus_read": False,
            "model_fits": 0, "portfolio_backtests": 0, "monthly_marks_streamed_annual_checkpoint": True}}
    plan["plan_sha256"] = canonical_sha256(plan)
    return plan


def verify_plan(plan, *, year=None):
    from importlib.metadata import version
    need(plan.get("schema") == SCHEMA and plan.get("protocol") == PROTOCOL, "Full path plan schema mismatch")
    need(canonical_sha256({k: v for k, v in plan.items() if k != "plan_sha256"}) == plan["plan_sha256"], "Reviewed full path plan changed")
    need(plan["ready_for_materialization"] and tuple(a["year"] for a in plan["annuals"]) == YEARS, "Plan is draft or wrong annual scope")
    need(plan["read_plan"]["cumulative_projected_compressed_bytes"] <= MAX_COMPRESSED_BYTES
        and plan["read_plan"]["source_candidate_rows"] <= MAX_SOURCE_ROWS, "Reviewed read budget exceeded")
    need(plan["limits"]["active_wall_seconds"] == 2700 and plan["limits"]["peak_rss_bytes"] == 4 * 1024 ** 3
        and plan["limits"]["output_bytes_per_year"] == 512 * 1024 ** 2 and plan["limits"]["output_bytes_total"] == 8 * 1024 ** 3,
        "Reviewed fixed resource ceiling changed")
    need(any(item["sha256"].removeprefix("sha256:") == CATALOG_SHA256 for item in plan["source_pins"]),
        "Reviewed full plan lacks the exact corrected catalog pin")
    need(_workers() == plan["worker_sha256"] and hash_file(Path(__file__).resolve().parents[2] / "scripts/run_full_path_audit.py") == plan["cli_sha256"],
        "Reviewed full path worker/CLI changed")
    need(all(version(name) == value for name, value in plan["runtime"].items()), "Reviewed runtime changed")
    sessions, windows, calendar = full_clock()
    need(plan["sessions"] == list(sessions) and plan["calendar"] == calendar
        and [w for a in plan["annuals"] for w in a["windows"]] == windows, "Reviewed independent168-month clock changed")
    paths = None
    if year is not None:
        annual = next(a for a in plan["annuals"] if a["year"] == year)
        paths = set(annual["paths"].values()) | {plan["execution_overlay"]["path"]}
        if annual["next_year_endpoint"]:
            paths.add(annual["next_year_endpoint"]["path"])
    for item in plan["source_pins"]:
        if paths is None or item["path"] in paths:
            verify_pin(item)
    return True


def load_annual_values(plan, year):
    """Bounded annual conversion, plus exact next-year first-open rows only."""
    verify_plan(plan, year=year)
    annual = next(a for a in plan["annuals"] if a["year"] == year)
    cohort, _, _ = _cohort_from_metadata(annual["paths"], year, annual["windows"])
    need(_metadata_hash(cohort) == annual["cohort_metadata_sha256"], "Corrected original metadata differs from reviewed plan")
    ids = set(annual["requested_ids"])
    # Event successors are request evidence only; their availability never
    # decides admission. Recurse because an acquired successor may itself retire.
    changed = True
    relevant = [e for e in plan["events"] if f"{year}-01-01" <= e["effective_date"] <= annual["windows"][-1]["exit_date"]]
    while changed:
        old = len(ids)
        ids.update(e["successor_id"] for e in relevant if e["predecessor_id"] in ids and e.get("successor_id"))
        changed = len(ids) != old
    inputs = [(annual["paths"]["daily_prices_raw"], year, None)]
    if annual["next_year_endpoint"]:
        endpoint = annual["next_year_endpoint"]
        inputs.append((endpoint["path"], year + 1, endpoint["date"]))
    frames = []
    for path, physical_year, exact_day in inputs:
        file = pq.ParquetFile(path)
        meta = file.read(columns=list(PRICE_METADATA), use_threads=False)
        guard_keys(meta, ("intrinio_security_id", "trading_date"), "trading_date", f"{physical_year}-01-01", f"{physical_year}-12-31")
        need(physical_year <= 2023, "Never decode a2024financial source")
        if exact_day is None:
            need(canonical_sha256(meta.select(["intrinio_security_id", "trading_date"]).to_pylist()) == annual["price_key_metadata_sha256"],
                "Daily physical metadata differs from reviewed plan")
        mask = pc.and_(pc.is_in(meta["intrinio_security_id"], value_set=pa.array(sorted(ids))),
            pc.is_in(meta["trading_date"], value_set=pa.array(plan["sessions"])))
        if exact_day:
            mask = pc.and_(mask, pc.equal(meta["trading_date"], exact_day))
        value = file.read(columns=list(PRICE_VALUES), use_threads=False).filter(mask)
        combined = meta.filter(mask)
        for name in PRICE_VALUES:
            combined = combined.append_column(name, value[name])
        frames.append(combined.to_pandas())
    overlay_file = pq.ParquetFile(plan["execution_overlay"]["path"])
    overlay_meta = overlay_file.read(columns=list(PRICE_METADATA), use_threads=False)
    need(canonical_sha256(overlay_meta.to_pylist()) == plan["execution_overlay"]["metadata_sha256"], "Execution overlay metadata changed")
    guard_keys(overlay_meta, ("intrinio_security_id", "trading_date"), "trading_date", "2010-01-01", "2023-12-31")
    last = annual["windows"][-1]["exit_date"]
    mask = pc.and_(pc.is_in(overlay_meta["intrinio_security_id"], value_set=pa.array(sorted(ids))),
        pc.and_(pc.greater_equal(overlay_meta["trading_date"], f"{year}-01-01"), pc.less_equal(overlay_meta["trading_date"], min(last, "2023-12-31"))))
    overlay_values = overlay_file.read(columns=list(PRICE_VALUES), use_threads=False).filter(mask)
    overlay = overlay_meta.filter(mask)
    for name in PRICE_VALUES:
        overlay = overlay.append_column(name, overlay_values[name])
    prices = pd.concat(frames, ignore_index=True)
    repair = overlay.to_pandas()
    keys = ["intrinio_security_id", "trading_date"]
    need(not prices.duplicated(keys).any(), "Conflicting annual/endpoint price keys")
    # _cohort_from_metadata already read/guarded this exact target metadata and
    # proved target order equal to the original ledger. Reuse its keys rather
    # than silently decode TARGET_METADATA a second unbudgeted time.
    legacy = pa.Table.from_pandas(cohort.loc[:, list(KEYS)], preserve_index=False)
    values = pq.ParquetFile(annual["paths"]["targets"]).read(columns=list(TARGET_VALUES), use_threads=False)
    for name in TARGET_VALUES:
        legacy = legacy.append_column(name, values[name])
    return cohort, prices, legacy.to_pandas(), repair


def core_cohort(frame):
    reserved = {"outcome_month", "outcome_start", "outcome_end", "entry_date", "exit_date", "exit_mark",
        "label_available_at", "side", "path_variant", "target_family", "cash_return", "maximum_drawdown", "utility",
        "target_value", "target_class", "target_available", "path_available", "path_requested", "target_status"}
    renamed = frame.rename(columns={name: "legacy_" + name for name in frame if name in reserved})
    need(len(renamed.columns) == len(set(renamed.columns)), "Legacy metadata renaming collision")
    return pa.Table.from_pandas(renamed, preserve_index=False)


def source_request_status(plan, native, date, role, outcome_year):
    if native.startswith(("unquoted_successor:", "execution_")):
        return "identity_unresolved_not_requestable", False
    annual = next(a for a in plan["annuals"] if a["year"] == int(date[:4]))
    if native in annual["declared_cache_requested_ids"]:
        return "declared_requested", False
    return ("not_requested_boundary" if int(date[:4]) > outcome_year else
        "not_requested_action_successor" if role == "action_successor" else "not_requested_cache_route"), True


def build_month_paths(plan, cohort, price_index, window, *, resource_check=None, registry=None):
    from .full_path_accounting import replay_unit_episode
    episodes, marks, details = [], [], []
    records = pa.Table.from_pandas(cohort, preserve_index=False).to_pylist()
    for position, row in enumerate(records):
        if position % 100 == 0:
            if resource_check:
                resource_check()
            print(json.dumps({"stage": "full_path_month", "outcome_month": window["outcome_month"],
                "processed_original_members": position, "total_original_members": len(records)}), flush=True)
        for side in SIDES:
            for variant in VARIANTS:
                key = {"source_security_id": row["source_security_id"], "signal_date": row["signal_date"], "side": side, "path_variant": variant}
                if not row["prediction_eligible"]:
                    result = {"requested": False, "status": "not_requested", "reason": "signal_time_prediction_ineligible", "marks": []}
                elif window["exit_date"] >= "2024-01-01":
                    result = {"requested": True, "status": "unavailable", "reason": "protected_scheduled_endpoint", "marks": []}
                else:
                    result = replay_unit_episode(price_index, source_security_id=row["source_security_id"],
                        intrinio_security_id=row["intrinio_security_id"], signal_date=row["signal_date"], window=window,
                        side=side, path_variant=variant, events=plan["events"], registry=registry, retain_signed_wealth=True)
                    reasons = []
                    source_request_incomplete = False
                    support = [(row["intrinio_security_id"], window["entry_date"], "selected")]
                    support.extend((a["successor_id"], a["date"], "action_successor") for a in result.get("actions", []) if a.get("successor_id"))
                    if result["status"] == "complete" and not result.get("stops"):
                        final_native = (result["actions"][-1].get("successor_id") if result.get("actions") else row["intrinio_security_id"])
                        if final_native is not None:
                            support.append((final_native, window["exit_date"], "held"))
                    for native, date, role in support:
                        status, incomplete = source_request_status(plan, native, date, role, int(window["outcome_month"][:4]))
                        source_request_incomplete |= incomplete
                        if status != "declared_requested":
                            reasons.append(status)
                    for failure in result.get("failures", []):
                        if failure["reason"] != "missing_quote":
                            continue
                        native = failure["intrinio_security_id"]
                        detail, incomplete = source_request_status(plan, native, failure["date"], failure.get("role"), int(window["outcome_month"][:4]))
                        source_request_incomplete |= incomplete
                        if detail == "identity_unresolved_not_requestable":
                            reasons.append("unsupported_successor_identity")
                        elif detail == "declared_requested":
                            detail = "requested_cache_quote_missing"
                        failure["cache_request_status"] = detail
                        reasons.append(detail)
                    if reasons:
                        result["reason"] += ";" + ";".join(sorted(set(reasons)))
                    if source_request_incomplete:
                        result["requested"] = None
                episodes.append({**key, **{k: result[k] for k in ("requested", "status", "reason")},
                    "audit_requested": bool(row["prediction_eligible"])})
                marks.extend({**key, **mark} for mark in result.get("marks", []))
                details.append({**key, **{name: result.get(name) for name in
                    ("status", "reason", "stop_count", "action_count", "unfilled_entry", "engine_status", "engine_capital_gate_triggered",
                     "legacy_capital_gate_would_trigger", "negative_reference_nav_retained")},
                    "arithmetic_version": result.get("arithmetic_version"),
                    "failure_evidence_json": json.dumps(result.get("failures", []), sort_keys=True),
                    "stop_evidence_json": json.dumps(result.get("stops", []), sort_keys=True),
                    "action_evidence_json": json.dumps(result.get("actions", []), sort_keys=True)})
    mark_schema = pa.schema([(name, pa.string()) for name in (*KEYS, "side", "path_variant", "date", "mark")] + [("wealth", pa.float64())])
    return pa.Table.from_pylist(episodes), pa.Table.from_pylist(marks, schema=mark_schema), pa.Table.from_pylist(details)


def read_full_path_audit(output_directory):
    """Verify small saved reports and artifact bytes; no financial table decode."""
    root = Path(output_directory).resolve()
    report = read_json(root / "audit.json")
    need(report.get("status") == "COMPLETE_FULL_PATH_AUDIT" and report.get("protocol") == PROTOCOL, "Full audit is incomplete")
    need(report["years"] == list(YEARS) and report["original_months"] == 168, "Saved full audit has truncated annual scope")
    reviewed = read_json(root / "reviewed_plan.json")
    need(reviewed["plan_sha256"] == report["plan_sha256"] and canonical_sha256({k: v for k, v in reviewed.items() if k != "plan_sha256"}) == report["plan_sha256"],
        "Saved full plan identity changed")
    validate_saved_plan_contract(reviewed)
    need(report["original_rows"] == sum(a["original_rows"] for a in reviewed["annuals"])
        and report["prediction_eligible_rows"] == sum(a["prediction_eligible_rows"] for a in reviewed["annuals"]), "Saved full denominator mismatch")
    for year in report["years"]:
        year_root = root / "years" / str(year)
        need(year_root.resolve() == year_root and year_root.parent.resolve().parent == root, "Unsafe annual directory")
        planned_year = next(a for a in reviewed["annuals"] if a["year"] == year)
        saved_year = report["year_audit_pins"][str(year)]
        need(hash_file(year_root / "audit.json") == saved_year["sha256"] and (year_root / "audit.json").stat().st_size == saved_year["size_bytes"],
            "Annual audit receipt changed")
        receipt = read_json(root / "years" / str(year) / "audit.json")
        need(receipt["status"] == "COMPLETE_ANNUAL_PATH_AUDIT" and receipt["plan_sha256"] == report["plan_sha256"], "Annual checkpoint identity mismatch")
        expected_months = [f"{year}-{month:02d}" for month in range(1, 13)]
        need(receipt["months"] == expected_months, "Saved annual audit has missing calendar months")
        need(receipt["original_rows"] == planned_year["original_rows"] and receipt["prediction_eligible_rows"] == planned_year["prediction_eligible_rows"], "Annual denominator mismatch")
        need(set(receipt["artifacts"]) == {f"months/{month}/checkpoint.json" for month in expected_months},
            "Annual checkpoint paths differ from exact monthly contract")
        for relative, item in receipt["artifacts"].items():
            path = (root / "years" / str(year) / relative).resolve()
            need(path.is_relative_to(year_root.resolve()) and path == year_root / relative, "Unsafe checkpoint artifact path")
            need(path.stat().st_size == item["size_bytes"] and hash_file(path) == item["sha256"], "Annual audit artifact changed")
            checkpoint = read_json(path)
            attempt = (path.parent / checkpoint["attempt"]).resolve()
            need(attempt.parent == path.parent and attempt.name == checkpoint["attempt"] and attempt.name.startswith("attempt-")
                and checkpoint["plan_sha256"] == report["plan_sha256"], "Unsafe or wrong-plan monthly attempt")
            month_audit = attempt / "audit.json"
            need(hash_file(month_audit) == checkpoint["audit_sha256"], "Monthly audit changed")
            month = read_json(month_audit)
            expected_month = next(m for m in planned_year["monthly"] if m["outcome_month"] == path.parent.name)
            window = next(w for w in planned_year["windows"] if w["outcome_month"] == path.parent.name)
            need(month["status"] == "COMPLETE_MONTHLY_PATH_AUDIT" and month["plan_sha256"] == report["plan_sha256"]
                and month["protocol"] == PROTOCOL and month["year"] == year and month["outcome_month"] == path.parent.name
                and month["signal_date"] == window["signal_date"] and month["exit_date"] == window["exit_date"], "Monthly checkpoint scope mismatch")
            need(all(month[name] == expected_month[name] for name in ("original_rows", "prediction_eligible_rows", "original_key_sha256", "eligible_key_sha256")), "Monthly original denominator/key identity mismatch")
            required = {name + ".parquet" for name in ("episodes", "marks", "episode_details", "targets", "diagnostics", "legacy_alignment")}
            if month["supplemental_execution_overlay"] is not None:
                required |= {"supplemental_targets.parquet", "supplemental_episode_details.parquet"}
            need(set(month["artifacts"]) == required, "Missing or unexpected monthly evidence artifacts")
            for name, saved in month["artifacts"].items():
                target = (attempt / name).resolve()
                need(target.parent == attempt and target.name == name, "Unsafe monthly artifact path")
                need(target.stat().st_size == saved["size_bytes"] and hash_file(target) == saved["sha256"], "Monthly artifact changed")
    for name, item in report["artifacts"].items():
        target = (root / name).resolve()
        need(target.parent == root and hash_file(target) == item["sha256"] and target.stat().st_size == item["size_bytes"], "Saved aggregate artifact changed")
    return report


def validate_saved_plan_contract(plan):
    """Saved metadata contract only: no source/runtime/price reconstruction."""
    need(plan.get("schema") == SCHEMA and plan.get("protocol") == PROTOCOL
        and plan.get("protocol_id") == "full-path-target-audit-2010-2023-20261022-v1", "Saved report uses another target protocol")
    need(any(p["sha256"].removeprefix("sha256:") == CATALOG_SHA256 for p in plan["source_pins"]), "Saved report lacks exact corrected catalog")
    need(tuple(a["year"] for a in plan["annuals"]) == YEARS and set(plan["worker_sha256"]) == set(WORKERS), "Saved report has wrong annual/worker scope")
    need(plan["limits"]["peak_rss_bytes"] == 4 * 1024 ** 3 and plan["limits"]["active_wall_seconds"] == 2700
        and plan["semantics"]["model_fits"] == plan["semantics"]["portfolio_backtests"] == 0, "Saved report violates zero-fit resource contract")
    need(plan["read_plan"]["original_months"] == 168 and plan["read_plan"]["original_years"] == 14, "Saved report has truncated metadata scope")
    sessions = plan["sessions"]
    need(sessions == sorted(set(sessions)) and canonical_sha256(sessions) == plan["calendar"]["session_dates_sha256"], "Saved calendar metadata changed")
    by_month = {}
    for day in sessions:
        need(type(day) is str and pd.Timestamp(day).strftime("%Y-%m-%d") == day and "2009-12-01" <= day <= "2024-01-05", "Saved calendar date outside metadata scope")
        by_month.setdefault(day[:7], []).append(day)
    for annual in plan["annuals"]:
        expected = [f"{annual['year']}-{m:02d}" for m in range(1, 13)]
        need([w["outcome_month"] for w in annual["windows"]] == [r["outcome_month"] for r in annual["monthly"]] == expected, "Saved annual calendar is incomplete")
        need(sum(r["original_rows"] for r in annual["monthly"]) == annual["original_rows"]
            and sum(r["prediction_eligible_rows"] for r in annual["monthly"]) == annual["prediction_eligible_rows"], "Saved annual denominator metadata mismatch")
        for window in annual["windows"]:
            month = pd.Period(window["outcome_month"], "M")
            need(window["signal_date"] == by_month[str(month-1)][-1] and window["entry_date"] == by_month[str(month)][0]
                and window["exit_date"] == by_month[str(month+1)][0] and window["exit_mark"] == "open", "Saved target clock mismatch")
    return True
