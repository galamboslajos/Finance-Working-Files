"""Explicit bounded initialization audit; no model or full feature warehouse.

The historical original cohort is fixed before price joins. Exact monthly
endpoint coverage for the paper's five raw momenta is separate from outcome
endpoint availability. Common-five normalization calls the shared active engineering policy;
the audit does not fit a model or claim that the qualified subset is representative.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from importlib.metadata import version
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .catalog import canonical_sha256
from .data import BRIDGE_COLUMNS, CALENDAR_COLUMNS, LINK_COLUMNS, MEMBERSHIP_COLUMNS, session_dates
from .exchange_sessions import EXCHANGE_CALENDARS_VERSION, _package_code_hash
from .identity_policy import PROVISIONAL_IDENTITY_POLICY, qualify_identity_records
from .loading import BoundedTableReader, BudgetExceeded, ReadBudget
from .monthly import _step_month
from .paper_features import (COMMON_MOMENTUM_POLICY, HORIZONS, PAPER_FEATURE_NAMES, PRICE_FIELDS,
                             _quote_value, _raw_momentum, normalize_common_momentum)
from .universe import DataContractError, build_snapshot, iso_date


AUDIT_VERSION = 1
CATEGORIES = ("complete_five_momenta", "no_exact_signal_quote", "identity_blocked",
              "invalid_signal_quote", "valid_signal_quote_incomplete_momentum_history")
METADATA_BUDGET = ReadBudget(max_metadata_files=16, max_data_files=8,
    max_scanned_rows=20_000_000, max_output_rows=250_000, max_read_bytes=128 * 1024**2)
PRICE_BUDGET = ReadBudget(max_metadata_files=128, max_data_files=128,
    max_scanned_rows=30_000_000, max_output_rows=1_000_000, max_read_bytes=256 * 1024**2)


def _aggregate(rows):
    counts = Counter(row["coverage_category"] for row in rows)
    total = len(rows)
    summary = {"original_stock_months": total,
        "admitted_stock_months": sum(row["identity_eligible"] for row in rows),
        "complete_five_momenta": counts["complete_five_momenta"],
        "paper_normalization_diagnostic_ready": sum(row["paper_normalization_diagnostic_ready"] for row in rows),
        "paired_outcome_endpoints_available": sum(row["paired_outcome_endpoints_available"] for row in rows),
        "exclusive_categories": {key: counts[key] for key in CATEGORIES},
        "fractions_of_original": {key: counts[key] / total if total else None for key in CATEGORIES},
        "identity_status_counts": dict(Counter(row["identity_status"] for row in rows))}
    summary["admission_fraction"] = summary["admitted_stock_months"] / total if total else None
    summary["complete_five_fraction"] = counts["complete_five_momenta"] / total if total else None
    return summary


def evaluate_start_coverage(ledger, month_ends, feature_quotes, *, outcome_quotes=None):
    """Pure exact-endpoint diagnostic, retaining every original constituent.

    ``feature_quotes`` must stop at the latest signal. Outcome quotes are a
    separate argument and never select the normalization population. All signal
    cohorts must include the entire original membership, even blocked routes.
    """
    originals = [dict(row) for row in ledger]
    if not originals:
        raise DataContractError("Initialization audit requires original historical members")
    grouped = defaultdict(list)
    seen = set()
    for row in originals:
        signal = iso_date(row.get("signal_date"), "signal_date")
        if signal >= "2024-01-01" or month_ends.get(signal[:7]) != signal:
            raise DataContractError("Signal must be an exact pre-holdout month end")
        if row.get("as_of") != signal:
            raise DataContractError("Cohort as_of differs from signal")
        expected_outcome = _step_month(signal[:7], 1)
        if row.get("outcome_month") != expected_outcome or expected_outcome >= "2024-01":
            raise DataContractError("Outcome month is not the following pre-holdout month")
        if expected_outcome not in month_ends:
            raise DataContractError("Missing exact outcome month end")
        key = (signal, row.get("source_security_id"))
        if not key[1] or key in seen:
            raise DataContractError("Missing/duplicate original historical member key")
        seen.add(key)
        grouped[signal].append(row)
    latest_signal = max(grouped)
    latest_outcome = max(month_ends[row["outcome_month"]] for row in originals)
    outcome_quotes = {} if outcome_quotes is None else outcome_quotes
    for quotes, upper, label in ((feature_quotes, latest_signal, "feature"),
                                  (outcome_quotes, latest_outcome, "outcome")):
        for key, quote in quotes.items():
            if not isinstance(key, tuple) or len(key) != 2 or not key[0]:
                raise DataContractError(f"Invalid {label} quote key")
            day = iso_date(key[1], "quote trading_date")
            if day >= "2024-01-01" or day > upper:
                raise DataContractError(f"Future/protected {label} quote")
            if quote.get("intrinio_security_id") != key[0] or quote.get("trading_date") != day:
                raise DataContractError("Quote metadata differs from its key")
    rows, signal_summary = [], []
    for signal, original_group in sorted(grouped.items()):
        # Defensive admission over every original dated member, never a filtered
        # subset or a caller's stale claimed-eligible field.
        records = qualify_identity_records(original_group, identity_policy=PROVISIONAL_IDENTITY_POLICY)
        requested = {row["intrinio_security_id"] for row in records if row["identity_eligible"]}
        signal_rows = []
        for source in records:
            row = dict(source)
            sid = row["intrinio_security_id"]
            quote = feature_quotes.get((sid, signal)) if row["identity_eligible"] else None
            _, signal_status = _quote_value(quote)
            row["signal_quote_status"] = signal_status if row["identity_eligible"] else "identity_unresolved"
            raw = {h: _raw_momentum(row, signal, h, month_ends, feature_quotes, requested) for h in HORIZONS}
            if not row["identity_eligible"]:
                category = "identity_blocked"
            elif signal_status == "missing_exact_endpoint":
                category = "no_exact_signal_quote"
            elif signal_status != "ok":
                category = "invalid_signal_quote"
            elif not all(status == "ok" for _, status in raw.values()):
                category = "valid_signal_quote_incomplete_momentum_history"
            else:
                category = "complete_five_momenta"
            row["coverage_category"] = category
            row["paper_common_input_eligible"] = category == "complete_five_momenta"
            for horizon, (value, status) in raw.items():
                row[f"MOM{horizon}_raw"] = value
                row[f"MOM{horizon}_raw_status"] = status
            # Outcome metadata is intentionally evaluated only after all feature
            # eligibility and raw moments were fixed. No target is imputed.
            outcome_day = month_ends[row["outcome_month"]]
            _, outcome_status = _quote_value(outcome_quotes.get((sid, outcome_day))) if row["identity_eligible"] else (None, "identity_unresolved")
            row["outcome_endpoint_status"] = outcome_status
            row["paired_outcome_endpoints_available"] = row["identity_eligible"] and signal_status == "ok" and outcome_status == "ok"
            signal_rows.append(row)
        # Give the shared normalizer only keys, identity admission and past raw
        # momenta. Outcome fields never enter this interface.
        raw_fields = ["source_security_id", "signal_date", "identity_eligible",
                      *(f"MOM{h}_raw" for h in HORIZONS),
                      *(f"MOM{h}_raw_status" for h in HORIZONS)]
        raw_frame = pd.DataFrame([{key: row[key] for key in raw_fields} for row in signal_rows])
        normalized = normalize_common_momentum(raw_frame, exhaustive_requested_by_signal={signal: True})
        coverage_row = normalized.coverage.iloc[0]
        n = int(coverage_row.common_momentum_population_count)
        norm_status = str(coverage_row.normalization_status)
        moments = {}
        for horizon in HORIZONS:
            ready = normalized.table[normalized.table.paper_prediction_eligible]
            moments[str(horizon)] = {
                "mean": float(ready[f"meanMOM{horizon}"].iloc[0]) if len(ready) else None,
                "sample_sd": float(ready[f"stdMOM{horizon}"].iloc[0]) if len(ready) else None}
        normalized_records = normalized.table.to_dict("records")
        for row, norm in zip(signal_rows, normalized_records, strict=True):
            row["paper_prediction_eligible"] = bool(norm["paper_prediction_eligible"])
            row["paper_normalization_status"] = norm["paper_normalization_status"]
            row["paper_normalization_diagnostic_ready"] = row["paper_prediction_eligible"]
            row["normalization_diagnostic_status"] = norm_status
            for name in PAPER_FEATURE_NAMES:
                row[name] = float(norm[name]) if math.isfinite(norm[name]) else None
        signal_summary.append({"signal_date": signal, "outcome_month": signal_rows[0]["outcome_month"],
            **_aggregate(signal_rows), "common_momentum_population_count": n,
            "normalization_diagnostic_status": norm_status, "raw_common_moments": moments})
        rows.extend(signal_rows)
    years = sorted({int(row["outcome_month"][:4]) for row in rows})
    yearly = [{"outcome_year": year, **_aggregate([row for row in rows if row["outcome_month"].startswith(str(year))])}
              for year in years]
    summary = {**_aggregate(rows), "signal_count": len(grouped),
        "first_signal": min(grouped), "last_signal": max(grouped),
        "outcome_start_month": min(row["outcome_month"] for row in rows),
        "outcome_end_month": max(row["outcome_month"] for row in rows),
        "yearly": yearly, "signals": signal_summary,
        "normalization_diagnostic": {"policy": COMMON_MOMENTUM_POLICY,
            "sample_sd_ddof": 1, "all_five_horizons_share_population": True,
            "shared_active_engineering_normalizer": True,
            "outcome_availability_used_for_population": False},
        "interpretation": "Absent quotes are absent under currently admitted native IDs; historical identity correctness and other provider histories are unverified.",
        "scope_limit": "Paper raw endpoint coverage, not all 715 predictors, daily-path completeness, labels, model fitting or economic accounting."}
    if len(years) >= 11:
        first = years[0]
        summary["initial_training_years"] = [first, first + 9]
        summary["initial_training_coverage"] = _aggregate([row for row in rows if first <= int(row["outcome_month"][:4]) <= first + 9])
        summary["nominal_test_years"] = [year for year in years if year >= first + 10]
        summary["nominal_test_coverage"] = _aggregate([row for row in rows if int(row["outcome_month"][:4]) >= first + 10])
    return rows, summary


def _totals(loads, budget):
    totals = {key: sum(item.stats.get(key, 0) for item in loads.values()) for key in
              ("metadata_files", "data_files", "scanned_rows", "returned_rows", "projected_compressed_bytes")}
    limits = {"metadata_files": budget.max_metadata_files, "data_files": budget.max_data_files,
              "scanned_rows": budget.max_scanned_rows, "returned_rows": budget.max_output_rows,
              "projected_compressed_bytes": budget.max_read_bytes}
    for key, value in totals.items():
        if value > limits[key]:
            raise BudgetExceeded(f"Cumulative initialization audit budget exceeded: {key}={value}>{limits[key]}")
    return totals


def _parameters(start_year, end_year):
    if type(start_year) is not int or type(end_year) is not int or not 2000 <= start_year <= end_year <= 2023:
        raise DataContractError("Initialization audit years must be integers in 2000..2023")
    if end_year - start_year + 1 > 14:
        raise DataContractError("This measured audit permits at most 14 outcome years")
    return {"start_year": start_year, "end_year": end_year, "universe": "combined",
            "identity_policy": PROVISIONAL_IDENTITY_POLICY, "audit_version": AUDIT_VERSION}


def _code_pins():
    names = ("start_coverage.py", "paper_features.py", "identity_policy.py", "universe.py",
             "loading.py", "catalog.py", "data.py", "monthly.py", "exchange_sessions.py")
    return {name: "sha256:" + hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in names}


def _runtime_pins():
    return {"python": platform.python_version(), **{name: version(name) for name in
            ("pyarrow", "pandas", "numpy", "exchange-calendars", "tzdata")}}


def _hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _calendar_parameters(start_year, end_year):
    if version("exchange-calendars") != EXCHANGE_CALENDARS_VERSION:
        raise DataContractError("Independent exchange calendar differs from reviewed pin")
    import exchange_calendars as xc
    first_signal_month = f"{start_year-1}-12"
    first_quote_month = f"{start_year-2}-12"
    last_signal_month = f"{end_year}-11"
    cal = xc.get_calendar("XNYS", start=first_quote_month + "-01", end=f"{end_year+1}-01-31")
    if type(cal).__module__ != "exchange_calendars.exchange_calendar_xnys":
        raise DataContractError("Calendar registry differs from reviewed XNYS implementation")
    dates = tuple(cal.sessions.strftime("%Y-%m-%d"))
    month_ends = {day[:7]: day for day in dates}
    signals = tuple(day for month, day in sorted(month_ends.items()) if first_signal_month <= month <= last_signal_month)
    if len(signals) != (end_year - start_year + 1) * 12:
        raise DataContractError("Independent calendar did not produce all required monthly signals")
    endpoints = tuple(day for month, day in sorted(month_ends.items()) if first_quote_month <= month <= f"{end_year}-12")
    return dates, month_ends, signals, endpoints


def _request_pin(summary, signals, endpoints, requested):
    return canonical_sha256({"parameters": summary["parameters"], "signals": signals,
        "endpoint_dates": endpoints, "provider_ids": requested, "source_pins": summary["source_pins"],
        "source_project_root": summary["source_project_root"], "code_sha256": summary["code_sha256"],
        "runtime": summary["runtime"], "calendar_provenance": summary["calendar_provenance"]})


def run_start_coverage(catalog, output_dir, *, start_year=2010, end_year=2023):
    """Run the explicit, pinned, measured monthly-endpoint audit once.

    This function is never called by ordinary notebook Run All. It projects only
    exact month ends, not a daily panel. Source blocks can transiently decode
    excluded dates inside Arrow; existing reader masks precede Python conversion.
    """
    parameters = _parameters(start_year, end_year)
    out = Path(output_dir)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise DataContractError("Coverage output already exists; preserve it and choose a new version directory")
    started = time.monotonic()
    # Pin imported helpers BEFORE any financial read. A concurrent code edit
    # invalidates the run, so its evidence cannot describe a different worker.
    code_pins = _code_pins()
    runtime_pins = _runtime_pins()
    dates, month_ends, signals, endpoint_dates = _calendar_parameters(start_year, end_year)
    reader = BoundedTableReader(catalog, METADATA_BUDGET)
    loads = {}
    loads["calendar"] = reader.read_metadata("siblis", "trading_calendar_sessions", columns=CALENDAR_COLUMNS,
        allowed_columns=CALENDAR_COLUMNS, max_rows=20_000, key_columns=("trading_date",))
    source_sessions = set(session_dates(loads["calendar"].table))
    if not set(signals) <= source_sessions:
        raise DataContractError("Independent month-end membership date absent from source calendar")
    loads["membership"] = reader.read("siblis", "membership_by_index_by_date", start=signals[0], end=signals[-1],
        columns=MEMBERSHIP_COLUMNS, allowed_columns=MEMBERSHIP_COLUMNS,
        key_columns=("index_id", "security_id", "trading_date"),
        filters={"index_id": ("sp500", "sp400"), "trading_date": signals}, financial=False)
    loads["bridge"] = reader.read_metadata("identity", "security_identity_bridge", columns=BRIDGE_COLUMNS,
        allowed_columns=BRIDGE_COLUMNS, max_rows=30_000, key_columns=("source_product_id", "source_security_id"))
    loads["links"] = reader.read_metadata("intrinio", "intrinio_identity_links", columns=LINK_COLUMNS,
        allowed_columns=LINK_COLUMNS, max_rows=20_000, key_columns=("unified_security_id",))
    metadata_totals = _totals(loads, METADATA_BUDGET)
    ledger, exclusions = [], []
    for signal in signals:
        members = loads["membership"].table.filter(pc.equal(loads["membership"].table["trading_date"], signal))
        snapshot = build_snapshot(members, loads["bridge"].table, loads["links"].table,
            as_of=signal, universe="combined", mode="historical_reconstruction")
        if not snapshot.table.num_rows:
            raise DataContractError("No original historical members at a required signal date")
        original = qualify_identity_records(snapshot.table.to_pylist(), identity_policy=PROVISIONAL_IDENTITY_POLICY)
        ledger.extend({**row, "signal_date": signal, "outcome_month": _step_month(signal[:7], 1)} for row in original)
        exclusions.extend(snapshot.excluded.to_pylist())
    requested = tuple(sorted({row["intrinio_security_id"] for row in ledger if row["identity_eligible"]}))
    loaded = BoundedTableReader(catalog, PRICE_BUDGET).read("intrinio", "intrinio_daily_prices",
        start=endpoint_dates[0], end=endpoint_dates[-1], columns=PRICE_FIELDS, allowed_columns=PRICE_FIELDS,
        key_columns=("intrinio_security_id", "trading_date"),
        filters={"intrinio_security_id": requested, "trading_date": endpoint_dates}, financial=True)
    # Arrow split before any financial rows reach Python; outcome data are never
    # passed to feature arithmetic beyond the latest declared signal.
    feature_table = loaded.table.filter(pc.less_equal(loaded.table["trading_date"], signals[-1]))
    feature_quotes = {(row["intrinio_security_id"], row["trading_date"]): row for row in feature_table.to_pylist()}
    outcome_dates = tuple(month_ends[_step_month(signal[:7], 1)] for signal in signals)
    outcome_table = loaded.table.filter(pc.is_in(loaded.table["trading_date"], value_set=pa.array(outcome_dates)))
    outcome_quotes = {(row["intrinio_security_id"], row["trading_date"]): row for row in outcome_table.to_pylist()}
    rows, summary = evaluate_start_coverage(ledger, month_ends, feature_quotes, outcome_quotes=outcome_quotes)
    if _code_pins() != code_pins or _runtime_pins() != runtime_pins:
        raise DataContractError("Coverage code/runtime changed during execution; rerun in a new version directory")
    out.mkdir(parents=True, exist_ok=True)
    ledger_path = out / "original_ledger.parquet"
    pq.write_table(pa.Table.from_pylist(rows), ledger_path, compression="zstd")
    year_frame = pd.DataFrame([{key: value for key, value in row.items() if not isinstance(value, dict)} for row in summary["yearly"]])
    yearly_path = out / "yearly.csv"
    year_frame.to_csv(yearly_path, index=False)
    import exchange_calendars as xc
    package_files, package_hash = _package_code_hash(xc)
    summary.update({"audit_version": AUDIT_VERSION, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "identity_policy": PROVISIONAL_IDENTITY_POLICY, "universe": "combined",
        "parameters": parameters, "source_project_root": str(catalog.root),
        "membership_exclusion_count": len(exclusions),
        "membership_exclusions_sha256": canonical_sha256(exclusions),
        "requested_provider_id_count": len(requested), "requested_endpoint_date_count": len(endpoint_dates),
        "feature_quote_row_count": feature_table.num_rows, "outcome_quote_row_count": outcome_table.num_rows,
        "feature_request_first_date": endpoint_dates[0], "feature_request_last_date": signals[-1],
        "outcome_request_last_date": endpoint_dates[-1],
        "source_pins": catalog.provenance(), "metadata_loading_stats": {key: value.stats for key, value in loads.items()},
        "metadata_loading_totals": metadata_totals, "price_loading_stats": loaded.stats,
        "metadata_loading_provenance": {key: value.provenance for key, value in loads.items()},
        "price_loading_provenance": loaded.provenance,
        "calendar_provenance": {"policy": "exchange_calendars_XNYS", "package_version": version("exchange-calendars"),
            "code_sha256": package_hash, "source_file_sha256": package_files,
            "month_ends_sha256": canonical_sha256(month_ends),
            "source_only_dates": sorted(day for day in source_sessions - set(dates) if dates[0] <= day <= dates[-1]),
            "independent_only_dates": sorted(set(dates) - source_sessions),
            "selection_independence": "No financial observations or eventual survivor availability."},
        "code_sha256": code_pins,
        "runtime": runtime_pins,
        "elapsed_seconds": time.monotonic() - started,
        "knowledge_time_verified": False, "historical_identity_verified": False,
        "all_predictor_coverage_verified": False, "daily_backtest_ready": False,
        "artifacts": {path.name: {"sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(), "size_bytes": path.stat().st_size}
                      for path in (ledger_path, yearly_path)}})
    summary["request_sha256"] = _request_pin(summary, signals, endpoint_dates, requested)
    if _code_pins() != code_pins or _runtime_pins() != runtime_pins:
        raise DataContractError("Coverage code/runtime changed before evidence write; choose a new version directory")
    with (out / "evidence.json").open("x") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return summary


def read_saved_start_coverage(catalog, output_dir, *, start_year=2010, end_year=2023):
    """Verify immutable evidence and return its summary without a price rescan.

    Artifact hashes, exact code/runtime/source pins and the original ledger's
    dates, years and denominators must match. Source objects are checked against
    pinned sizes; this does not reread all raw-object hashes on every Run All.
    """
    parameters = _parameters(start_year, end_year)
    out = Path(output_dir)
    try:
        summary = json.loads((out / "evidence.json").read_text())
    except (OSError, ValueError) as error:
        raise DataContractError("Saved initialization coverage evidence is missing or invalid; run an explicit new audit") from error
    if summary.get("parameters") != parameters:
        raise DataContractError("Saved coverage parameters differ; run a new versioned audit")
    if summary.get("source_project_root") != str(catalog.root) or summary.get("source_pins") != catalog.provenance():
        raise DataContractError("Saved coverage source root or release pins differ")
    if summary.get("code_sha256") != _code_pins() or summary.get("runtime") != _runtime_pins():
        raise DataContractError("Saved coverage code/runtime pins differ; run a new versioned audit")
    expected_artifacts = {"original_ledger.parquet", "yearly.csv"}
    if set(summary.get("artifacts", {})) != expected_artifacts:
        raise DataContractError("Saved coverage artifact inventory differs")
    for name, expected in summary["artifacts"].items():
        path = out / name
        if not path.is_file() or path.is_symlink() or path.stat().st_size != expected.get("size_bytes") or _hash_file(path) != expected.get("sha256"):
            raise DataContractError(f"Saved coverage artifact integrity mismatch: {name}")
    provenance_groups = list(summary.get("metadata_loading_provenance", {}).values()) + [summary.get("price_loading_provenance", {})]
    for provenance in provenance_groups:
        if "source_key" not in provenance or "table" not in provenance or not provenance.get("objects"):
            raise DataContractError("Saved coverage source-object provenance is incomplete")
        available = {obj.relative_path: obj for obj in catalog.table_objects(provenance["source_key"], provenance["table"])}
        for pin in provenance["objects"]:
            obj = available.get(pin.get("relative_path"))
            if obj is None or obj.size_bytes != pin.get("size_bytes") or obj.sha256 != pin.get("sha256"):
                raise DataContractError("Saved coverage source-object pin differs")
            obj.check()
    dates, month_ends, signals, endpoints = _calendar_parameters(start_year, end_year)
    import exchange_calendars as xc
    _, current_calendar_hash = _package_code_hash(xc)
    calendar = summary.get("calendar_provenance", {})
    if (calendar.get("code_sha256") != current_calendar_hash or
            calendar.get("month_ends_sha256") != canonical_sha256(month_ends)):
        raise DataContractError("Saved independent-calendar evidence differs")
    columns = ["source_security_id", "signal_date", "outcome_month", "intrinio_security_id",
        "identity_eligible", "identity_status", "coverage_category",
        "paper_normalization_diagnostic_ready", "paper_prediction_eligible", "paired_outcome_endpoints_available"]
    try:
        # Only audit metadata enter Python. The financial feature columns are
        # verified as artifact bytes but are unnecessary for displaying coverage.
        table = pq.read_table(out / "original_ledger.parquet", columns=columns)
        rows = table.to_pylist()
    except (pa.ArrowException, KeyError) as error:
        raise DataContractError("Saved original ledger schema is invalid") from error
    if not rows or set(row["signal_date"] for row in rows) != set(signals):
        raise DataContractError("Saved ledger does not contain every required monthly cohort")
    keys = set()
    for row in rows:
        key = (row["signal_date"], row["source_security_id"])
        if not key[1] or key in keys or row["outcome_month"] != _step_month(row["signal_date"][:7], 1):
            raise DataContractError("Saved original ledger member/chronology defect")
        keys.add(key)
        if row["signal_date"] >= "2024-01-01" or not start_year <= int(row["outcome_month"][:4]) <= end_year:
            raise DataContractError("Saved ledger enters protected/out-of-scope dates")
        if row["paper_prediction_eligible"] != row["paper_normalization_diagnostic_ready"]:
            raise DataContractError("Saved normalized eligibility disagrees with coverage")
    actual = _aggregate(rows)
    if any(summary.get(key) != value for key, value in actual.items()):
        raise DataContractError("Saved coverage totals differ from original ledger")
    expected_years = list(range(start_year, end_year + 1))
    if [row.get("outcome_year") for row in summary.get("yearly", [])] != expected_years:
        raise DataContractError("Saved yearly coverage does not match the requested history")
    for yearly in summary["yearly"]:
        actual = _aggregate([row for row in rows if int(row["outcome_month"][:4]) == yearly["outcome_year"]])
        if any(yearly.get(key) != value for key, value in actual.items()):
            raise DataContractError("Saved yearly coverage differs from original ledger")
    signal_evidence = summary.get("signals", [])
    if [row.get("signal_date") for row in signal_evidence] != list(signals):
        raise DataContractError("Saved per-signal coverage does not match the requested history")
    for item in signal_evidence:
        actual = _aggregate([row for row in rows if row["signal_date"] == item["signal_date"]])
        if any(item.get(key) != value for key, value in actual.items()):
            raise DataContractError("Saved signal coverage differs from original ledger")
    if end_year >= start_year + 10:
        if summary.get("initial_training_years") != [start_year, start_year + 9] or summary.get("nominal_test_years") != list(range(start_year + 10, end_year + 1)):
            raise DataContractError("Saved rolling training/test chronology differs")
        for name, selected in (
            ("initial_training_coverage", [row for row in rows if int(row["outcome_month"][:4]) <= start_year + 9]),
            ("nominal_test_coverage", [row for row in rows if int(row["outcome_month"][:4]) >= start_year + 10])):
            if summary.get(name) != _aggregate(selected):
                raise DataContractError("Saved rolling training/test coverage differs from original ledger")
    actual_csv = pd.read_csv(out / "yearly.csv")
    expected_csv = pd.DataFrame([{key: value for key, value in row.items() if not isinstance(value, dict)} for row in summary["yearly"]])
    try:
        if set(actual_csv) != set(expected_csv):
            raise AssertionError("CSV column set differs")
        pd.testing.assert_frame_equal(actual_csv[sorted(actual_csv)], expected_csv[sorted(expected_csv)],
                                      check_dtype=False, rtol=1e-14, atol=1e-14)
    except AssertionError as error:
        raise DataContractError("Saved yearly CSV differs from original-ledger evidence") from error
    requested = tuple(sorted({row["intrinio_security_id"] for row in rows if row["identity_eligible"]}))
    if summary.get("request_sha256") != _request_pin(summary, signals, endpoints, requested):
        raise DataContractError("Saved coverage request pin differs")
    return summary
