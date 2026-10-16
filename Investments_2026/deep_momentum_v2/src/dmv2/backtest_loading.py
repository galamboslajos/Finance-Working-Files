"""Metadata-first cache inputs for the bounded 2018–2019 daily development run.

This module never reads target classes, source price shards, fitted estimators,
feature matrices, or later-year financial columns. Quote availability cannot
select an original cohort or replace a failed holding.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .catalog import canonical_sha256
from .exchange_sessions import EXCHANGE_CALENDARS_VERSION, _package_code_hash
from .identity_policy import PROVISIONAL_IDENTITY_POLICY, qualify_identity_records
from .universe import DataContractError, iso_date

ARMS = ("rf_paper15", "xgb_paper15", "rf_extended", "xgb_extended")
YEARS = (2018, 2019)
KEYS = ("source_security_id", "signal_date")
PROBABILITIES = tuple(f"probability_{i}" for i in range(1, 11))
IDENTITY_FIELDS = ("intrinio_security_id", "identity_status", "source_identity_status",
    "identity_eligible", "identity_eligibility_policy", "identity_eligibility_reason",
    "identity_review_required", "resolution_status")
PREDICTION_METADATA = (*KEYS, "outcome_month", "outcome_start", "outcome_end",
    *IDENTITY_FIELDS, "requested", "paper_prediction_eligible", "prediction_eligible",
    "outcome_requested", "return_available", "observed_return_available", "target_available",
    "target_status", "return_observation_status", "source_return_status", "feature_complete",
    "model_ready", "knowledge_time_verified", "accounting_certified", "available_feature_count",
    "registered_feature_count", "feature_status", "prediction_status")
PRICE_KEYS = ("trading_date", "intrinio_security_id")
PRICE_COLUMNS = (*PRICE_KEYS, "adj_open", "adj_high", "adj_low", "adj_close", "frequency", "intraperiod",
    "price_quality_status", "adjusted_ohlc_bounds_valid", "raw_ohlc_bounds_valid",
    "factor", "split_ratio", "dividend")
DEFAULT_MAX_ROWS = 1_000_000
DEFAULT_MAX_PROJECTED_BYTES = 128 * 1024 * 1024


@dataclass(frozen=True)
class BacktestDataPlan:
    root: Path
    pilot_directory: Path
    prediction_paths: dict
    daily_paths: dict
    prediction_metadata: dict
    daily_metadata: dict
    requested_ids: tuple
    sessions: tuple
    windows: pd.DataFrame
    source_pins: tuple
    read_plan: dict
    provenance: dict
    fingerprint: str

    @property
    def summary(self):
        return self.read_plan


@dataclass(frozen=True)
class BacktestDataBundle:
    predictions: dict
    daily_prices: pd.DataFrame
    sessions: tuple
    windows: pd.DataFrame
    plan: BacktestDataPlan
    provenance: dict


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _within(root, path):
    root, path = Path(root).resolve(), Path(path).resolve()
    if not path.is_relative_to(root):
        raise DataContractError("Daily backtest artifact leaves its declared root")
    return path


def _json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError) as error:
        raise DataContractError("Missing or invalid daily backtest input manifest") from error


def _pin(root, path):
    root = Path(root).resolve()
    path = _within(root, path)
    if not path.is_file():
        raise DataContractError("Required saved backtest input is absent: " + str(path))
    return {"path": path.relative_to(root).as_posix(), "size_bytes": path.stat().st_size,
            "sha256": _hash(path)}


def _verify_pin(root, item):
    if not isinstance(item, dict) or not {"path", "size_bytes", "sha256"} <= set(item):
        raise DataContractError("Incomplete daily backtest artifact pin")
    path = _within(root, Path(root) / item["path"])
    if (not path.is_file() or path.stat().st_size != item["size_bytes"]
            or _hash(path) != item["sha256"]):
        raise DataContractError("Daily backtest input differs from its saved hash/size: " + item["path"])
    return path


def _artifact(root, directory, item, pins):
    path = _verify_pin(directory, item)
    pins.append(_pin(root, path))
    return path


def _table_hash(table):
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema.remove_metadata()) as writer:
        writer.write_table(table.replace_schema_metadata(None))
    return "sha256:" + hashlib.sha256(sink.getvalue().to_pybytes()).hexdigest()


def _projected_bytes(path, columns):
    file = pq.ParquetFile(path)
    if len(file.schema_arrow.names) != len(set(file.schema_arrow.names)) or not set(columns) <= set(file.schema_arrow.names):
        raise DataContractError("Backtest cache has missing or duplicate projected columns")
    return sum(file.metadata.row_group(group).column(column).total_compressed_size
        for group in range(file.metadata.num_row_groups)
        for column in range(file.metadata.num_columns)
        if file.metadata.schema.column(column).path.split(".")[0] in columns)


def _metadata(path, columns):
    _projected_bytes(path, columns)
    return pq.ParquetFile(path).read(columns=list(columns))


def _independent_clock():
    if version("exchange-calendars") != EXCHANGE_CALENDARS_VERSION:
        raise DataContractError("Daily backtest requires the reviewed exchange-calendars pin")
    import exchange_calendars as xcals
    calendar = xcals.get_calendar("XNYS", start="2017-12-01", end="2019-12-31")
    if calendar.name != "XNYS" or type(calendar).__module__ != "exchange_calendars.exchange_calendar_xnys":
        raise DataContractError("Unexpected independent XNYS implementation")
    dates = tuple(day for day in calendar.sessions.strftime("%Y-%m-%d")
                  if "2017-12-01" <= day <= "2019-12-31")
    sessions = tuple(day for day in dates if "2018-01-01" <= day <= "2019-12-31")
    windows = []
    months = [str(month) for month in pd.period_range("2018-01", "2019-12", freq="M")]
    for index, month in enumerate(months):
        previous = str(pd.Period(month, "M") - 1)
        current = [day for day in dates if day[:7] == month]
        preceding = [day for day in dates if day[:7] == previous]
        if not current or not preceding:
            raise DataContractError("Independent XNYS clock lacks a required month")
        following = str(pd.Period(month, "M") + 1)
        next_dates = [day for day in sessions if day[:7] == following]
        windows.append({"signal_date": preceding[-1], "outcome_month": month,
            "outcome_start": current[0], "outcome_end": current[-1],
            "entry_date": current[0], "execution_date": current[0],
            "exit_date": next_dates[0] if index < 23 else current[-1],
            "exit_price_field": "adj_open" if index < 23 else "adj_close"})
    _, code_hash = _package_code_hash(xcals)
    return sessions, pd.DataFrame(windows), {"calendar": "XNYS", "package_version": EXCHANGE_CALENDARS_VERSION,
        "calendar_code_sha256": code_hash, "sessions_sha256": canonical_sha256(list(sessions)),
        "calendar_start": "2017-12-01", "calendar_end": "2019-12-31",
        "selection_sessions": list(dates),
        "selection_sessions_sha256": canonical_sha256(list(dates)),
        "calendar_source": "independent_rule_based_calendar_not_quote_inventory"}


def _validate_prediction_metadata(table, windows):
    rows = table.to_pylist()
    key_list = [(row["source_security_id"], row["signal_date"]) for row in rows]
    if not key_list or len(set(key_list)) != len(key_list):
        raise DataContractError("Prediction ledger lost unique original cohort keys")
    expected = windows.set_index("signal_date").to_dict("index")
    if set(row["signal_date"] for row in rows) != set(expected):
        raise DataContractError("Prediction dates exceed or omit approved 2018–2019 signals")
    for row in rows:
        if not isinstance(row["source_security_id"], str) or not row["source_security_id"].strip():
            raise DataContractError("Prediction ledger has a null/blank source ID")
        iso_date(row["signal_date"], "signal_date")
        clock = expected[row["signal_date"]]
        if any(row[field] != clock[field] for field in ("outcome_month", "outcome_start", "outcome_end")):
            raise DataContractError("Prediction metadata differs from independent next-month clock")
        for field in ("identity_eligible", "requested", "paper_prediction_eligible", "prediction_eligible"):
            if type(row[field]) is not bool:
                raise DataContractError("Signal-time eligibility requires explicit booleans")
        if row["prediction_eligible"] != row["paper_prediction_eligible"]:
            raise DataContractError("Model prediction cohort differs from frozen Paper15 cohort")
        if row["prediction_eligible"] and (not row["identity_eligible"] or not row["requested"]
                or not isinstance(row["intrinio_security_id"], str) or not row["intrinio_security_id"].strip()):
            raise DataContractError("Prediction-eligible row lacks an admitted requested stable ID")
        if row["knowledge_time_verified"] is not False or row["accounting_certified"] is not False:
            raise DataContractError("Saved provisional predictions claim unsupported certification")
    return key_list


def _snapshot(plan):
    return {"root": str(plan.root), "pilot_directory": str(plan.pilot_directory),
        "prediction_paths": {name: str(path) for name, path in plan.prediction_paths.items()},
        "daily_paths": {str(year): str(path) for year, path in plan.daily_paths.items()},
        "prediction_metadata": {name: _table_hash(table) for name, table in plan.prediction_metadata.items()},
        "daily_metadata": {str(year): _table_hash(table) for year, table in plan.daily_metadata.items()},
        "requested_ids": list(plan.requested_ids), "sessions": list(plan.sessions),
        "windows": plan.windows.to_dict("records"), "source_pins": list(plan.source_pins),
        "read_plan": plan.read_plan, "provenance": plan.provenance}


def plan_daily_backtest(root, *, max_rows=DEFAULT_MAX_ROWS,
        max_projected_bytes=DEFAULT_MAX_PROJECTED_BYTES):
    """Verify hashes, clocks, admission, projections and budgets before values.

    The two annual outcome-only caches are selected from the completed pilot's
    frozen input plan. Missing requested quotes remain missing; no raw fallback
    or later-year cache is consulted, including for final liquidation.
    """
    root = Path(root if isinstance(root, (str, Path)) else root.root).resolve()
    if type(max_rows) is not int or max_rows <= 0 or type(max_projected_bytes) is not int or max_projected_bytes <= 0:
        raise DataContractError("Daily backtest budgets must be positive integers")
    parent = root / "outputs/model_pilot"
    pointer_path = parent / "LATEST.json"
    pointer = _json(pointer_path)
    directory = _within(parent, parent / pointer.get("path", ""))
    audit_path = directory / "audit.json"
    if not audit_path.is_file() or _hash(audit_path) != pointer.get("audit_sha256"):
        raise DataContractError("Classifier pilot pointer has an invalid root audit hash")
    audit = _json(audit_path)
    if audit.get("status") != "COMPLETE" or audit.get("build_identity_sha256") != pointer.get("build_identity_sha256"):
        raise DataContractError("Classifier pilot is incomplete or identity changed")
    pins = [_pin(root, pointer_path), _pin(root, audit_path)]
    identity_path = _artifact(root, directory, audit["artifacts"]["build_identity.json"], pins)
    identity = _json(identity_path)
    if canonical_sha256(identity) != pointer["build_identity_sha256"]:
        raise DataContractError("Classifier pilot build identity is inconsistent")
    configuration = identity.get("configuration", {})
    expected_scope = {"universe": "combined", "identity_policy": PROVISIONAL_IDENTITY_POLICY,
        "context_policy": "available_cohort_80pct", "train_outcome_start": "2010-01-01",
        "train_outcome_end": "2017-12-31", "validation_outcome_start": "2018-01-01",
        "validation_outcome_end": "2019-12-31"}
    if any(configuration.get(name) != value for name, value in expected_scope.items()):
        raise DataContractError("Saved pilot differs from the fixed four-arm development scope")
    sessions, windows, calendar_provenance = _independent_clock()
    predictions, metadata, projections = {}, {}, []
    first = None
    for arm in ARMS:
        arm_audit_path = _artifact(root, directory, audit["arms"][arm], pins)
        arm_audit = _json(arm_audit_path)
        if arm_audit.get("status") != "COMPLETE" or arm_audit.get("arm", {}).get("arm_id") != arm or arm_audit.get("build_identity_sha256") != pointer["build_identity_sha256"]:
            raise DataContractError("Classifier arm is incomplete or differs from pilot identity")
        path = _artifact(root, arm_audit_path.parent, arm_audit["artifacts"]["predictions"], pins)
        file = pq.ParquetFile(path)
        item = arm_audit["artifacts"]["predictions"]
        if file.metadata.num_rows != item.get("rows") or file.schema_arrow.names != item.get("columns"):
            raise DataContractError("Saved prediction schema/row count differs from its audit")
        table = _metadata(path, PREDICTION_METADATA)
        _validate_prediction_metadata(table, windows)
        if first is not None and not table.equals(first):
            raise DataContractError("Four classifier arms changed original cohorts or metadata flags")
        first = table if first is None else first
        predictions[arm], metadata[arm] = path, table
        projections.extend(((path, PREDICTION_METADATA), (path, PROBABILITIES)))
    requested_ids = tuple(sorted({row["intrinio_security_id"] for row in first.to_pylist() if row["prediction_eligible"]}))
    if not requested_ids:
        raise DataContractError("No signal-time prediction-eligible stable IDs")
    years = identity.get("data_plan", {}).get("years", {})
    daily_paths, daily_metadata, year_reports = {}, {}, {}
    for year in YEARS:
        frozen = years.get(str(year), {})
        cache = _within(root, root / frozen.get("feature_directory", ""))
        frozen_pins = {item["path"]: item for item in frozen.get("source_pins", [])}
        relative_audit = (cache / "audit.json").relative_to(root).as_posix()
        if relative_audit not in frozen_pins:
            raise DataContractError("Pilot lacks a pinned annual cache audit")
        annual_path = _verify_pin(root, frozen_pins[relative_audit]); pins.append(_pin(root, annual_path))
        annual = _json(annual_path)
        if annual.get("year") != year or annual.get("feature_provenance", {}).get("identity_policy") != PROVISIONAL_IDENTITY_POLICY:
            raise DataContractError("Annual cache year/provisional identity policy changed")
        provenance = annual["feature_provenance"]
        if provenance.get("mode") != "historical_reconstruction":
            raise DataContractError("Annual cache lacks historical reconstruction contract")
        year_windows = windows[windows.outcome_month.str.startswith(str(year))]
        expected_windows = {row["signal_date"]: {field: row[field] for field in ("outcome_month", "outcome_start", "outcome_end")}
            for row in year_windows.to_dict("records")}
        if provenance.get("target_windows") != expected_windows:
            raise DataContractError("Annual cache clock differs from independent exchange sessions")
        artifacts = annual["artifacts"]
        original_path = _artifact(root, cache, artifacts["original_cohort"], pins)
        coverage_path = _artifact(root, cache, artifacts["feature_coverage"], pins)
        original_columns = (*KEYS, "as_of", "unified_security_id", "intrinio_security_id", "identity_status",
                            "identity_review_required", "resolution_status")
        original = _metadata(original_path, original_columns)
        coverage_columns = (*KEYS, *IDENTITY_FIELDS, "requested", "paper_prediction_eligible")
        coverage = _metadata(coverage_path, coverage_columns)
        projections.extend(((original_path, original_columns), (coverage_path, coverage_columns)))
        subset = first.filter(pc.starts_with(first["outcome_month"], str(year)))
        keys = list(zip(subset[KEYS[0]].to_pylist(), subset[KEYS[1]].to_pylist()))
        if (keys != list(zip(original[KEYS[0]].to_pylist(), original[KEYS[1]].to_pylist()))
                or keys != list(zip(coverage[KEYS[0]].to_pylist(), coverage[KEYS[1]].to_pylist()))):
            raise DataContractError("Saved predictions lost ordered original annual cohort keys")
        qualified = qualify_identity_records(original.to_pylist(), identity_policy=PROVISIONAL_IDENTITY_POLICY)
        declared_requests = provenance.get("requested_ids")
        if (not isinstance(declared_requests, list) or len(set(declared_requests)) != len(declared_requests)
                or any(not isinstance(value, str) or not value.strip() for value in declared_requests)):
            raise DataContractError("Annual cache lacks its pre-price stable-ID request declaration")
        for original_row, admitted, source, prediction in zip(original.to_pylist(), qualified, coverage.to_pylist(), subset.to_pylist(), strict=True):
            if original_row["as_of"] != original_row["signal_date"]:
                raise DataContractError("Original historical cohort differs from signal cutoff")
            for field in IDENTITY_FIELDS:
                if admitted.get(field) != source[field] or source[field] != prediction[field]:
                    raise DataContractError("Prediction metadata changed original identity admission: " + field)
            for field in ("requested", "paper_prediction_eligible"):
                if source[field] != prediction[field]:
                    raise DataContractError("Prediction metadata changed frozen signal eligibility")
            if prediction["prediction_eligible"] and prediction["intrinio_security_id"] not in declared_requests:
                raise DataContractError("Eligible prediction omitted from original pre-price request")
        path = _artifact(root, cache, artifacts["daily_prices_raw"], pins)
        file = pq.ParquetFile(path)
        if file.metadata.num_rows != artifacts["daily_prices_raw"].get("rows") or file.schema_arrow.names != artifacts["daily_prices_raw"].get("columns"):
            raise DataContractError("Outcome-only daily cache differs from its saved schema/rows")
        daily_keys = _metadata(path, PRICE_KEYS)
        frame = daily_keys.to_pandas()
        if frame.isna().any().any() or frame.duplicated(list(PRICE_KEYS)).any():
            raise DataContractError("Daily cache contains null or duplicate stable-ID/date keys")
        for day in frame.trading_date.unique():
            if not f"{year}-01-01" <= iso_date(day, "trading_date") <= f"{year}-12-31":
                raise DataContractError("Daily cache would decode financial values outside its approved outcome year")
        if any(not isinstance(value, str) or not value.strip() for value in frame.intrinio_security_id.unique()):
            raise DataContractError("Daily cache has invalid stable security IDs")
        selected = frame.trading_date.isin(sessions) & frame.intrinio_security_id.isin(requested_ids)
        daily_paths[year], daily_metadata[year] = path, daily_keys
        projections.extend(((path, PRICE_KEYS), (path, PRICE_COLUMNS)))
        year_reports[str(year)] = {"daily_cache_path": path.relative_to(root).as_posix(),
            "file_bytes": path.stat().st_size, "candidate_rows": len(frame),
            "selected_rows": int(selected.sum()), "first_date": frame.trading_date.min(),
            "last_date": frame.trading_date.max(), "original_rows": len(subset),
            "prediction_eligible_rows": sum(subset["prediction_eligible"].to_pylist()),
            "pre_price_requested_ids": len(declared_requests),
            "non_exchange_rows_excluded_after_metadata_check": int((~frame.trading_date.isin(sessions)).sum())}
    projected = sum(_projected_bytes(path, columns) for path, columns in projections)
    rows = sum(pq.ParquetFile(path).metadata.num_rows for path in {path for path, _ in projections})
    if rows > max_rows or projected > max_projected_bytes:
        raise DataContractError("Measured daily backtest reads exceed the explicit row/byte ceiling")
    read_plan = {"outcome_years": list(YEARS), "original_rows_per_arm": len(first),
        "prediction_eligible_rows_per_arm": sum(first["prediction_eligible"].to_pylist()),
        "prediction_arms": list(ARMS), "signal_dates": windows.signal_date.tolist(),
        "independent_sessions": len(sessions), "requested_stable_ids": len(requested_ids),
        "candidate_rows": rows, "daily_candidate_rows": sum(report["candidate_rows"] for report in year_reports.values()),
        "daily_selected_rows": sum(report["selected_rows"] for report in year_reports.values()),
        "projected_compressed_bytes": projected, "max_rows": max_rows,
        "max_projected_bytes": max_projected_bytes, "price_projection": list(PRICE_COLUMNS),
        "years": year_reports, "financial_values_read": False,
        "financial_2020_plus_read": False, "target_classes_read": False,
        "feature_matrices_read": False, "serialized_models_read": False, "raw_source_fallback": False,
        "integrity_hashing_separate_from_projected_read_bytes": True}
    provenance = {"protocol": "saved_four_arm_2018_2019_daily_cache_inputs_v1",
        "pilot_build_identity_sha256": pointer["build_identity_sha256"],
        "pilot_audit_sha256": pointer["audit_sha256"], "calendar": calendar_provenance,
        "identity_policy": PROVISIONAL_IDENTITY_POLICY, "all_original_rows_retained": True,
        "eligibility_from_signal_metadata_only": True, "future_quote_availability_used_for_selection": False,
        "no_target_classes_or_return_values_loaded": True, "knowledge_time_verified": False,
        "accounting_certified": False, "final_liquidation": "2019-12-31_adj_close_no_2020_open"}
    pins = tuple({item["path"]: item for item in pins}.values())
    plan = BacktestDataPlan(root, directory, predictions, daily_paths, metadata, daily_metadata,
        requested_ids, sessions, windows, pins, read_plan, provenance, "")
    object.__setattr__(plan, "fingerprint", canonical_sha256(_snapshot(plan)))
    return plan


def verify_backtest_inputs(plan):
    """Recheck planned immutable inputs; hashing never decodes numerical values."""
    if not isinstance(plan, BacktestDataPlan) or canonical_sha256(_snapshot(plan)) != plan.fingerprint:
        raise DataContractError("Daily backtest metadata plan changed after validation")
    for item in plan.source_pins:
        _verify_pin(plan.root, item)


def load_daily_backtest(root, *, plan=None):
    """Materialize only the unchanged measured 2018–2019 cache projection."""
    root = Path(root if isinstance(root, (str, Path)) else root.root).resolve()
    plan = plan_daily_backtest(root) if plan is None else plan
    if not isinstance(plan, BacktestDataPlan) or plan.root != root:
        raise DataContractError("Daily backtest plan belongs to another root")
    verify_backtest_inputs(plan)
    predictions = {}
    for arm in ARMS:
        table = pq.ParquetFile(plan.prediction_paths[arm]).read(columns=list(PROBABILITIES))
        frame = plan.prediction_metadata[arm].to_pandas()
        values = table.to_pandas().to_numpy(dtype=float)
        eligible = frame.prediction_eligible.to_numpy(dtype=bool)
        if (len(values) != len(frame) or not np.isfinite(values[eligible]).all()
                or (values[eligible] < 0).any() or (values[eligible] > 1).any()
                or not np.allclose(values[eligible].sum(axis=1), 1.0, rtol=0, atol=1e-8)
                or not np.isnan(values[~eligible]).all()):
            raise DataContractError("Saved probabilities violate frozen eligibility/ten-class contract")
        for index, name in enumerate(PROBABILITIES):
            frame[name] = values[:, index]
        predictions[arm] = frame
    daily = []
    for year in YEARS:
        table = pq.ParquetFile(plan.daily_paths[year]).read(columns=list(PRICE_COLUMNS))
        if not table.select(list(PRICE_KEYS)).equals(plan.daily_metadata[year]):
            raise DataContractError("Daily cache keys changed after the approved metadata read")
        mask = pc.and_(pc.is_in(table["trading_date"], value_set=pa.array(plan.sessions)),
            pc.is_in(table["intrinio_security_id"], value_set=pa.array(plan.requested_ids)))
        daily.append(table.filter(mask).to_pandas())
    prices = pd.concat(daily, ignore_index=True).sort_values(list(PRICE_KEYS), ignore_index=True)
    if len(prices) != plan.read_plan["daily_selected_rows"] or prices.duplicated(list(PRICE_KEYS)).any():
        raise DataContractError("Bounded daily materialization changed measured keys/rows")
    verify_backtest_inputs(plan)
    provenance = {**plan.provenance, "plan_fingerprint": plan.fingerprint,
        "read_plan": {**plan.read_plan, "financial_values_read": True}, "source_files_unchanged": True,
        "no_price_or_target_based_prediction_filter": True, "no_price_filling": True,
        "financial_2020_plus_read": False, "models_fitted": False}
    return BacktestDataBundle(predictions, prices, plan.sessions, plan.windows.copy(deep=True), plan, provenance)
