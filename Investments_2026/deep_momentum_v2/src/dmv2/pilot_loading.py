"""Cache-only, metadata-first inputs for the fixed 2010–2019 classifier pilot.

Annual adapters limit peak memory and prevent 2020+ numerical values from being
decoded. The ten years are split by completed OUTCOME dates, retaining missing
labels and every original source-ID/signal row. No preprocessing is fitted here.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .catalog import canonical_sha256
from .classification import classification_training_slice
from .context_panel_roles import (PRIMARY_POLICY, _role_configuration, _verify_recipe)
from .features import FeaturePanel, KEYS, TrainingSlice
from .full_panel import _hash, _specs, _verify_artifacts, _verify_year, _within
from .full_target_panel import (SMALL_REPORTS, YEARS as FULL_TARGET_YEARS,
    _load_target_state, _year_audit, build_identity as target_build_identity,
    verify_preserved_inputs)
from .paper_features import PAPER_FEATURE_NAMES
from .targets import SAFE_METADATA
from .universe import DataContractError, iso_date


YEARS = tuple(range(2010, 2020))
TRAIN_YEARS = tuple(range(2010, 2018))
VALIDATION_YEARS = (2018, 2019)
ADMISSION = ("identity_eligible", "requested", "paper_prediction_eligible")
# These are status/count fields, never predictor or return values. Keep every
# original flag rather than reducing the saved ledger to admitted observations.
COVERAGE_METADATA = (*KEYS, "intrinio_security_id", "identity_status",
    "source_identity_status", "identity_eligible", "identity_eligibility_policy",
    "identity_eligibility_reason", "identity_review_required", "resolution_status",
    "paper_prediction_eligible", "requested", "knowledge_time_verified",
    "accounting_certified", "available_feature_count", "registered_feature_count",
    "feature_status", "model_ready")
TARGET_METADATA = (*SAFE_METADATA, "target_available")


@dataclass(frozen=True)
class PilotYearPlan:
    year: int
    feature_directory: Path
    target_directory: Path
    feature_audit: dict
    target_audit: dict
    coverage: pa.Table
    windows: dict
    source_pins: tuple
    read_plan: dict


@dataclass(frozen=True)
class PilotDataPlan:
    root: Path
    years: dict
    read_plan: dict
    provenance: dict
    fingerprint: str

    @property
    def summary(self):
        return self.read_plan


@dataclass(frozen=True)
class PilotDataBundle:
    train: TrainingSlice
    validation: TrainingSlice
    plan: PilotDataPlan
    provenance: dict


def _read_json(path):
    return json.loads(Path(path).read_text())


def _context_run_metadata(catalog, pin):
    """The frozen role reader's checks, without loading any report values."""
    directory = _within(catalog.root / "outputs", catalog.root / pin["directory"])
    audit_path = directory / "audit.json"
    if not audit_path.is_file() or _hash(audit_path) != pin["audit_sha256"]:
        raise DataContractError("Context role audit differs from its explicit pin")
    report = _read_json(audit_path)
    identity = _read_json(directory / "build_identity.json")
    identity_hash = pin["build_identity_sha256"]
    if canonical_sha256(identity) != identity_hash:
        raise DataContractError("Context role identity differs from its explicit pin")
    if (report.get("status") != "COMPLETE" or report.get("build_identity_sha256") != identity_hash
            or report.get("configuration") != identity.get("configuration")
            or identity.get("configuration", {}).get("context_policy") != pin["context_policy"]
            or any(report.get(key) is not False for key in
                   ("models_fitted", "imputation_fitted", "model_ready", "daily_backtest_ready"))
            or set(report.get("years", {})) != {str(year) for year in FULL_TARGET_YEARS}):
        raise DataContractError("Context role is not a complete diagnostic-only 2010–2023 run")
    if not {"build_identity.json", "feature_dictionary.json"} <= set(report.get("reports", {})):
        raise DataContractError("Context role is missing recipe/dictionary pins")
    _verify_artifacts(directory, report["reports"])
    _verify_artifacts(directory, report["years"])
    for year in FULL_TARGET_YEARS:
        if _verify_year(directory / "years" / str(year), identity_hash).get("year") != year:
            raise DataContractError("Context role annual partition is mislabelled")
    contract = identity.get("active_reference_contract", identity)
    dictionary_hash = canonical_sha256(_read_json(directory / "feature_dictionary.json"))
    summary = report["summary"]
    if (dictionary_hash != contract.get("feature_dictionary_sha256")
            or summary.get("feature_dictionary_sha256") != dictionary_hash
            or summary.get("original_stock_months") != 152307
            or summary.get("paper_qualified_stock_months") != 140777
            or summary.get("completed_signals") != 168):
        raise DataContractError("Context role differs from the verified cohort/dictionary scope")
    return {"directory": directory, "audit": report,
            "verification_mode": "pinned_immutable_role_metadata_only", "reference_pins": pin}, identity


def read_pilot_context_metadata(catalog):
    """Verify roles, archived recipes and comparison hashes without CSV reads.

    The ordinary notebook reader intentionally displays historical comparison
    reports, including numerical feature summaries after 2019. A pilot must not
    call that reporting reader. Here those files are integrity-hashed only.
    """
    roles = _role_configuration(catalog)
    primary, primary_identity = _context_run_metadata(catalog, roles["primary"])
    sensitivity, sensitivity_identity = _context_run_metadata(catalog, roles["sensitivity"])
    _verify_recipe(catalog, roles, primary_identity, sensitivity_identity)
    directory = primary["directory"] / "comparison"
    path = directory / "comparison_manifest.json"
    if not path.is_file() or _hash(path) != roles["comparison_manifest_sha256"]:
        raise DataContractError("Context comparison differs from its explicit manifest pin")
    manifest = _read_json(path)
    for name, run in (("reference_audit", sensitivity), ("sensitivity_audit", primary)):
        if manifest[name]["sha256"] != _hash(run["directory"] / "audit.json"):
            raise DataContractError("Context comparison uses a different completed panel")
    for name, item in manifest["reports"].items():
        artifact = _within(directory, directory / name)
        if (not artifact.is_file() or artifact.stat().st_size != item["bytes"]
                or _hash(artifact) != item["sha256"]):
            raise DataContractError("Context comparison report changed: " + name)
    if "comparison_summary.json" not in manifest["reports"]:
        raise DataContractError("Context comparison summary lacks a hash pin")
    return {"primary": primary, "sensitivity": sensitivity, "roles": roles,
        "verification_mode": "metadata_and_integrity_hashes_only_no_report_values"}


def read_pilot_target_metadata(catalog):
    """Verify full target persistence without decoding labels or reports."""
    identity = target_build_identity(catalog)
    pin = canonical_sha256(identity)
    parent = _within(catalog.root, catalog.root / "outputs/full_target_panel")
    pointer = _read_json(parent / "LATEST.json")
    directory = _within(parent, parent / pointer["path"])
    if (pointer.get("build_identity_sha256") != pin
            or _hash(directory / "audit.json") != pointer["audit_sha256"]):
        raise DataContractError("Full target pointer changed or is stale")
    audit = _read_json(directory / "audit.json")
    if (audit.get("status") != "COMPLETE" or audit.get("build_identity_sha256") != pin
            or set(audit.get("years", {})) != {str(year) for year in FULL_TARGET_YEARS}):
        raise DataContractError("Full target run is stale or incomplete")
    _verify_artifacts(directory, audit["artifacts"])
    _verify_artifacts(directory, audit["years"])
    required = {"monthly_returns", "targets", "target_diagnostics", "target_state",
        "build_identity.json", "source_plan.json", "preserved_source_manifest.json",
        "ret_month_class_grid", "ret_class_payoffs", *SMALL_REPORTS}
    if (not required <= set(audit["artifacts"])
            or _read_json(directory / "build_identity.json") != identity):
        raise DataContractError("Full target run is missing required artifacts or its current identity")
    for year in FULL_TARGET_YEARS:
        _year_audit(directory / "years" / str(year), pin)
    verify_preserved_inputs(catalog.root, _read_json(directory / "preserved_source_manifest.json"))
    return {"directory": directory, "audit": audit,
        "verification_mode": "metadata_and_integrity_hashes_only_no_report_values"}


def _artifact_path(directory, item):
    path = _within(directory, directory / item["path"])
    _verify_artifacts(directory, {item["path"]: item})
    return path


def _projected_bytes(path, columns):
    parquet = pq.ParquetFile(path)
    if not set(columns) <= set(parquet.schema_arrow.names):
        raise DataContractError("Pilot projection is missing registered columns")
    return sum(parquet.metadata.row_group(group).column(column).total_compressed_size
               for group in range(parquet.metadata.num_row_groups)
               for column in range(parquet.metadata.num_columns)
               if parquet.metadata.schema.column(column).path.split(".")[0] in columns)


def _metadata(path, columns):
    # Explicit projections are important: a mixed row group can contain financial
    # columns for an unauthorized date even when a later pandas filter removes it.
    return pq.ParquetFile(path).read(columns=list(columns))


def _keys(table, *, label):
    frame = table.select(list(KEYS)).to_pandas()
    if frame.isna().any().any() or frame.duplicated(list(KEYS)).any():
        raise DataContractError(label + " must retain unique original source-ID/signal keys")
    for sid, signal in frame.itertuples(index=False, name=None):
        if not isinstance(sid, str) or not sid.strip():
            raise DataContractError(label + " contains an invalid source security ID")
        if not "2009-12-01" <= iso_date(signal, "signal_date") < "2019-12-01":
            raise DataContractError("Pilot metadata exceeds the approved 2010–2019 outcomes")
    return pd.MultiIndex.from_frame(frame)


def _year_windows(provenance, year):
    windows = provenance.get("target_windows")
    signals = provenance.get("signal_dates")
    expected = {f"{year}-{month:02d}" for month in range(1, 13)}
    if not isinstance(windows, dict) or not isinstance(signals, list) or set(windows) != set(signals):
        raise DataContractError("Pilot feature cache lacks exact independent target windows")
    months = set()
    for signal, window in windows.items():
        signal = iso_date(signal, "signal_date")
        start = iso_date(window["outcome_start"], "outcome_start")
        end = iso_date(window["outcome_end"], "outcome_end")
        following = str(pd.Period(signal[:7], "M") + 1)
        if not (signal < start <= end and start[:7] == end[:7] == following
                and window["outcome_month"] == following):
            raise DataContractError("Pilot target windows overlap or change the next-month clock")
        months.add(window["outcome_month"])
    if len(windows) != 12 or months != expected:
        raise DataContractError("Pilot annual cache must contain exactly its twelve approved outcome months")
    return windows


def _pin(root, path):
    return {"path": str(path.relative_to(root)), "size_bytes": path.stat().st_size,
            "sha256": _hash(path)}


def _year_snapshot(root, annual):
    return {"read_plan": annual.read_plan, "source_pins": list(annual.source_pins),
        "target_windows": annual.windows,
        "feature_directory": str(_within(root, annual.feature_directory).relative_to(root)),
        "target_directory": str(_within(root, annual.target_directory).relative_to(root)),
        "coverage_metadata_sha256": canonical_sha256(annual.coverage.to_pylist())}


def _year_plan(root, primary, target_run, year, specs):
    feature_root = _within(root, primary["directory"])
    target_root = _within(root, target_run["directory"])
    feature_directory = feature_root / "years" / str(year)
    target_directory = target_root / "years" / str(year)
    feature_audit_path = _artifact_path(feature_root, primary["audit"]["years"][str(year)])
    target_audit_path = _artifact_path(target_root, target_run["audit"]["years"][str(year)])
    if (feature_audit_path != feature_directory / "audit.json"
            or target_audit_path != target_directory / "audit.json"):
        raise DataContractError("Pilot annual manifest redirects to another year")
    fa, ta = _read_json(feature_audit_path), _read_json(target_audit_path)
    if (type(fa.get("year")) is not int or fa["year"] != year
            or type(ta.get("year")) is not int or ta["year"] != year
            or ta.get("status") != "COMPLETE"
            or fa.get("build_identity_sha256") != primary["audit"]["build_identity_sha256"]
            or ta.get("build_identity_sha256") != target_run["audit"]["build_identity_sha256"]):
        raise DataContractError("Pilot annual manifest is stale or mislabelled")
    feature_names = [*KEYS, *(spec["name"] for spec in specs)]
    fpaths = {name: _artifact_path(feature_directory, fa["artifacts"][name])
              for name in ("features", "feature_coverage", "original_cohort")}
    tpaths = {name: _artifact_path(target_directory, ta["artifacts"][name])
              for name in ("targets", "target_diagnostics", "target_state")}
    if (any(path != feature_directory / (name + ".parquet") for name, path in fpaths.items())
            or any(path != target_directory / (name + (".json" if name == "target_state" else ".parquet"))
                   for name, path in tpaths.items())):
        raise DataContractError("Pilot annual manifest redirects a fixed cache artifact")
    if (set(pq.ParquetFile(fpaths["features"]).schema_arrow.names) != set(feature_names)
            or canonical_sha256(specs) != fa["feature_provenance"].get("feature_dictionary_sha256")):
        raise DataContractError("Pilot source differs from the registered 728-feature dictionary")
    windows = _year_windows(fa["feature_provenance"], year)
    feature_keys = _metadata(fpaths["features"], KEYS)
    coverage = _metadata(fpaths["feature_coverage"], COVERAGE_METADATA)
    original = _metadata(fpaths["original_cohort"], KEYS)
    target_meta = _metadata(tpaths["targets"], TARGET_METADATA)
    diagnostic_meta = _metadata(tpaths["target_diagnostics"], ("signal_date", "outcome_month"))
    expected_keys = _keys(feature_keys, label="Pilot features")
    for label, table in (("coverage", coverage), ("original cohort", original), ("targets", target_meta)):
        if set(_keys(table, label=label)) != set(expected_keys):
            raise DataContractError("Pilot cache changed the original historical cohort denominator")
    if set(feature_keys["signal_date"].to_pylist()) != set(windows):
        raise DataContractError("Pilot cache omits an independently declared signal cohort")
    cf = coverage.to_pandas().set_index(list(KEYS))
    tm = target_meta.to_pandas().set_index(list(KEYS)).reindex(cf.index)
    for name in (*ADMISSION, "target_available", "prediction_eligible"):
        frame = tm if name in ("target_available", "prediction_eligible") else cf
        if not frame[name].map(lambda value: isinstance(value, (bool, np.bool_))).all():
            raise DataContractError("Pilot admission and availability metadata must be explicit booleans")
    for name in ADMISSION:
        if not tm[name].equals(cf[name]):
            raise DataContractError("Pilot feature and target admission flags disagree")
    if (not tm.prediction_eligible.equals(cf.paper_prediction_eligible)
            or (tm.target_available & ~tm.prediction_eligible).any()
            or (cf.paper_prediction_eligible & ~(cf.identity_eligible & cf.requested)).any()):
        raise DataContractError("Future target availability changed the Paper15 prediction cohort")
    for row in target_meta.select([*KEYS, "outcome_month", "outcome_start", "outcome_end"]).to_pylist():
        expected_window = windows.get(row["signal_date"])
        if expected_window is None or any(row[name] != expected_window[name]
                for name in ("outcome_month", "outcome_start", "outcome_end")):
            raise DataContractError("Pilot target metadata exceeds its approved independent windows")
    expected_months = {window["outcome_month"] for window in windows.values()}
    if (diagnostic_meta.num_rows != 12
            or set(diagnostic_meta["outcome_month"].to_pylist()) != expected_months
            or set(diagnostic_meta["signal_date"].to_pylist()) != set(windows)):
        raise DataContractError("Pilot target diagnostics contain unapproved financial months")
    state = _read_json(tpaths["target_state"])
    if (state.get("artifact_sha256") != ta.get("target_artifact_sha256")
            or state.get("provenance", {}).get("label_observed_through", "9999") > f"{year}-12-31"):
        raise DataContractError("Pilot target state has stale identity or future observation bounds")
    declared = state.get("provenance", {}).get("month_windows", [])
    if (not isinstance(declared, list) or len(declared) != 12
            or {row["signal_date"]: {name: row[name] for name in
                ("outcome_month", "outcome_start", "outcome_end")} for row in declared} != windows):
        raise DataContractError("Pilot target state declares unapproved financial months")
    projections = ((fpaths["features"], feature_names), (fpaths["features"], KEYS),
        (fpaths["feature_coverage"], COVERAGE_METADATA), (fpaths["original_cohort"], KEYS),
        (tpaths["targets"], TARGET_METADATA),
        (tpaths["targets"], pq.ParquetFile(tpaths["targets"]).schema_arrow.names),
        (tpaths["target_diagnostics"], ("signal_date", "outcome_month")),
        (tpaths["target_diagnostics"], pq.ParquetFile(tpaths["target_diagnostics"]).schema_arrow.names))
    read_plan = {"year": year, "original_rows": len(cf),
        "prediction_eligible_rows": int(cf.paper_prediction_eligible.sum()),
        "labelled_rows": int(tm.target_available.sum()), "signal_dates": sorted(windows),
        "outcome_months": sorted(expected_months), "registered_feature_columns": len(specs),
        "model_feature_columns": sum(spec.get("model_eligible", True) for spec in specs),
        "projected_compressed_bytes": sum(_projected_bytes(path, columns) for path, columns in projections),
        "metadata_only": True, "financial_values_read": False}
    paths = [feature_audit_path, target_audit_path, *fpaths.values(), *tpaths.values()]
    return PilotYearPlan(year, feature_directory, target_directory, fa, ta, coverage,
        windows, tuple(_pin(root, path) for path in paths), read_plan)


def plan_pilot_data(catalog, *, roles=None, target_run=None):
    """Measure approved annual projections and verify clocks before any values.

    Optional run objects must come from the metadata-only verified readers. Only their
    pinned manifests are used; saved full-period feature summaries cannot select
    predictors or determine a fit mask. Hashing source bytes is separate from
    projected-column decoding and does not inspect numerical future values.
    """
    root = Path(catalog.root).resolve()
    roles = read_pilot_context_metadata(catalog) if roles is None else roles
    target_run = read_pilot_target_metadata(catalog) if target_run is None else target_run
    primary = roles["primary"]
    if roles.get("roles", {}).get("primary_policy") != PRIMARY_POLICY:
        raise DataContractError("The classifier pilot requires the selected 80% primary context panel")
    for run in (primary, target_run):
        if _read_json(run["directory"] / "audit.json") != run["audit"]:
            raise DataContractError("Pilot input manifest changed after its verified read")
    specs = _specs()
    years = {year: _year_plan(root, primary, target_run, year, specs) for year in YEARS}
    def total(years_, field):
        return sum(years[year].read_plan[field] for year in years_)
    rows = total(YEARS, "original_rows")
    # One annual Arrow/temporary copy plus persistent pandas slices; 4x the
    # full eligible numeric matrix is deliberately conservative for this loader.
    memory = rows * (715 * 8 * 4 + 4096)
    read_plan = {"outcome_years": list(YEARS), "train_outcome_years": list(TRAIN_YEARS),
        "validation_outcome_years": list(VALIDATION_YEARS),
        "rows_original_train": total(TRAIN_YEARS, "original_rows"),
        "rows_original_validation": total(VALIDATION_YEARS, "original_rows"),
        "rows_eligible_train": total(TRAIN_YEARS, "prediction_eligible_rows"),
        "rows_eligible_validation": total(VALIDATION_YEARS, "prediction_eligible_rows"),
        "rows_labelled_train": total(TRAIN_YEARS, "labelled_rows"),
        "rows_labelled_validation": total(VALIDATION_YEARS, "labelled_rows"),
        "projected_compressed_bytes": total(YEARS, "projected_compressed_bytes"),
        "estimated_peak_memory_bytes": memory, "signals": 120,
        "registered_feature_columns": 728, "model_feature_columns": 715,
        "excluded_registered_columns": [spec["name"] for spec in specs if not spec.get("model_eligible", True)],
        "feature_projection": "all_registered_728_for_frozen_table_schema_then_eligible_715_for_X",
        "financial_values_read": False, "financial_2020_plus_read": False,
        "saved_financial_report_values_read": False,
        "raw_source_reads": False, "full_global_target_table_read": False,
        "integrity_hashing_separate_from_projected_read_bytes": True}
    provenance = {"protocol": "fixed_2010_2017_train_2018_2019_validation_cache_inputs_v1",
        "read_plan": read_plan, "years": {str(year): _year_snapshot(root, plan)
            for year, plan in years.items()},
        "primary_audit_sha256": _hash(primary["directory"] / "audit.json"),
        "target_audit_sha256": _hash(target_run["directory"] / "audit.json"),
        "feature_dictionary_sha256": canonical_sha256(specs),
        "metadata_guards_before_financial_decode": True,
        "saved_financial_report_values_read": False,
        "same_classification_adapter_for_all_arms": True,
        "knowledge_time_verified": False, "accounting_certified": False}
    return PilotDataPlan(root, years, read_plan, provenance, canonical_sha256(provenance))


def _concat_slices(parts, *, start, end):
    X = pd.concat([part.X for part in parts])
    y = pd.concat([part.y for part in parts])
    ledger = pd.concat([part.ledger for part in parts])
    if not X.index.is_unique or not y.index.equals(X.index) or not ledger.index.equals(X.index):
        raise DataContractError("Annual pilot slices changed the original unique keys or row alignment")
    provenance = dict(parts[0].provenance)
    observed = max(part.provenance["label_observed_through"] for part in parts)
    provenance.update(train_start=start, train_end=end, label_observed_through=observed,
        rows_retained=len(X), original_keys_sha256=canonical_sha256([list(key) for key in X.index]),
        source_month_target_fingerprints={month: pin for part in parts
            for month, pin in part.provenance["source_month_target_fingerprints"].items()},
        source_annual_classification_slices_sha256=canonical_sha256([part.provenance for part in parts]),
        financial_2020_plus_read=False, no_preprocessing_fit_performed=True,
        estimator_fitted=False)
    return TrainingSlice(X, y, ledger, provenance)


def load_pilot_data(catalog, *, plan=None):
    """Read only measured 2010–2019 annual caches; never fit or source-fallback."""
    plan = plan_pilot_data(catalog) if plan is None else plan
    if (not isinstance(plan, PilotDataPlan) or plan.root != Path(catalog.root).resolve()
            or canonical_sha256(plan.provenance) != plan.fingerprint
            or tuple(plan.years) != YEARS):
        raise DataContractError("Pilot materialization requires the unchanged approved metadata plan")
    specs = _specs()
    if canonical_sha256(specs) != plan.provenance["feature_dictionary_sha256"]:
        raise DataContractError("Pilot feature dictionary changed after planning")
    parts = {}
    for year, annual in plan.years.items():
        if (annual.year != year or _year_snapshot(plan.root, annual) != plan.provenance["years"][str(year)]):
            raise DataContractError("Pilot annual metadata or paths changed after planning")
        _verify_artifacts(plan.root, {item["path"]: item for item in annual.source_pins})
        table = pq.ParquetFile(annual.feature_directory / "features.parquet").read(
            columns=[*KEYS, *(spec["name"] for spec in specs)])
        # All rows in each numerical file were admitted by its metadata-only
        # clock/key checks. No filter on future price or target existence occurs.
        panel = FeaturePanel(table, annual.coverage, specs, {}, {},
            {"target_windows": annual.windows, "source": "verified_primary80_annual_cache"})
        targets = _load_target_state(annual.target_directory)
        observed = min(f"{year}-12-31", targets.provenance["label_observed_through"])
        part = classification_training_slice(panel, targets,
            train_start=f"{year}-01-01", train_end=f"{year}-12-31",
            label_observed_through=observed)
        coverage = annual.coverage.to_pandas().set_index(list(KEYS)).reindex(part.X.index)
        extra_flags = [name for name in coverage if name not in part.ledger]
        part = replace(part, ledger=part.ledger.join(coverage[extra_flags]))
        if (len(part.X) != annual.read_plan["original_rows"]
                or int(part.ledger.paper_prediction_eligible.sum()) != annual.read_plan["prediction_eligible_rows"]
                or int(part.ledger.target_available.sum()) != annual.read_plan["labelled_rows"]):
            raise DataContractError("Pilot values changed the planned original/eligible/labelled denominators")
        parts[year] = part
        _verify_artifacts(plan.root, {item["path"]: item for item in annual.source_pins})
        del table, panel, targets
        gc.collect()
    train = _concat_slices([parts[y] for y in TRAIN_YEARS], start="2010-01-01", end="2017-12-31")
    validation = _concat_slices([parts[y] for y in VALIDATION_YEARS], start="2018-01-01", end="2019-12-31")
    if (set(train.X.index) & set(validation.X.index)
            or train.ledger.outcome_end.max() >= validation.ledger.outcome_start.min()
            or not set(PAPER_FEATURE_NAMES) <= set(train.X)
            or list(train.X) != list(validation.X)):
        raise DataContractError("Pilot train/validation clocks, original keys or predictor schemas overlap")
    provenance = {"plan_fingerprint": plan.fingerprint, "plan": plan.provenance,
        "read_plan": {**plan.read_plan, "financial_values_read": True},
        "train_slice_sha256": canonical_sha256(train.provenance),
        "validation_slice_sha256": canonical_sha256(validation.provenance),
        "same_cohorts_for_all_arms": True, "all_original_rows_retained": True,
        "financial_2020_plus_read": False, "raw_source_fallback": False,
        "source_files_unchanged": True, "preprocessing_fitted": False,
        "estimator_fitted": False, "knowledge_time_verified": False,
        "accounting_certified": False}
    return PilotDataBundle(train, validation, plan, provenance)
