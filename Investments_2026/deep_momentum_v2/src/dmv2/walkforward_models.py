"""Sixteen fixed rolling-ten-year fits and monthly2020--2023 forecasts.

Every financial read follows explicit annual metadata/identity/window checks.
An estimator is persisted immediately after fitting, before inference or score
generation. A started fit without a complete model receipt is never repeated
automatically; completed models can resume inference without another fit.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
import gc
from importlib.metadata import version
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .catalog import canonical_sha256
from .classification import classification_training_slice
from .classifier_pilot import ARMS, _finite32, _ledger, _predictions, _ResourceGuard
from .features import FeaturePanel, KEYS
from .full_panel import (_artifact, _hash, _json, _limits, _progress, _specs,
    _verify_artifacts, _within, _write_table)
from .full_target_panel import _load_target_state
from .identity_amendment import SOURCE_ROUTES
from .paper_features import PAPER_FEATURE_NAMES
from .pilot_loading import (ADMISSION, COVERAGE_METADATA, TARGET_METADATA,
    _concat_slices, _metadata, _projected_bytes, _year_windows)
from .pilot_models import (_constructor, _json_parameters, _matrix, _parameters,
    _sha, _training_labels)
from .preprocessing import fit_preprocessor
from .ret import CLASS_IDS
from .universe import DataContractError, iso_date
from .validation_reprediction import predict_saved_estimator
from .walkforward_metrics import evaluate_probabilities

YEARS = (2020, 2021, 2022, 2023)
PHASE = "walkforward_models_v1"
REPORTS = ("row_scores", "monthly", "classwise_calibration", "toplabel_calibration",
    "extreme_precision", "mode_class_performance", "confusion")
WORKERS = ("walkforward_models.py", "walkforward_metrics.py", "walkforward_loading.py",
    "classifier_pilot.py", "pilot_models.py", "pilot_loading.py", "classification.py",
    "preprocessing.py", "features.py", "paper_features.py", "targets.py", "ret.py",
    "validation_reprediction.py", "full_panel.py", "full_target_panel.py", "catalog.py", "universe.py")


@dataclass(frozen=True)
class AnnualInputs:
    year: int
    directory: Path
    audit: dict
    coverage: object
    windows: dict
    pins: tuple
    metadata: dict


def _configuration(root, cfg):
    execution = cfg.get("execution", {})
    if (cfg.get("protocol_id") != "dmv2-walkforward-2020-2023-v1"
            or cfg.get("test_years") != list(YEARS)
            or cfg.get("context_policy") != "available_cohort_80pct"
            or cfg.get("universe") != "historical_sp500_sp400_union"
            or execution.get("maximum_new_model_fits") != 16
            or execution.get("model_maximum_total_fits") != 16
            or execution.get("model_fits_per_fold") != 4
            or execution.get("maximum_new_preprocessor_fits") != 8
            or execution.get("workers") != 1 or execution.get("model_threads") != 1
            or execution.get("financial_2020_2023_analysis_authorized") is not True
            or any(execution.get(key) is not False for key in ("financial_2024_plus_analysis",
                "parameter_search", "early_stopping", "probability_calibration", "cloud_mutations",
                "automatic_notebook_build_or_fit"))):
        raise DataContractError("Walk-forward fitting needs the frozen16-fit/four-fold authority")
    expected_folds = [{"test_year": year, "train_outcome_start": f"{year-10}-01-01",
        "train_outcome_end": f"{year-1}-12-31", "prediction_outcome_start": f"{year}-01-01",
        "prediction_outcome_end": f"{year}-12-31",
        "label_observed_through": "last_XNYS_session_on_or_before_training_year_end"} for year in YEARS]
    if cfg.get("folds") != expected_folds:
        raise DataContractError("Walk-forward folds must each use exactly the preceding120 completed outcome months")
    caps = {"model_maximum_wall_seconds_per_fold": 5400,
        "model_aggregate_maximum_wall_seconds": 14400,
        "model_maximum_peak_rss_bytes": 8 * 1024**3,
        "model_maximum_projected_read_bytes": 1024**3,
        "maximum_output_bytes": 12 * 1024**3}
    for name, cap in caps.items():
        if type(execution.get(name)) is not int or not 0 < execution[name] <= cap:
            raise DataContractError("Walk-forward model execution ceiling is missing or too large: " + name)
    path = Path(root) / "configs/classifier_pilot.json"
    recipe = json.loads(path.read_text())
    if (cfg.get("model_recipe") != recipe
            or cfg.get("model_recipe_source", {}).get("sha256") != "sha256:" + _hash(path)
            or recipe.get("arms") != list(ARMS)):
        raise DataContractError("Walk-forward estimator template differs from the exact frozen development recipe")
    for package, expected in recipe["packages"].items():
        if version(package) != expected:
            raise DataContractError("Walk-forward package differs from the frozen pin: " + package)
    return recipe


def _keys(table, year, label):
    frame = table.select(list(KEYS)).to_pandas()
    if frame.isna().any().any() or frame.duplicated(list(KEYS)).any():
        raise DataContractError(label + " lost unique original source/signal keys")
    for sid, signal in frame.itertuples(index=False, name=None):
        if not isinstance(sid, str) or not sid.strip():
            raise DataContractError(label + " has a missing stable source identity")
        signal = iso_date(signal, "signal_date")
        if (not "2009-12-01" <= signal < "2023-12-01"
                or str(pd.Period(signal[:7], "M") + 1)[:4] != str(year)):
            raise DataContractError(label + " exceeds its explicitly approved outcome year")
    return pd.MultiIndex.from_frame(frame)


def _annual_inputs(root, reference, year):
    if type(year) is not int or not 2010 <= year <= 2023:
        raise DataContractError("Annual model inputs are restricted to2010--2023 outcomes")
    directory = _within(root, Path(reference["directory"]).resolve())
    audit = reference["audit"]
    if (json.loads((directory / "audit.json").read_text()) != audit
            or audit.get("status") != "COMPLETE" or audit.get("year") != year
            or audit.get("build_identity_sha256") != reference["build_identity_sha256"]):
        raise DataContractError("Walk-forward annual resolver supplied a stale or misidentified source")
    specs = _specs()
    names = [*KEYS, *(spec["name"] for spec in specs)]
    paths = {}
    for name in ("features", "feature_coverage", "original_cohort", "targets", "target_diagnostics", "target_state"):
        item = audit["artifacts"][name]
        _verify_artifacts(directory, {name: item})
        path = _within(directory, directory / item["path"])
        if path != directory / (name + (".json" if name == "target_state" else ".parquet")):
            raise DataContractError("Annual model manifest redirects a fixed input artifact")
        paths[name] = path
    if (set(pq.ParquetFile(paths["features"]).schema_arrow.names) != set(names)
            or audit["feature_provenance"].get("feature_dictionary_sha256") != canonical_sha256(specs)):
        raise DataContractError("Annual model input changes the frozen728-feature dictionary")
    windows = _year_windows(audit["feature_provenance"], year)
    feature_keys = _metadata(paths["features"], KEYS)
    coverage = _metadata(paths["feature_coverage"], COVERAGE_METADATA)
    original = _metadata(paths["original_cohort"], KEYS)
    target = _metadata(paths["targets"], TARGET_METADATA)
    key_index = _keys(feature_keys, year, "Features")
    for label, table in (("Coverage", coverage), ("Original cohort", original), ("Targets", target)):
        if set(_keys(table, year, label)) != set(key_index):
            raise DataContractError("Annual model inputs changed an original historical cohort denominator")
    if set(feature_keys["signal_date"].to_pylist()) != set(windows):
        raise DataContractError("Annual model input lost an independently declared monthly signal")
    cf = coverage.to_pandas().set_index(list(KEYS))
    tm = target.to_pandas().set_index(list(KEYS)).reindex(cf.index)
    for field in (*ADMISSION, "prediction_eligible", "target_available"):
        frame = tm if field in ("prediction_eligible", "target_available") else cf
        if not frame[field].map(lambda value: isinstance(value, (bool, np.bool_))).all():
            raise DataContractError("Annual admissions and class availability must be explicit booleans")
    if (any(not cf[field].equals(tm[field]) for field in ADMISSION)
            or not cf.paper_prediction_eligible.equals(tm.prediction_eligible)
            or (tm.target_available & ~tm.prediction_eligible).any()
            or (cf.paper_prediction_eligible & ~(cf.identity_eligible & cf.requested)).any()):
        raise DataContractError("Future outcomes changed the shared signal-time Paper15 cohort")
    for row in target.select([*KEYS, "outcome_month", "outcome_start", "outcome_end"]).to_pylist():
        if any(row[field] != windows[row["signal_date"]][field] for field in ("outcome_month", "outcome_start", "outcome_end")):
            raise DataContractError("Annual target metadata overlaps predictors or changes the next-month clock")
    for row in coverage.select(["source_security_id", "intrinio_security_id", "identity_eligible", "paper_prediction_eligible"]).to_pylist():
        if row["source_security_id"] in SOURCE_ROUTES and (row["intrinio_security_id"] is not None
                or row["identity_eligible"] or row["paper_prediction_eligible"]):
            raise DataContractError("Annual input readmits the known wrong old-BBBY route")
    state = json.loads(paths["target_state"].read_text())
    if (state.get("artifact_sha256") != audit.get("target_artifact_sha256")
            or state.get("provenance", {}).get("label_observed_through", "9999") > f"{year}-12-31"):
        raise DataContractError("Annual target state has a stale identity or future observation bound")
    declared = state.get("provenance", {}).get("month_windows", [])
    if (not isinstance(declared, list) or len(declared) != 12
            or {row["signal_date"]: {field: row[field] for field in
                ("outcome_month", "outcome_start", "outcome_end")} for row in declared} != windows):
        raise DataContractError("Annual target state changes its independently declared outcome windows")
    diagnostic_meta = _metadata(paths["target_diagnostics"], ("signal_date", "outcome_month"))
    if (len(diagnostic_meta) != 12 or set(diagnostic_meta["signal_date"].to_pylist()) != set(windows)
            or set(diagnostic_meta["outcome_month"].to_pylist()) != {w["outcome_month"] for w in windows.values()}):
        raise DataContractError("Annual class diagnostics exceed their approved twelve months")
    projections = ((paths["features"], names), (paths["features"], KEYS),
        (paths["feature_coverage"], COVERAGE_METADATA), (paths["original_cohort"], KEYS),
        (paths["targets"], TARGET_METADATA), (paths["targets"], pq.ParquetFile(paths["targets"]).schema_arrow.names),
        (paths["target_diagnostics"], pq.ParquetFile(paths["target_diagnostics"]).schema_arrow.names))
    pins = tuple(_artifact(root, path) for path in (directory / "audit.json", *paths.values()))
    meta = {"year": year, "directory": str(directory.relative_to(root)),
        "annual_build_identity_sha256": reference["build_identity_sha256"],
        "original_rows": len(cf), "prediction_rows": int(cf.paper_prediction_eligible.sum()),
        "labelled_rows": int(tm.target_available.sum()), "windows": windows,
        "coverage_metadata_sha256": canonical_sha256(coverage.to_pylist()), "source_pins": list(pins),
        "projected_compressed_bytes": sum(_projected_bytes(path, columns) for path, columns in projections),
        "financial_values_read": False}
    return AnnualInputs(year, directory, audit, coverage, windows, pins, meta)


def plan_walkforward_fold(root, run, configuration, identity_sha256, test_year):
    from .walkforward_loading import resolve_walkforward_year
    root, run = Path(root).resolve(), Path(run).resolve()
    run = _within(root, run)
    _configuration(root, configuration)
    if type(test_year) is not int or test_year not in YEARS:
        raise DataContractError("Unknown2020--2023 annual outer fold")
    root_identity = json.loads((run / "build_identity.json").read_text())
    if canonical_sha256(root_identity) != identity_sha256:
        raise DataContractError("Walk-forward model inputs belong to a different parent research identity")
    years = {year: _annual_inputs(root, resolve_walkforward_year(root, run, root_identity, year), year)
        for year in range(test_year - 10, test_year + 1)}
    import exchange_calendars as xcals
    calendar = xcals.get_calendar("XNYS", start=f"{test_year-1}-01-01", end=f"{test_year-1}-12-31")
    cutoff = calendar.sessions[-1].strftime("%Y-%m-%d")
    training_ends = [window["outcome_end"] for year, annual in years.items() if year < test_year for window in annual.windows.values()]
    if max(training_ends) != cutoff:
        raise DataContractError("Annual training outcomes do not end at the independently completed XNYS cutoff")
    rows = sum(p.metadata["original_rows"] for p in years.values())
    provenance = {"test_year": test_year, "train_start": f"{test_year-10}-01-01",
        "train_end": f"{test_year-1}-12-31", "label_observed_through": cutoff,
        "prediction_outcome_start": f"{test_year}-01-01", "prediction_outcome_end": f"{test_year}-12-31",
        "years": {str(year): annual.metadata for year, annual in years.items()},
        "feature_dictionary_sha256": canonical_sha256(_specs()),
        "projected_compressed_bytes": sum(p.metadata["projected_compressed_bytes"] for p in years.values()),
        "estimated_peak_memory_bytes": rows * (715 * 8 * 4 + 4096),
        "metadata_guards_before_financial_decode": True, "financial_values_read": False,
        "financial_2024_plus_decoded": False, "raw_source_reads": False}
    return {"root": root, "run": run, "test_year": test_year, "years": years,
        "provenance": provenance, "fingerprint": canonical_sha256(provenance)}


def _load_fold(plan):
    if canonical_sha256(plan["provenance"]) != plan["fingerprint"]:
        raise DataContractError("Annual outer input plan changed after metadata validation")
    specs, parts = _specs(), {}
    if canonical_sha256(specs) != plan["provenance"]["feature_dictionary_sha256"]:
        raise DataContractError("Frozen feature dictionary changed after metadata planning")
    for year, annual in plan["years"].items():
        if annual.metadata != plan["provenance"]["years"][str(year)]:
            raise DataContractError("Annual metadata plan changed after validation")
        if canonical_sha256(annual.coverage.to_pylist()) != annual.metadata["coverage_metadata_sha256"]:
            raise DataContractError("Annual coverage metadata changed after validation")
        _verify_artifacts(plan["root"], {item["path"]: item for item in annual.pins})
        table = pq.ParquetFile(annual.directory / "features.parquet").read(columns=[*KEYS, *(spec["name"] for spec in specs)])
        panel = FeaturePanel(table, annual.coverage, specs, {}, {}, {"target_windows": annual.windows,
            "source": "verified_walkforward_annual_reference"})
        targets = _load_target_state(annual.directory)
        observed = min(f"{year}-12-31", targets.provenance["label_observed_through"])
        part = classification_training_slice(panel, targets, train_start=f"{year}-01-01",
            train_end=f"{year}-12-31", label_observed_through=observed)
        coverage = annual.coverage.to_pandas().set_index(list(KEYS)).reindex(part.X.index)
        part = replace(part, ledger=part.ledger.join(coverage[[name for name in coverage if name not in part.ledger]]))
        if (len(part.X) != annual.metadata["original_rows"]
                or int(part.ledger.paper_prediction_eligible.sum()) != annual.metadata["prediction_rows"]
                or int(part.ledger.target_available.sum()) != annual.metadata["labelled_rows"]):
            raise DataContractError("Annual financial decoding changed the planned original/eligible/observed cohorts")
        parts[year] = part
        _verify_artifacts(plan["root"], {item["path"]: item for item in annual.pins})
        del table, panel, targets
        gc.collect()
    test_year = plan["test_year"]
    train = _concat_slices([parts[y] for y in range(test_year-10, test_year)],
        start=f"{test_year-10}-01-01", end=f"{test_year-1}-12-31")
    test = _concat_slices([parts[test_year]], start=f"{test_year}-01-01", end=f"{test_year}-12-31")
    if (train.provenance["label_observed_through"] != plan["provenance"]["label_observed_through"]
            or train.ledger.outcome_end.max() >= test.ledger.outcome_start.min()
            or set(train.X.index) & set(test.X.index) or list(train.X) != list(test.X)):
        raise DataContractError("Outer training/test clocks, keys or predictor schemas overlap")
    return train, test


def _phase_identity(root, run, cfg, identity):
    from .walkforward_loading import resolve_walkforward_year
    root_identity = json.loads((run / "build_identity.json").read_text())
    if canonical_sha256(root_identity) != identity:
        raise DataContractError("Walk-forward root identity changed before model-phase freezing")
    references = {str(year): resolve_walkforward_year(root, run, root_identity, year)["audit_pin"] for year in range(2010, 2024)}
    return {"schema_version": 1, "phase": PHASE, "parent_identity_sha256": identity,
        "configuration_sha256": canonical_sha256(cfg), "annual_source_audit_pins": references,
        "workers": {name: _hash(Path(__file__).parent / name) for name in WORKERS},
        "runtime": {"python": platform.python_version(), **{name: version(name)
            for name in ("numpy", "pandas", "pyarrow", "exchange-calendars", *cfg["model_recipe"]["packages"])}}}


@contextmanager
def _single_process(models):
    import fcntl
    with (models / ".fit.lock").open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise DataContractError("Only one sequential outer-fit process is permitted") from error
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _initialize(root, run, cfg, identity):
    phase = _phase_identity(root, run, cfg, identity)
    phase_pin = canonical_sha256(phase)
    models = run / "models"
    models.mkdir(exist_ok=True)
    path = models / "build_identity.json"
    if path.exists():
        if json.loads(path.read_text()) != phase:
            raise DataContractError("Existing outer-fit code/runtime/source identity changed")
    else:
        _json(path, phase)
    return models, phase_pin


def _save_evaluation(directory, ledger, probabilities, labels, index, artifacts, first_month, last_month):
    directory.mkdir(parents=True, exist_ok=True)
    _write_table(directory, "predictions", _predictions(ledger, probabilities, index), artifacts)
    evaluated = evaluate_probabilities(probabilities, labels.loc[index].reset_index(drop=True),
        class_ids=CLASS_IDS, row_metadata=ledger.loc[index].reset_index(),
        first_month=first_month, last_month=last_month)
    expected = len(pd.period_range(first_month, last_month, freq="M"))
    if evaluated.summary["outcome_months"] != expected or evaluated.summary["observed_label_months"] != expected:
        raise DataContractError("Outer evaluation lost a prediction month or has a whole unobserved-label month")
    reports = directory / "reports"
    reports.mkdir(exist_ok=True)
    _write_table(reports, "summary", pd.DataFrame([evaluated.summary]), {})
    artifacts["summary_report"] = _artifact(directory, reports / "summary.parquet")
    for name, table in evaluated.tables.items():
        _write_table(reports, name, table.reset_index(drop=True), {})
        artifacts[name] = _artifact(directory, reports / (name + ".parquet"))
    return evaluated.summary


def _load_estimator(kind, path):
    if kind == "rf":
        import joblib
        return joblib.load(path)
    from xgboost import XGBClassifier
    estimator = XGBClassifier()
    estimator.load_model(path)
    return estimator


def _fit_checkpoint(directory, *, kind, train_X, train_y, parameters, package_versions,
        identity, phase_pin, expected_metadata, maximum_total_fits, models_root):
    """Persist the fitted estimator before inference; never repeat a started fit."""
    complete_path = directory / "FIT_COMPLETE.json"
    if complete_path.exists():
        saved = json.loads(complete_path.read_text())
        if (saved.get("status") != "FIT_COMPLETE" or saved.get("build_identity_sha256") != identity
                or saved.get("model_identity_sha256") != phase_pin
                or saved.get("expected_metadata") != expected_metadata):
            raise DataContractError("Saved fitted-model checkpoint has a different training identity")
        _verify_artifacts(directory, {"model": saved["model"]})
        return _load_estimator(kind, directory / saved["model"]["path"]), saved
    if (directory / "FIT_ATTEMPT.json").exists():
        raise DataContractError("Preserve started fit without a complete model receipt; automatic duplicate fitting is forbidden")
    if len(list(models_root.glob("folds/*/arms/*/FIT_ATTEMPT.json"))) >= maximum_total_fits:
        raise DataContractError("The16-fit walk-forward budget has already been exhausted")
    directory.mkdir(parents=True, exist_ok=True)
    training = _matrix(train_X, "train_X")
    labels = _training_labels(train_y, len(training))
    parameters = _parameters(kind, parameters)
    encoded = labels if kind == "rf" else labels - 1
    package = "scikit-learn" if kind == "rf" else "xgboost"
    if version(package) != package_versions[package]:
        raise DataContractError("Fitting package changed after protocol freezing")
    _json(directory / "FIT_ATTEMPT.json", {"attempt": 1, "build_identity_sha256": identity,
        "model_identity_sha256": phase_pin, "expected_metadata": expected_metadata,
        "maximum_total_fits": maximum_total_fits, "estimator_kind": kind})
    started = time.monotonic()
    estimator = _constructor(kind)(**parameters)
    estimator.fit(training, encoded)
    # No prediction, labels for the test year, or metrics occur before this save.
    if kind == "rf":
        import joblib
        temporary, model = directory / "model.tmp.joblib", directory / "model.joblib"
        joblib.dump(estimator, temporary, compress=3)
    else:
        temporary, model = directory / "model.tmp.ubj", directory / "model.ubj"
        estimator.save_model(temporary)
    temporary.replace(model)
    native = [int(value) for value in estimator.classes_]
    expected_native = list(CLASS_IDS) if kind == "rf" else list(range(10))
    if native != expected_native:
        raise DataContractError("Fitted class map differs from the complete canonical native mapping")
    saved = {"status": "FIT_COMPLETE", "fit_count": 1,
        "build_identity_sha256": identity, "model_identity_sha256": phase_pin,
        "expected_metadata": expected_metadata, "model": _artifact(directory, model),
        "metadata": {"estimator_kind": kind, "package": package,
            "package_version": version(package), "native_class_ids": native,
            "native_to_canonical_mapping": [{"native_class_id": n, "canonical_class_id": n if kind == "rf" else n+1} for n in native],
            "probability_class_ids": list(CLASS_IDS), "fit_count": 1,
            "train_rows": len(training), "matrix_width": training.shape[1],
            "explicit_parameters": _json_parameters(parameters),
            "effective_parameters": _json_parameters(estimator.get_params(deep=False)),
            "prepared_train_matrix_sha256": _sha(training),
            "encoded_training_labels_sha256": _sha(np.ascontiguousarray(encoded)),
            "fit_seconds": time.monotonic() - started,
            "test_labels_supplied_to_estimator": False, "early_stopping_used": False,
            "parameter_search_performed": False, "probability_calibration_performed": False,
            "probability_extraction": "native_predict_proba_float64" if kind == "rf" else "native_output_margins_then_objective_softmax_float64"}}
    _json(complete_path, saved)
    return estimator, saved


def _recipe_for_limits(cfg):
    execution = cfg["execution"]
    return {"execution": {"maximum_wall_seconds": execution["model_maximum_wall_seconds_per_fold"],
        "maximum_peak_rss_bytes": execution["model_maximum_peak_rss_bytes"],
        "maximum_projected_read_bytes": execution["model_maximum_projected_read_bytes"],
        "maximum_output_bytes": 2 * 1024**3}}


def _check_aggregate(cfg, models, current_started, current_previous=0., current_fold=None):
    elapsed = current_previous + time.monotonic() - current_started
    for path in models.glob("folds/*/resource_checkpoint.json"):
        if current_fold is None or path.parent != current_fold:
            elapsed += json.loads(path.read_text())["elapsed_seconds"]
    if elapsed > cfg["execution"]["model_aggregate_maximum_wall_seconds"]:
        raise DataContractError("Walk-forward fitting exceeds the four-hour aggregate execution ceiling")
    if sum(path.stat().st_size for path in models.parent.rglob("*") if path.is_file()) > cfg["execution"]["maximum_output_bytes"]:
        raise DataContractError("Walk-forward outputs exceed the frozen12GiB ceiling")
    return elapsed


def _prepare_fold(plan, fold, phase_pin, identity):
    complete = fold / "DATA_COMPLETE.json"
    if complete.exists():
        saved = json.loads(complete.read_text())
        if saved["plan_fingerprint"] != plan["fingerprint"] or saved["model_identity_sha256"] != phase_pin:
            raise DataContractError("Prepared outer matrices belong to a different input plan")
        _verify_artifacts(fold, saved["artifacts"])
        return saved
    if (fold / "DATA_STARTED.json").exists():
        raise DataContractError("Preserve incomplete preprocessing checkpoint; no implicit repeated learning of a state")
    _json(fold / "DATA_STARTED.json", {"plan_fingerprint": plan["fingerprint"],
        "build_identity_sha256": identity, "model_identity_sha256": phase_pin})
    train, test = _load_fold(plan)
    train_ledger, test_ledger = _ledger(train, training=True), _ledger(test)
    train_mask, test_mask = train_ledger.fit_mask, test_ledger.prediction_eligible
    year = plan["test_year"]
    for frame in (train_ledger, test_ledger):
        frame["fit_year"] = year
        frame["training_outcome_start"] = plan["provenance"]["train_start"]
        frame["training_outcome_end"] = plan["provenance"]["train_end"]
        frame["training_label_observed_through"] = plan["provenance"]["label_observed_through"]
    data, artifacts = fold / "data", {}
    data.mkdir()
    for name, frame in (("train_ledger", train_ledger), ("test_ledger", test_ledger)):
        _write_table(data, name, frame.reset_index(), {})
        artifacts["data/" + name] = _artifact(fold, data / (name + ".parquet"))
    y = train.y.loc[train_mask].to_numpy(dtype=np.int64)
    _training_labels(y, len(y))
    np.save(data / "train_y.npy", y, allow_pickle=False)
    artifacts["data/train_y.npy"] = _artifact(fold, data / "train_y.npy")
    schemas = {}
    for feature_set in ("paper15", "expanded"):
        names = list(PAPER_FEATURE_NAMES) if feature_set == "paper15" else list(train.X)
        selected = replace(train, X=train.X[names])
        fitted = fit_preprocessor(selected, fit_mask=train_mask)
        prepared_train = fitted.transform(selected.X, prediction_mask=train_mask)
        prepared_test = fitted.transform(test.X[names], prediction_mask=test_mask)
        if not prepared_test.ledger.transformed.equals(test_mask):
            raise DataContractError("Outer preprocessing changed the common signal-time prediction population")
        train_X, test_X = _finite32(prepared_train.X.loc[train_mask]), _finite32(prepared_test.X.loc[test_mask])
        directory = fold / "preprocessors" / feature_set
        directory.mkdir(parents=True)
        _json(directory / "manifest.json", {"manifest": fitted.manifest, "fingerprint": fitted.fingerprint})
        np.save(directory / "train_X.npy", train_X, allow_pickle=False)
        np.save(directory / "test_X.npy", test_X, allow_pickle=False)
        for filename in ("manifest.json", "train_X.npy", "test_X.npy"):
            artifacts[f"preprocessors/{feature_set}/{filename}"] = _artifact(fold, directory / filename)
        schemas[feature_set] = {"input_predictors": len(names), "retained_predictors": len(fitted.manifest["retained_columns"]),
            "matrix_width": len(fitted.feature_names), "fingerprint": fitted.fingerprint,
            "dropped_columns": fitted.manifest["dropped_columns"]}
        del selected, fitted, prepared_train, prepared_test, train_X, test_X
        gc.collect()
    counts = {str(k): int((y == k).sum()) for k in CLASS_IDS}
    saved = {"status": "COMPLETE", "plan_fingerprint": plan["fingerprint"],
        "build_identity_sha256": identity, "model_identity_sha256": phase_pin,
        "artifacts": artifacts, "schemas": schemas,
        "fit_keys_sha256": canonical_sha256([list(key) for key in train_ledger.index[train_mask]]),
        "prediction_keys_sha256": canonical_sha256([list(key) for key in test_ledger.index[test_mask]]),
        "train_original_rows": len(train_ledger), "train_fit_rows": len(y),
        "test_original_rows": len(test_ledger), "test_prediction_rows": int(test_mask.sum()),
        "test_labelled_rows": int(test_ledger.target_available.sum()),
        "train_class_counts": counts, "training_prior_vector": [counts[str(k)] / len(y) for k in CLASS_IDS]}
    _json(complete, saved)
    del train, test, train_ledger, test_ledger, y
    gc.collect()
    return saved


def run_walkforward_fold(root, run, cfg, identity_sha256, test_year):
    root = Path(root).resolve(); run = _within(root, Path(run).resolve())
    recipe = _configuration(root, cfg)
    plan = plan_walkforward_fold(root, run, cfg, identity_sha256, test_year)
    if (plan["provenance"]["projected_compressed_bytes"] > cfg["execution"]["model_maximum_projected_read_bytes"]
            or plan["provenance"]["estimated_peak_memory_bytes"] > cfg["execution"]["model_maximum_peak_rss_bytes"]):
        raise DataContractError("Outer metadata projection exceeds its1GiB read or8GiB memory ceiling")
    models, phase_pin = _initialize(root, run, cfg, identity_sha256)
    with _single_process(models):
        fold = models / "folds" / str(test_year)
        if (fold / "audit.json").exists():
            _read_fold(fold, identity_sha256, phase_pin)
            return fold
        fold.mkdir(parents=True, exist_ok=True)
        _json(fold / "read_plan.json", plan["provenance"])
        previous = json.loads((fold / "resource_checkpoint.json").read_text())["elapsed_seconds"] if (fold / "resource_checkpoint.json").exists() else 0.
        started = time.monotonic()
        limits = _recipe_for_limits(cfg)
        limits["execution"]["maximum_wall_seconds"] -= previous
        if limits["execution"]["maximum_wall_seconds"] <= 0:
            raise DataContractError("This outer fold has exhausted its90-minute execution ceiling")
        with _ResourceGuard(limits, started, fold):
            try:
                prepared = _prepare_fold(plan, fold, phase_pin, identity_sha256)
                _verify_artifacts(fold, prepared["artifacts"])
                ledger = pq.ParquetFile(fold / "data/test_ledger.parquet").read().to_pandas().set_index(list(KEYS))
                index = ledger.index[ledger.prediction_eligible]
                labels = ledger.target_class.copy()
                train_y = np.load(fold / "data/train_y.npy", mmap_mode="r", allow_pickle=False)
                arms, summaries = {}, []
                from threadpoolctl import threadpool_limits
                for arm in ARMS:
                    name, feature_set, kind = arm["arm_id"], arm["feature_set"], arm["estimator"]
                    directory = fold / "arms" / name
                    if (directory / "audit.json").exists():
                        audit = _read_arm(directory, identity_sha256, phase_pin)
                        arms[name] = _artifact(fold, directory / "audit.json")
                        summaries.append({"arm_id": name, **audit["summary"]})
                        continue
                    train_X = np.load(fold / "preprocessors" / feature_set / "train_X.npy", mmap_mode="r", allow_pickle=False)
                    test_X = np.load(fold / "preprocessors" / feature_set / "test_X.npy", mmap_mode="r", allow_pickle=False)
                    expected = {"test_year": test_year, "fit_keys_sha256": prepared["fit_keys_sha256"],
                        "preprocessor_fingerprint": prepared["schemas"][feature_set]["fingerprint"],
                        "train_matrix": _artifact(fold, fold / "preprocessors" / feature_set / "train_X.npy"),
                        "training_labels": _artifact(fold, fold / "data/train_y.npy"),
                        **{key: plan["provenance"][key] for key in ("train_start", "train_end", "label_observed_through")}}
                    _check_aggregate(cfg, models, started, previous, fold)
                    _progress("walkforward_arm_started", test_year=test_year, arm_id=name,
                        supervised_rows=len(train_y), prediction_rows=len(test_X), matrix_width=train_X.shape[1])
                    with threadpool_limits(limits=1):
                        estimator, fit = _fit_checkpoint(directory, kind=kind, train_X=train_X, train_y=train_y,
                            parameters=recipe["estimators"][kind], package_versions=recipe["packages"],
                            identity=identity_sha256, phase_pin=phase_pin, expected_metadata=expected,
                            maximum_total_fits=16, models_root=models)
                        probabilities = predict_saved_estimator(estimator, kind, test_X,
                            expected_native_ids=fit["metadata"]["native_class_ids"])
                    arm_artifacts = {"model": _artifact(directory, directory / fit["model"]["path"]),
                        "fit_attempt": _artifact(directory, directory / "FIT_ATTEMPT.json"),
                        "fit_complete": _artifact(directory, directory / "FIT_COMPLETE.json")}
                    summary = _save_evaluation(directory, ledger, probabilities, labels, index,
                        arm_artifacts, f"{test_year}-01", f"{test_year}-12")
                    metadata = {**fit["metadata"], **expected,
                        "prediction_keys_sha256": prepared["prediction_keys_sha256"],
                        "prepared_prediction_matrix_sha256": _sha(np.asarray(test_X)),
                        "prediction_rows": len(index), "test_labels_used_for_fitting": False}
                    _json(directory / "audit.json", {"status": "COMPLETE", "phase": PHASE, "arm": arm,
                        "build_identity_sha256": identity_sha256, "model_identity_sha256": phase_pin,
                        "metadata": metadata, "summary": summary, "artifacts": arm_artifacts})
                    arms[name] = _artifact(fold, directory / "audit.json")
                    summaries.append({"arm_id": name, **summary})
                    del estimator, train_X, test_X, probabilities
                    gc.collect()
                    _limits(limits, started, fold, plan["provenance"]["projected_compressed_bytes"])
                    _json(fold / "resource_checkpoint.json", {"elapsed_seconds": previous + time.monotonic() - started})
                    _progress("walkforward_arm_completed", test_year=test_year, arm_id=name, **summary)
                for name, vector in (("uniform", np.full(10, .1)), ("train_prior", np.asarray(prepared["training_prior_vector"]))):
                    directory = fold / "refs" / name
                    ref_artifacts = {}
                    probabilities = np.broadcast_to(vector, (len(index), 10)).copy()
                    summary = _save_evaluation(directory, ledger, probabilities, labels, index, ref_artifacts,
                        f"{test_year}-01", f"{test_year}-12")
                    _json(directory / "audit.json", {"status": "COMPLETE", "phase": PHASE,
                        "build_identity_sha256": identity_sha256, "model_identity_sha256": phase_pin,
                        "metadata": {"fit_count": 0, "prior_vector": vector.tolist(), "test_year": test_year},
                        "summary": summary, "artifacts": ref_artifacts})
                    arms[name] = _artifact(fold, directory / "audit.json")
                    summaries.append({"arm_id": name, **summary})
                reports = fold / "reports"; reports.mkdir(exist_ok=True)
                _write_table(reports, "comparison", pd.DataFrame(summaries), {})
                _json(fold / "resource_checkpoint.json", {"elapsed_seconds": previous + time.monotonic() - started})
                artifacts = {"DATA_COMPLETE.json": _artifact(fold, fold / "DATA_COMPLETE.json"),
                    "read_plan.json": _artifact(fold, fold / "read_plan.json"),
                    "resource_checkpoint.json": _artifact(fold, fold / "resource_checkpoint.json"),
                    "comparison": _artifact(fold, reports / "comparison.parquet"), **prepared["artifacts"]}
                for annual in plan["years"].values():
                    _verify_artifacts(root, {item["path"]: item for item in annual.pins})
                _check_aggregate(cfg, models, started, previous, fold)
                _json(fold / "audit.json", {"status": "COMPLETE", "phase": PHASE, "test_year": test_year,
                    "build_identity_sha256": identity_sha256, "model_identity_sha256": phase_pin,
                    "summary": {"fitted_arms": 4, "preprocessor_fits": 2, "prediction_months": 12,
                        "train_original_rows": prepared["train_original_rows"], "train_fit_rows": prepared["train_fit_rows"],
                        "test_original_rows": prepared["test_original_rows"], "test_prediction_rows": prepared["test_prediction_rows"],
                        "test_labelled_rows": prepared["test_labelled_rows"], "schemas": prepared["schemas"],
                        "label_cutoff": plan["provenance"]["label_observed_through"],
                        "same_prediction_and_label_masks_all_arms": True,
                        "elapsed_seconds": previous + time.monotonic() - started,
                        "financial_2024_plus_decoded": False}, "artifacts": artifacts, "arms": arms})
            except BaseException as error:
                _json(fold / "resource_checkpoint.json", {"elapsed_seconds": previous + time.monotonic() - started})
                _json(fold / "failure.json", {"exception": type(error).__name__, "message": str(error),
                    "automatic_refitting_forbidden": True})
                raise
    return fold


def _read_arm(directory, identity, phase_pin):
    audit = json.loads((directory / "audit.json").read_text())
    if (audit.get("status") != "COMPLETE" or audit.get("phase") != PHASE
            or audit.get("build_identity_sha256") != identity or audit.get("model_identity_sha256") != phase_pin
            or not {"predictions", "summary_report", *REPORTS} <= set(audit.get("artifacts", {}))):
        raise DataContractError("Saved outer model arm is incomplete or has a different identity")
    _verify_artifacts(directory, audit["artifacts"])
    return audit


def _read_fold(directory, identity, phase_pin):
    audit = json.loads((directory / "audit.json").read_text())
    if (audit.get("status") != "COMPLETE" or audit.get("phase") != PHASE
            or audit.get("build_identity_sha256") != identity or audit.get("model_identity_sha256") != phase_pin
            or audit.get("test_year") not in YEARS or audit.get("summary", {}).get("fitted_arms") != 4
            or audit.get("summary", {}).get("preprocessor_fits") != 2
            or audit.get("summary", {}).get("prediction_months") != 12
            or set(audit.get("arms", {})) != {*(arm["arm_id"] for arm in ARMS), "uniform", "train_prior"}):
        raise DataContractError("Saved outer model fold is incomplete or has different fit authority")
    _verify_artifacts(directory, audit["artifacts"])
    _verify_artifacts(directory, audit["arms"])
    for name, item in audit["arms"].items():
        arm_directory = directory / item["path"].removesuffix("/audit.json")
        arm = _read_arm(arm_directory, identity, phase_pin)
        if name in {value["arm_id"] for value in ARMS}:
            if (arm.get("metadata", {}).get("fit_count") != 1
                    or not {"model", "fit_attempt", "fit_complete"} <= set(arm["artifacts"])):
                raise DataContractError("Saved annual fitted arm omits its unique model/fit checkpoint")
            fit = json.loads((arm_directory / "FIT_COMPLETE.json").read_text())
            if (fit.get("fit_count") != 1 or fit.get("status") != "FIT_COMPLETE"
                    or fit.get("build_identity_sha256") != identity or fit.get("model_identity_sha256") != phase_pin):
                raise DataContractError("Annual fitted-model checkpoint changed its original fit identity")
    return audit


def finalize_walkforward_models(root, run, cfg, identity_sha256):
    """Stitch only saved outer predictions; no estimator/preprocessor fitting."""
    root = Path(root).resolve(); run = _within(root, Path(run).resolve())
    _configuration(root, cfg)
    models, phase_pin = _initialize(root, run, cfg, identity_sha256)
    if (models / "audit.json").exists():
        return read_walkforward_models(models, root=root, identity_sha256=identity_sha256)["directory"]
    with _single_process(models):
        folds = {year: _read_fold(models / "folds" / str(year), identity_sha256, phase_pin) for year in YEARS}
        if sum(fold["summary"]["elapsed_seconds"] for fold in folds.values()) > cfg["execution"]["model_aggregate_maximum_wall_seconds"]:
            raise DataContractError("Completed fold execution exceeds the frozen aggregate model budget")
        if len(list(models.glob("folds/*/arms/*/FIT_ATTEMPT.json"))) != 16:
            raise DataContractError("Final outer panel must have exactly16 original estimator-fit attempts")
        artifacts = {"build_identity.json": _artifact(models, models / "build_identity.json")}
        arm_pins, comparisons, common_cohort = {}, [], None
        for name in [*(arm["arm_id"] for arm in ARMS), "uniform", "train_prior"]:
            frames = []
            for year in YEARS:
                location = models / "folds" / str(year) / ("arms" if name in {a["arm_id"] for a in ARMS} else "refs") / name
                path = location / "predictions.parquet"
                # Metadata guard before probability/label decode, even for saved
                # fold outputs whose whole files were hash authenticated above.
                meta = pq.ParquetFile(path).read(columns=[*KEYS, "outcome_month", "outcome_end", "fit_year"]).to_pandas()
                if (meta.duplicated(list(KEYS)).any() or not meta.outcome_month.str.startswith(str(year) + "-").all()
                        or not meta.fit_year.eq(year).all() or not meta.outcome_end.le(f"{year}-12-31").all()):
                    raise DataContractError("Saved outer probabilities leave their labelled test year")
                frames.append(pq.ParquetFile(path).read().to_pandas())
            frame = pd.concat(frames, ignore_index=True)
            if frame.duplicated(list(KEYS)).any() or frame.outcome_month.nunique() != 48:
                raise DataContractError("Stitched outer predictions changed original keys or lost a month")
            cohort = frame[[*KEYS, "outcome_month", "outcome_start", "outcome_end", "prediction_eligible",
                "paper_prediction_eligible", "target_available", "target_class", "fit_year"]]
            if common_cohort is not None and not cohort.equals(common_cohort):
                raise DataContractError("Outer model arms changed original cohorts, dates or observed-label masks")
            common_cohort = cohort.copy() if common_cohort is None else common_cohort
            ledger = frame.drop(columns=[*PROBABILITIES, "prediction_status"]).set_index(list(KEYS))
            index = ledger.index[ledger.prediction_eligible]
            probabilities = frame.loc[frame.prediction_eligible, PROBABILITIES].to_numpy(dtype=float)
            directory = models / ("arms" if name in {a["arm_id"] for a in ARMS} else "refs") / name
            arm_artifacts = {}
            summary = _save_evaluation(directory, ledger, probabilities, ledger.target_class, index,
                arm_artifacts, "2020-01", "2023-12")
            _json(directory / "audit.json", {"status": "COMPLETE", "phase": PHASE,
                "build_identity_sha256": identity_sha256, "model_identity_sha256": phase_pin,
                "metadata": {"fit_count": 4 if name in {a["arm_id"] for a in ARMS} else 0,
                    "test_years": list(YEARS), "new_fits_during_stitching": 0,
                    "original_rows_retained": True, "same_prediction_and_observed_label_masks": True},
                "summary": summary, "artifacts": arm_artifacts})
            arm_pins[name] = _artifact(models, directory / "audit.json")
            comparisons.append({"arm_id": name, **summary})
            del frame, frames, ledger, probabilities
            gc.collect()
        reports = models / "reports"; reports.mkdir(exist_ok=True)
        _write_table(reports, "comparison", pd.DataFrame(comparisons), {})
        artifacts["comparison"] = _artifact(models, reports / "comparison.parquet")
        fold_pins = {str(year): _artifact(models, models / "folds" / str(year) / "audit.json") for year in YEARS}
        _json(models / "audit.json", {"status": "COMPLETE", "phase": PHASE,
            "build_identity_sha256": identity_sha256, "model_identity_sha256": phase_pin,
            "summary": {"fitted_arms": 16, "preprocessor_fits": 8, "outer_years": list(YEARS),
                "months": 48, "same_prediction_cohorts_all_arms": True,
                "aggregate_fold_seconds": sum(f["summary"]["elapsed_seconds"] for f in folds.values()),
                "parameter_search_performed": False, "financial_2024_plus_decoded": False,
                "RET_strategy_evaluated": False}, "artifacts": artifacts, "arms": arm_pins, "folds": fold_pins})
    read_walkforward_models(models, root=root, identity_sha256=identity_sha256)
    return models


PROBABILITIES = [f"probability_{class_id}" for class_id in CLASS_IDS]


def read_walkforward_models(directory, *, identity_sha256=None, root=None):
    """Verify completed saved models/ledgers without deserializing or fitting."""
    directory = Path(directory).resolve()
    audit = json.loads((directory / "audit.json").read_text())
    phase = json.loads((directory / "build_identity.json").read_text())
    if (audit.get("status") != "COMPLETE" or audit.get("phase") != PHASE
            or audit.get("summary", {}).get("fitted_arms") != 16
            or audit.get("summary", {}).get("preprocessor_fits") != 8
            or audit.get("summary", {}).get("outer_years") != list(YEARS)
            or audit.get("summary", {}).get("months") != 48
            or identity_sha256 is not None and audit.get("build_identity_sha256") != identity_sha256
            or canonical_sha256(phase) != audit.get("model_identity_sha256")
            or phase.get("parent_identity_sha256") != audit.get("build_identity_sha256")
            or set(audit.get("arms", {})) != {*(arm["arm_id"] for arm in ARMS), "uniform", "train_prior"}
            or set(audit.get("folds", {})) != {str(year) for year in YEARS}):
        raise DataContractError("Saved16-fit walk-forward model phase is incomplete or has a different scope")
    _verify_artifacts(directory, audit["artifacts"])
    _verify_artifacts(directory, audit["arms"])
    _verify_artifacts(directory, audit["folds"])
    for item in audit["arms"].values():
        _read_arm(directory / item["path"].removesuffix("/audit.json"), audit["build_identity_sha256"], audit["model_identity_sha256"])
    for year in YEARS:
        _read_fold(directory / "folds" / str(year), audit["build_identity_sha256"], audit["model_identity_sha256"])
    if root is not None:
        root = Path(root).resolve(); _within(root, directory)
        _verify_artifacts(root, phase["annual_source_audit_pins"])
        for year in YEARS:
            plan = json.loads((directory / "folds" / str(year) / "read_plan.json").read_text())
            for annual in plan["years"].values():
                _verify_artifacts(root, {item["path"]: item for item in annual["source_pins"]})
    return {"directory": directory, "audit": audit}
