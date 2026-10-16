"""Reuse the four frozen fits on a separately repaired validation panel.

Only the 2018--2019 annual financial tables are decoded. Historical training
tables, matrices, fitted preprocessing states and estimators are integrity
checked without fitting or changing their bytes. New probabilities and labels
belong to this independent repair identity, never to the original fit identity.
"""
from __future__ import annotations

from dataclasses import replace
import gc
from importlib.metadata import version
import json
from numbers import Integral
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .catalog import canonical_sha256
from .classification import classification_training_slice
from .classifier_pilot import (ARMS, REPORT_NAMES, _ResourceGuard, _finite32,
    _ledger, _save_evaluation)
from .fallback_models import read_fallback_models
from .features import FeaturePanel, KEYS
from .full_panel import (_artifact, _hash, _json, _limits, _progress, _specs,
    _verify_artifacts, _within, _write_table)
from .full_target_panel import _load_target_state
from .pilot_loading import (_concat_slices, _keys, _metadata, _year_plan,
    _year_snapshot)
from .preprocessing import FittedPreprocessor
from .ret import CLASS_IDS, align_class_probabilities
from .universe import DataContractError


PHASE = "validation_reprediction_v1"
VALIDATION_YEARS = (2018, 2019)
FEATURE_SETS = ("paper15", "expanded")
WORKERS = ("validation_reprediction.py", "fallback_models.py", "classifier_pilot.py",
    "pilot_loading.py", "pilot_metrics.py", "classification.py", "preprocessing.py",
    "features.py", "paper_features.py", "ret.py", "targets.py", "full_panel.py",
    "full_target_panel.py", "catalog.py", "universe.py")
NUMERICAL_WORKERS = ("preprocessing.py", "pilot_models.py", "classification.py",
    "features.py", "paper_features.py", "ret.py", "targets.py")


def restore_fitted_preprocessor(saved):
    """Restore verified training-only medians/schema; never learn new state."""
    if not isinstance(saved, dict) or set(saved) != {"manifest", "fingerprint"}:
        raise DataContractError("Saved preprocessing state requires its exact manifest and fingerprint")
    manifest = saved["manifest"]
    if (canonical_sha256(manifest) != saved["fingerprint"]
            or manifest.get("train_start") != "2010-01-01"
            or manifest.get("train_end") != "2017-12-31"
            or not "2017-12-01" <= manifest.get("label_observed_through", "") <= "2017-12-31"
            or manifest.get("training_only_schema_selection") is not True
            or manifest.get("scaling") is not None
            or manifest.get("clipping") is not None
            or manifest.get("winsorization") is not None):
        raise DataContractError("Reused preprocessing state differs from the frozen completed training window")
    return FittedPreprocessor(json.dumps(manifest, sort_keys=True,
        separators=(",", ":"), allow_nan=False), saved["fingerprint"])


def predict_saved_estimator(estimator, kind, matrix, *, expected_native_ids=None):
    """Perform the original RF or XGBoost extraction without a fit entry point."""
    if kind not in {"rf", "xgb"}:
        raise DataContractError("Reused estimator must be RF or XGBoost")
    values = np.asarray(matrix)
    if (values.ndim != 2 or values.shape[1] < 1 or values.dtype != np.dtype("float32")
            or not np.isfinite(values).all()):
        raise DataContractError("Saved-model inference requires a finite float32 matrix")
    native = list(estimator.classes_)
    expected = list(CLASS_IDS) if kind == "rf" else list(range(10))
    if (len(native) != 10 or set(native) != set(expected)
            or any(isinstance(item, (bool, np.bool_)) or not isinstance(item, Integral) for item in native)
            or expected_native_ids is not None and native != expected_native_ids):
        raise DataContractError("Reused estimator class IDs differ from the saved native mapping")
    if getattr(estimator, "n_features_in_", values.shape[1]) != values.shape[1]:
        raise DataContractError("Reused model width differs from the frozen preprocessing schema")
    canonical = native if kind == "rf" else [int(item) + 1 for item in native]
    if not len(values):
        probabilities = np.empty((0, 10), dtype=np.float64)
    elif kind == "rf":
        probabilities = estimator.predict_proba(values)
    else:
        margins = np.asarray(estimator.predict(values, output_margin=True))
        if (margins.shape != (len(values), 10) or margins.dtype.kind not in "fiu"
                or not np.isfinite(margins).all()):
            raise DataContractError("Reused XGBoost must return finite ten-class native margins")
        shifted = margins.astype(np.float64) - margins.max(axis=1, keepdims=True).astype(np.float64)
        exponentials = np.exp(shifted)
        probabilities = exponentials / exponentials.sum(axis=1, keepdims=True)
    return align_class_probabilities(probabilities, class_ids=canonical)


def _parent(root, directory, configuration):
    authority = configuration.get("validation_reprediction", {})
    if (authority.get("approved") is not True
            or authority.get("financial_outcome_years") != list(VALIDATION_YEARS)
            or configuration.get("execution", {}).get("maximum_new_model_fits") != 0):
        raise DataContractError("Independent validation inference requires explicit zero-new-fit authority")
    parent = _within(root, Path(directory).resolve())
    pin = authority.get("fitted_parent_build_identity_sha256")
    saved = read_fallback_models(parent / "models", identity_sha256=pin)
    audit = json.loads((parent / "audit.json").read_text())
    identity = json.loads((parent / "build_identity.json").read_text())
    if (not isinstance(pin, str) or canonical_sha256(identity) != pin
            or audit.get("status") != "COMPLETE" or audit.get("build_identity_sha256") != pin):
        raise DataContractError("Fitted parent reconstruction is incomplete or differs from its explicit pin")
    phase = json.loads((parent / "models/build_identity.json").read_text())
    recipe = phase["model_recipe"]
    for package, expected in recipe["packages"].items():
        if version(package) != expected:
            raise DataContractError("Saved estimator inference package differs from its frozen version: " + package)
    for name in NUMERICAL_WORKERS:
        if _hash(Path(__file__).parent / name) != phase["workers"][name]:
            raise DataContractError("Reused prediction or numerical feature worker changed: " + name)
    pins = {str(path.relative_to(root)): _artifact(root, path) for path in (
        parent / "audit.json", parent / "build_identity.json",
        parent / "models/audit.json", parent / "models/build_identity.json")}
    # Hash every training input and saved matrix without decoding any of them.
    for year in range(2010, 2018):
        directory = parent / "years" / str(year)
        annual = json.loads((directory / "audit.json").read_text())
        if (annual.get("status") != "COMPLETE" or annual.get("year") != year
                or annual.get("build_identity_sha256") != pin):
            raise DataContractError("Fitted parent training annual identity changed")
        _verify_artifacts(directory, annual["artifacts"])
        for path in (directory / "audit.json", *(_within(directory, directory / item["path"])
                for item in annual["artifacts"].values())):
            pins[str(path.relative_to(root))] = _artifact(root, path)
    for feature_set in FEATURE_SETS:
        for filename in ("manifest.json", "train_X.npy"):
            path = parent / "models/preprocessors" / feature_set / filename
            pins[str(path.relative_to(root))] = _artifact(root, path)
    for name in ("train_ledger.parquet",):
        path = parent / "models/data" / name
        pins[str(path.relative_to(root))] = _artifact(root, path)
    for arm in ARMS:
        arm_path = parent / "models/arms" / arm["arm_id"] / "audit.json"
        arm_audit = json.loads(arm_path.read_text())
        for path in (arm_path, _within(arm_path.parent, arm_path.parent / arm_audit["artifacts"]["model"]["path"])):
            pins[str(path.relative_to(root))] = _artifact(root, path)
    return parent, saved["audit"], recipe, pins


def plan_validation_reprediction(root, run_directory, configuration, identity_sha256, *, fitted_run_directory):
    """Validate annual keys/clocks and all reused fits before financial decoding."""
    root, run = Path(root).resolve(), Path(run_directory).resolve()
    run = _within(root, run)
    parent, parent_audit, recipe, source_pins = _parent(root, fitted_run_directory, configuration)
    if (run == parent or run == root or not isinstance(identity_sha256, str)
            or not identity_sha256.startswith("sha256:")):
        raise DataContractError("Repaired inference requires a separate run and explicit build identity")
    expected = {"directory": run, "audit": {"build_identity_sha256": identity_sha256,
        "years": {str(year): _artifact(run, run / "years" / str(year) / "audit.json")
            for year in VALIDATION_YEARS}}}
    specs = _specs()
    years = {year: _year_plan(root, expected, expected, year, specs) for year in VALIDATION_YEARS}
    for year, annual in years.items():
        old = _keys(_metadata(parent / "years" / str(year) / "original_cohort.parquet", KEYS),
            label="Original validation denominator")
        new = _keys(annual.coverage, label="Repaired validation denominator")
        if set(old) != set(new):
            raise DataContractError("Quote repair changed original validation source-ID/signal membership")
        for item in annual.source_pins:
            source_pins[item["path"]] = item
    provenance = {"phase": PHASE, "build_identity_sha256": identity_sha256,
        "fitted_parent_run": str(parent.relative_to(root)),
        "fitted_parent_build_identity_sha256": parent_audit["build_identity_sha256"],
        "fitted_parent_model_identity_sha256": parent_audit["model_identity_sha256"],
        "new_estimator_fits": 0, "preprocessing_fits": 0,
        "training_financial_values_decoded": False, "financial_2020_plus_decoded": False,
        "source_pins": source_pins,
        "years": {str(year): _year_snapshot(root, annual) for year, annual in years.items()},
        "feature_dictionary_sha256": canonical_sha256(specs),
        "projected_compressed_bytes": sum(p.read_plan["projected_compressed_bytes"] for p in years.values()),
        "original_rows": sum(p.read_plan["original_rows"] for p in years.values()),
        "prediction_rows": sum(p.read_plan["prediction_eligible_rows"] for p in years.values()),
        "labelled_rows": sum(p.read_plan["labelled_rows"] for p in years.values()),
        "metadata_guards_before_financial_decode": True}
    return {"root": root, "run": run, "parent": parent, "parent_audit": parent_audit,
        "recipe": recipe, "years": years, "provenance": provenance,
        "fingerprint": canonical_sha256(provenance)}


def _load_validation(plan):
    if (tuple(plan["years"]) != VALIDATION_YEARS
            or canonical_sha256(plan["provenance"]) != plan["fingerprint"]
            or canonical_sha256(_specs()) != plan["provenance"]["feature_dictionary_sha256"]):
        raise DataContractError("Repaired validation plan or registered predictor dictionary changed")
    specs, parts = _specs(), []
    for year, annual in plan["years"].items():
        if (annual.year != year
                or _year_snapshot(plan["root"], annual) != plan["provenance"]["years"][str(year)]):
            raise DataContractError("Repaired validation annual metadata changed after planning")
        _verify_artifacts(plan["root"], {item["path"]: item for item in annual.source_pins})
        table = pq.ParquetFile(annual.feature_directory / "features.parquet").read(
            columns=[*KEYS, *(spec["name"] for spec in specs)])
        panel = FeaturePanel(table, annual.coverage, specs, {}, {},
            {"target_windows": annual.windows, "source": "independently_repaired_validation_cache"})
        targets = _load_target_state(annual.target_directory)
        part = classification_training_slice(panel, targets,
            train_start=f"{year}-01-01", train_end=f"{year}-12-31",
            label_observed_through=min(f"{year}-12-31", targets.provenance["label_observed_through"]))
        coverage = annual.coverage.to_pandas().set_index(list(KEYS)).reindex(part.X.index)
        part = replace(part, ledger=part.ledger.join(coverage[[name for name in coverage if name not in part.ledger]]))
        if (len(part.X) != annual.read_plan["original_rows"]
                or int(part.ledger.paper_prediction_eligible.sum()) != annual.read_plan["prediction_eligible_rows"]
                or int(part.ledger.target_available.sum()) != annual.read_plan["labelled_rows"]):
            raise DataContractError("Repaired validation financial decoding changed the planned cohorts")
        parts.append(part)
        _verify_artifacts(plan["root"], {item["path"]: item for item in annual.source_pins})
        del table, panel, targets
        gc.collect()
    return _concat_slices(parts, start="2018-01-01", end="2019-12-31")


def _phase_identity(plan, configuration, identity_sha256):
    return {"schema_version": 1, "phase": PHASE,
        "parent_identity_sha256": identity_sha256,
        "configuration_sha256": canonical_sha256(configuration),
        "data_plan_sha256": plan["fingerprint"], "data_plan": plan["provenance"],
        "source_root": str(plan["root"]), "new_estimator_fits": 0,
        "preprocessing_fits": 0,
        "workers": {name: _hash(Path(__file__).parent / name) for name in WORKERS},
        "runtime": {"python": platform.python_version(), **{name: version(name)
            for name in ("numpy", "pandas", "pyarrow", *plan["recipe"]["packages"])}}}


def read_validation_reprediction(directory, *, identity_sha256=None, model_identity_sha256=None, root=None):
    """Verify inference receipts and the unchanged parent files without loading models."""
    directory = Path(directory).resolve()
    audit = json.loads((directory / "audit.json").read_text())
    phase = json.loads((directory / "build_identity.json").read_text())
    source_root = Path(phase["source_root"]).resolve() if root is None else Path(root).resolve()
    _within(source_root, directory)
    if (audit.get("status") != "COMPLETE" or audit.get("phase") != PHASE
            or audit.get("summary", {}).get("newly_fitted_arms") != 0
            or audit.get("summary", {}).get("reused_fitted_arms") != 4
            or audit.get("summary", {}).get("preprocessing_fits") != 0
            or identity_sha256 is not None and audit.get("build_identity_sha256") != identity_sha256
            or model_identity_sha256 is not None and audit.get("model_identity_sha256") != model_identity_sha256
            or canonical_sha256(phase) != audit.get("model_identity_sha256")
            or phase.get("parent_identity_sha256") != audit.get("build_identity_sha256")
            or phase.get("phase") != PHASE or phase.get("new_estimator_fits") != 0
            or set(audit.get("arms", {})) != {*(a["arm_id"] for a in ARMS), "uniform", "train_prior"}):
        raise DataContractError("Saved validation inference phase is incomplete or has an inconsistent fit identity")
    _verify_artifacts(directory, audit["artifacts"])
    _verify_artifacts(directory, audit["arms"])
    required = {"build_identity.json", "read_plan.json", "ATTEMPT.json",
        "loading_provenance.json", "comparison", "data/validation_ledger",
        *(f"preprocessors/{feature_set}/{name}" for feature_set in FEATURE_SETS
            for name in ("manifest.json", "validation_X.npy"))}
    if not required <= set(audit["artifacts"]):
        raise DataContractError("Saved validation inference omits its bounded input/schema/report receipts")
    _verify_artifacts(source_root, phase["data_plan"]["source_pins"])
    for name, item in audit["arms"].items():
        path = _within(directory, directory / item["path"])
        arm = json.loads(path.read_text())
        if (arm.get("status") != "COMPLETE" or arm.get("phase") != PHASE
                or arm.get("build_identity_sha256") != audit["build_identity_sha256"]
                or arm.get("model_identity_sha256") != audit["model_identity_sha256"]
                or not {"predictions", "summary_report", *REPORT_NAMES} <= set(arm.get("artifacts", {}))
                or arm.get("metadata", {}).get("new_fit_count") != 0):
            raise DataContractError("Saved repaired-validation arm or zero-fit receipt changed")
        if name in {a["arm_id"] for a in ARMS} and (
                arm["metadata"].get("reused_fit_count") != 1
                or arm["metadata"].get("preprocessing_fit_count") != 0
                or arm["metadata"].get("parent_model_identity_sha256")
                    != phase["data_plan"]["fitted_parent_model_identity_sha256"]):
            raise DataContractError("Saved reused-model arm loses its original fitted identity")
        _verify_artifacts(path.parent, arm["artifacts"])
    return {"directory": directory, "audit": audit}


def run_validation_reprediction(root, run_directory, configuration, identity_sha256, *, fitted_run_directory):
    """Regenerate all validation forecasts with four existing fits, never refit."""
    plan = plan_validation_reprediction(root, run_directory, configuration, identity_sha256,
        fitted_run_directory=fitted_run_directory)
    phase = _phase_identity(plan, configuration, identity_sha256)
    phase_pin = canonical_sha256(phase)
    destination = plan["run"] / "models"
    if (destination / "audit.json").exists():
        return read_validation_reprediction(destination, identity_sha256=identity_sha256,
            model_identity_sha256=phase_pin, root=plan["root"])["directory"]
    if destination.exists():
        raise DataContractError("Preserve the incomplete inference attempt; do not silently overwrite: " + str(destination))
    recipe = plan["recipe"]
    if plan["provenance"]["projected_compressed_bytes"] > recipe["execution"]["maximum_projected_read_bytes"]:
        raise DataContractError("Validation inference exceeds the parent recipe's bounded local read ceiling")
    destination.mkdir()
    _json(destination / "build_identity.json", phase)
    _json(destination / "read_plan.json", plan["provenance"])
    _json(destination / "ATTEMPT.json", {"attempt": 1, "phase": PHASE,
        "build_identity_sha256": identity_sha256, "model_identity_sha256": phase_pin,
        "maximum_new_model_fits": 0, "maximum_new_preprocessor_fits": 0})
    started = time.monotonic()
    with _ResourceGuard(recipe, started, destination):
        return _execute(plan, destination, phase_pin, identity_sha256, started)


def _execute(plan, destination, phase_pin, identity_sha256, started):
    from threadpoolctl import threadpool_limits
    validation = _load_validation(plan)
    ledger = _ledger(validation)
    prediction_mask, prediction_index = ledger.prediction_eligible, ledger.index[ledger.prediction_eligible]
    labels = validation.y.copy()
    artifacts, schemas = {}, {}
    data = destination / "data"
    data.mkdir()
    _write_table(data, "validation_ledger", ledger.reset_index(), {})
    artifacts["data/validation_ledger"] = _artifact(destination, data / "validation_ledger.parquet")
    for feature_set in FEATURE_SETS:
        original = plan["parent"] / "models/preprocessors" / feature_set / "manifest.json"
        saved = json.loads(original.read_text())
        fitted = restore_fitted_preprocessor(saved)
        prepared = fitted.transform(validation.X[fitted.manifest["input_columns"]], prediction_mask=prediction_mask)
        if not prepared.ledger.transformed.equals(prediction_mask):
            raise DataContractError("Frozen preprocessing changed the repaired signal-time prediction population")
        matrix = _finite32(prepared.X.loc[prediction_mask])
        directory = destination / "preprocessors" / feature_set
        directory.mkdir(parents=True)
        # Byte-exact copy, with training medians/schema still owned by the parent.
        (directory / "manifest.json").write_bytes(original.read_bytes())
        np.save(directory / "validation_X.npy", matrix, allow_pickle=False)
        for filename in ("manifest.json", "validation_X.npy"):
            artifacts[f"preprocessors/{feature_set}/{filename}"] = _artifact(destination, directory / filename)
        schemas[feature_set] = {"fingerprint": fitted.fingerprint,
            "input_predictors": len(fitted.manifest["input_columns"]),
            "retained_predictors": len(fitted.manifest["retained_columns"]),
            "matrix_width": len(fitted.feature_names), "new_fit_count": 0,
            "manifest_bytes_identical_to_parent": _hash(directory / "manifest.json") == _hash(original)}
        del fitted, prepared, matrix
        gc.collect()
    del validation
    gc.collect()
    arms, summaries = {}, []
    for arm in ARMS:
        name, feature_set, kind = arm["arm_id"], arm["feature_set"], arm["estimator"]
        parent_arm = plan["parent"] / "models/arms" / name
        parent_audit = json.loads((parent_arm / "audit.json").read_text())
        if schemas[feature_set]["fingerprint"] != parent_audit["metadata"]["preprocessor_fingerprint"]:
            raise DataContractError("Reused estimator and preprocessing state came from different fitted inputs")
        model_pin = parent_audit["artifacts"]["model"]
        _verify_artifacts(parent_arm, {"model": model_pin})
        model_path = _within(parent_arm, parent_arm / model_pin["path"])
        matrix = np.load(destination / "preprocessors" / feature_set / "validation_X.npy", mmap_mode="r", allow_pickle=False)
        _progress("validation_reprediction_started", arm_id=name, new_fit_count=0, prediction_rows=len(matrix))
        if kind == "rf":
            import joblib
            estimator = joblib.load(model_path)
        else:
            from xgboost import XGBClassifier
            estimator = XGBClassifier()
            estimator.load_model(model_path)
        with threadpool_limits(limits=1):
            probabilities = predict_saved_estimator(estimator, kind, matrix,
                expected_native_ids=parent_audit["metadata"]["native_class_ids"])
        directory = destination / "arms" / name
        arm_artifacts = {}
        summary = _save_evaluation(directory, ledger, probabilities, labels, prediction_index, arm_artifacts)
        metadata = {"new_fit_count": 0, "reused_fit_count": 1,
            "preprocessing_fit_count": 0, "parent_model_identity_sha256": plan["parent_audit"]["model_identity_sha256"],
            "parent_arm_audit": _artifact(plan["root"], parent_arm / "audit.json"),
            "parent_model": _artifact(plan["root"], model_path),
            "fit_keys_sha256": parent_audit["metadata"]["fit_keys_sha256"],
            "prepared_train_matrix_sha256": parent_audit["metadata"]["prepared_train_matrix_sha256"],
            "train_rows": parent_audit["metadata"]["train_rows"],
            "native_class_ids": parent_audit["metadata"]["native_class_ids"],
            "preprocessor_fingerprint": schemas[feature_set]["fingerprint"],
            "prediction_keys_sha256": canonical_sha256([list(key) for key in prediction_index]),
            "validation_labels_supplied_to_estimator": False,
            "probability_extraction": parent_audit["metadata"]["probability_extraction"],
            "parameter_search_performed": False, "probability_calibration_performed": False}
        _json(directory / "audit.json", {"status": "COMPLETE", "phase": PHASE, "arm": arm,
            "build_identity_sha256": identity_sha256, "model_identity_sha256": phase_pin,
            "metadata": metadata, "summary": summary, "artifacts": arm_artifacts})
        arms[name] = _artifact(destination, directory / "audit.json")
        summaries.append({"arm_id": name, "feature_set": feature_set, "estimator": kind,
            "new_fit_count": 0, **summary})
        del estimator, matrix, probabilities
        gc.collect()
        _limits(plan["recipe"], started, destination, plan["provenance"]["projected_compressed_bytes"])
        _progress("validation_reprediction_completed", arm_id=name, new_fit_count=0,
            prediction_rows=len(prediction_index))
    prior = np.asarray(plan["parent_audit"]["summary"]["training_prior_vector"], dtype=np.float64)
    for name, vector in (("uniform", np.full(10, .1)), ("train_prior", prior)):
        directory, arm_artifacts = destination / "refs" / name, {}
        probabilities = np.broadcast_to(vector, (len(prediction_index), 10)).copy()
        summary = _save_evaluation(directory, ledger, probabilities, labels, prediction_index, arm_artifacts)
        _json(directory / "audit.json", {"status": "COMPLETE", "phase": PHASE, "reference": name,
            "build_identity_sha256": identity_sha256, "model_identity_sha256": phase_pin,
            "metadata": {"new_fit_count": 0, "prior_vector": vector.tolist()},
            "summary": summary, "artifacts": arm_artifacts})
        arms[name] = _artifact(destination, directory / "audit.json")
        summaries.append({"arm_id": name, "feature_set": "reference", "estimator": "reference", **summary})
    reports = destination / "reports"
    reports.mkdir()
    _write_table(reports, "comparison", pd.DataFrame(summaries), {})
    artifacts["comparison"] = _artifact(destination, reports / "comparison.parquet")
    _verify_artifacts(plan["root"], plan["provenance"]["source_pins"])
    _json(destination / "loading_provenance.json", {**plan["provenance"],
        "financial_values_read": True, "decoded_outcome_years": list(VALIDATION_YEARS),
        "new_estimator_fits": 0, "preprocessing_fits": 0, "source_files_unchanged": True})
    for filename in ("build_identity.json", "read_plan.json", "ATTEMPT.json", "loading_provenance.json"):
        artifacts[filename] = _artifact(destination, destination / filename)
    usage = _limits(plan["recipe"], started, destination, plan["provenance"]["projected_compressed_bytes"])
    summary = {"newly_fitted_arms": 0, "reused_fitted_arms": 4, "preprocessing_fits": 0,
        "schemas": schemas, "train_fit_rows": plan["parent_audit"]["summary"]["train_fit_rows"],
        "train_inputs_bytes_unchanged": True, "fitted_models_bytes_unchanged": True,
        "preprocessors_bytes_unchanged": True, "training_financial_values_decoded": False,
        "validation_original_rows": len(ledger), "validation_prediction_rows": len(prediction_index),
        "validation_labelled_rows": int(ledger.target_available.sum()),
        "validation_missing_label_prediction_rows": int((prediction_mask & ~ledger.target_available).sum()),
        "validation_months": 24, "financial_2020_plus_decoded": False,
        "same_prediction_keys_all_arms": True, "same_observed_label_mask_all_arms": True,
        "validation_labels_used_for_fit_or_stopping": False, "parameter_search_performed": False,
        "fitted_parent_model_identity_sha256": plan["parent_audit"]["model_identity_sha256"],
        "data_limited_reconstruction": True, "historical_identity_certified": False,
        "knowledge_time_certified": False, "resource_usage": usage,
        "elapsed_seconds": time.monotonic() - started}
    _json(destination / "audit.json", {"status": "COMPLETE", "phase": PHASE,
        "build_identity_sha256": identity_sha256, "model_identity_sha256": phase_pin,
        "summary": summary, "artifacts": artifacts, "arms": arms})
    read_validation_reprediction(destination, identity_sha256=identity_sha256,
        model_identity_sha256=phase_pin, root=plan["root"])
    return destination
