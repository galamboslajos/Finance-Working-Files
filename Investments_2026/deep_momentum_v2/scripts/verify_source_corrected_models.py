"""Independent corrected-source replay checks. Never imports fitting workers.

The frozen independent V2 numerical verifier is hash pinned. Only its run
orchestration, dictionary reference chain and reviewed-source checks are adapted.
All 715 training statistics/matrices and saved inference retain its exact checks.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
from importlib.metadata import version
import json
from pathlib import Path
import signal
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

BASE_VERIFIER_SHA = "3d9b6dd057341919f1b2c7b886ac327bc14d591ef50ee42f68a850a8fffbcab9"
ROOT_PHASE = "source_corrected_walkforward_2020_2023_v1"
CORRECTED = {2010, 2011, 2012, 2013, 2022, 2023}
MODEL_DETAILS = {}


def sha(path):
    state = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            state.update(block)
    return state.hexdigest()


def need(condition, message):
    if not condition:
        raise AssertionError(message)


def js(path):
    return json.loads(Path(path).read_text())


def save_once(path, value):
    path = Path(path).resolve()
    need(not path.exists(), "Preserve an existing independent receipt")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def metadata_pin(root, path):
    root, path = Path(root).resolve(), Path(path).resolve()
    need(path.is_relative_to(root) and path.is_file(), "Metadata seal leaves its approved root")
    return {"path": str(path.relative_to(root)), "size_bytes": path.stat().st_size, "sha256": sha(path)}


def import_frozen_verifier(root):
    source = root / "scripts/verify_walkforward_models_v2.py"
    need(sha(source) == BASE_VERIFIER_SHA, "Frozen independent verifier changed")
    spec = importlib.util.spec_from_file_location("_corrected_independent_v2", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _artifact_by_path(artifacts, relative):
    rows = [item for item in artifacts if item["path"] == relative]
    need(len(rows) == 1, "Required handoff artifact missing or duplicated: " + relative)
    return rows[0]


def _reviewed_sources(root, run, identity, audit, v):
    config = js(root / "configs/source_quality_repair_20261016.json")
    handoff_path = v._pin_path(root, config["handoff"])
    need(identity["source_correction_handoff_pin"] == config["handoff"], "Replay changed handoff pin")
    handoff = js(handoff_path)
    need(handoff["status"] == "REPAIRED_INPUTS_READY_MODELS_STALE"
         and handoff["financial_2024_plus_analysis"] is False, "Wrong source-repair handoff")
    manifest_pin = identity["source_correction_manifest_pin"]
    review_pin = identity["independent_corrected_input_review_pin"]
    for item in (manifest_pin, review_pin):
        need(_artifact_by_path(handoff["artifacts"], item["path"]) == item,
             "Required source evidence is not bound by the handoff")
        v._pin_path(root, item)
    manifest = js(root / manifest_pin["path"])
    need(manifest["financial_2024_plus_analysis"] is False
         and manifest["context_policy"] == "available_cohort_80pct", "Wrong source manifest scope")
    entries = manifest["years"]
    need([row["year"] for row in entries] == list(range(2010, 2024)), "Need fourteen ordered annual sources")
    need({row["year"] for row in entries if row["corrected"]} == CORRECTED,
         "Corrected annual source set changed")
    need(js(root / review_pin["path"])["status"] == "PASS_BOUNDED_SIX_ANNUAL_INPUT_RECOMPUTATION",
         "Actual independent repaired-input review is incomplete")
    need(set(identity["annual_sources"]) == {str(year) for year in range(2010, 2024)}
         and set(identity["reviewed_source_annuals"]) == set(identity["annual_sources"]),
         "Replay source references are incomplete")
    checked = []
    for entry in entries:
        year = entry["year"]
        source_directory = v.local(root, entry["directory"])
        source_audit_path = source_directory / "audit.json"
        need(sha(source_audit_path) == entry["audit_sha256"], "Reviewed annual source audit changed")
        source_audit = js(source_audit_path)
        need(source_audit["year"] == year and source_audit["status"] == "COMPLETE"
             and source_audit["build_identity_sha256"] == entry["build_identity_sha256"],
             "Wrong reviewed annual source identity")
        recorded = identity["reviewed_source_annuals"][str(year)]
        for field in ("directory", "build_identity_sha256", "corrected", "original_rows"):
            need(recorded[field] == entry[field], "Replay aliases a different source annual: " + field)
        need(v._pin_path(root, recorded["audit_pin"]) == source_audit_path,
             "Replay audit pin redirects reviewed source")
        directory, annual = v.annual_directory(root, run, identity, audit, year)
        if year < 2020:
            need(directory == source_directory, "Historical annual source redirect")
        else:
            lineage = annual["source_corrected_replay_lineage"]
            need(lineage == {**recorded, "numerical_artifact_bytes_unchanged": True,
                            "new_feature_calculations": 0,
                            "original_audit_preserved_as": "reference_annual_audit.json"},
                 "Copied outcome annual lost exact reuse provenance")
            reference = v._pin_path(directory, annual["artifacts"]["reference_annual_audit"])
            need(reference == directory / "reference_annual_audit.json"
                 and sha(reference) == entry["audit_sha256"], "Copied source audit changed")
            need(set(annual["artifacts"]) == {*source_audit["artifacts"], "reference_annual_audit"},
                 "Copied annual changed its numerical artifact inventory")
            for name, item in source_audit["artifacts"].items():
                need(annual["artifacts"][name] == item, "Copied numerical artifact pin changed: " + name)
                v._pin_path(directory, item)
        paths, metadata = v.annual_metadata(directory, annual, year)
        need(len(metadata) == entry["original_rows"], "Original cohort denominator changed")
        checked.append({"year": year, "corrected": entry["corrected"],
                        "source_directory": entry["directory"],
                        "model_directory": str(directory.relative_to(root)),
                        "original_rows": len(metadata),
                        "source_audit_sha256": entry["audit_sha256"]})
    return checked


def setup(root, run):
    root, run = Path(root).resolve(), Path(run).resolve()
    need(run.is_relative_to(root / "outputs/walkforward_source_corrected"), "Unapproved replay namespace")
    v = import_frozen_verifier(root)
    identity, audit = js(run / "build_identity.json"), js(run / "audit.json")
    identity_pin = v.canonical(identity)
    need(identity["phase"] == ROOT_PHASE and audit["status"] == "COMPLETE"
         and audit["phase"] == "reviewed_annual_input_reuse_only"
         and audit["build_identity_sha256"] == identity_pin
         and audit["model_fits"] == 0 and audit["backtest_runs"] == 0,
         "Fresh root must certify input reuse, not model/backtest completion")
    cfg = identity["configuration"]
    need(cfg == js(root / "configs/walkforward_2020_2023.json")
         and identity["configuration_sha256"] == v.canonical(cfg), "Frozen configuration changed")
    for package, expected in cfg["model_recipe"]["packages"].items():
        need(version(package) == expected, "Pinned inference package changed: " + package)
    scope = identity["scope"]
    for key, value in {"new_estimator_fits": 16, "new_preprocessor_fits": 8,
                       "parameter_search": False, "primary_context_coverage": .8,
                       "sensitivity_90pct_fits": 0, "new2018_2019_development_fits": 0,
                       "financial_2024_plus_analysis": False,
                       "new_cloud_reads_or_mutations": False, "existing_runs_changed": False}.items():
        need(scope[key] == value, "Replay scope changed: " + key)
    need(identity["historical_identity_or_survivorship_certified"] is False,
         "Provisional source policy was promoted")
    for name, expected in identity["workers"].items():
        need(sha(root / "src/dmv2" / name) == expected, "Frozen worker changed: " + name)
    need(sha(root / "src/dmv2/source_corrected_replay.py") == identity["source_replay_worker_sha256"],
         "New additive preparer changed")
    preserved = identity["preserved_reference_pins"]
    for item in preserved.values():
        v._pin_path(root, item)
    old_relative = "outputs/walkforward/2020-2023-aacd7579fcf21e6a/build_identity.json"
    need(old_relative in preserved, "Exact old dictionary/reference identity is not pinned")
    old_identity_path = v._pin_path(root, preserved[old_relative])
    old_identity = js(old_identity_path)
    need(old_identity["configuration"] == cfg, "Reference model recipe changed")
    need(identity["feature_dictionary_sha256"] == old_identity["feature_dictionary_sha256"],
         "Replay changed the canonical feature dictionary")
    checked = _reviewed_sources(root, run, identity, audit, v)
    # Only immutable dictionary-reference metadata is transported in memory.
    # The annual data resolver still sees all new source/copy references.
    dictionary_identity = {**identity, "feature_dictionary_sha256": old_identity["feature_dictionary_sha256"],
                           "years": old_identity["years"]}
    # Dictionary-only metadata does not alter the root identity used to admit
    # any annual. THIS_BUILD audits remain checked against the actual replay.
    original_resolver = v.annual_directory
    v.annual_directory = lambda _root, _run, _identity, _audit, year: original_resolver(
        root, run, identity, audit, year)
    try:
        dictionary, proof = v.resolve_feature_dictionary(root, run, dictionary_identity, audit)
    finally:
        v.annual_directory = original_resolver
    proof["transported_from_preserved_reference_identity"] = str(old_identity_path.relative_to(root))
    proof["transported_reference_raw_sha256"] = sha(old_identity_path)
    budget = v.VerificationBudget(cfg["execution"])
    budget.check()
    return v, identity, audit, cfg, dictionary, proof, checked, budget


def verify_actual_model(v, directory, kind, prepared, expected):
    """Validate reloadable structure/parameters and reconcile saved probabilities."""
    from threadpoolctl import threadpool_limits
    checkpoint = v.json_file(directory / "FIT_COMPLETE.json")
    parameters = checkpoint["metadata"]["explicit_parameters"]
    effective = checkpoint["metadata"]["effective_parameters"]
    need(all(effective.get(key) == value for key, value in parameters.items()),
         "Pre-save effective parameters differ from the explicit recipe")
    if kind == "rf":
        import joblib
        estimator = joblib.load(directory / "model.joblib")
        actual_parameters = estimator.get_params(deep=False)
        need(all(actual_parameters.get(key) == value for key, value in parameters.items()),
             "Serialized RF parameters differ from the frozen recipe")
        need(list(estimator.classes_) == list(range(1, 11)), "Reloaded RF class mapping changed")
        need(len(estimator.estimators_) == parameters["n_estimators"]
             and estimator.n_features_in_ == prepared.shape[1], "Serialized RF tree count/width changed")
        need(all(tree.tree_.max_depth <= parameters["max_depth"] for tree in estimator.estimators_),
             "Serialized RF tree exceeds maximum depth")
        with threadpool_limits(limits=1):
            actual = estimator.predict_proba(prepared)
        details = {"serialized_explicit_parameters_verified": True,
                   "tree_count": len(estimator.estimators_), "feature_width": int(estimator.n_features_in_)}
    else:
        from scipy.special import softmax
        from xgboost import XGBClassifier
        estimator = XGBClassifier(n_jobs=1, device="cpu")
        estimator.load_model(directory / "model.ubj")
        estimator.set_params(n_jobs=1, device="cpu")
        need(list(estimator.classes_) == list(range(10)), "Reloaded XGBoost classes changed")
        booster = estimator.get_booster()
        config = json.loads(booster.save_config())["learner"]
        need(config["objective"]["name"] == parameters["objective"]
             and int(config["learner_model_param"]["num_class"]) == parameters["num_class"]
             and int(config["learner_model_param"]["num_feature"]) == prepared.shape[1]
             and booster.num_boosted_rounds() == parameters["n_estimators"],
             "Serialized XGBoost objective/classes/width/boosting rounds differ")
        need(booster.attr("best_iteration") is None and booster.attr("best_score") is None,
             "Unexpected early-stopping state in serialized XGBoost")
        with threadpool_limits(limits=1):
            margins = estimator.predict(prepared, output_margin=True)
        actual = softmax(np.asarray(margins, dtype=np.float64), axis=1)
        details = {"serialized_objective_class_width_rounds_verified": True,
                   "boosting_rounds": booster.num_boosted_rounds(), "feature_width": booster.num_features(),
                   "all_training_hyperparameters_persisted_by_ubj": False,
                   "training_parameters_evidence": "pinned constructor plus pre-save explicit/effective receipt"}
    difference = float(np.max(np.abs(actual - expected)))
    need(np.allclose(actual, expected, rtol=1e-12, atol=2e-14), "Reloaded model probabilities differ")
    MODEL_DETAILS[str(directory)] = details
    del estimator, actual
    gc.collect()
    return difference


def verify(root, run, *, fold=None, metadata_only=False):
    started = time.monotonic()
    v, identity, audit, cfg, dictionary, proof, sources, budget = setup(root, run)
    root, run = Path(root).resolve(), Path(run).resolve()
    root_seals = [metadata_pin(root, run / name) for name in ("build_identity.json", "audit.json")]
    result = {"status": "METADATA_PASS" if metadata_only else "PASS",
              "run_directory": str(run), "root_identity_sha256": v.canonical(identity),
              "base_independent_verifier_sha256": BASE_VERIFIER_SHA,
              "adapter_sha256": sha(Path(__file__)), "annual_sources": sources,
              "feature_dictionary": proof, "models_fitted_by_verifier": 0,
              "preprocessors_fitted_by_verifier": 0, "financial_2024_plus_decoded": False,
              "financial_values_decoded": not metadata_only,
              "limits": ["All 715 preprocessing input statistics and matrices are checked; this does not independently certify all 715 feature formulas.",
                         "XGBoost UBJ does not persist every training hyperparameter; those are checked through pinned constructor code and the pre-save effective-parameter receipt.",
                         "PIT checks establish declared per-signal chronology under provisional retrieval-time identity and adjusted-return evidence, not full historical knowledge-time/survivorship certification."]}
    plans = {}
    for year in v.YEARS:
        months = []
        for y in range(year - 10, year):
            directory, annual = v.annual_directory(root, run, identity, audit, y)
            _, metadata = v.annual_metadata(directory, annual, y)
            months.extend(sorted(metadata.outcome_month.unique()))
        need(len(months) == 120 and len(set(months)) == 120
             and max(months) == f"{year-1}-12", "Fold does not have 120 completed outcome months")
        plans[str(year)] = {"train_start": f"{year-10}-01-01", "train_end": f"{year-1}-12-31",
                            "label_observed_through": v.CUTOFFS[year], "completed_outcome_months": 120}
    result["fold_windows"] = plans
    if not metadata_only:
        models = run / "models"
        phase = js(models / "build_identity.json")
        model_pin = v.canonical(phase)
        need(phase["parent_identity_sha256"] == v.canonical(identity)
             and phase["configuration_sha256"] == v.canonical(cfg), "Model phase changed root/configuration")
        v.verify_artifacts(root, phase["annual_source_audit_pins"])
        for name, expected in phase["workers"].items():
            need(sha(root / "src/dmv2" / name) == expected.removeprefix("sha256:"), "Model worker changed")
        v.verify_saved_model = lambda directory, kind, prepared, expected: verify_actual_model(v, directory, kind, prepared, expected)
        years = [fold] if fold is not None else list(v.YEARS)
        need(all(year in v.YEARS for year in years), "Unapproved fold")
        details = {}
        fold_seals = {}
        phase_seal = metadata_pin(root, models / "build_identity.json")
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            for year in years:
                fold_seals[str(year)] = metadata_pin(root, models / "folds" / str(year) / "audit.json")
                detail, frames = v.verify_fold(root, run, identity, audit, models, model_pin, year, budget, dictionary=dictionary)
                details[str(year)] = detail
                del frames, detail
                gc.collect()
                pa.default_memory_pool().release_unused()
                budget.check()
        result.update(model_identity_sha256=model_pin, folds=details,
                      estimator_structure=MODEL_DETAILS,
                      saved_models_reconciled=4 * len(years), preprocessor_states_reconciled=2 * len(years),
                      checked_fold_audit_pins=fold_seals, checked_model_phase_pin=phase_seal)
        for item in [*fold_seals.values(), phase_seal]:
            v._pin_path(root, item)
    for item in root_seals:
        v._pin_path(root, item)
    result["checked_root_metadata_pins"] = root_seals
    budget.check()
    result.update(elapsed_seconds=time.monotonic() - started, peak_rss_bytes=budget.peak)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--fold", type=int, choices=(2020, 2021, 2022, 2023))
    args = parser.parse_args(argv)
    need(not args.output.exists(), "Preserve existing receipt")
    need(not args.output.resolve().is_relative_to(args.run.resolve()), "Independent receipts stay outside the run")
    def timeout(signum, frame):
        raise AssertionError("Independent verifier exceeded its 30-minute ceiling")
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(1800)
    try:
        result = verify(args.root, args.run, fold=args.fold, metadata_only=args.metadata_only)
        save_once(args.output, result)
    finally:
        signal.alarm(0)
    print(json.dumps({key: result[key] for key in ("status", "root_identity_sha256", "models_fitted_by_verifier", "financial_values_decoded")}, indent=2))


if __name__ == "__main__":
    main()
