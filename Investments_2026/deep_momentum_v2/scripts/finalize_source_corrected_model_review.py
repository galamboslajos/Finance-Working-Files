"""Seal four no-fit fold reviews and reconcile the saved 48-month predictions.

Never reloads an estimator, fits preprocessing, or calls a production worker.
The saved fold-audit hashes bind each actual inference check to current files.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
from pathlib import Path
import signal
import time

import pandas as pd
import pyarrow as pa

ADAPTER_SHA = "d026f777a87445de5789e01279f457f15b5b7e98ef9ee197d5e9167e04d53c9d"


def load_adapter():
    import hashlib
    source = Path(__file__).with_name("verify_source_corrected_models.py")
    assert hashlib.sha256(source.read_bytes()).hexdigest() == ADAPTER_SHA, "Per-fold adapter changed"
    spec = importlib.util.spec_from_file_location("_corrected_final_adapter", source)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    return adapter


def finalize(root, run, receipt_paths):
    started = time.monotonic()
    a = load_adapter()
    root, run = Path(root).resolve(), Path(run).resolve()
    v, identity, root_audit, cfg, dictionary, proof, sources, budget = a.setup(root, run)
    identity_pin = v.canonical(identity)
    models = run / "models"
    phase = a.js(models / "build_identity.json")
    model_pin = v.canonical(phase)
    audit_path = models / "audit.json"
    audit = a.js(audit_path)
    a.need(audit["status"] == "COMPLETE" and audit["phase"] == "walkforward_models_v1"
           and audit["build_identity_sha256"] == identity_pin
           and audit["model_identity_sha256"] == model_pin
           and phase["parent_identity_sha256"] == identity_pin
           and phase["configuration_sha256"] == v.canonical(cfg), "Final model phase identity changed")
    summary = audit["summary"]
    a.need(summary["fitted_arms"] == 16 and summary["preprocessor_fits"] == 8
           and summary["outer_years"] == list(v.YEARS) and summary["months"] == 48,
           "Final phase does not contain exactly 16 fits/eight states/48 months")
    a.need(set(audit["folds"]) == {str(year) for year in v.YEARS}
           and set(audit["arms"]) == {*v.ARMS, *v.REFERENCES}, "Final fold/arm grid changed")
    a.need(set(receipt_paths) == set(v.YEARS)
           and len({Path(path).resolve() for path in receipt_paths.values()}) == 4,
           "Four distinct explicit fold receipts required")
    top_seals = [a.metadata_pin(root, path) for path in
                 (run / "build_identity.json", run / "audit.json", models / "build_identity.json", audit_path)]
    receipts, receipt_pins = {}, {}
    for year in v.YEARS:
        path = Path(receipt_paths[year]).resolve()
        a.need(path.is_file() and path.stat().st_size < 1024**2, "Fold receipt exceeds metadata budget")
        receipt = a.js(path)
        a.need(receipt["status"] == "PASS" and receipt["root_identity_sha256"] == identity_pin
               and receipt["model_identity_sha256"] == model_pin
               and receipt["adapter_sha256"] == ADAPTER_SHA
               and receipt["models_fitted_by_verifier"] == 0
               and receipt["preprocessors_fitted_by_verifier"] == 0
               and receipt["financial_2024_plus_decoded"] is False
               and receipt["financial_values_decoded"] is True
               and receipt["saved_models_reconciled"] == 4
               and receipt["preprocessor_states_reconciled"] == 2
               and set(receipt["folds"]) == {str(year)}, "Fold receipt scope changed")
        a.need(set(receipt["checked_fold_audit_pins"]) == {str(year)}, "Fold seal scope changed")
        fold = models / "folds" / str(year)
        seal = receipt["checked_fold_audit_pins"][str(year)]
        a.need(v._pin_path(root, seal) == fold / "audit.json", "Checked fold audit changed")
        a.need(v._pin_path(models, audit["folds"][str(year)]) == fold / "audit.json",
               "Final model manifest redirects its verified fold")
        a.need(v._pin_path(root, receipt["checked_model_phase_pin"]) == models / "build_identity.json",
               "Per-fold model phase changed")
        for item in receipt["checked_root_metadata_pins"]:
            v._pin_path(root, item)
        checked = a.js(fold / "audit.json")
        v.verify_artifacts(fold, checked["artifacts"])
        v.verify_artifacts(fold, checked["arms"])
        for name in (*v.ARMS, *v.REFERENCES):
            directory = fold / ("arms" if name in v.ARMS else "refs") / name
            state = a.js(directory / "audit.json")
            v.verify_artifacts(directory, state["artifacts"])
        receipts[str(year)] = receipt
        receipt_pins[str(year)] = {"absolute_path": str(path), "size_bytes": path.stat().st_size,
                                  "sha256": a.sha(path)}
        budget.check()
    a.need(len(list(models.glob("folds/*/arms/*/FIT_ATTEMPT.json"))) == 16
           and len(list(models.glob("folds/*/arms/*/FIT_COMPLETE.json"))) == 16,
           "Estimator attempts/completions are not exactly 16")
    artifact_count = sum(v.verify_artifacts(models, audit[key]) for key in ("artifacts", "arms", "folds"))
    stitched = {}
    for name in (*v.ARMS, *v.REFERENCES):
        directory = models / ("arms" if name in v.ARMS else "refs") / name
        state = a.js(directory / "audit.json")
        v.verify_artifacts(directory, state["artifacts"])
        a.need(state["status"] == "COMPLETE" and state["build_identity_sha256"] == identity_pin
               and state["model_identity_sha256"] == model_pin, "Aggregate arm identity changed")
        frame = v.read_table(directory / "predictions.parquet", first="2020-01", last="2023-12")
        parts = []
        for year in v.YEARS:
            path = models / "folds" / str(year) / ("arms" if name in v.ARMS else "refs") / name / "predictions.parquet"
            parts.append(v.read_table(path, first=f"{year}-01", last=f"{year}-12"))
        expected = pd.concat(parts, ignore_index=True)
        pd.testing.assert_frame_equal(frame, expected, check_dtype=False, check_exact=True)
        v.check_dates(frame, "2020-01", "2023-12")
        eligible = frame.loc[frame.prediction_eligible].reset_index(drop=True)
        metrics = v.verify_metrics(directory, eligible, state)
        v.verify_mode_metrics(directory, eligible)
        stitched[name] = {"original_rows": len(frame), "prediction_rows": len(eligible),
                          "observed_labels": int(eligible.target_class.notna().sum()),
                          "outcome_months": frame.outcome_month.nunique(), "metrics": metrics}
        del frame, expected, eligible, parts
        gc.collect()
        pa.default_memory_pool().release_unused()
        budget.check()
    for item in top_seals:
        v._pin_path(root, item)
    result = {"status": "PASSED", "schema": "dmv2-source-corrected-model-independent-review/v1",
              "build_identity_sha256": identity_pin, "model_identity_sha256": model_pin,
              "root_audit_sha256": a.sha(run / "audit.json"), "models_audit_sha256": a.sha(audit_path),
              "annual_fits_reconciled": 16, "preprocessor_states_reconciled": 8,
              "stitched_prediction_months": 48, "aggregate_arms_and_references_reconciled": 6,
              "models_fitted_by_verifier": 0, "preprocessors_fitted_by_verifier": 0,
              "models_reloaded_during_final_aggregation": 0,
              "financial_2024_plus_decoded": False, "maximum_financial_outcome_decoded": "2023-12-31",
              "base_independent_verifier_sha256": a.BASE_VERIFIER_SHA,
              "per_fold_adapter_sha256": ADAPTER_SHA, "verifier_sha256": a.sha(Path(__file__)),
              "artifact_files_verified": artifact_count, "original_fit_markers": 16,
              "fold_receipt_pins": receipt_pins,
              "folds": {year: receipt["folds"][year] for year, receipt in receipts.items()},
              "stitched": stitched, "annual_sources": sources, "feature_dictionary_source": proof,
              "checked_root_and_model_phase_pins": top_seals,
              "limits": next(iter(receipts.values()))["limits"],
              "execution_seconds": time.monotonic() - started, "peak_rss_bytes": budget.peak}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold-receipt", action="append", required=True,
                        help="Exactly four YEAR=/absolute/path mappings")
    args = parser.parse_args(argv)
    adapter = load_adapter()
    adapter.need(not args.output.exists() and not args.output.resolve().is_relative_to(args.run.resolve()),
                 "Use a fresh independent receipt outside the run")
    pairs = [item.split("=", 1) for item in args.fold_receipt]
    adapter.need(len(pairs) == 4 and all(len(item) == 2 for item in pairs)
                 and len({item[0] for item in pairs}) == 4, "Four unique explicit fold receipts required")
    paths = {int(year): Path(path) for year, path in pairs}
    def timeout(signum, frame):
        raise AssertionError("Final verification exceeded its 30-minute ceiling")
    signal.signal(signal.SIGALRM, timeout); signal.alarm(1800)
    try:
        result = finalize(args.root, args.run, paths)
        adapter.save_once(args.output, result)
    finally:
        signal.alarm(0)
    print(json.dumps({key: result[key] for key in ("status", "build_identity_sha256", "annual_fits_reconciled",
                                                "preprocessor_states_reconciled", "models_fitted_by_verifier")}, indent=2))


if __name__ == "__main__": main()
