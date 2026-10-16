"""Independently verify completed 2018--2019 repair evidence without refitting.

This script never imports research compute workers or serialized estimators.
Old training/source data are byte-hashed only. Financial table decoding is
limited to the completed new/parent 2018--2019 validation artifacts. The only
write is a small JSON receipt explicitly requested under /private/tmp.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PARENT = "sha256:50aa6e09f11b687c36cd3cf52ed8eb51fdde591fb2391cd58162d02edbb29b10"
ARMS = ("rf_paper15", "xgb_paper15", "rf_extended", "xgb_extended")
KEYS = ["source_security_id", "signal_date"]
PROBS = [f"probability_{i}" for i in range(1, 11)]
YEARS = (2018, 2019)
MAX_RSS = 2 * 1024**3
MAX_SECONDS = 900


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def canonical(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return "sha256:" + hashlib.sha256(raw.encode()).hexdigest()


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            value.update(block)
    return "sha256:" + value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def local(base, relative):
    base = Path(base).resolve()
    path = (base / relative).resolve()
    require(path.is_relative_to(base), "Pinned artifact escapes its declared directory")
    return path


def equal_missing(left, right):
    a, b = np.asarray(left), np.asarray(right)
    require(a.shape == b.shape, "Compared financial arrays have different shapes")
    return (a == b) | (pd.isna(a) & pd.isna(b))


class Verifier:
    def __init__(self, root, run, *, backtest_folder="backtest"):
        self.root = Path(root).resolve()
        self.run = local(self.root, run)
        require(self.run.is_relative_to(self.root / "outputs/independent_backtest"), "Wrong verification output namespace")
        require(backtest_folder in {"backtest", "backtest_completed"}, "Unknown bounded backtest phase")
        self.backtest_folder = backtest_folder
        self.started = time.monotonic()
        self.pins = {}
        self.cache = {}
        self.decoded_paths = set()

    def budget(self):
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss = int(rss if sys.platform == "darwin" else rss * 1024)
        require(rss <= MAX_RSS, "Independent verifier exceeded its 2 GiB peak RSS ceiling")
        require(time.monotonic() - self.started <= MAX_SECONDS, "Independent verifier exceeded its 900-second ceiling")
        return rss

    def pin(self, base, item, *, refresh=False):
        require(isinstance(item, dict) and {"path", "size_bytes", "sha256"} <= set(item), "Incomplete file pin")
        path = local(base, item["path"])
        require(path.is_relative_to(self.root), "Source pin leaves the approved workspace")
        require(path.is_file() and path.stat().st_size == item["size_bytes"], "Pinned file size changed: " + str(path))
        expected = "sha256:" + item["sha256"].removeprefix("sha256:")
        stat_key = (path.stat().st_size, path.stat().st_mtime_ns)
        if refresh or self.cache.get(path, (None, None))[0] != stat_key:
            self.cache[path] = (stat_key, digest(path))
        require(self.cache[path][1] == expected, "Pinned file hash changed: " + str(path))
        relative = str(path.relative_to(self.root))
        entry = {"path": relative, "size_bytes": item["size_bytes"], "sha256": expected}
        require(relative not in self.pins or self.pins[relative] == entry, "Conflicting source pins")
        self.pins[relative] = entry
        self.budget()
        return path

    def own_pin(self, path):
        path = local(self.root, path)
        return {"path": str(path.relative_to(self.root)), "size_bytes": path.stat().st_size,
            "sha256": digest(path)}

    def manifest(self, base, entries):
        records = entries.values() if isinstance(entries, dict) else entries
        return [self.pin(base, item) for item in records]

    def workers(self, phase):
        for name, expected in phase.get("workers", {}).items():
            require(Path(name).name == name, "Malformed worker filename")
            path = self.root / "src/dmv2" / name
            require(digest(path) == "sha256:" + expected.removeprefix("sha256:"), "Frozen worker changed: " + name)
        return len(phase.get("workers", {}))

    def table(self, path, columns=None):
        self.decoded_paths.add(str(Path(path).relative_to(self.root)))
        result = pq.ParquetFile(path).read(columns=columns).to_pandas()
        self.budget()
        return result

    def dated(self, path, *, year=None):
        fields = set(pq.ParquetFile(path).schema_arrow.names)
        columns = KEYS + [name for name in ("outcome_month", "outcome_start", "outcome_end") if name in fields]
        metadata = self.table(path, columns)
        require(not metadata[KEYS].isna().any().any() and not metadata.duplicated(KEYS).any(), "Duplicate/missing original dated keys")
        require(metadata.signal_date.map(lambda day: isinstance(day, str)
            and "2017-12-01" <= day <= "2019-11-30").all(), "Financial table metadata exceeds validation scope")
        if year is not None:
            following = metadata.signal_date.map(lambda day: str(pd.Period(day[:7], "M") + 1))
            require(following.str.startswith(str(year) + "-").all(), "Annual signal metadata crosses its outcome year")
        if "outcome_month" in metadata:
            for row in metadata.itertuples(index=False):
                require(str(pd.Period(row.signal_date[:7], "M") + 1) == row.outcome_month
                    and row.signal_date < row.outcome_start <= row.outcome_end
                    and row.outcome_start[:7] == row.outcome_month == row.outcome_end[:7]
                    and "2018-01" <= row.outcome_month <= "2019-12", "Outcome overlaps features or leaves validation scope")
        return metadata

    def annual(self, directory, year, identity):
        audit_path = directory / "audit.json"
        audit = read_json(audit_path)
        require(audit.get("status") == "COMPLETE" and audit.get("year") == year
            and audit.get("build_identity_sha256") == identity, "Incomplete or misidentified annual build")
        self.pin(self.root, self.own_pin(audit_path))
        self.manifest(directory, audit["artifacts"])
        windows = audit["feature_provenance"]["target_windows"]
        require(len(windows) == 12 and {entry["outcome_month"] for entry in windows.values()}
            == {f"{year}-{month:02d}" for month in range(1, 13)}, "Annual feature clock lost a validation month")
        keys = None
        for name in ("original_cohort", "features", "feature_coverage", "targets"):
            path = self.pin(directory, audit["artifacts"][name])
            meta = self.dated(path, year=year)
            current = pd.MultiIndex.from_frame(meta[KEYS])
            require(set(meta.signal_date) == set(windows), "Annual table lost a declared signal")
            require(keys is None or set(current) == set(keys), "Annual table changed original membership denominator")
            keys = current if keys is None else keys
        if "daily_prices_raw" in audit["artifacts"]:
            path = self.pin(directory, audit["artifacts"]["daily_prices_raw"])
            metadata = self.table(path, ["trading_date", "intrinio_security_id"])
            require(not metadata.duplicated(["trading_date", "intrinio_security_id"]).any()
                and metadata.trading_date.map(lambda day: isinstance(day, str)
                    and f"{year}-01-01" <= day <= f"{year}-12-31").all(), "Annual execution price metadata leaves authorized dates")
        return audit, keys

    def compare_annual(self, parent, year, pin, paper15):
        directory, original = self.run / "years" / str(year), parent / "years" / str(year)
        new_audit, new_keys = self.annual(directory, year, pin)
        old_audit, old_keys = self.annual(original, year, PARENT)
        require(set(new_keys) == set(old_keys), "Independent repair changed historical membership")
        coverage_cols = KEYS + ["intrinio_security_id", "identity_eligible", "requested", "paper_prediction_eligible"]
        new_coverage = self.table(directory / "feature_coverage.parquet", coverage_cols).set_index(KEYS)
        old_coverage = self.table(original / "feature_coverage.parquet", coverage_cols).set_index(KEYS).reindex(new_coverage.index)
        for field in ("intrinio_security_id", "identity_eligible", "requested"):
            require(equal_missing(new_coverage[field], old_coverage[field]).all(), "Quote repair changed identity/request cohort: " + field)
        eligible_changed = ~equal_missing(new_coverage.paper_prediction_eligible, old_coverage.paper_prediction_eligible)
        new_features = self.table(directory / "features.parquet").set_index(KEYS)
        old_features = self.table(original / "features.parquet").set_index(KEYS).reindex(new_features.index)
        require(list(new_features) == list(old_features) and len(new_features.columns) == 728, "Registered feature definitions/order changed")
        old_values, new_values = old_features.to_numpy(dtype=float), new_features.to_numpy(dtype=float)
        changed = ~equal_missing(old_values, new_values)
        baseline = np.isfinite(new_features[paper15].to_numpy(dtype=float)).all(axis=1)
        coverage = new_coverage.reindex(new_features.index)
        require(np.array_equal(baseline, coverage.paper_prediction_eligible.to_numpy(dtype=bool)), "Paper15 feature availability differs from prediction admission")
        feature_counts = {name: int(changed[:, column].sum()) for column, name in enumerate(new_features) if changed[:, column].any()}
        paper_columns = [new_features.columns.get_loc(name) for name in paper15]
        summary = {"year": year, "original_rows": len(new_features),
            "membership_keys_unchanged": True, "identity_and_request_cohorts_unchanged": True,
            "old_prediction_eligible_rows": int(old_coverage.paper_prediction_eligible.sum()),
            "new_prediction_eligible_rows": int(new_coverage.paper_prediction_eligible.sum()),
            "prediction_eligibility_rows_changed": int(eligible_changed.sum()),
            "registered_features": 728, "features_with_any_changed_cell": len(feature_counts),
            "feature_cells_changed": int(changed.sum()), "rows_with_changed_feature": int(changed.any(axis=1).sum()),
            "paper15_cells_changed": int(changed[:, paper_columns].sum()),
            "paper15_rows_with_changed_value": int(changed[:, paper_columns].any(axis=1).sum()),
            "changed_cells_by_feature": feature_counts}
        del old_values, new_values, old_features, new_features, changed
        gc.collect()
        target_columns = KEYS + ["prediction_eligible", "target_available", "target_class", "label_return"]
        new_targets = self.table(directory / "targets.parquet", target_columns).set_index(KEYS)
        old_targets = self.table(original / "targets.parquet", target_columns).set_index(KEYS).reindex(new_targets.index)
        require(new_targets.prediction_eligible.equals(new_coverage.reindex(new_targets.index).paper_prediction_eligible), "New target availability selected forecast rows")
        require(~(new_targets.target_available & ~new_targets.prediction_eligible).any(), "Ineligible row acquired a training/validation class")
        require(np.array_equal(new_targets.target_available.to_numpy(dtype=bool), new_targets.target_class.notna().to_numpy()), "Missing target class was imputed")
        summary.update(old_labelled_rows=int(old_targets.target_available.sum()),
            new_labelled_rows=int(new_targets.target_available.sum()),
            target_availability_rows_changed=int((~equal_missing(new_targets.target_available, old_targets.target_available)).sum()),
            target_class_rows_changed=int((~equal_missing(new_targets.target_class, old_targets.target_class)).sum()),
            label_return_rows_changed=int((~equal_missing(new_targets.label_return, old_targets.label_return)).sum()))
        self.budget()
        return summary

    def matrices(self, models, parent, ledger):
        result = {}
        prediction_index = ledger.index[ledger.prediction_eligible]
        for feature_set in ("paper15", "expanded"):
            original_manifest = parent / "models/preprocessors" / feature_set / "manifest.json"
            new_manifest = models / "preprocessors" / feature_set / "manifest.json"
            require(digest(original_manifest) == digest(new_manifest), "Training-only preprocessing manifest changed")
            state = read_json(original_manifest)
            manifest = state["manifest"]
            require(canonical(manifest) == state["fingerprint"] and manifest["train_start"] == "2010-01-01"
                and manifest["train_end"] == "2017-12-31" and manifest["label_observed_through"] <= "2017-12-31"
                and manifest["training_only_schema_selection"] is True, "Saved medians/schema exceed completed training window")
            names = manifest["input_columns"]
            sources = [self.table(self.run / "years" / str(year) / "features.parquet", KEYS + names).set_index(KEYS) for year in YEARS]
            numeric = pd.concat(sources).reindex(prediction_index).astype(float).replace([np.inf, -np.inf], np.nan)
            expected_columns = [column for name in manifest["retained_columns"] for column in (name, name + "__missing")]
            require(expected_columns == manifest["output_columns"], "Missingness indicator schema changed")
            saved_path = models / "preprocessors" / feature_set / "validation_X.npy"
            saved = np.load(saved_path, mmap_mode="r", allow_pickle=False)
            require(saved.dtype == np.float32 and saved.shape == (len(prediction_index), len(expected_columns)), "Prepared validation matrix shape/dtype changed")
            for column, name in enumerate(manifest["retained_columns"]):
                values = numeric[name]
                expected = values.fillna(manifest["medians"][name]).to_numpy(dtype=np.float32)
                indicators = values.isna().to_numpy(dtype=np.float32)
                require(np.isfinite(expected).all() and np.array_equal(saved[:, 2 * column], expected)
                    and np.array_equal(saved[:, 2 * column + 1], indicators), "Prepared matrix differs from immutable train-only median/missingness transform")
            result[feature_set] = {"frozen_manifest_sha256": digest(original_manifest),
                "manifest_fingerprint": state["fingerprint"], "fit_rows": manifest["fit_rows"],
                "prediction_rows": len(prediction_index), "matrix_width": len(expected_columns),
                "matrix_reconstructed_exactly": True, "new_preprocessing_fits": 0}
            del sources, numeric, saved
            gc.collect()
            self.budget()
        return result

    def verify(self):
        # No numerical table reads occur before every requested phase is complete.
        root_audit = read_json(self.run / "audit.json")
        model_audit = read_json(self.run / "models/audit.json")
        bt_audit = read_json(self.run / self.backtest_folder / "audit.json")
        require(all(audit.get("status") == "COMPLETE" for audit in (root_audit, model_audit, bt_audit)), "All three phases must complete before independent verification")
        identity = read_json(self.run / "build_identity.json")
        pin = canonical(identity)
        require(self.run.name == "2018-2019-" + pin[7:23] and root_audit["build_identity_sha256"] == pin, "New root identity/directory mismatch")
        cfg = identity["configuration"]
        require(cfg["execution"]["maximum_new_model_fits"] == cfg["execution"]["maximum_new_preprocessor_fits"] == 0
            and cfg["feature_outcome_years"] == [2018, 2019], "Wrong independent repair compute scope")
        parent = local(self.root, cfg["validation_reprediction"]["fitted_parent_directory"])
        require(canonical(read_json(parent / "build_identity.json")) == PARENT, "Original fitted parent identity changed")
        preservation = read_json(self.run / "source_preservation_manifest.json")
        self.manifest(self.root, preservation)
        self.manifest(self.run, root_audit["years"])
        worker_counts = {"feature_workers": self.workers(identity)}
        paper15 = read_json(parent / "models/preprocessors/paper15/manifest.json")["manifest"]["input_columns"]
        require(len(paper15) == 15, "Paper15 schema changed")
        years = [self.compare_annual(parent, year, pin, paper15) for year in YEARS]
        models = self.run / "models"
        require(model_audit["phase"] == "validation_reprediction_v1"
            and model_audit["build_identity_sha256"] == pin
            and model_audit["summary"]["newly_fitted_arms"] == 0
            and model_audit["summary"]["reused_fitted_arms"] == 4
            and model_audit["summary"]["preprocessing_fits"] == 0, "Inference claims a new fit or wrong parent")
        self.manifest(models, model_audit["artifacts"])
        self.manifest(models, model_audit["arms"])
        phase = read_json(models / "build_identity.json")
        require(canonical(phase) == model_audit["model_identity_sha256"] and phase["parent_identity_sha256"] == pin, "Inference phase identity changed")
        self.manifest(self.root, phase["data_plan"]["source_pins"])
        worker_counts["inference_workers"] = self.workers(phase)
        ledger_path = models / "data/validation_ledger.parquet"
        self.dated(ledger_path)
        ledger = self.table(ledger_path).set_index(KEYS)
        require(len(ledger) == sum(year["original_rows"] for year in years), "Inference changed original validation population")
        matrices = self.matrices(models, parent, ledger)
        arms, shared_metadata = {}, None
        for name in ARMS:
            directory = models / "arms" / name
            audit = read_json(directory / "audit.json")
            self.manifest(directory, audit["artifacts"])
            metadata = audit["metadata"]
            require(audit["build_identity_sha256"] == pin and audit["model_identity_sha256"] == model_audit["model_identity_sha256"]
                and metadata["new_fit_count"] == metadata["preprocessing_fit_count"] == 0
                and metadata["parent_model_identity_sha256"] == phase["data_plan"]["fitted_parent_model_identity_sha256"], "Arm lost the original fit identity")
            self.pin(self.root, metadata["parent_model"])
            original = read_json(parent / "models/arms" / name / "audit.json")
            require(metadata["fit_keys_sha256"] == original["metadata"]["fit_keys_sha256"]
                and metadata["prepared_train_matrix_sha256"] == original["metadata"]["prepared_train_matrix_sha256"], "Original training inputs or fit rows changed")
            path = directory / "predictions.parquet"
            self.dated(path)
            fields = [*KEYS, "prediction_eligible", "paper_prediction_eligible", "target_available", "target_class", *PROBS]
            frame = self.table(path, fields).set_index(KEYS)
            require(frame.index.equals(ledger.index), "Forecast ledger changed original key order")
            require(frame.prediction_eligible.equals(ledger.prediction_eligible)
                and equal_missing(frame.target_class, ledger.target_class).all(), "Forecast admission or labels differ from common ledger")
            meta = frame[["prediction_eligible", "paper_prediction_eligible", "target_available", "target_class"]]
            require(shared_metadata is None or meta.equals(shared_metadata), "Models use different cohort or observed-label masks")
            shared_metadata = meta if shared_metadata is None else shared_metadata
            probabilities = frame.loc[frame.prediction_eligible, PROBS].to_numpy(dtype=float)
            require(np.isfinite(probabilities).all() and (probabilities >= 0).all()
                and np.allclose(probabilities.sum(axis=1), 1., rtol=0, atol=1e-8)
                and frame.loc[~frame.prediction_eligible, PROBS].isna().all().all(), "Saved probability vectors are malformed or ineligible rows were predicted")
            arms[name] = {"parent_model_sha256": metadata["parent_model"]["sha256"],
                "fit_rows": metadata["train_rows"], "prediction_rows": int(frame.prediction_eligible.sum()),
                "original_rows": len(frame), "new_fit_count": 0}
            del frame, probabilities
            gc.collect()
        backtest = self.run / self.backtest_folder
        expected_phase, expected_events = (("independent_quote_backtest_v1", 9)
            if self.backtest_folder == "backtest" else ("independent_quote_action_completion_v1", 11))
        require(bt_audit["phase"] == expected_phase and bt_audit["build_identity_sha256"] == pin
            and bt_audit["models_fitted"] is False and bt_audit["financial_2020_plus_read"] is False, "Wrong backtest phase/scope")
        self.manifest(backtest, bt_audit["artifacts"])
        bt_phase = read_json(backtest / "build_identity.json")
        require(canonical(bt_phase) == bt_audit["phase_identity_sha256"]
            and len(bt_phase["events"]) == expected_events, "Backtest identity or reviewed corporate-action count changed")
        self.manifest(self.root, bt_phase["source_pins"])
        if self.backtest_folder == "backtest_completed":
            old = self.run / "backtest"
            old_audit = read_json(old / "audit.json")
            old_phase = read_json(old / "build_identity.json")
            self.pin(self.root, self.own_pin(old / "audit.json"))
            self.pin(self.root, self.own_pin(old / "build_identity.json"))
            self.manifest(old, old_audit["artifacts"])
            self.manifest(self.root, old_phase["source_pins"])
            require(old_audit["status"] == "COMPLETE" and old_audit["complete_proxy_paths"] == 56
                and old_audit["failed_closed_paths"] == 8 and len(old_phase["events"]) == 9
                and canonical(old_phase) == old_audit["phase_identity_sha256"]
                and bt_phase["events"][:9] == old_phase["events"], "Completion did not preserve the original nine-action 56/8 reference")
            completion_cfg = read_json(self.root / "configs/independent_backtest_completion.json")
            require(canonical(completion_cfg) == bt_phase["configuration_sha256"]
                and completion_cfg["parent_configuration_sha256"] == canonical(cfg), "Completion configuration changed its frozen parent")
            for arm, basket, name in itertools.product(ARMS, ("class_1_10", "class_2_9"),
                    ("original_ledger", "selected", "rebalances")):
                relative = f"selections/{arm}/{basket}/{name}.parquet"
                require(digest(backtest / relative) == digest(old / relative), "Completion changed a prior selection byte")
        worker_counts["backtest_workers"] = self.workers(bt_phase)
        import exchange_calendars as xcals
        calendar = xcals.get_calendar("XNYS", start="2017-12-01", end="2019-12-31")
        sessions = calendar.sessions.strftime("%Y-%m-%d").tolist()
        sessions = [day for day in sessions if "2018-01-01" <= day <= "2019-12-31"]
        months = pd.period_range("2018-01", "2019-12", freq="M").astype(str).tolist()
        require(len(sessions) == 503 and canonical(sessions) == bt_phase["clock"]["sessions_sha256"], "Independent session clock changed")
        comparison = self.table(backtest / "comparison.parquet", ["arm", "basket", "variant", "cost_bps", "complete_proxy_path", "expected_sessions", "valid_close_sessions"])
        expected_grid = set(itertools.product(ARMS, ("class_1_10", "class_2_9"), ("unstopped", "asset_stop_10pct"), (0, 5, 10, 25)))
        require(len(comparison) == 64 and set(comparison[["arm", "basket", "variant", "cost_bps"]].itertuples(index=False, name=None)) == expected_grid, "Incomplete or duplicated scenario grid")
        paths = []
        for row in comparison.itertuples(index=False):
            directory = backtest / "scenarios" / row.arm / row.basket / row.variant / f"{row.cost_bps}bps"
            daily = self.table(directory / "daily.parquet", ["date", "mark", "status", "nav"])
            dates = pd.to_datetime(daily.date).dt.strftime("%Y-%m-%d")
            require(dates.between("2018-01-01", "2019-12-31").all(), "Daily ledger leaves authorized financial period")
            closes = daily.loc[daily.mark.isin(["close", "terminal_close"])].copy()
            require(pd.to_datetime(closes.date).dt.strftime("%Y-%m-%d").tolist() == sessions, "Scenario lost one of 503 explicit close sessions")
            monthly = self.table(directory / "monthly.parquet", ["outcome_month", "nav", "return"])
            require(monthly.outcome_month.tolist() == months and row.expected_sessions == 503, "Scenario lost one of 24 outcome months")
            if row.complete_proxy_path:
                require(row.valid_close_sessions == 503 and np.isfinite(closes.nav.to_numpy(dtype=float)).all()
                    and np.isfinite(monthly[["nav", "return"]].to_numpy(dtype=float)).all(), "A complete path contains missing valuations or returns")
            paths.append({"arm": row.arm, "basket": row.basket, "variant": row.variant, "cost_bps": row.cost_bps,
                "complete_proxy_path": bool(row.complete_proxy_path), "close_sessions": len(closes), "monthly_rows": len(monthly)})
        ret = self.table(backtest / "ret_readiness.parquet", ["signal_date", "status", "arm"])
        require(len(ret) == 96 and ret.status.eq("withheld_incomplete_120_month_class_grid").all()
            and set(ret.arm) == set(ARMS), "The strict RET gate was shortened or altered")
        # All original files are independently rehashed after the numerical audit.
        for item in list(self.pins.values()):
            self.pin(self.root, item, refresh=True)
        peak = self.budget()
        return {"schema_version": 1, "status": "PASS", "run_directory": str(self.run),
            "build_identity_sha256": pin, "root_audit_sha256": digest(self.run / "audit.json"),
            "model_audit_sha256": digest(models / "audit.json"),
            "backtest_audit_sha256": digest(backtest / "audit.json"),
            "backtest_folder": self.backtest_folder, "corporate_action_events": expected_events,
            "original_nine_event_backtest_preserved": self.backtest_folder == "backtest_completed",
            "verifier_sha256": digest(Path(__file__).resolve()),
            "source_preservation_entries_verified": len(preservation), "total_file_pins_verified": len(self.pins),
            "all_source_files_unchanged_after_validation": True, "workers": worker_counts,
            "annual_input_differences": years, "reused_training_preprocessing": matrices,
            "reused_models": arms, "scenario_sessions": paths,
            "scenarios": 64, "complete_proxy_paths": int(comparison.complete_proxy_path.sum()),
            "all_paths_retain_503_close_sessions_and_24_month_rows": True,
            "new_estimator_fits": 0, "new_preprocessing_fits": 0,
            "training_financial_values_decoded": False, "financial_2020_plus_decoded": False,
            "old_training_and_daily_sources_integrity_hashed_only": True,
            "models_deserialized": False, "return_performance_printed": False,
            "research_status": "DATA_LIMITED_RESTRICTED_UNIVERSE",
            "retrospective_DF_RDC_execution_restrictions_preserved": True,
            "knowledge_time_or_survivorship_certification": False,
            "verification_scope": "integrity_chronology_frozen_inference_and_complete_saved_session_ledgers_not_executable_market_accounting_certification",
            "decoded_validation_paths": sorted(self.decoded_paths),
            "peak_rss_bytes": peak, "elapsed_seconds": time.monotonic() - self.started}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--backtest-folder", choices=("backtest", "backtest_completed"), default="backtest")
    args = parser.parse_args(argv)
    receipt_path = args.receipt.resolve()
    require(receipt_path.is_relative_to(Path("/private/tmp")), "Verification receipts may be written only under /private/tmp")
    verifier = Verifier(args.root, args.run, backtest_folder=args.backtest_folder)
    result = verifier.verify()
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    require(not receipt_path.exists(), "Preserve an existing verification receipt")
    receipt_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "receipt": str(receipt_path),
        "scenarios": result["scenarios"], "complete_proxy_paths": result["complete_proxy_paths"],
        "preservation_entries": result["source_preservation_entries_verified"],
        "peak_rss_bytes": result["peak_rss_bytes"], "new_fits": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
