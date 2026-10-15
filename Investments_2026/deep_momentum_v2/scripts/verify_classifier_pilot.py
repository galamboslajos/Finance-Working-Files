"""Independent saved-pilot verification, with no production pilot imports/fits.

Financial decoding is limited to the fixed 2010--2019 pilot. All 2020+ source
artifacts may be hashed for preservation but are never decoded here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ARMS = ("rf_paper15", "xgb_paper15", "rf_extended", "xgb_extended")
KEYS = ["source_security_id", "signal_date"]
PROBS = [f"probability_{i}" for i in range(1, 11)]
EPS = 1e-15


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def digest(path):
    state = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            state.update(block)
    return state.hexdigest()


def canonical(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return "sha256:" + hashlib.sha256(raw.encode()).hexdigest()


def json_file(path):
    return json.loads(path.read_text())


def local(base, relative):
    path = (base / relative).resolve()
    require(path.is_relative_to(base.resolve()), f"Artifact leaves approved directory: {relative}")
    return path


def receipt_binding(run, audit):
    """Bind the verification result to the exact, unchanged completed audit."""
    audit_path = run / "audit.json"
    require(json_file(audit_path) == audit, "Root audit changed while independent verification ran")
    return {"run_directory": str(run.resolve()),
        "build_identity_sha256": audit["build_identity_sha256"],
        "root_audit_sha256": digest(audit_path),
        "verifier_sha256": digest(Path(__file__).resolve())}


def verify_artifacts(base, artifacts):
    count = 0
    if isinstance(artifacts, dict):
        records = [{"path": key, **value} for key, value in artifacts.items()]
    else:
        records = artifacts
    require(isinstance(records, list), "Artifact manifest must be a dict/list")
    for item in records:
        path = local(base, item["path"])
        require(path.is_file(), f"Missing artifact: {path}")
        require(path.stat().st_size == item["size_bytes"], f"Wrong artifact size: {path}")
        require(digest(path) == item["sha256"].removeprefix("sha256:"), f"Changed artifact: {path}")
        count += 1
    return count


def read_pilot_table(path, *, first, last):
    """Read safe date/key metadata first, then the bounded financial columns."""
    schema = pq.ParquetFile(path).schema_arrow.names
    require(set(KEYS + ["outcome_month", "outcome_start", "outcome_end"]) <= set(schema), f"Missing dated keys: {path}")
    metadata = pq.read_table(path, columns=KEYS + ["outcome_month", "outcome_start", "outcome_end"]).to_pandas()
    require(not metadata.duplicated(KEYS).any(), f"Duplicate original keys: {path}")
    require(metadata.outcome_month.between(first, last).all(), f"Out-of-pilot financial rows: {path}")
    for sid, signal, month, start, end in metadata.itertuples(index=False, name=None):
        require(isinstance(sid, str) and sid.strip(), "Invalid source ID")
        require(str(pd.Period(signal[:7], freq="M") + 1) == month, "Wrong prediction month")
        require(signal < start <= end and start[:7] == month == end[:7], "Invalid outcome endpoints")
    return pq.read_table(path).to_pandas()


def independent_scores(frame):
    """Direct one-hot scores and equal-month aggregation, independent of workers."""
    values = frame[PROBS].to_numpy(dtype=float)
    require(np.isfinite(values).all(), "Prediction probabilities must be finite")
    require((values >= 0).all(), "Negative probability")
    require(np.all(np.abs(values.sum(axis=1) - 1) <= 1e-8), "Probability rows do not sum to one")
    y = frame.target_class.to_numpy(dtype=float, na_value=np.nan)
    observed = np.isfinite(y)
    require(np.isin(y[observed], np.arange(1, 11)).all(), "Bad canonical class")
    if "target_available" in frame:
        require(np.array_equal(observed, frame.target_available.to_numpy(dtype=bool)), "Target availability changed")
    hard = np.argmax(values, axis=1) + 1
    scored = frame[KEYS + ["outcome_month"]].copy()
    scored["log_loss"] = np.nan
    scored["brier_sum10"] = np.nan
    scored["accuracy"] = np.nan
    scored["observed"] = observed
    chosen = y[observed].astype(int) - 1
    true_probability = values[observed, chosen]
    onehot = np.eye(10)[chosen]
    scored.loc[observed, "log_loss"] = -np.log(np.maximum(true_probability, EPS))
    scored.loc[observed, "brier_sum10"] = np.square(values[observed] - onehot).sum(axis=1)
    scored.loc[observed, "accuracy"] = (hard[observed] == chosen + 1).astype(float)
    monthly = scored.groupby("outcome_month", sort=True).agg(
        prediction_rows=("observed", "size"), labelled_rows=("observed", "sum"),
        log_loss=("log_loss", "mean"), brier_sum10=("brier_sum10", "mean"), accuracy=("accuracy", "mean"))
    summary = {"prediction_rows": len(frame), "labelled_rows": int(observed.sum()),
        "missing_label_rows": int((~observed).sum()), "outcome_months": len(monthly),
        "mean_monthly_log_loss": float(monthly.log_loss.mean()),
        "mean_monthly_brier_sum10": float(monthly.brier_sum10.mean()),
        "mean_monthly_accuracy": float(monthly.accuracy.mean()),
        "pooled_log_loss": float(scored.log_loss.mean()),
        "pooled_brier_sum10": float(scored.brier_sum10.mean()),
        "pooled_accuracy": float(scored.accuracy.mean()),
        "zero_true_class_probability_rows": int(np.sum(true_probability == 0)),
        "floored_true_class_probability_rows": int(np.sum(true_probability < EPS))}
    return summary, monthly, scored


def equal_scalar(actual, expected, name):
    if expected is None or (isinstance(expected, float) and np.isnan(expected)):
        require(actual is None or pd.isna(actual), f"Unexpected value {name}")
    elif isinstance(expected, (int, np.integer)):
        require(actual == expected, f"Different count {name}: {actual} != {expected}")
    else:
        require(np.isclose(actual, expected, rtol=1e-12, atol=2e-14), f"Different score {name}: {actual} != {expected}")


def verify_metrics(directory, frame, state):
    summary, monthly, scored = independent_scores(frame)
    persisted = state["summary"]
    for name, value in summary.items():
        equal_scalar(persisted[name], value, name)
    saved_monthly = pq.read_table(directory / "reports/monthly.parquet").to_pandas().set_index("outcome_month")
    require(list(saved_monthly.index) == list(monthly.index), "Monthly report dates changed")
    for name in ("prediction_rows", "labelled_rows", "log_loss", "brier_sum10", "accuracy"):
        require(np.allclose(saved_monthly[name].astype(float), monthly[name].astype(float), rtol=1e-12, atol=2e-14, equal_nan=True), f"Monthly {name} differs")
    saved_rows = pq.read_table(directory / "reports/row_scores.parquet").to_pandas().set_index(KEYS)
    expected_rows = scored.set_index(KEYS)
    require(set(saved_rows.index) == set(expected_rows.index), "Row-score keys differ")
    saved_rows = saved_rows.reindex(expected_rows.index)
    for expected, actual in (("log_loss", "log_loss"), ("brier_sum10", "brier_sum10"), ("accuracy", "correct")):
        require(np.allclose(saved_rows[actual].astype(float), expected_rows[expected].astype(float), rtol=1e-12, atol=2e-14, equal_nan=True), f"Row {expected} differs")
    probability = frame[PROBS].to_numpy(dtype=float)
    y = frame.target_class.to_numpy(dtype=float, na_value=np.nan)
    labelled = np.isfinite(y)
    hard = np.argmax(probability, axis=1) + 1
    ece_parts = {}
    for kind in ("toplabel", "classwise"):
        table = pq.read_table(directory / f"reports/{kind}_calibration.parquet").to_pandas()
        require(len(table) == (10 if kind == "toplabel" else 100), "Calibration bins changed")
        weighted_gaps = 0.0
        for class_id in ([None] if kind == "toplabel" else range(1, 11)):
            confidence = probability[labelled].max(axis=1) if class_id is None else probability[labelled, class_id - 1]
            hit = hard[labelled] == y[labelled] if class_id is None else y[labelled] == class_id
            bins = np.searchsorted(np.arange(1, 10) / 10, confidence, side="right")
            for bucket in range(10):
                row = table.loc[table.bin_id == bucket]
                if class_id is not None:
                    row = row.loc[row.class_id == class_id]
                require(len(row) == 1, "Calibration cell duplicated or absent")
                row = row.iloc[0]
                selected = bins == bucket
                count = int(selected.sum())
                equal_scalar(row.labelled_count, count, "calibration labelled_count")
                mean = float(confidence[selected].mean()) if count else None
                frequency = float(hit[selected].mean()) if count else None
                gap = abs(mean - frequency) if count else None
                equal_scalar(row.mean_probability, mean, "calibration mean_probability")
                equal_scalar(row.observed_frequency, frequency, "calibration observed_frequency")
                equal_scalar(row.absolute_gap, gap, "calibration absolute_gap")
                weighted_gaps += count * (gap or 0)
        ece_parts[kind] = weighted_gaps / labelled.sum() / (10 if kind == "classwise" else 1)
    equal_scalar(persisted["pooled_top_label_ece"], ece_parts["toplabel"], "top-label ECE")
    equal_scalar(persisted["pooled_classwise_ece_mean"], ece_parts["classwise"], "classwise ECE")
    tail = pq.read_table(directory / "reports/extreme_precision.parquet").to_pandas()
    require(len(tail) == 2 * (1 + frame.outcome_month.nunique()), "Tail-precision calendar changed")
    for period in ["all", *sorted(frame.outcome_month.unique())]:
        period_mask = np.ones(len(frame), dtype=bool) if period == "all" else (frame.outcome_month == period).to_numpy()
        for class_id in (1, 10):
            row = tail.loc[(tail.period == period) & (tail.class_id == class_id)]
            require(len(row) == 1, "Tail-precision cell duplicated or absent")
            row = row.iloc[0]
            selected = period_mask & (hard == class_id)
            observed = selected & labelled
            support, observed_count = int(selected.sum()), int(observed.sum())
            hits = int((observed & (y == class_id)).sum())
            for name, value in (("prediction_rows", int(period_mask.sum())), ("hard_mode_prediction_count", support), ("labelled_hard_mode_count", observed_count), ("missing_label_hard_mode_count", support - observed_count), ("true_positive_count", hits)):
                equal_scalar(row[name], value, f"tail {name}")
            equal_scalar(row.precision, hits / observed_count if observed_count else None, "tail precision")
    return summary


def verify_preprocessor(path, ledger):
    state = json_file(path)
    manifest = state["manifest"]
    require(canonical(manifest) == state["fingerprint"], "Preprocessor state hash mismatch")
    for key, expected in (("train_start", "2010-01-01"), ("train_end", "2017-12-31")):
        require(manifest[key] == expected, f"Preprocessor fit cutoff differs: {key}")
    selected = ledger.loc[ledger.fit_mask]
    require("2017-12-01" <= manifest["label_observed_through"] <= "2017-12-31", "Preprocessor label observation exceeds the calendar fit boundary")
    require((selected.outcome_end <= manifest["label_observed_through"]).all(), "A selected training outcome was uncompleted at the recorded observation cutoff")
    require(manifest["fit_rows"] == len(selected), "Preprocessor fit row count changed")
    require(manifest["fit_keys_sha256"] == canonical(selected[KEYS].values.tolist()), "Preprocessor fit keys changed")
    require(not set(manifest["retained_columns"]) & set(manifest["dropped_columns"]), "Preprocessor retained/dropped overlap")
    require(set(manifest["input_columns"]) == set(manifest["retained_columns"]) | set(manifest["dropped_columns"]), "Input schema loses columns")
    expected = [value for name in manifest["retained_columns"] for value in (name, name + "__missing")]
    require(manifest["output_columns"] == expected, "Missing indicator schema changed")
    require(manifest["scaling"] is None and manifest["clipping"] is None and manifest["winsorization"] is None, "Unexpected normalization")
    return {"fingerprint": state["fingerprint"], "fit_rows": len(selected), "input_columns": len(manifest["input_columns"]), "retained_columns": len(manifest["retained_columns"]), "matrix_width": len(expected)}


def verify_source_ledgers(root, target_run, train, valid):
    """Reconcile saved pilot rows with immutable annual target/cohort sources."""
    roles = json_file(root / "configs/context_panel_roles.json")
    feature_run = local(root, roles["primary"]["directory"])
    require(digest(feature_run / "audit.json") == roles["primary"]["audit_sha256"].removeprefix("sha256:"), "Primary feature root audit changed")
    target_audit = json_file(target_run / "audit.json")
    require(target_audit["status"] == "COMPLETE", "Source target run incomplete")
    source_parts = []
    for year in range(2010, 2020):
        year_pin = target_audit["years"][str(year)]
        require(digest(local(target_run, year_pin["path"])) == year_pin["sha256"].removeprefix("sha256:"), f"Source target year audit changed: {year}")
        directory = target_run / "years" / str(year)
        annual = json_file(directory / "audit.json")
        target_entry = annual["artifacts"]["targets"]
        target_path = local(directory, target_entry["path"])
        require(digest(target_path) == target_entry["sha256"].removeprefix("sha256:"), f"Source annual target changed: {year}")
        target = read_pilot_table(target_path, first=f"{year}-01", last=f"{year}-12")
        cohort = pq.read_table(feature_run / "years" / str(year) / "feature_coverage.parquet", columns=KEYS + ["identity_eligible", "requested", "paper_prediction_eligible"]).to_pandas().set_index(KEYS)
        target = target.set_index(KEYS)
        require(set(cohort.index) == set(target.index), f"Source target/cohort keys differ: {year}")
        for name in ("identity_eligible", "requested", "paper_prediction_eligible"):
            require(cohort[name].equals(target.reindex(cohort.index)[name]), f"Source admission differs: {year}/{name}")
        source_parts.append(target)
    source = pd.concat(source_parts)
    actual = pd.concat([train, valid]).set_index(KEYS)
    require(set(actual.index) == set(source.index), "Pilot changed original source cohorts")
    expected = source.reindex(actual.index)
    columns = ("outcome_month", "outcome_start", "outcome_end", "identity_eligible", "requested", "paper_prediction_eligible", "prediction_eligible", "target_available", "target_class")
    for name in columns:
        left = actual[name].astype(object).where(actual[name].notna(), None)
        right = expected[name].astype(object).where(expected[name].notna(), None)
        require(left.equals(right), f"Pilot ledger differs from original source {name}")
    return feature_run


def verify_training_statistics(run, feature_run, train):
    """Recompute all train medians/exclusions from 2010--2017 values only."""
    manifests = {name: json_file(run / "preprocessors" / name / "manifest.json")["manifest"] for name in ("paper15", "expanded")}
    dictionary = json_file(feature_run / "feature_dictionary.json")
    require(isinstance(dictionary, list), "Feature dictionary shape changed")
    registered = [item["name"] for item in dictionary if item.get("model_eligible", True)]
    paper15 = [f"{prefix}{h}" for prefix in ("zMOM", "meanMOM", "stdMOM") for h in (1, 3, 6, 9, 12)]
    require(manifests["expanded"]["input_columns"] == registered and len(registered) == 715, "Expanded candidate allowlist changed")
    require(manifests["paper15"]["input_columns"] == paper15, "Paper15 candidate allowlist changed")
    columns = manifests["expanded"]["input_columns"]
    fit = train.loc[train.fit_mask].set_index(KEYS)
    parts = []
    for year in range(2010, 2018):
        path = feature_run / "years" / str(year) / "features.parquet"
        metadata = pq.read_table(path, columns=KEYS).to_pandas()
        require(metadata.signal_date.between(f"{year-1}-12-01", f"{year}-11-30").all(), "Out-of-training feature values would be decoded")
        require(not metadata.duplicated(KEYS).any(), "Duplicate source feature keys")
        frame = pq.read_table(path, columns=KEYS + columns).to_pandas().set_index(KEYS)
        selected = frame.index.isin(fit.index)
        parts.append(frame.loc[selected])
    features = pd.concat(parts).reindex(fit.index).replace([np.inf, -np.inf], np.nan)
    require(len(features) == len(fit) and features.index.is_unique, "Independent training matrix lost rows")
    checked = 0
    for name, manifest in manifests.items():
        matrix = np.load(run / "preprocessors" / name / "train_X.npy", mmap_mode="r", allow_pickle=False)
        require(matrix.dtype == np.float32 and matrix.shape == (len(fit), len(manifest["output_columns"])), "Prepared training matrix shape/dtype differs")
        for column in manifest["input_columns"]:
            values = features[column].dropna().to_numpy(dtype=float)
            if not len(values):
                require(manifest["dropped_columns"].get(column) == "all_missing_in_training_partition", f"All-missing exclusion differs: {name}/{column}")
            elif np.all(values == values[0]):
                require(manifest["dropped_columns"].get(column) == "constant_finite_values_in_training_partition", f"Constant exclusion differs: {name}/{column}")
            else:
                require(column in manifest["retained_columns"], f"Training-observed varying feature excluded: {name}/{column}")
                # np.median uses a separate partition implementation from the
                # production worker's sorted central-pair median computation.
                equal_scalar(manifest["medians"][column], float(np.median(values)), f"training median {name}/{column}")
            checked += 1
        for offset, column in enumerate(manifest["retained_columns"]):
            values = features[column].to_numpy(dtype=float)
            missing = ~np.isfinite(values)
            expected = np.where(missing, manifest["medians"][column], values).astype(np.float32)
            require(np.array_equal(matrix[:, 2 * offset], expected), f"Prepared training feature differs: {name}/{column}")
            require(np.array_equal(matrix[:, 2 * offset + 1], missing.astype(np.float32)), f"Training missing indicator differs: {name}/{column}")
    return {"independently_checked_input_columns": checked, "financial_feature_years": [2010, 2017], "fit_rows": len(fit)}


def verify_validation_matrices(run, feature_run, valid):
    manifests = {name: json_file(run / "preprocessors" / name / "manifest.json")["manifest"] for name in ("paper15", "expanded")}
    selected_keys = pd.MultiIndex.from_frame(valid.loc[valid.prediction_eligible, KEYS])
    matrices = {name: np.load(run / "preprocessors" / name / "validation_X.npy", mmap_mode="r", allow_pickle=False) for name in manifests}
    for name, matrix in matrices.items():
        require(matrix.dtype == np.float32 and matrix.shape == (len(selected_keys), len(manifests[name]["output_columns"])), "Prepared validation matrix shape/dtype changed")
    checked_rows = 0
    for year in (2018, 2019):
        path = feature_run / "years" / str(year) / "features.parquet"
        metadata = pq.read_table(path, columns=KEYS).to_pandas()
        require(metadata.signal_date.between(f"{year-1}-12-01", f"{year}-11-30").all(), "Out-of-validation feature values would be decoded")
        frame = pq.read_table(path, columns=KEYS + manifests["expanded"]["input_columns"]).to_pandas().set_index(KEYS)
        frame = frame.loc[frame.index.isin(selected_keys)]
        positions = selected_keys.get_indexer(frame.index)
        require((positions >= 0).all(), "Validation matrix row is outside the prediction cohort")
        for name, manifest in manifests.items():
            for offset, column in enumerate(manifest["retained_columns"]):
                values = frame[column].to_numpy(dtype=float)
                missing = ~np.isfinite(values)
                expected = np.where(missing, manifest["medians"][column], values).astype(np.float32)
                require(np.array_equal(matrices[name][positions, 2 * offset], expected), f"Prepared validation feature differs: {name}/{column}")
                require(np.array_equal(matrices[name][positions, 2 * offset + 1], missing.astype(np.float32)), f"Validation missing indicator differs: {name}/{column}")
        checked_rows += len(frame)
    require(checked_rows == len(selected_keys), "Validation matrix source rows lost")
    return checked_rows


def verify_saved_model(directory, kind, prepared, expected):
    """Reload hash-verified local models and predict; this never invokes fit."""
    from threadpoolctl import threadpool_limits
    if kind == "rf":
        import joblib
        estimator = joblib.load(directory / "model.joblib")
        require(list(estimator.classes_) == list(range(1, 11)), "Reloaded forest classes changed")
        with threadpool_limits(limits=1):
            actual = estimator.predict_proba(prepared)
    else:
        from scipy.special import softmax
        from xgboost import XGBClassifier
        estimator = XGBClassifier(n_jobs=1, device="cpu")
        estimator.load_model(directory / "model.ubj")
        estimator.set_params(n_jobs=1, device="cpu")
        require(list(estimator.classes_) == list(range(10)), "Reloaded XGBoost classes changed")
        with threadpool_limits(limits=1):
            margins = estimator.predict(prepared, output_margin=True)
        actual = softmax(np.asarray(margins, dtype=np.float64), axis=1)
    difference = float(np.max(np.abs(actual - expected)))
    require(np.allclose(actual, expected, rtol=1e-12, atol=2e-14), "Reloaded model probabilities differ from saved predictions")
    return difference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-run", type=Path, required=True, help="Exact immutable target source directory used by the pilot")
    args = parser.parse_args()
    root, run = args.root.resolve(), args.run.resolve()
    require(run.is_relative_to(root / "outputs/model_pilot"), "Run is outside the pilot output namespace")
    require(args.target_run.resolve().is_relative_to(root / "outputs/full_target_panel"), "Target source is outside its immutable namespace")
    audit = json_file(run / "audit.json")
    require(audit["status"] == "COMPLETE", "Pilot is incomplete")
    identity = json_file(run / "build_identity.json")
    require(canonical(identity) == audit["build_identity_sha256"], "Build identity hash differs")
    artifact_count = verify_artifacts(run, audit["artifacts"])
    artifact_count += verify_artifacts(run, audit["arms"])
    source_manifest = json_file(run / "source_preservation_manifest.json")
    source_count = verify_artifacts(root, source_manifest["artifacts"])
    train = read_pilot_table(run / "data/train_ledger.parquet", first="2010-01", last="2017-12")
    valid = read_pilot_table(run / "data/validation_ledger.parquet", first="2018-01", last="2019-12")
    require(train.outcome_month.nunique() == 96 and valid.outcome_month.nunique() == 24, "Incomplete pilot calendar")
    require((train.outcome_end <= "2017-12-31").all(), "Training labels enter validation period")
    require((train.signal_date < "2017-12-01").all(), "Training feature dates changed")
    require((valid.signal_date >= "2017-12-01").all() and (valid.signal_date < "2019-12-01").all(), "Validation signal dates changed")
    feature_run = verify_source_ledgers(root, args.target_run.resolve(), train, valid)
    expected_fit = train.prediction_eligible & train.target_available
    require(np.array_equal(train.fit_mask, expected_fit), "Fit mask depends on added-feature missingness or rejects valid labels")
    require(not train.loc[train.fit_mask, "target_class"].isna().any(), "Training labels imputed/missing")
    counts = train.loc[train.fit_mask, "target_class"].astype(int).value_counts().reindex(range(1, 11), fill_value=0)
    require((counts > 0).all(), "Training canonical class absent")
    prior = counts.to_numpy(dtype=float) / counts.sum()
    preprocessors = {name: verify_preprocessor(run / "preprocessors" / name / "manifest.json", train) for name in ("paper15", "expanded")}
    training_statistics = verify_training_statistics(run, feature_run, train)
    independently_checked_validation_rows = verify_validation_matrices(run, feature_run, valid)
    summaries = {}
    model_reload_max_differences = {}
    for arm in (*ARMS, "uniform", "train_prior"):
        directory = run / ("arms" if arm in ARMS else "refs") / arm
        state = json_file(directory / "audit.json")
        artifact_count += verify_artifacts(directory, state["artifacts"])
        predictions = read_pilot_table(directory / "predictions.parquet", first="2018-01", last="2019-12")
        require(list(map(tuple, predictions[KEYS].to_numpy())) == list(map(tuple, valid[KEYS].to_numpy())), f"Original validation row order/keys changed: {arm}")
        for column in ("outcome_month", "outcome_start", "outcome_end", "prediction_eligible", "target_available", "target_class"):
            left = predictions[column].astype(object).where(predictions[column].notna(), None)
            right = valid[column].astype(object).where(valid[column].notna(), None)
            require(left.equals(right), f"Validation {column} changed: {arm}")
        prediction_mask = predictions.prediction_eligible.to_numpy(dtype=bool)
        require(predictions.loc[~prediction_mask, PROBS].isna().all().all(), f"Ineligible rows received probabilities: {arm}")
        eligible = predictions.loc[prediction_mask].reset_index(drop=True)
        require(eligible[PROBS].notna().all().all(), f"Missing eligible predictions: {arm}")
        if arm == "uniform":
            require(np.array_equal(eligible[PROBS].to_numpy(), np.full((len(eligible), 10), .1)), "Uniform reference changed")
        if arm == "train_prior":
            require(np.allclose(eligible[PROBS].to_numpy(), prior[None, :], rtol=0, atol=1e-15), "Prior did not come solely from pooled supervised training classes")
        summaries[arm] = verify_metrics(directory, eligible, state)
        if arm in ARMS:
            metadata = state["metadata"]
            require(metadata["fit_count"] == 1, f"More than one fit: {arm}")
            for key in ("validation_labels_supplied_to_estimator", "early_stopping_used", "parameter_search_performed", "class_weighting_applied", "probability_calibration_performed"):
                require(metadata[key] is False, f"Unapproved model operation {key}: {arm}")
            require(metadata["train_rows"] == int(expected_fit.sum()), f"Estimator fit rows differ: {arm}")
            require(metadata["prediction_rows"] == int(valid.prediction_eligible.sum()), f"Estimator prediction rows differ: {arm}")
            require(metadata["probability_class_ids"] == list(range(1, 11)), f"Probability class order changed: {arm}")
            mappings = metadata["native_to_canonical_mapping"]
            offset = 1 if arm.startswith("xgb") else 0
            require(sorted((r["native_class_id"], r["canonical_class_id"]) for r in mappings) == [(i - offset, i) for i in range(1, 11)], f"Native class mapping changed: {arm}")
            require(metadata["train_class_counts"] == {str(i): int(counts[i]) for i in range(1, 11)}, f"Model class counts changed: {arm}")
            feature_set = "paper15" if arm.endswith("paper15") else "expanded"
            kind = "xgb" if arm.startswith("xgb") else "rf"
            require(metadata["explicit_parameters"] == identity["configuration"]["estimators"][kind], f"Model parameters differ from frozen configuration: {arm}")
            require(metadata["preprocessor_fingerprint"] == preprocessors[feature_set]["fingerprint"], f"Model uses different preprocessing state: {arm}")
            require(metadata["fit_keys_sha256"] == canonical(train.loc[expected_fit, KEYS].values.tolist()), f"Model fit-key hash changed: {arm}")
            require(metadata["prediction_keys_sha256"] == canonical(valid.loc[valid.prediction_eligible, KEYS].values.tolist()), f"Model prediction-key hash changed: {arm}")
            for filename, field in (("train_X.npy", "prepared_train_matrix_sha256"), ("validation_X.npy", "prepared_prediction_matrix_sha256")):
                matrix = np.load(run / "preprocessors" / feature_set / filename, mmap_mode="r", allow_pickle=False)
                require(metadata[field] == "sha256:" + hashlib.sha256(memoryview(matrix).cast("B")).hexdigest(), f"Estimator consumed different prepared matrix: {arm}/{filename}")
            model_reload_max_differences[arm] = verify_saved_model(directory, kind, matrix, eligible[PROBS].to_numpy(dtype=float))
    result = {**receipt_binding(run, audit), "status": "PASSED", "independent_verifier": "no_production_pilot_imports_no_fits", "verified_artifacts": artifact_count,
        "preserved_source_files": source_count, "training_original_rows": len(train), "supervised_fit_rows": int(expected_fit.sum()),
        "validation_original_rows": len(valid), "validation_prediction_rows": int(valid.prediction_eligible.sum()),
        "validation_labelled_rows": int((valid.prediction_eligible & valid.target_available).sum()),
        "train_class_counts": {str(i): int(counts[i]) for i in range(1, 11)}, "preprocessors": preprocessors,
        "training_statistics": training_statistics,
        "independently_checked_validation_matrix_rows": independently_checked_validation_rows,
        "model_reload_max_probability_difference": model_reload_max_differences,
        "summaries": summaries, "maximum_financial_outcome_decoded": "2019-12-31", "models_fitted_by_verifier": 0}
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "verified_artifacts", "preserved_source_files", "supervised_fit_rows", "validation_prediction_rows")}, sort_keys=True))


if __name__ == "__main__":
    main()
