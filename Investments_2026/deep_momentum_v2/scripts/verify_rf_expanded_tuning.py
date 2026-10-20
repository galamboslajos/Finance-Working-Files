"""Independent zero-fit verification of saved RF-expanded tuning checkpoints.

No production tuning module is imported.  Numerical checks use saved prepared
matrices, labels and probability ledgers.  Original financial source panels are
not decoded, and no estimator/preprocessor is fitted.  Matrix-level median
verification is explicitly limited by their saved float32 representation.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import resource
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

CLASS_IDS = tuple(range(1, 11))
KEYS = ("source_security_id", "signal_date")
PROBS = tuple(f"probability_{k}" for k in CLASS_IDS)
FLOOR = 1e-15
TIE_TOLERANCE = 1e-8


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def canonical(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return "sha256:" + hashlib.sha256(raw.encode()).hexdigest()


def digest(path):
    state = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            state.update(block)
    return state.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def verify_pin(pin, *, roots):
    require(isinstance(pin, dict), "Malformed artifact pin")
    declared = Path(pin["path"])
    path = declared.resolve()
    require(declared.is_absolute() and any(path.is_relative_to(Path(root).resolve()) for root in roots),
            f"Pinned path leaves approved roots: {path}")
    require(path.is_file(), f"Missing pinned file: {path}")
    require(type(pin["size_bytes"]) is int and path.stat().st_size == pin["size_bytes"],
            f"Pinned file size changed: {path}")
    require(digest(path) == pin["sha256"].removeprefix("sha256:"), f"Pinned file bytes changed: {path}")
    return path


def keys_hash(frame, mask=None):
    chosen = frame if mask is None else frame.loc[mask]
    return canonical(chosen.loc[:, list(KEYS)].values.tolist())


def boolean_column(frame, name):
    require(name in frame, f"Ledger lacks {name}")
    series = frame[name]
    require(not series.isna().any() and series.map(lambda x: isinstance(x, (bool, np.bool_))).all(),
            f"Ledger {name} must contain nonmissing boolean values")
    return series.to_numpy(dtype=bool)


def validate_dates(frame, *, first_year, last_year):
    """Inspect dates/keys before touching labels or probability values."""
    columns = (*KEYS, "outcome_month", "outcome_start", "outcome_end")
    require(set(columns) <= set(frame), "Ledger lacks dated original keys")
    require(not frame.duplicated(list(KEYS)).any(), "Duplicate original cohort keys")
    require(2010 <= first_year <= last_year <= 2023, "Financial bounds leave 2010–2023")
    for sid, signal, month, start, end in frame.loc[:, list(columns)].itertuples(index=False, name=None):
        require(isinstance(sid, str) and sid.strip(), "Invalid source security ID")
        for value in (signal, start, end):
            require(isinstance(value, str) and dt.date.fromisoformat(value).isoformat() == value,
                    "Ledger date is not an ISO date")
        require(isinstance(month, str) and len(month) == 7 and month + "-01" == dt.date.fromisoformat(month + "-01").isoformat(),
                "Invalid outcome month")
        require(f"{first_year}-01" <= month <= f"{last_year}-12", "Ledger leaves frozen financial year bounds")
        require(str(pd.Period(signal[:7], freq="M") + 1) == month, "Outcome is not month following signal")
        require(signal < start <= end and start[:7] == month == end[:7], "Invalid target interval")
    return True


def validate_chronology(train, validation, fit_mask, *, train_start_year, train_end_year, validation_year):
    validate_dates(train, first_year=train_start_year, last_year=train_end_year)
    validate_dates(validation, first_year=validation_year, last_year=validation_year)
    require(train_end_year == validation_year - 1, "Chronological partition has a gap or overlap")
    selected = train.loc[fit_mask]
    require(len(selected) > 0 and len(validation) > 0, "Empty train/validation partition")
    require(selected.outcome_end.max() <= validation.signal_date.min(), "Training target extends beyond validation signal")
    require(selected.outcome_end.max() < validation.outcome_start.min(), "Training and validation target intervals overlap")
    expected = [f"{validation_year}-{month:02d}" for month in range(1, 13)]
    require(sorted(validation.outcome_month.unique()) == expected, "Validation lacks a complete 12-month calendar")
    require(not set(map(tuple, selected.loc[:, list(KEYS)].values)) & set(map(tuple, validation.loc[:, list(KEYS)].values)),
            "Training and validation keys overlap")
    return {"maximum_train_outcome_end": selected.outcome_end.max(),
            "minimum_validation_signal_date": validation.signal_date.min(),
            "minimum_validation_outcome_start": validation.outcome_start.min()}


def independent_scores(probabilities, labels, metadata, *, expected_year):
    """Recompute scores directly; never clip/repair saved probabilities."""
    validate_dates(metadata, first_year=expected_year, last_year=expected_year)
    values = np.asarray(probabilities)
    require(values.dtype == np.float64 and values.shape == (len(metadata), 10), "Wrong native probability matrix shape/dtype")
    require(np.isfinite(values).all() and (values >= 0).all() and (values <= 1).all(), "Invalid probability value")
    require(np.all(np.abs(values.sum(axis=1) - 1) <= 1e-8), "Probability row does not sum to one")
    original_labels = np.asarray(labels, dtype=object)
    require(not any(isinstance(value, (bool, np.bool_, str)) for value in original_labels), "Noncanonical label type")
    y = np.asarray(labels, dtype=float)
    require(y.shape == (len(metadata),), "Label vector is not aligned")
    observed = np.isfinite(y)
    require(np.isin(y[observed], CLASS_IDS).all(), "Noncanonical observed label")
    hard = np.argmax(values, axis=1) + 1
    losses = np.full(len(y), np.nan)
    brier = np.full(len(y), np.nan)
    accuracy = np.full(len(y), np.nan)
    selected = y[observed].astype(int) - 1
    true = values[observed, selected]
    losses[observed] = -np.log(np.maximum(true, FLOOR))
    brier[observed] = np.square(values[observed] - np.eye(10)[selected]).sum(axis=1)
    accuracy[observed] = (hard[observed] == selected + 1).astype(float)
    months = metadata.outcome_month.to_numpy()
    expected = [f"{expected_year}-{month:02d}" for month in range(1, 13)]
    require(sorted(set(months)) == expected, "Prediction rows lack a complete 12-month calendar")
    monthly = []
    for month in expected:
        mask = months == month
        scored = mask & observed
        count = int(scored.sum())
        require(count > 0, f"Validation month has no observed label: {month}")
        monthly.append({"outcome_month": month, "prediction_rows": int(mask.sum()), "labelled_rows": count,
                        "log_loss": float(losses[scored].mean()), "brier_sum10": float(brier[scored].mean()),
                        "accuracy": float(accuracy[scored].mean()), "score_keys_sha256": keys_hash(metadata, scored)})
    summary = {"prediction_rows": len(y), "labelled_rows": int(observed.sum()),
               "missing_label_rows": int((~observed).sum()), "outcome_months": 12,
               "mean_monthly_log_loss": float(np.mean([row["log_loss"] for row in monthly])),
               "mean_monthly_brier_sum10": float(np.mean([row["brier_sum10"] for row in monthly])),
               "mean_monthly_accuracy": float(np.mean([row["accuracy"] for row in monthly])),
               "pooled_log_loss": float(losses[observed].mean()),
               "pooled_brier_sum10": float(brier[observed].mean()), "pooled_accuracy": float(accuracy[observed].mean()),
               "zero_true_class_probability_rows": int((true == 0).sum()),
               "floored_true_class_probability_rows": int((true < FLOOR).sum())}
    return {"summary": summary, "monthly": monthly, "observed_mask": observed,
            "predicted_class": hard, "row_log_loss": losses, "row_brier_sum10": brier, "row_accuracy": accuracy}


def nominal_features(candidate, width):
    value = candidate["max_features"]
    return max(1, int(math.sqrt(width))) if value == "sqrt" else max(1, int(float(value) * width))


def select_winner(candidates, monthly_by_candidate, widths):
    """Independent complete-grid 36-month selection and numerical tie rule."""
    ids = [candidate["candidate_id"] for candidate in candidates]
    require(len(ids) == len(set(ids)) == 8 and set(monthly_by_candidate) == set(ids), "Incomplete or duplicated candidate grid")
    require(len(widths) == 3 and all(type(width) is int and width > 0 for width in widths), "Three inner prepared widths required")
    scores, months_and_masks = {}, None
    for candidate_id in ids:
        rows = monthly_by_candidate[candidate_id]
        require(len(rows) == 36, "Candidate does not have 36 validation months")
        calendar = [(row["outcome_month"], row["score_keys_sha256"]) for row in rows]
        require(len(set(month for month, _ in calendar)) == 36 and calendar == sorted(calendar), "Duplicate/unsorted validation calendar")
        require(months_and_masks is None or calendar == months_and_masks, "Candidates use different scoring keys/months")
        months_and_masks = calendar
        values = [row["log_loss"] for row in rows]
        require(all(isinstance(value, (int, float)) and math.isfinite(value) and value >= 0 for value in values), "Incomplete candidate score")
        scores[candidate_id] = float(np.mean(values))
    minimum = min(scores.values())
    ties = [candidate for candidate in candidates if scores[candidate["candidate_id"]] <= minimum + TIE_TOLERANCE]
    references = [candidate for candidate in ties if candidate.get("reference_statistical_parameters") is True]
    require(sum(candidate.get("reference_statistical_parameters") is True for candidate in candidates) == 1, "Grid must identify exactly one reference")
    ordered = sorted(ties, key=lambda candidate: (candidate["max_depth"], -candidate["min_samples_leaf"],
                     sum(nominal_features(candidate, width) for width in widths), candidate["candidate_id"]))
    chosen = references[0] if references else ordered[0]
    return {"candidate_id": chosen["candidate_id"], "scores": scores,
            "tie_candidate_ids": sorted(candidate["candidate_id"] for candidate in ties),
            "validation_months": [month for month, _ in months_and_masks]}


def verify_prepared_matrix(matrix, preprocessing):
    """Independently check train-only missing flags, imputations and medians.

    These checks reconstruct retained medians from finite original training
    observations represented by flag=0.  Comparison accounts for the stored
    float32 quantization; exact raw-float64 medians and dropped source columns
    are not claimed to be independently reconstructed.
    """
    require(matrix.dtype == np.float32 and matrix.ndim == 2 and np.isfinite(matrix).all(), "Prepared matrix must be finite float32")
    retained = preprocessing["retained_columns"]
    expected_columns = [column for name in retained for column in (name, name + "__missing")]
    require(preprocessing["output_columns"] == expected_columns and matrix.shape[1] == len(expected_columns), "Prepared schema columns differ")
    require(len(preprocessing["input_columns"]) == 715 and len(set(preprocessing["input_columns"])) == 715, "Fixed 715-input bank changed")
    require(set(retained) | set(preprocessing["dropped_columns"]) == set(preprocessing["input_columns"]), "Dropped/retained input coverage changed")
    require(not set(retained) & set(preprocessing["dropped_columns"]), "Dropped/retained input overlap")
    require(matrix.shape[0] == preprocessing["fit_rows"], "Prepared training row count differs")
    largest_error = 0.0
    for index, name in enumerate(retained):
        values, flags = matrix[:, 2 * index], matrix[:, 2 * index + 1]
        require(np.isin(flags, (0, 1)).all(), f"Nonbinary missing indicator: {name}")
        finite = values[flags == 0]
        require(len(finite) > 0, f"Retained column is entirely missing: {name}")
        saved = np.float32(preprocessing["medians"][name])
        require(np.isfinite(saved), f"Nonfinite imputation median: {name}")
        require(np.array_equal(values[flags == 1], np.full(int((flags == 1).sum()), saved, dtype=np.float32)),
                f"Imputed values differ from frozen train median: {name}")
        ordered = np.sort(finite)
        middle = len(ordered) // 2
        center = ordered[middle:middle + 1] if len(ordered) % 2 else ordered[middle - 1:middle + 1]
        reconstructed = np.float32(np.mean(center.astype(np.float64)))
        # Each original float64 central value rounds into this neighboring
        # float32 interval.  Use both endpoints so opposite-sign cancellation
        # does not make a median-level ULP tolerance spuriously too strict.
        lower = float(np.mean(np.nextafter(center, np.float32(-np.inf)).astype(np.float64)))
        upper = float(np.mean(np.nextafter(center, np.float32(np.inf)).astype(np.float64)))
        tolerance = max(abs(float(np.spacing(saved))), abs(float(np.spacing(reconstructed))), np.finfo(np.float32).tiny)
        error = abs(float(saved) - float(reconstructed))
        require(lower - tolerance <= float(saved) <= upper + tolerance, f"Train-only retained median mismatch: {name}")
        largest_error = max(largest_error, error)
    for reason in preprocessing["dropped_columns"].values():
        require(reason in ("all_missing_in_training_partition", "constant_finite_values_in_training_partition"), "Unapproved column drop rule")
    return {"retained_predictors": len(retained), "dropped_predictors": len(preprocessing["dropped_columns"]),
            "maximum_saved_float32_median_absolute_difference": largest_error,
            "median_scope": "retained_training_values_reconstructed_from_float32_flags_not_exact_raw_float64_or_dropped_source_columns"}


class VerificationBudget:
    def __init__(self, *, seconds=3600, memory_bytes=20 * 1024**3):
        require(0 < seconds <= 3600 and 0 < memory_bytes <= 20 * 1024**3, "Invalid independent verification budget")
        self.started = time.monotonic()
        self.seconds = seconds
        self.memory_bytes = memory_bytes
        self.peak_rss_bytes = 0

    def check(self):
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self.peak_rss_bytes = max(self.peak_rss_bytes, int(peak if platform.system() == "Darwin" else peak * 1024))
        require(time.monotonic() - self.started <= self.seconds, "Independent verification exceeded one-hour ceiling")
        require(self.peak_rss_bytes <= self.memory_bytes, "Independent verification exceeded RSS ceiling")


def equal_number(actual, expected, label):
    if expected is None or (isinstance(expected, float) and np.isnan(expected)):
        require(actual is None or pd.isna(actual), f"Different missing value: {label}")
    elif isinstance(expected, (bool, np.bool_, int, np.integer)):
        require(actual == expected, f"Different count/flag: {label}")
    else:
        require(np.isclose(actual, expected, rtol=1e-12, atol=2e-14), f"Different numeric result: {label}")


def verify_event_chain(run_root, state):
    """Independently reconcile immutable snapshots, counters and causal seals."""
    previous = None
    previous_active = 0.0
    initiated = set()
    fit_count = preprocessor_count = 0
    winner_events = {}
    active_worker = None
    events = sorted((run_root / "events").glob("*.json"))
    require(len(events) == state["event_sequence"], "Durable event count changed")
    require(events, "No durable checkpoint events")
    for sequence, path in enumerate(events, 1):
        event = read_json(path)
        require(path.name == f"{sequence:06d}.json" and event["event_sequence"] == sequence, "Nonsequential checkpoint event")
        require(event["previous_event_sha256"] == previous, "Broken checkpoint event hash chain")
        snapshot = event["ledger_snapshot"]
        require(canonical(snapshot) == event["ledger_snapshot_sha256"], "Checkpoint snapshot hash changed")
        require(snapshot["active_seconds"] >= previous_active, "Active compute counter went backwards")
        previous_active = snapshot["active_seconds"]
        tasks = {task["task_id"]: task for task in snapshot["tasks"]}
        require(len(tasks) == len(snapshot["tasks"]), "Duplicate scheduled task ID")
        kind, payload = event["event_type"], event["payload"]
        if kind == "TASK_INITIATED":
            task_id = payload["task_id"]
            require(task_id not in initiated, "A scheduled task was initiated more than once")
            task = tasks[task_id]
            require(task["status"] == "RUNNING" and task.get("initiated") is True, "Task initiation not durable before work")
            initiated.add(task_id)
            fit_count += task["action"] == "fit"
            preprocessor_count += task["action"] == "prepare"
            if task["action"] == "predict_outer" or task["split_id"].endswith("_outer_train"):
                require(str(task["outer_year"]) in winner_events, "Outer action preceded its sealed winner")
            require(fit_count <= 100 and preprocessor_count <= 16, "Frozen initiated-fit ceiling exceeded")
        elif kind == "WINNER_SEALED":
            year = str(payload["outer_year"])
            require(year not in winner_events, "An outer winner was resealed")
            inner = [task for task in tasks.values() if task["outer_year"] == int(year)
                     and task["action"] == "fit" and "_validate_" in task["split_id"]]
            require(len(inner) == 24 and all(task["status"] == "COMPLETE" for task in inner), "Winner sealed before complete 24-fit grid")
            winner_events[year] = sequence
        elif kind == "WORKER_STARTED":
            require(active_worker is None, "Concurrent tuning child processes")
            active_worker = payload["worker_pid"]
        elif kind in ("WORKER_EXITED", "WORKER_SUPERVISOR_INTERRUPTED"):
            require(active_worker is not None, "Worker exit without durable start")
            active_worker = None
        require(snapshot["initiated_estimator_fits"] == fit_count, "Event estimator counter differs from initiations")
        require(snapshot["initiated_preprocessor_fits"] == preprocessor_count, "Event preprocessing counter differs from initiations")
        previous = canonical(event)
    require(previous == state["last_event_sha256"], "Ledger event head changed")
    final_payload = {key: value for key, value in state.items() if key not in ("event_sequence", "last_event_sha256")}
    require(snapshot == final_payload, "Mutable ledger differs from final immutable snapshot")
    require(active_worker is None, "Verification requires all tuning children exited")
    return {"event_count": len(events), "initiated_estimator_fits": fit_count,
            "initiated_preprocessor_fits": preprocessor_count, "winner_event_sequences": winner_events}


def expected_schedule(protocol):
    expected = []
    for fold in protocol["chronology"]["folds"]:
        year = fold["outer_outcome_year"]
        for split in fold["inner_splits"]:
            sid = split["split_id"]
            expected.append((f"{sid}__prepare", "prepare", year, sid, None))
            expected.extend((f"{sid}__{candidate}", "fit", year, sid, candidate)
                            for candidate in protocol["rf"]["candidate_fit_order_within_split"])
        sid = f"{year}_outer_train"
        expected.extend(((f"{sid}__prepare", "prepare", year, sid, None),
                         (f"{sid}__fit", "fit", year, sid, None),
                         (f"{year}__predict_outer", "predict_outer", year, sid, None)))
    require(len(expected) == 120 and sum(row[1] == "fit" for row in expected) == 100, "Frozen schedule changed")
    return expected


def verify_launch_policy(run_root, identity, protocol, amendment):
    """Check the separately pinned operational gate without changing statistics."""
    policy = amendment.get("resource_launch_policy")
    require(identity.get("resource_launch_policy") == policy, "Identity did not pin the execution launch policy")
    gate = protocol["local_budget"]["minimum_available_memory_before_launch_bytes"]
    pressure_required = False
    if policy is not None:
        require(policy == {"minimum_available_memory_before_launch_bytes": 7 * 1024**3,
                           "require_macos_pressure_dispatch_value": 1,
                           "stop_on_macos_pressure_not_normal": True}
                and type(policy["minimum_available_memory_before_launch_bytes"]) is int
                and type(policy["require_macos_pressure_dispatch_value"]) is int
                and policy["stop_on_macos_pressure_not_normal"] is True, "Unreviewed operational resource policy")
        require(gate == 12 * 1024**3, "Historical frozen launch gate changed")
        gate = policy["minimum_available_memory_before_launch_bytes"]
        pressure_required = True
    checked_launches = checked_exit_samples = 0
    for path in sorted((run_root / "events").glob("*.json")):
        event = read_json(path)
        snapshot = event["ledger_snapshot"]
        if event["event_type"] == "TASK_INITIATED":
            preflight = snapshot["last_preflight"]
            require(preflight["memory"]["available_bytes"] >= gate, "Financial child launched below its bound available-memory gate")
            require(preflight["free_disk_bytes"] >= protocol["local_budget"]["minimum_free_disk_before_launch_bytes"], "Financial child launched below frozen disk gate")
            if pressure_required:
                require(preflight["macos_memory_pressure"]["dispatch_value"] == 1, "Financial child launched under non-normal/unknown macOS pressure")
            checked_launches += 1
        if event["event_type"] == "WORKER_EXITED" and pressure_required:
            sample = snapshot["last_resource_sample"]["macos_memory_pressure"]
            require(sample is not None and sample["dispatch_value"] == 1, "Completed child has non-normal/unknown recorded pressure")
            checked_exit_samples += 1
    return {"minimum_available_memory_before_launch_bytes": gate, "normal_macos_pressure_dispatch_value": 1 if pressure_required else None,
            "checked_financial_child_launches": checked_launches, "checked_child_exit_pressure_samples": checked_exit_samples,
            "original_statistical_protocol_and_upper_limits_unchanged": True}


def verify_plan(plan, *, project_root, run_root, prepare=None):
    roots = [project_root, run_root]
    require(plan.get("planning_financial_values_decoded") is False, "Planner decoded financial values")
    verify_pin(plan["protocol_pin"], roots=roots)
    verify_pin(plan["reference_build_identity_pin"], roots=roots)
    years = [annual["year"] for annual in plan["annuals"]]
    require(years and all(type(year) is int and 2010 <= year <= 2023 for year in years), "Source plan includes protected financial year")
    if plan["planned_action"] == "predict_outer":
        require(years == [plan["outer_year"]], "Outer prediction plan includes another financial year")
    else:
        expected_years = list(range(int(plan["train_start"][:4]), int(plan["train_end"][:4]) + 1))
        if plan["stage"] == "inner":
            expected_years.append(plan["validation_year"])
        require(years == expected_years and max(years) < plan["outer_year"], "Premature outer source decode in training plan")
    projected = 0
    for annual in plan["annuals"]:
        verify_pin(annual["audit"], roots=roots)
        for item in annual["source_pins"]:
            verify_pin(item, roots=roots)
        total = sum(item["projected_compressed_bytes"] for item in annual["projections"])
        require(total == annual["projected_compressed_bytes"], "Annual source projection sum differs")
        projected += total
    require(projected == plan["projected_source_read_bytes"], "Source read reservation projection differs")
    if prepare is not None:
        require(prepare["source_plan"] == plan, "Prepared inputs are not bound to exact source plan")
    return projected


def load_manifest(pin, *, run_root):
    manifest = read_json(verify_pin(pin, roots=[run_root]))
    require(manifest.get("status") == "COMPLETE", "Incomplete checkpoint manifest")
    for item in manifest["artifacts"].values():
        verify_pin(item, roots=[run_root])
    return manifest


def verify_prepare(manifest, *, protocol, project_root, run_root, budget):
    artifacts = manifest["artifacts"]
    plan = read_json(verify_pin(manifest["plan"], roots=[run_root]))
    verify_plan(plan, project_root=project_root, run_root=run_root, prepare=manifest)
    train = pd.read_parquet(artifacts["train_ledger"]["path"])
    first, last = int(plan["train_start"][:4]), int(plan["train_end"][:4])
    validate_dates(train, first_year=first, last_year=last)
    prediction_eligible = boolean_column(train, "prediction_eligible")
    target_available = boolean_column(train, "target_available")
    fit_mask = boolean_column(train, "fit_mask")
    require(np.array_equal(fit_mask, prediction_eligible & target_available), "Supervised fit mask changed")
    y = train.target_class.to_numpy(dtype=float, na_value=np.nan)
    require(np.isin(y[fit_mask], CLASS_IDS).all() and set(y[fit_mask]) == set(CLASS_IDS), "Training does not contain all ten observed canonical classes")
    require(keys_hash(train) == manifest["train_original_keys_sha256"], "Original training membership keys differ")
    require(keys_hash(train, fit_mask) == manifest["train_keys_sha256"], "Supervised training keys differ")
    labels = np.load(artifacts["train_y"]["path"], mmap_mode="r", allow_pickle=False)
    require(labels.dtype == np.int64 and np.array_equal(labels, y[fit_mask]), "Prepared labels differ from exactly supervised original rows")
    preprocessing = read_json(verify_pin(manifest["preprocessor"], roots=[run_root]))
    require(preprocessing["fit_keys_sha256"] == manifest["train_keys_sha256"], "Preprocessor used different training keys")
    require(preprocessing["train_start"] == plan["train_start"] and preprocessing["train_end"] == plan["train_end"], "Preprocessor training window differs")
    require(train.loc[fit_mask, "outcome_end"].max() <= preprocessing["label_observed_through"] <= preprocessing["train_end"], "Preprocessor target completion cutoff differs")
    require(preprocessing["training_only_schema_selection"] is True and preprocessing["estimator_fitted"] is False
            and preprocessing["scaling"] is None and preprocessing["clipping"] is None and preprocessing["winsorization"] is None,
            "Preprocessor introduced an unapproved transform")
    matrix = np.load(artifacts["train_X"]["path"], mmap_mode="r", allow_pickle=False)
    median = verify_prepared_matrix(matrix, preprocessing)
    evidence = read_json(artifacts["train_schema_evidence"]["path"])
    require(set(evidence) == set(preprocessing["input_columns"]), "Train schema evidence omits input predictors")
    retained_positions = {name: index for index, name in enumerate(preprocessing["retained_columns"])}
    for name, item in evidence.items():
        require(type(item["finite_count"]) is int and 0 <= item["finite_count"] <= len(labels), "Invalid finite training count")
        require(type(item["finite_unique_count"]) is int and 0 <= item["finite_unique_count"] <= item["finite_count"], "Invalid unique training count")
        if name in retained_positions:
            index = retained_positions[name]
            require(item["finite_count"] == int((matrix[:, 2 * index + 1] == 0).sum()) and item["finite_unique_count"] >= 2,
                    "Retained training evidence differs from flags")
            require(item["median"] == preprocessing["medians"][name] and item["drop_reason"] is None, "Retained median evidence changed")
        else:
            reason = preprocessing["dropped_columns"][name]
            require(item["drop_reason"] == reason and item["median"] is None, "Dropped predictor evidence changed")
            require((item["finite_count"] == item["finite_unique_count"] == 0) if reason.startswith("all_missing")
                    else (item["finite_count"] > 0 and item["finite_unique_count"] == 1), "Dropped predictor does not satisfy frozen evidence rule")
    require(manifest["matrix_width"] == matrix.shape[1] and manifest["train_rows"] == matrix.shape[0], "Prepared train matrix dimensions differ")
    result = {"split_id": manifest["split_id"], "train_rows": len(labels), "matrix_width": matrix.shape[1], **median}
    if manifest["stage"] == "inner":
        validation = pd.read_parquet(artifacts["prediction_ledger"]["path"])
        prediction_mask = boolean_column(validation, "prediction_eligible")
        observed = boolean_column(validation, "target_available")
        require(keys_hash(validation) == manifest["prediction_original_keys_sha256"], "Original validation membership keys differ")
        require(keys_hash(validation, prediction_mask) == manifest["prediction_keys_sha256"], "Validation prediction keys differ")
        require(keys_hash(validation, prediction_mask & observed) == manifest["score_keys_sha256"], "Validation scoring keys differ")
        result["chronology"] = validate_chronology(train, validation, fit_mask, train_start_year=first, train_end_year=last,
                                                   validation_year=manifest["validation_year"])
        prediction = np.load(artifacts["prediction_X"]["path"], mmap_mode="r", allow_pickle=False)
        verify_prediction_matrix(prediction, preprocessing, expected_rows=int(prediction_mask.sum()))
        del prediction, validation
    budget.check()
    del train, labels, matrix
    gc.collect()
    return result


def verify_prediction_matrix(matrix, preprocessing, *, expected_rows):
    require(matrix.dtype == np.float32 and matrix.shape == (expected_rows, len(preprocessing["output_columns"]))
            and np.isfinite(matrix).all(), "Prediction prepared matrix dimensions/finite dtype differ")
    for index, name in enumerate(preprocessing["retained_columns"]):
        value, flags = matrix[:, 2 * index], matrix[:, 2 * index + 1]
        require(np.isin(flags, (0, 1)).all(), "Nonbinary prediction missing flags")
        require((value[flags == 1] == np.float32(preprocessing["medians"][name])).all(), "Prediction median differs from frozen training state")


def verify_reports(manifest, scoring, eligible, probabilities):
    """Reconcile all saved row/month diagnostics and descriptive class tables."""
    artifacts = manifest["artifacts"]
    summary = manifest["summary"]
    for name, expected in scoring["summary"].items():
        if name in summary:
            equal_number(summary[name], expected, f"summary {name}")
    require(summary["log_loss_evaluation_floor"] == FLOOR and summary["label_imputation_performed"] is False
            and summary["probability_clipping_or_renormalization"] is False, "Scoring policy changed")
    saved_summary = read_json(artifacts["summary"]["path"])
    require(saved_summary == summary, "Summary artifact differs from manifest")
    monthly = pd.read_parquet(artifacts["monthly"]["path"])
    require(list(monthly.outcome_month) == [row["outcome_month"] for row in scoring["monthly"]], "Monthly diagnostic calendar differs")
    persisted_losses = manifest["monthly_log_losses"]
    require(len(persisted_losses) == 12, "Saved manifest has incomplete monthly scores")
    for index, expected in enumerate(scoring["monthly"]):
        for name in ("prediction_rows", "labelled_rows", "log_loss", "brier_sum10", "accuracy"):
            equal_number(monthly.iloc[index][name], expected[name], f"monthly {expected['outcome_month']} {name}")
        require(monthly.iloc[index]["score_key_sha256"] == expected["score_keys_sha256"], "Monthly observed scoring keys differ")
        require(persisted_losses[index]["outcome_month"] == expected["outcome_month"]
                and persisted_losses[index]["score_key_sha256"] == expected["score_keys_sha256"], "Manifest monthly scoring keys differ")
        equal_number(persisted_losses[index]["log_loss"], expected["log_loss"], "manifest monthly loss")
    rows = pd.read_parquet(artifacts["row_scores"]["path"])
    require(keys_hash(rows) == keys_hash(eligible), "Row diagnostic prediction keys differ")
    for name, expected in (("log_loss", scoring["row_log_loss"]), ("brier_sum10", scoring["row_brier_sum10"]),
                           ("correct", scoring["row_accuracy"]), ("predicted_class", scoring["predicted_class"])):
        require(np.allclose(rows[name].to_numpy(dtype=float, na_value=np.nan), expected, rtol=1e-12, atol=2e-14, equal_nan=True),
                f"Row diagnostic differs: {name}")
    y = eligible.target_class.to_numpy(dtype=float, na_value=np.nan)
    observed = scoring["observed_mask"]
    hard = scoring["predicted_class"]
    for name, top in (("toplabel_calibration", True), ("classwise_calibration", False)):
        table = pd.read_parquet(artifacts[name]["path"])
        require(len(table) == (10 if top else 100), "Calibration bins changed")
        for class_id in ((None,) if top else CLASS_IDS):
            confidence = probabilities[observed].max(axis=1) if top else probabilities[observed, class_id - 1]
            hit = hard[observed] == y[observed] if top else y[observed] == class_id
            bins = np.minimum((confidence * 10).astype(int), 9)
            for bucket in range(10):
                row = table.loc[table.bin_id == bucket]
                if not top:
                    row = row.loc[row.class_id == class_id]
                require(len(row) == 1, "Calibration cell duplicated or absent")
                row = row.iloc[0]
                mask = bins == bucket
                count = int(mask.sum())
                equal_number(row.labelled_count, count, "calibration count")
                mean = float(confidence[mask].mean()) if count else None
                rate = float(hit[mask].mean()) if count else None
                equal_number(row.mean_probability, mean, "calibration confidence")
                equal_number(row.observed_frequency, rate, "calibration frequency")
                equal_number(row.absolute_gap, abs(mean - rate) if count else None, "calibration absolute gap")
    # The remaining class/confusion tables are checked cell-by-cell below.
    verify_class_tables(artifacts, eligible, y, observed, hard)


def verify_class_tables(artifacts, eligible, y, observed, hard):
    periods = ["all", *sorted(eligible.outcome_month.unique())]
    for table_name in ("extreme_precision", "mode_class_performance", "confusion"):
        table = pd.read_parquet(artifacts[table_name]["path"])
        expected_rows = len(periods) * (2 if table_name == "extreme_precision" else 10 if table_name == "mode_class_performance" else 100)
        key_columns = ["period", "actual_class", "predicted_class"] if table_name == "confusion" else ["period", "class_id"]
        require(len(table) == expected_rows and not table.columns.duplicated().any()
                and not table.duplicated(key_columns).any(), f"Incomplete/duplicate {table_name} table")
        for row in table.to_dict("records"):
            period = row.get("period", "all")
            period_mask = np.ones(len(eligible), dtype=bool) if period == "all" else eligible.outcome_month.to_numpy() == period
            require(period == "all" or period in set(eligible.outcome_month), "Diagnostic period outside validation")
            if "class_id" in row:
                k = row["class_id"]
                require(k in CLASS_IDS, "Diagnostic canonical class differs")
                predicted, actual = period_mask & (hard == k), period_mask & observed & (y == k)
                known_predicted = predicted & observed
                hits = int((actual & predicted).sum())
                checks = {"prediction_rows": int(period_mask.sum()), "hard_mode_prediction_count": int(predicted.sum()),
                          "labelled_hard_mode_count": int(known_predicted.sum()), "missing_label_hard_mode_count": int((predicted & ~observed).sum()),
                          "true_positive_count": hits, "actual_support": int(actual.sum()), "predicted_support": int(predicted.sum()),
                          "labelled_predicted_support": int(known_predicted.sum()),
                          "missing_label_predicted_support": int((predicted & ~observed).sum()),
                          "precision": hits / known_predicted.sum() if known_predicted.any() else None,
                          "recall": hits / actual.sum() if actual.any() else None}
                for key, expected in checks.items():
                    if key in row:
                        equal_number(row[key], expected, f"{table_name} {key}")
            if "actual_class" in row and "predicted_class" in row:
                require(row["actual_class"] in CLASS_IDS and row["predicted_class"] in CLASS_IDS, "Confusion canonical classes differ")
                count = int((period_mask & observed & (y == row["actual_class"]) & (hard == row["predicted_class"])).sum())
                for key in ("count", "labelled_count"):
                    if key in row:
                        equal_number(row[key], count, f"confusion {key}")


def verify_predictions(manifest, *, prepared, run_root, expected_year, reload_model=None, budget):
    artifacts = manifest["artifacts"]
    predictions = pd.read_parquet(artifacts["predictions"]["path"])
    validate_dates(predictions, first_year=expected_year, last_year=expected_year)
    original_pin = artifacts.get("prediction_ledger") or prepared["artifacts"]["prediction_ledger"]
    original = pd.read_parquet(verify_pin(original_pin, roots=[run_root]))
    if prepared["stage"] == "outer":
        train = pd.read_parquet(verify_pin(prepared["artifacts"]["train_ledger"], roots=[run_root]))
        fit_mask = boolean_column(train, "fit_mask")
        validate_chronology(train, original, fit_mask,
            train_start_year=int(prepared["source_plan"]["train_start"][:4]),
            train_end_year=int(prepared["source_plan"]["train_end"][:4]), validation_year=expected_year)
        del train
    for name in original.columns:
        require(name in predictions, "Prediction artifact lost original cohort metadata")
        pd.testing.assert_series_equal(predictions[name], original[name], check_names=False)
    eligible_mask = boolean_column(predictions, "prediction_eligible")
    observed = boolean_column(predictions, "target_available")
    eligible = predictions.loc[eligible_mask].reset_index(drop=True)
    require(keys_hash(predictions) == prepared.get("prediction_original_keys_sha256", manifest.get("prediction_original_keys_sha256")), "All original prediction keys changed")
    expected_keys = prepared.get("prediction_keys_sha256", manifest.get("prediction_keys_sha256"))
    require(keys_hash(eligible) == expected_keys, "Prediction matrix key order changed")
    values = np.load(artifacts["probabilities"]["path"], mmap_mode="r", allow_pickle=False)
    require(np.array_equal(values, eligible.loc[:, list(PROBS)].to_numpy(dtype=float)), "Ledger probabilities differ from saved native matrix")
    require(predictions.loc[~eligible_mask, list(PROBS)].isna().all().all(), "Ineligible original rows acquired predictions")
    scoring = independent_scores(values, eligible.target_class.to_numpy(dtype=float, na_value=np.nan), eligible, expected_year=expected_year)
    require(np.array_equal(scoring["observed_mask"], observed[eligible_mask]), "Scored label mask differs from original eligibility ledger")
    verify_reports(manifest, scoring, eligible, values)
    if reload_model is not None:
        import joblib
        from threadpoolctl import threadpool_limits
        estimator = joblib.load(verify_pin(reload_model["artifacts"]["model"], roots=[run_root]))
        require(list(estimator.classes_) == list(CLASS_IDS) and estimator.get_params(deep=False)["n_jobs"] == 1,
                "Selected saved model native classes or inference threads changed")
        require(estimator.get_params(deep=False) == reload_model["saved_inference_effective_parameters"],
                "Serialized selected estimator parameters differ from fit receipt")
        require(len(estimator.estimators_) == 500 and estimator.n_features_in_ == prepared["matrix_width"],
                "Serialized selected forest tree count or prepared width differs")
        x_pin = artifacts.get("prediction_X") or prepared["artifacts"]["prediction_X"]
        matrix = np.load(verify_pin(x_pin, roots=[run_root]), mmap_mode="r", allow_pickle=False)
        preprocessing = read_json(prepared["preprocessor"]["path"])
        verify_prediction_matrix(matrix, preprocessing, expected_rows=len(eligible))
        with threadpool_limits(limits=1):
            reproduced = estimator.predict_proba(matrix)
        require(np.allclose(reproduced, values, rtol=1e-12, atol=2e-14), "Selected saved model does not reproduce probabilities")
        difference = float(np.max(np.abs(reproduced - values)))
        del estimator, matrix, reproduced
    else:
        difference = None
    budget.check()
    result = {"summary": scoring["summary"], "monthly": scoring["monthly"],
              "selected_model_probability_maximum_absolute_difference": difference}
    del predictions, original, eligible, values
    gc.collect()
    return result


def verify_fit_parameters(manifest, protocol):
    matches = [candidate for candidate in protocol["rf"]["candidates"] if candidate["candidate_id"] == manifest["candidate_id"]]
    require(len(matches) == 1, "Unexpected candidate fit")
    candidate = matches[0]
    expected = {**protocol["rf"]["fixed_parameters"], **{key: candidate[key] for key in ("max_depth", "min_samples_leaf", "max_features")}}
    require(manifest["explicit_parameters"] == expected and manifest["fit_effective_parameters"] == expected, "Fit used unfrozen effective RF parameters")
    require(manifest["saved_inference_effective_parameters"] == {**expected, "n_jobs": 1}, "Inference parameters changed statistically")
    require(manifest["estimator_fits"] == 1 and manifest["preprocessing_fits"] == 0
            and manifest["native_class_ids"] == list(CLASS_IDS) and manifest["probability_class_ids"] == list(CLASS_IDS)
            and manifest["sample_weight"] is None and manifest["validation_labels_supplied_to_estimator"] is False,
            "Fit class/weight/label isolation contract differs")
    require(manifest["fit_tree_threads"] == 4 and manifest["inference_threads"] == 1, "Fit/inference parallelism changed")
    require(manifest["effective_feature_count"] == nominal_features(candidate, manifest["matrix_width"]), "Nominal feature count differs")
    require(manifest["runtime"] == protocol["runtime_snapshot_for_future_matching"], "Fit runtime differs from frozen snapshot")


def verify_run(project_root, run_root, *, allow_partial=False, check_runtime=True):
    budget = VerificationBudget()
    project_root, run_root = Path(project_root).resolve(), Path(run_root).resolve()
    require(run_root.is_relative_to(project_root / "outputs"), "Attempt is not a private project output")
    state = read_json(run_root / "ledger.json")
    chain = verify_event_chain(run_root, state)
    identity = read_json(verify_pin(state["build_identity_pin"], roots=[run_root]))
    auth = read_json(verify_pin(state["authorization_pin"], roots=[run_root]))
    require(identity["project_root"] == str(project_root) and identity["run_root"] == str(run_root), "Attempt identity root differs")
    original_identity = {key: value for key, value in identity.items() if key not in ("identity_sha256", "run_root", "created_at")}
    require(canonical(original_identity) == identity["identity_sha256"], "Build identity canonical hash differs")
    for item in (identity["protocol_pin"], identity["amendment_pin"], *identity["reference_pins"], *identity["code_pins"]):
        verify_pin(item, roots=[project_root])
    protocol = read_json(identity["protocol_pin"]["path"])
    if check_runtime:
        for name, expected in identity["runtime"].items():
            actual = platform.python_version() if name == "python" else importlib.metadata.version(name)
            require(actual == expected, f"Independent verification runtime changed: {name}")
    amendment = read_json(identity["amendment_pin"]["path"])
    require(amendment["protocol_sha256"].removeprefix("sha256:") == identity["protocol_pin"]["sha256"]
            and amendment["execution_authorized"] is True and amendment["statistical_protocol_changed"] is False,
            "Execution amendment does not preserve frozen authorized protocol")
    require(auth["execution_amendment_pin"] == identity["amendment_pin"] and auth["run_root"] == str(run_root), "Authorization binding differs")
    for name in ("fitting_authorized", "preprocessing_fitting_authorized", "prediction_authorized", "new_financial_reads_authorized"):
        require(auth.get(name) is True, "Explicit later execution authorization absent")
    require(auth.get("backtest_authorized") is False and auth.get("financial_2024_plus_authorized") is False, "Execution scope was broadened")
    launch_policy = verify_launch_policy(run_root, identity, protocol, amendment)
    schedule = expected_schedule(protocol)
    require(len(state["tasks"]) == len(schedule), "Scheduled action count changed")
    for task, (task_id, action, year, sid, candidate) in zip(state["tasks"], schedule):
        require((task["task_id"], task["action"], task["outer_year"], task["split_id"]) == (task_id, action, year, sid), "Frozen action schedule changed")
        if candidate is not None:
            require(task["candidate_id"] == candidate, "Candidate ordering changed")
    complete = all(task["status"] == "COMPLETE" for task in state["tasks"])
    if not allow_partial:
        require(complete and state["status"] == "COMPLETE_PENDING_INDEPENDENT_VERIFICATION", "Full verification requires all 100 fits,16 states,4 outer predictions")
        require(state["initiated_estimator_fits"] == 100 and state["initiated_preprocessor_fits"] == 16 and len(state["winners"]) == 4,
                "Incomplete complete-search fit/state/winner counts")
    else:
        require(state["status"] in ("PREPARED", "PAUSED", "WAITING_FOR_MEMORY", "WAITING_FOR_MEMORY_PRESSURE", "WAITING_FOR_DISK", "COMPLETE_PENDING_INDEPENDENT_VERIFICATION"),
                "Partial verification requires an idle nonfailed attempt")
    limits = protocol["local_budget"]
    require(state["active_seconds"] <= limits["total_preprocessing_fit_prediction_scoring_wall_seconds"], "Active compute budget exceeded")
    require(state["reserved_source_bytes"] <= limits["total_projected_compressed_source_read_stop_bytes"], "Reserved source read budget exceeded")
    require(state["resource_peaks"]["process_tree_rss_bytes"] < limits["process_tree_rss_stop_bytes"], "Observed resource RSS limit exceeded")
    require(state["resource_peaks"]["new_output_bytes"] < limits["new_output_stop_bytes"], "Observed output limit exceeded")
    require(limits["separate_zero_fit_verification_wall_seconds"] == budget.seconds, "Independent verification time ceiling changed")
    prepared, fits, predictions, output = {}, {}, {}, []
    reserved = 0
    for task in state["tasks"]:
        if task["status"] != "COMPLETE":
            continue
        receipt = read_json(verify_pin(task["receipt_pin"], roots=[run_root]))
        request = read_json(verify_pin(task["request_pin"], roots=[run_root]))
        require(receipt["status"] == "COMPLETE" and receipt["action"] == task["action"], "Task receipt binding differs")
        require(request["action"] == task["action"] and request["outer_year"] == task["outer_year"]
                and request["split_id"] == task["split_id"], "Task request binding differs")
        for name in ("outer_year", "split_id"):
            require(receipt[name] == task[name], "Task receipt clock differs")
        for item in receipt["artifacts"]:
            verify_pin(item, roots=[run_root])
        if task["action"] in ("prepare", "predict_outer"):
            plan = read_json(verify_pin(task["plan_path"], roots=[run_root]))
            projected = verify_plan(plan, project_root=project_root, run_root=run_root)
            reservation = read_json(verify_pin(task["reservation_pin"], roots=[run_root]))
            require(reservation["execution_authorized"] is True and reservation["plan_sha256"] == canonical(plan)
                    and reservation["projected_source_read_bytes"] == projected, "Source decoding was not exactly reserved")
            reserved += projected
        if task["action"] == "prepare":
            manifest = load_manifest(receipt["prepare_manifest"], run_root=run_root)
            output.append(verify_prepare(manifest, protocol=protocol, project_root=project_root, run_root=run_root, budget=budget))
            prepared[task["split_id"]] = manifest
        elif task["action"] == "fit":
            manifest = load_manifest(receipt["fit_manifest"], run_root=run_root)
            require(receipt["candidate_id"] == manifest["candidate_id"] == task["candidate_id"] == request["candidate_id"], "Fit candidate binding differs")
            verify_fit_parameters(manifest, protocol)
            require(manifest["prepare_manifest"] == read_json(next(t for t in state["tasks"] if t["split_id"] == task["split_id"] and t["action"] == "prepare")["receipt_pin"]["path"])["prepare_manifest"],
                    "Fit not bound to exact split-local prepared state")
            fits[(task["split_id"], task["candidate_id"])] = manifest
            if manifest["stage"] == "inner":
                result = verify_predictions(manifest, prepared=prepared[task["split_id"]], run_root=run_root,
                                            expected_year=prepared[task["split_id"]]["validation_year"], budget=budget)
                predictions[(task["split_id"], task["candidate_id"])] = result
        else:
            manifest = load_manifest(receipt["prediction_manifest"], run_root=run_root)
            winner = read_json(verify_pin(state["winners"][str(task["outer_year"])], roots=[run_root]))
            require(manifest["winner_path"] == state["winners"][str(task["outer_year"])] and manifest["candidate_id"] == winner["candidate_id"],
                    "Outer forecast not bound to sealed annual winner")
            fitted = fits[(task["split_id"], winner["candidate_id"])]
            result = verify_predictions(manifest, prepared=prepared[task["split_id"]], run_root=run_root,
                                        expected_year=task["outer_year"], reload_model=fitted, budget=budget)
            predictions[(task["split_id"], "outer_prediction")] = result
        budget.check()
    require(reserved == state["reserved_source_bytes"], "Completed action source reservations differ from ledger")
    selected = []
    for year, winner_pin in state["winners"].items():
        winner = read_json(verify_pin(winner_pin, roots=[run_root]))
        fold = next(f for f in protocol["chronology"]["folds"] if f["outer_outcome_year"] == int(year))
        split_ids = [split["split_id"] for split in fold["inner_splits"]]
        by_candidate = {candidate["candidate_id"]: [row for sid in split_ids for row in predictions[(sid, candidate["candidate_id"])]["monthly"]]
                        for candidate in protocol["rf"]["candidates"]}
        result = select_winner(protocol["rf"]["candidates"], by_candidate, [prepared[sid]["matrix_width"] for sid in split_ids])
        require(winner["candidate_id"] == result["candidate_id"] and winner["tie_candidates"] == result["tie_candidate_ids"], "Saved winner/tie selection differs from independent recomputation")
        for candidate_id, score in result["scores"].items():
            equal_number(winner["candidate_scores"][candidate_id], score, "winner equal-month score")
        selected.append({"outer_year": int(year), "candidate_id": result["candidate_id"], "candidate_scores": result["scores"]})
    budget.check()
    return {"schema": "dmv2-rf-expanded-tuning-independent-verification/v1", "status": "PASS_COMPLETE" if complete else "PASS_COMPLETED_CHECKPOINTS_ONLY",
            "complete_search_verified": complete, "run_root": str(run_root), "build_identity_sha256": identity["identity_sha256"],
            "new_estimator_fits": 0, "new_preprocessor_fits": 0, "new_source_financial_decodes": 0,
            "event_chain": chain, "verified_prepared_splits": output, "verified_fit_manifests": len(fits),
            "verified_resource_launch_policy": launch_policy,
            "verified_inner_probability_ledgers": sum(key[1] != "outer_prediction" for key in predictions),
            "verified_selected_outer_model_reproductions": sum(key[1] == "outer_prediction" for key in predictions),
            "selected_winners": selected, "reserved_source_bytes": reserved,
            "verification_seconds": time.monotonic() - budget.started, "verification_peak_rss_bytes": budget.peak_rss_bytes,
            "runtime_verified": check_runtime,
            "scope": {"all_saved_inner_probability_losses_masks_and_months": True,
                      "all_selected_outer_saved_models_reproduce_probabilities": complete,
                      "retained_train_medians": "independently_reconstructed_from_saved_float32_finite_values_with_quantization_intervals",
                      "dropped_columns": "pinned_train_only_schema_evidence_and_frozen_implementation_review",
                      "exact_raw_float64_or_dropped_source_reconstruction": False,
                      "all_feature_formula_identity_knowledge_time_survivorship_certification": False,
                      "backtest_rerun_or_portfolio_evaluation": False}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--completed-checkpoints-only", action="store_true")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    require(output.is_relative_to(args.run_root.resolve()) and not output.exists(), "Use a fresh verification receipt inside this attempt")
    report = verify_run(args.project_root, args.run_root, allow_partial=args.completed_checkpoints_only)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "report": str(output), "new_estimator_fits": 0}, sort_keys=True))


if __name__ == "__main__":
    main()

