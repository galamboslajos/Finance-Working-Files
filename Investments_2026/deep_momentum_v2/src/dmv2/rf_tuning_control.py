"""Explicit, sequential RF tuning supervisor; importing this module never fits.

One durable initiation is allowed per task. Pausing finishes the current child,
seals its receipt, then exits. Resuming rehashes every completed artifact. Failed
or interrupted tasks are preserved and cannot be retried by this protocol.
"""
from __future__ import annotations

import contextlib
import datetime as dt
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
from typing import Any, Callable


SCHEMA = "dmv2-rf-expanded-tuning-controller/v1"
TERMINAL_FAILURES = {"FAILED", "FAILED_INTERRUPTED", "STOPPED_BUDGET"}
WAITING = {"PAUSED", "WAITING_FOR_MEMORY", "WAITING_FOR_MEMORY_PRESSURE", "WAITING_FOR_DISK"}


class TuningControlError(RuntimeError):
    """A fail-closed protocol, integrity, lifecycle or budget violation."""


class TuningBudgetError(TuningControlError):
    """An observed fixed resource budget exhausted without an automatic retry."""


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def atomic_json(path: str | Path, value: Any, *, exclusive: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive and path.exists():
        raise TuningControlError(f"Refusing to overwrite {path}")
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    with tmp.open("x") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def file_pin(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def verify_pin(pin: dict[str, Any], *, within: Path | None = None) -> Path:
    path = Path(pin["path"]).resolve()
    if within is not None and not path.is_relative_to(within.resolve()):
        raise TuningControlError(f"Artifact escapes attempt root: {path}")
    actual = file_pin(path)
    if actual["size_bytes"] != pin["size_bytes"] or actual["sha256"] != pin["sha256"].removeprefix("sha256:"):
        raise TuningControlError(f"Hash/size mismatch: {path}")
    return path


def runtime_snapshot() -> dict[str, str]:
    names = ("numpy", "pandas", "pyarrow", "scikit-learn", "scipy", "joblib", "threadpoolctl", "exchange-calendars", "psutil")
    return {"python": ".".join(map(str, sys.version_info[:3])), **{name: importlib.metadata.version(name) for name in names}}


def effective_protocol(protocol: dict[str, Any], amendment: dict[str, Any]) -> dict[str, Any]:
    """Apply only the explicitly authorized operational launch amendment.

    The original research protocol and every statistical/maximum budget remain
    immutable. This is not permission to lower RSS/time/fit limits or alter data.
    macOS exposes DISPATCH masks here: NORMAL=1, WARN=2, CRITICAL=4.
    """
    result = json.loads(json.dumps(protocol))
    policy = amendment.get("resource_launch_policy")
    if policy is None:
        return result
    expected = {"minimum_available_memory_before_launch_bytes": 7 * 1024**3,
                "require_macos_pressure_dispatch_value": 1,
                "stop_on_macos_pressure_not_normal": True}
    if policy != expected or type(policy.get("minimum_available_memory_before_launch_bytes")) is not int or type(policy.get("require_macos_pressure_dispatch_value")) is not int or policy.get("stop_on_macos_pressure_not_normal") is not True:
        raise TuningControlError("Unreviewed operational launch policy; explicit revised amendment required")
    if protocol["local_budget"]["minimum_available_memory_before_launch_bytes"] != 12 * 1024**3:
        raise TuningControlError("Launch amendment applies only to the frozen original 12 GiB gate")
    result["local_budget"].update(policy)
    return result


def planned_tasks(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for fold in protocol["chronology"]["folds"]:
        year = fold["outer_outcome_year"]
        for split in fold["inner_splits"]:
            split_id = split["split_id"]
            result.append({"task_id": f"{split_id}__prepare", "action": "prepare", "outer_year": year, "split_id": split_id, "status": "PENDING"})
            for candidate in protocol["rf"]["candidate_fit_order_within_split"]:
                result.append({"task_id": f"{split_id}__{candidate}", "action": "fit", "outer_year": year, "split_id": split_id, "candidate_id": candidate, "status": "PENDING"})
        split_id = f"{year}_outer_train"
        result.append({"task_id": f"{split_id}__prepare", "action": "prepare", "outer_year": year, "split_id": split_id, "status": "PENDING"})
        result.append({"task_id": f"{split_id}__fit", "action": "fit", "outer_year": year, "split_id": split_id, "candidate_id": None, "status": "PENDING"})
        result.append({"task_id": f"{year}__predict_outer", "action": "predict_outer", "outer_year": year, "split_id": split_id, "status": "PENDING"})
    if sum(t["action"] == "fit" for t in result) != protocol["planned_fit_counts"]["maximum_estimator_fits"]:
        raise TuningControlError("Task fit count disagrees with frozen budget")
    if sum(t["action"] == "prepare" for t in result) != protocol["planned_fit_counts"]["maximum_preprocessor_fits"]:
        raise TuningControlError("Task preprocessing count disagrees with frozen budget")
    return result


def _state_payload(state: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in state.items() if key not in {"event_sequence", "last_event_sha256"}}


def commit_state(run_root: Path, state: dict[str, Any], event_type: str, payload: dict[str, Any] | None = None) -> None:
    sequence = state.get("event_sequence", 0) + 1
    snapshot = _state_payload(state)
    event = {"schema": SCHEMA + "/event", "event_sequence": sequence,
             "previous_event_sha256": state.get("last_event_sha256"), "created_at": utc_now(),
             "event_type": event_type, "payload": payload or {}, "ledger_snapshot": snapshot,
             "ledger_snapshot_sha256": canonical_sha256(snapshot)}
    atomic_json(run_root / "events" / f"{sequence:06d}.json", event, exclusive=True)
    state["event_sequence"] = sequence
    state["last_event_sha256"] = canonical_sha256(event)
    atomic_json(run_root / "ledger.json", state)


def verify_event_chain(run_root: Path, state: dict[str, Any]) -> None:
    previous = None
    events = sorted((run_root / "events").glob("*.json"))
    if len(events) != state["event_sequence"]:
        raise TuningControlError("Event count disagrees with ledger; preserve attempt for review")
    last = None
    for index, path in enumerate(events, 1):
        event = read_json(path)
        if path.name != f"{index:06d}.json" or event["event_sequence"] != index or event["previous_event_sha256"] != previous:
            raise TuningControlError("Broken durable event chain")
        if canonical_sha256(event["ledger_snapshot"]) != event["ledger_snapshot_sha256"]:
            raise TuningControlError("Event ledger snapshot changed")
        previous = canonical_sha256(event)
        last = event
    if previous != state.get("last_event_sha256") or last is None or last["ledger_snapshot"] != _state_payload(state):
        raise TuningControlError("Ledger and last immutable event disagree")


def prepare_run(project_root: str | Path, protocol_path: str | Path, amendment_path: str | Path,
                *, code_paths: list[str | Path] | None = None, output_parent: str | Path | None = None,
                runtime: dict[str, str] | None = None, attempt_suffix: str | None = None) -> Path:
    project_root = Path(project_root).resolve()
    protocol_path, amendment_path = Path(protocol_path).resolve(), Path(amendment_path).resolve()
    protocol, amendment = read_json(protocol_path), read_json(amendment_path)
    protocol_pin = file_pin(protocol_path)
    if amendment.get("protocol_id") != protocol["protocol_id"] or amendment.get("protocol_sha256", "").removeprefix("sha256:") != protocol_pin["sha256"]:
        raise TuningControlError("Execution amendment does not bind this exact frozen protocol")
    if amendment.get("execution_authorized") is not True:
        raise TuningControlError("Explicit execution authorization absent")
    authority = amendment.get("authority", {})
    for field in ("fitting_authorized", "preprocessing_fitting_authorized", "new_financial_reads_authorized", "prediction_authorized"):
        if authority.get(field) is not True:
            raise TuningControlError(f"Execution authority absent: {field}")
    if authority.get("backtest_authorized") is not False:
        raise TuningControlError("This runner cannot authorize a backtest")
    effective_protocol(protocol, amendment)  # Validate the exact operational-only override before creating any attempt.
    reference_pins = []
    for pin in protocol["reference"]["pins"]:
        absolute = dict(pin, path=str(project_root / pin["path"]))
        verify_pin(absolute)
        reference_pins.append(absolute)
    evidence = protocol["reference"].get("private_budget_evidence")
    if evidence:
        reference_pins.append(dict(evidence, path=str(project_root / evidence["path"])))
        verify_pin(reference_pins[-1])
    if code_paths is None:
        code_paths = sorted((project_root / "src" / "dmv2").glob("*.py")) + [project_root / "scripts" / "run_rf_expanded_tuning.py", project_root / "scripts" / "verify_rf_expanded_tuning.py"]
    code_pins = [file_pin(path) for path in code_paths]
    matching_runtime = runtime or runtime_snapshot()
    for name, expected in protocol.get("runtime_snapshot_for_future_matching", {}).items():
        if matching_runtime.get(name) != expected:
            raise TuningControlError(f"Runtime differs from frozen protocol: {name}")
    identity = {"schema": SCHEMA + "/identity", "project_root": str(project_root),
                "protocol_pin": protocol_pin, "amendment_pin": file_pin(amendment_path),
                "reference_pins": reference_pins, "code_pins": code_pins, "runtime": matching_runtime,
                "protocol_id": protocol["protocol_id"], "resource_launch_policy": amendment.get("resource_launch_policy"), "tasks": planned_tasks(protocol)}
    digest = canonical_sha256(identity)
    suffix = attempt_suffix or dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    parent = Path(output_parent or project_root / "outputs" / "rf_expanded_tuning").resolve()
    run_root = parent / f"{suffix}-{digest.split(':')[1][:12]}"
    parent.mkdir(parents=True, exist_ok=True)
    registry_path = parent / "execution_registry" / (identity["amendment_pin"]["sha256"] + ".json")
    with (parent / "preparation.lock").open("a+") as preparation_lock:
        fcntl.flock(preparation_lock, fcntl.LOCK_EX)
        if registry_path.exists():
            previous = read_json(registry_path)
            raise TuningControlError("This execution amendment already owns an attempt; use explicit resume, not a fresh fit counter: " + previous["run_root"])
        if run_root.exists():
            raise TuningControlError("New attempt root already exists; use status/resume explicitly")
        run_root.mkdir(parents=True, exist_ok=False)
        atomic_json(registry_path, {"schema": SCHEMA + "/execution-registry", "protocol_pin": protocol_pin,
                    "amendment_pin": identity["amendment_pin"], "run_root": str(run_root),
                    "registered_at": utc_now()}, exclusive=True)
        fcntl.flock(preparation_lock, fcntl.LOCK_UN)
    identity["identity_sha256"] = digest
    identity["run_root"] = str(run_root)
    identity["created_at"] = utc_now()
    atomic_json(run_root / "build_identity.json", identity, exclusive=True)
    auth = {"protocol_id": protocol["protocol_id"], "run_root": str(run_root), **authority,
            "execution_amendment_pin": identity["amendment_pin"]}
    atomic_json(run_root / "authorization.json", auth, exclusive=True)
    state = {"schema": SCHEMA, "run_root": str(run_root), "build_identity_pin": file_pin(run_root / "build_identity.json"),
             "authorization_pin": file_pin(run_root / "authorization.json"), "status": "PREPARED", "created_at": utc_now(),
             "updated_at": utc_now(), "initiated_estimator_fits": 0, "initiated_preprocessor_fits": 0,
             "reserved_source_bytes": 0, "active_seconds": 0.0, "verification_active_seconds": 0.0,
             "tasks": planned_tasks(protocol), "winners": {}, "running_task_id": None, "failure": None,
             "resource_peaks": {"process_tree_rss_bytes": 0, "new_output_bytes": 0}}
    commit_state(run_root, state, "PREPARED")
    return run_root


def verify_attempt(run_root: str | Path, *, check_runtime: bool = True) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    run_root = Path(run_root).resolve()
    state = read_json(run_root / "ledger.json")
    verify_event_chain(run_root, state)
    identity = read_json(verify_pin(state["build_identity_pin"], within=run_root))
    verify_pin(state["authorization_pin"], within=run_root)
    for pin in (identity["protocol_pin"], identity["amendment_pin"], *identity["reference_pins"], *identity["code_pins"]):
        verify_pin(pin)
    if check_runtime and runtime_snapshot() != identity["runtime"]:
        raise TuningControlError("Runtime changed; explicit amendment required")
    protocol = effective_protocol(read_json(identity["protocol_pin"]["path"]), read_json(identity["amendment_pin"]["path"]))
    fit_count = sum(t.get("initiated", False) for t in state["tasks"] if t["action"] == "fit")
    prep_count = sum(t.get("initiated", False) for t in state["tasks"] if t["action"] == "prepare")
    if fit_count != state["initiated_estimator_fits"] or prep_count != state["initiated_preprocessor_fits"]:
        raise TuningControlError("Durable initiation counters disagree")
    if fit_count > protocol["planned_fit_counts"]["maximum_estimator_fits"] or prep_count > protocol["planned_fit_counts"]["maximum_preprocessor_fits"]:
        raise TuningControlError("Fit counter exceeded")
    for task in state["tasks"]:
        for name in ("plan_receipt_pin", "plan_path", "request_pin", "reservation_pin"):
            if task.get(name):
                verify_pin(task[name], within=run_root)
        if task["status"] == "COMPLETE":
            receipt = read_json(verify_pin(task["receipt_pin"], within=run_root))
            validate_receipt(run_root, task, receipt)
    for pin in state["winners"].values():
        verify_pin(pin, within=run_root)
    return state, identity, protocol


def select_winner(protocol: dict[str, Any], outer_year: int, receipts: list[dict[str, Any]]) -> dict[str, Any]:
    fold = next(f for f in protocol["chronology"]["folds"] if f["outer_outcome_year"] == outer_year)
    split_ids = {s["split_id"] for s in fold["inner_splits"]}
    candidates = {c["candidate_id"]: c for c in protocol["rf"]["candidates"]}
    by_pair = {}
    common_keys: dict[str, str] = {}
    for receipt in receipts:
        key = (receipt["split_id"], receipt["candidate_id"])
        if key in by_pair or key[0] not in split_ids or key[1] not in candidates:
            raise TuningControlError("Duplicate or unexpected inner receipt")
        by_pair[key] = receipt
        values = receipt["monthly_log_losses"]
        validation_year = int(key[0].rsplit("_", 1)[1])
        expected = {f"{validation_year}-{month:02d}" for month in range(1, 13)}
        if len(values) != 12 or {v["outcome_month"] for v in values} != expected:
            raise TuningControlError("All twelve validation months required")
        for value in values:
            loss = value["log_loss"]
            if not isinstance(loss, (int, float)) or not math.isfinite(loss) or loss < 0:
                raise TuningControlError("Nonfinite or invalid monthly log loss")
            score_hash = value["score_key_sha256"]
            if not isinstance(score_hash, str) or not score_hash:
                raise TuningControlError("Missing scoring-key hash")
            month = value["outcome_month"]
            if month in common_keys and common_keys[month] != score_hash:
                raise TuningControlError("Candidates scored on different observed-label keys")
            common_keys[month] = score_hash
    if set(by_pair) != {(s, c) for s in split_ids for c in candidates}:
        raise TuningControlError("Cannot choose from an incomplete candidate grid")
    scores, effective = {}, {}
    for candidate in candidates:
        values = [v["log_loss"] for split in sorted(split_ids) for v in by_pair[(split, candidate)]["monthly_log_losses"]]
        scores[candidate] = math.fsum(values) / 36
        effective[candidate] = sum(by_pair[(split, candidate)]["effective_feature_count"] for split in split_ids)
    minimum = min(scores.values())
    ties = [key for key, score in scores.items() if score <= minimum + protocol["selection"]["numerical_tie_absolute_tolerance"]]
    def tie_key(key: str) -> tuple:
        candidate = candidates[key]
        return (not candidate["reference_statistical_parameters"], candidate["max_depth"], -candidate["min_samples_leaf"], effective[key], key)
    chosen = min(ties, key=tie_key)
    return {"status": "SEALED", "protocol_id": protocol["protocol_id"], "outer_year": outer_year,
            "candidate_id": chosen, "selection_metric": "equal_month_mean_log_loss", "candidate_scores": scores,
            "nominal_split_feature_count_sums": effective, "tie_candidates": sorted(ties),
            "observed_scoring_key_hashes": common_keys, "scored_months_per_candidate": 36, "sealed_at": utc_now()}


def validate_receipt(run_root: Path, task: dict[str, Any], receipt: dict[str, Any]) -> None:
    if receipt.get("status") != "COMPLETE":
        raise TuningControlError("Worker did not complete")
    for key in ("action", "outer_year", "split_id"):
        if receipt.get(key) != task[key]:
            raise TuningControlError(f"Worker receipt/task mismatch: {key}")
    if task["action"] == "fit" and receipt.get("candidate_id") != task["candidate_id"]:
        raise TuningControlError("Worker candidate differs from durable initiation")
    for pin in receipt.get("artifacts", []):
        verify_pin(pin, within=run_root)
    required = "prepare_manifest" if task["action"] == "prepare" else "fit_manifest" if task["action"] == "fit" else "prediction_manifest"
    if required not in receipt:
        raise TuningControlError(f"Worker receipt missing {required}")
    verify_pin(receipt[required], within=run_root)


def available_memory_snapshot() -> dict[str, Any]:
    result = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True)
    raw = result.stdout
    page_match = re.search(r"page size of\s+(\d+)\s+bytes", raw)
    if not page_match:
        raise TuningControlError("Cannot parse vm_stat page size")
    page_bytes = int(page_match.group(1))
    fields = {}
    for name in ("Pages free", "Pages inactive", "Pages speculative"):
        match = re.search(r"^" + re.escape(name) + r":\s+(\d+)\.?$", raw, re.MULTILINE)
        if not match:
            raise TuningControlError(f"Cannot parse vm_stat {name}")
        fields[name] = int(match.group(1))
    return {"available_bytes": page_bytes * sum(fields.values()), "page_bytes": page_bytes, "pages": fields, "raw_vm_stat": raw}


def macos_pressure_snapshot() -> dict[str, Any]:
    result = subprocess.run(["/usr/sbin/sysctl", "-n", "kern.memorystatus_vm_pressure_level"],
                            capture_output=True, text=True, check=True)
    raw = result.stdout.strip()
    if not raw.isdecimal():
        raise TuningControlError("Cannot parse macOS memory-pressure DISPATCH mask")
    value = int(raw)
    return {"dispatch_value": value, "normal_dispatch_value": 1, "raw_sysctl": raw,
            "meaning": {1: "NORMAL", 2: "WARN", 4: "CRITICAL"}.get(value, "UNKNOWN_NON_NORMAL")}


def process_tree_rss_bytes() -> int:
    import psutil
    current = psutil.Process(os.getpid())
    processes = [current, *current.children(recursive=True)]
    total = 0
    for process in processes:
        try:
            total += process.memory_info().rss
        except psutil.NoSuchProcess:
            if process.pid == current.pid:
                raise
        except psutil.AccessDenied:
            raise TuningControlError("Cannot observe descendant RSS; fail closed")
    return total


def output_bytes(run_root: Path) -> int:
    return sum(path.stat().st_size for path in run_root.rglob("*") if path.is_file())


def _kill_process_tree(process: subprocess.Popen, *, grace_seconds: float = 5.0) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()


@contextlib.contextmanager
def attempt_lock(run_root: Path):
    with (run_root / "controller.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise TuningControlError("Another controller already owns this attempt") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def request_pause(run_root: str | Path, *, immediate: bool = False) -> dict[str, Any]:
    run_root = Path(run_root).resolve()
    if not (run_root / "ledger.json").exists():
        raise TuningControlError("Not a prepared tuning attempt")
    request = {"requested_at": utc_now(), "immediate": immediate,
               "meaning": "interrupt_current_action_no_retry" if immediate else "finish_current_action_checkpoint_and_release_RAM"}
    atomic_json(run_root / ("interrupt.request.json" if immediate else "pause.request.json"), request)
    return request


def _find_task(state: dict[str, Any], split_id: str, action: str) -> dict[str, Any]:
    return next(t for t in state["tasks"] if t["split_id"] == split_id and t["action"] == action)


def _read_completed_receipt(task: dict[str, Any]) -> dict[str, Any]:
    if task["status"] != "COMPLETE":
        raise TuningControlError("Required predecessor is not complete")
    return read_json(task["receipt_pin"]["path"])


def _seal_winner(run_root: Path, state: dict[str, Any], protocol: dict[str, Any], year: int) -> dict[str, Any]:
    if str(year) in state["winners"]:
        return state["winners"][str(year)]
    tasks = [task for task in state["tasks"] if task["outer_year"] == year and task["action"] == "fit" and "_validate_" in task["split_id"]]
    receipts = [_read_completed_receipt(task) for task in tasks]
    winner = select_winner(protocol, year, receipts)
    winner["input_fit_receipt_pins"] = [task["receipt_pin"] for task in tasks]
    path = run_root / "winners" / f"{year}.json"
    atomic_json(path, winner, exclusive=True)
    state["winners"][str(year)] = file_pin(path)
    commit_state(run_root, state, "WINNER_SEALED", {"outer_year": year, "winner_pin": state["winners"][str(year)]})
    return state["winners"][str(year)]


def _base_request(run_root: Path, identity: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    request = {"root": identity["project_root"], "run_root": str(run_root), "protocol_path": identity["protocol_pin"]["path"],
               "authorization_path": str(run_root / "authorization.json"), "action": task["action"],
               "outer_year": task["outer_year"], "split_id": task["split_id"],
               "output_dir": str(run_root / "artifacts" / task["task_id"])}
    if task.get("candidate_id"):
        request["candidate_id"] = task["candidate_id"]
    return request


def _request_for_task(run_root: Path, state: dict[str, Any], identity: dict[str, Any], protocol: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    if "_outer_train" in task["split_id"]:
        winner_pin = _seal_winner(run_root, state, protocol, task["outer_year"])
        winner = read_json(winner_pin["path"])
        task["candidate_id"] = winner["candidate_id"]
    request = _base_request(run_root, identity, task)
    if "_outer_train" in task["split_id"]:
        request["winner_path"] = state["winners"][str(task["outer_year"])]
    if task["action"] in {"fit", "predict_outer"}:
        prepare = _read_completed_receipt(_find_task(state, task["split_id"], "prepare"))
        request["prepare_manifest"] = prepare["prepare_manifest"]
    if task["action"] == "predict_outer":
        fitted = _read_completed_receipt(_find_task(state, task["split_id"], "fit"))
        request["fit_manifest"] = fitted["fit_manifest"]
    return request


def _resource_preflight(run_root: Path, state: dict[str, Any], budget: dict[str, Any], memory_probe: Callable[[], dict[str, Any]], pressure_probe: Callable[[], dict[str, Any]]) -> bool:
    memory = memory_probe()
    disk = shutil.disk_usage(run_root)
    pressure = None
    if budget.get("stop_on_macos_pressure_not_normal"):
        try:
            pressure = pressure_probe()
        except Exception as exc:
            pressure = {"dispatch_value": None, "error": type(exc).__name__ + ": " + str(exc)}
    state["last_preflight"] = {"checked_at": utc_now(), "memory": memory, "free_disk_bytes": disk.free, "macos_memory_pressure": pressure}
    if pressure is not None and pressure.get("dispatch_value") != budget["require_macos_pressure_dispatch_value"]:
        state["status"] = "WAITING_FOR_MEMORY_PRESSURE"
        commit_state(run_root, state, "WAITING_FOR_MEMORY_PRESSURE", state["last_preflight"])
        return False
    if memory["available_bytes"] < budget["minimum_available_memory_before_launch_bytes"]:
        state["status"] = "WAITING_FOR_MEMORY"
        commit_state(run_root, state, "WAITING_FOR_MEMORY", state["last_preflight"])
        return False
    if disk.free < budget["minimum_free_disk_before_launch_bytes"]:
        state["status"] = "WAITING_FOR_DISK"
        commit_state(run_root, state, "WAITING_FOR_DISK", state["last_preflight"])
        return False
    return True


def _watch_child(run_root: Path, state: dict[str, Any], protocol: dict[str, Any], command: list[str],
                 log_path: Path, *, task_id: str, phase: str,
                 rss_probe: Callable[[], int] = process_tree_rss_bytes,
                 pressure_probe: Callable[[], dict[str, Any]] = macos_pressure_snapshot,
                 poll_seconds: float | None = None) -> tuple[int, str | None]:
    budget = protocol["local_budget"]
    env = dict(os.environ, **budget["numeric_thread_environment"])
    identity = read_json(state["build_identity_pin"]["path"])
    project_src = str(Path(identity["project_root"]) / "src")
    prior_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = project_src + (os.pathsep + prior_pythonpath if prior_pythonpath else "")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    before = time.monotonic()
    last_tick = before
    last_commit = before
    reason = None
    poll = budget["monitor_poll_seconds"] if poll_seconds is None else poll_seconds
    with log_path.open("x") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        state["worker_pid"] = process.pid
        commit_state(run_root, state, "WORKER_STARTED", {"task_id": task_id, "phase": phase, "worker_pid": process.pid})
        try:
            while True:
                now = time.monotonic()
                state["active_seconds"] += now - last_tick
                last_tick = now
                rss = rss_probe()
                size = output_bytes(run_root)
                state["resource_peaks"]["process_tree_rss_bytes"] = max(state["resource_peaks"]["process_tree_rss_bytes"], rss)
                state["resource_peaks"]["new_output_bytes"] = max(state["resource_peaks"]["new_output_bytes"], size)
                pressure = None
                if budget.get("stop_on_macos_pressure_not_normal"):
                    try:
                        pressure = pressure_probe()
                    except Exception as exc:
                        pressure = {"dispatch_value": None, "error": type(exc).__name__ + ": " + str(exc)}
                state["last_resource_sample"] = {"sampled_at": utc_now(), "rss_bytes": rss, "new_output_bytes": size,
                                                  "rss_warning": rss >= budget["process_tree_rss_warning_bytes"],
                                                  "macos_memory_pressure": pressure}
                if pressure is not None and pressure.get("dispatch_value") != budget["require_macos_pressure_dispatch_value"]:
                    reason = "MACOS_MEMORY_PRESSURE_NOT_NORMAL"
                elif (run_root / "interrupt.request.json").exists():
                    reason = "EXPLICIT_IMMEDIATE_INTERRUPT"
                elif rss >= budget["process_tree_rss_stop_bytes"]:
                    reason = "PROCESS_TREE_RSS_LIMIT"
                elif size >= budget["new_output_stop_bytes"]:
                    reason = "NEW_OUTPUT_LIMIT"
                elif state["active_seconds"] >= budget["total_preprocessing_fit_prediction_scoring_wall_seconds"]:
                    reason = "CUMULATIVE_ACTIVE_WALL_LIMIT"
                elif now - before >= budget["model_wall_seconds_per_fit"]:
                    reason = "CHILD_ACTION_WALL_LIMIT"
                if reason:
                    _kill_process_tree(process)
                    break
                code = process.poll()
                if code is not None:
                    break
                if now - last_commit >= 1:
                    atomic_json(run_root / "resource_checkpoint.json", {"task_id": task_id, "phase": phase,
                        "active_seconds": state["active_seconds"], "resource_peaks": state["resource_peaks"],
                        "last_resource_sample": state["last_resource_sample"]})
                    last_commit = now
                time.sleep(poll)
        except BaseException:
            _kill_process_tree(process)
            state["active_seconds"] += time.monotonic() - last_tick
            state["worker_pid"] = None
            commit_state(run_root, state, "WORKER_SUPERVISOR_INTERRUPTED", {"task_id": task_id})
            raise
        state["active_seconds"] += time.monotonic() - last_tick
        state["worker_pid"] = None
        commit_state(run_root, state, "WORKER_EXITED", {"task_id": task_id, "phase": phase, "exit_code": process.returncode, "stop_reason": reason})
        return process.returncode, reason


def _run_plan(run_root: Path, state: dict[str, Any], identity: dict[str, Any], protocol: dict[str, Any],
              task: dict[str, Any], request: dict[str, Any], worker_prefix: list[str], rss_probe: Callable[[], int], pressure_probe: Callable[[], dict[str, Any]], poll_seconds: float | None) -> None:
    if task.get("plan_path"):
        return
    planning = dict(request, action="plan", planned_action=task["action"], output_dir=str(run_root / "plans" / task["task_id"]))
    request_path = run_root / "requests" / f"{task['task_id']}__plan.json"
    receipt_path = run_root / "receipts" / f"{task['task_id']}__plan.json"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(request_path, planning, exclusive=True)
    code, reason = _watch_child(run_root, state, protocol, worker_prefix + ["--request", str(request_path), "--receipt", str(receipt_path)],
                                run_root / "logs" / f"{task['task_id']}__plan.log", task_id=task["task_id"], phase="metadata_plan", rss_probe=rss_probe, pressure_probe=pressure_probe, poll_seconds=poll_seconds)
    if code != 0 or reason:
        if reason:
            state["status"] = "FAILED_INTERRUPTED" if reason == "EXPLICIT_IMMEDIATE_INTERRUPT" else "STOPPED_BUDGET"
        raise TuningControlError(f"Metadata planning failed: exit={code} stop={reason}")
    receipt = read_json(receipt_path)
    if receipt.get("status") != "COMPLETE" or "plan_path" not in receipt:
        raise TuningControlError("Incomplete metadata plan")
    plan_path = verify_pin(receipt["plan_path"], within=run_root)
    plan = read_json(plan_path)
    projected = receipt["plan"]["projected_source_read_bytes"]
    if projected != plan["projected_source_read_bytes"] or not isinstance(projected, int) or projected < 0:
        raise TuningControlError("Invalid projected source read budget")
    task["plan_path"] = receipt["plan_path"]
    task["plan_receipt_pin"] = file_pin(receipt_path)
    task["projected_source_read_bytes"] = projected
    commit_state(run_root, state, "SOURCE_READ_PLAN_SEALED", {"task_id": task["task_id"], "projected_source_read_bytes": projected})


def _run_attempt_locked(run_root: str | Path, *, resume: bool = False, worker_prefix: list[str] | None = None,
                memory_probe: Callable[[], dict[str, Any]] = available_memory_snapshot,
                rss_probe: Callable[[], int] = process_tree_rss_bytes,
                pressure_probe: Callable[[], dict[str, Any]] = macos_pressure_snapshot,
                check_runtime: bool = True, poll_seconds: float | None = None,
                max_completed_actions: int | None = None,
                session_started: float, session_base_active: float) -> dict[str, Any]:
    run_root = Path(run_root).resolve()
    state, identity, protocol = verify_attempt(run_root, check_runtime=check_runtime)
    if state["status"] in TERMINAL_FAILURES:
        raise TuningControlError("Failed/interrupted/budget-stopped attempt is immutable; explicit amendment needed, no automatic retry")
    if any(t["status"] == "RUNNING" for t in state["tasks"]):
        state["status"] = "FAILED_INTERRUPTED"
        state["failure"] = {"reason": "ORPHAN_RUNNING_TASK_NO_AUTOMATIC_RETRY", "recorded_at": utc_now()}
        commit_state(run_root, state, "ORPHAN_RUNNING_TASK")
        raise TuningControlError("Incomplete initiated task preserved; no automatic retry")
    if state["status"] == "COMPLETE_PENDING_INDEPENDENT_VERIFICATION":
        return state
    if state["status"] in WAITING and not resume:
        raise TuningControlError("Use explicit resume for this paused/waiting attempt")
    if resume:
        pause_file = run_root / "pause.request.json"
        if pause_file.exists():
            archived = run_root / "pause_requests" / f"{state['event_sequence']:06d}.json"
            archived.parent.mkdir(parents=True, exist_ok=True)
            os.replace(pause_file, archived)
        commit_state(run_root, state, "EXPLICIT_RESUME")
    if (run_root / "interrupt.request.json").exists():
        raise TuningControlError("Immediate interrupt request remains; explicit amendment required")
    state["status"] = "RUNNING"
    commit_state(run_root, state, "RUN_STARTED")
    prefix = worker_prefix or [sys.executable, "-m", "dmv2.rf_tuning_fit"]
    budget = protocol["local_budget"]
    completed_this_session = 0
    current_task = None
    try:
        for task in state["tasks"]:
            current_task = task
            if task["status"] == "COMPLETE":
                continue
            if task.get("initiated"):
                raise TuningControlError("An initiated incomplete task cannot be retried")
            if (run_root / "pause.request.json").exists() or (max_completed_actions is not None and completed_this_session >= max_completed_actions):
                state["status"] = "PAUSED"
                commit_state(run_root, state, "GRACEFUL_PAUSE")
                return state
            if budget.get("stop_on_macos_pressure_not_normal"):
                try:
                    pressure = pressure_probe()
                except Exception as exc:
                    pressure = {"dispatch_value": None, "error": type(exc).__name__ + ": " + str(exc)}
                if pressure.get("dispatch_value") != budget["require_macos_pressure_dispatch_value"]:
                    _resource_preflight(run_root, state, budget, memory_probe, lambda: pressure)
                    return state
            request = _request_for_task(run_root, state, identity, protocol, task)
            if task["action"] in {"prepare", "predict_outer"}:
                _run_plan(run_root, state, identity, protocol, task, request, prefix, rss_probe, pressure_probe, poll_seconds)
            if (run_root / "pause.request.json").exists():
                state["status"] = "PAUSED"
                commit_state(run_root, state, "GRACEFUL_PAUSE_AFTER_METADATA_PLAN")
                return state
            state["active_seconds"] = max(state["active_seconds"], session_base_active + time.monotonic() - session_started)
            if not _resource_preflight(run_root, state, budget, memory_probe, pressure_probe):
                return state
            if state["active_seconds"] >= budget["total_preprocessing_fit_prediction_scoring_wall_seconds"]:
                raise TuningBudgetError("Cumulative active budget exhausted before launch")
            if output_bytes(run_root) >= budget["new_output_stop_bytes"]:
                raise TuningBudgetError("Output budget exhausted before launch")
            reserve = task.get("projected_source_read_bytes", 0)
            if state["reserved_source_bytes"] + reserve > budget["total_projected_compressed_source_read_stop_bytes"]:
                raise TuningBudgetError("Projected repeated-decode source read budget would be exceeded")
            if task["action"] == "fit":
                if state["initiated_estimator_fits"] >= protocol["planned_fit_counts"]["maximum_estimator_fits"]:
                    raise TuningBudgetError("Estimator fit budget exhausted")
                state["initiated_estimator_fits"] += 1
            if task["action"] == "prepare":
                if state["initiated_preprocessor_fits"] >= protocol["planned_fit_counts"]["maximum_preprocessor_fits"]:
                    raise TuningBudgetError("Preprocessor fit budget exhausted")
                state["initiated_preprocessor_fits"] += 1
            if task["action"] in {"prepare", "predict_outer"}:
                plan = read_json(task["plan_path"]["path"])
                reservation_path = run_root / "reservations" / f"{task['task_id']}.json"
                atomic_json(reservation_path, {"execution_authorized": True, "projected_source_read_bytes": reserve,
                                              "plan_sha256": canonical_sha256(plan), "task_id": task["task_id"]}, exclusive=True)
                task["reservation_pin"] = file_pin(reservation_path)
                request["plan_path"] = task["plan_path"]
                request["reservation_path"] = str(reservation_path)
            state["reserved_source_bytes"] += reserve
            request_path = run_root / "requests" / f"{task['task_id']}.json"
            receipt_path = run_root / "receipts" / f"{task['task_id']}.json"
            atomic_json(request_path, request, exclusive=True)
            task.update(status="RUNNING", initiated=True, started_at=utc_now(), request_pin=file_pin(request_path))
            state["running_task_id"] = task["task_id"]
            commit_state(run_root, state, "TASK_INITIATED", {"task_id": task["task_id"], "action": task["action"], "reserved_source_bytes": reserve})
            code, reason = _watch_child(run_root, state, protocol, prefix + ["--request", str(request_path), "--receipt", str(receipt_path)],
                                        run_root / "logs" / f"{task['task_id']}.log", task_id=task["task_id"], phase=task["action"], rss_probe=rss_probe, pressure_probe=pressure_probe, poll_seconds=poll_seconds)
            if code != 0 or reason:
                state["status"] = "FAILED_INTERRUPTED" if reason == "EXPLICIT_IMMEDIATE_INTERRUPT" else "STOPPED_BUDGET" if reason else "FAILED"
                raise TuningControlError(f"Worker failed; no retry: exit={code}, stop={reason}")
            receipt = read_json(receipt_path)
            validate_receipt(run_root, task, receipt)
            task.update(status="COMPLETE", completed_at=utc_now(), receipt_pin=file_pin(receipt_path))
            state["running_task_id"] = None
            commit_state(run_root, state, "TASK_COMPLETED", {"task_id": task["task_id"], "receipt_pin": task["receipt_pin"]})
            completed_this_session += 1
        state["status"] = "COMPLETE_PENDING_INDEPENDENT_VERIFICATION"
        state["completed_at"] = utc_now()
        commit_state(run_root, state, "ALL_SCHEDULED_ACTIONS_COMPLETE")
        return state
    except BaseException as exc:
        if current_task is not None and current_task.get("status") == "RUNNING":
            current_task["status"] = "FAILED"
        if state["status"] not in TERMINAL_FAILURES:
            state["status"] = "FAILED_INTERRUPTED" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "STOPPED_BUDGET" if isinstance(exc, TuningBudgetError) else "FAILED"
        state["failure"] = {"error_type": type(exc).__name__, "message": str(exc), "recorded_at": utc_now(),
                            "task_id": current_task.get("task_id") if current_task else None}
        state["running_task_id"] = None
        commit_state(run_root, state, "ATTEMPT_STOPPED", state["failure"])
        raise

def run_attempt(run_root: str | Path, *, resume: bool = False, worker_prefix: list[str] | None = None,
                memory_probe: Callable[[], dict[str, Any]] = available_memory_snapshot,
                rss_probe: Callable[[], int] = process_tree_rss_bytes,
                pressure_probe: Callable[[], dict[str, Any]] = macos_pressure_snapshot,
                check_runtime: bool = True, poll_seconds: float | None = None,
                max_completed_actions: int | None = None) -> dict[str, Any]:
    """Charge full active controller wall time once; paused hours are excluded."""
    run_root = Path(run_root).resolve()
    with attempt_lock(run_root):
        started = time.monotonic()
        base = read_json(run_root / "ledger.json")["active_seconds"]
        result = None
        try:
            result = _run_attempt_locked(run_root, resume=resume, worker_prefix=worker_prefix,
                memory_probe=memory_probe, rss_probe=rss_probe, pressure_probe=pressure_probe, check_runtime=check_runtime,
                poll_seconds=poll_seconds, max_completed_actions=max_completed_actions,
                session_started=started, session_base_active=base)
            return result
        finally:
            # Do not repair or append to an invalid event chain on a refusal.
            latest = read_json(run_root / "ledger.json")
            try:
                verify_event_chain(run_root, latest)
            except TuningControlError:
                pass
            else:
                latest["active_seconds"] = max(latest["active_seconds"], base + time.monotonic() - started)
                identity = read_json(latest["build_identity_pin"]["path"])
                protocol = effective_protocol(read_json(identity["protocol_pin"]["path"]), read_json(identity["amendment_pin"]["path"]))
                if latest["active_seconds"] >= protocol["local_budget"]["total_preprocessing_fit_prediction_scoring_wall_seconds"] and latest["status"] not in TERMINAL_FAILURES:
                    latest["status"] = "STOPPED_BUDGET"
                    latest["failure"] = {"reason": "CUMULATIVE_ACTIVE_CONTROLLER_WALL_LIMIT", "recorded_at": utc_now()}
                commit_state(run_root, latest, "ACTIVE_SESSION_ACCOUNTED", {"session_wall_seconds": time.monotonic() - started})
                if result is not None:
                    result.clear()
                    result.update(latest)


def status(run_root: str | Path) -> dict[str, Any]:
    run_root = Path(run_root).resolve()
    state = read_json(run_root / "ledger.json")
    checkpoint = run_root / "resource_checkpoint.json"
    if state["status"] == "RUNNING" and checkpoint.exists():
        monitor = read_json(checkpoint)
        state["active_seconds"] = max(state["active_seconds"], monitor["active_seconds"])
        state["resource_peaks"] = monitor["resource_peaks"]
    return {"run_root": str(run_root), "status": state["status"], "initiated_estimator_fits": state["initiated_estimator_fits"],
            "completed_estimator_fits": sum(t["action"] == "fit" and t["status"] == "COMPLETE" for t in state["tasks"]),
            "initiated_preprocessor_fits": state["initiated_preprocessor_fits"], "active_seconds": state["active_seconds"],
            "reserved_source_bytes": state["reserved_source_bytes"], "running_task_id": state["running_task_id"],
            "last_preflight": state.get("last_preflight"), "resource_peaks": state["resource_peaks"], "failure": state.get("failure")}
