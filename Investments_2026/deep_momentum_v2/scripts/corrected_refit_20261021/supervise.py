"""Run one pinned local phase with process-tree, pause and cumulative limits."""
from pathlib import Path
import argparse
import fcntl
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from control import ROOT, check, need, pin, read, replace_json, save_new


def resources():
    pressure = int(subprocess.check_output(
        ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"], text=True).strip())
    raw = subprocess.check_output(["vm_stat"], text=True)
    size = re.search(r"page size of (\d+) bytes", raw)
    need(size is not None, "Cannot read macOS page size")
    pages = {}
    for name in ("Pages free", "Pages inactive", "Pages speculative"):
        match = re.search(re.escape(name) + r":\s+(\d+)\.", raw)
        need(match is not None, "Cannot read available-memory proxy")
        pages[name] = int(match.group(1))
    return {"macos_memory_pressure": pressure,
        "available_proxy_bytes": int(size.group(1)) * sum(pages.values()),
        "available_proxy_definition": "page_size*(free+inactive+speculative)",
        "vm_stat": raw, "free_disk_bytes": shutil.disk_usage(ROOT).free}


def tree_rss():
    lines = subprocess.check_output(["ps", "-axo", "pid=,ppid=,rss="], text=True)
    rows = [tuple(map(int, line.split())) for line in lines.splitlines() if line.strip()]
    ids = {os.getpid()}
    while True:
        enlarged = ids | {pid for pid, parent, _ in rows if parent in ids}
        if enlarged == ids:
            return sum(rss * 1024 for pid, _, rss in rows if pid in ids)
        ids = enlarged


def run(launch_path, expected_hash):
    launch_path = Path(launch_path).resolve()
    need(launch_path.is_relative_to(ROOT) and pin(launch_path)["sha256"] == expected_hash,
         "Launch manifest differs")
    launch = read(launch_path)
    need(launch["schema"] == "dmv2-corrected-refit-phase-launch/v1", "Unknown phase launch")
    protocol_path = check(launch["protocol_pin"])
    protocol = read(protocol_path)
    need(protocol["status"] == "SEALED_BEFORE_FINANCIAL_READ", "Unsealed protocol")
    for item in launch["code_pins"]:
        check(item)
    phase = launch["phase"]
    need(isinstance(phase, str) and phase.replace("_", "").isalnum(), "Invalid phase")
    category = launch["time_category"]
    limits = protocol["limits"]
    need(category in limits["active_seconds_by_category"], "Unknown time budget")
    need(0 < launch["maximum_seconds"] <= limits["active_seconds_by_category"][category],
         "Phase wall time exceeds category budget")
    command = launch["command"]
    need(isinstance(command, list) and len(command) >= 2 and all(isinstance(x, str) for x in command),
         "Command must be an argument vector")
    need(command[0] == protocol["python_path"], "Unexpected runtime")
    need(Path(command[1]).resolve().is_relative_to(ROOT), "Worker outside sealed staging")
    need(any(Path(x["path"]) == Path(command[1]).resolve() for x in launch["code_pins"]),
         "Worker not pinned")
    out = ROOT / "run/control"
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".process.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status_path, log_path = out / (phase + ".json"), out / (phase + ".log")
        need(not status_path.exists() and not log_path.exists(), "Preserve the existing phase attempt")
        clock_path = out / "active_time.json"
        clock = read(clock_path) if clock_path.exists() else {
            "schema": "dmv2-corrected-refit-active-time/v1",
            "consumed_seconds": {key: 0.0 for key in limits["active_seconds_by_category"]}}
        remaining = limits["active_seconds_by_category"][category] - clock["consumed_seconds"][category]
        need(remaining > 0, "Cumulative phase time exhausted")
        maximum = min(remaining, launch["maximum_seconds"])
        before = resources()
        state = {"phase": phase, "launch_pin": pin(launch_path), "resources_before": before,
            "peak_process_tree_rss_bytes": 0, "active_seconds": 0, "time_category": category}
        if (ROOT / "PAUSE_REQUESTED.json").exists():
            save_new(status_path, {**state, "status": "PAUSED_BEFORE_DISPATCH"})
            print(json.dumps({"status": "PAUSED_BEFORE_DISPATCH", "phase": phase}), flush=True)
            return
        waiting = ("WAITING_FOR_MEMORY_PRESSURE" if before["macos_memory_pressure"] != 1 else
            "WAITING_FOR_MEMORY" if before["available_proxy_bytes"] < limits["available_proxy_minimum_bytes"] else
            "WAITING_FOR_DISK" if before["free_disk_bytes"] < limits["free_disk_minimum_bytes"] else None)
        if waiting:
            save_new(status_path, {**state, "status": waiting})
            print(json.dumps({"status": waiting, "phase": phase, "resources": before}), flush=True)
            return
        start, child = time.monotonic(), None
        previous_signals = {signum: signal.getsignal(signum) for signum in
            (signal.SIGTERM, signal.SIGHUP)}
        def terminate(signum, frame):
            raise SystemExit("Supervisor received signal " + str(signum))
        for signum in previous_signals:
            signal.signal(signum, terminate)
        try:
            with log_path.open("x", buffering=1) as stream:
                env = dict(os.environ, **{key: "1" for key in (
                    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")}, PYTHONDONTWRITEBYTECODE="1",
                    DMV2_GUARD_LAUNCH_PATH=str(launch_path), DMV2_GUARD_LAUNCH_SHA256=expected_hash)
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                    stderr=subprocess.STDOUT, start_new_session=True)
                while child.poll() is None:
                    current = tree_rss()
                    state["peak_process_tree_rss_bytes"] = max(current, state["peak_process_tree_rss_bytes"])
                    state["active_seconds"] = time.monotonic() - start
                    pressure = int(subprocess.check_output(
                        ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"], text=True).strip())
                    state.update(status="RUNNING", pid=child.pid, process_tree_rss_bytes=current,
                        macos_memory_pressure=pressure,
                        memory_warning=current >= limits["rss_warning_bytes"],
                        pause_requested=(ROOT / "PAUSE_REQUESTED.json").exists())
                    replace_json(status_path, state)
                    need(current <= limits["rss_stop_bytes"], "Process-tree RSS ceiling exceeded")
                    need(pressure == 1, "Memory pressure is not normal")
                    need(state["active_seconds"] <= maximum, "Phase or cumulative wall ceiling exceeded")
                    need(shutil.disk_usage(ROOT).free >= limits["free_disk_minimum_bytes"], "Disk ceiling")
                    size = sum(p.stat().st_size for p in ROOT.rglob("*") if p.is_file())
                    need(size <= limits["output_maximum_bytes"], "Output ceiling")
                    time.sleep(1)
                need(child.returncode == 0, "Child exit " + str(child.returncode))
            for item in launch["code_pins"]:
                check(item)
            check(launch["protocol_pin"])
            state["status"] = "COMPLETE"
        except BaseException as error:
            state.update(status="FAILED_OR_STOPPED", reason=str(error))
            raise
        finally:
            # A second termination request must not interrupt owned-group cleanup.
            for signum in previous_signals:
                signal.signal(signum, signal.SIG_IGN)
            if child is not None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait()
            state["active_seconds"] = time.monotonic() - start
            clock["consumed_seconds"][category] += state["active_seconds"]
            replace_json(clock_path, clock)
            replace_json(status_path, state)
            for signum, handler in previous_signals.items():
                signal.signal(signum, handler)
        print(json.dumps({key: state[key] for key in
            ("status", "phase", "active_seconds", "peak_process_tree_rss_bytes")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    run(args.launch, args.sha256)
