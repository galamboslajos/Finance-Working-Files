#!/usr/bin/env python3
"""Run independent year batches locally and retain restartable progress logs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from build_momentum_features import write_json
from prepare_sp500_daily import file_hash, require


ROOT = Path(__file__).resolve().parent


def run(output, first_year=2016, last_year=2025, workers=3):
    output.mkdir(parents=True, exist_ok=True)
    logs = output / "logs"; logs.mkdir(exist_ok=True)
    pending, completed, active = [], [], {}
    for year in range(first_year, last_year + 1):
        path = output / str(year) / "manifest.json"
        if path.exists():
            manifest = json.loads(path.read_text())
            require(manifest["status"] == "complete" and manifest["sample_dates"] is None and len(manifest["products"]) == 30,
                    "Existing year is not a complete grid batch")
            completed.append(year)
        else:
            pending.append(year)
    environment = os.environ.copy()
    for option in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        environment[option] = "1"
    last_progress = 0
    while pending or active:
        while pending and len(active) < workers:
            year = pending.pop(0)
            handle = (logs / f"{year}.log").open("a")
            command = [sys.executable, str(ROOT / "daily_network_series.py"), "--output", str(output / str(year)),
                       "--start", "2016-01-04" if year == 2016 else f"{year}-01-01", "--end", f"{year}-12-31"]
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT)
            active[year] = (process, handle)
        failed = []
        for year, (process, handle) in list(active.items()):
            result = process.poll()
            if result is not None:
                handle.close(); del active[year]
                if result:
                    failed.append(year)
                else:
                    require((output / str(year) / "manifest.json").exists(), "Worker exited without complete manifest")
                    completed.append(year)
        if failed:
            # No orphan jobs continue after the batch controller reports failure.
            for process, handle in active.values():
                process.terminate(); process.wait(); handle.close()
            for year in failed:
                print((logs / f"{year}.log").read_text()[-6000:], flush=True)
            raise RuntimeError(f"Year batch failed: {failed}; completed daily checkpoints are retained")
        if time.monotonic() - last_progress >= 30 or not active:
            progress = {}
            for year in active:
                markers = sorted((output / str(year) / "daily").glob("*.json"))
                progress[str(year)] = {"completed_dates": len(markers), "latest": markers[-1].stem if markers else None}
            print(json.dumps({"completed_years": sorted(completed), "active": progress, "queued_years": pending}), flush=True)
            last_progress = time.monotonic()
        time.sleep(1)
    manifests = []
    for year in sorted(completed):
        path = output / str(year) / "manifest.json"
        manifests.append({"year": year, "path": path.relative_to(output).as_posix(), "sha256": file_hash(path)})
    write_json(output / "manifest.json", {"status": "complete", "year_batches": manifests,
               "first_year": first_year, "last_year": last_year, "workers": workers, "script_sha256": file_hash(__file__)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/daily_networks/wfo_grid_by_year_v1")
    parser.add_argument("--first-year", type=int, default=2016)
    parser.add_argument("--last-year", type=int, default=2025)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    run(args.output.resolve(), args.first_year, args.last_year, args.workers)
