#!/usr/bin/env python3
"""Extend the immutable first-window features without changing their history.

Recompute indicator state from the same 2009 origin and persist only dates
after the baseline. Earlier partitions remain hash-pinned references. Only
the existing exact close-review keys are admitted beyond original quote gates.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

import build_momentum_features as feature_worker
from audit_first_window import read_slice
from build_momentum_features import FEATURES, POLICY, compute_features, write_json
from prepare_sp500_daily import canonical_hash, file_hash, require, write_parquet
from review_minor_closes import feature_view


ROOT = Path(__file__).resolve().parent


def checked_object(root, entry):
    path = (Path(root) / entry["path"]).resolve()
    require(path.stat().st_size == entry["size_bytes"] and file_hash(path) == entry["sha256"],
            f"Changed object: {path}")
    return path


def load_native_history(root, start, end, columns=None, verify=True):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    require(manifest["status"] == "complete", "Incomplete extended history")
    require(manifest["start"] <= start <= end <= manifest["end"], "History request exceeds saved bounds")
    parts = [p for p in manifest["history_parts"] if p["start"] <= end and p["end"] >= start]
    paths = [str(checked_object(root, p) if verify else (root / p["path"]).resolve()) for p in parts]
    predicate = (ds.field("date") >= pa.scalar(pd.Timestamp(start))) & (ds.field("date") <= pa.scalar(pd.Timestamp(end)))
    return ds.dataset(paths, format="parquet").to_table(columns=columns, filter=predicate).to_pandas()


def build(dataset, baseline, review, output, end):
    require(not output.exists(), "Output exists; choose a new version")
    source = json.loads((dataset / "manifest.json").read_text())
    base = json.loads((baseline / "manifest.json").read_text())
    reviewed = json.loads((review / "manifest.json").read_text())
    require(source["status"] == base["status"] == reviewed["status"] == "complete", "Incomplete inputs")
    require(base["end"] < end, "Extension must end after the baseline")
    require(base["input_dataset_manifest_sha256"] == reviewed["source_manifest_sha256"] == file_hash(dataset / "manifest.json"),
            "Source dataset lineage mismatch")
    require(base["input_review_manifest_sha256"] == file_hash(review / "manifest.json"), "Review lineage mismatch")
    require(base["script_sha256"] == file_hash(feature_worker.__file__), "Baseline feature code changed")
    require(canonical_hash(json.loads((baseline / "policy.json").read_text())) == canonical_hash(POLICY), "Feature policy changed")
    start, previous_end = base["start"], base["end"]
    calendar = xcals.get_calendar("XNYS", start=start, end=end)
    sessions = calendar.sessions.tz_localize(None)
    require(str(sessions[-1].date()) == end, "End must be an exchange session")
    new_sessions = sessions[sessions > pd.Timestamp(previous_end)]
    require(len(new_sessions) > 0, "No new exchange sessions")
    source_objects = [p for p in source["output_files"] if p["path"].startswith("price_history/")
                      and int(start[:4]) <= int(p["path"].split("/")[1][5:]) <= int(end[:4])]
    for entry in source_objects:
        checked_object(dataset, entry)
    base_parts = [p for p in base["outputs"] if p["path"].startswith("native_features/")]
    for entry in base_parts:
        checked_object(baseline, entry)
    ledger_entry = next(p for p in reviewed["outputs"] if p["path"] == "review_ledger.parquet")
    ledger = pd.read_parquet(checked_object(review, ledger_entry))
    accepted = set(zip(ledger.loc[ledger.admitted, "intrinio_security_id"], ledger.loc[ledger.admitted, "date"]))
    require(len(accepted) == reviewed["admitted_close_bars"] == 107, "Close-review scope changed")
    raw = read_slice(dataset, source, "price_history", ["date", "intrinio_security_id", "adj_close", "quote_status"], start, end)
    view = feature_view(raw, accepted, sessions)
    view = view.loc[view.date.isin(sessions)]
    require(int(view.feature_close_status.eq("reviewed_provisional_close").sum()) == 107, "Scoped review keys missing")
    new_source = view.loc[view.date.gt(previous_end)]
    require(not new_source.feature_close_status.eq("reviewed_provisional_close").any(), "New dates must use original quote gate")
    # Verify every native's previous endpoint before admitting an extension.
    old = ds.dataset([str(baseline / p["path"]) for p in base_parts], format="parquet").to_table(
        filter=ds.field("date") == pa.scalar(pd.Timestamp(previous_end)),
        columns=["intrinio_security_id"] + FEATURES + ["daily_volatility", "contiguous_price_sessions"]).to_pandas().set_index("intrinio_security_id")
    extension, checked, checked_prefix = [], 0, 0
    for native, group in view.groupby("intrinio_security_id", sort=True):
        grid = group.set_index("date").reindex(sessions)
        calculated = compute_features(grid.adj_close.where(grid.feature_close_eligible.eq(True)))
        np.testing.assert_allclose(calculated.feature_return_1d, grid.feature_return_1d, equal_nan=True, rtol=1e-12, atol=1e-14)
        if native in old.index:
            fields = FEATURES + ["daily_volatility", "contiguous_price_sessions"]
            np.testing.assert_allclose(calculated.loc[pd.Timestamp(previous_end), fields].to_numpy(float),
                old.loc[native, fields].to_numpy(float), equal_nan=True, rtol=0, atol=0)
            checked += 1
        # The added future tail cannot influence values on a mid-extension date.
        if checked_prefix < 12 and calculated.feature_ready.iloc[-1]:
            cutoff = len(sessions) - max(1, len(new_sessions) // 2)
            short = compute_features(calculated.feature_adj_close.iloc[:cutoff])
            np.testing.assert_allclose(short[FEATURES], calculated[FEATURES].iloc[:cutoff], equal_nan=True, rtol=0, atol=0)
            checked_prefix += 1
        calculated = calculated.loc[new_sessions].copy()
        calculated["intrinio_security_id"] = native
        calculated["feature_close_status"] = grid.loc[new_sessions, "feature_close_status"].fillna("absent_price")
        calculated["execution_bar_eligible"] = grid.loc[new_sessions, "execution_bar_eligible"].eq(True)
        extension.append(calculated.rename_axis("date").reset_index())
    require(checked == len(old), "Some baseline native histories disappeared")
    frame = pd.concat(extension, ignore_index=True)
    output.mkdir(parents=True)
    parts = []
    for entry in base_parts:
        year = int(entry["path"].split("/")[1][5:])
        parts.append({"path": str((baseline / entry["path"]).resolve()), "size_bytes": entry["size_bytes"], "sha256": entry["sha256"],
                      "start": max(start, f"{year}-01-01"), "end": min(previous_end, f"{year}-12-31")})
    for year, annual in frame.groupby(frame.date.dt.year, sort=True):
        path = output / "native_extension" / f"year={year}" / "part-00000.parquet"
        write_parquet(annual, path)
        parts.append({"path": path.relative_to(output).as_posix(), "size_bytes": path.stat().st_size, "sha256": file_hash(path),
                      "start": str(annual.date.min().date()), "end": str(annual.date.max().date())})
    validation = {"status": "passed", "baseline_endpoint_histories_unchanged": checked, "historical_prefix_checks": checked_prefix,
                  "new_dates_use_original_quote_gate_only": True, "new_close_reviews": 0, "old_partitions_referenced_unchanged": len(base_parts)}
    write_json(output / "validation.json", validation)
    manifest = {"status": "complete", "start": start, "end": end, "baseline_end": previous_end,
        "extension_sessions": len(new_sessions), "extension_rows": len(frame), "native_securities": len(extension),
        "policy_id": POLICY["id"], "reviewed_closes_inherited": 107, "new_reviewed_closes": 0,
        "history_parts": parts, "dataset": str(dataset), "baseline": str(baseline), "review": str(review),
        "input_dataset_manifest_sha256": file_hash(dataset / "manifest.json"),
        "input_baseline_manifest_sha256": file_hash(baseline / "manifest.json"),
        "input_review_manifest_sha256": file_hash(review / "manifest.json"),
        "feature_worker_sha256": file_hash(feature_worker.__file__), "script_sha256": file_hash(__file__),
        "source_objects": source_objects,
        "limits": ["Unchanged provisional close/identity policies; no independently authenticated new prices.",
                   "Later invalid OHLC bars remain withheld even if they contain a numeric close.",
                   "An extended feature table is not an investment universe; current dated PIT membership is still required."]}
    write_json(output / "manifest.json", manifest)
    print(json.dumps({k:v for k,v in manifest.items() if k not in {"history_parts", "source_objects", "limits"}}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/sp500_daily")
    parser.add_argument("--baseline", type=Path, default=ROOT / "data/momentum_features/v1")
    parser.add_argument("--review", type=Path, default=ROOT / "data/feature_close_review/v1")
    parser.add_argument("--output", type=Path, default=ROOT / "data/momentum_history/january_2016_v1")
    parser.add_argument("--end", default="2016-02-01")
    args = parser.parse_args()
    build(args.dataset.resolve(), args.baseline.resolve(), args.review.resolve(), args.output.resolve(), args.end)
