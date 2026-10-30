#!/usr/bin/env python3
"""Check dated graph eligibility and next-session label coverage without fitting models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import exchange_calendars as xcals
import numpy as np
import pandas as pd

from audit_first_window import read_slice
from build_momentum_features import write_json
from extend_momentum_history import load_native_history
from prepare_sp500_daily import file_hash, require, write_parquet


ROOT = Path(__file__).resolve().parent


def audit(history, dataset, output):
    require(not output.exists(), "Audit output exists")
    hm = json.loads((history / "manifest.json").read_text())
    dm = json.loads((dataset / "manifest.json").read_text())
    full = load_native_history(history, hm["start"], hm["end"],
        columns=["date", "intrinio_security_id", "feature_ready", "feature_return_1d", "daily_volatility"])
    sessions = xcals.get_calendar("XNYS", start=hm["start"], end=hm["end"]).sessions.tz_localize(None)
    next_date = pd.Series(sessions[1:], index=sessions[:-1])
    records = []
    for native, frame in full.groupby("intrinio_security_id", sort=False):
        frame = frame.sort_values("date").copy()
        ready = frame.feature_ready.to_numpy(bool)
        positions = np.arange(len(ready)) + 1
        run = positions - np.maximum.accumulate(np.where(~ready, positions, 0))
        frame["feature_run"] = run
        following = frame.date.shift(-1)
        frame["label_date"] = frame.date.map(next_date)
        frame["label_available"] = np.isfinite(frame.feature_return_1d.shift(-1)) & following.eq(frame.label_date)
        records.append(frame.loc[frame.date.between("2016-01-04", "2025-12-31"),
            ["date", "intrinio_security_id", "feature_run", "label_date", "label_available"]])
    states = pd.concat(records, ignore_index=True)
    panel = read_slice(dataset, dm, "panel", ["date", "source_security_id", "intrinio_security_id", "membership_ticker", "identity_eligible"],
                        "2016-01-04", "2025-12-31")
    rows = panel.merge(states, on=["date", "intrinio_security_id"], how="left", validate="many_to_one")
    require(len(rows) == len(panel), "PIT population changed")
    rows["graph_eligible"] = rows.identity_eligible & rows.feature_run.ge(1260)
    rows["available_label_for_node"] = rows.graph_eligible & rows.label_available.eq(True)
    daily = rows.groupby("date").agg(members=("source_security_id", "size"), nodes=("graph_eligible", "sum"),
                                      available_labels=("available_label_for_node", "sum")).reset_index()
    daily["complete_node_labels"] = daily.nodes.eq(daily.available_labels)
    require(int(daily.loc[daily.date.eq("2016-01-04"), "nodes"].iloc[0]) == 429, "First anchor no longer matches pilot")
    annual = daily.groupby(daily.date.dt.year).agg(dates=("date", "size"), member_days=("members", "sum"),
        graph_node_days=("nodes", "sum"), labeled_node_days=("available_labels", "sum"),
        complete_portfolio_dates=("complete_node_labels", "sum"), minimum_nodes=("nodes", "min"), maximum_nodes=("nodes", "max"))
    annual["graph_coverage"] = annual.graph_node_days / annual.member_days
    annual["complete_date_fraction"] = annual.complete_portfolio_dates / annual.dates
    output.mkdir(parents=True)
    daily.to_csv(output / "coverage_daily.csv", index=False)
    annual.to_csv(output / "coverage_annual.csv")
    missing = rows.loc[rows.graph_eligible & ~rows.label_available.eq(True)]
    write_parquet(missing, output / "missing_next_session_labels.parquet")
    write_json(output / "manifest.json", {"status":"complete", "purpose":"availability audit; no labels used for selection or model fit",
        "dates":len(daily), "member_rows":len(rows), "graph_node_rows":int(rows.graph_eligible.sum()),
        "missing_next_session_label_rows":len(missing), "incomplete_portfolio_dates":int((~daily.complete_node_labels).sum()),
        "input_history_manifest_sha256":file_hash(history / "manifest.json"), "input_dataset_manifest_sha256":file_hash(dataset / "manifest.json"),
        "script_sha256":file_hash(__file__)})
    print(annual.to_string())


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history",type=Path,default=ROOT/"data/momentum_history/wfo_2016_2025_v1")
    parser.add_argument("--dataset",type=Path,default=ROOT/"data/sp500_daily")
    parser.add_argument("--output",type=Path,default=ROOT/"data/wfo_availability/v1")
    args=parser.parse_args()
    audit(args.history.resolve(),args.dataset.resolve(),args.output.resolve())
