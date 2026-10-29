#!/usr/bin/env python3
"""Audit causal price-history readiness for the paper's graph lookbacks.

Reads only security/date/status metadata from the prepared dataset. This is a
necessary-history screen, not a feature build: zero-variance denominators, EWM
initialization, clipping and numerical feature validity are not evaluated.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parent
LOOKBACKS = (252, 504, 756, 1008, 1260)
# 63 prices for the first normalized MACD, followed by 252 valid normalized
# MACDs for its second denominator: 63 + 252 - 1 = 314 observed prices.
FEATURE_PRICES = 314


def consecutive_runs(valid):
    """Count uninterrupted observations, on an already complete session grid."""
    result = np.empty(valid.shape, dtype=np.uint16)
    state = np.zeros(valid.shape[1], dtype=np.uint16)
    for i, row in enumerate(valid):
        state = np.where(row, state + 1, 0)
        result[i] = state
    return result


def required_prices(lookback):
    return FEATURE_PRICES + lookback - 1


def audit(dataset, output, first_year=2010):
    manifest = json.loads((dataset / "manifest.json").read_text())
    if manifest.get("status") != "complete":
        raise ValueError("A completed dataset is required")
    output.mkdir(parents=True, exist_ok=False)
    sessions = xcals.get_calendar(
        "XNYS", start=manifest["price_history_start"],
        end=manifest["requested_end"]).sessions.tz_localize(None)
    history_paths = [dataset / item["path"] for item in manifest["output_files"]
                     if item["path"].startswith("price_history/")]
    panels = [dataset / item["path"] for item in manifest["output_files"]
              if item["path"].startswith("panel/")
              and int(item["path"].split("/")[1][5:]) >= first_year]
    frames = [pq.ParquetFile(p).read(columns=["intrinio_security_id", "date", "quote_status"]).to_pandas()
              for p in history_paths]
    prices = pd.concat(frames, ignore_index=True)
    del frames
    natives = pd.Index(sorted(prices.intrinio_security_id.unique()))
    valid = np.zeros((len(sessions), len(natives)), dtype=bool)
    date_pos = sessions.get_indexer(prices.date)
    native_pos = natives.get_indexer(prices.intrinio_security_id)
    usable = prices.quote_status.eq("ok").to_numpy() & (date_pos >= 0)
    valid[date_pos[usable], native_pos[usable]] = True
    runs = consecutive_runs(valid)
    del prices, valid, date_pos, native_pos
    origins = {"all_history": 0, "from_2004": sessions.searchsorted("2004-01-01"),
               "from_2010": sessions.searchsorted("2010-01-01")}
    daily_parts = []
    for path in panels:
        panel = pq.ParquetFile(path).read(columns=[
            "date", "source_security_id", "intrinio_security_id",
            "analysis_eligible"]).to_pandas()
        row_dates = sessions.get_indexer(panel.date)
        row_ids = natives.get_indexer(panel.intrinio_security_id)
        admitted = panel.analysis_eligible.to_numpy(dtype=bool) & (row_ids >= 0)
        history = np.zeros(len(panel), dtype=np.int32)
        history[admitted] = runs[row_dates[admitted], row_ids[admitted]]
        counts = pd.DataFrame({"date": panel.date, "members": 1,
                               "valid_current_quotes": admitted.astype(int)})
        for name, origin in origins.items():
            available = np.minimum(history, np.maximum(row_dates - origin + 1, 0))
            counts[f"{name}_features_314_prices"] = (available >= FEATURE_PRICES).astype(int)
            for lookback in LOOKBACKS:
                counts[f"{name}_graph_{lookback}"] = (available >= required_prices(lookback)).astype(int)
        daily_parts.append(counts.groupby("date").sum())
    daily = pd.concat(daily_parts).sort_index()
    daily.to_csv(output / "daily.csv")
    rows = []
    for year, group in daily.groupby(daily.index.year):
        total = int(group.members.sum())
        row = {"year": int(year), "sessions": len(group), "member_days": total,
               "current_quote_coverage": group.valid_current_quotes.sum() / total}
        for column in group.columns:
            if column in {"members", "valid_current_quotes"}:
                continue
            row[f"{column}_coverage"] = group[column].sum() / total
        for name in origins:
            col = f"{name}_graph_1260"
            row[f"{name}_graph_1260_first_day_members"] = int(group.iloc[0].members)
            row[f"{name}_graph_1260_first_day_ready"] = int(group.iloc[0][col])
            row[f"{name}_graph_1260_min_daily_ready"] = int(group[col].min())
        rows.append(row)
    annual = pd.DataFrame(rows)
    annual.to_csv(output / "annual.csv", index=False)
    anchors = {}
    for year in [2011, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]:
        index = sessions.searchsorted(f"{year}-01-01")
        if index >= len(sessions):
            continue
        anchors[str(year)] = {
            "first_session": str(sessions[index].date()),
            "latest_raw_start_for_1260_graph": str(sessions[index-required_prices(1260)+1].date()),
        }
    origin = origins["from_2010"]
    report = {
        "status": "price_history_readiness_audit_only",
        "feature_price_observations_required": FEATURE_PRICES,
        "graph_required_price_observations": {str(x): required_prices(x) for x in LOOKBACKS},
        "if_raw_starts_2010_first_possible": {
            "raw_start": str(sessions[origin].date()),
            "features": str(sessions[origin+FEATURE_PRICES-1].date()),
            **{f"graph_{x}": str(sessions[origin+required_prices(x)-1].date()) for x in LOOKBACKS},
        },
        "year_start_anchors": anchors,
        "conventions": [
            "Every XNYS session must have an admissible quote; a missing/bad quote resets the run.",
            "Current constituent eligibility comes from the dated panel; pre-entry price history is allowed.",
            "Prior quote history uses native IDs under the prepared dataset's identity policy.",
            "314-price requirement assumes full 63- and 252-observation rolling MACD normalization windows.",
            "No additional minimum EWM burn-in is imposed; adding one can reduce eligibility.",
            "Results are upper bounds on eight-feature readiness under these completeness rules; zero variance and numerical feature validity are untested.",
            "No feature values, labels, model fits, graph fits or performance returns were read or calculated.",
            "Yearly percentages are member-day weighted; first-day counts and minimum daily counts are also saved.",
        ],
        "calendar_version": xcals.__version__,
        "source_dataset_manifest": str(dataset / "manifest.json"),
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    display = ["year", "current_quote_coverage", "all_history_graph_1260_coverage",
               "from_2010_graph_1260_coverage", "all_history_graph_1260_first_day_ready",
               "from_2010_graph_1260_first_day_ready"]
    print(annual[display].to_string(index=False))
    print(json.dumps(report["if_raw_starts_2010_first_possible"], indent=2))
    print(f"Audit saved: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/sp500_daily")
    parser.add_argument("--output", type=Path, default=ROOT / "data/warmup_coverage")
    parser.add_argument("--first-year", type=int, default=2010)
    args = parser.parse_args()
    audit(args.dataset.resolve(), args.output.resolve(), args.first_year)
