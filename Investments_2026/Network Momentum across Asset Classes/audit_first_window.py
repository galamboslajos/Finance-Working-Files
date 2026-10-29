#!/usr/bin/env python3
"""Audit 2011-2015 PIT observations, warmup and first-2016 graph histories.

No source repair, feature/model fitting or return calculation is performed.
Gap classifications are retrospective audit evidence, not signal inputs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

from audit_warmup_coverage import consecutive_runs, required_prices, FEATURE_PRICES


ROOT = Path(__file__).resolve().parent
START, END = "2011-01-01", "2015-12-31"
WARMUP, ANCHOR = "2009-01-01", "2016-01-04"
PANEL_COLS = ["date", "source_security_id", "membership_ticker", "company_name",
              "intrinio_security_id", "identity_status", "identity_eligible", "quote_status"]


def read_slice(root, manifest, kind, columns, start, end):
    paths = [str(root / row["path"]) for row in manifest["output_files"]
             if row["path"].startswith(kind + "/")
             and int(start[:4]) <= int(row["path"].split("/")[1][5:]) <= int(end[:4])]
    data = ds.dataset(paths, format="parquet")
    condition = ((ds.field("date") >= pa.scalar(pd.Timestamp(start)))
                 & (ds.field("date") <= pa.scalar(pd.Timestamp(end))))
    return data.to_table(columns=columns, filter=condition).to_pandas()


def gap_runs(dates, status, expected=None):
    """Contiguous exchange-session gaps; out-of-universe days split episodes."""
    status = np.asarray(status, dtype=object)
    expected = np.ones(len(status), dtype=bool) if expected is None else np.asarray(expected, dtype=bool)
    good = expected & (status == "ok")
    good_idx = np.flatnonzero(good)
    rows, i = [], 0
    while i < len(status):
        if not expected[i] or status[i] == "ok":
            i += 1
            continue
        start, reason = i, status[i]
        while i + 1 < len(status) and expected[i + 1] and status[i + 1] == reason:
            i += 1
        last = i
        before = bool(len(good_idx) and good_idx[0] < start)
        after = bool(len(good_idx) and good_idx[-1] > last)
        position = ("interior" if before and after else "trailing" if before else
                    "leading" if after else "no_usable_quote")
        length = last - start + 1
        rows.append({"start": str(dates[start].date()), "end": str(dates[last].date()),
                     "sessions": length, "reason": str(reason), "position": position,
                     "length_bucket": "1" if length == 1 else "2-5" if length <= 5 else "6+"})
        i += 1
    return rows


def action_for(status, missing, invalid):
    if status == "known_wrong_issuer":
        return "Resolve reviewed historical issuer conflict; do not substitute a reused ticker."
    if status == "same_date_target_collision":
        return "Review dated native/source routes; retain both original members until resolved."
    if status.startswith("unmapped") or status.startswith("ambiguous"):
        return "Resolve historical security/share-class identity before obtaining prices."
    if missing:
        return "Check existing pinned archives/alternate native routes, then retrieve verified provider observations."
    if invalid:
        return "Review original bar and independent actual observations; keep rejected fields unavailable."
    return "No missing member-day quote; check required pre-membership feature history separately."


def build(dataset, output):
    manifest_path = dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete":
        raise ValueError("Completed source dataset required")
    output.mkdir(parents=True, exist_ok=False)
    all_sessions = xcals.get_calendar("XNYS", start=WARMUP, end=ANCHOR).sessions.tz_localize(None)
    member_sessions = all_sessions[(all_sessions >= START) & (all_sessions <= END)]
    panel = read_slice(dataset, manifest, "panel", PANEL_COLS, START, END)
    anchor = read_slice(dataset, manifest, "panel", PANEL_COLS, ANCHOR, ANCHOR)
    if panel.duplicated(["source_security_id", "date"]).any() or anchor.source_security_id.duplicated().any():
        raise ValueError("Duplicate source/date membership")
    if not panel.date.isin(member_sessions).all() or panel.date.nunique() != len(member_sessions):
        raise ValueError("Missing or invalid membership session")
    prices = read_slice(dataset, manifest, "price_history", [
        "date", "intrinio_security_id", "quote_status", "close", "adj_close"], WARMUP, ANCHOR)
    if prices.duplicated(["intrinio_security_id", "date"]).any():
        raise ValueError("Duplicate native/date prices")
    native_ids = pd.Index(sorted(set(prices.intrinio_security_id.dropna())
                                | set(anchor.intrinio_security_id.dropna())
                                | set(panel.intrinio_security_id.dropna())))
    state_names = ["missing_price", "ok", "invalid_ohlcv", "source_quality_flag", "known_bad_quote", "wrong_frequency"]
    other = set(prices.loc[prices.date.isin(all_sessions), "quote_status"]) - set(state_names)
    if other:
        raise ValueError(f"Unrecognized session quote states: {other}")
    status_grid = np.zeros((len(all_sessions), len(native_ids)), dtype=np.int8)
    finite_closes = np.zeros(status_grid.shape, dtype=bool)
    dp = all_sessions.get_indexer(prices.date)
    np_ = native_ids.get_indexer(prices.intrinio_security_id)
    inside = dp >= 0
    codes = pd.Categorical(prices.quote_status, categories=state_names).codes
    status_grid[dp[inside], np_[inside]] = codes[inside]
    finite = (np.isfinite(prices.close) & prices.close.gt(0)
              & np.isfinite(prices.adj_close) & prices.adj_close.gt(0)).to_numpy()
    finite_closes[dp[inside], np_[inside]] = finite[inside]
    runs = consecutive_runs(status_grid == state_names.index("ok"))
    pos = all_sessions.get_indexer(panel.date)
    ids = native_ids.get_indexer(panel.intrinio_security_id)
    good = panel.quote_status.eq("ok").to_numpy() & panel.identity_eligible.to_numpy(dtype=bool) & (ids >= 0)
    current_runs = np.zeros(len(panel), dtype=np.int32)
    current_runs[good] = runs[pos[good], ids[good]]
    panel["history_314_prices_ready"] = current_runs >= FEATURE_PRICES
    missing = panel.loc[~panel.quote_status.eq("ok")].copy()
    missing.to_parquet(output / "missing_member_days.parquet", index=False, compression="zstd")
    yearly = []
    for year, part in panel.groupby(panel.date.dt.year):
        states = part.quote_status.value_counts()
        row = {"year": int(year), "sessions": part.date.nunique(), "member_days": len(part),
               "unique_members": part.source_security_id.nunique(), "usable_quotes": int(states.get("ok", 0)),
               "missing_prices": int(states.get("missing_price", 0)),
               "identity_withheld": int(states.get("identity_unresolved", 0)),
               "invalid_or_flagged": int((~part.quote_status.isin(["ok", "missing_price", "identity_unresolved"])).sum()),
               "history_314_prices_ready": int(part.history_314_prices_ready.sum())}
        row["quote_coverage"] = row["usable_quotes"] / row["member_days"]
        row["history_314_price_coverage_upper_bound"] = row["history_314_prices_ready"] / row["member_days"]
        yearly.append(row)
    annual = pd.DataFrame(yearly)
    annual.to_csv(output / "coverage_by_year.csv", index=False)
    security_rows, episodes = [], []
    for sid, part in panel.groupby("source_security_id"):
        part = part.sort_values("date")
        status = np.full(len(member_sessions), "not_member", dtype=object)
        ix = member_sessions.get_indexer(part.date)
        status[ix] = part.quote_status
        expected = status != "not_member"
        gaps = gap_runs(member_sessions, status, expected)
        labels = " | ".join(sorted(part.membership_ticker.dropna().unique()))
        for gap in gaps:
            episodes.append({"source_security_id": sid, "membership_tickers": labels, **gap})
        usable_part = part.loc[part.quote_status.eq("ok")]
        problems = part.loc[~part.quote_status.eq("ok")]
        primary = problems.identity_status.mode().iloc[0] if len(problems) else part.identity_status.mode().iloc[0]
        n_missing = int(part.quote_status.eq("missing_price").sum())
        n_identity = int(part.quote_status.eq("identity_unresolved").sum())
        n_bad = len(problems) - n_missing - n_identity
        security_rows.append({
            "source_security_id": sid, "membership_tickers": labels, "company_name": part.iloc[-1].company_name,
            "native_ids": " | ".join(sorted(part.intrinio_security_id.dropna().unique())),
            "first_member_date_in_window": str(part.date.min().date()),
            "last_member_date_in_window": str(part.date.max().date()),
            "expected_member_days": len(part), "usable_quotes": len(usable_part),
            "missing_prices": n_missing, "identity_withheld": n_identity, "invalid_or_flagged": n_bad,
            "unusable_member_days": len(problems), "quote_coverage": len(usable_part) / len(part),
            "complete_while_member": not len(problems),
            "gap_episodes": len(gaps), "longest_gap_sessions": max((g["sessions"] for g in gaps), default=0),
            "interior_gap_sessions": sum(g["sessions"] for g in gaps if g["position"] == "interior"),
            "first_usable_member_date": str(usable_part.date.min().date()) if len(usable_part) else "",
            "last_usable_member_date": str(usable_part.date.max().date()) if len(usable_part) else "",
            "primary_identity_state_of_problem": primary,
            "action": action_for(primary, n_missing, n_bad),
        })
    securities = pd.DataFrame(security_rows).sort_values(["unusable_member_days", "source_security_id"], ascending=[False, True])
    securities.to_csv(output / "security_coverage.csv", index=False)
    pd.DataFrame(episodes).to_csv(output / "member_gap_runs.csv", index=False)
    securities.loc[securities.unusable_member_days.gt(0)].to_csv(output / "repair_queue.csv", index=False)
    anchor_i = all_sessions.get_loc(pd.Timestamp(ANCHOR))
    window = all_sessions[anchor_i-required_prices(1260)+1:anchor_i+1]
    anchor_rows, anchor_gaps = [], []
    for row in anchor.itertuples():
        native_i = native_ids.get_indexer([row.intrinio_security_id])[0]
        history = np.full(len(window), "identity_unresolved", dtype=object)
        finite = np.zeros(len(window), dtype=bool)
        if row.identity_eligible and native_i >= 0:
            sl = slice(anchor_i-required_prices(1260)+1, anchor_i+1)
            history = np.asarray(state_names, dtype=object)[status_grid[sl, native_i]]
            finite = finite_closes[sl, native_i]
        good = history == "ok"
        gaps = gap_runs(window, history)
        for gap in gaps:
            anchor_gaps.append({"source_security_id": row.source_security_id,
                                "membership_ticker": row.membership_ticker, **gap})
        observations = {name: int((history == name).sum()) for name in set(history)}
        quality_rejected_finite = int((~good & finite & (history != "missing_price")).sum())
        anchor_rows.append({
            "as_of": ANCHOR, "source_security_id": row.source_security_id,
            "membership_ticker": row.membership_ticker, "company_name": row.company_name,
            "intrinio_security_id": row.intrinio_security_id, "identity_status": row.identity_status,
            "identity_eligible": bool(row.identity_eligible),
            "required_raw_start": str(window[0].date()), "required_raw_sessions": len(window),
            "usable_history_sessions": int(good.sum()), "missing_or_withheld_history_sessions": int((~good).sum()),
            "price_history_complete": bool(good.all()), "gap_episodes": len(gaps),
            "longest_gap_sessions": max((g["sessions"] for g in gaps), default=0),
            "interior_gap_sessions": sum(g["sessions"] for g in gaps if g["position"] == "interior"),
            "leading_gap_sessions": sum(g["sessions"] for g in gaps if g["position"] == "leading"),
            "trailing_gap_sessions": sum(g["sessions"] for g in gaps if g["position"] == "trailing"),
            "quality_rejected_but_finite_closes": quality_rejected_finite,
            "history_status_counts": json.dumps(observations, sort_keys=True),
        })
    anchor_frame = pd.DataFrame(anchor_rows).sort_values(["price_history_complete", "missing_or_withheld_history_sessions"], ascending=[True, False])
    anchor_frame.to_csv(output / "first_2016_graph_history.csv", index=False)
    pd.DataFrame(anchor_gaps).to_csv(output / "anchor_gap_runs.csv", index=False)
    identity = missing.groupby(["identity_status", "quote_status"]).size().reset_index(name="member_days")
    identity.to_csv(output / "problem_states.csv", index=False)
    sums = Counter()
    for gap in episodes:
        sums[f"episodes_{gap['length_bucket']}_sessions"] += 1
        sums[f"member_days_{gap['length_bucket']}_sessions"] += gap["sessions"]
    summary = {
        "status": "complete_audit_no_repairs_applied",
        "window": {"start": START, "end": END, "raw_warmup_start": WARMUP,
                   "anchor_as_of_end_of_session": ANCHOR,
                   "anchor_required_raw_start": str(window[0].date()),
                   "anchor_required_raw_sessions": len(window)},
        "sessions": len(member_sessions), "unique_historical_members": len(securities),
        "expected_member_days": len(panel), "usable_member_days": len(panel)-len(missing),
        "quote_coverage": (len(panel)-len(missing))/len(panel),
        "member_day_quote_states": panel.quote_status.value_counts().to_dict(),
        "securities_complete_while_members": int(securities.complete_while_member.sum()),
        "securities_with_any_problem": int((~securities.complete_while_member).sum()),
        "member_gap_statistics": dict(sums),
        "interior_member_gap_episodes": sum(g["position"] == "interior" for g in episodes),
        "anchor_constituents": len(anchor_frame),
        "anchor_complete_required_price_history": int(anchor_frame.price_history_complete.sum()),
        "anchor_incomplete_but_at_most_5_bad_history_sessions": int((~anchor_frame.price_history_complete & anchor_frame.missing_or_withheld_history_sessions.le(5)).sum()),
        "anchor_rejected_bars_with_finite_close_candidates": int(anchor_frame.quality_rejected_but_finite_closes.sum()),
        "source_dataset_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "audit_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "limits": [
            "Actual feature values and zero-variance denominators are not tested; sufficient price history is necessary but not sufficient.",
            "The existing dataset's OHLCV quality policy is retained, even for close-only features; finite rejected closes are review candidates only.",
            "Leading history gaps do not establish IPO/listing dates; mapping, capture scope and true listing history require separate evidence.",
            "The anchor cohort is an audit of 2016-01-04 members, not a survivor-filtered replacement for 2011-2015 PIT membership.",
            "No prices, memberships or existing preparation outputs were altered, filled or interpolated.",
            "Interior/leading/trailing gap labels use the full audit interval retrospectively and must not enter predictors or eligibility.",
        ],
    }
    if sum(summary["member_day_quote_states"].values()) != len(panel):
        raise ValueError("Status reconciliation failed")
    if sum(g["sessions"] for g in episodes) != len(missing):
        raise ValueError("Gap-episode reconciliation failed")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(annual.to_string(index=False))
    print(json.dumps(summary, indent=2))
    print("Largest repair cases:")
    print(securities.head(15)[["membership_tickers", "unusable_member_days", "missing_prices", "identity_withheld", "longest_gap_sessions"]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/sp500_daily")
    parser.add_argument("--output", type=Path, default=ROOT / "data/first_window_audit")
    args = parser.parse_args()
    build(args.dataset.resolve(), args.output.resolve())
