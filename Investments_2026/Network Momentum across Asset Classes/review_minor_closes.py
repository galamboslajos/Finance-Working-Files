#!/usr/bin/env python3
"""Review scoped OHLC anomalies and publish a separate close-only feature view.

Provider prices and whole-bar quality flags stay unchanged. Admission is
provisional internal consistency, not independent confirmation of prices.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

from audit_first_window import read_slice
from prepare_sp500_daily import file_hash, write_parquet


ROOT = Path(__file__).resolve().parent
POLICY = "network-momentum-reviewed-close-only-v1"
# Explicit review assumption: a stored adjusted price represents a four-decimal
# value within +/- half a unit. This is a tolerance model, not a supplier promise.
ADJ_HALF_UNIT = 0.00005
NUMERICAL_SLACK = 1e-12
START, END = "2009-01-01", "2016-01-04"
OHLC = ["open", "high", "low", "close"]


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def positive(value):
    return pd.notna(value) and np.isfinite(value) and value > 0


def overlap(left, right):
    return max(left[0], right[0]) <= min(left[1], right[1]) + NUMERICAL_SLACK


def ratio_interval(adjusted, raw):
    return ((adjusted - ADJ_HALF_UNIT) / raw, (adjusted + ADJ_HALF_UNIT) / raw)


def check_close(row, previous):
    """Use this observation and a previously admitted exact-session close only."""
    need(row.frequency == "daily" and row.intraperiod == False, "frequency_or_intraperiod")
    need(positive(row.close) and positive(row.adj_close), "invalid_close")
    need(previous is not None and positive(previous.close) and positive(previous.adj_close),
         "missing_or_invalid_previous_close")
    need(positive(row.factor) and row.split_ratio == 1 and pd.notna(row.dividend)
         and np.isfinite(row.dividend) and row.dividend >= 0, "unexpected_or_missing_action_metadata")
    for prefix in ("", "adj_"):
        high, low, close = row[prefix + "high"], row[prefix + "low"], row[prefix + "close"]
        need(positive(high) and close <= high, "close_above_high_or_missing_high")
        if pd.notna(low):
            need(positive(low) and low <= high and close >= low, "close_below_low_or_invalid_range")
    open_anomaly = row.price_quality_status == "provider_ohlc_anomaly"
    incomplete = row.price_quality_status == "provider_incomplete_ohlc"
    if open_anomaly:
        for prefix in ("", "adj_"):
            need(all(positive(row[prefix + c]) for c in OHLC), "unexpected_missing_ohlc")
            need(not row[prefix + "low"] <= row[prefix + "open"] <= row[prefix + "high"],
                 "not_scoped_open_range_anomaly")
        issue = "open_range_anomaly_close_only"
    elif incomplete:
        need(pd.isna(row.low) and pd.isna(row.adj_low), "not_scoped_missing_low")
        need(positive(row.open) and positive(row.adj_open)
             and row.open <= row.high and row.adj_open <= row.adj_high, "unexpected_incomplete_ohlc")
        issue = "missing_low_close_only"
    else:
        raise ValueError("unreviewed_provider_quality_state")
    current_ratio = ratio_interval(row.adj_close, row.close)
    # Paired high/low and close must permit a common adjustment scale under the
    # declared precision model. The suspect open is deliberately excluded.
    intervals = [current_ratio]
    for field in ("high", "low"):
        raw, adjusted = row[field], row["adj_" + field]
        if pd.notna(raw) and pd.notna(adjusted):
            intervals.append(ratio_interval(adjusted, raw))
    need(max(x[0] for x in intervals) <= min(x[1] for x in intervals) + NUMERICAL_SLACK,
         "paired_price_scale_inconsistent")
    prev_ratio = ratio_interval(previous.adj_close, previous.close)
    need(overlap((current_ratio[0] * row.factor, current_ratio[1] * row.factor), prev_ratio),
         "adjustment_step_inconsistent_with_prior_close_and_recorded_factor")
    return {"issue": issue,
            "adjustment_step": (row.adj_close / row.close) / (previous.adj_close / previous.close),
            "recorded_factor": float(row.factor),
            "raw_return_from_previous": row.close / previous.close - 1,
            "adjusted_return_from_previous": row.adj_close / previous.adj_close - 1}


def review_targets(context, targets, sessions):
    need(not context.duplicated(["intrinio_security_id", "date"]).any(), "duplicate_context_key")
    need(not targets.duplicated(["intrinio_security_id", "date"]).any(), "duplicate_target_key")
    lookup = context.set_index(["intrinio_security_id", "date"])
    prior = pd.Series(sessions[:-1], index=sessions[1:])
    accepted, rows = set(), []
    for target in targets.sort_values(["intrinio_security_id", "date"]).itertuples():
        key = target.intrinio_security_id, target.date
        entry = {"source_security_id": target.source_security_id,
                 "membership_ticker": target.membership_ticker,
                 "intrinio_security_id": target.intrinio_security_id, "date": target.date,
                 "admitted": False, "reason": None, "policy_id": POLICY}
        try:
            need(target.date in prior.index, "non_session_target")
            need(key in lookup.index, "target_bar_absent")
            row = lookup.loc[key]
            prev_key = key[0], prior.loc[key[1]]
            need(prev_key in lookup.index, "previous_session_bar_absent")
            previous = lookup.loc[prev_key]
            need(previous.quote_status == "ok" or prev_key in accepted,
                 "previous_session_close_not_admitted")
            details = check_close(row, previous)
            entry.update(details)
            entry.update(admitted=True, reason="provisional_internal_consistency_pass",
                         original_close=float(row.close), original_adj_close=float(row.adj_close),
                         original_quote_status=row.quote_status,
                         original_price_quality_status=row.price_quality_status,
                         previous_session=prev_key[1], source_page_hash=row.get("source_page_hash"),
                         independently_price_verified=False)
            accepted.add(key)
        except ValueError as error:
            entry["reason"] = str(error)
        rows.append(entry)
    return pd.DataFrame(rows), accepted


def feature_view(prices, accepted, sessions):
    out = prices.copy().sort_values(["intrinio_security_id", "date"]).reset_index(drop=True)
    keys = list(zip(out.intrinio_security_id, out.date))
    reviewed = pd.Series([key in accepted for key in keys], index=out.index)
    original_ok = out.quote_status.eq("ok")
    out["feature_close_eligible"] = original_ok | reviewed
    out["feature_close_status"] = np.select([reviewed, original_ok],
        ["reviewed_provisional_close", "original_accepted_quote"], default="withheld")
    out["execution_bar_eligible"] = original_ok
    out["close_review_policy"] = pd.Series(np.where(reviewed, POLICY, None), dtype="string")
    trading = out.loc[out.date.isin(sessions)].copy()
    group = trading.groupby("intrinio_security_id", sort=False)
    previous_session = pd.Series(sessions[:-1], index=sessions[1:])
    exact_previous = group.date.shift().eq(trading.date.map(previous_session))
    previous_ok = group.feature_close_eligible.shift().eq(True)
    valid = trading.feature_close_eligible & previous_ok & exact_previous
    trading["feature_return_1d"] = (trading.adj_close / group.adj_close.shift() - 1).where(valid)
    trading["feature_log_return_1d"] = np.log1p(trading.feature_return_1d)
    trading["feature_return_status"] = np.select([
        ~trading.feature_close_eligible, ~exact_previous, ~previous_ok],
        ["current_close_withheld", "previous_session_missing", "previous_close_withheld"], default="ok")
    out = out.join(trading[["feature_return_1d", "feature_log_return_1d", "feature_return_status"]])
    out["feature_return_status"] = out.feature_return_status.fillna("non_session")
    return out


def build(dataset, audit, output):
    need(not output.exists(), "Output exists; select a new version")
    manifest_path = dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    need(manifest.get("status") == "complete", "Complete source dataset required")
    keys_path = audit / "minor_gap_keys.parquet"
    targets_all = pd.read_parquet(keys_path)
    targets = targets_all.loc[targets_all.gap_reason.eq("invalid_ohlcv")].copy()
    need(len(targets) == 107, "Reviewed target scope changed")
    sessions = xcals.get_calendar("XNYS", start=START, end=END).sessions.tz_localize(None)
    # Only previous-session controls are required; no future prices determine
    # admission. Original accepted controls and earlier admitted targets qualify.
    previous = pd.Series(sessions[:-1], index=sessions[1:])
    conditions = []
    for native, group in targets.groupby("intrinio_security_id"):
        days = sorted(set(group.date) | set(group.date.map(previous)))
        conditions.append((ds.field("intrinio_security_id") == native)
                          & ds.field("date").isin(pa.array(days, type=pa.timestamp("us"))))
    predicate = conditions[0]
    for term in conditions[1:]:
        predicate |= term
    source_objects = [x for x in manifest["output_files"] if x["path"].startswith("price_history/")
                      and 2009 <= int(x["path"].split("/")[1][5:]) <= 2016]
    for item in source_objects:
        p = dataset / item["path"]
        need(p.stat().st_size == item["size_bytes"] and file_hash(p) == item["sha256"],
             f"Source fingerprint changed: {p}")
    source = ds.dataset([str(dataset / x["path"]) for x in source_objects], format="parquet")
    context = source.to_table(filter=predicate).to_pandas()
    ledger, accepted = review_targets(context, targets, sessions)
    output.mkdir(parents=True)
    write_parquet(context, output / "review_context.parquet")
    ledger.to_csv(output / "review_ledger.csv", index=False)
    write_parquet(ledger, output / "review_ledger.parquet")
    # Store an explicit policy before materialization; all prices remain the
    # original supplied values. No independent-source authenticity is asserted.
    policy = {"id": POLICY, "target_scope": "107 exact existing native/date bars from the first-window minor-gap audit",
              "adjusted_price_precision_assumption": {"half_unit": ADJ_HALF_UNIT, "numerical_slack": NUMERICAL_SLACK},
              "checks": ["finite positive closes", "available raw/adjusted high-low bounds",
                         "only scoped open-range or missing-low anomaly", "common high/low/close adjustment scale",
                         "prior-session adjustment-ratio step consistent with recorded factor under declared precision model"],
              "independent_price_authentication": False, "admission": "provisional research close only",
              "future_observations_used_for_admission": False,
              "execution_fields_repaired": False, "source_prices_replaced": False}
    (output / "policy.json").write_text(json.dumps(policy, indent=2) + "\n")
    tail, close_delta, return_delta, price_rows = None, 0, 0, 0
    anchor_rows, expected_source = [], read_slice(dataset, manifest, "panel", [
        "date", "source_security_id", "intrinio_security_id", "membership_ticker", "identity_eligible"], END, END)
    need(len(expected_source) == 505, "Anchor membership changed")
    good_counts = {}
    required_start = pd.Timestamp("2009-10-05")
    for year in range(2009, 2017):
        start = f"{year}-01-01"
        end = min(f"{year}-12-31", END)
        raw = source.to_table(filter=(ds.field("date") >= pa.scalar(pd.Timestamp(start)))
                              & (ds.field("date") <= pa.scalar(pd.Timestamp(end)))).to_pandas()
        if raw.empty:
            continue
        combined = raw if tail is None else pd.concat([tail, raw], ignore_index=True)
        view = feature_view(combined, accepted, sessions)
        view = view.loc[view.date.dt.year.eq(year)].copy()
        original_columns = list(raw.columns)
        pd.testing.assert_frame_equal(
            view[original_columns].sort_values(["intrinio_security_id", "date"]).reset_index(drop=True),
            raw.sort_values(["intrinio_security_id", "date"]).reset_index(drop=True), check_dtype=False)
        close_delta += int((view.feature_close_eligible & ~view.quote_status.eq("ok")).sum())
        return_delta += int((view.feature_return_1d.notna() & view.provider_adjusted_return_1d.isna()).sum())
        eligible = view.loc[(view.date >= required_start) & view.feature_close_eligible]
        for native, count in eligible.groupby("intrinio_security_id").size().items():
            good_counts[native] = good_counts.get(native, 0) + int(count)
        write_parquet(view, output / "feature_prices" / f"year={year}" / "part-00000.parquet")
        price_rows += len(view)
        year_sessions = sessions[sessions.year == year]
        tail = raw.loc[raw.date.eq(year_sessions[-1])].copy() if len(year_sessions) else None
    before = pd.read_csv(audit / "first_2016_graph_history.csv").set_index("source_security_id")
    for row in expected_source.itertuples():
        count = good_counts.get(row.intrinio_security_id, 0) if row.identity_eligible else 0
        original = before.loc[row.source_security_id]
        anchor_rows.append({"source_security_id": row.source_security_id, "membership_ticker": row.membership_ticker,
                            "original_usable_history_sessions": int(original.usable_history_sessions),
                            "reviewed_usable_history_sessions": count,
                            "original_history_complete": bool(original.price_history_complete),
                            "reviewed_history_complete": count == 1573,
                            "remaining_unusable_sessions": 1573 - count})
    anchor = pd.DataFrame(anchor_rows)
    anchor.to_csv(output / "first_2016_history_after_review.csv", index=False)
    outputs = [{"path": p.relative_to(output).as_posix(), "size_bytes": p.stat().st_size, "sha256": file_hash(p)}
               for p in sorted(output.rglob("*")) if p.is_file()]
    summary = {"status": "complete", "policy_id": POLICY, "start": START, "end": END,
               "source_dataset": str(dataset), "source_manifest_sha256": file_hash(manifest_path),
               "target_keys_sha256": file_hash(keys_path), "preparation_script_sha256": file_hash(__file__),
               "target_bars": len(targets), "admitted_close_bars": len(accepted),
               "refused_close_bars": int((~ledger.admitted).sum()), "feature_price_rows": price_rows,
               "additional_feature_closes": close_delta, "additional_feature_return_rows": return_delta,
               "anchor_members": len(anchor), "anchor_original_complete": int(anchor.original_history_complete.sum()),
               "anchor_reviewed_complete": int(anchor.reviewed_history_complete.sum()),
               "restored_anchor_securities": int((anchor.reviewed_history_complete & ~anchor.original_history_complete).sum()),
               "remaining_eight_absent_bars": targets_all.loc[targets_all.gap_reason.eq("missing_price")]
                    .assign(date=lambda x: x.date.dt.strftime("%Y-%m-%d")).to_dict("records"),
               "source_objects": source_objects, "outputs": outputs,
               "limits": ["Provisional internal consistency under an explicit precision assumption; no independent validation of closing prices.",
                          "Only reviewed closing fields are usable for features. Source OHLC/quality flags and execution eligibility are preserved.",
                          "The eight absent observations remain missing; no forward/back fill, interpolation or synthetic bars.",
                          "Actual eight-feature values and graph/model fits have not been calculated."]}
    (output / "manifest.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k:v for k,v in summary.items() if k not in {"source_objects", "outputs", "remaining_eight_absent_bars", "limits"}}, indent=2))


def load_feature_prices(start, end, columns=None, layer=ROOT / "data/feature_close_review/v1"):
    layer = Path(layer)
    manifest = json.loads((layer / "manifest.json").read_text())
    need(manifest["status"] == "complete", "Feature-price layer is incomplete")
    need(manifest["start"] <= start <= end <= manifest["end"], "Request exceeds reviewed layer bounds")
    paths = [str(layer / x["path"]) for x in manifest["outputs"]
             if x["path"].startswith("feature_prices/")
             and int(start[:4]) <= int(x["path"].split("/")[1][5:]) <= int(end[:4])]
    data = ds.dataset(paths, format="parquet")
    condition = (ds.field("date") >= pa.scalar(pd.Timestamp(start))) & (ds.field("date") <= pa.scalar(pd.Timestamp(end)))
    return data.to_table(columns=columns, filter=condition).to_pandas()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/sp500_daily")
    parser.add_argument("--audit", type=Path, default=ROOT / "data/first_window_audit")
    parser.add_argument("--output", type=Path, default=ROOT / "data/feature_close_review/v1")
    args = parser.parse_args()
    build(args.dataset.resolve(), args.audit.resolve(), args.output.resolve())
