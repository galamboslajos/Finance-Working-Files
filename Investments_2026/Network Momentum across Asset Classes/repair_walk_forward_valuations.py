#!/usr/bin/env python3
"""Repair unavailable held valuations without changing the frozen WFO experiment.

The overlay affects realized labels only. No feature, forecast, membership,
cohort, position, scaler, graph or fitted model is revised. Mandatory merger
entitlements are values of received assets, not fabricated stock-price bars.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from build_momentum_features import write_json
from evaluate_walk_forward import ARMS, paper_proxy_returns, performance
from prepare_sp500_daily import file_hash, require

ROOT = Path(__file__).resolve().parent
INDEPENDENT_SHA = "4a4919d48128d5c573ddbca64649d8cc78783f7bbb818219a3ad5f34728e38a0"
ORIGINAL_SHA = "7fad2053eec5e87556e0930a7610ffd8156e8d103b1e72222a0827ddacb80f63"
KEY = ["date", "source_security_id"]
HALF_UNIT = .00005  # Intrinio adjusted prices are stored to four decimal places.


def positive(value):
    return pd.notna(value) and np.isfinite(value) and value > 0


def valid_quote(row):
    require(all(positive(row[x]) for x in ("open", "high", "low", "close")), "Invalid independent OHLC")
    require(row.low <= min(row.open, row.close) <= max(row.open, row.close) <= row.high,
            "Independent OHLC bounds fail")


def same_scale(a, b):
    """Both rounded adjusted/raw ratios must admit the same scale."""
    left = max((a.adj_close - HALF_UNIT) / a.close, (b.adj_close - HALF_UNIT) / b.close)
    right = min((a.adj_close + HALF_UNIT) / a.close, (b.adj_close + HALF_UNIT) / b.close)
    return left <= right + 1e-12


def quote_return(previous_raw, current_raw, factor=1.):
    require(all(positive(x) for x in (previous_raw, current_raw, factor)), "Invalid return inputs")
    return current_raw / (previous_raw * factor) - 1


def event_value(event, acquirer_raw=0.):
    require(event["cash"] >= 0 and event["ratio"] >= 0, "Invalid merger terms")
    if event.get("contingent_right_max", 0) > 0:
        return np.nan  # Cash alone or face value is not a certified rights mark.
    require(event["ratio"] == 0 or positive(acquirer_raw), "Acquirer raw close required")
    return event["cash"] + event["ratio"] * acquirer_raw


def checked_predictions(results):
    manifest = json.loads((results / "manifest.json").read_text())
    require(file_hash(results / "manifest.json") == ORIGINAL_SHA, "Wrong original WFO result version")
    paths = []
    for item in manifest["outputs"]:
        if item["path"].startswith("predictions/"):
            path = results / item["path"]
            require(file_hash(path) == item["sha256"], "Original predictions changed")
            paths.append(path)
    frame = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True).sort_values(KEY).reset_index(drop=True)
    require(not frame.duplicated(KEY).any(), "Duplicate original forecast key")
    return frame, manifest


def overlay(frame, ledger):
    require(not ledger.duplicated(KEY).any(), "Duplicate repair key")
    values = ledger.set_index(KEY).repaired_forward_return.reindex(pd.MultiIndex.from_frame(frame[KEY])).to_numpy()
    present = np.isfinite(values)
    require(not (present & np.isfinite(frame.forward_return.to_numpy())).any(), "Repair would replace an observed label")
    out = frame.copy()
    out.loc[present, "forward_return"] = values[present]
    out.loc[present, "target"] = values[present] / out.loc[present, "daily_volatility"]
    pd.testing.assert_frame_equal(out.drop(columns=["forward_return", "target"]),
                                  frame.drop(columns=["forward_return", "target"]))
    return out


def daily_returns(frame):
    result = None
    for arm in ARMS:
        part = paper_proxy_returns(frame, frame[arm + "_position"])
        if result is None:
            result = part.rename(columns={"return": arm, "net_weight": arm + "_net_weight",
                                          "gross_weight": arm + "_gross_weight"})
        else:
            np.testing.assert_array_equal(result.complete, part.complete)
            result[arm], result[arm + "_net_weight"], result[arm + "_gross_weight"] = (
                part["return"].to_numpy(), part.net_weight.to_numpy(), part.gross_weight.to_numpy())
    result["return_date"] = result.date.map(frame.groupby("date").label_date.first())
    result["test_year"] = result.return_date.dt.year
    return result


def independent_lookup(quotes, sid, day):
    found = quotes.loc[(quotes.security_id == sid) & (quotes.trading_date == str(pd.Timestamp(day).date()))]
    require(len(found) == 1, "Independent quote absent or ambiguous")
    row = found.iloc[0]
    require(row.adjustment == "raw" and row.feed == "sip" and row.timeframe == "1Day", "Wrong quote basis")
    valid_quote(row)
    return row


def ordinary_repair(target, history, quotes, manual):
    native = target.intrinio_security_id
    context = history.loc[history.intrinio_security_id.eq(native)].sort_values("date")
    prior = context.loc[context.date.eq(target.date)]
    require(len(prior) == 1 and prior.iloc[0].quote_status == "ok", "Original signal close not accepted")
    previous = prior.iloc[0]
    require(positive(previous.close) and positive(previous.adj_close), "Invalid signal units")
    local = context.loc[context.date.eq(target.label_date)]
    entry = {"prior_raw_close": float(previous.close), "prior_adjusted_close": float(previous.adj_close),
             "original_next_raw_close": float(local.close.iloc[0]) if len(local) else None,
             "original_next_quality": str(local.price_quality_status.iloc[0]) if len(local) else "absent"}
    if manual is None:
        q = independent_lookup(quotes, target.source_security_id, target.label_date)
        require(q.query_symbol == target.membership_ticker, "Independent symbol does not match scoped identity")
        price = float(q.close)
        # Match TWO existing controls; no future price is interpolated or used
        # to choose a security. Controls establish vendor units/identity only.
        controls = context.loc[context.date.le(target.date) & context.quote_status.eq("ok")].tail(2)
        require(len(controls) == 2, "Insufficient price controls")
        control_records = []
        for control in controls.itertuples():
            independent = independent_lookup(quotes, target.source_security_id, control.date)
            require(independent.query_symbol == q.query_symbol, "Control symbol differs")
            require(abs(independent.close - control.close) <= .005000001, "Independent control close differs")
            control_records.append({"date": str(control.date.date()), "raw_close": float(control.close)})
        entry.update(method="independent_sip_raw_close", quote_source="pinned Alpaca SIP daily_bars object",
                     independent_source_page_hash=q.source_page_hash, independent_request_id=q.source_request_id,
                     independent_security_id=q.security_id, quote_symbol=q.query_symbol,
                     controls_json=json.dumps(control_records))
    else:
        require(manual["native"] == native and manual["ticker"] == target.membership_ticker, "Manual identity mismatch")
        require(pd.Timestamp(manual["date"]) == target.label_date, "Manual date mismatch")
        for day, close in manual["prior_controls"].items():
            control = context.loc[context.date.eq(pd.Timestamp(day))]
            require(len(control) == 1 and control.iloc[0].quote_status == "ok", "Manual control absent")
            require(abs(control.iloc[0].close - close) < 1e-8, "Manual control units mismatch")
        if all(field in manual for field in ("open", "high", "low")):
            valid_quote(pd.Series(manual))
        price = manual["close"]
        entry.update(method="reviewed_predecessor_quote", quote_source=manual["source"],
                     identity_source=manual["identity_source"], quote_symbol=manual["symbol"],
                     controls_json=json.dumps(manual["prior_controls"]),
                     evidence_limit="Reviewed web transcription; no downloaded raw response")
    if len(local):
        bar = local.iloc[0]
        require(bar.frequency == "daily" and not bar.intraperiod, "Wrong original bar period")
        require(bar.factor == 1 and bar.split_ratio == 1 and bar.dividend == 0,
                "Action requires a separately reviewed return rule")
        require(same_scale(previous, bar), "Rejected bar's paired close scale is inconsistent")
        factor = float(bar.factor)
    else:
        after = context.loc[context.date.gt(target.label_date) & context.quote_status.eq("ok")].head(1)
        require(len(after) == 1, "Missing adjustment continuity control")
        following = after.iloc[0]
        require((following.date - target.label_date).days <= 10, "Adjustment bracket too wide")
        require(same_scale(previous, following), "Adjustment scales differ across the absent bar")
        between = context.loc[context.date.gt(target.date) & context.date.le(following.date)]
        require(between.factor.eq(1).all() and between.split_ratio.eq(1).all() and between.dividend.eq(0).all(),
                "Action within quote bracket requires review")
        factor = 1.
        entry.update(adjustment_bracket_end=str(following.date.date()),
                     adjustment_continuity="Unchanged rounded adjusted/raw scale across bracket; zero recorded intervening actions. Absent-bar action feed is not separately certified.")
    entry.update(repaired_raw_close=price, adjustment_factor=factor,
                 repaired_forward_return=quote_return(previous.close, price, factor), status="repaired")
    return entry


def merger_repair(target, event, history, event_quotes, independent):
    require(pd.Timestamp(event["effective_date"]) <= target.label_date, "Merger not effective by valuation")
    previous = history.loc[history.intrinio_security_id.eq(target.intrinio_security_id) & history.date.eq(target.date)]
    require(len(previous) == 1 and previous.iloc[0].quote_status == "ok", "Original entry quote absent")
    previous = previous.iloc[0]
    entry = {"method": "mandatory_merger_entitlement", "prior_raw_close": float(previous.close),
             "cash_per_old_share": event["cash"], "exchange_ratio": event["ratio"],
             "event_effective_date": event["effective_date"], "event_sources": json.dumps(event["sources"]),
             "note": event.get("note", ""), "quote_symbol": event.get("acquirer"),
             "repaired_forward_return": np.nan}
    acquirer_close = 0.
    if event["ratio"]:
        if "native" in event:
            q = event_quotes.loc[event_quotes.intrinio_security_id.eq(event["native"])
                                 & event_quotes.trading_date.eq(str(target.label_date.date()))]
            require(len(q) == 1 and q.iloc[0].price_quality_status == "accepted", "Acquirer quote unavailable")
            q = q.iloc[0]
            require(q.ticker == event.get("quote_ticker", event["acquirer"]), "Acquirer class/ticker mismatch")
            valid_quote(q)
            require(q.frequency == "daily" and not q.intraperiod and q.dividend == 0,
                    "Acquirer period/dividend requires explicit entitlement review")
            acquirer_close = float(q.close)
            entry.update(acquirer_native=event["native"], quote_source="pinned Intrinio accepted raw close",
                         acquirer_page_hash=q.source_page_hash)
            check = independent.loc[independent.query_symbol.eq(event["acquirer"])
                                    & independent.trading_date.eq(str(target.label_date.date()))]
            if len(check):
                require(len(check) == 1 and abs(check.close.iloc[0] - acquirer_close) < 1e-8, "Acquirer cross-source quote differs")
                entry["acquirer_sip_close_agrees"] = True
        else:
            q = independent_lookup(independent, event["independent_security_id"], target.label_date)
            require(q.query_symbol == event["acquirer"], "Acquirer independent identity mismatch")
            acquirer_close = float(q.close)
            entry.update(quote_source="pinned Alpaca SIP raw close", independent_source_page_hash=q.source_page_hash,
                         independent_security_id=q.security_id)
    value = event_value(event, acquirer_close)
    entry.update(acquirer_raw_close=acquirer_close if event["ratio"] else None,
                 total_entitlement_value=value, contingent_right_max=event.get("contingent_right_max", 0))
    if np.isfinite(value):
        require(value > 0, "Invalid terminal entitlement")
        entry.update(status="repaired", repaired_forward_return=value / previous.close - 1)
    else:
        entry.update(status="unresolved_contingent_right", note=event["contingent_right"],
                     stock_return_if_claim_mark_zero=event["cash"] / previous.close - 1,
                     stock_return_if_claim_mark_face=(event["cash"] + event["contingent_right_max"]) / previous.close - 1)
    return entry


def claim_sensitivity(frame, ledger):
    """One-day mark sensitivity only; no sale or subsequent CVR cash flow assumed."""
    rows = []
    for repair in ledger.loc[ledger.status.eq("unresolved_contingent_right")].itertuples():
        block = frame.loc[frame.date.eq(repair.date)]
        affected = block.source_security_id.eq(repair.source_security_id).to_numpy()
        require(affected.sum() == 1 and block.forward_return.isna().sum() == 1, "Unexpected unresolved cohort")
        for arm in ARMS:
            weights = block[arm + "_position"].to_numpy() * .15 / (np.sqrt(252) * block.daily_volatility.to_numpy() * len(block))
            realized = block.forward_return.to_numpy()
            known = float(np.sum(weights[~affected] * realized[~affected]))
            w = float(weights[affected][0])
            cash_mark = known + w * repair.stock_return_if_claim_mark_zero
            face_mark = known + w * repair.stock_return_if_claim_mark_face
            rows.append({"ticker": repair.membership_ticker, "return_date": repair.label_date, "strategy": arm,
                         "nodes": len(block), "position_weight": w, "claim_face_per_old_share": repair.contingent_right_max,
                         "portfolio_return_claim_zero_mark": cash_mark, "portfolio_return_claim_face_mark": face_mark,
                         "one_day_return_lower": min(cash_mark, face_mark), "one_day_return_upper": max(cash_mark, face_mark),
                         "return_range_basis_points": abs(face_mark - cash_mark) * 10000,
                         "interpretation": "Hypothetical one-day marks; NOT realized liquidation, full wealth bounds, or a carried-rights ledger"})
    return pd.DataFrame(rows)


def build(results, audit, output, event_config, manual_config):
    require(not output.exists(), "Repair output exists; use a new version")
    original, source = checked_predictions(results)
    triage_path = ROOT / "data/wfo_availability/source_triage_v1/missing_label_source_triage.csv"
    triage = pd.read_csv(triage_path, parse_dates=["date", "label_date"])
    missing = original.loc[~np.isfinite(original.forward_return), KEY + ["label_date", "intrinio_security_id", "membership_ticker"]]
    require(len(missing) == 62, "Original missing-label scope changed")
    require(not triage.duplicated(KEY).any(), "Ambiguous triage key")
    check = missing.merge(triage[KEY], on=KEY, how="left", indicator=True, validate="one_to_one")
    require(check._merge.eq("both").all(), "Missing row absent from source triage")
    independent_path = audit / "independent_daily_bars.parquet"
    require(file_hash(independent_path) == INDEPENDENT_SHA, "Independent whole-object SHA mismatch")
    # Filtering is for memory only, after full-object checksum verification.
    ids = set(missing.source_security_id)
    events = json.loads(event_config.read_text())["events"]
    acquirers = {e.get("acquirer", "") for e in events}
    independent = ds.dataset(independent_path).to_table(filter=(ds.field("adjustment") == "raw")
        & (ds.field("security_id").isin(ids) | ds.field("query_symbol").isin(acquirers))).to_pandas()
    history = pd.read_parquet(audit / "original_native_context.parquet")
    event_quotes = pd.read_parquet(audit / "event_date_source_prices.parquet")
    manual = json.loads(manual_config.read_text())["quotes"]
    event_map = {(e["ticker"], pd.Timestamp(e["label_date"])): e for e in events}
    manual_map = {(q["ticker"], pd.Timestamp(q["date"])): q for q in manual}
    require(len(event_map) == len(events) and len(manual_map) == len(manual), "Duplicate configured key")
    records = []
    for target in missing.sort_values(["label_date", "source_security_id"]).itertuples():
        key = (target.membership_ticker, target.label_date)
        record = {k: getattr(target, k) for k in missing.columns}
        if key in event_map:
            record.update(merger_repair(target, event_map[key], history, event_quotes, independent))
        else:
            record.update(ordinary_repair(target, history, independent, manual_map.get(key)))
        records.append(record)
    ledger = pd.DataFrame(records)
    repaired = overlay(original, ledger)
    daily = daily_returns(repaired)
    old_daily = pd.read_csv(results / "daily_paper_proxy_returns.csv", parse_dates=["date", "return_date"])
    old_daily = old_daily.sort_values("date").reset_index(drop=True)
    for col in ["date", "nodes", "return_date", "test_year", *[a+s for a in ARMS for s in ("_net_weight", "_gross_weight")]]:
        if pd.api.types.is_numeric_dtype(daily[col]):
            np.testing.assert_allclose(daily[col], old_daily[col], rtol=1e-13, atol=1e-14)
        else:
            np.testing.assert_array_equal(daily[col], old_daily[col])
    for arm in ARMS:
        known = old_daily.complete
        np.testing.assert_allclose(daily.loc[known, arm], old_daily.loc[known, arm], rtol=1e-12, atol=1e-14)
    output.mkdir(parents=True)
    ledger.to_csv(output / "valuation_ledger.csv", index=False)
    ledger.to_parquet(output / "valuation_ledger.parquet", index=False)
    daily.to_csv(output / "daily_paper_proxy_returns.csv", index=False)
    claim_sensitivity(repaired, ledger).to_csv(output / "contingent_claim_sensitivity.csv", index=False)
    metrics = pd.DataFrame([{"period": period, "strategy": arm, **performance(block[arm])}
        for period, block in [("all", daily), *[(str(y), b) for y, b in daily.groupby("test_year")]] for arm in ARMS])
    metrics.to_csv(output / "paper_proxy_metrics.csv", index=False)
    policy = {"id": "sp500-frozen-wfo-held-valuation-v1", "scope": "62 unavailable OOS realized labels; frozen original 2019-2025 forecasts",
        "ordinary_quote_rule": "Exact independent raw quote, accepted entry quote and matching raw controls; existing recorded action metadata or unchanged rounded adjustment-scale bracket required.",
        "merger_rule": "(USD cash + fixed share ratio * exact label-date acquirer raw close) / old raw entry close - 1.",
        "claim_rule": "Unquoted non-tradeable contingent rights remain unresolved; no zero, face-value, or ex-post realized payout is silently substituted.",
        "source_policy": "Alpaca whole-object checksum; existing Intrinio accepted source records; three reviewed predecessor-quote transcriptions with four prior raw controls and dated identity evidence.",
        "limitations": ["Valuation-only retrospective repair: original features, models, validation selection and eligibility remain frozen, including their original gaps.",
            "Raw source data and original WFO artifacts are never overwritten. This does not certify a fully corrected training dataset.",
            "Adjustment continuity across absent bars is a reviewed bracket assumption, not a complete independent corporate-action feed.",
            "Merger marks assume frictionless cash/share receivables valued at par or market; settlement delays, fractional-share cash, tax, borrow and costs are not modeled.",
            "An original close on a completion day may be stale or untradeable; these fixed-forecast corrections do not establish executable entries.",
            "Contingent-right sensitivity covers the one-day mark only, not subsequent rights value, payouts or a continuous realized strategy wealth path."]}
    write_json(output / "policy.json", policy)
    inputs = [event_config, manual_config, triage_path, audit / "original_native_context.parquet",
              audit / "event_date_source_prices.parquet", audit / "source_provenance.json"]
    record = {"status": "complete", "purpose": "Corrected held valuations on frozen original forecasts; unresolved rights preserved",
        "valuation_overlay": True, "models_refitted": 0, "forecasts_changed": 0, "original_observed_labels_changed": 0,
        "original_result_directory": str(results), "original_result_manifest_sha256": ORIGINAL_SHA,
        "original_portfolio_dates": len(old_daily), "original_missing_stock_labels": len(missing),
        "original_unavailable_portfolio_dates": int((~old_daily.complete).sum()),
        "repaired_stock_labels": int(ledger.status.eq("repaired").sum()),
        "remaining_unavailable_stock_labels": int(repaired.forward_return.isna().sum()),
        "complete_portfolio_dates": int(daily.complete.sum()), "unavailable_portfolio_dates": int((~daily.complete).sum()),
        "repair_methods": ledger.value_counts(["method", "status"]).reset_index(name="rows").to_dict("records"),
        "independent_object": {"uri": "gs://quant-research-trusty-agility-492109-v0/dataset-platform/curated/equity_us/sp_composite1500_market_daily/v1/tables/daily_bars/part-00000.parquet",
            "generation": "1783344954247444", "sha256": INDEPENDENT_SHA, "whole_object_verified": True,
            "raw_provider_response_bodies_reauthenticated": False},
        "inputs": [{"path": str(p), "sha256": file_hash(p)} for p in inputs], "script_sha256": file_hash(__file__),
        "outputs": [{"path": p.name, "sha256": file_hash(p), "size_bytes": p.stat().st_size} for p in sorted(output.iterdir()) if p.is_file()]}
    write_json(output / "manifest.json", record)
    print(json.dumps({k:v for k,v in record.items() if k not in {"inputs", "outputs", "independent_object"}}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "data/walk_forward/annual_results_v1")
    parser.add_argument("--audit", type=Path, default=ROOT / "data/performance_gap_audit/v1")
    parser.add_argument("--output", type=Path, default=ROOT / "data/walk_forward/valuation_results_v1")
    parser.add_argument("--events", type=Path, default=ROOT / "configs/valuation_events_v1.json")
    parser.add_argument("--manual-quotes", type=Path, default=ROOT / "configs/valuation_manual_quotes_v1.json")
    args = parser.parse_args()
    build(args.results.resolve(), args.audit.resolve(), args.output.resolve(), args.events.resolve(), args.manual_quotes.resolve())
