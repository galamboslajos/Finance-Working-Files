#!/usr/bin/env python3
"""Build the paper's eight causal momentum features and first-anchor eligibility.

The reviewed close mask is mandatory. An unavailable exchange-session close
resets all indicator state; neither prices nor returns are filled. Features are
calculated on native price histories, then joined to every dated PIT member.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import exchange_calendars as xcals
import numpy as np
import pandas as pd

from audit_first_window import read_slice
from prepare_sp500_daily import file_hash, require, write_parquet
from review_minor_closes import load_feature_prices


ROOT = Path(__file__).resolve().parent
HORIZONS = (1, 21, 63, 126, 252)
PAIRS = ((8, 24), (16, 48), (32, 96))
LOOKBACKS = (252, 504, 756, 1008, 1260)
FEATURES = [f"return_{h}" for h in HORIZONS] + [f"macd_{s}_{l}" for s, l in PAIRS]
POLICY = {
    "id": "network-momentum-eight-features-v1",
    "return_horizons_sessions": list(HORIZONS),
    "cumulative_returns": "adjusted-close endpoint ratio minus one; every intervening close required",
    "volatility": {"alpha": 2 / 61, "adjust": True, "population_variance": True,
                   "min_return_observations": 60, "includes_current_return": True},
    "macd": {"pairs": PAIRS, "ema_alpha": "1/J", "seed": "first close of contiguous run",
             "ema_adjust": False, "price_std_window": 63, "normalized_std_window": 252,
             "rolling_ddof": 0, "rolling_min_periods": "full window"},
    "winsorization": {"halflife_sessions": 252, "std_multiple": 5, "adjust": True,
                      "population_variance": True, "min_observations": 1,
                      "includes_current_observation": True, "moments_input": "raw unclipped feature"},
    "gap_treatment": "reset all indicator and clipping state after any missing or withheld session close",
    "denominators": "strictly positive finite; otherwise feature missing",
    "graph_inputs": "clipped features, without additional graph standardization",
    "price_source": "provisional reviewed close layer; original execution flags preserved",
    "as_of": "end of signal session; no future prices or labels",
    "paper_unspecified_choices_frozen_for_pilot": [
        "simple cumulative returns", "60-return volatility minimum", "rolling ddof=0",
        "EMA first-price seed", "current-inclusive clipping with one-observation minimum",
        "full state reset after unavailable closes"],
}


def contiguous_slices(valid):
    valid = np.asarray(valid, dtype=bool)
    edges = np.diff(np.r_[False, valid, False].astype(np.int8))
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))


def compute_features(close):
    """One exact-session series, with unavailable closes represented by NaN."""
    close = pd.Series(close, copy=True, dtype=float)
    valid = np.isfinite(close.to_numpy()) & close.gt(0).to_numpy()
    out = pd.DataFrame(index=close.index)
    out["feature_adj_close"] = close.where(valid)
    out["feature_return_1d"] = np.nan
    out["daily_volatility"] = np.nan
    out["contiguous_price_sessions"] = np.zeros(len(close), dtype=np.int32)
    for name in FEATURES:
        out[name] = np.nan
        out["raw_" + name] = np.nan
    for start, stop in contiguous_slices(valid):
        p = close.iloc[start:stop]
        r = p / p.shift(1) - 1
        vol = r.ewm(alpha=2 / 61, adjust=True, min_periods=60).std(bias=True)
        vol = vol.where(np.isfinite(vol) & vol.gt(0))
        raw = pd.DataFrame(index=p.index)
        for horizon in HORIZONS:
            raw[f"return_{horizon}"] = (p / p.shift(horizon) - 1) / (vol * np.sqrt(horizon))
        price_std = p.rolling(63, min_periods=63).std(ddof=0)
        price_std = price_std.where(price_std.gt(0))
        for short, long in PAIRS:
            macd = (p.ewm(alpha=1 / short, adjust=False).mean()
                    - p.ewm(alpha=1 / long, adjust=False).mean()) / price_std
            denom = macd.rolling(252, min_periods=252).std(ddof=0)
            raw[f"macd_{short}_{long}"] = macd / denom.where(denom.gt(0))
        raw = raw.where(np.isfinite(raw))
        moments = raw.ewm(halflife=252, adjust=True, min_periods=1)
        mean, std = moments.mean(), moments.std(bias=True)
        clipped = raw.clip(lower=mean - 5 * std, upper=mean + 5 * std, axis=None)
        out.loc[p.index, "feature_return_1d"] = r
        out.loc[p.index, "daily_volatility"] = vol
        out.loc[p.index, "contiguous_price_sessions"] = np.arange(1, len(p) + 1, dtype=np.int32)
        out.loc[p.index, FEATURES] = clipped.to_numpy()
        out.loc[p.index, ["raw_" + x for x in FEATURES]] = raw.to_numpy()
    out["feature_ready"] = np.isfinite(out[FEATURES]).all(axis=1)
    out["feature_status"] = np.select(
        [~valid, out.contiguous_price_sessions.lt(314), out.daily_volatility.isna(), out.feature_ready],
        ["close_unavailable", "indicator_warmup", "volatility_unavailable", "ready"],
        default="macd_denominator_unavailable")
    return out


def history_eligibility(cube, anchor, lookbacks=LOOKBACKS):
    """cube: session x ordered anchor member x feature. Keep every member row."""
    require(cube.shape[1] == len(anchor), "Anchor alignment mismatch")
    finite = np.isfinite(cube).all(axis=2)
    ledger = anchor.copy().reset_index(drop=True)
    for window in lookbacks:
        require(window <= len(cube), "History is shorter than requested lookback")
        count = finite[-window:].sum(axis=0)
        ledger[f"valid_feature_sessions_{window}"] = count
        eligible = ledger.identity_eligible & (count == window)
        ledger[f"eligible_{window}"] = eligible
        ledger[f"reason_{window}"] = np.select(
            [~ledger.identity_eligible, eligible, ~finite[-1]],
            ["identity_withheld", "eligible", "current_features_unavailable"],
            default="incomplete_feature_history")
    return ledger


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def build(layer, dataset, output):
    require(not output.exists(), "Output exists; choose a new version")
    source_manifest = json.loads((layer / "manifest.json").read_text())
    source_dataset_manifest = json.loads((dataset / "manifest.json").read_text())
    require(source_manifest["status"] == source_dataset_manifest["status"] == "complete", "Incomplete input")
    require(source_manifest["source_manifest_sha256"] == file_hash(dataset / "manifest.json"),
            "Reviewed price layer refers to another source dataset")
    for item in source_manifest["outputs"]:
        if item["path"].startswith("feature_prices/"):
            p = layer / item["path"]
            require(p.stat().st_size == item["size_bytes"] and file_hash(p) == item["sha256"],
                    f"Reviewed price fingerprint changed: {p}")
    start, end = source_manifest["start"], source_manifest["end"]
    sessions = xcals.get_calendar("XNYS", start=start, end=end).sessions.tz_localize(None)
    prices = load_feature_prices(start, end, columns=["date", "intrinio_security_id", "adj_close",
        "feature_close_eligible", "feature_close_status", "execution_bar_eligible", "feature_return_1d"], layer=layer)
    prices = prices.loc[prices.date.isin(sessions)].copy()
    require(not prices.duplicated(["intrinio_security_id", "date"]).any(), "Duplicate native price keys")
    anchor_cols = ["date", "source_security_id", "intrinio_security_id", "membership_ticker", "identity_eligible", "identity_status"]
    anchor = read_slice(dataset, source_dataset_manifest, "panel", anchor_cols, end, end)
    anchor = anchor.sort_values("source_security_id").reset_index(drop=True)
    admitted_anchor = anchor.loc[anchor.identity_eligible]
    require(not admitted_anchor.intrinio_security_id.duplicated().any(), "Admitted anchor identity collision")
    output.mkdir(parents=True)
    write_json(output / "policy.json", POLICY)
    histories, clip_counts, checked_returns = [], dict.fromkeys(FEATURES, 0), 0
    for native, group in prices.groupby("intrinio_security_id", sort=True):
        grid = group.set_index("date").reindex(sessions)
        close = grid.adj_close.where(grid.feature_close_eligible.eq(True))
        computed = compute_features(close)
        np.testing.assert_allclose(computed.feature_return_1d, grid.feature_return_1d,
                                   rtol=1e-12, atol=1e-14, equal_nan=True)
        checked_returns += int(computed.feature_return_1d.notna().sum())
        computed["intrinio_security_id"] = native
        computed["feature_close_status"] = grid.feature_close_status.fillna("absent_price")
        computed["execution_bar_eligible"] = grid.execution_bar_eligible.eq(True)
        for name in FEATURES:
            clip_counts[name] += int((computed[name].notna()
                & computed[name].ne(computed["raw_" + name])).sum())
        histories.append(computed.rename_axis("date").reset_index())
    features = pd.concat(histories, ignore_index=True)
    for year, annual in features.groupby(features.date.dt.year, sort=True):
        write_parquet(annual, output / "native_features" / f"year={year}" / "part-00000.parquet")
    print(json.dumps({"stage": "features_built", "native_securities": len(histories), "rows": len(features)}), flush=True)
    # All native histories, including pre-entry dates, are distinct from the PIT
    # population. A dated source-member join always determines availability.
    ready_by_year, ready_by_date, pit_rows = [], [], 0
    for year in range(max(2011, int(start[:4])), int(end[:4]) + 1):
        annual_start, annual_end = f"{year}-01-01", min(f"{year}-12-31", end)
        panel = read_slice(dataset, source_dataset_manifest, "panel", anchor_cols, annual_start, annual_end)
        annual = features.loc[features.date.dt.year.eq(year)]
        merged = panel.merge(annual, how="left", on=["intrinio_security_id", "date"], validate="many_to_one")
        require(len(merged) == len(panel) and not merged.duplicated(["source_security_id", "date"]).any(),
                "PIT rows changed during feature join")
        # Withheld identities cannot use values even if a raw native ID exists.
        withheld = ~merged.identity_eligible
        numeric = FEATURES + ["raw_" + x for x in FEATURES] + ["feature_adj_close", "feature_return_1d", "daily_volatility"]
        merged.loc[withheld, numeric] = np.nan
        merged["feature_ready"] = merged.identity_eligible & merged.feature_ready.eq(True)
        merged.loc[withheld, "feature_status"] = "identity_withheld"
        merged["feature_status"] = merged.feature_status.fillna("no_native_price_history")
        write_parquet(merged, output / "pit_features" / f"year={year}" / "part-00000.parquet")
        pit_rows += len(merged)
        ready_by_year.append({"year": year, "member_days": len(merged), "feature_ready_member_days": int(merged.feature_ready.sum()),
                              "feature_ready_fraction": float(merged.feature_ready.mean())})
        daily = merged.groupby("date").agg(members=("source_security_id", "size"), feature_ready=("feature_ready", "sum"))
        ready_by_date.append(daily.reset_index())
    pd.DataFrame(ready_by_year).to_csv(output / "coverage_annual.csv", index=False)
    pd.concat(ready_by_date, ignore_index=True).to_csv(output / "coverage_daily.csv", index=False)
    native_order = sorted(features.intrinio_security_id.unique())
    matrices = {native: frame.set_index("date")[FEATURES].to_numpy()
                for native, frame in features.groupby("intrinio_security_id", sort=True)}
    cube = np.full((len(sessions), len(anchor), len(FEATURES)), np.nan)
    for j, row in enumerate(anchor.itertuples()):
        if row.identity_eligible and row.intrinio_security_id in matrices:
            cube[:, j, :] = matrices[row.intrinio_security_id]
    ledger = history_eligibility(cube, anchor)
    ledger.to_csv(output / "anchor_eligibility.csv", index=False)
    common = ledger.eligible_1260.to_numpy()
    require(common.sum() >= 2, "Too few common nodes for network pilot")
    np.savez_compressed(output / "anchor_feature_history.npz", features=cube[-1260:, common, :],
        dates=sessions[-1260:].to_numpy(dtype="datetime64[ns]"), feature_names=np.array(FEATURES),
        source_security_ids=anchor.loc[common, "source_security_id"].to_numpy(dtype=str),
        intrinio_security_ids=anchor.loc[common, "intrinio_security_id"].to_numpy(dtype=str),
        membership_tickers=anchor.loc[common, "membership_ticker"].to_numpy(dtype=str))
    # Recompute representative full histories with independently shortened input;
    # compare every output value at the cutoff to detect future dependence.
    prefix_checks = 0
    for native in native_order[::max(1, len(native_order) // 12)]:
        frame = features.loc[features.intrinio_security_id.eq(native)].set_index("date")
        for cutoff in (600, 1100):
            short = compute_features(frame.feature_adj_close.iloc[:cutoff])
            np.testing.assert_allclose(short[FEATURES + ["daily_volatility"]],
                frame.iloc[:cutoff][FEATURES + ["daily_volatility"]], rtol=0, atol=0, equal_nan=True)
            prefix_checks += 1
    validation = {"status": "passed", "daily_returns_checked_against_review_layer": checked_returns,
        "historical_prefix_checks": prefix_checks, "pit_rows_preserved": pit_rows,
        "common_history_all_finite": bool(np.isfinite(cube[-1260:, common]).all()),
        "no_filled_prices_or_returns": True, "execution_flags_preserved": True}
    write_json(output / "validation.json", validation)
    summary = {"status": "complete", "policy_id": POLICY["id"], "start": start, "end": end,
        "anchor_date": end, "history_start_for_1260_features": str(sessions[-1260].date()),
        "native_securities": len(histories), "native_session_rows": len(features), "pit_member_rows": pit_rows,
        "anchor_members": len(anchor), "eligible_by_lookback": {str(w): int(ledger[f"eligible_{w}"].sum()) for w in LOOKBACKS},
        "common_ensemble_nodes": int(common.sum()), "clipped_observations_by_feature": clip_counts,
        "input_review_manifest_sha256": file_hash(layer / "manifest.json"),
        "input_dataset_manifest_sha256": file_hash(dataset / "manifest.json"),
        "script_sha256": file_hash(__file__), "policy_sha256": file_hash(output / "policy.json"),
        "versions": {"numpy": np.__version__, "pandas": pd.__version__},
        "limits": ["Features use the existing provisional close-only and exploratory identity policies.",
                   "This is a first-window feature build, not fitted regressions or a performance backtest.",
                   "Graph history includes the anchor close; first 1260-feature date precedes 2011 by one session."]}
    summary["outputs"] = [{"path": p.relative_to(output).as_posix(), "size_bytes": p.stat().st_size, "sha256": file_hash(p)}
                          for p in sorted(output.rglob("*")) if p.is_file()]
    write_json(output / "manifest.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in {"outputs", "limits"}}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=Path, default=ROOT / "data/feature_close_review/v1")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/sp500_daily")
    parser.add_argument("--output", type=Path, default=ROOT / "data/momentum_features/v1")
    args = parser.parse_args()
    build(args.layer.resolve(), args.dataset.resolve(), args.output.resolve())
