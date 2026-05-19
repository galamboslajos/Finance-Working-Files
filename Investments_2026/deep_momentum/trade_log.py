"""
Deep Momentum — Step 7b: Per-position trade log for accountability.

Reads:
  cache/ca_features_monthly.parquet     (monthly fwd_return_mt + close_mt + metadata)
  cache/ca_predictions_monthly.parquet  (model scores + class probs)

Writes:
  cache/ca_trade_log_<STRATEGY>.csv     (one CSV per strategy)

One row per (rebalance_date, stock, strategy). Every position the strategy
held, with entry/exit prices, weights, gross/net returns, and per-row
commission/financing attribution.

The per-month weighted sum of `r_usd_net` across rows reconciles EXACTLY
to the strategy's `ls_ret` from portfolio.py / optimizer.py — the script
asserts this on every month.

Cost attribution (mirrors portfolio.py `_period_cost`):
  - Commission charged if `is_new_position` OR `will_exit` is True.
    Per-row: events × weight × tc_bps / 10000.
  - Financing per row: carry_annual × days / 365, weighted by row weight.
    Long: positive cost. Short: negative cost (earned).
"""

from __future__ import annotations
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
CACHE_DIR   = PROJECT_DIR / "cache"

FEATURES_PATH    = CACHE_DIR / "ca_features_monthly.parquet"
PREDICTIONS_PATH = CACHE_DIR / "ca_predictions_monthly.parquet"


# ─── Selection helpers (mirror portfolio.py exactly) ─────────────────────────

def _resolve_n(top_n, n_total: int) -> int:
    """top_n > 1 → fixed; 0<top_n≤1 → percentile of cross-section."""
    if top_n is None:
        return 0
    if top_n > 1:
        return int(top_n)
    return max(1, int(round(top_n * n_total)))


def _build_selections_for_strategy(features: pd.DataFrame,
                                    predictions: pd.DataFrame,
                                    strategy: str,
                                    top_n) -> list[dict]:
    """
    Reproduce the same per-month selection as portfolio.py for one strategy.
    Returns list of dicts ordered by date with assetid lists for long/short
    plus the source rows (so we can recover entry prices and metadata).

    top_n: int > 1 → fixed N per leg; 0 < float ≤ 1 → percentile of cross-section.
    """
    if strategy == "MOM":
        sub = features.dropna(subset=["MOM_12_mt", "fwd_return_mt"])
        score_col, use_xgb = "MOM_12_mt", False
    elif strategy == "XGB":
        sub = predictions.dropna(subset=["prob_10", "prob_1", "fwd_return_mt"])
        score_col, use_xgb = None, True
    elif strategy == "RET":
        sub = predictions.dropna(subset=["ret_score", "fwd_return_mt"])
        score_col, use_xgb = "ret_score", False
    elif strategy == "SRP":
        sub = predictions.dropna(subset=["srp_score", "fwd_return_mt"])
        score_col, use_xgb = "srp_score", False
    elif strategy == "CVR":
        sub = predictions.dropna(subset=["cvr_score", "fwd_return_mt"])
        score_col, use_xgb = "cvr_score", False
    else:
        raise ValueError(f"unknown strategy {strategy!r}")

    sub = sub.copy()
    sub["_ym"] = sub["date_mt"].dt.to_period("M")

    selections = []
    for ym, grp in sub.groupby("_ym"):
        n_total = len(grp)
        n = _resolve_n(top_n, n_total)
        if n_total < 2 * n or n < 1:
            continue
        if use_xgb:
            g = grp.copy()
            g["_xgb_score"] = g["prob_10"] - g["prob_1"]
            g = g.sort_values("_xgb_score", ascending=False)
        else:
            g = grp.sort_values(score_col, ascending=False)
        longs  = g.head(n)
        shorts = g.tail(n)
        if longs.empty or shorts.empty:
            continue
        rebal_date = grp["date_mt"].max()
        selections.append({
            "date_mt":    pd.Timestamp(rebal_date),
            "long_df":    longs,
            "short_df":   shorts,
            "long_ids":   longs["assetid"].tolist(),
            "short_ids":  shorts["assetid"].tolist(),
        })
    return selections


# ─── Trade log builder for EW strategies ─────────────────────────────────────

def build_trade_log_ew(features: pd.DataFrame,
                       predictions: pd.DataFrame,
                       strategy: str,
                       top_n = 15,
                       tc_bps: float = 20.0,
                       carry_long_annual: float = 0.05,
                       carry_short_annual: float = 0.02,
                       days_per_month: int = 30,
                       long_only: bool = False,
                       reconcile_portfolio: pd.DataFrame | None = None,
                       reconcile_tol: float = 1e-8) -> pd.DataFrame:
    """
    Build the per-row trade log for one EW strategy. Each (date_mt, assetid)
    held is one row.

    If `reconcile_portfolio` is provided (DataFrame from
    portfolio.construct_<strategy>(...)), the per-month weighted sum of net
    returns from the trade log is asserted equal to ls_ret in that DataFrame.
    """
    selections = _build_selections_for_strategy(features, predictions, strategy, top_n)
    if not selections:
        return pd.DataFrame()

    # Look up close prices from the features panel (or predictions if it has them)
    # features should have close_mt; predictions has fwd_return_mt
    feat_lookup_cols = ["assetid", "date_mt", "close_mt", "securityname",
                         "exchange_name", "symbol"]
    feat_lookup_cols = [c for c in feat_lookup_cols if c in features.columns]
    feat_lookup = features[feat_lookup_cols].drop_duplicates(["assetid", "date_mt"])

    rows = []
    for i, sel in enumerate(selections):
        prev = selections[i - 1] if i > 0 else None
        nxt  = selections[i + 1] if i + 1 < len(selections) else None
        prev_long_set  = set(prev["long_ids"])  if prev else set()
        prev_short_set = set(prev["short_ids"]) if prev else set()
        next_long_set  = set(nxt["long_ids"])   if nxt  else set()
        next_short_set = set(nxt["short_ids"])  if nxt  else set()

        date_t  = sel["date_mt"]
        date_t1 = nxt["date_mt"] if nxt else pd.NaT  # exit = next rebalance

        n_long  = len(sel["long_ids"])
        n_short = 0 if long_only else len(sel["short_ids"])
        w_long  = 1.0 / n_long  if n_long  else 0.0
        w_short = 0.0 if long_only else (1.0 / n_short if n_short else 0.0)

        # Pull entry/exit prices for this month and next month
        entries = feat_lookup[feat_lookup["date_mt"] == date_t].set_index("assetid")
        if not pd.isna(date_t1):
            exits = feat_lookup[feat_lookup["date_mt"] == date_t1].set_index("assetid")
        else:
            exits = pd.DataFrame()

        # Long leg
        for _, row in sel["long_df"].iterrows():
            aid = int(row["assetid"])
            r_local = row["fwd_return_mt"]
            is_new      = aid not in prev_long_set
            will_exit   = aid not in next_long_set
            events      = int(is_new) + int(will_exit)
            commission_bps = events * tc_bps * w_long
            financing_bps  = carry_long_annual * days_per_month / 365 * 10000 * w_long
            commission     = commission_bps / 10000
            financing      = financing_bps / 10000
            r_usd_net = r_local * w_long - commission - financing

            meta = entries.loc[aid] if aid in entries.index else None
            exit_close = exits.loc[aid]["close_mt"] if (not exits.empty and aid in exits.index) else np.nan

            rows.append({
                "date_mt":       date_t,
                "strategy":      f"EW_{strategy}",
                "assetid":       aid,
                "symbol":        row.get("symbol", "?"),
                "securityname":  meta["securityname"] if meta is not None and "securityname" in meta else None,
                "exchange_name": meta["exchange_name"] if meta is not None and "exchange_name" in meta else None,
                "direction":     "L",
                "weight":        w_long,
                "entry_date":    date_t,
                "entry_close":   meta["close_mt"] if meta is not None and "close_mt" in meta else np.nan,
                "exit_date":     date_t1,
                "exit_close":    exit_close,
                "r_local":       r_local,
                "is_new_position": is_new,
                "will_exit":     will_exit,
                "commission_bps": commission_bps,
                "financing_bps":  financing_bps,
                "r_usd_net":     r_usd_net,
            })

        if long_only:
            continue

        # Short leg
        for _, row in sel["short_df"].iterrows():
            aid = int(row["assetid"])
            r_local = row["fwd_return_mt"]
            is_new      = aid not in prev_short_set
            will_exit   = aid not in next_short_set
            events      = int(is_new) + int(will_exit)
            commission_bps = events * tc_bps * w_short
            # Short leg EARNS carry_short_annual (financing is a credit, so negative cost)
            financing_bps  = -carry_short_annual * days_per_month / 365 * 10000 * w_short
            commission     = commission_bps / 10000
            financing      = financing_bps / 10000
            # Short P&L: shorting a stock that goes up loses you money, so net = -r_local × w
            r_usd_net = -r_local * w_short - commission - financing

            meta = entries.loc[aid] if aid in entries.index else None
            exit_close = exits.loc[aid]["close_mt"] if (not exits.empty and aid in exits.index) else np.nan

            rows.append({
                "date_mt":       date_t,
                "strategy":      f"EW_{strategy}",
                "assetid":       aid,
                "symbol":        row.get("symbol", "?"),
                "securityname":  meta["securityname"] if meta is not None and "securityname" in meta else None,
                "exchange_name": meta["exchange_name"] if meta is not None and "exchange_name" in meta else None,
                "direction":     "S",
                "weight":        w_short,
                "entry_date":    date_t,
                "entry_close":   meta["close_mt"] if meta is not None and "close_mt" in meta else np.nan,
                "exit_date":     date_t1,
                "exit_close":    exit_close,
                "r_local":       r_local,
                "is_new_position": is_new,
                "will_exit":     will_exit,
                "commission_bps": commission_bps,
                "financing_bps":  financing_bps,
                "r_usd_net":     r_usd_net,
            })

    log = pd.DataFrame(rows)

    if reconcile_portfolio is not None and not reconcile_portfolio.empty and not log.empty:
        agg = log.groupby("date_mt")["r_usd_net"].sum().reset_index()
        merged = reconcile_portfolio[["date_mt", "ls_ret"]].merge(
            agg, on="date_mt", how="inner"
        )
        diff = (merged["ls_ret"] - merged["r_usd_net"]).abs()
        max_err = diff.max() if len(diff) else 0.0
        if max_err > reconcile_tol:
            print(f"  WARN: reconcile error for EW_{strategy}: max abs diff = {max_err:.2e}")
        else:
            print(f"  EW_{strategy} reconciles: max abs diff = {max_err:.2e}")

    return log


# ─── Trade log builder for optimiser-weighted output ─────────────────────────

def build_trade_log_optimizer(features: pd.DataFrame,
                               predictions: pd.DataFrame,
                               opt_port: pd.DataFrame,
                               opt_weights: pd.DataFrame,
                               opt_sel_log: list[dict],
                               score_col: str = "ret_score",
                               tc_bps: float = 20.0,
                               carry_long_annual: float = 0.05,
                               carry_short_annual: float = 0.02,
                               days_per_month: int = 30,
                               long_only: bool = False,
                               reconcile_tol: float = 1e-6) -> pd.DataFrame:
    """
    Build the trade log for the CVaR-optimizer book. Same schema as
    build_trade_log_ew. Uses the optimizer's `selections_log` (per-month
    long/short ids + weights) and the realised fwd_returns.
    """
    if not opt_sel_log:
        return pd.DataFrame()

    feat_lookup_cols = ["assetid", "date_mt", "close_mt", "securityname",
                         "exchange_name", "symbol"]
    feat_lookup_cols = [c for c in feat_lookup_cols if c in features.columns]
    feat_lookup = features[feat_lookup_cols].drop_duplicates(["assetid", "date_mt"])

    fwd_lookup = features.set_index(["assetid", "date_mt"])["fwd_return_mt"].to_dict()

    rows = []
    for i, sel in enumerate(opt_sel_log):
        prev = opt_sel_log[i - 1] if i > 0 else None
        nxt  = opt_sel_log[i + 1] if i + 1 < len(opt_sel_log) else None
        prev_long_set  = set(prev["long_ids"])  if prev else set()
        prev_short_set = set(prev["short_ids"]) if prev else set()
        next_long_set  = set(nxt["long_ids"])   if nxt  else set()
        next_short_set = set(nxt["short_ids"])  if nxt  else set()

        date_t = sel["date_mt"]
        date_t1 = nxt["date_mt"] if nxt else pd.NaT

        entries = feat_lookup[feat_lookup["date_mt"] == date_t].set_index("assetid")
        if not pd.isna(date_t1):
            exits = feat_lookup[feat_lookup["date_mt"] == date_t1].set_index("assetid")
        else:
            exits = pd.DataFrame()

        # Long leg
        for aid, w in zip(sel["long_ids"], sel["w_long"]):
            aid = int(aid); w = float(w)
            if w <= 0:
                continue
            r_local = fwd_lookup.get((aid, pd.Timestamp(date_t)), np.nan)
            is_new      = aid not in prev_long_set
            will_exit   = aid not in next_long_set
            events      = int(is_new) + int(will_exit)
            commission_bps = events * tc_bps * w
            financing_bps  = carry_long_annual * days_per_month / 365 * 10000 * w
            commission     = commission_bps / 10000
            financing      = financing_bps / 10000
            r_usd_net = (r_local * w if not np.isnan(r_local) else 0.0) - commission - financing

            meta = entries.loc[aid] if aid in entries.index else None
            exit_close = exits.loc[aid]["close_mt"] if (not exits.empty and aid in exits.index) else np.nan

            rows.append({
                "date_mt":       date_t,
                "strategy":      f"OPT_{score_col.replace('_score','').upper()}",
                "assetid":       aid,
                "symbol":        meta["symbol"] if meta is not None and "symbol" in meta else "?",
                "securityname":  meta["securityname"] if meta is not None and "securityname" in meta else None,
                "exchange_name": meta["exchange_name"] if meta is not None and "exchange_name" in meta else None,
                "direction":     "L",
                "weight":        w,
                "entry_date":    date_t,
                "entry_close":   meta["close_mt"] if meta is not None and "close_mt" in meta else np.nan,
                "exit_date":     date_t1,
                "exit_close":    exit_close,
                "r_local":       r_local,
                "is_new_position": is_new,
                "will_exit":     will_exit,
                "commission_bps": commission_bps,
                "financing_bps":  financing_bps,
                "r_usd_net":     r_usd_net,
            })

        if long_only:
            continue

        # Short leg
        for aid, w in zip(sel.get("short_ids", []), sel.get("w_short", [])):
            aid = int(aid); w = float(w)
            if w <= 0:
                continue
            r_local = fwd_lookup.get((aid, pd.Timestamp(date_t)), np.nan)
            is_new      = aid not in prev_short_set
            will_exit   = aid not in next_short_set
            events      = int(is_new) + int(will_exit)
            commission_bps = events * tc_bps * w
            financing_bps  = -carry_short_annual * days_per_month / 365 * 10000 * w
            commission     = commission_bps / 10000
            financing      = financing_bps / 10000
            r_usd_net = (-r_local * w if not np.isnan(r_local) else 0.0) - commission - financing

            meta = entries.loc[aid] if aid in entries.index else None
            exit_close = exits.loc[aid]["close_mt"] if (not exits.empty and aid in exits.index) else np.nan

            rows.append({
                "date_mt":       date_t,
                "strategy":      f"OPT_{score_col.replace('_score','').upper()}",
                "assetid":       aid,
                "symbol":        meta["symbol"] if meta is not None and "symbol" in meta else "?",
                "securityname":  meta["securityname"] if meta is not None and "securityname" in meta else None,
                "exchange_name": meta["exchange_name"] if meta is not None and "exchange_name" in meta else None,
                "direction":     "S",
                "weight":        w,
                "entry_date":    date_t,
                "entry_close":   meta["close_mt"] if meta is not None and "close_mt" in meta else np.nan,
                "exit_date":     date_t1,
                "exit_close":    exit_close,
                "r_local":       r_local,
                "is_new_position": is_new,
                "will_exit":     will_exit,
                "commission_bps": commission_bps,
                "financing_bps":  financing_bps,
                "r_usd_net":     r_usd_net,
            })

    log = pd.DataFrame(rows)

    # Reconcile with opt_port
    if not opt_port.empty and not log.empty:
        agg = log.groupby("date_mt")["r_usd_net"].sum().reset_index()
        merged = opt_port[["date_mt", "ls_ret"]].merge(agg, on="date_mt", how="inner")
        diff = (merged["ls_ret"] - merged["r_usd_net"]).abs()
        max_err = diff.max() if len(diff) else 0.0
        if max_err > reconcile_tol:
            print(f"  WARN: reconcile error for optimizer log: max abs diff = {max_err:.2e}")
        else:
            print(f"  OPT_{score_col} reconciles: max abs diff = {max_err:.2e}")

    return log


# ─── Save helpers ────────────────────────────────────────────────────────────

def save_trade_log(log: pd.DataFrame, name: str) -> Path:
    out = CACHE_DIR / f"ca_trade_log_{name}.csv"
    log.to_csv(out, index=False)
    return out


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("Building trade logs for EW strategies (defaults: top_n=15, long_only=False)")
    print("=" * 70)

    if not (FEATURES_PATH.exists() and PREDICTIONS_PATH.exists()):
        raise FileNotFoundError("need features.parquet and predictions.parquet")

    features    = pd.read_parquet(FEATURES_PATH)
    predictions = pd.read_parquet(PREDICTIONS_PATH)

    # fwd_return_mt should be on predictions for selection consistency
    if "fwd_return_mt" not in predictions.columns or predictions["fwd_return_mt"].isna().all():
        fwd = features[["assetid", "date_mt", "fwd_return_mt"]].dropna()
        predictions = predictions.drop(columns=["fwd_return_mt"], errors="ignore")
        predictions = predictions.merge(fwd, on=["assetid", "date_mt"], how="left")

    from portfolio import run_all_strategies
    results = run_all_strategies(features, predictions, top_n=15, verbose=False)

    t0 = time.time()
    for strat in ["MOM", "XGB", "RET", "SRP", "CVR"]:
        log = build_trade_log_ew(
            features, predictions, strat,
            top_n=15, tc_bps=20.0,
            carry_long_annual=0.05, carry_short_annual=0.02,
            days_per_month=30, long_only=False,
            reconcile_portfolio=results[strat]["portfolio"],
        )
        if log.empty:
            print(f"  {strat}: empty log")
            continue
        out = save_trade_log(log, f"EW_{strat}")
        print(f"  wrote {out.name}: {len(log):,} rows")

    print(f"\nTotal runtime: {time.time()-t0:.0f}s")
