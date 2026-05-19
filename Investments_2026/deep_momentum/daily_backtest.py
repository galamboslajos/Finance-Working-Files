"""
Deep Momentum — Step 8: Daily-frequency backtester.

Replaces the monthly-aggregated P&L view with realistic daily mark-to-market.
This is the **canonical performance reporter** — monthly stats from
portfolio.py / optimizer.py are useful for selection diagnostics only.

Inputs:
  cache/ca_equities_daily.parquet     daily OHLCV panel (data_load.py)
  monthly_selections + weights from EW (portfolio.py) or OPT (optimizer.py)

Outputs (returned, not written by default):
  daily_results: DataFrame with columns
      date, n_positions, daily_ret, equity, peak, drawdown, commission_$, financing_$
  summary: dict with realistic Sharpe, ann.ret, ann.vol, MaxDD, win rate, VaR95

Architecture:
  1. Each rebalance month's selections are observed at close of month-end day T.
  2. Execution happens at close of NEXT trading day T+1 (1-day lag).
  3. Holdings + weights become "live" from T+1 close → next rebalance's T+1 close.
  4. Daily mark-to-market on every trading day in between using actual stock
     closes. If a stock's data ends mid-month (delisting), forward-fill for up
     to 3 trading days, then **mark its price to zero** (lose the position).
  5. Commission charged on $ traded at each rebalance day (open positions +
     close positions). Financing accrues daily on overnight positions.

All cost params + LONG_ONLY are kwargs.
"""

from __future__ import annotations
import time
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
CACHE_DIR   = PROJECT_DIR / "cache"

DAILY_PATH       = CACHE_DIR / "ca_equities_daily.parquet"
FEATURES_PATH    = CACHE_DIR / "ca_features_monthly.parquet"
PREDICTIONS_PATH = CACHE_DIR / "ca_predictions_monthly.parquet"


# Defaults
STARTING_CAPITAL          = 100_000.0  # USD-equivalent (CAD here, fine for relative reporting)
EXECUTION_LAG_DAYS        = 1
DELIST_FORWARD_FILL_DAYS  = 3      # forward-fill missing prices this many days, then mark to zero
TRADING_DAYS_PER_YEAR     = 252


# ─── Build the universal price panel ─────────────────────────────────────────

def build_daily_close_panel(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot daily df to a (date × assetid) close-price panel.
    Used by the backtester to mark portfolios daily.
    """
    df = daily[["assetid", "date", "close"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    panel = df.pivot_table(index="date", columns="assetid", values="close")
    return panel.sort_index()


def apply_delist_logic(prices: pd.Series, max_ffill: int = DELIST_FORWARD_FILL_DAYS) -> pd.Series:
    """
    For one stock's price series sliced over the holding window:
    - Forward-fill missing prices for up to `max_ffill` consecutive trading days.
    - After that many missing days in a row, mark the price to zero (delisting).
    """
    s = prices.copy()
    if s.isna().all():
        return s.fillna(0.0)
    # Forward-fill with limit
    s = s.ffill(limit=max_ffill)
    # Any remaining NaN means we went > max_ffill days without data → mark to zero
    s = s.fillna(0.0)
    return s


# ─── EW selection harvest ────────────────────────────────────────────────────

def _resolve_n(top_n, n_total: int) -> int:
    """top_n > 1 → fixed; 0<top_n≤1 → percentile of cross-section."""
    if top_n is None:
        return 0
    if top_n > 1:
        return int(top_n)
    return max(1, int(round(top_n * n_total)))


def _ew_selections_from_strategy(features: pd.DataFrame,
                                   predictions: pd.DataFrame,
                                   strategy: str,
                                   top_n) -> list[dict]:
    """
    Build EW per-month ranked candidate lists.
    Returns list ordered by rebalance date.

    top_n: int > 1 → fixed N per leg; 0 < float ≤ 1 → percentile of cross-section.
    The daily executor fills down these ranked lists until it finds `target_n`
    executable names at the actual execution close.
    """
    if strategy == "MOM":
        sub = features.dropna(subset=["MOM_12_mt"]).copy()
        score_col, use_xgb = "MOM_12_mt", False
    elif strategy == "XGB":
        sub = predictions.dropna(subset=["prob_10", "prob_1"]).copy()
        score_col, use_xgb = None, True
    elif strategy == "RET":
        sub = predictions.dropna(subset=["ret_score"]).copy()
        score_col, use_xgb = "ret_score", False
    elif strategy == "SRP":
        sub = predictions.dropna(subset=["srp_score"]).copy()
        score_col, use_xgb = "srp_score", False
    elif strategy == "CVR":
        sub = predictions.dropna(subset=["cvr_score"]).copy()
        score_col, use_xgb = "cvr_score", False
    else:
        raise ValueError(f"unknown strategy {strategy!r}")

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
        long_candidates = g["assetid"].astype(int).tolist()
        short_candidates = g["assetid"].astype(int).iloc[::-1].tolist()
        if not long_candidates or not short_candidates:
            continue
        rebal_date = grp["date_mt"].max()
        selections.append({
            "date_mt":          pd.Timestamp(rebal_date),
            "target_n_long":    n,
            "target_n_short":   n,
            "long_candidates":  long_candidates,
            "short_candidates": short_candidates,
            # Backwards-compatible seed lists for diagnostics/older callers.
            "long_ids":         long_candidates[:n],
            "short_ids":        short_candidates[:n],
            "w_long":           [1.0 / n] * n,
            "w_short":          [1.0 / n] * n,
        })
    return selections


# ─── Daily backtest engine ───────────────────────────────────────────────────

def backtest_daily(selections: list[dict],
                    close_panel: pd.DataFrame,
                    starting_capital: float = STARTING_CAPITAL,
                    tc_bps: float = 20.0,
                    carry_long_annual: float = 0.05,
                    carry_short_annual: float = 0.02,
                    execution_lag_days: int = EXECUTION_LAG_DAYS,
                    delist_ffill_days: int = DELIST_FORWARD_FILL_DAYS,
                    long_only: bool = False,
                    verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    """
    Run the daily backtest.

    Args:
        selections: list of dicts ordered by date with:
                    {date_mt, long_ids, short_ids, w_long, w_short}
        close_panel: (date × assetid) adjusted close prices
        starting_capital: initial portfolio value
        tc_bps: commission, bps of $ traded (one-way)
        carry_long_annual / carry_short_annual: annualized financing rates
        execution_lag_days: how many trading days after the signal we execute
        delist_ffill_days: forward-fill missing prices this many days, then 0
        long_only: if True, ignore short_ids/w_short entirely

    Returns:
        daily_df: per-day DataFrame (date, n_positions, daily_ret, equity, peak,
                  drawdown, commission_$, financing_$)
        summary:  dict of realistic performance metrics
    """
    if not selections:
        return pd.DataFrame(), {}

    if verbose:
        print(f"  Selections to process: {len(selections):,}")
        print(f"  Daily panel: {close_panel.shape[0]:,} days × {close_panel.shape[1]:,} assetids")

    panel_dates = close_panel.index
    panel_index_of = {d: i for i, d in enumerate(panel_dates)}
    last_valid_dates = close_panel.apply(lambda s: s.last_valid_index()).to_dict()

    # Find the execution date for each rebalance (T + execution_lag trading days)
    def _exec_date(signal_date):
        idx = panel_dates.searchsorted(signal_date, side="right")
        target_idx = idx + execution_lag_days - 1  # first trading day strictly after signal_date
        if target_idx >= len(panel_dates):
            return None
        return panel_dates[target_idx]

    # Build per-rebalance execution schedule
    schedule = []
    for sel in selections:
        ed = _exec_date(sel["date_mt"])
        if ed is None:
            continue
        schedule.append({**sel, "exec_date": ed})

    if not schedule:
        return pd.DataFrame(), {}

    # Backtest loop: iterate trading days from first exec date to end of panel
    start_idx = panel_dates.searchsorted(schedule[0]["exec_date"])
    end_idx   = len(panel_dates) - 1
    backtest_days = panel_dates[start_idx:end_idx + 1]

    # Holdings state: dict assetid → {direction: 'L'/'S', value: $}
    holdings = {}
    equity = starting_capital
    rows = []

    sched_idx = 0
    next_rebal_exec = schedule[sched_idx]["exec_date"]
    next_rebal_sel  = schedule[sched_idx]

    t0 = time.time()
    last_print = t0

    for day_i, d in enumerate(backtest_days):
        equity_start_of_day = equity
        is_rebal = (d == next_rebal_exec)

        # ── 1. Daily mark-to-market old holdings through today's close ──
        # If today is a rebalance day, these are still the holdings that were
        # live overnight. New targets are installed only after this close.
        daily_pnl = 0.0
        financing_dollar = 0.0
        new_holdings = {}
        if day_i > 0 and holdings:
            for aid, h in holdings.items():
                # Get this day's price for the stock. The prior close is stored
                # on the holding so missing bars can be handled consistently.
                this_close = close_panel.iloc[panel_index_of[d]].get(aid, np.nan)
                prev_close = h.get("last_close", np.nan)
                missing_days = int(h.get("missing_days", 0))

                # Apply delist logic: forward-fill for N missing trading days,
                # then mark to zero so the final loss/gain is booked once.
                if pd.isna(this_close):
                    missing_days += 1
                    last_valid_date = last_valid_dates.get(aid)
                    is_past_final_bar = (
                        last_valid_date is not None and d > last_valid_date
                    )
                    if missing_days <= delist_ffill_days or not is_past_final_bar:
                        this_close = prev_close
                    else:
                        this_close = 0.0
                else:
                    missing_days = 0

                if pd.isna(prev_close) or prev_close == 0:
                    daily_ret_i = 0.0
                else:
                    daily_ret_i = this_close / prev_close - 1

                v0 = h["value"]
                if h["direction"] == "L":
                    pnl_i = v0 * daily_ret_i
                    # Long financing: daily cost
                    fin_i = v0 * carry_long_annual / 365
                else:
                    pnl_i = -v0 * daily_ret_i
                    # Short financing: daily earn (negative cost)
                    fin_i = -v0 * carry_short_annual / 365
                daily_pnl   += pnl_i
                financing_dollar += fin_i
                new_v = v0 + pnl_i - fin_i
                new_holdings[aid] = {
                    "direction":    h["direction"],
                    "value":        new_v,
                    "last_close":   this_close,
                    "missing_days": missing_days,
                }
            holdings = new_holdings

        equity     += daily_pnl - financing_dollar  # commission already deducted on rebal day

        # ── 2. Rebalance at today's close after old holdings are marked ──
        commission_dollar = 0.0
        if is_rebal:
            target_positions = {}

            # Determine the leg notional from post-MTM, pre-commission equity.
            if long_only:
                long_notional = equity
                short_notional = 0.0
            else:
                # 50/50 of equity across legs (dollar-neutral L/S construction)
                long_notional  = equity / 2
                short_notional = equity / 2

            long_candidates = next_rebal_sel.get("long_candidates", next_rebal_sel["long_ids"])
            short_candidates = next_rebal_sel.get("short_candidates", next_rebal_sel["short_ids"])
            target_n_long = int(next_rebal_sel.get("target_n_long", len(next_rebal_sel["long_ids"])))
            target_n_short = 0 if long_only else int(next_rebal_sel.get("target_n_short", len(next_rebal_sel["short_ids"])))

            selected_longs = []
            selected_shorts = []

            for aid in long_candidates:
                aid_i = int(aid)
                px = close_panel.iloc[panel_index_of[d]].get(aid_i, np.nan)
                if pd.isna(px) or px <= 0:
                    continue
                selected_longs.append((aid_i, px))
                if len(selected_longs) >= target_n_long:
                    break

            n_exec_long = len(selected_longs)
            if n_exec_long > 0:
                w_long_exec = 1.0 / n_exec_long
                for aid_i, px in selected_longs:
                    target_positions[aid_i] = {
                        "direction":    "L",
                        "value":        w_long_exec * long_notional,
                        "last_close":   px,
                        "missing_days": 0,
                    }

            if not long_only:
                for aid in short_candidates:
                    aid_i = int(aid)
                    if aid_i in target_positions:
                        continue
                    px = close_panel.iloc[panel_index_of[d]].get(aid_i, np.nan)
                    if pd.isna(px) or px <= 0:
                        continue
                    selected_shorts.append((aid_i, px))
                    if len(selected_shorts) >= target_n_short:
                        break

                n_exec_short = len(selected_shorts)
                if n_exec_short > 0:
                    w_short_exec = 1.0 / n_exec_short
                    for aid_i, px in selected_shorts:
                        target_positions[aid_i] = {
                            "direction":    "S",
                            "value":        w_short_exec * short_notional,
                            "last_close":   px,
                            "missing_days": 0,
                        }

            trade_value = 0.0
            for aid, tgt in target_positions.items():
                curr_v = holdings.get(aid, {}).get("value", 0.0)
                trade_value += abs(tgt["value"] - curr_v)
            for aid, h in holdings.items():
                if aid not in target_positions:
                    trade_value += abs(h["value"])
            commission_dollar = trade_value * tc_bps / 10000

            equity -= commission_dollar
            holdings = target_positions

            sched_idx += 1
            if sched_idx < len(schedule):
                next_rebal_exec = schedule[sched_idx]["exec_date"]
                next_rebal_sel  = schedule[sched_idx]
            else:
                next_rebal_exec = None
                next_rebal_sel  = None

        daily_ret_p = (equity / equity_start_of_day) - 1 if equity_start_of_day > 0 else 0.0

        rows.append({
            "date":           d,
            "n_positions":    len(holdings),
            "is_rebal_day":   is_rebal,
            "daily_ret":      daily_ret_p,
            "equity":         equity,
            "commission_$":   commission_dollar,
            "financing_$":    financing_dollar,
        })

        if verbose and (time.time() - last_print > 30):
            print(f"    {d.date()}  {day_i}/{len(backtest_days)}  equity={equity:,.0f}  ({time.time()-t0:.0f}s)")
            last_print = time.time()

    daily_df = pd.DataFrame(rows)
    daily_df["peak"]     = daily_df["equity"].cummax()
    daily_df["drawdown"] = (daily_df["equity"] / daily_df["peak"]) - 1

    # Summary metrics
    n_days = len(daily_df)
    if n_days < 2:
        return daily_df, {}

    final_eq = daily_df["equity"].iloc[-1]
    total_ret = final_eq / starting_capital - 1
    days_actual = n_days
    ann_ret = (final_eq / starting_capital) ** (TRADING_DAYS_PER_YEAR / days_actual) - 1
    daily_returns = daily_df["daily_ret"]
    ann_vol = daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe  = (daily_returns.mean() * TRADING_DAYS_PER_YEAR) / ann_vol if ann_vol > 0 else 0.0
    max_dd  = daily_df["drawdown"].min()
    win_rate = (daily_returns > 0).mean()
    var95   = np.percentile(daily_returns, 5)

    summary = {
        "starting_capital":   starting_capital,
        "final_equity":       float(final_eq),
        "total_return":       float(total_ret),
        "annualized_return":  float(ann_ret),
        "annualized_vol":     float(ann_vol),
        "sharpe_ratio":       float(sharpe),
        "max_drawdown":       float(max_dd),
        "win_rate":           float(win_rate),
        "var_95_daily":       float(var95),
        "n_trading_days":     int(days_actual),
        "n_rebalances":       int(daily_df["is_rebal_day"].sum()),
        "avg_positions":      float(daily_df["n_positions"].mean()),
        "total_commission_$": float(daily_df["commission_$"].sum()),
        "total_financing_$":  float(daily_df["financing_$"].sum()),
    }
    return daily_df, summary


# ─── Convenience wrappers ────────────────────────────────────────────────────

def backtest_ew_strategy(features: pd.DataFrame,
                          predictions: pd.DataFrame,
                          daily: pd.DataFrame,
                          strategy: str,
                          top_n = 15,
                          **kwargs) -> tuple[pd.DataFrame, dict]:
    """Daily backtest for an EW strategy (MOM/XGB/RET/SRP/CVR)."""
    selections = _ew_selections_from_strategy(features, predictions, strategy, top_n)
    panel = build_daily_close_panel(daily)
    return backtest_daily(selections, panel, **kwargs)


def backtest_optimizer_selections(opt_sel_log: list[dict],
                                    daily: pd.DataFrame,
                                    **kwargs) -> tuple[pd.DataFrame, dict]:
    """
    Daily backtest using the Mean-CVaR optimizer's selections + weights.
    opt_sel_log comes from run_two_leg_optimizer's third return value.
    """
    # Normalize: opt_sel_log uses 'long_ids', 'short_ids', 'w_long', 'w_short', 'date_mt'
    selections = [{
        "date_mt":   pd.Timestamp(sel["date_mt"]),
        "long_ids":  [int(a) for a in sel.get("long_ids", [])],
        "short_ids": [int(a) for a in sel.get("short_ids", [])],
        "w_long":    list(sel.get("w_long", [])),
        "w_short":   list(sel.get("w_short", [])),
    } for sel in opt_sel_log]
    panel = build_daily_close_panel(daily)
    return backtest_daily(selections, panel, **kwargs)


def print_summary(name: str, summary: dict):
    """One-line summary for a daily backtest."""
    if not summary:
        print(f"  {name}: empty")
        return
    print(f"  {name}: final ${summary['final_equity']:>14,.0f} | "
          f"ann.ret {summary['annualized_return']:>7.1%} | "
          f"ann.vol {summary['annualized_vol']:>7.1%} | "
          f"sharpe {summary['sharpe_ratio']:>5.2f} | "
          f"MaxDD {summary['max_drawdown']:>7.1%} | "
          f"days {summary['n_trading_days']:>5d}")


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("Daily backtester — EW strategies (long-only by default)")
    print("=" * 70)

    if not (DAILY_PATH.exists() and FEATURES_PATH.exists() and PREDICTIONS_PATH.exists()):
        raise FileNotFoundError("need daily, features, and predictions parquets")

    daily       = pd.read_parquet(DAILY_PATH)
    features    = pd.read_parquet(FEATURES_PATH)
    predictions = pd.read_parquet(PREDICTIONS_PATH)
    if "fwd_return_mt" not in predictions.columns:
        fwd = features[["assetid", "date_mt", "fwd_return_mt"]].dropna()
        predictions = predictions.merge(fwd, on=["assetid", "date_mt"], how="left")

    for strat in ["MOM", "XGB", "RET", "SRP", "CVR"]:
        daily_df, summary = backtest_ew_strategy(
            features, predictions, daily, strat,
            top_n=15,
            tc_bps=20.0,
            carry_long_annual=0.05,
            carry_short_annual=0.02,
            long_only=True,
            verbose=False,
        )
        print_summary(f"EW_{strat}", summary)
