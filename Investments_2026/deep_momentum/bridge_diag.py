"""
Bridge diagnostics between monthly selection returns and daily realized returns.

This is an audit helper, not a production backtester. It answers:
if we pick the same forecast-ranked names each month, how far are monthly
`fwd_return_mt` returns from daily close-to-close execution-window returns?
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from daily_backtest import build_daily_close_panel


PROJECT_DIR = Path(__file__).resolve().parent
CACHE_DIR = PROJECT_DIR / "cache"


def _exec_date(panel_dates: pd.Index, signal_date, execution_lag_days: int = 1):
    idx = panel_dates.searchsorted(pd.Timestamp(signal_date), side="right")
    target_idx = idx + execution_lag_days - 1
    if target_idx >= len(panel_dates):
        return None
    return panel_dates[target_idx]


def build_monthly_daily_bridge(
    features: pd.DataFrame,
    predictions: pd.DataFrame,
    daily: pd.DataFrame,
    strategy: str = "RET",
    top_n=15,
    execution_lag_days: int = 1,
    long_only: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return (per_position, per_month) bridge diagnostics.

    monthly return:
      feature/prediction row's fwd_return_mt, i.e. month-end t to month-end t+1.

    daily return:
      close at exec_date(t) to close at exec_date(t+1), using the same selected
      assetid. If either close is missing, daily return is NaN and counted.
    """
    selections = _build_bridge_selections(features, predictions, strategy, top_n)
    if not selections:
        return pd.DataFrame(), pd.DataFrame()

    panel = build_daily_close_panel(daily)
    panel_dates = panel.index

    schedule = []
    for sel in selections:
        ed = _exec_date(panel_dates, sel["date_mt"], execution_lag_days)
        if ed is not None:
            schedule.append({**sel, "exec_date": ed})

    if len(schedule) < 2:
        return pd.DataFrame(), pd.DataFrame()

    rows = []
    for i, sel in enumerate(schedule[:-1]):
        next_sel = schedule[i + 1]
        date_mt = pd.Timestamp(sel["date_mt"])
        entry_date = pd.Timestamp(sel["exec_date"])
        exit_date = pd.Timestamp(next_sel["exec_date"])

        long_rows = _pick_executable_rows(
            sel["long_candidates_df"], panel, entry_date, sel["target_n_long"]
        )
        short_rows = pd.DataFrame()
        if not long_only:
            short_rows = _pick_executable_rows(
                sel["short_candidates_df"], panel, entry_date, sel["target_n_short"],
                exclude_ids=set(long_rows["assetid"].astype(int)) if not long_rows.empty else set(),
            )

        legs = [("L", long_rows)]
        if not long_only:
            legs.append(("S", short_rows))

        for direction, leg_df in legs:
            if leg_df.empty:
                continue
            w = 1.0 / len(leg_df)
            for _, row in leg_df.iterrows():
                aid = int(row["assetid"])
                source_date = pd.Timestamp(row["date_mt"])
                entry_px = panel.at[entry_date, aid] if aid in panel.columns else np.nan
                exit_px = panel.at[exit_date, aid] if aid in panel.columns else np.nan
                if pd.notna(entry_px) and entry_px > 0 and pd.notna(exit_px):
                    daily_stock_ret = exit_px / entry_px - 1
                else:
                    daily_stock_ret = np.nan

                monthly_stock_ret = row.get("fwd_return_mt", np.nan)
                if direction == "L":
                    monthly_leg_ret = monthly_stock_ret
                    daily_leg_ret = daily_stock_ret
                else:
                    monthly_leg_ret = -monthly_stock_ret if pd.notna(monthly_stock_ret) else np.nan
                    daily_leg_ret = -daily_stock_ret if pd.notna(daily_stock_ret) else np.nan

                rows.append({
                    "date_mt": date_mt,
                    "source_date_mt": source_date,
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "strategy": strategy,
                    "assetid": aid,
                    "symbol": row.get("symbol"),
                    "securityname": row.get("securityname"),
                    "exchange_name": row.get("exchange_name"),
                    "direction": direction,
                    "weight": w,
                    "entry_px": entry_px,
                    "exit_px": exit_px,
                    "monthly_stock_ret": monthly_stock_ret,
                    "daily_stock_ret": daily_stock_ret,
                    "monthly_leg_ret": monthly_leg_ret,
                    "daily_leg_ret": daily_leg_ret,
                    "weighted_monthly": w * monthly_leg_ret if pd.notna(monthly_leg_ret) else np.nan,
                    "weighted_daily": w * daily_leg_ret if pd.notna(daily_leg_ret) else np.nan,
                    "diff": (w * daily_leg_ret - w * monthly_leg_ret)
                            if pd.notna(daily_leg_ret) and pd.notna(monthly_leg_ret) else np.nan,
                    "missing_monthly": pd.isna(monthly_leg_ret),
                    "missing_daily": pd.isna(daily_leg_ret),
                    "source_before_rebal": source_date < date_mt,
                })

    pos = pd.DataFrame(rows)
    if pos.empty:
        return pos, pd.DataFrame()

    month = (pos.groupby(["date_mt", "direction"], as_index=False)
               .agg(monthly_ret=("weighted_monthly", "sum"),
                    daily_ret=("weighted_daily", "sum"),
                    n=("assetid", "size"),
                    missing_monthly=("missing_monthly", "sum"),
                    missing_daily=("missing_daily", "sum")))
    wide = month.pivot(index="date_mt", columns="direction")
    out = pd.DataFrame(index=wide.index)
    for leg in ["L", "S"]:
        for col in ["monthly_ret", "daily_ret", "n", "missing_monthly", "missing_daily"]:
            out[f"{leg}_{col}"] = wide[(col, leg)] if (col, leg) in wide.columns else np.nan
    out = out.reset_index()
    out["monthly_ls_ret"] = out["L_monthly_ret"].fillna(0) + out["S_monthly_ret"].fillna(0)
    out["daily_ls_ret"] = out["L_daily_ret"].fillna(0) + out["S_daily_ret"].fillna(0)
    out["ls_diff"] = out["daily_ls_ret"] - out["monthly_ls_ret"]
    return pos, out


def _pick_executable_rows(candidates: pd.DataFrame,
                          panel: pd.DataFrame,
                          entry_date: pd.Timestamp,
                          n: int,
                          exclude_ids: set[int] | None = None) -> pd.DataFrame:
    """Take the first n ranked rows with a valid execution close."""
    if exclude_ids is None:
        exclude_ids = set()
    rows = []
    for _, row in candidates.iterrows():
        aid = int(row["assetid"])
        if aid in exclude_ids or aid not in panel.columns:
            continue
        px = panel.at[entry_date, aid]
        if pd.isna(px) or px <= 0:
            continue
        rows.append(row)
        if len(rows) >= n:
            break
    if not rows:
        return pd.DataFrame(columns=candidates.columns)
    return pd.DataFrame(rows)


def _resolve_n(top_n, n_total: int) -> int:
    if top_n is None:
        return 0
    if top_n > 1:
        return int(top_n)
    return max(1, int(round(top_n * n_total)))


def _build_bridge_selections(features: pd.DataFrame,
                             predictions: pd.DataFrame,
                             strategy: str,
                             top_n) -> list[dict]:
    """Forecast-only selections that preserve the selected source rows."""
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
    for _, grp in sub.groupby("_ym"):
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
        long_candidates = g.copy()
        short_candidates = g.iloc[::-1].copy()
        if long_candidates.empty or short_candidates.empty:
            continue
        rebal_date = grp["date_mt"].max()
        selections.append({
            "date_mt": pd.Timestamp(rebal_date),
            "target_n_long": n,
            "target_n_short": n,
            "long_candidates_df": long_candidates,
            "short_candidates_df": short_candidates,
        })
    return selections


def print_bridge_report(pos: pd.DataFrame, month: pd.DataFrame, long_only: bool = False) -> None:
    if pos.empty or month.empty:
        print("Bridge is empty.")
        return

    print("Bridge coverage")
    print("-" * 72)
    print(f"Positions: {len(pos):,}")
    print(f"Months:    {month['date_mt'].nunique():,}")
    print(f"Missing monthly leg returns: {int(pos['missing_monthly'].sum()):,}")
    print(f"Missing daily leg returns:   {int(pos['missing_daily'].sum()):,}")
    print(f"Source rows before rebalance date: {int(pos['source_before_rebal'].sum()):,}")
    print()

    cols = ["monthly_ls_ret", "daily_ls_ret", "ls_diff"]
    print("Monthly aggregate comparison")
    print("-" * 72)
    print(month[cols].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).to_string())
    corr = month[["monthly_ls_ret", "daily_ls_ret"]].corr().iloc[0, 1]
    print(f"\nCorrelation monthly vs daily: {corr:.3f}")
    print()

    print("Worst divergence months by |daily - monthly|")
    print("-" * 72)
    worst = month.reindex(month["ls_diff"].abs().sort_values(ascending=False).index).head(10)
    show_cols = [
        "date_mt", "monthly_ls_ret", "daily_ls_ret", "ls_diff",
        "L_missing_daily", "S_missing_daily",
    ]
    show_cols = [c for c in show_cols if c in worst.columns]
    print(worst[show_cols].to_string(index=False))
    print()

    print("Worst position-level divergences")
    print("-" * 72)
    worst_pos = pos.dropna(subset=["diff"]).copy()
    worst_pos = worst_pos.reindex(worst_pos["diff"].abs().sort_values(ascending=False).index).head(20)
    show_pos_cols = [
        "date_mt", "symbol", "direction", "weight", "entry_date", "exit_date",
        "source_date_mt", "monthly_stock_ret", "daily_stock_ret", "diff", "entry_px", "exit_px",
    ]
    print(worst_pos[show_pos_cols].to_string(index=False))


if __name__ == "__main__":
    features = pd.read_parquet(CACHE_DIR / "ca_features_monthly.parquet")
    predictions = pd.read_parquet(CACHE_DIR / "ca_predictions_monthly.parquet")
    daily = pd.read_parquet(CACHE_DIR / "ca_equities_daily.parquet")

    pos, month = build_monthly_daily_bridge(
        features, predictions, daily,
        strategy="RET",
        top_n=15,
        execution_lag_days=1,
        long_only=False,
    )
    print_bridge_report(pos, month, long_only=False)
