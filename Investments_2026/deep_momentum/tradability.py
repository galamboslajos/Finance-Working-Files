"""
Deep Momentum — Tradability and daily-derived state features.

Builds point-in-time monthly tradability flags from the daily Norgate panel and
merges them onto the monthly panel before feature engineering/model training.

The goal is to keep the ML universe aligned with what the daily backtester can
actually trade, rather than training on sparse penny-stock rows that generate
large monthly labels but cannot be executed cleanly.
"""

from __future__ import annotations
import time
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
CACHE_DIR = PROJECT_DIR / "cache"

DAILY_PATH = CACHE_DIR / "ca_equities_daily.parquet"
MONTHLY_PATH = CACHE_DIR / "ca_equities_monthly.parquet"
TRADABLE_MONTHLY_PATH = CACHE_DIR / "ca_equities_monthly_tradable.parquet"


def build_daily_monthly_state(daily: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Aggregate daily execution/liquidity/range state to asset-month rows.

    Uses unadjusted close for price-level filters when available, adjusted close
    for daily return/range behavior. All statistics are known by month end.
    """
    df = daily.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["_ym"] = df["date"].dt.to_period("M")
    market_days = df.groupby("_ym")["date"].nunique().rename("market_days_mt")

    price_col = "unadjusted_close" if "unadjusted_close" in df.columns else "close"
    df["_price"] = df[price_col]
    df["_has_price"] = df["_price"].notna() & (df["_price"] > 0)
    df["_traded"] = df["_has_price"] & (df["volume"].fillna(0) > 0)
    df["_zero_volume"] = df["volume"].fillna(0) <= 0

    df = df.sort_values(["assetid", "date"])
    df["_daily_ret"] = df.groupby("assetid", sort=False)["close"].pct_change()
    df["_downside_ret"] = df["_daily_ret"].where(df["_daily_ret"] < 0, 0.0)

    if verbose:
        print(f"  Daily rows for tradability: {len(df):,}")

    t0 = time.time()
    grouped = df.groupby(["assetid", "_ym"], as_index=False)
    out = grouped.agg(
        date_mt=("date", "last"),
        last_trade_date_mt=("date", "last"),
        last_price_mt=("_price", "last"),
        min_price_mt=("_price", "min"),
        median_price_mt=("_price", "median"),
        valid_price_days_mt=("_has_price", "sum"),
        traded_days_mt=("_traded", "sum"),
        zero_volume_days_mt=("_zero_volume", "sum"),
        median_daily_turnover_mt=("turnover", "median"),
        mean_daily_turnover_mt=("turnover", "mean"),
        max_daily_ret_mt=("_daily_ret", "max"),
        min_daily_ret_mt=("_daily_ret", "min"),
        daily_vol_mt=("_daily_ret", "std"),
        downside_daily_vol_mt=("_downside_ret", "std"),
        daily_high_mt=("high", "max"),
        daily_low_mt=("low", "min"),
    )
    out = out.merge(market_days, on="_ym", how="left")
    out["zero_volume_share_mt"] = (
        out["zero_volume_days_mt"] / out["valid_price_days_mt"].replace(0, np.nan)
    )
    out["range_hl_mt"] = out["daily_high_mt"] / out["daily_low_mt"].replace(0, np.nan) - 1
    out["close_pos_range_mt"] = (
        (out["last_price_mt"] - out["daily_low_mt"]) /
        (out["daily_high_mt"] - out["daily_low_mt"]).replace(0, np.nan)
    )
    out["dist_high_mt"] = out["last_price_mt"] / out["daily_high_mt"].replace(0, np.nan) - 1
    out["dist_low_mt"] = out["last_price_mt"] / out["daily_low_mt"].replace(0, np.nan) - 1

    out = out.drop(columns=["daily_high_mt", "daily_low_mt"])
    if verbose:
        print(f"  Built daily state rows: {len(out):,} in {time.time()-t0:.0f}s")
    return out


def add_tradability_to_monthly(monthly: pd.DataFrame,
                               daily: pd.DataFrame,
                               min_price: float = 5.0,
                               turnover_bottom_pct: float = 0.20,
                               max_zero_volume_share: float = 0.50,
                               require_full_month_trading: bool = True,
                               verbose: bool = True) -> pd.DataFrame:
    """
    Merge daily state onto monthly rows and add `is_tradable_mt`.

    The turnover filter is cross-sectional: each month drops the bottom
    `turnover_bottom_pct` by median daily turnover. This is less arbitrary than
    a fixed dollar floor and adapts across eras.
    """
    state = build_daily_monthly_state(daily, verbose=verbose)
    df = monthly.copy()
    df["_ym"] = pd.to_datetime(df["date_mt"]).dt.to_period("M")
    merged = df.merge(
        state.drop(columns=["date_mt"]),
        on=["assetid", "_ym"],
        how="left",
    ).drop(columns=["_ym"])

    merged["turnover_ratio_3_mt"] = (
        merged["turnover_mt"] /
        merged.groupby("assetid", sort=False)["turnover_mt"]
              .rolling(3, min_periods=2).mean()
              .reset_index(level=0, drop=True)
              .replace(0, np.nan)
    )
    merged["turnover_ratio_12_mt"] = (
        merged["turnover_mt"] /
        merged.groupby("assetid", sort=False)["turnover_mt"]
              .rolling(12, min_periods=6).mean()
              .reset_index(level=0, drop=True)
              .replace(0, np.nan)
    )

    merged["turnover_pct_rank_mt"] = (
        merged.groupby(merged["date_mt"].dt.to_period("M"))["median_daily_turnover_mt"]
              .rank(pct=True)
    )

    turnover_ok = merged["turnover_pct_rank_mt"] > turnover_bottom_pct
    turnover_label = f"drop bottom {turnover_bottom_pct:.0%} by monthly median daily turnover"

    if require_full_month_trading:
        traded_ok = merged["traded_days_mt"] >= merged["market_days_mt"]
        traded_label = "full-month trading"
    else:
        traded_ok = merged["traded_days_mt"] > 0
        traded_label = "at least one traded day"

    tradable = (
        (merged["last_price_mt"] >= min_price) &
        turnover_ok &
        traded_ok &
        (merged["zero_volume_share_mt"].fillna(1.0) <= max_zero_volume_share)
    )
    merged["is_tradable_mt"] = tradable

    if verbose:
        n_before = len(merged)
        n_after = int(tradable.sum())
        print("  Tradability flags:")
        print(f"    min price:             {min_price:,.2f} CAD")
        print(f"    turnover filter:       {turnover_label}")
        print(f"    trading-days filter:   {traded_label}")
        print(f"    max zero-volume share: {max_zero_volume_share:.0%}")
        print(f"    tradable rows:         {n_after:,} / {n_before:,} ({n_after/max(n_before,1):.1%})")
    return merged


def filter_tradable_monthly(monthly: pd.DataFrame,
                            min_price: float = 5.0,
                            turnover_bottom_pct: float = 0.20,
                            max_zero_volume_share: float = 0.50,
                            require_full_month_trading: bool = True,
                            verbose: bool = True) -> pd.DataFrame:
    """Filter a monthly panel that already has tradability columns."""
    required = [
        "last_price_mt", "median_daily_turnover_mt",
        "traded_days_mt", "zero_volume_share_mt",
        "turnover_pct_rank_mt", "market_days_mt",
    ]
    missing = [c for c in required if c not in monthly.columns]
    if missing:
        raise ValueError(f"monthly panel missing tradability columns: {missing}")

    turnover_ok = monthly["turnover_pct_rank_mt"] > turnover_bottom_pct

    if require_full_month_trading:
        traded_ok = monthly["traded_days_mt"] >= monthly["market_days_mt"]
    else:
        traded_ok = monthly["traded_days_mt"] > 0

    mask = (
        (monthly["last_price_mt"] >= min_price) &
        turnover_ok &
        traded_ok &
        (monthly["zero_volume_share_mt"].fillna(1.0) <= max_zero_volume_share)
    )
    out = monthly[mask].copy()
    if verbose:
        print(f"  Tradability filter: dropped {len(monthly)-len(out):,} obs "
              f"({100*(len(monthly)-len(out))/max(len(monthly),1):.2f}%)")
        print(f"    final tradable rows: {len(out):,}, assets: {out['assetid'].nunique():,}")
    return out


if __name__ == "__main__":
    if not DAILY_PATH.exists() or not MONTHLY_PATH.exists():
        raise FileNotFoundError("Need daily and monthly parquets first.")
    t0 = time.time()
    daily = pd.read_parquet(DAILY_PATH)
    monthly = pd.read_parquet(MONTHLY_PATH)
    out = add_tradability_to_monthly(monthly, daily)
    out.to_parquet(TRADABLE_MONTHLY_PATH, index=False, compression="snappy")
    print(f"Wrote {TRADABLE_MONTHLY_PATH} in {time.time()-t0:.0f}s")
