"""
Deep Momentum — Step 2: Aggregate daily Norgate panel into monthly.

Reads:
  cache/ca_equities_daily.parquet   (output of data_load.py)

Writes:
  cache/ca_equities_monthly.parquet

Convention: every column derived from a daily-to-monthly aggregation has the
suffix `_mt` so daily and monthly columns can never silently collide when the
two parquets are joined later (the optimizer/CVaR layer uses daily; the
ML/forecasting layer uses monthly).

Per (assetid, calendar month) collapse:
  date_mt              last trading day of the month
  open_mt              first trading day's open
  high_mt              max of daily high
  low_mt               min of daily low
  close_mt             last day's close (adjusted)
  unadjusted_close_mt  last day's unadjusted close
  volume_mt            sum of daily volume
  turnover_mt          sum of daily turnover (CAD)
  dividend_mt          sum of dividends paid in the month
  n_obs_mt             number of trading days observed

Then per assetid, after sorting by date_mt:
  return_mt            close_mt / close_mt.shift(1) - 1
  volume_mt_prev       volume_mt.shift(1)
  turnover_mt_prev     turnover_mt.shift(1)

Metadata columns are carried through unchanged from the daily input.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
CACHE_DIR   = PROJECT_DIR / "cache"

DAILY_PATH   = CACHE_DIR / "ca_equities_daily.parquet"
MONTHLY_PATH = CACHE_DIR / "ca_equities_monthly.parquet"


# Columns that vary across rows of the same assetid (we aggregate these)
AGG_NUMERIC = {
    "open":              ("open_mt",              "first"),
    "high":              ("high_mt",              "max"),
    "low":               ("low_mt",               "min"),
    "close":             ("close_mt",             "last"),
    "unadjusted_close":  ("unadjusted_close_mt",  "last"),
    "volume":            ("volume_mt",            "sum"),
    "turnover":          ("turnover_mt",          "sum"),
    "dividend":          ("dividend_mt",          "sum"),
}
# `date` aggregation is special — last trading day → date_mt

# Metadata columns to carry through (constant per assetid, taken as `last`)
META_COLS = [
    "symbol", "securityname",
    "subtype1", "subtype2", "subtype3",
    "exchange_name", "exchange_name_full",
    "is_operating_company", "is_delisted", "delist_date",
    "currency",
]


def aggregate_to_monthly(daily: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Collapse a daily OHLCV panel into a monthly panel."""
    if verbose:
        print(f"  Input rows:       {len(daily):,}")
        print(f"  Unique assetids:  {daily['assetid'].nunique():,}")

    df = daily.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["_ym"]  = df["date"].dt.to_period("M")

    t0 = time.time()

    # Numeric aggregations + n_obs + last date
    agg_dict: dict[str, tuple[str, str]] = {}
    for src_col, (out_col, how) in AGG_NUMERIC.items():
        if src_col in df.columns:
            agg_dict[out_col] = (src_col, how)
    agg_dict["date_mt"]  = ("date", "last")
    agg_dict["n_obs_mt"] = ("date", "count")

    monthly = df.groupby(["assetid", "_ym"], as_index=False).agg(**agg_dict)
    if verbose:
        print(f"  After groupby:    {len(monthly):,} rows  ({time.time()-t0:.0f}s)")

    # Carry through metadata (constant per assetid; pick the last observation)
    meta_present = [c for c in META_COLS if c in df.columns]
    if meta_present:
        meta = (df.sort_values(["assetid", "date"])
                  .groupby("assetid", as_index=False)[meta_present].last())
        monthly = monthly.merge(meta, on="assetid", how="left")

    # Per-assetid lagged columns + return_mt
    monthly = monthly.sort_values(["assetid", "date_mt"]).reset_index(drop=True)
    grp = monthly.groupby("assetid", sort=False)
    monthly["return_mt"]        = grp["close_mt"].pct_change()
    monthly["volume_mt_prev"]   = grp["volume_mt"].shift(1)
    monthly["turnover_mt_prev"] = grp["turnover_mt"].shift(1)

    monthly = monthly.drop(columns=["_ym"])

    # Tidy ordering
    leading = ["assetid", "symbol", "date_mt",
               "open_mt", "high_mt", "low_mt", "close_mt", "unadjusted_close_mt",
               "volume_mt", "turnover_mt", "dividend_mt", "n_obs_mt",
               "return_mt", "volume_mt_prev", "turnover_mt_prev",
               "is_delisted", "delist_date",
               "securityname", "subtype1", "subtype2", "subtype3",
               "exchange_name", "exchange_name_full", "is_operating_company",
               "currency"]
    cols = [c for c in leading if c in monthly.columns] + \
           [c for c in monthly.columns if c not in leading]
    monthly = monthly[cols]

    if verbose:
        print(f"  Output rows:      {len(monthly):,}")
        print(f"  Date range (mt):  {monthly['date_mt'].min().date()} → "
              f"{monthly['date_mt'].max().date()}")
        print(f"  Compression:      {len(daily)/len(monthly):.1f}× (daily→monthly)")

    return monthly


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("Aggregating daily → monthly")
    print("=" * 70)

    if not DAILY_PATH.exists():
        raise FileNotFoundError(
            f"Missing {DAILY_PATH}. Run data_load.py first."
        )

    t0 = time.time()
    daily = pd.read_parquet(DAILY_PATH)
    print(f"  Loaded {DAILY_PATH.name} in {time.time()-t0:.0f}s")

    t1 = time.time()
    monthly = aggregate_to_monthly(daily)
    print(f"\nAggregation done in {time.time()-t1:.0f}s")

    print(f"\nWriting {MONTHLY_PATH}...")
    monthly.to_parquet(MONTHLY_PATH, index=False, compression="snappy")
    size_mb = MONTHLY_PATH.stat().st_size / (1024 * 1024)
    print(f"  wrote {size_mb:.0f} MB")

    print(f"\nTotal runtime: {time.time()-t0:.0f}s")
