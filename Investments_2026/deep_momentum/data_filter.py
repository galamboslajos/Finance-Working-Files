"""
Deep Momentum — Step 2: Data Filtering
Applies all filters from Section 3.1 of the paper, exactly as described.

Filters (in order):
1. Exclude if trading volumes in current AND previous months are zero AND return is zero
2. Exclude if market capitalization is not available in current or previous month
3. Exclude if market cap is below bottom 5% within a country in any month
4. Remove observations with monthly returns > 300% or < -95%
5. Winsorize remaining returns at 1% and 99% within each country

After filtering:
- Keep only countries with data from 2000 or earlier
- Keep only countries with at least 30 stocks in ALL months
"""

import pandas as pd
import numpy as np
from pathlib import Path

from config import (
    MAX_MONTHLY_RETURN, MIN_MONTHLY_RETURN,
    WINSORIZE_LOWER, WINSORIZE_UPPER,
    MCAP_BOTTOM_PCT, MCAP_BOTTOM_PCT_US, DATA_DIR, CACHE_DIR,
    COUNTRIES, MIN_TRAIN_YEARS,
)


def filter_zero_volume(df):
    """
    Paper: "Observations are excluded from the sample if the trading volumes
    in the current and previous months are zero, and the return is zero"
    """
    n_before = len(df)

    mask = (
        (df["volume_month"] == 0) &
        (df["volume_prev_month"].fillna(0) == 0) &
        (df["return"].fillna(0).abs() < 1e-10)
    )
    df = df[~mask].copy()

    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f"    Zero volume filter: dropped {n_dropped} obs ({100*n_dropped/n_before:.1f}%)")

    return df


def filter_market_cap_missing(df):
    """
    Paper: "the market capitalization is not available in the current or previous month"
    """
    n_before = len(df)

    df = df.sort_values(["symbol", "date"])
    df["marketCap_prev"] = df.groupby("symbol")["marketCap"].shift(1)

    mask = df["marketCap"].isna() | df["marketCap_prev"].isna()
    df = df[~mask].copy()

    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f"    MCap missing filter: dropped {n_dropped} obs ({100*n_dropped/n_before:.1f}%)")

    df = df.drop(columns=["marketCap_prev"])
    return df


def filter_market_cap_bottom(df, mcap_pct=None):
    """
    Paper: "Observations are excluded from the sample if the market capitalization
    is below the bottom 5% within a country in any month"

    Per-observation filter: for each month, compute the threshold percentile of
    market cap and remove only those observations below it in that month.

    mcap_pct: override threshold (e.g., 0.25 for US to remove OTC junk)
    """
    if mcap_pct is None:
        mcap_pct = MCAP_BOTTOM_PCT

    n_before = len(df)

    df = df.copy()
    df["_ym"] = df["date"].dt.to_period("M")
    df["_mcap_threshold"] = df.groupby("_ym")["marketCap"].transform(
        lambda x: x.quantile(mcap_pct)
    )

    mask = df["marketCap"] < df["_mcap_threshold"]
    n_dropped = mask.sum()
    df = df[~mask].copy()
    df = df.drop(columns=["_ym", "_mcap_threshold"])

    if n_dropped > 0:
        print(f"    MCap bottom {mcap_pct:.0%} filter: dropped {n_dropped} obs "
              f"({100*n_dropped/n_before:.1f}%)")

    return df


def filter_extreme_returns(df):
    """
    Paper: "Observations with monthly returns greater than 300% or lower than -95% are removed"
    """
    n_before = len(df)

    mask = (df["return"] > MAX_MONTHLY_RETURN) | (df["return"] < MIN_MONTHLY_RETURN)
    n_extreme = mask.sum()
    df = df[~mask].copy()

    if n_extreme > 0:
        print(f"    Extreme return filter: dropped {n_extreme} obs ({100*n_extreme/n_before:.1f}%)")

    return df


def winsorize_returns(df):
    """
    Paper: "the remaining returns are winsorized at 1% and 99% within each country"
    """
    lower = df["return"].quantile(WINSORIZE_LOWER)
    upper = df["return"].quantile(WINSORIZE_UPPER)

    n_clipped = ((df["return"] < lower) | (df["return"] > upper)).sum()
    df["return"] = df["return"].clip(lower, upper)

    if n_clipped > 0:
        print(f"    Winsorize: clipped {n_clipped} obs at [{lower:.4f}, {upper:.4f}]")

    return df


def filter_country(df, country_name, suffix=""):
    """
    Apply all filters to one country's data.
    Returns filtered DataFrame.
    """
    print(f"\n  {country_name}: {df['symbol'].nunique()} stocks, {len(df)} obs")

    # Must have return computed (drop first obs per stock where return is NaN)
    df = df.dropna(subset=["return"]).copy()

    # 1. Zero volume
    df = filter_zero_volume(df)

    # 2. Market cap missing
    df = filter_market_cap_missing(df)

    # 3. Market cap bottom % (25% for US to remove OTC, 5% for others)
    mcap_pct = MCAP_BOTTOM_PCT_US if suffix == "US" else MCAP_BOTTOM_PCT
    df = filter_market_cap_bottom(df, mcap_pct=mcap_pct)

    # 4. Extreme returns
    df = filter_extreme_returns(df)

    # 5. Winsorize within country
    df = winsorize_returns(df)

    print(f"    After all filters: {df['symbol'].nunique()} stocks, {len(df)} obs")

    return df


def check_country_eligibility(df, suffix):
    """
    Paper: "we choose countries with data available from 2000 or earlier
    and at least 30 stocks in all months throughout the sample period"

    Returns (eligible: bool, reason: str)
    """
    # Check data available from 2000 or earlier
    first_date = df["date"].min()
    if first_date > pd.Timestamp("2000-12-31"):
        return False, f"data starts {first_date.date()}, need <=2000"

    # Check at least 30 stocks in all months
    stocks_per_month = df.groupby(df["date"].dt.to_period("M"))["symbol"].nunique()
    min_stocks = stocks_per_month.min()
    if min_stocks < 30:
        return False, f"min stocks/month = {min_stocks}, need >=30"

    return True, "OK"


def filter_all_countries(save=True):
    """
    Load raw data for all countries, apply filters, check eligibility.
    Returns dict of {suffix: filtered_DataFrame} for eligible countries.
    """
    data_dir = Path(DATA_DIR)
    files = sorted(data_dir.glob("monthly_*.parquet"))

    if not files:
        print("No data files found. Run data_fetch.py first.")
        return {}

    print("=" * 70)
    print("APPLYING PAPER FILTERS (Section 3.1)")
    print("=" * 70)

    results = {}
    eligibility = []

    for f in files:
        suffix = f.stem.replace("monthly_", "")
        if suffix not in COUNTRIES:
            continue

        _, country_name, _, _ = COUNTRIES[suffix]
        df = pd.read_parquet(f)

        # Apply filters
        filtered = filter_country(df, country_name)

        if filtered.empty:
            eligibility.append({
                "suffix": suffix, "country": country_name,
                "eligible": False, "reason": "no data after filtering"
            })
            continue

        # Check eligibility
        eligible, reason = check_country_eligibility(filtered, suffix)
        eligibility.append({
            "suffix": suffix, "country": country_name,
            "eligible": eligible, "reason": reason,
            "n_stocks": filtered["symbol"].nunique(),
            "n_obs": len(filtered),
            "first_date": str(filtered["date"].min().date()),
            "last_date": str(filtered["date"].max().date()),
        })

        if save:
            out_path = Path(CACHE_DIR) / f"filtered_{suffix}.parquet"
            filtered.to_parquet(out_path, index=False)

        results[suffix] = filtered

    # Print eligibility summary
    elig_df = pd.DataFrame(eligibility)
    print(f"\n{'='*70}")
    print(f"ELIGIBILITY SUMMARY")
    print(f"{'='*70}")
    print(f"\n{'Country':<25s} {'Suffix':>6s} {'Stocks':>6s} {'Obs':>7s} "
          f"{'First':>12s} {'Eligible':>8s} {'Reason'}")
    print("-" * 100)

    for _, row in elig_df.iterrows():
        print(f"{row['country']:<25s} {row['suffix']:>6s} "
              f"{row.get('n_stocks', 0):>6.0f} {row.get('n_obs', 0):>7.0f} "
              f"{row.get('first_date', 'N/A'):>12s} "
              f"{'YES' if row['eligible'] else 'NO':>8s} "
              f"{row['reason']}")

    n_eligible = elig_df["eligible"].sum()
    print(f"\nEligible countries: {n_eligible} / {len(elig_df)}")

    return results


# ═══════════════════════════════════════════════════════
if __name__ == "__main__":
    results = filter_all_countries(save=True)

    print(f"\n{'='*70}")
    print(f"FILTERED DATA SUMMARY")
    print(f"{'='*70}")
    total_stocks = 0
    total_obs = 0
    for suffix, df in sorted(results.items()):
        name = COUNTRIES[suffix][1]
        n = df["symbol"].nunique()
        total_stocks += n
        total_obs += len(df)
        ret = df["return"]
        print(f"  {name:<25s} {n:>5d} stocks  {len(df):>7d} obs  "
              f"ret: [{ret.min():.2%}, {ret.max():.2%}]  "
              f"mean={ret.mean():.4f}")

    print(f"\n  TOTAL: {total_stocks} stocks, {total_obs} obs")
