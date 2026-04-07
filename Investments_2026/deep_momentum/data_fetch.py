"""
Deep Momentum — Step 1: Data Fetching
Downloads daily price data + market cap from FMP for all countries.
Converts to monthly returns (last trading day of each month).

Paper reference: Section 3.1
- Monthly stock data are generated from the daily data by choosing the last records
  of the stocks in each month
- Monthly returns are calculated using the Return Index (adjusted close in our case)
- Market capitalization must be available
"""

import time
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

from config import (
    FMP_API_KEY, FMP_BASE, FMP_RATE_LIMIT,
    COUNTRIES, DATA_DIR,
)


def fmp_get(endpoint, **params):
    """Single FMP API call with rate limiting."""
    params["apikey"] = FMP_API_KEY
    r = requests.get(f"{FMP_BASE}/{endpoint}", params=params)
    r.raise_for_status()
    return r.json()


def get_stock_list_for_suffix(suffix):
    """Get all stock symbols for a given exchange suffix from FMP stock list."""
    data = fmp_get("stock-list")
    df = pd.DataFrame(data)
    if suffix == "US":
        # US stocks have no suffix
        mask = ~df["symbol"].str.contains(r"\.", na=False)
    else:
        mask = df["symbol"].str.endswith(f".{suffix}")
    return df.loc[mask, "symbol"].tolist()


def fetch_daily_prices(symbol, from_date="1990-01-01", to_date="2026-12-31"):
    """
    Fetch full daily price history for one symbol.
    FMP caps at 5000 rows per request, so we paginate.
    Returns DataFrame with columns: date, open, high, low, close, adjClose, volume.
    """
    all_data = []
    current_to = to_date

    while True:
        data = fmp_get(
            "historical-price-eod/full",
            symbol=symbol,
            **{"from": from_date, "to": current_to},
        )
        if not isinstance(data, list) or len(data) == 0:
            break

        all_data.extend(data)

        # If we got fewer than 5000, we have all the data
        if len(data) < 5000:
            break

        # Otherwise paginate: next request ends one day before earliest date
        earliest = data[-1]["date"]
        current_to = (pd.Timestamp(earliest) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates(subset=["date"])
    return df


def daily_to_monthly(df):
    """
    Convert daily prices to monthly observations.
    Paper: "Monthly stock data are generated from the daily data by choosing
    the last records of the stocks in each month"

    Returns one row per month with:
    - date: last trading day of the month
    - close: adjusted close on that day
    - volume: total volume for the month
    - market_cap: market cap on last trading day (if available)
    """
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["year_month"] = df["date"].dt.to_period("M")

    # Last trading day per month
    monthly = df.groupby("year_month").last().reset_index()

    # Sum volume over month (for zero-volume filter)
    vol_sum = df.groupby("year_month")["volume"].sum().reset_index()
    vol_sum.columns = ["year_month", "volume_month"]

    monthly = monthly.merge(vol_sum, on="year_month", how="left")

    # Previous month volume (for the filter: "trading volumes in current and previous months")
    monthly = monthly.sort_values("date")
    monthly["volume_prev_month"] = monthly["volume_month"].shift(1)

    return monthly


def compute_monthly_returns(monthly_df):
    """
    Compute monthly returns from adjusted close prices.
    Paper uses Return Index (total return including dividends).
    FMP stable API 'close' is already split/dividend adjusted.

    r_t = close_t / close_{t-1} - 1
    """
    if monthly_df.empty:
        return monthly_df

    df = monthly_df.copy().sort_values("date")
    df["return"] = df["close"].pct_change()

    return df


def fetch_market_cap(symbol, from_date="1990-01-01"):
    """
    Fetch historical market cap for a symbol.
    FMP caps at 5000 rows per request, so we paginate.
    Returns DataFrame with date and marketCap columns.
    """
    all_data = []
    current_to = "2026-12-31"

    while True:
        data = fmp_get(
            "historical-market-capitalization",
            symbol=symbol,
            limit=5000,
            **{"from": from_date, "to": current_to},
        )
        if not isinstance(data, list) or len(data) == 0:
            break

        all_data.extend(data)

        if len(data) < 5000:
            break

        earliest = data[-1]["date"]
        current_to = (pd.Timestamp(earliest) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(subset=["date"])
    return df[["date", "marketCap"]].sort_values("date")


def process_one_stock(symbol, min_date="1990-01-01"):
    """
    Full pipeline for one stock:
    1. Fetch daily prices (paginated)
    2. Convert to monthly
    3. Compute returns
    4. Fetch market cap and merge

    Returns monthly DataFrame or empty DataFrame on failure.
    """
    try:
        # Daily prices
        daily = fetch_daily_prices(symbol, from_date=min_date)
        if daily.empty or len(daily) < 20:
            return pd.DataFrame()

        # Monthly
        monthly = daily_to_monthly(daily)
        if monthly.empty:
            return pd.DataFrame()

        # Returns
        monthly = compute_monthly_returns(monthly)

        # Market cap
        mcap = fetch_market_cap(symbol, from_date=min_date)
        if not mcap.empty:
            # Merge market cap: for each monthly date, get closest prior market cap
            mcap = mcap.sort_values("date")
            monthly = monthly.sort_values("date")
            monthly = pd.merge_asof(
                monthly, mcap, on="date", direction="backward"
            )
        else:
            monthly["marketCap"] = np.nan

        monthly["symbol"] = symbol
        return monthly

    except Exception as e:
        print(f"  ERROR {symbol}: {e}")
        return pd.DataFrame()


def fetch_country(suffix, country_name, max_stocks=None, min_date="1990-01-01"):
    """
    Fetch all stocks for one country (exchange suffix).

    Args:
        suffix: FMP exchange suffix (e.g., 'L' for UK, 'T' for Japan)
        country_name: for logging
        max_stocks: limit for testing (None = all)
        min_date: earliest date to fetch

    Returns:
        DataFrame with all monthly observations for the country.
    """
    print(f"\n{'='*60}")
    print(f"Fetching {country_name} (.{suffix})")
    print(f"{'='*60}")

    # Get symbol list
    symbols = get_stock_list_for_suffix(suffix)
    print(f"  Found {len(symbols)} symbols")

    if max_stocks is not None:
        symbols = symbols[:max_stocks]
        print(f"  Limited to {max_stocks} for testing")

    # Rate limiter: simple sleep
    delay = 1.0 / FMP_RATE_LIMIT

    all_monthly = []
    errors = 0

    for sym in tqdm(symbols, desc=f"  {suffix}", ncols=80):
        result = process_one_stock(sym, min_date=min_date)
        if not result.empty:
            all_monthly.append(result)
        else:
            errors += 1
        time.sleep(delay)  # rate limit

    if not all_monthly:
        print(f"  No data retrieved for {country_name}")
        return pd.DataFrame()

    df = pd.concat(all_monthly, ignore_index=True)
    print(f"  Retrieved: {df['symbol'].nunique()} stocks, "
          f"{len(df)} monthly obs, {errors} errors")

    return df


def fetch_all_countries(suffixes=None, max_stocks_per_country=None,
                        min_date="1990-01-01", save=True):
    """
    Fetch data for all (or selected) countries.

    Args:
        suffixes: list of suffixes to fetch, or None for all
        max_stocks_per_country: limit per country for testing
        min_date: earliest date
        save: whether to save per-country parquet files

    Returns:
        dict of {suffix: DataFrame}
    """
    if suffixes is None:
        suffixes = list(COUNTRIES.keys())

    results = {}
    for suffix in suffixes:
        if suffix not in COUNTRIES:
            print(f"Unknown suffix: {suffix}, skipping")
            continue

        _, country_name, _, _ = COUNTRIES[suffix]
        df = fetch_country(
            suffix, country_name,
            max_stocks=max_stocks_per_country,
            min_date=min_date,
        )

        if not df.empty:
            df["country_suffix"] = suffix
            df["country_code"] = COUNTRIES[suffix][0]
            df["country_name"] = country_name

            if save:
                path = Path(DATA_DIR) / f"monthly_{suffix}.parquet"
                df.to_parquet(path, index=False)
                print(f"  Saved: {path} ({len(df)} rows)")

            results[suffix] = df

    return results


# ═══════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch stock data from FMP")
    parser.add_argument("--countries", nargs="+", default=None,
                        help="Exchange suffixes to fetch (e.g., US TO T). Default: all")
    parser.add_argument("--max-stocks", type=int, default=None,
                        help="Max stocks per country (for testing)")
    parser.add_argument("--min-date", default="1990-01-01",
                        help="Earliest date to fetch (default: 1990-01-01)")
    args = parser.parse_args()

    results = fetch_all_countries(
        suffixes=args.countries,
        max_stocks_per_country=args.max_stocks,
        min_date=args.min_date,
    )

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    for suffix, df in results.items():
        name = COUNTRIES[suffix][1]
        n_stocks = df["symbol"].nunique()
        date_range = f"{df['date'].min().date()} to {df['date'].max().date()}"
        print(f"  {name:25s} {n_stocks:>5d} stocks  {len(df):>8d} obs  {date_range}")
