"""
Deep Momentum — FX layer
Converts local-currency stock returns to USD for a USD-base investor.

Uses Yahoo Finance (yfinance) for monthly FX rates. Pair convention:
    {CCY}USD=X → USD per one unit of {CCY}.

`fx_return[t]` is FORWARD (fx change from month t to t+1), aligned with
`fwd_return` in the features/predictions DataFrames so a simple
multiplicative compounding gives the USD return over the same holding period:

    fwd_return_usd = (1 + fwd_return_local) * (1 + fx_return) - 1
"""

import pandas as pd
import numpy as np

from config import COUNTRY_CURRENCY


def fetch_fx_series(currency, start="1990-01-01", end="2026-12-31"):
    """
    Fetch monthly USD/{currency} rates from Yahoo Finance.
    Returns DataFrame: date (month-end), fx_rate, fx_return (forward).
    """
    if currency == "USD":
        dates = pd.date_range(start, end, freq="ME")
        return pd.DataFrame({
            "date": dates,
            "fx_rate": 1.0,
            "fx_return": 0.0,
        })

    import yfinance as yf

    symbol = f"{currency}USD=X"
    df = yf.download(symbol, start=start, end=end,
                     progress=False, auto_adjust=False)
    if df.empty:
        raise RuntimeError(f"No Yahoo data for {symbol}")

    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    monthly = close.resample("ME").last().to_frame("fx_rate").reset_index()
    monthly.columns = ["date", "fx_rate"]
    monthly["date"] = pd.to_datetime(monthly["date"])

    # Forward FX return: fx[t+1]/fx[t] - 1, indexed at t
    monthly["fx_return"] = monthly["fx_rate"].pct_change().shift(-1)

    return monthly


def build_fx_table(country_suffixes, start="1990-01-01"):
    """
    Fetch FX series for all unique currencies needed by the given country
    suffixes. Returns {currency: DataFrame}.
    """
    currencies = sorted({COUNTRY_CURRENCY[s] for s in country_suffixes
                         if s in COUNTRY_CURRENCY})
    table = {}
    for ccy in currencies:
        print(f"  Fetching FX: {ccy}USD")
        table[ccy] = fetch_fx_series(ccy, start=start)
    return table


def convert_fwd_return_to_usd(df, country_suffix, fx_table):
    """
    Add `fwd_return_usd` column derived from local `fwd_return` and the
    matching FX series. The input DataFrame must have `date` and `fwd_return`.
    Rows outside the FX series date range get NaN USD returns.
    """
    ccy = COUNTRY_CURRENCY.get(country_suffix)
    if ccy is None:
        raise ValueError(f"No currency mapping for suffix {country_suffix}")
    if ccy not in fx_table:
        raise ValueError(f"FX table missing {ccy} for {country_suffix}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    fx = fx_table[ccy][["date", "fx_return"]].copy()
    fx["_ym"] = fx["date"].dt.to_period("M")
    fx_slim = fx[["_ym", "fx_return"]].rename(columns={"fx_return": "_fx"})

    df["_ym"] = df["date"].dt.to_period("M")
    df = df.merge(fx_slim, on="_ym", how="left").drop(columns=["_ym"])

    df["fwd_return_usd"] = (1 + df["fwd_return"]) * (1 + df["_fx"]) - 1
    df = df.drop(columns=["_fx"])
    return df


def replace_fwd_return_with_usd(df, country_suffix, fx_table):
    """
    Same as convert_fwd_return_to_usd but swaps `fwd_return` for its USD
    version in-place so existing portfolio functions operate on USD returns
    without code changes.
    """
    df = convert_fwd_return_to_usd(df, country_suffix, fx_table)
    df = df.drop(columns=["fwd_return"])
    df = df.rename(columns={"fwd_return_usd": "fwd_return"})
    return df
