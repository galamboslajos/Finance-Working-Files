"""
Deep Momentum — Step 5: Portfolio Construction
Constructs long-short portfolios for MOM, XGB, and RET strategies.

Paper reference: Sections 3.2, 4.2.2

Three strategies:
- MOM: Traditional momentum — buy top decile of past 11-month return (skip 1 month),
       sell bottom decile. Rebalance monthly.
- XGB: Naive ML — buy stocks classified into highest class (10),
       sell stocks classified into lowest class (1).
- RET: Deep Momentum — sort stocks by predicted expected return,
       buy top decile, sell bottom decile.

Portfolio construction:
- Long-short: buy top decile, sell bottom decile
- Equal-weighted (primary analysis)
- Monthly rebalance
"""

import pandas as pd
import numpy as np
from config import N_CLASSES, OOS_START


def construct_mom_portfolio(df):
    """
    Traditional momentum strategy.

    Paper Section 3.2:
    "the momentum strategy makes predictions based on the past eleven-month
    return with a one-month lag and buys (sells) stocks whose past returns
    belong to the highest (lowest) decile"

    MOM_12 in our features = cumulative return from t-11 to t-1 (11 months,
    skip current month). This is exactly the Jegadeesh-Titman momentum signal.

    Returns DataFrame with monthly long-short returns.
    """
    df = df.copy()

    # Need MOM_12 and forward return
    required = ["MOM_12", "fwd_return", "date", "symbol"]
    df = df.dropna(subset=[c for c in required if c in df.columns])

    if df.empty:
        return pd.DataFrame()

    results = []
    for date, group in df.groupby(df["date"].dt.to_period("M")):
        if len(group) < N_CLASSES:
            continue

        # Sort into deciles by MOM_12
        group = group.copy()
        group["mom_decile"] = pd.qcut(
            group["MOM_12"], q=N_CLASSES, labels=False, duplicates="drop"
        ) + 1

        # Long = top decile, Short = bottom decile
        long_stocks = group[group["mom_decile"] == N_CLASSES]
        short_stocks = group[group["mom_decile"] == 1]

        if long_stocks.empty or short_stocks.empty:
            continue

        # Equal-weighted returns
        long_ret = long_stocks["fwd_return"].mean()
        short_ret = short_stocks["fwd_return"].mean()
        ls_ret = long_ret - short_ret

        results.append({
            "date": group["date"].iloc[0],
            "long_ret": long_ret,
            "short_ret": short_ret,
            "ls_ret": ls_ret,
            "n_long": len(long_stocks),
            "n_short": len(short_stocks),
            "strategy": "MOM",
        })

    return pd.DataFrame(results)


def construct_xgb_portfolio(predictions):
    """
    Naive ML strategy.

    Paper Section 4.2.2:
    For XGB, select all stocks classified into the highest or lowest return class.

    "buy stocks classified into highest class (10),
     sell stocks classified into lowest class (1)"
    """
    df = predictions.copy()
    df = df.dropna(subset=["xgb_class", "fwd_return"])

    if df.empty:
        return pd.DataFrame()

    results = []
    for date, group in df.groupby(df["date"].dt.to_period("M")):
        long_stocks = group[group["xgb_class"] == N_CLASSES]
        short_stocks = group[group["xgb_class"] == 1]

        if long_stocks.empty or short_stocks.empty:
            continue

        long_ret = long_stocks["fwd_return"].mean()
        short_ret = short_stocks["fwd_return"].mean()
        ls_ret = long_ret - short_ret

        results.append({
            "date": group["date"].iloc[0],
            "long_ret": long_ret,
            "short_ret": short_ret,
            "ls_ret": ls_ret,
            "n_long": len(long_stocks),
            "n_short": len(short_stocks),
            "strategy": "XGB",
        })

    return pd.DataFrame(results)


def construct_ret_portfolio(predictions):
    """
    Deep Momentum (RET) strategy.

    Paper Section 3.3.2 / 4.2.2:
    "RET selects the top and bottom 10% of all stocks based on their
    predicted return" (i.e., predicted expected return from reclassification)

    Sort stocks by ret_score, buy top decile, sell bottom decile.
    """
    df = predictions.copy()
    df = df.dropna(subset=["ret_score", "fwd_return"])

    if df.empty:
        return pd.DataFrame()

    results = []
    for date, group in df.groupby(df["date"].dt.to_period("M")):
        if len(group) < N_CLASSES:
            continue

        group = group.copy()
        group["ret_decile"] = pd.qcut(
            group["ret_score"], q=N_CLASSES, labels=False, duplicates="drop"
        ) + 1

        long_stocks = group[group["ret_decile"] == N_CLASSES]
        short_stocks = group[group["ret_decile"] == 1]

        if long_stocks.empty or short_stocks.empty:
            continue

        long_ret = long_stocks["fwd_return"].mean()
        short_ret = short_stocks["fwd_return"].mean()
        ls_ret = long_ret - short_ret

        results.append({
            "date": group["date"].iloc[0],
            "long_ret": long_ret,
            "short_ret": short_ret,
            "ls_ret": ls_ret,
            "n_long": len(long_stocks),
            "n_short": len(short_stocks),
            "strategy": "RET",
        })

    return pd.DataFrame(results)


def construct_srp_portfolio(predictions):
    """
    Deep Momentum (SRP) strategy — Sharpe ratio reclassification.

    Same as RET but sorts by predicted Sharpe ratio instead of expected return.
    """
    df = predictions.copy()
    df = df.dropna(subset=["srp_score", "fwd_return"])

    if df.empty:
        return pd.DataFrame()

    results = []
    for date, group in df.groupby(df["date"].dt.to_period("M")):
        if len(group) < N_CLASSES:
            continue

        group = group.copy()
        group["srp_decile"] = pd.qcut(
            group["srp_score"], q=N_CLASSES, labels=False, duplicates="drop"
        ) + 1

        long_stocks = group[group["srp_decile"] == N_CLASSES]
        short_stocks = group[group["srp_decile"] == 1]

        if long_stocks.empty or short_stocks.empty:
            continue

        long_ret = long_stocks["fwd_return"].mean()
        short_ret = short_stocks["fwd_return"].mean()
        ls_ret = long_ret - short_ret

        results.append({
            "date": group["date"].iloc[0],
            "long_ret": long_ret,
            "short_ret": short_ret,
            "ls_ret": ls_ret,
            "n_long": len(long_stocks),
            "n_short": len(short_stocks),
            "strategy": "SRP",
        })

    return pd.DataFrame(results)


def filter_oos(portfolio_df, oos_start=OOS_START):
    """
    Paper: "All empirical results in this section are from the out-of-sample
    period of January 2010 to December 2023"

    Filter portfolio returns to OOS period only.
    """
    if portfolio_df.empty:
        return portfolio_df
    return portfolio_df[portfolio_df["date"] >= pd.Timestamp(oos_start)].copy()


def compute_performance(portfolio_df, strategy_name=""):
    """
    Compute standard performance metrics for a long-short portfolio.

    Returns dict with:
    - mean_return (annualized)
    - std (annualized)
    - sharpe (annualized)
    - cumulative_return
    - max_drawdown
    - t_stat
    - n_months
    """
    if portfolio_df.empty:
        return {}

    ret = portfolio_df["ls_ret"]
    n = len(ret)

    mean_monthly = ret.mean()
    std_monthly = ret.std()

    # Annualize
    mean_annual = mean_monthly * 12
    std_annual = std_monthly * np.sqrt(12)
    sharpe = mean_annual / std_annual if std_annual > 0 else 0

    # Cumulative
    cum = (1 + ret).cumprod()
    cum_return = cum.iloc[-1] - 1

    # Max drawdown
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()

    # T-stat
    t_stat = mean_monthly / (std_monthly / np.sqrt(n)) if std_monthly > 0 else 0

    return {
        "strategy": strategy_name,
        "n_months": n,
        "mean_annual": mean_annual,
        "std_annual": std_annual,
        "sharpe": sharpe,
        "cum_return": cum_return,
        "max_drawdown": max_dd,
        "t_stat": t_stat,
        "mean_monthly": mean_monthly,
    }


def run_all_strategies(features_df, predictions_df, oos_start=OOS_START):
    """
    Construct portfolios for all strategies and compute performance.

    Args:
        features_df: DataFrame with features (needs MOM_12, fwd_return)
        predictions_df: DataFrame with model predictions (xgb_class, ret_score, srp_score)
        oos_start: start of out-of-sample period

    Returns:
        dict with portfolio DataFrames and performance metrics
    """
    results = {}

    # MOM
    mom_port = construct_mom_portfolio(features_df)
    mom_oos = filter_oos(mom_port, oos_start)
    results["MOM"] = {
        "portfolio": mom_oos,
        "metrics": compute_performance(mom_oos, "MOM"),
    }

    # XGB
    xgb_port = construct_xgb_portfolio(predictions_df)
    xgb_oos = filter_oos(xgb_port, oos_start)
    results["XGB"] = {
        "portfolio": xgb_oos,
        "metrics": compute_performance(xgb_oos, "XGB"),
    }

    # RET
    ret_port = construct_ret_portfolio(predictions_df)
    ret_oos = filter_oos(ret_port, oos_start)
    results["RET"] = {
        "portfolio": ret_oos,
        "metrics": compute_performance(ret_oos, "RET"),
    }

    # SRP
    srp_port = construct_srp_portfolio(predictions_df)
    srp_oos = filter_oos(srp_port, oos_start)
    results["SRP"] = {
        "portfolio": srp_oos,
        "metrics": compute_performance(srp_oos, "SRP"),
    }

    return results


def print_performance_table(results):
    """Print a formatted comparison table of all strategies."""
    print(f"\n{'Strategy':<10s} {'Ann.Ret':>10s} {'Ann.Vol':>10s} {'Sharpe':>8s} "
          f"{'Cum.Ret':>10s} {'MaxDD':>8s} {'t-stat':>8s} {'Months':>7s}")
    print("-" * 75)

    for name in ["MOM", "XGB", "RET", "SRP"]:
        if name not in results or not results[name]["metrics"]:
            print(f"{name:<10s} {'N/A':>10s}")
            continue
        m = results[name]["metrics"]
        print(f"{name:<10s} {m['mean_annual']:>9.1%} {m['std_annual']:>9.1%} "
              f"{m['sharpe']:>8.3f} {m['cum_return']:>9.1%} "
              f"{m['max_drawdown']:>7.1%} {m['t_stat']:>8.2f} {m['n_months']:>7d}")


# ═══════════════════════════════════════════════════════
if __name__ == "__main__":
    from pathlib import Path
    from config import CACHE_DIR, COUNTRIES
    from features import build_features

    cache_dir = Path(CACHE_DIR)
    suffix = "TO"
    _, country_name, _, _ = COUNTRIES[suffix]

    # Load filtered data and build features
    filtered_path = cache_dir / f"filtered_{suffix}.parquet"
    predictions_path = cache_dir / f"predictions_{suffix}.parquet"

    if not filtered_path.exists() or not predictions_path.exists():
        print("Need filtered data and predictions. Run data_filter.py and model.py first.")
    else:
        df = pd.read_parquet(filtered_path)
        df, feature_cols = build_features(df, country_name)
        predictions = pd.read_parquet(predictions_path)

        # Merge fwd_return into predictions (from features df)
        if "fwd_return" not in predictions.columns or predictions["fwd_return"].isna().all():
            fwd = df[["symbol", "date", "fwd_return"]].dropna()
            predictions = predictions.drop(columns=["fwd_return"], errors="ignore")
            predictions = predictions.merge(fwd, on=["symbol", "date"], how="left")

        print(f"\n  Running portfolio construction for {country_name}...")
        results = run_all_strategies(
            df, predictions,
            oos_start="2016-01-01",  # adjusted for our short test sample
        )

        print_performance_table(results)
