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
from config import N_CLASSES, OOS_START, TC_BPS, TOP_N_FIXED, FX_TC_BPS


def apply_fx_cost(portfolio_df, fx_tc_bps=FX_TC_BPS, tc_bps=TC_BPS):
    """
    Deduct FX roundtrip cost from an existing portfolio's ls_ret.
    Assumes the portfolio was built with compute_turnover_cost, so we can
    recover monthly turnover from the stored `tc` column:
        turnover = tc * 10000 / tc_bps
        fx_cost  = turnover * fx_tc_bps / 10000
    """
    if portfolio_df.empty or "tc" not in portfolio_df.columns:
        return portfolio_df
    df = portfolio_df.copy()
    turnover = df["tc"] * 10000 / tc_bps if tc_bps else 0.0
    df["fx_cost"] = turnover * fx_tc_bps / 10000
    df["ls_ret"] = df["ls_ret"] - df["fx_cost"]
    return df


def _pick_top_bottom(group, score_col, top_n):
    """
    Fixed-N selection: rank `group` by `score_col` desc and return
    (long_stocks, short_stocks) as the top-N and bottom-N rows.
    Returns (None, None) if the group doesn't have at least 2*top_n rows.
    """
    if len(group) < 2 * top_n:
        return None, None
    ranked = group.sort_values(score_col, ascending=False)
    long_stocks = ranked.head(top_n)
    short_stocks = ranked.tail(top_n)
    return long_stocks, short_stocks


def compute_turnover_cost(prev_long, prev_short, curr_long, curr_short, tc_bps=TC_BPS):
    """
    Compute turnover and transaction cost for one rebalance.

    Turnover = fraction of portfolio that changed (both legs).
    Equal-weighted: turnover = (stocks_traded / total_stocks) for each leg.
    Cost = turnover * tc_bps / 10000 (applied to both legs).

    Returns cost as a fraction of portfolio value (to subtract from return).
    """
    if not prev_long or not prev_short:
        return 0.0  # first month, no turnover

    # Long leg turnover
    prev_l = set(prev_long)
    curr_l = set(curr_long)
    n_long = max(len(prev_l), len(curr_l), 1)
    long_sold = len(prev_l - curr_l)
    long_bought = len(curr_l - prev_l)
    long_turnover = (long_sold + long_bought) / (2 * n_long)

    # Short leg turnover
    prev_s = set(prev_short)
    curr_s = set(curr_short)
    n_short = max(len(prev_s), len(curr_s), 1)
    short_sold = len(prev_s - curr_s)
    short_bought = len(curr_s - prev_s)
    short_turnover = (short_sold + short_bought) / (2 * n_short)

    # Average turnover across both legs, apply cost
    avg_turnover = (long_turnover + short_turnover) / 2
    cost = avg_turnover * tc_bps / 10000

    return cost


def construct_mom_portfolio(df, top_n=None):
    """
    Traditional momentum strategy.

    Paper Section 3.2:
    "the momentum strategy makes predictions based on the past eleven-month
    return with a one-month lag and buys (sells) stocks whose past returns
    belong to the highest (lowest) decile"

    MOM_12 in our features = cumulative return from t-11 to t-1 (11 months,
    skip current month). This is exactly the Jegadeesh-Titman momentum signal.

    Args:
        top_n: if None, decile selection (paper). If int, pick top_n/bottom_n
               stocks by MOM_12 instead.

    Returns DataFrame with monthly long-short returns.
    """
    df = df.copy()

    # Need MOM_12 and forward return
    required = ["MOM_12", "fwd_return", "date", "symbol"]
    df = df.dropna(subset=[c for c in required if c in df.columns])

    if df.empty:
        return pd.DataFrame()

    results = []
    prev_long_syms = []
    prev_short_syms = []

    for date, group in df.groupby(df["date"].dt.to_period("M")):
        if top_n is None:
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
        else:
            long_stocks, short_stocks = _pick_top_bottom(group, "MOM_12", top_n)
            if long_stocks is None:
                continue

        if long_stocks.empty or short_stocks.empty:
            continue

        curr_long_syms = long_stocks["symbol"].tolist()
        curr_short_syms = short_stocks["symbol"].tolist()

        # Transaction cost from turnover
        tc = compute_turnover_cost(prev_long_syms, prev_short_syms,
                                   curr_long_syms, curr_short_syms)

        # Equal-weighted returns
        long_ret = long_stocks["fwd_return"].mean()
        short_ret = short_stocks["fwd_return"].mean()
        ls_ret = long_ret - short_ret
        ls_ret_net = ls_ret - tc

        results.append({
            "date": group["date"].iloc[0],
            "long_ret": long_ret,
            "short_ret": short_ret,
            "ls_ret": ls_ret_net,
            "ls_ret_gross": ls_ret,
            "tc": tc,
            "n_long": len(long_stocks),
            "n_short": len(short_stocks),
            "strategy": "MOM",
        })

        prev_long_syms = curr_long_syms
        prev_short_syms = curr_short_syms

    return pd.DataFrame(results)


def construct_xgb_portfolio(predictions, top_n=None):
    """
    Naive ML strategy.

    Paper Section 4.2.2:
    For XGB, select all stocks classified into the highest or lowest return class.

    "buy stocks classified into highest class (10),
     sell stocks classified into lowest class (1)"

    Args:
        top_n: if None, class-membership selection (paper). If int, rank by
               scalar score `prob_10 - prob_1` and pick top_n/bottom_n —
               a symmetric fixed-N analogue to RET/SRP.
    """
    df = predictions.copy()
    df = df.dropna(subset=["xgb_class", "fwd_return"])

    if df.empty:
        return pd.DataFrame()

    results = []
    prev_long_syms = []
    prev_short_syms = []

    if top_n is not None:
        df = df.copy()
        df["_xgb_score"] = df[f"prob_{N_CLASSES}"] - df["prob_1"]

    for date, group in df.groupby(df["date"].dt.to_period("M")):
        if top_n is None:
            long_stocks = group[group["xgb_class"] == N_CLASSES]
            short_stocks = group[group["xgb_class"] == 1]
        else:
            long_stocks, short_stocks = _pick_top_bottom(group, "_xgb_score", top_n)
            if long_stocks is None:
                continue

        if long_stocks.empty or short_stocks.empty:
            continue

        curr_long_syms = long_stocks["symbol"].tolist()
        curr_short_syms = short_stocks["symbol"].tolist()

        tc = compute_turnover_cost(prev_long_syms, prev_short_syms,
                                   curr_long_syms, curr_short_syms)

        long_ret = long_stocks["fwd_return"].mean()
        short_ret = short_stocks["fwd_return"].mean()
        ls_ret = long_ret - short_ret
        ls_ret_net = ls_ret - tc

        results.append({
            "date": group["date"].iloc[0],
            "long_ret": long_ret,
            "short_ret": short_ret,
            "ls_ret": ls_ret_net,
            "ls_ret_gross": ls_ret,
            "tc": tc,
            "n_long": len(long_stocks),
            "n_short": len(short_stocks),
            "strategy": "XGB",
        })

        prev_long_syms = curr_long_syms
        prev_short_syms = curr_short_syms

    return pd.DataFrame(results)


def construct_ret_portfolio(predictions, top_n=None):
    """
    Deep Momentum (RET) strategy.

    Paper Section 3.3.2 / 4.2.2:
    "RET selects the top and bottom 10% of all stocks based on their
    predicted return" (i.e., predicted expected return from reclassification)

    Sort stocks by ret_score, buy top decile, sell bottom decile.

    Args:
        top_n: if None, decile selection (paper). If int, pick top_n/bottom_n
               stocks by ret_score instead.
    """
    df = predictions.copy()
    df = df.dropna(subset=["ret_score", "fwd_return"])

    if df.empty:
        return pd.DataFrame()

    results = []
    prev_long_syms = []
    prev_short_syms = []

    for date, group in df.groupby(df["date"].dt.to_period("M")):
        if top_n is None:
            if len(group) < N_CLASSES:
                continue

            group = group.copy()
            group["ret_decile"] = pd.qcut(
                group["ret_score"], q=N_CLASSES, labels=False, duplicates="drop"
            ) + 1

            long_stocks = group[group["ret_decile"] == N_CLASSES]
            short_stocks = group[group["ret_decile"] == 1]
        else:
            long_stocks, short_stocks = _pick_top_bottom(group, "ret_score", top_n)
            if long_stocks is None:
                continue

        if long_stocks.empty or short_stocks.empty:
            continue

        curr_long_syms = long_stocks["symbol"].tolist()
        curr_short_syms = short_stocks["symbol"].tolist()

        tc = compute_turnover_cost(prev_long_syms, prev_short_syms,
                                   curr_long_syms, curr_short_syms)

        long_ret = long_stocks["fwd_return"].mean()
        short_ret = short_stocks["fwd_return"].mean()
        ls_ret = long_ret - short_ret
        ls_ret_net = ls_ret - tc

        results.append({
            "date": group["date"].iloc[0],
            "long_ret": long_ret,
            "short_ret": short_ret,
            "ls_ret": ls_ret_net,
            "ls_ret_gross": ls_ret,
            "tc": tc,
            "n_long": len(long_stocks),
            "n_short": len(short_stocks),
            "strategy": "RET",
        })

        prev_long_syms = curr_long_syms
        prev_short_syms = curr_short_syms

    return pd.DataFrame(results)


def construct_srp_portfolio(predictions, top_n=None):
    """
    Deep Momentum (SRP) strategy — Sharpe ratio reclassification.

    Same as RET but sorts by predicted Sharpe ratio instead of expected return.

    Args:
        top_n: if None, decile selection (paper). If int, pick top_n/bottom_n
               stocks by srp_score instead.
    """
    df = predictions.copy()
    df = df.dropna(subset=["srp_score", "fwd_return"])

    if df.empty:
        return pd.DataFrame()

    results = []
    prev_long_syms = []
    prev_short_syms = []

    for date, group in df.groupby(df["date"].dt.to_period("M")):
        if top_n is None:
            if len(group) < N_CLASSES:
                continue

            group = group.copy()
            group["srp_decile"] = pd.qcut(
                group["srp_score"], q=N_CLASSES, labels=False, duplicates="drop"
            ) + 1

            long_stocks = group[group["srp_decile"] == N_CLASSES]
            short_stocks = group[group["srp_decile"] == 1]
        else:
            long_stocks, short_stocks = _pick_top_bottom(group, "srp_score", top_n)
            if long_stocks is None:
                continue

        if long_stocks.empty or short_stocks.empty:
            continue

        curr_long_syms = long_stocks["symbol"].tolist()
        curr_short_syms = short_stocks["symbol"].tolist()

        tc = compute_turnover_cost(prev_long_syms, prev_short_syms,
                                   curr_long_syms, curr_short_syms)

        long_ret = long_stocks["fwd_return"].mean()
        short_ret = short_stocks["fwd_return"].mean()
        ls_ret = long_ret - short_ret
        ls_ret_net = ls_ret - tc

        results.append({
            "date": group["date"].iloc[0],
            "long_ret": long_ret,
            "short_ret": short_ret,
            "ls_ret": ls_ret_net,
            "ls_ret_gross": ls_ret,
            "tc": tc,
            "n_long": len(long_stocks),
            "n_short": len(short_stocks),
            "strategy": "SRP",
        })

        prev_long_syms = curr_long_syms
        prev_short_syms = curr_short_syms

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


def run_all_strategies(features_df, predictions_df, oos_start=OOS_START, top_n=None):
    """
    Construct portfolios for all strategies and compute performance.

    Args:
        features_df: DataFrame with features (needs MOM_12, fwd_return)
        predictions_df: DataFrame with model predictions (xgb_class, ret_score, srp_score)
        oos_start: start of out-of-sample period
        top_n: if None, paper-faithful decile selection. If int (e.g. 10),
               fixed top-N / bottom-N per country-month for all strategies.

    Returns:
        dict with portfolio DataFrames and performance metrics
    """
    results = {}

    # MOM
    mom_port = construct_mom_portfolio(features_df, top_n=top_n)
    mom_oos = filter_oos(mom_port, oos_start)
    results["MOM"] = {
        "portfolio": mom_oos,
        "metrics": compute_performance(mom_oos, "MOM"),
    }

    # XGB
    xgb_port = construct_xgb_portfolio(predictions_df, top_n=top_n)
    xgb_oos = filter_oos(xgb_port, oos_start)
    results["XGB"] = {
        "portfolio": xgb_oos,
        "metrics": compute_performance(xgb_oos, "XGB"),
    }

    # RET
    ret_port = construct_ret_portfolio(predictions_df, top_n=top_n)
    ret_oos = filter_oos(ret_port, oos_start)
    results["RET"] = {
        "portfolio": ret_oos,
        "metrics": compute_performance(ret_oos, "RET"),
    }

    # SRP
    srp_port = construct_srp_portfolio(predictions_df, top_n=top_n)
    srp_oos = filter_oos(srp_port, oos_start)
    results["SRP"] = {
        "portfolio": srp_oos,
        "metrics": compute_performance(srp_oos, "SRP"),
    }

    return results


def print_performance_table(results):
    """Print a formatted comparison table of all strategies."""
    print(f"\n{'Strategy':<10s} {'Ann.Ret':>10s} {'Ann.Vol':>10s} {'Sharpe':>8s} "
          f"{'Cum.Ret':>10s} {'MaxDD':>8s} {'t-stat':>8s} {'Months':>7s} {'Avg TC':>8s}")
    print("-" * 83)

    for name in ["MOM", "XGB", "RET", "SRP"]:
        if name not in results or not results[name]["metrics"]:
            print(f"{name:<10s} {'N/A':>10s}")
            continue
        m = results[name]["metrics"]
        port = results[name]["portfolio"]
        avg_tc = port["tc"].mean() * 10000 if "tc" in port.columns else 0
        print(f"{name:<10s} {m['mean_annual']:>9.1%} {m['std_annual']:>9.1%} "
              f"{m['sharpe']:>8.3f} {m['cum_return']:>9.1%} "
              f"{m['max_drawdown']:>7.1%} {m['t_stat']:>8.2f} {m['n_months']:>7d} "
              f"{avg_tc:>6.1f}bp")


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
