"""
Deep Momentum — Step 6: Metrics
Bimodality measure, crash rate, classification accuracy, and reporting.

Paper reference: Sections 3.2, 4.1, 4.2.1

Bimodality measure (Section 3.2):
- HH = precision of predicted winners - 0.1
- HL = proportion of actual losers among predicted winners - 0.1
- LL = precision of predicted losers - 0.1
- LH = proportion of actual winners among predicted losers - 0.1
- BM = -(HH - HL + LL - LH) / 2

Crash rate (Section 4.1):
- Number of years where max drawdown > 20%, divided by total years
- Calculated 12 times shifting by one month, averaged

Classification accuracy (Section 4.2.1):
- Overall accuracy = correct predictions / total
- Precision, recall per class
"""

import pandas as pd
import numpy as np
from config import N_CLASSES


def compute_bimodality(predictions_df, actual_col="fwd_return", pred_col="xgb_class"):
    """
    Paper Section 3.2, Equations (1)-(5):

    Stocks are classified based on their PAST return (momentum) into
    winners (top decile) and losers (bottom decile). Then we check
    where they end up in FUTURE return deciles.

    HH = TP / (TP + FP) - 0.1  (precision of winners)
    HL = FP / (TP + FP) - 0.1  (losers among predicted winners)
    LL = TN / (TN + FN) - 0.1  (precision of losers)
    LH = FN / (TN + FN) - 0.1  (winners among predicted losers)

    BM = -(HH - HL + LL - LH) / 2

    Here "positive" = winners (H), "negative" = losers (L)
    TP = predicted winner, actually winner
    FP = predicted winner, actually loser
    TN = predicted loser, actually loser
    FN = predicted loser, actually winner
    """
    df = predictions_df.copy()
    df = df.dropna(subset=[actual_col])

    if df.empty:
        return {}

    results = []

    for date, group in df.groupby(df["date"].dt.to_period("M")):
        if len(group) < N_CLASSES:
            continue

        # Actual future return deciles
        group = group.copy()
        group["actual_decile"] = pd.qcut(
            group[actual_col], q=N_CLASSES, labels=False, duplicates="drop"
        ) + 1

        # Predicted class (momentum-based for MOM, or xgb_class for ML)
        pred_winners = group[group[pred_col] == N_CLASSES]
        pred_losers = group[group[pred_col] == 1]

        if pred_winners.empty or pred_losers.empty:
            continue

        # Among predicted winners: how many are actual winners (H) vs losers (L)?
        tp = (pred_winners["actual_decile"] == N_CLASSES).sum()
        fp = (pred_winners["actual_decile"] == 1).sum()
        n_pred_winners = len(pred_winners)

        # Among predicted losers: how many are actual losers (L) vs winners (H)?
        tn = (pred_losers["actual_decile"] == 1).sum()
        fn = (pred_losers["actual_decile"] == N_CLASSES).sum()
        n_pred_losers = len(pred_losers)

        if n_pred_winners == 0 or n_pred_losers == 0:
            continue

        hh = tp / n_pred_winners - 0.1
        hl = fp / n_pred_winners - 0.1
        ll = tn / n_pred_losers - 0.1
        lh = fn / n_pred_losers - 0.1

        bm = -((hh - hl) + (ll - lh)) / 2

        results.append({
            "date": group["date"].iloc[0],
            "HH": hh, "HL": hl, "LL": ll, "LH": lh, "BM": bm,
        })

    if not results:
        return {}

    res_df = pd.DataFrame(results)
    return {
        "HH": res_df["HH"].mean(),
        "HL": res_df["HL"].mean(),
        "LL": res_df["LL"].mean(),
        "LH": res_df["LH"].mean(),
        "BM": res_df["BM"].mean(),
    }


def compute_crash_rate(portfolio_df, dd_threshold=0.20):
    """
    Paper Section 4.1:
    "The crash rate is defined as the number of years in which the maximum
    drawdown (MDD) exceeds 20%, divided by the total number of years in
    the sample. Since the return could fall more than 20% across two years,
    the crash rate is calculated twelve times, shifting the sample by a month,
    and the average is reported."
    """
    if portfolio_df.empty or len(portfolio_df) < 12:
        return np.nan

    ret = portfolio_df["ls_ret"].values
    dates = pd.to_datetime(portfolio_df["date"].values)

    crash_rates = []

    for shift in range(12):
        # Shift the starting month
        shifted_ret = ret[shift:]
        shifted_dates = dates[shift:]

        if len(shifted_ret) < 12:
            continue

        # Group into non-overlapping 12-month windows
        n_years = len(shifted_ret) // 12
        crash_count = 0

        for y in range(n_years):
            year_ret = shifted_ret[y * 12:(y + 1) * 12]
            cum = np.cumprod(1 + year_ret)
            peak = np.maximum.accumulate(cum)
            dd = (cum - peak) / peak
            mdd = dd.min()

            if mdd < -dd_threshold:
                crash_count += 1

        if n_years > 0:
            crash_rates.append(crash_count / n_years)

    if crash_rates:
        return np.mean(crash_rates)
    return np.nan


def compute_classification_accuracy(predictions_df):
    """
    Paper Section 4.2.1:
    - Overall accuracy = correct predictions / total
    - Precision of H (highest class) and L (lowest class)
    - Recall of H and L
    - Prediction ratio (proportion classified into each class)
    """
    df = predictions_df.copy()
    df = df.dropna(subset=["fwd_return", "xgb_class"])

    if df.empty:
        return {}

    # Compute actual deciles per month
    all_results = []
    for date, group in df.groupby(df["date"].dt.to_period("M")):
        if len(group) < N_CLASSES:
            continue
        group = group.copy()
        group["actual_decile"] = pd.qcut(
            group["fwd_return"], q=N_CLASSES, labels=False, duplicates="drop"
        ) + 1
        all_results.append(group)

    if not all_results:
        return {}

    df = pd.concat(all_results, ignore_index=True)

    # Overall accuracy
    correct = (df["xgb_class"] == df["actual_decile"]).sum()
    accuracy = correct / len(df)

    # Precision and recall for H (class N_CLASSES) and L (class 1)
    pred_h = df[df["xgb_class"] == N_CLASSES]
    pred_l = df[df["xgb_class"] == 1]
    actual_h = df[df["actual_decile"] == N_CLASSES]
    actual_l = df[df["actual_decile"] == 1]

    precision_h = (pred_h["actual_decile"] == N_CLASSES).mean() if len(pred_h) > 0 else 0
    precision_l = (pred_l["actual_decile"] == 1).mean() if len(pred_l) > 0 else 0
    recall_h = (actual_h["xgb_class"] == N_CLASSES).mean() if len(actual_h) > 0 else 0
    recall_l = (actual_l["xgb_class"] == 1).mean() if len(actual_l) > 0 else 0

    pred_ratio_h = len(pred_h) / len(df) if len(df) > 0 else 0
    pred_ratio_l = len(pred_l) / len(df) if len(df) > 0 else 0

    return {
        "accuracy": accuracy,
        "precision_H": precision_h,
        "precision_L": precision_l,
        "recall_H": recall_h,
        "recall_L": recall_l,
        "pred_ratio_H": pred_ratio_h,
        "pred_ratio_L": pred_ratio_l,
    }


def full_report(features_df, predictions_df, portfolio_results, country_name=""):
    """
    Print a comprehensive report matching the paper's tables.
    """
    print(f"\n{'='*70}")
    print(f"FULL REPORT — {country_name}")
    print(f"{'='*70}")

    # 1. Portfolio performance (Table 5)
    print(f"\n--- Portfolio Performance (Table 5 equivalent) ---")
    print(f"{'Strategy':<10s} {'Ann.Ret':>10s} {'Ann.Vol':>10s} {'Sharpe':>8s} "
          f"{'Cum.Ret':>10s} {'MaxDD':>8s} {'t-stat':>8s} {'Months':>7s}")
    print("-" * 75)

    for name in ["MOM", "XGB", "RET", "SRP"]:
        if name not in portfolio_results or not portfolio_results[name]["metrics"]:
            print(f"{name:<10s} {'N/A':>10s}")
            continue
        m = portfolio_results[name]["metrics"]
        print(f"{name:<10s} {m['mean_annual']:>9.1%} {m['std_annual']:>9.1%} "
              f"{m['sharpe']:>8.3f} {m['cum_return']:>9.1%} "
              f"{m['max_drawdown']:>7.1%} {m['t_stat']:>8.2f} {m['n_months']:>7d}")

    # 2. Bimodality (Table 2 equivalent)
    print(f"\n--- Bimodality (Table 2 equivalent) ---")
    # MOM bimodality: use MOM_12 decile as predictor
    if "MOM_12" in features_df.columns and "fwd_return" in features_df.columns:
        mom_df = features_df.dropna(subset=["MOM_12", "fwd_return"]).copy()
        # Assign MOM decile per month
        all_mom = []
        for date, group in mom_df.groupby(mom_df["date"].dt.to_period("M")):
            if len(group) < N_CLASSES:
                continue
            group = group.copy()
            group["mom_decile"] = pd.qcut(
                group["MOM_12"], q=N_CLASSES, labels=False, duplicates="drop"
            ) + 1
            all_mom.append(group)
        if all_mom:
            mom_df = pd.concat(all_mom, ignore_index=True)
            mom_bm = compute_bimodality(mom_df, actual_col="fwd_return", pred_col="mom_decile")
            if mom_bm:
                print(f"  MOM:  HH={mom_bm['HH']:.3f}  HL={mom_bm['HL']:.3f}  "
                      f"LL={mom_bm['LL']:.3f}  LH={mom_bm['LH']:.3f}  BM={mom_bm['BM']:.3f}")

    # XGB bimodality
    if not predictions_df.empty:
        xgb_bm = compute_bimodality(predictions_df, actual_col="fwd_return", pred_col="xgb_class")
        if xgb_bm:
            print(f"  XGB:  HH={xgb_bm['HH']:.3f}  HL={xgb_bm['HL']:.3f}  "
                  f"LL={xgb_bm['LL']:.3f}  LH={xgb_bm['LH']:.3f}  BM={xgb_bm['BM']:.3f}")

    # 3. Crash rate
    print(f"\n--- Crash Rate ---")
    for name in ["MOM", "XGB", "RET", "SRP"]:
        if name in portfolio_results and not portfolio_results[name]["portfolio"].empty:
            cr = compute_crash_rate(portfolio_results[name]["portfolio"])
            print(f"  {name}: {cr:.3f}" if not np.isnan(cr) else f"  {name}: N/A")

    # 4. Classification accuracy (Table IA2 equivalent)
    if not predictions_df.empty:
        print(f"\n--- Classification Accuracy (Table IA2 equivalent) ---")
        acc = compute_classification_accuracy(predictions_df)
        if acc:
            print(f"  Overall accuracy: {acc['accuracy']:.1%}")
            print(f"  Precision H: {acc['precision_H']:.1%}  "
                  f"Precision L: {acc['precision_L']:.1%}")
            print(f"  Recall H: {acc['recall_H']:.1%}  "
                  f"Recall L: {acc['recall_L']:.1%}")
            print(f"  Pred ratio H: {acc['pred_ratio_H']:.1%}  "
                  f"Pred ratio L: {acc['pred_ratio_L']:.1%}")


# ═══════════════════════════════════════════════════════
if __name__ == "__main__":
    from pathlib import Path
    from config import CACHE_DIR, COUNTRIES
    from features import build_features
    from portfolio import run_all_strategies

    cache_dir = Path(CACHE_DIR)
    suffix = "TO"
    _, country_name, _, _ = COUNTRIES[suffix]

    filtered_path = cache_dir / f"filtered_{suffix}.parquet"
    predictions_path = cache_dir / f"predictions_{suffix}.parquet"

    if not filtered_path.exists() or not predictions_path.exists():
        print("Need filtered data and predictions. Run data_filter.py and model.py first.")
    else:
        df = pd.read_parquet(filtered_path)
        df, feature_cols = build_features(df, country_name)
        predictions = pd.read_parquet(predictions_path)

        # Merge fwd_return
        fwd = df[["symbol", "date", "fwd_return"]].dropna()
        predictions = predictions.drop(columns=["fwd_return"], errors="ignore")
        predictions = predictions.merge(fwd, on=["symbol", "date"], how="left")

        # Run portfolios (adjusted OOS for test sample)
        results = run_all_strategies(df, predictions, oos_start="2016-01-01")

        # Full report
        full_report(df, predictions, results, country_name)
