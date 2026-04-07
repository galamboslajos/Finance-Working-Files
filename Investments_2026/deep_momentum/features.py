"""
Deep Momentum — Step 3: Feature Construction
Builds the 16 features exactly as described in Section 3.3.1 of the paper.

Features (16 total):
- 5 cross-sectionally z-scored momentum features: zMOM_1, zMOM_3, zMOM_6, zMOM_9, zMOM_12
- 5 cross-sectional means: MMOM_1, MMOM_3, MMOM_6, MMOM_9, MMOM_12
- 5 cross-sectional stds: SMOM_1, SMOM_3, SMOM_6, SMOM_9, SMOM_12
- 1 size feature: SIZE (market cap decile 1-10 within country per month)

Target variable:
- LABEL: return decile (1-10) based on next month's return, within country per month
"""

import pandas as pd
import numpy as np
from config import MOMENTUM_HORIZONS, N_CLASSES


def compute_momentum(df):
    """
    Compute raw momentum features per stock.

    Paper Section 3.3.1:
    - MOM_1 = previous one-month return r_{i,t}
    - MOM_m = cumulative return from t-m+1 to t-1 for m = 3, 6, 9, 12

    Note: at month t, we predict month t+1. So:
    - MOM_1 = return at month t (already in the 'return' column)
    - MOM_3 = cumulative return over months t-2 to t (3 months ending at t, skip nothing)
      But paper says "from t-m+1 to t-1" — so MOM_3 uses months t-2 to t-1 (2 months).
      Wait — re-reading: for m=3,6,9,12 it's t-m+1 to t-1. For m=1, it's r_{i,t}.

    So MOM_m (m>1) skips the current month:
    - MOM_3 = (1+r_{t-2})(1+r_{t-1}) - 1   (2 months: t-2, t-1)
    Actually re-reading more carefully:
    - MOM_m = product from j=t-m+1 to t-1 of (1+r_j) - 1

    For m=3: j goes from t-2 to t-1 → 2 months
    For m=6: j goes from t-5 to t-1 → 5 months
    For m=9: j goes from t-8 to t-1 → 9-1=8 months? No:
      t-m+1 = t-9+1 = t-8, to t-1 → months t-8, t-7, ..., t-1 = 8 months

    Wait that gives m-1 months for m>1. Let me re-read equation (6):
    MOM_m = prod_{j=t-m+1}^{t-1} (r_{i,j} + 1) - 1, for m = 3,6,9,12

    For m=12: prod from t-11 to t-1 = 11 months.
    This is the standard Jegadeesh-Titman convention: 12-month momentum
    skips the most recent month (short-term reversal) and uses months t-11 to t-1.

    For m=1: MOM_1 = r_{i,t} (the current month return, i.e. the reversal signal)

    So:
    - MOM_1 = return at t (1 month)
    - MOM_3 = cumulative return from t-2 to t-1 (2 months, skip current)
    - MOM_6 = cumulative return from t-5 to t-1 (5 months, skip current)
    - MOM_9 = cumulative return from t-8 to t-1 (8 months, skip current)
    - MOM_12 = cumulative return from t-11 to t-1 (11 months, skip current)
    """
    df = df.sort_values(["symbol", "date"]).copy()

    for m in MOMENTUM_HORIZONS:
        if m == 1:
            # MOM_1 = current month return
            df[f"MOM_{m}"] = df["return"]
        else:
            # MOM_m = cumulative return from t-m+1 to t-1 (skip current month)
            # = rolling product of (1+r) over m-1 months, shifted by 1
            df[f"_1pr"] = 1 + df["return"]
            cum = (
                df.groupby("symbol")["_1pr"]
                .rolling(m - 1)
                .apply(np.prod, raw=True)
                .reset_index(level=0, drop=True)
            )
            # Shift by 1 to exclude current month (use t-m+1 to t-1, not t-m+2 to t)
            df[f"MOM_{m}"] = df.groupby("symbol")[cum.name].shift(1) if cum.name else None
            # Actually cum doesn't have a name, let me fix this:
            df[f"_cum_{m}"] = cum.values
            df[f"MOM_{m}"] = df.groupby("symbol")[f"_cum_{m}"].shift(1) - 1
            df = df.drop(columns=[f"_cum_{m}"])

    if "_1pr" in df.columns:
        df = df.drop(columns=["_1pr"])

    return df


def compute_cross_sectional_features(df):
    """
    Paper Section 3.3.1:
    - zMOM_m = (MOM_m - MMOM_m) / SMOM_m  (cross-sectional z-score per month)
    - MMOM_m = cross-sectional mean of MOM_m that month
    - SMOM_m = cross-sectional std of MOM_m that month

    These are computed within each country, per month.
    """
    df = df.copy()
    df["_ym"] = df["date"].dt.to_period("M")

    for m in MOMENTUM_HORIZONS:
        col = f"MOM_{m}"
        if col not in df.columns:
            continue

        # Cross-sectional mean and std per month
        mean_col = f"MMOM_{m}"
        std_col = f"SMOM_{m}"

        df[mean_col] = df.groupby("_ym")[col].transform("mean")
        df[std_col] = df.groupby("_ym")[col].transform("std")

        # Z-score
        df[f"zMOM_{m}"] = (df[col] - df[mean_col]) / df[std_col].replace(0, np.nan)

    df = df.drop(columns=["_ym"])
    return df


def compute_size_feature(df):
    """
    Paper Section 3.3.1:
    "The size feature, S_i, is a categorical variable that represents stocks'
    size deciles and has a value between 1 and 10."

    Computed within country per month based on market cap.
    """
    df = df.copy()
    date_col = df["date"].dt.to_period("M")

    def assign_decile(group):
        if len(group) < N_CLASSES:
            # Not enough stocks to form 10 deciles — use available bins
            n_bins = max(1, len(group))
            group["SIZE"] = pd.qcut(
                group["marketCap"], q=n_bins, labels=False, duplicates="drop"
            ) + 1
        else:
            group["SIZE"] = pd.qcut(
                group["marketCap"], q=N_CLASSES, labels=False, duplicates="drop"
            ) + 1
        return group

    df = df.groupby(date_col, group_keys=False).apply(assign_decile)

    return df


def compute_target(df):
    """
    Paper Section 3.3.1 / 3.2:
    Target = return decile (1-10) based on NEXT month's return.
    Stocks sorted into 10 deciles cross-sectionally within country per month.
    1 = lowest return, 10 = highest return.
    """
    df = df.sort_values(["symbol", "date"]).copy()

    # Forward return = next month's return
    df["fwd_return"] = df.groupby("symbol")["return"].shift(-1)

    # Assign deciles per month based on forward return
    date_col = df["date"].dt.to_period("M")

    def assign_label(group):
        fwd = group["fwd_return"]
        valid = fwd.dropna()
        if len(valid) < N_CLASSES:
            n_bins = max(1, len(valid))
            group.loc[valid.index, "LABEL"] = pd.qcut(
                valid, q=n_bins, labels=False, duplicates="drop"
            ) + 1
        else:
            group.loc[valid.index, "LABEL"] = pd.qcut(
                valid, q=N_CLASSES, labels=False, duplicates="drop"
            ) + 1
        return group

    df["LABEL"] = np.nan
    df = df.groupby(date_col, group_keys=False).apply(assign_label)
    df["LABEL"] = df["LABEL"].astype("Int64")  # nullable integer

    return df


def build_features(df, country_name=""):
    """
    Full feature pipeline for one country.
    Input: filtered monthly DataFrame (from data_filter.py)
    Output: DataFrame with 16 features + target label

    Feature columns: zMOM_1, zMOM_3, zMOM_6, zMOM_9, zMOM_12,
                     MMOM_1, MMOM_3, MMOM_6, MMOM_9, MMOM_12,
                     SMOM_1, SMOM_3, SMOM_6, SMOM_9, SMOM_12,
                     SIZE
    Target column:   LABEL (1-10)
    """
    print(f"  Building features for {country_name}...")

    # 1. Momentum
    df = compute_momentum(df)

    # 2. Cross-sectional z-scores, means, stds
    df = compute_cross_sectional_features(df)

    # 3. Size decile
    df = compute_size_feature(df)

    # 4. Target
    df = compute_target(df)

    # Define feature columns
    feature_cols = []
    for m in MOMENTUM_HORIZONS:
        feature_cols.extend([f"zMOM_{m}", f"MMOM_{m}", f"SMOM_{m}"])
    feature_cols.append("SIZE")

    # Report
    n_complete = df[feature_cols + ["LABEL"]].dropna().shape[0]
    print(f"    Stocks: {df['symbol'].nunique()}")
    print(f"    Total obs: {len(df)}")
    print(f"    Complete obs (all features + label): {n_complete}")
    print(f"    Feature cols: {len(feature_cols)}")

    return df, feature_cols


# ═══════════════════════════════════════════════════════
if __name__ == "__main__":
    from pathlib import Path
    from config import CACHE_DIR, COUNTRIES

    cache_dir = Path(CACHE_DIR)
    files = sorted(cache_dir.glob("filtered_*.parquet"))

    if not files:
        print("No filtered data found. Run data_filter.py first.")
    else:
        for f in files:
            suffix = f.stem.replace("filtered_", "")
            if suffix not in COUNTRIES:
                continue
            _, country_name, _, _ = COUNTRIES[suffix]
            df = pd.read_parquet(f)

            df, feature_cols = build_features(df, country_name)

            # Save
            out_path = cache_dir / f"features_{suffix}.parquet"
            df.to_parquet(out_path, index=False)
            print(f"    Saved: {out_path}")

            # Show sample
            sample = df[feature_cols + ["LABEL"]].dropna().head(3)
            print(f"    Sample:\n{sample.to_string()}\n")
