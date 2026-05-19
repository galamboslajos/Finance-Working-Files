"""
Deep Momentum — Step 4: Feature engineering on the filtered monthly panel.

Reads:
  cache/ca_filtered_monthly.parquet   (output of data_filter.py)

Writes:
  cache/ca_features_monthly.parquet

Builds the paper's 16 features (Section 3.3.1 of Han & Qin 2026), with the
single substitution that SIZE_mt is the per-month decile of `turnover_mt`
(dollar-turnover proxy for market cap, since shares-outstanding is not in
the Norgate export).

Baseline features (all month-end values, suffix `_mt`):
  zMOM_1_mt   zMOM_3_mt   zMOM_6_mt   zMOM_9_mt   zMOM_12_mt   (cross-sectional z-score)
  MMOM_1_mt   MMOM_3_mt   MMOM_6_mt   MMOM_9_mt   MMOM_12_mt   (cross-sectional mean)
  SMOM_1_mt   SMOM_3_mt   SMOM_6_mt   SMOM_9_mt   SMOM_12_mt   (cross-sectional std)
  SIZE_mt                                                       (1..10, integer)

Additional tradability/state features are included when present:
  PRICE_DECILE_mt, TURNOVER_DECILE_mt, RANGE_HL_mt, CLOSE_POS_RANGE_mt,
  DIST_HIGH_mt, DIST_LOW_mt, DAILY_VOL_mt, DOWNSIDE_DAILY_VOL_mt,
  MAX_DAILY_RET_mt, MIN_DAILY_RET_mt, TURNOVER_RATIO_3_mt,
  TURNOVER_RATIO_12_mt, ZERO_VOLUME_SHARE_mt, TRADED_DAYS_mt

Raw momentum (kept for portfolio.py to use directly):
  MOM_1_mt    = r_t                                  (current-month return, reversal signal)
  MOM_m_mt    = ∏_{j=t-m+1}^{t-1} (1 + r_j) - 1      for m ∈ {3, 6, 9, 12}
                (Jegadeesh-Titman convention: skips current month)

Target (label):
  fwd_return_mt = next-month return per assetid (return_mt.shift(-1))
  LABEL_mt      = decile of fwd_return_mt within the country-month
                  (1 = lowest, 10 = highest), nullable Int

Gap-aware: when the filter removes an intermediate month for a stock, the
shift-based MOM/fwd-return calculations would silently use non-consecutive
months. We NaN-out those rows so they don't enter training or portfolio
construction.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
CACHE_DIR   = PROJECT_DIR / "cache"

FILTERED_PATH = CACHE_DIR / "ca_filtered_monthly.parquet"
FEATURES_PATH = CACHE_DIR / "ca_features_monthly.parquet"

MOMENTUM_HORIZONS = [1, 3, 6, 9, 12]
N_CLASSES = 10  # paper uses 10

OPTIONAL_STATE_FEATURES = [
    "PRICE_DECILE_mt",
    "TURNOVER_DECILE_mt",
    "RANGE_HL_mt",
    "CLOSE_POS_RANGE_mt",
    "DIST_HIGH_mt",
    "DIST_LOW_mt",
    "DAILY_VOL_mt",
    "DOWNSIDE_DAILY_VOL_mt",
    "MAX_DAILY_RET_mt",
    "MIN_DAILY_RET_mt",
    "TURNOVER_RATIO_3_mt",
    "TURNOVER_RATIO_12_mt",
    "ZERO_VOLUME_SHARE_mt",
    "TRADED_DAYS_mt",
]


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _gap_aware_mom(df: pd.DataFrame, m: int) -> pd.Series:
    """
    Compute MOM_m for m >= 3 with gap-awareness.

    MOM_m at month t uses returns from months t-m+1 ... t-1 (m-1 obs).
    Require that the (m-1)-prior row exists and is exactly (m-1) calendar
    months earlier; otherwise NaN.
    """
    df = df.sort_values(["assetid", "date_mt"])
    one_plus_r = 1.0 + df["return_mt"]
    cum = (one_plus_r.groupby(df["assetid"], sort=False)
                     .rolling(m - 1).apply(np.prod, raw=True)
                     .reset_index(level=0, drop=True))
    mom = (cum.groupby(df["assetid"], sort=False).shift(1) - 1)

    # Gap check: the row m-1 positions earlier must be exactly (m-1) months
    # before the current row.
    date_shifted = df.groupby("assetid", sort=False)["date_mt"].shift(m - 1)
    cur_period   = df["date_mt"].dt.to_period("M")
    shift_period = date_shifted.dt.to_period("M")
    mask_gap = shift_period.isna() | (shift_period != cur_period - (m - 1))
    mom = mom.where(~mask_gap, np.nan)
    return mom


def _gap_aware_fwd_return(df: pd.DataFrame) -> pd.Series:
    """fwd_return at t = return at t+1, but NaN if next surviving row isn't t+1."""
    df = df.sort_values(["assetid", "date_mt"])
    fwd = df.groupby("assetid", sort=False)["return_mt"].shift(-1)
    next_date = df.groupby("assetid", sort=False)["date_mt"].shift(-1)
    cur_period  = df["date_mt"].dt.to_period("M")
    next_period = next_date.dt.to_period("M")
    mask_gap = next_period.isna() | (next_period != cur_period + 1)
    return fwd.where(~mask_gap, np.nan)


def _xs_decile(s: pd.Series, n_classes: int = N_CLASSES) -> pd.Series:
    """qcut into n_classes deciles (1..n_classes); returns NaN where input is NaN."""
    valid = s.dropna()
    if len(valid) < n_classes:
        # Too few stocks — assign integer ranks 1..k instead
        n_bins = max(1, len(valid))
        ranks = pd.qcut(valid, q=n_bins, labels=False, duplicates="drop") + 1
    else:
        ranks = pd.qcut(valid, q=n_classes, labels=False, duplicates="drop") + 1
    out = pd.Series(np.nan, index=s.index)
    out.loc[valid.index] = ranks
    return out


# ─── Feature pipeline ────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Build 16 features + label on the filtered monthly panel.
    Input must contain: assetid, date_mt, return_mt, turnover_mt.
    """
    df = df.sort_values(["assetid", "date_mt"]).copy()

    if verbose:
        print(f"  Input rows: {len(df):,}, "
              f"unique assetids: {df['assetid'].nunique():,}")

    t0 = time.time()

    # ── 1. Raw momentum MOM_m_mt ──
    # m=1: short-term reversal, current-month return
    df["MOM_1_mt"] = df["return_mt"]
    # m∈{3,6,9,12}: Jegadeesh-Titman cumulative skipping current month
    for m in MOMENTUM_HORIZONS:
        if m == 1:
            continue
        df[f"MOM_{m}_mt"] = _gap_aware_mom(df, m)

    # ── 2. Cross-sectional z-score, mean, std per month for each MOM_m ──
    for m in MOMENTUM_HORIZONS:
        col = f"MOM_{m}_mt"
        df[f"MMOM_{m}_mt"] = df.groupby("date_mt")[col].transform("mean")
        df[f"SMOM_{m}_mt"] = df.groupby("date_mt")[col].transform("std")
        df[f"zMOM_{m}_mt"] = ((df[col] - df[f"MMOM_{m}_mt"])
                              / df[f"SMOM_{m}_mt"].replace(0, np.nan))

    # ── 3. SIZE_mt: per-month decile of turnover_mt (proxy for mcap) ──
    df["SIZE_mt"] = (df.groupby("date_mt", group_keys=False)["turnover_mt"]
                       .apply(_xs_decile)).astype("Int64")

    # ── 4. Daily-derived tradability/risk/liquidity state features ──
    if "last_price_mt" in df.columns:
        df["PRICE_DECILE_mt"] = (df.groupby("date_mt", group_keys=False)["last_price_mt"]
                                   .apply(_xs_decile)).astype("Int64")
    if "median_daily_turnover_mt" in df.columns:
        df["TURNOVER_DECILE_mt"] = (
            df.groupby("date_mt", group_keys=False)["median_daily_turnover_mt"]
              .apply(_xs_decile)
        ).astype("Int64")

    rename_map = {
        "range_hl_mt": "RANGE_HL_mt",
        "close_pos_range_mt": "CLOSE_POS_RANGE_mt",
        "dist_high_mt": "DIST_HIGH_mt",
        "dist_low_mt": "DIST_LOW_mt",
        "daily_vol_mt": "DAILY_VOL_mt",
        "downside_daily_vol_mt": "DOWNSIDE_DAILY_VOL_mt",
        "max_daily_ret_mt": "MAX_DAILY_RET_mt",
        "min_daily_ret_mt": "MIN_DAILY_RET_mt",
        "turnover_ratio_3_mt": "TURNOVER_RATIO_3_mt",
        "turnover_ratio_12_mt": "TURNOVER_RATIO_12_mt",
        "zero_volume_share_mt": "ZERO_VOLUME_SHARE_mt",
        "traded_days_mt": "TRADED_DAYS_mt",
    }
    for src_col, dst_col in rename_map.items():
        if src_col in df.columns:
            df[dst_col] = df[src_col]

    # ── 5. Forward return + label ──
    df["fwd_return_mt"] = _gap_aware_fwd_return(df)
    df["LABEL_mt"] = (df.groupby("date_mt", group_keys=False)["fwd_return_mt"]
                        .apply(_xs_decile)).astype("Int64")

    if verbose:
        feature_cols = get_feature_columns(df)
        n_complete = df[feature_cols + ["LABEL_mt"]].dropna().shape[0]
        print(f"  Features built in {time.time()-t0:.0f}s")
        print(f"  Rows with full features + label: {n_complete:,} "
              f"({100*n_complete/len(df):.1f}% of input)")
        print(f"  Feature columns ({len(feature_cols)}): {feature_cols}")

    return df


def get_feature_columns(df: pd.DataFrame | None = None,
                        include_state_features: bool = True) -> list[str]:
    """Canonical model input features, plus optional state features if present."""
    cols = []
    for m in MOMENTUM_HORIZONS:
        cols.extend([f"zMOM_{m}_mt", f"MMOM_{m}_mt", f"SMOM_{m}_mt"])
    cols.append("SIZE_mt")
    if include_state_features:
        if df is None:
            cols.extend(OPTIONAL_STATE_FEATURES)
        else:
            cols.extend([c for c in OPTIONAL_STATE_FEATURES if c in df.columns])
    return cols


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("Building features")
    print("=" * 70)

    if not FILTERED_PATH.exists():
        raise FileNotFoundError(
            f"Missing {FILTERED_PATH}. Run data_filter.py first."
        )

    t0 = time.time()
    filtered = pd.read_parquet(FILTERED_PATH)
    print(f"  Loaded {FILTERED_PATH.name} in {time.time()-t0:.0f}s")

    out = build_features(filtered)

    print(f"\nWriting {FEATURES_PATH}...")
    out.to_parquet(FEATURES_PATH, index=False, compression="snappy")
    size_mb = FEATURES_PATH.stat().st_size / (1024 * 1024)
    print(f"  wrote {size_mb:.0f} MB")

    print(f"\nTotal runtime: {time.time()-t0:.0f}s")
