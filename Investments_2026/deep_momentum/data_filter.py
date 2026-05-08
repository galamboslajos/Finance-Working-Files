"""
Deep Momentum — Step 3: Filter monthly panel to a tradable universe.

Reads:
  cache/ca_equities_monthly.parquet   (output of data_aggregate.py)

Writes:
  cache/ca_filtered_monthly.parquet

Filter sequence (all point-in-time, no look-ahead):
  1. Drop rows with NaN return_mt (first observation per assetid).
  2. Zero-volume rule (paper-faithful):
        drop if volume_mt == 0 AND volume_mt_prev == 0 AND return_mt == 0
  3. Liquidity-bottom-pct (replaces paper's mcap-bottom-5%):
        for each month t, rank all alive stocks by trailing X-month avg
        turnover (in CAD), drop the bottom `liq_pct` of that month's
        cross-section. Strictly past data.
  4. Operating-company gate (replaces paper's `typecode='EQ'`):
        keep only is_operating_company == True (Norgate's classification)
  5. Optional exchange filter (kwarg `exchanges`).

Defaults match the paper's spirit: 5% liquidity bottom, 12-month lookback,
6-month warm-up, all four CA exchanges allowed.

NO winsorize, NO extreme-returns clip — both removed for honest realised P&L.
The pipeline never clips per-stock returns; portfolio-level compounding is
the only place this can interact, and that's handled in portfolio.py.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
CACHE_DIR   = PROJECT_DIR / "cache"

MONTHLY_PATH  = CACHE_DIR / "ca_equities_monthly.parquet"
FILTERED_PATH = CACHE_DIR / "ca_filtered_monthly.parquet"


# ─── Individual filters ──────────────────────────────────────────────────────

def filter_zero_volume(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Drop obs with zero volume both months AND zero return (stale ticker)."""
    n_before = len(df)
    mask = (
        (df["volume_mt"].fillna(0) == 0) &
        (df["volume_mt_prev"].fillna(0) == 0) &
        (df["return_mt"].fillna(0).abs() < 1e-10)
    )
    out = df[~mask].copy()
    n_dropped = n_before - len(out)
    if verbose and n_dropped > 0:
        print(f"  Zero-volume filter:  dropped {n_dropped:,} obs "
              f"({100*n_dropped/n_before:.2f}%)")
    return out


def filter_liquidity_bottom(df: pd.DataFrame, pct: float = 0.05,
                             lookback: int = 12, min_periods: int = 6,
                             verbose: bool = True) -> pd.DataFrame:
    """
    Drop stock-months where the TURNOVER RATIO (current month vs trailing avg)
    is in the bottom `pct` of that month's cross-section.

    Liquidity proxy:   liq_t = turnover_mt_t / mean(turnover_mt over [t-lookback+1, t])
    Cross-sectional:   per-month rank, drop bottom `pct`.

    A ratio < 1 means this month was quieter than the stock's recent average;
    ranking cross-sectionally normalises for absolute size differences and
    surfaces stocks that are illiquid RELATIVE to their own history.

    Point-in-time: only past+current data, no look-ahead. A stock with
    fewer than `min_periods` of history is excluded (cannot evaluate ratio).
    """
    df = df.sort_values(["assetid", "date_mt"]).copy()

    # Trailing average turnover per assetid — denominator
    avg = (df.groupby("assetid", sort=False)["turnover_mt"]
             .rolling(lookback, min_periods=min_periods).mean()
             .reset_index(level=0, drop=True))

    # Liquidity ratio = current month / trailing avg
    df["_liq"] = df["turnover_mt"] / avg.replace(0, np.nan)

    n_before = len(df)
    n_no_history = df["_liq"].isna().sum()

    # Per-month cross-sectional threshold on the ratio
    df["_thresh"] = (df.groupby("date_mt")["_liq"]
                       .transform(lambda x: x.quantile(pct)))

    keep = df["_liq"].notna() & (df["_liq"] >= df["_thresh"])
    out = df[keep].drop(columns=["_liq", "_thresh"])

    n_dropped_low_liq = (~keep).sum() - n_no_history
    if verbose:
        print(f"  Liquidity filter (bottom {pct:.0%} by turnover ratio, "
              f"{lookback}m lookback):")
        print(f"    insufficient history (<{min_periods}m): {n_no_history:,} obs")
        print(f"    below ratio threshold:                  {n_dropped_low_liq:,} obs")
        print(f"    total dropped:                          "
              f"{n_before - len(out):,} ({100*(n_before-len(out))/n_before:.2f}%)")
    return out


def filter_operating_companies(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Keep only is_operating_company == True (Norgate classification)."""
    if "is_operating_company" not in df.columns:
        if verbose:
            print("  Operating-company filter: column missing, skipped")
        return df
    n_before = len(df)
    out = df[df["is_operating_company"] == True].copy()
    n_dropped = n_before - len(out)
    if verbose:
        print(f"  Operating-company filter: dropped {n_dropped:,} obs "
              f"({100*n_dropped/n_before:.2f}%)  "
              f"[ETFs/CEFs/SPACs/hybrids/derivatives]")
    return out


def filter_exchanges(df: pd.DataFrame, exchanges: list[str] | None = None,
                     verbose: bool = True) -> pd.DataFrame:
    """
    If `exchanges` is None: pass-through (keep all). Otherwise restrict to
    the named exchanges (matching `exchange_name` column, e.g. 'TSX',
    'TSX Venture', 'CSE', 'NEO').
    """
    if exchanges is None or "exchange_name" not in df.columns:
        return df
    n_before = len(df)
    out = df[df["exchange_name"].isin(exchanges)].copy()
    n_dropped = n_before - len(out)
    if verbose:
        print(f"  Exchange filter (keep {exchanges}): "
              f"dropped {n_dropped:,} obs ({100*n_dropped/n_before:.2f}%)")
    return out


# ─── Master ──────────────────────────────────────────────────────────────────

def filter_monthly(monthly: pd.DataFrame,
                    liq_pct: float = 0.05,
                    liq_lookback: int = 12,
                    liq_min_periods: int = 6,
                    operating_only: bool = True,
                    exchanges: list[str] | None = None,
                    verbose: bool = True) -> pd.DataFrame:
    """
    Apply the full filter sequence to the monthly panel.

    Args:
        liq_pct:         bottom-X cross-sectional liquidity cut (default 5%).
        liq_lookback:    trailing months for avg turnover (default 12).
        liq_min_periods: minimum months of history to be evaluated (default 6).
        operating_only:  keep only is_operating_company == True.
        exchanges:       optional whitelist (e.g. ['TSX', 'TSX Venture']);
                         None keeps all.

    Returns:
        Filtered DataFrame ready for features.py.
    """
    if verbose:
        print(f"\nInput rows: {len(monthly):,}, "
              f"unique assetids: {monthly['assetid'].nunique():,}")

    df = monthly.dropna(subset=["return_mt"]).copy()
    if verbose:
        n_dropped = len(monthly) - len(df)
        print(f"  Drop NaN return_mt:  dropped {n_dropped:,} obs (first month per assetid)")

    df = filter_zero_volume(df, verbose=verbose)
    df = filter_liquidity_bottom(df, pct=liq_pct,
                                  lookback=liq_lookback,
                                  min_periods=liq_min_periods,
                                  verbose=verbose)
    if operating_only:
        df = filter_operating_companies(df, verbose=verbose)
    df = filter_exchanges(df, exchanges=exchanges, verbose=verbose)

    if verbose:
        print(f"\nFinal: {len(df):,} rows, "
              f"{df['assetid'].nunique():,} unique assetids")
    return df


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("Filtering monthly panel")
    print("=" * 70)

    if not MONTHLY_PATH.exists():
        raise FileNotFoundError(
            f"Missing {MONTHLY_PATH}. Run data_aggregate.py first."
        )

    t0 = time.time()
    monthly = pd.read_parquet(MONTHLY_PATH)
    print(f"  Loaded {MONTHLY_PATH.name} in {time.time()-t0:.0f}s")

    filtered = filter_monthly(
        monthly,
        liq_pct=0.05,
        liq_lookback=12,
        liq_min_periods=6,
        operating_only=True,
        exchanges=None,           # all CA venues; tighten in run.ipynb if desired
    )

    print(f"\nWriting {FILTERED_PATH}...")
    filtered.to_parquet(FILTERED_PATH, index=False, compression="snappy")
    size_mb = FILTERED_PATH.stat().st_size / (1024 * 1024)
    print(f"  wrote {size_mb:.0f} MB")

    print(f"\nTotal runtime: {time.time()-t0:.0f}s")
