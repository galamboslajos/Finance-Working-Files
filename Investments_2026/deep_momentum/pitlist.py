"""
Deep Momentum — Step 2b: Point-in-time index membership filter.

Reads:
  norgate_ca_pit_export.zip   (Norgate's index_constituent_timeseries export,
                               one zip with 7 indices × 3 file formats each)

Public API:
  load_membership_intervals(index_name) -> DataFrame[symbol, start_date, end_date]
  is_member(symbol, date, index_name) -> bool
  get_members(date, index_name) -> set[str]   (Norgate symbols, e.g. 'ABX.ca')
  filter_to_index(df, index_name, date_col, symbol_col) -> DataFrame

Available indices (per manifest):
  'sptsx_composite'                  ~225-300 members daily
  'sptsx_60'                          60 names
  's_p_tsx_completion'                Composite minus TSX 60 (~165)
  's_p_tsx_midcap_inferred'           ~85 names
  's_p_tsx_smallcap'                  ~200 names
  's_p_tsx_canadian_dividend_aristocrats'   ~80 dividend payers
  'all_ca_indexes'                    union across the above

Intervals are contiguous on Norgate TRADING DAYS (not calendar days). For
monthly filtering we treat membership as "covers calendar month M" iff the
interval overlaps any trading day in M. A symbol can re-enter the index
multiple times — each interval is a separate row.
"""

from __future__ import annotations
import zipfile
from pathlib import Path
from functools import lru_cache

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
PIT_ZIP     = PROJECT_DIR / "norgate_ca_pit_export.zip"

# index_name → filename inside the zip
_INTERVAL_FILES = {
    "sptsx_composite":               "sptsx_composite_pit_membership_intervals.csv",
    "s_p_tsx_composite":             "s_p_tsx_composite_pit_membership_intervals.csv",
    "sptsx_60":                      "s_p_tsx_60_pit_membership_intervals.csv",
    "s_p_tsx_60":                    "s_p_tsx_60_pit_membership_intervals.csv",
    "s_p_tsx_completion":            "s_p_tsx_completion_pit_membership_intervals.csv",
    "s_p_tsx_midcap_inferred":       "s_p_tsx_midcap_inferred_pit_membership_intervals.csv",
    "s_p_tsx_smallcap":              "s_p_tsx_smallcap_pit_membership_intervals.csv",
    "s_p_tsx_canadian_dividend_aristocrats":
        "s_p_tsx_canadian_dividend_aristocrats_pit_membership_intervals.csv",
    "all_ca_indexes":                "all_ca_indexes_pit_membership_intervals.csv",
}


# ─── Loaders ─────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_symbol_to_assetid() -> dict:
    """
    Build a symbol → assetid mapping from norgate_exports/metadata_all.csv.
    The metadata's `symbol` column has the full Norgate convention (e.g.
    'ABX.ca', 'A-200710.ca'), matching the pit-list intervals' symbol col.
    """
    meta_path = PROJECT_DIR / "norgate_exports" / "metadata_all.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} not found — needed to resolve "
                                "pit-list Norgate symbols to assetids")
    meta = pd.read_csv(meta_path, usecols=["symbol", "assetid"])
    return dict(zip(meta["symbol"], meta["assetid"].astype(int)))


@lru_cache(maxsize=8)
def load_membership_intervals(index_name: str = "sptsx_composite") -> pd.DataFrame:
    """
    Load the per-index intervals CSV from the zip. Resolves each Norgate-
    formatted symbol (e.g. 'ABX.ca') to an `assetid` so downstream filters
    can merge on the integer PK — robust to symbol-normalization differences
    in our cached parquets (our loader strips '.ca' and delisting suffixes).

    Returns DataFrame with columns:
        index_name, watchlist, symbol, assetid, start_date, end_date, trading_days
    Rows whose symbol can't be resolved to an assetid are dropped (rare;
    these would be index members that aren't in our equities panel).
    """
    if index_name not in _INTERVAL_FILES:
        raise ValueError(
            f"unknown index_name {index_name!r}. "
            f"available: {sorted(_INTERVAL_FILES.keys())}"
        )
    if not PIT_ZIP.exists():
        raise FileNotFoundError(f"{PIT_ZIP} not found — expected the Norgate pit export zip in the project dir")

    fname = _INTERVAL_FILES[index_name]
    with zipfile.ZipFile(PIT_ZIP) as zf:
        with zf.open(fname) as f:
            df = pd.read_csv(f)
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["end_date"]   = pd.to_datetime(df["end_date"])

    # Resolve Norgate symbols → assetids using metadata_all.csv
    sym_to_aid = _load_symbol_to_assetid()
    df["assetid"] = df["symbol"].map(sym_to_aid)
    n_unresolved = df["assetid"].isna().sum()
    if n_unresolved > 0:
        print(f"  WARN: {n_unresolved} pit-list intervals have unresolved symbols "
              f"(not in metadata_all.csv) — dropped")
    df = df.dropna(subset=["assetid"]).copy()
    df["assetid"] = df["assetid"].astype(int)
    return df.reset_index(drop=True)


# ─── Membership queries ──────────────────────────────────────────────────────

def is_member(symbol: str, date, index_name: str = "sptsx_composite") -> bool:
    """True iff `symbol` was a member of the index on `date`."""
    df = load_membership_intervals(index_name)
    d = pd.Timestamp(date)
    sub = df[df["symbol"] == symbol]
    return bool(((sub["start_date"] <= d) & (sub["end_date"] >= d)).any())


def get_members(date, index_name: str = "sptsx_composite") -> set:
    """Set of Norgate symbols that were members of the index on `date`."""
    df = load_membership_intervals(index_name)
    d = pd.Timestamp(date)
    mask = (df["start_date"] <= d) & (df["end_date"] >= d)
    return set(df.loc[mask, "symbol"].unique())


# ─── Bulk filter ─────────────────────────────────────────────────────────────

def filter_to_index(panel: pd.DataFrame,
                    index_name: str = "sptsx_composite",
                    date_col: str = "date_mt",
                    assetid_col: str = "assetid",
                    verbose: bool = True) -> pd.DataFrame:
    """
    Filter `panel` to only (assetid, date) pairs that were in `index_name` at
    that date. Membership check uses the calendar month containing `date`:
    a stock is "in" for month M iff its membership interval overlaps any day
    of month M.

    Joins on `assetid` (not `symbol`) because the panel's `symbol` column
    has been normalised (.ca and delisting suffixes stripped) while the
    pit-list export uses full Norgate symbols. assetid is the unambiguous PK.

    Args:
        panel:        monthly panel with `assetid_col` and `date_col` columns.
        index_name:   one of the keys in _INTERVAL_FILES.
        date_col:     panel column with month-end dates (default 'date_mt').
        assetid_col:  panel column with Norgate assetid (default 'assetid').

    Returns:
        Filtered DataFrame.
    """
    intervals = load_membership_intervals(index_name)
    n_before = len(panel)

    # Snap the panel's dates to month-end period
    panel = panel.copy()
    panel["_ym"] = panel[date_col].dt.to_period("M")

    # Expand intervals to (assetid, year-month) pairs the index covers
    iv = intervals[["assetid", "start_date", "end_date"]].copy()
    iv["start_ym"] = iv["start_date"].dt.to_period("M")
    iv["end_ym"]   = iv["end_date"].dt.to_period("M")

    rows = []
    for _, r in iv.iterrows():
        aid = int(r["assetid"])
        s_ym = r["start_ym"]
        e_ym = r["end_ym"]
        n_months = (e_ym - s_ym).n + 1
        for k in range(n_months):
            rows.append((aid, s_ym + k))
    member_pairs = pd.DataFrame(rows, columns=["assetid", "_ym"])
    member_pairs = member_pairs.drop_duplicates()

    # Inner join on (assetid, year-month)
    out = panel.merge(member_pairs, on=["assetid", "_ym"], how="inner").drop(columns=["_ym"])
    n_after = len(out)
    n_dropped = n_before - n_after

    if verbose:
        n_unique_in = panel[assetid_col].nunique()
        n_unique_out = out[assetid_col].nunique()
        print(f"  Pit-list filter ({index_name}): "
              f"dropped {n_dropped:,} obs ({100*n_dropped/max(n_before,1):.2f}%)")
        print(f"    universe shrinks from {n_unique_in:,} → {n_unique_out:,} unique assetids")
    return out


# ─── Main: smoke test ────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Sanity check: TSX Composite membership counts over time")
    print("=" * 60)
    df = load_membership_intervals("sptsx_composite")
    print(f"  {len(df):,} intervals, {df['symbol'].nunique():,} unique symbols ever")
    print(f"  date range: {df['start_date'].min().date()} → {df['end_date'].max().date()}")

    # Members at end of each year
    print("\nMembers at year-end (sample):")
    for year in [1992, 1995, 2000, 2005, 2010, 2015, 2020, 2024]:
        d = pd.Timestamp(f"{year}-12-31")
        mask = (df["start_date"] <= d) & (df["end_date"] >= d)
        n = mask.sum()
        print(f"  {year}-12-31:  {n:>4d} members")

    print("\n" + "=" * 60)
    print("Available indices:")
    for k in _INTERVAL_FILES:
        print(f"  {k}")
