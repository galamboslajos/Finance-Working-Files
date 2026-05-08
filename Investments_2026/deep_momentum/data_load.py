"""
Deep Momentum — Step 1: Load Norgate exports into unified daily parquets.

Reads from:
  norgate_exports/CA Equities/             (currently-listed Canadian stocks)
  norgate_exports/CA Equities Delisted/    (historical delisted Canadian stocks)
  norgate_exports/Forex Spot/              (CADUSD spot for USD-base conversion)
  norgate_exports/metadata_all.csv         (per-symbol classification + exchange)

Writes:
  cache/ca_equities_daily.parquet          (every CA equity, listed AND delisted,
                                            daily OHLCV + classification flags)
  cache/cad_fx_daily.parquet               (CADUSD daily close)

No filtering is applied here. Classification columns (is_operating_company,
subtype1, exchange_name, etc.) are MERGED IN so downstream scripts can filter
freely without re-loading.

Symbol convention:
  Listed:    "ABX.ca"            → symbol = "ABX",   is_delisted = False
  Delisted:  "GLA.U-201307.ca"   → symbol = "GLA.U", is_delisted = True,
                                                     delist_date = last bar date
  assetid (Norgate stable PK) is the unique key — symbols can collide
  across active/delisted reuse, so always merge on assetid.
"""

import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
NORGATE_DIR = PROJECT_DIR / "norgate_exports"
CACHE_DIR   = PROJECT_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)

PRICE_COLS_RENAME = {
    "Date": "date",
    "Open": "open",
    "High": "high",
    "Low":  "low",
    "Close": "close",
    "Volume": "volume",
    "Turnover": "turnover",
    "Unadjusted Close": "unadjusted_close",
    "Dividend": "dividend",
}

DELIST_SUFFIX_RE = re.compile(r"-(\d{6})\.ca$")  # captures "-YYYYMM" before ".ca"


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _parse_filename(filename: str) -> tuple[int, str, bool]:
    """
    'norgate' filenames look like:
        1509941_ABX.ca.csv               (listed)
        1507901_GLA.U-201307.ca.csv      (delisted, suffix = last trading YYYYMM)

    Returns (assetid, symbol, is_delisted). The symbol is normalised:
    trailing '.ca' stripped; for delisted, the '-YYYYMM' suffix also stripped.
    """
    stem = filename[:-4] if filename.endswith(".csv") else filename
    aid_str, _, rest = stem.partition("_")
    assetid = int(aid_str)
    is_delisted = bool(DELIST_SUFFIX_RE.search(rest))
    if is_delisted:
        symbol = DELIST_SUFFIX_RE.sub("", rest)
    else:
        symbol = rest[:-3] if rest.endswith(".ca") else rest
    return assetid, symbol, is_delisted


def _read_one_csv(path: Path, assetid: int, symbol: str, is_delisted: bool) -> pd.DataFrame:
    """Read one Norgate price CSV and return a DataFrame with normalised columns."""
    df = pd.read_csv(path)
    if df.empty:
        return df  # empty shell file (some Norgate delisted entries have no rows)
    df = df.rename(columns=PRICE_COLS_RENAME)
    df["date"] = pd.to_datetime(df["date"])
    df["assetid"] = assetid
    df["symbol"] = symbol
    df["is_delisted"] = is_delisted
    if is_delisted:
        df["delist_date"] = df["date"].iloc[-1]
    else:
        df["delist_date"] = pd.NaT
    return df


def load_norgate_equities(verbose: bool = True) -> pd.DataFrame:
    """
    Walk both CA folders, read every CSV, concat into one daily panel.
    Merges in classification columns from metadata_all.csv on assetid.
    """
    folders = [
        ("CA Equities",          False),
        ("CA Equities Delisted", True),
    ]

    frames: list[pd.DataFrame] = []
    n_files_total = 0
    t0 = time.time()

    for folder_name, expected_delisted in folders:
        folder = NORGATE_DIR / folder_name
        if not folder.exists():
            print(f"  WARN: {folder} not found, skipping")
            continue

        files = [f for f in os.scandir(folder)
                 if f.name.endswith(".csv") and f.name != "metadata.csv"]
        n_files = len(files)
        n_files_total += n_files
        if verbose:
            print(f"  {folder_name}: {n_files:,} CSVs")

        for i, f in enumerate(files):
            try:
                aid, sym, is_del = _parse_filename(f.name)
                df = _read_one_csv(Path(f.path), aid, sym, is_del)
                if not df.empty:
                    frames.append(df)
            except Exception as e:
                print(f"    ERR {f.name}: {e}")

            if verbose and (i + 1) % 2000 == 0:
                print(f"    {folder_name}: {i+1:,}/{n_files:,}  ({time.time()-t0:.0f}s)")

    if not frames:
        return pd.DataFrame()

    if verbose:
        print(f"  Concatenating {len(frames):,} frames...")
    daily = pd.concat(frames, ignore_index=True)
    if verbose:
        print(f"  Concatenated: {len(daily):,} rows  ({time.time()-t0:.0f}s)")

    # Merge classification metadata (per assetid)
    meta_path = NORGATE_DIR / "metadata_all.csv"
    if meta_path.exists():
        meta = pd.read_csv(meta_path)
        keep_cols = ["assetid", "securityname", "subtype1", "subtype2", "subtype3",
                     "exchange_name", "exchange_name_full", "is_operating_company"]
        meta = meta[keep_cols].drop_duplicates(subset=["assetid"])
        daily = daily.merge(meta, on="assetid", how="left")
    else:
        print(f"  WARN: {meta_path} not found — no classification columns")

    daily["currency"] = "CAD"

    # Tidy ordering
    leading = ["assetid", "symbol", "date", "is_delisted", "delist_date",
               "open", "high", "low", "close", "volume", "turnover",
               "unadjusted_close", "dividend",
               "securityname", "subtype1", "subtype2", "subtype3",
               "exchange_name", "exchange_name_full", "is_operating_company",
               "currency"]
    cols = [c for c in leading if c in daily.columns] + \
           [c for c in daily.columns if c not in leading]
    daily = daily[cols].sort_values(["assetid", "date"]).reset_index(drop=True)

    return daily


def load_cadusd(verbose: bool = True) -> pd.DataFrame:
    """Load Forex Spot/CADUSD daily series."""
    fx_dir = NORGATE_DIR / "Forex Spot"
    if not fx_dir.exists():
        print(f"  WARN: {fx_dir} not found")
        return pd.DataFrame()

    candidates = [f for f in os.scandir(fx_dir)
                  if "CADUSD" in f.name.upper() and f.name.endswith(".csv")]
    if not candidates:
        print("  WARN: no CADUSD file in Forex Spot/")
        return pd.DataFrame()

    df = pd.read_csv(candidates[0].path)
    df = df.rename(columns={c: c.lower() for c in df.columns})
    df["date"] = pd.to_datetime(df["date"])
    df["pair"] = "CADUSD"
    df = df[["date", "pair", "open", "high", "low", "close"]].sort_values("date")
    if verbose:
        print(f"  CADUSD: {len(df):,} rows, {df['date'].min().date()} → {df['date'].max().date()}")
    return df


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("Loading Norgate CA equities (listed + delisted)")
    print("=" * 70)
    t0 = time.time()
    daily = load_norgate_equities()
    print(f"\nLoaded daily panel in {time.time()-t0:.0f}s:")
    print(f"  rows:           {len(daily):,}")
    print(f"  unique assetids:{daily['assetid'].nunique():,}")
    print(f"  date range:     {daily['date'].min().date()} → {daily['date'].max().date()}")
    print(f"  is_delisted:    {daily['is_delisted'].sum():,} rows from delisted "
          f"({daily.loc[daily['is_delisted'], 'assetid'].nunique():,} unique)")
    if "is_operating_company" in daily.columns:
        ops = daily.loc[daily["is_operating_company"] == True, "assetid"].nunique()
        print(f"  operating cos:  {ops:,} unique assetids "
              f"(of {daily['assetid'].nunique():,} total)")

    out = CACHE_DIR / "ca_equities_daily.parquet"
    print(f"\nWriting {out}...")
    daily.to_parquet(out, index=False, compression="snappy")
    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"  wrote {size_mb:.0f} MB")

    print("\n" + "=" * 70)
    print("Loading CADUSD")
    print("=" * 70)
    fx = load_cadusd()
    if not fx.empty:
        out_fx = CACHE_DIR / "cad_fx_daily.parquet"
        fx.to_parquet(out_fx, index=False, compression="snappy")
        print(f"  wrote {out_fx} ({out_fx.stat().st_size/1024:.0f} KB)")

    print(f"\nTotal runtime: {time.time()-t0:.0f}s")
