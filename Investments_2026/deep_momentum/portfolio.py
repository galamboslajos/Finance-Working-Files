"""
Deep Momentum — Step 6: Portfolio construction.

Reads:
  cache/ca_features_monthly.parquet     (for MOM_12_mt — used by MOM strategy)
  cache/ca_predictions_monthly.parquet  (for xgb_class, ret_score, srp_score, cvr_score)

Writes (when run as __main__):
  cache/ca_portfolios.parquet           (all five strategies stacked)

Strategies (top_n per leg, paper-departing for realism):
  MOM    long top_n / short bottom_n by MOM_12_mt
  XGB    long top_n / short bottom_n by (prob_10 - prob_1)
  RET    long top_n / short bottom_n by ret_score
  SRP    long top_n / short bottom_n by srp_score
  CVR    long top_n / short bottom_n by cvr_score        (extension)

Selection is top-N each leg (default 15) — NOT paper's decile cuts, which
produce hundreds of names and aren't CFD-tradeable in practice.

Cost model (per-stock-month attribution, two-pass; trade-log reconciles):
  Commission: tc_bps × {entry + exit} events / 10000
              entry = stock new in current month vs previous
              exit  = stock not present in next month
              persistent positions pay 0 commission that month
  Financing : carry_long_annual  × days/365 on each long
              carry_short_annual × days/365 EARNED on each short
              defaults: long 5%/yr (Saxo-style), short 2%/yr (IBKR-style)

All cost params are kwargs and adjustable from run.ipynb.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
CACHE_DIR   = PROJECT_DIR / "cache"

FEATURES_PATH    = CACHE_DIR / "ca_features_monthly.parquet"
PREDICTIONS_PATH = CACHE_DIR / "ca_predictions_monthly.parquet"
PORTFOLIOS_PATH  = CACHE_DIR / "ca_portfolios.parquet"


# ─── Cost helpers ────────────────────────────────────────────────────────────

def _period_cost(curr: list, prev: list, nxt: list,
                 tc_bps: float, carry_annual: float, days: int = 30):
    """
    Commission + financing for one leg over the holding month [t → t+1].

    Entry events: names in curr but not in prev.
    Exit events:  names in curr but not in nxt (their exit will be charged here).

    Commission per position is multiplied by 1/n_leg because each name carries
    1/n_leg of the leg's notional under equal weighting. Sum across positions
    gives the leg-level commission as a fraction of leg notional.

    Returns (commission, financing) — both fractions of leg notional.
    """
    n_leg = max(len(curr), 1)
    curr_set, prev_set, nxt_set = set(curr), set(prev), set(nxt)
    entries = len(curr_set - prev_set)
    exits   = len(curr_set - nxt_set)
    commission = (entries + exits) / n_leg * tc_bps / 10000
    financing  = carry_annual * days / 365
    return commission, financing


# ─── Strategy selection ──────────────────────────────────────────────────────

def _resolve_n(top_n, n_total: int) -> int:
    """
    Convert top_n into an absolute count of names per leg.
      - top_n > 1 (int)        → fixed N            (e.g. top_n=15 → 15 names)
      - 0 < top_n <= 1 (float) → percentile of cross-section
                                  (e.g. top_n=0.10 on 250 names → 25 per leg)
    """
    if top_n is None:
        return 0
    if top_n > 1:
        return int(top_n)
    return max(1, int(round(top_n * n_total)))


def _select_top_bottom(group: pd.DataFrame, score_col: str, top_n):
    """Sort group desc by score; return (top, bottom) DataFrames or (None, None).
    top_n is either fixed int or percentile float (see _resolve_n)."""
    n_total = len(group)
    n = _resolve_n(top_n, n_total)
    if n_total < 2 * n or n < 1:
        return None, None
    g = group.sort_values(score_col, ascending=False)
    return g.head(n), g.tail(n)


def _select_xgb_class(group: pd.DataFrame, top_n, n_classes: int = 10):
    """
    XGB-specific: rank by (prob_{N} - prob_1) — symmetric class-membership score.
    Allows top-N or percentile selection consistent with other strategies.
    """
    n_total = len(group)
    n = _resolve_n(top_n, n_total)
    if n_total < 2 * n or n < 1:
        return None, None
    g = group.copy()
    g["_xgb_score"] = g[f"prob_{n_classes}"] - g["prob_1"]
    g = g.sort_values("_xgb_score", ascending=False)
    return g.head(n), g.tail(n)


# ─── Two-pass strategy runner ────────────────────────────────────────────────

def _build_selections(df: pd.DataFrame, score_col: str, top_n: int,
                       date_col: str, return_col: str, *,
                       use_xgb_class: bool = False, n_classes: int = 10):
    """
    Loop months in order, compute long/short rows + names per month.
    Returns list of dicts: {date, long_df, short_df, long_names, short_names}.
    """
    sub = df.dropna(subset=[score_col, return_col]) if not use_xgb_class \
          else df.dropna(subset=[f"prob_{n_classes}", "prob_1", return_col])
    if sub.empty:
        return []

    # Group by CALENDAR MONTH (not raw date_mt) — each stock's last trading day
    # within a month can vary (holidays, mid-month delistings). Without snapping
    # to month, a Toronto stock with date_mt=2007-03-30 and a Calgary stock with
    # date_mt=2007-03-29 would land in two different "months" and get scored in
    # isolation. We want one cross-section per calendar month.
    sub = sub.copy()
    sub["_ym"] = sub[date_col].dt.to_period("M")

    selections = []
    for ym, grp in sub.groupby("_ym"):
        if use_xgb_class:
            longs, shorts = _select_xgb_class(grp, top_n, n_classes)
        else:
            longs, shorts = _select_top_bottom(grp, score_col, top_n)
        if longs is None or longs.empty or shorts.empty:
            continue
        # Use the latest stock-level date_mt within this month as the "rebalance date"
        rebal_date = grp[date_col].max()
        selections.append({
            "date":         rebal_date,
            "long_df":      longs,
            "short_df":     shorts,
            "long_names":   longs["assetid"].tolist(),
            "short_names":  shorts["assetid"].tolist(),
        })
    return selections


def _run_strategy(selections, strategy_name: str,
                   tc_bps: float, carry_long_annual: float,
                   carry_short_annual: float, days: int,
                   return_col: str = "fwd_return_mt",
                   long_only: bool = False):
    """
    Aggregate selections into a per-month portfolio DataFrame with cost breakdown.

    long_only=True drops the short leg entirely: short_ret=0, no short
    commission, no short financing earned. Useful for diagnosing whether short-leg
    blow-ups are corrupting the strategy's reported P&L.
    """
    if not selections:
        return pd.DataFrame()

    rows = []
    for i, sel in enumerate(selections):
        prev = selections[i - 1] if i > 0 else None
        nxt  = selections[i + 1] if i + 1 < len(selections) else None
        prev_l = prev["long_names"]  if prev else []
        prev_s = prev["short_names"] if prev else []
        nxt_l  = nxt["long_names"]   if nxt  else []
        nxt_s  = nxt["short_names"]  if nxt  else []

        comm_l, fin_l = _period_cost(sel["long_names"],  prev_l, nxt_l,
                                      tc_bps, carry_long_annual,  days)
        if long_only:
            comm_s, fin_s = 0.0, 0.0
        else:
            comm_s, fin_s = _period_cost(sel["short_names"], prev_s, nxt_s,
                                          tc_bps, carry_short_annual, days)

        # Long financing is a cost; short financing is earned (subtract from cost).
        commission = comm_l + comm_s
        financing  = fin_l - fin_s
        tc = commission + financing

        long_ret  = sel["long_df"][return_col].mean()
        short_ret = 0.0 if long_only else sel["short_df"][return_col].mean()
        ls_gross  = long_ret - short_ret  # long-only: ls_gross = long_ret

        rows.append({
            "date_mt":       sel["date"],
            "strategy":      strategy_name,
            "n_long":        len(sel["long_names"]),
            "n_short":       0 if long_only else len(sel["short_names"]),
            "long_ret":      long_ret,
            "short_ret":     short_ret,
            "ls_ret_gross":  ls_gross,
            "commission":    commission,
            "financing":     financing,
            "tc":            tc,
            "ls_ret":        ls_gross - tc,
        })

    return pd.DataFrame(rows)


# ─── Per-strategy public constructors ────────────────────────────────────────

def construct_mom(features: pd.DataFrame, top_n = 15,
                   tc_bps: float = 20.0,
                   carry_long_annual: float = 0.05,
                   carry_short_annual: float = 0.02,
                   days: int = 30,
                   long_only: bool = False) -> pd.DataFrame:
    """MOM: rank by MOM_12_mt (Jegadeesh-Titman 11-month momentum, skip-1)."""
    sels = _build_selections(features, score_col="MOM_12_mt", top_n=top_n,
                              date_col="date_mt", return_col="fwd_return_mt")
    return _run_strategy(sels, "MOM", tc_bps, carry_long_annual,
                          carry_short_annual, days, long_only=long_only)


def construct_xgb(predictions: pd.DataFrame, top_n = 15,
                   tc_bps: float = 20.0,
                   carry_long_annual: float = 0.05,
                   carry_short_annual: float = 0.02,
                   days: int = 30,
                   n_classes: int = 10,
                   long_only: bool = False) -> pd.DataFrame:
    """XGB: rank by (prob_{N} - prob_1) — symmetric class-edge score."""
    sels = _build_selections(predictions, score_col="", top_n=top_n,
                              date_col="date_mt", return_col="fwd_return_mt",
                              use_xgb_class=True, n_classes=n_classes)
    return _run_strategy(sels, "XGB", tc_bps, carry_long_annual,
                          carry_short_annual, days, long_only=long_only)


def construct_ret(predictions: pd.DataFrame, top_n = 15,
                   tc_bps: float = 20.0,
                   carry_long_annual: float = 0.05,
                   carry_short_annual: float = 0.02,
                   days: int = 30,
                   long_only: bool = False) -> pd.DataFrame:
    """RET: rank by ret_score (predicted expected return)."""
    sels = _build_selections(predictions, score_col="ret_score", top_n=top_n,
                              date_col="date_mt", return_col="fwd_return_mt")
    return _run_strategy(sels, "RET", tc_bps, carry_long_annual,
                          carry_short_annual, days, long_only=long_only)


def construct_srp(predictions: pd.DataFrame, top_n = 15,
                   tc_bps: float = 20.0,
                   carry_long_annual: float = 0.05,
                   carry_short_annual: float = 0.02,
                   days: int = 30,
                   long_only: bool = False) -> pd.DataFrame:
    """SRP: rank by srp_score (predicted Sharpe)."""
    sels = _build_selections(predictions, score_col="srp_score", top_n=top_n,
                              date_col="date_mt", return_col="fwd_return_mt")
    return _run_strategy(sels, "SRP", tc_bps, carry_long_annual,
                          carry_short_annual, days, long_only=long_only)


def construct_cvr(predictions: pd.DataFrame, top_n = 15,
                   tc_bps: float = 20.0,
                   carry_long_annual: float = 0.05,
                   carry_short_annual: float = 0.02,
                   days: int = 30,
                   long_only: bool = False) -> pd.DataFrame:
    """CVR: rank by cvr_score (predicted return-over-CVaR; extension)."""
    sels = _build_selections(predictions, score_col="cvr_score", top_n=top_n,
                              date_col="date_mt", return_col="fwd_return_mt")
    return _run_strategy(sels, "CVR", tc_bps, carry_long_annual,
                          carry_short_annual, days, long_only=long_only)


# ─── Performance + driver ────────────────────────────────────────────────────

def compute_performance(port: pd.DataFrame, name: str = "") -> dict:
    """Standard L/S metrics. Operates on `ls_ret` (net of costs)."""
    if port.empty:
        return {}
    r = port["ls_ret"]
    n = len(r)
    mean_m, std_m = r.mean(), r.std()
    cum = (1 + r).cumprod()
    peak = cum.cummax()
    return {
        "strategy":     name,
        "n_months":     n,
        "mean_annual":  mean_m * 12,
        "std_annual":   std_m * np.sqrt(12),
        "sharpe":       (mean_m * 12) / (std_m * np.sqrt(12)) if std_m > 0 else 0,
        "cum_return":   cum.iloc[-1] - 1,
        "max_drawdown": ((cum - peak) / peak).min(),
        "t_stat":       mean_m / (std_m / np.sqrt(n)) if std_m > 0 else 0,
        "mean_monthly": mean_m,
        "avg_comm_bps": port["commission"].mean() * 10000,
        "avg_fin_bps":  port["financing" ].mean() * 10000,
    }


def run_all_strategies(features: pd.DataFrame, predictions: pd.DataFrame,
                        top_n = 15,
                        tc_bps: float = 20.0,
                        carry_long_annual: float = 0.05,
                        carry_short_annual: float = 0.02,
                        days: int = 30,
                        long_only: bool = False,
                        verbose: bool = True) -> dict:
    """
    Run all five strategies. Returns:
        {strategy: {'portfolio': DataFrame, 'metrics': dict}}

    long_only=True drops the short leg from every strategy — useful for a
    diagnostic comparison vs the L/S book (the short leg is the source of
    monthly compounding blow-ups when a single name surges 5×+).
    """
    common = dict(top_n=top_n, tc_bps=tc_bps,
                  carry_long_annual=carry_long_annual,
                  carry_short_annual=carry_short_annual,
                  days=days, long_only=long_only)
    constructors = {
        "MOM": (construct_mom, features),
        "XGB": (construct_xgb, predictions),
        "RET": (construct_ret, predictions),
        "SRP": (construct_srp, predictions),
        "CVR": (construct_cvr, predictions),
    }
    out = {}
    for name, (fn, src) in constructors.items():
        port = fn(src, **common)
        out[name] = {"portfolio": port,
                     "metrics":   compute_performance(port, name)}
        if verbose and not port.empty:
            m = out[name]["metrics"]
            print(f"  {name}: ann.ret {m['mean_annual']:>7.1%}  "
                  f"sharpe {m['sharpe']:>5.2f}  "
                  f"comm {m['avg_comm_bps']:>5.1f}bp/mo  "
                  f"fin {m['avg_fin_bps']:>5.1f}bp/mo  "
                  f"months {m['n_months']}")
    return out


def print_performance_table(results: dict):
    """Formatted table — same shape as legacy."""
    print(f"\n{'Strategy':<8s} {'Ann.Ret':>9s} {'Ann.Vol':>9s} {'Sharpe':>8s} "
          f"{'Cum.Ret':>10s} {'MaxDD':>8s} {'t-stat':>8s} {'Months':>7s} "
          f"{'Comm':>7s} {'Fin':>7s}")
    print("-" * 90)
    for name in ["MOM", "XGB", "RET", "SRP", "CVR"]:
        if name not in results or not results[name]["metrics"]:
            print(f"{name:<8s} N/A")
            continue
        m = results[name]["metrics"]
        print(f"{name:<8s} {m['mean_annual']:>8.1%} {m['std_annual']:>8.1%} "
              f"{m['sharpe']:>8.3f} {m['cum_return']:>9.1%} "
              f"{m['max_drawdown']:>7.1%} {m['t_stat']:>8.2f} {m['n_months']:>7d} "
              f"{m['avg_comm_bps']:>5.1f}bp {m['avg_fin_bps']:>5.1f}bp")


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("Building portfolios (top-15 per leg, default cost params)")
    print("=" * 70)

    if not FEATURES_PATH.exists() or not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Need both {FEATURES_PATH.name} and {PREDICTIONS_PATH.name}. "
            "Run features.py and model.py first."
        )

    t0 = time.time()
    features    = pd.read_parquet(FEATURES_PATH)
    predictions = pd.read_parquet(PREDICTIONS_PATH)

    # Predictions need fwd_return_mt for realised P&L; merge from features panel
    if "fwd_return_mt" not in predictions.columns or predictions["fwd_return_mt"].isna().all():
        merge_keys = ["assetid", "date_mt"]
        fwd = features[merge_keys + ["fwd_return_mt"]].dropna()
        predictions = predictions.drop(columns=["fwd_return_mt"], errors="ignore")
        predictions = predictions.merge(fwd, on=merge_keys, how="left")

    results = run_all_strategies(features, predictions,
                                  top_n=15, tc_bps=20.0,
                                  carry_long_annual=0.05,
                                  carry_short_annual=0.02)

    print_performance_table(results)

    # Stack all five portfolios into one parquet with `strategy` column
    all_ports = pd.concat([r["portfolio"] for r in results.values() if not r["portfolio"].empty],
                           ignore_index=True)
    all_ports.to_parquet(PORTFOLIOS_PATH, index=False, compression="snappy")
    print(f"\nWrote {PORTFOLIOS_PATH} ({all_ports.shape[0]:,} rows)")
    print(f"Total runtime: {time.time()-t0:.0f}s")
