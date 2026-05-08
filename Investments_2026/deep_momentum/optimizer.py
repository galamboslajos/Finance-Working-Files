"""
Deep Momentum — Step 7: Mean-CVaR two-leg optimizer (Vorobets-style).

Reads:
  cache/ca_equities_daily.parquet           (daily lookback for cov + scenarios)
  cache/ca_predictions_monthly.parquet      (monthly model scores → monthly selections)
  cache/ca_features_monthly.parquet         (fwd_return_mt for realised P&L)

Writes (when run as __main__):
  cache/ca_optimizer_portfolio.parquet      (optimised L/S monthly returns + costs)
  cache/ca_optimizer_weights.parquet        (per-month weight panel: date_mt × symbol)

Architecture (Vorobets canonical pattern, Chapter 9):
  Each rebalance month t (where the score-based strategy picks 15L + 15S):
    1. Build daily-return matrix R_long (S × 15) and R_short (S × 15) over the
       last `lookback_days` (default 504 = 2y) for the selected names.
    2. For each leg, run B parameter-uncertainty bootstraps:
         (a) MVN-sample bootstrap means: N×B draws from MVN(empirical_mean, cov)
             → produces B candidate mean vectors `means_b`
         (b) Entropy-pool the historical scenario probabilities to satisfy
             `q' R = means_b` (re-weights history; preserves higher moments)
         (c) Optimize MeanCVaR(R, G=G, h=h, p=q, alpha=cvar_alpha) and grab
             the efficient frontier (I × P).
    3. Take the middle frontier point (`pf_index=4` of P=9) from each
       bootstrap iteration → matrix of B candidate weights per leg.
    4. Apply Exposure Stacking (`ft.exposure_stacking(L, ...)`) → stable
       single weight vector per leg.
    5. Apply costs (commission + financing) using the same per-stock-month
       attribution as portfolio.py for reconciliation.

Two-leg convention:
  - Long leg: optimize on positive returns of the long picks; max E[r]/min CVaR.
  - Short leg: optimize on NEGATED returns of the short picks (so the short
    leg's CVaR penalises tail loss-of-shorting = stocks surging).
  - Combined L/S P&L = long_ret_realised − short_ret_realised, applied
    monthly using fwd_return_mt of each name.

Liquidity / history sanity:
  - A name needs >= `min_history_days` of daily history at month t (default
    252 = 1y). Names with insufficient history are dropped from that month's
    optimization — the leg may end up with fewer than top_n names; that's
    fine, the LP just optimizes the smaller universe.

All params are kwargs; defaults are Vorobets-canonical.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import fortitudo.tech as ft
    _HAS_FT = True
except ImportError:
    _HAS_FT = False


PROJECT_DIR = Path(__file__).resolve().parent
CACHE_DIR   = PROJECT_DIR / "cache"

DAILY_PATH        = CACHE_DIR / "ca_equities_daily.parquet"
PREDICTIONS_PATH  = CACHE_DIR / "ca_predictions_monthly.parquet"
FEATURES_PATH     = CACHE_DIR / "ca_features_monthly.parquet"
OPT_PORT_PATH     = CACHE_DIR / "ca_optimizer_portfolio.parquet"
OPT_WEIGHTS_PATH  = CACHE_DIR / "ca_optimizer_weights.parquet"


# Vorobets-style defaults
LOOKBACK_DAYS    = 504    # 2 years daily
MIN_HISTORY_DAYS = 252    # 1 year minimum per stock
B_RESAMPLES      = 200
N_RESAMPLE       = 100
P_FRONTIER       = 9
PF_INDEX         = 4
L_STACK          = 20
CVAR_ALPHA       = 0.10   # Vorobets convention: tail mass
W_MAX            = 0.30


# ─── Single-leg optimizer ────────────────────────────────────────────────────

def optimize_leg(R: np.ndarray,
                 w_max: float = W_MAX,
                 alpha: float = CVAR_ALPHA,
                 B: int = B_RESAMPLES,
                 N_resample: int = N_RESAMPLE,
                 P: int = P_FRONTIER,
                 pf_idx: int = PF_INDEX,
                 L: int = L_STACK,
                 seed: int | None = None) -> np.ndarray:
    """
    Vorobets-canonical Mean-CVaR + parameter-uncertainty + Exposure-Stacking
    on one leg.

    Args:
        R: daily-return matrix shape (S, I) for the I stocks in the leg.
        w_max: cap on any single name (default 30% of leg notional).
        alpha: CVaR tail mass — paper convention (default 0.10 = worst 10%).
        B, N_resample, P, pf_idx, L: bootstrap / stacking / frontier params.

    Returns:
        Weight vector shape (I,), sum=1, 0 <= w_i <= w_max.
        Falls back to equal-weight 1/I on any pipeline failure.
    """
    if not _HAS_FT:
        raise ImportError("fortitudo.tech is required: pip install fortitudo_tech")

    S, I = R.shape
    if I == 0:
        return np.zeros(0)
    if I == 1:
        return np.array([1.0])

    means = R.mean(axis=0)
    try:
        cov = np.cov(R.T)
    except Exception:
        return np.ones(I) / I

    # Constraints: 0 <= w_i <= w_max  (sum(w)=1 is built into MeanCVaR via v)
    G = np.vstack((np.eye(I), -np.eye(I)))
    h = np.hstack((w_max * np.ones(I), np.zeros(I)))

    # Bootstrap candidate means (parameter uncertainty)
    if seed is not None:
        np.random.seed(seed)
    try:
        return_sim = np.random.multivariate_normal(means, cov, (N_resample, B))
    except np.linalg.LinAlgError:
        return np.ones(I) / I

    p_prior = np.ones((S, 1)) / S
    frontier_mean = np.full((I, P, B), np.nan)

    for b in range(B):
        means_b = return_sim[:, b, :].mean(axis=0)
        try:
            # Entropy pooling: re-weight history so weighted mean matches means_b.
            # A is (I, S) — one row per asset, each column a scenario.
            q = ft.entropy_pooling(p_prior, A=R.T, b=means_b[:, np.newaxis])
            cvar_b = ft.MeanCVaR(R, G=G, h=h, p=q, alpha=alpha)
            frontier_mean[:, :, b] = cvar_b.efficient_frontier(P)
        except Exception:
            pass  # infeasible bootstrap; skipped

    good = ~np.isnan(frontier_mean[0, pf_idx, :])
    if good.sum() < L * 2:
        return np.ones(I) / I  # fallback if too few feasible bootstraps

    return ft.exposure_stacking(L, frontier_mean[:, pf_idx, good])


# ─── Selection helpers (mirror portfolio.py) ─────────────────────────────────

def _select_top_bottom(group: pd.DataFrame, score_col: str, top_n: int):
    if len(group) < 2 * top_n:
        return None, None
    g = group.sort_values(score_col, ascending=False)
    return g.head(top_n), g.tail(top_n)


def _select_xgb_class(group: pd.DataFrame, top_n: int, n_classes: int = 10):
    if len(group) < 2 * top_n:
        return None, None
    g = group.copy()
    g["_xgb_score"] = g[f"prob_{n_classes}"] - g["prob_1"]
    g = g.sort_values("_xgb_score", ascending=False)
    return g.head(top_n), g.tail(top_n)


# ─── Daily-return panel slicer ───────────────────────────────────────────────

def _build_daily_panel(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot daily df to (date × assetid) of returns, computed as
    close.pct_change() per assetid. Used to slice lookback windows.
    """
    df = daily[["assetid", "date", "close"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["assetid", "date"])
    df["ret"] = df.groupby("assetid")["close"].pct_change()
    panel = df.pivot_table(index="date", columns="assetid", values="ret")
    return panel


def _slice_lookback(panel: pd.DataFrame, end_date: pd.Timestamp,
                    assetids: list[int], lookback_days: int,
                    min_history: int) -> tuple[np.ndarray, list[int]]:
    """
    Return (R, kept_assetids) where R is a (≤lookback_days, len(kept)) array
    of daily returns ending at or before `end_date`, including only stocks
    with at least `min_history` non-NaN observations in that window.
    """
    end_idx = panel.index.searchsorted(end_date, side="right")
    start_idx = max(end_idx - lookback_days, 0)
    window = panel.iloc[start_idx:end_idx]

    cols = [a for a in assetids if a in window.columns]
    if not cols:
        return np.zeros((0, 0)), []
    sub = window[cols].dropna(how="all")
    counts = sub.count()
    keep_assetids = [a for a in cols if counts.get(a, 0) >= min_history]
    if not keep_assetids:
        return np.zeros((0, 0)), []
    R = sub[keep_assetids].fillna(0.0).values
    return R, keep_assetids


# ─── Cost helper (same convention as portfolio.py) ───────────────────────────

def _period_cost_weighted(curr_ids, prev_ids, next_ids, weights,
                           tc_bps, carry_annual, days):
    """
    Cost for a leg in one period, using the actual optimizer weights (not 1/N).
    Commission per stock = (entry+exit_events) * tc_bps / 10000 * w_i.
    Financing per stock  = carry_annual * days/365 * w_i.
    """
    prev_set, next_set = set(prev_ids), set(next_ids)
    commission = 0.0
    for sym, w in zip(curr_ids, weights):
        if w <= 0:
            continue
        events = int(sym not in prev_set) + int(sym not in next_set)
        commission += events * w * tc_bps / 10000
    financing = carry_annual * days / 365  # per leg, on full notional (sum of weights = 1)
    return commission, financing


# ─── Main rolling driver ─────────────────────────────────────────────────────

def run_two_leg_optimizer(predictions: pd.DataFrame,
                            features: pd.DataFrame,
                            daily: pd.DataFrame,
                            score_col: str = "ret_score",
                            top_n: int = 15,
                            lookback_days: int = LOOKBACK_DAYS,
                            min_history_days: int = MIN_HISTORY_DAYS,
                            cvar_alpha: float = CVAR_ALPHA,
                            w_max: float = W_MAX,
                            B: int = B_RESAMPLES,
                            N_resample: int = N_RESAMPLE,
                            P: int = P_FRONTIER,
                            pf_idx: int = PF_INDEX,
                            L: int = L_STACK,
                            tc_bps: float = 20.0,
                            carry_long_annual: float = 0.05,
                            carry_short_annual: float = 0.02,
                            days_per_month: int = 30,
                            long_only: bool = False,
                            verbose: bool = True,
                            seed: int = 3) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """
    Run the two-leg Mean-CVaR optimizer over all months in `predictions`.

    Args:
        predictions: monthly model output with `assetid`, `date_mt`,
                     `score_col`, `fwd_return_mt`, prob_1..prob_10.
        features:    monthly feature panel (used as fallback for fwd_return_mt
                     if predictions doesn't have it).
        daily:       daily OHLCV from data_load.py (uses `close` for returns).

    Returns (portfolio_df, weights_df, selections_log):
        portfolio_df:  one row per month with same schema as portfolio.py
                       outputs (date_mt, strategy, ls_ret_gross, commission,
                       financing, tc, ls_ret, n_long, n_short).
        weights_df:    date_mt index × assetid columns of signed weights
                       (positive long, negative short; NaN = not in book).
        selections_log: list of per-month dicts with picked names + weights;
                       used by the trade log builder.
    """
    if not _HAS_FT:
        raise ImportError("fortitudo.tech is required: pip install fortitudo_tech")

    # Ensure fwd_return_mt is present in predictions
    pred = predictions.copy()
    if "fwd_return_mt" not in pred.columns or pred["fwd_return_mt"].isna().all():
        fwd = features[["assetid", "date_mt", "fwd_return_mt"]].dropna()
        pred = pred.drop(columns=["fwd_return_mt"], errors="ignore")
        pred = pred.merge(fwd, on=["assetid", "date_mt"], how="left")

    # Build daily return panel once
    if verbose:
        print("  Building daily return panel...")
    panel = _build_daily_panel(daily)
    if verbose:
        print(f"  Daily panel: {panel.shape[0]:,} days × {panel.shape[1]:,} assetids")

    # Per-month name selection
    is_xgb = (score_col == "xgb")
    rows = []
    weight_records = {}
    selections_log = []

    np.random.seed(seed)
    t0 = time.time()

    # Group by CALENDAR MONTH (not raw date_mt) — see portfolio._build_selections
    # for the same fix and rationale.
    pred = pred.copy()
    pred["_ym"] = pred["date_mt"].dt.to_period("M")
    months = sorted(pred["_ym"].unique())

    # Pre-collect selections so we can look at prev/next months for cost attribution
    selections = []
    for m in months:
        grp = pred[pred["_ym"] == m]
        if is_xgb:
            longs, shorts = _select_xgb_class(grp, top_n)
        else:
            longs, shorts = _select_top_bottom(grp, score_col, top_n)
        if longs is None:
            continue
        rebal_date = grp["date_mt"].max()  # latest stock-level rebalance date in this month
        selections.append({
            "date_mt":       pd.Timestamp(rebal_date),
            "long_ids":      longs["assetid"].tolist(),
            "short_ids":     shorts["assetid"].tolist(),
            "long_fwd":      longs.set_index("assetid")["fwd_return_mt"].to_dict(),
            "short_fwd":     shorts.set_index("assetid")["fwd_return_mt"].to_dict(),
            "long_symbols":  longs.set_index("assetid")["symbol"].to_dict(),
            "short_symbols": shorts.set_index("assetid")["symbol"].to_dict(),
        })

    n_total = len(selections)
    if verbose:
        print(f"  Months to process: {n_total:,}")

    for i, sel in enumerate(selections):
        date_t = sel["date_mt"]

        # Lookback windows for each leg
        R_l, kept_long = _slice_lookback(panel, date_t,
                                          sel["long_ids"],
                                          lookback_days, min_history_days)
        if long_only:
            R_s, kept_short = np.zeros((0, 0)), []
        else:
            R_s, kept_short = _slice_lookback(panel, date_t,
                                               sel["short_ids"],
                                               lookback_days, min_history_days)

        # Sanity gate: long leg always required; short leg only if not long_only
        if len(kept_long) < 3 or (not long_only and len(kept_short) < 3):
            if verbose:
                print(f"    {date_t.date()}: too few names with history "
                      f"(L={len(kept_long)}, S={len(kept_short)}) — skipped")
            continue

        # Optimize long leg always; short leg only if not long_only.
        w_long  = optimize_leg(R_l,  w_max=w_max, alpha=cvar_alpha,
                                B=B, N_resample=N_resample, P=P,
                                pf_idx=pf_idx, L=L)
        if long_only:
            w_short = np.zeros(0)
        else:
            w_short = optimize_leg(-R_s, w_max=w_max, alpha=cvar_alpha,
                                    B=B, N_resample=N_resample, P=P,
                                    pf_idx=pf_idx, L=L)

        # Realised P&L (apply weights to next-month forward returns)
        fwd_l = np.array([sel["long_fwd"].get(a, 0.0) for a in kept_long])
        long_ret  = float((w_long * fwd_l).sum())
        if long_only:
            short_ret = 0.0
        else:
            fwd_s = np.array([sel["short_fwd"].get(a, 0.0) for a in kept_short])
            short_ret = float((w_short * fwd_s).sum())
        ls_gross = long_ret - short_ret  # long_only → ls_gross == long_ret

        # Cost attribution
        prev = selections[i - 1] if i > 0 else None
        nxt  = selections[i + 1] if i + 1 < n_total else None
        prev_l = prev["long_ids"]  if prev else []
        prev_s = prev["short_ids"] if prev else []
        nxt_l  = nxt["long_ids"]   if nxt  else []
        nxt_s  = nxt["short_ids"]  if nxt  else []

        comm_l, fin_l = _period_cost_weighted(kept_long,  prev_l, nxt_l, w_long,
                                               tc_bps, carry_long_annual, days_per_month)
        if long_only:
            comm_s, fin_s = 0.0, 0.0
        else:
            comm_s, fin_s = _period_cost_weighted(kept_short, prev_s, nxt_s, w_short,
                                                   tc_bps, carry_short_annual, days_per_month)
        commission = comm_l + comm_s
        financing  = fin_l - fin_s   # long pays, short earns (or 0 in long_only)
        tc = commission + financing

        rows.append({
            "date_mt":      date_t,
            "strategy":     f"OPT_{score_col.replace('_score','').upper()}",
            "n_long":       len(kept_long),
            "n_short":      0 if long_only else len(kept_short),
            "long_ret":     long_ret,
            "short_ret":    short_ret,
            "ls_ret_gross": ls_gross,
            "commission":   commission,
            "financing":    financing,
            "tc":           tc,
            "ls_ret":       ls_gross - tc,
        })

        # Signed weights for the panel
        wrec = {}
        for a, w in zip(kept_long,  w_long):  wrec[a] =  float(w)
        if not long_only:
            for a, w in zip(kept_short, w_short): wrec[a] = -float(w)
        weight_records[date_t] = wrec

        selections_log.append({
            "date_mt":       date_t,
            "long_ids":      kept_long,
            "short_ids":     [] if long_only else kept_short,
            "w_long":        w_long.tolist(),
            "w_short":       [] if long_only else w_short.tolist(),
            "long_symbols":  [sel["long_symbols" ].get(a, "?") for a in kept_long],
            "short_symbols": [] if long_only else [sel["short_symbols"].get(a, "?") for a in kept_short],
        })

        if verbose and (i % 12 == 0 or i == n_total - 1):
            print(f"    {date_t.date()}  {i+1}/{n_total}  ({time.time()-t0:.0f}s)")

    portfolio = pd.DataFrame(rows)
    weights   = pd.DataFrame(weight_records).T.sort_index()
    weights.index.name = "date_mt"
    return portfolio, weights, selections_log


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("Two-leg Mean-CVaR optimizer (RET signal, 2y daily lookback)")
    print("=" * 70)

    if not (DAILY_PATH.exists() and PREDICTIONS_PATH.exists() and FEATURES_PATH.exists()):
        raise FileNotFoundError(
            "Need ca_equities_daily.parquet + ca_predictions_monthly.parquet + "
            "ca_features_monthly.parquet. Run earlier scripts first."
        )

    t0 = time.time()
    daily = pd.read_parquet(DAILY_PATH)
    predictions = pd.read_parquet(PREDICTIONS_PATH)
    features = pd.read_parquet(FEATURES_PATH)
    print(f"  Loaded inputs in {time.time()-t0:.0f}s")

    portfolio, weights, sel_log = run_two_leg_optimizer(
        predictions, features, daily,
        score_col="ret_score",
        top_n=15,
        lookback_days=504,
        min_history_days=252,
        cvar_alpha=0.10,
        w_max=0.30,
        B=200, N_resample=100, P=9, pf_idx=4, L=20,
        tc_bps=20.0,
        carry_long_annual=0.05,
        carry_short_annual=0.02,
    )

    portfolio.to_parquet(OPT_PORT_PATH, index=False, compression="snappy")
    weights.to_parquet(OPT_WEIGHTS_PATH, compression="snappy")
    print(f"\nWrote {OPT_PORT_PATH}")
    print(f"Wrote {OPT_WEIGHTS_PATH}")

    if not portfolio.empty:
        from portfolio import compute_performance, print_performance_table
        m = compute_performance(portfolio, "OPT_RET")
        print(f"\nOPT_RET: ann.ret {m['mean_annual']:>7.1%}  "
              f"sharpe {m['sharpe']:>5.2f}  "
              f"comm {m['avg_comm_bps']:>5.1f}bp/mo  "
              f"fin {m['avg_fin_bps']:>5.1f}bp/mo  "
              f"months {m['n_months']}")

    print(f"\nTotal runtime: {time.time()-t0:.0f}s")
