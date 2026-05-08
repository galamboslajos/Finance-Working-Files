"""
Deep Momentum — Step 5: XGBoost walk-forward training & prediction.

Reads:
  cache/ca_features_monthly.parquet   (output of features.py)

Writes:
  cache/ca_predictions_monthly.parquet

Paper-faithful walk-forward (Section 3.3.3):
  - Require at least MIN_TRAIN_YEARS (10) of history before first prediction
  - Retrain every RETRAIN_FREQUENCY (12) months, accumulating the sample
  - Train N_ENSEMBLE (100) times per training date with different random
    80/20 train/val splits; average predicted probabilities
  - XGBoost with default hyperparameters except early stopping

Reclassifications (Section 3.3.2 + extension):
  XGB    = mode class of predicted probability distribution
  RET    = E[r] = Σ p_k · μ_k         where μ_k is past-10y mean of class k
  SRP    = E[r] / σ_r  using law of total variance
  CVR    = E[r] / |CVaR_α|            (extension; α = 10%)

All operates on the new schema: assetid as the row key, date_mt / LABEL_mt /
fwd_return_mt with `_mt` suffixes. Uses the canonical 16-feature list from
features.get_feature_columns().
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

from features import get_feature_columns


PROJECT_DIR = Path(__file__).resolve().parent
CACHE_DIR   = PROJECT_DIR / "cache"

FEATURES_PATH    = CACHE_DIR / "ca_features_monthly.parquet"
PREDICTIONS_PATH = CACHE_DIR / "ca_predictions_monthly.parquet"

# Paper hyperparameters (Section 3.3)
N_CLASSES                  = 10
MIN_TRAIN_YEARS            = 10
N_ENSEMBLE                 = 100
TRAIN_VAL_RATIO            = 0.8
RETRAIN_FREQUENCY          = 12   # months
CLASS_RETURN_LOOKBACK_YEARS = 10
CVAR_ALPHA                 = 0.10  # extension (CVR)

XGB_PARAMS = dict(
    objective="multi:softprob",
    num_class=N_CLASSES,
    eval_metric="mlogloss",
    verbosity=0,
    n_estimators=10000,
    early_stopping_rounds=50,
)


# ─── Training primitives ─────────────────────────────────────────────────────

def train_single_xgb(X_train, y_train, X_val, y_val, random_state=0):
    """One XGBoost fit with early stopping on the validation set."""
    model = xgb.XGBClassifier(random_state=random_state, **XGB_PARAMS)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def train_ensemble(X, y, n_ensemble=N_ENSEMBLE):
    """
    100 fits with different random 80/20 train/val splits (paper-faithful).
    Returns list of fitted models.
    """
    models = []
    for i in range(n_ensemble):
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=1 - TRAIN_VAL_RATIO, random_state=i
        )
        models.append(train_single_xgb(X_tr, y_tr, X_val, y_val, random_state=0))
    return models


def predict_ensemble(models, X):
    """
    Average probabilities across ensemble. Pads to N_CLASSES if a sub-fit
    saw fewer classes (small training samples can hit this).
    """
    probs_stack = []
    for m in models:
        p = m.predict_proba(X)
        if p.shape[1] < N_CLASSES:
            padded = np.zeros((p.shape[0], N_CLASSES))
            for j, c in enumerate(m.classes_):
                padded[:, int(c)] = p[:, j]
            p = padded
        probs_stack.append(p)
    return np.mean(probs_stack, axis=0)


# ─── Reclassification ────────────────────────────────────────────────────────

def naive_classify(probs):
    """XGB strategy — mode class. Returns 1..N_CLASSES."""
    return np.argmax(probs, axis=1) + 1


def compute_class_stats(df, date, lookback_years=CLASS_RETURN_LOOKBACK_YEARS):
    """
    Past-10y per-class mean and std of fwd_return, by LABEL_mt.
    Strictly past data: only rows with date_mt < `date`.
    """
    cutoff = date - pd.DateOffset(years=lookback_years)
    hist = df[(df["date_mt"] >= cutoff) & (df["date_mt"] < date)]

    mu = {}
    sigma = {}
    for k in range(1, N_CLASSES + 1):
        cls = hist.loc[hist["LABEL_mt"] == k, "fwd_return_mt"].dropna()
        mu[k]    = cls.mean() if len(cls) > 0 else 0.0
        sigma[k] = cls.std()  if len(cls) > 1 else 0.01
    return mu, sigma


def reclassify_ret(probs, mu_dict):
    mu = np.array([mu_dict.get(k, 0.0) for k in range(1, N_CLASSES + 1)])
    return probs @ mu


def reclassify_srp(probs, mu_dict, sigma_dict):
    mu    = np.array([mu_dict.get(k, 0.0)    for k in range(1, N_CLASSES + 1)])
    sigma = np.array([sigma_dict.get(k, 0.0) for k in range(1, N_CLASSES + 1)])
    e_r = probs @ mu
    var = probs @ (sigma**2 + mu**2) - e_r**2
    var = np.maximum(var, 1e-10)
    return e_r / np.sqrt(var)


def reclassify_cvr(probs, mu_dict, alpha=CVAR_ALPHA):
    """
    Return-over-CVaR score (extension):
        score_i = E[r_i] / |CVaR_α(r_i)|
    where CVaR is computed from the predicted discrete distribution.
    """
    mu = np.array([mu_dict.get(k, 0.0) for k in range(1, N_CLASSES + 1)])
    e_r = probs @ mu
    cumprob = np.cumsum(probs, axis=1)
    cum_prev = np.concatenate(
        [np.zeros((probs.shape[0], 1)), cumprob[:, :-1]], axis=1
    )
    included = np.minimum(probs, np.maximum(alpha - cum_prev, 0.0))
    cvar = (included * mu[None, :]).sum(axis=1) / alpha
    return e_r / (np.abs(cvar) + 1e-6)


# ─── Walk-forward orchestration ──────────────────────────────────────────────

def get_training_schedule(df):
    """
    First train = first_date + 10 years. Retrain every 12 months thereafter.
    Returns list of dicts: {train_date, train_idx, predict_months}.
    """
    months = sorted(df["date_mt"].dt.to_period("M").unique())
    if not months:
        return [], months

    first_date  = pd.Timestamp(months[0].to_timestamp())
    first_train = first_date + pd.DateOffset(years=MIN_TRAIN_YEARS)

    schedule = []
    cur_idx = None
    for i, ym in enumerate(months):
        d = ym.to_timestamp()
        if d < first_train:
            continue
        if cur_idx is None or i - cur_idx >= RETRAIN_FREQUENCY:
            cur_idx = i
            schedule.append({"train_date": d, "train_idx": i})

    for j, sched in enumerate(schedule):
        next_idx = schedule[j + 1]["train_idx"] if j + 1 < len(schedule) else len(months)
        sched["predict_months"] = months[sched["train_idx"]:next_idx]
    return schedule, months


def run_walk_forward(df: pd.DataFrame, feature_cols: list[str] | None = None,
                     n_ensemble: int = N_ENSEMBLE, verbose: bool = True) -> pd.DataFrame:
    """
    Full walk-forward over the panel. One model trained per year, predicts the
    next 12 months. Returns predictions DataFrame with all four scores
    (xgb_class, ret_score, srp_score, cvr_score) plus per-class probs.
    """
    if feature_cols is None:
        feature_cols = get_feature_columns()

    schedule, all_months = get_training_schedule(df)
    if not schedule:
        print("    No training months found (insufficient history).")
        return pd.DataFrame()

    if verbose:
        print(f"    Training schedule: {len(schedule)} retrainings")
        print(f"    First train: {schedule[0]['train_date'].date()}")
        print(f"    Last train:  {schedule[-1]['train_date'].date()}")

    rows = []
    t0 = time.time()

    for sched in schedule:
        train_date = sched["train_date"]
        predict_months = sched["predict_months"]

        if verbose:
            print(f"    Training at {train_date.date()}, predicting {len(predict_months)} months...")

        # Training rows: everything up to and including the train_date
        mask_train = df["date_mt"] <= train_date
        sub = df.loc[mask_train, feature_cols + ["LABEL_mt"]].dropna()
        if len(sub) < 100:
            if verbose:
                print(f"      skip: only {len(sub)} complete training rows")
            continue

        X_train = sub[feature_cols].values
        y_train = sub["LABEL_mt"].astype(int).values - 1  # 0-indexed for XGBoost

        models = train_ensemble(X_train, y_train, n_ensemble=n_ensemble)

        # Past-10y class stats — used by RET / SRP / CVR
        mu_dict, sigma_dict = compute_class_stats(df, train_date)

        # Predict month-by-month over the prediction window
        for ym in predict_months:
            month_d = ym.to_timestamp()
            month_mask = df["date_mt"].dt.to_period("M") == ym
            month = df.loc[month_mask].copy()
            if month.empty:
                continue

            valid = month[feature_cols].dropna()
            if valid.empty:
                continue
            pred_idx = valid.index

            X_pred = month.loc[pred_idx, feature_cols].values
            probs = predict_ensemble(models, X_pred)

            xgb_class = naive_classify(probs)
            ret_score = reclassify_ret(probs, mu_dict)
            srp_score = reclassify_srp(probs, mu_dict, sigma_dict)
            cvr_score = reclassify_cvr(probs, mu_dict)

            for i, idx in enumerate(pred_idx):
                rec = {
                    "assetid":      int(month.loc[idx, "assetid"]),
                    "symbol":       month.loc[idx, "symbol"],
                    "date_mt":      month.loc[idx, "date_mt"],
                    "fwd_return_mt": month.loc[idx, "fwd_return_mt"]
                                     if "fwd_return_mt" in month.columns else np.nan,
                    "xgb_class":    int(xgb_class[i]),
                    "ret_score":    float(ret_score[i]),
                    "srp_score":    float(srp_score[i]),
                    "cvr_score":    float(cvr_score[i]),
                }
                for k in range(N_CLASSES):
                    rec[f"prob_{k+1}"] = float(probs[i, k])
                rows.append(rec)

        if verbose:
            print(f"      done — elapsed {time.time()-t0:.0f}s")

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows).sort_values(["assetid", "date_mt"]).reset_index(drop=True)
    out["date_mt"] = pd.to_datetime(out["date_mt"])
    if verbose:
        print(f"    Total predictions: {len(out):,}")
        print(f"    Date range:        {out['date_mt'].min().date()} → "
              f"{out['date_mt'].max().date()}")
    return out


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("Walk-forward training")
    print("=" * 70)

    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Missing {FEATURES_PATH}. Run features.py first.")

    t0 = time.time()
    feat = pd.read_parquet(FEATURES_PATH)
    print(f"  Loaded {FEATURES_PATH.name} in {time.time()-t0:.0f}s "
          f"({len(feat):,} rows)")

    preds = run_walk_forward(feat, n_ensemble=N_ENSEMBLE, verbose=True)

    print(f"\nWriting {PREDICTIONS_PATH}...")
    preds.to_parquet(PREDICTIONS_PATH, index=False, compression="snappy")
    size_mb = PREDICTIONS_PATH.stat().st_size / (1024 * 1024)
    print(f"  wrote {size_mb:.0f} MB")

    print(f"\nTotal runtime: {time.time()-t0:.0f}s")
