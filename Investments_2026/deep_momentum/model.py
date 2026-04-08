"""
Deep Momentum — Step 4: Model
XGBoost multiclass classifier + 100x ensemble + RET reclassification.

Paper reference: Sections 3.3.1, 3.3.2, 3.3.3

Training procedure (Section 3.3.3):
- Require at least 10 years of data before first prediction
- Retrain every year, accumulating the sample
- 80/20 random train/val split (NOT chronological — paper is explicit about this)
- Train 100 times per training month, average predicted probabilities
- XGBoost with default hyperparameters, only early stopping

Prediction (Section 3.3.1):
- XGBoost multiclass classifier: objective = multi:softprob, 10 classes
- Output: P(class=k) for k=1..10 for each stock

Reclassification (Section 3.3.2):
- XGB (naive): classify into mode class (highest probability)
- RET: compute expected return = sum(P(class=k) * mu_k) where mu_k is
  the average historical return of class k over the past 10 years
- SRP: compute Sharpe ratio from predicted distribution
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

from config import (
    N_CLASSES, N_ENSEMBLE, TRAIN_VAL_RATIO,
    MIN_TRAIN_YEARS, RETRAIN_FREQUENCY,
    CLASS_RETURN_LOOKBACK_YEARS,
)


def train_single_xgb(X_train, y_train, X_val, y_val, random_state=0):
    """
    Train one XGBoost classifier.

    Paper: "XGBoost is trained using its default hyperparameters,
    except for early stopping"
    """
    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=N_CLASSES,
        eval_metric="mlogloss",
        verbosity=0,
        n_estimators=10000,
        early_stopping_rounds=50,
        random_state=random_state,
        use_label_encoder=False,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    return model


def train_ensemble(X, y, n_ensemble=N_ENSEMBLE):
    """
    Paper Section 3.3.3:
    "we train the algorithm 100 times in each training month and use the
    average probabilities as the final predicted probabilities"

    "The training sample is randomly split into a training set and a
    validation set in a ratio of 8:2"

    Each of the 100 trainings uses a different random split.
    """
    models = []

    for i in range(n_ensemble):
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=1 - TRAIN_VAL_RATIO,
            random_state=i,  # different split each time
        )

        # Only vary the data split (random_state=i above), keep XGBoost
        # tree internals fixed. Paper only mentions randomizing the split.
        model = train_single_xgb(X_train, y_train, X_val, y_val, random_state=0)
        models.append(model)

    return models


def predict_ensemble(models, X):
    """
    Average predicted probabilities across all ensemble members.
    Returns: array of shape (n_samples, N_CLASSES)

    Note: if training data didn't contain all N_CLASSES,
    XGBoost only outputs columns for seen classes. We pad to N_CLASSES.
    """
    all_probs = []
    for model in models:
        probs = model.predict_proba(X)

        # Pad to N_CLASSES if needed (small samples may not have all classes)
        if probs.shape[1] < N_CLASSES:
            padded = np.zeros((probs.shape[0], N_CLASSES))
            seen_classes = model.classes_
            for i, c in enumerate(seen_classes):
                padded[:, int(c)] = probs[:, i]
            probs = padded

        all_probs.append(probs)

    avg_probs = np.mean(all_probs, axis=0)
    return avg_probs


def naive_classify(probs):
    """
    XGB strategy (naive): assign each stock to its mode class.
    Returns class labels (1-10).
    """
    return np.argmax(probs, axis=1) + 1  # +1 because classes are 1-indexed


def compute_class_returns(df, date, lookback_years=CLASS_RETURN_LOOKBACK_YEARS):
    """
    Paper Section 3.3.2 (RET method):
    "The return of class k is defined as the average return of the stocks
    belonging to the class, and the estimate of mu_k is their time-series
    average over the past ten years."

    For each class k (1-10), compute the average monthly return of stocks
    that were assigned to class k, averaged over the past 10 years.

    Args:
        df: DataFrame with 'date', 'LABEL', 'return' columns
        date: current date (compute lookback from here)
        lookback_years: number of years to look back (default 10)

    Returns:
        dict {class_k: mean_return} for k=1..N_CLASSES
    """
    cutoff = date - pd.DateOffset(years=lookback_years)
    hist = df[(df["date"] >= cutoff) & (df["date"] < date)]

    class_returns = {}
    for k in range(1, N_CLASSES + 1):
        # LABEL is assigned based on fwd_return (next month's return).
        # mu_k = average forward return of stocks in class k.
        col = "fwd_return" if "fwd_return" in hist.columns else "return"
        class_data = hist[hist["LABEL"] == k][col]
        if len(class_data) > 0:
            class_returns[k] = class_data.mean()
        else:
            class_returns[k] = 0.0

    return class_returns


def reclassify_ret(probs, class_returns):
    """
    Paper Section 3.3.2 (RET — Reclassification on expected return):
    mu_i = sum_{k=1}^{10} P(class=k) * mu_k

    Args:
        probs: array (n_samples, N_CLASSES) — predicted probabilities
        class_returns: dict {k: mu_k} for k=1..N_CLASSES

    Returns:
        array of expected returns, shape (n_samples,)
    """
    mu = np.array([class_returns.get(k, 0.0) for k in range(1, N_CLASSES + 1)])
    expected_returns = probs @ mu
    return expected_returns


def reclassify_srp(probs, class_returns, class_stds):
    """
    Paper Section 3.3.2 (SRP — Reclassification on Sharpe ratio):
    sigma_i^2 = sum P(k) * (sigma_k^2 + mu_k^2) - mu_i^2
    SRP_i = mu_i / sigma_i

    Args:
        probs: array (n_samples, N_CLASSES)
        class_returns: dict {k: mu_k}
        class_stds: dict {k: sigma_k}

    Returns:
        array of Sharpe ratios, shape (n_samples,)
    """
    mu = np.array([class_returns.get(k, 0.0) for k in range(1, N_CLASSES + 1)])
    sigma = np.array([class_stds.get(k, 0.0) for k in range(1, N_CLASSES + 1)])

    expected_returns = probs @ mu
    variance = probs @ (sigma**2 + mu**2) - expected_returns**2
    variance = np.maximum(variance, 1e-10)  # avoid division by zero
    sharpe = expected_returns / np.sqrt(variance)

    return sharpe


def compute_class_stds(df, date, lookback_years=CLASS_RETURN_LOOKBACK_YEARS):
    """
    Compute standard deviation of returns per class over lookback window.
    Used for SRP reclassification.
    """
    cutoff = date - pd.DateOffset(years=lookback_years)
    hist = df[(df["date"] >= cutoff) & (df["date"] < date)]

    class_stds = {}
    for k in range(1, N_CLASSES + 1):
        col = "fwd_return" if "fwd_return" in hist.columns else "return"
        class_data = hist[hist["LABEL"] == k][col]
        if len(class_data) > 1:
            class_stds[k] = class_data.std()
        else:
            class_stds[k] = 0.01  # small default

    return class_stds


def get_training_months(df):
    """
    Paper Section 3.3.3:
    "We require at least ten years of data to train XGBoost and retrain
    the algorithm every year"

    Returns list of (train_date, predict_months) tuples.
    train_date: the month when the model is trained
    predict_months: list of months to predict using this model (up to 12)
    """
    dates = sorted(df["date"].unique())
    first_date = pd.Timestamp(dates[0])

    # First training date: 10 years after first data
    first_train = first_date + pd.DateOffset(years=MIN_TRAIN_YEARS)

    # Get all unique year-months
    all_months = sorted(df["date"].dt.to_period("M").unique())

    # Find training months (January of each year, starting from first eligible)
    training_schedule = []
    current_train_idx = None

    for i, ym in enumerate(all_months):
        month_date = ym.to_timestamp()
        if month_date < first_train:
            continue

        if current_train_idx is None:
            # First training
            current_train_idx = i
            training_schedule.append({
                "train_date": month_date,
                "train_idx": i,
            })
        elif i - current_train_idx >= RETRAIN_FREQUENCY:
            # Retrain every 12 months
            current_train_idx = i
            training_schedule.append({
                "train_date": month_date,
                "train_idx": i,
            })

    # Assign prediction months to each training
    for j, sched in enumerate(training_schedule):
        if j + 1 < len(training_schedule):
            next_train_idx = training_schedule[j + 1]["train_idx"]
        else:
            next_train_idx = len(all_months)

        sched["predict_months"] = all_months[sched["train_idx"]:next_train_idx]

    return training_schedule, all_months


def run_walk_forward(df, feature_cols, n_ensemble=N_ENSEMBLE, verbose=True):
    """
    Full walk-forward procedure for one country.

    Paper Section 3.3.3:
    - Train with all data up to training date (accumulating)
    - Retrain every year
    - Predict monthly using latest model
    - 100x ensemble, average probabilities

    Returns DataFrame with predictions:
    - symbol, date, probs (P(k) for each class), xgb_class, ret_score, srp_score
    """
    training_schedule, all_months = get_training_months(df)

    if not training_schedule:
        print("    No training months found (not enough history)")
        return pd.DataFrame()

    if verbose:
        print(f"    Training schedule: {len(training_schedule)} retrainings")
        print(f"    First train: {training_schedule[0]['train_date'].date()}")
        print(f"    Last train: {training_schedule[-1]['train_date'].date()}")

    all_predictions = []

    for sched in training_schedule:
        train_date = sched["train_date"]
        predict_months = sched["predict_months"]

        if verbose:
            print(f"    Training at {train_date.date()}, "
                  f"predicting {len(predict_months)} months...")

        # Training data: all data up to (and including) training date
        train_mask = df["date"] <= train_date
        train_df = df[train_mask].copy()

        # Must have complete features and label
        train_complete = train_df[feature_cols + ["LABEL"]].dropna()
        train_idx = train_complete.index

        if len(train_idx) < 100:
            if verbose:
                print(f"      Skipping: only {len(train_idx)} training samples")
            continue

        X_train = train_df.loc[train_idx, feature_cols].values
        y_train = train_df.loc[train_idx, "LABEL"].values.astype(int) - 1  # 0-indexed for XGBoost

        # Train ensemble
        models = train_ensemble(X_train, y_train, n_ensemble=n_ensemble)

        # Compute class returns and stds for RET/SRP reclassification
        class_returns = compute_class_returns(df, train_date)
        class_stds = compute_class_stds(df, train_date)

        # Predict for each month in the prediction window
        for ym in predict_months:
            month_date = ym.to_timestamp()
            month_mask = df["date"].dt.to_period("M") == ym
            month_df = df[month_mask].copy()

            if month_df.empty:
                continue

            # Complete features for prediction
            pred_complete = month_df[feature_cols].dropna()
            if pred_complete.empty:
                continue

            pred_idx = pred_complete.index
            X_pred = month_df.loc[pred_idx, feature_cols].values

            # Predict probabilities (ensemble average)
            probs = predict_ensemble(models, X_pred)

            # Naive classification
            xgb_class = naive_classify(probs)

            # RET reclassification
            ret_score = reclassify_ret(probs, class_returns)

            # SRP reclassification
            srp_score = reclassify_srp(probs, class_returns, class_stds)

            # Store results
            for i, idx in enumerate(pred_idx):
                row = {
                    "symbol": month_df.loc[idx, "symbol"],
                    "date": month_df.loc[idx, "date"],
                    "fwd_return": month_df.loc[idx, "fwd_return"]
                        if "fwd_return" in month_df.columns else np.nan,
                    "xgb_class": int(xgb_class[i]),
                    "ret_score": ret_score[i],
                    "srp_score": srp_score[i],
                }
                # Store individual class probabilities
                for k in range(N_CLASSES):
                    row[f"prob_{k+1}"] = probs[i, k]

                all_predictions.append(row)

    if not all_predictions:
        return pd.DataFrame()

    result = pd.DataFrame(all_predictions)
    result["date"] = pd.to_datetime(result["date"])

    if verbose:
        print(f"    Total predictions: {len(result)}")
        print(f"    Date range: {result['date'].min().date()} to {result['date'].max().date()}")

    return result


# ═══════════════════════════════════════════════════════
if __name__ == "__main__":
    from pathlib import Path
    from config import CACHE_DIR, COUNTRIES
    from features import build_features

    cache_dir = Path(CACHE_DIR)

    # Test on Canada with small ensemble for speed
    suffix = "TO"
    _, country_name, _, _ = COUNTRIES[suffix]

    filtered_path = cache_dir / f"filtered_{suffix}.parquet"
    if not filtered_path.exists():
        print(f"No filtered data for {suffix}. Run data_filter.py first.")
    else:
        df = pd.read_parquet(filtered_path)
        df, feature_cols = build_features(df, country_name)

        print(f"\n  Running walk-forward for {country_name} (test: 5 ensemble members)...")
        predictions = run_walk_forward(
            df, feature_cols,
            n_ensemble=5,  # small for testing
            verbose=True,
        )

        if not predictions.empty:
            out_path = cache_dir / f"predictions_{suffix}.parquet"
            predictions.to_parquet(out_path, index=False)
            print(f"\n  Saved: {out_path}")

            print(f"\n  --- Sample predictions ---")
            print(predictions[["symbol", "date", "xgb_class", "ret_score",
                             "srp_score", "fwd_return"]].tail(10).to_string(index=False))
