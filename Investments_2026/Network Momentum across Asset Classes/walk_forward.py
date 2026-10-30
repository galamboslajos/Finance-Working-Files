#!/usr/bin/env python3
"""Annual expanding OLS chronology, causal splits and pooled linear models."""
from __future__ import annotations

import argparse
from decimal import Decimal
import json
from pathlib import Path

import exchange_calendars as xcals
import numpy as np
import pandas as pd

from build_momentum_features import LOOKBACKS, write_json
from prepare_sp500_daily import file_hash, require


ROOT = Path(__file__).resolve().parent
PROTOCOL = {
    "id": "sp500-annual-expanding-ols-three-initial-years-v2",
    "status": "user_selected_2026-10-30",
    "initial_training_years": [2016, 2017, 2018],
    "training_start": "2016-01-04", "first_test_year": 2019, "last_complete_test_year": 2025,
    "training_window": "expanding", "refit_frequency": "annual",
    "validation_fraction_dates": .10, "validation_rounding": "ceiling; whole exchange dates",
    "graph_lookbacks_sessions": list(LOOKBACKS), "graph_frequency": "daily",
    "node_alignment": "current PIT members with all eight features throughout the longest lookback",
    "target": "next exchange-session adjusted-close return / signal-date daily volatility",
    "split_boundary": "label must be observed by each fitting cutoff; purge crossing labels",
    "validation_labels": "observed by last training-year session",
    "test_year_index": "return realization date; first signal is the previous year's final session after refitting",
    "ols": "pooled eight features plus intercept; population standardization fitted on fit rows only",
    "post_selection_refit": "fit selected model/scaler on all labeled pre-test rows known by fit cutoff",
    "frozen_during_test": ["selected graph hyperparameters", "feature scaler", "OLS coefficients"],
    "updates_during_test": ["dated PIT membership", "causal features", "five daily graphs"],
    "selection_score": "annualized Sharpe of zero-cost paper-style close-to-close validation proxy",
    "selection_scope": "one shared alpha/beta pair for all five windows",
    "tie_break": "highest score, then smallest alpha, then smallest beta; no OOS comparisons",
    "parameter_grid": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10],
    "reporting": "paper-style research proxy; execution, financing and full terminal accounting are not certified",
    "paper_departures": ["three initial OLS training years instead of ten", "annual refits instead of five-year refits"],
}


def make_folds(sessions, training_start="2016-01-04", first_test_year=2019, last_test_year=2025, validation_fraction=.1):
    sessions = pd.DatetimeIndex(sessions).sort_values()
    require(sessions.is_unique and 0 < validation_fraction < 1, "Invalid fold calendar/fraction")
    next_date = pd.Series(sessions[1:], index=sessions[:-1])
    previous_date = pd.Series(sessions[:-1], index=sessions[1:])
    folds = []
    for year in range(first_test_year, last_test_year + 1):
        training = sessions[(sessions >= training_start) & (sessions.year < year)]
        test = sessions[sessions.year == year]
        require(len(training) > 10 and len(test) > 0, "Insufficient fold dates")
        n_val = int(np.ceil(len(training) * validation_fraction))
        validation = training[-n_val:]
        fit = training[:-n_val]
        fit_cutoff, recalibration_cutoff = fit[-1], training[-1]
        fit_labeled = fit[next_date.reindex(fit).notna() & next_date.reindex(fit).le(fit_cutoff)]
        val_labeled = validation[next_date.reindex(validation).notna() & next_date.reindex(validation).le(recalibration_cutoff)]
        all_labeled = training[next_date.reindex(training).notna() & next_date.reindex(training).le(recalibration_cutoff)]
        folds.append({"test_year": year, "training_start": str(training[0].date()), "training_end": str(training[-1].date()),
            "fit_start": str(fit[0].date()), "fit_cutoff": str(fit_cutoff.date()),
            "validation_start": str(validation[0].date()), "validation_end": str(validation[-1].date()),
            "test_start": str(previous_date.loc[test[0]].date()), "test_end": str(previous_date.loc[test[-1]].date()),
            "test_label_start": str(test[0].date()), "test_label_end": str(test[-1].date()),
            "training_signal_dates": len(training), "fit_signal_dates": len(fit), "fit_labeled_dates_max": len(fit_labeled),
            "validation_signal_dates": len(validation), "validation_labeled_dates_max": len(val_labeled),
            "refit_labeled_dates_max": len(all_labeled), "test_signal_dates": len(test),
            "last_test_label_date": str(test[-1].date())})
    return pd.DataFrame(folds)


def causal_rows(frame, start, end, cutoff):
    """Future labels never change predictor availability; this selects fit rows only."""
    return frame.date.between(start, end) & frame.label_date.le(pd.Timestamp(cutoff)) & np.isfinite(frame.target)


def fit_ols(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    require(x.ndim == 2 and y.shape == (len(x),) and len(x) > x.shape[1] + 1, "Insufficient OLS rows")
    require(np.isfinite(x).all() and np.isfinite(y).all(), "OLS input contains unavailable values")
    mean, scale = x.mean(axis=0), x.std(axis=0, ddof=0)
    require((scale > 0).all(), "OLS feature has zero variance")
    design = np.column_stack([np.ones(len(x)), (x - mean) / scale])
    coefficients, _, rank, singular_values = np.linalg.lstsq(design, y, rcond=None)
    require(rank == design.shape[1], "Rank-deficient OLS design")
    return {"mean": mean, "scale": scale, "intercept": float(coefficients[0]), "coefficients": coefficients[1:],
            "fit_rows": len(x), "rank": int(rank), "condition_number": float(singular_values[0] / singular_values[-1])}


def predict_ols(model, x):
    x = np.asarray(x, dtype=float)
    require(np.isfinite(x).all(), "Prediction inputs unavailable")
    return model["intercept"] + ((x - model["mean"]) / model["scale"]) @ model["coefficients"]


def grid_groups(grid=PROTOCOL["parameter_grid"]):
    """A(alpha,beta)=alpha*A(1,alpha*beta); normalization cancels alpha.

    This identity holds for the shared pair and raw-average-before-normalizing
    ensemble. Decimal grouping retains all 121 original pairs exactly.
    """
    groups = {}
    for alpha in grid:
        for beta in grid:
            product = Decimal(str(alpha)) * Decimal(str(beta))
            groups.setdefault(product, []).append([alpha, beta])
    return [{"product": float(product), "pairs": pairs, "canonical_pair": min(pairs)}
            for product, pairs in sorted(groups.items())]


def build(output):
    require(not output.exists(), "WFO plan exists; choose a new version")
    sessions = xcals.get_calendar("XNYS", start="2016-01-01", end="2026-01-02").sessions.tz_localize(None)
    folds = make_folds(sessions)
    output.mkdir(parents=True)
    folds.to_csv(output / "folds.csv", index=False)
    write_json(output / "protocol.json", PROTOCOL)
    write_json(output / "parameter_groups.json", grid_groups())
    manifest = {"status": "complete", "purpose": "frozen WFO chronology, not fitted models or results",
        "folds": len(folds), "test_years": folds.test_year.tolist(), "original_parameter_pairs": 121,
        "distinct_normalized_graph_products": len(grid_groups()), "script_sha256": file_hash(__file__),
        "ols_models_fitted": False, "out_of_sample_evaluated": False}
    write_json(output / "manifest.json", manifest)
    print(folds.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/walk_forward/annual_expanding_v2")
    args = parser.parse_args()
    build(args.output.resolve())
