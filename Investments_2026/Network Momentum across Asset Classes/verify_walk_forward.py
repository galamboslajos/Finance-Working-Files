#!/usr/bin/env python3
"""Independently check WFO fits, saved predictions, boundaries and proxy returns."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import exchange_calendars as xcals

from build_momentum_features import write_json
from evaluate_walk_forward import ARMS, NETWORK, OWN, read_objects, source_batches
from extend_momentum_history import checked_object
from prepare_sp500_daily import file_hash, require


ROOT = Path(__file__).resolve().parent


def independent_ols(x, y, saved):
    x, y = np.asarray(x, float), np.asarray(y, float)
    mean = np.mean(x, axis=0)
    centered = x - mean
    scale = np.sqrt(np.mean(centered ** 2, axis=0))
    # Independent centered normal equations, rather than the evaluator's
    # standardized augmented-design least-squares routine.
    normalized = centered / scale
    coefficients = np.linalg.solve(normalized.T @ normalized, normalized.T @ (y - y.mean()))
    np.testing.assert_allclose(saved["mean"], mean, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(saved["scale"], scale, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(saved["coefficients"], coefficients, rtol=2e-7, atol=1e-9)
    np.testing.assert_allclose(saved["intercept"], y.mean(), rtol=1e-10, atol=1e-10)
    return float(np.max(np.abs(coefficients - saved["coefficients"])))


def verify(series, results, output):
    require(not output.exists(), "Verification report exists")
    manifest = json.loads((results / "manifest.json").read_text())
    require(manifest["status"] == "complete", "Incomplete WFO evaluation")
    for entry in manifest["outputs"]:
        checked_object(results, entry)
    for entry in manifest["source_year_manifests"]:
        require(file_hash(entry["path"]) == entry["sha256"], "Changed source year manifest")
    batches = source_batches(series, max(manifest["test_years"]))
    base = read_objects(batches, "base")
    scores = pd.read_csv(results / "validation_scores.csv")
    selected = pd.read_csv(results / "selected_parameters.csv")
    folds = pd.read_csv(results / "folds.csv")
    daily = pd.read_csv(results / "daily_paper_proxy_returns.csv", parse_dates=["date", "return_date"])
    first_year, last_year = min(manifest["test_years"]), max(manifest["test_years"])
    expected_calendar = xcals.get_calendar("XNYS",start=f"{first_year}-01-01",end=f"{last_year}-12-31").sessions.tz_localize(None)
    require(daily.return_date.is_unique and set(daily.return_date) == set(expected_calendar), "Unexpected OOS return calendar")
    require(scores.groupby("test_year").size().eq(30).all(), "Missing validation candidates")
    require(scores.groupby("test_year").equivalent_grid_pairs.sum().eq(121).all(), "Original grid aliases missing")
    winners = scores.sort_values(["test_year", "validation_sharpe", "alpha", "beta"], ascending=[True,False,True,True]).groupby("test_year").head(1)
    np.testing.assert_allclose(winners.sort_values("test_year")["product"], selected.sort_values("test_year")["product"], rtol=1e-14)
    max_coefficient_error, max_prediction_error, checked_models, forecasts = 0., 0., 0, 0
    source_samples = []
    for fold in folds.itertuples():
        model_record = json.loads((results / f"models_{fold.test_year}.json").read_text())
        require(model_record["training_end"] == fold.training_end, "Wrong model cutoff")
        product = float(model_record["product"])
        network = read_objects(batches, "network", product)
        pd.testing.assert_frame_equal(base[["date", "source_security_id"]], network[["date", "source_security_id"]])
        fit = base.date.between(fold.training_start, fold.training_end) & base.label_date.le(pd.Timestamp(fold.training_end)) & np.isfinite(base.target)
        require(base.loc[fit, "label_date"].max() <= pd.Timestamp(fold.training_end), "Fit contains a future label")
        require(base.loc[fit, "date"].max() < pd.Timestamp(fold.test_start), "Test signal entered its own fitting sample")
        path = results / "predictions" / f"year={fold.test_year}" / "part-00000.parquet"
        prediction = ds.dataset([str(path)], format="parquet").to_table().to_pandas().sort_values(["date", "source_security_id"], ignore_index=True)
        test = base.date.between(fold.test_start, fold.test_end) & base.label_date.between(fold.test_label_start, fold.test_label_end)
        expected = base.loc[test].reset_index(drop=True)
        pd.testing.assert_frame_equal(prediction[["date", "source_security_id"]], expected[["date", "source_security_id"]])
        np.testing.assert_allclose(prediction.target, expected.target, equal_nan=True, rtol=0, atol=0)
        np.testing.assert_allclose(prediction.forward_return, expected.forward_return, equal_nan=True, rtol=0, atol=0)
        require(prediction.label_date.dt.year.eq(fold.test_year).all(), "Forecast has the wrong return year")
        positions = set(np.linspace(0, len(prediction)-1, 3, dtype=int).tolist())
        unavailable = np.flatnonzero(~np.isfinite(prediction.target))
        if len(unavailable):
            positions.add(int(unavailable[0]))
        source_samples.append(prediction.iloc[sorted(positions)][["date", "label_date", "intrinio_security_id", "daily_volatility", "forward_return", "target"]])
        own, peer = base[OWN].to_numpy(), network[NETWORK].to_numpy()
        designs = {"LinReg": (own[fit], own[test]), "GMOM": (peer[fit], peer[test]),
                   "RegCombo": (np.column_stack([own[fit], peer[fit]]), np.column_stack([own[test], peer[test]]))}
        for name, (x_fit, x_test) in designs.items():
            saved = model_record["models"][name]
            error = independent_ols(x_fit, base.loc[fit, "target"], saved)
            max_coefficient_error = max(max_coefficient_error, error)
            independent_prediction = saved["intercept"] + np.sum((x_test - np.array(saved["mean"])) / np.array(saved["scale"]) * saved["coefficients"], axis=1)
            actual = prediction[name + "_prediction"].to_numpy()
            np.testing.assert_allclose(independent_prediction, actual, rtol=1e-11, atol=1e-11)
            max_prediction_error = max(max_prediction_error, float(np.max(abs(independent_prediction-actual))))
            checked_models += 1
        for date, frame in prediction.groupby("date", sort=True):
            saved_day = daily.loc[daily.date.eq(date)]
            require(len(saved_day) == 1, "Missing/duplicate portfolio date")
            saved_day = saved_day.iloc[0]
            require(int(saved_day.nodes) == len(frame), "Future labels changed N")
            available = np.isfinite(frame.forward_return).to_numpy()
            require(int(saved_day.available_labels) == int(available.sum()) and bool(saved_day.complete) == bool(available.all()), "Label coverage changed")
            for name in ARMS:
                position = frame[name + "_position"].to_numpy()
                weights = .15 * position / (len(frame) * np.sqrt(252) * frame.daily_volatility.to_numpy())
                np.testing.assert_allclose(saved_day[name + "_net_weight"], weights.sum(), rtol=1e-11, atol=1e-12)
                np.testing.assert_allclose(saved_day[name + "_gross_weight"], np.abs(weights).sum(), rtol=1e-11, atol=1e-12)
                realized = float(weights @ frame.forward_return.to_numpy()) if available.all() else np.nan
                np.testing.assert_allclose(saved_day[name], realized, rtol=1e-11, atol=1e-12, equal_nan=True)
        forecasts += len(prediction)
    first_frozen = ROOT / "data/walk_forward/first_fold_frozen_v1/models.json"
    if first_frozen.exists():
        frozen = json.loads(first_frozen.read_text()); evaluated = json.loads((results / "models_2019.json").read_text())
        require(frozen["product"] == evaluated["product"], "First-fold parameter changed after freezing")
        for name in ("LinReg", "GMOM", "RegCombo"):
            for field in ("mean", "scale", "coefficients", "intercept"):
                np.testing.assert_allclose(frozen["models"][name][field], evaluated["models"][name][field], rtol=1e-11, atol=1e-11)
    samples = pd.concat(source_samples, ignore_index=True)
    history = Path(batches[0][1]["history"])
    hm = json.loads((history / "manifest.json").read_text())
    wanted_dates = pd.DatetimeIndex(samples.date.tolist() + samples.label_date.tolist()).unique()
    paths = [str(checked_object(history, part)) for part in hm["history_parts"]
             if part["start"] <= str(wanted_dates.max().date()) and part["end"] >= str(wanted_dates.min().date())]
    predicate = ds.field("date").isin(pa.array(wanted_dates, type=pa.timestamp("us"))) & ds.field("intrinio_security_id").isin(samples.intrinio_security_id.unique().tolist())
    prices = ds.dataset(paths, format="parquet").to_table(columns=["date", "intrinio_security_id", "feature_adj_close", "daily_volatility"], filter=predicate).to_pandas().set_index(["intrinio_security_id", "date"])
    for row in samples.itertuples():
        current = prices.loc[(row.intrinio_security_id, row.date)]
        following = prices.loc[(row.intrinio_security_id, row.label_date)]
        expected_return = following.feature_adj_close / current.feature_adj_close - 1
        np.testing.assert_allclose(row.daily_volatility, current.daily_volatility, rtol=0, atol=0)
        np.testing.assert_allclose(row.forward_return, expected_return, rtol=1e-12, atol=1e-14, equal_nan=True)
        np.testing.assert_allclose(row.target, expected_return / current.daily_volatility, rtol=1e-12, atol=1e-12, equal_nan=True)
    metrics = pd.read_csv(results / "paper_proxy_metrics.csv")
    for row in metrics.itertuples():
        frame = daily if row.period == "all" else daily.loc[daily.test_year.eq(int(row.period))]
        values = frame[row.strategy].dropna().to_numpy()
        expected_sharpe = np.sqrt(252) * values.mean() / values.std(ddof=1)
        np.testing.assert_allclose(row.sharpe_complete_dates, expected_sharpe, rtol=1e-12)
        if frame[row.strategy].isna().any():
            require(pd.isna(row.cagr) and pd.isna(row.maximum_drawdown), "Continuous wealth reported across missing valuations")
    result = {"status":"passed", "independent_ols_models_checked":checked_models, "forecasts_preserved":forecasts,
        "portfolio_return_dates_checked":len(daily), "strategy_returns_and_exposures_checked":len(daily)*len(ARMS),
        "maximum_standardized_coefficient_error":max_coefficient_error, "maximum_prediction_absolute_error":max_prediction_error,
        "first_fold_frozen_models_unchanged":first_frozen.exists(), "selection_uses_test_labels":False,
        "independent_source_label_samples_checked":len(samples),
        "continuous_wealth_withheld_across_gaps":True, "result_manifest_sha256":file_hash(results / "manifest.json"),
        "verification_script_sha256":file_hash(__file__)}
    output.parent.mkdir(parents=True, exist_ok=True); write_json(output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series",type=Path,default=ROOT/"data/daily_networks/wfo_grid_by_year_v1")
    parser.add_argument("--results",type=Path,default=ROOT/"data/walk_forward/annual_results_v1")
    parser.add_argument("--output",type=Path,default=ROOT/"data/walk_forward/verification_v1/report.json")
    args=parser.parse_args()
    verify(args.series.resolve(),args.results.resolve(),args.output.resolve())
