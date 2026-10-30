#!/usr/bin/env python3
"""Select graph parameters on past validation dates, refit, then evaluate annual WFO.

Results are the paper's zero-cost close-to-close research proxy. Missing held
valuations, executable fills, financing and terminal cash flows are not solved
by this evaluator. Missing forward returns preserve forecasts and withhold the
whole common-cohort portfolio date; continuous wealth metrics are then absent.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from build_momentum_features import FEATURES, write_json
from extend_momentum_history import checked_object
from prepare_sp500_daily import canonical_hash, file_hash, require, write_parquet
from walk_forward import PROTOCOL, causal_rows, fit_ols, grid_groups, make_folds, predict_ols


ROOT = Path(__file__).resolve().parent
OWN = ["own_" + x for x in FEATURES]
NETWORK = ["network_" + x for x in FEATURES]
ARMS = ("LongOnly", "MACD", "LinReg", "GMOM", "RegCombo", "SignCombo")


def paper_proxy_returns(frame, positions, cutoff=None):
    """All predicted nodes count in N; labels never redefine the portfolio."""
    position = np.asarray(positions, float)
    require(position.shape == (len(frame),) and np.isfinite(position).all(), "Invalid positions")
    known = np.ones(len(frame), bool) if cutoff is None else frame.label_date.le(pd.Timestamp(cutoff)).to_numpy()
    realized = frame.forward_return.to_numpy(float)
    available = known & np.isfinite(realized)
    realized = np.where(known, realized, np.nan)
    work = pd.DataFrame({"date": frame.date.to_numpy(), "available": available,
        "risk_unit": position * .15 / (np.sqrt(252) * frame.daily_volatility.to_numpy(float)),
        "contribution": position * .15 * realized / (np.sqrt(252) * frame.daily_volatility.to_numpy(float))})
    grouped = work.groupby("date", sort=True)
    count = grouped.size()
    available_count = grouped.available.sum()
    # Sum with missing observations is only a diagnostic; it cannot be reported
    # as a complete portfolio return, even after dividing by all current nodes.
    result = grouped.contribution.sum() / count
    result = result.where(available_count.eq(count))
    return pd.DataFrame({"date": count.index, "nodes": count.to_numpy(), "available_labels": available_count.to_numpy(),
                         "return": result.to_numpy(), "complete": available_count.eq(count).to_numpy(),
                         "net_weight": (grouped.risk_unit.sum() / count).to_numpy(),
                         "gross_weight": (work.risk_unit.abs().groupby(work.date).sum() / count).to_numpy()})


def sharpe(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if len(values) < 20 or np.std(values, ddof=1) == 0:
        return np.nan
    return float(np.sqrt(252) * values.mean() / values.std(ddof=1))


def performance(values):
    values = np.asarray(values, float)
    finite = values[np.isfinite(values)]
    result = {"expected_dates": len(values), "complete_dates": len(finite), "missing_dates": len(values) - len(finite),
        "complete_date_fraction": len(finite) / len(values),
        "annualized_mean": float(252 * finite.mean()) if len(finite) else None,
        "annualized_volatility": float(np.sqrt(252) * finite.std(ddof=1)) if len(finite) > 1 else None,
        "sharpe_complete_dates": float(sharpe(finite)) if np.isfinite(sharpe(finite)) else None,
        "cagr": None, "maximum_drawdown": None}
    if len(finite) == len(values) and len(values) and (values > -1).all():
        wealth = np.r_[1., np.exp(np.cumsum(np.log1p(values)))]
        result["cagr"] = float(np.expm1(np.log(wealth[-1]) * 252 / len(values)))
        result["maximum_drawdown"] = float(np.min(wealth / np.maximum.accumulate(wealth) - 1))
    return result


def json_model(model):
    return {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in model.items()}


def source_batches(root, last_year):
    manifests = []
    for year in range(2016, last_year + 1):
        directory = root / str(year)
        path = directory / "manifest.json"
        require(path.exists(), f"Daily grid year {year} has not completed")
        manifest = json.loads(path.read_text())
        require(manifest["status"] == "complete" and manifest["sample_dates"] is None and len(manifest["products"]) == 30,
                f"Year {year} is not the full causal grid")
        manifests.append((directory, manifest))
    require(len({m["history_manifest_sha256"] for _, m in manifests}) == 1, "Mixed price histories")
    require(len({tuple(m["products"]) for _, m in manifests}) == 1, "Mixed parameter grids")
    return manifests


def read_objects(batches, kind, product=None):
    paths = []
    for directory, manifest in batches:
        candidates = manifest["base_files"] if kind == "base" else [p for p in manifest["network_files"] if f"product={product:.12g}/" in p]
        objects = {p["path"]:p for p in manifest["outputs"]}
        require(len(candidates) == 1, "Unexpected annual file count")
        paths.extend(str(checked_object(directory, objects[p])) for p in candidates)
    return ds.dataset(paths, format="parquet").to_table().to_pandas().sort_values(["date", "source_security_id"], ignore_index=True)


def evaluate(series, plan, output, last_test_year=2025):
    require(not output.exists(), "Evaluation exists; choose a new version")
    saved_protocol = json.loads((plan / "protocol.json").read_text())
    require(canonical_hash(saved_protocol) == canonical_hash(PROTOCOL), "Protocol changed since the plan was frozen")
    batches = source_batches(series, last_test_year)
    base = read_objects(batches, "base")
    require(not base.duplicated(["date", "source_security_id"]).any(), "Duplicate predictor rows")
    sessions = xcals.get_calendar("XNYS", start="2016-01-01", end="2026-01-02").sessions.tz_localize(None)
    folds = make_folds(sessions, last_test_year=last_test_year)
    expected = sessions[(sessions >= PROTOCOL["training_start"]) & (sessions.year <= last_test_year)]
    require(set(base.date.unique()) == set(expected), "Missing daily network dates")
    require(np.isfinite(base[OWN + ["daily_volatility"]]).all().all(), "Unavailable current predictors")
    require((base.daily_volatility > 0).all(), "Invalid volatility")
    output.mkdir(parents=True)
    scores = []
    for group in grid_groups():
        product = group["product"]
        network = read_objects(batches, "network", product)
        pd.testing.assert_frame_equal(base[["date", "source_security_id"]], network[["date", "source_security_id"]])
        x = network[NETWORK].to_numpy()
        require(np.isfinite(x).all(), "Unavailable network predictors")
        for fold in folds.itertuples():
            fit = causal_rows(base, fold.fit_start, fold.fit_cutoff, fold.fit_cutoff)
            model = fit_ols(x[fit], base.loc[fit, "target"])
            validation = base.date.between(fold.validation_start, fold.validation_end)
            frame = base.loc[validation].copy()
            prediction = predict_ols(model, x[validation])
            returns = paper_proxy_returns(frame, np.sign(prediction), cutoff=fold.training_end)
            score = sharpe(returns["return"])
            require(np.isfinite(score), f"Insufficient complete validation dates in fold {fold.test_year}")
            label_known = frame.label_date.le(pd.Timestamp(fold.training_end)) & np.isfinite(frame.target)
            alpha, beta = group["canonical_pair"]
            scores.append({"test_year": fold.test_year, "product": product, "alpha": alpha, "beta": beta,
                "equivalent_grid_pairs": len(group["pairs"]), "validation_sharpe": score,
                "validation_mse": float(np.mean(np.square(prediction[label_known] - frame.loc[label_known, "target"]))),
                "fit_rows": model["fit_rows"], "validation_signal_dates": len(returns),
                "validation_complete_portfolio_dates": int(returns.complete.sum()),
                "latest_fitted_label_date": str(base.loc[fit, "label_date"].max().date())})
        print(json.dumps({"stage": "validation", "product": product, "completed_products": len(scores)//len(folds), "total_products": 30}), flush=True)
    score_table = pd.DataFrame(scores)
    score_table.to_csv(output / "validation_scores.csv", index=False)
    selected = score_table.sort_values(["test_year", "validation_sharpe", "alpha", "beta"],
        ascending=[True,False,True,True]).groupby("test_year", sort=True).head(1).sort_values("test_year")
    selected.to_csv(output / "selected_parameters.csv", index=False)
    # Selection is fixed before accessing OOS outcomes for performance reporting.
    predictions, daily, model_records = [], [], []
    for fold in folds.itertuples():
        selection = selected.loc[selected.test_year.eq(fold.test_year)].iloc[0]
        product = float(selection["product"])
        network = read_objects(batches, "network", product)
        x_network, x_own = network[NETWORK].to_numpy(), base[OWN].to_numpy()
        fit = causal_rows(base, fold.training_start, fold.training_end, fold.training_end)
        require(base.loc[fit, "label_date"].max() <= pd.Timestamp(fold.training_end), "Future label entered refit")
        test = base.date.between(fold.test_start, fold.test_end) & base.label_date.between(fold.test_label_start, fold.test_label_end)
        frame = base.loc[test].copy()
        models = {"LinReg": fit_ols(x_own[fit], base.loc[fit, "target"]),
                  "GMOM": fit_ols(x_network[fit], base.loc[fit, "target"]),
                  "RegCombo": fit_ols(np.column_stack([x_own[fit], x_network[fit]]), base.loc[fit, "target"])}
        frame["LinReg_prediction"] = predict_ols(models["LinReg"], x_own[test])
        frame["GMOM_prediction"] = predict_ols(models["GMOM"], x_network[test])
        frame["RegCombo_prediction"] = predict_ols(models["RegCombo"], np.column_stack([x_own[test], x_network[test]]))
        frame["LongOnly_position"] = 1.
        macd = frame[["own_macd_8_24", "own_macd_16_48", "own_macd_32_96"]].to_numpy()
        frame["MACD_position"] = np.mean(macd * np.exp(-macd ** 2 / 4) / .89, axis=1)
        for name in ("LinReg", "GMOM", "RegCombo"):
            frame[name + "_position"] = np.sign(frame[name + "_prediction"])
        frame["SignCombo_position"] = .5 * (frame.LinReg_position + frame.GMOM_position)
        frame["test_year"] = fold.test_year
        frame["alpha"], frame["beta"], frame["product"] = float(selection.alpha), float(selection.beta), product
        write_parquet(frame, output / "predictions" / f"year={fold.test_year}" / "part-00000.parquet")
        results = None
        for arm in ARMS:
            values = paper_proxy_returns(frame, frame[arm + "_position"])
            if results is None:
                results = values.rename(columns={"return": arm, "net_weight": arm + "_net_weight", "gross_weight": arm + "_gross_weight"})
            else:
                np.testing.assert_array_equal(results.complete, values.complete)
                results[arm] = values["return"].to_numpy()
                results[arm + "_net_weight"] = values.net_weight.to_numpy()
                results[arm + "_gross_weight"] = values.gross_weight.to_numpy()
        label_lookup = frame.groupby("date").label_date.first()
        results["return_date"] = results.date.map(label_lookup)
        require(results.return_date.dt.year.eq(fold.test_year).all(), "Return realized outside its assigned test year")
        results["test_year"] = fold.test_year
        daily.append(results); predictions.append(frame[["date", "target", "LinReg_prediction", "GMOM_prediction", "RegCombo_prediction"]])
        record = {"test_year": fold.test_year, "training_start": fold.training_start, "training_end": fold.training_end,
            "latest_fitted_label_date": str(base.loc[fit, "label_date"].max().date()),
            "alpha": float(selection.alpha), "beta": float(selection.beta), "product": product,
            "feature_order": {"LinReg": OWN, "GMOM": NETWORK, "RegCombo": OWN + NETWORK},
            "models": {name: json_model(model) for name, model in models.items()}}
        write_json(output / f"models_{fold.test_year}.json", record); model_records.append(record)
    daily = pd.concat(daily, ignore_index=True)
    daily.to_csv(output / "daily_paper_proxy_returns.csv", index=False)
    metrics = []
    for label, frame in [(str(y), daily.loc[daily.test_year.eq(y)]) for y in folds.test_year] + [("all", daily)]:
        for arm in ARMS:
            metrics.append({"period": label, "strategy": arm, **performance(frame[arm])})
    pd.DataFrame(metrics).to_csv(output / "paper_proxy_metrics.csv", index=False)
    forecasts = pd.concat(predictions, ignore_index=True)
    available = np.isfinite(forecasts.target)
    forecast_metrics = {name: {"labeled_rows": int(available.sum()),
        "mse": float(np.mean(np.square(forecasts.loc[available, name + "_prediction"] - forecasts.loc[available, "target"]))) }
        for name in ("LinReg", "GMOM", "RegCombo")}
    forecast_metrics["ZeroForecast"] = {"labeled_rows": int(available.sum()), "mse": float(np.mean(np.square(forecasts.loc[available, "target"])))}
    write_json(output / "forecast_metrics.json", forecast_metrics)
    write_json(output / "protocol.json", PROTOCOL)
    folds.to_csv(output / "folds.csv", index=False)
    manifest = {"status": "complete", "purpose": "annual expanding WFO; zero-cost paper-style research proxy",
        "protocol_id": PROTOCOL["id"], "test_years": folds.test_year.tolist(), "folds": len(folds),
        "validation_candidates_per_fold": 121, "distinct_normalized_graph_candidates": 30,
        "models_refitted": len(model_records) * 3, "out_of_sample_signal_dates": len(daily),
        "out_of_sample_predictor_rows": len(forecasts), "out_of_sample_labeled_rows": int(available.sum()),
        "complete_portfolio_dates": int(daily.complete.sum()), "unavailable_portfolio_dates": int((~daily.complete).sum()),
        "selection_uses_test_labels": False, "missing_labels_change_nodes": False,
        "source_year_manifests": [{"path": str(directory / "manifest.json"), "sha256": file_hash(directory / "manifest.json")} for directory, _ in batches],
        "script_sha256": file_hash(__file__), "protocol_script_sha256": file_hash(ROOT / "walk_forward.py"),
        "limits": ["Research proxy uses final signal close to next close; these are not certified executable returns.",
                   "Costs, borrow/financing, portfolio volatility targeting, terminal distributions and missing-held valuation accounting are not included.",
                   "All source identity, corporate-action and provisional close-review limitations remain.",
                   "With missing forward observations, Sharpe and moments describe only complete portfolio dates; CAGR and drawdown are withheld across gaps."]}
    manifest["outputs"] = [{"path": p.relative_to(output).as_posix(), "size_bytes": p.stat().st_size, "sha256": file_hash(p)}
                           for p in sorted(output.rglob("*")) if p.is_file()]
    write_json(output / "manifest.json", manifest)
    print(json.dumps({k:v for k,v in manifest.items() if k not in {"outputs", "source_year_manifests", "limits"}}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series", type=Path, default=ROOT / "data/daily_networks/wfo_grid_by_year_v1")
    parser.add_argument("--plan", type=Path, default=ROOT / "data/walk_forward/annual_expanding_v2")
    parser.add_argument("--output", type=Path, default=ROOT / "data/walk_forward/annual_results_v1")
    parser.add_argument("--last-test-year", type=int, default=2025)
    args = parser.parse_args()
    evaluate(args.series.resolve(), args.plan.resolve(), args.output.resolve(), args.last_test_year)
