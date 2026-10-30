#!/usr/bin/env python3
"""Freeze the first fold using only completed 2016-2018 network tables."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import exchange_calendars as xcals
import numpy as np
import pandas as pd

from build_momentum_features import write_json
from evaluate_walk_forward import NETWORK, OWN, json_model, paper_proxy_returns, read_objects, sharpe
from prepare_sp500_daily import file_hash, require
from walk_forward import PROTOCOL, causal_rows, fit_ols, grid_groups, make_folds, predict_ols


ROOT = Path(__file__).resolve().parent


def freeze(series, output):
    require(not output.exists(), "Frozen first fold exists")
    batches = []
    for year in (2016,2017,2018):
        directory = series / str(year)
        manifest = json.loads((directory / "manifest.json").read_text())
        require(manifest["status"] == "complete", "Incomplete training year")
        batches.append((directory,manifest))
    base = read_objects(batches,"base")
    # Make unavailable boundary labels explicit before any selection code runs.
    unknown = base.label_date.gt(pd.Timestamp("2018-12-31"))
    base.loc[unknown,["forward_return","target"]] = np.nan
    sessions = xcals.get_calendar("XNYS",start="2016-01-01",end="2019-12-31").sessions.tz_localize(None)
    fold = make_folds(sessions,last_test_year=2019).iloc[0]
    fit = causal_rows(base,fold.fit_start,fold.fit_cutoff,fold.fit_cutoff)
    validation = base.date.between(fold.validation_start,fold.validation_end)
    scores=[]
    for group in grid_groups():
        network = read_objects(batches,"network",group["product"])
        pd.testing.assert_frame_equal(base[["date","source_security_id"]],network[["date","source_security_id"]])
        model=fit_ols(network.loc[fit,NETWORK],base.loc[fit,"target"])
        prediction=predict_ols(model,network.loc[validation,NETWORK])
        returns=paper_proxy_returns(base.loc[validation],np.sign(prediction),cutoff=fold.training_end)
        score=sharpe(returns["return"])
        require(np.isfinite(score),"Insufficient validation dates")
        scores.append({"product":group["product"],"alpha":group["canonical_pair"][0],"beta":group["canonical_pair"][1],
                       "validation_sharpe":score,"equivalent_grid_pairs":len(group["pairs"]),"fit_rows":model["fit_rows"],
                       "validation_complete_dates":int(returns.complete.sum())})
    table=pd.DataFrame(scores)
    chosen=table.sort_values(["validation_sharpe","alpha","beta"],ascending=[False,True,True]).iloc[0]
    network=read_objects(batches,"network",float(chosen["product"]))
    refit=causal_rows(base,fold.training_start,fold.training_end,fold.training_end)
    models={"LinReg":fit_ols(base.loc[refit,OWN],base.loc[refit,"target"]),
            "GMOM":fit_ols(network.loc[refit,NETWORK],base.loc[refit,"target"]),
            "RegCombo":fit_ols(np.column_stack([base.loc[refit,OWN],network.loc[refit,NETWORK]]),base.loc[refit,"target"])}
    output.mkdir(parents=True)
    table.to_csv(output/"validation_scores.csv",index=False)
    record={"status":"complete","test_return_year":2019,"training_start":fold.training_start,"training_end":fold.training_end,
        "latest_fitted_label_date":str(base.loc[refit,"label_date"].max().date()),
        "first_test_signal_date":fold.test_start,"first_test_return_date":fold.test_label_start,
        "alpha":float(chosen.alpha),"beta":float(chosen.beta),"product":float(chosen["product"]),
        "models":{name:json_model(model) for name,model in models.items()},"protocol_id":PROTOCOL["id"],
        "past_year_manifests":[{"path":str(directory/"manifest.json"),"sha256":file_hash(directory/"manifest.json")} for directory,_ in batches],
        "test_returns_read_for_selection":False,"test_forecasts_evaluated":False,"script_sha256":file_hash(__file__)}
    write_json(output/"models.json",record)
    print(json.dumps({k:v for k,v in record.items() if k not in {"models","past_year_manifests"}},indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series",type=Path,default=ROOT/"data/daily_networks/wfo_grid_by_year_v1")
    parser.add_argument("--output",type=Path,default=ROOT/"data/walk_forward/first_fold_frozen_v1")
    args=parser.parse_args()
    freeze(args.series.resolve(),args.output.resolve())
