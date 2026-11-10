"""Read-only model analysis: all-member gain importance and held-out return fit.

No fits, source bars, one-minute data or execution simulations are used. Outputs
are private under data/reports. Gain shares are NOT explained return variance.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd


LABELS = {
    "nm_scaled_1h": "Return: last 60 minutes",
    "nm_scaled_1d": "Return: 1 session",
    "nm_scaled_3d": "Return: 3 sessions",
    "nm_scaled_5d": "Return: 5 sessions",
    "nm_scaled_10d": "Return: 10 sessions",
    "nm_macd_1": "Normalized MACD: short",
    "nm_macd_2": "Normalized MACD: medium",
    "nm_macd_3": "Normalized MACD: long",
    "vol_relative_activity_10d": "Relative dollar activity",
    "vol_relative_last_hour_10d": "Relative closing-hour activity",
    "vol_activity_trend_3v7": "Dollar activity: 3 vs 7 sessions",
    "vol_signed_balance_1d": "Signed dollar balance: 1 session",
    "vol_signed_balance_5d": "Signed dollar balance: 5 sessions",
    "vol_close_vwap_gap": "Prior close vs session VWAP",
    "overnight_gap": "Trade-day overnight gap",
}
for _h in ("1h","1d","3d","5d","10d"):
    for _kind,_label in [("z","Cross-sectional z"),("mean","Universe mean"),("std","Universe dispersion")]:
        LABELS[f"{_kind}_ret_{_h}"]=f"{_label}: {_h}"
LABELS.update(dict(zip([
    "path_rv_relative_10d", "path_downside_variance_share", "path_signed_efficiency",
    "path_late_variance_concentration", "path_late_vs_early_return", "path_max_drawdown_scaled",
    "path_vwap_above_fraction", "path_return_autocorr_1", "path_extrema_time_balance",
    "path_return_volume_alignment", "path_late_relative_activity", "path_signed_late_relative_activity"], [
    "Relative five-minute volatility", "Downside variance share", "Signed trend efficiency",
    "Closing-hour variance concentration", "Late vs early return", "Scaled intraday drawdown",
    "Time above cumulative VWAP", "Adjacent return correlation", "High vs low timing",
    "Return/activity alignment", "Profile-relative closing activity", "Signed closing activity"])))
GROUP_COLORS = {"Original momentum":"#428577","Momentum": "#3674aa", "Volume": "#df8d31", "Opening gap": "#9962ad", "Five-minute path":"#b85858"}


def feature_group(name):
    return "Five-minute path" if name.startswith("path_") else "Momentum" if name.startswith("nm_") else "Volume" if name.startswith("vol_") else "Opening gap" if name=="overnight_gap" else "Original momentum"


def score_returns(y, forecast):
    """Population-variance EV and standard R2 on an identical finite cohort."""
    y, forecast = np.asarray(y, dtype=float), np.asarray(forecast, dtype=float)
    assert y.shape == forecast.shape and np.isfinite(y).all() and np.isfinite(forecast).all()
    residual = y - forecast
    variance = np.var(y)
    assert variance > 0
    return dict(stock_days=len(y), explained_variance=1-np.var(residual)/variance,
        r2=1-np.mean(residual**2)/variance,
        target_std_bps=np.std(y)*10000, forecast_std_bps=np.std(forecast)*10000,
        residual_std_bps=np.std(residual)*10000,
        mean_error_bps=np.mean(forecast-y)*10000,
        rmse_bps=np.sqrt(np.mean(residual**2))*10000)


def extract_importance(run, out, *, allow_constant_models=False):
    import xgboost as xgb
    config = json.loads((run/"CONFIG.json").read_text())
    features = config["model_features"]
    models = sorted(run.glob("folds/*/members/member_*.ubj"))
    assert len(models) == 4000
    inventory = [(str(p.relative_to(run)), p.stat().st_size, p.stat().st_mtime_ns) for p in models]
    signature = hashlib.sha256(json.dumps(inventory).encode()+(run/"CONFIG.json").read_bytes()).hexdigest()
    cache, meta = out/"member_feature_importance.parquet", out/"importance_source.json"
    if cache.exists() and meta.exists() and json.loads(meta.read_text())["inventory_sha256"] == signature:
        members = pd.read_parquet(cache)
        print("Reusing verified 4,000-model importance extraction.", flush=True)
    else:
        records, started = [], time.monotonic()
        for i, path in enumerate(models, 1):
            model = xgb.Booster(params={"nthread": 1})
            model.load_model(path)
            assert model.feature_names == features
            gain, splits = model.get_score(importance_type="total_gain"), model.get_score(importance_type="weight")
            total_gain, total_splits = sum(gain.values()), sum(splits.values())
            assert (total_gain > 0 and total_splits > 0) or (allow_constant_models and total_gain==0 and total_splits==0)
            for name in features:
                records.append(dict(fold_id=path.parents[1].name, member=path.stem,
                    feature=name, group=feature_group(name), total_gain=gain.get(name, 0),
                    split_count=splits.get(name, 0), gain_share=gain.get(name, 0)/total_gain if total_gain else 0.,
                    split_share=splits.get(name, 0)/total_splits if total_splits else 0.))
            if i % 500 == 0 or i == 1:
                print(f"Importance: {i}/4000 saved models; {time.monotonic()-started:.1f}s", flush=True)
        members = pd.DataFrame(records)
        members.to_parquet(cache, index=False)
        meta.write_text(json.dumps(dict(inventory_sha256=signature, models=len(models),
            definition="Normalize summed split gain within each saved best-iteration model; equal-weight all 4,000 models."), indent=2)+"\n")
    counts = members.groupby("fold_id").member.nunique()
    assert len(counts) == 40 and counts.eq(100).all()
    shares=members.groupby(["fold_id", "member"]).gain_share.sum()
    assert np.all(np.isclose(shares,1) | (allow_constant_models & np.isclose(shares,0)))
    if allow_constant_models:
        (out/"constant_models.json").write_text(json.dumps(dict(constant_models=int(np.isclose(shares,0).sum()),total_models=len(shares),
            definition="A model with no splits uses no features and contributes zero to all gain shares. All 4,000 models retain equal weight."),indent=2)+"\n")
    quarterly = members.groupby(["fold_id", "feature", "group"], as_index=False)[["gain_share", "split_share"]].mean()
    summary = quarterly.groupby(["feature", "group"]).agg(
        mean_gain_share=("gain_share", "mean"), quarter_p10=("gain_share", lambda x:x.quantile(.1)),
        quarter_p90=("gain_share", lambda x:x.quantile(.9)),
        mean_split_share=("split_share", "mean")).reset_index().sort_values("mean_gain_share", ascending=False)
    summary["label"] = summary.feature.map(LABELS)
    quarterly.to_parquet(out/"quarterly_feature_importance.parquet", index=False)
    summary.to_parquet(out/"feature_importance.parquet", index=False)
    summary.to_csv(out/"feature_importance.csv", index=False)
    return summary, quarterly


def analyze_forecasts(run, baseline, out, old_label="momentum8",new_label="combined15"):
    columns = ["security_id", "trade_date", "fold_id", "future_oc_return", "ret_score"]
    new = pd.read_parquet(run/"predictions.parquet", columns=columns)
    old = pd.read_parquet(baseline/"predictions.parquet", columns=columns)
    keys = ["security_id", "trade_date", "fold_id"]
    joint = new.merge(old, on=keys, how="outer", suffixes=("_new", "_old"), validate="one_to_one", indicator=True)
    assert joint["_merge"].eq("both").all() and len(joint) == len(new) == len(old)
    assert np.array_equal(joint.future_oc_return_new, joint.future_oc_return_old, equal_nan=True)
    assert np.isfinite(joint[["ret_score_new", "ret_score_old"]]).all().all()
    missing = ~np.isfinite(joint.future_oc_return_new)
    data = joint.loc[~missing].copy()
    data["year"] = data.trade_date.dt.year
    day_groups = data.groupby("trade_date")
    data["y_cs"] = data.future_oc_return_new-day_groups.future_oc_return_new.transform("mean")
    records, yearly, quarterly, calibration = [], [], [], []
    for label, column in [(old_label, "ret_score_old"), (new_label, "ret_score_new")]:
        data["prediction_cs"] = data[column]-day_groups[column].transform("mean")
        for scope, ycol, pcol in [("All observed stock-days", "future_oc_return_new", column),
                                   ("Within-day cross-section", "y_cs", "prediction_cs")]:
            records.append(dict(feature_set=label, scope=scope, **score_returns(data[ycol], data[pcol])))
            for year, group in data.groupby("year"):
                yearly.append(dict(feature_set=label, scope=scope, year=int(year),
                    **score_returns(group[ycol], group[pcol])))
            for fold_id, group in data.groupby("fold_id"):
                quarterly.append(dict(feature_set=label, scope=scope, fold_id=fold_id,
                    **score_returns(group[ycol], group[pcol])))
        # Rank within each day; use every stock, not just the selected 2/3.
        rank_fraction = data.groupby("trade_date")[column].rank(method="average", pct=True)
        bins = np.ceil(rank_fraction*10).clip(1, 10).astype(int)
        daily = data.assign(prediction_decile=bins).groupby(["trade_date", "prediction_decile"]).agg(
            actual=("future_oc_return_new", "mean"), predicted=(column, "mean"), stocks=(column, "size"))
        table = daily.groupby("prediction_decile").agg(
            actual=("actual", "mean"), predicted=("predicted", "mean"),
            day_mean_std=("actual", "std"), days=("actual", "size"), stock_days=("stocks", "sum")).reset_index()
        table["feature_set"] = label
        calibration.append(table)
    metrics, annual, quarter = pd.DataFrame(records), pd.DataFrame(yearly), pd.DataFrame(quarterly)
    for table in (metrics, annual, quarter):
        assert np.allclose(table.explained_variance-table.r2,
            (table.mean_error_bps/table.target_std_bps)**2, atol=1e-12)
    deciles = pd.concat(calibration, ignore_index=True)
    for name, frame in [("explained_variance", metrics), ("explained_variance_by_year", annual),
                        ("explained_variance_by_quarter", quarter), ("return_calibration_by_decile", deciles)]:
        frame.to_parquet(out/f"{name}.parquet", index=False)
        frame.to_csv(out/f"{name}.csv", index=False)
    info = dict(forecast_rows=len(joint), observed_stock_days=len(data),
        unresolved_stock_days=int(missing.sum()), observed_dates=int(data.trade_date.nunique()),
        first_date=str(data.trade_date.min().date()), last_date=str(data.trade_date.max().date()),
        cohort="Exact matched stock-days with finite outcomes in both saved prediction files; no selected-portfolio-date mask.",
        metrics=metrics.to_dict("records"))
    return metrics, annual, quarter, deciles, info


def plot_results(importance, quarterly, metrics, annual, deciles, out):
    os.environ.setdefault("MPLCONFIGDIR", str(out/".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "axes.titleweight": "bold", "savefig.facecolor": "white"})
    ordered = importance.sort_values("mean_gain_share")
    fig, ax = plt.subplots(figsize=(12, max(7.6,.37*len(importance)+2)))
    bars = ax.barh(ordered.label, ordered.mean_gain_share*100,
        color=[GROUP_COLORS[g] for g in ordered.group], height=.69)
    # Quarter quantiles describe stability; these are not confidence intervals.
    for i, row in enumerate(ordered.itertuples()):
        ax.plot([row.quarter_p10*100, row.quarter_p90*100], [i, i], color="#303b47", lw=1.4)
        ax.plot([row.quarter_p10*100, row.quarter_p10*100], [i-.12, i+.12], color="#303b47", lw=1.1)
        ax.plot([row.quarter_p90*100, row.quarter_p90*100], [i-.12, i+.12], color="#303b47", lw=1.1)
        ax.text(max(row.mean_gain_share, row.quarter_p90)*100+.35, i, f"{row.mean_gain_share*100:.1f}%", va="center")
    ax.set_xlim(0, ordered.quarter_p90.max()*100+5)
    ax.set_xlabel("Share of training split gain (%)")
    ax.set_title("Which inputs the 4,000 saved models use", loc="left", pad=28, fontsize=16)
    ax.text(0, 1.015, "Equal model weights · lines: 10th–90th percentile across 40 quarterly fits", transform=ax.transAxes, color="#56616e")
    ax.legend(handles=[Patch(color=c, label=g) for g,c in GROUP_COLORS.items() if g in set(importance.group)], loc="lower right", frameon=False)
    ax.grid(axis="x", alpha=.16)
    ax.set_axisbelow(True)
    fig.text(.015, .015, "Gain is an in-training importance measure, not explained return variance or a causal effect.", color="#56616e", fontsize=10)
    fig.tight_layout(rect=(0,.04,1,1))
    fig.savefig(out/"feature_importance.png", dpi=170)
    plt.close(fig)

    names = importance.feature.tolist()
    heat = quarterly.pivot(index="feature", columns="fold_id", values="gain_share").reindex(names)*100
    fig, ax = plt.subplots(figsize=(14, max(7.1,.30*len(importance)+2)))
    im = ax.imshow(heat, aspect="auto", cmap="YlGnBu", vmin=0)
    ax.set_yticks(range(len(names)), [LABELS[n] for n in names])
    ticks = list(range(0,len(heat.columns),4)) + [len(heat.columns)-1]
    ax.set_xticks(ticks, heat.columns[ticks], rotation=45, ha="right")
    ax.set_title("Feature importance through the walk-forward", loc="left", fontsize=16, pad=15)
    fig.colorbar(im, ax=ax, label="Mean normalized training gain (%)", shrink=.8)
    fig.tight_layout()
    fig.savefig(out/"feature_importance_by_quarter.png", dpi=160)
    plt.close(fig)

    old_label,new_label=metrics.feature_set.drop_duplicates().tolist()
    colors = {old_label:"#8d9aa9",new_label:"#267ba7"}
    labels = {name:{"momentum8":"Momentum 8","combined15":"Momentum + volume + gap (15)","joint30":"Joint momentum + volume + gap (30)","path42":"Joint + five-minute path (42)"}[name] for name in colors}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.1))
    for ax, scope in zip(axes, ["All observed stock-days", "Within-day cross-section"]):
        table = metrics[metrics.scope.eq(scope)].set_index("feature_set")
        x = np.arange(2)
        for shift, name in [(-.18,old_label),(.18,new_label)]:
            vals = table.loc[name,["explained_variance", "r2"]].to_numpy(dtype=float)*100
            ax.bar(x+shift, vals, width=.32, color=colors[name], label=labels[name])
            for xx, value in zip(x+shift, vals):
                ax.annotate(f"{value:+.3f}%", (xx,value), xytext=(0,6 if value>=0 else -14), textcoords="offset points", ha="center", fontsize=10)
        ax.axhline(0, color="#444", lw=.8)
        ax.set_xticks(x,["Explained variance", "$R^2$"])
        ax.set_title(scope)
        ax.set_ylabel("Return variance score (%)")
        ax.margins(y=.36)
        ax.grid(axis="y", alpha=.16)
        ax.set_axisbelow(True)
    axes[0].legend(fontsize=9, frameon=False, loc="lower left")
    fig.suptitle("Out-of-sample return fit: exact matching observed stock-days", x=.05, ha="left", fontsize=15, weight="bold")
    fig.text(.025,.015,"RET is the frozen quarterly probability-weighted return forecast. No calibration on held-out returns. Negative values are retained.",color="#56616e", fontsize=9)
    fig.tight_layout(rect=(0,.045,1,.94))
    fig.savefig(out/"out_of_sample_explained_variance.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1,2,figsize=(13,4.8))
    for ax, scope in zip(axes,["All observed stock-days","Within-day cross-section"]):
        table=annual[annual.scope.eq(scope)]
        for name in colors:
            group=table[table.feature_set.eq(name)]
            ax.plot(group.year,group.explained_variance*100,marker="o",ms=4,color=colors[name],label=labels[name])
        ax.axhline(0,color="#444",lw=.8)
        ax.set_title(scope)
        ax.set_ylabel("Explained variance (%)")
        ax.set_xticks(range(2017,2027),labels=range(2017,2027),rotation=45)
        ax.grid(alpha=.16)
    axes[0].legend(fontsize=9,frameon=False)
    fig.suptitle("Out-of-sample explained variance by year",x=.05,ha="left",fontsize=15,weight="bold")
    fig.text(.025,.01,"2026 ends October 30. Within-day scores remove each day's stock-universe mean from both returns and forecasts.",fontsize=9,color="#56616e")
    fig.tight_layout(rect=(0,.04,1,.94))
    fig.savefig(out/"explained_variance_by_year.png",dpi=170)
    plt.close(fig)

    fig, axes=plt.subplots(1,2,figsize=(13,5.0),sharey=True)
    for ax,name in zip(axes,[old_label,new_label]):
        group=deciles[deciles.feature_set.eq(name)]
        ax.bar(group.prediction_decile,group.actual*10000,color=colors[name],alpha=.8,label="Realized open-to-close")
        ax.plot(group.prediction_decile,group.predicted*10000,color="#36333b",marker="o",ms=4,label="RET forecast")
        ax.axhline(0,color="#444",lw=.8)
        ax.set_title(labels[name])
        ax.set_xticks(range(1,11))
        ax.set_xlabel("Within-day forecast decile (1 = lowest RET)")
        ax.grid(axis="y",alpha=.16)
        ax.set_axisbelow(True)
        ax.legend(fontsize=9,frameon=False)
    axes[0].set_ylabel("Mean return (basis points)")
    fig.suptitle("Do higher RET scores identify higher subsequent intraday returns?",x=.05,ha="left",fontsize=14,weight="bold")
    fig.text(.025,.012,"Equal-weight stocks within each daily decile, then equal-weight days. Same-opening-price gap observation remains optimistic.",fontsize=9,color="#56616e")
    fig.tight_layout(rect=(0,.045,1,.94))
    fig.savefig(out/"out_of_sample_return_calibration.png",dpi=170)
    plt.close(fig)


def analyze(project,experiment="combined15"):
    project=Path(project).resolve()
    if experiment=="path42":
        run=project/"data/models/10_xgb_ret_rich_path100_v1"
        baseline=project/"data/models/09_xgb_ret_joint100_v1"
        out=project/"data/reports/10_train_xgb_ret_rich_path/feature_diagnostics"
        old_label,new_label="joint30","path42"
    elif experiment=="joint30":
        run=project/"data/models/09_xgb_ret_joint100_v1"
        baseline=project/"data/models/07_xgb_ret_volume_overnight100_v1"
        out=project/"data/reports/09_train_xgb_ret_joint/feature_diagnostics"
        old_label,new_label="combined15","joint30"
    else:
        run=project/"data/models/07_xgb_ret_volume_overnight100_v1"
        baseline=project/"data/models/05_xgb_ret_network_features100_uncapped_v1"
        out=project/"data/reports/07_train_xgb_ret_volume_overnight/feature_diagnostics"
        old_label,new_label="momentum8","combined15"
    out.mkdir(parents=True,exist_ok=True)
    assert json.loads((run/"RUN_STATUS.json").read_text())["status"]=="complete"
    importance,quarterly=extract_importance(run,out)
    metrics,annual,quarter,deciles,info=analyze_forecasts(run,baseline,out,old_label,new_label)
    plot_results(importance,quarterly,metrics,annual,deciles,out)
    info.update(status="complete",updated_at_utc=datetime.now(timezone.utc).isoformat(),
        model_count=4000,quarter_count=40,group_gain_shares=importance.groupby("group").mean_gain_share.sum().to_dict(),
        top_features=importance.head(5).to_dict("records"),
        explained_variance_definition="1 - population variance(actual minus RET forecast) / population variance(actual)",
        r2_definition="1 - mean squared error(actual, RET forecast) / population variance(actual)",
        within_day_definition="Subtract each observed day's universe mean from actual and forecast, then pool residuals.",
        qualifications=["Gain measures training classification-loss splits, not explained variance in returns.",
            "Correlated inputs can share or substitute importance; no feature ablation or permutation was fitted.",
            "Gain direction does not indicate bullish or bearish effects.",
            "Unresolved outcomes omitted only from scoring, not forecasting; no outcomes are filled with zero.",
            "Same-opening-price gap observation and execution remain optimistic; no one-minute data used.",
            "These reused development periods are not a new final holdout."])
    (out/"SUMMARY.json").write_text(json.dumps(info,indent=2,allow_nan=False)+"\n")
    print(json.dumps(info,indent=2),flush=True)
    return out


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment",choices=["combined15","joint30","path42"],default="combined15")
    parser.add_argument("--project",type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args()
    analyze(args.project,args.experiment)
