#!/usr/bin/env python3
"""Render the completed research WFO and verification results as a repository report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_walk_forward import ARMS
from prepare_sp500_daily import require


ROOT = Path(__file__).resolve().parent


def number(value, percent=False):
    if pd.isna(value):
        return "withheld"
    return f"{100*value:.2f}%" if percent else f"{value:.3f}"


def report(results, verification, output):
    manifest=json.loads((results/"manifest.json").read_text())
    checks=json.loads(verification.read_text())
    require(manifest["status"]=="complete" and checks["status"]=="passed","Completed verified results required")
    metrics=pd.read_csv(results/"paper_proxy_metrics.csv")
    selected=pd.read_csv(results/"selected_parameters.csv")
    scores=pd.read_csv(results/"validation_scores.csv")
    forecasts=json.loads((results/"forecast_metrics.json").read_text())
    daily=pd.read_csv(results/"daily_paper_proxy_returns.csv")
    all_metrics=metrics.loc[metrics.period.eq("all")].set_index("strategy")
    lines=["# Annual expanding walk-forward results", "",
        "The selected experiment is complete: three initial OLS training years (2016–2018), expanding annual refits, "
        "and seven unseen return years (2019–2025). All five daily graph lookbacks are retained. "
        "These results use the paper's zero-cost close-to-close research proxy, without an additional portfolio-volatility target.", "",
        f"The run preserves {manifest['out_of_sample_predictor_rows']:,} out-of-sample stock/date forecasts over "
        f"{manifest['out_of_sample_signal_dates']:,} return dates. The original 121 graph-parameter pairs are all represented "
        "by 30 mathematically equivalent-product groups. Each annual selection uses past validation labels only.", "",
        f"**Coverage limits the return interpretation:** {manifest['unavailable_portfolio_dates']} portfolio dates have "
        "at least one unavailable next-session return. Their forecasts remain saved, but the entire matched-cohort date is "
        "unavailable for return reporting. Sharpe and moments below describe complete dates. Continuous CAGR and drawdown "
        "are withheld across gaps; no missing return is treated as zero.", "", "## Matched-cohort research-proxy comparison", "",
        "| Strategy | Annualized mean, complete dates | Annualized volatility, complete dates | Sharpe, complete dates |", "| --- | ---: | ---: | ---: |"]
    for arm in ARMS:
        row=all_metrics.loc[arm]
        lines.append(f"| {arm} | {number(row.annualized_mean,True)} | {number(row.annualized_volatility,True)} | {number(row.sharpe_complete_dates)} |")
    gmom, own=all_metrics.loc["GMOM"],all_metrics.loc["LinReg"]
    comparison="higher" if gmom.sharpe_complete_dates>own.sharpe_complete_dates else "lower"
    lines += ["",f"GMOM's complete-date Sharpe is **{number(gmom.sharpe_complete_dates)}**, {comparison} than "
        f"matched own-stock LinReg's **{number(own.sharpe_complete_dates)}**. This comparison does not establish executable "
        "profits or causal information transmission. The small difference and unresolved valuation dates warrant cautious interpretation. LongOnly is the paper's inverse-volatility-sized cohort, not a "
        "market-cap-weighted S&P 500 total-return index.","", "## Annual unseen blocks", "",
        "| Return year | OLS history | GMOM Sharpe | LinReg Sharpe | Complete portfolio dates |", "| --- | --- | ---: | ---: | ---: |"]
    for year in manifest["test_years"]:
        gm=metrics.loc[metrics.period.eq(str(year))&metrics.strategy.eq("GMOM")].iloc[0]
        lr=metrics.loc[metrics.period.eq(str(year))&metrics.strategy.eq("LinReg")].iloc[0]
        lines.append(f"| {year} | 2016–{year-1} | {number(gm.sharpe_complete_dates)} | {number(lr.sharpe_complete_dates)} | {int(gm.complete_dates)}/{int(gm.expected_dates)} |")
    lines += ["", "Test years use the return realization date. For example, the first 2019 return is realized on "
        "2019-01-02 from a forecast after the 2018-12-31 refit. That following return cannot enter fitting or validation. "
        "Models and scalers remain fixed within each return year, while networks and dated membership update daily.","",
        "## Graph parameter selection", "", "| Test year | Graph alpha | Graph beta | Product | Validation Sharpe | Candidate groups within 1e-12 of best |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in selected.itertuples():
        sample=scores.loc[scores.test_year.eq(row.test_year)]
        near=int(np.isclose(sample.validation_sharpe,row.validation_sharpe,rtol=0,atol=1e-12).sum())
        lines.append(f"| {row.test_year} | {row.alpha:g} | {row.beta:g} | {getattr(row,'product'):g} | {number(row.validation_sharpe)} | {near}/30 |")
    lines += ["", "A flat validation score across candidates is not evidence of a uniquely identified optimum. "
        "Many distinct small penalties can produce the same signed validation positions. Exact score ties follow the "
        "predeclared smallest-alpha, then smallest-beta rule. No test score enters selection.", "", "## Forecast diagnostics", "",
        "| Forecast | Volatility-scaled target MSE | Labeled stock/date observations |", "| --- | ---: | ---: |"]
    for name in ("ZeroForecast","LinReg","GMOM","RegCombo"):
        item=forecasts[name]
        lines.append(f"| {name} | {item['mse']:.6f} | {item['labeled_rows']:,} |")
    correlation=daily[["GMOM","LinReg"]].dropna().corr().iloc[0,1]
    lines += ["",f"GMOM/LinReg complete-date return correlation is {correlation:.3f}. "
        f"GMOM's mean intended net weight is {daily.GMOM_net_weight.mean():.3f}, and mean gross weight is "
        f"{daily.GMOM_gross_weight.mean():.3f}. The method does not enforce market neutrality. "
        "Forecast MSE uses only observed labels; missing-label forecasts remain in the prediction files.", "",
        "## Verification", "",f"The independent verifier checked {checks['independent_ols_models_checked']} fitted OLS models "
        "using centered normal equations, separately from the evaluator's least-squares routine. It replayed every saved "
        "forecast and all strategy returns/exposures, verified parameter selection, label cutoffs and unchanged first-fold "
        "models frozen before OOS evaluation. Maximum standardized coefficient discrepancy was "
        f"{checks['maximum_standardized_coefficient_error']:.3g}; maximum prediction discrepancy was "
        f"{checks['maximum_prediction_absolute_error']:.3g}.", "",
        "Structural tests cover feature formulas, missing-session resets, future-prefix invariance, source/member preservation, "
        "graph objective/normalization, exact grid grouping, chronological fitting boundaries, training-only standardization, "
        "missing-label preservation and withholding continuous wealth metrics across gaps.","",
        "## Additional performance diagnostics", "", "The [extended performance report](ANNUAL_WALK_FORWARD_PERFORMANCE.md) "
        "adds Sortino, positive-day frequency, tail losses, monthly returns, correlations, rolling risk and exposures. "
        "Its equity and drawdown charts reset within each uninterrupted complete-data run. "
        "The self-contained HTML, PDF charts and CSV tables are saved in `data/walk_forward/performance_v1/`.", "",
        "## Practical limits and saved artifacts", "", "The calculation does not include executable fills, trading costs, "
        "borrow/financing, complete corporate-action/terminal distributions or a solution for missing held valuations. "
        "The 107 close reviews remain provisional, and historical identity/price-vintage limitations remain. "
        "PIT membership and WFO prevent specific forms of lookahead; they do not certify complete or unbiased economic returns.","",
        "| Artifact | Location |", "| --- | --- |",
        "| Frozen chronology and protocol | `data/walk_forward/annual_expanding_v2/` |",
        "| First-fold models frozen before OOS evaluation | `data/walk_forward/first_fold_frozen_v1/` |",
        "| All validation scores and selected graph parameters | `data/walk_forward/annual_results_v1/validation_scores.csv`, `selected_parameters.csv` |",
        "| Annual scalers and OLS coefficients | `data/walk_forward/annual_results_v1/models_YYYY.json` |",
        "| Every OOS forecast, including missing labels | `data/walk_forward/annual_results_v1/predictions/` |",
        "| Daily returns, date coverage and net/gross weights | `data/walk_forward/annual_results_v1/daily_paper_proxy_returns.csv` |",
        "| Annual/aggregate research-proxy metrics | `data/walk_forward/annual_results_v1/paper_proxy_metrics.csv` |",
        "| Independent numerical verification | `data/walk_forward/verification_v1/report.json` |",
        "| Source/policy details and reproduction | [Annual WFO protocol](ANNUAL_WALK_FORWARD.md) |", ""]
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text("\n".join(lines))
    print(str(output))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results",type=Path,default=ROOT/"data/walk_forward/annual_results_v1")
    parser.add_argument("--verification",type=Path,default=ROOT/"data/walk_forward/verification_v1/report.json")
    parser.add_argument("--output",type=Path,default=ROOT/"docs/ANNUAL_WALK_FORWARD_RESULTS.md")
    args=parser.parse_args()
    report(args.results.resolve(),args.verification.resolve(),args.output.resolve())
