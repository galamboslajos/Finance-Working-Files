#!/usr/bin/env python3
"""Export compact, hash-checked original WFO outputs for repository review."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from prepare_sp500_daily import file_hash, require

ROOT = Path(__file__).resolve().parent


def export(output):
    require(not output.exists(), "Review outputs already exist; choose a fresh directory")
    result = ROOT / "data/walk_forward/annual_results_v1"
    charts = ROOT / "data/walk_forward/performance_v1"
    verification = ROOT / "data/walk_forward/verification_v1/report.json"
    plan = ROOT / "data/walk_forward/annual_expanding_v2"
    result_manifest = json.loads((result / "manifest.json").read_text())
    chart_manifest = json.loads((charts / "manifest.json").read_text())
    checked = json.loads(verification.read_text())
    require(result_manifest["status"] == chart_manifest["status"] == "complete" and checked["status"] == "passed", "Completed verified results required")
    source_hash = file_hash(result / "manifest.json")
    require(source_hash == chart_manifest["results_manifest_sha256"] == checked["result_manifest_sha256"], "Mixed source results")
    source_objects = {obj["path"]: obj for obj in result_manifest["outputs"]}
    for source in result.iterdir():
        if source.is_file() and source.name != "manifest.json":
            require(file_hash(source) == source_objects[source.name]["sha256"], "Original result file changed")
    for obj in chart_manifest["outputs"]:
        require(file_hash(charts / obj["path"]) == obj["sha256"], "Performance output changed")
    copied = []

    def copy(source, relative):
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        require(file_hash(source) == file_hash(destination), "Export copy differs")
        copied.append({"source_relative_path": source.relative_to(ROOT).as_posix(),
                       "path": relative, "sha256": file_hash(destination), "size_bytes": destination.stat().st_size})

    for source in sorted(result.iterdir()):
        if source.is_file():
            copy(source, "results/" + source.name)
    for source in sorted(charts.iterdir()):
        if source.is_file():
            copy(source, "performance/" + source.name)
    for source in sorted(plan.iterdir()):
        if source.is_file():
            copy(source, "protocol/" + source.name)
    copy(verification, "verification/wfo_report.json")
    readme = """# Original full universe walk forward outputs

These saved outputs cover the original S&P 500 network-momentum experiment:
2016–2018 initial OLS training, expanding annual refits and 2019–2025 test
returns. All eligible PIT members receive a long or short position based on
their forecast sign, sized by asset volatility. There is no top-N selector.
The test holds 426–467 stocks daily, averaging about 439.

## Start here

- [Results and annual comparisons](../../docs/ANNUAL_WALK_FORWARD_RESULTS.md)
- [Additional performance metrics](../../docs/ANNUAL_WALK_FORWARD_PERFORMANCE.md)
- [Four-page chart PDF](performance/performance_charts.pdf)
- [Self-contained HTML report](performance/performance_report.html): download and open locally; GitHub displays HTML source rather than running the report.
- [WFO chronology and methodology](../../docs/ANNUAL_WALK_FORWARD.md)

## Performance overview

![Annual performance, monthly returns and correlations](performance/performance_overview.png)

## Equity and drawdown

The 38 unavailable portfolio dates remain gaps. Each uninterrupted run starts
from a new 100 baseline; these lines do not represent continuous seven-year
wealth. Full-period CAGR and maximum drawdown remain withheld.

![Segment equity and drawdown with missing dates preserved](performance/equity_drawdown_segments.png)

The [longest complete interval](performance/longest_complete_period.png) covers
140 sessions, 2024-05-06 through 2024-11-21. It was selected by continuity only.
The [rolling risk and exposure chart](performance/rolling_risk_and_exposure.png)
requires all 63 daily returns for each rolling statistic.

## Tables and fitted models

| Output | File |
| --- | --- |
| Daily returns, coverage and intended net/gross exposures for all six strategies | [daily_paper_proxy_returns.csv](results/daily_paper_proxy_returns.csv) |
| Annual and aggregate performance | [paper_proxy_metrics.csv](results/paper_proxy_metrics.csv) |
| Sortino, positive-day frequency and tail diagnostics | [extended_metrics.csv](performance/extended_metrics.csv) |
| Monthly returns with incomplete months withheld | [monthly_returns.csv](performance/monthly_returns.csv) |
| Return correlations | [return_correlations.csv](performance/return_correlations.csv) |
| Equity and drawdown within uninterrupted runs | [segment_curves.csv](performance/segment_curves.csv) |
| Every graph-validation candidate score | [validation_scores.csv](results/validation_scores.csv) |
| Annual selected graph parameters | [selected_parameters.csv](results/selected_parameters.csv) |
| Annual model coefficients and feature scalers | [2019](results/models_2019.json), [2020](results/models_2020.json), [2021](results/models_2021.json), [2022](results/models_2022.json), [2023](results/models_2023.json), [2024](results/models_2024.json), [2025](results/models_2025.json) |
| Exact annual fold boundaries | [folds.csv](protocol/folds.csv) |
| Forecast MSE diagnostics | [forecast_metrics.json](results/forecast_metrics.json) |
| Independent full-run numerical verification | [wfo_report.json](verification/wfo_report.json) |

## Interpretation and provenance

The source run contains 771,773 OOS forecasts and 21 fitted OLS models across
1,760 return dates. The verifier independently replayed every forecast and
strategy return. Complete-date GMOM Sharpe is 0.757, compared with 0.774 for
own-stock LinReg. All 63 tests for the original pipeline and reporting pass.

Returns are a zero-cost close-to-close research proxy. They exclude executable
fills, trading costs, borrow/financing and complete terminal/corporate-action
cash flows. PIT membership and chronological fitting do not resolve all source
identity or price-history limitations. LongOnly is an inverse-volatility-sized
matched cohort, not a capital-weighted S&P 500 index.

This folder contains compact review outputs, without raw prices, feature
matrices, daily graph arrays or the approximately 100 MB of stock/date
prediction tables. The original [result manifest](results/manifest.json) and
verification describe the complete local run and reference those omitted
artifacts. Re-running their full verification needs the pinned local dataset;
viewing these tables and charts does not.

The [export manifest](export_manifest.json) records hashes for every copied
artifact. The analysis and chart values are unchanged from the verified run.
"""
    (output / "README.md").write_text(readme)
    manifest = {"status": "complete", "strategy": "original full eligible universe; sign forecast; asset-volatility sizing",
        "purpose": "compact GitHub review outputs", "source_result_manifest_sha256": source_hash,
        "source_performance_manifest_sha256": file_hash(charts / "manifest.json"),
        "export_script_sha256": file_hash(__file__), "original_test_suite_count": 63,
        "top_n_selector": False, "raw_dataset_included": False, "stock_date_prediction_tables_included": False,
        "copied_outputs": copied, "readme_sha256": file_hash(output / "README.md")}
    (output / "export_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": "complete", "copied_files": len(copied), "bytes": sum(obj["size_bytes"] for obj in copied), "output": str(output)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/original_wfo")
    args = parser.parse_args()
    export(args.output.resolve())
