#!/usr/bin/env python3
"""Additional WFO diagnostics; equity/drawdown never cross a missing valuation.

Creates CSV tables, PNG/SVG charts, a PDF and a self-contained HTML report from
the verified saved returns. No model fitting or missing-return imputation.
"""
from __future__ import annotations

import argparse
import base64
import html
import json
from pathlib import Path

import exchange_calendars as xcals
import numpy as np
import pandas as pd

from build_momentum_features import contiguous_slices, write_json
from evaluate_walk_forward import ARMS, performance
from prepare_sp500_daily import file_hash, require

ROOT = Path(__file__).resolve().parent
PRIMARY = ("GMOM", "LinReg", "LongOnly")
COLORS = {"GMOM": "#087f8c", "LinReg": "#de7851", "LongOnly": "#596480",
          "MACD": "#a89b4a", "RegCombo": "#9877ad", "SignCombo": "#5d9771"}


def extended_metrics(frame, arm):
    raw = frame[arm].to_numpy(float)
    r = raw[np.isfinite(raw)]
    out = performance(raw)
    if not len(r):
        return out
    downside = float(np.sqrt(np.mean(np.minimum(r, 0) ** 2)))
    positive, negative = r[r > 0], r[r < 0]
    tail_n = max(1, int(np.ceil(.05 * len(r))))
    observed = frame.loc[np.isfinite(raw)]
    out.update({
        "sortino_zero_threshold": float(np.sqrt(252) * r.mean() / downside) if downside else None,
        "positive_day_fraction": float(np.mean(r > 0)),
        "zero_day_fraction": float(np.mean(r == 0)),
        "average_positive_day": float(positive.mean()) if len(positive) else None,
        "average_negative_day": float(negative.mean()) if len(negative) else None,
        "gain_loss_size_ratio": float(positive.mean() / -negative.mean()) if len(positive) and len(negative) else None,
        "omega_zero_threshold": float(positive.sum() / -negative.sum()) if len(negative) else None,
        "daily_quantile_05": float(np.quantile(r, .05)),
        "mean_worst_5pct_days": float(np.sort(r)[:tail_n].mean()),
        "tail_observations": tail_n,
        "best_day": float(r.max()), "worst_day": float(r.min()),
        "best_return_date": str(observed.loc[observed[arm].idxmax(), "return_date"].date()),
        "worst_return_date": str(observed.loc[observed[arm].idxmin(), "return_date"].date()),
        "skewness": float(pd.Series(r).skew()) if len(r) >= 3 else None,
        "excess_kurtosis": float(pd.Series(r).kurt()) if len(r) >= 4 else None,
        # Positions are known even on dates with unavailable realized returns.
        "mean_intended_net_weight": float(frame[arm + "_net_weight"].mean()),
        "mean_intended_gross_weight": float(frame[arm + "_gross_weight"].mean()),
    })
    return out


def segment_curves(frame):
    """Reset wealth to 100 before each complete run, including initial capital in HWM."""
    rows, segments = [], []
    for segment_id, (start, stop) in enumerate(contiguous_slices(frame.complete), 1):
        block = frame.iloc[start:stop]
        segments.append({"segment": segment_id, "start": str(block.return_date.iloc[0].date()),
                         "end": str(block.return_date.iloc[-1].date()), "sessions": len(block),
                         "first_signal_date": str(block.date.iloc[0].date())})
        for arm in ARMS:
            r = block[arm].to_numpy(float)
            require(np.isfinite(r).all() and (r > -1).all(), "Invalid complete-run return")
            wealth = np.r_[100., 100. * np.exp(np.cumsum(np.log1p(r))) ]
            drawdown = wealth / np.maximum.accumulate(wealth) - 1
            dates = [block.date.iloc[0], *block.return_date.tolist()]
            for offset, (date, value, dd) in enumerate(zip(dates, wealth, drawdown)):
                rows.append({"segment": segment_id, "strategy": arm, "date": date,
                             "is_initial_capital": offset == 0,
                             "segment_equity": value, "segment_drawdown": dd})
    return pd.DataFrame(rows), pd.DataFrame(segments)


def monthly_returns(frame):
    rows = []
    for month, block in frame.groupby(frame.return_date.dt.to_period("M"), sort=True):
        complete = bool(block.complete.all())
        row = {"month": str(month), "year": month.year, "month_number": month.month,
               "expected_dates": len(block), "complete_dates": int(block.complete.sum()),
               "complete": complete}
        for arm in ARMS:
            row[arm] = float(np.expm1(np.log1p(block[arm]).sum())) if complete else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def rolling_statistics(frame, window=63):
    """A missing session invalidates every rolling window containing it."""
    out = frame[["return_date", "complete"]].copy()
    for arm in ARMS:
        r = frame[arm]
        mean, vol = r.rolling(window, min_periods=window).mean(), r.rolling(window, min_periods=window).std(ddof=1)
        out[arm + "_sharpe"] = np.sqrt(252) * mean / vol.where(vol > 0)
        out[arm + "_volatility"] = np.sqrt(252) * vol
    return out


def save_charts(frame, metrics, curves, segments, monthly, rolling, correlations, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.ticker import PercentFormatter

    plt.rcParams.update({"figure.facecolor": "#f7f8fa", "axes.facecolor": "white",
        "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "bold", "axes.labelcolor": "#424c5a", "text.color": "#202b39",
        "axes.edgecolor": "#cdd4df", "grid.alpha": .2, "savefig.facecolor": "#f7f8fa"})
    figures = []

    def finish(fig, name, title, subtitle):
        fig.suptitle(title, x=.08, y=.985, ha="left", fontsize=19, fontweight="bold")
        fig.text(.08, .945, subtitle, fontsize=10, color="#505d70")
        fig.text(.08, .018, "2019–2025 annual expanding WFO • zero-cost close-to-close research proxy • no executable-fill or terminal-cash accounting", fontsize=8, color="#667286")
        fig.subplots_adjust(top=.88, bottom=.09, left=.08, right=.97, hspace=.48, wspace=.3)
        for ax in fig.axes:
            if ax.get_label() != "<colorbar>":
                ax.grid(axis="y", alpha=.18)
        for extension in ("png", "svg"):
            fig.savefig(output / f"{name}.{extension}", dpi=170)
        figures.append((name, fig))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    years = sorted(frame.test_year.unique())
    x = np.arange(len(years))
    for i, arm in enumerate(PRIMARY):
        values = [metrics.loc[(metrics.period == str(y)) & (metrics.strategy == arm), "sharpe_complete_dates"].iloc[0] for y in years]
        axes[0, 0].bar(x + (i - 1) * .24, values, .24, color=COLORS[arm], label=arm)
    axes[0, 0].set(xticks=x, xticklabels=years, title="Annual Sharpe · complete dates", ylabel="Sharpe, zero risk-free rate")
    axes[0, 0].legend(frameon=False, fontsize=9)
    for arm in ("GMOM", "LinReg"):
        axes[0, 1].hist(100 * frame[arm].dropna(), bins=60, color=COLORS[arm], alpha=.5, label=arm)
    axes[0, 1].set(title="Daily return distribution", xlabel="Return (%)", ylabel="Complete dates")
    axes[0, 1].legend(frameon=False)
    grid = monthly.pivot(index="year", columns="month_number", values="GMOM").reindex(index=years, columns=range(1, 13))
    limit = max(.01, float(np.nanmax(np.abs(grid.to_numpy()))))
    cmap = plt.get_cmap("RdYlGn").copy(); cmap.set_bad("#dce1e8")
    image = axes[1, 0].imshow(grid, aspect="auto", cmap=cmap, vmin=-limit, vmax=limit)
    axes[1, 0].set(title="GMOM monthly return · grey = unavailable", xticks=range(12), xticklabels=list("JFMAMJJASOND"), yticks=range(len(years)), yticklabels=years)
    for i in range(len(years)):
        for j in range(12):
            value = grid.iloc[i, j]
            axes[1, 0].text(j, i, "—" if pd.isna(value) else f"{100*value:.1f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=axes[1, 0], format=PercentFormatter(1), fraction=.03, pad=.03)
    image = axes[1, 1].imshow(correlations, cmap="coolwarm", vmin=-1, vmax=1)
    axes[1, 1].set(title="Daily return correlations · complete dates", xticks=range(len(ARMS)), xticklabels=ARMS, yticks=range(len(ARMS)), yticklabels=ARMS)
    axes[1, 1].tick_params(axis="x", labelrotation=30)
    for i in range(len(ARMS)):
        for j in range(len(ARMS)):
            axes[1, 1].text(j, i, f"{correlations.iloc[i,j]:.2f}", ha="center", va="center", fontsize=8)
    finish(fig, "performance_overview", "Walk-forward performance diagnostics", "1,722 / 1,760 complete dates. Monthly returns require every session; missing months stay unavailable.")

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    for arm in PRIMARY:
        for i, (_, block) in enumerate(curves.loc[curves.strategy.eq(arm)].groupby("segment")):
            axes[0].plot(block.date, block.segment_equity, color=COLORS[arm], lw=1.15, label=arm if i == 0 else None)
            axes[1].plot(block.date, block.segment_drawdown, color=COLORS[arm], lw=1.15)
    for date in frame.loc[~frame.complete, "return_date"]:
        for ax in axes[:2]:
            ax.axvline(date, color="#b55151", alpha=.15, lw=1)
    axes[0].axhline(100, color="#8693a7", lw=.6)
    axes[0].set(title="Equity within each uninterrupted run · resets to 100 after every gap", ylabel="Segment equity")
    axes[0].legend(frameon=False, ncols=3)
    axes[1].set(title="Drawdown from the high-water mark within each run", ylabel="Segment drawdown")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[2].plot(frame.return_date, frame.nodes, color="#8e9aaf", lw=.9, label="Forecast cohort")
    axes[2].scatter(frame.loc[~frame.complete, "return_date"], frame.loc[~frame.complete, "nodes"], marker="x", color="#b55151", s=25, label="Unavailable valuation")
    axes[2].set(title="Forecast cohort and unavailable portfolio dates", ylabel="Stocks", xlabel="Return realization date")
    axes[2].legend(frameon=False)
    finish(fig, "equity_drawdown_segments", "Equity and drawdown with gaps preserved", f"{len(segments)} independent complete runs. Lines stop at gaps; a new run has a new 100 baseline. These are not continuous seven-year wealth paths.")

    longest = segments.sort_values(["sessions", "start"], ascending=[False, True]).iloc[0]
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    for arm in PRIMARY:
        block = curves.loc[curves.segment.eq(longest.segment) & curves.strategy.eq(arm)]
        axes[0].plot(block.date, block.segment_equity, color=COLORS[arm], lw=2, label=arm)
        axes[1].plot(block.date, block.segment_drawdown, color=COLORS[arm], lw=1.6)
    axes[0].set(title="Equity · initial capital = 100", ylabel="Equity")
    axes[0].legend(frameon=False, ncols=3)
    axes[1].set(title="Drawdown · initial capital included in high-water mark", ylabel="Drawdown", xlabel="Return realization date")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    finish(fig, "longest_complete_period", "Equity and drawdown · longest complete interval", f"{longest.start} to {longest.end} · {int(longest.sessions)} sessions · selected only by data continuity, not performance. Short-period results are not full-WFO results.")

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    for arm in PRIMARY:
        axes[0].plot(rolling.return_date, rolling[arm + "_sharpe"], color=COLORS[arm], lw=1.1, label=arm)
        axes[1].plot(rolling.return_date, rolling[arm + "_volatility"], color=COLORS[arm], lw=1.1)
    axes[0].set(title="Rolling 63-session Sharpe · full 63 observations required", ylabel="Sharpe")
    axes[0].legend(frameon=False, ncols=3)
    axes[1].set(title="Rolling 63-session annualized volatility", ylabel="Volatility")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[2].plot(frame.return_date, frame.GMOM_gross_weight, color=COLORS["GMOM"], label="GMOM gross", lw=1.2)
    axes[2].plot(frame.return_date, frame.GMOM_net_weight, color=COLORS["LinReg"], label="GMOM net", lw=1.2)
    axes[2].set(title="Intended exposures · retained even when a return is unavailable", ylabel="Portfolio weight", xlabel="Return realization date")
    axes[2].yaxis.set_major_formatter(PercentFormatter(1)); axes[2].legend(frameon=False, ncols=2)
    finish(fig, "rolling_risk_and_exposure", "Rolling risk and portfolio exposures", "Rolling statistics do not skip missing dates. Ratios use a zero risk-free / target-return threshold; exposures are signal-date intended weights.")
    with PdfPages(output / "performance_charts.pdf") as pdf:
        for _, fig in figures:
            pdf.savefig(fig)
    for _, fig in figures:
        plt.close(fig)
    return [name for name, _ in figures]


def render_report(metrics, longest_metrics, segments, monthly, charts, output):
    overall = metrics.loc[metrics.period.eq("all")].set_index("strategy")
    summary = overall[["annualized_mean", "annualized_volatility", "sharpe_complete_dates", "sortino_zero_threshold",
                       "positive_day_fraction", "mean_worst_5pct_days", "worst_day", "best_day"]].copy()
    labels = ["Annualized mean*", "Annualized vol*", "Sharpe*", "Sortino*", "Positive days*", "Worst 5% mean*", "Worst day*", "Best day*"]
    percent_columns = [0, 1, 4, 5, 6, 7]
    text = summary.astype(object)
    for j, col in enumerate(summary.columns):
        text[col] = summary[col].map(lambda x: f"{100*x:.2f}%" if j in percent_columns else f"{x:.3f}")
    text.columns = labels
    longest = segments.sort_values(["sessions", "start"], ascending=[False, True]).iloc[0]
    lp = longest_metrics.set_index("strategy")[["total_return", "maximum_drawdown", "positive_day_fraction"]].map(lambda x: f"{100*x:.2f}%")
    lp.columns = ["Interval return", "Interval max drawdown", "Positive days"]
    notes = [
        "*Complete-date statistics describe 1,722 observed dates out of 1,760, conditional on availability. They are not a continuous seven-year performance record.",
        "The 38 incomplete dates and all forecasts remain saved. No missing return is filled or reweighted away. Equity and drawdown restart at each uninterrupted run; cross-gap CAGR, total return, maximum drawdown, Calmar and recovery duration remain withheld.",
        "Sortino = sqrt(252) × mean daily return / sqrt(mean(min(return, 0)²)); zero threshold. Sharpe uses sample daily volatility and zero risk-free rate. Annualized mean = 252 × daily arithmetic mean, not CAGR.",
        "Positive-day fraction includes zero-return days in its denominator. Worst 5% mean averages the worst ceil(0.05 × observed days) returns. These are portfolio-day statistics, not trade win rates or trade profit factors.",
        "Monthly compounded returns require every scheduled session to be observed. Rolling 63-session statistics require all 63 returns and stay unavailable across gaps. Intended exposures use all signal dates, including unavailable valuation dates.",
        "LongOnly is an inverse-volatility-sized matched cohort, not the capital-weighted S&P 500. Trading costs, financing, borrow, executable fills and complete terminal/corporate-action cash flows are not included.",
        "No models, hyperparameters, membership rules, prices or saved returns were changed for these diagnostics. No model selection uses these test results.",
    ]
    body = "<h1>Walk-forward performance</h1><p class='lead'>S&P 500 network momentum · annual expanding WFO · 2019–2025</p>"
    body += "<div class='warning'><strong>Research proxy with 38 unavailable portfolio dates.</strong> Equity and drawdown stop at each gap and restart at 100. A continuous seven-year equity curve and full-period maximum drawdown remain unavailable.</div>"
    body += "<h2>Daily performance statistics</h2>" + text.to_html(border=0)
    for name in charts:
        encoded = base64.b64encode((output / f"{name}.png").read_bytes()).decode()
        body += f'<img alt="{html.escape(name.replace("_", " "))}" src="data:image/png;base64,{encoded}">'
        if name == "longest_complete_period":
            body += f"<h2>Longest complete interval: {longest.start} to {longest.end}</h2><p>{int(longest.sessions)} sessions, chosen by continuity only. This interval is a small part of the WFO.</p>" + lp.to_html(border=0)
    body += "<h2>Definitions and interpretation</h2><ul>" + "".join(f"<li>{html.escape(note)}</li>" for note in notes) + "</ul>"
    document = "<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'><title>Network Momentum WFO performance</title><style>body{font:16px/1.55 system-ui,sans-serif;color:#202b39;background:#f7f8fa;max-width:1300px;margin:40px auto;padding:0 28px}h1{font-size:38px;margin-bottom:4px}h2{margin-top:30px}.lead{color:#596480}.warning{background:#fff3d8;padding:18px;border-left:4px solid #ba7c10;margin:24px 0}table{width:100%;border-collapse:collapse;background:white;font-size:13px}th,td{text-align:right;padding:10px;border-bottom:1px solid #e5e8ef}th:first-child{text-align:left}img{width:100%;height:auto;margin:28px 0 5px}li{margin:12px 0}@media(max-width:800px){body{padding:0 10px}table{font-size:10px}th,td{padding:5px}}</style><main>" + body + "</main></html>"
    (output / "performance_report.html").write_text(document)
    lines = ["# Additional walk-forward performance diagnostics", "", "These charts extend the completed WFO without changing its models or returns.", "",
        "Open the self-contained report at `data/walk_forward/performance_v1/performance_report.html`; the same folder contains a four-page `performance_charts.pdf`, PNG/SVG charts and CSV tables.", "",
        "| Strategy | Annualized mean* | Annualized vol* | Sharpe* | Sortino* | Positive days* | Worst 5% mean* |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for arm in ARMS:
        row = text.loc[arm]
        lines.append("| " + " | ".join([arm, *row.iloc[:6].tolist()]) + " |")
    lines += ["", f"The longest complete interval is **{longest.start}–{longest.end}**, {int(longest.sessions)} sessions, selected by coverage only.", "",
        "| Strategy | Return in longest complete interval | Max drawdown in that interval |", "| --- | ---: | ---: |"]
    for arm in ARMS:
        lines.append(f"| {arm} | {lp.loc[arm].iloc[0]} | {lp.loc[arm].iloc[1]} |")
    lines += ["", f"There are {len(segments)} uninterrupted complete runs and {int(monthly.complete.sum())}/{len(monthly)} fully observed months. "
        "The equity and drawdown charts reset independently at each run, including initial capital in the drawdown high-water mark.", "",
        *[note + "\n" for note in notes], "## Reproduce", "", "```bash", "python analyze_walk_forward.py", "```", "",
        "The analysis verifies the result/verification hashes first and refuses to overwrite an existing output directory. "
        "Use `--output` for a fresh version. Matplotlib is a reporting dependency; the core WFO evaluation is unchanged.", ""]
    (ROOT / "docs/ANNUAL_WALK_FORWARD_PERFORMANCE.md").write_text("\n".join(lines))


def analyze(results, verification, output):
    require(not output.exists(), "Performance report exists; choose a new version")
    manifest_path = results / "manifest.json"
    manifest, checks = json.loads(manifest_path.read_text()), json.loads(verification.read_text())
    require(manifest["status"] == "complete" and checks["status"] == "passed", "Verified completed results required")
    require(file_hash(manifest_path) == checks["result_manifest_sha256"], "Verification belongs to different results")
    objects = {item["path"]: item for item in manifest["outputs"]}
    daily_path = results / "daily_paper_proxy_returns.csv"
    require(file_hash(daily_path) == objects[daily_path.name]["sha256"], "Saved daily returns changed")
    frame = pd.read_csv(daily_path, parse_dates=["date", "return_date"]).sort_values("return_date").reset_index(drop=True)
    require(frame.complete.dtype == bool, "Unexpected complete-date mask")
    require(not frame.return_date.duplicated().any(), "Duplicate return dates")
    sessions = xcals.get_calendar("XNYS", start=frame.return_date.min(), end=frame.return_date.max()).sessions.tz_localize(None)
    require(pd.DatetimeIndex(frame.return_date).equals(sessions), "Missing scheduled dates")
    require(np.array_equal(frame[list(ARMS)].notna().to_numpy(), np.repeat(frame.complete.to_numpy()[:, None], len(ARMS), axis=1)), "Strategy availability differs from matched cohort")
    require(np.isfinite(frame.loc[frame.complete, list(ARMS)]).all().all(), "Nonfinite return")
    output.mkdir(parents=True)
    metrics = pd.DataFrame([{"period": period, "strategy": arm, **extended_metrics(block, arm)}
        for period, block in [("all", frame), *[(str(year), block) for year, block in frame.groupby("test_year")]] for arm in ARMS])
    curves, segments = segment_curves(frame)
    longest = segments.sort_values(["sessions", "start"], ascending=[False, True]).iloc[0]
    interval = frame.loc[frame.return_date.between(longest.start, longest.end)]
    longest_metrics = pd.DataFrame([{"strategy": arm, "start": longest.start, "end": longest.end,
        **extended_metrics(interval, arm), "total_return": float(np.expm1(np.log1p(interval[arm]).sum()))} for arm in ARMS])
    monthly, rolling = monthly_returns(frame), rolling_statistics(frame)
    correlations = frame.loc[frame.complete, list(ARMS)].corr()
    for name, table in [("extended_metrics", metrics), ("segment_curves", curves), ("complete_segments", segments),
                        ("longest_complete_interval_metrics", longest_metrics), ("monthly_returns", monthly), ("rolling_63_sessions", rolling)]:
        table.to_csv(output / f"{name}.csv", index=False)
    correlations.to_csv(output / "return_correlations.csv")
    charts = save_charts(frame, metrics, curves, segments, monthly, rolling, correlations, output)
    render_report(metrics, longest_metrics, segments, monthly, charts, output)
    require(file_hash(manifest_path) == checks["result_manifest_sha256"] and file_hash(daily_path) == objects[daily_path.name]["sha256"], "Original results changed during reporting")
    record = {"status": "complete", "purpose": "Additional descriptive diagnostics; no model or strategy changes",
        "results_manifest_sha256": file_hash(manifest_path), "daily_returns_sha256": file_hash(daily_path),
        "verification_sha256": file_hash(verification), "script_sha256": file_hash(__file__),
        "sessions": len(frame), "complete_sessions": int(frame.complete.sum()), "unavailable_sessions": int((~frame.complete).sum()),
        "complete_runs": len(segments), "complete_months": int(monthly.complete.sum()), "months": len(monthly),
        "longest_run": longest.to_dict(), "equity_across_missing_dates": False,
        "full_period_cagr_and_drawdown": "withheld", "rolling_window_sessions": 63,
        "outputs": [{"path": p.name, "sha256": file_hash(p), "size_bytes": p.stat().st_size} for p in sorted(output.iterdir()) if p.is_file()]}
    write_json(output / "manifest.json", record)
    print(json.dumps({k: v for k, v in record.items() if k != "outputs"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "data/walk_forward/annual_results_v1")
    parser.add_argument("--verification", type=Path, default=ROOT / "data/walk_forward/verification_v1/report.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/walk_forward/performance_v1")
    args = parser.parse_args()
    analyze(args.results.resolve(), args.verification.resolve(), args.output.resolve())
