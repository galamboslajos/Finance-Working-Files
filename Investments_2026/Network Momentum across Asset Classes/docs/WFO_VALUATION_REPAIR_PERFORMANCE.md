# Additional walk-forward performance diagnostics

For the current unbroken equity curve, see the [continuous accounting-scenario
report](CONTINUOUS_WFO_PERFORMANCE.md). This earlier report preserves the two
unobserved source marks and therefore still has two calculation gaps.

These charts describe repaired held valuations on the frozen original WFO. Original observed returns and models are unchanged.

Open the self-contained report at `data/walk_forward/valuation_performance_v2/performance_report.html`; the same folder contains a four-page `performance_charts.pdf`, PNG/SVG charts and CSV tables.

| Strategy | Annualized mean* | Annualized vol* | Sharpe* | Sortino* | Positive days* | Worst 5% mean* |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LongOnly | 6.69% | 8.62% | 0.776 | 1.076 | 54.61% | -1.31% |
| MACD | -0.48% | 3.11% | -0.153 | -0.202 | 52.84% | -0.51% |
| LinReg | 6.09% | 8.09% | 0.753 | 1.045 | 54.44% | -1.23% |
| GMOM | 5.84% | 7.94% | 0.736 | 1.019 | 54.38% | -1.22% |
| RegCombo | 5.69% | 7.79% | 0.731 | 1.014 | 54.21% | -1.19% |
| SignCombo | 5.97% | 8.00% | 0.745 | 1.033 | 54.32% | -1.22% |

The longest complete interval is **2019-01-02–2022-12-21**, 1002 sessions, selected by coverage only.

| Strategy | Return in longest complete interval | Max drawdown in that interval |
| --- | ---: | ---: |
| LongOnly | 27.96% | -16.40% |
| MACD | -3.57% | -6.09% |
| LinReg | 26.00% | -16.18% |
| GMOM | 24.88% | -16.05% |
| RegCombo | 25.09% | -15.93% |
| SignCombo | 25.44% | -16.12% |

There are 3 uninterrupted complete runs and 82/84 fully observed months. The equity and drawdown charts reset independently at each run, including initial capital in the drawdown high-water mark.

*Complete-date statistics describe 1,758 observed dates out of 1,760, conditional on availability. They are not a continuous seven-year performance record.

The 2 incomplete dates and all forecasts remain saved. No missing return is filled or reweighted away. Equity and drawdown restart at each uninterrupted run; cross-gap CAGR, total return, maximum drawdown, Calmar and recovery duration remain withheld.

Sortino = sqrt(252) × mean daily return / sqrt(mean(min(return, 0)²)); zero threshold. Sharpe uses sample daily volatility and zero risk-free rate. Annualized mean = 252 × daily arithmetic mean, not CAGR.

Positive-day fraction includes zero-return days in its denominator. Worst 5% mean averages the worst ceil(0.05 × observed days) returns. These are portfolio-day statistics, not trade win rates or trade profit factors.

Monthly compounded returns require every scheduled session to be observed. Rolling 63-session statistics require all 63 returns and stay unavailable across gaps. Intended exposures use all signal dates, including unavailable valuation dates.

LongOnly is an inverse-volatility-sized matched cohort, not the capital-weighted S&P 500. Trading costs, financing, borrow, executable fills and complete terminal/corporate-action cash flows are not included.

This report uses a retrospective realized-label overlay on frozen original forecasts. Sixty unavailable stock/date valuations were recovered: 30 SIP quotes, three predecessor-symbol quotes, and 27 fixed cash/share merger entitlements. Raw prices and original WFO outputs remain unchanged.

ABMD on 2022-12-22 and WBA on 2025-08-28 remain unavailable because their consideration includes non-tradeable contingent rights. Their face-value versus zero-mark sensitivity is saved separately; neither assumption is substituted into these primary returns.

Models, networks, scalers, hyperparameters, positions, cohorts, and original observed labels are unchanged. Original feature/eligibility gaps and training-label limitations are retained; this is an accounting correction of the saved experiment, not a refitted clean-data backtest.

Merger entitlements use contemporaneous raw USD acquirer closes and final closing terms. Cash/share receivables are marked without settlement friction. Source close-to-close entries may be stale or untradeable around completion; executable trading and subsequent contingent-right accounting remain unmodeled.

Three predecessor quotes were transcribed from source tables with matching controls. Unchanged adjustment-scale brackets on absent bars are a reviewed continuity assumption, not a separately complete corporate-action feed.

## Reproduce

```bash
python analyze_walk_forward.py --results data/walk_forward/valuation_results_v1 --verification data/walk_forward/valuation_verification_v1/report.json --output data/walk_forward/valuation_performance_v1
```

The analysis verifies the result/verification hashes first and refuses to overwrite an existing output directory. Use `--output` for a fresh version. Matplotlib is a reporting dependency; the core WFO evaluation is unchanged.
