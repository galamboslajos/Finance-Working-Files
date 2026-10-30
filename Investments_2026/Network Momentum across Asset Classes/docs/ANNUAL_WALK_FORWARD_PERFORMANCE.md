# Additional walk-forward performance diagnostics

The [published review outputs](../outputs/original_wfo/README.md) include the
[chart PDF](../outputs/original_wfo/performance/performance_charts.pdf),
[HTML report](../outputs/original_wfo/performance/performance_report.html)
and source tables for these diagnostics.

These charts extend the completed WFO without changing its models or returns.

Open the self-contained report at `data/walk_forward/performance_v1/performance_report.html`; the same folder contains a four-page `performance_charts.pdf`, PNG/SVG charts and CSV tables.

| Strategy | Annualized mean* | Annualized vol* | Sharpe* | Sortino* | Positive days* | Worst 5% mean* |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LongOnly | 6.79% | 8.65% | 0.786 | 1.091 | 54.70% | -1.30% |
| MACD | -0.66% | 3.11% | -0.213 | -0.281 | 52.56% | -0.51% |
| LinReg | 6.27% | 8.11% | 0.774 | 1.075 | 54.53% | -1.23% |
| GMOM | 6.03% | 7.97% | 0.757 | 1.051 | 54.53% | -1.21% |
| RegCombo | 5.90% | 7.82% | 0.755 | 1.049 | 54.30% | -1.19% |
| SignCombo | 6.15% | 8.03% | 0.767 | 1.065 | 54.41% | -1.22% |

The longest complete interval is **2024-05-06–2024-11-21**, 140 sessions, selected by coverage only.

| Strategy | Return in longest complete interval | Max drawdown in that interval |
| --- | ---: | ---: |
| LongOnly | 9.36% | -2.73% |
| MACD | 2.57% | -0.89% |
| LinReg | 8.12% | -2.52% |
| GMOM | 7.03% | -2.58% |
| RegCombo | 6.40% | -2.58% |
| SignCombo | 7.57% | -2.55% |

There are 37 uninterrupted complete runs and 53/84 fully observed months. The equity and drawdown charts reset independently at each run, including initial capital in the drawdown high-water mark.

*Complete-date statistics describe 1,722 observed dates out of 1,760, conditional on availability. They are not a continuous seven-year performance record.

The 38 incomplete dates and all forecasts remain saved. No missing return is filled or reweighted away. Equity and drawdown restart at each uninterrupted run; cross-gap CAGR, total return, maximum drawdown, Calmar and recovery duration remain withheld.

Sortino = sqrt(252) × mean daily return / sqrt(mean(min(return, 0)²)); zero threshold. Sharpe uses sample daily volatility and zero risk-free rate. Annualized mean = 252 × daily arithmetic mean, not CAGR.

Positive-day fraction includes zero-return days in its denominator. Worst 5% mean averages the worst ceil(0.05 × observed days) returns. These are portfolio-day statistics, not trade win rates or trade profit factors.

Monthly compounded returns require every scheduled session to be observed. Rolling 63-session statistics require all 63 returns and stay unavailable across gaps. Intended exposures use all signal dates, including unavailable valuation dates.

LongOnly is an inverse-volatility-sized matched cohort, not the capital-weighted S&P 500. Trading costs, financing, borrow, executable fills and complete terminal/corporate-action cash flows are not included.

No models, hyperparameters, membership rules, prices or saved returns were changed for these diagnostics. No model selection uses these test results.

## Reproduce

```bash
python analyze_walk_forward.py
```

The analysis verifies the result/verification hashes first and refuses to overwrite an existing output directory. Use `--output` for a fresh version. Matplotlib is a reporting dependency; the core WFO evaluation is unchanged.
