# Annual expanding walk-forward results

The [published review outputs](../outputs/original_wfo/README.md) include the
charts, daily returns, fitted models and verification reports referenced below.

The selected experiment is complete: three initial OLS training years (2016–2018), expanding annual refits, and seven unseen return years (2019–2025). All five daily graph lookbacks are retained. These results use the paper's zero-cost close-to-close research proxy, without an additional portfolio-volatility target.

The run preserves 771,773 out-of-sample stock/date forecasts over 1,760 return dates. The original 121 graph-parameter pairs are all represented by 30 mathematically equivalent-product groups. Each annual selection uses past validation labels only.

**Coverage limits the return interpretation:** 38 portfolio dates have at least one unavailable next-session return. Their forecasts remain saved, but the entire matched-cohort date is unavailable for return reporting. Sharpe and moments below describe complete dates. Continuous CAGR and drawdown are withheld across gaps; no missing return is treated as zero.

## Matched-cohort research-proxy comparison

| Strategy | Annualized mean, complete dates | Annualized volatility, complete dates | Sharpe, complete dates |
| --- | ---: | ---: | ---: |
| LongOnly | 6.79% | 8.65% | 0.786 |
| MACD | -0.66% | 3.11% | -0.213 |
| LinReg | 6.27% | 8.11% | 0.774 |
| GMOM | 6.03% | 7.97% | 0.757 |
| RegCombo | 5.90% | 7.82% | 0.755 |
| SignCombo | 6.15% | 8.03% | 0.767 |

GMOM's complete-date Sharpe is **0.757**, lower than matched own-stock LinReg's **0.774**. This comparison does not establish executable profits or causal information transmission. The small difference and unresolved valuation dates warrant cautious interpretation. LongOnly is the paper's inverse-volatility-sized cohort, not a market-cap-weighted S&P 500 total-return index.

## Annual unseen blocks

| Return year | OLS history | GMOM Sharpe | LinReg Sharpe | Complete portfolio dates |
| --- | --- | ---: | ---: | ---: |
| 2019 | 2016–2018 | 2.103 | 2.109 | 246/252 |
| 2020 | 2016–2019 | 0.205 | 0.234 | 250/253 |
| 2021 | 2016–2020 | 1.672 | 1.665 | 246/252 |
| 2022 | 2016–2021 | -0.283 | -0.283 | 244/251 |
| 2023 | 2016–2022 | 1.104 | 0.972 | 244/250 |
| 2024 | 2016–2023 | 1.290 | 1.480 | 249/252 |
| 2025 | 2016–2024 | 0.326 | 0.376 | 243/250 |

Test years use the return realization date. For example, the first 2019 return is realized on 2019-01-02 from a forecast after the 2018-12-31 refit. That following return cannot enter fitting or validation. Models and scalers remain fixed within each return year, while networks and dated membership update daily.

## Graph parameter selection

| Test year | Graph alpha | Graph beta | Product | Validation Sharpe | Candidate groups within 1e-12 of best |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2019 | 0.0001 | 0.0001 | 1e-08 | -2.487 | 29/30 |
| 2020 | 5 | 5 | 25 | 2.298 | 1/30 |
| 2021 | 0.1 | 10 | 1 | 2.626 | 2/30 |
| 2022 | 10 | 10 | 100 | 1.270 | 1/30 |
| 2023 | 0.0001 | 0.0001 | 1e-08 | -0.282 | 26/30 |
| 2024 | 0.05 | 10 | 0.5 | 1.542 | 2/30 |
| 2025 | 10 | 10 | 100 | 1.362 | 1/30 |

A flat validation score across candidates is not evidence of a uniquely identified optimum. Many distinct small penalties can produce the same signed validation positions. Exact score ties follow the predeclared smallest-alpha, then smallest-beta rule. No test score enters selection.

## Forecast diagnostics

| Forecast | Volatility-scaled target MSE | Labeled stock/date observations |
| --- | ---: | ---: |
| ZeroForecast | 1.143111 | 771,711 |
| LinReg | 1.142705 | 771,711 |
| GMOM | 1.142857 | 771,711 |
| RegCombo | 1.142839 | 771,711 |

GMOM/LinReg complete-date return correlation is 0.995. GMOM's mean intended net weight is 0.489, and mean gross weight is 0.564. The method does not enforce market neutrality. Forecast MSE uses only observed labels; missing-label forecasts remain in the prediction files.

## Verification

The independent verifier checked 21 fitted OLS models using centered normal equations, separately from the evaluator's least-squares routine. It replayed every saved forecast and all strategy returns/exposures, verified parameter selection, label cutoffs and unchanged first-fold models frozen before OOS evaluation. Maximum standardized coefficient discrepancy was 4.48e-14; maximum prediction discrepancy was 5.55e-17.

Structural tests cover feature formulas, missing-session resets, future-prefix invariance, source/member preservation, graph objective/normalization, exact grid grouping, chronological fitting boundaries, training-only standardization, missing-label preservation and withholding continuous wealth metrics across gaps.

## Additional performance diagnostics

The [extended performance report](ANNUAL_WALK_FORWARD_PERFORMANCE.md) adds Sortino, positive-day frequency, tail losses, monthly returns, correlations, rolling risk and exposures. Its equity and drawdown charts reset within each uninterrupted complete-data run. The self-contained HTML, PDF charts and CSV tables are saved in `data/walk_forward/performance_v1/`.

## Practical limits and saved artifacts

The calculation does not include executable fills, trading costs, borrow/financing, complete corporate-action/terminal distributions or a solution for missing held valuations. The 107 close reviews remain provisional, and historical identity/price-vintage limitations remain. PIT membership and WFO prevent specific forms of lookahead; they do not certify complete or unbiased economic returns.

| Artifact | Location |
| --- | --- |
| Frozen chronology and protocol | `data/walk_forward/annual_expanding_v2/` |
| First-fold models frozen before OOS evaluation | `data/walk_forward/first_fold_frozen_v1/` |
| All validation scores and selected graph parameters | `data/walk_forward/annual_results_v1/validation_scores.csv`, `selected_parameters.csv` |
| Annual scalers and OLS coefficients | `data/walk_forward/annual_results_v1/models_YYYY.json` |
| Every OOS forecast, including missing labels | `data/walk_forward/annual_results_v1/predictions/` |
| Daily returns, date coverage and net/gross weights | `data/walk_forward/annual_results_v1/daily_paper_proxy_returns.csv` |
| Annual/aggregate research-proxy metrics | `data/walk_forward/annual_results_v1/paper_proxy_metrics.csv` |
| Independent numerical verification | `data/walk_forward/verification_v1/report.json` |
| Source/policy details and reproduction | [Annual WFO protocol](ANNUAL_WALK_FORWARD.md) |
