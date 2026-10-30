# Annual expanding walk-forward experiment

The user selected this adaptation on **30 October 2026**: retain all five
network lookbacks, use three initial OLS training years, and recalibrate each
year with an expanding training sample. The paper declares an initial ten-year
pre-test training span and five-year recalibration schedule. It does not fully
document how initial feature/graph warmup affects the number of usable OLS
observations inside that span. This experiment specifies those stages and
counts separately rather than assuming ten complete years of network inputs.

| Pre-test OLS training sample | Unseen return year |
| --- | --- |
| 2016–2018 | 2019 |
| 2016–2019 | 2020 |
| 2016–2020 | 2021 |
| 2016–2021 | 2022 |
| 2016–2022 | 2023 |
| 2016–2023 | 2024 |
| 2016–2024 | 2025 |

The exact XNYS dates, label cutoffs and row counts are saved in
`data/walk_forward/annual_expanding_v2/folds.csv`. The policy is defined by
`walk_forward.PROTOCOL` and saved beside that table. The previous v1 plan
indexed test years by signal date; v2 fixes them by return realization date
before any out-of-sample results are evaluated.

## What is fitted, and when

On every signal date, calculate the eight own-stock features from admitted
closing prices, learn the five networks from the trailing 252, 504, 756, 1,008
and 1,260 feature sessions, average raw adjacency matrices, normalize, and
propagate that day's features. Current dated membership and complete historical
features select the nodes. Pre-entry history is allowed; future membership,
future prices and future target availability cannot select a node.

The OLS target is the following exchange-session return divided by the
signal-date daily volatility. In each annual fold:

1. Reserve the latest **10% of pre-test exchange dates**, rounding upward, for
   validation. All stocks of a date stay together.
2. Fit the scaler and pooled OLS on earlier dates. Exclude any label not known
   by the fitting cutoff. In particular, the last fitting signal's following
   return cannot cross into the validation interval.
3. Select the shared graph parameters using validation Sharpe of the paper's
   raw, zero-cost close-to-close portfolio proxy. Validation labels must have
   matured by the final training-year session. Preserve validation MSE as a
   diagnostic; it is not the selection score.
4. Refit the selected scaler and OLS using all eligible pre-test labeled rows
   known by the final training-year close.
5. Freeze the selected graph parameters, scaler and OLS coefficients throughout
   the next return year. Membership, own features and networks continue updating
   each day. Recalibrate again at the following year-end.

For the first fold, the pre-test sample contains **754 signal dates**.
The earlier fitting interval is 2016-01-04 through 2018-09-11; validation starts
2018-09-12. At most 677 fit dates and 75 validation dates have labels completed
within their respective cutoffs. After selection, the final refit can use at
most 753 pre-test labeled dates, before stock-specific missing labels.

The first 2019 return is realized on 2019-01-02 from the 2018-12-31 close.
Its forecast is generated after the year-end refit, using only information
available by that close; its return is excluded from fitting. Keeping both
`date` (signal) and `label_date`/`return_date` (realization) avoids skipping the
first return of a test year or accidentally training on it.

## Full parameter grid with equivalent calculations grouped

The paper's alpha and beta grids each have 11 values, producing 121 pairs.
Under the chosen shared-pair ensemble convention, they produce **30 distinct
normalized network candidates**. No grid candidates are discarded.

For undirected edge vector `w`, distances `z`, and degree operator `S`, substitute
`w=alpha*v` into the objective:

```text
[z'w + 2*beta*w'w - alpha*sum(log(Sw))] / alpha
  = z'v + 2*alpha*beta*v'v - sum(log(Sv)) - N*log(alpha)
```

The final term is constant in `v`, so
`A(alpha,beta) = alpha*A(1,alpha*beta)`. The same alpha scales all five raw
adjacency matrices. Averaging and symmetric degree normalization cancel that
common scale. Consequently, propagated features depend only on `alpha*beta`.
Decimal arithmetic groups the original pairs exactly, rather than rounding
similar products together. A numerical unit test independently verifies this
identity across an ensemble.

Each distinct product receives its own causal historical network-feature
series and validation score. Equivalent original pairs retain their aliases.
Score ties resolve by smallest alpha, then smallest beta; test performance
never breaks a tie. This reduction applies to this shared-pair convention,
not arbitrary different parameter pairs for each lookback.

## Numerical implementation

`graph_grid.py` solves the original objective. With very small alpha*beta,
recovering edges by subtracting dual variables and distances can lose useful
digits. A scaled primal active-set optimizer with Newton polishing avoids that
subtraction. It checks every edge's KKT conditions and a primal-dual objective
bound. The internal change of optimization units does not normalize graph
distances or change their objective. Moderate parameter values use the verified
convex dual solver. There is no synthetic edge, degree floor or edge cutoff.

Daily squared distances update by adding the newest eight-feature contribution
and removing the oldest. Node entries receive direct distances. Every 21
sessions, and after a discontinuity, a direct rebase controls floating-point
drift. Direct checks run on selected pairs every date. Warm starts affect
numerical initialization; all results must satisfy the frozen optimality
tolerances.

Independent year processes use the same immutable source pins and parameter
grid. Daily arrays, full original-cohort eligibility ledgers, solver
certificates and hashes are checkpointed before a date is marked complete.
After interruption, complete checkpoints are verified and reused; the first
new date can start cold. Annual tables make subsequent OLS fitting efficient.

## Data availability and interpretation

`extend_momentum_history.py` extends features through **2026-01-02**, using
the same 2009 initialization, 107 exact reviewed-close keys, and original
quality gate on later observations. It introduces **no new close reviews**.
All 1,082 baseline native endpoints match exactly. Earlier partitions remain
unchanged hash-pinned references, and indicators reset after unavailable
closes as in the first pilot.

`audit_wfo_availability.py` preserves each original member/date and measures
eligibility before fitting any models. Across 2016–2025 the common graph cohort
ranges from **426 to 467 securities**. Annual eligible node-days cover roughly
85.5–88.8% of original member-days. The 1,260-session complete-feature rule
therefore remains a meaningful coverage restriction even with PIT membership.

Whole-cohort next-session returns are available on approximately 96.4–98.8%
of signal dates in each year. A missing target never removes a forecast or
shrinks its portfolio denominator. If any node lacks a following return, the
entire common-cohort portfolio date is marked unavailable. Metrics identify
their complete-date coverage; continuous CAGR and drawdown are withheld across
gaps. This prevents a missing holding valuation from becoming an assumed zero
return. It does not establish unbiasedness of the observed sample.

The evaluator compares **LongOnly, MACD, LinReg, GMOM, RegCombo and SignCombo**
on the same dated graph cohort. LongOnly uses the paper's inverse-volatility
risk sizing and is not a market-cap-weighted S&P 500 benchmark. GMOM is compared
primarily with own-stock LinReg. Net and gross risk weights are retained.

The initial results are explicitly the paper's **zero-cost close-to-close
research proxy**. They do not certify final-close fills, financing, borrow,
portfolio volatility targeting, terminal distributions, or a complete path
through missing valuations. Source identity and provisional closing-price
limitations remain. None of those limits is repaired by a walk-forward split.

## Reproduction

Run with the existing Deep Momentum V2 Python environment, or install the
project requirements. Use new output versions when repeating completed builds.

```bash
python walk_forward.py
python extend_momentum_history.py --end 2026-01-02 --output data/momentum_history/wfo_2016_2025_v1
python audit_wfo_availability.py
python run_daily_years.py --workers 3
python evaluate_walk_forward.py
python -m unittest discover -s tests -v
```

The January-only extension and the six-date engineering sample remain separate
from the full WFO inputs. The engineering sample checks solver behavior and
date handling; it cannot supply regression or performance conclusions.

The main output locations are:

| Location | Contents |
| --- | --- |
| `data/walk_forward/annual_expanding_v2/` | Frozen protocol, seven folds, and all original parameter aliases. |
| `data/momentum_history/wfo_2016_2025_v1/` | Feature extension and hash-pinned references to earlier history. |
| `data/wfo_availability/v1/` | Daily/annual coverage and unresolved future-label keys. |
| `data/daily_networks/wfo_grid_by_year_v1/` | Complete year batches, daily checkpoints, eligibility ledgers, 30 network candidates and numerical certificates. |
| `data/walk_forward/annual_results_v1/` | Validation scores, selected parameters, frozen annual models, all OOS forecasts, daily research-proxy returns and coverage-aware metrics. |

The full experiment is complete. Read the [verified results](ANNUAL_WALK_FORWARD_RESULTS.md)
for the annual comparisons, selected parameters and remaining valuation limits.
