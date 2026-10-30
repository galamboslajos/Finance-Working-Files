# Network Momentum methodology and S&P 500 replication

The proposed equity study asks whether momentum from a stock's learned peers
improves next-day forecasts and portfolio performance beyond that stock's own
momentum. It preserves the paper's feature, graph, regression and trading
architecture while replacing its cross-asset futures universe with historical
S&P 500 constituents. This is an application of the methodology to a different
universe; the paper's performance figures are reference results, not targets for
the equity study.

Source: Xingyue (Stacy) Pu, Stephen Roberts, Xiaowen Dong and Stefan Zohren,
*Network Momentum across Asset Classes*, supplied August 2023 version, 32 pages.
Page references below use the PDF's printed page numbers, which match its page
positions. Original PDF:
[source paper](</Users/lajosgalambos/Desktop/Network Momentum across Asset Classes.pdf>).
SHA256: `a375a7370900dc485ce4793aa68ef6f4c34f2f7d2fb44658a1b28d7edba2e245`.

Paper settings are also recorded in
[network_momentum_paper.json](../configs/network_momentum_paper.json).
Statements under proposed equity rules and open choices describe this project's
adaptation, rather than parameters established by the authors.

## What the strategy learns

Each asset is a node. The model learns an undirected network from similarities
in historical momentum features. It then aggregates the current momentum of
each node's neighbors, using learned edge weights, and predicts that node's
next-day volatility-scaled return with a single pooled linear regression.

The graph is learned without future return labels. Future returns enter the
regression and the validation used to choose graph hyperparameters. The network
itself is symmetric, with no explicit estimated direction or lag on an edge.
Predictive performance therefore tests whether this aggregation is useful; it
does not by itself establish a causal direction of information transmission.

The original universe is 64 ratio-adjusted continuous futures contracts spanning
commodities, equity indices, fixed income and currencies. The paper's equity
nodes are index futures, rather than constituent stocks (pp. 7 and 31).

## Individual momentum features

Let `p[i,t]` be an asset's price, `r[i,t]` its daily return, and `s[i,t]` its
estimated daily volatility. All inputs at a signal date must be available by
that date.

### Five return features

For horizons `h = 1, 21, 63, 126, 252` trading sessions, the paper uses
(p. 8):

```text
return_feature[i,t,h] = cumulative_return[i,t-h:t] / (s[i,t] * sqrt(h))
```

Daily volatility uses exponentially weighted standard deviation with a 60-day
span and decay parameter `a = 2/61`. A span controls exponential decay; it is
not a hard cutoff to the most recent 60 observations. Appendix A.2 uses a
weighted population variance, without an unbiased sample correction (p. 32).
The lag-weight interpretation is `w[k] = (1-a)^k`, normalized over the observed
past. The appendix's subscripts on the weight symbol are inconsistent; the
prose establishes exponential decay with lag.

The paper does not explicitly settle simple versus log multi-session returns.
For the equity baseline, using `p[t]/p[t-h]-1` on adjusted close is a proposed
convention. It must be frozen before comparing performance. No most-recent-month
skip appears in these five paper features.

### Three normalized MACD features

For `(S,L) = (8,24), (16,48), (32,96)`, equations (1)-(3) specify:

```text
m_J[t] = (1/J) * p[t] + (1 - 1/J) * m_J[t-1]
macd[t,S,L] = m_S[t] - m_L[t]
macd_norm[t,S,L] = macd[t,S,L] / rolling_std_63(p)[t]
macd_feature[t,S,L] = macd_norm[t,S,L] / rolling_std_252(macd_norm)[t]
```

Here the EMA smoothing coefficient is **1/J**, as confirmed in Appendix A.2.
Using an ordinary library `span=J` would instead use `2/(J+1)` and change the
method. Equation (3)'s printed denominator has incomplete window notation;
the accompanying text explicitly defines the 252-day rolling standard
deviation. Rolling standard-deviation degrees of freedom and minimum periods
are not specified.

### Feature winsorization and regression scaling

For each asset and feature separately, cap the feature to its exponentially
weighted mean plus or minus five exponentially weighted standard deviations;
the half-life is 252 sessions (p. 8). This applies to momentum features, not
realized investment returns. A causal implementation must freeze whether the
current observation enters its own clipping limits, initialization, minimum
history, and how gaps are handled. It must not estimate clipping limits from
the full sample.

Before linear regression, the paper removes feature means and scales to unit
variance (p. 26). Estimate these parameters on the regression fit portion only,
apply them unchanged to validation/test observations, then refit on the full
pre-test sample once hyperparameters are selected. The paper does not prescribe
an additional standardization of the concatenated graph inputs; adding one
would be a separate specification, since it changes graph distances.

## Daily graph learning

For each signal date `t`, collect the eight features over the preceding `delta`
sessions, including the current available feature vector, into a matrix
`V_t` of shape `N_t x (8*delta)`. Each row is an asset with an observed feature
history throughout that window (p. 9).

For each lookback, equation (4) minimizes:

```text
trace(V.T @ (D - A) @ V)
    - alpha * sum(log(A @ ones))
    + beta * sum(A**2)

D = diag(A @ ones)
A = A.T
A[i,j] >= 0
A[i,i] = 0
```

The zero diagonal follows the paper's explicit no-self-connections description
on p. 5, although it is omitted from the displayed constraints on p. 9. Include
it in the implementation; a graph which can use self-loops would change the
method materially.

The first term favors links between assets whose momentum histories are
similar. For pairwise squared distances `Z[i,j] = ||V[i,:]-V[j,:]||^2`, it equals
`sum(i<j, A[i,j]*Z[i,j])`. The log-degree term requires positive weighted degrees
at the optimum. It prevents isolated nodes, but does not mathematically ensure
that the entire network is one connected component. The squared Frobenius term
regularizes edge magnitudes. This is not a correlation-threshold graph, an
estimated directed lead-lag graph, or a graph neural network.

The paper uses CVXPY with MOSEK (p. 12). An alternative solver can implement the
same objective, but should be checked on small reference problems for objective
values, residuals and propagated signals. Freeze solver tolerances, treatment
of near-zero edges and any numerical degree floor. Do not tune an edge cutoff
using test performance.

### Ensemble and normalization

The primary method learns five graphs every trading day, with lookbacks
`252, 504, 756, 1008, 1260` sessions. It first averages the adjacency matrices,
then applies symmetric degree normalization (equations (5)-(6), p. 10):

```text
A_bar = mean(A_252, A_504, A_756, A_1008, A_1260)
D_bar = diag(A_bar @ ones)
A_tilde = D_bar**(-1/2) @ A_bar @ D_bar**(-1/2)
```

Normalization happens **after** averaging. This is not the same as averaging
five separately normalized graphs. Symmetric normalization is also different
from row normalization: rows of `A_tilde` need not sum to one. Consequently,
equation (7) is a weighted neighbor aggregation, despite the prose's loose
description as a weighted average.

The paper does not explicitly resolve differences in node availability across
the five lookbacks. A proposed primary equity implementation uses a common
node set with complete feature history for the longest window, aligned by
stable security ID. This makes all five matrices directly comparable. It can
exclude recent IPOs and stocks with insufficient usable history; those names
must remain in the original membership and eligibility ledger. Alternatives
that pad or combine differing node sets are separate methodological choices.

## Regression and portfolio construction

Propagate current features through the graph (equation (7)):

```text
U_network[t] = A_tilde[t] @ U_individual[t]
```

Fit one OLS regression, with eight coefficients and an intercept, pooling all
eligible asset/date observations (equation (8), p. 11):

```text
target[i,t] = return[i,t:t+1] / s[i,t]
prediction[i,t] = intercept + coefficients @ standardized_network_features[i,t]
position_direction[i,t] = sign(prediction[i,t])
```

The denominator of the target is known at `t`. It must not use tomorrow's
realized volatility. OLS is shared across stocks rather than fitted separately
to each stock. Negative coefficients are allowed, so a feature can express
reversal. Preserve the fitted signs; do not hardcode the signs in Table 4.

With annualized asset volatility `sigma[i,t] = sqrt(252)*s[i,t]`, equation (9)
corresponds to target weights:

```text
w[i,t] = sign(prediction[i,t]) * 0.15 / (N_t * sigma[i,t])
portfolio_return[t:t+1] = sum_i(w[i,t] * return[i,t:t+1])
```

The 15% target in this formula is an **asset-level** risk scale. The strategy
can be net long or net short; it does not enforce equal long and short books,
market neutrality, a top/bottom-decile selection, fixed gross exposure, or
market-cap weighting. Pooled OLS and this position rule are the paper baseline.

Panel A of Table 1 already includes this asset-level volatility scaling. Panel B
adds a second, portfolio-level scaling step targeting 15% annualized volatility
(pp. 12-14). The exact estimator, lag, initialization and leverage limits for
this second step are not specified. A proposed implementation uses a lagged
60-span EWM estimate of the strategy's own realized past returns. This must be
labeled an implementation convention; full-sample realized volatility scaling
would introduce future information.

## Training and validation chronology

Graph structures and signals change daily, but graph hyperparameters and OLS
coefficients are recalibrated **every five years**, using an expanding sample
(p. 12):

| Fit sample | Following test sample |
| --- | --- |
| 1990-1999 | 2000-2004 |
| 1990-2004 | 2005-2009 |
| 1990-2009 | 2010-2014 |
| 1990-2014 | 2015-2019 |
| 1990-2019 | 2020-2022 |

These are the authors' futures dates. The paper calls the aggregate period
22 years, while the stated inclusive calendar-year range is 2000-2022; retain
the explicit dates rather than deriving chronology from that count. The
occasional description as rolling on p. 26 does not change the expanding
samples listed on p. 12.

For each fit period, reserve the most recent 10% for validation. Both graph
hyperparameters use the grid:

```text
0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10
```

A full Cartesian interpretation has 121 pairs. The paper does not state the
validation score, whether parameters differ by lookback, or the exact final
refit sequence. A proposed primary convention is one common `(alpha,beta)`
pair for the ensemble, selected by validation portfolio Sharpe before the
extra portfolio-level scaling; validation MSE is retained as a diagnostic.
Any alternative score is a separate declared sensitivity. The convention is
not an author-confirmed implementation detail.

For an equity adaptation, define the last 10% in chronological **trading dates**,
keeping all stocks of a date together. Do not randomly split pooled stock-days.
Discard training examples whose next-session label is not complete by the fit
cutoff, and apply the same rule at the fit/validation boundary. Each historical
graph used to construct a fit example must use only features available on that
example's date, rather than a graph learned once using the whole fit period.
During the five-year test, graphs may incorporate newly observed past/current
features; regression coefficients, training scalers and selected hyperparameters
remain fixed until the scheduled recalibration.

## Comparison strategies

Use the same point-in-time population, price conventions, execution rules and
accounting across these paper arms (pp. 11-16):

| Strategy | Signal |
| --- | --- |
| Long Only | `x=1` with the same asset volatility scaling. |
| MACD | Average `phi(y)` across the three MACDs, where `phi(y)=y*exp(-y*y/4)/0.89`. Positions are continuous. |
| LinReg | Pooled OLS on the eight own-stock momentum features; take the prediction sign. |
| GMOM | Pooled OLS on eight graph-propagated features; take the prediction sign. |
| RegCombo | One pooled OLS with all 16 own-stock and network features; take its prediction sign. |
| SignCombo | Half the LinReg signed position plus half the GMOM signed position. |

Equation (12) mistakenly labels some RegCombo quantities as SignCombo; the
surrounding prose defines the 16-feature regression. The two combinations
should therefore remain distinct implementations.

The primary comparison is **GMOM versus LinReg on matched eligible dates and
stocks**. Report each method's broader coverage separately so that a smaller,
older or more liquid GMOM population cannot silently create an unfair comparison.
An actual S&P 500 total-return benchmark and an equal-weight PIT equity
benchmark would add context, using separately verified benchmark data.

## Reported results and their relevance to equities

Table 1 reports GMOM Sharpe `1.363` before the additional portfolio scaling and
`1.511` afterward. The corresponding annualized mean-return figures are 7.0%
and 22.2%. The headline is before the transaction-cost sensitivity; it is not
a net equity trading result or a stated CAGR.

Table 2 (p. 21) is particularly relevant to this project. Its equity-index
futures portfolio using the full cross-asset network has Sharpe `0.948`; the
equity-only network has Sharpe `0.302`, with 4.4% annualized return and 77.7%
maximum drawdown after portfolio scaling. The all-asset result has Sharpe
`1.511`. Individual S&P 500 stocks differ from both sets of futures. The paper
supports testing an equity-only adaptation, but gives no basis to assume that
the cross-asset advantage will transfer unchanged.

The paper's costs are also material. Equations (13)-(14) define:

```text
turnover_i[t] = 0.15 * abs(x[i,t]/sigma[i,t] - x[i,t-1]/sigma[i,t-1])
net_return[t] = mean_i(0.15*x[i,t]*r[i,t:t+1]/sigma[i,t]
                       - (cost_bps/10000)*turnover_i[t])
```

The tested pseudo-costs are `0, 0.5, 1, 2, 3, 4, 5` bps. The prose reports GMOM
remaining positive through 3 bps and negative at 5 bps (pp. 16-17). Reproduce
this convention for comparability, then report equity execution costs and
short financing separately. Even a constant positive signal trades as its
volatility weight changes; compute turnover from the equations/holdings, rather
than relying on the p. 17 description attributing long-only turnover only to
assets ceasing to trade.

## Proposed S&P 500 data and execution rules

### Historical membership and usable history

Use every exact-date `sp500` membership record from the existing prepared panel
as the original cohort. Eligibility follows membership effective/availability
dates, stable source IDs, documented identity admission, usable historical
prices and feature history available by the signal. It must never depend on
current membership, later survival or a complete future quote path.

For a stock that joins the index, pre-entry prices can supply its lookback
features while its tradability still starts with PIT membership. Requiring
complete price/features over the graph window does not mean requiring index
membership for that whole window. At removal, stop initiating positions under
the membership rule, but retain held positions and valuations through the
documented executable exit. Missing exit prices cannot erase a loss or remove
the position from the accounting denominator.

Our prepared panel covers 1970-03-31 through 2026-03-31. The separate price
history retains available earlier observations. Usable member-day quotes cover
only 3.5% in 1970, 48.9% in 1999, 90.6% in 2010, 95.0% in 2016 and 98.6% in
2023. See `data/sp500_daily/coverage_annual.csv` and `manifest.json`. Therefore
1970 cannot be treated as the start of a broad complete-price backtest.

The 1260-session graph requires five years of **feature** observations, with
additional raw-price history for constructing each feature. With the nested
MACD normalization, this means roughly six years or more of price history,
depending on EMA initialization and minimum-period conventions. A current
quote-coverage figure is not a measure of graph eligibility. Choose training
and test start dates from a feature-history coverage audit, before comparing
strategy results. Preserve unavailable constituents and all exclusion reasons.

The subsequent metadata audit in `data/warmup_coverage/` measures necessary
uninterrupted usable-price histories. It assumes full 63-observation and
252-observation MACD normalization windows: the first complete eight-feature
vector needs at least 314 prices, and 1260 such vectors need 1573 prices.
It imposes no extra EWM burn-in and does not evaluate zero-variance denominators
or actual feature values, so these are upper bounds on feature readiness under
the current quote-quality policy.

| Calendar year | Current usable quote coverage | History sufficient for the full graph ensemble |
| --- | --- | --- |
| 2011 | 90.3% | 13.9% |
| 2015 | 92.5% | 62.1% |
| 2016 | 95.0% | 78.5% |
| 2017 | 96.7% | 83.6% |
| 2020 | 97.6% | 84.9% |
| 2023 | 98.6% | 87.4% |

These are annual member-day percentages using all available earlier prices.
With raw history deliberately truncated to 2010-01-04, the first theoretically
complete feature vector falls on 2011-03-31 and the first full five-window
ensemble on 2016-04-04. The user subsequently clarified that 2011 means the
start of the **history used to learn the networks**, not the first date on
which a full five-window graph must already be available. The intended first
network-learning window is therefore approximately 2011-2015, with raw prices
from 2009-2010 used to initialize features and the first full ensemble around
early 2016. The regression fitting period and first out-of-sample test remain
separate choices. The earlier suggestion to postpone the start to 2017 is
superseded by this clarified definition.

The diagnostic is reproducible with `audit_warmup_coverage.py`. Read
`annual.csv`, `daily.csv` and `summary.json` together for coverage denominators,
history origins, exact earliest dates and untested feature conditions.
The subsequent [first-window completeness audit](FIRST_WINDOW_DATA_AUDIT.md)
checks this clarified 2011-2015 window and saves a per-security repair queue.

Adjusted close and its endpoint ratios are the current reconstruction inputs.
Do not add source dividends again to provider-adjusted returns. Revision-as-of
prices, fully dated identity validity and complete delisting/distribution
accounting remain limits. Equity splits, acquisitions, spin-offs, delistings
and ticker reuse require identity and corporate-action evidence beyond the
dated membership list.

### Decision timing and equity costs

The paper's `t` to `t+1` return formula is a research timing convention, not a
specified order-execution schedule. A feature using the final closing price at
`t` cannot automatically obtain a fill at that same close. Keep a paper-style
close-to-close proxy separate from an executable schedule such as next-session
open execution or a full-session signal lag. Freeze the executable schedule
before performance comparisons and charge entry, exit and rebalancing costs.
Execution timing can also change which return is appropriate as the label.

Report gross exposure, net exposure, beta, sector concentrations and realized
volatility. Inverse-volatility sizing and an additional portfolio scale can
create leverage. Dollar neutrality, beta neutrality, leverage caps, long-only
variants and rank-based baskets are useful separate experiments; none should
silently replace the paper's signed OLS position rule.

For cash equities, use actual traded weights/holdings to account for membership
changes, denominator changes, price drift, corporate actions, spreads, fees,
market impact, borrow availability/costs and financing. The paper's average
pseudo-turnover is not a complete stock-portfolio execution model. When a
holding lacks a valid mark or terminal payoff, retain the position and flag or
withhold the affected performance path rather than reallocating it away.

### Equity-specific network controls

An equity network may primarily capture the common market factor or sector
co-movement. Test that possibility rather than interpreting every learned link
as new spillover alpha. Alongside LinReg, compare fixed uniform-neighbor and
randomized-network controls with matched populations and risk scaling. A
nonnegative correlation-network comparison can test whether the learned graph
adds value beyond a simpler network construction.

Within-sector versus between-sector ablations, sector neutrality and
market/sector-residual input features are proposed equity extensions. They
require dated sector classifications or an explicitly disclosed substitute;
today's sector labels cannot establish historical classifications. They do not
belong in the primary price-only baseline automatically.

## Replication and testing sequence

1. **Freeze feature conventions and audit coverage.** Record simple/log returns,
   exact endpoints, EWM and rolling variance conventions, initialization,
   clipping timing, gap handling and stable-ID routes. Measure daily original
   membership, valid features, full-window graph nodes and known exclusions.
2. **Verify features independently.** Use hand-calculated examples, EMA
   recurrences, missing-session and split examples, and prefix-invariance tests:
   appending future observations must not change a past feature or graph.
3. **Verify graph mathematics on small examples.** Check symmetry, zero diagonal,
   nonnegative edges, positive degrees, objective/residuals, node permutation
   equivariance and stable-ID alignment. Compare direct and distance-form
   objectives and test ensemble-before-normalization. A degree-regular graph
   and a graph with unequal degrees can expose accidental row normalization.
4. **Verify pooled regression and chronology.** Check scaler fit dates, complete
   labels, temporal validation, prediction alignment and five-year parameter
   freezing. Using an identity matrix as a test propagation operator should
   reduce propagated features to own-stock features; this is a unit-test
   control, not an admissible learned graph with zero diagonal.
5. **Run a bounded engineering pilot.** Validate daily graphs, signals, positions,
   costs and missing-mark handling on a declared small date/stock scope. A
   reduced universe or single-lookback pilot verifies code, not full-method
   performance. Freeze the chronological folds and selection score before
   launching model comparisons.
6. **Evaluate the primary arms on successive unseen blocks.** Keep Long Only,
   MACD, LinReg and GMOM, then the two paper combinations, on consistent data,
   membership, execution and accounting. Preserve predictions and weights so
   performance can be independently reconstructed.
7. **Run declared robustness checks.** Compare all five individual graph
   lookbacks with the ensemble, graph controls, topology/stability, prediction
   errors, return/sign correlations, cost sensitivities and equity exposures.
   Test sector variations only with appropriate historical classification data.

Report annualized mean return and CAGR separately, volatility, Sharpe, downside
deviation, Sortino, Calmar, drawdown depth and duration, hit rate, profit/loss
ratio, turnover, costs and coverage. Add year/subperiod and crisis performance.
For uncertainty, use time blocks that preserve common stock-date shocks rather
than treating all pooled stock-days as independent. The paper's coefficient
t-statistics are descriptive reference results; stock-panel inference should
allow cross-sectional and serial dependence. Track all tested variants to
avoid presenting a selected result as a single unsearched hypothesis.

### Computation at S&P 500 scale

At 500 nodes there are 124,750 undirected pair variables versus 2,016 for 64
nodes, about 62 times as many. Five daily graphs and a 121-pair parameter grid
require a measured compute plan. Cache causal pairwise feature distances,
reuse past solution warm starts and verify solver tolerances before a full run.
For a fixed node set, rolling squared distances can be updated by adding the
latest eight-feature contribution and removing the oldest; membership changes
require explicit ID-aligned updates. Weekly graph estimation, fewer lookbacks,
a smaller hyperparameter grid or a different objective are named engineering
variants, not an exact reproduction of the daily ensemble.

## Implementation choices and remaining execution questions

The PDF specifies the architecture sufficiently to build it, but does not
uniquely specify all implementation details. The following must be recorded
before results are used to choose a variant:

| Choice | Paper evidence or proposed convention |
| --- | --- |
| Multi-session return definition | Simple adjusted-price ratios frozen for the first pilot. |
| Rolling variance, EWM initialization and clipping timing | First-pilot policy freezes ddof=0, full rolling windows, first-price EMA seeds, population EWM volatility with 60 returns, and current-inclusive clipping. |
| Missing data and multi-lookback node alignment | First pilot resets all indicator state after missing closes and uses the common longest-window node set. |
| Hyperparameter score and common/per-window parameters | Annual equity WFO uses a shared pair and complete-date raw proxy validation Sharpe; the paper does not specify the score/scope. |
| Post-selection refit | Annual equity WFO refits OLS/scalers on all pre-test labels known by the cutoff after selection. |
| Portfolio-level volatility estimator and leverage | 15% target stated; estimator, lag and limits need a frozen convention. |
| Equity training/test calendar | User selected 2016–2018 initial OLS training and annual expanding tests in 2019–2025, departing from the paper's five-year recalibrations. |
| Executable trading schedule and return label | Daily positions and next-day return target stated; equity fill timing must be specified. |
| Corporate actions, terminal values and short financing | Require equity accounting in addition to the futures pseudo-cost model. |

The [first feature and network pilot](FIRST_NETWORK_PILOT.md) now implements
the eight features, retains every dated historical member, and fits the first
five networks after the 2016-01-04 close. All 429 price-history-qualified anchor
members also qualify on actual features. Alpha and beta remain untuned at one;
that pilot contains no regressions, selected hyperparameters or backtest results.
The subsequent [annual expanding WFO](ANNUAL_WALK_FORWARD.md) implements
the user-selected 2016–2018 initial OLS window and seven unseen return years
in 2019–2025. The [verified research-proxy results](ANNUAL_WALK_FORWARD_RESULTS.md)
are separate from the first-anchor pilot and retain all unavailable valuation dates.
