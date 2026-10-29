# First S&P 500 feature and network pilot

The eight momentum features and the first five-network ensemble have been
built for **4 January 2016, after the close**. Of that day's **505 PIT members,
429 have complete actual feature histories for all five windows**. The earlier
429-stock price-history screen therefore survives the feature and denominator
checks under the conventions below.

This completes feature construction and a numerical graph pilot. Graph
hyperparameters have not been selected, and no OLS regression, positions or
performance backtest has been fitted. The 2011–2015 period supplies the graph's
historical feature observations; it is not a completed OLS training sample.

## Feature history and eligibility

Initialization uses reviewed closing-price history from 2009-01-01 through
2016-01-04. The longest graph uses **1,260 feature sessions from 2010-12-31 to
2016-01-04 inclusive**. That slight extension of the calendar-year window
follows the paper's exact session counts: 2011–2015 contains 1,258 XNYS sessions.

| Graph history | First feature session | Independently eligible PIT stocks | Stocks used in ensemble |
| --- | --- | ---: | ---: |
| 252 sessions | 2015-01-05 | 450 | 429 |
| 504 sessions | 2014-01-03 | 446 | 429 |
| 756 sessions | 2013-01-03 | 438 | 429 |
| 1,008 sessions | 2011-12-30 | 433 | 429 |
| 1,260 sessions | 2010-12-31 | 429 | 429 |

All five matrices use the same ordered 429 source IDs so they can be averaged
directly. There are 456 members with complete current-day features, but some
lack the history required for a graph. The remaining 76 longest-window rows
are retained: 8 have withheld identities, 41 have unavailable current features,
and 27 have current features but incomplete history. These categories describe
feature availability rather than certifying the cause of each missing price.

The historical feature panel preserves all **630,009 member-days in
2011–2015**, plus the 505 anchor rows. Historical cohorts come from each date's
membership, rather than applying the January 2016 selection retrospectively.
Pre-entry prices can warm indicators for a later member.

| Year | Historical member-days | Complete current eight-feature vectors | Coverage |
| --- | ---: | ---: | ---: |
| 2011 | 126,000 | 111,568 | 88.55% |
| 2012 | 125,000 | 111,557 | 89.25% |
| 2013 | 126,000 | 113,551 | 90.12% |
| 2014 | 126,289 | 114,105 | 90.35% |
| 2015 | 126,720 | 113,238 | 89.36% |

These percentages concern current own-stock features, not five-year graph or
OLS training availability on every historical date.

## Frozen feature conventions for this pilot

The five return horizons, three MACDs, 60-span volatility, nested 63/252-session
normalization and 252-session clipping half-life follow the supplied paper.
Its unspecified details are fixed in
`data/momentum_features/v1/policy.json`:

- Cumulative returns are adjusted-close endpoint ratios minus one. Every
  intervening session needs an admitted close.
- Daily volatility is a normalized, current-inclusive exponential weighted
  population standard deviation, alpha `2/61`, with at least 60 daily returns.
- MACD EMAs use alpha `1/J`, recurrence initialization at the first admitted
  price, and no adjusted finite-history normalization.
- Both rolling standard deviations use `ddof=0` and full windows. The first
  eight-feature vector needs 314 uninterrupted prices. Denominators must be
  strictly positive and finite.
- Winsorization uses each raw feature's current-inclusive exponential weighted
  population moments, half-life 252, and five standard deviations. Moments
  begin at the first finite feature and use unclipped observations.
- Every absent or withheld session close resets all indicator and clipping
  state. Prices and returns are never filled. An invalid denominator withholds
  its feature; no denominator floor manufactures a signal.

These are declared implementation choices, rather than author-confirmed
settings. The 107 internally reviewed closes remain provisional research
inputs. Their source OHLC warnings and execution rejections remain intact.
The eight absent observations in SPGI, CELG, COL and LVLT are still absent.
Existing identity and provider-adjustment limitations also remain.

## Graph objective and numerical results

The pilot uses **alpha=1 and beta=1**, both values in the paper's parameter
grid. They were selected as engineering defaults, without future labels or
validation scores. No claim about their investment merit follows.

For each window, the code uses the unscaled concatenated feature vectors and
the exact objective recorded in the methodology document. It enforces
nonnegative symmetric adjacency with zero diagonal. The implementation solves
an equivalent convex dual with one variable per node, using damped Newton
steps in SciPy; it does not require a MOSEK license. It checks all 91,806
candidate edges and applies no edge pruning, distance normalization or degree
floor.

For edge vector `w`, degree-incidence operator `S` and squared distances `z`:

```text
primal: min(w>=0) z'w + 2*beta*w'w - alpha*sum(log(Sw))
dual:   min(y>0) sum((y_i+y_j-z_ij)_+^2)/(8*beta) - alpha*sum(log(y))
w_ij = (y_i+y_j-z_ij)_+/(4*beta)
optimality: degree_i*y_i/alpha = 1
```

The factor two in the primal penalty accounts for both symmetric entries in
the paper's Frobenius norm. The stable primal-dual gap is
`alpha*sum(x-1-log(x))`, with `x=degree*y/alpha`.

| Lookback | Positive undirected edges | Connected components | Newton iterations |
| --- | ---: | ---: | ---: |
| 252 | 349 | 93 | 142 |
| 504 | 352 | 88 | 81 |
| 756 | 358 | 81 | 114 |
| 1,008 | 346 | 92 | 138 |
| 1,260 | 361 | 76 | 341 |

All five converge below the declared `1e-6` relative degree tolerance; the
largest attained residual is `1.28e-8`. The largest stable primal-dual gap is
approximately `1.03e-15`; direct objective subtraction differs by floating-point
roundoff. An independent check of all edge optimality conditions has maximum
relative residual `5.86e-9`.

The ensemble averages raw adjacency matrices before symmetric degree
normalization, then propagates the current eight features. It contains
**1,068 positive edges and four components of sizes 423, 2, 2 and 2**. Every
node has positive degree. The objective permits disconnected components; no
extra edges were added to force connectivity. These sparse untuned graphs need
parameter validation before investment interpretation.

## Saved artifacts and reproduction

| Location | Contents |
| --- | --- |
| `data/momentum_features/v1/native_features/` | Exact-session history per native ID: masked close, returns, volatility, raw/clipped features, readiness and original execution eligibility. Includes nonmember warmup and explicit missing sessions. |
| `data/momentum_features/v1/pit_features/` | All dated historical members joined to features, with withheld identities masked. |
| `data/momentum_features/v1/anchor_eligibility.csv` | All 505 members, per-lookback counts, eligibility and reasons. |
| `data/momentum_features/v1/anchor_feature_history.npz` | Ordered, finite 1,260-session history for the common 429 nodes. |
| `data/network_pilot/2016-01-04_a1_b1_v1/` | Five adjacency/distance/dual arrays, convergence traces, ordered nodes, ensemble, edge ledger and propagated current features. |
| `data/momentum_pilot_verification/v1/report.json` | Independent source-integrity, feature-formula, dated-node and graph checks. |

Both builds include manifests with source, script, policy and output hashes.
The feature build occupies approximately 365 MiB. Outputs are ignored by Git;
code, tests, configuration and this report remain outside the ignored data
directories.

```bash
python build_momentum_features.py
python learn_momentum_network.py
python validate_momentum_pilot.py
python -m unittest discover -s tests -v
```

The existing Deep Momentum V2 Python environment can run all commands. Build
directories are immutable: select fresh `--output` locations to rerun, and pass
those locations to later stages. The verification script also refuses to
overwrite an existing report.

For example, inspect the saved own and propagated features with:

```python
import pandas as pd

signals = pd.read_parquet(
    "data/network_pilot/2016-01-04_a1_b1_v1/anchor_network_features.parquet"
)
eligibility = pd.read_csv("data/momentum_features/v1/anchor_eligibility.csv")
```

## Verification and next modeling stage

All **42 structural tests pass**, including explicit weighted-volatility and
EMA formulas, feature clipping, missing-session resets, future-prefix
invariance, dated eligibility, an analytic two-node graph, an independent
primal SLSQP comparison, Frobenius/trace factors, node permutation and ensemble
normalization order.

The separate verifier checks **35 saved output objects by SHA256**, confirms
the 429 nodes against the original dated membership, and rederives 128 current
feature values for 16 spread-out securities directly from source prices using
NumPy formulas. Maximum feature discrepancy is `5.53e-12`; maximum daily
volatility discrepancy is `1.39e-17`. It also checks saved distances, all graph
edge optimality conditions and ensemble propagation independently.

The next modeling stage requires a causal **daily graph-feature series**,
next-session volatility-scaled labels, and explicit OLS fitting, validation and
out-of-sample dates. The single January 2016 graph cannot be reused backward
as historical training features. Execution timing and equity holding/exit
accounting still need specifications before a strategy backtest.
