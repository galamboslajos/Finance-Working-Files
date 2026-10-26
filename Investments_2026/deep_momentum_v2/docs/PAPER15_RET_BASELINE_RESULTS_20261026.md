# Paper15 executable-RET baseline — verified results, 26 October 2026

The isolated `codex/paper15-ret-5y-10y` experiment completed **16 fixed-setting
annual fits and independent checks of all 32 daily scenarios**. There are
**24 complete proxy paths and eight withheld RF / 5-year paths**. All old runs
remain unchanged. This is development evidence, not an untouched test.

Read [the frozen method](PAPER15_RET_BASELINE_20261026.md) and use
[the separate passive notebook](../notebooks/10_paper15_ret_baseline.ipynb).

## The target and RET score

The supervised return is **an unstopped, gross executable daily-path return**:
buy one unit at the first trading open after the historical monthly signal,
track daily open/close cash and share wealth, and exit at the following monthly
rebalance open. Reviewed cash/stock actions are applied; required missing daily
marks leave the target unavailable. Without an action, compounding the daily
marked returns telescopes to the adjusted open-to-open ratio. Intramonth
volatility/drawdown is not a penalty in this return-only target.

Those realized executable returns are ranked into ten within-month classes,
best = 1 and worst = 10, with exact ties kept together. Thus classes remain a
cross-sectional monthly classification interface, but their underlying return
is derived from executable daily prices, rather than calendar-close returns.
The 15 past momentum/state features keep the agreed paper indexing.

RET = sum of all ten predicted class probabilities times the trailing historical
mean executable return in that realized class. Monthly class means receive equal
weight. The five-year arm requires all 60 × 10 cells; the ten-year arm requires
all 120 × 10 cells. Actual scheduled outcome completion governs both annual
training and historical payoff cutoffs. Neither missing targets nor payoff
cells are imputed. This target adaptation is separate from the paper's original
calendar-month return definition and from prior return/drawdown experiments.

Training, complete past daily OHLC eligibility and RET payoff windows each
match their five/ten-year arm. RF and XGBoost within an arm use the same rows,
15 inputs, classes and payoff vectors. No tuning, expanded features, missing
indicators or feature imputation were used.

## Dates and primary daily results

The common economic period is **3 February 2020–29 December 2023: 47 months,
985 trading sessions**. Portfolio NAV is continuous across annual refits.
January 2020 forecasts remain saved, but that month is outside both arms'
paired economics because the ten-year executable-payoff history is incomplete.
The terminal December close values the backtest only: it does not substitute
for the protected January 2024 open in the December 2023 learning label.

Primary scenario: RET-ranked up to ten longs / ten shorts, matched counts,
50% long / 50% short initial weights, **10% per-asset stops**. Gross uses zero
trading costs; net uses **10 bp per one-way traded notional**. Net volatility
and Sharpe use daily returns and 252-session annualization, with zero cash rate.
CAGR uses geometric NAV growth. Borrow and funding costs remain zero.

| Arm | Gross total return | Net total return | Net CAGR | Net volatility | Net Sharpe | Net maximum DD |
| --- | --- | --- | --- | --- | --- | --- |
| RF / 5 years | Withheld | Withheld | Withheld | Withheld | — | Withheld |
| XGBoost / 5 years | +6.25% | -2.95% | -0.76% | +19.89% | 0.06 | -32.99% |
| RF / 10 years | -27.34% | -33.62% | -9.95% | +14.99% | -0.62 | -42.75% |
| XGBoost / 10 years | -3.51% | -11.79% | -3.16% | +16.67% | -0.11 | -32.91% |

The corresponding no-stop comparator is also retained, under the same 10 bp
cost and period. These alternatives were fixed before this run's results.

| Arm | No stop, net total return | 10% asset stops, net total return |
| --- | --- | --- |
| RF / 5 years | Withheld | Withheld |
| XGBoost / 5 years | +4.64% | -2.95% |
| RF / 10 years | -10.52% | -33.62% |
| XGBoost / 10 years | +12.09% | -11.79% |

No complete four-arm economic ranking is available while RF / 5 years is
withheld. Differences between five and ten years also change the eligible
population, cross-sectional normalization, return classes and payoff history;
they do not isolate training length alone.

## Prediction performance and coverage

Each model produces 48 forecast months, January 2020–December 2023. The
observable target months are January 2020–November 2023 (47), a different set
from the economic February–December period above. December 2023 targets need
protected January 2024 financial values, which were not read. Missing later
outcomes never remove an otherwise eligible forecast.

| Arm | Forecast rows | Observed labels | Mean monthly log loss | Mean monthly accuracy |
| --- | --- | --- | --- | --- |
| RF / 5 years | 39,756 | 38,876 | 2.284384 | 12.46% |
| XGBoost / 5 years | 39,756 | 38,876 | 2.289197 | 12.30% |
| RF / 10 years | 33,011 | 32,244 | 2.287213 | 12.31% |
| XGBoost / 10 years | 33,011 | 32,244 | 2.294208 | 12.32% |

Loss and accuracy above give each observed month equal weight. Uniform
probabilities have log loss 2.302585 and approximately 10% accuracy. The saved
notebook also exposes pooled scores, Brier scores, uniform/training-prior
references and each of the ten classes' separate precision, recall and support.

Eligibility summary across all 48 test signals:

| History arm | Original members, mean | Identity admitted, mean | Forecast eligible, mean | Forecast eligible, min–max |
| --- | --- | --- | --- | --- |
| 5 years | 909.48 | 900.31 | 828.25 | 814–835 |
| 10 years | 909.48 | 900.31 | 687.73 | 581–730 |

All 152,307 original 2010–2023 stock-month rows remain in each historical
panel's audit denominators. Membership is historical at each signal; five/ten
years of index membership is not required. Complete pre-index provider history
may qualify a stock. Raw 2000–2023 history is projected only for the admitted
identity universe and reviewed action routes; corrected cached observations
override their raw keys. The execution-only quote amendments stay outside
learning histories. Historical identities, adjustment vintage and complete
corporate-action/delisting coverage remain provisional.

## The one new-order gap and all eight withheld paths

RF / 5 years selected a new **short Eaton Vance (membership alias EV)** at the
26 February 2021 signal, for execution on **1 March 2021**. The pinned provider
route `sec_yRvj1y` lacks the required adjusted open. The first failure is
`missing_quote`, mark `open`, role `selected`; it is **not an existing holding
with a missing valuation**. EV was absent from the preceding January selection.

Maximum-holding preflight finds 24 missing quote keys on this one route from
1 March through 1 April 2021 and no other missing route or quality flags in
this arm. The other three arms have zero missing maximum-horizon quote keys.
All eight RF / 5-year combinations of two stops and four costs encounter the
same new-order failure, rather than eight separate issuer defects. Their
full-period and annual returns remain unavailable; partial results are not
substituted. The partial stop count is not a full-period statistic.

The existing retired-new-order policy would leave a verified retired security's
slot in cash, without replacement or redistribution. This run's pinned action
registry has no applicable EV retirement entry, so it does not assume that
exception, fabricate a quote or silently skip the order. Resolving this exact
case would need separately pinned legal retirement/timing evidence and a
versioned execution-only replay of the same selections, not new model fits.
Known execution repairs remain deferred under the user's earlier direction.

## Verification, resources and preserved attempts

Independent source reconstruction checked daily history, eligibility, Paper15
formulas, executable cash/share targets, exact classes and 60/120-month causal
payoffs. A separate model check reloaded all 16 saved estimators and reproduced
inference, class mapping and reported prediction metrics without refitting.
Independent selection, preflight and cash/share/stop/cost checks cover all 32
daily scenarios, including correct withholding. Calculation checks do not
certify the completeness of provider identities, actions or investability.

The final private run is `outputs/paper15_ret_baseline/20261026-v3`, status
`COMPLETE_VERIFIED`. Active controller time was 495.91 seconds
(including the preserved zero-fit failed attempt); peak measured worker RSS was
2.42 GiB. Thirteen targeted synthetic tests pass. Financial 2024+ values,
expanded feature panels and cloud resources were not accessed.

The superseded calendar-close plan was metadata-only: no financial projections
or fits. Executable preparation v2 stopped on an independent-checker local
variable collision with zero fits. Recovery v3 reused prepared bytes, fixed
only that independent lookup and reserved an extra bounded reconstruction pass.
All earlier plans, code, failed evidence and checkpoints remain preserved.
The 16 empirical fits were each consumed once; no settings were selected from
these test outcomes.

Git contains original source, the frozen protocol, tests, an output-free passive
notebook and these aggregate summaries. Raw data, models, per-stock predictions,
private receipts, holdings and credentials remain outside Git. A fresh clone
does not contain private evidence or start computation through Run All.

## Every fixed daily scenario

| Arm | Stop | One-way cost, bp | Total return | CAGR | Sharpe | Maximum DD | Coverage |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RF / 5 years | None | 0 | Withheld | Withheld | — | Withheld | Withheld from 2021-03-01 |
| RF / 5 years | None | 5 | Withheld | Withheld | — | Withheld | Withheld from 2021-03-01 |
| RF / 5 years | None | 10 | Withheld | Withheld | — | Withheld | Withheld from 2021-03-01 |
| RF / 5 years | None | 25 | Withheld | Withheld | — | Withheld | Withheld from 2021-03-01 |
| RF / 5 years | 10% per asset | 0 | Withheld | Withheld | — | Withheld | Withheld from 2021-03-01 |
| RF / 5 years | 10% per asset | 5 | Withheld | Withheld | — | Withheld | Withheld from 2021-03-01 |
| RF / 5 years | 10% per asset | 10 | Withheld | Withheld | — | Withheld | Withheld from 2021-03-01 |
| RF / 5 years | 10% per asset | 25 | Withheld | Withheld | — | Withheld | Withheld from 2021-03-01 |
| XGBoost / 5 years | None | 0 | +13.58% | +3.31% | 0.25 | -30.40% | 985 / 985 sessions |
| XGBoost / 5 years | None | 5 | +9.03% | +2.24% | 0.21 | -31.82% | 985 / 985 sessions |
| XGBoost / 5 years | None | 10 | +4.64% | +1.17% | 0.17 | -33.36% | 985 / 985 sessions |
| XGBoost / 5 years | None | 25 | -7.52% | -1.98% | 0.04 | -37.80% | 985 / 985 sessions |
| XGBoost / 5 years | 10% per asset | 0 | +6.25% | +1.56% | 0.18 | -27.53% | 985 / 985 sessions |
| XGBoost / 5 years | 10% per asset | 5 | +1.55% | +0.39% | 0.12 | -30.20% | 985 / 985 sessions |
| XGBoost / 5 years | 10% per asset | 10 | -2.95% | -0.76% | 0.06 | -32.99% | 985 / 985 sessions |
| XGBoost / 5 years | 10% per asset | 25 | -15.32% | -4.17% | -0.11 | -40.73% | 985 / 985 sessions |
| RF / 10 years | None | 0 | -2.88% | -0.74% | 0.03 | -29.12% | 985 / 985 sessions |
| RF / 10 years | None | 5 | -6.78% | -1.78% | -0.03 | -30.20% | 985 / 985 sessions |
| RF / 10 years | None | 10 | -10.52% | -2.80% | -0.09 | -31.26% | 985 / 985 sessions |
| RF / 10 years | None | 25 | -20.92% | -5.83% | -0.28 | -34.37% | 985 / 985 sessions |
| RF / 10 years | 10% per asset | 0 | -27.34% | -7.85% | -0.47 | -39.43% | 985 / 985 sessions |
| RF / 10 years | 10% per asset | 5 | -30.55% | -8.90% | -0.55 | -41.11% | 985 / 985 sessions |
| RF / 10 years | 10% per asset | 10 | -33.62% | -9.95% | -0.62 | -42.75% | 985 / 985 sessions |
| RF / 10 years | 10% per asset | 25 | -42.07% | -13.04% | -0.85 | -47.42% | 985 / 985 sessions |
| XGBoost / 10 years | None | 0 | +21.66% | +5.14% | 0.37 | -20.72% | 985 / 985 sessions |
| XGBoost / 10 years | None | 5 | +16.78% | +4.05% | 0.31 | -20.87% | 985 / 985 sessions |
| XGBoost / 10 years | None | 10 | +12.09% | +2.96% | 0.25 | -21.03% | 985 / 985 sessions |
| XGBoost / 10 years | None | 25 | -0.92% | -0.24% | 0.08 | -24.69% | 985 / 985 sessions |
| XGBoost / 10 years | 10% per asset | 0 | -3.51% | -0.91% | 0.03 | -28.27% | 985 / 985 sessions |
| XGBoost / 10 years | 10% per asset | 5 | -7.74% | -2.04% | -0.04 | -30.63% | 985 / 985 sessions |
| XGBoost / 10 years | 10% per asset | 10 | -11.79% | -3.16% | -0.11 | -32.91% | 985 / 985 sessions |
| XGBoost / 10 years | 10% per asset | 25 | -22.94% | -6.45% | -0.31 | -39.82% | 985 / 985 sessions |
