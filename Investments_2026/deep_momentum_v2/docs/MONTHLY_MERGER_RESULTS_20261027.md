# Full monthly Paper15 comparison — completed 27 October 2026

**VERIFIED LOCAL:** all 32 declared scenarios are complete, with zero withheld.
This is a realized-return repair of the saved monthly experiment: **zero new
fits**, unchanged forecasts, strict training labels, features, RET payoff
estimates and selected baskets. The original sixteen annual RF/XGBoost models
remain the same. The user's explicit choice was to repair portfolio returns
only. Read [the frozen method and primary action evidence](MONTHLY_MERGER_REPLAY_20261026.md)
and use [passive notebook 12](../notebooks/12_monthly_merger_replay.ipynb).

## Primary full period: January 2020–December 2023

There are **48 monthly returns**, from the 31 December 2019 signal close to
the 29 December 2023 terminal close. All four arms use the same calendar.
Historical S&P 500/400 union, Paper15 inputs, annual rolling five/ten-year fits
and matching past-history eligibility remain the prior experiment's rules.
RET uses all ten probabilities and historical class-return means, with its
original matching 60/120-month payoff windows. Top/bottom ten, 50% long / 50%
short, equal within-side weights; no intramonth stops. Primary net costs are
10 bp per one-way traded notional. Gross is the separate zero-cost replay.

| Arm | Gross total | Net total | Net CAGR | Net annualized volatility | Net Sharpe | Net month-end max drawdown |
| --- | --- | --- | --- | --- | --- | --- |
| RF · 5 years | +28.94% | +18.49% | +4.33% | 22.65% | 0.30 | -23.83% |
| XGBoost · 5 years | +22.20% | +12.56% | +3.00% | 20.99% | 0.24 | -19.46% |
| RF · 10 years | -1.21% | -9.16% | -2.37% | 16.12% | -0.07 | -20.49% |
| XGBoost · 10 years | +59.62% | +46.84% | +10.08% | 17.04% | 0.65 | -12.07% |

CAGR is compounded growth, distinct from annualized arithmetic mean. Volatility
and zero-cash-rate Sharpe use monthly returns and sqrt(12); maximum drawdown is
observed only at month ends. These cannot reveal intramonth drawdowns. Same-close
entry is idealized; borrow and funding costs are zero. No terminal liquidation
is imposed. Costs are paid from cash with the existing pre-trade NAV target
weights; mandatory conversions do not incur a market-trading fee.

## Calendar-year net returns at 10 bp

The continuous portfolio is not reset at annual refits. These annual returns
compound to each full-period total above.

| Arm | 2020 | 2021 | 2022 | 2023 |
| --- | --- | --- | --- | --- |
| RF · 5 years | +12.84% | +6.12% | -15.17% | +16.64% |
| XGBoost · 5 years | +17.08% | +18.86% | -18.29% | -1.02% |
| RF · 10 years | +5.31% | -4.08% | -9.40% | -0.74% |
| XGBoost · 10 years | +31.65% | +5.63% | -10.39% | +17.84% |

## Matched February 2020–December 2023 period

This separate 47-month replay starts flat at the 31 January 2020 signal close.
It uses the same saved forecasts and calendar as the earlier daily experiment,
but retains monthly same-close economics. It is not a cropped continuation of
the full48 portfolio. Comparing it with the daily experiment also changes
learning labels, completion/payoff clocks and execution conventions; that
comparison cannot isolate only the effect of fills.

| Arm | Gross total | Net total | Net CAGR | Net annualized volatility | Net Sharpe | Net month-end max drawdown |
| --- | --- | --- | --- | --- | --- | --- |
| RF · 5 years | +27.05% | +16.96% | +4.08% | 22.89% | 0.28 | -23.83% |
| XGBoost · 5 years | +21.77% | +12.36% | +3.02% | 21.22% | 0.24 | -19.46% |
| RF · 10 years | -5.05% | -12.56% | -3.37% | 16.16% | -0.13 | -20.49% |
| XGBoost · 10 years | +55.11% | +42.93% | +9.55% | 17.19% | 0.61 | -12.02% |

## Complete cost sensitivity

All 32 paths below are complete: four arms × two periods × four fee rates.
Cells are cumulative portfolio returns, not annual returns.

| Period | Arm | 0 bp total | 5 bp total | 10 bp total | 25 bp total |
| --- | --- | --- | --- | --- | --- |
| full48 | RF · 5 years | +28.94% | +23.61% | +18.49% | +4.33% |
| full48 | XGBoost · 5 years | +22.20% | +17.28% | +12.56% | -0.56% |
| full48 | RF · 10 years | -1.21% | -5.26% | -9.16% | -19.94% |
| full48 | XGBoost · 10 years | +59.62% | +53.11% | +46.84% | +29.49% |
| matched47 | RF · 5 years | +27.05% | +21.90% | +16.96% | +3.25% |
| matched47 | XGBoost · 5 years | +21.77% | +16.97% | +12.36% | -0.46% |
| matched47 | RF · 10 years | -5.05% | -8.88% | -12.56% | -22.76% |
| matched47 | XGBoost · 10 years | +55.11% | +48.90% | +42.93% | +26.35% |

## What the repair changes

Four distinct missing stock-month returns affected six selected arm exposures:
RF5 had EV in March 2021 and CDK in July 2022; XGBoost5 had CDK in July 2022;
RF10 had XEC in October 2021 and XLNX in February 2022; XGBoost10 had XEC in
October 2021. The earlier missing old-stock month-end prices were not created.
Instead, exact cached raw/adjusted close pairs establish former-share units,
then documented cash and/or successor shares establish the realized payoff.
The same selected positions remain, including signed short liabilities.
Cash remains cash and successor shares survive/net into the next rebalance.
The next voluntary trade therefore uses the actual post-merger holdings.

The [method](MONTHLY_MERGER_REPLAY_20261026.md) records EV's non-election mixed
consideration, inherited CDK/XEC terms, XLNX/AMD conversion, effective dates,
primary sources and adjustment-unit formula. Twelve exact cached close points
support the four corrections. No new download, full raw-price scan, hindsight
stock removal, fictional retired-stock close or duplicate dividend was used.

## Verification and preserved references

The ledger is COMPLETE_VERIFIED. Independent amendment checks reconstruct all
four stock-months from twelve exact cache points. Independent accounting checks
reconstruct all 32 scenarios, all 128 annual slices, signed action applications,
voluntary trades, fees, long/short contributions and monthly NAV without
importing the producer accounting. There are 48 signed action applications
across the scenario grid. The original first fourteen months reproduce the
prior common-period report at every cost. Saved features, labels, RET inputs,
selected baskets and all sixteen fit/forecast receipt bindings are unchanged.

Private run: `outputs/paper15_monthly_merger_replay/20261026-v1`.
Backtest verification: `verification/backtest.json`, plan digest
`951647ee7a677c615c51fae7e4df42755b823cab4a1391d710baaa83697c5dde`. All generated panels, quotes, models, per-stock
ledgers and receipts remain outside Git. Notebook 12 authenticates the saved
proof chain and reads small reports only. Run All starts no compute or replay.
The optional figure is a presentation of the verified aggregate ledgers,
with its source/figure hashes in a separate private presentation manifest.

The [original strict run and fourteen-month report](PAPER15_MONTHLY_RESULTS_20261026.md)
and [prior executable daily experiment](PAPER15_RET_BASELINE_RESULTS_20261026.md)
remain unchanged. The fourteen-month totals must not be compared directly
with these four-year totals as evidence of improvement. Prediction metrics
are unchanged and remain in the original strict monthly report/notebook 11.

Code and reporting validation: 41 focused synthetic tests passed. The passive
notebook executed against the saved proofs without importing an estimator,
with all committed outputs empty. All seven frozen worker/helper/config pins
still match the empirical plan. The saved plot was visually inspected.

## Interpretation and remaining limits

XGBoost with ten-year history has the highest realized return in this fixed
four-arm monthly comparison; RF ten-year is negative after costs. Five/ten-year
arms differ in both rolling training/payoff history and complete-past-history
eligibility, so this is not a controlled estimate of training-window effects
alone. Four years of already examined outcomes are development evidence, not
a new untouched holdout or a basis to claim statistical superiority.

**Learning inputs remain frozen as requested.** Strict labels and RET still
use the originally observed monthly-return population; these four reconstructed
payoffs were not promoted into targets, historical class means or later
training. The metadata-only dependency audit records possible affected windows,
with no input changes or new fits. Repairing those learning returns and
recomputing predictions would be a separate experiment. This full realized
comparison does not certify all historical learning outcomes as repaired.

Provisional identity, source adjustment/distribution coverage, historical
knowledge-time and investability limits persist. Fractional successor units
are economic proxies; same-close fills, zero borrow/funding and no intramonth
stop/risk observations remain material assumptions. Protected 2024+ financial
values were not read. The broader expanded-feature and stopped daily runs
remain separate historical comparisons.
