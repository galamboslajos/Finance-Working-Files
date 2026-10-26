# Paper15 strict monthly benchmark — 26 October 2026

The user-requested monthly-close experiment completed **16 annual model fits**
and independent formula/inference/accounting checks of **32 monthly scenarios**.
All full-period paths are withheld for missing selected monthly returns.
The user explicitly chose to keep the strict adjusted-close definition and
show the common available period, rather than add corporate-action returns or
refit. The separately checked common diagnostic is **January 2020–February
2021: 14 monthly returns**. It must not be presented as a four-year result.

Read [the frozen monthly method](PAPER15_MONTHLY_BASELINE_20261026.md), use
[the separate passive notebook](../notebooks/11_paper15_monthly_baseline.ipynb),
and retain [the executable daily-target baseline](PAPER15_RET_BASELINE_RESULTS_20261026.md)
as a different experiment. Everything remains on `codex/paper15-ret-5y-10y`;
the original master, sources and prior runs are unchanged.

## What was run

The same 15 paper momentum/state inputs and matching five/ten-year complete
past OHLC eligibility were reused exactly from the independently verified
baseline. RF and XGBoost refit every year for 2020–2023 on the preceding five
or ten calendar outcome years, with identical fixed parameters and no tuning.
At the monthly close, both trainable December outcomes and RET's latest
completed signal-month outcome are now available. There is no next-open purge
for this target. There are 48 completed monthly label months through December
2023, without reading any January 2024 financial value.

The target is the exact provider-adjusted end-close / preceding month-end-close
ratio minus one, ranked into the same ten exact-tie classes. RET dots all ten
probabilities with the equal-time-weighted class payoffs from exactly 60/120
completed monthly outcomes. All 48 payoff vectors are complete in both arms.
Missing outcomes remain unlabelled and never filter signal eligibility.

The portfolio assumes each selected stock earns that monthly return from the
signal close. This is **idealized same-close monthly accounting**, with no
claim that already observed closing information can be traded at that close.
The prior top-ten long / bottom-ten short selection and 50%/50% allocation
remain. Costs use drifted monthly notionals. No daily execution or intramonth
stop is inferred, and no missing return is filled. This is the paper's monthly
return convention under our retained V2 source/universe/training/breadth
adaptations, not the paper's complete CRSP/16-input/100-fit implementation.

## Common available period — 14 months only

The period starts at the 31 December 2019 signal close and covers **January
2020–February 2021**, ending at the 26 February 2021 close. Every arm has all
selected monthly returns over this same prefix. The endpoint was chosen from
the earliest missing selected-return month (March 2021), before opening the
prefix's financial aggregates. It was not selected for performance, but it is
still a **post-run, coverage-limited diagnostic**, not a new untouched test.
It is dominated by a particular short market interval and cannot establish a
durable advantage or the full 2020–2023 performance.

Gross below has zero trading costs; net uses **10 bp per one-way traded
notional**. Borrow and funding are zero. CAGR annualizes only 14 months.

| Arm | Gross total return | Net total return | Net CAGR | Net monthly-based Sharpe | Net month-end maximum DD |
| --- | --- | --- | --- | --- | --- |
| RF / 5 years | +24.20% | +21.18% | +17.90% | 0.62 | -18.30% |
| XGBoost / 5 years | +28.19% | +25.48% | +21.48% | 0.74 | -14.08% |
| RF / 10 years | +9.18% | +6.58% | +5.62% | 0.35 | -16.82% |
| XGBoost / 10 years | +36.80% | +33.51% | +28.11% | 1.15 | -12.07% |

Sharpe uses monthly returns with sqrt(12) annualization. Drawdown is observed
only at month ends and can miss substantially worse intramonth losses. These
are not the earlier daily Sharpe/drawdown measures. Do not compare these
14-month totals directly with the old 47-month daily totals as a label effect.
Labels, completion cutoffs, RET payoff clock and accounting convention also
changed together, so even aligned dates would not isolate execution alone.

Net calendar slices, with no NAV reset:

| Arm | 2020 total net return | January–February 2021 total net return |
| --- | --- | --- |
| RF / 5 years | +12.84% | +7.39% |
| XGBoost / 5 years | +17.08% | +7.18% |
| RF / 10 years | +5.31% | +1.21% |
| XGBoost / 10 years | +31.65% | +1.41% |

Additive contributions to the net common-period result, in percentage points
of initial capital (not standalone leg compounded returns):

| Arm | Long P&L / initial capital | Short P&L / initial capital | Trading costs / initial capital | Net total return |
| --- | --- | --- | --- | --- |
| RF / 5 years | +33.64% | -9.97% | -2.50% | +21.18% |
| XGBoost / 5 years | +43.66% | -15.87% | -2.30% | +25.48% |
| RF / 10 years | +20.00% | -10.98% | -2.44% | +6.58% |
| XGBoost / 10 years | +40.20% | -3.93% | -2.76% | +33.51% |

All four short legs detract in this common interval. That is descriptive of
the saved baskets; no selection, stop or model setting was changed in response.

## Full 2020–2023 results remain unavailable

The original full48 (January–December 2020–2023) and matched47
(February 2020–December 2023) scenarios remain **0 complete / 32 withheld**.
No partial path is promoted to a full result. The common 14-month report adds
no fills, fits, replays or changes to those full-period artifacts.

A metadata-only join audited all **3,840 selected episodes** across four arms,
including later months after the first failure. It found six arm/stock/month
episodes representing four distinct missing stock-month endpoints:

| Arm | First unavailable month | All selected missing episodes | Membership aliases / months | 2020–2023 result |
| --- | --- | --- | --- | --- |
| RF / 5 years | 2021-03 | 2 | EV 2021-03, CDK 2022-07 | Withheld |
| XGBoost / 5 years | 2022-07 | 1 | CDK 2022-07 | Withheld |
| RF / 10 years | 2021-10 | 2 | XEC 2021-10, XLNX 2022-02 | Withheld |
| XGBoost / 10 years | 2021-10 | 1 | XEC 2021-10 | Withheld |

The repeated costs and periods explain why these gaps withhold 32 scenarios.
They are not 32 different issuer problems. EV lacks the March 2021 month-end
return; CDK lacks July 2022, XEC October 2021 and XLNX February 2022. These are
membership aliases on the unchanged provisional source routes, not new provider
identity certifications. The report identifies missing accepted endpoints;
it does not treat absence as a zero return or invent a legal entitlement.

The earlier executable baseline's EV failure was a new order at the next
March open. This monthly experiment assumes entry at the preceding February
close, so the gap is now the **monthly outcome endpoint**. These are distinct
accounting clocks even though the same predecessor history is involved.
By user choice, action-aware monthly return construction and dependent refits
are not part of this completion. Known execution cases stay deferred.

## Prediction metrics — all 48 outcome months

These scores use **January 2020–December 2023**, not the 14-month economic
diagnostic. Each observed month receives equal weight. Missing outcome labels
remain absent from scoring while their forecasts remain in the ledger.

| Arm | Forecast rows | Observed monthly labels | Mean monthly log loss | Mean monthly accuracy |
| --- | --- | --- | --- | --- |
| RF / 5 years | 39,756 | 39,699 | 2.284916 | 12.39% |
| XGBoost / 5 years | 39,756 | 39,699 | 2.289059 | 12.14% |
| RF / 10 years | 33,011 | 32,971 | 2.286551 | 12.23% |
| XGBoost / 10 years | 33,011 | 32,971 | 2.291960 | 12.48% |

Uniform log loss is 2.302585 with approximately 10% accuracy. The five-year arm
has 57 unavailable outcomes among 39,756 forecasts; the ten-year arm has 40
among 33,011. The 48-month scoring sample differs from the earlier daily-path
target's 47 observable label months. Small differences across those headline
scores are not a controlled equal-label comparison or a significance test.
Class-mean RET payoffs remain conditional on observed returns; a retained
missingness ledger does not repair missing terminal observations or certify
survivorship freedom.

## Checks, resources and use

Independent checks authenticated the parent source/feature proof, exact X and
eligibility reuse, monthly return formulas, ties, coverage and 60/120-month
payoff clocks before fitting. All 16 saved models reproduced their inference,
metrics, class mappings and confusion counts without refitting. The independent
monthly cash/notional oracle checked every selection, cost, contribution,
turnover, NAV and correct withholding across all 32 scenarios. The common-prefix
report binds those verified ledgers and checks every reported aggregate with
two distinct formulas. It does not supply absent monthly returns.

All **26 focused tests** passed. The controller finished in
217.53 active seconds with
0.63 GiB peak measured worker RSS. The sealed
source projection was 11,860,397 bytes (11.31 MiB). No daily financial arrays,
raw source panels, expanded inputs, previous models, cloud resources or
financial 2024+ values were read by this experiment. Monthly closes and verified
signal inputs were sufficient.

Private run: `outputs/paper15_monthly_baseline/20261026-v1`, status
`COMPLETE_VERIFIED`. Its full-period results remain withheld; the status means
the fixed computation and independent checks completed. The separate
`reports/common_period_v1` records the user-selected diagnostic. The standalone
notebook authenticates these receipts and reads small saved reports only.
Run All launches no fits, data builds or backtest. Git contains code, protocol,
tests, empty-output notebook and aggregate notes; private data/models/ledgers,
source evidence and credentials remain ignored. No merge into main is implied.

## Common-period cost sensitivity

All rows below cover the same **14 months**, no stops, gross-one initial
allocation and monthly risk observation:

| Arm | One-way cost, bp | 14-month total return | CAGR | Monthly-based Sharpe | Month-end maximum DD |
| --- | --- | --- | --- | --- | --- |
| RF / 5 years | 0 | +24.20% | +20.42% | 0.68 | -18.12% |
| RF / 5 years | 5 | +22.68% | +19.15% | 0.65 | -18.21% |
| RF / 5 years | 10 | +21.18% | +17.90% | 0.62 | -18.30% |
| RF / 5 years | 25 | +16.76% | +14.20% | 0.53 | -18.57% |
| XGBoost / 5 years | 0 | +28.19% | +23.72% | 0.80 | -13.91% |
| XGBoost / 5 years | 5 | +26.83% | +22.60% | 0.77 | -14.00% |
| XGBoost / 5 years | 10 | +25.48% | +21.48% | 0.74 | -14.08% |
| XGBoost / 5 years | 25 | +21.52% | +18.18% | 0.66 | -14.33% |
| RF / 10 years | 0 | +9.18% | +7.82% | 0.44 | -16.52% |
| RF / 10 years | 5 | +7.88% | +6.71% | 0.40 | -16.67% |
| RF / 10 years | 10 | +6.58% | +5.62% | 0.35 | -16.82% |
| RF / 10 years | 25 | +2.79% | +2.39% | 0.21 | -17.27% |
| XGBoost / 10 years | 0 | +36.80% | +30.81% | 1.24 | -11.75% |
| XGBoost / 10 years | 5 | +35.14% | +29.45% | 1.19 | -11.91% |
| XGBoost / 10 years | 10 | +33.51% | +28.11% | 1.15 | -12.07% |
| XGBoost / 10 years | 25 | +28.69% | +24.14% | 1.01 | -12.55% |
