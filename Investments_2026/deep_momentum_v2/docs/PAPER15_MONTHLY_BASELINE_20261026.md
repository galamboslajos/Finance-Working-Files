# Paper15 calendar-month label and monthly-return backtest

USER DIRECTION, 26 October 2026: run the paper-style monthly return label and
a backtest that assumes those monthly returns can be earned as stated. This
is additive to the completed executable-return baseline on the same separate
`codex/paper15-ret-5y-10y` branch. All old runs and worker bytes are preserved.

## Frozen scope before numerical reads

Use `configs/paper15_monthly_baseline_20261026.json`. Retain the four RF/XGBoost
by five/ten-year arms, original historical S&P 500/400 membership, matching
complete past daily OHLC eligibility, the same 15 finite paper inputs,
cross-sectional normalization, seeds and fixed estimator settings. Reuse the
previous independently verified signal features and eligibility byte-for-byte.
There are exactly 16 new annual fits for 2020–2023, no tuning, no imputation,
no expanded inputs and no automatic model execution from the notebook.

This tests the paper's monthly return convention. It is not a full replication:
the paper uses broad CRSP, 16 inputs including size, expanding classifier
history, 100-fit annual XGBoost ensembles and full top/bottom-decile baskets.
Our existing source/universe, no-capitalization, rolling-history, one-fit,
forest, top-ten and gross-one adaptations remain. No new outlier filtering or
winsorization is introduced by this comparison. The paper's learning-sample
cleaning must not silently replace economic returns with winsorized values.

## Target, completion and RET

For the original cohort at signal month-end t, define the next calendar month
return as `adjusted_close[t+1] / adjusted_close[t] - 1` at exact exchange
month-end observations. Use the pinned verified monthly close cache. No stale
last-observed substitution, daily execution overlay, separate action payout,
double dividend addition, fill, clipping or missing-to-zero substitution.
Provider-adjusted ratios are research monthly returns, not certified CRSP
total returns, delisting payouts or executable cash flows.

Rank observed eligible returns into ten exact-tie midrank classes (1 best,
10 worst). Keep every original cohort row and every signal-qualified forecast
when a later outcome is unavailable. Missing future values never screen X.
Retain the previously accepted provisional identities and quarantines.

The label is complete at the outcome month-end close, including December.
Each annual fit uses the preceding N calendar outcome years, restricted to
labels completed by the preceding December close. No next-January-open purge
is appropriate for this monthly definition. At each signal, RET uses exactly
60 or 120 completed outcome months ending in that signal month. Every month
and all ten actual return-class means are required; average monthly means
equally through time, then dot with all ten aligned predicted probabilities.
The current signal's next-month return never enters its payoff vector.

## Idealized monthly economics

Assume the signal, annual fitting, monthly predictions and RET calculation
can all be reflected in an investment at that same month-end close. This is
an explicit idealized availability/fill convention, not a claim that closing
features can be calculated and traded at the already observed closing price.

Keep RET top ten / bottom ten, disjoint stable-ID ties, matching available
counts and +50% / -50% equal weights within each side. Keep the existing
DF/RDC execution restrictions, without new outcome-based exclusions.
The portfolio directly earns the signed weighted monthly source returns.
Use continuous NAV across annual refits, drifted month-end return notionals
for one-way turnover, and 0/5/10/25 bp costs per traded notional. Targets use
pre-trade NAV; fees reduce cash without retrospectively resizing the orders.
Borrow/funding are zero. The terminal portfolio is marked, not forcibly
liquidated. A missing selected monthly return withholds the entire continuous
period; never drop the stock, redistribute its weight or assume a payout.

No intramonth stops are simulated. Monthly returns cannot identify threshold
crossings, gap fills or post-exit paths. This monthly baseline is no-stop only;
the existing daily stopped/no-stop results stay separate references.

The primary period is all 48 months January 2020–December 2023. December 2023
closes are sufficient; protected January 2024 prices are unnecessary and stay
unread. Also replay February 2020–December 2023 (47 months), starting flat at
that February signal, to match the old calendar. Four arms × two fixed periods
× four costs give 32 monthly scenarios. The extra period changes no fits.

Report 12 × monthly mean, sqrt(12) × sample monthly SD, zero-cash-rate Sharpe,
geometric CAGR and maximum drawdown observed at monthly boundaries. These are
monthly-frequency risk measures and cannot reveal daily drawdown or support
a claim that intramonth risk was reduced. Prediction metrics use all available
48 label months and are not numerically the same sample as the prior 47-label
daily-target evaluation. The change bundles labels, completion, payoff clock
and economic return convention; it does not isolate execution alone.

## Bounded execution and independent checks

Read only the parent proof/metadata, projected signal columns, and 3.3 MiB
monthly-close array. Do not read raw/daily/expanded financial panels or models
from prior runs. The plan pins exact files, code, projections, byte reservations
and protocol before decoding source values. Per-pass input ceiling is 64 MiB;
derived verification reads 2 GiB, peak worker RSS 8 GiB, output 4 GiB, one thread,
600 seconds per fit, 1,800 seconds per phase, three active hours total.

Before fitting, an independent checker authenticates the parent source/feature
proof, checks exact feature/eligibility reuse and reconstructs all new monthly
returns, classes, completion dates, coverage and causal payoff grids. It then
reloads all 16 new models to reproduce inference and independently reconstructs
probability scores and class confusion reports. A final independent cash/notional
oracle checks RET selection, monthly costs/turnover, contributions, NAV, missing
return withholding, annual slices and all 32 aggregate scenarios. It does not
import the producer's monthly arithmetic or fit a model.

Use the explicit `scripts/run_paper15_monthly.py` plan/run/pause/resume CLI.
Fit attempts are consumed once; a failed attempt is preserved and not retried
automatically. Clean checkpoint pause/resume is supported. The controller
checks resources between and during phases. The standalone monthly notebook
will read independently verified small reports only. Keep source data, models,
forecasts, receipts and per-stock positions private. No cloud operations,
protected financial 2024+ values or changes to the original master are included.


## Completed run and explicit missing-return decision

The independently checked run completed all16fits and32monthly scenarios.
Every full48/matched47 path is withheld for at least one missing selected
monthly return. The user explicitly chose to preserve this strict source
definition and report the common available period, without legal-action
return reconstruction or a dependent refit. The separate report uses the
existing verified ledgers for January2020–February2021 (14months), chosen from
the earliest failure metadata before prefix return aggregation. It adds no
fits, fills, strategy trials or backtests. This is a post-run coverage diagnostic,
not the full four-year outcome or an untouched test. See
[the exact results and all-selection missingness audit](PAPER15_MONTHLY_RESULTS_20261026.md).
