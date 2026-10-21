# RET selection baseline — 20 October 2026

Status: frozen additive execution scope; numerical payoff coverage, replay and
independent verification must complete before any results are called verified.
The user directed prioritizing the originally intended RET strategy before
alternative labels. Existing class-basket backtests remain reference strategies.
Their published performance does not describe a RET-ranked portfolio.

## What changes

Only the probability-to-stock-selection layer changes. Keep the current
source-corrected ten return classes, features, historical S&P 500/400 cohorts,
80% primary context, training-only preprocessing and saved annual probabilities.
Use all ten probabilities to calculate RET as their weighted sum of historical
realized-class payoffs. No new fit, inference, feature/label rebuild, calibration,
parameter search or alternative path-quality label is part of this experiment.

Evaluate the original four arms (RF/XGBoost on Paper15/expanded715) and the
verified tuned RF-expanded forecasts. There are five RET arms, each with two
stop choices and four cost settings: forty predefined scenarios. RF expanded
with 10% asset stops and 10 bp one-way costs remains the main reporting focus;
show both original and tuned RF and the other models without discarding
withheld comparisons.

## Historical payoffs and clocks

At each of the 48 signals from 2019-12-31 through 2023-11-30, use exactly the
preceding 120 completed calendar months, ending in the signal's calendar month.
For each realized class and month, calculate the equal-weight mean of observed
label returns; average the resulting 120 monthly means with equal time weights.
Payoffs use actual realized classes, never predicted-mode groups. Every one
of the 1,200 month/class cells must be available; otherwise withhold the whole
vector. Preserve missing outcomes and original denominators. Do not shorten,
pool, forward-fill or impute the history.

The first January 2020 forecast therefore uses January 2010–December 2019
outcomes. December 2019's completed outcome is keyed by the November 2019
signal, not the December signal predicting January 2020. Across all 48 signals,
only 167 outcome months (January 2010–November 2023) are needed. December 2023
outcomes are unnecessary for scoring. The independent XNYS calendar and exact
outcome-end cutoffs must precede financial conversion for each selected month.

Recalculate the payoff states from the exact corrected annual target lineage
used by the saved classifiers, authenticated through their completed parent
identities and immutable pins. The old full-target-panel RET states predate
later source/BBBY corrections and are not admissible simply because they once
passed a 48-vector readiness check. Reuse corrected target_class/label_return;
do not rerank labels or manufacture a new learning population.

Arrow may decode mixed compressed column blocks while extracting a requested
month. Filter to authorized rows before financial Python conversion; do not
claim per-cell physical isolation. Never decode protected 2024+ financial data.
Saved shared payoffs must be sliced causally for every signal, and later
outcomes must not affect an earlier score. Same-close data availability remains
a historical-reconstruction assumption, not knowledge-time certification.

## Selection

Rank all original prediction-eligible forecasts after the unchanged DF/RDC
execution restriction, with no predicted-class filter and no later quote or
label-availability filter. Keep every original row, identity state and admission
flag in the ledger. The existing wrong-BBBY quarantine remains in force.

For M trade-eligible names, N=min(10, floor(M/2)). Select longs by RET descending,
then stable source ID ascending. Reserve those N names, then select shorts by
RET ascending/source ID ascending from the remaining names. This preserves the
existing generic RET tie convention and guarantees disjoint equal-sized sides.
If N is zero or the payoff vector is unavailable, withhold that rebalance and
continuous path. Report exact cutoff ties and zero score spread. A flat score
retains the inherited stable-ID convention rather than introducing a new
performance-dependent filter. No positive/negative RET sign gate is added.

Ten names per side is the user's V2 breadth adaptation. The paper uses highest
and lowest ten percent; this experiment is not a paper-exact breadth replication.

## Daily execution and coverage

Preserve +50% long/−50% short initial monthly capital, equal within-side weights,
next-session-open execution, monthly rebalance, drift between rebalances,
continuous NAV without annual liquidation, and terminal close liquidation on
2023-12-29. Keep unstopped and fixed monthly-entry 10% per-asset stop variants;
stopped proceeds stay cash until the next rebalance. Keep the current idealized
OHLC threshold/gap-open fills, signed corporate-action entitlements, eleven
historical retirements and exact previously approved PENN execution handover.
No canonical identity promotion, new exclusion or fabricated exit is permitted.

Costs remain 0/5/10/25 bp per one-way traded notional; borrow and funding remain
zero. The horizon is 2020–2023, 48 months and 1,006 sessions. New selections need
their own maximum-holding-horizon quote preflight. Missing or rejected held
marks retain exposure and withhold full-period metrics. They do not authorize
replacing a selected name with a better-covered name. The frozen scope permits
explicit fail-closed replay of such paths after documenting the coverage gaps.

## Resource and preservation contract

The JSON contract is configs/ret_backtest_20261020.json. Use a fresh private run
under outputs/ret_backtest and separate controls under outputs/ret_backtest_control.
Preserve all existing source files, protocols, forecasts, models, runs, pointers,
failed attempts and pause/resume checkpoints. No cloud operations are included.

Use one process and one numerical thread. The separate limits are 4 GiB aggregate
process-tree RSS stop/3 GiB warning, normal macOS pressure plus 7 GiB available
memory and 30 GiB free disk at dispatch, 1 GiB cumulative conservative compressed
read reservations and 512 MiB total new run/control outputs. Phase ceilings are
five minutes for preparation, fifteen for payoff/selection preflight, twenty
for daily replay and sixty for independent verification. The external watchdog
samples each second; these limits can overshoot and are not an OS reservation.
Failures preserve receipts and consumed budgets; no automatic retry is allowed.

Calculate and reserve exact column/read-pass projections before conversion.
Byte hashing is separately identified integrity I/O. Metadata planning may not
be presented as numeric payoff availability or successful access to a full path.

## Verification and reporting

Synthetic tests cover chronology, 120-month availability, probability alignment,
full-distribution ranking, ties, disjointness, fixed restrictions, missing paths
and resource supervision. The independent actual verifier must reconstruct
monthly class means, all 48 payoff windows, scores and selections without using
production selection/RET arithmetic as its expected answer. It separately checks
cash/units, fees, actions, stops, metrics and retained failed exposures using the
preserved independent accounting equations. Keep raw artifact pins and exact
semantic frame checks; harmless Parquet storage-width changes must never permit
value/order/null differences.

Report RET alongside saved class-basket comparisons, selection overlap, score
and payoff coverage, costs, annual slices and all withheld paths. A complete
subset cannot establish a full five-arm economic ranking. The already examined
2020–2023 interval remains development evidence. Identity, knowledge time,
observed-label selection, missing terminal events, retrospective DF/RDC
restrictions, borrow availability and idealized stop fills remain limitations.

The master notebook remains passive. Explicit workers and CLIs produce a new
private experiment; opening or running the notebook must not launch it.
