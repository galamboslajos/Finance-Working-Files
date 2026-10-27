# Fixed-forecast monthly/daily reconciliation

USER DIRECTION, 27 October 2026: reconcile the latest monthly model's existing
forecasts and RET baskets through daily accounting, then change execution
timing, costs and stops separately. Keep all four Paper15 RF/XGBoost five/ten-year
arms, with XGBoost ten-year as the diagnostic focus. Zero new models,
preprocessors, forecasts, labels, features, RET means or stock selections.

## Frozen comparison

Use configs/fixed_forecast_reconciliation_20261027.json. All 28 paths cover the
47 outcome months February 2020–December 2023. The initial reference is the
31 January close; 985 following sessions end on 29 December 2023. Every stage
marks terminal holdings without a forced liquidation. Original daily runs'
terminal liquidation is therefore a separately labelled historical convention.

Each arm has seven predefined stages:

1. Daily cash/share accounting at signal close, original cached OC quotes,
   successor effective-close adjustment units, zero costs and no stops.
2. The same close-clock control with 10 bp costs, to check original monthly
   rebalance fees and successor netting as well as gross return arithmetic.
3. Close-clock accounting with the existing repaired execution quote view.
4. The same execution quotes and close fills, changing only the successor
   factor from effective close to effective open.
5. The same open-unit convention, moving monthly fills to the following open.
6. The same next-open path with 10 bp one-way costs.
7. The same next-open path and costs with the existing 10% per-asset stop.

Stages 1→3→4→5→6→7 form the sequential attribution. Stage 2 is an additional
cost reconciliation control. The explicit source/unit controls prevent
different quote vintages or corporate-action conventions from being counted
as a pure execution-clock effect.

Signals, basket keys, probabilities and RET scores stay byte-pinned. Holdings
receive +50% long / −50% short target capital, equal within each side, with the
original ten names per side. Retired new orders keep their selected rows and
unfilled cash slots, with no replacement, redistribution or acquisition
entitlement for a never-filled order. Stops retain the original gap-at-open
and idealized daily high/low threshold fills, with stopped capital in cash.
Zero borrow/funding, fractional successor units and provisional identity,
distribution and historical knowledge-time limits remain explicit.

## Sources and legal terms

Reuse only the independently verified Paper15 daily-history OC cache and the
existing annual execution OHLC tables plus forty reviewed quote overlays.
The plan seals original source, forecast/selection/learning and verifier pins,
all calendar/native projections, output limits and code before daily values.
No raw source fallback, full-universe new scan, cloud mutation or protected
2024+ financial values are authorized by this comparison.

Retain the twenty daily action definitions, eleven prior retirements and exact
dated PENN handover. Add only EV and XLNX's already reviewed legal terms from
the completed monthly merger replay. Existing CDK/XEC terms must agree exactly.
This extends those legal definitions to execution of the frozen monthly
baskets, including the appropriate distinction between a position held before
conversion and a new order placed after conversion. No new primary-source
claim or identity certification is made.

## Daily and monthly boundary accounting

The additive fixed_forecast_accounting worker retains the old engine's
cash/share, stop, net market order and identity equations. It adds explicit
close/open schedules and successor-unit basis controls, and marks terminal
holdings rather than liquidating them. Old workers and runs remain unchanged.

For a close rebalance, the ending month's NAV is the close **before** the next
month's entry trade/fee. Thus month-end reconciliation uses pre-rebalance NAV,
and the following day's return incorporates the next entry fee. Daily output
keeps both pretrade and posttrade NAV so the boundary is auditable. This
avoids moving a fee between months merely because both decisions use one close.

Exact month-end NAV and cash/share/fee controls compare with the independently
verified monthly merger replay at 0/10 bp. Relative tolerance is 1e-9 and
absolute tolerance 1e-10. Disagreements remain explicit diagnostics; no prices,
events or selections are adjusted after seeing results to force equality.
An internal missing mark preserves exposure and withholds continuous-path risk
and performance. A calculation PASS does not certify source completeness.

## Bounded execution and independent checks

The explicit CLI has preparation, independent input verification, replay and
independent accounting verification phases. It caps cache-file reservations
at 512 MiB, execution projections at 128 MiB per pass, worker RSS at 4 GiB,
outputs at 1 GiB, independent derived reads at 2 GiB, each phase at 30 minutes
and total active time at 90 minutes. Numerical source reads occur once in
preparation and once in the independent reconstruction. Integrity hashing is
separate from financial decoding.

The checker reconstructs prepared quote views, original basket projections,
legal-event additions independently. The auxiliary quote-difference inventory
is not independently recalculated or used as a certified result. A separate
cash/share oracle validates actions, targets, net physical orders, stops,
every valid open/close mark, failed-path retention and summary metrics without
importing the producer accounting. Reference discrepancies and withheld paths
remain visible in every report. Cooperative clean-phase pause/resume preserves
completed work; failed attempts are not silently retried. The notebook reader
will expose only authenticated small reports after independent checks pass.
