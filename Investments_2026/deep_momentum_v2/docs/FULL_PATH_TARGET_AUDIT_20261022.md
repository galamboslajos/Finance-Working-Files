# Corrected full-history daily-path target audit — 22 October 2026

Status: **COMPLETE; independent formula/rank/clock/denominator checks passed.**
This is a separately versioned 2010–2023 target audit. It fits no model, changes
no portfolio rule and preserves the original endpoint targets and prior runs.
The earlier January/October 2020 sample remains a bounded engineering reference.

## Which corrected data this uses

The source is the installed corrected-refit catalog at
`outputs/corrected_refit/2020-2023-20261021-v1/input_catalog.json`, SHA-256
`7ebc434ae394410ff5f5377a538b0e43eb290d6685f7f9659ef098d8378c7394`.
Its fourteen annual inputs and 48 original RET states match the independently
checked BK/XOM shadow catalog byte for byte. The source chain retains:

- Historical S&P 500/400 membership, original source/signal keys and the existing
  provisional identity policy; both known wrong BBBY routes stay quarantined.
- The El Paso/Empire issuer-route correction and fifteen OPCH OHLC repairs.
  OPCH volume on those repaired bars remains unavailable.
- BK's reviewed computational route and XOM raw/all SIP observations from 2016.
  XOM's pre-2016 history remains missing and its 69 precision warnings survive.
- The accepted 80% feature-context rule, daily feature histories and unchanged
  Paper15/expanded feature definitions. Target coverage does not reuse that 80%
  feature threshold.
- Rejection of the proposed RCM/DISCA learning patches by the frozen volume
  controls: **zero further learning bars were admitted**. The corrected-refit
  input verification reused the BK/XOM annual inputs without another rebuild.

These are versioned corrected inputs with documented arithmetic and preservation
checks. They are not a claim that every quote, identity, adjustment vintage or
historical knowledge-time assumption has been independently certified.

## Primary paths and separate execution evidence

The primary audit uses the corrected **learning** daily caches. It preserves
152,307 original constituent-months, 168 outcome months and 140,931 signal-time
prediction-eligible rows in the reviewed metadata plan. The original endpoint
control has 140,638 observed labels; it is read for comparison, not reconstructed
or overwritten by this audit. Its 29 known event-overlap labels remain missing.

The action registry carries 29 reviewed events: 18 later actions and eleven
2018–2019 actions with their cash/share terms. The full-history adapter applies
applicable old terms as in-period actions, rather than using their stripped
retired-order metadata as a substitute for entitlements. The registry is not a
complete corporate-action inventory for 2010–2017. Unsupported paths, missing
successor observations and unresolved identity conflicts remain unavailable.

Forty earlier execution-only quote amendments—22 RCM, five DISCA and thirteen
ADS observations—remain a **supplemental diagnostic**. A separate quote index
replays potentially affected original episodes, then recomputes ranks over the
full original cross section. It does not mutate the primary prices, promote
these amendments into learning data, change prediction eligibility or silently
combine the diagnostic target values with the primary target panel.

## Target definitions and completion clock

The original provider-adjusted monthly close-to-close target remains the
control. New paths enter at the next XNYS open after the monthly signal and
exit at the following monthly rebalance open. Both long and short paths are
calculated independently, with unstopped and fixed 10% per-asset-stop variants.
They are gross unit-notional outcomes: trading, borrow and funding costs are
zero. Existing portfolio cost sensitivities and 50%/50% basket rules are unchanged.

The new target families are executable cash return and
`cash_return - 1.0 * maximum_drawdown`. Drawdown is the positive peak-to-trough
loss of the signed economic path's unit-initial-capital NAV, observed at opens
and closes and including initial NAV 1. High/low fields support only the
existing idealized stop rule; no intraday price sequence is invented. Finite
short losses below −100% and drawdowns above 100% remain explicit, rather than
being clipped or excluded merely for nonpositive wealth. Lambda 1 is a fixed
engineering candidate, not a choice selected from performance.

All original rows remain present. Classes use valid observed outcomes within
the signal-time eligible population, separately by month, side, stop and target
family. Exact ties retain the midpoint convention. Missing outcomes are never
imputed; fewer than ten observed outcomes withholds that month's classes. Ties
can leave classes empty even when a larger observed population is available.

An outcome stopped early still waits until its common scheduled next-open
endpoint for cross-sectional release. December labels are unavailable at the
preceding December month-end annual fit and must be purged from its supervised
rows without extending the ten-calendar-year range. A new payoff window needs
120 consecutive completed calendar outcome months and all ten class cells;
missing latest cells cannot be skipped. With history starting in January 2010,
February 2020 is the earliest possible executable-payoff forecast, subject to
actual coverage.

December 2023 requires the protected 2 January 2024 open. That endpoint remains
unread and the target unavailable, including for early-stopped episodes. The
old portfolio backtest's terminal December close is not substituted. Calendar
metadata may identify the next session without reading its price. The plan
includes thirteen adjacent-annual first-open projections and identifies 25
requested boundary observations missing from those caches; cache absence does
not prove the provider lacks the historical quote.

## Actual coverage and verification

The completed primary audit retains **152,307 original stock-month rows and
140,931 signal-time eligible rows over all 168 months**. Each row has two sides,
two stop variants and two new target families: 1,218,456 target rows in total,
including every unavailable/ineligible state. There are 1,115,498 available target
values. Coverage below uses the 140,931 eligible original rows as its denominator;
it is the same for cash-return and utility targets, whose class values differ.

| Direction | Path variant | Available paths/labels, each family | Eligible coverage |
|---|---|---:|---:|
| Long | Unstopped | 139,414 | 98.92% |
| Long | 10% asset stop | 139,466 | 98.96% |
| Short | Unstopped | 139,414 | 98.92% |
| Short | 10% asset stop | 139,455 | 98.95% |

The 609,228 distinct directional/stop episodes comprise 557,749 complete,
45,504 unrequested because of existing signal-time ineligibility, and 5,975
unavailable eligible episodes. Counts below are episodes, not unique stocks or
new exclusions, and must not be doubled again for the two target families.

| Unavailable reason | Episodes |
|---|---:|
| Protected December 2023 next-open horizon | 3,548 |
| Requested cached observation missing | 1,293 |
| Existing source-quality flag | 1,106 |
| Retired before entry; unfilled, not assigned a zero payoff | 20 |
| Unsupported successor identity/quote basis | 8 |

December 2023 preserves all 909 original rows, including 887 eligible members.
Its 3,548 eligible directional/stop episodes remain unavailable; no January 2024
financial observation was read. The other 22 members remain signal-ineligible.
Twenty-five requested cross-year endpoint observations were absent from the
caches; this is an inventory diagnosis, not proof that no quote can be obtained.

### Ties and payoff readiness are distinct from label coverage

The exact-equal rank rule leaves **230 empty cells among 13,440 month/class
cells**. The largest exact-value group has 661 members. Labels are still
available for valid observed rows; an empty class does not erase those labels.
It does withhold a score requiring all ten historical class-payoff cells.

Across all 168 signals and eight side/stop/family groups there are 1,344 candidate
120-month states; 188 have a complete class grid. Most early states cannot have
120 completed outcomes yet. Within the intended 48 outcome months of 2020–2023:

| Target family | Variant | Complete states per direction /48 |
|---|---|---:|
| Executable cash return | Unstopped | 47 /48 |
| Return minus drawdown | Unstopped | 47 /48 |
| Executable cash return | 10% asset stop | 0 /48 |
| Return minus drawdown | 10% asset stop | 0 /48 |

January 2020 lacks the required completed executable history; the first possible
unstopped state is February 2020. For stopped targets, exact ties leave missing
class cells in every otherwise history-ready 120-month window. This is a real
obstacle to the unchanged ten-class payoff-scoring rule, not missing target
values or permission to jitter ties, shorten history or alter bins after seeing
results. A later experiment must explicitly decide whether to learn unstopped
path targets while retaining stops in execution, or adopt and justify a different
stopped-target scoring contract. No choice is promoted by this audit.

The 32 annual side/variant/family training summaries apply the actual next-open
completion gate. For the 2020 fit, 872 otherwise observed December 2019 labels
per group are withheld until their scheduled January open; completed supervised
counts range from 97,253 to 97,298. These are target-availability counts, not
model fits or a new forecast/trading cohort. The exact annual summaries remain
in the saved report and notebook.

### Supplemental diagnostics and signed losses

The execution-only overlay changes diagnostic availability in June, October and
November 2021 and April 2022. Each affected month replays four directional/stop
episodes for one original member and changes availability in eight target-family rows. The
full-cohort rerank changes 31, 41, 58 and 41 class entries respectively. No shared
previously available numerical target changes in this diagnostic. These outputs
remain separate; primary labels and source learning files are unchanged.

One April 2020 episode would have triggered the old portfolio capital gate.
The standalone signed path retains three negative reference-NAV marks with an
explicit diagnostic. Finite severe short losses and drawdowns above one are not
clipped or discarded solely because wealth becomes nonpositive. This retention
does not certify that its underlying provider observations are correct.

### Independent check and resource record

The separate verifier returned
`PASS_FULL_FORMULAS_RANKS_CLOCK_AND_ORIGINAL_DENOMINATORS`, bound to audit SHA-256
`8f09fec5ed2b0404f9518ef4b6be58e50a4a1a7751e9768ed7e49fe320124547`
and plan identity
`sha256:5e29693f8f2b41b65e1663932255beb7a8e773cb13fa108bf0bdb2f426fba825`.
It checked all 168 months, 1,218,456 target rows, 1,344 payoff-readiness states and
32 training-completion summaries. The independent arithmetic uses saved derived
marks/targets, not a fresh provider-price or entitlement reconstruction. It read
no source prices, features or models and makes no identity/action certification
claim. It took 51.49 seconds; no model was fitted or portfolio backtest run.

Earlier calibration receipts and their arithmetic-order reconciliation remain
preserved; the frozen successor control matched all 308,059 compared values
exactly. This separate calibration history does not alter the full-audit proof.

The materialization took 239.53 active seconds and peaked at 991,248,384 bytes
(about 0.92 GiB) process RSS. The recorded materialized package size was 230,719,783
bytes before later verification/calibration/installation metadata. The reserved materialization
projection was 221,621,837 bytes plus 4,610,428 bytes of planning metadata;
no reservation was reset. All models, original targets and previous runs remain
unchanged.

The sealed V3 metadata plan covers all fourteen years and 168 months. It reserves
226,232,265 compressed source-projection bytes against a 512 MiB ceiling; this
is a planned conservative reservation, not a measurement of physical disk I/O.
Integrity hashing is separate. The execution ceiling is 45 active minutes,
4 GiB process RSS, one worker, 512 MiB output per year and 8 GiB total output.
Immutable monthly and annual checkpoints preserve completed work and charged
read/time budgets across explicit resume. No automatic retry or budget reset
is implied by a failed attempt.

The passive notebook reader authenticates the saved plan, exact monthly/annual
receipt chain, denominator/key identities and artifact bytes without decoding
financial tables. This is distinct from an independent numerical target audit.
A successful formula/coverage audit does not supply unresolved prices,
certify every corporate action or establish an investable return dataset.

## What can follow this audit

The intended small model comparison has the same expanded features and frozen
RF settings, with three target arms: original endpoint control, executable
cash return and return-minus-drawdown utility. It is not another RF tuning search.
Before fitting, name the frozen RF configuration to reuse and freeze its
supervised masks, completion-aware rolling ten-year
and inner-validation cutoffs, compute budget and directional score-to-balanced,
disjoint-basket mapping. Refit preprocessing only on each arm's completed
training rows; retain monthly prediction timing and the existing daily backtest
rules to separate the effect of the target definition.

Old RET probabilities/payoff states cannot be applied to new classes. Cash
means conditional on new classes estimate a separately named expected return;
utility means estimate expected utility, **not original RET**. No fitting,
inference, new portfolio replay, active-pointer promotion or automatic next
experiment is authorized by completion of this audit. The already inspected
2020–2023 period remains development evidence; protected 2024+ values stay unread.

See [the target engineering contract](PATH_TARGET_ENGINEERING_20261022.md),
[the frozen full-audit protocol](../configs/full_path_target_audit_20261022.json)
and the latest [corrected-input RET comparison](CORRECTED_BACKTEST_COMPARISON_20261021.md).

Private installed location: `outputs/full_path_target_audit/2010-2023-20261022-v1/`.
`audit.json` is the completed aggregate record; `reviewed_plan.json` identifies the
exact read/source/worker contract; `independent_full_verification.json` is the
separate proof. The notebook uses these saved records only. Source prices,
per-stock target/mark panels and proof bodies remain private and are not supplied
by a Git clone.
