# Forward daily-path targets — engineering version 1

USER DIRECTION, 22 October 2026: implement alternative target workers and a
bounded alignment/coverage audit before comparative model fitting. This is a
V2 research extension. It does not change the locked Han–Qin paper account,
the original endpoint target, existing features, models, selections or runs.

The engineering contract is
[`path_target_protocol_20261022.json`](../configs/path_target_protocol_20261022.json).
Its status is **ENGINEERING_ONLY_NOT_FIT_READY**. The notebook displays saved
results and protocol metadata; Run All never constructs these paths or fits a
model. No additional estimator fit is authorized by this engineering contract.

## What the experiment separates

The existing target is an exact provider-adjusted close-to-close ratio for the
next calendar month, ranked into ten classes. Existing RET is a score formed
from the corresponding class probabilities and causal historical class-return
means. Changing RET selection did not change that supervised target.

The new workers keep three concepts separate:

1. **Original endpoint control:** retain the existing target unchanged.
2. **Executable-path return:** entry at the next XNYS session open after the
   signal, exit at the following monthly rebalance open, under the current
   adjusted-quote/action accounting proxy.
3. **Path utility:** `U = R - lambda * D`, where `R` is that directional cash
   return and `D` is positive maximum drawdown of its unit-initial-capital
   economic NAV. Version 1 uses `lambda = 1` as a fixed engineering candidate;
   it was not selected by looking at target/model/portfolio performance.

The daily marks for `D` are the accounting engine's open and close marks,
including initial NAV 1. This is peak-to-trough drawdown, not maximum adverse
excursion from entry, daily high/low drawdown, a return/drawdown ratio, or a
portfolio drawdown objective. Daily high/low fields only implement the existing
idealized stop crossing. No intraday sequence is reconstructed from OHLC.

Actual cash returns remain separate from utilities. Utility-class historical
means produce **expected utility**, not original RET. Executable cash-return
means conditional on new classes produce a separately named expected-return
score. Old class-payoff vectors cannot be reused for new class definitions.

## Observation clock and target release

A signal at January month-end enters at February's first open. Its normal exit
is March's first open. Consequently that February executable outcome is **not
known at February month-end**. Cross-sectional classes are released only when
the common scheduled horizon is complete; a stock that stopped earlier does
not release its relative class earlier.

Training and inner validation must check actual completion dates as well as
nominal outcome months. An annual fit at December month-end cannot consume
that December's next-open-exit target. Training remains inside the selected
ten-calendar-year outcome range and drops its not-yet-completed boundary; it
must not quietly extend the range or use January's open in a December fit.

The new payoff window is exactly 120 consecutive **completed calendar outcome
months** with all ten class cells. It ends at the most recent month whose
scheduled open endpoint is available at the signal cutoff. An unavailable
latest month must not be skipped in favor of any earlier available month.
With history beginning January 2010, January 2020 predictions cannot yet have
120 completed executable outcome months. February 2020 is the earliest
possible forecast for that payoff window, subject to actual class coverage.
Classifier prediction can exist earlier without that financial score.

December 2023's normal exit requires a protected January 2024 financial open.
Keep that target unavailable. The old backtest's final December 2023 terminal
close is a separate boundary convention and must not silently replace it.
Calendar metadata may establish the next session date without reading its
price. This engineering phase reads no protected 2024+ financial values.
Missing December labels do not remove December signal-time predictions. A
portfolio may have a declared terminal-close valuation without that valuation
becoming its next-open training label. Future comparisons must state their
common scoring/trading period explicitly: a new RET-style score cannot have
a January2020 value when its 120 completed executable months are unavailable.
Do not pad the first unavailable score with zero or compare different periods
as if the only change were the target.

## Population, missingness and directions

Start from every original source-ID/signal key. Preserve historical membership,
provisional identity flags, original denominators, signal-time Paper15
eligibility and all unavailable outcomes. Extra-feature completeness and future
quote availability never select the prediction cohort. Do not select only
previously traded stocks when measuring target coverage.

Rank valid observed outcomes within the original prediction-eligible population
for each month, direction, stop variant and target family separately. Class 1
means the most desirable value for that family and direction. Use the existing
descending exact-equal midpoint convention, without jitter or tolerance-based
tie splitting; fewer than ten observed outcomes withholds all classes. Report
empty classes and reasons. No label completeness threshold is borrowed from
the 80% feature-context rule. Preserve explicit exhaustive request evidence.

Long and short paths are independent, signed, unit-notional economic outcomes.
The zero-cost entry-open wealth must equal 1 within absolute tolerance 1e-12;
reject incorrectly scaled paths instead of rescaling them silently.
Stops, short losses and corporate actions make short utility different from
the negative of long utility. Short losses can exceed initial notional; the
original simple-return `R >= -1` rule is inappropriate for these targets.
No loss may disappear through clipping or a nonpositive-NAV filter. Any
accounting limitation must remain an explicit unavailable-path reason.

This phase does not define a new portfolio selector. A later fitting protocol
must state how two directional forecasts become balanced, disjoint baskets,
including ties and a name attractive in both directions. The existing RET and
class-basket strategies keep their current rules until such a protocol exists.

## Accounting and costs

Reuse the reviewed signed cash/share entitlement, quote-quality, identity,
stop and retired-order rules. A missing quote does not prove retirement or
create an entitlement. Unsupported events stay unavailable. Do not add source
dividends/splits a second time on top of adjusted prices. Source identity,
adjustment semantics, knowledge time, terminal-event coverage, borrow and
investability remain provisional.

Both unstopped and 10% per-asset stopped paths are engineering variants. Stops
use the existing entry-relative threshold, gap-open execution and idealized
intraday threshold fill; stopped capital remains cash until the scheduled exit.
The target outcomes are **gross unit-notional** values with zero trading,
borrow and funding costs. Standalone monthly labels cannot reproduce portfolio
turnover and netting across retained holdings. Keep the actual portfolio's
0/5/10/25bp cost sensitivity and 50%/50% exposure rules unchanged.

## Bounded audit and the fit gate

First inspect a pinned metadata/Parquet-footer read plan. Execute only bounded
full historical monthly cohorts after checking its source projections, dates,
row groups and byte limits. Save a new private audit directory; never overwrite
the original inputs or previous attempts. Measure original and eligible counts,
path/request coverage, direction/stop/failure reasons, endpoint-return alignment,
class changes and actual completion-date exclusions. Report matched observed
label comparisons separately from full denominators. Matched masks are audit
diagnostics, never a trading-universe rule.

The initial adapter deliberately reuses the existing 2020–2023 accounting
kernel. Its isolated constant-value reference leg contributes zero P&L and is
an internal accounting device, not a constituent or research data source.
Convert the real half-notional contribution to unit notional explicitly.
Synthetic controls must never enter original membership, predictors, labels or
published financial observations. Earlier training-year support requires a
separately tested adapter; do not date-shift records or relax frozen guards.

Passing a bounded audit is engineering evidence, **not full 2010–2023 target
coverage or authorization for fitting**. Before a comparative rerun:

- Extend the adapter to training years with explicit protected-date checks and
  verify full original-cohort target/action/missingness coverage.
- Independently check forward returns, signed entitlements, long/short stops,
  drawdown arithmetic, class ranks and completion-aware training/payoff cutoffs.
- Freeze the small RF-expanded target comparison, chronological validation,
  directional score-to-basket mapping and compute budget. Preserve features,
  frozen RF parameters and portfolio assumptions to isolate the target change.
- Refit preprocessing on each explicit supervised training mask; rerank targets
  and rebuild dependent causal payoff states. Do not silently reuse old models
  or class-payoff states after the labels change.

The fit protocol must distinguish the historical published baseline from a
paired refit control using common completed training rows. The new completion
lag and missing daily paths can change the supervised mask as well as the
label. Quantify those changes before attributing a performance difference only
to target design. A matched training diagnostic must not become a hindsight
filter on future prediction/trading cohorts.

Universal identity certification and completion of every comparator execution
path are not prerequisites for all target engineering. Missing terminal returns
can still bias supervised learning, so their prevalence and affected population
must be measured. Keep old results as dated references. The already reviewed
2020–2023 period remains development evidence, not a newly untouched test.

## Reusable worker interfaces

`dmv2.path_targets.build_path_targets` accepts the original cohort, requested
episode status ledger and unit-notional marks, plus independent windows,
XNYS sessions and an observation cutoff. It returns separate target values,
classes, original-denominator diagnostics and scoped numerical fingerprints.
`completed_training_slice` keeps all original slice metadata and returns only
completed observed supervised rows after authenticating their month values.
`build_path_payoffs` retains the full 120-month/class grid and declares whether
the chosen payoff field means executable cash return or utility.
`score_path_probabilities` requires explicit class-column alignment and matching
signal_date/as_of arguments; an unavailable payoff state produces an unavailable
score, not zero. These workers perform no source I/O or model fitting.

From the configured V2 directory, the bounded audit is an explicit command:

```sh
PYTHONPATH=src .venv/bin/python scripts/run_path_target_audit.py plan --project . --output outputs/path_target_audit/next-plan.json
PYTHONPATH=src .venv/bin/python scripts/run_path_target_audit.py run --plan outputs/path_target_audit/next-plan.json --output outputs/path_target_audit/next-audit
PYTHONPATH=src .venv/bin/python scripts/verify_path_target_audit.py outputs/path_target_audit/next-audit
```

Create the parent output directory first. Existing plans, audit directories and
verification receipts are refused, preserving unsuccessful attempts. These
commands retain the fixed January/October2020 audit scope; changing the study
start in the notebook does not expand this audit into a full target build.
The independent checker reads only three bounded generated ledgers and the
reviewed plan, recomputes return/drawdown/utility and exact ranks, and checks
completion dates and original denominators. It does not certify source prices
or independently establish legal entitlements.
