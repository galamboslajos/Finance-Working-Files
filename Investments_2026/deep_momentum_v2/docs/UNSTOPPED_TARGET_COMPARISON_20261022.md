# Fixed RF comparison of unstopped daily-path targets

The user accepted unstopped learning targets on 22 October 2026 after the full
2010–2023 target audit. The daily backtester retains its separate 10% per-asset
stop. This is a new development experiment; the paper account, original runs,
failed attempts, source corrections and active pointers remain preserved.

The comparison holds the corrected 715-input feature catalog, historical S&P
500/400 union, 80% context policy, provisional identity admissions and prediction
cohorts fixed. Three arms change the supervised outcome:

| Arm | Learning target | Classifiers per year |
|---|---|---:|
| Endpoint control | Original adjusted month-end return classes | 1 |
| Executable cash return | Own-direction return from next open to following rebalance open | 2: long and short |
| Return minus drawdown | Own-direction cash return minus positive open/close-marked drawdown | 2: long and short |

The utility coefficient is fixed at one. It was not selected using model or
portfolio performance. Learning targets are gross, unstopped, unit-notional
outcomes. Finite short losses below minus initial notional remain. Missing
quotes, unsupported successor paths and December 2023's protected next-open
endpoint stay unavailable. No target is imputed. Future outcome availability
never determines a forecast or trade admission.

## Fits, chronology and paired training population

Four annual windows train on the preceding ten calendar outcome years for
2020–2023 monthly forecasts. A shared supervised mask requires original
signal-time eligibility, all five preserved classes and actual completion by
the December signal cutoff. The latest December executable outcome is excluded
until its following January open. The range is not extended backwards to replace
it. Missing supervised outcomes can still introduce selection bias; this mask
is necessary for the paired experiment, not evidence of complete survivorship
accounting.

Classes keep their original full observed eligible monthly ranking. They are
never reranked on the fitting intersection. One shared training-only
median/missing-indicator preprocessor per year feeds all five heads. Prediction
retains every signal-time Paper15-qualified row, including future missing
outcomes. This gives **20 estimator fits and four preprocessing states**.

Every head uses 500 trees, seed 1729 and `max_features="sqrt"`. The previously
selected annual depth/minimum-leaf settings are 12/20, 8/20, 12/100 and 12/100
for 2020 through 2023. There is no new search, inner fit or calibration. These
settings were selected on the original target; transfer does not establish
optimality for new outcomes. Raw log loss across different target definitions
cannot by itself select the target family.

## Causal scores and separate baskets

Each payoff vector requires 120 consecutive completed calendar outcome months,
all ten class cells and equal weighting of monthly realized-class means. Its
population is the full original observed target cohort, rather than the paired
supervised intersection. All arms use a window ending at signal month minus one.

The fresh endpoint arm is **completion-clock-matched endpoint RET**, distinct
from archived original RET. Executable-return probabilities times own-class
cash means form expected executable return. Utility probabilities times
own-class utility means form expected utility; that score is not original RET.
New classes never reuse old payoff states.

January 2020 lacks completed payoff history. All 48 forecast months remain in
coverage ledgers, but the common economic comparison begins **3 February 2020**,
resets NAV to one there and runs through **29 December 2023: 47 monthly outcomes
and 985 sessions**. January's mode selections are outside this common comparison;
its score basket is withheld. Archived 48-month totals remain dated references.

Three selection styles remain separate for each arm:

- **Score:** rank each side by its own expected desirable outcome. Endpoint
  shorts use negative endpoint expected return.
- **Best mode:** endpoint long class 1/short class 10; directional long class
  1/short class 1, since new short classes rank profitable shorts first.
- **Second mode:** endpoint long 2/short 9; directional long 2/short 2.

Stable source ID resolves exact score/probability ties. Reserve ranked longs
first, then choose disjoint shorts. Score breadth is `min(10, floor(N/2))`.
Mode breadth is the greatest count up to ten whose top-ranked longs leave the
same count of eligible shorts. The fixed DF/RDC execution restrictions apply
before ranking. There is no positive-score filter, future quote requirement or
realized-target selection. Zero breadth withholds the path.

## Execution, resources and independent verification

The unchanged accounting engine receives balanced selections: 50% long/50%
short, equal weights within each side, next-session open execution, continuous
NAV across annual model changes and original action/quote/identity guards.
Unstopped execution and 10% per-asset stops are reported at 0/5/10/25 bp per
one-way traded notional, with unchanged zero borrow/funding proxies. Stopped
capital stays cash until rebalance. Three arms × three baskets × two stop
variants × four costs produce **72 paths**. Missing held observations preserve
exposure and withhold full-period metrics; no stock is dropped to complete a
result. The stated focus is the score basket with stops and 10 bp costs.

The frozen local protocol allows one sequential process, four fitting threads,
one inference thread, 16 GiB RAM, three active hours, 20 estimator attempts and
four preprocessors. Cumulative disk-read limits are 4 GiB of model source
projections, a separate 1 GiB backtest source allowance, and 32 GiB of derived
reads, with at most 16 GiB reserved for each independent
verification phase. These are read totals, not RAM requirements. Output is
capped at 24 GiB. No automatic retry or new target/tuning search is permitted.

The independent verifier seals metadata, artifact pins and read budgets before
financial values. It checks source-derived target classes, original keys,
actual completion, common masks, training-only schema/medians, saved float32
matrices, RF classes/parameters, exact saved-model probability reproduction,
summary and monthly probability metrics, full confusion counts and class
precision/recall/support. It reconstructs causal
payoffs and disjoint signal selections separately, then uses the established
independent quote-overlay and signed cash/units equations to verify all 72 paths
and their daily-derived performance. Fitting is never called by verification.

Source feature formulas are inherited from the pinned corrected-input proof,
not independently reimplemented in this experiment. No check certifies every
historical identity, provider adjustment, entitlement, missing terminal return
or investability assumption. The 2020–2023 period is already examined development
evidence; protected 2024+ financial observations remain unread.

The master notebook remains passive. Run All authenticates saved results and
never starts fitting or replay. Explicit run checkpoints support pause/resume
between heads, preserve completed work and charge consumed attempts. Hard-killed
fits are not automatically repeated.
