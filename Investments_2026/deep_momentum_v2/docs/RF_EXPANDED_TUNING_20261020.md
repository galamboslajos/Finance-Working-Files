# Frozen RF-expanded tuning protocol — 20 October 2026

**FROZEN DESIGN ONLY. No fitting or backtesting is authorized by this file.**
The user requested a small chronological search selected by log loss, with the
existing daily backtest rules preserved and a realistic budget for this local
32 GiB computer. The contract is
[`configs/rf_expanded_tuning_20261020.json`](../configs/rf_expanded_tuning_20261020.json),
protocol `dmv2-rf-expanded-temporal-tuning-20261020-v1`.

This task saves the design and checks its metadata. It creates no executor and
does not fit preprocessing or models, reload estimators, prepare matrices,
decode financial panels, generate forecasts or run a backtest. The master
notebook is unchanged. A later execution instruction and a reviewed additive
runner are required. Historical workers and their old budgets stay frozen.

## 1. Research scope and reference

USER DIRECTION: tune only RF on the expanded 715-candidate feature bank for the
historical S&P 500/400 union, with primary 80% context, rolling ten-year outer
training, annual refits and monthly probabilities for outcome years 2020–2023.
Keep Paper15 formulas, target classes, provisional identity admission,
missingness and signal populations unchanged. No capitalization is used.

Use the reviewed source-corrected annual input map, including the El Paso route
correction and OPCH bar repair. Do not use the obsolete uncorrected panels or
the wrong BBBY routes. The fixed BBBY quarantine, DF/RDC trading restrictions
and provisional source limitations remain in effect. Training rows for valid
DF/RDC history remain available; their trading restriction is separate.

The reference is the completed corrected run
`outputs/walkforward_source_corrected/2020-2023-20261016-v1`, build identity
`sha256:559ba2f7fbe187f192b40af2ab8fbe0a4fd70a1b843e2cbd0503f7e35729fcae`.
The new JSON pins its identity, completion, model/input/accounting audits and
verification receipts, the corrected annual source map and frozen recipes.
Recheck these before future financial decoding. See
[the corrected handoff](SOURCE_CORRECTED_WALKFORWARD_20261016.md).

PAPER versus V2: the locked paper uses expanding classifier history, XGBoost,
random historical 80:20 validation and a 100-fit probability ensemble per
annual date. This experiment is a V2 temporal RF tuning adaptation. Its 100
prospective fits are a search/refit count, not that paper ensemble. RET is not
introduced by this experiment.

## 2. Chronological inner validation

For outer outcome year `Y`, keep the final training window `Y−10 … Y−1`.
Within that window use its last three completed years as separate, forward
validation blocks. Each inner training partition starts at `Y−10` and ends
before its validation year: seven, eight and nine outcome years respectively.

| Outer outcome year | Full outer training | Inner training → validation outcome year |
|---|---|---|
| 2020 | 2010–2019 | 2010–2016 → 2017; 2010–2017 → 2018; 2010–2018 → 2019 |
| 2021 | 2011–2020 | 2011–2017 → 2018; 2011–2018 → 2019; 2011–2019 → 2020 |
| 2022 | 2012–2021 | 2012–2018 → 2019; 2012–2019 → 2020; 2012–2020 → 2021 |
| 2023 | 2013–2022 | 2013–2019 → 2020; 2013–2020 → 2021; 2013–2021 → 2022 |

The shorter inner histories are an explicit tuning compromise; the selected
annual outer fit still receives all ten years. Every validation block contains
12 complete outcome months. Keep each complete monthly cross-section together;
never randomly split stock rows. Earlier inner validation years can enter the
next inner training set after completion, as chronological retraining allows.
This freezes the new three-block design; the earlier two-year inner-validation
proposal in `MODEL_COMPARISON.md` remains historical and is not this protocol.

Features end at the preceding exchange month-end close. Targets are admitted
only after their recorded outcome end and by the last XNYS session on or before
the training year-end. Assert `max(train.outcome_end) <= min(valid.signal_date)`
and `max(train.outcome_end) < min(valid.outcome_start)`. Purge and record any
target interval that crosses these boundaries before fitting; never compress
missing calendar months or shift labels to make a split fit.

The completed December label may share the December close with January's
signal under the existing after-close fit / next-session-open convention. No
extra calendar-month embargo is imposed. Shared trailing feature history does
not by itself justify deleting otherwise completed labels; future target
interval overlap does. Record actual exchange cutoffs and interval assertions.
This preserves the existing convention, including its practical after-close
data/compute availability assumptions; it does not certify provider knowledge
time.

Process outer years in order. Seal that year's winner and training/input key
hashes before decoding its outer financial values for forecasts or evaluation.
Do not preload all 2010–2023 financial values into the tuning process. A later
outer year may use an earlier outer year's now-completed outcomes, but cannot
change the frozen grid or earlier choices.

## 3. Cohorts, preprocessing and probabilities

Use the same target-aware training adapter and observed completed-label mask
as the corrected baseline. Require all ten classes in each training partition.
Keep original historical membership keys, identity flags and denominators.
Predict every signal-time Paper15-qualified validation row, including rows
whose future outcome is missing. Score only the identical observed-label keys
for all eight candidates. Missing labels are reported and never imputed.

Fit the active `train_partition_median_missing_indicator_v1` separately on each
inner training partition, then share that state and its matrices across the
eight candidate fits. Drop all-missing/constant-finite columns and calculate
finite medians using only those training rows. Include one missingness flag per
retained predictor. Apply the frozen state to validation; do not refit on it.
The final outer winner receives a fresh preprocessor fitted on its ten-year
training window. An outer-fitted schema or imputer must not enter inner CV.

The candidate dictionary stays 715, including Paper15; a training-only drop
may reduce a split's prepared width below 1,430. Do not force all-history column
retention or remove a prediction row because added features are missing. No
extra z-scoring, clipping, winsorization, coverage threshold search or feature
selection is added. Signal-local normalization still uses only that signal's
past inputs. Daily history and daily execution data remain separate.

Keep finite float32 matrices and the existing float64 ten-class probability
contract. Persist native class mapping and align classes 1 (best) through 10
(worst). Reject invalid probabilities without repairing or renormalizing them.
Mode ties keep the lowest canonical class ID. Use one thread for probability
inference/reproduction; fitting may use four threads. Record all effective
parameters and the runtime. The four-thread fit is not claimed bit-identical
to the historical single-thread baseline.

## 4. Eight candidates and fixed parameters

Use the full Cartesian product, fixed before any new results:

| Parameter | Values |
|---|---|
| `max_depth` | 8, 12 |
| `min_samples_leaf` | 20, 100 |
| `max_features` | `sqrt`, 0.05 |

Every fit has **500 trees**, seed **1729**, Gini splitting, bootstrap enabled,
`min_samples_split=2`, no class/sample weighting, no OOB selection, no warm
start, no calibration and no early stopping. All other effective RF settings
are explicit in the JSON. The existing statistical baseline
`depth=12 / leaf=20 / sqrt` is included as `rf_d12_l020_sqrt`; it runs first in
each split, followed by candidate IDs in ascending order. This is a bounded
regularization check, not an exhaustive RF search.

The 0.05 fraction applies to all prepared columns, including missingness flags.
At the historical 1,430-column width its nominal subset is 71 columns versus
37 for `sqrt`. It samples predictors at each split, not a permanent 5% feature
screen. The fraction limits this first search's cost; the library may inspect
more columns when necessary to obtain a valid partition. Four `n_jobs` threads
parallelize trees. These meanings follow the
[official RF parameter documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).

## 5. Selection by log loss only

For a candidate and validation month, calculate the stock mean of
`−ln(max(probability_of_observed_class, 1e−15))` on the shared observed-label
mask. Its score is the **equally weighted mean of all 36 monthly losses** from
the three validation years. Months with more stocks receive no extra weight.
This is also the equal mean of the three complete annual scores.

Use the project's existing log-loss calculation and evaluation floor rather
than silently substituting library defaults. The floor affects the logarithm
only; saved probabilities stay native. The
[library metric](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html)
has dtype-dependent clipping, so a default scorer is not the frozen project
scorer. A validation month with no observed label makes selection incomplete;
an individual class absent in a validation month is reported, not an excuse to
change masks or class mapping.

Choose a separate lowest-loss candidate for each outer year. Scores within
absolute `1e−8` of the minimum form the numerical tie set. Within it prefer the
reference candidate, then smaller depth, larger leaf minimum, the smaller sum
of nominal split-feature counts over the three inner schemas, then candidate
ID. This tolerates numerical noise without using portfolio performance.
There is no global winner selected across outer years.

Persist all eight candidates' three annual and 36 monthly scores, key/mask
hashes, coverage, zero-probability/floor counts, schemas and fit receipts. Report
pooled loss, Brier, accuracy, calibration and class 1/2/9/10 precision/recall as
descriptive diagnostics. They do not break ties or select settings. No Sharpe,
return, drawdown, stop exit or outer score enters selection. A baseline win is
a valid completed result. No minimum improvement is required for completion;
the fixed grid cannot expand.

All eight candidates must finish all three inner blocks. Any failed fit,
missing month, invalid prediction or exceeded budget stops the attempt. Keep
its evidence; never select from successful candidates only, retry silently,
reduce trees, omit slow candidates or widen the grid. After complete selection,
refit that winner once on the full preceding ten years with fresh preprocessing.
No extra baseline fit or independent-seed ensemble is added.

## 6. Prospective local compute budget

VERIFIED LOCAL: `sysctl` reports **34,359,738,368 bytes = 32 GiB RAM**, ten
physical and ten logical CPUs. The four corrected RF-expanded annual fit
receipts report 450.4–468.7 seconds per single-thread fit, 98,307–101,463
supervised rows, 1,430 prepared columns, roughly 562–580 MB float32 training
matrices and 40–44 MB saved forests. These are observations from old runs, not
a benchmark of this new search or its four-thread speedup.

| Budget | Frozen limit |
|---|---:|
| Inner estimator fits | 8 × 3 × 4 = 96 |
| Selected annual outer refits | 4 |
| Maximum RF fits / trees | **100 / 50,000** |
| Fresh preprocessing states | 12 inner + 4 outer = **16** |
| Parallelism | **one fit process; four RF tree threads** |
| BLAS/OpenMP and inference threads | 1 |
| Aggregate process-tree RSS | warn at **16 GiB**; stop at **20 GiB** |
| Minimum available/reclaimable RAM before launch | **12 GiB** |
| Minimum free disk before launch | **30 GiB** |
| Per-fit wall time | **20 minutes** |
| Total preprocessing/fit/prediction/scoring wall time | **16 hours** |
| Separate zero-fit verification wall time | **1 hour** |
| Cumulative projected compressed source reads | **8 GiB** |
| All new attempt artifacts, including failed fits/matrices | **20 GiB** |

These are stop ceilings, not expected running times or permission to run now.
All 100 old full-size single-thread baseline fits would total about 12.7 hours;
shorter inner histories, more feature candidates per split and four-thread
parallelism change that cost in unbenchmarked ways. Do not promise an ETA or
silently expand resources to finish a partial search.

Use a supervisor that samples its own plus all descendant processes' current
RSS each second, without concurrent candidates, outer windows or verification.
Stop the process tree on the observed ceiling or timeout and preserve receipts.
Sampling may overshoot; do not call this an already implemented operating-system
memory reservation or exact hard cap. Count all new output bytes. Count repeat
source decodes in the compressed read projection, separately from matrix mmap
I/O. For the launch RAM check, use macOS `vm_stat`: header page size times the
sum of free, inactive and speculative pages. Do not also add overlapping
purgeable/file-backed counts. This is a conservative availability proxy, not
a guarantee that every inactive page is immediately reclaimable. Record the
raw memory reading. Recompute exact column/footer read plans before execution.
Existing raw data may be reused; no raw-source fallback, source download or
cloud job is added.

The old four fold read plans project 2.247 GB including their test inputs;
their training portions project 2.038 GB. The new twelve inner plus four outer
states need a new cumulative plan. The historical 8 GiB RF limit belonged to
an earlier experiment. The small streaming verifier's cap belongs to that
verifier. Neither truncates the data, and neither is edited here. This new
prospective 20 GiB allowance leaves nominal 12 GiB of the physical machine for
the OS and other applications; it does not force the model to allocate 20 GiB.

## 7. Backtest rules remain fixed

RF-expanded / class 1/10 / 10% per-asset stop / 10 bp one-way costs remains the
primary reporting focus. Keep the corrected four-model results as comparators,
including all withheld paths. No additional comparator is fitted here.

Keep class 1/10 and class 2/9 separate. Select by predicted mode, then the
corresponding class probability and stable ID; use the same available count
on both sides, `min(10, eligible longs, eligible shorts)`. Start each monthly
rebalance at 50% long / 50% short, equal weights within each side, executing at
the next trading session's open. Keep unstopped and fixed monthly-entry 10%
per-asset stop variants. Stops are not trailing or a guaranteed portfolio loss
cap: gaps use opening fills; the intraday rule retains its idealized OHLC fill.

Stopped capital stays cash until the next monthly rebalance, with no replacement
or exposure reset. Long/short exposure can drift between rebalances. In
particular, the directional exposure seen in the corrected drawdown is not
changed by this tuning. Preserve corporate-action accounting, quote overlays,
the exact provisional PENN handover, continuous NAV with no annual reset,
0/5/10/25 bp one-way cost scenarios and zero modeled borrow/funding charges.
Missing held marks retain exposure and withhold the affected full-path metrics;
no synthetic prices or additional performance-driven exclusions are allowed.

A new backtest is outside this model-compute budget and needs its separate
execution scope. It must consume verified selected forecasts under these same
rules. Better class log loss does not guarantee better portfolio returns.

## 8. Execution gate, interpretation and handoff

Before a later authorized run, implement a separate runner and independent
review with synthetic chronology, tie-selection, cohort and resource-stop
checks. Freeze its code hashes, exact runtime, fit counter, input read plan,
monitor and a fresh private attempt root before financial access. Preserve
every original run/pointer and failed attempt. Independently reconstruct loss
selection, split-local preprocessing and saved-model probabilities without
refitting before declaring completion. Do not invoke the frozen single-thread
worker with relaxed settings or rewrite its recipe to bypass its guard.

2020–2023 has already been examined and motivated this development. Later
walk-forward tuned results there are a development comparison, not a fresh
untouched test. Do not tune again based on those outer results under this
protocol. Protected 2024+ financial values stay unread.

The accepted identity waiver, historical reconstruction versus unavailable
knowledge-time evidence, missing terminal outcomes, retrospective DF/RDC
restriction, incomplete comparators, short availability and idealized fills
remain limitations. Passing this protocol's causal checks would not certify
all identities, all 715 formulas, survivorship completeness or investability.

Private budget metadata, preservation hashes and the freeze receipt are under
`outputs/rf_tuning_protocol/dmv2-rf-expanded-temporal-tuning-20261020-v1/`.
Only the new JSON contract and this original methodological note are intended
for future Git publication; raw panels, estimators and outputs stay private.
