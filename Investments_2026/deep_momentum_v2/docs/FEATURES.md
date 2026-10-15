# Signal-only feature workers

Implemented under user direction on 2026-10-14. Feature construction is separate
from labels, model fitting and portfolio accounting. The master calls worker
files and preserves daily histories independently of the monthly outcome ledger.

USER UPDATE: the master now passes the explicitly accepted
[provisional identity policy](PROVISIONAL_IDENTITY_POLICY.md) through every
feature worker. Unique resolved review-flagged routes may receive values while
their original flags/statuses remain in separate coverage. Missing/ambiguous
routes and collisions remain blocked. Formula definitions, predictor allowlists,
and past-input clocks are preserved. The later active paper normalization uses
one common five-momentum population after an exhaustive original-cohort request;
the active context policy now uses 80% original-cohort coverage, with 90% sensitivity. Earlier complete-cohort results remain historical evidence.

## Definition bank and explicit adaptations

The fixed bank has **728 registered columns**: 298 daily technical definitions,
415 historical S&P 500/400 context definitions and the paper's 15 monthly
momentum/state inputs. It reimplements formula definitions inspected at the
organizational `GalambosC/deep-momentum` commit
`e9a7a01881a46374694f60fc34eb7d7e4a26c2b8`. Private source references and the
745-name reuse audit remain under ignored `outputs/feature_audits/`.

The additional 22 company/macro source-dependent columns and ten S&P 600
spreads are outside this bank. Capitalization, cap-based screens and size are
excluded by user direction. Raw dollar volume is activity, not size or turnover.

The predictor allowlist has **715** names: three one-session market constants
and ten algebraically duplicate ranks are retained in the dictionary but
excluded from X. The ten aliases are ranks of mean log returns versus ranks of
simple horizon returns at 5/20/63/126/252 sessions, for universe and sleeve,
under our complete equal-length window policy. Robust z scores remain distinct.
Different horizons and transformations are not independent economic signals;
no distinct variable is removed simply because it is correlated with another.

| Worker | Function |
|---|---|
| `daily_features.py` | 298 security-local daily definitions, exact-session rolling history, then signal snapshots |
| `paper_features.py` | Exact monthly momentum indexing and explicit signal population normalization, 15 inputs |
| `context_features.py` | Balanced 80-parent ranks/MAD scores, market descriptors and paired relative risk |
| `feature_loading.py` | Independent bounded historical price request ending at the latest signal |
| `features.py` | Original-cohort left joins, predictor allowlist and explicit completed-outcome training interface |
| `feature_workflow.py` | Notebook orchestration and separate fingerprinted private evidence |

Daily families: 18 returns, seven skip momenta, 77 return-distribution features,
72 trends, 12 oscillators, 45 OHLC descriptors, 63 activity/liquidity proxies and
four event observations. Context families: 320 cross-sectional, 65 market-state
and 30 relative definitions. Formula strings, required inputs, exact windows,
roles, source reference and model exclusions are emitted by each worker's
`*_feature_specs()` and saved to `feature_dictionary.json`.

The strict V2 engineering policy requires complete exact required session
inputs, for individual time windows. The 80% context threshold is a cross-sectional coverage rule; it does not relax temporal windows. Incomplete
or undefined features remain null with reasons; no missing-date collapse,
forward fill, stale-price fallback, missing-to-zero conversion or sample ranks.
The paper does not specify its SD divisor: **sample SD, ddof=1**, is a declared
V2 adaptation. The active common-five paper population and explicit context
coverage policies are V2 choices, not author-specified cleaning rules.

## Clock and separation

At a verified exchange month-end `t`, all predictor inputs end at that closed
session. The next-month target return occupies **(signal close, next month-end
close]**. A signal-date close can be the feature's last observation and the
target's starting anchor; this shared boundary does not reveal the subsequent
return. It does not assume a fill at the already observed close. Actual entry,
publication timestamps, costs and execution still need their separate protocol.

The master exposes:

```python
feature_history_df  # independent price request, ending at latest signal
features_df        # source_security_id, signal_date, registered values only
feature_coverage_df  # identity/request/missingness audit; never predictors
daily_df           # unchanged daily outcome sample for later daily accounting
monthly_df         # next-month outcome and coverage ledger, never feature input
```

Raw `MOM*_raw` values and calculation statuses are diagnostics outside the
predictor dictionary. The paper inputs are `zMOM1/3/6/9/12`, `meanMOM*` and
`stdMOM*`. MOM1 uses month `t`; long MOM3/6/9/12 compound 2/5/8/11 returns from
`t-m+1` to `t-1`. Every required monthly endpoint must be present and accepted.
Neither a missing month nor an intermediate absent endpoint can be hidden by
telescoping or shifting the remaining observations together.

Each daily rolling feature is evaluated on the complete exchange-session grid
before selecting signal month-ends. Future dates between two requested signals
may support the later signal, but cannot alter the earlier snapshot. Raw price
rows after the final signal, protected 2024+ rows, duplicate keys, noncalendar
dates and outcome/target contamination are refused. Future calendar rows are
metadata boundary evidence, never permission to read future financial values.

The engineering read requests 503 sessions through the first signal, then ends
at the last signal. Abnormal-volume features first divide daily raw volume by
its trailing 252-session mean, then average those ratios over `w` sessions:
complete history needs `w+251` observations, hence 503 for `w=252`. Return
volatility over 252 returns separately needs 253 closes. Pre-entry price
history can support a currently eligible member; membership does not truncate
its past price history. Current initialization in 2010 refers to training outcome
years, not to cutting off feature warmup at January 2010.

## Historical cohorts and missing context

Prediction cohorts are reconstructed at each signal before price joins. All
original source IDs remain, including unrequested, unresolved, review-required
and colliding identities. Current delisted flags, future quotes, missing future
targets and future membership never select these rows. Price attachment uses
defensive quarantine even if a caller incorrectly labels a flagged route resolved.

The default notebook is a two-ID engineering sample. Statistical features
require exhaustive admitted-ID requests over the original group, then their
own population/coverage gate. Two requested stocks cannot stand in for the
intended market. For added context, each sleeve and the declared union require
at least 80% valid original members independently;90% remains sensitivity.
Blocked identities remain in the denominator. Individual temporal windows
must still be complete.

Market/relative workers additionally require **dated DAILY original cohorts**.
Monthly snapshots cannot be frozen backward to manufacture daily benchmark
membership. The bounded loader does not scan that full membership history;
missing context returns explicit nulls. Pure workers support a supplied complete
daily context for a later measured build. Market descriptors average current
members' horizon returns; they are not official or investable index-return
series. Relative statistics use complete jointly observed daily stock/benchmark
pairs; index switches or entries can shorten qualified sleeve history even
when raw security prices exist.

## Predictor and completed-label interface

```python
from dmv2.features import predictor_frame, training_slice

X_all = predictor_frame(feature_run.panel)  # whitelist only; no fill/drop/fit
training = training_slice(
    feature_run.panel, monthly_run.sample.panel,
    train_start="2010-01-01", train_end="2019-12-31",
    label_observed_through="2019-12-31",
)
X, y, audit = training.X, training.y, training.ledger
```

The example requires feature/outcome rows for that scope; the present two-month
sample does not establish a ten-year panel. Training selection uses completed
OUTCOME dates, not a signal's year. It verifies exact next-month calendar
endpoints and unique original source-ID/signal keys. Missing target ledger rows
fail the denominator check; explicit missing outcomes remain as unavailable y.
Future target status or numeric values do not change X. A December signal whose
target is January of the test year cannot enter the preceding training year.

This interface performs no imputation, scaling, winsorization, selection,
probability calibration, class labels or model fitting. Those transformations
must later be fit only on training months. Validation must respect completed
label cutoffs; do not randomly split stock rows across the same calendar month.
Multi-month overlapping targets would need separately specified purging/embargo;
this stage's targets are distinct next-calendar-month intervals. Shared past
lookbacks across signals are normal information reuse, not future-label leakage.

## Present limits and verification

Numeric success does not establish `knowledge_time_verified`,
`accounting_certified`, survivorship-free economics or `model_ready`. These
remain false. Membership availability and identity routes are retrieval-time
reconstructions without original publication/validity evidence. Adjusted prices
can contain later revisions or merger stitching, and terminal payouts are absent.
Strict knowledge-time mode fails closed. Action features retain the same source
revision limits; dividend divided by price is an event proxy, not annual yield.

Private feature artifacts are separate from outcome artifacts and include past
daily history, features, original-cohort coverage, diagnostics, dictionary,
source/calendar/request/code/runtime hashes and an audit. Identical reruns reuse
identical evidence; differing results under the same pins raise. The versioned
notebook has no execution outputs. Full-cohort/full-period feature construction,
models, backtests and cloud work require the later measured execution plan.

Tests use structural fixtures only: exact paper windows, daily formulas,
future-price perturbations, target contamination, missing sessions, warmup,
identity quarantine, entries/removals, original denominators, monthly boundaries,
completed-label cutoffs and duplicate-index handling. No fixture is a research
observation or evidence of economic performance.

## Verified bounded local execution, 2026-10-14

All **265** worker/integration tests passed. The installed master completed all
**13** code cells in two fresh V2 kernels, from both the V2 root and notebooks
folder. The versioned notebook remained without outputs. Both successful runs
reused identical private feature evidence under unchanged source/settings/code
pins; all ten artifact hashes and all 16 worker hashes were independently checked.

The feature read returned **1,046 daily observations**, dated 1998-01-05 through
2000-01-31, with 503 sessions through the first 1999-12-31 signal. The two signal
cohorts retained **1,799** original rows: four requested stock/signal rows each
had all 298 technical definitions available (**1,192** numeric values); 603
unrequested and 1,192 unresolved-identity rows remained visible. The original
daily outcome sample remained separate with 82 observations.

All 15 paper definitions are implemented and verified on complete structural
cohorts. In this real two-ID sample their complete-cohort normalization/state
values are correctly withheld, as are all context outputs without full cohort
and daily context inputs. This is a bounded engineering result, not a full
model-ready panel or certified PIT/economic return history. The bank has 728
registered definitions and 715 structurally eligible predictor names, not 715
empirically admitted features across 2000–2023.

The measured feature stage considered 9,741,712 candidate rows (primarily
membership metadata), returned 51,870 total source rows including metadata, read
five data files/109 footers and 43,454,076 projected compressed bytes. Budgets
remained within the declared stage limits. No model fit, protected financial
inspection, full feature build, cloud mutation or portfolio simulation occurred.

## Active paper population and training-only missingness

The [active normalization/missingness protocol](NORMALIZATION_PROTOCOL.md) is
`configs/normalization_protocol.json`. The proposed JSON and old strict-identity
anchor fractions remain archival evidence. The current paper worker requires an
exhaustive request over admitted original members, then one shared population
with all five exact finite raw momenta and positive finite sample dispersions.
Its shared means/SDs preserve their time-series levels. Arbitrary two-ID samples
withhold normalized values rather than redefining the universe.

The [2010 initialization audit](INITIALIZATION_2010.md) measures those 15 paper
inputs across the original historical cohorts. It does not certify all 715
candidate columns or full daily context. Extra predictor missingness does not
remove paper-qualified prediction rows. The preprocessing worker fits medians,
constant/all-missing exclusions and a fixed indicator schema on selected
completed training rows only; prediction transformation consumes no labels.

## Active added-context availability, 2026-10-15

The user first authorized a 90% rebuild, then an 80% sensitivity comparison, and
subsequently selected `available_cohort_80pct` as primary with 90% as sensitivity.
The already verified panels are retained byte-for-byte under explicit role pins.
These choices supersede the complete-cohort context policy in earlier engineering
records. Exhaustive admitted-ID requests remain mandatory;
coverage uses all original historical members, including blocked identities.
Ranks/MAD use per-base valid values only; no pre-rank imputation occurs. Daily
benchmarks check 80% on every historical daily union/sleeve, sleeve spreads gate
both sleeves, and individual exact lookbacks/relative pairs remain unchanged.
The paper15 and 298 technical definitions are unchanged. The frozen728 definitions
and715-name allowlist are preserved. Read [the detailed policy](CONTEXT_AVAILABILITY.md)
and [full-build results](FULL_FEATURE_PANEL.md).
