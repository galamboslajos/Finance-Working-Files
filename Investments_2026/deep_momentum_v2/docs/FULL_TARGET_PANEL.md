# Full monthly targets and RET coverage, 2010–2023

Status: full local build and independent numerical verification complete, 2026-10-15.
This is the user's explicit target/coverage task, under
[`full_target_build.json`](../configs/full_target_build.json). It does not fit
preprocessing or either classifier. The [shared target/RET contract](TARGET_CLASSES_RET.md)
still controls the formulas, ties, missingness and causal cutoffs.

## Scope and immutable inputs

Outcome months are January 2010 through December 2023: 168 signals from
December 31, 2009 through November 30, 2023. Preserve all 152,307 historical
S&P 500/400-union source-ID/signal keys. The same 140,777 Paper15-qualified
prediction rows apply to XGBoost and random forest, and to both the 80% primary
and 90% sensitivity feature panels. Added-feature missingness does not select
the target cohort. The 80%/90% context thresholds are not label thresholds.

`full_target_loading.py` verifies each immutable feature-year checkpoint,
original cohort, identity admission, feature-coverage ledger, exchange calendar
and the original executed forward request before reading quote values. It uses
the saved `daily_history_raw.parquet` because that cache preserves the preceding
December anchors, including entrants. Later `daily_prices_raw.parquet` files
alone can omit those anchors. A cache miss cannot trigger another source read
or survivor-based replacement. All admitted originals were requested; missing
quotes do not change that request evidence.

The measured plan projects six quote columns plus a two-column metadata pass:
74,578,827 compressed bytes across 9,507,584 annually overlapping cached rows.
The conservative peak-memory estimate is 1,499,355,136 bytes. Recorded limits
are one local worker, three hours, 4 GiB RSS, 1 GiB new output and 2 GiB projected
reads. Full source-file hashing also reads bytes for integrity, without decoding
their financial values; it is separate from this projected-column budget.
No new download or cloud action is part of the build.

Only approved outcome-year exchange sessions and the preceding signal anchor
reach the monthly-return worker. Warmup and source holiday quotes are excluded
using metadata before Python financial conversion. Raw histories, disjoint
daily backtest tables, features, original flags, configs and old run pointers
remain unchanged and are checked against a preserved SHA-256 manifest.

## Verified class coverage

| Population/state | Stock-months |
|---|---:|
| Original historical members retained | 152,307 |
| Identity eligible under the accepted provisional policy | 149,950 |
| Prediction eligible with the unchanged Paper15 policy | 140,777 |
| Valid observed returns and released classes | 140,485 |
| Prediction eligible but missing ending price | 276 |
| Prediction eligible but source-quality flag | 16 |
| Identity ineligible, retained without a class | 2,357 |
| Identity eligible but Paper15 ineligible, retained without a class | 9,173 |

Labels cover 99.7926% of the prediction cohort and 92.2380% of all original
members. All 168 months pass the ranking rules and contain all ten classes.
Exact-return ties occur in 43 months; the largest tie group has four members.
No month has an empty class. All requests are exhaustive, with no unknown
forward request, unrequested admitted member or missing outcome record.
Known requested rows with absent endpoint prices remain explicit missing-price
states; they are not absent ledger records.

Classes rank valid observed next-month outcomes inside the signal-time
Paper15 prediction cohort, with class 1 best and 10 worst. Exact ties receive
the same descending midrank bin; they are never split or jittered. Missing
labels are never imputed and never change prediction eligibility. The 292
eligible missing outcomes must remain visible in later prediction evaluation
and event accounting. Retaining them does not recover missing terminal payouts.

Class counts, missingness reasons and original/prediction denominators are
saved monthly, annually and by development period. The four ten-year training
windows have their own coverage and class-presence reports. These descriptive
reports do not authorize full-history feature selection or any fitted statistic.

## RET coverage

All **48/48 nominal 2020–2023 payoff vectors are available**, starting with the
December 31, 2019 signal for January 2020. Its 120 outcome months are January
2010–December 2019. The last audited test signal is November 30, 2023 for
December 2023; its payoff window ends in November 2023. All 48 states match
the existing shared `build_ret_payoffs` worker exactly.

The earlier 120 signals are withheld because they lack a complete preceding
120-month outcome history. This is expected initialization, not a missing-class
failure. No forecast is moved earlier by shortening the window. All four
annual rolling classifier windows contain the full 120 months and all classes:

| Test year | Training outcomes | Labelled rows | Eligible label coverage |
|---|---|---:|---:|
| 2020 | 2010–2019 | 98,399 | 99.77% |
| 2021 | 2011–2020 | 99,684 | 99.78% |
| 2022 | 2012–2021 | 100,617 | 99.79% |
| 2023 | 2013–2022 | 101,553 | 99.79% |

The audit retains the full 168 × 120 × 10 calendar grid (201,600 rows), even for
signals with too little history. It also saves 1,680 dated class-payoff rows
and the 48 canonical worker states for the nominal 2020–2023 model-test signals.
At each signal, every financial month consumed by the canonical payoff worker
must already be completed at that signal's cutoff. Future labels are neither
decoded nor used by a past payoff calculation.

Each payoff is an equal-weight mean of the 120 monthly equal-weight realized
class returns. Every ten-class cell must be present in every month. Any absent
month or class withholds the whole vector; there is no shortened, pooled,
filled or compressed calendar. Payoff availability establishes structural
coverage conditional on observed outcomes, not investability or expected profits.

## Saved artifacts and repeatable access

The complete run is selected by `outputs/full_target_panel/LATEST.json`.
Its audit pins its code/runtime/source identity, annual checkpoints and every
artifact. `targets.parquet` and `monthly_returns.parquet` are separate from the
existing feature and daily datasets. Each annual checkpoint also retains its
monthly returns, target state and loading evidence. `reports/` contains the
class, missingness, rolling-training and RET-grid reports. `ret_states/` contains
the 48 separately dated shared payoff objects.

Parquet standardizes nested-list child names. `target_state.json` preserves
the exact original serialized Arrow schemas, which are restored before target
fingerprints are verified. This storage fix changes no class or return formula.
An earlier incomplete attempt is preserved for diagnosis and is not selected
by `LATEST.json`; completion is published only after all checks pass.

From the V2 root, using the existing `.venv`:

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m dmv2.full_target_panel plan --root .
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m dmv2.full_target_panel verify --root .
```

The independent numerical audit can also be repeated explicitly:

```bash
.venv/bin/python scripts/verify_full_target_panel.py --root . \
  --run outputs/full_target_panel/2010-2023-c7e2bde2d1781be7 \
  --output outputs/full_target_panel/verification/independent_target_verification.json
```

`build --root .` is a separate explicit action; the notebook does not run it.
Section 16 of the master reads verified small reports and exposes lazy label
and monthly-outcome datasets. It clears state before verification and leaves
the bounded `daily_df`, `monthly_df` and `features_df` untouched.

```python
from pathlib import Path
import sys
sys.path.insert(0, str(Path.cwd() / "src"))  # V2 root; master already bootstraps this
from dmv2.catalog import SourceCatalog
from dmv2.full_target_panel import read_full_target_run, read_full_targets

run = read_full_target_run(SourceCatalog(Path.cwd()))
annual_coverage = run['tables']['target_coverage_year']
ret_coverage = run['tables']['ret_coverage_signal']
targets = read_full_targets(run)  # explicit shared class-panel load; no fit
```

Both estimators must use `classification.classification_training_slice` on
these same targets, with completed outcomes, training-only preprocessing and
the unchanged rolling ten-year annual fits/monthly predictions. The actual
supervised fit mask requires a released class; the prediction population stays
independent of later label availability.

## Validation and limits

The completed run is `2010-2023-c7e2bde2d1781be7`, built in 102.33 seconds
with 16,755,907 bytes of output. Its recipe is
`sha256:c7e2bde2d1781be765397ee15dbeddf34b124ead87b8a0de7a68738fee673b85`;
the shared target artifact is
`sha256:916629ef57f157a561c1617e8b02d5cff5159a49e5436d17a7f53624174a527c`.
All 475 preserved input/config/pointer files matched their hashes.

The portable `scripts/verify_full_target_panel.py`, without production target/RET
imports, recomputed all
140,485 classes, all 43 tied groups, 201,600 RET grid cells, 1,680 dated payoffs
and 48 saved canonical states. It also checked 2,716 original rows and 5,373
quote endpoints at three fixed signals against cached adjusted closes; observed
ratios and missing/identity/quality states matched. All 620 tests passed,
including schema round trips, strict manifests, stale-state guards, target ties,
causal cutoffs, protected-value rejection and complete-calendar payoff rules.
All 23 original master code cells also passed in a fresh kernel, with three
private verification cells enforcing no full build/preprocessing fit, checking
exact unchanged bounded daily/monthly/feature frames and explicitly reloading
the saved shared target state. The versioned notebook retains empty outputs.
Private independent and notebook receipts plus the executed notebook are saved
under ignored `outputs/full_target_panel/verification/`. The independent receipt
pins the exact run, recipe, root audit, target fingerprint and verifier; the
verifier checks timing metadata before financial conversion and rechecks all
250 target artifacts plus 475 preserved inputs before and after its audit.

PIT metadata guards, dated historical membership and source integrity checks
remain active. Current identities and provider-adjusted endpoint ratios remain
historical reconstruction: knowledge-time identities/publication, adjustment
semantics, delisting cash flows and survivorship-bias-free economics are not
certified. No 2024+ financial values, preprocessing fit, model fit, probability
score or backtest is part of this audit. Daily data remain separate for their
future backtesting role as well as their existing feature-calculation role.
