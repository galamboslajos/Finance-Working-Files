# First four-arm classifier development pilot

Status: four-arm execution and independent result verification complete, 2026-10-15.
The authoritative recipe is
[`classifier_pilot.json`](../configs/classifier_pilot.json), fixed before the
first research-data fit. The original `model_comparison.proposed.json` remains
a historical design snapshot; its open execution/readiness fields do not
describe the separate active pilot.

## Question, population and chronology

Compare random forest and XGBoost on the same Paper15 inputs, then repeat both
on the expanded 715-name candidate bank. This separates an estimator change
from an input-bank change. Use the existing historical S&P 500/400 union,
provisional identity admission and 80% primary context panel. No 90% sensitivity
fits, searches or feature-group trials are part of this pilot.

Training outcomes are January 2010–December 2017 (96 months). Their features
end at December 2009–November 2017 signals. Validation outcomes are January
2018–December 2019 (24 months), using December 2017–November 2019 signals.
The last training outcome ends before the first validation outcome starts.
Saved trading-session evidence, rather than an assumed December 31 quote,
determines actual label observation cutoffs: the training slice ends with the
completed December 29, 2017 session, within the calendar December 31 bound.

| Population | Training | Validation |
|---|---:|---:|
| Original historical stock-month rows | 86,825 | 21,827 |
| Signal-time Paper15 prediction eligible | 77,871 | 20,755 |
| Available canonical class label | 77,690 | 20,709 |
| Eligible but missing label, retained | 181 | 46 |

All four arms fit exactly the same 77,690 supervised stock-months and predict
all 20,755 eligible validation rows. Missing future labels do not remove a
prediction, redefine a feature population or select an estimator cohort.
The same 20,709 observed validation labels contribute to every arm's scores.
All other original rows and their original admission/status flags remain in
the saved ledgers. Neither `feature_complete` across 715 inputs nor complete
added-feature coverage is a cohort filter.

This is the inner development pilot, with one fitted object per arm throughout
the 24 validation months. It does not change the project's later rolling
ten-year annual refits/monthly forecasts. No 2020–2023 model evaluation, outer
refit or protected 2024+ financial inspection is authorized by this recipe.

## Frozen preprocessing and estimator settings

Use the existing `classification_training_slice` and
`train_partition_median_missing_indicator_v1` contract. Fit preprocessing only
on the 2010–2017 supervised mask, independently for Paper15 and the expanded
bank. Each fitted state is shared between the two estimators using that bank.
Drop only training-partition all-missing/constant predictors, fit finite-value
medians, and add one fixed missing indicator for every retained predictor.
Do not refit on validation, impute a label, add scaling, clip/winsorize inputs,
or use full-history coverage to select predictors. The saved manifests report
semantic names, medians, drop reasons and final matrix width; missing indicators
mean 15/715 base names can become up to 30/1,430 estimator columns.

The prepared matrices use float32 only after a finite/range guard. Persist
their exact ordered schemas, original fit/prediction keys and input hashes.

| Setting | Random forest | XGBoost |
|---|---|---|
| One fixed-seed fit per feature bank | 500 trees | 300 boosting rounds |
| Maximum depth | 12 | 3 |
| Leaf/child regularization | Minimum 20 samples per leaf | `min_child_weight=20` |
| Feature sampling | `sqrt` | `colsample_bytree=0.8` |
| Row sampling | Bootstrap, full sample size | `subsample=0.8` |
| Other | Gini, no class weighting/OOB/warm start | Learning rate 0.05, lambda 10, alpha/gamma 0, histogram 256 bins |
| Seed / threads | 1729 / 1 | 1729 / 1, CPU |

No validation labels or evaluation set are supplied to either fit. There is
no early stopping, tuning, learned calibration, blending, resampling or
validation-driven refit. Effective library settings and XGBoost's native
booster configuration are saved beside each model.

These are untuned V2 pilot settings. They are not undocumented author defaults
and one fit is not the paper's annual 100-fit probability ensemble. The paper
also uses 16 inputs and expanding classifier history; V2 omits capitalization
and uses its separately selected rolling chronology. See the
[source-locked paper account](PAPER_REFERENCE_HAN_QIN_2026.md).

Exact additional package pins are in
[`requirements-models.lock.txt`](../requirements-models.lock.txt): scikit-learn
1.9.1, XGBoost 3.4.1, SciPy 1.18.1, joblib 1.6.0, threadpoolctl 3.7.0,
narwhals 2.27.1 and cloudpickle 3.1.2. The run identity also pins Python, NumPy,
pandas, PyArrow, exchange-calendars, code and source manifests. These binaries
were resolved and import-tested in the existing Python 3.14.6 macOS ARM64 V2
environment. [The forest API](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
and [XGBoost estimator API](https://xgboost.readthedocs.io/en/stable/python/sklearn_estimator.html)
document the estimator interfaces; our choices above are the frozen V2 protocol.

## Probability, metric and reference conventions

Canonical class 1 is best and 10 worst. Persist each native class mapping;
RF fits 1–10 directly, while XGBoost fits 0–9 with an explicit +1 mapping.
Reject any missing training class. RF uses native probability averages.
XGBoost evaluates the objective's stable softmax from native margins in
float64, avoiding float32 row-sum drift under the existing 1e-8 probability
contract. This constructs probabilities from logits; it does not repair,
clip or renormalize an already supplied probability vector.

Primary metric is the equal-weight mean of 24 monthly log-loss means, with
lower values better. Secondary Brier is the sum of ten squared class errors
per stock, then averaged within months and equally across months (range 0–2,
not divided by ten). Report monthly accuracy, pooled stock-month scores,
fixed ten-bin classwise/top-label calibration, and hard-mode class-1/class-10
precision with prediction/observed/missing support and explicit empty groups.
Hard-mode probability ties select the lowest canonical class ID.

Only the log-loss calculation floors a true-class probability at 1e-15, with
zero/floored cases counted. Saved probabilities, Brier, modes and calibration
use the original probabilities. There is no calibration fit. Missing labels
remain unscored rather than being filled with zero or another class.

Two fixed reference forecasts use the same validation cohort and label mask:
uniform 0.1 per class and the pooled supervised-training class frequencies.
The latter uses training stock-month counts, not validation labels or equal
weighting of training-month priors. Both are reference predictions, not extra
estimator fits. Uniform log loss is `ln(10)` and Brier is 0.9.

RET and portfolios remain disabled: the 2018–2019 development dates cannot
provide 120 completed prior outcome months from the selected 2010 start.
There is no shortened payoff history or new pre-2010 supervised sample.

## Loading, execution and preservation

`pilot_loading.py` first verifies JSON manifests, immutable source/role pins,
archived recipes, original keys, flags and independent windows. It hashes
comparison artifacts without decoding their numerical reports. Only approved
annual 2010–2019 feature/target values are then loaded, through the shared
classification adapter. All 728 registered feature columns are read to honor
the frozen table schema; the 13 known constants/rank aliases never enter X.
No source download, raw-price fallback or daily-data rebuild occurs.

The projected cache reads are 497,294,113 compressed bytes (about 474 MiB).
The conservative loader estimate is 2.73 GiB. The frozen execution ceiling
is one sequential CPU fit, one estimator thread, 8 GiB peak RSS, 90 minutes
total, 1 GiB projected reads and 2 GiB new outputs. A watchdog stops the run
if the time or memory ceiling is exceeded. Matrices are prepared once per
feature bank, persisted privately, and released before sequential model fits.

All source feature/daily files, shared targets and old pointers are checked
before and after execution. Hashing later-year files verifies byte integrity;
it does not decode their financial values. A completed audit alone publishes
the new pilot `LATEST.json`. Incomplete attempts remain unselected and cannot
silently cause an automatic duplicate fit.

Pre-fit engineering notes: the first planning probe used older reporting
readers that decoded saved full-period financial summary tables. Those values
were not used for fitting, parameter/predictor selection or evaluation. The
production path was replaced with tested metadata-only verification before
any research-data fit. The first materialization then stopped on the
calendar/session cutoff mismatch above, also before preprocessing or fitting.
Its incomplete run is preserved. The fix clamps to saved completed-session
evidence and leaves all labels, populations and frozen model settings unchanged.

## Repeating the explicit workflow

From the configured V2 root:

```bash
PYTHONPATH=src .venv/bin/python -m dmv2.classifier_pilot plan --root .
PYTHONPATH=src .venv/bin/python -m dmv2.classifier_pilot run --root .
PYTHONPATH=src .venv/bin/python -m dmv2.classifier_pilot verify --root .
```

The direct CLI enforces estimator thread limits and the frozen resource
ceiling. The installed task runner also limits BLAS/OpenMP thread environment
variables. A completed identical recipe is verified and reused; it is not
fitted again. A changed protocol/source/runtime creates a distinct identity.

Outputs remain ignored under `outputs/model_pilot/`. Every run records:
original train/validation ledgers; frozen preprocessing manifests and prepared
matrices; four fitted models and native class maps; original-row probability
ledgers; fixed references; monthly and pooled probability/calibration/tail
reports; source/recipe/artifact hashes; runtime and resource evidence.

Section 17 of the master verifies saved reports and displays the six-way
comparison without loading serialized estimators, preparing full matrices or
starting a fit. The existing daily/monthly/feature views stay separate. Saved
models are local trusted artifacts; the notebook reader only verifies hashes.

## Results and independent verification

The completed run is
`outputs/model_pilot/2010-2017__2018-2019-a2cd7669fde3f582`.
Its root audit SHA-256 is
`d7d773059f69af6a2feaf95b73391915f0118cd66901d0d6e29e67035b262ab1`.
All numbers below give each validation month equal weight, rather than giving
larger monthly cohorts more influence.

| Forecast | Base predictors | Log loss ↓ | Brier sum10 ↓ | Accuracy ↑ | Pooled top-label ECE ↓ |
|---|---:|---:|---:|---:|---:|
| RF / Paper15 | 15 | 2.277891 | 0.894768 | 12.56% | 0.004277 |
| XGBoost / Paper15 | 15 | 2.276638 | 0.894704 | 12.69% | 0.011031 |
| RF / expanded | 715 | 2.263056 | 0.891507 | 13.60% | 0.001209 |
| XGBoost / expanded | 715 | 2.264372 | 0.892057 | 13.45% | 0.013307 |
| Uniform reference | — | 2.302585 | 0.900000 | 10.00% | 0.000005 |
| Training-prior reference | — | 2.302585 | 0.900000 | 10.03% | 0.000043 |

Both expanded arms improve on their Paper15 versions in this window. Expanded
RF has the lowest observed log loss, but its lead over expanded XGBoost is only
0.001316. This is a descriptive comparison from one development split, without
a significance or economic-performance claim. ECE is computed from fixed-bin
pooled observed predictions; very low reference ECE does not imply useful
ranking power. Uniform hard modes use the declared lowest-class tie rule.

Training-only preprocessing retained all 15 and all 715 candidate inputs, with
no all-missing/constant drops, producing 30 and 1,430 matrix columns. Each arm
fit once on 77,690 rows and predicted 20,755 rows across all 24 months; the same
20,709 labels were scored and 46 missing-label predictions retained. No
true-class probability required the log-loss floor. All 726 recorded source
feature/daily/target/pointer files were unchanged. Total runtime was 911.08
seconds (15m11s), peak RSS 5,382,127,616 bytes (5.01 GiB), and saved output usage
at the final resource check was 658,759,144 bytes (about 628 MiB).

The full 690-test worker suite passed before the final audit-receipt binding
addition; all seven focused verifier tests then passed, including rejection of
a root audit changed during verification. The independent
[`verify_classifier_pilot.py`](../scripts/verify_classifier_pilot.py) imports no
production pilot workers and fits no models. It verified 71 saved artifacts,
reconciled all 108,652 original train/validation source keys, admission flags
and labels, independently recomputed all 730 input-column training statistics
and every prepared matrix value, and reproduced row/month/summary scores,
calibration and extreme-class precision/support for all six forecasts. Reloading
each of the four hash-verified local models reproduced every probability
exactly (maximum difference 0.0). Financial decoding stopped at 2019 outcomes.
The saved independent receipt lives under
`outputs/model_pilot/verification/independent_verification.json`.

The installed master completed all 24 original code cells in a fresh kernel
in 32.99 seconds. Verification patches blocked full pilot loads/fits, imputer
fits, full target rebuilds and NumPy/joblib/pickle loads; none was called.
All six saved comparison rows were displayed, and exact before/after frame
checks preserved separate daily/monthly/feature samples. The versioned notebook
remained output-free. Private notebook receipts and the executed copy are in
`outputs/model_pilot/verification/`.

These are temporal development diagnostics on 24 dependent validation months.
They are not final 2020–2023 outer-test performance or investable returns.
Historical identity knowledge time, adjusted-return semantics and missing
terminal cash flows retain the existing provisional limitations. This pilot
does not select a winner, revise thresholds/settings or authorize a backtest.
