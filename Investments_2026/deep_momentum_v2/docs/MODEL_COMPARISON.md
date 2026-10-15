# Random forest and paper-mechanism baseline: initial design

Status: **model comparison proposed**, 2026-10-15. The shared target, training
interface and RET workers are now implemented under the separate
[target/RET engineering contract](TARGET_CLASSES_RET.md). Its bounded checks do
not fit a classifier or preprocessor or run a backtest. The separately authorized
[full 2010–2023 target and RET coverage audit](FULL_TARGET_PANEL.md) is complete
from verified saved caches. It retains 152,307 original rows
and releases 140,485 labels from 140,777 signal-time prediction-eligible rows
(99.79%). All 48 nominal 2020–2023 monthly test signals have complete RET payoff
vectors; the preceding 120 signals correctly withhold them for insufficient history.
The user selected the historical **S&P 500 + S&P 400 union** as the primary
modelling universe. Existing saved panels already use that engineering union.

The machine-readable draft is
[`model_comparison.proposed.json`](../configs/model_comparison.proposed.json).
`dmv2.model_design` validates it and produces the experiment matrix, calendar
schedule and unresolved-decision report without reading market observations.
The master notebook displays these plans. The draft cannot enable execution.

## 1. What we compare

| Arm | Probability estimator | Semantic predictor bank | Question |
|---|---|---|---|
| A | XGBoost multiclass classifier | Paper15 | Closest V2 paper-mechanism baseline |
| B | Random forest classifier | Paper15 | Does changing the estimator help? |
| C | XGBoost multiclass classifier | Expanded715, including Paper15 | Do added features help XGBoost? |
| D | Random forest classifier | Expanded715, including Paper15 | Do added features help the forest? |

B versus A isolates the estimator; C versus A isolates added features. D versus
B and D versus C show the corresponding effects in the expanded forest. Run
the same primary 80% context panel throughout; repeat C and D with 90% as the
declared coverage sensitivity. A and B need no duplicate context-threshold run:
their paper inputs are identical. Do not choose a threshold from test scores.

"715" describes the candidate feature dictionary, including the 15 paper inputs.
Training-only all-missing/constant exclusions can reduce it. Missing indicators
also change the actual estimator matrix width. Record retained semantic names,
indicator columns and matrix width for every fit; neither 15 nor 715 promises
that many effective estimator columns after preprocessing.

**PAPER:** the locked Han–Qin version uses 16 inputs, expanding classifier
history, random historical 80:20 fit/validation splits and 100 XGBoost fits at
each annual refit. Average their aligned ten-class probabilities. RET then maps
probabilities to trailing ten-year realized class payoffs. See the
[source-locked reference](PAPER_REFERENCE_HAN_QIN_2026.md), sections 4–7.

**V2 USER DIRECTION / PROPOSED DESIGN:** our baseline retains that probability
and payoff mechanism, uses 15 inputs without capitalization, Intrinio/Siblis
historical S&P membership, rolling ten-year training and a proposed temporal
inner split. It is a declared adaptation, not an exact reproduction of the
paper's original experiment or published returns. An expanding/random-split
reference variant would be separately named and budgeted; it is outside the
initial four-arm matrix.

## 2. Shared prediction and payoff interface

At each closed signal month t, both classifiers predict probabilities for the
ten relative-return classes of calendar month t+1. Canonical class 1 is best
and class 10 worst. A forest changes the probability estimator; it does not
regress on integer class codes or replace probabilities with hard votes.
Scikit-learn's forest averages tree probability estimates; its `n_estimators`
counts internal trees, not the paper's 100 separately fitted XGBoost models.
See the [official forest API](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).

Persist each estimator's class encoding and align every probability column
before scoring or averaging. Require finite nonnegative rows summing to one;
missing training classes must be reported explicitly and resolved under a
frozen rule. No silent positional assumptions, class resampling, weighting,
calibration or probability blending.

For a common dated payoff vector, score each model with:

`RET[i,t] = sum(probability[i,k,t] * historical_class_mean[k,t], k=1..10)`

The proposed payoff window is the previous 120 **completed calendar outcome
months**, updated monthly. First compute each month's equal-weight mean of
actual realized returns within each realized class, then average those monthly
means with equal month weights. Do not pool stock rows, use predicted buckets,
compress missing months or substitute artificial decile-number payoffs.
All four arms share the same payoff ledger. A completed January 2020 outcome
may enter February's causal payoff update; it cannot enter January's forecast,
classifier tuning or a revised research rule. Missing-month/class handling is now fixed by the shared engineering contract:
require all 120 calendar months and ten classes, otherwise withhold the vector.
Empirical historical targets and RET payoffs now have measured coverage in the
separate full-panel audit. The 292 eligible missing
labels remain in the forecasting cohort and coverage ledgers. Terminal-event
accounting remains unresolved.

## 3. Time separation and proposed validation

| Test outcome year | Completed training outcome years | First test feature month |
|---|---|---|
| 2020 | 2010–2019 | December 2019 |
| 2021 | 2011–2020 | December 2020 |
| 2022 | 2012–2021 | December 2021 |
| 2023 | 2013–2022 | December 2022 |

Refit annually and predict monthly. For the first fold, December 2019's
completed outcome may enter training using its **November 2019 signal**.
The December 2019 signal predicts January 2020 and belongs to the test. Features
end at the signal; the target begins strictly afterward. These are date-only
research bounds, not a certified same-close fill or publication-latency model.

**PROPOSED:** within each outer ten-year window, fit on its first eight outcome
years and validate on its last two, keeping whole months together. Fit medians,
schema exclusions and indicators on the inner fitting portion only. After
settings and boosting-round count are fixed from historical validation, refit
preprocessing and the estimator on all ten completed training years. XGBoost
early stopping requires a held-out evaluation set and a recorded best-iteration
policy; see its [official estimator guide](https://xgboost.readthedocs.io/en/stable/python/sklearn_estimator.html).
The paper's row-random split and RF out-of-bag estimates are not the primary
temporal validation. Use the implemented target worker's actual outcome intervals
to enforce nonoverlapping fit/validation boundaries; overlapping past feature windows alone do not
authorize dropping arbitrary years or using future labels.

Keep signal-time prediction eligibility independent of future quote/label
availability. Construct historical membership and provisional identity admission
first, then Paper15 eligibility from past inputs. Expanded-feature missingness
must not remove a baseline-qualified prediction row. Keep all original rows and
missing outcomes in coverage ledgers; never impute a target. The 80% context
gate does not define a target-completeness threshold. The shared contract fixes
the ranking population to observed valid returns within the signal-time
Paper15-qualified cohort, uses exact-equal midrank ties, and withholds classes
below ten observed members or without exhaustive original-member requests.
Missing outcomes remain unlabelled. These rules do not recover missing terminal
returns or establish that the observed ranking population is representative.

Apply the existing [normalization/missingness contract](NORMALIZATION_PROTOCOL.md)
inside every training partition. No new universal z-score is proposed. Preserve
the paper's shared mean/SD levels, provisional identity flags outside predictors,
daily feature histories and separate daily backtesting tables. Protected 2024+
financial values remain outside this design.

## 4. Scope of the first implementation

Start with one explicit development pilot, not an automatic full search. Its
single fit per arm uses 2010–2017 inner-development outcomes and evaluates only
2018–2019 validation probabilities. The 2020 fold name identifies the enclosing
calendar window; the pilot does not refit on all ten years or inspect 2020 test
metrics. Full-window refits are separate fitted objects in the later frozen
outer evaluation. With initialization in 2010, inner validation in 2018–2019
does not yet have 120 prior outcome months: evaluate classification probabilities
there, and keep RET structural until its complete window is available. Do not
shorten the payoff history or silently add earlier outcome years.

The forest proposal is one 500-tree classifier per fit, with regularized leaf size
and a fixed seed; every setting in the draft is a V2 proposal. The paper's
100-fit XGBoost ensemble is the intended reference mechanism, but version pins,
early-stopping settings, ensemble variation and compute/trial ceilings must be
specified before execution. One-fit pilot results would not represent that
100-fit reference. Pin scikit-learn as well as XGBoost before execution.
Repeating a deterministic fit with unchanged data/settings
does not create ensemble diversity. Do not automatically fit 100 forests.

Next coding sequence:

1. The shared workers and full 2010–2023 target/RET coverage audit are complete.
   The audit finds all ten classes populated in every month and all 48 nominal
   test payoff vectors available, with original prediction denominators retained.
   Use the same verified target and payoff states for every estimator arm.
2. Class alignment, the dated 120-month payoff worker and class-label training
   adapter are implemented. Next add estimator adapters using those same
   interfaces and the existing training-only preprocessing.
3. Measure and record a bounded pilot's rows, memory, runtime and trial budget;
   pin dependencies and model settings before fitting real data.
4. Produce matched monthly predictions and diagnostics. Evaluate mean monthly
   log loss and multiclass Brier score first, plus tail precision, calibration
   and RET rank association with realized returns. Report observed-label and
   original-cohort coverage. Use the same evaluation mask across arms.

The 48 nominal outer-test months are dependent observations, not 150,000
independent experiments. Keep uncertainty and any later resampling time-aware.
Calibration, feature selection, stacking and ensemble weights may use only
chronological development predictions; no outer-test-driven redesign.

Portfolio evaluation follows later with one shared execution, event, cost,
exposure and borrowing ledger. Keep MOM12, mode-selected groups and RET score
deciles distinct. A usable probability score is not a backtest or evidence of
investable returns.
