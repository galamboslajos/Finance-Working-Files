# Active normalization and missingness protocol

**ACTIVE ENGINEERING, IDENTITY UNVERIFIED — version2.0.0, 2026-10-15.**
The machine-readable contract is
[`configs/normalization_protocol.json`](../configs/normalization_protocol.json).
The user authorized completing this protocol under provisional identity admission,
then selected **2010 initialization** after seeing the early-history coverage gap.
The former [`normalization_protocol.proposed.json`](../configs/normalization_protocol.proposed.json)
remains a historical inactive proposal. Its strict-identity anchor fractions do
not describe the current provisional policy.

This freezes signal normalization and a reusable train-only missingness worker.
It does not certify identities, complete economic returns or authorize a model
fit. The primary S&P universe, label/ranking policy, inner validation and
experiment budget remain separate decisions. No model results selected these
rules. The 2010 choice is explicitly a coverage-motivated sample change.

## Chronology and separate clocks

USER DIRECTION: use rolling ten-year training, annual refits and monthly
predictions. With supervised outcomes beginning in January2010, the nominal
first training window is **2010–2019** and first test outcome year is **2020**.
The pre-holdout development period ends with December2023 outcomes; protected
2024+ financial values remain uninspected. Exact pre-2010 warmup prices support
features but do not add supervised training years.

Features for an outcome month stop at the preceding exchange month-end close.
January2010 therefore uses the December2009 signal; December2023 uses the
November2023 signal. Daily quotes remain separate for later daily backtesting.
Missing outcomes never determine the signal population or its statistics.

## Signal-specific common momentum population

The active paper policy is `signal_qualified_common_momentum_v1`:

1. Construct the full original dated historical membership cohort before prices.
2. Apply `provisional_unique_resolved_link`, preserving original review flags and
   unresolved/collision rows in the ledger. Admission is not identity certification.
3. Request **every admitted native ID** in that original signal cohort. If the
   request is partial, withhold all15 statistical features. Missing quotes after
   an exhaustive request are source coverage evidence; an unrequested ID is not.
4. Qualify one common population of admitted members with all five finite raw
   momenta. Each signal qualifies independently; do not intersect future survivors.
5. Require at least two common members and finite, strictly positive sample SD
   for **all five horizons** (`ddof=1`). If any gate fails, withhold all15.
6. On that same common population, calculate five z-scores, five cross-sectional
   means and five SDs. Only its members receive these15 inputs. Every original
   member remains in the coverage ledger with explicit eligibility/status.

The exact paper indexing remains `MOM1=r[t]`. For `m=3,6,9,12`, compound
`t-m+1` through`t-1`, giving2/5/8/11 completed monthly returns. Require every
intermediate exact exchange month-end endpoint, with no stale substitution,
forward filling, zero imputation or missing-month compression. The union of
all five dependencies spans the thirteen anchors`t-12` through`t`.

The means and SDs are shared state variables whose levels vary through time.
Do not standardize them across stocks within a month: that would erase their
information. Current test-signal cross-sectional normalization uses only that
signal's past inputs; it does not fit a pooled-time transform on future data.

The common population, sample-SD divisor and fail-closed gates are declared V2
adaptations to paper equations6–8. The mathematical minimum of two is a
computability gate, **not a claim that a selected subset represents the full
index**. There is no invented empirical coverage-percentage threshold.
Selection from missing history and unverified identities remains reportable.
Private measured evidence is in `outputs/initialization_coverage/2010-v3`.

`paper_features.normalize_common_momentum(raw_frame,
exhaustive_requested_by_signal=...)` is the shared implementation. Its input
contains original keys, identity admission and five past-only raw momenta.
Outcome/target fields are rejected. The price worker independently derives the
exhaustive-request attestation from original admitted IDs and requested IDs.
The standalone normalizer necessarily relies on the caller's original ledger
and explicit attestation; it cannot prove a source query it did not perform.

## Added features and missingness

The715 base predictor allowlist stays fixed. Do not require all715 to be
complete or let added-feature availability change the paper-qualified prediction
population. Unavailable/undefined added inputs remain missing; no forward fill
or global zero replacement occurs. Original audit flags remain outside X.

For forest experiments, the frozen missingness policy is
`train_partition_median_missing_indicator_v1`, implemented in
[`src/dmv2/preprocessing.py`](../src/dmv2/preprocessing.py):

- Fit from an existing `TrainingSlice` with an explicitly aligned boolean
  `fit_mask`. Selected rows must independently have all15 finite paper inputs,
  finite available labels, and completed outcomes inside the recorded training
  and label-observation cutoffs. Invalid selected rows raise; they are not
  silently removed. Labels are never imputed.
- In that training partition only, drop all-missing predictors and predictors
  with constant finite values, recording each reason. A later validation/test
  value cannot revive a dropped column. This includes dropping an otherwise
  constant finite predictor even if its missingness varies.
- Fit the finite-value median of each retained predictor. Transform missing or
  nonfinite added inputs with that training median. Include a binary missingness
  column for every retained predictor, even if training had no missing values.
  Baseline missingness never triggers baseline imputation: those rows remain
  in the ledger and are withheld from prediction.
- Fit independently for every inner-training partition and outer-training
  window. No full-history coverage statistic chooses a feature, median or
  threshold for an earlier forecast. Validation/test transform never refits.
- Use no automatic scaler, clipping or winsorization. The paper's particular
  normalization does not justify an additional universal z-score for forests.
- Fingerprint the fitted input/output schema, medians, drop reasons, training
  keys and cutoffs. Missingness columns belong to this serialized derived
  preprocessing schema; they do not silently expand the715 source definitions.

Example for a later modeling stage, after completing the separate label policy:

```python
from dmv2.preprocessing import fit_preprocessor

# train is an existing TrainingSlice. Choose this mask from baseline eligibility
# and valid completed training labels; the worker independently checks it.
preprocessor = fit_preprocessor(train, fit_mask=supervised_fit_mask)
prepared = preprocessor.transform(signal_X, prediction_mask=signal_prediction_mask)
X_prediction = prepared.X.loc[prepared.ledger.transformed]
# Persist preprocessor.manifest and preprocessor.fingerprint with future fits.
```

`transform` returns all original rows: `prepared.X` is finite only for requested
baseline-complete prediction rows, and `prepared.ledger` records why others
were withheld. It consumes no target information. Missing future labels exclude
rows from supervised loss only; original prediction, coverage and accounting
rows remain. Retention documents missing terminal outcomes but does not
reconstruct them.

## Context and empirical limits

The active context worker still uses explicit `complete_cohort`. Cross-sectional
and market/relative context stays withheld when its exhaustive original or
DAILY historical-cohort requirements are unmet. Reusing one month-end cohort
across historical daily benchmark windows is prohibited. The active paper
policy and technical definitions do not claim all715 features have full-history
coverage. Full context adaptation needs its own tested implementation.

Bounded two-ID notebook samples can demonstrate exact raw momentum and
technical formulas. They cannot establish full-cohort statistics; their paper
normalization remains withheld under the exhaustive-request gate. Coverage
reports separately preserve original membership, identity admission, common
past-input population and actual normalized outputs.

`knowledge_time_verified`, `accounting_certified`, `daily_backtest_ready` and
`model_ready` remain false. Retrieval-time identities/adjusted history, missing
quotes and terminal payouts remain material empirical limitations. Provisional
identity uncertainty was accepted by the user and does not block all engineering.
