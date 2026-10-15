# Shared monthly targets and RET payoffs

Status: **implemented engineering contract**, 2026-10-15. This implements the
user-requested shared target and scoring layer for XGBoost and random forest.
It does not fit either model or certify historical identity, economic returns,
source publication timing or investable performance.

The versioned rules are in
[`target_ret_protocol.json`](../configs/target_ret_protocol.json). The workers
consume explicit supplied tables; they do not discover files, download data or
launch full-history builds. The master reuses its already loaded bounded
monthly/feature sample to check the guards. Daily data remain separate.

## 1. Same cohort for both models

Start with every original historical S&P 500/400-union member at the preceding
month-end signal. Retain every source-ID/signal key, including missing identities
and missing outcomes. A prediction-eligible member must pass the existing
provisional identity and Paper15 signal-time policies. Added-feature availability
does not change this population. Target availability never changes prediction
eligibility, past feature calculations or historical membership.

The label-ranking population is the **valid observed outcomes within that
signal-time prediction cohort**. This is a declared V2 convention; the source
paper does not resolve its precise missingness/ranking cohort. Require complete past-feature requests and separate forward-outcome request
evidence for every identity-admitted original member before computing classes.
Wrapped monthly panels derive forward requests from their declared native-ID
request manifest, independently of quote availability. Bare tables must supply
explicit forward `requested` flags. Partial or unknown requests withhold the
whole month. Missing outcome records remain explicit unlabelled rows with
unknown forward-request evidence; genuine missing-price rows with a known request
are distinct. Extra or duplicate outcome keys are rejected.

## 2. Ten classes, exact ties and missing returns

Class 1 means the highest realized return and class 10 the lowest. Rank each
completed next-calendar-month cross-section independently, in descending order.
For n valid observed returns, give an exact-equal tie group occupying sorted
positions L through U (one-based) the class:

`class = clip(1 + (10 * (L + U - 1)) // (2 * n), 1, 10)`

This is the average-rank rule
`1 + floor(10 * (average_rank - 0.5) / n)` using exact integer arithmetic at bin
boundaries. No rounding tolerance, ticker sorting, random jitter or artificial
tie splitting is applied. Input row order cannot change a security's class.

With 20 distinct returns, each class has two names. If the top ten returns tie
and the bottom ten tie, those groups receive classes 3 and 8. If all 20 returns
are identical, all receive class 6. These outcomes are intentional: report empty
classes rather than manufacture ten balanced groups. Fewer than ten observed
eligible returns withholds that month's classes; this is a mathematical sample
size gate, not an empirical completeness or representativeness standard.

Use only source status `ok` and finite numerical returns at least -1. Preserve
an explicitly supplied -100% return; do not erase terminal losses by rule.
Boolean/text values are not numerical returns. No new outlier clipping,
winsorization, zero fill, stale-price substitution or target imputation is
introduced. Other source statuses, absent records, nulls, infinities and invalid
returns retain explicit reasons and no class. Bad-status values and outcomes
after the observation cutoff are not decoded numerically.

`return_available` / `observed_return_available` describe an observed valid
return. `target_available` specifically means a released class. For example,
nine valid returns remain observed while all nine classes are withheld.

Diagnostics report original members, prediction-eligible members, observed
returns, labelled rows, request completeness, empty classes and missingness
reasons. The `observed_original_coverage` numerator is observed returns **within
the prediction cohort**, divided by all original members; it is not a measure
of all original stocks' source-price availability. Prediction-cohort coverage
has its own denominator. There is no invented percentage target gate: the 80%
primary/90% sensitivity context thresholds do not become label thresholds.

Labels and payoffs remain conditional on observed valid outcomes. Keeping
missing names in a ledger makes omissions visible; it does not recover missing
delisting cash flows or prove absence of survivorship/selection bias. The separately authorized [full 2010–2023 target audit](FULL_TARGET_PANEL.md) now measures
label/payoff coverage before model fitting; economic completeness remains unverified.

## 3. Time and integrity checks

Require explicit independent monthly windows, normally from the existing
`MonthlyReturnPanel.windows`. The signal precedes the outcome interval, which
must be in exactly the following calendar month. Accept outcome years 2010–2023
only. Validate all keys and date metadata, including later supplied rows, before
numerical conversion. Only completed outcomes at or before the explicit
`label_observed_through` cutoff can produce labels.

The output pins the policy, original keys, safe signal metadata, calendar
windows and each month's exact values/diagnostics. Scoped verification first
checks all timing metadata, then hashes only the requested completed financial
months. A past payoff or training slice need not decode later returns to verify
its inputs. Protected 2024+ outcomes fail before numerical conversion.

These are date-only historical-reconstruction cutoffs. The supplied windows do
not establish historically published identities, unrevised adjusted prices or
same-close operational availability.

## 4. Shared classifier training interface

Use `classification_training_slice` to create the existing `TrainingSlice`
interface with `y = next_month_return_class`. It validates original feature and
target keys, admission flags and independent endpoints; it subsets Arrow rows
by completed-outcome metadata before converting financial columns. Missing
class rows remain in X/y/ledger. The ledger's `target_available` follows class
availability, with return availability retained separately.

Apply the existing training-only preprocessor with an explicit supervised fit
mask. Keep all original prediction rows separately; use the baseline-qualified
signal population for both estimators. Extra-feature missingness and later label
availability must not select a different forecasting cohort.

The rolling setup is unchanged: train on 2010–2019 outcomes for the 2020 test,
through 2013–2022 for the 2023 test, with annual classifier refits and monthly
predictions. December 2019's completed training return uses a November 2019
signal; the December 2019 signal predicts January 2020 and is outside that fit.
Use the same adapter for the chronological inner development/validation slices.

## 5. RET uses 120 actual calendar months

At a closed monthly signal, the payoff window ends in that signal's calendar
month. December 2019 therefore uses January 2010–December 2019. January 2020
uses February 2010–January 2020. The cutoff cannot exceed the signal date, and
independent outcome-end evidence must establish completion. An earlier cutoff
does not silently shift the window to older months.

For every calendar month s and realized class k:

`monthly_class_mean[k,s] = equal-weight mean of observed returns in true class k`

`payoff[k,t] = equal-weight mean of those 120 monthly class means`

Use actual realized classes, not predicted buckets. Months with larger
cross-sections get the same time weight as smaller months. Do not pool stock
rows. Every one of the 120 × 10 month/class cells must be available. Any absent
month, uncompleted month, unavailable target group or empty class withholds the
**entire** payoff vector. Keep the full calendar grid and reasons; never shorten,
compress, pad or impute the window. This strict history convention is a V2
choice, and may delay usable RET forecasts beyond the nominal first test.

The classifier can stay frozen throughout a test year while the fixed payoff
estimator updates with newly completed past outcomes. January 2020 outcomes
can enter February scores, and cannot enter January scores or retrospective
tuning. The 2018–2019 inner pilot remains probability-only because 2010-start
history cannot provide 120 prior outcome months there.

## 6. Probability alignment and scoring

Provide the canonical class ID for every probability column. Reorder explicitly
to 1–10; require all ten unique integer IDs and finite nonnegative probabilities
whose row sums differ from one by at most 1e-8. Missing classes raise an error;
no zero padding, clipping or renormalization is applied. Estimator adapters must
persist their native-label-to-canonical-class mapping, including XGBoost's
eventual numerical encoding.

`RET = aligned_probabilities @ shared_payoff_vector`

Score only against the matching signal date and observation cutoff. Preserve
every supplied forecast row and its unique key. A withheld payoff vector
produces null RET scores with status, rather than dropping names or producing
zero returns. Scoring accepts probabilities and verified payoffs; it has no
future target argument. It does not form portfolios or execute trades.

## 7. Notebook scope and APIs

```python
from dmv2.targets import build_monthly_targets
from dmv2.classification import classification_training_slice
from dmv2.ret import build_ret_payoffs, score_ret

targets = build_monthly_targets(
    feature_panel.coverage, monthly_returns,
    label_observed_through=completed_cutoff,
)
train = classification_training_slice(
    feature_panel, targets, train_start="2010-01-01", train_end="2019-12-31",
    label_observed_through="2019-12-31",
)
payoffs = build_ret_payoffs(
    targets, signal_date=signal_date, label_observed_through=signal_date,
)
scores = score_ret(
    model_probabilities, payoffs, class_ids=canonical_ids_in_model_column_order,
    signal_date=signal_date, label_observed_through=signal_date,
)
```

This illustrates interfaces, not an automatically executable full-data example.
Both models consume the same targets and dated payoff state. The notebook's
bounded check uses existing samples only, capped at three outcome months and
3,000 original rows. Its two requested IDs do not justify cohort ranks or a
ten-year RET vector; correctly withheld outputs are expected. No full-history
target build, preprocessing fit, classifier fit or backtest runs implicitly. Section 16
separately verifies the completed [full target audit](FULL_TARGET_PANEL.md), displays
small saved reports and exposes lazy target/outcome datasets. Both models share
140,485 released classes over 168 months and all 48 nominal 2020–2023 RET states;
the 292 eligible missing outcomes remain explicit. The bounded check above retains
its original scope and correctly withheld sample results.
