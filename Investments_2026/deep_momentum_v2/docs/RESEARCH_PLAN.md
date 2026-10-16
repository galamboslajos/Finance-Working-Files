# Starting research workflow

Status: DATA FOUNDATIONS, FULL FEATURES AND SHARED TARGET/RET WORKERS IMPLEMENTED;
FULL TARGET COVERAGE AUDITED; FIRST FOUR-ARM CLASSIFIER PILOT COMPLETE AND VERIFIED.
New strategy research begins here; the previous attempt is a reference.

The user has now authorized the [first classifier pilot](CLASSIFIER_PILOT.md):
one RF and one XGBoost fit on each of Paper15 and the expanded candidate bank,
using 2010–2017 completed outcomes for training and 2018–2019 for validation.
[`configs/classifier_pilot.json`](../configs/classifier_pilot.json) freezes the
settings before fitting. It uses primary 80% context, seed 1729 and the same
cohorts and observed-label evaluation mask across four arms. Training-only
preprocessing, 500-tree RF and 300-round XGBoost are separately recorded; no
parameter search, early stopping, 90% model sensitivity, outer refit, RET or
backtest is included. The completed run retained 77,690 supervised training rows
and all 20,755 eligible validation predictions, including 46 missing labels;
20,709 observed labels were scored identically across all arms.
The earlier `model_comparison.proposed.json` remains a design snapshot.

The verified mean monthly log losses are RF/Paper15 2.277891, XGBoost/Paper15
2.276638, RF/expanded 2.263056 and XGBoost/expanded 2.264372, versus 2.302585 for
the uniform reference. Expanded RF has the lowest observed loss and 13.60%
monthly accuracy in this development split. This is a descriptive pilot result;
later outer tests and economic evaluation remain separate work. Four fits took
911.08 seconds with 5.01 GiB peak RSS and 726 recorded source files unchanged.
Independent reconstruction, exact model-probability reload checks, the 690-test
suite and seven later focused verifier tests passed. The fresh master ran all
24 code cells while guards blocked implicit fits/builds and preserved its
separate daily/monthly/feature frames. Full metrics and evidence are in the
[pilot verification record](CLASSIFIER_PILOT.md).

**USER DIRECTION, 2026-10-15:** initialize research in 2010 and use rolling ten-year
training / one-year outcome tests. The canonical settings and readable master
notebook are described in [the notebook workflow](NOTEBOOK_WORKFLOW.md).
Nominal tests run from 2020 through the existing pre-holdout boundary in 2023:
2010–2019 training precedes 2020 testing, advancing annually to 2013–2022
training / 2023 testing. There are four nominal folds; the rolling window and
monthly prediction frequency are unchanged. The [2010–2023 endpoint audit](INITIALIZATION_2010.md)
measures complete 15-input paper coverage under the accepted provisional identity
policy. Full-bank coverage is now measured in [the full-panel audit](FULL_FEATURE_PANEL.md); unavailable context coverage limitations and economic certification remain unresolved. The feature worker
uses exact pre-signal warmup, including pre-2010 prices, described in [FEATURES.md](FEATURES.md).

The first source/date/identity/quote contracts and adversarial tests are in
[data foundations](DATA_FOUNDATIONS.md). One real 2010-01-29 combined snapshot
retains all 902 names. Its original strict-policy audit counted 130 clear
identity routes, 755 review-flagged routes, ten collisions and seven unlinked
names. The [active provisional policy](PROVISIONAL_IDENTITY_POLICY.md) accepts
unique resolved review-flagged routes while preserving the flags; missing/ambiguous
routes and collisions remain blocked. Dated identity evidence remains unverified,
without blocking provisional development or changing original denominators.

The [monthly worker and notebook stage](MONTHLY_RETURNS.md) now preserve daily
data independently and construct bounded provider-adjusted ratios from full
preceding-month cohorts. Missing/identity/unrequested states remain visible.
Economic event accounting and knowledge-time certification remain unresolved.

The locked paper is Han-Qin, *Bimodality Everywhere*, March 26, 2026. Read
[its precise reference](PAPER_REFERENCE_HAN_QIN_2026.md) and
[the U.S. decision ledger](US_REPLICATION_PLAN.md). It uses monthly ten-class
XGBoost probabilities and historical-payoff reclassification, not a required
neural architecture. Historical S&P 500/400/union data, additional features and
forest ensembles are our intended adaptations.

1. Audit dated U.S. membership and identity joins, return adjustments,
   availability, delisted names and corporate actions on a bounded approved
   pre-holdout sample. Preserve exclusion denominators and prior failure evidence.
2. Apply the user's decision to exclude market capitalization. Start from
   the 15 momentum/state inputs before extensions, omitting the paper's size
   feature, cap-based screens and cap-value weighting. Keep this departure
   explicit; obtaining capitalization is not a task or blocker. Portfolio
   weighting remains a separate OPEN decision.
3. Use the selected historical S&P 500/400 union and audit usable history within
   the selected 2010 initialization / rolling 10-1 chronology. The feature and
   shared target/class/RET conventions are implemented in their named contracts.
   Freeze remaining estimator validation/settings, operational signal/entry
   timing, costs and trial budgets before fitting or portfolio evaluation.
   Resolve paper underspecification in the ledger instead of inventing author settings.
4. Implement and verify the paper mechanism under the declared 15-input
   V2 adaptation, with monthly momentum, mode-XGB
   and RET under a common cohort and economic ledger. Keep prediction quality
   distinct from portfolio/valuation completeness. The paper's random historical
   inner split and any temporal variant must have different named protocols.
5. Compare forest probability estimators on the same baseline inputs. Compare
   added feature groups separately, then combine using development-only
   calibration/weights if justified. Neural models or direct regressors are
   optional extensions, not prerequisites from this PDF.
6. Measure a small engineering sample, then prepare a bounded full experiment
   with resource estimates, cost limits, trial bounds, acceptance criteria,
   output provenance and cleanup controls under the project execution workflow.

The prior recovery used 1998 warmup, a 1999-2023 accepted signal panel,
2009-2023 primary inputs and 2015-2023 test folds. These are reference roles,
not the new design. Ten years of usable history are required before a
paper-style initial fit. Product-wide early dates do not establish that history.
The prior 2024+ holdout remains protected until a new protocol explicitly
determines its use.

The previous comparison's economics were inconclusive due to missing valuations
and event/identity limitations. Successful model fits, source-file verification
or feature construction are not evidence of complete economic performance.

The user-authorized [feature stage](FEATURES.md) registers 728 definitions with
715 eligible predictor names. It preserves original cohorts, signal-only history
and a separate completed-outcome X/y interface. Two-ID samples withhold complete
cohort statistics; no model fit or full-context scan is implicit.

The historical user-authorized 90% context rebuild is complete: all 402 previously empty
context predictors have observations somewhere in 2010–2023; none are entirely
missing over that full period. Paper15,298 technical features and every preserved
daily artifact match the previous run exactly. Among paper-qualified rows,
context-value coverage is 59.52% in 2010–2019 and 98.08% in 2020–2023. Early missingness
remains material; these are descriptive counts, not full-history feature selection.
See [the full-panel verification](FULL_FEATURE_PANEL.md) and the active
context contract for detailed annual/feature coverage.

## Active context panel roles, 2026-10-15

By user direction, the verified 80% original-cohort panel is now primary; the 90%
panel is retained as sensitivity. The [promotion/comparison record](CONTEXT_SENSITIVITY_80.md)
preserves original 90% →80% comparison tables and exact immutable run pins.
Context-value coverage among paper-qualified rows is 96.27% versus 71.06% over
2010–2023, and 92.09% versus 33.57% in 2010–2015 (80% versus 90%). Exhaustive requests,
original denominators, paper15, technical298, PIT cutoffs and separate daily
artifacts are unchanged. This is a user policy choice, not selection by model
performance or proof that remaining omissions are random.

## Initial modelling design and active pilot

The primary model universe is now the user-selected historical S&P 500/400
union. The [initial comparison sketch](MODEL_COMPARISON.md) separates XGBoost
versus random forest and Paper15 versus expanded inputs in four arms. The master
displays metadata-only plans from `dmv2.model_design`; no model fit is implied.
The shared target/class/payoff contracts are implemented. The separately
authorized [full historical target audit](FULL_TARGET_PANEL.md) is complete
under the same PIT, missingness, chronology and protected-data rules. Estimator
adapters and fixed probability metrics are now implemented under the active
pilot contract above. Its completed run retains 86,825 training originals,
77,690 supervised fit rows, 21,827 validation originals and 20,755 predictions.
All 46 eligible missing validation classes remain in prediction coverage;
20,709 observed classes supply the common evaluation mask. Validation labels
do not enter preprocessing, fitting or stopping. Run All continues to verify
saved reports without fitting; the explicit local pilot is complete and its
saved models are not loaded by notebook Run All.

## Shared monthly class/RET engineering, 2026-10-15

The [shared workers](TARGET_CLASSES_RET.md) now implement exact tie-aware monthly
classes, original-cohort/missing-outcome ledgers, the label-aware training adapter
and strict 120-calendar-month realized-class RET payoffs. Forward-outcome requests
are distinguished from past feature requests. The master checks already loaded
bounded samples and keeps daily/feature data separate. A separate explicit
full-history build has measured label and strict RET payoff coverage. No
estimator or backtest ran during this target/RET implementation stage,
and economic accounting remains unresolved.

## Full monthly target and RET coverage audit, 2026-10-15

The user authorized the full 2010–2023 target panel and class/payoff coverage
audit before fitting either model. The cache-only workflow reuses the primary
panel's original cohorts, provisional identity admission, Paper15 eligibility
and independent calendar evidence. It keeps forward-outcome requests separate
from feature-history requests and preserves existing feature and daily artifacts.
The [full target audit record](FULL_TARGET_PANEL.md) describes the explicit build
and saved readers. The completed audit retains 152,307 original rows,
149,950 identity-admitted rows and 140,777 signal-time prediction-eligible rows.
It releases 140,485 labels (99.79% of prediction-eligible rows). The 292 eligible
unlabelled rows comprise 276 missing end quotes and 16 source quality flags;
they remain in prediction ledgers. All 168 months pass exhaustive-request and
rank guards, and all ten classes appear in every month. Forty-three months
contain exact-return ties; the largest tie has four members.

All 48 nominal test signals have complete RET payoff vectors,
cross-checked through the canonical shared payoff worker. The first available
signal is 2019-12-31 for the January 2020 prediction. The preceding 120 signals
withhold RET because they lack 120 completed outcome months; history is never
shortened. The saved audit contains 201,600 month/class grid rows and 1,680
class-payoff rows, and verifies 475 recorded input files unchanged. Rolling
ten-year classifier windows and monthly 120-calendar-month payoff updates remain
separate and unchanged.


The separately authorized [daily development stage](DAILY_BACKTEST.md) uses
saved 2018–2019 probabilities, fixed-ten class 1/10 and 2/9 baskets,50% / 50%
capital and an independent 10% per-asset stop alternative. RET stays gated;
no earlier payoff extension, new fit or 2020–2023 evaluation is included.
