# Deep Momentum V2 working instructions

Read README.md, docs/RESEARCH_PLAN.md and docs/DATA_STATUS.md before work here.
For features, labels, models or portfolios, also read
docs/PAPER_REFERENCE_HAN_QIN_2026.md and docs/US_REPLICATION_PLAN.md. Apply the
repository-local .agents/skills/han-qin-deep-momentum/SKILL.md when relevant.
For loading or universe changes, also read docs/DATA_FOUNDATIONS.md.
For identity evidence or mapping changes, also read docs/IDENTITY_AUDIT.md.
Read docs/PROVISIONAL_IDENTITY_POLICY.md for the user's active research waiver.
The active notebook admits unique resolved review-flagged routes provisionally;
preserve original flags and retain missing/ambiguous/collision exclusions.
Do not reopen full identity certification as a prerequisite to all development.
Source/platform aliases are not independent returned-provider confirmation.
The identity audit is non-promoting; keep its diagnostics outside predictors.
For notebook or chronology changes, also read docs/NOTEBOOK_WORKFLOW.md and
configs/research_workflow.json. The master notebook orchestrates worker modules;
keep reusable data/model rules in src/dmv2 and test them there.
For monthly return changes, also read docs/MONTHLY_RETURNS.md.
For feature changes, also read docs/FEATURES.md and docs/NORMALIZATION_PROTOCOL.md. Keep the original
daily data separate for future daily backtesting; monthly frequency applies to predictions.

- This is a U.S. equity project. Keep new code and outputs in this directory.
- User-selected chronology starts in 2010 with rolling ten-year training and
  one-year outcome tests (nominal 2020–2023 tests). Keep the paper's expanding history as a source fact,
  separate from this V2 adaptation. Feature workers use exact per-signal warmup;
  full historical usable coverage remains open. Preserve earlier raw history
  for feature warmup. The fixed 2000–2009 identity diagnostic remains historical
  reference evidence and does not measure the selected 2010–2019 training window.
- The notebook defaults to one bounded 2010-01-29 engineering audit and no price
  sample in the optional one-date check. Its separate monthly stage defaults
  to January–February 2010 and two requested price IDs. Keep committed outputs empty and save private audit evidence
  under ignored outputs. Do not treat Run All as an implicit full-panel/model run.
- Feature defaults are a bounded 503-session history for two first-cohort IDs.
  Keep features_df signal-only and separate from monthly_df outcomes and coverage.
  Statistics must not be computed on an arbitrary requested subset. The active
  common-five paper population needs exhaustive admitted-original requests;
  context uses `available_cohort_80pct`: exhaustive admitted-ID requests plus at least 80% valid original members per variable/date/cohort. Exact individual lookbacks stay unchanged.
  Preserve 728 dictionary definitions; the predictor allowlist excludes three
  constants and ten exact rank aliases. No outcomes or audit flags enter X.
- Monthly outcomes use the preceding month-end historical cohort. Preserve
  removals, missing/unrequested outcomes and original denominators. Exact
  provider-adjusted endpoint ratios are provisional research returns; do not
  double count dividends, substitute stale endpoints or certify cash-flow/PIT
  completeness. Daily path completeness is distinct from backtest readiness.
- Preserve ../deep_momentum/ as an observation and navigation reference.
- Use the existing Finance-Working-Files Git repository; do not create nested Git repos.
- Keep raw data, credentials, paper PDFs, source snapshots and generated outputs
  outside Git. Publish original methodological notes, not the downloaded PDF.
- Treat reference/sadig_us_recovery/ as a pinned reference, not our active code.
- The locked Han-Qin paper uses monthly ten-class probability prediction,
  XGBoost and RET reclassification. It does not require an LSTM or Sharpe-loss
  neural network. Added features and forest ensembles are intended extensions;
  distinguish them from the paper baseline and isolate their effects.
- Preserve the paper account: exact momentum indexing, 16 original inputs,
  expanding annual classifier history, 100-fit probability averaging and
  trailing ten-year class payoffs. Do not invent missing author parameters.
  V2 deliberately starts from 15 momentum/state inputs before extensions;
  omit the market-cap size feature, cap-based screens and cap-value weighting.
- Distinguish source fields, engineered features and model-ready panels.
  Historical capitalization is absent locally and excluded by user direction;
  obtaining it is not a V2 task or blocker. Do not substitute price or dollar
  volume for the omitted size feature.
- Use stable security IDs and dated membership. Record identity, adjustment,
  missing-price and corporate-action limitations from the prior U.S. attempt.
  Future quote availability must not determine signal-time eligibility.
- Build cohorts from dated source membership before price joins. Retain identity,
  missing-price and unrequested-price states and original cohort denominators.
  Current links/adjusted prices are retrieval-time evidence; distinguish
  historical_reconstruction from unavailable knowledge_time certification.
  Identity states are review evidence, not silently removed constituents.
  Apply the explicit identity admission policy consistently across workers;
  provisionally admitted review flags remain unverified and outside predictors.
- Use pinned manifest objects, explicit projections/date bounds and measured
  read budgets. Never return a silently truncated request or read protected
  financial values through metadata keys/filters. Read the documented holdout
  output guard and Parquet block-decoding limits before changing loaders.
- Keep PAPER, INTERPRETATION, VERIFIED LOCAL, USER DIRECTION, PROPOSED and
  OPEN claims separate.
  Paper-reading completion is not empirical replication or a frozen protocol.
- Freeze formulas, timing, eligibility, chronology, costs and trial bounds before
  results. Do not inspect protected 2024+ financial values unless a new research
  protocol explicitly authorizes that access.
- Small local metadata/sample checks are appropriate. Full feature builds and
  backtests must follow an agreed execution plan. Cloud compute, GCS writes,
  IAM changes and deletions need their applicable authorization and policies.
  Registered-repository cloud mutations use quant-gcp-ops gcp-run.

- `configs/normalization_protocol.json` is the active engineering contract under
  the accepted provisional identity policy. The proposed JSON is archival.
  Request all admitted original members and use one common five-momentum
  population at each signal. Preserve shared mean/dispersion time levels.
  Fit imputation and schema only on explicit completed training rows, including
  inside inner-training splits. Extra missing predictors must not remove
  baseline-qualified prediction rows; never impute a target.
- The 2010–2023 initialization audit is an explicit monthly-endpoint check,
  separate from the two-ID daily sample. The master verifies saved evidence;
  Run All does not launch the full audit. Complete paper inputs do not certify
  all 715 added predictor coverage or representative full-index performance.

- The user-authorized full 2010–2023 feature build is implemented in `full_loading.py`, `full_features.py`, `full_panel.py` and `panel_coverage.py`. Read docs/FULL_FEATURE_PANEL.md and configs/full_panel_build.json before changing it. The master reads verified saved reports; full building is an explicit CLI action. Preserve full annual raw warmup and past-only feature histories separately from disjoint outcome-year daily tables. Context uses the user-approved 80% primary original-cohort rule, with 90% retained as sensitivity; read docs/CONTEXT_AVAILABILITY.md. Missingness reports never authorize full-history predictor selection or fitting.

By subsequent user direction, **80% is primary and 90% is the sensitivity check**.
The verified panels retain their original paths, identities and bytes; their
roles are recorded in `configs/context_panel_roles.json`. Read both panels and
saved comparison evidence with `dmv2.context_panel_roles.read_context_panel_roles`.
Historical comparison tables retain their original **90% → 80%** orientation.
Notebook Run All verifies saved evidence without starting a build or fitting a model.
See [the comparison and promotion record](docs/CONTEXT_SENSITIVITY_80.md).

For model planning, read docs/MODEL_COMPARISON.md and the standalone
configs/model_comparison.proposed.json. The primary modelling universe is the
user-selected historical S&P 500/400 union. The four-arm design and temporal
inner validation in that draft remain historical proposals, not execution authority. model_design.py
reads draft metadata only. Keep the paper's original expanding/16-input/100-fit
method distinct from V2's rolling/no-size baseline. Existing panel recipe
configs are pinned historical contracts; model planning must not rewrite them.

For classifier labels or RET scoring, read docs/TARGET_CLASSES_RET.md and
configs/target_ret_protocol.json. Use targets.build_monthly_targets and
classification.classification_training_slice for BOTH XGBoost and random forest;
features.training_slice still returns continuous research returns. Preserve
independent past-feature and forward-outcome request evidence, all original keys
and outcome metadata before financial conversion. Targets use exact-equal
descending midrank bins on valid observed Paper15-cohort outcomes. RET requires
every ten-class cell in 120 completed calendar months and withholds the whole
vector otherwise. The notebook check reuses existing <=3-month/3,000-row samples;
it does not authorize a full label build or fit.


The user-authorized full 2010–2023 target panel is implemented separately in
`full_target_loading.py`, `target_coverage.py` and `full_target_panel.py`. Read
`docs/FULL_TARGET_PANEL.md` and `configs/full_target_build.json` before changing it.
Use verified annual raw-history caches (including prior December anchors), not
raw-source fallback or future quote inventories. Preserve all original keys and
the same Paper15 prediction cohorts for both estimators. The saved build has
140,485 classes for 140,777 prediction-eligible rows, with 292 eligible missing
outcomes retained; all 48 nominal 2020–2023 RET vectors have complete 120-month
ten-class grids. Earlier RET states are withheld rather than shortened. Section 16
of the master verifies saved reports and exposes lazy target/outcome views;
Run All never rebuilds full targets, fits preprocessing/models or backtests.
Original daily/feature files and the protected 2024+ financial guard remain unchanged.
Do not treat observed-label coverage as certified delisting or survivorship accounting.

For the first authorized classifier development pilot, read
`docs/CLASSIFIER_PILOT.md` and `configs/classifier_pilot.json`. This separate
frozen recipe supersedes the old draft's pilot readiness fields only for four
fits: RF/XGBoost on Paper15/expanded, 2010–2017 training outcomes and 2018–2019
validation outcomes. Use `pilot_loading.py`, `pilot_models.py`,
`pilot_metrics.py` and `classifier_pilot.py`; never rewrite the pinned feature,
target or normalization recipes to run a model. The approved local ceiling is
four sequential single-thread fits total, 8 GiB peak RSS and 90 minutes; no
searches, early stopping, validation-driven refits or implicit additional fits.
Future experiments need their own explicit scope and frozen protocol.

Fit both preprocessing states on the same supervised training mask only, and
share each state between RF and XGBoost. Retain all original cohort keys and
flags, predict every signal-time Paper15-qualified validation row, and score
only the identical observed-label mask. Do not filter by added-feature
completeness or future labels. Use saved observed trading-session cutoffs
within calendar bounds. No 2020–2023 financial values enter this pilot's
preparation/fitting/evaluation; integrity hashing does not decode those values.
RET, outer refits/evaluation, protected 2024+ values and backtesting are disabled.

Section 17 calls `read_classifier_pilot_run` to verify saved hashes and small
reports only. It must not load serialized models, prepare full matrices or fit
anything. The explicit CLI `run` reuses a completed identical recipe and refuses
to repeat an incomplete same-identity attempt automatically. Keep generated
matrices, estimators, probability ledgers and verification receipts private under
ignored outputs. `scripts/verify_classifier_pilot.py` independently reconciles
source populations, training-only medians/matrices, scores and saved-model
inference without refitting. Preserve separate daily/monthly/feature data and
the provisional identity, source knowledge-time and terminal-event limitations.


For the separately authorized 2018–2019 daily development backtest, read
`docs/DAILY_BACKTEST.md` and frozen `configs/daily_backtest.json`. Use new
`backtest_loading.py`, `backtest_selection.py`, `backtest_accounting.py` and
`daily_backtest.py`; preserve all pinned pilot/feature/target workers and recipes.
User selected ten longs / ten shorts within each mode class pair (1/10 and 2/9
separately), 50% / 50% equal weighting, and a separate 10% per-asset stop variant.
RET's 120-month gate remains intact and RET is withheld in 2018–2019. Reuse saved
probabilities only; no new fitting, earlier payoff extension or 2020–2023 financial
evaluation. Daily adjusted-price accounting is a provisional proxy. Missing held
valuations retain positions and withhold continuous full-period performance.
Section 18 verifies saved small reports only; Run All never starts a backtest.

For the additive corporate-action/bar-repair rerun, read docs/ACTION_BACKTEST.md,
docs/CORPORATE_ACTION_EVIDENCE.md and configs/action_backtest.json. Preserve the
original backtest workers, pointer and eight selected/rebalance artifacts byte
for byte. Use the separate action workers and independently pinned reviewed
quote overlays. Recognize verified signed raw-share entitlements for existing
holdings; retain retired new orders as unfilled cash slots. Missing successor,
halt or held-price evidence still withholds the affected continuous path; do
not drop stocks, invent payouts or rank models on complete paths alone. Section
19 verifies saved reports only. This rerun is restricted to 2018–2019, with no
new fit or 2020–2023 evaluation. Partial remote source projections may transiently
decode mixed Arrow blocks; excluded values must never reach returned tables,
Python conversion or analysis, and the limitation must remain explicit.

The 2018–2019 action audit found a specific old-BBBY → historical Overstock
native-route conflict. Read docs/ACTION_BACKTEST_RESULTS.md before interpreting
saved pilot/backtest performance. The provisional waiver does not certify this
known conflict. Keep frozen artifacts as evidence; a correction needs a separate
versioned mapping and affected feature/target/model rebuild, without silently
changing the current selections or evaluating later years. This is a specific
data repair, not a requirement to reopen full identity certification for all work.

The additive `identity_amendment.py` and `identity_impact.py` implement the
October 16 review; read docs/BBBY_IDENTITY_AMENDMENT.md before any dependent new
build. The saved amendment retains all original rows and source review flags,
invalidates the demonstrated wrong route, and explicitly leaves its replacement
unresolved. Use the reviewed amendment in a new, separately versioned corrected
workflow; never silently reuse the old native route or treat blocked exposure as
repaired price history. Original feature/request eligibility is audit-only in
the amended view and must be recomputed after actual data repair. The current
reference workers/configs/pointers remain frozen. `require_corrected_price_route`
blocks a corrected rebuild while replacement evidence is absent. Section 20 of
the master verifies saved metadata/hash receipts only; no panel decoding, fit,
backtest or 2020–2023 financial evaluation is authorized by this review. This
specific unresolved route does not suspend unrelated provisional development.

Read docs/DATA_REPAIRS_20261016.md before consuming the separately reviewed
`bracket_ohlc_repairs.py` amendment. It replaces only the already-rejected
ACC/K/SO August 12, 2019 raw/adjusted OHLC fields using pinned SIP bars and
preceding valid adjustment units, checked against a following control. It does
not widen the old close-match policy, repair volume, mutate source files,
rebuild features or authorize new performance results. Keep its separate ledger
and provisional unit/source limitations. The corresponding private package is
data/derived/data_repairs/2018-2019-resolution-20261016-v1. Existing old overlays
may be applied first; their original audit fields must survive the new amendment.
Dean Foods' eleven observed November/December bars were already cached. Missing
November 18–21 OTC bars must not be filled by an unapproved halt convention;
November 12 is a separate documented suspension. No alternative halt valuation
or OTC execution treatment has been implemented. Preserve all existing run and
selection pointers and the outstanding BBBY rebuild guard.

## Latest user override: immediate data-limited reconstruction

On October 16, 2026, the user directed a practical solution without waiting for
Sadig, explicitly allowing a documented exclusion. Read
`docs/DATA_LIMITED_BACKTEST.md` and frozen `configs/data_limited_fallback.json`.
This new authority permits the separate data-limited path to quarantine both
known wrong BBBY routes, retain all original membership rows/denominators, leave
their unavailable features/labels missing, rebuild 2010–2019 inputs, fit exactly
four frozen 2010–2017 classifiers and backtest the 24 months of 2018–2019.
It supersedes requiring a verified replacement native route before this
specifically labelled fallback can proceed; `require_corrected_price_route`
still applies to a claim of fully repaired identity/history.

Dean Foods and Rowan remain in valid historical features/targets and model
prediction ledgers, but are excluded from trading at every 2018–2019 signal
before top-ten selection, under the fixed retrospective trading restriction.
Do not delete old holdings/losses, fabricate missing quotes/exit prices, or add
new performance-driven exclusions. Retrospective coverage restrictions can
introduce selection bias and must remain explicit in all result claims.
Use new private outputs and independent feature/model/backtest phase identities;
preserve every original source/run/pointer. Run exactly four sequential frozen
fits and 64 predefined scenarios, with no searches, no notebook-triggered
compute, no cloud mutations and no 2020+ financial evaluation. Any additional
unexplained held-price gap retains exposure and withholds the affected complete
path, rather than creating an assumed exit or silently dropping a stock.


## Latest additive independent-source repair

Read docs/INDEPENDENT_QUOTE_BACKTEST.md and configs/independent_backtest.json.
The user authorized independent quote repairs and completion of 2018–2019.
Rebuild only the affected 2018/2019 validation panels, reuse the exact prior four
2010–2017 fits and training-only preprocessing, and perform zero new fits.
Use independent_rebuild.py, validation_reprediction.py, independent_backtest.py,
balanced_baskets.py and sni_action.py; preserve all older workers/runs/pointers.
The declared basket adapts contemporaneously to N=min(10,nlong,nshort), equal
names on both sides at 50%/50%; zero-sided months withhold. No new exclusions,
future-price eligibility filters, fabricated quotes or 2020+ evaluation. Add the
verified SNI cash+DISCK entitlement to the original eight actions. Passive
notebook reports initially use independent_reporting.read_independent_run.

The terminal-event completion in configs/independent_backtest_completion.json
adds VVC and TSS to the unchanged nine-action ledger. Read the completion section
of docs/INDEPENDENT_QUOTE_BACKTEST.md. completed_backtest.py writes a separate
backtest_completed directory and verifies every forecast and selection artifact
against the preserved 56-complete/eight-withheld parent. VVC includes the verified
$0.41145 closing dividend once. No new features, labels, fitting, inference or
quote changes. Final passive reports use completed_reporting.read_completed_run.


## Latest user override: annual 2020–2023 walk-forward

The October 16, 2026 instruction authorizes the separate four-fold experiment
in docs/WALKFORWARD_2020_2023.md and configs/walkforward_2020_2023.json.
It supersedes the older 2018–2019-only limits for this additive workflow only.
Use the preceding ten completed outcome years for each annual refit, monthly
forecasts, the same balanced class baskets and 10% per-asset stop alternative.
Exactly 16 sequential one-thread estimator fits and eight training-only
preprocessing states; no searches or performance-driven refits. Keep every
original source, worker, run and pointer unchanged, retain the fixed BBBY
quarantine and DF/RDC execution restrictions, and preserve all original cohort
keys. Read the new protocol before financial decoding. Do not inspect 2024+
financial values. New quote/action corrections require separately pinned
independent evidence; no new hindsight exclusions or invented exits. The
continuous daily ledger has no forced annual liquidation or NAV reset. The
master reads saved reports only and never starts this workflow implicitly.


## Current saved walk-forward handoff, 2026-10-16

Read docs/WALKFORWARD_EVIDENCE.md before interpreting the saved 2020–2023 run.
All sixteen fits and eight preprocessing states are finalized; independent
feature/model checks passed with their documented formula-sampling limits.
The separate additive V2 daily-accounting run and independent zero-fit audit
completed all 64 scenarios: 28 complete proxy paths and 36 withheld paths.
The accounting PASS covers source-based cash/units arithmetic and preservation
of withheld exposures; it does not fill missing marks. Do not rank strategies
using only the complete subset.

For a separately authorized accounting-only rerun, use
scripts/run_walkforward_backtest_v2.py and its explicit parent identity,
new output, failed-attempt and private registry arguments. It performs zero
fits. Preserve the failed V1 attempt and every frozen source, original worker,
configuration, feature/label panel, model, forecast and selection. Do not rerun
the original V1 backtest or any fitting merely to inspect saved reports.

The single private handover registry is an exact dated, provisional execution
exception supported by separately pinned evidence and approval. It does not
promote source identity, merge canonical records or change model inputs or
selected rows. Every other runtime route conflict remains fail-closed;
verified merger-converted-lot handling is unchanged. Missing held observations
retain exposure and withhold affected full-period metrics.

Master section 23 is a passive saved-report reader. It requires the exact three
completed phase audits and three matching independent verification receipts;
Run All starts no build, fit, model reload or source-price scan. Keep daily
feature history, monthly signals/outcomes and daily execution data separate.
The actual independent accounting script is
scripts/verify_walkforward_backtest_v2.py; the preserved V1 verifier is historical.
All three phase audits and matching independent receipts passed. No further fit
or backtest is needed to display the saved section 23 reports.


## Current source-corrected handoff, 2026-10-16

Read CURRENT_RESEARCH_STATUS.md and docs/SOURCE_CORRECTED_WALKFORWARD_20261016.md
for the completed corrected-source experiment: 16 annual fits, eight train-only
preprocessing states, 24 complete daily scenarios and 40 withheld scenarios.
RF expanded / class1/10 / 10% asset stops is the primary focus. Master section24
reads saved evidence only; no fitting or backtesting is triggered by publication.
The source-repair phase pinned README.md, SOURCE_QUALITY_STATUS.md and its repair
report before refitting. Keep those exact historical bytes; their pending-refit
statements describe that earlier phase. Earlier current-status blocks here and
in other reference documents describe their own source versions. Never weaken
hashes, remove unsupported holdings, or rank all models on complete paths only.
Raw data, source evidence, outputs, fitted models and PDFs remain outside Git.


## Completed RF tuning and daily replay, 2026-10-20

CURRENT_RESEARCH_STATUS.md is the current entry point. Read
`docs/RF_EXPANDED_TUNING_RESULTS_20261020.md` and
`docs/RF_EXPANDED_TUNED_BACKTEST_RESULTS_20261020.md` before interpreting the new
results. The immutable design-only protocol, later execution authority and
restart amendment describe separate historical stages. The user-authorized
search completed 96 inner fits, four annual refits and sixteen training-only
preprocessing states. Annual selection uses only three chronological validation
years / 36 equally weighted monthly log losses. No portfolio score chooses RF
settings. The independent tuning receipt is PASS_COMPLETE with its documented
float32/schema limitations.

The separate additive zero-fit replay completed sixteen scenarios: eight
complete class 1/10 paths and eight withheld class 2/9 paths. Use
`scripts/verify_rf_tuned_backtest_v2.py` and
`scripts/watch_rf_tuned_backtest_v2.py` for the reviewed verification-only
correction; preserve their V1 predecessors and the failed verification evidence.
Only the side column's Arrow string storage width differs in four selection
files; exact semantic frame equality, all raw pins and independent accounting
remain required. Never loosen value comparisons to resolve a failed assertion.

Keep the original corrected models/backtests as comparators, RF expanded with
10% per-asset stops as primary, the fixed basket/cost/execution rules, withheld
exposures, stopped attempt and pause/resume checkpoints. The master remains
passive; a fresh clone lacks private datasets and run evidence. No new fitting,
resume or replay is needed to read these summaries. Already examined 2020–2023
results remain development evidence; protected 2024+ values stay unread.


## Current corrected-input refit and RET handoff, 2026-10-21

CURRENT_RESEARCH_STATUS.md is the current entry point. Read
`docs/CORRECTED_REFIT_AND_RET_BACKTEST_20261021.md`,
`docs/CORRECTED_BACKTEST_COMPARISON_20261021.md` and the source-only guide at
`scripts/corrected_refit_20261021/README.md` before interpreting this version.
The corrected BK/XOM catalog and48 causal RET states supply20 fixed-setting
annual fits and8 shared training-only preprocessing states. The fifth RF arm
reuses prior annual CV settings; no new search was run. Original targets,
cohorts, primary80% context, daily/monthly separation and trading rules remain.
All120 paths were independently checked:56 complete and64 withheld. Keep
missing/unvalued holdings and the selected-provider/held-source collision
explicit; do not compare models on a favorable available subset alone.

The frozen executed worker/test bytes are published in the dated source folder.
Their safe `run_offline_synthetic_tests.py` entrypoint uses only explicit invented
tests; do not indiscriminately import historical dated audit helpers or run their
CLIs. Some preserve absolute private paths and top-level one-off operations.
A fresh empirical run needs a separately versioned workspace, approved private
inputs, explicit measured projections/read reservations and guarded launches.
Publication starts no fitting, resume, data download or backtest. The current
notebook remains a passive reader of its own older frozen evidence; no active
pointer is promoted. Preserve the historical README/source-status bytes.

The final verifierV5 preserves Arrow-null optional identifiers in four closed
nonfinancial diagnostics after unchanged metadata fingerprints, with no NaN
comparison waiver or financial rule change. Preserve failed attempts, checkpoints
and all old runs. See the exact failure/gap notes and endpoint-label decision in
the dated source folder. Repair new quote/action/identity evidence and audit any
learning/RET impact before changing labels. Protected2024+ financial values,
knowledge-time identity, survivorship and investability remain outside a full
certification claim. Raw data, receipts, plans, model artifacts and ledgers stay
out of Git; only original code/methods and aggregate summaries are published.
