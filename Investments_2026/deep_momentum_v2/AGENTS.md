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
inner validation remain proposals, not execution authority. model_design.py
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
