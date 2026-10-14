# Deep Momentum V2 working instructions

Read README.md, docs/RESEARCH_PLAN.md and docs/DATA_STATUS.md before work here.
For features, labels, models or portfolios, also read
docs/PAPER_REFERENCE_HAN_QIN_2026.md and docs/US_REPLICATION_PLAN.md. Apply the
repository-local .agents/skills/han-qin-deep-momentum/SKILL.md when relevant.
For loading or universe changes, also read docs/DATA_FOUNDATIONS.md.
For notebook or chronology changes, also read docs/NOTEBOOK_WORKFLOW.md and
configs/research_workflow.json. The master notebook orchestrates worker modules;
keep reusable data/model rules in src/dmv2 and test them there.

- This is a U.S. equity project. Keep new code and outputs in this directory.
- User-selected chronology starts in 2000 with rolling ten-year training and
  one-year outcome tests. Keep the paper's expanding history as a source fact,
  separate from this V2 adaptation. Feature warmup and usable coverage remain open.
- The notebook defaults to one bounded 2000 engineering audit and no price
  sample. Keep its committed outputs empty and save private audit evidence
  under ignored outputs. Do not treat Run All as an implicit full-panel/model run.
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
  Quarantined identities are review tasks, not silently removed constituents.
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
