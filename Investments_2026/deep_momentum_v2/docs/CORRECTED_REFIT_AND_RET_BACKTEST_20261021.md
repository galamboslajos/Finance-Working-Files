# Corrected-input model refit and RET daily backtest, 2020–2023

This is the completed, independently verified conditional research baseline. The previous runs, stopped/failed attempts and checkpoints are preserved. No parameter search, target-family change or trading-rule change was performed.

Annual refits use the previous ten completed outcome years: 2010–2019 for 2020, 2011–2020 for 2021, 2012–2021 for 2022, and 2013–2022 for 2023. The historical S&P 500 + S&P 400 union, primary 80% feature-availability policy, Paper15 and expanded 715 predictors remain fixed. Five arms (four baseline RF/XGBoost combinations plus RF expanded using prior annual CV settings) require 20 estimator fits and eight shared training-only preprocessing states. Prior CV parameters were reused; they were not retuned on corrected inputs.

**Backtest period: 2 January 2020–29 December 2023; 48 monthly outcomes, 1,006 sessions.** Monthly predictions feed fresh RET, class1/10 and class2/9 selections. NAV continues across annual refits. The 120 predefined paths are five arms × three baskets × two stop variants × four costs. Independent verification retained all120: **56 complete / 64 withheld**. Missing exposure remains in the ledgers; withheld full-period and annual performance stays null.

## RET with the 10% per-asset stop

The stated focus is RF expanded with SL; the baseline and prior-CV RF remain distinct. Gross uses0bp. Net uses10bp per one-way traded notional. Borrow/funding costs are0 in this frozen proxy. Returns and drawdowns cover the full four-year continuous path; annualization uses252 sessions/year, and Sharpe uses daily returns with a zero cash rate. A dash is unavailable, never zero.

| Model | Gross total | Net total | Net CAGR | Net volatility | Net Sharpe | Net max DD | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| RF · Paper15 | — | — | — | — | — | — | withheld |
| XGBoost · Paper15 | +39.41% | +27.16% | +6.20% | +21.17% | 0.39 | -25.56% | complete |
| RF · expanded baseline | +0.18% | -8.35% | -2.16% | +18.90% | -0.02 | -31.01% | complete |
| XGBoost · expanded | — | — | — | — | — | — | withheld |
| RF · expanded, prior CV settings | +26.69% | +15.87% | +3.76% | +19.11% | 0.29 | -22.71% | complete |

Full class1/10 and class2/9 comparisons, unstopped comparators, all0/5/10/25bp cost assumptions, continuous annual slices, coverage, stop/trade counts and turnover/cost totals are saved in run/reports/backtest_v1/report.md and report.json. These are simulated historical proxies, not live or investability-certified results.

## Prediction diagnostics

All five arms share prediction and observed-label keys. Accuracy is ten-class mode accuracy, separate from portfolio performance. Class1 is best and class10 worst. Observed-label coverage is measured among prediction-eligible stock-months, not the entire historical universe.

| Model | Pooled log loss | Equal-month log loss | Pooled accuracy | Observed labels / predictions |
|---|---:|---:|---:|---:|
| RF · Paper15 | 2.283152 | 2.283136 | 12.34% | 42,176 / 42,241 |
| XGBoost · Paper15 | 2.284890 | 2.284871 | 12.13% | 42,176 / 42,241 |
| RF · expanded baseline | 2.268530 | 2.268524 | 13.26% | 42,176 / 42,241 |
| XGBoost · expanded | 2.268343 | 2.268338 | 13.40% | 42,176 / 42,241 |
| RF · expanded, prior CV settings | 2.268729 | 2.268725 | 13.28% | 42,176 / 42,241 |

Separate class1/2/9/10 precision, recall and support, monthly diagnostics and annual metrics are retained in run/reports/models_v1/comparison.md and comparison.json. No paired improvement claim against old runs is made because input/cohort changes differ. This period was already reviewed during research; it is not a new untouched holdout. No 2024+ financial evaluation is included.

## Selection, remaining gaps and limits

RET ranks the full predicted probability distribution against the shared causal120-completed-calendar-month class payoffs. The separate class baskets select within their predicted classes. Both use up to10 names per side, equal counts per side, 50% long/50% short, equal weights within a side and next-session open execution. Stops use the unchanged daily-OHLC observation/fill assumptions; gaps can exceed10%, and stopped capital remains cash until the next scheduled rebalance. Future quote or target availability never chooses the basket.

Monthly training labels remain provider-adjusted close-to-close endpoint returns. RET therefore predicts that label payoff, not the exact next-open, stop-dependent realized portfolio payoff. This definitional difference remains a target-alignment research question; the completed run does not establish that label and daily execution returns coincide.

All15 fresh selections have zero unavailable selection months. The maximum-horizon preflight finds127 unique missing native/date keys across nine IDs, plus one quality-flagged key; these affect168 and2 episode rows respectively. The saved failure state, rather than this maximum-horizon check alone, determines each path’s availability. SELECTED_HORIZON_GAPS_20261021.md lists the exact identities, dates, affected models and remaining evidence needs. The descriptive names are PBCT, EV, NBL, HRC, PSB, WWE, WRI, FBIN and HWM; a name match does not independently establish an entitlement or historical identity.

The verified actual first failures are44 missing-quote paths,16 source-quality-flag paths and4 selected-provider/held-source identity-collision paths. The collision affects XGBoost expanded, unstopped class1/10, on2022-10-03; it is separate from quote availability. ACTUAL_FAILURE_DIAGNOSTICS_20261021.md and its hash-pinned JSON retain exact first-failure keys and costs. No tolerance or identity override was added.

The reviewed BK/XOM routes and prior40 execution-only quote repairs remain pinned. RCM/DISCA learning-bar candidates were rejected by the frozen volume-compatibility control; no such learning bars were admitted. XOM’s69 retained precision warnings remain audit evidence and no tolerance was widened. Provisional identity decisions, historical reconstruction/knowledge-time, provider adjustment vintage, terminal-return coverage and survivorship limitations remain unresolved. Independent checks establish the saved timing, preprocessing, forecasts, selections and accounting within this scope; they do not certify every source observation or eliminate these research biases.

Next priority: verify the remaining selected-horizon quotes, dated corporate-action entitlements and selected/held identity handover collision, then replay the same saved forecasts/selections in a separately versioned execution repair. Audit any resulting feature/label/RET-payoff impact before changing labels. A further model refit is warranted only if that audited learning input changes; this list does not authorize a new search or alternative label experiment.

## Evidence and archive

The installed package is outputs/corrected_refit/2020-2023-20261021-v1/. installed_catalog.json lists the actual archive locations; relocation_index.json translates the staging prefix while preserving embedded provenance byte-for-byte. No active pointer is promoted. The private archive includes model artifacts, predictions, all120 ledgers, failed attempts/checkpoints, frozen plans, source decisions, reviews and proofs. Reproduction requires a fresh attempt and frozen plan; never execute into this sealed archive.

- Model proof: run/verification/models.json, SHA256 9d0d6be8fee01428b6550c60659672e4f6a46b665cfafc72e0fc3200e76c94b2.
- Backtest proof: run/verification/backtest.json, SHA256 a0374f97f6193969f741cd2f59b19ea8e367eb851bdd48276b6763c04fe56aea.
- Passive report receipts: run/reports/models_v1/receipt.json and run/reports/backtest_v1/receipt.json.
- Input decisions: RECOVERY_INPUT_DECISIONS.md and MONTHLY_ENDPOINT_LABEL_POLICY.md.
- Integration/attempt provenance: RUNTIME_ENTRYPOINTS_20261021.md and run/control/.
- The installer adds archive_receipt.json only after opaque copy/hash verification; this pre-install document makes no earlier archive-success claim.
