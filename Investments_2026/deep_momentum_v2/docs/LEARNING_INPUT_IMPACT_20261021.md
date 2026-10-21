# Learning-input impact audit — 21 October 2026

Status: **audits completed; production inputs remain frozen**. The unresolved source coverage and economic-accounting work below comes before alternative labels, model tuning or another model comparison. No fitting, inference or backtesting was performed.

## Scope and lineage

The metadata audit covers all **168 outcome months, January 2010–December 2023**, retaining every original historical S&P500/S&P400 cohort row. It follows the fourteen corrected annual source records pinned in the original RET run at `outputs/ret_backtest/2020-2023-20261020-v1/build_identity.json`, rather than the older full-target-panel cache. Signals run from 2009-12-31 through 2023-11-30. Existing 80% context, provisional unique-route identity, Paper15, missingness and rolling-ten-year rules remain in force. No financial 2024+ source columns were decoded.

The numerical experiment is deliberately narrower: one **conditional RCM November 2021 endpoint repair**, under the existing close-to-close adjusted monthly-return and ten-class midpoint-rank definition. It is a shadow diagnosis, not a promoted replacement panel. The separate feature audit is structural and does not measure changed feature values.

## Full original-cohort reconciliation

| Saved row population | Stock-months |
|---|---:|
| Original historical membership | 152,307 |
| Identity eligible under existing provisional policy | 149,824 |
| Signal-time prediction eligible | 140,680 |
| Observed valid labels | 140,387 |
| Prediction eligible but missing label | 293 |

Prediction eligibility covers **92.37% of original rows**. Label availability is **99.79% of eligible rows**, or **92.17% of original rows**. These are distinct denominators. Of 11,627 signal-ineligible rows, 2,483 are identity-ineligible and 9,144 are Paper15-ineligible. No rows were dropped from the audit.

The 293 missing labels span 139 months and 288 original source IDs: **277 missing end prices plus 16 end-price quality flags**. All are retained in the exact-key review queue. Twenty-nine missing-end cases overlap the pinned 18 current and 11 historical legal events; **248 missing-end cases and all 16 quality cases have no overlap in that registry**. A registry overlap is a candidate dependency, not proof that it explains the missing price; an unmatched gap is not evidence that a dataset is absent or that a stock delisted. All 29 known events have an original event-month overlap and no original post-effective membership rows in these saved cohorts.

The metadata retains 622 signal/native collision groups (1,244 blocked rows) and 1,113 unknown review-flag values. Unknown review flags are counted explicitly, never treated as passed checks.

## Coverage before eligibility is a material separate queue

The eligible-only missing-label queue does not expose all source-history loss. Saved routes labelled **XOM, BK, SW, PSKY, OEI and BGEN** each have 168 original cohort months, no matching start/end prices and zero prediction-eligible months. Other routes have long partial gaps. These are **route/source-coverage candidates**. This audit does not establish that the companies lacked market history or that those canonical identities are wrong; ticker labels alone are not authority for a remap. The full source/native/year diagnostics retain original keys, flags and denominators.

For RCM's exact original source key, membership begins in the saved June 2021 outcome cohort:

| Outcome year | Original rows | Prediction eligible | Labels available |
|---|---:|---:|---:|
| 2021, June–December | 7 | 6 | 5 |
| 2022 | 12 | 0 | 0 |
| 2023 | 12 | 6 | 6 |

Some later returns exist while lookback/feature eligibility remains unavailable. This is why restoring one missing target is not a complete RCM repair. The old/new provider first/last inventory dates and 138 later old-route date keys do not validate a full historical remap. The reviewed execution remap remains limited to its 22 dated observations.

## Measured conditional RCM effect

Before applying the conditional repair, every saved November 2021 class/status was reproduced, the month's class means matched the original RET ledger, and **all 48 saved RET logical payoff fingerprints reproduced exactly**. Inherited source byte pins were checked before and after.

The same 910 original rows and 873 prediction-eligible rows remain intact. Restoring the observed November 30 RCM adjusted close raises observed labels from **872 to 873**. Its unchanged-formula monthly return is +9.7696%, placing it in class 1. **Five existing stocks also change boundary classes**: 1→2, 3→4, 4→5, 5→6 and 8→9. Six class assignments change in total; only one return changes.

Eight monthly class means change. Their causal contribution changes 200 payoff cells across **25 monthly RET vectors**, starting at the **2021-11-30 signal for the December 2021 decision** and ending at the November 2023 signal. All 23 earlier vectors are unchanged; all 48 remain available. Maximum absolute payoff change is **0.04254 basis points** of return, consistent with the changed monthly class mean divided by 120. This isolated payoff magnitude is not an estimate of strategy P&L or a bound on later feature and selection effects.

December 2021 remains originally Paper15-ineligible. A repaired November anchor does not supply a missing December end price or establish a new December return. Whole selected annual Arrow column chunks can transiently decode excluded 2021 rows; only the filtered November target cross-section, old monthly means and two selected RCM endpoints were converted for numerical analysis. See the separate read-semantics disclosure; the December receipt is not a claim that every excluded physical column block was unread.

## Feature and fold implications

The 715 model-eligible names match the 728 registered definitions, including nonpredictor definitions. Exact daily dependencies can extend to **503 trading sessions**, due to a long abnormal-volume aggregation. The execution supplement has OHLC/adjusted OHLC but **no volume**; field presence also does not prove complete split/dividend semantics or valid history. It cannot be treated as a complete feature input.

RCM's approved dates can structurally affect Paper15/common-population normalization from November 2021 through November 2022. Discovery's five April 2022 interior bars do not directly change monthly endpoint labels or Paper15 endpoints. Daily/context feature dependencies for both cases can extend further, with per-field/date spans recorded in the private audit. Shared ranks, market states, original-denominator 80% gates and daily benchmarks can propagate constituent changes to other stocks. These are potential effects, not counts of changed values or newly eligible rows.

For the measured November label change alone:

| Saved fold | Potential consequence if shadow inputs are later promoted |
|---|---|
| 2020 | No dependency from this dated RCM repair |
| 2021 | November evaluation labels change; no change to the preceding training window from this label |
| 2022 | Full outer training includes changed 2021 labels; inner validation 2021 requires updated scoring |
| 2023 | Full outer training includes changed labels; inner validation 2021 and inner training for validation 2022 are affected |

Feature changes can additionally affect December 2021 predictions and later preprocessing/training. **Do not automatically repeat all 100 tuning fits.** Compare exact input hashes first; validation-only label changes may permit rescoring saved probabilities, whereas changed training inputs require their dependent preprocessing/fit. Artifact availability must be checked. The earlier unresolved routes and 2010–2014 quality cases can affect earlier training windows; the narrow statement about 2020 above does not certify the whole 2020 input history.

## Next work, in order

1. Investigate the full original-cohort route/source-coverage queue, including the fully absent routes, rather than only currently eligible missing labels. Require dated supporting evidence; preserve original membership and unresolved flags.
2. Obtain and validate complete dated RCM/Discovery **OHLCV and action-field** histories for the required intervals. Freeze sources, adjustment conventions, lookbacks and a bounded shadow-build budget. Do not extend the RCM remap from master first/last dates alone.
3. Specify monthly **economic-return accounting** for cash, stock and ADS events while retaining the monthly-return objective and original signal cohort. Cash and successor entitlements through month end need explicit rules; never manufacture predecessor month-end closes. Stops and trading costs remain execution policies.
4. Rebuild a separate shadow feature/target/class/120-month payoff version. Compare eligibility, raw/normalized values, cross-sectional effects and inner/outer fold hashes against the frozen production chain.
5. Reuse unchanged artifacts, rescore or refit only affected dependencies under a separately frozen compute scope, then rerun matched RET/class comparisons with the existing basket, stop and execution rules. Alternative risk-adjusted labels come after this foundation work.

## Validation and evidence

**47 focused synthetic tests passed**: 20 metadata, 19 conditional-return/RET and eight dependency tests. Independent readers reconciled saved metadata and reproduced the new class/support/mean/payoff results with an alternative rank implementation. There were zero fits, inference calls and backtests. **250 inherited production pins (36,083,911 bytes) were rechecked unchanged**; this is not a fresh hash of every model in every old run.

The first metadata attempt stopped on a nullable review-flag counter. Its frozen code, configuration and failure receipt are preserved. A separately sealed revision explicitly counts true/false/unknown flags and passed the audit. It did not change any production identity policy.

Primary private evidence is under `outputs/learning_input_impact/20261021-v1`:

- `handoff.json` — machine-readable next tasks and measured scope.
- `metadata/result-v2/audit.json` — full 168-month reconciliation.
- `metadata/result-v2/eligible_missing_target_review_queue.jsonl` — exact 293 keys.
- `metadata/gap_queue_findings.json` and `metadata/result-v2/source_native_year_diagnostics.jsonl` — broader route/history coverage.
- `counterfactual/result-v1/summary.json`, `changed_rows.parquet`, `causal_ret_vector_deltas.parquet` — conditional measured changes.
- `counterfactual/independent_result_review.json`, `feature_design/final_review.json` — independent output checks and limits.
- `feature_design/feature_dependency_summary.json` and `feature_dependency_audit.json` — input readiness and exact structural spans.
- `preservation.json` — inherited byte pins.

The frozen protocol is `configs/learning_input_impact_20261021.json`; tested audit source snapshots are in `scripts/learning_input_audit_20261021/`. They are additive audit tools, not a new production notebook stage. Original staged plans/receipts are immutable. `counterfactual/replay_plan.json` relocates the same code/content pins to the installed paths and is explicitly **not executed**. Use a fresh output directory for any replay; the metadata CLI also accepts the saved plan/config and a new output path. Structural planning scripts retain their original staging paths as provenance snapshots and need a new explicit output scope before reuse.

This establishes saved-metadata consistency and the bounded conditional effect. It does **not** certify every historical OHLCV value, knowledge-time identity, full survivorship correctness, a repaired learning panel or new model performance.
