# RET selection baseline: verified execution and repair handoff — 20 October 2026

Status: scoring, selection and accounting independently verified; all 40 full-period performance paths withheld. This is not a completed economic comparison or evidence that RET has poor returns.

The user directed finishing the originally intended RET strategy before alternative labels. Earlier published daily backtests used predicted-class baskets. Their returns were not RET results. Alternative labels remain on hold.

## What was implemented and run

Use all ten saved class probabilities at every signal, with `RET = sum(probability[class] * historical_realized_class_payoff[class])`. The payoff vector uses exactly the preceding 120 completed outcome months, equal-weighting stocks within a realized class/month and then equal-weighting monthly class means. All 1,200 month/class cells must exist for a signal to be admissible.

The five saved annual walk-forward model arms are RF and XGBoost on Paper15 and expanded features, plus tuned RF expanded. No estimator or preprocessor was fitted, no model was reloaded for inference, and no feature or target was changed. Corrected target lineage replaced the inadmissible older cached RET states. Original cohorts and missing-label forecast rows were preserved.

The experiment covers 2 January 2020–29 December 2023: 48 outcome months and 1,006 trading sessions. Rank RET over the full trade-eligible prediction cohort, with deterministic stable-ID ties and disjoint sides. All five arms selected 10 longs and 10 shorts at each of the 48 signals. This is the user's V2 breadth rule; the paper's percentage breadth is different.

Keep monthly rebalancing, 50% long/50% short starting allocation, next-session open execution, continuous NAV, existing corporate actions and restrictions, unstopped and 10% per-asset stops, and 0/5/10/25 bp costs per one-way traded notional. Thus there are 5 models × 2 stop variants × 4 costs = 40 RET scenarios. Original class-basket runs remain intact as comparators.

## Verified results and limits

- All 48 RET payoff vectors were available. The verifier independently recomputed 167 source outcome months, 1,670 unique class/month means and all 57,600 rolling window cells.
- The first signal, 31 December 2019, uses January 2010–December 2019 completed outcomes. The last signal uses December 2013–November 2023.
- All five original prediction cohorts were preserved: 43,655 original forecast rows, 42,145 prediction-eligible rows and 65 predictions with missing future labels per arm.
- There were no withheld selection months. Each model has 960 selected stock-month slots.
- Independent replay verification passed all 40 scenarios, including retained exposures at failures. None has a complete full-period price path. Total return, CAGR, volatility, Sharpe and drawdown remain null for every RET scenario.
- The independent PASS verifies the stated calculations and failure handling. It does not certify identity, historical knowledge-time availability, investability or the economic validity of an incomplete path.

## Exact first failures

These are first failures, not an exhaustive list of remaining data issues. All are missing adjusted opening quotes. Provider IDs below have not been promoted to confirmed company identities by this audit.

| Model arms | First failure | Intrinio security ID | Source security ID | Role | Affected scenarios |
| --- | --- | --- | --- | --- | ---: |
| RF expanded, RF expanded tuned, XGBoost expanded | 2020-08-03 | `sec_ypOpLz` | `siblis_ticker_company_2f8767243edbdb8465a4` | Held | 24 |
| XGBoost Paper15 | 2021-06-15 | `sec_zEbnBy` | `siblis_ticker_company_f7b9f16263798e24cff7` | Held | 8 |
| RF Paper15 | 2021-11-01 | `sec_zMM1vd` | `siblis_ticker_company_8500ae2d7e99a52f8713` | New selection | 8 |

The 40 failures count scenario variants, not 40 distinct bad stocks. The maximum-holding-horizon preflight found 61/32/37/45/31 unique missing date/security keys respectively for RF Paper15, XGBoost Paper15, RF expanded, XGBoost expanded and tuned RF expanded. These overlap across arms and must not be summed. Some potential gaps can fall after a stop exit. Fixing only the three first failures does not guarantee complete paths.

Missing selected/held observations did not authorize dropping names, replacing them with better-covered names, filling a zero return, or reporting partial-period gains as full-period results.

## Next RET work

1. Resolve company/security identity and event status for the three first failures. Check whether each missing opening quote reflects a genuine quote gap, terminal corporate action or an identity transition; do not infer the cause from absence alone.
2. Reconcile all remaining maximum-horizon preflight gaps for the fixed RET selections, including terminal entitlements and any prohibited new orders after retirement. Obtain dated independent evidence where needed. Do not add retrospective selection exclusions to improve coverage or returns.
3. Freeze a versioned repair overlay and its evidence, preserving this first run. Rerun the same 40 scenarios with the same saved forecasts, RET scores, selection rules and trading mechanics, and independently verify the new run.
4. Compare the completed RET paths against the preserved class1/10 and class2/9 reference strategies, prioritizing both RF-expanded versions with stops. Continue alternative-label work only after this RET comparison is clearly resolved or an explicit incomplete-data policy is agreed.

RET scoring targets expected monthly label return. It does not itself eliminate the difference between close-to-close labels and next-open daily trading, stops, costs and corporate-action cash flows. That alignment remains an empirical diagnostic rather than an assertion of equality. The already examined 2020–2023 interval remains development evidence, with provisional identity, same-close availability assumptions, provider-adjusted price proxies, retrospective DF/RDC restrictions, zero borrow/funding costs and idealized daily stop fills.

## Reproduction and evidence

The frozen method is `docs/RET_BACKTEST_20261020.md`; configuration is `configs/ret_backtest_20261020.json`. The entry point is `scripts/run_ret_backtest.py` and the independent verifier is `scripts/verify_ret_backtest.py`. Do not rerun into existing directories or interpret the frozen method document's pre-execution status as the latest result. The master notebook remains passive.

Run: `outputs/ret_backtest/2020-2023-20261020-v1`.

- Root audit: `audit.json`.
- Payoff/selection preflight: `preflight/audit.json` and pinned artifacts.
- Daily replay: `backtest/audit.json`, scenario failures and retained holdings.
- Independent verification: `verification/ret_backtest.json`.
- Passive comparison: `reports/verified_20261020/verified_ret_comparison.md` and `.json`; the JSON retains every RET and saved class-basket scenario, annual slices and first-failure details. Withheld metrics remain null and no incomplete subset is ranked.
- Resource receipts: sibling `outputs/ret_backtest_control/20261020-v1/{prepare,preflight,backtest,verify}.json`.

Validation: 95 new tests; 126 focused RET and target-integration tests passed. All 105 preserved prior code/protocol pins and all 12 new installation file hashes reconciled after execution (including the explicitly documented extra pre-execution Parquet round-trip test). No old runs or pointers were altered.

All four execution phases passed their process budgets. Maximum watchdog-sampled process-tree RSS was 1.66 GiB; cumulative conservative compressed read reservations including verification/reporting were 268,162,957 bytes against 1 GiB. New run/control artifacts were approximately 25.3 MiB before this handoff, against 512 MiB. No protected 2024+ financial rows were converted or used. Mixed annual Arrow blocks may decode before authorized-row filtering, as the frozen method explains.

Independent receipt SHA-256: `be73bbac727f36d090ec34367bebbd3d09d5dc92ba49fe1c35319bd00207f49c`.

These additive RET changes and this report are local; this request did not ask for another commit or merge.
