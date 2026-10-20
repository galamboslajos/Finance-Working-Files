# Corrected-source 2020–2023 walk-forward, 16 October 2026

Status: COMPLETE for the authorized refit/replay. All independent phase checks passed; 24 full paths are available and 40 remain withheld for unresolved coverage.

This separately authorized replay refits RF and XGBoost on Paper15 and expanded715, then repeats the frozen continuous daily backtest. Its root is `outputs/walkforward_source_corrected/2020-2023-20261016-v1`, identity `sha256:559ba2f7fbe187f192b40af2ab8fbe0a4fd70a1b843e2cbd0503f7e35729fcae`. All original runs, source snapshots, workers and pointers remain reference evidence.

The inputs incorporate the reviewed El Paso/Empire route correction and OPCH's 15 independently repaired 2022 OHLC bars. OPCH volume on those bars remains missing. Both known incorrect BBBY routes remain quarantined under the existing data-limited policy. Annual numerical artifacts are reused from the independently reviewed corrected input handoff; 2020–2023 copies have explicit original-audit lineage. This replay does not claim new feature calculations or certify all 715 formulas independently.

For outcome year Y, each estimator trains on completed outcomes Y−10 through Y−1 and predicts monthly in Y. The four windows are 2010–2019→2020, 2011–2020→2021, 2012–2021→2022 and 2013–2022→2023. There are exactly 16 sequential one-thread fits and eight training-only preprocessing states, with the original parameters and seed. Paper15 and expanded use the same supervised mask, retain every signal-qualified prediction row, and never impute a target. Expanded predictors retain train-only medians and explicit missingness indicators; the 80% context policy remains primary.

The daily replay keeps separate class1/10 and class2/9 baskets; symmetric name counts N=min(10, available long candidates, available short candidates); +50%/−50% initial allocations with equal weights within each side; next-session open execution; monthly rebalance; and unstopped/10%-per-asset-stop variants. Stops leave cash until the next monthly decision. Costs remain 0/5/10/25 bps per traded notional, with zero borrow/funding assumptions. There is no annual liquidation or NAV reset. RET is not included in this class-basket experiment. The frozen outcome horizon is 48 months and 1,006 XNYS sessions from 2 January 2020 through 29 December 2023.

Every newly selected maximum holding horizon is checked for available metadata before PnL. Any unresolved held mark retains exposure and withholds the affected continuous-path metrics. No new exclusions, synthetic prices, retirement terms or identity waivers are introduced. Existing reviewed action terms and the dated provisional PENN execution handover are rebound explicitly to this root; they do not promote source identity.

Explicit CLI entry points are `scripts/run_source_corrected_walkforward.py` and `scripts/run_source_corrected_backtest.py`. Model fits are separate annual commands, followed by finalization. Independent model verification uses `scripts/verify_source_corrected_models.py` per completed fold and `scripts/finalize_source_corrected_model_review.py` for the final stitched phase. Fresh independent accounting verification uses `scripts/verify_source_corrected_backtest.py`. None is triggered by opening or running the master notebook's saved-report section.

The old reference is `outputs/walkforward/2020-2023-aacd7579fcf21e6a/backtest_identity_v2`. Its returns used the previous source inputs and must be labelled obsolete-source reference in comparisons. A correction can change training data, forecasts and selected stocks; it cannot be represented by changing the old portfolio's PnL alone.

Historical reconstructed membership and the provisional identity policy remain limitations. Neither independent accounting checks nor a successful backtest certify knowledge-time identity, delisting completeness, attainable intraday stop fills, or borrow availability. No protected 2024+ financial values are used.

## Completed verified handoff

All sixteen frozen fits and eight training-only preprocessing states passed independent checks. Every saved probability reproduced exactly. Each arm retains 42,145 forecasts and 42,080 observed labels across 48 months; the 65 unlabelled forecasts remain present. The combined ledgers preserve all 43,655 original membership rows.

All 64 daily scenarios and their independent source-based cash/units audit completed. There are 24 complete proxy paths and 40 withheld paths. The complete set is the class1/10 baskets for XGBoost/Paper15, RF/expanded and XGBoost/expanded, with both stop alternatives and four cost settings. RF/Paper15 and every class2/9 full-period result remain withheld. This is not a complete four-model economic ranking.

RF expanded / class1/10 / 10% asset stops, 2020–2023:

| Measure | Gross | 10bps per one-way traded notional |
|---|---:|---:|
| Total four-year return | +29.18% | +18.82% |
| Annualized return | +6.62% | +4.41% |
| Annualized volatility | 20.78% | 20.82% |
| Annualized zero-cash-rate Sharpe | 0.41 | 0.31 |
| Daily-close maximum drawdown | −34.57% | −35.14% |

Net annual returns are +8.75% (2020), −5.97% (2021), −1.87% (2022), +18.42% (2023). These are slices of one continuous ledger. The net drawdown peak is 8 June 2020 and its trough is 6 November 2020. During that interval, long/short/cost contributions are −25.28/−9.28/−0.58 percentage points of peak portfolio NAV. The full net path has 513 stop exits: 261 long, 252 short, including 86 gap-open fills. Price-only stock attribution reconciles to the saved side ledgers.

The old RF expanded SL reference was +16.74% gross / +7.31% net, with −43.65% net drawdown. New forecasts, selections and NAV must be used together. This is not an isolated OPCH-fill correction. The unchanged unstopped RF expanded result is +47.79% net; the stop alternative remains the user's selected focus, rather than a retrospectively selected winning variant.

The net path's direct long/short/cost contributions are +38.60/−11.08/−8.70 percentage points. The difference between separately compounded gross and net runs is larger than direct cash fees because fees also change subsequent capital and position sizes. Contributions are not standalone leg returns and stop PnL is already included.

New maximum-horizon coverage flags involve membership labels NBL/WBD, ELV, CXO, RTN, VYX and FBIN. Actual first blocking marks are separately saved in `backtest_source_corrected/failures.parquet`; later preflight gaps must still be resolved after earlier blocks. Preserve source/quote identity and independently verify missing bars or legal conversion/entitlement terms before any further accounting repair. No selected stock was removed to complete a path.

Use master notebook section24 for the current passive saved summaries. Earlier sections retain their frozen source versions. Corrected model inputs are the exact annual references recorded in this run's build identity and `outputs/source_quality/confirmed-repairs-20261016-v1/corrected_annual_sources.json`. The old90% context sensitivity is still a reference with its original source history.

Private results: `outputs/walkforward_source_corrected/2020-2023-20261016-v1/reports/comparison_20261016/RESULTS.md`. All64 scenarios, all256 annual rows, old/new comparisons, stock/side contributions and stop exits are separate saved files in that directory. The final NAV/drawdown chart is in the sibling `plots_20261016` directory. Exact phase receipts are in this run's `verification` directory.

The new notebook cell passed an actual summary-only smoke check: exactly two bounded saved Parquet tables, no estimator reloads, no source-price reads and no output cells written. Every old notebook cell is preserved and its original exact bytes are archived with the run.

## Prediction and selection changes

The receipt-bound passive comparison in `reports/forecast_selection_changes_v2_20261016` confirms identical original keys, signal-time admission metadata and42,145 joint forecasts across source versions. RF expanded's hard class changes on34.31% of those forecasts. The saved class1/10 baskets retain323/480 previous long names and320/480 previous short names (67.29%/66.67%) across48monthly decisions. Model parameters, packages, seed, selection, clock, costs and stop definitions match exactly. Source repair plus refitting is a whole-workflow comparison; these differences do not isolate a fill-only causal effect.

The first optional comparison helper rejected224 null review flags on nonpredicted rows. Its code/error evidence are preserved. The additive V2 reporter keeps those raw unknown flags explicit and retains strict Boolean admission and valid unique native identities for every predicted row. It passed22synthetic tests, an eight-file actual metadata check and independent code review, then completed212forecast summaries and768selection-overlap rows. No model, feature, label, selection or daily accounting result changed for this reporting repair.
