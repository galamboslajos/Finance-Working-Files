# Paper15 extension models: independently verified results

The 12 new annual fits and monthly predictions for **January 2024–April 2026 (28 months)** are complete and independently verified. These are the four isolated Paper15 RF/XGBoost × rolling five/ten-year arms, using the strict monthly close-return target and unchanged RET. They do not extend or replace the main expanded-RF experiment.

This phase produces **classification performance and fixed RET picks only**. No new portfolio-return, CAGR, Sharpe or drawdown result has been calculated. A continuous daily comparison over February 2020–April 2026 would cover 75 months; it remains the next economic phase.

| Paper15 arm | Mean monthly log loss ↓ | Mean monthly accuracy ↑ | Observed / predicted stock-months |
| --- | ---: | ---: | ---: |
| RF / 5 years | 2.282878 | 12.57% | 23,558 / 23,589 |
| XGBoost / 5 years | 2.285083 | 12.46% | 23,558 / 23,589 |
| RF / 10 years | 2.284015 | 12.13% | 20,925 / 20,949 |
| XGBoost / 10 years | 2.286926 | 12.04% | 20,925 / 20,949 |

Each metric averages monthly values equally across all 28 months. Uniform probabilities give log loss 2.302585; full uniform and annual training-prior references and pooled metrics are saved separately. Accuracy is the fraction of exact ten-class mode predictions that match the observed next-month class. It is neither precision for selected trades nor a trading win rate. The two history arms have different eligible universes. The 31/24 missing labels remain in the forecast ledger and are not used to screen prediction eligibility.

## Fits, provenance and preserved history

There are exactly 12 new estimators: both RF and XGBoost for each history window and forecast year 2024, 2025 and 2026. The 2026 metrics cover four months. All source/feature/target/RET pins, class mappings, parameters, seeds and training clocks remain unchanged from the sealed model extension recipe. No tuning, imputation, calibration, early stopping, source upgrade or old-model refit occurred.

Independent checks reconstructed the six training/prediction matrices before fitting, verified causal cutoffs, reloaded all twelve new saved estimators, reproduced inference and reconciled probability metrics and ten-class confusion/support/precision/recall. Old 2020–2023 forecasts and all selected names, sides, weights and RET scores remain exactly unchanged in the joined 76-month monthly forecast ledger.

The original controller completed all12fits and its independent model checker, then stopped during the old-pick parity check: pandas represented `side` as object in one view and a string dtype in the saved view. The original run remains `FAILED_PRESERVED`, including its error/logs and consumed fit attempts. No numeric inequality triggered that failure.

A separate sealed zero-fit recovery casts only the two parity-check `side` views to the same string dtype. All numeric values and exact comparisons remain unchanged. Structural tests confirm a one-ULP RET change or any side/weight change still fails. The independent recovered selector checked all 304 monthly baskets (4×76), including112added baskets. The recovery fitted and loaded no estimator and ran no backtest. Its proof handoff authenticates the inherited 12-model PASS and the separate recovered selection PASS without rewriting the old ledger.

The original run is `outputs/walkforward_extension_models/20261027-v1`; its original sealed plan remains in sibling `plans/`. Recovery evidence is `outputs/walkforward_extension_selection_recovery/20261027-r1`; the original run's `verification_recovery/selection_handoff.json` is the composite proof entry point. The passive reader reports original and recovered states explicitly. All generated data/models/receipts remain private and ignored by Git.

Total active compute through independent selection recovery was207.24seconds (about3.5minutes), including the original consumed attempt. Controller-observed peak worker RAM was662,601,728bytes, approximately0.62GiB. The 8GiB budget was a ceiling rather than allocated/reserved RAM. Readonly notebook15 imports no model; it exposes progress before verified completion and only bounded authenticated JSON afterward.

## Selected coverage and the next daily backtest

Every arm has560new selected stock-month rows:10long/10short over28months, with unchanged50%long/50%short equal side weights. Selection used probabilities and past RET means only. It did not use future return or daily quote availability.

| Paper15 arm | Added monthly baskets | Selected rows | Selected missing monthly endpoints |
| --- | ---: | ---: | ---: |
| RF / 5 years | 28 | 560 | 1 |
| XGBoost / 5 years | 28 | 560 | 0 |
| RF / 10 years | 28 | 560 | 0 |
| XGBoost / 10 years | 28 | 560 | 0 |

The one selected endpoint case is **U.S.Steel (X), June2025, short in RF/5year**, signalled30May2025. Public legal terms identify a cash-merger treatment: common shares converted to a $55cash entitlement at the18June2025 closing, with trading removal before that day's open. See the issuer's [18June2025Form8-K, introductory note and Item3.01](https://www.sec.gov/Archives/edgar/data/1163302/000110465925060674/tm2517314d1_8k.htm). These terms have been researched but have **not been installed into any price, target, RET or economic ledger**. The next execution recipe must pin the evidence and apply signed raw-share entitlements, including the liability on a short, rather than dropping X, inventing a quote or changing its forecast.

The endpoint count is not a full daily preflight. Unselected labels may remain missing, and selected intramonth open/high/low/close, boundary-sale, stop and action marks still need an independent execution check. Zero monthly endpoint gaps is not proof that a daily path is complete. The original cash/stock-event and quote repairs must survive continuous year boundaries; no annual NAV restart or implicit liquidation at the old2023endpoint is permitted.

BK/XOM's audited post2023route limits remain explicit in the signal population; they were not silently repaired or reverted. A metadata check found neither in December2023orJanuary2024 boundary baskets. Their missing route continuation still limits full-universe claims. General historical identity/knowledge-time/adjustment/investability uncertainty remains; the existing provisional waiver is unchanged.

Next: seal the75month accounting extension, carry the saved baskets continuously, pin the Xcash entitlement and check every required new daily mark. Then compare monthly/same-close/next-open realization, gross/net costs and both no-stop/10%asset-stop variants under the existing rules. Report exact periods, cumulative total return and annualized CAGR separately. Do not change or refit learning inputs from execution-gap repairs without a separate versioned decision.

Open `notebooks/15_walkforward_extension_models.ipynb` for verified model metrics, annual diagnostics and selection coverage. No merge is authorized by this round.
