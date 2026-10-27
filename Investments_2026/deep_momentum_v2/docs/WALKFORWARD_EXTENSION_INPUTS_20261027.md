# Paper15 walk-forward extension: verified input readiness

USER DIRECTION, 27 October 2026: extend the existing four-arm walk-forward as far as the data supports. This first bounded phase is complete: **28 new outcome months, January 2024–April 2026, zero fits and zero backtests**. All old sources, panels, models, forecasts, performance runs and frozen guards remain unchanged.

The current four arms are RF/XGBoost on Paper15, with rolling five/ten-year training, matching complete past daily OHLC eligibility and matching 60/120-month RET histories. This does not extend the separate expanded-RF experiment or change the target. The strict target remains the adjusted-close ratio at exact calendar-month exchange endpoints, not an executable open-to-open return. There is no tuning, new winsorization, missing-target imputation or expanded feature bank.

## Supported horizon and measured coverage

The old 2023 stop was a frozen protocol cutoff. Pinned downloaded price dates reach July 2026, while historical S&P 500/400 membership reaches March 2026. March's signal supports April's holding period. The new numerical input audit reaches **30 April 2026**. A continuous matched economic comparison could therefore cover **February 2020–April 2026, 75 months**; the idealized monthly-only January 2020 start would cover 76. Neither longer backtest has run yet. Later source price dates alone do not extend historical index membership.

All **25,460 original constituent-month rows** survive in each history arm. The provisional identity rule admits 25,330 before complete-history screening. No future price availability enters signal eligibility. Features use exact source-locked momentum indexing, one common history-qualified population and sample SD; all fifteen are finite for eligible rows.

| Matching history arm | Eligible stock-months | Share of original cohort | Observed monthly labels | Missing labels | Complete RET vectors |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5 years | 23,589 | 92.65% | 23,558 | 31 | 28 / 28 |
| 10 years | 20,949 | 82.28% | 20,925 | 24 | 28 / 28 |

These are stock-month coverage counts, not model accuracy, portfolio performance or counts of failed portfolios. Missing label rates among eligible rows are approximately 0.131% and 0.115%. A missing future outcome remains in prediction coverage; only observed, completed labels can enter supervised fitting.

| History arm | Outcome year | Months | Eligible / original stock-months | Eligible share | Missing monthly labels |
| --- | --- | ---: | ---: | ---: | ---: |
| 5 years | 2024 | 12 | 10,020 / 10,904 | 91.89% | 9 |
| 5 years | 2025 | 12 | 10,165 / 10,920 | 93.09% | 15 |
| 5 years | 2026 through April | 4 | 3,404 / 3,636 | 93.62% | 7 |
| 10 years | 2024 | 12 | 8,898 / 10,904 | 81.60% | 6 |
| 10 years | 2025 | 12 | 9,032 / 10,920 | 82.71% | 13 |
| 10 years | 2026 through April | 4 | 3,019 / 3,636 | 83.03% | 5 |

## Explicit route and economic dependencies

BK and XOM's independently reviewed shadow source routes end in 2023. December 2023 uses their approved computational keys; their 54 later constituent-months remain visible with `reviewed_route_interval_ended`. The extension does not extend those intervals by assumption, revert to the disputed originals, drop their original member rows or call them repaired. XOM's January 2024 future endpoint is unavailable on the old SIP computational route and is one of the five-year arm's missing labels. Six other constituent-month rows have target collisions and 70 have no Intrinio route. The active provisional waiver remains; there is no new general identity-certification prerequisite.

The other missing monthly endpoints include securities involved in later acquisitions or corporate events. The saved private missing-label ledger records exact source/native IDs, signal/outcome dates and missing-price state. Event types and legal terms have not been newly verified in this phase; a missing quote is not an assumed payout. Future economic action/quote handling must retain selected exposure and may not remove a stock using hindsight. These missing outcomes have not been used to alter eligibility, fit settings or RET gates.

The strict learning convention remains frozen, including unavailable merger-month labels and their omitted realized class means. A later execution-only repair must not silently change training labels or RET. Any learning-data correction needs a separate declared impact/refit decision.

## Six annual windows, twelve future estimator fits

| Annual forecast | 5-year completed outcome training | Supervised rows | 10-year completed outcome training | Supervised rows |
| --- | --- | ---: | --- | ---: |
| 2024 | 2019–2023 | 49,313 | 2014–2023 | 49,013 |
| 2025 | 2020–2024 | 49,710 | 2015–2024 | 56,716 |
| 2026 through April | 2021–2025 | 50,022 | 2016–2025 | 64,514 |

Each window contains all expected 60/120 outcome months and ten classes. RF and XGBoost share the same input and label masks within each arm. Twelve additional fixed annual fits are the prospective extension when these input pins remain unchanged. None has started. Source/route corrections that alter historical cross sections must be audited before claiming the old models can remain reused.

## Verification, source boundaries and resources

The sealed recipe is `configs/walkforward_extension_inputs_20261027.json`, with explicit 2000–April 2026 financial scope only for this new worker. Original loading/holdout guards are unchanged. Existing downloaded membership/identity manifest versions stay pinned; newer cloud versions were not promoted. The separate metadata check's three exact new-release object 403s do not block reading these pinned local inputs and are not dataset-absence evidence.

The plan verifies full SHA-256 hashes for the source objects it reads, pins parent proofs, projected columns, row groups, code and date bounds before financial conversion. It reserves 907,824,692 projected/cache bytes per pass under a 2 GiB ceiling. Two bounded passes produced and independently checked the numerical inputs; the supplementary coverage reconciler reads only a separately sealed metadata/class projection under 64 MiB. Parquet chunks can transiently decode excluded Arrow values; masks apply before Python conversion or returned/saved values. Dates after April 2026 never enter outputs or analysis.

The producer reuses exact verified pre-2024 validity and monthly anchors. It reads pre-2024 raw history only for 85 newly requested native identities and post-2023 OHLC for the bounded native inventory. It saves the 584 new daily sessions separately as raw/adjusted eight-field OHLC with quality masks, plus history validity, monthly anchors, original cohorts, features, labels and RET grids. Daily bars remain available for the next backtester; they are not replaced by monthly aggregation.

Independent verification authenticates the old prefix unchanged, reconstructs all 906,634 requested raw source rows and new cache cells, and checks 668,070 eligible Paper15 cells, all monthly return/class formulas and all 56 RET vectors. Membership reconstruction and routing use the same pinned metadata adapter in both passes; they are not an independently recertified identity source. A second independent aggregation checks monthly coverage and all six annual training masks against the saved counts. Source identity, retrieval-vintage adjustments, historical knowledge time, ordinary distribution completeness, actions and investability remain provisional.

The two input phases finished in 28.12 active seconds at approximately 1.00 GiB peak worker RSS, below the 8 GiB ceiling. Eight adversarial structural tests passed, including future-price poisoning, matching-window missingness, collisions, route interval boundaries and the zero-fit scope. No fitting, cloud mutation, source upgrade or old-run rewrite is hidden in notebook Run All. The numerical input workers do use NumPy/pandas; the passive JSON reader imports no model or numerical library.

## Follow in VS Code and continue

Open `notebooks/14_walkforward_extension_inputs.ipynb`. Its ignored local JSON selects the private run; a clone without private evidence displays that state and launches nothing. Run All authenticates proofs and reads only the small saved JSON coverage summary.

The private run is `outputs/walkforward_extension_inputs/20261027-v1`, with the sealed plan in its sibling `plans/` directory. `prepared/` contains immutable daily and monthly artifacts, `verification/data.json` binds the source/formula check, and `report/verification.json` binds the aggregate coverage report. Failed or paused attempts must remain preserved; the explicit runner supports only clean phase checkpoint resume.

Next, settle the two dated source-route continuations and prepare a separately sealed twelve-fit/inference/RET execution recipe. Then extend the corporate-action-aware monthly and daily realization comparison continuously, with the same baskets, costs, stops and terminal marking. Preserve all original 47/48-month result versions. Report the new totals with exact periods and separate annualized CAGR; there are no new strategy returns to report from this input audit.
