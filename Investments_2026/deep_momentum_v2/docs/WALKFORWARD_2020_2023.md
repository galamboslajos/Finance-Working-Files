# Annual walk-forward experiment, 2020–2023

USER DIRECTION, October 16, 2026: merge the completed development repairs and
continue with annual retraining on the preceding ten years, monthly forecasts,
and the same baskets and stops. This document and
`configs/walkforward_2020_2023.json` define the next experiment before its first
feature build or fit. The exact quote-acquisition and acceptance plan is frozen
separately before its source/control price projections; its completed bundle is
bound into the feature recipe before building. Prior runs, source data and
pointers remain immutable.

| Prediction outcome year | Training outcome years | Latest training information |
| --- | --- | --- |
| 2020 | 2010–2019 | December 31, 2019 |
| 2021 | 2011–2020 | December 31, 2020 |
| 2022 | 2012–2021 | December 31, 2021 |
| 2023 | 2013–2022 | December 30, 2022 |

Training labels must be fully observed by the final XNYS session of the prior
year. Each fitted model forecasts twelve monthly outcomes from preceding
month-end signals. January's signal is the prior December month-end. The
prediction cohort depends on historical membership, the provisional identity
policy and past-only Paper15 availability; extra predictor missingness and
future labels never remove a prediction row. All original keys and denominators
remain in saved ledgers, including missing outcomes.

The four arms remain RF/Paper15, XGBoost/Paper15, RF/expanded and
XGBoost/expanded. Estimator parameters, package versions, seed 1729, probability
construction and metrics come from the frozen pilot recipe. Its old dates and
four-fit budget are a template reference only: the new explicit folds replace
them. There are exactly 16 sequential, single-thread estimator fits and at most
eight preprocessing fits. Each preprocessor uses only the common supervised
training mask and is shared between the two estimators for that feature set.
Medians, missing indicators and schema decisions never use the test year.
No parameter search, early stopping, probability calibration, performance-driven
refits or sensitivity-model fits are included. The 80% context rule is primary;
the saved 90% sensitivity remains unchanged. The paper's 100-fit, expanding
history, 16-input method remains distinct from this rolling, no-size adaptation.

The new feature phase reuses the corrected 2010–2017 training annuals and the
independently repaired 2018/2019 annuals. It computes new 2020–2023 annuals with
the same feature formulas and full past-only daily warmup. Earlier quote
corrections propagate into later warmup where applicable. The two demonstrated
wrong BBBY routes remain quarantined, with no invented replacement. Daily
feature history, disjoint outcome-year daily prices, monthly returns, predictors
and targets are saved separately. A metadata-only inventory records quote
quality flags, source extents and interior holes before financial decoding.
This inventory does not filter signal cohorts. New corrections require pinned,
real independent evidence and a separately versioned rebuild impact record.

Trading continues the fixed class 1/10 and class 2/9 baskets separately.
Within each mode-class pair, rank the corresponding class probability descending,
then stable source ID ascending. After the previously fixed Dean Foods/Rowan
execution restrictions, use N=min(10, available long candidates, available short
candidates). Hold N names on each side, each long weighted +0.5/N and each short
weighted −0.5/N. A zero candidate side withholds the full path. Restrictions do
not remove forecasts or valid historical training rows. New data problems never
authorize new hindsight exclusions or automatic replacement of selected names.

The daily ledger spans all 1,006 XNYS sessions from January 2, 2020 through
December 29, 2023 and 48 monthly signal decisions. Annual model updates do not
force liquidation or reset NAV. Normal monthly rebalancing uses next-session
open execution and the existing adjusted-price/share-unit accounting. The
unstopped variant and 10% adverse-price stop per asset are separate. Existing
daily gap and OHLC stop/fill conventions, per-trade cost sensitivity of
0/5/10/25 basis points and terminal liquidation remain unchanged. Stops reset
at the regular monthly rebalance. The final liquidation uses December 29, 2023
close; no 2024 execution price is read. Annual summaries are slices of the same
continuous ledger. Borrow and funding charges remain zero, as in development.

Before performance calculations, a preflight checks every selected maximum
holding horizon, required corporate-action predecessor/successor observations,
and invalid or missing quotes. Existing pre-2020 actions are a retirement
registry only: a new order after retirement never receives an old entitlement.
New-period entitlements require dated primary evidence and verified quote units.
Unsupported actions or unexplained held-price gaps retain exposure and withhold
full-period metrics. No fabricated exit, zero terminal value or deleted loss is
allowed. RET is not evaluated in this class-basket experiment; its 120-month
gate remains unchanged.

Execution is local against verified cached annuals, one process/fold and one
estimator thread. Feature bounds: 4 GiB peak RSS, 4 GiB projected compressed
reads and two hours. Model bounds: 8 GiB RSS, 1 GiB projected reads per fold,
90 minutes per fold and four hours aggregate. Backtest bounds: 2 GiB RSS and
15 minutes. Total new output ceiling is 12 GiB. Before each estimator fit,
write an attempt receipt; immediately after fitting, persist the model and a
completion receipt before inference or metrics. A saved completed fit can
resume inference without refitting. An interrupted fit without completion is
preserved and blocks automatic repetition. Notebook Run All never launches
this work. Every phase has its own pinned identity and source preservation
checks. No cloud mutation or financial analysis of 2024+ is authorized.

Independent verification is a separate bounded pass, with no estimator fits,
one thread, at most 8 GiB RSS and 30 minutes. It checks training-only medians and
prepared matrices, saved-model inference, monthly scores and daily accounting.

Explicit entry point, from the V2 directory:

```sh
.venv/bin/python -B scripts/run_walkforward.py plan
.venv/bin/python -B scripts/run_walkforward.py build
.venv/bin/python -B scripts/run_walkforward.py fold --run outputs/walkforward/RUN --year 2020
```

Repeat the fold command for 2021, 2022 and 2023, then use `finalize-models`,
`preflight` and `backtest` with that same `--run`. Run each phase in a fresh
process with BLAS/OpenMP thread ceilings of one. The initial backtest registry
is empty for in-period actions; preflight findings must be resolved with
evidence or explicitly retained as withheld paths before results are claimed.

INTERPRETATION: this remains a data-limited, restricted-universe research
simulation under retrieval-time identity evidence. Historical membership is
preserved, but full identity, knowledge-time, delisting and survivorship
certification are not established. The fixed retrospective trading restrictions
and remaining source/accounting limitations accompany every performance claim.
This is a new sequential walk-forward evaluation, not an untouched static model
test: completed earlier test years subsequently enter later training windows.
