# Tuned RF daily backtest results — 20 October 2026

Status: COMPLETE AND INDEPENDENTLY VERIFIED for the frozen sixteen-scenario replay. Eight class 1/10 paths are complete; eight class 2/9 paths remain withheld. No new models, preprocessing or inference were run.

The period is 2 January 2020–29 December 2023: 48 prediction months and 1,006 XNYS sessions. Training rolls over the preceding ten completed years, with annual model refits already saved by the separate tuning run and monthly predictions. NAV is continuous across year boundaries.

## Primary comparison

Class 1/10 baskets, fixed 10% per-asset stops and 10 bp per one-way traded notional. These are simulated research returns; existing identity, data coverage, borrow and idealized fill limitations remain.

| Model | Total return | CAGR | Annualized volatility | Sharpe | Maximum drawdown |
|---|---:|---:|---:|---:|---:|
| RF · Paper15 | withheld | withheld | withheld | withheld | withheld |
| XGBoost · Paper15 | +23.45% | 5.42% | 21.86% | 0.35 | -31.18% |
| RF · expanded original | +18.82% | 4.41% | 20.82% | 0.31 | -35.14% |
| XGBoost · expanded | +3.41% | 0.84% | 18.64% | 0.14 | -31.82% |
| RF · expanded tuned | +42.78% | 9.33% | 20.69% | 0.53 | -35.14% |

Tuned RF with stops returned +55.11% gross and +42.78% at 10 bp costs. The net maximum drawdown remains −35.14%. The 2020 forecasts and all sixteen first-year policy accounting controls match the original RF exactly; this tuning does not repair the original 2020 risk path.

The same tuned RF without stops returned +94.49% net, with 18.13% CAGR, 22.28% volatility, 0.86 Sharpe and −31.82% maximum drawdown. This is a predefined comparator, not a retrospective promotion over the user-selected stop variant.

## Calendar-year returns

| Year | Original RF expanded with stops | Tuned RF expanded with stops |
|---|---:|---:|
| 2020 | +8.75% | +8.75% |
| 2021 | -5.97% | +4.26% |
| 2022 | -1.87% | +10.25% |
| 2023 | +18.42% | +14.23% |

These returns use 10 bp costs and are slices of one continuous portfolio. The selected RF parameters remain fixed by past-only 36-month validation log loss: depth12/leaf20/sqrt for 2020, depth8/leaf20/sqrt for 2021, and depth12/leaf100/sqrt for 2022–2023. No portfolio metric selected those settings.

## Costs, stops and coverage

| One-way cost | Tuned RF with stops: total return | CAGR | Sharpe | Maximum drawdown |
|---|---:|---:|---:|---:|
| 0 bp | +55.11% | 11.62% | 0.63 | -34.57% |
| 5 bp | +48.82% | 10.47% | 0.58 | -34.86% |
| 10 bp | +42.78% | 9.33% | 0.53 | -35.14% |
| 25 bp | +26.05% | 5.97% | 0.38 | -36.00% |

The primary path has 489 stop exits: 253 long and 236 short; 77 use gap-open fills. Stop proceeds remain cash until the next monthly rebalance. Ten names per side were available in every class 1/10 signal.

Class 2/9 has 21 missing maximum-horizon quote keys across native IDs sec_X6drPy and sec_z5lK44. All eight full-period variants remain withheld, with exposure retained rather than a synthetic exit or replacement. Four signals use the previously authorized balanced counts of 8, 2, 6 and 9 names per side. No zero-sided month occurs. The original four-model comparison still has 24 complete and 40 withheld paths; it is not a complete economic ranking of every arm.

## Verification and provenance

The independent V2 verifier passed all sixteen scenarios, including cash/units, fees, actions, stops, metrics and first-failure exposure preservation. All eight preflight/replay selection frames match exactly. The preserved V1 audit stopped before numeric verification because its raw-byte equality assertion rejected a side-column Arrow string-to-large_string re-encoding. The additive correction preserves both raw hashes and allows only that physical difference before exact frame checks. No backtest replay was required.

See [the frozen replay contract](RF_EXPANDED_TUNED_BACKTEST_20261020.md) and [the verification-only correction](RF_TUNED_BACKTEST_VERIFICATION_CORRECTION_20261020.md). Original code, data, runs, failed attempts and pointers are preserved. No protected 2024+ financial values were read.

Private run: `outputs/rf_expanded_tuned_backtest/2020-2023-20261020-v1`.
Numerical verification: `verification/backtest_20261020_v2.json`.
Saved comparison: `reports/verified_comparison_20261020.json`.

The replay took 29.5 seconds and peaked at 1.40 GiB aggregate process-tree RSS. Successful independent verification took 54.9 seconds and peaked at 1.51 GiB. The cumulative conservative read reservation including reporting is 207,903,470 bytes, below 512 MiB. The earlier failed check and diagnostic allowances remain charged.

2020–2023 is previously examined development evidence. Passing accounting and chronology checks does not certify all security identities, provider knowledge time, survivorship completeness, actual borrow availability or attainable intraday fills.
