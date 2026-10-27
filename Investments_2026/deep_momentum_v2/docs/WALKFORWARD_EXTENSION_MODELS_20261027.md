# Fixed Paper15 annual model extension through April 2026

USER DIRECTION, 27 October 2026: continue the longest-supported current
four-arm comparison after the independently checked input build. This
separately sealed model phase fits exactly twelve estimators: RF/XGBoost,
five/ten-year rolling training, forecast years 2024, 2025 and 2026 (through
April). It does not launch an economic backtest or change a source, target,
feature, seed, estimator setting, eligibility rule, RET gate or trading rule.

Use `configs/walkforward_extension_models_20261027.json` and the explicit
`scripts/run_extension_models.py` plan/run/pause/resume CLI. The original
workers and their protected-2024 guards remain unchanged. New probability
metrics have the same formulas under a separately bounded extension worker.
The old 2020–2023 sixteen models and forecasts are reused as immutable evidence;
this phase loads only the twelve new serialized estimators during verification.

## Input choice and chronology

The source version is exactly the verified
`outputs/walkforward_extension_inputs/20261027-v1` audit, plus its byte-pinned
original monthly learning history. All original 177,767 stock-months per arm
remain. Fifteen finite predictors are allowed; no size, expanded features,
imputation, missing indicators, searches, validation-driven refits, early
stopping or calibration. Both estimators share each arm's exact X/y masks.

The strict training target is next-calendar-month adjusted close/preceding
signal close minus one, ranked into exact-tie ten-class bins. Class 1 is best.
Annual fits include only completed outcomes from the preceding N calendar
years released by the preceding December close. The 2024 five/ten-year windows
are 2019–2023 / 2014–2023; 2025 uses 2020–2024 / 2015–2024; 2026 uses
2021–2025 / 2016–2025. Forward target availability never filters predictions.
The four arms’ added predictions cover January 2024–April 2026, 28 months per arm.

BK/XOM's source-route intervals remain exactly as in the audited input build:
their 54 post-2023 constituent-month rows remain visible and unqualified rather
than falling back to disputed IDs. No route is silently repaired or extended.
This is the declared data-limited population, not a certified full-universe
result. Its limitation does not require reopening general identity certification
or refitting unchanged old models. A later route correction that changes X,
normalization or targets needs an explicit impact audit and affected refits.

Missing strict monthly endpoints remain missing learning labels, including
corporate-event months. No guessed payout enters training or RET. The prior
user decision to repair realized portfolio accounting separately remains.
An execution-only repair after predictions cannot silently change this learning
input version or choose replacement stocks.

## RET selection and checks

After independent saved-model inference checks, score every eligible name
using all ten aligned probabilities times the unchanged 60/120 completed-month
class payoff means. Select up to ten disjoint long/short names, matching counts,
+50% / −50% equal side weights, stable source-ID ties and the original DF/RDC
restrictions. Selection accepts no forward target/return fields or quote
coverage screen. Save every original forecast row, including ineligible rows
with missing probabilities, and preserve missing subsequent outcomes.

Join the 28 new forecasts to the old 48 monthly forecasts. Independently verify
all 76 monthly RET baskets and prove every old selected name, side, weight and
RET score remains exactly unchanged. The prospective paired daily horizon is
February 2020–April 2026, 75 months; the monthly-only January 2020 start is 76.
Neither longer economic result is produced by this model phase. Planned costs,
10% asset stops, next-open execution and continuous NAV remain unchanged for
the following separately sealed realization step.

Before any estimator fit, an independent checker compares combined monthly
panels to their verified sources exactly, checks all 76 causal payoff states,
reconstructs six chronological training/prediction masks, and binds float32 X
and integer y fingerprints. Each fit checks those fingerprints before calling
the unchanged estimator helper. The independent model checker reloads all
twelve new estimators, reproduces inference, class mappings and probability
metrics, and verifies saved settings, missing-label denominators, confusion
counts and separate ten-class precision/recall.

The final independent selection checker reconstructs all RET scores and picks,
verifies old-pick parity, and inventories added selected monthly endpoint gaps
without deleting those positions. It aggregates prediction metrics over all
28 months using equal monthly weights; four-month 2026 must not receive the
same weight as a full year. Uniform and training-prior references accompany
each model. Five/ten-year cohorts differ, so their contrast changes more than
training length alone.

## Bounded local execution

Source/derived projections, code, package versions, parameters and hashes are
sealed before monthly-panel reads or fitting. Budgets: 64 MiB reserved source
projections per pass, 2 GiB independent derived reads per phase, 8 GiB peak
worker RSS, 4 GiB output, twelve fits, one sequential fit/thread, 600 seconds
per fit, 1,800 seconds per phase and three active hours total. These are stop
ceilings, not RAM reservations or a runtime estimate. No raw financial source,
daily panel, cloud operation or source upgrade is included.

Each attempted fit is consumed once. A failure preserves the attempt and stops;
there is no implicit retry. Pause/resume is allowed only at clean completed
phase/fit checkpoints. Preserve all original runs, incomplete attempts and
checkpoints. Notebook Run All reads verified small JSON reports and launches
no fitting, model loading or replay. All data, estimators, forecasts and detailed
ledgers remain private and ignored by Git. No main-branch merge is part of this
phase. Completion and metrics must come from independent saved proofs.


The twelve fits, independent inference and zero-fit recovered RET selection are
now complete. Read [the verified results and preserved representation failure](WALKFORWARD_EXTENSION_MODEL_RESULTS_20261027.md).
No extended economic backtest has run.
