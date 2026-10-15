# Full feature panel, 2010–2023

USER DIRECTION, 2026-10-15: build the historical S&P 500/S&P 400 union feature
panel and inspect coverage/missingness before modelling. This authorizes the
explicit local build below. It does not choose a final modelling universe,
fit preprocessing/models, run a backtest or authorize cloud mutations.

The build uses the same 728 definitions and 715 candidate predictor names as
the bounded workers, the active common-five paper normalization and the
accepted provisional identity policy. The active added-context policy now uses
the user-approved 80% primary original-cohort gate, with 90% as sensitivity; see [its exact contract](CONTEXT_AVAILABILITY.md).
The paper15 and individual temporal lookbacks are unchanged. No missing input is filled to make a feature appear available.

## Date roles and daily calculations

- Outcome years: 2010–2023. Their 168 monthly signals run from December 31,
  2009 through November 30, 2023; 152,307 original constituent-months are expected.
- Daily technical indicators use actual daily OHLCV/action observations on
  independent XNYS sessions, then take exact month-end snapshots. They are
  not calculated from monthly candles or from a monthly average of prices.
- Each annual batch begins with exactly 503 sessions through its first signal.
  The initial daily warmup begins January 4, 2008. Paper momentum retains its
  separate exact monthly endpoints and all intermediate observations.
- Context uses original membership on each historical daily date. A month-end
  cohort is never projected backward across a benchmark window. Missing
  identities, membership, inputs or benchmark pairs retain explicit nulls.
- Raw daily quotes are read through each year's final outcome month for later
  backtesting, including December 29, 2023 in the last batch. Before numerical
  feature conversion, Arrow masks the feature view at the final signal and
  filters off-session quotes with an exclusion audit. Raw data are preserved.
- The predictor table contains only original source-ID/signal keys and feature
  values. Identity flags, raw momentum, missingness causes and future-outcome
  fields are separate. Protected2024+ financial values do not enter analysis.

Daily quotes are therefore needed twice in the workflow: now as inputs to
technical/context features, and later for portfolio execution and valuation.
The original pinned daily source files remain unchanged. Annual raw-daily
artifacts keep disjoint outcome-year dates, plus the initial warmup in 2010.
Separate `daily_history_raw.parquet` artifacts preserve every annual raw read,
including a new entrant's preceding December signal and earlier warmup; these
overlap across years and must not be blindly concatenated. Separate
`feature_history_daily.parquet` artifacts preserve the actual exact-session,
past-only feature inputs. The raw daily artifacts are not a certified economic
backtesting panel.

## Recorded execution plan

`configs/full_panel_build.json` records the authorized period, population,
policies and resource ceilings. Metadata and footer planning precedes prices.
The measured metadata pilot reproduces all 168 signals/152,307 original keys.
For 2010 it reconstructs 434,764 daily constituent rows at 482 daily dates and
requests 954 provisionally admitted native IDs, determined before prices.
The conservative full-year price projection is about 1.15 GB of compressed
column chunks; fourteen batches project about 16 GB in total. Projected chunk
bytes are not the same as actual matching rows, RAM or physical disk I/O.

The local host has 32 GiB RAM / 10 logical CPUs. The plan uses one worker,
annual checkpoints, a 4 GiB peak-RSS ceiling, 20 GiB cumulative projected reads,
10 GiB artifact ceiling and three-hour elapsed ceiling. A measured first-year
build precedes the remaining years. Resource checks occur at annual boundaries;
external monitoring can interrupt a batch without invalidating completed years.
This uses downloaded sources and incurs no cloud compute or GCS write cost.

New compact workers reuse the frozen technical series, paper normalizer and
cross-sectional/market functions. Equivalence tests compare every technical
and context value against the original workers, including missing windows,
invalid metadata, identity/sleeve changes and future-input rejection. Compact
uint8 daily diagnostics preserve exact status categories without constructing
one Python dictionary per stock/month/feature.

## Run and resume explicitly

From a configured V2 environment:

```sh
PYTHONPATH=src .venv/bin/python -m dmv2.full_panel plan --root .
PYTHONPATH=src .venv/bin/python -m dmv2.full_panel build --root . --years 2010
PYTHONPATH=src .venv/bin/python -m dmv2.full_panel build --root .
PYTHONPATH=src .venv/bin/python -m dmv2.full_panel verify --root .
```

The build command is explicit; notebook Run All does not launch it. A new build
under the active 80% configuration receives a fresh code/configuration identity;
this promotion reuses the already verified panels and performs no rebuild.
The retained 80% primary and 90% sensitivity panels were built under their
original code/configuration pins. The master uses
`dmv2.context_panel_roles.read_context_panel_roles(catalog)` to verify their
archived recipes, artifact hashes and explicit role contract. The strict
`dmv2.full_panel verify` command intentionally rejects a historical run whose
worker hashes differ from current code; role promotion does not rebuild or
misrepresent that run as generated by the current code. Completed
year checkpoints are reused only after source/code/runtime/policy and artifact
hash checks. A code or source change creates a different run directory. A
partial run never updates `LATEST.json` or reports complete coverage. Interrupted
temporary directories are retained for investigation; they are not successful
partitions and may be removed separately after their contents are understood.

## Private artifacts and coverage

Under ignored `outputs/full_feature_panel/2010-2023-<identity>/`:

- `years/<outcome-year>/features.parquet`: all original monthly rows and728
  definitions, with real missing values preserved.
- `feature_coverage.parquet` and `original_cohort.parquet`: original identity,
  request, paper-population and availability ledgers, separate from predictors.
- `daily_prices_raw.parquet`: unchanged raw daily observations stored separately.
- `daily_history_raw.parquet`: complete annual raw history, including warmup
  and signal anchors; overlaps other annual history artifacts.
- `feature_history_daily.parquet`: exact-session daily inputs ending at the
  final feature signal; separate from raw outcome/backtesting observations.
- `daily_cohorts.parquet` and `exchange_calendar.parquet`: dated context and
  independent calendar evidence.
- Separate technical status codes, paper raw momentum and context diagnostics.
- Yearly audits with exact requests, source pins, row counts and artifact hashes.
- Descriptive CSV reports by feature, family, outcome year, universe sleeve and
  original versus paper-qualified population; initial training2010–2019 and
  nominal test2020–2023 are reported separately.

Only a completed fourteen-year build updates that build workflow's `LATEST.json`.
Role promotion does not update this historical pointer.
`configs/context_panel_roles.json` selects the verified 80% primary and 90%
sensitivity for the master, which displays their saved coverage reports. Feature and daily
datasets remain lazy until an explicitly bounded partition is selected.

The reports describe missingness; they do not select features or fit medians.
Use the [active normalization/missingness protocol](NORMALIZATION_PROTOCOL.md)
inside each later training partition and inner split. Never use full-period
coverage to eliminate a predictor for an earlier forecast. Common-five baseline
eligibility and added-feature missingness are separate denominators.

The user accepted provisional identity uncertainty. Original denominators and
causal feature cutoffs reduce specific selection/leakage risks; retrieval-time
identity mappings and adjusted-price revisions still do not certify historical
knowledge-time PIT or survivorship-free economic results. Terminal events,
cash flows, labels and execution remain separate future work. No model,
preprocessing fit or backtest is performed by this build.

## Historical complete-cohort execution, 2026-10-15

All fourteen annual partitions completed: **168 signals and 152,307 original stock-months**. Complete paper inputs occur on **140,777 rows (92.43%)**. The 715-name predictor allowlist contains **313 predictors with finite observations**; **402 are all missing** under the unchanged context gates. This is observed availability, not a full-history feature-selection rule.

| Outcome year | Original rows | All 15 paper inputs | Paper coverage | Technical value coverage among paper-qualified rows |
|---|---:|---:|---:|---:|
| 2010 | 10,824 | 9,226 | 85.24% | 93.71% |
| 2011 | 10,824 | 9,547 | 88.20% | 98.83% |
| 2012 | 10,824 | 9,608 | 88.77% | 99.32% |
| 2013 | 10,829 | 9,727 | 89.82% | 99.33% |
| 2014 | 10,848 | 9,778 | 90.14% | 98.92% |
| 2015 | 10,869 | 9,756 | 89.76% | 99.38% |
| 2016 | 10,899 | 10,046 | 92.17% | 99.67% |
| 2017 | 10,908 | 10,183 | 93.35% | 99.87% |
| 2018 | 10,909 | 10,337 | 94.76% | 99.36% |
| 2019 | 10,918 | 10,418 | 95.42% | 98.66% |
| 2020 | 10,916 | 10,500 | 96.19% | 99.20% |
| 2021 | 10,920 | 10,478 | 95.95% | 99.76% |
| 2022 | 10,911 | 10,542 | 96.62% | 99.87% |
| 2023 | 10,908 | 10,631 | 97.46% | 99.74% |

The initial training period 2010–2019 has 98,626/108,652 complete paper rows (90.77%); nominal tests 2020–2023 have 42,151/43,655 (96.55%). Every original key, identity-admission flag, paper eligibility flag, all 15 paper values and raw momentum diagnostics match the independent endpoint audit exactly. All saved artifact hashes were independently checked. All 149 raw source objects (1,792,717,079 bytes) were rehashed successfully without decoding protected values.

All 298 daily technical definitions have finite observations. Their feature-value coverage is **92.27% against all original rows** and **99.01% among paper-qualified rows**. All 415 context definitions (402 eligible predictors) are missing: every signal and index sleeve contains an identity-blocked original member, sufficient to fail the complete-cohort gates. This does not establish that every underlying price history is absent. Context was not relaxed or filled.

The disjoint daily table contains **3,563,099 observations including the initial warmup**. Complete annual raw reads preserve 9,507,584 rows with overlap, and separate past-only feature histories preserve 9,224,882 rows with overlap. The coverage aggregator reads completed feature panels; its `daily_data_consumed=false` field describes that aggregation worker, not the daily-input feature construction.

**427 structural/adversarial tests passed. All 19 master code cells** completed in a fresh kernel against the installed project; the versioned notebook remains output-free. Private evidence is in the completed run referenced by `outputs/full_feature_panel/LATEST.json`, with `coverage_review.json`, `independent_verification.json`, `source_hash_verification.json` and `notebook_verification.json`. The reviewed original/context denominators and missingness reasons are separate from predictor values. No preprocessing or model was fitted.

## Historical user-authorized 90% context rebuild

The `available_cohort_90pct` rebuild runs into a separate source/code/policy-pinned
directory. The older complete-cohort run is retained. Completion is accepted only
after all 14 partitions and independent comparison of original keys, all 15 paper
values/eligibility,298 technical values and every daily artifact against the
historical run. The master reads the new completed pointer and never rebuilds
implicitly. Rebuild results are recorded below after verification.

## Historical verified 90% context rebuild, 2026-10-15

All 14 annual partitions completed in the separate run
`outputs/full_feature_panel/2010-2023-9992dcb5c83acdf2`. The active
`LATEST.json` points to this new completed run. The historical complete-cohort
run is retained. There are168 signals and152,307 original stock-months.

**All 402 previously empty eligible context predictors now have at least one
finite observation:310 cross-sectional,62 market-state and30 relative-context.
None remain entirely missing over the full period.** Together with 15 paper and
298 technical inputs, all 715 eligible predictor names have observations somewhere
in the period. This is descriptive availability, not admission to every fold.

| Outcome year | Context names with observations /402 | Finite context values / original possible values |
|---|---:|---:|
|2010|56|2.19%|
|2011|95|3.59%|
|2012|233|20.34%|
|2013|315|44.07%|
|2014|333|48.83%|
|2015|349|64.09%|
|2016|402|89.24%|
|2017|402|94.45%|
|2018|402|94.95%|
|2019|402|87.85%|
|2020|400|91.66%|
|2021|402|96.12%|
|2022|402|97.02%|
|2023|402|97.52%|

Among paper-qualified rows, context feature-value coverage is 59.52% in initial
TRAIN2010–2019 and 98.08% in nominal TEST2020–2023; full-period coverage is 71.06%.
Against all original rows the corresponding fractions are 55.08%,95.58% and 66.69%.
Early missingness and the changing information set remain material. Do not use
these full-history reports to select predictors or fit an earlier forecast.

The two all-missing2020 columns are the universe rank and robust z-score of
`liq_abnormal_volume_ratio_252s`; every2020 signal fails that base's 90% group gate.
They are available elsewhere in the period. Individual exact temporal requirements
remain unchanged, including this base's503-session dependency.

Independent verification confirms every original key, identity admission and
paper eligibility flag, all 15 paper values and all 298 technical values are exactly
unchanged. The paper-qualified count remains140,777/152,307(92.43%). All 10
preserved artifact classes per year have identical SHA-256 bytes, including
3,563,099 disjoint-core daily rows,9,507,584 annual raw-history rows and9,224,882
past-only feature-history rows (the latter two include annual overlap). Raw daily
coverage still reaches2023-12-29; the final feature signal is2023-11-30.

All 310 eligible cross-sectional columns were independently recomputed from
unchanged technical values and original membership.41 return-based market columns
were also independently recomputed, and monthly/daily original denominators,
exhaustive requests and 90% gates checked. The source-file rehash verified all 149
objects,1,792,717,079 bytes, without decoding financial values.

442 structural/adversarial tests passed. The new run used1,858,192,840 output bytes,
2,906,701,824 peak RSS bytes and16,425,449,561 projected read bytes; elapsed build
time was1,791.38 seconds, inside the existing limits. No preprocessing/model was
fitted and no backtest or cloud mutation was performed. Historical identity,
adjustment vintage, knowledge-time and economic-accounting limitations remain.

Private handoff artifacts in the completed run include
`context90_feature_availability.csv`(all 402 names with full/train/test original
and paper-qualified coverage),`context90_coverage_by_year.csv`,
`context90_coverage_by_period.csv`,`context90_independent_verification.json`,
`source_hash_verification.json`and`tests_verification.json`. The master exposes
the saved context summaries and separate lazy feature/daily datasets.

By subsequent user direction, **80% is primary and 90% is the sensitivity check**.
The verified panels retain their original paths, identities and bytes; their
roles are recorded in `configs/context_panel_roles.json`. Read both panels and
saved comparison evidence with `dmv2.context_panel_roles.read_context_panel_roles`.
Historical comparison tables retain their original **90% → 80%** orientation.
Notebook Run All verifies saved evidence without starting a build or fitting a model.
See [the comparison and promotion record](CONTEXT_SENSITIVITY_80.md).

The [verified 80% primary panel](CONTEXT_SENSITIVITY_80.md) has 96.27% finite
context values among paper-qualified rows over 2010–2023 and 92.09% in 2010–2015.
The retained 90% sensitivity has 71.06% and 33.57%, respectively. All shared finite
values, paper15, technical298 and daily artifacts are unchanged. Missingness may
be systematic; promoting 80% does not establish representative or certified PIT data.
