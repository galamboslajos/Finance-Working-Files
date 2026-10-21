# Current Deep Momentum V2 handoff — 21 October 2026

The corrected-input five-arm refit and daily replay are complete. Start with the
[latest handoff](docs/CORRECTED_REFIT_AND_RET_BACKTEST_20261021.md),
[full portfolio comparison](docs/CORRECTED_BACKTEST_COMPARISON_20261021.md),
[model diagnostics](docs/CORRECTED_MODEL_DIAGNOSTICS_20261021.md) and
[executed source / offline test guide](scripts/corrected_refit_20261021/README.md).

Twenty annual estimator fits and eight shared training-only preprocessing states
cover 2020–2023 using the preceding ten completed outcome years, with monthly
predictions. Four baseline RF/XGBoost × Paper15/expanded arms are accompanied by
RF expanded using the previously selected annual CV settings. Those parameters
were reused; this is not a new tuning search. The historical S&P 500/400 union,
80% primary context, 715 expanded predictors, endpoint target and trading rules
remain fixed. Daily feature history and daily execution inputs remain separate
from monthly features/targets.

For **2 January 2020–29 December 2023: 48 months and 1,006 sessions**, RET with
10% per-asset stops and 10bp per one-way traded notional:

| Model | Gross four-year return | Net four-year return | Net CAGR | Net Sharpe | Net max DD |
|---|---:|---:|---:|---:|---:|
| RF · Paper15 | Withheld | Withheld | — | — | — |
| XGBoost · Paper15 | +39.41% | +27.16% | +6.20% | 0.39 | −25.56% |
| **RF · expanded baseline** | **+0.18%** | **−8.35%** | **−2.16%** | **−0.02** | **−31.01%** |
| XGBoost · expanded | Withheld | Withheld | — | — | — |
| RF · expanded, prior CV settings | +26.69% | +15.87% | +3.76% | 0.29 | −22.71% |

RET ranks the entire class-probability vector against the shared causal 120-month
class payoff states. Class1/10 and class2/9 remain separate alternatives; their
results, unstopped comparators, all four cost assumptions and annual slices are
in the full comparison. These are continuous four-year returns. Sharpe uses
daily returns and a zero cash rate; CAGR uses252 sessions/year. Borrow/funding
costs remain zero and daily-OHLC stop fills are idealized.

Independent checks account for all120 paths: **56 complete /64 withheld**.
Actual first failures are44 missing-quote paths,16 source-quality-flag paths and
four selected-provider/held-source collision paths. Every selected exposure and
failed path remains present; full-period and annual-prefix performance is null.
The [exact first-failure note](scripts/corrected_refit_20261021/ACTUAL_FAILURE_DIAGNOSTICS_20261021.md)
and [maximum-horizon gap list](scripts/corrected_refit_20261021/SELECTED_HORIZON_GAPS_20261021.md)
separate the causes. A complete subset cannot establish a complete model ranking.

The [BK/XOM shadow input rebuild](docs/BK_XOM_SHADOW_REBUILD_20261021.md) was
reused byte-for-byte, including48 corrected causal RET states. RCM/DISCA learning
candidates failed frozen volume controls and admitted no learning bars. Their
previously checked execution-only quotes remain separately pinned. The
[endpoint policy](scripts/corrected_refit_20261021/MONTHLY_ENDPOINT_LABEL_POLICY.md)
keeps29 registered-event overlap targets missing; no cash/stock payout is invented
as a predecessor close. [Input decisions](scripts/corrected_refit_20261021/RECOVERY_INPUT_DECISIONS.md)
and the [data flow](scripts/corrected_refit_20261021/DATA_FLOW_20261021.md) explain
the boundary between monthly labels, RET estimates and next-open/stop-managed
realized payoffs. Identity, source vintage, terminal returns, knowledge time and
survivorship remain conditional. This already reviewed period is development
evidence; no2024+ financial evaluation is included.

The final independent backtest checker is verifierV5. A physical Arrow-null
successor identifier is kept as None by reading only four registered diagnostic
projections directly as Arrow records after the same fingerprint check. Strict
NaN rejection and all selection/accounting equations remain unchanged. Earlier
failed checker attempts and all prior runs are preserved. The
[source entrypoint record](scripts/corrected_refit_20261021/RUNTIME_ENTRYPOINTS_20261021.md)
distinguishes the versions and targeted regression tests.

Next, establish dated quote/action entitlements and the selected/held identity
handover for the withheld paths, then replay the same saved selections in a new
execution-data version. Audit any learning/label/RET impact before changing
labels; another refit follows only if those inputs actually change.

Git contains original source, protocols, tests and aggregate research summaries.
Raw/provider data, credentials, source evidence bundles, models, matrices,
probabilities, numerical receipts and daily ledgers remain private. The verified
local package is `outputs/corrected_refit/2020-2023-20261021-v1/`; its installed
catalog/relocation map locate every artifact. A fresh clone has no private run
or automatic model/backtest execution. The notebook's existing section24 remains
a reader for its older frozen baseline; publication does not promote a new
notebook or active-run pointer. The historical README and source-quality notes
retain their pinned bytes. Use this file as the current entry point.

The older handoffs below are historical input/run versions. Their class-basket
results must not be confused with the RET comparison above or treated as a
paired causal estimate of the latest repairs.

---

# Current Deep Momentum V2 handoff — 20 October 2026

The RF-expanded temporal tuning and its separate daily backtest are complete
and independently verified. Start with [the tuning completion and prediction
diagnostics](docs/RF_EXPANDED_TUNING_RESULTS_20261020.md) and
[the verified daily results](docs/RF_EXPANDED_TUNED_BACKTEST_RESULTS_20261020.md).
The dated source-corrected baseline handoff below remains reference evidence.

The search completed 96 inner fits and four selected annual ten-year refits,
using training-only preprocessing and three chronological validation years per
outer year. Annual settings were selected only by 36-month mean log loss.
All original cohorts, targets, source corrections, monthly prediction timing and
basket/stop/accounting rules remain unchanged. The first memory-stopped attempt
and all pause/resume checkpoints are preserved; no automatic retry is added.

For 2 January 2020–29 December 2023 (48 months, 1,006 sessions), class 1/10,
10% per-asset stops and 10 bp one-way costs:

| Model | Total net return | CAGR | Annualized volatility | Sharpe | Maximum drawdown |
| --- | ---: | ---: | ---: | ---: | ---: |
| RF / Paper15 | Withheld | — | — | — | — |
| XGBoost / Paper15 | +23.45% | 5.42% | 21.86% | 0.35 | −31.18% |
| RF / expanded original | +18.82% | 4.41% | 20.82% | 0.31 | −35.14% |
| XGBoost / expanded | +3.41% | 0.84% | 18.64% | 0.14 | −31.82% |
| **RF / expanded tuned** | **+42.78%** | **9.33%** | **20.69%** | **0.53** | **−35.14%** |

RF expanded with stops remains the primary focus. Its tuned return is +55.11%
gross. The unchanged 2020 forecasts explain the unchanged maximum drawdown.
Aggregate monthly prediction log loss is 2.268171 versus 2.268053 for the
original RF; accuracy is 13.11% versus 13.36%. Better portfolio performance
is not evidence of better probability forecasts or a new untouched test.

The additive replay completed sixteen scenarios: eight class 1/10 complete
proxy paths and eight class 2/9 paths with full-period metrics withheld for
missing held observations. Exposure remains present. The original comparison
still has 24 complete / 40 withheld scenarios, so a complete four-model economic
ranking is unavailable. Borrow/funding remain zero; intraday stop fills are
idealized; source identity, knowledge time and survivorship remain provisional.
No protected 2024+ financial values were read.

The [V2 verification correction](docs/RF_TUNED_BACKTEST_VERIFICATION_CORRECTION_20261020.md)
addresses a physical Arrow string-width re-encoding in selection files. Exact
values, row order, nulls and accounting were independently checked. The first
failed verifier, both raw hashes, original replay and all old runs stay preserved;
no fitting or replay was repeated to resolve that assertion.

Git contains source, protocols, tests and aggregate summaries. Datasets,
credentials, models, matrices, ledgers and verification receipts stay private.
The historical README and repair-phase notes remain hash-pinned evidence;
this file is the current entry point. Notebook section24 displays the original
source-corrected baseline, while the new tuned results are documented in the
links above. Publication starts no financial computation.

---

# Source-corrected baseline handoff — 16 October 2026

The corrected 2020–2023 walk-forward is complete and independently checked.
This is the starting point for the next research round. The original runs are
preserved as dated references. No model tuning or new experiment is part of
this publication.

## What was corrected and rerun

The new inputs repair the historical El Paso/Empire issuer route and fifteen
corrupt OPCH OHLC observations in 2022. OPCH volume remains missing on those
fifteen observations. Annual input views for 2010–2013 and 2022–2023 were
rebuilt; unaffected annual inputs were reused with explicit byte-level lineage.
The source screen reviewed 3,792,168 cached observations and produced 3,490
review clusters. Those clusters are not an error count or a guarantee that all
source problems have been discovered.

All sixteen annual RF/XGBoost fits and eight train-only preprocessing states
completed under the original frozen recipe. Each arm retains 42,145 forecasts
and 42,080 observed labels across 43,655 original stock-month rows. Missing
future labels do not remove predictions. Saved probabilities reproduced exactly
in the independent checks. Input reuse, model reconstruction and daily cash/units
checks passed; the formula audit does not independently certify every expanded
feature. Source identity, knowledge time and survivorship remain provisional.

## Results to discuss

These are continuous **four-year total returns**, 2 January 2020–29 December
2023: 48 months and 1,006 XNYS sessions. The table uses class 1/10 baskets and
10 bps of trading costs per one-way traded notional. Net here excludes borrow
and funding costs, which remain zero under the frozen assumptions.

| Model | Total return, unstopped | Total return, 10% asset stop | Stop CAGR | Stop maximum drawdown | Stop annualized Sharpe |
| --- | ---: | ---: | ---: | ---: | ---: |
| RF / Paper15 | Withheld | Withheld | — | — | — |
| XGBoost / Paper15 | +56.16% | +23.45% | +5.42% | −31.18% | 0.35 |
| **RF / expanded 715** | **+47.79%** | **+18.82%** | **+4.41%** | **−35.14%** | **0.31** |
| XGBoost / expanded 715 | +23.55% | +3.41% | +0.84% | −31.82% | 0.14 |

RF expanded with stops is the user-selected focus, not a variant selected after
these returns were observed. It returns +29.18% gross and +18.82% after the
stated trading costs, with 20.82% annualized net volatility. Its net annual
slices are +8.75% in 2020, −5.97% in 2021, −1.87% in 2022 and +18.42% in 2023.
The maximum drawdown runs from 8 June to 6 November 2020; long, short and cost
contributions during that interval are −25.28, −9.28 and −0.58 percentage points
of peak NAV. There are 513 stop exits: 261 long and 252 short, including 86
gap-open fills. Daily OHLC stop fills remain idealized execution assumptions.

The obsolete-source RF expanded stop reference was +7.31% net for the same four
years. The new comparison includes corrected training inputs, refitted models,
changed forecasts, changed holdings and daily execution inputs. It is not a
fill-only adjustment: 34.31% of RF expanded hard-class forecasts changed, and
67.29% of old long selections / 66.67% of old short selections were retained.

## Coverage that still needs repair

All 64 predefined scenarios were processed: **24 complete proxy paths and 40
withheld paths**. Complete paths are class 1/10 for XGBoost/Paper15, RF/expanded
and XGBoost/expanded, each with two stop choices and four cost settings.
RF/Paper15 and every class 2/9 full-period comparison remain withheld. Missing
held marks retain exposure and suppress full-period metrics; no selected stock
was dropped to make a path complete. These are not 40 distinct issuer problems,
and the available subset cannot establish a complete four-model ranking.

| Model / basket | First actual blocking mark | Membership label / native source ID |
| --- | --- | --- |
| RF / Paper15 / 1–10 | 2020-10-02, source quality flag | NBL / sec_y3QbRX |
| RF / Paper15 / 2–9 | 2022-06-27, missing quote | ELV / sec_XO8Y0b |
| XGBoost / Paper15 / 2–9 | 2021-01-19, missing quote | CXO / sec_qge3qz |
| RF / expanded / 2–9 | 2020-04-03, missing quote | RTN / sec_X6drPy |
| XGBoost / expanded / 2–9 | 2020-04-03, missing quote | RTN / sec_X6drPy |

Later maximum-holding-horizon checks also flag WBD (sec_Xnq2jn, 4–8 April
2022), VYX (sec_z5lK44, 16 October 2023) and FBIN (sec_z5xK8O, 14 December
2022). These are source membership labels, not new identity certifications.
Before a repair, establish dated issuer identity and actual quote or legal
conversion/entitlement evidence. Preserve selection rules, earlier evidence and
all held exposures. KEM/KMPR identity ambiguity remains unresolved; both known
wrong BBBY routes stay quarantined under the existing provisional policy.

## Frozen research rules

- Historical S&P 500/400 union; 80% context rule primary; Paper15 and expanded715.
- Train on 2010–2019 for 2020, then advance annually to 2013–2022 for 2023.
  Monthly predictions; train-only medians and missing indicators; no target imputation.
- RF 500 trees and XGBoost 300 rounds, seed 1729, original parameters;
  no search, early stopping, calibration or performance-driven refit.
- Separate class 1/10 and class 2/9 mode-class baskets, ranked by own-class
  probability with stable source-ID ties; N=min(10, long candidates, short candidates).
  Equal names on both sides, +50%/−50% initial capital, equal within-side weights.
- Next-session open execution, monthly rebalance; unstopped and 10% per-asset
  stop alternatives; stop proceeds remain cash until the next monthly decision.
- Costs 0/5/10/25 bps per one-way traded notional; continuous NAV across years;
  original action terms, dated provisional PENN handover and DF/RDC trading
  restrictions. No synthetic quotes, new exclusions or annual NAV reset.
- Separate daily feature history and execution data; RET is not evaluated;
  no protected 2024+ financial observations are analyzed.

## Code and evidence entry points

Start with [the corrected protocol and completed evidence summary](docs/SOURCE_CORRECTED_WALKFORWARD_20261016.md)
and master notebook section24 in [00_research_master.ipynb](notebooks/00_research_master.ipynb).
The notebook reads verified saved summaries only and has no committed outputs.
Earlier notebook sections preserve earlier source versions.

Explicit workers and CLIs are in `src/dmv2/source_corrected_*.py` and
`scripts/*source_corrected*.py`. They separate input preparation, annual fitting,
independent verification, evidence rebinding, coverage preflight, daily replay
and passive reporting. Use `scripts/compare_source_corrected_forecasts_v2.py`
for the completed forecast/selection diagnostic; its V1 predecessor is retained
as documented historical code. CLI `--help` is safe; executing a fit or replay
requires the frozen execution plan and separate fresh output directories.

A Git clone includes code, original methodology and aggregate results only.
It does **not** include the approved datasets, fitted models, forecast/holding
ledgers, quote evidence or verification receipts. Cloud access is separate from
GitHub access. Restore approved private artifacts to their recorded relative
paths in a separately configured evidence workspace; do not weaken checks to
make an incomplete checkout pass. The corrected replay requires the reviewed
412-artifact repair handoff, fourteen annual source references, original
completed reference run and separately pinned execution/action evidence.

The historical repair-phase files `README.md`, `SOURCE_QUALITY_STATUS.md` and
`docs/SOURCE_QUALITY_REPAIR_20261016.md` are deliberately preserved byte for
byte: the private repair handoff hashes them. Their pending-refit language
describes the earlier phase, before the completed replay documented here.
Keep immutable evidence checkouts distinct from subsequent source development.

Private completed run (relative to the V2 workspace):
`outputs/walkforward_source_corrected/2020-2023-20261016-v1`, build identity
`sha256:559ba2f7fbe187f192b40af2ab8fbe0a4fd70a1b843e2cbd0503f7e35729fcae`.
Its `completion.json` binds the final artifacts. Comparative summaries, all64
scenarios, all256 annual rows, stock contributions and stop exits are under
`reports/comparison_20261016`; forecast differences and selections are under
`reports/forecast_selection_changes_v2_20261016`. Full model and accounting
receipts remain in `verification`. Original reference runs stay unchanged.

For code-only checks after installing the pinned dependencies, run:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[notebook,models]'
PYTHONPATH=src .venv/bin/python -B -m unittest discover -s tests
```

Tests use synthetic fixtures and do not start the research fits or backtest.
Private-data actual audits are separate explicit verification commands.
