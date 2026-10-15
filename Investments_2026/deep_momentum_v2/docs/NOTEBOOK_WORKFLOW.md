# Master notebook and worker files

V2 uses one readable master notebook plus small tested Python modules. The
master owns the sequence and displays intermediate results; worker files own
loading, chronology, identity/coverage rules and future modelling functions.
This keeps one Run All entry point while making the same rules usable from
tests and the existing command line.

The master now also calls the [monthly return stage](MONTHLY_RETURNS.md), keeping
separate `daily_df` and `monthly_df` views. Daily data is preserved for later
daily backtesting even though predictions and targets will be monthly.

## Start in VS Code

Open the V2 folder and `notebooks/00_research_master.ipynb`. Select the project's
`.venv` kernel, displayed as **Deep Momentum V2 (.venv)**, then run cells in
order or use **Run All**. Python and Jupyter VS Code extensions are already
installed on this machine. The notebook bootstrap explicitly imports the
current `src/` code, avoiding the host's hidden editable-path issue.

For a fresh configured checkout:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[notebook]'
.venv/bin/python -m ipykernel install --sys-prefix --name dmv2-v2 --display-name 'Deep Momentum V2 (.venv)'
```

Kernel registration is confined to this environment. The notebook does not
install software, authenticate, download data or launch cloud work. A new
clone needs its own approved local source files and private source pins.
See [VS Code's notebook guide](https://code.visualstudio.com/docs/datascience/jupyter-notebooks)
for kernel selection and Run All, and
[IPython's kernel guide](https://ipython.readthedocs.io/en/stable/install/kernel_install.html)
for environment-specific kernels.

## One configuration, separate date roles

Edit [`configs/research_workflow.json`](../configs/research_workflow.json) and
rerun from the top. It is the source of truth for notebook chronology and
bounded inspection controls. The paper reference/draft retains author facts;
it points to this configuration for selected V2 workflow decisions.

Schema version 4 now also records the user's explicit provisional identity
policy. The master shows `research_identity_df` separately from original
identity evidence; all monthly/feature workers use the same policy. See
[PROVISIONAL_IDENTITY_POLICY.md](PROVISIONAL_IDENTITY_POLICY.md).

The user selected:

- Research initialization: **2010-01-01** (user change, 2026-10-15).
- Training: **rolling ten completed outcome years**.
- Testing: **one completed outcome year**, advancing annually.
- Nominal tests: **2020–2023**, subject to usable coverage and warmup.
- Protected financial holdout: **2024+**, unchanged.

For example, outcome training 2010–2019 precedes test outcomes in 2020;
training 2011–2020 precedes 2021. There are four nominal annual folds, ending
with training 2013–2022 / testing 2023. The code does not put a December 2019
signal's January 2020 label into the 2019 training window. The shared target
and training workers enforce completed-outcome observation cutoffs separately
from feature dates. Proposed calendar-year cutoffs differ from
the paper's January-31 annual fit convention and expanding classifier history.

The engineering snapshot defaults to **2010-01-29** and is independent of
the research start and fold boundaries. Raw files retain their earlier
histories: price footers begin in 1962, and the historical constituent sources
extend before 2000. The start change selects later analysis outcomes; it does
not truncate raw histories or erase the earlier coverage diagnosis.

Pre-initialization feature warmup is per-signal: the bounded feature read starts exactly
503 independent sessions through the first signal, then ends at the last signal.
The paper's monthly endpoints and each technical dependency have their own
complete-window gates. `warmup.raw_start` stays null because a global date is not
an exact rolling window. This does not certify 120 usable training months. The
primary modelling universe is now the user-selected historical S&P 500/400
union. The bounded `engineering_sample=combined` remains a loader demonstration,
not evidence of full-history target coverage. See [feature contracts](FEATURES.md).

## What Run All currently executes

1. Resolve the project/kernel and load validated settings.
2. Display the rolling fold calendar without reading outcomes or fitting models.
3. Check the pinned local source inventory and file sizes.
4. Run one bounded historical membership/identity audit.
   Run the separate three-anchor metadata identity evidence worker, retaining
   the original cohorts and saving its non-promoting review queue. Its dates
   are fixed historical reference anchors at 2000-01-31, 2005-01-31 and
   2009-01-30, independent of the selected research initialization and monthly
   price sample; see [the exact diagnostic contract](IDENTITY_AUDIT.md).
5. Optionally sample a few exact-date quotes, disabled by default.
6. Save complete audit evidence under ignored `outputs/notebook_audits/`.
7. Build a separate past-only feature history and month-end snapshots; withhold
   full-cohort normalization and daily context where unavailable.
8. Save feature history/values/coverage/dictionary/diagnostics independently.
9. Construct the configured January–February 2010 monthly engineering sample
   using preceding-month cohorts and two requested daily price IDs.
10. Save independent daily/monthly Parquet tables and a private return audit.
11. Verify saved full feature panels and the active 80% primary / 90% sensitivity
    roles; expose feature and daily datasets separately without rebuilding them.
12. Display the four-arm model-comparison plan without importing or fitting an estimator.
13. Check shared class/RET guards using already loaded bounded samples.
14. Verify the separately authorized full target build's saved class/RET coverage
    in section 16. Expose lazy full-target and monthly-outcome datasets, keeping
    full target materialization an explicit separate call. Run All does not
    rebuild outcomes or fit a model. See [the full target workflow](FULL_TARGET_PANEL.md).
15. Verify the completed first classifier pilot's saved artifact hashes and
    display six small comparison rows in section 17. This reader does not load
    estimators, prepare full matrices or fit preprocessing/models.

**HISTORICAL VERIFICATION, 2026-10-14 (former 2000 settings):** the installed
master completed Run All in two
fresh kernels, from both the V2 root and the notebook directory, using Python
3.14.6, PyArrow 26.0.0 and ipykernel 7.4.0. Each run retained all **900** source
constituents on 2000-01-31: 302 uniquely mapped without review flags, 529 with
review flags, 32 target collisions and 37 without Intrinio links. These are
one-date identity audit counts, not certified historical training coverage.
The optional one-date price sampling remained disabled. This smoke check
preceded the separately controlled monthly stage. Repeated runs produced identical private
audit evidence; the versioned master stayed without outputs. All 103 worker
tests passed, using structural fixtures rather than financial outcomes.

Run All does not turn the selected research period into an implicit full scan. Every
loader read uses explicit row/file/column-byte limits. The one-date membership
query has a 12-million candidate-row ceiling because the vendor storage is
ordered by security/index history; that ceiling is not permission for a full
panel or backtest. Budget errors do not return partial successful samples.

Re-running with identical settings, source pins, worker code and runtime reuses
identical private evidence. The actual worker files and Python/PyArrow versions
are fingerprinted, so editing a worker produces a separate artifact. A different
result under otherwise identical pins fails instead of overwriting
it. Disabling inspection resets inspection state, so an old result is not
silently reused. Changing worker code reloads in the interactive kernel;
restart the kernel after changing class layouts or switching project folders.

Versioned notebooks contain no execution outputs. Use the notebook's Clear
All Outputs before committing; the full private audit is already retained
separately. The optional quote check summarizes coverage rather than embedding
raw market records in the public notebook.

## Coded files and inspection scripts

| File | Role |
|---|---|
| [`notebooks/00_research_master.ipynb`](../notebooks/00_research_master.ipynb) | Readable notebook entry point |
| [`src/dmv2/catalog.py`](../src/dmv2/catalog.py) | Manifest pins, source inventory, schema/integrity checks |
| [`src/dmv2/loading.py`](../src/dmv2/loading.py) | Bounded date/ID price and metadata loading |
| [`src/dmv2/universe.py`](../src/dmv2/universe.py) | Dated cohort, identity states and retained quote coverage |
| [`src/dmv2/identity_audit.py`](../src/dmv2/identity_audit.py) | Non-promoting historical identity evidence, metadata-only loader and private review queue |
| [`src/dmv2/data.py`](../src/dmv2/data.py) | Source adapters and source-calendar helpers |
| [`src/dmv2/exchange_sessions.py`](../src/dmv2/exchange_sessions.py) | Independent XNYS sessions and vendor-calendar reconciliation |
| [`src/dmv2/monthly.py`](../src/dmv2/monthly.py) | Full-cohort monthly ratios, endpoint/path and volume audits |
| [`src/dmv2/monthly_loading.py`](../src/dmv2/monthly_loading.py) | Capped cumulative-budget sample with unchanged daily prices |
| [`src/dmv2/chronology.py`](../src/dmv2/chronology.py) | Pure rolling outcome-window calendar |
| [`src/dmv2/targets.py`](../src/dmv2/targets.py), [`src/dmv2/ret.py`](../src/dmv2/ret.py) | Shared tie-aware classes and strictly trailing realized-class RET payoffs |
| [`src/dmv2/classification.py`](../src/dmv2/classification.py) | Shared class-label training adapter with completed-outcome guards |
| [`src/dmv2/full_target_loading.py`](../src/dmv2/full_target_loading.py) | Cache-only annual target planning and exact monthly outcome construction |
| [`src/dmv2/target_coverage.py`](../src/dmv2/target_coverage.py) | Original-denominator class/missingness reports and full RET calendar grids |
| [`src/dmv2/full_target_panel.py`](../src/dmv2/full_target_panel.py) | Explicit full-target build, immutable checkpoints and saved report/target readers |
| [`src/dmv2/workflow.py`](../src/dmv2/workflow.py) | Notebook settings, orchestration and private audit persistence |
| `tests/` | Adversarial worker tests; structural fixtures only |

The loader was already implemented as code in these source files. Separate
one-off scripts were used to inspect source metadata and validate the first
real sample. Their reusable loading/inspection sequence is now called through
the master notebook, with source and request provenance saved locally.

## Next worker stages

Historical identity adjudication remains separate from provisional admission;
a producer review flag is not a confirmed incorrect link or missing-price diagnosis. The prior 755 flagged
2010 links all had resolved provider IDs, predominantly through ticker/company
matching. All original constituents remain in the audit ledger.

Provisional monthly endpoint arithmetic and daily preservation are implemented.
The feature dictionary and workers are implemented in [FEATURES.md](FEATURES.md).
Shared monthly class labels and class-payoff workers are now implemented; the
full-history empirical coverage audit is complete. Estimator adapters and the
first measured development pilot are implemented in
[CLASSIFIER_PILOT.md](CLASSIFIER_PILOT.md). Complete monthly event accounting and
portfolio evaluation retain their remaining methodology decisions. There are no
placeholder model calls that pretend to execute the research. Source
knowledge-time and terminal-event limitations remain documented in
[`DATA_FOUNDATIONS.md`](DATA_FOUNDATIONS.md).

## Feature-stage verification, 2026-10-14

The updated installed master completed all 13 code cells from both the V2 root
and notebooks folder in fresh kernels. All 265 tests passed. The separate
503-session warmup read produced 1,046 daily records; features retained 1,799
original cohort rows and all 298 technical values on each of four requested
stock/signal rows. Full-cohort paper/context outputs remained withheld rather
than using two-stock sample statistics. Repeated runs reused identical private
evidence and the versioned notebook remained without outputs. See
[FEATURES.md](FEATURES.md) for exact policies and source certification limits.

## Identity-evidence-stage verification, 2026-10-14

The installed master completed all **15 code cells** in a fresh V2-root kernel
with the separate three-anchor identity worker. All **298 tests** passed,
including 21 new identity/clock/contamination/persistence tests. Its metadata
audit retained all 2,701 original constituent/date rows and promoted no mappings.
The existing bounded two-ID feature/monthly samples completed unchanged in
scope; the identity worker itself read no financial values or outcomes. The
versioned notebook remained output-free. See [IDENTITY_AUDIT.md](IDENTITY_AUDIT.md)
for evidence classes, exact populations, budgets and unresolved source evidence.

## Provisional-policy-stage verification, 2026-10-14

Under the user's explicit identity waiver, the installed master completed all
**16 code cells** and **330 tests** passed. The sample now exercises one clean
route and one review-flagged route without clearing its flag. Both produce 298
technical values at both signals and four accepted monthly endpoint ratios;
all 1,799 original rows and the independent daily tables remain intact. The
separate admission display reports 831/860/878 eligible routes at the three
anchor dates. No predictor contains admission metadata, no fit was performed,
and no cross-sectional sample normalization or historical certification is
inferred. See [the recorded provisional decision](PROVISIONAL_IDENTITY_POLICY.md).

## Full feature panel verification, 2026-10-15

The [full 2010–2023 feature panel](FULL_FEATURE_PANEL.md) is now built and audited under the provisional identity policy. It retains 152,307 original rows and 168 monthly signals. Daily observations feed technical/context calculations and are also preserved independently for later backtesting, including complete annual warmup and signal anchors. All 715 eligible names remain in the dictionary; missing context is reported rather than filled or silently excluded. The master verifies saved coverage and exposes separate lazy feature/daily datasets. Notebook Run All does not relaunch the full build, fit imputation/models or backtest.

## Active 80% primary context policy; 90% sensitivity

`features.context_policy=available_cohort_80pct` agrees with the active normalization
and full-build configurations. The master verifies explicit immutable panel pins
and exposes the 80% run as `full_feature_run`/`full_feature_dataset`, the 90% run as
`context_sensitivity_run`/`context_sensitivity_dataset`, and raw daily data as the
separate lazy `full_daily_dataset`. Historical comparison columns remain 90% →80%.
It displays the active policy and saved full-panel context coverage. A two-ID request still cannot qualify a historical cross-section:
all admitted IDs must be requested before 80% observed coverage is considered.
The paper15 normalization and daily/outcome separation are unchanged. Read
[the context contract](CONTEXT_AVAILABILITY.md).

By subsequent user direction, **80% is primary and 90% is the sensitivity check**.
The verified panels retain their original paths, identities and bytes; their
roles are recorded in `configs/context_panel_roles.json`. Read both panels and
saved comparison evidence with `dmv2.context_panel_roles.read_context_panel_roles`.
Historical comparison tables retain their original **90% → 80%** orientation.
Notebook Run All verifies saved evidence without starting a build or fitting a model.
See [the comparison and promotion record](CONTEXT_SENSITIVITY_80.md).

## Model-comparison planning cell

Section 14 reads `configs/model_comparison.proposed.json` through
`dmv2.model_design`, displays four proposed estimator/input arms, and generates
48 monthly calendar specifications for the 2020–2023 nominal tests. This stage
reads no market observations and imports no estimator. It clears planning state
on rerun and displays unresolved decisions; Run All does not start fitting.
See [MODEL_COMPARISON.md](MODEL_COMPARISON.md) before implementing estimator
adapters. The combined primary universe is selected, and the shared target
contract fixes exact ties, missing-outcome ranking and strict RET grid handling.
That draft is historical planning evidence. The separately authorized first
pilot's validation/settings and measured fit budget are fixed in
`configs/classifier_pilot.json`; section 17 displays its saved results.

## Shared class labels and RET

Section 15 loads `configs/target_ret_protocol.json`, defines the shared
`prepare_classification_inputs` helper and reuses existing bounded feature and
monthly samples through `target_workflow.run_shared_target_sample`. It adds no
source-price request. The partial two-ID sample correctly withholds class ranks
and 120-month RET; inspect reasons rather than treating it as population coverage.
Both models must use the class-label helper; the older continuous-return
preparation helper serves a different target interface. Missing class rows stay
in prediction/coverage ledgers. See [TARGET_CLASSES_RET.md](TARGET_CLASSES_RET.md).

## Full target audit and saved-evidence cell

The separately authorized full 2010–2023 build is complete from verified saved
caches. Section 16 uses
`dmv2.full_target_panel.read_full_target_run(catalog)` to verify immutable saved
state and display small aggregate class/RET reports. It keeps
`full_feature_dataset` and `full_daily_dataset` separate from lazy
`full_target_dataset` and `full_monthly_outcome_dataset` views. Full labels are
loaded only by the explicit `read_full_targets(run)` call. Neither reader launches
a target build or estimator. The audit has verified 140,485 labels among
140,777 signal-time prediction-eligible rows, retaining all 152,307 original rows.
All 168 months populate all ten classes; 292 eligible missing labels remain
explicit. All 48 nominal test signals have strict 120-month RET payoff vectors;
the first is 2019-12-31 for January 2020. Earlier 120 signals remain explicitly
withheld for insufficient history. The saved audit verifies 475 recorded input
files unchanged. Read [FULL_TARGET_PANEL.md](FULL_TARGET_PANEL.md) for exact
coverage, artifact pins and the remaining identity/knowledge-time/economic limits.


The installed master completed all **23 original code cells** in a fresh kernel
after the full target audit. Three private verification cells blocked implicit
full builds/preprocessing fits, asserted exact unchanged bounded daily/monthly/
feature frames, and reloaded the full shared class state. All **620 tests** and
the independent full class/RET verifier passed. Private receipts and the executed
notebook are under ignored `outputs/full_target_panel/verification/`; the
versioned notebook remains output-free. No classifier or backtest ran.

## First four-arm classifier pilot

Section 17 uses `dmv2.classifier_pilot.read_classifier_pilot_run(catalog)` and
displays RF and XGBoost on Paper15 and the expanded bank, alongside uniform and
training-prior references. The explicit pilot trains on 2010–2017 completed
outcomes and validates on 2018–2019. Its input preparation and four fits occur
only through the separate CLI action, never Run All. The reader verifies
current recipe/source/artifact hashes and reads only the flat comparison table;
it does not load serialized estimators or prepared matrices. A missing run
prints a message; an invalid saved run fails after clearing old notebook state.

`classifier_pilot_run` and `classifier_pilot_comparison_df` hold the verified
result. The existing bounded `daily_df`, `monthly_df`, `features_df` and
`feature_history_df` stay independent and unchanged. RET, 2020–2023 evaluation
and backtesting remain outside this development pilot. See
[the frozen protocol and complete results](CLASSIFIER_PILOT.md).

The installed master completed all **24 original code cells** in a fresh
kernel in **32.99 seconds** after the pilot. Added verification cells blocked
full pilot loading/fitting, imputer fitting, full target rebuilding and saved
matrix/model loads; none was called. All six comparison rows were displayed
and exact before/after checks preserved the separate daily/monthly/feature
samples. All **690 tests** and the independent full pilot verifier passed; the
four reloaded models reproduced every probability exactly. Private receipts and
the executed notebook live under ignored `outputs/model_pilot/verification/`;
the versioned notebook remains output-free.
