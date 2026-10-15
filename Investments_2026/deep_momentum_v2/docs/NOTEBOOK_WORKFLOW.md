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
signal's January 2020 label into the 2019 training window. Label observation timing must also be enforced when
actual labels are implemented. Proposed calendar-year cutoffs differ from
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
final modelling universe remains open; `engineering_sample=combined` is a
loader demonstration. See [feature contracts](FEATURES.md).

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
| [`src/dmv2/workflow.py`](../src/dmv2/workflow.py) | Notebook settings, orchestration and private audit persistence |
| `tests/` | Adversarial worker tests; structural fixtures only |

The loader was already implemented as code in these source files. Separate
one-off scripts were used to inspect source metadata and validate the first
real sample. Their reusable loading/inspection sequence is now called through
the master notebook, with source and request provenance saved locally.

## Next worker stages

Historical identity adjudication comes first; a producer review flag is not
a confirmed incorrect link or missing-price diagnosis. The prior 755 flagged
2010 links all had resolved provider IDs, predominantly through ticker/company
matching. All original constituents remain in the audit ledger.

Provisional monthly endpoint arithmetic and daily preservation are implemented.
The feature dictionary and workers are implemented in [FEATURES.md](FEATURES.md).
Next are complete monthly event accounting and labels,
class-payoff estimation, forest/XGBoost estimators and portfolio evaluation.
Those modules will be added when implemented and verified; there are no
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
