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

The user selected:

- Research initialization: **2000-01-01**.
- Training: **rolling ten completed outcome years**.
- Testing: **one completed outcome year**, advancing annually.
- Nominal tests: **2010–2023**, subject to usable coverage and warmup.
- Protected financial holdout: **2024+**, unchanged.

For example, outcome training 2000–2009 precedes test outcomes in 2010;
training 2001–2010 precedes 2011. There are fourteen nominal annual folds.
The code does not put a December 2009 signal's January 2010 label into the
2009 training window. Label observation timing must also be enforced when
actual labels are implemented. Proposed calendar-year cutoffs differ from
the paper's January-31 annual fit convention and expanding classifier history.

The engineering snapshot defaults to **2000-01-31** and is independent of
the research start and fold boundaries. The old 2010 snapshot was only a
loader test. Raw files retain their earlier histories: price footers begin
1962, and the historical constituent sources extend before 2000.

Pre-2000 feature warmup remains to be specified. Early 2000 momentum values
may need earlier monthly prices; initialization does not truncate the raw
files or certify 120 complete training months. `warmup.raw_start` stays null
until the return/feature contract is defined. The final modelling universe
also remains open; `engineering_sample=combined` is a loader demonstration.

## What Run All currently executes

1. Resolve the project/kernel and load validated settings.
2. Display the rolling fold calendar without reading outcomes or fitting models.
3. Check the pinned local source inventory and file sizes.
4. Run one bounded historical membership/identity audit.
5. Optionally sample a few exact-date quotes, disabled by default.
6. Save complete audit evidence under ignored `outputs/notebook_audits/`.
7. Construct the configured January–February 2000 monthly engineering sample
   using preceding-month cohorts and two requested daily price IDs.
8. Save independent daily/monthly Parquet tables and a private return audit.

**VERIFIED LOCAL, 2026-10-14:** the installed master completed Run All in two
fresh kernels, from both the V2 root and the notebook directory, using Python
3.14.6, PyArrow 26.0.0 and ipykernel 7.4.0. Each run retained all **900** source
constituents on 2000-01-31: 302 uniquely mapped without review flags, 529 with
review flags, 32 target collisions and 37 without Intrinio links. These are
one-date identity audit counts, not certified historical training coverage.
The optional one-date price sampling remained disabled. This smoke check
preceded the separately controlled monthly stage. Repeated runs produced identical private
audit evidence; the versioned master stayed without outputs. All 103 worker
tests passed, using structural fixtures rather than financial outcomes.

It does not turn the entire 2000–2023 period into an implicit scan. Every
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
Next are complete monthly event accounting, a feature dictionary and labels,
class-payoff estimation, forest/XGBoost estimators and portfolio evaluation.
Those modules will be added when implemented and verified; there are no
placeholder model calls that pretend to execute the research. Source
knowledge-time and terminal-event limitations remain documented in
[`DATA_FOUNDATIONS.md`](DATA_FOUNDATIONS.md).
