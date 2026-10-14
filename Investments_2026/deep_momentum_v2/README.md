# Deep Momentum V2

This is the new U.S. equity research workspace in the existing
`Finance-Working-Files` Git repository. The existing `../deep_momentum/` folder
is retained for observation and reference; there is no nested Git repository.

Our source paper is Han and Qin's *Bimodality Everywhere: International Evidence
of Deep Momentum*, March 26, 2026. Its mechanism is **monthly ten-class return
probabilities → historical class payoffs → RET portfolio rankings**. This PDF
uses XGBoost and probability averaging. We intend to study historical S&P 500,
S&P 400 or their union, then added features and forest ensembles. These are
explicit adaptations to its broader CRSP/international analysis, not a promise
to reproduce its published returns. By user decision, V2 does not use market
capitalization: its starting set is the 15 momentum/state inputs, omitting the
size decile, cap-based screening and cap-value weighting. Additional features
remain to be specified.

## Read before research

Start with [`notebooks/00_research_master.ipynb`](notebooks/00_research_master.ipynb)
in VS Code. Select **Deep Momentum V2 (.venv)** and use **Run All** to execute
the implemented bounded foundation stages. It calls the tested source modules;
it does not hide a full scan or model fit. See [the notebook guide](docs/NOTEBOOK_WORKFLOW.md).
The selected study initialization is **2000**, with **rolling 10-year training /
1-year testing**. The nominal first outcome test is 2010, subject to warmup and
coverage; the earlier 2010 loader check was not a research-start decision.

- [Precise paper reference](docs/PAPER_REFERENCE_HAN_QIN_2026.md): equations,
  timing, features, training, portfolio rules, reported evidence and ambiguities.
- [U.S. adaptation and decision ledger](docs/US_REPLICATION_PLAN.md): data
  mapping, missing inputs, causal workflow and decisions still OPEN.
- [Reusable paper skill](.agents/skills/han-qin-deep-momentum/SKILL.md): concise
  instructions for future implementation and review.
- [Draft decision configuration](configs/paper_replication_draft.json): fixed
  paper facts separated from unresolved project conventions; not runnable policy.
- [Research workflow](docs/RESEARCH_PLAN.md) and [data status](docs/DATA_STATUS.md).
- [Data foundations](docs/DATA_FOUNDATIONS.md): loader APIs, historical-universe
  reconstruction, coverage states, source limits and repeatable checks.

The supplied 115-page PDF is identified by its SHA-256 in the reference and
source manifest. Its ignored local copy is
`reference/papers/ssrn-4452964-2026-03-26.pdf`; Git contains original notes and
the skill, not the PDF. A later paper version needs a separate review.

## Local setup and data checks

The configured local workspace already has an isolated Python environment and
authorized source data. For a fresh checkout:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e .
```

Data and source pins are excluded from Git. Obtain the approved source releases
through your own cloud access, copy `configs/data_sources.example.json` to
`configs/data_sources.local.json`, and populate local paths and canonical
manifest hashes. A clone alone does not include data.

From a configured workspace:

```sh
PYTHONPATH=src .venv/bin/python -m dmv2 status
PYTHONPATH=src .venv/bin/python -m dmv2 schema intrinio
PYTHONPATH=src .venv/bin/python -m dmv2 verify-data --hashes
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
PYTHONPATH=src .venv/bin/python -m dmv2 universe --as-of 2010-01-29 --universe combined --mode historical_reconstruction --max-scanned-rows 9000000
```

The first checks input files against manifests. The second shows schemas
without reading financial values. The last verifies local file sizes and
SHA-256 values. Local reads do not require Google login. The package now also
provides bounded price readers and historical constituent reconstruction.
The explicit mode records source limitations; knowledge-time certification is
unavailable from these inputs. The modelling packages are not installed.

## Layout

- `src/dmv2/`: source catalog, bounded loaders, historical universe and quote audits.
- `notebooks/00_research_master.ipynb`: the readable Run All entry point.
- `configs/research_workflow.json`: selected study chronology and inspection controls.
- `tests/`: adversarial source/PIT/coverage tests using structural fixtures only.
- `configs/data_sources.local.json`: ignored private paths and source pins.
- `configs/paper_replication_draft.json`: public paper facts/design placeholders.
- `data/raw/`: ignored independent Intrinio, Siblis membership and unified
  identity releases, with manifests/certificates.
- `data/derived/`: future feature and model-input panels.
- `reference/sadig_us_recovery/`: ignored pinned recovery source, including
  daily five-feature and earlier 740-feature construction code.
- `reference/papers/`: ignored local source PDFs.
- `.agents/skills/han-qin-deep-momentum/`: repository-local paper skill.
- `docs/`: original paper analysis, data status and research design.
- `outputs/`: future reports and experiment artifacts, ignored.

## Current state

MASTER NOTEBOOK AND DATA FOUNDATIONS IMPLEMENTED; REMAINING DESIGN OPEN. Source-file integrity
and table footer counts were verified. All 115 paper pages were reviewed.
The 2010-01-29 combined universe audit retained all 902 constituents; only 130
had a unique identity route without a review flag. Two exact-date quotes loaded
successfully. Unresolved identities remain in the ledger; do not model the
mapped subset as if it represented the intended complete cohort.
No new features, model fits or backtests have been run. Historical capitalization
is absent from the downloaded schemas and deliberately excluded from V2; it
is not a remaining data task or blocker. The 15-input starting set is a declared
adaptation, not the paper's complete 16-input baseline. The primary S&P universe,
usable training coverage, pre-2000 feature warmup, return/execution rules,
inner validation and experiment budget remain OPEN. Selected rolling chronology
is recorded in configs/research_workflow.json, separate from paper facts.
Protected 2024+ financial values remain uninspected.
