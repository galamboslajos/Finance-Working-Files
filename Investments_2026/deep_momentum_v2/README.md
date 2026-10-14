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
to reproduce its published returns.

## Read before research

- [Precise paper reference](docs/PAPER_REFERENCE_HAN_QIN_2026.md): equations,
  timing, features, training, portfolio rules, reported evidence and ambiguities.
- [U.S. adaptation and decision ledger](docs/US_REPLICATION_PLAN.md): data
  mapping, missing inputs, causal workflow and decisions still OPEN.
- [Reusable paper skill](.agents/skills/han-qin-deep-momentum/SKILL.md): concise
  instructions for future implementation and review.
- [Draft decision configuration](configs/paper_replication_draft.json): fixed
  paper facts separated from unresolved project conventions; not runnable policy.
- [Research workflow](docs/RESEARCH_PLAN.md) and [data status](docs/DATA_STATUS.md).

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
.venv/bin/python -m dmv2 status
.venv/bin/python -m dmv2 schema intrinio
.venv/bin/python -m dmv2 verify-data --hashes
```

The first checks input files against manifests. The second shows schemas
without reading financial values. The last verifies local file sizes and
SHA-256 values. Local reads do not require Google login. The existing package
is a data catalog/verification CLI; paper-model software is not installed by
this documentation change.

## Layout

- `src/dmv2/`: active V2 code, currently the data catalog and verification CLI.
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

SETUP AND PAPER REVIEW COMPLETE; EXPERIMENT DESIGN OPEN. Source-file integrity
and table footer counts were verified. All 115 paper pages were reviewed.
No new features, model fits or backtests have been run. Historical capitalization
is absent from the downloaded schemas, so the paper's complete 16-feature
baseline is not yet supported. The primary S&P universe, usable history,
return/execution rules, temporal validation and experiment budget remain OPEN.
Protected 2024+ financial values remain uninspected.
