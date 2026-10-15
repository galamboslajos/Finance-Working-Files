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
size decile, cap-based screening and cap-value weighting. The bounded feature bank now registers the 15 paper inputs plus daily technical
and historical context definitions; see [feature contracts](docs/FEATURES.md).

## Read before research

Start with [`notebooks/00_research_master.ipynb`](notebooks/00_research_master.ipynb)
in VS Code. Select **Deep Momentum V2 (.venv)** and use **Run All** to execute
the implemented bounded foundation stages. It calls the tested source modules;
it does not hide a full scan or model fit. See [the notebook guide](docs/NOTEBOOK_WORKFLOW.md).
The selected study initialization is **2010**, with **rolling 10-year training /
1-year testing**. The nominal annual outcome tests are **2020–2023**, subject to
warmup and coverage. Starting later does not remove the earlier raw history.
Pre-2010 observations remain available for exact feature warmup.

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
- [Monthly returns and daily preservation](docs/MONTHLY_RETURNS.md): exact
  endpoint ratios, prior-month cohorts, missing outcomes and event/PIT limits.

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
- `src/dmv2/monthly.py` and `monthly_loading.py`: monthly return/coverage worker
  and bounded source adapter, retaining daily data for later backtesting.
- `src/dmv2/daily_features.py`, `paper_features.py`, `context_features.py`: daily-to-month-end technical/context features and exact paper monthly inputs.
- `src/dmv2/features.py`, `feature_loading.py`, `feature_workflow.py`: predictor/target separation, bounded past-only reads and private feature evidence.
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

MASTER NOTEBOOK, DATA FOUNDATIONS, MONTHLY RETURNS AND BOUNDED FEATURE WORKERS IMPLEMENTED; REMAINING EXPERIMENT DESIGN OPEN. Source-file integrity
and table footer counts were verified. All 115 paper pages were reviewed.
The initial strict-policy 2010-01-29 combined audit retained all 902 constituents;
130 had unique routes without review flags, 755 required review, ten had target
collisions and seven were unlinked. Two exact-date quotes loaded successfully.
Under the current provisional policy, unique resolved review-flagged routes may
enter research with their flags retained; missing/ambiguous routes and collisions
remain blocked. Original cohort denominators remain intact.
The separate [2010–2023 endpoint audit](docs/INITIALIZATION_2010.md) measured
complete 15-input paper coverage on 140,777 of 152,307 original stock-months
(92.43%). Full-bank coverage is now measured in [the full-panel audit](docs/FULL_FEATURE_PANEL.md); model fits and backtests remain outstanding. Historical capitalization
is absent from the downloaded schemas and deliberately excluded from V2; it
is not a remaining data task or blocker. The 15-input starting set is a declared
adaptation, not the paper's complete 16-input baseline. The primary S&P universe,
economic return/execution rules, inner validation
and experiment budget remain OPEN. Exact per-signal warmup is implemented; the
15-input endpoint audit does not establish complete daily technical/context history. Selected rolling chronology
is recorded in configs/research_workflow.json, separate from paper facts.
Protected 2024+ financial values remain uninspected.

The master now constructs separate `daily_df` and `monthly_df` views for a
bounded January–February 2010 sample. Monthly cohorts use the preceding month
end, retaining every historical member and missing/unrequested outcome. Numeric
provider-adjusted ratios are provisional research returns; knowledge-time,
economic-event completeness and daily backtest readiness are not certified.

The [normalization/missingness protocol](docs/NORMALIZATION_PROTOCOL.md) now
freezes the active common-five paper population and training-only median/missing
indicator rules under the accepted provisional identity policy. Earlier proposals
remain archival. The master verifies and displays the explicit
[2010-start coverage audit](docs/INITIALIZATION_2010.md), with no automatic
full-history scan or model fit.

The [historical identity evidence audit](docs/IDENTITY_AUDIT.md) is now called
from the master as historical reference evidence. It retains all 2,701
constituent/date rows at three fixed 2000–2009 anchors, distinguishes source aliases from declared lookup evidence and saves
a private review queue. No mappings were promoted; dated provider identity and
publication evidence remain missing. A [draft evidence request](docs/IDENTITY_EVIDENCE_REQUEST.md)
is available to forward.

By user direction, [provisional identity admission](docs/PROVISIONAL_IDENTITY_POLICY.md)
is active in the master configuration. Unique resolved review-flagged links may
enter research calculations with their original flags and uncertainty intact;
missing/ambiguous routes and same-date collisions remain blocked and visible.
Historical identity/PIT certification no longer stops all provisional development.

The explicitly authorized [full 2010–2023 feature build](docs/FULL_FEATURE_PANEL.md)
uses annual checkpoints and separate raw daily data. The master verifies saved
coverage/missingness reports and exposes lazy feature/daily datasets; Run All
does not launch a full build or fit preprocessing/models.

The active added-context policy is [80% original-cohort coverage](docs/CONTEXT_AVAILABILITY.md),
with 90% retained as a separate sensitivity check,
with exhaustive admitted-ID requests, per-variable/date/sleeve diagnostics and
unchanged exact temporal windows. The paper15 keep their existing common-five
population. The earlier complete-cohort build remains historical comparison evidence.

The historical user-authorized 90% context rebuild is complete: all 402 previously empty
context predictors have observations somewhere in 2010–2023; none are entirely
missing over that full period. Paper15,298 technical features and every preserved
daily artifact match the previous run exactly. Among paper-qualified rows,
context-value coverage is 59.52% in 2010–2019 and 98.08% in 2020–2023. Early missingness
remains material; these are descriptive counts, not full-history feature selection.
See [the full-panel verification](docs/FULL_FEATURE_PANEL.md) and the active
context contract for detailed annual/feature coverage.

By subsequent user direction, **80% is primary and 90% is the sensitivity check**.
The verified panels retain their original paths, identities and bytes; their
roles are recorded in `configs/context_panel_roles.json`. Read both panels and
saved comparison evidence with `dmv2.context_panel_roles.read_context_panel_roles`.
Historical comparison tables retain their original **90% → 80%** orientation.
Notebook Run All verifies saved evidence without starting a build or fitting a model.
`configs/context_panel_roles.json` selects the active roles; original output
pointers retain their historical build meanings.
See [the comparison and promotion record](docs/CONTEXT_SENSITIVITY_80.md).

The [verified 80% primary panel](docs/CONTEXT_SENSITIVITY_80.md) has 96.27% finite
context values among paper-qualified rows over 2010–2023 and 92.09% in 2010–2015.
The retained 90% sensitivity has 71.06% and 33.57%, respectively. All shared finite
values, paper15, technical298 and daily artifacts are unchanged. Missingness may
be systematic; promoting 80% does not establish representative or certified PIT data.
