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
unavailable from these inputs. The local pilot uses the exact modelling package
pins in `requirements-models.lock.txt`; a fresh checkout can install them with
`.venv/bin/python -m pip install -r requirements-models.lock.txt`.

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
- `outputs/`: saved reports, panels and experiment artifacts, ignored.

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
(92.43%). Full-bank coverage is measured in [the full-panel audit](docs/FULL_FEATURE_PANEL.md).
The first four-arm classifier pilot is described below; backtesting remains outstanding. Historical capitalization
is absent from the downloaded schemas and deliberately excluded from V2; it
is not a remaining data task or blocker. The 15-input starting set is a declared
adaptation, not the paper's complete 16-input baseline. The selected primary modelling universe is the historical S&P 500/400 union.
Economic return/execution rules and subsequent experiment budgets remain OPEN.
The first development pilot has its own frozen validation, settings and budget
in `configs/classifier_pilot.json`. Exact per-signal warmup is implemented; the
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

## Model comparison and first development pilot

The user selected the historical S&P 500 + S&P 400 union. The master now displays
a metadata-only [four-arm XGBoost/random-forest design](docs/MODEL_COMPARISON.md):
both estimators on Paper15, then both on the expanded candidate bank. The draft
keeps the shared ten-class probability-to-RET interface, rolling ten-year annual
fits, monthly predictions and 80% primary/90% sensitivity context policy.
`configs/model_comparison.proposed.json` records proposed settings and open
label/validation/budget decisions; `src/dmv2/model_design.py` produces calendar
plans without estimator dependencies or data reads. This draft is retained as
historical planning evidence; it is not the active pilot recipe.

The completed, separately authorized [four-arm classifier pilot](docs/CLASSIFIER_PILOT.md)
uses 2010–2017 completed outcomes for training and 2018–2019 for validation,
with one fixed-seed fit each of RF/Paper15, XGBoost/Paper15, RF/expanded and
XGBoost/expanded. Both feature banks share the same 77,690 supervised training
rows and 20,755 validation predictions; 20,709 observed labels are scored.
Preprocessing uses training-only medians and missing indicators. Section 17
verifies saved comparison reports; **Run All does not start fitting**. Exact
settings, metrics, verification and repeat commands are in the pilot document.
Expanded RF had the lowest development log loss (2.2631), followed closely by
expanded XGBoost (2.2644); both improved on their Paper15 arms. All 690 tests,
independent probability/score verification and fresh-kernel Run All passed.
This pilot does not evaluate 2020–2023, score RET or run a backtest.

## Shared target and RET workers

[The exact target/RET contract](docs/TARGET_CLASSES_RET.md) is implemented in
`targets.py`, `classification.py` and `ret.py`. Both probability estimators use
the same original cohorts, completed-outcome class labels and dated 120-month
payoff vectors. Ties share exact midrank classes; missing returns remain
unlabelled and never remove prediction rows. Past feature requests and forward
outcome requests are checked separately. The master keeps its bounded
check and separately verifies the completed [full 2010–2023 target audit](docs/FULL_TARGET_PANEL.md)
in section 16. The full panel retains 152,307 original rows and 140,485 released
classes (99.79% of the 140,777 prediction-eligible rows); 292 eligible missing
outcomes remain visible. All ten classes occur in all 168 months, and all 48
nominal 2020–2023 RET payoff vectors have complete 120-month histories. Daily
and feature data are unchanged. Run All reads saved reports and lazy datasets;
model/preprocessor fitting remains a separate explicit action. The first
development pilot uses the same saved class state and leaves this audit intact.


The [daily development backtester](docs/DAILY_BACKTEST.md) reuses saved 2018–2019
probabilities for separate fixed-ten-per-side class 1/10 and 2/9 portfolios,
50% long / 50% short, with an independent 10% per-asset stop alternative and
explicit cost sensitivities. RET stays withheld under its 120-month gate.
Master section18 verifies saved reports only; execution is a separate CLI action.
No new fitting or 2020–2023 evaluation is included.

The separate [corporate-action rerun](docs/ACTION_BACKTEST.md) retains the same
saved selections and original daily sources, adding reviewed entitlements and
independent OHLC overlays. Master section 19 verifies its saved coverage and
remaining failures. Missing valuations continue to withhold affected performance.
The [rerun audit](docs/ACTION_BACKTEST_RESULTS.md) records 12 complete proxy paths
and 52 withheld paths. A known historical BBBY/Overstock mapping conflict also
affects prior inputs; these results must not be used as a validated strategy comparison.

The [next session checklist](docs/NEXT_SESSION.md) orders the remaining identity/data
repairs, corrected development rerun, and proposed 2020–2023 rolling evaluation.

The [BBBY identity amendment](docs/BBBY_IDENTITY_AMENDMENT.md) records the verified
old issuer ISIN and disables the known Overstock native route in a separate
metadata view. The replacement native ID and matched historical prices remain
unresolved. All original runs and cohort rows are retained. Master section 20
verifies the saved correction and dependency audit without reading feature or
return values, rebuilding panels, fitting models or changing existing pointers.
The [precise lookup request](docs/BBBY_LOOKUP_REQUEST.md) is prepared for forwarding;
it has not been sent.

The [October 16 targeted data repair pass](docs/DATA_REPAIRS_20261016.md)
adds a separate, three-key ACC/K/SO full-OHLC amendment and narrows the remaining
BBBY, Ensco Rowan and Dean Foods requirements. The eleven available Dean Foods
bars were already cached; November 18–21 remain missing and November 12 needs
an explicit halt treatment. Fresh Cloud metadata reads succeeded, with unchanged
release pins. Original panels, models and backtest results remain untouched.
See the [updated unsent owner request](docs/REMAINING_BACKTEST_DATA_REQUEST_20261016.md).

The user's subsequent October 16 instruction authorizes a separate
[data-limited development backtest](docs/DATA_LIMITED_BACKTEST.md) without
waiting for the owner: quarantine the demonstrated wrong BBBY routes while
retaining all original membership rows and denominators; rebuild 2010–2019
features and labels; refit the same four frozen classifiers on 2010–2017; and
rerun 2018–2019's 24 months. Dean Foods and Rowan remain in valid training data
and predictions but are excluded from trading before top-ten selection for the
whole development period. This retrospective restriction can introduce bias;
the new result is conditional on the documented restricted universe. Missing
prices and exits are not invented. Original runs and pointers remain unchanged,
new phases have separate identities, and 2020–2023 evaluation remains outside
this run. The method document states the contract; completion and results must
be read from the new saved audits.


The additive independent-source 2018–2019 repair and balanced-basket rerun is
documented in [INDEPENDENT_QUOTE_BACKTEST.md](docs/INDEPENDENT_QUOTE_BACKTEST.md).
It reuses the four saved 2010–2017 models, with zero new fits, and preserves all
reference runs. The full-universe identity/terminal-accounting limitations remain.
