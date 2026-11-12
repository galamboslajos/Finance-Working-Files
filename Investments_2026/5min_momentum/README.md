# 5min_momentum

Five-minute Alpaca bars, historical S&P 500 membership, and next-session
open-to-close long/short momentum research. Current work: review the completed
267-feature RET/corrected-ranker comparison against the 42-feature versions.
The strict historical representation uses ten completed sessions and nine internal
gaps. Earlier RET/ranker and cost experiments remain historical comparisons. ML experiments
retain 100-member ensembles, 12/3 WFO and N=2/3. Network and one-minute work remain
deferred. Historical completed model and feature releases are immutable.

Canonical source: `Finance-Working-Files/Investments_2026/5min_momentum`.
Local `data/` and `.venv/` point to private storage under
`Documents/ChatGPT/deep_momentum_5min`; that is not a second edited source tree.

## Start here

- [Research contract](docs/RESEARCH_CONTRACT.md): exact features, clocks, labels,
  validation, stopping and portfolios.
- [Audit findings](docs/AUDIT_FINDINGS.md): corrected defects and unresolved biases.
- `data/reports/08_research_review/AUDIT.json`: machine-readable audit evidence.

```sh
python3.14 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -B -m unittest discover -s tests
.venv/bin/python -B scripts/audit_research.py
```

## Active notebooks

| Notebook | Purpose |
| --- | --- |
| `notebooks/01_load_sp500_5min.ipynb` | Coverage, missing slots, membership, sample bars |
| `notebooks/06_build_volume_and_overnight_features.ipynb` | Inspect pinned inputs; no automatic rebuild |
| `notebooks/07_train_xgb_ret_volume_overnight.ipynb` | Completed N=2/3 results, importance and explained variance |
| `notebooks/09_train_xgb_ret_joint.ipynb` | Joint original/network momentum + volume + overnight experiment and comparison |
| `notebooks/10_train_xgb_ret_rich_path.ipynb` | Joint30 vs 42-input five-minute path model; N=2/3, importance and explained variance |
| `notebooks/11_ret_tuning_and_ranker.ipynb` | Tuned RET and untuned ranker on 42 frozen inputs, with historical RET42 reference |
| `notebooks/12_ranker_portfolio.ipynb` | Completed ranker with corrected top/bottom selector and equal-N portfolio stopping |
| `notebooks/13_stop_overlays.ipynb` | ATR, opening-gap and hybrid stops; standard 10bp per entry/exit side on frozen policies |
| `notebooks/14_stop_cost_recalibration.ipynb` | Stops recalibrated separately at 5/10/20bp per side; N=2/3 and frozen-policy comparison |
| `notebooks/15_build_session_window_features.ipynb` | Strict ten-session / nine-gap feature panel; coverage, duplicates, units and lookahead audit before training |
| `notebooks/16_train_session_window.ipynb` | Completed 267-feature RET/corrected-ranker comparison; N=2/3, 5/10/20bp costs, training gain and matched RET explained variance |

One runner replaces seven separate runners:

```sh
.venv/bin/python scripts/run_notebook.py 06
.venv/bin/python scripts/run_notebook.py 07
```

It enforces inspection flags, closes its kernel and saves executed outputs only
under `data/reports/`. Source notebooks have no outputs. Notebook 01 scans local
five-minute bars; 06/07 inspect derived artifacts. No one-minute work is added.

## Code map

| Files under `scripts/` | Responsibility |
| --- | --- |
| `build_momentum_features.py` | Calendar anchors and original eligibility ledger/baseline |
| `network_paper_features.py`, `build_network_paper_features.py` | Eight individual momentum inputs; no graph/OLS |
| `volume_features.py`, `build_volume_overnight_features.py` | Six relative-volume inputs and opening gap |
| `joint_features.py` | Join the two frozen feature blocks; remove exact duplicates without outcome-based selection |
| `rich_5min_features.py` | Twelve prior-close path features, dense clocks, strict causal references and exact cohort preservation |
| `ret_backtest.py` | Labels, folds, class payoffs and portfolio accounting; no fitting entry point |
| `train_xgb_ret_ensemble.py`, `xgb_validation_stopping.py` | Shared 100-member engine with explicit momentum8/combined15/joint30/path42 contracts |
| `learner_experiments.py` | Parallel 24-setting RET tuning and fixed pairwise ranking protocol; shared CPU scheduler and result inspection |
| `train_ranker_portfolio.py`, `portfolio_stopping.py` | Corrected full-candidate, fixed-date portfolio validation and shared selector |
| `stop_loss_math.py`, `backtest_stop_overlays.py` | Prior-year ATR/gap/hybrid calibration, first-touch replay and exact coverage ledgers |
| `trading_costs.py`, `report_stop_costs.py` | Standard 10bp per executed side, actual exit-notional costs and frozen-fill reports |
| `recalibrate_stop_costs.py` | Cost-specific 12/3 stop calibration and comparison against original frozen stops |
| `research_checks.py`, `audit_research.py` | Strict guards and independent frozen-run inspection |
| `analyze_volume_overnight_results.py` | All-model gain and held-out return-fit figures |
| `download_published_5min.py`, `prepare_published_sp500.py`, `gcs_readonly.py` | Pinned read-only ingestion and selection |

The joint run uses `--experiment joint30`, earliest fitting time
`2026-11-06T14:40:00+00:00`, ten CPU workers and zero available-memory launch
threshold. There is no round budget, fixed RAM ceiling or wall-clock deadline;
validation patience remains 30. Its runtime benchmark also waits for the gate.
An exclusive controller lock prevents duplicate launches. The completed job
automatically executes notebook 09, including importance and explained variance.

The shared trainer exposes `--help`. A new implementation requires a new private
run directory; old checkpoints/configurations cannot be silently mixed with it.

The new run uses `--experiment path42`, no start gate, the same ten workers,
zero available-memory threshold and no round/RAM/runtime ceilings. Validation
patience is thirty. Its feature release is `data/features/10_rich_5min_path_v1`;
its private run is `data/models/10_xgb_ret_rich_path100_v1`. Completion automatically
executes notebook 10. Inspection with `scripts/run_notebook.py 10` never fits or
rebuilds absent releases. The original joint30 panel, models and reports remain
unchanged.

The two-stream experiment uses `train_xgb_ret_ensemble.py --mode comparison`
with private run directory `data/models/11_ret_ranker_parallel_v1`. It evaluates
2,880 past-only tuning fits and 8,000 final members across the two experiments.
Ten CPU workers are shared, without fixed RAM/round/time ceilings. Final RET
member row splits remain unchanged; ranker splits use whole dates. This
validation difference is explicit in notebook 11. Its source stays output-free;
completion writes the private comparison and importance/RET-EV plots automatically.

## Private data and history

| Directory | Role |
| --- | --- |
| `data/alpaca_5min_20261103` | Verified published multi-index five-minute source, both adjustments |
| `data/sp500_5min_20261103` | S&P 500 member-day selection, raw and adjusted |
| `data/sp500_pit_20261103` | Retrospective membership, including unresolved identities |
| `data/features`, `data/models` | Immutable releases and completed experiments/checkpoints |
| `data/reports` | Private executed notebooks, audits and figures |
| `data/maintenance/20261106_workspace_review` | Full inventory, pre-cleanup source archive, recovery/deletion ledger and legacy metadata |

Pinned release: `equity_us_intraday_5min_alpaca/v1/_releases/professional-source-build-20261103-r1`
under `gs://quant-research-trusty-agility-492109-v0/dataset-platform/curated/equity_us/`.
Local manifests retain coverage, object generations, hashes, rights and issues.
The full release and S&P 500 subset intentionally overlap; they are not additional
historical coverage and both pinned artifacts are retained.

Retired notebooks 02–05, controllers and documents remain in the private source
archive. Their executed reports, completed models and exact source snapshots are
retained. Refactoring does not relabel old results as outputs of new code.

Licensed data, models, executed outputs, credentials and `.venv` remain excluded
from Git. Cloud operations are read/download only. Notify before downloads over
50 GB. GitHub publication contains reviewed source and output-free notebooks only.

Notebook 12 inspects the corrected portfolio-stopped ranker (same learner/features/100-member 12/3 WFO). Its fixed complete-outcome validation date policy and full trace ledger are specified in docs/RESEARCH_CONTRACT.md. Existing RET runs use pinned sources and continue independently.


## Completed experiments and cost reporting

RET tuning and both ranking protocols have completed their original schedules.
The corrected ranker has 100 members for each of forty quarters. Inspection
reports, stopping traces, importance plots and comparison audits remain private.
No learner parameters, seed/split schedules or completed releases were changed
by the stop and transaction-cost experiments.

Performance standard: **10bp per executed entry/exit side**. Entry and actual
exit dollar notionals both pay the cost, including stopped exits. At 100% gross,
a flat round trip costs approximately 20bp of equity. Five/ten/twenty-bp
sensitivities and cost-specific stop calibrations retain the original signals,
weights, WFO dates and conditional observation masks. Notebook 13 keeps the
original 1bp stop calibration; notebook 14 recalibrates at each scenario's cost.
Neither inspection notebook starts fits or rebuilds missing releases.

```sh
.venv/bin/python -B scripts/run_notebook.py 12
.venv/bin/python -B scripts/run_notebook.py 13
.venv/bin/python -B scripts/run_notebook.py 14
```

Stop outputs are versioned under `data/backtests/13_stop_overlays_v1` and
`data/backtests/14_stop_cost_recalibration_v1`; cost reports and private executed
notebooks are under `data/reports`. See the research contract for calibration,
unknown-outcome, warm-up and date-mask rules. Aggregate assumed costs do not
certify opening/stop fills, complete tradability, PIT membership or borrow.


## Strict ten-session input representation

Notebook 15 inspects a new label-free feature release with 267 session-history
columns. It uses exactly ten completed exchange sessions and nine internal gaps,
including no price or smoothing state outside the block. Normalization uses
only the completed block; activity references the other nine same-window days.
Every resolved signal-day member is retained without a legacy/future-price gate.
The independent audit and exact missing-reason ledger precede any training.

```sh
.venv/bin/python -B scripts/build_session_window_features.py
.venv/bin/python -B scripts/audit_session_window_features.py
.venv/bin/python -B scripts/run_notebook.py 15
```

These commands construct/audit inputs and inspect results; no ML fits occur.
Source modules separate numerical definitions, input construction and independent
inspection. Completed historical feature/model releases remain unchanged.


Notebook 16 inspects the completed private run
`16_session_window_ret_ranker100_v1`: 4,000 RET models and 4,000 corrected-ranker
models. Both retain 100-member ensembles, 12/3 WFO, 30-round patience and fixed
learner settings, with no tuning or arbitrary round/RAM/runtime ceilings.
Reports cover N=2/3, 5/10/20bp per-side costs, conditional date masks, training
gain and matched RET return EV/R². Inspection cannot train or rebuild.

The 267-feature experiment excludes the trade-day opening gap. This was an
assistant interpretation of the history constraint, rather than an explicit
request to remove that existing input. Eligibility, normalization and
cohort-conditioned labels also differ from the 42-feature versions. Restoring
the opening gap as a separate 268th input is proposed and has not been
implemented or fitted. See the clarification in `docs/AUDIT_FINDINGS.md`.
Completed historical releases retain their original definitions.
