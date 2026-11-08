# 5min_momentum

Five-minute Alpaca bars, historical S&P 500 membership, and next-session
open-to-close long/short momentum research. The current phase is workspace and
methodology review. No new strategy fit is part of the cleanup.

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
| `ret_backtest.py` | Labels, folds, class payoffs and portfolio accounting; no fitting entry point |
| `train_xgb_ret_ensemble.py`, `xgb_validation_stopping.py` | Shared 100-member engine with explicit momentum8/combined15 contracts |
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
50 GB. No upload or GitHub publication is part of this cleanup.
