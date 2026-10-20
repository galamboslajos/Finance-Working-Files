# Completed RF-expanded temporal tuning — 20 October 2026

Status: COMPLETE; independent zero-fit verification returned `PASS_COMPLETE`.
The separately authorized daily replay also passed its independent V2 audit.
This note records completed evidence; the original design-only protocol and
later execution/restart amendments retain their exact historical bytes.

## Selection and chronology

The frozen eight-candidate search completed all 96 inner fits and four selected
annual refits. Each outer outcome year 2020–2023 uses the preceding ten completed
outcome years. Inside that window, three chronological validation years follow
7/8/9 training years. Selection minimizes the equally weighted mean of 36 monthly
log losses, without portfolio metrics, random row splits or outer-year selection.
Sixteen preprocessing states use their own training rows only. All final annual
states retain the 715 candidate predictors plus their missingness indicators;
three inner states drop six predictors under the frozen all-missing/constant
training-only rule. Missing future labels never remove eligible forecasts.

Every candidate has 500 trees and seed 1729. The grid is depth 8/12, leaf minimum
20/100 and `max_features` sqrt/0.05. The selected annual settings are:

| Outcome year | Depth | Minimum leaf | Max features | Inner monthly log loss |
| --- | ---: | ---: | --- | ---: |
| 2020 | 12 | 20 | sqrt | 2.259940 |
| 2021 | 8 | 20 | sqrt | 2.265133 |
| 2022 | 12 | 100 | sqrt | 2.266077 |
| 2023 | 12 | 100 | sqrt | 2.268901 |

The same historical S&P 500/400 union, primary 80% context, Paper15 qualification,
source-corrected inputs, BBBY quarantine, provisional identity policy and fixed
DF/RDC trading restrictions remain in force. Daily feature history and daily
execution data remain separate. No protected 2024+ financial values were read.

## Prediction diagnostics and economic comparison

Both expanded RF versions retain the same 43,655 original stock-month rows,
42,145 eligible forecasts and 42,080 observed labels over 48 months. The 65
eligible missing outcomes remain explicit. On the common observed-label mask,
equally weighted monthly outer-year diagnostics are:

| Model | Log loss | Accuracy |
| --- | ---: | ---: |
| RF Paper15 | 2.282741 | 12.35% |
| XGBoost Paper15 | 2.284619 | 12.16% |
| RF expanded original | 2.268053 | 13.36% |
| XGBoost expanded | 2.267959 | 13.49% |
| RF expanded tuned | 2.268171 | 13.11% |

The search did not improve aggregate outer log loss or accuracy. The separately
frozen class 1/10 daily replay improved the user-selected RF stop path from
+18.82% to +42.78% total net return at 10 bp costs, while maximum drawdown stayed
−35.14%. These are distinct diagnostics; portfolio returns did not choose the
hyperparameters. See [the full backtest comparison, costs, annual slices and
withheld paths](RF_EXPANDED_TUNED_BACKTEST_RESULTS_20261020.md).

2020–2023 was already examined development evidence before this tuning. It is
not a fresh untouched test. Do not expand this grid or choose another protocol
from these outer results without recording a new research decision.

## Execution, preservation and independent checks

The completed attempt used 3 hours 52 minutes of active preparation/fitting/
prediction/scoring and peaked at 6.09 GiB aggregate process-tree RSS. It reserved
7.20 GiB of compressed source reads and wrote 12.36 GiB of private artifacts.
All 100 fits completed without automatic retries or concurrent candidates.
A stopped earlier preparation attempt initiated zero estimator fits; its time,
read and output charges were subtracted from the restart's ceilings and its 22
files remain unchanged. Pause/resume checkpoints and both attempts are preserved.

The independent verifier checked all 100 fit manifests, 96 inner probability
ledgers, 16 prepared states, chronological keys/masks, all winner selections and
four selected saved-model probability reproductions. It performed zero fits and
no source financial decoding. Verification took 516 seconds and peaked at
1.50 GiB RSS. Its matrix checks use retained float32 values; dropped columns
rely on pinned training-only schema evidence. This is not independent float64
source reconstruction, all-715-formula validation or identity/PIT certification.

Private completed run:
`outputs/rf_expanded_tuning/20261020T092321879709Z-792c5bb9b38e`, identity
`sha256:792c5bb9b38e2bc35dfae80ef03d395314f0247817a7e74a1c34dc827719fc3b`.
Its final independent receipt is `independent_verification_20261020.json`.
The postrun comparison is `postrun_comparison_20261020.json`. These private
artifacts, sources, models, matrices and ledgers are not distributed by Git.

## Code and reproduction boundaries

- Design: [frozen statistical protocol](RF_EXPANDED_TUNING_20261020.md).
- Execution and checkpoints: [execution amendment](RF_EXPANDED_TUNING_EXECUTION_20261020.md)
  and [explicit restart/budget accounting](RF_TUNING_RESTART_20261020.md).
- Tuning: `scripts/run_rf_expanded_tuning.py` and
  `scripts/verify_rf_expanded_tuning.py`; workers `src/dmv2/rf_tuning_*.py`.
- Daily replay: `scripts/run_rf_tuned_backtest.py`; use the reviewed V2 verifier
  `scripts/verify_rf_tuned_backtest_v2.py` and supervisor
  `scripts/watch_rf_tuned_backtest_v2.py`. Their V1 predecessors remain historical.
- Three new test files use checkout-relative roots in this publication. The
  executing source and frozen protocol files are copied byte for byte; the
  original research checkout and run pins remain unchanged.

Use each explicit CLI's `--help` for its arguments. A fresh clone lacks the
approved private inputs and verification authority; restore those in a separately
configured evidence workspace without weakening hashes. Do not start a fresh
search, resume a completed attempt or replay a backtest merely to view the saved
results. The master notebook stays passive and does not launch this experiment.
