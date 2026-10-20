# Explicit RF tuning restart — 20 October 2026

The user closed applications and explicitly requested a RAM recheck and restart
while keeping the stopped attempt and checkpoint behavior. The recheck showed
about 13.9 GiB available/reclaimable by the recorded macOS proxy and normal
memory pressure (userspace dispatch mask 1). The controller rechecks immediately
at launch and before every financial child phase; this dated reading is not a
standing override of the live guard.

## Preserved attempt and unchanged research

Preserve `outputs/rf_expanded_tuning/20261020T090703447545Z-58984518c932`
byte for byte, with its `STOPPED_BUDGET` status and memory-pressure evidence.
It initiated **zero RF fits**, one preparation phase, and completed no
preprocessing state or model checkpoint. It used 34.837866625 active seconds,
reserved 391,450,896 compressed source bytes and wrote 1,142,498 output bytes.
There are therefore no completed checkpoints to reuse from that attempt.

The separately authorized restart begins in a new attempt root. It uses the
same reviewed controller, data worker, fit worker and independent verifier;
their bytes are unchanged. Keep all eight RF candidates, 500 trees, seed 1729,
715 predictors, split-local preprocessing, original cohorts, chronological
7/8/9-year inner training and 36 equally weighted monthly log losses, followed
by one selected ten-year refit for each outer year. No loss, fit order, eligibility,
backtest rule, daily data or protected 2024+ boundary changes.

## Remaining budgets and identity

The original frozen design and original execution amendment are immutable.
[`rf_expanded_tuning_restart_budget_20261020_v2.json`](../configs/rf_expanded_tuning_restart_budget_20261020_v2.json)
is a separately pinned execution copy of that design. Its semantic protocol ID
is retained for the existing worker contract, and its unique
`execution_variant_id` and raw file hash distinguish this recovery. Only three
current-attempt ceilings are reduced: original time minus 35 seconds (rounding
up the parent's consumed time), original source projection minus 391,450,896
bytes, and original output limit minus 1,142,498 bytes. Parent ledger, identity,
event head, resource evidence and original design/amendment are pinned.

The [new restart amendment](../configs/rf_expanded_tuning_restart_20261020_v2.json)
records the user's specific retry authorization. All future automatic retries
remain disabled. This exception permits **one failed inherited preparation plus
16 newly scheduled preparation initiations**, yielding 17 lifetime preparation
initiations and still at most 16 completed states. RF estimator initiations
remain at most 100 across both attempts because the parent initiated none.

Keep the new ledger's fit, preprocessing and source-reservation counters local
to the new attempt so their event reconciliation and independent verification
remain valid. Do not silently seed source counters with old reservations.
Subtracting the old costs from the new limits preserves the original cumulative
16-hour active, 8-GiB projected-read and 20-GiB output ceilings. The metadata-only
preflight plans all 20 source-materialization phases and checks that parent plus
new projections fit the original read limit before any new financial decoding.

## Memory and pause/resume

Keep the current seven-GiB available-memory launch gate plus normal macOS
pressure, one fit process, four RF tree threads, one inference/numerical thread,
16-GiB RSS warning and 20-GiB aggregate process-tree RSS stop. The original
design's historical 12-GiB launch field remains unchanged; the explicit
operational amendment supplies the same reviewed seven-GiB/pressure guard.

Graceful pause finishes the current child phase, verifies/saves its artifacts,
starts no next fit and exits to release RAM. Explicit resume verifies the same
code/runtime/input/checkpoint hashes and skips completed work. Paused hours
are excluded from active compute time; already used budgets stay charged.
A new memory-pressure warning stops the worker and preserves evidence rather
than automatically retrying or choosing from incomplete candidates.

The explicit CLI remains `scripts/run_rf_expanded_tuning.py`; prepare the new
attempt with the restart-budget protocol and restart amendment above, then use
its returned `--run-root` for `run`, `status`, `pause` and `resume`. Nothing in
the master notebook launches or restarts fitting. No backtest is authorized by
this restart. Private preflight/recovery receipts are under
`outputs/rf_tuning_protocol/dmv2-rf-expanded-tuning-restart-20261020-v2/`.
