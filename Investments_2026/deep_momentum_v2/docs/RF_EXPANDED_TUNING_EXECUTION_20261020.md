# RF-expanded tuning execution and pause/resume — 20 October 2026

The user authorized proceeding after the
[frozen statistical protocol](RF_EXPANDED_TUNING_20261020.md) and the discussion
of checkpointing to free RAM. The original design-only JSON stays unchanged.
[`rf_expanded_tuning_execution_20261020.json`](../configs/rf_expanded_tuning_execution_20261020.json)
records this later execution authority and the operational pause/resume policy.

## Scope

Implement and execute the eight-candidate RF-expanded search: three annual
chronological validation splits inside each of four rolling ten-year training
windows, 36 equally weighted monthly log losses for selection per outer year,
then one selected annual refit. Maximum 96 inner fits plus four selected outer
refits, 16 fresh preprocessing states, 500 trees per fit and seed 1729.

The 715-candidate dictionary, Paper15, primary 80% context, original historical
S&P 500/400 cohorts, target/feature cutoffs and missing-label handling remain
fixed. Reuse only pinned reviewed corrected annual inputs. Do not rebuild or
overwrite sources, daily data, old runs, old configs or the notebook. Protected
2024+ financial values stay unread. Existing comparator results remain saved;
this execution adds no other estimator fits or backtest runs.

## Durable checkpoints and freeing RAM

Save prepared split matrices, preprocessing manifests and source/key/mask
receipts before candidate fitting. Share each split's state across its eight
candidate fits; do not refit preprocessing on resumption. Save each fit's model,
native probabilities, metrics and artifact hashes before marking it complete.
Persist fit counters before fitting and track all active time/read/output
consumption cumulatively across sessions.

A graceful pause request allows the current child phase to finish, verifies its
checkpoint and starts no next fit. The worker then exits, releasing its model
and matrix memory. An operating-system suspended process would keep memory
allocated and is not the pause mechanism. Resuming explicitly verifies the same
protocol, code/runtime, input pins and completed artifact hashes, and skips
completed work. It must never reinterpret a pause as a fresh experiment with
reset budgets or change candidate order, validation masks or selection rules.

If RAM is needed immediately, an interruption may lose the current unfinished
fit. Preserve its files and initiated-fit counter. The frozen failure rule
still stops the attempt, and an interrupted/failed fit is never retried
automatically. An explicit later retry amendment would be needed; this
checkpoint design guarantees reuse of completed fits, not mid-tree recovery.

The controller writes `WAITING_FOR_MEMORY`, `WAITING_FOR_MEMORY_PRESSURE` or
`WAITING_FOR_DISK` and exits without starting a financial preparation, fit or
prediction child when a launch gate is unmet. Bounded metadata-only planning
may occur first at low available RAM; non-normal or unknown pressure waits
before that metadata child as well. Those states can resume
after resources become available without consuming an estimator fit. A pause
or resource wait is not a completed search and cannot produce a partial-grid
winner. Successful completion still requires all predefined candidates and
validation months for each annual choice.

## Resource enforcement

Keep one fit process at a time and four RF tree threads; numerical libraries
and probability inference use one thread. Warn at 16 GiB aggregate current
process-tree RSS and stop at the observed 20 GiB ceiling. Poll once per second;
sampling can overshoot. Before a financial preparation/fit/prediction child
launch require **7 GiB** by the
macOS free+inactive+speculative-page proxy, normal operating-system memory
pressure and 30 GiB free disk. Stop the worker if pressure ceases to be normal;
an unavailable pressure reading fails closed. Keep the
20-minute fit timeout, 16-hour cumulative active preparation/fit/prediction/
scoring ceiling, 8 GiB cumulative compressed source projection and 20 GiB new
artifact limit. Paused hours are excluded from active time, while resource use
already consumed remains charged. A separate zero-fit verifier has a one-hour
active-time limit. These are stop limits and do not guarantee a completion time.

The original design retains its historical 12 GiB gate. After the user's
32 GiB memory question, a read-only audit found about 8.5 GiB currently
available/reclaimable, normal macOS pressure and zero currently allocated swap.
The prior four-arm pilot measured 5,382,127,616 bytes (5.01 GiB) peak RSS; that
does not guarantee a new fit's peak. The new execution amendment records the
judgment that a 7 GiB launch floor plus continuous real pressure monitoring is
more appropriate than the initial 12 GiB precaution. The 20 GiB upper limit,
dataset, grid, fit/time/read budgets and all research rules stay fixed. This
operational amendment precedes every new fit and is unrelated to results.
Recheck resources at each dispatch; physical capacity and available memory are
different measurements. Initial free disk was 186 GiB.

Read only `sysctl -n kern.memorystatus_vm_pressure_level`. Its userspace result
uses dispatch masks: **1 normal, 2 warning, 4 critical**. The sysctl handler
[converts the internal enum to dispatch values](https://github.com/apple-oss-distributions/xnu/blob/main/bsd/kern/kern_memorystatus_notify.c);
the [private event header defines those masks](https://github.com/apple-oss-distributions/xnu/blob/main/bsd/sys/event_private.h).
Do not mistake internal enum value 1 (warning) for dispatch value 1 (normal).
Do not run memory-pressure generation or simulation commands.

## Validation and verification

Before real execution, test chronology, key/mask preservation, training-only
preprocessing, monthly scoring/ties, checkpoint tampering, pause/resume, fit
counters and watchdog failures using synthetic data/processes. Freeze exact
new executor code hashes and runtime in a fresh private run identity.

The independent zero-fit verifier reconciles saved probabilities and all
36-month candidate scores, winner/tie selection, exact training keys/masks and
completed-label cutoffs. It checks prepared matrix medians, missing flags and
imputed slots with stated float32 quantization tolerance, and reloads selected
models for inference reproduction. Dropped-column rules rely on pinned
training-only schema evidence and the unchanged preprocessing implementation;
this is not an independent float64 source-data reconstruction or certification
of all 715 feature formulas. Record the verification scope precisely.

Preserve the original run and source integrity receipts. A new daily backtest
requires a separate execution scope and budget, with the same basket, stop,
cash, execution, event, cost and withheld-path rules. The already examined
2020–2023 interval remains development evidence, not a new untouched test.

## Commands

The explicit entry point is `scripts/run_rf_expanded_tuning.py`. It provides
metadata preparation, execution, status, graceful pause and explicit resume.
No notebook cell starts this search. Run the command's `--help` for exact
arguments; the private run identity and state record are the authority for
resumption. Exact commands and the observed launch status are recorded with
the eventual execution receipt, without modifying this frozen policy.
