# Corrected-input refit: source snapshot, 21 October 2026

This folder contains our executed worker code and fabricated regression tests for the corrected-input 2020–2023 refit. The source files are byte-identical to the preserved research attempt. This is a source snapshot, not a new execution authorization or a self-contained dataset release.

The [public handoff](../../docs/CORRECTED_REFIT_AND_RET_BACKTEST_20261021.md) explains the measured results and limitations. Financial tables, fitted models, probabilities, raw/provider snapshots, private catalogs/plans/protocols, numerical receipts and the multi-gigabyte archive remain local and are excluded from this folder.

The model workflow uses five annual arms: RF/XGBoost on Paper15 and expanded predictors, plus RF expanded using prior annual CV settings. Four previous-ten-year windows produce 20 estimator fits and eight training-only preprocessing states. No fresh tuning search is performed. Fresh RET and separate class1/10 and class2/9 selections feed the unchanged daily accounting rules. All 120 stop/cost/model/basket scenarios remain represented, including withheld paths. A verification PASS does not certify full source coverage, knowledge-time identity, survivorship completeness or investability.

The current code groups are:

- `model_worker.py`, `verify_models.py`, `run_model_pipeline.py`, `report_models.py`: model preparation, fixed fits, predictions, independent checks and passive diagnostics.
- `learning_repairs.py`, `finalize_learning_repairs.py`, `verify_learning_repairs.py`, `terminal_guard_v2.py`, `freeze_model_inputs.py`, `freeze_model_protocol.py`: source-quality decisions and the input/terminal boundaries. The actual additional RCM/DISCA learning candidates were rejected; these general workers also have fabricated accepted-case tests.
- `backtest_integration.py`, `backtest_runner.py`, `backtest_make_control.py`, `backtest_make_protocol.py`: source planning, fresh selection, quote preflight and daily replay.
- `verify_backtest_v5.py`, `backtest_verification_math_v2.py`, `report_backtest_v5.py`: final independent checker and passive reporting. Verifier V5 retains physical Arrow-null identifiers in four closed metadata diagnostics. It does not grant true NaN equivalence or change accounting. `verify_backtest_v4.py` is included only because a V5 regression test confirms the strict comparer is unchanged; V4 is not the current execution entrypoint.
- `control.py`, `supervise.py`, `progress_status.py`: shared read/resource accounting, guarded dispatch and passive status.
- `build_install_scope.py`, `install_corrected_refit.py`, `root_final_handoff_v1.py`: explicit final archive and passive handoff operations. See `ARCHIVE_INSTALL_INTERFACE.md`.

`RUNTIME_ENTRYPOINTS_20261021.md` distinguishes current workers from preserved prior attempts. `DATA_FLOW_20261021.md`, `MONTHLY_ENDPOINT_LABEL_POLICY.md` and `RECOVERY_INPUT_DECISIONS.md` describe timing, target alignment and the provisional input policy. Earlier code versions, full fabricated integration packets and failed attempts remain in the private archive; they are not all duplicated into Git.

## Offline fabricated tests

With Python and the project's NumPy, pandas and PyArrow dependencies available, run from this folder:

```sh
PYTHONDONTWRITEBYTECODE=1 python run_offline_synthetic_tests.py .
```

The exact selected suite passed **214 tests**, with zero skipped tests, in a new temporary code-only copy. The new test-only entrypoint refuses original research/staging reads, `dmv2`/estimator-library imports, networking and actual subprocess dispatch. It reads only this copied code, installed third-party libraries and invented temporary files. Fake estimator interfaces exercise prediction checks; no estimator is fitted or loaded.

The original tests retain the Unix/macOS `/private/tmp` temporary-directory convention, so that directory must exist and be writable. Their executed bytes were not rewritten to change platform paths. The suite deliberately excludes the `InventedPreparationTests` class and the `FailedMetricNullRepresentation` class: these import/authenticate project code at the original local research path. It also explicitly excludes the two incomplete-fit/predict marker methods from `StructuralTests`, because those methods import the project helpers before reaching their marker check. Those exact tests remain available in the files, and the fabricated controller tests still check consumed-fit refusal. A full `unittest discover` is not the published offline recipe.

## Execution is separately guarded

Do not run the worker CLIs directly in this source folder or in the sealed archive. `control.ROOT` follows the copied file location, some original scripts retain fixed personal-machine paths, and the frozen source/runtime/parent hashes are research-attempt identities. Simply relocating paths or replacing old hashes would create a different experiment. Several commands create single-use plans, reservations, STARTED markers or artifact directories before later phases execute.

A new empirical run requires a fresh versioned workspace, approved local inputs, explicit scope and code/runtime pins, metadata-before-values projections, cumulative read reservations and authenticated resource launches. The current model design uses one sequential estimator process; prior-CV RF fitting uses four tree threads, baseline fits/inference retain their frozen thread rules. The resource/phase budgets, pause checkpoints and no-automatic-retry rules are enforced by the preserved workers. Private numerical receipts and the installed artifact catalog are required to interpret or verify an existing run; their bodies are not bundled here.

This source bundle depends on the repository's existing `src/dmv2` workers and established independent verification scripts for an authorized empirical run. The accompanying publication manifest lists the required RET/BK-XOM additions; none of the offline suite imports those project modules or opens their data. Keep all local data/model artifacts ignored. No active research pointer is promoted by publishing this code.
