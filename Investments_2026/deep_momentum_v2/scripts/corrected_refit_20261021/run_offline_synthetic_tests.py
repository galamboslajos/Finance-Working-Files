#!/usr/bin/env python3
"""Checkout-portable fabricated suite; never opens the original research inputs.

Invoke with the curated code directory as its sole argument. This new test-only
entrypoint does not modify the executed worker/test bytes. It deliberately omits
tests that authenticate the original local research layout or real model files.
"""
from pathlib import Path
import json
import os
import sys
import unittest

sys.dont_write_bytecode = True
SOURCE = Path('/private/tmp/dmv2-corrected-refit-20261021')
RESEARCH = Path('/Users/lajosgalambos/Desktop/Finance-Working-Files/Investments_2026/deep_momentum_v2')
TESTS = [
    'test_control', 'test_model_worker.StructuralTests', 'test_verify_models',
    'test_run_model_pipeline', 'backtest_test_integration',
    'backtest_test_metadata_runtime', 'backtest_test_planning_budget',
    'backtest_test_make_control',
    'test_verify_backtest_v5.Math',
    'test_verify_backtest_v5.PreflightSummaryClockException',
    'test_verify_backtest_v5.Planning',
    'test_verify_backtest_v5.SavedRepresentation',
    'test_verify_backtest_v5.ArrowPreflightNullRepresentation',
    'test_verify_backtest_v5.FreshV5Attempt',
    'test_report_models', 'test_report_backtest_v5',
    'test_build_install_scope', 'test_install_corrected_refit',
    'test_terminal_guard_v2', 'test_progress_status',
    'test_learning_repairs', 'test_verify_learning_repairs',
    'test_finalize_learning_repairs',
]
EXCLUDED_TESTS = {
    'test_model_worker.StructuralTests.test_incomplete_fit_never_repeats',
    'test_model_worker.StructuralTests.test_incomplete_prediction_never_repeats',
}


def main():
    bundle = Path(sys.argv[1] if len(sys.argv) == 2 else Path(__file__).parent).resolve()
    if bundle == SOURCE or bundle.is_relative_to(RESEARCH/'outputs'):
        raise ValueError('Use the public code bundle, never the sealed research archive')
    # /private/tmp is a literal retained by the original fabricated tests.
    if not Path('/private/tmp').is_dir():
        raise ValueError('This unchanged test snapshot requires a writable /private/tmp')
    sys.path.insert(0, str(bundle))
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    denied = []
    def audit(event, args):
        if event == 'open' and args and isinstance(args[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(args[0])).resolve()
            if path.is_relative_to(SOURCE) or (path.is_relative_to(RESEARCH)
                    and not path.is_relative_to(RESEARCH/'.venv')):
                denied.append(str(path)); raise RuntimeError('Original research path is forbidden: '+str(path))
        if event == 'import' and args and str(args[0]).split('.')[0] in {'sklearn', 'xgboost', 'joblib', 'dmv2'}:
            raise RuntimeError('Estimator/project-source imports are outside the offline suite: '+str(args[0]))
        if event in ('socket.connect', 'socket.connect_ex', 'subprocess.Popen'):
            raise RuntimeError('Network/real subprocess dispatch is outside the offline suite')
    sys.addaudithook(audit)
    loaded = unittest.defaultTestLoader.loadTestsFromNames(TESTS)
    def leaves(value):
        for item in value:
            if isinstance(item, unittest.TestSuite):
                yield from leaves(item)
            else:
                yield item
    suite = unittest.TestSuite(item for item in leaves(loaded) if item.id() not in EXCLUDED_TESTS)
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    print(json.dumps({'schema': 'dmv2-curated-offline-synthetic-tests/v1',
        'tests_run': result.testsRun, 'failures': len(result.failures),
        'errors': len(result.errors), 'skipped': len(result.skipped),
        'explicitly_excluded_project_import_tests': sorted(EXCLUDED_TESTS),
        'status': 'PASS' if result.wasSuccessful() else 'FAIL',
        'original_research_reads_denied': denied,
        'original_source_or_model_reads_allowed': False,
        'actual_estimator_imports_or_fits_allowed': False,
        'network_or_real_subprocess_dispatch_allowed': False}, sort_keys=True))
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
