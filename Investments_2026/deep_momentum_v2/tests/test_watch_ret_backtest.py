"""Independent resource-supervisor tests. Synthetic processes and /tmp only."""
from contextlib import ExitStack, redirect_stdout
from importlib.util import spec_from_file_location, module_from_spec
from io import StringIO
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/watch_ret_backtest.py'
if not SCRIPT.exists():
    SCRIPT = Path(__file__).resolve().parents[2] / 'control/scripts/watch_ret_backtest.py'
SPEC = spec_from_file_location('rf_replay_watch_review', SCRIPT)
M = module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class WatchdogReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='dmv2-watch-review-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.run = self.root / 'outputs/ret_backtest/run'
        self.control = self.root / 'outputs/ret_backtest_control/run'
        self.child = Mock(pid=31415)
        self.child.poll.return_value = 0
        self.child.wait.return_value = 0

    def argv(self, phase='backtest'):
        script = 'verify_ret_backtest.py' if phase == 'verify' else 'run_ret_backtest.py'
        return ['--root', str(self.root), '--run', str(self.run),
                '--control', str(self.control), '--phase', phase, '--',
                str(self.root / '.venv/bin/python'), str(self.root / 'scripts' / script),
                *([] if phase == 'verify' else [phase]),
                '--root', str(self.root),
                *(['--output-directory', str(self.run)] if phase == 'prepare' else
                  ['--run', str(self.run)]),
                *(['--backtest', str(self.run / 'backtest'),
                   '--output', str(self.run / 'verification.json')] if phase == 'verify' else [])]

    def execute(self, *, masks=(1, 1), rss=1024**2, elapsed=1,
                output=0, stop_error=None, fallback_error=None, phase='backtest'):
        process = Mock()
        process.memory_info.return_value = SimpleNamespace(rss=rss)
        process.children.return_value = []
        ticks = iter([0., float(elapsed), float(elapsed), float(elapsed)])
        with ExitStack() as stack:
            stack.enter_context(patch.object(M, 'memory', return_value={'available_proxy_bytes': 9*M.GIB}))
            stack.enter_context(patch.object(M, 'pressure', side_effect=[{'dispatch_value': x, 'normal': x == 1} for x in masks]))
            stack.enter_context(patch.object(M.shutil, 'disk_usage', return_value=SimpleNamespace(free=50*M.GIB)))
            self.launcher = stack.enter_context(patch.object(M.subprocess, 'Popen', return_value=self.child))
            stack.enter_context(patch.object(M.psutil, 'Process', return_value=process))
            stack.enter_context(patch.object(M.time, 'monotonic', side_effect=lambda: next(ticks)))
            stack.enter_context(patch.object(M, 'files_size', return_value=output))
            stopper = stack.enter_context(patch.object(M, 'stop_tree', side_effect=stop_error))
            stack.enter_context(redirect_stdout(StringIO()))
            if fallback_error is not None:
                self.child.kill.side_effect = fallback_error
            M.main(self.argv(phase))
        return stopper

    def receipt(self, phase='backtest'):
        return json.loads((self.control / (phase+'.json')).read_text())

    def test_pressure_dispatch_masks_and_unknown_are_not_interchanged(self):
        for value in (0, 1, 2, 4, 8):
            with self.subTest(value=value), patch.object(M.subprocess, 'check_output', return_value=str(value)+'\n'):
                result = M.pressure()
                self.assertEqual(result['normal'], value == 1)
                self.assertEqual(result['dispatch_value'], value)
        with patch.object(M.subprocess, 'check_output', return_value='unknown\n'):
            with self.assertRaises(ValueError): M.pressure()

    def test_memory_proxy_uses_free_inactive_speculative_only(self):
        vm = ('Mach Virtual Memory Statistics: (page size of 16384 bytes)\n'
              'Pages free: 10.\nPages inactive: 20.\nPages speculative: 30.\n'
              'Pages active: 999999.\nPages occupied by compressor: 999999.\n')
        with patch.object(M.subprocess, 'check_output', return_value=vm):
            sample = M.memory()
        self.assertEqual(sample['available_proxy_bytes'], 60*16384)

    def test_happy_child_exit_records_limits_and_threads(self):
        with patch.dict(M.os.environ, {'OMP_NUM_THREADS': '8'}):
            self.execute()
        for name in M.THREADS:
            self.assertEqual(self.launcher.call_args.kwargs['env'][name], '1')
        result = self.receipt()
        self.assertEqual(result['status'], 'PASS_PROCESS_BUDGET')
        self.assertEqual(result['exit_code'], 0)
        self.assertEqual(result['limits']['process_tree_rss_stop_bytes'], 4*M.GIB)
        self.assertEqual(result['limits']['process_tree_rss_warn_bytes'], 3*M.GIB)
        self.assertFalse(self.child.kill.called)

    def test_runtime_warn_critical_and_unknown_stop_with_terminal_receipt(self):
        for mask in (2, 4, 0):
            # Use separate output paths for each attempt; no overwritten receipts.
            self.control = self.control.parent / ('mask-'+str(mask))
            self.child.poll.return_value = None
            with self.subTest(mask=mask), self.assertRaisesRegex(RuntimeError, 'memory pressure'):
                self.execute(masks=(1, mask))
            result = self.receipt()
            self.assertEqual(result['status'], 'STOPPED_OR_FAILED')
            self.assertEqual(result['pressure']['dispatch_value'], mask)

    def test_timeout_stops_without_retry(self):
        self.child.poll.return_value = None
        with self.assertRaisesRegex(RuntimeError, 'wall-time'):
            self.execute(elapsed=1201)
        self.assertEqual(self.receipt()['status'], 'STOPPED_OR_FAILED')
        self.assertIn('wall-time', self.receipt()['failure']['message'])

    def test_aggregate_rss_includes_supervisor_and_child(self):
        self.child.poll.return_value = None
        with self.assertRaisesRegex(RuntimeError, 'RSS'):
            self.execute(rss=2*M.GIB)
        self.assertEqual(self.receipt()['peak_process_tree_rss_bytes'], 4*M.GIB)

    def test_warning_does_not_change_execution_or_stop_threshold(self):
        self.execute(rss=3*M.GIB//2)
        result = self.receipt()
        self.assertTrue(result['rss_warning'])
        self.assertEqual(result['status'], 'PASS_PROCESS_BUDGET')
        self.assertFalse(self.child.kill.called)

    def test_output_budget_stops(self):
        self.child.poll.return_value = None
        with self.assertRaisesRegex(RuntimeError, 'output budget'):
            self.execute(output=512*1024**2+1)
        self.assertEqual(self.receipt()['new_output_bytes'], 512*1024**2+1)

    def test_nonzero_child_exit_is_not_success(self):
        self.child.poll.return_value = 7
        with self.assertRaisesRegex(RuntimeError, 'code 7'):
            self.execute()
        self.assertEqual(self.receipt()['exit_code'], 7)
        self.assertEqual(self.receipt()['status'], 'STOPPED_OR_FAILED')

    def test_termination_error_still_writes_receipt_and_owned_child_fallback(self):
        self.child.poll.return_value = None
        with self.assertRaisesRegex(RuntimeError, 'memory pressure'):
            self.execute(masks=(1,2), stop_error=PermissionError('enumeration denied'))
        result = self.receipt()
        self.assertEqual(result['termination_error']['type'], 'PermissionError')
        self.child.kill.assert_called_once()
        self.child.wait.assert_called_once_with(timeout=10)

    def test_termination_and_fallback_errors_still_preserve_receipt(self):
        self.child.poll.return_value = None
        with self.assertRaisesRegex(RuntimeError, 'memory pressure'):
            self.execute(masks=(1,2), stop_error=PermissionError('denied'), fallback_error=TimeoutError('kill failed'))
        result = self.receipt()
        self.assertEqual(result['termination_fallback_error']['type'], 'TimeoutError')
        self.assertEqual(result['status'], 'STOPPED_OR_FAILED')

    def test_existing_attempt_cannot_be_reused(self):
        self.execute()
        original = (self.control / 'backtest.json').read_bytes()
        with self.assertRaisesRegex(AssertionError, 'Preserve previous'):
            self.execute()
        self.assertEqual((self.control / 'backtest.json').read_bytes(), original)

    def test_command_run_is_bound_to_monitored_output_before_launch(self):
        args = self.argv()
        child_start = args.index('--') + 1
        child_run = args.index('--run', child_start) + 1
        args[child_run] = str(self.root / 'outputs/unmonitored')
        with patch.object(M.subprocess, 'Popen', side_effect=AssertionError('Launched escaped command')) as launch:
            with self.assertRaises(AssertionError): M.main(args)
        launch.assert_not_called()

    def test_duplicate_child_scope_flag_rejected_before_launch(self):
        args = [*self.argv(), '--run', str(self.run)]
        with patch.object(M.subprocess, 'Popen', side_effect=AssertionError('Launched duplicate scope')) as launch:
            with self.assertRaisesRegex(AssertionError, 'one exact'): M.main(args)
        launch.assert_not_called()

    def test_prepare_and_verification_have_their_own_bound_scopes(self):
        self.execute(phase='prepare')
        self.assertEqual(self.receipt('prepare')['limits']['wall_seconds'], 300)
        self.execute(phase='verify')
        self.assertEqual(self.receipt('verify')['limits']['wall_seconds'], 3600)


if __name__ == '__main__': unittest.main()
