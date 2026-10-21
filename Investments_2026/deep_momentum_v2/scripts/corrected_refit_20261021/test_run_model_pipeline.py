"""Fabricated controller tests; no real model/source/guard dispatches."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, Mock
import contextlib
import copy
import importlib.util
import json
import signal
import subprocess
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('pipeline_under_test', HERE / 'run_model_pipeline.py')
g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)


class Fixture:
    def __init__(self, directory):
        self.root = Path(directory); self.run = self.root / 'run/models'
        self.source_root = self.root / 'source'; self.source_root.mkdir()
        self.protocol = self.root / 'protocol.json'; g.save_new(self.protocol, {'status': 'SEALED'})
        self.p = {'python_path': '/fabricated/python', 'limits': {'per_fit_seconds': 1200}}
        for name in ('pipeline_pin', 'worker_pin', 'verification_worker_pin', 'control_pin', 'supervisor_pin'):
            path = self.root / (name + '.py'); path.write_text('# fabricated code\n'); self.p[name] = g.pin(path)
        self.reservations = []; self.dispatched = []; self.wait_once = False; self.pause_after = None; self.fail_once = False
        self.worker = SimpleNamespace(plan=self.source_plan, action_plan=self.action_plan)
        self.verifier = SimpleNamespace(make_plan=self.verify_plan, make_final_plan=self.final_plan)

    def source_plan(self, root, protocol, run):
        run.mkdir(parents=True); g.save_new(run / 'read_plan.json', {'no_actual_data': True})
        g.save_new(run / 'build_identity.json', {'no_actual_data': True})

    def action_plan(self, root, protocol, run, action, year, arm, write=False):
        return {'action': action, 'year': year, 'arm': arm,
                'projected_reads': {'compressed_source': 2 if action in ('prepare', 'predict', 'finalize') else 0,
                                    'derived_matrix': 3 if action in ('fit', 'predict') else 0}}

    def verify_plan(self, root, protocol, run, year, output):
        return g.save_new(output, self.action_plan(root, protocol, run, 'verify', year, None))

    def final_plan(self, protocol, run, output):
        return g.save_new(output, self.action_plan(None, protocol, run, 'verify_final', None, None))

    def reserve(self, root, label, amount, detail, category):
        if label in [r[0] for r in self.reservations]: raise ValueError('reservation reused')
        self.reservations.append((label, amount, category, copy.deepcopy(detail)))

    def dispatch(self, p, launch_pin):
        launch = g.read(g.check(launch_pin)); plan = g.read(g.check(launch['action_plan_pin']))
        step = {k: plan[k] for k in ('action', 'year', 'arm')}
        self.dispatched.append(step)
        if self.wait_once:
            self.wait_once = False
            g.save_new(self.root / 'run/control' / (launch['phase'] + '.json'), {'status': 'WAITING_FOR_MEMORY'})
            return 0
        if self.fail_once:
            self.fail_once = False
            out = g.result_path(self.run, step)
            if step['action'] == 'fit': g.save_new(out.parent / 'FIT_ATTEMPT.json', {'consumed': True})
            g.save_new(self.root / 'run/control' / (launch['phase'] + '.json'), {'status': 'FAILED_OR_STOPPED'})
            return 1
        status = {'prepare': 'COMPLETE', 'fit': 'FIT_COMPLETE', 'predict': 'COMPLETE', 'verify': 'PASS',
                  'finalize': 'COMPLETE_PENDING_INDEPENDENT_VERIFICATION', 'verify_final': 'PASS'}[step['action']]
        value = {'status': status, 'year': step['year'], 'source_plan_pin': g.pin(self.run / 'read_plan.json'),
                 'verified_estimator_fits': 20, 'verified_training_only_preprocessing_states': 8}
        item = g.save_new(g.result_path(self.run, step), value)
        if step['action'] not in ('verify', 'verify_final'):
            g.save_new(self.run / 'checkpoints' / f"{step['action']}_{step['year']}_{step['arm']}.json",
                       {'status': 'COMPLETE', 'result_pin': item})
        g.save_new(self.root / 'run/control' / (launch['phase'] + '.json'),
                   {'status': 'COMPLETE', 'launch_pin': launch_pin})
        if self.pause_after == len(self.dispatched):
            g.save_new(self.root / 'PAUSE_REQUESTED.json', {'pause': True})
        return 0

    @contextlib.contextmanager
    def patches(self):
        with patch.object(g, 'ROOT', self.root), \
             patch.object(g, 'configuration', return_value=(self.source_root, self.protocol, self.p, self.run)), \
             patch.object(g, 'code_checks'), patch.object(g, 'controller_lock', contextlib.nullcontext), \
             patch.object(g, 'load_module', side_effect=lambda path, name: self.worker if 'worker' in name else self.verifier), \
             patch.object(g, 'reserve_read', side_effect=self.reserve), patch.object(g, 'dispatch', side_effect=self.dispatch):
            yield

    def execute(self, resume=False):
        with self.patches(): return g.pipeline(self.source_root, self.protocol, g.pin(self.protocol)['sha256'], resume)


class Tests(unittest.TestCase):
    def test_schedule_twenty_fits_eight_states_and_annual_verification_barrier(self):
        steps = g.schedule(); self.assertEqual(len(steps), 50)
        self.assertEqual(sum(s['action'] == 'fit' for s in steps), 20)
        self.assertEqual(g.counts(steps)['completed_preprocessors'], 8)
        for year in g.YEARS:
            annual = [s for s in steps if s['year'] == year]
            self.assertEqual(annual[0]['action'], 'prepare'); self.assertEqual(annual[-1]['action'], 'verify')
            for i, arm in enumerate(g.ARMS):
                self.assertEqual(annual[1 + 2*i], dict(action='fit', year=year, arm=arm))
                self.assertEqual(annual[2 + 2*i], dict(action='predict', year=year, arm=arm))

    def test_full_fabricated_pipeline_completes_exact_fifty_phases(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            f=Fixture(d); result=f.execute()
            self.assertEqual(result['status'], 'COMPLETE_VERIFIED')
            self.assertEqual(result['completed_estimator_fits'], 20)
            self.assertEqual(result['completed_preprocessors'], 8)
            self.assertEqual(f.dispatched, g.schedule())
            self.assertEqual(len(list((f.run/'pipeline/checkpoints').glob('checkpoint.*.json'))), 50)
            self.assertEqual(len([r for r in f.reservations if r[0].startswith('pipeline_result_')]), 50)

    def test_pause_finishes_current_phase_then_explicit_resume_skips_it(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            f=Fixture(d);f.pause_after=2; result=f.execute()
            self.assertEqual(result['status'],'PAUSED');self.assertEqual(len(f.dispatched),2)
            with self.assertRaisesRegex(ValueError,'Explicit resume'):f.execute()
            (f.root/'PAUSE_REQUESTED.json').unlink();f.pause_after=None
            result=f.execute(True);self.assertEqual(result['status'],'COMPLETE_VERIFIED')
            self.assertEqual(f.dispatched,g.schedule())

    def test_wait_resume_uses_fresh_guard_and_reservation_but_identical_plan(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            f=Fixture(d);f.wait_once=True; result=f.execute()
            self.assertEqual(result['status'],'WAITING_FOR_MEMORY')
            old=result['attempts'][0]; old_launch=g.pin(old['launch_pin']['path'])
            result=f.execute(True)
            self.assertEqual(result['status'],'COMPLETE_VERIFIED')
            a,b=result['attempts'][:2]
            self.assertEqual(a['action_plan_pin'],b['action_plan_pin'])
            self.assertNotEqual(a['phase'],b['phase']);self.assertNotEqual(a['reservations'],b['reservations'])
            self.assertEqual(g.pin(old_launch['path']),old_launch)
            self.assertEqual(len(f.dispatched),51)

    def test_failed_guard_is_not_resumable_or_retried(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            f=Fixture(d);f.fail_once=True
            with self.assertRaisesRegex(ValueError,'Guarded phase failed'):f.execute()
            self.assertEqual(g.read(f.run/'pipeline/progress.json')['status'],'FAILED_OR_STOPPED')
            with self.assertRaisesRegex(ValueError,'Explicit resume'):f.execute(True)
            self.assertEqual(len(f.dispatched),1)

    def test_consumed_fit_and_preparation_markers_cannot_retry(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            f=Fixture(d)
            for step in (g.schedule()[0],g.schedule()[1],g.schedule()[2],g.schedule()[11]):
                out=g.result_path(f.run,step)
                name={'prepare':'DATA_STARTED.json','fit':'FIT_ATTEMPT.json','predict':'PREDICT_STARTED.json',
                      'verify':out.name+'.STARTED.json'}[step['action']]
                marker=out.parent/name;g.save_new(marker,{'consumed':True})
                with self.assertRaisesRegex(ValueError,'Incomplete attempt'):g.ensure_fresh_action(f.run,step)

    def test_changed_completed_artifact_blocks_resume(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            f=Fixture(d);f.pause_after=1;f.execute();(f.root/'PAUSE_REQUESTED.json').unlink()
            g.result_path(f.run,g.schedule()[0]).write_text('{}')
            with self.assertRaisesRegex(ValueError,'Pinned bytes changed'):f.execute(True)

    def test_reservation_before_dispatch_and_no_zero_category(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            f=Fixture(d);path=f.root/'plan.json';item=g.save_new(path,{'metadata':True})
            with f.patches():labels=g.reserve_categories('fakephase',item,{'projected_reads':{'compressed_source':7,'derived_matrix':0}})
            self.assertEqual(labels,{'compressed_source':'fakephase_compressed_source'})
            self.assertEqual(f.reservations[0][1],7)
            with f.patches(),self.assertRaisesRegex(ValueError,'Both read categories'):
                g.reserve_categories('bad',item,{'projected_reads':{'compressed_source':7}})

    def test_model_command_binds_exact_plan_and_launch_without_hash_cycle(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            f=Fixture(d);step=g.schedule()[1]; pp=g.save_new(f.root/'p.json',{})
            with f.patches():command=g.command_vector(f.p,f.source_root,f.protocol,f.run,step,pp,{'derived_matrix':'fitbytes'},f.root/'launch.json')
            self.assertIn('--launch',command);self.assertNotIn('--launch-sha256',command)
            self.assertEqual(command[command.index('--action-plan-sha256')+1],pp['sha256'])
            self.assertIn('derived_matrix=fitbytes',command)

    def test_guard_cleanup_sigterm_waits_for_supervisor(self):
        child=Mock(pid=87654);child.poll.return_value=None
        with patch.object(g.os,'killpg') as kill:g.terminate_guard(child)
        kill.assert_called_once_with(87654,signal.SIGTERM);child.wait.assert_called_once_with(timeout=25)

    def test_guard_cleanup_escalates_only_after_grace(self):
        child=Mock(pid=87654);child.poll.return_value=None
        child.wait.side_effect=[subprocess.TimeoutExpired('guard',25),0]
        with patch.object(g.os,'killpg') as kill:g.terminate_guard(child)
        self.assertEqual(kill.call_args_list[0].args,(87654,signal.SIGTERM))
        self.assertEqual(kill.call_args_list[1].args,(87654,signal.SIGKILL))

    def test_guard_cleanup_masks_repeat_signals_and_restores_handlers(self):
        child=Mock(pid=87654);child.poll.return_value=None
        old_term,old_hup=object(),object()
        handlers={signal.SIGTERM:old_term,signal.SIGHUP:old_hup}
        def wait(**kwargs):
            self.assertEqual(handlers[signal.SIGTERM],signal.SIG_IGN)
            self.assertEqual(handlers[signal.SIGHUP],signal.SIG_IGN)
            self.assertEqual(kwargs,{'timeout':25})
            # A repeated TERM/HUP during the grace period is ignored, allowing
            # the supervisor's own child-group cleanup to finish.
            return 0
        child.wait.side_effect=wait
        with patch.object(g.os,'killpg'),patch.object(g.signal,'getsignal',side_effect=lambda s:handlers[s]), \
             patch.object(g.signal,'signal',side_effect=lambda s,h:handlers.__setitem__(s,h)):
            g.terminate_guard(child)
        self.assertIs(handlers[signal.SIGTERM],old_term);self.assertIs(handlers[signal.SIGHUP],old_hup)

    def test_dispatch_signal_unwinds_owned_guard_before_propagating(self):
        child=Mock(pid=76543);child.wait.side_effect=SystemExit('fabricated signal')
        with patch.object(g.subprocess,'Popen',return_value=child),patch.object(g,'terminate_guard') as cleanup:
            with self.assertRaises(SystemExit):g.dispatch({'python_path':'/fake/python'},{'path':'/fake/launch','sha256':'abc'})
        cleanup.assert_called_once_with(child)


if __name__=='__main__':unittest.main()
