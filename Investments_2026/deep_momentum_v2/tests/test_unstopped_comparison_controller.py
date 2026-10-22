"""Controller guards with invented metadata and fake child completion only."""
import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/run_unstopped_target_comparison.py'
spec=importlib.util.spec_from_file_location('unstopped_controller_test',SCRIPT)
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


def fixture(directory):
    root=Path(directory).resolve();root.mkdir(exist_ok=True)
    protocol={'project_root':str(root),'limits':{'source_projection_bytes':100,'backtest_source_projection_bytes':80,
        'derived_read_bytes':200,'max_fits':20,'per_fit_seconds':1200,'verification_seconds':3600,
        'backtest_seconds':3600,'rss_stop_bytes':16*1024**3,'active_seconds':10800,'output_bytes':24*1024**3}}
    protocol_path=root/'protocol.json';c.write(protocol_path,protocol)
    plan={'output_directory':str(root),'protocol_pin':c.pin(protocol_path),'folds':{'2020':{'source_projection_bytes':40}}}
    plan_path=root/'read_plan.json';c.write(plan_path,plan)
    launch={'model_plan_pin':c.pin(plan_path),'protocol_pin':c.pin(protocol_path),'code_pins':c.code_pins(),
        'python_path':str(root/'.venv/bin/python'),'maximum_estimator_fits':20,'automatic_retry':False}
    launch_path=root/'launch.json';c.write(launch_path,launch)
    ledger=dict(status='RUNNING',model_plan_pin=c.pin(plan_path),fits_started=0,fits_completed=0,
        active_seconds=0.0,current_phase_seconds=0.0,peak_process_tree_rss_bytes=0,
        active_phase=None,reservations=[],completed_phases={})
    return root,plan_path,launch_path,protocol,ledger


class ControllerTest(unittest.TestCase):
    def setUp(self):
        self.rss_patch=patch.object(c,'rss',return_value=1024);self.rss_patch.start();self.addCleanup(self.rss_patch.stop)


class ReservationTests(ControllerTest):
    def test_read_cap_cumulative_and_single_use(self):
        ledger={'reservations':[]};limits={'source_projection_bytes':100,'backtest_source_projection_bytes':80,'derived_read_bytes':200}
        c.reserve(ledger,'model_source','year2020',60,limits)
        with self.assertRaisesRegex(ValueError,'single-use'):c.reserve(ledger,'model_source','year2020',1,limits)
        with self.assertRaisesRegex(ValueError,'allowance exhausted'):c.reserve(ledger,'model_source','year2021',41,limits)
        self.assertEqual(len(ledger['reservations']),1)
        with self.assertRaisesRegex(ValueError,'nonnegative'):c.reserve(ledger,'derived','negative',-1,limits)

    def test_pause_preserves_fit_and_read_budgets(self):
        with tempfile.TemporaryDirectory() as d:
            root,p,l,protocol,ledger=fixture(d);(root/'PAUSE_REQUESTED').write_text('pause')
            result=c.run_phase(p,l,root/'ledger.json',ledger,'fit',2020,'endpoint')
            self.assertIsNone(result);self.assertEqual(ledger['status'],'PAUSED')
            self.assertEqual(ledger['fits_started'],0);self.assertEqual(ledger['reservations'],[])
            self.assertFalse((root/'controller/tickets/fit-2020-endpoint.json').exists())

    def test_unfinished_phase_is_never_retried(self):
        with tempfile.TemporaryDirectory() as d:
            root,p,l,protocol,ledger=fixture(d);ledger['active_phase']='fit-2020-endpoint'
            with self.assertRaisesRegex(ValueError,'unfinished phase'):
                c.run_phase(p,l,root/'ledger.json',ledger,'fit',2020,'endpoint')
            self.assertEqual(ledger['fits_started'],0)

    def test_twenty_fit_attempt_cap_before_dispatch(self):
        with tempfile.TemporaryDirectory() as d:
            root,p,l,protocol,ledger=fixture(d);ledger['fits_started']=20
            with self.assertRaisesRegex(ValueError,'Fit-attempt budget'):
                c.run_phase(p,l,root/'ledger.json',ledger,'fit',2020,'endpoint')
            self.assertFalse((root/'controller').exists())

    def test_changed_completed_phase_bytes_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root,p,l,protocol,ledger=fixture(d);path=root/'done.json';c.write(path,{'status':'COMPLETE_PHASE','worker_result':{}})
            ledger['completed_phases']['fit-2020-endpoint']=c.pin(path);path.write_text('{}')
            with self.assertRaisesRegex(ValueError,'Pinned controller input changed'):
                c.run_phase(p,l,root/'ledger.json',ledger,'fit',2020,'endpoint')

    def test_wrong_ticket_hash_fails_before_worker_import(self):
        with tempfile.TemporaryDirectory() as d:
            root,p,l,protocol,ledger=fixture(d);path=root/'ticket.json';c.write(path,{'phase':'fit'})
            with self.assertRaisesRegex(ValueError,'ticket hash mismatch'):
                c.worker(path,'wrongsha')

    def test_resume_requires_contiguous_phase_prefix(self):
        with tempfile.TemporaryDirectory() as d:
            root,p,l,protocol,ledger=fixture(d)
            ledger['completed_phases']={'prepare-2021':{'path':'/not/read'}}
            with self.assertRaisesRegex(ValueError,'contiguous fixed schedule'):
                c.validate_ledger(ledger,p,l)

    def test_resume_fit_counters_must_match_completed_prefix(self):
        with tempfile.TemporaryDirectory() as d:
            root,p,l,protocol,ledger=fixture(d);ledger['fits_started']=1
            with self.assertRaisesRegex(ValueError,'fit counts differ'):
                c.validate_ledger(ledger,p,l)

    def test_completed_worker_receipt_bytes_are_still_checked(self):
        with tempfile.TemporaryDirectory() as d:
            root,p,l,protocol,ledger=fixture(d)
            worker=root/'prepared_receipt.json';c.write(worker,{'status':'PREPARED'})
            ticket=root/'ticket.json';c.write(ticket,dict(phase='prepare',year=2020,head=None,launch_pin=c.pin(l)))
            done=root/'done.json';c.write(done,dict(status='COMPLETE_PHASE',phase='prepare',year=2020,head=None,
                ticket_pin=c.pin(ticket),worker_result=c.pin(worker)))
            completed=c.pin(done);worker.write_text('{}')
            with self.assertRaisesRegex(ValueError,'Pinned controller input changed'):
                c.validate_completed(completed,'prepare-2020',l)


class ReviewRegressionTests(ControllerTest):
    def test_launch_protocol_must_match_frozen_model_plan(self):
        with tempfile.TemporaryDirectory() as d:
            root,p,l,protocol,ledger=fixture(d);launch=c.read(l)
            changed=root/'changed_protocol.json';c.write(changed,{'limits':{'rss_stop_bytes':100*1024**3}})
            launch['protocol_pin']=c.pin(changed)
            with self.assertRaises(ValueError):c.validate_launch(p,launch)

    def test_completed_phase_scope_is_authenticated(self):
        with tempfile.TemporaryDirectory() as d:
            root,p,l,protocol,ledger=fixture(d);path=root/'wrong_done.json'
            ticket=root/'wrong_ticket.json';c.write(ticket,dict(phase='fit',year=2021,head='utility_long',launch_pin=c.pin(l)))
            c.write(path,dict(status='COMPLETE_PHASE',phase='fit',year=2021,head='utility_long',ticket_pin=c.pin(ticket),worker_result={'other':True}))
            ledger['completed_phases']['fit-2020-endpoint']=c.pin(path)
            with self.assertRaises(ValueError):c.run_phase(p,l,root/'ledger.json',ledger,'fit',2020,'endpoint')

    def test_global_budget_checked_before_fast_child_dispatch(self):
        with tempfile.TemporaryDirectory() as d:
            root,p,l,protocol,ledger=fixture(d);ledger['active_seconds']=10800.01
            with patch.object(c.subprocess,'Popen',side_effect=AssertionError('child must not dispatch')):
                with self.assertRaisesRegex(ValueError,'[Tt]ime|budget|allowance'):
                    c.run_phase(p,l,root/'ledger.json',ledger,'fit',2020,'endpoint')


if __name__=='__main__':unittest.main()
