"""Synthetic lifecycle checks: child only writes fake data, never imports RF/data."""
from __future__ import annotations
import hashlib
import contextlib
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from dmv2 import rf_tuning_control as control

FROZEN = Path(__file__).resolve().parents[1] / 'configs' / 'rf_expanded_tuning_20261020.json'
FAKE_WORKER = r'''
import argparse,hashlib,json,os,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--request');p.add_argument('--receipt');a=p.parse_args()
r=json.loads(Path(a.request).read_text()); out=Path(r['output_dir']);out.mkdir(parents=True)
def save(path,value):
 path.write_text(json.dumps(value,sort_keys=True));return {'path':str(path.resolve()),'size_bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
if os.environ.get('FAKE_SLEEP'):time.sleep(float(os.environ['FAKE_SLEEP']))
if os.environ.get('FAKE_FAIL_ACTION')==r['action']:raise SystemExit(7)
if r['split_id'].endswith('_outer_train') and r['action']!='plan':assert r['candidate_id']=='rf_d12_l020_sqrt'
if r['action']=='plan':
 plan={'projected_source_read_bytes':int(os.environ.get('FAKE_READ_BYTES','0'))}
 planpin=save(out/'plan.json',plan)
 result={'status':'COMPLETE','plan':plan,'plan_path':planpin}
 if os.environ.get('FAKE_PAUSE_DURING_PLAN'):
  save(Path(r['run_root'])/'pause.request.json',{'test':True})
else:
 manifest=save(out/'manifest.json',{'synthetic':True,'action':r['action']})
 key={'prepare':'prepare_manifest','fit':'fit_manifest','predict_outer':'prediction_manifest'}[r['action']]
 result={'status':'COMPLETE','action':r['action'],'outer_year':r['outer_year'],'split_id':r['split_id'],'artifacts':[manifest],key:manifest}
 if 'candidate_id'in r:result['candidate_id']=r['candidate_id']
 if r['action']=='fit':
  vy=int(r['split_id'].split('_')[-1]) if '_validate_' in r['split_id'] else r['outer_year']
  result['monthly_log_losses']=[{'outcome_month':f'{vy}-{m:02d}','log_loss':2.3,'score_key_sha256':str(m)}for m in range(1,13)]
  result['effective_feature_count']=37
Path(a.receipt).parent.mkdir(parents=True,exist_ok=True);save(Path(a.receipt),result)
'''


class ControlTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.protocol=json.loads(FROZEN.read_text())
        self.protocol['reference']={'pins':[]}
        self.protocol['runtime_snapshot_for_future_matching']={}
        self.protocol['local_budget'].update(minimum_free_disk_before_launch_bytes=0, minimum_available_memory_before_launch_bytes=1,
                                            process_tree_rss_warning_bytes=100000,process_tree_rss_stop_bytes=1000000,
                                            monitor_poll_seconds=.01,model_wall_seconds_per_fit=10)
        self.protocol_path=self.root/'protocol.json';control.atomic_json(self.protocol_path,self.protocol)
        amendment={'protocol_id':self.protocol['protocol_id'],'protocol_sha256':control.file_pin(self.protocol_path)['sha256'],
                   'execution_authorized':True,'authority':{name:True for name in ('fitting_authorized','preprocessing_fitting_authorized','new_financial_reads_authorized','prediction_authorized')}}
        amendment['authority']['backtest_authorized']=False
        self.amendment=self.root/'amendment.json';control.atomic_json(self.amendment,amendment)
        self.worker=self.root/'fake_worker.py';self.worker.write_text(FAKE_WORKER)
        self.run=control.prepare_run(self.root,self.protocol_path,self.amendment,code_paths=[self.worker],runtime={'python':'test'},attempt_suffix='test')
        self.kwargs={'worker_prefix':[sys.executable,str(self.worker)],'memory_probe':lambda:{'available_bytes':100},
                     'rss_probe':lambda:100,'check_runtime':False,'poll_seconds':.005}

    def tearDown(self):self.tmp.cleanup()

    def go(self,**kw):return control.run_attempt(self.run,**self.kwargs,**kw)

    def test_full_synthetic_schedule_and_outer_winner_binding(self):
        state=self.go()
        self.assertEqual(state["status"],"COMPLETE_PENDING_INDEPENDENT_VERIFICATION")
        self.assertEqual(state["initiated_estimator_fits"],100)
        self.assertEqual(state["initiated_preprocessor_fits"],16)
        self.assertEqual(len(state["winners"]),4)
        control.verify_attempt(self.run,check_runtime=False)

    def test_failed_fit_consumes_slot_without_retry(self):
        with patch.dict("os.environ",{"FAKE_FAIL_ACTION":"fit"}):
            with self.assertRaisesRegex(control.TuningControlError,"Worker failed"):self.go()
        self.assertEqual(control.status(self.run)["initiated_estimator_fits"],1)
        with self.assertRaisesRegex(control.TuningControlError,"no automatic retry"):self.go(resume=True)

    def test_same_execution_amendment_cannot_reset_fit_counter_in_new_root(self):
        with self.assertRaisesRegex(control.TuningControlError,"already owns an attempt"):
            control.prepare_run(self.root,self.protocol_path,self.amendment,code_paths=[self.worker],runtime={"python":"test"},attempt_suffix="duplicate")

    def test_active_time_includes_parent_work_but_excludes_paused_hours(self):
        real_monotonic=time.monotonic
        offset=[0.]
        with patch.object(control.time,"monotonic",side_effect=lambda:real_monotonic()+offset[0]):
            state=self.go(max_completed_actions=1)
            before=state["active_seconds"]
            offset[0]+=7200.
            state=self.go(resume=True,max_completed_actions=1)
        self.assertLess(state["active_seconds"]-before,10.)
        def parent_memory_probe():
            time.sleep(.03)
            return {"available_bytes":0}
        self.kwargs["memory_probe"]=parent_memory_probe
        before=state["active_seconds"]
        state=self.go(resume=True)
        self.assertGreaterEqual(state["active_seconds"]-before,.03)
        self.assertEqual(state["status"],"WAITING_FOR_MEMORY")

    def test_default_child_imports_pinned_project_package_from_unrelated_cwd(self):
        package=self.root/"src"/"dmv2"
        package.mkdir(parents=True)
        (package/"__init__.py").write_text("")
        (package/"rf_tuning_fit.py").write_text(FAKE_WORKER)
        unrelated=self.root/"unrelated";unrelated.mkdir()
        kwargs=dict(self.kwargs);kwargs.pop("worker_prefix")
        with contextlib.chdir(unrelated),patch.dict("os.environ",{"PYTHONPATH":""}):
            state=control.run_attempt(self.run,max_completed_actions=1,**kwargs)
        self.assertEqual(state["status"],"PAUSED")
        self.assertEqual(state["initiated_preprocessor_fits"],1)
        self.assertEqual(state["initiated_estimator_fits"],0)

    def test_operational_amendment_allows_8_5GiB_only_with_normal_pressure(self):
        original=self.protocol["local_budget"]["minimum_available_memory_before_launch_bytes"]
        self._pressure_amendment()
        self.kwargs["memory_probe"]=lambda:{"available_bytes":int(8.5*1024**3)}
        self.kwargs["pressure_probe"]=lambda:{"dispatch_value":1}
        state=self.go(max_completed_actions=1)
        self.assertEqual(state["initiated_preprocessor_fits"],1)
        self.assertEqual(control.read_json(self.protocol_path)["local_budget"]["minimum_available_memory_before_launch_bytes"],12*1024**3)
        self.assertEqual(control.verify_attempt(self.run,check_runtime=False)[2]["local_budget"]["minimum_available_memory_before_launch_bytes"],7*1024**3)

    def test_warn_pressure_waits_before_any_child_or_fit(self):
        self._pressure_amendment()
        self.kwargs["pressure_probe"]=lambda:{"dispatch_value":2}
        state=self.go()
        self.assertEqual(state["status"],"WAITING_FOR_MEMORY_PRESSURE")
        self.assertEqual(state["initiated_preprocessor_fits"],0)
        self.assertNotIn("plan_path",state["tasks"][0])

    def test_unknown_pressure_fails_closed_without_fit(self):
        self._pressure_amendment()
        self.kwargs["pressure_probe"]=lambda:(_ for _ in ()).throw(OSError("synthetic unreadable pressure"))
        state=self.go()
        self.assertEqual(state["status"],"WAITING_FOR_MEMORY_PRESSURE")
        self.assertIsNone(state["last_preflight"]["macos_memory_pressure"]["dispatch_value"])

    def test_pressure_warning_during_child_stops_worker(self):
        self._pressure_amendment()
        self.kwargs["memory_probe"]=lambda:{"available_bytes":int(8.5*1024**3)}
        readings=iter([1,2])
        self.kwargs["pressure_probe"]=lambda:{"dispatch_value":next(readings,2)}
        with patch.dict("os.environ",{"FAKE_SLEEP":".1"}):
            with self.assertRaisesRegex(control.TuningControlError,"MACOS_MEMORY_PRESSURE_NOT_NORMAL"):self.go()
        self.assertEqual(control.status(self.run)["status"],"STOPPED_BUDGET")

    def test_unreviewed_launch_override_rejected(self):
        policy={"resource_launch_policy":{"minimum_available_memory_before_launch_bytes":6*1024**3,
            "require_macos_pressure_dispatch_value":1,"stop_on_macos_pressure_not_normal":True}}
        with self.assertRaisesRegex(control.TuningControlError,"Unreviewed operational"):control.effective_protocol(self.protocol,policy)

    def _pressure_amendment(self):
        self.protocol["local_budget"]["minimum_available_memory_before_launch_bytes"]=12*1024**3
        control.atomic_json(self.protocol_path,self.protocol)
        amendment=control.read_json(self.amendment)
        amendment["protocol_sha256"]=control.file_pin(self.protocol_path)["sha256"]
        amendment["resource_launch_policy"]={"minimum_available_memory_before_launch_bytes":7*1024**3,
            "require_macos_pressure_dispatch_value":1,"stop_on_macos_pressure_not_normal":True}
        control.atomic_json(self.amendment,amendment)
        self.run=control.prepare_run(self.root,self.protocol_path,self.amendment,code_paths=[self.worker],runtime={"python":"test"},attempt_suffix="pressure")

    def test_fixed_count_and_order(self):
        tasks=control.planned_tasks(self.protocol)
        self.assertEqual(sum(t['action']=='fit'for t in tasks),100)
        self.assertEqual(sum(t['action']=='prepare'for t in tasks),16)
        self.assertEqual(tasks[1]['candidate_id'],'rf_d12_l020_sqrt')

    def test_checkpoint_resume_does_not_refit_completed_state_or_model(self):
        state=self.go(max_completed_actions=2)
        self.assertEqual(state['status'],'PAUSED');self.assertEqual(state['initiated_estimator_fits'],1)
        self.assertEqual(state['initiated_preprocessor_fits'],1)
        first=dict(state['tasks'][1]['receipt_pin'])
        with self.assertRaisesRegex(control.TuningControlError,'explicit resume'):self.go()
        state=self.go(resume=True,max_completed_actions=1)
        self.assertEqual(state['initiated_estimator_fits'],2);self.assertEqual(state['initiated_preprocessor_fits'],1)
        self.assertEqual(first,state['tasks'][1]['receipt_pin'])
        control.verify_attempt(self.run,check_runtime=False)

    def test_low_memory_wait_is_zero_preprocessing_and_zero_fit(self):
        self.kwargs['memory_probe']=lambda:{'available_bytes':0}
        state=self.go()
        self.assertEqual(state['status'],'WAITING_FOR_MEMORY')
        self.assertEqual(state['initiated_estimator_fits'],0);self.assertEqual(state['initiated_preprocessor_fits'],0)
        self.assertIsNotNone(state['tasks'][0].get('plan_path'))
        self.kwargs['memory_probe']=lambda:{'available_bytes':100}
        state=self.go(resume=True,max_completed_actions=1)
        self.assertEqual(state['initiated_preprocessor_fits'],1)

    def test_pause_during_metadata_plan_prevents_financial_launch(self):
        with patch.dict('os.environ',{'FAKE_PAUSE_DURING_PLAN':'1'}):state=self.go()
        self.assertEqual(state['status'],'PAUSED');self.assertEqual(state['initiated_preprocessor_fits'],0)

    def test_completed_artifact_tamper_refuses_resume(self):
        state=self.go(max_completed_actions=1)
        receipt=control.read_json(state['tasks'][0]['receipt_pin']['path'])
        Path(receipt['prepare_manifest']['path']).write_text('tampered')
        with self.assertRaisesRegex(control.TuningControlError,'Hash/size mismatch'):self.go(resume=True)

    def test_code_tamper_refuses_resume(self):
        self.go(max_completed_actions=1);self.worker.write_text('tampered')
        with self.assertRaisesRegex(control.TuningControlError,'Hash/size mismatch'):self.go(resume=True)

    def test_incomplete_initiated_fit_refuses_retry(self):
        state=control.read_json(self.run/'ledger.json')
        state['tasks'][1].update(status='RUNNING',initiated=True);state['initiated_estimator_fits']=1
        control.commit_state(self.run,state,'SYNTHETIC_CRASH')
        with self.assertRaisesRegex(control.TuningControlError,'Incomplete initiated'):self.go(resume=True)
        self.assertEqual(control.status(self.run)['status'],'FAILED_INTERRUPTED')
        with self.assertRaisesRegex(control.TuningControlError,'no automatic retry'):self.go(resume=True)

    def test_source_budget_is_reserved_before_prepare(self):
        self.protocol['local_budget']['total_projected_compressed_source_read_stop_bytes']=1
        self._rebuild()
        with patch.dict('os.environ',{'FAKE_READ_BYTES':'2'}):
            with self.assertRaisesRegex(control.TuningControlError,'source read budget'):self.go()
        state=control.status(self.run)
        self.assertEqual(state['status'],'STOPPED_BUDGET');self.assertEqual(state['initiated_preprocessor_fits'],0)

    def test_watchdog_stops_child_on_current_rss(self):
        self.kwargs['rss_probe']=lambda:2000000
        with patch.dict('os.environ',{'FAKE_SLEEP':'.1'}):
            with self.assertRaises(control.TuningControlError):self.go()
        self.assertEqual(control.status(self.run)['status'],'STOPPED_BUDGET')

    def test_watchdog_stops_child_on_wall_timeout(self):
        self.protocol['local_budget']['model_wall_seconds_per_fit']=.03;self._rebuild()
        with patch.dict('os.environ',{'FAKE_SLEEP':'.1'}):
            with self.assertRaises(control.TuningControlError):self.go()
        self.assertEqual(control.status(self.run)['status'],'STOPPED_BUDGET')

    def test_event_chain_tamper_refused(self):
        path=self.run/'events'/'000001.json';event=control.read_json(path);event['ledger_snapshot']['initiated_estimator_fits']=9
        control.atomic_json(path,event)
        with self.assertRaisesRegex(control.TuningControlError,'snapshot changed'):self.go()

    def _rebuild(self):
        control.atomic_json(self.protocol_path,self.protocol)
        amendment=control.read_json(self.amendment);amendment['protocol_sha256']=control.file_pin(self.protocol_path)['sha256'];control.atomic_json(self.amendment,amendment)
        self.run=control.prepare_run(self.root,self.protocol_path,self.amendment,code_paths=[self.worker],runtime={'python':'test'},attempt_suffix='rebuild')


class SelectionTests(unittest.TestCase):
    def setUp(self):self.protocol=json.loads(FROZEN.read_text())
    def receipts(self):
        return [{'split_id':split['split_id'],'candidate_id':candidate['candidate_id'],'monthly_log_losses':[
          {'outcome_month':f"{split['validation_outcome_start'][:4]}-{m:02d}",'log_loss':2.,'score_key_sha256':f'keys{m}'}for m in range(1,13)],'effective_feature_count':37}
          for split in self.protocol['chronology']['folds'][0]['inner_splits'] for candidate in self.protocol['rf']['candidates']]
    def test_reference_wins_numerical_tie(self):
        result=control.select_winner(self.protocol,2020,self.receipts())
        self.assertEqual(result['candidate_id'],'rf_d12_l020_sqrt')
    def test_incomplete_grid_refused(self):
        with self.assertRaisesRegex(control.TuningControlError,'incomplete candidate grid'):control.select_winner(self.protocol,2020,self.receipts()[:-1])
    def test_one_missing_month_refused(self):
        receipts=self.receipts();receipts[0]['monthly_log_losses'].pop()
        with self.assertRaisesRegex(control.TuningControlError,'twelve'):control.select_winner(self.protocol,2020,receipts)
    def test_different_scoring_keys_refused(self):
        receipts=self.receipts();receipts[-1]['monthly_log_losses'][0]['score_key_sha256']='different'
        with self.assertRaisesRegex(control.TuningControlError,'different observed'):control.select_winner(self.protocol,2020,receipts)
    def test_equal_month_mean_and_regularization_tie(self):
        receipts=self.receipts()
        for r in receipts:
            if r['candidate_id'] in ('rf_d08_l100_sqrt','rf_d08_l020_sqrt'):
                for value in r['monthly_log_losses']:value['log_loss']=1.8
        result=control.select_winner(self.protocol,2020,receipts)
        self.assertEqual(result['candidate_id'],'rf_d08_l100_sqrt');self.assertAlmostEqual(result['candidate_scores']['rf_d08_l100_sqrt'],1.8)

if __name__=='__main__':unittest.main()
