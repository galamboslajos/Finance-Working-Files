import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from dmv2.extension_model_reporting import read_extension_models


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
    return dict(path=str(path),size_bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())


class ExtensionModelReportingTests(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.r=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()
    def ledger(self,status='RUNNING'):
        return dict(status=status,phase='fit-5-2024-rf',fits_started=1,fits_completed=0,active_seconds=3,peak_worker_rss_bytes=123,completed=[])
    def fixture(self):
        plan=save(self.r/'plan.json',{});worker=save(self.r/'worker.json',{});vp=save(self.r/'readplan.json',{});prep=save(self.r/'prepared.json',{})
        data=save(self.r/'data.json',dict(status='PASS',plan_pin=plan,prepared_receipt_pin=prep,worker_pin=worker,verification_read_plan_pin=vp))
        model=save(self.r/'models.json',dict(status='PASS',plan_pin=plan,data_proof_pin=data,verified_fits=12,worker_pin=worker,verification_read_plan_pin=vp))
        summary=save(self.r/'summary.json',dict(status='VERIFIED_MODELS_AND_RET_SELECTIONS_ONLY',completed_fits=12,new_backtests=0))
        audit=save(self.r/'audit.json',{})
        selection=save(self.r/'selection.json',dict(status='PASS',plan_pin=plan,model_proof_pin=model,summary_pin=summary,audit_pin=audit,worker_pin=worker,verification_read_plan_pin=vp))
        ledger=self.ledger('COMPLETE_VERIFIED_MODELS_AND_SELECTION');ledger.update(phase=None,fits_started=12,fits_completed=12,plan_pin=plan)
        for key,value in [('verify_data',data),('verify_models',model),('verify_selection',selection)]:
            pp=save(self.r/(key+'_phase.json'),dict(result=value));ledger['completed'].append(dict(key=key,receipt=pp))
        save(self.r/'ledger.json',ledger)
    def test_running_reads_progress_without_numeric_reports(self):
        save(self.r/'ledger.json',self.ledger());result=read_extension_models(self.r)
        self.assertEqual(result['fits_completed'],0);self.assertNotIn('metrics',result)
    def test_complete_requires_independent_chain(self):
        self.fixture();self.assertEqual(read_extension_models(self.r)['completed_fits'],12)
    def test_summary_tampering_rejected(self):
        self.fixture();p=self.r/'summary.json';p.write_text(p.read_text()+' ')
        with self.assertRaises(ValueError):read_extension_models(self.r)
    def test_false_completion_rejected(self):
        d=self.ledger('COMPLETE_VERIFIED_MODELS_AND_SELECTION');save(self.r/'ledger.json',d)
        with self.assertRaises(ValueError):read_extension_models(self.r)
    def test_excess_fit_count_rejected(self):
        d=self.ledger();d.update(fits_started=13);save(self.r/'ledger.json',d)
        with self.assertRaises(ValueError):read_extension_models(self.r)
