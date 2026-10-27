import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from dmv2.extension_reporting import read_extension_inputs


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
    return dict(path=str(path),size_bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())


class PassiveCoverageTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        worker=save(self.root/'checker.json',{'test':'structural fixture'})
        prep=save(self.root/'prepared.json',{'no_prices_or_models':True})
        source=save(self.root/'source.json',dict(status='PASS_INPUT_SOURCE_ARRAYS_AND_MONTHLY_FORMULAS',plan_pin={'test_plan':True},prepared_receipt_pin=prep))
        phase=save(self.root/'phase.json',{'result':source})
        save(self.root/'ledger.json',dict(status='COMPLETE_VERIFIED_INPUTS',fits_started=0,plan_pin={'test_plan':True},completed=[dict(key='verify_data',receipt=phase)]))
        rp=save(self.root/'report/read_plan.json',dict(source_proof_pin=source,worker_pin=worker))
        summary=save(self.root/'report/summary.json',dict(status='VERIFIED_INPUT_COVERAGE_ONLY',new_fits=0,new_backtests=0))
        save(self.root/'report/verification.json',dict(status='PASS_AGGREGATE_RECONCILIATION',source_proof_pin=source,read_plan_pin=rp,worker_pin=worker,summary_pin=summary))

    def tearDown(self):self.temp.cleanup()

    def test_verified_small_report_without_source_arrays_or_models(self):
        self.assertEqual(read_extension_inputs(self.root)['new_fits'],0)

    def test_incomplete_audit_is_rejected(self):
        p=self.root/'ledger.json';v=json.loads(p.read_text());v['status']='RUNNING';p.write_text(json.dumps(v))
        with self.assertRaises(ValueError):read_extension_inputs(self.root)

    def test_summary_tampering_is_rejected(self):
        p=self.root/'report/summary.json';p.write_text(p.read_text()+' ')
        with self.assertRaises(ValueError):read_extension_inputs(self.root)

    def test_fit_attempts_are_not_coverage_only(self):
        p=self.root/'ledger.json';v=json.loads(p.read_text());v['fits_started']=1;p.write_text(json.dumps(v))
        with self.assertRaises(ValueError):read_extension_inputs(self.root)
