import json
from pathlib import Path
import unittest
import test_extension_model_reporting as fixtures
from dmv2.extension_model_reporting import read_extension_models

class RecoveredReportingTests(unittest.TestCase):
    def setUp(self):
        self.base=fixtures.ExtensionModelReportingTests();self.base.setUp();self.base.fixture();self.r=self.base.r
        ledger=json.loads((self.r/'ledger.json').read_text());plan=ledger['plan_pin']
        modelphase=next(x['receipt'] for x in ledger['completed'] if x['key']=='verify_models')
        mp=json.loads(Path(modelphase['path']).read_text())['result']
        original=next(x['receipt'] for x in ledger['completed'] if x['key']=='verify_selection')
        oldsp=json.loads(Path(original['path']).read_text())['result'];old=json.loads(Path(oldsp['path']).read_text())
        ledger.update(status='FAILED_PRESERVED',phase='selection');ledger['completed']=[x for x in ledger['completed'] if x['key']!='verify_selection']
        lp=fixtures.save(self.r/'ledger.json',ledger)
        rp=fixtures.save(self.r/'recovery/plan.json',dict(max_fits=0,max_backtests=0,parent_plan_pin=plan,model_proof_pin=mp))
        proof=dict(old,plan_pin=rp)
        sp=fixtures.save(self.r/'recovery/selection.json',proof)
        phase=fixtures.save(self.r/'recovery/phase.json',dict(result=sp))
        rl=fixtures.save(self.r/'recovery/ledger.json',dict(status='COMPLETE_VERIFIED_SELECTION_RECOVERY',phase=None,
            fits_started=0,fits_completed=0,plan_pin=rp,active_seconds=10,peak_worker_rss_bytes=100,
            completed=[dict(key='verify_selection',receipt=phase)]))
        fixtures.save(self.r/'verification_recovery/selection_handoff.json',dict(status='PASS_INHERITED_12_MODELS_AND_RECOVERED_SELECTIONS',
            new_fits=0,new_backtests=0,parent_plan_pin=plan,parent_ledger_pin=lp,inherited_model_proof_pin=mp,
            recovery_plan_pin=rp,recovery_ledger_pin=rl,selection_proof_pin=sp,summary_pin=old['summary_pin']))
    def tearDown(self):self.base.tearDown()
    def test_verified_zero_fit_recovery_preserves_original_status(self):
        d=read_extension_models(self.r);self.assertEqual(d['original_status'],'FAILED_PRESERVED')
        self.assertEqual(d['status'],'COMPLETE_VERIFIED_MODELS_AND_SELECTION');self.assertTrue(d['zero_fit_selection_recovery'])
    def test_changed_original_ledger_is_rejected(self):
        p=self.r/'ledger.json';p.write_text(p.read_text()+' ')
        with self.assertRaises(ValueError):read_extension_models(self.r)
    def test_recovery_cannot_hide_new_fits(self):
        p=self.r/'verification_recovery/selection_handoff.json';d=json.loads(p.read_text());d['new_fits']=1;p.write_text(json.dumps(d))
        with self.assertRaises(ValueError):read_extension_models(self.r)
