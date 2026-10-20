"""Handoff preparation is gated by saved receipts and never edits the master."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests.test_walkforward_reporting import fixture,pin,save

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/prepare_walkforward_handoff.py'
spec=importlib.util.spec_from_file_location('prepare_walkforward_handoff',SCRIPT)
handoff=importlib.util.module_from_spec(spec); spec.loader.exec_module(handoff)


def complete_fixture(root,*,withheld=0):
    run,identity,expected=fixture(root,withheld=withheld)
    # Reuse the passive-report schema, adding only the verifier summary fields
    # that a completed handoff gate independently requires.
    model_path=run/'verification/models.json'; model=json.loads(model_path.read_text())
    model['financial_2024_plus_decoded']=False; save(model_path,model)
    import exchange_calendars as xcals
    phase_path=run/'backtest/build_identity.json'; phase=json.loads(phase_path.read_text())
    phase['clock']={'selection_sessions':xcals.get_calendar('XNYS',start='2019-12-01',end='2023-12-31').sessions.strftime('%Y-%m-%d').tolist()}
    save(phase_path,phase); phase_pin=handoff.canonical(phase)
    audit_path=run/'backtest/audit.json'; audit=json.loads(audit_path.read_text())
    audit['phase_identity_sha256']=phase_pin; audit['artifacts']['build_identity.json']=pin(run/'backtest',phase_path); save(audit_path,audit)
    receipt_path=run/'verification/backtest.json'; receipt=json.loads(receipt_path.read_text())
    receipt['backtest_phase_identity_sha256']=phase_pin; receipt['backtest_audit_sha256']=pin(run,audit_path)['sha256']; save(receipt_path,receipt)
    return run,identity


def template(path):
    markdown='## 23. Annual rolling walk-forward — verified saved reports\n'
    code='''from dmv2.walkforward_reporting import read_walkforward_run
walkforward_saved_report = None
WALKFORWARD_RUN = ROOT / "outputs/walkforward/__PENDING_VALIDATED_RUN__"
WALKFORWARD_IDENTITY = "__PENDING_VALIDATED_IDENTITY__"
WALKFORWARD_RECEIPTS = {
    "features": "__PENDING_FEATURE_AUDIT_SHA256__",
    "models": "__PENDING_MODEL_AUDIT_SHA256__",
    "backtest": "__PENDING_BACKTEST_AUDIT_SHA256__",
    "feature_verification": "__PENDING_FEATURE_VERIFICATION_SHA256__",
    "model_verification": "__PENDING_MODEL_VERIFICATION_SHA256__",
    "backtest_verification": "__PENDING_BACKTEST_VERIFICATION_SHA256__",
}
walkforward_saved_report = read_walkforward_run(ROOT, WALKFORWARD_RUN,
    identity_sha256=WALKFORWARD_IDENTITY,
    expected_receipts=WALKFORWARD_RECEIPTS, verify_sources=False)
'''
    cells=[{'cell_type':'markdown','metadata':{},'source':markdown.splitlines(keepends=True)},
        {'cell_type':'code','metadata':{},'source':code.splitlines(keepends=True),'execution_count':None,'outputs':[]}]
    save(path,cells); return path


def notebook(root):
    value={'cells':[{'cell_type':'markdown','id':'old-description','metadata':{'keep':'unchanged'},'source':['# Existing research\n']},
        {'cell_type':'code','id':'old-code','metadata':{},'execution_count':None,'outputs':[],'source':['daily_df = monthly_df = features_df = None\n']}],
        'metadata':{'kernelspec':{'display_name':'Deep Momentum V2 (.venv)','language':'python','name':'dmv2-v2'}},
        'nbformat':4,'nbformat_minor':5}
    path=root/'notebooks/00_research_master.ipynb'; save(path,value); return path,value


class HandoffPreparationTests(unittest.TestCase):
    def test_missing_verification_stops_before_any_saved_audit_read(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve(); run=root/'outputs/walkforward/incomplete'; run.mkdir(parents=True)
            with patch.object(handoff,'json_file',side_effect=AssertionError('Audit opened too early')):
                with self.assertRaisesRegex(AssertionError,'All three completed independent'):
                    handoff.final_bindings(root,run,'sha256:'+'0'*64)

    def test_only_complete_inference_verification_binds_all_six_receipts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve(); run,identity=complete_fixture(root)
            result=handoff.final_bindings(root,run,identity)
            self.assertEqual(len(result['expected_receipts']),6)
            self.assertEqual(result['expected_receipts']['models'],'sha256:'+pin(run,run/'models/audit.json')['sha256'])
            path=run/'verification/models.json'; receipt=json.loads(path.read_text()); receipt['status']='METADATA_PASSED'; save(path,receipt)
            with self.assertRaisesRegex(AssertionError,'completed independent verification'):
                handoff.final_bindings(root,run,identity)

    def test_notebook_candidate_preserves_every_previous_cell_and_installed_bytes(self):
        with tempfile.TemporaryDirectory() as temporary,tempfile.TemporaryDirectory() as other:
            root=Path(temporary).resolve(); run,identity=complete_fixture(root); source,original=notebook(root)
            before=source.read_bytes(); reviewed=template(root/'section.json'); output=Path(other)/'candidate'
            with patch('numpy.load',side_effect=AssertionError('Matrix opened')), \
                    patch('joblib.load',side_effect=AssertionError('Model opened')):
                result=handoff.prepare_notebook(root,run,identity,reviewed,output)
            candidate=json.loads((output/source.name).read_text())
            self.assertEqual(source.read_bytes(),before); self.assertEqual(candidate['cells'][:2],original['cells'])
            self.assertEqual(len(candidate['cells']),4); self.assertEqual(candidate['metadata'],original['metadata'])
            self.assertEqual(candidate['cells'][-1]['outputs'],[]); self.assertIsNone(candidate['cells'][-1]['execution_count'])
            self.assertIn(identity,''.join(candidate['cells'][-1]['source']))
            self.assertEqual(result['original_notebook_sha256'],handoff.digest(source))
            self.assertFalse(result['installed_notebook_changed']); self.assertFalse(result['reports_or_source_prices_decoded'])
            with self.assertRaisesRegex(AssertionError,'existing notebook handoff'):
                handoff.prepare_notebook(root,run,identity,reviewed,output)

    def test_existing_outputs_section_or_numerical_template_refused(self):
        for kind in ('outputs','existing_section','fit_call'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as temporary,tempfile.TemporaryDirectory() as other:
                root=Path(temporary).resolve(); run,identity=complete_fixture(root); source,original=notebook(root)
                reviewed=template(root/'section.json')
                if kind=='outputs':
                    original['cells'][1]['outputs']=[{'output_type':'stream','name':'stdout','text':'private'}]; save(source,original)
                elif kind=='existing_section':
                    original['cells'][0]['source']=['## 23. Existing section\n']; save(source,original)
                else:
                    cells=json.loads(reviewed.read_text()); cells[1]['source'].append('estimator.fit(X,y)\n'); save(reviewed,cells)
                with self.assertRaises(AssertionError):
                    handoff.prepare_notebook(root,run,identity,reviewed,Path(other)/'candidate')

    def test_passive_private_report_preserves_withheld_values_and_all_grids(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve(); run,identity=complete_fixture(root,withheld=32)
            destination=run/'reports/synthetic-handoff'
            with patch('numpy.load',side_effect=AssertionError('Matrix opened')), \
                    patch('joblib.load',side_effect=AssertionError('Model deserialized')):
                result=handoff.prepare_report(root,run,identity,destination)
            self.assertEqual(result['status'],'SAVED_PASSIVE_REPORT_SNAPSHOT'); self.assertEqual(result['models_fitted'],0)
            text=(destination/'RESULTS.md').read_text()
            self.assertIn('32 complete proxies and 32 withheld',text)
            self.assertIn('annual retraining adds no liquidation',text)
            self.assertIn('unavailable',text); self.assertIn('Protected 2024+',text)
            import pandas as pd
            self.assertEqual(len(pd.read_csv(destination/'backtest_comparison.csv')),64)
            self.assertEqual(len(pd.read_csv(destination/'backtest_yearly.csv')),256)
            self.assertEqual(len(pd.read_csv(destination/'monthly_10bps.csv')),768)


if __name__=='__main__':unittest.main()
