import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from dmv2.return_dd_reporting import read_return_dd_audit


def save(p, value):
    p.write_text(json.dumps(value))
    return {'path':str(p),'size_bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}


class SavedReportTests(unittest.TestCase):
    def fixture(self, root):
        plan = save(root/'reviewed_plan.json', {'synthetic':'plan'})
        worker = save(root/'verifier.py', {'synthetic':'verifier'})
        schema = 'dmv2-return-dd-independent-verification/v1'
        cov=[];changes=[]
        for i in range(168):
            for side in ('long','short'):
                cov.append({'side':side,'original_members':10,'prediction_eligible_members':9,
                    'labelled_members':8,'floor_hit_count':1,'negative_return_count':3,
                    'minus_one_ratio_count':2,'largest_exact_tie_count':2,'empty_classes':[10],
                    'rank_status':'available'})
                changes.append({'side':side,'class_comparison_rows':8,'class_changed_rows':2})
        tables={'coverage.json':cov,'class_changes.json':changes,
                'payoff_readiness.json':[{'side':s,'vector_available':False} for _ in range(48) for s in ('long','short')],
                'training_completion.json':[{} for _ in range(8)]}
        pins={name:save(root/name,rows) for name,rows in tables.items()}
        audit={'status':'COMPLETE_PENDING_INDEPENDENT_VERIFICATION','reviewed_plan_pin':plan,
            'observed_through':'2023-12-29','denominator_floor':.01,'months':168,
            'original_rows':152307,'direction_rows':304614,'financial_2024_plus_read':False,
            'models_fitted':False,'portfolio_backtest_run':False,
            'artifacts':{name:{**item,'path':name} for name,item in pins.items()}}
        pins['audit.json']=save(root/'audit.json',audit)
        vp=save(root/'verification_plan.json',{'schema':schema,'worker_pin':worker,'run_directory':str(root),'plan_pin':plan,'artifacts':pins})
        proof={'schema':schema,'worker_pin':worker,'status':'PASS','months_checked':168,'directional_rows_checked':304614,
            'models_fitted':0,'portfolio_backtest_run':False,'financial_2024_plus_read':False,
            'plan_pin':plan,'verification_plan_pin':vp}
        save(root/'independent_verification.json',proof)

    def test_valid_saved_aggregate_only(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();self.fixture(root)
            result=read_return_dd_audit(root)
            self.assertEqual(result['summary'][0]['observed_ratio_labels'],1344)
            self.assertEqual(result['summary'][1]['payoff_vectors_available'],0)

    def test_changed_aggregate_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();self.fixture(root)
            (root/'coverage.json').write_text('[]')
            with self.assertRaises(ValueError):read_return_dd_audit(root)

    def test_unverified_result_not_displayed(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();self.fixture(root)
            p=root/'independent_verification.json';proof=json.loads(p.read_text());proof['status']='PENDING';save(p,proof)
            with self.assertRaises(ValueError):read_return_dd_audit(root)

    def test_changed_plan_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();self.fixture(root)
            (root/'reviewed_plan.json').write_text('{}')
            with self.assertRaises(ValueError):read_return_dd_audit(root)

    def test_changed_verifier_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();self.fixture(root)
            (root/'verifier.py').write_text('{}')
            with self.assertRaises(ValueError):read_return_dd_audit(root)

    def test_oversized_pin_refused_before_hash(self):
        from unittest.mock import patch
        from dmv2.return_dd_reporting import _pin
        with patch('dmv2.return_dd_reporting._sha', side_effect=AssertionError('must not hash')):
            with self.assertRaises(ValueError):_pin({'path':'/nonexistent','size_bytes':17*1024**2,'sha256':'x'})

if __name__=='__main__':unittest.main()
