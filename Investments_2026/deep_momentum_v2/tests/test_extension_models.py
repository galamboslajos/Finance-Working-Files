"""Structural scope/leakage tests; no financial panels or estimator fitting."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from dmv2.extension_models import masks, matrices, FEATURES, array_sha
from dmv2.extension_model_metrics import evaluate_probabilities
from dmv2.walkforward_metrics import evaluate_probabilities as old_metrics


class ExtensionModelsTests(unittest.TestCase):
    def frame(self):
        rows=[]
        for month,available in [('2014-01','2014-01-31'),('2018-12','2018-12-31'),('2019-01','2019-01-31'),
            ('2023-12','2023-12-29'),('2023-12','2024-01-02'),('2024-01','2024-01-31'),
            ('2025-12','2025-12-31'),('2026-04','2026-04-30'),('2026-05','2026-05-29')]:
            rows.append(dict(source_security_id=str(len(rows)),signal_date='unused',outcome_month=month,
                label_available_at=available,prediction_eligible=True,target_available=True,target_class=1))
        return pd.DataFrame(rows)

    def test_completed_outcomes_and_matching_training_bounds(self):
        f=self.frame();train,_,_=masks(f,5,2024)
        self.assertEqual(np.flatnonzero(train).tolist(),[2,3])
        ten,_,_=masks(f,10,2024);self.assertEqual(np.flatnonzero(ten).tolist(),[0,1,2,3])

    def test_forward_target_state_cannot_screen_prediction(self):
        f=self.frame();before=masks(f,5,2024)
        f.loc[f.outcome_month.ge('2024-01'),'target_available']=False
        f.loc[f.outcome_month.ge('2024-01'),'target_class']=np.nan
        after=masks(f,5,2024)
        for a,b in zip(before,after):np.testing.assert_array_equal(a,b)
        self.assertTrue(after[2][5])

    def test_april2026_terminal_scope(self):
        f=self.frame();_,test,qualified=masks(f,5,2026)
        self.assertEqual(np.flatnonzero(test).tolist(),[7]);self.assertTrue(qualified[7])
        with self.assertRaises(ValueError):masks(f,5,2027)
        with self.assertRaises(ValueError):masks(f,7,2026)

    def test_exact15_matrix_no_forward_column_and_no_imputation(self):
        rows=[]
        for k in range(1,11):rows.append(dict(source_security_id=str(k),signal_date='2018-12-31',outcome_month='2019-01',
            label_available_at='2019-01-31',prediction_eligible=True,target_available=True,target_class=k,**{n:k for n in FEATURES}))
        rows.append(dict(source_security_id='new',signal_date='2023-12-29',outcome_month='2024-01',label_available_at='2024-01-31',
            prediction_eligible=True,target_available=False,target_class=np.nan,**{n:2 for n in FEATURES}))
        f=pd.DataFrame(rows);f['future_return']=1e99
        _,_,_,X,y,XP=matrices(f,5,2024);self.assertEqual(X.shape,(10,15));self.assertEqual(XP.shape,(1,15))
        self.assertEqual(set(y),set(range(1,11)));self.assertEqual(XP[0,0],2)
        f.loc[len(f)-1,FEATURES[0]]=np.nan
        with self.assertRaises(ValueError):matrices(f,5,2024)

    def test_scope_is_twelve_fits_and_two_independent_selection_phases(self):
        path=Path(__file__).resolve().parents[1]/'scripts/run_extension_models.py'
        spec=importlib.util.spec_from_file_location('_scope',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        tasks=module.schedule();fits=[r for r in tasks if r[0]=='fit']
        self.assertEqual(len(fits),12);self.assertEqual({r[2] for r in fits},{2024,2025,2026})
        self.assertEqual(tasks[-2][0],'selection');self.assertEqual(tasks[-1][0],'verify_selection')
        self.assertFalse(any('backtest' in r[0] for r in tasks))

    def test_new_metric_scope_preserves_original_guard(self):
        meta=pd.DataFrame([dict(source_security_id='test',signal_date='2023-12-29',outcome_month='2024-01',outcome_end='2024-01-31')])
        values=np.full((1,10),.1)
        result=evaluate_probabilities(values,[1.],class_ids=list(range(1,11)),row_metadata=meta,first_month='2024-01',last_month='2024-01')
        self.assertAlmostEqual(result.summary['pooled_log_loss'],np.log(10))
        with self.assertRaises(ValueError):old_metrics(values,[1.],class_ids=list(range(1,11)),row_metadata=meta,first_month='2024-01',last_month='2024-01')
        with self.assertRaises(ValueError):evaluate_probabilities(values,[1.],class_ids=list(range(1,11)),row_metadata=meta,first_month='2024-01',last_month='2026-05')

    def test_matrix_fingerprint_includes_float32_values_and_order(self):
        a=np.array([[1,2],[3,4]],np.float32)
        self.assertNotEqual(array_sha(a),array_sha(a[::-1].copy()))


if __name__=='__main__':unittest.main()
