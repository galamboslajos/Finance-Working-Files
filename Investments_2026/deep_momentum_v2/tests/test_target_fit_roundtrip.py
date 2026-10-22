"""One deliberately tiny invented500-tree forest, never empirical panels."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from dmv2 import target_comparison_models as models


class SyntheticFitRoundTrip(unittest.TestCase):
    def test_one_synthetic_fit_serializes_reloads_and_preserves_original_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve()
            prepared_dir=root/'folds/2020/prepared'
            generator=np.random.default_rng(1729)
            train=generator.normal(size=(100,4)).astype(np.float32)
            labels=np.tile(np.arange(1,11,dtype=np.int64),10)
            keys=[];records=[]
            for month in range(1,13):
                period=pd.Period(f'2020-{month:02d}',freq='M')
                signal=str((period-1).end_time.date())
                for member in range(3):
                    eligible=member<2
                    # One eligible missing future label must remain predicted.
                    observed=eligible and (month,member)!=(6,0)
                    keys.append((f'INVENTED{member}',signal))
                    records.append(dict(outcome_month=str(period),outcome_start=str(period.start_time.date()),
                        outcome_end=str(period.end_time.date()),prediction_eligible=eligible,
                        paper_prediction_eligible=eligible,target_available=observed,
                        target_class=(1+((month*2+member)%10)) if observed else np.nan,
                        label_available_at=str((period+1).start_time.date()),side='long',target_family='executable_return',
                        path_variant='unstopped',intrinio_security_id=f'INVENTED_NATIVE{member}',head='cash_long'))
            ledger=pd.DataFrame(records,index=pd.MultiIndex.from_tuples(keys,names=models.KEYS))
            selected=ledger.prediction_eligible
            prediction=generator.normal(size=(int(selected.sum()),4)).astype(np.float32)
            artifacts={
                'train_X':models.save_array(prepared_dir/'train_X.npy',train),
                'train_y_cash_long':models.save_array(prepared_dir/'train_y_cash_long.npy',labels),
                'prediction_X':models.save_array(prepared_dir/'prediction_X.npy',prediction),
                'prediction_ledger_cash_long':models.save_frame(prepared_dir/'prediction_ledger_cash_long.parquet',ledger.reset_index())}
            receipt={'status':'PREPARED','year':2020,'plan_sha256':'sha256:invented',
                'artifacts':artifacts,'train_fit_rows':100,'matrix_width':4,
                'preprocessor_fingerprint':'sha256:invented_preprocessor',
                'prediction_keys_sha256':models.canonical([list(k) for k in ledger.index[selected]])}
            models.save(prepared_dir/'receipt.json',receipt)
            plan={'output_directory':str(root),'plan_sha256':'sha256:invented'}
            with patch.object(models,'verify_plan'),patch.object(models,'_prepared',return_value=(prepared_dir,receipt)):
                result=models.fit_head(plan,2020,'cash_long')
            complete=models.read(result['path'])
            head=root/'folds/2020/heads/cash_long'
            fit=models.read(head/'FIT_COMPLETE.json')
            self.assertEqual(fit['estimator_fits'],1)
            self.assertEqual(fit['explicit_parameters']['n_estimators'],500)
            self.assertEqual(fit['fit_effective_parameters']['n_jobs'],4)
            self.assertEqual(fit['saved_inference_effective_parameters']['n_jobs'],1)
            self.assertEqual(fit['native_class_ids'],list(range(1,11)))
            self.assertEqual(complete['estimator_fits'],0)
            forecasts=pd.read_parquet(complete['artifacts']['predictions']['path'])
            self.assertEqual(len(forecasts),36)
            self.assertEqual(forecasts.prediction_eligible.sum(),24)
            self.assertEqual(complete['summary']['prediction_rows'],24)
            self.assertEqual(complete['summary']['labelled_rows'],23)
            self.assertEqual(complete['summary']['outcome_months'],12)
            columns=[f'probability_{k}' for k in range(1,11)]
            self.assertTrue(forecasts.loc[~forecasts.prediction_eligible,columns].isna().all().all())
            self.assertTrue(forecasts.loc[forecasts.prediction_eligible,columns].notna().all().all())
            missing=forecasts.prediction_eligible & ~forecasts.target_available
            self.assertEqual(missing.sum(),1)
            self.assertTrue(forecasts.loc[missing,columns].notna().all().all())
            probability=np.load(complete['artifacts']['probabilities']['path'],allow_pickle=False)
            restored=joblib.load(models.check(fit['model']))
            self.assertEqual(len(restored.estimators_),500)
            np.testing.assert_array_equal(restored.predict_proba(prediction),probability)
            np.testing.assert_array_equal(forecasts.loc[forecasts.prediction_eligible,columns].to_numpy(),probability)
            np.testing.assert_allclose(probability.sum(axis=1),1,rtol=0,atol=1e-14)
            expected_bytes=sum(artifacts[k]['size_bytes'] for k in ('train_X','train_y_cash_long','prediction_X','prediction_ledger_cash_long'))
            self.assertEqual(complete['derived_prepared_read_bytes'],expected_bytes)
            self.assertEqual(len(list(root.glob('folds/*/heads/*/FIT_ATTEMPT.json'))),1)
            with patch.object(models,'verify_plan'),patch.object(models,'_prepared',return_value=(prepared_dir,receipt)):
                with self.assertRaisesRegex(ValueError,'prediction attempt'):
                    models.fit_head(plan,2020,'cash_long')


if __name__=='__main__':unittest.main()
