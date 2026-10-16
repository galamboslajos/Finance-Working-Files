"""Independent verifier rejects leakage/tampering using synthetic data only."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/verify_walkforward_models.py'
spec = importlib.util.spec_from_file_location('independent_walkforward_verifier', SCRIPT)
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


def preprocessing_fixture(root):
    root.mkdir(parents=True,exist_ok=True)
    names = [*verifier.PAPER15,'missing_varying','all_missing','constant']
    keys = [(f's{i}','2019-11-29') for i in range(4)]
    index = pd.MultiIndex.from_tuples(keys,names=verifier.KEYS)
    train = pd.DataFrame({'source_security_id':[k[0] for k in keys],'signal_date':[k[1] for k in keys],
        'fit_mask':True})
    X = pd.DataFrame({name:np.arange(1.,5.) for name in names},index=index)
    X['missing_varying']=[1.,np.nan,5.,np.nan]; X['all_missing']=np.nan; X['constant']=2.
    test = X.iloc[:2].copy(); test.iloc[0,0]=np.nan
    for bank,columns in (('paper15',verifier.PAPER15),('expanded',names)):
        directory=root/'preprocessors'/bank; directory.mkdir(parents=True)
        retained=[name for name in columns if name not in {'all_missing','constant'}]
        dropped={name:reason for name,reason in [('all_missing','all_missing_in_training_partition'),
            ('constant','constant_finite_values_in_training_partition')] if name in columns}
        medians={name:float(np.nanmedian(X[name])) for name in retained}
        manifest={'input_columns':columns,'retained_columns':retained,'dropped_columns':dropped,
            'output_columns':[part for name in retained for part in (name,name+'__missing')],
            'medians':medians,'train_start':'2010-01-01','train_end':'2019-12-31',
            'label_observed_through':'2019-12-31','fit_rows':4,
            'fit_keys_sha256':verifier.canonical(train[verifier.KEYS].values.tolist()),
            'baseline_required_for_population':verifier.PAPER15,'training_only_schema_selection':True,
            'scaling':None,'clipping':None,'winsorization':None}
        (directory/'manifest.json').write_text(json.dumps({'manifest':manifest,'fingerprint':verifier.canonical(manifest)}))
        for filename,source in (('train_X.npy',X),('test_X.npy',test)):
            matrix=np.empty((len(source),2*len(retained)),dtype=np.float32)
            for offset,name in enumerate(retained):
                missing=source[name].isna().to_numpy()
                matrix[:,2*offset]=np.where(missing,medians[name],source[name])
                matrix[:,2*offset+1]=missing
            np.save(directory/filename,matrix,allow_pickle=False)
    return train,test,X,names


class IndependentVerifierTests(unittest.TestCase):
    def test_direct_training_statistics_and_matrix_flags(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); train,test,X,names=preprocessing_fixture(root)
            result=verifier.verify_preprocessing(root,train,test,X,test,2020,expanded_names=names)
            self.assertEqual(result['paper15']['retained_columns'],15)
            self.assertEqual(result['expanded']['dropped_columns'],2)
            self.assertEqual(result['expanded']['retained_columns'],16)

    def test_new_test_value_cannot_change_training_median(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); train,test,X,names=preprocessing_fixture(root)
            path=root/'preprocessors/expanded/manifest.json'
            state=json.loads(path.read_text()); state['manifest']['medians']['missing_varying']=999.
            state['fingerprint']=verifier.canonical(state['manifest']); path.write_text(json.dumps(state))
            with self.assertRaisesRegex(AssertionError,'training-only median'):
                verifier.verify_preprocessing(root,train,test,X,test,2020,expanded_names=names)

    def test_missing_flag_or_schema_tamper_rejected(self):
        for kind in ('flag','constant','future_cutoff'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary); train,test,X,names=preprocessing_fixture(root)
                path=root/'preprocessors/expanded/manifest.json'
                if kind=='flag':
                    matrix=np.load(root/'preprocessors/expanded/test_X.npy'); matrix[0,1]=0.
                    np.save(root/'preprocessors/expanded/test_X.npy',matrix)
                else:
                    state=json.loads(path.read_text())
                    if kind=='constant':state['manifest']['dropped_columns'].pop('constant')
                    else:state['manifest']['label_observed_through']='2020-12-31'
                    state['fingerprint']=verifier.canonical(state['manifest']); path.write_text(json.dumps(state))
                with self.assertRaises(AssertionError):
                    verifier.verify_preprocessing(root,train,test,X,test,2020,expanded_names=names)

    def test_future_window_refused_before_any_parquet_read(self):
        with patch.object(verifier.pq,'ParquetFile',side_effect=AssertionError('decoded before window guard')):
            with self.assertRaisesRegex(AssertionError,'financial bounds'):
                verifier.read_table(Path('/nonexistent'),first='2024-01',last='2024-12')

    def test_resource_budget_and_no_fit_authority(self):
        execution={'verification_maximum_wall_seconds':1800,'verification_maximum_peak_rss_bytes':8*1024**3,
            'verification_model_fits':0}
        verifier.VerificationBudget(execution).check()
        for key,value in [('verification_model_fits',1),('verification_maximum_wall_seconds',1801),
                ('verification_maximum_peak_rss_bytes',9*1024**3)]:
            with self.subTest(key=key),self.assertRaises(AssertionError):
                verifier.VerificationBudget({**execution,key:value})

    def test_cli_receipt_writes_once_in_separate_verification_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve(); run=root/'outputs/walkforward/synthetic'
            output=run/'verification/models.json'
            result={'status':'PASSED','artifact_files_verified':0,'preserved_source_files':0,
                'models_fitted_by_verifier':0,'financial_values_decoded':False}
            args=['--root',str(root),'--run',str(run),'--output',str(output)]
            with patch.object(verifier,'verify_walkforward_models',return_value=result):
                verifier.main(args)
                self.assertTrue(output.is_file())
                with self.assertRaisesRegex(AssertionError,'existing verification'):
                    verifier.main(args)
                with self.assertRaisesRegex(AssertionError,'separate verification'):
                    verifier.main(['--root',str(root),'--run',str(run),'--output',str(run/'models/audit.json')])

    def test_full_saved_synthetic16_fit_workflow_without_verifier_fits(self):
        # Fake training creates a complete source/model contract. The verifier
        # itself only reloads these fake saved estimators for inference.
        from test_walkforward_models import setup,FakeEstimator
        import dmv2.walkforward_models as worker
        import dmv2.walkforward_loading as loader
        from dmv2.full_panel import _artifact,_json,_specs
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve(); run,cfg,old_pin,references=setup(root)
            cfg['execution'].update(verification_maximum_wall_seconds=1800,
                verification_maximum_peak_rss_bytes=8*1024**3,verification_model_fits=0)
            parent=root/'outputs/historical'; parent.mkdir()
            parent_identity={'synthetic_historical_parent':True}; parent_pin=verifier.canonical(parent_identity)
            _json(parent/'build_identity.json',parent_identity); parent_years={}
            for year in range(2010,2020):
                old=run/'years'/str(year); new=parent/'years'/str(year); new.parent.mkdir(exist_ok=True)
                shutil.move(str(old),new)
                audit=references[year]['audit']; audit['build_identity_sha256']=parent_pin
                _json(new/'audit.json',audit); parent_years[str(year)]=_artifact(parent,new/'audit.json')
                references[year].update(directory=new,audit=audit,build_identity_sha256=parent_pin,
                    audit_pin=_artifact(root,new/'audit.json'))
            _json(parent/'audit.json',{'status':'COMPLETE','build_identity_sha256':parent_pin,'years':parent_years})
            entries={}
            for year in range(2010,2020):
                entries[str(year)]={'directory':str(references[year]['directory'].relative_to(root)),'directory_scope':'root',
                    'build_identity_sha256':parent_pin,'audit_pin':references[year]['audit_pin'],
                    'parent_identity_pin':_artifact(root,parent/'build_identity.json'),
                    'parent_audit_pin':_artifact(root,parent/'audit.json')}
            entries.update({str(year):{'directory':f'years/{year}','directory_scope':'run','parent_reference':'THIS_BUILD'} for year in verifier.YEARS})
            identity={'phase':'synthetic_root','configuration':cfg,'annual_sources':entries}; pin=verifier.canonical(identity)
            _json(run/'build_identity.json',identity); _json(root/'configs/walkforward_2020_2023.json',cfg)
            root_years={}
            for year in verifier.YEARS:
                directory=references[year]['directory']; audit=references[year]['audit']; audit['build_identity_sha256']=pin
                _json(directory/'audit.json',audit); root_years[str(year)]=_artifact(run,directory/'audit.json')
                references[year].update(audit=audit,build_identity_sha256=pin,audit_pin=_artifact(root,directory/'audit.json'))
            _json(run/'audit.json',{'status':'COMPLETE','build_identity_sha256':pin,'years':root_years})
            _json(run/'source_preservation_manifest.json',{'artifacts':[]}); _json(run/'feature_dictionary.json',_specs())
            source=Path(worker.__file__).parent; (root/'src/dmv2').mkdir(parents=True)
            for path in source.glob('*.py'):shutil.copy2(path,root/'src/dmv2'/path.name)
            FakeEstimator.fit_calls=0
            class FakeXgb(FakeEstimator):
                def set_params(self,**kwargs):return self
            def load_model(kind,path):
                if kind=='rf':
                    import joblib
                    return joblib.load(path)
                estimator=FakeXgb(); estimator.load_model(path); return estimator
            with patch.object(loader,'resolve_walkforward_year',side_effect=lambda root,run,identity,year:references[year]), \
                    patch.object(worker,'_constructor',return_value=FakeEstimator), \
                    patch.object(worker,'_load_estimator',side_effect=load_model):
                for year in worker.YEARS:worker.run_walkforward_fold(root,run,cfg,pin,year)
                worker.finalize_walkforward_models(root,run,cfg,pin)
            calls=FakeEstimator.fit_calls
            with patch('xgboost.XGBClassifier',FakeXgb),patch.object(FakeEstimator,'fit',side_effect=AssertionError('Verifier called fit')):
                result=verifier.verify_walkforward_models(root,run)
            self.assertEqual(calls,16); self.assertEqual(FakeEstimator.fit_calls,calls)
            self.assertEqual(result['status'],'PASSED'); self.assertEqual(result['annual_fits_reconciled'],16)
            self.assertEqual(result['stitched_prediction_months'],48)


if __name__=='__main__':unittest.main()
