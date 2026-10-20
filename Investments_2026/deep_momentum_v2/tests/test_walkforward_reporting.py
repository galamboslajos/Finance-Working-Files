"""Passive small-report fixtures; models, matrices and prices are never read."""
import hashlib
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from dmv2.catalog import canonical_sha256
from dmv2.universe import DataContractError
from dmv2.walkforward_reporting import read_walkforward_run,ARMS,REFERENCES,BASKETS,VARIANTS


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')


def pin(base,path):
    return {'path':str(path.relative_to(base)),'size_bytes':path.stat().st_size,
        'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def table(base,relative,frame,artifacts,key=None):
    path=base/relative; path.parent.mkdir(parents=True,exist_ok=True); frame.to_parquet(path,index=False)
    artifacts[relative if key is None else key]=pin(base,path)


def fixture(root,*,withheld=0):
    cfg={'protocol_id':'dmv2-walkforward-2020-2023-v1','test_years':[2020,2021,2022,2023],
        'execution':{'maximum_new_model_fits':16,'maximum_new_preprocessor_fits':8,'financial_2024_plus_analysis':False}}
    identity={'phase':'walkforward_features_and_targets_2020_2023_v1','configuration':cfg,'workers':{}}
    identity_pin=canonical_sha256(identity)
    run=root/'outputs/walkforward'/('2020-2023-'+identity_pin[7:23]); run.mkdir(parents=True)
    save(run/'build_identity.json',identity); annuals={}
    for year in range(2020,2024):
        path=run/f'years/{year}/audit.json'
        save(path,{'status':'COMPLETE','year':year,'build_identity_sha256':identity_pin,'summary':{'original_rows':100},'artifacts':{}})
        annuals[str(year)]=pin(run,path)
    save(run/'audit.json',{'status':'COMPLETE','phase':'walkforward_features_and_targets',
        'build_identity_sha256':identity_pin,'years':annuals})
    save(run/'source_preservation_manifest.json',{})
    models=run/'models'; model_phase={'parent_identity_sha256':identity_pin,'configuration_sha256':canonical_sha256(cfg),
        'workers':{},'annual_source_audit_pins':{}}
    save(models/'build_identity.json',model_phase); model_pin=canonical_sha256(model_phase)
    model_artifacts={'build_identity.json':pin(models,models/'build_identity.json')}; model_arms={}
    model_comparison=pd.DataFrame({'arm_id':[*ARMS,*REFERENCES],'mean_monthly_accuracy':.1,
        'outcome_months':48,'first_outcome_month':'2020-01','last_outcome_month':'2023-12'})
    table(models,'reports/comparison.parquet',model_comparison,model_artifacts,'comparison')
    months=[str(p) for p in pd.period_range('2020-01','2023-12',freq='M')]
    for name in (*ARMS,*REFERENCES):
        directory=models/('arms' if name in ARMS else 'refs')/name
        artifacts={}; mode=pd.DataFrame([{'period':period,'class_id':k,'precision':.1,'recall':.1}
            for period,k in itertools.product(['all',*months],range(1,11))])
        table(directory,'reports/mode_class_performance.parquet',mode,artifacts,'mode_class_performance')
        save(directory/'audit.json',{'status':'COMPLETE','build_identity_sha256':identity_pin,
            'model_identity_sha256':model_pin,'metadata':{'new_fits_during_stitching':0},'artifacts':artifacts})
        model_arms[name]=pin(models,directory/'audit.json')
    folds={}
    for year in range(2020,2024):
        path=models/f'folds/{year}/audit.json'; save(path,{'artifacts':{},'arms':{}})
        save(path.parent/'read_plan.json',{'years':{}}); folds[str(year)]=pin(models,path)
    save(models/'audit.json',{'status':'COMPLETE','phase':'walkforward_models_v1','build_identity_sha256':identity_pin,
        'model_identity_sha256':model_pin,'summary':{'fitted_arms':16,'preprocessor_fits':8,'months':48,'outer_years':[2020,2021,2022,2023]},
        'artifacts':model_artifacts,'arms':model_arms,'folds':folds})
    backtest=run/'backtest'; bt_phase={'parent_identity_sha256':identity_pin,'configuration_sha256':canonical_sha256(cfg),'workers':{},'source_pins':[]}
    save(backtest/'build_identity.json',bt_phase); bt_pin=canonical_sha256(bt_phase)
    artifacts={'build_identity.json':pin(backtest,backtest/'build_identity.json')}
    records=[dict(zip(('arm','basket','variant','cost_bps'),item)) for item in itertools.product(ARMS,BASKETS,VARIANTS,[0,5,10,25])]
    comparison=pd.DataFrame(records); comparison['expected_sessions']=1006
    comparison['complete_proxy_path']=[i>=withheld for i in range(64)]
    comparison['total_return']=[None if i<withheld else .1 for i in range(64)]
    table(backtest,'comparison.parquet',comparison,artifacts)
    table(backtest,'yearly.parquet',pd.DataFrame([{**row,'year':year} for row in records for year in range(2020,2024)]),artifacts)
    coverage=pd.DataFrame([{'arm':arm,'basket':basket,'signal_date':(pd.Period(month,'M')-1).end_time.strftime('%Y-%m-%d'),
        'long_candidates':2,'short_candidates':4,'selected_names_per_side':2,'long_selected':2,'short_selected':2,
        'effective_per_name_abs_weight':.25} for arm,basket,month in itertools.product(ARMS,BASKETS,months)])
    table(backtest,'selection_coverage.parquet',coverage,artifacts)
    for arm,basket,variant in itertools.product(ARMS,BASKETS,VARIANTS):
        table(backtest,f'arms/{arm}/{basket}/{variant}/cost_10bps/monthly.parquet',
            pd.DataFrame({'outcome_month':months,'nav':1.,'return':0.}),artifacts)
    save(backtest/'audit.json',{'status':'COMPLETE','phase':'walkforward_continuous_backtest_v1','build_identity_sha256':identity_pin,
        'phase_identity_sha256':bt_pin,'scenarios':64,'sessions':1006,'outcome_months':48,'complete_proxy_paths':64-withheld,
        'failed_closed_paths':withheld,'models_fitted':False,'preprocessing_fitted':False,'annual_liquidation':False,
        'financial_2024_plus_read':False,'artifacts':artifacts})
    raw={label:pin(run,run/relative)['sha256'] for label,relative in (('features','audit.json'),('models','models/audit.json'),('backtest','backtest/audit.json'))}
    receipts={'features':{'status':'PASS','build_identity_sha256':identity_pin,'root_audit_sha256':raw['features'],
            'models_fitted':0,'financial_2024_plus_read':False},
        'models':{'status':'PASSED','build_identity_sha256':identity_pin,'root_audit_sha256':raw['features'],
            'model_identity_sha256':model_pin,'models_audit_sha256':raw['models'],'models_fitted_by_verifier':0,
            'preprocessors_fitted_by_verifier':0,'annual_fits_reconciled':16,'stitched_prediction_months':48,
            'financial_2024_plus_decoded':False},
        'backtest':{'status':'PASS','build_identity_sha256':identity_pin,'root_audit_sha256':raw['features'],
            'backtest_phase_identity_sha256':bt_pin,'backtest_audit_sha256':raw['backtest'],
            'backtest_directory':str(backtest.relative_to(root)),'models_fitted':0,'preprocessors_fitted':0,
            'source_pins_unchanged':True,'financial_2024_plus_read':False,'sessions':1006,'outcome_months':48,'scenarios':64}}
    for label,receipt in receipts.items():save(run/f'verification/{label}.json',receipt)
    expected={key:'sha256:'+value for key,value in raw.items()}
    for key,label in (('feature_verification','features'),('model_verification','models'),('backtest_verification','backtest')):
        expected[key]='sha256:'+pin(run,run/f'verification/{label}.json')['sha256']
    return run,identity_pin,expected


class PassiveWalkforwardReportsTests(unittest.TestCase):
    def test_missing_saved_run_never_starts_numerical_work(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); identity='sha256:'+'a'*64; expected={key:identity for key in
                ('features','models','backtest','feature_verification','model_verification','backtest_verification')}
            with patch('numpy.load',side_effect=AssertionError('Matrix opened')),self.assertRaises(FileNotFoundError):
                read_walkforward_run(root,root/'outputs/walkforward'/('2020-2023-'+'a'*16),
                    identity_sha256=identity,expected_receipts=expected)

    def test_only_bounded_report_parquet_is_opened(self):
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve(); run,identity,expected=fixture(root)
            real=pq.ParquetFile; opened=[]
            def guarded(path,*args,**kwargs):
                path=Path(path); opened.append(str(path))
                self.assertTrue(path.name in {'comparison.parquet','yearly.parquet','selection_coverage.parquet',
                    'mode_class_performance.parquet','monthly.parquet'})
                return real(path,*args,**kwargs)
            with patch('dmv2.walkforward_reporting.pq.ParquetFile',side_effect=guarded), \
                    patch('numpy.load',side_effect=AssertionError('Matrix opened')), \
                    patch('joblib.load',side_effect=AssertionError('Model deserialized')):
                result=read_walkforward_run(root,run,identity_sha256=identity,expected_receipts=expected)
            self.assertEqual(len(result['model_comparison']),6); self.assertEqual(len(result['backtest_comparison']),64)
            self.assertEqual(len(result['backtest_yearly']),256); self.assertEqual(len(result['selection_coverage']),384)
            self.assertEqual(len(result['model_classes_1_2_9_10']),24); self.assertEqual(len(result['monthly_10bps']),768)
            self.assertFalse(result['summary']['models_deserialized']); self.assertFalse(result['summary']['numerical_work_started'])
            self.assertTrue(opened)

    def test_all_receipts_pass_before_any_report_value_decode(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve(); run,identity,expected=fixture(root)
            for kind in ('byte_hash','wrong_identity','metadata_only'):
                with self.subTest(kind=kind):
                    path=run/'verification/models.json'; original=path.read_text(); current=dict(expected)
                    if kind=='byte_hash':current['model_verification']='sha256:'+'0'*64
                    else:
                        receipt=json.loads(original)
                        if kind=='wrong_identity':receipt['build_identity_sha256']='sha256:'+'0'*64
                        else:receipt['status']='METADATA_PASSED'
                        save(path,receipt); current['model_verification']='sha256:'+pin(run,path)['sha256']
                    with patch('dmv2.walkforward_reporting.pq.ParquetFile',side_effect=AssertionError('Report decoded too early')), \
                            self.assertRaises(DataContractError):
                        read_walkforward_run(root,run,identity_sha256=identity,expected_receipts=current)
                    path.write_text(original)

    def test_verified_withheld_paths_keep_null_full_period_return(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve(); run,identity,expected=fixture(root,withheld=32)
            result=read_walkforward_run(root,run,identity_sha256=identity,expected_receipts=expected)
            self.assertEqual(result['backtest_summary']['failed_closed_paths'],32)
            self.assertTrue(result['backtest_comparison'].loc[lambda f:~f.complete_proxy_path,'total_return'].isna().all())

    def test_explicit_other_verification_directory_and_source_hashing_only(self):
        import shutil
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve(); run,identity,expected=fixture(root)
            target=root/'verification_copy'; shutil.copytree(run/'verification',target)
            result=read_walkforward_run(root,run,identity_sha256=identity,expected_receipts=expected,
                verification_directory=target,verify_sources=True)
            self.assertTrue(result['summary']['source_bytes_fully_hashed'])

    def test_partial_or_unpinned_report_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve(); run,identity,expected=fixture(root)
            path=run/'models/reports/comparison.parquet'; path.write_bytes(path.read_bytes()+b'x')
            with self.assertRaises(DataContractError):
                read_walkforward_run(root,run,identity_sha256=identity,expected_receipts=expected)

    def test_authenticated_receipt_selects_additive_backtest_folder(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve(); run,identity,expected=fixture(root)
            target=run/'backtest_action_completion'; (run/'backtest').rename(target)
            path=run/'verification/backtest.json'; receipt=json.loads(path.read_text())
            receipt['backtest_directory']=str(target.relative_to(root)); save(path,receipt)
            expected['backtest_verification']='sha256:'+pin(run,path)['sha256']
            result=read_walkforward_run(root,run,identity_sha256=identity,expected_receipts=expected)
            self.assertEqual(result['backtest_directory'],target)
            receipt['backtest_directory']='outputs/another_run'; save(path,receipt)
            expected['backtest_verification']='sha256:'+pin(run,path)['sha256']
            with self.assertRaises(DataContractError):
                read_walkforward_run(root,run,identity_sha256=identity,expected_receipts=expected)


if __name__=='__main__':unittest.main()
