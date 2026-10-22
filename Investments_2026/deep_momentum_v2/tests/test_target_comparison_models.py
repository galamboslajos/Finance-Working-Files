import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from dmv2 import target_comparison_models as m
from dmv2.features import registered_feature_specs
from dmv2.paper_features import PAPER_FEATURE_NAMES


def fixture(year):
    names = [x['name'] for x in registered_feature_specs() if x.get('model_eligible', True)]
    rows=[]
    for month in (1,12):
        period=pd.Period(f'{year}-{month:02d}',freq='M'); signal=str((period-1).end_time.date())
        for i in range(12): rows.append((f'S{i:02d}',signal))
    index=pd.MultiIndex.from_tuples(rows,names=m.KEYS)
    X=pd.DataFrame({name:np.arange(len(index),dtype=float)+n/100 for n,name in enumerate(names)},index=index)
    ledgers={}
    for head in m.HEADS:
        f=pd.DataFrame(index=index)
        f['outcome_month']=[str(pd.Period(k[1][:7],freq='M')+1) for k in index]
        f['outcome_start']=[str(pd.Period(x,freq='M').start_time.date()) for x in f.outcome_month]
        f['outcome_end']=[str(pd.Period(x,freq='M').end_time.date()) for x in f.outcome_month]
        f['label_available_at']=f.outcome_end if head=='endpoint' else [str((pd.Period(x,freq='M')+1).start_time.date()) for x in f.outcome_month]
        f['prediction_eligible']=True;f['paper_prediction_eligible']=True;f['target_available']=True
        f['target_class']=[1+i%10 for i in range(len(f))]
        f['identity_eligible']=True;f['requested']=True;f['side']=m.HEAD_SPEC[head][1]
        f['target_family']=m.HEAD_SPEC[head][0]
        ledgers[head]=f
    return X,{h:f.target_class for h,f in ledgers.items()},ledgers


def metadata_fixture(year=2023):
    windows=[]; original=[]; coverage=[]; endpoint=[]; monthly=[]
    for mth in range(1,13):
        period=pd.Period(f'{year}-{mth:02d}',freq='M')
        signal=str((period-1).end_time.date()); start=str(period.start_time.date()); end=str(period.end_time.date())
        exit_date=str((period+1).start_time.date())
        windows.append(dict(signal_date=signal,outcome_month=str(period),outcome_start=start,outcome_end=end,
            entry_date=start,exit_date=exit_date,exit_mark='open'))
        key=dict(source_security_id='S1',signal_date=signal);original.append(key)
        coverage.append(dict(**key,identity_eligible=True,requested=True,paper_prediction_eligible=True))
        endpoint.append(dict(**coverage[-1],outcome_month=str(period),outcome_start=start,outcome_end=end,
            prediction_eligible=True,target_available=True))
        records=[]
        for side in ('long','short'):
            for variant in ('unstopped','asset_stop_10pct'):
                for family in ('executable_return','return_minus_drawdown'):
                    records.append(dict(**key,outcome_month=str(period),entry_date=start,exit_date=exit_date,
                        exit_mark='open',label_available_at=exit_date,side=side,path_variant=variant,target_family=family,
                        prediction_eligible=True,target_available=exit_date<'2024-01-01',target_status='available'))
        digest=m.canonical([['S1',signal]])
        monthly.append((dict(outcome_month=str(period),original_rows=1,original_key_sha256=digest,eligible_key_sha256=digest),pd.DataFrame(records)))
    annual=dict(year=year,windows=windows,original_rows=12,prediction_eligible_rows=12)
    return annual,pd.DataFrame(original),pd.DataFrame(coverage),pd.DataFrame(original),pd.DataFrame(endpoint),monthly


class MetadataGuardTests(unittest.TestCase):
    def test_protected_endpoint_metadata_remains_missing(self):
        args=metadata_fixture()
        m.validate_annual_metadata(*args)
        self.assertEqual(len(args[-1]),12)

    def test_protected_financial_target_flag_rejected(self):
        args=metadata_fixture();args[-1][-1][1]['target_available']=True
        with self.assertRaisesRegex(ValueError,'Protected target'):
            m.validate_annual_metadata(*args)

    def test_forward_availability_cannot_change_prediction_keys(self):
        args=metadata_fixture();args[-1][0][1]['prediction_eligible']=False
        with self.assertRaisesRegex(ValueError,'prediction cohort'):
            m.validate_annual_metadata(*args)

    def test_bad_actual_completion_clock_rejected(self):
        args=metadata_fixture();args[-1][0][1]['label_available_at']='2023-01-31'
        with self.assertRaisesRegex(ValueError,'completion clock'):
            m.validate_annual_metadata(*args)

    def test_duplicate_path_key_rejected(self):
        args=list(metadata_fixture());month,frame=args[-1][0];args[-1][0]=(month,pd.concat([frame,frame.iloc[[0]]]))
        with self.assertRaisesRegex(ValueError,'Duplicate path'):
            m.validate_annual_metadata(*args)


class CommonMaskTests(unittest.TestCase):
    def test_actual_next_open_completion_purges_december(self):
        _,_,f=fixture(2019)
        mask=m.common_fit_mask(f,train_start='2010-01-01',train_end='2019-12-31',as_of='2019-12-31')
        self.assertEqual(mask.sum(),12)
        self.assertTrue(mask.iloc[:12].all());self.assertFalse(mask.iloc[12:].any())
        self.assertTrue(f['endpoint'].prediction_eligible.all())

    def test_missing_one_target_changes_fit_only(self):
        _,_,f=fixture(2018);key=f['endpoint'].index[0]
        f['utility_short'].loc[key,'target_available']=False;f['utility_short'].loc[key,'target_class']=np.nan
        mask=m.common_fit_mask(f,train_start='2010-01-01',train_end='2019-12-31',as_of='2019-12-31')
        self.assertEqual(mask.sum(),23);self.assertTrue(all(x.prediction_eligible.all() for x in f.values()))

    def test_misaligned_head_rejected(self):
        _,_,f=fixture(2018);f['cash_long']=f['cash_long'].iloc[::-1]
        with self.assertRaisesRegex(ValueError,'align'):
            m.common_fit_mask(f,train_start='2010-01-01',train_end='2019-12-31',as_of='2019-12-31')

    def test_target_cannot_admit_ineligible_row(self):
        _,_,f=fixture(2018);f['cash_long'].iloc[0,f['cash_long'].columns.get_loc('prediction_eligible')]=False
        with self.assertRaisesRegex(ValueError,'signal cohort'):
            m.common_fit_mask(f,train_start='2010-01-01',train_end='2019-12-31',as_of='2019-12-31')

    def test_protected_training_bound_rejected(self):
        _,_,f=fixture(2018)
        with self.assertRaises(ValueError):m.common_fit_mask(f,train_start='2010-01-01',train_end='2024-01-01',as_of='2023-12-31')


class FixedSettingsTests(unittest.TestCase):
    def test_inherited_settings_have_no_search(self):
        self.assertEqual([(m.fixed_parameters(y)['max_depth'],m.fixed_parameters(y)['min_samples_leaf']) for y in m.YEARS],[(12,20),(8,20),(12,100),(12,100)])
        for y in m.YEARS:
            p=m.fixed_parameters(y);self.assertEqual(p['n_estimators'],500);self.assertEqual(p['n_jobs'],4)
            self.assertEqual(p['random_state'],1729);self.assertIsNone(p['class_weight']);self.assertFalse(p['warm_start'])
        with self.assertRaises(ValueError):m.fixed_parameters(2024)

    def test_pin_and_array_hash_tampering(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a.npy';item=m.save_array(p,np.arange(6,dtype=np.float32).reshape(2,3))
            self.assertEqual(m.check(item),p.resolve())
            self.assertEqual(item['array_sha256'],m.array_hash(np.arange(6,dtype=np.float32).reshape(2,3)))
            with p.open('ab') as f:f.write(b'x')
            with self.assertRaisesRegex(ValueError,'Changed pinned'):m.check(item)

    def test_immutable_json_output(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'receipt.json';m.save(p,{'a':1})
            with self.assertRaises(FileExistsError):m.save(p,{'a':2})


class PreparationTests(unittest.TestCase):
    def test_one_shared_mask_preprocessor_and_all_predictions(self):
        with tempfile.TemporaryDirectory() as d:
            names=[x['name'] for x in registered_feature_specs() if x.get('model_eligible',True)]
            plan={'plan_sha256':'sha256:fixture','output_directory':d,'predictor_names':names,
                'annuals':{str(y):{'year':y} for y in (2010,2019,2020)},
                'folds':{'2020':{'annual_years':[2010,2019,2020], 'train_start':'2010-01-01','train_end':'2019-12-31',
                    'as_of':'2019-12-31','source_projection_bytes':123}}}
            def values(a,p):return fixture(a['year'])
            with patch.object(m,'verify_plan'),patch.object(m,'_annual_values',side_effect=values):
                item=m.prepare_fold(plan,2020)
            receipt=m.read(item['path']);self.assertEqual(receipt['preprocessor_fits'],1)
            self.assertEqual(receipt['train_fit_rows'],36);self.assertEqual(receipt['prediction_rows'],24)
            self.assertEqual(set(receipt['heads']),set(m.HEADS))
            ledger=pd.read_parquet(receipt['artifacts']['train_ledger']['path'])
            self.assertEqual(ledger.common_fit_mask.sum(),36)
            self.assertEqual(len(ledger),48)
            for head in m.HEADS:
                labels=np.load(receipt['artifacts']['train_y_'+head]['path'],allow_pickle=False)
                self.assertEqual(len(labels),36);self.assertEqual(set(labels),set(range(1,11)))
                pred=pd.read_parquet(receipt['artifacts']['prediction_ledger_'+head]['path'])
                self.assertTrue(pred.prediction_eligible.all());self.assertEqual(len(pred),24)
            raw=np.load(receipt['artifacts']['raw_train_X']['path'],allow_pickle=False)
            self.assertEqual(raw.shape,(36,715));self.assertEqual(raw.dtype,np.float64)
            prep=m.read(receipt['artifacts']['preprocessor']['path'])
            self.assertEqual(prep['manifest']['fit_rows'],36)
            self.assertEqual(prep['fingerprint'],receipt['preprocessor_fingerprint'])
            with patch.object(m,'verify_plan'),self.assertRaisesRegex(ValueError,'Preserve existing'):
                m.prepare_fold(plan,2020)

    def test_incomplete_fit_never_retried(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'folds/2020/heads/cash_long';m.save(out/'FIT_ATTEMPT.json',{'status':'CONSUMED'})
            plan={'output_directory':d}
            with patch.object(m,'verify_plan'),patch.object(m,'_prepared',return_value=(Path(d),{})),self.assertRaisesRegex(ValueError,'Incomplete fit'):
                m.fit_head(plan,2020,'cash_long')

    def test_incomplete_prediction_never_retried(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'folds/2020/heads/cash_long';m.save(out/'PREDICT_STARTED.json',{'status':'CONSUMED'})
            with patch.object(m,'verify_plan'),patch.object(m,'_prepared',return_value=(Path(d),{})),self.assertRaisesRegex(ValueError,'prediction attempt'):
                m.fit_head({'output_directory':d},2020,'cash_long')


if __name__=='__main__':unittest.main()
