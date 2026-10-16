"""Small synthetic-only tests for the bounded post-fit unit reviewer."""
from pathlib import Path
import importlib.util
import json
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

SPEC=importlib.util.spec_from_file_location('action_units',Path(__file__).parents[1]/'scripts/review_walkforward_action_units.py')
m=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(m)


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True)+'\n')
    return path


def localpin(base,path):
    result=m.pin(path);result['path']=str(Path(path).relative_to(base));return result


def event(cash=5.,ratio=.5):
    return {'event_id':'SYNTHETIC_20210107','predecessor_id':'sec_old','successor_id':'sec_new' if ratio else None,
        'effective_date':'2021-01-07','cash_per_raw_share':cash,'stock_ratio':ratio,
        'verified':False,'legal_terms_verified':True,'evidence_reference':'https://example.test/primary-terms',
        'separate_dividends_not_added_to_merger_cash':[{'amount':.5,'ex_date':'2021-01-20'}]}


def bar(date,native,factor=.5):
    raw={'open':80.,'high':85.,'low':75.,'close':82.}
    return {'trading_date':date,'intrinio_security_id':native,'frequency':'daily','intraperiod':False,
        'price_quality_status':'accepted','raw_ohlc_bounds_valid':True,'adjusted_ohlc_bounds_valid':True,
        **raw,**{'adj_'+k:value*factor for k,value in raw.items()}}


def completed_fixture(root):
    run=root/'run';run.mkdir()
    cfg={'protocol_id':'synthetic-only'};config=write(root/'configs/frozen.json',cfg)
    identity={'configuration':cfg};write(run/'build_identity.json',identity);identity_sha=m.canonical(identity)
    phase={'parent_identity_sha256':identity_sha};write(run/'models/build_identity.json',phase);model_sha=m.canonical(phase)
    folds={};cutoffs={2020:'2019-12-31',2021:'2020-12-31',2022:'2021-12-31',2023:'2022-12-30'}
    for year in m.YEARS:
        directory=run/'models/folds'/str(year);arms={}
        for arm in m.ARMS:
            arm_directory=directory/'arms'/arm
            expected={'test_year':year,'train_start':f'{year-10}-01-01','train_end':f'{year-1}-12-31',
                'label_observed_through':cutoffs[year]}
            fit=write(arm_directory/'FIT_COMPLETE.json',{'status':'FIT_COMPLETE','fit_count':1,
                'build_identity_sha256':identity_sha,'model_identity_sha256':model_sha,'expected_metadata':expected,
                'metadata':{'test_labels_supplied_to_estimator':False}})
            audit=write(arm_directory/'audit.json',{'status':'COMPLETE','build_identity_sha256':identity_sha,
                'model_identity_sha256':model_sha,'metadata':expected,'artifacts':{'fit_complete':localpin(arm_directory,fit)}})
            arms[arm]=localpin(directory,audit)
        fold=write(directory/'audit.json',{'status':'COMPLETE','test_year':year,'build_identity_sha256':identity_sha,
            'model_identity_sha256':model_sha,'summary':{'fitted_arms':4},'arms':arms})
        folds[str(year)]=localpin(run/'models',fold)
    model=write(run/'models/audit.json',{'status':'COMPLETE','phase':'walkforward_models_v1',
        'build_identity_sha256':identity_sha,'model_identity_sha256':model_sha,'folds':folds,
        'summary':{'fitted_arms':16,'preprocessor_fits':8,'months':48,'outer_years':list(m.YEARS)}})
    pre_phase={'parent_identity_sha256':identity_sha,'events':[],'source_pins':[localpin(root,model)]}
    pre_dir=run/'preflight/test';pre_identity=write(pre_dir/'build_identity.json',pre_phase)
    pre_audit=write(pre_dir/'audit.json',{'status':'COMPLETE','phase':'walkforward_selected_horizon_preflight_v1',
        'build_identity_sha256':identity_sha,'phase_identity_sha256':m.canonical(pre_phase),
        'artifacts':{'build_identity.json':localpin(pre_dir,pre_identity)}})
    years={}
    for year in m.YEARS:
        directory=run/'years'/str(year)
        if year==2021:rows=[bar('2021-01-06','sec_old'),bar('2021-01-07','sec_new',.25)]
        else:rows=[bar(f'{year}-06-01','sec_unused')]
        directory.mkdir(parents=True);path=directory/'daily_prices_raw.parquet'
        pd.DataFrame(rows).to_parquet(path,index=False)
        audit=write(directory/'audit.json',{'status':'COMPLETE','year':year,'build_identity_sha256':identity_sha,
            'feature_provenance':{'requested_ids':sorted({row['intrinio_security_id'] for row in rows})},
            'artifacts':{'daily_prices_raw':localpin(directory,path)}})
        years[str(year)]=localpin(run,audit)
    annual=write(run/'audit.json',{'status':'COMPLETE','build_identity_sha256':identity_sha,'years':years})
    draft=write(root/'evidence/draft.json',event())
    plan={'schema':m.SCHEMA,'project_root':str(root),'run_directory':'run','build_identity_sha256':identity_sha,
        'configuration':localpin(root,config),'run_audit':localpin(run,annual),'model_audit':localpin(run,model),
        'preflight_audit':localpin(run,pre_audit),'events':[m.pin(draft)]}
    plan_path=write(root/'plan.json',plan)
    return run,plan,plan_path


def add_supplement(root,plan,path,native='sec_supp'):
    draft=root/'evidence/draft.json';terms=json.loads(draft.read_text());terms['successor_id']=native;write(draft,terms)
    source=root/'data/derived/execution-only/supp.parquet';source.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame([bar('2021-01-07',native,.25)]).to_parquet(source,index=False)
    plan['events']=[m.pin(draft)]
    plan['supplemental_quote_sources']=[{'pin':localpin(root,source),'allowed_native_ids':[native],
        'start':'2021-01-07','end':'2021-01-07','evidence_reference':'https://example.test/closing',
        'identity_evidence_reference':'https://example.test/ordinary-shares'}]
    write(path,plan)
    return source


class UnitAlgebraTests(unittest.TestCase):
    def test_explicit_spinoff_or_unknown_cash_stock_type_is_rejected(self):
        for action_type in ('spinoff','unknown_cash_stock',None):
            with self.subTest(action_type=action_type):
                e=event();e['action_type']=action_type
                with self.assertRaisesRegex(m.ReviewError,'unsupported action_type'):m.event_requests([e],m.clock())
        m.event_requests([event()],m.clock()) # Existing CMD source has no action_type.
        for action_type in ('cash_merger','stock_merger','mixed_merger'):
            e=event();e['action_type']=action_type;m.event_requests([e],m.clock())

    def test_mixed_signed_cash_stock_and_fractional_adjusted_units(self):
        terms,keys=m.event_requests([event()],m.clock())
        result=m.review_event(terms[0],{('2021-01-06','sec_old'):bar('2021-01-06','sec_old'),
            ('2021-01-07','sec_new'):bar('2021-01-07','sec_new',.25)})
        self.assertEqual(result['status'],'PASS_UNIT_ALGEBRA')
        long,short=result['signed_examples']
        self.assertEqual((long['old_adjusted_units'],long['successor_signed_adjusted_units']),(2.,2.))
        self.assertEqual((long['signed_cash_receivable_or_liability'],short['signed_cash_receivable_or_liability']),(5.,-5.))
        self.assertEqual(long['raw_stock_plus_cash_mark'],-short['raw_stock_plus_cash_mark'])
        self.assertFalse(result['draft_event_promoted']);self.assertFalse(result['source_draft_verified'])
        self.assertEqual(len(result['separate_dividends_not_added']),1)

    def test_cash_only_uses_no_invented_successor_quote(self):
        terms,keys=m.event_requests([event(23.85,0)],m.clock())
        self.assertEqual(keys,{('2021-01-06','sec_old'):{'predecessor_basis'}})
        result=m.review_event(terms[0],{('2021-01-06','sec_old'):bar('2021-01-06','sec_old')})
        self.assertIsNone(result['successor_basis'])
        self.assertEqual(result['signed_examples'][1]['raw_stock_plus_cash_mark'],-23.85)

    def test_frozen_absolute_residual_threshold_and_bounds(self):
        row=bar('2021-01-06','sec_old');row['adj_high']+=.000199
        self.assertTrue(m.review_bar(row,'close')['pass'])
        row['adj_high']+=.000002
        result=m.review_bar(row,'close');self.assertFalse(result['pass'])
        self.assertEqual(result['threshold_absolute_adjusted_price'],.0002)
        row=bar('2021-01-06','sec_old');row['adj_low']=row['adj_high']+1
        with self.assertRaisesRegex(m.ReviewError,'OHLC bounds'):m.review_bar(row,'close')

    def test_future_event_and_zero_retirement_refuse_before_numeric_quotes(self):
        e=event();e['effective_date']='2024-01-02'
        with self.assertRaisesRegex(m.ReviewError,'prior authorized'):m.event_requests([e],m.clock())
        with self.assertRaisesRegex(m.ReviewError,'invented zero'):m.event_requests([event(0,0)],m.clock())

    def test_nonfinite_unit_factor_is_refused(self):
        row=bar('2021-01-06','sec_old')
        for key in m.OHLC:row[key]=1e-300;row['adj_'+key]=1e300
        with self.assertRaisesRegex(m.ReviewError,'positive finite'):m.review_bar(row,'close')

    def test_cy_diagnostic_requires_exact_cash_event(self):
        with self.assertRaisesRegex(m.ReviewError,'exact reviewed'):m.event_requests([event()],m.clock(),True)
        e=event(23.85,0);e.update(predecessor_id='sec_rg0Y2X',effective_date='2020-04-16')
        _,keys=m.event_requests([e],m.clock(),True)
        self.assertEqual(set(keys),{('2020-04-15','sec_rg0Y2X'),('2020-04-16','sec_rg0Y2X')})

    def test_external_cusip_route_never_claims_invented_intrinio_identity(self):
        e=event();e['successor_id']='execution_cusip:48214T305'
        with self.assertRaisesRegex(m.ReviewError,'External execution successor'):m.event_requests([e],m.clock())
        e.update(successor_identity_namespace='execution_cusip',successor_permanent_identifier='48214T305',
            successor_provider_identity_asserted=False)
        _,keys=m.event_requests([e],m.clock())
        self.assertIn(('2021-01-07','execution_cusip:48214T305'),keys)
        e['identity_namespace']=e.pop('successor_identity_namespace');e['permanent_identifier']=e.pop('successor_permanent_identifier')
        m.event_requests([e],m.clock())
        e['successor_provider_identity_asserted']=True
        with self.assertRaisesRegex(m.ReviewError,'invented provider'):m.event_requests([e],m.clock())

    def test_cap_rejects_excess_legal_event_keys(self):
        events=[]
        for i in range(32):
            e=event();e.update(event_id=str(i),predecessor_id=f'sec_old{i}',successor_id=f'sec_new{i}')
            events.append(e)
        _,keys=m.event_requests(events,m.clock());self.assertEqual(len(keys),64)
        events.append(event())
        with self.assertRaisesRegex(m.ReviewError,'One to32'):m.event_requests(events,m.clock())


class BoundedReadTests(unittest.TestCase):
    def test_new_execution_only_successor_has_separate_pinned_scope(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);run,plan,path=completed_fixture(root);add_supplement(root,plan,path)
            result=m.execute(path,m.sha(path),root/'result',authorize_postfit=True)
            self.assertEqual(result['status'],'PASS')
            self.assertEqual(result['supplemental_new_successor_natives'],['sec_supp'])
            bars=pd.read_parquet(root/'result/exact_paired_bar_evidence.parquet')
            self.assertEqual(set(bars.intrinio_security_id),{'sec_old','sec_supp'})
            self.assertEqual(result['budgets']['returned_financial_rows'],2)

    def test_supplement_cannot_extend_original_feature_native(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);run,plan,path=completed_fixture(root);add_supplement(root,plan,path,'sec_new')
            with patch.object(m,'scan',side_effect=AssertionError('No numeric scan')):
                with self.assertRaisesRegex(m.ReviewError,'original feature-requested'):m.execute(path,m.sha(path),root/'result',authorize_postfit=True)

    def test_supplement_rejects_native_hidden_in_original_keys_before_finance(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);run,plan,path=completed_fixture(root);add_supplement(root,plan,path)
            source=run/'years/2021/daily_prices_raw.parquet'
            rows=pd.read_parquet(source);rows.loc[1,'intrinio_security_id']='sec_supp';rows.to_parquet(source,index=False)
            annual_path=source.parent/'audit.json';annual=json.loads(annual_path.read_text())
            annual['artifacts']['daily_prices_raw']=localpin(source.parent,source);write(annual_path,annual)
            audit_path=run/'audit.json';audit=json.loads(audit_path.read_text());audit['years']['2021']=localpin(run,annual_path)
            write(audit_path,audit);plan['run_audit']=localpin(run,audit_path);write(path,plan)
            with patch.object(m,'scan',side_effect=AssertionError('No numeric scan')):
                with self.assertRaisesRegex(m.ReviewError,'already occurs'):m.execute(path,m.sha(path),root/'result',authorize_postfit=True)

    def test_supplement_future_key_is_rejected_by_metadata_scope(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);run,plan,path=completed_fixture(root);source=add_supplement(root,plan,path)
            pd.DataFrame([bar('2024-01-02','sec_supp')]).to_parquet(source,index=False)
            plan['supplemental_quote_sources'][0]['pin']=localpin(root,source);write(path,plan)
            with patch.object(m,'scan',side_effect=AssertionError('No numeric scan')):
                with self.assertRaisesRegex(m.ReviewError,'escape approved'):m.execute(path,m.sha(path),root/'result',authorize_postfit=True)

    def test_real_small_pinned_json_and_exact_parquet_projection(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);run,plan,path=completed_fixture(root);original=(root/'evidence/draft.json').read_bytes()
            result=m.execute(path,m.sha(path),root/'result',authorize_postfit=True)
            self.assertEqual(result['status'],'PASS')
            self.assertEqual(result['budgets']['requested_distinct_keys'],2)
            self.assertEqual(result['budgets']['returned_financial_rows'],2)
            self.assertLessEqual(result['budgets']['projected_compressed_bytes'],64*1024*1024)
            self.assertEqual((root/'evidence/draft.json').read_bytes(),original)
            self.assertFalse(result['event_drafts_changed_or_promoted'])
            self.assertFalse(result['execution_ready_certified'])
            self.assertEqual(result['model_fits'],0)
            self.assertEqual(len(pd.read_parquet(root/'result/exact_paired_bar_evidence.parquet')),2)
            with self.assertRaisesRegex(m.ReviewError,'Preserve'):m.execute(path,m.sha(path),root/'result',authorize_postfit=True)

    def test_incomplete_fit_or_stale_plan_has_no_parquet_scan(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);run,plan,path=completed_fixture(root)
            with patch.object(m,'scan',side_effect=AssertionError('No source scan allowed')):
                with self.assertRaisesRegex(m.ReviewError,'authorization'):m.execute(path,m.sha(path),root/'result')
                with self.assertRaisesRegex(m.ReviewError,'plan changed'):m.execute(path,'0'*64,root/'result',authorize_postfit=True)
                model=run/'models/audit.json';value=json.loads(model.read_text());value['summary']['fitted_arms']=15
                write(model,value);plan['model_audit']=localpin(run,model);write(path,plan)
                with self.assertRaisesRegex(m.ReviewError,'All16fits'):m.execute(path,m.sha(path),root/'result',authorize_postfit=True)

    def test_bad_metadata_never_projects_financial_columns(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);run,plan,path=completed_fixture(root)
            source=run/'years/2021/daily_prices_raw.parquet'
            rows=pd.read_parquet(source);rows.loc[0,'raw_ohlc_bounds_valid']=False;rows.to_parquet(source,index=False)
            annual_path=run/'years/2021/audit.json';annual=json.loads(annual_path.read_text())
            annual['artifacts']['daily_prices_raw']=localpin(source.parent,source);write(annual_path,annual)
            root_audit=run/'audit.json';annuals=json.loads(root_audit.read_text());annuals['years']['2021']=localpin(run,annual_path)
            write(root_audit,annuals);plan['run_audit']=localpin(run,root_audit);write(path,plan)
            original=m.scan;observed=[]
            def spy(item,columns):
                observed.append(columns)
                self.assertFalse(set(m.PAIRED)&set(columns),'Numeric source projection occurred before strict quality guard')
                return original(item,columns)
            with patch.object(m,'scan',side_effect=spy):
                with self.assertRaisesRegex(m.ReviewError,'source-quality'):m.execute(path,m.sha(path),root/'result',authorize_postfit=True)
            self.assertEqual(len(observed),1)

    def test_compressed_budget_fails_before_even_metadata_scan(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);run,plan,path=completed_fixture(root)
            original=m.file_plan
            def oversized(*args):
                item=original(*args);item['projected_compressed_bytes']=m.MAX_COMPRESSED_BYTES+1;return item
            with patch.object(m,'file_plan',side_effect=oversized),patch.object(m,'scan',side_effect=AssertionError('No scan')):
                with self.assertRaisesRegex(m.ReviewError,'64MiB'):m.execute(path,m.sha(path),root/'result',authorize_postfit=True)

    def test_absent_or_wrong_identity_and_future_footer_fail_closed(self):
        with tempfile.TemporaryDirectory() as td:
            source=Path(td)/'small.parquet';pd.DataFrame([bar('2021-01-06','sec_old')]).to_parquet(source,index=False)
            item=m.file_plan(source,2021,{('2021-01-06','sec_old')},False)
            rows=m.scan(item,list(m.KEYS));m.guard_rows(rows,item['keys'])
            with self.assertRaisesRegex(m.ReviewError,'absent or duplicated'):m.guard_rows([],{('2021-01-06','sec_old')})
            with self.assertRaisesRegex(m.ReviewError,'escaped exact'):m.guard_rows(rows,{('2021-01-06','sec_other')})
            pd.DataFrame([bar('2024-01-02','sec_old')]).to_parquet(source,index=False)
            with self.assertRaisesRegex(m.ReviewError,'unauthorized date'):m.file_plan(source,2021,{('2021-01-06','sec_old')},False)


if __name__=='__main__':unittest.main()
