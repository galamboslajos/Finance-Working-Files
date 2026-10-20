"""Synthetic new-year artifact chronology, financial guards and passive preflight."""
from pathlib import Path
from types import SimpleNamespace
import importlib.util,sys,tempfile,unittest,json
from unittest.mock import patch
import pyarrow as pa,pyarrow.parquet as pq
from test_walkforward_loading import m as loading,pin
try:from dmv2 import walkforward_panel as p
except ImportError:
 spec=importlib.util.spec_from_file_location('dmv2.walkforward_panel',Path(__file__).parents[1]/'src/dmv2/walkforward_panel.py');p=importlib.util.module_from_spec(spec);sys.modules[spec.name]=p;spec.loader.exec_module(p)

def fake_result():
 keys=pa.table({'source_security_id':['A'],'signal_date':['2019-12-31']});coverage=keys.append_column('paper_prediction_eligible',pa.array([True]));target=keys.append_column('target_available',pa.array([True]));daily=pa.table({'intrinio_security_id':['native','native'],'trading_date':['2019-12-31','2020-01-02'],'open':[1.,2.]});feature=SimpleNamespace(table=keys,coverage=coverage,provenance={'target_windows':{'2019-12-31':{'outcome_month':'2020-01','outcome_end':'2020-01-31'}},'feature_dictionary_sha256':'synthetic'},summary={});monthly=SimpleNamespace(table=keys,provenance={},summary={});targets=SimpleNamespace(table=target,diagnostics=pa.table({'test':['synthetic']}),provenance={},artifact_sha256='synthetic');plan=SimpleNamespace(year=2020,original_cohort=keys,daily_cohorts=keys,calendar=pa.table({'trading_date':['2020-01-02'],'calendar_name':['XNYS'],'is_trading_session':[True]}),read_plan={'monthly_wrong_route_rows':0,'daily_wrong_route_rows':0});loaded=SimpleNamespace(plan=plan,feature_prices=daily.slice(0,1),outcome_prices=daily);return SimpleNamespace(loaded=loaded,features=feature,monthly=monthly,targets=targets,provenance={'test':'synthetic'})
class WalkforwardPanelContractTests(unittest.TestCase):
 def test_save_separates_warmup_from_disjoint_2020_outcomes(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);result=fake_result();audit=p._save_walkforward_year(root,result,{'configuration':{'research_status':'DATA_LIMITED_RESTRICTED_UNIVERSE'},'quote_amendments_fingerprint':'synthetic'},'synthetic',.1)
   self.assertEqual(pq.read_table(root/'daily_prices_raw.parquet')['trading_date'].to_pylist(),['2020-01-02']);self.assertEqual(pq.read_table(root/'daily_history_raw.parquet')['trading_date'].to_pylist(),['2019-12-31','2020-01-02']);self.assertEqual(pq.read_table(root/'feature_history_daily.parquet')['trading_date'].to_pylist(),['2019-12-31']);self.assertTrue(audit['summary']['financial_2020_2023_analysis']);self.assertFalse(audit['summary']['financial_2024_plus_analysis']);self.assertTrue({'features','feature_coverage','original_cohort','targets','target_diagnostics','target_state','daily_cohorts'}<=set(audit['artifacts']))
 def test_guard_precedes_any_planning_or_build(self):
  with patch.object(p,'plan_walkforward_panel',side_effect=AssertionError('unexpected metadata/planning')):
   with self.assertRaises(p.DataContractError):p.build_walkforward_panel('/private/tmp',{},None)
 def test_plan_rejects_another_legitimate_bundle_under_frozen_recipe(self):
  cfg={'protocol_id':'dmv2-walkforward-2020-2023-v1','feature_outcome_years':[2020,2021,2022,2023],
   'context_policy':'available_cohort_80pct','research_status':'DATA_LIMITED_RESTRICTED_UNIVERSE',
   'execution':{'financial_2020_2023_analysis_authorized':True,'financial_2024_plus_analysis':False},
   'execution_restrictions':{'native_ids':{'sec_gjP4xM':'same','sec_zmYNnz':'same'}},'quote_cache_fingerprint':'expected'}
  bundle=SimpleNamespace(fingerprint='different',verify=lambda root:None)
  with patch.object(p,'wrap_walkforward_quotes',return_value=bundle),patch.object(p,'load_identity_amendment',side_effect=AssertionError('should stop before input planning')):
   with self.assertRaises(p.DataContractError):p.plan_walkforward_panel('/private/tmp',cfg,bundle)
 def test_old_loaded_class_cannot_bypass_new_scope(self):
  with self.assertRaises(p.DataContractError):p.build_walkforward_year(SimpleNamespace())
class QuoteMetadataPreflightTests(unittest.TestCase):
 def plan(self,root,extra_date=None,extra_id=None):
  rows=[dict(intrinio_security_id='native',trading_date='2020-01-02',price_quality_status='accepted',raw_ohlc_bounds_valid=True,adjusted_ohlc_bounds_valid=True,open='NOT_A_FINANCIAL_VALUE'),dict(intrinio_security_id=extra_id or'native',trading_date=extra_date or'2020-01-06',price_quality_status='provider_ohlc_anomaly',raw_ohlc_bounds_valid=False,adjusted_ohlc_bounds_valid=False,open='POISON_FINANCIAL_COLUMN')];path=root/'daily_history_raw.parquet';pq.write_table(pa.Table.from_pylist(rows),path);calendar=pa.table({'trading_date':['2020-01-02','2020-01-03','2020-01-06'],'calendar_name':['XNYS']*3,'is_trading_session':[True]*3});empty=pa.table({'source_security_id':[],'signal_date':[]});return loading.WalkforwardYearPlan(root,root,2020,{},calendar,(),empty,empty,(),('native',),('native',),'2019-01-01','2020-11-30','2020-12-31',{}, {'used_artifacts':{'daily_history_raw':pin(root,path)}},loading.WalkforwardReadLimits(),'unused')
 def test_global_states_and_holes_without_financial_projection(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d).resolve();plan=self.plan(root);real=pq.ParquetFile;calls=[]
   class Guard:
    def __init__(self,path):self.actual=real(path);self.schema_arrow=self.actual.schema_arrow
    def read(self,columns=None,**kwargs):
     assert columns and 'open' not in columns;calls.append(columns);return self.actual.read(columns=columns,**kwargs)
   with patch.object(loading.pq,'ParquetFile',Guard):r=loading.preflight_walkforward_quote_metadata(plan)
   self.assertEqual(r['invalid_quote_key_count'],1);self.assertEqual(r['interior_missing_keys'],[dict(intrinio_security_id='native',trading_date='2020-01-03')]);self.assertFalse(r['eligibility_changed']);self.assertFalse(r['financial_columns_read']);self.assertTrue(calls)
 def test_protected_metadata_refused_before_any_financial_projection(self):
  with tempfile.TemporaryDirectory() as d:
   plan=self.plan(Path(d).resolve(),extra_date='2024-01-02')
   with self.assertRaises(loading.DataContractError):loading.preflight_walkforward_quote_metadata(plan)
 def test_unrequested_native_metadata_refused(self):
  with tempfile.TemporaryDirectory() as d:
   plan=self.plan(Path(d).resolve(),extra_id='UNREQUESTED')
   with self.assertRaises(loading.DataContractError):loading.preflight_walkforward_quote_metadata(plan)
if __name__=='__main__':unittest.main()
