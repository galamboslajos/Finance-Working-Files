"""Synthetic overlap/unit/source checks; no private fixtures or remote calls."""
import unittest,copy,importlib.util,sys
from pathlib import Path
# Install into the active package only for staged testing; production imports normally.
MODULE=Path(__file__).parents[1]/'src/dmv2/independent_quotes.py'
if MODULE.exists():
 import dmv2
 spec=importlib.util.spec_from_file_location('dmv2.independent_quotes',MODULE);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
else:
 from dmv2 import independent_quotes as m

def fixture(missing=False):
 c={'intrinio_security_id':'native','trading_date':'2019-08-12','source_security_id':'source','symbol_candidates':['ABC'],'preceding_control':['2019-08-09'],'following_control':['2019-08-13'],'dates':['2019-08-09','2019-08-12','2019-08-13']}
 def original(day):
  r=dict(intrinio_security_id='native',trading_date=day,open=100.,high=102.,low=99.,close=101.,adj_open=80.,adj_high=81.6,adj_low=79.2,adj_close=80.8,volume=1000.,adj_volume=1000.,factor=1.,split_ratio=1.,dividend=0.,price_quality_status='accepted',raw_ohlc_bounds_valid=True,adjusted_ohlc_bounds_valid=True,frequency='daily',intraperiod=False);return r
 originals={'preceding':original('2019-08-09'),'following':original('2019-08-13')}
 if not missing:
  originals['target']=original('2019-08-12');originals['target'].update(low=104.,price_quality_status='provider_ohlc_anomaly',raw_ohlc_bounds_valid=False,adjusted_ohlc_bounds_valid=False)
 sip={}
 for day in c['dates']:
  sip['source',day]=dict(security_id='source',trading_date=day,query_symbol='ABC',feed='sip',adjustment='raw',timeframe='1Day',source_page_hash='a'*64,source_task_id='task',source_request_id='request',open=100.,high=102.,low=99.,close=101.)
 return c,originals,sip
class IndependentQuoteChecks(unittest.TestCase):
 def test_replaces_actual_full_bar_preserves_action_volume(self):
  c,o,s=fixture();result=m.review_independent_case(c,o,s);self.assertEqual(result['status'],'ACCEPTED_REPLACE');self.assertEqual(result['replacement_fields']['adj_close'],80.8);self.assertNotIn('volume',result['replacement_fields']);self.assertNotIn('factor',result['replacement_fields'])
 def test_missing_appends_real_quote_unknown_volume_actions(self):
  c,o,s=fixture(True);r=m.review_independent_case(c,o,s);self.assertEqual(r['status'],'ACCEPTED_APPEND');self.assertIsNone(r['replacement_fields']['volume']);self.assertIsNone(r['replacement_fields']['split_ratio'])
 def test_no_future_value_supplies_multiplier(self):
  c,o,s=fixture();a=m.review_independent_case(c,o,s);o['following']['close']=201.;s['source','2019-08-13']['close']=201.;o['following']['high']=202.;s['source','2019-08-13']['high']=202.;o['following']['adj_close']=160.8;o['following']['adj_high']=161.6;b=m.review_independent_case(c,o,s);self.assertEqual(a['replacement_fields'],b['replacement_fields']);self.assertEqual(a['multiplier_source_date'],'2019-08-09')
 def test_bad_following_units_refused(self):
  c,o,s=fixture();o['following']['adj_close']+=.001;r=m.review_independent_case(c,o,s);self.assertEqual(r['status'],'REFUSED_UNCHANGED');self.assertIn('0.0002',r['reason'])
 def test_bad_raw_control_refused_for_replacement(self):
  c,o,s=fixture();s['source','2019-08-09']['high']+=.01;r=m.review_independent_case(c,o,s);self.assertIn('full raw',r['reason'])
 def test_append_uses_frozen_close_overlap_contract(self):
  c,o,s=fixture(True);s['source','2019-08-09']['high']+=.01;r=m.review_independent_case(c,o,s);self.assertEqual(r['status'],'ACCEPTED_APPEND');self.assertFalse(r['full_raw_control_equality_required'])
 def test_close_absolute_and_relative_both_required(self):
  for old,new in [(101.,101.06),(1.,1.006)]:
   c,o,s=fixture(True)
   if old==1.:
    for role in ['preceding','following']:
     for field in ['open','high','low','close']:o[role][field]=1.;o[role]['adj_'+field]=.8
    for row in s.values():row.update(open=1.,high=1.01,low=.99,close=1.)
   s['source','2019-08-09']['close']=new
   self.assertEqual(m.review_independent_case(c,o,s)['status'],'REFUSED_UNCHANGED')
 def test_target_action_refused(self):
  for field,value in [('split_ratio',2.),('dividend',.1)]:
   c,o,s=fixture();o['target'][field]=value;self.assertIn('Target action',m.review_independent_case(c,o,s)['reason'])
 def test_following_action_refused(self):
  c,o,s=fixture();o['following']['dividend']=.1;self.assertIn('Following dividend',m.review_independent_case(c,o,s)['reason'])
 def test_preceding_dividend_already_in_pair_accepted(self):
  c,o,s=fixture();o['preceding']['dividend']=.1;r=m.review_independent_case(c,o,s);self.assertEqual(r['status'],'ACCEPTED_REPLACE');self.assertEqual(r['preceding_dividend_already_in_paired_units'],.1)
 def test_accepted_target_never_replaced(self):
  c,o,s=fixture();o['target']['price_quality_status']='accepted';self.assertEqual(m.review_independent_case(c,o,s)['status'],'REFUSED_UNCHANGED')
 def test_missing_source_target_refused(self):
  c,o,s=fixture();del s['source','2019-08-12'];self.assertIn('absent',m.review_independent_case(c,o,s)['reason'])
 def test_wrong_symbol_refused(self):
  c,o,s=fixture();s['source','2019-08-12']['query_symbol']='OTHER';self.assertIn('symbol',m.review_independent_case(c,o,s)['reason'])
 def test_wrong_feed_refused(self):
  c,o,s=fixture();s['source','2019-08-12']['feed']='iex';self.assertIn('feed',m.review_independent_case(c,o,s)['reason'])
 def test_wrong_chronology_refused(self):
  c,o,s=fixture();c['preceding_control']=['2019-08-13'];self.assertIn('chronology',m.review_independent_case(c,o,s)['reason'])
 def test_scope_refuses_2020_before_numeric_conversion(self):
  c,o,s=fixture();c['trading_date']='2020-01-02';o['target']['open']='not-a-number';self.assertIn('authorized',m.review_independent_case(c,o,s)['reason'])
 def test_incomplete_target_close_does_not_define_units(self):
  c,o,s=fixture();o['target'].update(open=None,high=None,low=None,close=999.,adj_close=999.,price_quality_status='provider_incomplete_ohlc');r=m.review_independent_case(c,o,s);self.assertEqual(r['status'],'ACCEPTED_REPLACE');self.assertAlmostEqual(r['multiplier'],.8)
 def test_global_review_retains_refusals_without_predictions(self):
  c,o,s=fixture();controls=[dict(row,target_native_id='native',target_date=c['trading_date'],role=role) for role,row in o.items()];r=m.review_independent_quotes([c],controls,list(s.values()));self.assertEqual(r['target_count'],1);self.assertEqual(r['accepted_count'],1)
 def test_duplicate_source_quote_rejected(self):
  c,o,s=fixture();controls=[dict(row,target_native_id='native',target_date=c['trading_date'],role=role) for role,row in o.items()]
  with self.assertRaises(m.DataContractError):m.review_independent_quotes([c],controls,list(s.values())+[next(iter(s.values()))])
 def test_conflicting_aliases_not_chosen_for_advantage(self):
  c,o,s=fixture();other=copy.deepcopy(c);other['source_security_id']='alias';rows=list(s.values());aliases=[]
  for row in rows:
   x=copy.deepcopy(row);x['security_id']='alias'
   if x['trading_date']==c['trading_date']:x['close']=100.
   aliases.append(x)
  controls=[dict(row,target_native_id='native',target_date=c['trading_date'],role=role) for role,row in o.items()];r=m.review_independent_quotes([c,other],controls,rows+aliases);self.assertEqual(r['accepted_count'],0);self.assertIn('Conflicting',r['target_summary'][0]['reason'])
class IndependentCompiledChecks(unittest.TestCase):
 def bundle(self,missing=False):
  import pandas as pd
  c,o,s=fixture(missing);controls=[dict(row,target_native_id='native',target_date=c['trading_date'],role=role) for role,row in o.items()];review=m.review_independent_quotes([c],controls,list(s.values()));original=pd.DataFrame(list(o.values()));corrected=original.copy(deep=True)
  if missing:
   row={field:review['accepted'][0]['replacement_fields'].get(field) for field in original.columns};corrected=pd.concat([corrected,pd.DataFrame([row])],ignore_index=True)
  else:
   index=corrected.index[corrected.trading_date==c['trading_date']][0]
   for field,value in review['accepted'][0]['replacement_fields'].items():corrected.at[index,field]=value
  compiled=m.build_quote_amendments(original,corrected,ledger=pd.DataFrame(review['route_ledger']),sessions=c['dates']);fp=m.canonical_sha256({'policy_id':m.POLICY_ID,'compiled':compiled.fingerprint,'review':review,'pins':[]});return m.IndependentQuoteAmendments(compiled,review,(),fp),original
 def test_compiled_replacement_preserves_volume_and_existing_audit(self):
  import pyarrow as pa
  bundle,original=self.bundle();original['original_close']=777.;raw=pa.Table.from_pandas(original,preserve_index=False);after=bundle.apply(raw).to_pandas();self.assertEqual(after.volume.tolist(),original.volume.tolist());self.assertEqual(after.original_close.tolist(),[777.]*len(original));self.assertEqual(after.loc[after.trading_date=='2019-08-12','low'].iloc[0],99.)
 def test_compiled_append_preserves_every_source_row(self):
  import pyarrow as pa
  bundle,original=self.bundle(True);after=bundle.apply(pa.Table.from_pandas(original,preserve_index=False)).to_pandas();self.assertEqual(len(after),len(original)+1);self.assertEqual(after.iloc[:len(original)].volume.tolist(),original.volume.tolist());self.assertTrue(after.iloc[-1].isna()['volume'])
 def test_compiled_integrity_detects_review_tampering(self):
  bundle,original=self.bundle();bundle.review['target_count']=999
  with self.assertRaises(m.DataContractError):bundle.verify(Path('/private/tmp'))
if __name__=='__main__':unittest.main()
