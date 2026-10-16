"""Synthetic future quote admission, complete-bundle routing and source guards."""
import unittest,copy,json,hashlib
from pathlib import Path
import pandas as pd
import pyarrow as pa
from dmv2 import walkforward_quotes as m
from dmv2.walkforward_loading import wrap_walkforward_quotes

def fixture(missing=False):
 c={'intrinio_security_id':'native','trading_date':'2021-05-12','source_security_id':'source','symbol_candidates':['ABC'],'preceding_control':['2021-05-09'],'following_control':['2021-05-13'],'dates':['2021-05-09','2021-05-12','2021-05-13']}
 def original(day):
  r=dict(intrinio_security_id='native',trading_date=day,open=100.,high=102.,low=99.,close=101.,adj_open=80.,adj_high=81.6,adj_low=79.2,adj_close=80.8,volume=1000.,adj_volume=1000.,factor=1.,split_ratio=1.,dividend=0.,price_quality_status='accepted',raw_ohlc_bounds_valid=True,adjusted_ohlc_bounds_valid=True,frequency='daily',intraperiod=False);return r
 originals={'preceding':original('2021-05-09'),'following':original('2021-05-13')}
 if not missing:
  originals['target']=original('2021-05-12');originals['target'].update(low=104.,price_quality_status='provider_ohlc_anomaly',raw_ohlc_bounds_valid=False,adjusted_ohlc_bounds_valid=False)
 sip={}
 for day in c['dates']:
  sip['source',day]=dict(security_id='source',trading_date=day,query_symbol='ABC',feed='sip',adjustment='raw',timeframe='1Day',source_page_hash='a'*64,source_task_id='task',source_request_id='request',open=100.,high=102.,low=99.,close=101.)
 return c,originals,sip
class IndependentQuoteChecks(unittest.TestCase):
 def test_replaces_actual_full_bar_preserves_action_volume(self):
  c,o,s=fixture();result=m.review_walkforward_quote_case(c,o,s);self.assertEqual(result['status'],'ACCEPTED_REPLACE');self.assertEqual(result['replacement_fields']['adj_close'],80.8);self.assertNotIn('volume',result['replacement_fields']);self.assertNotIn('factor',result['replacement_fields'])
 def test_missing_appends_real_quote_unknown_volume_actions(self):
  c,o,s=fixture(True);r=m.review_walkforward_quote_case(c,o,s);self.assertEqual(r['status'],'ACCEPTED_APPEND');self.assertIsNone(r['replacement_fields']['volume']);self.assertIsNone(r['replacement_fields']['split_ratio'])
 def test_no_future_value_supplies_multiplier(self):
  c,o,s=fixture();a=m.review_walkforward_quote_case(c,o,s);o['following']['close']=201.;s['source','2021-05-13']['close']=201.;o['following']['high']=202.;s['source','2021-05-13']['high']=202.;o['following']['adj_close']=160.8;o['following']['adj_high']=161.6;b=m.review_walkforward_quote_case(c,o,s);self.assertEqual(a['replacement_fields'],b['replacement_fields']);self.assertEqual(a['multiplier_source_date'],'2021-05-09')
 def test_bad_following_units_refused(self):
  c,o,s=fixture();o['following']['adj_close']+=.001;r=m.review_walkforward_quote_case(c,o,s);self.assertEqual(r['status'],'REFUSED_UNCHANGED');self.assertIn('0.0002',r['reason'])
 def test_bad_raw_control_refused_for_replacement(self):
  c,o,s=fixture();s['source','2021-05-09']['high']+=.01;r=m.review_walkforward_quote_case(c,o,s);self.assertIn('full raw',r['reason'])
 def test_append_uses_frozen_close_overlap_contract(self):
  c,o,s=fixture(True);s['source','2021-05-09']['high']+=.01;r=m.review_walkforward_quote_case(c,o,s);self.assertEqual(r['status'],'ACCEPTED_APPEND');self.assertFalse(r['full_raw_control_equality_required'])
 def test_close_absolute_and_relative_both_required(self):
  for old,new in [(101.,101.06),(1.,1.006)]:
   c,o,s=fixture(True)
   if old==1.:
    for role in ['preceding','following']:
     for field in ['open','high','low','close']:o[role][field]=1.;o[role]['adj_'+field]=.8
    for row in s.values():row.update(open=1.,high=1.01,low=.99,close=1.)
   s['source','2021-05-09']['close']=new
   self.assertEqual(m.review_walkforward_quote_case(c,o,s)['status'],'REFUSED_UNCHANGED')
 def test_target_action_refused(self):
  for field,value in [('split_ratio',2.),('dividend',.1)]:
   c,o,s=fixture();o['target'][field]=value;self.assertIn('Target action',m.review_walkforward_quote_case(c,o,s)['reason'])
 def test_following_action_refused(self):
  c,o,s=fixture();o['following']['dividend']=.1;self.assertIn('Following dividend',m.review_walkforward_quote_case(c,o,s)['reason'])
 def test_preceding_dividend_already_in_pair_accepted(self):
  c,o,s=fixture();o['preceding']['dividend']=.1;r=m.review_walkforward_quote_case(c,o,s);self.assertEqual(r['status'],'ACCEPTED_REPLACE');self.assertEqual(r['preceding_dividend_already_in_paired_units'],.1)
 def test_accepted_target_never_replaced(self):
  c,o,s=fixture();o['target']['price_quality_status']='accepted';self.assertEqual(m.review_walkforward_quote_case(c,o,s)['status'],'REFUSED_UNCHANGED')
 def test_missing_source_target_refused(self):
  c,o,s=fixture();del s['source','2021-05-12'];self.assertIn('absent',m.review_walkforward_quote_case(c,o,s)['reason'])
 def test_wrong_symbol_refused(self):
  c,o,s=fixture();s['source','2021-05-12']['query_symbol']='OTHER';self.assertIn('symbol',m.review_walkforward_quote_case(c,o,s)['reason'])
 def test_wrong_feed_refused(self):
  c,o,s=fixture();s['source','2021-05-12']['feed']='iex';self.assertIn('feed',m.review_walkforward_quote_case(c,o,s)['reason'])
 def test_wrong_chronology_refused(self):
  c,o,s=fixture();c['preceding_control']=['2021-05-13'];self.assertIn('chronology',m.review_walkforward_quote_case(c,o,s)['reason'])
 def test_scope_refuses_2020_before_numeric_conversion(self):
  c,o,s=fixture();c['trading_date']='2024-01-02';o['target']['open']='not-a-number';self.assertIn('authorized',m.review_walkforward_quote_case(c,o,s)['reason'])
 def test_incomplete_target_close_does_not_define_units(self):
  c,o,s=fixture();o['target'].update(open=None,high=None,low=None,close=999.,adj_close=999.,price_quality_status='provider_incomplete_ohlc');r=m.review_walkforward_quote_case(c,o,s);self.assertEqual(r['status'],'ACCEPTED_REPLACE');self.assertAlmostEqual(r['multiplier'],.8)
 def test_global_review_retains_refusals_without_predictions(self):
  c,o,s=fixture();controls=[dict(row,target_native_id='native',target_date=c['trading_date'],role=role) for role,row in o.items()];r=m.review_walkforward_quotes([c],controls,list(s.values()));self.assertEqual(r['target_count'],1);self.assertEqual(r['accepted_count'],1)
 def test_duplicate_source_quote_rejected(self):
  c,o,s=fixture();controls=[dict(row,target_native_id='native',target_date=c['trading_date'],role=role) for role,row in o.items()]
  with self.assertRaises(m.DataContractError):m.review_walkforward_quotes([c],controls,list(s.values())+[next(iter(s.values()))])
 def test_conflicting_aliases_not_chosen_for_advantage(self):
  c,o,s=fixture();other=copy.deepcopy(c);other['source_security_id']='alias';rows=list(s.values());aliases=[]
  for row in rows:
   x=copy.deepcopy(row);x['security_id']='alias'
   if x['trading_date']==c['trading_date']:x['close']=100.
   aliases.append(x)
  controls=[dict(row,target_native_id='native',target_date=c['trading_date'],role=role) for role,row in o.items()];r=m.review_walkforward_quotes([c,other],controls,rows+aliases);self.assertEqual(r['accepted_count'],0);self.assertIn('Conflicting',r['target_summary'][0]['reason'])
class FutureBundleApplicationTests(unittest.TestCase):
 def bundle(self,missing=False):
  c,o,s=fixture(missing);controls=[dict(row,target_native_id='native',target_date=c['trading_date'],role=role) for role,row in o.items()];review=m.review_walkforward_quotes([c],controls,list(s.values()));entry=review['accepted'][0];before=o.get('target',{});after={**before,**entry['replacement_fields']};record=dict(intrinio_security_id='native',trading_date=c['trading_date'],kind=entry['kind'],original_bar_sha256=entry['original_target_bar_sha256'],replacement_bar_sha256=m.original_bar_sha256(after),fields_json=json.dumps(entry['replacement_fields'],sort_keys=True),policy_id=m.POLICY_ID);amendments=pa.Table.from_pylist([record],schema=m.AMENDMENT_SCHEMA);worker_sha=hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest();fp=m.canonical_sha256(m._snapshot(amendments,review,(),worker_sha));return m.ReviewedWalkforwardQuotes(amendments,review,(),worker_sha,fp),pa.Table.from_pylist(list(o.values()))
 def legacy(self):
  class Legacy:
   fingerprint='synthetic_old';provenance={'scope':'old2018-2019'}
   def verify(self,root):return None
   def apply(self,t,*,start=None,end=None):
    assert all(d<'2020-01-01' for d in t['trading_date'].to_pylist());return t
  return Legacy()
 def test_future_replacement_through_repeated_plan_load_wrapper(self):
  recent,table=self.bundle();complete=m.compose_complete_walkforward_quotes(self.legacy(),recent);wrapped=wrap_walkforward_quotes(wrap_walkforward_quotes(complete));self.assertIs(wrapped,complete);after=wrapped.apply(table);target=after.to_pandas().query("trading_date=='2021-05-12'").iloc[0];self.assertEqual(target.low,99.);self.assertEqual(target.volume,1000.)
 def test_future_append_through_repeated_plan_load_wrapper(self):
  recent,table=self.bundle(True);complete=m.compose_complete_walkforward_quotes(self.legacy(),recent);after=wrap_walkforward_quotes(wrap_walkforward_quotes(complete)).apply(table);self.assertEqual(len(after),len(table)+1);self.assertEqual(after['trading_date'][-1].as_py(),'2021-05-12');self.assertIsNone(after['volume'][-1].as_py());self.assertIsNone(after['split_ratio'][-1].as_py())
 def test_append_is_idempotent_only_for_exact_reviewed_replacement_hash(self):
  recent,t=self.bundle(True);after=recent.apply(t);again=recent.apply(after);self.assertTrue(after.equals(again));rows=after.to_pylist();rows[-1]['open']+=1.
  with self.assertRaises(m.DataContractError):recent.apply(pa.Table.from_pylist(rows,schema=after.schema))
 def test_existing_volume_and_action_fields_remain_byte_equal(self):
  recent,t=self.bundle();after=recent.apply(t)
  for f in ['volume','adj_volume','factor','split_ratio','dividend']:self.assertTrue(after[f].equals(t[f]))
 def test_original_bar_change_cannot_be_silently_repaired(self):
  recent,t=self.bundle();rows=t.to_pylist();rows[-1]['close']+=.01
  with self.assertRaises(m.DataContractError):recent.apply(pa.Table.from_pylist(rows,schema=t.schema))
 def test_protected2024_metadata_stops_before_price_parsing(self):
  recent,t=self.bundle();poison=pa.table({'intrinio_security_id':['native'],'trading_date':['2024-01-02'],'open':['poison']})
  with self.assertRaises(m.DataContractError):recent.apply(poison)
 def test_new_quote_bundle_integrity_guard(self):
  recent,t=self.bundle();recent.review['accepted_count']=999
  with self.assertRaises(m.DataContractError):recent.apply(t)

class OriginalControlsMetadataGuards(unittest.TestCase):
 def test_2024_control_poison_rejected_before_full_table_read(self):self.check('2024-01-02','native','2021-05-12')
 def test_unrequested_native_poison_rejected_before_full_table_read(self):self.check('2021-05-09','unrequested','2021-05-12')
 def check(self,day,native,target_day):
  import tempfile
  import pyarrow.parquet as pq
  from unittest.mock import patch
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'controls.parquet';pq.write_table(pa.Table.from_pylist([dict(target_native_id=native,target_date=target_day,role='preceding',intrinio_security_id=native,trading_date=day,open='POISON_NEVER_PROJECT')]),path);real=pq.ParquetFile;calls=[]
   class Guard:
    def __init__(self,path):self.actual=real(path);self.schema_arrow=self.actual.schema_arrow;self.metadata=self.actual.metadata
    def read(self,columns=None,**kwargs):
     if columns is None:raise AssertionError('Original financial columns decoded before metadata rejection')
     calls.append(columns);return self.actual.read(columns=columns,**kwargs)
   c,_,_=fixture();c['kind']='invalid_existing'
   with patch.object(m.pq,'ParquetFile',Guard):
    with self.assertRaises(m.DataContractError):m.read_guarded_original_controls(path,[c])
   self.assertEqual(len(calls),1)

if __name__=='__main__':unittest.main()
