"""Synthetic signed-stock entitlement and fixed ten-event preservation checks."""
import unittest,copy,json,hashlib,tempfile,importlib.util
from pathlib import Path
import pandas as pd
from dmv2.action_loading import EVENT_TERMS
from dmv2.sni_action import SNI_EVENT
from dmv2.vvc_action import VVC_EVENT
from dmv2.action_accounting import run_action_accounting
try:from dmv2 import tss_action as m
except ImportError:
 spec=importlib.util.spec_from_file_location('dmv2.tss_action',Path(__file__).with_name('tss_action.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def prices():
 rows=[]
 for day,native,value in [('2019-09-16',m.TSS_PREDECESSOR_ID,100.),('2019-09-17',m.TSS_PREDECESSOR_ID,100.),('2019-09-18',m.TSS_SUCCESSOR_ID,120.),('2019-09-19',m.TSS_SUCCESSOR_ID,121.),*[(d,'flat',100.) for d in ['2019-09-16','2019-09-17','2019-09-18','2019-09-19']]]:
  rows.append(dict(trading_date=day,intrinio_security_id=native,open=value,close=value,adj_open=value,adj_close=value,adj_high=value,adj_low=value,frequency='daily',intraperiod=False,price_quality_status='accepted'))
 return pd.DataFrame(rows)
def old10():return [dict(event_id=t+'-'+v[1],predecessor_id=v[0],effective_date=v[2],cash_per_raw_share=v[3],stock_ratio=v[4],arbitrary_audit={'preserve':t}) for t,v in EVENT_TERMS.items()]+[copy.deepcopy(SNI_EVENT),copy.deepcopy(VVC_EVENT)]
def evidence(directory):
 identity={'master':dict(intrinio_security_id=m.TSS_SUCCESSOR_ID,ticker='GPN',type='Ordinary Shares'),'links':[dict(intrinio_security_id=m.TSS_SUCCESSOR_ID,query_identifier_type='isin',query_identifier='US37940X1028',resolution_status='resolved')]}
 (directory/'successor_identity.json').write_text(json.dumps(identity));prices().to_parquet(directory/'predecessor_successor_controls.parquet',index=False)
 pins={}
 for name in ['successor_identity.json','predecessor_successor_controls.parquet']:
  p=directory/name;pins[name]=dict(path=name,size_bytes=p.stat().st_size,sha256='sha256:'+hashlib.sha256(p.read_bytes()).hexdigest())
 r=dict(schema='dmv2-tss-action-evidence/v1',event=copy.deepcopy(m.TSS_EVENT),no_new_quote_created=True,successor_class_verified=True,settlement_or_fractional_cash_invented=False,artifacts=pins);(directory/'tss_action_evidence.json').write_text(json.dumps(r));return r
class TssActionTests(unittest.TestCase):
 def account(self,side='long',p=None):
  dates=['2019-09-16','2019-09-17','2019-09-18','2019-09-19'];selected=pd.DataFrame([dict(signal_date='2019-08-30',execution_date=dates[0],source_security_id='tss',intrinio_security_id=m.TSS_PREDECESSOR_ID,side=side),dict(signal_date='2019-08-30',execution_date=dates[0],source_security_id='other',intrinio_security_id='flat',side='short' if side=='long' else'long')]);return run_action_accounting(selected,prices() if p is None else p,dates,rebalances=pd.DataFrame([dict(signal_date='2019-08-30',execution_date=dates[0])]),terminal_date=dates[-1],events=[m.TSS_EVENT])
 def test_long_exact_signed_stock_ratio_and_zero_ordinary_cash(self):
  r=self.account();self.assertEqual(r.status,'complete_proxy');a=r.action_events.iloc[0];self.assertEqual(a.signed_cash,0.);self.assertAlmostEqual(a.new_adjusted_units,.005*.8101);self.assertAlmostEqual(r.summary['final_nav'],1.+.005*(.8101*121.-100.));self.assertEqual(str(a.date.date()),'2019-09-18')
 def test_short_inherits_stock_liability(self):
  r=self.account('short');a=r.action_events.iloc[0];self.assertEqual(a.signed_cash,0.);self.assertAlmostEqual(a.new_adjusted_units,-.005*.8101);self.assertAlmostEqual(r.summary['final_nav'],1.-.005*(.8101*121.-100.))
 def test_both_paired_unit_bases_transport_once(self):
  p=prices();pred=p.intrinio_security_id.eq(m.TSS_PREDECESSOR_ID);succ=p.intrinio_security_id.eq(m.TSS_SUCCESSOR_ID)
  for f in ['adj_open','adj_close','adj_high','adj_low']:p.loc[pred,f]*=.5;p.loc[succ,f]*=.25
  r=self.account(p=p);a=r.action_events.iloc[0];self.assertAlmostEqual(a.signed_raw_entitlement,.005);self.assertAlmostEqual(a.new_adjusted_units,.005*.8101/.25);self.assertAlmostEqual(r.summary['final_nav'],1.+.005*(.8101*121.-100.))
 def test_action_is_not_fictitious_market_trade(self):
  r=self.account();self.assertFalse(r.trades.date.eq(pd.Timestamp('2019-09-18')).any())
 def test_legal_and_application_dates_distinct(self):
  self.assertEqual(m.TSS_EVENT['legal_effective_date'],'2019-09-17');self.assertEqual(m.TSS_EVENT['effective_date'],'2019-09-18')
 def test_cached_basis_accepts_accepted_observations(self):m.validate_tss_action_prices(prices())
 def test_no_successor_basis_refuses(self):
  p=prices();p=p[~((p.intrinio_security_id==m.TSS_SUCCESSOR_ID)&(p.trading_date=='2019-09-18'))]
  with self.assertRaises(m.DataContractError):m.validate_tss_action_prices(p)
 def test_post_retirement_tss_quote_refuses(self):
  p=prices();x=p.iloc[0].copy();x['trading_date']='2019-09-18';p=pd.concat([p,pd.DataFrame([x])],ignore_index=True)
  with self.assertRaises(m.DataContractError):m.validate_tss_action_prices(p)
 def test_future_dates_refused(self):
  p=prices();p.loc[0,'trading_date']='2020-01-02'
  with self.assertRaises(m.DataContractError):m.validate_tss_action_prices(p)
 def test_bad_paired_open_refused(self):
  p=prices();p.loc[(p.intrinio_security_id==m.TSS_SUCCESSOR_ID)&(p.trading_date=='2019-09-18'),'adj_open']=0.
  with self.assertRaises(m.DataContractError):m.validate_tss_action_prices(p)
 def test_small_exact_evidence_and_append_preserve_old10(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);evidence(root);before=old10();after=m.append_tss_action(before,root,prices=prices());self.assertEqual(after[:10],before);self.assertEqual(after[-1],m.TSS_EVENT);after[0]['arbitrary_audit']['preserve']='changed';self.assertNotEqual(after[0],before[0])
 def test_cannot_change_vvc(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);evidence(root);old=old10();old[-1]['cash_per_raw_share']=0.
   with self.assertRaises(m.DataContractError):m.append_tss_action(old,root)
 def test_cannot_omit_an_old_action(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);evidence(root)
   with self.assertRaises(m.DataContractError):m.append_tss_action(old10()[:-1],root)
 def test_evidence_tamper_refuses(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);evidence(root);(root/'successor_identity.json').write_text('{}')
   with self.assertRaises(m.DataContractError):m.load_tss_action(root)
 def test_share_ratio_not_proposed_merger_guess(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);r=evidence(root);r['event']['stock_ratio']=.81;(root/'tss_action_evidence.json').write_text(json.dumps(r))
   with self.assertRaises(m.DataContractError):m.load_tss_action(root)
if __name__=='__main__':unittest.main()
