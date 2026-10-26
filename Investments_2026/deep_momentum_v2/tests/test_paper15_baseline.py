"""Synthetic prices/metadata only; no empirical estimator attempts."""
from pathlib import Path
import importlib.util,sys,unittest
import numpy as np,pandas as pd
from dmv2.paper15_baseline import strict_history,build_panels as core_build_panels,deciles,ret_payoffs,select_ret,FEATURES,PROBS,shift,executable_return

def build_panels(*args):
    cohorts,valid,days,closes,natives,months,years=args
    prices=np.zeros((len(days),len(natives),4))
    for j in range(len(natives)):
        value=100+j+np.arange(len(days))*(j+1)*.01
        prices[:,j,:]=value[:,None]
    q=np.repeat(valid[:,:,None],2,axis=2)
    return core_build_panels(cohorts,valid,days,closes,natives,months,years,prices=prices,quote_validity=q,exit_calendar=['2024-01-02'])

class Rules(unittest.TestCase):
 def test_history_matches_years_and_ignores_future(self):
  days=list(pd.bdate_range('2000-01-01','2023-12-31').strftime('%Y-%m-%d'));v=np.ones((len(days),1),bool)
  v[days.index('2012-01-03'),0]=False
  self.assertTrue(strict_history(v,days,0,'2019-12-31',5)['history_complete'])
  self.assertFalse(strict_history(v,days,0,'2019-12-31',10)['history_complete'])
  v[days.index('2020-01-02'),0]=False
  self.assertTrue(strict_history(v,days,0,'2019-12-31',5)['history_complete'])
  with self.assertRaises(ValueError):strict_history(v,days,0,'2024-01-02',5)
 def test_decile_ties_nulls_and_permutation(self):
  v=np.r_[np.arange(20,dtype=float),np.nan];labels=deciles(v)
  self.assertTrue(np.isnan(labels[-1]));self.assertEqual(labels[0],10);self.assertEqual(labels[19],1)
  self.assertTrue(np.array_equal(labels[:-1],deciles(v[:-1][::-1])[::-1]));self.assertTrue((deciles([1]*20)==6).all())
  self.assertTrue(np.isnan(deciles([1,2,3])).all())
 def fixture(self):
  days=list(pd.bdate_range('2000-01-01','2023-12-31').strftime('%Y-%m-%d'));months=sorted(set(d[:7] for d in days));natives=['N'+str(j) for j in range(20)]
  prices=np.array([[(100+j)*(1+.001*(j+1))**m for j in range(20)] for m in range(len(months))]);valid=np.ones((len(days),20),bool)
  rows=[dict(source_security_id=f'S{j:02d}',signal_date=s,intrinio_security_id=natives[j],identity_eligible=True) for s in ['2019-12-31','2020-01-31'] for j in range(20)]
  return pd.DataFrame(rows),valid,days,prices,natives,months
 def test_only15features_and_future_does_not_change_past_eligibility_or_X(self):
  args=self.fixture();a,cov=build_panels(*args,5);bargs=list(args);bargs[3]=args[3].copy();bargs[3][args[5].index('2020-01'),0]*=10;bargs[1]=args[1].copy();bargs[1][args[2].index('2020-01-02'),0]=False
  b,_=build_panels(*bargs,5);mask=a.signal_date.eq('2019-12-31')
  np.testing.assert_array_equal(a.loc[mask,list(FEATURES)],b.loc[mask,list(FEATURES)]);self.assertTrue(a.prediction_eligible.all());self.assertTrue(b.loc[mask,'prediction_eligible'].all())
  self.assertEqual(len(FEATURES),15);self.assertEqual(cov.original_members.tolist(),[20,20]);self.assertFalse(np.array_equal(a.loc[mask,'target_class'],b.loc[mask,'target_class']))
 def test_one_past_gap_does_not_remove_original_cohort_row(self):
  args=list(self.fixture());args[1][args[2].index('2019-12-30'),0]=False
  panel,cov=build_panels(*args,5);self.assertEqual(len(panel),40);self.assertFalse(panel.loc[panel.source_security_id.eq('S00'),'prediction_eligible'].any());self.assertTrue(panel.loc[panel.source_security_id.eq('S00'),'target_class'].isna().all())
 def test_RET_60months_gate_and_future_invariance(self):
  rows=[dict(outcome_month=shift('2019-12',j),outcome_end=shift('2019-12',j)+'-28',target_class=k,label_return=k/100,label_available_at=shift('2019-12',j+1)+'-02') for j in range(-60,0) for k in range(1,11)]
  p=pd.DataFrame(rows);mu,cells=ret_payoffs(p,'2019-12-31',5);np.testing.assert_allclose(mu,np.arange(1,11)/100);self.assertEqual(len(cells),600)
  bad=p.iloc[1:];self.assertIsNone(ret_payoffs(bad,'2019-12-31',5)[0]);self.assertIsNone(ret_payoffs(p,'2019-12-31',10)[0])
  future=pd.DataFrame([dict(outcome_month='2020-01',outcome_end='2020-01-31',target_class=1,label_return=1e9,label_available_at='2020-02-03')])
  self.assertEqual(mu,ret_payoffs(pd.concat([p,future]),'2019-12-31',5)[0])
 def test_RET_selection_is_signal_only_balanced_and_stable(self):
  rows=[]
  for j in range(22):
   row=dict(source_security_id=f'S{j:02d}',signal_date='2020-01-31',intrinio_security_id='N'+str(j),prediction_eligible=True)
   row.update({n:float(k==j%10) for k,n in enumerate(PROBS)});rows.append(row)
  pred=pd.DataFrame(rows);w=[dict(signal_date='2020-01-31',outcome_month='2020-02',outcome_start='2020-02-03',outcome_end='2020-02-28')]
  ledger,s,rb,m=select_ret(pred,{'2020-01-31':list(np.arange(10,dtype=float))},w)
  self.assertEqual(len(s),20);self.assertAlmostEqual(s.intended_weight.sum(),0);self.assertEqual(len(set(s.source_security_id)),20)
  self.assertEqual(set(s.side),{'long','short'});self.assertEqual(m.iloc[0].selected_per_side,10)
  pred['label_return']=0
  with self.assertRaises(ValueError):select_ret(pred,{'2020-01-31':[0.]*10},w)
 def test_controller_freezes16_fits_and_independent_checks(self):
  p=Path(__file__).parents[1]/'scripts/run_paper15_baseline.py';spec=importlib.util.spec_from_file_location('_baseline_controller_test',p);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
  phases=m.schedule();self.assertEqual(sum(p[0]=='fit' for p in phases),16);self.assertLess(next(i for i,p in enumerate(phases) if p[0]=='verify_data'),next(i for i,p in enumerate(phases) if p[0]=='fit'))
  self.assertEqual(phases[-1][0],'verify_backtest')

 def test_executable_return_uses_opens_and_checks_interior_daily_marks(self):
  days=['2020-01-02','2020-01-03','2020-02-03'];p=np.array([[[100.,99.,50.,49.5]],[[90.,80.,45.,40.]],[[110.,109.,55.,54.5]]]);q=np.ones((3,1,2),bool)
  value,status,marks,actions=executable_return(p,q,days,['A'],'A',days[0],days[-1])
  self.assertAlmostEqual(value,.1);self.assertEqual(status,'ok');self.assertEqual(marks,5)
  self.assertAlmostEqual(np.prod([45/50,40/45,55/40])-1,value)
  q[1,0,1]=False;self.assertTrue(np.isnan(executable_return(p,q,days,['A'],'A',days[0],days[-1])[0]))
 def test_cash_entitlement_uses_raw_units_without_inventing_retired_prices(self):
  days=['2020-01-02','2020-01-03','2020-02-03'];p=np.full((3,1,4),np.nan);p[0,0]=[100,100,50,50];q=np.zeros((3,1,2),bool);q[0,0]=True
  event=dict(predecessor_id='A',successor_id=None,effective_date=days[1],cash_per_raw_share=110.,stock_ratio=0.)
  r=executable_return(p,q,days,['A'],'A',days[0],days[-1],[event]);self.assertAlmostEqual(r[0],.1);self.assertEqual(r[3],1)
 def test_mixed_conversion_and_protected_exit(self):
  days=['2020-01-02','2020-01-03','2020-02-03'];p=np.full((3,2,4),np.nan);q=np.zeros((3,2,2),bool)
  p[0,0]=[100,100,50,50];q[0,0]=True;p[1,1]=[40,40,20,20];p[2,1]=[50,50,25,25];q[1:,1]=True
  e=dict(predecessor_id='A',successor_id='B',effective_date=days[1],cash_per_raw_share=20.,stock_ratio=2.)
  self.assertAlmostEqual(executable_return(p,q,days,['A','B'],'A',days[0],days[-1],[e])[0],.2)
  r=executable_return(p,q,days,['A','B'],'A',days[0],'2024-01-02',[e]);self.assertTrue(np.isnan(r[0]));self.assertEqual(r[1],'protected_scheduled_endpoint')

if __name__=='__main__':unittest.main()
