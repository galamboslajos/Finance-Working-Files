import unittest

import numpy as np
import pandas as pd

from continuous_walk_forward import apply_policy, conservative_mark, path_metrics
from analyze_walk_forward import monthly_returns, segment_curves
from evaluate_walk_forward import ARMS


class ContinuousWFOTests(unittest.TestCase):
    def fixture(self):
        dates=pd.bdate_range('2024-01-02',periods=5)
        f=pd.DataFrame({'date':dates-pd.Timedelta(days=1),'return_date':dates,'test_year':2024,
                        'complete':[True,False,True,False,True], 'nodes':500,'available_labels':[500,499,500,499,500]})
        for arm in ARMS:
            f[arm]=[.01,np.nan,-.02,np.nan,.01]
            f[arm+'_net_weight']=.2;f[arm+'_gross_weight']=.5
        events=[{'ticker':'ABMD','return_date':str(dates[1].date()),'cash_per_old_share':90.,'right_payout_cap':20.,'source':'fixture'},
                {'ticker':'WBA','return_date':str(dates[3].date()),'cash_per_old_share':90.,'right_payout_cap':20.,'source':'fixture'}]
        ledger=[];s=[]
        for i,e in enumerate(events):
            day=pd.Timestamp(e['return_date']);w=-.01 if i==0 else .01
            ledger.append({'status':'unresolved_contingent_right','label_date':day,'date':day-pd.Timedelta(days=1),
                'source_security_id':e['ticker'],'membership_ticker':e['ticker'],'prior_raw_close':100.,
                'cash_per_old_share':90.,'contingent_right_max':20.})
            zero=.003+w*(.9-1);face=.003+w*(1.1-1)
            for arm in ARMS:
                s.append({'return_date':day,'ticker':e['ticker'],'strategy':arm,'nodes':500,'position_weight':w,
                    'portfolio_return_claim_zero_mark':zero,'portfolio_return_claim_face_mark':face,
                    'one_day_return_lower':min(zero,face)})
        return f,pd.DataFrame(s),pd.DataFrame(ledger),{'events':events}

    def test_long_zero_short_cap_and_zero_exposure(self):
        self.assertEqual(conservative_mark(.01,35),0.)
        self.assertEqual(conservative_mark(-.01,35),35.)
        self.assertEqual(conservative_mark(0.,35),0.)
        with self.assertRaises(ValueError):conservative_mark(np.nan,35)

    def test_marks_contributions_without_zeroing_or_reweighting_portfolio(self):
        f,s,l,p=self.fixture();out,register,alternatives=apply_policy(f,s,l,p)
        self.assertTrue(out.complete.all())
        self.assertEqual(out.has_contingent_right_assumption.sum(),2)
        self.assertEqual(out.observed_or_recovered_complete.sum(),3)
        np.testing.assert_allclose(out.GMOM,[.01,.002,-.02,.002,.01])
        np.testing.assert_array_equal(out.nodes,f.nodes)
        np.testing.assert_array_equal(out.available_labels,f.available_labels)
        np.testing.assert_array_equal(out.GMOM_net_weight,f.GMOM_net_weight)
        self.assertEqual(set(register.conservative_mark_rule),{'short_full_cap','long_zero'})
        self.assertFalse(register.right_sold.any())
        self.assertFalse(register.subsequent_claim_pnl_credited.any())
        self.assertEqual(len(alternatives),4)
        self.assertNotIn('forward_return',out.columns)

    def test_curve_compounds_through_both_assumptions_without_resets(self):
        f,s,l,p=self.fixture();out,_,_=apply_policy(f,s,l,p)
        curves,segments=segment_curves(out)
        self.assertEqual(len(segments),1)
        curve=curves[curves.strategy.eq('GMOM')]
        self.assertEqual(curve.is_initial_capital.sum(),1)
        self.assertEqual(len(curve),len(out)+1)
        self.assertAlmostEqual(curve.segment_equity.iloc[-1],100*np.prod(1+out.GMOM))
        self.assertTrue(monthly_returns(out).complete.all())
        self.assertIsNotNone(path_metrics(out,'GMOM')['cagr'])

    def test_unconfigured_gap_is_rejected(self):
        f,s,l,p=self.fixture();f.loc[0,'complete']=False
        for arm in ARMS:f.loc[0,arm]=np.nan
        with self.assertRaises(ValueError):apply_policy(f,s,l,p)

    def test_already_observed_event_is_not_overwritten(self):
        f,s,l,p=self.fixture();f.loc[1,'complete']=True
        for arm in ARMS:f.loc[1,arm]=.015
        with self.assertRaises(ValueError):apply_policy(f,s,l,p)

    def test_duplicate_right_row_is_rejected(self):
        f,s,l,p=self.fixture();s=pd.concat([s,s.iloc[[0]]],ignore_index=True)
        with self.assertRaises(ValueError):apply_policy(f,s,l,p)


if __name__=='__main__':unittest.main()
