"""Cost-aware calibration, original objective and strictly prior outcomes."""
from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
import pandas as pd
from stop_loss_math import response_segments,calibrate,replay,net_contribution
from recalibrate_stop_costs import cost_segments,learn_quarter,replay_quarter,COSTS

def fixture():
    rng=np.random.default_rng(23);rows=[];paths={}
    dates=pd.to_datetime(['2019-01-03','2019-12-31','2020-01-02'])
    for date in dates:
        for sid in 'ABCDEF':
            o=np.r_[100,100+np.cumsum(rng.normal(0,1.2,4))]
            c=o+rng.normal(0,.4,5);h=np.maximum(o,c)+rng.uniform(.1,1.5,5);l=np.minimum(o,c)-rng.uniform(.1,1.5,5)
            paths[(sid,date)]=dict(open=o,high=h,low=l,close=c,timestamps=pd.date_range(date+pd.Timedelta(hours=14,minutes=30),periods=5,freq='5min',tz='UTC').to_numpy())
        for n in (2,3):
            ids=list('ABCDEF')[:n]+list('ABCDEF')[-n:]
            for j,sid in enumerate(ids):
                weight=(1 if j<n else -1)*.5/n
                rows.append(dict(security_id=sid,ticker=sid,trade_date=date,n_per_side=n,
                    weight=weight,session_open=100.,atr_prior=2.,gap_price=1.,
                    opening_available_at=date+pd.Timedelta(hours=14,minutes=35),
                    future_oc_return=paths[(sid,date)]['close'][-1]/100-1))
    return pd.DataFrame(rows),paths,dates

class Recalibration(unittest.TestCase):
    def test_affine_cost_transform_matches_direct_price_path_math(self):
        p,paths,dates=fixture()
        for row in p.iloc[:10].itertuples():
            direction=int(np.sign(row.weight));path=paths[(row.security_id,row.trade_date)]
            gross=response_segments(path,direction,.02,row.weight,0)
            for cost in (1,)+COSTS:
                converted=cost_segments(gross,direction,abs(row.weight),cost)
                direct=response_segments(path,direction,.02,row.weight,cost)
                for a,b in zip(converted,direct):np.testing.assert_allclose(a,b,rtol=0,atol=1e-14)

    def test_all_costs_match_brute_force_and_equal_N_utility(self):
        p,paths,dates=fixture();start=pd.Timestamp('2020-01-01')
        choices,days,count=learn_quarter(p,paths,'atr',start,dates)
        self.assertEqual(days,2);self.assertEqual(count,20)
        past=p[p.trade_date<start]
        candidates=np.unique(np.concatenate([response_segments(paths[(r.security_id,r.trade_date)],int(np.sign(r.weight)),.02,r.weight*.5/days,0)[0] for r in past.itertuples()]))
        for cost,choice in choices.items():
            def utility(k):
                return sum(net_contribution(replay(paths[(r.security_id,r.trade_date)],int(np.sign(r.weight)),k*2)['raw_return'],r.weight,cost)*.5/days for r in past.itertuples())
            scores=[utility(k) for k in candidates]+[utility(np.inf)]
            self.assertAlmostEqual(choice['score'],max(scores),places=12)
            self.assertAlmostEqual(choice['score'],utility(choice['multiplier']),places=12)
            self.assertGreaterEqual(choice['score']+1e-12,choice['baseline'])

    def test_future_labels_paths_and_scales_cannot_change_current_policy(self):
        p,paths,dates=fixture()
        before=learn_quarter(p,paths,'hybrid','2020-01-01',dates)
        p.loc[p.trade_date.ge('2020-01-01'),['future_oc_return','atr_prior','gap_price']]=1e9
        poisoned={key:({**value,'low':np.full(5,np.nan),'high':np.full(5,np.nan)} if key[1]>=pd.Timestamp('2020-01-01') else value) for key,value in paths.items()}
        after=learn_quarter(p,poisoned,'hybrid','2020-01-01',dates)
        self.assertEqual(before,after)

    def test_original_cash_dates_remain_in_equal_day_denominator(self):
        p,paths,dates=fixture();past_dates=dates[:2]
        a,days_a,_=learn_quarter(p,paths,'atr','2020-01-01',past_dates)
        b,days_b,_=learn_quarter(p,paths,'atr','2020-01-01',past_dates.append(pd.DatetimeIndex(['2019-06-03'])))
        self.assertEqual(days_a,2);self.assertEqual(days_b,3)
        for cost in COSTS:
            self.assertAlmostEqual(a[cost]['score']*2,b[cost]['score']*3,places=12)

    def test_threshold_is_shared_by_N_and_costs_are_charged_at_actual_exits(self):
        p,paths,dates=fixture();choices,_,_=learn_quarter(p,paths,'gap','2020-01-01',dates)
        current=p[p.trade_date.eq(dates[-1])]
        t=replay_quarter(current,paths,'gap',choices,pd.Period('2020Q1'),'fixture')
        self.assertEqual(len(t),30)
        self.assertTrue(t.groupby('cost_bps_per_side').multiplier.nunique(dropna=False).eq(1).all())
        for row in t.itertuples():
            self.assertAlmostEqual(row.net_contribution,net_contribution(row.raw_return,row.weight,row.cost_bps_per_side),places=14)
            self.assertAlmostEqual(row.trade_notional,abs(row.weight)*(1+row.fill_price/row.entry_price),places=14)
        np.testing.assert_allclose(t.weight.to_numpy(),np.tile(current.weight.to_numpy(),3))

if __name__=='__main__':unittest.main()
