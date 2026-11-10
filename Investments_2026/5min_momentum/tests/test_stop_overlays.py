"""Exit semantics, empirical optimization, causal scales and dense calendars."""
from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
import pandas as pd
from stop_loss_math import (replay,net_contribution,response_segments,calibrate,
                           prior_context,select_past,scale_price)
from backtest_stop_overlays import annotate_positions
from build_momentum_features import session_calendar

def path(o=(100,99,98),h=(101,100,100),l=(99,97,97),c=(100,98,99)):
    return dict(open=np.array(o,float),high=np.array(h,float),low=np.array(l,float),close=np.array(c,float))

def daily_fixture():
    dates=pd.bdate_range('2020-01-01',periods=30)
    daily=pd.DataFrame(dict(security_id='A',trade_date=dates,path_valid=True,
        source='source1',opening_source='source1',session_open=101.,session_high=103.,
        session_low=99.,session_close=100.))
    return daily,pd.DataFrame(dict(trade_date=dates))

class StopTests(unittest.TestCase):
    def test_long_short_first_bar_and_gap_through(self):
        p=path()
        r=replay(p,1,1);self.assertEqual(r['exit_slot'],0);self.assertAlmostEqual(r['raw_return'],-.01)
        p=path(o=(100,95),h=(101,99),l=(99,94),c=(100,98))
        r=replay(p,1,3);self.assertTrue(r['gap_through']);self.assertEqual(r['fill_price'],95)
        p=path(o=(100,105),h=(101,106),l=(99,102),c=(100,103))
        r=replay(p,-1,3);self.assertTrue(r['gap_through']);self.assertEqual(r['fill_price'],105)
        self.assertAlmostEqual(r['directional_return'],-.05)

    def test_no_stop_reproduces_close_and_exit_notional(self):
        p=path();r=replay(p,1,np.inf)
        self.assertFalse(r['stopped']);self.assertAlmostEqual(r['raw_return'],-.01)
        self.assertAlmostEqual(net_contribution(-.01,.25,5),-.0025-.25*1.99*.0005)
        self.assertAlmostEqual(net_contribution(-.01,-.25,5),.0025-.25*1.99*.0005)

    def test_fixed_stop_does_not_need_high_low_ordering(self):
        p=path();a=replay(p,1,2)
        p['close'][1]=99.5
        b=replay(p,1,2)
        self.assertEqual(a,b)
        with self.assertRaises(ValueError):replay(p,1,0)

    def test_empirical_sweep_matches_brute_force_including_gaps(self):
        rng=np.random.default_rng(5)
        segments=[];cases=[]
        for i in range(40):
            o=np.r_[100,100+np.cumsum(rng.normal(0,2,11))]
            c=o+rng.normal(0,.3,len(o));h=np.maximum(o,c)+rng.uniform(0,2,len(o));l=np.minimum(o,c)-rng.uniform(0,2,len(o))
            p=path(o,h,l,c);direction=1 if i%2 else -1
            scale=.01+.002*(i%3);weight=direction*.25/40
            seg=response_segments(p,direction,scale,weight,1)
            segments.append(seg);cases.append((p,direction,scale,weight))
        result=calibrate(segments)
        ks=np.unique(np.concatenate([s[0] for s in segments]))
        scores=[sum(net_contribution(replay(p,d,k*scale*100)['raw_return'],w,1)
                    for p,d,scale,w in cases) for k in ks]
        baseline=sum(net_contribution(p['close'][-1]/100-1,w,1) for p,d,s,w in cases)
        self.assertAlmostEqual(result['score'],max(max(scores),baseline),places=11)
        actual=baseline if np.isinf(result['multiplier']) else sum(net_contribution(
            replay(p,d,result['multiplier']*s*100)['raw_return'],w,1) for p,d,s,w in cases)
        self.assertAlmostEqual(actual,result['score'],places=11)

    def test_no_stop_wins_when_protection_adds_no_value(self):
        p=path(o=(100,101),h=(102,103),l=(99,100),c=(101,102))
        choice=calibrate([response_segments(p,1,.02,.25,1)])
        self.assertTrue(np.isinf(choice['multiplier']))

    def test_atr_units_and_future_ohlc_perturbation(self):
        d,c=daily_fixture();original=prior_context(d,c)
        self.assertAlmostEqual(original.loc[15,'atr_prior'],4)
        self.assertAlmostEqual(original.loc[15,'gap_price'],1)
        changed=d.copy();changed.loc[15:,'session_high']=100000;changed.loc[15:,'session_low']=.1
        changed.loc[15:,'session_close']=300
        future=prior_context(changed,c)
        self.assertEqual(original.loc[15,'atr_prior'],future.loc[15,'atr_prior'])
        self.assertEqual(original.loc[15,'gap_price'],future.loc[15,'gap_price'])
        scaled=d.copy()
        for k in ['session_open','session_high','session_low','session_close']:scaled[k]*=7
        scaled=prior_context(scaled,c)
        self.assertAlmostEqual(scaled.loc[15,'atr_prior']/scaled.loc[15,'session_open'],4/101)
        self.assertEqual(scale_price('hybrid',4,7),7)

    def test_missing_session_not_compressed_and_source_break_rejected(self):
        d,c=daily_fixture();missing=prior_context(d.drop(index=10),c)
        self.assertTrue(np.isnan(missing.loc[20,'atr_prior']))
        self.assertTrue(np.isfinite(missing.loc[22,'atr_prior']))
        d.loc[15:,'source']='source2';d.loc[15:,'opening_source']='source2'
        switched=prior_context(d,c)
        self.assertTrue(np.isnan(switched.loc[15,'atr_prior']))
        self.assertTrue(np.isnan(switched.loc[15,'gap_price']))
        self.assertTrue(np.isfinite(switched.loc[26,'atr_prior']))

    def test_early_close_uses_only_scheduled_bars(self):
        p=path(o=np.full(42,100),h=np.full(42,101),l=np.full(42,99),c=np.full(42,100.5))
        r=replay(p,1,5);self.assertEqual(r['exit_slot'],41)
        self.assertAlmostEqual(r['raw_return'],.005)

    def test_quarter_calibration_never_uses_current_or_future_outcomes(self):
        frame=pd.DataFrame(dict(trade_date=pd.to_datetime(['2019-12-31','2020-01-02','2020-12-31','2021-01-04']),value=[0,1,2,3]))
        past=select_past(frame,'2021-01-01')
        self.assertEqual(past.value.tolist(),[1,2])
        changed=frame.copy();changed.loc[3,'value']=100000
        pd.testing.assert_frame_equal(past,select_past(changed,'2021-01-01'))

    def test_unknown_exit_and_zero_gap_preserve_selected_rows_with_reasons(self):
        p=pd.DataFrame(dict(security_id=['A','B'],trade_date=pd.to_datetime(['2020-01-02']*2),
            weight=[.25,-.25],n_per_side=2,future_oc_return=[.01,np.nan]))
        context=pd.DataFrame(dict(security_id=['A','B'],trade_date=pd.to_datetime(['2020-01-02']*2),
            path_valid=True,atr_prior=2.,gap_price=[0.,1.],session_open=100.,session_close=101.))
        checked=annotate_positions(p,context,{})
        self.assertEqual(len(checked),len(p))
        self.assertFalse(checked.overlay_observed.any())
        self.assertIn('undefined_zero',checked.exclusion_reason.iloc[0])
        self.assertIn('unknown_original_exit',checked.exclusion_reason.iloc[1])
        np.testing.assert_array_equal(checked.weight,p.weight)
        context['path_valid']=context.path_valid.astype(object)
        context.loc[1,'path_valid']=None
        nullable=annotate_positions(p,context,{})
        self.assertEqual(len(nullable),2)
        self.assertIn('incomplete_or_invalid',nullable.exclusion_reason.iloc[1])

    def test_dense_exchange_calendar_has_early_close_and_no_holiday(self):
        c=session_calendar('2020-11-25','2020-11-30')
        self.assertEqual(c.signal_date.dt.strftime('%Y-%m-%d').tolist(),['2020-11-25','2020-11-27','2020-11-30'])
        self.assertEqual(c.expected_slots.tolist(),[78,42,78])
        early=c[c.expected_slots.eq(42)].iloc[0]
        self.assertEqual((early.market_close_utc-early.market_open_utc).total_seconds(),42*300)

if __name__=='__main__':unittest.main()
