"""Strict boundary, nine gaps, clocks, units, provenance and exact missingness."""
from pathlib import Path
import sys,tempfile,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
import pandas as pd
import duckdb
from session_window_features import (FEATURES,DAY_NAMES,make_grid,daily_descriptors,window_matrix,ColumnAudit)
from build_session_window_features import read_month
from build_momentum_features import session_calendar


def fixture(days=12,early=False):
    rng=np.random.default_rng(51);shape=(days,78)
    n=np.full(days,78);n[-1]=42 if early else 78
    grid={name:np.zeros(shape,float) for name in ('open','high','low','close','raw_open','raw_high','raw_low','raw_close','volume','vwap','minute_bars','raw_minute_bars')}
    grid.update({name:np.ones(shape,int) for name in ('count','raw_count','source','raw_source')})
    grid.update({name:np.ones(shape,bool) for name in ('time_valid','raw_time_valid')})
    for day in range(days):
        o=100+day+np.r_[0,np.cumsum(rng.normal(0,.07,77))]
        c=o+rng.normal(.01,.05,78);h=np.maximum(o,c)+.08;l=np.minimum(o,c)-.08
        for name,a in [('open',o),('high',h),('low',l),('close',c)]:grid[name][day]=a;grid['raw_'+name][day]=a
        grid['volume'][day]=1000*(day+1);grid['vwap'][day]=(o+c)/2
        grid['minute_bars'][day]=5;grid['raw_minute_bars'][day]=5
    return grid,n


def matrix(grid,n):return window_matrix(daily_descriptors(grid,n))


class WindowFeatures(unittest.TestCase):
    def test_feature_names_and_exactly_nine_gaps(self):
        self.assertEqual(len(DAY_NAMES),24);self.assertEqual(len(FEATURES),267)
        self.assertEqual(len(set(FEATURES)),267)
        gaps=[s for s in FEATURES if s.endswith('_gap')]
        self.assertEqual(gaps,[f'lag{d:02}_gap' for d in range(1,10)])
        self.assertFalse(any(s.startswith('lag10_gap') for s in FEATURES))

    def test_future_and_prewindow_perturbations_cannot_change_chosen_window(self):
        g,n=fixture();before=matrix(g,n)
        changed={k:v.copy() for k,v in g.items()}
        # Window ending at day 10 includes days 1..10; days 0 and 11 are outside.
        for name in ('open','high','low','close','raw_open','raw_high','raw_low','raw_close','vwap','volume'):
            changed[name][[0,11]]=1e10
        changed['source'][[0,11]]=99;changed['raw_source'][[0,11]]=99
        after=matrix(changed,n)
        np.testing.assert_array_equal(before[0][1],after[0][1])
        np.testing.assert_array_equal(before[1][1],after[1][1])

    def test_first_10_minute_anchor_and_followthrough_exclude_opening_window(self):
        g,n=fixture(10);d=daily_descriptors(g,n);x,_,stats=window_matrix(d)
        scale=stats['window_risk_unit'][0]
        opening=np.log(g['close'][-1,1]/g['open'][-1,0])
        self.assertAlmostEqual(x[0,FEATURES.index('lag01_open_10m')],opening/scale,places=6)
        expected=np.sign(opening)*np.log(g['close'][-1,77]/g['close'][-1,1])/scale
        self.assertAlmostEqual(x[0,FEATURES.index('lag01_follow_10m')],expected,places=6)

    def test_gap_anchors_are_internal_adjacent_sessions(self):
        g,n=fixture(10);x,_,stats=matrix(g,n);scale=stats['window_risk_unit'][0]
        for lag in range(1,10):
            day=10-lag
            expected=np.log(g['open'][day,0]/g['close'][day-1,77])/scale
            self.assertAlmostEqual(x[0,FEATURES.index(f'lag{lag:02}_gap')],expected,places=6)

    def test_price_and_quantity_unit_changes_cancel(self):
        g,n=fixture(10);before=matrix(g,n)[0]
        changed={k:v.copy() for k,v in g.items()}
        for name in ('open','high','low','close','raw_open','raw_high','raw_low','raw_close','vwap'):changed[name]*=7
        changed['volume']*=1000
        np.testing.assert_allclose(before,matrix(changed,n)[0],rtol=1e-5,atol=1e-6,equal_nan=True)

    def test_relative_activity_uses_other_nine_sessions_only(self):
        g,n=fixture(10);d=daily_descriptors(g,n);x,_,_=window_matrix(d)
        a=d['values']['activity_05m'];expected=a[-1]/np.median(a[:-1])
        self.assertAlmostEqual(x[0,FEATURES.index('lag01_activity_05m')],expected,places=6)

    def test_missing_calendar_day_is_not_compressed_or_filled(self):
        g,n=fixture(11);g['count'][5]=0;g['raw_count'][5]=0
        x,reasons,stats=matrix(g,n)
        self.assertFalse(stats['window_price_complete'].any());self.assertTrue(np.isnan(x[:,FEATURES.index('lag01_open_05m')]).all())
        self.assertTrue((stats['missing_slots_10']==78).all())
        self.assertTrue((reasons[:,FEATURES.index('lag01_open_05m')]==1).all())

    def test_source_break_blocks_cross_session_normalization(self):
        g,n=fixture(10);g['source'][5]=2;g['raw_source'][5]=2
        x,reasons,stats=matrix(g,n)
        self.assertFalse(stats['source_consistent'][0])
        self.assertEqual(reasons[0,FEATURES.index('lag01_open_05m')],2)
        self.assertTrue(np.isnan(x[0,FEATURES.index('lag01_activity_05m')]))
        self.assertTrue(np.isfinite(x[0,FEATURES.index('lag01_realized_vol')]))

    def test_raw_adjusted_source_mismatch_invalidates_activity(self):
        g,n=fixture(10);g['raw_source'][4]=2
        x,reasons,stats=matrix(g,n)
        self.assertFalse(stats['window_activity_complete'][0]);self.assertTrue(stats['window_price_complete'][0])
        self.assertEqual(reasons[0,FEATURES.index('lag01_activity_05m')],7)

    def test_early_close_excludes_after_close_poison(self):
        g,n=fixture(10,early=True);before=matrix(g,n)
        for name in ('open','high','low','close','volume','vwap'):g[name][-1,42:]=1e12
        g['source'][-1,42:]=999
        after=matrix(g,n)
        np.testing.assert_array_equal(before[0],after[0]);np.testing.assert_array_equal(before[1],after[1])

    def test_flat_opening_and_zero_gap_are_explicit_not_filled(self):
        g,n=fixture(10)
        g['close'][-1,0]=g['open'][-1,0]
        g['open'][-1,0]=g['close'][-2,77]
        g['close'][-1,0]=g['open'][-1,0]
        g['low'][-1,0]=min(g['low'][-1,0],g['open'][-1,0]-.1)
        g['high'][-1,0]=max(g['high'][-1,0],g['open'][-1,0]+.1)
        x,why,_=matrix(g,n)
        self.assertEqual(why[0,FEATURES.index('lag01_persist_05m')],4)
        self.assertEqual(x[0,FEATURES.index('lag01_gap')],0)
        self.assertEqual(why[0,FEATURES.index('lag01_gap_filled')],6)

    def test_zero_risk_and_zero_volume_reference_stay_undefined(self):
        g,n=fixture(10)
        for name in ('open','high','low','close'):g[name][:]=100
        g['volume'][:]=0
        x,why,_=matrix(g,n)
        self.assertEqual(why[0,FEATURES.index('lag01_open_05m')],3)
        self.assertEqual(why[0,FEATURES.index('lag01_efficiency')],5)
        self.assertEqual(why[0,FEATURES.index('lag01_activity_05m')],8)
        self.assertFalse(np.isinf(x).any())

    def test_duplicate_hash_diagnostics_include_nan_masks_and_signed_zero(self):
        a=ColumnAudit(['a','b','c']);a.update(np.array([[1,1,1],[np.nan,np.nan,2],[0,-0.,0]],np.float32))
        self.assertEqual(a.result()['duplicate_groups'],[['a','b']])

    def test_late_adjusted_bar_cannot_leak_through_activity(self):
        g,n=fixture(10);g['time_valid'][-1,3]=False
        x,why,stats=matrix(g,n)
        self.assertFalse(stats['window_activity_complete'][0]);self.assertFalse(stats['window_price_complete'][0])
        self.assertTrue(np.isnan(x[0,FEATURES.index('lag01_activity_05m')]))

    def test_sql_timestamps_duplicates_and_slot_boundaries(self):
        calendar=session_calendar('2020-11-27','2020-11-27')
        base=pd.Timestamp(calendar.market_open_utc.iloc[0])
        records=[]
        for k in range(42):
            ts=base+pd.Timedelta(minutes=5*k)
            records.append(dict(security_id='A',trading_date='2020-11-27',bar_timestamp_utc=ts.isoformat(),
                available_at_utc=(ts+pd.Timedelta(minutes=5)).isoformat(),open=100.,high=101.,low=99.,close=100.1,volume=1000,vwap=100.,minute_bars=5,source='s'))
        with tempfile.TemporaryDirectory() as temp:
            a=Path(temp)/'a.parquet';r=Path(temp)/'r.parquet'
            pd.DataFrame(records).to_parquet(a);pd.DataFrame(records).to_parquet(r)
            con=duckdb.connect();con.register('sessions',calendar)
            f=read_month(con,a,r);self.assertEqual(len(f),42);self.assertTrue(f.time_valid.all())
            self.assertEqual(f.slot.max(),41)
            bad=pd.DataFrame(records+[records[0]])
            bad.loc[1,'available_at_utc']=(base+pd.Timedelta(minutes=11)).isoformat()
            bad.to_parquet(a);f=read_month(con,a,r)
            self.assertEqual(f.loc[f.slot.eq(0),'count'].iloc[0],2)
            self.assertFalse(f.loc[f.slot.eq(1),'time_valid'].iloc[0])
            con.close()

if __name__=='__main__':unittest.main()
