"""Financial units, split invariance, calendar gaps and feature cutoff tests."""
import sys
from pathlib import Path
import tempfile
import unittest

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from volume_features import VOLUME_FEATURES,make_volume_features,opening_gap


def daily_fixture():
    dates=pd.bdate_range('2024-01-02',periods=24)
    minutes=np.repeat(390.,24)
    minutes[15]=210.
    dollars=1000*minutes
    return pd.DataFrame(dict(dollar_volume=dollars,last_hour_dollar_volume=np.repeat(60000.,24),
        signed_dollar_volume=.2*dollars,share_volume=dollars/100,
        closing_price_raw=np.repeat(102.,24),session_minutes=minutes,
        activity_valid=True,signed_valid=True),index=dates)


class VolumeTests(unittest.TestCase):
    def test_past_only_reference_does_not_absorb_current_spike(self):
        d=daily_fixture()
        d.loc[d.index[-1],'dollar_volume']*=100
        d.loc[d.index[-1],'last_hour_dollar_volume']*=100
        x=make_volume_features(d)
        self.assertAlmostEqual(x.iloc[-1].vol_relative_activity_10d,100.)
        self.assertAlmostEqual(x.iloc[-1].vol_relative_last_hour_10d,100.)

    def test_early_close_activity_rate_and_vwap_units(self):
        x=make_volume_features(daily_fixture())
        self.assertAlmostEqual(x.iloc[15].vol_relative_activity_10d,1.)
        self.assertAlmostEqual(x.iloc[15].vol_relative_last_hour_10d,1.)
        self.assertAlmostEqual(x.iloc[15].vol_activity_trend_3v7,1.)
        self.assertAlmostEqual(x.iloc[15].vol_signed_balance_1d,.2)
        self.assertAlmostEqual(x.iloc[15].vol_signed_balance_5d,.2)
        self.assertAlmostEqual(x.iloc[15].vol_close_vwap_gap,.02)

    def test_recent_three_versus_preceding_seven_and_weighted_balance(self):
        d=daily_fixture()
        d.loc[d.index[-3:],'dollar_volume']*=2
        d.loc[d.index[-3:],'signed_dollar_volume']*=2
        x=make_volume_features(d)
        self.assertAlmostEqual(x.iloc[-1].vol_activity_trend_3v7,2.)
        d.loc[d.index[-1],'signed_dollar_volume']=-d.iloc[-1].dollar_volume
        x=make_volume_features(d)
        self.assertAlmostEqual(x.iloc[-1].vol_signed_balance_1d,-1.)
        expected=d.signed_dollar_volume.iloc[-5:].sum()/d.dollar_volume.iloc[-5:].sum()
        self.assertAlmostEqual(x.iloc[-1].vol_signed_balance_5d,expected)

    def test_future_activity_does_not_change_past_values(self):
        d=daily_fixture()
        base=make_volume_features(d)
        changed=d.copy()
        changed.iloc[19:,changed.columns.get_loc('dollar_volume')]*=1000
        changed.iloc[19:,changed.columns.get_loc('closing_price_raw')]*=20
        other=make_volume_features(changed)
        pd.testing.assert_frame_equal(base.iloc[:19],other.iloc[:19])

    def test_missing_calendar_day_is_not_zero_or_a_shorter_window(self):
        d=daily_fixture()
        d.loc[d.index[15],'activity_valid']=False
        x=make_volume_features(d)
        self.assertTrue(np.isnan(x.iloc[15].vol_relative_activity_10d))
        self.assertTrue(np.isnan(x.iloc[19].vol_signed_balance_5d))
        self.assertTrue(np.isnan(x.iloc[23].vol_activity_trend_3v7))
        self.assertEqual(x.iloc[16].volume_reference_days,9)
        self.assertEqual(x.iloc[16].vol_relative_activity_10d,1.)

    def test_split_does_not_change_raw_dollar_volume_or_vwap_gap(self):
        d=daily_fixture()
        transformed=d.copy()
        # A 4-for-1 split: dollar notional stays fixed, shares x4 and prices /4.
        transformed.loc[d.index[12]:,'share_volume']*=4
        transformed.loc[d.index[12]:,'closing_price_raw']/=4
        np.testing.assert_allclose(make_volume_features(d)[VOLUME_FEATURES],
            make_volume_features(transformed)[VOLUME_FEATURES],equal_nan=True)

    def test_opening_gap_is_current_morning_and_missing_is_explicit(self):
        previous=pd.Series([100.,100.,100.,100.])
        opened=pd.Series([102.,98.,np.nan,105.])
        sources=pd.Series([True,True,True,False])
        g=opening_gap(previous,opened,sources)
        self.assertAlmostEqual(g.iloc[0],.02)
        self.assertAlmostEqual(g.iloc[1],-.02)
        self.assertTrue(g.iloc[2:].isna().all())
        self.assertTrue(opening_gap(previous,opened,sources,unique_open=False).isna().all())

    def test_observed_zero_activity_is_not_missing_volume(self):
        d=daily_fixture()
        d.loc[d.index[-1],['dollar_volume','last_hour_dollar_volume',
            'share_volume','signed_dollar_volume']]=0.
        x=make_volume_features(d).iloc[-1]
        self.assertEqual(x.vol_relative_activity_10d,0.)
        self.assertEqual(x.vol_relative_last_hour_10d,0.)
        self.assertTrue(np.isnan(x.vol_signed_balance_1d))
        self.assertTrue(np.isnan(x.vol_close_vwap_gap))


class ExtractionTests(unittest.TestCase):
    def fixture(self):
        from build_momentum_features import session_calendar
        sessions=session_calendar('2024-11-27','2024-11-29')
        rows=[]
        for s in sessions.itertuples(index=False):
            for ts in pd.date_range(s.market_open_utc,s.market_close_utc,
                                     freq='5min',inclusive='left'):
                rows.append(dict(security_id='A',ticker='AAA',trading_date=str(s.signal_date.date()),
                    bar_timestamp_utc=ts.isoformat(),available_at_utc=(ts+pd.Timedelta(minutes=5)).isoformat(),
                    open=10.,close=10.,vwap=10.,volume=100,trade_count=10,source='same'))
            for ts in [s.market_open_utc-pd.Timedelta(minutes=5),s.market_close_utc]:
                rows.append(dict(security_id='A',ticker='AAA',trading_date=str(s.signal_date.date()),
                    bar_timestamp_utc=ts.isoformat(),available_at_utc=(ts+pd.Timedelta(minutes=5)).isoformat(),
                    open=10.,close=10.,vwap=10.,volume=10**9,trade_count=10,source='same'))
        return sessions,pd.DataFrame(rows)

    def test_calendar_aggregation_excludes_extended_hours_and_handles_half_day(self):
        from build_volume_overnight_features import aggregate_volume_month
        sessions,bars=self.fixture()
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'bars.parquet'
            bars.to_parquet(path,index=False)
            c=duckdb.connect()
            c.execute("SET TimeZone='UTC'")
            c.register('sessions',sessions)
            aggregate_volume_month(c,path,first=True)
            rows=c.execute('SELECT * FROM daily_volume ORDER BY signal_date').df()
            c.close()
        self.assertEqual(rows.rth_rows.tolist(),[78,42])
        self.assertEqual(rows.dollar_volume.tolist(),[78000.,42000.])
        self.assertEqual(rows.last_hour_dollar_volume.tolist(),[12000.,12000.])
        self.assertEqual(rows.invalid_quantity_rows.tolist(),[0.,0.])

    def test_opening_input_ignores_first_bar_future_fields_and_later_source(self):
        from build_volume_overnight_features import aggregate_opening_month
        sessions,bars=self.fixture()
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'bars.parquet'
            c=duckdb.connect()
            c.execute("SET TimeZone='UTC'")
            c.register('sessions',sessions)
            bars.to_parquet(path,index=False)
            aggregate_opening_month(c,path,first=True)
            before=c.execute('SELECT * FROM opening ORDER BY trade_date').df()
            bars['close']=-999.
            bars['vwap']=12345.
            bars['volume']=99
            # A later source change must not enter the opening-source decision.
            times=pd.to_datetime(bars.bar_timestamp_utc,utc=True)
            opens=set(sessions.market_open_utc)
            bars.loc[~times.isin(opens),'source']='future-different'
            bars.to_parquet(path,index=False)
            c.execute('DROP TABLE opening')
            aggregate_opening_month(c,path,first=True)
            after=c.execute('SELECT * FROM opening ORDER BY trade_date').df()
            c.close()
        pd.testing.assert_frame_equal(before,after)


if __name__=='__main__':
    unittest.main()
