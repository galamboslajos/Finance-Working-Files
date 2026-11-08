"""Financial timing, calendar gaps and cross-sectional cohort checks."""
import sys
from pathlib import Path
import tempfile
import unittest

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from build_momentum_features import (MODEL_FEATURES, assemble_features,
                                     scan_daily, session_calendar)


def fixture():
    sessions = session_calendar("2023-11-01","2023-11-30")
    members, bars = [], []
    for stock in range(3):
        rate = 0.00001*(stock+1)
        for row in sessions.itertuples(index=False):
            members.append(dict(security_id=str(stock),signal_date=row.signal_date,
                has_conflicting_membership_series=False,membership_tickers=[f"S{stock}"]))
            opening = 100*np.exp((stock+1)*0.002*row.session_num)
            for b in range(row.expected_slots):
                ts = row.market_open_utc + pd.Timedelta(minutes=5*b)
                op,cl = opening*np.exp(rate*b),opening*np.exp(rate*(b+1))
                bars.append(dict(security_id=str(stock),trading_date=str(row.signal_date.date()),
                    ticker=f"S{stock}",bar_timestamp_utc=ts.isoformat(),
                    available_at_utc=(ts+pd.Timedelta(minutes=5)).isoformat(),
                    open=op,high=cl,low=op,close=cl,minute_bars=3,source="vintage-a"))
    return sessions,pd.DataFrame(members),pd.DataFrame(bars)


def features(sessions,members,bars):
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder)/"bars.parquet"
        bars.to_parquet(path,index=False)
        con = duckdb.connect()
        con.execute("SET TimeZone='UTC'")
        con.register("sessions",sessions)
        con.register("members",members)
        scan_daily(con,[path],progress=lambda *args,**kwargs:None)
        assemble_features(con)
        result = con.sql("SELECT * FROM features ORDER BY signal_date,security_id").df()
        con.close()
        return result


class FeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sessions,cls.members,cls.bars=fixture()

    def run_fixture(self,bars=None,members=None,sessions=None):
        return features(self.sessions if sessions is None else sessions,
                        self.members if members is None else members,
                        self.bars if bars is None else bars)

    def test_exact_endpoints_hour_anchor_and_paper15(self):
        result=self.run_fixture()
        date=self.sessions.signal_date.iloc[12]
        rows=result[result.signal_date.eq(date)]
        self.assertTrue(rows.prediction_eligible.all())
        for row in rows.itertuples(index=False):
            stock=int(row.security_id)
            self.assertAlmostEqual(row.ret_1h,np.expm1(0.00001*(stock+1)*12),places=12)
            for h in (1,3,5,10):
                # Both endpoint sessions here have the same normal-session length.
                self.assertAlmostEqual(getattr(row,f"ret_{h}d"),
                                       np.expm1((stock+1)*0.002*h),places=12)
        self.assertEqual(len(MODEL_FEATURES),15)
        for h in ("1h","1d","3d","5d","10d"):
            self.assertAlmostEqual(rows[f"z_ret_{h}"].mean(),0,places=10)
            self.assertAlmostEqual(rows[f"z_ret_{h}"].std(ddof=0),1,places=10)
            self.assertEqual(rows[f"mean_ret_{h}"].nunique(),1)
        self.assertTrue(result.loc[result.signal_date.lt(self.sessions.signal_date.iloc[10]),
                                   "prediction_eligible"].eq(False).all())

    def test_internal_missing_bar_blocks_window_without_compressing_dates(self):
        bars=self.bars.copy()
        missing_date=str(self.sessions.signal_date.iloc[5].date())
        mask=bars.security_id.eq("0") & bars.trading_date.eq(missing_date)
        bars=bars.drop(bars.index[mask][20])
        result=self.run_fixture(bars=bars)
        affected=result[result.security_id.eq("0") & result.signal_date.eq(
            self.sessions.signal_date.iloc[14])].iloc[0]
        self.assertFalse(affected.prediction_eligible)
        self.assertEqual(affected.missing_slots_10d,1)
        self.assertIn("incomplete_or_invalid_history",affected.exclusion_reasons)
        # At session 15 the hole is outside the required ten-session path.
        later=result[result.security_id.eq("0") & result.signal_date.eq(
            self.sessions.signal_date.iloc[15])].iloc[0]
        self.assertTrue(later.prediction_eligible)

    def test_future_prices_and_membership_do_not_change_past_features(self):
        baseline=self.run_fixture()
        cutoff=self.sessions.signal_date.iloc[12]
        members=self.members[self.members.signal_date.le(cutoff)].copy()
        bars=self.bars.copy()
        future=pd.to_datetime(bars.trading_date).gt(cutoff)
        bars.loc[future,["open","high","low","close"]]*=1000
        alternate=self.run_fixture(bars=bars,members=members)
        pd.testing.assert_frame_equal(
            baseline[baseline.signal_date.le(cutoff)].reset_index(drop=True),
            alternate.reset_index(drop=True))

    def test_membership_gap_and_missing_anchor_are_not_skipped(self):
        members=self.members.copy()
        date=self.sessions.signal_date.iloc[8]
        members=members[~(members.security_id.eq("0") & members.signal_date.eq(date))]
        bars=self.bars[~(self.bars.security_id.eq("0") & self.bars.trading_date.eq(str(date.date())))]
        result=self.run_fixture(bars=bars,members=members)
        affected=result[result.security_id.eq("0") & result.signal_date.eq(
            self.sessions.signal_date.iloc[18])].iloc[0]
        self.assertFalse(affected.prediction_eligible)
        self.assertIn("invalid_10_session_anchor",affected.exclusion_reasons)

    def test_source_transition_and_late_bar_are_excluded(self):
        bars=self.bars.copy()
        date=self.sessions.signal_date.iloc[12]
        mask=bars.security_id.eq("0") & pd.to_datetime(bars.trading_date).ge(date)
        bars.loc[mask,"source"]="vintage-b"
        late=bars.index[bars.security_id.eq("1") & bars.trading_date.eq(str(date.date()))][-1]
        bars.loc[late,"available_at_utc"]=(date.tz_localize("UTC")+pd.Timedelta(days=2)).isoformat()
        result=self.run_fixture(bars=bars)
        for stock,reason in [("0","source_change_in_lookback"),
                             ("1","incomplete_or_invalid_history")]:
            row=result[result.security_id.eq(stock) & result.signal_date.eq(date)].iloc[0]
            self.assertFalse(row.prediction_eligible)
            self.assertIn(reason,row.exclusion_reasons)

    def test_early_close_uses_actual_close_and_short_minute_buckets_are_valid(self):
        result=self.run_fixture()
        rows=result[result.signal_date.eq(pd.Timestamp("2023-11-24"))]
        self.assertTrue(rows.expected_slots.eq(42).all())
        self.assertTrue(rows.prediction_eligible.all())
        self.assertTrue(rows.short_minute_buckets_10d.gt(0).all())
        for row in rows.itertuples(index=False):
            self.assertAlmostEqual(row.ret_1h,np.expm1(0.00001*(int(row.security_id)+1)*12),places=12)

    def test_conflicting_membership_and_duplicate_slots_are_excluded(self):
        date=self.sessions.signal_date.iloc[12]
        members=self.members.copy()
        members.loc[members.security_id.eq("0") & members.signal_date.eq(date),
                    "has_conflicting_membership_series"]=True
        bars=pd.concat([self.bars,self.bars[self.bars.security_id.eq("1") &
            self.bars.trading_date.eq(str(date.date()))].iloc[[10]]],ignore_index=True)
        result=self.run_fixture(bars=bars,members=members)
        self.assertFalse(result[result.signal_date.eq(date) &
            result.security_id.isin(["0","1"])].prediction_eligible.any())


if __name__=="__main__":
    unittest.main()
