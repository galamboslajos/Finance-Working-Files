import unittest
import numpy as np
import pandas as pd
from review_minor_closes import check_close, review_targets, feature_view


SESSIONS = pd.DatetimeIndex(['2012-01-03','2012-01-04','2012-01-05','2012-01-06'])


def bar(day, close=10., ratio=.5, bad=False):
    values = dict(intrinio_security_id='n',date=pd.Timestamp(day),close=close,
                  low=close-.1,high=close+.1,open=close+.5 if bad else close,
                  adj_close=round(close*ratio,4),adj_low=round((close-.1)*ratio,4),
                  adj_high=round((close+.1)*ratio,4),adj_open=round((close+.5 if bad else close)*ratio,4),
                  frequency='daily',intraperiod=False,factor=1.,split_ratio=1.,dividend=0.,
                  price_quality_status='provider_ohlc_anomaly' if bad else 'accepted',
                  quote_status='invalid_ohlcv' if bad else 'ok')
    return pd.Series(values)


class MinorCloseTests(unittest.TestCase):
    def test_bad_open_can_preserve_close(self):
        result=check_close(bar(SESSIONS[1],bad=True),bar(SESSIONS[0]))
        self.assertEqual(result['issue'],'open_range_anomaly_close_only')

    def test_corrupt_adjusted_close_is_refused(self):
        row=bar(SESSIONS[1],bad=True)
        row['adj_close']=20.
        with self.assertRaises(ValueError):check_close(row,bar(SESSIONS[0]))

    def test_undocumented_adjustment_jump_is_refused(self):
        row=bar(SESSIONS[1],ratio=.6,bad=True)
        with self.assertRaises(ValueError):check_close(row,bar(SESSIONS[0]))

    def test_recorded_dividend_factor_permits_matching_step(self):
        row=bar(SESSIONS[1],ratio=.5/.99,bad=True)
        row['factor']=.99;row['dividend']=.1
        self.assertEqual(check_close(row,bar(SESSIONS[0]))['issue'],'open_range_anomaly_close_only')

    def test_only_missing_low_is_permitted(self):
        row=bar(SESSIONS[1]);row['low']=np.nan;row['adj_low']=np.nan
        row['price_quality_status']='provider_incomplete_ohlc'
        self.assertEqual(check_close(row,bar(SESSIONS[0]))['issue'],'missing_low_close_only')
        row['adj_close']=np.nan
        with self.assertRaises(ValueError):check_close(row,bar(SESSIONS[0]))

    def test_missing_previous_session_blocks_review(self):
        context=pd.DataFrame([bar(SESSIONS[0]),bar(SESSIONS[2],bad=True)])
        targets=pd.DataFrame([dict(source_security_id='s',membership_ticker='X',intrinio_security_id='n',date=SESSIONS[2])])
        ledger,accepted=review_targets(context,targets,SESSIONS)
        self.assertFalse(ledger.iloc[0].admitted);self.assertFalse(accepted)

    def test_consecutive_reviews_and_future_prefix_invariance(self):
        context=pd.DataFrame([bar(SESSIONS[0]),bar(SESSIONS[1],bad=True),bar(SESSIONS[2],bad=True)])
        targets=pd.DataFrame([dict(source_security_id='s',membership_ticker='X',intrinio_security_id='n',date=d) for d in SESSIONS[1:3]])
        first,accepted=review_targets(context,targets,SESSIONS)
        future=pd.concat([context,pd.DataFrame([bar(SESSIONS[3],close=100.)])],ignore_index=True)
        second,accepted2=review_targets(future,targets,SESSIONS)
        pd.testing.assert_frame_equal(first,second)
        self.assertEqual(accepted,accepted2);self.assertEqual(len(accepted),2)

    def test_feature_admission_preserves_execution_rejection_and_prices(self):
        raw=pd.DataFrame([bar(SESSIONS[0]),bar(SESSIONS[1],close=11.,bad=True),bar(SESSIONS[2],close=12.)])
        accepted={('n',SESSIONS[1])}
        out=feature_view(raw,accepted,SESSIONS)
        self.assertTrue(out.iloc[1].feature_close_eligible)
        self.assertFalse(out.iloc[1].execution_bar_eligible)
        self.assertEqual(out.iloc[1].quote_status,'invalid_ohlcv')
        self.assertAlmostEqual(out.iloc[1].feature_return_1d,.1)
        np.testing.assert_array_equal(out.adj_close,raw.adj_close)

    def test_feature_returns_never_bridge_a_missing_session(self):
        raw=pd.DataFrame([bar(SESSIONS[0]),bar(SESSIONS[2])])
        out=feature_view(raw,set(),SESSIONS)
        self.assertTrue(np.isnan(out.iloc[1].feature_return_1d))


if __name__=='__main__':unittest.main()
