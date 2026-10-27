import unittest
import numpy as np,pandas as pd
from dmv2.fixed_forecast_completion import completed_quote_view


class CompletionTests(unittest.TestCase):
    def sources(self):
        old=pd.DataFrame([dict(trading_date='2022-12-29',intrinio_security_id='n',open=100.,close=100.,
           adj_open=95.,adj_close=95.,adj_high=96.,adj_low=94.,frequency='daily',intraperiod=False,
           price_quality_status='accepted',raw_ohlc_bounds_valid=True,adjusted_ohlc_bounds_valid=True)])
        point=pd.DataFrame([dict(trading_date='2022-12-30',intrinio_security_id='n',open=101.,close=102.,
            adj_open=96.,adj_close=97.,price_accepted=True)])
        return old,point

    def test_one_observed_point_without_high_low(self):
        old,point=self.sources();v=completed_quote_view(old,point,'2022-12-30','n')
        self.assertEqual(len(v),2);self.assertTrue(v.price_accepted.all())
        self.assertTrue(v.iloc[1][['adj_high','adj_low']].isna().all())
        self.assertEqual(v.iloc[1].adj_close,97.)

    def test_existing_rejection_is_preserved(self):
        old,point=self.sources();old.loc[0,'raw_ohlc_bounds_valid']=False
        v=completed_quote_view(old,point,'2022-12-30','n');self.assertFalse(v.iloc[0].price_accepted)

    def test_integer_one_cannot_promote_quality_acceptance(self):
        old,point=self.sources();old['raw_ohlc_bounds_valid']=old.raw_ohlc_bounds_valid.astype(object);old.loc[0,'raw_ohlc_bounds_valid']=1
        v=completed_quote_view(old,point,'2022-12-30','n');self.assertFalse(v.iloc[0].price_accepted)

    def test_no_existing_key_overwrite(self):
        old,point=self.sources();point.trading_date='2022-12-29'
        with self.assertRaises(ValueError):completed_quote_view(old,point,'2022-12-29','n')

    def test_invalid_or_unaccepted_point_rejected(self):
        for field,value in [('price_accepted',False),('price_accepted','False'),('adj_close',np.nan),('open',0.)]:
            old,point=self.sources();point[field]=value
            with self.assertRaises(ValueError):completed_quote_view(old,point,'2022-12-30','n')

    def test_high_low_cannot_be_invented_and_future_blocked(self):
        old,point=self.sources();point['adj_high']=100.
        with self.assertRaises(ValueError):completed_quote_view(old,point,'2022-12-30','n')
        old,point=self.sources();point.trading_date='2024-01-02'
        with self.assertRaises(ValueError):completed_quote_view(old,point,'2024-01-02','n')


if __name__=='__main__':unittest.main()
