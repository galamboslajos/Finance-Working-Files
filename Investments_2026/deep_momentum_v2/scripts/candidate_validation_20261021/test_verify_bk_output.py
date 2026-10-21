import copy
import importlib.util
from pathlib import Path
import unittest
import pyarrow as pa

p=Path(__file__).with_name('verify_bk_output.py')
spec=importlib.util.spec_from_file_location('independent_bk_verifier',p)
qa=importlib.util.module_from_spec(spec)
spec.loader.exec_module(qa)

def quote(day='2008-01-04'):
    return {'intrinio_security_id':'sec_zq1J05','trading_date':day,
            'open':10.,'high':12.,'low':9.,'close':11.,'volume':100.,
            'adj_open':5.,'adj_high':6.,'adj_low':4.5,'adj_close':5.5,'adj_volume':100.,
            'factor':.5,'split_ratio':1.,'dividend':0.}

class IndependentQATests(unittest.TestCase):
    def run_rows(self,rows,sessions=None):
        return qa.verify(pa.Table.from_pylist(rows),sessions or {r['trading_date'] for r in rows})
    def test_valid_price_volume_and_factor(self):
        x=self.run_rows([quote()])
        self.assertFalse(x['raw_OHLC_failures'])
        self.assertFalse(x['adjusted_OHLC_failures'])
        self.assertFalse(x['within_row_adjustment_ratio_failures'])
    def test_bracket_and_adjustment_failure_distinguished(self):
        q=quote(); q['high']=9.
        x=self.run_rows([q])
        self.assertEqual(x['raw_OHLC_failures'],['2008-01-04'])
        self.assertFalse(x['adjusted_OHLC_failures'])
        self.assertEqual(x['within_row_adjustment_ratio_failures'],['2008-01-04'])
    def test_volume_and_nonfinite_price_reported(self):
        q=quote(); q['volume']=-1.; q['adj_volume']=0.; q['close']=float('nan')
        x=self.run_rows([q])
        self.assertEqual(x['volume_negative']['volume'],1)
        self.assertEqual(x['volume_zero']['adj_volume'],1)
        self.assertEqual(x['price_null_or_nonfinite']['close'],1)
    def test_scope_and_duplicate_keys_rejected(self):
        q=quote('2024-01-02')
        with self.assertRaises(AssertionError): self.run_rows([q])
        with self.assertRaises(AssertionError): self.run_rows([quote(),quote()])
        q=quote(); q['intrinio_security_id']='another'
        with self.assertRaises(AssertionError): self.run_rows([q])
    def test_missing_session_not_stale_daily_jump(self):
        a=quote('2008-01-04'); b=quote('2008-01-08'); b['adj_close']=9.
        x=self.run_rows([a,b],{'2008-01-04','2008-01-07','2008-01-08'})
        self.assertEqual(x['missing_sessions'],['2008-01-07'])
        self.assertFalse(x['daily_jump_over_30pct_dates'])
    def test_factor_comparison_is_diagnostic(self):
        q=quote(); q['factor']=1.; q['dividend']=.2; q['split_ratio']=2.
        x=self.run_rows([q])
        self.assertEqual(x['source_factor_vs_adj_close_ratio_mismatch_dates'],['2008-01-04'])
        self.assertEqual(x['positive_dividend_dates'],['2008-01-04'])
        self.assertEqual(x['split_ratio_nontrivial_dates'],['2008-01-04'])

if __name__=='__main__': unittest.main()
