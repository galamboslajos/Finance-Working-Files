"""Independent synthetic review only; no research data reads."""
import unittest

from dmv2 import ret_repair_backtest as m
from dmv2.universe import DataContractError
import pandas as pd


def bar(native, day, price):
    return dict(intrinio_security_id=native, trading_date=day, frequency='daily',
        intraperiod=False, price_quality_status='accepted', raw_ohlc_bounds_valid=True,
        adjusted_ohlc_bounds_valid=True, open=price, high=price+1, low=price-1,
        close=price, adj_open=price/2, adj_high=(price+1)/2,
        adj_low=(price-1)/2, adj_close=price/2, factor=.5, split_ratio=None, dividend=None)


class IndependentListingControlReview(unittest.TestCase):
    def fixture(self):
        before=bar('old_common','2021-06-14',18.)
        source=bar('published_route','2021-06-15',5.5)
        source[m.SOURCE_NATIVE_COLUMN]='published_ads_context'
        target=dict(source,intrinio_security_id='execution_cusip_new_ads')
        literal=[]
        for b,context in ((before,'published_old_context'),(source,'published_ads_context')):
            for adjustment,prefix in (('raw',''),('all','adj_')):
                literal.append(dict(trading_date=b['trading_date'],adjustment=adjustment,
                    selected_source_security_id=context,
                    **{key:b[prefix+key] for key in ('open','high','low','close')}))
        proof=dict(intrinio_security_id='execution_cusip_new_ads',source_native='published_route',
            method=m.PUBLISHED_METHOD, factor_method=m.FACTOR_METHOD,
            control_coverage_policy=m.LISTING_CONTROL_POLICY, control_dates=[],
            predecessor_control_date='2021-06-14', stock_ratio=3.355,
            max_paired_residual=2e-4,max_control_relative_residual=2e-4)
        observations={'execution_cusip_new_ads':dict(source=pd.DataFrame([source]),target_controls=None,
            published_pairs=pd.DataFrame(literal),predecessor_control=dict(row=before))}
        return pd.DataFrame([before]),pd.DataFrame([target]),[proof],observations

    def test_new_ads_uses_own_raw_all_units_not_predecessor_price_scale(self):
        m.verify_quote_units(*self.fixture())

    def test_new_ads_predecessor_reference_must_match_original_cached_close(self):
        original,target,proofs,obs=self.fixture()
        obs['execution_cusip_new_ads']['predecessor_control']['row']['close']=1.
        with self.assertRaisesRegex(DataContractError,'predecessor raw-price control'):
            m.verify_quote_units(original,target,proofs,obs)

    def test_published_predecessor_raw_price_must_match_original(self):
        original,target,proofs,obs=self.fixture()
        obs['execution_cusip_new_ads']['published_pairs'].loc[0,'close']=1.
        with self.assertRaisesRegex(DataContractError,'predecessor raw-price control'):
            m.verify_quote_units(original,target,proofs,obs)

    def test_successor_raw_all_pair_cannot_be_rescaled_to_old_common(self):
        original,target,proofs,obs=self.fixture()
        for column in ('open','high','low','close','adj_open','adj_high','adj_low','adj_close'):
            obs['execution_cusip_new_ads']['source'].loc[0,column]*=3.355
            target.loc[0,column]*=3.355
        with self.assertRaisesRegex(DataContractError,'exact published raw/all copy'):
            m.verify_quote_units(original,target,proofs,obs)

    def test_missing_original_predecessor_cannot_pass_on_literal_rows_alone(self):
        original,target,proofs,obs=self.fixture()
        with self.assertRaisesRegex(DataContractError,'predecessor raw-price control'):
            m.verify_quote_units(original.iloc[:0],target,proofs,obs)

if __name__=='__main__': unittest.main()
