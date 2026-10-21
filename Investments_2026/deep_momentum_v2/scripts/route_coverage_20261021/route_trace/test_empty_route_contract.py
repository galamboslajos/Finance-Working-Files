"""Tiny fabricated fixtures only: empty native route != missing request."""
from types import SimpleNamespace
import unittest
import pyarrow as pa
from dmv2.loading import PriceLoader,ReadError
from dmv2.universe import build_snapshot,attach_quotes,DataContractError

DAY='2020-01-31'
NATIVE='sec_declared';ALTERNATIVE='sec_candidate_not_admitted'
PRICE_SCHEMA=pa.schema([('intrinio_security_id',pa.string()),('trading_date',pa.string()),('close',pa.float64())])

def snapshot(review=False):
    membership=pa.Table.from_pylist([dict(index_id='sp500',trading_date=DAY,security_id='source_original',
        membership_ticker='FAKE',is_member=True,membership_effective_date='2019-01-01',membership_end_date=None,
        membership_availability_date='2019-01-01',pit_policy='fixture',identity_confidence='fixture')])
    bridge=pa.Table.from_pylist([dict(source_product_id='sp_composite1500',source_security_id='source_original',
        unified_security_id='unified_original',bridge_method='fixture',bridge_confidence='fixture')])
    links=pa.Table.from_pylist([dict(unified_security_id='unified_original',intrinio_security_id=NATIVE,
        identity_review_required=review,resolution_status='resolved')])
    return build_snapshot(membership,bridge,links,as_of=DAY,universe='sp500',mode='historical_reconstruction')

def quotes(native=None):
    return pa.Table.from_pylist([] if native is None else [dict(intrinio_security_id=native,trading_date=DAY,close=10.)],schema=PRICE_SCHEMA)

class EmptyNativeRouteContracts(unittest.TestCase):
    def test_resolved_explicitly_requested_empty_route_is_missing_price(self):
        s=snapshot();got=attach_quotes(s,quotes(),requested_security_ids=[NATIVE]).table.to_pylist()
        self.assertEqual(len(got),1);self.assertEqual(got[0]['source_security_id'],'source_original')
        self.assertEqual(got[0]['intrinio_security_id'],NATIVE)
        self.assertEqual(got[0]['identity_status'],'resolved');self.assertEqual(got[0]['quote_status'],'missing_price')
        self.assertIsNone(got[0]['quote_close'])

    def test_unrequested_state_stays_distinct(self):
        got=attach_quotes(snapshot(),quotes(),requested_security_ids=[ALTERNATIVE]).table.to_pylist()[0]
        self.assertEqual(got['quote_status'],'not_requested')

    def test_alternative_native_quote_never_falls_back_to_original_route(self):
        got=attach_quotes(snapshot(),quotes(ALTERNATIVE),requested_security_ids=[NATIVE,ALTERNATIVE]).table.to_pylist()[0]
        self.assertEqual(got['intrinio_security_id'],NATIVE);self.assertEqual(got['quote_status'],'missing_price')
        self.assertIsNone(got['quote_close'])

    def test_extra_unrequested_candidate_quote_fails(self):
        with self.assertRaisesRegex(DataContractError,'outside.*declared'):
            attach_quotes(snapshot(),quotes(ALTERNATIVE),requested_security_ids=[NATIVE])

    def test_review_flag_provisional_admission_does_not_certify_route(self):
        got=attach_quotes(snapshot(True),quotes(),requested_security_ids=[NATIVE],
            identity_policy='provisional_unique_resolved_link').table.to_pylist()[0]
        self.assertEqual(got['quote_status'],'missing_price');self.assertTrue(got['identity_review_required'])
        self.assertEqual(got['source_identity_status'],'needs_review')
        self.assertTrue(got['identity_eligible'])

    def test_strict_rule_does_not_promote_review_flag_because_quote_exists(self):
        got=attach_quotes(snapshot(True),quotes(NATIVE),requested_security_ids=[NATIVE]).table.to_pylist()[0]
        self.assertEqual(got['quote_status'],'identity_unresolved');self.assertFalse(got['identity_eligible'])
        self.assertTrue(got['identity_review_required'])

    def test_price_loader_requests_only_exact_declared_native_ids(self):
        calls=[]
        class Reader:
            def read(self,*args,**kwargs):calls.append((args,kwargs));return 'fixture'
        loader=PriceLoader.__new__(PriceLoader);loader.reader=Reader()
        self.assertEqual(loader.load([NATIVE],DAY,DAY,['adj_close']),'fixture')
        args,kwargs=calls[0]
        self.assertEqual(args,('intrinio','intrinio_daily_prices'))
        self.assertEqual(kwargs['filters'],{'intrinio_security_id':(NATIVE,)})
        self.assertEqual(kwargs['key_columns'],('intrinio_security_id','trading_date'))
        self.assertTrue(kwargs['financial'])
        self.assertFalse(any(x in kwargs['filters'] for x in ('ticker','isin','figi','company_id')))

if __name__=='__main__':unittest.main()
