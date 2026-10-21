import copy
import datetime
import unittest
import learning_repairs as l


def row(native, date, price=10.0, volume=1000.0):
    return dict(intrinio_security_id=native,trading_date=date,frequency='daily',intraperiod=False,
        open=price,high=price+1,low=price-1,close=price+.5,volume=volume,
        adj_open=price/2,adj_high=(price+1)/2,adj_low=(price-1)/2,adj_close=(price+.5)/2,adj_volume=volume*2,
        factor=.5,split_ratio=None,dividend=0.,raw_ohlc_bounds_valid=True,
        adjusted_ohlc_bounds_valid=True,price_quality_status='accepted')


def rcm_fixture():
    start=datetime.date(2021,11,1)
    dates=[]
    current=start
    while len(dates)<160:
        if current.weekday()<5: dates.append(current.isoformat())
        current+=datetime.timedelta(days=1)
    # Synthetic session vector remains inside the frozen interval.
    dates[-1]='2022-06-21'
    controls=['2021-10-28','2021-10-29']
    target=controls+['2022-06-22','2022-06-23']
    inventory={'routes':{l.RCM_OLD:controls+dates,l.RCM_TARGET:target}}
    rows=[row(l.RCM_OLD,d) for d in controls+dates]+[row(l.RCM_TARGET,d) for d in target]
    return rows,inventory


def disca_fixture():
    literals=[]
    for date,(symbol,native) in l.DISCA_ROUTES.items():
        r=row(l.DISCA_TARGET,date)
        for adjustment,prefix in [('raw',''),('all','adj_')]:
            literals.append(dict(unified_security_id=l.DISCA_UNIFIED,trading_date=date,
                adjustment=adjustment,query_symbol=symbol,feed='sip',timeframe='1Day',
                selected_source_product_id='russell_1000_2000_3000',selected_source_security_id=native,
                **{c:r[prefix+c] for c in (*l.OHLC,'volume')}))
    controls=[row(l.DISCA_TARGET,d) for d in ('2022-04-01','2022-04-11')]
    return literals,controls


class LearningRepairs(unittest.TestCase):
    def test_rcm_160_append_preserves_all_fields_and_actions(self):
        rows,inv=rcm_fixture(); original=copy.deepcopy(rows)
        receipt,admitted=l.rcm_case(rows,inv)
        self.assertEqual(len(admitted),160)
        self.assertEqual(receipt['status'],'ACCEPTED_PROVISIONAL_DATED_APPEND_ONLY')
        self.assertEqual(len(receipt['control_comparisons']),20)
        self.assertEqual(rows,original)
        self.assertEqual(admitted[0]['factor'],.5)
        self.assertEqual(admitted[0]['dividend'],0.)
    def test_rcm_volume_rejection_is_normal(self):
        rows,inv=rcm_fixture(); rows[0]['volume']*=1.001
        receipt,admitted=l.rcm_case(rows,inv)
        self.assertEqual(receipt['status'],'REJECTED_SOURCE_QUALITY');self.assertEqual(admitted,[])
    def test_rcm_future_key_is_structural(self):
        rows,inv=rcm_fixture();rows[0]['trading_date']='2024-01-01'
        with self.assertRaises(ValueError): l.rcm_case(rows,inv)
    def test_rcm_duplicate_is_structural(self):
        rows,inv=rcm_fixture();rows[0]=rows[1].copy()
        with self.assertRaises(ValueError): l.rcm_case(rows,inv)
    def test_disca_accepts_5_without_imputing_actions(self):
        raw,controls=disca_fixture();receipt,admitted=l.disca_case(raw,controls)
        self.assertEqual(len(admitted),5);self.assertEqual(len(receipt['control_comparisons']),20)
        for r in admitted:
            self.assertIsNone(r['factor']);self.assertIsNone(r['split_ratio']);self.assertIsNone(r['dividend'])
            self.assertEqual(r['adj_volume'],2000.)
    def test_disca_volume_mismatch_rejects_every_learning_row(self):
        raw,controls=disca_fixture();controls[0]['volume']*=1.001
        receipt,admitted=l.disca_case(raw,controls)
        self.assertEqual(receipt['status'],'REJECTED_SOURCE_QUALITY');self.assertEqual(admitted,[])
        self.assertIn('control_volume:2022-04-01',receipt['quality_reasons'])
    def test_disca_adjusted_volume_separate_rejection(self):
        raw,controls=disca_fixture();controls[1]['adj_volume']*=1.001
        receipt,admitted=l.disca_case(raw,controls)
        self.assertEqual(admitted,[])
        self.assertIn('control_adj_volume:2022-04-11',receipt['quality_reasons'])
    def test_disca_frozen_tolerance_boundary(self):
        raw,controls=disca_fixture();controls[0]['volume']*=1.0001
        receipt,admitted=l.disca_case(raw,controls);self.assertEqual(len(admitted),5)
    def test_disca_wrong_share_class_hard_failure(self):
        raw,controls=disca_fixture();raw[0]['query_symbol']='DISCK'
        with self.assertRaises(ValueError):l.disca_case(raw,controls)
    def test_missing_pair_hard_failure(self):
        raw,controls=disca_fixture()
        with self.assertRaises(ValueError):l.disca_case(raw[:-1],controls)
    def test_bad_bar_normal_rejection(self):
        raw,controls=disca_fixture();raw[2]['low']=100
        receipt,admitted=l.disca_case(raw,controls);self.assertEqual(admitted,[])
    def test_append_preserves_originals_and_rejects_overwrite(self):
        original=[row(l.RCM_TARGET,'2021-10-29')];patch=[row(l.RCM_TARGET,'2021-11-01')]
        out=l.append_only(original,patch);self.assertIs(out[0],original[0]);self.assertEqual(out[1],patch[0])
        with self.assertRaises(ValueError):l.append_only(original,original)
    def test_zero_volume_compatibility(self):
        self.assertEqual(l.relative(0,0),0);self.assertIsNone(l.relative(1,0))
    def test_malformed_iso_date_rejected(self):
        with self.assertRaises(ValueError):l.day('2022-4-01')
    def test_infinite_volume_rejection(self):
        r=row(l.RCM_OLD,'2021-11-01');r['volume']=float('inf')
        self.assertIn('invalid_volume',l.quality(r))


if __name__=='__main__':unittest.main()
