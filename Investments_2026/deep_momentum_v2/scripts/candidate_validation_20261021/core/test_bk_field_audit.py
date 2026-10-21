"""Synthetic source-row/Arrow-mask checks; no research financial reads."""
import math,unittest
import pyarrow as pa
from bk_field_audit import (COLUMNS,META,NATIVE,AuditError,audit_rows,key_and_warmup_summary)
from read_bk_candidate import mask

SESSIONS=['2007-07-02','2007-07-03','2007-07-05','2007-07-06']
def row(d='2007-07-02',**changes):
 r={k:None for k in COLUMNS};r.update(intrinio_security_id=NATIVE,trading_date=d,frequency='daily',intraperiod=False,
  price_quality_status='accepted',raw_ohlc_bounds_valid=True,adjusted_ohlc_bounds_valid=True,
  open=100.,high=110.,low=90.,close=105.,volume=1000.,adj_open=90.,adj_high=99.,adj_low=81.,adj_close=94.5,
  adj_volume=1200.,factor=.9,split_ratio=None,dividend=None)
 r.update(changes);return r

class BKFieldContracts(unittest.TestCase):
 def test_good_paired_prices_volume_not_forced_inverse_dividend_factor(self):
  summary,quality,jumps=audit_rows([row(),row('2007-07-03')],SESSIONS)
  self.assertEqual(summary['rows_with_field_flags'],0);self.assertEqual(len(quality),2)
  self.assertLess(summary['max_OHLC_common_scale_relative_residual'],1e-14)
  self.assertEqual(summary['source_factor_direct_max_relative_residual'],0.)
  self.assertFalse(summary['adjusted_volume_assumed_inverse_of_dividend_inclusive_price_factor'])
  self.assertFalse(summary['canonical_mapping_promoted']);self.assertEqual(jumps,[])

 def test_protected_wrong_native_premerger_or_nonsession_fail_before_numeric(self):
  class Poison:
   def __float__(self):raise AssertionError('Unauthorized numeric conversion')
  for changes in ({'trading_date':'2024-01-02'},{'trading_date':'2007-06-29'},
      {'trading_date':'2007-07-04'},{'intrinio_security_id':'another'}):
   with self.subTest(changes=changes),self.assertRaises(AuditError):
    audit_rows([row(close=Poison(),**changes)],SESSIONS)

 def test_duplicate_key_rejected(self):
  with self.assertRaises(AuditError):audit_rows([row(),row()],SESSIONS)

 def test_bad_OHLC_and_false_source_flag_reported(self):
  summary,q,_=audit_rows([row(high=80.)],SESSIONS)
  self.assertIn('raw_price_or_OHLC_bracket_invalid',q[0]['flags'])
  self.assertIn('source_raw_bounds_flag_disagrees',q[0]['flags'])
  self.assertGreater(summary['rows_with_field_flags'],0)

 def test_cross_OHLC_scale_inconsistency_and_missingfactor(self):
  _,q,_=audit_rows([row(adj_high=109.,factor=None)],SESSIONS)
  self.assertIn('raw_adjusted_OHLC_scales_disagree',q[0]['flags'])
  self.assertIn('source_factor_missing_or_invalid',q[0]['flags'])

 def test_null_and_invalid_action_volume_values_never_imputed(self):
  summary,q,_=audit_rows([row(volume=-1,adj_volume=math.inf)],SESSIONS)
  self.assertIn('volume_invalid',q[0]['flags']);self.assertIn('adj_volume_invalid',q[0]['flags'])
  self.assertEqual(summary['split_ratio_null_rows'],1);self.assertEqual(summary['dividend_null_rows'],1)
  self.assertIsNone(q[0]['split_ratio_source']);self.assertIsNone(q[0]['dividend_source'])

 def test_large_move_is_review_flag_not_verified_action_or_exclusion(self):
  second=row('2007-07-03',open=200,high=220,low=180,close=210,adj_open=180,adj_high=198,adj_low=162,adj_close=189,split_ratio=2.)
  summary,q,jumps=audit_rows([row(),second],SESSIONS)
  self.assertEqual(len(jumps),1);self.assertTrue(jumps[0]['source_action_marker_present'])
  self.assertFalse(jumps[0]['verified_economic_explanation']);self.assertFalse(jumps[0]['synthetic_fill_or_exclusion'])
  self.assertEqual(summary['financial_rows'],2)

 def test_discontinuity_does_not_bridge_missing_true_session(self):
  second=row('2007-07-06',open=200,high=220,low=180,close=210,adj_open=180,adj_high=198,adj_low=162,adj_close=189)
  summary,_,jumps=audit_rows([row(),second],SESSIONS)
  self.assertEqual(jumps,[]);self.assertEqual(summary['missing_postmerger_sessions'],['2007-07-03','2007-07-05'])

 def test_source_quality_and_factor_interpretation_are_not_promoted(self):
  summary,q,_=audit_rows([row(price_quality_status='unreviewed',factor=1/.9)],SESSIONS)
  self.assertIn('source_quality_not_accepted',q[0]['flags'])
  self.assertFalse(summary['source_factor_semantics_certified'])
  self.assertAlmostEqual(summary['source_factor_inverse_max_relative_residual'],0.)

 def test_arrowmask_filters_native_identity_interval_XNYS_and_protected_before_python(self):
  t=pa.Table.from_pylist([row(),row('2007-06-29'),row('2007-07-04'),row('2024-01-02'),row(intrinio_security_id='other')])
  got=mask(t,'2007-07-02','2023-12-31',SESSIONS)
  self.assertEqual(got.num_rows,1);self.assertEqual(got['trading_date'].to_pylist(),['2007-07-02'])

if __name__=='__main__':unittest.main()
