"""Synthetic paired SIP cases only; no source data or network calls."""
import json,math,unittest
from xom_field_audit import SOURCE_ID,AuditError,audit_rows
SESSIONS=['2016-01-04','2016-01-05','2016-01-06','2016-01-07']

def row(d='2016-01-04',adjustment='raw',**changes):
 r=dict(security_id=SOURCE_ID,trading_date=d,query_symbol='XOM',adjustment=adjustment,
  open=100.,high=110.,low=90.,close=105.,volume=1000.,feed='sip',timeframe='1Day')
 if adjustment=='all':r.update(open=80.,high=88.,low=72.,close=84.,volume=1100.)
 r.update(changes);return r

def pair(d='2016-01-04'):return [row(d),row(d,'all')]

class XOMQAContracts(unittest.TestCase):
 def test_valid_pair_and_volume_difference_not_forced_to_pricefactor(self):
  summary,q=audit_rows(pair()+pair(SESSIONS[1]),SESSIONS[:2])
  self.assertEqual(summary['input_rows'],4);self.assertEqual(summary['complete_raw_all_pairs'],2)
  self.assertEqual(summary['dates_with_quality_flags'],0)
  self.assertEqual(summary['volume_difference_pair_dates'],2)
  self.assertFalse(summary['volume_assumed_inverse_of_price_adjustment_factor'])
  self.assertEqual(q[0]['paired_all_over_raw_close'],.8)
  self.assertFalse(summary['explicit_dividend_split_action_fields_available'])
  self.assertFalse(summary['economic_total_return_certified']);self.assertFalse(summary['canonical_mapping_promoted'])

 def test_missing_pair_and_missing_whole_session_remain_on_grid(self):
  summary,q=audit_rows([row()],SESSIONS[:2])
  self.assertEqual(len(q),2);self.assertEqual(summary['missing_all_dates'],SESSIONS[:2])
  self.assertEqual(summary['missing_raw_dates'],[SESSIONS[1]])
  self.assertIsNone(q[0]['paired_all_over_raw_close']);self.assertFalse(summary['missing_values_filled'])

 def test_duplicate_original_source_date_adjustment_rejected(self):
  with self.assertRaises(AuditError):audit_rows(pair()+[row()],SESSIONS)

 def test_metadata_guards_before_unauthorized_numeric_use(self):
  class Poison:
   def __float__(self):raise AssertionError('Unauthorized conversion')
  for update in ({'security_id':'alias'},{'query_symbol':'XOM_ALIAS'},{'adjustment':'split'},
       {'trading_date':'2024-01-02'},{'trading_date':'2016-01-03'},{'trading_date':'2016-01-08'},
       {'feed':'not_sip'},{'timeframe':'1Min'}):
   with self.subTest(update=update),self.assertRaises(AuditError):
    audit_rows([row(close=Poison(),**update)],SESSIONS)

 def test_unapproved_financial_fields_and_duplicate_calendar_fail(self):
  for extra in ('vwap','trade_count','dividend','split_ratio','label_return'):
   with self.subTest(extra=extra),self.assertRaises(AuditError):audit_rows([row(**{extra:1})],SESSIONS)
  with self.assertRaises(AuditError):audit_rows(pair(),[SESSIONS[0],SESSIONS[0]])

 def test_nonpositive_brackets_and_null_or_invalid_volume_flagged(self):
  summary,q=audit_rows([row(high=80.,volume=-1),row(adjustment='all',low=0.,volume=None)],SESSIONS[:1])
  self.assertIn('raw_OHLC_bracket_invalid',q[0]['flags'])
  self.assertIn('all_OHLC_nonpositive_or_nonfinite',q[0]['flags'])
  self.assertIn('raw_volume_negative_or_nonfinite',q[0]['flags'])
  self.assertIn('all_volume_negative_or_nonfinite',q[0]['flags'])
  self.assertEqual(summary['dates_with_quality_flags'],1)

 def test_common_scale_mismatch_flagged_preread_two_bp_tolerance(self):
  summary,q=audit_rows([row(),row(adjustment='all',high=90.)],SESSIONS[:1])
  self.assertEqual(summary['OHLC_common_price_scale_relative_tolerance'],.0002)
  self.assertIn('raw_all_OHLC_price_scales_disagree',q[0]['flags'])

 def test_large_jump_is_flag_not_exclusion_or_action_inference(self):
  second=[row(SESSIONS[1],open=200.,high=220.,low=180.,close=210.),
      row(SESSIONS[1],'all',open=160.,high=176.,low=144.,close=168.)]
  summary,q=audit_rows(pair()+second,SESSIONS[:2]);jumps=summary['flagged_consecutive_session_discontinuities']
  self.assertEqual(len(jumps),1);self.assertEqual(len(q),2)
  self.assertFalse(jumps[0]['verified_economic_explanation'])
  self.assertFalse(jumps[0]['synthetic_fill_or_automatic_exclusion'])

 def test_no_jump_compression_over_missing_expected_session(self):
  later=[row(SESSIONS[2],open=200.,high=220.,low=180.,close=210.),
      row(SESSIONS[2],'all',open=160.,high=176.,low=144.,close=168.)]
  summary,_=audit_rows(pair()+later,SESSIONS[:3])
  self.assertEqual(summary['flagged_consecutive_session_discontinuities'],[])

 def test_partial_pair_does_not_hide_available_side_jump(self):
  summary,_=audit_rows(pair()+[row(SESSIONS[1],open=200.,high=220.,low=180.,close=210.)],SESSIONS[:2])
  jumps=summary['flagged_consecutive_session_discontinuities'];self.assertEqual(len(jumps),1)
  self.assertEqual(jumps[0]['raw_close_change'],1.);self.assertIsNone(jumps[0]['all_close_change'])

 def test_derived_overflow_cannot_be_reported_as_valid_or_nonstandardJSON(self):
  raw=row(open=1e-308,high=2e-308,low=1e-309,close=1e-308)
  adjusted=row(adjustment='all',open=1e308,high=1e308,low=1e308,close=1e308)
  summary,q=audit_rows([raw,adjusted],SESSIONS[:1])
  self.assertIn('raw_all_OHLC_price_scales_disagree',q[0]['flags'])
  json.dumps({'summary':summary,'quality':q},allow_nan=False)

if __name__=='__main__':unittest.main()
