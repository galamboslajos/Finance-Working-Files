import unittest
from dataclasses import replace
import numpy as np
import pandas as pd
from dmv2.quality_audit import AuditPolicy, audit_daily_history


def frame(prices, *, native='sec_test', start='2021-01-04'):
    p=np.array(prices,dtype=float);dates=pd.bdate_range(start,periods=len(p));x=pd.DataFrame({'intrinio_security_id':native,'date':dates,'open':p,'high':p*1.01,'low':p*.99,'close':p,'frequency':'daily','split_ratio':1.,'dividend':0.,'volume':1000.})
    for c in ['open','high','low','close']:x['adj_'+c]=x[c]
    return x

class QualityAuditTest(unittest.TestCase):
    def test_ordinary_noise_has_no_flags_and_input_immutable(self):
        x=frame([100,101,100.5,102,101.5,103]);old=x.copy(deep=True);r=audit_daily_history(x,expected_sessions=x.date)
        self.assertTrue(r.flags.empty);pd.testing.assert_frame_equal(x,old)
    def test_genuine_oneway_gap_is_review_not_repaired(self):
        x=frame([100,101,50,51,52]);r=audit_daily_history(x);reasons=set(r.flags.reason)
        self.assertIn('adjusted_close_move_30pct',reasons);self.assertIn('raw_open_gap_30pct',reasons)
        self.assertNotIn('adjusted_roundtrip_regime_candidate',reasons)
        self.assertEqual(r.summary['status'],'FLAG_AUDIT_ONLY_NOT_ERROR_COUNT');self.assertEqual(x.close.iloc[2],50)
        self.assertTrue(r.flags.disposition.eq('REVIEW_ONLY_NO_DATA_CHANGE').all())
    def test_coherent_two_for_one_split(self):
        x=frame([100,102,51,52]);x.loc[2,'split_ratio']=2.
        for c in ['open','high','low','close']:x.loc[:1,'adj_'+c]=x.loc[:1,c]/2
        r=audit_daily_history(x,policy=replace(AuditPolicy(),split_ratio_convention='new_shares_per_old'))
        self.assertIn('declared_split_review',set(r.flags.reason));self.assertNotIn('split_adjustment_factor_conflict',set(r.flags.reason));self.assertNotIn('adjusted_close_move_30pct',set(r.flags.reason))
    def test_bad_split_factor_is_flag_not_fixed(self):
        x=frame([100,102,51,52]);x.loc[2,'split_ratio']=2.
        r=audit_daily_history(x,policy=replace(AuditPolicy(),split_ratio_convention='new_shares_per_old'))
        self.assertIn('split_adjustment_factor_conflict',set(r.flags.reason))
    def test_unspecified_split_convention_does_not_assert_conflict(self):
        x=frame([100,102,51,52]);x.loc[2,'split_ratio']=2.;r=audit_daily_history(x)
        self.assertIn('declared_split_review',set(r.flags.reason));self.assertNotIn('split_adjustment_factor_conflict',set(r.flags.reason))
    def test_opch_like_valid_bars_bad_series(self):
        x=frame([23.7,9.88,25.34,9.86,25.82,9.89,25.75]);r=audit_daily_history(x)
        self.assertFalse(any('bounds' in s for s in r.flags.reason));self.assertIn('adjusted_roundtrip_regime_candidate',set(r.flags.reason));self.assertIn('adjusted_repeated_regime_switches',set(r.flags.reason))
    def test_sub30pct_repeated_switches_still_flagged(self):
        x=frame([100,125,100,125,100,125,100]);r=audit_daily_history(x)
        self.assertNotIn('adjusted_close_move_30pct',set(r.flags.reason));self.assertIn('adjusted_repeated_regime_switches',set(r.flags.reason))
    def test_null_duplicate_and_conflicting_keys(self):
        x=frame([100,101,102,103]);dup=x.iloc[[1]].copy();conf=x.iloc[[2]].copy();conf['close']=105.;conf['adj_close']=105.;null=x.iloc[[0]].copy();null['intrinio_security_id']=None;y=pd.concat([x,dup,conf,null],ignore_index=True);r=audit_daily_history(y)
        self.assertTrue({'null_or_invalid_key','duplicate_observation','conflicting_duplicate_observation'}.issubset(set(r.flags.reason)));self.assertEqual(r.summary['ambiguous_native_date_keys'],1)
    def test_cache_version_replica_deduplicates(self):
        x=frame([100,101,102]);x['source_version']='v1';y=x.copy();y['source_version']='v2';r=audit_daily_history(pd.concat([x,y],ignore_index=True))
        self.assertTrue(r.flags.empty);self.assertEqual(r.summary['collapsed_cross_version_equal_copies'],3);self.assertEqual(r.summary['diagnostic_unique_daily_rows'],3)
    def test_equal_duplicate_within_source_is_not_cache_copy(self):
        x=frame([100,101,102]);x['source_version']='v1';r=audit_daily_history(pd.concat([x,x.iloc[[1]]],ignore_index=True));self.assertIn('duplicate_observation',set(r.flags.reason));self.assertEqual(r.summary['collapsed_cross_version_equal_copies'],0)
    def test_hole_only_within_observed_extent(self):
        x=frame([100,101,102,103,104]);sessions=pd.bdate_range('2020-12-01','2021-02-01');x=x.drop(index=[2]);r=audit_daily_history(x,expected_sessions=sessions);f=r.flags[r.flags.reason.eq('missing_session_cluster')];self.assertEqual(len(f),1);self.assertEqual(f.observations.iloc[0],1);self.assertEqual(f.span_start.iloc[0],'2021-01-06')
    def test_staleness_and_missing_session_break(self):
        x=frame([100]*5);r=audit_daily_history(x,expected_sessions=x.date);self.assertIn('adjusted_stale_identical_ohlc_run',set(r.flags.reason));sessions=pd.bdate_range(x.date.min(),periods=6);x.loc[3:,'date']=sessions[4:];r=audit_daily_history(x,expected_sessions=sessions);self.assertNotIn('adjusted_stale_identical_ohlc_run',set(r.flags.reason))
    def test_invalid_ohlc_missing_adjusted_and_factor_mismatch(self):
        x=frame([100,101,102]);x.loc[0,'high']=90;x.loc[1,'adj_low']=np.nan;x.loc[2,'adj_open']=80.;r=audit_daily_history(x);self.assertTrue({'raw_ohlc_bounds','invalid_or_missing_ohlc','within_bar_adjustment_conflict'}.issubset(set(r.flags.reason)))
    def test_frequency_duplicate_and_intraperiod(self):
        x=frame([100,101,102]);y=x.iloc[[0]].copy();y.frequency='weekly';z=x.iloc[[2]].copy();z['intraperiod']=True;x['intraperiod']=False;x.loc[2,'intraperiod']=True;r=audit_daily_history(pd.concat([x,y],ignore_index=True));self.assertIn('conflicting_duplicate_observation',set(r.flags.reason));self.assertIn('non_daily_observation',set(r.flags.reason))
    def test_overlapping_raw_adjusted_evidence_is_one_review_episode(self):
        r=audit_daily_history(frame([100,50,100,50,100,50,100]))
        self.assertGreater(len(r.flags),1);self.assertEqual(r.flags.event_cluster.nunique(),1)
        self.assertEqual(r.flags.evidence_id.nunique(),len(r.flags))
    def test_separated_events_are_distinct_review_episodes(self):
        r=audit_daily_history(frame([100,50,51,52,53,54,55,56,57,58,100]))
        self.assertGreaterEqual(r.flags.event_cluster.nunique(),2)
    def test_null_key_summary_is_strict_json_serializable(self):
        import json
        x=frame([100]);x.loc[0,'intrinio_security_id']=None;x.loc[0,'date']=pd.NaT
        r=audit_daily_history(x);json.dumps(r.summary,allow_nan=False)
        self.assertEqual(r.summary['by_native_id'][0]['intrinio_security_id'],None)
    def test_input_budget_fails_before_result(self):
        with self.assertRaisesRegex(ValueError,'Input row budget'):
            audit_daily_history(frame([100,101]),policy=replace(AuditPolicy(),max_input_rows=1))
    def test_protected_dates_refused(self):
        x=frame([100],start='2024-01-02')
        with self.assertRaisesRegex(ValueError,'Protected'):audit_daily_history(x)
    def test_budget_fails_without_truncation(self):
        x=frame([100,50,100,50,100]);
        with self.assertRaisesRegex(ValueError,'Flag budget'):audit_daily_history(x,policy=replace(AuditPolicy(),max_flags=1))
    def test_peryear_pernative_perreason_summary(self):
        x=frame([100,50,51]);r=audit_daily_history(x,provenance={'overlay_status':'frozen_inputs','source_versions':['v1']});self.assertEqual(r.summary['by_year'][0]['year'],'2021');self.assertEqual(r.summary['by_native_id'][0]['intrinio_security_id'],'sec_test');self.assertTrue(r.summary['by_reason']);self.assertEqual(r.summary['provenance']['overlay_status'],'frozen_inputs')
    def test_fine_bars_not_proof_of_correct_series(self):
        r=audit_daily_history(frame([100,50,100]));self.assertFalse(any('bounds' in s for s in r.flags.reason));self.assertTrue(len(r.flags));self.assertIn('No flag does not establish source correctness',r.summary['limits'][3])

if __name__=='__main__':unittest.main()
