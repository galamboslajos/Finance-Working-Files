"""Synthetic-only integration checks. Never uses research source fixtures."""
from pathlib import Path
from dataclasses import replace
import tempfile,json,unittest,hashlib
import pandas as pd
import pyarrow as pa,pyarrow.parquet as pq
from dmv2.source_repair_loading import *
from dmv2.el_paso_route import SOURCE,UNIFIED,OLD,NEW,RouteError
from test_el_paso_route import RouteTest,row
from test_opch_amendment import OpchAmendmentTest
from dmv2.fallback_loading import FallbackYearPlan,FallbackReadLimits,FALLBACK_POLICY
from dmv2.walkforward_loading import WalkforwardYearPlan,WalkforwardReadLimits,FALLBACK_POLICY as WALK_POLICY
from dmv2.monthly import MonthWindow
from dmv2.universe import UniverseSnapshot,EXCLUDED_SCHEMA

def pin(p):return PinnedFile(str(p.resolve()),file_hash(p),p.stat().st_size)
def quote(native,d,value=10.):
    z={'intrinio_security_id':native,'trading_date':d,'frequency':'daily','intraperiod':False,'volume':5.,'adj_volume':5.,'factor':1.,'split_ratio':1.,'dividend':0.,'raw_ohlc_bounds_valid':True,'adjusted_ohlc_bounds_valid':True,'price_quality_status':'accepted'}
    z.update({c:value for c in ['open','high','low','close','adj_open','adj_high','adj_low','adj_close']});return z
def calendar(dates):return pa.Table.from_pylist([{'trading_date':d,'calendar_name':'XNYS','is_trading_session':True,'session_open_utc':d+'T14:30:00Z','session_close_utc':d+'T21:00:00Z','is_early_close':False}for d in dates])
def loaded(kind='fallback',monthly=None,daily=None,feature=None,outcome=None,dates=None,raw_start='2010-01-28',feature_end='2010-01-29',raw_end='2010-02-26',year=2010):
    monthly=monthly if monthly is not None else pa.Table.from_pylist([row()]);daily=daily if daily is not None else monthly
    dates=dates or ['2010-01-28','2010-01-29','2010-02-01','2010-02-26'];cal=calendar(dates)
    ids=tuple(sorted({r['intrinio_security_id']for t in[monthly,daily]for r in t.select(['intrinio_security_id','identity_eligible']).to_pylist()if r['identity_eligible']}))
    feature=feature if feature is not None else pa.Table.from_pylist([quote(OLD,'2010-01-28'),quote(OLD,'2010-01-29')]);outcome=outcome if outcome is not None else pa.Table.from_pylist([quote(OLD,'2010-01-29'),quote(OLD,'2010-02-01')])
    signals=sorted(set(monthly['signal_date'].to_pylist()));snaps=tuple(UniverseSnapshot(monthly.filter(pc.equal(monthly['signal_date'],d)),pa.Table.from_pylist([],schema=EXCLUDED_SCHEMA),d,'combined',{'mode':'historical_reconstruction'},{'constituents':monthly.num_rows})for d in signals)
    windows=tuple(MonthWindow(str(year)+'-02',d,dates[-2],raw_end,tuple(x for x in dates if x>=d))for d in signals)
    cls=FallbackYearPlan if kind=='fallback'else WalkforwardYearPlan;lim=FallbackReadLimits()if kind=='fallback'else WalkforwardReadLimits();policy=FALLBACK_POLICY if kind=='fallback'else WALK_POLICY
    p=cls(Path('/private/tmp'),Path('/private/tmp'),year,{},cal,windows,monthly,daily,snaps,ids,ids,raw_start,feature_end,raw_end,{}, {'policy':policy},lim,'');f=fallback_fingerprint if kind=='fallback'else walkforward_fingerprint;p=replace(p,fingerprint=f(p));lc=LoadedFallbackYear if kind=='fallback'else LoadedWalkforwardYear
    return lc(feature,outcome,p,{}, {'plan_fingerprint':p.fingerprint,'policy':policy,'approved_missing_route_fallback':True,'fully_identity_repaired':False,'quote_amendments_fingerprint':'frozen_historical_plus_recent_overlay'})

class IntegrationTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir='/private/tmp');self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.rf=RouteTest(methodName='test_exact_source_route_only_and_all_flags_unified_values_preserved');self.rf.setUp();self.addCleanup(self.rf.doCleanups);self.route=self.rf.worker()
        # A fully pinned small synthetic source object and1111 distinct projection rows.
        days=[d.date().isoformat()for d in pd.date_range(EP_START,periods=1110)]+[EP_END]
        rows=[dict(quote(NEW,d),company_id='com_gDvxOz',ticker='EP',figi='BBG000DST4Q5',source_page_hash='f'*64,source_relative_path='raw_pages/intrinio_security_id=sec_XO6Eab-fixture/prices/page.json.gz')for d in days]
        self.proj=self.root/'el_paso_sec_XO6Eab_2008_2012.parquet';pq.write_table(pa.Table.from_pylist(rows),self.proj)
        self.sourcefile=self.root/'source.bin';self.sourcefile.write_bytes(b'SYNTHETIC raw source fixture, no market data')
        self.plan=self.root/'plan.json';self.plan.write_text(json.dumps({'expected_provider_metadata':{'intrinio_security_id':NEW,'company_id':'com_gDvxOz','figi':'BBG000DST4Q5','source_relative_path':'raw_pages/query_identifier=US28336L1098-fixture/security.json.gz'},'source_objects':[{'path':str(self.sourcefile),'relative_path':'tables/prices/fixture.bin','size_bytes':self.sourcefile.stat().st_size,'manifest_sha256':file_hash(self.sourcefile)}]}))
        self.receipt=self.root/'receipt.json';self.receipt.write_text(json.dumps({'status':'COMPLETE_CORRECT_NATIVE_LOCAL_HISTORY_ACQUIRED','plan_sha256':file_hash(self.plan).removeprefix('sha256:'),'output':{'path':'/recorded/original/'+self.proj.name,'sha256':file_hash(self.proj).removeprefix('sha256:'),'rows':EP_ROWS},'whole_source_hash_authenticated_before_and_after':True,'native_id':NEW,'figi':'BBG000DST4Q5','provider_lookup_ISIN':'US28336L1098','actual_date_min':EP_START,'actual_date_max':EP_END,'missing_session_dates':[],'no_synthetic_quotes':True,'financial_2024_plus_returned':False,'provenance':{'objects':[{'relative_path':'tables/prices/fixture.bin','size_bytes':self.sourcefile.stat().st_size,'sha256':file_hash(self.sourcefile)}]}}))
        self.src=CorrectElPasoProjection.from_pins(projection=pin(self.proj),receipt=pin(self.receipt),plan=pin(self.plan));self.state=JointSourceRepairs(self.route,self.src,None)
    def apply(self,value=None,repairs=None):return apply_source_repairs(loaded()if value is None else value,self.state if repairs is None else repairs,approved_source_repair=True)
    def test_route_snapshots_requests_and_exact_frozen_fingerprint(self):
        x=loaded();a=self.apply(x);self.assertEqual(a.plan.requested_ids,(NEW,));self.assertEqual(a.plan.original_requested_ids,(OLD,));self.assertEqual(a.plan.snapshots[0].table['intrinio_security_id'].to_pylist(),[NEW]);self.assertEqual(a.plan.fingerprint,fallback_fingerprint(a.plan));self.assertEqual(a.provenance['plan_fingerprint'],a.plan.fingerprint);self.assertEqual(a.provenance['policy'],FALLBACK_POLICY);self.assertEqual(x.plan.original_cohort['intrinio_security_id'].to_pylist(),[OLD]);self.assertEqual(a.plan.original_cohort['identity_status'].to_pylist(),['needs_review'])
    def test_only_allowed_calendar_dates_appended_no_non_session_or_outcome_leak(self):
        # January2010 lies beyond synthetic1110-day projection; use exact earlier dates present.
        x=loaded(raw_start='2009-01-28',feature_end='2009-01-29',raw_end='2009-02-26',year=2010,monthly=pa.Table.from_pylist([row(when='2009-01-29')]),dates=['2009-01-28','2009-01-29','2009-02-02','2009-02-26'],feature=pa.Table.from_pylist([quote(OLD,'2009-01-28')]),outcome=pa.Table.from_pylist([quote(OLD,'2009-01-29')]))
        a=self.apply(x);self.assertEqual(set(a.feature_prices['trading_date'].to_pylist()),{'2009-01-28','2009-01-29'});self.assertEqual(set(a.outcome_prices['trading_date'].to_pylist()),{'2009-01-29','2009-02-02','2009-02-26'});self.assertNotIn('2009-01-30',a.feature_prices['trading_date'].to_pylist())
    def test_genuine_empire_other_date_preserves_old_native_and_quote(self):
        m=pa.Table.from_pylist([row(),row('genuine_empire',OLD,'2010-02-26')]);q=pa.Table.from_pylist([quote(OLD,'2010-01-29'),quote(OLD,'2010-02-26')]);x=loaded(monthly=m,daily=m,feature=q,outcome=q,dates=['2010-01-29','2010-02-26','2010-03-01'],feature_end='2010-02-26',raw_end='2010-03-01');a=self.apply(x);self.assertEqual(set(a.plan.requested_ids),{OLD,NEW});self.assertTrue(pc.any(pc.equal(a.feature_prices['intrinio_security_id'],OLD)).as_py())
    def test_collision_even_blocked_original_member_refuses(self):
        other=dict(row('blocked',NEW),identity_eligible=False,identity_status='unresolved');x=loaded(monthly=pa.Table.from_pylist([row(),other]));self.assertRaises(RouteError,self.apply,x)
    def test_duplicate_base_quotes_fail_closed(self):
        x=loaded();x=replace(x,feature_prices=pa.concat_tables([x.feature_prices,x.feature_prices.slice(0,1)]));self.assertRaisesRegex(RepairError,'Duplicate native',self.apply,x)
    def test_changed_base_plan_fingerprint_refused(self):
        x=loaded();self.assertRaisesRegex(RepairError,'fingerprint changed',self.apply,replace(x,plan=replace(x.plan,raw_end='2010-03-01')))
    def test_changed_projection_or_whole_source_refused(self):
        self.sourcefile.write_bytes(b'corrupted');self.assertRaisesRegex(RepairError,'whole source changed',self.apply)
    def test_changed_projection_file_refused(self):
        self.proj.write_bytes(self.proj.read_bytes()+b' ');self.assertRaisesRegex(RepairError,'Pinned file changed',self.apply)
    def test_approval_required(self):self.assertRaises(RepairError,apply_source_repairs,loaded(),self.state)
    def test_source_pair_required(self):self.assertRaises(RepairError,self.apply,None,JointSourceRepairs(self.route,None,None))
    def test_wrong_bbb_y_quarantine_refused(self):
        bad=dict(row('siblis_ticker_company_00326c7d9ad13903e5e3','sec_ydbQnj'));self.assertRaisesRegex(RepairError,'BBBY quarantine',self.apply,loaded(monthly=pa.Table.from_pylist([row(),bad])))
    def test_already_amended_reapply_and_tampered_price_both_refused(self):
        a=self.apply();self.assertRaisesRegex(RepairError,'Already amended',self.apply,a);b=replace(a,feature_prices=a.feature_prices.slice(0,0));self.assertRaisesRegex(RepairError,'Already amended',self.apply,b)
    def test_existing_correct_native_base_overlap_refused(self):
        x=loaded();x=replace(x,plan=replace(x.plan,requested_ids=(OLD,NEW),original_requested_ids=(OLD,NEW)));x=replace(x,plan=replace(x.plan,fingerprint=fallback_fingerprint(x.plan)));x=replace(x,provenance={**x.provenance,'plan_fingerprint':x.plan.fingerprint},feature_prices=pa.Table.from_pylist([quote(OLD,'2010-01-29'),quote(NEW,'2010-01-29')]));self.assertRaisesRegex(RepairError,'already present',self.apply,x)
    def test_portable_projection_original_absolute_path_is_provenance(self):
        # Receipt deliberately records another absolute parent; basename/hash/rows authenticate copied bytes.
        self.assertEqual(self.src.verify_metadata().num_rows,EP_ROWS)
    def test_protected_base_numeric_date_refused(self):
        x=loaded();x=replace(x,feature_prices=pa.Table.from_pylist([quote(OLD,'2024-01-02')]));self.assertRaisesRegex(RepairError,'Protected',self.apply,x)
    def opch_loaded(self,year=2023):
        of=OpchAmendmentTest(methodName='test_exact_fifteen_repaired_volume_null_and_other_fields_preserved');of.setUp();self.addCleanup(of.doCleanups)
        t=of.table;d=t['trading_date'].to_pylist();m=pa.Table.from_pylist([row('opch_source','sec_zmox6E','2023-11-30')]);empty=t.slice(0,0)
        x=loaded(kind='walkforward',year=year,monthly=m,daily=m,feature=t,outcome=empty,dates=[*d,'2023-11-30','2023-12-29'],raw_start='2022-01-27',feature_end='2023-11-30',raw_end='2023-12-29');return x,of.worker()
    def test_opch_post_overlay_2023_warmup_fifteen_bars_and_null_volume(self):
        x,w=self.opch_loaded();a=self.apply(x,JointSourceRepairs(opch=w));self.assertEqual(a.feature_prices['volume'].null_count,15);self.assertEqual(a.feature_prices['adj_volume'].null_count,15);self.assertEqual(a.feature_prices['close'][1].as_py(),100.);self.assertEqual(a.plan.fingerprint,walkforward_fingerprint(a.plan));self.assertEqual(a.provenance['joint_source_repairs']['base_quote_amendments_fingerprint'],'frozen_historical_plus_recent_overlay');self.assertEqual(a.provenance['joint_source_repairs']['opch_applied_rows']['feature'],15)
    def test_opch_only_protected_cohort_metadata_is_refused(self):
        x,w=self.opch_loaded();m=pa.Table.from_pylist([row('opch_source','sec_zmox6E','2024-01-02')]);p=replace(x.plan,original_cohort=m);p=replace(p,fingerprint=walkforward_fingerprint(p));x=replace(x,plan=p,provenance={**x.provenance,'plan_fingerprint':p.fingerprint});self.assertRaisesRegex(RepairError,'Protected',self.apply,x,JointSourceRepairs(opch=w))
    def test_joint_ep_state_no_target_later_year_does_not_append_ep(self):
        x,w=self.opch_loaded();a=self.apply(x,JointSourceRepairs(self.route,self.src,w));self.assertNotIn(NEW,a.plan.requested_ids);self.assertEqual(a.provenance['joint_source_repairs']['correct_native_added_rows'],{'feature':0,'outcome':0});self.assertEqual(a.feature_prices['volume'].null_count,15)
if __name__=='__main__':unittest.main()
