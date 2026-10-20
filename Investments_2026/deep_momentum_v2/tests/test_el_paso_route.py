"""Structural metadata fixtures only; no source prices or financial outcomes."""
import hashlib,json,tempfile,unittest
from pathlib import Path
import pyarrow as pa
from dmv2.el_paso_route import *

def primary():
    def f(url,issuer,cusip,when):return {'url':url,'issuer':issuer,'ticker':'EP','cusip':cusip,'fact_date':when,'evidence_scope':'issuer_and_security_identifier_only'}
    return {'schema':'dmv2-el-paso-primary-findings/v1','status':'REVIEWED_PRIMARY_SOURCE_FACTS','record_kind':'documented_primary_source_findings_not_raw_provider_response','el_paso':f(EL_PASO_URL,'El Paso Corporation','28336L109','2011-05-17'),'empire':f(EMPIRE_URL,'Empire Petroleum Corporation','292034303','2022-03-08')}

def proposal():
    return {'schema':'dmv2-el-paso-identity-route-override-proposal/v1','status':'REVIEW_READY_NONPROMOTING_NOT_APPLIED','source_security_id':SOURCE,'source_named_issuer':'El Paso Corp.','source_membership_labels':['EP'],'original_native_id':OLD,'original_native_issuer':'Empire Petroleum Corporation','original_provider_lookup_isin':'US2920343033','original_native_figi':DEFINITION['original_figi'],'candidate_native_id':NEW,'candidate_native_issuer':'El Paso Corporation','candidate_provider_lookup_isin':'US28336L1098','candidate_native_figi':DEFINITION['replacement_figi'],'candidate_security_master_last_quote':END,'applicability':{},'collision_check':{},'evidence':[],'replacement_history_already_downloaded':True,'required_before_new_build':[],'source_incorrect_route_confirmed':True,'newproviderlookup_performed':False,'canonical_identity_promoted':False,'old_artifacts_modified':False,'model_fits':0}

def row(source=SOURCE,native=OLD,when='2010-01-29'):
    return {'source_security_id':source,'intrinio_security_id':native,'unified_security_id':UNIFIED if source==SOURCE else 'u_other','signal_date':when,'identity_status':'needs_review','source_identity_status':'needs_review','identity_review_required':True,'resolution_status':'resolved','identity_eligible':True,'identity_eligibility_policy':'provisional_unique_resolved_link','identity_eligibility_reason':'provisional_review_flag_accepted','index_id':'sp500'}

class RouteTest(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);self.root=Path(tmp.name)
        self.data={'proposal':proposal(),'primary_findings':primary(),'provider_metadata':[{'intrinio_security_id':native,'figi':figi,'requested_identifier':native,'ticker':'EP','source_relative_path':'raw_pages/query_identifier='+isin+'-abc/security.json.gz','source_response_hash':'a'*64,'source_run_id':'synthetic_metadata_fixture'} for native,figi,isin in [(OLD,DEFINITION['original_figi'],'US2920343033'),(NEW,DEFINITION['replacement_figi'],'US28336L1098')]],'source_bridge_metadata':[{'source_security_id':SOURCE,'source_product_id':'sp_composite1500','unified_security_id':UNIFIED,'identity_key':'ticker_company:EP|EL PASO','company_name':'El Paso Corp.','first_membership_date':'1999-10-26','last_membership_date':END}]}
        for k,v in self.data.items():self.write(k+'.json',v)
        self.pins={k:artifact_pin(self.root,self.root/(k+'.json'),'synthetic_metadata_fixture:'+k) for k in ARTIFACT_ROLES};self.manifest=build_route_manifest(evidence_root=self.root,artifacts=self.pins,reviewed=True);self.save_manifest();self.table=pa.Table.from_pylist([row(),row('other','sec_other')])
    def write(self,path,obj):self.root.joinpath(path).write_text(json.dumps(obj,indent=2)+'\n')
    def save_manifest(self):self.write('manifest.json',self.manifest);self.pin=sha(self.root/'manifest.json')
    def worker(self):return ElPasoRouteAmendment.from_manifest(evidence_root=self.root,manifest_relative_path='manifest.json',expected_sha256=self.pin)
    def apply(self,t=None,**kw):return self.worker().apply_with_ledger(self.table if t is None else t,all_original_members=True,**kw)
    def test_exact_source_route_only_and_all_flags_unified_values_preserved(self):
        r=self.apply();self.assertEqual(r.table.num_rows,2);self.assertEqual(r.table['intrinio_security_id'].to_pylist(),[NEW,'sec_other']);self.assertEqual(len(r.ledger),1)
        for c in self.table.column_names:
            if c!='intrinio_security_id':self.assertTrue(r.table[c].equals(self.table[c]))
        self.assertEqual(r.table['identity_status'][0].as_py(),'needs_review');self.assertTrue(r.table['identity_review_required'][0].as_py());self.assertFalse(r.table['el_paso_route_historical_identity_certified'][0].as_py())
    def test_idempotent_exact_new_route_and_audit(self):
        a=self.apply().table;b=self.apply(a);self.assertTrue(a.equals(b.table));self.assertEqual(b.ledger[0]['state'],'already_applied_exact')
    def test_new_without_audit_refused(self):
        with self.assertRaisesRegex(RouteError,'lacks exact idempotent'):self.apply(pa.Table.from_pylist([row(native=NEW)]))
    def test_unknown_native_not_overwritten(self):
        with self.assertRaisesRegex(RouteError,'Unknown El Paso'):self.apply(pa.Table.from_pylist([row(native='unknown')]))
    def test_unrelated_genuine_empire_route_preserved(self):
        t=pa.Table.from_pylist([row(),row('genuine_empire',OLD)]);r=self.apply(t);self.assertEqual(r.table['intrinio_security_id'].to_pylist(),[NEW,OLD]);ids=self.worker().requested_ids(t,all_original_members=True);self.assertEqual(ids,tuple(sorted([NEW,OLD])))
    def test_collision_with_blocked_original_member_refuses_entire_amendment(self):
        other=row('blocked',NEW);other['identity_eligible']=False;other['identity_status']='unresolved'
        t=pa.Table.from_pylist([row(),other]);before=t
        with self.assertRaisesRegex(RouteError,'collides with another ORIGINAL'):self.apply(t)
        self.assertTrue(before.equals(t))
    def test_other_date_candidate_route_not_collision(self):
        t=pa.Table.from_pylist([row(),row('other',NEW,'2010-02-26')]);self.assertEqual(len(self.apply(t).ledger),1)
    def test_same_original_source_across_sleeves_retained(self):
        b=dict(row(),index_id='sp400');t=pa.Table.from_pylist([row(),b]);r=self.apply(t);self.assertEqual(len(r.ledger),2);self.assertEqual(r.summary['distinct_target_source_dates'],1);self.assertEqual(r.table.num_rows,2)
    def test_pre_warmup_target_refused(self):
        with self.assertRaisesRegex(RouteError,'outside approved'):self.apply(pa.Table.from_pylist([row(when='2008-01-03')]))
    def test_post_terminal_target_refused(self):
        with self.assertRaisesRegex(RouteError,'outside approved'):self.apply(pa.Table.from_pylist([row(when='2012-05-25')]))
    def test_exact_scope_boundaries_allowed(self):
        t=pa.Table.from_pylist([row(when=START),row(when=END)]);self.assertEqual(len(self.apply(t).ledger),2)
    def test_protected_unrelated_row_also_refused(self):
        with self.assertRaisesRegex(RouteError,'Protected'):self.apply(pa.Table.from_pylist([row(),row('other','sec_other','2024-01-02')]))
    def test_wrong_unified_refused(self):
        with self.assertRaisesRegex(RouteError,'unified ID changed'):self.apply(pa.Table.from_pylist([dict(row(),unified_security_id='wrong')]))
    def test_producer_promotion_or_review_change_refused(self):
        for changes in [{'identity_status':'resolved'},{'identity_review_required':False},{'resolution_status':'unresolved'},{'source_identity_status':'resolved'}]:
            with self.subTest(changes=changes),self.assertRaises(RouteError):self.apply(pa.Table.from_pylist([dict(row(),**changes)]))
    def test_all_original_scope_assertion_required(self):
        with self.assertRaisesRegex(RouteError,'ALL original'):self.worker().apply(self.table)
    def test_original_denominators_must_match(self):
        with self.assertRaisesRegex(RouteError,'denominators differ'):self.apply(original_counts={'2010-01-29':3})
        self.assertTrue(self.apply(original_counts={'2010-01-29':2}).summary['original_denominators_supplied'])
    def test_daily_asof_and_trading_date_supported(self):
        for c in ['as_of','trading_date']:
            t=self.table.rename_columns([c if n=='signal_date' else n for n in self.table.column_names]);self.assertEqual(len(self.apply(t,date_column=c).ledger),1)
    def test_conflicting_date_alias_refused(self):
        t=self.table.append_column('as_of',pa.array(['2010-02-01']*2))
        with self.assertRaisesRegex(RouteError,'Conflicting cohort'):self.apply(t)
    def test_null_native_other_member_preserved(self):
        t=pa.Table.from_pylist([row(),row('unlinked',None)]);self.assertIsNone(self.apply(t).table['intrinio_security_id'][1].as_py())
    def test_null_source_refused(self):
        with self.assertRaisesRegex(RouteError,'stable source identity'):self.apply(pa.Table.from_pylist([row(),row(None,'sec_other')]))
    def test_partial_audit_schema_refused(self):
        t=self.table.append_column('el_paso_route_amendment_id',pa.array([None,None],type=pa.string()))
        with self.assertRaisesRegex(RouteError,'Partial route audit'):self.apply(t)
    def test_foreign_audit_on_unrelated_source_refused(self):
        t=self.apply().table;v=t['el_paso_route_amendment_id'].to_pylist();v[1]='foreign';t=t.set_column(t.schema.get_field_index('el_paso_route_amendment_id'),'el_paso_route_amendment_id',pa.array(v))
        with self.assertRaisesRegex(RouteError,'unrelated source'):self.apply(t)
    def test_changed_evidence_after_loading_refused(self):
        w=self.worker();(self.root/'provider_metadata.json').write_text('[]')
        with self.assertRaisesRegex(RouteError,'pin changed'):w.apply(self.table,all_original_members=True)
    def test_changed_manifest_even_if_repinned_refused(self):
        self.manifest['definition']=dict(DEFINITION,maximum_cohort_date='2023-12-29');self.save_manifest()
        with self.assertRaisesRegex(RouteError,'Closed admitted'):self.worker()
    def test_unreviewed_manifest_refused(self):
        self.manifest['status']='REVIEW_READY_NOT_APPLIED';self.save_manifest()
        with self.assertRaises(RouteError):self.worker()
    def test_provider_cached_figi_or_lookup_path_refused(self):
        self.data['provider_metadata'][1]['figi']='wrong';self.write('provider_metadata.json',self.data['provider_metadata']);self.pins['provider_metadata']=artifact_pin(self.root,self.root/'provider_metadata.json','fixture')
        with self.assertRaisesRegex(RouteError,'FIGI/query'):build_route_manifest(evidence_root=self.root,artifacts=self.pins)
    def test_wrong_primary_cusip_refused(self):
        self.data['primary_findings']['el_paso']['cusip']='292034303';self.write('primary_findings.json',self.data['primary_findings']);self.pins['primary_findings']=artifact_pin(self.root,self.root/'primary_findings.json','fixture')
        with self.assertRaisesRegex(RouteError,'CUSIP facts'):build_route_manifest(evidence_root=self.root,artifacts=self.pins)
    def test_unknown_proposal_field_refused(self):
        self.data['proposal']['extra']='scope expansion';self.write('proposal.json',self.data['proposal']);self.pins['proposal']=artifact_pin(self.root,self.root/'proposal.json','fixture')
        with self.assertRaisesRegex(RouteError,'Closed exact'):build_route_manifest(evidence_root=self.root,artifacts=self.pins)
    def test_no_price_source_admission_implied_by_requested_ids(self):
        ids=self.worker().requested_ids(self.table,all_original_members=True);self.assertIn(NEW,ids);self.assertNotIn(OLD,ids)
        r=self.apply();self.assertFalse(r.summary['canonical_identity_promoted']);self.assertFalse(r.summary['historical_knowledge_time_certified'])
    def test_admitted_requests_retain_only_existing_eligibility(self):
        t=pa.Table.from_pylist([row(),dict(row('other','sec_other'),identity_eligible=False)]);self.assertEqual(self.worker().requested_ids(t,all_original_members=True),(NEW,))

if __name__=='__main__':unittest.main()
