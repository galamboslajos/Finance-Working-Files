"""All quotes in these tests are synthetic; dates/ID are structural scope keys."""
import hashlib,json,tempfile,unittest
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from dmv2.opch_amendment import *

class OpchAmendmentTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.original=[];sip=[];targets=[]
        for d in ['2022-01-27',*DATES,'2022-03-11']:
            bad=d in DATES;value=50. if bad else 100.
            row={'intrinio_security_id':NATIVE,'trading_date':d,'frequency':'daily','intraperiod':False,'volume':7.,'adj_volume':7.,'factor':1.,'split_ratio':1.,'dividend':0.,'raw_ohlc_bounds_valid':True,'adjusted_ohlc_bounds_valid':True,'price_quality_status':'accepted','source_page_hash':'a'*64}
            row.update({c:value for c in OHLC+ADJ});self.original.append(row)
            independent={'security_id':SOURCE,'trading_date':d,'query_symbol':'OPCH','feed':'sip','adjustment':'raw','timeframe':'1Day','source_page_hash':'b'*64};independent.update({c:100. for c in OHLC});sip.append(independent)
            if bad:targets.append({'intrinio_security_id':NATIVE,'source_security_id':SOURCE,'trading_date':d,'original_raw_ohlc':{c:50. for c in OHLC},'independent_raw_ohlc':{c:100. for c in OHLC},'preceding_full_ohlc_match_control':'2022-01-27','following_full_ohlc_match_control':'2022-03-11','paired_multiplier_from_preceding':1.,'maximum_following_adjusted_control_residual':0.,'original_volume':7.,'original_adj_volume':7.,'proposed_volume':None,'proposed_adj_volume':None,'other_source_fields_unchanged':True,'original_source_page_hash':'a'*64,'independent_source_page_hash':'b'*64})
        pq.write_table(pa.Table.from_pylist(self.original),self.root/'original.parquet');pq.write_table(pa.Table.from_pylist(sip),self.root/'sip.parquet');(self.root/'source.bin').write_bytes(b'fictional source object for structural tests only')
        self.write('proposal.json',{'schema':'dmv2-opch-confirmed-contamination-repair-evidence/v1','target_count':15,'targets':targets})
        self.write('provenance.json',{'source_objects_hash_reread_matched':True,'provenance':{'objects':[{'sha256':'sha256:'+hashlib.sha256((self.root/'source.bin').read_bytes()).hexdigest()}]}})
        self.write('receipt.json',{'object':{'uri':SOURCE_URI,'generation':SOURCE_GENERATION,'sha256':SOURCE_SHA},'output_sha256':hashlib.sha256((self.root/'sip.parquet').read_bytes()).hexdigest(),'metadata_before_financial':True,'no_unrequested_financial_rows_returned_or_converted':True})
        self.write('publication.json',{'files':[{'uri':SOURCE_URI,'generation':SOURCE_GENERATION,'sha256':SOURCE_SHA}]})
        self.artifacts={k:artifact_pin(self.root,self.root/f,'synthetic-fixture:'+k) for k,f in {'proposal':'proposal.json','original_projection':'original.parquet','original_provenance':'provenance.json','original_source':'source.bin','independent_projection':'sip.parquet','independent_receipt':'receipt.json','publication_manifest':'publication.json'}.items()}
        self.manifest=build_manifest(evidence_root=self.root,artifacts=self.artifacts,reviewed=True);self.save_manifest()
        self.table=pa.Table.from_pylist(self.original).select(list(CORE)).append_column('unchanged_audit_field',pa.array(list(range(len(self.original)))))
    def write(self,path,obj):self.root.joinpath(path).write_text(json.dumps(obj,indent=2)+'\n')
    def save_manifest(self):
        self.write('manifest.json',self.manifest);self.sha='sha256:'+hashlib.sha256((self.root/'manifest.json').read_bytes()).hexdigest()
    def worker(self):return ReviewedOpchAmendment.from_manifest(evidence_root=self.root,manifest_relative_path='manifest.json',expected_sha256=self.sha)
    def changed(self,table,column,index,value):
        vals=table[column].to_pylist();vals[index]=value;return table.set_column(table.schema.get_field_index(column),table.schema.field(column),pa.array(vals,type=table[column].type))
    def apply(self,table=None):return self.worker().apply_with_ledger(self.table if table is None else table,start='2022-01-01',end='2022-03-31')
    def test_exact_fifteen_repaired_volume_null_and_other_fields_preserved(self):
        r=self.apply();self.assertEqual(r.table.num_rows,self.table.num_rows);self.assertEqual(r.table.column_names,self.table.column_names);self.assertEqual(len(r.ledger),15)
        for a,b in zip(self.table.to_pylist(),r.table.to_pylist()):
            if a['trading_date'] in DATES:
                for c in OHLC+ADJ:self.assertEqual(b[c],100.)
                self.assertIsNone(b['volume']);self.assertIsNone(b['adj_volume'])
            else:self.assertEqual(a,b)
            for c in self.table.column_names:
                if c not in CHANGED:self.assertEqual(a[c],b[c])
        self.assertEqual(self.table['volume'].null_count,0)
    def test_idempotent_exact_replacement(self):
        a=self.apply().table;b=self.apply(a);self.assertTrue(a.equals(b.table));self.assertTrue(all(r['state']=='already_applied_exact' for r in b.ledger))
    def test_partial_replacement_refused(self):
        with self.assertRaisesRegex(AmendmentError,'partial/unreviewed'):self.apply(self.changed(self.table,'open',1,100.))
    def test_changed_original_volume_refused(self):
        with self.assertRaisesRegex(AmendmentError,'source fields differ'):self.apply(self.changed(self.table,'volume',1,8.))
    def test_changed_original_adjusted_price_refused(self):
        with self.assertRaisesRegex(AmendmentError,'source fields differ'):self.apply(self.changed(self.table,'adj_close',1,51.))
    def test_changed_present_control_refused(self):
        with self.assertRaisesRegex(AmendmentError,'control source fields changed'):self.apply(self.changed(self.table,'open',0,101.))
    def test_protected_input_date_refused(self):
        with self.assertRaisesRegex(AmendmentError,'Protected'):self.worker().apply(self.changed(self.table,'trading_date',0,'2024-01-02'),start='2022-01-01',end='2023-12-29')
    def test_protected_request_refused(self):
        with self.assertRaisesRegex(AmendmentError,'Protected'):self.worker().apply(self.table,start='2022-01-01',end='2024-01-01')
    def test_out_of_request_rows_refused(self):
        with self.assertRaisesRegex(AmendmentError,'outside explicit request'):self.worker().apply(self.table,start='2022-02-01',end='2022-03-31')
    def test_subset_does_not_append_missing_keys(self):
        row=self.table.slice(1,1);r=self.worker().apply_with_ledger(row,start=DATES[0],end=DATES[0]);self.assertEqual(r.table.num_rows,1);self.assertEqual(len(r.ledger),1)
    def test_other_native_and_non_target_unchanged(self):
        t=self.table.set_column(0,'intrinio_security_id',pa.array(['other']*len(self.table)));r=self.apply(t);self.assertTrue(t.equals(r.table));self.assertEqual(r.ledger,())
    def test_duplicate_input_keys_refused(self):
        with self.assertRaisesRegex(AmendmentError,'Duplicate native'):self.apply(pa.concat_tables([self.table,self.table.slice(1,1)]))
    def test_null_native_refused(self):
        with self.assertRaisesRegex(AmendmentError,'Null or malformed'):self.apply(self.changed(self.table,'intrinio_security_id',0,None))
    def test_missing_schema_refused(self):
        with self.assertRaisesRegex(AmendmentError,'full original'):self.apply(self.table.drop(['volume']))
    def test_manifest_sha_change_refused(self):
        w=self.worker();(self.root/'manifest.json').write_text((self.root/'manifest.json').read_text()+' ')
        with self.assertRaisesRegex(AmendmentError,'SHA256 changed'):w.apply(self.table,start='2022-01-01',end='2022-03-31')
    def test_evidence_pin_change_after_loading_refused(self):
        w=self.worker();(self.root/'source.bin').write_bytes(b'changed source')
        with self.assertRaisesRegex(AmendmentError,'Evidence size or path changed'):w.apply(self.table,start='2022-01-01',end='2022-03-31')
    def test_fabricated_replacement_even_rehashed_manifest_refused(self):
        self.manifest['cases'][0]['replacement_record']['close']=999.;self.save_manifest()
        with self.assertRaisesRegex(AmendmentError,'differs from verified'):self.worker()
    def test_wrong_manifest_native_refused(self):
        self.manifest['native_id']='wrong';self.save_manifest()
        with self.assertRaisesRegex(AmendmentError,'identity or admission'):self.worker()
    def test_unreviewed_manifest_refused(self):
        self.manifest['status']='REVIEW_READY_NOT_APPLIED';self.save_manifest()
        with self.assertRaisesRegex(AmendmentError,'identity or admission'):self.worker()
    def test_expanded_target_scope_refused(self):
        p=json.loads((self.root/'proposal.json').read_text());p['targets'][0]['trading_date']='2022-01-26';self.write('proposal.json',p);self.artifacts['proposal']=artifact_pin(self.root,self.root/'proposal.json','synthetic')
        with self.assertRaisesRegex(AmendmentError,'exact reviewed OPCH'):build_manifest(evidence_root=self.root,artifacts=self.artifacts,reviewed=True)
    def test_path_escape_refused(self):
        a=dict(self.artifacts);a['proposal']=dict(a['proposal'],path='../escape.json')
        with self.assertRaisesRegex(AmendmentError,'traversal'):build_manifest(evidence_root=self.root,artifacts=a)
    def test_volume_schema_nullable_upgrade_only(self):
        fields=[f.with_nullable(False) if f.name in ('volume','adj_volume') else f for f in self.table.schema];t=pa.Table.from_arrays(self.table.columns,schema=pa.schema(fields,metadata=self.table.schema.metadata));r=self.apply(t).table
        self.assertTrue(r.schema.field('volume').nullable);self.assertTrue(r.schema.field('adj_volume').nullable)
        for name in t.column_names:
            if name not in ('volume','adj_volume'):self.assertEqual(t.schema.field(name),r.schema.field(name))
    def test_non_numeric_changed_column_refused(self):
        t=self.table.set_column(self.table.schema.get_field_index('open'),'open',pa.array([str(x) for x in self.table['open'].to_pylist()]))
        with self.assertRaises(AmendmentError):self.apply(t)
    def test_changed_price_precision_schema_refused(self):
        idx=self.table.schema.get_field_index('open');t=self.table.set_column(idx,'open',self.table['open'].cast(pa.float32()))
        with self.assertRaisesRegex(AmendmentError,'float64 OHLC'):self.apply(t)
    def test_protected_evidence_date_refused_before_financial_selection(self):
        extra=dict(self.original[0],trading_date='2024-01-02');pq.write_table(pa.Table.from_pylist(self.original+[extra]),self.root/'original.parquet');self.artifacts['original_projection']=artifact_pin(self.root,self.root/'original.parquet','synthetic')
        with self.assertRaisesRegex(AmendmentError,'Protected'):build_manifest(evidence_root=self.root,artifacts=self.artifacts,reviewed=True)
    def test_missing_evidence_target_refused(self):
        pq.write_table(pa.Table.from_pylist(self.original[1:]),self.root/'original.parquet');self.artifacts['original_projection']=artifact_pin(self.root,self.root/'original.parquet','synthetic')
        with self.assertRaisesRegex(AmendmentError,'exactly one evidence row'):build_manifest(evidence_root=self.root,artifacts=self.artifacts,reviewed=True)
    def test_source_publication_or_generation_mismatch_refused(self):
        r=json.loads((self.root/'receipt.json').read_text());r['object']['generation']=1;self.write('receipt.json',r);self.artifacts['independent_receipt']=artifact_pin(self.root,self.root/'receipt.json','synthetic')
        with self.assertRaisesRegex(AmendmentError,'identity/generation/hash'):build_manifest(evidence_root=self.root,artifacts=self.artifacts,reviewed=True)
    def test_independent_quote_volume_never_admitted(self):
        r=self.apply().table;self.assertEqual(r['volume'].null_count,15);self.assertNotIn('independent_volume',r.column_names)

if __name__=='__main__':unittest.main()
