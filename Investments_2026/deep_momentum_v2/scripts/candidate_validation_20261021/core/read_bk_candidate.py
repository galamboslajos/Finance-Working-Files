"""Explicit bounded BK field audit; financial Arrow masking precedes Python."""
import os
for _n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[_n]='1'
import argparse,hashlib,json,resource,struct,sys,time
from pathlib import Path
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from bk_field_audit import COLUMNS,KEYS,NATIVE,START,END,audit_rows,key_and_warmup_summary,need


def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def pin(p):
 p=Path(p).resolve();return {'path':str(p),'sha256':sha(p),'size_bytes':p.stat().st_size}

def checked(p):
 q=Path(p['path']);need(q.is_file() and q.stat().st_size==p['size_bytes'] and sha(q)==p['sha256'],'Frozen source/code pin changed');return q

def write(p,obj):
 need(not Path(p).exists(),'Preserve previous artifact');Path(p).write_text(json.dumps(obj,sort_keys=True,indent=2,allow_nan=False)+'\n')

def mask(table,start,end,sessions=None):
 keep=pc.and_(pc.equal(table['intrinio_security_id'],NATIVE),pc.and_(pc.greater_equal(table['trading_date'],start),pc.less_equal(table['trading_date'],end)))
 if sessions is not None:keep=pc.and_(keep,pc.is_in(table['trading_date'],value_set=pa.array(sessions)))
 return table.filter(keep)

def run(config_path,output):
 began=time.monotonic();cfg=json.loads(Path(config_path).read_text());plan_path=checked(cfg['plan_pin']);plan=json.loads(plan_path.read_text())
 for p in cfg['code_pins']:checked(p)
 need(cfg['schema']=='dmv2-BK-field-config/v1' and plan['schema']=='dmv2-BK-field-plan/v1'
      and plan['financial_start']==START and plan['financial_end']==END and plan['native']==NATIVE
      and plan['financial_projection']==list(COLUMNS) and plan['metadata_projection']==list(KEYS),'Frozen BK field scope changed')
 protocol=json.loads(checked(plan['protocol_pin']).read_text());limits=protocol['budget']
 need(protocol['status']=='FROZEN_SCOPE_REQUIRES_EXACT_SUBPLANS_BEFORE_FINANCIAL_READS'
      and limits['threads']==1 and limits['automatic_retries']==0,'Unfrozen authority or expanded execution')
 for p in protocol['original_pins']:checked({**p,'path':str(Path(protocol['root'])/p['path'])})
 for name in ('calendar_pin','prior_keys_pin','prior_key_receipt_pin','footer_plan_pin'):checked(plan[name])
 calendar=json.loads(Path(plan['calendar_pin']['path']).read_text());sessions=calendar['sessions']
 need(calendar['package_version']=='4.13.2' and calendar['calendar_implementation']=='exchange_calendars.exchange_calendar_xnys','Independent calendar contract changed')
 import exchange_calendars as xc
 from importlib.metadata import version
 need(version('exchange-calendars')=='4.13.2','Calendar package version changed')
 package=Path(xc.__file__).resolve().parent
 for name,digest in calendar['package_source_hashes'].items():need(sha(package/name)==digest,'Calendar code changed')
 source=plan['source_object'];path=checked(source);pf=pq.ParquetFile(path)
 with path.open('rb') as f:f.seek(source['footer_offset']);raw=f.read(source['footer_size_bytes'])
 need(hashlib.sha256(raw).hexdigest()==source['footer_sha256'],'Physical footer changed')
 need(pf.metadata.num_rows==plan['candidate_rows'] and pf.metadata.num_row_groups==1,'Source rowgroup changed')
 need([[c,str(pf.schema_arrow.field(c).type)] for c in COLUMNS]==plan['projected_schema'],'Source projection schema changed')
 def cost(cols):return sum(pf.metadata.row_group(0).column(pf.schema_arrow.names.index(c)).total_compressed_size for c in cols)
 prior_pf=pq.ParquetFile(plan['prior_keys_pin']['path'])
 prior_cost=sum(prior_pf.metadata.row_group(i).column(prior_pf.schema_arrow.names.index(c)).total_compressed_size
     for i in range(prior_pf.metadata.num_row_groups) for c in KEYS)
 need(cost(KEYS)==plan['metadata_projected_bytes'] and cost(COLUMNS)==plan['financial_projected_bytes']
      and prior_cost==plan['prior_key_projected_bytes']
      and plan['total_projected_bytes']==cost(KEYS)+cost(COLUMNS)+prior_cost<=limits['max_BK_projected_bytes'], 'Unreserved projection or BK budget exceeded')
 need(plan['returned_financial_rows_upper_bound']<=limits['max_financial_returned_rows_per_case'],'Returned financial upperbound exceeded')
 def guard():
  rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
  need(rss<=limits['max_rss_bytes'] and time.monotonic()-began<=limits['max_worker_wall_seconds'],'Resource ceiling reached');return rss
 output=Path(output);need(not output.exists(),'Preserve prior attempt');output.mkdir(parents=True)
 write(output/'read_reservation.json',{'schema':'dmv2-BK-field-read-reservation/v1','config_pin':pin(config_path),'plan_pin':pin(plan_path),
  'protocol_pin':plan['protocol_pin'],'metadata_projected_bytes':plan['metadata_projected_bytes'],'financial_projected_bytes':plan['financial_projected_bytes'],
  'total_projected_bytes':plan['total_projected_bytes'],'candidate_rows':plan['candidate_rows'],'maximum_financial_output_rows':plan['returned_financial_rows_upper_bound'],
  'numeric_Python_mask':'native=sec_zq1J05 AND2007-07-02<=date<=2023-12-31 AND independentXNYSsession',
  'before_any_source_column_decode':True})
 pa.set_cpu_count(1);pa.set_io_thread_count(1)
 meta=pf.read(columns=list(KEYS),use_threads=False)
 need(meta['intrinio_security_id'].null_count==0 and meta['trading_date'].null_count==0,'Null source date/native metadata')
 # Source date validity is established in Arrow before any financial-column read.
 dates=meta['trading_date'];parsed=pc.strptime(dates,format='%Y-%m-%d',unit='s',error_is_null=False)
 need(pc.all(pc.equal(pc.strftime(parsed,format='%Y-%m-%d'),dates)).as_py(),'Malformed source date metadata')
 bounded_meta=mask(meta,'2007-01-01',END);metadata_rows=bounded_meta.to_pylist()
 prior=prior_pf.read(columns=list(KEYS),use_threads=False).to_pylist()
 original_keys=sorted(r['trading_date'] for r in prior if r['intrinio_security_id']==NATIVE)
 need(sorted(r['trading_date'] for r in metadata_rows)==original_keys and len(set(original_keys))==len(original_keys)==4289,'Candidate keys differ from original authenticated inventory')
 warmup=key_and_warmup_summary(original_keys,sessions);guard()
 finance=pf.read(columns=list(COLUMNS),use_threads=False)
 finance=mask(finance,START,END,[d for d in sessions if START<=d<=END])
 need(finance.num_rows<=plan['returned_financial_rows_upper_bound'],'Financial output upperbound exceeded; no truncation')
 expected=sorted(d for d in original_keys if START<=d<=END and d in set(sessions))
 need(sorted(finance['trading_date'].to_pylist())==expected,'Financial exactkey projection changed')
 rows=finance.to_pylist();summary,quality,jumps=audit_rows(rows,sessions);guard()
 # Raw immutable file bytes rechecked; no new numerical projection is needed.
 checked(source)
 for name in ('calendar_pin','prior_keys_pin','prior_key_receipt_pin','footer_plan_pin','protocol_pin'):checked(plan[name])
 for p in cfg['code_pins']:checked(p)
 checked(cfg['plan_pin'])
 pq.write_table(finance,output/'candidate_postmerger_XNYS_rows.parquet')
 pq.write_table(pa.Table.from_pylist(quality),output/'row_quality.parquet')
 write(output/'discontinuities.json',jumps);write(output/'key_and_warmup_summary.json',warmup)
 summary.update({'schema':'dmv2-BK-field-validation-result/v1','status':'COMPLETE_DIAGNOSTIC_AUDIT_NOT_PROMOTED',
  'plan_pin':pin(plan_path),'config_pin':pin(config_path),'source_pin':{k:source[k] for k in ('path','size_bytes','sha256')},
  'total_projected_bytes':plan['total_projected_bytes'],'financial_fields_read':list(COLUMNS),
  'financial_start':START,'financial_end':END,'financial_Python_output_XNYS_only':True,
  'premerger_financial_values_selected_or_converted':False,'provider_only_date_financial_values_selected_or_converted':False,
  'mixed_whole_Parquet_column_chunks_may_transiently_decode_excluded_2024plus_or_other_native_values':True,
  'excluded_2024plus_financial_values_returned_converted_or_analyzed':False,
  'source_bounds_quality_provenance_preserved':True,'auto_approved_for_consumption':False,'peak_rss_bytes':guard(),'elapsed_seconds':time.monotonic()-began,
  'limitations':['Field diagnostics do not establish independent economic totalreturn or dated ordinaryshare identity.',
   'Returned ticker/FIGI/provenance fields are retrieval-time metadata and never past cohort admission.',
   'Null action values remain unknown; no dividend doublecount or inverse price-factor volume adjustment is imposed.',
   'Large moves and adjustment changes are review flags, not automatic exclusions or invented repairs.',
   'Pre-July2,2007 prices and all provider-only dates remain metadata-only. No production mapping, inputs or runs changed.']})
 write(output/'summary.json',summary)
 need(sum(p.stat().st_size for p in output.iterdir())<=limits['max_output_bytes'],'Output byte limit exceeded')
 write(output/'receipt.json',{'status':'COMPLETE_DIAGNOSTIC_AUDIT_NOT_PROMOTED','summary_pin':pin(output/'summary.json'),
  'candidate_rows_pin':pin(output/'candidate_postmerger_XNYS_rows.parquet'),'row_quality_pin':pin(output/'row_quality.parquet'),
  'discontinuities_pin':pin(output/'discontinuities.json'),'key_and_warmup_pin':pin(output/'key_and_warmup_summary.json'),
  'financial_rows':len(rows),'projected_bytes':plan['total_projected_bytes'],'source_and_code_pins_unchanged':True,
  'canonical_mapping_promoted':False,'features_targets_models_or_backtests_built':False})
 print(json.dumps({'status':summary['status'],'financial_rows':len(rows),'rows_with_field_flags':summary['rows_with_field_flags'],
  'source_factor_direct_max_relative_residual':summary['source_factor_direct_max_relative_residual']}))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);args=p.parse_args();run(args.config,args.output)
