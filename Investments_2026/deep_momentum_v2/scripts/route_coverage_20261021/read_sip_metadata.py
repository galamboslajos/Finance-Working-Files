from pathlib import Path
import hashlib,json,time,resource,sys
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
S=Path('/private/tmp/dmv2-route-coverage-20261021')
P=S/'sip_column_read_plan.json'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(P)=='7ffdeee6129db3080c8e0361aa72ab5cd2a51a52d094116cf28843b992b9959b'
plan=json.loads(P.read_text());proto=json.loads(Path(plan['protocol_pin']['path']).read_text());began=time.monotonic()
def check(item):
 p=Path(item['path']);assert p.stat().st_size==item['size_bytes'] and sha(p)==item['sha256'];return p
def write(p,a):
 assert not p.exists();p.write_text(json.dumps(a,sort_keys=True,indent=2,allow_nan=False,default=lambda x:x.isoformat())+'\n')
def resources():
 rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
 assert rss<=proto['bounds']['max_rss_bytes'] and time.monotonic()-began<=proto['bounds']['max_wall_seconds'];return rss
check(plan['protocol_pin']);check(plan['download_receipt_pin']);assert plan['financial_columns']==[] and plan['filter_symbols']==['XOM','BK']
assert plan['projected_compressed_bytes']<=1048576 and plan['candidate_rows']<=101074 and len(plan['tables'])==2
for entry in plan['tables']:
 p=check(entry['pin']);pf=pq.ParquetFile(p);allowed=next(x for x in proto['objects'] if x['table']==entry['table'])['allowed_columns']
 assert entry['columns']==allowed and pf.metadata.num_rows==entry['rows']
 assert [[c,str(pf.schema_arrow.field(c).type)] for c in entry['columns']]==entry['schema']
 cost=sum(pf.metadata.row_group(i).column(pf.schema_arrow.names.index(c)).total_compressed_size for i in range(pf.metadata.num_row_groups) for c in entry['columns'])
 assert cost==entry['projected_compressed_bytes']
resources();out=S/'sip_metadata_analysis-v1';assert not out.exists();out.mkdir()
write(out/'read_reservation.json',dict(plan_sha256=sha(P),worker_sha256=sha(Path(__file__)),protocol_sha256=sha(Path(plan['protocol_pin']['path'])),projected_compressed_bytes=plan['projected_compressed_bytes'],financial_columns=[],conversion_filter='literalXOM/BKbeforePython'))
pa.set_cpu_count(1);pa.set_io_thread_count(1);result=[]
for entry in plan['tables']:
 table=pq.ParquetFile(entry['pin']['path']).read(columns=entry['columns'],use_threads=False)
 mask=pa.array([False]*table.num_rows)
 for c in ['query_symbol','raw_membership_ticker']:
  if c in table.column_names:mask=pc.or_(mask,pc.fill_null(pc.is_in(table[c],value_set=pa.array(plan['filter_symbols'])),False))
 selected=table.filter(mask);assert selected.num_rows<=entry['output_row_limit'];rows=selected.to_pylist();resources()
 result.append(dict(table=entry['table'],source_rows=entry['rows'],selected_rows=len(rows),columns=entry['columns'],records=rows))
for entry in plan['tables']:check(entry['pin'])
write(out/'cases.json',result)
receipt=dict(schema='dmv2-sip-query-metadata-result/v1',status='PASS_FILTERED_METADATA_ONLY',plan_sha256=sha(P),worker_sha256=sha(Path(__file__)),metadata_rows_converted=sum(x['selected_rows'] for x in result),projected_compressed_bytes=plan['projected_compressed_bytes'],case_counts=[dict(table=x['table'],selected_rows=x['selected_rows']) for x in result],financial_columns_decoded=[],canonical_mapping_promoted=False,new_fits_inference_backtests=0,peak_rss_bytes=resources(),elapsed_seconds=time.monotonic()-began)
write(out/'receipt.json',receipt);print(json.dumps(receipt))
