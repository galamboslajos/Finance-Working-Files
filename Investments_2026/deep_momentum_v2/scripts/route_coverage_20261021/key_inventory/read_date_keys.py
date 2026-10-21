"""Frozen native-ID/date-key inventory only. No prices, identity promotion or fits."""
from __future__ import annotations
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_name]='1'
import argparse
from collections import Counter
from datetime import date
import hashlib,json,resource,struct,sys,time
from pathlib import Path
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

COLUMNS=['intrinio_security_id','trading_date']
IDS={'sec_z5YpGE','sec_gA3NYQ','sec_zq1J05'}
START='2007-01-01';END='2023-12-31'

class ContractError(ValueError):pass

def need(value,message):
    if not bool(value):raise ContractError(message)

def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()

def pin(path):
    p=Path(path).resolve();return {'path':str(p),'size_bytes':p.stat().st_size,'sha256':sha(p)}

def checked(item):
    p=Path(item['path']);need(p.is_file() and p.stat().st_size==item['size_bytes']
        and sha(p)==item['sha256'].removeprefix('sha256:'),'Frozen bytepin changed: '+str(p));return p

def write(path,value):
    need(not Path(path).exists(),'Preserve existing audit artifact')
    Path(path).write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n')

def canonical_day(value):
    need(isinstance(value,str),'Date keys must remain explicit strings')
    try:parsed=date.fromisoformat(value)
    except ValueError as error:raise ContractError('Malformed date key') from error
    need(parsed.isoformat()==value,'Noncanonical date key');return value

def included_group(group,ids,start,end):
    """Only safe key-field statistics may decide pruning; missing bounds include."""
    d0,d1=group.get('date_min'),group.get('date_max');n0,n1=group.get('native_min'),group.get('native_max')
    if d0 is not None and d1 is not None:
        canonical_day(d0);canonical_day(d1);need(d0<=d1,'Invalid safe date bounds')
    if n0 is not None and n1 is not None:
        need(isinstance(n0,str) and isinstance(n1,str) and n0<=n1,'Invalid safe native bounds')
    if d0 is not None and d1 is not None and (d1<start or d0>end):return False
    if n0 is not None and n1 is not None and not any(n0<=value<=n1 for value in ids):return False
    return True

def cumulative_usage(stage1,stage2,limits):
    mapping={'candidate_rows':'max_candidate_rows','data_files':'max_data_files','footer_files':'max_footer_files',
        'projected_compressed_bytes':'max_projected_column_bytes','metadata_rows_converted':'max_returned_metadata_rows'}
    total={key:int(stage1[key])+int(stage2[key]) for key in mapping}
    need(all(value>=0 and value<=limits[mapping[key]] for key,value in total.items()),'Cumulative stage1+stage2 budget exceeded')
    return total

def inspect_footer(obj):
    path=Path(obj['path']);need(path.is_file() and path.stat().st_size==obj['size_bytes'],'Source file size changed')
    with path.open('rb') as handle:
        handle.seek(-8,2);tail=handle.read(8)
        need(tail[4:]==b'PAR1','Not a Parquet footer')
        length=struct.unpack('<I',tail[:4])[0];offset=path.stat().st_size-8-length
        handle.seek(offset);raw=handle.read(length)
    need(length==len(raw)==obj['footer_size_bytes'] and offset==obj['footer_offset']
        and hashlib.sha256(raw).hexdigest()==obj['footer_sha256'],'Raw footer bytepin changed')
    pf=pq.ParquetFile(path)
    schema=[[c,str(pf.schema_arrow.field(c).type)] for c in COLUMNS]
    need(schema==obj['key_schema']==[[c,'string'] for c in COLUMNS],'Native/date key schema changed')
    need(pf.metadata.num_row_groups==len(obj['groups']),'Source row-group count changed')
    groups=[]
    for index in range(pf.metadata.num_row_groups):
        rg=pf.metadata.row_group(index);row={'index':index,'rows':rg.num_rows,'projected_compressed_bytes':0}
        for name,prefix in [('intrinio_security_id','native'),('trading_date','date')]:
            # No financial-column statistics are accessed.
            column=rg.column(pf.schema_arrow.names.index(name));stats=column.statistics
            row['projected_compressed_bytes']+=column.total_compressed_size
            row[prefix+'_min']=stats.min if stats is not None and stats.has_min_max else None
            row[prefix+'_max']=stats.max if stats is not None and stats.has_min_max else None
        row['included']=included_group(row,IDS,START,END);groups.append(row)
    need(groups==obj['groups'],'Safe native/date footer pruning changed')
    return pf,groups

def filtered_keys(table,ids,start,end):
    need(table.column_names==COLUMNS and set(ids)<=IDS and ids,'Exact key-only allowlist required')
    need(start==START and end==END,'Frozen date bounds changed')
    if table.num_rows==0:return table
    need(all(table[c].null_count==0 for c in COLUMNS),'Null native/date keys fail closed')
    need(pc.all(pc.not_equal(pc.utf8_trim_whitespace(table[COLUMNS[0]]),'' )).as_py(),'Blank native key')
    need(pc.all(pc.match_substring_regex(table[COLUMNS[1]],r'^\d{4}-\d{2}-\d{2}$')).as_py(),'Malformed date key')
    try:parsed=pc.strptime(table[COLUMNS[1]],format='%Y-%m-%d',unit='s',error_is_null=False)
    except pa.ArrowInvalid as error:raise ContractError('Malformed calendar date key') from error
    need(pc.all(pc.equal(pc.strftime(parsed,format='%Y-%m-%d'),table[COLUMNS[1]])).as_py(),
        'Normalized or impossible calendar date key')
    mask=pc.and_(pc.is_in(table[COLUMNS[0]],value_set=pa.array(sorted(ids))),pc.and_(
        pc.greater_equal(table[COLUMNS[1]],start),pc.less_equal(table[COLUMNS[1]],end)))
    return table.filter(mask)

def validate_keys(rows,ids,start,end):
    need(set(ids)==IDS and start==START and end==END,'No native substitution or date widening')
    keys=[]
    for row in rows:
        need(set(row)==set(COLUMNS) and row[COLUMNS[0]] in ids,'Unexpected source key')
        day=canonical_day(row[COLUMNS[1]]);need(start<=day<=end,'Out-of-bound returned date key')
        keys.append((row[COLUMNS[0]],day))
    need(len(keys)==len(set(keys)),'Duplicate native/date keys fail closed')
    return sorted(keys)

def anchor_summary(keys,windows,cases):
    need(len(windows)==168 and len({w['outcome_month'] for w in windows})==168,'Exact168 original outcome windows required')
    need([w['outcome_month'] for w in windows]==[f'{y}-{m:02d}' for y in range(2010,2024) for m in range(1,13)],'Original monthly calendar changed')
    for w in windows:
        for field in ('signal_date','outcome_start','outcome_end'):canonical_day(w[field])
        need('2009-12-01'<=w['signal_date']<w['outcome_start']<=w['outcome_end']<=END
            and w['outcome_start'][:7]==w['outcome_end'][:7]==w['outcome_month'],'Unsafe original calendar clock')
    anchors=sorted({w['signal_date'] for w in windows}|{w['outcome_end'] for w in windows})
    need(len(anchors)==169,'Exact169 original signal/outcome endpoints required')
    per_native={native:{day for route,day in keys if route==native} for native in IDS}
    result=[]
    for case in cases:
        native=case['native_security_id'];dates=per_native[native]
        result.append({**case,'qualification':'METADATA_KEY_COVERAGE_ONLY_NOT_PROMOTED','date_key_rows':len(dates),
            'first_date_key':min(dates) if dates else None,'last_date_key':max(dates) if dates else None,
            'year_key_counts':dict(sorted(Counter(day[:4] for day in dates).items())),
            'anchor_dates':169,'matched_anchor_dates':sum(day in dates for day in anchors),
            'missing_anchor_dates':[day for day in anchors if day not in dates],
            'month_endpoints':[{'outcome_month':w['outcome_month'],'signal_date':w['signal_date'],
                'outcome_end':w['outcome_end'],'start_key_exists':w['signal_date'] in dates,
                'end_key_exists':w['outcome_end'] in dates} for w in windows],
            'all_original_cohorts_unchanged':True,'historical_identity_or_quote_validity_certified':False})
    return result

def run(config_path,output):
    began=time.monotonic();cfg=json.loads(Path(config_path).read_text());need(cfg['worker_sha256']==sha(__file__),'Frozen worker changed')
    plan_path=checked(cfg['plan_pin']);plan=json.loads(plan_path.read_text())
    protocol_path=checked(plan['protocol_pin']);protocol=json.loads(protocol_path.read_text());limits=protocol['budget']
    need(cfg['schema']=='dmv2-route-date-key-config/v1' and cfg['protocol_sha256']==sha(protocol_path)
        and protocol['status']=='FROZEN_METADATA_ONLY','Frozen metadata authority changed')
    need(plan['schema']=='dmv2-route-date-key-plan/v2' and plan['status']=='FROZEN_METADATA_ONLY_NOT_EXECUTED'
        and plan['columns']==COLUMNS and {c['native_security_id'] for c in plan['cases']}==IDS
        and plan['date_start']==START and plan['date_end']==END,'Frozen key scope changed')
    need(protocol['budget']['arrow_and_numeric_threads']==1 and protocol['budget']['automatic_retries']==0,'No threads/retries expansion')
    checked(protocol['parent_protocol_pin'])
    for item in protocol['sources']:checked({**item,'path':str(Path(plan['root'])/item['path'])})
    for field in ('source_inventory_reference_pin','source_windows_pin','stage1_cases_pin','stage1_receipt_pin'):checked(plan[field])
    stage1=json.loads(Path(plan['stage1_receipt_pin']['path']).read_text())
    need(stage1['status']=='PASS_METADATA_CANDIDATE_INVENTORY' and stage1['financial_columns_decoded']==[],'Original stage1 receipt not PASS metadata-only')
    expected_prior={'candidate_rows':stage1['candidate_rows_considered'],'data_files':stage1['source_data_files'],
        'footer_files':stage1['source_footers'],'metadata_rows_converted':stage1['metadata_rows_converted'],
        'projected_compressed_bytes':stage1['projected_compressed_bytes']}
    need(expected_prior==plan['stage1_usage'],'Cumulative prior stage usage changed')
    stage2={'candidate_rows':plan['candidate_rows'],'data_files':plan['data_files'],'footer_files':plan['footer_files'],
        'metadata_rows_converted':plan['returned_key_rows_upper_bound'],'projected_compressed_bytes':plan['projected_compressed_bytes']}
    cumulative=cumulative_usage(expected_prior,stage2,limits)
    def resources():
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
        need(rss<=limits['max_rss_bytes'] and time.monotonic()-began<=limits['max_wall_seconds'],'Metadata resource ceiling reached');return rss
    selected=[];computed={'candidate_rows':0,'projected_compressed_bytes':0};objects=plan['objects']
    need(len(objects)==plan['footer_files']==105 and len({o['path'] for o in objects})==105,'Exact105 source footer inventory required')
    # ALL safe physical footers and pruning decisions are verified before decoding any key column.
    for obj in objects:
        _,groups=inspect_footer(obj);resources()
        if any(g['included'] for g in groups):
            selected.append(obj)
            for group in groups:
                if group['included']:
                    computed['candidate_rows']+=group['rows'];computed['projected_compressed_bytes']+=group['projected_compressed_bytes']
    need([o['path'] for o in selected]==plan['selected_object_paths'] and len(selected)==plan['data_files']==3
        and computed['candidate_rows']==plan['candidate_rows'] and computed['projected_compressed_bytes']==plan['projected_compressed_bytes'],
        'Selected source files/rows/byte projection changed')
    for obj in selected:checked(obj);resources()
    output=Path(output).resolve();need(not output.exists(),'Preserve previous inventory attempt');output.mkdir(parents=True)
    write(output/'read_reservation.json',{'schema':'dmv2-route-date-key-reservation/v1','config_pin':pin(config_path),
        'worker_pin':pin(__file__),'plan_pin':pin(plan_path),'protocol_pin':pin(protocol_path),
        'stage1_usage':expected_prior,'stage2_upper':stage2,'cumulative_upper':cumulative,
        'financial_columns_decoded':[],'reservation_persisted_before_any_key_column_decode':True})
    pa.set_cpu_count(1);pa.set_io_thread_count(1);rows=[]
    for obj in selected:
        pf=pq.ParquetFile(obj['path'])
        for group in obj['groups']:
            if not group['included']:continue
            table=pf.read_row_group(group['index'],columns=COLUMNS,use_threads=False)
            table=filtered_keys(table,IDS,START,END)
            need(len(rows)+table.num_rows<=plan['returned_key_rows_upper_bound'],'Returned key upper bound exceeded; no truncation')
            rows.extend(table.to_pylist());resources()
    keys=validate_keys(rows,IDS,START,END)
    actual_usage=cumulative_usage(expected_prior,{**stage2,'metadata_rows_converted':len(keys)},limits)
    windows=json.loads(Path(plan['source_windows_pin']['path']).read_text());summary=anchor_summary(keys,windows,plan['cases'])
    # No decoded value replay: raw selected bytes + all physical footers reauthenticated.
    for obj in objects:inspect_footer(obj);resources()
    for obj in selected:checked(obj);resources()
    for field in ('protocol_pin','source_inventory_reference_pin','source_windows_pin','stage1_cases_pin','stage1_receipt_pin'):checked(plan[field])
    checked(cfg['plan_pin']);need(cfg['worker_sha256']==sha(__file__),'Worker changed during run')
    pq.write_table(pa.Table.from_pylist([dict(zip(COLUMNS,key)) for key in keys],schema=pa.schema([(c,pa.string()) for c in COLUMNS])),output/'date_keys.parquet')
    write(output/'cases.json',summary)
    receipt={'schema':'dmv2-route-date-key-result/v1','status':'PASS_METADATA_KEYS_ONLY','config_pin':pin(config_path),
        'worker_pin':pin(__file__),'plan_pin':pin(plan_path),'protocol_pin':pin(protocol_path),'date_keys_pin':pin(output/'date_keys.parquet'),
        'cases_pin':pin(output/'cases.json'),'stage1_usage':expected_prior,'stage2_usage':{**stage2,'metadata_rows_converted':len(keys)},
        'cumulative_usage':actual_usage,'source_footer_inventory_count':105,'exact_original_anchor_dates':169,
        'case_counts':[{k:c[k] for k in ('ticker_label','native_security_id','role','date_key_rows','first_date_key','last_date_key','matched_anchor_dates')} for c in summary],
        'financial_columns_decoded':[],'cloud_reads':False,'canonical_mapping_promoted':False,'original_cohorts_or_prior_outputs_changed':False,
        'models_loaded_fitted_or_predicted':False,'backtests_run':False,'source_byte_and_footer_pins_unchanged':True,
        'elapsed_seconds':time.monotonic()-began,'peak_rss_bytes':resources(),
        'limitations':['Date keys do not certify valid prices, historical identity, adjustment units, liquidity or economic returns.',
            'BK ordinary-share native remains an unpromoted candidate; coverage is not an admission rule.',
            'No OHLCV, return, feature, class or probability column or financial footer statistic was accessed.']}
    need(sum(p.stat().st_size for p in output.iterdir())+len(json.dumps(receipt))<=limits['max_output_bytes'],'Output budget exceeded')
    write(output/'receipt.json',receipt);print(json.dumps({'status':receipt['status'],'case_counts':receipt['case_counts'],'cumulative_usage':actual_usage}))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();run(args.config,args.output)
