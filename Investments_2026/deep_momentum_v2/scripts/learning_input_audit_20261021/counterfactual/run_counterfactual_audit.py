"""Explicit frozen-plan conditional audit. Never fits, infers or backtests."""
from __future__ import annotations
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_name]='1'
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from learning_repair_counterfactual import (MONTH_FIELDS, TARGET_FIELDS, annual_dependency_table,
    day, need, propagate_ret_payoff_impact, rerank_month_after_endpoint_repair,
    verify_preserved_ret_states)

pa.set_cpu_count(1);pa.set_io_thread_count(1)


def load_json(path):return json.loads(Path(path).read_text())


def file_hash(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def check_pin(pin):
    path=Path(pin['path'])
    need(path.is_file() and path.stat().st_size==pin['size_bytes'] and file_hash(path)==pin['sha256'],
        f'Authenticated input bytepin changed: {path}')


def target_fingerprint(table):
    canonical=pa.Table.from_pylist(table.to_pylist(),schema=table.schema.remove_metadata())
    sink=pa.BufferOutputStream()
    with pa.ipc.new_stream(sink,canonical.schema) as writer:writer.write_table(canonical)
    return 'sha256:'+hashlib.sha256(sink.getvalue()).hexdigest()


def select(table,filters):
    mask=pa.array([True]*table.num_rows)
    for name,value in filters.items():
        if name=='outcome_month_range':
            mask=pc.and_(mask,pc.and_(pc.greater_equal(table['outcome_month'],value[0]),
                pc.less_equal(table['outcome_month'],value[1])))
        else:mask=pc.and_(mask,pc.equal(table[name],value))
    return table.filter(mask)


def run(plan_path,output):
    began=time.monotonic();plan=load_json(plan_path);output=Path(output)
    need(not output.exists(),'Preserve prior results; output must be new')
    need(plan['schema']=='dmv2-learning-conditional-numeric-plan/v1'
        and plan['numeric_threads']==1 and plan['automatic_retries']==0,'Unexpected numerical audit contract')
    need(plan['original_source_security_id']=='siblis_ticker_company_8500ae2d7e99a52f8713'
        and plan['original_native']=='sec_zMM1vd' and plan['outcome_month']=='2021-11'
        and plan['signal_date']=='2021-10-29' and plan['outcome_start']=='2021-11-01'
        and plan['outcome_end']=='2021-11-30'
        and [t['name'] for t in plan['tables']]==['targets','class_means','start_quote','end_quote'],
        'Only the frozen conditional November2021 RCM case is authorized')
    check_pin(plan['master_protocol_pin']);protocol=load_json(plan['master_protocol_pin']['path'])
    need(protocol['status']=='FROZEN_AUDIT_ONLY'
        and protocol['budget']['conditional_numeric_max_projected_bytes']==plan['max_projected_bytes']==268435456
        and plan['max_rss_bytes']==4294967296 and plan['max_wall_seconds']==600,'Master audit budget changed')
    pins=[*plan['source_pins'],*plan['code_pins'],*[t['pin'] for t in plan['tables']],
        *[p for state in plan['old_vectors'] for p in state['pins']]]
    for pin in pins:check_pin(pin)
    root=Path(plan['project_root']);old=root/'outputs/ret_backtest/2020-2023-20261020-v1'
    need(load_json(old/'verification/ret_backtest.json')['status']=='PASS'
        and load_json(old/'preflight/audit.json')['status']=='COMPLETE','Original RET independent PASS required')
    pre=load_json(old/'preflight/audit.json');oldroot=load_json(old/'build_identity.json')
    annual=next(row for row in oldroot['annual_sources'] if row['year']==2021)
    def auth_relative(pin,base=root):check_pin({**pin,'path':str(base/pin['path'])})
    auth_relative(annual['targets_pin']);auth_relative(annual['target_state_pin']);auth_relative(annual['audit_pin'])
    annual_audit=load_json(root/annual['audit_pin']['path'])
    inherited={
        'targets':(root/annual['targets_pin']['path'],annual['targets_pin']),
        'class_means':(old/'preflight/source_month_class_means.parquet',pre['artifacts']['source_month_class_means.parquet']),
        'start_quote':(root/annual['directory']/'daily_prices_raw.parquet',annual_audit['artifacts']['daily_prices_raw']),
    }
    manifest=load_json(root/'data/derived/ret_repairs/20261020-v1/repair_manifest.json')
    inherited['end_quote']=(root/manifest['quotes_pin']['path'],manifest['quotes_pin'])
    for entry in plan['tables']:
        path,saved=inherited[entry['name']]
        need(Path(entry['pin']['path'])==path
            and (entry['pin']['sha256'],entry['pin']['size_bytes'])==(saved['sha256'],saved['size_bytes']),
            'Numeric input is not the original pinned target/mean/quote artifact')
    for state in plan['old_vectors']:
        for pin in state['pins']:
            relative=str(Path(pin['path']).relative_to(old/'preflight'))
            saved=pre['artifacts'][relative]
            need((saved['sha256'],saved['size_bytes'])==(pin['sha256'],pin['size_bytes']),
                'Saved baseline vector not authenticated by original preflight')
    tables={};reserved=0;metadata={}
    def resource_guard():
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform!='darwin':rss*=1024
        need(rss<=plan['max_rss_bytes'] and time.monotonic()-began<=plan['max_wall_seconds'],
            'Conditional audit resource ceiling reached')
        return rss
    def reserve(entry,columns,label):
        nonlocal reserved
        footer=pq.ParquetFile(entry['pin']['path']);names=footer.schema_arrow.names
        cost=sum(footer.metadata.row_group(g).column(names.index(c)).total_compressed_size
            for g in range(footer.metadata.num_row_groups) for c in columns)
        need(cost==entry[label] and reserved+cost<=plan['max_projected_bytes'],
            'Projected financial read not reserved within frozen budget')
        reserved+=cost;resource_guard()
    # All date metadata for ALL four files is checked before any financial read.
    for entry in plan['tables']:
        reserve(entry,entry['metadata_columns'],'metadata_projected_bytes')
        meta=pq.read_table(entry['pin']['path'],columns=entry['metadata_columns'],use_threads=False)
        need(meta.num_rows==entry['rows'],'Frozen source row count changed')
        names=meta.column_names
        for field in ('signal_date','outcome_start','outcome_end','trading_date'):
            if field in names:
                for value in meta[field].to_pylist():day(value)
        if 'outcome_month' in names:
            months=meta['outcome_month'].to_pylist()
            need(all(isinstance(m,str) and '2010-01'<=m<='2023-12' for m in months),'Protected or malformed source month')
        if entry['name'] in ('start_quote','end_quote'):
            selected=select(meta,entry['filter']).to_pylist()
            need(len(selected)==1,'Actual unique endpoint metadata required')
            row=selected[0]
            need(row['frequency']=='daily' and row['intraperiod'] is False
                and row['price_quality_status']=='accepted'
                and row['adjusted_ohlc_bounds_valid'] is True
                and row['raw_ohlc_bounds_valid'] is True,'Accepted paired endpoint metadata required')
        metadata[entry['name']]=meta
    for entry in plan['tables']:
        reserve(entry,entry['numeric_columns'],'numeric_projected_bytes')
        # Whole selected-column blocks may transiently decode within Arrow.
        # No excluded numeric row is converted to Python or analyzed.
        raw=pq.read_table(entry['pin']['path'],columns=entry['numeric_columns'],use_threads=False)
        tables[entry['name']]=select(raw,entry['filter']);resource_guard()
    need(reserved==plan['projected_compressed_bytes'],'Reserved read total differs from frozen plan')
    december_metadata=select(metadata['targets'],{'source_security_id':plan['original_source_security_id'],
        'outcome_month':'2021-12'}).to_pylist()
    need(len(december_metadata)<=1,'Original December metadata contains duplicate RCM keys')
    selected=tables['targets'];state=load_json(root/annual['target_state_pin']['path'])
    need(target_fingerprint(selected)==state['provenance']['month_target_fingerprints']['2021-11'],
        'Original November target Arrow fingerprint mismatch')
    target=selected.to_pandas();means=tables['class_means'].to_pandas()
    source_key=plan['original_source_security_id'];row=target[target.source_security_id.eq(source_key)]
    need(len(row)==1 and row.iloc[0].signal_date==plan['signal_date']
        and bool(row.iloc[0].prediction_eligible) and not bool(row.iloc[0].return_available)
        and pd.isna(row.iloc[0].label_return),'Frozen original RCM missing endpoint case changed')
    states={entry['signal_date']:load_json(entry['pins'][0]['path']) for entry in plan['old_vectors']}
    # Exact saved vector fingerprints must reproduce before constructing shadow.
    fingerprints=verify_preserved_ret_states(means,states)
    start=tables['start_quote']['adj_close'][0].as_py();end=tables['end_quote']['adj_close'][0].as_py()
    change={'scope':'conditional_shadow_only','same_security_price_only':True,
        'uses_cash_or_successor_equity':False,'source_security_id':source_key,
        'signal_date':plan['signal_date'],'start_date':plan['signal_date'],
        'end_date':plan['outcome_end'],'repaired_date':plan['outcome_end'],
        'expected_old_label_return':None,'start_adj_close':start,'end_adj_close':end}
    result=rerank_month_after_endpoint_repair(target,change,observed_through='2023-12-31')
    vectors=propagate_ret_payoff_impact(means,[result],pd.DataFrame({'signal_date':sorted(states)}),
        baseline_vector_fingerprints=fingerprints)
    dependencies=annual_dependency_table(['2021-11'],load_json(root/'configs/rf_expanded_tuning_20261020.json'))
    # Reauthenticate all preserved inputs before output, including old vector bytes.
    for pin in pins:check_pin(pin)
    peak=resource_guard();output.mkdir()
    frames={'original_month':result.baseline,'conditional_shadow_month':result.shadow,
        'changed_rows':result.changes,'monthly_class_means_before':result.monthly_before,
        'monthly_class_means_after':result.monthly_after,'causal_ret_vector_deltas':vectors,
        'training_and_cv_date_dependencies':dependencies}
    for name,frame in frames.items():
        pq.write_table(pa.Table.from_pandas(frame,preserve_index=False),output/(name+'.parquet'))
    summary={**result.summary,'schema':'dmv2-learning-conditional-shadow-result/v1',
        'master_protocol_sha256':plan['master_protocol_pin']['sha256'],'numeric_read_plan_sha256':file_hash(plan_path),
        'ret_vectors':48,'baseline_logical_vector_fingerprints_reproduced':48,
        'baseline_vector_bytepins_unchanged_before_after':True,
        'first_affected_signal_date':vectors.loc[vectors.patched_month_in_window,'signal_date'].min(),
        'first_affected_decision_outcome_month':'2021-12','december2021_label_rebuilt':False,
        'december_anchor_dependency_receipt':{'original_safe_target_metadata':december_metadata,
            'November30_anchor_restored_from_observed_repair':True,
            'December31_endpoint_evidence_requested_or_read':False,
            'December_label_financial_values_converted_to_Python':False,
            'new_December_return_inferred':False},
        'projected_compressed_bytes':reserved,'peak_rss_bytes':peak,'execution_seconds':time.monotonic()-began,
        'financial_2024_plus_decoded':False,'production_inputs_or_prior_runs_modified':False,
        'limitations':['Conditional RCM price-only shadow; no canonical route promotion.',
            'Other security class boundaries may change under the unchanged observed-return rank policy.',
            'December anchor repair alone does not establish December31 endpoint or a December return.',
            'No features, preprocessing, classifiers, probabilities or backtests rebuilt.',
            'Later Paper15 feature/admission and cross-sectional normalization dependencies are outside this fixed-cohort conditional experiment.',
            'Cash/stock/ADS economic labels remain outside this audit.']}
    (output/'summary.json').write_text(json.dumps(summary,sort_keys=True,indent=2,allow_nan=False)+'\n')
    need(sum(p.stat().st_size for p in output.iterdir())<=protocol['budget']['new_output_max_bytes'],
        'Output budget exceeded')
    print(json.dumps({'status':'CONDITIONAL_SHADOW_COMPLETE','output':str(output),
        'class_rows_changed':summary['class_rows_changed'],'models_fitted':0}))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();run(args.plan,args.output)
