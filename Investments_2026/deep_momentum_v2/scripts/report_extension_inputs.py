#!/usr/bin/env python3
"""Bounded independent aggregation of verified input coverage; zero model I/O."""
from pathlib import Path
import argparse
import math
import pandas as pd
import pyarrow.parquet as pq
from dmv2.extension_inputs import validate
from dmv2.paper15_baseline_data import pin,check,read,save


def main(plan_path):
    p=validate(plan_path);root=Path(p['output_directory']);ledger=read(root/'ledger.json')
    if ledger['status']!='COMPLETE_VERIFIED_INPUTS':raise ValueError('Verified input audit required')
    phase=next(x['receipt'] for x in ledger['completed'] if x['key']=='verify_data')
    phase=read(check(phase));proof_pin=phase['result'];proof=read(check(proof_pin))
    if proof['status']!='PASS_INPUT_SOURCE_ARRAYS_AND_MONTHLY_FORMULAS':raise ValueError('Independent proof')
    receipt=read(check(proof['prepared_receipt_pin']));a=receipt['artifacts'];report=root/'report'
    if report.exists():raise ValueError('Preserve prior report attempt')
    report.mkdir();specs=[];columns=['source_security_id','signal_date','outcome_month','label_available_at',
        'identity_eligible','history_complete','prediction_eligible','target_available','target_class',
        'identity_status','identity_eligibility_reason','outcome_invalid_daily_sessions']
    for n in (5,10):
        for kind,item in [('old',p['parent_inputs'][f'panel{n}']),('new',a[f'panel{n}'])]:
            pf=pq.ParquetFile(check(item));cols=[c for c in columns if c in pf.schema_arrow.names]
            cost=sum(pf.metadata.row_group(g).column(c).total_compressed_size for g in range(pf.metadata.num_row_groups)
                     for c in range(pf.metadata.row_group(g).num_columns) if pf.metadata.row_group(g).column(c).path_in_schema in cols)
            specs.append(dict(years=n,kind=kind,pin=item,columns=cols,compressed_projection_bytes=cost))
    total=sum(s['compressed_projection_bytes'] for s in specs)
    if total>64*1024**2:raise ValueError('Report projection exceeds64MiB')
    readplan=save(report/'read_plan.json',dict(status='SEALED_BEFORE_PROJECTED_REPORT_READ',source_proof_pin=proof_pin,
        prepared_receipt_pin=proof['prepared_receipt_pin'],worker_pin=pin(__file__),specs=specs,
        maximum_projection_bytes=64*1024**2,reserved_compressed_bytes=total,new_fits=0,new_backtests=0))
    frames={(s['years'],s['kind']):pq.read_table(check(s['pin']),columns=s['columns']).to_pandas() for s in specs}
    annual=[];summary=[];cohorts=[];coverage=[]
    for n in (5,10):
        new=frames[n,'new'];combined=pd.concat([frames[n,'old'],new],ignore_index=True)
        if combined.duplicated(['source_security_id','signal_date']).any():raise ValueError('Overlapping histories')
        for y in (2024,2025,2026):
            train=combined.loc[combined.outcome_month.between(f'{y-n}-01',f'{y-1}-12')&combined.label_available_at.le(f'{y-1}-12-31')&combined.prediction_eligible&combined.target_available]
            pred=new.loc[new.outcome_month.str.startswith(str(y))]
            annual.append(dict(history_years=n,year=y,train_start=f'{y-n}-01',train_end=f'{y-1}-12',training_rows=len(train),
                training_months=int(train.outcome_month.nunique()),training_classes=int(train.target_class.nunique()),
                prediction_months=int(pred.outcome_month.nunique()),prediction_eligible=int(pred.prediction_eligible.sum()),
                labelled=int(pred.target_available.sum()),missing_targets=int((pred.prediction_eligible&~pred.target_available).sum())))
        for signal,g in new.groupby('signal_date',sort=True):
            coverage.append(dict(history_years=n,signal_date=signal,outcome_month=g.outcome_month.iloc[0],original_members=len(g),
                identity_admitted=int(g.identity_eligible.sum()),complete_history=int((g.identity_eligible&g.history_complete).sum()),
                prediction_eligible=int(g.prediction_eligible.sum()),labelled=int(g.target_available.sum()),
                missing_targets=int((g.prediction_eligible&~g.target_available).sum()),
                eligible_with_incomplete_outcome_daily=int((g.prediction_eligible&g.outcome_invalid_daily_sessions.gt(0)).sum())))
        summary.append(dict(history_years=n,original_stock_months=len(new),identity_admitted=int(new.identity_eligible.sum()),
            prediction_eligible=int(new.prediction_eligible.sum()),eligible_pct=100*float(new.prediction_eligible.mean()),
            observed_labels=int(new.target_available.sum()),missing_labels=int((new.prediction_eligible&~new.target_available).sum()),
            observed_label_pct_of_eligible=100*float(new.target_available.sum()/new.prediction_eligible.sum()),
            complete_RET_vectors=next(x['complete_RET_vectors'] for x in proof['summary'] if x['history_years']==n)))
    actual=pd.DataFrame(annual);saved=pq.read_table(check(a['annual_readiness'])).to_pandas()
    pd.testing.assert_frame_equal(actual,saved,check_dtype=False)
    for n in (5,10):
        saved=pq.read_table(check(a[f'coverage{n}'])).to_pandas()
        expected=pd.DataFrame([r for r in coverage if r['history_years']==n])
        pd.testing.assert_frame_equal(expected,saved,check_dtype=False)
    new=frames[5,'new']
    blocked=new.loc[~new.identity_eligible].groupby(['identity_status','identity_eligibility_reason']).size().reset_index(name='member_months')
    pd.testing.assert_frame_equal(blocked,pq.read_table(check(a['identity_blockers'])).to_pandas(),check_dtype=False)
    yearly=[]
    for n in (5,10):
        f=frames[n,'new']
        for year,g in f.groupby(f.outcome_month.str[:4],sort=True):
            yearly.append(dict(history_years=n,year=int(year),months=int(g.outcome_month.nunique()),original_stock_months=len(g),
                prediction_eligible=int(g.prediction_eligible.sum()),eligible_pct=100*float(g.prediction_eligible.mean()),
                observed_labels=int(g.target_available.sum()),missing_labels=int((g.prediction_eligible&~g.target_available).sum())))
    # Small final JSON is the passive notebook's entire numerical data source.
    result=save(report/'summary.json',dict(status='VERIFIED_INPUT_COVERAGE_ONLY',first_added_outcome='2024-01',last_added_outcome='2026-04',added_months=28,
        common_candidate_months=75,monthly_only_candidate_months=76,summary=summary,annual_training=annual,yearly_coverage=yearly,
        identity_blockers=blocked.to_dict('records'),new_fits=0,new_backtests=0,
        open_dependencies=['BK/XOM reviewed route continuation beyond2023; original disputed routes not reused',
          'Forward endpoint gaps and corporate-action accounting must be handled without filtering predictions',
          '12 new fixed annual model fits, independent inference and continuous75month daily/monthly replay remain unrun'],
        source_identity_and_knowledge_time_certified=False,execution_action_coverage_certified=False))
    proofout=save(report/'verification.json',dict(status='PASS_AGGREGATE_RECONCILIATION',source_proof_pin=proof_pin,
        read_plan_pin=readplan,summary_pin=result,worker_pin=pin(__file__),new_fits=0,new_backtests=0))
    print(str(proofout['path']))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);main(ap.parse_args().plan)
