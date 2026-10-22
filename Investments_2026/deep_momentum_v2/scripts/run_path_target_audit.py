"""Explicit two-month path engineering; planning reads no financial values."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import resource
import sys
import time
import pandas as pd
import pyarrow as pa
from dmv2.catalog import canonical_sha256
from dmv2.path_target_audit import (PROTOCOL, _pin, _json, _need,
    plan_path_target_audit, verify_path_audit_plan, load_path_audit_values,
    build_audit_paths, read_path_target_audit)
from dmv2.path_targets import build_path_targets


def write_json(path, value):
    Path(path).write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False,default=str)+'\n')


def resources(plan, started, output):
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss=int(rss if sys.platform=='darwin' else rss*1024)
    _need(time.monotonic()-started<=plan['limits']['wall_seconds'] and rss<=plan['limits']['peak_rss_bytes'],
          'Bounded path audit exceeded time/RSS ceiling')
    _need(sum(p.stat().st_size for p in Path(output).rglob('*') if p.is_file())<=plan['limits']['output_bytes'],
          'Bounded path audit exceeded output ceiling')
    return rss


def core_cohort(plan):
    # Old label/outcome fields are audit controls, never the new target family's
    # own columns. Arrow preserves nullable identity metadata under pandas3.
    reserved = {"outcome_month", "outcome_start", "outcome_end", "entry_date", "exit_date", "exit_mark",
        "label_available_at", "side", "path_variant", "target_family", "cash_return", "maximum_drawdown",
        "utility", "target_value", "target_class", "target_available", "path_available", "path_requested", "target_status"}
    rows = []
    for original in plan["original_cohort"]:
        rows.append({("legacy_" + name if name in reserved else name): value for name, value in original.items()})
    return pa.Table.from_pylist(rows)


def run(plan, output):
    output=Path(output).resolve()
    _need(not output.exists(),'Use a fresh output directory for an explicit path audit')
    verify_path_audit_plan(plan)
    started=time.monotonic();output.mkdir(parents=True)
    write_json(output/'reviewed_plan.json',plan)
    write_json(output/'STARTED.json',{'protocol':PROTOCOL,'plan_sha256':plan['plan_sha256']})
    try:
        resources(plan,started,output)
        cohort_table=core_cohort(plan)
        cohort=pd.DataFrame(plan['original_cohort'])
        prices,legacy=load_path_audit_values(plan)
        resources(plan,started,output)
        episodes,marks,details=build_audit_paths(plan,prices,resource_check=lambda:resources(plan,started,output))
        panel=build_path_targets(cohort_table,episodes,marks,windows=pd.DataFrame(plan['windows']),
            sessions=plan['sessions'],observed_through=max(w['exit_date'] for w in plan['windows']),
            risk_penalty=1.0,endpoint_policy='next_open')
        table=panel.table.to_pandas()
        control=table.loc[(table.side=='long')&(table.path_variant=='unstopped')&(table.target_family=='executable_return'),
            ['source_security_id','signal_date','prediction_eligible','path_available','cash_return','target_class']]
        comparison=control.merge(legacy[['source_security_id','signal_date','label_return','target_class']].rename(
            columns={'label_return':'legacy_label_return','target_class':'legacy_target_class'}),
            on=['source_security_id','signal_date'],how='left',validate='one_to_one')
        comparison['return_difference']=comparison.cash_return-comparison.legacy_label_return
        matched=comparison.loc[comparison.path_available & comparison.legacy_label_return.notna()]
        label_overlap=comparison.target_class.notna() & comparison.legacy_target_class.notna()
        month_by_signal={w['signal_date']:w['outcome_month'] for w in plan['windows']}
        episode_coverage=episodes.assign(outcome_month=episodes.signal_date.map(month_by_signal)).groupby(
            ['outcome_month','side','path_variant','status'],dropna=False).size().rename('rows').reset_index().to_dict('records')
        plan_pin=_pin(output/'reviewed_plan.json');plan_pin['path']='reviewed_plan.json'
        artifacts={'reviewed_plan.json':plan_pin}
        for name,frame in [('episodes',episodes),('marks',marks),('episode_details',details),
                ('targets',table),('diagnostics',panel.diagnostics.to_pandas()),('legacy_alignment',comparison)]:
            path=output/(name+'.parquet');frame.to_parquet(path,index=False)
            pin=_pin(path);pin['path']=path.name;artifacts[path.name]=pin
        write_json(output/'target_provenance.json',panel.provenance)
        pin=_pin(output/'target_provenance.json');pin['path']='target_provenance.json';artifacts[pin['path']]=pin
        verify_path_audit_plan(plan)
        peak=resources(plan,started,output)
        audit={'status':'COMPLETE_BOUNDED_AUDIT','protocol':PROTOCOL,'plan_sha256':plan['plan_sha256'],
            'audit_months':[w['outcome_month'] for w in plan['windows']],
            'original_rows':len(cohort),'prediction_eligible_rows':int(cohort.prediction_eligible.sum()),
            'requested_episode_count':int(episodes.requested.sum()),'complete_episode_count':int(episodes.status.eq('complete').sum()),
            'unavailable_episode_count':int(episodes.status.eq('unavailable').sum()),
            'not_requested_episode_count':int(episodes.status.eq('not_requested').sum()),'reason_counts':dict(Counter(episodes.reason)),
            'monthly_episode_coverage':episode_coverage,'monthly_target_diagnostics':panel.diagnostics.to_pylist(),
            'scheduled_availability_policy':'All labels wait for the scheduled next-month first-open exit, including early stops.',
            'legacy_alignment':{'both_available_rows':len(matched),'class_comparison_rows':int(label_overlap.sum()),
                'class_changed_rows':int((comparison.loc[label_overlap,'target_class']!=comparison.loc[label_overlap,'legacy_target_class']).sum()),
                'legacy_only_rows':int((comparison.legacy_label_return.notna()&~comparison.path_available).sum()),
                'executable_only_rows':int((comparison.path_available&comparison.legacy_label_return.isna()).sum()),
                'mean_absolute_return_difference':None if matched.empty else float(matched.return_difference.abs().mean()),
                'maximum_absolute_return_difference':None if matched.empty else float(matched.return_difference.abs().max())},
            'unit_cash_reference':plan['semantics'],'target_provenance_sha256':canonical_sha256(panel.provenance),
            'training_history_path_targets_ready':False,'financial_2024_plus_read':False,'models_fitted':False,
            'portfolio_backtest_run':False,'original_sources_unchanged':True,
            'read_plan':{**plan['read_plan'],'financial_values_read':True},
            'wall_seconds':time.monotonic()-started,'peak_rss_bytes':peak,'artifacts':artifacts}
        write_json(output/'audit.json',audit)
        return read_path_target_audit(output)
    except BaseException as error:
        write_json(output/'failure.json',{'exception':type(error).__name__,'message':str(error),
            'models_fitted':False,'portfolio_backtest_run':False})
        raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['plan','run','report'])
    parser.add_argument('--project',type=Path,default=Path.cwd())
    parser.add_argument('--archive',type=Path)
    parser.add_argument('--plan',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.action=='plan':
        plan=plan_path_target_audit(args.project,archive_directory=args.archive)
        _need(not args.output.exists(),'Preserve existing reviewed plan attempts')
        write_json(args.output,plan);print(json.dumps(plan['read_plan'],indent=2))
    elif args.action=='run':
        _need(args.plan is not None,'An explicit reviewed --plan JSON is required')
        print(json.dumps(run(_json(args.plan),args.output),indent=2))
    else: print(json.dumps(read_path_target_audit(args.output),indent=2))

if __name__=='__main__': main()
