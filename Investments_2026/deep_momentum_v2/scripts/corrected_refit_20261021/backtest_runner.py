"""Sealed phase runner for new forecasts: selection, preflight, then120 replays.

No fitting or inference APIs are imported. Real phase execution requires all20
independent model proofs through the source plan, exact shared read reservation,
and the existing external process-tree supervisor. Planning inspects footers only.
"""
from pathlib import Path
import argparse
import json
import os
import sys
from control import ROOT,check,pin,read,save_new,require_reservation,need
import backtest_integration as bi

PHASES=('selection','preflight','replay')
BASE=ROOT/'run/backtest'
NUMERIC_ENV=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')
MAX_BACKTEST_OUTPUT=1024**3


def uniq_pins(items):
    out={}
    for item in items:
        clean=bi.exact_pin(item);out[clean['path']]=clean
    return list(out.values())


def source_pins(source):
    pins=[source[n] for n in ('control_pin','catalog_pin','final_model_verification_pin','base_configuration_pin',
        'execution_phase_pin','execution_manifest_pin','execution_verification_pin','execution_backtest_audit_pin',
        'scope_pin','planning_JSON_plan_pin')]
    for f in source['forecasts']:
        pins += [f['predictions_pin'],f['prediction_receipt_pin']]
    for d in source['daily']:pins += [d['metadata']['pin']]
    for p in source['payoffs']:pins += [p['state_pin'],p['values']['pin'],p['monthly']['pin']]
    pins += [source['execution_repairs']['values']['pin'],*source['worker_pins'],
             source['model_protocol_pin'],source['input_verification_pin'],source['terminal_guard_pin'],
             *source['model_verification_pins'].values(),*source['execution_evidence_pins']]
    return uniq_pins(pins)


def projection_all(item):
    import pyarrow.parquet as pq
    pf=pq.ParquetFile(check(item))
    return bi.footer_projection(item,pf.schema_arrow.names,150000)


def saved_selection_plans(audit):
    entries={};cost=0
    for arm in bi.ARMS:
        for basket in bi.BASKETS:
            values={}
            for name in ('selected','rebalances','monthly'):
                item=audit['artifacts'][f'{arm}/{basket}/{name}.parquet']
                full=projection_all(item)
                datecols=[n for n in ('signal_date','execution_date','source_security_id','intrinio_security_id','side') if n in full['columns']]
                meta=bi.footer_projection(item,datecols,150000)
                values[name]={'metadata':meta,'values':full}
                cost+=meta['projected_compressed_bytes']+full['projected_compressed_bytes']
            entries[arm+'/'+basket]=values
    return entries,cost


def make_plan(source_path,protocol_path,phase,output):
    need(phase in PHASES,'Unknown backtest phase')
    source_pin=pin(source_path);source=read(source_path);control=read(check(source['control_pin']))
    protocol_pin=pin(protocol_path);protocol=read(protocol_path)
    need(source['schema']=='dmv2-corrected-forecasts-backtest-source-plan/v1'
         and source['scenario_count']==120 and protocol['schema']=='dmv2-corrected-forecasts-backtest-protocol/v1'
         and protocol['status']=='SEALED_BEFORE_FINANCIAL_READ'
         and protocol['source_plan_pin']==source_pin and protocol['allow_withheld_paths'] is True
         and protocol['scenarios']==120 and protocol['financial_2024_plus_authorized'] is False
         and protocol['runtime']==source['runtime'],
         'Explicit120-path no-new-model backtest protocol required')
    need(protocol['maximum_backtest_output_bytes']<=MAX_BACKTEST_OUTPUT
         and protocol['limits']['rss_stop_bytes']<=4*1024**3
         and protocol['limits']['rss_warning_bytes']<=3*1024**3
         and protocol['limits']['active_seconds_by_category']['backtest']<=7200,'Frozen backtest ceilings changed')
    pins=source_pins(source)
    for item in pins:check(item)
    common_json=uniq_pins([source['control_pin'],source['base_configuration_pin']] if phase=='selection' else
                          [source['control_pin'],source['execution_phase_pin'],source['execution_manifest_pin']])
    cost=sum(x['size_bytes'] for x in common_json)
    body={'schema':'dmv2-corrected-backtest-phase-plan/v1','status':'SEALED_BEFORE_FINANCIAL_READ',
        'phase':phase,'output_directory':str(BASE/phase),'protocol_pin':protocol_pin,'source_plan_pin':source_pin,
        'control_pin':source['control_pin'],'JSON_pins':common_json,'source_pins':pins,
        'execution_authority':source['execution_authority'],
        'integration_pin':pin(bi.__file__),'worker_pin':pin(__file__),'control_worker_pin':pin(ROOT/'control.py'),
        'code_pins':[pin(__file__),pin(bi.__file__),pin(ROOT/'control.py'),*source['worker_pins']],
        'new_estimator_fits':0,'new_inference_calls':0,'backtest_scenarios':120,
        'financial_2024_plus_authorized':False,'allow_withheld_paths':True}
    if phase=='selection':
        cost+=source['source_read_projections']['selection']
    else:
        selection_pin=pin(BASE/'selection/audit.json');selection=read(selection_pin['path'])
        need(selection['status']=='COMPLETE' and selection['source_plan_pin']==source_pin
             and selection['selection_sets']==15,'Exact completed fresh selection phase required')
        body['selection_audit_pin']=selection_pin
        body['selection_read_plan'],selected_cost=saved_selection_plans(selection)
        cost+=selected_cost
        cost+=source['source_read_projections']['quote_preflight' if phase=='preflight' else 'quote_replay']
        if phase=='replay':
            pre_pin=pin(BASE/'preflight/audit.json');pre=read(pre_pin['path'])
            need(pre['status']=='COMPLETE' and pre['source_plan_pin']==source_pin
                 and pre['selection_audit_pin']==selection_pin,'Exact fresh quote preflight required')
            body['preflight_audit_pin']=pre_pin
    body['projected_read_bytes']=cost
    return save_new(output,body)


def require_guard(plan_pin,plan):
    path=os.environ.get('DMV2_GUARD_LAUNCH_PATH');sha=os.environ.get('DMV2_GUARD_LAUNCH_SHA256')
    need(path and sha and pin(path)['sha256']==sha,'External backtest resource supervisor required')
    launch=read(path)
    need(launch['schema']=='dmv2-corrected-refit-phase-launch/v1'
         and launch['protocol_pin']==plan['protocol_pin'] and launch['action_plan_pin']==plan_pin
         and launch['time_category']=='backtest','Backtest supervisor binds another phase')
    command=launch['command']
    need(Path(command[0]).resolve()==Path(sys.executable).resolve()
         and Path(command[1]).resolve()==Path(__file__).resolve() and command[2:]==sys.argv[1:],
         'Backtest command differs from sealed launch')
    protocol=read(check(plan['protocol_pin']))
    need(launch['maximum_seconds']<=protocol['phase_maximum_seconds'][plan['phase']],'Backtest per-phase wall limit changed')
    for item in plan['code_pins']:
        need(item in launch['code_pins'],'Backtest integration/runner/control not supervisor-pinned')
    need(all(os.environ.get(n)=='1' for n in NUMERIC_ENV),'Single numeric thread environment required')


def read_projection(item):
    import pyarrow.parquet as pq
    check(item['pin'])
    need(bi.footer_projection(item['pin'],item['columns'],item['rows'])==item,'Exact backtest footer projection changed')
    table=pq.ParquetFile(item['pin']['path']).read(columns=item['columns'],use_threads=False)
    need(table.num_rows==item['rows'],'Backtest projection row count changed')
    return table


def save_table(directory,relative,frame,artifacts):
    import pyarrow as pa
    import pyarrow.parquet as pq
    target=Path(directory)/relative;target.parent.mkdir(parents=True,exist_ok=True)
    table=frame if isinstance(frame,pa.Table) else pa.Table.from_pandas(frame,preserve_index=False)
    with target.open('xb') as stream:pq.write_table(table,stream,compression='zstd')
    artifacts[relative]={**pin(target),'rows':table.num_rows,'columns':table.column_names}
    return artifacts[relative]


def begin(plan_path,label):
    pp=pin(plan_path);plan=read(plan_path)
    need(plan['schema']=='dmv2-corrected-backtest-phase-plan/v1'
         and plan['status']=='SEALED_BEFORE_FINANCIAL_READ' and plan['phase'] in PHASES
         and plan['worker_pin']==pin(__file__) and plan['integration_pin']==pin(bi.__file__),
         'Backtest phase code/identity changed')
    require_reservation(ROOT,label,plan['projected_read_bytes'],pp)
    require_guard(pp,plan)
    for item in (*plan['source_pins'],plan['protocol_pin'],plan['source_plan_pin'],plan['control_worker_pin']):check(item)
    protocol=read(plan['protocol_pin']['path'])
    from importlib.metadata import version
    for package,expected in protocol['runtime'].items():need(version(package)==expected,'Frozen backtest runtime changed: '+package)
    directory=Path(plan['output_directory']);need(directory==BASE/plan['phase'] and not directory.exists(),'Preserve prior backtest phase attempt')
    directory.mkdir(parents=True)
    started=save_new(directory/'STARTED.json',{'schema':'dmv2-corrected-backtest-phase-attempt/v1',
        'status':'ATTEMPT_CONSUMED','phase_plan_pin':pp,'reservation_label':label})
    source=read(plan['source_plan_pin']['path']);control=read(source['control_pin']['path'])
    return pp,plan,source,control,directory,started


def finish(plan_pin,plan,directory,started,artifacts,extra):
    for item in (*plan['source_pins'],plan['protocol_pin'],plan['source_plan_pin'],plan['worker_pin'],plan['integration_pin'],plan['control_worker_pin']):check(item)
    size=sum(p.stat().st_size for p in BASE.rglob('*') if p.is_file())
    need(size<=MAX_BACKTEST_OUTPUT,'Separate1GiB backtest output ceiling exceeded')
    return save_new(directory/'audit.json',{'schema':'dmv2-corrected-backtest-'+plan['phase']+'-audit/v1',
        'status':'COMPLETE','phase_plan_pin':plan_pin,'source_plan_pin':plan['source_plan_pin'],
        'protocol_pin':plan['protocol_pin'],'started_pin':started,'artifacts':artifacts,
        'reserved_projected_read_bytes':plan['projected_read_bytes'],'new_estimator_fits':0,'new_inference_calls':0,
        'original_source_pins_unchanged':True,'financial_2024_plus_values_converted':False,
        'execution_authority':plan['execution_authority'],
        'integrity_hash_bytes_are_separate_from_projected_column_budget':True,**extra})


def run_selection(args):
    pp,plan,source,control,directory,started=begin(args.plan,args.reservation)
    import pandas as pd
    import pyarrow as pa
    worker=bi._workers(control['project_root']);_,windows,_=worker['walkforward_clock']()
    windows_rows=windows.to_dict('records');metadata={};common={}
    # ALL20 original forecast metadata guards precede any probability conversion.
    for item in source['forecasts']:
        rows=read_projection(item['metadata']).to_pylist()
        cohort=bi.forecasts_metadata(rows,item['year'],windows_rows)
        if item['year'] in common:need(common[item['year']]==cohort,'New arms changed original forecast cohort/order')
        else:common[item['year']]=cohort
        metadata[(item['arm'],item['year'])]=rows
    payoff_meta={};states={}
    for item in source['payoffs']:
        state=read(check(item['state_pin']));rows=read_projection(item['monthly_metadata']).to_pylist()
        bi.validate_ret_metadata(rows,state,item['signal_date'])
        states[item['signal_date']]=state;payoff_meta[item['signal_date']]=rows
    expected_signals=windows.signal_date.tolist()
    need(sorted(states)==expected_signals,'Exactly48canonical corrected RET states required')
    predictions={arm:[] for arm in bi.ARMS};payoffs={}
    for item in source['forecasts']:
        values=read_projection(item['probabilities']);rows=metadata[(item['arm'],item['year'])]
        need(values.select(bi.KEYS).to_pylist()==[{k:r[k] for k in bi.KEYS} for r in rows],'Forecast probability row order differs from metadata')
        frame=pd.DataFrame(rows)
        for column in bi.PROBS:frame[column]=values[column].to_numpy(zero_copy_only=False)
        predictions[item['arm']].append((item['year'],frame))
    predictions={arm:pd.concat([frame for _,frame in sorted(parts,key=lambda z:z[0])],ignore_index=True) for arm,parts in predictions.items()}
    for item in source['payoffs']:
        monthly=read_projection(item['monthly']);values=read_projection(item['values']);signal=item['signal_date']
        need(monthly.select([n for n in monthly.column_names if n!='class_mean_return']).to_pylist()==payoff_meta[signal],'RET source metadata changed between passes')
        payoffs[signal]=bi.restore_payoff_in_memory(values,monthly,states[signal],signal=signal)
    selections,validation=bi.select_in_memory(predictions,payoffs,project=control['project_root'],configuration=read(source['base_configuration_pin']['path']))
    artifacts={}
    for (arm,basket),selected in selections.items():
        for attr,name in bi.FRAME_NAMES.items():save_table(directory,f'{arm}/{basket}/{name}.parquet',getattr(selected,attr),artifacts)
    save_new(directory/'validation.json',{'status':'PASS_PRODUCTION_RECONSTRUCTION','results':validation})
    result=finish(pp,plan,directory,started,artifacts,{'selection_sets':15,'forecast_arms':5,'outcome_months':48,
        'all_original_forecast_rows_and_admissions_preserved':True,'old_selection_reuse':False,
        'selection_used_quote_or_future_target_availability':False,'RET_states_from_corrected_catalog':48,
        'selection_validation_pin':pin(directory/'validation.json')})
    print(json.dumps({'phase':'selection','status':'COMPLETE','selection_sets':15,'audit_pin':result}))


def load_saved_selections(plan,windows):
    import pandas as pd
    from dmv2.walkforward_selection import BasketSelection
    expected=dict(zip(windows.signal_date,windows.execution_date));output={}
    for key,entries in plan['selection_read_plan'].items():
        frames={}
        for name,entry in entries.items():
            meta=read_projection(entry['metadata']).to_pylist()
            for r in meta:
                need(r['signal_date'] in expected and r.get('execution_date',expected[r['signal_date']])==expected[r['signal_date']],
                     'Saved selection timing differs before scores/weights')
            values=read_projection(entry['values'])
            need(values.select(entry['metadata']['columns']).to_pylist()==meta,'Saved selection metadata row order changed')
            frames[name]=values.to_pandas()
        arm,basket=key.split('/');output[(arm,basket)]=BasketSelection(pd.DataFrame(),frames['selected'],frames['rebalances'],frames['monthly'])
    need(set(output)==set(__import__('itertools').product(bi.ARMS,bi.BASKETS)),'Saved selection grid changed')
    return output


def load_quote_metadata(source,sessions,manifest):
    import pandas as pd
    from dmv2.ret_repair_backtest import overlay_quotes
    pieces=[];positions={};originals={}
    for item in source['daily']:
        table=read_projection(item['metadata']);rows=table.select(['intrinio_security_id','trading_date']).to_pylist()
        selected,excluded=bi.scoped_quote_positions(rows,item['year'],sessions)
        frame=table.take(__import__('pyarrow').array(selected,type=__import__('pyarrow').int64())).to_pandas()
        pieces.append(frame);positions[item['year']]=selected;originals[item['year']]=frame
    original=pd.concat(pieces,ignore_index=True)
    need(not original.duplicated(['intrinio_security_id','trading_date']).any(),'Cross-annual quote key collision')
    repairs=read_projection(source['execution_repairs']['metadata']).to_pandas()
    need(repairs.trading_date.isin(sessions).all(),'Old execution quote repair leaves approved sessions')
    amended=overlay_quotes(original,repairs,manifest['quote_amendments'],metadata_only=True)
    return original,amended,positions,originals,repairs


def run_preflight(args):
    pp,plan,source,control,directory,started=begin(args.plan,args.reservation)
    worker=bi._workers(control['project_root']);sessions,windows,_=worker['walkforward_clock']()
    selections=load_saved_selections(plan,windows)
    phase=read(source['execution_phase_pin']['path']);manifest=read(source['execution_manifest_pin']['path'])
    _,metadata,_,_,_=load_quote_metadata(source,sessions,manifest)
    coverage=bi.preflight_in_memory(selections,metadata,project=control['project_root'],execution_phase=phase)
    artifacts={};summaries=[]
    for (arm,basket),result in coverage.items():
        monthly=selections[(arm,basket)].monthly
        summaries.append({'arm':arm,'basket':basket,**result['summary'],
                          'withheld_selection_months':int(monthly.status.ne('available').sum())})
        for name,frame in result.items():
            if name!='summary':save_table(directory,f'coverage/{arm}/{basket}/{name}.parquet',frame,artifacts)
    import pandas as pd
    save_table(directory,'summary.parquet',pd.DataFrame(summaries),artifacts)
    result=finish(pp,plan,directory,started,artifacts,{'selection_audit_pin':plan['selection_audit_pin'],
        'selection_sets':15,'summary':summaries,'performance_computed':False,
        'selection_changed_using_quote_coverage':False,'all_maximum_unstopped_horizons_checked':True,
        'event_count':18,'historical_retirements':11,'PENN_exception_unchanged':True,'execution_quote_repairs':40})
    print(json.dumps({'phase':'preflight','status':'COMPLETE','selection_sets':15,'audit_pin':result}))


def run_replay(args):
    pp,plan,source,control,directory,started=begin(args.plan,args.reservation)
    worker=bi._workers(control['project_root']);sessions,windows,_=worker['walkforward_clock']()
    selections=load_saved_selections(plan,windows)
    phase=read(source['execution_phase_pin']['path']);manifest=read(source['execution_manifest_pin']['path'])
    original_meta,_,positions,annual_meta,repair_metadata=load_quote_metadata(source,sessions,manifest)
    import pandas as pd
    import pyarrow as pa
    from dmv2.ret_repair_backtest import overlay_quotes
    pieces=[]
    # All quote ID/date/quality metadata is guarded before numeric prices enter Python.
    for item in source['daily']:
        table=read_projection(item['values']).take(pa.array(positions[item['year']],type=pa.int64()))
        need(table.select(['trading_date','intrinio_security_id']).to_pylist()==annual_meta[item['year']][['trading_date','intrinio_security_id']].to_dict('records'),
             'Quote value keys differ from admitted metadata')
        pieces.append(table.to_pandas())
    original=pd.concat(pieces,ignore_index=True)
    repairs=read_projection(source['execution_repairs']['values'])
    need(repairs.select(['trading_date','intrinio_security_id']).to_pylist()==[
        {'trading_date':r['trading_date'],'intrinio_security_id':r['intrinio_security_id']}
        for r in repair_metadata.to_dict('records')], 'Old repair value keys differ from metadata')
    prices=overlay_quotes(original,repairs.to_pandas(),manifest['quote_amendments'])
    handovers=phase['identity_handover_registry']['handovers']
    artifacts={};comparison=[];yearly=[];coverage=[]
    for (arm,basket),selected in selections.items():coverage.append(selected.monthly.assign(arm=arm,basket=basket))
    for case,value,row,monthly,annual in bi.replay_in_memory(selections,prices,project=control['project_root'],execution_phase=phase,identity_handovers=handovers):
        arm,basket,variant,cost=case;relative=f'arms/{arm}/{basket}/{variant}/cost_{cost:02d}bps'
        for name,frame in value.tables.items():save_table(directory,relative+'/'+name+'.parquet',frame,artifacts)
        save_table(directory,relative+'/monthly.parquet',monthly,artifacts)
        comparison.append(row);yearly.append(annual)
        need(sum(p.stat().st_size for p in BASE.rglob('*') if p.is_file())<=MAX_BACKTEST_OUTPUT,'Backtest output limit')
        print(json.dumps({'phase':'replay','completed_scenarios':len(comparison),'expected_scenarios':120}),flush=True)
    need(len(comparison)==120,'Exactly120 retained scenarios required')
    for name,frame in {'comparison':pd.DataFrame(comparison),'yearly':pd.concat(yearly,ignore_index=True),
                       'selection_coverage':pd.concat(coverage,ignore_index=True)}.items():save_table(directory,name+'.parquet',frame,artifacts)
    result=finish(pp,plan,directory,started,artifacts,{'selection_audit_pin':plan['selection_audit_pin'],
        'preflight_audit_pin':plan['preflight_audit_pin'],'scenarios':120,'sessions':1006,'outcome_months':48,
        'complete_proxy_paths':sum(bool(r['complete_proxy_path']) for r in comparison),
        'withheld_paths':sum(not bool(r['complete_proxy_path']) for r in comparison),
        'old_execution_quote_units_inherited_from_exact_independent_PASS':True,
        'old_execution_quotes_or_actions_changed':False,'missing_selected_positions_dropped':False,
        'XOM_quality_flags_and_precision_warnings_overridden':False,
        'status_note':'Numerical outputs require independent120-path verification before performance handoff.'})
    print(json.dumps({'phase':'replay','status':'COMPLETE_PENDING_INDEPENDENT_VERIFICATION','scenarios':120,'audit_pin':result}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['plan',*PHASES]);p.add_argument('--source-plan')
    p.add_argument('--protocol');p.add_argument('--phase',choices=PHASES);p.add_argument('--output');p.add_argument('--plan');p.add_argument('--reservation')
    a=p.parse_args()
    if a.command=='plan':print(json.dumps(make_plan(a.source_plan,a.protocol,a.phase,a.output)))
    else:{'selection':run_selection,'preflight':run_preflight,'replay':run_replay}[a.command](a)
