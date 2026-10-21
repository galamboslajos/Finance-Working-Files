"""New-forecast backtest integration scaffold; no model/feature/label loading.

Import is standard-library only. `plan` inspects pinned metadata and footer sizes.
The in-memory adapters reuse unchanged production selection/accounting workers;
there is intentionally no command that decodes quotes or runs a replay yet.
A separately sealed root phase must own source reads, checkpoints and saving.
"""
from pathlib import Path
import argparse
import datetime as dt
import itertools
import hashlib
import json
import sys
from control import ROOT,check,pin,read,save_new,need,require_reservation

YEARS=(2020,2021,2022,2023)
ARMS=('rf_paper15','xgb_paper15','rf_extended','xgb_extended','rf_tuned_extended')
OLD_ARM_ALIAS={'rf_tuned_extended':'rf_extended_tuned'}
BASKETS=('ret','class_1_10','class_2_9')
VARIANTS={'unstopped':None,'asset_stop_10pct':.1}
COSTS=(0,5,10,25)
SCENARIOS=tuple(itertools.product(ARMS,BASKETS,VARIANTS,COSTS))
KEYS=('source_security_id','signal_date')
PROBS=tuple(f'probability_{n}' for n in range(1,11))
LINEAGE=('fit_year','training_outcome_start','training_outcome_end','training_label_observed_through')
REQUIRED_META=(*KEYS,'intrinio_security_id','outcome_month','outcome_start','outcome_end',
 'identity_eligible','paper_prediction_eligible','prediction_eligible','prediction_status',*LINEAGE)
# Exact old accounting inputs: intentionally no feature-only volume fields.
ACTION_COLUMNS=('trading_date','intrinio_security_id','adj_open','adj_high','adj_low','adj_close',
 'frequency','intraperiod','price_quality_status','adjusted_ohlc_bounds_valid','raw_ohlc_bounds_valid',
 'factor','split_ratio','dividend','open','high','low','close')
QUOTE_META=('trading_date','intrinio_security_id','frequency','intraperiod','price_quality_status',
 'raw_ohlc_bounds_valid','adjusted_ohlc_bounds_valid')
BK_SOURCE='siblis_ticker_company_1c21a7b293fd137c59c9';BK_KEY='sec_zq1J05'
XOM_SOURCE='siblis_isin_b284bb7497616f5afba0'
XOM_KEY='dmv2_sip:siblis_isin_b284bb7497616f5afba0:all'
WORKERS=('ret_selection.py','walkforward_selection.py','ret.py','walkforward_clock.py',
 'walkforward_preflight.py','walkforward_accounting_v2.py','walkforward_accounting.py',
 'backtest_accounting.py','ret_repair_backtest.py','action_loading.py','backtest_loading.py',
 'daily_backtest.py','walkforward_backtest.py','catalog.py','universe.py','monthly.py',
 'exchange_sessions.py','identity_policy.py','backtest_selection.py','walkforward_backtest_v2.py')
FRAME_NAMES={'ledger':'original_ledger','selections':'selected','rebalances':'rebalances','monthly':'monthly'}


def canonical(value):
    return 'sha256:'+hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def day(value):
    need(type(value) is str and dt.date.fromisoformat(value).isoformat()==value and value<'2024-01-01','Canonical bounded backtest date required')
    return value


def exact_pin(item):
    return {k:item[k] for k in ('path','sha256','size_bytes')}


def forecasts_metadata(rows,year,windows):
    """Validate exact original metadata before numerical probabilities enter Python."""
    need(year in YEARS and len(windows)==48,'Frozen four-year clock required')
    expected={r['signal_date']:r for r in windows if r['outcome_month'].startswith(str(year)+'-')}
    need(len(expected)==12 and 0<len(rows)<=12000,'Twelve original monthly cohorts required')
    seen=set()
    cutoffs={2020:'2019-12-31',2021:'2020-12-31',2022:'2021-12-31',2023:'2022-12-30'}
    for row in rows:
        need(set(REQUIRED_META)<=set(row),'New forecast original metadata incomplete')
        sid,signal=row['source_security_id'],day(row['signal_date'])
        need(type(sid) is str and sid.strip()==sid and bool(sid),'Nonblank original source required')
        need((sid,signal) not in seen and signal in expected,'Duplicate or unknown forecast source/signal')
        seen.add((sid,signal));window=expected[signal]
        need(all(row[k]==window[k] for k in ('outcome_month','outcome_start','outcome_end')),'Forecast clocks differ from independent XNYS')
        need(all(type(row[k]) is bool for k in ('identity_eligible','paper_prediction_eligible','prediction_eligible')),'Explicit admission booleans required')
        need(row['prediction_eligible']==row['paper_prediction_eligible'] and (not row['prediction_eligible'] or row['identity_eligible']),'Model changed original cohort')
        need(row['prediction_status']==('predicted' if row['prediction_eligible'] else 'withheld_signal_time_baseline'),'Label availability changed prediction admission')
        need(row['fit_year']==year and row['training_outcome_start']==f'{year-10}-01-01'
            and row['training_outcome_end']==f'{year-1}-12-31' and row['training_label_observed_through']==cutoffs[year],'New forecast rolling fit lineage differs')
        native=row['intrinio_security_id']
        if row['prediction_eligible']:
            need(type(native) is str and bool(native.strip()),'Eligible forecast missing computational price key')
            if sid==BK_SOURCE:need(native==BK_KEY,'BK amended native route lost')
            if sid==XOM_SOURCE:need(native==XOM_KEY,'Typed SIP XOM key lost or promoted to provider native')
    need({signal for _,signal in seen}==set(expected),'Missing one of twelve original monthly cohorts')
    eligible=[r for r in rows if r['prediction_eligible']]
    need(len({(r['signal_date'],r['intrinio_security_id']) for r in eligible})==len(eligible),'Dated eligible native collision')
    return [(r['source_security_id'],r['signal_date'],r['intrinio_security_id'],r['identity_eligible'],r['paper_prediction_eligible'],r['prediction_eligible'],*[r[k] for k in LINEAGE]) for r in rows]


def footer_projection(item,columns,max_rows):
    """Footer names/types/lengths only, never financial statistics or values."""
    import pyarrow.parquet as pq
    source=pq.ParquetFile(check(item));names=source.schema_arrow.names
    need(len(names)==len(set(names)) and set(columns)<=set(names),'Backtest projected schema missing/duplicated')
    need(source.metadata.num_rows<=max_rows,'Backtest original row ceiling exceeded')
    sizes=[]
    for i in range(source.metadata.num_row_groups):
        group=source.metadata.row_group(i)
        found={}
        for j in range(group.num_columns):
            chunk=group.column(j);root=chunk.path_in_schema.split('.')[0]
            if root in columns:
                need(chunk.path_in_schema==root and root not in found,'Nested projected fields require a separate exact leaf plan')
                found[root]=chunk.total_compressed_size
        need(set(found)==set(columns),'Projected physical roots missing from footer')
        sizes.append(sum(found.values()))
    return {'pin':exact_pin(item),'columns':list(columns),'rows':source.metadata.num_rows,
        'schema_fields':{n:str(source.schema_arrow.field(n).type) for n in columns},
        'row_group_compressed_bytes':sizes,'projected_compressed_bytes':sum(sizes)}


def planning_json_pins(control):
    items=[control[n] for n in ('catalog_pin','final_model_verification_pin','model_protocol_pin',
        'execution_phase_pin','execution_verification_pin','execution_backtest_audit_pin',
        'execution_manifest_pin','base_configuration_pin')]
    items+=list(control['model_verification_pins'].values())
    items+=[e['prediction_receipt_pin'] for e in control['forecast_entries']]
    distinct={str(Path(x['path']).resolve()):exact_pin(x) for x in items}
    return [distinct[name] for name in sorted(distinct)]


def planning_json_plan(control_path,output):
    """Hash/size-only reservation plan; no metric-containing receipt is parsed."""
    control=read(control_path);items=planning_json_pins(control)
    for item in items:check(item)
    return save_new(output,{'schema':'dmv2-corrected-backtest-planning-JSON-plan/v1',
        'status':'SEALED_BEFORE_FINANCIAL_JSON_PARSE','control_pin':pin(control_path),
        'worker_pin':pin(__file__),'JSON_pins':items,
        'projected_bytes':sum(x['size_bytes'] for x in items)})


def plan_sources(control_path,output,*,planning_plan,planning_reservation):
    """Root calls only after all20 forecasts and four independent model PASSes exist."""
    control=read(control_path)
    charged=read(planning_plan);charged_pin=pin(planning_plan)
    need(charged['schema']=='dmv2-corrected-backtest-planning-JSON-plan/v1'
         and charged['status']=='SEALED_BEFORE_FINANCIAL_JSON_PARSE'
         and charged['control_pin']==pin(control_path) and charged['worker_pin']==pin(__file__),'Planning JSON/code binding changed')
    expected_json=planning_json_pins(control)
    need(charged['JSON_pins']==expected_json and type(charged['projected_bytes']) is int
         and charged['projected_bytes']==sum(x['size_bytes'] for x in expected_json),'Planning JSON exact set/bytes changed')
    require_reservation(ROOT,planning_reservation,charged['projected_bytes'],charged_pin)
    for item in charged['JSON_pins']:check(item)
    save_new(str(output)+'.STARTED.json',{'status':'ATTEMPT_CONSUMED','planning_plan_pin':charged_pin})
    need(control['schema']=='dmv2-corrected-forecasts-backtest-control/v1'
        and control['status']=='FROZEN_AFTER_TWENTY_MODEL_VERIFICATIONS'
        and control['arm_ids']==list(ARMS) and control['baskets']==list(BASKETS)
        and control['scenarios']==120 and control['financial_2024_plus_authorized'] is False,
        'Explicit new-forecast120-scenario protocol required')
    project=Path(control['project_root']).resolve()
    need(project.name=='deep_momentum_v2','V2 project root required')
    catalog=read(check(control['catalog_pin']));need(len(catalog['annuals'])==14 and len(catalog['RET_states'])==48,'Complete corrected catalog+48RET states required')
    # Before ANY source/footer access, freeze exact path/column metadata scope.
    entries=control['forecast_entries']
    need(len(entries)==20 and {(e['year'],e['arm']) for e in entries}==set(itertools.product(YEARS,ARMS)),'Exactly20 new forecast entries required')
    scope={'schema':'dmv2-corrected-backtest-footer-scope/v1','control_pin':pin(control_path),
        'forecasts':[exact_pin(e['predictions_pin']) for e in entries],
        'daily':[exact_pin(a['artifacts']['daily_prices_raw']) for a in catalog['annuals'] if a['year'] in YEARS],
        'RET':[exact_pin(s['artifacts'][n]) for s in catalog['RET_states'] for n in ('payoffs.parquet','monthly.parquet')],
        'statistics_read':False,'financial_values_read':False,'maximum_unique_footer_files':121,
        'maximum_footer_open_passes':214}
    scope_pin=save_new(ROOT/'backtest_footer_scope.json',scope)
    model_protocol=read(check(control['model_protocol_pin']))
    need(model_protocol['catalog_pin']==control['catalog_pin'],'Model protocol catalog binding changed')
    final_proof=read(check(control['final_model_verification_pin']))
    need(final_proof['schema']=='dmv2-independent-corrected-model-final-verification/v1'
         and final_proof['status']=='PASS' and final_proof['verified_estimator_fits']==20
         and final_proof['verified_training_only_preprocessing_states']==8
         and final_proof['all_stitched_forecasts_exact_to_independent_annual_proofs'] is True
         and final_proof['same_cohort_all_arms'] is True and final_proof['catalog_pin']==control['catalog_pin']
         and final_proof['protocol_pin']==control['model_protocol_pin']
         and final_proof['annual_verification_pins']==control['model_verification_pins'],'Final20/8model proof binding required')
    model_proofs={}
    for year in YEARS:
        proof=read(check(control['model_verification_pins'][str(year)]))
        need(proof['status']=='PASS' and proof['year']==year and set(proof['arms'])==set(ARMS)
             and proof['catalog_pin']==control['catalog_pin'],'Exact independently verified new forecasts required')
        model_proofs[year]=proof
    forecasts=[]
    for entry in entries:
        verified=model_proofs[entry['year']]['arms'][entry['arm']]
        need(verified['prediction_receipt_pin']==entry['prediction_receipt_pin'],'New forecast not bound to independent model PASS')
        receipt=read(check(entry['prediction_receipt_pin']))
        need(receipt['status']=='COMPLETE' and receipt['year']==entry['year'] and receipt['arm']==entry['arm']
             and exact_pin(receipt['artifacts']['predictions'])==exact_pin(entry['predictions_pin']),'New forecast artifact differs from verified prediction receipt')
        source=Path(check(entry['predictions_pin']))
        import pyarrow.parquet as pq
        names=pq.ParquetFile(source).schema_arrow.names
        from_columns=[n for n in names if n in (*REQUIRED_META,'requested','identity_status','source_identity_status',
          'identity_eligibility_policy','identity_eligibility_reason','identity_review_required','resolution_status')]
        need(set(REQUIRED_META)<=set(from_columns),'New forecast metadata schema incomplete')
        forecasts.append({**entry,'metadata':footer_projection(entry['predictions_pin'],from_columns,12000),
            'probabilities':footer_projection(entry['predictions_pin'],[*KEYS,*PROBS],12000)})
    daily=[]
    for a in catalog['annuals']:
        if a['year'] in YEARS:
            source=a['artifacts']['daily_prices_raw']
            daily.append({'year':a['year'],'metadata':footer_projection(source,QUOTE_META,300000),
                'values':footer_projection(source,ACTION_COLUMNS,300000)})
    payoffs=[]
    for entry in catalog['RET_states']:
        signal=day(entry['signal_date']);art=entry['artifacts']
        payoffs.append({'signal_date':signal,'state_pin':exact_pin(art['payoff_state.json']),
            'values':footer_projection(art['payoffs.parquet'],['class_id','payoff','status'],10),
            'monthly_metadata':footer_projection(art['monthly.parquet'],['outcome_month','outcome_end','class_id',
                'original_member_count','prediction_eligible_count','member_count','status'],1200),
            'monthly':footer_projection(art['monthly.parquet'],['outcome_month','outcome_end','class_id',
                'original_member_count','prediction_eligible_count','member_count','class_mean_return','status'],1200)})
    phase=read(check(control['execution_phase_pin']));manifest=read(check(control['execution_manifest_pin']))
    old_pass=read(check(control['execution_verification_pin']))
    need(old_pass['schema']=='dmv2-ret-repair-independent-verification/v1'
        and old_pass['backtest_phase_identity_sha256']==canonical(phase)
        and old_pass['backtest_audit_sha256']==exact_pin(control['execution_backtest_audit_pin'])['sha256']
        and old_pass['scenarios']==40 and old_pass['sessions']==1006 and old_pass['outcome_months']==48
        and old_pass['source_pins_unchanged'] is True
        and old_pass['prior_retirements_and_PENN_exception_preserved'] is True,'Old execution PASS binding changed')
    check(control['execution_backtest_audit_pin'])
    need(old_pass['repair_manifest_pin']['sha256']==control['execution_manifest_pin']['sha256']
         and phase['repair_manifest_pin']['sha256']==control['execution_manifest_pin']['sha256'],'Execution manifest differs from original independent PASS')
    need(old_pass['status']=='PASS' and len(phase['event_registry']['events'])==18
        and len(phase['historical_retirements'])==11 and len(manifest['quote_amendments'])==40,'Pinned18+11event and40quote execution reference required')
    old_quotes=manifest['quotes_pin'];old_quotes={**old_quotes,'path':str(project/old_quotes['path'])} if not Path(old_quotes['path']).is_absolute() else old_quotes
    repairs={'metadata':footer_projection(old_quotes,QUOTE_META,40),'values':footer_projection(old_quotes,ACTION_COLUMNS,40)}
    evidence={}
    def collect(value):
        if isinstance(value,dict):
            if {'path','sha256','size_bytes'}<=set(value):
                item=exact_pin(value);item['path']=str((project/Path(item['path'])).resolve()) if not Path(item['path']).is_absolute() else str(Path(item['path']).resolve())
                evidence[item['path']]=item
            for child in value.values():collect(child)
        elif isinstance(value,list):
            for child in value:collect(child)
    collect(phase);collect(manifest)
    read_cost={'selection':sum(f['metadata']['projected_compressed_bytes']+f['probabilities']['projected_compressed_bytes'] for f in forecasts)
        +sum(p['values']['projected_compressed_bytes']+p['monthly_metadata']['projected_compressed_bytes']+p['monthly']['projected_compressed_bytes']+p['state_pin']['size_bytes'] for p in payoffs),
        'quote_preflight':sum(d['metadata']['projected_compressed_bytes'] for d in daily)+repairs['metadata']['projected_compressed_bytes'],
        'quote_replay':sum(d['metadata']['projected_compressed_bytes']+d['values']['projected_compressed_bytes'] for d in daily)+repairs['metadata']['projected_compressed_bytes']+repairs['values']['projected_compressed_bytes']}
    result={'schema':'dmv2-corrected-forecasts-backtest-source-plan/v1','status':'PLANNED_NOT_EXECUTION_AUTHORIZED',
        'control_pin':pin(control_path),'scope_pin':scope_pin,'catalog_pin':control['catalog_pin'],
        'planning_JSON_plan_pin':charged_pin,'planning_JSON_reservation':planning_reservation,
        'final_model_verification_pin':control['final_model_verification_pin'],
        'model_protocol_pin':control['model_protocol_pin'],'model_verification_pins':control['model_verification_pins'],
        'input_verification_pin':model_protocol['input_verification_pin'],
        'terminal_guard_pin':model_protocol['terminal_guard_pin'],'runtime':model_protocol['runtime'],
        'base_configuration_pin':control['base_configuration_pin'],
        'forecasts':forecasts,'daily':daily,'payoffs':payoffs,'execution_repairs':repairs,
        'execution_phase_pin':control['execution_phase_pin'],'execution_manifest_pin':control['execution_manifest_pin'],
        'execution_verification_pin':control['execution_verification_pin'],
        'execution_backtest_audit_pin':control['execution_backtest_audit_pin'],
        'execution_evidence_pins':list(evidence.values()),
        'execution_authority':{'base_configuration_pin':control['base_configuration_pin'],
            'execution_phase_pin':control['execution_phase_pin'],'execution_manifest_pin':control['execution_manifest_pin'],
            'execution_verification_pin':control['execution_verification_pin'],
            'event_registry_sha256':canonical(phase['event_registry']),
            'historical_retirements_sha256':canonical(phase['historical_retirements']),
            'identity_handover_registry_sha256':canonical(phase['identity_handover_registry']),
            'identity_handover_approval_sha256':canonical(phase['identity_handover_approval']),
            'event_count':18,'historical_retirements':11,'execution_quote_repairs':40},
        'worker_pins':[pin(project/'src/dmv2'/name) for name in WORKERS],
        'scaffold_pin':pin(__file__),'source_read_projections':read_cost,
        'additional_read_reservations_required':['Small identity/financial JSON and saved60selection frames per later phase.','Independent verification outputs/ledgers must have separate exact projections.'],
        'scenario_count':120,'selection_sets':15,'fresh_selections_required':True,'old_selection_reuse':False,
        'new_model_fits':0,'new_inference_calls':0,'financial_source_values_read':False,
        'limitations':['Planning scaffold only: phase-specific launch/read reservation and independent120-path verifier are still required.','Maximum-horizon quote preflight diagnoses gaps; never changes selection or replaces failed holdings.']}
    for item in charged['JSON_pins']:check(item)
    return save_new(output,result)


def scoped_quote_positions(rows,year,sessions):
    """Calendar/year projection only; holdings/future quote availability never enter."""
    need(year in YEARS and len(sessions)==1006 and tuple(sessions)==tuple(sorted(set(sessions)))
         and sessions[0]>='2020-01-01' and sessions[-1]<'2024-01-01','Frozen1006session quote scope required')
    calendar=set(sessions);seen=set();positions=[];outside=[]
    for i,row in enumerate(rows):
        native=row['intrinio_security_id'];value=row['trading_date']
        need(type(native) is str and bool(native.strip()),'Blank computational quote identity')
        need(type(value) is str and dt.date.fromisoformat(value).isoformat()==value,'Noncanonical quote metadata date')
        key=native,value;need(key not in seen,'Duplicate original source quote key');seen.add(key)
        if value.startswith(str(year)+'-') and value in calendar:positions.append(i)
        else:outside.append({'intrinio_security_id':native,'trading_date':value})
    return positions,outside


def shift_month(value,offset):
    year,month=map(int,value.split('-'));total=year*12+month-1+offset
    return f'{total//12:04d}-{total%12+1:02d}'


def validate_ret_metadata(meta,state,signal):
    day(signal);provenance=state['provenance'];first=shift_month(signal[:7],-119)
    need(provenance['signal_date']==signal and provenance['label_observed_through']<=signal
         and provenance['first_outcome_month']==first and provenance['last_outcome_month']==signal[:7]
         and type(provenance['vector_available']) is bool,'RET causal provenance changed')
    expected=[(shift_month(first,i),c) for i in range(120) for c in range(1,11)]
    need(len(meta)==1200 and [(r['outcome_month'],r['class_id']) for r in meta]==expected,'RET ordered120x10 metadata grid changed')
    for row in meta:
        end=row['outcome_end']
        need(end is None or day(end)[:7]==row['outcome_month'] and end<=signal,'RET endpoint not completed at signal')
        need(all(type(row[n]) is int and row[n]>=0 for n in ('original_member_count','prediction_eligible_count','member_count')),'RET source count metadata invalid')
        if row['status']=='available':need(row['member_count']>=1 and end is not None,'RET observed cell lacks completed nonempty source')
    need(provenance['vector_available']==all(r['status']=='available' for r in meta),'RET whole-vector metadata availability changed')


def restore_payoff_in_memory(table,monthly,state,*,signal):
    """Restore saved Arrow schema after causal metadata checks, never new payoffs."""
    day(signal)
    from dmv2.ret import RetPayoffPanel,PAYOFF_SCHEMA,MONTHLY_PAYOFF_SCHEMA,verify_ret_payoffs
    need(state['provenance']['signal_date']==signal and state['provenance']['label_observed_through']<=signal,
         'RET saved state has future/wrong cutoff')
    meta=monthly.select([n for n in monthly.column_names if n!='class_mean_return']).to_pylist()
    validate_ret_metadata(meta,state,signal)
    need(table.num_rows==10,'RET ten-class vector size changed')
    need(table.select(['class_id','status']).to_pylist()==[{'class_id':n,'status':('available' if state['provenance']['vector_available'] else 'withheld_incomplete_120_month_class_grid')} for n in range(1,11)],'RET vector class/status metadata changed')
    for row in meta:
        date=row['outcome_end']
        need(date is None or day(date)[:7]==row['outcome_month'] and date<=signal,'RET source endpoint not completed at signal')
    result=RetPayoffPanel(table.cast(PAYOFF_SCHEMA),monthly.cast(MONTHLY_PAYOFF_SCHEMA),state['provenance'],state['fingerprint'])
    verify_ret_payoffs(result)
    return result


def _workers(project):
    sys.path.insert(0,str(Path(project)/'src'))
    from dmv2.walkforward_clock import walkforward_clock
    from dmv2.walkforward_selection import select_walkforward_basket,validate_walkforward_basket,SAFE_COLUMNS
    from dmv2.ret_selection import select_ret_walkforward_basket,validate_ret_walkforward_basket
    from dmv2.walkforward_preflight import preflight_selected_horizons
    return locals()


def select_in_memory(predictions,payoffs,*,project,configuration):
    """Fresh signal-only15selection sets; no quote/target/model arguments exist."""
    import pandas as pd
    worker=_workers(project);sessions,windows,clock=worker['walkforward_clock']()
    selection_sessions=clock['selection_sessions']
    need(tuple(predictions)==ARMS,'New forecast arm order differs')
    common=None;result={};validation=[]
    for arm in ARMS:
        original=predictions[arm]
        safe=[n for n in worker['SAFE_COLUMNS'] if n in original]+list(PROBS)
        need(len(safe)==len(set(safe)) and set(REQUIRED_META)<=set(original),'Forecast selection metadata incomplete')
        source=original[safe].copy(deep=True)
        cohort=original[[*KEYS,'intrinio_security_id','prediction_eligible','paper_prediction_eligible','identity_eligible']]
        if common is None:common=cohort
        else:pd.testing.assert_frame_equal(common,cohort,check_exact=True)
        for basket in BASKETS:
            if basket=='ret':
                selected=worker['select_ret_walkforward_basket'](source,payoffs=payoffs,sessions=selection_sessions,configuration=configuration)
                check_result=worker['validate_ret_walkforward_basket'](selected,source,payoffs=payoffs,sessions=selection_sessions,configuration=configuration)
            else:
                selected=worker['select_walkforward_basket'](source,basket=basket,sessions=selection_sessions,configuration=configuration)
                check_result=worker['validate_walkforward_basket'](selected,source,basket=basket,sessions=selection_sessions,configuration=configuration)
            result[(arm,basket)]=selected;validation.append({'arm':arm,'basket':basket,**check_result})
    need(len(result)==15,'Fifteen fresh selection sets required')
    return result,validation


def preflight_in_memory(selections,quote_metadata,*,project,execution_phase):
    worker=_workers(project);sessions,_,_=worker['walkforward_clock']()
    need(len(execution_phase['event_registry']['events'])==18 and len(execution_phase['historical_retirements'])==11,'Execution event scope changed')
    need(len(selections)==15,'Fresh15selection sets required before quote preflight')
    return {(arm,basket):worker['preflight_selected_horizons'](s.selections,s.rebalances,quote_metadata,sessions,
        events=execution_phase['event_registry']['events'],historical_retirements=execution_phase['historical_retirements'])
        for (arm,basket),s in selections.items()}


def replay_in_memory(selections,prices,*,project,execution_phase,identity_handovers):
    """Yield all120paths, retaining failures. Caller seals/reserves/saves each path."""
    worker=_workers(project);sessions,_,_=worker['walkforward_clock']()
    from dmv2.walkforward_accounting_v2 import run_walkforward_accounting_v2
    from dmv2.backtest_accounting import DailyAccountingPolicy
    from dmv2.daily_backtest import summarize_daily
    from dmv2.walkforward_backtest import yearly_breakdown
    need(set(selections)==set(itertools.product(ARMS,BASKETS)),'Fresh15selection sets required')
    for arm,basket,variant,cost in SCENARIOS:
        s=selections[(arm,basket)]
        value=run_walkforward_accounting_v2(s.selections,prices,sessions,rebalances=s.rebalances,
            terminal_date='2023-12-29',events=execution_phase['event_registry']['events'],
            historical_retirements=execution_phase['historical_retirements'],identity_handovers=identity_handovers,
            policy=DailyAccountingPolicy(cost_bps=cost,borrow_annual_bps=0,required_names_per_side=None,asset_stop_fraction=VARIANTS[variant]))
        row,monthly=summarize_daily(value,arm=arm,basket=basket,variant=variant,cost_bps=cost,expected_sessions=1006)
        row.update(outcome_start='2020-01-01',outcome_end='2023-12-31',outcome_months=48)
        annual=yearly_breakdown(value,arm=arm,basket=basket,variant=variant,cost_bps=cost)
        yield (arm,basket,variant,cost),value,row,monthly,annual


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['planning-plan','plan']);p.add_argument('--control',required=True);p.add_argument('--output',required=True)
    p.add_argument('--planning-plan');p.add_argument('--planning-reservation')
    a=p.parse_args();value=planning_json_plan(a.control,a.output) if a.command=='planning-plan' else plan_sources(a.control,a.output,planning_plan=a.planning_plan,planning_reservation=a.planning_reservation)
    print(json.dumps(value))
