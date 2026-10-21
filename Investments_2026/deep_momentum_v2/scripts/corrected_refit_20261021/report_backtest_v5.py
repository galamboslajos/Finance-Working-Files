#!/usr/bin/env python3
"""Passive saved backtest tables after the new120-path independent PASS.

No models, probabilities, positions, prices, features or targets are opened.
Financial JSON parsing is precharged separately from exact saved-table reads.
"""
from pathlib import Path
import argparse
import collections
import itertools
import json
import math
import os
import sys
import time

from control import ROOT,check,pin,read,need,require_reservation,save_new

ARMS=('rf_paper15','xgb_paper15','rf_extended','xgb_extended','rf_tuned_extended')
LABELS=dict(zip(ARMS,('RF · Paper15','XGBoost · Paper15','RF · expanded baseline',
    'XGBoost · expanded','RF · expanded, prior CV settings')))
BASKETS=('ret','class_1_10','class_2_9')
VARIANTS=('asset_stop_10pct','unstopped')
COSTS=(0,5,10,25)
KEYS=('arm','basket','variant','cost_bps')
BASE=ROOT/'run/backtest'
MAX_READ=64*1024**2
MAX_OUTPUT=4*1024**2
MAX_SECONDS=180
VERIFIER_SHA='6fccbc68d858800100d0c359f8544b899b718e3ecb21ca2bdcba923181771a19'
VERIFIER_MATH_SHA='60c6e30176fe58930e8680a6a06504e80a2835c495422f95caf3b6e18bf43cb1'
COMPARISON_META=(*KEYS,'status','expected_sessions','valid_close_sessions','first_failure_date',
    'complete_proxy_path','outcome_start','outcome_end','outcome_months','economic_accounting_certified')
PERFORMANCE=('final_nav','total_return','cagr','annualized_volatility','sharpe_zero_cash_rate',
    'maximum_drawdown','total_one_way_turnover','trading_cost_paid')
COMPARISON=(*COMPARISON_META,'initial_nav','stop_count','trade_count',*PERFORMANCE)
YEAR_META=(*KEYS,'year','expected_sessions','complete_full_period_proxy')
YEAR_FINANCIAL=('initial_nav','final_nav','total_return','annualized_volatility','sharpe_zero_cash_rate',
    'maximum_drawdown','trading_cost_paid')
YEARLY=(*YEAR_META,*YEAR_FINANCIAL)
SELECTION=('arm','basket','signal_date','execution_date','outcome_month','status','original_rows',
    'prediction_eligible','long_candidates','short_candidates','long_selected','short_selected','selected_names_per_side')
PREFLIGHT=('arm','basket','selected_episode_rows','distinct_maximum_horizon_quote_keys','missing_key_episode_rows',
    'quality_flag_episode_rows','unique_missing_keys','unique_quality_flag_keys','retired_new_orders',
    'ready_for_accounting','withheld_selection_months','source_metadata_only','prices_or_outcomes_read',
    'selection_changed','unstopped_maximum_horizons_reviewed_for_both_stop_variants')
NUMERIC_ENV=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')


def clean_pin(item):return {k:item[k] for k in ('path','sha256','size_bytes')}


def unique_pins(named):
    result={}
    for item in named.values():
        item=clean_pin(item);check(item)
        need(item['path'] not in result or result[item['path']]==item,'Conflicting JSON byte pins')
        result[item['path']]=item
    return [result[k] for k in sorted(result)]


def json_budget(z):
    expected=unique_pins(z['JSON_pins'])
    need(z['unique_JSON_pins']==expected,'Report JSON dispatch set changed')
    return sum(p['size_bytes'] for p in expected)


def plan_json(source_path,protocol_path,output):
    """Raw hashes and safe paths/projections only; no PASS/metric JSON parsing."""
    source_path,protocol_path,output=map(lambda p:Path(p).resolve(),(source_path,protocol_path,output))
    need(source_path.is_relative_to(ROOT) and protocol_path.is_relative_to(ROOT)
         and output.is_relative_to(ROOT/'plans'),'New scoped report source/plan paths required')
    source=read(source_path);protocol=read(protocol_path)
    sp,pp=pin(source_path),pin(protocol_path)
    need(source['schema']=='dmv2-corrected-forecasts-backtest-source-plan/v1'
         and source['scenario_count']==120 and protocol['source_plan_pin']==sp
         and protocol['schema']=='dmv2-corrected-forecasts-backtest-protocol/v1'
         and protocol['status']=='SEALED_BEFORE_FINANCIAL_READ','Exact new120-case source/protocol required')
    named={'source':sp,'protocol':pp,'verification':pin(ROOT/'run/verification/backtest.json'),
        'replay':pin(BASE/'replay/audit.json'),'preflight':pin(BASE/'preflight/audit.json')}
    unique=unique_pins(named);amount=sum(p['size_bytes'] for p in unique)
    need(0<amount<=MAX_READ,'Report planning JSON exceeds64MiB')
    codes=[pin(__file__),pin(ROOT/'control.py'),pin(ROOT/'supervise.py')]
    return save_new(output,{'schema':'dmv2-backtest-report-JSON-plan/v1',
        'status':'SEALED_BEFORE_FINANCIAL_JSON_PARSE','JSON_pins':named,'unique_JSON_pins':unique,
        'worker_pin':codes[0],'control_pin':codes[1],'supervisor_pin':codes[2],'code_pins':codes,
        'projected_reads':{'compressed_source':amount,'derived_matrix':0},'new_fits':0,'new_inference_calls':0})


def parsed_json(named):
    cache={};result={}
    for name,item in named.items():
        if item['path'] not in cache:cache[item['path']]=read(check(item))
        result[name]=cache[item['path']]
    return result


def authority(z,data):
    proof,source,protocol,replay,preflight=(data[k] for k in ('verification','source','protocol','replay','preflight'))
    need(proof['schema']=='dmv2-independent-corrected-backtest-verification/v1' and proof['status']=='PASS'
         and proof['worker_pin']['sha256']==VERIFIER_SHA and proof['math_pin']['sha256']==VERIFIER_MATH_SHA
         and proof['scenarios']==120
         and proof['sessions']==1006 and proof['outcome_months']==48 and proof['selection_sets']==15
         and proof['all_scenarios_retained'] is True and proof['original_source_pins_unchanged'] is True
         and proof['new_estimator_fits']==proof['new_inference_calls']==0
         and proof['missing_selected_positions_dropped'] is False
         and proof['source_precision_tolerance_overridden'] is False
         and proof['financial_2024_plus_values_converted'] is False,
         'Final new120-path independent PASS required before any report values')
    check(proof['worker_pin']);check(proof['math_pin'])
    need(proof['source_plan_pin']==z['JSON_pins']['source']
         and proof['backtest_protocol_pin']==z['JSON_pins']['protocol']
         and proof['replay_audit_pin']==z['JSON_pins']['replay']
         and proof['preflight_audit_pin']==z['JSON_pins']['preflight']
         and protocol['source_plan_pin']==z['JSON_pins']['source']
         and proof['final_model_verification_pin']==source['final_model_verification_pin'],
         'Report source or verified phase identity detached')
    for name,value in (('replay',replay),('preflight',preflight)):
        need(value['schema']=='dmv2-corrected-backtest-'+name+'-audit/v1' and value['status']=='COMPLETE'
             and value['source_plan_pin']==proof['source_plan_pin']
             and value['protocol_pin']==proof['backtest_protocol_pin'],'Backtest audit authority changed')
    need(replay['scenarios']==120 and replay['sessions']==1006 and replay['outcome_months']==48
         and replay['complete_proxy_paths']==proof['complete_proxy_paths']
         and replay['withheld_paths']==proof['withheld_paths']
         and proof['complete_proxy_paths']+proof['withheld_paths']==120,'Completed/withheld counts detached')
    return proof


def projection(item,columns,rows):
    import pyarrow.parquet as pq
    item=clean_pin(item);f=pq.ParquetFile(check(item));columns=list(columns)
    need(f.metadata.num_rows==rows and len(set(columns))==len(columns)
         and set(columns)<=set(f.schema_arrow.names),'Passive report table row/schema envelope changed')
    sizes=[]
    for i in range(f.metadata.num_row_groups):
        group=f.metadata.row_group(i);found=collections.defaultdict(int)
        for j in range(group.num_columns):
            field=group.column(j);need(field.total_compressed_size>=0,'Negative report projection cost')
            name=field.path_in_schema.split('.')[0]
            if name in columns:need(field.path_in_schema==name,'Nested report values are not authorized')
            found[name]+=field.total_compressed_size
        need(set(columns)<=set(found),'Projected root missing from report footer')
        sizes.append(sum(found[n] for n in columns))
    return {'pin':item,'columns':columns,'rows':rows,
        'schema_fields':{n:str(f.schema_arrow.field(n).type) for n in columns},
        'row_group_compressed_bytes':sizes,'projected_compressed_bytes':sum(sizes)}


def plan(json_plan_path,expected_sha,reservation,output):
    pp=pin(json_plan_path);need(pp['sha256']==expected_sha,'Report JSON plan hash changed')
    z=read(check(pp));amount=json_budget(z)
    need(z['schema']=='dmv2-backtest-report-JSON-plan/v1' and z['worker_pin']==pin(__file__)
         and z['status']=='SEALED_BEFORE_FINANCIAL_JSON_PARSE'
         and z['projected_reads']=={'compressed_source':amount,'derived_matrix':0},'Report JSON budget/identity changed')
    require_reservation(ROOT,reservation,amount,pp)
    output=Path(output).resolve();marker=output.with_name(output.name+'.STARTED.json')
    need(output.is_relative_to(ROOT/'plans') and not output.exists() and not marker.exists(),
         'Fresh scoped report table-plan attempt required')
    started=save_new(marker,{'schema':'dmv2-passive-backtest-report-planning-attempt/v1',
        'status':'ATTEMPT_CONSUMED','JSON_plan_pin':pp,'reservation':reservation})
    data=parsed_json(z['JSON_pins']);authority(z,data)
    tables=table_plan(data)
    total=amount+sum(p['projected_compressed_bytes'] for specs in tables.values() for p in specs.values())
    need(0<total<=MAX_READ,'Passive report exact reads exceed64MiB')
    return save_new(output,{**z,'schema':'dmv2-backtest-report-table-plan/v1','status':'SEALED_BEFORE_REPORT_VALUES',
        'JSON_plan_pin':pp,'planning_started_pin':started,'tables':tables,'backtest_protocol_pin':z['JSON_pins']['protocol'],
        'projected_reads':{'compressed_source':total,'derived_matrix':0},'maximum_output_bytes':MAX_OUTPUT,
        'maximum_seconds':MAX_SECONDS,'scenarios':120,'sessions':1006,'outcome_months':48})


def table_plan(data):
    """Exact four-audit-artifact authority, reconstructed before each read phase."""
    replay,preflight=data['replay'],data['preflight']
    entries={'comparison':replay['artifacts']['comparison.parquet'],
        'yearly':replay['artifacts']['yearly.parquet'],
        'selection':replay['artifacts']['selection_coverage.parquet'],
        'preflight':preflight['artifacts']['summary.parquet']}
    return {'comparison':{'metadata':projection(entries['comparison'],COMPARISON_META,120),
            'values':projection(entries['comparison'],COMPARISON,120)},
        'yearly':{'metadata':projection(entries['yearly'],YEAR_META,480),
            'values':projection(entries['yearly'],YEARLY,480)},
        'selection':{'metadata':projection(entries['selection'],SELECTION,720)},
        'preflight':{'metadata':projection(entries['preflight'],PREFLIGHT,15)}}


def key(r):return tuple(r[n] for n in KEYS)


def canonical_day(value):
    from datetime import date
    return type(value) is str and date.fromisoformat(value).isoformat()==value


def count(value):return type(value) is int and value>=0


def metadata(comparison,yearly,selection,preflight):
    """All clock/identity/admission guards before financial values are converted."""
    expected=set(itertools.product(ARMS,BASKETS,VARIANTS,COSTS))
    need(len(comparison)==120 and {key(r) for r in comparison}==expected,'All120 unique report scenarios required')
    for r in comparison:
        need(type(r['complete_proxy_path']) is bool and r['expected_sessions']==1006
             and count(r['valid_close_sessions']) and r['valid_close_sessions']<=1006
             and r['outcome_start']=='2020-01-01' and r['outcome_end']=='2023-12-31'
             and r['outcome_months']==48 and r['economic_accounting_certified'] is False,
             'Saved report completeness/period/certification changed')
        need(r['first_failure_date'] is None or canonical_day(r['first_failure_date'])
             and '2020-01-01'<=r['first_failure_date']<'2024-01-01','Protected or malformed failure date')
        if r['complete_proxy_path']:need(r['status']=='complete_proxy' and r['valid_close_sessions']==1006
            and r['first_failure_date'] is None,'Complete report row contains an accounting failure')
    need(len(yearly)==480 and {(key(r),r['year']) for r in yearly}==set(itertools.product(expected,range(2020,2024))),
         'Exact480 annual report rows required')
    compared={key(r):r for r in comparison}
    for r in yearly:
        need(type(r['year']) is int and r['expected_sessions']=={2020:253,2021:252,2022:251,2023:250}[r['year']]
             and type(r['complete_full_period_proxy']) is bool
             and r['complete_full_period_proxy']==compared[key(r)]['complete_proxy_path'],
             'Annual report changed full-period completeness or session clock')
    months={f'{y}-{m:02d}' for y in range(2020,2024) for m in range(1,13)}
    expected_selection=set(itertools.product(ARMS,BASKETS,months))
    need(len(selection)==720 and {(r['arm'],r['basket'],r['outcome_month']) for r in selection}==expected_selection,
         'Exact15-by48 selection coverage grid required')
    for r in selection:
        month=r['outcome_month'];y,m=map(int,month.split('-'));previous=f'{y-1}-12' if m==1 else f'{y}-{m-1:02d}'
        need(canonical_day(r['signal_date']) and r['signal_date'][:7]==previous
             and canonical_day(r['execution_date']) and r['execution_date'][:7]==month,
             'Saved selection report timing changed')
        for n in SELECTION[6:]:need(count(r[n]),'Unknown selection coverage count')
        n=r['selected_names_per_side']
        need(n<=10 and r['long_selected']==r['short_selected']==n
             and (r['status']=='available')==(n>0),'Unbalanced or silently filled selection shortage')
    need(len(preflight)==15 and {(r['arm'],r['basket']) for r in preflight}==set(itertools.product(ARMS,BASKETS)),
         'Exact15 preflight summaries required')
    for r in preflight:
        need(type(r['ready_for_accounting']) is bool and r['source_metadata_only'] is True
             and r['prices_or_outcomes_read'] is False and r['selection_changed'] is False
             and r['unstopped_maximum_horizons_reviewed_for_both_stop_variants'] is True,
             'Preflight report implies a changed selection or numerical source read')
        for n in PREFLIGHT[2:9]:need(count(r[n]),'Unknown preflight count')
        need(count(r['withheld_selection_months']) and r['withheld_selection_months']<=48,'Unknown selection shortage count')
    return True


def finite(value):return type(value) in (float,int) and math.isfinite(value)


def clean_rows(rows):
    return [{k:(None if type(v) is float and not math.isfinite(v) else v) for k,v in r.items()} for r in rows]


def close(a,b):return finite(a) and finite(b) and abs(a-b)<=1e-12*max(1.,abs(b))


def build_report(comparison,yearly,selection,preflight,proof):
    comparison,yearly,selection,preflight=map(clean_rows,(comparison,yearly,selection,preflight))
    metadata(comparison,yearly,selection,preflight)
    compared={key(r):r for r in comparison};annuals=collections.defaultdict(list)
    for r in yearly:annuals[key(r)].append(r)
    complete=sum(r['complete_proxy_path'] for r in comparison)
    need(complete==proof['complete_proxy_paths'] and 120-complete==proof['withheld_paths'],
         'Saved metrics completion counts differ from independent proof')
    for r in comparison:
        need(r['initial_nav']==1. and count(r['stop_count']) and count(r['trade_count']),'Initial capital or recorded ledger counts changed')
        if not r['complete_proxy_path']:
            need(all(r[n] is None for n in PERFORMANCE),'Withheld path exposed full-period performance')
            need(all(all(a[n] is None for n in YEAR_FINANCIAL) for a in annuals[key(r)]),
                 'Failed path exposed annual prefix performance')
            continue
        need(all(finite(r[n]) for n in PERFORMANCE if n!='sharpe_zero_cash_rate')
             and (r['sharpe_zero_cash_rate'] is None or finite(r['sharpe_zero_cash_rate'])),
             'Complete path contains unknown performance')
        need(close(r['final_nav']-1,r['total_return']) and r['final_nav']>0
             and close(r['cagr'],r['final_nav']**(252/1006)-1),'Saved total return/CAGR period differs')
        previous=1.
        for a in sorted(annuals[key(r)],key=lambda a:a['year']):
            need(all(finite(a[n]) for n in YEAR_FINANCIAL if n!='sharpe_zero_cash_rate')
                 and (a['sharpe_zero_cash_rate'] is None or finite(a['sharpe_zero_cash_rate'])),'Complete annual metrics contain unknown values')
            need(close(a['initial_nav'],previous) and close(a['total_return'],a['final_nav']/previous-1),
                 'Annual report resets capital or changes continuous NAV')
            previous=a['final_nav']
        need(close(previous,r['final_nav']),'Annual NAV does not finish at saved full-period NAV')
    coverage=[]
    for arm,basket in itertools.product(ARMS,BASKETS):
        selected=[r for r in selection if (r['arm'],r['basket'])==(arm,basket)]
        pre=next(r for r in preflight if (r['arm'],r['basket'])==(arm,basket))
        withheld=sum(r['status']!='available' for r in selected)
        need(withheld==pre['withheld_selection_months'],'Preflight and saved selection shortage counts differ')
        coverage.append({'arm':arm,'basket':basket,'label':LABELS[arm],
            'selection_months':48,'withheld_selection_months':withheld,
            'minimum_selected_per_side':min(r['selected_names_per_side'] for r in selected),
            'maximum_selected_per_side':max(r['selected_names_per_side'] for r in selected),
            **{k:pre[k] for k in PREFLIGHT[2:]}})
    return {'schema':'dmv2-passive-corrected-backtest-report/v1','status':'FROM_INDEPENDENTLY_VERIFIED_SAVED_OUTPUTS',
        'outcome_start':'2020-01-01','outcome_end':'2023-12-31','first_session':'2020-01-02','last_session':'2023-12-29',
        'outcome_months':48,'sessions':1006,'continuous_NAV':True,'annual_capital_reset':False,
        'scenarios':120,'complete_proxy_paths':complete,'withheld_paths':120-complete,'selection_sets':15,
        'focus':{'arms':['rf_extended','rf_tuned_extended'],'basket':'ret','variant':'asset_stop_10pct',
            'prior_CV_settings_reused_without_search':True,'new_strategy_choice_made':False},
        'labels':LABELS,'primary_basket':'ret','primary_variant':'asset_stop_10pct','costs_bps':list(COSTS),
        'rows':sorted(comparison,key=lambda r:(ARMS.index(r['arm']),BASKETS.index(r['basket']),VARIANTS.index(r['variant']),r['cost_bps'])),
        'yearly':sorted(yearly,key=lambda r:(key(r),r['year'])),'coverage':coverage,
        'new_estimator_fits':0,'new_inference_calls':0,'accounting_replays':0,'old_result_comparison_performed':False,
        'certification':'Conditional historical proxy; identity knowledge-time, survivorship, investability and economic-return certification remain unresolved.',
        'learning_limitation':'RCM/DISCA OHLCV learning candidates failed frozen volume compatibility; no new learning bars admitted. Existing execution-only repairs remain separate.'}


def percent(v):return '—' if v is None else f'{100*v:+.2f}%'


def number(v):return '—' if v is None else f'{v:.2f}'


def state(r):return 'complete' if r['complete_proxy_path'] else 'withheld'


def markdown(report):
    rows={key(r):r for r in report['rows']};lines=['# Corrected-input walk-forward backtest', '',
        'All results cover **2 January 2020–29 December 2023: four years, 48 monthly outcomes, 1,006 trading sessions**. NAV is continuous across annual refits; capital is never reset.',
        f"Independent verification retained all120 paths: **{report['complete_proxy_paths']} complete / {report['withheld_paths']} withheld**. A dash denotes unavailable performance, not a zero return.",
        'RF expanded, the RET basket and the 10% per-asset stop remain the stated primary focus. The class1–10 and class2–9 baskets remain separate comparators. “Baseline” and “prior CV settings” are distinct frozen estimators; no new tuning or strategy selection is performed here.', '']
    for variant,title in (('asset_stop_10pct','10% per-asset stop: primary comparison'),('unstopped','Unstopped comparator')):
        lines.extend(['## '+title,'','Gross is 0bp; net uses10bp per one-way traded notional. Borrow/funding are0 in this frozen proxy. Sharpe uses daily returns and a zero cash rate; CAGR annualizes the full1,006-session path. Drawdown is shown as a negative return.',
            '','| Model | Basket | Gross total | Net10bp total | Net CAGR | Net vol | Net Sharpe | Net max DD | Gross / net status |',
            '|---|---|---:|---:|---:|---:|---:|---:|---|'])
        for arm,basket in itertools.product(ARMS,BASKETS):
            g,n=(rows[(arm,basket,variant,c)] for c in (0,10))
            lines.append(f"| {LABELS[arm]} | {basket} | {percent(g['total_return'])} | {percent(n['total_return'])} | {percent(n['cagr'])} | {percent(n['annualized_volatility'])} | {number(n['sharpe_zero_cash_rate'])} | {percent(n['maximum_drawdown'])} | {state(g)} / {state(n)} |")
        lines.append('')
    lines.extend(['## Trading-cost sensitivity','',
        'Each cell is the complete four-year total return and its availability. All four frozen cost assumptions and both variants are retained.',
        '','| Model | Basket | Stop | 0bp | 5bp | 10bp | 25bp |','|---|---|---|---:|---:|---:|---:|'])
    for arm,basket,variant in itertools.product(ARMS,BASKETS,VARIANTS):
        cells=[f"{percent(rows[(arm,basket,variant,c)]['total_return'])} ({state(rows[(arm,basket,variant,c)])})" for c in COSTS]
        lines.append(f"| {LABELS[arm]} | {basket} | {'10%' if variant=='asset_stop_10pct' else 'none'} | "+' | '.join(cells)+' |')
    lines.extend(['','## Annual returns: 10% stop, 10bp','',
        'These are slices of the same continuous NAV. If any part of a full path fails, every annual return for that path is withheld.',
        '','| Model | Basket | 2020 | 2021 | 2022 | 2023 | Status |','|---|---|---:|---:|---:|---:|---|'])
    annual={(key(r),r['year']):r for r in report['yearly']}
    for arm,basket in itertools.product(ARMS,BASKETS):
        k=(arm,basket,'asset_stop_10pct',10)
        lines.append(f"| {LABELS[arm]} | {basket} | "+' | '.join(percent(annual[(k,y)]['total_return']) for y in range(2020,2024))+f" | {state(rows[k])} |")
    lines.extend(['','## Selection and quote coverage','',
        '| Model | Basket | Withheld selection months /48 | Names per side, min–max | Missing quote keys | Quality-flag keys | Retired new orders |',
        '|---|---|---:|---:|---:|---:|---:|'])
    for r in report['coverage']:lines.append(f"| {r['label']} | {r['basket']} | {r['withheld_selection_months']} | {r['minimum_selected_per_side']}–{r['maximum_selected_per_side']} | {r['unique_missing_keys']} | {r['unique_quality_flag_keys']} | {r['retired_new_orders']} |")
    lines.extend(['',report['learning_limitation'],report['certification'],
        'This passive report reads saved summary tables only. It does not fit, infer, replay, choose a new strategy, certify the source data, or claim improvement against an older run. Recorded stop/trade counts and turnover/cost totals are retained in JSON; failed paths expose no full-period or annual performance.', ''])
    return '\n'.join(lines)


def table(spec):
    import pyarrow.parquet as pq
    need(projection(spec['pin'],spec['columns'],spec['rows'])==spec,'Saved report projection changed')
    return pq.ParquetFile(spec['pin']['path']).read(columns=spec['columns'],use_threads=False).to_pylist()


def require_guard(pp,z):
    path,sha=(os.environ.get(k) for k in ('DMV2_GUARD_LAUNCH_PATH','DMV2_GUARD_LAUNCH_SHA256'))
    need(path and sha and pin(path)['sha256']==sha,'External passive-report resource guard required')
    launch=read(path)
    need(launch['schema']=='dmv2-corrected-refit-phase-launch/v1'
         and launch['protocol_pin']==z['backtest_protocol_pin'] and launch['action_plan_pin']==pp
         and launch['time_category'] in ('backtest','verification') and launch['maximum_seconds']<=MAX_SECONDS
         and launch['command']==[sys.executable,*sys.argv],'Passive report guard command/limits changed')
    need(z['code_pins']==[z['worker_pin'],z['control_pin'],z['supervisor_pin']],'Passive report code closure changed')
    for item in z['code_pins']:need(item in launch['code_pins'],'Unpinned passive-report code')
    need(all(os.environ.get(k)=='1' for k in NUMERIC_ENV),'Single numeric thread environment required')


def run(plan_path,expected_sha,reservation,output):
    start=time.monotonic();pp=pin(plan_path);need(pp['sha256']==expected_sha,'Report table plan hash changed')
    z=read(check(pp));amount=json_budget(z)+sum(p['projected_compressed_bytes'] for specs in z['tables'].values() for p in specs.values())
    need(z['schema']=='dmv2-backtest-report-table-plan/v1' and z['worker_pin']==pin(__file__)
         and z['status']=='SEALED_BEFORE_REPORT_VALUES'
         and z['projected_reads']=={'compressed_source':amount,'derived_matrix':0}
         and 0<amount<=MAX_READ and z['maximum_output_bytes']==MAX_OUTPUT and z['maximum_seconds']==MAX_SECONDS,
         'Passive report sealed identity/read/output limits changed')
    require_reservation(ROOT,reservation,amount,pp);require_guard(pp,z)
    for item in (z['JSON_plan_pin'],z['worker_pin'],z['control_pin'],z['supervisor_pin']):check(item)
    output=Path(output).resolve();need(output==ROOT/'run/reports/backtest_v1' and not output.exists(),'Fresh exact report output required')
    output.mkdir(parents=True)
    started=save_new(output/'STARTED.json',{'schema':'dmv2-passive-backtest-report-attempt/v1',
        'status':'ATTEMPT_CONSUMED','plan_pin':pp,'reservation':reservation})
    data=parsed_json(z['JSON_pins']);proof=authority(z,data)
    need(z['tables']==table_plan(data),'Runtime report projections detached from the four verified audit artifacts')
    meta={name:table(specs['metadata']) for name,specs in z['tables'].items()}
    metadata(meta['comparison'],meta['yearly'],meta['selection'],meta['preflight'])
    values={name:table(z['tables'][name]['values']) for name in ('comparison','yearly')}
    for name,fields in (('comparison',COMPARISON_META),('yearly',YEAR_META)):
        need([{k:r[k] for k in fields} for r in values[name]]==meta[name],'Report value keys differ from guarded metadata')
    report=build_report(values['comparison'],values['yearly'],meta['selection'],meta['preflight'],proof)
    report.update(independent_verification_pin=z['JSON_pins']['verification'],source_plan_pin=z['JSON_pins']['source'],
        backtest_protocol_pin=z['backtest_protocol_pin'],plan_pin=pp)
    need(time.monotonic()-start<=MAX_SECONDS,'Passive report wall limit exceeded')
    for item in unique_pins(z['JSON_pins']):check(item)
    for specs in z['tables'].values():
        for spec in specs.values():check(spec['pin'])
    for item in (z['JSON_plan_pin'],z['worker_pin'],z['control_pin'],z['supervisor_pin']):check(item)
    result=save_new(output/'report.json',report)
    prose=markdown(report);need(len(prose.encode())<=MAX_OUTPUT,'Passive markdown output exceeds budget')
    with (output/'report.md').open('x') as stream:stream.write(prose)
    artifacts={'report':result,'markdown':pin(output/'report.md')}
    need(sum(p.stat().st_size for p in output.iterdir())<=MAX_OUTPUT,'Passive report output exceeds4MiB')
    return save_new(output/'receipt.json',{'schema':'dmv2-passive-corrected-backtest-report-receipt/v1',
        'status':'COMPLETE_FROM_INDEPENDENTLY_VERIFIED_OUTPUTS','plan_pin':pp,'started_pin':started,'worker_pin':pin(__file__),
        'independent_verification_pin':z['JSON_pins']['verification'],'source_plan_pin':z['JSON_pins']['source'],
        'artifacts':artifacts,'scenarios':120,'complete_proxy_paths':report['complete_proxy_paths'],
        'withheld_paths':report['withheld_paths'],'sessions':1006,'outcome_months':48,
        'reserved_compressed_source_bytes':amount,'source_or_model_tables_read':False,
        'new_estimator_fits':0,'new_inference_calls':0,'accounting_replays':0,'old_result_comparison_performed':False,
        'execution_seconds':time.monotonic()-start,'source_and_artifact_pins_unchanged':True})


def readiness():
    """No proof contents parsed; readiness is not a numerical PASS claim."""
    paths={'independent_backtest':ROOT/'run/verification/backtest.json',
        'source_plan':ROOT/'backtest_source_plan.json','protocol':ROOT/'backtest_protocol.json',
        'replay_audit':BASE/'replay/audit.json','preflight_audit':BASE/'preflight/audit.json'}
    present={name:pin(path) if path.is_file() else None for name,path in paths.items()}
    return {'schema':'dmv2-passive-backtest-report-readiness/v1',
        'status':'FILES_PRESENT_REQUIRES_PRECHARGED_PASS_CHECK' if all(present.values()) else 'WAITING_FOR_VERIFIED_OUTPUT_FILES',
        'pins':present,'proof_contents_parsed':False,'source_tables_decoded':False,
        'commands':['plan-json','plan','run'],'maximum_read_bytes':MAX_READ,
        'maximum_output_bytes':MAX_OUTPUT,'maximum_seconds':MAX_SECONDS,
        'output_directory':str(ROOT/'run/reports/backtest_v1')}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('readiness','plan-json','plan','run'))
    for name in ('source-plan','protocol','json-plan','json-plan-sha256','plan','plan-sha256','reservation','output'):p.add_argument('--'+name)
    a=p.parse_args()
    if a.command=='readiness':value=readiness()
    elif a.command=='plan-json':value=plan_json(a.source_plan,a.protocol,a.output)
    elif a.command=='plan':value=plan(a.json_plan,a.json_plan_sha256,a.reservation,a.output)
    else:value=run(a.plan,a.plan_sha256,a.reservation,a.output)
    print(json.dumps(value,sort_keys=True))
