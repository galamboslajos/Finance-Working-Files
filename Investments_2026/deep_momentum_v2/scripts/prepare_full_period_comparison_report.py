#!/usr/bin/env python3
"""Display-only comparison of independently verified monthly and daily aggregates."""
from pathlib import Path
import argparse,json,hashlib,math
import pyarrow.parquet as pq
from dmv2.extension_backtest_reporting import read_extension_backtest


def read(path):return json.loads(Path(path).read_text())
def pin(path):
    p=Path(path).resolve();return dict(path=str(p),size_bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def check(item):
    if pin(item['path'])!=item:raise ValueError('Changed verified aggregate')
    return Path(item['path'])
def save(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    return pin(path)


def prepare(run,output):
    root=Path(run).resolve();out=Path(output).resolve()
    if out.exists():raise ValueError('Preserve earlier display export')
    saved=read_extension_backtest(root);proofpin=pin(saved['independent_proof']);proof=read(check(proofpin));audit=read(check(proof['audit_pin']))
    if proof['monthly_references_verified']!=16 or proof['complete_paths']!=48 or proof['withheld_paths']!=0:raise ValueError('Full comparison verification required')
    monthly={k:v for k,v in audit['artifacts'].items() if '/monthly_cost' in k and k.endswith('/monthly')}
    if len(monthly)!=16 or sum(v['size_bytes'] for v in monthly.values())>1024**2:raise ValueError('Bounded sixteen monthly tables required')
    out.mkdir(parents=True);source={**monthly,'independent_proof':proofpin,'daily_report':pin(root/'report/summary.json'),'daily_display_proof':pin(root/'report/verification.json')}
    save(out/'read_plan.json',dict(status='SEALED_BEFORE_MONTHLY_AGGREGATE_READ',inputs=source,columns=['outcome_month','nav_start','nav_end','net_return'],maximum_rows_per_table=75,source_array_or_model_reads=False,new_fits=0,new_backtests=0))
    arms=['rf_paper15_5y','xgb_paper15_5y','rf_paper15_10y','xgb_paper15_10y'];names=['RF · 5-year','XGBoost · 5-year','RF · 10-year','XGBoost · 10-year'];lookup={(r['arm'],r['stage']):r for r in saved['comparison']};rows=[];monthly_curves=[]
    for arm,name in zip(arms,names):
        values={}
        for cost in (0,5,10,25):
            key=f'{arm}/monthly_cost{cost}/monthly';f=pq.read_table(check(monthly[key]),columns=['outcome_month','nav_start','nav_end','net_return']).to_pandas()
            if len(f)!=75 or f.outcome_month.iloc[0]!='2020-02' or f.outcome_month.iloc[-1]!='2026-04':raise ValueError('Full75month monthly reference required')
            before=1.
            for r in f.itertuples():
                if not math.isclose(r.nav_start,before,rel_tol=1e-9,abs_tol=1e-10) or not math.isclose(r.nav_end/r.nav_start-1,r.net_return,rel_tol=1e-9,abs_tol=1e-10):raise ValueError('Monthly cash-flow continuity changed')
                before=r.nav_end
                monthly_curves.append(dict(arm=arm,cost_bps=cost,outcome_month=r.outcome_month,wealth=r.nav_end))
            values[cost]=dict(total_return=float(f.nav_end.iloc[-1]-1),cagr_same_daily_convention=float(f.nav_end.iloc[-1]**(252/1569)-1))
            if cost in (0,10):
                stage='cache_close_gross' if cost==0 else 'cache_close_net';daily=lookup[arm,stage]
                if not math.isclose(values[cost]['total_return'],daily['total_return'],rel_tol=1e-9,abs_tol=1e-10):raise ValueError('Monthly/daily close total mismatch')
                if cost==0:
                    dc=[r for r in saved['curves'] if r['arm']==arm and r['stage']==stage]
                    if len(dc)!=75 or any(r.outcome_month!=d['date'][:7] or not math.isclose(r.nav_end,d['wealth'],rel_tol=1e-9,abs_tol=1e-10) for r,d in zip(f.itertuples(),dc)):raise ValueError('Monthly/daily close curve mismatch')
        rows.append(dict(arm=arm,model=name,monthly_gross=values[0]['total_return'],monthly_net10=values[10]['total_return'],
            daily_same_close_gross=lookup[arm,'cache_close_gross']['total_return'],daily_same_close_net10=lookup[arm,'cache_close_net']['total_return'],
            daily_next_open_gross=lookup[arm,'execution_open_gross']['total_return'],daily_next_open_net10=lookup[arm,'execution_open_net']['total_return'],daily_next_open_net10_stop=lookup[arm,'execution_open_net_stop']['total_return'],
            daily_net10_cagr=lookup[arm,'execution_open_net']['cagr'],daily_net10_stop_cagr=lookup[arm,'execution_open_net_stop']['cagr']))
    body=dict(status='VERIFIED_MONTHLY_DAILY_DISPLAY',period='February 2020–April 2026',months=75,sessions=1569,start_date='2020-02-03',end_date='2026-04-30',initial_nav_date='2020-01-31',comparison=rows,
        monthly_curves=monthly_curves,daily=saved,monthly_reference_definition='Realized adjusted-close monthly portfolio returns, with approved execution-only cash/share entitlements for missing delisting endpoints. RET is the selection score, not this realized return.',
        total_return_units='Cumulative compounded total over75months; CAGR separately annualized with252/1569 for comparable tables.',new_fits=0,new_backtests=0)
    bodypin=save(out/'summary.json',body);vp=save(out/'verification.json',dict(status='PASS_VERIFIED_MONTHLY_AND_DAILY_DISPLAY',summary_pin=bodypin,independent_proof_pin=proofpin,source_pins=source,worker_pin=pin(__file__),monthly_tables_verified=16,models_changed=False,rules_changed=False,new_fits=0,new_backtests=0))
    print(json.dumps(dict(status=body['status'],output=str(out),verification_pin=vp,comparison=rows)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);a=p.parse_args();prepare(a.run,a.output)
