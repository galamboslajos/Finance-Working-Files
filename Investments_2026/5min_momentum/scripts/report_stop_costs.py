"""Reprice frozen stop overlays at the standard cost; no calibration or fits."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse,json,hashlib,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
from ret_backtest import performance
from trading_costs import STANDARD_COST_BPS,net_after_costs,round_trip_notional

PARENT='13_stop_overlays_v1'
REPORT='13_stop_overlays_10bps_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_json(p,value):p.write_text(json.dumps(value,indent=2,default=str,allow_nan=False)+'\n')


def summarize(daily, shared_dates=None):
    results=[];excluded=[]
    for model,frame in daily.groupby('model',sort=False):
        p=frame[~frame.calibration_warmup].pivot(index='trade_date',columns=['n_per_side','approach'],values='standard_net_return')
        complete=p.notna().all(axis=1)
        dates=p.index[complete]
        if shared_dates is not None:dates=dates.intersection(shared_dates)
        for date in p.index.difference(dates):excluded.append(dict(model=model,trade_date=date))
        for (n,approach),g in frame[frame.trade_date.isin(dates)].groupby(['n_per_side','approach']):
            if len(g)<2:continue
            g=g.sort_values('trade_date');r=g.standard_net_return.to_numpy()
            results.append(dict(model=model,n_per_side=int(n),approach=approach,cost_bps=STANDARD_COST_BPS,
                excluded_dates=len(p)-len(dates),start=str(dates.min().date()),end=str(dates.max().date()),
                mean_cost_bps=float(g.standard_cost_return.mean()*10000),
                mean_gross_bps=float(g.gross_return.mean()*10000),
                mean_net_bps=float(r.mean()*10000),
                stop_rate=float(g.stops.sum()/g.selected_positions.sum()),
                worst_day=float(r.min()),expected_shortfall_5pct=float(r[r<=np.quantile(r,.05)].mean()),
                **performance(r)))
    return pd.DataFrame(results),pd.DataFrame(excluded)


def run(project):
    started=time.monotonic();project=Path(project).resolve();private=(project/'data').resolve()
    parent=private/'backtests'/PARENT;out=private/'reports'/REPORT
    if out.exists():raise FileExistsError('Never overwrite a completed cost report; choose a new version')
    manifest=json.loads((parent/'MANIFEST.json').read_text())
    if not manifest['complete']:raise ValueError('Completed parent stop release required')
    required=['portfolio_daily.parquet','metrics.csv','quarterly_stop_policies.csv','excluded_comparison_dates.parquet']
    required += [f"{r['model']}_trades.parquet" for r in manifest['model_catalog']]
    inputs={name:sha(parent/name) for name in required}
    for name,digest in inputs.items():
        if manifest['outputs_sha256'][name]!=digest:raise ValueError('Changed frozen input: '+name)
    sources={p.name:sha(p) for p in [Path(__file__),Path(__file__).with_name('trading_costs.py'),Path(__file__).with_name('ret_backtest.py')]}
    parent_manifest_sha=sha(parent/'MANIFEST.json')
    out.mkdir(parents=True)
    os.environ['MPLCONFIGDIR']=str(private/'reports/13_stop_overlays/.matplotlib')
    d=pd.read_parquet(parent/'portfolio_daily.parquet')
    d['standard_cost_bps']=STANDARD_COST_BPS
    d['standard_cost_return']=d.trade_notional*STANDARD_COST_BPS/10000
    d['standard_net_return']=net_after_costs(d.gross_return,d.trade_notional)
    checks=[]
    def check(name,ok,**details):
        checks.append(dict(check=name,passed=bool(ok),**details))
        if not ok:raise AssertionError((name,details))
    check('standard cost equals frozen 10bp sensitivity',np.allclose(d.standard_net_return,d.net_return_10bps,equal_nan=True,rtol=0,atol=1e-14))
    check('unknown outcomes preserved',np.array_equal(d.standard_net_return.isna(),d.gross_return.isna()|d.trade_notional.isna()))
    cash=d.selected_positions.eq(0)&d.gross_return.notna()
    check('cash has no execution cost',d.loc[cash,'standard_cost_return'].eq(0).all())
    check('1 to 10bp cost difference uses actual notional',np.allclose(d.net_return_1bps-d.standard_net_return,.0009*d.trade_notional,equal_nan=True,rtol=0,atol=1e-14))
    # Reconstruct execution notional independently from the saved trade exits.
    for item in manifest['model_catalog']:
        model=item['model'];t=pd.read_parquet(parent/f'{model}_trades.parquet',columns=['trade_date','n_per_side','approach','weight','raw_return'])
        t['notional']=round_trip_notional(t.weight,t.raw_return)
        t['gross']=t.weight*t.raw_return
        grouped=t.groupby(['trade_date','n_per_side','approach'])[['notional','gross']].sum().reset_index()
        joined=d[d.model.eq(model)].merge(grouped,on=['trade_date','n_per_side','approach'],validate='one_to_one')
        check(model+' saved exits reproduce gross and notional',
            np.allclose(joined.notional,joined.trade_notional,rtol=0,atol=1e-12)
            and np.allclose(joined.gross,joined.gross_return,rtol=0,atol=1e-12))
    metrics,excluded=summarize(d)
    previous=pd.read_csv(parent/'metrics.csv').query('cost_bps == 10')
    joined=metrics.merge(previous,on=['model','n_per_side','approach'],suffixes=('_new','_old'),validate='one_to_one')
    check('80 recomputed portfolios equal prior 10bp sensitivity',len(joined)==80 and all(np.allclose(joined[c+'_new'],joined[c+'_old'],rtol=0,atol=1e-11) for c in ('annual_compound','sharpe','max_drawdown','sessions')))
    old_excluded=pd.read_parquet(parent/'excluded_comparison_dates.parquet')
    keys=['model','trade_date']
    check('exact original conditional date exclusions',np.array_equal(excluded.sort_values(keys).to_numpy(dtype=object),old_excluded[keys].sort_values(keys).to_numpy(dtype=object)))
    d.to_parquet(out/'portfolio_daily.parquet',index=False)
    metrics.to_csv(out/'metrics.csv',index=False);excluded.to_parquet(out/'excluded_comparison_dates.parquet',index=False)
    full=d[~d.calibration_warmup&d.model.ne('ret_validation_only')]
    wide=full.pivot(index='trade_date',columns=['model','n_per_side','approach'],values='standard_net_return')
    dates=wide.index[wide.notna().all(axis=1)]
    common,_=summarize(full,dates);common.to_csv(out/'nine_models_shared_dates_metrics.csv',index=False)
    pd.DataFrame(dict(trade_date=wide.index,observed_for_all_nine=wide.notna().all(axis=1).to_numpy())).to_parquet(out/'nine_models_shared_date_ledger.parquet',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(14,9),sharex=True)
    for col,model in enumerate(('ret_rich42','ret_tuned42')):
        for n in (2,3):
            frame=d[d.model.eq(model)&d.n_per_side.eq(n)&~d.calibration_warmup]
            p=frame.pivot(index='trade_date',columns='approach',values='standard_net_return').dropna()
            for approach in ('no_stop','atr','gap','hybrid'):
                axes[n-2,col].plot(p.index,(1+p[approach]).cumprod(),label=approach)
            axes[n-2,col].set_title(f'{model}: {n} per side');axes[n-2,col].grid(alpha=.2);axes[n-2,col].legend()
    fig.suptitle('10 bp per entry/exit side; frozen stops; within-model observed dates')
    fig.tight_layout();fig.savefig(out/'stop_equity_comparison.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(15,7),sharey=True)
    for ax,n in zip(axes,(2,3)):
        metrics[metrics.n_per_side.eq(n)].pivot(index='model',columns='approach',values='sharpe').plot.barh(ax=ax)
        ax.set_title(f'{n} per side | 10 bp per entry/exit side');ax.grid(axis='x',alpha=.2)
    fig.tight_layout();fig.savefig(out/'all_models_stop_sharpe.png',dpi=150);plt.close(fig)
    html='''<html><head><meta charset="utf-8"><title>Standard 10bp costs</title><style>body{font:15px system-ui;margin:32px;max-width:1500px}td,th{padding:6px;text-align:right}img{max-width:100%}</style></head><body>
    <h1>Stop comparison: standard 10 bp per entry/exit side</h1>
    <p>Net return = gross return − 0.001 × (entry dollar notional + actual exit dollar notional), relative to initial equity. At 100% gross exposure, a flat round trip costs 20 bp of equity. Both long and short executions pay costs; stops do not remove the exit charge. Cash has zero execution cost. Unknown exits remain unresolved.</p>
    <p>The original signals, selected stocks, weights, stop multipliers, fills, date masks and warm-up are frozen. Stops were calibrated at the original 1 bp/side assumption; this report applies 10 bp/side without recalibration or ML fits. Completed parent results remain unchanged.</p>
    <p>Within-model dates are matched across both N and all approaches. Cash dates remain; omitted dates are exported. Different models may have different dates. The separate nine-model common-date table permits direct model comparisons. The bounded validation-only experiment remains separate from that intersection.</p>
    <p>All curves and annualization are conditional on observed dates. Historical development reuse, adjusted-price/PIT limitations, optimistic opening knowledge and fills, partial underlying-minute buckets, first-bar stops and very tight barriers remain. A fixed cost allowance does not certify executable fills or borrow and is not an observed estimate of these components.</p>'''
    columns=['model','n_per_side','approach','sessions','mean_gross_bps','mean_cost_bps','annual_compound','sharpe','max_drawdown']
    html+=metrics[columns].to_html(index=False,float_format=lambda x:f'{x:.4f}')
    html+='<img src="stop_equity_comparison.png"><img src="all_models_stop_sharpe.png"></body></html>'
    (out/'STOP_COMPARISON.html').write_text(html)
    # Fingerprints are rechecked at the end; even historical inputs stay frozen.
    check('parent inputs unchanged throughout reporting',all(sha(parent/f)==digest for f,digest in inputs.items()) and sha(parent/'MANIFEST.json')==parent_manifest_sha)
    check('report sources unchanged',all(sha(Path(__file__).with_name(f))==digest for f,digest in sources.items()))
    write_json(out/'VERIFICATION.json',dict(all_passed=True,checks=checks))
    outputs={p.name:sha(p) for p in out.iterdir() if p.is_file()}
    write_json(out/'MANIFEST.json',dict(complete=True,created_at_utc=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=time.monotonic()-started,standard_cost_bps_per_side=STANDARD_COST_BPS,
        round_trip_flat_cost_bps_at_100pct_gross=20,stop_calibration_cost_bps_per_side=1,
        stop_recalibrations=0,model_fits=0,parent_release=PARENT,parent_manifest_sha256=parent_manifest_sha,
        input_sha256=inputs,source_sha256=sources,outputs_sha256=outputs,shared_nine_model_dates=len(dates),
        usage_rights='Internal Galambos Capital only; inherited licensed vendor-derived outputs',
        primary_sample='Original fixed within-model observed dates; unknown dates/warmup/cash preserved'))
    print(f'COMPLETE {out}; {len(checks)} checks passed; 0 fits; 0 stop recalibrations',flush=True)
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[1])
    run(p.parse_args().project)
