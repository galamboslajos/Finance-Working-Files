#!/usr/bin/env python3
"""Bounded post-verification display export; no source panels/models/replay."""
from pathlib import Path
import argparse,json,math,hashlib
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


def pin(path):
    p=Path(path).resolve();return dict(path=str(p),size_bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def check(v):
    if pin(v['path'])!=v:raise ValueError('Changed independent artifact')
    return Path(v['path'])


def read(p):return json.loads(Path(p).read_text())


def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    return pin(p)


def records(f):return [{k:None if isinstance(v,float) and math.isnan(v) else v for k,v in row.items()} for row in f.to_dict('records')]


def build(root):
    root=Path(root).resolve();ledger=read(root/'ledger.json')
    if ledger['status']!='COMPLETE_VERIFIED':raise ValueError('Verification must finish first')
    phase=next(x for x in ledger['completed'] if x['key']=='verify_replay');proofpin=read(check(phase['receipt']))['result'];proof=read(check(proofpin))
    if proof['status']!='PASS' or proof['verified_scenarios']!=48:raise ValueError('Incomplete replay proof')
    check(proof['plan_pin']);audit=read(check(proof['audit_pin']));arts=audit['artifacts'];out=root/'report'
    if out.exists():raise ValueError('Preserve report attempt')
    out.mkdir();sources=dict(proof['report_artifacts']);comp=pq.read_table(check(sources['comparison'])).to_pandas();annual=pq.read_table(check(sources['annual'])).to_pandas();rec=pq.read_table(check(sources['reconciliation'])).to_pandas()
    stages=['cache_close_gross','execution_open_gross','execution_open_net','execution_open_net_stop'];arms=['rf_paper15_5y','xgb_paper15_5y','rf_paper15_10y','xgb_paper15_10y']
    if len(comp)!=48 or set(comp.arm)!=set(arms):raise ValueError('Comparison denominator changed')
    curves=[];daily={}
    for arm in arms:
        for stage in stages:
            key=f'{arm}/{stage}/close_wealth';sources[key]=arts[key];f=pq.read_table(check(sources[key]),columns=['date','wealth']).to_pandas()
            row=comp.loc[comp.arm.eq(arm)&comp.stage.eq(stage)].iloc[0]
            if not row.complete_path:continue
            if len(f)!=1569 or str(f.date.iloc[-1])[:10]!='2026-04-30':raise ValueError('Truncated displayed curve')
            if not math.isclose(f.wealth.iloc[-1]-1,row.total_return,rel_tol=1e-9,abs_tol=1e-10):raise ValueError('Curve/metric mismatch')
            ends=f.groupby(f.date.dt.strftime('%Y-%m'),sort=True).tail(1)
            if len(ends)!=75:raise ValueError('75month display required')
            curves.extend(dict(arm=arm,stage=stage,date=str(r.date)[:10],wealth=float(r.wealth)) for r in ends.itertuples())
            daily[arm,stage]=f
    def label(a):return ('RF' if a.startswith('rf') else 'XGBoost')+' · '+('5-year' if a.endswith('_5y') else '10-year')
    colors=['#186a90','#df8d28','#499c77','#a44f90'];titles=['Same-close gross: monthly/daily control','Next-open gross, no stop','Next-open net: 10 bp, no stop','Next-open net: 10 bp + 10% asset stop']
    fig,axes=plt.subplots(2,2,figsize=(13,8),sharex=True,sharey=True,layout='constrained')
    for ax,stage,title in zip(axes.flat,stages,titles):
        for arm,color in zip(arms,colors):
            f=daily.get((arm,stage));row=comp.loc[comp.arm.eq(arm)&comp.stage.eq(stage)].iloc[0]
            if f is None:ax.plot([],[],color=color,label=label(arm)+' WITHHELD');continue
            ends=f.groupby(f.date.dt.strftime('%Y-%m')).tail(1)
            x=[pd.Timestamp('2020-01-31'),*ends.date];y=[1.,*ends.wealth]
            ax.plot(x,y,color=color,lw=1.8,label=f'{label(arm)}: total {row.total_return:+.1%}; CAGR {row.cagr:+.1%}')
        ax.set_title(title,fontsize=11,loc='left');ax.axhline(1,color='#aab5bf',lw=.7);ax.axvline(pd.Timestamp('2023-12-29'),color='#9aa9b7',ls=':',lw=1)
        ax.grid(axis='y',alpha=.2);ax.legend(loc='best',fontsize=7.8,frameon=False);ax.set_ylabel('Month-end wealth per $1 starting NAV')
        ax.spines[['right','top']].set_visible(False)
    fig.suptitle('Paper15 fixed RET baskets · February 2020–April 2026\n75 months / 1,569 sessions · cumulative wealth; CAGR separately labelled',fontsize=14,x=.04,ha='left')
    equity=out/'equity_comparison.png';fig.savefig(equity,dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,4),layout='constrained')
    for arm,color in zip(arms,colors):
        f=daily.get((arm,'execution_open_net_stop'))
        if f is None:continue
        import numpy as np
        wealth=np.r_[1.,f.wealth.to_numpy()];dd=(wealth/np.maximum.accumulate(wealth)-1)[1:]
        ax.plot(f.date,dd,color=color,lw=1.2,label=label(arm))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1));ax.legend(frameon=False,ncol=4);ax.grid(axis='y',alpha=.2);ax.spines[['right','top']].set_visible(False)
    ax.set_title('Daily drawdown · next-open net 10 bp + 10% asset stop\n3 February 2020–30 April 2026; same fixed baskets',loc='left')
    drawdown=out/'daily_drawdown.png';fig.savefig(drawdown,dpi=180);plt.close(fig)
    summary=dict(first_outcome='2020-02',last_outcome='2026-04',start_date='2020-02-03',end_date='2026-04-30',initial_nav_date='2020-01-31',months=75,sessions=1569,
      complete_paths=proof['complete_paths'],withheld_paths=proof['withheld_paths'],comparison=records(comp),annual=records(annual),curves=curves,
      same_cache_same_close_maximum_nav_difference=float(rec.loc[rec.stage.isin(['cache_close_gross','cache_close_net']),'absolute_nav_difference'].abs().max()),
      old_prefix_paths_unchanged=proof['old_prefix_paths_unchanged'],monthly_references_verified=16,
      return_units='Total return is cumulative compounded over75months. CAGR annualized with252/1569daily convention. Sharpe/volatility use daily returns and zero cash rate; drawdown uses daily closes.',
      limits=['Provisional source identity/dated membership universe; source knowledge-time is not a fully certified historical release archive.',
       'Strict monthly labels/RET payoffs remain frozen; cash/share legal entitlements repair execution only.',
       'Adjusted OHLC/distribution proxy and reviewed action set; zero borrow, financing and cash interest. Cash settlement at effective open is an economic entitlement proxy.',
       'Stops assume observed gap-open or idealized intraday threshold fills; no guarantee of attainable fill, liquidity or stock borrow.',
       'Same-close benchmark assumes immediate close execution. Terminal holdings marked without forced liquidation. No year resets or target/selection changes.'])
    body=save(out/'summary.json',summary)
    figures={'equity_comparison':pin(equity),'daily_drawdown':pin(drawdown)}
    proofout=save(out/'verification.json',dict(status='PASS_VERIFIED_AGGREGATES_AND_CURVES',replay_proof_pin=proofpin,ledger_pin=pin(root/'ledger.json'),worker_pin=pin(__file__),report_pin=body,source_report_pins=sources,figures=figures))
    print(json.dumps(dict(status='REPORT_VERIFIED',complete_paths=proof['complete_paths'],withheld_paths=proof['withheld_paths'],report_pin=body,verification_pin=proofout,figures=figures)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);build(p.parse_args().run)
