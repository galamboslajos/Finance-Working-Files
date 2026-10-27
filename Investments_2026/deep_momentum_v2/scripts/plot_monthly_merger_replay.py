#!/usr/bin/env python3
"""Plot only independently verified small monthly replay ledgers; zero fits."""
from pathlib import Path
import argparse


def main():
    import numpy as np,pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from dmv2.monthly_merger_reporting import read_monthly_merger_replay
    from dmv2.paper15_baseline_data import save,pin,read
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);a=parser.parse_args()
    root=Path(a.run).resolve();r=read_monthly_merger_replay(root)
    if r['status']!='COMPLETE_VERIFIED':raise ValueError('Verified replay required')
    out=root/'reports/presentation_v1'
    if out.exists():raise ValueError('Preserve existing presentation')
    out.mkdir(parents=True)
    audit=read(root/'backtest/audit.json')
    inputs={k:v for k,v in audit['artifacts'].items() if k.endswith('/full48/cost_10bps/monthly') or k in ('comparison','annual')}
    plan=save(out/'read_plan.json',dict(status='BOUND_VERIFIED_SMALL_REPORTS',parent_proof_pin=pin(root/'verification/backtest.json'),
        input_artifacts=inputs,worker_pin=pin(__file__),new_fits=0,new_backtests=0,financial_2024_plus_read=False))
    names={'rf_paper15_5y':'RF · 5 years','xgb_paper15_5y':'XGBoost · 5 years',
           'rf_paper15_10y':'RF · 10 years','xgb_paper15_10y':'XGBoost · 10 years'}
    colors=['#176a9d','#d48417','#6e6f73','#2c8b61']
    fig,axes=plt.subplots(1,2,figsize=(13,5.2),gridspec_kw={'width_ratios':[1.2,1]})
    for (arm,label),color in zip(names.items(),colors):
        g=r['monthly'].loc[r['monthly'].arm.eq(arm)].sort_values('outcome_month')
        dates=pd.to_datetime([g.signal_date.iloc[0],*g.signal_date.iloc[1:].tolist(),'2023-12-29'])
        nav=np.r_[1.,g.nav_end.to_numpy(dtype=float)]
        axes[0].plot(dates,nav*100,label=label,color=color,linewidth=2)
        axes[1].plot(dates,(nav/np.maximum.accumulate(nav)-1)*100,color=color,linewidth=1.8)
    axes[0].axhline(100,color='#b7bdc4',linewidth=.8);axes[1].axhline(0,color='#b7bdc4',linewidth=.8)
    axes[0].set_title('Monthly NAV · initial capital = 100',loc='left',fontsize=12)
    axes[1].set_title('Drawdown at month ends',loc='left',fontsize=12)
    axes[0].set_ylabel('Capital');axes[1].set_ylabel('Drawdown (%)')
    for ax in axes:
        ax.grid(axis='y',alpha=.22);ax.spines[['top','right']].set_visible(False)
        ax.xaxis.set_major_locator(mdates.YearLocator());ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.set_xlim(pd.Timestamp('2019-12-31'),pd.Timestamp('2023-12-29'))
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.93),ncol=4,frameon=False)
    fig.suptitle('Paper15 RET · full 2020–2023 monthly comparison',x=.065,y=.995,ha='left',fontsize=15,fontweight='bold')
    fig.text(.065,.025,'48 monthly returns · 10 bp one-way costs · 50% long / 50% short · no stops · borrow/funding = 0',fontsize=9,color='#444c55')
    fig.text(.065,.002,'Realized merger payoffs repaired; original forecasts, training labels and RET selections unchanged. Same-close entry is idealized.',fontsize=8,color='#5b626a')
    fig.subplots_adjust(left=.065,right=.985,top=.79,bottom=.14,wspace=.24)
    path=out/'full48_monthly_nav_drawdown.png';fig.savefig(path,dpi=170,facecolor='white');plt.close(fig)
    manifest=save(out/'manifest.json',dict(status='VERIFIED_AGGREGATE_PRESENTATION',read_plan_pin=plan,
        parent_proof_pin=pin(root/'verification/backtest.json'),figure_pin=pin(path),period='January2020–December2023;48monthlyreturns',
        cost_bps=10,stops=False,drawdown_frequency='monthly',new_fits=0,new_backtests=0,matplotlib_version=matplotlib.__version__))
    print(path);print('Presentation manifest:',manifest['path'])


if __name__=='__main__':main()
