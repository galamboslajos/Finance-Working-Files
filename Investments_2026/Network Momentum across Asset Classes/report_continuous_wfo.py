#!/usr/bin/env python3
"""Render an unbroken accounting-scenario curve with disclosed rights assumptions."""
from __future__ import annotations

import argparse
import base64
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_walk_forward import COLORS, PRIMARY
from build_momentum_features import write_json
from evaluate_walk_forward import ARMS
from prepare_sp500_daily import file_hash, require

ROOT = Path(__file__).resolve().parent


def charts(daily, curves, monthly, metrics, envelope, rolling, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.ticker import PercentFormatter

    plt.rcParams.update({'figure.facecolor':'#f7f8fa','axes.facecolor':'white','font.size':10,
        'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold',
        'text.color':'#202b39','axes.labelcolor':'#424c5a','axes.edgecolor':'#cdd4df'})
    figures=[]

    def finish(fig,name,title,subtitle):
        fig.suptitle(title,x=.085,y=.985,ha='left',fontsize=19,fontweight='bold')
        fig.text(.085,.945,subtitle,fontsize=10,color='#505d70')
        fig.text(.085,.018,'Conservative accounting scenario: long rights = 0; short rights = payout cap. Subsequent claim P&L excluded. Before trading costs.',fontsize=8,color='#667286')
        fig.subplots_adjust(top=.87,bottom=.09,left=.085,right=.93 if name=='performance_overview' else .97,hspace=.40,wspace=.32)
        for ax in fig.axes:
            ax.grid(axis='y',alpha=.18)
        for extension in ('png','svg'):
            fig.savefig(output/(name+'.'+extension),dpi=170)
        figures.append((name,fig))

    fig,axes=plt.subplots(2,1,figsize=(14,10),sharex=True)
    for arm in PRIMARY:
        block=curves.loc[curves.strategy.eq(arm)].sort_values('date')
        require(block.segment.nunique()==1 and block.is_initial_capital.sum()==1,'Chart would reset equity')
        band=envelope.loc[envelope.strategy.eq(arm)].sort_values('date')
        axes[0].fill_between(band.date,band.lower_equity,band.upper_equity,color=COLORS[arm],alpha=.2)
        axes[0].plot(block.date,block.equity,color=COLORS[arm],label=arm,lw=1.7)
        axes[1].plot(block.date,block.drawdown,color=COLORS[arm],lw=1.4)
    marked=daily.loc[daily.has_contingent_right_assumption,'return_date']
    for i,date in enumerate(marked):
        for ax in axes:
            ax.axvline(date,color='#b47a19',alpha=.65,lw=.9,ls='--',label='Rights assumption date' if i==0 and ax is axes[0] else None)
    axes[0].axhline(100,color='#8693a7',lw=.6)
    axes[0].set(title='Continuous compounded equity · one initial 100 baseline',ylabel='Scenario equity')
    axes[0].legend(frameon=False,ncols=4,loc='upper left',fontsize=9)
    axes[1].set(title='Continuous drawdown · full-history high-water mark',ylabel='Drawdown',xlabel='Return realization date')
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    finish(fig,'equity_drawdown','Continuous WFO equity and drawdown · accounting scenario',
        '2019–2025 · 1,760 sessions · no breaks or resets · 1,758 observed/recovered days + two explicit contingent-right marks')

    fig,axes=plt.subplots(2,2,figsize=(14,10))
    years=sorted(daily.test_year.unique());x=np.arange(len(years))
    for i,arm in enumerate(PRIMARY):
        values=[metrics.loc[metrics.period.eq(str(y)) & metrics.strategy.eq(arm),'sharpe_complete_dates'].iloc[0] for y in years]
        axes[0,0].bar(x+(i-1)*.24,values,.24,color=COLORS[arm],label=arm)
    axes[0,0].set(xticks=x,xticklabels=years,title='Annual Sharpe · conservative rights scenario',ylabel='Sharpe, zero risk-free rate')
    axes[0,0].legend(frameon=False,ncols=3,fontsize=9)
    grid=monthly.pivot(index='year',columns='month_number',values='GMOM').reindex(index=years,columns=range(1,13))
    require(np.isfinite(grid.to_numpy()).all(),'Monthly chart has a performance gap')
    limit=max(.01,float(np.abs(grid.to_numpy()).max()))
    heat=axes[0,1].imshow(grid,aspect='auto',cmap='RdYlGn',vmin=-limit,vmax=limit)
    axes[0,1].set(title='GMOM monthly returns · all 84 months calculated',xticks=range(12),xticklabels=list('JFMAMJJASOND'),yticks=range(len(years)),yticklabels=years)
    for i in range(len(years)):
        for j in range(12):
            label=f'{100*grid.iloc[i,j]:.1f}'
            if (years[i],j+1) in {(2022,12),(2025,8)}:label+='*'
            axes[0,1].text(j,i,label,ha='center',va='center',fontsize=8)
    fig.colorbar(heat,ax=axes[0,1],format=PercentFormatter(1),fraction=.03,pad=.02)
    for arm in PRIMARY:
        axes[1,0].plot(rolling.return_date,rolling[arm+'_sharpe'],color=COLORS[arm],lw=1,label=arm)
    axes[1,0].set(title='Rolling 63-session Sharpe · all scheduled returns',ylabel='Sharpe',xlabel='Return realization date')
    axes[1,0].legend(frameon=False,ncols=3,fontsize=9)
    axes[1,1].plot(daily.return_date,daily.GMOM_gross_weight,color=COLORS['GMOM'],label='GMOM gross',lw=1.1)
    axes[1,1].plot(daily.return_date,daily.GMOM_net_weight,color=COLORS['LinReg'],label='GMOM net',lw=1.1)
    axes[1,1].set(title='Original intended exposures · unchanged',ylabel='Portfolio weight',xlabel='Return realization date')
    axes[1,1].yaxis.set_major_formatter(PercentFormatter(1));axes[1,1].legend(frameon=False,ncols=2,fontsize=9)
    finish(fig,'performance_overview','Continuous WFO performance · accounting scenario',
        '* December 2022 and August 2025 include the declared rights marks. No month or portfolio day is omitted.')
    with PdfPages(output/'performance_charts.pdf') as pdf:
        for _,fig in figures:pdf.savefig(fig)
    for _,fig in figures:plt.close(fig)
    return [name for name,_ in figures]


def report(results,verification,output):
    require(not output.exists(),'Report output exists; choose a new version')
    manifest=json.loads((results/'manifest.json').read_text());proof=json.loads(verification.read_text())
    require(manifest['status']=='complete' and proof['status']=='passed'
            and proof['result_manifest_sha256']==file_hash(results/'manifest.json'),'Verified scenario required')
    for item in manifest['outputs']:
        require(file_hash(results/item['path'])==item['sha256'],'Scenario output changed')
    daily=pd.read_csv(results/'daily_paper_proxy_returns.csv',parse_dates=['date','return_date'])
    curves=pd.read_csv(results/'equity_drawdown.csv',parse_dates=['date'])
    envelope=pd.read_csv(results/'claim_mark_sensitivity_envelope.csv',parse_dates=['date'])
    metrics=pd.read_csv(results/'paper_proxy_metrics.csv');monthly=pd.read_csv(results/'monthly_returns.csv')
    rolling=pd.read_csv(results/'rolling_63_sessions.csv',parse_dates=['return_date'])
    register=pd.read_csv(results/'claim_register.csv',parse_dates=['date','return_date'])
    sensitivity=pd.read_csv(results/'claim_mark_scenario_metrics.csv')
    require(daily.complete.all() and monthly.complete.all(),'Scenario calculation gaps remain')
    require(curves.groupby('strategy').segment.nunique().eq(1).all(),'Equity chart resets')
    output.mkdir(parents=True)
    names=charts(daily,curves,monthly,metrics,envelope,rolling,output)
    overall=metrics.loc[metrics.period.eq('all')].set_index('strategy')
    view=overall[['total_return','cagr','maximum_drawdown','sharpe_complete_dates','annualized_volatility','sortino_zero_threshold','calmar']].copy()
    for column in ('total_return','cagr','maximum_drawdown','annualized_volatility'):
        view[column]=view[column].map(lambda x:f'{100*x:.2f}%')
    for column in ('sharpe_complete_dates','sortino_zero_threshold','calmar'):
        view[column]=view[column].map(lambda x:f'{x:.3f}')
    view.columns=['Total return*','CAGR*','Max drawdown*','Sharpe*','Ann. volatility*','Sortino*','Calmar*']
    event_view=register.loc[register.strategy.eq('GMOM'),['ticker','return_date','position_weight','right_mark_per_unit','portfolio_return']].copy()
    event_view['return_date']=event_view.return_date.dt.strftime('%Y-%m-%d')
    event_view.position_weight=event_view.position_weight.map(lambda x:f'{100*x:.5f}%')
    event_view.portfolio_return=event_view.portfolio_return.map(lambda x:f'{100*x:.5f}%')
    event_view.right_mark_per_unit=event_view.right_mark_per_unit.map(lambda x:f'${x:.2f}')
    event_view.columns=['Stock','Date','GMOM entry weight','Assumed right mark','GMOM portfolio-day return']
    ranges=sensitivity.groupby('strategy').agg(cagr_low=('cagr','min'),cagr_high=('cagr','max'),
        total_low=('total_return','min'),total_high=('total_return','max'))
    ranges_view=pd.DataFrame({'CAGR range*':ranges.apply(lambda r:f'{100*r.cagr_low:.4f}% – {100*r.cagr_high:.4f}%',axis=1),
        'Total-return range*':ranges.apply(lambda r:f'{100*r.total_low:.4f}% – {100*r.total_high:.4f}%',axis=1)})
    notes=[
        '*All path metrics are conditional on the declared contingent-right accounting scenario, including subsequent claim P&L being zero. Two source valuations are still unobserved; those dates are visibly flagged as assumptions.',
        'Cash consideration is recognized at face value. Long non-traded rights are marked zero; short obligations are reserved at their full contractual payout cap. The rule follows position sign, never whichever mark improves test performance.',
        'The claim register retains the rights/obligations. No right is assumed sold. Later payouts, fair-value changes, reserve releases and financing are excluded; this is not a fully reconciled realized cash-flow ledger.',
        'The four alternative zero/face event-mark combinations use the same exclusion of subsequent claim P&L. Their envelope measures only the two initial-mark assumptions and is not a bound on actual strategy wealth with later cash flows.',
        'All 1,758 observed or previously recovered portfolio days, every forecast, position, cohort and original model are unchanged. No stock label is invented and no whole portfolio return is replaced by zero, dropped or reweighted.',
        'Equity starts at 100 once and compounds all 1,760 scheduled daily returns. Drawdown uses the running high-water mark including initial capital. All 84 monthly returns compound every scheduled session.',
        'CAGR annualizes with 252 trading sessions. Sharpe uses sample daily volatility and zero risk-free rate. Sortino uses a zero return threshold. LongOnly is the inverse-volatility-sized cohort, not a capitalization-weighted index.',
        'The existing research-proxy limits remain: executable fills, trading costs, borrow/financing, source identity certification and complete training-feature repairs are not supplied by this report.',
    ]
    body="<h1>Continuous WFO equity and performance</h1><p class='lead'>S&P 500 · frozen annual WFO · 2019–2025 · conservative contingent-right scenario</p>"
    body+=f"<div class='notice'><strong>{len(daily):,} / {len(daily):,} sessions · {len(monthly)} / {len(monthly)} months · one continuous equity path.</strong><br>1,758 days use observed/recovered returns; two use explicit conservative rights marks. Long rights = $0; short rights = full payout cap. Later claim P&amp;L is excluded. This is an accounting scenario, not newly recovered market data.</div>"
    body+="<h2>Full-period performance under the policy</h2>"+view.to_html(border=0)
    for name in names:
        encoded=base64.b64encode((output/(name+'.png')).read_bytes()).decode()
        body+=f'<img alt="{html.escape(name.replace("_"," "))}" src="data:image/png;base64,{encoded}">'
    body+='<h2>The two assumed GMOM days</h2>'+event_view.to_html(index=False,border=0)
    body+='<p>The cash/right terms come from <a href="https://www.sec.gov/Archives/edgar/data/200406/000119312522311072/d428734dex991.htm">the Abiomed completion announcement</a> and <a href="https://classic.nasdaqtrader.com/TraderNews.aspx?id=ECA2025-463">Nasdaq’s Walgreens notice</a>.</p>'
    body+='<h2>Event-mark sensitivity</h2>'+ranges_view.to_html(border=0)
    body+='<p>The narrow shaded equity envelopes vary both event marks from zero to their contractual maxima. They share the same assumption of zero subsequent claim P&amp;L.</p>'
    body+='<h2>Definitions, verification and limits</h2><ul>'+''.join('<li>'+html.escape(note)+'</li>' for note in notes)+'</ul>'
    body+=f"<p>Independent verification replayed 12 assumed strategy/event returns from the original holdings, checked every monthly return, and confirmed a single uninterrupted path for each strategy. Maximum event-return error: {proof['maximum_assumed_return_error']:.3g}.</p>"
    document="<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Continuous Network Momentum WFO</title><style>body{font:16px/1.55 system-ui,sans-serif;color:#202b39;background:#f7f8fa;max-width:1300px;margin:35px auto;padding:0 25px}h1{font-size:36px;margin-bottom:4px}.lead{color:#596480}.notice{background:#fff3d8;border-left:4px solid #ba7c10;padding:18px;margin:25px 0}table{width:100%;border-collapse:collapse;background:white;font-size:13px}th,td{padding:10px;text-align:right;border-bottom:1px solid #e5e8ef}th:first-child{text-align:left}img{width:100%;height:auto;margin:24px 0}li{margin:12px 0}@media(max-width:800px){body{padding:0 10px}table{font-size:10px}th,td{padding:5px}}</style><main>"+body+'</main></html>'
    (output/'performance_report.html').write_text(document)
    lines=['# Continuous WFO accounting scenario','',
        'The latest report compounds all **1,760 sessions and 84 months** without equity resets. Two unquoted contingent-right valuations use a disclosed conservative convention; they have not become observed prices.','',
        'Long rights are marked zero; short rights are reserved at the full contractual payout cap. Cash consideration is recognized. No subsequent claim payouts, financing, revaluations or reserve releases are credited. Rights are retained in a separate register and no sale is assumed.','',
        '| Strategy | Total return* | CAGR* | Max drawdown* | Sharpe* |','| --- | ---: | ---: | ---: | ---: |']
    for arm in ARMS:
        row=view.loc[arm];lines.append('| '+' | '.join([arm,*row.iloc[:4].tolist()])+' |')
    lines+=['','*Conditional accounting-scenario metrics, before trading costs. The sensitivity below varies only the two initial rights marks and shares the assumption of zero subsequent claim P&L.','',
        '| Stock | Assumed date | GMOM right mark | GMOM portfolio-day return |','| --- | --- | ---: | ---: |']
    for row in register.loc[register.strategy.eq('GMOM')].itertuples():
        lines.append(f'| {row.ticker} | {row.return_date.date()} | ${row.right_mark_per_unit:.2f} | {100*row.portfolio_return:.5f}% |')
    gm=ranges.loc['GMOM'];lines+=['',f"GMOM CAGR across all four event-mark combinations is **{100*gm.cagr_low:.4f}%–{100*gm.cagr_high:.4f}%**. These are not bounds on actual wealth with later rights cash flows.",'',
        'The other 1,758 dates, all portfolio weights, forecast cohorts, features and fitted models are unchanged. The two raw stock labels remain unavailable in the source audit.','',
        '[Review the continuous curves and tables](../outputs/continuous_wfo/README.md). The [earlier valuation audit](WFO_VALUATION_REPAIRS.md) preserves the two source gaps and the preceding 60 recoveries.','',
        *[note+'\n' for note in notes], '## Reproduce','', '```bash','python continuous_walk_forward.py','python verify_continuous_wfo.py','python report_continuous_wfo.py','python export_continuous_wfo.py','```','',
        'Existing versioned output directories are preserved; choose fresh paths when rebuilding. Viewing the compact export does not require raw data, but independent reconstruction requires the pinned local sources.','']
    (ROOT/'docs/CONTINUOUS_WFO_PERFORMANCE.md').write_text('\n'.join(lines))
    record={'status':'complete','purpose':'Continuous accounting-scenario equity and drawdown; no observed data fabrication',
        'results_manifest_sha256':file_hash(results/'manifest.json'),'verification_sha256':file_hash(verification),
        'script_sha256':file_hash(__file__),'sessions':len(daily),'calculated_months':len(monthly),'equity_runs':1,
        'assumption_dates':daily.loc[daily.has_contingent_right_assumption,'return_date'].dt.strftime('%Y-%m-%d').tolist(),
        'plots':names,'outputs':[{'path':p.name,'sha256':file_hash(p),'size_bytes':p.stat().st_size} for p in sorted(output.iterdir()) if p.is_file()]}
    write_json(output/'manifest.json',record);print(json.dumps({k:v for k,v in record.items() if k!='outputs'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=ROOT/'data/walk_forward/continuous_results_v1')
    p.add_argument('--verification',type=Path,default=ROOT/'data/walk_forward/continuous_verification_v1/report.json')
    p.add_argument('--output',type=Path,default=ROOT/'data/walk_forward/continuous_performance_v2')
    a=p.parse_args();report(a.results.resolve(),a.verification.resolve(),a.output.resolve())
