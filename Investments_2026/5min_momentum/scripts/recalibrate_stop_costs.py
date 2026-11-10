"""Past-year cost-specific stop calibration on frozen model selections.

All model parameters, forecasts, original entries/weights, 12/3 WFO dates,
five-minute path rules and conditional observation masks remain unchanged.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
import argparse,hashlib,json,shutil,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
from stop_loss_math import FAMILIES,scale_price,response_segments,calibrate,replay,select_past
from trading_costs import net_after_costs,round_trip_notional
from ret_backtest import performance

COSTS=(5,10,20)
PARENT='13_stop_overlays_v1'
RELEASE='14_stop_cost_recalibration_v1'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def save(path,value):Path(path).write_text(json.dumps(value,indent=2,default=str,allow_nan=False)+'\n')


def cost_segments(gross_segments,direction,absolute_weight,cost_bps):
    """Exact affine conversion: net = gross*(1-c*direction)-2*c*|w|.

    Thresholds and first-touch intervals do not depend on execution cost.
    This avoids reconstructing the same price path for every cost scenario.
    """
    boundaries,m,b=gross_segments;c=cost_bps/10000
    return boundaries,m*(1-c*direction),b*(1-c*direction)-2*c*absolute_weight


def learn_quarter(positions,paths,family,quarter_start,observed_dates,costs=COSTS):
    start=pd.Timestamp(quarter_start)
    dates=pd.DatetimeIndex(observed_dates)
    if not dates.is_unique:raise ValueError('Duplicate calibration dates')
    dates=dates[(dates>=start-pd.DateOffset(months=12))&(dates<start)]
    past=select_past(positions,start)
    past=past[past.trade_date.isin(dates)]
    if not len(past) or not len(dates):raise ValueError('No observed past calibration trades')
    if set(past.n_per_side)!={2,3}:raise ValueError('Both N utilities required')
    segments={cost:[] for cost in costs}
    for row in past.itertuples():
        direction=int(np.sign(row.weight))
        # Half the utility to each N and equal utility to each observed date,
        # including original cash dates in the denominator.
        weight=row.weight*.5/len(dates)
        scale=scale_price(family,row.atr_prior,row.gap_price)/row.session_open
        gross=response_segments(paths[(row.security_id,row.trade_date)],direction,scale,weight,0)
        for cost in costs:
            segments[cost].append(cost_segments(gross,direction,abs(weight),cost))
    return {cost:calibrate(values) for cost,values in segments.items()},len(dates),len(past)


def replay_quarter(current,paths,family,choices,quarter,model):
    """Saved positions only; one fixed stop, cash after exit, no replacement."""
    rows=[];cache={}
    for cost in COSTS:
        k=choices[cost]['multiplier']
        for row in current.itertuples():
            path=paths[(row.security_id,row.trade_date)]
            direction=int(np.sign(row.weight))
            scale=scale_price(family,row.atr_prior,row.gap_price)
            distance=np.inf if np.isinf(k) else k*scale
            key=(row.security_id,row.trade_date,direction,float(k))
            if key not in cache:cache[key]=replay(path,direction,distance)
            result=cache[key]
            duration=(result['exit_slot']+1)/len(path['open']) if result['stopped'] else 1.
            gross=row.weight*result['raw_return']
            notional=float(round_trip_notional(row.weight,result['raw_return']))
            rows.append(dict(model=model,quarter=str(quarter),trade_date=row.trade_date,
                security_id=row.security_id,ticker=row.ticker,n_per_side=int(row.n_per_side),
                approach=family,cost_bps_per_side=cost,weight=row.weight,
                entry_price=row.session_open,entry_timestamp=path['timestamps'][0],
                opening_available_at=row.opening_available_at,
                multiplier=float(k) if np.isfinite(k) else None,
                distance_price=float(distance) if np.isfinite(distance) else None,
                gross_contribution=gross,trade_notional=notional,
                net_contribution=float(net_after_costs(gross,notional,cost)),
                first_bar_stop=int(result['stopped'] and result['exit_slot']==0),
                winners_stopped=int(result['stopped'] and direction*row.future_oc_return>0),
                gross_exposure_time=abs(row.weight)*duration,net_exposure_time=row.weight*duration,
                exit_bar_timestamp=path['timestamps'][result['exit_slot']],**result))
    return pd.DataFrame(rows)


def model_replay(model,positions,parent_daily,old_policies,paths,out,progress):
    parent=parent_daily[parent_daily.model.eq(model)].copy()
    base=parent[parent.approach.eq('no_stop')].copy()
    calendar=pd.DatetimeIndex(sorted(base.trade_date.unique()))
    mask=base.pivot(index='trade_date',columns='n_per_side',values='gross_return').notna().all(axis=1)
    observed_dates=mask.index[mask]
    eligible=positions[positions.trade_date.isin(observed_dates)].copy()
    policies=[];books=[];all_trades=[];checks=[]
    quarters=pd.period_range(calendar.min(),calendar.max(),freq='Q')
    first=quarters[0].start_time
    for q in quarters:
        start=q.start_time;end=(q+1).start_time
        current_base=base[base.trade_date.ge(start)&base.trade_date.lt(end)].copy()
        ready=start>=first+pd.DateOffset(months=12)
        current=eligible[eligible.trade_date.ge(start)&eligible.trade_date.lt(end)]
        for cost in COSTS:
            held=current_base.copy();held['cost_bps_per_side']=cost
            held['net_return']=net_after_costs(held.gross_return,held.trade_notional,cost)
            held['policy_status']='hold_to_close'
            books.append(held)
        for family in FAMILIES:
            if ready:
                # 1bp is a reproduction check, not a new trading scenario.
                choices,days,count=learn_quarter(eligible,paths,family,start,observed_dates,(1,)+COSTS)
                old=old_policies[(old_policies.model==model)&(old_policies.quarter==str(q))&(old_policies.family==family)]
                if len(old)!=1:raise ValueError('Missing original calibration control')
                old=old.iloc[0];control=choices[1]
                same_status=np.isinf(control['multiplier'])==(old.policy_status=='no_stop_selected')
                same_k=(np.isinf(control['multiplier']) or np.isclose(control['multiplier'],old.multiplier,rtol=1e-10,atol=1e-12))
                same_score=abs(control['score']-old.calibration_mean_net)<1e-11
                if not (same_status and same_k and same_score):raise AssertionError(f'Original calibration not reproduced: {model} {q} {family}')
                checks.append(dict(model=model,quarter=str(q),family=family,control_1bp_reproduced=True,
                                   absolute_score_error=float(abs(control['score']-old.calibration_mean_net))))
                trades=replay_quarter(current,paths,family,choices,q,model)
                if len(trades):all_trades.append(trades)
            else:
                choices={};days=count=0;trades=pd.DataFrame()
            for cost in COSTS:
                choice=choices.get(cost);k=choice['multiplier'] if choice else np.nan
                status=('no_stop_selected' if np.isinf(k) else 'active_stop') if ready else 'calibration_warmup'
                policies.append(dict(model=model,quarter=str(q),family=family,cost_bps_per_side=cost,
                    calibration_start=start-pd.DateOffset(months=12),calibration_end_exclusive=start,
                    calibration_observed_days=days,calibration_positions=count,
                    multiplier=float(k) if np.isfinite(k) else None,policy_status=status,
                    calibration_mean_net=choice['score'] if choice else None,
                    calibration_no_stop_mean_net=choice['baseline'] if choice else None,
                    empirical_breakpoints=choice['breakpoints'] if choice else 0))
                book=current_base.drop(columns=[c for c in current_base if c.startswith('net_return_')]).copy()
                book['approach']=family;book['cost_bps_per_side']=cost;book['policy_status']=status
                stats=['gross_return','trade_notional','stops','gap_throughs','winners_stopped',
                       'net_exposure_time','gross_exposure_time','first_bar_stops']
                for name in stats:book[name]=np.nan
                observed=book.trade_date.isin(observed_dates)&ready
                book.loc[observed,stats]=0.
                if ready and len(trades):
                    selected=trades[trades.cost_bps_per_side.eq(cost)]
                    grouped=selected.groupby(['trade_date','n_per_side']).agg(
                        gross_return=('gross_contribution','sum'),trade_notional=('trade_notional','sum'),
                        stops=('stopped','sum'),gap_throughs=('gap_through','sum'),
                        winners_stopped=('winners_stopped','sum'),net_exposure_time=('net_exposure_time','sum'),
                        gross_exposure_time=('gross_exposure_time','sum'),first_bar_stops=('first_bar_stop','sum'))
                    book=book.set_index(['trade_date','n_per_side'])
                    book.loc[grouped.index,stats]=grouped[stats]
                    book=book.reset_index()
                book['net_return']=net_after_costs(book.gross_return,book.trade_notional,cost)
                books.append(book)
        progress(model,str(q),ready)
    daily=pd.concat(books,ignore_index=True)
    trades=pd.concat(all_trades,ignore_index=True)
    daily.to_parquet(out/f'{model}_daily.parquet',index=False)
    trades.to_parquet(out/f'{model}_trades.parquet',index=False)
    pd.DataFrame(policies).to_parquet(out/f'{model}_policies.parquet',index=False)
    return daily,pd.DataFrame(policies),checks


def metrics_for(daily,shared_dates=None):
    rows=[];excluded=[]
    for model,frame in daily.groupby('model',sort=False):
        pivot=frame[~frame.calibration_warmup].pivot(index='trade_date',columns=['n_per_side','approach','cost_bps_per_side'],values='net_return')
        observed=pivot.notna().all(axis=1);dates=pivot.index[observed]
        if shared_dates is not None:dates=dates.intersection(shared_dates)
        excluded.extend(dict(model=model,trade_date=date) for date in pivot.index.difference(dates))
        for (n,approach,cost),g in frame[frame.trade_date.isin(dates)].groupby(['n_per_side','approach','cost_bps_per_side']):
            g=g.sort_values('trade_date');r=g.net_return.to_numpy()
            rows.append(dict(model=model,n_per_side=int(n),approach=approach,cost_bps_per_side=int(cost),
                excluded_dates=len(pivot)-len(dates),start=str(dates.min().date()),end=str(dates.max().date()),
                mean_gross_bps=float(g.gross_return.mean()*10000),mean_cost_bps=float(g.trade_notional.mean()*cost),
                mean_net_bps=float(r.mean()*10000),stop_rate=float(g.stops.sum()/g.selected_positions.sum()),
                first_bar_fraction_of_stops=float(g.first_bar_stops.sum()/max(1,g.stops.sum())),
                mean_gross_exposure_time=float(g.gross_exposure_time.mean()),
                mean_absolute_net_exposure_time=float(g.net_exposure_time.abs().mean()),
                worst_day=float(r.min()),expected_shortfall_5pct=float(r[r<=np.quantile(r,.05)].mean()),**performance(r)))
    return pd.DataFrame(rows),pd.DataFrame(excluded)


def report(out,daily,policies,parent_daily):
    metrics,excluded=metrics_for(daily)
    if len(metrics)!=240:raise AssertionError('Expected 240 model/N/approach/cost metrics')
    metrics.to_csv(out/'metrics.csv',index=False);excluded.to_parquet(out/'excluded_comparison_dates.parquet',index=False)
    frozen=[]
    for cost in COSTS:
        g=parent_daily.copy();g['cost_bps_per_side']=cost
        g['net_return']=net_after_costs(g.gross_return,g.trade_notional,cost)
        frozen.append(g)
    frozen=pd.concat(frozen,ignore_index=True)
    frozen['first_bar_stops']=np.nan
    old,_=metrics_for(frozen)
    comparison=metrics.merge(old,on=['model','n_per_side','approach','cost_bps_per_side'],suffixes=('_recalibrated','_frozen_1bp'),validate='one_to_one')
    comparison['cagr_change']=comparison.annual_compound_recalibrated-comparison.annual_compound_frozen_1bp
    comparison.to_csv(out/'recalibrated_vs_frozen.csv',index=False)
    full=daily[~daily.calibration_warmup&daily.model.ne('ret_validation_only')]
    wide=full.pivot(index='trade_date',columns=['model','n_per_side','approach','cost_bps_per_side'],values='net_return')
    dates=wide.index[wide.notna().all(axis=1)]
    shared,_=metrics_for(full,dates);shared.to_csv(out/'nine_models_shared_dates_metrics.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    colors={'no_stop':'#64748b','atr':'#2563eb','gap':'#d97706','hybrid':'#059669'}
    rich=metrics[metrics.model.eq('ret_rich42')]
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),sharey=True)
    for ax,n in zip(axes,(2,3)):
        for approach,color in colors.items():
            points=rich[rich.n_per_side.eq(n)&rich.approach.eq(approach)].sort_values('cost_bps_per_side')
            ax.plot(points.cost_bps_per_side,points.annual_compound,marker='o',label=approach.replace('_',' '),color=color)
        ax.axhline(0,color='#111827',lw=.8);ax.set_xticks(COSTS);ax.set_xlabel('Calibration and execution cost per side (bp)')
        ax.set_title(f'Richer42: {n} stocks per side');ax.grid(alpha=.2);ax.legend();ax.yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].set_ylabel('Conditional net annualized compound return')
    fig.suptitle('Cost-specific stops | past 12-month calibration, next 3-month freeze | same observed dates')
    fig.tight_layout();fig.savefig(out/'richer42_recalibrated_sensitivity.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(15,8),sharex=True,sharey=True)
    for i,n in enumerate((2,3)):
        for j,cost in enumerate(COSTS):
            ax=axes[i,j];g=daily[daily.model.eq('ret_rich42')&daily.n_per_side.eq(n)&daily.cost_bps_per_side.eq(cost)&~daily.calibration_warmup]
            p=g.pivot(index='trade_date',columns='approach',values='net_return').dropna()
            for approach,color in colors.items():ax.plot(p.index,(1+p[approach]).cumprod(),label=approach.replace('_',' '),color=color)
            ax.set_title(f'{n} per side | calibrated and charged at {cost} bp/side');ax.grid(alpha=.2);ax.legend()
    axes[0,0].set_ylabel('Conditional equity, initial = 1');axes[1,0].set_ylabel('Conditional equity, initial = 1')
    fig.suptitle('Richer42: cost-specific stops on frozen original model entries')
    fig.tight_layout();fig.savefig(out/'richer42_recalibrated_equity.png',dpi=140);plt.close(fig)
    table=rich.pivot(index=['n_per_side','approach'],columns='cost_bps_per_side',values='annual_compound')
    html='''<html><head><meta charset="utf-8"><title>Cost-specific stop calibration</title><style>body{font:15px system-ui;margin:32px;max-width:1500px}td,th{padding:6px;text-align:right}img{max-width:100%}</style></head><body>
    <h1>Stops recalibrated at 5 / 10 / 20 bp per side</h1>
    <p>Each model/family/cost quarter chooses a positive empirical stop multiplier using only the preceding 12 calendar months of original saved trades. Maximize the equal-day average of N=2 and N=3 net returns at that scenario's cost, then freeze for the next three months. The same multiplier is shared by both N and both directions. No-stop remains a hold-to-close candidate and wins ties; it never cancels an original entry. The first year is calibration warm-up. All original ML models, predictions, selections, initial weights, WFO dates and cohort rules are unchanged. No ML fits or one-minute work.</p>
    <p>Entry and actual exit notional each incur the stated per-side cost. Fixed complete-path/ATR/nonzero-gap observation masks and original cash days are inherited exactly. Unknown outcomes remain unresolved, even if an earlier stop could appear to resolve an exit. Within-model conditional dates are matched across all costs, N and approaches. Different models may have different dates; the separate nine-model table uses their intersection, excluding the bounded validation-only experiment.</p>
    <p>Original 1bp-calibrated stops are also evaluated at the same scenario costs. The comparison isolates recalibration, without choosing a winner on forward-quarter returns. One-bp calibration controls reproduce the original policies. In-sample calibration gains need not improve forward returns.</p>
    <p>These remain development comparisons on reused history. Opening-price knowledge and same-price fills, five-minute touch/gap-through fills, very tight barriers, incomplete underlying-minute buckets, adjustment/PIT vintages, and unmeasured quotes/borrow remain. Costs do not certify executable fills. Exposure durations are bar-end approximations. Conditional curves and annualization are not complete investable paths.</p>
    <h2>Richer42 net annualized compound return</h2>'''
    html+=table.to_html(float_format=lambda x:f'{x:.2%}')
    html+='<img src="richer42_recalibrated_sensitivity.png"><img src="richer42_recalibrated_equity.png"><h2>All models, N and costs</h2>'
    html+=metrics[['model','n_per_side','approach','cost_bps_per_side','sessions','annual_compound','sharpe','max_drawdown','stop_rate']].to_html(index=False,float_format=lambda x:f'{x:.4f}')
    html+='</body></html>'; (out/'STOP_RECALIBRATION.html').write_text(html)
    return metrics,excluded,comparison


def run(project):
    started=time.monotonic();project=Path(project).resolve();private=(project/'data').resolve()
    parent=private/'backtests'/PARENT;out=private/'backtests'/RELEASE
    if out.exists():raise FileExistsError('Never overwrite a completed release; choose a new version')
    parent_manifest=json.loads((parent/'MANIFEST.json').read_text())
    if not parent_manifest['complete']:raise ValueError('Completed parent stop release required')
    required=['portfolio_daily.parquet','quarterly_stop_policies.csv','selected_five_minute_paths.parquet','excluded_comparison_dates.parquet']
    required += [f"{item['model']}_selected_coverage.parquet" for item in parent_manifest['model_catalog']]
    inputs={name:sha(parent/name) for name in required}
    for name,digest in inputs.items():
        if digest!=parent_manifest['outputs_sha256'][name]:raise ValueError('Changed parent input '+name)
    historical=parent_manifest['input_sha256']
    for name,digest in historical.items():
        if sha(private/name)!=digest:raise ValueError('Changed historical model artifact '+name)
    source_paths=[Path(__file__)]+[Path(__file__).with_name(f) for f in ('stop_loss_math.py','trading_costs.py','ret_backtest.py')]
    source_hashes={p.name:sha(p) for p in source_paths}
    out.mkdir(parents=True);os.environ['MPLCONFIGDIR']=str(private/'reports/13_stop_overlays/.matplotlib')
    total_books=3*len(COSTS)*sum(len(pd.period_range(pd.Timestamp(x['first_trade']).to_period('Q')+4,pd.Timestamp(x['last_trade']).to_period('Q'),freq='Q')) for x in parent_manifest['model_catalog'])
    completed_books=0;completed_quarters=0
    def progress(model,quarter,ready):
        nonlocal completed_books,completed_quarters
        completed_quarters+=1
        if ready:completed_books+=3*len(COSTS)
        save(out/'RUN_STATUS.json',dict(status='running',updated_at_utc=now(),current_model=model,
            quarter=quarter,completed_model_quarters=completed_quarters,
            completed_calibration_books=completed_books,total_calibration_books=total_books,
            elapsed_seconds=time.monotonic()-started,model_fits=0))
        print(f'{model} {quarter}: {completed_books}/{total_books} cost-specific calibration books',flush=True)
    try:
        save(out/'RUN_STATUS.json',dict(status='running',started_at_utc=now(),total_calibration_books=total_books,model_fits=0))
        frame=pd.read_parquet(parent/'selected_five_minute_paths.parquet')
        paths={(r.security_id,r.trade_date):{k:getattr(r,k) for k in ('open','high','low','close','timestamps')} for r in frame.itertuples()}
        parent_daily=pd.read_parquet(parent/'portfolio_daily.parquet')
        old_policies=pd.read_csv(parent/'quarterly_stop_policies.csv')
        all_daily=[];all_policies=[];controls=[]
        for item in parent_manifest['model_catalog']:
            model=item['model'];positions=pd.read_parquet(parent/f'{model}_selected_coverage.parquet')
            daily,policies,checks=model_replay(model,positions,parent_daily,old_policies,paths,out,progress)
            all_daily.append(daily);all_policies.append(policies);controls.extend(checks)
        daily=pd.concat(all_daily,ignore_index=True);policies=pd.concat(all_policies,ignore_index=True)
        daily.to_parquet(out/'portfolio_daily.parquet',index=False);policies.to_csv(out/'quarterly_stop_policies.csv',index=False)
        metrics,excluded,comparison=report(out,daily,policies,parent_daily)
        old_excluded=pd.read_parquet(parent/'excluded_comparison_dates.parquet')
        keys=['model','trade_date']
        if not np.array_equal(excluded[keys].sort_values(keys).to_numpy(dtype=object),old_excluded[keys].sort_values(keys).to_numpy(dtype=object)):
            raise AssertionError('Original observation exclusions changed')
        for name,digest in inputs.items():
            if sha(parent/name)!=digest:raise AssertionError('Parent modified during recalibration')
        for name,digest in historical.items():
            if sha(private/name)!=digest:raise AssertionError('Historical model modified')
        if source_hashes!={p.name:sha(p) for p in source_paths}:raise AssertionError('Active sources changed')
        if completed_books!=total_books:raise AssertionError('Incomplete cost calibrations')
        save(out/'CALIBRATION_CONTROL.json',dict(all_passed=True,controls=controls))
        outputs={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name not in ('MANIFEST.json','RUN_STATUS.json')}
        save(out/'MANIFEST.json',dict(complete=True,created_at_utc=now(),elapsed_seconds=time.monotonic()-started,
            cost_bps_per_side=list(COSTS),stop_calibration_bps_per_side=list(COSTS),model_fits=0,
            calibration_books=completed_books,original_1bp_controls=len(controls),source_sha256=source_hashes,
            input_sha256=inputs,historical_model_sha256=historical,parent_release=PARENT,
            parent_manifest_sha256=sha(parent/'MANIFEST.json'),outputs_sha256=outputs,
            model_catalog=parent_manifest['model_catalog'],wfo='Unchanged 12 calendar months / 3 forward months',
            objective='Equal-day mean of N=2 and N=3 net portfolio return at the scenario cost; same empirical candidates and no-stop/tie rules',
            observation_sample='Exact inherited conditional date masks; first year warm-up, cash retained',
            usage_rights='Licensed derived outputs: internal Galambos Capital only',one_minute_data_used=False))
        save(out/'RUN_STATUS.json',dict(status='complete',completed_at_utc=now(),completed_calibration_books=completed_books,
            total_calibration_books=total_books,model_fits=0,elapsed_seconds=time.monotonic()-started))
        print(f'COMPLETE {out} in {time.monotonic()-started:.1f}s; {len(controls)} original calibration controls passed',flush=True)
        return out
    except Exception as error:
        save(out/'RUN_STATUS.json',dict(status='failed',updated_at_utc=now(),error=repr(error),model_fits=0));raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[1])
    run(p.parse_args().project)
