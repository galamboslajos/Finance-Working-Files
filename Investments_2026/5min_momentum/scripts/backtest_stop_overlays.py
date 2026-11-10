"""Three frozen-signal five-minute exit overlays; never starts ML fits.

Private immutable release, exact input fingerprints, prior-year calibration,
quarterly frozen decisions and complete-path conditional comparisons.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
from datetime import datetime, timezone
import hashlib, inspect, json, time
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from build_momentum_features import session_calendar
from ret_backtest import performance
from stop_loss_math import (FAMILIES, prior_context, scale_price, replay,
                           response_segments, calibrate, select_past)

RELEASE = '13_stop_overlays_v1'
RUNS = {
    'ret_initial': '03_xgb_ret_wfo_full_history_v1',
    'ret_validation_only': '03_xgb_ret_wfo_validation_v1',
    'ret_paper15': '04_xgb_ret_ensemble100_v1',
    'ret_network8': '05_xgb_ret_network_features100_uncapped_v1',
    'ret_volume_gap15': '07_xgb_ret_volume_overnight100_v1',
    'ret_joint30': '09_xgb_ret_joint100_v1',
    'ret_rich42': '10_xgb_ret_rich_path100_v1',
    'ret_tuned42': '11_ret_ranker_parallel_v1/ret_tuned',
    'ranker_original42': '11_ret_ranker_parallel_v1/ranker',
    'ranker_corrected42': '12_ranker_portfolio100_v1',
}

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(path, value):
    Path(path).write_text(json.dumps(value,indent=2,default=str,allow_nan=False)+'\n')
def stamp(): return datetime.now(timezone.utc).isoformat()


def scan_prices(private, out, needed, sessions):
    """One monthly scan, only local published five-minute adjusted files."""
    con=duckdb.connect(); con.execute('SET threads=6');con.execute("SET TimeZone='UTC'")
    con.register('sessions',sessions.rename(columns={'trade_date':'signal_date'}))
    con.register('needed',needed)
    daily, selected, inventory = [], [], []
    root=private/'sp500_5min_20261103'
    manifest=json.loads((root/'MANIFEST.json').read_text())
    files=[x for x in manifest['objects'] if '/adjustment=all/' in x['path']]
    if len(files)!=130: raise ValueError('Unexpected price manifest coverage')
    for j,obj in enumerate(files,1):
        p=root/'adjustment=all'/Path(obj['path']).parent.name/Path(obj['path']).name
        digest=sha(p)
        if digest != obj['sha256'].removeprefix('sha256:'):
            raise ValueError('Changed source '+str(p))
        inventory.append(dict(path=str(p.relative_to(private)),sha256=digest,bytes=p.stat().st_size))
        con.read_parquet(str(p),hive_partitioning=False).create_view('bars',replace=True)
        con.execute('''CREATE OR REPLACE TEMP TABLE rth AS
          WITH t AS (SELECT b.security_id,CAST(b.trading_date AS DATE) trade_date,
            try_cast(b.bar_timestamp_utc AS TIMESTAMPTZ) ts,
            try_cast(b.available_at_utc AS TIMESTAMPTZ) available,
            b.open,b.high,b.low,b.close,b.minute_bars,b.source,
            s.market_open_utc,s.market_close_utc,s.expected_slots
            FROM bars b JOIN sessions s ON CAST(b.trading_date AS DATE)=s.signal_date)
          SELECT *,coalesce(isfinite(open) AND isfinite(high) AND isfinite(low)
            AND isfinite(close) AND low>0 AND high>=low AND open BETWEEN low AND high
            AND close BETWEEN low AND high AND ts=date_trunc('second',ts)
            AND epoch(ts-market_open_utc)::BIGINT%300=0
            AND available=ts+INTERVAL '5 minutes' AND minute_bars BETWEEN 1 AND 5
            AND source IS NOT NULL AND source!='',false) valid
          FROM t WHERE ts>=market_open_utc AND ts<market_close_utc''')
        daily.append(con.execute('''SELECT security_id,trade_date,
          count(*) rows,count(DISTINCT ts) slots,max(expected_slots) expected_slots,
          count_if(NOT valid) invalid_rows,count(DISTINCT source) source_count,min(source) source,
          count_if(minute_bars<5) partial_buckets,
          max(open) FILTER(WHERE ts=market_open_utc AND valid) session_open,
          max(close) FILTER(WHERE ts=market_close_utc-INTERVAL '5 minutes' AND valid) session_close,
          max(high) session_high,min(low) session_low,
          min(source) FILTER(WHERE ts=market_open_utc AND valid) opening_source,
          max(available) FILTER(WHERE ts=market_open_utc) opening_available_at
          FROM rth GROUP BY 1,2''').df())
        selected.append(con.execute('''SELECT b.security_id,b.trade_date,
          list(b.open ORDER BY ts) open,list(high ORDER BY ts) high,
          list(low ORDER BY ts) low,list(close ORDER BY ts) close,
          list(ts ORDER BY ts) timestamps
          FROM rth b JOIN needed n USING(security_id,trade_date) GROUP BY 1,2''').df())
        if j==1 or j%12==0 or j==len(files):print(f'Prices {j}/{len(files)} months',flush=True)
    con.close()
    d=pd.concat(daily,ignore_index=True)
    d['path_valid']=(d.rows.eq(d.expected_slots)&d.slots.eq(d.expected_slots)
        &d.invalid_rows.eq(0)&d.source_count.eq(1)&d.source.notna()
        &d.session_open.gt(0)&d.session_close.gt(0))
    d.to_parquet(out/'daily_price_quality.parquet',index=False)
    context=prior_context(d,sessions)
    context.to_parquet(out/'daily_stop_context.parquet',index=False)
    paths=pd.concat(selected,ignore_index=True)
    if paths.duplicated(['security_id','trade_date']).any(): raise ValueError('Duplicate selected paths')
    paths.to_parquet(out/'selected_five_minute_paths.parquet',index=False)
    save(out/'PRICE_INVENTORY.json',inventory)
    return context,paths,inventory


def annotate_positions(positions, context, paths):
    p=positions.merge(context,on=['security_id','trade_date'],how='left',validate='many_to_one')
    # Outcome availability and execution coverage are evaluated after the
    # original selections. They never change forecasts, weights or selections.
    p['exclusion_reason']=''
    tests=[('unknown_original_exit',np.isfinite(p.future_oc_return)),
        ('incomplete_or_invalid_five_minute_path',p.path_valid.fillna(False)),
        ('missing_prior_atr',np.isfinite(p.atr_prior)&p.atr_prior.gt(0)),
        ('undefined_zero_or_missing_gap',np.isfinite(p.gap_price)&p.gap_price.ne(0)),
        ('closing_return_mismatch',np.isclose(p.future_oc_return,p.session_close/p.session_open-1,
                                              rtol=0,atol=1e-10,equal_nan=False))]
    reasons=[]
    for reason,ok in tests:
        bad=~np.asarray(ok,dtype=bool)
        reasons.append(pd.Series(np.where(bad,reason,''),index=p.index))
        p.loc[bad,'exclusion_reason'] += reason+';'
    p['overlay_observed']=p.exclusion_reason.eq('')
    return p


def policy_calibration(past, paths, family, days):
    segments=[]
    for row in past.itertuples():
        direction=int(np.sign(row.weight)); path=paths[(row.security_id,row.trade_date)]
        scale=scale_price(family,row.atr_prior,row.gap_price)/row.session_open
        # Each N receives half the utility and each complete calendar date the
        # same utility. These are the saved weights, without stock replacement.
        segments.append(response_segments(path,direction,scale,row.weight*.5/days,1))
    return calibrate(segments)


def simulate(model, positions, original_daily, context, path_lookup, out):
    p=annotate_positions(positions,context,path_lookup)
    original_daily=original_daily.copy()
    original_daily['trade_date']=pd.to_datetime(original_daily.trade_date)
    # One fixed mask across both N and all distances/families. Cash days remain.
    day_good=p.groupby('trade_date').overlay_observed.all()
    calendar=pd.DatetimeIndex(sorted(original_daily.trade_date.unique()))
    mask=day_good.reindex(calendar,fill_value=True)
    original_ok=original_daily.assign(ok=np.isfinite(original_daily.gross_return)).groupby('trade_date').ok.all()
    mask &= original_ok.reindex(calendar,fill_value=False)
    p.to_parquet(out/f'{model}_selected_coverage.parquet',index=False)
    eligible=p[p.trade_date.isin(calendar[mask])].copy()
    quarters=pd.period_range(calendar.min(),calendar.max(),freq='Q')
    first=calendar.min().to_period('Q').start_time
    policy_rows, daily_rows, trade_rows=[],[],[]
    for quarter in quarters:
        start=quarter.start_time; end=(quarter+1).start_time
        current_dates=calendar[(calendar>=start)&(calendar<end)]
        ready=start>=first+pd.DateOffset(months=12)
        past_dates=calendar[(calendar>=start-pd.DateOffset(months=12))&(calendar<start)&mask]
        past=select_past(eligible,start)
        policies={}
        if ready and not len(past):raise ValueError('No observed prior-year calibration positions')
        for family in FAMILIES:
            choice=policy_calibration(past,path_lookup,family,len(past_dates)) if ready and len(past) else None
            k=choice['multiplier'] if choice else np.nan
            policies[family]=k
            policy_rows.append(dict(model=model,quarter=str(quarter),family=family,
                calibration_start=start-pd.DateOffset(months=12),calibration_end_exclusive=start,
                calibration_observed_days=len(past_dates),calibration_positions=len(past),
                multiplier=float(k) if np.isfinite(k) else None,
                policy_status=('no_stop_selected' if np.isinf(k) else 'active_stop') if ready and choice else 'calibration_warmup',
                calibration_mean_net=choice['score'] if choice else None,
                calibration_baseline_mean_net=choice['baseline'] if choice else None,
                empirical_breakpoints=choice['breakpoints'] if choice else 0))
        current=p[p.trade_date.isin(current_dates)]
        for family in ('no_stop',)+FAMILIES:
            for n in (2,3):
                for date in current_dates:
                    original=original_daily[(original_daily.trade_date==date)&original_daily.n_per_side.eq(n)]
                    if len(original)!=1:raise ValueError('Missing original calendar row')
                    group=current[current.trade_date.eq(date)&current.n_per_side.eq(n)]
                    observed=bool(mask.loc[date]); warmup=not ready
                    record=dict(model=model,trade_date=date,n_per_side=n,approach=family,
                        observation_status='observed' if observed else 'excluded_fixed_path_cohort',
                        calibration_warmup=warmup,selected_positions=len(group),stops=0,
                        gap_throughs=0,winners_stopped=0,partial_buckets=int(group.partial_buckets.fillna(0).sum()),
                        net_exposure_time=np.nan,gross_exposure_time=np.nan,gross_return=np.nan,trade_notional=np.nan)
                    if observed and (family=='no_stop' or ready):
                        contributions=[]; notional=[]; net_exp=[];gross_exp=[]
                        k=np.inf if family=='no_stop' else policies[family]
                        for row in group.itertuples():
                            path=path_lookup[(row.security_id,row.trade_date)]
                            direction=int(np.sign(row.weight))
                            distance=np.inf if np.isinf(k) else k*scale_price(family,row.atr_prior,row.gap_price)
                            result=replay(path,direction,distance)
                            contributions.append(row.weight*result['raw_return'])
                            notional.append(abs(row.weight)*(2+result['raw_return']))
                            # Conservative bar-end duration: exact crossing time
                            # inside the five-minute bucket is unknown.
                            duration=(result['exit_slot']+1)/len(path['open']) if result['stopped'] else 1.
                            net_exp.append(row.weight*duration);gross_exp.append(abs(row.weight)*duration)
                            record['stops']+=int(result['stopped']);record['gap_throughs']+=int(result['gap_through'])
                            record['winners_stopped']+=int(result['stopped'] and direction*row.future_oc_return>0)
                            trade_rows.append(dict(model=model,trade_date=date,security_id=row.security_id,
                                ticker=row.ticker,n_per_side=n,approach=family,weight=row.weight,
                                direction=direction,scale_price=None if family=='no_stop' else scale_price(family,row.atr_prior,row.gap_price),
                                multiplier=float(k) if np.isfinite(k) else None,
                                distance_price=float(distance) if np.isfinite(distance) else None,
                                stop_price=float(row.session_open-direction*distance) if np.isfinite(distance) else None,
                                gap_fraction=row.gap_price/row.session_open,atr_fraction=row.atr_prior/row.session_open,
                                signal_type='continuation' if direction*row.gap_price>0 else 'reversal',
                                original_raw_return=row.future_oc_return,scheduled_slots=len(path['open']),
                                exit_bar_timestamp=path['timestamps'][result['exit_slot']],**result))
                        record.update(gross_return=sum(contributions),trade_notional=sum(notional),
                                      net_exposure_time=sum(net_exp),gross_exposure_time=sum(gross_exp))
                        if family=='no_stop' and not np.isclose(record['gross_return'],original.gross_return.iloc[0],rtol=0,atol=1e-10):
                            raise ValueError('No-stop accounting changed')
                    for cost in (0,1,5,10):record[f'net_return_{cost}bps']=record['gross_return']-cost/10000*record['trade_notional']
                    daily_rows.append(record)
        print(f'{model} {quarter}: {len(past_dates)} past observed days; {len(current_dates)} test days',flush=True)
    daily=pd.DataFrame(daily_rows); trades=pd.DataFrame(trade_rows); policies=pd.DataFrame(policy_rows)
    daily.to_parquet(out/f'{model}_daily.parquet',index=False)
    trades.to_parquet(out/f'{model}_trades.parquet',index=False)
    policies.to_parquet(out/f'{model}_policies.parquet',index=False)
    return daily,policies


def summarize(all_daily, private, out):
    """Conditional within-model comparisons; exact omitted dates retained."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    results=[]; exclusions=[]
    fig,axes=plt.subplots(2,2,figsize=(14,9),sharex=True)
    focus={'ret_rich42':0,'ret_tuned42':1}
    for model,g in all_daily.groupby('model',sort=False):
        observed=g[~g.calibration_warmup].pivot(index='trade_date',columns=['n_per_side','approach'],values='net_return_1bps')
        complete=observed.notna().all(axis=1)
        for date in observed.index[~complete]:exclusions.append(dict(model=model,trade_date=date,reason='fixed_path_or_scale_or_original_outcome_unavailable'))
        dates=observed.index[complete]
        for (n,approach),s in g.groupby(['n_per_side','approach']):
            sample=s[s.trade_date.isin(dates)].sort_values('trade_date')
            if len(sample)<2:continue
            for cost in (0,1,5,10):
                metrics=performance(sample[f'net_return_{cost}bps'])
                values=sample[f'net_return_{cost}bps'].to_numpy()
                tail=values[values<=np.quantile(values,.05)]
                results.append(dict(model=model,n_per_side=int(n),approach=approach,cost_bps=cost,
                    start=str(dates.min().date()),end=str(dates.max().date()),excluded_dates=int((~complete).sum()),
                    stop_rate=float(sample.stops.sum()/sample.selected_positions.sum()),
                    gap_through_rate=float(sample.gap_throughs.sum()/max(1,sample.stops.sum())),
                    winners_cut_fraction=float(sample.winners_stopped.sum()/max(1,sample.stops.sum())),
                    mean_gross_exposure_time=float(sample.gross_exposure_time.mean()),
                    mean_absolute_net_exposure_time=float(sample.net_exposure_time.abs().mean()),
                    worst_day=float(values.min()),expected_shortfall_5pct=float(tail.mean()),**metrics))
            if model in focus:
                ax=axes[n-2,focus[model]]
                ax.plot(sample.trade_date,np.cumprod(1+sample.net_return_1bps),label=approach)
                ax.set_title(f'{model} | {n} per side | conditional dates');ax.grid(alpha=.2);ax.legend()
    fig.suptitle('Frozen selections: stop overlays at 1 bp per side; within-model matched dates')
    fig.tight_layout();fig.savefig(out/'stop_equity_comparison.png',dpi=150);plt.close(fig)
    result=pd.DataFrame(results);result.to_csv(out/'metrics.csv',index=False)
    pd.DataFrame(exclusions).to_parquet(out/'excluded_comparison_dates.parquet',index=False)
    base=result[result.cost_bps.eq(1)].copy()
    fig,axes=plt.subplots(1,2,figsize=(15,7),sharey=True)
    for ax,n in zip(axes,(2,3)):
        x=base[base.n_per_side.eq(n)].pivot(index='model',columns='approach',values='sharpe')
        x.plot.barh(ax=ax);ax.set_title(f'{n} per side | Sharpe at 1 bp/side');ax.grid(axis='x',alpha=.2)
    fig.tight_layout();fig.savefig(out/'all_models_stop_sharpe.png',dpi=150);plt.close(fig)
    table=base[['model','n_per_side','approach','sessions','annual_compound','sharpe','max_drawdown','stop_rate','worst_day','expected_shortfall_5pct']]
    html='''<html><head><meta charset="utf-8"><title>Three stop overlays</title>
      <style>body{font:15px system-ui;margin:32px;max-width:1500px}td,th{padding:6px;text-align:right}img{max-width:100%}h1{font-size:26px}</style></head><body>
      <h1>ATR, overnight gap and hybrid protective stops</h1>
      <p>Research replay, not certified execution. Frozen original selections and initial 50% long/50% short weights. Stops leave cash; no replacements or re-entry. Includes legacy runs with inherited research limitations.</p>
      <p>D = k × ATR(10 prior sessions); k × |opening gap|; or k × max(ATR, |gap|). ATR is a simple mean of true range from ten preceding scheduled sessions. Same adjusted vendor basis as historical price targets; adjustment vintage and retrospective PIT limitations persist.</p>
      <p>Each quarter uses only the prior 12 months of saved out-of-sample trades. Equal-day mean net return at 1 bp/side, with equal utility for N=2 and N=3, chooses a positive empirical adverse-excursion threshold. No-stop wins ties. The first 12 months are calibration warm-up (2017 for full-history runs; 2022 for the bounded validation-only run). No model refits.</p>
      <p>Fixed complete-path, positive-ATR, nonzero-gap date cohort across all four policies and both N within each model. Cash days retained; omitted dates and selected-row reasons exported. Different models can have different observed dates. Curves and annualization are conditional on observed dates; they are not complete investable paths.</p>
      <p>Five-minute stop active from the assumed opening fill, including the first bar; touch exits at barrier, gap-through exits at worse bar open. No additional stop slippage or queue model. Opening bar delivery follows the assumed decision/fill; no execution certification. Partial underlying minute buckets (1–4) are flagged and admitted under the existing five-minute data policy; missing intrabucket trades cannot be ruled out. Independent stops create intraday net exposure; duration diagnostics use bar-end approximations. No one-minute data used.</p>
      <p>Repeatedly inspected historical results are development comparisons, not a new untouched holdout. Calibration is a search over past observations, not evidence that a future optimal stop has been found.</p>'''
    html+=table.to_html(index=False,float_format=lambda x:f'{x:.4f}')
    html+='<img src="stop_equity_comparison.png"><img src="all_models_stop_sharpe.png"></body></html>'
    (out/'STOP_COMPARISON.html').write_text(html)
    return result


def load_price_cache(cache, private):
    cache=Path(cache).resolve()
    if not cache.is_relative_to(private):raise ValueError('Price cache must remain in private data')
    manifest=json.loads((cache/'PRICE_CACHE_MANIFEST.json').read_text())
    if not manifest['complete']:raise ValueError('Incomplete price cache')
    for name,digest in manifest['files_sha256'].items():
        if sha(cache/name)!=digest:raise ValueError('Changed cache: '+name)
    if sha(Path(__file__).with_name('stop_loss_math.py'))!=manifest['context_math_sha256']:
        raise ValueError('Changed context construction')
    if hashlib.sha256(inspect.getsource(scan_prices).encode()).hexdigest()!=manifest['scanner_sha256']:
        raise ValueError('Changed price scanner')
    inventory=json.loads((cache/'PRICE_INVENTORY.json').read_text())
    for obj in inventory:
        if sha(private/obj['path'])!=obj['sha256']:raise ValueError('Changed raw price input')
    return (pd.read_parquet(cache/'daily_stop_context.parquet'),
            pd.read_parquet(cache/'selected_five_minute_paths.parquet'),inventory)


def run(project,price_cache=None):
    started=time.monotonic();project=Path(project).resolve();private=(project/'data').resolve()
    out=private/'backtests'/RELEASE
    if out.exists():raise FileExistsError('Choose a new version; never overwrite a stop release')
    out.mkdir(parents=True)
    source_paths=[Path(__file__),Path(__file__).with_name('stop_loss_math.py'),
        Path(__file__).with_name('build_momentum_features.py'),Path(__file__).with_name('ret_backtest.py')]
    source_at_start={p.name:sha(p) for p in source_paths}
    save(out/'RUN_STATUS.json',dict(status='running',started_at_utc=stamp()))
    try:
        positions={};books={};inputs={};catalog=[]
        for model,relative in RUNS.items():
            root=private/'models'/relative
            for name in ('positions.parquet','portfolio_daily.parquet','CONFIG.json'):
                inputs[str((root/name).relative_to(private))]=sha(root/name)
            p=pq.ParquetFile(root/'positions.parquet').read().to_pandas()
            p['trade_date']=pd.to_datetime(p.trade_date);p['security_id']=p.security_id.astype(str)
            if p.duplicated(['trade_date','security_id','n_per_side']).any():raise ValueError('Duplicate original position')
            if not np.allclose(p.weight.abs(),.5/p.n_per_side):raise ValueError('Changed original weights')
            positions[model]=p[['security_id','ticker','trade_date','n_per_side','weight','future_oc_return']].copy()
            books[model]=pq.ParquetFile(root/'portfolio_daily.parquet').read().to_pandas()
            catalog.append(dict(model=model,run=relative,first_trade=str(p.trade_date.min()),last_trade=str(p.trade_date.max()),
                                original_positions=len(p),qualification='bounded validation experiment' if 'validation' in model else 'historical model release; original limitations retained'))
        needed=pd.concat([p[['security_id','trade_date']] for p in positions.values()]).drop_duplicates()
        print(f'{len(needed)} distinct selected security/days across {len(positions)} versions',flush=True)
        sessions=session_calendar('2016-01-04','2026-10-30').rename(columns={'signal_date':'trade_date'})
        sessions.to_parquet(out/'exchange_calendar.parquet',index=False)
        if price_cache is None:
            context,paths,inventory=scan_prices(private,out,needed,sessions)
        else:
            import shutil
            context,paths,inventory=load_price_cache(price_cache,private)
            for name in ('daily_price_quality.parquet','daily_stop_context.parquet',
                         'selected_five_minute_paths.parquet','PRICE_INVENTORY.json'):
                shutil.copy2(Path(price_cache)/name,out/name)
            save(out/'PRICE_CACHE_PROVENANCE.json',dict(path=str(Path(price_cache).resolve()),
                 manifest_sha256=sha(Path(price_cache)/'PRICE_CACHE_MANIFEST.json')))
            missing=needed.merge(paths[['security_id','trade_date']],how='left',indicator=True)
            # Missing original paths remain unknown, not filled. Ensure every
            # cached key belongs to the current requested union of positions.
            extra=paths[['security_id','trade_date']].merge(needed,how='left',indicator=True)
            if extra._merge.ne('both').any():raise ValueError('Cache belongs to another selected universe')
        lookup={(r.security_id,r.trade_date):{k:getattr(r,k) for k in ('open','high','low','close','timestamps')}
                for r in paths.itertuples()}
        all_daily=[];all_policies=[]
        for model,p in positions.items():
            d,policy=simulate(model,p,books[model],context,lookup,out)
            all_daily.append(d);all_policies.append(policy)
        all_daily=pd.concat(all_daily,ignore_index=True);all_policies=pd.concat(all_policies,ignore_index=True)
        all_daily.to_parquet(out/'portfolio_daily.parquet',index=False)
        all_policies.to_csv(out/'quarterly_stop_policies.csv',index=False)
        summarize(all_daily,private,out)
        # Confirm no mutable input changed during the analysis.
        for name,digest in inputs.items():
            if sha(private/name)!=digest:raise ValueError('Historical run changed during replay: '+name)
        source={p.name:sha(p) for p in source_paths}
        if source!=source_at_start:raise ValueError('Source changed during analysis')
        outputs={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name not in ('MANIFEST.json','RUN_STATUS.json')}
        manifest=dict(complete=True,created_at_utc=stamp(),elapsed_seconds=time.monotonic()-started,
            source_sha256=source,input_sha256=inputs,price_manifest_sha256=sha(private/'sp500_5min_20261103/MANIFEST.json'),
            verified_price_files=len(inventory),outputs_sha256=outputs,model_catalog=catalog,
            policies=list(FAMILIES),model_fits=0,one_minute_data_used=False,
            calibration='Prior 12 calendar months, quarterly freeze, positive empirical first-touch breakpoints; shared N=2/3 mean daily net utility at 1 bp/side; no-stop wins ties',
            primary_sample='Within-model common observed dates across all policies and both N; original cash retained; first 12 months calibration warm-up',
            usage_rights='Licensed vendor-derived outputs: internal Galambos Capital only',
            execution='Optimistic opening knowledge/same-price fill; five-minute touch fill with worse gap-through open; no extra slippage; partial minute buckets flagged, no certified execution')
        save(out/'MANIFEST.json',manifest)
        save(out/'RUN_STATUS.json',dict(status='complete',completed_at_utc=stamp(),elapsed_seconds=time.monotonic()-started))
        print(f'COMPLETE {out} in {time.monotonic()-started:.1f}s',flush=True)
        return out
    except Exception as error:
        save(out/'RUN_STATUS.json',dict(status='failed',updated_at_utc=stamp(),error=repr(error)))
        raise

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[1])
    ap.add_argument('--price-cache',type=Path,default=None,help='Verified private preflight cache, never unpinned prices')
    args=ap.parse_args();run(args.project,args.price_cache)
