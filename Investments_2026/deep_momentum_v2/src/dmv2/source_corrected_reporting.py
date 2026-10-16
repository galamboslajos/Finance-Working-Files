"""Passive, pinned summaries and ledger diagnostics; no fitting or execution.

Input bindings are created only after the corrected workflow's independent
feature/input, model and accounting checks finish. This worker never imports a
model, feature calculator, quote loader or accounting engine.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ARMS=('rf_paper15','xgb_paper15','rf_extended','xgb_extended')
BASKETS=('class_1_10','class_2_9')
VARIANTS=('unstopped','asset_stop_10pct')
COSTS=(0,5,10,25)
KEYS=['arm','basket','variant','cost_bps']
FOCUS=('rf_extended','class_1_10','asset_stop_10pct')
METRICS=['total_return','cagr','maximum_drawdown','annualized_volatility','sharpe_zero_cash_rate']
MAX_BYTES=16*1024**2
INPUT_REUSE_SCOPE='new_root_input_reference_reuse_and_prior_formula_evidence'

def require(ok,msg):
    if not ok: raise ValueError(msg)

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''): h.update(block)
    return h.hexdigest()

def load_json(path):
    path=Path(path);require(path.stat().st_size<=MAX_BYTES,'JSON exceeds bounded size')
    return json.loads(path.read_text())

def pinned(root,pin):
    require({'path','sha256','size_bytes'}<=set(pin),'Artifact pin missing')
    root=Path(root).resolve();path=(root/pin['path']).resolve()
    require(path.is_relative_to(root),'Artifact escapes explicit root')
    require(path.is_file() and path.stat().st_size==pin['size_bytes'],'Artifact size changed')
    require(digest(path)==pin['sha256'].removeprefix('sha256:'),'Artifact bytes changed')
    return path

def preserve(snapshot):
    snap=load_json(snapshot);root=Path(snap['root'])
    for pin in snap['selected_metadata_code_report_files']:pinned(root,pin)
    return {'status':'PASS','selected_files':snap['selected_files_count'],
        'snapshot_sha256':digest(snapshot),'scope':snap['scope_limit']}

def read_report(root,pin,rows):
    path=pinned(root,pin);require(path.stat().st_size<=MAX_BYTES,'Report too large')
    f=pq.ParquetFile(path);require(f.metadata.num_rows==rows,'Report row grid changed')
    return f.read().to_pandas()

def read_ledger(root,pin,max_rows):
    """Check bounded accounting dates before financial ledger conversion."""
    path=pinned(root,pin);require(path.stat().st_size<=MAX_BYTES,'Ledger exceeds bounded byte ceiling')
    f=pq.ParquetFile(path);require(f.metadata.num_rows<=max_rows,'Ledger exceeds bounded row ceiling')
    if not f.metadata.num_rows:return f.read().to_pandas()
    require('date' in f.schema_arrow.names,'Ledger requires explicit date metadata')
    dates=pd.to_datetime(f.read(columns=['date'])['date'].to_pylist())
    require(dates.notna().all() and dates.isin(pd.date_range('2020-01-02','2023-12-29')).all(),'Protected/out-of-horizon ledger date')
    return f.read().to_pandas()

def check_comparison(frame):
    require(set(KEYS+METRICS+['status','complete_proxy_path','expected_sessions'])<=set(frame),'Comparison schema changed')
    expected=set(itertools.product(ARMS,BASKETS,VARIANTS,COSTS))
    require(len(frame)==64 and not frame.duplicated(KEYS).any() and set(map(tuple,frame[KEYS].values))==expected,'Incomplete64-scenario comparison')
    require(frame.expected_sessions.eq(1006).all(),'Trading horizon changed')
    complete=frame.complete_proxy_path.eq(True)
    require(frame.loc[complete,'status'].eq('complete_proxy').all(),'Complete status changed')
    require(frame.loc[~complete,METRICS].isna().all().all(),'Withheld full-period metrics must be null')
    finite_metrics(frame,complete,[name for name in METRICS if name!='sharpe_zero_cash_rate'])
    return frame

def finite_metrics(frame,complete,names):
    require(np.isfinite(frame.loc[complete,names].to_numpy(dtype=float)).all(),'Complete-path metrics absent or nonfinite')
    require(frame.loc[complete,'annualized_volatility'].ge(0).all(),'Negative annualized volatility')
    sharpe=frame.loc[complete,'sharpe_zero_cash_rate'];vol=frame.loc[complete,'annualized_volatility']
    require((sharpe.notna()|vol.eq(0)).all(),'Nonzero-volatility complete path lacks Sharpe')
    require(np.isfinite(sharpe.dropna().to_numpy(dtype=float)).all(),'Nonfinite complete-path Sharpe')

def check_yearly(yearly,comparison):
    annual=['initial_nav','final_nav','total_return','annualized_volatility','sharpe_zero_cash_rate','maximum_drawdown','trading_cost_paid']
    require(set(KEYS+['year','complete_full_period_proxy','expected_sessions',*annual])<=set(yearly),'Annual schema changed')
    require(not yearly.duplicated(KEYS+['year']).any(),'Duplicate annual breakdown')
    require(set(map(tuple,yearly[KEYS+['year']].values))==set(itertools.product(ARMS,BASKETS,VARIANTS,COSTS,[2020,2021,2022,2023])),'Incomplete annual breakdown')
    parent=yearly.merge(comparison[KEYS+['complete_proxy_path']],on=KEYS,how='left',validate='many_to_one')
    require(parent.complete_full_period_proxy.eq(parent.complete_proxy_path).all(),'Annual completion status differs from parent scenario')
    require(parent.expected_sessions.eq(parent.year.map({2020:253,2021:252,2022:251,2023:250})).all(),'Annual session horizon changed')
    complete=parent.complete_proxy_path.eq(True)
    require(parent.loc[~complete,annual].isna().all().all(),'Withheld annual metrics must be null')
    finite_metrics(parent,complete,[name for name in annual if name!='sharpe_zero_cash_rate'])
    return yearly

def saved_tables(root,backtest,audit_pin,accounting_receipt_pin,gate_pins,identity,phase_pins):
    """Read complete saved summaries after exact independent receipt bindings.

    gate_pins contain the already completed feature/input and model verification
    receipts. Their exact pins and identity are required, never an inferred PASS.
    """
    root=Path(root).resolve();bt=(root/backtest).resolve()
    require(bt.is_relative_to(root),'Backtest outside root')
    require(set(gate_pins)=={'feature_verification','model_verification'},'Both upstream independent gates required')
    require(set(phase_pins)=={'features','models'},'Exact completed feature/model audits required')
    raw={name:pin['sha256'].removeprefix('sha256:') for name,pin in phase_pins.items()}
    phases={name:load_json(pinned(root,pin)) for name,pin in phase_pins.items()}
    require(all(p.get('status')=='COMPLETE' and p.get('build_identity_sha256')==identity for p in phases.values()),'Upstream phase audit incomplete or wrong run')
    gates={}
    for role,pin in gate_pins.items():
        j=load_json(pinned(root,pin));require(j.get('status') in {'PASS','PASSED'},role+' not passed')
        require(j.get('build_identity_sha256')==identity,role+' refers to another run')
        require(j.get('root_audit_sha256')==raw['features'],role+' does not bind the exact input audit')
        gates[role]=j
    feature_gate=gates['feature_verification']
    require(feature_gate.get('models_fitted')==0 and feature_gate.get('financial_2024_plus_read') is False,'Feature verifier fit or protected scope must be explicit')
    require(feature_gate.get('verification_scope')==INPUT_REUSE_SCOPE,'Exact new-root input-reuse verification scope required')
    model_gate=gates['model_verification']
    require(model_gate.get('annual_fits_reconciled')==16 and model_gate.get('preprocessor_states_reconciled')==8,'All 16 annual fits / eight preprocessing states must be independently reconciled')
    require(model_gate.get('models_audit_sha256')==raw['models'],'Model verifier does not bind the exact completed audit')
    require(model_gate.get('model_identity_sha256')==phases['models'].get('model_identity_sha256') and model_gate.get('model_identity_sha256') is not None,'Model verifier phase changed')
    require(model_gate.get('models_fitted_by_verifier')==0 and model_gate.get('preprocessors_fitted_by_verifier')==0 and model_gate.get('financial_2024_plus_decoded') is False,'Model verifier fit or protected scope changed')
    audit_path=pinned(root,audit_pin);require(audit_path==bt/'audit.json','Backtest audit redirect')
    audit=load_json(audit_path);v=load_json(pinned(root,accounting_receipt_pin))
    require(audit.get('status')=='COMPLETE' and audit.get('build_identity_sha256')==identity,'Backtest incomplete or wrong identity')
    require(audit.get('fresh_source_replay')=='fresh_source_corrected_walkforward_v1' and audit.get('old_completed_reference_preserved') is True,'Corrected replay provenance absent')
    require(audit.get('scenarios')==64 and audit.get('sessions')==1006 and audit.get('outcome_months')==48,'Backtest scope changed')
    require(audit.get('annual_liquidation') is False and audit.get('financial_2024_plus_read') is False,'Chronology changed')
    require(v.get('status') in {'PASS','PASSED'} and v.get('build_identity_sha256')==identity,'Accounting verification not passed for this run')
    require(v.get('backtest_audit_sha256')==audit_pin['sha256'].removeprefix('sha256:'),'Accounting receipt does not bind this exact audit')
    require(v.get('root_audit_sha256')==raw['features'],'Accounting receipt does not bind the exact input audit')
    require(v.get('backtest_phase_identity_sha256')==audit.get('phase_identity_sha256') and v.get('backtest_phase_identity_sha256') is not None,'Accounting receipt phase changed')
    require(v.get('scenarios')==64 and v.get('sessions')==1006 and v.get('outcome_months')==48,'Accounting verification scope changed')
    require(v.get('models_fitted')==0 and v.get('preprocessors_fitted')==0,'Accounting verifier must perform no fits')
    require(v.get('source_pins_unchanged') is True and v.get('financial_2024_plus_read') is False,'Accounting source/protected scope changed')
    if 'backtest_directory' in v:require((root/v['backtest_directory']).resolve()==bt,'Accounting receipt redirects the backtest directory')
    comparison=check_comparison(read_report(bt,audit['artifacts']['comparison.parquet'],64))
    yearly=read_report(bt,audit['artifacts']['yearly.parquet'],256)
    return comparison,check_yearly(yearly,comparison),audit

def comparative(old,new):
    check_comparison(old);check_comparison(new)
    result=old.merge(new,on=KEYS,how='outer',validate='one_to_one',suffixes=('_obsolete_reference','_corrected'))
    result['paired_full_period_available']=result.complete_proxy_path_obsolete_reference&result.complete_proxy_path_corrected
    # Never rank the complete subset. Delta exists only for the identical arm,
    # basket, stop and cost path with full-period metrics on both source versions.
    for name in METRICS:
        result[name+'_change']=np.where(result.paired_full_period_available,
            result[name+'_corrected']-result[name+'_obsolete_reference'],np.nan)
    return result

def close_nav(daily,initial_nav):
    d=daily.copy().reset_index(drop=True);d['date']=pd.to_datetime(d.date)
    require(d.date.between('2020-01-02','2023-12-29').all(),'Ledger date exceeds2020–2023')
    require(not d.duplicated(['date','mark']).any() and d.date.is_monotonic_increasing,'Unordered/duplicate ledger marks')
    require(d.status.eq('ok').all(),'Full-period diagnostics require a complete valued path')
    require(np.allclose(d.nav.diff().fillna(d.nav.iloc[0]-initial_nav),d.net_pnl,atol=1e-10,rtol=0),'Saved NAV/PnL mismatch')
    require(np.allclose(d.long_price_pnl+d.short_price_pnl-d.trading_cost-d.borrow_cost,d.net_pnl,atol=1e-10,rtol=0),'Saved leg/fee PnL mismatch')
    c=d.loc[d.mark.isin(['close','terminal_close'])].copy()
    require(len(c)==1006 and not c.date.duplicated().any(),'Expected1006 daily closes')
    c['running_peak']=np.maximum.accumulate(np.r_[initial_nav,c.nav.to_numpy()])[1:]
    c['drawdown']=c.nav/c.running_peak-1
    return d,c

def period_attribution(daily,stops,initial_nav,expected_maximum_drawdown):
    """Sum saved daily leg PnL; does not reconstruct orders, prices or actions."""
    d,c=close_nav(daily,initial_nav);trough=int(c.drawdown.idxmin())
    prior=c.loc[c.index<=trough]
    if prior.nav.max()<initial_nav:
        peak=-1;base=initial_nav;peak_date='2019-12-31'
    else:
        peak=int(prior.nav.idxmax());base=float(d.loc[peak,'nav']);peak_date=d.loc[peak,'date'].date().isoformat()
    require(np.isclose(c.loc[trough,'drawdown'],expected_maximum_drawdown,atol=1e-10,rtol=0),'Diagnostic DD differs from saved official metric')
    monthly=d.groupby(d.date.dt.strftime('%Y-%m'),sort=True)
    intervals=[('full_2020_2023',-1,len(d)-1,initial_nav),('maximum_drawdown',peak,trough,base)]
    months=[]
    for month,g in monthly:
        start=int(g.index.min())-1;end=int(g.index.max());b=initial_nav if start<0 else float(d.loc[start,'nav'])
        months.append((month,start,end,b))
    worst=sorted(months,key=lambda x:float(d.loc[x[2],'nav'])/x[3]-1)[:5]
    for year,g in d.groupby(d.date.dt.year,sort=True):
        start=int(g.index.min())-1;end=int(g.index.max());b=initial_nav if start<0 else float(d.loc[start,'nav'])
        intervals.append(('year_'+str(year),start,end,b))
    intervals+=months;rows=[]
    stop=stops.copy();stop['date']=pd.to_datetime(stop.date)
    require(stop.date.between('2020-01-02','2023-12-29').all(),'Stop date exceeds approved horizon')
    for name,start,end,b in intervals:
        q=d.iloc[start+1:end+1]
        if q.empty:
            require(name=='maximum_drawdown' and start==end,'Unexpected empty contribution interval')
            s=stop.iloc[:0];start_date=end_date=d.loc[end,'date'].date().isoformat();ending_nav=float(d.loc[end,'nav'])
        else:
            s=stop.loc[stop.date.between(q.date.iloc[0],q.date.iloc[-1])]
            start_date=q.date.iloc[0].date().isoformat();end_date=q.date.iloc[-1].date().isoformat();ending_nav=float(q.nav.iloc[-1])
        values={k:float(q[k].sum())/b*100 for k in ['long_price_pnl','short_price_pnl','trading_cost','borrow_cost']}
        ret=(ending_nav/b-1)*100
        require(np.isclose(values['long_price_pnl']+values['short_price_pnl']-values['trading_cost']-values['borrow_cost'],ret,atol=1e-8,rtol=0),'Period attribution not additive')
        rows.append({'period':name,'start_step_exclusive':start,'end_step_inclusive':end,'start_date':start_date,'end_date':end_date,'zero_length_interval':q.empty,'starting_nav':b,'ending_nav':ending_nav,'long_contribution_pp':values['long_price_pnl'],'short_contribution_pp':values['short_price_pnl'],'trading_cost_contribution_pp':-values['trading_cost'],'borrow_cost_contribution_pp':-values['borrow_cost'],'net_return_percent':ret,'stop_count':len(s),'long_stops':int(s.side.eq('long').sum()),'short_stops':int(s.side.eq('short').sum()),'gap_stops':int(s.fill_type.eq('gap_open').sum())})
    return pd.DataFrame(rows),{'peak_date':peak_date,'trough_date':d.loc[trough,'date'].date().isoformat(),'worst_months':[x[0] for x in worst],'maximum_drawdown':float(c.loc[trough,'drawdown'])}

def stock_attribution_if_supported(daily,holdings,actions,handovers,intervals):
    """Reuse the prior price-only ledger identity; refuse unsupported actions.

    No successor or entitlement algebra is reconstructed here. Aggregated daily
    long/short diagnostics remain available when stock attribution is withheld.
    """
    if len(actions) or len(handovers):
        return pd.DataFrame(),{'status':'WITHHELD_STOCK_LEVEL_ACTION_OR_HANDOVER_REQUIRES_SEPARATE_RECONCILER'}
    groups={key:g.set_index('source_security_id') for key,g in holdings.groupby(['date','mark'],sort=False)}
    previous={};records=[];maxerr=0.
    for step,row in daily.iterrows():
        current=groups[row.date,row.mark];sums={'long':0.,'short':0.}
        for sid,old in previous.items():
            if old['units']==0:continue
            require(sid in current.index,'Prior lot lost from holdings audit')
            now=current.loc[sid];pnl=float(old['units'])*(float(now.valuation_price)-float(old['valuation_price']))
            sums[old['side']]+=pnl;records.append({'step':step,'date':row.date,'source_security_id':sid,'intrinio_security_id':old['intrinio_security_id'],'signal_date':old['signal_date'],'side':old['side'],'gross_pnl':pnl})
        for side in sums:
            err=abs(sums[side]-float(row[side+'_price_pnl']));maxerr=max(maxerr,err)
            require(err<1e-10,'Stock attribution does not reconcile to saved side PnL')
        previous={sid:v.to_dict() for sid,v in current.iterrows() if v.units!=0}
    stock=pd.DataFrame(records);output=[]
    for q in intervals.itertuples():
        p=stock.loc[stock.step.gt(q.start_step_exclusive)&stock.step.le(q.end_step_inclusive)]
        z=p.groupby(['source_security_id','side'],dropna=False).gross_pnl.sum().reset_index();z['period']=q.period;z['contribution_pp']=z.gross_pnl/q.starting_nav*100;output.append(z)
    return pd.concat(output,ignore_index=True),{'status':'RECONCILED_PRICE_ONLY_SAVED_LOTS','maximum_side_mark_error':maxerr}

def focus_rows(frame):
    return frame.loc[frame.arm.eq(FOCUS[0])&frame.basket.eq(FOCUS[1])&frame.variant.eq(FOCUS[2])&frame.cost_bps.isin([0,10])].copy()

def write_focus_diagnostics(bt,audit,comparison,out):
    row=focus_rows(comparison).loc[lambda x:x.cost_bps.eq(10)].iloc[0]
    if not row.complete_proxy_path:
        return {'status':'WITHHELD_FULL_PERIOD_FOCUS_PATH','reason':'No complete-period diagnostic is manufactured from a valid prefix.'}
    prefix='arms/rf_extended/class_1_10/asset_stop_10pct/cost_10bps/'
    def load(name,limit):
        key=prefix+name+'.parquet';require(key in audit['artifacts'],'Saved focus ledger absent: '+key)
        return read_ledger(bt,audit['artifacts'][key],limit)
    daily=load('daily',2012);stops=load('stops',3000)
    intervals,summary=period_attribution(daily,stops,float(row.initial_nav),float(row.maximum_drawdown))
    intervals.to_csv(out/'rf_expanded_SL_period_long_short_cost_stops.csv',index=False)
    stops.to_csv(out/'rf_expanded_SL_stop_exits.csv',index=False)
    stop=stops.copy();stop['month']=pd.to_datetime(stop.date).dt.strftime('%Y-%m')
    stop.groupby(['month','side','fill_type']).size().rename('stop_count').reset_index().to_csv(out/'rf_expanded_SL_monthly_stop_counts.csv',index=False)
    # The prior reconciler is safe only without actions/identity handovers.
    # Refuse those stock-level cases; daily side PnL already includes actions.
    actions=load('action_events',1000);handovers=load('identity_handovers',1000)
    stock,status=stock_attribution_if_supported(daily,load('holdings',100000),actions,handovers,intervals)
    if len(stock):stock.to_csv(out/'rf_expanded_SL_stock_contributions.csv',index=False)
    return {'status':'SAVED_LEDGER_PERIODS_RECONCILED',**summary,'stock_attribution':status,
        'contribution_unit':'percentage points of period-start total portfolio NAV; not standalone long/short returns',
        'stop_episode_caution':'Stop exits are already included in saved stock/leg PnL and are not added again.',
        'allocation_caution':'50/50 applies at monthly rebalance; stops retain cash, permitting intra-month direction drift.'}

def markdown_table(frame,columns):
    def fmt(value,name):
        if pd.isna(value):return '—'
        if name in METRICS and name!='sharpe_zero_cash_rate':return f'{value*100:+.2f}%'
        if name=='sharpe_zero_cash_rate':return f'{value:.2f}'
        return str(value)
    return '\n'.join(['| '+' | '.join(columns)+' |','|'+'|'.join(['---']*len(columns))+'|']+['| '+' | '.join(fmt(row[c],c) for c in columns)+' |' for _,row in frame.iterrows()])

def read_source_corrected_reports(bindings_path,*,root,expected_bindings_sha256,identity_sha256):
    """Side-effect-free notebook API: exact receipts, then saved summaries only.

    This API never reads daily ledgers, source quotes, features, matrices or
    estimators. Detailed diagnostics are an explicit separate reporting step.
    """
    bindings_path=Path(bindings_path).resolve()
    require(digest(bindings_path)==expected_bindings_sha256.removeprefix('sha256:'),'Reporting bindings changed')
    bindings=load_json(bindings_path)
    require(bindings.get('schema')=='dmv2-corrected-passive-report-bindings/v1','Explicit corrected reporting bindings required')
    require(bindings.get('build_identity_sha256')==identity_sha256,'Reporting bindings refer to another corrected run')
    comp,year,audit=saved_tables(root,bindings['backtest_directory'],bindings['backtest_audit'],bindings['backtest_verification'],bindings['upstream_verification'],identity_sha256,bindings['upstream_audits'])
    return {'directory':str((Path(root)/bindings['backtest_directory']).resolve()),'audit':audit,
        'comparison':comp,'yearly':year,'rf_expanded_sl':focus_rows(comp),
        'rf_expanded_sl_yearly':focus_rows(year),'new_fits':0,'model_reload':False,
        'source_price_reads':False,'daily_ledgers_read':False,'financial_2024_plus_read':False}

def render(root,bindings_path,out,old_snapshot):
    root=Path(root).resolve();bindings=load_json(bindings_path);out=Path(out).resolve()
    require(not out.exists(),'Preserve existing reporting snapshot')
    require(bindings.get('schema')=='dmv2-corrected-passive-report-bindings/v1','Explicit corrected reporting bindings required')
    before=preserve(old_snapshot)
    saved=read_source_corrected_reports(bindings_path,root=root,expected_bindings_sha256=digest(bindings_path),identity_sha256=bindings['build_identity_sha256'])
    comp,year,audit=saved['comparison'],saved['yearly'],saved['audit']
    snap=load_json(old_snapshot);old_report=Path(snap['run_directory'])/'reports/completed_20261016'
    old=pd.read_csv(old_report/'backtest_comparison.csv');check_comparison(old)
    out.mkdir(parents=True);joined=comparative(old,comp);joined.to_csv(out/'old_obsolete_vs_corrected_all64.csv',index=False)
    comp.to_csv(out/'corrected_all64.csv',index=False);year.to_csv(out/'corrected_all256_annual.csv',index=False)
    focus_rows(comp).to_csv(out/'rf_expanded_SL_gross_and_10bps.csv',index=False);focus_rows(year).to_csv(out/'rf_expanded_SL_annual_gross_and_10bps.csv',index=False)
    rows=comp.loc[comp.cost_bps.eq(10)].copy();lines=['# Corrected 2020–2023 saved walk-forward reports','',
        'Annual rolling 10-year training; monthly forecasts; 48 months and 1,006 daily sessions. The original source run is obsolete reference evidence. No complete-only model ranking is made.','',
        'RF expanded, class 1/10, 10% per-asset stop is the primary focus. Gross and 10 bps outcomes remain separate.','',
        markdown_table(focus_rows(comp),['arm','basket','variant','cost_bps','status',*METRICS]),'',
        'Annual slices of the same continuous RF expanded SL ledger:','',
        markdown_table(focus_rows(year),['year','cost_bps','complete_full_period_proxy','total_return','maximum_drawdown','annualized_volatility','sharpe_zero_cash_rate']),'',
        'All four models, both class baskets and both stop rules at 10 bps:','',markdown_table(rows,['arm','basket','variant','status',*METRICS]),'',
        'CAGR, volatility and Sharpe are annualized; total return covers four years; drawdown uses daily close NAV. Costs are bps per one-way traded notional. Zero borrow/funding and idealized daily OHLC fills remain research assumptions. RET is not evaluated.','',
        'This is a data-limited historical reconstruction with provisional retrieval-time identities and adjustments, and fixed retrospective Dean Foods/Rowan trading restrictions. The repaired subset does not certify survivorship, borrow or terminal-event completeness. Old-to-new changes combine corrected inputs, training/preprocessing refits and changed forecasts/selections; they do not isolate an OPCH-fill-only effect.']
    diagnostic=write_focus_diagnostics((root/bindings['backtest_directory']).resolve(),audit,comp,out)
    (out/'rf_expanded_SL_diagnostic.json').write_text(json.dumps(diagnostic,indent=2)+'\n')
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    after=preserve(old_snapshot)
    receipt={'schema':'dmv2-corrected-passive-report-snapshot/v1','status':'SAVED_REPORTS_AND_RECONCILED_LEDGER_DIAGNOSTICS','bindings_sha256':digest(bindings_path),'build_identity_sha256':bindings['build_identity_sha256'],'old_reference_preservation_before':before,'old_reference_preservation_after':after,'scenarios':64,'complete_proxy_paths':int(comp.complete_proxy_path.sum()),'withheld_paths':int((~comp.complete_proxy_path).sum()),'focus_diagnostic':diagnostic,'new_fits':0,'model_reload':False,'source_prices_read':False,'accounting_rerun':False,'complete_subset_ranked':False,'financial_2024_plus_read':False,'artifacts':{p.name:{'sha256':digest(p),'size_bytes':p.stat().st_size} for p in out.iterdir() if p.is_file()}}
    (out/'report_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--bindings',required=True);p.add_argument('--output',required=True);p.add_argument('--old-snapshot',required=True);a=p.parse_args()
    print(json.dumps(render(a.root,a.bindings,a.output,a.old_snapshot),indent=2))
