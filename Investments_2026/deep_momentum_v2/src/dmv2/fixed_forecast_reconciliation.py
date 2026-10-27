"""Metadata-sealed, zero-fit fixed-forecast accounting reconciliation."""
from pathlib import Path
from copy import deepcopy
import itertools,json,math
from .paper15_baseline_data import pin,check,read,save,table_save,canonical

ARMS=tuple(f'{kind}_paper15_{n}y' for n in (5,10) for kind in ('rf','xgb'))
STAGES=(
 dict(key='cache_close_gross',view='cache',clock='signal_close',unit='close',cost=0,stop=None),
 dict(key='cache_close_net',view='cache',clock='signal_close',unit='close',cost=10,stop=None),
 dict(key='execution_close_gross',view='execution',clock='signal_close',unit='close',cost=0,stop=None),
 dict(key='execution_close_open_units',view='execution',clock='signal_close',unit='open',cost=0,stop=None),
 dict(key='execution_open_gross',view='execution',clock='next_open',unit='open',cost=0,stop=None),
 dict(key='execution_open_net',view='execution',clock='next_open',unit='open',cost=10,stop=None),
 dict(key='execution_open_net_stop',view='execution',clock='next_open',unit='open',cost=10,stop=.1),
)
SELECT=('source_security_id','intrinio_security_id','signal_date','outcome_month','outcome_start','outcome_end','side','intended_weight')
CODE=('src/dmv2/fixed_forecast_accounting.py','src/dmv2/fixed_forecast_reconciliation.py',
      'scripts/run_fixed_forecast_reconciliation.py','scripts/verify_fixed_forecast_reconciliation.py')
HELPERS=('paper15_baseline_data.py','paper15_baseline.py','target_comparison_backtest.py',
 'ret_repair_backtest.py','walkforward_accounting_v2.py','walkforward_accounting.py','backtest_accounting.py','universe.py')


def need(v,m):
    if not v:raise ValueError(m)


def seal(code_root,data_root,protocol,output,destination):
    code,data=Path(code_root).resolve(),Path(data_root).resolve();cfg=read(protocol)
    need(cfg['maximum_fits']==0 and cfg['stages']==list(STAGES) and cfg['arms']==list(ARMS),'Frozen zero-fit scope')
    parent=data/'outputs/paper15_monthly_merger_replay/20261026-v1'
    ledger=read(parent/'ledger.json');need(ledger['status']=='COMPLETE_VERIFIED','Verified realized monthly reference required')
    item=next(read(check(c['receipt']))['result'] for c in ledger['completed'] if c['key']=='verify_backtest')
    proof=read(check(item));need(proof['status']=='PASS' and proof['complete_paths']==32 and proof['new_fits']==0,'Full parent accounting proof')
    old=read(check(ledger['plan_pin']));audit=read(check(proof['audit_pin']))
    source_plan_pin=pin(data/'outputs/paper15_ret_baseline/plans/20261026-v3.json')
    source=read(check(source_plan_pin));phase=deepcopy(source['execution_phase'])
    additions=[]
    for e in read(check(old['protocol_pin']))['events']:
        existing=[v for v in phase['event_registry']['events'] if v['predecessor_id']==e['predecessor_id']]
        if existing:
            need(len(existing)==1 and all(existing[0][k]==e[k] for k in ('effective_date','cash_per_raw_share','stock_ratio','successor_id')),'Conflicting reviewed action')
        else:
            event={k:e[k] for k in ('event_id','effective_date','predecessor_id','successor_id','cash_per_raw_share','stock_ratio')}
            event.update(verified=True,evidence_reference='; '.join(e['evidence_references']),
                         identity_certified=False,execution_only=True)
            phase['event_registry']['events'].append(event);additions.append(event)
    need(len(additions)==2 and len(phase['event_registry']['events'])==22,'Only EV/XLNX legal additions to original20 events')
    idx=read(check(old['quote_cache']['index']))
    sessions=[d for d in idx['sessions'] if '2020-01-31'<=d<='2023-12-29']
    need(len(sessions)==986 and sessions[0]=='2020-01-31' and sessions[-1]=='2023-12-29','Exact initial anchor and985sessions')
    bt=deepcopy(source['execution_source']);bt['sessions']=sessions;bt['expected_sessions']=len(sessions)
    execution_bytes=sum(s[t]['projected_compressed_bytes'] for s in [*bt['daily'],bt['repairs']] for t in ('metadata','values'))
    cache_bytes=sum(v['size_bytes'] for v in old['quote_cache'].values())
    need(execution_bytes<=cfg['budgets']['execution_projection_bytes'] and cache_bytes<=cfg['budgets']['cached_input_file_bytes'],'Explicit source reservations')
    import pyarrow.parquet as pq
    selected={};identities=set();metadata_counts={}
    for arm,item2 in old['selected'].items():
        pf=pq.ParquetFile(check(item2));need(pf.metadata.num_rows==960 and set(SELECT)<=set(pf.schema_arrow.names),'Original48x20selected schema')
        f=pf.read(columns=['source_security_id','intrinio_security_id','signal_date','outcome_month','outcome_start','outcome_end']).to_pandas()
        f=f.loc[f.outcome_month.ge('2020-02')]
        need(len(f)==940 and f.signal_date.nunique()==47,'Frozen matched47 baskets')
        for r in f.to_dict('records'):
            own=[d for d in sessions if d.startswith(r['outcome_month'])]
            need(r['outcome_start']==own[0] and r['outcome_end']==own[-1] and r['signal_date']<own[0], 'Exact next-open metadata clock')
        identities.update(f.intrinio_security_id)
        metadata_counts[arm]=dict(selected_episodes=940,signals=47)
        selected[arm]=item2
    identities.update(e[k] for e in phase['event_registry']['events'] for k in ('predecessor_id','successor_id') if e.get(k))
    for e in phase['historical_retirements']:identities.add(e['predecessor_id'])
    references={}
    for arm,cost in itertools.product(ARMS,(0,10)):
        for name in ('monthly','positions','trades','actions'):
            k=f'{arm}/matched47/cost_{cost:02d}bps/{name}';references[k]=audit['artifacts'][k];check(references[k])
    p=dict(schema='dmv2_fixed_forecast_reconciliation_v1',status='SEALED_BEFORE_DAILY_VALUE_READ',
      code_root=str(code),data_root=str(data),output_directory=str(Path(output).resolve()),protocol_pin=pin(protocol),
      monthly_reference_directory=str(parent),monthly_plan_pin=ledger['plan_pin'],monthly_proof_pin=item,
      monthly_audit_pin=proof['audit_pin'],source_plan_pin=source_plan_pin,quote_cache=old['quote_cache'],
      selected=selected,metadata_counts=metadata_counts,natives=sorted(identities),sessions=sessions,
      reference_artifacts=references,execution_source=bt,execution_phase=phase,additional_actions=additions,
      retention={**old['retention'],**{'selection/'+k:v for k,v in old['selected'].items()},
                 **{'original_ledger/'+k:v for k,v in old['original_ledgers'].items()}},
      code_pins={n:pin(code/n) for n in CODE},helper_pins={n:pin(code/'src/dmv2'/n) for n in HELPERS},
      execution_projection_bytes=execution_bytes,cached_file_reservation_bytes=cache_bytes,
      independent_overlay_pin=source['verifier_helper_pins']['verify_ret_repair_backtest.py'],
      maximum_fits=0,financial_2024_plus_read=False,models_loaded=False)
    p['plan_sha256']=canonical(p);save(destination,p);return p


def validate(path):
    p=read(path)
    need(p['plan_sha256']==canonical({k:v for k,v in p.items() if k!='plan_sha256'}),'Reconciliation plan changed')
    items=[p[k] for k in ('protocol_pin','monthly_plan_pin','monthly_proof_pin','monthly_audit_pin','source_plan_pin','independent_overlay_pin')]
    for name in ('quote_cache','selected','reference_artifacts','retention','code_pins','helper_pins'):items+=list(p[name].values())
    bt=p['execution_source']
    items+=[*bt['execution_pins'].values(),*bt['execution_evidence_pins']]
    for v in [*bt['daily'],bt['repairs']]:items.extend(v[t]['pin'] for t in ('metadata','values'))
    for item in items:check(item)
    need(p['maximum_fits']==0 and not p['financial_2024_plus_read'] and not p['models_loaded'],'Zero-fit protected scope')
    return p


def prepare(plan_path,guard=lambda:None):
    import numpy as np,pandas as pd,pyarrow.parquet as pq
    from .target_comparison_backtest import _load_prices
    p=validate(plan_path);root=Path(p['output_directory']);out=root/'prepared';need(not out.exists(),'Preserve preparation attempt');out.mkdir(parents=True)
    save(out/'read_plan.json',dict(status='SEALED_BEFORE_NUMERIC_READ',plan_pin=pin(plan_path),quote_cache=p['quote_cache'],
       execution_source=p['execution_source'],sessions=p['sessions'],natives=p['natives'],new_fits=0))
    idx=read(check(p['quote_cache']['index']));di={d:i for i,d in enumerate(idx['sessions'])};ni={n:i for i,n in enumerate(idx['natives'])}
    a=np.load(check(p['quote_cache']['daily_prices']),mmap_mode='r',allow_pickle=False)
    q=np.load(check(p['quote_cache']['quote_validity']),mmap_mode='r',allow_pickle=False)
    cache=[]
    for native in p['natives']:
        guard();j=ni.get(native)
        if j is None:continue
        ii=[di[d] for d in p['sessions']];block=a[ii,j,:];flags=q[ii,j,:]
        cache.append(pd.DataFrame(dict(trading_date=p['sessions'],intrinio_security_id=native,
          open=block[:,0],close=block[:,1],adj_open=block[:,2],adj_close=block[:,3],
          price_accepted=flags[:,0]&flags[:,1],open_accepted=flags[:,0],close_accepted=flags[:,1])))
    cf=pd.concat(cache,ignore_index=True)
    execution,overlay,_=_load_prices(p['execution_source'],guard)
    ef=execution.loc[execution.intrinio_security_id.isin(p['natives'])].copy()
    del execution,a,q,cache
    keep=['trading_date','intrinio_security_id','open','close','adj_open','adj_close','adj_high','adj_low',
          'frequency','intraperiod','price_quality_status','raw_ohlc_bounds_valid','adjusted_ohlc_bounds_valid']
    ef=ef[keep]
    artifacts={'cache_prices':table_save(out/'cache_prices.parquet',cf),'execution_prices':table_save(out/'execution_prices.parquet',ef),
               'quote_overlay':save(out/'quote_overlay.json',overlay)}
    for arm,item in p['selected'].items():
        f=pq.read_table(check(item),columns=list(SELECT)).to_pandas()
        f=f.loc[f.outcome_month.ge('2020-02')].sort_values(['signal_date','side','source_security_id'],kind='stable').reset_index(drop=True)
        need(len(f)==940,'Preserved940 selected episodes');artifacts[arm+'/selected']=table_save(out/arm/'selected.parquet',f)
    # Compare quote views, not predictor or source-label values.
    merged=cf.merge(ef,on=['trading_date','intrinio_security_id'],suffixes=('_cache','_execution'),how='outer',indicator=True)
    differences=[]
    for field in ('open','close','adj_open','adj_close'):
        x=pd.to_numeric(merged[field+'_cache']);y=pd.to_numeric(merged[field+'_execution'])
        available=np.isfinite(x)&np.isfinite(y);changed=available&~np.isclose(x,y,rtol=1e-10,atol=1e-10)
        missing=np.isfinite(x)^np.isfinite(y)
        f=merged.loc[changed|missing,['trading_date','intrinio_security_id',field+'_cache',field+'_execution']].copy()
        f['field']=field;f['difference_type']=np.where(changed.loc[f.index],'different_value','availability_difference');differences.append(f)
    artifacts['quote_differences']=table_save(out/'quote_differences.parquet',pd.concat(differences,ignore_index=True))
    return save(out/'receipt.json',dict(status='PENDING_INDEPENDENT_INPUT_CHECK',plan_pin=pin(plan_path),artifacts=artifacts,
      cache_rows=len(cf),execution_rows=len(ef),selected_episodes=3760,new_fits=0,labels_changed=False,RET_changed=False,
      financial_2024_plus_read=False))


def stats(daily,clock,complete):
    import numpy as np,pandas as pd
    closes=daily.loc[daily.mark.ne('open')&daily.date.ge(pd.Timestamp('2020-02-03'))].copy()
    closes['wealth']=np.where(closes.rebalance& (clock=='signal_close'),closes.nav_before_cost,closes.nav)
    if not complete:
        return dict(total_return=None,cagr=None,daily_sharpe=None,daily_volatility=None,daily_maximum_drawdown=None,monthly_sharpe=None,monthly_maximum_drawdown=None),closes,pd.DataFrame()
    need(len(closes)==985 and closes.wealth.notna().all(),'Complete985dailyclosewealth')
    v=closes.wealth.to_numpy(dtype=float);r=v/np.r_[1.,v[:-1]]-1
    monthly=closes.groupby(closes.date.dt.strftime('%Y-%m'),sort=True).tail(1)[['date','wealth']].copy()
    monthly['outcome_month']=monthly.date.dt.strftime('%Y-%m');m=monthly.wealth.to_numpy(dtype=float);mr=m/np.r_[1.,m[:-1]]-1
    monthly['return']=mr
    sd=np.std(r,ddof=1);ms=np.std(mr,ddof=1)
    peak=np.maximum.accumulate(np.r_[1.,v]);mp=np.maximum.accumulate(np.r_[1.,m])
    out=dict(total_return=float(v[-1]-1),cagr=float(v[-1]**(252/985)-1),
      daily_sharpe=float(np.mean(r)/sd*np.sqrt(252)) if sd else None,daily_volatility=float(sd*np.sqrt(252)),
      daily_maximum_drawdown=float(np.min(np.r_[1.,v]/peak-1)),
      monthly_sharpe=float(np.mean(mr)/ms*np.sqrt(12)) if ms else None,
      monthly_maximum_drawdown=float(np.min(np.r_[1.,m]/mp-1)))
    return out,closes,monthly


def replay(plan_path,guard=lambda:None):
    import pandas as pd,pyarrow.parquet as pq,numpy as np
    from .fixed_forecast_accounting import run_fixed_forecast_accounting
    from .backtest_accounting import DailyAccountingPolicy
    p=validate(plan_path);root=Path(p['output_directory']);proof=read(root/'verification/inputs.json');need(proof['status']=='PASS','Independent inputs before replay')
    prep=read(check(proof['prepared_receipt_pin']));out=root/'replay';need(not out.exists(),'Preserve replay attempt');out.mkdir(parents=True)
    frames={v:pq.read_table(check(prep['artifacts'][v+'_prices'])).to_pandas() for v in ('cache','execution')}
    artifacts={};rows=[];reconciliation=[]
    for arm in ARMS:
        original=pq.read_table(check(prep['artifacts'][arm+'/selected'])).to_pandas()
        for st in STAGES:
            guard();sel=original.copy()
            sel['execution_date']=sel.signal_date if st['clock']=='signal_close' else sel.outcome_start
            rb=sel[['signal_date','execution_date']].drop_duplicates().sort_values('execution_date')
            handovers=[] if st['clock']=='signal_close' else p['execution_phase']['identity_handover_registry']['handovers']
            result=run_fixed_forecast_accounting(sel,frames[st['view']],p['sessions'],rebalances=rb,
              terminal_date='2023-12-29',events=p['execution_phase']['event_registry']['events'],
              historical_retirements=p['execution_phase']['historical_retirements'],identity_handovers=handovers,
              clock=st['clock'],successor_factor_mark=st['unit'],
              policy=DailyAccountingPolicy(cost_bps=st['cost'],borrow_annual_bps=0,asset_stop_fraction=st['stop']))
            complete=result.status=='complete_proxy';measure,closes,monthly=stats(result.daily,st['clock'],complete)
            row=dict(arm=arm,stage=st['key'],quote_view=st['view'],execution_clock=st['clock'],successor_factor_mark=st['unit'],
                cost_bps=st['cost'],asset_stop_fraction=st['stop'],complete_path=complete,
                first_failure_date=result.summary['first_failure_date'],stop_count=result.summary['stop_count'],
                unfilled_retired_orders=result.summary['unfilled_retired_order_count'],**measure)
            rows.append(row);base=arm+'/'+st['key']+'/'
            for name,f in [*result.tables.items(),('close_wealth',closes),('monthly',monthly)]:
                artifacts[base+name]=table_save(out/(base+name+'.parquet'),f)
            artifacts[base+'summary']=save(out/(base+'summary.json'),result.summary)
            if st['key'] in ('cache_close_gross','cache_close_net'):
                ref=pq.read_table(check(p['reference_artifacts'][f'{arm}/matched47/cost_{st["cost"]:02d}bps/monthly'])).to_pandas()
                if complete:
                    merged=ref.merge(monthly,on='outcome_month',how='outer',validate='one_to_one')
                    for r in merged.to_dict('records'):
                        reconciliation.append(dict(arm=arm,stage=st['key'],outcome_month=r['outcome_month'],
                           original_month_end_nav=r['nav_end'],daily_engine_month_end_nav=r['wealth'],
                           absolute_nav_difference=r['wealth']-r['nav_end'],matched=bool(math.isclose(r['wealth'],r['nav_end'],rel_tol=1e-9,abs_tol=1e-10))))
                else:reconciliation.append(dict(arm=arm,stage=st['key'],outcome_month=None,matched=False,reason='daily_path_withheld'))
            save(out/'checkpoints'/(arm+'-'+st['key']+'.json'),dict(status='COMPLETE_PENDING_VERIFICATION',row=row,
                  artifacts={k:v for k,v in artifacts.items() if k.startswith(base)}))
            print(json.dumps(dict(arm=arm,stage=st['key'],completed=len(rows),maximum=28,new_fits=0)),flush=True)
    artifacts['comparison']=table_save(out/'comparison.parquet',pd.DataFrame(rows))
    artifacts['reconciliation']=table_save(out/'reconciliation.parquet',pd.DataFrame(reconciliation))
    return save(out/'audit.json',dict(status='PENDING_INDEPENDENT_ACCOUNTING',plan_pin=pin(plan_path),
        input_proof_pin=pin(root/'verification/inputs.json'),artifacts=artifacts,scenarios=28,
        complete_paths=sum(r['complete_path'] for r in rows),withheld_paths=sum(not r['complete_path'] for r in rows),
        selected=p['selected'],retention=p['retention'],new_fits=0,forecasts_changed=False,
        labels_changed=False,RET_changed=False,financial_2024_plus_read=False))
