"""Sealed zero-fit realization of fixed Paper15 baskets, February2020--April2026."""
from pathlib import Path
from copy import deepcopy
import itertools,json,math
from .paper15_baseline_data import pin,check,read,save,table_save,canonical

ARMS=tuple(f'{kind}_paper15_{n}y' for n in (5,10) for kind in ('rf','xgb'))
SELECT=('source_security_id','intrinio_security_id','signal_date','outcome_month','outcome_start','outcome_end','side','intended_weight')
OC=('open','close','adj_open','adj_close')
STAGES=(
 dict(key='cache_close_gross',view='cache',clock='signal_close',unit='close',cost=0,stop=None),
 dict(key='cache_close_net',view='cache',clock='signal_close',unit='close',cost=10,stop=None),
 dict(key='execution_close_gross',view='execution',clock='signal_close',unit='close',cost=0,stop=None),
 dict(key='execution_close_open_units',view='execution',clock='signal_close',unit='open',cost=0,stop=None),
 *tuple(dict(key=('execution_open_gross' if cost==0 else 'execution_open_net' if cost==10 else f'execution_open_cost{cost}')+('_stop' if stop else ''),
             view='execution',clock='next_open',unit='open',cost=cost,stop=stop)
        for cost,stop in itertools.product((0,5,10,25),(None,.1))),
)
CODE=('src/dmv2/extension_backtest.py','src/dmv2/extension_accounting.py','scripts/run_extension_backtest.py','scripts/verify_extension_backtest.py')
HELPERS=('paper15_baseline_data.py','monthly_merger_accounting.py','walkforward_accounting_v2.py','walkforward_accounting.py','backtest_accounting.py','universe.py')


def need(v,m):
    if not v:raise ValueError(m)


def seal(code_root,data_root,protocol,output,destination):
    import pyarrow.parquet as pq
    code,data=Path(code_root).resolve(),Path(data_root).resolve();cfg=read(protocol);bindings=[]
    need(cfg['maximum_fits']==0 and cfg['stages']==list(STAGES) and cfg['arms']==list(ARMS)
         and cfg['first_outcome']=='2020-02' and cfg['last_outcome']=='2026-04','Exact75month zero-fit scope')
    need(not Path(output).exists(),'Preserve prior realization attempts')
    def bind(path):
        item=pin(path);bindings.append(item);return read(check(item))
    def phase(root,key):
        ledger=bind(root/'ledger.json');need(ledger['status']=='COMPLETE_VERIFIED','Verified source ledger required')
        plan=read(check(ledger['plan_pin']));bindings.append(ledger['plan_pin'])
        receipt=next(x['receipt'] for x in ledger['completed'] if x['key']==key);bindings.append(receipt)
        proof_pin=read(check(receipt))['result'];bindings.append(proof_pin);proof=read(check(proof_pin))
        need(proof['status']=='PASS','Independent source proof required');return plan,proof
    fp,fr=phase(data/'outputs/fixed_forecast_reconciliation/20261027-v1','verify_replay')
    bindings.extend([fr['audit_pin'],fr['prepared_input_proof_pin']]);fa=read(check(fr['audit_pin']))
    fi=read(check(fr['prepared_input_proof_pin']));bindings.append(fi['prepared_receipt_pin']);fprep=read(check(fi['prepared_receipt_pin']))
    cp,cr=phase(data/'outputs/fixed_forecast_anchor_completion/20261027-v1','verify_completion')
    bindings.append(cr['audit_pin']);ca=read(check(cr['audit_pin']));need(cr['complete_paths']==28 and cr['withheld_paths']==0,'Completed28path reference')
    iproot=data/'outputs/walkforward_extension_inputs/20261027-v1';il=bind(iproot/'ledger.json')
    need(il['status']=='COMPLETE_VERIFIED_INPUTS','Verified extension inputs')
    bindings.append(il['plan_pin']);ip=read(check(il['plan_pin']))
    iproof=bind(iproot/'verification/data.json');need(iproof['status']=='PASS_INPUT_SOURCE_ARRAYS_AND_MONTHLY_FORMULAS','Source array/formula proof')
    bindings.append(iproof['prepared_receipt_pin']);iprep=read(check(iproof['prepared_receipt_pin']))
    handoff=bind(data/'outputs/walkforward_extension_models/20261027-v1/verification_recovery/selection_handoff.json')
    need(handoff['status']=='PASS_INHERITED_12_MODELS_AND_RECOVERED_SELECTIONS','Verified fixed model/selection handoff')
    for k,v in handoff.items():
        if k.endswith('_pin'):check(v);bindings.append(v)
    sp=read(check(handoff['selection_proof_pin']));mp=read(check(handoff['inherited_model_proof_pin']))
    need(sp['status']=='PASS' and mp['status']=='PASS' and mp['verified_fits']==12,'12models and selection independent PASS')
    bindings.append(sp['audit_pin']);sa=read(check(sp['audit_pin']))
    md=read(check(mp['data_proof_pin']));bindings.append(mp['data_proof_pin']);bindings.append(md['prepared_receipt_pin'])
    mprep=read(check(md['prepared_receipt_pin']))
    _,amproof=phase(data/'outputs/paper15_monthly_merger_replay/20261026-v1','verify_amendments')
    amreceipt=bind(data/'outputs/paper15_monthly_merger_replay/20261026-v1/amendments/receipt.json')
    # Proof authenticates the exact original correction receipt, retained verbatim.
    need(any(v==pin(data/'outputs/paper15_monthly_merger_replay/20261026-v1/amendments/receipt.json') for v in amproof.values() if isinstance(v,dict)),'Original legal correction binding')
    inputs={'old_cache_prices':fprep['artifacts']['cache_prices'],
        'old_execution_prices':ca['artifacts']['completed_execution_prices'],
        'old_index':fp['quote_cache']['index'],'old_daily_prices':fp['quote_cache']['daily_prices'],'old_quote_validity':fp['quote_cache']['quote_validity'],
        'index':iprep['artifacts']['index'],'extension_ohlc':iprep['artifacts']['extension_ohlc'],
        'extension_quote_validity':iprep['artifacts']['extension_quote_validity'],'valid':iprep['artifacts']['valid'],
        'panel5':mprep['artifacts']['panel5'],'panel10':mprep['artifacts']['panel10'],
        'old_corrections':amreceipt['artifacts']['corrections'],'legal_evidence':pin(cfg['legal_evidence_path'])}
    idx=read(check(inputs['index']));oldidx=read(check(inputs['old_index']))
    sessions=[d for d in idx['sessions'] if '2020-01-31'<=d<='2026-04-30']
    need(len(sessions)==1570 and sessions[0]=='2020-01-31' and sessions[-1]=='2026-04-30'
         and len(idx['extension_sessions'])==584,'Exact75month/1569following sessions')
    months=sorted({d[:7] for d in sessions if d>='2020-02-03'});need(len(months)==75,'75outcome months')
    selected={};natives=set();counts={};anchor_ids=set()
    for arm in ARMS:
        item=sa['artifacts'][arm+'/selected'];check(item);selected[arm]=item
        f=pq.read_table(check(item),columns=list(SELECT)).to_pandas();f=f.loc[f.outcome_month.ge('2020-02')]
        need(len(f)==1500 and sorted(f.outcome_month.unique())==months and (f.groupby('outcome_month').size()==20).all(),'Exact75x20 fixed basket')
        natives.update(f.intrinio_security_id);anchor_ids.update(f.loc[f.outcome_month.eq('2024-01')].intrinio_security_id)
        counts[arm]=dict(months=75,selected_episodes=1500)
    need(anchor_ids<=set(oldidx['natives']),'Boundary raw-anchor requires separately authorized source projection')
    phase2=deepcopy(fp['execution_phase']);event=deepcopy(cfg['additional_event'])
    need(event['predecessor_id']=='sec_g4KJZy' and event['effective_date']=='2025-06-18'
         and event['cash_per_raw_share']==55 and event['stock_ratio']==0 and event['successor_id'] is None
         and event['verified'] is True,'Only reviewed U.S.Steel cash entitlement')
    need(not any(e['predecessor_id']==event['predecessor_id'] for e in phase2['event_registry']['events']),'New distinct legal event')
    phase2['event_registry']['events'].append(event)
    natives.update(e[k] for e in phase2['event_registry']['events'] for k in ('predecessor_id','successor_id') if e.get(k))
    natives.update(e['predecessor_id'] for e in phase2['historical_retirements'])
    # Full source bytes are reserved even though memory maps decode only exact date/native projections.
    reservation=sum(v['size_bytes'] for v in inputs.values())+sum(v['size_bytes'] for v in selected.values())
    need(reservation<=cfg['budgets']['source_read_bytes_per_pass'],'Source byte ceiling')
    references={}
    for key,item in fa['artifacts'].items():
        if key.endswith('/close_wealth'):references[key]=ca['artifacts'].get(key,item)
    p=dict(schema=cfg['schema'],status='SEALED_BEFORE_DAILY_FINANCIAL_READ',code_root=str(code),data_root=str(data),
        output_directory=str(Path(output).resolve()),protocol_pin=pin(protocol),source_binding_pins=bindings,inputs=inputs,
        selected=selected,reference_artifacts=references,sessions=sessions,months=months,natives=sorted(natives),anchor_ids=sorted(anchor_ids),
        metadata_counts=counts,execution_phase=phase2,original_execution_phase=fp['execution_phase'],
        source_read_bytes_reserved=reservation,maximum_fits=0,models_loaded=False,financial_end='2026-04-30',
        code_pins={n:pin(code/n) for n in CODE},helper_pins={n:pin(code/'src/dmv2'/n) for n in HELPERS})
    p['plan_sha256']=canonical(p);save(destination,p);return p


def validate(plan_path):
    p=read(plan_path);need(p['plan_sha256']==canonical({k:v for k,v in p.items() if k!='plan_sha256'}),'Changed realization plan')
    need(p['maximum_fits']==0 and not p['models_loaded'] and p['financial_end']=='2026-04-30','Bounded zero-fit scope')
    for item in [p['protocol_pin'],*p['source_binding_pins'],*p['inputs'].values(),*p['selected'].values(),*p['reference_artifacts'].values(),*p['code_pins'].values(),*p['helper_pins'].values()]:check(item)
    return p


def prepare(plan_path,guard=lambda:None):
    import numpy as np,pandas as pd,pyarrow.parquet as pq
    from .monthly_merger_accounting import merger_components
    p=validate(plan_path);cfg=read(check(p['protocol_pin']));root=Path(p['output_directory']);out=root/'prepared';need(not out.exists(),'Preserve consumed preparation');out.mkdir(parents=True)
    save(out/'read_plan.json',dict(status='SEALED_BEFORE_NUMERIC_READ',plan_pin=pin(plan_path),inputs=p['inputs'],
         source_bytes_reserved=p['source_read_bytes_reserved'],financial_end=p['financial_end'],new_fits=0))
    idx=read(check(p['inputs']['index']));ni={n:i for i,n in enumerate(idx['natives'])};di={d:i for i,d in enumerate(idx['sessions'])}
    arr=np.load(check(p['inputs']['extension_ohlc']),mmap_mode='r',allow_pickle=False)
    flags=np.load(check(p['inputs']['extension_quote_validity']),mmap_mode='r',allow_pickle=False)
    valid=np.load(check(p['inputs']['valid']),mmap_mode='r',allow_pickle=False)
    parts=[];extdates=idx['extension_sessions'];ii=[di[d] for d in extdates]
    for n in p['natives']:
        guard()
        if n not in ni:continue
        j=ni[n];values=arr[:,j,:];q=flags[:,j,:]
        f=pd.DataFrame(values,columns=idx['ohlc_columns']);f['trading_date']=extdates;f['intrinio_security_id']=n
        f['open_accepted']=q[:,0];f['close_accepted']=q[:,1];f['price_accepted']=q[:,0]&q[:,1]
        f['full_ohlc_accepted']=valid[ii,j];parts.append(f)
    ext=pd.concat(parts,ignore_index=True)
    oldcache=pq.read_table(check(p['inputs']['old_cache_prices'])).to_pandas()
    oldexec=pq.read_table(check(p['inputs']['old_execution_prices'])).to_pandas()
    # Supplement only previously unrequested Dec2023 anchor IDs from the immutable observed OC cache.
    oi=read(check(p['inputs']['old_index']));on={n:i for i,n in enumerate(oi['natives'])};od=oi['sessions'].index('2023-12-29')
    oa=np.load(check(p['inputs']['old_daily_prices']),mmap_mode='r',allow_pickle=False)
    oq=np.load(check(p['inputs']['old_quote_validity']),mmap_mode='r',allow_pickle=False)
    supplements=[]
    for n in p['anchor_ids']:
        row=dict(trading_date='2023-12-29',intrinio_security_id=n,**dict(zip(OC,map(float,oa[od,on[n],:]))))
        row.update(price_accepted=bool(oq[od,on[n],:].all()),open_accepted=bool(oq[od,on[n],0]),close_accepted=bool(oq[od,on[n],1]))
        supplements.append(row)
    anchor=pd.DataFrame(supplements);keys=['trading_date','intrinio_security_id']
    frames={}
    for view,old in [('cache',oldcache),('execution',oldexec)]:
        add=anchor.merge(old[keys],on=keys,how='left',indicator=True).query('_merge == "left_only"').drop(columns='_merge')
        new=ext.copy()
        if view=='cache':new=new[[*keys,*OC,'price_accepted','open_accepted','close_accepted']]
        else:
            new['price_accepted']=new.full_ohlc_accepted
            new=new[[*keys,*OC,'adj_high','adj_low','price_accepted']]
        f=pd.concat([old,add,new],ignore_index=True)
        need(not f.duplicated(keys).any(),'Unique continuous quote grid')
        frames[view]=f
    artifacts={v+'_prices':table_save(out/(v+'_prices.parquet'),f) for v,f in frames.items()}
    artifacts['boundary_observed_anchors']=table_save(out/'boundary_observed_anchors.parquet',anchor)
    # Cash entitlement for exactly the selected X episode. It never repairs a learning label.
    corrections=read(check(p['inputs']['old_corrections']));need(len(corrections)==4,'Unchanged four original monthly corrections')
    ep=p['execution_phase']['event_registry']['events'][-1];native=ep['predecessor_id']
    points=ext.loc[ext.intrinio_security_id.eq(native)&ext.trading_date.isin(['2025-05-30','2025-06-17'])]
    need(len(points)==2 and points.price_accepted.all(),'Observed X paired close factors')
    start=points.loc[points.trading_date.eq('2025-05-30')].iloc[0];base=points.loc[points.trading_date.eq('2025-06-17')].iloc[0]
    component=merger_components(float(start.adj_close),float(base.close),float(base.adj_close),55.,0.)
    correction=dict(**component,event_id=ep['event_id'],source_security_id='siblis_ticker_company_1ca6af788db837c4ecee',
       predecessor_id=native,successor_id=None,cash_per_raw_share=55.,stock_ratio=0.,signal_date='2025-05-30',
       outcome_month='2025-06',outcome_end='2025-06-30',effective_date='2025-06-18',predecessor_basis_date='2025-06-17',
       cash_settlement_convention='economic_entitlement_zero_rate_to_month_end',execution_only=True,evidence_pin=p['inputs']['legal_evidence'])
    corrections.append(correction);artifacts['corrections']=save(out/'corrections.json',corrections)
    artifacts['action_observed_points']=table_save(out/'action_observed_points.parquet',points)
    for arm,item in p['selected'].items():
        f=pq.read_table(check(item),columns=list(SELECT)).to_pandas();f=f.loc[f.outcome_month.ge('2020-02')].sort_values(['signal_date','side','source_security_id'],kind='stable').reset_index(drop=True)
        need(len(f)==1500,'Unchanged75x20 selected rows');artifacts[arm+'/selected']=table_save(out/arm/'selected.parquet',f)
    for n in (5,10):
        f=pq.read_table(check(p['inputs'][f'panel{n}']),columns=['source_security_id','signal_date','label_return']).to_pandas()
        artifacts[f'outcomes{n}']=table_save(out/f'outcomes{n}.parquet',f)
    return save(out/'receipt.json',dict(status='PENDING_INDEPENDENT_INPUT_CHECK',plan_pin=pin(plan_path),artifacts=artifacts,
       new_fits=0,models_loaded=False,forecasts_changed=False,labels_changed=False,RET_changed=False,new_legal_events=1))


def stats(daily,clock,complete):
    import numpy as np,pandas as pd
    closes=daily.loc[daily.mark.ne('open')&daily.date.ge(pd.Timestamp('2020-02-03'))].copy()
    closes['wealth']=np.where(closes.rebalance& (clock=='signal_close'),closes.nav_before_cost,closes.nav)
    if not complete:
        return dict(total_return=None,cagr=None,daily_sharpe=None,daily_volatility=None,daily_maximum_drawdown=None,monthly_sharpe=None,monthly_maximum_drawdown=None),closes,pd.DataFrame()
    need(len(closes)==1569 and closes.wealth.notna().all(),'Complete1569dailyclosewealth')
    v=closes.wealth.to_numpy(dtype=float);r=v/np.r_[1.,v[:-1]]-1
    monthly=closes.groupby(closes.date.dt.strftime('%Y-%m'),sort=True).tail(1)[['date','wealth']].copy()
    monthly['outcome_month']=monthly.date.dt.strftime('%Y-%m');m=monthly.wealth.to_numpy(dtype=float);mr=m/np.r_[1.,m[:-1]]-1
    monthly['return']=mr
    sd=np.std(r,ddof=1);ms=np.std(mr,ddof=1)
    peak=np.maximum.accumulate(np.r_[1.,v]);mp=np.maximum.accumulate(np.r_[1.,m])
    out=dict(total_return=float(v[-1]-1),cagr=float(v[-1]**(252/1569)-1),
      daily_sharpe=float(np.mean(r)/sd*np.sqrt(252)) if sd else None,daily_volatility=float(sd*np.sqrt(252)),
      daily_maximum_drawdown=float(np.min(np.r_[1.,v]/peak-1)),
      monthly_sharpe=float(np.mean(mr)/ms*np.sqrt(12)) if ms else None,
      monthly_maximum_drawdown=float(np.min(np.r_[1.,m]/mp-1)))
    return out,closes,monthly


def replay(plan_path,guard=lambda:None):
    import numpy as np,pandas as pd,pyarrow.parquet as pq
    from .extension_accounting import run_fixed_forecast_accounting
    from .backtest_accounting import DailyAccountingPolicy
    from .monthly_merger_accounting import replay_monthly_mergers
    p=validate(plan_path);root=Path(p['output_directory']);proof=read(root/'verification/inputs.json');need(proof['status']=='PASS','Independent input check before realization')
    prep=read(check(proof['prepared_receipt_pin']));out=root/'replay';need(not out.exists(),'Preserve consumed replay');out.mkdir(parents=True)
    frames={v:pq.read_table(check(prep['artifacts'][v+'_prices'])).to_pandas() for v in ('cache','execution')}
    corrections=read(check(prep['artifacts']['corrections']));artifacts={};rows=[];prefix=[]
    for arm in ARMS:
        original=pq.read_table(check(prep['artifacts'][arm+'/selected'])).to_pandas()
        n=5 if arm.endswith('_5y') else 10
        outcomes=pq.read_table(check(prep['artifacts'][f'outcomes{n}'])).to_pandas()
        for cost in (0,5,10,25):
            guard();result=replay_monthly_mergers(original,outcomes,p['months'],cost,corrections)
            base=f'{arm}/monthly_cost{cost}/'
            for name,f in result.items():artifacts[base+name]=table_save(out/(base+name+'.parquet'),f)
        for st in STAGES:
            guard();sel=original.copy();sel['execution_date']=sel.signal_date if st['clock']=='signal_close' else sel.outcome_start
            rb=sel[['signal_date','execution_date']].drop_duplicates().sort_values('execution_date')
            hand=[] if st['clock']=='signal_close' else p['execution_phase']['identity_handover_registry']['handovers']
            result=run_fixed_forecast_accounting(sel,frames[st['view']],p['sessions'],rebalances=rb,terminal_date='2026-04-30',
              events=p['execution_phase']['event_registry']['events'],historical_retirements=p['execution_phase']['historical_retirements'],
              identity_handovers=hand,clock=st['clock'],successor_factor_mark=st['unit'],
              policy=DailyAccountingPolicy(cost_bps=st['cost'],borrow_annual_bps=0,asset_stop_fraction=st['stop']))
            complete=result.status=='complete_proxy';measure,closes,monthly=stats(result.daily,st['clock'],complete)
            rows.append(dict(arm=arm,stage=st['key'],quote_view=st['view'],execution_clock=st['clock'],successor_factor_mark=st['unit'],
                cost_bps=st['cost'],asset_stop_fraction=st['stop'],complete_path=complete,months=75,sessions=1569,
                first_failure_date=result.summary['first_failure_date'],stop_count=result.summary['stop_count'],
                unfilled_retired_orders=result.summary['unfilled_retired_order_count'],**measure))
            base=f'{arm}/{st["key"]}/'
            for name,f in [*result.tables.items(),('close_wealth',closes),('monthly',monthly)]:artifacts[base+name]=table_save(out/(base+name+'.parquet'),f)
            artifacts[base+'summary']=save(out/(base+'summary.json'),result.summary)
            ref=p['reference_artifacts'].get(base+'close_wealth')
            if ref:
                old=pq.read_table(check(ref),columns=['date','wealth']).to_pandas()
                new=closes.loc[closes.date.le(pd.Timestamp('2023-12-29')),['date','wealth']]
                joined=old.merge(new,on='date',how='outer',suffixes=('_old','_new'),validate='one_to_one')
                ok=len(joined)==985 and np.allclose(joined.wealth_old,joined.wealth_new,rtol=1e-9,atol=1e-10,equal_nan=False)
                prefix.append(dict(arm=arm,stage=st['key'],matched=bool(ok),observations=len(joined),
                    maximum_absolute_nav_difference=float((joined.wealth_old-joined.wealth_new).abs().max())))
            save(out/'checkpoints'/(arm+'-'+st['key']+'.json'),dict(status='REALIZED_PENDING_INDEPENDENT_CHECK',row=rows[-1],
                artifacts={k:v for k,v in artifacts.items() if k.startswith(base)}))
            print(json.dumps(dict(arm=arm,stage=st['key'],completed=len(rows),maximum=48,new_fits=0,complete=complete)),flush=True)
    artifacts['comparison']=table_save(out/'comparison.parquet',pd.DataFrame(rows))
    artifacts['prefix_parity']=table_save(out/'prefix_parity.parquet',pd.DataFrame(prefix))
    return save(out/'audit.json',dict(status='PENDING_INDEPENDENT_ACCOUNTING',plan_pin=pin(plan_path),input_proof_pin=pin(root/'verification/inputs.json'),
        artifacts=artifacts,scenarios=48,monthly_references=16,complete_paths=sum(r['complete_path'] for r in rows),withheld_paths=sum(not r['complete_path'] for r in rows),
        new_fits=0,forecasts_changed=False,labels_changed=False,RET_changed=False,models_loaded=False))
