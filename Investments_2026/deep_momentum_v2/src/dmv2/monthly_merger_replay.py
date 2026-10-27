"""Pinned zero-fit realized-return repairs for the existing monthly comparison."""
from pathlib import Path
import itertools, json
from .paper15_baseline_data import pin, check, read, save, table_save, canonical
from .monthly_merger_accounting import need, merger_components, replay_monthly_mergers

ARMS = [f'{k}_paper15_{n}y' for n in (5,10) for k in ('rf','xgb')]
CODE = ['src/dmv2/monthly_merger_accounting.py','src/dmv2/monthly_merger_replay.py',
        'scripts/run_monthly_merger_replay.py','scripts/verify_monthly_merger_replay.py']


def seal(code_root,data_root,protocol,output,destination):
    code,data=Path(code_root).resolve(),Path(data_root).resolve();cfg=read(protocol)
    need(cfg['maximum_fits']==0 and len(cfg['events'])==4 and cfg['change_training_labels'] is False,'Authorized realized-return-only scope')
    parent=data/'outputs/paper15_monthly_baseline/20261026-v1';ledger=read(parent/'ledger.json')
    need(ledger['status']=='COMPLETE_VERIFIED' and ledger['fits_completed']==16,'Verified original monthly run')
    proofs={}
    for name in ('data','models','backtest'):
        item=next(read(check(x['receipt']))['result'] for x in ledger['completed'] if x['key']=='verify_'+name)
        proof=read(check(item));need(proof['status']=='PASS' and proof['plan_pin']==ledger['plan_pin'],'Parent proof binding')
        proofs[name]=item
    data_proof=read(check(proofs['data']));prep_pin=data_proof['prepared_receipt_pin'];prep=read(check(prep_pin))
    back_proof=read(check(proofs['backtest']));audit_pin=back_proof['audit_pin'];audit=read(check(audit_pin))
    source_plan=read(data/'outputs/paper15_ret_baseline/plans/20261026-v3.json')
    source_root=data/'outputs/paper15_ret_baseline/20261026-v3';source_proof=read(source_root/'verification/data.json')
    need(source_proof['status']=='PASS','Reviewed daily cache source proof')
    source_prep_pin=source_proof['prepared_receipt_pin'];source_prep=read(check(source_prep_pin))
    source_vp=read(check(source_proof['verification_read_plan_pin']))
    sources={k:source_prep['artifacts'][k] for k in ('index','daily_prices','quote_validity','month_closes')}
    for k,item in sources.items():need(item==source_vp['prepared_artifacts'][k],'Unbound quote cache');check(item)
    idx=read(check(sources['index']));need(max(idx['sessions'])=='2023-12-29','No protected cache dates')
    points=set()
    for e in cfg['events']:
        need(e['signal_date']<e['effective_date']<=e['outcome_end']<'2024-01-01','Held-before-conversion monthly scope')
        need(idx['sessions'][idx['sessions'].index(e['effective_date'])-1]==e['predecessor_basis_date'],'Last exact predecessor session')
        for date,native in [(e['signal_date'],e['predecessor_id']),(e['predecessor_basis_date'],e['predecessor_id'])]:points.add((date,native))
        if e['stock_ratio']:
            for date in (e['effective_date'],e['outcome_end']):points.add((date,e['successor_id']))
        need(all(n in idx['natives'] for n in (e['predecessor_id'],e['successor_id']) if n),'Cached original/successor routes')
        if e['ticker'] in ('XEC','CDK'):
            old=next(x for x in source_plan['target_events'] if x['predecessor_id']==e['predecessor_id'])
            need(old['verified'] is True and all(e[k]==old[k] for k in ('effective_date','cash_per_raw_share','stock_ratio','successor_id')),'Reviewed existing legal terms changed')
    selected={a:audit['artifacts'][a+'/selection/selected'] for a in ARMS}
    originals={a:audit['artifacts'][a+'/selection/original_ledger'] for a in ARMS}
    outcomes={str(n):prep['artifacts'][f'panel{n}'] for n in (5,10)}
    retention={**{f'panel{n}':prep['artifacts'][f'panel{n}'] for n in (5,10)},
               **{f'payoffs{n}':prep['artifacts'][f'payoffs{n}'] for n in (5,10)},
               **{f'payoff_grid{n}':prep['artifacts'][f'payoff_grid{n}'] for n in (5,10)}}
    model_proof=read(check(proofs['models']));retention.update({'fit/'+k:v for k,v in model_proof['fit_receipts'].items()})
    retention.update({'forecast/'+k:v for k,v in model_proof['forecasts'].items()})
    for item in [*selected.values(),*originals.values(),*retention.values()]:check(item)
    # Nonfinancial check: native accounting matches the parent's canonical
    # turnover keys outside legal conversions; no unrelated route/alias change.
    import pyarrow.parquet as pq
    continuity={}
    for arm,item in selected.items():
        frame=pq.read_table(check(item),columns=['signal_date','source_security_id','intrinio_security_id']).to_pandas()
        old_source={};old_native={}
        for _,g in frame.groupby('signal_date',sort=True):
            by_source=dict(zip(g.source_security_id,g.intrinio_security_id));by_native=dict(zip(g.intrinio_security_id,g.source_security_id))
            need(len(by_source)==len(by_native)==len(g),'Source/native signal collision')
            need(all(old_source[k]==by_source[k] for k in old_source.keys()&by_source.keys()),'Unreviewed consecutive provider-route change')
            need(all(old_native[k]==by_native[k] for k in old_native.keys()&by_native.keys()),'Unreviewed consecutive canonical alias change')
            old_source,old_native=by_source,by_native
        continuity[arm]={'selected_episodes':len(frame),'consecutive_route_changes':0,'consecutive_alias_changes':0}
    budget=sum(v['size_bytes'] for v in sources.values())+sum(v['size_bytes'] for v in outcomes.values())+sum(v['size_bytes'] for v in selected.values())
    need(budget<=cfg['budgets']['cached_input_file_bytes'],'Conservative cached source reservation')
    p=dict(schema='dmv2_monthly_merger_replay_v1',status='SEALED_BEFORE_QUOTE_POINT_READ',code_root=str(code),
        data_root=str(data),output_directory=str(Path(output).resolve()),parent_directory=str(parent),protocol_pin=pin(protocol),
        parent_proofs=proofs,parent_audit_pin=audit_pin,parent_prepared_receipt_pin=prep_pin,
        source_binding_pins=[pin(parent/'ledger.json'),ledger['plan_pin'],pin(source_root/'verification/data.json'),source_prep_pin,
                             source_proof['verification_read_plan_pin'],pin(data/'outputs/paper15_ret_baseline/plans/20261026-v3.json')],
        quote_cache=sources,quote_points=[dict(date=d,intrinio_security_id=n) for d,n in sorted(points)],
        selected=selected,original_ledgers=originals,source_route_continuity=continuity,outcomes=outcomes,retention=retention,cached_file_reservation_bytes=budget,
        code_pins={n:pin(code/n) for n in CODE},helper_pins={n:pin(code/'src/dmv2'/n) for n in ('paper15_baseline_data.py','paper15_monthly.py')},
        maximum_fits=0,forecasts_changed=False,RET_changed=False,financial_2024_plus_read=False)
    p['plan_sha256']=canonical(p);save(destination,p);return p


def validate(path):
    p=read(path);need(p['plan_sha256']==canonical({k:v for k,v in p.items() if k!='plan_sha256'}),'Replay plan changed')
    for item in [p['protocol_pin'],*p['parent_proofs'].values(),p['parent_audit_pin'],p['parent_prepared_receipt_pin'],
                 *p['source_binding_pins'],*p['quote_cache'].values(),*p['selected'].values(),*p['original_ledgers'].values(),
                 *p['retention'].values(),*p['code_pins'].values(),*p['helper_pins'].values()]:check(item)
    need(p['maximum_fits']==0 and not p['forecasts_changed'] and not p['RET_changed'] and not p['financial_2024_plus_read'],'Zero-fit preservation scope')
    return p


def prepare_amendments(plan_path,guard=lambda:None):
    import numpy as np,pandas as pd,pyarrow.parquet as pq
    p=validate(plan_path);cfg=read(check(p['protocol_pin']));root=Path(p['output_directory']);out=root/'amendments'
    need(not out.exists(),'Preserve consumed quote audit');out.mkdir(parents=True)
    save(out/'read_plan.json',dict(status='SEALED_BEFORE_NUMERIC_READ',plan_pin=pin(plan_path),quote_points=p['quote_points'],
                                 quote_cache=p['quote_cache'],cached_file_reservation_bytes=p['cached_file_reservation_bytes'],new_fits=0))
    idx=read(check(p['quote_cache']['index']));days={d:i for i,d in enumerate(idx['sessions'])};natives={n:i for i,n in enumerate(idx['natives'])};months={m:i for i,m in enumerate(idx['months'])}
    prices=np.load(check(p['quote_cache']['daily_prices']),mmap_mode='r',allow_pickle=False)
    quality=np.load(check(p['quote_cache']['quote_validity']),mmap_mode='r',allow_pickle=False)
    closes=np.load(check(p['quote_cache']['month_closes']),mmap_mode='r',allow_pickle=False)
    points=[];quotes={}
    for item in p['quote_points']:
        guard();d,n=item['date'],item['intrinio_security_id'];i,j=days[d],natives[n]
        raw,adj=float(prices[i,j,1]),float(prices[i,j,3]);accepted=bool(quality[i,j,1])
        need(accepted and np.isfinite([raw,adj]).all() and min(raw,adj)>0,'Missing/rejected paired close: '+d+' '+n)
        row=dict(date=d,intrinio_security_id=n,raw_close=raw,adj_close=adj,accepted=accepted)
        quotes[(d,n)]=row;points.append(row)
    corrections=[]
    for e in cfg['events']:
        begin=quotes[(e['signal_date'],e['predecessor_id'])];basis=quotes[(e['predecessor_basis_date'],e['predecessor_id'])]
        need(begin['adj_close']==float(closes[months[e['signal_date'][:7]],natives[e['predecessor_id']]]),'Exact monthly start boundary')
        missing=closes[months[e['outcome_month']],natives[e['predecessor_id']]]
        need(not np.isfinite(missing),'Correction must not overwrite an observed endpoint')
        args={}
        if e['stock_ratio']:
            successor=quotes[(e['effective_date'],e['successor_id'])];end=quotes[(e['outcome_end'],e['successor_id'])]
            need(end['adj_close']==float(closes[months[e['outcome_month']],natives[e['successor_id']]]),'Exact successor month-end mark')
            args=dict(successor_raw_close=successor['raw_close'],successor_adjusted_close=successor['adj_close'],successor_end_adjusted_close=end['adj_close'])
        result=merger_components(begin['adj_close'],basis['raw_close'],basis['adj_close'],e['cash_per_raw_share'],e['stock_ratio'],**args)
        corrections.append({**e,**result,'original_month_end_quote_missing':True,'source_identity_promoted':False,
                            'economic_fractional_units':True,'cash_settlement_convention':'economic_entitlement_zero_rate_to_month_end'})
    artifacts=dict(quote_points=table_save(out/'quote_points.parquet',pd.DataFrame(points)),
                   corrections=save(out/'corrections.json',corrections))
    # Metadata-only dependency record: no X, labels, ranks or RET values rebuilt.
    dependencies=[]
    for n in (5,10):
        meta=pq.read_table(check(p['outcomes'][str(n)]),columns=['source_security_id','signal_date','prediction_eligible','target_available']).to_pandas()
        for e in cfg['events']:
            g=meta.loc[meta.source_security_id.eq(e['source_security_id'])&meta.signal_date.eq(e['signal_date'])]
            need(len(g)==1,'Original correction cohort row retained')
            eligible=bool(g.prediction_eligible.iloc[0])
            dependencies.append(dict(ticker=e['ticker'],history_years=n,signal_date=e['signal_date'],
                eligible=eligible,training_target_available_before=bool(g.target_available.iloc[0]),
                affected_annual_fits_if_promoted=[y for y in (2020,2021,2022,2023) if eligible and y-n<=int(e['outcome_month'][:4])<y],
                promoted_to_learning=False,labels_modified=0,features_modified=0,RET_modified=0))
    artifacts['learning_scope_audit']=save(out/'learning_scope_audit.json',dependencies)
    return save(out/'receipt.json',dict(status='PENDING_INDEPENDENT_UNIT_CHECK',plan_pin=pin(plan_path),artifacts=artifacts,
        quote_point_count=len(points),correction_count=4,training_inputs_changed=False,RET_changed=False,new_fits=0,financial_2024_plus_read=False))


def backtest(plan_path,guard=lambda:None):
    import pandas as pd,pyarrow.parquet as pq
    from .paper15_monthly import monthly_statistics
    p=validate(plan_path);root=Path(p['output_directory']);cfg=read(check(p['protocol_pin']))
    proof=read(root/'verification/amendments.json');need(proof['status']=='PASS','Independent unit proof before replay')
    receipt=read(check(proof['amendment_receipt_pin']));corrections=read(check(receipt['artifacts']['corrections']))
    out=root/'backtest';need(not out.exists(),'Preserve replay attempt');out.mkdir(parents=True)
    artifacts={};comparison=[];annuals=[]
    for arm in ARMS:
        n=5 if arm.endswith('_5y') else 10;selected=pq.read_table(check(p['selected'][arm])).to_pandas()
        outcomes=pq.read_table(check(p['outcomes'][str(n)]),columns=['source_security_id','signal_date','label_return']).to_pandas()
        need(len(selected)==960,'Exactly saved48x20selected episodes')
        for period,first in [('full48','2020-01'),('matched47','2020-02')]:
            months=sorted(selected.loc[selected.outcome_month.ge(first),'outcome_month'].unique())
            for fee in cfg['cost_bps']:
                guard();tables=replay_monthly_mergers(selected,outcomes,months,fee,corrections)
                f=tables['monthly'];complete=len(f)==len(months) and tables['failures'].empty
                row=dict(arm=arm,period=period,cost_bps=fee,complete_monthly_path=complete,expected_months=len(months),observed_months=len(f),
                         first_failure_month=None if tables['failures'].empty else tables['failures'].iloc[0].outcome_month,
                         **monthly_statistics(f,complete));comparison.append(row)
                annual=[]
                for year in (2020,2021,2022,2023):
                    g=f.loc[f.outcome_month.str.startswith(str(year))]
                    annual.append(dict(arm=arm,period=period,cost_bps=fee,year=year,months=len(g),
                                       total_return=float((1+g.net_return).prod()-1) if complete else None))
                base=f'{arm}/{period}/cost_{fee:02d}bps/'
                for name,frame in [*tables.items(),('annual',pd.DataFrame(annual))]:artifacts[base+name]=table_save(out/(base+name+'.parquet'),frame)
                annuals.extend(annual)
        print(json.dumps(dict(completed_arm=arm,scenarios_completed=len(comparison),maximum=32,new_fits=0)),flush=True)
    artifacts['comparison']=table_save(out/'comparison.parquet',pd.DataFrame(comparison));artifacts['annual']=table_save(out/'annual.parquet',pd.DataFrame(annuals))
    return save(out/'audit.json',dict(status='PENDING_INDEPENDENT_ACCOUNTING',plan_pin=pin(plan_path),amendment_proof_pin=pin(root/'verification/amendments.json'),
        artifacts=artifacts,scenarios=32,complete_paths=sum(bool(r['complete_monthly_path']) for r in comparison),
        withheld_paths=sum(not bool(r['complete_monthly_path']) for r in comparison),selected=p['selected'],
        original_ledgers=p['original_ledgers'],upstream_retention=p['retention'],new_fits=0,
        training_labels_changed=False,features_changed=False,RET_changed=False,forecasts_changed=False,financial_2024_plus_read=False))
