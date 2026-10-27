#!/usr/bin/env python3
"""Independent point-unit and frozen-selection cash/share monthly replay proof.

No producer accounting, target construction, estimator or selection imports.
"""
from pathlib import Path
import hashlib,json,math


def need(value,message):
    if not value:raise ValueError(message)


def pin(path):
    p=Path(path).resolve();h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return dict(path=str(p),size_bytes=p.stat().st_size,sha256=h.hexdigest())


def check(item):
    need(pin(item['path'])=={k:item[k] for k in ('path','size_bytes','sha256')},'Independent immutable bytes changed')
    return Path(item['path'])


def read(path):
    need(Path(path).stat().st_size<=32*1024**2,'Independent JSON ceiling')
    return json.loads(Path(path).read_text())


def save(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
    return pin(p)


def context(plan_path,phase):
    p=read(plan_path);body={k:v for k,v in p.items() if k!='plan_sha256'}
    need(hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()==p['plan_sha256'],'Independent plan identity')
    items=[p['protocol_pin'],*p['parent_proofs'].values(),p['parent_audit_pin'],p['parent_prepared_receipt_pin'],*p['source_binding_pins'],
           *p['quote_cache'].values(),*p['selected'].values(),*p['original_ledgers'].values(),*p['retention'].values(),*p['code_pins'].values(),*p['helper_pins'].values()]
    for item in items:check(item)
    root=Path(p['output_directory']);cfg=read(check(p['protocol_pin']));a=read(root/'amendments/receipt.json')
    need(p['maximum_fits']==cfg['maximum_fits']==0 and p['financial_2024_plus_read'] is False and a['plan_pin']==pin(plan_path),'Independent zero-fit scope')
    for item in a['artifacts'].values():check(item)
    inventory={}
    if phase=='backtest':
        audit=read(root/'backtest/audit.json');inventory=audit['artifacts']
        need(sum(v['size_bytes'] for v in inventory.values())<=cfg['budgets']['output_bytes'],'Independent saved artifact ceiling')
        for item in inventory.values():check(item)
    vp=save(root/'verification'/('read_plan_'+phase+'.json'),dict(status='SEALED_BEFORE_NUMERIC_READ',plan_pin=pin(plan_path),
        amendment_receipt_pin=pin(root/'amendments/receipt.json'),amendment_artifacts=a['artifacts'],
        source_cache=p['quote_cache'],source_points=p['quote_points'],backtest_artifacts=inventory,
        retained_upstream=p['retention'],worker_pin=pin(__file__),new_fits=0))
    return p,cfg,root,a,vp


def verify_amendments(plan_path,guard=lambda:None):
    import numpy as np,pandas as pd,pyarrow.parquet as pq
    p,cfg,root,a,vp=context(plan_path,'amendments');idx=read(check(p['quote_cache']['index']))
    di,ni={d:i for i,d in enumerate(idx['sessions'])},{n:i for i,n in enumerate(idx['natives'])};mi={m:i for i,m in enumerate(idx['months'])}
    prices=np.load(check(p['quote_cache']['daily_prices']),mmap_mode='r',allow_pickle=False)
    good=np.load(check(p['quote_cache']['quote_validity']),mmap_mode='r',allow_pickle=False)
    close=np.load(check(p['quote_cache']['month_closes']),mmap_mode='r',allow_pickle=False)
    q=pq.read_table(check(a['artifacts']['quote_points'])).to_pandas()
    need(len(q)==len(p['quote_points'])==12 and not q.duplicated(['date','intrinio_security_id']).any(),'Exact12observedclosepoints')
    quote={}
    for row in q.itertuples():
        guard();i,j=di[row.date],ni[row.intrinio_security_id]
        need(row.date<'2024-01-01' and bool(good[i,j,1]) and row.accepted,'Independent close quality')
        need(row.raw_close==prices[i,j,1] and row.adj_close==prices[i,j,3] and min(row.raw_close,row.adj_close)>0,'Paired source observations')
        quote[(row.date,row.intrinio_security_id)]=row
    repairs=read(check(a['artifacts']['corrections']));need(len(repairs)==4,'Exactly four amended stock-months')
    for event,repair in zip(cfg['events'],repairs):
        need(all(event[k]==repair[k] for k in event),'Legal source terms modified')
        begin=quote[(event['signal_date'],event['predecessor_id'])];pre=quote[(event['predecessor_basis_date'],event['predecessor_id'])]
        need(event['signal_date']<event['effective_date']<=event['outcome_end'] and idx['sessions'][di[event['effective_date']]-1]==event['predecessor_basis_date'],'Held-before-event calendar')
        need(not np.isfinite(close[mi[event['outcome_month']],ni[event['predecessor_id']]]),'No observed old-stock endpoint overwritten')
        need(begin.adj_close==close[mi[event['signal_date'][:7]],ni[event['predecessor_id']]],'Independent monthly start mark')
        pre_event_wealth=pre.adj_close/begin.adj_close;raw_shares=pre_event_wealth/pre.raw_close
        cash=raw_shares*event['cash_per_raw_share'];stock=0.
        if event['stock_ratio']:
            at=quote[(event['effective_date'],event['successor_id'])];end=quote[(event['outcome_end'],event['successor_id'])]
            stock=raw_shares*event['stock_ratio']*at.raw_close*(end.adj_close/at.adj_close)
            need(end.adj_close==close[mi[event['outcome_month']],ni[event['successor_id']]],'Independent successor end mark')
        for name,wanted in [('asset_return',cash+stock-1),('cash_component',cash),('successor_component',stock),
                            ('predecessor_raw_shares_per_initial_capital',raw_shares)]:
            need(abs(repair[name]-wanted)<=1e-12,'Independent raw-share formula '+name)
        need(repair['source_identity_promoted'] is False and repair['original_month_end_quote_missing'] is True,'No invented price/identity promotion')
    dependencies=read(check(a['artifacts']['learning_scope_audit']))
    need(len(dependencies)==8 and all(not d['promoted_to_learning'] and d['labels_modified']==d['features_modified']==d['RET_modified']==0 for d in dependencies),'No learning-input mutation')
    need(not (root/'folds').exists(),'Replay must not fit models')
    return save(root/'verification/amendments.json',dict(status='PASS',plan_pin=pin(plan_path),amendment_receipt_pin=pin(root/'amendments/receipt.json'),
        verification_read_plan_pin=vp,verified_stock_months=4,verified_quote_points=12,verified_raw_share_formulas=True,
        new_fits=0,labels_changed=False,RET_changed=False,financial_2024_plus_read=False,worker_pin=pin(__file__)))


def verify_backtest(plan_path,guard=lambda:None):
    import numpy as np,pandas as pd,pyarrow.parquet as pq
    p,cfg,root,a,vp=context(plan_path,'backtest');audit=read(root/'backtest/audit.json');art=audit['artifacts']
    need(audit['plan_pin']==pin(plan_path) and audit['selected']==p['selected'] and audit['original_ledgers']==p['original_ledgers']
         and audit['upstream_retention']==p['retention'],'Frozen forecast/selection/input preservation')
    unit=read(root/'verification/amendments.json');need(unit['status']=='PASS' and unit['amendment_receipt_pin']==pin(root/'amendments/receipt.json'),'Unit proof binding')
    fix={(x['source_security_id'],x['signal_date']):x for x in read(check(a['artifacts']['corrections']))}
    comp=pq.read_table(check(art['comparison'])).to_pandas();annual_all=pq.read_table(check(art['annual'])).to_pandas()
    need(len(comp)==32 and len(annual_all)==128 and not comp.duplicated(['arm','period','cost_bps']).any(),'Full scenario/annual grid')
    oldaudit=read(check(p['parent_audit_pin']));total_actions=0
    for arm,selected_pin in p['selected'].items():
        guard();n='5' if arm.endswith('_5y') else '10'
        pick=pq.read_table(check(selected_pin)).to_pandas();outcomes=pq.read_table(check(p['outcomes'][n]),columns=['source_security_id','signal_date','label_return']).to_pandas()
        values=outcomes.set_index(['source_security_id','signal_date']).label_return.to_dict()
        need(len(pick)==960 and selected_pin==oldaudit['artifacts'][arm+'/selection/selected'],'Same960originalselectedepisodes')
        for period,first in [('full48','2020-01'),('matched47','2020-02')]:
            months=sorted(pick.loc[pick.outcome_month.ge(first),'outcome_month'].unique())
            need(len(months)==(48 if period=='full48' else 47),'Fixed comparison dates')
            for fee in (0,5,10,25):
                base=f'{arm}/{period}/cost_{fee:02d}bps/'
                frames={k:pq.read_table(check(art[base+k])).to_pandas() for k in ('monthly','positions','actions','trades','failures','annual')}
                f=frames['monthly'];need(frames['failures'].empty and f.outcome_month.tolist()==months,'Complete monthly path, no missing-return waiver')
                before=1.;held={};expected_positions=[];expected_actions=[];expected_trades=[]
                for i,month in enumerate(months):
                    g=pick.loc[pick.outcome_month.eq(month)];need(len(g)==20 and g.intrinio_security_id.is_unique,'Unchanged20-namebasket')
                    target={row.intrinio_security_id:before*row.intended_weight for row in g.itertuples()}
                    changes={native:target.get(native,0)-held.get(native,0) for native in target.keys()|held.keys()}
                    cost=fee/10000*math.fsum(abs(v) for v in changes.values())
                    # Reconstruct cash from NAV and new target notionals, independently
                    # of the producer's carried-cash and trade-settlement update.
                    cash=before-math.fsum(target.values())-cost;ending={};lp=0.;sp=0.
                    for native in sorted(changes):
                        expected_trades.append(dict(outcome_month=month,intrinio_security_id=native,
                            pretrade_notional=held.get(native,0.),target_notional=target.get(native,0.),
                            signed_trade_notional=changes[native],trading_cost=abs(changes[native])*fee/10000))
                    for row in g.itertuples():
                        key=(row.source_security_id,row.signal_date);event=fix.get(key);amount=target[row.intrinio_security_id]
                        r=event['asset_return'] if event else values[key]
                        need(np.isfinite(r) and r>=-1,'Every selected realized return independently supported')
                        pnl=amount*r
                        if row.side=='long':lp+=pnl
                        else:sp+=pnl
                        if event:
                            ce=amount*event['cash_component'];se=amount*event['successor_component'];successor=event['successor_id'];cash+=ce;remaining=0.
                            if successor:ending[successor]=ending.get(successor,0)+se
                            expected_actions.append(dict(outcome_month=month,source_security_id=row.source_security_id,side=row.side,
                                event_id=event['event_id'],predecessor_id=row.intrinio_security_id,successor_id=successor,
                                signed_initial_notional=amount,signed_cash_entitlement=ce,signed_successor_end_notional=se,
                                mandatory_conversion_market_cost=0.))
                        else:
                            ce=0.;se=0.;successor=None;remaining=amount*(1+r)
                            ending[row.intrinio_security_id]=ending.get(row.intrinio_security_id,0)+remaining
                        expected_positions.append(dict(outcome_month=month,source_security_id=row.source_security_id,intrinio_security_id=row.intrinio_security_id,
                            side=row.side,target_notional=amount,asset_return=r,pnl=pnl,predecessor_end_notional=remaining,
                            signed_cash_entitlement=ce,successor_id=successor,signed_successor_end_notional=se,
                            return_source='verified_merger_entitlement' if event else 'original_adjusted_close_ratio'))
                    after=cash+math.fsum(ending.values());actual=f.iloc[i]
                    expected=dict(nav_start=before,nav_end=after,gross_return=(lp+sp)/before,net_return=after/before-1,
                        long_contribution=lp/before,short_contribution=sp/before,trading_cost=cost,cost_contribution=-cost/before,
                        one_way_turnover=math.fsum(abs(v) for v in changes.values())/before,cash_end=cash)
                    for name,wanted in expected.items():need(abs(actual[name]-wanted)<2e-10,'Independent cash/share '+name)
                    need(after>0 and actual.selected_long==actual.selected_short==10,'Positive NAV/same allocation')
                    before,held=after,ending
                for name,expected,keys in [('positions',expected_positions,['outcome_month','source_security_id']),
                                           ('actions',expected_actions,['outcome_month','source_security_id']),
                                           ('trades',expected_trades,['outcome_month','intrinio_security_id'])]:
                    left=frames[name].sort_values(keys).reset_index(drop=True);right=pd.DataFrame(expected).sort_values(keys).reset_index(drop=True)
                    pd.testing.assert_frame_equal(left,right,check_exact=False,rtol=2e-11,atol=2e-12)
                total_actions+=len(expected_actions)
                row=comp.loc[comp.arm.eq(arm)&comp.period.eq(period)&comp.cost_bps.eq(fee)].iloc[0]
                need(row.complete_monthly_path and row.expected_months==row.observed_months==len(months) and pd.isna(row.first_failure_month),'No truncated result')
                returns=f.net_return.to_numpy();mean=math.fsum(returns)/len(returns);sd=math.sqrt(math.fsum((r-mean)**2 for r in returns)/(len(returns)-1))
                wealth=np.r_[1.,np.cumprod(1+returns)]
                want=dict(total_return=wealth[-1]-1,cagr=wealth[-1]**(12/len(returns))-1,annualized_arithmetic_mean=12*mean,
                    annualized_volatility=sd*math.sqrt(12),sharpe_zero_cash_rate=mean/sd*math.sqrt(12),
                    monthly_maximum_drawdown=float((wealth/np.maximum.accumulate(wealth)-1).min()),
                    total_one_way_turnover=math.fsum(f.one_way_turnover),trading_cost_paid=math.fsum(f.trading_cost))
                for name,v in want.items():need(abs(row[name]-v)<5e-10,'Independent aggregate '+name)
                annual=frames['annual'];need(annual.year.tolist()==[2020,2021,2022,2023],'No year dropped')
                for year in (2020,2021,2022,2023):
                    g=f.loc[f.outcome_month.str.startswith(str(year))];ar=annual.loc[annual.year.eq(year)].iloc[0]
                    need(ar.months==len(g) and abs(ar.total_return-(math.prod(1+x for x in g.net_return)-1))<5e-11,'Continuous annual return')
                if period=='full48':
                    # Before any corrected event, every return, cost and NAV must
                    # match the published14-month diagnostic under all costs.
                    oldkey=f'{arm}/full48/cost_{fee:02d}bps/monthly'
                    old=pq.read_table(check(oldaudit['artifacts'][oldkey]),columns=['outcome_month','nav_start','nav_end','net_return','trading_cost','one_way_turnover']).to_pandas()
                    old=old.loc[old.outcome_month.le('2021-02')].reset_index(drop=True)
                    now=f.loc[f.outcome_month.le('2021-02'),old.columns].reset_index(drop=True)
                    pd.testing.assert_frame_equal(old,now,check_exact=False,rtol=2e-11,atol=2e-12)
    need(total_actions==48 and bool(comp.complete_monthly_path.all()) and audit['complete_paths']==32 and audit['withheld_paths']==0,'All32complete/48signedactionapplications')
    for item in p['retention'].values():check(item)
    need(not (root/'folds').exists(),'No refits allowed')
    return save(root/'verification/backtest.json',dict(status='PASS',plan_pin=pin(plan_path),audit_pin=pin(root/'backtest/audit.json'),
        amendment_proof_pin=pin(root/'verification/amendments.json'),verification_read_plan_pin=vp,verified_scenarios=32,
        complete_paths=32,withheld_paths=0,verified_action_applications=48,forecasts_and_selection_byte_identical=True,
        labels_features_and_RET_byte_identical=True,common14month_results_preserved=True,new_fits=0,financial_2024_plus_read=False,worker_pin=pin(__file__)))
