#!/usr/bin/env python3
"""Independent source and cash/share oracle for frozen-forecast reconciliation.

Never imports the new accounting/producer module, a model, features or labels.
"""
from pathlib import Path
from collections import defaultdict
import hashlib,importlib.util,json,math,statistics,sys

META=('source_security_id','intrinio_security_id','signal_date','outcome_month','outcome_start','outcome_end','side','intended_weight')


def need(v,m):
    if not v:raise ValueError(m)


def pin(path):
    p=Path(path).resolve();h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return dict(path=str(p),size_bytes=p.stat().st_size,sha256=h.hexdigest())


def check(v):
    need(pin(v['path'])=={k:v[k] for k in ('path','size_bytes','sha256')},'Independent artifact bytes changed')
    return Path(v['path'])


def read(p):return json.loads(Path(p).read_text())


def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    return pin(p)


def number(a,b,m):
    if b is None:need(a is None or (isinstance(a,float) and math.isnan(a)),m);return
    need(a is not None and math.isfinite(float(a)) and math.isclose(float(a),float(b),rel_tol=1e-9,abs_tol=1e-10),m+': '+str(a)+' != '+str(b))


class Halt(Exception):
    def __init__(self,date,mark,reason,key=None,native=None,field=None):
        self.details=dict(date=date,mark=mark,reason=reason,source_security_id=key,intrinio_security_id=native,price_field=field)


def oracle(selected,prices,sessions,events,historical,handovers,stage,tables,guard=lambda:None):
    """Independent transition equations, computed from inputs rather than saved NAV."""
    import numpy as np,pandas as pd
    prices=prices.to_dict('records') if hasattr(prices,'to_dict') else prices
    quotes={(str(r.get('trading_date',r.get('date')))[:10],r['intrinio_security_id']):r for r in prices}
    records=selected.to_dict('records') if hasattr(selected,'to_dict') else selected
    clock,unit,fee,stop=stage['clock'],stage['unit'],stage['cost']/10000,stage['stop']
    bydate={}
    for r in records:
        d=str(r['signal_date'] if clock=='signal_close' else r['outcome_start'])[:10]
        bydate.setdefault(d,[]).append(r)
    action_by=defaultdict(list)
    for e in sorted(events,key=lambda r:(r['effective_date'],r['event_id'])):action_by[e['effective_date']].append(e)
    retired=sorted([*historical,*events],key=lambda r:(r['effective_date'],r['event_id']))
    positions={};last={};cash=1.;previous=1.;verified=0;failure=None
    days=list(sessions);daily=tables['daily'].copy()
    daily['date']=daily.date.astype(str).str[:10]
    need(len(daily)==len(days)*2,'Complete mark/withholding grid')
    indexed=daily.set_index(['date','mark'])
    trade=tables['trades'].copy();actions=tables['action_events'].copy();holds=tables['holdings'].copy();stops=tables['stops'].copy()
    for f in (trade,actions,holds,stops):
        if not f.empty:f['date']=f.date.astype(str).str[:10]
    def positive(v):return v is not None and isinstance(v,(int,float,np.integer,np.floating)) and math.isfinite(v) and v>0
    def quote(date,mark,native,key,field=None):
        field=field or ('adj_open' if mark=='open' else 'adj_close');r=quotes.get((date,native))
        if r is None:raise Halt(date,mark,'missing_quote',key,native,field)
        if 'price_accepted' in r:accepted=isinstance(r['price_accepted'],(bool,np.bool_)) and bool(r['price_accepted'])
        else:accepted=r.get('frequency')=='daily' and r.get('intraperiod') is False and r.get('price_quality_status') in ('ok','accepted')
        accepted=accepted and all(isinstance(r[k],(bool,np.bool_)) and bool(r[k]) for k in ('raw_ohlc_bounds_valid','adjusted_ohlc_bounds_valid') if k in r)
        if not accepted:raise Halt(date,mark,'source_quality_flag',key,native,field)
        if not positive(r.get(field)):raise Halt(date,mark,'invalid_price',key,native,field)
        return float(r[field])
    expected_closes=[];expected_actions=0;expected_stops=0;unfilled=0
    for di,date in enumerate(days):
      for mark in ('open','terminal_close' if date==days[-1] else 'close'):
        guard();saved=indexed.loc[(date,mark)]
        if failure is not None:
            need(saved.status=='failed_closed' and pd.isna(saved.nav),'Failed path retains unavailable NAV');continue
        try:
            before={k:dict(v) for k,v in positions.items()};prior={k:v['units']*last[k] for k,v in positions.items()}
            action_cash={'long':0.,'short':0.};action_expected=[]
            if mark=='open':
                for e in action_by[date]:
                    for key in sorted(list(positions)):
                        pos=positions[key]
                        if pos['native']!=e['predecessor_id']:continue
                        if di==0:raise Halt(date,mark,'missing_event_predecessor_basis',key,pos['native'],'raw_close/adj_close')
                        base=quotes.get((days[di-1],pos['native']),{});raw=base.get('raw_close',base.get('close'));adj=base.get('adj_close')
                        if not positive(raw) or not positive(adj):raise Halt(date,mark,'missing_event_predecessor_basis',key,pos['native'],'raw_close/adj_close')
                        old_units=pos['units'];raw_units=old_units*adj/raw;credit=raw_units*e['cash_per_raw_share'];new_units=0.
                        factor=None
                        if e['stock_ratio']:
                            successor=e['successor_id'];op=quote(date,mark,successor,key,'adj_open');sr=quotes[(date,successor)];raw_open=sr.get('raw_open',sr.get('open'))
                            if not positive(raw_open):raise Halt(date,mark,'missing_event_successor_basis',key,successor,'raw_open/adj_open')
                            if unit=='open':factor=op/raw_open
                            else:
                                ac=quote(date,mark,successor,key,'adj_close');rc=sr.get('raw_close',sr.get('close'))
                                if not positive(rc):raise Halt(date,mark,'missing_successor_close_factor',key,successor,'raw_close/adj_close')
                                factor=ac/rc
                            new_units=raw_units*e['stock_ratio']/factor
                        cash+=credit;action_cash[pos['side']]+=credit
                        action_expected.append(dict(source_security_id=key,event_id=e['event_id'],old_adjusted_units=old_units,
                            signed_raw_entitlement=raw_units,signed_cash=credit,new_adjusted_units=new_units))
                        if new_units:
                            pos['native']=e['successor_id'];pos['units']=new_units
                            pos['entry']=(old_units*pos['entry']-credit)/new_units
                            pos['threshold']=None if pos['threshold'] is None else (old_units*pos['threshold']-credit)/new_units
                            pos['action']=True
                        else:del positions[key]
            rebalance=date in bydate and ((clock=='next_open' and mark=='open') or (clock=='signal_close' and mark!='open'))
            chosen=bydate[date] if rebalance else None;executable=[];order_count=0
            for r in chosen or []:
                if any(v['predecessor_id']==r['intrinio_security_id'] and v['effective_date']<=date for v in retired):
                    unfilled+=1;continue
                executable.append(r)
            required={k:(v['native'],'held') for k,v in positions.items()}
            for r in executable:
                key,native=r['source_security_id'],r['intrinio_security_id']
                if key in positions and positions[key]['native']!=native:raise Halt(date,mark,'held_identity_route_changed',key,native)
                for oldkey,pos in positions.items():
                    if oldkey==key or pos['native']!=native or pos.get('action'):continue
                    allowed=clock=='next_open' and any(h['effective_date']==date and h['old_source_security_id']==oldkey and h['new_source_security_id']==key and h['intrinio_security_id']==native and h['signal_date']==r['signal_date'] and h['held_signal_date']==pos['signal'] and oldkey not in {v['source_security_id'] for v in chosen} for h in handovers)
                    if not allowed:raise Halt(date,mark,'selected_provider_held_source_collision',key,native)
                required.setdefault(key,(native,'selected'))
            marks={k:quote(date,mark,native,k) for k,(native,_) in required.items()};stop_fills={}
            if stop is not None:
                for key,pos in positions.items():
                    threshold=pos['threshold']
                    if threshold is None:continue
                    if mark=='open' and chosen is None:
                        crossed=marks[key]<=threshold if pos['side']=='long' else marks[key]>=threshold
                        if crossed:stop_fills[key]='gap_open'
                    elif mark!='open':
                        low=quote(date,mark,pos['native'],key,'adj_low');high=quote(date,mark,pos['native'],key,'adj_high');op=quote(date,mark,pos['native'],key,'adj_open')
                        if not low<=min(op,marks[key])<=max(op,marks[key])<=high:raise Halt(date,mark,'invalid_adjusted_ohlc_bounds',key,pos['native'],'adj_low/adj_high')
                        crossed=low<=threshold if pos['side']=='long' else high>=threshold
                        if crossed:stop_fills[key]='idealized_intraday_threshold';marks[key]=threshold
            pre=cash+sum(pos['units']*marks[k] for k,pos in positions.items())
            if not math.isfinite(pre) or pre<=0:raise Halt(date,mark,'nonpositive_or_nonfinite_nav')
            lp=sum(pos['units']*marks[k] for k,pos in positions.items() if pos['side']=='long')-sum(v for k,v in prior.items() if before[k]['side']=='long')+action_cash['long']
            sp=sum(pos['units']*marks[k] for k,pos in positions.items() if pos['side']=='short')-sum(v for k,v in prior.items() if before[k]['side']=='short')+action_cash['short']
            target=None;phase=None
            if chosen is not None:
                n=sum(r['side']=='long' for r in chosen);need(n==len(chosen)-n==10,'Frozen ten-per-side cardinality')
                target={}
                for r in executable:
                    key=r['source_security_id'];weight=(.5 if r['side']=='long' else -.5)/n;reference=marks[key]
                    target[key]=dict(native=r['intrinio_security_id'],side=r['side'],units=weight*pre/reference,entry=reference,
                        threshold=None if stop is None else reference*(1-stop if r['side']=='long' else 1+stop),
                        signal=r['signal_date'],action=False)
                phase='rebalance'
            elif stop_fills:
                target={k:dict(v) for k,v in positions.items() if k not in stop_fills};phase='stop_gap' if mark=='open' else 'stop_intraday'
            fee_paid=0.;traded=0.;instructions={}
            if target is not None:
                groups=defaultdict(list)
                for key in sorted(set(positions)|set(target)):
                    old_units=positions.get(key,{}).get('units',0.);new_units=target.get(key,{}).get('units',0.);delta=new_units-old_units
                    if delta==0:continue
                    pos=target.get(key,positions.get(key));signed=delta*marks[key]
                    bucket=key if key in stop_fills and mark!='open' else 'simultaneous_mark'
                    groups[(pos['native'],marks[key],bucket)].append((key,old_units,new_units,delta,signed))
                for group in groups.values():
                    signed=sum(v[4] for v in group);cost=abs(signed)*fee;fee_paid+=cost;traded+=abs(signed)
                    denom=sum(abs(v[4]) for v in group)
                    for key,old_u,new_u,delta,signed_i in group:
                        instructions[key]=dict(units_before=old_u,units_after=new_u,units_delta=delta,price=marks[key],
                            signed_notional=signed_i,cost=cost*abs(signed_i)/denom if denom else 0.)
                cash-=sum(v['signed_notional'] for v in instructions.values())+fee_paid
                positions=target
            nav=cash+sum(v['units']*marks[k] for k,v in positions.items())
            if not math.isfinite(nav) or nav<=0:raise Halt(date,mark,'nonpositive_or_nonfinite_nav_after_cost')
            need(saved.status=='ok','Valid mark unexpectedly withheld')
            for name,value in dict(nav=nav,nav_before_cost=pre,cash=cash,long_price_pnl=lp,short_price_pnl=sp,
                trading_cost=fee_paid,traded_notional=traded,net_pnl=lp+sp-fee_paid,borrow_cost=0.).items():number(saved[name],value,'Independent mark '+name)
            number(saved['return'],nav/previous-1,'Independent mark return')
            number(nav-previous,lp+sp-fee_paid,'Independent cash/share PnL conservation')
            own_trades=trade.loc[trade.date.eq(date)&trade.mark.eq(mark)] if not trade.empty else trade
            actual_keys=set(own_trades.source_security_id) if not own_trades.empty else set()
            mandatory={k for k,v in instructions.items() if abs(v['signed_notional'])>1e-10}
            need(mandatory<=actual_keys,'Missing material trade instruction')
            for r in own_trades.to_dict('records'):
                expected=instructions.get(r['source_security_id'])
                if expected is None:need(abs(r['signed_notional'])<=1e-10,'Unexpected material trade');continue
                need(r['phase']==phase,'Changed trade phase')
                for name,value in expected.items():number(r[name],value,'Independent trade '+name)
            own_actions=actions.loc[actions.date.eq(date)&actions.mark.eq(mark)] if not actions.empty else actions
            need(len(own_actions)==len(action_expected),'Mandatory action count')
            for a in action_expected:
                r=own_actions.loc[own_actions.source_security_id.eq(a['source_security_id'])&own_actions.event_id.eq(a['event_id'])]
                need(len(r)==1,'Independent legal action key')
                for name in ('old_adjusted_units','signed_raw_entitlement','signed_cash','new_adjusted_units'):number(r.iloc[0][name],a[name],'Independent entitlement '+name)
            own_holds=holds.loc[holds.date.eq(date)&holds.mark.eq(mark)] if not holds.empty else holds
            nonzero=own_holds.loc[own_holds.units.ne(0)] if not own_holds.empty else own_holds
            need(set(nonzero.source_security_id)==set(positions),'Held lot retention')
            for r in nonzero.to_dict('records'):
                pos=positions[r['source_security_id']];need(pos['native']==r['intrinio_security_id'],'Held successor route');number(r['units'],pos['units'],'Held signed units')
            own_stops=stops.loc[stops.date.eq(date)&stops.mark.eq(mark)] if not stops.empty else stops
            got_stops=set(own_stops.source_security_id) if not own_stops.empty else set()
            need(got_stops==set(stop_fills),'Independent stop keys')
            for r in own_stops.to_dict('records'):
                need(r['fill_type']==stop_fills[r['source_security_id']],'Stop fill convention');number(r['fill_price'],marks[r['source_security_id']],'Stop fill')
            expected_actions+=len(action_expected);expected_stops+=len(stop_fills);last={k:marks[k] for k in positions}
            if mark!='open' and date>='2020-02-03':expected_closes.append((date,pre if clock=='signal_close' and rebalance else nav))
            previous=nav;verified+=1
        except Halt as exc:
            failure=exc.details;need(saved.status=='failed_closed' and pd.isna(saved.nav),'Missing/invalid mark did not fail closed')
            own=tables['failures'];datestr=own.date.astype(str).str[:10]
            first=own.loc[datestr.eq(date)&own.mark.eq(mark)]
            need(not first.empty and failure['reason'] in set(first.reason),'First independent failure reason')
    if failure:
        fd,fm=failure['date'],failure['mark'];after=trade.loc[trade.date.gt(fd)|(trade.date.eq(fd)&(trade.mark.ne('open') if fm=='open' else False))] if not trade.empty else trade
        need(after.empty,'No trades after withheld mark')
    return dict(complete=failure is None,closes=expected_closes,verified_marks=verified,actions=expected_actions,stops=expected_stops,
                unfilled_retired_orders=unfilled,first_failure=failure)


def independent_metrics(closes,complete):
    if not complete:return dict(total_return=None,cagr=None,daily_sharpe=None,daily_volatility=None,daily_maximum_drawdown=None,monthly_sharpe=None,monthly_maximum_drawdown=None),[]
    need(len(closes)==1569,'Independent1569close observations')
    v=[x[1] for x in closes];returns=[v[0]-1]+[v[i]/v[i-1]-1 for i in range(1,len(v))]
    ends={}
    for d,value in closes:ends[d[:7]]=(d,value)
    months=list(ends.values());m=[x[1] for x in months];mr=[m[0]-1]+[m[i]/m[i-1]-1 for i in range(1,len(m))]
    def dd(values):
        peak=1.;low=0.
        for x in values:peak=max(peak,x);low=min(low,x/peak-1)
        return low
    sd=statistics.stdev(returns);ms=statistics.stdev(mr)
    return dict(total_return=v[-1]-1,cagr=v[-1]**(252/1569)-1,
      daily_sharpe=statistics.mean(returns)/sd*math.sqrt(252) if sd else None,
      daily_volatility=sd*math.sqrt(252),daily_maximum_drawdown=dd(v),
      monthly_sharpe=statistics.mean(mr)/ms*math.sqrt(12) if ms else None,monthly_maximum_drawdown=dd(m)),months


def context(plan_path):
    p=read(plan_path);digest=hashlib.sha256(json.dumps({k:v for k,v in p.items() if k!='plan_sha256'},sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
    need(p['plan_sha256']==digest and p['maximum_fits']==0 and p['models_loaded'] is False and p['financial_end']=='2026-04-30','Independent bounded zero-fit plan')
    for v in [p['protocol_pin'],*p['source_binding_pins'],*p['inputs'].values(),*p['selected'].values(),*p['reference_artifacts'].values(),*p['code_pins'].values(),*p['helper_pins'].values()]:check(v)
    cfg=read(check(p['protocol_pin']));need(cfg['scenarios']==48 and len(cfg['stages'])==12 and len(cfg['arms'])==4,'Full predeclared grid')
    old=p['original_execution_phase'];now=p['execution_phase'];need(now['historical_retirements']==old['historical_retirements'] and now['identity_handover_registry']==old['identity_handover_registry'],'Old retirements/handover preserved')
    need(now['event_registry']['events']==old['event_registry']['events']+[cfg['additional_event']],'Only evidenced X addition')
    # Evidence is the complete official SEC HTML, not a generated quote or secondary price.
    import re,html
    document=html.unescape(re.sub('<[^>]+>',' ',check(p['inputs']['legal_evidence']).read_text())).split()
    document=' '.join(document)
    need('$55.00 in cash' in document and 'prior to the open of trading on June 18, 2025' in document,'Official legal consideration/removal evidence')
    return p,cfg,Path(p['output_directory'])


def verify_inputs(plan_path,guard=lambda:None):
    import numpy as np,pandas as pd,pyarrow.parquet as pq
    p,cfg,root=context(plan_path);rp=pin(root/'prepared/receipt.json');receipt=read(check(rp));arts=receipt['artifacts']
    need(receipt['plan_pin']==pin(plan_path) and receipt['new_fits']==0 and not receipt['labels_changed'] and not receipt['RET_changed'],'Learning scope preserved')
    need(sum(v['size_bytes'] for v in arts.values())<=cfg['budgets']['derived_verification_bytes'],'Derived read ceiling')
    vp=save(root/'verification/input_read_plan.json',dict(status='SEALED_BEFORE_VERIFICATION_READ',plan_pin=pin(plan_path),prepared_receipt_pin=rp,artifacts=arts,worker_pin=pin(__file__)))
    for v in arts.values():check(v)
    idx=read(check(p['inputs']['index']));dates=idx['extension_sessions'];di={d:i for i,d in enumerate(idx['sessions'])};ni={n:i for i,n in enumerate(idx['natives'])}
    a=np.load(check(p['inputs']['extension_ohlc']),mmap_mode='r',allow_pickle=False);q=np.load(check(p['inputs']['extension_quote_validity']),mmap_mode='r',allow_pickle=False);v=np.load(check(p['inputs']['valid']),mmap_mode='r',allow_pickle=False)
    oi=read(check(p['inputs']['old_index']));on={n:i for i,n in enumerate(oi['natives'])};od=oi['sessions'].index('2023-12-29')
    oa=np.load(check(p['inputs']['old_daily_prices']),mmap_mode='r',allow_pickle=False);oq=np.load(check(p['inputs']['old_quote_validity']),mmap_mode='r',allow_pickle=False)
    anchors=pq.read_table(check(arts['boundary_observed_anchors'])).to_pandas()
    need(sorted(anchors.intrinio_security_id)==p['anchor_ids'] and anchors.trading_date.eq('2023-12-29').all(),'Exact boundary source projection')
    for row in anchors.itertuples():
        need(np.array_equal(np.array([row.open,row.close,row.adj_open,row.adj_close]),oa[od,on[row.intrinio_security_id],:],equal_nan=True),'Original observed anchor values')
        need(row.price_accepted==bool(oq[od,on[row.intrinio_security_id],:].all()),'Anchor source acceptance')
    keys=['trading_date','intrinio_security_id'];oc=['open','close','adj_open','adj_close']
    for view in ('cache','execution'):
        f=pq.read_table(check(arts[view+'_prices'])).to_pandas();old=pq.read_table(check(p['inputs']['old_'+view+'_prices'])).to_pandas()
        need(not f.duplicated(keys).any() and f.trading_date.isin(p['sessions']).all(),'Independent all-row date/key guard')
        merged=old[keys].merge(f,on=keys,validate='one_to_one')
        pd.testing.assert_frame_equal(old.reset_index(drop=True),merged[old.columns].reset_index(drop=True),check_dtype=False,check_exact=True)
        added=f.loc[f.trading_date.lt('2024-01-01')].merge(old[keys],on=keys,how='left',indicator=True).query('_merge == "left_only"').drop(columns='_merge')
        expected=anchors.merge(old[keys],on=keys,how='left',indicator=True).query('_merge == "left_only"').drop(columns='_merge')
        pd.testing.assert_frame_equal(added[expected.columns].sort_values(keys).reset_index(drop=True),expected.sort_values(keys).reset_index(drop=True),check_dtype=False,check_exact=True)
        extra=f.loc[f.trading_date.ge('2024-01-01')];need(len(extra)==584*len(set(p['natives'])&set(ni)),'Exact bounded extension grid')
        for native,g in extra.groupby('intrinio_security_id'):
            guard();g=g.sort_values('trading_date');j=ni[native];need(g.trading_date.tolist()==dates,'No date omitted')
            fields=oc if view=='cache' else [*oc,'adj_high','adj_low'];columns=[idx['ohlc_columns'].index(k) for k in fields]
            need(np.array_equal(g[fields].to_numpy(float),a[:,j,:][:,columns],equal_nan=True),'Independent source array projection')
            wanted=q[:,j,0]&q[:,j,1] if view=='cache' else v[[di[d] for d in dates],j]
            need(np.array_equal(g.price_accepted.to_numpy(),wanted),'Source acceptance retained')
    original=read(check(p['inputs']['old_corrections']));fix=read(check(arts['corrections']));need(fix[:4]==original and len(fix)==5,'Only X realized correction')
    x=fix[-1];j=ni['sec_g4KJZy'];sd=dates.index('2025-05-30');bd=dates.index('2025-06-17')
    cash=float(a[bd,j,7]/a[bd,j,3]*55/a[sd,j,7]);number(x['cash_component'],cash,'Independent signed raw-share entitlement basis');number(x['asset_return'],cash-1,'Independent X economic return')
    need(x['stock_ratio']==0 and x['successor_id'] is None and x['effective_date']=='2025-06-18' and x['execution_only'],'Cash-only execution correction')
    for arm,item in p['selected'].items():
        source=pq.read_table(check(item),columns=list(META)).to_pandas();source=source.loc[source.outcome_month.ge('2020-02')].sort_values(['signal_date','side','source_security_id'],kind='stable').reset_index(drop=True)
        actual=pq.read_table(check(arts[arm+'/selected'])).to_pandas();pd.testing.assert_frame_equal(source,actual,check_dtype=False,check_exact=True)
        need(len(actual)==1500 and actual.signal_date.nunique()==75,'Fixed75x20 selections')
        need(not actual.intrinio_security_id.isin(cfg['execution_restrictions']).any(),'Existing execution restrictions retained')
    for n in (5,10):
        cols=['source_security_id','signal_date','label_return'];source=pq.read_table(check(p['inputs'][f'panel{n}']),columns=cols).to_pandas();actual=pq.read_table(check(arts[f'outcomes{n}'])).to_pandas()
        pd.testing.assert_frame_equal(source,actual,check_dtype=False,check_exact=True)
    return save(root/'verification/inputs.json',dict(status='PASS',plan_pin=pin(plan_path),prepared_receipt_pin=rp,verification_read_plan_pin=vp,
        independently_projected_sources=True,new_fits=0,forecasts_changed=False,labels_changed=False,RET_changed=False))


def verify_monthly(selected,outcomes,months,cost,corrections,tables):
    """Independent monthly notional equations; does not import replay workers."""
    values=outcomes.set_index(['source_security_id','signal_date']).label_return.to_dict();fix={(v['source_security_id'],v['signal_date']):v for v in corrections}
    f=tables['monthly'];need(tables['failures'].empty and f.outcome_month.tolist()==months,'Full75monthly payoff coverage')
    nav=1.;held={};action_count=0
    for i,month in enumerate(months):
        g=selected.loc[selected.outcome_month.eq(month)];target={r.intrinio_security_id:nav*r.intended_weight for r in g.itertuples()}
        delta={n:target.get(n,0)-held.get(n,0) for n in target.keys()|held.keys()};fee=math.fsum(abs(v) for v in delta.values())*cost/10000
        cash=nav-math.fsum(target.values())-fee;ending={};lp=0.;sp=0.
        for r in g.itertuples():
            e=fix.get((r.source_security_id,r.signal_date));value=e['asset_return'] if e else values[(r.source_security_id,r.signal_date)]
            need(math.isfinite(value) and value>=-1,'Observed legal or original monthly return')
            amount=target[r.intrinio_security_id];pnl=amount*value
            if r.side=='long':lp+=pnl
            else:sp+=pnl
            if e:
                cash+=amount*e['cash_component'];action_count+=1
                if e['successor_id']:ending[e['successor_id']]=ending.get(e['successor_id'],0)+amount*e['successor_component']
            else:ending[r.intrinio_security_id]=ending.get(r.intrinio_security_id,0)+amount*(1+value)
        after=cash+math.fsum(ending.values());actual=f.iloc[i]
        for k,w in dict(nav_start=nav,nav_end=after,gross_return=(lp+sp)/nav,net_return=after/nav-1,long_contribution=lp/nav,short_contribution=sp/nav,trading_cost=fee,cash_end=cash).items():number(actual[k],w,'Independent monthly '+k)
        need(actual.selected_long==actual.selected_short==10,'Original monthly cardinality');nav,held=after,ending
    need(len(tables['actions'])==action_count,'Monthly legal event count');return action_count


def verify_replay(plan_path,guard=lambda:None):
    import numpy as np,pandas as pd,pyarrow.parquet as pq
    p,cfg,root=context(plan_path);ap=pin(root/'replay/audit.json');audit=read(check(ap));arts=audit['artifacts']
    ip=pin(root/'verification/inputs.json');inp=read(check(ip));need(inp['status']=='PASS' and audit['input_proof_pin']==ip and audit['plan_pin']==pin(plan_path),'Independent input proof required')
    prep=read(check(inp['prepared_receipt_pin']));need(sum(v['size_bytes'] for v in arts.values())<=cfg['budgets']['derived_verification_bytes'],'Bounded output verification')
    vp=save(root/'verification/replay_read_plan.json',dict(status='SEALED_BEFORE_INDEPENDENT_REPLAY',plan_pin=pin(plan_path),audit_pin=ap,artifacts=arts,worker_pin=pin(__file__)))
    for v in arts.values():check(v)
    frames={v:pq.read_table(check(prep['artifacts'][v+'_prices'])).to_pandas() for v in ('cache','execution')}
    comp=pq.read_table(check(arts['comparison'])).to_pandas();need(len(comp)==48 and not comp.duplicated(['arm','stage']).any(),'Full48path denominator')
    corrections=read(check(prep['artifacts']['corrections']));scenarios=[];reconcile=[];prefix=[];annual=[]
    for arm in cfg['arms']:
        selected=pq.read_table(check(prep['artifacts'][arm+'/selected'])).to_pandas();n=5 if arm.endswith('_5y') else 10
        outcomes=pq.read_table(check(prep['artifacts'][f'outcomes{n}'])).to_pandas();monthly_refs={}
        for cost in (0,5,10,25):
            base=f'{arm}/monthly_cost{cost}/';mf={k:pq.read_table(check(arts[base+k])).to_pandas() for k in ('monthly','actions','failures')}
            verify_monthly(selected,outcomes,p['months'],cost,corrections,mf);monthly_refs[cost]=mf['monthly']
        for stage in cfg['stages']:
            guard();base=f'{arm}/{stage["key"]}/';names=('daily','trades','action_events','holdings','stops','failures')
            tables={k:pq.read_table(check(arts[base+k])).to_pandas() for k in names}
            hand=[] if stage['clock']=='signal_close' else p['execution_phase']['identity_handover_registry']['handovers']
            result=oracle(selected,frames[stage['view']],p['sessions'],p['execution_phase']['event_registry']['events'],p['execution_phase']['historical_retirements'],hand,stage,tables,guard)
            row=comp.loc[comp.arm.eq(arm)&comp.stage.eq(stage['key'])].iloc[0];need(bool(row.complete_path)==result['complete'],'Independent full-path availability')
            metrics,ends=independent_metrics(result['closes'],result['complete'])
            for k,w in metrics.items():number(row[k],w,'Independent aggregate '+k)
            saved=pq.read_table(check(arts[base+'close_wealth'])).to_pandas()
            if result['complete']:
                need(len(saved)==1569 and saved.date.astype(str).str[:10].tolist()==[x[0] for x in result['closes']],'Exact1569close dates')
                for got,want in zip(saved.wealth,result['closes']):number(got,want[1],'Independent saved curve')
                byyear={}
                for d,w in ends:byyear.setdefault(int(d[:4]),[]).append((d,w))
                previous=1.
                for year,points in byyear.items():
                    end=points[-1][1];annual.append(dict(arm=arm,stage=stage['key'],year=year,months=len(points),total_return=end/previous-1));previous=end
                if stage['key'] in ('cache_close_gross','cache_close_net','execution_close_gross','execution_close_open_units'):
                    reference=monthly_refs[stage['cost']];ref=dict(zip(reference.outcome_month,reference.nav_end))
                    for d,w in ends:
                        diff=w-ref[d[:7]];reconcile.append(dict(arm=arm,stage=stage['key'],outcome_month=d[:7],absolute_nav_difference=diff,matched=math.isclose(w,ref[d[:7]],rel_tol=1e-9,abs_tol=1e-10)))
                    # Only same cache/close successor convention is an equality claim.
                    if stage['key'] in ('cache_close_gross','cache_close_net'):need(all(math.isclose(w,ref[d[:7]],rel_tol=1e-9,abs_tol=1e-10) for d,w in ends),'Monthly returns do not translate to identical same-close wealth')
            refpin=p['reference_artifacts'].get(base+'close_wealth')
            if refpin:
                old=pq.read_table(check(refpin),columns=['date','wealth']).to_pandas();new=saved.loc[saved.date.le(pd.Timestamp('2023-12-29')),['date','wealth']]
                pd.testing.assert_frame_equal(old.reset_index(drop=True),new.reset_index(drop=True),check_exact=False,rtol=1e-9,atol=1e-10)
                prefix.append(dict(arm=arm,stage=stage['key'],observations=len(old),matched=True))
            scenarios.append(dict(arm=arm,stage=stage['key'],complete=result['complete'],first_failure=result['first_failure'],verified_marks=result['verified_marks'],stop_count=result['stops'],action_count=result['actions']))
            print(json.dumps(dict(verified=len(scenarios),maximum=48,arm=arm,stage=stage['key'])),flush=True)
    need(len(prefix)==28 and all(x['matched'] for x in prefix),'All28old controls retained')
    def table(name,records):
        import pyarrow as pa
        path=root/'verification'/(name+'.parquet');pq.write_table(pa.Table.from_pandas(pd.DataFrame(records),preserve_index=False),path);return pin(path)
    report={'comparison':arts['comparison'],'annual':table('annual',annual),'reconciliation':table('reconciliation',reconcile),'prefix_parity':table('prefix_parity',prefix)}
    complete=sum(x['complete'] for x in scenarios)
    return save(root/'verification/replay.json',dict(status='PASS',plan_pin=pin(plan_path),audit_pin=ap,input_proof_pin=ip,verification_read_plan_pin=vp,
        verified_scenarios=48,complete_paths=complete,withheld_paths=48-complete,monthly_references_verified=16,old_prefix_paths_unchanged=28,
        scenarios=scenarios,report_artifacts=report,new_fits=0,models_loaded=False,forecasts_changed=False,labels_changed=False,RET_changed=False,
        first_outcome='2020-02',last_outcome='2026-04',months=75,sessions=1569,worker_pin=pin(__file__)))
