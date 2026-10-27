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


def context(plan_path):
    p=read(plan_path);canonical=hashlib.sha256(json.dumps({k:v for k,v in p.items() if k!='plan_sha256'},sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
    need(p['plan_sha256']==canonical and p['maximum_fits']==0 and p['models_loaded'] is False and p['financial_2024_plus_read'] is False,'Independent zero-fit plan identity')
    for k in ('protocol_pin','monthly_plan_pin','monthly_proof_pin','monthly_audit_pin','source_plan_pin','independent_overlay_pin'):check(p[k])
    for group in ('code_pins','helper_pins','quote_cache','retention','selected','reference_artifacts'):
        for v in p[group].values():check(v)
    cfg=read(check(p['protocol_pin']));need(cfg['scenarios']==28 and len(cfg['stages'])==7 and len(cfg['arms'])==4,'Frozen28-path grid')
    source=read(check(p['source_plan_pin']));old=source['execution_phase'];current=p['execution_phase']
    need(current['historical_retirements']==old['historical_retirements'] and current['identity_handover_registry']==old['identity_handover_registry'],'Prior retirements/handover changed')
    need(current['event_registry']['events']==old['event_registry']['events']+p['additional_actions'] and len(p['additional_actions'])==2,'Only two legal additions')
    monthly=read(check(p['monthly_plan_pin']));terms=read(check(monthly['protocol_pin']))['events']
    for event in p['additional_actions']:
        legal=next(e for e in terms if e['predecessor_id']==event['predecessor_id'])
        need(event['verified'] is True and all(event[k]==legal[k] for k in ('effective_date','cash_per_raw_share','stock_ratio','successor_id')),'Changed legal entitlement')
    return p,cfg,Path(p['output_directory'])


def projection(spec):
    import pyarrow.parquet as pq
    pf=pq.ParquetFile(check(spec['pin']))
    cost=sum(pf.metadata.row_group(i).column(j).total_compressed_size for i in range(pf.metadata.num_row_groups)
        for j in range(pf.metadata.row_group(i).num_columns) if pf.metadata.row_group(i).column(j).path_in_schema in spec['columns'])
    need(cost==spec['projected_compressed_bytes'] and pf.metadata.num_rows==spec['rows'],'Independent projection reservation')
    return pf.read(columns=spec['columns'],use_threads=False)


def verify_inputs(plan_path,guard=lambda:None):
    import numpy as np,pandas as pd,pyarrow as pa,pyarrow.parquet as pq
    p,cfg,root=context(plan_path);receipt_pin=pin(root/'prepared/receipt.json');receipt=read(check(receipt_pin));arts=receipt['artifacts']
    need(receipt['plan_pin']==pin(plan_path) and receipt['new_fits']==0 and not receipt['labels_changed'] and not receipt['RET_changed'],'Unchanged learning scope')
    need(sum(v['size_bytes'] for v in arts.values())<=cfg['budgets']['derived_verification_bytes'],'Derived input read ceiling')
    vp=save(root/'verification/input_read_plan.json',dict(status='SEALED_BEFORE_INDEPENDENT_NUMERIC_READ',plan_pin=pin(plan_path),
       prepared_receipt_pin=receipt_pin,artifacts=arts,worker_pin=pin(__file__),new_fits=0,financial_2024_plus_read=False))
    for v in arts.values():check(v)
    cache=pq.read_table(check(arts['cache_prices'])).to_pandas()
    idx=read(check(p['quote_cache']['index']));di={d:i for i,d in enumerate(idx['sessions'])};ni={n:i for i,n in enumerate(idx['natives'])}
    a=np.load(check(p['quote_cache']['daily_prices']),mmap_mode='r',allow_pickle=False)
    q=np.load(check(p['quote_cache']['quote_validity']),mmap_mode='r',allow_pickle=False)
    need(cache.trading_date.isin(p['sessions']).all() and cache.intrinio_security_id.isin(p['natives']).all()
         and not cache.duplicated(['trading_date','intrinio_security_id']).any(),'Independent cache projection keys')
    expected_ids=set(p['natives'])&set(ni)
    need(set(cache.intrinio_security_id)==expected_ids and len(cache)==len(expected_ids)*len(p['sessions']),'Complete planned cache grid')
    for native,g in cache.groupby('intrinio_security_id',sort=True):
        guard();g=g.sort_values('trading_date');need(g.trading_date.tolist()==p['sessions'],'Exact cached dates')
        rows=[di[d] for d in p['sessions']];expected=a[rows,ni[native],:]
        actual=g[['open','close','adj_open','adj_close']].to_numpy(dtype=float)
        need(np.array_equal(actual,expected,equal_nan=True),'Cached OC differs from pinned source')
        flags=q[rows,ni[native],:]
        need(np.array_equal(g.open_accepted.to_numpy(),flags[:,0]) and np.array_equal(g.close_accepted.to_numpy(),flags[:,1])
             and np.array_equal(g.price_accepted.to_numpy(),flags[:,0]&flags[:,1]),'Independent quote acceptance')
    del a,q
    # Reconstruct the old forty overlays through the independently pinned historical checker.
    spec=importlib.util.spec_from_file_location('_reconciliation_independent_overlay',check(p['independent_overlay_pin']))
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    bt=p['execution_source'];sessions=set(p['sessions']);parts=[]
    for v in [*bt['execution_pins'].values(),*bt['execution_evidence_pins']]:check(v)
    for pair in bt['daily']:
        guard();metadata=projection(pair['metadata']);rows=metadata.to_pylist()
        need(all('2010-01-01'<=r['trading_date']<'2024-01-01' for r in rows),'Protected execution metadata')
        take=[i for i,r in enumerate(rows) if r['trading_date'] in sessions and r['trading_date'].startswith(str(pair['year']))]
        values=projection(pair['values']);need(values.select(metadata.column_names).equals(metadata),'Execution metadata binding')
        parts.append(values.take(pa.array(take,type=pa.int64())).to_pandas())
    metadata=projection(bt['repairs']['metadata']);need(all(r['trading_date'] in sessions for r in metadata.to_pylist()),'Exact overlay dates')
    values=projection(bt['repairs']['values']);need(values.select(metadata.column_names).equals(metadata),'Repair metadata changed')
    manifest=read(check(bt['execution_pins']['execution_manifest_pin']));hashes={v['sha256'] for v in bt['execution_evidence_pins']}
    def rejection(v):
        it=v['rejection_evidence_pin'];need(it['sha256'] in hashes,'Unbound rejection')
        check(dict(path=str((Path(p['data_root'])/it['path']).resolve()),sha256=it['sha256'],size_bytes=it.get('size_bytes',it.get('bytes'))))
    reconstructed,changes=mod.reconstruct_overlay(pd.concat(parts,ignore_index=True),values.to_pandas(),
        manifest['quote_amendments'],sessions=p['sessions'],authenticate_rejection=rejection)
    need(len(changes)==40,'Original forty overlays')
    actual=pq.read_table(check(arts['execution_prices'])).to_pandas()
    expected=reconstructed.loc[reconstructed.intrinio_security_id.isin(p['natives']),actual.columns]
    keys=['trading_date','intrinio_security_id']
    pd.testing.assert_frame_equal(actual.sort_values(keys).reset_index(drop=True),expected.sort_values(keys).reset_index(drop=True),check_dtype=False,check_exact=True)
    for arm in cfg['arms']:
        original=pq.read_table(check(p['selected'][arm]),columns=list(META)).to_pandas()
        original=original.loc[original.outcome_month.ge('2020-02')].sort_values(['signal_date','side','source_security_id'],kind='stable').reset_index(drop=True)
        selected=pq.read_table(check(arts[arm+'/selected'])).to_pandas()
        pd.testing.assert_frame_equal(original,selected,check_dtype=False,check_exact=True)
        need(len(selected)==940 and selected.signal_date.nunique()==47,'Unchanged47x20 baskets')
    return save(root/'verification/inputs.json',dict(status='PASS',plan_pin=pin(plan_path),prepared_receipt_pin=receipt_pin,
        verification_read_plan_pin=vp,quote_views_independently_reconstructed=True,selected_episodes=3760,
        artifacts=arts,new_fits=0,labels_changed=False,RET_changed=False,financial_2024_plus_read=False))


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
    need(len(closes)==985,'Independent985close observations')
    v=[x[1] for x in closes];returns=[v[0]-1]+[v[i]/v[i-1]-1 for i in range(1,len(v))]
    ends={}
    for d,value in closes:ends[d[:7]]=(d,value)
    months=list(ends.values());m=[x[1] for x in months];mr=[m[0]-1]+[m[i]/m[i-1]-1 for i in range(1,len(m))]
    def dd(values):
        peak=1.;low=0.
        for x in values:peak=max(peak,x);low=min(low,x/peak-1)
        return low
    sd=statistics.stdev(returns);ms=statistics.stdev(mr)
    return dict(total_return=v[-1]-1,cagr=v[-1]**(252/985)-1,
      daily_sharpe=statistics.mean(returns)/sd*math.sqrt(252) if sd else None,
      daily_volatility=sd*math.sqrt(252),daily_maximum_drawdown=dd(v),
      monthly_sharpe=statistics.mean(mr)/ms*math.sqrt(12) if ms else None,monthly_maximum_drawdown=dd(m)),months


def verify_replay(plan_path,guard=lambda:None):
    import pandas as pd,pyarrow.parquet as pq
    p,cfg,root=context(plan_path);ip=read(root/'verification/inputs.json');need(ip['status']=='PASS','Input proof before independent accounting')
    prep=read(check(ip['prepared_receipt_pin']));audit=read(root/'replay/audit.json');arts=audit['artifacts']
    need(audit['plan_pin']==pin(plan_path) and audit['selected']==p['selected'] and audit['retention']==p['retention'] and audit['new_fits']==0,'Original selections and learning retention')
    need(sum(v['size_bytes'] for v in arts.values())<=cfg['budgets']['derived_verification_bytes'],'Independent aggregate/ledger ceiling')
    vp=save(root/'verification/replay_read_plan.json',dict(status='SEALED_BEFORE_INDEPENDENT_REPLAY_READ',plan_pin=pin(plan_path),
       audit_pin=pin(root/'replay/audit.json'),artifacts=arts,worker_pin=pin(__file__),new_fits=0))
    for v in arts.values():check(v)
    prices={view:pq.read_table(check(prep['artifacts'][view+'_prices'])).to_pandas() for view in ('cache','execution')}
    comp=pq.read_table(check(arts['comparison'])).to_pandas();recon=pq.read_table(check(arts['reconciliation'])).to_pandas()
    need(len(comp)==28 and not comp.duplicated(['arm','stage']).any(),'Complete declared comparison grid')
    checked=[];reconciliation=[];phase=p['execution_phase']
    for arm in cfg['arms']:
        selected=pq.read_table(check(prep['artifacts'][arm+'/selected'])).to_pandas()
        for stage in cfg['stages']:
            guard();base=arm+'/'+stage['key']+'/'
            tables={n:pq.read_table(check(arts[base+n])).to_pandas() for n in ('daily','trades','holdings','failures','stops','action_events')}
            result=oracle(selected,prices[stage['view']],p['sessions'],phase['event_registry']['events'],
              phase['historical_retirements'],phase['identity_handover_registry']['handovers'],stage,tables,guard)
            metrics,months=independent_metrics(result['closes'],result['complete'])
            saved=comp.loc[comp.arm.eq(arm)&comp.stage.eq(stage['key'])];need(len(saved)==1,'Unique saved scenario');saved=saved.iloc[0]
            need(bool(saved.complete_path)==result['complete'],'Independent completeness')
            for name,value in metrics.items():number(saved[name],value,'Independent metric '+name)
            need(saved.stop_count==result['stops'] and saved.unfilled_retired_orders==result['unfilled_retired_orders'],'Stop/retired order counts')
            if stage['key'] in ('cache_close_gross','cache_close_net'):
                if result['complete']:
                    ref=pq.read_table(check(p['reference_artifacts'][f'{arm}/matched47/cost_{stage["cost"]:02d}bps/monthly'])).to_pandas().set_index('outcome_month')
                    for date,value in months:
                        old=float(ref.loc[date[:7],'nav_end']);matched=math.isclose(old,value,rel_tol=1e-9,abs_tol=1e-10)
                        row=recon.loc[recon.arm.eq(arm)&recon.stage.eq(stage['key'])&recon.outcome_month.eq(date[:7])]
                        need(len(row)==1 and bool(row.iloc[0].matched)==matched,'Independent reference reconciliation flag')
                        number(row.iloc[0].absolute_nav_difference,value-old,'Reference discrepancy')
                        reconciliation.append(dict(arm=arm,stage=stage['key'],outcome_month=date[:7],matched=matched,absolute_nav_difference=value-old))
                else:
                    row=recon.loc[recon.arm.eq(arm)&recon.stage.eq(stage['key'])]
                    need(len(row)==1 and not bool(row.iloc[0].matched),'Withheld reference control')
                    reconciliation.append(dict(arm=arm,stage=stage['key'],outcome_month=None,matched=False))
            checked.append(dict(arm=arm,stage=stage['key'],complete=result['complete'],verified_marks=result['verified_marks'],actions=result['actions'],stops=result['stops']))
            print(json.dumps(dict(verified=len(checked),maximum=28,new_fits=0)),flush=True)
    need(audit['complete_paths']==sum(r['complete'] for r in checked) and audit['withheld_paths']==sum(not r['complete'] for r in checked),'Coverage totals')
    return save(root/'verification/replay.json',dict(status='PASS',plan_pin=pin(plan_path),audit_pin=pin(root/'replay/audit.json'),
      prepared_input_proof_pin=pin(root/'verification/inputs.json'),verification_read_plan_pin=vp,
      verified_scenarios=28,scenarios=checked,reconciliation=reconciliation,
      all_close_controls_reconcile=all(r['matched'] for r in reconciliation),complete_paths=audit['complete_paths'],
      withheld_paths=audit['withheld_paths'],new_fits=0,producer_accounting_imported=False,
      forecasts_and_selections_unchanged=True,learning_inputs_unchanged=True,financial_2024_plus_read=False))
