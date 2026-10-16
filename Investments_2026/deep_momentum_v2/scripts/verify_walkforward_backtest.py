#!/usr/bin/env python3
"""Independent saved-result audit. No estimator, fitting, or accounting imports.

Decode only authenticated 2020--2023 execution quotes and saved forecasts after
key/date checks. Reconstruct each cash/units transition, signed entitlement,
physical net order, monthly stop episode, and metric directly from source values.
Outputs are new audit evidence; existing research outputs are never overwritten.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import itertools
import json
from math import isfinite
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ARMS=('rf_paper15','xgb_paper15','rf_extended','xgb_extended')
BASKETS={'class_1_10':(1,10),'class_2_9':(2,9)}
FEES=(0,5,10,25)
EXCLUDED={'sec_gjP4xM','sec_zmYNnz'}
PROBABILITIES=[f'probability_{k}' for k in range(1,11)]
PRICE_FIELDS=['trading_date','intrinio_security_id','open','close','adj_open','adj_close','adj_high','adj_low',
    'frequency','intraperiod','price_quality_status','raw_ohlc_bounds_valid','adjusted_ohlc_bounds_valid']


class VerificationError(RuntimeError):pass


def require(condition,message):
    if not bool(condition):raise VerificationError(message)


def number(actual,expected,message):
    if expected is None or pd.isna(expected):
        require(actual is None or pd.isna(actual),message)
    else:
        require(actual is not None, f'{message}: missing numeric value')
        observed,wanted=float(actual),float(expected)
        require(isfinite(observed) and abs(observed-wanted)<=2e-11+2e-9*abs(wanted),
            f'{message}: saved={actual!r}, independent={expected!r}')


def digest(path):
    h=sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def canonical(value):return 'sha256:'+sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def local(root,path):
    path=(Path(root)/path).resolve(); require(path.is_relative_to(Path(root).resolve()),'Artifact leaves declared root')
    return path


def pinned(root,item):
    require(isinstance(item,dict) and {'path','size_bytes','sha256'}<=set(item),'Missing artifact pin')
    path=local(root,item['path']); require(path.is_file(),'Pinned artifact absent')
    require(path.stat().st_size==item['size_bytes'] and digest(path)==item['sha256'],'Artifact hash/size changed: '+str(path))
    return path


def json_file(path):
    path=Path(path); require(path.stat().st_size<=16*1024**2,'Unbounded JSON read')
    return json.loads(path.read_text())


def iso(value):
    day=pd.Timestamp(value); require(not pd.isna(day) and day.tz is None and day==day.normalize(),'Invalid/intraday date')
    return day.strftime('%Y-%m-%d')


def key_table(path,*,kind,columns=None,maximum_rows=2_000_000):
    """Date/source keys first, independently of values and production loaders."""
    pf=pq.ParquetFile(path); names=set(pf.schema_arrow.names)
    require(pf.metadata.num_rows<=maximum_rows,'Bounded verification row limit')
    dates=[name for name in ('date','trading_date','signal_date','execution_date') if name in names]
    require(dates or pf.metadata.num_rows==0,'Saved financial table lacks timing keys')
    keys=pf.read(columns=dates).to_pandas()
    for column in dates:
        for value in keys[column].dropna():
            day=iso(value)
            low='2019-12-01' if column=='signal_date' else '2020-01-01'
            require(low<=day<='2023-12-31','Protected period decoded before financial projection: '+kind)
    # Only the explicit projection is read after all timing keys are checked.
    return pf.read(columns=columns).to_pandas()


def normalize(frame):
    frame=frame.copy()
    for column in ('date','signal_date','execution_date','period_start_date'):
        if column in frame:frame[column]=frame[column].map(lambda value:None if pd.isna(value) else iso(value))
    return frame


def group_records(frame):
    frame=normalize(frame)
    return {(day,mark):part.to_dict('records') for (day,mark),part in frame.groupby(['date','mark'],sort=False)} if not frame.empty else {}


def records_by_source(records,message):
    result={row['source_security_id']:row for row in records}
    require(len(result)==len(records),message+' duplicate source lot')
    return result


def verify_selection(forecasts,ledger,selected,monthly,windows,*,basket):
    """Recreate candidate counts, first-argmax ties and ranks from forecasts."""
    forecasts=normalize(forecasts); ledger=normalize(ledger); selected=normalize(selected); monthly=normalize(monthly)
    keys=['source_security_id','signal_date']; require(not forecasts.duplicated(keys).any(),'Duplicate forecast key')
    require(list(map(tuple,forecasts[keys].to_numpy()))==list(map(tuple,ledger[keys].to_numpy())),'Original cohort/order changed in selector ledger')
    for field in ('intrinio_security_id','prediction_eligible','paper_prediction_eligible'):
        require(field in forecasts and field in ledger,'Cohort admission field missing')
        # None/NaN native IDs on rejected rows compare equal.
        require(forecasts[field].fillna('<NA>').equals(ledger[field].fillna('<NA>')),'Selector changed original admission/identity '+field)
    require(set(forecasts.signal_date)==set(windows.signal_date),'Forecast month grid changed')
    expected_restricted=forecasts.intrinio_security_id.isin(EXCLUDED)
    expected_eligible=forecasts.prediction_eligible.astype(bool)&~expected_restricted
    require(ledger.execution_restricted.equals(expected_restricted) and ledger.trade_eligible.equals(expected_eligible),'New hindsight route exclusion or admission change')
    admitted=forecasts.prediction_eligible.astype(bool)
    p_all=forecasts.loc[admitted,PROBABILITIES].to_numpy(dtype=float)
    require(np.isfinite(p_all).all() and (p_all>=0).all() and np.allclose(p_all.sum(axis=1),1,rtol=1e-9,atol=1e-9),'Original forecast simplex changed')
    require(np.array_equal(ledger.loc[admitted,'predicted_class'].to_numpy(dtype=int),p_all.argmax(axis=1)+1),'Original class modes changed')
    require(len(monthly)==len(windows) and not monthly.signal_date.duplicated().any(),'Monthly breadth grid changed')
    require(not selected.duplicated(keys).any(),'Duplicate selected source')
    total=0; expected_all=[]
    for window in windows.to_dict('records'):
        signal=window['signal_date']; group=forecasts.loc[forecasts.signal_date.eq(signal)]
        eligible=group.prediction_eligible.astype(bool) & ~group.intrinio_security_id.isin(EXCLUDED)
        candidate=group.loc[eligible].copy(); p=candidate[PROBABILITIES].to_numpy(dtype=float)
        require(np.isfinite(p).all() and (p>=0).all() and np.allclose(p.sum(axis=1),1,rtol=1e-9,atol=1e-9),'Invalid signal probability simplex')
        candidate['mode']=p.argmax(axis=1)+1
        candidates={}
        for side,cls in zip(('long','short'),BASKETS[basket]):
            candidates[side]=candidate.loc[candidate['mode'].eq(cls)].sort_values(
                [f'probability_{cls}','source_security_id'],ascending=[False,True],kind='stable')
        n=min(10,len(candidates['long']),len(candidates['short']))
        row=monthly.loc[monthly.signal_date.eq(signal)].iloc[0]
        for side in ('long','short'):
            number(row[side+'_candidates'],len(candidates[side]),'Candidate count')
            number(row[side+'_selected'],n,'Balanced selected count')
            actual=selected.loc[selected.signal_date.eq(signal)&selected.side.eq(side)]
            expected=candidates[side].source_security_id.iloc[:n].tolist()
            require(sorted(actual.source_security_id.tolist())==sorted(expected),'Signal-only class rank/tie selection changed')
            require(actual.execution_date.eq(window['entry_date']).all(),'Next-session execution clock changed')
            for item in actual.to_dict('records'):
                original=candidates[side].loc[candidates[side].source_security_id.eq(item['source_security_id'])].iloc[0]
                number(item['selection_score'],original[f'probability_{BASKETS[basket][0 if side=="long" else 1]}'],'Chosen class probability')
                number(item['intended_weight'],(.5 if side=='long' else -.5)/n,'Chosen per-name weight')
            expected_all.extend((source,signal) for source in expected)
        number(row.selected_names_per_side,n,'Declared breadth')
        number(row.effective_per_name_abs_weight,None if n==0 else .5/n,'Declared per-asset weight')
        original=ledger.loc[ledger.signal_date.eq(signal)]
        require(set(original.loc[original.selected,'source_security_id'])==set(source for source,sig in expected_all if sig==signal),'Selection flag changed original ledger')
        total+=2*n
    require(len(selected)==total,'Extra selected rows outside monthly grid')
    return {'months':len(windows),'original_rows':len(forecasts),'selected_slots':total,'target_columns_read':False}


def _quote(quotes,day,native,field):
    require((day,native) in quotes,'Missing independently required held/source quote')
    row=quotes[(day,native)]
    require(row.get('frequency')=='daily' and row.get('intraperiod') in (False,np.bool_(False)) and
        row.get('price_quality_status') in ('accepted','ok'),'Rejected source quote used')
    for flag in ('raw_ohlc_bounds_valid','adjusted_ohlc_bounds_valid'):
        require(flag not in row or row[flag] in (True,np.bool_(True)),'Rejected OHLC bounds used')
    value=row.get(field); require(value is not None and isfinite(float(value)) and float(value)>0,'Invalid held price '+field)
    return float(value)


def verify_scenario(tables,prices,selections,rebalances,sessions,*,events=(),historical_retirements=(),cost_bps=0,stop_fraction=None,initial_nav=1.,prepared_quotes=None):
    """Audit an entire continuous path from source values, never call its engine.

    Supports a bounded synthetic subperiod for tests; the public saved-run entry
    point below additionally requires exactly 1006 sessions and 48 rebalances.
    """
    dates=[iso(day) for day in sessions]
    require(dates==sorted(set(dates)) and dates and '2020-01-01'<=dates[0]<=dates[-1]<='2023-12-31','Scenario escaped explicit period')
    if prepared_quotes is None:
        prices=prices.copy(); price_date='trading_date' if 'trading_date' in prices else 'date'
        prices[price_date]=prices[price_date].map(iso)
        require(prices[price_date].isin(dates).all(),'Protected/unrelated quote day')
        require(not prices.duplicated([price_date,'intrinio_security_id']).any(),'Duplicate execution quote')
        quotes={(row[price_date],row['intrinio_security_id']):row for row in prices.to_dict('records')}
    else:
        # Built once by the metadata-guarded saved-run entry point, not a worker.
        quotes=prepared_quotes
    daily=normalize(tables['daily']); selected=normalize(selections); schedule=normalize(rebalances)
    wanted=[(day,mark) for day in dates for mark in ('open','terminal_close' if day==dates[-1] else 'close')]
    require(list(zip(daily.date,daily.mark))==wanted,'Continuous daily mark grid changed')
    require(daily.status.eq('ok').all(),'Independent complete-path audit cannot certify failed exposure')
    require(tables['failures'].empty,'Complete result contains failures')
    grouped={name:group_records(tables[name]) for name in ('trades','holdings','stops','action_events','market_trades')}
    order_frame=normalize(tables['orders']); orders={day:part.to_dict('records') for day,part in order_frame.groupby('date')} if not order_frame.empty else {}
    selected_by_day={day:part.to_dict('records') for day,part in selected.groupby('execution_date')}
    schedule_by_day=dict(zip(schedule.execution_date,schedule.signal_date))
    require(set(schedule_by_day)==set(selected_by_day),'Zero-sided/absent selection cannot produce complete path')
    current=sorted([dict(event,effective_date=iso(event['effective_date'])) for event in events],key=lambda event:(event['effective_date'],event['event_id']))
    history=[dict(event,effective_date=iso(event['effective_date'])) for event in historical_retirements]
    require(all(event['effective_date']<'2020-01-01' for event in history),'Prior entitlement applied in new period')
    require(all(event.get('verified') is True and event.get('evidence_reference') for event in current),'Unverified action')
    require(all(event['effective_date'] in dates for event in current),'Wrong-period/non-session current action')
    require(len({event['event_id'] for event in current})==len(current) and len({(event['effective_date'],event['predecessor_id']) for event in current})==len(current),'Duplicate/conflicting current action')
    require(not ({event['predecessor_id'] for event in history}&{event['predecessor_id'] for event in current}),'Retired native reused by new security')
    for event in current:
        require(event.get('action_type','verified_retirement') in {'cash_merger','stock_merger','mixed_merger','verified_retirement'},'Unsupported action type')
        require(all(not isinstance(event.get(field),(bool,np.bool_)) and isinstance(event.get(field),(int,float,np.integer,np.floating)) and isfinite(float(event[field])) and event[field]>=0 for field in ('cash_per_raw_share','stock_ratio')),'Unsupported monetary entitlement')
        require(event['cash_per_raw_share']>0 or event['stock_ratio']>0,'Invented zero extinguishment')
        require((isinstance(event.get('successor_id'),str) and event['successor_id']!=event['predecessor_id']) if event['stock_ratio'] else event.get('successor_id') is None,'Wrong successor for entitlement')
        require(not any(other['predecessor_id']==event.get('successor_id') and other['effective_date']==event['effective_date'] for other in current),'Unsupported same-session action chain')
    state={}; last_prices={}; cash=initial_nav; previous_nav=initial_nav; previous_day=None; previous_mark=None
    event_count=stop_count=trade_count=0
    for row in daily.to_dict('records'):
        day,mark=row['date'],row['mark']; stamp=(day,mark); old=deepcopy(state)
        prior_value={source:p['units']*last_prices[source] for source,p in old.items()}
        signed_action_cash={'long':0.,'short':0.}; expected_events={}
        if mark=='open':
            for event in current:
                if event['effective_date']!=day:continue
                for source,position in list(state.items()):
                    if position['intrinio_security_id']!=event['predecessor_id']:continue
                    require(previous_day is not None and previous_mark=='close','Nonconsecutive action predecessor')
                    old_units=position['units']; basis=_quote(quotes,previous_day,event['predecessor_id'],'adj_close')/_quote(quotes,previous_day,event['predecessor_id'],'close')
                    entitlement=old_units*basis; action_cash=entitlement*event['cash_per_raw_share']; cash+=action_cash
                    signed_action_cash[position['side']]+=action_cash
                    new_units=0.; factor=None; opening=None
                    old_reference=position['entry_reference']; old_stop=position['stop_price']
                    if event['stock_ratio']:
                        opening=_quote(quotes,day,event['successor_id'],'adj_open')
                        factor=opening/_quote(quotes,day,event['successor_id'],'open')
                        new_units=entitlement*event['stock_ratio']/factor
                        position.update(intrinio_security_id=event['successor_id'],units=new_units,
                            entry_reference=(old_units*old_reference-action_cash)/new_units,
                            stop_price=None if old_stop is None else (old_units*old_stop-action_cash)/new_units)
                    else:del state[source]
                    expected_events[source]={'event_id':event['event_id'],'old_adjusted_units':old_units,
                        'predecessor_factor':basis,'signed_raw_entitlement':entitlement,'signed_cash':action_cash,
                        'cash_per_raw_share':event['cash_per_raw_share'],'stock_ratio':event['stock_ratio'],
                        'successor_factor':factor,'new_adjusted_units':new_units,'successor_open':opening,
                        'old_entry_reference':old_reference,'old_stop_price':old_stop,
                        'new_entry_reference':position['entry_reference'] if new_units else None,
                        'new_stop_price':position['stop_price'] if new_units else None}
        saved_events=records_by_source(grouped['action_events'].get(stamp,[]),'Action')
        require(set(saved_events)==set(expected_events),'Missing/extra signed held entitlement')
        for source,expected in expected_events.items():
            saved=saved_events[source]; definition=next(event for event in current if event['event_id']==expected['event_id'])
            require(saved['event_id']==expected.pop('event_id') and saved['predecessor_id']==definition['predecessor_id']
                and (pd.isna(saved['successor_id']) if definition.get('successor_id') is None else saved['successor_id']==definition['successor_id'])
                and saved['evidence_reference']==definition['evidence_reference'],'Wrong corporate action applied')
            for field,value in expected.items():number(saved_events[source][field],value,'Entitlement '+field)
        event_count+=len(expected_events)
        monthly=mark=='open' and day in schedule_by_day
        intended=selected_by_day.get(day,[]) if monthly else []
        executable=[]; saved_orders=records_by_source(orders.get(day,[]) if monthly else [],'Order')
        require(not monthly or set(saved_orders)=={item['source_security_id'] for item in intended},'Intended order slots changed')
        for item in intended:
            source=item['source_security_id']; retired=next((event for event in reversed(history+current) if event['predecessor_id']==item['intrinio_security_id'] and event['effective_date']<=day),None)
            order=saved_orders[source]
            require(order['signal_date']==item['signal_date'] and order['intrinio_security_id']==item['intrinio_security_id'] and order['side']==item['side'],'Intended order route/episode changed')
            require(order['status']==('unfilled_retired_security' if retired else 'executed'),'Retired order replaced/filled')
            require(pd.isna(order.get('retirement_event_id')) if retired is None else order.get('retirement_event_id')==retired['event_id'],'Retired event provenance changed')
            number(order['intended_weight'],(.5 if item['side']=='long' else -.5)/sum(member['side']==item['side'] for member in intended),'Unfilled slot denominator')
            if retired:number(order['new_adjusted_units'],0.,'Retired order received units')
            else:executable.append(item)
        needed=deepcopy(state)
        for item in executable:
            source=item['source_security_id']
            require(source not in needed or needed[source]['intrinio_security_id']==item['intrinio_security_id'],'Held identity changed')
            needed.setdefault(source,item)
        marks={source:_quote(quotes,day,position['intrinio_security_id'],'adj_open' if mark=='open' else 'adj_close') for source,position in needed.items()}
        stop_fills={}
        if stop_fraction is not None:
            for source,position in state.items():
                threshold=position['stop_price']
                if threshold is None:continue
                if mark=='open' and not monthly:
                    crossed=marks[source]<=threshold if position['side']=='long' else marks[source]>=threshold
                    if crossed:stop_fills[source]='gap_open'
                elif mark!='open':
                    low=_quote(quotes,day,position['intrinio_security_id'],'adj_low'); high=_quote(quotes,day,position['intrinio_security_id'],'adj_high')
                    opening=_quote(quotes,day,position['intrinio_security_id'],'adj_open')
                    require(low<=min(opening,marks[source])<=max(opening,marks[source])<=high,'Invalid held source OHLC')
                    crossed=low<=threshold if position['side']=='long' else high>=threshold
                    if crossed:stop_fills[source]='idealized_intraday_threshold'; marks[source]=threshold
        saved_stops=records_by_source(grouped['stops'].get(stamp,[]),'Stop')
        require(set(saved_stops)==set(stop_fills),'Missing/extra 10% asset stop')
        for source,fill in stop_fills.items():
            saved=saved_stops[source]; require(saved['fill_type']==fill,'Wrong gap/intraday fill assumption')
            number(saved['stop_price'],state[source]['stop_price'],'Stop threshold'); number(saved['fill_price'],marks[source],'Stop fill')
            number(saved['units_before'],state[source]['units'],'Stop units')
        stop_count+=len(stop_fills)
        pretrade=cash+sum(position['units']*marks[source] for source,position in state.items())
        target=None; phase=None
        if monthly:
            target={}; phase='rebalance'
            for item in executable:
                source=item['source_security_id']; weight=(.5 if item['side']=='long' else -.5)/sum(member['side']==item['side'] for member in intended)
                price=marks[source]; threshold=None if stop_fraction is None else price+(-price*stop_fraction if item['side']=='long' else price*stop_fraction)
                target[source]={**item,'units':weight*pretrade/price,'entry_reference':price,'stop_price':threshold}
            for item in intended:
                order=saved_orders[item['source_security_id']]; number(order['target_notional'],order['intended_weight']*pretrade,'Monthly pre-cost target capital')
                if order['status']=='executed':number(order['new_adjusted_units'],target[item['source_security_id']]['units'],'Order target units')
        elif stop_fills:target={source:deepcopy(position) for source,position in state.items() if source not in stop_fills}; phase='stop_gap' if mark=='open' else 'stop_intraday'
        if mark=='terminal_close':target={}; phase='terminal_liquidation'
        instructions={}; physical={}
        if target is not None:
            for source in sorted(set(state)|set(target)):
                before=state.get(source,{}).get('units',0.); after=target.get(source,{}).get('units',0.); delta=after-before
                if delta==0:continue
                position=target.get(source,state.get(source)); notional=delta*marks[source]
                bucket=source if source in stop_fills and mark!='open' else 'simultaneous_mark'
                grouping=(position['intrinio_security_id'],float(marks[source]),bucket)
                physical.setdefault(grouping,[]).append(source)
                instructions[source]={'units_before':before,'units_after':after,'units_delta':delta,'price':marks[source],
                    'signed_notional':notional,'absolute_notional':abs(notional),'group':grouping,
                    'phase':('stop_gap' if mark=='open' else 'stop_intraday') if source in stop_fills else phase}
        saved_trades=records_by_source(grouped['trades'].get(stamp,[]),'Trade')
        require(set(saved_trades)==set(instructions),f'Missing/extra market lot trade at {stamp}: saved={set(saved_trades)}, expected={set(instructions)}')
        market=grouped['market_trades'].get(stamp,[])
        saved_physical={(item['intrinio_security_id'],float(item['price']),item['execution_bucket']):item for item in market}
        require(len(saved_physical)==len(market) and set(saved_physical)==set(physical),'Physical net order groups changed')
        cost_total=notional_total=0.
        for grouping in sorted(physical):
            sources=physical[grouping]
            signed=sum(instructions[source]['signed_notional'] for source in sources); absolute=abs(signed)
            gross=sum(instructions[source]['absolute_notional'] for source in sources); cost=absolute*cost_bps/10000
            saved=saved_physical[grouping]
            physical_phase=('stop_gap' if mark=='open' else 'stop_intraday') if all(source in stop_fills for source in sources) else phase
            require(saved['phase']==physical_phase,'Physical execution phase changed')
            for field,value in [('signed_notional',signed),('absolute_notional',absolute),('cost',cost),('lot_instruction_notional',gross),('internally_netted_notional',gross-absolute),('source_lot_count',len(sources))]:number(saved[field],value,'Physical net '+field)
            cash-=signed+cost; cost_total+=cost; notional_total+=absolute
            for source in sources:
                instruction=instructions[source]; saved=saved_trades[source]
                require(saved['phase']==instruction.pop('phase'),'Unscheduled/yearly liquidation or wrong trade phase')
                instruction.pop('group')
                for field,value in instruction.items():number(saved[field],value,'Lot trade '+field)
                number(saved['cost'],cost*instruction['absolute_notional']/gross if gross else 0.,'Allocated physical fee')
                if source in saved_stops:number(saved_stops[source]['cost'],saved['cost'],'Stop fee allocation')
        trade_count+=len(instructions)
        if target is not None:state=target
        saved_holdings=records_by_source(grouped['holdings'].get(stamp,[]),'Holding')
        active={source:item for source,item in saved_holdings.items() if item['units']!=0}
        require(set(active)==set(state),'Held slots replaced/dropped')
        for source,position in state.items():
            saved=active[source]; require(saved['intrinio_security_id']==position['intrinio_security_id'] and saved['side']==position['side'] and saved['signal_date']==position['signal_date'],'Held successor route/episode changed')
            for field,value in [('units',position['units']),('valuation_price',marks[source]),('market_value',position['units']*marks[source]),('entry_reference',position['entry_reference']),('stop_price',position['stop_price'])]:number(saved[field],value,'Holding '+field)
        for source,item in saved_holdings.items():
            if item['units']==0 and pd.notna(item['market_value']):number(item['market_value'],0.,'Closed lot market value')
        long_value=sum(position['units']*marks[source] for source,position in state.items() if position['units']>0)
        short_value=sum(position['units']*marks[source] for source,position in state.items() if position['units']<0)
        nav=cash+long_value+short_value
        # Attribute price movement using post-action PRE-trade units, independently
        # of target scaling, stops removed after the mark, and cash transaction.
        before_trade=deepcopy(old)
        for source in expected_events:
            if source in needed:before_trade[source]=needed[source]
            else:before_trade.pop(source,None)
        long_pnl=sum(position['units']*marks[source] for source,position in before_trade.items() if position['side']=='long')-sum(value for source,value in prior_value.items() if old[source]['side']=='long')+signed_action_cash['long']
        short_pnl=sum(position['units']*marks[source] for source,position in before_trade.items() if position['side']=='short')-sum(value for source,value in prior_value.items() if old[source]['side']=='short')+signed_action_cash['short']
        fields={'nav_before_cost':pretrade,'nav':nav,'cash':cash,'long_market_value':long_value,'short_market_value':short_value,
            'return':nav/previous_nav-1,'long_price_pnl':long_pnl,'short_price_pnl':short_pnl,'price_pnl':long_pnl+short_pnl,
            'trading_cost':cost_total,'borrow_cost':0.,'net_pnl':nav-previous_nav,'traded_notional':notional_total,
            'turnover_one_way':notional_total/pretrade,'gross_exposure':(long_value-short_value)/nav,
            'net_exposure':(long_value+short_value)/nav,'long_count':sum(p['units']>0 for p in state.values()),'short_count':sum(p['units']<0 for p in state.values())}
        for field,value in fields.items():number(row[field],value,'Continuous ledger '+field)
        number(nav-previous_nav,long_pnl+short_pnl-cost_total,'Independent price/cash reconciliation')
        require(bool(row['rebalance'])==monthly,'Extra annual rebalance')
        require((pd.isna(row['period_start_date']) if previous_day is None else row['period_start_date']==previous_day)
            and (pd.isna(row['period_start_mark']) if previous_mark is None else row['period_start_mark']==previous_mark),'Ledger continuity/year boundary changed')
        previous_nav,previous_day,previous_mark=nav,day,mark
        last_prices={source:marks[source] for source in state}
    require(not state,'Final authorized close not liquidated')
    for name,groups in grouped.items():require(set(groups)<=set(wanted),'Extra accounting marks: '+name)
    require(set(orders)<=set(schedule_by_day),'Orders outside monthly rebalances')
    return {'sessions':len(dates),'marks':len(daily),'final_nav':previous_nav,'trade_count':trade_count,
        'stop_count':stop_count,'action_count':event_count,'continuous_cash_units_reconciled':True}


def verify_failed_scenario(tables,prices,selections,sessions,*,events=(),historical_retirements=(),cost_bps=0,prepared_quotes=None):
    """Certify withheld performance and preserved exposures, never a return.

    A failure is an observable source/selection/economic condition. Failed marks
    must remain unvalued and the remaining lot state must persist unchanged.
    Only independently verified actions and already executed transactions at the
    first failure may explain a state change. No later invented exit is allowed.
    """
    dates=[iso(day) for day in sessions]; daily=normalize(tables['daily'])
    wanted=[(day,mark) for day in dates for mark in ('open','terminal_close' if day==dates[-1] else 'close')]
    require(list(zip(daily.date,daily.mark))==wanted,'Failed path lost continuous mark grid')
    blocked=daily.status.ne('ok'); require(blocked.any(),'Withheld scenario lacks actual failed mark')
    first=int(np.flatnonzero(blocked.to_numpy())[0]); require(blocked.iloc[first:].all() and daily.status.iloc[first:].eq('failed_closed').all(),'Failed path silently restarted or changed failure state')
    stamp=(daily.iloc[first].date,daily.iloc[first].mark); grouped={name:group_records(tables[name]) for name in ('trades','holdings','stops','action_events','market_trades')}
    failures=normalize(tables['failures']); require(not failures.empty,'Withheld path lacks failure evidence')
    require(set(zip(failures.date,failures.mark))=={stamp},'Failure chronology changed')
    if prepared_quotes is None:
        price_date='trading_date' if 'trading_date' in prices else 'date'
        quotes={(iso(row[price_date]),row['intrinio_security_id']):row for row in prices.to_dict('records')}
    else:quotes=prepared_quotes
    previous_stamp=wanted[first-1] if first else None
    state=records_by_source(grouped['holdings'].get(previous_stamp,[]),'Last valued holding') if first else {}
    state={source:dict(position) for source,position in state.items() if position['units']!=0}
    current={event['event_id']:event for event in events}; cash=float(daily.iloc[first-1].cash) if first else 1.
    for action in grouped['action_events'].get(stamp,[]):
        source=action['source_security_id']; require(source in state,'Failed-mark action invented a prior holding')
        require(stamp[1]=='open' and action['event_id'] in current,'Failed-mark action has no verified definition')
        definition=current[action['event_id']]; position=state[source]
        require(definition['verified'] is True and iso(definition['effective_date'])==stamp[0]
            and definition['predecessor_id']==position['intrinio_security_id'],'Failed-mark action changed dated native identity')
        require(previous_stamp is not None and previous_stamp[1]=='close','Failed-mark event has no preceding close basis')
        factor=_quote(quotes,previous_stamp[0],position['intrinio_security_id'],'adj_close')/_quote(quotes,previous_stamp[0],position['intrinio_security_id'],'close')
        raw=position['units']*factor; signed_cash=raw*definition['cash_per_raw_share']; cash+=signed_cash
        for field,value in [('old_adjusted_units',position['units']),('predecessor_factor',factor),('signed_raw_entitlement',raw),('signed_cash',signed_cash)]:number(action[field],value,'Failed-mark entitlement '+field)
        if definition['stock_ratio']:
            successor=definition['successor_id']; sf=_quote(quotes,stamp[0],successor,'adj_open')/_quote(quotes,stamp[0],successor,'open')
            units=raw*definition['stock_ratio']/sf; number(action['new_adjusted_units'],units,'Failed-mark successor units')
            position.update(intrinio_security_id=successor,units=units,
                entry_reference=(position['units']*position['entry_reference']-signed_cash)/units,
                stop_price=None if pd.isna(position['stop_price']) else (position['units']*position['stop_price']-signed_cash)/units)
        else:
            number(action['new_adjusted_units'],0.,'Failed-mark cash retirement units'); del state[source]
    before_first_trades=deepcopy(state)
    reasons=set(failures.reason)
    intended=normalize(selections)
    order_frame=normalize(tables['orders'])
    require(order_frame.empty or order_frame.date.le(stamp[0]).all(),'Orders created after blocked exposure')
    expected_order_dates=set(intended.loc[intended.execution_date.lt(stamp[0]) | (intended.execution_date.eq(stamp[0]) & (stamp[1] in ('open','close','terminal_close'))),'execution_date'])
    actual_order_dates=set(order_frame.date) if not order_frame.empty else set()
    require(actual_order_dates==expected_order_dates,'Failed path dropped intended order episodes before failure')
    for order_day in actual_order_dates:
        planned=records_by_source(intended.loc[intended.execution_date.eq(order_day)].to_dict('records'),'Failed-path intended order')
        saved=records_by_source(order_frame.loc[order_frame.date.eq(order_day)].to_dict('records'),'Failed-path order')
        require(set(saved)==set(planned),'Failed path replaced/dropped intended order slot')
        for source,item in planned.items():
            order=saved[source]; retired=next((event for event in reversed([*historical_retirements,*events]) if event['predecessor_id']==item['intrinio_security_id'] and iso(event['effective_date'])<=order_day),None)
            expected_status='unfilled_retired_security' if retired else 'pending_validation' if order_day==stamp[0] and stamp[1]=='open' and 'nonpositive_or_nonfinite_nav_after_cost' not in reasons else 'executed'
            require(order['status']==expected_status and order['intrinio_security_id']==item['intrinio_security_id'] and order['side']==item['side'],'Failed-path order route/status changed')
            number(order['intended_weight'],(.5 if item['side']=='long' else -.5)/sum(member['side']==item['side'] for member in planned.values()),'Failed-path intended denominator')
    first_trades=grouped['trades'].get(stamp,[])
    require(not first_trades or 'nonpositive_or_nonfinite_nav_after_cost' in reasons,'Unpriced failure created a transaction')
    for trade in first_trades:
        source=trade['source_security_id']; before=state.get(source,{}).get('units',0.)
        number(trade['units_before'],before,'Failed-mark actual prior units')
        number(trade['units_after']-trade['units_before'],trade['units_delta'],'Failed-mark actual unit delta')
        number(trade['signed_notional'],trade['units_delta']*trade['price'],'Failed-mark actual cash trade')
        if trade['units_after']==0:state.pop(source,None)
        else:state[source]=dict(trade,units=trade['units_after'],side='long' if trade['units_after']>0 else 'short')
    for market in grouped['market_trades'].get(stamp,[]):
        number(market['absolute_notional'],abs(market['signed_notional']),'Failed-mark physical notional')
        number(market['cost'],abs(market['signed_notional'])*cost_bps/10000,'Failed-mark physical fee')
        cash-=market['signed_notional']+market['cost']
    expected=state
    for index in range(first,len(daily)):
        row=daily.iloc[index]; key=(row.date,row.mark)
        for field in ('nav_before_cost','nav','return','cash','long_market_value','short_market_value','gross_exposure','net_exposure',
            'long_price_pnl','short_price_pnl','price_pnl','trading_cost','borrow_cost','net_pnl','traded_notional','turnover_one_way'):
            require(pd.isna(row[field]),'Failed exposure received an invented mark/return: '+field)
        holdings=records_by_source(grouped['holdings'].get(key,[]),'Retained failed holding')
        require(set(holdings)==set(expected),'Failed holding dropped or replaced')
        for source,position in expected.items():
            holding=holdings[source]
            require(holding['status']=='retained_unvalued_after_failure' and holding['intrinio_security_id']==position['intrinio_security_id'],'Failed holding was silently closed/rerouted')
            for field in ('units','entry_reference','stop_price'):number(holding[field],position[field],'Retained unvalued '+field)
            require(pd.isna(holding['valuation_price']) and pd.isna(holding['market_value']),'Failed holding received fabricated price')
        if index>first:
            for name in ('trades','stops','action_events','market_trades'):require(not grouped[name].get(key),'Transaction/stop/action after a failed exposure')
    day,mark=stamp
    for failure in failures.to_dict('records'):
        quote=quotes.get((day,failure['intrinio_security_id'])); reason=failure['reason']; field=failure.get('price_field')
        if reason not in ('empty_long_or_short_leg','nonpositive_or_nonfinite_nav','nonpositive_or_nonfinite_nav_after_cost'):
            source=failure['source_security_id']; native=failure['intrinio_security_id']; role=failure.get('role')
            held=before_first_trades.get(source); planned=intended.loc[intended.execution_date.eq(day)&intended.source_security_id.eq(source)]
            possible_action=any(iso(event['effective_date'])==day and (held is not None and held['intrinio_security_id'] in (event['predecessor_id'],event.get('successor_id'))) and native in (event['predecessor_id'],event.get('successor_id')) for event in events)
            valid_route=(held is not None and held['intrinio_security_id']==native) or (mark=='open' and len(planned)==1 and planned.iloc[0].intrinio_security_id==native) or (role in ('action','action_successor') and possible_action)
            require(valid_route,'Failure attributed to an unrelated/unselected native')
        if reason=='missing_quote':require(quote is None,'Missing-quote failure contradicted by execution source')
        elif reason=='source_quality_flag':
            bad=quote is None or quote.get('frequency')!='daily' or quote.get('intraperiod') not in (False,np.bool_(False)) or quote.get('price_quality_status') not in ('accepted','ok') or any(flag in quote and quote[flag] not in (True,np.bool_(True)) for flag in ('adjusted_ohlc_bounds_valid','raw_ohlc_bounds_valid'))
            require(bad,'Quality failure contradicted by source metadata')
        elif reason=='invalid_price':require(quote is not None and (quote.get(field) is None or not isfinite(float(quote[field])) or float(quote[field])<=0),'Invalid-price failure contradicted by source')
        elif reason=='invalid_adjusted_ohlc_bounds':require(quote is not None and not quote['adj_low']<=min(quote['adj_open'],quote['adj_close'])<=max(quote['adj_open'],quote['adj_close'])<=quote['adj_high'],'OHLC failure contradicted by source')
        elif reason=='missing_event_predecessor_basis':
            previous=quotes.get((previous_stamp[0],failure['intrinio_security_id'])) if previous_stamp else None
            require(previous is None or any(previous.get(name) is None or not isfinite(float(previous[name])) or float(previous[name])<=0 for name in ('close','adj_close')),'Missing predecessor basis contradicted by source')
        elif reason=='missing_event_successor_basis':require(quote is None or quote.get('open') is None or not isfinite(float(quote['open'])) or float(quote['open'])<=0,'Missing successor basis contradicted by source')
        elif reason=='empty_long_or_short_leg':
            intended=normalize(selections); sides=set(intended.loc[intended.execution_date.eq(day),'side'])
            require(sides!={'long','short'},'Empty-side failure contradicted by unchanged selection')
        elif reason in ('nonpositive_or_nonfinite_nav','nonpositive_or_nonfinite_nav_after_cost'):
            marks={source:_quote(quotes,day,position['intrinio_security_id'],'adj_open' if mark=='open' else 'adj_close') for source,position in expected.items()}
            monthly=mark=='open' and day in set(normalize(selections).execution_date)
            for source,position in expected.items():
                threshold=position.get('stop_price')
                if pd.isna(threshold) or monthly:continue
                if mark=='open':crossed=marks[source]<=threshold if position['side']=='long' else marks[source]>=threshold
                else:
                    adverse=_quote(quotes,day,position['intrinio_security_id'],'adj_low' if position['side']=='long' else 'adj_high')
                    crossed=adverse<=threshold if position['side']=='long' else adverse>=threshold
                if crossed and mark!='open':marks[source]=threshold
            nav=cash+sum(position['units']*marks[source] for source,position in expected.items())
            require(not isfinite(nav) or nav<=0,'Nonpositive-NAV failure contradicted by actual holdings/cash')
        else:raise VerificationError('Unsupported failure requires independent review: '+str(reason))
    # Every previously valued mark still reconciles to its own cash/holdings and
    # prior NAV, without converting that prefix into a full-period performance.
    previous_nav=1.
    for row in daily.iloc[:first].to_dict('records'):
        holdings=grouped['holdings'].get((row['date'],row['mark']),[])
        value=sum(item['market_value'] for item in holdings if pd.notna(item['market_value']))
        number(row['nav'],row['cash']+value,'Valid-prefix cash/holdings bridge')
        number(row['return'],row['nav']/previous_nav-1,'Valid-prefix return bridge')
        number(row['net_pnl'],row['nav']-previous_nav,'Valid-prefix NAV bridge'); previous_nav=row['nav']
    return {'sessions':len(dates),'marks':len(daily),'first_failure_date':day,'first_failure_mark':mark,
        'withheld_marks':len(daily)-first,'retained_unvalued_lots':len(expected),
        'economic_performance_available':False,'failure_exposures_preserved':True}


def verify_failed_metrics(daily,trades,monthly,summary,annual):
    daily=normalize(daily); closes=daily.drop_duplicates('date',keep='last')
    require(summary['status']=='failed_closed' and not summary['complete_proxy_path'],'Failed path claimed complete performance')
    for field in ('final_nav','total_return','annualized_arithmetic_mean','cagr','annualized_volatility','sharpe_zero_cash_rate',
        'maximum_drawdown','total_one_way_turnover','trading_cost_paid'):require(pd.isna(summary[field]),'Full-period metric invented for failed exposure: '+field)
    available=closes.assign(month=closes.date.str[:7]).groupby('month').nav.apply(lambda values:values.notna().all())
    expected=closes.assign(month=closes.date.str[:7]).groupby('month').nav.last(); expected.loc[~available]=np.nan
    require(monthly.outcome_month.tolist()==expected.index.tolist(),'Withheld monthly grid changed')
    require(np.allclose(monthly.nav,expected.to_numpy(),equal_nan=True),'Withheld monthly NAV fabricated')
    values=expected.to_numpy(); require(np.allclose(monthly['return'],values/np.r_[1.,values[:-1]]-1,equal_nan=True),'Withheld monthly return fabricated')
    for field in ('initial_nav','final_nav','total_return','annualized_volatility','sharpe_zero_cash_rate','maximum_drawdown','trading_cost_paid'):
        require(annual[field].isna().all(),'Annual metric invented for incomplete full path: '+field)
    require(annual.complete_full_period_proxy.eq(False).all(),'Failed path annual report claimed completeness')


def independent_clock():
    import exchange_calendars as xc
    calendar=xc.get_calendar('XNYS',start='2019-12-01',end='2023-12-31')
    all_days=calendar.sessions.strftime('%Y-%m-%d').tolist()
    dates=[day for day in all_days if day>='2020-01-01']
    require(len(dates)==1006 and dates[0]=='2020-01-02' and dates[-1]=='2023-12-29','Independent session count changed')
    months={month:[day for day in all_days if day[:7]==month] for month in sorted({day[:7] for day in all_days})}
    windows=[]
    for month in pd.period_range('2020-01','2023-12',freq='M'):
        previous=str(month-1); current=str(month)
        windows.append({'signal_date':months[previous][-1],'entry_date':months[current][0],
            'outcome_month':current,'outcome_start':months[current][0],'outcome_end':months[current][-1]})
    return dates,pd.DataFrame(windows)


def verify_metrics(daily,trades,monthly,summary,annual,*,initial_nav=1.):
    """Recalculate daily/monthly/annual statistics from the continuous close NAV."""
    daily=normalize(daily); trades=normalize(trades); closes=daily.drop_duplicates('date',keep='last')
    nav=closes.nav.to_numpy(dtype=float); returns=nav/np.r_[initial_nav,nav[:-1]]-1
    sd=float(np.std(returns,ddof=1)); peak=np.maximum.accumulate(np.r_[initial_nav,nav])[1:]
    expected={'initial_nav':initial_nav,'final_nav':nav[-1],'total_return':nav[-1]/initial_nav-1,
        'annualized_arithmetic_mean':returns.mean()*252,'cagr':(nav[-1]/initial_nav)**(252/len(nav))-1,
        'annualized_volatility':sd*np.sqrt(252),'sharpe_zero_cash_rate':None if sd==0 else returns.mean()/sd*np.sqrt(252),
        'maximum_drawdown':(nav/peak-1).min(),'total_one_way_turnover':daily.turnover_one_way.sum(),
        'trading_cost_paid':trades.cost.sum(),'valid_close_sessions':len(nav)}
    for field,value in expected.items():number(summary[field],value,'Full-period metric '+field)
    require(summary['complete_proxy_path'] and summary['status']=='complete_proxy','Incomplete full-period metric')
    expected_monthly=closes.assign(outcome_month=closes.date.str[:7]).groupby('outcome_month',sort=True).nav.last()
    require(monthly.outcome_month.tolist()==expected_monthly.index.tolist(),'Monthly return grid changed')
    require(np.allclose(monthly.nav,expected_monthly.to_numpy(),rtol=1e-10,atol=1e-11),'Monthly closing NAV')
    require(np.allclose(monthly['return'],expected_monthly.to_numpy()/np.r_[initial_nav,expected_monthly.to_numpy()[:-1]]-1,rtol=1e-10,atol=1e-11),'Monthly compounded return')
    previous=initial_nav
    for year,part in closes.groupby(closes.date.str[:4],sort=True):
        rows=annual.loc[annual.year.eq(int(year))]; require(len(rows)==1,'Calendar-year result grid changed')
        saved=rows.iloc[0]; yearly_nav=part.nav.to_numpy(dtype=float); yearly_returns=yearly_nav/np.r_[previous,yearly_nav[:-1]]-1
        yearly_sd=float(np.std(yearly_returns,ddof=1)); peak=np.maximum.accumulate(np.r_[previous,yearly_nav])[1:]
        for field,value in {'initial_nav':previous,'final_nav':yearly_nav[-1],'total_return':yearly_nav[-1]/previous-1,
            'expected_sessions':len(part),'annualized_volatility':yearly_sd*np.sqrt(252),
            'sharpe_zero_cash_rate':None if yearly_sd==0 else yearly_returns.mean()/yearly_sd*np.sqrt(252),
            'maximum_drawdown':(yearly_nav/peak-1).min(),'trading_cost_paid':trades.loc[trades.date.str[:4].eq(year),'cost'].sum()}.items():number(saved[field],value,'Continuous calendar-year '+field)
        previous=yearly_nav[-1]
    require(len(annual)==closes.date.str[:4].nunique(),'Extra calendar-year results')


def _quote_inputs(root,run,phase,sessions):
    # Annual inputs are exactly this run's four execution caches. They are not
    # replaced by a convenient later dataset, and supplements remain append-only.
    pins={item['path']:item for item in phase['source_pins']}
    paths=[]; original_ids=set()
    for year in range(2020,2024):
        directory=run/'years'/str(year); audit=json_file(directory/'audit.json')
        require(audit.get('status')=='COMPLETE' and audit.get('year')==year and audit.get('build_identity_sha256')==phase['parent_identity_sha256'],'Annual execution audit identity/year changed')
        expected=(directory/'daily_prices_raw.parquet').relative_to(root).as_posix()
        require(expected in pins,'Execution annual cache missing from backtest source pins')
        path=pinned(root,pins[expected]); apath=directory/'audit.json'
        require(apath.relative_to(root).as_posix() in pins,'Annual execution audit not pinned')
        pinned(root,pins[apath.relative_to(root).as_posix()])
        original_ids.update(audit['feature_provenance']['requested_ids'])
        pf=pq.ParquetFile(path); keys=pf.read(columns=['trading_date','intrinio_security_id']).to_pandas()
        require(keys.trading_date.map(lambda day:isinstance(day,str) and f'{year}-01-01'<=day<=f'{year}-12-31').all(),'Annual quote key escapes declared year')
        require(keys.trading_date.isin(sessions).all(),'Annual quote keys escape independent XNYS calendar')
        original_ids.update(keys.intrinio_security_id); paths.append(path)
    registry=phase['event_registry']; successors={event['successor_id'] for event in registry['events'] if event.get('successor_id')}
    for source in registry.get('supplemental_quote_sources',[]):
        path=pinned(root,source['pin']); require(path.is_relative_to(root/'data/derived') and path.suffix=='.parquet','Execution supplement outside private derived namespace')
        ids=set(source['allowed_native_ids']); require(ids and ids<=successors and not ids&original_ids,'Supplement extends a feature-requested identity')
        require('2020-01-01'<=source['start']<=source['end']<='2023-12-31','Supplement outside explicit period')
        keys=pq.ParquetFile(path).read(columns=['trading_date','intrinio_security_id']).to_pandas()
        require(set(keys.intrinio_security_id)<=ids and keys.trading_date.between(source['start'],source['end']).all() and keys.trading_date.isin(sessions).all(),'Supplement exact native/date bounds violated')
        paths.append(path)
    require(sum(path.stat().st_size for path in paths)<=512*1024**2,'Independent execution input file budget')
    # All four annual metadata key scans finish before any OHLC projection.
    parts=[key_table(path,kind='execution source',columns=PRICE_FIELDS) for path in paths]
    prices=pd.concat(parts,ignore_index=True); require(not prices.duplicated(['trading_date','intrinio_security_id']).any(),'Quote overwrite across annuals/supplements')
    return prices


def verify_saved_run(root,run,backtest,*,output=None):
    """Metadata-authenticate the frozen scope, then independently audit 64 paths."""
    import platform
    import resource
    started=time.monotonic(); root=Path(root).resolve(); run=local(root,Path(run).resolve()); backtest=local(run,Path(backtest).resolve())
    if output is not None:
        output=local(run,Path(output).resolve()); require(not output.exists(),'Preserve existing independent verification audit')
    identity=json_file(run/'build_identity.json'); identity_hash=canonical(identity); cfg=identity['configuration']; ex=cfg['execution']
    require(ex.get('verification_model_fits')==0 and ex.get('financial_2020_2023_analysis_authorized') is True and ex.get('financial_2024_plus_analysis') is False,'Verification has no frozen zero-fit outer-period authorization')
    wall=ex.get('verification_maximum_wall_seconds'); memory=ex.get('verification_maximum_peak_rss_bytes')
    require(type(wall)is int and 0<wall<=1800 and type(memory)is int and 0<memory<=8*1024**3,'Explicit bounded independent verification resources required')
    def budget():
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if platform.system()!='Darwin':rss*=1024
        require(time.monotonic()-started<=wall and rss<=memory,'Independent verification resource ceiling exceeded')
        return rss
    require(json_file(root/'configs/walkforward_2020_2023.json')==cfg,'Saved outer configuration changed')
    require((run/'audit.json').is_file(),'Completed root audit required before independent accounting verification')
    root_audit_hash=digest(run/'audit.json')
    root_audit=json_file(run/'audit.json')
    require(root_audit.get('status')=='COMPLETE' and root_audit.get('build_identity_sha256')==identity_hash,'Root feature audit is stale or incomplete')
    backtest_audit_hash=digest(backtest/'audit.json')
    audit=json_file(backtest/'audit.json'); phase=json_file(backtest/'build_identity.json')
    require(audit.get('status')=='COMPLETE' and audit.get('phase')=='walkforward_continuous_backtest_v1','Backtest is not a completed continuous saved run')
    require(audit.get('build_identity_sha256')==identity_hash==phase['parent_identity_sha256'] and canonical(phase)==audit['phase_identity_sha256'],'Backtest/parent identity changed')
    require(phase['configuration_sha256']==canonical(cfg),'Backtest protocol differs from frozen parent')
    for item in phase['source_pins']:pinned(root,item)
    for item in audit['artifacts'].values():pinned(backtest,item)
    old=cfg['historical_retirement_registry']; old_dir=local(root,old['directory'])
    require(digest(old_dir/'audit.json')==old['audit_sha256'] and digest(old_dir/'build_identity.json')==old['build_identity_sha256'],'Original 2018--2019 registry changed')
    historical=json_file(old_dir/'build_identity.json')['events']; require(len(historical)==11 and historical==phase['historical_retirements'],'Original eleven actions changed or reused as new entitlements')
    require(audit.get('models_fitted') is False and audit.get('preprocessing_fitted') is False and audit.get('annual_liquidation') is False and audit.get('financial_2024_plus_read') is False,'Backtest changed fitting, year boundary or financial scope')
    require(cfg.get('protocol_id')=='dmv2-walkforward-2020-2023-v1' and cfg['backtest']['start']=='2020-01-01' and cfg['backtest']['end']=='2023-12-31','Frozen outer protocol/date scope changed')
    require(cfg['backtest']['arms']==list(ARMS) and cfg['backtest']['baskets']==list(BASKETS) and cfg['backtest']['cost_bps']==list(FEES) and cfg['backtest']['variants']=={'unstopped':None,'asset_stop_10pct':.1},'Frozen model/basket/stop/cost grid changed')
    require(cfg['accounting']['continuous_2020_2023_ledger'] is True and cfg['accounting']['annual_fit_forced_liquidation'] is False and cfg['accounting']['terminal_session']=='2023-12-29' and cfg['accounting']['borrow_annual_bps']==0 and cfg['accounting']['funding_annual_bps']==0,'Frozen continuous accounting convention changed')
    require(set(cfg['execution_restrictions']['native_ids'])==EXCLUDED,'Fixed trade restrictions changed')
    require(cfg['backtest']['top_k_per_side']==10 and cfg['backtest']['long_weight']==.5 and cfg['backtest']['short_weight']==-.5,'Frozen equal-side cap changed')
    dates,windows=independent_clock(); require(audit['sessions']==1006 and audit['outcome_months']==48 and audit['scenarios']==64,'Expected full saved grid changed')
    comparison=pd.read_parquet(backtest/'comparison.parquet'); yearly=pd.read_parquet(backtest/'yearly.parquet')
    coverage=key_table(backtest/'selection_coverage.parquet',kind='selection coverage')
    expected_grid=set(itertools.product(ARMS,BASKETS,('unstopped','asset_stop_10pct'),FEES))
    require(len(comparison)==64 and set(map(tuple,comparison[['arm','basket','variant','cost_bps']].to_numpy()))==expected_grid,'64-scenario grid changed')
    require(len(yearly)==256 and len(coverage)==384,'48-month coverage/continuous annual grid changed')
    require(comparison.complete_proxy_path.map(lambda value:isinstance(value,(bool,np.bool_))).all(),'Explicit complete/withheld path states required')
    require(audit.get('complete_proxy_paths')==int(comparison.complete_proxy_path.sum()) and audit.get('failed_closed_paths')==int((~comparison.complete_proxy_path).sum()),'Audit complete/withheld counts disagree with saved scenarios')
    prices=_quote_inputs(root,run,phase,dates); budget()
    quote_index={(row['trading_date'],row['intrinio_security_id']):row for row in prices.to_dict('records')}
    selection_reports=[]; scenario_reports=[]
    for arm in ARMS:
        forecast_path=run/'models/arms'/arm/'predictions.parquet'
        require(forecast_path.relative_to(root).as_posix() in {item['path'] for item in phase['source_pins']},'Forecast not pinned by accounting phase')
        fields=set(pq.ParquetFile(forecast_path).schema_arrow.names)
        forecasts=key_table(forecast_path,kind='saved forecasts',columns=[name for name in ['source_security_id','signal_date','intrinio_security_id','prediction_eligible','paper_prediction_eligible',*PROBABILITIES] if name in fields],maximum_rows=60_000)
        for basket in BASKETS:
            directory=backtest/'selections'/arm/basket
            tables={name:key_table(directory/(name+'.parquet'),kind='selection '+name,maximum_rows=60_000) for name in ('original_ledger','selected','monthly','rebalances')}
            chosen=tables['selected']; rebalance=tables['rebalances']
            joined=coverage.loc[coverage.arm.eq(arm)&coverage.basket.eq(basket)].drop(columns=['arm']).reset_index(drop=True)
            require(normalize(joined).equals(normalize(tables['monthly']).reset_index(drop=True)),'Aggregate coverage changed individual monthly breadth')
            selection_reports.append(dict(arm=arm,basket=basket,**verify_selection(forecasts,tables['original_ledger'],chosen,tables['monthly'],windows,basket=basket)))
            for variant,stop in [('unstopped',None),('asset_stop_10pct',.1)]:
                for fee in FEES:
                    budget(); path=backtest/'arms'/arm/basket/variant/f'cost_{fee:02d}bps'
                    names=('daily','trades','holdings','failures','stops','action_events','market_trades','orders')
                    ledgers={name:key_table(path/(name+'.parquet'),kind=name) for name in names}
                    summary=comparison.loc[comparison.arm.eq(arm)&comparison.basket.eq(basket)&comparison.variant.eq(variant)&comparison.cost_bps.eq(fee)].iloc[0]
                    annual=yearly.loc[yearly.arm.eq(arm)&yearly.basket.eq(basket)&yearly.variant.eq(variant)&yearly.cost_bps.eq(fee)]
                    monthly=pd.read_parquet(path/'monthly.parquet')
                    if summary['complete_proxy_path']:
                        checked=verify_scenario(ledgers,prices,chosen,rebalance,dates,events=phase['event_registry']['events'],historical_retirements=historical,cost_bps=fee,stop_fraction=stop,prepared_quotes=quote_index)
                        verify_metrics(ledgers['daily'],ledgers['trades'],monthly,summary,annual)
                        checked.update(path_status='complete_proxy',economic_performance_available=True)
                    else:
                        checked=verify_failed_scenario(ledgers,prices,chosen,dates,events=phase['event_registry']['events'],historical_retirements=historical,cost_bps=fee,prepared_quotes=quote_index)
                        verify_failed_metrics(ledgers['daily'],ledgers['trades'],monthly,summary,annual)
                        checked['path_status']='failed_closed'
                    scenario_reports.append(dict(arm=arm,basket=basket,variant=variant,cost_bps=fee,**checked))
                    print(json.dumps({'phase':'independent_walkforward_backtest_verification','verified_scenarios':len(scenario_reports),'expected_scenarios':64}),flush=True)
    # A verifier never repairs its inputs; verify every original and new pin again.
    for item in phase['source_pins']:pinned(root,item)
    for item in audit['artifacts'].values():pinned(backtest,item)
    require(digest(backtest/'audit.json')==backtest_audit_hash and digest(run/'audit.json')==root_audit_hash,'Output audit bytes changed during independent verification')
    result={'status':'PASS','phase':'independent_walkforward_backtest_verification_v1',
        'root_audit_sha256':root_audit_hash,'backtest_audit_sha256':backtest_audit_hash,
        'backtest_directory':backtest.relative_to(root).as_posix(),
        'build_identity_sha256':identity_hash,'backtest_phase_identity_sha256':audit['phase_identity_sha256'],
        'script_sha256':digest(__file__),'models_fitted':0,'preprocessors_fitted':0,'production_accounting_called':False,
        'source_pins_unchanged':True,'original_2018_2019_registry_unchanged':True,'financial_2024_plus_read':False,
        'sessions':1006,'outcome_months':48,'scenarios':64,
        'complete_proxy_paths':int(comparison.complete_proxy_path.sum()),'withheld_paths':int((~comparison.complete_proxy_path).sum()),
        'verification_scope':'complete_path_cash_units_math_and_failed_path_exposure_preservation',
        'withheld_economic_performance_certified':False,'selections':selection_reports,'scenario_checks':scenario_reports,
        'execution_seconds':time.monotonic()-started,'peak_rss_bytes':budget()}
    if output is not None:
        output=local(run,Path(output).resolve()); require(not output.exists(),'Preserve existing independent verification audit')
        output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True); parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--backtest',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); result=verify_saved_run(args.root,args.run,args.backtest,output=args.output)
    print(json.dumps({key:value for key,value in result.items() if key not in ('selections','scenario_checks')},sort_keys=True))


if __name__=='__main__':main()
