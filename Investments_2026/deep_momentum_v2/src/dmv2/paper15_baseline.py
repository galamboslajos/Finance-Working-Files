"""Pure Paper15/complete-history/decile/RET baseline rules. No I/O or fits.

All historical membership rows survive; only backward-looking history governs
signal eligibility. Five-year RET is an explicit adaptation of the paper.
"""
import math
import numpy as np
import pandas as pd

FEATURES=tuple(p+str(h) for p in ('zMOM','meanMOM','stdMOM') for h in (1,3,6,9,12))
KEYS=('source_security_id','signal_date')
PROBS=tuple('probability_'+str(k) for k in range(1,11))


def need(v,message):
    if not v:raise ValueError(message)


def shift(month,offset):
    y,m=map(int,month.split('-'));n=y*12+m-1+offset;return f'{n//12:04d}-{n%12+1:02d}'


def strict_history(valid,sessions,native_position,signal,years):
    need(years in (5,10) and signal<'2024-01-01','Approved historical signal/window required')
    lower=(pd.Timestamp(signal)-pd.DateOffset(years=years)).strftime('%Y-%m-%d')
    days=np.asarray(sessions);lo=int(np.searchsorted(days,lower,side='right'));hi=int(np.searchsorted(days,signal,side='right'))
    need(lower>='1999-12-31' and hi and days[hi-1]==signal and hi>lo,'Complete calendar bounds required')
    observed=int(valid[lo:hi,native_position].sum()) if native_position is not None else 0
    return dict(history_start_exclusive=lower,history_end=signal,expected_sessions=hi-lo,
       valid_sessions=observed,history_complete=observed==hi-lo)


def deciles(values):
    """Exact equal-return ties share descending midpoint bins; nulls stay null."""
    v=np.asarray(values,dtype=float);result=np.full(len(v),np.nan);observed=np.flatnonzero(np.isfinite(v));n=len(observed)
    if n<10:return result
    order=observed[np.argsort(-v[observed],kind='stable')];start=0
    while start<n:
        end=start+1
        while end<n and v[order[end]]==v[order[start]]:end+=1
        result[order[start:end]]=min(10,max(1,1+(10*(start+end))//(2*n)));start=end
    return result


def build_panels(cohorts,valid,sessions,month_closes,natives,months,years,*,prices=None,quote_validity=None,events=(),exit_calendar=()):
    """Recompute all15inputs and labels for each history-qualified cross-section."""
    need(years in (5,10) and len(FEATURES)==15 and prices is not None and quote_validity is not None,'Paper15 and executable daily target inputs required')
    frame=cohorts.sort_values(list(KEYS),kind='stable').reset_index(drop=True).copy()
    need(not frame.duplicated(list(KEYS)).any() and frame.signal_date.max()<'2024-01-01','Unique protected original cohort keys')
    ni={n:i for i,n in enumerate(natives)};maps=({d:i for i,d in enumerate(sessions)},ni);mi={m:i for i,m in enumerate(months)}
    days=np.asarray(sessions);by_month={m:days[np.char.startswith(days.astype(str),m)].tolist() for m in months}
    invalid=np.vstack([np.zeros((1,len(natives)),dtype=np.int32),np.cumsum(~valid,axis=0,dtype=np.int32)])
    rows=[];coverage=[]
    for signal,part in frame.groupby('signal_date',sort=True):
        need(signal==by_month[signal[:7]][-1] and signal[:7]<'2023-12','Signal must be exact supported month end')
        future=shift(signal[:7],1);window=by_month[future]
        lower=(pd.Timestamp(signal)-pd.DateOffset(years=years)).strftime('%Y-%m-%d')
        lo=int(np.searchsorted(days,lower,side='right'));hi=int(np.searchsorted(days,signal,side='right'));need(lower>='1999-12-31','Unobserved older eligibility calendar')
        block=part.copy();raw=np.full((len(part),5),np.nan);counts=[];complete=[]
        for j,r in enumerate(part.to_dict('records')):
            native=ni.get(r['intrinio_security_id']);missing=(int(invalid[hi,native]-invalid[lo,native]) if native is not None else hi-lo)
            counts.append(hi-lo-missing);admitted=bool(r['identity_eligible']) and missing==0;complete.append(missing==0)
            if admitted:
                for k,h in enumerate((1,3,6,9,12)):
                    required=([shift(signal[:7],-1),signal[:7]] if h==1 else [shift(signal[:7],q) for q in range(-h,0)])
                    values=[month_closes[mi[m],native] if m in mi else np.nan for m in required]
                    if all(math.isfinite(v) and v>0 for v in values):raw[j,k]=math.prod(values[i+1]/values[i] for i in range(len(values)-1))-1
        admitted=part.identity_eligible.to_numpy(dtype=bool);common=admitted & np.asarray(complete) & np.isfinite(raw).all(axis=1)
        x=np.full((len(part),15),np.nan);population=raw[common]
        if len(population)>=2:
            mean=np.array([math.fsum(v)/len(v) for v in population.T]);sd=np.array([math.sqrt(math.fsum((z-a)**2 for z in v)/(len(v)-1)) for v,a in zip(population.T,mean)])
            if np.isfinite(mean).all() and np.isfinite(sd).all() and (sd>0).all():
                x[common,:5]=(population-mean)/sd;x[common,5:10]=mean;x[common,10:]=sd
        eligible=np.isfinite(x).all(axis=1)&common
        block['history_years']=years;block['history_start_exclusive']=lower;block['history_end']=signal
        block['expected_history_sessions']=hi-lo;block['valid_history_sessions']=counts;block['history_complete']=complete
        block['prediction_eligible']=eligible;block['outcome_month']=future;block['outcome_start']=window[0];block['outcome_end']=window[-1]
        for k,n in enumerate(FEATURES):block[n]=x[:,k]
        returns=np.full(len(block),np.nan);statuses=[];markcounts=[];actioncounts=[]
        exit_month=shift(future,1);exit_date=by_month[exit_month][0] if exit_month in by_month else next((d for d in exit_calendar if d.startswith(exit_month)),None)
        need(exit_date is not None,'Scheduled exit calendar metadata required')
        block['label_available_at']=exit_date;block['target_entry_date']=window[0];block['target_exit_date']=exit_date
        for j,r in enumerate(part.to_dict('records')):
            if eligible[j]:value,status,marks,actions=executable_return(prices,quote_validity,sessions,natives,r['intrinio_security_id'],window[0],exit_date,events,maps=maps)
            else:value,status,marks,actions=np.nan,'not_prediction_eligible',0,0
            returns[j]=value;statuses.append(status);markcounts.append(marks);actioncounts.append(actions)
        block['target_return_status']=statuses;block['target_mark_count']=markcounts;block['target_action_count']=actioncounts
        block['label_return']=returns;block['target_class']=deciles(returns);block['target_available']=block.target_class.notna()
        block['prediction_status']=np.where(eligible,'eligible',np.where(~admitted,'identity_blocked',np.where(~np.asarray(complete),'incomplete_past_daily_history','paper_input_or_population_unavailable')))
        rows.append(block)
        coverage.append(dict(history_years=years,signal_date=signal,outcome_month=future,original_members=len(block),identity_admitted=int(admitted.sum()),
          complete_history=int((admitted&np.asarray(complete)).sum()),prediction_eligible=int(eligible.sum()),observed_returns=int(np.isfinite(returns).sum()),
          labelled=int(block.target_available.sum()),missing_targets=int((eligible&~block.target_available.to_numpy()).sum()),expected_history_sessions=hi-lo))
    return pd.concat(rows,ignore_index=True),pd.DataFrame(coverage)


def ret_payoffs(panel,signal,years):
    need(years in (5,10) and signal<'2024-01-01','Approved RET clock')
    months=[shift(signal[:7],i) for i in range(-12*years,0)];cells=[]
    historical=panel.loc[panel.outcome_month.isin(months)&panel.label_available_at.le(signal)]
    for month in months:
        p=historical.loc[historical.outcome_month.eq(month)]
        for k in range(1,11):
            v=p.loc[p.target_class.eq(k),'label_return'].to_numpy(dtype=float)
            cells.append({'signal_date':signal,'history_years':years,'outcome_month':month,'class_id':k,'count':len(v),
              'mean_return':math.fsum(v)/len(v) if len(v) else None,'status':'available' if len(v) else 'missing_month_or_empty_class'})
    need(len(cells)==120*years,'Exact month/class grid')
    mu=[math.fsum(c['mean_return'] for c in cells if c['class_id']==k)/(12*years) for k in range(1,11)] if all(c['count'] for c in cells) else None
    return mu,cells


def select_ret(predictions,payoffs,windows,excluded=()):
    """Signal-only selection. Neither targets nor future quotes are accepted."""
    required={*KEYS,'intrinio_security_id','prediction_eligible',*PROBS}
    need(required<=set(predictions),'Missing signal predictions')
    forbidden=[c for c in predictions if str(c).startswith(('target','label','future','realized'))]
    need(not forbidden,'Future outcome fields cannot enter selection')
    ledger=predictions[[c for c in predictions if c not in PROBS]].copy();ledger['selected']=False;ledger['side']=None;ledger['selection_score']=np.nan
    ledger['selection_status']='not_selected';ledger['execution_restricted']=ledger.intrinio_security_id.isin(excluded)
    selected=[];rebalances=[];monthly=[]
    for w in windows:
        signal=w['signal_date'];idx=ledger.index[ledger.signal_date.eq(signal)];eligible=idx[ledger.loc[idx,'prediction_eligible'].to_numpy(dtype=bool)]
        mu=payoffs[signal];n=0
        if mu is not None:
            need(len(mu)==10 and np.isfinite(mu).all(),'Finite complete RET vector')
            p=predictions.loc[eligible,list(PROBS)].to_numpy(dtype=float);need(np.isfinite(p).all() and (p>=0).all() and np.allclose(p.sum(axis=1),1,atol=1e-8,rtol=0),'Aligned probabilities')
            ledger.loc[eligible,'selection_score']=p@np.asarray(mu)
            tradable=[i for i in eligible if not ledger.at[i,'execution_restricted']]
            need(not ledger.loc[tradable].duplicated('intrinio_security_id').any(),'Signal native collision')
            n=min(10,len(tradable)//2);long=sorted(tradable,key=lambda i:(-float(ledger.at[i,'selection_score']),ledger.at[i,'source_security_id']))[:n]
            short=sorted((i for i in tradable if i not in long),key=lambda i:(float(ledger.at[i,'selection_score']),ledger.at[i,'source_security_id']))[:n]
            for side,chosen in [('long',long),('short',short)]:
                ledger.loc[chosen,'selected']=True;ledger.loc[chosen,'side']=side;ledger.loc[chosen,'selection_status']='selected'
                for i in chosen:selected.append({**ledger.loc[i].to_dict(),'execution_date':w['outcome_start'],'intended_weight':(.5 if side=='long' else -.5)/n})
        status='available' if n else 'withheld_incomplete_RET_grid' if mu is None else 'withheld_zero_candidates'
        if w['outcome_month']=='2020-01':
            selected=[r for r in selected if r['signal_date']!=signal];ledger.loc[idx,'selected']=False;ledger.loc[idx,'side']=None;ledger.loc[idx,'selection_status']='outside_common_comparison_period';n=0;status='outside_common_comparison_period'
        if not n:ledger.loc[idx,'selection_status']=status
        rebalances.append(dict(signal_date=signal,execution_date=w['outcome_start'],selection_status=status,selected_names_per_side=n))
        monthly.append(dict(signal_date=signal,outcome_month=w['outcome_month'],original_rows=len(idx),eligible=len(eligible),selected_per_side=n,status=status,future_data_used=False))
    return ledger,pd.DataFrame(selected),pd.DataFrame(rebalances),pd.DataFrame(monthly)


def executable_return(prices,quality,sessions,natives,native,entry,exit_date,events=(),maps=None):
    """Long-unit daily cash/share wealth; no stop, costs, filling or target clipping.

    Price order is raw open, raw close, adjusted open, adjusted close. Quality
    has independent open/close flags. Cash conversion needs no retired quote.
    Return is cumulative executable wealth minus one; intramonth prices are
    validated even though a no-action path algebraically telescopes.
    """
    if exit_date>='2024-01-01':return np.nan,'protected_scheduled_endpoint',0,0
    positions,ids=maps if maps is not None else ({d:i for i,d in enumerate(sessions)},{n:i for i,n in enumerate(natives)})
    if native not in ids:return np.nan,'missing_native_history',0,0
    a,b=positions.get(entry),positions.get(exit_date)
    if a is None or b is None or a>=b:return np.nan,'missing_calendar_boundary',0,0
    if any(e['predecessor_id']==native and e['effective_date']<=entry for e in events):return np.nan,'retired_new_entry',0,0
    n=ids[native]
    if not quality[a,n,0]:return np.nan,'invalid_or_missing_entry_open',0,0
    relevant=[e for e in events if entry<e['effective_date']<=exit_date]
    if not any(e['predecessor_id']==native for e in relevant):
        if not quality[a:b+1,n,0].all() or not quality[a:b,n,1].all():return np.nan,'missing_or_invalid_daily_mark',0,0
        value=prices[b,n,2]/prices[a,n,2]-1
        return (float(value),'ok',2*(b-a)+1,0) if math.isfinite(value) else (np.nan,'nonfinite_return',0,0)
    units=1/float(prices[a,n,2]);cash=0.;marks=0;actions=0
    grouped={}
    for e in relevant:grouped.setdefault(e['effective_date'],[]).append(e)
    for d in range(a,b+1):
        date=sessions[d]
        if units and d>a:
            for e in grouped.get(date,[]):
                if e['predecessor_id']!=native:continue
                if not quality[d-1,n,1]:return np.nan,'missing_action_basis',marks,actions
                raw_units=units*prices[d-1,n,3]/prices[d-1,n,1]
                cash+=raw_units*e['cash_per_raw_share'];actions+=1
                if e['stock_ratio']:
                    successor=e['successor_id'];n=ids.get(successor)
                    if n is None or not quality[d,n,0]:return np.nan,'missing_successor_open',marks,actions
                    units=raw_units*e['stock_ratio']/(prices[d,n,2]/prices[d,n,0]);native=successor
                else:units=0.
        for mark in (0,) if d==b else (0,1):
            if units and not quality[d,n,mark]:return np.nan,'missing_or_invalid_daily_mark',marks,actions
            nav=cash+(units*prices[d,n,2+mark] if units else 0.);marks+=1
            if not math.isfinite(nav):return np.nan,'nonfinite_wealth',marks,actions
    return float(nav-1),'ok',marks,actions
