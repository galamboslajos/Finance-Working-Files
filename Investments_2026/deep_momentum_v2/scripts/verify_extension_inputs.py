#!/usr/bin/env python3
"""Independent source-array and formula verification; never fits or trades."""
from pathlib import Path
import math
import resource
import time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from dmv2.extension_inputs import validate, quote_blocks, read_metadata, OHLC, COHORT
from dmv2.paper15_baseline_data import pin, check, read, save


def require(value, message):
    if not value: raise AssertionError(message)


def verify_data(plan_path):
    p=validate(plan_path);root=Path(p['output_directory']);rp=pin(root/'prepared/receipt.json');receipt=read(check(rp))
    require(receipt['plan_pin']==pin(plan_path) and receipt['new_fits']==receipt['new_backtests']==0,'Input receipt binding')
    a=receipt['artifacts'];idx=read(check(a['index']));old=p['parent_inputs'];oi=read(check(old['index']))
    arrays={k:np.load(check(a[k]),mmap_mode='r',allow_pickle=False) for k in ('valid','present','month_closes','extension_ohlc','extension_quote_validity')}
    valid,present,closes=arrays['valid'],arrays['present'],arrays['month_closes'];natives=idx['natives'];days=idx['sessions']
    ni=pd.Index(natives);di=pd.Index(days);ei=pd.Index(idx['extension_sessions']);mi={m:i for i,m in enumerate(idx['months'])}
    ends={d[:7]:d for d in days};parentpos=ni.get_indexer(oi['natives']);oldlen=len(oi['sessions'])
    require(days==p['sessions'] and days[-1]=='2026-04-30','Scoped calendar')
    # Authenticate inherited corrected history without recasting or revising it.
    for name in ('valid','present'):
        parent=np.load(check(old[name]),mmap_mode='r',allow_pickle=False)
        require(np.array_equal(arrays[name][:oldlen,parentpos],parent),'Old '+name+' changed')
    parent=np.load(check(old['month_closes']),mmap_mode='r',allow_pickle=False)
    require(np.array_equal(closes[:len(oi['months']),parentpos],parent,equal_nan=True),'Old close anchors changed')
    cohort=pq.read_table(check(a['cohorts'])).to_pandas();expected,exclusions=read_metadata(p)
    # Same pinned membership/route adapter is shared; independent price and all
    # feature/target/payoff arithmetic below do not call the producer arithmetic.
    require(cohort.to_json()==expected.to_json(),'Dated membership/route adapter consistency')
    seen=np.zeros(valid.shape,bool);row_count=0
    for block in quote_blocks(p,natives,idx['new_native_ids']):
        d=di.get_indexer(block.trading_date);n=ni.get_indexer(block.intrinio_security_id);keep=(d>=0)&(n>=0)
        block=block.loc[keep].reset_index(drop=True);d,n=d[keep],n[keep]
        require(not seen[d,n].any() and len(set(zip(d,n)))==len(d),'Duplicate source keys')
        seen[d,n]=True;row_count+=len(block)
        values=block[list(OHLC)].to_numpy(float)
        quality=block.frequency.eq('daily').to_numpy() & block.intraperiod.eq(False).to_numpy() & block.price_quality_status.isin(['accepted','ok']).to_numpy()
        flags=block.raw_ohlc_bounds_valid.eq(True).to_numpy()&block.adjusted_ohlc_bounds_valid.eq(True).to_numpy()
        complete=quality&flags
        for col in range(8):complete=complete&np.isfinite(values[:,col])&(values[:,col]>0)
        for o,h,l,c in ((0,1,2,3),(4,5,6,7)):
            complete=complete&(values[:,h]>=values[:,o])&(values[:,h]>=values[:,c])&(values[:,l]<=values[:,o])&(values[:,l]<=values[:,c])&(values[:,h]>=values[:,l])
        require(np.array_equal(valid[d,n],complete) and present[d,n].all(),'Independent OHLC validity mismatch')
        fresh=block.trading_date.ge('2024-01-01').to_numpy();e=ei.get_indexer(block.trading_date)
        require(np.array_equal(arrays['extension_ohlc'][e[fresh],n[fresh]],values[fresh],equal_nan=True),'New daily OHLC bytes mismatch')
        for k,cols in ((0,(0,4)),(1,(3,7))):
            flagsok=quality&flags
            for col in cols:flagsok=flagsok&np.isfinite(values[:,col])&(values[:,col]>0)
            require(np.array_equal(arrays['extension_quote_validity'][e[fresh],n[fresh],k],flagsok[fresh]),'Quote mark validity')
        for i,date in enumerate(block.trading_date):
            if date!=ends[date[:7]]:continue
            val=values[i,7] if quality[i] and np.isfinite(values[i,7]) and values[i,7]>0 else np.nan
            observed=closes[mi[date[:7]],n[i]]
            require((np.isnan(val) and np.isnan(observed)) or val==observed,'Month-end source anchor')
    # Unobserved quotes must stay absent, invalid and NaN in the financial cache.
    newrows=np.array(days)>='2024-01-01';newcols=ni.get_indexer(idx['new_native_ids'])
    require(np.array_equal(present[newrows],seen[newrows]),'Future present mask manufactured')
    require(not valid[newrows][~seen[newrows]].any(),'Missing future quote admitted')
    require(np.isnan(arrays['extension_ohlc'][~seen[newrows]]).all(),'Unobserved OHLC fabricated')
    if len(newcols):
        require(np.array_equal(present[:oldlen,newcols],seen[:oldlen,newcols]),'New ID past observation mask')
        require(not valid[:oldlen,newcols][~seen[:oldlen,newcols]].any(),'New ID missing past fabricated')
    require(row_count==idx['returned_source_rows'],'Source row receipt')
    summaries=[]
    for years in (5,10):
        panel=pq.read_table(check(a[f'panel{years}'])).to_pandas()
        require(not panel.duplicated(['source_security_id','signal_date']).any() and len(panel)==len(cohort),'Original denominators')
        left=panel[list(COHORT)].sort_values(['source_security_id','signal_date']).reset_index(drop=True)
        right=cohort[list(COHORT)].sort_values(['source_security_id','signal_date']).reset_index(drop=True)
        require(left.to_json()==right.to_json(),'Panel changed cohort metadata')
        cells_checked=0
        for signal,b in panel.groupby('signal_date',sort=True):
            lower=(pd.Timestamp(signal)-pd.DateOffset(years=years)).strftime('%Y-%m-%d')
            history=[i for i,d in enumerate(days) if lower<d<=signal]
            require(b.expected_history_sessions.eq(len(history)).all(),'Calendar history expected count')
            raw=np.full((len(b),5),np.nan);complete=[]
            for j,r in enumerate(b.to_dict('records')):
                native=natives.index(r['intrinio_security_id']) if r['intrinio_security_id'] in natives else None
                count=int(np.count_nonzero(valid[history,native])) if native is not None else 0
                require(count==r['valid_history_sessions'] and r['history_complete']==(count==len(history)),'Complete past history')
                complete.append(count==len(history))
                if not r['identity_eligible'] or count!=len(history):continue
                # Independent telescoping endpoints, while checking every required anchor.
                sm=signal[:7];smpos=mi[sm]
                for k,h in enumerate((1,3,6,9,12)):
                    indices=[smpos-1,smpos] if h==1 else list(range(smpos-h,smpos))
                    v=np.asarray(closes[indices,native])
                    if np.isfinite(v).all() and np.min(v)>0:raw[j,k]=v[-1]/v[0]-1
            common=b.identity_eligible.to_numpy(bool)&np.array(complete)&np.isfinite(raw).all(axis=1)
            expectedx=np.full((len(b),15),np.nan)
            if common.sum()>=2:
                pop=raw[common];mean=np.mean(pop,axis=0);sd=np.std(pop,axis=0,ddof=1)
                if np.isfinite(sd).all() and np.min(sd)>0:
                    expectedx[common]=np.column_stack(((pop-mean)/sd,np.tile(mean,(len(pop),1)),np.tile(sd,(len(pop),1))))
            names=[prefix+str(h) for prefix in ('zMOM','meanMOM','stdMOM') for h in (1,3,6,9,12)]
            require(np.allclose(b[names].to_numpy(float),expectedx,rtol=2e-11,atol=2e-11,equal_nan=True),'Independent Paper15 formulas')
            eligible=np.isfinite(expectedx).all(axis=1)&common
            require(np.array_equal(eligible,b.prediction_eligible.to_numpy(bool)),'Future values affected signal eligibility')
            returns=np.full(len(b),np.nan)
            for j,r in enumerate(b.to_dict('records')):
                require(r['target_entry_date']==signal and r['label_available_at']==ends[r['outcome_month']],'Label clock')
                if not eligible[j]:continue
                n=natives.index(r['intrinio_security_id']);start=closes[mi[signal[:7]],n];end=closes[mi[r['outcome_month']],n]
                if np.isfinite(start) and start>0 and np.isfinite(end) and end>0:returns[j]=end/start-1
            require(np.allclose(returns,b.label_return.to_numpy(float),rtol=0,atol=0,equal_nan=True),'Independent monthly return')
            actual=b.target_class.to_numpy(float);observed=np.flatnonzero(np.isfinite(returns));expectedclasses=np.full(len(b),np.nan)
            if len(observed)>=10:
                ranks=pd.Series(returns[observed]).rank(method='average',ascending=False).to_numpy()
                expectedclasses[observed]=np.clip(1+np.floor(10*(ranks-.5)/len(observed)),1,10)
            require(np.array_equal(expectedclasses,actual,equal_nan=True),'Independent exact ties/deciles')
            require(np.array_equal(np.isfinite(actual),b.target_available.to_numpy(bool)),'Missing label retention')
            cells_checked+=int(eligible.sum())*15
        past=pq.read_table(check(old[f'panel{years}'])).to_pandas();combined=pd.concat([past,panel],ignore_index=True)
        payoffs=read(check(a[f'payoffs{years}']));grid=pq.read_table(check(a[f'payoff_grid{years}'])).to_pandas()
        for signal,mu in payoffs.items():
            y,m=map(int,signal[:7].split('-'));last=y*12+m-1;months=[f'{q//12:04d}-{q%12+1:02d}' for q in range(last-12*years+1,last+1)]
            pastmask=combined.outcome_month.isin(months)&combined.label_available_at.le(signal)
            means=combined.loc[pastmask].groupby(['outcome_month','target_class']).label_return.mean()
            full=pd.MultiIndex.from_product([months,range(1,11)]);means=means.reindex(full)
            expectedmu=means.to_numpy().reshape(12*years,10).mean(axis=0) if means.notna().all() else None
            require((mu is None)==(expectedmu is None),'RET coverage gate')
            if mu is not None:require(np.allclose(mu,expectedmu,rtol=1e-13,atol=1e-14),'RET causal temporal averages')
            g=grid.loc[grid.signal_date.eq(signal)]
            require(len(g)==years*120 and list(g.outcome_month.unique())==months,'RET full month/class grid')
        summaries.append(dict(history_years=years,original_members=len(panel),verified_feature_cells=cells_checked,
            prediction_eligible=int(panel.prediction_eligible.sum()),missing_targets=int((panel.prediction_eligible&~panel.target_available).sum()),
            complete_RET_vectors=sum(x is not None for x in payoffs.values())))
    return save(root/'verification/data.json',dict(status='PASS_INPUT_SOURCE_ARRAYS_AND_MONTHLY_FORMULAS',plan_pin=pin(plan_path),
        prepared_receipt_pin=rp,worker_pin=pin(__file__),source_rows_checked=row_count,summary=summaries,new_fits=0,new_backtests=0,
        scope='Shared pinned metadata/routing adapter; independent full price-array, eligibility, Paper15, monthly-return, class and RET-mean arithmetic',
        execution_action_coverage_certified=False,post2023_BK_XOM_routes_approved=False))
