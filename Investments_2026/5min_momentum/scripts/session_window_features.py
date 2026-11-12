"""Session-by-session descriptors with a sealed ten-session dependency window.

No model, label, future price, inherited eligibility or smoothing state enters.
All normalization is as of the completed window end, not the older day's open.
"""
from __future__ import annotations
import numpy as np

WINDOW=10
MAX_SLOTS=78
HORIZONS=(5,10,15,30)
DAY_NAMES=([f'open_{h:02}m' for h in HORIZONS]
    +[f'follow_{h:02}m' for h in HORIZONS]
    +[f'persist_{h:02}m' for h in HORIZONS]
    +['intraday','efficiency']+[f'range_{h:02}m' for h in HORIZONS]
    +['realized_vol','downside_share']+[f'activity_{h:02}m' for h in HORIZONS])
GAP_NAMES=('gap','gap_response','gap_filled')
FEATURES=[f'lag{lag:02}_{name}' for lag in range(1,11)
          for name in DAY_NAMES+ (list(GAP_NAMES) if lag<10 else [])]
REASONS={0:'observed',1:'missing_duplicate_invalid_price',2:'incomparable_source',
         3:'zero_window_risk',4:'flat_opening_no_direction',5:'zero_session_path_variance',
         6:'zero_gap_no_fill_event',7:'missing_duplicate_invalid_quantity',8:'zero_activity_reference'}


def ratio(a,b):
    a,b=np.broadcast_arrays(np.asarray(a,float),np.asarray(b,float))
    return np.divide(a,b,out=np.full(a.shape,np.nan),where=np.isfinite(b)&(b>0))


def make_grid(frame,calendar):
    """Dense scheduled cells. Missing bars remain absent; duplicates stay bad."""
    shape=(len(calendar),MAX_SLOTS)
    fields=('open','high','low','close','raw_open','raw_high','raw_low','raw_close',
            'volume','vwap','minute_bars','raw_minute_bars')
    grid={name:np.full(shape,np.nan) for name in fields}
    grid.update({name:np.zeros(shape,int) for name in ('count','raw_count','source','raw_source')})
    grid.update({name:np.zeros(shape,bool) for name in ('time_valid','raw_time_valid')})
    if len(frame):
        if frame.duplicated(['session_num','slot']).any():raise ValueError('Slot duplicates must be counted before gridding')
        # read_month already aggregates duplicate timestamps and records counts.
        rows=calendar.session_num.to_numpy().searchsorted(frame.session_num.to_numpy())
        if np.any(rows>=len(calendar)) or not np.array_equal(calendar.session_num.to_numpy()[rows],frame.session_num.to_numpy()):
            raise ValueError('Bars outside dense calendar')
        slots=frame.slot.to_numpy(int)
        if np.any((slots<0)|(slots>=MAX_SLOTS)):raise ValueError('Invalid bar slot')
        for name in grid:grid[name][rows,slots]=frame[name].to_numpy()
    return grid


def daily_descriptors(grid,expected_slots):
    """Every descriptor here depends only on its own scheduled session."""
    n=np.asarray(expected_slots,int);days=len(n)
    if np.any((n<6)|(n>MAX_SLOTS)):raise ValueError('Unsupported session length')
    slot=np.arange(MAX_SLOTS)[None,:];scheduled=slot<n[:,None]
    o,h,l,c=[np.asarray(grid[name],float) for name in ('open','high','low','close')]
    source=grid['source'];raw_source=grid['raw_source']
    def price_valid(prefix):
        oo,hh,ll,cc=[grid[prefix+name] for name in ('open','high','low','close')]
        count=grid['raw_count' if prefix else 'count']
        time=grid['raw_time_valid' if prefix else 'time_valid']
        src=raw_source if prefix else source
        return (np.isfinite(oo)&np.isfinite(hh)&np.isfinite(ll)&np.isfinite(cc)
            &(ll>0)&(hh>=ll)&(oo>=ll)&(oo<=hh)&(cc>=ll)&(cc<=hh)
            &(count==1)&time&(src>0))
    valid=price_valid('')
    homogeneous=((source==source[:,:1])|~scheduled).all(axis=1)&(source[:,0]>0)
    price_ok=((valid|~scheduled).all(axis=1))&homogeneous
    raw_valid=price_valid('raw_')
    raw_homogeneous=((raw_source==raw_source[:,:1])|~scheduled).all(axis=1)&(raw_source[:,0]>0)
    v,w=grid['volume'],grid['vwap']
    quantity=(raw_valid&(grid['count']==1)&grid['time_valid']&(source>0)
        &np.isfinite(v)&(v>=0)&((v==0)|(np.isfinite(w)&(w>0)&(w>=grid['raw_low'])&(w<=grid['raw_high']))))
    cross_source=((source==raw_source)|~scheduled).all(axis=1)
    quantity_ok=(quantity|~scheduled).all(axis=1)&raw_homogeneous&cross_source&homogeneous
    previous=np.concatenate([o[:,:1],c[:,:-1]],axis=1)
    with np.errstate(divide='ignore',invalid='ignore'):
        increments=np.where(scheduled,np.log(c/previous),0.)
        net=np.log(c[np.arange(days),n-1]/o[:,0])
    rv2=np.sum(increments**2,axis=1);absolute=np.sum(abs(increments),axis=1)
    result=dict(price_ok=price_ok,quantity_ok=quantity_ok,source=source[:,0],
        session_open=o[:,0],session_close=c[np.arange(days),n-1],
        open_ok=valid[:,0],close_ok=valid[np.arange(days),n-1],
        close_source=source[np.arange(days),n-1],
        high=np.where(scheduled,h,-np.inf).max(axis=1),
        low=np.where(scheduled,l,np.inf).min(axis=1),
        missing_slots=((grid['count']==0)&scheduled).sum(axis=1),
        duplicate_slots=((grid['count']>1)&scheduled).sum(axis=1),
        invalid_price_slots=((~valid)&(grid['count']>0)&scheduled).sum(axis=1),
        invalid_quantity_slots=((~quantity)&scheduled).sum(axis=1),
        partial_buckets=(((grid['minute_bars']<5)|(grid['raw_minute_bars']<5))&scheduled).sum(axis=1))
    values={name:np.full(days,np.nan) for name in DAY_NAMES}
    values['intraday']=np.where(price_ok,net,np.nan)
    values['efficiency']=np.where(price_ok,ratio(net,absolute),np.nan)
    values['realized_vol']=np.where(price_ok,np.sqrt(rv2),np.nan)
    values['downside_share']=np.where(price_ok,ratio(np.sum(np.minimum(increments,0)**2,axis=1),rv2),np.nan)
    reasons={name:np.where(price_ok,0,1).astype(np.uint8) for name in DAY_NAMES}
    for name in ('efficiency','downside_share'):
        reasons[name]=np.where(price_ok&(rv2==0),5,reasons[name]).astype(np.uint8)
    dollars=np.where(v==0,0,v*w)
    for horizon in HORIZONS:
        k=horizon//5
        with np.errstate(divide='ignore',invalid='ignore'):
            opening=np.log(c[:,k-1]/o[:,0])
            following=np.log(c[np.arange(days),n-1]/c[:,k-1])
            width=np.log(np.max(h[:,:k],axis=1)/np.min(l[:,:k],axis=1))
        values[f'open_{horizon:02}m']=np.where(price_ok,opening,np.nan)
        values[f'follow_{horizon:02}m']=np.where(price_ok,np.sign(opening)*following,np.nan)
        balance=np.sum(np.where(scheduled&(slot>=k),np.sign(increments),0),axis=1)/(n-k)
        values[f'persist_{horizon:02}m']=np.where(price_ok&(opening!=0),np.sign(opening)*balance,np.nan)
        reasons[f'persist_{horizon:02}m']=np.where(price_ok&(opening==0),4,reasons[f'persist_{horizon:02}m']).astype(np.uint8)
        values[f'range_{horizon:02}m']=np.where(price_ok,width,np.nan)
        values[f'activity_{horizon:02}m']=np.where(quantity_ok,np.sum(dollars[:,:k],axis=1),np.nan)
        reasons[f'activity_{horizon:02}m']=np.where(quantity_ok,0,7).astype(np.uint8)
    result.update(values=values,reasons=reasons)
    return result


def window_matrix(daily):
    """Return all dense ten-day windows; no pre-window state is accepted.

    Session range/return units are the RMS of ten daily realized volatilities.
    Realized-vol columns retain absolute decimal log-return units. Activity
    compares each day with the other nine days of the same completed window.
    """
    length=len(daily['price_ok'])
    if length<WINDOW:raise ValueError('Ten scheduled sessions required')
    index=np.arange(length-WINDOW+1)[:,None]+np.arange(WINDOW)[None,:]
    price_ok=daily['price_ok'][index];quantity_ok=daily['quantity_ok'][index]
    src=daily['source'][index]
    same=(src>0).all(axis=1)&(src==src[:,:1]).all(axis=1)
    price_complete=price_ok.all(axis=1)&same
    activity_complete=quantity_ok.all(axis=1)&same
    risk=np.sqrt(np.mean(daily['values']['realized_vol'][index]**2,axis=1))
    risk=np.where(price_complete,risk,np.nan)
    risk_reason=np.where(~price_ok.all(axis=1),1,np.where(~same,2,np.where(risk>0,0,3))).astype(np.uint8)
    count=len(index);values={};reasons={}
    scaled=[name for name in DAY_NAMES if name.startswith(('open_','follow_','range_')) or name=='intraday']
    for day in range(WINDOW):
        lag=WINDOW-day
        for name in DAY_NAMES:
            key=f'lag{lag:02}_{name}';v=daily['values'][name][index[:,day]].copy()
            reason=daily['reasons'][name][index[:,day]].copy()
            if name in scaled:
                v=ratio(v,risk)
                reason=np.where(risk_reason!=0,risk_reason,reason).astype(np.uint8)
            if name.startswith('activity_'):
                all_values=daily['values'][name][index]
                reference=np.median(np.delete(all_values,day,axis=1),axis=1)
                v=ratio(v,reference)
                valid=activity_complete&np.isfinite(reference)&(reference>0)
                v=np.where(valid,v,np.nan)
                reason=np.where(~quantity_ok.all(axis=1),7,np.where(~same,2,np.where(reference>0,0,8))).astype(np.uint8)
            values[key]=v;reasons[key]=reason
    # Nine internal gaps only. The first chronological session has no gap input.
    previous=daily['session_close'][index[:,:-1]]
    opening=daily['session_open'][index[:,1:]]
    endpoints=daily['close_ok'][index[:,:-1]]&daily['open_ok'][index[:,1:]]
    pair_source=(daily['close_source'][index[:,:-1]]==src[:,1:])&(src[:,1:]>0)
    with np.errstate(divide='ignore',invalid='ignore'):gaps=np.log(opening/previous)
    for day in range(1,WINDOW):
        lag=WINDOW-day;j=day-1
        anchors=endpoints[:,j]&pair_source[:,j]
        base_reason=np.where(~endpoints[:,j],1,np.where(~pair_source[:,j],2,0)).astype(np.uint8)
        reason=np.where(base_reason!=0,base_reason,risk_reason).astype(np.uint8)
        gap=gaps[:,j]
        v=ratio(gap,risk);values[f'lag{lag:02}_gap']=np.where(anchors,v,np.nan)
        reasons[f'lag{lag:02}_gap']=reason
        response=np.sign(gap)*daily['values']['intraday'][index[:,day]]
        values[f'lag{lag:02}_gap_response']=np.where(anchors,ratio(response,risk),np.nan)
        reasons[f'lag{lag:02}_gap_response']=np.where(~price_ok[:,day],1,reason).astype(np.uint8)
        filled=np.where(gap>0,daily['low'][index[:,day]]<=previous[:,j],daily['high'][index[:,day]]>=previous[:,j])
        valid=anchors&price_ok[:,day]&(gap!=0)
        values[f'lag{lag:02}_gap_filled']=np.where(valid,filled.astype(float),np.nan)
        reasons[f'lag{lag:02}_gap_filled']=np.where(~price_ok[:,day],1,np.where(base_reason!=0,base_reason,np.where(gap==0,6,0))).astype(np.uint8)
    x=np.column_stack([values[name] for name in FEATURES]).astype(np.float32)
    why=np.column_stack([reasons[name] for name in FEATURES]).astype(np.uint8)
    if np.isinf(x).any():raise ValueError('Infinite feature')
    if np.any(np.isnan(x)!=(why!=0)):raise ValueError('Missing feature/reason mask mismatch')
    stats=dict(window_price_complete=price_complete,window_activity_complete=activity_complete,
        source_consistent=same,price_days_valid=price_ok.sum(axis=1),activity_days_valid=quantity_ok.sum(axis=1),
        available_gap_anchors=(endpoints&pair_source).sum(axis=1),window_risk_unit=risk,
        complete_feature_vector=np.isfinite(x).all(axis=1),missing_features=np.isnan(x).sum(axis=1))
    for name in ('missing_slots','duplicate_slots','invalid_price_slots','invalid_quantity_slots','partial_buckets'):
        stats[name+'_10']=daily[name][index].sum(axis=1)
    return x,why,stats


class ColumnAudit:
    """Outcome-blind full-release duplicate diagnostics; never selects columns."""
    def __init__(self,names):
        import hashlib
        if len(names)!=len(set(names)):raise ValueError('Duplicate feature names')
        self.names=names;self.hashes=[hashlib.sha256() for _ in names]
        self.finite=np.zeros(len(names),np.int64);self.minimum=np.full(len(names),np.inf);self.maximum=np.full(len(names),-np.inf)
    def update(self,x):
        for i,h in enumerate(self.hashes):
            a=x[:,i].astype(np.float32,copy=True);a[a==0]=0.;a[np.isnan(a)]=np.nan
            h.update(a.tobytes());valid=np.isfinite(a);self.finite[i]+=valid.sum()
            if valid.any():self.minimum[i]=min(self.minimum[i],float(a[valid].min()));self.maximum[i]=max(self.maximum[i],float(a[valid].max()))
    def result(self):
        groups={}
        for name,h in zip(self.names,self.hashes):groups.setdefault(h.hexdigest(),[]).append(name)
        return dict(duplicate_groups=[names for names in groups.values() if len(names)>1],
            constants=[name for i,name in enumerate(self.names) if self.finite[i] and self.minimum[i]==self.maximum[i]],
            all_missing=[name for i,name in enumerate(self.names) if not self.finite[i]],
            column_sha256={name:h.hexdigest() for name,h in zip(self.names,self.hashes)})
