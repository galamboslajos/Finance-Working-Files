"""Frozen-threshold diagnostics; flags never change prices or research eligibility.

Canonical input is already date/ID bounded by the caller. This worker accepts no
paths and performs no IO, fitting, provider lookup, clipping or price repair.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any
import numpy as np
import pandas as pd

KEY = 'intrinio_security_id'
OHLC = ('open', 'high', 'low', 'close')
ADJ = tuple('adj_' + x for x in OHLC)
FLAG_COLUMNS = [KEY, 'date', 'reason', 'span_start', 'span_end', 'observations',
                'event_cluster', 'evidence_id', 'evidence_json', 'disposition']

@dataclass(frozen=True)
class AuditPolicy:
    version: str = 'daily-history-diagnostic-v1'
    protected_start: str = '2024-01-01'
    single_move_fraction: float = .30
    repeated_ratio_separation: float = .20
    repeated_window: int = 20
    repeated_switches: int = 4
    roundtrip_max_observations: int = 3
    roundtrip_return_tolerance: float = .10
    stale_min_observations: int = 5
    within_bar_adjustment_spread: float = .01
    adjustment_factor_jump: float = .20
    split_factor_tolerance: float = .05
    split_ratio_convention: str = 'unspecified'
    max_input_rows: int = 12_000_000
    max_flags: int = 500_000

@dataclass
class AuditResult:
    flags: pd.DataFrame
    summary: dict[str, Any]


def audit_daily_history(frame: pd.DataFrame, *, expected_sessions=None,
                        policy: AuditPolicy = AuditPolicy(), provenance=None) -> AuditResult:
    """Screen immutable daily observations, returning review candidates and counts.

    Required: stable native ID, date, raw and adjusted OHLC. Optional frequency,
    intraperiod, split_ratio (new/old only if explicitly declared), dividend,
    volume, source_version, source_hash, overlay_status are retained as evidence.
    The caller supplies canonical source/overlay selection and approved read
    bounds. Conflicting ID/date observations are flagged and never chosen for
    temporal diagnostics. Equal cache copies from distinct source versions are
    collapsed for diagnostics; duplicate rows within a version remain flagged.
    Calendar holes are only between each ID's first/last observed valid date.
    """
    required = {KEY, 'date', *OHLC, *ADJ}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError('Missing canonical columns: ' + ', '.join(sorted(missing)))
    if len(frame) > policy.max_input_rows:
        raise ValueError('Input row budget exceeded; no partial result returned')
    if policy.split_ratio_convention not in {'unspecified', 'new_shares_per_old'}:
        raise ValueError('Unreviewed split ratio convention')
    dates = pd.to_datetime(frame.date, errors='coerce')
    if dates.dt.tz is not None:
        raise ValueError('Dates must be timezone-naive canonical session dates')
    if dates.ge(pd.Timestamp(policy.protected_start)).any():
        raise ValueError('Protected 2024+ dates refused before financial conversion')
    if ((dates.notna()) & (dates != dates.dt.normalize())).any():
        raise ValueError('Dates must be canonical date-only observations')
    x = frame.copy(deep=True)
    x['date'] = dates
    flags = []
    def add(native, date, reason, *, start=None, end=None, n=1, **evidence):
        if len(flags) >= policy.max_flags:
            raise ValueError('Flag budget exceeded; no silently truncated result')
        def ds(v): return None if v is None or pd.isna(v) else pd.Timestamp(v).strftime('%Y-%m-%d')
        date_s, start_s, end_s = ds(date), ds(start if start is not None else date), ds(end if end is not None else date)
        native = None if pd.isna(native) else str(native)
        cluster = hashlib.sha256(json.dumps([native, reason, start_s, end_s], separators=(',', ':')).encode()).hexdigest()[:20]
        flags.append({KEY:native, 'date':date_s, 'reason':reason, 'span_start':start_s,
                      'span_end':end_s, 'observations':int(n), 'event_cluster':cluster, 'evidence_id':cluster,
                      'evidence_json':json.dumps(evidence, sort_keys=True, default=str, allow_nan=False),
                      'disposition':'REVIEW_ONLY_NO_DATA_CHANGE'})
    badkey = x[KEY].isna() | x[KEY].astype('string').str.strip().eq('').fillna(True) | x.date.isna()
    for row in x.loc[badkey, [KEY, 'date']].itertuples(index=False, name=None):
        add(row[0], row[1], 'null_or_invalid_key')
    x = x.loc[~badkey].copy()
    x[KEY] = x[KEY].astype(str)
    compare = [*OHLC, *ADJ] + [c for c in ['frequency', 'intraperiod', 'split_ratio', 'dividend', 'volume'] if c in x]
    copies = 0
    ambiguous = set()
    dup = x.duplicated([KEY, 'date'], keep=False)
    for (native, date), group in x.loc[dup].groupby([KEY, 'date'], sort=True):
        distinct = group[compare].drop_duplicates()
        versions = group.source_version.nunique(dropna=False) if 'source_version' in group else 1
        version_dup = ('source_version' not in group or group.duplicated(['source_version'], keep=False).any())
        if len(distinct) > 1:
            ambiguous.add((native, date))
            add(native, date, 'conflicting_duplicate_observation', n=len(group), distinct_payloads=len(distinct),
                source_versions=int(versions))
        elif version_dup:
            add(native, date, 'duplicate_observation', n=len(group), distinct_payloads=1, source_versions=int(versions))
        else:
            copies += len(group)-1
    if ambiguous:
        mask = pd.Series([(a,b) not in ambiguous for a,b in zip(x[KEY],x.date)], index=x.index)
        x = x.loc[mask].copy()
    x = x.drop_duplicates([KEY, 'date'], keep='first').sort_values([KEY, 'date'], kind='stable')
    non_daily = pd.Series(False,index=x.index)
    if 'frequency' in x:
        non_daily |= x.frequency.astype('string').str.lower().ne('daily').fillna(True)
    if 'intraperiod' in x:
        non_daily |= x.intraperiod.fillna(False).astype(bool)
    for row in x.loc[non_daily,[KEY,'date']].itertuples(index=False,name=None):
        add(*row,'non_daily_observation')
    x = x.loc[~non_daily].copy()
    for c in OHLC+ADJ:
        x[c] = pd.to_numeric(x[c], errors='coerce')
    for c in ['split_ratio','dividend','volume']:
        if c in x: x[c] = pd.to_numeric(x[c], errors='coerce')
    sessions = None
    if expected_sessions is not None:
        sessions = pd.DatetimeIndex(pd.to_datetime(expected_sessions, errors='raise')).normalize()
        if sessions.has_duplicates or not sessions.is_monotonic_increasing:
            raise ValueError('Expected sessions must be unique and sorted')
        if sessions.tz is not None or (sessions >= pd.Timestamp(policy.protected_start)).any():
            raise ValueError('Expected sessions cross timezone/protected bound')
    coverage = []
    for native, g in x.groupby(KEY,sort=True):
        g = g.reset_index(drop=True)
        n = len(g)
        days = g.date.to_numpy()
        a = g[list(OHLC)].to_numpy(float)
        b = g[list(ADJ)].to_numpy(float)
        valid_raw = np.isfinite(a).all(axis=1) & (a>0).all(axis=1)
        valid_adj = np.isfinite(b).all(axis=1) & (b>0).all(axis=1)
        for i in np.flatnonzero(~valid_raw | ~valid_adj):
            add(native,days[i],'invalid_or_missing_ohlc',raw_valid=bool(valid_raw[i]),adjusted_valid=bool(valid_adj[i]))
        rawbounds = valid_raw & (a[:,2] <= np.minimum(a[:,0],a[:,3])) & (a[:,1] >= np.maximum(a[:,0],a[:,3])) & (a[:,2] <= a[:,1])
        adjbounds = valid_adj & (b[:,2] <= np.minimum(b[:,0],b[:,3])) & (b[:,1] >= np.maximum(b[:,0],b[:,3])) & (b[:,2] <= b[:,1])
        for label, valid, bounds in [('raw',valid_raw,rawbounds),('adjusted',valid_adj,adjbounds)]:
            for i in np.flatnonzero(valid & ~bounds): add(native,days[i],label+'_ohlc_bounds')
        with np.errstate(divide='ignore',invalid='ignore'):
            factors=b/a
            spread=np.nanmax(factors,axis=1)/np.nanmin(factors,axis=1)-1
        for i in np.flatnonzero(valid_raw & valid_adj & (spread > policy.within_bar_adjustment_spread)):
            add(native,days[i],'within_bar_adjustment_conflict',ratio_spread=float(spread[i]))
        for name, values, valid in [('raw',a,valid_raw),('adjusted',b,valid_adj)]:
            close = values[:,3]
            with np.errstate(divide='ignore',invalid='ignore'):
                ret=close[1:]/close[:-1]-1
                gap=values[1:,0]/close[:-1]-1
                switches=np.maximum(close[1:]/close[:-1],close[:-1]/close[1:])-1
            pairvalid=valid[1:] & valid[:-1]
            for kind, arr in [('close_move',ret),('open_gap',gap)]:
                for j in np.flatnonzero(pairvalid & (np.abs(arr)>=policy.single_move_fraction)):
                    i=j+1;add(native,days[i],name+'_'+kind+'_30pct',start=days[i-1],previous_close=float(close[i-1]),observed=float(values[i,0] if kind=='open_gap' else close[i]),fraction=float(arr[j]))
            # Revisit an earlier level after a large excursion, bounded to <=3 observed rows.
            for i in range(1,n-1):
                if not valid[i-1] or not valid[i]: continue
                if max(close[i]/close[i-1],close[i-1]/close[i])-1 < policy.repeated_ratio_separation: continue
                for j in range(i+1,min(n,i+policy.roundtrip_max_observations+1)):
                    if not valid[j]: break
                    if abs(close[j]/close[i-1]-1)<=policy.roundtrip_return_tolerance:
                        add(native,days[j],name+'_roundtrip_regime_candidate',start=days[i-1],n=j-i+2,anchor=float(close[i-1]),excursion=float(close[i]),return_level=float(close[j]));break
            large=pd.Series(pairvalid & (switches>=policy.repeated_ratio_separation))
            rolling=large.rolling(policy.repeated_window,min_periods=policy.repeated_switches).sum().to_numpy()
            active=rolling>=policy.repeated_switches
            for j in np.flatnonzero(active & ~np.r_[False,active[:-1]]):
                i=j+1;left=max(0,i-policy.repeated_window)
                add(native,days[i],name+'_repeated_regime_switches',start=days[left],n=i-left+1,switch_count=int(rolling[j]),switch_fraction=policy.repeated_ratio_separation)
            # Identical entire bars only; missing dates interrupt staleness where a calendar exists.
            same=(values[1:]==values[:-1]).all(axis=1) & pairvalid
            if sessions is not None:
                ix=sessions.get_indexer(pd.DatetimeIndex(g.date));same &= (ix[1:]==ix[:-1]+1) & (ix[1:]>=0) & (ix[:-1]>=0)
            run_start=0
            for i in range(1,n+1):
                if i<n and same[i-1]: continue
                if i-run_start>=policy.stale_min_observations:
                    add(native,days[i-1],name+'_stale_identical_ohlc_run',start=days[run_start],n=i-run_start,close=float(close[run_start]))
                run_start=i
        with np.errstate(divide='ignore',invalid='ignore'):
            f=factors[:,3];fchange=np.maximum(f[1:]/f[:-1],f[:-1]/f[1:])-1
        fv=valid_raw & valid_adj
        for j in np.flatnonzero(fv[1:] & fv[:-1] & (fchange>=policy.adjustment_factor_jump)):
            i=j+1;add(native,days[i],'crossday_adjustment_factor_jump',start=days[i-1],factor_before=float(f[i-1]),factor_after=float(f[i]))
        if 'split_ratio' in g:
            splits=g.split_ratio.to_numpy(float)
            for i in np.flatnonzero(np.isfinite(splits) & (splits!=1)):
                if splits[i]<=0:
                    add(native,days[i],'invalid_split_ratio',split_ratio=float(splits[i]));continue
                add(native,days[i],'declared_split_review',split_ratio=float(splits[i]),convention=policy.split_ratio_convention)
                if policy.split_ratio_convention=='new_shares_per_old' and i>0 and fv[i] and fv[i-1]:
                    factor_jump=f[i]/f[i-1]
                    residual=abs(factor_jump/splits[i]-1)
                    if residual>policy.split_factor_tolerance:
                        add(native,days[i],'split_adjustment_factor_conflict',factor_jump=float(factor_jump),split_ratio=float(splits[i]),relative_residual=float(residual))
        if 'volume' in g:
            for i in np.flatnonzero(np.isfinite(g.volume.to_numpy(float)) & (g.volume.to_numpy(float)<0)):
                add(native,days[i],'negative_volume')
        if 'dividend' in g:
            for i in np.flatnonzero(np.isfinite(g.dividend.to_numpy(float)) & (g.dividend.to_numpy(float)<0)):
                add(native,days[i],'negative_dividend_review',dividend=float(g.dividend.iloc[i]))
        missing_count=0
        if sessions is not None and n:
            expected=sessions[(sessions>=g.date.iloc[0])&(sessions<=g.date.iloc[-1])]
            miss=expected.difference(pd.DatetimeIndex(g.date));missing_count=len(miss)
            # One flag per contiguous missing-session cluster, preserving exact dates as evidence.
            if len(miss):
                ix=sessions.get_indexer(miss);cuts=np.r_[0,np.flatnonzero(np.diff(ix)>1)+1,len(miss)]
                for left,right in zip(cuts[:-1],cuts[1:]):
                    dd=miss[left:right];add(native,dd[-1],'missing_session_cluster',start=dd[0],n=len(dd),missing_dates=[d.strftime('%Y-%m-%d') for d in dd])
            for day in pd.DatetimeIndex(g.date).difference(sessions):add(native,day,'non_session_observation')
        coverage.append({KEY:native,'first_observation':g.date.min().strftime('%Y-%m-%d'),'last_observation':g.date.max().strftime('%Y-%m-%d'),'diagnostic_observations':n,'missing_sessions_within_observed_extent':missing_count})
    ff=pd.DataFrame(flags,columns=FLAG_COLUMNS)
    if len(ff):
        ff=ff.drop_duplicates([KEY,'date','reason','span_start','span_end']).sort_values([KEY,'date','reason'],na_position='first',kind='stable').reset_index(drop=True)
        # Merge overlapping evidence intervals per native ID into review episodes.
        # Raw/adjusted and multiple reasons for one interval are not separate cases.
        for native, group in ff.groupby(KEY, dropna=False, sort=True):
            group=group.sort_values(['span_start','span_end','reason'],na_position='first')
            pending=[]; left=right=None
            def assign_cluster(indices, lo, hi):
                if not indices: return
                payload=[None if pd.isna(native) else str(native),lo,hi]
                value=hashlib.sha256(json.dumps(payload,separators=(',',':')).encode()).hexdigest()[:20]
                ff.loc[indices,'event_cluster']=value
            for row in group.itertuples():
                start,end=row.span_start,row.span_end
                if start is None or end is None:
                    continue
                if pending and start > right:
                    assign_cluster(pending,left,right);pending=[]
                if not pending: left,right=start,end
                else: right=max(right,end)
                pending.append(row.Index)
            assign_cluster(pending,left,right)
        yy=pd.to_datetime(ff.date,errors='coerce').dt.year.astype('Int64').astype('string').fillna('invalid_date')
        byyear=ff.groupby(yy,dropna=False).agg(flag_records=('event_cluster','size'),distinct_native_ids=(KEY,'nunique'),distinct_event_clusters=('event_cluster','nunique')).reset_index(names='year').to_dict('records')
        byreason=ff.groupby('reason').agg(flag_records=('event_cluster','size'),distinct_native_ids=(KEY,'nunique')).reset_index().to_dict('records')
        bynative=ff.groupby(KEY,dropna=False).agg(flag_records=('event_cluster','size'),distinct_dates=('date','nunique'),distinct_reasons=('reason','nunique')).reset_index().to_dict('records')
    else: byyear=byreason=bynative=[]
    for row in bynative:
        if pd.isna(row[KEY]): row[KEY]=None
    policy_dict=asdict(policy)
    summary={'status':'FLAG_AUDIT_ONLY_NOT_ERROR_COUNT','policy':policy_dict,'policy_sha256':hashlib.sha256(json.dumps(policy_dict,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
             'input_rows':len(frame),'null_key_rows':int(badkey.sum()),'collapsed_cross_version_equal_copies':copies,'ambiguous_native_date_keys':len(ambiguous),'diagnostic_unique_daily_rows':len(x),
             'flag_records':len(ff),'distinct_flagged_native_ids':int(ff[KEY].nunique()),'distinct_native_date_keys':len(ff[[KEY,'date']].drop_duplicates()),'distinct_event_clusters':int(ff.event_cluster.nunique()),
             'by_year':byyear,'by_reason':byreason,'by_native_id':bynative,'observed_extent_coverage':coverage,'provenance':provenance or {},
             'limits':['Flags identify review candidates, never confirmed errors or automatic exclusions.','No source edits, price fills, winsorization, model fitting, or return recalculation.','Calendar holes are limited to observed extents; listing/delisting and temporary halts need separate evidence.','No flag does not establish source correctness, identity correctness, PIT knowledge time, or survivorship completeness.','Duplicate conflicts are not resolved for temporal screening; candidate counts remain distinct from cache-copy counts.','Selected-portfolio, execution and exact-month-end impact intersections must be joined by the caller using original stable IDs.','Corporate actions, plausible crashes and valid highly volatile histories can trigger review flags.']}
    return AuditResult(ff,summary)
