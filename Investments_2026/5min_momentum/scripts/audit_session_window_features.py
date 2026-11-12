"""Independently audit completed ten-session inputs before any training."""
from pathlib import Path
import os,hashlib,json
from datetime import datetime,timezone
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import argparse
import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from session_window_features import FEATURES,make_grid,daily_descriptors,window_matrix
from build_session_window_features import read_month,RELEASE

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def audit(project):
    project=Path(project).resolve();private=(project/'data').resolve()
    root=private/'features'/RELEASE;out=private/'reports/15_build_session_window_features';out.mkdir(parents=True,exist_ok=True)
    m=json.loads((root/'MANIFEST.json').read_text());checks=[]
    def check(name,ok,**details):
        checks.append(dict(check=name,passed=bool(ok),**details))
        print(('PASS ' if ok else 'FAIL ')+name,flush=True)
        if not ok:
            (out/'WINDOW_AUDIT.json').write_text(json.dumps(dict(all_passed=False,checks=checks),indent=2,default=str)+'\n')
            raise AssertionError((name,details))
    check('completed feature-only release',m['complete'] and m['model_fits']==0 and not m['construction_reads_labels'])
    check('267 unique names and exactly nine gap inputs',len(FEATURES)==267 and len(set(FEATURES))==267 and len([n for n in FEATURES if n.endswith('_gap')])==9 and not any(n.startswith('lag10_gap') for n in FEATURES))
    check('all feature release output fingerprints',all(sha(root/name)==digest for name,digest in m['outputs_sha256'].items()))
    check('source fingerprints',all(sha(Path(__file__).with_name(name))==digest for name,digest in m['source_sha256'].items()))
    files={'price_manifest':private/'sp500_5min_20261103/MANIFEST.json',
        'membership':private/'sp500_pit_20261103/membership_by_security_date.parquet',
        'membership_daily':private/'sp500_pit_20261103/membership_daily.parquet',
        'membership_manifest':private/'sp500_pit_20261103/MANIFEST.json'}
    check('membership and price manifest inputs unchanged',all(sha(path)==m['input_sha256'][name] for name,path in files.items()))
    inventory=json.loads((root/'SOURCE_INVENTORY.json').read_text())
    check('all 260 five-minute input object fingerprints',len(inventory)==260 and all(sha(private/obj['path'])==obj['sha256'] for obj in inventory))
    schema=pq.read_schema(root/'features.parquet')
    check('no labels or old feature inputs in panel',schema.names==['security_id','signal_date','trade_date']+FEATURES)
    q=pd.read_parquet(root/'coverage_ledger.parquet');cal=pd.read_parquet(root/'exchange_calendar.parquet')
    check('unique row identities and preserved row count',not q.duplicated(['security_id','signal_date']).any() and len(q)==m['summary']['rows'])
    expected=cal.assign(expected_start=cal.signal_date.shift(9))
    joined=q.merge(expected[['signal_date','expected_start','market_close_utc','next_session']],on='signal_date',validate='many_to_one')
    check('every row has exact ten scheduled sessions and nine gaps',joined.window_start_date.eq(joined.expected_start).all() and joined.window_end_date.eq(joined.signal_date).all() and joined.expected_sessions.eq(10).all() and joined.expected_gaps.eq(9).all())
    check('prior-close feature clock and next-session target date',joined.feature_time_upper_bound_utc.eq(joined.market_close_utc).all() and joined.trade_date.eq(joined.next_session).all() and joined.signal_date.lt(joined.trade_date).all())
    con=duckdb.connect();con.execute('SET threads=4');con.register('sessions',cal)
    p=private/'sp500_pit_20261103/membership_by_security_date.parquet'
    count=con.execute("SELECT count(*) FROM read_parquet(?) WHERE is_member AND index_id='sp500' AND security_id IS NOT NULL AND security_id!='' AND has_conflicting_membership_series=false",[str(p)]).fetchall()[0][0]
    check('all resolved nonconflicting members retained without price/outcome gate',count==len(q),expected_rows=count,actual_rows=len(q))
    duplicates=json.loads((root/'DUPLICATE_AUDIT.json').read_text())
    check('no exact duplicate, constant or all-missing feature columns',not any(duplicates[k] for k in ('duplicate_groups','constants','all_missing')))
    offset=0;finite=np.zeros(267,np.int64)
    for batch in pq.ParquetFile(root/'features.parquet').iter_batches(batch_size=30000):
        f=batch.to_pandas();x=f[FEATURES].to_numpy();quality=q.iloc[offset:offset+len(f)]
        why=np.frombuffer(b''.join(quality.missing_reason_bytes),np.uint8).reshape(len(f),267)
        if np.isinf(x).any() or not np.array_equal(np.isnan(x),why!=0):raise AssertionError('Feature values and missing-reason bytes disagree')
        if not np.array_equal(np.isnan(x).sum(axis=1),quality.missing_features):raise AssertionError('Row missing counts differ')
        for i,name in enumerate(FEATURES):
            good=np.isfinite(x[:,i]);v=x[good,i]
            if name.endswith(('efficiency',)) or '_persist_' in name:
                if np.any(abs(v)>1+1e-6):raise AssertionError('Signed feature out of domain')
            if name.endswith('downside_share') and np.any((v<0)|(v>1+1e-6)):raise AssertionError('Variance share out of domain')
            if name.endswith('gap_filled') and not np.isin(v,[0,1]).all():raise AssertionError('Gap fill indicator out of domain')
            if ('_range_' in name or '_activity_' in name or name.endswith('realized_vol')) and np.any(v<0):raise AssertionError('Nonnegative feature out of domain')
        finite+=np.isfinite(x).sum(axis=0);offset+=len(f)
    distribution=pd.read_csv(root/'feature_coverage.csv')
    check('all stored values have valid units/domains and exact missing reasons',offset==len(q) and np.array_equal(finite,distribution.finite.to_numpy()))
    # Actual-data tests use complete historical windows selected by coverage,
    # never by returns/labels. Include a known XNYS early close and a normal day.
    sample_rows=[]
    for date in ('2020-11-27','2024-03-22'):
        candidates=q[q.signal_date.eq(pd.Timestamp(date))&q.window_price_complete&q.window_activity_complete]
        check(date+' actual complete windows available',len(candidates)>0)
        selected=candidates.head(2)
        month=date[:7]
        a=private/f'sp500_5min_20261103/adjustment=all/month={month}/part-00000.parquet'
        r=private/f'sp500_5min_20261103/adjustment=raw/month={month}/part-00000.parquet'
        bars=read_month(con,a,r)
        codes={s:i+1 for i,s in enumerate(sorted(set(bars.source.dropna())|set(bars.raw_source.dropna())))}
        for field in ('source','raw_source'):bars[field]=bars[field].map(codes).fillna(0).astype(int)
        dense=cal[cal.signal_date.dt.strftime('%Y-%m').eq(month)].reset_index(drop=True)
        end=int(dense.index[dense.signal_date.eq(pd.Timestamp(date))][0]);index=end-9
        if index<1 or end>=len(dense)-1:raise AssertionError('Audit fixture lacks pre/post-window poison dates')
        for row in selected.itertuples():
            grid=make_grid(bars[bars.security_id.eq(row.security_id)],dense)
            x,why,stats=window_matrix(daily_descriptors(grid,dense.expected_slots.to_numpy()))
            original=x[index].copy()
            stored=con.execute('SELECT '+','.join(FEATURES)+' FROM read_parquet(?) WHERE security_id=? AND signal_date=CAST(? AS DATE)',[str(root/'features.parquet'),row.security_id,date]).df().to_numpy(np.float32)[0]
            check(f'{row.security_id}/{date} rebuild matches released values',np.array_equal(original,stored,equal_nan=True))
            poison={k:v.copy() for k,v in grid.items()}
            outside=np.r_[np.arange(index),np.arange(end+1,len(dense))]
            for name in ('open','high','low','close','raw_open','raw_high','raw_low','raw_close','vwap','volume'):poison[name][outside]=1e12
            poison['source'][outside]=999;poison['raw_source'][outside]=999
            altered=window_matrix(daily_descriptors(poison,dense.expected_slots.to_numpy()))
            check(f'{row.security_id}/{date} prewindow and future perturbation invariant',np.array_equal(original,altered[0][index],equal_nan=True) and np.array_equal(why[index],altered[1][index]))
            scale_grid={k:v.copy() for k,v in grid.items()}
            for name in ('open','high','low','close','raw_open','raw_high','raw_low','raw_close','vwap'):scale_grid[name]*=7
            scale_grid['volume']*=1000
            rescaled=window_matrix(daily_descriptors(scale_grid,dense.expected_slots.to_numpy()))[0][index]
            check(f'{row.security_id}/{date} price and volume units invariant',np.allclose(original,rescaled,rtol=1e-5,atol=1e-6,equal_nan=True))
            # Independent direct loops reconstruct risk and anchor returns.
            rv=[]
            for day in range(index,end+1):
                slots=int(dense.expected_slots.iloc[day]);close=grid['close'][day,:slots]
                prev=np.r_[grid['open'][day,0],close[:-1]]
                rv.append(np.sum(np.log(close/prev)**2))
            scale=np.sqrt(np.mean(rv))
            gaps=[np.log(grid['open'][end-lag+1,0]/grid['close'][end-lag,int(dense.expected_slots.iloc[end-lag])-1])/scale for lag in range(1,10)]
            actual=[original[FEATURES.index(f'lag{lag:02}_gap')] for lag in range(1,10)]
            check(f'{row.security_id}/{date} nine direct internal gap anchors',np.allclose(gaps,actual,rtol=1e-6,atol=1e-6))
            for horizon in (5,10,15,30):
                value=np.log(grid['close'][end,horizon//5-1]/grid['open'][end,0])/scale
                check(f'{row.security_id}/{date} {horizon}min closing timestamp anchor',abs(value-original[FEATURES.index(f'lag01_open_{horizon:02}m')])<1e-6)
            sample_rows.append(dict(security_id=row.security_id,signal_date=date,window_start=str(dense.signal_date.iloc[index].date()),expected_slots=int(dense.expected_slots.iloc[end]),window_risk_unit=scale))
    con.close()
    pd.DataFrame(sample_rows).to_csv(out/'actual_data_audit_samples.csv',index=False)
    result=dict(all_passed=True,created_at_utc=datetime.now(timezone.utc).isoformat(),checks=checks,
        audit_source_sha256=sha(__file__),feature_manifest_sha256=sha(root/'MANIFEST.json'),model_fits=0,
        limitations='Numerical construction checks do not certify historical as-known membership/adjustment vintages, complete underlying minutes or executable trading.')
    (out/'WINDOW_AUDIT.json').write_text(json.dumps(result,indent=2,default=str)+'\n')
    print(f'COMPLETE: {len(checks)} construction checks passed; zero fits',flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[1])
    audit(p.parse_args().project)
