"""Build and audit a label-free ten-session feature release; never fits models."""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[_key]='1'
import argparse,hashlib,json,time
from pathlib import Path
from datetime import datetime,timezone
import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from build_momentum_features import session_calendar
from session_window_features import (WINDOW,FEATURES,DAY_NAMES,GAP_NAMES,REASONS,
                                    make_grid,daily_descriptors,window_matrix,ColumnAudit)

RELEASE='15_session_window_10_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,value):Path(p).write_text(json.dumps(value,indent=2,default=str,allow_nan=False)+'\n')


def read_month(con,adjusted,raw):
    """Aggregate exact scheduled slots, preserving invalid/duplicate counts."""
    for alias,path,is_raw in [('a',adjusted,False),('r',raw,True)]:
        con.read_parquet(str(path),hive_partitioning=False).create_view('input_bars',replace=True)
        fields=['open','high','low','close']+(['volume','vwap'] if is_raw else [])
        aggregate=','.join(f'min({name}) {name}' for name in fields)
        con.execute(f'''CREATE OR REPLACE TEMP TABLE {alias} AS
          WITH typed AS (SELECT b.security_id,CAST(b.trading_date AS DATE) signal_date,
            try_cast(b.bar_timestamp_utc AS TIMESTAMPTZ) ts,
            try_cast(b.available_at_utc AS TIMESTAMPTZ) available_at,
            b.open,b.high,b.low,b.close,b.volume::DOUBLE volume,b.vwap,b.minute_bars,b.source
            FROM input_bars b), regular AS (
            SELECT b.*,s.session_num,s.market_open_utc,
              floor(epoch(ts-s.market_open_utc)/300)::INTEGER slot
            FROM typed b JOIN sessions s USING(signal_date)
            WHERE ts>=market_open_utc AND ts<s.market_close_utc)
          SELECT security_id,session_num,slot,count(*) count,{aggregate},
            min(minute_bars) minute_bars,
            CASE WHEN count(DISTINCT source)=1 AND count_if(source IS NULL OR source='')=0
              THEN min(source) END source,
            bool_and(coalesce(ts=market_open_utc+slot*INTERVAL '5 minutes'
              AND ts=date_trunc('second',ts) AND available_at=ts+INTERVAL '5 minutes'
              AND minute_bars BETWEEN 1 AND 5,false)) time_valid
          FROM regular GROUP BY 1,2,3''')
    return con.execute('''SELECT coalesce(a.security_id,r.security_id) security_id,
      coalesce(a.session_num,r.session_num) session_num,coalesce(a.slot,r.slot) slot,
      coalesce(a.count,0) count,a.open,a.high,a.low,a.close,a.minute_bars,a.source,
      coalesce(a.time_valid,false) time_valid,
      coalesce(r.count,0) raw_count,r.open raw_open,r.high raw_high,r.low raw_low,r.close raw_close,
      r.volume,r.vwap,r.minute_bars raw_minute_bars,r.source raw_source,
      coalesce(r.time_valid,false) raw_time_valid
      FROM a FULL OUTER JOIN r USING(security_id,session_num,slot)
      ORDER BY 1,2,3''').df()


def feature_dictionary():
    rows=[]
    for name in FEATURES:
        lag=int(name[3:5]);kind=name[6:]
        unit=('decimal log-return volatility' if kind=='realized_vol' else
              'fraction' if kind=='downside_share' else
              'indicator (zero gaps undefined)' if kind=='gap_filled' else
              'signed path balance' if kind.startswith(('persist_','efficiency')) else
              'relative dollar activity' if kind.startswith('activity_') else 'window intraday risk units')
        rows.append(dict(feature=name,session_lag=lag,descriptor=kind,units=unit,
            reference='Other nine sessions of this same completed window' if kind.startswith('activity_') else
              'RMS of ten daily realized volatilities in this same window' if unit=='window intraday risk units' else 'Own historical session; internal adjacent pair for gaps'))
    return rows


def write_report(out,summary,coverage,distribution):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    yearly=coverage.groupby(coverage.signal_date.dt.year)[['window_price_complete','window_activity_complete','complete_feature_vector']].mean()
    yearly.plot.bar(ax=axes[0]);axes[0].set_title('Resolved member-day coverage');axes[0].yaxis.set_major_formatter(PercentFormatter(1));axes[0].legend(fontsize=7);axes[0].set_xlabel('Signal year')
    lag=distribution.assign(lag=distribution.feature.str.slice(0,5)).groupby('lag').finite_fraction.mean()
    lag.plot.bar(ax=axes[1],color='#2563eb');axes[1].set_title('Mean feature availability by session lag');axes[1].yaxis.set_major_formatter(PercentFormatter(1));axes[1].set_xlabel('lag01 = latest completed session')
    fig.tight_layout();fig.savefig(out/'coverage.png',dpi=150);plt.close(fig)
    text='''<html><head><meta charset="utf-8"><title>Ten-session features</title><style>body{font:15px system-ui;margin:32px;max-width:1400px}td,th{padding:5px;text-align:right}img{max-width:100%}</style></head><body>
    <h1>Session-by-session features: strict ten-session window</h1>
    <p>Feature-only release; no targets, labels, old features, opening-day prices, inherited forecast eligibility or model fitting enter construction. Every resolved, nonconflicting signal-day S&amp;P 500 member remains a row, including incomplete histories. Membership is retrospective, not certified as-known PIT. The old model cohort is compared separately using keys only.</p>
    <p>Exactly ten completed XNYS sessions ending at the signal close; exactly nine internal gaps. The oldest session has no gap, gap response or fill column. No pre-window price, EMA/MACD state, return, normalization or volume reference is used. Trade day is the next exchange session; its opening gap and OHLC are excluded.</p>
    <p>Returns/ranges use the RMS of ten session realized volatilities as a common risk unit. Individual realized volatilities remain in decimal log-return units. Each opening activity value is divided by the median from the other nine sessions of this same window. These references are known at window end; they are not claimed to have been known at each older day's open. Undefined values remain NaN, with exact per-feature reason bytes and row diagnostics. No epsilon floors, price filling, compressed dates or outcome filtering.</p>
    <p>Prices use the published adjusted basis; dollar activity uses raw shares × raw VWAP. Source consistency and scheduled timestamps are checked. Present 1–4-minute underlying buckets are flagged and admitted under the historical five-minute policy. Early-close scheduled lengths are respected. Adjusted-price vintage, membership, quotes/borrow and executable-fill limitations remain.</p>
    <p>Duplicate-column checks are full-release diagnostics only, never future-data feature selection. No column is silently dropped based on observed outcomes or future values. Exact duplicates/constants would require review before training.</p>'''
    text+='<pre>'+json.dumps(summary,indent=2,default=str)+'</pre><img src="coverage.png">'
    text+=yearly.to_html(float_format=lambda x:f'{x:.2%}')
    text+='<h2>Feature coverage and domains</h2>'+distribution.to_html(index=False,float_format=lambda x:f'{x:.5g}')+'</body></html>'
    (out/'FEATURE_AUDIT.html').write_text(text)


def build(project):
    started=time.monotonic();project=Path(project).resolve();private=(project/'data').resolve()
    out=private/'features'/RELEASE
    if out.exists():raise FileExistsError('Immutable feature release exists; inspect it or choose a new version')
    prices=private/'sp500_5min_20261103';members_path=private/'sp500_pit_20261103/membership_by_security_date.parquet'
    price_manifest=json.loads((prices/'MANIFEST.json').read_text())
    sources=[Path(__file__),Path(__file__).with_name('session_window_features.py'),Path(__file__).with_name('build_momentum_features.py')]
    source_hash={p.name:sha(p) for p in sources}
    membership_daily=private/'sp500_pit_20261103/membership_daily.parquet'
    input_hash=dict(price_manifest=sha(prices/'MANIFEST.json'),membership=sha(members_path),
        membership_daily=sha(membership_daily),membership_manifest=sha(private/'sp500_pit_20261103/MANIFEST.json'))
    out.mkdir(parents=True);os.environ['MPLCONFIGDIR']=str(private/'reports/15_build_session_window_features/.matplotlib')
    save(out/'RUN_STATUS.json',dict(status='building',started_at_utc=datetime.now(timezone.utc).isoformat(),model_fits=0))
    con=duckdb.connect();con.execute('SET threads=6');con.execute("SET TimeZone='UTC'")
    members=con.execute('SELECT * FROM read_parquet(?) WHERE is_member AND index_id=\'sp500\'',[str(members_path)]).df()
    members['signal_date']=pd.to_datetime(members.trading_date)
    conflict=members.has_conflicting_membership_series.fillna(True).astype(bool)
    unresolved=members.security_id.isna()|members.security_id.astype(str).str.len().eq(0)
    rejected=members[conflict|unresolved].copy();rejected['exclusion_reason']=np.where(unresolved[conflict|unresolved],'unresolved_security_id','conflicting_membership_series')
    rejected.to_parquet(out/'membership_exclusions.parquet',index=False)
    identity_exclusions=con.execute("SELECT *, 'unresolved_security_id' exclusion_reason FROM read_parquet(?) WHERE is_member AND index_id='sp500' AND (security_id IS NULL OR security_id='')",[str(membership_daily)]).df()
    identity_exclusions.to_parquet(out/'unresolved_membership_rows.parquet',index=False)
    members=members[~conflict&~unresolved].copy()
    if members.duplicated(['security_id','signal_date']).any():raise ValueError('Duplicate membership keys')
    # Calendar predates the first download only to represent initial missing
    # scheduled history. No price data outside the selected ten cells is used.
    calendar=session_calendar('2015-12-01',members.signal_date.max().date().isoformat())
    con.register('sessions',calendar)
    members=members.merge(calendar[['signal_date','session_num','next_session']],on='signal_date',validate='many_to_one')
    if members.session_num.isna().any():raise ValueError('Membership on a non-exchange date')
    members['month']=members.signal_date.dt.strftime('%Y-%m')
    calendar.to_parquet(out/'exchange_calendar.parquet',index=False)
    manifest_objects={str(Path(x['path']).relative_to(prices)):x for x in price_manifest['objects']}
    paths=sorted((prices/'adjustment=all').glob('month=*/*.parquet'))
    column_audit=ColumnAudit(FEATURES);source_codes={};carry=pd.DataFrame();coverage=[];inventory=[]
    writer=None;rows=0
    try:
        for number,adjusted in enumerate(paths,1):
            month=adjusted.parent.name.removeprefix('month=');raw=prices/'adjustment=raw'/adjusted.parent.name/adjusted.name
            for path in (adjusted,raw):
                obj=manifest_objects[str(path.relative_to(prices))];digest=sha(path)
                if digest!=obj['sha256'].removeprefix('sha256:'):raise ValueError('Changed raw five-minute input '+str(path))
                inventory.append(dict(path=str(path.relative_to(private)),sha256=digest,bytes=path.stat().st_size))
            current=read_month(con,adjusted,raw)
            for field in ('source','raw_source'):
                for name in current[field].dropna().unique():
                    if name and name not in source_codes:source_codes[name]=len(source_codes)+1
                current[field]=current[field].map(source_codes).fillna(0).astype(int)
            month_calendar=calendar[calendar.signal_date.dt.strftime('%Y-%m').eq(month)]
            lo,hi=int(month_calendar.session_num.min()),int(month_calendar.session_num.max())
            dense=calendar[calendar.session_num.between(lo-9,hi)].reset_index(drop=True)
            bars=pd.concat([carry,current],ignore_index=True) if len(carry) else current
            groups={str(s):g for s,g in bars.groupby('security_id',sort=False)}
            wanted=members[members.month.eq(month)]
            matrices=[];metadata=[]
            for security,targets in wanted.groupby('security_id',sort=True):
                frame=groups.get(str(security),bars.iloc[:0])
                grid=make_grid(frame,dense)
                descriptors=daily_descriptors(grid,dense.expected_slots.to_numpy())
                x,reason,stats=window_matrix(descriptors)
                index=targets.session_num.to_numpy(int)-int(dense.session_num.iloc[0])-9
                if np.any((index<0)|(index>=len(x))):raise ValueError('Window index outside calendar')
                matrices.append(x[index])
                end=dense.iloc[index+9]
                start=dense.iloc[index]
                meta=pd.DataFrame(dict(security_id=targets.security_id.to_numpy(),signal_date=targets.signal_date.to_numpy(),
                    trade_date=targets.next_session.to_numpy(),window_start_date=start.signal_date.to_numpy(),
                    window_end_date=end.signal_date.to_numpy(),feature_time_upper_bound_utc=end.market_close_utc.to_numpy(),
                    expected_sessions=10,expected_gaps=9,
                    source_history_warmup=start.signal_date.lt(pd.Timestamp('2016-01-04')).to_numpy(),
                    missing_reason_bytes=[a.tobytes() for a in reason[index]]))
                for name,values in stats.items():meta[name]=values[index]
                metadata.append(meta)
            if matrices:
                matrix=np.concatenate(matrices);meta=pd.concat(metadata,ignore_index=True)
                column_audit.update(matrix)
                feature_frame=pd.DataFrame(matrix,columns=FEATURES)
                table=pa.Table.from_pandas(pd.concat([meta[['security_id','signal_date','trade_date']],feature_frame],axis=1),preserve_index=False)
                if writer is None:writer=pq.ParquetWriter(out/'features.parquet',table.schema,compression='zstd')
                writer.write_table(table)
                coverage.append(meta);rows+=len(meta)
            # Exactly nine previous scheduled sessions may cross a month edge.
            carry=current[current.session_num.ge(hi-8)].copy()
            save(out/'RUN_STATUS.json',dict(status='building',months=number,total_months=len(paths),rows=rows,
                elapsed_seconds=time.monotonic()-started,model_fits=0))
            if number==1 or number%12==0 or number==len(paths):print(f'SESSION WINDOWS {number}/{len(paths)} months; {rows:,} rows; {time.monotonic()-started:.1f}s',flush=True)
    finally:
        if writer is not None:writer.close()
        con.close()
    if rows!=len(members):raise AssertionError('Feature construction dropped member rows')
    quality=pd.concat(coverage,ignore_index=True)
    if quality.duplicated(['security_id','signal_date']).any():raise AssertionError('Duplicate output keys')
    quality.to_parquet(out/'coverage_ledger.parquet',index=False)
    duplicate_audit=column_audit.result()
    # Cryptographic hashes identify candidates; verify matching columns against
    # every stored row before reporting any actual exact duplicates.
    for group in duplicate_audit['duplicate_groups']:
        arrays=pq.ParquetFile(out/'features.parquet').read(columns=group).to_pandas()
        if not all(np.array_equal(arrays[group[0]],arrays[name],equal_nan=True) for name in group[1:]):raise AssertionError('Column hash collision')
    distribution=pd.DataFrame(dict(feature=FEATURES,finite=column_audit.finite,
        finite_fraction=column_audit.finite/rows,missing=rows-column_audit.finite,
        minimum=np.where(np.isfinite(column_audit.minimum),column_audit.minimum,np.nan),
        maximum=np.where(np.isfinite(column_audit.maximum),column_audit.maximum,np.nan)))
    distribution.to_csv(out/'feature_coverage.csv',index=False)
    pd.DataFrame(feature_dictionary()).to_csv(out/'feature_dictionary.csv',index=False)
    save(out/'DUPLICATE_AUDIT.json',duplicate_audit);save(out/'MISSING_REASONS.json',REASONS)
    # Coverage comparison reads keys only; it never filters construction rows or
    # imports labels, old features or legacy lookback-dependent eligibility.
    legacy_path=private/'features/10_rich_5min_path_v1/supervised_panel.parquet'
    legacy=pq.ParquetFile(legacy_path).read(columns=['security_id','signal_date','trade_date']).to_pandas()
    input_hash['legacy_comparison_keys']=hashlib.sha256(pd.util.hash_pandas_object(legacy,index=False).to_numpy().tobytes()).hexdigest()
    joined=legacy.merge(quality,on=['security_id','signal_date','trade_date'],how='left',validate='one_to_one')
    if joined.expected_sessions.isna().any():raise AssertionError('Legacy row not represented in broad member feature panel')
    joined.to_parquet(out/'legacy_cohort_coverage.parquet',index=False)
    summary=dict(rows=rows,feature_columns=len(FEATURES),resolved_member_rows=len(members),membership_exclusions=len(rejected),
        unresolved_membership_series_rows=len(identity_exclusions),
        duplicate_keys=0,duplicate_feature_names=0,exact_duplicate_column_groups=len(duplicate_audit['duplicate_groups']),
        constant_columns=len(duplicate_audit['constants']),all_missing_columns=len(duplicate_audit['all_missing']),
        complete_price_windows=int(quality.window_price_complete.sum()),complete_activity_windows=int(quality.window_activity_complete.sum()),
        complete_feature_vectors=int(quality.complete_feature_vector.sum()),windows_with_partial_buckets=int(quality.partial_buckets_10.gt(0).sum()),
        warmup_rows=int(quality.source_history_warmup.sum()),legacy_rows=len(legacy),
        legacy_complete_price_windows=int(joined.window_price_complete.sum()),legacy_complete_feature_vectors=int(joined.complete_feature_vector.sum()))
    write_report(out,summary,quality,distribution)
    if input_hash['membership']!=sha(members_path) or input_hash['price_manifest']!=sha(prices/'MANIFEST.json'):raise AssertionError('Input changed during construction')
    if source_hash!={p.name:sha(p) for p in sources}:raise AssertionError('Sources changed during construction')
    save(out/'SOURCE_INVENTORY.json',inventory)
    outputs={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name not in ('MANIFEST.json','RUN_STATUS.json')}
    save(out/'MANIFEST.json',dict(complete=True,created_at_utc=datetime.now(timezone.utc).isoformat(),summary=summary,
        model_features=FEATURES,source_sha256=source_hash,input_sha256=input_hash,outputs_sha256=outputs,
        source_codes=source_codes,scope_start=str(members.signal_date.min().date()),scope_end=str(members.signal_date.max().date()),
        window_sessions=10,internal_overnight_gaps=9,current_trade_day_inputs=False,older_smoothing_state=False,
        construction_reads_labels=False,cohort='All resolved nonconflicting signal-day members; no legacy eligibility or target-outcome filtering',
        legacy_comparison='Keys-only coverage diagnostic; no impact on construction population',
        normalization='Window-end RMS of ten daily realized volatilities; activity uses the other nine sessions inside this same completed window',
        missing_policy='Native NaNs; exact per-feature reason bytes in coverage ledger; no filling, epsilon floors, compressed dates or row drops',
        price_basis='Published adjustment=all OHLC; raw volume times raw VWAP for relative activity',
        duplicate_policy='Diagnostics only; no full-history data-driven column pruning',
        construction_checks_passed=not any(duplicate_audit[k] for k in ('duplicate_groups','constants','all_missing')),
        model_fits=0,one_minute_data_used=False,elapsed_seconds=time.monotonic()-started,
        usage_rights='Internal Galambos Capital only; inherited vendor licenses and retrospective vintage/PIT limitations'))
    save(out/'RUN_STATUS.json',dict(status='complete',completed_at_utc=datetime.now(timezone.utc).isoformat(),rows=rows,model_fits=0,elapsed_seconds=time.monotonic()-started))
    print(json.dumps(summary,indent=2),flush=True)
    return out

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args()
    try:build(args.project)
    except Exception as error:
        failed=(args.project/'data').resolve()/'features'/RELEASE
        if failed.exists() and not (failed/'MANIFEST.json').exists():
            save(failed/'RUN_STATUS.json',dict(status='failed',updated_at_utc=datetime.now(timezone.utc).isoformat(),error=repr(error),model_fits=0))
        raise
