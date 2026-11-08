"""Add six volume variables and an optimistic morning gap to frozen eight inputs.

Only local five-minute price files and existing derived panels are read. No
one-minute data, downloads, execution simulation or model fitting occurs.
"""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import resource
import time

import duckdb
import numpy as np
import pandas as pd

from build_momentum_features import session_calendar
from build_network_paper_features import MODEL_FEATURES as MOMENTUM_FEATURES
from volume_features import VOLUME_FEATURES,make_volume_features

MODEL_FEATURES=MOMENTUM_FEATURES+VOLUME_FEATURES+["overnight_gap"]
FEATURE_SETS={"momentum8":MOMENTUM_FEATURES,
              "momentum_volume14":MOMENTUM_FEATURES+VOLUME_FEATURES,
              "momentum_volume_gap15":MODEL_FEATURES}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,default=str,allow_nan=False)+'\n')


def aggregate_volume_month(con,path,first=False):
    con.read_parquet(str(path),hive_partitioning=False).create_view('month_bars',replace=True)
    query="""WITH typed AS (
      SELECT b.security_id,CAST(b.trading_date AS DATE) signal_date,b.ticker,
        try_cast(b.bar_timestamp_utc AS TIMESTAMPTZ) ts,
        try_cast(b.available_at_utc AS TIMESTAMPTZ) available_at,
        b.open,b.close,b.vwap,b.volume,b.trade_count,b.source,
        s.market_open_utc,s.market_close_utc,s.expected_slots
      FROM month_bars b JOIN sessions s ON CAST(b.trading_date AS DATE)=s.signal_date
    ), regular AS (
      SELECT *,coalesce(volume>=0 AND (volume=0 OR (isfinite(vwap) AND vwap>0))
          AND available_at=ts+INTERVAL '5 minutes'
          AND epoch(ts)::BIGINT%300=0,false) quantity_ok
      FROM typed WHERE ts>=market_open_utc AND ts<market_close_utc
    ), values_ AS (
      SELECT *,CASE WHEN quantity_ok AND volume=0 THEN 0.
        WHEN quantity_ok THEN volume::DOUBLE*vwap END dollars,
        coalesce(volume=0 OR (isfinite(open) AND open>0
          AND isfinite(close) AND close>0),false) direction_ok
      FROM regular
    )
    SELECT security_id,signal_date,min(ticker) ticker,
      count(*) rth_rows,count(DISTINCT ts) observed_slots,
      count_if(NOT quantity_ok OR NOT isfinite(dollars)) invalid_quantity_rows,
      count_if(NOT direction_ok) invalid_direction_rows,
      sum(dollars) dollar_volume,sum(volume::DOUBLE) share_volume,
      sum(trade_count::DOUBLE) trade_count,
      sum(dollars) FILTER (WHERE ts>=market_close_utc-INTERVAL '60 minutes')
        last_hour_dollar_volume,
      sum(CASE WHEN dollars=0 THEN 0. WHEN direction_ok
        THEN sign(close-open)*dollars END) signed_dollar_volume,
      count_if(ts=market_close_utc-INTERVAL '5 minutes') closing_rows,
      max(close) FILTER (WHERE ts=market_close_utc-INTERVAL '5 minutes') closing_price_raw,
      count(DISTINCT source) source_count,min(source) raw_source,
      max(available_at) volume_available_at_utc
    FROM values_ GROUP BY 1,2"""
    con.execute(('CREATE TEMP TABLE daily_volume AS ' if first else 'INSERT INTO daily_volume ')+query)


def aggregate_opening_month(con,path,first=False):
    con.read_parquet(str(path),hive_partitioning=False).create_view('month_open_bars',replace=True)
    query="""SELECT b.security_id,CAST(b.trading_date AS DATE) trade_date,
      count(*) opening_rows,
      CASE WHEN count(*)=1 AND bool_and(isfinite(b.open) AND b.open>0)
        THEN max(b.open) END gap_opening_price,
      CASE WHEN count(*)=1 THEN min(b.source) END opening_source,
      max(try_cast(b.available_at_utc AS TIMESTAMPTZ)) opening_bar_available_at_utc
      FROM month_open_bars b JOIN sessions s ON CAST(b.trading_date AS DATE)=s.signal_date
      WHERE try_cast(b.bar_timestamp_utc AS TIMESTAMPTZ)=s.market_open_utc
      GROUP BY 1,2"""
    con.execute(('CREATE TEMP TABLE opening AS ' if first else 'INSERT INTO opening ')+query)


def build(project, output=None):
    project=Path(project).resolve()
    private=(project/'data').resolve()
    out=Path(output).resolve() if output is not None else private/'features/06_momentum_volume_overnight_v1'
    if not out.is_relative_to(private):
        raise ValueError('Feature releases must remain inside private project data/.')
    parent_features=private/'features/05_network_individual_8_v1'
    parent_run=private/'models/05_xgb_ret_network_features100_uncapped_v1'
    parent_manifest=json.loads((parent_features/'MANIFEST.json').read_text())
    hashes=dict(builder_sha256=sha(__file__),
                volume_math_sha256=sha(Path(__file__).with_name('volume_features.py')),
                quality_math_sha256=sha(Path(__file__).with_name('research_checks.py')),
                parent_feature_manifest_sha256=sha(parent_features/'MANIFEST.json'),
                parent_panel_sha256=sha(parent_run/'supervised_panel.parquet'))
    if (out/'MANIFEST.json').exists():
        old=json.loads((out/'MANIFEST.json').read_text())
        if old['input_hashes']!=hashes:
            raise FileExistsError('Completed release uses different inputs/code; choose a new release.')
        print('Reusing completed volume/overnight release.',flush=True)
        return out
    started=time.monotonic()
    out.mkdir(parents=True,exist_ok=True)
    con=duckdb.connect()
    con.execute('SET threads=10')
    con.execute("SET TimeZone='UTC'")
    con.execute('SET temp_directory=?',[str(out/'.duckdb_tmp')])
    sessions=session_calendar(parent_manifest['scope_start'],parent_manifest['scope_end'])
    con.register('sessions',sessions)
    raw_paths=sorted((private/'sp500_5min_20261103/adjustment=raw').glob('month=*/*.parquet'))
    adjusted_paths=sorted((private/'sp500_5min_20261103/adjustment=all').glob('month=*/*.parquet'))
    if not raw_paths or len(raw_paths)!=len(adjusted_paths):
        raise ValueError('Expected aligned raw and adjusted monthly five-minute files.')
    inventory=[]
    for k,(raw,adjusted) in enumerate(zip(raw_paths,adjusted_paths),1):
        if raw.parent.name!=adjusted.parent.name:
            raise ValueError('Raw/adjusted month mismatch.')
        aggregate_volume_month(con,raw,first=k==1)
        aggregate_opening_month(con,adjusted,first=k==1)
        inventory += [dict(path=str(p.relative_to(private)),bytes=p.stat().st_size,
                            mtime_ns=p.stat().st_mtime_ns) for p in [raw,adjusted]]
        if k==1 or k%12==0 or k==len(raw_paths):
            print(f'VOLUME {k}/{len(raw_paths)} months; {time.monotonic()-started:.1f}s',flush=True)
    for table,key in [('daily_volume','security_id,signal_date'),('opening','security_id,trade_date')]:
        duplicates=con.execute(f'SELECT count(*) FROM (SELECT {key} FROM {table} GROUP BY ALL HAVING count(*)>1)').fetchone()[0]
        if duplicates:
            raise ValueError('Duplicate month-boundary keys in '+table)
    con.execute('''CREATE TEMP TABLE daily_ready AS SELECT d.*,s.expected_slots,
      epoch(s.market_close_utc-s.market_open_utc)/60. session_minutes,
      coalesce(d.rth_rows=s.expected_slots AND d.observed_slots=s.expected_slots
        AND d.invalid_quantity_rows=0 AND isfinite(d.dollar_volume)
        AND d.dollar_volume>=0 AND d.volume_available_at_utc<=s.market_close_utc,false) activity_valid,
      coalesce(d.invalid_direction_rows=0,false) signed_valid
      FROM daily_volume d JOIN sessions s USING(signal_date)''')
    con.execute('COPY (SELECT * FROM daily_ready ORDER BY signal_date,security_id) TO ? (FORMAT PARQUET,COMPRESSION ZSTD)',
                [str(out/'daily_volume_anchors.parquet')])
    con.execute('COPY (SELECT * FROM opening ORDER BY trade_date,security_id) TO ? (FORMAT PARQUET,COMPRESSION ZSTD)',
                [str(out/'opening_anchors.parquet')])
    con.read_parquet(str(parent_features/'features.parquet')).create_view('parent_features')
    ids=con.execute('SELECT DISTINCT security_id FROM parent_features ORDER BY 1').df().security_id.tolist()
    daily=con.execute('SELECT * FROM daily_ready ORDER BY security_id,signal_date').df()
    groups={key:value.set_index('signal_date') for key,value in daily.groupby('security_id',sort=False)}
    dates=pd.DatetimeIndex(sessions.signal_date)
    pieces=[]
    for security in ids:
        group=groups.get(security)
        if group is None:
            continue
        aligned=group.reindex(dates)
        aligned['session_minutes']=sessions.set_index('signal_date').market_close_utc.sub(
            sessions.set_index('signal_date').market_open_utc).dt.total_seconds()/60
        values=make_volume_features(aligned,expected_sessions=dates).rename_axis('signal_date').reset_index()
        values.insert(0,'security_id',security)
        pieces.append(values)
    volume=pd.concat(pieces,ignore_index=True)
    con.register('volume_values',volume)
    con.read_parquet(str(private/'features/02_momentum_total_v1/daily_anchors.parquet')).create_view('price_anchors')
    con.execute('''CREATE TEMP TABLE joined AS SELECT f.*,v.* EXCLUDE(security_id,signal_date),
      prev.session_close gap_previous_close,o.gap_opening_price,o.opening_rows,
      o.opening_source,o.opening_bar_available_at_utc,
      upcoming.market_open_utc decision_event_utc,
      coalesce(o.opening_rows=1 AND o.opening_source=f.source
        AND o.opening_source IS NOT NULL AND o.opening_source!='',false) gap_source_matched,
      CASE WHEN o.gap_opening_price>0 AND prev.session_close>0
        THEN o.gap_opening_price/prev.session_close-1 END overnight_gap_unchecked
      FROM parent_features f LEFT JOIN volume_values v USING(security_id,signal_date)
      LEFT JOIN price_anchors prev USING(security_id,signal_date)
      LEFT JOIN opening o ON f.security_id=o.security_id AND f.next_session=o.trade_date
      LEFT JOIN sessions upcoming ON f.next_session=upcoming.signal_date''')
    con.execute('''CREATE TEMP TABLE final_features AS SELECT *,
      CASE WHEN gap_source_matched THEN overnight_gap_unchecked END overnight_gap,
      CASE WHEN opening_rows IS NULL THEN 'missing_opening_bar'
        WHEN opening_rows!=1 THEN 'duplicate_opening_bar'
        WHEN gap_opening_price IS NULL THEN 'invalid_opening_price'
        WHEN gap_previous_close IS NULL OR gap_previous_close<=0 THEN 'missing_prior_close'
        WHEN NOT gap_source_matched THEN 'incomparable_price_source'
        ELSE 'observed' END opening_gap_status
      FROM joined''')
    con.execute('COPY (SELECT * FROM final_features ORDER BY signal_date,security_id) TO ? (FORMAT PARQUET,COMPRESSION ZSTD)',
                [str(out/'features.parquet')])
    con.read_parquet(str(parent_run/'supervised_panel.parquet')).create_view('parent_panel')
    original_columns=con.execute('DESCRIBE parent_panel').df().column_name.tolist()
    keep=[x for x in original_columns if x!='signal_cutoff_utc']
    extras=VOLUME_FEATURES+['overnight_gap','overnight_gap_unchecked','decision_event_utc',
        'opening_bar_available_at_utc','gap_source_matched','opening_gap_status',
        'volume_reference_days','last_hour_reference_days']
    con.execute('CREATE TEMP TABLE panel AS SELECT '+','.join('p.'+x for x in keep)+
        ',p.signal_cutoff_utc history_cutoff_utc,f.decision_event_utc signal_cutoff_utc,'+
        ','.join('f.'+x for x in extras)+
        ' FROM parent_panel p LEFT JOIN final_features f USING(security_id,signal_date)')
    con.execute('COPY (SELECT * FROM panel ORDER BY trade_date,security_id) TO ? (FORMAT PARQUET,COMPRESSION ZSTD)',
                [str(out/'supervised_panel.parquet')])
    parent_count=con.execute('SELECT count(*) FROM parent_panel').fetchone()[0]
    panel_count=con.execute('SELECT count(*) FROM panel').fetchone()[0]
    if parent_count!=panel_count:
        raise ValueError('Forecast cohort changed during feature join.')
    stable=[x for x in original_columns if x!='signal_cutoff_utc']
    cols=','.join(stable)
    different=con.execute('WITH diff AS ((SELECT '+cols+' FROM panel EXCEPT ALL SELECT '+cols+
        ' FROM parent_panel) UNION ALL (SELECT '+cols+' FROM parent_panel EXCEPT ALL SELECT '+cols+
        ' FROM panel)) SELECT count(*) FROM diff').fetchone()[0]
    if different:
        raise ValueError('Original inputs/labels/outcomes changed.')
    new_missing_known_opens=con.execute('''SELECT count(*) FROM parent_panel p
        LEFT JOIN opening o ON p.security_id=o.security_id AND p.trade_date=o.trade_date
        WHERE p.outcome_status='ok' AND (o.gap_opening_price IS NULL
          OR abs(o.gap_opening_price-p.session_open)>1e-12)''').fetchone()[0]
    if new_missing_known_opens:
        raise ValueError('Opening-only extraction disagrees with known target openings.')
    records=[]
    for name in MODEL_FEATURES:
        row=con.execute(f'''SELECT count(*) AS n_rows,count_if({name} IS NULL OR isnan({name})) missing,
          count_if(isinf({name})) infinite,min({name}) minimum,max({name}) maximum FROM panel''').df().iloc[0].to_dict()
        row['feature']=name
        records.append(row)
        if row['infinite']:
            raise ValueError('Infinite feature: '+name)
        if name in MOMENTUM_FEATURES and row['missing']:
            raise ValueError('Original finite momentum input became missing.')
    availability=pd.DataFrame(records)
    availability.to_parquet(out/'feature_availability.parquet',index=False)
    gap_counts=con.execute('SELECT opening_gap_status,count(*) AS n_rows FROM panel GROUP BY 1 ORDER BY 2 DESC').df()
    gap_counts.to_parquet(out/'opening_gap_status_counts.parquet',index=False)
    con.execute('''COPY (SELECT * FROM daily_ready WHERE NOT activity_valid OR NOT signed_valid)
        TO ? (FORMAT PARQUET,COMPRESSION ZSTD)''',[str(out/'invalid_volume_days.parquet')])
    violations=con.execute('''SELECT count(*) FROM panel WHERE
        abs(vol_signed_balance_1d)>1+1e-12 OR abs(vol_signed_balance_5d)>1+1e-12
        OR vol_relative_activity_10d<0 OR vol_relative_last_hour_10d<0 OR vol_activity_trend_3v7<0
        OR vol_close_vwap_gap<=-1 OR overnight_gap<=-1
        OR signal_cutoff_utc IS NULL OR (history_cutoff_utc>=signal_cutoff_utc)''').fetchone()[0]
    if violations:
        raise ValueError('Feature domain or cutoff validation failed.')
    summary=dict(parent_panel_rows=parent_count,panel_rows=panel_count,
        full_membership_feature_rows=con.execute('SELECT count(*) FROM final_features').fetchone()[0],
        forecast_cohort_identical=True,original_eight_inputs_labels_outcomes_identical=True,
        original_opening_proxy_mismatches=new_missing_known_opens,
        volume_feature_count=6,total_feature_count=15,
        elapsed_seconds=time.monotonic()-started,
        peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**3)
    audit=dict(**summary,availability=availability.to_dict('records'),opening_gap_status=gap_counts.to_dict('records'))
    save_json(out/'AUDIT.json',audit)
    manifest=dict(version='momentum8_volume6_opening_gap1_v1',created_at_utc=datetime.now(timezone.utc).isoformat(),
        feature_sets=FEATURE_SETS,model_features=MODEL_FEATURES,input_hashes=hashes,summary=summary,
        source_file_inventory=inventory,
        information_timing=dict(momentum8='Previous completed regular-session close.',
            volume6='Previous completed regular-session close; historical references exclude the current session.',
            overnight_gap='Trade-day opening price observed with zero latency; same-price entry is an optimistic research assumption.',
            stored_opening_bar='Opening field only; no first-bar high/low/close/volume in gap. Published five-minute bar availability is retained separately, generally five minutes later.'),
        volume_definitions=dict(
            vol_relative_activity_10d='Current raw dollar-volume proxy per scheduled trading minute / median of available prior ten scheduled sessions.',
            vol_relative_last_hour_10d='Closing sixty trading minutes raw dollar volume / median of prior ten closing hours.',
            vol_activity_trend_3v7='Mean dollar-volume-per-minute in latest three sessions / mean in preceding seven sessions; full ten scheduled sessions required for this feature value.',
            vol_signed_balance_1d='SUM(sign(raw close - raw open) * raw VWAP * raw volume) / total dollar volume, one session.',
            vol_signed_balance_5d='Five-session signed-dollar-volume sum / five-session dollar-volume sum.',
            vol_close_vwap_gap='Last raw regular-session close / volume-weighted raw bar VWAP - 1.'),
        missing_policy='Original price eligibility and forecast rows preserved. Incomplete/invalid quantity days and undefined ratios remain NaN, never zero-filled. A future XGBoost fit must use native missing-value handling for added inputs. Original eight inputs remain finite.',
        overnight_policy='Adjusted opening / previous adjusted close - 1; incomparable opening-source tags or invalid/duplicate opens yield NaN. This uses only information about the opening bar opening field, not same-day closing availability.',
        clipping='Existing eight inputs copied unchanged. No new clipping or winsorization of volume or gap values or realized returns.',
        one_minute_work='None. No one-minute data were read, downloaded or checked.',
        usage_rights='internal_research_only',known_issues=parent_manifest['known_issues']+[
            'Dollar volume and session VWAP are derived from aggregated vendor bars, not certified trade-level cash flow.',
            'Signed activity uses bar direction and is not observed aggressor order flow.',
            'The fifteen-input same-opening-price strategy is an optimistic signal experiment, not executable trading evidence.'])
    manifest['output_sha256']={name:sha(out/name) for name in ['features.parquet','supervised_panel.parquet','daily_volume_anchors.parquet','opening_anchors.parquet']}
    save_json(out/'MANIFEST.json',manifest)
    con.close()
    print(json.dumps(summary),flush=True)
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--output',type=Path,help='A new private release directory; existing releases stay immutable.')
    args=p.parse_args()
    build(args.project,args.output)
