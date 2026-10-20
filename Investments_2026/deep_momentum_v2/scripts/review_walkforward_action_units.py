"""Bounded post-fit corporate-action unit review, with no portfolio calculation.

This standalone tool never promotes a draft event, repairs a source, changes
selection or fits a model. Its requested raw/adjusted bars are fixed by dated
legal terms, not by a performance result. Parquet filters return exact requested
keys in small batches; source compressed column-chunk budgets are checked first.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
import re

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

SCHEMA='dmv2-postfit-action-unit-plan/v1'
MAX_KEYS=64
MAX_RETURNED_ROWS=128
MAX_COMPRESSED_BYTES=64*1024*1024
MAX_PAIRED_RESIDUAL=0.0002
YEARS=(2020,2021,2022,2023)
ARMS=('rf_paper15','xgb_paper15','rf_extended','xgb_extended')
KEYS=('trading_date','intrinio_security_id')
QUALITY=('frequency','intraperiod','price_quality_status','raw_ohlc_bounds_valid','adjusted_ohlc_bounds_valid')
OHLC=('open','high','low','close')
PAIRED=(*OHLC,*('adj_'+name for name in OHLC))
DIAGNOSTIC=('volume','adj_volume','dividend','split_ratio','factor')


class ReviewError(ValueError):
    pass


def need(condition,message):
    if not condition: raise ReviewError(message)


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def canonical(value):
    return 'sha256:'+hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),
        ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def read_json(path):
    need(Path(path).stat().st_size<=4*1024*1024,'JSON receipt exceeds4MiB')
    return json.loads(Path(path).read_text())


def verify(base,pin):
    need(isinstance(pin,dict) and {'path','sha256','size_bytes'}<=set(pin),'Exact path/hash/size pin required')
    path=(Path(base)/pin['path']).resolve()
    need(path.is_relative_to(Path(base).resolve()),'Source pin escapes its root')
    expected=pin['sha256'].removeprefix('sha256:')
    need(path.is_file() and path.stat().st_size==pin['size_bytes'] and sha(path)==expected,'Pinned source changed: '+str(path))
    return path


def pin(path):
    path=Path(path).resolve()
    return {'path':str(path),'sha256':sha(path),'size_bytes':path.stat().st_size}


def day(value):
    stamp=pd.Timestamp(value)
    need(stamp.tz is None and not pd.isna(stamp) and stamp==stamp.normalize(),'Complete timezone-naive date required')
    return stamp.strftime('%Y-%m-%d')


def clock():
    dates=xcals.get_calendar('XNYS').sessions_in_range('2020-01-01','2023-12-31')
    result=[stamp.strftime('%Y-%m-%d') for stamp in dates]
    need(len(result)==1006 and result[0]=='2020-01-02' and result[-1]=='2023-12-29','Authorized1006-session calendar changed')
    return result


def guard_completed(root,run,plan):
    """Only bounded pinned JSON is read before the post-fit authorization gate."""
    identity=read_json(run/'build_identity.json')
    expected=plan['build_identity_sha256']
    need(canonical(identity)==expected,'Root build identity changed')
    config=read_json(verify(root,plan['configuration']))
    need(config==identity['configuration'],'Frozen configuration changed')
    annual_path=verify(run,plan['run_audit']); annual=read_json(annual_path)
    need(annual_path==run/'audit.json' and annual.get('status')=='COMPLETE' and annual.get('build_identity_sha256')==expected
        and set(annual.get('years',{}))==set(map(str,YEARS)),'Four-year feature phase incomplete')
    model_path=verify(run,plan['model_audit']); model=read_json(model_path)
    phase_path=run/'models/build_identity.json'; phase=read_json(phase_path)
    need(model_path==run/'models/audit.json' and model.get('status')=='COMPLETE' and model.get('phase')=='walkforward_models_v1'
        and model.get('build_identity_sha256')==expected
        and model.get('model_identity_sha256')==canonical(phase)
        and phase.get('parent_identity_sha256')==expected
        and model.get('summary',{}).get('fitted_arms')==16
        and model['summary'].get('preprocessor_fits')==8
        and model['summary'].get('months')==48
        and model['summary'].get('outer_years')==list(YEARS),'All16fits must be finalized before any numerical quote read')
    need(set(model.get('folds',{}))==set(map(str,YEARS)),'Four completed fold pins required')
    receipt_pins=[pin(annual_path),pin(model_path),pin(phase_path),pin(run/'build_identity.json'),
        pin(verify(root,plan['configuration']))]
    cutoffs={2020:'2019-12-31',2021:'2020-12-31',2022:'2021-12-31',2023:'2022-12-30'}
    for year in YEARS:
        fold_path=verify(run/'models',model['folds'][str(year)]); fold=read_json(fold_path)
        need(fold_path==run/'models/folds'/str(year)/'audit.json'
            and fold.get('status')=='COMPLETE' and fold.get('test_year')==year
            and fold.get('build_identity_sha256')==expected
            and fold.get('model_identity_sha256')==model['model_identity_sha256']
            and fold.get('summary',{}).get('fitted_arms')==4,'Completed fold scope changed')
        receipt_pins.append(pin(fold_path))
        for arm in ARMS:
            arm_path=verify(fold_path.parent,fold['arms'][arm]); arm_audit=read_json(arm_path)
            need(arm_path==fold_path.parent/'arms'/arm/'audit.json'
                and arm_audit.get('status')=='COMPLETE'
                and arm_audit.get('build_identity_sha256')==expected
                and arm_audit.get('model_identity_sha256')==model['model_identity_sha256'],'Completed arm receipt changed')
            fit_path=verify(arm_path.parent,arm_audit['artifacts']['fit_complete']); fit=read_json(fit_path)
            need(fit.get('status')=='FIT_COMPLETE' and fit.get('fit_count')==1
                and fit.get('build_identity_sha256')==expected
                and fit.get('model_identity_sha256')==model['model_identity_sha256']
                and fit.get('metadata',{}).get('test_labels_supplied_to_estimator') is False,'Fit receipt is incomplete or used outer labels')
            for field,value in {'test_year':year,'train_start':f'{year-10}-01-01',
                    'train_end':f'{year-1}-12-31','label_observed_through':cutoffs[year]}.items():
                need(fit.get('expected_metadata',{}).get(field)==value
                    and arm_audit.get('metadata',{}).get(field)==value,'Fit chronology changed')
            receipt_pins.extend([pin(arm_path),pin(fit_path)])
    preflight_path=verify(run,plan['preflight_audit']); preflight=read_json(preflight_path)
    need(preflight_path.parent.parent==run/'preflight' and preflight_path.name=='audit.json'
        and preflight.get('status')=='COMPLETE' and preflight.get('phase')=='walkforward_selected_horizon_preflight_v1'
        and preflight.get('build_identity_sha256')==expected,'Initial complete selected-horizon preflight is required')
    preflight_identity=verify(preflight_path.parent,preflight['artifacts']['build_identity.json'])
    preflight_phase=read_json(preflight_identity)
    need(canonical(preflight_phase)==preflight['phase_identity_sha256']
        and preflight_phase.get('parent_identity_sha256')==expected,'Preflight identity changed')
    matching=[source for source in preflight_phase.get('source_pins',[])
        if (root/source['path']).resolve()==model_path]
    need(len(matching)==1 and verify(root,matching[0])==model_path,'Preflight is not bound to this exact finalized model audit')
    receipt_pins.extend([pin(preflight_path),pin(preflight_identity)])
    return annual,receipt_pins


def event_requests(events,sessions,diagnose_cy=False):
    need(isinstance(events,list) and events and len(events)<=32,'One to32 dated proposed events required')
    requests={}; normalized=[]; seen=set()
    for event in events:
        need('action_type' not in event or event['action_type'] in {'cash_merger','stock_merger','mixed_merger'},
            'Explicit unsupported action_type requires a separately reviewed economic ledger')
        event_id=event.get('event_id'); date=day(event.get('effective_date'))
        need(isinstance(event_id,str) and event_id and event_id not in seen,'Unique event identities required')
        seen.add(event_id)
        need(date in sessions and sessions.index(date)>0,'Event needs a prior authorized session in2020–2023')
        predecessor=event.get('predecessor_id'); successor=event.get('successor_id')
        need(isinstance(predecessor,str) and predecessor.startswith('sec_'),'Predecessor native identity required')
        need(event.get('verified') is True or event.get('primary_event_terms_verified') is True
            or event.get('legal_terms_verified') is True,'Primary legal event terms must already be reviewed')
        need(isinstance(event.get('evidence_reference'),str) and event['evidence_reference'].startswith('https://'),'Primary evidence URL required')
        for name in ('cash_per_raw_share','stock_ratio'):
            number=event.get(name)
            need(not isinstance(number,bool) and isinstance(number,Real) and math.isfinite(number) and number>=0,'Finite nonnegative legal terms required')
        need(event['cash_per_raw_share']>0 or event['stock_ratio']>0,'No invented zero retirement')
        previous=sessions[sessions.index(date)-1]
        requests.setdefault((previous,predecessor),set()).add('predecessor_basis')
        if event['stock_ratio']>0:
            need(isinstance(successor,str) and successor and successor!=predecessor,'Stock consideration requires a distinct successor')
            if not successor.startswith('sec_'):
                namespace=event.get('successor_identity_namespace',event.get('identity_namespace'))
                identifier=event.get('successor_permanent_identifier',event.get('permanent_identifier'))
                patterns={'execution_cusip':r'[A-Z0-9]{9}','execution_isin':r'[A-Z]{2}[A-Z0-9]{10}'}
                need(namespace in patterns and isinstance(identifier,str)
                    and re.fullmatch(patterns[namespace],identifier)
                    and successor==namespace+':'+identifier
                    and event.get('identity_namespace',namespace)==namespace
                    and event.get('permanent_identifier',identifier)==identifier
                    and event.get('successor_provider_identity_asserted') is False,
                    'External execution successor must explicitly declare a matching CUSIP/ISIN namespace, never an invented provider native')
            requests.setdefault((date,successor),set()).add('successor_basis')
        else:need(successor is None,'Cash-only action cannot carry successor stock')
        normalized.append({**event,'prior_session':previous})
    if diagnose_cy:
        need(any(e['predecessor_id']=='sec_rg0Y2X' and e['effective_date']=='2020-04-16'
            and e['cash_per_raw_share']==23.85 and e['stock_ratio']==0 for e in events),'CY diagnostic requires exact reviewed cash terms')
        for date in ('2020-04-15','2020-04-16'):
            requests.setdefault((date,'sec_rg0Y2X'),set()).add('cy_halted_vendor_key_diagnostic')
    need(len(requests)<=MAX_KEYS,'Exact requested paired-bar key cap exceeded')
    return normalized,requests


def _filter(schema,keys):
    expression=None
    for date,native in sorted(keys):
        dtype=schema.field('trading_date').type
        value=date if pa.types.is_string(dtype) or pa.types.is_large_string(dtype) else pd.Timestamp(date).to_pydatetime()
        term=(ds.field('trading_date')==pa.scalar(value,type=dtype))&(ds.field('intrinio_security_id')==native)
        expression=term if expression is None else expression|term
    return expression


def file_plan(path,year,keys,diagnostic,*,bounds=None):
    pf=pq.ParquetFile(path); names=pf.schema_arrow.names
    need(set((*KEYS,*QUALITY,*PAIRED))<=set(names),'Source lacks complete paired OHLC or strict quality fields')
    columns=list((*KEYS,*QUALITY,*PAIRED))
    if diagnostic:columns.extend(name for name in DIAGNOSTIC if name in names)
    date_index=names.index('trading_date'); native_index=names.index('intrinio_security_id')
    first,last=bounds or (f'{year}-01-01',f'{year}-12-31')
    need('2020-01-01'<=first<=last<='2023-12-31','Source planning leaves authorized dates')
    groups=[]; compressed=0
    for index in range(pf.metadata.num_row_groups):
        group=pf.metadata.row_group(index)
        dstat=group.column(date_index).statistics; nstat=group.column(native_index).statistics
        if dstat is not None and dstat.has_min_max:
            low,high=day(dstat.min),day(dstat.max)
            need(first<=low<=high<=last,'Annual footer contains an unauthorized date')
        else:low,high=first,last
        nlow,nhigh=(nstat.min,nstat.max) if nstat is not None and nstat.has_min_max else ('','\uffff')
        if not any(low<=date<=high and nlow<=native<=nhigh for date,native in keys):continue
        groups.append(index)
        for column in columns:
            size=group.column(names.index(column)).total_compressed_size
            need(isinstance(size,int) and size>=0,'Compressed column-chunk size unavailable')
            compressed+=size
    need(groups,'No source row group can contain the exact requested keys')
    return {'path':Path(path),'year':year,'keys':keys,'groups':groups,'columns':columns,
        'schema':pf.schema_arrow,'projected_compressed_bytes':compressed,'diagnostic':diagnostic}


def supplemental_plans(root,sources,annual_sources,events,requests,sessions):
    """Only newly evidenced successors may use explicitly scoped extra files."""
    need(isinstance(sources,list) and len(sources)<=16,'At most16 explicit supplemental quote sources allowed')
    original_ids=set().union(*(set(item['audit']['feature_provenance']['requested_ids']) for item in annual_sources.values()))
    successors={event['successor_id'] for event in events if event['successor_id']}
    result=[];pins=[];declared_ids=set();seen_keys=set()
    for source in sources:
        required={'pin','allowed_native_ids','start','end','evidence_reference','identity_evidence_reference'}
        need(isinstance(source,dict) and required<=set(source) and not set(source)-required-{'receipt_pin'},'Exact supplemental source declarations required')
        ids=source['allowed_native_ids'];first,last=day(source['start']),day(source['end'])
        need(isinstance(ids,list) and 0<len(ids)<=8 and len(set(ids))==len(ids)
            and set(ids)<=successors and not set(ids)&original_ids
            and not set(ids)&declared_ids,'Supplements cannot extend original feature-requested natives or repeat successor routes')
        need('2020-01-01'<=first<=last<='2023-12-31'
            and all(isinstance(source[name],str) and source[name].startswith('https://')
                for name in ('evidence_reference','identity_evidence_reference')),'Supplemental evidence/dates invalid')
        path=verify(root,source['pin']);pins.append(pin(path))
        need(path.is_relative_to(root/'data/derived') and path.suffix=='.parquet'
            and path.stat().st_size<=16*1024*1024,'Supplemental quotes must be small pinned private derived Parquet')
        pf=pq.ParquetFile(path)
        need(0<pf.metadata.num_rows<=10_000,'Supplemental metadata row budget exceeded')
        # This is key-only streaming. No numerical quote columns are decoded.
        actual=set()
        for batch in pf.iter_batches(batch_size=64,columns=list(KEYS),use_threads=False):
            for row in batch.to_pylist():
                key=(day(row['trading_date']),row['intrinio_security_id'])
                need(key[1] in ids and first<=key[0]<=last and key[0] in sessions
                    and key not in actual and key not in seen_keys,'Supplemental keys overlap or escape approved metadata scope')
                actual.add(key)
                need(len(actual)+len(seen_keys)<=10_000,'Supplemental aggregate metadata exceeds10000rows')
        selected={key for key in requests if key[1] in ids}
        need(selected and selected<=actual,'Supplemental source does not supply its exact reviewed action-basis keys')
        result.append(file_plan(path,None,selected,False,bounds=(first,last)))
        seen_keys.update(actual);declared_ids.update(ids)
        if 'receipt_pin' in source:pins.append(pin(verify(root,source['receipt_pin'])))
    # Budget the exact key-only absence scans before any financial read.
    absence_bytes=0
    if declared_ids:
        for item in annual_sources.values():
            pf=pq.ParquetFile(item['path']);names=pf.schema_arrow.names
            absence_bytes+=sum(pf.metadata.row_group(group).column(names.index(name)).total_compressed_size
                for group in range(pf.metadata.num_row_groups) for name in KEYS)
    return result,pins,declared_ids,absence_bytes


def scan(plan,columns):
    fragment=ds.ParquetFileFormat().make_fragment(str(plan['path']),filesystem=pa.fs.LocalFileSystem(),row_groups=plan['groups'])
    scanner=ds.Scanner.from_fragment(fragment,schema=plan['schema'],columns=columns,
        filter=_filter(plan['schema'],plan['keys']),batch_size=64,use_threads=False)
    rows=[]
    for batch in scanner.to_batches():
        rows.extend(batch.to_pylist())
        need(len(rows)<=MAX_RETURNED_ROWS,'Exact filtered return exceeds128rows')
    return rows


def guard_rows(rows,keys):
    need(len(rows)<=MAX_RETURNED_ROWS,'Returned-row cap exceeded')
    observed=[]
    for row in rows:
        date=day(row['trading_date']); native=row['intrinio_security_id']
        need('2020-01-01'<=date<='2023-12-31' and (date,native) in keys,'Filtered output escaped exact authorized keys')
        observed.append((date,native))
    need(len(set(observed))==len(observed) and set(observed)==set(keys),'Requested paired basis is absent or duplicated')


def quality(row):
    need(row.get('frequency')=='daily' and row.get('intraperiod') is False
        and row.get('price_quality_status')=='accepted'
        and row.get('raw_ohlc_bounds_valid') is True
        and row.get('adjusted_ohlc_bounds_valid') is True,'Paired bar requires accepted complete source-quality metadata')


def number(value):
    need(not isinstance(value,(bool,np.bool_)) and isinstance(value,Real) and math.isfinite(value) and value>0,
        'Paired OHLC must be positive finite observations')
    return float(value)


def review_bar(row,anchor):
    quality(row)
    raw={name:number(row[name]) for name in OHLC}
    adjusted={name:number(row['adj_'+name]) for name in OHLC}
    for values in (raw,adjusted):
        need(values['low']<=min(values['open'],values['close'])<=max(values['open'],values['close'])<=values['high'],
            'Observed numerical OHLC bounds fail')
    multiplier=number(adjusted[anchor]/raw[anchor])
    residuals={name:abs(adjusted[name]-raw[name]*multiplier) for name in OHLC}
    ratios={name:number(adjusted[name]/raw[name]) for name in OHLC}
    need(all(math.isfinite(value) for value in residuals.values()),'Paired-unit residual is nonfinite')
    return {'factor_anchor':anchor,'adjusted_per_raw_factor':multiplier,'raw':raw,'adjusted':adjusted,
        'adjusted_raw_ratios':ratios,'absolute_adjusted_price_residuals':residuals,
        'maximum_absolute_residual':max(residuals.values()),
        'threshold_absolute_adjusted_price':MAX_PAIRED_RESIDUAL,
        'pass':max(residuals.values())<=MAX_PAIRED_RESIDUAL}


def review_event(event,bars):
    pre=review_bar(bars[(event['prior_session'],event['predecessor_id'])],'close')
    post=review_bar(bars[(event['effective_date'],event['successor_id'])],'open') if event['stock_ratio'] else None
    examples=[]
    for raw_shares in (1.0,-1.0):
        old_units=raw_shares/pre['adjusted_per_raw_factor']
        recovered_raw=old_units*pre['adjusted_per_raw_factor']
        need(math.isfinite(old_units) and abs(recovered_raw-raw_shares)<=1e-12,'Adjusted/raw predecessor entitlement identity fails')
        cash=recovered_raw*event['cash_per_raw_share']
        successor_raw=recovered_raw*event['stock_ratio']
        successor_adjusted=successor_raw/post['adjusted_per_raw_factor'] if post else 0.0
        raw_stock_mark=successor_raw*post['raw']['open'] if post else 0.0
        adjusted_stock_mark=successor_adjusted*post['adjusted']['open'] if post else 0.0
        raw_total=cash+raw_stock_mark; adjusted_total=cash+adjusted_stock_mark
        residual=abs(raw_total-adjusted_total)
        need(residual<=1e-10*max(1.,abs(raw_total)),'Signed raw/adjusted entitlement algebra fails')
        examples.append({'signed_raw_share':raw_shares,'old_adjusted_units':old_units,
            'recovered_signed_raw_entitlement':recovered_raw,'signed_cash_receivable_or_liability':cash,
            'successor_signed_raw_shares':successor_raw,'successor_signed_adjusted_units':successor_adjusted,
            'raw_stock_plus_cash_mark':raw_total,'adjusted_stock_plus_cash_mark':adjusted_total,
            'mark_identity_absolute_residual':residual})
    return {'event_id':event['event_id'],'status':'PASS_UNIT_ALGEBRA' if pre['pass'] and (post is None or post['pass'])
        else 'REFUSED_PAIRED_UNIT_RESIDUAL','predecessor_basis':pre,'successor_basis':post,
        'signed_examples':examples,'source_draft_verified':event.get('verified',False),
        'successor_identity_namespace':event.get('successor_identity_namespace',event.get('identity_namespace','intrinio_security_id' if event['successor_id'] else None)),
        'successor_permanent_identifier':event.get('successor_permanent_identifier',event.get('permanent_identifier')),
        'successor_provider_identity_asserted':event.get('successor_provider_identity_asserted',None),
        'draft_event_promoted':False,'portfolio_pnl_computed':False,
        'separate_dividends_not_added':[{'field':key,'documented_value':value} for key,value in event.items()
            if 'dividend' in key.lower()],
        'physical_settlement_or_fractional_cash_simulated':False}


def execute(plan_path,plan_sha256,output,*,authorize_postfit=False):
    need(authorize_postfit is True,'Explicit post-fit financial-read authorization is required')
    plan_path=Path(plan_path).resolve(); need(sha(plan_path)==plan_sha256.removeprefix('sha256:'),'Input plan changed')
    plan=read_json(plan_path); need(plan.get('schema')==SCHEMA,'Unknown bounded action-unit plan schema')
    root=Path(plan['project_root']).resolve(); run=(root/plan['run_directory']).resolve()
    need(run.is_relative_to(root),'Run escapes project root')
    output=Path(output).resolve(); need(not output.exists(),'Preserve prior unit-review output')
    annual,pins=guard_completed(root,run,plan)
    drafts=[]; draft_pins=[]
    for source in plan['events']:
        path=Path(source['path']).resolve()
        need(path.is_file() and path.stat().st_size==source['size_bytes']
            and sha(path)==source['sha256'].removeprefix('sha256:'),'Proposed event source changed')
        drafts.append(read_json(path));draft_pins.append(pin(path))
    events,requests=event_requests(drafts,clock(),plan.get('diagnose_cy_april16') is True)
    files=[];annual_sources={}
    for year in YEARS:
        directory=run/'years'/str(year)
        audit_path=verify(run,annual['years'][str(year)]); local=read_json(audit_path)
        need(audit_path==directory/'audit.json' and local.get('status')=='COMPLETE'
            and local.get('year')==year and local.get('build_identity_sha256')==plan['build_identity_sha256'],
            'Requested annual source audit changed')
        path=verify(directory,local['artifacts']['daily_prices_raw']);pins.extend([pin(audit_path),pin(path)])
        need(isinstance(local.get('feature_provenance',{}).get('requested_ids'),list),'Annual original requested-ID registry is required')
        annual_sources[year]={'audit':local,'path':path}
    extra,extra_pins,extra_ids,absence_bytes=supplemental_plans(root,plan.get('supplemental_quote_sources',[]),
        annual_sources,events,requests,clock())
    pins.extend(extra_pins)
    for year,item in annual_sources.items():
        keys={key for key in requests if key[0].startswith(str(year)) and key[1] not in extra_ids}
        if keys:files.append(file_plan(item['path'],year,keys,
            any('cy_halted_vendor_key_diagnostic' in requests[key] for key in keys)))
    files.extend(extra)
    compressed=sum(item['projected_compressed_bytes'] for item in files)+absence_bytes
    need(compressed<=MAX_COMPRESSED_BYTES,'64MiB projected compressed column-chunk budget exceeded')
    if extra_ids:
        for item in annual_sources.values():
            scanner=ds.dataset(str(item['path']),format='parquet').scanner(columns=list(KEYS),
                filter=ds.field('intrinio_security_id').isin(sorted(extra_ids)),batch_size=64,use_threads=False)
            for batch in scanner.to_batches():
                need(batch.num_rows==0,'Supplemental native already occurs in an original annual cache; no overwrite/extension allowed')
    # Every key, date and strict quality flag passes before any numeric projection.
    for item in files:
        metadata=scan(item,list((*KEYS,*QUALITY)))
        guard_rows(metadata,item['keys'])
        for row in metadata:quality(row)
    bars={};returned=0
    for item in files:
        rows=scan(item,list((*KEYS,*QUALITY,*PAIRED)));guard_rows(rows,item['keys'])
        returned+=len(rows);need(returned<=MAX_RETURNED_ROWS,'Total financial rows exceed128')
        for row in rows:bars[(day(row['trading_date']),row['intrinio_security_id'])]=row
        diagnostic_keys={key for key in item['keys'] if 'cy_halted_vendor_key_diagnostic' in requests[key]}
        if diagnostic_keys:
            diagnostic_plan={**item,'keys':diagnostic_keys}
            diagnostic_rows=scan(diagnostic_plan,item['columns']);guard_rows(diagnostic_rows,diagnostic_keys)
            returned+=len(diagnostic_rows);need(returned<=MAX_RETURNED_ROWS,'Total financial rows exceed128')
            for row in diagnostic_rows:bars[(day(row['trading_date']),row['intrinio_security_id'])].update(row)
    reviews=[review_event(event,bars) for event in events]
    diagnostics=[]
    if plan.get('diagnose_cy_april16') is True:
        for date in ('2020-04-15','2020-04-16'):
            row=bars[(date,'sec_rg0Y2X')]
            diagnostics.append({'date':date,'native':'sec_rg0Y2X','source_observation':row,
                'raw_ohlc_all_equal_legal_cash23_85':all(number(row[field])==23.85 for field in OHLC),
                'actual_trading_bar_certified':False,'bar_is_execution_basis':date=='2020-04-15',
                'limitation':'Nasdaq says CY was halted throughout April16; vendor key alone proves no trading. No source repair or feature/refit decision follows from this diagnostic.'})
    for source in [pin(plan_path),*pins,*draft_pins]:
        path=Path(source['path']);need(path.stat().st_size==source['size_bytes'] and sha(path)==source['sha256'],'A bound source changed during review')
    output.mkdir(parents=True)
    evidence=output/'exact_paired_bar_evidence.parquet'
    pd.DataFrame(list(bars.values())).to_parquet(evidence,index=False)
    receipt={'schema':'dmv2-postfit-action-unit-review/v1','status':'PASS' if all(r['status']=='PASS_UNIT_ALGEBRA' for r in reviews) else 'REFUSED',
        'reviewer_script_sha256':sha(Path(__file__)),
        'created_at_utc':datetime.now(timezone.utc).isoformat(),'build_identity_sha256':plan['build_identity_sha256'],
        'plan':pin(plan_path),'completion_and_source_pins':pins,'original_event_draft_pins':draft_pins,
        'paired_bar_evidence':pin(evidence),'event_reviews':reviews,'cy_diagnostic':diagnostics,
        'supplemental_new_successor_natives':sorted(extra_ids),
        'budgets':{'requested_distinct_keys':len(requests),'maximum_distinct_keys':MAX_KEYS,
            'projected_compressed_bytes':compressed,'maximum_projected_compressed_bytes':MAX_COMPRESSED_BYTES,
            'returned_financial_rows':returned,'maximum_returned_rows':MAX_RETURNED_ROWS},
        'paired_absolute_adjusted_price_residual_gate':MAX_PAIRED_RESIDUAL,
        'metadata_dates_keys_and_quality_guarded_before_numeric_projection':True,
        'full_financial_panel_materialized':False,'numeric_projection':'exact key-filtered Arrow scanner, batch_size64; Parquet may internally process source pages within the compressed-byte cap',
        'portfolio_performance_computed':False,'targets_or_model_objects_decoded':False,'model_fits':0,
        'event_drafts_changed_or_promoted':False,'source_or_feature_or_prediction_or_selection_changed':False,
        'execution_ready_certified':False,'limitations':['Legal and local provisional identities retain their original evidence limitations.',
            'PASS verifies quote-unit algebra only, not actual security settlement, source-wide corporate-action completeness, identity certification or portfolio performance.']}
    (output/'receipt.json').write_text(json.dumps(receipt,sort_keys=True,indent=2,allow_nan=False,default=str)+'\n')
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True);parser.add_argument('--plan-sha256',required=True)
    parser.add_argument('--output-directory',required=True);parser.add_argument('--authorize-postfit',action='store_true')
    args=parser.parse_args()
    receipt=execute(args.plan,args.plan_sha256,args.output_directory,authorize_postfit=args.authorize_postfit)
    print(json.dumps({'status':receipt['status'],'events':len(receipt['event_reviews']),'budgets':receipt['budgets']}))


if __name__=='__main__':main()
