"""Sealed cache-only input plan and streaming complete-history evidence.

No new technical feature bank is read or computed. Metadata masks precede
financial conversion; mixed Parquet blocks may decode excluded Arrow values.
"""
from pathlib import Path
from copy import deepcopy
import hashlib,json,os
from .paper15_baseline import need

META=('trading_date','intrinio_security_id','frequency','intraperiod','price_quality_status','raw_ohlc_bounds_valid','adjusted_ohlc_bounds_valid')
OHLC=('open','high','low','close','adj_open','adj_high','adj_low','adj_close')
COHORT=('source_security_id','signal_date','intrinio_security_id','identity_eligible','identity_status','identity_review_required','resolution_status','indices','membership_tickers','unified_security_id','source_identity_status','identity_eligibility_policy')
CODE=('src/dmv2/paper15_baseline.py','src/dmv2/paper15_baseline_data.py','src/dmv2/paper15_baseline_run.py','scripts/run_paper15_baseline.py','scripts/verify_paper15_baseline.py')


def pin(path):
    p=Path(path).resolve();h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return dict(path=str(p),size_bytes=p.stat().st_size,sha256=h.hexdigest())

def clean(p):return {k:p[k] for k in ('path','size_bytes','sha256')}

def check(item):
    p=Path(item['path']);need(p.is_absolute() and pin(p)==clean(item),'Pinned input changed: '+str(p));return p

def read(p):
    p=Path(p);need(p.stat().st_size<=32*1024**2,'JSON ceiling');return json.loads(p.read_text())

def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
    return pin(p)

def table_save(p,v):
    import pyarrow as pa
    import pyarrow.parquet as pq
    p=Path(p);need(not p.exists(),'Preserve prior numerical artifact');p.parent.mkdir(parents=True,exist_ok=True)
    if not isinstance(v,pa.Table):v=pa.Table.from_pandas(v,preserve_index=False)
    pq.write_table(v,p,compression='zstd');return pin(p)

def canonical(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()

def projection(item,cols,lower,upper):
    import pyarrow.parquet as pq
    pf=pq.ParquetFile(check(item));need(set(cols)<=set(pf.schema_arrow.names),'Source projection schema')
    groups=[];cost=0;rows=0
    for g in range(pf.metadata.num_row_groups):
        rg=pf.metadata.row_group(g);stats=rg.column(pf.schema_arrow.names.index('trading_date')).statistics
        if stats and stats.has_min_max and (stats.max<lower or stats.min>upper):continue
        selected=[rg.column(c).total_compressed_size for c in range(rg.num_columns) if rg.column(c).path_in_schema in cols]
        groups.append(g);cost+=sum(selected);rows+=rg.num_rows
    return dict(pin=clean(item),columns=list(cols),row_groups=groups,start=lower,end=upper,projected_compressed_bytes=cost,candidate_rows=rows)


def seal(code_root,data_root,protocol,output,destination):
    import exchange_calendars as xc
    import pyarrow.parquet as pq
    from .catalog import SourceCatalog
    code=Path(code_root).resolve();data=Path(data_root).resolve();cfg=read(protocol)
    need(cfg['schema']=='dmv2_paper15_executable_ret_baseline_v2' and cfg['maximum_fits']==16 and cfg['windows_years']==[5,10]
        and cfg['RET']['months_by_arm']=={'5':60,'10':120} and cfg['imputation'] is False,'Frozen side-test scope changed')
    catalog_path=data/'outputs/bk_xom_shadow/2010-2023-20261021-v1/installed_catalog.json';catalog=read(catalog_path)
    proof=read(check(catalog['independent_verification']));need(proof['status']=='PASS','Corrected annual source proof required')
    corrected=[];cohorts=[];bound_pins=[pin(catalog_path),clean(catalog['independent_verification'])]
    for annual in catalog['annuals']:
        year=annual['year'];need(2010<=year<=2023,'Source annual year escaped study')
        ap=clean(annual['audit']);check(ap)
        need(any(all(ap[k]==p[k] for k in ('sha256','size_bytes')) for p in proof['annual_audit_pins']),'Unverified corrected annual audit')
        audit=read(ap['path']);need(audit['status']=='COMPLETE_SHADOW_NOT_PROMOTED','Annual source incomplete')
        for name in ('original_cohort','daily_prices_raw'):
            item=annual['artifacts'][name];a=audit['artifacts'][name]
            need(all(item[k]==a[k] for k in ('size_bytes','sha256')),'Annual corrected artifact not proof-bound')
        bound_pins.append(ap)
        cp=clean(annual['artifacts']['original_cohort']);pf=pq.ParquetFile(check(cp));need(set(COHORT)<=set(pf.schema_arrow.names),'Historical membership schema changed')
        cost=sum(pf.metadata.row_group(g).column(c).total_compressed_size for g in range(pf.metadata.num_row_groups) for c in range(pf.metadata.row_group(g).num_columns) if pf.metadata.row_group(g).column(c).path_in_schema in COHORT)
        cohorts.append(dict(year=year,pin=cp,columns=list(COHORT),rows=pf.metadata.num_rows,projected_compressed_bytes=cost))
        corrected.append(projection(annual['artifacts']['daily_prices_raw'],(*META,*OHLC),f'{year}-01-01',f'{year}-12-31'))
    raw=SourceCatalog(data);prefix=[]
    bound_pins.extend([pin(data/'configs/data_sources.local.json'),pin(raw.source('intrinio').base/'MANIFEST.json')])
    for obj in raw.table_objects('intrinio','intrinio_daily_prices'):
        obj.check();item={'path':str(obj.path),'size_bytes':obj.size_bytes,'sha256':obj.sha256.removeprefix('sha256:')}
        # Full-byte integrity checking opens no financial columns; no statistic
        # other than date bounds is consulted while selecting row groups.
        spec=projection(item,(*META,*OHLC),'2000-01-01','2023-12-31')
        if spec['row_groups']:prefix.append(spec)
    cal=xc.get_calendar('XNYS',start='2000-01-01',end='2023-12-31');sessions=list(cal.sessions.strftime('%Y-%m-%d'))
    execution=read(data/'outputs/ratio_cash_execution/plans/2020-2023-tif-lhcg-20261026-v1.json')
    handoff_path=data/'outputs/ratio_cash_execution/2020-2023-tif-lhcg-20261026-v1/verification_recovery/recovery_handoff.json'
    handoff=read(handoff_path);execution_proof=read(check(handoff['proof_pin']));need(execution_proof['status']=='PASS','Reviewed action/overlay source proof required')
    bound_pins.extend([pin(handoff_path),handoff['proof_pin'],pin(data/'outputs/ratio_cash_execution/plans/2020-2023-tif-lhcg-20261026-v1.json')])
    bt=deepcopy(execution['source_backtest']);bt['sessions']=[d for d in sessions if d>='2020-02-03'];bt['expected_sessions']=len(bt['sessions'])
    need(len(bt['sessions'])==985,'Common executable RET comparison requires985sessions')
    phase=read(check(bt['execution_pins']['execution_phase_pin']));need(len(phase['event_registry']['events'])==18,'Original18actions')
    phase['event_registry']['events']+=cfg['backtest']['additional_reviewed_actions'];need(len(phase['event_registry']['events'])==20,'Two approved additions')
    target_plan_path=data/'outputs/full_path_target_audit/2010-2023-20261022-v1/reviewed_plan.json'
    target_proof_path=target_plan_path.parent/'independent_full_verification.json'
    target_source=read(target_plan_path);target_proof=read(target_proof_path)
    need(target_proof['status']=='PASS_FULL_FORMULAS_RANKS_CLOCK_AND_ORIGINAL_DENOMINATORS' and target_proof['plan_sha256']==target_source['plan_sha256'],'Reviewed learning action source proof')
    events=target_source['events']+cfg['backtest']['additional_reviewed_actions']
    need(len(events)==31 and len({e['event_id'] for e in events})==31 and all(e['verified'] is True and e['cash_per_raw_share']>=0 and e['stock_ratio']>=0 for e in events),'31approved full cash/share terms')
    need(not any(a.get('successor_id')==b['predecessor_id'] and a['effective_date']==b['effective_date'] for a in events for b in events),'Unsupported same-session action chain')
    bound_pins.extend([pin(target_plan_path),pin(target_proof_path)])
    exit_calendar=list(xc.get_calendar('XNYS',start='2024-01-01',end='2024-01-03').sessions.strftime('%Y-%m-%d'))
    source_cost=sum(x['projected_compressed_bytes'] for x in prefix+corrected+cohorts)
    need(source_cost<=cfg['budgets']['source_projection_bytes_per_pass'],'Source projection budget; seal larger reviewed bound before reading')
    helpers={n:pin(code/'src/dmv2'/n) for n in ('pilot_models.py','ret.py','walkforward_metrics.py','walkforward_accounting_v2.py','walkforward_accounting.py','backtest_accounting.py','daily_backtest.py','walkforward_backtest.py','walkforward_preflight.py','target_comparison_backtest.py','ret_repair_backtest.py')}
    verifier={n:pin(code/'scripts'/n) for n in ('verify_walkforward_backtest_v2.py','verify_ret_repair_backtest.py')}
    plan=dict(schema=cfg['schema'],status='SEALED_BEFORE_FINANCIAL_READ',code_root=str(code),data_root=str(data),output_directory=str(Path(output).resolve()),
      protocol_pin=pin(protocol),source_binding_pins=bound_pins,cohorts=cohorts,raw_prefix=prefix,corrected_daily=corrected,sessions=sessions,
      source_projection_bytes=source_cost,target_events=events,target_exit_calendar_metadata=exit_calendar,execution_source=bt,execution_phase=phase,code_pins={n:pin(code/n) for n in CODE},helper_pins=helpers,verifier_helper_pins=verifier,
      max_fits=16,financial_2024_plus_read=False,expanded_features_read=False)
    plan['plan_sha256']=canonical(plan);save(destination,plan);return plan


def validate(plan_path):
    p=read(plan_path);need(p['plan_sha256']==canonical({k:v for k,v in p.items() if k!='plan_sha256'}),'Sealed plan changed')
    for item in [p['protocol_pin'],*p['source_binding_pins'],*p['code_pins'].values(),*p['helper_pins'].values(),*p['verifier_helper_pins'].values()]:check(item)
    return p


def read_quotes(spec,natives):
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    pf=pq.ParquetFile(check(spec['pin']));value_cols=[n for n in spec['columns'] if n not in ('trading_date','intrinio_security_id')]
    for g in spec['row_groups']:
        meta=pf.read_row_group(g,columns=['trading_date','intrinio_security_id'],use_threads=False)
        need(meta.null_count==0 if hasattr(meta,'null_count') else all(c.null_count==0 for c in meta.columns),'Null source keys')
        mask=pc.and_(pc.and_(pc.greater_equal(meta['trading_date'],spec['start']),pc.less_equal(meta['trading_date'],spec['end'])),pc.is_in(meta['intrinio_security_id'],value_set=pa.array(natives)))
        if not pc.any(mask).as_py():continue
        financial=pf.read_row_group(g,columns=value_cols,use_threads=False).filter(mask)
        selected=meta.filter(mask)
        for name in value_cols:selected=selected.append_column(name,financial[name])
        yield selected.to_pandas()


def ingest(plan,guard=lambda:None):
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    cohorts=[]
    for item in plan['cohorts']:
        guard();t=pq.read_table(check(item['pin']),columns=item['columns']);need(t.num_rows==item['rows'],'Original cohort rows changed')
        c=t.to_pandas();need(c.signal_date.min()>=f"{item['year']-1}-12-01" and c.signal_date.max()<f"{item['year']}-12-01",'Cohort clock changed');cohorts.append(c)
    cohorts=pd.concat(cohorts,ignore_index=True);need(len(cohorts)==152307 and not cohorts.duplicated(['source_security_id','signal_date']).any(),'All original152307stock-months required')
    natives=sorted(set(cohorts.loc[cohorts.identity_eligible,'intrinio_security_id'].dropna())|{e[k] for e in plan['target_events'] for k in ('predecessor_id','successor_id') if e.get(k)});days=plan['sessions'];months=sorted(set(d[:7] for d in days));month_ends={m:max(d for d in days if d.startswith(m)) for m in months}
    dayindex=pd.Index(days);nativeindex=pd.Index(natives);mi={d:i for i,d in enumerate(month_ends.values())}
    daily=np.full((len(days),len(natives),4),np.nan);quote_validity=np.zeros((len(days),len(natives),2),bool)
    valid=np.zeros((len(days),len(natives)),dtype=bool);present=np.zeros_like(valid);corrected_seen=np.zeros_like(valid);closes=np.full((len(months),len(natives)),np.nan)
    for no,spec in enumerate([*plan['raw_prefix'],*plan['corrected_daily']]):
        guard()
        for t in read_quotes(spec,natives):
            di=dayindex.get_indexer(t.trading_date);ni=nativeindex.get_indexer(t.intrinio_security_id);in_calendar=(di>=0)&(ni>=0)
            t=t.loc[in_calendar].reset_index(drop=True);di=di[in_calendar];ni=ni[in_calendar]
            flat=di*len(natives)+ni;need(len(np.unique(flat))==len(flat),'Duplicate source native/session within block')
            if no<len(plan['raw_prefix']):need(not present[di,ni].any(),'Duplicate raw-source native/session')
            else:
                need(not corrected_seen[di,ni].any(),'Duplicate corrected-source native/session');corrected_seen[di,ni]=True
            meta=t.frequency.eq('daily').to_numpy()&t.intraperiod.eq(False).to_numpy()&t.price_quality_status.isin(('accepted','ok')).to_numpy()
            values=t[list(OHLC)].to_numpy(dtype=float);good=meta & np.isfinite(values).all(axis=1)&(values>0).all(axis=1)
            good &= t.raw_ohlc_bounds_valid.eq(True).to_numpy()&t.adjusted_ohlc_bounds_valid.eq(True).to_numpy()
            for offset in (0,4):
                o,h,l,c=values[:,offset:offset+4].T;good &= (h>=np.maximum(o,c))&(l<=np.minimum(o,c))&(h>=l)
            present[di,ni]=True;valid[di,ni]=good
            daily[di,ni]=values[:,[0,3,4,7]]
            accepted=meta&t.raw_ohlc_bounds_valid.eq(True).to_numpy()&t.adjusted_ohlc_bounds_valid.eq(True).to_numpy()
            for mark,cols in [(0,[0,4]),(1,[3,7])]:quote_validity[di,ni,mark]=accepted&np.isfinite(values[:,cols]).all(axis=1)&(values[:,cols]>0).all(axis=1)
            at_end=t.trading_date.isin(mi).to_numpy();ci=np.array([mi.get(d,-1) for d in t.trading_date]);cv=meta&np.isfinite(values[:,-1])&(values[:,-1]>0)&at_end
            closes[ci[at_end],ni[at_end]]=np.nan
            closes[ci[cv],ni[cv]]=values[cv,-1]
        if (no+1)%10==0:print(json.dumps({'phase':'history_ingestion','source_files_completed':no+1,'maximum':len(plan['raw_prefix'])+len(plan['corrected_daily'])}),flush=True)
    return cohorts,valid,present,closes,natives,months,daily,quote_validity


def prepare(plan_path,guard=lambda:None):
    import numpy as np
    from .paper15_baseline import build_panels,ret_payoffs
    p=validate(plan_path);out=Path(p['output_directory'])/'prepared';need(not out.exists(),'Preserve preparation attempt');out.mkdir(parents=True)
    save(out/'STARTED.json',{'plan_pin':pin(plan_path),'status':'ATTEMPT_CONSUMED'})
    cohorts,valid,present,closes,natives,months,daily,quote_validity=ingest(p,guard);artifacts={}
    for n,a in [('valid',valid),('present',present),('month_closes',closes),('daily_prices',daily),('quote_validity',quote_validity)]:
        path=out/(n+'.npy')
        with path.open('xb') as f:np.save(f,a,allow_pickle=False)
        artifacts[n]=pin(path)
    artifacts['index']=save(out/'index.json',{'natives':natives,'months':months,'sessions':p['sessions']})
    artifacts['cohorts']=table_save(out/'cohorts.parquet',cohorts)
    for years in (5,10):
        guard();panel,coverage=build_panels(cohorts,valid,p['sessions'],closes,natives,months,years,prices=daily,quote_validity=quote_validity,events=p['target_events'],exit_calendar=p['target_exit_calendar_metadata'])
        artifacts[f'panel{years}']=table_save(out/f'panel{years}.parquet',panel)
        artifacts[f'coverage{years}']=table_save(out/f'coverage{years}.parquet',coverage)
        records={};cells=[]
        for signal in sorted(panel.loc[panel.outcome_month.ge('2020-01'),'signal_date'].unique()):
            mu,grid=ret_payoffs(panel,signal,years);records[signal]=mu;cells.extend(grid)
        artifacts[f'payoffs{years}']=save(out/f'payoffs{years}.json',records)
        import pandas as pd
        artifacts[f'payoff_grid{years}']=table_save(out/f'payoff_grid{years}.parquet',pd.DataFrame(cells))
    guard();return save(out/'receipt.json',{'status':'PREPARED_PENDING_INDEPENDENT_CHECK','plan_pin':pin(plan_path),'artifacts':artifacts,
        'original_stock_months':len(cohorts),'expanded_features_read':False,'source_projection_bytes':p['source_projection_bytes'],'new_fits':0})
