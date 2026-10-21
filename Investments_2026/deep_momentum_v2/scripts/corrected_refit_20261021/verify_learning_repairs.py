"""Independent zero-fit verification of dated OHLCV repair source composition.

Expected rows, admissibility and control comparisons do not call the production
learning adapter. Import is standard-library only; real source/output projections
require an exact sealed plan and cumulative reservation.
"""
from pathlib import Path
import argparse
import datetime
import json
import math
from control import ROOT,check,pin,read,require_reservation,save_new,need
COLS=('intrinio_security_id','trading_date','frequency','intraperiod','open','high','low','close','volume',
 'adj_open','adj_high','adj_low','adj_close','adj_volume','factor','split_ratio','dividend',
 'raw_ohlc_bounds_valid','adjusted_ohlc_bounds_valid','price_quality_status')
FIELDS=('open','high','low','close','volume','adj_open','adj_high','adj_low','adj_close','adj_volume')
META=('unified_security_id','trading_date','adjustment','query_symbol','feed','timeframe',
      'selected_source_product_id','selected_source_security_id')
RCM_OLD='sec_yKbEk4';RCM_TARGET='sec_zMM1vd';DISCA_TARGET='sec_Xnq2jn'
TOL=2e-4


def canonical_day(value):
    need(type(value) is str and datetime.date.fromisoformat(value).isoformat()==value and value<'2024-01-01', 'Invalid/protected repair date')
    return value


def index(rows):
    out={}
    for r in rows:
        k=(r['intrinio_security_id'],canonical_day(r['trading_date']))
        need(k not in out,'Duplicate repair verification source key');out[k]=r
    return out


def finite(value):
    return type(value) in (float,int) and math.isfinite(value)


def admissible(row,flags=True):
    if row.get('frequency')!='daily' or row.get('intraperiod') is not False:return False
    for prefix in ('','adj_'):
        o,h,l,c=(row.get(prefix+k) for k in ('open','high','low','close'))
        if not all(finite(x) and x>0 for x in (o,h,l,c)):return False
        if not (l<=o<=h and l<=c<=h):return False
        v=row.get(prefix+'volume')
        if not finite(v) or v<0:return False
    paired=row['adj_close']/row['close']
    for field in ('open','high','low','close'):
        if abs(row['adj_'+field]/row[field]/paired-1)>TOL:return False
    return (not flags or (row.get('price_quality_status')=='accepted' and
        row.get('raw_ohlc_bounds_valid') is True and row.get('adjusted_ohlc_bounds_valid') is True))


def controls(left,right,dates):
    rows=[];ok=True
    for date in dates:
        for field in FIELDS:
            a,b=left[date].get(field),right[date].get(field)
            residual=(0.0 if finite(a) and finite(b) and a==b==0 else abs(a-b)/abs(b) if finite(a) and finite(b) and b!=0 else None)
            accepted=residual is not None and residual<=TOL
            ok=ok and accepted
            rows.append({'trading_date':date,'column':field,'relative_residual':residual,'accepted':accepted})
    return ok,rows


def same_bar(left,right):
    if set(left)!=set(right):return False
    for field,value in left.items():
        other=right[field]
        if value is None or other is None:
            if value is not other:return False
        elif type(value) is float and type(other) is float and math.isnan(value) and math.isnan(other):
            continue
        elif type(value) is bool or type(other) is bool:
            if type(value) is not type(other) or value!=other:return False
        elif value!=other:return False
    return True


def verify_rows(rcm,disca,target_controls,quotes,manifest,inventory):
    old=index(rcm);need(len(rcm)==166 and all(set(r)==set(COLS) for r in rcm),'RCM verified source schema/count differs')
    expected={(n,canonical_day(d)) for n,dates in inventory['routes'].items() for d in dates}
    need(set(old)==expected and len(expected)==166,'RCM authenticated source key set differs')
    before=['2021-10-28','2021-10-29']
    dates=[d for d in inventory['routes'][RCM_OLD] if '2021-11-01'<=d<='2022-06-21']
    need(len(dates)==160 and inventory['routes'][RCM_OLD]==before+dates and
         inventory['routes'][RCM_TARGET]==before+['2022-06-22','2022-06-23'],'RCM frozen dated inventory differs')
    rc_ok,rc_controls=controls({d:old[(RCM_OLD,d)] for d in before},{d:old[(RCM_TARGET,d)] for d in before},before)
    rc_ok=rc_ok and all(admissible(r) for r in rcm)
    expected_rcm=[{**old[(RCM_OLD,d)],'intrinio_security_id':RCM_TARGET} for d in dates] if rc_ok else []
    pairs={}
    for row in disca:
        need(set(row)==set((*META,'open','high','low','close','volume')),'DISCA literal schema differs')
        d=canonical_day(row['trading_date']);need(d in ['2022-04-01','2022-04-04','2022-04-05','2022-04-06','2022-04-07','2022-04-08','2022-04-11'],'DISCA unapproved literal date')
        symbol='WBD' if d=='2022-04-11' else 'DISCA'
        route='siblis_isin_context_bd09bf2365326584bc2c' if symbol=='WBD' else 'siblis_isin_context_f53b6add4120f42b1c53'
        need(row['unified_security_id']=='equity_us_4f745ae6337bd9ef8d4bb45b'
            and row['query_symbol']==symbol and row['selected_source_security_id']==route
            and row['selected_source_product_id']=='russell_1000_2000_3000' and row['feed']=='sip'
            and row['timeframe']=='1Day' and row['adjustment'] in ('raw','all'),'DISCA dated share-class literal differs')
        k=d,row['adjustment'];need(k not in pairs,'Duplicate DISCA source pair');pairs[k]=row
    all_dates=['2022-04-01','2022-04-04','2022-04-05','2022-04-06','2022-04-07','2022-04-08','2022-04-11']
    need(set(pairs)=={(d,a) for d in all_dates for a in ('raw','all')},'DISCA literal pair count differs')
    normalized={}
    for d in all_dates:
        raw=pairs[(d,'raw')];adj=pairs[(d,'all')]
        normalized[d]={'intrinio_security_id':DISCA_TARGET,'trading_date':d,'frequency':'daily','intraperiod':False,
            **{f:raw[f] for f in ('open','high','low','close','volume')},
            **{'adj_'+f:adj[f] for f in ('open','high','low','close','volume')},
            'factor':None,'split_ratio':None,'dividend':None,
            'raw_ohlc_bounds_valid':True,'adjusted_ohlc_bounds_valid':True,'price_quality_status':'accepted'}
    ti=index(target_controls);need(set(ti)=={(DISCA_TARGET,d) for d in ('2022-04-01','2022-04-11')}
        and all(set(r)==set(COLS) for r in target_controls),'DISCA original target controls differ')
    dc_ok,dc_controls=controls(normalized,{d:ti[(DISCA_TARGET,d)] for d in ('2022-04-01','2022-04-11')},['2022-04-01','2022-04-11'])
    dc_ok=dc_ok and all(admissible(r,False) for r in normalized.values()) and all(admissible(r) for r in target_controls)
    expected_disca=[normalized[d] for d in all_dates[1:-1]] if dc_ok else []
    expected_quotes=expected_rcm+expected_disca
    actual_quotes=index(quotes);expected_map=index(expected_quotes)
    need(len(quotes)==len(expected_quotes) and set(actual_quotes)==set(expected_map)
         and all(same_bar(actual_quotes[k],expected_map[k]) for k in actual_quotes),'Admitted repair rows differ from independent exact source composition')
    need(all(set(r)==set(COLS) for r in quotes),'Admitted repair schema differs')
    need(manifest['status']=='PASS' and manifest['verification_status']=='VERIFIED_PROVISIONAL'
         and manifest['row_count']==len(quotes) and manifest['columns']==list(COLS)
         and manifest['append_only'] is True and manifest['replace_existing_rows'] is False
         and manifest['canonical_identity_promoted'] is False and manifest['terminal_target_policy_changed'] is False,
         'Learning manifest semantics differ')
    cases={}
    for name,accepted,expected_controls,expected_count,source_count in [('RCM',rc_ok,rc_controls,len(expected_rcm),166),('DISCA',dc_ok,dc_controls,len(expected_disca),14)]:
        decision=manifest['cases'][name]
        status='ACCEPTED_PROVISIONAL_DATED_APPEND_ONLY' if accepted else 'REJECTED_SOURCE_QUALITY'
        need(decision['status']==status and decision['control_relative_tolerance']==TOL
             and decision['source_rows']==source_count and decision['admitted_repair_rows']==expected_count,
             'Case admission differs from independent frozen quality rule: '+name)
        actual=decision['control_comparisons'];need(len(actual)==20,'Case control denominator changed')
        for a,e in zip(actual,expected_controls):
            need(a['trading_date']==e['trading_date'] and a['column']==e['column'] and a['accepted'] is e['accepted'],'Case control metadata changed')
            av,ev=a['relative_residual'],e['relative_residual']
            need((av is ev is None) or (finite(av) and finite(ev) and math.isclose(av,ev,rel_tol=1e-14,abs_tol=1e-16)), 'Case control residual changed')
        cases[name]={'status':status,'admitted_rows':expected_count,'independently_checked_control_fields':20}
    return {'status':'PASS','cases':cases,'admitted_rows':len(quotes),'source_rows':182,
        'source_controls_checked':4,'model_fits':0,'identity_promoted':False,
        'scope':'Exact dated source composition, quality/admission arithmetic and append-only patch keys; not canonical identity or adjustment-vintage certification.'}


def validate_metadata(rows,inventory):
    """Reject wrong original routes/windows before any financial conversion."""
    expected={(n,canonical_day(d)) for n,dates in inventory['routes'].items() for d in dates}
    need(len(rows['rcm'])==166 and set(index(rows['rcm']))==expected,'Independent RCM metadata differs')
    need(len(rows['controls'])==2 and set(index(rows['controls']))=={(DISCA_TARGET,'2022-04-01'),(DISCA_TARGET,'2022-04-11')},'Independent target control metadata differs')
    literal=rows['disca'];need(len(literal)==14,'Independent DISCA metadata count differs')
    dates=['2022-04-01','2022-04-04','2022-04-05','2022-04-06','2022-04-07','2022-04-08','2022-04-11'];seen=set()
    for row in literal:
        d=canonical_day(row['trading_date']);need(d in dates,'Unapproved literal metadata date')
        symbol='WBD' if d=='2022-04-11' else 'DISCA'
        route='siblis_isin_context_bd09bf2365326584bc2c' if symbol=='WBD' else 'siblis_isin_context_f53b6add4120f42b1c53'
        need(row['unified_security_id']=='equity_us_4f745ae6337bd9ef8d4bb45b' and row['query_symbol']==symbol
             and row['selected_source_security_id']==route and row['selected_source_product_id']=='russell_1000_2000_3000'
             and row['feed']=='sip' and row['timeframe']=='1Day' and row['adjustment'] in ('raw','all'),'Unapproved literal metadata route')
        k=d,row['adjustment'];need(k not in seen,'Duplicate literal metadata key');seen.add(k)
    need(seen=={(d,a) for d in dates for a in ('raw','all')},'Independent literal pair metadata differs')
    allowed={(RCM_TARGET,d) for d in inventory['routes'][RCM_OLD] if '2021-11-01'<=d<='2022-06-21'}|{(DISCA_TARGET,d) for d in dates[1:-1]}
    need(set(index(rows['quotes']))<=allowed,'Independent admitted quote metadata escaped dated repair scope')


def projection(path,columns,maximum):
    import pyarrow.parquet as pq
    pf=pq.ParquetFile(path);need(pf.metadata.num_rows<=maximum,'Independent source row ceiling exceeded')
    need(set(columns)<=set(pf.schema_arrow.names),'Independent projected field missing')
    cost=0
    for i in range(pf.metadata.num_row_groups):
        g=pf.metadata.row_group(i)
        cost+=sum(g.column(j).total_compressed_size for j in range(g.num_columns) if g.column(j).path_in_schema in columns)
    return {'pin':pin(path),'columns':list(columns),'rows':pf.metadata.num_rows,'projected_bytes':cost,
            'schema_fields':{n:str(pf.schema_arrow.field(n).type) for n in columns}}


def make_plan(output):
    files={'rcm':('rcm_source_full_rows.parquet',COLS,166),
           'disca':('disca_source_raw_all.parquet',(*META,'open','high','low','close','volume'),14),
           'controls':('disca_target_controls.parquet',COLS,2),
           'quotes':('learning_quote_repairs.parquet',COLS,165)}
    p={n:projection(ROOT/'repairs'/path,cols,maxrows) for n,(path,cols,maxrows) in files.items()}
    manifest=pin(ROOT/'repairs/learning_quote_repair_manifest.json');inventory=pin(ROOT/'repairs/rcm_dated_key_inventory.json')
    return save_new(output,{'schema':'dmv2-independent-learning-repair-plan/v1','status':'SEALED_BEFORE_FINANCIAL_READ',
        'projections':p,'manifest_pin':manifest,'inventory_pin':inventory,'worker_pin':pin(__file__),
        'control_pin':pin(ROOT/'control.py'),
        'projected_bytes':sum(v['projected_bytes'] for v in p.values())+manifest['size_bytes']+inventory['size_bytes']})


def run(plan_path,launch_path,output):
    pp=pin(plan_path);plan=read(plan_path);launch=read(launch_path)
    need(launch['schema']=='dmv2-independent-learning-repair-launch/v1' and launch['status']=='SEALED_BEFORE_FINANCIAL_READ'
         and launch['plan_pin']==pp and launch['worker_pin']==pin(__file__) and plan['worker_pin']==pin(__file__),'Independent repair launch unsealed')
    require_reservation(ROOT,launch['reservation_label'],plan['projected_bytes'],pp)
    check(plan['control_pin']);inventory=read(check(plan['inventory_pin']))
    import pyarrow.parquet as pq
    tables={};all_metadata={}
    for name,item in plan['projections'].items():
        path=check(item['pin']);need(projection(path,item['columns'],item['rows'])==item,'Independent repair footer projection changed')
        table=pq.ParquetFile(path).read(columns=item['columns'],use_threads=False)
        # Metadata guard BEFORE any source/repair numeric values enter Python.
        cols=list(META) if name=='disca' else ['intrinio_security_id','trading_date']
        meta=table.select(cols).to_pylist()
        for row in meta:canonical_day(row['trading_date'])
        if name!='disca':index(meta)
        all_metadata[name]=meta
        tables[name]=table
    validate_metadata(all_metadata,inventory)
    manifest=read(check(plan['manifest_pin']))
    result=verify_rows(tables['rcm'].to_pylist(),tables['disca'].to_pylist(),tables['controls'].to_pylist(),tables['quotes'].to_pylist(),manifest,inventory)
    need(manifest['inventory_pin']==plan['inventory_pin'],'Manifest inventory byte binding changed')
    need(manifest['quotes_pin']==plan['projections']['quotes']['pin'],'Manifest quote byte binding changed')
    need(manifest['source_pins']=={n:plan['projections'][k]['pin'] for n,k in [('rcm','rcm'),('disca','disca'),('disca_controls','controls')]},'Manifest source byte binding changed')
    for item in manifest['evidence_pins']:check(item)
    for item in plan['projections'].values():check(item['pin'])
    check(plan['manifest_pin']);check(plan['inventory_pin']);check(plan['control_pin']);check(pp);check(launch['worker_pin'])
    result.update({'schema':'dmv2-independent-learning-repair-verification/v1','plan_pin':pp,
        'launch_pin':pin(launch_path),'manifest_pin':plan['manifest_pin'],'worker_pin':pin(__file__),
        'reserved_read_bytes':plan['projected_bytes']})
    save_new(output,result);print(json.dumps({'status':result['status'],'cases':result['cases'],'admitted_rows':result['admitted_rows']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['plan','run']);p.add_argument('--output',required=True);p.add_argument('--plan');p.add_argument('--launch');a=p.parse_args()
    if a.command=='plan':print(json.dumps(make_plan(a.output)))
    else:run(Path(a.plan),Path(a.launch),Path(a.output))
