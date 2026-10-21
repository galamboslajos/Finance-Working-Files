"""Assemble independently source-pinned dated learning decisions, never fit models."""
from pathlib import Path
import argparse
import datetime
import json
from control import ROOT, check, pin, read, require_reservation, save_new, need
import learning_repairs as lr


def projection(path, columns, rows):
    import pyarrow.parquet as pq
    source=pq.ParquetFile(path)
    need(source.metadata.num_rows==rows,'Repair input footer rows differ')
    need(set(columns)<=set(source.schema_arrow.names),'Repair input columns missing')
    sizes=[]
    for i in range(source.metadata.num_row_groups):
        g=source.metadata.row_group(i)
        sizes.append(sum(g.column(j).total_compressed_size for j in range(g.num_columns)
                         if g.column(j).path_in_schema in columns))
    return dict(pin=pin(path),columns=columns,rows=rows,projected_bytes=sum(sizes),
                schema_fields={c:str(source.schema_arrow.field(c).type) for c in columns})


def prepare_plan(control_path, output):
    """Only called by root after outputs exist; footers/hash bytes, no values."""
    control=read(control_path)
    need(control['schema']=='dmv2-learning-repair-finalization-control/v1','Unknown finalization control')
    files={
        'rcm':(ROOT/'repairs/rcm_source_full_rows.parquet',list(lr.PRICE_COLUMNS),166),
        'disca':(ROOT/'repairs/disca_source_raw_all.parquet',[
            'unified_security_id','trading_date','adjustment','query_symbol','feed','timeframe',
            'selected_source_product_id','selected_source_security_id','open','high','low','close','volume'],14),
        'disca_controls':(check(control['disca_target_controls_pin']),list(lr.PRICE_COLUMNS),2)}
    projections={name:projection(*args) for name,args in files.items()}
    plan={'schema':'dmv2-learning-repair-finalization-plan/v1','status':'SEALED_BEFORE_DERIVED_PRICE_READ',
        'control_pin':pin(control_path),'projections':projections,
        'projected_bytes':sum(x['projected_bytes'] for x in projections.values()),
        'inventory_pin':pin(ROOT/'repairs/rcm_dated_key_inventory.json'),
        'dependency_pins':[pin(ROOT/'control.py'),pin(ROOT/'learning_repairs.py'),*control['evidence_pins']],
        'fixed_quality_tolerance':lr.TOLERANCE,'protected_2024_plus':True}
    return save_new(output,plan)


def validate_metadata(name, table, inventory):
    """Guard all exact date/route keys before financial Python conversion."""
    if name=='rcm':
        rows=table.select(['intrinio_security_id','trading_date']).to_pylist()
        expected={(n,d) for n,dates in inventory['routes'].items() for d in dates}
        need(len(rows)==166 and len(expected)==166 and {lr.key(r) for r in rows}==expected,'RCM finalization metadata differs')
    elif name=='disca_controls':
        rows=table.select(['intrinio_security_id','trading_date']).to_pylist()
        need(len(rows)==2 and {lr.key(r) for r in rows}=={(lr.DISCA_TARGET,'2022-04-01'),(lr.DISCA_TARGET,'2022-04-11')},'DISCA target control keys differ')
    else:
        cols=['unified_security_id','trading_date','adjustment','query_symbol','feed','timeframe',
              'selected_source_product_id','selected_source_security_id']
        rows=table.select(cols).to_pylist();seen=set()
        need(len(rows)==14,'DISCA source finalization rows differ')
        for r in rows:
            date=lr.day(r['trading_date']);need(date in lr.DISCA_ROUTES,'Unapproved DISCA source date')
            symbol,route=lr.DISCA_ROUTES[date]
            need(r['unified_security_id']==lr.DISCA_UNIFIED and r['query_symbol']==symbol
                and r['selected_source_security_id']==route and r['selected_source_product_id']=='russell_1000_2000_3000'
                and r['feed']=='sip' and r['timeframe']=='1Day' and r['adjustment'] in ('raw','all'),'DISCA source finalization route differs')
            k=date,r['adjustment'];need(k not in seen,'Duplicate DISCA source pair');seen.add(k)
        need(seen=={(d,a) for d in lr.DISCA_ROUTES for a in ('raw','all')},'DISCA source pair set incomplete')


def repair_schema():
    import pyarrow as pa
    bools={'intraperiod','raw_ohlc_bounds_valid','adjusted_ohlc_bounds_valid'}
    strings={'intrinio_security_id','trading_date','frequency','price_quality_status'}
    return pa.schema([(n,pa.bool_() if n in bools else pa.string() if n in strings else pa.float64()) for n in lr.PRICE_COLUMNS])


def run(plan_path, launch_path):
    pp=pin(plan_path);plan=read(plan_path);launch=read(launch_path)
    need(launch['schema']=='dmv2-learning-repair-finalization-launch/v1'
         and launch['status']=='SEALED_BEFORE_DERIVED_PRICE_READ' and launch['plan_pin']==pp
         and launch['worker_pin']==pin(__file__),'Unsealed repair finalization')
    require_reservation(ROOT,launch['reservation_label'],plan['projected_bytes'],pp)
    need(plan['fixed_quality_tolerance']==2e-4,'Frozen repair tolerance changed')
    for item in plan['dependency_pins']:check(item)
    control=read(check(plan['control_pin']));inventory=read(check(plan['inventory_pin']))
    import pyarrow as pa
    import pyarrow.parquet as pq
    tables={}
    for name,item in plan['projections'].items():
        path=check(item['pin']);need(projection(path,item['columns'],item['rows'])==item,'Repair input footer plan changed')
        table=pq.ParquetFile(path).read(columns=item['columns'],use_threads=False)
        validate_metadata(name,table,inventory);tables[name]=table
    # No financial rows entered Python before all three metadata guards succeeded.
    rcm,rr=lr.rcm_case(tables['rcm'].to_pylist(),inventory)
    disca,dr=lr.disca_case(tables['disca'].to_pylist(),tables['disca_controls'].to_pylist())
    rows=rr+dr;need(len(rows)<=165 and len(lr.unique(rows))==len(rows),'Admitted patch structural ceiling changed')
    target=ROOT/'repairs/learning_quote_repairs.parquet'
    with target.open('xb') as stream:pq.write_table(pa.Table.from_pylist(rows,schema=repair_schema()),stream,compression='zstd')
    for item in plan['dependency_pins']:check(item)
    for item in plan['projections'].values():check(item['pin'])
    check(plan['control_pin']);check(plan['inventory_pin']);check(pp);check(launch['worker_pin'])
    manifest={'schema':'dmv2-learning-OHLCV-repair-manifest/v1','status':'PASS',
        'verification_status':'VERIFIED_PROVISIONAL','cases':{'RCM':rcm,'DISCA':disca},
        'quotes_pin':pin(target),'row_count':len(rows),'columns':list(lr.PRICE_COLUMNS),
        'plan_pin':pp,'launch_pin':pin(launch_path),'worker_pin':pin(__file__),
        'source_pins':{n:x['pin'] for n,x in plan['projections'].items()},
        'evidence_pins':control['evidence_pins'],'inventory_pin':plan['inventory_pin'],
        'append_only':True,'replace_existing_rows':False,'canonical_identity_promoted':False,
        'terminal_target_policy_changed':False,'feature_or_target_values_rebuilt':False,
        'remaining_limitations':['Dated provisional route exceptions; no canonical mapping promotion.',
            'Original provider adjustments/knowledge-time limitations persist; paired controls do not certify full adjustment semantics.',
            'A source-quality rejected case admits zero rows and preserves original missing input bars.'],
        'models_fitted':0,'backtests_run':0}
    save_new(ROOT/'repairs/learning_quote_repair_manifest.json',manifest)
    print(json.dumps({'status':'PASS','admitted_rows':len(rows),'cases':{n:r['status'] for n,r in manifest['cases'].items()}}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['plan','run']);p.add_argument('--control')
    p.add_argument('--output');p.add_argument('--plan');p.add_argument('--launch');a=p.parse_args()
    if a.command=='plan':print(json.dumps(prepare_plan(a.control,a.output)))
    else:run(Path(a.plan),Path(a.launch))
