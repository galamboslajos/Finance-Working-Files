#!/usr/bin/env python3
"""Independent one-point completion and six unchanged close-clock controls."""
from pathlib import Path
import hashlib,importlib.util,json,math,sys


def need(v,m):
    if not v:raise ValueError(m)


def pin(p):
    p=Path(p).resolve();h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return dict(path=str(p),sha256=h.hexdigest(),size_bytes=p.stat().st_size)


def check(v):
    need(pin(v['path'])=={k:v[k] for k in ('path','sha256','size_bytes')},'Completion proof fingerprint')
    return Path(v['path'])


def read(p):return json.loads(Path(p).read_text())


def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
    return pin(p)


def verify_completion(plan_path,guard=lambda:None):
    import numpy as np,pandas as pd,pyarrow.parquet as pq
    p=read(plan_path);cfg=read(check(p['protocol_pin']))
    canonical=hashlib.sha256(json.dumps({k:v for k,v in p.items() if k!='plan_sha256'},sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
    need(p['plan_sha256']==canonical and p['maximum_fits']==0 and p['financial_2024_plus_read'] is False,'Frozen completion identity')
    for k in ('parent_ledger_pin','parent_plan_pin','parent_proof_pin','parent_audit_pin','parent_input_proof_pin','prepared_receipt_pin','parent_engine_pin','parent_producer_pin','independent_oracle_pin','parent_controller_pin'):check(p[k])
    for v in p['code_pins'].values():check(v)
    parent=read(check(p['parent_plan_pin']));oldproof=read(check(p['parent_proof_pin']));oldaudit=read(check(p['parent_audit_pin']))
    need(oldproof['status']=='PASS' and oldproof['complete_paths']==22 and oldproof['withheld_paths']==6,'Original coverage reference')
    root=Path(p['output_directory']);audit=read(root/'completion/audit.json');arts=audit['artifacts']
    need(audit['plan_pin']==pin(plan_path) and audit['new_fits']==0 and audit['parent_proof_pin']==p['parent_proof_pin']
        and audit['new_observed_points']==1 and audit['high_low_values_added'] is False,'Exact observed-OC-only completion')
    need(sum(v['size_bytes'] for v in arts.values())<=cfg['budgets']['derived_verification_bytes'],'Independent completion read ceiling')
    vp=save(root/'verification/read_plan.json',dict(status='SEALED_BEFORE_INDEPENDENT_VALUES',plan_pin=pin(plan_path),
       artifacts=arts,audit_pin=pin(root/'completion/audit.json'),new_fits=0))
    for v in arts.values():check(v)
    cp=pq.read_table(check(p['prepared_artifacts']['cache_prices'])).to_pandas()
    point=cp.loc[cp.trading_date.eq(cfg['date'])&cp.intrinio_security_id.eq(cfg['native'])].reset_index(drop=True)
    actual=pq.read_table(check(arts['observed_point'])).to_pandas()
    pd.testing.assert_frame_equal(point,actual,check_dtype=False,check_exact=True)
    need(len(point)==1 and bool(point.price_accepted.iloc[0]) and not {'adj_high','adj_low'}&set(point),'Existing accepted observed OC only')
    old=pq.read_table(check(p['prepared_artifacts']['execution_prices'])).to_pandas()
    need(not (old.trading_date.eq(cfg['date'])&old.intrinio_security_id.eq(cfg['native'])).any(),'Missing execution request boundary')
    old['price_accepted']=[r['frequency']=='daily' and isinstance(r['intraperiod'],(bool,np.bool_)) and not r['intraperiod'] and r['price_quality_status'] in ('ok','accepted') and isinstance(r['raw_ohlc_bounds_valid'],(bool,np.bool_)) and bool(r['raw_ohlc_bounds_valid']) and isinstance(r['adjusted_ohlc_bounds_valid'],(bool,np.bool_)) and bool(r['adjusted_ohlc_bounds_valid']) for r in old.to_dict('records')]
    old=old.drop(columns=['frequency','intraperiod','price_quality_status','raw_ohlc_bounds_valid','adjusted_ohlc_bounds_valid'])
    expected=pd.concat([old,point[[c for c in old.columns if c in point.columns]]],ignore_index=True)
    prices=pq.read_table(check(arts['completed_execution_prices'])).to_pandas()
    pd.testing.assert_frame_equal(prices,expected,check_dtype=False,check_exact=True)
    added=prices.loc[prices.trading_date.eq(cfg['date'])&prices.intrinio_security_id.eq(cfg['native'])]
    need(added.adj_high.isna().all() and added.adj_low.isna().all(),'High/low not manufactured')
    spec=importlib.util.spec_from_file_location('_fixed_completion_independent_oracle',check(p['independent_oracle_pin']))
    oracle=importlib.util.module_from_spec(spec);sys.modules[spec.name]=oracle;spec.loader.exec_module(oracle)
    comp=pq.read_table(check(arts['combined_comparison'])).to_pandas()
    original=pq.read_table(check(oldaudit['artifacts']['comparison'])).to_pandas()
    replacement=pq.read_table(check(arts['replacement_comparison'])).to_pandas();cases={tuple(x) for x in cfg['cases']}
    need(len(comp)==28 and len(replacement)==6 and set(zip(replacement.arm,replacement.stage))==cases,'Exact six-path replacement scope')
    reused=original.loc[[(r.arm,r.stage) not in cases for r in original.itertuples()]].sort_values(['arm','stage']).reset_index(drop=True)
    observed=comp.loc[[(r.arm,r.stage) not in cases for r in comp.itertuples()]].sort_values(['arm','stage']).reset_index(drop=True)
    pd.testing.assert_frame_equal(reused,observed,check_dtype=False,check_exact=True)
    verified=[];differences=[];phase=parent['execution_phase'];protocol=read(check(parent['protocol_pin']))
    for arm,stagekey in cfg['cases']:
        guard();st=next(s for s in protocol['stages'] if s['key']==stagekey)
        need(st['clock']=='signal_close' and st['cost']==0 and st['stop'] is None,'No main execution/stop path consumes added OC row')
        selected=pq.read_table(check(p['prepared_artifacts'][arm+'/selected'])).to_pandas()
        base=arm+'/'+stagekey+'/'
        tables={n:pq.read_table(check(arts[base+n])).to_pandas() for n in ('daily','trades','holdings','failures','stops','action_events')}
        result=oracle.oracle(selected,prices,parent['sessions'],phase['event_registry']['events'],
            phase['historical_retirements'],[],st,tables,guard)
        metrics,months=oracle.independent_metrics(result['closes'],result['complete'])
        row=comp.loc[comp.arm.eq(arm)&comp.stage.eq(stagekey)].iloc[0]
        need(bool(row.complete_path)==result['complete'],'Independent completion coverage')
        for name,value in metrics.items():oracle.number(row[name],value,'Independent completed metric '+name)
        need(row.stop_count==0 and row.unfilled_retired_orders==result['unfilled_retired_orders'],'No stops or altered retired orders')
        reference=pq.read_table(check(parent['reference_artifacts'][arm+'/matched47/cost_00bps/monthly'])).to_pandas().set_index('outcome_month')
        for date,wealth in months:
            diff=wealth-float(reference.loc[date[:7],'nav_end'])
            differences.append(dict(arm=arm,stage=stagekey,outcome_month=date[:7],absolute_nav_difference=diff,
              matched=math.isclose(wealth,float(reference.loc[date[:7],'nav_end']),rel_tol=1e-9,abs_tol=1e-10)))
        verified.append(dict(arm=arm,stage=stagekey,complete=result['complete'],verified_marks=result['verified_marks']))
        print(json.dumps(dict(verified=len(verified),maximum=6,new_fits=0)),flush=True)
    complete=int(comp.complete_path.sum())
    return save(root/'verification/completion.json',dict(status='PASS',plan_pin=pin(plan_path),
       audit_pin=pin(root/'completion/audit.json'),verification_read_plan_pin=vp,parent_proof_pin=p['parent_proof_pin'],
       verified_replays=6,original_rows_byte_equivalent=22,combined_scenarios=28,complete_paths=complete,withheld_paths=28-complete,
       scenarios=verified,reference_differences=differences,all_completed_controls_reconcile=all(r['matched'] for r in differences),
       observed_points_added=1,source_dates=[cfg['date']],source_natives=[cfg['native']],high_low_values_added=False,
       new_fits=0,new_predictions=0,learning_inputs_changed=False,selected_baskets_changed=False,financial_2024_plus_read=False))
