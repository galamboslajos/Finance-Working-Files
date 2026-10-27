"""One observed PBF boundary-point completion for six failed close controls."""
from pathlib import Path
import json,itertools
from .paper15_baseline_data import pin,check,read,save,table_save,canonical

CASES=tuple((a,s) for a in ('xgb_paper15_5y','rf_paper15_10y','xgb_paper15_10y')
 for s in ('execution_close_gross','execution_close_open_units'))
CODE=('src/dmv2/fixed_forecast_completion.py','scripts/run_fixed_forecast_completion.py',
      'scripts/verify_fixed_forecast_completion.py')


def need(v,m):
    if not v:raise ValueError(m)


def completed_quote_view(execution,point,date,native):
    import numpy as np,pandas as pd
    need('2020-01-01'<=date<'2024-01-01','Protected boundary date')
    need(len(point)==1 and point.trading_date.iloc[0]==date and point.intrinio_security_id.iloc[0]==native,'Exact observed boundary point')
    need(not (execution.trading_date.eq(date)&execution.intrinio_security_id.eq(native)).any(),'Never overwrite an observed execution key')
    need(isinstance(point.price_accepted.iloc[0],(bool,np.bool_)) and bool(point.price_accepted.iloc[0])
       and np.isfinite(point[['open','close','adj_open','adj_close']].to_numpy()).all()
       and (point[['open','close','adj_open','adj_close']].to_numpy()>0).all(),'Positive observed accepted OC')
    need(not {'adj_high','adj_low','high','low'}&set(point),'No invented high/low source fields')
    result=execution.copy()
    result['price_accepted']=result.frequency.eq('daily')&result.intraperiod.map(lambda x:isinstance(x,(bool,np.bool_)) and not bool(x))&result.price_quality_status.isin(('ok','accepted'))&result.raw_ohlc_bounds_valid.map(lambda x:isinstance(x,(bool,np.bool_)) and bool(x))&result.adjusted_ohlc_bounds_valid.map(lambda x:isinstance(x,(bool,np.bool_)) and bool(x))
    result=result.drop(columns=['frequency','intraperiod','price_quality_status','raw_ohlc_bounds_valid','adjusted_ohlc_bounds_valid'])
    result=pd.concat([result,point[list(result.columns.intersection(point.columns))]],ignore_index=True)
    need(not result.duplicated(['trading_date','intrinio_security_id']).any(),'One observed boundary addition only')
    return result


def seal(code_root,data_root,protocol,output,destination):
    import pyarrow.parquet as pq
    code,data=Path(code_root).resolve(),Path(data_root).resolve();cfg=read(protocol)
    need(cfg['maximum_fits']==0 and cfg['date']=='2022-12-30' and cfg['native']=='sec_z9QxBz' and cfg['cases']==[list(x) for x in CASES],'Exact PBF completion scope')
    parent=data/'outputs/fixed_forecast_reconciliation/20261027-v1';ledger=read(parent/'ledger.json')
    need(ledger['status']=='COMPLETE_VERIFIED' and ledger['fits_completed']==0,'Completed original reconciliation')
    phase=next(read(check(v['receipt']))['result'] for v in ledger['completed'] if v['key']=='verify_replay')
    proof=read(check(phase));need(proof['status']=='PASS' and proof['complete_paths']==22 and proof['withheld_paths']==6,'Exact original22complete/6withheld grid')
    p=read(check(ledger['plan_pin']));ip=read(check(proof['prepared_input_proof_pin']));prep=read(check(ip['prepared_receipt_pin']))
    audit=read(check(proof['audit_pin']));target=(cfg['date'],cfg['native'])
    for key in ('cache_prices','execution_prices'):
        f=pq.read_table(check(prep['artifacts'][key]),columns=['trading_date','intrinio_security_id']).to_pandas()
        count=int((f.trading_date.eq(target[0])&f.intrinio_security_id.eq(target[1])).sum())
        need(count==(1 if key=='cache_prices' else 0),'Metadata-only point presence/absence changed')
    for arm,stage in CASES:
        f=pq.read_table(check(audit['artifacts'][arm+'/'+stage+'/failures'])).to_pandas()
        need(len(f)==1 and str(f.iloc[0].date)[:10]==target[0] and f.iloc[0].intrinio_security_id==target[1]
             and f.iloc[0].reason=='missing_quote' and f.iloc[0].role=='selected','Only one absent boundary source key')
    result=dict(schema='dmv2_fixed_forecast_completion_v1',status='SEALED_BEFORE_OBSERVED_POINT_VALUE_READ',
       code_root=str(code),data_root=str(data),output_directory=str(Path(output).resolve()),protocol_pin=pin(protocol),
       parent_directory=str(parent),parent_ledger_pin=pin(parent/'ledger.json'),parent_plan_pin=ledger['plan_pin'],
       parent_proof_pin=phase,parent_audit_pin=proof['audit_pin'],parent_input_proof_pin=proof['prepared_input_proof_pin'],
       prepared_receipt_pin=ip['prepared_receipt_pin'],prepared_artifacts=prep['artifacts'],
       parent_engine_pin=p['code_pins']['src/dmv2/fixed_forecast_accounting.py'],
       parent_producer_pin=p['code_pins']['src/dmv2/fixed_forecast_reconciliation.py'],
       independent_oracle_pin=p['code_pins']['scripts/verify_fixed_forecast_reconciliation.py'],
       parent_controller_pin=p['code_pins']['scripts/run_fixed_forecast_reconciliation.py'],
       cached_file_reservation_bytes=sum(prep['artifacts'][k]['size_bytes'] for k in ('cache_prices','execution_prices')),
       code_pins={n:pin(code/n) for n in CODE},maximum_fits=0,financial_2024_plus_read=False)
    result['plan_sha256']=canonical(result);save(destination,result);return result


def validate(path):
    p=read(path);need(p['plan_sha256']==canonical({k:v for k,v in p.items() if k!='plan_sha256'}),'Completion plan changed')
    for k in ('protocol_pin','parent_ledger_pin','parent_plan_pin','parent_proof_pin','parent_audit_pin','parent_input_proof_pin','prepared_receipt_pin','parent_engine_pin','parent_producer_pin','independent_oracle_pin','parent_controller_pin'):check(p[k])
    for v in [*p['code_pins'].values(),*p['prepared_artifacts'].values()]:check(v)
    need(p['maximum_fits']==0 and p['financial_2024_plus_read'] is False,'Zero-fit protected completion')
    return p


def complete(plan_path,guard=lambda:None):
    import numpy as np,pandas as pd,pyarrow.parquet as pq
    from .fixed_forecast_accounting import run_fixed_forecast_accounting
    from .fixed_forecast_reconciliation import stats
    from .backtest_accounting import DailyAccountingPolicy
    p=validate(plan_path);cfg=read(check(p['protocol_pin']));parent=read(check(p['parent_plan_pin']))
    root=Path(p['output_directory']);out=root/'completion';need(not out.exists(),'Preserve completion attempt');out.mkdir(parents=True)
    save(out/'read_plan.json',dict(status='SEALED_BEFORE_NUMERIC_READ',plan_pin=pin(plan_path),date=cfg['date'],native=cfg['native'],
        source=p['prepared_artifacts']['cache_prices'],execution=p['prepared_artifacts']['execution_prices'],new_fits=0))
    cache=pq.read_table(check(p['prepared_artifacts']['cache_prices'])).to_pandas()
    point=cache.loc[cache.trading_date.eq(cfg['date'])&cache.intrinio_security_id.eq(cfg['native'])].copy()
    need(len(point)==1 and bool(point.price_accepted.iloc[0]) and np.isfinite(point[['open','close','adj_open','adj_close']].to_numpy()).all(),'Observed accepted OC boundary point')
    execution=pq.read_table(check(p['prepared_artifacts']['execution_prices'])).to_pandas()
    execution=completed_quote_view(execution,point,cfg['date'],cfg['native'])
    artifacts={'observed_point':table_save(out/'observed_point.parquet',point),
        'completed_execution_prices':table_save(out/'completed_execution_prices.parquet',execution)}
    prior=read(check(p['parent_audit_pin']))
    comp=pq.read_table(check(prior['artifacts']['comparison'])).to_pandas();replacement=[]
    for arm,key in CASES:
        guard();stage=next(s for s in read(check(parent['protocol_pin']))['stages'] if s['key']==key)
        sel=pq.read_table(check(p['prepared_artifacts'][arm+'/selected'])).to_pandas();sel['execution_date']=sel.signal_date
        rb=sel[['signal_date','execution_date']].drop_duplicates().sort_values('execution_date')
        phase=parent['execution_phase']
        value=run_fixed_forecast_accounting(sel,execution,parent['sessions'],rebalances=rb,terminal_date='2023-12-29',
           clock='signal_close',successor_factor_mark=stage['unit'],events=phase['event_registry']['events'],
           historical_retirements=phase['historical_retirements'],policy=DailyAccountingPolicy())
        good=value.status=='complete_proxy';measure,closes,monthly=stats(value.daily,'signal_close',good)
        row=comp.loc[comp.arm.eq(arm)&comp.stage.eq(key)].iloc[0].to_dict()
        row.update(complete_path=good,first_failure_date=value.summary['first_failure_date'],stop_count=value.summary['stop_count'],
           unfilled_retired_orders=value.summary['unfilled_retired_order_count'],**measure)
        replacement.append(row);base=arm+'/'+key+'/'
        for name,frame in [*value.tables.items(),('close_wealth',closes),('monthly',monthly)]:artifacts[base+name]=table_save(out/(base+name+'.parquet'),frame)
        artifacts[base+'summary']=save(out/(base+'summary.json'),value.summary)
        print(json.dumps(dict(completed=len(replacement),maximum=6,new_fits=0)),flush=True)
    rep=pd.DataFrame(replacement);keys=set(CASES)
    keep=comp.loc[[ (r.arm,r.stage) not in keys for r in comp.itertuples()]]
    combined=pd.concat([keep,rep],ignore_index=True)
    artifacts['replacement_comparison']=table_save(out/'replacement_comparison.parquet',rep)
    artifacts['combined_comparison']=table_save(out/'combined_comparison.parquet',combined)
    return save(out/'audit.json',dict(status='PENDING_INDEPENDENT_CONTROL_CHECK',plan_pin=pin(plan_path),parent_proof_pin=p['parent_proof_pin'],
       artifacts=artifacts,replayed_paths=6,original_rows_reused=22,new_fits=0,forecasts_changed=False,selection_changed=False,
       new_observed_points=1,high_low_values_added=False,financial_2024_plus_read=False))
