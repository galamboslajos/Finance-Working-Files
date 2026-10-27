"""Zero-fit selection recovery after a side-column representation mismatch.

Original fit attempt, twelve models, independent model proof and failed ledger
remain immutable. Only the parity-check side representation is normalized.
"""
from pathlib import Path
import itertools
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from .paper15_baseline import FEATURES, PROBS, KEYS, need
from .paper15_baseline_data import pin, check, read, save, table_save, canonical
from .extension_models import FORECAST_META, validate as validate_parent

CODE=('src/dmv2/extension_selection_recovery.py','scripts/run_extension_selection_recovery.py','scripts/verify_extension_selection_recovery.py')


def assert_old_selection_equal(left,right):
    left=left.copy();right=right.copy()
    left['side']=left['side'].astype('string');right['side']=right['side'].astype('string')
    pd.testing.assert_frame_equal(left,right,check_exact=True)


def seal(code_root,parent_plan,protocol,output,destination):
    code=Path(code_root).resolve();cfg=read(protocol);parent=validate_parent(parent_plan);oldroot=Path(parent['output_directory'])
    need(cfg['schema']=='dmv2_extension_selection_recovery_v1' and cfg['maximum_fits']==cfg['maximum_backtests']==0,'Zero-fit representation-only recovery')
    ledger=read(oldroot/'ledger.json')
    need(ledger['status']=='FAILED_PRESERVED' and ledger['phase']=='selection' and ledger['fits_completed']==ledger['fits_started']==12,'Preserved selection-only failure required')
    model_pin=pin(oldroot/'verification/models.json');model=read(check(model_pin))
    need(model['status']=='PASS' and model['verified_fits']==12 and model['plan_pin']==pin(parent_plan),'All12models already independently passed')
    prep_pin=pin(oldroot/'prepared/receipt.json');data=read(check(model['data_proof_pin']))
    need(data['prepared_receipt_pin']==prep_pin,'Inherited source receipt')
    log_pin=pin(oldroot/'logs/selection.log');log=Path(log_pin['path']).read_text()
    need('column name="side"' in log and 'Attribute "dtype" are different' in log,'Exact representation mismatch required')
    need(not (oldroot/'selection/audit.json').exists(),'No completed prior selection may be overwritten')
    need(not Path(output).exists(),'New recovery output required')
    p=dict(schema=cfg['schema'],status='SEALED_BEFORE_ZERO_FIT_SELECTION_RECOVERY',code_root=str(code),output_directory=str(Path(output).resolve()),
        parent_plan_pin=pin(parent_plan),parent_ledger_pin=pin(oldroot/'ledger.json'),parent_failure_log_pin=log_pin,
        protocol_pin=pin(protocol),model_proof_pin=model_pin,data_proof_pin=model['data_proof_pin'],prepared_receipt_pin=prep_pin,
        max_fits=0,max_backtests=0,inherited_verified_fits=12,inherited_active_seconds=ledger['active_seconds'],
        projected_bytes_per_pass=parent['source_projection_bytes'],code_pins={n:pin(code/n) for n in CODE})
    p['plan_sha256']=canonical(p);save(destination,p);return p


def validate(plan_path):
    p=read(plan_path);need(p['plan_sha256']==canonical({k:v for k,v in p.items() if k!='plan_sha256'}),'Recovery plan changed')
    for item in [p['parent_plan_pin'],p['parent_ledger_pin'],p['parent_failure_log_pin'],p['protocol_pin'],p['model_proof_pin'],p['data_proof_pin'],p['prepared_receipt_pin'],*p['code_pins'].values()]:check(item)
    validate_parent(check(p['parent_plan_pin']))
    need(p['max_fits']==p['max_backtests']==0 and p['inherited_verified_fits']==12,'No new fitting or trading')
    return p


def publish(plan_path):
    p=validate(plan_path);root=Path(p['output_directory']);ledger=read(root/'ledger.json')
    need(ledger['status']=='COMPLETE_VERIFIED_SELECTION_RECOVERY' and ledger['fits_started']==ledger['fits_completed']==0,'Verified zero-fit recovery required')
    phase=next(x['receipt'] for x in ledger['completed'] if x['key']=='verify_selection');proof_pin=read(check(phase))['result'];proof=read(check(proof_pin))
    need(proof['status']=='PASS' and proof['plan_pin']==pin(plan_path) and proof['model_proof_pin']==p['model_proof_pin'],'Independent recovered selection proof')
    parent=read(check(p['parent_plan_pin']));oldroot=Path(parent['output_directory'])
    handoff=dict(status='PASS_INHERITED_12_MODELS_AND_RECOVERED_SELECTIONS',parent_plan_pin=p['parent_plan_pin'],parent_ledger_pin=p['parent_ledger_pin'],
        inherited_model_proof_pin=p['model_proof_pin'],recovery_plan_pin=pin(plan_path),recovery_ledger_pin=pin(root/'ledger.json'),
        selection_proof_pin=proof_pin,summary_pin=proof['summary_pin'],inherited_verified_fits=12,new_fits=0,new_backtests=0,
        original_failed_ledger_preserved=True,representation_change='Only side parity views cast to pandas string; numeric values/tolerances unchanged')
    return save(oldroot/'verification_recovery/selection_handoff.json',handoff)

def selection(plan_path,guard=lambda:None):
    from .paper15_monthly import select_monthly
    recovery=validate(plan_path);p=read(check(recovery['parent_plan_pin']));cfg=read(check(p['protocol_pin']));root=Path(recovery['output_directory'])
    proof=read(check(recovery['model_proof_pin']));need(proof['status']=='PASS' and proof['verified_fits']==12,'Inherited verified12fits')
    prep=read(check(recovery['prepared_receipt_pin']));out=root/'selection';need(not out.exists(),'Preserve selection attempt');out.mkdir(parents=True)
    save(out/'STARTED.json',dict(plan_pin=pin(plan_path),new_fits=0,new_backtests=0));artifacts={};rows=[];gaps=[]
    for n,kind in itertools.product((5,10),('rf','xgb')):
        guard();arm=f'{kind}_paper15_{n}y'
        pins=[p['old_forecasts'][f'{n}/{y}/{kind}'] for y in range(2020,2024)]+[proof['forecasts'][f'{n}/{y}/{kind}'] for y in range(2024,2027)]
        pred=pd.concat([pq.read_table(check(item)).to_pandas()[FORECAST_META+list(PROBS)] for item in pins],ignore_index=True)
        need(pred.outcome_month.nunique()==76 and not pred.duplicated(list(KEYS)).any(),'Full forecast chronology')
        payoff=read(check(prep['artifacts'][f'payoffs{n}']));ledger,picks=select_monthly(pred,payoff,restricted=cfg['execution_restrictions'])
        old=pq.read_table(check(p['old_selected'][arm])).to_pandas()
        compare=['source_security_id','signal_date','side','intended_weight','RET']
        left=picks.loc[picks.outcome_month.le('2023-12'),compare].sort_values(list(KEYS)).reset_index(drop=True)
        right=old[compare].sort_values(list(KEYS)).reset_index(drop=True)
        assert_old_selection_equal(left,right)
        panel=pq.read_table(check(prep['artifacts'][f'panel{n}']),columns=[*KEYS,'label_return','target_available']).to_pandas()
        joined=picks.merge(panel,on=list(KEYS),how='left',validate='one_to_one')
        missing=joined.loc[joined.outcome_month.ge('2024-01') & ~joined.target_available].copy()
        missing['arm']=arm;gaps.append(missing[['arm',*KEYS,'outcome_month','intrinio_security_id','side','RET']])
        for name,frame in [('forecasts',pred),('original_ledger',ledger),('selected',picks)]:artifacts[arm+'/'+name]=table_save(out/arm/(name+'.parquet'),frame)
        rows.append(dict(arm=arm,added_months=28,added_original_rows=int(pred.outcome_month.ge('2024-01').sum()),
            added_predictions=int((pred.outcome_month.ge('2024-01') & pred.prediction_eligible).sum()),
            added_selected_rows=int(picks.outcome_month.ge('2024-01').sum()),added_selected_monthly_endpoint_gaps=len(missing),
            full_forecast_months=76,common_candidate_months=75,old_selected_rows_byte_values_unchanged=True))
    artifacts['coverage']=table_save(out/'coverage.parquet',pd.DataFrame(rows))
    artifacts['selected_endpoint_gaps']=table_save(out/'selected_endpoint_gaps.parquet',pd.concat(gaps,ignore_index=True))
    return save(out/'audit.json',dict(status='SELECTION_PENDING_INDEPENDENT_CHECK',plan_pin=pin(plan_path),model_proof_pin=recovery['model_proof_pin'],
        artifacts=artifacts,summary=rows,new_fits=0,new_backtests=0,no_future_quote_selection_filter=True,old_selections_preserved=True,
        endpoint_gap_inventory_is_not_an_execution_or_corporate_action_audit=True))
