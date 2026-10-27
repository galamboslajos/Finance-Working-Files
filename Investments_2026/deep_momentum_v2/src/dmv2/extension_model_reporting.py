"""Passive extension progress/verified small reports; no estimator imports."""
from pathlib import Path
import hashlib
import json


def _read(path):
    p=Path(path)
    if p.stat().st_size>128*1024:raise ValueError('Small JSON report ceiling')
    return json.loads(p.read_text())


def _check(item):
    p=Path(item['path'])
    if not p.is_absolute() or p.stat().st_size!=item['size_bytes']:raise ValueError('Pinned report path/size')
    if hashlib.sha256(p.read_bytes()).hexdigest()!=item['sha256']:raise ValueError('Pinned report hash')
    return p


def read_extension_models(run):
    root=Path(run).resolve();ledger=_read(root/'ledger.json')
    if not 0<=ledger['fits_completed']<=ledger['fits_started']<=12:raise ValueError('Frozen fit count')
    result={k:ledger[k] for k in ('status','phase','fits_started','fits_completed','active_seconds','peak_worker_rss_bytes')}
    result['maximum_fits']=12;result['new_backtests']=0
    handoff=root/'verification_recovery/selection_handoff.json'
    if ledger['status']=='FAILED_PRESERVED' and handoff.exists():
        return _recovered(root,ledger,result)
    if ledger['status']!='COMPLETE_VERIFIED_MODELS_AND_SELECTION':
        if 'error' in ledger:result['error']=ledger['error']
        return result
    if ledger['fits_completed']!=12 or ledger['phase'] is not None:raise ValueError('Verified completion count/state')
    _check(ledger['plan_pin'])
    proofs={}
    for key in ('verify_data','verify_models','verify_selection'):
        item=next(x['receipt'] for x in ledger['completed'] if x['key']==key)
        phase=_read(_check(item));proofs[key]=(phase['result'],_read(_check(phase['result'])))
    dp,data=proofs['verify_data'];mp,models=proofs['verify_models'];sp,selection=proofs['verify_selection']
    for _,proof in proofs.values():
        if proof['status']!='PASS' or proof['plan_pin']!=ledger['plan_pin']:raise ValueError('Independent phase proof')
        _check(proof['worker_pin']);_check(proof['verification_read_plan_pin'])
    if models['data_proof_pin']!=dp or models['verified_fits']!=12 or selection['model_proof_pin']!=mp:
        raise ValueError('Independent input/model/selection chain')
    _check(data['prepared_receipt_pin']);_check(selection['audit_pin'])
    summary=_read(_check(selection['summary_pin']))
    if summary['status']!='VERIFIED_MODELS_AND_RET_SELECTIONS_ONLY' or summary['completed_fits']!=12 or summary['new_backtests']!=0:
        raise ValueError('Model-only summary scope')
    result.update(summary);result['status']=ledger['status'];return result


def _recovered(root,ledger,result):
    handoff=_read(root/'verification_recovery/selection_handoff.json')
    if handoff['status']!='PASS_INHERITED_12_MODELS_AND_RECOVERED_SELECTIONS' or handoff['new_fits']!=0 or handoff['new_backtests']!=0:
        raise ValueError('Zero-fit recovery handoff')
    if ledger['fits_completed']!=12 or handoff['parent_plan_pin']!=ledger['plan_pin']:
        raise ValueError('Inherited model count/plan')
    _check(handoff['parent_ledger_pin']);_check(handoff['parent_plan_pin'])
    rp=handoff['recovery_plan_pin'];plan=_read(_check(rp));recovery=_read(_check(handoff['recovery_ledger_pin']))
    if plan['max_fits']!=0 or plan['max_backtests']!=0 or recovery['status']!='COMPLETE_VERIFIED_SELECTION_RECOVERY' or recovery['fits_started']!=0 or recovery['fits_completed']!=0:
        raise ValueError('Recovery performed no new fitting/trading')
    if recovery['plan_pin']!=rp or plan['parent_plan_pin']!=ledger['plan_pin']:
        raise ValueError('Recovered plan binding')
    phases={}
    for key in ('verify_data','verify_models'):
        phase=_read(_check(next(x['receipt'] for x in ledger['completed'] if x['key']==key)))
        phases[key]=(phase['result'],_read(_check(phase['result'])))
    dp,data=phases['verify_data'];mp,models=phases['verify_models']
    if models['status']!='PASS' or models['verified_fits']!=12 or models['data_proof_pin']!=dp or mp!=handoff['inherited_model_proof_pin'] or plan['model_proof_pin']!=mp:
        raise ValueError('Inherited independent twelve-model proof')
    if data['status']!='PASS' or models['plan_pin']!=ledger['plan_pin'] or data['plan_pin']!=ledger['plan_pin']:
        raise ValueError('Inherited independent data proof')
    phase=_read(_check(next(x['receipt'] for x in recovery['completed'] if x['key']=='verify_selection')))
    if phase['result']!=handoff['selection_proof_pin']:raise ValueError('Recovered selection phase')
    selection=_read(_check(handoff['selection_proof_pin']))
    if selection['status']!='PASS' or selection['plan_pin']!=rp or selection['model_proof_pin']!=mp or selection['summary_pin']!=handoff['summary_pin']:
        raise ValueError('Recovered independent selection proof')
    for proof in (data,models,selection):
        _check(proof['worker_pin']);_check(proof['verification_read_plan_pin'])
    _check(data['prepared_receipt_pin']);_check(selection['audit_pin'])
    summary=_read(_check(handoff['summary_pin']))
    if summary['status']!='VERIFIED_MODELS_AND_RET_SELECTIONS_ONLY' or summary['completed_fits']!=12 or summary['new_backtests']!=0:
        raise ValueError('Verified recovered model-only summary')
    result.update(summary);result.update(status='COMPLETE_VERIFIED_MODELS_AND_SELECTION',phase=None,
        original_status='FAILED_PRESERVED',zero_fit_selection_recovery=True,active_seconds=recovery['active_seconds'],
        peak_worker_rss_bytes=max(ledger['peak_worker_rss_bytes'],recovery['peak_worker_rss_bytes']))
    return result
