"""Passive standard-library reader for independently checked75month results."""
from pathlib import Path
import hashlib,json


def read_extension_backtest(run_directory):
    root=Path(run_directory).resolve()
    def read(p):
        p=Path(p)
        if p.stat().st_size>2*1024**2:raise ValueError('Small saved JSON report ceiling')
        return json.loads(p.read_text())
    def check(item):
        p=Path(item['path']).resolve()
        if not Path(item['path']).is_absolute() or item['size_bytes']>8*1024**2:raise ValueError('Bounded saved report pin required')
        h=hashlib.sha256(p.read_bytes()).hexdigest()
        if p.stat().st_size!=item['size_bytes'] or h!=item['sha256']:raise ValueError('Changed verified report artifact')
        return p
    ledger=read(root/'ledger.json')
    if ledger.get('fits_started')!=0 or ledger.get('fits_completed')!=0:raise ValueError('Zero-fit realization required')
    if ledger['status']!='COMPLETE_VERIFIED' or ledger['phase'] is not None:raise ValueError('Independent realization verification is not complete')
    phase=next(x for x in ledger['completed'] if x['key']=='verify_replay');receipt=read(check(phase['receipt']));proof_pin=receipt['result'];proof=read(check(proof_pin))
    if proof['plan_pin']!=ledger['plan_pin'] or proof['status']!='PASS' or proof['verified_scenarios']!=48 or proof['months']!=75 or proof['sessions']!=1569:raise ValueError('Incomplete comparison proof')
    check(ledger['plan_pin']);check(proof['audit_pin']);check(proof['input_proof_pin'])
    vp=root/'report/verification.json';verified=read(vp)
    if verified['replay_proof_pin']!=proof_pin or verified['status']!='PASS_VERIFIED_AGGREGATES_AND_CURVES':raise ValueError('Report proof mismatch')
    check(verified['ledger_pin']);check(verified['worker_pin']);body=read(check(verified['report_pin']))
    if len(body['comparison'])!=48 or body['complete_paths']!=proof['complete_paths'] or body['withheld_paths']!=proof['withheld_paths']:raise ValueError('Full comparison denominator changed')
    for item in verified['source_report_pins'].values():check(item)
    for item in verified['figures'].values():check(item)
    return dict(status='COMPLETE_VERIFIED',**body,figures={k:v['path'] for k,v in verified['figures'].items()},
        independent_proof=str(check(proof_pin)),report_proof=str(vp),new_fits=0)
