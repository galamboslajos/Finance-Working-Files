"""Passive small-JSON coverage reader; no source arrays or estimator imports."""
import hashlib
import json
from pathlib import Path


def _read(path):
    p=Path(path)
    if p.stat().st_size>128*1024:raise ValueError('Small report ceiling')
    return json.loads(p.read_text())


def _check(item):
    p=Path(item['path'])
    if not p.is_absolute() or p.stat().st_size!=item['size_bytes']:raise ValueError('Report pin size/path')
    if hashlib.sha256(p.read_bytes()).hexdigest()!=item['sha256']:raise ValueError('Report pin hash')
    return p


def read_extension_inputs(run):
    root=Path(run).resolve();ledger=_read(root/'ledger.json')
    if ledger['status']!='COMPLETE_VERIFIED_INPUTS' or ledger['fits_started']!=0:raise ValueError('Verified zero-fit input audit required')
    receipt=next(x['receipt'] for x in ledger['completed'] if x['key']=='verify_data')
    phase=_read(_check(receipt));source=_read(_check(phase['result']))
    report=_read(root/'report/verification.json')
    if report['status']!='PASS_AGGREGATE_RECONCILIATION' or report['source_proof_pin']!=phase['result']:
        raise ValueError('Independent coverage proof')
    readplan=_read(_check(report['read_plan_pin']))
    if readplan['source_proof_pin']!=phase['result']:raise ValueError('Wrong coverage source')
    if source['status']!='PASS_INPUT_SOURCE_ARRAYS_AND_MONTHLY_FORMULAS' or source['plan_pin']!=ledger['plan_pin']:
        raise ValueError('Independent source proof')
    # Authenticating JSON receipts reads no financial panel or source price.
    _check(source['prepared_receipt_pin']);_check(report['worker_pin']);_check(readplan['worker_pin'])
    result=_read(_check(report['summary_pin']))
    if result['status']!='VERIFIED_INPUT_COVERAGE_ONLY' or result['new_fits']!=0 or result['new_backtests']!=0:
        raise ValueError('Coverage-only report required')
    return result
