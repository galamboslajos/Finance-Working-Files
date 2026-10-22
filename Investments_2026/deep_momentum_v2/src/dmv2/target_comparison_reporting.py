"""Passive saved status and independently verified aggregate tables only."""
from pathlib import Path
import hashlib
import json


def _need(ok, message):
    if not ok:
        raise ValueError(message)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def _read(path):
    path = Path(path)
    _need(path.stat().st_size <= 16*1024**2, 'Saved comparison JSON exceeds metadata cap')
    return json.loads(path.read_text())


def _check(item):
    path = Path(item['path'])
    _need(path.stat().st_size == item['size_bytes'] and _sha(path) == item['sha256'], 'Saved comparison receipt changed')
    return path


def read_target_comparison(directory):
    """Never fit, reload a model, decode panels, resume or replay anything."""
    root = Path(directory).resolve()
    plan_path = root / 'read_plan.json'
    plan, ledger, launch = (_read(p) for p in (plan_path, root/'ledger.json', root/'launch.json'))
    expected_pin = dict(path=str(plan_path), size_bytes=plan_path.stat().st_size, sha256=_sha(plan_path))
    _need(ledger['model_plan_pin'] == launch['model_plan_pin'] == expected_pin
          and plan['expected_fits'] == 20 and plan['expected_preprocessors'] == 4,
          'Saved progress belongs to another comparison')
    body = {k:v for k,v in plan.items() if k != 'plan_sha256'}
    digest = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                                      allow_nan=False).encode()).hexdigest()
    _need(plan['plan_sha256'] == 'sha256:' + digest, 'Saved plan fingerprint changed')
    _need(0 <= ledger['fits_completed'] <= ledger['fits_started'] <= 20 and plan['financial_2024_plus_allowed'] is False,
          'Saved fit/financial scope differs')
    status = {k:ledger.get(k) for k in ('status','fits_started','fits_completed','active_phase','active_seconds',
              'current_phase_seconds','peak_process_tree_rss_bytes','updated_at','error')}
    result = dict(status=status, comparison=None, yearly=None, models_independently_verified=False,
                  backtest_independently_verified=False, period='3 February 2020–29 December 2023:47months',
                  automatic_computation=False, source_identity_or_investability_certified=False)
    if ledger['status'] != 'COMPLETE_VERIFIED':
        return result
    mp = _read(root/'verification/models.json')
    bp = _read(root/'backtest/independent_verification.json')
    _need(mp.get('status') == bp.get('status') == 'PASS' and mp['phase'] == 'models' and bp['phase'] == 'backtest'
          and mp['model_plan_pin'] == bp['model_plan_pin'] == expected_pin
          and mp['verified_estimator_fits'] == 20 and mp['verified_shared_preprocessors'] == 4
          and bp['verified_scenarios'] == 72 and not mp['financial_2024_plus_values_read']
          and not bp['financial_2024_plus_values_read'], 'Exact independent model/backtest proof required')
    _check(mp['verification_plan_pin']); vp = _read(_check(bp['verification_plan_pin']))
    _check(bp['backtest_plan_pin'])
    import pyarrow.parquet as pq
    for name, rows in [('comparison',72),('yearly',288)]:
        path = root/'backtest'/(name+'.parquet')
        spec = vp['artifacts'][str(path)]
        _check(spec['pin'])
        _need(path.stat().st_size <= 8*1024**2 and pq.ParquetFile(path).metadata.num_rows == rows,
              'Saved aggregate table scope differs')
        result[name] = pq.read_table(path).to_pandas()
    result['models_independently_verified'] = result['backtest_independently_verified'] = True
    return result
