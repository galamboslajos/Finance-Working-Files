"""Passive, byte-authenticated reporting of a separately verified recovery.

The original failed producer ledger is preserved. Only a complete, pinned V3
verification handoff permits decoding the two small original aggregate tables.
Temporary evidence can be relocated by explicit byte-exact, one-to-one copies;
original run artifacts cannot be remapped. No fitting, model reload or replay.
"""
from pathlib import Path
import hashlib
import itertools
import json
import math
import re

SCHEMA = 'dmv2_target_comparison_recovery_handoff_v1'
PROOF_SCHEMAS = frozenset(('dmv2-unstopped-target-independent-verification/v3',
                           'dmv2-unstopped-target-independent-verification/v4',
                           'dmv2-unstopped-target-independent-verification/v5'))
HEADS = ('endpoint', 'cash_long', 'cash_short', 'utility_long', 'utility_short')
ARMS = ('endpoint_matched_clock', 'cash', 'utility')
BASKETS = ('score', 'best_mode', 'second_mode')
CASES = set(itertools.product(ARMS, BASKETS, ('unstopped', 'asset_stop_10pct'), (0, 5, 10, 25)))
RELOCATED_FIELDS = ('independent_backtest_proof_pin', 'verification_plan_pin',
                    'verifier_worker_pin', 'completion_pin', 'launch_pin')
PIN_FIELDS = {'path', 'size_bytes', 'sha256'}


def _need(ok, message):
    if not ok:
        raise ValueError(message)


def _inside(path, parent):
    return path == parent or parent in path.parents


def _read(path):
    path = Path(path)
    _need(path.stat().st_size <= 16 * 1024 ** 2, 'Recovery metadata byte cap exceeded')
    return json.loads(path.read_text())


def _valid_pin(item):
    _need(isinstance(item, dict) and set(item) == PIN_FIELDS,
          'Exact recovery pin fields required')
    _need(type(item['path']) is str and Path(item['path']).is_absolute()
          and type(item['size_bytes']) is int and 0 <= item['size_bytes'] <= 16 * 1024 ** 2
          and type(item['sha256']) is str and re.fullmatch('[0-9a-f]{64}', item['sha256']) is not None,
          'Invalid recovery byte pin')
    return Path(item['path'])


def _check_at(item, path):
    _valid_pin(item)
    path = Path(path).resolve()
    _need(path.stat().st_size == item['size_bytes'], 'Recovery byte size changed')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b''):
            digest.update(block)
    _need(digest.hexdigest() == item['sha256'], 'Recovery byte digest changed')
    return path


def _canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _cases(rows):
    return [(r['arm'], r['basket'], r['variant'], r['cost_bps']) for r in rows]


def read_target_comparison_recovery(directory):
    """Read verified aggregate results; never repair or alter the failed run."""
    root = Path(directory).resolve()
    sidecar_dir = root / 'verification_recovery'
    handoff = _read(sidecar_dir / 'recovery_handoff.json')
    _need(handoff.get('schema') == SCHEMA and handoff.get('status') == 'COMPLETE_VERIFIED_RECOVERY'
          and Path(handoff['original_run_directory']).resolve() == root,
          'Complete recovery handoff required')
    _need(handoff.get('new_estimator_fits') == 0 and handoff.get('new_backtest_replays') == 0
          and handoff.get('financial_2024_plus_values_read') is False
          and handoff.get('source_identity_action_completeness_or_investability_certified') is False,
          'Recovery computation or source-certification scope changed')
    relocations = handoff['evidence_relocations']
    _need(isinstance(relocations, list) and len(relocations) == len(RELOCATED_FIELDS),
          'Exactly five evidence relocations required')
    relocated = {}; copied_paths = set()
    for relocation in relocations:
        _need(set(relocation) == {'original_pin', 'copy_pin'}, 'Explicit evidence relocation required')
        original, copied = relocation['original_pin'], relocation['copy_pin']
        old_path = _valid_pin(original); new_path = _valid_pin(copied).resolve()
        _need(not _inside(old_path.resolve(), root), 'Original run artifacts cannot be relocated')
        _need(new_path.parent == sidecar_dir.resolve(), 'Evidence copies must remain beside recovery handoff')
        _need(original['sha256'] == copied['sha256'] and original['size_bytes'] == copied['size_bytes'],
              'Relocated evidence is not byte-exact')
        _need(str(old_path) not in relocated and new_path not in copied_paths,
              'Evidence relocations must be one-to-one')
        relocated[str(old_path)] = (original, _check_at(copied, new_path)); copied_paths.add(new_path)
    required = {handoff[field]['path'] for field in RELOCATED_FIELDS}
    _need(len(required) == len(RELOCATED_FIELDS) and set(relocated) == required,
          'Relocations differ from exact recovery evidence identities')

    def evidence(item):
        old_path = _valid_pin(item)
        _need(str(old_path) in relocated and relocated[str(old_path)][0] == item,
              'Unsealed recovery evidence identity')
        return relocated[str(old_path)][1]

    def original(item, expected):
        path = _valid_pin(item)
        expected = Path(expected).resolve()
        _need(path.resolve() == expected and _inside(expected, root),
              'Original run evidence redirected')
        return _check_at(item, expected)

    mpin = handoff['model_plan_pin']; lpin = handoff['original_ledger_pin']
    mproofpin = handoff['model_verification_pin']; bpin = handoff['backtest_plan_pin']
    auditpin = handoff['producer_backtest_audit_pin']
    model = _read(original(mpin, root / 'read_plan.json'))
    ledger = _read(original(lpin, root / 'ledger.json'))
    mproof = _read(original(mproofpin, root / 'verification/models.json'))
    backtest = _read(original(bpin, root / 'backtest_plan.json'))
    audit = _read(original(auditpin, root / 'backtest/audit.json'))
    _need(model['plan_sha256'] == 'sha256:' + _canonical({k: v for k, v in model.items() if k != 'plan_sha256'})
          and model['expected_fits'] == 20 and model['expected_preprocessors'] == 4
          and model['financial_2024_plus_allowed'] is False, 'Original model plan contract changed')
    _need(ledger['status'] == 'FAILED_PRESERVED' and ledger['fits_started'] == ledger['fits_completed'] == 20
          and ledger['model_plan_pin'] == mpin, 'Original failed producer must remain preserved')
    _need(mproof['status'] == 'PASS' and mproof['phase'] == 'models' and mproof['model_plan_pin'] == mpin
          and mproof['verified_estimator_fits'] == 20 and mproof['verified_shared_preprocessors'] == 4
          and mproof['new_estimator_fits'] == 0 and mproof['financial_2024_plus_values_read'] is False,
          'Independent twenty-fit model proof required')
    _need({(r['year'], r['head']) for r in mproof['results']} == set(itertools.product((2020, 2021, 2022, 2023), HEADS))
          and len(mproof['results']) == 20, 'Model proof lacks exact twenty distinct year/head checks')
    mvp = _read(original(mproof['verification_plan_pin'], root / 'verification/models_read_plan.json'))
    _need(mvp['phase'] == 'models' and mvp['model_plan_pin'] == mpin
          and mvp['worker_pin'] == mproof['worker_pin'], 'Model proof/read-plan chain changed')
    _need(backtest['plan_sha256'] == _canonical({k: v for k, v in backtest.items() if k != 'plan_sha256'})
          and backtest['model_plan_sha256'] == model['plan_sha256']
          and backtest['scenarios'] == 72 and backtest['economic_months'] == 47
          and backtest['financial_2024_plus_read'] is False
          and Path(backtest['output_directory']).resolve() == root / 'backtest'
          and len(backtest['sessions']) == 985 and backtest['sessions'] == sorted(set(backtest['sessions']))
          and backtest['sessions'][0] == '2020-02-03' and backtest['sessions'][-1] == '2023-12-29',
          'Original backtest plan contract changed')
    _need(audit['plan_sha256'] == backtest['plan_sha256'] and audit['model_verification_pin'] == mproofpin
          and audit['status'] == 'COMPLETE_PENDING_INDEPENDENT_VERIFICATION',
          'Original backtest production audit changed')
    vpin = handoff['verification_plan_pin']; wpin = handoff['verifier_worker_pin']
    ppin = handoff['independent_backtest_proof_pin']
    vplan = _read(evidence(vpin)); evidence(wpin)
    proof = _read(evidence(ppin)); launch = _read(evidence(handoff['launch_pin']))
    completion = _read(evidence(handoff['completion_pin']))
    _need(vplan['schema'] == proof['schema'] and proof['schema'] in PROOF_SCHEMAS
          and vplan['status'] == 'SEALED_BEFORE_FINANCIAL_READ' and vplan['phase'] == 'backtest'
          and vplan['model_plan_pin'] == mpin and vplan['backtest_plan_pin'] == bpin
          and vplan['model_verification_pin'] == mproofpin and vplan['worker_pin'] == wpin
          and Path(vplan['run_directory']).resolve() == root
          and Path(vplan['receipt_directory']) == Path(vpin['path']).parent
          and vplan['new_estimator_fits'] == 0 and vplan['financial_2024_plus_authorized'] is False,
          'Separate recovery verification plan chain changed')
    _need(proof['status'] == 'PASS' and proof['phase'] == 'backtest'
          and proof['verification_plan_pin'] == vpin and proof['worker_pin'] == wpin
          and proof['model_plan_pin'] == mpin and proof['backtest_plan_pin'] == bpin
          and proof['model_verification_pin'] == mproofpin and proof['producer_backtest_audit_pin'] == auditpin
          and proof['new_estimator_fits'] == 0 and proof['financial_2024_plus_values_read'] is False
          and proof['source_identity_action_completeness_or_investability_certified'] is False
          and proof['production_selection_or_accounting_used_as_expected_answer'] is False,
          'Independent recovery backtest PASS required')
    proof_cases = _cases(proof['scenario_checks'])
    _need(proof['verified_scenarios'] == len(proof_cases) == 72 and set(proof_cases) == CASES,
          'Independent proof lacks complete distinct scenario grid')
    _need(launch['status'] == 'SEALED_VERIFICATION_ONLY' and launch['parent_ledger_pin'] == lpin
          and launch['model_verification_pin'] == mproofpin and launch['verification_plan_pin'] == vpin
          and launch['worker_pin'] == wpin and launch['new_estimator_fits'] == launch['new_backtests'] == 0
          and launch['original_run_mutations'] is False
          and launch['cumulative_reserved_derived_bytes'] <= launch['maximum_cumulative_derived_bytes']
          and launch['new_reserved_derived_bytes'] == vplan['derived_read_bytes'],
          'Verification-only launch chain or budget changed')
    _need(completion['status'] == 'PASS_VERIFICATION_ONLY' and completion['original_run_preserved'] is True
          and completion['proof_pin'] == ppin and completion['new_estimator_fits'] == completion['new_backtests'] == 0
          and math.isfinite(completion['active_seconds']) and 0 <= completion['active_seconds'] <= launch['maximum_new_active_seconds']
          and 0 <= completion['peak_rss_bytes'] <= launch['rss_maximum_bytes'],
          'Completed resource-bounded verification-only receipt required')
    inventory = vplan['artifacts']
    # Authenticate the production audit through the independently sealed inventory too.
    _need(inventory[str(root / 'backtest/audit.json')]['pin'] == auditpin,
          'Recovery read inventory does not authenticate original production audit')
    import pyarrow.parquet as pq
    result = {}
    for name, expected_rows in (('comparison', 72), ('yearly', 288)):
        path = root / 'backtest' / (name + '.parquet')
        spec = inventory[str(path)]
        _need(audit['artifacts'][name + '.parquet'] == spec['pin'], 'Aggregate producer/independent byte pins differ')
        original(spec['pin'], path)
        _need(path.stat().st_size <= 8 * 1024 ** 2 and pq.ParquetFile(path).metadata.num_rows == expected_rows,
              'Aggregate report exceeds exact small-table scope')
        result[name] = pq.read_table(path).to_pandas()
    rows = result['comparison'].to_dict('records')
    _need(len(rows) == 72 and set(_cases(rows)) == CASES, 'Aggregate scenario grid differs from independent proof')
    expected_complete = dict(zip(proof_cases, (r['complete_proxy_path'] for r in proof['scenario_checks'])))
    _need(all(type(r['complete_proxy_path']) is bool and r['complete_proxy_path'] is expected_complete[c]
              for r, c in zip(rows, _cases(rows))), 'Aggregate withheld status differs from independent proof')
    year_rows = result['yearly'].to_dict('records')
    _need(len(year_rows) == 288 and {(*c, r['year']) for c, r in zip(_cases(year_rows), year_rows)}
          == {(*c, year) for c in CASES for year in (2020, 2021, 2022, 2023)},
          'Annual aggregate comparison grid changed')
    return dict(status='COMPLETE_VERIFIED_RECOVERY', original_producer_status=ledger['status'],
        comparison=result['comparison'], yearly=result['yearly'],
        models_independently_verified=True, backtest_independently_verified=True,
        automatic_computation=False, new_estimator_fits=0, new_backtest_replays=0,
        financial_2024_plus_values_read=False, source_identity_or_investability_certified=False,
        period='3 February 2020–29 December 2023: 47 months / 985 sessions',
        handoff_path=str(sidecar_dir / 'recovery_handoff.json'))
