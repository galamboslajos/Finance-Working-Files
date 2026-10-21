#!/usr/bin/env python3
"""Independent execution-repair audit preserving the original verified RET.

No target, probability, feature or model is decoded. The original RET PASS and
twenty immutable selection frames are inherited by exact hash. Quote overlays
and added legal actions are checked independently before reusing the reviewed
independent cash/units equations for all forty fixed scenarios.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import importlib.util
import itertools
import json
import math
from numbers import Real
import os
from pathlib import Path
import platform
import resource
import time
from types import SimpleNamespace

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

BASE_SHA = '16c3651289258bed24a83d500f6b8589fac3bcb4d358e516fe981acec527a794'
PARENT_VERIFIER_SHA = 'cd315f90848349b3f64045923e5410ee1adea92bec0eaafcd72f974f03c7c89e'
ARMS = ('rf_paper15', 'xgb_paper15', 'rf_extended', 'xgb_extended', 'rf_extended_tuned')
VARIANTS = ('unstopped', 'asset_stop_10pct')
FEES = (0, 5, 10, 25)
BASKET = 'ret'
LEDGERS = ('daily', 'trades', 'holdings', 'failures', 'stops', 'action_events',
           'market_trades', 'orders', 'identity_handovers')
ACTION_FIELDS = ('trading_date', 'intrinio_security_id', 'adj_open', 'adj_high', 'adj_low', 'adj_close',
    'frequency', 'intraperiod', 'price_quality_status', 'adjusted_ohlc_bounds_valid',
    'raw_ohlc_bounds_valid', 'factor', 'split_ratio', 'dividend', 'open', 'high', 'low', 'close')
READ_LIMIT = 1024**3


class VerificationError(RuntimeError):
    pass


def need(condition, message):
    if not bool(condition): raise VerificationError(message)


def canonical(value):
    return 'sha256:' + hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def normalize_scalar(value):
    if value is None or pd.isna(value): return None
    if isinstance(value, (datetime, pd.Timestamp)): return value.strftime('%Y-%m-%d')
    if isinstance(value, np.generic): return value.item()
    return value


def bar_record(row):
    fields = set(row.index) if isinstance(row, pd.Series) else set(row)
    need(set(ACTION_FIELDS) <= fields, 'Paired quote row lacks a frozen accounting field')
    return {field: normalize_scalar(row[field]) for field in ACTION_FIELDS}


def bar_hash(row):
    return canonical(bar_record(row))


def acceptable(row):
    row = bar_record(row)
    if (row['frequency'] != 'daily' or row['intraperiod'] is not False
            or row['price_quality_status'] not in ('accepted', 'ok')
            or row['raw_ohlc_bounds_valid'] is not True or row['adjusted_ohlc_bounds_valid'] is not True):
        return False
    for field in ('open', 'high', 'low', 'close', 'adj_open', 'adj_high', 'adj_low', 'adj_close'):
        if isinstance(row[field], bool) or not isinstance(row[field], Real) or not math.isfinite(row[field]) or row[field] <= 0:
            return False
    return (row['low'] <= min(row['open'], row['close']) <= max(row['open'], row['close']) <= row['high']
        and row['adj_low'] <= min(row['adj_open'], row['adj_close']) <= max(row['adj_open'], row['adj_close']) <= row['adj_high'])


def append_events(original, additions, *, sessions):
    """Append reviewed definitions; old terms remain byte-logically identical."""
    need(isinstance(original, list) and isinstance(additions, list), 'Explicit action lists required')
    result = deepcopy(original)
    event_ids = {event['event_id'] for event in original}
    predecessor_dates = {(event['predecessor_id'], event['effective_date']) for event in original}
    for event in additions:
        need(isinstance(event, dict) and event.get('verified') is True
             and event.get('evidence_reference') and event.get('event_id') not in event_ids
             and isinstance(event.get('predecessor_id'), str) and bool(event['predecessor_id'].strip())
             and event.get('effective_date') in sessions
             and (event['predecessor_id'], event['effective_date']) not in predecessor_dates,
             'Added action lacks unique dated primary evidence')
        cash, ratio = event.get('cash_per_raw_share'), event.get('stock_ratio')
        need(all(isinstance(value, Real) and not isinstance(value, (bool, np.bool_))
                 and math.isfinite(value) and value >= 0 for value in (cash, ratio))
             and (cash > 0 or ratio > 0), 'Zero/negative/nonfinite action entitlement is unsupported')
        need(event.get('action_type', 'verified_retirement') in
             ('cash_merger', 'stock_merger', 'mixed_merger', 'verified_retirement'), 'Unsupported legal action type')
        if ratio:
            need(isinstance(event.get('successor_id'), str) and bool(event['successor_id'].strip())
                 and event['successor_id'] != event['predecessor_id'], 'Stock entitlement needs an explicit distinct successor')
        else:
            need(event.get('successor_id') in (None, ''), 'Cash-only entitlement cannot invent successor holdings')
        event_ids.add(event['event_id']); predecessor_dates.add((event['predecessor_id'], event['effective_date']))
        result.append(deepcopy(event))
    need(result[:len(original)] == original, 'Original verified action definitions changed')
    return result


def reconstruct_overlay(original, repairs, amendments, *, sessions, authenticate_rejection=None):
    """Independently apply exact append/replace keys and before/after hashes."""
    need(set(ACTION_FIELDS) <= set(original) and set(ACTION_FIELDS) <= set(repairs), 'Frozen paired quote schema required')
    need(len(repairs) <= 10000 and not original.duplicated(['trading_date', 'intrinio_security_id']).any()
         and not repairs.duplicated(['trading_date', 'intrinio_security_id']).any(), 'Duplicate/unbounded paired quote source')
    old = { (row['trading_date'], row['intrinio_security_id']): bar_record(row)
            for row in original.to_dict('records') }
    new = { (row['trading_date'], row['intrinio_security_id']): bar_record(row)
            for row in repairs.to_dict('records') }
    keys = [(row['trading_date'], row['intrinio_security_id']) for row in amendments]
    need(len(keys) == len(set(keys)) and set(keys) == set(new), 'Repair rows must exactly equal the declared amendment keys')
    changes = []
    for declaration in amendments:
        key = (declaration['trading_date'], declaration['intrinio_security_id'])
        need(key[0] in sessions and '2020-01-01' <= key[0] <= '2023-12-31'
             and isinstance(key[1], str) and bool(key[1].strip()), 'Repair quote key escaped approved dates/identity')
        replacement = new[key]
        need(acceptable(replacement), 'New paired quote fails source or OHLC quality checks')
        need(bar_hash(replacement) == declaration['replacement_bar_sha256'], 'Replacement paired quote hash changed')
        if declaration['kind'] == 'append':
            need(key not in old and declaration.get('original_bar_sha256') is None, 'Append would overwrite an existing quote')
        elif declaration['kind'] == 'replace':
            need(key in old and declaration.get('original_bar_sha256') == bar_hash(old[key]), 'Replacement before-bar hash changed')
            if acceptable(old[key]):
                need(declaration.get('rejection_evidence_pin') is not None and authenticate_rejection is not None,
                     'Accepted old quote cannot be replaced without explicit pinned rejection evidence')
                authenticate_rejection(declaration)
        else:
            raise VerificationError('Unknown quote amendment kind')
        changes.append({'trading_date': key[0], 'intrinio_security_id': key[1], 'kind': declaration['kind'],
                        'before_bar_sha256': None if key not in old else bar_hash(old[key]),
                        'after_bar_sha256': bar_hash(replacement)})
        old[key] = replacement
    output = pd.DataFrame([old[key] for key in sorted(old)], columns=list(ACTION_FIELDS))
    return output, changes


def import_pinned(parent, name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def load_parent_helper(root, identity):
    pin = identity['parent_verifier_pin']
    path = (Path(root)/pin['path']).resolve()
    need(path.is_relative_to(root) and path.is_file() and path.stat().st_size == pin['size_bytes'], 'Parent verifier pin leaves root/changes size')
    need(pin['sha256'] == PARENT_VERIFIER_SHA
         and hashlib.sha256(path.read_bytes()).hexdigest() == PARENT_VERIFIER_SHA,
         'Original independent RET verifier changed')
    module = import_pinned(None, '_preserved_independent_ret_audit', path)
    need(module.BASE_SHA == BASE_SHA, 'Original independent cash/units equation identity changed')
    return module


def json_file(path):
    path = Path(path); need(path.is_file() and path.stat().st_size <= 16*1024**2, 'Bounded metadata JSON required')
    return json.loads(path.read_text())


def authenticate(root, run):
    from importlib.metadata import version
    identity, root_audit = json_file(run/'build_identity.json'), json_file(run/'audit.json')
    need(identity['schema'] == 'dmv2-ret-repair-root/v1' and identity['project_root'] == str(root)
         and identity['run_root'] == str(run) and root_audit['status'] == 'COMPLETE_PENDING_INDEPENDENT_VERIFICATION'
         and root_audit['build_identity_sha256'] == canonical(identity), 'Completed separate execution-repair identity required')
    parent = load_parent_helper(root, identity)
    cfg = json_file(parent.pinned(root, identity['configuration_pin']))
    need(cfg == identity['configuration'] and cfg.get('schema') == 'dmv2-ret-repair-execution/v1'
         and cfg.get('status') == 'FROZEN_EXECUTION_AUTHORIZED' and cfg.get('execution_authorized') is True
         and cfg.get('arm_ids') == list(ARMS) and cfg.get('scenarios') == 40
         and cfg.get('resource_limits') == parent.LIMITS and cfg.get('allow_withheld_paths') is True
         and cfg.get('scores_and_selections_unchanged') is True
         and cfg.get('cloud_operations') is False and cfg.get('financial_2024_plus_authorized') is False
         and all(type(cfg.get(key)) is int and cfg[key] == 0 for key in
                 ('new_estimator_fits', 'new_preprocessor_fits', 'new_inference_calls')),
         'Frozen execution-only repair authority/resources changed')
    runtime = {'python': platform.python_version(), **{name: version(name) for name in
        ('numpy', 'pandas', 'pyarrow', 'exchange-calendars', 'psutil', 'threadpoolctl')}}
    need(identity['runtime'] == runtime and identity['scores_and_selections_unchanged'] is True
         and identity['models_fitted'] == identity['new_inference_calls'] == 0
         and identity['learning_inputs_rebuilt'] is False
         and identity['potential_learning_input_impact_not_rebuilt'] is True,
         'Repair changed learning inputs, models or numerical runtime')
    for pin in [*identity['code_pins'], *identity['source_pins']]: parent.pinned(root, pin)
    need(any(parent.pinned(root, pin) == Path(__file__).resolve() for pin in identity['code_pins']),
         'Independent repair verifier absent from sealed code identity')
    for field in ('parent_identity_pin', 'parent_audit_pin', 'parent_verification_pin',
                  'parent_verifier_pin', 'independent_equations_pin', 'repair_manifest_pin'):
        need(identity[field] == cfg[field], 'Repair dependency pin differs from frozen configuration')
        parent.pinned(root, identity[field])
    v = parent.load_equations(root)
    need(identity['independent_equations_pin']['sha256'] == BASE_SHA, 'Reviewed independent cash/units equations changed')
    original_run = parent.pinned(root, identity['parent_identity_pin']).parent
    need(original_run == parent.local(root, identity['parent_directory'])
         and original_run.is_relative_to(root/'outputs/ret_backtest')
         and parent.pinned(root, identity['parent_audit_pin']) == original_run/'audit.json',
         'Original independently checked RET run was substituted')
    values = parent.authenticate(root, original_run, v)
    old_identity, old_pre, old_pre_audit, old_phase, old_audit, dates, windows, source, handovers = values
    receipt_path = parent.pinned(root, identity['parent_verification_pin']); receipt = json_file(receipt_path)
    need(receipt['schema'] == 'dmv2-ret-backtest-independent-verification/v1' and receipt['status'] == 'PASS'
         and receipt['build_identity_sha256'] == identity['parent_identity_sha256'] == canonical(old_identity)
         and receipt['root_audit_sha256'] == identity['parent_audit_pin']['sha256']
         and receipt['backtest_audit_sha256'] == parent.digest(original_run/'backtest/audit.json')
         and receipt['preflight_audit_sha256'] == parent.digest(original_run/'preflight/audit.json')
         and receipt['backtest_phase_identity_sha256'] == canonical(old_phase)
         and receipt['script_sha256'] == identity['parent_verifier_pin']['sha256'] == PARENT_VERIFIER_SHA
         and receipt['scenarios'] == 40 and receipt['sessions'] == 1006 and receipt['outcome_months'] == 48
         and receipt['all_original_forecast_keys_and_admissions_preserved'] is True
         and receipt['all_five_arms_share_identical_payoff_vectors'] is True,
         'Original RET arithmetic/selection/accounting PASS detached')
    need(identity['parent_phase'] == old_phase
         and identity['source_reference_directory'] == old_identity['source_reference_directory'],
         'Original RET execution evidence changed before repair')
    expected_selection = {f'selections/{arm}/{name}.parquet': old_phase['selection_artifacts'][f'selections/{arm}/{name}.parquet']
        for arm in ARMS for name in ('original_ledger', 'selected', 'monthly', 'rebalances')}
    need(identity['parent_selection_artifacts'] == expected_selection and len(expected_selection) == 20,
         'Original twenty RET decision frames were copied, replaced or omitted')
    for pin in expected_selection.values(): parent.pinned(root, pin)
    manifest = json_file(parent.pinned(root, identity['repair_manifest_pin']))
    need(manifest['schema'] == 'dmv2-ret-execution-repairs/v1' and manifest['status'] == 'REVIEWED_FROZEN'
         and manifest['parent_ret_identity_sha256'] == canonical(old_identity)
         and isinstance(manifest['source_pins'], list) and bool(manifest['source_pins'])
         and isinstance(manifest['additional_events'], list) and isinstance(manifest['quote_amendments'], list),
         'Separately reviewed execution-repair manifest required')
    for pin in manifest['source_pins']: parent.pinned(root, pin)
    additional_events = append_events(old_phase['event_registry']['events'], manifest['additional_events'], sessions=dates)
    for change in manifest['quote_amendments']:
        need(change['trading_date'] in dates and change['kind'] in ('append', 'replace'), 'Repair quote scope changed')
        if change.get('rejection_evidence_pin'): parent.pinned(root, change['rejection_evidence_pin'])
    need(bool(manifest['quote_amendments']) == bool(manifest.get('quotes_pin')), 'Repair quote data/declarations disagree')
    if manifest.get('quotes_pin'):
        path = parent.pinned(root, manifest['quotes_pin'])
        need(path.is_relative_to(root/'data/derived') and path.suffix == '.parquet'
             and path.stat().st_size <= 16*1024**2 and pq.ParquetFile(path).metadata.num_rows <= 10000
             and set(ACTION_FIELDS) <= set(pq.ParquetFile(path).schema_arrow.names), 'Bounded complete paired repair quote file required')
    pre, backtest = run/'preflight', run/'backtest'
    pre_phase, pre_audit = json_file(pre/'build_identity.json'), json_file(pre/'audit.json')
    phase, audit = json_file(backtest/'build_identity.json'), json_file(backtest/'audit.json')
    need(pre_phase['schema'] == 'dmv2-ret-repair-preflight/v1'
         and pre_audit['status'] == 'COMPLETE' and pre_audit['build_identity_sha256'] == canonical(identity)
         and pre_audit['phase_identity_sha256'] == canonical(pre_phase)
         and pre_audit['performance_computed'] is pre_phase['performance_computed'] is False,
         'Execution-repair coverage preflight missing before performance')
    need(phase['schema'] == 'dmv2-ret-repair-phase/v1' and phase['phase'] == audit['phase'] == 'ret_repaired_continuous_backtest_v1'
         and audit['status'] == 'COMPLETE' and audit['phase_identity_sha256'] == canonical(phase)
         and audit['build_identity_sha256'] == phase['parent_identity_sha256'] == canonical(identity)
         and parent.pinned(root, root_audit['backtest_audit_pin']) == backtest/'audit.json'
         and parent.pinned(root, phase['preflight_audit_pin']) == pre/'audit.json'
         and phase['performance_computed'] is True
         and audit['scenarios'] == 40 and audit['sessions'] == 1006 and audit['outcome_months'] == 48
         and audit['scores_and_selections_unchanged'] is True and audit['old_runs_and_pointers_changed'] is False
         and audit['models_fitted'] == audit['new_inference_calls'] == 0 and audit['financial_2024_plus_read'] is False,
         'Repaired forty-scenario accounting identity/scope changed')
    for name in ('parent_selection_artifacts', 'repair_manifest_pin'):
        need(phase[name] == pre_phase[name] == identity[name], 'Replay changed inherited decisions or repair manifest')
    need(phase['event_registry']['events'] == pre_phase['event_registry']['events'] == additional_events,
         'Original legal events changed or an unapproved event was introduced')
    expected_registry = deepcopy(old_phase['event_registry']); expected_registry['events'] = additional_events
    need(phase['event_registry'] == pre_phase['event_registry'] == expected_registry,
         'Repair mutated original event-registry metadata')
    for field in ('historical_retirements', 'identity_handover_registry', 'identity_handover_registry_pin',
                  'identity_handover_approval', 'clock'):
        need(phase[field] == pre_phase[field] == old_phase[field], 'Original retirement/PENN/clock changed: '+field)
    need(phase['original_ret_identity_sha256'] == pre_phase['original_ret_identity_sha256'] == canonical(old_identity),
         'Repaired phase detached the original RET proof')
    need(phase['source_pins'] == pre_phase['source_pins'] == identity['source_pins'], 'Repair source pins changed after preflight')
    for pin in pre_audit['artifacts'].values(): parent.pinned(pre, pin)
    for pin in audit['artifacts'].values(): parent.pinned(backtest, pin)
    return parent, v, identity, cfg, manifest, old_phase, phase, audit, dates, windows, source, handovers


def read_plan(root, run, identity, manifest, old_phase, parent):
    entries = []
    def add(path, passes):
        path = Path(path); need(path.is_file(), 'A reserved independent input is missing')
        entries.append({'path': str(path), 'passes': passes, 'file_size_bytes': path.stat().st_size,
                        'reserved_compressed_bytes': path.stat().st_size * passes})
    source = parent.local(root, identity['source_reference_directory'])
    for year in range(2020, 2024): add(source/'years'/str(year)/'daily_prices_raw.parquet', 2)
    for quote in old_phase['event_registry'].get('supplemental_quote_sources', []): add(parent.pinned(root, quote['pin']), 2)
    if manifest.get('quotes_pin'): add(parent.pinned(root, manifest['quotes_pin']), 2)
    for proof in manifest.get('quote_unit_proofs', []):
        add(parent.pinned(root, proof['source_quotes_pin']), 2)
        if proof.get('paired_source_pin'): add(parent.pinned(root, proof['paired_source_pin']), 2)
        if proof.get('target_controls_pin'): add(parent.pinned(root, proof['target_controls_pin']), 2)
        for field in ('predecessor_control_pin', 'after_control_unavailable_evidence_pin'):
            if proof.get(field): add(parent.pinned(root, proof[field]), 1)
    for pin in identity['parent_selection_artifacts'].values(): add(parent.pinned(root, pin), 2)
    for arm, variant, fee in itertools.product(ARMS, VARIANTS, FEES):
        directory = run/f'backtest/arms/{arm}/{BASKET}/{variant}/cost_{fee:02d}bps'
        for name in (*LEDGERS, 'monthly'): add(directory/(name+'.parquet'), 2)
    for name in ('comparison', 'yearly', 'selection_coverage'): add(run/'backtest'/(name+'.parquet'), 2)
    return entries, sum(item['reserved_compressed_bytes'] for item in entries)


def read_repair_quotes(root, manifest, parent, budget, dates):
    if not manifest.get('quotes_pin'): return pd.DataFrame(columns=list(ACTION_FIELDS))
    path = parent.pinned(root, manifest['quotes_pin'])
    keys = budget.read(path, ['trading_date', 'intrinio_security_id'])
    need(len(keys) == len(manifest['quote_amendments'])
         and keys.trading_date.isin(dates).all()
         and keys.intrinio_security_id.map(lambda value: isinstance(value, str) and bool(value.strip())).all()
         and not keys.duplicated(['trading_date', 'intrinio_security_id']).any(), 'Repair quote keys rejected before financial conversion')
    return budget.read(path, list(ACTION_FIELDS))


def check_unit_proof_metadata(proof, repair_dates, *, unavailable=None):
    """Default two-sided controls; one exact declared RCM provisional waiver."""
    need(proof.get('method') == 'same_provider_dated_route_with_paired_controls'
         and proof.get('factor_method') == 'per_row_observed_source_adj_over_raw'
         and proof.get('max_paired_residual') == 2e-4
         and proof.get('max_control_relative_residual') == 2e-4
         and isinstance(proof.get('intrinio_security_id'), str)
         and isinstance(proof.get('source_native'), str)
         and proof['source_native'] != proof['intrinio_security_id']
         and isinstance(proof.get('identity_evidence_pin'), dict), 'Frozen paired native-route proof required')
    controls = proof.get('control_dates')
    need(isinstance(controls, list) and len(controls) >= 2 and controls == sorted(set(controls))
         and repair_dates and not set(controls) & set(repair_dates), 'Distinct declared source/target unit controls required')
    policy = proof.get('control_coverage_policy', 'before_and_after')
    if policy == 'before_and_after':
        need(controls[0] < min(repair_dates) and controls[-1] > max(repair_dates),
             'Paired native-route controls must surround the quote gap')
    elif policy == 'two_pre_gap_controls_and_primary_dated_one_for_one_reorganization':
        need(proof['intrinio_security_id'] == 'sec_zMM1vd' and proof['source_native'] == 'sec_yKbEk4'
             and controls == ['2021-10-28', '2021-10-29']
             and len(repair_dates) == 22 and min(repair_dates) == '2021-11-01' and max(repair_dates) == '2021-12-01'
             and isinstance(proof.get('after_control_unavailable_evidence_pin'), dict),
             'Provisional two-pre-gap-control exception exceeds the exact dated RCM scope')
        need(isinstance(unavailable, dict) and unavailable.get('financial_columns') == []
             and unavailable.get('common_dates') == [], 'Exact metadata-only unavailable post-control evidence required')
        request = unavailable.get('provenance', {}).get('request', {})
        need(request.get('start') == '2021-12-02' and request.get('end') == '2022-06-21'
             and request.get('date_column') == 'trading_date'
             and set(request.get('filters', {}).get('intrinio_security_id', [])) == {'sec_yKbEk4', 'sec_zMM1vd'}
             and unavailable.get('provenance', {}).get('columns') == ['trading_date', 'intrinio_security_id']
             and unavailable.get('dates', {}).get('sec_zMM1vd', []) == []
             and len(unavailable.get('dates', {}).get('sec_yKbEk4', [])) == 138,
             'RCM post-control absence evidence does not match its original bounded metadata request')
    else:
        raise VerificationError('Unknown native-route unit-control coverage policy')
    return policy


def verify_unit_rows(proof, source, repairs, original, *, target_controls=None, unavailable=None):
    """Validate actual paired bars and copy them with only the execution ID changed."""
    target, native = proof['intrinio_security_id'], proof['source_native']
    repaired = repairs.loc[repairs.intrinio_security_id.eq(target)]
    wanted_dates = sorted(repaired.trading_date.tolist())
    policy = check_unit_proof_metadata(proof, wanted_dates, unavailable=unavailable)
    need(set((*ACTION_FIELDS, 'original_source_native')) <= set(source)
         and source.intrinio_security_id.eq(native).all() and source.original_source_native.eq(native).all()
         and not source.duplicated(['trading_date', 'intrinio_security_id']).any()
         and set(source.trading_date) == set(wanted_dates) | set(proof['control_dates']),
         'Paired native-route source differs from exact repair/control keys')
    indexed = {row['trading_date']: row for row in source.to_dict('records')}
    maxima = []
    for day, row in indexed.items():
        need(acceptable(row), 'Paired route source bar has invalid daily OHLC/quality')
        factor = row['adj_close'] / row['close']
        residual = max(abs((row['adj_'+field]/row[field])/factor - 1.) for field in ('open', 'high', 'low', 'close'))
        need(residual <= proof['max_paired_residual'], 'Per-row observed raw/adjusted source units are inconsistent')
        maxima.append(residual)
    originals = {(row['trading_date'], row['intrinio_security_id']): row for row in original.to_dict('records')
                 if row['intrinio_security_id'] == target and row['trading_date'] in proof['control_dates']}
    if target_controls is not None:
        need(target_controls.intrinio_security_id.eq(target).all()
             and set(target_controls.trading_date) == set(proof['control_dates'])
             and not target_controls.duplicated(['trading_date', 'intrinio_security_id']).any(),
             'External target paired controls changed their exact target/date keys')
        for row in target_controls.to_dict('records'):
            key = (row['trading_date'], target)
            if key in originals: need(bar_hash(originals[key]) == bar_hash(row), 'External paired target control changed an existing original bar')
            else: originals[key] = row
    controls = []
    for day in proof['control_dates']:
        need((day, target) in originals and acceptable(originals[(day, target)]), 'Target-native paired control is missing or rejected')
        a, b = indexed[day], originals[(day, target)]
        residual = max(abs(a[field]/b[field] - 1.) for field in
                       ('open', 'high', 'low', 'close', 'adj_open', 'adj_high', 'adj_low', 'adj_close'))
        need(residual <= proof['max_control_relative_residual'], 'Observed source/target paired controls do not match')
        controls.append({'trading_date': day, 'maximum_raw_and_adjusted_relative_residual': residual,
                         'source_bar_sha256': bar_hash(a), 'target_bar_sha256': bar_hash(b)})
    for row in repaired.to_dict('records'):
        expected = bar_record(indexed[row['trading_date']]); expected['intrinio_security_id'] = target
        need(bar_hash(row) == canonical(expected), 'Repair did not copy exact observed paired source bar with native-only remap')
    return {'intrinio_security_id': target, 'source_native': native, 'method': proof['method'],
        'control_coverage_policy': policy, 'repaired_rows': len(repaired), 'controls': controls,
        'maximum_paired_source_residual': max(maxima), 'exact_observed_source_rows_only': True,
        'factor_interpolation_or_price_fill': False,
        'historical_identity_or_units_certified': False,
        'provisional_one_for_one_reorganization_exception': policy != 'before_and_after'}


PUBLISHED_META = ('unified_security_id', 'trading_date', 'adjustment', 'query_symbol', 'feed',
                  'timeframe', 'selected_source_product_id', 'selected_source_security_id')
PUBLISHED_METHOD = 'published_raw_all_pair_with_dated_share_class'
LISTING_POLICY = 'primary_dated_listing_and_predecessor_price_control'


def published_mask(metadata, proof, dates):
    """Select exact published identity/date/raw-all pairs before OHLC conversion."""
    need(set(PUBLISHED_META) <= set(metadata)
         and proof.get('method') == PUBLISHED_METHOD
         and proof.get('factor_method') == 'per_row_observed_source_adj_over_raw'
         and proof.get('published_source_product_id') == 'russell_1000_2000_3000'
         and isinstance(proof.get('published_unified_security_id'), str)
         and isinstance(proof.get('published_routes'), list), 'Exact published raw/all route contract required')
    routes = proof['published_routes']
    need(bool(routes) and all(isinstance(row, dict) and {'start', 'end', 'query_symbol', 'selected_source_security_id'} <= set(row)
         and row['start'] <= row['end'] and '2020-01-01' <= row['start'] <= row['end'] <= '2023-12-31'
         for row in routes), 'Bounded explicit published source symbol routes required')
    need(all(sum(row['start'] <= day <= row['end'] for row in routes) == 1 for day in dates),
         'Published route intervals overlap or miss a required day')
    routing = pd.Series(False, index=metadata.index)
    for route in routes:
        routing |= (metadata.trading_date.between(route['start'], route['end'])
                    & metadata.query_symbol.eq(route['query_symbol'])
                    & metadata.selected_source_security_id.eq(route['selected_source_security_id']))
    mask = (metadata.unified_security_id.eq(proof['published_unified_security_id'])
            & metadata.selected_source_product_id.eq(proof['published_source_product_id'])
            & metadata.feed.eq('sip') & metadata.timeframe.eq('1Day')
            & metadata.adjustment.isin(['raw', 'all']) & metadata.trading_date.isin(dates) & routing)
    selected = metadata.loc[mask]
    expected = {(day, adjustment) for day in dates for adjustment in ('raw', 'all')}
    need(len(selected) == len(expected) and not selected.duplicated(['trading_date', 'adjustment']).any()
         and set(map(tuple, selected[['trading_date', 'adjustment']].to_numpy())) == expected,
         'Published identity/date/raw-all pair is missing, duplicated or ambiguous')
    return mask


def published_rows(proof, paired, dates):
    """Derive normalized paired accounting fields from literal published values."""
    mask = published_mask(paired.loc[:, list(PUBLISHED_META)], proof, dates)
    selected = paired.loc[mask]
    output = []
    for day in sorted(dates):
        group = selected.loc[selected.trading_date.eq(day)]
        raw = group.loc[group.adjustment.eq('raw')].iloc[0]
        adjusted = group.loc[group.adjustment.eq('all')].iloc[0]
        need(raw.selected_source_security_id == adjusted.selected_source_security_id
             and raw.query_symbol == adjusted.query_symbol, 'Published raw/all pair changed its share-class context')
        need(all(isinstance(row[field], Real) and not isinstance(row[field], (bool, np.bool_))
                 and math.isfinite(row[field]) and row[field] > 0
                 for row in (raw, adjusted) for field in ('open', 'high', 'low', 'close')),
             'Published literal raw/all prices must be finite and positive')
        record = {'trading_date': day, 'intrinio_security_id': proof['source_native'],
            **{field: float(raw[field]) for field in ('open', 'high', 'low', 'close')},
            **{'adj_'+field: float(adjusted[field]) for field in ('open', 'high', 'low', 'close')},
            'frequency': 'daily', 'intraperiod': False, 'price_quality_status': 'accepted',
            'raw_ohlc_bounds_valid': True, 'adjusted_ohlc_bounds_valid': True,
            'factor': float(adjusted['close'])/float(raw['close']), 'split_ratio': None, 'dividend': None,
            'original_source_native': raw.selected_source_security_id}
        need(acceptable(record), 'Published raw/all OHLC pair is invalid')
        factor = record['factor']
        residual = max(abs((record['adj_'+field]/record[field])/factor-1.) for field in ('open', 'high', 'low', 'close'))
        need(residual <= 2e-4, 'Published raw/all pair has inconsistent adjustment units')
        output.append(record)
    return pd.DataFrame(output)


def verify_published_unit_rows(proof, paired, source, repairs, original, *, target_controls=None,
                               predecessor=None, events=()):
    target = proof['intrinio_security_id']; repaired = repairs.loc[repairs.intrinio_security_id.eq(target)]
    control_dates = proof.get('control_dates', [])
    need(proof.get('max_paired_residual') == proof.get('max_control_relative_residual') == 2e-4
         and control_dates == sorted(set(control_dates)), 'Frozen published unit tolerances/control dates changed')
    source_dates = set(repaired.trading_date) | set(control_dates)
    need(source.intrinio_security_id.eq(proof['source_native']).all()
         and set(source.trading_date) == source_dates and len(source) == len(source_dates)
         and 'original_source_native' in source, 'Normalized published source keys changed')
    policy = proof.get('control_coverage_policy', 'before_and_after')
    pair_dates = set(source_dates)
    if policy == LISTING_POLICY:
        need(target == proof['source_native'] == 'execution_cusip:48214T305'
             and control_dates == [] and proof.get('predecessor_control_date') == '2021-06-14'
             and proof.get('stock_ratio') == 3.355 and min(source_dates) == '2021-06-15'
             and max(source_dates) == '2021-07-01' and len(source_dates) == 13,
             'Published listing exception exceeds the exact Grubhub ADS dated scope')
        need(isinstance(predecessor, dict) and predecessor.get('schema') == 'dmv2-ret-repair-predecessor-control/v1',
             'Pinned original Grubhub predecessor control required')
        pair_dates.add(proof['predecessor_control_date'])
    else:
        need(policy == 'before_and_after' and len(control_dates) == 2
             and control_dates[0] < min(repaired.trading_date) <= max(repaired.trading_date) < control_dates[-1],
             'Published raw/all target-native controls must bracket the gap')
    normalized = published_rows(proof, paired, pair_dates)
    expected = normalized.loc[normalized.trading_date.isin(source_dates)].set_index('trading_date')
    for row in source.to_dict('records'):
        wanted = expected.loc[row['trading_date']].to_dict(); wanted['trading_date'] = row['trading_date']
        need(row['original_source_native'] == wanted['original_source_native']
             and bar_hash(row) == bar_hash(wanted), 'Normalized quote differs from literal published raw/all pair')
    for row in repaired.to_dict('records'):
        wanted = expected.loc[row['trading_date']].to_dict()
        wanted.update(trading_date=row['trading_date'], intrinio_security_id=target)
        need(bar_hash(row) == bar_hash(wanted), 'Published repair changed fields beyond its explicit execution-native rebind')
    controls_report = []
    if policy == LISTING_POLICY:
        control = predecessor['row']; day = proof['predecessor_control_date']
        need(control['intrinio_security_id'] == 'sec_zEbnBy' and control['trading_date'] == day,
             'Predecessor control changed its original Grubhub native/date')
        cached = original.loc[original.trading_date.eq(day) & original.intrinio_security_id.eq('sec_zEbnBy')]
        need(len(cached) == 1 and acceptable(cached.iloc[0]), 'Original Grubhub predecessor control missing/rejected')
        for field in ('open', 'high', 'low', 'close', 'adj_open', 'adj_high', 'adj_low', 'adj_close'):
            need(float(control[field]) == float(cached.iloc[0][field]), 'Pinned predecessor control differs from original paired bar')
        literal = normalized.loc[normalized.trading_date.eq(day)].iloc[0]
        residual = max(abs(float(literal[field])/float(control[field])-1.) for field in ('open', 'high', 'low', 'close'))
        need(residual <= 2e-4, 'Published predecessor raw prices do not match the original holding')
        additions = [event for event in events if event['predecessor_id'] == 'sec_zEbnBy'
            and event['successor_id'] == target and event['effective_date'] == '2021-06-15']
        need(len(additions) == 1 and additions[0]['stock_ratio'] == 3.355 and additions[0]['cash_per_raw_share'] == 0.,
             'Published ADS quotes lack the separately verified 3.355 stock entitlement')
        controls_report.append({'trading_date': day, 'maximum_raw_relative_residual': residual,
            'separate_new_successor_share_class_basis': True})
    else:
        control_rows = original.loc[original.intrinio_security_id.eq(target) & original.trading_date.isin(control_dates)]
        mapping = {row['trading_date']: row for row in control_rows.to_dict('records')}
        if target_controls is not None:
            need(target_controls.intrinio_security_id.eq(target).all()
                 and set(target_controls.trading_date) == set(control_dates)
                 and len(target_controls) == len(control_dates), 'Published target-control keys changed')
            for row in target_controls.to_dict('records'):
                if row['trading_date'] in mapping: need(bar_hash(row) == bar_hash(mapping[row['trading_date']]), 'Published external control changed original target bar')
                else: mapping[row['trading_date']] = row
        for day in control_dates:
            need(day in mapping and acceptable(mapping[day]), 'Published paired target-native control missing/rejected')
            literal = expected.loc[day]
            residual = max(abs(float(literal[field])/float(mapping[day][field])-1.) for field in
                ('open', 'high', 'low', 'close', 'adj_open', 'adj_high', 'adj_low', 'adj_close'))
            need(residual <= 2e-4, 'Published raw/all target-native unit controls differ')
            controls_report.append({'trading_date': day, 'maximum_raw_and_adjusted_relative_residual': residual})
    return {'intrinio_security_id': target, 'source_native': proof['source_native'], 'method': PUBLISHED_METHOD,
        'control_coverage_policy': policy, 'repaired_rows': len(repaired), 'controls': controls_report,
        'literal_raw_and_all_pairs_independently_reconstructed': True, 'raw_all_factor_observed_not_assumed': True,
        'source_context_identity_canonically_promoted': False, 'execution_only_dated_share_class_rebind': True,
        'historical_identity_or_units_certified': False}


def verify_quote_units(root, manifest, parent, budget, original, repairs, dates):
    proofs = manifest.get('quote_unit_proofs', [])
    need(isinstance(proofs, list), 'Explicit paired quote unit proof list required')
    targets = {row['intrinio_security_id'] for row in manifest['quote_amendments']}
    need(len(proofs) == len(targets) and {proof['intrinio_security_id'] for proof in proofs} == targets,
         'Every repaired native requires exactly one paired source/unit proof')
    results = []
    def bounded_json(pin):
        path = parent.pinned(root, pin); size = path.stat().st_size
        need(budget.used+size <= budget.reserved and budget.prior+budget.used+size <= budget.limit,
             'Independent source-control JSON exceeded its prior read reservation')
        budget.used += size
        return json_file(path)
    for proof in proofs:
        parent.pinned(root, proof['identity_evidence_pin'])
        unavailable = (bounded_json(proof['after_control_unavailable_evidence_pin'])
                       if proof.get('after_control_unavailable_evidence_pin') else None)
        source_path = parent.pinned(root, proof['source_quotes_pin'])
        pf = pq.ParquetFile(source_path)
        need(source_path.is_relative_to(root/'data/derived') and source_path.stat().st_size <= 16*1024**2
             and pf.metadata.num_rows <= 10000 and set((*ACTION_FIELDS, 'original_source_native')) <= set(pf.schema_arrow.names),
             'Bounded complete paired source-control quote file required')
        meta = budget.read(source_path, ['trading_date', 'intrinio_security_id', 'original_source_native'])
        allowed_dates = set(repairs.loc[repairs.intrinio_security_id.eq(proof['intrinio_security_id']), 'trading_date']) | set(proof['control_dates'])
        need(set(meta.trading_date) == allowed_dates and meta.trading_date.isin(dates).all()
             and meta.intrinio_security_id.eq(proof['source_native']).all(), 'Paired source keys rejected before financial read')
        if proof['method'] != PUBLISHED_METHOD:
            need(meta.original_source_native.eq(proof['source_native']).all(), 'Original paired provider source identity changed')
        source = budget.read(source_path, list((*ACTION_FIELDS, 'original_source_native')))
        controls = None
        if proof.get('target_controls_pin'):
            path = parent.pinned(root, proof['target_controls_pin'])
            need(pq.ParquetFile(path).metadata.num_rows <= 10, 'Bounded external target control projection required')
            meta = budget.read(path, ['trading_date', 'intrinio_security_id'])
            need(meta.trading_date.isin(dates).all() and set(meta.trading_date) == set(proof['control_dates'])
                 and meta.intrinio_security_id.eq(proof['intrinio_security_id']).all(), 'External target control keys rejected before financial read')
            controls = budget.read(path, list(ACTION_FIELDS))
        if proof['method'] == PUBLISHED_METHOD:
            path = parent.pinned(root, proof['paired_source_pin']); pf = pq.ParquetFile(path)
            need(path.is_relative_to(root/'data/derived') and pf.metadata.num_rows <= 20000
                 and set((*PUBLISHED_META, 'open', 'high', 'low', 'close')) <= set(pf.schema_arrow.names),
                 'Bounded literal published raw/all pair source required')
            literal_meta = budget.read(path, list(PUBLISHED_META))
            need(literal_meta.trading_date.map(lambda day: isinstance(day, str)).all()
                 and literal_meta.trading_date.isin(dates).all(),
                 'Published literal file contains protected or unrelated dates before financial Arrow decoding')
            required = set(allowed_dates)
            if proof.get('control_coverage_policy') == LISTING_POLICY: required.add(proof['predecessor_control_date'])
            need(set(required) <= set(dates), 'Published literal required date outside independent execution calendar')
            mask = published_mask(literal_meta, proof, required)
            # A physical Arrow block may contain additional rows. Those rows
            # are excluded before any OHLC value reaches Python or comparison.
            numbers = budget.arrow(path, ['open', 'high', 'low', 'close']).filter(pa.array(mask.to_numpy(dtype=bool))).to_pandas()
            paired = pd.concat([literal_meta.loc[mask].reset_index(drop=True), numbers.reset_index(drop=True)], axis=1)
            predecessor = bounded_json(proof['predecessor_control_pin']) if proof.get('predecessor_control_pin') else None
            results.append(verify_published_unit_rows(proof, paired, source, repairs, original,
                target_controls=controls, predecessor=predecessor, events=manifest['additional_events']))
        else:
            results.append(verify_unit_rows(proof, source, repairs, original, target_controls=controls, unavailable=unavailable))
    return results


def verify_saved_run(root, run, *, output):
    started = time.monotonic()
    root, run, output = [Path(value).resolve() for value in (root, run, output)]
    need(run.is_relative_to(root/'outputs/ret_repair_backtest') and run != root/'outputs/ret_repair_backtest'
         and output.is_relative_to(run/'verification') and not output.exists(), 'Fresh separate repair verification output required')
    parent, v, identity, cfg, manifest, old_phase, phase, audit, dates, windows, source, handovers = authenticate(root, run)
    hashes = {name: parent.digest(run/name) for name in ('audit.json', 'preflight/audit.json', 'backtest/audit.json')}
    plan, amount = read_plan(root, run, identity, manifest, old_phase, parent)
    prior = parent.reserve_verification(run, amount)
    budget = parent.ProjectionBudget(prior=prior, reserved=amount)

    def resources():
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if platform.system() != 'Darwin': peak *= 1024
        need(peak <= cfg['resource_limits']['rss_stop_bytes']
             and time.monotonic()-started <= cfg['resource_limits']['verification_wall_seconds']
             and budget.used <= amount, 'Independent repaired-RET verification resource ceiling exceeded')
        return peak

    # The old helper is itself an independently sealed verifier, not a
    # production loader. All 18 raw/adjusted accounting fields are requested
    # so the whole before-image hash and unit proofs can be reconstructed.
    original = parent.read_quotes(SimpleNamespace(PRICE_FIELDS=list(ACTION_FIELDS)),
        root, old_phase, source, dates, budget)
    repair_rows = read_repair_quotes(root, manifest, parent, budget, dates)
    unit_reports = verify_quote_units(root, manifest, parent, budget, original, repair_rows, dates)

    def rejection(change):
        # This exact source is an explicit separately reviewed rejection, rather
        # than an implicit permission to overwrite any formerly accepted quote.
        path = parent.pinned(root, change['rejection_evidence_pin'])
        need(path.stat().st_size > 0 and any(parent.pinned(root, pin) == path for pin in manifest['source_pins']),
             'Accepted-bar rejection evidence absent from reviewed repair sources')

    prices, overlay_report = reconstruct_overlay(original, repair_rows,
        manifest['quote_amendments'], sessions=dates, authenticate_rejection=rejection)
    del original, repair_rows
    quote_index = {(row['trading_date'], row['intrinio_security_id']): row for row in prices.to_dict('records')}
    resources()
    comparison, yearly = parent.read_reports(run, budget)
    coverage = parent.guarded_table(v, budget, run/'backtest/selection_coverage.parquet')
    parent.verify_grid(comparison, yearly, coverage)
    need(audit['complete_proxy_paths'] == int(comparison.complete_proxy_path.sum())
         and audit['failed_closed_paths'] == int((~comparison.complete_proxy_path).sum()),
         'Repaired complete/withheld counts disagree with saved rows')
    selection_reports, scenario_reports = [], []
    for arm in ARMS:
        tables = {}
        for name in ('original_ledger', 'selected', 'monthly', 'rebalances'):
            pin = identity['parent_selection_artifacts'][f'selections/{arm}/{name}.parquet']
            path = parent.pinned(root, pin)
            tables[name] = parent.guarded_table(v, budget, path, maximum_rows=60000)
        need(len(tables['original_ledger']) == 43655 and len(tables['selected']) == 960
             and len(tables['rebalances']) == len(tables['monthly']) == 48,
             'Inherited original RET decisions lost rows or months')
        need(tables['monthly'].signal_date.tolist() == windows.signal_date.tolist()
             and tables['rebalances'].signal_date.tolist() == windows.signal_date.tolist(),
             'Inherited RET monthly clock changed')
        # The original score/rank/payoff arithmetic is inherited from its exact
        # PASS receipt and original immutable files, without decoding any model,
        # probability or target panel again.
        need(tables['monthly'].selected_names_per_side.eq(10).all()
             and tables['monthly'].status.eq('available').all(), 'Previously verified RET selection status changed')
        joined = coverage.loc[coverage.arm.eq(arm)].drop(columns=['arm']).reset_index(drop=True)
        parent.exact_frame(joined, tables['monthly'].reset_index(drop=True), 'New coverage report changed inherited RET breadth')
        selection_reports.append({'arm': arm, 'original_rows': 43655, 'selected_slots': 960,
            'months': 48, 'exact_original_selection_paths_and_hashes_preserved': True,
            'original_RET_score_and_payoff_PASS_inherited': True})
        for variant, fee in itertools.product(VARIANTS, FEES):
            resources()
            directory = run/f'backtest/arms/{arm}/{BASKET}/{variant}/cost_{fee:02d}bps'
            ledgers = {name: parent.guarded_table(v, budget, directory/(name+'.parquet')) for name in LEDGERS}
            monthly = parent.monthly_table(budget, directory/'monthly.parquet')
            mask = comparison.arm.eq(arm) & comparison.basket.eq(BASKET) & comparison.variant.eq(variant) & comparison.cost_bps.eq(fee)
            summary = comparison.loc[mask].iloc[0]
            annual = yearly.loc[yearly.arm.eq(arm) & yearly.basket.eq(BASKET) & yearly.variant.eq(variant) & yearly.cost_bps.eq(fee)]
            stop = None if variant == 'unstopped' else .1
            if summary.complete_proxy_path:
                checked = v.verify_scenario(ledgers, prices, tables['selected'], tables['rebalances'], dates,
                    events=phase['event_registry']['events'], historical_retirements=phase['historical_retirements'],
                    cost_bps=fee, stop_fraction=stop, prepared_quotes=quote_index, identity_handovers=handovers)
                v.verify_metrics(ledgers['daily'], ledgers['trades'], monthly, summary, annual)
                checked.update(path_status='complete_proxy', economic_performance_available=True)
            else:
                checked = v.verify_failed_scenario(ledgers, prices, tables['selected'], dates,
                    events=phase['event_registry']['events'], historical_retirements=phase['historical_retirements'],
                    cost_bps=fee, stop_fraction=stop, prepared_quotes=quote_index,
                    rebalances=tables['rebalances'], identity_handovers=handovers)
                v.verify_failed_metrics(ledgers['daily'], ledgers['trades'], monthly, summary, annual)
                checked['path_status'] = 'failed_closed'
            scenario_reports.append({'arm': arm, 'basket': BASKET, 'variant': variant, 'cost_bps': fee, **checked})
            print(json.dumps({'phase': 'independent_ret_repair_verification',
                'verified_scenarios': len(scenario_reports), 'expected_scenarios': 40}), flush=True)
    for pin in [*identity['code_pins'], *identity['source_pins']]: parent.pinned(root, pin)
    for pin in identity['parent_selection_artifacts'].values(): parent.pinned(root, pin)
    for pin in audit['artifacts'].values(): parent.pinned(run/'backtest', pin)
    need(all(parent.digest(run/name) == value for name, value in hashes.items()), 'Repaired run changed during independent verification')
    receipt = {'schema': 'dmv2-ret-repair-independent-verification/v1', 'status': 'PASS',
        'phase': 'independent_ret_repair_verification', 'build_identity_sha256': canonical(identity),
        'root_audit_sha256': hashes['audit.json'], 'preflight_audit_sha256': hashes['preflight/audit.json'],
        'backtest_audit_sha256': hashes['backtest/audit.json'], 'backtest_phase_identity_sha256': canonical(phase),
        'script_sha256': parent.digest(__file__), 'independent_equations_script_sha256': BASE_SHA,
        'original_ret_verification_pin': identity['parent_verification_pin'],
        'original_ret_identity_sha256': identity['parent_identity_sha256'],
        'repair_manifest_pin': identity['repair_manifest_pin'], 'source_pins_unchanged': True,
        'original_twenty_selection_frames_preserved': True,
        'probabilities_targets_features_or_models_decoded': False,
        'RET_scores_or_payoffs_recomputed': False,
        'models_fitted': 0, 'preprocessors_fitted': 0, 'new_inference_calls': 0,
        'production_accounting_or_quote_overlay_called': False,
        'financial_2024_plus_read': False, 'scores_and_selections_unchanged': True,
        'learning_inputs_rebuilt': False, 'potential_learning_input_impact_not_rebuilt': True,
        'source_quote_rows_after_amendment': len(prices), 'quote_overlay_checks': overlay_report,
        'paired_quote_unit_checks': unit_reports,
        'original_current_legal_event_count': len(old_phase['event_registry']['events']),
        'added_legal_event_count': len(manifest['additional_events']),
        'prior_retirements_and_PENN_exception_preserved': True,
        'sessions': 1006, 'outcome_months': 48, 'scenarios': 40,
        'complete_proxy_paths': int(comparison.complete_proxy_path.sum()),
        'withheld_paths': int((~comparison.complete_proxy_path).sum()),
        'selections': selection_reports, 'scenario_checks': scenario_reports,
        'historical_identity_knowledge_time_survivorship_or_investability_certified': False,
        'withheld_economic_performance_certified': False,
        'prior_phase_projected_compressed_bytes': prior,
        'verification_reserved_compressed_file_upper_bound_bytes': amount,
        'verification_actual_compressed_file_upper_bound_bytes': budget.used,
        'verification_read_plan': plan, 'execution_seconds': time.monotonic()-started,
        'peak_rss_bytes': resources()}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(receipt, stream, sort_keys=True, indent=2, allow_nan=False); stream.write('\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('root', 'run', 'backtest', 'output'): parser.add_argument('--'+key, required=True, type=Path)
    args = parser.parse_args()
    need(args.backtest.resolve() == args.run.resolve()/'backtest', 'Exact separate repaired backtest directory required')
    result = verify_saved_run(args.root, args.run, output=args.output)
    print(json.dumps({key: value for key, value in result.items()
        if key not in ('quote_overlay_checks', 'paired_quote_unit_checks', 'selections', 'scenario_checks', 'verification_read_plan')}))


if __name__ == '__main__': main()
