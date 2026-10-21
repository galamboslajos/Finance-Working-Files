"""Versioned execution-only repairs of the exact independently checked RET orders.

No scores, selections, labels, forecasts or models are regenerated. The parent
RET PASS authenticates the inherited decision layer; only separately reviewed
paired quote rows and legal cash/stock entitlements amend daily accounting.
"""
from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

from .action_loading import ACTION_PRICE_COLUMNS
from .backtest_accounting import DailyAccountingPolicy
from .backtest_loading import _pin, _within
from .catalog import canonical_sha256
from .daily_backtest import _save_table, summarize_daily
from .ret_backtest import ARMS, BASKET, LIMITS
from .rf_tuned_backtest import (NUMERIC_ENV, authenticated, daily_footer_plan,
    exclusive, need, read_json, resources, runtime, write_json)
from .walkforward_accounting_v2 import IMPLEMENTATION_REVISION, run_walkforward_accounting_v2
from .walkforward_backtest import _price_plan, yearly_breakdown
from .walkforward_preflight import QUOTE_METADATA_COLUMNS, preflight_selected_horizons, validate_current_events

PHASE = 'ret_repaired_continuous_backtest_v1'
NEW_FILES = ('src/dmv2/ret_repair_backtest.py', 'scripts/run_ret_repair_backtest.py',
    'scripts/verify_ret_repair_backtest.py', 'scripts/watch_ret_repair_backtest.py')
PARENT_FIELDS = ('parent_identity_pin', 'parent_audit_pin', 'parent_verification_pin',
    'parent_verifier_pin', 'independent_equations_pin')
FRAME_NAMES = ('original_ledger', 'selected', 'rebalances', 'monthly')
UNIT_METHOD = 'same_provider_dated_route_with_paired_controls'
FACTOR_METHOD = 'per_row_observed_source_adj_over_raw'
SOURCE_NATIVE_COLUMN = 'original_source_native'
PROVISIONAL_CONTROL_POLICY = 'two_pre_gap_controls_and_primary_dated_one_for_one_reorganization'
PUBLISHED_METHOD = 'published_raw_all_pair_with_dated_share_class'
LISTING_CONTROL_POLICY = 'primary_dated_listing_and_predecessor_price_control'
PUBLISHED_META = ('unified_security_id', 'trading_date', 'adjustment', 'query_symbol', 'feed',
    'timeframe', 'selected_source_product_id', 'selected_source_security_id')


def _module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


def configuration(path):
    cfg = read_json(path)
    need(cfg.get('schema') == 'dmv2-ret-repair-execution/v1'
         and cfg.get('status') == 'FROZEN_EXECUTION_AUTHORIZED'
         and cfg.get('execution_authorized') is True
         and cfg.get('resource_limits') == LIMITS
         and cfg.get('arm_ids') == list(ARMS) and cfg.get('scenarios') == 40
         and cfg.get('financial_2024_plus_authorized') is False
         and cfg.get('cloud_operations') is False
         and all(type(cfg.get(k)) is int and cfg[k] == 0 for k in
                 ('new_estimator_fits', 'new_preprocessor_fits', 'new_inference_calls'))
         and cfg.get('scores_and_selections_unchanged') is True
         and cfg.get('allow_withheld_paths') is True,
         'Explicit execution-only forty-scenario repair authority required')
    need(all(isinstance(cfg.get(k), dict) for k in (*PARENT_FIELDS, 'repair_manifest_pin')),
         'Exact parent PASS, workers and reviewed repair pins required')
    return cfg


def parent_context(root, cfg):
    paths = {name: authenticated(root, cfg[name]) for name in PARENT_FIELDS}
    parent = paths['parent_identity_pin'].parent
    need(parent.is_relative_to(root / 'outputs/ret_backtest')
         and paths['parent_identity_pin'] == parent / 'build_identity.json'
         and paths['parent_audit_pin'] == parent / 'audit.json', 'Exact original RET parent required')
    verifier = _module(paths['parent_verifier_pin'], '_ret_repair_parent_verifier')
    equations = _module(paths['independent_equations_pin'], '_ret_repair_independent_equations')
    need(cfg['independent_equations_pin']['sha256'] == verifier.BASE_SHA,
         'Original independent accounting equations changed')
    values = verifier.authenticate(root, parent, equations)
    identity, pre_phase, pre_audit, phase, audit, sessions, windows, source, handovers = values
    receipt = read_json(paths['parent_verification_pin'])
    need(receipt.get('schema') == 'dmv2-ret-backtest-independent-verification/v1'
         and receipt.get('status') == 'PASS' and receipt.get('scenarios') == 40
         and receipt.get('build_identity_sha256') == canonical_sha256(identity)
         and receipt.get('root_audit_sha256') == cfg['parent_audit_pin']['sha256']
         and receipt.get('preflight_audit_sha256') == _pin(root, parent / 'preflight/audit.json')['sha256']
         and receipt.get('backtest_audit_sha256') == _pin(root, parent / 'backtest/audit.json')['sha256']
         and receipt.get('backtest_phase_identity_sha256') == canonical_sha256(phase),
         'Exact original RET decision/accounting PASS required, including withheld paths')
    selection = {f'selections/{arm}/{name}.parquet': _pin(root, authenticated(parent / 'preflight',
        pre_audit['artifacts'][f'selections/{arm}/{name}.parquet']))
        for arm in ARMS for name in FRAME_NAMES}
    return {'directory': str(parent.relative_to(root)), 'identity': identity,
        'phase': phase, 'sessions': sessions, 'windows': windows,
        'source': source, 'handovers': handovers, 'selection_artifacts': selection}


def repair_manifest(root, cfg, parent):
    manifest = read_json(authenticated(root, cfg['repair_manifest_pin']))
    need(manifest.get('schema') == 'dmv2-ret-execution-repairs/v1'
         and manifest.get('status') == 'REVIEWED_FROZEN'
         and manifest.get('parent_ret_identity_sha256') == canonical_sha256(parent['identity'])
         and isinstance(manifest.get('source_pins'), list) and manifest['source_pins']
         and isinstance(manifest.get('additional_events'), list)
         and len(manifest['additional_events']) <= 32
         and isinstance(manifest.get('quote_amendments'), list)
         and 'quotes_pin' in manifest, 'Reviewed execution-repair manifest required')
    for pin in manifest['source_pins']: authenticated(root, pin)
    old = parent['phase']['event_registry']['events']; additions = manifest['additional_events']
    need(len(old) == 9 and len(parent['phase']['historical_retirements']) == 11,
         'Original action/retirement authority changed')
    validate_current_events([*old, *additions], parent['sessions'])
    keys = [(row['intrinio_security_id'], row['trading_date']) for row in manifest['quote_amendments']]
    need(len(keys) == len(set(keys)) <= 10000, 'Unique bounded exact quote repair keys required')
    for row in manifest['quote_amendments']:
        need(row.get('kind') in ('append', 'replace') and isinstance(row['intrinio_security_id'], str)
             and row['intrinio_security_id'].strip() and row['trading_date'] in parent['sessions']
             and isinstance(row.get('replacement_bar_sha256'), str)
             and ((row['kind'] == 'append' and row.get('original_bar_sha256') is None)
                  or (row['kind'] == 'replace' and isinstance(row.get('original_bar_sha256'), str))),
             'Quote amendment must specify exact observed append/replacement and before/after hashes')
        if row.get('rejection_evidence_pin'): authenticated(root, row['rejection_evidence_pin'])
    quote_pin = manifest.get('quotes_pin')
    need(bool(keys) == (quote_pin is not None), 'Quote amendments and quote file must correspond exactly')
    if quote_pin:
        path = authenticated(root, quote_pin); pf = pq.ParquetFile(path)
        need(path.is_relative_to(root / 'data/derived') and path.suffix == '.parquet'
             and path.stat().st_size <= 16 * 1024**2 and pf.metadata.num_rows == len(keys)
             and set(ACTION_PRICE_COLUMNS) <= set(pf.schema_arrow.names), 'Bounded complete paired quote file required')
    proofs = manifest.get('quote_unit_proofs', [])
    need(isinstance(proofs, list) and len(proofs) == len({native for native, _ in keys})
         and {proof.get('intrinio_security_id') for proof in proofs} == {native for native, _ in keys},
         'Every repaired native requires an independent dated route and paired control proof')
    for proof in proofs:
        dates = proof.get('control_dates', [])
        need(proof.get('method') in (UNIT_METHOD, PUBLISHED_METHOD) and proof.get('factor_method') == FACTOR_METHOD
             and proof.get('max_paired_residual') == proof.get('max_control_relative_residual') == 2e-4
             and isinstance(proof.get('source_native'), str) and proof['source_native'].strip()
             and isinstance(dates, list) and dates == sorted(set(dates))
             and all(day in parent['sessions'] for day in dates), 'Exact reviewed same-provider adjustment proof required')
        target_dates = [day for native, day in keys if native == proof['intrinio_security_id']]
        coverage = proof.get('control_coverage_policy', 'before_and_after')
        if coverage == LISTING_CONTROL_POLICY:
            need(proof['method'] == PUBLISHED_METHOD and not dates
                 and proof.get('predecessor_control_date') in parent['sessions']
                 and proof['predecessor_control_date'] < min(target_dates)
                 and proof.get('stock_ratio') == 3.355, 'Dated new ADS needs its exact approved predecessor/share-class bridge')
            predecessor = read_json(authenticated(root, proof['predecessor_control_pin']))
            need(predecessor.get('schema') == 'dmv2-ret-repair-predecessor-control/v1'
                 and predecessor['row']['trading_date'] == proof['predecessor_control_date']
                 and any(event['predecessor_id'] == predecessor['row']['intrinio_security_id']
                     and event.get('successor_id') == proof['intrinio_security_id']
                     and event['stock_ratio'] == proof['stock_ratio'] and event['cash_per_raw_share'] == 0
                     and event['effective_date'] == min(target_dates) for event in additions),
                 'New listing must match the pinned original predecessor and mandatory legal stock entitlement')
        elif coverage == PROVISIONAL_CONTROL_POLICY:
            need(len(dates) == 2 and dates[-1] < min(target_dates)
                 and isinstance(proof.get('after_control_unavailable_evidence_pin'), dict),
                 'Provisional one-for-one route needs two pre-gap controls and explicit missing-after evidence')
            authenticated(root, proof['after_control_unavailable_evidence_pin'])
        else:
            need(coverage == 'before_and_after' and len(dates) == 2
                 and dates[0] < min(target_dates) <= max(target_dates) < dates[1],
                 'Independent controls must bracket every repaired date')
        authenticated(root, proof['identity_evidence_pin'])
        for key in ('source_quotes_pin', 'target_controls_pin'):
            if not proof.get(key): continue
            path = authenticated(root, proof[key]); pf = pq.ParquetFile(path)
            need(path.is_relative_to(root / 'data/derived') and path.stat().st_size <= 16 * 1024**2
                 and pf.metadata.num_rows <= 10000 and set(ACTION_PRICE_COLUMNS) <= set(pf.schema_arrow.names)
                 and (key != 'source_quotes_pin' or SOURCE_NATIVE_COLUMN in pf.schema_arrow.names), 'Bounded paired source/control quotes required')
        need(proof.get('source_quotes_pin') is not None, 'Independent paired source proof file required')
        if proof['method'] == PUBLISHED_METHOD:
            source = authenticated(root, proof['paired_source_pin']); pf = pq.ParquetFile(source)
            need(source.is_relative_to(root / 'data/derived') and source.stat().st_size <= 16 * 1024**2
                 and pf.metadata.num_rows <= 10000 and set((*PUBLISHED_META, 'open', 'high', 'low', 'close')) <= set(pf.schema_arrow.names)
                 and proof.get('published_source_product_id') == 'russell_1000_2000_3000'
                 and isinstance(proof.get('published_unified_security_id'), str)
                 and isinstance(proof.get('published_routes'), list) and 0 < len(proof['published_routes']) <= 8,
                 'Literal published raw/all rows and exact dated listing route required')
    return manifest


def _bar(row):
    return {key: (None if pd.isna(row[key]) else row[key].item() if isinstance(row[key], np.generic) else row[key])
            for key in ACTION_PRICE_COLUMNS}


def _accepted(row):
    return (row.get('frequency') == 'daily'
        and isinstance(row.get('intraperiod'), (bool, np.bool_)) and not bool(row['intraperiod'])
        and row.get('price_quality_status') in ('ok', 'accepted')
        and all(isinstance(row.get(key), (bool, np.bool_)) and bool(row[key]) for key in
                ('raw_ohlc_bounds_valid', 'adjusted_ohlc_bounds_valid')))


def overlay_quotes(original, reviewed, amendments, *, metadata_only=False):
    """Apply exact keyed overrides; all other observed rows remain untouched."""
    original = original.reset_index(drop=True)
    keys = ['intrinio_security_id', 'trading_date']
    need(not original.duplicated(keys).any() and not reviewed.duplicated(keys).any(), 'Duplicate quote key')
    expected = {(row['intrinio_security_id'], row['trading_date']): row for row in amendments}
    need(len(expected) == len(amendments), 'Duplicate amendment target key')
    replacements = {(row['intrinio_security_id'], row['trading_date']): row for row in reviewed.to_dict('records')}
    need(set(expected) == set(replacements), 'Reviewed quote rows differ from the frozen exact repair request')
    output = original.copy(deep=True); positions = {tuple(row): index for index, row in enumerate(original[keys].itertuples(index=False, name=None))}
    appends = []
    for key, request in expected.items():
        replacement = replacements[key]; need(_accepted(replacement), 'Replacement quote metadata is not accepted')
        if key in positions:
            need(request['kind'] == 'replace', 'An append would overwrite an existing observed quote')
            previous = original.iloc[positions[key]].to_dict()
            need(not _accepted(previous) or bool(request.get('rejection_evidence_pin')),
                 'An accepted original bar requires explicit pinned rejection evidence')
            if not metadata_only:
                need(canonical_sha256(_bar(previous)) == request['original_bar_sha256'], 'Original quote before-image changed')
        else:
            need(request['kind'] == 'append', 'A reviewed replacement lacks its original observed key')
        if not metadata_only:
            for prefix in ('', 'adj_'):
                values = [replacement[prefix + field] for field in ('open', 'high', 'low', 'close')]
                need(all(isinstance(value, (int, float)) and not isinstance(value, bool)
                         and math.isfinite(value) and value > 0 for value in values), 'Nonpositive/nonfinite reviewed OHLC')
                opening, high, low, close = values
                need(low <= min(opening, close) <= max(opening, close) <= high, 'Invalid reviewed OHLC bounds')
            need(canonical_sha256(_bar(replacement)) == request['replacement_bar_sha256'], 'Reviewed quote after-image changed')
        if key in positions:
            for column in original.columns: output.at[positions[key], column] = replacement[column]
        else:
            appends.append({column: replacement[column] for column in original.columns})
    if appends: output = pd.concat([output, pd.DataFrame(appends)], ignore_index=True)
    need(len(output) == len(original) + len(appends) and not output.duplicated(keys).any(), 'Repair changed original quote population')
    return output


def verify_quote_units(original, reviewed, proofs, observations):
    """Require exact paired-provider copying and independently bracketed units."""
    ohlc = [prefix + field for prefix in ('', 'adj_') for field in ('open', 'high', 'low', 'close')]
    for proof in proofs:
        native, source = proof['intrinio_security_id'], proof['source_native']
        source_rows = observations[native]['source'].set_index('trading_date')
        target = reviewed.loc[reviewed.intrinio_security_id.eq(native)]
        controls = observations[native]['target_controls']
        if controls is None: controls = original.loc[original.intrinio_security_id.eq(native)
            & original.trading_date.isin(proof['control_dates'])]
        listing = proof.get('control_coverage_policy') == LISTING_CONTROL_POLICY
        need(listing or (len(controls) == 2 and set(controls.trading_date) == set(proof['control_dates'])), 'Both original target paired controls required')
        for control in controls.to_dict('records'):
            need(control['intrinio_security_id'] == native and _accepted(control), 'Target paired control is not accepted')
            cached = original.loc[original.intrinio_security_id.eq(native) & original.trading_date.eq(control['trading_date'])]
            need(cached.empty or (len(cached) == 1 and canonical_sha256(_bar(cached.iloc[0])) == canonical_sha256(_bar(control))),
                 'Supplemental target control contradicts immutable cached control')
            observed = source_rows.loc[control['trading_date']]
            for field in ohlc:
                value, baseline = observed[field], control[field]
                need(isinstance(value, (int, float, np.number)) and isinstance(baseline, (int, float, np.number))
                     and math.isfinite(value) and math.isfinite(baseline) and value > 0 and baseline > 0
                     and abs(value / baseline - 1) <= proof['max_control_relative_residual'], 'Independent raw/adjusted control mismatch')
        for observed in source_rows.to_dict('records'):
            need(_accepted(observed), 'Independent paired source quote not accepted')
            need(all(isinstance(observed[field], (int, float, np.number))
                     and not isinstance(observed[field], (bool, np.bool_))
                     and math.isfinite(observed[field]) and observed[field] > 0 for field in ohlc),
                 'Independent source OHLC must be finite and positive')
            factor = observed['adj_close'] / observed['close']
            need(math.isfinite(factor) and factor > 0, 'Invalid observed paired source adjustment factor')
            for field in ('open', 'high', 'low', 'close'):
                raw, adjusted = observed[field], observed['adj_' + field]
                need(math.isfinite(raw) and math.isfinite(adjusted) and raw > 0 and adjusted > 0
                     and abs((adjusted / raw) / factor - 1) <= proof['max_paired_residual'], 'Independent source paired OHLC units disagree')
        for after in target.to_dict('records'):
            before = source_rows.loc[after['trading_date']].to_dict()
            before.update(intrinio_security_id=native, trading_date=after['trading_date'])
            need(canonical_sha256(_bar(before)) == canonical_sha256(_bar(after)), 'Overlay differs from source paired fields beyond explicit native remap')
        if proof.get('method') == PUBLISHED_METHOD:
            literal = observations[native]['published_pairs'].set_index(['trading_date', 'adjustment'])
            for after in observations[native]['source'].to_dict('records'):
                raw, adjusted = literal.loc[(after['trading_date'], 'raw')], literal.loc[(after['trading_date'], 'all')]
                need(after[SOURCE_NATIVE_COLUMN] == raw.selected_source_security_id == adjusted.selected_source_security_id,
                     'Normalized paired row changed its literal publisher source identity')
                for field in ('open', 'high', 'low', 'close'):
                    need(after[field] == raw[field] and after['adj_' + field] == adjusted[field], 'Normalized quote is not an exact published raw/all copy')
                need(pd.isna(after['dividend']) and pd.isna(after['split_ratio'])
                     and math.isclose(after['factor'], adjusted['close'] / raw['close'], rel_tol=1e-15, abs_tol=0),
                     'Published quote invented action fields or a nonobserved adjustment factor')
            if listing:
                control = observations[native]['predecessor_control']['row']; day = proof['predecessor_control_date']
                cached = original.loc[original.intrinio_security_id.eq(control['intrinio_security_id']) & original.trading_date.eq(day)]
                need(len(cached) == 1 and _accepted(cached.iloc[0])
                     and math.isclose(control['close'], cached.iloc[0]['close'], rel_tol=2e-4, abs_tol=0)
                     and math.isclose(literal.loc[(day, 'raw'), 'close'], control['close'], rel_tol=2e-4, abs_tol=0),
                     'New dated ADS listing lacks the original predecessor raw-price control')


def published_mask(metadata, proof, dates):
    """Choose exact publisher product, dated share class and raw/all cells."""
    need(set(PUBLISHED_META) <= set(metadata) and metadata.trading_date.map(
        lambda day: isinstance(day, str) and '2020-01-01' <= day <= '2023-12-31').all(),
        'Published source dates rejected before financial conversion')
    selected = (metadata.selected_source_product_id.eq(proof['published_source_product_id'])
        & metadata.unified_security_id.eq(proof['published_unified_security_id'])
        & metadata.feed.eq('sip') & metadata.timeframe.eq('1Day') & metadata.trading_date.isin(dates))
    route = pd.Series(False, index=metadata.index)
    for item in proof['published_routes']:
        need('2020-01-01' <= item['start'] <= item['end'] <= '2023-12-31', 'Published dated route leaves authorized years')
        route |= (metadata.trading_date.between(item['start'], item['end'])
            & metadata.query_symbol.eq(item['query_symbol'])
            & metadata.selected_source_security_id.eq(item['selected_source_security_id']))
    selected &= route & metadata.adjustment.isin(('raw', 'all'))
    result = metadata.loc[selected]
    need(len(result) == 2 * len(dates) and not result.duplicated(['trading_date', 'adjustment']).any()
         and set(map(tuple, result[['trading_date', 'adjustment']].to_numpy())) == {(day, adjustment) for day in dates for adjustment in ('raw', 'all')},
         'Every exact published date requires one approved raw/all pair without aliases')
    return selected.to_numpy(dtype=bool)


def reserve_reads(run, phase, amount):
    path = run / 'read_usage.json'; state = read_json(path)
    need(type(amount) is int and amount >= 0 and state['schema'] == 'dmv2-ret-repair-read-usage/v1'
         and state['limit_bytes'] == LIMITS['cumulative_compressed_read_bytes']
         and not any(row['phase'] == phase for row in state['reservations'])
         and state['total_projected_compressed_bytes'] == sum(row['projected_compressed_bytes'] for row in state['reservations'])
         and state['total_projected_compressed_bytes'] + amount <= state['limit_bytes'], 'Repair cumulative read budget/retry guard')
    state['reservations'].append({'phase': phase, 'projected_compressed_bytes': amount})
    state['total_projected_compressed_bytes'] += amount; write_json(path, state, replace=True)


def prepare(root, *, configuration_path, output_directory):
    root = Path(root).resolve(); run = _within(root / 'outputs/ret_repair_backtest', Path(output_directory).resolve())
    need(run != root / 'outputs/ret_repair_backtest' and not run.exists(), 'Fresh separate repair output required')
    started = time.monotonic(); cfg = configuration(configuration_path); parent = parent_context(root, cfg)
    manifest = repair_manifest(root, cfg, parent)
    source_pins = [*parent['identity']['source_pins'], *parent['phase']['source_pins'],
        *(cfg[name] for name in PARENT_FIELDS), cfg['repair_manifest_pin'], *manifest['source_pins'],
        *parent['selection_artifacts'].values()]
    if manifest['quotes_pin']: source_pins.append(manifest['quotes_pin'])
    for proof in manifest.get('quote_unit_proofs', []):
        source_pins += [proof['source_quotes_pin'], proof['identity_evidence_pin']]
        if proof.get('target_controls_pin'): source_pins.append(proof['target_controls_pin'])
        if proof.get('after_control_unavailable_evidence_pin'): source_pins.append(proof['after_control_unavailable_evidence_pin'])
        for key in ('paired_source_pin', 'predecessor_control_pin'):
            if proof.get(key): source_pins.append(proof[key])
    source_pins += [row['rejection_evidence_pin'] for row in manifest['quote_amendments'] if row.get('rejection_evidence_pin')]
    identity = {'schema': 'dmv2-ret-repair-root/v1', 'project_root': str(root), 'run_root': str(run),
        'configuration': cfg, 'configuration_pin': _pin(root, configuration_path),
        **{name: cfg[name] for name in (*PARENT_FIELDS, 'repair_manifest_pin')},
        'parent_directory': parent['directory'], 'parent_identity_sha256': canonical_sha256(parent['identity']),
        'parent_selection_artifacts': parent['selection_artifacts'], 'parent_phase': parent['phase'],
        'source_reference_directory': parent['identity']['source_reference_directory'],
        'code_pins': [*parent['identity']['code_pins'], *[_pin(root, root / name) for name in NEW_FILES]],
        'source_pins': source_pins, 'runtime': runtime(), 'models_fitted': 0,
        'new_inference_calls': 0, 'scores_and_selections_unchanged': True,
        'learning_inputs_rebuilt': False, 'potential_learning_input_impact_not_rebuilt': True}
    resources(run, cfg, started, LIMITS['prepare_wall_seconds'], launch=True); run.mkdir(parents=True)
    write_json(run / 'build_identity.json', identity)
    write_json(run / 'read_usage.json', {'schema': 'dmv2-ret-repair-read-usage/v1',
        'limit_bytes': LIMITS['cumulative_compressed_read_bytes'], 'total_projected_compressed_bytes': 0, 'reservations': []})
    write_json(run / 'audit.json', {'status': 'PREPARED', 'build_identity_sha256': canonical_sha256(identity),
        'quote_values_decoded': False, 'models_fitted': 0, 'new_inference_calls': 0})
    return run


def load_run(root, run):
    root = Path(root).resolve(); run = _within(root / 'outputs/ret_repair_backtest', Path(run).resolve())
    identity, audit = read_json(run / 'build_identity.json'), read_json(run / 'audit.json')
    need(identity['schema'] == 'dmv2-ret-repair-root/v1' and identity['project_root'] == str(root)
         and identity['run_root'] == str(run) and canonical_sha256(identity) == audit['build_identity_sha256']
         and identity['runtime'] == runtime(), 'Frozen repair run/runtime changed')
    cfg = configuration(authenticated(root, identity['configuration_pin']))
    need(cfg == identity['configuration'] and all(os.environ.get(name) == '1' for name in NUMERIC_ENV), 'Repair config/thread contract changed')
    for item in (*identity['source_pins'], *identity['code_pins']): authenticated(root, item)
    parent = parent_context(root, cfg); manifest = repair_manifest(root, cfg, parent)
    need(parent['selection_artifacts'] == identity['parent_selection_artifacts']
         and parent['phase'] == identity['parent_phase'], 'Inherited RET decisions or execution authority changed')
    return identity, audit, cfg, parent, manifest


def _read_inputs(root, run, identity, parent, manifest, *, financial):
    plan, metadata_bytes, financial_bytes = daily_footer_plan(root, parent['identity'])
    selections = identity['parent_selection_artifacts']
    selection_bytes = sum(2 * authenticated(root, pin).stat().st_size for pin in selections.values())
    quote_path = authenticated(root, manifest['quotes_pin']) if manifest['quotes_pin'] else None
    amount = metadata_bytes + (financial_bytes if financial else 0) + selection_bytes
    amount += (2 if financial else 1) * quote_path.stat().st_size if quote_path else 0
    for proof in manifest.get('quote_unit_proofs', []):
        amount += (2 if financial else 1) * sum(authenticated(root, proof[key]).stat().st_size
            for key in ('source_quotes_pin', 'target_controls_pin', 'paired_source_pin') if proof.get(key))
    reserve_reads(run, 'backtest' if financial else 'preflight', amount)
    _, metadata, _ = _price_plan(root, parent['source'],
        parent['identity']['execution_reference_root_identity_sha256'], list(parent['phase']['source_pins']))
    quote_columns = [name for name in QUOTE_METADATA_COLUMNS if name in ACTION_PRICE_COLUMNS]
    reviewed_meta = (pq.ParquetFile(quote_path).read(columns=quote_columns).to_pandas()
        if quote_path else metadata.iloc[:0].copy())
    need(reviewed_meta.trading_date.isin(parent['sessions']).all(), 'Repair quotes leave approved sessions')
    amended_metadata = overlay_quotes(metadata, reviewed_meta, manifest['quote_amendments'], metadata_only=True)
    observations = {}
    for proof in manifest.get('quote_unit_proofs', []):
        native, source = proof['intrinio_security_id'], proof['source_native']
        target_dates = set(reviewed_meta.loc[reviewed_meta.intrinio_security_id.eq(native), 'trading_date'])
        observations[native] = {'source': None, 'target_controls': None}
        for key, name in (('source_quotes_pin', 'source'), ('target_controls_pin', 'target_controls')):
            if not proof.get(key): continue
            path = authenticated(root, proof[key]); pf = pq.ParquetFile(path)
            columns = ['trading_date', 'intrinio_security_id'] + ([SOURCE_NATIVE_COLUMN] if name == 'source' else [])
            keys = pf.read(columns=columns).to_pandas()
            need(not keys.duplicated(['trading_date', 'intrinio_security_id']).any()
                 and keys.trading_date.isin(parent['sessions']).all()
                 and keys.intrinio_security_id.eq(source if name == 'source' else native).all()
                 and set(keys.trading_date) == (target_dates | set(proof['control_dates']) if name == 'source' else set(proof['control_dates']))
                 and (name != 'source' or proof['method'] == PUBLISHED_METHOD or keys[SOURCE_NATIVE_COLUMN].eq(source).all()), 'Paired proof metadata leaves exact approved controls/targets')
            if financial: observations[native][name] = pf.read(columns=list(ACTION_PRICE_COLUMNS) + ([SOURCE_NATIVE_COLUMN] if name == 'source' else [])).to_pandas()
        if proof['method'] == PUBLISHED_METHOD:
            path = authenticated(root, proof['paired_source_pin']); pf = pq.ParquetFile(path)
            literal_meta = pf.read(columns=list(PUBLISHED_META)).to_pandas()
            needed = target_dates | set(proof['control_dates'])
            if proof.get('control_coverage_policy') == LISTING_CONTROL_POLICY:
                needed.add(proof['predecessor_control_date'])
                observations[native]['predecessor_control'] = read_json(authenticated(root, proof['predecessor_control_pin']))
            mask = published_mask(literal_meta, proof, needed)
            if financial: observations[native]['published_pairs'] = pf.read(columns=[*PUBLISHED_META, 'open', 'high', 'low', 'close']).filter(pa.array(mask)).to_pandas()
    tables = {}
    for arm in ARMS:
        tables[arm] = {}
        for name in FRAME_NAMES:
            path = authenticated(root, selections[f'selections/{arm}/{name}.parquet']); pf = pq.ParquetFile(path)
            date_columns = [column for column in ('signal_date', 'execution_date') if column in pf.schema_arrow.names]
            keys = pf.read(columns=date_columns).to_pandas()
            need(keys.signal_date.isin(parent['windows'].signal_date).all()
                 and ('execution_date' not in keys or keys.execution_date.isin(parent['sessions']).all()), 'Inherited selection dates changed')
            tables[arm][name] = pf.read().to_pandas()
    prices = None
    if financial:
        prices = pd.concat([pq.ParquetFile(item['path']).read(columns=list(ACTION_PRICE_COLUMNS)).to_pandas() for item in plan], ignore_index=True)
        need(prices[['trading_date', 'intrinio_security_id']].equals(metadata[['trading_date', 'intrinio_security_id']]), 'Base daily keys changed after guard')
        reviewed = pq.ParquetFile(quote_path).read(columns=list(ACTION_PRICE_COLUMNS)).to_pandas() if quote_path else prices.iloc[:0].copy()
        verify_quote_units(prices, reviewed, manifest.get('quote_unit_proofs', []), observations)
        prices = overlay_quotes(prices, reviewed, manifest['quote_amendments'])
    return amended_metadata, prices, tables, amount


def preflight(root, run_directory):
    root, run = Path(root).resolve(), Path(run_directory).resolve()
    identity, audit, cfg, parent, manifest = load_run(root, run); directory = run / 'preflight'
    need(audit['status'] == 'PREPARED' and not directory.exists(), 'Preserve prior repair preflight attempts')
    started = time.monotonic()
    with exclusive(run), threadpool_limits(limits=1):
        resources(run, cfg, started, LIMITS['preflight_wall_seconds'], launch=True); directory.mkdir()
        metadata, _, tables, amount = _read_inputs(root, run, identity, parent, manifest, financial=False)
        registry = deepcopy(parent['phase']['event_registry']); registry['events'] += deepcopy(manifest['additional_events'])
        summaries, artifacts = [], {}
        for arm, values in tables.items():
            result = preflight_selected_horizons(values['selected'], values['rebalances'], metadata, parent['sessions'],
                events=registry['events'], historical_retirements=parent['phase']['historical_retirements'])
            summaries.append({'arm': arm, 'basket': BASKET, **result['summary'],
                'withheld_selection_months': int(values['monthly'].status.ne('available').sum())})
            for name, frame in result.items():
                if name != 'summary': _save_table(directory, f'coverage/{arm}/{name}.parquet', frame, artifacts)
        phase = {**deepcopy(parent['phase']), 'schema': 'dmv2-ret-repair-preflight/v1', 'phase': PHASE,
            'parent_identity_sha256': canonical_sha256(identity), 'original_ret_identity_sha256': identity['parent_identity_sha256'],
            'parent_selection_artifacts': identity['parent_selection_artifacts'], 'repair_manifest_pin': identity['repair_manifest_pin'],
            'source_pins': identity['source_pins'], 'event_registry': registry, 'performance_computed': False,
            'read_reserved_compressed_bytes': amount, 'scores_and_selections_unchanged': True}
        write_json(directory / 'build_identity.json', phase)
        _save_table(directory, 'summary.parquet', pd.DataFrame(summaries), artifacts)
        write_json(directory / 'audit.json', {'status': 'COMPLETE', 'build_identity_sha256': canonical_sha256(identity),
            'phase_identity_sha256': canonical_sha256(phase), 'artifacts': artifacts, 'summary': summaries, 'performance_computed': False})
        resources(run, cfg, started, LIMITS['preflight_wall_seconds'])
    return directory


def backtest(root, run_directory, *, allow_withheld_paths=False):
    root, run = Path(root).resolve(), Path(run_directory).resolve()
    identity, audit, cfg, parent, manifest = load_run(root, run); directory = run / 'backtest'
    pre = read_json(run / 'preflight/audit.json'); phase = read_json(run / 'preflight/build_identity.json')
    need(not directory.exists() and type(allow_withheld_paths) is bool and pre['status'] == 'COMPLETE'
         and pre['build_identity_sha256'] == canonical_sha256(identity)
         and pre['phase_identity_sha256'] == canonical_sha256(phase), 'Completed exact new repair preflight required')
    need(all(row['ready_for_accounting'] and not row['withheld_selection_months'] for row in pre['summary']) or allow_withheld_paths,
         'Unresolved repaired-path coverage requires explicit fail-closed authorization')
    for pin in pre['artifacts'].values(): authenticated(run / 'preflight', pin)
    started = time.monotonic()
    with exclusive(run), threadpool_limits(limits=1):
        resources(run, cfg, started, LIMITS['backtest_wall_seconds'], launch=True); directory.mkdir()
        _, prices, tables, amount = _read_inputs(root, run, identity, parent, manifest, financial=True)
        phase = {**phase, 'schema': 'dmv2-ret-repair-phase/v1', 'performance_computed': True,
            'preflight_audit_pin': _pin(root, run / 'preflight/audit.json'), 'code_pins': identity['code_pins'],
            'runtime': identity['runtime'], 'read_reserved_compressed_bytes': amount}
        write_json(directory / 'build_identity.json', phase); artifacts = {'build_identity.json': _pin(directory, directory / 'build_identity.json')}
        comparison, yearly, coverage = [], [], []
        for arm, values in tables.items():
            coverage.append(values['monthly'].assign(arm=arm, basket=BASKET))
            for variant, stop in (('unstopped', None), ('asset_stop_10pct', .1)):
                for cost in (0, 5, 10, 25):
                    resources(run, cfg, started, LIMITS['backtest_wall_seconds'])
                    result = run_walkforward_accounting_v2(values['selected'], prices, parent['sessions'],
                        rebalances=values['rebalances'], terminal_date='2023-12-29', events=phase['event_registry']['events'],
                        historical_retirements=phase['historical_retirements'], identity_handovers=parent['handovers'],
                        policy=DailyAccountingPolicy(cost_bps=cost, borrow_annual_bps=0, required_names_per_side=None, asset_stop_fraction=stop))
                    row, monthly = summarize_daily(result, arm=arm, basket=BASKET, variant=variant, cost_bps=cost, expected_sessions=1006)
                    row.update(outcome_start='2020-01-01', outcome_end='2023-12-31', outcome_months=48)
                    comparison.append(row); yearly.append(yearly_breakdown(result, arm=arm, basket=BASKET, variant=variant, cost_bps=cost))
                    relative = f'arms/{arm}/{BASKET}/{variant}/cost_{cost:02d}bps'
                    for name, frame in result.tables.items(): _save_table(directory, relative + '/' + name + '.parquet', frame, artifacts)
                    _save_table(directory, relative + '/monthly.parquet', monthly, artifacts)
                    print(json.dumps({'phase': PHASE, 'completed_scenarios': len(comparison), 'expected_scenarios': 40}), flush=True)
        need(len(comparison) == 40, 'Exact forty repaired RET scenarios required')
        for name, frame in {'comparison': pd.DataFrame(comparison), 'yearly': pd.concat(yearly, ignore_index=True),
                'selection_coverage': pd.concat(coverage, ignore_index=True)}.items(): _save_table(directory, name + '.parquet', frame, artifacts)
        for pin in (*identity['source_pins'], *identity['code_pins']): authenticated(root, pin)
        resources(run, cfg, started, LIMITS['backtest_wall_seconds'])
        write_json(directory / 'audit.json', {'status': 'COMPLETE', 'phase': PHASE, 'implementation_revision': IMPLEMENTATION_REVISION,
            'build_identity_sha256': canonical_sha256(identity), 'phase_identity_sha256': canonical_sha256(phase),
            'artifacts': artifacts, 'scenarios': 40, 'sessions': 1006, 'outcome_months': 48,
            'complete_proxy_paths': sum(row['complete_proxy_path'] for row in comparison),
            'failed_closed_paths': sum(not row['complete_proxy_path'] for row in comparison),
            'scores_and_selections_unchanged': True, 'old_runs_and_pointers_changed': False, 'models_fitted': 0,
            'new_inference_calls': 0, 'financial_2024_plus_read': False, 'execution_seconds': time.monotonic() - started})
        write_json(run / 'audit.json', {**audit, 'status': 'COMPLETE_PENDING_INDEPENDENT_VERIFICATION',
            'backtest_audit_pin': _pin(root, directory / 'audit.json')}, replace=True)
    return directory
