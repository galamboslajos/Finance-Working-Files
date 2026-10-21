#!/usr/bin/env python3
"""Catalog-bound, additive20-fit annual model refit; no RF search.

Planning inspects JSON and Parquet footers only. Each numerical action requires
its own immutable plan, exact shared-ledger reservations and an external sealed
resource-guard launch. Imports do not read panels, load estimators or fit.
"""
import argparse
import base64
from contextlib import contextmanager
from dataclasses import replace
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

YEARS = (2020, 2021, 2022, 2023)
ARMS = {'rf_paper15': ('rf', 'paper15'), 'xgb_paper15': ('xgb', 'paper15'),
    'rf_extended': ('rf', 'expanded'), 'xgb_extended': ('xgb', 'expanded'),
    'rf_tuned_extended': ('rf', 'expanded')}
WINNERS = {2020: (12, 20), 2021: (8, 20), 2022: (12, 100), 2023: (12, 100)}
KEYS = ('source_security_id', 'signal_date')
CATEGORIES = ('compressed_source', 'derived_matrix')
HELPERS = ('classification.py', 'features.py', 'preprocessing.py', 'paper_features.py',
    'targets.py', 'ret.py', 'classifier_pilot.py', 'pilot_models.py',
    'validation_reprediction.py', 'walkforward_metrics.py', 'catalog.py', 'universe.py',
    'daily_features.py', 'context_features.py', 'identity_policy.py', 'monthly.py',
    'data.py', 'pilot_loading.py', 'full_panel.py', 'full_target_panel.py')
HASH_IO_BYTES = 0
META_COVERAGE = (*KEYS, 'intrinio_security_id', 'identity_status', 'source_identity_status',
    'identity_eligible', 'identity_eligibility_policy', 'identity_eligibility_reason',
    'identity_review_required', 'resolution_status', 'paper_prediction_eligible',
    'requested', 'knowledge_time_verified', 'accounting_certified',
    'available_feature_count', 'registered_feature_count', 'feature_status', 'model_ready')
ORIGINAL_METADATA = (*KEYS, 'as_of', 'indices', 'membership_tickers', 'unified_security_id',
    'intrinio_security_id', 'identity_status', 'bridge_method', 'bridge_confidence',
    'identity_review_required', 'resolution_status', 'source_identity_status',
    'identity_eligible', 'identity_eligibility_policy', 'identity_eligibility_reason',
    'original_intrinio_security_id', 'original_identity_status', 'original_resolution_status',
    'original_identity_eligible', 'reviewed_security_isin', 'identity_amendment_id',
    'identity_amendment_sha256', 'replacement_native_status', 'el_paso_route_amendment_id',
    'el_paso_route_amendment_sha256', 'el_paso_route_original_native_id',
    'el_paso_route_replacement_native_id', 'el_paso_route_reviewed_cusip',
    'el_paso_route_lookup_identifier', 'el_paso_route_historical_identity_certified',
    'shadow_original_intrinio_security_id', 'shadow_original_identity_status',
    'shadow_original_identity_review_required', 'shadow_original_resolution_status',
    'shadow_original_identity_eligible', 'shadow_route_policy', 'computational_price_key',
    'price_route_provider', 'price_route_source_key', 'provider_intrinio_security_id',
    'price_route_status', 'shadow_canonical_route_promoted')


def need(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    global HASH_IO_BYTES
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 ** 2), b''):
            HASH_IO_BYTES += len(block)
            h.update(block)
    return h.hexdigest()


def pin(path):
    p = Path(path).resolve()
    need(p.is_file(), 'Pinned regular file required')
    return dict(path=str(p), size_bytes=p.stat().st_size, sha256=digest(p))


def check(item):
    p = Path(item['path'])
    need(p.is_absolute() and p.is_file() and p.stat().st_size == item['size_bytes']
         and digest(p) == item['sha256'], 'Changed pinned bytes: ' + str(p))
    return p


def read(path):
    return json.loads(Path(path).read_text())


def canonical(value):
    return 'sha256:' + hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode()).hexdigest()


def save(path, value):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False); f.write('\n')
        f.flush(); os.fsync(f.fileno())
    return pin(p)


def checked_read(item):
    return read(check(item))


def config(root, protocol_path):
    root = Path(root).resolve(); p = read(protocol_path)
    need(p['schema'] == 'dmv2-corrected-refit-protocol/v1'
         and p['status'] == 'SEALED_BEFORE_FINANCIAL_READ', 'Unsealed refit protocol')
    need(p['parameter_search'] is False and p['protected_2024_plus'] is True,
         'Search/protected-scope flags changed')
    need(p['outcome_years'] == list(YEARS), 'Exactly four annual outcome windows required')
    bind(p['worker_pin'], Path(__file__))
    for name in HELPERS:
        bind(p['helper_pins'][name], root / 'src/dmv2' / name)
    limits = p['limits']
    need(limits['max_fits'] == 20 and limits['max_preprocessors'] == 8,
         'Exactly20 estimator fits/eight preprocessing states maximum')
    need(limits['per_fit_seconds'] <= 1200
         and limits['rss_warning_bytes'] == 16 * 1024 ** 3
         and limits['rss_stop_bytes'] == 20 * 1024 ** 3
         and limits['output_maximum_bytes'] == 20 * 1024 ** 3, 'Frozen execution ceilings changed')
    for package, expected in p['runtime'].items():
        need(importlib.metadata.version(package) == expected, 'Frozen runtime changed: ' + package)
    bind(p['baseline_recipe_pin'], root / 'configs/classifier_pilot.json')
    recipe = checked_read(p['baseline_recipe_pin'])
    need([a['arm_id'] for a in recipe['arms']] == list(ARMS)[:4], 'Four baseline arms changed')
    for package, expected in recipe['packages'].items():
        need(importlib.metadata.version(package) == expected, 'Baseline package changed: ' + package)
    catalog = checked_read(p['catalog_pin'])
    need(catalog['schema'] == p['catalog_schema'] and catalog['status'] == p['catalog_status'],
         'Input catalog state differs from frozen practical version')
    proof = checked_read(p['input_verification_pin'])
    need(proof['status'] == 'PASS', 'Independent repaired-input verification required')
    need(all(catalog['independent_verification'][k] == p['input_verification_pin'][k]
         for k in ('path', 'size_bytes', 'sha256')), 'Catalog and independent input receipt are not bound')
    guard = checked_read(p['terminal_guard_pin'])
    validate_terminal_guard(guard)
    tuned = {int(year): checked_read(item) for year, item in p['tuned_fit_manifest_pins'].items()}
    need(set(tuned) == set(YEARS), 'Four frozen annual tuned settings required')
    for year, manifest in tuned.items():
        tuned_parameters(year, manifest)
    return p, recipe, catalog, tuned


def bind(item, path):
    need(Path(item['path']).resolve() == Path(path).resolve(), 'Pin redirects expected file')
    return check(item)


def validate_terminal_guard(guard):
    need(guard['status'] == 'PASS' and guard['events_reviewed'] == 29
         and guard['target_definition_unchanged'] is True
         and guard['missing_terminal_targets_imputed'] is False,
         'Full29-event metadata review with unchanged/missing terminal labels required')


def tuned_parameters(year, manifest):
    need(manifest['status'] == 'COMPLETE' and manifest['stage'] == 'outer'
         and manifest['outer_year'] == year, 'Tuned manifest is not its fixed outer winner')
    params = manifest['explicit_parameters']
    need((params['max_depth'], params['min_samples_leaf']) == WINNERS[year]
         and params['max_features'] == 'sqrt' and params['n_estimators'] == 500
         and params['n_jobs'] == 4 and params['random_state'] == 1729
         and params['class_weight'] is None and params['bootstrap'] is True
         and params['warm_start'] is False and params['oob_score'] is False,
         'Frozen annual RF winner/settings changed')
    need(manifest['fit_effective_parameters'] == params
         and manifest['saved_inference_effective_parameters'] == {**params, 'n_jobs': 1},
         'Tuned statistical parameters or inference convention changed')
    return dict(params)


def year_window(year):
    need(type(year) is int and year in YEARS, 'Unapproved outcome year')
    return dict(train_start=f'{year-10}-01-01', train_end=f'{year-1}-12-31',
        prediction_start=f'{year}-01-01', prediction_end=f'{year}-12-31')


def keys(rows):
    result = []
    for row in rows:
        key = tuple(row[k] for k in KEYS)
        need(all(isinstance(x, str) and bool(x.strip()) for x in key), 'Missing original key')
        result.append(key)
    need(len(set(result)) == len(result), 'Duplicate original source/signal key')
    return result


def validate_metadata(year, feature_rows, coverage, original, targets, windows):
    """Whole original annual cohort/clock check before financial Python values."""
    original_keys = set(keys(original))
    for table in (feature_rows, coverage, targets):
        need(set(keys(table)) == original_keys, 'Original cohort denominator changed')
    need(len(windows) == 12 and {w['outcome_month'] for w in windows.values()} ==
         {f'{year}-{m:02d}' for m in range(1, 13)}, 'Exact12 next-month windows required')
    need({r['signal_date'] for r in feature_rows} == set(windows), 'Monthly signals differ')
    for signal, w in windows.items():
        from datetime import date
        first, start, end = map(date.fromisoformat, (signal, w['outcome_start'], w['outcome_end']))
        following = first.year * 12 + first.month
        need((start.year * 12 + start.month - 1) == following
             and start.year == end.year == year and start.month == end.month
             and first < start <= end and end.isoformat() < '2024-01-01', 'Future/overlapping target clock')
    by_coverage = {tuple(r[k] for k in KEYS): r for r in coverage}
    for row in targets:
        need(all(row[k] == windows[row['signal_date']][k] for k in
            ('outcome_month', 'outcome_start', 'outcome_end')), 'Target clock differs from independent windows')
        cf = by_coverage[tuple(row[k] for k in KEYS)]
        for name in ('identity_eligible', 'requested', 'paper_prediction_eligible'):
            need(type(row[name]) is bool and type(cf[name]) is bool and row[name] == cf[name], 'Signal admission changed')
        need(type(row['prediction_eligible']) is bool and type(row['target_available']) is bool,
             'Explicit prediction/target masks required')
        need(row['prediction_eligible'] == cf['paper_prediction_eligible']
             and (not row['target_available'] or row['prediction_eligible'])
             and (not cf['paper_prediction_eligible'] or cf['identity_eligible'] and cf['requested']),
             'Future target availability changed prediction population')
    return original_keys


def projection(item, columns):
    import pyarrow.parquet as pq
    path = check(item); pf = pq.ParquetFile(path)
    need(len(columns) == len(set(columns)) and set(columns) <= set(pf.schema_arrow.names), 'Projection/schema mismatch')
    leaves = [pf.metadata.row_group(g).column(c) for g in range(pf.metadata.num_row_groups)
        for c in range(pf.metadata.row_group(g).num_columns)]
    roots = {leaf.path_in_schema.split('.')[0] for leaf in leaves}
    need(set(columns) <= roots, 'Requested Parquet column lacks physical chunks')
    compressed = sum(leaf.total_compressed_size for leaf in leaves
        if leaf.path_in_schema.split('.')[0] in columns)
    return dict(pin=item, columns=list(columns), rows=pf.metadata.num_rows, projected_compressed_bytes=compressed)


def plan(root, protocol_path, run):
    """No table reads: exact JSON/Parquet-footer scope before numerical actions."""
    root, run = Path(root).resolve(), Path(run).resolve()
    p, recipe, catalog, tuned = config(root, protocol_path)
    need(run == Path(p['run_directory']).resolve() and not run.exists(), 'Fresh exact run directory required')
    sys.path.insert(0, str(root / 'src'))
    from dmv2.features import registered_feature_specs
    from dmv2.targets import SAFE_METADATA
    TARGET_METADATA = (*SAFE_METADATA, 'target_available')
    from dmv2.paper_features import PAPER_FEATURE_NAMES
    specs = registered_feature_specs(); names = [x['name'] for x in specs]
    predictors = [x['name'] for x in specs if x.get('model_eligible', True)]
    need(len(names) == 728 and len(predictors) == 715, 'Frozen728/715 dictionary required')
    annuals = {}
    for record in catalog['annuals']:
        year = record['year']; need(type(year) is int and 2010 <= year <= 2023 and str(year) not in annuals, 'Annual catalog scope/duplicate')
        a = record['artifacts']; state = checked_read(a['target_state'])
        windows = {r['signal_date']: {k: r[k] for k in ('outcome_month', 'outcome_start', 'outcome_end')}
            for r in state['provenance']['month_windows']}
        need(len(windows) == 12 and state['provenance']['label_observed_through'] <= f'{year}-12-31', 'Annual state scope')
        import pyarrow.parquet as pq
        original_fields = set(pq.ParquetFile(check(a['original_cohort'])).schema_arrow.names)
        metadata = {'feature_keys': projection(a['features'], KEYS),
            'coverage': projection(a['feature_coverage'], META_COVERAGE),
            'original': projection(a['original_cohort'], [n for n in ORIGINAL_METADATA if n in original_fields]),
            'target_metadata': projection(a['targets'], TARGET_METADATA),
            'diagnostic_metadata': projection(a['target_diagnostics'], ('signal_date', 'outcome_month'))}
        need(set(pq.ParquetFile(check(a['features'])).schema_arrow.names) == set([*KEYS, *names]),
             'Feature file changes the registered728-only schema')
        numeric = {'features': projection(a['features'], [*KEYS, *names]),
            'targets': projection(a['targets'], pq.ParquetFile(check(a['targets'])).schema_arrow.names),
            'target_diagnostics': projection(a['target_diagnostics'], pq.ParquetFile(check(a['target_diagnostics'])).schema_arrow.names)}
        annuals[str(year)] = dict(year=year, metadata=metadata, numeric=numeric,
            target_state_pin=a['target_state'], windows=windows,
            target_state_sha256=a['target_state']['sha256'], catalog_audit_pin=record.get('audit'))
    need(set(annuals) == {str(y) for y in range(2010, 2024)}, 'Complete fourteen-year catalog required')
    need(sum(a['metadata']['original']['rows'] for a in annuals.values()) == 152307,
         'Original152307-row historical cohort envelope changed')
    folds = {}
    for year in YEARS:
        ys = [str(y) for y in range(year - 10, year + 1)]
        import exchange_calendars as xcals
        completed = xcals.get_calendar('XNYS', start=f'{year-1}-01-01', end=f'{year-1}-12-31').sessions[-1].strftime('%Y-%m-%d')
        need(max(w['outcome_end'] for y in ys[:-1] for w in annuals[y]['windows'].values()) == completed,
             'Ten-year labels do not end at independent completed XNYS cutoff')
        cost = sum(v['projected_compressed_bytes'] for y in ys for group in ('metadata', 'numeric')
            for v in annuals[y][group].values())
        folds[str(year)] = dict(**year_window(year), annual_years=ys, label_observed_through=completed,
            projected_reads={'compressed_source': cost, 'derived_matrix': 0})
    body = dict(schema='dmv2-corrected-refit-source-plan/v1', status='SEALED_BEFORE_FINANCIAL_READ',
        root=str(root), run=str(run), protocol_pin=pin(protocol_path), worker_pin=pin(__file__),
        catalog_pin=p['catalog_pin'], annuals=annuals, folds=folds,
        registered_specs=specs, predictor_names=predictors, paper_names=list(PAPER_FEATURE_NAMES),
        baseline_recipe_pin=p['baseline_recipe_pin'], tuned_fit_manifest_pins=p['tuned_fit_manifest_pins'],
        expected_estimator_fits=20, expected_preprocessor_fits=8,
        source_read_projection_sum=sum(x['projected_reads']['compressed_source'] for x in folds.values()),
        feature_and_outcome_metadata_guard_before_values=True,
        block_decode_disclosure='Selected compressed columns may transiently decode Arrow blocks; only approved annual keys/windows reach Python financial values.')
    run.mkdir(parents=True)
    save(run / 'build_identity.json', dict(schema='dmv2-corrected-refit-identity/v1', protocol_pin=pin(protocol_path),
        catalog_pin=p['catalog_pin'], source_plan_sha256=canonical(body), worker_pin=pin(__file__),
        parameter_search=False, fitted_reference_models_reused=False, old_runs_unchanged=True))
    return save(run / 'read_plan.json', body)


def runtime(root, protocol_path, run):
    p, recipe, catalog, tuned = config(root, protocol_path)
    run = Path(run).resolve(); source = read(run / 'read_plan.json')
    need(run == Path(p['run_directory']).resolve() and source['protocol_pin'] == pin(protocol_path)
         and source['worker_pin'] == pin(__file__) and source['catalog_pin'] == p['catalog_pin'], 'Frozen run plan changed')
    need(read(run / 'build_identity.json')['source_plan_sha256'] == canonical(source), 'Source plan identity changed')
    return p, recipe, tuned, source


def table_pin(path):
    return projection(pin(path), __import__('pyarrow.parquet', fromlist=['ParquetFile']).ParquetFile(path).schema_arrow.names)


def action_plan(root, protocol_path, run, action, year=None, arm=None, *, write=True):
    p, recipe, tuned, source = runtime(root, protocol_path, run); run = Path(run).resolve()
    need(action in ('prepare', 'fit', 'predict', 'finalize'), 'Unknown numerical action')
    if action != 'finalize': year_window(year)
    if action in ('fit', 'predict'): need(arm in ARMS, 'Unknown fixed arm')
    fold = run / 'folds' / str(year); files = {}; reads = dict.fromkeys(CATEGORIES, 0)
    if action == 'prepare':
        reads = source['folds'][str(year)]['projected_reads']
    elif action in ('fit', 'predict'):
        prepared = read(fold / 'DATA_COMPLETE.json'); validate_prepared(prepared, source, year)
        bank = ARMS[arm][1]
        needed = [f'preprocessors/{bank}/train_X.npy', 'data/train_y.npy'] if action == 'fit' else [f'preprocessors/{bank}/test_X.npy']
        for name in needed:
            files[name] = prepared['artifacts'][name]; check(files[name]); reads['derived_matrix'] += files[name]['size_bytes']
        if action == 'predict':
            fit_pin = pin(fold / 'arms' / arm / 'FIT_COMPLETE.json'); fitted = checked_read(fit_pin)
            files['model'] = fitted['model']; files['fit_receipt'] = fit_pin
            reads['derived_matrix'] += fitted['model']['size_bytes']
            spec = table_pin(fold / 'data/test_ledger.parquet'); files['test_ledger'] = spec
            reads['compressed_source'] += spec['projected_compressed_bytes']
    else:
        for y in YEARS:
            for a in ARMS:
                path = run / 'folds' / str(y) / 'arms' / a / 'PREDICT_COMPLETE.json'
                receipt = checked_read(pin(path)); spec = table_pin(check(receipt['artifacts']['predictions']))
                files[f'{y}/{a}'] = spec; reads['compressed_source'] += spec['projected_compressed_bytes']
    result = dict(schema='dmv2-corrected-refit-action-plan/v1', status='SEALED_BEFORE_ACTION_READ', action=action,
        year=year, arm=arm, source_plan_pin=pin(run / 'read_plan.json'), protocol_pin=pin(protocol_path),
        worker_pin=pin(__file__), files=files, projected_reads=reads)
    label = action + (f'_{year}' if year is not None else '') + (f'_{arm}' if arm else '')
    return save(run / 'plans' / (label + '.json'), result) if write else result


def validate_prepared(prepared, source, year):
    need(prepared['status'] == 'COMPLETE' and prepared['year'] == year
         and prepared['source_plan_sha256'] == canonical(source)
         and set(prepared['schemas']) == {'paper15', 'expanded'}, 'Prepared annual identity changed')
    for item in prepared['artifacts'].values(): check(item)


def require_reservations(run, plan_path, plan, labels):
    from control import ROOT, require_reservation
    need(Path(run).resolve().is_relative_to(ROOT / 'run'), 'Model artifacts must use the shared rerun namespace')
    need(set(labels) == {k for k, amount in plan['projected_reads'].items() if amount}, 'One exact reservation per nonzero read category required')
    for category, amount in plan['projected_reads'].items():
        if not amount: continue
        require_reservation(ROOT, labels[category], amount, pin(plan_path), category)


def require_launch(launch_path, launch_sha, plan_path, plan, protocol, protocol_path, command=None):
    need(launch_path, 'Sealed external resource-guard launch required')
    need(Path(os.environ.get('DMV2_GUARD_LAUNCH_PATH', '/missing')).resolve() == Path(launch_path).resolve()
         and os.environ.get('DMV2_GUARD_LAUNCH_SHA256') == digest(launch_path),
         'Authenticated external guard environment required')
    if launch_sha: need(digest(launch_path) == launch_sha, 'Launch changed')
    launch = read(launch_path)
    actual = [protocol['python_path'], *sys.argv] if command is None else command
    need(launch['schema'] == 'dmv2-corrected-refit-phase-launch/v1'
         and launch['protocol_pin'] == pin(protocol_path)
         and launch['command'] == actual
         and any(item == pin(__file__) for item in launch['code_pins'])
         and launch['maximum_seconds'] > 0
         and launch['time_category'] in protocol['limits']['active_seconds_by_category'],
         'External resource launch differs from exact worker/action/budget')
    for flag, value in (('--action-plan', str(plan_path)), ('--action-plan-sha256', digest(plan_path))):
        need(actual.count(flag) == 1 and actual[actual.index(flag)+1] == value,
             'Guard command does not bind the exact action plan')


@contextmanager
def single_worker(run):
    import fcntl
    with (Path(run) / '.worker.lock').open('a+') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error: raise ValueError('One model action process only') from error
        yield


def paused(run):
    if (Path(__file__).parent / 'PAUSE_REQUESTED.json').exists():
        save(Path(run) / 'checkpoints' / ('paused_' + str(time.time_ns()) + '.json'),
             dict(status='PAUSED_BEFORE_NEXT_ACTION', no_partial_fit_resumed=True))
        return True
    return False


def load_annual(spec, registered_specs):
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    from dmv2.features import FeaturePanel
    from dmv2.targets import MonthlyTargetPanel
    from dmv2.classification import classification_training_slice
    year = spec['year']; meta = {}
    for name, record in spec['metadata'].items():
        meta[name] = pq.ParquetFile(check(record['pin'])).read(columns=record['columns'])
        need(meta[name].num_rows == record['rows'], 'Metadata rowcount drift')
    validate_metadata(year, meta['feature_keys'].to_pylist(), meta['coverage'].to_pylist(),
        meta['original'].to_pylist(), meta['target_metadata'].to_pylist(), spec['windows'])
    diagnostic = meta['diagnostic_metadata'].to_pylist()
    need(len(diagnostic) == 12 and {r['signal_date'] for r in diagnostic} == set(spec['windows'])
         and all(r['outcome_month'] == spec['windows'][r['signal_date']]['outcome_month'] for r in diagnostic), 'Diagnostic clock changed')
    # Entire annual metadata admitted before numerical values are materialized.
    values = {}
    for name, record in spec['numeric'].items():
        t = pq.ParquetFile(check(record['pin'])).read(columns=record['columns'])
        t = t.filter(pc.is_in(t['signal_date'], value_set=pa.array(list(spec['windows']))))
        need(t.num_rows == record['rows'], 'Financial rows differ from admitted annual metadata')
        values[name] = t
    state = checked_read(spec['target_state_pin'])
    schemas = {name: pa.ipc.read_schema(pa.BufferReader(base64.b64decode(raw, validate=True))) for name, raw in state['schemas'].items()}
    target = MonthlyTargetPanel(values['targets'].cast(schemas['targets']),
        values['target_diagnostics'].cast(schemas['target_diagnostics']), state['provenance'], state['artifact_sha256'])
    panel = FeaturePanel(values['features'], meta['coverage'], registered_specs, {}, {},
        dict(target_windows=spec['windows'], source='explicit_verified_practical_catalog'))
    part = classification_training_slice(panel, target, train_start=f'{year}-01-01', train_end=f'{year}-12-31',
        label_observed_through=min(f'{year}-12-31', state['provenance']['label_observed_through']))
    coverage = meta['coverage'].to_pandas().set_index(list(KEYS)).reindex(part.X.index)
    part = replace(part, ledger=part.ledger.join(coverage[[c for c in coverage if c not in part.ledger]]))
    original = meta['original'].to_pandas().set_index(list(KEYS)).reindex(part.X.index)
    part = replace(part, ledger=part.ledger.join(original[[c for c in original if c not in part.ledger]]))
    for group in ('metadata', 'numeric'):
        for record in spec[group].values(): check(record['pin'])
    return part


def concat_parts(parts, start, end):
    import pandas as pd
    from dmv2.features import TrainingSlice
    X = pd.concat([p.X for p in parts]); y = pd.concat([p.y for p in parts]); ledger = pd.concat([p.ledger for p in parts])
    need(X.index.is_unique and y.index.equals(X.index) and ledger.index.equals(X.index), 'Concatenation changed original keys')
    provenance = {**parts[0].provenance, 'train_start': start, 'train_end': end,
        'label_observed_through': max(p.provenance['label_observed_through'] for p in parts)}
    return TrainingSlice(X, y, ledger, provenance)


def save_array(path, array):
    import numpy as np
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f: np.save(f, array, allow_pickle=False)
    return {**pin(path), 'shape': list(array.shape), 'dtype': str(array.dtype), 'array_sha256': array_sha(array)}


def array_sha(array):
    import numpy as np
    a = np.ascontiguousarray(array)
    return 'sha256:' + hashlib.sha256(memoryview(a).cast('B') if a.size else b'').hexdigest()


def save_frame(path, frame):
    import pyarrow as pa
    import pyarrow.parquet as pq
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), stream, compression='zstd')
    return pin(path)


def prepare(root, run, source, year):
    import numpy as np
    from dmv2.classifier_pilot import _ledger, _finite32
    from dmv2.preprocessing import fit_preprocessor
    out = Path(run) / 'folds' / str(year)
    need(not (out / 'DATA_STARTED.json').exists() and not (out / 'DATA_COMPLETE.json').exists(), 'Preserve existing preprocessing attempt; no automatic refit')
    save(out / 'DATA_STARTED.json', dict(year=year, source_plan_sha256=canonical(source), preprocessing_fits=2))
    parts = {y: load_annual(source['annuals'][str(y)], source['registered_specs']) for y in range(year - 10, year + 1)}
    w = year_window(year)
    train = concat_parts([parts[y] for y in range(year - 10, year)], w['train_start'], w['train_end'])
    test = concat_parts([parts[year]], w['prediction_start'], w['prediction_end'])
    need(not set(train.X.index) & set(test.X.index) and train.ledger.outcome_end.max() < test.ledger.outcome_start.min(), 'Train/prediction intervals overlap')
    need(train.provenance['label_observed_through'] == source['folds'][str(year)]['label_observed_through'],
         'Prepared labels changed independently completed training cutoff')
    train_ledger, test_ledger = _ledger(train, training=True), _ledger(test)
    for ledger in (train_ledger, test_ledger):
        ledger['fit_year'] = year; ledger['training_outcome_start'] = w['train_start']
        ledger['training_outcome_end'] = w['train_end']; ledger['training_label_observed_through'] = train.provenance['label_observed_through']
    artifacts = {'data/train_ledger.parquet': save_frame(out / 'data/train_ledger.parquet', train_ledger.reset_index()),
        'data/test_ledger.parquet': save_frame(out / 'data/test_ledger.parquet', test_ledger.reset_index())}
    y = train.y.loc[train_ledger.fit_mask].to_numpy(dtype=np.int64)
    artifacts['data/train_y.npy'] = save_array(out / 'data/train_y.npy', y)
    schemas = {}
    for bank in ('paper15', 'expanded'):
        names = source['paper_names'] if bank == 'paper15' else source['predictor_names']
        selected = replace(train, X=train.X[names]); fitted = fit_preprocessor(selected, fit_mask=train_ledger.fit_mask)
        a = fitted.transform(selected.X, prediction_mask=train_ledger.fit_mask)
        b = fitted.transform(test.X[names], prediction_mask=test_ledger.prediction_eligible)
        need(b.ledger.transformed.equals(test_ledger.prediction_eligible), 'Preprocessing changed prediction cohort')
        directory = out / 'preprocessors' / bank
        artifacts[f'preprocessors/{bank}/manifest.json'] = save(directory / 'manifest.json', dict(manifest=fitted.manifest, fingerprint=fitted.fingerprint))
        artifacts[f'preprocessors/{bank}/train_X.npy'] = save_array(directory / 'train_X.npy', _finite32(a.X.loc[train_ledger.fit_mask]))
        artifacts[f'preprocessors/{bank}/test_X.npy'] = save_array(directory / 'test_X.npy', _finite32(b.X.loc[test_ledger.prediction_eligible]))
        schemas[bank] = dict(fingerprint=fitted.fingerprint, input_predictors=len(names), matrix_width=len(fitted.feature_names))
        del selected, fitted, a, b; gc.collect()
    result = dict(status='COMPLETE', year=year, source_plan_sha256=canonical(source), artifacts=artifacts, schemas=schemas,
        preprocessor_fits=2, train_original_rows=len(train.X), train_fit_rows=int(train_ledger.fit_mask.sum()),
        test_original_rows=len(test.X), test_prediction_rows=int(test_ledger.prediction_eligible.sum()),
        test_labelled_rows=int(test_ledger.target_available.sum()),
        fit_keys_sha256=canonical([list(k) for k in train_ledger.index[train_ledger.fit_mask]]),
        prediction_keys_sha256=canonical([list(k) for k in test_ledger.index[test_ledger.prediction_eligible]]))
    return save(out / 'DATA_COMPLETE.json', result)


def fit_one(run, source, recipe, tuned, year, arm, maximum_seconds):
    import numpy as np
    from dmv2.pilot_models import _constructor, _training_labels, _json_parameters
    from threadpoolctl import threadpool_limits
    kind, bank = ARMS[arm]; fold = Path(run) / 'folds' / str(year); out = fold / 'arms' / arm
    need(not (out / 'FIT_ATTEMPT.json').exists() and not (out / 'FIT_COMPLETE.json').exists(), 'Preserve existing fit; never automatically repeat/resume an incomplete fit')
    need(len(list(Path(run).glob('folds/*/arms/*/FIT_ATTEMPT.json'))) < 20, 'Twenty estimator attempts exhausted')
    prepared = read(fold / 'DATA_COMPLETE.json'); validate_prepared(prepared, source, year)
    params = tuned_parameters(year, tuned[year]) if arm == 'rf_tuned_extended' else dict(recipe['estimators'][kind])
    need(params['n_jobs'] == (4 if arm == 'rf_tuned_extended' else 1), 'Frozen fit parallelism changed')
    X = np.load(check(prepared['artifacts'][f'preprocessors/{bank}/train_X.npy']), mmap_mode='r', allow_pickle=False)
    y = np.load(check(prepared['artifacts']['data/train_y.npy']), mmap_mode='r', allow_pickle=False)
    need(X.dtype == np.float32 and X.ndim == 2 and np.isfinite(X).all(), 'Prepared train matrix invalid')
    labels = _training_labels(y, len(X)); encoded = labels if kind == 'rf' else labels - 1
    expected = dict(year=year, arm=arm, prepared_pin=pin(fold / 'DATA_COMPLETE.json'),
        source_plan_sha256=canonical(source), preprocessor_fingerprint=prepared['schemas'][bank]['fingerprint'])
    save(out / 'FIT_ATTEMPT.json', dict(status='ATTEMPT_CONSUMED', estimator_fits=1, expected_metadata=expected))
    estimator = _constructor(kind)(**params); began = time.monotonic()
    old_handler = signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Per-fit time budget exceeded')))
    signal.setitimer(signal.ITIMER_REAL, maximum_seconds)
    try:
        with threadpool_limits(limits=1): estimator.fit(X, encoded)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0); signal.signal(signal.SIGALRM, old_handler)
    effective = _json_parameters(estimator.get_params(deep=False))
    native = [int(k) for k in estimator.classes_]
    need(native == (list(range(1, 11)) if kind == 'rf' else list(range(10))), 'Model class mapping changed')
    if arm == 'rf_tuned_extended': estimator.set_params(n_jobs=1)
    inference = _json_parameters(estimator.get_params(deep=False))
    if kind == 'rf':
        import joblib
        model = out / 'model.joblib'
        with model.open('xb') as stream: joblib.dump(estimator, stream, compress=0)
    else:
        model = out / 'model.ubj'; need(not model.exists(), 'Preserve model bytes'); estimator.save_model(model)
    result = dict(status='FIT_COMPLETE', fit_count=1, expected_metadata=expected, model=pin(model),
        metadata=dict(estimator_kind=kind, explicit_parameters=params, effective_parameters=effective,
            saved_inference_effective_parameters=inference, native_class_ids=native,
            probability_class_ids=list(range(1, 11)), train_rows=len(X), matrix_width=X.shape[1],
            prepared_train_matrix_sha256=array_sha(X), encoded_training_labels_sha256=array_sha(encoded),
            fit_seconds=time.monotonic()-began, test_labels_supplied_to_estimator=False,
            early_stopping_used=False, parameter_search_performed=False, probability_calibration_performed=False))
    return save(out / 'FIT_COMPLETE.json', result)


def predict_one(run, source, year, arm):
    import numpy as np
    import pandas as pd
    from dmv2.classifier_pilot import _predictions
    from dmv2.validation_reprediction import predict_saved_estimator
    from dmv2.walkforward_metrics import evaluate_probabilities
    from threadpoolctl import threadpool_limits
    kind, bank = ARMS[arm]; fold = Path(run) / 'folds' / str(year); out = fold / 'arms' / arm
    need(not (out / 'PREDICT_STARTED.json').exists() and not (out / 'PREDICT_COMPLETE.json').exists(), 'Preserve existing inference attempt')
    prepared = read(fold / 'DATA_COMPLETE.json'); validate_prepared(prepared, source, year)
    fitted = read(out / 'FIT_COMPLETE.json'); need(fitted['expected_metadata']['prepared_pin'] == pin(fold / 'DATA_COMPLETE.json'), 'Model/preprocessing identity differs')
    save(out / 'PREDICT_STARTED.json', dict(fit_receipt_pin=pin(out / 'FIT_COMPLETE.json'), estimator_fits=0))
    matrix = np.load(check(prepared['artifacts'][f'preprocessors/{bank}/test_X.npy']), mmap_mode='r', allow_pickle=False)
    ledger = pd.read_parquet(check(prepared['artifacts']['data/test_ledger.parquet'])).set_index(list(KEYS))
    need(canonical([list(k) for k in ledger.index[ledger.prediction_eligible]]) == prepared['prediction_keys_sha256'], 'Prediction key order drift')
    if kind == 'rf':
        import joblib
        estimator = joblib.load(check(fitted['model']))
    else:
        from xgboost import XGBClassifier
        estimator = XGBClassifier(n_jobs=1, device='cpu'); estimator.load_model(check(fitted['model'])); estimator.set_params(n_jobs=1, device='cpu')
    with threadpool_limits(limits=1):
        probabilities = predict_saved_estimator(estimator, kind, matrix, expected_native_ids=fitted['metadata']['native_class_ids'])
    prediction_index = ledger.index[ledger.prediction_eligible]
    frame = _predictions(ledger, probabilities, prediction_index)
    result = evaluate_probabilities(probabilities, ledger.loc[prediction_index].target_class.reset_index(drop=True),
        class_ids=tuple(range(1, 11)), row_metadata=ledger.loc[prediction_index].reset_index(),
        first_month=f'{year}-01', last_month=f'{year}-12')
    artifacts = dict(predictions=save_frame(out / 'predictions.parquet', frame),
        probabilities=save_array(out / 'probabilities.npy', probabilities),
        summary=save(out / 'reports/summary.json', result.summary))
    for name, table in result.tables.items(): artifacts[name] = save_frame(out / 'reports' / (name + '.parquet'), table.reset_index(drop=True))
    return save(out / 'PREDICT_COMPLETE.json', dict(status='COMPLETE', year=year, arm=arm,
        fit_receipt_pin=pin(out / 'FIT_COMPLETE.json'), prepared_pin=pin(fold / 'DATA_COMPLETE.json'),
        artifacts=artifacts, summary=result.summary, estimator_fits=0, preprocessing_fits=0))


def finalize(run):
    import pandas as pd
    out = Path(run) / 'stitched'; need(not out.exists(), 'Preserve prior stitched attempt')
    out.mkdir(); artifacts = {}; reference = None
    for arm in ARMS:
        frames = []
        for year in YEARS:
            receipt = read(Path(run) / 'folds' / str(year) / 'arms' / arm / 'PREDICT_COMPLETE.json')
            need(receipt['status'] == 'COMPLETE' and receipt['year'] == year and receipt['arm'] == arm, 'Annual forecast completion mismatch')
            frames.append(pd.read_parquet(check(receipt['artifacts']['predictions'])))
        frame = pd.concat(frames, ignore_index=True)
        need(not frame.duplicated(list(KEYS)).any() and frame.outcome_month.nunique() == 48, 'Stitched original keys/months changed')
        cohort = frame[[c for c in frame if not c.startswith('probability_') and c != 'prediction_status']]
        if reference is None: reference = cohort
        else: need(cohort.equals(reference), 'Five arms must retain identical original cohort/masks/labels')
        artifacts[arm] = save_frame(out / (arm + '.parquet'), frame)
    fits = list(Path(run).glob('folds/*/arms/*/FIT_COMPLETE.json'))
    prepared = list(Path(run).glob('folds/*/DATA_COMPLETE.json'))
    need(len(fits) == 20 and len(prepared) == 4 and sum(read(p)['preprocessor_fits'] for p in prepared) == 8, 'Final20/8 counters incomplete')
    return save(Path(run) / 'audit.json', dict(schema='dmv2-corrected-refit-model-audit/v1', status='COMPLETE_PENDING_INDEPENDENT_VERIFICATION',
        estimator_fits=20, preprocessor_fits=8, outer_years=list(YEARS), forecast_artifacts=artifacts,
        source_plan_pin=pin(Path(run) / 'read_plan.json'), identity_pin=pin(Path(run) / 'build_identity.json'),
        parameter_search=False, protected_2024_plus=True, old_runs_unchanged=True))


def execute(root, protocol_path, run, action_path, action_sha, labels, launch, launch_sha):
    root, run = Path(root).resolve(), Path(run).resolve(); sys.path.insert(0, str(root / 'src'))
    p, recipe, tuned, source = runtime(root, protocol_path, run)
    need(digest(action_path) == action_sha, 'Action plan hash changed'); a = read(action_path)
    need(a['status'] == 'SEALED_BEFORE_ACTION_READ' and a['source_plan_pin'] == pin(run / 'read_plan.json')
         and a['protocol_pin'] == pin(protocol_path) and a['worker_pin'] == pin(__file__), 'Action plan identity changed')
    need(a == action_plan(root, protocol_path, run, a['action'], a['year'], a['arm'], write=False),
         'Action files/projections differ from fresh metadata-only reconstruction')
    require_reservations(run, action_path, a, labels)
    require_launch(launch, launch_sha, action_path, a, p, protocol_path)
    with single_worker(run):
        if paused(run): return dict(status='PAUSED_BEFORE_NEXT_ACTION')
        began = time.monotonic()
        if a['action'] == 'prepare': result = prepare(root, run, source, a['year'])
        elif a['action'] == 'fit': result = fit_one(run, source, recipe, tuned, a['year'], a['arm'], p['limits']['per_fit_seconds'])
        elif a['action'] == 'predict': result = predict_one(run, source, a['year'], a['arm'])
        elif a['action'] == 'finalize': result = finalize(run)
        else: raise ValueError('Unknown sealed action')
        for record in source['annuals'].values():
            for group in ('metadata', 'numeric'):
                for item in record[group].values(): check(item['pin'])
        save(run / 'checkpoints' / f"{a['action']}_{a['year']}_{a['arm']}.json", dict(status='COMPLETE',
            action_plan_pin=pin(action_path), result_pin=result, active_seconds=time.monotonic()-began,
            worker_integrity_hash_io_bytes=HASH_IO_BYTES,
            estimator_fits=1 if a['action']=='fit' else 0, preprocessor_fits=2 if a['action']=='prepare' else 0,
            pause_requested_after_action=(Path(__file__).parent / 'PAUSE_REQUESTED.json').exists()))
        return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('command', choices=['plan', 'plan-action', 'execute'])
    cli.add_argument('--root', required=True); cli.add_argument('--protocol', required=True); cli.add_argument('--run', required=True)
    cli.add_argument('--action', choices=['prepare', 'fit', 'predict', 'finalize']); cli.add_argument('--year', type=int)
    cli.add_argument('--arm', choices=list(ARMS)); cli.add_argument('--action-plan'); cli.add_argument('--action-plan-sha256')
    cli.add_argument('--reservation', action='append', default=[]); cli.add_argument('--launch'); cli.add_argument('--launch-sha256')
    args = cli.parse_args()
    if args.command == 'plan': result = plan(args.root, args.protocol, args.run)
    elif args.command == 'plan-action': result = action_plan(args.root, args.protocol, args.run, args.action, args.year, args.arm)
    else:
        labels = dict(x.split('=', 1) for x in args.reservation)
        need(len(labels) == len(args.reservation), 'Duplicate category reservation argument')
        result = execute(args.root, args.protocol, args.run, args.action_plan, args.action_plan_sha256, labels, args.launch, args.launch_sha256)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__': main()
