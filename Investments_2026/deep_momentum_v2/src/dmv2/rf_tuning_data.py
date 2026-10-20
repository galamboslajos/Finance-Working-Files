"""Additive, explicitly authorized temporal RF inputs and durable matrices.

Planning reads JSON, byte hashes and Parquet footers only. Materialization is a
separate action after a supervisor has reserved its exact column-chunk budget.
Nothing in this module changes the frozen pilot or walk-forward guards.
"""
from __future__ import annotations

from dataclasses import replace
import gc
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .catalog import canonical_sha256
from .classification import classification_training_slice
from .classifier_pilot import _finite32, _ledger
from .features import FeaturePanel, KEYS
from .full_panel import _specs, _verify_artifacts
from .full_target_panel import _load_target_state
from .pilot_loading import COVERAGE_METADATA, TARGET_METADATA, _concat_slices, _projected_bytes
from .pilot_models import _training_labels
from .preprocessing import FittedPreprocessor, fit_preprocessor
from .universe import DataContractError
from .walkforward_loading import resolve_walkforward_year
from .walkforward_models import _annual_inputs

PROTOCOL_ID = 'dmv2-rf-expanded-temporal-tuning-20261020-v1'


def need(condition, message):
    if not condition:
        raise DataContractError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def pin(path):
    path = Path(path).resolve()
    need(path.is_file(), 'Artifact is absent: ' + str(path))
    return {'path': str(path), 'size_bytes': path.stat().st_size, 'sha256': file_hash(path)}


def verify_pin(item, *, within=None):
    need(isinstance(item, dict), 'An explicit artifact pin is required')
    path = Path(item['path']).resolve()
    if within is not None:
        need(path.is_relative_to(Path(within).resolve()), 'Artifact leaves its declared root')
    need(path.is_file() and path.stat().st_size == item['size_bytes']
         and file_hash(path) == item['sha256'], 'Artifact pin changed: ' + str(path))
    return path


def write_json(path, value):
    """Publish once by atomic rename; never replace a completed artifact."""
    path = Path(path)
    need(not path.exists(), 'Refusing to replace an artifact: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial-' + str(os.getpid()))
    with temporary.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    need(not path.exists(), 'Artifact appeared during publication')
    os.rename(temporary, path)
    return pin(path)


def _settings(request):
    root, run = Path(request['root']).resolve(), Path(request['run_root']).resolve()
    need(run.is_relative_to(root / 'outputs') and run != root / 'outputs',
         'Use a separate private output attempt inside the project')
    protocol_path = Path(request['protocol_path']).resolve()
    need(protocol_path.is_relative_to(root), 'Frozen protocol must be inside the project')
    protocol = read_json(protocol_path)
    need(protocol.get('protocol_id') == PROTOCOL_ID and
         protocol.get('rf', {}).get('fixed_parameters', {}).get('n_estimators') == 500 and
         protocol['rf']['fixed_parameters'].get('n_jobs') == 4 and
         protocol['rf']['fixed_parameters'].get('random_state') == 1729 and
         protocol.get('planned_fit_counts', {}).get('maximum_estimator_fits') == 100,
         'The exact frozen RF temporal protocol is required')
    year = request['outer_year']
    need(type(year) is int and year in (2020, 2021, 2022, 2023), 'Unknown outer year')
    fold = next(f for f in protocol['chronology']['folds'] if f['outer_outcome_year'] == year)
    split_id = request['split_id']
    if split_id == f'{year}_outer_train':
        split = {'split_id': split_id, 'train_outcome_start': fold['train_outcome_start'],
                 'train_outcome_end': fold['train_outcome_end'], 'validation_year': None}
        stage = 'outer'
    else:
        choices = [s for s in fold['inner_splits'] if s['split_id'] == split_id]
        need(len(choices) == 1, 'Split is not one of the twelve frozen inner splits')
        split = dict(choices[0])
        split['validation_year'] = int(split['validation_outcome_start'][:4])
        stage = 'inner'
    need(int(split['train_outcome_start'][:4]) == year - 10 and
         int(split['train_outcome_end'][:4]) < year,
         'Temporal training window changed')
    return root, run, protocol, split, stage


def check_authorization(request):
    root, run, protocol, split, stage = _settings(request)
    auth_path = Path(request['authorization_path']).resolve()
    need(auth_path.is_relative_to(run), 'Execution authorization must be pinned inside the new attempt')
    auth = read_json(auth_path)
    need(auth.get('protocol_id') == PROTOCOL_ID and auth.get('run_root') == str(run)
         and all(auth.get(k) is True for k in ('fitting_authorized',
             'preprocessing_fitting_authorized', 'new_financial_reads_authorized', 'prediction_authorized')),
         'Explicit later user execution authority is required')
    return root, run, protocol, split, stage


def _check_reference_pins(root, protocol):
    for item in protocol['reference']['pins']:
        relative = Path(item['path'])
        need(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe frozen reference path')
        verify_pin({**item, 'path': str(root / relative)}, within=root)


def _annual_plan(root, reference_run, identity, year):
    """Authenticate an annual source without decoding a single column."""
    reference = resolve_walkforward_year(root, reference_run, identity, year)
    directory, audit = Path(reference['directory']), reference['audit']
    names = [*KEYS, *(s['name'] for s in _specs())]
    artifacts = {}
    for name in ('features', 'feature_coverage', 'original_cohort', 'targets',
                 'target_diagnostics', 'target_state'):
        item = audit['artifacts'][name]
        _verify_artifacts(directory, {name: item})
        path = (directory / item['path']).resolve()
        need(path.parent == directory and path.name == name + ('.json' if name == 'target_state' else '.parquet'),
             'Annual artifact redirects a fixed source')
        artifacts[name] = path
    need(set(pq.ParquetFile(artifacts['features']).schema_arrow.names) == set(names),
         'Frozen feature dictionary changed')
    # These are exactly the physical reads in _annual_inputs + _load_annual.
    projections = [
        (artifacts['features'], list(KEYS)),
        (artifacts['feature_coverage'], list(COVERAGE_METADATA)),
        (artifacts['original_cohort'], list(KEYS)),
        (artifacts['targets'], list(TARGET_METADATA)),
        (artifacts['target_diagnostics'], ['signal_date', 'outcome_month']),
        (artifacts['features'], names),
        (artifacts['targets'], pq.ParquetFile(artifacts['targets']).schema_arrow.names),
        (artifacts['target_diagnostics'], pq.ParquetFile(artifacts['target_diagnostics']).schema_arrow.names),
    ]
    records = [{'path': str(path), 'columns': list(columns),
                'projected_compressed_bytes': _projected_bytes(path, columns)}
               for path, columns in projections]
    return {'year': year, 'directory': str(directory), 'audit': pin(directory / 'audit.json'),
            'build_identity_sha256': reference['build_identity_sha256'],
            'source_pins': [pin(path) for path in artifacts.values()],
            'projections': records,
            'projected_compressed_bytes': sum(r['projected_compressed_bytes'] for r in records)}


def _winner(request):
    _, run, _, _, stage = _settings(request)
    need(stage == 'outer', 'A winner seal applies only to an outer-year action')
    winner_path = verify_pin(request['winner_path'], within=run)
    winner = read_json(winner_path)
    need(winner.get('status') == 'SEALED' and winner.get('protocol_id') == PROTOCOL_ID
         and winner.get('outer_year') == request['outer_year']
         and winner.get('candidate_id') == request['candidate_id'],
         'Outer action requires the matching completed and sealed inner-grid winner')
    return winner


def plan_action(request):
    root, run, protocol, split, stage = _settings(request)
    planned_action = request.get('planned_action', request.get('action'))
    need(planned_action in ('prepare', 'predict_outer'), 'Only a bounded materialization can be planned')
    _check_reference_pins(root, protocol)
    reference_run = root / protocol['reference']['source_corrected_run']
    identity = read_json(reference_run / 'build_identity.json')
    need(canonical_sha256(identity) == protocol['reference']['source_corrected_build_identity_sha256'],
         'Reviewed corrected annual source identity changed')
    if planned_action == 'predict_outer':
        _winner(request)
        years = [request['outer_year']]
    else:
        years = list(range(int(split['train_outcome_start'][:4]), int(split['train_outcome_end'][:4]) + 1))
        if stage == 'inner':
            years.append(split['validation_year'])
    need(years and min(years) >= 2010 and max(years) <= 2023
         and (planned_action == 'predict_outer' or max(years) < request['outer_year']),
         'Refusing premature outer-year or protected financial values')
    annuals = [_annual_plan(root, reference_run, identity, year) for year in years]
    return {'schema': 'dmv2-rf-tuning-read-plan/v1', 'protocol_id': PROTOCOL_ID,
            'protocol_pin': pin(request['protocol_path']), 'outer_year': request['outer_year'],
            'split_id': split['split_id'], 'stage': stage, 'planned_action': planned_action,
            'train_start': split['train_outcome_start'], 'train_end': split['train_outcome_end'],
            'validation_year': split['validation_year'], 'annuals': annuals,
            'feature_dictionary_sha256': canonical_sha256(_specs()),
            'projected_source_read_bytes': sum(a['projected_compressed_bytes'] for a in annuals),
            'planning_financial_values_decoded': False,
            'reference_build_identity_pin': pin(reference_run / 'build_identity.json')}


def authorized_plan(request):
    root, run, protocol, split, stage = check_authorization(request)
    plan_path = verify_pin(request['plan_path'], within=run)
    plan = read_json(plan_path)
    need(plan['protocol_id'] == PROTOCOL_ID and plan['outer_year'] == request['outer_year']
         and plan['split_id'] == request['split_id'] and plan['planned_action'] == request['action']
         and plan['protocol_pin'] == pin(request['protocol_path'])
         and plan['feature_dictionary_sha256'] == canonical_sha256(_specs()),
         'Source plan changed or is not bound to this action')
    expected_years = ([request['outer_year']] if request['action'] == 'predict_outer' else
                      list(range(int(split['train_outcome_start'][:4]), int(split['train_outcome_end'][:4]) + 1))
                      + ([split['validation_year']] if stage == 'inner' else []))
    need([a['year'] for a in plan['annuals']] == expected_years
         and plan['train_start'] == split['train_outcome_start']
         and plan['train_end'] == split['train_outcome_end']
         and plan['validation_year'] == split['validation_year']
         and plan['stage'] == stage,
         'Reserved plan changed its complete frozen annual boundaries')
    reservation_path = Path(request['reservation_path']).resolve()
    need(reservation_path.is_relative_to(run), 'Read reservation must belong to the attempt')
    reservation = read_json(reservation_path)
    need(reservation.get('execution_authorized') is True
         and reservation.get('plan_sha256') == canonical_sha256(plan)
         and reservation.get('projected_source_read_bytes') == plan['projected_source_read_bytes'],
         'Exact persisted supervisor source-read reservation is required before decoding')
    _check_reference_pins(root, protocol)
    verify_pin(plan['reference_build_identity_pin'], within=root)
    for annual in plan['annuals']:
        verify_pin(annual['audit'], within=root)
        for item in annual['source_pins']:
            verify_pin(item, within=root)
    return root, run, protocol, plan


def _load_annual(root, annual):
    directory = Path(annual['directory'])
    reference = {'directory': directory, 'audit': read_json(directory / 'audit.json'),
                 'build_identity_sha256': annual['build_identity_sha256']}
    checked = _annual_inputs(root, reference, annual['year'])
    specs = _specs()
    table = pq.ParquetFile(directory / 'features.parquet').read(columns=[*KEYS, *(s['name'] for s in specs)])
    panel = FeaturePanel(table, checked.coverage, specs, {}, {},
                         {'target_windows': checked.windows, 'source': 'pinned_source_corrected_rf_tuning'})
    target = _load_target_state(directory)
    year = annual['year']
    part = classification_training_slice(panel, target, train_start=f'{year}-01-01',
        train_end=f'{year}-12-31', label_observed_through=min(f'{year}-12-31', target.provenance['label_observed_through']))
    coverage = checked.coverage.to_pandas().set_index(list(KEYS)).reindex(part.X.index)
    part = replace(part, ledger=part.ledger.join(coverage[[c for c in coverage if c not in part.ledger]]))
    need(len(part.X) == checked.metadata['original_rows']
         and int(part.ledger.paper_prediction_eligible.sum()) == checked.metadata['prediction_rows']
         and int(part.ledger.target_available.sum()) == checked.metadata['labelled_rows'],
         'Source decoding changed original membership or signal/target masks')
    for item in annual['source_pins']:
        verify_pin(item, within=root)
    return part


def validate_partition_clocks(train, prediction=None):
    """Fail closed rather than silently deleting a crossing/misdated target."""
    cutoff = train.provenance['label_observed_through']
    need(train.ledger.outcome_end.max() <= cutoff and train.ledger.outcome_start.min() >= train.provenance['train_start'],
         'Training target interval crosses its explicit completed-outcome cutoff')
    if prediction is not None:
        need(not set(train.X.index) & set(prediction.X.index)
             and train.ledger.outcome_end.max() <= min(prediction.X.index.get_level_values('signal_date'))
             and train.ledger.outcome_end.max() < prediction.ledger.outcome_start.min()
             and list(train.X) == list(prediction.X),
             'Training labels overlap validation signals/outcomes, keys or schema')


def _completed_exchange_cutoff(year):
    import exchange_calendars as xcals
    calendar = xcals.get_calendar('XNYS', start=f'{year}-01-01', end=f'{year}-12-31')
    return calendar.sessions[-1].strftime('%Y-%m-%d')


def _save_array(path, array):
    need(not Path(path).exists(), 'Prepared matrix already exists')
    temporary = Path(str(path) + '.partial')
    with temporary.open('xb') as stream:
        np.save(stream, array, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.rename(temporary, path)
    return pin(path)


def _save_frame(path, frame):
    """Finish and fsync Parquet bytes before publishing a checkpoint name."""
    path = Path(path)
    need(not path.exists(), 'Refusing to replace a saved ledger/report')
    temporary = path.with_name(path.name + '.partial')
    with temporary.open('xb') as stream:
        frame.to_parquet(stream, index=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.rename(temporary, path)
    return pin(path)


def _key_hash(index):
    return canonical_sha256([list(key) for key in index])


def _fresh_output(request):
    run, output = Path(request['run_root']).resolve(), Path(request['output_dir']).resolve()
    need(output.is_relative_to(run) and output != run and not output.exists(),
         'Use a fresh non-overwriting action output directory within the attempt')
    output.mkdir(parents=True)
    return output


def prepare_split(request):
    root, run, protocol, plan = authorized_plan(request)
    output = _fresh_output(request)
    parts = {a['year']: _load_annual(root, a) for a in plan['annuals']}
    train_years = range(int(plan['train_start'][:4]), int(plan['train_end'][:4]) + 1)
    train = _concat_slices([parts[y] for y in train_years], start=plan['train_start'], end=plan['train_end'])
    train = replace(train, provenance={**train.provenance,
        'financial_2020_plus_read': max(train_years) >= 2020,
        'approved_inner_training_partition': plan['stage'] == 'inner',
        'source_corrected_temporal_tuning': True})
    expected_cutoff = _completed_exchange_cutoff(int(plan['train_end'][:4]))
    need(train.provenance['label_observed_through'] == expected_cutoff
         and train.ledger.outcome_end.max() == expected_cutoff,
         'Training labels do not end at the independently completed XNYS cutoff')
    validation = parts[plan['validation_year']] if plan['stage'] == 'inner' else None
    validate_partition_clocks(train, validation)
    need(len(train.X.columns) == 715, 'Fixed expanded candidate bank must have exactly715 registered predictors')
    ledger = _ledger(train, training=True)
    fit_mask = ledger.fit_mask
    labels = _training_labels(train.y.loc[fit_mask], int(fit_mask.sum()))
    fitted = fit_preprocessor(train, fit_mask=fit_mask)
    prepared = fitted.transform(train.X, prediction_mask=ledger.prediction_eligible)
    matrix = _finite32(prepared.X.loc[fit_mask])
    artifacts = {'train_X': _save_array(output / 'train_X.npy', matrix),
                 'train_y': _save_array(output / 'train_y.npy', labels)}
    artifacts['train_ledger'] = _save_frame(output / 'train_ledger.parquet', ledger.reset_index())
    artifacts['preprocessor'] = write_json(output / 'preprocessor.json', fitted.manifest)
    evidence = {}
    numeric = train.X.loc[fit_mask].astype(float).replace([np.inf, -np.inf], np.nan)
    for name in numeric:
        finite = numeric[name].dropna()
        evidence[name] = {'finite_count': len(finite), 'finite_unique_count': int(finite.nunique()),
                          'finite_values_sha256': hashlib.sha256(np.ascontiguousarray(finite.to_numpy(dtype=np.float64)).tobytes()).hexdigest(),
                          'median': fitted.manifest['medians'].get(name),
                          'drop_reason': fitted.manifest['dropped_columns'].get(name)}
    artifacts['train_schema_evidence'] = write_json(output / 'train_schema_evidence.json', evidence)
    manifest = {'schema': 'dmv2-rf-tuning-prepared/v1', 'status': 'COMPLETE',
                'protocol_id': PROTOCOL_ID, 'outer_year': plan['outer_year'], 'split_id': plan['split_id'],
                'stage': plan['stage'], 'validation_year': plan['validation_year'],
                'plan': request['plan_path'], 'source_plan': plan,
                'preprocessor': artifacts['preprocessor'], 'artifacts': artifacts,
                'matrix_width': matrix.shape[1], 'train_rows': len(matrix),
                'train_original_rows': len(train.X), 'train_keys_sha256': _key_hash(train.X.index[fit_mask]),
                'train_original_keys_sha256': _key_hash(train.X.index),
                'train_provenance': train.provenance, 'preprocessing_fits': 1,
                'financial_2024_plus_decoded': False, 'prediction_rows': 0}
    if validation is not None:
        prediction_ledger = _ledger(validation)
        prediction_mask = prediction_ledger.prediction_eligible
        prediction_prepared = fitted.transform(validation.X, prediction_mask=prediction_mask)
        need(prediction_prepared.ledger.transformed.equals(prediction_mask),
             'Expanded missingness changed signal-time prediction rows')
        prediction_matrix = _finite32(prediction_prepared.X.loc[prediction_mask])
        artifacts['prediction_X'] = _save_array(output / 'prediction_X.npy', prediction_matrix)
        artifacts['prediction_ledger'] = _save_frame(output / 'prediction_ledger.parquet', prediction_ledger.reset_index())
        score = prediction_mask & prediction_ledger.target_available
        manifest.update(prediction_rows=int(prediction_mask.sum()), prediction_original_rows=len(validation.X),
                        prediction_keys_sha256=_key_hash(validation.X.index[prediction_mask]),
                        prediction_original_keys_sha256=_key_hash(validation.X.index),
                        score_keys_sha256=_key_hash(validation.X.index[score]),
                        prediction_provenance=validation.provenance)
    manifest_pin = write_json(output / 'prepare_manifest.json', manifest)
    del parts, train, numeric, matrix, prepared
    gc.collect()
    return {'status': 'COMPLETE', 'action': 'prepare', 'preprocessing_fits': 1, 'estimator_fits': 0,
            'prepare_manifest': manifest_pin, 'artifacts': [*artifacts.values(), manifest_pin],
            'projected_source_read_bytes': plan['projected_source_read_bytes'],
            'matrix_width': manifest['matrix_width']}


def load_prepared(request):
    _, run, _, _, _ = check_authorization(request)
    manifest_path = verify_pin(request['prepare_manifest'], within=run)
    prepared = read_json(manifest_path)
    need(prepared.get('status') == 'COMPLETE' and prepared.get('protocol_id') == PROTOCOL_ID
         and prepared.get('outer_year') == request['outer_year'] and prepared.get('split_id') == request['split_id'],
         'Prepared split is incomplete or belongs to a different clock')
    for item in prepared['artifacts'].values():
        verify_pin(item, within=run)
    return prepared


def prepare_outer_prediction(request):
    root, run, protocol, plan = authorized_plan(request)
    _winner(request)
    prepared = load_prepared(request)
    need(prepared['stage'] == 'outer' and plan['stage'] == 'outer' and len(plan['annuals']) == 1,
         'Only sealed outer prediction can decode its single outcome year')
    part = _load_annual(root, plan['annuals'][0])
    train_ledger = pd.read_parquet(verify_pin(prepared['artifacts']['train_ledger'], within=run)).set_index(list(KEYS))
    need(train_ledger.outcome_end.max() <= min(part.X.index.get_level_values('signal_date'))
         and train_ledger.outcome_end.max() < part.ledger.outcome_start.min(),
         'Outer completed targets overlap prediction timing')
    state = read_json(verify_pin(prepared['preprocessor'], within=run))
    fitted = FittedPreprocessor(json.dumps(state, sort_keys=True, separators=(',', ':'), allow_nan=False), canonical_sha256(state))
    ledger = _ledger(part)
    prediction_mask = ledger.prediction_eligible
    result = fitted.transform(part.X, prediction_mask=prediction_mask)
    need(result.ledger.transformed.equals(prediction_mask), 'Added missingness removed outer prediction rows')
    return _finite32(result.X.loc[prediction_mask]), ledger, part, plan
