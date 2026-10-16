"""Four explicit refits on the new data-limited 2010--2019 reconstruction.

This phase never follows the original feature/target/pilot pointers. It checks
only the ten annual artifacts supplied by the new run, validates metadata before
decoding financial columns, and uses the unchanged classification/preprocessing
and estimator workers. An incomplete attempt is never resumed implicitly.
"""
from __future__ import annotations

from dataclasses import replace
import gc
from importlib.metadata import version
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd

from .catalog import canonical_sha256
from .classifier_pilot import (ARMS, REPORT_NAMES, _finite32, _ledger,
    _ResourceGuard, _save_evaluation)
from .full_panel import (_artifact, _hash, _json, _limits, _progress, _specs,
    _verify_artifacts, _within, _write_table)
from .paper_features import PAPER_FEATURE_NAMES
from .pilot_loading import (PilotDataPlan, YEARS, TRAIN_YEARS, VALIDATION_YEARS,
    _year_plan, _year_snapshot, load_pilot_data)
from .pilot_models import fit_predict_pilot_arm
from .preprocessing import fit_preprocessor
from .ret import CLASS_IDS
from .universe import DataContractError


PHASE_WORKERS = ('fallback_models.py', 'classifier_pilot.py', 'pilot_loading.py',
    'pilot_models.py', 'pilot_metrics.py', 'classification.py', 'preprocessing.py',
    'features.py', 'paper_features.py', 'ret.py', 'targets.py',
    'full_target_panel.py', 'full_panel.py', 'catalog.py', 'universe.py')


def _recipe(root, configuration):
    """Use the frozen pilot settings under the NEW outer execution authority."""
    if (configuration.get('protocol_id') != 'dmv2-data-limited-restricted-universe-2018-2019-v1'
            or configuration.get('research_status') != 'DATA_LIMITED_RESTRICTED_UNIVERSE'
            or configuration.get('missing_route_fallback', {}).get('approved') is not True
            or configuration.get('execution_restrictions', {}).get('valid_historical_training_rows_retained') is not True
            or configuration.get('execution', {}).get('model_maximum_total_fits') != 4):
        raise DataContractError('Four refits require the frozen explicit data-limited fallback authority')
    supplied = configuration.get('model_recipe')
    source_path = Path(root) / 'configs/classifier_pilot.json'
    source = json.loads(source_path.read_text())
    if configuration.get('model_recipe_source', {}).get('sha256') != 'sha256:' + _hash(source_path):
        raise DataContractError('Fallback frozen model-recipe source pin changed')
    if supplied != source or not isinstance(supplied, dict):
        raise DataContractError('Fallback refits must retain the complete frozen pilot recipe')
    if (supplied.get('arms') != list(ARMS)
            or supplied.get('train_outcome_start') != '2010-01-01'
            or supplied.get('train_outcome_end') != '2017-12-31'
            or supplied.get('validation_outcome_start') != '2018-01-01'
            or supplied.get('validation_outcome_end') != '2019-12-31'
            or supplied.get('context_policy') != 'available_cohort_80pct'):
        raise DataContractError('Fallback permits the same four-arm development split only')
    execution = supplied.get('execution', {})
    if (execution.get('maximum_total_fits') != 4
            or execution.get('fits_per_arm') != 1
            or execution.get('workers') != 1
            or execution.get('estimator_threads') != 1
            or execution.get('maximum_peak_rss_bytes', 2**64) > 8 * 1024**3
            or execution.get('maximum_wall_seconds', 2**64) > 5400):
        raise DataContractError('Fallback model phase exceeds its four-fit local budget')
    for name, expected in supplied['packages'].items():
        if version(name) != expected:
            raise DataContractError('Fallback installed model package differs from frozen pin: ' + name)
    return supplied


def plan_fallback_model_inputs(root, run_directory, identity_sha256):
    """Inspect/hash only NEW annual caches and their projected metadata.

    The proven annual guard checks all source/date keys, exact independent
    next-month windows, admission flags and target-state metadata before any
    feature, return, or class value is decoded. Counts come from this new run,
    never from the original pilot's populations.
    """
    root = Path(root).resolve()
    run = _within(root, Path(run_directory).resolve())
    if run == root or not isinstance(identity_sha256, str) or not identity_sha256.startswith('sha256:'):
        raise DataContractError('Fallback model inputs need a separate run and explicit parent identity')
    annual_pins = {str(year): _artifact(run, run / 'years' / str(year) / 'audit.json')
                   for year in YEARS}
    expected = {'directory': run,
                'audit': {'build_identity_sha256': identity_sha256, 'years': annual_pins}}
    specs = _specs()
    years = {year: _year_plan(root, expected, expected, year, specs) for year in YEARS}
    from .identity_amendment import SOURCE_ROUTES
    for annual in years.values():
        for row in annual.coverage.to_pylist():
            if (row['source_security_id'] in SOURCE_ROUTES
                    and (row['identity_eligible'] or row['paper_prediction_eligible']
                         or row['intrinio_security_id'] is not None)):
                raise DataContractError('Fallback metadata readmits a known wrong historical BBBY route')
    rows = sum(part.read_plan['original_rows'] for part in years.values())
    read_plan = {
        'outcome_years': list(YEARS), 'train_outcome_years': list(TRAIN_YEARS),
        'validation_outcome_years': list(VALIDATION_YEARS),
        'rows_original_train': sum(years[y].read_plan['original_rows'] for y in TRAIN_YEARS),
        'rows_original_validation': sum(years[y].read_plan['original_rows'] for y in VALIDATION_YEARS),
        'rows_eligible_train': sum(years[y].read_plan['prediction_eligible_rows'] for y in TRAIN_YEARS),
        'rows_eligible_validation': sum(years[y].read_plan['prediction_eligible_rows'] for y in VALIDATION_YEARS),
        'rows_labelled_train': sum(years[y].read_plan['labelled_rows'] for y in TRAIN_YEARS),
        'rows_labelled_validation': sum(years[y].read_plan['labelled_rows'] for y in VALIDATION_YEARS),
        'projected_compressed_bytes': sum(part.read_plan['projected_compressed_bytes'] for part in years.values()),
        'estimated_peak_memory_bytes': rows * (715 * 8 * 4 + 4096),
        'signals': 120, 'registered_feature_columns': len(specs),
        'model_feature_columns': sum(spec.get('model_eligible', True) for spec in specs),
        'financial_values_read': False, 'financial_2020_plus_read': False,
        'raw_source_reads': False, 'original_pointers_followed': False}
    provenance = {'protocol': 'data_limited_four_arm_development_inputs_v1',
        'parent_identity_sha256': identity_sha256,
        'read_plan': read_plan,
        'years': {str(year): _year_snapshot(root, annual) for year, annual in years.items()},
        'feature_dictionary_sha256': canonical_sha256(specs),
        'metadata_guards_before_financial_decode': True,
        'same_classification_adapter_for_all_arms': True,
        'financial_2020_plus_read': False, 'knowledge_time_verified': False,
        'data_limited_reconstruction': True}
    return PilotDataPlan(root, years, read_plan, provenance, canonical_sha256(provenance))


def _phase_identity(root, configuration, parent_identity, plan):
    return {'schema_version': 1, 'phase': 'data_limited_four_arm_models_v1',
        'parent_identity_sha256': parent_identity,
        'configuration_sha256': canonical_sha256(configuration),
        'model_recipe': configuration['model_recipe'],
        'model_recipe_source': _artifact(root, Path(root) / 'configs/classifier_pilot.json'),
        'data_plan_sha256': plan.fingerprint, 'data_plan': plan.provenance,
        'workers': {name: _hash(Path(__file__).parent / name) for name in PHASE_WORKERS},
        'runtime': {'python': platform.python_version(),
            **{name: version(name) for name in ('numpy', 'pandas', 'pyarrow',
                *configuration['model_recipe']['packages'])}}}


def _verify_input_plan(plan):
    for annual in plan.years.values():
        _verify_artifacts(plan.root, {item['path']: item for item in annual.source_pins})


def read_fallback_models(directory, *, identity_sha256=None, model_identity_sha256=None):
    """Verify saved phase hashes only; never deserialize a model or fit."""
    directory = Path(directory).resolve()
    audit = json.loads((directory / 'audit.json').read_text())
    if (audit.get('status') != 'COMPLETE'
            or audit.get('summary', {}).get('fitted_arms') != 4
            or audit.get('summary', {}).get('data_limited_reconstruction') is not True
            or identity_sha256 is not None and audit.get('build_identity_sha256') != identity_sha256
            or model_identity_sha256 is not None and audit.get('model_identity_sha256') != model_identity_sha256
            or set(audit.get('arms', {})) != {*(arm['arm_id'] for arm in ARMS), 'uniform', 'train_prior'}):
        raise DataContractError('Saved fallback model phase is stale or incomplete')
    _verify_artifacts(directory, audit['artifacts'])
    _verify_artifacts(directory, audit['arms'])
    required = {'build_identity.json', 'read_plan.json', 'ATTEMPT.json',
        'loading_provenance.json', 'comparison', 'data/train_ledger',
        'data/validation_ledger', *(f'preprocessors/{feature_set}/{name}'
            for feature_set in ('paper15', 'expanded')
            for name in ('manifest.json', 'train_X.npy', 'validation_X.npy'))}
    if not required <= set(audit['artifacts']):
        raise DataContractError('Saved fallback model phase lacks its required matrix/schema/report receipts')
    phase = json.loads((directory / 'build_identity.json').read_text())
    if (canonical_sha256(phase) != audit['model_identity_sha256']
            or phase.get('parent_identity_sha256') != audit['build_identity_sha256']):
        raise DataContractError('Saved fallback model phase identity changed')
    for name, item in audit['arms'].items():
        path = _within(directory, directory / item['path'])
        arm = json.loads(path.read_text())
        if (arm.get('status') != 'COMPLETE'
                or arm.get('build_identity_sha256') != audit['build_identity_sha256']
                or arm.get('model_identity_sha256') != audit['model_identity_sha256']
                or not {'predictions', 'summary_report', *REPORT_NAMES} <= set(arm.get('artifacts', {}))):
            raise DataContractError('Saved fallback arm is incomplete')
        if name in {arm['arm_id'] for arm in ARMS} and not {'model', 'attempt'} <= set(arm['artifacts']):
            raise DataContractError('Saved fallback fitted arm lacks its model/attempt receipt')
        _verify_artifacts(path.parent, arm['artifacts'])
    return {'directory': directory, 'audit': audit}


def run_fallback_models(root, run_directory, configuration, identity_sha256):
    """Execute exactly four sequential fixed fits; explicit caller only."""
    root = Path(root).resolve()
    run = _within(root, Path(run_directory).resolve())
    recipe = _recipe(root, configuration)
    plan = plan_fallback_model_inputs(root, run, identity_sha256)
    phase = _phase_identity(root, configuration, identity_sha256, plan)
    phase_pin = canonical_sha256(phase)
    destination = run / 'models'
    if (destination / 'audit.json').exists():
        return read_fallback_models(destination, identity_sha256=identity_sha256,
            model_identity_sha256=phase_pin)['directory']
    if destination.exists():
        raise DataContractError('Preserve incomplete fallback model attempt; automatic refitting is forbidden: ' + str(destination))
    limits = recipe['execution']
    read_bytes = plan.read_plan['projected_compressed_bytes']
    if (read_bytes > limits['maximum_projected_read_bytes']
            or plan.read_plan['estimated_peak_memory_bytes'] > limits['maximum_peak_rss_bytes']):
        raise DataContractError('Fallback model input plan exceeds its measured local read/memory budget')
    destination.mkdir()
    _json(destination / 'build_identity.json', phase)
    _json(destination / 'read_plan.json', plan.provenance)
    _json(destination / 'ATTEMPT.json', {'attempt': 1, 'build_identity_sha256': identity_sha256,
        'model_identity_sha256': phase_pin, 'maximum_total_fits': 4})
    started = time.monotonic()
    with _ResourceGuard(recipe, started, destination):
        return _execute(root, plan, recipe, identity_sha256, phase_pin, destination, started, read_bytes)


def _execute(root, plan, recipe, parent_pin, phase_pin, destination, started, read_bytes):
    from types import SimpleNamespace
    from threadpoolctl import threadpool_limits
    bundle = load_pilot_data(SimpleNamespace(root=root), plan=plan)
    train_ledger = _ledger(bundle.train, training=True)
    validation_ledger = _ledger(bundle.validation)
    if train_ledger.outcome_end.max() >= validation_ledger.outcome_start.min():
        raise DataContractError('Fallback training and validation outcomes overlap')
    artifacts = {}
    data_directory = destination / 'data'
    data_directory.mkdir()
    for name, frame in (('train_ledger', train_ledger), ('validation_ledger', validation_ledger)):
        _write_table(data_directory, name, frame.reset_index(), {})
        artifacts['data/' + name] = _artifact(destination, data_directory / (name + '.parquet'))
    fit_mask = train_ledger.fit_mask
    prediction_mask = validation_ledger.prediction_eligible
    fit_y, validation_y = bundle.train.y.loc[fit_mask].copy(), bundle.validation.y.copy()
    prediction_index = validation_ledger.index[prediction_mask]
    counts = {str(k): int((fit_y == k).sum()) for k in CLASS_IDS}
    if not all(counts.values()):
        raise DataContractError('Fallback supervised training must retain all ten canonical classes')
    prior = np.array([counts[str(k)] / len(fit_y) for k in CLASS_IDS])
    schemas = {}
    for feature_set in ('paper15', 'expanded'):
        names = list(PAPER_FEATURE_NAMES) if feature_set == 'paper15' else list(bundle.train.X)
        train = replace(bundle.train, X=bundle.train.X[names])
        validation = replace(bundle.validation, X=bundle.validation.X[names])
        fitted = fit_preprocessor(train, fit_mask=fit_mask)
        prepared_train = fitted.transform(train.X, prediction_mask=fit_mask)
        prepared_validation = fitted.transform(validation.X, prediction_mask=prediction_mask)
        if not prepared_validation.ledger.transformed.equals(prediction_mask):
            raise DataContractError('Fallback preprocessing changed the shared prediction population')
        train_X = _finite32(prepared_train.X.loc[fit_mask])
        validation_X = _finite32(prepared_validation.X.loc[prediction_mask])
        directory = destination / 'preprocessors' / feature_set
        directory.mkdir(parents=True)
        _json(directory / 'manifest.json', {'manifest': fitted.manifest, 'fingerprint': fitted.fingerprint})
        np.save(directory / 'train_X.npy', train_X, allow_pickle=False)
        np.save(directory / 'validation_X.npy', validation_X, allow_pickle=False)
        schemas[feature_set] = {'input_predictors': len(names),
            'retained_predictors': len(fitted.manifest['retained_columns']),
            'matrix_width': len(fitted.feature_names),
            'dropped_columns': fitted.manifest['dropped_columns'], 'fingerprint': fitted.fingerprint}
        for name in ('manifest.json', 'train_X.npy', 'validation_X.npy'):
            artifacts[f'preprocessors/{feature_set}/{name}'] = _artifact(destination, directory / name)
        del train, validation, fitted, prepared_train, prepared_validation, train_X, validation_X
        gc.collect()
        _limits(recipe, started, destination, read_bytes)
    _json(destination / 'loading_provenance.json', bundle.provenance)
    del bundle
    gc.collect()
    summaries, arms = [], {}
    for arm in recipe['arms']:
        name, feature_set, kind = arm['arm_id'], arm['feature_set'], arm['estimator']
        directory = destination / 'arms' / name
        directory.mkdir(parents=True)
        _json(directory / 'ATTEMPT.json', {'arm': arm, 'attempt': 1,
            'build_identity_sha256': parent_pin, 'model_identity_sha256': phase_pin})
        prep = destination / 'preprocessors' / feature_set
        train_X = np.load(prep / 'train_X.npy', mmap_mode='r', allow_pickle=False)
        validation_X = np.load(prep / 'validation_X.npy', mmap_mode='r', allow_pickle=False)
        arm_started = time.monotonic()
        _progress('fallback_fit_started', arm_id=name, train_rows=len(fit_y),
            prediction_rows=len(prediction_index), matrix_width=train_X.shape[1])
        with threadpool_limits(limits=1):
            result = fit_predict_pilot_arm(kind, train_X, fit_y, validation_X,
                parameters=recipe['estimators'][kind], package_versions=recipe['packages'])
        arm_artifacts = {}
        summary = _save_evaluation(directory, validation_ledger, result.probabilities,
            validation_y, prediction_index, arm_artifacts)
        if kind == 'rf':
            import joblib
            model = directory / 'model.joblib'
            joblib.dump(result.estimator, model, compress=3)
        else:
            model = directory / 'model.ubj'
            result.estimator.save_model(model)
            _json(directory / 'booster_configuration.json', json.loads(result.estimator.get_booster().save_config()))
            arm_artifacts['booster_configuration'] = _artifact(directory, directory / 'booster_configuration.json')
        arm_artifacts['model'] = _artifact(directory, model)
        arm_artifacts['attempt'] = _artifact(directory, directory / 'ATTEMPT.json')
        metadata = {**result.metadata,
            **{key: recipe[key] for key in ('train_outcome_start', 'train_outcome_end',
                'validation_outcome_start', 'validation_outcome_end')},
            'preprocessor_fingerprint': schemas[feature_set]['fingerprint'],
            'fit_keys_sha256': canonical_sha256([list(key) for key in train_ledger.index[fit_mask]]),
            'prediction_keys_sha256': canonical_sha256([list(key) for key in prediction_index]),
            'elapsed_seconds': time.monotonic() - arm_started,
            'data_limited_reconstruction': True,
            'execution_only_restrictions_change_training': False}
        _json(directory / 'audit.json', {'status': 'COMPLETE', 'arm': arm,
            'build_identity_sha256': parent_pin, 'model_identity_sha256': phase_pin,
            'metadata': metadata, 'summary': summary, 'artifacts': arm_artifacts})
        arms[name] = _artifact(destination, directory / 'audit.json')
        summaries.append({'arm_id': name, 'feature_set': feature_set, 'estimator': kind,
            **schemas[feature_set], **summary, 'fit_seconds': metadata['elapsed_seconds']})
        _progress('fallback_fit_completed', arm_id=name, **summary,
            elapsed_seconds=metadata['elapsed_seconds'])
        del result, train_X, validation_X
        gc.collect()
        _limits(recipe, started, destination, read_bytes)
    for name, vector in (('uniform', np.full(10, .1)), ('train_prior', prior)):
        directory = destination / 'refs' / name
        reference_artifacts = {}
        probabilities = np.broadcast_to(vector, (len(prediction_index), 10)).copy()
        summary = _save_evaluation(directory, validation_ledger, probabilities,
            validation_y, prediction_index, reference_artifacts)
        _json(directory / 'audit.json', {'status': 'COMPLETE', 'reference': name,
            'build_identity_sha256': parent_pin, 'model_identity_sha256': phase_pin,
            'metadata': {'train_class_counts': counts, 'prior_vector': vector.tolist()},
            'summary': summary, 'artifacts': reference_artifacts})
        arms[name] = _artifact(destination, directory / 'audit.json')
        summaries.append({'arm_id': name, 'feature_set': 'reference',
            'estimator': 'reference', **summary})
    reports = destination / 'reports'
    reports.mkdir()
    _write_table(reports, 'comparison', pd.DataFrame([
        {key: value for key, value in row.items() if key != 'dropped_columns'}
        for row in summaries]), {})
    artifacts['comparison'] = _artifact(destination, reports / 'comparison.parquet')
    for name in ('build_identity.json', 'read_plan.json', 'ATTEMPT.json', 'loading_provenance.json'):
        artifacts[name] = _artifact(destination, destination / name)
    _verify_input_plan(plan)
    usage = _limits(recipe, started, destination, read_bytes)
    summary = {'fitted_arms': 4, 'fits_per_arm': 1, 'schemas': schemas,
        'train_original_rows': len(train_ledger), 'train_fit_rows': int(fit_mask.sum()),
        'train_prediction_eligible_rows': int(train_ledger.prediction_eligible.sum()),
        'validation_original_rows': len(validation_ledger),
        'validation_prediction_rows': len(prediction_index),
        'validation_labelled_rows': int(validation_ledger.target_available.sum()),
        'validation_missing_label_prediction_rows': int((prediction_mask & ~validation_ledger.target_available).sum()),
        'train_class_counts': counts, 'training_prior_vector': prior.tolist(),
        'same_fit_keys_all_arms': True, 'same_prediction_keys_all_arms': True,
        'same_observed_label_mask_all_arms': True, 'validation_months': 24,
        'financial_2020_plus_decoded': False, 'parameter_search_performed': False,
        'validation_labels_used_for_fit_or_stopping': False,
        'data_limited_reconstruction': True,
        'historical_identity_certified': False, 'knowledge_time_certified': False,
        'execution_only_restrictions_change_training': False,
        'original_pilot_pointer_changed': False, 'resource_usage': usage,
        'elapsed_seconds': time.monotonic() - started}
    _json(destination / 'audit.json', {'status': 'COMPLETE',
        'build_identity_sha256': parent_pin, 'model_identity_sha256': phase_pin,
        'summary': summary, 'artifacts': artifacts, 'arms': arms})
    _progress('fallback_models_completed', directory=str(destination), **summary)
    return destination
