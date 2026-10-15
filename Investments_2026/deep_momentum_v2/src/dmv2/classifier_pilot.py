"""Explicit four-fit temporal development pilot and read-only saved reports.

Only ``run`` fits models. The notebook reader never loads serialized estimators
or launches data preparation, fitting, calibration, RET or backtesting.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import gc
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import threading
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .catalog import SourceCatalog, canonical_sha256
from .features import KEYS
from .full_panel import (_artifact, _hash, _json, _limits, _progress, _rss_bytes,
                         _verify_artifacts, _within, _write_table)
from .paper_features import PAPER_FEATURE_NAMES
from .pilot_loading import plan_pilot_data, load_pilot_data
from .pilot_metrics import evaluate_probabilities, LOG_LOSS_FLOOR
from .pilot_models import fit_predict_pilot_arm
from .preprocessing import fit_preprocessor, PREPROCESSING_POLICY
from .ret import CLASS_IDS, align_class_probabilities
from .universe import DataContractError

ARMS = (
    {'arm_id': 'rf_paper15', 'estimator': 'rf', 'feature_set': 'paper15'},
    {'arm_id': 'xgb_paper15', 'estimator': 'xgb', 'feature_set': 'paper15'},
    {'arm_id': 'rf_extended', 'estimator': 'rf', 'feature_set': 'expanded'},
    {'arm_id': 'xgb_extended', 'estimator': 'xgb', 'feature_set': 'expanded'},
)
WORKERS = ('classifier_pilot.py', 'pilot_loading.py', 'pilot_models.py', 'pilot_metrics.py',
    'classification.py', 'preprocessing.py', 'features.py', 'paper_features.py',
    'ret.py', 'targets.py', 'target_coverage.py', 'full_target_panel.py',
    'catalog.py', 'context_panel_roles.py', 'full_panel.py', 'universe.py')
REPORT_NAMES = ('monthly', 'classwise_calibration', 'toplabel_calibration',
                'extreme_precision', 'row_scores')
DISABLED = ('parameter_search', 'early_stopping', 'outer_refit', 'outer_test_evaluation',
            'RET_evaluation', 'backtesting', 'cloud_mutations', 'source_fallback',
            'automatic_notebook_fitting')


def configuration(catalog):
    cfg = json.loads((catalog.root / 'configs/classifier_pilot.json').read_text())
    if (type(cfg.get('schema_version')) is not int or cfg['schema_version'] != 1
            or cfg.get('protocol_id') != 'dmv2-classifier-pilot-2010-2017-validate-2018-2019-v1'
            or cfg.get('status') != 'FROZEN_BEFORE_FIRST_FIT'
            or cfg.get('universe') != 'combined'
            or cfg.get('identity_policy') != 'provisional_unique_resolved_link'
            or cfg.get('feature_panel_role') != 'primary'
            or cfg.get('context_policy') != 'available_cohort_80pct'
            or cfg.get('arms') != list(ARMS)):
        raise DataContractError('Classifier pilot scope or four-arm design differs from user direction')
    dates = {'train_outcome_start': '2010-01-01', 'train_outcome_end': '2017-12-31',
        'validation_outcome_start': '2018-01-01', 'validation_outcome_end': '2019-12-31',
        'train_label_observed_through': '2017-12-31',
        'validation_label_observed_through': '2019-12-31'}
    if any(cfg.get(key) != value for key, value in dates.items()):
        raise DataContractError('Classifier pilot must retain its exact completed-outcome split')
    if (cfg.get('preprocessing_policy') != PREPROCESSING_POLICY
            or cfg.get('preprocessing_fit_period') != '2010-2017_only'
            or cfg.get('supervised_fit_population') != 'paper_prediction_eligible_and_target_available'
            or cfg.get('prediction_population') != 'paper_prediction_eligible_independent_of_future_labels'
            or cfg.get('evaluation_population') != 'same_observed_target_mask_for_all_arms'
            or cfg.get('preprocessor_shared_between_estimators_with_same_feature_set') is not True
            or cfg.get('references') != ['uniform', 'train_prior']
            or cfg.get('class_prior_weighting') != 'pooled_supervised_training_stock_month_counts'):
        raise DataContractError('Pilot preprocessing, prediction cohort or comparator policy changed')
    metrics, probabilities = cfg.get('metrics', {}), cfg.get('probability_construction', {})
    if (metrics.get('primary') != 'equal_month_mean_log_loss'
            or metrics.get('log_loss_true_class_probability_floor') != LOG_LOSS_FLOOR
            or metrics.get('floor_applies_only_to_log_evaluation_not_saved_probabilities') is not True
            or metrics.get('calibration_fitted') is not False
            or metrics.get('mode_ties') != 'lowest_canonical_class_id'
            or metrics.get('brier') != 'sum_squared_errors_over_ten_classes_then_equal_month_mean'
            or probabilities.get('canonical_classes') != list(CLASS_IDS)
            or any(type(k) is not int for k in probabilities.get('canonical_classes', []))
            or probabilities.get('repair_clipping_or_renormalization') is not False
            or probabilities.get('rf') != 'native_predict_proba_float64'
            or probabilities.get('xgb') != 'native_output_margins_then_objective_softmax_float64'
            or probabilities.get('matrix_dtype') != 'float32_after_finite_range_guard'):
        raise DataContractError('Pilot canonical probabilities or fixed metrics changed')
    execution = cfg.get('execution', {})
    for key, value in (('workers', 1), ('estimator_threads', 1), ('fits_per_arm', 1), ('maximum_total_fits', 4)):
        if type(execution.get(key)) is not int or execution[key] != value:
            raise DataContractError('Pilot permits exactly one sequential fit per arm')
    if execution.get('backend') != 'local_verified_saved_panels' or any(execution.get(k) is not False for k in DISABLED):
        raise DataContractError('Pilot cannot search, use outer tests, backtest or mutate cloud resources')
    for key in ('maximum_wall_seconds', 'maximum_peak_rss_bytes', 'maximum_output_bytes', 'maximum_projected_read_bytes'):
        if type(execution.get(key)) is not int or execution[key] <= 0:
            raise DataContractError('Pilot requires explicit positive resource ceilings')
    packages = cfg.get('packages', {})
    if set(packages) != {'scikit-learn', 'xgboost', 'scipy', 'joblib', 'threadpoolctl', 'narwhals', 'cloudpickle'}:
        raise DataContractError('Pilot model dependencies must be pinned explicitly')
    if any(not isinstance(v, str) or len(v.split('.')) != 3 or not all(p.isdigit() for p in v.split('.')) for v in packages.values()):
        raise DataContractError('Pilot package versions require exact numeric release pins')
    for kind, parameters in cfg.get('estimators', {}).items():
        if (kind not in ('rf', 'xgb') or type(parameters.get('n_jobs')) is not int or parameters['n_jobs'] != 1
                or type(parameters.get('n_estimators')) is not int or parameters['n_estimators'] <= 0
                or parameters.get('early_stopping_rounds') is not None or parameters.get('callbacks') is not None):
            raise DataContractError('Pilot estimators require explicit single-thread fixed iterations')
    if set(cfg.get('estimators', {})) != {'rf', 'xgb'}:
        raise DataContractError('Pilot needs both frozen estimator parameter dictionaries')
    return cfg


def build_identity(catalog, plan):
    cfg = configuration(catalog)
    for package, expected in cfg['packages'].items():
        if version(package) != expected:
            raise DataContractError('Installed pilot dependency differs from its frozen pin: ' + package)
    return {'schema_version': 1, 'configuration': cfg, 'data_plan_sha256': plan.fingerprint,
        'data_plan': plan.provenance, 'source_pins': catalog.provenance(),
        'workers': {name: _hash(Path(__file__).parent / name) for name in WORKERS},
        'runtime': {'python': platform.python_version(), **{name: version(name)
            for name in ('numpy', 'pandas', 'pyarrow', 'exchange-calendars', *cfg['packages'])}}}


def _verify_sources(root, manifest):
    items = manifest.get('artifacts', [])
    if not items or len({item['path'] for item in items}) != len(items):
        raise DataContractError('Pilot source preservation manifest must have unique nonempty paths')
    _verify_artifacts(root, {item['path']: item for item in items})
    return len(items)


def _source_manifest(catalog, plan):
    target_root = plan.years[2010].target_directory.parent.parent
    original = json.loads((target_root / 'preserved_source_manifest.json').read_text())
    entries = {item['path']: item for item in original['artifacts']}
    audit = json.loads((target_root / 'audit.json').read_text())
    for directory, items in ((target_root, audit['artifacts']), (target_root, audit['years'])):
        for item in items.values():
            path = _within(directory, directory / item['path'])
            relative = str(path.relative_to(catalog.root))
            entries[relative] = {**item, 'path': relative}
    for year in range(2010, 2024):
        directory = target_root / 'years' / str(year)
        annual = json.loads((directory / 'audit.json').read_text())
        for item in annual['artifacts'].values():
            path = _within(directory, directory / item['path'])
            relative = str(path.relative_to(catalog.root))
            entries[relative] = {**item, 'path': relative}
    for path in (target_root / 'audit.json', catalog.root / 'outputs/full_target_panel/LATEST.json'):
        relative = str(path.relative_to(catalog.root))
        entries[relative] = {'path': relative, 'sha256': _hash(path), 'size_bytes': path.stat().st_size}
    return {'artifacts': list(entries.values()),
        'scope': 'Immutable feature/daily runs, full shared targets and existing pointers; byte hashing does not decode future financial values'}


def _ledger(part, *, training=False):
    frame = part.ledger.copy()
    frame['prediction_eligible'] = frame.paper_prediction_eligible.astype(bool)
    frame['target_class'] = pd.array(part.y, dtype='Int64')
    if training:
        frame['fit_mask'] = frame.prediction_eligible & frame.target_available
    return frame


def _predictions(ledger, probabilities, prediction_index):
    if not isinstance(ledger.index, pd.MultiIndex) or tuple(ledger.index.names) != KEYS or not ledger.index.is_unique:
        raise DataContractError('Pilot prediction ledger must retain all original unique keys')
    expected = ledger.index[ledger.prediction_eligible]
    if not expected.equals(prediction_index):
        raise DataContractError('Prediction matrix must match the entire ordered signal-time eligible cohort')
    values = align_class_probabilities(probabilities, class_ids=CLASS_IDS)
    if len(values) != len(expected):
        raise DataContractError('Pilot probabilities changed prediction row coverage')
    frame = ledger.copy()
    columns = [f'probability_{k}' for k in CLASS_IDS]
    for column in columns:
        frame[column] = np.nan
    frame.loc[prediction_index, columns] = values
    frame['prediction_status'] = np.where(frame.prediction_eligible, 'predicted', 'withheld_signal_time_baseline')
    return frame.reset_index()


def _finite32(frame):
    array = frame.to_numpy(dtype=np.float64)
    if not np.isfinite(array).all() or (array.size and np.max(np.abs(array)) > np.finfo(np.float32).max):
        raise DataContractError('Frozen prepared inputs must fit finite float32 without clipping')
    return np.ascontiguousarray(array, dtype=np.float32)


def _save_evaluation(directory, frame, probabilities, labels, prediction_index, artifacts):
    directory.mkdir(parents=True, exist_ok=True)
    predictions = _predictions(frame, probabilities, prediction_index)
    _write_table(directory, 'predictions', predictions, artifacts)
    row_metadata = frame.loc[prediction_index].reset_index()
    evaluation = evaluate_probabilities(probabilities, labels.loc[prediction_index].reset_index(drop=True),
        class_ids=CLASS_IDS, row_metadata=row_metadata)
    reports = directory / 'reports'
    reports.mkdir(exist_ok=True)
    _write_table(reports, 'summary', pd.DataFrame([evaluation.summary]), {})
    artifacts['summary_report'] = _artifact(directory, reports / 'summary.parquet')
    for name, table in evaluation.tables.items():
        _write_table(reports, name, table.reset_index(drop=True), {})
        artifacts[name] = _artifact(directory, reports / (name + '.parquet'))
    if evaluation.summary['outcome_months'] != 24 or evaluation.summary['observed_label_months'] != 24:
        raise DataContractError('Pilot validation must retain all 24 completed calendar months')
    return evaluation.summary


class _ResourceGuard:
    """Stop an explicit CLI run if its frozen wall/RSS ceiling is exceeded."""
    def __init__(self, cfg, started, destination):
        self.cfg, self.started, self.destination = cfg, started, destination
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._watch, daemon=True)

    def _watch(self):
        last_report = self.started
        while not self.stop.wait(1):
            elapsed, peak = time.monotonic() - self.started, _rss_bytes()
            if time.monotonic() - last_report >= 60:
                _progress('pilot_resource_usage', elapsed_seconds=elapsed, peak_rss_bytes=peak)
                last_report = time.monotonic()
            limits = self.cfg['execution']
            if elapsed > limits['maximum_wall_seconds'] or peak > limits['maximum_peak_rss_bytes']:
                _json(self.destination / 'resource_failure.json', {'elapsed_seconds': elapsed, 'peak_rss_bytes': peak})
                os._exit(75)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=2)


def run_classifier_pilot(catalog):
    started = time.monotonic()
    cfg = configuration(catalog)
    plan = plan_pilot_data(catalog)
    identity = build_identity(catalog, plan)
    pin = canonical_sha256(identity)
    read_bytes = plan.read_plan['projected_compressed_bytes']
    if (read_bytes > cfg['execution']['maximum_projected_read_bytes']
            or plan.read_plan['estimated_peak_memory_bytes'] > cfg['execution']['maximum_peak_rss_bytes']):
        raise DataContractError('Pilot metadata plan exceeds the frozen local read/memory budget')
    parent = _within(catalog.root, catalog.root / 'outputs/model_pilot')
    destination = _within(parent, parent / ('2010-2017__2018-2019-' + pin[7:23]))
    if (destination / 'audit.json').exists():
        return read_classifier_pilot_run(catalog, directory=destination)['directory']
    if destination.exists():
        raise DataContractError('Preserve and investigate this incomplete pilot; no automatic duplicate fit: ' + str(destination))
    destination.mkdir(parents=True)
    _json(destination / 'build_identity.json', identity)
    _json(destination / 'read_plan.json', plan.provenance)
    manifest = _source_manifest(catalog, plan)
    _verify_sources(catalog.root, manifest)
    _json(destination / 'source_preservation_manifest.json', manifest)
    _progress('classifier_pilot_plan', **plan.read_plan)
    with _ResourceGuard(cfg, started, destination):
        return _execute(catalog, plan, cfg, identity, pin, destination, manifest, started, read_bytes)


def _execute(catalog, plan, cfg, identity, pin, destination, manifest, started, read_bytes):
    from threadpoolctl import threadpool_limits
    bundle = load_pilot_data(catalog, plan=plan)
    train_ledger, validation_ledger = _ledger(bundle.train, training=True), _ledger(bundle.validation)
    if train_ledger.outcome_end.max() >= validation_ledger.outcome_start.min():
        raise DataContractError('Pilot fit and validation outcome intervals overlap')
    artifacts = {}
    data_directory = destination / 'data'
    data_directory.mkdir()
    _write_table(data_directory, 'train_ledger', train_ledger.reset_index(), {})
    _write_table(data_directory, 'validation_ledger', validation_ledger.reset_index(), {})
    for name in ('train_ledger', 'validation_ledger'):
        artifacts['data/' + name] = _artifact(destination, data_directory / (name + '.parquet'))
    fit_mask = train_ledger.fit_mask
    prediction_mask = validation_ledger.prediction_eligible
    fit_y = bundle.train.y.loc[fit_mask].copy()
    validation_y = bundle.validation.y.copy()
    prediction_index = validation_ledger.index[prediction_mask]
    counts = {str(k): int((fit_y == k).sum()) for k in CLASS_IDS}
    if not all(counts.values()):
        raise DataContractError('All ten classes must be present in the supervised training partition')
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
            raise DataContractError('Added feature missingness changed the shared validation prediction cohort')
        matrix_train = _finite32(prepared_train.X.loc[fit_mask])
        matrix_validation = _finite32(prepared_validation.X.loc[prediction_mask])
        directory = destination / 'preprocessors' / feature_set
        directory.mkdir(parents=True)
        _json(directory / 'manifest.json', {'manifest': fitted.manifest, 'fingerprint': fitted.fingerprint})
        np.save(directory / 'train_X.npy', matrix_train, allow_pickle=False)
        np.save(directory / 'validation_X.npy', matrix_validation, allow_pickle=False)
        schemas[feature_set] = {'input_predictors': len(names),
            'retained_predictors': len(fitted.manifest['retained_columns']),
            'matrix_width': len(fitted.feature_names), 'dropped_columns': fitted.manifest['dropped_columns'],
            'fingerprint': fitted.fingerprint}
        for name in ('manifest.json', 'train_X.npy', 'validation_X.npy'):
            artifacts[f'preprocessors/{feature_set}/{name}'] = _artifact(destination, directory / name)
        _progress('pilot_preprocessing_completed', feature_set=feature_set,
            fit_rows=len(fit_y), prediction_rows=len(prediction_index), **schemas[feature_set])
        del train, validation, fitted, prepared_train, prepared_validation, matrix_train, matrix_validation
        gc.collect()
        _limits(cfg, started, destination, read_bytes)
    _json(destination / 'loading_provenance.json', bundle.provenance)
    del bundle
    gc.collect()
    summaries, arm_audits = [], {}
    for arm in cfg['arms']:
        arm_id, feature_set, kind = arm['arm_id'], arm['feature_set'], arm['estimator']
        directory = destination / 'arms' / arm_id
        directory.mkdir(parents=True)
        _json(directory / 'ATTEMPT.json', {'arm': arm, 'build_identity_sha256': pin,
            'attempt': 1, 'start_elapsed_seconds': time.monotonic() - started})
        prep = destination / 'preprocessors' / feature_set
        train_X = np.load(prep / 'train_X.npy', mmap_mode='r', allow_pickle=False)
        validation_X = np.load(prep / 'validation_X.npy', mmap_mode='r', allow_pickle=False)
        arm_started = time.monotonic()
        _progress('pilot_fit_started', arm_id=arm_id, train_rows=len(fit_y), prediction_rows=len(prediction_index), matrix_width=train_X.shape[1])
        with threadpool_limits(limits=1):
            result = fit_predict_pilot_arm(kind, train_X, fit_y, validation_X,
                parameters=cfg['estimators'][kind], package_versions=cfg['packages'])
        arm_artifacts = {}
        summary = _save_evaluation(directory, validation_ledger, result.probabilities,
            validation_y, prediction_index, arm_artifacts)
        if kind == 'rf':
            import joblib
            joblib.dump(result.estimator, directory / 'model.joblib', compress=3)
            model_path = directory / 'model.joblib'
        else:
            result.estimator.save_model(directory / 'model.ubj')
            model_path = directory / 'model.ubj'
            _json(directory / 'booster_configuration.json', json.loads(result.estimator.get_booster().save_config()))
            arm_artifacts['booster_configuration'] = _artifact(directory, directory / 'booster_configuration.json')
        arm_artifacts['model'] = _artifact(directory, model_path)
        arm_artifacts['attempt'] = _artifact(directory, directory / 'ATTEMPT.json')
        metadata = {**result.metadata, 'train_outcome_start': cfg['train_outcome_start'],
            'train_outcome_end': cfg['train_outcome_end'],
            'validation_outcome_start': cfg['validation_outcome_start'],
            'validation_outcome_end': cfg['validation_outcome_end'],
            'preprocessor_fingerprint': schemas[feature_set]['fingerprint'],
            'fit_keys_sha256': canonical_sha256([list(key) for key in train_ledger.index[fit_mask]]),
            'prediction_keys_sha256': canonical_sha256([list(key) for key in prediction_index]),
            'elapsed_seconds': time.monotonic() - arm_started}
        _json(directory / 'audit.json', {'status': 'COMPLETE', 'arm': arm,
            'build_identity_sha256': pin, 'metadata': metadata, 'summary': summary, 'artifacts': arm_artifacts})
        arm_audits[arm_id] = _artifact(destination, directory / 'audit.json')
        summaries.append({'arm_id': arm_id, 'feature_set': feature_set, 'estimator': kind,
            **schemas[feature_set], **summary, 'fit_seconds': metadata['elapsed_seconds']})
        _progress('pilot_fit_completed', arm_id=arm_id, **summary, elapsed_seconds=metadata['elapsed_seconds'])
        del result, train_X, validation_X
        gc.collect()
        _limits(cfg, started, destination, read_bytes)
    for reference, vector in (('uniform', np.full(10, .1)), ('train_prior', prior)):
        probabilities = np.broadcast_to(vector, (len(prediction_index), 10)).copy()
        directory = destination / 'refs' / reference
        reference_artifacts = {}
        summary = _save_evaluation(directory, validation_ledger, probabilities, validation_y, prediction_index, reference_artifacts)
        _json(directory / 'audit.json', {'status': 'COMPLETE', 'reference': reference,
            'build_identity_sha256': pin,
            'metadata': {'train_class_counts': counts, 'prior_vector': vector.tolist(),
                'prior_weighting': cfg['class_prior_weighting'], 'training_outcome_end': cfg['train_outcome_end']},
            'summary': summary, 'artifacts': reference_artifacts})
        arm_audits[reference] = _artifact(destination, directory / 'audit.json')
        summaries.append({'arm_id': reference, 'feature_set': 'reference', 'estimator': 'reference', **summary})
    reports = destination / 'reports'
    reports.mkdir()
    # Drop nested drop-reason dictionaries from this flat comparison view; the
    # exact schema manifest remains in every preprocessor and root summary.
    flat = [{key: value for key, value in row.items() if key != 'dropped_columns'} for row in summaries]
    _write_table(reports, 'comparison', pd.DataFrame(flat), {})
    artifacts['comparison'] = _artifact(destination, reports / 'comparison.parquet')
    for name in ('build_identity.json', 'read_plan.json', 'source_preservation_manifest.json', 'loading_provenance.json'):
        artifacts[name] = _artifact(destination, destination / name)
    preserved = _verify_sources(catalog.root, manifest)
    limits = _limits(cfg, started, destination, read_bytes)
    summary = {'fitted_arms': 4, 'fits_per_arm': 1, 'schemas': schemas,
        'train_original_rows': len(train_ledger), 'train_prediction_eligible_rows': int(train_ledger.prediction_eligible.sum()),
        'train_fit_rows': int(fit_mask.sum()), 'validation_original_rows': len(validation_ledger),
        'validation_prediction_rows': len(prediction_index),
        'validation_labelled_rows': int(validation_ledger.target_available.sum()),
        'validation_missing_label_prediction_rows': int((prediction_mask & ~validation_ledger.target_available).sum()),
        'train_class_counts': counts, 'training_prior_vector': prior.tolist(),
        'same_fit_keys_all_arms': True, 'same_prediction_keys_all_arms': True,
        'same_observed_label_mask_all_arms': True, 'source_files_unchanged': preserved,
        'preprocessing_fit_end': cfg['train_outcome_end'], 'validation_months': 24,
        'financial_2020_plus_decoded': False, 'parameter_search_performed': False,
        'validation_labels_used_for_fit_or_stopping': False,
        'RET_evaluated': False, 'backtest_performed': False, 'outer_test_evaluated': False,
        'paper_100_fit_ensemble': False, 'knowledge_time_certified': False,
        'economic_return_certified': False, 'resource_usage': limits,
        'elapsed_seconds': time.monotonic() - started}
    _json(destination / 'audit.json', {'status': 'COMPLETE', 'build_identity_sha256': pin,
        'summary': summary, 'artifacts': artifacts, 'arms': arm_audits})
    _json(destination.parent / 'LATEST.json', {'path': destination.name,
        'build_identity_sha256': pin, 'audit_sha256': _hash(destination / 'audit.json')})
    _progress('classifier_pilot_completed', directory=str(destination), **summary)
    return destination


def read_classifier_pilot_run(catalog, *, directory=None):
    parent = _within(catalog.root, catalog.root / 'outputs/model_pilot')
    pointer = None
    if directory is None:
        pointer = json.loads((parent / 'LATEST.json').read_text())
        directory = _within(parent, parent / pointer['path'])
    directory = _within(parent, directory)
    audit = json.loads((directory / 'audit.json').read_text())
    if pointer is not None and _hash(directory / 'audit.json') != pointer['audit_sha256']:
        raise DataContractError('Saved classifier pilot pointer changed')
    plan = plan_pilot_data(catalog)
    identity = build_identity(catalog, plan)
    pin = canonical_sha256(identity)
    if (audit.get('status') != 'COMPLETE' or audit.get('build_identity_sha256') != pin
            or (pointer is not None and pointer.get('build_identity_sha256') != pin)
            or set(audit.get('arms', {})) != {*(arm['arm_id'] for arm in ARMS), 'uniform', 'train_prior'}):
        raise DataContractError('Saved classifier pilot is stale or incomplete')
    _verify_artifacts(directory, audit['artifacts'])
    _verify_artifacts(directory, audit['arms'])
    required = {'build_identity.json', 'read_plan.json', 'source_preservation_manifest.json',
        'loading_provenance.json', 'comparison', 'data/train_ledger', 'data/validation_ledger',
        *(f'preprocessors/{feature_set}/{name}' for feature_set in ('paper15', 'expanded')
          for name in ('manifest.json', 'train_X.npy', 'validation_X.npy'))}
    if not required <= set(audit['artifacts']):
        raise DataContractError('Saved pilot is missing required data/schema/report artifacts')
    if json.loads((directory / 'build_identity.json').read_text()) != identity:
        raise DataContractError('Saved classifier pilot recipe differs from the current frozen protocol')
    expected_arms = {arm['arm_id']: arm for arm in ARMS}
    for name, item in audit['arms'].items():
        path = _within(directory, directory / item['path'])
        arm_audit = json.loads(path.read_text())
        required_arm = {'predictions', 'summary_report', *REPORT_NAMES}
        if name in expected_arms:
            required_arm |= {'model', 'attempt'}
            if expected_arms[name]['estimator'] == 'xgb':
                required_arm.add('booster_configuration')
        if (arm_audit.get('status') != 'COMPLETE'
                or arm_audit.get('build_identity_sha256') != pin
                or not required_arm <= set(arm_audit.get('artifacts', {}))
                or (name in expected_arms and arm_audit.get('arm') != expected_arms[name])
                or (name not in expected_arms and arm_audit.get('reference') != name)):
            raise DataContractError('Saved pilot arm is incomplete')
        _verify_artifacts(path.parent, arm_audit['artifacts'])
    _verify_sources(catalog.root, json.loads((directory / 'source_preservation_manifest.json').read_text()))
    return {'directory': directory, 'audit': audit,
        'tables': {'comparison': pd.read_parquet(directory / audit['artifacts']['comparison']['path'])}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'run', 'verify'))
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    catalog = SourceCatalog(args.root)
    if args.action == 'plan':
        cfg, plan = configuration(catalog), plan_pilot_data(catalog)
        _progress('classifier_pilot_plan', protocol=cfg['protocol_id'], **plan.read_plan)
    elif args.action == 'run':
        run_classifier_pilot(catalog)
    else:
        run = read_classifier_pilot_run(catalog)
        _progress('classifier_pilot_verified', directory=str(run['directory']), **run['audit']['summary'])


if __name__ == '__main__':
    main()
