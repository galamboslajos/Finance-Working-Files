"""One-action child process for the separately authorized RF tuning protocol.

The supervisor owns scheduling, persisted counters, resources and pause. Each
``fit`` invocation performs exactly one forest fit. Imports and planning do not
fit or decode financial source panels. Outputs are immutable per action.
"""
from __future__ import annotations

import argparse
import math
import os
from numbers import Integral
from pathlib import Path
import time
from importlib.metadata import version

import numpy as np
import pandas as pd

from .catalog import canonical_sha256
from .classifier_pilot import _predictions
from .features import KEYS
from .pilot_models import _json_parameters, _matrix, _training_labels
from .ret import CLASS_IDS, align_class_probabilities
from .rf_tuning_data import (PROTOCOL_ID, _fresh_output, _key_hash, _save_array, _save_frame,
    _winner, check_authorization, load_prepared, need, pin, plan_action,
    prepare_outer_prediction, prepare_split, read_json, verify_pin, write_json)
from .walkforward_metrics import (_calibration, _confusion, _extremes, _labels,
    _mode_classes)


def _candidate(protocol, candidate_id):
    matches = [candidate for candidate in protocol['rf']['candidates'] if candidate['candidate_id'] == candidate_id]
    need(len(matches) == 1, 'Candidate is not in the frozen eight-setting grid')
    candidate = matches[0]
    parameters = dict(protocol['rf']['fixed_parameters'])
    parameters.update({k: candidate[k] for k in ('max_depth', 'min_samples_leaf', 'max_features')})
    need(parameters['n_estimators'] == 500 and parameters['n_jobs'] == 4
         and parameters['random_state'] == 1729 and parameters['max_depth'] in (8, 12)
         and parameters['min_samples_leaf'] in (20, 100)
         and parameters['max_features'] in ('sqrt', 0.05)
         and parameters['class_weight'] is None and parameters['bootstrap'] is True
         and parameters['oob_score'] is False and parameters['warm_start'] is False,
         'Frozen RF parameter contract changed')
    return candidate, parameters


def nominal_feature_count(max_features, width):
    return max(1, int(math.sqrt(width) if max_features == 'sqrt' else max_features * width))


def evaluate_tuning_probabilities(probabilities, ledger, *, year):
    """Same frozen loss/calibration formulas, with explicit 2017--2023 bounds.

    Old metric modules intentionally reject earlier tuning years. This additive
    adapter widens only this experiment's date gate, without changing globals.
    """
    need(type(year) is int and 2017 <= year <= 2023, 'Tuning scores require an explicit approved year')
    need(isinstance(ledger, pd.DataFrame) and not ledger.columns.duplicated().any()
         and ledger.index.is_unique and tuple(ledger.index.names) == KEYS,
         'Score ledger must retain all original unique source/signal keys')
    required = {'prediction_eligible', 'target_available', 'target_class', 'outcome_month', 'outcome_start', 'outcome_end'}
    need(required <= set(ledger), 'Score ledger lacks explicit masks, labels or target intervals')
    for (sid, signal), row in ledger.iterrows():
        month = row.outcome_month
        need(isinstance(sid, str) and bool(sid.strip()) and isinstance(signal, str)
             and isinstance(month, str) and f'{year}-01' <= month <= f'{year}-12'
             and str(pd.Period(signal[:7], 'M') + 1) == month
             and signal < row.outcome_start <= row.outcome_end
             and row.outcome_start[:7] == row.outcome_end[:7] == month,
             'Score target interval overlaps its signal or exceeds the approved year')
    for mask in ('prediction_eligible', 'target_available'):
        need(ledger[mask].map(lambda v: isinstance(v, (bool, np.bool_))).all(), 'Score masks must be explicit booleans')
    chosen = ledger.loc[ledger.prediction_eligible].copy()
    probabilities = align_class_probabilities(probabilities, class_ids=CLASS_IDS)
    need(len(probabilities) == len(chosen), 'Probability rows differ from the complete signal-time cohort')
    targets = _labels(chosen.target_class, len(chosen))
    observed = np.isfinite(targets)
    need(np.array_equal(observed, chosen.target_available.to_numpy(dtype=bool)), 'Missing target mask changed at scoring')
    months = [f'{year}-{m:02d}' for m in range(1, 13)]
    need(sorted(chosen.outcome_month.unique()) == months, 'A validation year lost a complete month')
    mode = probabilities.argmax(axis=1) + 1
    true_p, losses, brier, correct = [np.full(len(chosen), np.nan) for _ in range(4)]
    ids = targets[observed].astype(int)
    selected = probabilities[observed]
    true_p[observed] = selected[np.arange(len(ids)), ids - 1]
    losses[observed] = -np.log(np.maximum(true_p[observed], 1e-15))
    errors = selected.copy()
    errors[np.arange(len(ids)), ids - 1] -= 1
    brier[observed] = np.square(errors).sum(axis=1)
    correct[observed] = mode[observed] == ids
    rows = chosen.reset_index()[['source_security_id', 'signal_date', 'outcome_month', 'outcome_end']]
    rows['true_class'] = pd.array(targets, dtype='Int64')
    rows['label_available'], rows['predicted_class'] = observed, mode
    rows['top_probability'], rows['true_class_probability'] = probabilities.max(axis=1), true_p
    rows['log_loss'], rows['brier_sum10'], rows['correct'] = losses, brier, correct
    rows['log_loss_probability_floored'] = observed & (true_p < 1e-15)
    monthly, scores = [], []
    for month in months:
        positions = chosen.outcome_month.to_numpy() == month
        scored = positions & observed
        need(scored.any(), 'Every validation month requires at least one observed label')
        loss = float(losses[scored].mean())
        score_pin = _key_hash(chosen.index[scored])
        scores.append({'outcome_month': month, 'log_loss': loss, 'score_key_sha256': score_pin})
        monthly.append({'outcome_month': month, 'prediction_rows': int(positions.sum()),
            'labelled_rows': int(scored.sum()), 'missing_label_rows': int((positions & ~observed).sum()),
            'label_coverage': float(scored.sum() / positions.sum()), 'log_loss': loss,
            'brier_sum10': float(brier[scored].mean()), 'accuracy': float(correct[scored].mean()),
            'zero_true_class_probability_rows': int(np.sum(positions & (true_p == 0))),
            'floored_true_class_probability_rows': int(np.sum(positions & (true_p < 1e-15))),
            'score_key_sha256': score_pin, 'status': 'available'})
    monthly = pd.DataFrame(monthly)
    summary = {'prediction_rows': len(chosen), 'labelled_rows': int(observed.sum()),
        'missing_label_rows': int((~observed).sum()), 'outcome_months': 12, 'observed_label_months': 12,
        'label_coverage': float(observed.mean()),
        'mean_monthly_log_loss': float(monthly.log_loss.mean()),
        'mean_monthly_brier_sum10': float(monthly.brier_sum10.mean()),
        'mean_monthly_accuracy': float(monthly.accuracy.mean()),
        'pooled_log_loss': float(losses[observed].mean()),
        'pooled_brier_sum10': float(brier[observed].mean()),
        'pooled_accuracy': float(correct[observed].mean()),
        'zero_true_class_probability_rows': int(np.sum(true_p == 0)),
        'floored_true_class_probability_rows': int(np.sum(observed & (true_p < 1e-15))),
        'log_loss_evaluation_floor': 1e-15,
        'hard_mode_probability_tie_rule': 'lowest_canonical_class_id',
        'primary_aggregation': 'equal_weight_mean_of_observed_monthly_means',
        'label_imputation_performed': False, 'probability_clipping_or_renormalization': False}
    return {'summary': summary, 'monthly_log_losses': scores, 'tables': {
        'row_scores': rows, 'monthly': monthly, 'classwise_calibration': _calibration(selected, ids, mode[observed], top_label=False),
        'toplabel_calibration': _calibration(selected, ids, mode[observed], top_label=True),
        'extreme_precision': _extremes(rows), 'mode_class_performance': _mode_classes(rows), 'confusion': _confusion(rows)}}


def _save_prediction_artifacts(output, ledger, probabilities, *, year):
    prediction_index = ledger.index[ledger.prediction_eligible]
    predictions = _predictions(ledger, probabilities, prediction_index)
    evaluation = evaluate_tuning_probabilities(probabilities, ledger, year=year)
    artifacts = {'probabilities': _save_array(output / 'probabilities.npy', probabilities)}
    artifacts['predictions'] = _save_frame(output / 'predictions.parquet', predictions)
    reports = output / 'reports'
    reports.mkdir()
    for name, table in evaluation['tables'].items():
        artifacts[name] = _save_frame(reports / (name + '.parquet'), table)
    artifacts['summary'] = write_json(reports / 'summary.json', evaluation['summary'])
    return artifacts, evaluation


def _validate_estimator_classes(estimator):
    classes = list(estimator.classes_)
    need(len(classes) == 10 and all(isinstance(k, Integral) and not isinstance(k, (bool, np.bool_)) for k in classes)
         and set(classes) == set(CLASS_IDS), 'Forest must retain all ten native canonical classes; no zero padding')
    return [int(k) for k in classes]


def _package_guard(protocol):
    runtime = protocol['runtime_snapshot_for_future_matching']
    for package in ('numpy', 'pandas', 'pyarrow', 'scikit-learn', 'scipy', 'joblib', 'threadpoolctl', 'exchange-calendars'):
        need(version(package) == runtime[package], 'Runtime differs from frozen tuning snapshot: ' + package)


def fit_one(request):
    _, run, protocol, _, stage = check_authorization(request)
    prepared = load_prepared(request)
    need(stage == prepared['stage'], 'Fit stage changed from prepared partition')
    if stage == 'outer':
        _winner(request)
    _package_guard(protocol)
    candidate, parameters = _candidate(protocol, request['candidate_id'])
    output = _fresh_output(request)
    train = np.load(verify_pin(prepared['artifacts']['train_X'], within=run), mmap_mode='r', allow_pickle=False)
    labels = np.load(verify_pin(prepared['artifacts']['train_y'], within=run), mmap_mode='r', allow_pickle=False)
    need(train.dtype == np.float32 and train.ndim == 2 and np.isfinite(train).all()
         and train.shape == (prepared['train_rows'], prepared['matrix_width']),
         'Prepared training matrix changed or is not finite float32')
    labels = _training_labels(labels, len(train))
    from sklearn.ensemble import RandomForestClassifier
    from joblib import parallel_backend, dump
    from threadpoolctl import threadpool_limits
    estimator = RandomForestClassifier(**parameters)
    started = time.monotonic()
    # The supervisor persists the consumed attempt slot before launching here.
    write_json(output / 'fit_started.json', {'candidate_id': candidate['candidate_id'],
        'split_id': prepared['split_id'], 'fit_count': 1, 'prepare_manifest': request['prepare_manifest']})
    with threadpool_limits(limits=1), parallel_backend('threading', n_jobs=4):
        estimator.fit(train, labels)
    fit_seconds = time.monotonic() - started
    native = _validate_estimator_classes(estimator)
    fitted_parameters = _json_parameters(estimator.get_params(deep=False))
    # Single-thread inference/reproduction is deterministic in accumulation order.
    estimator.set_params(n_jobs=1)
    temporary = output / 'model.joblib.partial'
    with temporary.open('xb') as stream:
        dump(estimator, stream, compress=0)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.rename(output / 'model.joblib')
    artifacts = {'model': pin(output / 'model.joblib'), 'fit_started': pin(output / 'fit_started.json')}
    manifest = {'schema': 'dmv2-rf-tuning-fit/v1', 'status': 'COMPLETE',
        'protocol_id': PROTOCOL_ID, 'outer_year': request['outer_year'], 'split_id': prepared['split_id'],
        'stage': stage, 'candidate_id': candidate['candidate_id'], 'prepare_manifest': request['prepare_manifest'],
        'artifacts': artifacts, 'estimator_fits': 1, 'preprocessing_fits': 0,
        'matrix_width': prepared['matrix_width'],
        'effective_feature_count': nominal_feature_count(parameters['max_features'], prepared['matrix_width']),
        'fit_seconds': fit_seconds, 'explicit_parameters': parameters, 'fit_effective_parameters': fitted_parameters,
        'saved_inference_effective_parameters': _json_parameters(estimator.get_params(deep=False)),
        'native_class_ids': native, 'probability_class_ids': list(CLASS_IDS),
        'probability_dtype': 'float64', 'inference_threads': 1, 'fit_tree_threads': 4,
        'validation_labels_supplied_to_estimator': False, 'sample_weight': None,
        'runtime': protocol['runtime_snapshot_for_future_matching']}
    evaluation = None
    if stage == 'inner':
        prediction = np.load(verify_pin(prepared['artifacts']['prediction_X'], within=run), mmap_mode='r', allow_pickle=False)
        need(prediction.dtype == np.float32 and prediction.shape == (prepared['prediction_rows'], prepared['matrix_width'])
             and np.isfinite(prediction).all(), 'Prepared prediction matrix changed')
        ledger = pd.read_parquet(verify_pin(prepared['artifacts']['prediction_ledger'], within=run)).set_index(list(KEYS))
        need(_key_hash(ledger.index[ledger.prediction_eligible]) == prepared['prediction_keys_sha256'],
             'Prepared prediction key order changed')
        with threadpool_limits(limits=1):
            probabilities = align_class_probabilities(estimator.predict_proba(prediction), class_ids=native)
        saved, evaluation = _save_prediction_artifacts(output, ledger, probabilities, year=prepared['validation_year'])
        artifacts.update(saved)
        manifest['monthly_log_losses'] = evaluation['monthly_log_losses']
        manifest['summary'] = evaluation['summary']
    manifest_pin = write_json(output / 'fit_manifest.json', manifest)
    return {'status': 'COMPLETE', 'action': 'fit', 'estimator_fits': 1, 'preprocessing_fits': 0,
        'fit_manifest': manifest_pin, 'artifacts': [*artifacts.values(), manifest_pin],
        'effective_feature_count': manifest['effective_feature_count'], 'matrix_width': manifest['matrix_width'],
        'monthly_log_losses': [] if evaluation is None else evaluation['monthly_log_losses'],
        'fit_seconds': fit_seconds, 'projected_source_read_bytes': 0}


def predict_outer(request):
    _, run, protocol, _, stage = check_authorization(request)
    _winner(request)
    _package_guard(protocol)
    fit_path = verify_pin(request['fit_manifest'], within=run)
    fit = read_json(fit_path)
    need(fit.get('status') == 'COMPLETE' and fit.get('stage') == 'outer'
         and fit.get('candidate_id') == request['candidate_id'] and fit.get('outer_year') == request['outer_year']
         and fit.get('prepare_manifest') == request['prepare_manifest'], 'Selected outer model binding changed')
    matrix, ledger, part, plan = prepare_outer_prediction(request)
    output = _fresh_output(request)
    from joblib import load
    from threadpoolctl import threadpool_limits
    estimator = load(verify_pin(fit['artifacts']['model'], within=run))
    need(estimator.get_params(deep=False)['n_jobs'] == 1, 'Saved model must reproduce with one inference thread')
    native = _validate_estimator_classes(estimator)
    with threadpool_limits(limits=1):
        probabilities = align_class_probabilities(estimator.predict_proba(matrix), class_ids=native)
    artifacts, evaluation = _save_prediction_artifacts(output, ledger, probabilities, year=request['outer_year'])
    artifacts['prediction_X'] = _save_array(output / 'prediction_X.npy', matrix)
    artifacts['prediction_ledger'] = _save_frame(output / 'prediction_ledger.parquet', ledger.reset_index())
    manifest = {'schema': 'dmv2-rf-tuning-outer-prediction/v1', 'status': 'COMPLETE', 'protocol_id': PROTOCOL_ID,
        'outer_year': request['outer_year'], 'split_id': request['split_id'], 'candidate_id': request['candidate_id'],
        'winner_path': request['winner_path'], 'prepare_manifest': request['prepare_manifest'],
        'fit_manifest': request['fit_manifest'], 'plan': request['plan_path'], 'artifacts': artifacts,
        'prediction_keys_sha256': _key_hash(ledger.index[ledger.prediction_eligible]),
        'prediction_original_keys_sha256': _key_hash(ledger.index),
        'score_keys_sha256': _key_hash(ledger.index[ledger.prediction_eligible & ledger.target_available]),
        'prediction_provenance': part.provenance, 'summary': evaluation['summary'],
        'monthly_log_losses': evaluation['monthly_log_losses'], 'estimator_fits': 0, 'preprocessing_fits': 0,
        'projected_source_read_bytes': plan['projected_source_read_bytes']}
    manifest_pin = write_json(output / 'prediction_manifest.json', manifest)
    return {'status': 'COMPLETE', 'action': 'predict_outer', 'estimator_fits': 0, 'preprocessing_fits': 0,
            'prediction_manifest': manifest_pin, 'artifacts': [*artifacts.values(), manifest_pin],
            'projected_source_read_bytes': plan['projected_source_read_bytes']}


def execute_request(request):
    result = _execute_action(request)
    result.update({key: request[key] for key in ('action', 'outer_year', 'split_id', 'candidate_id') if key in request})
    return result


def _execute_action(request):
    action = request.get('action')
    if action == 'plan':
        plan = plan_action(request)
        output = _fresh_output(request)
        plan_pin = write_json(output / 'plan.json', plan)
        return {'status': 'COMPLETE', 'action': 'plan', 'plan': plan, 'plan_path': plan_pin,
                'artifacts': [plan_pin], 'estimator_fits': 0, 'preprocessing_fits': 0,
                'projected_source_read_bytes': 0}
    if action == 'prepare':
        return prepare_split(request)
    if action == 'fit':
        return fit_one(request)
    if action == 'predict_outer':
        return predict_outer(request)
    raise ValueError('Unknown explicit RF tuning child action')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', required=True)
    parser.add_argument('--receipt', required=True)
    args = parser.parse_args(argv)
    started = time.monotonic()
    receipt = execute_request(read_json(args.request))
    receipt['wall_seconds'] = time.monotonic() - started
    write_json(args.receipt, receipt)


if __name__ == '__main__':
    main()
