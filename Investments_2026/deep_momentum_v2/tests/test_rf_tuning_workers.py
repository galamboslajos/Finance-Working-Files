"""Synthetic contracts only: no real financial panels and no estimator fitting."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from dmv2.catalog import canonical_sha256
from dmv2.features import TrainingSlice, KEYS, registered_feature_specs
from dmv2.paper_features import PAPER_FEATURE_NAMES
from dmv2.preprocessing import fit_preprocessor
from dmv2.rf_tuning_data import (PROTOCOL_ID, _settings, check_authorization,
    pin, prepare_split, validate_partition_clocks, verify_pin, write_json)
from dmv2.rf_tuning_fit import (evaluate_tuning_probabilities,
    nominal_feature_count, _validate_estimator_classes, execute_request, fit_one)
from dmv2.universe import DataContractError


def ledger(year, names=3):
    rows = []
    for month in range(1, 13):
        period = pd.Period(f'{year}-{month:02d}', 'M')
        signal = str((period - 1).end_time.date())
        for position in range(names):
            rows.append({'source_security_id': f'S{position}', 'signal_date': signal,
                'outcome_month': str(period), 'outcome_start': str(period.start_time.date()),
                'outcome_end': str(period.end_time.date()), 'prediction_eligible': True,
                'paper_prediction_eligible': True, 'target_available': position != names - 1,
                'target_class': 1 + position if position != names - 1 else pd.NA})
    frame = pd.DataFrame(rows).set_index(list(KEYS))
    frame['target_class'] = pd.array(frame.target_class, dtype='Int64')
    return frame


def slice_for(year):
    frame = ledger(year)
    X = pd.DataFrame({name: np.arange(len(frame), dtype=float) + i
                      for i, name in enumerate(PAPER_FEATURE_NAMES)}, index=frame.index)
    return TrainingSlice(X, frame.target_class.astype(float), frame,
        {'train_start': f'{year}-01-01', 'train_end': f'{year}-12-31',
         'label_observed_through': f'{year}-12-31'})


class FakeForest:
    """A no-compute estimator double; no sklearn fitting is performed."""
    def __init__(self, **parameters):
        self.parameters = dict(parameters)
        self.fit_calls = 0

    def fit(self, X, y):
        self.fit_calls += 1
        self.classes_ = np.unique(y)
        self.fit_shape = X.shape
        return self

    def get_params(self, deep=False):
        return dict(self.parameters)

    def set_params(self, **parameters):
        self.parameters.update(parameters)
        return self

    def predict_proba(self, X):
        return np.full((len(X), 10), .1)


class WorkerContracts(unittest.TestCase):
    def test_equal_month_loss_missing_labels_and_native_values(self):
        frame = ledger(2017)
        probabilities = np.full((len(frame), 10), 0.1)
        result = evaluate_tuning_probabilities(probabilities, frame, year=2017)
        self.assertAlmostEqual(result['summary']['mean_monthly_log_loss'], np.log(10))
        self.assertEqual(result['summary']['missing_label_rows'], 12)
        self.assertEqual(result['summary']['prediction_rows'], 36)
        self.assertEqual(len(result['monthly_log_losses']), 12)
        np.testing.assert_array_equal(probabilities, np.full((36, 10), .1))
        self.assertEqual(set(result['tables']), {'row_scores', 'monthly', 'classwise_calibration',
            'toplabel_calibration', 'extreme_precision', 'mode_class_performance', 'confusion'})

    def test_new_scoring_adapter_matches_frozen_outer_metric_formulas(self):
        from dmv2.walkforward_metrics import evaluate_probabilities
        frame = ledger(2020)
        raw = np.random.default_rng(1729).uniform(.01, 1., (36, 10))
        probabilities = raw / raw.sum(axis=1, keepdims=True)
        ours = evaluate_tuning_probabilities(probabilities, frame, year=2020)
        old = evaluate_probabilities(probabilities, frame.target_class.reset_index(drop=True),
            class_ids=range(1, 11), row_metadata=frame.reset_index(),
            first_month='2020-01', last_month='2020-12')
        for name in ('mean_monthly_log_loss', 'mean_monthly_brier_sum10', 'mean_monthly_accuracy',
                     'pooled_log_loss', 'pooled_brier_sum10', 'pooled_accuracy'):
            self.assertAlmostEqual(ours['summary'][name], old.summary[name], places=14)
        for name in ('classwise_calibration', 'toplabel_calibration', 'extreme_precision',
                     'mode_class_performance', 'confusion'):
            pd.testing.assert_frame_equal(ours['tables'][name], old.tables[name])

    def test_floor_is_evaluation_only_and_mode_tie_lowest(self):
        frame = ledger(2020)
        probabilities = np.zeros((36, 10))
        probabilities[:, 3] = .5
        probabilities[:, 5] = .5
        result = evaluate_tuning_probabilities(probabilities, frame, year=2020)
        self.assertAlmostEqual(result['summary']['mean_monthly_log_loss'], -np.log(1e-15))
        self.assertTrue((result['tables']['row_scores'].predicted_class == 4).all())
        self.assertTrue((probabilities[:, 0] == 0).all())

    def test_missing_month_or_all_missing_month_labels_rejected(self):
        frame = ledger(2018)
        with self.assertRaises(DataContractError):
            evaluate_tuning_probabilities(np.full((33, 10), .1), frame.iloc[3:], year=2018)
        frame.loc[frame.outcome_month == '2018-02', 'target_available'] = False
        frame.loc[frame.outcome_month == '2018-02', 'target_class'] = pd.NA
        with self.assertRaises(DataContractError):
            evaluate_tuning_probabilities(np.full((36, 10), .1), frame, year=2018)

    def test_signal_cutoff_and_protected_year_rejected_before_scores(self):
        frame = ledger(2021)
        bad = frame.reset_index()
        bad.loc[0, 'signal_date'] = '2021-01-01'
        with self.assertRaises(DataContractError):
            evaluate_tuning_probabilities(np.full((36, 10), .1), bad.set_index(list(KEYS)), year=2021)
        with self.assertRaises(DataContractError):
            evaluate_tuning_probabilities(np.full((36, 10), .1), ledger(2024), year=2024)

    def test_masks_must_match_labels_and_probability_coverage(self):
        frame = ledger(2022)
        frame.iloc[0, frame.columns.get_loc('target_available')] = False
        with self.assertRaises(DataContractError):
            evaluate_tuning_probabilities(np.full((36, 10), .1), frame, year=2022)
        with self.assertRaises(DataContractError):
            evaluate_tuning_probabilities(np.full((35, 10), .1), ledger(2022), year=2022)

    def test_training_preprocessor_does_not_learn_validation_median(self):
        train, valid = slice_for(2016), slice_for(2017)
        name = next(s['name'] for s in registered_feature_specs()
                    if s.get('model_eligible', True) and s['name'] not in PAPER_FEATURE_NAMES)
        train.X[name] = np.nan
        train.X.iloc[0, train.X.columns.get_loc(name)] = 2.
        train.X.iloc[1, train.X.columns.get_loc(name)] = 4.
        valid.X[name] = np.nan
        fitted = fit_preprocessor(train, fit_mask=train.ledger.target_available)
        self.assertEqual(fitted.manifest['medians'][name], 3.)
        transformed = fitted.transform(valid.X, prediction_mask=valid.ledger.prediction_eligible)
        self.assertTrue(transformed.ledger.transformed.all())
        self.assertTrue((transformed.X[name] == 3.).all())
        self.assertTrue((transformed.X[name + '__missing'] == 1.).all())
        self.assertEqual(fitted.manifest['medians'][name], 3.)

    def test_temporal_partition_boundary_after_close_is_allowed(self):
        train, valid = slice_for(2016), slice_for(2017)
        validate_partition_clocks(train, valid)
        crossing = train.ledger.copy()
        crossing.iloc[-1, crossing.columns.get_loc('outcome_end')] = '2017-01-03'
        with self.assertRaises(DataContractError):
            validate_partition_clocks(replace(train, ledger=crossing), valid)

    def test_native_classes_no_zero_padding(self):
        class Good:
            classes_ = np.arange(1, 11)
        class Bad:
            classes_ = np.arange(1, 10)
        self.assertEqual(_validate_estimator_classes(Good()), list(range(1, 11)))
        with self.assertRaises(DataContractError):
            _validate_estimator_classes(Bad())
        self.assertEqual(nominal_feature_count('sqrt', 1430), 37)
        self.assertEqual(nominal_feature_count(.05, 1430), 71)

    def test_immutable_artifacts_and_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'state.json'
            original = write_json(path, {'data': 1})
            self.assertEqual(verify_pin(original, within=directory), path.resolve())
            with self.assertRaises(DataContractError):
                write_json(path, {'data': 2})
            path.write_text('{}')
            with self.assertRaises(DataContractError):
                verify_pin(original, within=directory)

    def test_no_implicit_action(self):
        with self.assertRaises(ValueError):
            execute_request({})

    def test_action_receipt_keeps_controller_identity(self):
        request = {'action': 'fit', 'outer_year': 2020, 'split_id': '2020_validate_2017',
                   'candidate_id': 'rf_d12_l020_sqrt'}
        with patch('dmv2.rf_tuning_fit._execute_action', return_value={'status': 'COMPLETE'}):
            result = execute_request(request)
        for key, value in request.items():
            self.assertEqual(result[key], value)

    def test_all_twelve_inner_and_four_outer_windows_are_explicit(self):
        folds = []
        for year in range(2020, 2024):
            folds.append({'outer_outcome_year': year, 'train_outcome_start': f'{year-10}-01-01',
                'train_outcome_end': f'{year-1}-12-31', 'inner_splits': [
                    {'split_id': f'{year}_validate_{valid}', 'train_outcome_start': f'{year-10}-01-01',
                     'train_outcome_end': f'{valid-1}-12-31', 'validation_outcome_start': f'{valid}-01-01',
                     'validation_outcome_end': f'{valid}-12-31'} for valid in range(year-3, year)]})
        protocol = {'protocol_id': PROTOCOL_ID, 'rf': {'fixed_parameters': {
            'n_estimators': 500, 'n_jobs': 4, 'random_state': 1729}},
            'planned_fit_counts': {'maximum_estimator_fits': 100}, 'chronology': {'folds': folds}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            path = root / 'protocol.json'
            path.write_text(json.dumps(protocol))
            for fold in folds:
                year = fold['outer_outcome_year']
                for i, split in enumerate(fold['inner_splits']):
                    request = {'root': str(root), 'run_root': str(root / 'outputs' / 'attempt'),
                        'protocol_path': str(path), 'outer_year': year, 'split_id': split['split_id']}
                    _, _, _, actual, stage = _settings(request)
                    self.assertEqual(stage, 'inner')
                    self.assertEqual(int(actual['train_outcome_end'][:4]) - int(actual['train_outcome_start'][:4]) + 1, 7 + i)
                request['split_id'] = f'{year}_outer_train'
                _, _, _, actual, stage = _settings(request)
                self.assertEqual(stage, 'outer')
                self.assertEqual(int(actual['train_outcome_end'][:4]) - int(actual['train_outcome_start'][:4]) + 1, 10)

    def test_worker_receipt_passes_controller_integrity_contract(self):
        from dmv2.rf_tuning_control import validate_receipt
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory).resolve()
            item = write_json(run / 'manifest.json', {'status': 'COMPLETE'})
            request = {'action': 'prepare', 'outer_year': 2020, 'split_id': '2020_validate_2017'}
            with patch('dmv2.rf_tuning_fit._execute_action', return_value={
                'status': 'COMPLETE', 'prepare_manifest': item, 'artifacts': [item]}):
                receipt = execute_request(request)
            validate_receipt(run, request, receipt)

    def test_full_synthetic_prepare_shared_missingness_and_single_fit_artifacts(self):
        from dmv2.rf_tuning_control import validate_receipt
        columns = [s['name'] for s in registered_feature_specs() if s.get('model_eligible', True)]
        self.assertEqual(len(columns), 715)
        extra = next(name for name in columns if name not in PAPER_FEATURE_NAMES)
        def annual(year):
            frame = ledger(year, names=11)
            values = np.arange(len(frame), dtype=float)
            X = pd.DataFrame({name: values + i for i, name in enumerate(columns)}, index=frame.index)
            X[extra] = np.nan  # training-only drop, without dropping any prediction row
            return TrainingSlice(X, frame.target_class.astype(float), frame,
                {'train_start': f'{year}-01-01', 'train_end': f'{year}-12-31',
                 'label_observed_through': f'{year}-12-31', 'source_month_target_fingerprints': {
                    f'{year}-{m:02d}': f'fake-{year}-{m}' for m in range(1, 13)}})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            run = root / 'outputs' / 'attempt'
            run.mkdir(parents=True)
            plan = {'stage': 'inner', 'outer_year': 2020, 'split_id': '2020_validate_2017',
                'validation_year': 2017, 'train_start': '2010-01-01', 'train_end': '2016-12-31',
                'annuals': [{'year': y} for y in range(2010, 2018)], 'projected_source_read_bytes': 123}
            request = {'action': 'prepare', 'root': str(root), 'run_root': str(run),
                'output_dir': str(run / 'prepared'), 'plan_path': write_json(run / 'plan.json', plan),
                'outer_year': 2020, 'split_id': '2020_validate_2017'}
            with patch('dmv2.rf_tuning_data.authorized_plan', return_value=(root, run, {}, plan)), \
                    patch('dmv2.rf_tuning_data._load_annual', side_effect=lambda _, item: annual(item['year'])), \
                    patch('dmv2.rf_tuning_data._completed_exchange_cutoff', return_value='2016-12-31'):
                receipt = execute_request(request)
            validate_receipt(run, request, receipt)
            prepared = json.loads(Path(receipt['prepare_manifest']['path']).read_text())
            self.assertEqual(prepared['preprocessing_fits'], 1)
            self.assertEqual(prepared['prediction_rows'], 132)
            self.assertEqual(prepared['matrix_width'], 1428)
            saved_ledger = pd.read_parquet(prepared['artifacts']['prediction_ledger']['path'])
            self.assertEqual(len(saved_ledger), 132)
            self.assertEqual(int(saved_ledger.target_class.isna().sum()), 12)
            parameters = {'n_estimators': 500, 'n_jobs': 4, 'random_state': 1729,
                'class_weight': None, 'bootstrap': True, 'oob_score': False, 'warm_start': False}
            protocol = {'rf': {'fixed_parameters': parameters, 'candidates': [
                {'candidate_id': 'rf_d12_l020_sqrt', 'max_depth': 12, 'min_samples_leaf': 20, 'max_features': 'sqrt'}]},
                'runtime_snapshot_for_future_matching': {}}
            fit_request = {**request, 'action': 'fit', 'candidate_id': 'rf_d12_l020_sqrt',
                'output_dir': str(run / 'fit'), 'prepare_manifest': receipt['prepare_manifest']}
            with patch('dmv2.rf_tuning_fit.check_authorization', return_value=(root, run, protocol, {}, 'inner')), \
                    patch('dmv2.rf_tuning_fit.load_prepared', return_value=prepared), \
                    patch('dmv2.rf_tuning_fit._package_guard'), \
                    patch('sklearn.ensemble.RandomForestClassifier', FakeForest):
                fitted = execute_request(fit_request)
            validate_receipt(run, fit_request, fitted)
            self.assertEqual(fitted['estimator_fits'], 1)
            self.assertEqual(len(fitted['monthly_log_losses']), 12)
            manifest = json.loads(Path(fitted['fit_manifest']['path']).read_text())
            self.assertEqual(manifest['fit_effective_parameters']['n_jobs'], 4)
            self.assertEqual(manifest['saved_inference_effective_parameters']['n_jobs'], 1)


if __name__ == '__main__':
    unittest.main()
