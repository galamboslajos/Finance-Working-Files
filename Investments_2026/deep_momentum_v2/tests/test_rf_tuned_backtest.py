"""Synthetic adapter, scope, source-parent and predecode guard regressions."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STAGING = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('dmv2.rf_tuned_backtest', STAGING / 'src/dmv2/rf_tuned_backtest.py')
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def metadata(year=2020):
    _, windows, _ = m.walkforward_clock()
    frame = windows.loc[windows.outcome_month.str.startswith(str(year) + '-'),
                        ['signal_date', 'outcome_month', 'outcome_start', 'outcome_end']].copy()
    frame['source_security_id'] = 'source-one'
    frame['intrinio_security_id'] = 'native-one'
    for name in ('identity_eligible', 'paper_prediction_eligible', 'prediction_eligible'):
        frame[name] = True
    return frame.reset_index(drop=True), windows


class SourceParentTests(unittest.TestCase):
    def test_relative_and_absolute_same_pin_are_accepted(self):
        root = Path('/private/tmp/test')
        actual = {'path': '/private/tmp/test/years/2020/audit.json', 'sha256': 'old', 'size_bytes': 4}
        frozen = [{'path': 'years/2020/audit.json', 'sha256': 'old', 'size_bytes': 4}]
        m.require_frozen_pin(root, actual, frozen)

    def test_rewritten_daily_and_selfhashed_child_audit_cannot_promote(self):
        root = Path('/private/tmp/test')
        frozen = [{'path': 'years/2020/audit.json', 'sha256': 'original', 'size_bytes': 12},
                  {'path': 'years/2020/daily_prices_raw.parquet', 'sha256': 'original-data', 'size_bytes': 20}]
        for path, sha, size in [('years/2020/audit.json', 'new-selfhash', 12),
                                ('years/2020/daily_prices_raw.parquet', 'new-data-selfhash', 20)]:
            with self.subTest(path=path), self.assertRaises(m.DataContractError):
                m.require_frozen_pin(root, {'path': path, 'sha256': sha, 'size_bytes': size}, frozen)

    def test_unregistered_or_conflicting_parent_pin_is_refused(self):
        actual = {'path': 'audit.json', 'sha256': 'new', 'size_bytes': 10}
        for frozen in ([], [actual, {**actual, 'sha256': 'other'}]):
            with self.subTest(frozen=frozen), self.assertRaises(m.DataContractError):
                m.require_frozen_pin(Path('/private/tmp'), actual, frozen)


class ForecastGuardTests(unittest.TestCase):
    def test_full_independent_calendar_is_required(self):
        frame, windows = metadata()
        m.validate_forecast_metadata(frame, 2020, windows)
        for changed in (frame.iloc[:-1], pd.concat([frame, frame.iloc[:1]])):
            with self.assertRaises(m.DataContractError):
                m.validate_forecast_metadata(changed, 2020, windows)

    def test_target_overlap_wrong_month_and_noncanonical_eligibility_fail(self):
        frame, windows = metadata()
        for field, value in [('signal_date', '2024-01-31'), ('outcome_month', '2024-02'),
                             ('outcome_end', '2024-02-29'), ('prediction_eligible', 'True'),
                             ('intrinio_security_id', None)]:
            changed = frame.astype({field: object}).copy()
            changed.loc[0, field] = value
            with self.subTest(field=field), self.assertRaises(m.DataContractError):
                m.validate_forecast_metadata(changed, 2020, windows)

    def test_extra_predictors_cannot_narrow_baseline_prediction_cohort(self):
        frame, windows = metadata()
        frame.loc[0, 'prediction_eligible'] = False
        with self.assertRaises(m.DataContractError): m.validate_forecast_metadata(frame, 2020, windows)

    def test_blank_or_nonstring_source_id_is_refused_before_values(self):
        frame, windows = metadata()
        for value in ('', '   ', 123):
            changed = frame.astype({'source_security_id': object}).copy()
            changed.loc[0, 'source_security_id'] = value
            with self.subTest(value=value), self.assertRaises(m.DataContractError):
                m.validate_forecast_metadata(changed, 2020, windows)

    def test_eligible_dated_native_collision_is_refused(self):
        frame, windows = metadata()
        duplicate = frame.iloc[[0]].copy(); duplicate['source_security_id'] = 'different-source'
        with self.assertRaises(m.DataContractError):
            m.validate_forecast_metadata(pd.concat([frame, duplicate]), 2020, windows)

    def test_bad_dates_stop_before_any_probability_conversion(self):
        frame, windows = metadata(); frame.loc[0, 'signal_date'] = '2024-01-31'
        reads = []
        class File:
            def read(self, columns):
                reads.append(columns)
                if any(name.startswith('probability_') for name in columns):
                    raise AssertionError('Protected numeric conversion occurred')
                return Mock(to_pandas=lambda: frame)
        entry = {'outer_year': 2020, 'predictions_pin': {'path': '/private/tmp/new'},
                 'original_rf_predictions_pin': {'path': '/private/tmp/old'},
                 'metadata_columns': list(frame)}
        with patch.object(m.pq, 'ParquetFile', return_value=File()), self.assertRaises(m.DataContractError):
            m.load_forecasts([entry], windows)
        self.assertEqual(len(reads), 1)

    def test_native_cohort_change_stops_before_probabilities(self):
        frame, windows = metadata(); old = frame.copy(); old.loc[0, 'intrinio_security_id'] = 'different-native'
        reads = []
        class File:
            def __init__(self, path): self.path = path
            def read(self, columns):
                reads.append((self.path, columns))
                if any(name.startswith('probability_') for name in columns):
                    raise AssertionError('Numeric conversion before original cohort check')
                return Mock(to_pandas=lambda: frame if self.path.endswith('new') else old)
        entry = {'outer_year': 2020, 'predictions_pin': {'path': '/private/tmp/new'},
                 'original_rf_predictions_pin': {'path': '/private/tmp/old'},
                 'metadata_columns': list(frame)}
        with patch.object(m.pq, 'ParquetFile', side_effect=File), self.assertRaises(AssertionError):
            m.load_forecasts([entry], windows)
        self.assertEqual(len(reads), 2)

    def test_forecast_projection_excludes_targets_and_outcomes(self):
        names = [*m.COHORT_COLUMNS, *m.PROBABILITIES, 'target_class', 'target_available', 'realized_return']
        pf = Mock(schema_arrow=Mock(names=names), metadata=Mock(num_rows=12))
        identity = {'forecast_entries': [{'outer_year': 2020, 'predictions_pin': {'path': '/private/tmp/new'},
                                        'original_rf_predictions_pin': {'path': '/private/tmp/old'}}]}
        with patch.object(m.pq, 'ParquetFile', return_value=pf), patch.object(m, '_projected_bytes', return_value=3):
            plan, amount = m.forecast_footer_plan(identity)
        self.assertEqual(amount, 9)
        self.assertNotIn('target_class', plan[0]['metadata_columns'])
        self.assertNotIn('target_available', plan[0]['metadata_columns'])
        self.assertNotIn('realized_return', plan[0]['metadata_columns'])

    def test_full_four_year_result_has_unique_43655_original_rows(self):
        # A successful first-year conversion still cannot return a truncated
        # requested subset as a supposedly full research panel.
        frame, windows = metadata()
        probability = pd.DataFrame({name: [0.1] * 12 for name in m.PROBABILITIES})
        class File:
            def read(self, columns):
                return Mock(to_pandas=lambda: probability if columns == list(m.PROBABILITIES) else frame.loc[:, columns])
        entry = {'outer_year': 2020, 'predictions_pin': {'path': '/private/tmp/new'},
                 'original_rf_predictions_pin': {'path': '/private/tmp/old'}, 'metadata_columns': list(frame),
                 'training_outcome_start': '2010-01-01', 'training_outcome_end': '2019-12-31',
                 'training_label_observed_through': '2019-12-31'}
        with patch.object(m.pq, 'ParquetFile', return_value=File()), self.assertRaises(m.DataContractError):
            m.load_forecasts([entry], windows)


class BudgetCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(); self.run = Path(self.temp.name)
        m.write_json(self.run / 'read_usage.json', {'schema': 'dmv2-rf-tuned-read-usage/v1',
            'limit_bytes': 100, 'reservations': [], 'total_projected_compressed_bytes': 0})
    def tearDown(self): self.temp.cleanup()
    def test_cumulative_repeat_reads_are_charged(self):
        m.reserve_reads(self.run, 'preflight', 30, limit=100)
        m.reserve_reads(self.run, 'backtest', 40, limit=100)
        actual = m.reserve_reads(self.run, 'independent_verification', 25, limit=100)
        self.assertEqual(actual['total_projected_compressed_bytes'], 95)
    def test_exhaustion_stops_before_mutating_prior_ledger(self):
        m.reserve_reads(self.run, 'preflight', 90, limit=100)
        before = (self.run / 'read_usage.json').read_bytes()
        with self.assertRaises(m.DataContractError): m.reserve_reads(self.run, 'backtest', 11, limit=100)
        self.assertEqual(before, (self.run / 'read_usage.json').read_bytes())
    def test_same_phase_cannot_silently_retry_or_reset_budget(self):
        m.reserve_reads(self.run, 'preflight', 30, limit=100)
        with self.assertRaises(m.DataContractError): m.reserve_reads(self.run, 'preflight', 1, limit=100)
    def test_tampered_counter_is_refused(self):
        m.write_json(self.run / 'read_usage.json', {'schema': 'dmv2-rf-tuned-read-usage/v1',
            'limit_bytes': 100, 'reservations': [], 'total_projected_compressed_bytes': 5}, replace=True)
        with self.assertRaises(m.DataContractError): m.reserve_reads(self.run, 'preflight', 1, limit=100)
    def test_existing_checkpoint_or_partial_is_preserved(self):
        path = self.run / 'saved.json'; m.write_json(path, {'original': True})
        with self.assertRaises(m.DataContractError): m.write_json(path, {'original': False})
        self.assertTrue(m.read_json(path)['original'])
        (self.run / 'pending.json.partial').write_text('interrupted')
        with self.assertRaises(m.DataContractError): m.write_json(self.run / 'pending.json', {})


class FrozenScopeTests(unittest.TestCase):
    def test_exact_sixteen_scenarios_and_no_estimator_entrypoints(self):
        self.assertEqual(len(set(m.SCENARIOS)), 16)
        self.assertEqual({cost for _, _, cost in m.SCENARIOS}, {0, 5, 10, 25})
        self.assertEqual({b for b, _, _ in m.SCENARIOS}, {'class_1_10', 'class_2_9'})
        self.assertEqual({v for _, v, _ in m.SCENARIOS}, {'unstopped', 'asset_stop_10pct'})
        source = (STAGING / 'src/dmv2/rf_tuned_backtest.py').read_text()
        for forbidden in ('.fit(', 'predict_proba(', 'joblib.load(', 'pickle.load(', 'requests.get('):
            self.assertNotIn(forbidden, source)
    def test_frozen_configuration_has_separate_authority_and_headroom(self):
        cfg = json.loads((STAGING / 'configs/rf_expanded_tuned_backtest_20261020.json').read_text())
        self.assertTrue(cfg['execution_authorized'])
        self.assertEqual(cfg['resource_limits']['rss_stop_bytes'], 4 * 1024**3)
        self.assertEqual(cfg['resource_limits']['rss_warning_bytes'], 3 * 1024**3)
        self.assertEqual(cfg['new_estimator_fits'], 0)
        self.assertEqual(cfg['new_inference_calls'], 0)


if __name__ == '__main__': unittest.main()
