"""Synthetic RET execution boundaries; no research data or estimator activity."""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from dmv2 import ret_backtest as m
from dmv2.catalog import canonical_sha256
from dmv2.monthly import _step_month
from dmv2.ret import verify_ret_payoffs
from dmv2.universe import DataContractError
from dmv2.walkforward_clock import walkforward_clock


def means_fixture():
    means, windows, pins = {}, {}, {}
    for position in range(167):
        month = _step_month('2010-01', position)
        end = str(pd.Period(month, freq='M').end_time.date())
        window = {'signal_date': str((pd.Period(month, freq='M') - 1).end_time.date()),
            'outcome_month': month, 'outcome_start': month + '-01', 'outcome_end': end}
        means[month] = [{'outcome_month': month, 'outcome_end': end, 'class_id': k,
            'original_member_count': 40, 'prediction_eligible_count': 30, 'member_count': k,
            'class_mean_return': .01 * (11-k) + position * 1e-5, 'status': 'available'} for k in range(1, 11)]
        windows[month] = window; pins[month] = 'sha256:' + str(position).zfill(64)
    return {'monthly': means, 'windows': windows, 'target_fingerprints': pins,
        'diagnostic_fingerprints': dict(pins), 'source_target_policy_sha256': canonical_sha256({'test': True})}


def signals_fixture():
    # The payoff worker only needs a completed calendar endpoint. Use the exact
    # independent trading clock and move synthetic completion evidence to it.
    return list(walkforward_clock()[1].signal_date)


def completed_means_fixture():
    result = means_fixture()
    for signal in signals_fixture():
        month = signal[:7]
        result['windows'][month]['outcome_end'] = signal
        for row in result['monthly'][month]: row['outcome_end'] = signal
    return result


class RetPayoffConstructionTests(unittest.TestCase):
    def test_exact_shared_48_by_120_by_10_grid(self):
        payoffs = m.build_shared_payoffs(completed_means_fixture(), signals_fixture())
        self.assertEqual(len(payoffs), 48)
        self.assertEqual(sum(p.monthly.num_rows for p in payoffs.values()), 57600)
        self.assertTrue(all(verify_ret_payoffs(p) for p in payoffs.values()))
        first = payoffs[signals_fixture()[0]]
        self.assertEqual(first.provenance['first_outcome_month'], '2010-01')
        self.assertEqual(first.provenance['last_outcome_month'], '2019-12')

    def test_equal_month_weights_not_stock_row_weights(self):
        means = completed_means_fixture(); signal = signals_fixture()[0]
        means['monthly']['2010-01'][0]['member_count'] = 10000
        means['monthly']['2010-01'][0]['class_mean_return'] = .9
        payoffs = m.build_shared_payoffs(means, signals_fixture())
        expected = math.fsum(means['monthly'][_step_month('2010-01', i)][0]['class_mean_return'] / 120 for i in range(120))
        self.assertEqual(payoffs[signal].table.to_pylist()[0]['payoff'], expected)

    def test_newly_completed_month_cannot_change_previous_signal(self):
        means = completed_means_fixture(); before = m.build_shared_payoffs(means, signals_fixture())
        for row in means['monthly']['2020-01']: row['class_mean_return'] += 100
        after = m.build_shared_payoffs(means, signals_fixture())
        first, second = signals_fixture()[:2]
        self.assertEqual(before[first].fingerprint, after[first].fingerprint)
        self.assertNotEqual(before[second].fingerprint, after[second].fingerprint)

    def test_empty_class_withholds_whole_vector(self):
        means = completed_means_fixture()
        means['monthly']['2010-01'][0].update(member_count=0, class_mean_return=None, status='empty_realized_class')
        payoffs = m.build_shared_payoffs(means, signals_fixture()); first, second = signals_fixture()[:2]
        self.assertFalse(payoffs[first].provenance['vector_available'])
        self.assertTrue(all(row['payoff'] is None for row in payoffs[first].table.to_pylist()))
        self.assertTrue(payoffs[second].provenance['vector_available'])

    def test_missing_calendar_month_cannot_compress_window(self):
        means = completed_means_fixture(); del means['monthly']['2010-02']
        with self.assertRaisesRegex(DataContractError, 'actual calendar month'):
            m.build_shared_payoffs(means, signals_fixture())

    def test_uncompleted_outcome_is_rejected(self):
        means = completed_means_fixture()
        for row in means['monthly']['2019-12']: row['outcome_end'] = '2020-01-01'
        with self.assertRaisesRegex(DataContractError, 'Future outcome'):
            m.build_shared_payoffs(means, signals_fixture())

    def test_duplicate_or_missing_signal_cannot_shrink_states(self):
        with self.assertRaisesRegex(DataContractError, 'all48'):
            m.build_shared_payoffs(completed_means_fixture(), signals_fixture()[:-1])

    def test_protected_annual_rejected_before_state_or_parquet_read(self):
        with patch.object(m, 'authenticated', side_effect=AssertionError('must not open financial/state source')):
            with self.assertRaisesRegex(DataContractError, 'Protected annual'):
                m.load_monthly_class_means(Path('/synthetic'), [{'year': 2024}])


class RetReadLedgerTests(unittest.TestCase):
    def ledger(self, directory):
        state = {'schema': 'dmv2-ret-read-usage/v1', 'limit_bytes': m.LIMITS['cumulative_compressed_read_bytes'],
            'total_projected_compressed_bytes': 0, 'reservations': []}
        (directory / 'read_usage.json').write_text(json.dumps(state))
        return state

    def test_reservation_charged_and_repeated_phase_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp); self.ledger(directory)
            result = m.reserve_reads(directory, 'preflight', 123)
            self.assertEqual(result['total_projected_compressed_bytes'], 123)
            with self.assertRaisesRegex(DataContractError, 'automatic phase retries'):
                m.reserve_reads(directory, 'preflight', 1)
            self.assertEqual(json.loads((directory / 'read_usage.json').read_text())['total_projected_compressed_bytes'], 123)

    def test_budget_failure_does_not_erase_prior_charge(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp); self.ledger(directory)
            m.reserve_reads(directory, 'preflight', 17)
            with self.assertRaisesRegex(DataContractError, 'before decoding'):
                m.reserve_reads(directory, 'backtest', m.LIMITS['cumulative_compressed_read_bytes'])
            self.assertEqual(json.loads((directory / 'read_usage.json').read_text())['total_projected_compressed_bytes'], 17)

    def test_corrupt_totals_fail_before_new_reservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp); state = self.ledger(directory); state['total_projected_compressed_bytes'] = 1
            (directory / 'read_usage.json').write_text(json.dumps(state))
            with self.assertRaisesRegex(DataContractError, 'does not reconcile'):
                m.reserve_reads(directory, 'preflight', 5)

    def test_boolean_projection_is_not_an_integer_budget(self):
        with self.assertRaisesRegex(DataContractError, 'Invalid compressed'):
            m.reserve_reads(Path('/never-read'), 'preflight', True)


class RetConfigurationTests(unittest.TestCase):
    def config(self):
        return json.loads((Path(__file__).resolve().parents[1] / 'configs/ret_backtest_20261020.json').read_text())

    def check(self, cfg):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'cfg.json'; path.write_text(json.dumps(cfg))
            return m.configuration(Path(tmp), path)

    def test_frozen_configuration_and_grid(self):
        cfg = self.config(); self.assertEqual(self.check(cfg), cfg)
        self.assertEqual(len(m.SCENARIOS), 40)
        self.assertEqual(len(set(m.SCENARIOS)), 40)

    def test_changed_score_formula_rejected(self):
        cfg = self.config(); cfg['selection']['ranking'] = 'mode_probability'
        with self.assertRaisesRegex(DataContractError, 'score ranking'):
            self.check(cfg)

    def test_shortened_payoffs_rejected(self):
        cfg = self.config(); cfg['payoffs']['completed_calendar_months'] = 119
        with self.assertRaisesRegex(DataContractError, 'payoff policy'):
            self.check(cfg)

    def test_stop_or_cost_changes_rejected(self):
        for field, value in [('asset_stop_choices', [None, .2]), ('cost_bps', [0, 10]), ('long_weight', 1.)]:
            cfg = self.config(); cfg['execution'][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(DataContractError, 'daily execution'):
                self.check(cfg)

    def test_fit_or_inference_authority_rejected(self):
        for field in ('new_estimator_fits', 'new_preprocessor_fits', 'new_inference_calls'):
            cfg = self.config(); cfg[field] = 1
            with self.subTest(field=field), self.assertRaisesRegex(DataContractError, 'zero fits'):
                self.check(cfg)

    def test_resource_amendment_cannot_be_silent(self):
        cfg = self.config(); cfg['resource_limits']['rss_stop_bytes'] *= 2
        with self.assertRaisesRegex(DataContractError, 'resource'):
            self.check(cfg)


class RetForecastDecodeBoundaryTests(unittest.TestCase):
    def test_future_forecast_fails_before_probability_conversion(self):
        metadata = pd.DataFrame({'signal_date': ['2023-12-29'], 'outcome_month': ['2024-01']})
        calls = []
        class Table:
            def to_pandas(self): return metadata
        class Footer:
            def read(self, *, columns):
                calls.append(columns)
                if columns != ['signal_date', 'outcome_month']:
                    raise AssertionError('protected probability conversion')
                return Table()
        plan = [{'arm': m.ARMS[0], 'files': [{'path': '/synthetic', 'metadata_columns': ['signal_date', 'outcome_month']}]}]
        with patch.object(m.pq, 'ParquetFile', return_value=Footer()):
            with self.assertRaisesRegex(DataContractError, 'financial dates'):
                m.load_forecasts(plan, pd.DataFrame())
        self.assertEqual(len(calls), 1)

    def test_noncanonical_dates_fail_before_probabilities(self):
        metadata = pd.DataFrame({'signal_date': [123], 'outcome_month': ['2020-01']})
        class Table:
            def to_pandas(self): return metadata
        class Footer:
            def read(self, *, columns): return Table()
        plan = [{'arm': m.ARMS[0], 'files': [{'path': '/synthetic', 'metadata_columns': ['signal_date', 'outcome_month']}]}]
        with patch.object(m.pq, 'ParquetFile', return_value=Footer()):
            with self.assertRaisesRegex(DataContractError, 'canonical strings'):
                m.load_forecasts(plan, pd.DataFrame())


if __name__ == '__main__': unittest.main()
