"""Synthetic independent RET verification tests; no local research data."""
from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import types
import unittest

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/verify_ret_backtest.py'
SPEC = importlib.util.spec_from_file_location('independent_ret_audit_under_test', SCRIPT)
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def monthly_cells(signal='2019-12-31'):
    result = {}
    for month in m.trailing_months(signal):
        for cls in range(1, 11):
            result[(month, cls)] = {'outcome_month': month, 'outcome_end': month + '-28',
                'class_id': cls, 'original_member_count': 100, 'prediction_eligible_count': 90,
                'member_count': 9, 'class_mean_return': .011 - .002 * cls, 'status': 'available'}
    return result


def candidate_frame(n=24):
    return pd.DataFrame({'source_security_id': [f'source_{i:03}' for i in range(n)],
        'signal_date': ['2019-12-31'] * n, 'intrinio_security_id': [f'native_{i:03}' for i in range(n)],
        'prediction_eligible': [True] * n, 'target_available': [True] * n})


def target_table():
    window = {'signal_date': '2019-11-29', 'outcome_month': '2019-12',
              'outcome_start': '2019-12-02', 'outcome_end': '2019-12-31'}
    rows = []
    for i, (cls, value) in enumerate(((1, .2), (1, .4), (2, -.1), (2, -.3))):
        rows.append({'source_security_id': f'source{i}', **window,
            'identity_eligible': True, 'requested': True, 'paper_prediction_eligible': True,
            'prediction_eligible': True, 'outcome_requested': True,
            'target_available': True, 'target_class': cls, 'label_return': value})
    rows.append({'source_security_id': 'missing', **window,
        'identity_eligible': True, 'requested': True, 'paper_prediction_eligible': True,
        'prediction_eligible': True, 'outcome_requested': True,
        'target_available': False, 'target_class': None, 'label_return': None})
    return pa.Table.from_pylist(rows), window


class IndependentMeanTests(unittest.TestCase):
    def test_true_classes_missing_rows_and_original_denominators(self):
        table, window = target_table()
        result = m.class_means(table, window)
        self.assertEqual(len(result), 10)
        self.assertAlmostEqual(result[0]['class_mean_return'], .3)
        self.assertAlmostEqual(result[1]['class_mean_return'], -.2)
        self.assertEqual(result[0]['original_member_count'], 5)
        self.assertEqual(result[0]['prediction_eligible_count'], 5)
        self.assertEqual(result[0]['member_count'], 2)
        self.assertEqual(result[2]['status'], 'empty_realized_class')
        self.assertIsNone(result[2]['class_mean_return'])

    def test_unlabelled_is_never_imputed(self):
        table, window = target_table()
        rows = table.to_pylist()
        rows[-1]['label_return'] = 1000.
        rows[-1]['target_class'] = 1
        result = m.class_means(pa.Table.from_pylist(rows), window)
        self.assertAlmostEqual(result[0]['class_mean_return'], .3)

    def test_future_or_misdated_outcome_rejected(self):
        table, window = target_table()
        rows = table.to_pylist(); rows[0]['outcome_end'] = '2020-01-31'
        with self.assertRaisesRegex(m.VerificationError, 'completion window'):
            m.class_means(pa.Table.from_pylist(rows), window)

    def test_prediction_population_cannot_depend_on_target(self):
        table, window = target_table()
        rows = table.to_pylist(); rows[0]['prediction_eligible'] = False
        with self.assertRaises(m.VerificationError):
            m.class_means(pa.Table.from_pylist(rows), window)

    def test_duplicate_source_is_rejected(self):
        table, window = target_table()
        with self.assertRaisesRegex(m.VerificationError, 'Duplicate'):
            m.class_means(pa.concat_tables([table, table.slice(0, 1)]), window)

    def test_invalid_return_rejected(self):
        table, window = target_table()
        for bad in (-1.01, float('nan'), float('inf')):
            rows = table.to_pylist(); rows[0]['label_return'] = bad
            with self.subTest(bad=bad), self.assertRaises(m.VerificationError):
                m.class_means(pa.Table.from_pylist(rows), window)

    def test_fingerprint_independent_of_arrow_offsets_and_chunks(self):
        table, _ = target_table()
        original = m.source_month_fingerprint(table)
        other = pa.concat_tables([table.slice(0, 2), table.slice(2)])
        self.assertEqual(original, m.source_month_fingerprint(other))
        changed = table.to_pylist(); changed[0]['label_return'] += .01
        self.assertNotEqual(original, m.source_month_fingerprint(pa.Table.from_pylist(changed, schema=table.schema)))


class IndependentPayoffTests(unittest.TestCase):
    def test_first_signal_uses_december2019_not_january2020(self):
        months = m.trailing_months('2019-12-31')
        self.assertEqual(months[0], '2010-01')
        self.assertEqual(months[-1], '2019-12')
        self.assertEqual(len(months), 120)
        self.assertNotIn('2020-01', months)

    def test_last_signal_excludes_december2023(self):
        months = m.trailing_months('2023-11-30')
        self.assertEqual(months[0], '2013-12')
        self.assertEqual(months[-1], '2023-11')
        with self.assertRaises(m.VerificationError):
            m.trailing_months('2023-12-29')

    def test_exact_equal_calendar_month_weights_not_pooled_stock_rows(self):
        cells = monthly_cells()
        for month in m.trailing_months('2019-12-31'):
            cells[(month, 1)]['class_mean_return'] = 0.
            cells[(month, 1)]['member_count'] = 1
        cells[('2010-01', 1)]['class_mean_return'] = 1.2
        cells[('2010-01', 1)]['member_count'] = 1000
        result, grid = m.independent_payoff(cells, '2019-12-31')
        self.assertAlmostEqual(result[0], .01)
        self.assertEqual(len(grid), 1200)

    def test_missing_calendar_month_is_not_compressed_or_shifted(self):
        cells = monthly_cells(); del cells[('2011-03', 1)]
        with self.assertRaisesRegex(m.VerificationError, 'Missing'):
            m.independent_payoff(cells, '2019-12-31')

    def test_empty_class_withholds_whole_vector(self):
        cells = monthly_cells(); cells[('2010-01', 10)]['status'] = 'empty_realized_class'
        with self.assertRaisesRegex(m.VerificationError, 'Incomplete'):
            m.independent_payoff(cells, '2019-12-31')

    def test_uncompleted_last_month_cannot_enter(self):
        cells = monthly_cells(); cells[('2019-12', 1)]['outcome_end'] = '2019-12-31'
        with self.assertRaises(m.VerificationError):
            m.independent_payoff(cells, '2019-12-31', cutoff='2019-12-30')

    def test_future_cutoff_cannot_enter(self):
        with self.assertRaises(m.VerificationError):
            m.independent_payoff(monthly_cells(), '2019-12-31', cutoff='2020-01-01')

    def test_out_of_window_outlier_cannot_change_payoff(self):
        cells = monthly_cells()
        expected, _ = m.independent_payoff(cells, '2019-12-31')
        cells[('2020-01', 1)] = {'class_mean_return': 1e200}
        observed, _ = m.independent_payoff(cells, '2019-12-31')
        np.testing.assert_array_equal(observed, expected)


class IndependentScoreRankTests(unittest.TestCase):
    def test_scores_use_all_ten_probabilities(self):
        p = np.zeros((2, 10)); p[0, 0] = .51; p[0, -1] = .49; p[1, 0] = .5; p[1, 1] = .5
        mu = np.array([.2, .1, 0, 0, 0, 0, 0, 0, 0, -.2])
        result = m.independent_scores(p, mu)
        self.assertAlmostEqual(result[0], .004)
        self.assertAlmostEqual(result[1], .15)
        self.assertGreater(result[1], result[0])

    def test_invalid_probability_vectors_are_not_renormalized(self):
        for bad in (np.full((1, 10), .2), np.full((1, 10), -.1), np.full((1, 10), np.nan)):
            with self.subTest(bad=bad), self.assertRaises(m.VerificationError):
                m.independent_scores(bad, np.arange(10.))

    def test_missing_class_rejected(self):
        with self.assertRaises(m.VerificationError):
            m.independent_scores(np.full((1, 9), 1/9), np.arange(10.))

    def test_flat_scores_disjoint_stable_source_ties(self):
        group = candidate_frame()
        longs, shorts, n = m.independent_rank(group, np.zeros(len(group)))
        self.assertEqual(n, 10)
        self.assertEqual(longs.source_security_id.tolist(), [f'source_{i:03}' for i in range(10)])
        self.assertEqual(shorts.source_security_id.tolist(), [f'source_{i:03}' for i in range(10, 20)])

    def test_positive_only_scores_still_form_short_side(self):
        group = candidate_frame()
        longs, shorts, n = m.independent_rank(group, np.arange(1., 25.))
        self.assertEqual(n, 10)
        self.assertEqual(shorts.source_security_id.iloc[0], 'source_000')
        self.assertEqual(longs.source_security_id.iloc[0], 'source_023')

    def test_no_class_mode_gate(self):
        group = candidate_frame()
        group['predicted_class'] = 5
        longs, shorts, n = m.independent_rank(group, np.arange(24.))
        self.assertEqual((len(longs), len(shorts), n), (10, 10, 10))

    def test_missing_future_labels_do_not_change_selection(self):
        group = candidate_frame(); group.target_available = False
        longs, shorts, n = m.independent_rank(group, np.arange(24.))
        self.assertEqual((len(longs), len(shorts), n), (10, 10, 10))

    def test_fixed_exclusions_only_and_balanced_shortage(self):
        group = candidate_frame(19); group.loc[18, 'intrinio_security_id'] = sorted(m.EXCLUDED)[0]
        longs, shorts, n = m.independent_rank(group, np.arange(19.))
        self.assertEqual(n, 9)
        self.assertNotIn('source_018', set(longs.source_security_id) | set(shorts.source_security_id))
        self.assertEqual(set(longs.source_security_id) | set(shorts.source_security_id), set(group.source_security_id[:18]))

    def test_insufficient_two_sides_is_zero_not_unbalanced(self):
        longs, shorts, n = m.independent_rank(candidate_frame(1), [1.])
        self.assertEqual((len(longs), len(shorts), n), (0, 0, 0))


class IndependentPersistenceTests(unittest.TestCase):
    def test_read_budget_charges_each_projection(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'synthetic.parquet'; pq.write_table(pa.table({'a': [1, 2]}), path)
            size = path.stat().st_size
            budget = m.ProjectionBudget(prior=3, reserved=2*size, limit=3+2*size)
            budget.read(path, ['a']); budget.read(path, ['a'])
            self.assertEqual(budget.used, 2*size)
            with self.assertRaises(m.VerificationError):
                budget.read(path, ['a'])

    def test_future_key_rejected_before_numeric_conversion(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'synthetic.parquet'
            pq.write_table(pa.table({'date': ['2024-01-02'], 'nav': [10000.]}), path)
            budget = m.ProjectionBudget(reserved=2*path.stat().st_size)
            v = types.SimpleNamespace(iso=lambda value: pd.Timestamp(value).strftime('%Y-%m-%d'))
            with self.assertRaisesRegex(m.VerificationError, 'before numeric'):
                m.guarded_table(v, budget, path)
            self.assertEqual(budget.used, path.stat().st_size)

    def test_read_reservation_is_cumulative_and_cannot_repeat(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder); path = run / 'read_usage.json'
            path.write_text(json.dumps({'limit_bytes': m.READ_LIMIT, 'total_projected_compressed_bytes': 5,
                'reservations': [{'phase': 'preflight', 'projected_compressed_bytes': 5}]}))
            self.assertEqual(m.reserve_verification(run, 7), 5)
            self.assertEqual(json.loads(path.read_text())['total_projected_compressed_bytes'], 12)
            with self.assertRaisesRegex(m.VerificationError, 'previous verification'):
                m.reserve_verification(run, 7)

    def test_read_reservation_rejects_ledger_tampering(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            (run/'read_usage.json').write_text(json.dumps({'limit_bytes': m.READ_LIMIT,
                'total_projected_compressed_bytes': 0,
                'reservations': [{'phase': 'preflight', 'projected_compressed_bytes': 5}]}))
            with self.assertRaises(m.VerificationError):
                m.reserve_verification(run, 7)

    def test_pin_must_remain_inside_root_and_exact(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); path = root / 'sample.json'; path.write_text('{}')
            pin = {'path': 'sample.json', 'size_bytes': path.stat().st_size, 'sha256': m.digest(path)}
            self.assertEqual(m.pinned(root, pin), path.resolve())
            path.write_text('{"changed":true}')
            with self.assertRaises(m.VerificationError):
                m.pinned(root, pin)
            with self.assertRaises(m.VerificationError):
                m.local(root, '../escape')


if __name__ == '__main__':
    unittest.main()
