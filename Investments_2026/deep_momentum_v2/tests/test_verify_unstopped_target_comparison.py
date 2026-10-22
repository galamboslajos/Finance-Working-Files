import importlib.util
import math
from pathlib import Path
import tempfile
import unittest

P = Path(__file__).parents[1] / 'scripts/verify_unstopped_target_comparison.py'
spec = importlib.util.spec_from_file_location('independent_unstopped', P)
v = importlib.util.module_from_spec(spec); spec.loader.exec_module(v)


class MaskTests(unittest.TestCase):
    def row(self, sid='a', month='2019-11', completion='2019-12-02'):
        r = {'source_security_id': sid, 'signal_date': v.shift(month, -1) + '-28',
             'outcome_month': month, 'prediction_eligible': True, 'common_fit_mask': True}
        for head in v.HEADS:
            r.update({head + '_target_available': True, head + '_target_class': 1,
                      head + '_label_available_at': completion})
        return r

    def test_completion_boundary_changes_common_mask_without_discarding_original_row(self):
        a, b = self.row(), self.row('b', '2019-12', '2020-01-02')
        b['common_fit_mask'] = False
        result = v.verify_common_mask([a, b], 2020)
        self.assertEqual(result['mask'], [True, False]); self.assertEqual(len(result['keys']), 2)

    def test_early_release_or_changed_mask_rejected(self):
        a = self.row(); a['common_fit_mask'] = False
        with self.assertRaisesRegex(ValueError, 'completion gate'):
            v.verify_common_mask([a], 2020)
        a = self.row(); a['cash_long_label_available_at'] = a['signal_date']
        with self.assertRaisesRegex(ValueError, 'completed before'):
            v.verify_common_mask([a], 2020)

    def test_missing_class_cannot_be_imputed(self):
        a = self.row(); a['cash_short_target_available'] = False
        with self.assertRaisesRegex(ValueError, 'imputed'):
            v.verify_common_mask([a], 2020)

    def test_duplicate_original_key_rejected(self):
        a = self.row()
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            v.verify_common_mask([a, a], 2020)


class PayoffTests(unittest.TestCase):
    def rows(self):
        rows = []
        for m in range(120):
            month = v.shift('2010-01', m)
            for c in range(1, 11):
                rows.append({'source_security_id': str(c), 'signal_date': v.shift(month, -1) + '-28',
                    'outcome_month': month, 'label_available_at': v.shift(month, 1) + '-02',
                    'target_available': True, 'prediction_eligible': True, 'target_class': c,
                    'cash_return': float(c) / 100})
        return rows

    def test_exact120calendar_window_and_shared_lag(self):
        state = v.independent_payoff(self.rows(), '2020-01-31', value_field='cash_return')
        self.assertTrue(state['available']); self.assertEqual(state['first_outcome_month'], '2010-01')
        self.assertEqual(state['last_outcome_month'], '2019-12')
        self.assertAlmostEqual(state['payoffs'][0], .01)

    def test_uncompleted_calendar_cell_never_skipped(self):
        rows = self.rows()[10:]
        state = v.independent_payoff(rows, '2020-01-31', value_field='cash_return')
        self.assertFalse(state['available']); self.assertIsNone(state['payoffs'])
        bad = self.rows(); bad[-1]['label_available_at'] = '2020-02-03'
        with self.assertRaisesRegex(ValueError, 'Uncompleted'):
            v.independent_payoff(bad, '2020-01-31', value_field='cash_return')

    def test_short_endpoint_payoff_is_signed_not_bottom_score_positive(self):
        s = v.independent_payoff(self.rows(), '2020-01-31', value_field='cash_return', short_endpoint=True)
        self.assertAlmostEqual(s['payoffs'][-1], -.1)


class SelectionTests(unittest.TestCase):
    def rows(self, count=6, l=1, s=1):
        rows = []
        for i in range(count):
            r = {'source_security_id': f'{i:02}', 'signal_date': '2020-01-31',
                 'intrinio_security_id': 'native' + str(i), 'prediction_eligible': True}
            r.update({'long_probability_' + str(c): float(c == l) for c in range(1, 11)})
            r.update({'short_probability_' + str(c): float(c == s) for c in range(1, 11)})
            rows.append(r)
        return rows

    def test_mode_overlap_has_balanced_disjoint_stable_id_choice(self):
        r = v.independent_selection(self.rows(), arm='cash', basket='class_1_10')
        self.assertEqual(r['long'], ['00', '01', '02']); self.assertEqual(r['short'], ['03', '04', '05'])
        self.assertEqual(r['n'], 3); self.assertAlmostEqual(r['long_weight'], 1/6)

    def test_short_mode_one_is_correct_for_new_directional_head(self):
        r = v.independent_selection(self.rows(), arm='utility', basket='class_1_10')
        self.assertEqual(r['n'], 3)
        r = v.independent_selection(self.rows(), arm='endpoint', basket='class_1_10')
        self.assertEqual(r['n'], 0)

    def test_score_ranks_each_direction_descending(self):
        rows = self.rows()
        r = v.independent_selection(rows, arm='cash', basket='score', long_payoffs=list(range(10)), short_payoffs=list(range(10)))
        self.assertEqual(r['long'], ['00', '01', '02']); self.assertEqual(r['short'], ['03', '04', '05'])

    def test_missing_payoff_withholds_not_zero(self):
        r = v.independent_selection(self.rows(), arm='cash', basket='score')
        self.assertEqual(r['status'], 'withheld_incomplete_120_month_class_grid')

    def test_restricted_names_removed_before_breadth(self):
        rows = self.rows(); rows[0]['intrinio_security_id'] = 'sec_gjP4xM'
        r = v.independent_selection(rows, arm='cash', basket='class_1_10')
        self.assertNotIn('00', r['long'] + r['short']); self.assertEqual(r['n'], 2)

    def test_probability_is_not_silently_normalized(self):
        rows = self.rows(); rows[0]['long_probability_1'] = .5
        with self.assertRaisesRegex(ValueError, 'simplex'):
            v.independent_selection(rows, arm='cash', basket='class_1_10')

    def test_forecast_ineligible_probabilities_are_not_required(self):
        rows = self.rows(); rows[0] = {'source_security_id': '00', 'signal_date': '2020-01-31',
                                     'prediction_eligible': False, 'intrinio_security_id': None}
        self.assertEqual(v.independent_selection(rows, arm='cash', basket='class_1_10')['n'], 2)



class PayloadAndIndependentMathTests(unittest.TestCase):
    def test_npy_payload_matches_independent_matrix_bytes_and_rejects_tamper(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'matrix.npy'; x = np.array([[1., 2.], [3., 4.]], dtype=np.float32)
            np.save(p, x, allow_pickle=False)
            state = v.array_payload(v.pin(p))
            self.assertEqual(state['shape'], [2, 2]); self.assertEqual(state['array_sha256'], v.array_sha(x))
            with p.open('ab') as f: f.write(b'extra')
            with self.assertRaisesRegex(ValueError, 'payload length'):
                v.array_payload(v.pin(p))

    def test_independent_preprocessing_uses_common_train_only(self):
        import numpy as np
        old = v.load_independent(P.parent / 'corrected_refit_20261021/verify_models.py', 'invented_old_model_math')
        raw = np.array([[1., 2., np.nan, 7.], [3., 4., np.nan, 7.]], float)
        keys = [('a', '2010-01-29'), ('b', '2010-01-29')]
        state = old.derive_preprocessor(raw, ['p', 'q', 'empty', 'constant'], np.array([True, True]), keys,
            ['p', 'q'], '2010-01-01', '2019-12-31', '2019-12-31')
        self.assertEqual(state['manifest']['medians'], {'p': 2., 'q': 3.})
        prediction = np.array([[np.nan, 8., 100., 99.]], float)
        x = old.prepare_matrix(prediction, ['p', 'q', 'empty', 'constant'], state, np.array([True]))
        self.assertEqual(x.tolist(), [[2., 1., 8., 0.]])

    def test_common_preflight_is_explicitly47months(self):
        sessions = [v.shift('2020-02', i) + '-03' for i in range(47)]
        rebalances = [{'execution_date': d} for d in sessions]
        events = [{'effective_date': sessions[1], 'event_id': str(i), 'predecessor_id': 'a' + str(i)} for i in range(18)]
        history = [{'effective_date': '2019-01-03', 'event_id': str(i), 'predecessor_id': 'old' + str(i)} for i in range(11)]
        out = v.independent_common_preflight([], rebalances, [], sessions, events, history)
        self.assertEqual(out['summary']['selected_episode_rows'], 0)
        with self.assertRaisesRegex(ValueError, 'rebalance'):
            v.independent_common_preflight([], rebalances[:-1], [], sessions, events, history)

    def test_payoff_groups_preserve_original_denominator_not_only_observed(self):
        rows = [{'source_security_id': 'a', 'signal_date': '2009-12-31', 'outcome_month': '2010-01',
            'prediction_eligible': True, 'target_available': True, 'target_class': 1, 'payoff_value': -.3},
            {'source_security_id': 'b', 'signal_date': '2009-12-31', 'outcome_month': '2010-01',
            'prediction_eligible': True, 'target_available': False, 'target_class': None, 'payoff_value': None}]
        grouped, counts = v.full_payoff_groups(rows, {'2010-01': {}})
        self.assertEqual(counts['2010-01'], [2, 2]); self.assertEqual(grouped[('2010-01', 1)], [-.3])

    def test_plan_is_metadata_only_and_reserves_projection_bytes(self):
        from unittest.mock import patch
        import json
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); run = root / 'run'; (run / 'folds').mkdir(parents=True)
            (run / 'folds/fabricated.json').write_text('{}')
            target = root / 'target.parquet'; pq.write_table(pa.table({'source_security_id': ['a'], 'target_class': [1]}), target)
            spec = v.footer(v.pin(target))
            model = {'status': 'SEALED_BEFORE_FINANCIAL_READ', 'heads': list(v.HEADS), 'expected_fits': 20,
                'expected_preprocessors': 4, 'financial_2024_plus_allowed': False, 'output_directory': str(run),
                'annuals': {'2010': {'endpoint_metadata': spec, 'endpoint_values': spec, 'path_months': []}}}
            model['plan_sha256'] = v.canonical(model)
            path = root / 'model_plan.json'; path.write_text(json.dumps(model))
            with patch.object(pq.ParquetFile, 'read', side_effect=AssertionError('No financial table reads in planning')):
                result = v.plan_verification(path, None, root / 'verification_plan.json')
            body = v.load_json(result['path'])
            self.assertGreater(body['derived_read_bytes'], 0)
            self.assertEqual(body['derived_read_bytes'], body['derived_read_reservation_bytes'])


class ClassMetricTests(unittest.TestCase):
    def test_confusion_and_separate_class_support_include_missing_predictions(self):
        import numpy as np
        p = np.array([[1.] + [0.] * 9] * 3)
        labels = [1, 2, float('nan')]; months = ['2020-01'] * 3
        confusion, performance = [], []
        for period in ('all', '2020-01'):
            for actual in range(1, 11):
                for predicted in range(1, 11):
                    confusion.append({'period': period, 'actual_class': actual, 'predicted_class': predicted,
                        'count': int(actual in (1, 2) and predicted == 1)})
            for c in range(1, 11):
                performance.append({'period': period, 'class_id': c, 'prediction_rows': 3,
                    'predicted_support': 3 if c == 1 else 0, 'labelled_predicted_support': 2 if c == 1 else 0,
                    'missing_label_predicted_support': 1 if c == 1 else 0, 'actual_support': int(c in (1, 2)),
                    'true_positive_count': int(c == 1), 'precision': .5 if c == 1 else None,
                    'recall': 1. if c == 1 else 0. if c == 2 else None,
                    'status': 'available' if c == 1 else 'missing_predicted_or_actual_support'})
        result = v.verify_class_reports(p, labels, months, confusion, performance)
        self.assertEqual(result['confusion_rows'], 200)
        confusion[0]['count'] = 999
        with self.assertRaisesRegex(ValueError, 'confusion'):
            v.verify_class_reports(p, labels, months, confusion, performance)


if __name__ == '__main__': unittest.main()
