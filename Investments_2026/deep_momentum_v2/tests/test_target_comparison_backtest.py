"""Structural fixtures only: no project datasets, models or financial outputs."""
import copy
from functools import lru_cache
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from dmv2 import target_comparison_backtest as b
from dmv2.walkforward_clock import walkforward_clock


@lru_cache(None)
def fixture():
    sessions, windows, clock = walkforward_clock()
    windows = windows.to_dict('records')
    history_windows, endpoint, paths = [], [], []
    for n in range(168):
        month = b.shift('2010-01', n)
        available = b.shift(month, 1) + '-02'
        history_windows.append({'outcome_month': month, 'label_available_at': available})
        for c in range(1, 11):
            row = {'source_security_id': 'historical_' + str(c),
                'signal_date': b.shift(month, -1) + '-28', 'outcome_month': month,
                'prediction_eligible': True, 'target_available': True,
                'target_class': c, 'label_return': (11 - c) / 1000}
            endpoint.append(row)
            for side in ('long', 'short'):
                for arm, family in (('cash', 'executable_return'), ('utility', 'return_minus_drawdown')):
                    paths.append({**row, 'side': side, 'path_variant': 'unstopped',
                        'target_family': family, 'label_available_at': available,
                        'cash_return': (11 - c) / 1000, 'utility': (11 - c) / 1000 - .02})
    signals = [w['signal_date'] for w in windows]
    payoffs = {(arm, side): b.build_payoffs(endpoint if arm == 'endpoint_matched_clock' else paths,
        signals, history_windows, arm=arm, side=side)
        for arm in b.ARMS for side in ('long', 'short')}
    forecasts = {}
    for head in b.HEADS:
        rows = []
        for w in windows:
            for k in range(26):
                eligible = k != 25
                mode = (1 if k < 12 else 10) if head == 'endpoint' else 1
                probabilities = [0.] * 10
                probabilities[mode - 1] = 1.
                rows.append({'source_security_id': f'source_{k:02d}', 'signal_date': w['signal_date'],
                    'intrinio_security_id': (b.EXCLUDED[0] if k == 24 else f'native_{k:02d}') if eligible else None,
                    'outcome_month': w['outcome_month'], 'outcome_start': w['outcome_start'], 'outcome_end': w['outcome_end'],
                    'prediction_eligible': eligible, 'paper_prediction_eligible': eligible,
                    'identity_eligible': eligible, 'prediction_status': 'predicted' if eligible else 'withheld_signal_time_baseline',
                    **{f'probability_{c}': probabilities[c - 1] if eligible else np.nan for c in range(1, 11)}})
        forecasts[head] = pd.DataFrame(rows)
    return sessions, windows, clock, endpoint, paths, history_windows, payoffs, forecasts


class PayoffTests(unittest.TestCase):
    def test_equal_month_causal_window_and_january_warmup(self):
        *_, payoffs, forecasts = fixture()
        for arm in b.ARMS:
            for side in ('long', 'short'):
                state = payoffs[(arm, side)]['2020-01-31']
                self.assertEqual(state['provenance']['first_outcome_month'], '2010-01')
                self.assertEqual(state['provenance']['last_outcome_month'], '2019-12')
                self.assertTrue(state['provenance']['vector_available'])
                self.assertEqual(len(state['monthly']), 1200)
                self.assertFalse(payoffs[(arm, side)]['2019-12-31']['provenance']['vector_available'])

    def test_own_utility_and_endpoint_short_sign(self):
        payoffs = fixture()[-2]
        signal = '2020-01-31'; p = [1.] + [0.] * 9
        left = b.score(p, payoffs[('endpoint_matched_clock', 'long')][signal], signal_date=signal,
            arm='endpoint_matched_clock', side='long')
        right = b.score(p, payoffs[('endpoint_matched_clock', 'short')][signal], signal_date=signal,
            arm='endpoint_matched_clock', side='short')
        self.assertEqual(left, -right)
        cash = b.score(p, payoffs[('cash', 'short')][signal], signal_date=signal, arm='cash', side='short')
        utility = b.score(p, payoffs[('utility', 'short')][signal], signal_date=signal, arm='utility', side='short')
        self.assertAlmostEqual(cash - utility, .02)
        self.assertLess(utility, 0)  # No positive-score eligibility filter.

    def test_empty_class_withholds_whole_vector(self):
        _, _, _, endpoint, _, windows, _, _ = fixture()
        rows = [r for r in endpoint if not (r['outcome_month'] == '2010-01' and r['target_class'] == 10)]
        state = b.build_payoffs(rows, ['2020-01-31'], windows, arm='endpoint_matched_clock', side='long')['2020-01-31']
        self.assertFalse(state['provenance']['vector_available'])
        self.assertTrue(all(r['payoff'] is None for r in state['table']))
        self.assertIsNone(b.score([.1] * 10, state, signal_date='2020-01-31', arm='endpoint_matched_clock', side='long'))

    def test_uncompleted_cell_and_duplicate_original_fail(self):
        _, _, _, endpoint, _, windows, _, _ = fixture()
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            b.build_payoffs(endpoint + endpoint[:1], ['2020-01-31'], windows, arm='endpoint_matched_clock', side='long')
        future = copy.deepcopy(windows)
        next(w for w in future if w['outcome_month'] == '2019-12')['label_available_at'] = '2020-02-03'
        state = b.build_payoffs(endpoint, ['2020-01-31'], future, arm='endpoint_matched_clock', side='long')['2020-01-31']
        self.assertFalse(state['provenance']['vector_available'])

    def test_actual_full_audit_exit_date_window_schema(self):
        _, _, _, endpoint, _, windows, _, _ = fixture()
        actual_schema = [{'outcome_month': w['outcome_month'], 'exit_date': w['label_available_at']} for w in windows]
        state = b.build_payoffs(endpoint, ['2020-01-31'], actual_schema, arm='endpoint_matched_clock', side='long')['2020-01-31']
        self.assertTrue(state['provenance']['vector_available'])
        self.assertEqual(state['monthly'][-1]['label_available_at'], '2020-01-02')

    def test_rehashed_wrong_grid_or_payoff_is_rejected(self):
        state = copy.deepcopy(fixture()[-2][('cash', 'long')]['2020-01-31'])
        state['table'][0]['payoff'] += .01
        state['fingerprint'] = b.canonical({k: state[k] for k in ('table', 'monthly', 'provenance')})
        with self.assertRaisesRegex(ValueError, 'formula'):
            b.score([.1] * 10, state, signal_date='2020-01-31', arm='cash', side='long')


class SelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, cls.windows, cls.clock, _, _, _, cls.payoffs, cls.forecasts = fixture()
        cls.selection = b.select_baskets(cls.forecasts, cls.payoffs, cls.windows, cls.clock['selection_sessions'])

    def test_all_arms_baskets_originals_and_48_states_retained(self):
        self.assertEqual(set(self.selection), set(__import__('itertools').product(b.ARMS, b.BASKETS)))
        for key, item in self.selection.items():
            self.assertEqual(len(item.ledger), 26 * 48)
            self.assertEqual(len(item.monthly), 48)
            first = item.monthly.iloc[0]
            self.assertEqual(first.status, 'withheld_incomplete_120_month_class_grid'
                if key[1] == 'score' else 'outside_common_comparison_period')
            self.assertFalse(item.selections.outcome_month.eq('2020-01').any())

    def test_score_ties_long_first_disjoint_short_and_weights(self):
        item = self.selection[('cash', 'score')]
        group = item.selections.loc[item.selections.outcome_month.eq('2020-02')]
        longs = group.loc[group.side.eq('long'), 'source_security_id'].tolist()
        shorts = group.loc[group.side.eq('short'), 'source_security_id'].tolist()
        self.assertEqual(longs, [f'source_{i:02d}' for i in range(10)])
        self.assertEqual(shorts, [f'source_{i:02d}' for i in range(10, 20)])
        self.assertAlmostEqual(group.loc[group.side.eq('long'), 'intended_weight'].sum(), .5)
        self.assertAlmostEqual(group.loc[group.side.eq('short'), 'intended_weight'].sum(), -.5)
        self.assertFalse(group.source_security_id.duplicated().any())
        self.assertFalse(group.intrinio_security_id.isin(b.EXCLUDED).any())

    def test_directional_best_is_one_one_endpoint_one_ten(self):
        new = self.selection[('cash', 'best_mode')].selections
        self.assertTrue(new.predicted_class_short.eq(1).all())
        old = self.selection[('endpoint_matched_clock', 'best_mode')].selections
        self.assertTrue(old.loc[old.side.eq('short'), 'predicted_class_short'].eq(10).all())

    def test_overlap_shortage_reduces_both_sides(self):
        forecasts = {head: frame.loc[frame.source_security_id.isin([f'source_{i:02d}' for i in range(9)])].copy()
            for head, frame in self.forecasts.items()}
        result = b.select_baskets(forecasts, self.payoffs, self.windows, self.clock['selection_sessions'])
        monthly = result[('cash', 'best_mode')].monthly.iloc[1:]
        self.assertTrue(monthly.selected_names_per_side.eq(4).all())

    def test_future_targets_ignored_and_ineligible_rows_not_imputed(self):
        changed = {head: frame.assign(target_available=False, label_return=999., target_class=10)
            for head, frame in self.forecasts.items()}
        actual = b.select_baskets(changed, self.payoffs, self.windows, self.clock['selection_sessions'])
        for key in actual:
            pd.testing.assert_frame_equal(actual[key].selections, self.selection[key].selections, check_exact=True)

    def test_missing_short_payoff_withholds_score_not_mode(self):
        payoffs = dict(self.payoffs); payoffs[('cash', 'short')] = dict(payoffs[('cash', 'short')])
        del payoffs[('cash', 'short')]['2020-01-31']
        result = b.select_baskets(self.forecasts, payoffs, self.windows, self.clock['selection_sessions'])
        self.assertEqual(result[('cash', 'score')].monthly.iloc[1].status, 'withheld_incomplete_120_month_class_grid')
        self.assertEqual(result[('cash', 'best_mode')].monthly.iloc[1].status, 'available')

    def test_bad_probability_or_changed_cohort_fails(self):
        values = {k: v.copy() for k, v in self.forecasts.items()}
        values['utility_short'].loc[0, 'probability_1'] = .5
        with self.assertRaisesRegex(ValueError, 'simplex'):
            b.select_baskets(values, self.payoffs, self.windows, self.clock['selection_sessions'])
        values = dict(self.forecasts); values['utility_short'] = values['utility_short'].iloc[1:].copy()
        with self.assertRaises(AssertionError):
            b.select_baskets(values, self.payoffs, self.windows, self.clock['selection_sessions'])


class AccountingIntegrationTests(unittest.TestCase):
    def test_price_adapter_masks_adjacent_annual_anchor_before_financial_python(self):
        import json
        import pyarrow as pa
        def bar(day):
            return {'trading_date': day, 'intrinio_security_id': 'synthetic_native',
                'adj_open': 100., 'adj_high': 101., 'adj_low': 99., 'adj_close': 100.,
                'open': 100., 'high': 101., 'low': 99., 'close': 100., 'factor': 1.,
                'split_ratio': 1., 'dividend': 0., 'frequency': 'daily', 'intraperiod': False,
                'price_quality_status': 'accepted', 'raw_ohlc_bounds_valid': True, 'adjusted_ohlc_bounds_valid': True}
        tables = {}
        daily = []
        for year, days in ((2020, ['2020-02-03', '2020-12-31']), (2021, ['2020-12-31', '2021-01-04'])):
            values = pa.Table.from_pylist([bar(day) for day in days])
            tables[str(year) + '_metadata'] = values.select(b.QUOTE_META)
            tables[str(year) + '_values'] = values
            daily.append({'year': year, 'metadata': str(year) + '_metadata', 'values': str(year) + '_values'})
        empty = pa.Table.from_pylist([bar('2020-02-03')]).slice(0, 0)
        tables['repair_meta'] = empty.select(b.QUOTE_META); tables['repair_values'] = empty
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); phase = root / 'phase.json'; manifest = root / 'manifest.json'
            phase.write_text('{}'); manifest.write_text(json.dumps({'quote_amendments': []}))
            plan = {'sessions': ['2020-02-03', '2020-12-31', '2021-01-04'], 'daily': daily,
                'repairs': {'metadata': 'repair_meta', 'values': 'repair_values'},
                'execution_pins': {'execution_phase_pin': b.pin(phase), 'execution_manifest_pin': b.pin(manifest)}}
            with patch.object(b, '_read_projection', side_effect=lambda spec: tables[spec]):
                prices, receipt, _ = b._load_prices(plan, lambda: None)
            self.assertEqual(len(prices), 3)
            self.assertFalse(prices.duplicated(['trading_date', 'intrinio_security_id']).any())
            self.assertEqual(receipt['excluded_source_rows_metadata_only']['2021'], 1)

    def test_all72_paths_use_unchanged_cash_stop_engine_on_structural_clock(self):
        """Two independently known sessions/month keep this fixture small."""
        sessions, windows, clock, endpoint, paths, history, _, forecasts = fixture()
        selected_sessions = sorted({w[k] for w in windows[1:] for k in ('outcome_start', 'outcome_end')})
        forecasts = {head: frame.loc[frame.source_security_id.isin(['source_00', 'source_01', 'source_12', 'source_13'])].copy()
            for head, frame in forecasts.items()}
        price_rows = [{'trading_date': day, 'intrinio_security_id': f'native_{k:02d}',
            'adj_open': 100., 'adj_high': 101., 'adj_low': 99., 'adj_close': 100.,
            'open': 100., 'high': 101., 'low': 99., 'close': 100.,
            'frequency': 'daily', 'intraperiod': False, 'price_quality_status': 'accepted',
            'raw_ohlc_bounds_valid': True, 'adjusted_ohlc_bounds_valid': True}
            for day in selected_sessions for k in (0, 1, 12, 13)]
        phase = {'event_registry': {'events': []}, 'historical_retirements': [],
            'identity_handover_registry': {'handovers': []}}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); source = root / 'target_plan.json'
            source.write_text(__import__('json').dumps({'annuals': [{'windows': history}]}))
            model_plan = root / 'read_plan.json'; model_plan.write_text('{}')
            (root / 'verification').mkdir()
            (root / 'verification/models.json').write_text(__import__('json').dumps({
                'status': 'PASS', 'phase': 'models', 'verified_estimator_fits': 20,
                'verified_shared_preprocessors': 4, 'new_estimator_fits': 0,
                'financial_2024_plus_values_read': False, 'model_plan_pin': b.pin(model_plan)}))
            plan = {'output_directory': str(root / 'backtest'), 'plan_sha256': 'synthetic_only',
                'model_output_directory': str(root), 'model_plan_pin': b.pin(model_plan),
                'model_source_pins': {'target_plan_pin': b.pin(source)},
                'windows': windows, 'selection_sessions': clock['selection_sessions'], 'sessions': selected_sessions,
                'expected_sessions': len(selected_sessions), 'projected_read_bytes': 0,
                'limits': {'rss_stop_bytes': 8 * 1024 ** 3, 'active_seconds': 120}}
            def target_rows(plan, *, arm, side, resource_check):
                return endpoint if arm == 'endpoint_matched_clock' else paths
            with patch.object(b, 'verify_plan', side_effect=lambda p: p), \
                    patch.object(b, '_forecast_plan', return_value=[]), \
                    patch.object(b, '_target_rows', side_effect=target_rows), \
                    patch.object(b, '_load_forecasts', return_value=forecasts), \
                    patch.object(b, '_load_prices', return_value=(pd.DataFrame(price_rows), {'synthetic': True}, phase)):
                result = b.run_backtest(plan)
            self.assertEqual(result['scenarios'], 72)
            self.assertEqual(result['complete_proxy_paths'], 48)
            self.assertEqual(result['withheld_paths'], 24)  # Empty second-mode cohort is retained.
            self.assertEqual(result['fits'], 0)
            self.assertEqual(len(list((root / 'backtest/checkpoints').glob('*/*/*/*.json'))), 72)
            comparison = pd.read_parquet(root / 'backtest/comparison.parquet')
            self.assertEqual(len(comparison), 72)
            missing = comparison.loc[~comparison.complete_proxy_path]
            self.assertTrue(missing.total_return.isna().all())
            gross = comparison.loc[comparison.complete_proxy_path & comparison.cost_bps.eq(0)]
            self.assertTrue(gross.total_return.eq(0).all())
            net = comparison.loc[comparison.complete_proxy_path & comparison.cost_bps.eq(10)]
            self.assertTrue(net.total_return.lt(0).all())


if __name__ == '__main__':
    unittest.main()
