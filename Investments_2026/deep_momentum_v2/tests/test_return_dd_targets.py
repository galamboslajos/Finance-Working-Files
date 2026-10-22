"""Invented records only. Never load research panels, prices or estimators."""
import copy
import importlib.util
import math
from pathlib import Path
import sys
import unittest

import pyarrow as pa

spec = importlib.util.spec_from_file_location('ratio_worker_under_test',
    Path(__file__).parents[1] / 'src/dmv2/return_dd_targets.py')
p = importlib.util.module_from_spec(spec); sys.modules[spec.name] = p; spec.loader.exec_module(p)


def fixture(n=10, month='2020-01', sides=p.SIDES):
    rows = []
    for side in sides:
        for i in range(n):
            ret = (i + 1) / 100 if side == 'long' else (n - i) / 100
            rows.append({'source_security_id': f'S{i:03}',
                'signal_date': p._shift(month, -1) + '-28', 'side': side,
                'path_variant': 'unstopped', 'target_family': 'executable_return',
                'outcome_month': month, 'entry_date': month + '-02',
                'exit_date': p._shift(month, 1) + '-03', 'exit_mark': 'open',
                'label_available_at': p._shift(month, 1) + '-03',
                'prediction_eligible': True, 'path_requested': True, 'path_available': True,
                'target_available': True, 'target_status': 'available',
                'target_value': ret, 'target_class': min(10,1+10*(2*(n-i-1 if side=='long' else i)+1)//(2*n)),
                'cash_return': ret, 'maximum_drawdown': .02,
                'intrinio_security_id': 'N'+str(i), 'identity_review_required': i == 3})
    return rows


def build(rows=None, **kwargs):
    return p.build_return_dd_targets(rows or fixture(),
        observed_through=kwargs.pop('observed_through', '2023-12-31'), **kwargs)


def one(panel, source='S009', side='long'):
    return next(row for row in panel.table.to_pylist()
                if row['source_security_id'] == source and row['side'] == side)


class Ratios(unittest.TestCase):
    def test_frozen_literal_signed_formula_floor_and_zero(self):
        self.assertEqual(p.signed_ratio(.08, .04), (2., 'available', False, False))
        self.assertEqual(p.signed_ratio(.08, 0), (8., 'available', True, False))
        self.assertEqual(p.signed_ratio(-.05, .05)[0], -1.)
        self.assertEqual(p.signed_ratio(-.05, .20)[0], -.25)
        self.assertGreater(p.signed_ratio(-.05, .20)[0], p.signed_ratio(-.05, .05)[0])
        self.assertEqual(p.signed_ratio(-1.5, 2.)[0], -.75)
        self.assertEqual(p.signed_ratio(0, 0)[0], 0.)
        self.assertFalse(p.signed_ratio(.1, .01)[2])

    def test_null_invalid_and_nonfinite_never_coerced(self):
        for ret, dd in [(None,.2),(.2,None)]:
            self.assertEqual(p.signed_ratio(ret,dd)[1], 'missing_return_or_drawdown')
        for ret, dd in [(float('nan'),.2),(.2,float('inf')),(-float('inf'),.2)]:
            self.assertEqual(p.signed_ratio(ret,dd)[1], 'invalid_nonfinite_input')
        for ret, dd in [(True,.2),(.2,False),('0.2',.2)]:
            self.assertEqual(p.signed_ratio(ret,dd)[1], 'invalid_non_numeric_input')
        self.assertEqual(p.signed_ratio(.2,-.01)[1], 'invalid_negative_drawdown')

    def test_original_keys_admission_metadata_and_clocks_unchanged(self):
        rows = fixture(); rows[0].update(prediction_eligible=False, intrinio_security_id=None)
        panel = build(rows)
        self.assertEqual(panel.table.num_rows, len(rows))
        lookup = {(r['source_security_id'],r['side']): r for r in rows}
        for row in panel.table.to_pylist():
            old = lookup[(row['source_security_id'],row['side'])]
            for name in (*p.KEYS, 'outcome_month', 'entry_date', 'exit_date', 'exit_mark',
                         'label_available_at', 'prediction_eligible', 'path_available',
                         'path_requested', 'intrinio_security_id', 'identity_review_required'):
                self.assertEqual(row[name], old[name])
            self.assertEqual(row['source_target_family'], 'executable_return')
            self.assertEqual(row['source_target_available'], old['target_available'])
            self.assertEqual(row['source_target_status'], old['target_status'])
        self.assertEqual(one(panel,'S000')['target_status'], 'prediction_ineligible')
        self.assertIsNone(one(panel,'S000')['target_value'])

    def test_distinct_direction_classes_and_class_one_highest(self):
        panel = build()
        self.assertEqual(one(panel)['target_value'],5.)
        self.assertEqual(one(panel)['target_class'],1)
        self.assertEqual(one(panel,side='short')['target_class'],10)
        self.assertEqual(one(panel)['source_cash_target_class'],1)
        self.assertEqual(one(panel)['source_cash_target_value'],.1)
        self.assertEqual(set(panel.table['target_family'].to_pylist()), {p.FAMILY})

    def test_exact_ties_midrank_no_jitter_and_permutation_invariant(self):
        rows = fixture(sides=('long',))
        for row in rows:
            row['cash_return'] = .1
        panel = build(rows)
        self.assertEqual(set(panel.table['target_class'].to_pylist()), {6})
        reversed_panel = build(list(reversed(rows)))
        self.assertEqual({r['source_security_id']: r['target_class'] for r in panel.table.to_pylist()},
                         {r['source_security_id']: r['target_class'] for r in reversed_panel.table.to_pylist()})
        self.assertEqual(panel.diagnostics.to_pylist()[0]['empty_classes'], [1,2,3,4,5,7,8,9,10])

    def test_different_losses_at_initial_peak_final_trough_remain_minus_one_ties(self):
        rows=fixture(sides=('long',))
        for i,row in enumerate(rows):
            row['cash_return']=-(i+1)/100
            row['maximum_drawdown']=-row['cash_return']
            row['target_value']=row['cash_return']
        panel=build(rows)
        self.assertEqual(set(panel.table['target_value'].to_pylist()),{-1.})
        self.assertEqual(set(panel.table['target_class'].to_pylist()),{6})
        self.assertEqual(len(set(panel.table['cash_return'].to_pylist())),10)
        diag=panel.diagnostics.to_pylist()[0]
        self.assertEqual(diag['largest_exact_tie_count'],10)
        self.assertEqual(diag['minus_one_ratio_count'],10)
        self.assertEqual(diag['empty_classes'],[1,2,3,4,5,7,8,9,10])

    def test_less_than_ten_and_missing_paths_preserve_population(self):
        rows = fixture(); rows[0].update(path_available=False, cash_return=None)
        panel = build(rows)
        self.assertEqual(panel.table.num_rows,20)
        self.assertEqual(one(panel,'S000')['path_available'],False)
        self.assertFalse(any(r['target_available'] for r in panel.table.to_pylist() if r['side']=='long'))
        self.assertTrue(all(r['target_available'] for r in panel.table.to_pylist() if r['side']=='short'))

    def test_source_class_withheld_does_not_filter_new_ranking(self):
        rows = fixture()
        for row in rows:
            row.update(target_available=False,target_class=None,target_status='source_ranking_withheld')
        panel = build(rows)
        self.assertTrue(all(panel.table['target_available'].to_pylist()))
        self.assertTrue(all(not x for x in panel.table['source_target_available'].to_pylist()))

    def test_non_exhaustive_request_withholds_whole_month_side(self):
        rows = fixture(11); rows[0]['path_requested'] = False
        panel = build(rows)
        self.assertFalse(any(r['target_available'] for r in panel.table.to_pylist() if r['side']=='long'))
        self.assertTrue(all(r['target_available'] for r in panel.table.to_pylist() if r['side']=='short'))

    def test_metadata_first_does_not_touch_uncompleted_numeric_payload(self):
        class Explosive:
            def __float__(self):
                raise AssertionError('Uncompleted numbers were inspected')
        rows = fixture(month='2023-12')
        for row in rows:
            row['cash_return'] = Explosive(); row['maximum_drawdown'] = Explosive()
        panel = build(rows)
        self.assertTrue(all(x is None for x in panel.table['target_value'].to_pylist()))
        self.assertTrue(all(not x for x in panel.table['target_available'].to_pylist()))
        self.assertTrue(all(panel.table['prediction_eligible'].to_pylist()))

    def test_invalid_numerics_retained_as_missing_not_dropped(self):
        rows = fixture(12)
        rows[0]['maximum_drawdown'] = -.1; rows[1]['cash_return'] = float('nan')
        panel = build(rows)
        self.assertEqual(panel.table.num_rows,24)
        self.assertEqual(one(panel,'S000')['target_status'],'invalid_negative_drawdown')
        self.assertEqual(one(panel,'S001')['target_status'],'invalid_nonfinite_input')
        self.assertTrue(all(r['target_available'] for r in panel.table.to_pylist()
                            if r['side']=='long' and r['source_security_id'] not in ('S000','S001')))

    def test_floor_negative_zero_and_dd_above_one_audit_only(self):
        rows = fixture()
        rows[0].update(cash_return=-.03,maximum_drawdown=0)
        rows[1].update(cash_return=-1.5,maximum_drawdown=2.)
        panel = build(rows)
        diag = next(r for r in panel.diagnostics.to_pylist() if r['side']=='long')
        self.assertEqual((diag['floor_hit_count'],diag['zero_drawdown_count'],diag['negative_return_count'],
                          diag['drawdown_above_one_count']), (1,1,2,1))
        self.assertFalse(panel.provenance['diagnostics_are_features'])
        self.assertEqual(panel.provenance['estimator_fits'],0)

    def test_arrow_equivalence_and_source_unchanged(self):
        rows = fixture(); original = copy.deepcopy(rows)
        a, b = build(rows), build(pa.Table.from_pylist(rows))
        self.assertEqual(a.table.to_pylist(), b.table.to_pylist())
        self.assertEqual(rows,original)

    def test_invalid_scope_keys_flags_or_clock_rejected(self):
        for field,value in [('target_family','return_minus_drawdown'),('path_variant','asset_stop_10pct'),
                            ('prediction_eligible',1),('exit_mark','close'),
                            ('outcome_month','2020-02'),('label_available_at','2020-01-03')]:
            rows = fixture(); rows[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                build(rows)
        rows=fixture();rows.append(dict(rows[0]))
        with self.assertRaises(ValueError):build(rows)

    def test_training_slice_exact_completion_cutoff_no_admission_change(self):
        panel=build()
        before=p.completed_training_slice(panel,train_start_month='2020-01',train_end_month='2020-01',
            as_of='2020-02-02',side='long')
        after=p.completed_training_slice(panel,train_start_month='2020-01',train_end_month='2020-01',
            as_of='2020-02-03',side='long')
        self.assertEqual(before.ledger.num_rows,10);self.assertEqual(before.fit_table.num_rows,0)
        self.assertEqual(after.fit_table.num_rows,10)
        self.assertTrue(all(before.ledger['prediction_eligible'].to_pylist()))

    def test_metadata_and_completed_numeric_tampering_fail_closed(self):
        panel=build();panel.provenance['denominator_floor']=.02
        with self.assertRaises(ValueError):
            p.completed_training_slice(panel,train_start_month='2020-01',train_end_month='2020-01',as_of='2020-02-03',side='long')
        panel=build();rows=panel.table.to_pylist();rows[0]['target_value']=123.
        altered=p.ReturnDDPanel(pa.Table.from_pylist(rows,schema=panel.table.schema),panel.diagnostics,panel.provenance,panel.fingerprint)
        with self.assertRaises(ValueError):
            p.completed_training_slice(altered,train_start_month='2020-01',train_end_month='2020-01',as_of='2020-02-03',side='long')
        panel=build();rows=panel.table.to_pylist();rows[0]['intrinio_security_id']='WRONG'
        altered=p.ReturnDDPanel(pa.Table.from_pylist(rows,schema=panel.table.schema),panel.diagnostics,panel.provenance,panel.fingerprint)
        with self.assertRaises(ValueError):
            p.completed_training_slice(altered,train_start_month='2020-01',train_end_month='2020-01',as_of='2020-02-02',side='long')


def payoff_panel():
    rows=[]
    for i in range(120):
        month=p._shift('2010-01',i)
        group=fixture(n=10,month=month,sides=('long',))
        for row in group:
            # Unequal observation counts by month check equal-month weighting.
            row['cash_return'] += i/10000
            rows.append(row)
            if i%2:
                duplicate=dict(row);duplicate['source_security_id'] += 'extra';rows.append(duplicate)
    return build(rows)


class Payoffs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel=payoff_panel()

    def test_dual_full_120_month_grid_and_equal_month_means(self):
        state=p.build_return_dd_payoffs(self.panel,signal_date='2020-01-31',side='long')
        self.assertTrue(state.provenance['vector_available'])
        self.assertEqual(state.monthly.num_rows,1200)
        self.assertEqual(state.provenance['first_outcome_month'],'2010-01')
        self.assertEqual(state.provenance['last_outcome_month'],'2019-12')
        best=state.table.to_pylist()[0]
        self.assertAlmostEqual(best['class_mean_cash_return'],.1+59.5/10000)
        self.assertAlmostEqual(best['class_mean_ratio'],best['class_mean_cash_return']/.02)
        self.assertNotEqual(best['class_mean_ratio'],best['class_mean_cash_return'])

    def test_shorter_history_never_shrinks_gate(self):
        state=p.build_return_dd_payoffs(self.panel,signal_date='2019-12-31',side='long')
        self.assertFalse(state.provenance['vector_available'])
        self.assertTrue(all(row['class_mean_ratio'] is None and row['class_mean_cash_return'] is None
                            for row in state.table.to_pylist()))

    def test_one_missing_cell_withholds_both_entire_vectors(self):
        rows=fixture(month='2010-01',sides=('long',))
        for row in rows:row['cash_return']=.1
        all_rows=[]
        for i in range(120):
            all_rows.extend(rows if i==0 else fixture(month=p._shift('2010-01',i),sides=('long',)))
        state=p.build_return_dd_payoffs(build(all_rows),signal_date='2020-01-31',side='long')
        self.assertFalse(state.provenance['vector_available'])
        self.assertTrue(any(row['status']=='empty_realized_class' for row in state.monthly.to_pylist()))

    def test_actual_availability_must_precede_signal(self):
        # Last prior-month endpoint can be later than a month-start signal.
        state=p.build_return_dd_payoffs(self.panel,signal_date='2020-01-01',side='long')
        self.assertFalse(state.provenance['vector_available'])
        last=[row for row in state.monthly.to_pylist() if row['outcome_month']=='2019-12']
        self.assertTrue(all(row['status']=='scheduled_endpoint_not_observed' for row in last))

    def test_scores_explicit_distinct_and_class_column_mapping(self):
        state=p.build_return_dd_payoffs(self.panel,signal_date='2020-01-31',side='long')
        result=p.score_return_dd_probabilities([0.]*9+[1.],state,class_ids=list(range(10,0,-1)),
            signal_date='2020-01-31',as_of='2020-01-31')
        self.assertAlmostEqual(result['expected_return_over_drawdown'],state.table.to_pylist()[0]['class_mean_ratio'])
        self.assertAlmostEqual(result['expected_executable_cash_return'],state.table.to_pylist()[0]['class_mean_cash_return'])
        for kwargs in [{'as_of':'2020-01-30'}, {'class_ids':[1]*10}]:
            args={'class_ids':p.CLASSES,'signal_date':'2020-01-31','as_of':'2020-01-31',**kwargs}
            with self.assertRaises(ValueError):p.score_return_dd_probabilities([.1]*10,state,**args)

    def test_streaming_monthly_stats_and_payoffs_match_full_panel_api(self):
        stats=p.monthly_class_means(self.panel)
        self.assertEqual(len(stats),1200)
        full=p.build_return_dd_payoffs(self.panel,signal_date='2020-01-31',side='long')
        streamed=p.payoffs_from_monthly_means(stats,self.panel.provenance['windows'],
            signal_date='2020-01-31',side='long',observed_through='2023-12-31',
            source_policy_fingerprint=self.panel.fingerprint)
        self.assertEqual(streamed.table.to_pylist(),full.table.to_pylist())
        self.assertEqual(streamed.monthly.to_pylist(),full.monthly.to_pylist())
        self.assertEqual(stats[0]['original_members'],10)
        self.assertEqual(stats[0]['member_count'],1)
        self.assertAlmostEqual(stats[0]['mean_ratio'],5.)
        self.assertEqual(stats[10]['original_members'],20)
        self.assertEqual(stats[10]['member_count'],2)

    def test_streaming_monthly_processing_retains_same_statistics(self):
        all_rows=fixture(month='2020-01')+fixture(month='2020-02')
        together=build(all_rows)
        separate=[]
        for month in ('2020-01','2020-02'):
            separate.extend(p.monthly_class_means(build([row for row in all_rows if row['outcome_month']==month])))
        self.assertEqual(p.monthly_class_means(together),separate)

    def test_streaming_rejects_duplicate_invalid_or_forged_clock_cells(self):
        original=p.monthly_class_means(self.panel);windows=self.panel.provenance['windows']
        def run(stats):
            return p.payoffs_from_monthly_means(stats,windows,signal_date='2020-01-31',
                side='long',observed_through='2023-12-31')
        bad=copy.deepcopy(original);bad.append(dict(bad[0]))
        with self.assertRaises(ValueError):run(bad)
        for field,value in [('member_count',0),('mean_ratio',float('nan')),
                            ('label_available_at','2010-02-04')]:
            bad=copy.deepcopy(original);bad[0][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):run(bad)

    def test_streaming_missing_cell_and_incomplete_actual_clock_withhold(self):
        stats=p.monthly_class_means(self.panel);windows=self.panel.provenance['windows']
        incomplete=p.payoffs_from_monthly_means(stats[1:],windows,signal_date='2020-01-31',
            side='long',observed_through='2023-12-31')
        self.assertFalse(incomplete.provenance['vector_available'])
        self.assertTrue(any(row['status']=='missing_month_class_cell' for row in incomplete.monthly.to_pylist()))
        early=p.payoffs_from_monthly_means(stats,windows,signal_date='2020-01-01',
            side='long',observed_through='2023-12-31')
        self.assertFalse(early.provenance['vector_available'])
        self.assertTrue(any(row['status']=='scheduled_endpoint_not_observed' for row in early.monthly.to_pylist()))

    def test_streaming_future_and_other_side_means_are_never_inspected(self):
        class Explosive:
            def __float__(self):
                raise AssertionError('Future or other-side mean was inspected')
            def __repr__(self):
                raise AssertionError('Future or other-side mean was serialized')
        stats=p.monthly_class_means(self.panel)
        baseline=p.payoffs_from_monthly_means(stats,self.panel.provenance['windows'],
            signal_date='2020-01-31',side='long',observed_through='2023-12-31')
        future_panel=build(fixture(month='2020-01'))
        future=p.monthly_class_means(future_panel)
        for row in future:
            row['mean_ratio']=Explosive();row['mean_cash_return']=Explosive()
        other_side=[]
        for old in stats:
            row={**old,'side':'short','mean_ratio':Explosive(),'mean_cash_return':Explosive()}
            other_side.append(row)
        windows=self.panel.provenance['windows']+future_panel.provenance['windows']
        result=p.payoffs_from_monthly_means(stats+other_side+future,windows,
            signal_date='2020-01-31',side='long',observed_through='2023-12-31')
        self.assertEqual(result.table.to_pylist(),baseline.table.to_pylist())
        self.assertEqual(result.provenance['source_monthly_statistics_fingerprint'],
                         baseline.provenance['source_monthly_statistics_fingerprint'])
        self.assertEqual(result.provenance['inspected_monthly_class_cells'],1200)

    def test_streaming_in_window_but_uncompleted_means_are_never_inspected(self):
        class Explosive:
            def __float__(self):
                raise AssertionError('Uncompleted mean was inspected')
        stats=p.monthly_class_means(self.panel)
        for row in stats:
            if row['outcome_month']=='2019-12':
                row['mean_ratio']=Explosive();row['mean_cash_return']=Explosive()
        result=p.payoffs_from_monthly_means(stats,self.panel.provenance['windows'],
            signal_date='2020-01-01',side='long',observed_through='2023-12-31')
        self.assertFalse(result.provenance['vector_available'])
        self.assertEqual(result.provenance['inspected_monthly_class_cells'],1190)


if __name__=='__main__':
    unittest.main()
