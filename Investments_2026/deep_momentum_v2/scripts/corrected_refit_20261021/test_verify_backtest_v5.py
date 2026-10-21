"""Tiny fabricated verification tests; never read research tables or models."""
import calendar
import copy
import datetime as dt
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import backtest_verification_math_v2 as m
import verify_backtest_v5 as v


def payoff_fixture(signal='2019-12-31'):
    months = m.trailing_months(signal); rows = []; windows = []
    for i, month in enumerate(months):
        year, number = map(int, month.split('-'))
        end = f'{month}-{calendar.monthrange(year, number)[1]:02d}'
        windows.append({'outcome_month': month, 'outcome_end': end})
        rows.extend({'outcome_month': month, 'outcome_end': end, 'class_id': c,
            'original_member_count': 100, 'prediction_eligible_count': 90, 'member_count': 9,
            'class_mean_return': .001 * (c - 5) + i * .00001, 'status': 'available'} for c in range(1, 11))
    provenance = {'signal_date': signal, 'label_observed_through': signal,
        'first_outcome_month': months[0], 'last_outcome_month': months[-1],
        'expected_calendar_months': 120, 'expected_classes': list(range(1, 11)), 'expected_month_class_cells': 1200,
        'monthly_class_mean': 'equal_weight_observed_realized_class_returns',
        'payoff_time_weighting': 'equal_calendar_month_weights',
        'missing_month_or_class_policy': 'withhold_entire_payoff_vector',
        'uses_predicted_classes': False, 'financial_rows_after_cutoff_consumed': False,
        'return_basis': 'provisional_provider_adjusted_endpoint_ratio',
        'economic_return_certified': False, 'knowledge_time_certified': False,
        'same_close_operational_availability_certified': False, 'model_ready': False,
        'estimator_fit_performed': False, 'source_completion_windows': windows,
        'vector_available': True, 'available_month_class_cells': 1200}
    import math
    table = [{'class_id': c, 'payoff': math.fsum(r['class_mean_return']/120 for r in rows if r['class_id'] == c),
              'status': 'available'} for c in range(1, 11)]
    state = {'provenance': provenance}
    state['fingerprint'] = m.canonical({'table': table, 'monthly': rows, 'provenance': provenance})
    return rows, table, state


def windows_fixture(year=2020):
    rows = []
    for month in range(1, 13):
        first = dt.date(year, month, 1); previous = first - dt.timedelta(days=1)
        rows.append({'signal_date': previous.isoformat(), 'outcome_month': f'{year}-{month:02d}',
            'outcome_start': first.isoformat(), 'outcome_end': f'{year}-{month:02d}-{calendar.monthrange(year, month)[1]:02d}'})
    return rows


def forecast_fixture():
    return [{**r, 'source_security_id': 'source', 'intrinio_security_id': 'anything',
        'identity_eligible': True, 'paper_prediction_eligible': True, 'prediction_eligible': True,
        'prediction_status': 'predicted', 'fit_year': 2020, 'training_outcome_start': '2010-01-01',
        'training_outcome_end': '2019-12-31', 'training_label_observed_through': '2019-12-31'} for r in windows_fixture()]


class Math(unittest.TestCase):
    def test_exact_120_month_equal_mean_and_logical_fingerprint(self):
        rows, table, state = payoff_fixture()
        actual = m.payoff_values(rows, table, state, '2019-12-31')
        self.assertEqual(actual['mu'], [r['payoff'] for r in table])
        self.assertEqual(len(rows), 1200)

    def test_missing_duplicate_class_grid_rejected_before_values(self):
        rows, table, state = payoff_fixture(); rows[1] = rows[0].copy()
        with self.assertRaisesRegex(ValueError, '120-calendar-month'):
            m.payoff_metadata(rows, table, state, '2019-12-31')

    def test_future_or_wrong_signal_payoff_rejected(self):
        rows, table, state = payoff_fixture(); state['provenance']['label_observed_through'] = '2020-01-01'
        with self.assertRaisesRegex(ValueError, 'causal scope'):
            m.payoff_metadata(rows, table, state, '2019-12-31')
        rows, table, state = payoff_fixture(); rows[-1]['outcome_end'] = '2020-01-01'
        with self.assertRaisesRegex(ValueError, 'completion clock'):
            m.payoff_metadata(rows, table, state, '2019-12-31')

    def test_bool_coverage_count_does_not_equal_one(self):
        rows, table, state = payoff_fixture(); rows[0]['member_count'] = True
        with self.assertRaisesRegex(ValueError, 'coverage count'):
            m.payoff_metadata(rows, table, state, '2019-12-31')

    def test_one_missing_cell_withholds_entire_vector_without_imputation(self):
        rows, table, state = payoff_fixture(); rows[0].update(status='empty_realized_class', class_mean_return=None, member_count=0)
        state['provenance'].update(vector_available=False, available_month_class_cells=1199)
        for r in table: r.update(payoff=None, status='withheld_incomplete_120_month_class_grid')
        state['fingerprint'] = m.canonical({'table': table, 'monthly': rows, 'provenance': state['provenance']})
        self.assertIsNone(m.payoff_values(rows, table, state, '2019-12-31')['mu'])

    def test_modified_payoff_and_provenance_fingerprint_rejected(self):
        rows, table, state = payoff_fixture(); table[0]['payoff'] += 1e-15
        with self.assertRaisesRegex(ValueError, 'exact equal-month'):
            m.payoff_values(rows, table, state, '2019-12-31')
        rows, table, state = payoff_fixture(); state['provenance']['extra_claim'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'fingerprint'):
            m.payoff_values(rows, table, state, '2019-12-31')

    def test_future_target_availability_never_changes_forecast_admission(self):
        a = forecast_fixture(); b = copy.deepcopy(a)
        for r in b: r['target_available'] = False; r['future_quote_available'] = False
        self.assertEqual(m.forecast_metadata(a, 2020, windows_fixture()), m.forecast_metadata(b, 2020, windows_fixture()))

    def test_bk_and_typed_xom_preserved_without_provider_id_prefix_rule(self):
        rows = forecast_fixture(); rows[0].update(source_security_id=m.XOM_SOURCE, intrinio_security_id=m.XOM_NATIVE)
        rows[1].update(source_security_id=m.BK_SOURCE, intrinio_security_id=m.BK_NATIVE)
        m.forecast_metadata(rows, 2020, windows_fixture())
        rows[0]['intrinio_security_id'] = 'sec_wrong'
        with self.assertRaisesRegex(ValueError, 'Typed SIP'):
            m.forecast_metadata(rows, 2020, windows_fixture())

    def test_source_year_positions_clip_future_and_non_session_metadata(self):
        sessions = [(dt.date(2020, 1, 1) + dt.timedelta(days=i)).isoformat() for i in range(1006)]
        rows = [{'trading_date': d, 'intrinio_security_id': m.XOM_NATIVE} for d in ('2019-12-31', '2020-01-02', '2024-01-02')]
        positions, outside = m.quote_positions(rows, 2020, sessions)
        self.assertEqual(positions, [1]); self.assertEqual(len(outside), 2)

    def test_saved_table_scope_rejects_future_before_financial_values(self):
        with self.assertRaisesRegex(ValueError, 'protected financial date'):
            m.generic_table_metadata([{'date': '2024-01-02'}], ['date'])
        with self.assertRaisesRegex(ValueError, 'yearly metric'):
            m.generic_table_metadata([{'year': 2024}], ['year'])

    def test_saved_midnight_timestamp_is_guarded_without_financial_conversion(self):
        self.assertTrue(m.generic_table_metadata([{'date': dt.datetime(2020, 1, 2)}], ['date']))
        with self.assertRaisesRegex(ValueError, 'Intraday'):
            m.generic_table_metadata([{'date': dt.datetime(2020, 1, 2, 1)}], ['date'])

    def test_unicode_canonical_hash_matches_declared_legacy_policy(self):
        self.assertNotEqual(m.canonical({'name': 'é'}), m.canonical({'name': '\\u00e9'}))


class PreflightSummaryClockException(unittest.TestCase):
    def fixture(self):
        rows = [{'arm': arm, 'basket': basket} for arm in m.ARMS for basket in m.BASKETS]
        source = {'path': '/invented/source-plan.json', 'sha256': '0'*64, 'size_bytes': 1}
        scope = {'sessions': 1006, 'outcome_months': 48, 'selection_sets': 15,
            'source_plan_pin': source, 'preflight_source_plan_pin': source, 'replay_source_plan_pin': source}
        return rows, scope

    def test_only_fifteen_coverage_summary_rows_can_inherit_run_clock(self):
        rows, scope = self.fixture()
        self.assertTrue(m.preflight_summary_metadata(rows, ['arm', 'basket'], m.PREFLIGHT_SUMMARY_SCHEMA, run_scope=scope))

    def test_closed_string_width_allowance_keeps_other_types_exact(self):
        rows, scope = self.fixture(); schema = dict(m.PREFLIGHT_SUMMARY_SCHEMA)
        schema['arm'] = 'large_string'; schema['basket'] = 'large_string'
        self.assertTrue(m.preflight_summary_metadata(rows, ['arm', 'basket'], schema, run_scope=scope))
        schema['selected_episode_rows'] = 'double'
        with self.assertRaisesRegex(ValueError, '17-field'):
            m.preflight_summary_metadata(rows, ['arm', 'basket'], schema, run_scope=scope)

    def test_extra_financial_or_unknown_field_is_never_clock_exempt(self):
        rows, scope = self.fixture(); schema = {**m.PREFLIGHT_SUMMARY_SCHEMA, 'total_return': 'double'}
        with self.assertRaisesRegex(ValueError, '17-field'):
            m.preflight_summary_metadata(rows, ['arm', 'basket'], schema, run_scope=scope)

    def test_incomplete_or_duplicate_summary_grid_rejected(self):
        rows, scope = self.fixture(); rows[-1] = rows[0]
        with self.assertRaisesRegex(ValueError, 'fifteen'):
            m.preflight_summary_metadata(rows, ['arm', 'basket'], m.PREFLIGHT_SUMMARY_SCHEMA, run_scope=scope)

    def test_wrong_run_scope_is_rejected(self):
        rows, scope = self.fixture(); scope['sessions'] = 1005
        with self.assertRaisesRegex(ValueError, '48-month/1006'):
            m.preflight_summary_metadata(rows, ['arm', 'basket'], m.PREFLIGHT_SUMMARY_SCHEMA, run_scope=scope)

    def test_wrong_source_phase_binding_is_rejected(self):
        rows, scope = self.fixture(); scope['preflight_source_plan_pin'] = {'sha256': 'different'}
        with self.assertRaisesRegex(ValueError, '48-month/1006'):
            m.preflight_summary_metadata(rows, ['arm', 'basket'], m.PREFLIGHT_SUMMARY_SCHEMA, run_scope=scope)

    def test_generic_clock_guard_remains_strict(self):
        rows, _ = self.fixture()
        with self.assertRaisesRegex(ValueError, 'lacks a scoped clock'):
            m.generic_table_metadata(rows, ['arm', 'basket'])

    def test_only_exact_artifact_role_uses_the_summary_branch(self):
        import pyarrow as pa
        rows, scope = self.fixture(); table = pa.Table.from_pylist(rows)
        pair = {'metadata': {}, 'values': {'schema_fields': dict(m.PREFLIGHT_SUMMARY_SCHEMA)}}
        with patch.object(v, 'read_projection', return_value=table):
            digest = v.guarded_metadata(pair, context=('preflight', 'summary.parquet'), run_scope=scope)
            self.assertEqual(digest, v.metadata_fingerprint(table))
            with self.assertRaisesRegex(ValueError, 'replay/holdings.parquet.*lacks a scoped clock'):
                v.guarded_metadata(pair, context=('replay', 'holdings.parquet'), run_scope=scope)


class Planning(unittest.TestCase):
    def test_conflicting_path_pins_fail(self):
        a = {'path': '/same', 'sha256': 'one', 'size_bytes': 1}; b = {**a, 'sha256': 'two'}
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            v.collect_pins([a, b])
        self.assertEqual(v.collect_pins([a, a]), [a])

    def test_financial_json_parse_cannot_precede_reservation(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            p = Path(d)/'plan.json'
            opaque = Path(d)/'opaque.json'; opaque.write_text('metric values must never parse')
            item = v.pin(opaque)
            v.save_new(p, {'schema': 'dmv2-corrected-backtest-verification-JSON-plan/v1',
                'worker_pin': v.pin(v.__file__), 'math_pin': v.pin(m.__file__),
                'named_JSON_pins': {'opaque': item}, 'unique_JSON_pins': [item],
                'projected_reads': {'compressed_source': item['size_bytes'], 'derived_matrix': 0}})
            with patch.object(v, 'require_reservation', side_effect=ValueError('no reservation')), \
                 patch.object(v, 'parsed_json', side_effect=AssertionError('financial JSON parsed')) as parser:
                with self.assertRaisesRegex(ValueError, 'no reservation'):
                    v.make_plan(p, v.pin(p)['sha256'], 'invented', Path(d)/'output.json')
                parser.assert_not_called()

    def test_json_budget_cannot_omit_or_undercharge_a_metric_file(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            p = Path(d)/'opaque.json'; p.write_text('not parsed')
            item = v.pin(p)
            z = {'named_JSON_pins': {'one': item, 'same': item}, 'unique_JSON_pins': [item],
                'projected_reads': {'compressed_source': item['size_bytes'], 'derived_matrix': 0}}
            self.assertEqual(v.validate_json_budget(z), item['size_bytes'])
            z['projected_reads']['compressed_source'] -= 1
            with self.assertRaisesRegex(ValueError, 'byte sum'):
                v.validate_json_budget(z)
            z['unique_JSON_pins'] = []
            with self.assertRaisesRegex(ValueError, 'distinct pin set'):
                v.validate_json_budget(z)

    def test_compact_metadata_fingerprint_survives_chunk_layout_only(self):
        import pyarrow as pa
        a = pa.table({'date': ['2020-01-02', '2020-01-03']})
        b = pa.concat_tables([a.slice(0, 1), a.slice(1)])
        self.assertEqual(v.metadata_fingerprint(a), v.metadata_fingerprint(b))
        self.assertNotEqual(v.metadata_fingerprint(a), v.metadata_fingerprint(a.slice(0, 1)))

    def test_value_metadata_mismatch_fails_before_pandas(self):
        table = Mock(); table.select.return_value.to_pylist.return_value = [{'date': '2020-01-02'}]
        with patch.object(v, 'read_projection', return_value=table):
            with self.assertRaisesRegex(ValueError, 'metadata/order'):
                v.read_values({'values': {}, 'metadata': {'columns': ['date']}}, [{'date': '2020-01-03'}])
        table.to_pandas.assert_not_called()

    def test_nested_footer_chunks_are_counted_without_value_read(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            path = Path(d)/'toy.parquet'; pq.write_table(pa.Table.from_pylist([{'safe': ['x', 'y'], 'price': 123.}]), path)
            with patch.object(pq.ParquetFile, 'read', side_effect=AssertionError('no numerical read at plan')):
                p = v.projection(v.pin(path), ['safe'], 10)
            group = pq.ParquetFile(path).metadata.row_group(0)
            self.assertEqual(p['projected_compressed_bytes'], sum(group.column(i).total_compressed_size
                for i in range(group.num_columns) if group.column(i).path_in_schema.split('.')[0] == 'safe'))


class SavedRepresentation(unittest.TestCase):
    def monthlies(self):
        import pandas as pd
        class_monthly = pd.DataFrame([{'signal_date': '2019-12-31', 'basket': 'class_1_10',
            'class_only': 1., 'selected_names_per_side': 1}])
        ret_monthly = pd.DataFrame([{'signal_date': '2019-12-31', 'basket': 'ret',
            'available_names_per_side': 2, 'selected_names_per_side': 2, 'ret_only': .1}])
        return {('rf_paper15', 'ret'): {'monthly': ret_monthly},
                ('rf_paper15', 'class_1_10'): {'monthly': class_monthly}}

    def test_registered_union_uses_sorted_groups_and_original_column_order(self):
        import pandas as pd
        frames = self.monthlies()
        actual = pd.concat([frames[key]['monthly'].assign(arm=key[0], basket=key[1])
            for key in sorted(frames)], ignore_index=True)
        expected = v.reconstruct_coverage(frames)
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
        self.assertEqual(expected.basket.tolist(), ['class_1_10', 'ret'])
        self.assertEqual(expected.columns.tolist(), ['signal_date', 'basket', 'class_only',
            'selected_names_per_side', 'arm', 'available_names_per_side', 'ret_only'])
        self.assertTrue(pd.isna(expected.iloc[0].ret_only))

    def test_union_comparison_preserves_null_padding_dtype_and_unregistered_field_checks(self):
        import pandas as pd
        expected = v.reconstruct_coverage(self.monthlies())
        for kind in ('null_padding', 'dtype', 'extra_column', 'row_order'):
            bad = expected.copy()
            if kind == 'null_padding': bad.loc[0, 'ret_only'] = 0.
            elif kind == 'dtype': bad['selected_names_per_side'] = bad.selected_names_per_side.astype(float)
            elif kind == 'extra_column': bad['unregistered'] = None
            else: bad = bad.iloc[::-1].reset_index(drop=True)
            with self.subTest(kind=kind), self.assertRaises(AssertionError):
                pd.testing.assert_frame_equal(bad, expected, check_exact=True)

    def test_only_date_fields_normalize_midnight_datetimes_and_dates(self):
        import pandas as pd
        expected = [{'date': '2020-01-02', 'signal_date': '2019-12-31',
                     'execution_date': '2020-01-02', 'reason': '2020-01-02', 'successor_id': None}]
        actual = [{'date': pd.Timestamp('2020-01-02'), 'signal_date': dt.date(2019, 12, 31),
                   'execution_date': dt.datetime(2020, 1, 2), 'reason': '2020-01-02', 'successor_id': None}]
        v.rows_equal(actual, expected, 'invented preflight metadata')

    def test_timezone_intraday_and_nanosecond_dates_are_rejected(self):
        import pandas as pd
        values = [dt.datetime(2020, 1, 2, 1), dt.datetime(2020, 1, 2, tzinfo=dt.timezone.utc),
                  pd.Timestamp('2020-01-02 00:00:00.000000001')]
        for value in values:
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'timezone/intraday'):
                v.rows_equal([{'date': value}], [{'date': '2020-01-02'}], 'invented')

    def test_arbitrary_objects_never_receive_default_string_conversion(self):
        class Object:
            def __str__(self): raise AssertionError('No string coercion allowed')
        for field in ('date', 'reason'):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'arbitrary object'):
                v.rows_equal([{field: Object()}], [{field: '2020-01-02'}], 'invented')

    def test_timestamp_in_unregistered_field_and_protected_dates_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'outside date fields'):
            v.rows_equal([{'reason': dt.datetime(2020, 1, 2)}], [{'reason': '2020-01-02'}], 'invented')
        with self.assertRaisesRegex(ValueError, 'protected financial date'):
            v.rows_equal([{'date': dt.datetime(2024, 1, 2)}], [{'date': '2024-01-02'}], 'invented')

    def test_none_missingness_is_not_stringified(self):
        v.rows_equal([{'date': None, 'successor_id': None}], [{'date': None, 'successor_id': None}], 'invented')
        with self.assertRaisesRegex(ValueError, 'logical rows changed'):
            v.rows_equal([{'date': None}], [{'date': '2020-01-02'}], 'invented')


class FailedMetricNullRepresentation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Code-only authenticated import. Tests below contain fabricated values.
        import hashlib
        import importlib.util
        path = Path('/Users/lajosgalambos/Desktop/Finance-Working-Files/Investments_2026/deep_momentum_v2/scripts/verify_walkforward_backtest_v2.py')
        if hashlib.sha256(path.read_bytes()).hexdigest() != v.HELPERS['cash_units'][1]:
            raise AssertionError('Preserved independent helper code changed')
        spec = importlib.util.spec_from_file_location('tiny_preserved_failed_metrics', path)
        cls.legacy = importlib.util.module_from_spec(spec); spec.loader.exec_module(cls.legacy)

    def fixture(self, prefix=False):
        import numpy as np
        import pandas as pd
        nav = [1.1, 1.2, None, None] if prefix else [None]*4
        daily = pd.DataFrame({'date': ['2020-01-02','2020-01-31','2020-02-03','2020-02-28'],
            'nav': pd.Series(nav, dtype=object), 'untouched': ['a','b','c','d']})
        monthly = pd.DataFrame({'outcome_month':['2020-01','2020-02'],
            'nav': pd.Series([1.2,None] if prefix else [None,None],dtype=object),
            'return':pd.Series([.2,None] if prefix else [None,None],dtype=object)})
        fields = ('final_nav','total_return','annualized_arithmetic_mean','cagr','annualized_volatility',
            'sharpe_zero_cash_rate','maximum_drawdown','total_one_way_turnover','trading_cost_paid')
        summary = {'status':'failed_closed','complete_proxy_path':False,**{k:None for k in fields}}
        annual = pd.DataFrame([{**{k:None for k in ('initial_nav','final_nav','total_return',
            'annualized_volatility','sharpe_zero_cash_rate','maximum_drawdown','trading_cost_paid')},
            'complete_full_period_proxy':False}])
        return daily,monthly,summary,annual

    def test_all_missing_float_copies_pass_real_legacy_helper(self):
        import pandas as pd
        daily,monthly,summary,annual=self.fixture(); d0=daily.copy(deep=True); m0=monthly.copy(deep=True)
        d,m=v.failed_metrics_float_views(daily,monthly)
        self.legacy.verify_failed_metrics(d,pd.DataFrame(),m,summary,annual)
        self.assertEqual(str(d.nav.dtype),'float64'); self.assertEqual(str(m.nav.dtype),'float64')
        self.assertEqual(str(m['return'].dtype),'float64')
        pd.testing.assert_frame_equal(daily,d0); pd.testing.assert_frame_equal(monthly,m0)
        self.assertTrue(d.nav.isna().all()); self.assertTrue(m.nav.isna().all())
        pd.testing.assert_series_equal(d.untouched,daily.untouched)

    def test_finite_prefix_then_missing_passes_real_legacy_helper(self):
        import pandas as pd
        daily,monthly,summary,annual=self.fixture(prefix=True); d,m=v.failed_metrics_float_views(daily,monthly)
        self.legacy.verify_failed_metrics(d,pd.DataFrame(),m,summary,annual)
        self.assertEqual(d.nav.iloc[:2].tolist(),[1.1,1.2]); self.assertEqual(m.nav.iloc[0],1.2)
        self.assertEqual(d.nav.isna().tolist(),daily.nav.isna().tolist())
        self.assertEqual(m.nav.isna().tolist(),monthly.nav.isna().tolist())
        self.assertEqual(m['return'].isna().tolist(),monthly['return'].isna().tolist())

    def test_allowed_null_sentinels_preserve_only_their_masks(self):
        import numpy as np
        import pandas as pd
        daily=pd.DataFrame({'nav':pd.Series([None,np.nan,pd.NA,np.float32('nan'),0.,-0.,2],dtype=object)})
        monthly=pd.DataFrame({'nav':[None],'return':[None]})
        d,m=v.failed_metrics_float_views(daily,monthly)
        self.assertEqual(d.nav.isna().tolist(),[True]*4+[False]*3)
        self.assertEqual(d.nav.iloc[4:].tolist(),[0.,-0.,2.]); self.assertTrue(np.signbit(d.nav.iloc[5]))

    def test_poison_scalars_rejected_in_each_registered_column(self):
        import numpy as np
        import pandas as pd
        for poison in (True,np.bool_(False),'1.0',object(),complex(1),np.inf,-np.inf,2**53+1,10**400):
            for which in ('daily','monthly_nav','monthly_return'):
                with self.subTest(poison=repr(poison),which=which):
                    daily=pd.DataFrame({'nav':pd.Series([1.],dtype=object)})
                    monthly=pd.DataFrame({'nav':pd.Series([1.],dtype=object),'return':pd.Series([0.],dtype=object)})
                    if which=='daily': daily.at[0,'nav']=poison
                    elif which=='monthly_nav': monthly.at[0,'nav']=poison
                    else: monthly.at[0,'return']=poison
                    with self.assertRaisesRegex(ValueError,'Failed-metric float view'):
                        v.failed_metrics_float_views(daily,monthly)

    def test_adapter_is_only_on_failed_metrics_call_boundary(self):
        import inspect
        source=inspect.getsource(v.verify)
        self.assertEqual(source.count('failed_metrics_float_views('),1)
        self.assertIn("equations.verify_failed_scenario(tables, prices",source)
        self.assertIn("equations.verify_metrics(tables['daily'], tables['trades'], monthly, summary, annual)",source)
        self.assertIn('equations.verify_failed_metrics(metric_daily,',source)


class ArrowPreflightNullRepresentation(unittest.TestCase):
    def records(self):
        base={'signal_date':'2019-12-31','source_security_id':'invented',
            'date':'2020-01-15','status':'verified_retirement_definition_required_and_present'}
        return [{**base,'event_id':'cash','predecessor_id':'old_cash','successor_id':None},
                {**base,'event_id':'stock','predecessor_id':'old_stock','successor_id':'new_stock'}]

    def pair(self, path):
        import pyarrow.parquet as pq
        names=pq.ParquetFile(path).schema_arrow.names; item=v.pin(path)
        return {'metadata':v.projection(item,['signal_date','date','event_id'],10),
                'values':v.projection(item,names,10)}

    def test_mixed_cash_stock_parquet_preserves_arrow_null_before_strict_rows_compare(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            path=Path(d)/'cash_stock.parquet'; expected=self.records()
            pq.write_table(pa.Table.from_pylist(expected),path)
            pair=self.pair(path); fingerprint=v.metadata_fingerprint(v.read_projection(pair['metadata']))
            actual=v.read_preflight_records(pair,fingerprint,name='action_requirements')
            self.assertIsNone(actual[0]['successor_id']);self.assertEqual(actual[1]['successor_id'],'new_stock')
            v.rows_equal(actual,expected,'mixed cash/stock metadata')
            # No saved bytes are modified and no nullable identifiers are stringified.
            self.assertEqual(actual,pq.ParquetFile(path).read().to_pylist())
            self.assertEqual(v.pin(path),pair['values']['pin'])

    def test_true_nan_and_infinity_remain_rejected_by_unchanged_comparer(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        for poison in (float('nan'),float('inf'),-float('inf')):
            with self.subTest(poison=poison),tempfile.TemporaryDirectory(dir='/private/tmp') as d:
                path=Path(d)/'poison.parquet'; row=self.records()[0];row['successor_id']=poison
                pq.write_table(pa.Table.from_pylist([row]),path)
                pair=self.pair(path); fingerprint=v.metadata_fingerprint(v.read_projection(pair['metadata']))
                actual=v.read_preflight_records(pair,fingerprint,name='action_requirements')
                with self.assertRaises(ValueError): v.rows_equal(actual,[self.records()[0]],'poison')

    def test_metadata_fingerprint_mutation_fails_before_record_conversion(self):
        table=Mock(); table.column_names=list(self.records()[0]);table.num_rows=2
        pair={'metadata':{'columns':['date']},'values':{'columns':table.column_names}}
        with patch.object(v,'read_projection',return_value=table),patch.object(v,'metadata_fingerprint',return_value='changed'):
            with self.assertRaisesRegex(ValueError,'metadata/order'):
                v.read_preflight_records(pair,'prior',name='action_requirements')
        table.to_pylist.assert_not_called()

    def test_foreign_or_financial_fields_cannot_use_metadata_record_reader(self):
        for name,fields in [('summary',[]),('action_requirements',['date','cash_per_raw_share']),
                ('missing_quotes',['date','price'])]:
            with self.subTest(name=name),patch.object(v,'read_projection',side_effect=AssertionError('must reject before read')):
                with self.assertRaisesRegex(ValueError,'registered preflight metadata'):
                    v.read_preflight_records({'values':{'columns':fields}},'prior',name=name)

    def test_nonempty_requires_exact_registered_schema_but_empty_diagnostics_remain_empty(self):
        import pyarrow as pa
        pair={'values':{'columns':['date']},'metadata':{'columns':['date']}}
        table=pa.table({'date':['2020-01-15']})
        with patch.object(v,'read_projection',return_value=table),self.assertRaisesRegex(ValueError,'exact registered'):
            v.read_preflight_records(pair,v.metadata_fingerprint(table),name='action_requirements')
        empty=pa.table({});pair={'values':{'columns':[]},'metadata':{'columns':[]}}
        with patch.object(v,'read_projection',return_value=empty):
            self.assertEqual(v.read_preflight_records(pair,v.metadata_fingerprint(empty),name='action_requirements'),[])

    def test_global_row_comparison_and_accounting_rules_are_not_changed(self):
        import inspect
        import verify_backtest_v4 as old
        self.assertEqual(inspect.getsource(v.rows_equal),inspect.getsource(old.rows_equal))
        source=inspect.getsource(v.verify)
        self.assertEqual(source.count('read_preflight_records('),1)
        self.assertNotIn("read_values(specs['preflight'][key]",source)
        self.assertIn('equations.verify_failed_scenario(tables, prices',source)
        self.assertIn("equations.verify_metrics(tables['daily'], tables['trades'], monthly, summary, annual)",source)


class FreshV5Attempt(unittest.TestCase):
    def test_versioned_marker_preserves_old_attempt_is_single_use_and_keeps_proof_fresh(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            root=Path(d);output=root/'run/verification/backtest.json';output.parent.mkdir(parents=True)
            old=Path(str(output)+'.STARTED.json');old.write_bytes(b'preserve failed V4 attempt')
            old_pin=v.pin(old); plan={'path':'/invented/new-v5-plan.json','sha256':'a'*64,'size_bytes':123}
            with patch.object(v,'ROOT',root):
                marker=v.begin_verification_attempt(output,plan)
                self.assertEqual(Path(marker['path']),Path(str(output)+'.v5.STARTED.json'))
                self.assertEqual(v.read(marker['path'])['plan_pin'],plan)
                self.assertEqual(v.read(marker['path'])['verifier_revision'],5)
                with self.assertRaises(FileExistsError):v.begin_verification_attempt(output,plan)
                self.assertFalse(output.exists());self.assertEqual(v.pin(old),old_pin)
                output.write_bytes(b'preserve final proof')
                with self.assertRaisesRegex(ValueError,'Fresh canonical'):v.begin_verification_attempt(output,plan)
                self.assertEqual(output.read_bytes(),b'preserve final proof')


if __name__ == '__main__': unittest.main()
