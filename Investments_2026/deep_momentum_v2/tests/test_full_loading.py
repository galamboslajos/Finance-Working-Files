"""Adversarial source partition tests using structural metadata/quotes only."""
import unittest
from dataclasses import replace
from unittest.mock import Mock, patch

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from dmv2.full_loading import (FullFeatureSourcePlan, _daily_cohorts,
    annual_feature_calendar, estimate_bounded_read, load_feature_year,
    plan_full_feature_sources, prepare_feature_year)
from dmv2.identity_policy import PROVISIONAL_IDENTITY_POLICY, qualify_identity_records
from dmv2.loading import BudgetExceeded, ProtectedDataError, ReadBudget
from dmv2.universe import DataContractError, KnowledgeTimeUnavailable, build_snapshot
from test_monthly_loading import fixtures, loaded


class FullLoadingGuards(unittest.TestCase):
    def test_invalid_scope_fails_before_reader_or_catalog(self):
        for changes, error in [({'start_year': 2000}, DataContractError),
                ({'start_year': True}, DataContractError),
                ({'end_year': 2009}, DataContractError),
                ({'end_year': 2024}, ProtectedDataError),
                ({'universe': 'sp1500'}, DataContractError),
                ({'history_sessions': 252}, DataContractError),
                ({'mode': 'knowledge_time'}, KnowledgeTimeUnavailable),
                ({'identity_policy': 'unknown'}, DataContractError)]:
            with self.subTest(changes=changes), patch('dmv2.full_loading.BoundedTableReader') as reader:
                catalog = Mock()
                with self.assertRaises(error):
                    plan_full_feature_sources(catalog, **changes)
                reader.assert_not_called()
                self.assertEqual(catalog.mock_calls, [])

    def test_estimate_protected_before_footer(self):
        catalog = Mock()
        with patch('dmv2.full_loading.BoundedTableReader') as reader:
            with self.assertRaises(ProtectedDataError):
                estimate_bounded_read(catalog, 'intrinio', 'intrinio_daily_prices',
                    start='2023-01-01', end='2024-01-01', columns=('adj_close',),
                    allowed_columns=('adj_close',))
            reader.assert_not_called()

    def test_financial_estimate_filter_refused_before_footers(self):
        with patch('dmv2.full_loading.BoundedTableReader') as reader:
            with self.assertRaises(DataContractError):
                estimate_bounded_read(Mock(), 'intrinio', 'intrinio_daily_prices',
                    start='2010-01-01', end='2010-12-31', columns=('adj_close',),
                    allowed_columns=('adj_close',), filters={'adj_close':(1.,)})
            reader.assert_not_called()


class DailyCohortTests(unittest.TestCase):
    def setUp(self):
        source = fixtures()
        self.members = source['membership'].table
        self.bridge = source['bridge'].table
        self.links = source['links'].table
        self.dates = ('1999-12-31', '2000-01-03', '2000-01-31')

    def cohorts(self):
        return _daily_cohorts(self.members, self.bridge, self.links,
                             self.dates, 'combined', 'historical_reconstruction')

    def test_each_observed_day_matches_original_snapshot_identity_and_cohort(self):
        result, audit = self.cohorts()
        columns = ['source_security_id', 'indices', 'unified_security_id',
            'intrinio_security_id', 'identity_status', 'identity_review_required', 'resolution_status']
        for day in ('1999-12-31', '2000-01-31'):
            expected = build_snapshot(self.members, self.bridge, self.links,
                as_of=day, universe='combined').table.select(columns).to_pylist()
            actual = result.filter(pc.equal(result['signal_date'], day)).select(columns).to_pylist()
            self.assertEqual(actual, expected)
        self.assertEqual(audit['original_daily_rows'], 6)

    def test_missing_daily_membership_not_carried_forward(self):
        result, audit = self.cohorts()
        self.assertEqual(audit['missing_dates'], ['2000-01-03'])
        self.assertEqual(set(result['signal_date'].to_pylist()), {'1999-12-31', '2000-01-31'})
        self.assertFalse(audit['membership_filled'])

    def test_future_availability_excluded_without_hiding_original_source(self):
        source = self.members.to_pylist()
        source[0]['membership_availability_date'] = '2000-01-01'
        self.members = pa.Table.from_pylist(source)
        result, audit = self.cohorts()
        self.assertEqual(result.num_rows, 5)
        self.assertEqual(audit['membership_clock_exclusions'][0]['reason'], 'future_availability')
        self.assertEqual(self.members.num_rows, 6)

    def test_same_date_collision_retained_and_blocked(self):
        rows = self.links.to_pylist()
        rows[0]['intrinio_security_id'] = 'p-b'
        self.links = pa.Table.from_pylist(rows)
        result, audit = self.cohorts()
        first = result.filter(pc.equal(result['signal_date'], '1999-12-31')).to_pylist()
        qualified = qualify_identity_records(first, identity_policy=PROVISIONAL_IDENTITY_POLICY)
        self.assertEqual(sum(r['identity_status']=='target_collision' for r in qualified), 2)
        self.assertEqual(sum(r['identity_eligible'] for r in qualified), 0)
        self.assertEqual(audit['original_daily_rows'], 6)

    def test_overlap_original_index_list_and_clock_aggregates_retained(self):
        rows = self.members.to_pylist()
        other = dict(rows[0], index_id='sp400', membership_effective_date='1999-12-15')
        rows.append(other)
        self.members = pa.Table.from_pylist(rows)
        result, _ = self.cohorts()
        row = next(r for r in result.to_pylist()
                   if r['signal_date']=='1999-12-31' and r['source_security_id']=='z')
        self.assertEqual(row['indices'], ['sp400', 'sp500'])
        self.assertEqual(row['membership_record_count'], 2)
        self.assertEqual(row['membership_effective_date'], '1999-12-15')


class AnnualPartitionTests(unittest.TestCase):
    def setUp(self):
        self.source = fixtures()
        snapshots = tuple(build_snapshot(self.source['membership'].table,
            self.source['bridge'].table, self.source['links'].table,
            as_of=day, universe='combined') for day in ('1999-12-31', '2000-01-31'))
        self.request = {'year': 2010, 'signal_dates': [s.as_of for s in snapshots],
            'raw_start': '1999-12-30', 'daily_context_start': '1999-12-31',
            'feature_end': '2000-01-31', 'raw_end': '2000-02-29'}
        self.catalog = Mock()
        self.catalog.provenance.return_value = {'source': 'fixture-pin'}
        self.plan = FullFeatureSourcePlan(self.catalog, (2010,), (), snapshots,
            self.source['calendar'].table, self.source['calendar'].table,
            self.source['bridge'].table, self.source['links'].table,
            {2010: self.request}, {}, {'request_sha256': 'fixture-plan-pin'},
            PROVISIONAL_IDENTITY_POLICY, 'combined')
        rows = self.source['prices'].table.to_pylist()
        rows += [dict(rows[0], trading_date='2000-02-29', adj_close=999.),
                 dict(rows[0], trading_date='2000-01-15', adj_close=888.)]
        self.raw = loaded(pa.Table.from_pylist(rows), 'prices')
        self.reader = Mock()
        self.reader.read.return_value = self.source['membership']
        self.price = Mock()
        self.price.load.return_value = self.raw

    def run_year(self, **kwargs):
        with patch('dmv2.full_loading.BoundedTableReader', return_value=self.reader), \
                patch('dmv2.full_loading.PriceLoader', return_value=self.price), \
                patch('dmv2.full_loading.estimate_bounded_read', return_value={'footer_only':True}):
            return load_feature_year(self.plan, 2010, **kwargs)

    def test_raw_daily_unchanged_and_future_outcome_filtered_before_feature_worker(self):
        result = self.run_year()
        self.assertIs(result.raw_daily_prices, self.raw)
        self.assertEqual(result.raw_daily_prices.table.num_rows, 3)
        self.assertEqual(result.feature_daily_prices.table.num_rows, 1)
        self.assertEqual(result.feature_daily_prices.table['adj_close'].to_pylist(), [10.])
        self.assertEqual(result.loading_provenance['feature_financial_cutoff'], '2000-01-31')
        self.assertEqual(result.loading_provenance['raw_daily_financial_cutoff'], '2000-02-29')
        self.assertFalse(result.loading_provenance['outcome_tables_consumed_by_features'])
        self.assertEqual(result.loading_provenance['quote_session_audit']['excluded_non_session_rows'], 1)

    def test_annual_calendar_removes_prior_decades_and_keeps_boundary_metadata(self):
        dates = pd.bdate_range('2000-01-01','2024-12-31').strftime('%Y-%m-%d').tolist()
        calendar = pa.table({'trading_date':dates, 'calendar_name':['XNYS']*len(dates),
                             'is_trading_session':[True]*len(dates)})
        plan = replace(self.plan, years=(2023,), calendar=calendar,
            year_requests={2023:{'raw_start':'2021-01-04'}})
        annual = annual_feature_calendar(plan,2023)
        expected = [d for d in dates if '2021-01-04'<=d<='2024-01-31']
        self.assertEqual(annual['trading_date'].to_pylist(), expected)
        self.assertLess(annual.num_rows, 820)
        self.assertEqual(calendar.num_rows, len(dates))

    def test_prepared_metadata_reused_without_second_membership_read(self):
        with patch('dmv2.full_loading.BoundedTableReader', return_value=self.reader), \
                patch('dmv2.full_loading.estimate_bounded_read', return_value={'footer_only':True}):
            prepared = prepare_feature_year(self.plan,2010)
        self.reader.read.reset_mock()
        with patch('dmv2.full_loading.BoundedTableReader', return_value=self.reader), \
                patch('dmv2.full_loading.PriceLoader', return_value=self.price), \
                patch('dmv2.full_loading.estimate_bounded_read', return_value={'footer_only':True}):
            result = load_feature_year(self.plan,2010,prepared=prepared)
        self.reader.read.assert_not_called()
        self.assertIs(result.daily_membership, prepared.daily_membership)

    def test_prepared_different_pin_refused_before_prices(self):
        with patch('dmv2.full_loading.BoundedTableReader', return_value=self.reader), \
                patch('dmv2.full_loading.estimate_bounded_read', return_value={'footer_only':True}):
            prepared = prepare_feature_year(self.plan,2010)
        with patch('dmv2.full_loading.PriceLoader') as price:
            with self.assertRaises(DataContractError):
                load_feature_year(self.plan,2010,prepared=replace(prepared,source_plan_sha256='wrong'))
            price.assert_not_called()

    def test_id_requests_exhaustive_from_original_metadata_without_price_replacement(self):
        result = self.run_year()
        self.assertEqual(result.requested_ids, ('p-a', 'p-b', 'p-z'))
        self.assertEqual([s.table.num_rows for s in result.snapshots], [3,3])
        self.assertFalse(result.loading_provenance['future_quote_eligibility_used'])
        self.price.load.assert_called_once()
        self.assertEqual(self.price.load.call_args.args[:3],
                         (('p-a','p-b','p-z'), '1999-12-30', '2000-02-29'))

    def test_membership_read_never_uses_outcome_dates_or_quotes(self):
        result = self.run_year()
        request = self.reader.read.call_args.kwargs
        self.assertEqual(request['start'], '1999-12-31')
        self.assertEqual(request['end'], '2000-01-31')
        self.assertFalse(request['financial'])
        self.assertIs(result.daily_membership, self.source['membership'])
        self.assertNotIn('2000-02-29', request['filters']['trading_date'])

    def test_can_stop_raw_at_signal_explicitly(self):
        result = self.run_year(include_outcome_daily=False)
        self.assertEqual(self.price.load.call_args.args[2], '2000-01-31')
        self.assertEqual(result.loading_provenance['raw_daily_financial_cutoff'], '2000-01-31')

    def test_invalid_annual_requests_fail_before_sources(self):
        for year, overrides in [(2024, {}), (True, {}), (2010, {'include_outcome_daily':1}),
                                 (2010, {'price_budget':{}})]:
            with self.subTest(year=year, overrides=overrides), patch('dmv2.full_loading.PriceLoader') as price:
                with self.assertRaises(DataContractError):
                    load_feature_year(self.plan, year, **overrides)
                price.assert_not_called()

    def test_budget_rejection_before_price_read_returns_no_partial_year(self):
        with patch('dmv2.full_loading.BoundedTableReader', return_value=self.reader), \
                patch('dmv2.full_loading.estimate_bounded_read', side_effect=BudgetExceeded('too large')), \
                patch('dmv2.full_loading.PriceLoader') as price:
            with self.assertRaises(BudgetExceeded):
                load_feature_year(self.plan, 2010)
            price.assert_not_called()


if __name__ == '__main__':
    unittest.main()
