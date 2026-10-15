"""Feature history reads stop at signals and retain first-cohort sample selection."""
import unittest
from unittest.mock import Mock, patch

import pandas as pd
import pyarrow as pa

from dmv2.exchange_sessions import ReconciledExchangeCalendar
from dmv2.feature_loading import load_feature_sample
from dmv2.loading import BudgetExceeded, ReadBudget
from dmv2.universe import DataContractError, KnowledgeTimeUnavailable
from test_monthly_loading import fixtures


class FeatureLoadingTests(unittest.TestCase):
    def setUp(self):
        self.source = fixtures()
        dates = pd.bdate_range('1997-01-01', '2000-03-31').strftime('%Y-%m-%d').tolist()
        self.calendar = pa.table({'trading_date': dates, 'calendar_name': ['XNYS']*len(dates),
                                  'is_trading_session': [True]*len(dates)})
        self.catalog = Mock()
        self.catalog.provenance.return_value = {'pin': 'structural-fixture'}
        self.reader = Mock()
        self.reader.read_metadata.side_effect = [self.source['calendar'], self.source['bridge'], self.source['links']]
        self.reader.read.return_value = self.source['membership']
        self.price = Mock()
        self.price.load.return_value = self.source['prices']

    def run_sample(self, **overrides):
        params = dict(start_month='2000-01', end_month='2000-02', universe='combined')
        params.update(overrides)
        with patch('dmv2.feature_loading.BoundedTableReader', return_value=self.reader), \
             patch('dmv2.feature_loading.PriceLoader', return_value=self.price), \
             patch('dmv2.feature_loading.reconcile_exchange_calendar', return_value=ReconciledExchangeCalendar(self.calendar, self.source['calendar'].table, {'fixture': True})):
            return load_feature_sample(self.catalog, **params)

    def test_history_stops_latest_signal_not_next_month_target(self):
        sample = self.run_sample()
        self.assertEqual(sample.signal_dates, ('1999-12-31', '2000-01-31'))
        args = self.price.load.call_args.args
        self.assertEqual(args[2], '2000-01-31')
        first = self.calendar['trading_date'].to_pylist().index('1999-12-31')
        self.assertEqual(args[1], self.calendar['trading_date'].to_pylist()[first-502])
        self.assertFalse(sample.loading_provenance['outcome_tables_consumed'])

    def test_ids_chosen_before_later_entries_or_quotes(self):
        sample = self.run_sample()
        self.assertEqual(sample.requested_ids, ('p-b', 'p-z'))
        self.assertNotIn('p-a', sample.requested_ids)
        self.assertEqual([s.table.num_rows for s in sample.snapshots], [3, 3])
        self.assertFalse(sample.loading_provenance['daily_context_loaded'])

    def test_invalid_warmup_fails_before_sources(self):
        for value in (252, True, 503.0, None):
            with self.subTest(value=value), patch('dmv2.feature_loading.BoundedTableReader') as reader:
                with self.assertRaises(DataContractError):
                    load_feature_sample(Mock(), start_month='2000-01', end_month='2000-02', universe='combined', history_sessions=value)
                reader.assert_not_called()

    def test_knowledge_time_claim_refused_before_sources(self):
        with patch('dmv2.feature_loading.BoundedTableReader') as reader:
            with self.assertRaises(KnowledgeTimeUnavailable):
                load_feature_sample(Mock(), start_month='2000-01', end_month='2000-02', universe='combined', mode='knowledge_time')
            reader.assert_not_called()

    def test_insufficient_calendar_fails_before_membership_prices(self):
        self.calendar = self.source['calendar'].table
        with self.assertRaisesRegex(DataContractError, 'warmup'):
            self.run_sample()
        self.reader.read.assert_not_called()
        self.price.load.assert_not_called()

    def test_cumulative_budget_no_partial_success(self):
        with self.assertRaises(BudgetExceeded):
            self.run_sample(budget=ReadBudget(max_scanned_rows=1))

    def test_non_session_quotes_are_filtered_before_feature_arithmetic(self):
        from test_monthly_loading import loaded
        rows=self.source['prices'].table.to_pylist()
        rows.append({**rows[0],'trading_date':'2000-01-15','adj_close':999.})
        raw=loaded(pa.Table.from_pylist(rows),'prices')
        self.price.load.return_value=raw
        result=self.run_sample()
        self.assertEqual(raw.table.num_rows,2)
        self.assertEqual(result.daily_prices.table.num_rows,1)
        self.assertEqual(result.loading_provenance['quote_session_audit']['excluded_non_session_rows'],1)


if __name__ == '__main__':
    unittest.main()
