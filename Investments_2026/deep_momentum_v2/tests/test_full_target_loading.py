"""Tiny synthetic cache contracts; no research observations in these fixtures."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from dmv2.full_target_loading import build_cached_target_year, plan_cached_target_year
from dmv2.identity_policy import PROVISIONAL_IDENTITY_POLICY, qualify_identity_records
from dmv2.monthly import REQUIRED_PRICE_COLUMNS, build_month_windows
from dmv2.universe import DataContractError, UNIVERSE_SCHEMA

PRICE_SCHEMA = pa.schema([(n, pa.float64() if n == 'adj_close' else pa.bool_()
                          if n == 'intraperiod' else pa.string()) for n in REQUIRED_PRICE_COLUMNS])


def artifact(path):
    return {'path': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'size_bytes': path.stat().st_size}


class StructuralCache:
    def __init__(self, root):
        self.root = root
        self.year = root / 'years/2010'
        self.year.mkdir(parents=True)
        self.calendar = pa.Table.from_pylist([{'trading_date': d, 'calendar_name': 'STRUCTURAL_ONLY',
            'is_trading_session': True} for d in ['2009-12-31', *[f'2010-{m:02d}-{day}'
            for m in range(1, 13) for day in ('01', '28')], '2011-01-03']])
        self.windows = build_month_windows(self.calendar, start_month='2010-01', end_month='2010-12')
        rows = []
        for index, window in enumerate(self.windows):
            for number in range(20):
                sid = 'removed' if number == 19 and index == 0 else 'entrant' if number == 19 else f'member{number:02d}'
                rows.append({'source_security_id': sid, 'as_of': window.signal_date,
                    'indices': ['sp500' if number < 10 else 'sp400'],
                    'membership_tickers': ['STRUCTURAL_ONLY'], 'membership_records_json': '[]',
                    'unified_security_id': 'u-' + sid, 'intrinio_security_id': 'p-' + sid,
                    'identity_status': 'resolved', 'bridge_method': 'structural', 'bridge_confidence': 'structural',
                    'identity_review_required': False, 'resolution_status': 'resolved'})
        rows.sort(key=lambda r: (r['as_of'], r['source_security_id']))
        self.original = pa.Table.from_pylist(rows, schema=UNIVERSE_SCHEMA)
        self.original = self.original.append_column('signal_date', self.original['as_of'])
        qualified = qualify_identity_records(self.original.to_pylist(), identity_policy=PROVISIONAL_IDENTITY_POLICY)
        self.coverage = pa.Table.from_pylist([{**r, 'requested': True, 'paper_prediction_eligible': True} for r in qualified])
        self.requests = sorted({r['intrinio_security_id'] for r in rows})
        dates = ['2008-01-04', '2009-12-31', *[f'2010-{m:02d}-{day}' for m in range(1, 13) for day in ('01', '28')]]
        self.prices = pa.Table.from_pylist([{'intrinio_security_id': sid, 'trading_date': day,
            'adj_close': 10 + i * .1 + day.count('28') * (i + 1),
            'frequency': 'daily', 'intraperiod': False, 'price_quality_status': 'accepted'}
            for i, sid in enumerate(self.requests) for day in dates] + [{
                'intrinio_security_id': self.requests[0], 'trading_date': '2010-01-02',
                'adj_close': 999999.0, 'frequency': 'daily', 'intraperiod': False,
                'price_quality_status': 'accepted'}], schema=PRICE_SCHEMA)
        self.audit = {'year': 2010, 'build_identity_sha256': 'sha256:' + 'a' * 64,
            'feature_provenance': {'identity_policy': PROVISIONAL_IDENTITY_POLICY,
                'mode': 'historical_reconstruction', 'requested_ids': self.requests,
                'signal_dates': [w.signal_date for w in self.windows],
                'target_windows': {w.signal_date: {'outcome_month': w.outcome_month,
                    'outcome_start': w.outcome_start, 'outcome_end': w.outcome_end} for w in self.windows}},
            'loading_provenance': {'reference_loading_provenance': {'stages': {'raw_daily_prices': {'request': {
                'start': '2008-01-04', 'end': '2010-12-28', 'filters': {'intrinio_security_id': self.requests}}}}}}}
        self.write()

    def write(self):
        self.audit['artifacts'] = {}
        for name, table in [('original_cohort', self.original), ('feature_coverage', self.coverage),
                             ('exchange_calendar', self.calendar), ('daily_history_raw', self.prices)]:
            path = self.year / (name + '.parquet')
            pq.write_table(table, path, compression='zstd')
            self.audit['artifacts'][name] = artifact(path)
        self.rebind()

    def rebind(self):
        path = self.year / 'audit.json'
        path.write_text(json.dumps(self.audit))
        item = artifact(path)
        item['path'] = 'years/2010/audit.json'
        self.reference = {'directory': self.root,
            'audit': {'status': 'COMPLETE', 'build_identity_sha256': self.audit['build_identity_sha256'],
                      'years': {'2010': item}}}


class CachedTargetLoadingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cache = StructuralCache(Path(self.temp.name))

    def test_plan_reads_only_metadata_and_footers(self):
        original_read = pq.read_table
        calls = []
        def recording(path, *args, **kwargs):
            calls.append((Path(path).name, kwargs.get('columns')))
            self.assertNotEqual(Path(path).name, 'daily_history_raw.parquet')
            return original_read(path, *args, **kwargs)
        with patch('dmv2.full_target_loading.pq.read_table', side_effect=recording):
            plan = plan_cached_target_year(self.cache.reference, 2010)
        self.assertFalse(plan.read_plan['financial_values_read'])
        self.assertEqual(plan.read_plan['original_rows'], 240)
        self.assertEqual(plan.read_plan['paper_qualified_rows'], 240)
        self.assertEqual(plan.read_plan['daily_cache_rows'], self.cache.prices.num_rows)
        self.assertEqual(plan.read_plan['price_columns'], list(REQUIRED_PRICE_COLUMNS))
        self.assertEqual(len(calls), 3)

    def test_build_preserves_signal_membership_and_prior_december_anchor(self):
        result = build_cached_target_year(self.cache.reference, 2010)
        self.assertEqual(result.targets.table.num_rows, 240)
        self.assertEqual(result.loading_stats['discarded_offsession_rows'], 1)
        self.assertEqual(result.loading_stats['discarded_warmup_rows'], 21)
        jan = [r for r in result.monthly.table.to_pylist() if r['outcome_month'] == '2010-01']
        feb = [r for r in result.monthly.table.to_pylist() if r['outcome_month'] == '2010-02']
        self.assertIn('removed', {r['source_security_id'] for r in jan})
        self.assertNotIn('entrant', {r['source_security_id'] for r in jan})
        self.assertIn('entrant', {r['source_security_id'] for r in feb})
        self.assertNotIn('removed', {r['source_security_id'] for r in feb})
        self.assertTrue(all(r['start_adj_close'] < 100 for r in jan))
        self.assertFalse(result.provenance['raw_daily_artifacts_modified'])
        self.assertTrue(all(result.targets.diagnostics['exhaustive_requests'].to_pylist()))

    def test_missing_terminal_quote_remains_in_original_cohort_unlabelled(self):
        records = self.cache.prices.to_pylist()
        self.cache.prices = pa.Table.from_pylist([r for r in records if not (
            r['intrinio_security_id'] == 'p-removed' and r['trading_date'] == '2010-01-28')], schema=PRICE_SCHEMA)
        self.cache.write()
        result = build_cached_target_year(self.cache.reference, 2010)
        monthly = next(r for r in result.monthly.table.to_pylist() if r['source_security_id'] == 'removed')
        target = next(r for r in result.targets.table.to_pylist() if r['source_security_id'] == 'removed')
        self.assertEqual(monthly['return_status'], 'missing_end')
        self.assertTrue(target['prediction_eligible'])
        self.assertFalse(target['target_available'])
        self.assertEqual(result.targets.table.num_rows, 240)

    def test_request_cannot_be_inferred_from_available_price_inventory(self):
        missing = self.cache.requests.pop()
        self.cache.audit['feature_provenance']['requested_ids'] = list(self.cache.requests)
        self.cache.audit['loading_provenance']['reference_loading_provenance']['stages']['raw_daily_prices']['request']['filters']['intrinio_security_id'] = list(self.cache.requests)
        self.cache.rebind()
        with self.assertRaisesRegex(DataContractError, 'request'):
            plan_cached_target_year(self.cache.reference, 2010)

    def test_actual_executed_request_must_match_declared_request(self):
        request = self.cache.audit['loading_provenance']['reference_loading_provenance']['stages']['raw_daily_prices']['request']
        request['filters']['intrinio_security_id'] = self.cache.requests[:-1]
        self.cache.rebind()
        with self.assertRaisesRegex(DataContractError, 'executed ID/date'):
            plan_cached_target_year(self.cache.reference, 2010)

    def test_bad_identity_ledger_fails_before_prices(self):
        rows = self.cache.coverage.to_pylist()
        rows[0]['identity_eligible'] = False
        self.cache.coverage = pa.Table.from_pylist(rows)
        self.cache.write()
        with self.assertRaisesRegex(DataContractError, 'identity metadata'):
            plan_cached_target_year(self.cache.reference, 2010)

    def test_protected_metadata_is_rejected_before_financial_projection(self):
        rows = self.cache.prices.to_pylist()
        rows.append({**rows[0], 'trading_date': '2024-01-02'})
        self.cache.prices = pa.Table.from_pylist(rows, schema=PRICE_SCHEMA)
        self.cache.write()
        original_read = pq.read_table
        def guarded(path, *args, **kwargs):
            if Path(path).name == 'daily_history_raw.parquet':
                self.assertNotIn('adj_close', kwargs.get('columns', []))
            return original_read(path, *args, **kwargs)
        with patch('dmv2.full_target_loading.pq.read_table', side_effect=guarded):
            with self.assertRaisesRegex(DataContractError, 'Protected 2024'):
                build_cached_target_year(self.cache.reference, 2010)

    def test_changed_cache_bytes_fail_hash_verification(self):
        path = self.cache.year / 'daily_history_raw.parquet'
        path.write_bytes(path.read_bytes() + b'changed')
        with self.assertRaisesRegex(DataContractError, 'artifact changed'):
            build_cached_target_year(self.cache.reference, 2010)

    def test_changed_plan_and_wrong_year_are_rejected(self):
        plan = plan_cached_target_year(self.cache.reference, 2010)
        plan.provenance['year'] = 2011
        with self.assertRaisesRegex(DataContractError, 'plan changed'):
            build_cached_target_year(self.cache.reference, 2010, plan=plan)
        for year in [2009, 2024, True, 2010.0]:
            with self.subTest(year=year), self.assertRaises(DataContractError):
                plan_cached_target_year(self.cache.reference, year)

    def test_non_session_quote_does_not_fill_absent_endpoint(self):
        rows = self.cache.prices.to_pylist()
        sid = 'p-member00'
        self.cache.prices = pa.Table.from_pylist([r for r in rows if not (
            r['intrinio_security_id'] == sid and r['trading_date'] == '2010-01-28')], schema=PRICE_SCHEMA)
        self.cache.write()
        result = build_cached_target_year(self.cache.reference, 2010)
        row = next(r for r in result.monthly.table.to_pylist() if r['intrinio_security_id'] == sid)
        self.assertEqual(row['return_status'], 'missing_end')
        self.assertIsNone(row['provider_adjusted_return'])


if __name__ == '__main__':
    unittest.main()
