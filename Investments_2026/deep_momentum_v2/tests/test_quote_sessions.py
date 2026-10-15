"""Metadata-only quote/calendar partition guards; no market observations read."""
import unittest

import pyarrow as pa

from dmv2.catalog import canonical_sha256
from dmv2.quote_sessions import exact_session_quotes
from dmv2.universe import DataContractError


class _FinancialScalar(pa.ExtensionScalar):
    conversions = 0

    def as_py(self, *args, **kwargs):
        type(self).conversions += 1
        raise AssertionError('Financial scalar reached Python during metadata partition')


class _FinancialType(pa.ExtensionType):
    def __init__(self):
        super().__init__(pa.string(), 'dmv2.tests.no_python_financial_conversion')

    def __arrow_ext_serialize__(self):
        return b''

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized):
        return cls()

    def __arrow_ext_scalar_class__(self):
        return _FinancialScalar


SESSIONS = ('2009-01-16', '2009-01-20')


def quotes(days, ids=None):
    ids = ['security-a'] * len(days) if ids is None else ids
    opaque_values = pa.ExtensionArray.from_storage(
        _FinancialType(), pa.array(['opaque-value'] * len(days), type=pa.string()))
    return pa.table({'intrinio_security_id': ids, 'trading_date': days,
                     'adj_close': opaque_values})


class QuoteSessionTests(unittest.TestCase):
    def setUp(self):
        _FinancialScalar.conversions = 0

    def run_partition(self, table, sessions=SESSIONS, **overrides):
        args = dict(start=SESSIONS[0], end=SESSIONS[-1])
        args.update(overrides)
        return exact_session_quotes(table, sessions, **args)

    def test_no_exclusions_preserve_same_source_object_without_value_conversion(self):
        source = quotes(list(SESSIONS))
        filtered, audit = self.run_partition(source)
        self.assertIs(filtered, source)
        self.assertEqual(audit['source_rows_read'], 2)
        self.assertEqual(audit['analysis_rows'], 2)
        self.assertEqual(audit['excluded_non_session_rows'], 0)
        self.assertEqual(audit['excluded_quote_keys'], [])
        self.assertEqual(audit['calendar_sessions_sha256'], canonical_sha256(SESSIONS))
        self.assertTrue(audit['original_source_rows_unchanged'])
        self.assertFalse(audit['financial_values_converted_to_python'])
        self.assertEqual(_FinancialScalar.conversions, 0)

    def test_weekend_and_holiday_rows_are_retained_as_exclusion_metadata(self):
        days = ['2009-01-16', '2009-01-17', '2009-01-18', '2009-01-19', '2009-01-20']
        source = quotes(days)
        original_buffer = source['adj_close'].chunk(0).storage.buffers()[2].address
        filtered, audit = self.run_partition(source)
        self.assertEqual(filtered['trading_date'].to_pylist(), list(SESSIONS))
        self.assertEqual((audit['source_rows_read'], audit['analysis_rows'],
                          audit['excluded_non_session_rows']), (5, 2, 3))
        self.assertEqual(audit['excluded_quote_keys'], [
            {'intrinio_security_id': 'security-a', 'trading_date': day,
             'reason': 'not_independent_exchange_session'} for day in days[1:4]])
        self.assertEqual(source['trading_date'].to_pylist(), days)
        self.assertEqual(source.num_rows, 5)
        self.assertEqual(source['adj_close'].chunk(0).storage.buffers()[2].address, original_buffer)
        self.assertEqual(filtered.schema, source.schema)
        self.assertEqual(_FinancialScalar.conversions, 0)

    def test_empty_quote_table_is_valid_and_preserves_source(self):
        source = quotes([], ids=pa.array([], type=pa.string()))
        # Empty date arrays still carry the ordinary text key schema.
        source = source.set_column(1, 'trading_date', pa.array([], type=pa.string()))
        filtered, audit = self.run_partition(source)
        self.assertIs(filtered, source)
        self.assertEqual(audit['analysis_rows'], 0)
        self.assertEqual(audit['excluded_quote_keys'], [])

    def test_outside_or_protected_quotes_cannot_be_hidden_as_non_sessions(self):
        for day in ('2009-01-15', '2009-01-21', '2024-01-02'):
            with self.subTest(day=day), self.assertRaisesRegex(DataContractError, 'outside declared bounds'):
                self.run_partition(quotes([day]))
        self.assertEqual(_FinancialScalar.conversions, 0)

    def test_duplicate_source_keys_fail_even_when_non_session_rows(self):
        for day in ('2009-01-16', '2009-01-19'):
            with self.subTest(day=day), self.assertRaisesRegex(DataContractError, 'duplicate'):
                self.run_partition(quotes([day, day]))
        self.assertEqual(_FinancialScalar.conversions, 0)

    def test_bad_ids_and_dates_fail_without_financial_conversion(self):
        for native in (None, '', '   ', 123):
            with self.subTest(native=native), self.assertRaises(DataContractError):
                self.run_partition(quotes(['2009-01-16'], ids=[native]))
        for day in (None, '2009-1-16', '20090116', '2009-02-30'):
            with self.subTest(day=day), self.assertRaises(DataContractError):
                self.run_partition(quotes([day]))
        self.assertEqual(_FinancialScalar.conversions, 0)

    def test_calendar_requires_ordered_unique_exact_boundaries(self):
        for sessions in ('2009-01-16', b'2009-01-16', (),
                         ('2009-01-20', '2009-01-16'),
                         ('2009-01-16', '2009-01-16', '2009-01-20'),
                         ('2009-01-16',), ('2009-01-20',)):
            with self.subTest(sessions=sessions), self.assertRaises(DataContractError):
                self.run_partition(quotes(list(SESSIONS)), sessions=sessions)

    def test_invalid_or_protected_bounds_fail_before_financial_values(self):
        cases = [dict(start='2009-01-20', end='2009-01-16'),
                 dict(start='2009-01-16', end='2024-01-02'),
                 dict(start='2009-01-19'), dict(end='2009-01-19'),
                 dict(start='2009-1-16')]
        for bounds in cases:
            with self.subTest(bounds=bounds), self.assertRaises(DataContractError):
                self.run_partition(quotes(list(SESSIONS)), **bounds)
        self.assertEqual(_FinancialScalar.conversions, 0)

    def test_financial_holdout_guard_is_independent_of_extended_calendar_metadata(self):
        calendar = (*SESSIONS, '2024-01-02')
        source = quotes(list(SESSIONS))
        filtered, _ = self.run_partition(source, sessions=calendar)
        self.assertIs(filtered, source)
        with self.assertRaises(DataContractError):
            self.run_partition(quotes(['2024-01-02']), sessions=calendar)
        self.assertEqual(_FinancialScalar.conversions, 0)

    def test_only_arrow_tables_with_both_metadata_keys_are_accepted(self):
        for source in (None, {}, [], pa.table({'trading_date': list(SESSIONS)}),
                       pa.table({'intrinio_security_id': ['security-a']})):
            with self.subTest(source=source), self.assertRaises(DataContractError):
                self.run_partition(source)


if __name__ == '__main__':
    unittest.main()
