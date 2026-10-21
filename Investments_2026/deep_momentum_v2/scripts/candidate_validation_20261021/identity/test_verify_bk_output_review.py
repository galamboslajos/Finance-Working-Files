"""Fabricated verifier contracts; no research table or numeric source reads."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verify_bk_output import verify


class KeyOnly:
    def __init__(self, rows):
        self.rows = rows

    def to_pylist(self):
        return self.rows


class GuardTable:
    def __init__(self, keys, financial_rows=None):
        self.keys = keys
        self.num_rows = len(keys)
        self.financial_rows = financial_rows
        self.financial_conversion_attempted = False

    def select(self, fields):
        assert fields == ['intrinio_security_id', 'trading_date']
        return KeyOnly(self.keys)

    def to_pylist(self):
        self.financial_conversion_attempted = True
        if self.financial_rows is None:
            raise RuntimeError('Financial conversion before metadata guard')
        return self.financial_rows


def row(day, **changes):
    out = dict(intrinio_security_id='sec_zq1J05', trading_date=day,
        open=10., high=12., low=9., close=11., volume=100.,
        adj_open=5., adj_high=6., adj_low=4.5, adj_close=5.5, adj_volume=200.,
        factor=.5, split_ratio=None, dividend=None)
    out.update(changes)
    return out


def table(rows):
    return GuardTable([{k: r[k] for k in ('intrinio_security_id', 'trading_date')} for r in rows], rows)


class VerifierContracts(unittest.TestCase):
    def test_scope_and_calendar_failure_precede_financial_conversion(self):
        for native, day in [('other', '2010-01-04'), ('sec_zq1J05', '2007-06-29'),
                            ('sec_zq1J05', '2024-01-02'), ('sec_zq1J05', '2010-01-03')]:
            bad = GuardTable([{'intrinio_security_id': native, 'trading_date': day}])
            with self.subTest(native=native, day=day), self.assertRaises(AssertionError):
                verify(bad, {'2010-01-04'})
            self.assertFalse(bad.financial_conversion_attempted)

    def test_duplicate_precedes_financial_conversion(self):
        key = {'intrinio_security_id': 'sec_zq1J05', 'trading_date': '2010-01-04'}
        bad = GuardTable([key, key])
        with self.assertRaises(AssertionError):
            verify(bad, {'2010-01-04'})
        self.assertFalse(bad.financial_conversion_attempted)

    def test_valid_zero_volume_and_null_actions_remain_separate_diagnostics(self):
        stats = verify(table([row('2010-01-04', volume=0.)]), {'2010-01-04'})
        self.assertEqual(stats['volume_zero']['volume'], 1)
        self.assertEqual(stats['volume_negative']['volume'], 0)
        self.assertEqual(stats['split_ratio_nontrivial_dates'], [])
        self.assertEqual(stats['positive_dividend_dates'], [])
        self.assertEqual(stats['raw_OHLC_failures'], [])

    def test_missing_nonfinite_price_and_ohlc_are_separate(self):
        stats = verify(table([row('2010-01-04', open=None, adj_open=float('nan'), low=12.)]), {'2010-01-04'})
        self.assertEqual(stats['price_null_or_nonfinite']['open'], 1)
        self.assertEqual(stats['price_null_or_nonfinite']['adj_open'], 1)
        # Missing values prevent an OHLC bracket claim; the separate null flag retains the failure.
        self.assertEqual(stats['raw_OHLC_failures'], [])
        stats = verify(table([row('2010-01-04', low=12.)]), {'2010-01-04'})
        self.assertEqual(stats['raw_OHLC_failures'], ['2010-01-04'])

    def test_common_factor_diagnostic_does_not_assume_inverse_volume_semantics(self):
        stats = verify(table([row('2010-01-04', factor=2., adj_volume=100.)]), {'2010-01-04'})
        self.assertEqual(stats['within_row_adjustment_ratio_failures'], [])
        self.assertEqual(stats['source_factor_vs_adj_close_ratio_mismatch_dates'], ['2010-01-04'])
        self.assertEqual(stats['volume_null_or_nonfinite']['adj_volume'], 0)

    def test_missing_session_prevents_bridged_jump_diagnostic(self):
        rows = [row('2010-01-04'), row('2010-01-06', adj_close=11.)]
        stats = verify(table(rows), {'2010-01-04', '2010-01-05', '2010-01-06'})
        self.assertEqual(stats['missing_sessions'], ['2010-01-05'])
        self.assertEqual(stats['daily_jump_over_30pct_dates'], [])
        stats = verify(table(rows), {'2010-01-04', '2010-01-06'})
        self.assertEqual(stats['daily_jump_over_30pct_dates'], ['2010-01-06'])


if __name__ == '__main__':
    unittest.main()
