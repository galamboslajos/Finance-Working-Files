"""Fabricated independent XOM checker contracts; never original data."""
import unittest
from verify_xom_saved_output import SOURCE, calculate

DATES = ['2020-01-02', '2020-01-03']


def row(day='2020-01-02', mode='raw', **changes):
    scale = 1. if mode == 'raw' else .5
    r = dict(security_id=SOURCE, trading_date=day, query_symbol='XOM', adjustment=mode,
             open=10. * scale, high=12. * scale, low=9. * scale, close=11. * scale, volume=100.)
    r.update(changes)
    return r


class OutputCheckerContracts(unittest.TestCase):
    def test_complete_pairs_and_volume_do_not_require_inverse_factor(self):
        got = calculate([row(d, m) for d in DATES for m in ['raw', 'all']], DATES)
        self.assertEqual(got['complete_pairs'], 2)
        self.assertEqual(got['maximum_common_scale_relative_residual'], 0.)
        self.assertTrue(all(not v for v in got['per_date_flags'].values()))
        self.assertEqual(got['differing_volume_dates'], [])

    def test_missing_pair_retained(self):
        got = calculate([row(), row(mode='all')], DATES)
        self.assertEqual(got['missing_dates']['raw'], ['2020-01-03'])
        self.assertEqual(got['missing_dates']['all'], ['2020-01-03'])
        self.assertEqual(got['complete_pairs'], 1)

    def test_scope_is_checked_before_numeric_cells(self):
        class Poison:
            def __float__(self):
                raise AssertionError('Unauthorized numeric conversion')
        for changes in [{'security_id': 'other'}, {'trading_date': '2024-01-02'},
                        {'trading_date': '2020-01-04'}, {'query_symbol': 'OTHER'}, {'adjustment': 'split'}]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                calculate([row(close=Poison(), **changes)], DATES)

    def test_duplicate_key_fails(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            calculate([row(), row()], DATES)

    def test_ohlc_and_scale_failures_are_separate(self):
        got = calculate([row(high=10.), row(mode='all', adj_extra='unused')], DATES)
        self.assertEqual(got['OHLC_bracket_dates']['raw'], ['2020-01-02'])
        self.assertEqual(got['price_scale_failure_dates'], ['2020-01-02'])

    def test_zero_and_negative_volumes_remain_separate_diagnostics(self):
        got = calculate([row(volume=0.), row(mode='all', volume=-1.)], DATES)
        self.assertEqual(got['zero_volume']['raw'], 1)
        self.assertEqual(got['negative_volume']['all'], 1)
        self.assertIn('all_volume_negative_or_nonfinite', got['per_date_flags']['2020-01-02'])

    def test_discontinuities_compare_adjacent_expected_sessions(self):
        rows = [row(mode=m) for m in ['raw', 'all']]
        rows += [row(day='2020-01-03', mode='raw', close=22., high=24.),
                 row(day='2020-01-03', mode='all', close=11., high=12.)]
        got = calculate(rows, DATES)
        self.assertEqual(len(got['consecutive_session_discontinuities']), 1)
        self.assertEqual(got['consecutive_session_discontinuities'][0]['raw_close_change'], 1.)

    def test_discontinuities_do_not_bridge_a_missing_true_session(self):
        rows = [row(mode=m) for m in ['raw', 'all']]
        rows += [row(day='2020-01-06', mode='raw', close=22., high=24.),
                 row(day='2020-01-06', mode='all', close=11., high=12.)]
        got = calculate(rows, DATES + ['2020-01-06'])
        self.assertEqual(got['consecutive_session_discontinuities'], [])


if __name__ == '__main__':
    unittest.main()
