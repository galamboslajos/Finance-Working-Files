import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from repair_walk_forward_valuations import event_value, overlay, quote_return, same_scale, claim_sensitivity
from evaluate_walk_forward import ARMS


class ValuationRepairTests(unittest.TestCase):
    def test_no_dividend_or_reverse_split_double_count(self):
        # A historical future reverse split can make adjusted/raw scale > 1;
        # the one-day return is still on two contemporaneous raw units.
        self.assertAlmostEqual(quote_return(9.15, 9.015), 9.015 / 9.15 - 1)
        self.assertAlmostEqual(quote_return(100, 49, .5), -.02)

    def test_rounded_scale_intersection(self):
        a = pd.Series(dict(close=100., adj_close=75.0000))
        b = pd.Series(dict(close=80., adj_close=60.0000))
        self.assertTrue(same_scale(a, b))
        b.adj_close = 59.
        self.assertFalse(same_scale(a, b))

    def test_merger_uses_raw_exchange_units_and_final_anss_terms(self):
        config = json.loads((Path(__file__).resolve().parents[1] / 'configs/valuation_events_v1.json').read_text())
        e = next(x for x in config['events'] if x['ticker'] == 'ANSS')
        self.assertEqual(e['cash'], 199.91)
        self.assertEqual(e['ratio'], .3399)
        self.assertAlmostEqual(event_value(e, 589.01), 199.91 + .3399 * 589.01)
        rtn = next(x for x in config['events'] if x['ticker'] == 'RTN')
        self.assertAlmostEqual(event_value(rtn, 49.93), 2.3348 * 49.93)

    def test_contingent_right_is_not_silently_marked_zero_or_face(self):
        self.assertTrue(np.isnan(event_value({'cash': 380., 'ratio': 0., 'contingent_right_max': 35.})))
        self.assertEqual(event_value({'cash': 95., 'ratio': 0.}), 95.)

    def test_overlay_preserves_observed_outcomes_and_forecasts(self):
        f = pd.DataFrame({'date': pd.to_datetime(['2024-01-02', '2024-01-02']),
            'source_security_id': ['a', 'b'], 'forward_return': [.1, np.nan],
            'target': [1., np.nan], 'daily_volatility': [.1, .2], 'GMOM_prediction': [3., 4.]})
        ledger = pd.DataFrame({'date': pd.to_datetime(['2024-01-02']), 'source_security_id': ['b'],
            'repaired_forward_return': [-.04]})
        out = overlay(f, ledger)
        np.testing.assert_array_equal(out.GMOM_prediction, f.GMOM_prediction)
        self.assertEqual(out.forward_return.iloc[0], .1)
        self.assertAlmostEqual(out.target.iloc[1], -.2)
        ledger.source_security_id = 'a'
        with self.assertRaises(ValueError):
            overlay(f, ledger)

    def test_claim_bounds_reverse_for_short_without_dropping_held_stock(self):
        day = pd.Timestamp('2025-08-27')
        f = pd.DataFrame({'date': [day, day], 'source_security_id': ['a', 'b'],
            'forward_return': [np.nan, .01], 'daily_volatility': [.02, .03]})
        for arm in ARMS:
            f[arm + '_position'] = [-1., 1.]
        ledger = pd.DataFrame({'date': [day], 'label_date': [pd.Timestamp('2025-08-28')],
            'source_security_id': ['a'], 'status': ['unresolved_contingent_right'], 'membership_ticker': ['WBA'],
            'stock_return_if_claim_mark_zero': [-.05], 'stock_return_if_claim_mark_face': [.2], 'contingent_right_max': [3.]})
        sensitivity = claim_sensitivity(f, ledger)
        self.assertTrue(sensitivity.nodes.eq(2).all())
        self.assertTrue((sensitivity.portfolio_return_claim_face_mark < sensitivity.portfolio_return_claim_zero_mark).all())
        self.assertTrue((sensitivity.one_day_return_upper > sensitivity.one_day_return_lower).all())


if __name__ == '__main__':
    unittest.main()
