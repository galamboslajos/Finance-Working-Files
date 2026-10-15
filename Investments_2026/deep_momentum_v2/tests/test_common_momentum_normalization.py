"""Adversarial population/timing tests using synthetic returns only."""
import math
import unittest

import numpy as np
import pandas as pd

from dmv2.paper_features import (COMMON_MOMENTUM_POLICY, HORIZONS, PAPER_FEATURE_NAMES,
                                 normalize_common_momentum)
from dmv2.universe import DataContractError


def raw():
    rows = []
    for sid, base, admitted in [('a', .01, True), ('b', .03, True), ('c', .1, True), ('blocked', 100, False)]:
        rows.append({'source_security_id': sid, 'signal_date': '2009-12-31',
                     'identity_eligible': admitted,
                     **{f'MOM{h}_raw': base * h for h in HORIZONS},
                     **{f'MOM{h}_raw_status': 'ok' for h in HORIZONS}})
    return pd.DataFrame(rows)


def run(frame=None, exhaustive=True):
    frame = raw() if frame is None else frame
    return normalize_common_momentum(frame, exhaustive_requested_by_signal={s: exhaustive for s in frame.signal_date.unique()})


class CommonMomentumTests(unittest.TestCase):
    def test_shared_five_population_excludes_missing_horizon_from_every_statistic(self):
        frame = raw()
        frame.loc[2, 'MOM12_raw'] = np.nan
        before = frame.copy(deep=True)
        result = run(frame)
        table = result.table.set_index('source_security_id')
        self.assertEqual(table.paper_prediction_eligible.to_dict(), {'a': True, 'b': True, 'c': False, 'blocked': False})
        for h in HORIZONS:
            self.assertAlmostEqual(table.at['a', f'meanMOM{h}'], .02*h)
            self.assertAlmostEqual(table.at['a', f'stdMOM{h}'], math.sqrt(.0002)*h)
            self.assertAlmostEqual(table.loc[['a','b'], f'zMOM{h}'].mean(), 0)
            self.assertAlmostEqual(table.loc[['a','b'], f'zMOM{h}'].std(ddof=1), 1)
        self.assertTrue(table.loc[['c','blocked'], list(PAPER_FEATURE_NAMES)].isna().all().all())
        pd.testing.assert_frame_equal(frame, before)
        self.assertEqual(result.coverage.original_cohort_count.tolist(), [4]*5)
        self.assertEqual(result.coverage.common_momentum_population_count.tolist(), [2]*5)
        self.assertFalse(result.provenance['outcomes_consumed'])

    def test_partial_requested_sample_withholds_all_fifteen_without_deleting_raw(self):
        result = run(exhaustive=False)
        self.assertEqual(len(result.table), 4)
        self.assertTrue(result.table[list(PAPER_FEATURE_NAMES)].isna().all().all())
        self.assertTrue(result.table.MOM1_raw.notna().all())
        self.assertFalse(result.table.paper_prediction_eligible.any())
        self.assertEqual(set(result.table.paper_normalization_status), {'incomplete_request'})

    def test_one_degenerate_horizon_withholds_all_fifteen(self):
        frame = raw()
        frame['MOM6_raw'] = .5
        result = run(frame)
        self.assertTrue(result.table[list(PAPER_FEATURE_NAMES)].isna().all().all())
        self.assertEqual(set(result.table.paper_normalization_status), {'degenerate_dispersion'})

    def test_bool_infinite_and_non_ok_raw_are_not_finite_past_momentum(self):
        for value, status in [(True,'ok'),(np.inf,'ok'),(.02,'missing_exact_endpoint')]:
            frame = raw().astype({'MOM12_raw':object})
            frame.loc[1,'MOM12_raw'] = value
            frame.loc[1,'MOM12_raw_status'] = status
            result = run(frame)
            self.assertFalse(result.table.loc[1,'paper_prediction_eligible'])
            self.assertEqual(result.coverage.common_momentum_population_count.tolist(), [2]*5)

    def test_independent_signal_population_preserves_shared_state_levels(self):
        first = raw()
        second = raw()
        second.signal_date = '2010-01-29'
        for h in HORIZONS:
            second[f'MOM{h}_raw'] *= 2
        combined = run(pd.concat([first, second], ignore_index=True)).table
        a = combined[combined.source_security_id.eq('a')]
        self.assertAlmostEqual(a.iloc[1].meanMOM1, 2*a.iloc[0].meanMOM1)
        self.assertAlmostEqual(a.iloc[1].stdMOM1, 2*a.iloc[0].stdMOM1)
        pd.testing.assert_frame_equal(run(first).table, combined.iloc[:4].reset_index(drop=True))

    def test_minimum_common_population_is_two_not_coverage_threshold(self):
        frame = raw()
        frame.loc[[1,2], 'MOM1_raw'] = np.nan
        result = run(frame)
        self.assertEqual(set(result.table.paper_normalization_status), {'insufficient_common_population'})
        self.assertFalse(result.table.paper_prediction_eligible.any())

    def test_attestations_cannot_be_missing_extra_or_truthy_strings(self):
        for evidence in ({}, {'2009-12-31':True,'2010-01-29':True}, {'2009-12-31':'yes'}):
            with self.assertRaises(DataContractError):
                normalize_common_momentum(raw(), exhaustive_requested_by_signal=evidence)

    def test_reject_target_columns_protected_signals_duplicate_keys_and_unknown_identity(self):
        for change in ('target','protected','duplicate','identity'):
            frame = raw()
            if change == 'target': frame['outcome_end'] = '2010-01-29'
            elif change == 'protected': frame.signal_date = '2024-01-31'
            elif change == 'duplicate': frame.loc[1,'source_security_id'] = 'a'
            else: frame['identity_eligible'] = 'yes'
            with self.subTest(change=change), self.assertRaises(DataContractError):
                run(frame)


if __name__ == '__main__': unittest.main()
