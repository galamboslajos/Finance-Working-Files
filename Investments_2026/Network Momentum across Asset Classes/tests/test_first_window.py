import unittest
import pandas as pd
from audit_first_window import gap_runs


class FirstWindowTests(unittest.TestCase):
    def test_single_interior_missing_day(self):
        days = pd.bdate_range('2015-01-05', periods=3)
        gaps = gap_runs(days, ['ok','missing_price','ok'])
        self.assertEqual(len(gaps),1)
        self.assertEqual(gaps[0]['sessions'],1)
        self.assertEqual(gaps[0]['position'],'interior')

    def test_membership_exit_splits_gaps(self):
        days = pd.bdate_range('2015-01-05', periods=5)
        gaps = gap_runs(days, ['missing_price']*5, [True, True, False, True, True])
        self.assertEqual([g['sessions'] for g in gaps],[2,2])

    def test_reason_change_splits_but_keeps_all_days(self):
        days = pd.bdate_range('2015-01-05', periods=5)
        gaps = gap_runs(days,['missing_price','ok','invalid_ohlcv','identity_unresolved','ok'])
        self.assertEqual([g['position'] for g in gaps],['leading','interior','interior'])
        self.assertEqual(sum(g['sessions'] for g in gaps),3)

    def test_full_absence_and_trailing_run(self):
        days = pd.bdate_range('2015-01-05', periods=6)
        self.assertEqual(gap_runs(days,['missing_price']*6)[0]['position'],'no_usable_quote')
        gap = gap_runs(days,['ok']+['missing_price']*5)[0]
        self.assertEqual(gap['position'],'trailing')
        self.assertEqual(gap['length_bucket'],'2-5')


if __name__ == '__main__':
    unittest.main()
