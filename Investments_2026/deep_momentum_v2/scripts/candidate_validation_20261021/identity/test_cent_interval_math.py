import unittest
from decimal import Decimal
from cent_interval_math import factor_intersection


class IntervalMathTests(unittest.TestCase):
    def test_exact_common_scale(self):
        result = factor_intersection([100, 110, 90, 105], [50, 55, 45, 52.5])
        self.assertTrue(result['common_factor_possible_under_assumption'])
        self.assertLessEqual(Decimal(result['lower']), Decimal('.5'))
        self.assertGreaterEqual(Decimal(result['upper']), Decimal('.5'))

    def test_a_single_differently_scaled_field_is_not_explained(self):
        result = factor_intersection([100, 110, 90, 105], [50, 56, 45, 52.5])
        self.assertFalse(result['common_factor_possible_under_assumption'])

    def test_independently_rounded_observations_can_share_a_factor(self):
        raw = [10.01, 10.13, 9.98, 10.05]
        factor = Decimal('.73456')
        adj = [str((Decimal(str(r)) * factor).quantize(Decimal('.01'))) for r in raw]
        result = factor_intersection(raw, adj)
        self.assertTrue(result['common_factor_possible_under_assumption'])
        self.assertLessEqual(Decimal(result['lower']), factor)
        self.assertGreaterEqual(Decimal(result['upper']), factor)

    def test_zero_crossing_is_unavailable_not_division_by_nonpositive(self):
        result = factor_intersection([.005, 1, 1, 1], [1, 1, 1, 1])
        self.assertIsNone(result['common_factor_possible_under_assumption'])

    def test_nonfinite_is_unavailable(self):
        result = factor_intersection([100, 110, 90, 105], [50, 55, 45, float('inf')])
        self.assertIsNone(result['common_factor_possible_under_assumption'])

    def test_malformed_input_or_halfwidth_rejected(self):
        with self.assertRaises(ValueError):
            factor_intersection([1], [1])
        with self.assertRaises(ValueError):
            factor_intersection([1] * 4, [1] * 4, half_width='0')


if __name__ == '__main__':
    unittest.main()
