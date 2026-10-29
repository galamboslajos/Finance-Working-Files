import unittest
import numpy as np
from audit_warmup_coverage import consecutive_runs, required_prices


class WarmupCoverageTests(unittest.TestCase):
    def test_gap_resets_each_security_independently(self):
        valid = np.array([[True, False], [True, True], [False, True], [True, True]])
        np.testing.assert_array_equal(consecutive_runs(valid), [[1,0], [2,1], [0,2], [1,3]])

    def test_nested_normalization_then_graph_window(self):
        self.assertEqual(required_prices(1), 314)
        self.assertEqual(required_prices(252), 565)
        self.assertEqual(required_prices(1260), 1573)


if __name__ == "__main__":
    unittest.main()
