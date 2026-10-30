import unittest

import numpy as np
import pandas as pd

from analyze_walk_forward import extended_metrics, monthly_returns, rolling_statistics, segment_curves
from evaluate_walk_forward import ARMS


class PerformanceDiagnosticsTests(unittest.TestCase):
    def frame(self, values, dates=None):
        dates = pd.bdate_range("2024-01-02", periods=len(values)) if dates is None else pd.to_datetime(dates)
        frame = pd.DataFrame({"date": dates - pd.Timedelta(days=1), "return_date": dates,
                              "complete": np.isfinite(values)})
        for arm in ARMS:
            frame[arm] = values
            frame[arm + "_net_weight"] = .5
            frame[arm + "_gross_weight"] = .6
        return frame

    def test_gap_breaks_equity_and_resets_high_water_mark(self):
        curves, segments = segment_curves(self.frame([-.1, .2, np.nan, -.2, .1]))
        self.assertEqual(segments.sessions.tolist(), [2, 2])
        a = curves.loc[curves.strategy.eq("GMOM")]
        np.testing.assert_allclose(a.segment_equity, [100, 90, 108, 100, 80, 88])
        np.testing.assert_allclose(a.segment_drawdown, [0, -.1, 0, 0, -.2, -.12])

    def test_month_with_missing_session_is_wholly_unavailable(self):
        frame = self.frame([.1, np.nan, .2, -.1], ["2024-01-30", "2024-01-31", "2024-02-01", "2024-02-02"])
        monthly = monthly_returns(frame)
        self.assertTrue(np.isnan(monthly.GMOM.iloc[0]))
        self.assertAlmostEqual(monthly.GMOM.iloc[1], .08)
        self.assertEqual(monthly.complete.tolist(), [False, True])

    def test_rolling_window_requires_every_scheduled_return(self):
        frame = self.frame([.01, .02, .03, np.nan, .01, .02, .03])
        rolling = rolling_statistics(frame, window=3)
        self.assertTrue(np.isnan(rolling.GMOM_sharpe.iloc[3:6]).all())
        self.assertAlmostEqual(rolling.GMOM_sharpe.iloc[2], rolling.GMOM_sharpe.iloc[6])

    def test_metrics_withhold_path_statistics_but_use_defined_tail_and_sortino(self):
        frame = self.frame([-.02, .01, 0., np.nan, .03])
        result = extended_metrics(frame, "GMOM")
        self.assertIsNone(result["cagr"])
        self.assertIsNone(result["maximum_drawdown"])
        self.assertAlmostEqual(result["positive_day_fraction"], .5)
        self.assertAlmostEqual(result["mean_worst_5pct_days"], -.02)
        self.assertEqual(result["tail_observations"], 1)
        self.assertAlmostEqual(result["sortino_zero_threshold"], np.sqrt(252) * .005 / .01)
        self.assertEqual(result["worst_return_date"], "2024-01-02")

    def test_run_can_cross_refit_year_without_artificial_reset(self):
        curves, segments = segment_curves(self.frame([.01, .02], ["2023-12-29", "2024-01-02"]))
        self.assertEqual(len(segments), 1)
        self.assertAlmostEqual(curves.loc[curves.strategy.eq("GMOM"), "segment_equity"].iloc[-1], 103.02)


if __name__ == "__main__":
    unittest.main()
