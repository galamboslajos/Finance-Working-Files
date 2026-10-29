import unittest

import numpy as np
import pandas as pd

from build_momentum_features import FEATURES, compute_features, history_eligibility


def prices(n=700):
    t = np.arange(n)
    return pd.Series(80 * np.exp(.0004 * t + .025 * np.sin(t / 9) + .009 * np.cos(t / 3)))


class MomentumFeatureTests(unittest.TestCase):
    def test_full_nested_macd_warmup(self):
        result = compute_features(prices())
        self.assertFalse(result.feature_ready.iloc[:313].any())
        self.assertTrue(result.feature_ready.iloc[313:].all())

    def test_volatility_matches_explicit_population_weights(self):
        p = prices(100)
        result = compute_features(p)
        r = (p / p.shift() - 1).dropna().to_numpy()
        w = (1 - 2 / 61) ** np.arange(len(r) - 1, -1, -1)
        mean = np.sum(w * r) / w.sum()
        expected = np.sqrt(np.sum(w * (r - mean) ** 2) / w.sum())
        self.assertAlmostEqual(result.daily_volatility.iloc[-1], expected, places=14)

    def test_endpoint_return_and_exact_macd_recurrence(self):
        p = prices()
        result = compute_features(p)
        self.assertAlmostEqual(result.raw_return_252.iloc[-1],
            (p.iloc[-1] / p.iloc[-253] - 1) / (result.daily_volatility.iloc[-1] * np.sqrt(252)), places=12)
        emas = {}
        for scale in (8, 24):
            values = [p.iloc[0]]
            for value in p.iloc[1:]:
                values.append(value / scale + (1 - 1 / scale) * values[-1])
            emas[scale] = pd.Series(values)
        normalized = (emas[8] - emas[24]) / p.rolling(63).std(ddof=0)
        expected = normalized / normalized.rolling(252).std(ddof=0)
        np.testing.assert_allclose(result.raw_macd_8_24, expected, equal_nan=True, atol=1e-11)

    def test_future_changes_cannot_change_features(self):
        p = prices()
        first = compute_features(p.iloc[:500])
        p.iloc[500:] *= 7
        second = compute_features(p).iloc[:500]
        pd.testing.assert_frame_equal(first, second)

    def test_gap_resets_all_state_and_never_bridges_returns(self):
        p = prices(1000)
        p.iloc[400] = np.nan
        result = compute_features(p)
        self.assertTrue(result[FEATURES].iloc[400].isna().all())
        self.assertFalse(result.feature_ready.iloc[400:714].any())
        self.assertTrue(result.feature_ready.iloc[714:].all())
        self.assertTrue(np.isnan(result.feature_return_1d.iloc[401]))
        fresh = compute_features(p.iloc[401:].reset_index(drop=True))
        np.testing.assert_allclose(result[FEATURES].iloc[401:], fresh[FEATURES], equal_nan=True)

    def test_constant_price_and_invalid_close_cannot_form_features(self):
        result = compute_features(pd.Series(np.ones(500)))
        self.assertFalse(result.feature_ready.any())
        p = prices(); p.iloc[350] = -1
        self.assertTrue(np.isnan(compute_features(p).feature_adj_close.iloc[350]))

    def test_clipping_uses_raw_feature_population_moments(self):
        p = prices(); p.iloc[550] *= 1.2
        result = compute_features(p)
        raw = result[["raw_" + name for name in FEATURES]].set_axis(FEATURES, axis=1)
        moments = raw.ewm(halflife=252, adjust=True, min_periods=1)
        expected = raw.clip(moments.mean() - 5 * moments.std(bias=True),
                            moments.mean() + 5 * moments.std(bias=True), axis=None)
        np.testing.assert_allclose(result[FEATURES], expected, equal_nan=True)
        self.assertGreater(int((raw.notna() & raw.ne(expected)).to_numpy().sum()), 0)

    def test_shorter_lookback_and_identity_ledger(self):
        cube = np.ones((6, 3, 8)); cube[0, 1, 0] = np.nan
        anchor = pd.DataFrame({"source_security_id": ["a", "b", "c"], "identity_eligible": [True, True, False]})
        result = history_eligibility(cube, anchor, (3, 6))
        self.assertEqual(len(result), 3)
        self.assertEqual(result.eligible_3.tolist(), [True, True, False])
        self.assertEqual(result.eligible_6.tolist(), [True, False, False])
        self.assertEqual(result.reason_6.tolist(), ["eligible", "incomplete_feature_history", "identity_withheld"])


if __name__ == "__main__":
    unittest.main()
