"""Structural feature fixtures; no numerical research data or fitted models."""
from collections import Counter
import math
import unittest

import numpy as np
import pandas as pd
import pyarrow as pa

from dmv2.daily_features import compute_daily_features, daily_feature_specs
from dmv2.universe import DataContractError


def fixture(n=520):
    # A declared artificial session grid, not a claim about historical exchanges.
    dates = pd.bdate_range("1997-01-01", periods=n).strftime("%Y-%m-%d").tolist()
    x = np.arange(n)
    c = 100 * np.exp(.001 * x + .01 * np.sin(x))
    prices = pd.DataFrame({"intrinio_security_id": "security-a", "trading_date": dates,
                           "frequency": "daily", "intraperiod": False,
                           "price_quality_status": "accepted", "adj_close": c,
                           "adj_open": c * .998, "adj_high": c * 1.02,
                           "adj_low": c * .98, "close": c, "high": c * 1.02,
                           "low": c * .98, "volume": 100. + x, "factor": 1.,
                           "dividend": 0., "split_ratio": 1.})
    return prices, dates


def run(prices, dates, signals=None, requested=("security-a",)):
    return compute_daily_features(prices, dates, signal_dates=signals or [dates[-1]],
                                  requested_security_ids=requested)


def status(result, feature, signal=None, sid="security-a"):
    selected = result.coverage[(result.coverage.intrinio_security_id == sid)
                               & (result.coverage.feature_name == feature)]
    if signal is not None: selected = selected[selected.signal_date == signal]
    return selected.iloc[0].status


class FinancialConversionTrap:
    """Keys may be read; financial conversion must not precede rejection."""
    def __init__(self, table): self.table = table
    @property
    def column_names(self): return self.table.column_names
    def select(self, columns):
        if set(columns) <= {"intrinio_security_id", "trading_date"}:
            return self.table.select(columns)
        raise AssertionError("Financial columns converted before boundary rejection")


class DailyFeatureTests(unittest.TestCase):
    def test_exact_298_family_counts_and_native_dependencies(self):
        specs = daily_feature_specs()
        self.assertEqual(len(specs), 298)
        self.assertEqual(len({spec["name"] for spec in specs}), 298)
        self.assertEqual(Counter(s["family"] for s in specs), {
            "security_return": 18, "security_momentum": 7,
            "security_return_distribution": 77, "security_trend": 72,
            "security_oscillator": 12, "security_ohlc": 45,
            "security_liquidity": 63, "security_event_and_level": 4})
        self.assertTrue(all(s["role"] == "technical" and s["formula"] for s in specs))
        self.assertFalse(any("turnover" in s["name"] for s in specs))
        self.assertEqual(next(s["lookback_sessions"] for s in specs
                              if s["name"] == "liq_abnormal_volume_ratio_252s"), 503)
        self.assertEqual(next(s["lookback_sessions"] for s in specs
                              if s["name"] == "risk_return_autocorr_1_252s"), 254)

    def test_complete_fixture_has_298_finite_values_without_audit_columns(self):
        prices, dates = fixture()
        result = run(prices, dates)
        self.assertEqual(result.table.shape, (1, 300))
        self.assertTrue(np.isfinite(result.table.iloc[:, 2:].to_numpy(dtype=float)).all())
        self.assertEqual(set(result.coverage.status), {"ok"})
        self.assertFalse(result.provenance["knowledge_time_verified"])
        self.assertFalse(result.provenance["model_ready"])
        self.assertFalse(any("status" in column for column in result.table.columns))

    def test_future_perturbation_cannot_change_earlier_signal(self):
        prices, dates = fixture()
        signals = [dates[300], dates[-1]]
        before = run(prices, dates, signals)
        later = prices.copy(deep=True)
        mask = later.trading_date > dates[300]
        for column in ("adj_close", "adj_open", "adj_high", "adj_low", "close", "high", "low", "volume"):
            later.loc[mask, column] *= 1000
        later.loc[mask, "dividend"] = 30.
        after = run(later, dates, signals)
        pd.testing.assert_series_equal(before.table.iloc[0], after.table.iloc[0])
        pd.testing.assert_frame_equal(before.coverage[before.coverage.signal_date == dates[300]].reset_index(drop=True),
                                      after.coverage[after.coverage.signal_date == dates[300]].reset_index(drop=True))
        self.assertNotEqual(before.table.iloc[1].liq_log_volume_5s, after.table.iloc[1].liq_log_volume_5s)

    def test_input_is_unchanged_and_arrow_pandas_parity(self):
        prices, dates = fixture(280)
        original = prices.copy(deep=True)
        pandas = run(prices, dates)
        arrow = run(pa.Table.from_pandas(prices, preserve_index=False), dates)
        pd.testing.assert_frame_equal(prices, original)
        pd.testing.assert_frame_equal(pandas.table, arrow.table)
        pd.testing.assert_frame_equal(pandas.coverage, arrow.coverage)

    def test_future_and_holdout_rejected_before_financial_conversion(self):
        prices, dates = fixture(30)
        with self.assertRaisesRegex(DataContractError, "future or protected"):
            run(FinancialConversionTrap(pa.Table.from_pandas(prices)), dates, [dates[-2]])
        protected = prices.copy()
        protected.loc[len(protected) - 1, "trading_date"] = "2024-01-02"
        with self.assertRaisesRegex(DataContractError, "future or protected"):
            run(FinancialConversionTrap(pa.Table.from_pandas(protected)), dates, [dates[-2]])

    def test_target_and_outcome_fields_rejected_even_when_empty(self):
        prices, dates = fixture(30)
        for name in ("target", "target_return", "provider_adjusted_return", "outcome_end",
                     "next_month_return", "future_rank", "label_class", "prediction_score",
                     "realized_return", "delisted", "last_price_date"):
            with self.subTest(name=name), self.assertRaisesRegex(DataContractError, "target/outcome"):
                run(prices.assign(**{name: np.nan}), dates)

    def test_missing_exchange_session_does_not_compress_horizons(self):
        prices, dates = fixture()
        hole = len(prices) - 3
        result = run(prices.drop(index=hole), dates)
        self.assertTrue(math.isnan(result.table.iloc[0].px_simple_return_20s))
        self.assertEqual(status(result, "px_simple_return_20s"), "missing_session")
        # A skipped return excludes the most recent five sessions, so this gap
        # is outside its actual historical dependency interval.
        self.assertTrue(math.isfinite(result.table.iloc[0].px_skip_return_42_5s))
        self.assertEqual(status(result, "event_dividend_event"), "ok")

    def test_invalid_quote_metadata_masks_complete_windows(self):
        prices, dates = fixture(50)
        for field, value in (("frequency", "weekly"), ("intraperiod", True),
                             ("price_quality_status", "rejected"), ("intraperiod", None)):
            changed = prices.copy()
            changed[field] = changed[field].astype(object)
            changed.loc[len(changed) - 3, field] = value
            result = run(changed, dates)
            self.assertEqual(status(result, "px_simple_return_20s"), "invalid_quote_metadata")
            self.assertTrue(math.isnan(result.table.iloc[0].px_simple_return_20s))
            self.assertEqual(status(result, "event_dividend_event"), "ok")

    def test_invalid_ohlc_does_not_discard_usable_close_only_features(self):
        prices, dates = fixture(50)
        changed = prices.copy()
        changed.loc[len(changed) - 1, "adj_high"] = changed.iloc[-1].adj_low / 2
        result = run(changed, dates)
        baseline = run(prices, dates)
        self.assertEqual(result.table.iloc[0].px_simple_return_20s, baseline.table.iloc[0].px_simple_return_20s)
        self.assertEqual(status(result, "trend_high_distance_20s"), "missing_or_invalid_input")
        changed = prices.assign(adjusted_ohlc_bounds_valid=True)
        changed.loc[len(changed) - 1, "adjusted_ohlc_bounds_valid"] = False
        flagged = run(changed, dates)
        self.assertEqual(status(flagged, "px_simple_return_20s"), "ok")
        self.assertEqual(status(flagged, "ohlc_high_low_range_1s"), "missing_or_invalid_input")

    def test_missing_optional_inputs_preserve_close_only_features(self):
        prices, dates = fixture(50)
        result = run(prices.drop(columns=["adj_open", "adj_high", "adj_low", "volume"]), dates)
        self.assertEqual(status(result, "px_simple_return_20s"), "ok")
        self.assertEqual(status(result, "ohlc_body_fraction_1s"), "missing_or_invalid_input")
        self.assertEqual(status(result, "liq_log_volume_20s"), "missing_or_invalid_input")

    def test_missing_security_is_retained_without_future_survival_filter(self):
        prices, dates = fixture(50)
        result = run(prices, dates, requested=("security-a", "no-quotes"))
        self.assertEqual(list(result.table.intrinio_security_id), ["security-a", "no-quotes"])
        self.assertTrue(result.table.iloc[1, 2:].isna().all())
        self.assertEqual(status(result, "event_dividend_event", sid="no-quotes"), "missing_session")

    def test_nested_volume_requires_503_complete_sessions(self):
        prices, dates = fixture(503)
        result = run(prices, dates, [dates[-2], dates[-1]])
        name = "liq_abnormal_volume_ratio_252s"
        self.assertEqual(status(result, name, dates[-2]), "insufficient_calendar_warmup")
        self.assertEqual(status(result, name, dates[-1]), "ok")
        volume = prices.volume.to_numpy()
        expected = np.mean([volume[i] / np.mean(volume[i - 251:i + 1]) for i in range(251, 503)])
        self.assertAlmostEqual(result.table.iloc[1][name], expected)
        missing = run(prices.drop(index=0), dates)
        self.assertEqual(status(missing, name), "missing_session")
        self.assertTrue(math.isnan(missing.table.iloc[0][name]))

    def test_endpoint_skip_and_distribution_formulas(self):
        prices, dates = fixture(50)
        result = run(prices, dates).table.iloc[0]
        c = prices.adj_close.to_numpy()
        self.assertAlmostEqual(result.px_simple_return_20s, c[-1] / c[-21] - 1)
        self.assertAlmostEqual(result.px_skip_return_21_1s, c[-2] / c[-22] - 1)
        log_r = np.log(c[-6:-1] / c[-7:-2])  # prior window for autocorr comparison below
        current_r = np.log(c[-5:] / c[-6:-1])
        self.assertAlmostEqual(result.risk_mean_log_return_5s, current_r.mean())
        self.assertAlmostEqual(result.risk_realized_vol_5s, current_r.std(ddof=1) * math.sqrt(252))
        self.assertAlmostEqual(result.risk_return_autocorr_1_5s, np.corrcoef(current_r, log_r)[0, 1])
        self.assertAlmostEqual(result.risk_downside_vol_5s, math.sqrt(252 * np.minimum(current_r, 0).dot(np.minimum(current_r, 0)) / 5))
        n = len(current_r)
        centered = current_r - current_r.mean()
        m2, m3, m4 = (np.mean(centered ** power) for power in (2, 3, 4))
        skew = math.sqrt(n * (n - 1)) / (n - 2) * m3 / m2 ** 1.5
        kurt = (n - 1) / ((n - 2) * (n - 3)) * ((n + 1) * (m4 / m2 ** 2 - 3) + 6)
        self.assertAlmostEqual(result.risk_skew_5s, skew)
        self.assertAlmostEqual(result.risk_excess_kurtosis_5s, kurt)

    def test_finite_ema_simple_rsi_and_peak_distance_conventions(self):
        prices, dates = fixture(50)
        result = run(prices, dates).table.iloc[0]
        c = prices.adj_close.to_numpy()
        weights = np.exp((np.arange(20) - 19) * 2 / 21)
        expected_ema = np.dot(c[-20:], weights) / weights.sum()
        self.assertAlmostEqual(result.trend_ema_gap_20s, c[-1] / expected_ema - 1)
        changes = np.diff(c[-15:])
        gain, loss = np.maximum(changes, 0).mean(), np.maximum(-changes, 0).mean()
        self.assertAlmostEqual(result.osc_rsi_14s, 100 - 100 / (1 + gain / loss))
        self.assertAlmostEqual(result.trend_drawdown_20s, c[-1] / c[-20:].max() - 1)
        self.assertAlmostEqual(result.trend_path_efficiency_20s,
                               abs(c[-1] - c[-21]) / np.abs(np.diff(c[-21:])).sum())
        self.assertTrue(0 <= result.osc_stochastic_20s <= 1)

    def test_true_flat_and_zero_volume_windows_stay_undefined_when_required(self):
        prices, dates = fixture(280)
        for field in ("adj_close", "adj_open", "adj_high", "adj_low", "close", "high", "low"):
            prices[field] = 100.
        prices["volume"] = 0.
        result = run(prices, dates)
        row = result.table.iloc[0]
        self.assertEqual(row.px_simple_return_20s, 0.)
        self.assertEqual(row.liq_zero_volume_share_20s, 1.)
        self.assertEqual(row.risk_realized_vol_20s, 0.)
        for feature in ("osc_rsi_20s", "osc_stochastic_20s", "trend_path_efficiency_20s",
                        "risk_skew_20s", "risk_excess_kurtosis_20s", "risk_return_autocorr_1_20s",
                        "liq_abnormal_volume_ratio_5s", "liq_amihud_20s"):
            with self.subTest(feature=feature):
                self.assertTrue(math.isnan(row[feature]))
                self.assertEqual(status(result, feature), "undefined_formula")
        self.assertFalse(np.isinf(result.table.iloc[:, 2:].to_numpy(dtype=float)).any())

    def test_calendar_duplicate_unknown_session_and_unrequested_id_reject(self):
        prices, dates = fixture(30)
        with self.assertRaisesRegex(DataContractError, "Duplicate daily"):
            run(pd.concat([prices, prices.iloc[[0]]]), dates)
        with self.assertRaisesRegex(DataContractError, "exact supplied sessions"):
            run(prices, dates[1:])
        with self.assertRaisesRegex(DataContractError, "unrequested"):
            run(prices, dates, requested=("other-id",))
        with self.assertRaisesRegex(DataContractError, "unique"):
            run(prices, dates + [dates[-1]])
        with self.assertRaisesRegex(DataContractError, "increasing"):
            run(prices, list(reversed(dates)))


if __name__ == "__main__":
    unittest.main()
