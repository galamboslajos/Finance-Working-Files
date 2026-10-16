"""Hand-accounted units/cash cases; no research outcomes or model fitting."""
import unittest

import numpy as np
import pandas as pd

from dmv2.backtest_accounting import DailyAccountingPolicy, run_daily_accounting
from dmv2.universe import DataContractError


def selections(months=(("2017-12-29", "2018-01-02"),)):
    return pd.DataFrame([
        {"signal_date": signal, "execution_date": execution, "source_security_id": name,
         "intrinio_security_id": name.upper(), "side": side, "identity_review_flag": True}
        for signal, execution in months for name, side in (("a", "long"), ("b", "short"))
    ])


def schedule(months=(("2017-12-29", "2018-01-02"),)):
    return pd.DataFrame(months, columns=["signal_date", "execution_date"])


def quotes(dates=("2018-01-02", "2018-01-03"), a=(100., 110.), b=(100., 90.),
           aclose=None, bclose=None):
    return pd.DataFrame([
        {"trading_date": date, "intrinio_security_id": name,
         "adj_open": value, "adj_close": close, "frequency": "daily",
         "intraperiod": False, "price_quality_status": "accepted"}
        for i, date in enumerate(dates) for name, value, close in (
            ("A", a[i], a[i] if aclose is None else aclose[i]),
            ("B", b[i], b[i] if bclose is None else bclose[i]))
    ])


def ohlc(prices):
    result = prices.copy(deep=True)
    result["adj_high"] = result[["adj_open", "adj_close"]].max(axis=1)
    result["adj_low"] = result[["adj_open", "adj_close"]].min(axis=1)
    return result


class DailyAccountingTests(unittest.TestCase):
    def run_case(self, **changes):
        args = dict(selections=selections(), prices=quotes(),
                    sessions=["2018-01-02", "2018-01-03"], rebalances=schedule(),
                    terminal_date="2018-01-03")
        args.update(changes)
        return run_daily_accounting(**args)

    def test_fixed_units_cash_and_two_sided_price_attribution(self):
        result = self.run_case()
        self.assertEqual(result.status, "complete_proxy")
        self.assertAlmostEqual(result.summary["final_nav"], 1.1)
        self.assertAlmostEqual(result.daily.iloc[0].cash, 1.)
        next_open = result.daily[result.daily.mark.eq("open")].iloc[1]
        self.assertAlmostEqual(next_open.long_price_pnl, .05)
        self.assertAlmostEqual(next_open.short_price_pnl, .05)
        self.assertAlmostEqual(next_open.nav, 1.1)
        # Fixed units imply drift, not daily restoration of 50%/50% weights.
        self.assertAlmostEqual(next_open.long_market_value, .55)
        self.assertAlmostEqual(next_open.short_market_value, -.45)
        self.assertAlmostEqual(next_open.gross_exposure, 1 / 1.1)
        self.assertAlmostEqual(next_open.net_exposure, .1 / 1.1)
        self.assertAlmostEqual(result.daily.iloc[-1].cash, 1.1)
        self.assertEqual(result.daily.iloc[-1].gross_exposure, 0)

    def test_terminal_intraday_separate_close_point_and_liquidation(self):
        prices = quotes(aclose=[100., 121.], bclose=[100., 81.])
        result = self.run_case(prices=prices)
        self.assertEqual(result.daily.mark.tolist(), ["open", "close", "open", "terminal_close"])
        self.assertAlmostEqual(result.daily.iloc[-1].long_price_pnl, .055)
        self.assertAlmostEqual(result.daily.iloc[-1].short_price_pnl, .045)
        self.assertAlmostEqual(result.summary["final_nav"], 1.2)
        self.assertEqual(result.daily.iloc[-1].period_start_mark, "open")
        self.assertEqual(result.daily.iloc[-1].date, result.daily.iloc[-2].date)
        self.assertTrue(result.holdings.query("mark == 'terminal_close'").units.eq(0).all())
        self.assertEqual(result.trades.phase.tolist(), ["rebalance", "rebalance", "terminal_liquidation", "terminal_liquidation"])

    def test_one_way_entry_and_exit_costs_have_hand_calculated_cash(self):
        result = self.run_case(policy=DailyAccountingPolicy(cost_bps=10))
        # Entry absolute notional 1; exit absolute notional .55 + .45 = 1.
        self.assertAlmostEqual(result.summary["final_nav"], 1.098)
        self.assertAlmostEqual(result.trades.cost.sum(), .002)
        self.assertAlmostEqual(result.daily.net_pnl.sum(), .098)
        self.assertAlmostEqual(result.daily.iloc[0]["return"], -.001)
        self.assertAlmostEqual(np.prod(1 + result.daily["return"]), 1.098)
        self.assertAlmostEqual(result.daily.iloc[0].turnover_one_way, 1)

    def test_monthly_rebalance_uses_pre_cost_nav_and_drifted_units(self):
        months = (("2017-12-29", "2018-01-02"), ("2018-01-31", "2018-02-01"))
        dates = ["2018-01-02", "2018-01-31", "2018-02-01", "2018-02-02"]
        result = self.run_case(selections=selections(months), rebalances=schedule(months),
                               sessions=dates, terminal_date=dates[-1],
                               prices=quotes(dates, [100, 110, 110, 110], [100, 90, 90, 90]))
        feb = result.daily[result.daily.date.eq(pd.Timestamp("2018-02-01"))].iloc[0]
        self.assertAlmostEqual(feb.nav_before_cost, 1.1)
        self.assertAlmostEqual(feb.long_market_value, .55)
        self.assertAlmostEqual(feb.short_market_value, -.55)
        # Long leg needs no trade; short leg increases absolute value by .10.
        self.assertAlmostEqual(feb.traded_notional, .1)
        self.assertAlmostEqual(result.summary["final_nav"], 1.1)

    def test_borrow_uses_previous_short_value_actual_calendar_days(self):
        dates = ["2018-01-05", "2018-01-08"]
        month = (("2017-12-29", dates[0]),)
        result = self.run_case(selections=selections(month), rebalances=schedule(month),
                               prices=quotes(dates, [100, 100], [100, 100]),
                               sessions=dates, terminal_date=dates[-1],
                               policy=DailyAccountingPolicy(borrow_annual_bps=100))
        expected = .5 * .01 * 3 / 365
        self.assertAlmostEqual(result.daily.borrow_cost.sum(), expected)
        self.assertAlmostEqual(result.daily.iloc[-1].borrow_cost, 0)
        self.assertAlmostEqual(result.summary["final_nav"], 1 - expected)

    def test_missing_held_quote_fails_permanently_and_retains_position_units(self):
        dates = ["2018-01-02", "2018-01-03", "2018-01-04"]
        prices = quotes(dates, [100, 110, 120], [100, 90, 80])
        prices = prices[~(prices.trading_date.eq("2018-01-03") & prices.intrinio_security_id.eq("A"))]
        result = self.run_case(prices=prices, sessions=dates, terminal_date=dates[-1])
        self.assertEqual(result.status, "failed_closed")
        self.assertIsNone(result.summary["total_return"])
        self.assertTrue(result.daily.iloc[2:].nav.isna().all())
        self.assertTrue(result.daily.iloc[2:]["return"].isna().all())
        self.assertEqual(len(result.trades), 2)
        self.assertEqual(result.failures.iloc[0].role, "held")
        retained = result.holdings[result.holdings.date.eq(pd.Timestamp(dates[-1]))]
        self.assertEqual(set(retained.source_security_id), {"a", "b"})
        self.assertTrue(retained.valuation_price.isna().all())
        self.assertTrue(retained.loc[retained.source_security_id.eq("a"), "units"].eq(.005).all())

    def test_missing_selected_quote_never_substitutes_or_partial_trades(self):
        prices = quotes().query("intrinio_security_id != 'A'")
        result = self.run_case(prices=prices)
        self.assertEqual(result.status, "failed_closed")
        self.assertTrue(result.trades.empty)
        self.assertTrue(result.daily.nav.isna().all())
        self.assertEqual(result.failures.iloc[0].source_security_id, "a")
        self.assertEqual(result.failures.iloc[0].role, "selected")

    def test_empty_later_month_fails_instead_of_cashing_out_or_restarting(self):
        months = (("2017-12-29", "2018-01-02"), ("2018-01-31", "2018-02-01"),
                  ("2018-02-28", "2018-03-01"))
        dates = ["2018-01-02", "2018-02-01", "2018-03-01"]
        selected = selections(months)
        selected = selected[selected.execution_date.ne("2018-02-01")]
        result = self.run_case(selections=selected, rebalances=schedule(months),
                               sessions=dates, terminal_date=dates[-1],
                               prices=quotes(dates, [100]*3, [100]*3))
        self.assertEqual(result.failures.iloc[0].reason, "empty_long_or_short_leg")
        self.assertTrue(result.daily.iloc[2:].nav.isna().all())
        self.assertEqual(len(result.trades), 2)
        self.assertEqual(len(result.holdings[result.holdings.date.eq(pd.Timestamp("2018-03-01"))]), 4)

    def test_required_ten_names_per_leg_withholds_underfilled_selection(self):
        result = self.run_case(policy=DailyAccountingPolicy(required_names_per_side=10))
        self.assertEqual(result.failures.iloc[0].reason, "incorrect_names_per_side")
        self.assertTrue(result.trades.empty)
        self.assertTrue(result.daily.nav.isna().all())

    def test_terminal_missing_price_does_not_reuse_open_or_read_next_year(self):
        prices = quotes()
        prices.loc[prices.trading_date.eq("2018-01-03") & prices.intrinio_security_id.eq("B"), "adj_close"] = np.nan
        result = self.run_case(prices=prices)
        self.assertEqual(result.daily.iloc[-1].status, "failed_closed")
        self.assertEqual(result.failures.iloc[0].price_field, "adj_close")
        self.assertEqual(len(result.trades), 2)

    def test_bad_quote_states_fail_closed_without_false_or_zero_coercions(self):
        changes = [{"adj_open": 0}, {"adj_open": np.inf}, {"adj_open": -1},
                   {"adj_open": "100"}, {"adj_open": True}, {"intraperiod": None},
                   {"intraperiod": 0}, {"frequency": "weekly"},
                   {"frequency": pd.NA}, {"price_quality_status": pd.NA},
                   {"price_quality_status": "unknown"}, {"price_accepted": "True"},
                   {"adjusted_ohlc_bounds_valid": False}, {"raw_ohlc_bounds_valid": None}]
        for change in changes:
            with self.subTest(change=change):
                prices = quotes().astype(object)
                for column, value in change.items():
                    prices.loc[0, column] = value
                self.assertEqual(self.run_case(prices=prices).status, "failed_closed")

    def test_metadata_and_inputs_are_preserved_without_double_counting_actions(self):
        prices, selected = quotes(), selections()
        prices["dividend"], prices["split_ratio"] = 100., 100.
        before_prices, before_selected = prices.copy(deep=True), selected.copy(deep=True)
        result = self.run_case(prices=prices, selections=selected)
        self.assertAlmostEqual(result.summary["final_nav"], 1.1)
        self.assertTrue(result.holdings.identity_review_flag.all())
        self.assertTrue(result.trades.identity_review_flag.all())
        pd.testing.assert_frame_equal(prices, before_prices)
        pd.testing.assert_frame_equal(selected, before_selected)
        self.assertFalse(result.summary["economic_accounting_certified"])
        self.assertFalse(result.summary["knowledge_time_certified"])

    def test_duplicate_identity_and_price_routes_raise(self):
        for name in ("source", "provider", "price"):
            with self.subTest(name=name), self.assertRaises(DataContractError):
                chosen, prices = selections(), quotes()
                if name == "source":
                    chosen.loc[1, "source_security_id"] = "a"
                elif name == "provider":
                    chosen.loc[1, "intrinio_security_id"] = "A"
                else:
                    prices = pd.concat([prices, prices.iloc[[0]]], ignore_index=True)
                self.run_case(selections=chosen, prices=prices)

    def test_missing_schedule_or_unannounced_signal_or_route_changes_raise(self):
        with self.assertRaises(DataContractError):
            self.run_case(rebalances=schedule().iloc[:0])
        with self.assertRaises(DataContractError):
            chosen = selections()
            chosen.loc[0, "signal_date"] = "2018-01-01"
            self.run_case(selections=chosen)
        months = (("2017-12-29", "2018-01-02"), ("2018-01-31", "2018-02-01"))
        chosen = selections(months)
        chosen.loc[2, "intrinio_security_id"] = "C"
        with self.assertRaisesRegex(DataContractError, "route changes"):
            self.run_case(selections=chosen, rebalances=schedule(months),
                          sessions=["2018-01-02", "2018-02-01"], terminal_date="2018-02-01",
                          prices=quotes(["2018-01-02", "2018-02-01"]))

    def test_held_to_new_source_provider_collision_raises(self):
        months = (("2017-12-29", "2018-01-02"), ("2018-01-31", "2018-02-01"))
        chosen = selections(months)
        chosen.loc[2, "source_security_id"] = "c"
        with self.assertRaisesRegex(DataContractError, "different held"):
            self.run_case(selections=chosen, rebalances=schedule(months),
                          sessions=["2018-01-02", "2018-02-01"], terminal_date="2018-02-01",
                          prices=quotes(["2018-01-02", "2018-02-01"]))

    def test_future_financial_rows_rejected_before_numeric_conversion(self):
        class PoisonPrice:
            def __float__(self):
                raise AssertionError("Future financial values were converted")
        prices = quotes().astype(object)
        prices.loc[0, "trading_date"] = "2020-01-02"
        prices.loc[0, "adj_open"] = PoisonPrice()
        with self.assertRaisesRegex(DataContractError, "outside authorized"):
            self.run_case(prices=prices)

    def test_invalid_policy_and_terminal_bounds_raise(self):
        for kwargs in ({"cost_bps": -1}, {"borrow_annual_bps": np.nan},
                       {"initial_nav": 0}, {"initial_nav": True},
                       {"required_names_per_side": 0}, {"required_names_per_side": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(DataContractError):
                DailyAccountingPolicy(**kwargs)
        with self.assertRaises(DataContractError):
            self.run_case(terminal_date="2018-01-04")
        with self.assertRaisesRegex(DataContractError, "2018–2019"):
            self.run_case(sessions=["2020-01-02"], terminal_date="2020-01-02")

    def test_path_insolvency_is_failure_and_does_not_resume(self):
        dates = ["2018-01-02", "2018-01-03", "2018-01-04"]
        result = self.run_case(sessions=dates, terminal_date=dates[-1],
                               prices=quotes(dates, [100, 100, 100], [100, 400, 100]))
        self.assertEqual(result.status, "failed_closed")
        self.assertEqual(result.failures.iloc[0].reason, "nonpositive_or_nonfinite_nav")
        self.assertTrue(result.daily.iloc[2:].nav.isna().all())

    def test_explicit_adapter_quality_can_be_used_without_raw_flags(self):
        prices = quotes().drop(columns=["frequency", "intraperiod", "price_quality_status"])
        prices["price_accepted"] = True
        self.assertAlmostEqual(self.run_case(prices=prices).summary["final_nav"], 1.1)

    def test_long_intraday_stop_uses_adverse_low_and_not_closing_recovery(self):
        prices = ohlc(quotes(a=[100, 100], b=[100, 100]))
        prices.loc[0, "adj_low"] = 85.
        result = self.run_case(prices=prices, policy=DailyAccountingPolicy(asset_stop_fraction=.1))
        self.assertAlmostEqual(result.summary["final_nav"], .95)
        self.assertEqual(len(result.stops), 1)
        stop = result.stops.iloc[0]
        self.assertEqual(stop.source_security_id, "a")
        self.assertEqual(stop.fill_price, 90)
        self.assertEqual(stop.fill_type, "idealized_intraday_threshold")
        self.assertEqual(stop.mark, "close")
        self.assertEqual(result.trades.query("phase == 'stop_intraday'").iloc[0].price, 90)
        self.assertFalse((result.holdings.source_security_id.eq("a")
                          & result.holdings.date.eq(pd.Timestamp("2018-01-03"))
                          & result.holdings.units.gt(0)).any())

    def test_short_intraday_stop_triggers_on_exact_boundary_high(self):
        prices = ohlc(quotes(a=[100, 100], b=[100, 100]))
        prices.loc[1, "adj_high"] = 110.
        result = self.run_case(prices=prices, policy=DailyAccountingPolicy(asset_stop_fraction=.1))
        self.assertAlmostEqual(result.summary["final_nav"], .95)
        self.assertEqual(result.stops.iloc[0].source_security_id, "b")
        self.assertEqual(result.stops.iloc[0].stop_price, 110)
        self.assertEqual(result.stops.iloc[0].fill_price, 110)

    def test_gap_stop_does_not_fabricate_a_ten_percent_loss_cap(self):
        prices = ohlc(quotes(a=[100, 80], b=[100, 100], aclose=[100, 100]))
        result = self.run_case(prices=prices, policy=DailyAccountingPolicy(asset_stop_fraction=.1))
        self.assertAlmostEqual(result.summary["final_nav"], .9)
        self.assertEqual(result.stops.iloc[0].fill_type, "gap_open")
        self.assertEqual(result.stops.iloc[0].fill_price, 80)
        self.assertEqual(result.stops.iloc[0].stop_price, 90)
        self.assertEqual(result.stops.iloc[0].mark, "open")
        short_prices = ohlc(quotes(a=[100, 100], b=[100, 130]))
        short_result = self.run_case(prices=short_prices, policy=DailyAccountingPolicy(asset_stop_fraction=.1))
        self.assertAlmostEqual(short_result.summary["final_nav"], .85)
        self.assertEqual(short_result.stops.iloc[0].fill_price, 130)

    def test_both_assets_stop_independently_and_cash_does_not_renormalize(self):
        prices = ohlc(quotes(a=[100, 100], b=[100, 100]))
        prices.loc[0, "adj_low"] = 90.
        prices.loc[1, "adj_high"] = 110.
        result = self.run_case(prices=prices, policy=DailyAccountingPolicy(asset_stop_fraction=.1))
        self.assertEqual(result.summary["stop_count"], 2)
        self.assertAlmostEqual(result.summary["final_nav"], .9)
        self.assertTrue(result.daily.iloc[1:].gross_exposure.eq(0).all())
        self.assertEqual(len(result.trades), 4)
        self.assertAlmostEqual(result.daily.iloc[1].long_price_pnl, -.05)
        self.assertAlmostEqual(result.daily.iloc[1].short_price_pnl, -.05)

    def test_stop_cost_charged_on_actual_one_way_exit_notional(self):
        prices = ohlc(quotes(a=[100, 100], b=[100, 100]))
        prices.loc[0, "adj_low"] = 90.
        result = self.run_case(prices=prices, policy=DailyAccountingPolicy(cost_bps=10, asset_stop_fraction=.1))
        # Entry 1.0, long-stop .45, remaining short liquidation .5.
        self.assertAlmostEqual(result.trades.cost.sum(), .00195)
        self.assertAlmostEqual(result.stops.iloc[0].cost, .00045)
        self.assertAlmostEqual(result.summary["final_nav"], .94805)

    def test_monthly_rebalance_reopens_stopped_name_and_resets_reference(self):
        months = (("2017-12-29", "2018-01-02"), ("2018-01-31", "2018-02-01"))
        dates = ["2018-01-02", "2018-01-31", "2018-02-01", "2018-02-02"]
        prices = ohlc(quotes(dates, [100]*4, [100]*4))
        prices.loc[0, "adj_low"] = 85.
        result = self.run_case(selections=selections(months), rebalances=schedule(months),
                               sessions=dates, terminal_date=dates[-1], prices=prices,
                               policy=DailyAccountingPolicy(asset_stop_fraction=.1))
        jan_end = result.holdings[result.holdings.date.eq(pd.Timestamp("2018-01-31"))]
        self.assertFalse(jan_end.source_security_id.eq("a").any())
        feb_entry = result.holdings[(result.holdings.date.eq(pd.Timestamp("2018-02-01")))
                                   & result.holdings.mark.eq("open")
                                   & result.holdings.source_security_id.eq("a")].iloc[0]
        self.assertGreater(feb_entry.units, 0)
        self.assertEqual(feb_entry.entry_reference, 100)
        self.assertEqual(feb_entry.stop_price, 90)
        self.assertEqual(result.summary["stop_count"], 1)

    def test_retained_name_has_new_monthly_reference_not_trailing_high(self):
        months = (("2017-12-29", "2018-01-02"), ("2018-01-31", "2018-02-01"))
        dates = ["2018-01-02", "2018-01-31", "2018-02-01", "2018-02-02"]
        prices = ohlc(quotes(dates, [100, 120, 120, 105], [100]*4))
        result = self.run_case(selections=selections(months), rebalances=schedule(months),
                               sessions=dates, terminal_date=dates[-1], prices=prices,
                               policy=DailyAccountingPolicy(asset_stop_fraction=.1))
        self.assertEqual(result.summary["stop_count"], 1)
        self.assertEqual(result.stops.iloc[0].entry_reference, 120)
        self.assertEqual(result.stops.iloc[0].stop_price, 108)
        self.assertEqual(result.stops.iloc[0].fill_price, 105)
        self.assertAlmostEqual(result.summary["final_nav"], 1.03125)

    def test_incomplete_stop_bar_fails_closed_instead_of_guessing_order(self):
        for field, value in (("adj_low", np.nan), ("adj_high", 50.), ("adj_low", -1)):
            with self.subTest(field=field, value=value):
                prices = ohlc(quotes())
                prices.loc[0, field] = value
                result = self.run_case(prices=prices, policy=DailyAccountingPolicy(asset_stop_fraction=.1))
                self.assertEqual(result.status, "failed_closed")
                self.assertTrue(result.daily.iloc[1:].nav.isna().all())
                self.assertEqual(len(result.trades), 2)
                self.assertTrue(result.stops.empty)

    def test_terminal_bar_stop_fill_takes_precedence_over_terminal_close(self):
        prices = ohlc(quotes(a=[100, 100], b=[100, 100]))
        prices.loc[2, "adj_low"] = 85.
        result = self.run_case(prices=prices, policy=DailyAccountingPolicy(asset_stop_fraction=.1))
        self.assertEqual(result.stops.iloc[0].mark, "terminal_close")
        stopped_trade = result.trades[result.trades.source_security_id.eq("a")].iloc[-1]
        self.assertEqual(stopped_trade.phase, "stop_intraday")
        self.assertEqual(stopped_trade.price, 90)
        self.assertAlmostEqual(result.summary["final_nav"], .95)

    def test_stop_parameters_and_required_ohlc_are_validated(self):
        for value in (0, 1, -1, True, "0.1", np.nan, np.inf):
            with self.subTest(value=value), self.assertRaises(DataContractError):
                DailyAccountingPolicy(asset_stop_fraction=value)
        with self.assertRaisesRegex(DataContractError, "adjusted high and low"):
            self.run_case(policy=DailyAccountingPolicy(asset_stop_fraction=.1))

    def test_scheduled_monthly_open_cancels_old_stop_before_new_episode(self):
        months = (("2017-12-29", "2018-01-02"), ("2018-01-31", "2018-02-01"))
        dates = ["2018-01-02", "2018-01-31", "2018-02-01", "2018-02-02"]
        prices = ohlc(quotes(dates, [100, 100, 80, 80], [100]*4))
        result = self.run_case(selections=selections(months), rebalances=schedule(months),
                               sessions=dates, terminal_date=dates[-1], prices=prices,
                               policy=DailyAccountingPolicy(asset_stop_fraction=.1))
        self.assertTrue(result.stops.empty)
        entry = result.holdings[(result.holdings.date.eq(pd.Timestamp("2018-02-01")))
                                & result.holdings.mark.eq("open")
                                & result.holdings.source_security_id.eq("a")].iloc[0]
        self.assertEqual(entry.entry_reference, 80)
        self.assertEqual(entry.stop_price, 72)
        self.assertAlmostEqual(result.summary["final_nav"], .9)

    def test_stop_is_fixed_to_entry_and_does_not_trail_a_monthly_high(self):
        dates = ["2018-01-02", "2018-01-03", "2018-01-04"]
        prices = ohlc(quotes(dates, [100, 120, 100], [100]*3))
        result = self.run_case(prices=prices, sessions=dates, terminal_date=dates[-1],
                               policy=DailyAccountingPolicy(asset_stop_fraction=.1))
        self.assertTrue(result.stops.empty)
        self.assertAlmostEqual(result.summary["final_nav"], 1.)


if __name__ == "__main__":
    unittest.main()
