"""Hand-reconciled signed entitlements, exposure continuity and fail-closed cases."""
import unittest

import pandas as pd

from dmv2.action_accounting import run_action_accounting
from dmv2.backtest_accounting import DailyAccountingPolicy, run_daily_accounting
from dmv2.universe import DataContractError
from test_backtest_accounting import selections, schedule, quotes, ohlc


def ledger(predecessor='A', successor=None, cash=110., ratio=0., date='2018-01-03'):
    return [dict(event_id='verified-case', effective_date=date, predecessor_id=predecessor,
                 successor_id=successor, cash_per_raw_share=cash, stock_ratio=ratio,
                 verified=True, evidence_reference='synthetic-legal-terms')]


def raw_quotes(prices):
    prices = ohlc(prices)
    prices['open'] = prices.adj_open
    prices['close'] = prices.adj_close
    return prices


def successor_quotes(dates=('2018-01-03',), opening=(50.,), closing=None, factor=1.):
    closing = opening if closing is None else closing
    return pd.DataFrame([dict(trading_date=date, intrinio_security_id='C', adj_open=value * factor,
                              adj_close=close * factor, adj_high=max(value, close) * factor,
                              adj_low=min(value, close) * factor, open=value, close=close,
                              frequency='daily', intraperiod=False, price_quality_status='accepted')
                         for date, value, close in zip(dates, opening, closing)])


class ActionAccountingTests(unittest.TestCase):
    def run_case(self, **changes):
        args = dict(selections=selections(), prices=raw_quotes(quotes()),
                    sessions=['2018-01-02', '2018-01-03'], rebalances=schedule(),
                    terminal_date='2018-01-03', events=[])
        args.update(changes)
        return run_action_accounting(**args)

    def event_prices(self, predecessor='A', successor=False, dates=('2018-01-02', '2018-01-03'),
                     a=(100., 110.), b=(100., 90.), successor_factor=1., successor_open=(50.,)):
        prices = raw_quotes(quotes(dates, a, b))
        prices = prices[~(prices.trading_date.gt(dates[0]) & prices.intrinio_security_id.eq(predecessor))]
        if successor:
            prices = pd.concat([prices, successor_quotes(dates[1:], successor_open, factor=successor_factor)], ignore_index=True)
        return prices

    def test_no_events_matches_frozen_engine_all_ledgers(self):
        args = dict(selections=selections(), prices=raw_quotes(quotes()),
                    sessions=['2018-01-02', '2018-01-03'], rebalances=schedule(), terminal_date='2018-01-03')
        original = run_daily_accounting(**args, policy=DailyAccountingPolicy(cost_bps=10, borrow_annual_bps=50))
        new = run_action_accounting(**args, policy=DailyAccountingPolicy(cost_bps=10, borrow_annual_bps=50))
        for key in original.tables:
            pd.testing.assert_frame_equal(original.tables[key], new.tables[key], check_exact=False, rtol=1e-12, atol=1e-15)
        self.assertTrue(new.action_events.empty)
        self.assertAlmostEqual(new.market_trades.cost.sum(), original.trades.cost.sum())

    def test_cash_only_long_redeems_real_signed_shares_without_missing_quote(self):
        result = self.run_case(prices=self.event_prices(), events=ledger())
        self.assertEqual(result.status, 'complete_proxy')
        self.assertAlmostEqual(result.summary['final_nav'], 1.1)
        self.assertAlmostEqual(result.action_events.iloc[0].signed_cash, .55)
        self.assertAlmostEqual(result.action_events.iloc[0].signed_raw_entitlement, .005)
        self.assertEqual(result.trades.source_security_id.tolist(), ['a', 'b', 'b'])
        self.assertAlmostEqual(result.daily.iloc[2].long_price_pnl, .05)
        self.assertTrue(result.holdings.loc[result.holdings.date.eq(pd.Timestamp("2018-01-03")) & result.holdings.source_security_id.eq("a")].units.eq(0).all())

    def test_cash_only_short_owes_cash_instead_of_treating_extinction_as_gain(self):
        prices = self.event_prices(predecessor='B', a=(100.,100.))
        result = self.run_case(prices=prices, events=ledger(predecessor='B', cash=120.))
        self.assertEqual(result.status, 'complete_proxy')
        self.assertAlmostEqual(result.action_events.iloc[0].signed_cash, -.6)
        self.assertAlmostEqual(result.summary['final_nav'], .9)
        self.assertAlmostEqual(result.daily.iloc[2].short_price_pnl, -.1)

    def test_adjusted_to_raw_basis_and_successor_basis_are_both_applied(self):
        prices = self.event_prices(successor=True, successor_factor=.25)
        prices.loc[prices.intrinio_security_id.eq('A'), ['adj_open','adj_close','adj_high','adj_low']] *= .5
        result = self.run_case(prices=prices, events=ledger(successor='C', cash=20., ratio=2.))
        self.assertEqual(result.status, 'complete_proxy')
        event = result.action_events.iloc[0]
        self.assertAlmostEqual(event.old_adjusted_units, .01)
        self.assertAlmostEqual(event.signed_raw_entitlement, .005)
        self.assertAlmostEqual(event.new_adjusted_units, .04)
        self.assertAlmostEqual(event.signed_cash, .1)
        # 0.005 raw predecessor shares -> .01 raw C shares ($.50) + $.10 cash.
        self.assertAlmostEqual(result.summary['final_nav'], 1.15)

    def test_mixed_short_cash_and_stock_liabilities_have_correct_sign(self):
        prices = self.event_prices(predecessor='B', successor=True, a=(100.,100.))
        result = self.run_case(prices=prices, events=ledger(predecessor='B', successor='C', cash=20., ratio=2.))
        self.assertEqual(result.status, 'complete_proxy')
        self.assertAlmostEqual(result.action_events.iloc[0].signed_cash, -.1)
        self.assertAlmostEqual(result.action_events.iloc[0].new_adjusted_units, -.01)
        self.assertAlmostEqual(result.summary['final_nav'], .9)

    def test_successor_remains_held_until_monthly_rebalance(self):
        dates = ['2018-01-02','2018-01-03','2018-01-04']
        prices = self.event_prices(successor=True, dates=dates, a=(100.,100.,100.), b=(100.,100.,100.), successor_open=(50.,60.))
        result = self.run_case(prices=prices, events=ledger(successor='C', cash=0., ratio=2.), sessions=dates, terminal_date=dates[-1])
        self.assertEqual(result.status, 'complete_proxy')
        self.assertAlmostEqual(result.summary['final_nav'], 1.1)
        held = result.holdings.loc[result.holdings.date.eq(pd.Timestamp("2018-01-03")) & result.holdings.source_security_id.eq("a")]
        self.assertEqual(set(held.intrinio_security_id), {'C'})
        self.assertTrue(held.units.eq(.01).all())
        self.assertFalse(result.trades.date.eq(pd.Timestamp('2018-01-03')).any())

    def test_conversion_has_no_fictitious_trading_cost_or_turnover(self):
        prices = self.event_prices(successor=True, b=(100.,100.))
        result = self.run_case(prices=prices, events=ledger(successor='C', cash=0., ratio=2.), policy=DailyAccountingPolicy(cost_bps=10))
        self.assertAlmostEqual(result.summary['final_nav'], .998)
        self.assertAlmostEqual(result.trades.cost.sum(), .002)
        self.assertEqual(result.trades.phase.tolist(), ['rebalance','rebalance','terminal_liquidation','terminal_liquidation'])
        self.assertAlmostEqual(result.daily.iloc[2].traded_notional, 0)

    def test_mixed_action_carries_same_economic_long_stop(self):
        prices = self.event_prices(successor=True, b=(100.,100.), successor_open=(35.,))
        # Original entry100, stop90; $20 cash +2*C: successor stop35.
        result = self.run_case(prices=prices, events=ledger(successor='C', cash=20., ratio=2.), policy=DailyAccountingPolicy(asset_stop_fraction=.1))
        self.assertEqual(result.status, 'complete_proxy')
        self.assertEqual(len(result.stops), 1)
        self.assertAlmostEqual(result.action_events.iloc[0].new_stop_price, 35.)
        self.assertAlmostEqual(result.stops.iloc[0].fill_price, 35.)
        self.assertAlmostEqual(result.summary['final_nav'], .95)
        self.assertEqual(result.stops.iloc[0].fill_type, 'gap_open')

    def test_mixed_action_short_stop_uses_cash_plus_successor_value(self):
        prices = self.event_prices(predecessor='B', successor=True, a=(100.,100.), successor_open=(45.,))
        result = self.run_case(prices=prices, events=ledger(predecessor='B', successor='C', cash=20., ratio=2.), policy=DailyAccountingPolicy(asset_stop_fraction=.1))
        self.assertEqual(result.status, 'complete_proxy')
        self.assertAlmostEqual(result.action_events.iloc[0].new_stop_price, 45.)
        self.assertAlmostEqual(result.summary['final_nav'], .95)

    def test_adjusted_split_or_dividend_fields_not_double_counted(self):
        prices = raw_quotes(quotes(a=(100.,100.),b=(100.,100.)))
        prices['dividend'] = 99.
        prices['split_ratio'] = 4.
        result = self.run_case(prices=prices)
        self.assertAlmostEqual(result.summary['final_nav'], 1.)
        self.assertTrue(result.action_events.empty)

    def test_missing_successor_quote_fails_closed_without_invented_cashout(self):
        result = self.run_case(prices=self.event_prices(), events=ledger(successor='C', cash=20., ratio=2.))
        self.assertEqual(result.status, 'failed_closed')
        self.assertIsNone(result.summary['final_nav'])
        self.assertTrue(result.action_events.empty)
        self.assertEqual(set(result.holdings.loc[result.holdings.date.eq(pd.Timestamp("2018-01-03"))].intrinio_security_id), {'A','B'})

    def test_missing_predecessor_raw_basis_fails_closed(self):
        prices = self.event_prices()
        prices.loc[prices.intrinio_security_id.eq('A'),'close'] = float('nan')
        result = self.run_case(prices=prices, events=ledger())
        self.assertEqual(result.status, 'failed_closed')
        self.assertEqual(result.failures.iloc[0].reason, 'missing_event_predecessor_basis')
        self.assertTrue(result.action_events.empty)

    def test_unevidenced_action_rejected(self):
        for update in ({'verified':False},{'evidence_reference':''},{'cash_per_raw_share':0.,'stock_ratio':0.}, {'stock_ratio':1.}):
            events = ledger()
            events[0].update(update)
            with self.assertRaises(DataContractError):
                self.run_case(events=events)

    def test_future_action_metadata_guard_runs_before_financial_conversion(self):
        class Poison:
            def __float__(self):
                raise AssertionError('Financial future value decoded')
        events = ledger(date='2020-01-02')
        events[0]['cash_per_raw_share'] = Poison()
        with self.assertRaisesRegex(DataContractError, 'outside authorized'):
            self.run_case(events=events)

    def test_duplicate_event_conflict_rejected(self):
        events = ledger() * 2
        with self.assertRaisesRegex(DataContractError, 'Duplicate'):
            self.run_case(events=events)

    def test_cash_merger_gap_is_not_magically_capped_by_stop(self):
        prices = self.event_prices(predecessor='B', a=(100.,100.))
        result = self.run_case(prices=prices, events=ledger(predecessor='B', cash=130.),
                               policy=DailyAccountingPolicy(asset_stop_fraction=.1))
        self.assertEqual(result.status, 'complete_proxy')
        self.assertTrue(result.stops.empty)
        self.assertAlmostEqual(result.summary['final_nav'], .85)
        self.assertAlmostEqual(result.action_events.signed_cash.sum(), -.65)

    def test_missing_later_successor_valuation_retains_successor_exposure(self):
        dates = ['2018-01-02','2018-01-03','2018-01-04']
        prices = self.event_prices(successor=True, dates=dates, a=(100.,100.,100.), b=(100.,100.,100.), successor_open=(50.,60.))
        prices = prices.loc[~(prices.trading_date.eq(dates[-1]) & prices.intrinio_security_id.eq('C'))]
        result = self.run_case(prices=prices, events=ledger(successor='C', cash=0., ratio=2.), sessions=dates, terminal_date=dates[-1])
        self.assertEqual(result.status, 'failed_closed')
        self.assertIsNone(result.summary['total_return'])
        remaining = result.holdings.loc[result.holdings.date.eq(pd.Timestamp(dates[-1])) & result.holdings.source_security_id.eq('a')]
        self.assertFalse(remaining.empty)
        self.assertEqual(set(remaining.intrinio_security_id), {'C'})
        self.assertTrue(remaining.units.eq(.01).all())
        self.assertTrue(remaining.valuation_price.isna().all())

    def test_action_not_held_is_not_used_to_change_the_selection(self):
        event = ledger(predecessor='Z', cash=9_999.)
        result = self.run_case(events=event)
        self.assertTrue(result.action_events.empty)
        self.assertAlmostEqual(result.summary['final_nav'], 1.1)
        self.assertEqual(set(result.trades.source_security_id), {'a','b'})

    def test_scheduled_rebalance_cancels_old_stop_after_mandatory_conversion(self):
        months = (("2017-12-29", "2018-01-02"), ("2018-01-31", "2018-02-01"))
        dates = ['2018-01-02','2018-01-31','2018-02-01','2018-02-02']
        selected = selections(months)
        selected.loc[selected.execution_date.eq('2018-02-01') & selected.source_security_id.eq('a'), ['source_security_id','intrinio_security_id']] = ['c','C']
        prices = raw_quotes(quotes(dates, (100.,100.,100.,100.), (100.,100.,100.,100.)))
        prices = prices.loc[~(prices.trading_date.ge('2018-02-01') & prices.intrinio_security_id.eq('A'))]
        prices = pd.concat([prices, successor_quotes(dates[2:], (30.,30.))], ignore_index=True)
        result = self.run_case(selections=selected, rebalances=schedule(months), prices=prices,
                               sessions=dates, terminal_date=dates[-1],
                               events=ledger(successor='C', cash=0., ratio=2., date='2018-02-01'),
                               policy=DailyAccountingPolicy(cost_bps=10, asset_stop_fraction=.1))
        self.assertEqual(result.status, 'complete_proxy')
        self.assertTrue(result.stops.empty)
        market = result.market_trades.loc[result.market_trades.date.eq(pd.Timestamp('2018-02-01')) & result.market_trades.intrinio_security_id.eq('C')]
        instructions = result.trades.loc[result.trades.date.eq(pd.Timestamp('2018-02-01')) & result.trades.intrinio_security_id.eq('C')]
        self.assertEqual(len(instructions), 2)
        self.assertEqual(len(market), 1)
        self.assertGreater(float(market.iloc[0].internally_netted_notional), .5)
        self.assertAlmostEqual(instructions.cost.sum(), market.cost.sum())
        self.assertAlmostEqual(market.iloc[0].absolute_notional, abs(instructions.signed_notional.sum()))
        # Existing converted lot has original ID a; new selected lot is c.
        self.assertEqual(set(result.action_events.source_security_id), {'a'})
        self.assertEqual(set(instructions.source_security_id), {'a','c'})

    def test_successor_can_coexist_with_opposite_side_original_lot(self):
        prices = self.event_prices(successor=False, b=(50.,50.))
        # B is already the selected short. Conversion gives a separate long B lot.
        result = self.run_case(prices=prices, events=ledger(successor='B', cash=0., ratio=2.),
                               policy=DailyAccountingPolicy(cost_bps=10))
        self.assertEqual(result.status, 'complete_proxy')
        last = result.market_trades.loc[result.market_trades.date.eq(pd.Timestamp('2018-01-03'))]
        self.assertEqual(len(last), 1)
        self.assertAlmostEqual(last.iloc[0].absolute_notional, 0.)
        self.assertAlmostEqual(last.iloc[0].internally_netted_notional, 1.)
        self.assertAlmostEqual(result.summary['final_nav'], .999)
        self.assertAlmostEqual(result.trades.cost.sum(), .001)

    def test_inputs_remain_bytewise_equal_after_entitlements(self):
        prices = self.event_prices(successor=True)
        selected = selections()
        events = pd.DataFrame(ledger(successor='C', cash=0., ratio=2.))
        copies = [frame.copy(deep=True) for frame in (prices,selected,events)]
        self.run_case(prices=prices, selections=selected, events=events)
        for original, frozen in zip((prices,selected,events), copies):
            pd.testing.assert_frame_equal(original, frozen)

    def test_future_price_date_guard_precedes_event_payout_conversion(self):
        prices = self.event_prices()
        prices.loc[prices.index[-1], 'trading_date'] = '2020-01-02'
        with self.assertRaisesRegex(DataContractError, 'outside authorized'):
            self.run_case(prices=prices, events=ledger())


    def test_retired_new_order_does_not_receive_windfall_entitlement(self):
        dates = ['2018-01-02','2018-01-31','2018-02-01','2018-02-02']
        months = (("2017-12-29", dates[0]), ("2018-01-31", dates[2]))
        selected = selections(months)
        selected.loc[selected.execution_date.eq(dates[0]) & selected.source_security_id.eq('a'), ['source_security_id','intrinio_security_id']] = ['c','C']
        prices = raw_quotes(quotes(dates, (100.,100.,100.,100.), (100.,100.,100.,100.)))
        prices = prices.loc[~(prices.trading_date.ge('2018-02-01') & prices.intrinio_security_id.eq('A'))]
        prices = pd.concat([prices, successor_quotes(dates, (100.,100.,100.,100.))], ignore_index=True)
        result = self.run_case(selections=selected, rebalances=schedule(months), prices=prices,
                               sessions=dates, terminal_date=dates[-1], events=ledger(cash=9999., date=dates[2]))
        self.assertEqual(result.status, 'complete_proxy')
        self.assertTrue(result.action_events.empty)
        self.assertAlmostEqual(result.summary['final_nav'], 1.)
        order = result.orders.loc[result.orders.date.eq(pd.Timestamp(dates[2])) & result.orders.source_security_id.eq('a')].iloc[0]
        self.assertEqual(order.status, 'unfilled_retired_security')
        self.assertAlmostEqual(order.intended_weight, .5)
        self.assertAlmostEqual(order.target_notional, .5)
        self.assertEqual(order.new_adjusted_units, 0)
        self.assertEqual(result.summary['unfilled_retired_order_count'], 1)
        febopen = result.daily.loc[result.daily.date.eq(pd.Timestamp(dates[2])) & result.daily.mark.eq('open')].iloc[0]
        self.assertEqual(febopen.long_count, 0)
        self.assertEqual(febopen.short_count, 1)
        self.assertAlmostEqual(febopen.short_market_value, -.5)
        self.assertAlmostEqual(febopen.gross_exposure, .5)

    def test_preexisting_retired_holding_gets_entitlement_new_order_does_not(self):
        dates = ['2018-01-02','2018-01-31','2018-02-01','2018-02-02']
        months = (("2017-12-29", dates[0]), ("2018-01-31", dates[2]))
        prices = raw_quotes(quotes(dates, (100.,100.,100.,100.), (100.,100.,100.,100.)))
        prices = prices.loc[~(prices.trading_date.ge('2018-02-01') & prices.intrinio_security_id.eq('A'))]
        result = self.run_case(selections=selections(months), rebalances=schedule(months), prices=prices,
                               sessions=dates, terminal_date=dates[-1], events=ledger(cash=110., date=dates[2]))
        self.assertEqual(result.status, 'complete_proxy')
        self.assertEqual(len(result.action_events), 1)
        self.assertAlmostEqual(result.action_events.signed_cash.sum(), .55)
        self.assertEqual(result.summary['unfilled_retired_order_count'], 1)
        self.assertAlmostEqual(result.summary['final_nav'], 1.05)
        febopen = result.daily.loc[result.daily.date.eq(pd.Timestamp(dates[2])) & result.daily.mark.eq('open')].iloc[0]
        self.assertEqual(febopen.long_count, 0)
        self.assertAlmostEqual(febopen.short_market_value, -.525)
        self.assertAlmostEqual(febopen.gross_exposure, .5)

    def test_non_action_source_identity_collision_remains_blocked(self):
        months = (("2017-12-29", "2018-01-02"), ("2018-01-31", "2018-02-01"))
        dates = ['2018-01-02','2018-02-01']
        selected = selections(months)
        selected.loc[selected.execution_date.eq(dates[-1]) & selected.source_security_id.eq('a'), 'source_security_id'] = 'impostor'
        with self.assertRaisesRegex(DataContractError, 'different held'):
            self.run_case(selections=selected, rebalances=schedule(months), prices=raw_quotes(quotes(dates)),
                           sessions=dates, terminal_date=dates[-1])


    def test_consecutive_stock_cash_actions_preserve_original_package_stop(self):
        dates = ['2018-01-02','2018-01-03','2018-01-04']
        prices = self.event_prices(successor=True, dates=dates, a=(100.,100.,100.), b=(100.,100.,100.), successor_open=(40.,40.))
        prices = prices.loc[~(prices.trading_date.eq(dates[-1]) & prices.intrinio_security_id.eq('C'))]
        second_successor = successor_quotes([dates[-1]], [50.]).assign(intrinio_security_id='D')
        prices = pd.concat([prices, second_successor], ignore_index=True)
        # A=$100 -> $20+2*C ($40); C -> $10+.5*D ($50).
        # Episode original long stop90 -> C threshold35 -> D threshold50.
        events = ledger(successor='C', cash=20., ratio=2.)
        events += [dict(events[0], event_id='second-action', effective_date=dates[-1], predecessor_id='C', successor_id='D', cash_per_raw_share=10., stock_ratio=.5)]
        result = self.run_case(prices=prices, events=events, sessions=dates, terminal_date=dates[-1], policy=DailyAccountingPolicy(asset_stop_fraction=.1))
        self.assertEqual(result.status, 'complete_proxy')
        self.assertEqual(len(result.action_events), 2)
        self.assertAlmostEqual(result.action_events.iloc[0].new_stop_price, 35.)
        self.assertAlmostEqual(result.action_events.iloc[1].new_stop_price, 50.)
        self.assertAlmostEqual(result.action_events.signed_cash.sum(), .2)
        self.assertEqual(len(result.stops), 1)
        self.assertEqual(result.stops.iloc[0].intrinio_security_id, 'D')
        self.assertAlmostEqual(result.summary['final_nav'], .95)

    def test_ten_slots_preserved_and_retired_new_orders_never_paid_again(self):
        dates = ['2018-01-02','2018-01-31','2018-02-01','2018-02-28','2018-03-01','2018-03-02']
        months = (("2017-12-29",dates[0]),("2018-01-31",dates[2]),("2018-02-28",dates[4]))
        selected = pd.DataFrame([dict(signal_date=signal, execution_date=date, source_security_id=f'{side}{i}',
                                     intrinio_security_id=('A' if side=='long' and i==0 else f'{side.upper()}{i}'), side=side)
                                for signal,date in months for side in ('long','short') for i in range(10)])
        selected.loc[selected.execution_date.eq(dates[0]) & selected.source_security_id.eq('long0'),'intrinio_security_id'] = 'C'
        # This is a fixed dated source mapping in the original strategy: source
        # long0 before event and the retired A entry are distinct source slots.
        selected.loc[selected.execution_date.eq(dates[0]) & selected.source_security_id.eq('long0'),'source_security_id'] = 'initial-C'
        names = set(selected.intrinio_security_id)
        prices = pd.DataFrame([dict(trading_date=date, intrinio_security_id=name, adj_open=100., adj_close=100.,
                                    adj_high=100.,adj_low=100.,open=100.,close=100.,frequency='daily',intraperiod=False,price_quality_status='accepted')
                               for date in dates for name in names if not(name=='A' and date>='2018-02-01')])
        frozen = selected.copy(deep=True)
        result = self.run_case(selections=selected, rebalances=schedule(months), prices=prices, sessions=dates,
                               terminal_date=dates[-1], events=ledger(cash=9999., date='2018-02-01'),
                               policy=DailyAccountingPolicy(required_names_per_side=10))
        pd.testing.assert_frame_equal(selected, frozen)
        self.assertEqual(result.status,'complete_proxy')
        self.assertEqual(len(result.orders),60)
        self.assertTrue(result.orders.groupby(['date','side']).size().eq(10).all())
        self.assertTrue(result.orders.intended_weight.abs().eq(.05).all())
        unfilled = result.orders.loc[result.orders.status.eq('unfilled_retired_security')]
        self.assertEqual(len(unfilled),2)
        self.assertEqual(set(unfilled.source_security_id),{'long0'})
        self.assertTrue(result.action_events.empty)
        self.assertAlmostEqual(result.summary['final_nav'],1.)
        march = result.daily.loc[result.daily.date.eq(pd.Timestamp('2018-03-01')) & result.daily.mark.eq('open')].iloc[0]
        self.assertEqual(march.long_count,9)
        self.assertEqual(march.short_count,10)
        self.assertAlmostEqual(march.long_market_value,.45)
        self.assertAlmostEqual(march.short_market_value,-.5)
        self.assertAlmostEqual(march.gross_exposure,.95)

    def test_fractional_successor_shares_retained_without_rounding(self):
        result = self.run_case(prices=self.event_prices(successor=True, b=(100.,100.)), events=ledger(successor='C',cash=0.,ratio=.2434))
        self.assertEqual(result.status,'complete_proxy')
        self.assertAlmostEqual(result.action_events.iloc[0].new_adjusted_units,.005*.2434)
        self.assertEqual(result.summary['fractional_successor_share_policy'],'theoretical_fractional_units_retained_no_cash_in_lieu_invented')

    def test_short_borrow_ends_at_cash_conversion_not_future_rebalance(self):
        dates=['2018-01-05','2018-01-08','2018-01-09']
        months=(("2017-12-29",dates[0]),)
        result=self.run_case(selections=selections(months),rebalances=schedule(months),sessions=dates,terminal_date=dates[-1],
                             prices=self.event_prices(predecessor='B',dates=dates,a=(100.,100.,100.),b=(100.,100.,100.)),
                             events=ledger(predecessor='B',cash=100.,date=dates[1]),policy=DailyAccountingPolicy(borrow_annual_bps=100))
        self.assertEqual(result.status,'complete_proxy')
        self.assertAlmostEqual(result.daily.borrow_cost.sum(),.5*.01*3/365)
        self.assertTrue(result.daily.loc[result.daily.date.eq(pd.Timestamp(dates[-1])),'borrow_cost'].eq(0).all())

    def test_same_session_action_chain_requires_ordered_terms(self):
        events=ledger(successor='C',cash=0.,ratio=1.)
        events+=[dict(events[0],event_id='second',predecessor_id='C',successor_id='D')]
        with self.assertRaisesRegex(DataContractError,'explicit ordered terms'):
            self.run_case(events=events)



if __name__ == '__main__':
    unittest.main()
