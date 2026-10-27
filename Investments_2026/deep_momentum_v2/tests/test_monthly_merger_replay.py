"""Invented prices and baskets only; no private source or estimator reads."""
from pathlib import Path
import importlib.util,unittest
import numpy as np
import pandas as pd
from dmv2.monthly_merger_accounting import merger_components,replay_monthly_mergers
from dmv2.paper15_monthly import monthly_account


class MonthlyMergerTests(unittest.TestCase):
    def test_cash_entitlement_uses_observed_raw_adjusted_units(self):
        r=merger_components(50.,100.,55.,110.,0.)
        self.assertAlmostEqual(r['predecessor_raw_shares_per_initial_capital'],.011)
        self.assertAlmostEqual(r['cash_component'],1.21);self.assertAlmostEqual(r['asset_return'],.21)
        self.assertEqual(r['successor_component'],0)

    def test_stock_factor_is_unit_conversion_not_trade_price(self):
        r=merger_components(50.,100.,50.,0.,2.,40.,20.,25.)
        self.assertAlmostEqual(r['successor_component'],1.)
        self.assertAlmostEqual(r['asset_return'],0.)

    def test_mixed_cash_and_stock_wealth(self):
        r=merger_components(100.,100.,100.,20.,2.,40.,20.,25.)
        self.assertAlmostEqual(r['cash_component'],.2);self.assertAlmostEqual(r['successor_component'],1.)
        self.assertAlmostEqual(r['asset_return'],.2)

    def test_invalid_observation_or_missing_successor_rejected(self):
        for start in (0.,np.nan,-1.):
            with self.assertRaises(ValueError):merger_components(start,100.,100.,20.,0.)
        with self.assertRaises(ValueError):merger_components(100.,100.,100.,0.,1.)
        with self.assertRaises(ValueError):merger_components(100.,100.,100.,0.,0.)

    def basket(self,successor='C'):
        rows=[]
        for signal,month,end,short in [('2021-02-26','2021-03','2021-03-31','A'),('2021-03-31','2021-04','2021-04-30',successor)]:
            for sid,side,w in [('B','long',.5),(short,'short',-.5)]:
                rows.append(dict(source_security_id=sid,signal_date=signal,outcome_month=month,outcome_end=end,
                                 intrinio_security_id=sid,side=side,intended_weight=w))
        selected=pd.DataFrame(rows);returns=selected[['source_security_id','signal_date']].copy();returns['label_return']=0.
        return selected,returns

    def correction(self,cash=1.1,stock=0.):
        return dict(event_id='A_MERGER',source_security_id='A',signal_date='2021-02-26',outcome_month='2021-03',
                    predecessor_id='A',successor_id='C' if stock else None,effective_date='2021-03-01',
                    asset_return=cash+stock-1,cash_component=cash,successor_component=stock)

    def test_no_action_matches_existing_monthly_accounting(self):
        selected,outcomes=self.basket();outcomes['label_return']=[.2,-.1,.03,.07]
        for fee in (0,5,10,25):
            old,_,_=monthly_account(selected,outcomes,['2021-03','2021-04'],fee)
            new=replay_monthly_mergers(selected,outcomes,['2021-03','2021-04'],fee,[])['monthly']
            for col in ('nav_start','nav_end','net_return','trading_cost','one_way_turnover','cash_end'):
                np.testing.assert_allclose(old[col],new[col],rtol=0,atol=1e-14)

    def test_cash_merger_short_is_signed_liability_and_cash_not_stock(self):
        selected,outcomes=self.basket();outcomes.loc[outcomes.source_security_id.eq('A'),'label_return']=np.nan
        r=replay_monthly_mergers(selected,outcomes,['2021-03','2021-04'],0,[self.correction()])
        self.assertTrue(r['failures'].empty);self.assertAlmostEqual(r['monthly'].nav_end.iloc[0],.95)
        self.assertAlmostEqual(r['actions'].signed_cash_entitlement.iloc[0],-.55)
        self.assertAlmostEqual(r['monthly'].one_way_turnover.iloc[1],.5/.95)
        self.assertTrue(r['positions'].loc[r['positions'].source_security_id.eq('A'),'predecessor_end_notional'].eq(0).all())

    def test_stock_conversion_carried_to_successor_and_net_rebalance(self):
        selected,outcomes=self.basket();outcomes.loc[outcomes.source_security_id.eq('A'),'label_return']=np.nan
        r=replay_monthly_mergers(selected,outcomes,['2021-03','2021-04'],0,[self.correction(0.,1.1)])
        self.assertAlmostEqual(r['actions'].signed_successor_end_notional.iloc[0],-.55)
        self.assertAlmostEqual(r['monthly'].one_way_turnover.iloc[1],.1/.95)
        self.assertEqual(r['actions'].mandatory_conversion_market_cost.iloc[0],0.)
        april=r['trades'].loc[r['trades'].outcome_month.eq('2021-04')]
        self.assertNotIn('A',april.intrinio_security_id.tolist())

    def test_mixed_merger_keeps_cash_and_successor_separate(self):
        selected,outcomes=self.basket();r=replay_monthly_mergers(selected,outcomes,['2021-03','2021-04'],10,[self.correction(.4,.7)])
        self.assertAlmostEqual(r['actions'].signed_cash_entitlement.iloc[0],-.2)
        self.assertAlmostEqual(r['actions'].signed_successor_end_notional.iloc[0],-.35)
        self.assertAlmostEqual(r['monthly'].nav_end.iloc[0],.949)
        self.assertAlmostEqual(r['monthly'].trading_cost.iloc[0],.001)

    def test_missing_unamended_return_remains_fail_closed(self):
        selected,outcomes=self.basket();outcomes.loc[0,'label_return']=np.nan
        r=replay_monthly_mergers(selected,outcomes,['2021-03','2021-04'],0,[self.correction()])
        self.assertEqual(r['failures'].reason.iloc[0],'unresolved_selected_monthly_return')
        self.assertTrue(r['monthly'].empty);self.assertEqual(len(selected),4)

    def test_new_order_after_effective_date_has_no_prior_holder_entitlement(self):
        selected,outcomes=self.basket();fix=self.correction();fix['effective_date']='2021-02-26'
        with self.assertRaises(ValueError):replay_monthly_mergers(selected,outcomes,['2021-03','2021-04'],0,[fix])

    def test_duplicate_case_rejected(self):
        selected,outcomes=self.basket()
        with self.assertRaises(ValueError):replay_monthly_mergers(selected,outcomes,['2021-03'],0,[self.correction(),self.correction()])

    def test_controller_has_no_fit_or_selection_phase(self):
        p=Path(__file__).parents[1]/'scripts/run_monthly_merger_replay.py'
        spec=importlib.util.spec_from_file_location('_zero_fit_controller_test',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        self.assertEqual(m.PHASES,('amendments','verify_amendments','backtest','verify_backtest'))


if __name__=='__main__':unittest.main()
