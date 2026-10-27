import importlib.util,sys,unittest
from pathlib import Path
import numpy as np,pandas as pd
from dmv2.fixed_forecast_accounting import run_fixed_forecast_accounting
from dmv2.backtest_accounting import DailyAccountingPolicy
from dmv2.monthly_merger_accounting import replay_monthly_mergers
from dmv2.universe import DataContractError

HERE=Path(__file__).resolve().parents[1]
sp=importlib.util.spec_from_file_location('_fixed_independent_fixture',HERE/'scripts/verify_fixed_forecast_reconciliation.py')
iv=importlib.util.module_from_spec(sp);sys.modules[sp.name]=iv;sp.loader.exec_module(iv)


def fixture(clock='signal_close',dates=('2020-01-31','2020-02-03','2020-02-04')):
    records=[]
    for j in range(20):
        records.append(dict(source_security_id='s'+str(j),intrinio_security_id='n'+str(j),signal_date='2020-01-31',
         outcome_month='2020-02',outcome_start='2020-02-03',outcome_end=dates[-1],side='long' if j<10 else 'short',
         intended_weight=.05 if j<10 else -.05))
    sel=pd.DataFrame(records);sel['execution_date']=sel.signal_date if clock=='signal_close' else sel.outcome_start
    rows=[]
    for d,n in ((d,'n'+str(j)) for d in dates for j in range(21)):
        rows.append(dict(trading_date=d,intrinio_security_id=n,open=100.,close=100.,adj_open=100.,adj_close=100.,
          adj_high=100.,adj_low=100.,price_accepted=True))
    return sel,pd.DataFrame(rows),list(dates)


def event(native='n0',successor=None,cash=120.,ratio=0.,date='2020-02-04'):
    return dict(event_id='reviewed-'+native,predecessor_id=native,successor_id=successor,effective_date=date,
      cash_per_raw_share=cash,stock_ratio=ratio,verified=True,evidence_reference='synthetic reviewed legal terms')


def run(sel,prices,days,clock='signal_close',cost=0,stop=None,unit='open',events=()):
    return run_fixed_forecast_accounting(sel,prices,days,rebalances=sel[['signal_date','execution_date']].drop_duplicates(),
      terminal_date=days[-1],clock=clock,successor_factor_mark=unit,events=events,
      policy=DailyAccountingPolicy(cost_bps=cost,asset_stop_fraction=stop))


def verify(sel,prices,days,result,clock='signal_close',cost=0,stop=None,unit='open',events=()):
    return iv.oracle(sel,prices,days,list(events),[],[],dict(clock=clock,unit=unit,cost=cost,stop=stop),result.tables)


class FixedForecastTests(unittest.TestCase):
    def test_flat_close_retains_terminal_holdings(self):
        s,p,d=fixture();v=run(s,p,d)
        self.assertEqual(v.status,'complete_proxy');self.assertAlmostEqual(v.summary['final_nav'],1.)
        h=v.holdings.loc[v.holdings.mark.eq('terminal_close')]
        self.assertEqual(h.units.ne(0).sum(),20);self.assertNotIn('terminal_liquidation',v.trades.phase.tolist())
        self.assertEqual(verify(s,p,d,v)['verified_marks'],6)

    def test_close_gap_and_open_gap_are_distinct(self):
        s,p,d=fixture();mask=p.trading_date.ne('2020-01-31')&p.intrinio_security_id.isin(['n'+str(i) for i in range(10)])
        p.loc[mask,['open','close','adj_open','adj_close','adj_high','adj_low']]=110.
        c=run(s,p,d);self.assertAlmostEqual(c.summary['final_nav'],1.05)
        s['execution_date']=s.outcome_start;o=run(s,p,d,clock='next_open');self.assertAlmostEqual(o.summary['final_nav'],1.)
        verify(s,p,d,o,clock='next_open')

    def test_close_cost_matches_monthly_reference(self):
        days=pd.bdate_range('2020-01-31','2020-03-03').strftime('%Y-%m-%d').tolist()
        s,p,d=fixture(dates=days)
        second=s.copy();second['signal_date']='2020-02-28';second['outcome_month']='2020-03';second['outcome_start']='2020-03-02';second['outcome_end']='2020-03-03';second['execution_date']=second.signal_date
        s=pd.concat([s,second],ignore_index=True)
        for j,date in enumerate(days):
            mask=p.trading_date.eq(date)&p.intrinio_security_id.isin(['n'+str(i) for i in range(10)])
            p.loc[mask,['open','close','adj_open','adj_close','adj_high','adj_low']]=100+j
        for cost in (0,10):
            v=run(s,p,d,cost=cost)
            end={'2020-02':'2020-02-28','2020-03':'2020-03-03'}
            outcomes=[]
            for row in s.to_dict('records'):
                a=p.loc[p.trading_date.eq(row['signal_date'])&p.intrinio_security_id.eq(row['intrinio_security_id']),'adj_close'].iloc[0]
                b=p.loc[p.trading_date.eq(end[row['outcome_month']])&p.intrinio_security_id.eq(row['intrinio_security_id']),'adj_close'].iloc[0]
                outcomes.append(dict(source_security_id=row['source_security_id'],signal_date=row['signal_date'],label_return=b/a-1))
            parent=replay_monthly_mergers(s,pd.DataFrame(outcomes),['2020-02','2020-03'],cost,[])['monthly']
            for row in parent.to_dict('records'):
                r=v.daily.loc[v.daily.date.eq(pd.Timestamp(end[row['outcome_month']]))&v.daily.mark.ne('open')].iloc[0]
                got=r.nav_before_cost if bool(r.rebalance) else r.nav
                self.assertAlmostEqual(got,row['nav_end'],places=12)
            verify(s,p,d,v,cost=cost)

    def test_intraday_stops_apply_and_stay_cash(self):
        s,p,d=fixture('next_open');mask=p.trading_date.eq('2020-02-03')&p.intrinio_security_id.isin(['n'+str(i) for i in range(10)])
        p.loc[mask,'adj_low']=89.
        v=run(s,p,d,clock='next_open',cost=10,stop=.1)
        self.assertEqual(v.summary['stop_count'],10);self.assertAlmostEqual(v.summary['final_nav'],.94855)
        verify(s,p,d,v,clock='next_open',cost=10,stop=.1)

    def test_gap_stop_fills_at_gap_open(self):
        s,p,d=fixture('next_open');mask=p.trading_date.eq('2020-02-04')&p.intrinio_security_id.isin(['n'+str(i) for i in range(10)])
        p.loc[mask,['open','close','adj_open','adj_close','adj_high','adj_low']]=80.
        v=run(s,p,d,clock='next_open',stop=.1);self.assertAlmostEqual(v.summary['final_nav'],.9)
        self.assertTrue(v.stops.fill_type.eq('gap_open').all())
        verify(s,p,d,v,clock='next_open',stop=.1)

    def test_retired_order_remains_cash_without_payout(self):
        s,p,d=fixture('next_open');e=event(date='2020-02-03');p=p.loc[~(p.trading_date.ge('2020-02-03')&p.intrinio_security_id.eq('n0'))]
        v=run(s,p,d,clock='next_open',events=[e]);self.assertEqual(v.status,'complete_proxy')
        self.assertEqual(v.summary['unfilled_retired_order_count'],1);self.assertTrue(v.action_events.empty)
        self.assertAlmostEqual(v.summary['final_nav'],1.)
        self.assertAlmostEqual(v.daily.iloc[-1].long_market_value,.45)
        verify(s,p,d,v,clock='next_open',events=[e])

    def test_held_cash_entitlement_is_signed_and_has_no_market_fee(self):
        for native,sign in [('n0',1),('n10',-1)]:
            s,p,d=fixture();e=event(native);p=p.loc[~(p.trading_date.eq('2020-02-04')&p.intrinio_security_id.eq(native))]
            v=run(s,p,d,cost=10,events=[e]);self.assertEqual(v.status,'complete_proxy')
            self.assertAlmostEqual(v.action_events.signed_cash.iloc[0],sign*.06)
            self.assertEqual(len(v.trades),20)
            self.assertAlmostEqual(v.summary['final_nav'],1+sign*.01-.001)
            verify(s,p,d,v,cost=10,events=[e])

    def test_successor_unit_control_and_signed_conversion(self):
        for unit in ('open','close'):
            s,p,d=fixture();p.loc[p.intrinio_security_id.eq('n0'),['adj_open','adj_close','adj_high','adj_low']]=50.
            mask=p.trading_date.eq('2020-02-04')&p.intrinio_security_id.eq('n20')
            p.loc[mask,['adj_open','adj_low']]=40.;p.loc[mask,['adj_close','adj_high']]=50.
            p=p.loc[~(p.trading_date.eq('2020-02-04')&p.intrinio_security_id.eq('n0'))]
            e=event(successor='n20',cash=10.,ratio=2.)
            v=run(s,p,d,unit=unit,events=[e]);self.assertEqual(v.status,'complete_proxy')
            self.assertAlmostEqual(v.action_events.new_adjusted_units.iloc[0],.0025 if unit=='open' else .002)
            verify(s,p,d,v,unit=unit,events=[e])

    def test_missing_quote_retains_exposure_and_withholds(self):
        s,p,d=fixture();p=p.loc[~(p.trading_date.eq('2020-02-04')&p.intrinio_security_id.eq('n0'))]
        v=run(s,p,d);self.assertEqual(v.status,'failed_closed');self.assertIsNone(v.summary['final_nav'])
        self.assertTrue(v.holdings.loc[v.holdings.date.eq(pd.Timestamp('2020-02-04'))].units.ne(0).any())
        self.assertFalse(verify(s,p,d,v)['complete'])

    def test_all_row_future_guard(self):
        s,p,d=fixture();extra=p.iloc[[0]].copy();extra.trading_date='2024-01-02'
        with self.assertRaises(DataContractError):run(s,pd.concat([p,extra]),d)

    def test_close_clock_cannot_silently_enable_stops(self):
        s,p,d=fixture()
        with self.assertRaises(DataContractError):run(s,p,d,stop=.1)

    def test_source_collision_and_schedule_guards(self):
        s,p,d=fixture();bad=s.copy();bad.loc[0,'intrinio_security_id']=bad.loc[1,'intrinio_security_id']
        with self.assertRaises(DataContractError):run(bad,p,d)
        bad=s.copy();bad['execution_date']='2020-02-03'
        with self.assertRaises(DataContractError):run(bad,p,d)

    def test_input_frames_unchanged_and_oracle_rejects_nav_tampering(self):
        s,p,d=fixture();before=s.copy(deep=True);prices=p.copy(deep=True)
        v=run(s,p,d);pd.testing.assert_frame_equal(s,before);pd.testing.assert_frame_equal(p,prices)
        v.daily.loc[v.daily.index[-1],'nav']+=.01
        with self.assertRaises(ValueError):verify(s,p,d,v)

    def test_mixed_action_preserves_package_stop(self):
        s,p,d=fixture('next_open')
        mask=p.trading_date.eq('2020-02-04')&p.intrinio_security_id.eq('n20')
        p.loc[mask,['open','close','adj_open','adj_close','adj_high']]=50.
        p.loc[mask,'adj_low']=39.
        p=p.loc[~(p.trading_date.eq('2020-02-04')&p.intrinio_security_id.eq('n0'))]
        e=event(successor='n20',cash=10.,ratio=2.)
        v=run(s,p,d,clock='next_open',stop=.1,events=[e])
        self.assertAlmostEqual(v.summary['final_nav'],.995)
        self.assertEqual(v.summary['stop_count'],1)
        self.assertAlmostEqual(v.action_events.new_stop_price.iloc[0],40.)
        verify(s,p,d,v,clock='next_open',stop=.1,events=[e])

    def test_short_gap_stop_has_unlimited_gap_room(self):
        s,p,d=fixture('next_open')
        mask=p.trading_date.eq('2020-02-04')&p.intrinio_security_id.isin(['n'+str(i) for i in range(10,20)])
        p.loc[mask,['open','close','adj_open','adj_close','adj_high','adj_low']]=125.
        v=run(s,p,d,clock='next_open',stop=.1)
        self.assertAlmostEqual(v.summary['final_nav'],.875)
        self.assertTrue(v.stops.fill_price.eq(125.).all())
        verify(s,p,d,v,clock='next_open',stop=.1)

    def test_source_quality_flag_does_not_become_an_exit(self):
        s,p,d=fixture()
        p.loc[p.trading_date.eq('2020-02-04')&p.intrinio_security_id.eq('n0'),'price_accepted']=False
        v=run(s,p,d)
        self.assertEqual(v.status,'failed_closed')
        self.assertEqual(v.failures.iloc[0].reason,'source_quality_flag')
        self.assertFalse(verify(s,p,d,v)['complete'])

    def test_missing_raw_action_basis_cannot_create_cash(self):
        s,p,d=fixture()
        p.loc[p.trading_date.eq('2020-02-03')&p.intrinio_security_id.eq('n0'),'close']=np.nan
        e=event()
        v=run(s,p,d,events=[e])
        self.assertEqual(v.status,'failed_closed')
        self.assertTrue(v.action_events.empty)
        self.assertFalse(verify(s,p,d,v,events=[e])['complete'])

    def test_successor_can_net_with_an_existing_selected_native(self):
        s,p,d=fixture();e=event(successor='n1',cash=0.,ratio=1.)
        p=p.loc[~(p.trading_date.eq('2020-02-04')&p.intrinio_security_id.eq('n0'))]
        v=run(s,p,d,cost=10,events=[e])
        self.assertEqual(v.status,'complete_proxy')
        self.assertAlmostEqual(v.summary['final_nav'],.999)
        verify(s,p,d,v,cost=10,events=[e])

    def test_controller_has_only_zero_fit_phases(self):
        import ast
        code=(HERE/'scripts/run_fixed_forecast_reconciliation.py').read_text()
        tree=ast.parse(code)
        values=[ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PHASES' for t in n.targets)]
        self.assertEqual(values,[('prepare','verify_inputs','replay','verify_replay')])
        self.assertIn('Model import forbidden',code)


if __name__=='__main__':unittest.main()
