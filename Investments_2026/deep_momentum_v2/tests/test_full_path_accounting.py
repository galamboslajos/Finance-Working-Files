"""Fabricated stock paths and frozen-engine parity; no research inputs."""
from copy import deepcopy
from math import isclose
import unittest
from unittest.mock import patch

import pandas as pd
import pyarrow as pa

from dmv2 import full_path_accounting as fast
from dmv2 import path_target_audit as old
from dmv2.universe import DataContractError


def fixture():
    sessions, windows, _ = old._clock()
    window = windows[0]
    dates = sorted({r['date'] for r in window['required_marks']})
    prices = pd.DataFrame([{'trading_date':d,'intrinio_security_id':'native',
        'open':100.,'close':100.,'adj_open':100.,'adj_high':100.,'adj_low':100.,'adj_close':100.,
        'frequency':'daily','intraperiod':False,'price_quality_status':'accepted',
        'raw_ohlc_bounds_valid':True,'adjusted_ohlc_bounds_valid':True} for d in dates])
    return prices, window, sessions


def call(prices, window, sessions, side='long', variant='unstopped', events=(), historical=()):
    index = fast.prepare_price_index(prices, sessions)
    return fast.replay_unit_episode(index,source_security_id='source',intrinio_security_id='native',
        signal_date=window['signal_date'],window=window,side=side,path_variant=variant,
        events=events,historical_retirements=historical)


class FullPathAccounting(unittest.TestCase):
    def setUp(self): self.prices,self.window,self.sessions=fixture()
    def set_quote(self, date, opening, high, low, close):
        at=self.prices.trading_date.eq(date)
        for field,value in {'open':opening,'close':close,'adj_open':opening,'adj_high':high,'adj_low':low,'adj_close':close}.items():
            self.prices.loc[at,field]=value
    def parity(self, events=(), historical=()):
        for side in ('long','short'):
            for variant in ('unstopped','asset_stop_10pct'):
                original=old.replay_asset_episode(self.prices,source_security_id='source',intrinio_security_id='native',
                    signal_date=self.window['signal_date'],window=self.window,side=side,path_variant=variant,
                    events=events,historical_retirements=historical)
                result=call(self.prices,self.window,self.sessions,side,variant,events,historical)
                for field in ('status','reason','stop_count','action_count','unfilled_entry','engine_status','engine_capital_gate_triggered'):
                    self.assertEqual(result[field],original[field],(side,variant,field,result,original))
                self.assertEqual([(r['date'],r['mark']) for r in result['marks']],[(r['date'],r['mark']) for r in original['marks']])
                for new,prior in zip(result['marks'],original['marks']):
                    self.assertTrue(isclose(new['wealth'],prior['wealth'],rel_tol=0.,abs_tol=1e-12),(side,variant,new,prior))
    def event(self, ratio=0., cash=120., successor=None):
        return {'event_id':'event','predecessor_id':'native','effective_date':'2020-01-10','successor_id':successor,
            'cash_per_raw_share':cash,'stock_ratio':ratio,'verified':True,'evidence_reference':'fabricated-primary'}
    def test_flat_and_ordinary_path_parity(self):
        self.parity()
        self.set_quote('2020-01-06',101,105,96,103)
        self.set_quote(self.window['exit_date'],110,110,110,110)
        self.parity()
    def test_exit_uses_exact_pretrade_sum_before_cash_only_rebalance(self):
        self.set_quote(self.window['exit_date'],20.,20.,20.,20.)
        original=old.replay_asset_episode(self.prices,source_security_id='source',intrinio_security_id='native',
            signal_date=self.window['signal_date'],window=self.window,side='long',path_variant='unstopped')
        result=call(self.prices,self.window,self.sessions)
        self.assertEqual(result['marks'],original['marks'])
    def test_gap_stop_parity_both_directions(self):
        for price in (85.,115.):
            self.set_quote('2020-01-06',price,price+2,price-2,price+1)
            self.parity()
    def test_intraday_stop_parity(self):
        self.set_quote('2020-01-06',100,115,85,102)
        self.parity()
    def test_stop_then_cash_needs_no_later_asset_quotes(self):
        self.set_quote('2020-01-06',85,87,83,86)
        self.prices=self.prices.loc[self.prices.trading_date<='2020-01-06']
        self.parity()
    def test_missing_price_and_source_quality_parity(self):
        self.prices=self.prices.loc[self.prices.trading_date!='2020-01-06']
        self.parity()
        self.prices,self.window,self.sessions=fixture()
        self.prices.loc[self.prices.trading_date=='2020-01-06','price_quality_status']='unknown'
        self.parity()
    def test_invalid_ohlc_and_boolean_price_parity(self):
        self.set_quote('2020-01-06',100,95,90,100)
        self.parity()
    def test_negative_unit_wealth_and_legacy_capital_gate_parity(self):
        for price in (250.,350.):
            self.set_quote(self.window['exit_date'],price,price,price,price)
            self.parity()
        result=call(self.prices,self.window,self.sessions,'short')
        self.assertTrue(result['engine_capital_gate_triggered'])
        self.assertEqual(result['status'],'unavailable')
    def test_explicit_signed_unit_domain_retains_extreme_short_loss(self):
        self.set_quote('2020-01-06',400,400,400,400)
        self.set_quote(self.window['exit_date'],400,400,400,400)
        index=fast.prepare_price_index(self.prices,self.sessions)
        result=fast.replay_unit_episode(index,source_security_id='source',intrinio_security_id='native',
            signal_date=self.window['signal_date'],window=self.window,side='short',path_variant='unstopped',
            retain_signed_wealth=True)
        self.assertEqual(result['status'],'complete')
        self.assertAlmostEqual(result['marks'][-1]['wealth']-1.,-3.)
        self.assertTrue(result['legacy_capital_gate_would_trigger'])
        self.assertGreater(result['negative_reference_nav_retained'],0)
        self.assertFalse(result['engine_capital_gate_triggered'])
        self.assertIn('signed_unit_domain',result['arithmetic_version'])
    def test_cash_and_stock_entitlement_parity(self):
        self.prices=self.prices.loc[self.prices.trading_date<'2020-01-10']
        self.parity([self.event()])
        self.prices,self.window,self.sessions=fixture()
        successor=self.prices.loc[self.prices.trading_date>='2020-01-10'].copy()
        successor['intrinio_security_id']='successor'
        for field in ('open','close','adj_open','adj_high','adj_low','adj_close'):successor[field]=220.
        self.prices=pd.concat([self.prices.loc[self.prices.trading_date<'2020-01-10'],successor],ignore_index=True)
        self.parity([self.event(ratio=.5,cash=0.,successor='successor')])
    def test_mixed_adjusted_raw_entitlement_and_carried_stop_parity(self):
        for field in ('adj_open','adj_high','adj_low','adj_close'):self.prices[field]=50.
        successor=self.prices.loc[self.prices.trading_date>='2020-01-10'].copy()
        successor['intrinio_security_id']='successor'
        for field in ('open','close'):successor[field]=200.
        for field in ('adj_open','adj_high','adj_low','adj_close'):successor[field]=80.
        self.prices=pd.concat([self.prices.loc[self.prices.trading_date<'2020-01-10'],successor],ignore_index=True)
        self.parity([self.event(ratio=.5,cash=20.,successor='successor')])
    def test_missing_successor_and_basis_parity(self):
        self.parity([self.event(ratio=.5,cash=0.,successor='missing')])
        self.prices.loc[self.prices.trading_date=='2020-01-09','close']=float('nan')
        self.parity([self.event()])
    def test_retired_entry_not_zero_return_parity(self):
        event=self.event();event['effective_date']=self.window['entry_date']
        self.parity([event])
    def test_exit_close_is_not_needed(self):
        self.prices.loc[self.prices.trading_date==self.window['exit_date'],'adj_close']=float('nan')
        self.parity()
    def test_generic_2010_and_protected_endpoint(self):
        days=['2009-12-31','2010-01-04','2010-01-05','2010-02-01']
        window={'signal_date':days[0],'entry_date':days[1],'exit_date':days[-1],
            'required_marks':[{'date':d,'mark':m} for d in days[1:-1] for m in ('open','close')]+[{'date':days[-1],'mark':'open'}]}
        rows=self.prices.iloc[:3].copy();rows['trading_date']=days[1:]
        result=call(rows,window,days)
        self.assertEqual(result['status'],'complete')
        window['exit_date']='2024-01-02'
        result=call(rows,window,days)
        self.assertEqual(result['reason'],'protected_scheduled_endpoint')
    def test_metadata_guard_before_price_conversion_and_duplicates(self):
        row={'trading_date':'2024-01-02','intrinio_security_id':'native','adj_open':object(),'adj_close':object()}
        with self.assertRaisesRegex(DataContractError,'Unauthorized financial'):
            fast.prepare_price_index([row],['2024-01-02'])
        rows=pd.concat([self.prices,self.prices.iloc[:1]],ignore_index=True)
        with self.assertRaisesRegex(DataContractError,'Duplicate provider'):fast.prepare_price_index(rows,self.sessions)
    def test_missing_grid_and_cash_action_rejected(self):
        window=deepcopy(self.window);window['required_marks'].pop(2)
        with self.assertRaisesRegex(DataContractError,'Missing or extra'):call(self.prices,window,self.sessions)
        event=self.event();event['predecessor_id']=fast.CASH_LONG
        with self.assertRaisesRegex(DataContractError,'synthetic cash'):call(self.prices,self.window,self.sessions,events=[event])
    def test_registry_historical_event_is_real_economic_action(self):
        prices=self.prices.copy();prices.trading_date=prices.trading_date.str.replace('2020','2019',regex=False)
        days=tuple(d.replace('2020','2019') for d in self.sessions if '2020' in d)
        # Isolate a literal old-year window; financial terms are not discarded.
        window=deepcopy(self.window)
        for field in ('signal_date','entry_date','exit_date'):window[field]=window[field].replace('2020','2019').replace('2019-12','2018-12')
        window['required_marks']=[{'date':r['date'].replace('2020','2019'),'mark':r['mark']} for r in window['required_marks']]
        event=self.event();event['effective_date']='2019-01-10'
        prices=prices.loc[prices.trading_date<'2019-01-10']
        result=call(prices,window,days,events=[event])
        self.assertEqual(result['action_count'],1)
        self.assertAlmostEqual(result['marks'][-1]['wealth'],1.2)


if __name__=='__main__':unittest.main()
