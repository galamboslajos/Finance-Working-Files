"""Behavioral accounting checks for the bounded continuous extension."""
from pathlib import Path
import numpy as np
import pandas as pd
import math,unittest
from dmv2.extension_accounting import run_fixed_forecast_accounting,prepare_fixed
from dmv2.backtest_accounting import DailyAccountingPolicy
from dmv2.universe import DataContractError


def fixture():
    days=pd.DatetimeIndex([pd.Timestamp('2020-01-31'),*pd.bdate_range('2020-02-03',periods=1568),pd.Timestamp('2026-04-30')])
    assert len(days)==1570 and pd.Timestamp('2025-06-18') in days
    sel=pd.DataFrame([dict(source_security_id=k,intrinio_security_id=k,signal_date='2020-01-31',execution_date='2020-02-03',side=s,intended_weight=w) for k,s,w in [('long','long',.5),('short','short',-.5)]])
    rb=sel[['signal_date','execution_date']].drop_duplicates()
    prices=pd.DataFrame([dict(trading_date=d,intrinio_security_id=k,open=50.,close=50.,adj_open=50.,adj_close=50.,adj_high=50.,adj_low=50.,price_accepted=True) for d in days for k in ('long','short')])
    return days,sel,rb,prices


def run(prices,days,sel,rb,**kwargs):
    return run_fixed_forecast_accounting(sel,prices,days,rebalances=rb,terminal_date='2026-04-30',**kwargs)


def test_cash_takeover_removes_short_with_signed_liability_and_no_invented_quote():
    days,sel,rb,p=fixture();p=p.loc[~(p.intrinio_security_id.eq('short')&p.trading_date.ge('2025-06-18'))]
    event=dict(event_id='cash',effective_date='2025-06-18',predecessor_id='short',successor_id=None,cash_per_raw_share=55.,stock_ratio=0.,verified=True,evidence_reference='test legal entitlement')
    result=run(p,days,sel,rb,events=[event]);assert result.status=='complete_proxy'
    assert math.isclose(result.action_events.signed_cash.iloc[0],-.55,rel_tol=1e-12)
    assert math.isclose(result.daily.nav.iloc[-1],.95,rel_tol=1e-12)
    assert len(result.trades)==2 and not (result.trades.date==pd.Timestamp('2026-04-30')).any()


def test_intraday_short_stop_and_gap_long_stop_keep_slots_in_cash():
    days,sel,rb,p=fixture()
    p.loc[p.trading_date.eq('2020-02-03')&p.intrinio_security_id.eq('short'),'adj_high']=56.
    take=p.trading_date.eq('2020-02-04')&p.intrinio_security_id.eq('long')
    p.loc[take,['open','close','adj_open','adj_close','adj_high','adj_low']]=44.
    result=run(p,days,sel,rb,policy=DailyAccountingPolicy(asset_stop_fraction=.1))
    assert result.status=='complete_proxy' and len(result.stops)==2
    assert set(result.stops.fill_type)=={'idealized_intraday_threshold','gap_open'}
    assert math.isclose(result.daily.nav.iloc[-1],.89,rel_tol=1e-12)
    assert len(result.trades)==4


def test_bad_required_quote_withholds_without_dropping_holding():
    days,sel,rb,p=fixture();p=p.loc[~(p.trading_date.eq('2024-01-02')&p.intrinio_security_id.eq('short'))]
    result=run(p,days,sel,rb);assert result.status=='failed_closed'
    assert result.failures.reason.iloc[0]=='missing_quote'
    assert pd.isna(result.daily.nav.iloc[-1])


def test_outside_explicit_calendar_is_rejected():
    days,sel,rb,p=fixture()
    with unittest.TestCase().assertRaises(DataContractError):prepare_fixed(sel,p,[*days,pd.Timestamp('2026-05-01')],rb,'2026-05-01','next_open')


def test_only_parent_calendar_guard_changes_financial_equations():
    root=Path(__file__).resolve().parents[1]
    old=(root/'src/dmv2/fixed_forecast_accounting.py').read_text();new=(root/'src/dmv2/extension_accounting.py').read_text()
    old=old[old.index('from math import'):];new=new[new.index('from math import'):]
    old=old.replace("terminal>=pd.Timestamp('2024-01-01')", "terminal>pd.Timestamp('2026-04-30') or sessions[0]!=pd.Timestamp('2020-01-31') or terminal!=pd.Timestamp('2026-04-30') or len(sessions)!=1570")
    assert old==new


def load_tests(loader,tests,pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(v) for k,v in sorted(globals().items()) if k.startswith("test_") and callable(v))
