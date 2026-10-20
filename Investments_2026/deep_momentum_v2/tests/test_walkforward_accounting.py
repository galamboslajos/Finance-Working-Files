"""Structural clocks, continuous year boundaries, signed actions and preflight."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from dmv2.action_accounting import run_action_accounting
from dmv2.backtest_accounting import DailyAccountingPolicy
from dmv2.backtest_loading import _pin
from dmv2.walkforward_accounting import run_walkforward_accounting
from dmv2.walkforward_backtest import _configuration, _price_plan, yearly_breakdown
from dmv2.walkforward_clock import walkforward_clock
from dmv2.walkforward_preflight import preflight_selected_horizons, validate_current_events
from dmv2.walkforward_selection import select_walkforward_basket, validate_walkforward_basket
from dmv2.universe import DataContractError


def configuration():
    return {'protocol_id':'dmv2-walkforward-2020-2023-v1',
        'execution_restrictions':{'native_ids':{'sec_gjP4xM':'fixed Dean Foods','sec_zmYNnz':'fixed Rowan'},
            'period':'all_2020_2023_signals_before_selection','valid_historical_training_rows_retained':True,
            'prediction_rows_and_probabilities_retained':True},
        'backtest':{'start':'2020-01-01','end':'2023-12-31','top_k_per_side':10,
            'baskets':['class_1_10','class_2_9'],'long_weight':.5,'short_weight':-.5,
            'arms':['rf_paper15','xgb_paper15','rf_extended','xgb_extended'], 'cost_bps':[0,5,10,25],
            'variants':{'unstopped':None,'asset_stop_10pct':.1},
            'basket_breadth':{'policy':'balanced_available_capped_10',
                'names_per_side':'min(10,eligible_mode_long_count,eligible_mode_short_count)',
                'equal_side_name_counts':True,'long_total_weight':.5,'short_total_weight':-.5,
                'zero_candidate_side':'withhold_full_path',
                'candidate_information':'signal_time_forecasts_and_fixed_execution_restrictions_only',
                'rank_and_ties':'unchanged_class_probability_descending_then_source_id_ascending'},
            'RET':'not_evaluated_in_this_class_basket_experiment'},
        'execution':{'financial_2020_2023_analysis_authorized':True,'financial_2024_plus_analysis':False,
            'cloud_mutations':False,'backtest_maximum_wall_seconds':900,'backtest_maximum_peak_rss_bytes':2*1024**3,
            'maximum_output_bytes':12*1024**3},
        'accounting':{'continuous_2020_2023_ledger':True,'annual_fit_forced_liquidation':False,
            'terminal_session':'2023-12-29','new_postselection_hindsight_exclusions':False,
            'borrow_annual_bps':0,'funding_annual_bps':0}}


def predictions(nlong=9,nshort=37,classes=(1,10)):
    result=[]
    for side,n,mode in [('long',nlong,classes[0]),('short',nshort,classes[1])]:
        for i in range(n):
            prob=.8-i/1000
            result.append({'source_security_id':f'{side}-{i:03d}','signal_date':'2019-12-31',
                'intrinio_security_id':f'native-{side}-{i:03d}','prediction_eligible':True,
                'paper_prediction_eligible':True,'prediction_status':'predicted',
                **{f'probability_{k}':prob if k==mode else (1-prob)/9 for k in range(1,11)}})
    return pd.DataFrame(result)


def selection(months=(('2019-12-31','2020-01-02'),)):
    return pd.DataFrame([dict(signal_date=signal,execution_date=entry,source_security_id=source,
        intrinio_security_id=native,side=side) for signal,entry in months
        for source,native,side in [('a','A','long'),('b','B','short')]])


def schedule(months=(('2019-12-31','2020-01-02'),)):
    return pd.DataFrame(months,columns=['signal_date','execution_date'])


def prices(dates=('2020-01-02','2020-01-03'),a=(100.,110.),b=(100.,100.)):
    return pd.DataFrame([dict(trading_date=day,intrinio_security_id=native,open=value,close=value,
        adj_open=value,adj_close=value,adj_high=value,adj_low=value,frequency='daily',intraperiod=False,
        price_quality_status='accepted') for i,day in enumerate(dates) for native,value in [('A',a[i]),('B',b[i])]])


def event(predecessor='A',cash=110.,successor=None,ratio=0.,date='2020-01-03'):
    return dict(event_id='synthetic-retirement',predecessor_id=predecessor,effective_date=date,
        cash_per_raw_share=cash,stock_ratio=ratio,successor_id=successor,verified=True,
        evidence_reference='https://www.sec.gov/synthetic-test-evidence')


class WalkforwardStructuralTests(unittest.TestCase):
    def test_clock_exact_four_year_grid_and_last_authorized_close(self):
        sessions,windows,clock=walkforward_clock()
        self.assertEqual(len(sessions),1006); self.assertEqual(len(windows),48)
        self.assertEqual(sessions[0],'2020-01-02'); self.assertEqual(sessions[-1],'2023-12-29')
        self.assertEqual(windows.iloc[0].signal_date,'2019-12-31')
        self.assertEqual(windows.iloc[-1].signal_date,'2023-11-30')
        self.assertEqual(windows.iloc[-1].exit_price_field,'adj_close')
        self.assertFalse(clock['annual_liquidation'])

    def test_balanced_nine_and_nine_preserves_rank_and_target_independence(self):
        _,_,clock=walkforward_clock(); source=predictions()
        result=select_walkforward_basket(source,basket='class_1_10',sessions=clock['selection_sessions'],configuration=configuration())
        self.assertEqual(result.selections.groupby('side').size().to_dict(),{'long':9,'short':9})
        self.assertAlmostEqual(result.selections.query("side=='long'").intended_weight.sum(),.5)
        validate_walkforward_basket(result,source,basket='class_1_10',sessions=clock['selection_sessions'],configuration=configuration())
        source['target_return']=np.nan; source['future_quote_complete']=False
        changed=select_walkforward_basket(source,basket='class_1_10',sessions=clock['selection_sessions'],configuration=configuration())
        pd.testing.assert_frame_equal(result.selections,changed.selections)

    def test_fixed_native_exclusions_retained_as_prediction_rows_and_separate_pairs(self):
        _,_,clock=walkforward_clock(); source=predictions(10,38,(2,9))
        source.loc[0,'intrinio_security_id']='sec_gjP4xM'; source.loc[10,'intrinio_security_id']='sec_zmYNnz'
        result=select_walkforward_basket(source,basket='class_2_9',sessions=clock['selection_sessions'],configuration=configuration())
        self.assertEqual(len(result.ledger),48); self.assertTrue(result.ledger.prediction_eligible.all())
        self.assertEqual(result.ledger.execution_restricted.sum(),2)
        self.assertEqual(set(result.selections.predicted_class),{2,9})
        self.assertFalse(result.selections.execution_restricted.any())

    def test_numerical_parity_with_old_frozen_engine_on_distinct_structural_dates(self):
        new=run_walkforward_accounting(selection(),prices(),['2020-01-02','2020-01-03'],
            rebalances=schedule(),terminal_date='2020-01-03',policy=DailyAccountingPolicy(cost_bps=10))
        old_months=(('2017-12-29','2018-01-02'),)
        old=run_action_accounting(selection(old_months),prices(['2018-01-02','2018-01-03']),
            ['2018-01-02','2018-01-03'],rebalances=schedule(old_months),terminal_date='2018-01-03',
            policy=DailyAccountingPolicy(cost_bps=10))
        for name in ['nav','price_pnl','trading_cost','traded_notional','turnover_one_way','cash']:
            np.testing.assert_allclose(new.daily[name],old.daily[name],rtol=0,atol=0)
        self.assertAlmostEqual(new.summary['final_nav'],old.summary['final_nav'])

    def test_year_boundary_keeps_nav_overnight_move_and_only_regular_monthly_stop_reset(self):
        months=(('2020-11-30','2020-12-01'),('2020-12-31','2021-01-04'))
        days=['2020-12-01','2020-12-31','2021-01-04','2021-01-05']
        result=run_walkforward_accounting(selection(months),prices(days,[100,110,121,121],[100,90,81,81]),days,
            rebalances=schedule(months),terminal_date=days[-1],policy=DailyAccountingPolicy(asset_stop_fraction=.1))
        self.assertEqual(result.status,'complete_proxy'); self.assertAlmostEqual(result.summary['final_nav'],1.2)
        self.assertFalse(((result.trades.date==pd.Timestamp('2020-12-31'))&result.trades.phase.eq('terminal_liquidation')).any())
        before=result.holdings.query("source_security_id=='a' and date=='2020-12-31'")
        self.assertTrue(before.stop_price.eq(90.).all())
        after=result.holdings.query("source_security_id=='a' and date=='2021-01-04'")
        np.testing.assert_allclose(after.stop_price,121*.9)
        annual=yearly_breakdown(result,arm='synthetic',basket='class_1_10',variant='asset_stop_10pct',cost_bps=0)
        self.assertAlmostEqual(np.prod(1+annual.total_return),1.2)
        self.assertAlmostEqual(annual.iloc[1].initial_nav,1.1)

    def test_verified_current_cash_short_liability_and_historical_new_order_no_entitlement(self):
        p=prices(); p=p[~(p.intrinio_security_id.eq('B')&p.trading_date.eq('2020-01-03'))]
        result=run_walkforward_accounting(selection(),p,['2020-01-02','2020-01-03'],rebalances=schedule(),
            terminal_date='2020-01-03',events=[event(predecessor='B',cash=120.)])
        self.assertEqual(result.status,'complete_proxy'); self.assertAlmostEqual(result.action_events.iloc[0].signed_cash,-.6)
        historical=event(date='2019-12-20'); p=prices(); p=p[p.intrinio_security_id.eq('B')]
        result=run_walkforward_accounting(selection(),p,['2020-01-02','2020-01-03'],rebalances=schedule(),
            terminal_date='2020-01-03',historical_retirements=[historical])
        self.assertEqual(result.status,'complete_proxy'); self.assertTrue(result.action_events.empty)
        self.assertAlmostEqual(result.summary['final_nav'],1.)
        self.assertEqual(result.orders.iloc[0].status,'unfilled_retired_security')

    def test_protected_quote_date_rejected_even_if_unheld_before_monetary_conversion(self):
        p=prices(); row=p.iloc[0].copy(); row.trading_date='2024-01-02'; row.intrinio_security_id='unheld'
        p=pd.concat([p,row.to_frame().T],ignore_index=True)
        with self.assertRaises(DataContractError):
            run_walkforward_accounting(selection(),p,['2020-01-02','2020-01-03'],
                rebalances=schedule(),terminal_date='2020-01-03')

    def test_preflight_lists_entire_missing_horizon_before_any_performance(self):
        p=prices(); meta=p[['trading_date','intrinio_security_id','frequency','intraperiod','price_quality_status']]
        meta=meta[~(meta.intrinio_security_id.eq('A')&meta.trading_date.eq('2020-01-03'))]
        result=preflight_selected_horizons(selection(),schedule(),meta,['2020-01-02','2020-01-03'])
        self.assertEqual(result['summary']['unique_missing_keys'],1)
        self.assertFalse(result['summary']['ready_for_accounting'])
        self.assertFalse(result['summary']['prices_or_outcomes_read'])
        self.assertFalse(result['summary']['selection_changed'])
        fixed=preflight_selected_horizons(selection(),schedule(),meta,['2020-01-02','2020-01-03'],events=[event()])
        self.assertTrue(fixed['summary']['ready_for_accounting'])
        self.assertEqual(len(fixed['action_requirements']),1)

    def test_preflight_quality_failures_and_financial_column_refusal(self):
        p=prices(); meta=p[['trading_date','intrinio_security_id','frequency','intraperiod','price_quality_status']].copy()
        meta.loc[0,'price_quality_status']='rejected'
        result=preflight_selected_horizons(selection(),schedule(),meta,['2020-01-02','2020-01-03'])
        self.assertEqual(result['summary']['unique_quality_flag_keys'],1)
        with self.assertRaises(DataContractError):
            preflight_selected_horizons(selection(),schedule(),p,['2020-01-02','2020-01-03'])

    def test_unsupported_spin_off_and_same_day_action_chain_fail_closed(self):
        item=event(); item['action_type']='spinoff_multiple_assets'
        with self.assertRaises(DataContractError): validate_current_events([item],['2020-01-02','2020-01-03'])
        one=event(successor='C',ratio=1.); two=event(predecessor='C',successor='D',ratio=1.); two['event_id']='second'
        with self.assertRaises(DataContractError): validate_current_events([one,two],['2020-01-02','2020-01-03'])

    def test_runner_does_not_plan_financial_projection_for_wrong_year_keys(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); run=root/'new'; directory=run/'years/2020'; directory.mkdir(parents=True)
            path=directory/'daily_prices_raw.parquet'; pq.write_table(pa.table({'trading_date':['2024-01-02'],'intrinio_security_id':['A']}),path)
            (directory/'audit.json').write_text(__import__('json').dumps({'status':'COMPLETE','year':2020,
                'build_identity_sha256':'identity','artifacts':{'daily_prices_raw':_pin(directory,path)}}))
            with patch('dmv2.walkforward_backtest._projected_bytes',side_effect=AssertionError('Financial projection before date guard')):
                with self.assertRaises(DataContractError): _price_plan(root,run,'identity',[])

    def test_configuration_refuses_yearly_liquidation_future_scope_and_new_exclusions(self):
        for field in ['annual_liquidation','future','exclusion']:
            cfg=deepcopy(configuration())
            if field=='annual_liquidation': cfg['accounting']['annual_fit_forced_liquidation']=True
            elif field=='future': cfg['backtest']['end']='2024-12-31'
            else: cfg['execution_restrictions']['native_ids']['new-hindsight-drop']='unapproved'
            with self.subTest(field=field),self.assertRaises(DataContractError): _configuration(cfg)


if __name__=='__main__': unittest.main()
