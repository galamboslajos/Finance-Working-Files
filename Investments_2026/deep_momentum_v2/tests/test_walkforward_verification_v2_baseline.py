"""Independent replay catches faults in small genuine synthetic engine results."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest
import tempfile
from unittest.mock import patch
import pyarrow.parquet as pq

import pandas as pd

from dmv2.backtest_accounting import DailyAccountingPolicy
from dmv2.walkforward_accounting import run_walkforward_accounting
from dmv2.walkforward_clock import walkforward_clock
from dmv2.walkforward_selection import select_walkforward_basket
from test_walkforward_accounting import configuration,event,prices,predictions,schedule,selection

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/verify_walkforward_backtest_v2.py'
if not SCRIPT.exists():SCRIPT=Path(__file__).resolve().parents[1]/'verify_walkforward_backtest_v2.py'
spec=importlib.util.spec_from_file_location('independent_backtest_verifier',SCRIPT)
verify=importlib.util.module_from_spec(spec); spec.loader.exec_module(verify)


def audit(result,quote,chosen,rebalance,dates,**kwargs):
    return verify.verify_scenario(result.tables,quote,chosen,rebalance,dates,**kwargs)


class IndependentBacktestVerificationTests(unittest.TestCase):
    def test_complete_cash_units_fee_stop_and_year_continuity(self):
        dates=('2020-12-31','2021-01-04','2021-01-05')
        months=(('2020-11-30','2020-12-31'),('2020-12-31','2021-01-04'))
        quote=prices(dates,a=(100.,105.,94.5),b=(100.,100.,100.))
        chosen=selection(months); rebalance=schedule(months)
        # Structural partial month is intentional; no financial source involved.
        result=run_walkforward_accounting(chosen,quote,dates,rebalances=rebalance,terminal_date=dates[-1],
            policy=DailyAccountingPolicy(cost_bps=5,required_names_per_side=None,asset_stop_fraction=.1))
        report=audit(result,quote,chosen,rebalance,dates,cost_bps=5,stop_fraction=.1)
        self.assertTrue(report['continuous_cash_units_reconciled']); self.assertEqual(report['sessions'],3)
        self.assertEqual(report['stop_count'],1)

    def test_mixed_entitlement_long_and_short_and_retired_cash_slots(self):
        dates=('2020-01-02','2020-01-03'); chosen=selection(); rebalance=schedule()
        quote=prices(dates,a=(100.,100.),b=(100.,100.))
        for predecessor in ('A','B'):
            action=event(predecessor=predecessor,cash=30.,successor='NEW',ratio=.7)
            successor=quote.loc[quote.trading_date.eq(dates[-1])&quote.intrinio_security_id.eq('A')].copy()
            successor['intrinio_security_id']='NEW'; successor['open']=50.; successor['close']=50.
            for column in ['adj_open','adj_close','adj_high','adj_low']:successor[column]=100.
            source=pd.concat([quote,successor],ignore_index=True)
            result=run_walkforward_accounting(chosen,source,dates,rebalances=rebalance,terminal_date=dates[-1],
                events=[action],policy=DailyAccountingPolicy(cost_bps=10,required_names_per_side=None,asset_stop_fraction=.1))
            report=audit(result,source,chosen,rebalance,dates,events=[action],cost_bps=10,stop_fraction=.1)
            self.assertEqual(report['action_count'],1)
            corrupt=deepcopy(result); corrupt.action_events.loc[0,'signed_cash']+=.01
            with self.assertRaisesRegex(verify.VerificationError,'Entitlement signed_cash'):
                audit(corrupt,source,chosen,rebalance,dates,events=[action],cost_bps=10,stop_fraction=.1)
        history=[dict(event(),effective_date='2019-12-30')]
        result=run_walkforward_accounting(chosen,quote,dates,rebalances=rebalance,terminal_date=dates[-1],
            historical_retirements=history,policy=DailyAccountingPolicy(cost_bps=0,required_names_per_side=None))
        report=audit(result,quote,chosen,rebalance,dates,historical_retirements=history)
        self.assertEqual(report['action_count'],0)

    def test_mutated_cash_units_fees_missing_stop_or_year_liquidation_fail(self):
        dates=('2020-01-02','2020-01-03'); chosen=selection(); rebalance=schedule()
        quote=prices(dates,a=(100.,88.),b=(100.,100.))
        result=run_walkforward_accounting(chosen,quote,dates,rebalances=rebalance,terminal_date=dates[-1],
            policy=DailyAccountingPolicy(cost_bps=5,required_names_per_side=None,asset_stop_fraction=.1))
        audit(result,quote,chosen,rebalance,dates,cost_bps=5,stop_fraction=.1)
        for mutation in ('cash','units','fee','stop','phase'):
            with self.subTest(mutation=mutation):
                changed=deepcopy(result)
                if mutation=='cash':changed.daily.loc[1,'cash']+=.01
                if mutation=='units':changed.holdings.loc[0,'units']*=1.01
                if mutation=='fee':changed.market_trades.loc[0,'cost']+=.01
                if mutation=='stop':changed.stops=changed.stops.iloc[0:0]
                if mutation=='phase':changed.trades.loc[0,'phase']='year_end_liquidation'
                with self.assertRaises(verify.VerificationError):audit(changed,quote,chosen,rebalance,dates,cost_bps=5,stop_fraction=.1)

    def test_independent_selection_ranks_ties_original_rows_and_denominators(self):
        forecast=predictions(); selected=select_walkforward_basket(forecast,basket='class_1_10',
            sessions=walkforward_clock()[2]['selection_sessions'],configuration=configuration())
        windows=pd.DataFrame([{'signal_date':'2019-12-31','entry_date':'2020-01-02'}])
        report=verify.verify_selection(forecast,selected.ledger,selected.selections,selected.monthly,windows,basket='class_1_10')
        self.assertEqual(report['selected_slots'],18)
        changed=selected.selections.copy(); changed.loc[changed.side.eq('short').idxmax(),'source_security_id']='short-036'
        with self.assertRaisesRegex(verify.VerificationError,'rank/tie'):
            verify.verify_selection(forecast,selected.ledger,changed,selected.monthly,windows,basket='class_1_10')
        changed=selected.monthly.copy(); changed.loc[0,'long_candidates']=999
        with self.assertRaisesRegex(verify.VerificationError,'Candidate count'):
            verify.verify_selection(forecast,selected.ledger,selected.selections,changed,windows,basket='class_1_10')

    def test_action_successor_opposite_lots_net_actual_market_costs(self):
        dates=('2020-01-02','2020-01-03'); chosen=selection(); rebalance=schedule()
        quote=prices(dates,a=(100.,100.),b=(100.,100.)); action=event(cash=0.,successor='B',ratio=1.)
        result=run_walkforward_accounting(chosen,quote,dates,rebalances=rebalance,terminal_date=dates[-1],events=[action],
            policy=DailyAccountingPolicy(cost_bps=25,required_names_per_side=None))
        report=audit(result,quote,chosen,rebalance,dates,events=[action],cost_bps=25)
        self.assertEqual(report['action_count'],1)
        final=result.market_trades.iloc[-1]; self.assertEqual(final['source_lot_count'],2)
        self.assertEqual(final['cost'],0.); self.assertGreater(final['internally_netted_notional'],0.)

    def test_date_guard_rejects_unheld_2024_before_monetary_projection(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'quotes.parquet'; frame=prices(); frame.loc[0,'trading_date']='2024-01-02'
            frame.to_parquet(path,index=False); original=pq.ParquetFile; reads=[]
            class Guard:
                def __init__(self,*args,**kwargs):self.pf=original(*args,**kwargs)
                def __getattr__(self,name):return getattr(self.pf,name)
                def read(self,columns=None,*args,**kwargs):
                    reads.append(columns)
                    if columns is None or 'adj_close' in columns:raise AssertionError('Protected financial decode')
                    return self.pf.read(columns=columns,*args,**kwargs)
            with patch.object(verify.pq,'ParquetFile',side_effect=Guard):
                with self.assertRaisesRegex(verify.VerificationError,'Protected period'):
                    verify.key_table(path,kind='execution',columns=['trading_date','adj_close'])
            self.assertEqual(reads,[['trading_date']])

    def test_direct_metrics_from_continuous_close_nav_and_bad_year_reset(self):
        from dmv2.daily_backtest import summarize_daily
        from dmv2.walkforward_backtest import yearly_breakdown
        dates=('2020-12-30','2020-12-31','2021-01-04','2021-01-05')
        months=(('2020-11-30','2020-12-30'),('2020-12-31','2021-01-04'))
        quote=prices(dates,a=(100.,102.,105.,110.),b=(100.,100.,100.,100.))
        result=run_walkforward_accounting(selection(months),quote,dates,rebalances=schedule(months),terminal_date=dates[-1],
            policy=DailyAccountingPolicy(cost_bps=5,required_names_per_side=None))
        summary,monthly=summarize_daily(result,arm='synthetic',basket='class_1_10',variant='unstopped',cost_bps=5,expected_sessions=4)
        annual=yearly_breakdown(result,arm='synthetic',basket='class_1_10',variant='unstopped',cost_bps=5)
        verify.verify_metrics(result.daily,result.trades,monthly,summary,annual)
        changed=annual.copy(); changed.loc[1,'initial_nav']=1.
        with self.assertRaisesRegex(verify.VerificationError,'calendar-year initial_nav'):
            verify.verify_metrics(result.daily,result.trades,monthly,summary,changed)
        changed=dict(summary); changed['total_return']+=.01
        with self.assertRaisesRegex(verify.VerificationError,'Full-period metric total_return'):
            verify.verify_metrics(result.daily,result.trades,monthly,changed,annual)

    def test_failed_quote_paths_retain_lots_null_metrics_and_no_inferred_exit(self):
        from dmv2.daily_backtest import summarize_daily
        from dmv2.walkforward_backtest import yearly_breakdown
        dates=('2020-01-02','2020-01-03'); chosen=selection(); rebalance=schedule()
        quote=prices(dates); quote=quote.loc[~(quote.trading_date.eq('2020-01-03')&quote.intrinio_security_id.eq('A'))]
        result=run_walkforward_accounting(chosen,quote,dates,rebalances=rebalance,terminal_date=dates[-1],
            policy=DailyAccountingPolicy(cost_bps=5,required_names_per_side=None))
        checked=verify.verify_failed_scenario(result.tables,quote,chosen,dates,cost_bps=5)
        self.assertFalse(checked['economic_performance_available']); self.assertEqual(checked['retained_unvalued_lots'],2)
        summary,monthly=summarize_daily(result,arm='synthetic',basket='class_1_10',variant='unstopped',cost_bps=5,expected_sessions=2)
        annual=yearly_breakdown(result,arm='synthetic',basket='class_1_10',variant='unstopped',cost_bps=5)
        verify.verify_failed_metrics(result.daily,result.trades,monthly,summary,annual)
        changed=deepcopy(result); failed=changed.holdings.date.eq(pd.Timestamp('2020-01-03'))
        changed.holdings.loc[failed,'units']=0.
        with self.assertRaisesRegex(verify.VerificationError,'Retained unvalued units'):
            verify.verify_failed_scenario(changed.tables,quote,chosen,dates,cost_bps=5)
        changed=deepcopy(result); changed.daily.loc[changed.daily.status.eq('failed_closed'),'nav']=1.
        with self.assertRaisesRegex(verify.VerificationError,'invented mark'):
            verify.verify_failed_scenario(changed.tables,quote,chosen,dates,cost_bps=5)
        with self.assertRaisesRegex(verify.VerificationError,'contradicted'):
            verify.verify_failed_scenario(result.tables,prices(dates),chosen,dates,cost_bps=5)
        changed=dict(summary); changed['total_return']=0.
        with self.assertRaisesRegex(verify.VerificationError,'metric invented'):
            verify.verify_failed_metrics(result.daily,result.trades,monthly,changed,annual)

    def test_source_quality_and_nonpositive_gap_failures_are_not_silent_exits(self):
        dates=('2020-01-02','2020-01-03'); chosen=selection(); rebalance=schedule()
        for kind in ('quality','nonpositive_gap'):
            quote=prices(dates,a=(100.,100.),b=(100.,400. if kind=='nonpositive_gap' else 100.))
            if kind=='quality':quote.loc[quote.trading_date.eq('2020-01-03')&quote.intrinio_security_id.eq('A'),'price_quality_status']='rejected'
            result=run_walkforward_accounting(chosen,quote,dates,rebalances=rebalance,terminal_date=dates[-1],
                policy=DailyAccountingPolicy(cost_bps=5,required_names_per_side=None,asset_stop_fraction=.1))
            checked=verify.verify_failed_scenario(result.tables,quote,chosen,dates,cost_bps=5,stop_fraction=.1)
            self.assertFalse(checked['economic_performance_available']); self.assertEqual(checked['retained_unvalued_lots'],2)

    def test_saved_config_and_original_bytes_rejected_before_source_decode(self):
        from test_walkforward_preflight_runner import fixture,save
        from dmv2.backtest_loading import _pin
        from dmv2.catalog import canonical_sha256
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve(); run,cfg,_,_=fixture(root)
            cfg['execution'].update(verification_maximum_peak_rss_bytes=8*1024**3,
                verification_maximum_wall_seconds=1800,verification_model_fits=0)
            parent={'configuration':cfg}; identity=canonical_sha256(parent); save(run/'build_identity.json',parent)
            save(run/'audit.json',{'status':'COMPLETE','build_identity_sha256':identity})
            save(root/'configs/walkforward_2020_2023.json',dict(cfg,protocol_id='changed'))
            with patch.object(verify,'_quote_inputs',side_effect=AssertionError('Financial decode after config drift')):
                with self.assertRaisesRegex(verify.VerificationError,'configuration changed'):
                    verify.verify_saved_run(root,run,run/'backtest')
            save(root/'configs/walkforward_2020_2023.json',cfg)
            old=root/cfg['historical_retirement_registry']['directory']
            history=verify.json_file(old/'build_identity.json')['events']
            phase={'implementation_revision':'identity_route_failure_v2','identity_handover_registry':None,'parent_identity_sha256':identity,'configuration_sha256':canonical_sha256(cfg),
                'source_pins':[_pin(root,old/'build_identity.json')],'historical_retirements':history}
            directory=run/'backtest'; save(directory/'build_identity.json',phase)
            save(directory/'audit.json',{'status':'COMPLETE','phase':'walkforward_continuous_backtest_v1',
                'build_identity_sha256':identity,'phase_identity_sha256':canonical_sha256(phase),'artifacts':{}})
            old_path=old/'build_identity.json'; old_path.write_bytes(old_path.read_bytes()+b'\n')
            with patch.object(verify,'_quote_inputs',side_effect=AssertionError('Financial decode after original drift')):
                with self.assertRaisesRegex(verify.VerificationError,'hash/size'):
                    verify.verify_saved_run(root,run,directory)

    def test_no_production_accounting_or_fitting_import_and_utf8_canonical(self):
        from dmv2.catalog import canonical_sha256
        source=SCRIPT.read_text()
        self.assertNotIn('from dmv2',source); self.assertNotIn('import sklearn',source)
        self.assertNotIn('import xgboost',source); self.assertNotIn('.fit(',source)
        self.assertEqual(verify.canonical({'unicode':'2020–2023'}),canonical_sha256({'unicode':'2020–2023'}))


if __name__=='__main__':unittest.main()
