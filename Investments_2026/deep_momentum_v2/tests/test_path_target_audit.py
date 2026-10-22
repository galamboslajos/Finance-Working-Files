"""Synthetic analytical parity; no local research cache required."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow as pa

import dmv2.path_target_audit as m
from dmv2.universe import DataContractError


def fixture():
    sessions,windows,_=m._clock()
    window=windows[0]
    days=sorted({r['date'] for r in window['required_marks']})
    prices=pd.DataFrame([{'trading_date':day,'intrinio_security_id':'native',
        'open':100.,'close':100.,'adj_open':100.,'adj_high':100.,'adj_low':100.,'adj_close':100.,
        'frequency':'daily','intraperiod':False,'price_quality_status':'accepted',
        'raw_ohlc_bounds_valid':True,'adjusted_ohlc_bounds_valid':True} for day in days])
    return prices,window


def replay(prices,window,side='long',variant='unstopped',events=(),historical=()):
    return m.replay_asset_episode(prices,source_security_id='source',intrinio_security_id='native',
        signal_date=window['signal_date'],window=window,side=side,path_variant=variant,
        events=events,historical_retirements=historical)


class PathAuditEpisodeTests(unittest.TestCase):
    def setUp(self): self.prices,self.window=fixture()
    def set_quote(self,date,opening,high,low,close):
        at=self.prices.trading_date.eq(date)
        for key,value in {'open':opening,'close':close,'adj_open':opening,'adj_high':high,'adj_low':low,'adj_close':close}.items():
            self.prices.loc[at,key]=value

    def test_long_short_are_standalone_unit_notional_not_half_returns(self):
        self.set_quote(self.window['exit_date'],110,110,110,110)
        before=self.prices.copy(deep=True)
        for side,expected in [('long',1.1),('short',.9)]:
            result=replay(self.prices,self.window,side)
            self.assertEqual(result['status'],'complete')
            self.assertAlmostEqual(result['marks'][-1]['wealth'],expected)
            self.assertEqual(result['marks'][0]['wealth'],1)
            self.assertEqual(len(result['marks']),len(self.window['required_marks']))
        pd.testing.assert_frame_equal(self.prices,before)

    def test_finite_negative_short_wealth_is_not_clipped(self):
        self.set_quote(self.window['exit_date'],250,250,250,250)
        result=replay(self.prices,self.window,'short')
        self.assertEqual(result['status'],'complete')
        self.assertAlmostEqual(result['marks'][-1]['wealth'],-.5)

    def test_kernel_capital_failure_is_reported_unknown_never_minus100(self):
        self.set_quote(self.window['exit_date'],350,350,350,350)
        result=replay(self.prices,self.window,'short')
        self.assertEqual(result['status'],'unavailable')
        self.assertTrue(result['engine_capital_gate_triggered'])
        self.assertNotEqual(result['marks'][-1]['wealth'],0.)

    def test_gap_stop_can_exceed_ten_percent(self):
        self.set_quote('2020-01-06',85,87,83,86)
        result=replay(self.prices,self.window,variant='asset_stop_10pct')
        self.assertEqual(result['status'],'complete')
        self.assertEqual(result['stop_count'],1)
        self.assertAlmostEqual(result['marks'][-1]['wealth'],.85)
        self.assertTrue(all(abs(row['wealth']-.85)<1e-12 for row in result['marks'] if row['date']>='2020-01-06'))

    def test_short_gap_stop(self):
        self.set_quote('2020-01-06',115,118,112,116)
        result=replay(self.prices,self.window,side='short',variant='asset_stop_10pct')
        self.assertEqual(result['status'],'complete')
        self.assertAlmostEqual(result['marks'][-1]['wealth'],.85)

    def test_long_intraday_stop_uses_original_threshold(self):
        self.set_quote('2020-01-06',100,102,85,98)
        result=replay(self.prices,self.window,variant='asset_stop_10pct')
        self.assertEqual(result['stop_count'],1)
        self.assertAlmostEqual(result['marks'][-1]['wealth'],.9)

    def test_short_intraday_stop(self):
        self.set_quote('2020-01-06',100,115,98,102)
        result=replay(self.prices,self.window,side='short',variant='asset_stop_10pct')
        self.assertEqual(result['stop_count'],1)
        self.assertAlmostEqual(result['marks'][-1]['wealth'],.9)

    def test_missing_interior_held_mark_keeps_unknown(self):
        self.prices=self.prices[self.prices.trading_date!='2020-01-06']
        result=replay(self.prices,self.window)
        self.assertEqual(result['status'],'unavailable')
        self.assertIn('missing_quote',result['reason'])

    def test_early_stop_retains_cash_marks_to_common_exit_without_later_quotes(self):
        self.set_quote('2020-01-06',85,87,83,86)
        self.prices=self.prices[self.prices.trading_date<='2020-01-06']
        result=replay(self.prices,self.window,variant='asset_stop_10pct')
        self.assertEqual(result['status'],'complete')
        self.assertEqual(result['marks'][-1]['date'],self.window['exit_date'])
        self.assertEqual(result['marks'][-1]['mark'],'open')
        self.assertAlmostEqual(result['marks'][-1]['wealth'],.85)

    def test_exit_close_is_not_part_of_episode_or_a_required_asset_price(self):
        self.set_quote(self.window['exit_date'],110,110,110,np.nan)
        result=replay(self.prices,self.window)
        self.assertEqual(result['status'],'complete')
        self.assertAlmostEqual(result['marks'][-1]['wealth'],1.1)
        self.assertNotIn({'date':self.window['exit_date'],'mark':'close'},[{k:r[k] for k in ['date','mark']} for r in result['marks']])

    def test_signed_cash_action_long_and_short(self):
        self.prices=self.prices[self.prices.trading_date<'2020-01-10']
        event={'event_id':'cash','effective_date':'2020-01-10','predecessor_id':'native','successor_id':None,
            'cash_per_raw_share':120.,'stock_ratio':0.,'verified':True,'evidence_reference':'synthetic-primary'}
        for side,expected in [('long',1.2),('short',.8)]:
            result=replay(self.prices,self.window,side,events=[event])
            self.assertEqual(result['status'],'complete')
            self.assertEqual(result['action_count'],1)
            self.assertAlmostEqual(result['marks'][-1]['wealth'],expected)

    def test_signed_stock_action_uses_successor_basis(self):
        successor=self.prices[self.prices.trading_date>='2020-01-10'].copy()
        successor['intrinio_security_id']='successor'
        for field in ['open','close','adj_open','adj_high','adj_low','adj_close']:successor[field]=220.
        self.prices=pd.concat([self.prices[self.prices.trading_date<'2020-01-10'],successor],ignore_index=True)
        event={'event_id':'stock','effective_date':'2020-01-10','predecessor_id':'native','successor_id':'successor',
            'cash_per_raw_share':0.,'stock_ratio':.5,'verified':True,'evidence_reference':'synthetic-primary'}
        for side,expected in [('long',1.1),('short',.9)]:
            result=replay(self.prices,self.window,side,events=[event])
            self.assertEqual(result['status'],'complete')
            self.assertAlmostEqual(result['marks'][-1]['wealth'],expected)

    def test_mixed_entitlement_uses_paired_raw_adjusted_units(self):
        for field in ['adj_open','adj_high','adj_low','adj_close']:self.prices[field]=50.
        successor=self.prices[self.prices.trading_date>='2020-01-10'].copy()
        successor['intrinio_security_id']='successor'
        for field in ['open','close']:successor[field]=200.
        for field in ['adj_open','adj_high','adj_low','adj_close']:successor[field]=80.
        self.prices=pd.concat([self.prices[self.prices.trading_date<'2020-01-10'],successor],ignore_index=True)
        event={'event_id':'mixed','effective_date':'2020-01-10','predecessor_id':'native','successor_id':'successor',
            'cash_per_raw_share':20.,'stock_ratio':.5,'verified':True,'evidence_reference':'synthetic-primary'}
        for side,expected in [('long',1.2),('short',.8)]:
            result=replay(self.prices,self.window,side,events=[event])
            self.assertEqual(result['status'],'complete')
            self.assertAlmostEqual(result['marks'][-1]['wealth'],expected)

    def test_full_original_episode_ledger_retains_unrequested_and_unknown(self):
        cohort=[{'source_security_id':'source','signal_date':self.window['signal_date'],'intrinio_security_id':'native','prediction_eligible':True},
            {'source_security_id':'missing','signal_date':self.window['signal_date'],'intrinio_security_id':'missing','prediction_eligible':True},
            {'source_security_id':'blocked','signal_date':self.window['signal_date'],'intrinio_security_id':None,'prediction_eligible':False}]
        plan={'original_cohort':cohort,'windows':[self.window],'events':[],'historical_retirements':[]}
        episodes,marks,details=m.build_audit_paths(plan,self.prices)
        self.assertEqual(len(episodes),12)
        self.assertEqual(episodes.status.value_counts().to_dict(),{'complete':4,'unavailable':4,'not_requested':4})
        self.assertEqual(episodes.requested.sum(),8)
        from dmv2.path_targets import build_path_targets
        sessions,_,_=m._clock()
        panel=build_path_targets(pa.Table.from_pylist(cohort),episodes,marks,windows=pd.DataFrame([self.window]),
            sessions=sessions,observed_through=self.window['exit_date'])
        self.assertEqual(panel.table.num_rows,24)
        self.assertEqual(set(panel.table['source_security_id'].to_pylist()),{'source','missing','blocked'})

    def test_cash_control_cannot_receive_a_corporate_action(self):
        event={'event_id':'bad','effective_date':'2020-01-10','predecessor_id':m.CASH_LONG,'successor_id':None,
            'cash_per_raw_share':100.,'stock_ratio':0.,'verified':True,'evidence_reference':'bad'}
        with self.assertRaisesRegex(DataContractError,'synthetic cash'):replay(self.prices,self.window,events=[event])

    def test_extra_2024_quote_metadata_refuses_before_any_numerical_use(self):
        row=self.prices.iloc[[0]].copy();row['trading_date']='2024-01-02'
        prices=pd.concat([self.prices,row],ignore_index=True)
        with patch.object(m,'run_walkforward_accounting_v2',side_effect=AssertionError('numeric kernel called')):
            with self.assertRaisesRegex(DataContractError,'unauthorized date'):replay(prices,self.window)

    def test_missing_successor_does_not_turn_into_cash_or_stale_price(self):
        event={'event_id':'stock','effective_date':'2020-01-10','predecessor_id':'native','successor_id':'missing',
            'cash_per_raw_share':0.,'stock_ratio':.5,'verified':True,'evidence_reference':'synthetic-primary'}
        result=replay(self.prices,self.window,events=[event])
        self.assertEqual(result['status'],'unavailable')
        self.assertIn('missing_quote',result['reason'])

    def test_retired_at_entry_is_not_a_fabricated_zero_payoff(self):
        event={'event_id':'retired','effective_date':'2020-01-02','predecessor_id':'native','successor_id':None,
            'cash_per_raw_share':120.,'stock_ratio':0.,'verified':True,'evidence_reference':'synthetic-primary'}
        result=replay(self.prices,self.window,events=[event])
        self.assertEqual(result['status'],'unavailable')
        self.assertIn('retired_before_entry',result['reason'])

    def test_2010_and_2024_episodes_refuse_without_date_warping(self):
        for date in ['2010-01-04','2024-01-02']:
            w=deepcopy(self.window);w['entry_date']=date
            with self.assertRaises(DataContractError):replay(self.prices,w)

    def test_unknown_source_quality_is_not_accepted(self):
        self.prices.loc[self.prices.trading_date=='2020-01-06','price_quality_status']='unknown'
        result=replay(self.prices,self.window)
        self.assertEqual(result['status'],'unavailable')
        self.assertIn('quality',result['reason'])

    def test_metadata_guard_refuses_future_dates_before_values(self):
        t=pa.table({'source_security_id':['source'],'signal_date':['2024-01-01']})
        with self.assertRaisesRegex(DataContractError,'approved financial dates'):
            m._guard_keys(t,'signal_date',lower='2019-12-01',upper='2020-11-30',keys=m.KEYS)

    def test_reader_is_passive_and_missing_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):m.read_path_target_audit(directory)
            p=Path(directory)/'audit.json';p.write_text(json.dumps({'status':'COMPLETE_BOUNDED_AUDIT','protocol':m.PROTOCOL,'artifacts':{},'models_fitted':False}))
            with patch.object(m.pq,'ParquetFile',side_effect=AssertionError('decoded')),patch.object(m,'run_walkforward_accounting_v2',side_effect=AssertionError('computed')):
                self.assertFalse(m.read_path_target_audit(directory)['models_fitted'])

    def test_synthetic_explicit_CLI_run_strips_legacy_fields_and_preserves_null_ids(self):
        import importlib.util
        path=Path(m.__file__).resolve().parents[2]/'scripts/run_path_target_audit.py'
        spec=importlib.util.spec_from_file_location('synthetic_path_audit_cli',path)
        cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
        sessions,windows,_=m._clock()
        prices=[];cohort=[];legacy=[]
        for window in windows:
            dates=sorted({r['date'] for r in window['required_marks']})
            for i in range(11):
                source='source'+str(i);native='native'+str(i) if i<10 else None
                cohort.append({'source_security_id':source,'signal_date':window['signal_date'],
                    'intrinio_security_id':native,'prediction_eligible':i<10,'outcome_month':window['outcome_month'],
                    'target_available':i<10,'target_status':'old_available','identity_review_required':True})
                legacy.append({'source_security_id':source,'signal_date':window['signal_date'],
                    'label_return':0. if i<10 else None,'target_class':5 if i<10 else None})
                if native:
                    for day in dates:
                        prices.append({'trading_date':day,'intrinio_security_id':native,'open':100.,'close':100.,
                            'adj_open':100.,'adj_high':100.,'adj_low':100.,'adj_close':100.,'frequency':'daily',
                            'intraperiod':False,'price_quality_status':'accepted','raw_ohlc_bounds_valid':True,
                            'adjusted_ohlc_bounds_valid':True})
        plan={'protocol':m.PROTOCOL,'plan_sha256':'synthetic','original_cohort':cohort,'windows':windows,'sessions':sessions,
            'events':[],'historical_retirements':[],'semantics':{},'read_plan':{},
            'limits':{'wall_seconds':30,'peak_rss_bytes':2*1024**3,'output_bytes':64*1024**2}}
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'fresh'
            with patch.object(cli,'verify_path_audit_plan',return_value=True), \
                 patch.object(cli,'load_path_audit_values',return_value=(pd.DataFrame(prices),pd.DataFrame(legacy))):
                audit=cli.run(plan,out)
            self.assertEqual(audit['original_rows'],22)
            self.assertEqual(audit['complete_episode_count'],80)
            self.assertEqual(audit['not_requested_episode_count'],8)
            self.assertFalse(audit['training_history_path_targets_ready'])
            self.assertTrue(audit['monthly_episode_coverage'])
            self.assertEqual(audit['legacy_alignment']['class_changed_rows'],20)
            self.assertFalse(audit['models_fitted'])
            table=pd.read_parquet(out/'targets.parquet')
            self.assertEqual(len(table),176)
            self.assertTrue(table.loc[~table.prediction_eligible,'intrinio_security_id'].isna().all())
            self.assertIn('legacy_target_available',table)

    def test_legacy_target_future_outcome_clock_refuses_before_value_read(self):
        table=pa.table({'signal_date':['2020-01-31'],'outcome_month':['2024-02'],
            'outcome_start':['2024-02-01'],'outcome_end':['2024-02-29']})
        with self.assertRaisesRegex(DataContractError,'physically scoped'):
            m._guard_target_clock(table)

    def test_reader_caps_json_size(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory)/'audit.json').write_bytes(b' '* (1024*1024+1))
            with self.assertRaisesRegex(DataContractError,'byte ceiling'):m.read_path_target_audit(directory)


if __name__=='__main__':unittest.main()
