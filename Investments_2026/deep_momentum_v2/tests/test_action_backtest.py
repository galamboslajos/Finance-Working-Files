"""Bounded orchestration contracts using synthetic prices and actual file hashes."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from dmv2 import action_backtest as worker
from dmv2.backtest_loading import _hash, _pin
from dmv2.catalog import canonical_sha256
from dmv2.universe import DataContractError

ROOT=Path(__file__).resolve().parents[1]


def selections():
    return pd.DataFrame([dict(signal_date='2017-12-29',execution_date='2018-01-02',source_security_id=f'{side}{i}',
                              intrinio_security_id=f'{side}{i}',side=side,identity_review_required=True)
                         for side in ('long','short') for i in range(10)])


class ActionBacktestTests(unittest.TestCase):
    def test_fixed_configuration_and_mandatory_original_repairs(self):
        cfg=worker.configuration(ROOT)
        self.assertEqual(cfg['trading_cost_bps'],[0,5,10,25])
        self.assertEqual(len(cfg['ohlc_repairs']['mandatory_original_keys']),3)
        self.assertEqual(cfg['ohlc_repairs']['maximum_raw_close_absolute_difference'],.05)
        self.assertEqual(cfg['ohlc_repairs']['maximum_raw_close_relative_difference'],.005)
        self.assertFalse(cfg['execution']['financial_2020_plus_read'])
        self.assertFalse(cfg['reporting']['complete_paths_only_model_leaderboard'])
        for key,value in [('end','2020-12-31'),('required_names_per_side',9),('short_weight',-.4)]:
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory);(root/'configs').mkdir()
                altered=copy.deepcopy(cfg);altered[key]=value
                (root/'configs/action_backtest.json').write_text(json.dumps(altered))
                with self.assertRaisesRegex(DataContractError,'Frozen'):
                    worker.configuration(root)

    def test_selection_metadata_has_ten_original_names_each_side(self):
        frame=selections()
        before=frame.copy(deep=True)
        worker._check_selection_metadata(frame)
        pd.testing.assert_frame_equal(frame,before)
        with self.assertRaisesRegex(DataContractError,'ten original'):
            worker._check_selection_metadata(frame.iloc[1:])

    def test_dates_guarded_before_financial_selection_columns_are_read(self):
        frame=selections()
        frame['probability_1']=object()
        frame.loc[0,'execution_date']='2020-01-02'
        with self.assertRaisesRegex(DataContractError,'authorized'):
            worker._check_selection_metadata(frame)
        frame=selections();frame.loc[0,'signal_date']='2017-11-30'
        with self.assertRaisesRegex(DataContractError,'chronology'):
            worker._check_selection_metadata(frame)

    def test_both_sides_required_at_every_execution(self):
        frame=selections();frame.loc[frame.side.eq('short'),'execution_date']='2018-02-01'
        frame.loc[frame.side.eq('short'),'signal_date']='2018-01-31'
        with self.assertRaisesRegex(DataContractError,'ten original'):
            worker._check_selection_metadata(frame)

    def test_missing_saved_reader_never_plans_loads_repairs_or_runs(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(worker,'plan_action_backtest') as plan,\
             patch.object(worker,'load_action_inputs') as load,patch.object(worker,'_prepare_prices') as repair,\
             patch.object(worker,'run_action_accounting') as accounting:
            with self.assertRaises(FileNotFoundError):worker.read_action_backtest(directory)
            for method in (plan,load,repair,accounting):method.assert_not_called()

    def test_missing_actual_reviewed_manifest_has_clear_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(DataContractError,'three bars are not fixed'):
                worker._reviewed_manifest(Path(directory),Path(directory))

    def test_all_three_mandatory_repairs_required(self):
        cfg=worker.configuration(ROOT)
        manifest=dict(projection_evidence={},source_provenance={},native_mapping={},review_notes='reviewed')
        for keys,passes in [(cfg['ohlc_repairs']['mandatory_original_keys'][:2],False),(cfg['ohlc_repairs']['mandatory_original_keys'],True)]:
            prices=pd.DataFrame([dict(intrinio_security_id=sid,trading_date=day,**{field:1. for field in worker.ORIGINAL_FIELDS}) for sid,day in keys])
            bundle=SimpleNamespace(daily_prices=prices)
            prepared=SimpleNamespace(repairs=tuple(dict(intrinio_security_id=sid,trading_date=day) for sid,day in keys),identity_and_control_ledger=())
            def checked(subset, repairs, **kwargs):
                return SimpleNamespace(prices=subset,ledger=pd.DataFrame(repairs))
            with patch.object(worker,'prepare_published_alpaca_repair_declarations',return_value=prepared),\
                 patch.object(worker,'apply_reviewed_ohlc_repairs',side_effect=checked) as apply:
                if passes:worker._prepare_prices(bundle,manifest,cfg,ROOT)
                else:
                    with self.assertRaisesRegex(DataContractError,'not all repaired'):
                        worker._prepare_prices(bundle,manifest,cfg,ROOT)
                policy=apply.call_args.kwargs['policy']
                self.assertEqual(policy.maximum_raw_close_absolute_difference,.05)
                self.assertEqual(policy.maximum_raw_close_relative_difference,.005)
                self.assertEqual(policy.maximum_original_adjusted_rounding_residual,.0002)

    def synthetic_plan(self,root):
        (root/'configs').mkdir()
        (root/'configs/action_backtest.json').write_bytes((ROOT/'configs/action_backtest.json').read_bytes())
        original_path=root/'outputs/daily_backtest/synthetic-frozen-run'
        original_path.mkdir(parents=True)
        pointer=original_path.parent/'LATEST.json';pointer.write_text('{"synthetic":true}\n')
        selected=selections().assign(signal_date='2019-11-29',execution_date='2019-12-30')
        schedule=pd.DataFrame([dict(signal_date='2019-11-29',execution_date='2019-12-30')])
        paths={};selected_sha={};rebalance_sha={};pins=[_pin(root,pointer)]
        for arm in worker.ARMS:
            for basket in ('class_1_10','class_2_9'):
                key=f'{arm}/{basket}'
                for name,frame in [('selected',selected),('rebalances',schedule),('original_ledger',selected)]:
                    relative=f'selections/{key}/{name}.parquet'
                    path=original_path/relative;path.parent.mkdir(parents=True,exist_ok=True)
                    frame.to_parquet(path,index=False);paths[relative]=path;pins.append(_pin(root,path))
                    if name=='selected':selected_sha[key]=_hash(path)
                    if name=='rebalances':rebalance_sha[key]=_hash(path)
        cfg=worker.configuration(root)
        original=SimpleNamespace(path=original_path,selection_coverage=pd.DataFrame({'preserved':[True]}),
                                 ret_readiness=pd.DataFrame({'signal':range(24),'status':['withheld']*24}))
        identity=dict(schema_version=1,configuration=cfg,workers=worker._worker_hashes(),source_pins=pins,
                      original_selected_sha256=selected_sha,original_rebalances_sha256=rebalance_sha)
        plan=SimpleNamespace(readiness_failures=())
        planned=dict(root=root,configuration=cfg,evidence_dir=root,original=original,original_paths=paths,
                     reviewed_manifest=dict(projection_evidence={},source_provenance={},native_mapping={},review_notes='synthetic'),
                     input_plan=plan,identity=identity,build_identity_sha256=canonical_sha256(identity))
        sessions=('2019-12-30','2019-12-31')
        prices=pd.DataFrame([dict(trading_date=date,intrinio_security_id=provider,adj_open=100.,adj_close=100.,
                                  adj_high=100.,adj_low=100.,open=100.,close=100.,price_accepted=True)
                             for date in sessions for provider in selected.intrinio_security_id])
        bundle=SimpleNamespace(daily_prices=prices,events=pd.DataFrame(),sessions=sessions,provenance={'synthetic':True})
        repair_ledger=pd.DataFrame(cfg['ohlc_repairs']['mandatory_original_keys'],columns=['intrinio_security_id','trading_date'])
        return planned,bundle,prices,repair_ledger,pointer

    def test_full_64_synthetic_orchestration_preserves_bytes_and_reader_is_passive(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);planned,bundle,prices,repairs,pointer=self.synthetic_plan(root)
            original_pointer=pointer.read_bytes()
            with patch.object(worker,'plan_action_backtest',return_value=planned),\
                 patch.object(worker,'load_action_inputs',return_value=bundle),\
                 patch.object(worker,'_prepare_prices',return_value=(prices,repairs,pd.DataFrame({'review':['synthetic']}))),\
                 patch.object(worker,'admit_reviewed_missing_bars',return_value=SimpleNamespace(prices=prices,ledger=pd.DataFrame(),evidence_pins=())),\
                 patch.object(worker,'verify_action_inputs'),patch('builtins.print'):
                saved=worker.run_action_backtest(root)
            self.assertEqual(len(saved.comparison),64)
            self.assertEqual(saved.audit['complete_proxy_paths'],64)
            self.assertEqual(pointer.read_bytes(),original_pointer)
            for key,expected in planned['identity']['original_selected_sha256'].items():
                self.assertEqual(_hash(saved.path/f'selections/{key}/selected.parquet'),expected)
            for key,expected in planned['identity']['original_rebalances_sha256'].items():
                self.assertEqual(_hash(saved.path/f'selections/{key}/rebalances.parquet'),expected)
            with patch.object(worker,'plan_action_backtest') as plan,patch.object(worker,'load_action_inputs') as load,\
                 patch.object(worker,'run_action_accounting') as run,patch.object(worker,'_prepare_prices') as repair:
                read=worker.read_action_backtest(root)
                self.assertEqual(len(read.comparison),64)
                for method in (plan,load,run,repair):method.assert_not_called()
            (saved.path/'comparison.parquet').write_bytes(b'tampered')
            with self.assertRaises(DataContractError):worker.read_action_backtest(root)

    def test_missing_mandatory_repairs_do_not_create_output_or_change_original_pointer(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);planned,bundle,prices,repairs,pointer=self.synthetic_plan(root)
            original_pointer=pointer.read_bytes()
            with patch.object(worker,'plan_action_backtest',return_value=planned),patch.object(worker,'load_action_inputs',return_value=bundle),\
                 patch.object(worker,'_prepare_prices',side_effect=DataContractError('missing actual compatible repairs')),\
                 patch.object(worker,'run_action_accounting') as run:
                with self.assertRaisesRegex(DataContractError,'missing actual'):
                    worker.run_action_backtest(root)
                run.assert_not_called()
            self.assertFalse((root/'outputs/action_backtest').exists())
            self.assertEqual(pointer.read_bytes(),original_pointer)


if __name__=='__main__':unittest.main()
