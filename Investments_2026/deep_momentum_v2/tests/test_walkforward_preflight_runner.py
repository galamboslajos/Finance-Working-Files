"""Real small source pins exercise runner preflight without price/model decode."""
import json
from importlib.machinery import ModuleSpec
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from dmv2.action_loading import ACTION_PRICE_COLUMNS
from dmv2.backtest_loading import ARMS, _hash, _pin
from dmv2.catalog import canonical_sha256
from dmv2.walkforward_backtest import _historical_registry, _prediction_plan, run_walkforward_backtest
from dmv2.walkforward_clock import walkforward_clock
from test_walkforward_accounting import configuration
from dmv2.universe import DataContractError


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))


def fixture(root):
    cfg=configuration(); cfg['corrected_validation_directory']='outputs/development'
    cfg['corrected_validation_build_identity_sha256']='synthetic-old-identity'
    save(root/'configs/walkforward_2020_2023.json',cfg)
    events=[dict(event_id=f'old-{i}',predecessor_id=f'old-native-{i}',effective_date='2019-01-02',
        verified=True,evidence_reference='https://www.sec.gov/synthetic') for i in range(11)]
    old_phase={'events':events}; old=root/'outputs/development/backtest_completed'
    save(old/'build_identity.json',old_phase)
    save(old/'audit.json',{'status':'COMPLETE','phase':'independent_quote_action_completion_v1',
        'build_identity_sha256':cfg['corrected_validation_build_identity_sha256'],
        'phase_identity_sha256':canonical_sha256(old_phase)})
    cfg['historical_retirement_registry']={'directory':str(old.relative_to(root)),
        'audit_sha256':_hash(old/'audit.json'),'build_identity_sha256':_hash(old/'build_identity.json')}
    save(root/'configs/walkforward_2020_2023.json',cfg)
    run=root/'outputs/walkforward/synthetic'
    identity={'configuration':cfg}; pin=canonical_sha256(identity)
    save(run/'build_identity.json',identity)
    model_phase={'parent_identity_sha256':pin}; model_pin=canonical_sha256(model_phase)
    save(run/'models/build_identity.json',model_phase)
    sessions,windows,_=walkforward_clock()
    arms={}
    for arm in ARMS:
        directory=run/'models/arms'/arm; directory.mkdir(parents=True)
        records=[]
        for window in windows.to_dict('records'):
            year=int(window['outcome_month'][:4])
            for native,mode in [('A',1),('B',10),('C',2),('D',9)]:
                records.append(dict(source_security_id=native,signal_date=window['signal_date'],
                    intrinio_security_id=native,prediction_eligible=True,paper_prediction_eligible=True,
                    fit_year=year,training_outcome_start=f'{year-10}-01-01',training_outcome_end=f'{year-1}-12-31',
                    training_label_observed_through={2020:'2019-12-31',2021:'2020-12-31',2022:'2021-12-31',2023:'2022-12-30'}[year],
                    outcome_month=window['outcome_month'],outcome_start=window['outcome_start'],outcome_end=window['outcome_end'],
                    **{f'probability_{k}':.82 if k==mode else .02 for k in range(1,11)}))
        path=directory/'predictions.parquet'; pd.DataFrame(records).to_parquet(path,index=False)
        save(directory/'audit.json',{'status':'COMPLETE','build_identity_sha256':pin,
            'model_identity_sha256':model_pin,'artifacts':{'predictions':_pin(directory,path)}})
        arms[arm]=_pin(run/'models',directory/'audit.json')
    model_audit={'status':'COMPLETE','phase':'walkforward_models_v1','build_identity_sha256':pin,
        'model_identity_sha256':model_pin,'summary':{'fitted_arms':16,'outer_years':[2020,2021,2022,2023],'months':48},'arms':arms}
    save(run/'models/audit.json',model_audit)
    for year in range(2020,2024):
        directory=run/'years'/str(year); directory.mkdir(parents=True)
        records=[]
        for day in sessions:
            if day[:4]!=str(year):continue
            for native in ['A','B','C','D']:
                row={name:100. for name in ACTION_PRICE_COLUMNS}
                row.update(trading_date=day,intrinio_security_id=native,frequency='daily',intraperiod=False,
                    price_quality_status='accepted',raw_ohlc_bounds_valid=True,adjusted_ohlc_bounds_valid=True,
                    factor=1.,split_ratio=1.,dividend=0.)
                records.append(row)
        path=directory/'daily_prices_raw.parquet'; pq.write_table(pa.Table.from_pylist(records),path)
        save(directory/'audit.json',{'status':'COMPLETE','year':year,'build_identity_sha256':pin,
            'feature_provenance':{'requested_ids':['A','B','C','D']},
            'artifacts':{'daily_prices_raw':_pin(directory,path)}})
    return run,cfg,pin,model_audit


class WalkforwardPreflightRunnerTests(unittest.TestCase):
    def test_historical_registry_bytes_checked_before_any_json_decoding(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve(); run,cfg,pin,model_audit=fixture(root)
            old=root/cfg['historical_retirement_registry']['directory']
            for name in ('audit.json','build_identity.json'):
                with self.subTest(name=name):
                    path=old/name; original=path.read_bytes(); path.write_bytes(original+b'\n')
                    with patch('dmv2.walkforward_backtest._json',side_effect=AssertionError('JSON decoded before old registry byte validation')):
                        with self.assertRaisesRegex(DataContractError,'retirement bytes changed'):
                            _historical_registry(root,cfg,[])
                    path.write_bytes(original)
            changed=dict(cfg); changed['historical_retirement_registry']=dict(cfg['historical_retirement_registry'])
            changed['historical_retirement_registry']['directory']='outputs/another_run'
            with self.assertRaises(DataContractError):_historical_registry(root,changed,[])

    def test_wrong_annual_training_lineage_rejected_before_probabilities(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve(); run,cfg,pin,model_audit=fixture(root)
            arm=ARMS[0]; directory=run/'models/arms'/arm
            path=directory/'predictions.parquet'; frame=pd.read_parquet(path)
            frame.loc[0,'training_outcome_end']='2020-12-31'; frame.to_parquet(path,index=False)
            audit_path=directory/'audit.json'; audit=json.loads(audit_path.read_text())
            audit['artifacts']['predictions']=_pin(directory,path); save(audit_path,audit)
            model_audit['arms'][arm]=_pin(run/'models',audit_path)
            save(run/'models/audit.json',model_audit)
            module=ModuleType('dmv2.walkforward_models')
            module.read_walkforward_models=lambda *args,**kwargs:{'audit':model_audit}
            with patch.dict(sys.modules,{'dmv2.walkforward_models':module}):
                with self.assertRaisesRegex(DataContractError,'lineage'):
                    _prediction_plan(root,run,pin,walkforward_clock()[1],[])

    def test_preflight_only_saves_all_eight_48_month_baskets_without_prices_or_pnl(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve(); run,cfg,pin,model_audit=fixture(root)
            module=ModuleType('dmv2.walkforward_models')
            module.__spec__=ModuleSpec('dmv2.walkforward_models',loader=None,origin=__file__)
            module.read_walkforward_models=lambda *args,**kwargs:{'audit':model_audit}
            original=pq.ParquetFile
            class GuardedParquet:
                def __init__(self,path,*args,**kwargs):self.path=Path(path); self.base=original(path,*args,**kwargs)
                def __getattr__(self,name):return getattr(self.base,name)
                def read(self,columns=None,*args,**kwargs):
                    if self.path.name=='daily_prices_raw.parquet' and (columns is None or {'open','close','adj_open','adj_close'}&set(columns)):
                        raise AssertionError('Financial quote decode in preflight-only phase')
                    return self.base.read(columns=columns,*args,**kwargs)
            with patch.dict(sys.modules,{'dmv2.walkforward_models':module}), \
                 patch('dmv2.walkforward_backtest.run_walkforward_accounting',side_effect=AssertionError('PnL in preflight-only phase')), \
                 patch('dmv2.walkforward_backtest.pq.ParquetFile',side_effect=GuardedParquet):
                directory=run_walkforward_backtest(root,run,cfg,pin,preflight_only=True)
            audit=json.loads((directory/'audit.json').read_text())
            self.assertFalse(audit['performance_computed']); self.assertFalse(audit['selection_changed'])
            summary=pd.read_parquet(directory/'summary.parquet')
            self.assertEqual(len(summary),8); self.assertTrue(summary.ready_for_accounting.all())
            self.assertFalse((run/'backtest').exists())
            for arm in ARMS:
                for basket in ['class_1_10','class_2_9']:
                    monthly=pd.read_parquet(directory/arm/basket/'monthly.parquet')
                    self.assertEqual(len(monthly),48)
                    self.assertTrue(monthly.selected_names_per_side.eq(1).all())


if __name__=='__main__':unittest.main()
