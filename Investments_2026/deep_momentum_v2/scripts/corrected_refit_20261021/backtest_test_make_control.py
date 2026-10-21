"""Fabricated control files only; no real model/forecast/source reads."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import control as c
import backtest_make_control as m


class MakeControl(unittest.TestCase):
    def fixture(self,root,status='COMPLETE_VERIFIED'):
        project=root/'deep_momentum_v2';project.mkdir();run=root/'run/models'
        (run/'pipeline').mkdir(parents=True);(run.parent/'verification').mkdir()
        proof=run.parent/'verification/models.json';proof.write_text('opaque numerical JSON must not parse')
        cat=root/'catalog.json';cat.write_text('opaque catalog pointer')
        protocol={'schema':'dmv2-corrected-refit-protocol/v1','status':'SEALED_BEFORE_FINANCIAL_READ',
            'outcome_years':list(m.YEARS),'parameter_search':False,'run_directory':str(run),
            'catalog_pin':c.pin(cat),'runtime':{}}
        pp=c.save_new(root/'model_protocol.json',protocol)
        ip=c.save_new(run/'pipeline/identity.json',{'protocol_pin':pp})
        c.save_new(run/'pipeline/progress.json',{'schema':'dmv2-corrected-refit-pipeline-progress/v1',
            'status':status,'identity_pin':ip,'final_verification_pin':c.pin(proof)})
        for year in m.YEARS:
            (run.parent/'verification'/f'models_{year}.json').write_text('opaque annual score receipt')
            for arm in m.ARMS:
                path=run/'folds'/str(year)/'arms'/arm;path.mkdir(parents=True)
                (path/'predictions.parquet').write_bytes(b'opaque forecast bytes; never decode')
                (path/'PREDICT_COMPLETE.json').write_text('opaque metrics; never parse')
        refs={}
        for name in m.EXECUTION_REFERENCE:
            relative=name+'.json';path=project/relative;path.write_text('opaque old proof; hash only')
            p=c.pin(path);refs[name]=(relative,p['sha256'],p['size_bytes'])
        return project,refs
    def test_exact20pins_constructed_without_parsing_metric_json(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name).resolve();project,refs=self.fixture(root)
            with patch.object(m,'ROOT',root),patch.object(m,'EXECUTION_REFERENCE',refs):
                output=m.make_control(project,root/'model_protocol.json',root/'control.json')
            obj=c.read(output['path']);self.assertEqual(len(obj['forecast_entries']),20)
            self.assertFalse(obj['construction_reads']['metric_JSON_parsed']);self.assertEqual(len(obj['model_verification_pins']),4)
    def test_running_models_block_control_and_write_nothing(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name).resolve();project,refs=self.fixture(root,'RUNNING')
            with patch.object(m,'ROOT',root),patch.object(m,'EXECUTION_REFERENCE',refs):
                with self.assertRaises(ValueError):m.make_control(project,root/'model_protocol.json',root/'control.json')
            self.assertFalse((root/'control.json').exists())

if __name__=='__main__':unittest.main()
