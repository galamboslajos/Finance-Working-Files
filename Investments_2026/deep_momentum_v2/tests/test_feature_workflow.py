"""Feature stage controls, separate evidence and stale notebook-state guards."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import pandas as pd
import pyarrow as pa

from dmv2.feature_workflow import FeatureRun, run_feature_sample, save_feature_sample
from dmv2.loading import LoadedTable
from dmv2.workflow import validate_workflow
from test_master_notebook import cell_containing


def configuration():
    return json.loads((Path(__file__).resolve().parents[1]/'configs/research_workflow.json').read_text())


def fixture():
    sample = SimpleNamespace(signal_dates=('1999-12-31',),
        daily_prices=LoadedTable(pa.table({'intrinio_security_id':['fixture-a'], 'trading_date':['1999-12-31'], 'adj_close':[float('nan')]}), {}, {}),
        calendar=pa.table({'trading_date':['1999-12-31']}), source_calendar=pa.table({'trading_date':['1999-12-31']}))
    panel = SimpleNamespace(table=pa.table({'source_security_id':['member-a'], 'signal_date':['1999-12-31'], 'fixture_feature':[None]}),
        coverage=pa.table({'source_security_id':['member-a'], 'feature_status':['no_usable_features']}),
        specs=({'name':'fixture_feature','formula':'structural persistence fixture'},),
        diagnostics={'daily':pd.DataFrame({'status':['missing_input']})})
    report={'workflow_sha256':'fixture-settings','source_pins':{'release':'fixture-r1'},
            'code_provenance':{'worker_sha256':{'features.py':'fixture-code'}},'summary':{'rows':1}}
    return FeatureRun(sample,panel,report)


class FeatureWorkflowTests(unittest.TestCase):
    def test_disabled_feature_stage_reads_nothing(self):
        settings=configuration();settings['features']['enabled']=False
        with patch('dmv2.feature_workflow.load_feature_sample') as loader:
            self.assertIsNone(run_feature_sample(Mock(),settings))
            loader.assert_not_called()

    def test_invalid_feature_controls_fail_before_reads(self):
        for key,value in [('enabled',1),('persist','true'),('history_sessions',252),
                          ('history_sessions',True),('cohort_policy','sample'),('context_scope','sp1500'),
                          ('read_budget',[]),('read_budget',{'max_scanned_rows':0}),('read_budget',{'unknown':1})]:
            settings=configuration();settings['features'][key]=value
            with self.subTest(key=key,value=value),patch('dmv2.feature_workflow.load_feature_sample') as loader:
                with self.assertRaises(ValueError):run_feature_sample(Mock(),settings)
                loader.assert_not_called()

    def test_feature_controls_required_even_when_disabled(self):
        settings=configuration();del settings['features']['history_sessions']
        with self.assertRaises(ValueError):validate_workflow(settings)

    def test_separate_private_tables_dictionary_and_idempotency(self):
        with tempfile.TemporaryDirectory() as root:
            run=fixture();path=save_feature_sample(root,run)
            self.assertEqual(path,save_feature_sample(root,run))
            audit=json.loads((path/'audit.json').read_text())
            self.assertIn('features.parquet',audit['artifacts'])
            self.assertIn('feature_history_daily.parquet',audit['artifacts'])
            self.assertIn('feature_dictionary.json',audit['artifacts'])
            self.assertFalse((path/'monthly.parquet').exists())
            self.assertNotIn('NaN',(path/'audit.json').read_text())

    def test_changed_values_under_identical_pins_never_overwrite(self):
        with tempfile.TemporaryDirectory() as root:
            run=fixture();path=save_feature_sample(root,run);before=(path/'features.parquet').read_bytes()
            run.panel.table=pa.table({'source_security_id':['member-a'],'signal_date':['1999-12-31'],'fixture_feature':[1.]})
            with self.assertRaisesRegex(ValueError,'changed'):save_feature_sample(root,run)
            self.assertEqual(before,(path/'features.parquet').read_bytes())

    def test_changed_code_gets_new_artifact(self):
        with tempfile.TemporaryDirectory() as root:
            run=fixture();first=save_feature_sample(root,run)
            run.report['code_provenance']['worker_sha256']['features.py']='new-code'
            self.assertNotEqual(first,save_feature_sample(root,run))

    def test_output_symlink_escape_refused(self):
        with tempfile.TemporaryDirectory() as root,tempfile.TemporaryDirectory() as outside:
            (Path(root)/'outputs').symlink_to(outside,target_is_directory=True)
            with self.assertRaisesRegex(ValueError,'leaves'):save_feature_sample(root,fixture())
            self.assertEqual(list(Path(outside).iterdir()),[])

    def test_failed_or_disabled_notebook_feature_rerun_clears_all_prior_results(self):
        names=('feature_run','feature_history_df','features_df','feature_coverage_df','feature_catalog_df','feature_artifact_path')
        for result in [None,ValueError('feature read failed')]:
            state={name:object() for name in names}
            state.update(catalog=Mock(),settings={},display=Mock(),Markdown=str,
                         run_feature_sample=Mock(side_effect=result) if isinstance(result,Exception) else Mock(return_value=result))
            cell=cell_containing('run_feature_sample(catalog, settings)')
            if isinstance(result,Exception):
                with self.assertRaisesRegex(ValueError,'failed'):exec(cell,state)
            else:exec(cell,state)
            for name in names:self.assertIsNone(state[name])


if __name__=='__main__':unittest.main()
