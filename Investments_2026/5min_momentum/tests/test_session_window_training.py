"""Cohort/target separation, native missing states and resumable orchestration."""
from pathlib import Path
import sys,json,tempfile,unittest
from unittest import mock
import duckdb
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import train_session_window as exp
import train_xgb_ret_ensemble as engine
import ret_backtest as base
from session_window_features import FEATURES


class StrictWindowTrainingTests(unittest.TestCase):
    def test_gate_uses_historical_quality_not_finite_state_or_future_labels(self):
        frame=pd.DataFrame(dict(window_price_complete=[True,True,False,True],
            window_activity_complete=[True,True,True,False],source_consistent=[True,False,True,True],
            complete_feature_vector=[False,True,True,True],future_oc_return=[np.nan,1,2,3]))
        np.testing.assert_array_equal(exp.eligible_history(frame),[True,False,False,False])
        frame['future_oc_return']=-999.;frame['realized_class']=999
        frame['complete_feature_vector']=False
        np.testing.assert_array_equal(exp.eligible_history(frame),[True,False,False,False])

    def test_matrix_allows_declared_native_nans_but_not_inf_or_wrong_columns(self):
        x=np.zeros((4,267),np.float32);x[0,8]=np.nan
        engine.validate_matrix(x,FEATURES)
        x[0,8]=np.inf
        with self.assertRaises(ValueError):engine.validate_matrix(x,FEATURES)
        with self.assertRaises(ValueError):engine.validate_matrix(np.zeros((4,267)),FEATURES[::-1])
        old=engine.EXPERIMENTS['combined15']['features'];a=np.zeros((4,len(old)));a[0,0]=np.nan
        with self.assertRaises(ValueError):engine.validate_matrix(a,old)

    def test_dedicated_controller_prevents_inheriting_legacy_headroom_default(self):
        with self.assertRaisesRegex(ValueError,'dedicated'):
            engine.full_run(Path('.'),Path('.'),experiment='session267')

    def examples(self):
        candidates=pd.DataFrame(dict(security_id=[str(i) for i in range(13)],
            signal_date=pd.Timestamp('2020-01-02'),trade_date=pd.Timestamp('2020-01-03'),
            lag01_open_05m=np.arange(13,dtype=float)))
        anchors=pd.DataFrame(dict(security_id=[str(i) for i in range(12)],signal_date=pd.Timestamp('2020-01-03'),
            opening_rows=1,closing_rows=1,session_open=100.,session_close=np.r_[100.,100.,np.arange(102.,112.)],
            duplicate_rows=0,source_count=1,missing_source_rows=0))
        return candidates,anchors

    def test_unknown_outcomes_remain_candidates_ties_are_preserved(self):
        c,a=self.examples();con=duckdb.connect();con.register('candidates',c);con.register('anchors',a)
        exp.attach_outcomes(con)
        result=con.execute('SELECT * FROM examples LEFT JOIN labels USING(security_id,signal_date) ORDER BY cast(security_id AS INT)').df();con.close()
        self.assertEqual(len(result),13)
        self.assertTrue(pd.isna(result.iloc[-1].future_oc_return));self.assertTrue(pd.isna(result.iloc[-1].realized_class))
        self.assertEqual(result.iloc[0].realized_class,result.iloc[1].realized_class)
        np.testing.assert_allclose(result.iloc[:12].future_oc_return,a.session_close/100-1)

    def test_future_outcome_perturbation_does_not_change_features_or_candidate_keys(self):
        c,a=self.examples();outputs=[]
        for poison in (False,True):
            b=a.copy()
            if poison:b['session_close']=1e8;b.loc[0,'opening_rows']=0
            con=duckdb.connect();con.register('candidates',c);con.register('anchors',b)
            exp.attach_outcomes(con)
            outputs.append(con.execute('SELECT security_id,trade_date,lag01_open_05m FROM examples ORDER BY security_id').df());con.close()
        pd.testing.assert_frame_equal(outputs[0],outputs[1])

    def test_target_status_rejects_ambiguous_forward_anchors_without_replacement(self):
        c,a=self.examples();a.loc[0,'duplicate_rows']=1;a.loc[1,'source_count']=2;a.loc[2,'closing_rows']=0
        con=duckdb.connect();con.register('candidates',c);con.register('anchors',a);exp.attach_outcomes(con)
        result=con.execute('SELECT * FROM examples ORDER BY cast(security_id AS INT)').df();con.close()
        self.assertEqual(result.iloc[0].outcome_status,'duplicate_forward_bar')
        self.assertEqual(result.iloc[1].outcome_status,'ambiguous_forward_source')
        self.assertEqual(result.iloc[2].outcome_status,'missing_or_invalid_close')
        self.assertTrue(result.iloc[:3].future_oc_return.isna().all());self.assertEqual(len(result),13)

    def test_graceful_stop_cannot_finalize_an_incomplete_run(self):
        with tempfile.TemporaryDirectory() as folder:
            suite=Path(folder)
            for variant in exp.VARIANTS:(suite/variant).mkdir()
            (suite/'PRETRAIN_AUDIT.json').write_text(json.dumps({'all_passed':True}))
            (suite/'STOP_REQUESTED.json').write_text('{}')
            with mock.patch.object(exp,'verify',return_value={'workers':10}),mock.patch.object(exp,'checkpoint',return_value=False),mock.patch.object(exp,'finalize') as finalize,mock.patch.object(exp.subprocess,'Popen') as popen:
                exp.run_controller(suite)
            finalize.assert_not_called();popen.assert_not_called()
            self.assertEqual(json.loads((suite/'RUN_STATUS.json').read_text())['status'],'stopped_resumable')

    def test_inspection_rejects_absent_results_without_training(self):
        with tempfile.TemporaryDirectory() as folder:
            project=Path(folder);(project/'data').mkdir()
            with mock.patch.object(engine,'batch_worker') as ret,mock.patch.object(exp.ranking,'batch') as ranker:
                with self.assertRaises(FileNotFoundError):exp.report(project)
            ret.assert_not_called();ranker.assert_not_called()

    def test_authorized_seed_methods_and_uncapped_settings_remain_distinct(self):
        self.assertEqual(len(base.folds()),40)
        self.assertEqual(engine.MEMBERS,100)
        self.assertEqual(base.PARAMETERS['max_depth'],3)
        self.assertEqual(exp.protocol.ranking_parameters(12)['lambdarank_num_pair_per_sample'],8)
        self.assertNotIn('num_class',exp.protocol.ranking_parameters(12))
        self.assertEqual(engine.member_seeds('2017Q1',0),engine.member_seeds('2017Q1',0))
        self.assertNotEqual(engine.member_seeds('2017Q1',0),engine.member_seeds('2017Q1',1))


if __name__=='__main__':unittest.main()
