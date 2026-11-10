"""Ranking query, temporal selection and unchanged RET protocol tests."""
import sys,json,tempfile,unittest
from pathlib import Path
from unittest import mock
import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import learner_experiments as exp
import train_xgb_ret_ensemble as engine
import ret_backtest as base
from xgb_validation_stopping import train_until_validation_stops


def frame(days=40,stocks=12):
    dates=np.repeat(pd.bdate_range('2016-10-01',periods=days),stocks)
    return pd.DataFrame(dict(trade_date=dates,security_id=np.tile([str(i) for i in range(stocks)],days),
        future_oc_return=np.tile(np.linspace(-.01,.01,stocks),days)))


class LearnerProtocolTests(unittest.TestCase):
    def test_all_twenty_four_configurations_include_original_baseline(self):
        grid=exp.config_grid()
        self.assertEqual(len(grid),24)
        self.assertEqual(len({json.dumps(x,sort_keys=True) for x in grid}),24)
        self.assertEqual(grid[0],{k:base.PARAMETERS[k] for k in grid[0]})

    def test_date_split_preserves_complete_queries_and_reproduces(self):
        data=frame();a,b=exp.date_split(data,123);c,d=exp.date_split(data,123)
        np.testing.assert_array_equal(a,c);np.testing.assert_array_equal(b,d)
        self.assertEqual(data.iloc[a].trade_date.nunique(),32)
        self.assertEqual(data.iloc[b].trade_date.nunique(),8)
        self.assertFalse(set(data.iloc[a].trade_date)&set(data.iloc[b].trade_date))
        np.testing.assert_array_equal(np.sort(np.r_[a,b]),np.arange(len(data)))
        self.assertTrue((exp.group_sizes(data.iloc[a])==12).all())

    def test_relevance_retains_true_ties_and_increases_with_return(self):
        data=frame(days=2,stocks=12);data.loc[1,'future_oc_return']=data.loc[0,'future_oc_return']
        y=exp.relevance(data)
        self.assertEqual(y[0],y[1]);self.assertGreater(y[11],y[10])
        self.assertEqual(y[12],0)
        data.loc[0,'future_oc_return']=np.nan
        with self.assertRaises(ValueError):exp.relevance(data)

    def test_percentiles_are_day_local_and_ignore_margin_scale(self):
        dates=np.repeat(['a','b'],4)
        scores=np.array([1,4,2,2,20,10,40,30],float)
        y=exp.percentiles(scores,dates)
        np.testing.assert_allclose(y[:4],[0,1,.5,.5])
        changed=scores.copy();changed[4:]=changed[4:]*100+70
        np.testing.assert_array_equal(y,exp.percentiles(changed,dates))

    def test_symmetric_metric_rewards_both_ends_and_equal_weights_dates(self):
        labels=np.array([0,1,2,3,4,0,1,2],float)
        metric=exp.SymmetricNDCG(labels,[5,3])
        self.assertAlmostEqual(metric(labels),1)
        discount=1/np.log2(3)
        expected=.5*((discount+1)/(5+3*discount)+(discount+1)/(2+discount))
        self.assertAlmostEqual(metric(-labels),expected)
        # Perfect long ordering alone is insufficient if the short order fails.
        corrupted=labels.copy();corrupted[:2]=[1,0]
        self.assertLess(metric(corrupted),1)
        self.assertAlmostEqual(metric(-labels),exp.SymmetricNDCG(np.r_[4-labels[:5],2-labels[5:]],[5,3])(labels))
        self.assertEqual(exp.SymmetricNDCG([0,0,0],[3])([1,2,3]),0)

    def test_ranker_parameters_are_fixed_and_do_not_inherit_ten_classes(self):
        p=exp.ranking_parameters(123)
        self.assertEqual(p['objective'],'rank:pairwise');self.assertNotIn('num_class',p)
        self.assertEqual(p['max_depth'],3);self.assertEqual(p['min_child_weight'],20)
        self.assertEqual(p['seed'],123);self.assertFalse(p['ndcg_exp_gain'])
        self.assertEqual(p['lambdarank_pair_method'],'mean')

    def test_maximize_early_stopping_and_no_implicit_round_limit(self):
        data=xgb.DMatrix(np.arange(60,dtype=np.float32).reshape(30,2),label=np.arange(30)%3,nthread=1)
        scores=iter([.1,.3,.2,.25,.1])
        model,history=train_until_validation_stops(dict(objective='multi:softprob',num_class=3,nthread=1),
            data,data,patience=3,metric='custom_rank',maximize=True,score_function=lambda _:next(scores))
        self.assertEqual(model.best_iteration,1)
        self.assertEqual(history['random20']['custom_rank'],[.1,.3,.2,.25,.1])

    def test_native_pairwise_fit_with_real_query_groups(self):
        rng=np.random.default_rng(4);data=frame(days=40,stocks=12)
        y=exp.relevance(data);x=np.column_stack([y+rng.normal(0,.5,len(y)),rng.normal(size=(len(y),3))]).astype(np.float32)
        ti,vi=exp.date_split(data,123)
        a=xgb.DMatrix(x[ti],label=y[ti],nthread=1);b=xgb.DMatrix(x[vi],label=y[vi],nthread=1)
        a.set_group(exp.group_sizes(data.iloc[ti]));b.set_group(exp.group_sizes(data.iloc[vi]))
        metric=exp.SymmetricNDCG(y[vi],exp.group_sizes(data.iloc[vi]))
        model,h=train_until_validation_stops(exp.ranking_parameters(1),a,b,30,
            metric='symmetric_ndcg3',maximize=True,score_function=metric)
        self.assertEqual(len(h['random20']['symmetric_ndcg3']),model.best_iteration+31)
        self.assertGreaterEqual(float(model.best_score),.8)
        self.assertTrue(np.isfinite(model.predict(b)).all())

    def test_future_outcome_is_not_an_input_to_rank_percentiles(self):
        f=frame(days=2);scores=np.arange(len(f),dtype=float)
        before=exp.percentiles(scores,f.trade_date)
        f['future_oc_return']=-999.;f['realized_class']=999
        np.testing.assert_array_equal(before,exp.percentiles(scores,f.trade_date))

    def test_ranker_summary_keeps_unknown_selected_exits_and_no_fake_return_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp);(run/'folds/toy').mkdir(parents=True)
            f=frame(days=5,stocks=12)
            f['ranking_score']=np.tile(np.linspace(0,1,12),5);f['ret_score']=f.ranking_score
            f['signal_date']=f.trade_date-pd.Timedelta(days=1);f['realized_class']=np.tile(np.arange(12)%10+1,5)
            f['ticker']=f.security_id;f['outcome_status']='ok';f['fold_id']='toy';f['ensemble_members']=100
            f.loc[11,'future_oc_return']=np.nan;f.loc[11,'outcome_status']='unknown'
            f.to_parquet(run/'folds/toy/predictions.parquet',index=False)
            pd.DataFrame({'trade_date':f.trade_date.unique()}).to_parquet(run/'wfo_calendar.parquet',index=False)
            with mock.patch.object(base,'folds',return_value=[{'fold_id':'toy'}]):base.summarize(run,score_type='ranking')
            result=json.loads((run/'RESULTS.json').read_text())
            self.assertIsNone(result['classification']['logloss']);self.assertIsNone(result['classification']['accuracy'])
            self.assertEqual(result['score_type'],'ranking')
            book=pd.read_parquet(run/'portfolio_daily.parquet')
            self.assertTrue(book[book.trade_date.eq(f.trade_date.iloc[0])].gross_return.isna().all())

    def test_generic_checkpoint_verifies_ranker_prediction_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp);p=engine.member_prefix(run,'2017Q1',0);p.parent.mkdir(parents=True)
            p.with_suffix('.ubj').write_bytes(b'model');p.with_suffix('.npz').write_bytes(b'scores')
            info=dict(member=0,fold_id='2017Q1',model_sha256=base.fingerprint(p.with_suffix('.ubj')),
                prediction_sha256=base.fingerprint(p.with_suffix('.npz')))
            p.with_suffix('.json').write_text(json.dumps(info))
            self.assertTrue(engine.checkpoint_complete(run,'2017Q1',0))
            p.with_suffix('.npz').write_bytes(b'changed')
            with self.assertRaises(ValueError):engine.checkpoint_complete(run,'2017Q1',0)

    def test_tuning_selection_and_future_outcome_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp);folder=run/'folds/2017Q1';folder.mkdir(parents=True)
            grid=exp.config_grid();base.save_json(run/'CONFIG.json',{'tuning_grid':grid})
            trials=[dict(candidate=i,split=j,parameters=p,best_validation_logloss=2+i*.001,
                training_last_outcome='2016-12-30',outer_test_start='2017-01-01') for i,p in enumerate(grid) for j in range(3)]
            base.save_json(folder/'TUNING_PROGRESS.json',dict(config_sha256=base.fingerprint(run/'CONFIG.json'),trials=trials))
            selected=dict(fold_id='2017Q1',selected_candidate=0,selected_parameters=grid[0],trials_sha256=base.fingerprint(folder/'TUNING_PROGRESS.json'))
            base.save_json(folder/'TUNING.json',selected)
            self.assertEqual(exp.read_tuning(run,'2017Q1')['selected_candidate'],0)
            trials[0]['training_last_outcome']='2017-01-02'
            base.save_json(folder/'TUNING_PROGRESS.json',dict(config_sha256=base.fingerprint(run/'CONFIG.json'),trials=trials))
            selected['trials_sha256']=base.fingerprint(folder/'TUNING_PROGRESS.json');base.save_json(folder/'TUNING.json',selected)
            with self.assertRaises(ValueError):exp.read_tuning(run,'2017Q1')

    def test_heldout_outcome_poisoning_cannot_change_ranker_forecasts(self):
        rng=np.random.default_rng(11)
        training=frame(days=40);training['realized_class']=np.tile(np.arange(12)%10+1,40)
        testing=frame(days=3);testing['trade_date']+=pd.DateOffset(years=1)
        testing['realized_class']=1
        for name in exp.MODEL_FEATURES:
            training[name]=rng.normal(size=len(training))
            testing[name]=rng.normal(size=len(testing))
        config=dict(variant='ranker',model_features=exp.MODEL_FEATURES)
        with tempfile.TemporaryDirectory() as tmp:
            left,right=Path(tmp)/'left',Path(tmp)/'right';left.mkdir();right.mkdir()
            with mock.patch.object(exp,'verify_source',return_value=config),mock.patch.object(base,'read_fold',return_value=(training,testing)):
                exp.batch_worker(left,'2017Q1',[0])
            changed=testing.copy();changed['future_oc_return']=np.nan;changed['realized_class']=999
            changed['session_close']=-1000.;changed['high']=np.inf;changed['low']=-np.inf
            with mock.patch.object(exp,'verify_source',return_value=config),mock.patch.object(base,'read_fold',return_value=(training,changed)):
                exp.batch_worker(right,'2017Q1',[0])
            a=np.load(engine.member_prefix(left,'2017Q1',0).with_suffix('.npz'))
            b=np.load(engine.member_prefix(right,'2017Q1',0).with_suffix('.npz'))
            np.testing.assert_array_equal(a['raw_scores'],b['raw_scores'])
            np.testing.assert_array_equal(a['percentile_scores'],b['percentile_scores'])

    def test_tuning_checkpoint_reuse_and_ret_member_split_preservation(self):
        rng=np.random.default_rng(21);training=frame(days=40)
        training['realized_class']=np.tile(np.arange(12)%10+1,40)
        testing=frame(days=3);testing['trade_date']+=pd.DateOffset(years=1);testing['realized_class']=1
        for name in exp.MODEL_FEATURES:
            training[name]=rng.normal(size=len(training));testing[name]=rng.normal(size=len(testing))
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp);cfg=dict(variant='ret_tuned',model_features=exp.MODEL_FEATURES,tuning_grid=[exp.config_grid()[0]])
            base.save_json(run/'CONFIG.json',cfg)
            with mock.patch.object(exp,'verify_source',return_value=cfg),mock.patch.object(base,'read_fold',return_value=(training,testing)):
                exp.tuning_worker(run,'2017Q1')
                with mock.patch.object(exp,'train_until_validation_stops',side_effect=AssertionError('Completed tuning must not refit')):
                    exp.tuning_worker(run,'2017Q1')
                exp.batch_worker(run,'2017Q1',[0])
            meta=json.loads(engine.member_prefix(run,'2017Q1',0).with_suffix('.json').read_text())
            seed,_=engine.member_seeds('2017Q1',0);a,b=engine.split_indices(len(training),seed)
            self.assertEqual(meta['split_train_index_sha256'],exp.digest(a))
            self.assertEqual(meta['split_validation_index_sha256'],exp.digest(b))
            with np.load(engine.member_prefix(run,'2017Q1',0).with_suffix('.npz')) as prediction:
                np.testing.assert_allclose(prediction['probabilities'].sum(axis=1),1,atol=1e-6)

    def test_initial_queues_have_two_streams_and_full_ranker_member_sets(self):
        with tempfile.TemporaryDirectory() as tmp:
            suite=Path(tmp)
            with mock.patch.object(engine,'checkpoint_complete',return_value=False):queues=exp.task_queues(suite)
            self.assertEqual(set(queues),{'ret_tuned','ranker'})
            self.assertEqual(len(queues['ret_tuned']),40)
            self.assertTrue(all(j['kind']=='tune' for j in queues['ret_tuned']))
            for fid in {j['fold_id'] for j in queues['ranker']}:
                ids=[m for j in queues['ranker'] if j['fold_id']==fid for m in j['members']]
                self.assertEqual(sorted(ids),list(range(100)))


if __name__=='__main__':unittest.main()
