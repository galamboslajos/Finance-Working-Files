"""Portfolio goal, tied-ID, missing-exit and split-preservation checks."""
from pathlib import Path
import sys,json,tempfile,unittest
from unittest import mock
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
import pandas as pd
import ret_backtest as base
import learner_experiments as old
import train_xgb_ret_ensemble as engine
import train_ranker_portfolio as corrected
from portfolio_stopping import PortfolioStopping,METRIC

def fixture(days=4,stocks=12):
    return pd.DataFrame(dict(trade_date=np.repeat(pd.bdate_range('2016-10-01',periods=days),stocks),
        security_id=np.tile([f'id_{i:02}' for i in range(stocks)],days),
        future_oc_return=np.tile(np.linspace(-.02,.03,stocks),days),outcome_status='ok'))

class PortfolioGoal(unittest.TestCase):
    def test_ties_use_disjoint_extremes_and_are_row_order_invariant(self):
        ids=np.array(['F','C','A','E','B','D']);s=np.zeros(6)
        for n in (2,3):
            a,b=base.select_portfolio_indices(s,ids,n)
            self.assertEqual(ids[a].tolist(),sorted(ids)[:n]);self.assertEqual(ids[b].tolist(),sorted(ids)[-n:])
            self.assertFalse(set(a)&set(b))
            p=np.array([4,2,0,5,1,3]);c,d=base.select_portfolio_indices(s[p],ids[p],n)
            np.testing.assert_array_equal(ids[a],ids[p][c]);np.testing.assert_array_equal(ids[b],ids[p][d])

    def test_shared_selector_matches_historical_pandas_order(self):
        rng=np.random.default_rng(13)
        ids=np.array([f'ID{i:03}' for i in range(100)])[rng.permutation(100)]
        scores=rng.integers(-3,4,100).astype(float)
        reference=pd.DataFrame(dict(score=scores,id=ids)).sort_values(['score','id'],ascending=[False,True])
        for n in (2,3):
            a,b=base.select_portfolio_indices(scores,ids,n)
            self.assertEqual(a.tolist(),reference.index[:n].tolist());self.assertEqual(b.tolist(),reference.index[-n:].tolist())

    def test_selector_rejects_invalid_inputs(self):
        with self.assertRaises(ValueError):base.select_portfolio_indices([0]*6,['A']*6,2)
        with self.assertRaises(ValueError):base.select_portfolio_indices([np.nan]*6,list('ABCDEF'),2)
        with self.assertRaises(ValueError):base.select_portfolio_indices([0]*3,list('ABC'),2)

    def test_stopping_matches_final_accounting_and_both_N_equal_weights(self):
        f=fixture();scores=np.tile(np.arange(12),4)
        criterion=PortfolioStopping(f);value=criterion(scores)
        f['ret_score']=scores;calendar=f[['trade_date']].drop_duplicates();means=[]
        for n in (2,3):
            b,_=base.portfolio_days(f,calendar,n);means.append(b.gross_return.mean())
            self.assertAlmostEqual(criterion.history[0][f'mean_gross_n{n}'],means[-1])
        self.assertAlmostEqual(value,.5*sum(means))
        self.assertEqual(criterion.history[0]['fixed_scoring_dates'],4)

    def test_equal_weight_metric_ignores_internal_selected_name_order(self):
        f=fixture();s=np.tile(np.arange(12,dtype=float),4);changed=s.copy()
        # Swap top two and bottom two; the selected sets for BOTH N stay fixed.
        for a in range(0,len(s),12):changed[a:a+2]=changed[a:a+2][::-1];changed[a+10:a+12]=changed[a+10:a+12][::-1]
        c=PortfolioStopping(f)
        self.assertAlmostEqual(c.evaluate(s,record=False),c.evaluate(changed,record=False))

    def test_return_units_are_decimal_and_scale_linearly(self):
        f=fixture();s=np.arange(len(f),dtype=float);a=PortfolioStopping(f)(s)
        f.future_oc_return*=100
        self.assertAlmostEqual(PortfolioStopping(f)(s),a*100)

    def test_unknown_outcomes_never_remove_candidate_or_change_fixed_mask_by_score(self):
        f=fixture();f.loc[0,'future_oc_return']=np.nan;f.loc[0,'outcome_status']='unknown'
        c=PortfolioStopping(f);s=np.tile(np.arange(12,dtype=float),4);s[0]=100
        a=c(s);first=pd.DataFrame(c.daily_history);self.assertTrue(np.isnan(first.gross_n2.iloc[0]));self.assertEqual(first.missing_selected_n2.iloc[0],1)
        changed=s.copy();changed[0]=5.5;b=c(changed)
        self.assertEqual(c.history[0]['fixed_scoring_dates'],3);self.assertEqual(c.history[1]['fixed_scoring_dates'],3)
        self.assertEqual(c.history[0]['excluded_fixed_dates'],1);self.assertAlmostEqual(a,b)
        self.assertEqual(c.history[0]['unresolved_selected_dates_n2'],1);self.assertEqual(c.history[1]['unresolved_selected_dates_n2'],0)
        self.assertEqual(c.coverage.forecast_candidates.iloc[0],12)

    def test_incomplete_date_never_zero_fills_or_replaces_selected_unknown(self):
        f=fixture();f.loc[11,'future_oc_return']=np.nan;f.loc[11,'outcome_status']='unknown'
        s=np.tile(np.arange(12),4);c=PortfolioStopping(f);c(s)
        d=pd.DataFrame(c.daily_history)
        self.assertTrue(np.isnan(d.gross_n2.iloc[0]));self.assertTrue(np.isnan(d.net_1bp_n2.iloc[0]))
        self.assertFalse(d.fixed_stopping_date.iloc[0])

    def test_inspection_never_builds_or_fits_missing_run(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(corrected,'batch',side_effect=AssertionError('No fits')):
            with self.assertRaises(FileNotFoundError):corrected.report(Path(tmp))

    def test_native_worker_retains_original_splits_full_candidates_trace_and_resume(self):
        rng=np.random.default_rng(27);training=fixture(days=40)
        training['realized_class']=np.tile(np.arange(12)%10+1,40)
        testing=fixture(days=3);testing.trade_date+=pd.DateOffset(years=1);testing['realized_class']=1
        for name in old.MODEL_FEATURES:
            training[name]=rng.normal(size=len(training));testing[name]=rng.normal(size=len(testing))
        split_seed,seed=engine.member_seeds('2017Q1',0);ti,vi=old.date_split(training,split_seed)
        extra=training.iloc[[vi[0]]].copy();extra['security_id']='unknown_candidate';extra['future_oc_return']=np.nan
        extra['realized_class']=np.nan;extra['outcome_status']='unknown'
        full=pd.concat([training,extra],ignore_index=True).sort_values(['trade_date','security_id']).reset_index(drop=True)
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp)/'new';reference=Path(tmp)/'old';run.mkdir();reference.mkdir()
            prefix=engine.member_prefix(reference,'2017Q1',0);prefix.parent.mkdir(parents=True)
            old_meta=dict(parameters=old.ranking_parameters(seed),split_train_index_sha256=old.digest(ti),
                split_validation_index_sha256=old.digest(vi),prediction_row_key_sha256=engine.row_key_hash(testing))
            base.save_json(prefix.with_suffix('.json'),old_meta)
            config=dict(model_features=old.MODEL_FEATURES,ranker_v1_reference=str(reference))
            with mock.patch.object(corrected,'verify_source',return_value=config),mock.patch.object(corrected,'read_full_fold',return_value=(training,full,testing)):
                corrected.batch(run,'2017Q1',[0])
                with mock.patch.object(corrected,'train_until_validation_stops',side_effect=AssertionError('Do not refit completed checkpoint')):
                    corrected.batch(run,'2017Q1',[0])
            self.assertTrue(corrected.checkpoint_complete(run,'2017Q1',0))
            p=engine.member_prefix(run,'2017Q1',0);meta=json.loads(p.with_suffix('.json').read_text())
            self.assertEqual(meta['validation_metric'],METRIC);self.assertEqual(meta['parameters'],old_meta['parameters'])
            self.assertEqual(meta['split_train_index_sha256'],old_meta['split_train_index_sha256'])
            self.assertEqual(meta['split_validation_index_sha256'],old_meta['split_validation_index_sha256'])
            self.assertEqual(meta['full_validation_candidate_rows'],len(vi)+1)
            self.assertEqual(meta['rounds_examined'],meta['best_rounds']+30)
            curve=pd.read_parquet(p.with_suffix('.validation_curve.parquet'))
            dates=pd.read_parquet(p.with_suffix('.validation_dates.parquet'))
            self.assertEqual(int(curve.selected_best.sum()),1);self.assertEqual(int(dates.unknown_candidate_outcomes.sum()),1)
            self.assertEqual(curve.fixed_scoring_dates.nunique(),1)
            self.assertAlmostEqual(float(curve.loc[curve.selected_best,'metric'].iloc[0]),meta['best_validation_score'])
            changed=testing.copy();changed['future_oc_return']=np.nan;changed['realized_class']=999
            changed['high']=np.inf;changed['low']=-np.inf;changed['close']=-999.
            other=Path(tmp)/'poisoned';other.mkdir()
            with mock.patch.object(corrected,'verify_source',return_value=config),mock.patch.object(corrected,'read_full_fold',return_value=(training,full,changed)):
                corrected.batch(other,'2017Q1',[0])
            with np.load(p.with_suffix('.npz')) as a,np.load(engine.member_prefix(other,'2017Q1',0).with_suffix('.npz')) as b:
                np.testing.assert_array_equal(a['raw_scores'],b['raw_scores'])
                np.testing.assert_array_equal(a['percentile_scores'],b['percentile_scores'])

if __name__=='__main__':unittest.main()
