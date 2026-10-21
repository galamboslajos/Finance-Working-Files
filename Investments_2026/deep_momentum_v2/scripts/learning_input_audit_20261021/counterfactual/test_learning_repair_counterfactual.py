"""Synthetic-only tests; no research-file loads and no model/backtest calls."""
import copy
import hashlib
import json
import math
import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from learning_repair_counterfactual import (CLASS_IDS, CounterfactualError,
    _reconstruct_classes, annual_dependency_table, classify_repair_label_domain,
    month_step, monthly_class_means, propagate_ret_payoff_impact,
    rerank_month_after_endpoint_repair,verify_preserved_ret_states)


def original_month(*, n=20, missing_repair=True, ties=False):
    rows=[]
    for i in range(n):
        rows.append(dict(source_security_id=f'id{i:03d}', signal_date='2021-10-29',
            outcome_month='2021-11', outcome_start='2021-11-01', outcome_end='2021-11-30',
            identity_eligible=True, requested=True, paper_prediction_eligible=True,
            prediction_eligible=True, outcome_requested=True, target_class=1+i//2,
            label_return=.05 if ties else (n-i)/100, return_available=True,
            observed_return_available=True, target_available=True, target_status='ok',
            return_observation_status='ok', source_return_status='ok'))
    if ties:
        for row in rows: row['target_class']=6
    if missing_repair:
        rows.append({**rows[0], 'source_security_id':'repair', 'label_return':None,
            'target_class':None, 'return_available':False, 'observed_return_available':False,
            'target_available':False, 'target_status':'source_return_status:missing_end_price',
            'return_observation_status':'source_return_status:missing_end_price',
            'source_return_status':'missing_end_price'})
    return pd.DataFrame(rows)


def patch(**kwargs):
    return dict(scope='conditional_shadow_only', same_security_price_only=True,
        uses_cash_or_successor_equity=False, source_security_id='repair',
        signal_date='2021-10-29',start_date='2021-10-29',end_date='2021-11-30',
        repaired_date='2021-11-30',expected_old_label_return=None,
        start_adj_close=100.,end_adj_close=121.,**kwargs)


def shadow(frame=None, change=None):
    return rerank_month_after_endpoint_repair(original_month() if frame is None else frame,
        patch() if change is None else change, observed_through='2023-12-31')


def means_and_signals(change):
    rows=[]
    for i in range(167):
        month=month_step('2010-01',i)
        end=(pd.Timestamp(month+'-01')+pd.offsets.MonthEnd()).strftime('%Y-%m-%d')
        for cls in CLASS_IDS:
            rows.append(dict(outcome_month=month,outcome_end=end,class_id=cls,
                original_member_count=20,prediction_eligible_count=20,member_count=2,
                class_mean_return=(11-cls)/1000,status='available'))
    data=pd.DataFrame(rows)
    data.loc[data.outcome_month.eq('2021-11'),:]=change.monthly_before.to_numpy()
    signals=pd.DataFrame({'signal_date':[(pd.Timestamp(month_step('2019-12',i)+'-01')+
        pd.offsets.MonthEnd()).strftime('%Y-%m-%d') for i in range(48)]})
    fingerprints={s:'sha256:'+'0'*64 for s in signals.signal_date}
    return data,signals,fingerprints


class ConditionalMonthTests(unittest.TestCase):
    def test_exact_manual_old_bins_and_frozen_cohort(self):
        frame=original_month(); prior=frame.copy(deep=True); got=shadow(frame)
        self.assertTrue(got.summary['baseline_numeric_classes_reproduced'])
        self.assertEqual(got.baseline.target_class.iloc[:20].tolist(),[1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10])
        self.assertEqual(got.shadow.target_class.tolist(),[1,2,2,3,3,4,4,5,5,6,6,6,7,7,8,8,9,9,10,10,1])
        self.assertEqual(got.summary['return_rows_changed'],1)
        self.assertGreater(got.summary['class_rows_changed'],1)
        assert_frame_equal(frame,prior)
        self.assertEqual(len(got.shadow),21)
        self.assertFalse(got.summary['production_promoted'])
        self.assertEqual(got.summary['models_fitted'],0)

    def test_exact_ties_use_midrank_no_jitter(self):
        got=shadow(original_month(ties=True), {**patch(),'end_adj_close':105.})
        # 105/100-1 has a different IEEE representation than literal .05;
        # exact equality is deliberate. Use exactly representable endpoints.
        frame=original_month(ties=True); frame.loc[frame.return_available,"label_return"] = .125
        got=shadow(frame,{**patch(),'end_adj_close':112.5})
        self.assertEqual(got.shadow.target_class.tolist(),[6]*21)
        self.assertEqual(got.monthly_after.member_count.tolist(),[0,0,0,0,0,21,0,0,0,0])

    def test_reordering_has_no_class_effect(self):
        a=shadow(); b=shadow(original_month().sample(frac=1,random_state=11))
        assert_frame_equal(a.shadow.sort_values('source_security_id').reset_index(drop=True),
            b.shadow.sort_values('source_security_id').reset_index(drop=True))

    def test_wrong_baseline_class_fails_before_patch(self):
        frame=original_month(); frame.loc[0,'target_class']=10
        with self.assertRaisesRegex(CounterfactualError,'Baseline exact'): shadow(frame)

    def test_non_exhaustive_month_retains_observed_returns(self):
        frame=original_month(); frame.loc[0,'requested']=False
        frame.loc[frame.return_available,'target_status']='non_exhaustive_original_requests'
        frame.target_class=pd.array([pd.NA]*len(frame),dtype='Int64');frame.target_available=False
        got=shadow(frame)
        self.assertEqual(got.summary['baseline']['observed_return_members'],20)
        self.assertFalse(got.shadow.target_available.any())
        self.assertEqual(got.summary['shadow']['rank_status'],'non_exhaustive_original_requests')

    def test_insufficient_population_remains_withheld(self):
        frame=original_month(n=8)
        frame.loc[frame.return_available,'target_status']='insufficient_observed_population'
        frame.target_class=pd.array([pd.NA]*len(frame),dtype='Int64');frame.target_available=False
        got=shadow(frame)
        self.assertEqual(got.summary['shadow']['observed_return_members'],9)
        self.assertFalse(got.shadow.target_available.any())

    def test_endpoint_price_preconditions(self):
        for update in ({'expected_old_label_return':.1},{'start_adj_close':0},
                       {'end_adj_close':math.inf},{'end_adj_close':True},
                       {'repaired_date':'2021-11-15'},{'end_date':'2021-11-29'},
                       {'uses_cash_or_successor_equity':True},
                       {'same_security_price_only':False},{'source_security_id':'new'}):
            with self.subTest(update=update),self.assertRaises(CounterfactualError):
                shadow(change={**patch(),**update})

    def test_noneligible_member_is_not_newly_admitted(self):
        frame=original_month(); frame.loc[len(frame)-1,['identity_eligible','prediction_eligible']]=False
        with self.assertRaises(CounterfactualError):shadow(frame)

    def test_metadata_and_boolean_class_guards(self):
        for field,value in [('outcome_end','2024-01-31'),('requested',1),('target_class',True)]:
            frame=original_month(); frame[field]=frame[field].astype(object);frame.loc[0,field]=value
            with self.subTest(field=field),self.assertRaises(CounterfactualError):shadow(frame)

    def test_domain_november_end_december_anchor_and_wbd_interior(self):
        windows=pd.DataFrame([dict(signal_date='2021-10-29',outcome_start='2021-11-01',outcome_end='2021-11-30',outcome_month='2021-11'),
            dict(signal_date='2021-11-30',outcome_start='2021-12-01',outcome_end='2021-12-31',outcome_month='2021-12'),
            dict(signal_date='2022-03-31',outcome_start='2022-04-01',outcome_end='2022-04-29',outcome_month='2022-04')])
        got=classify_repair_label_domain([dict(repair_id='RCM',date='2021-11-30',same_security_price_only=True),
            dict(repair_id='WBD',date='2022-04-05',same_security_price_only=True),
            dict(repair_id='ADS',date='2021-11-15',same_security_price_only=False)],windows)
        self.assertEqual(got[got.repair_id.eq('RCM')].role.tolist(),['outcome_month_end','prior_month_end_anchor'])
        self.assertEqual(got[got.repair_id.eq('WBD')].classification.tolist(),['no_endpoint_label_change'])
        self.assertEqual(got[got.repair_id.eq('ADS')].classification.tolist(),['requires_separate_economic_return_policy'])
        self.assertFalse(got.automatic_label_change.any())


class PayoffTests(unittest.TestCase):
    def baseline_states(self,data,signals):
        lookup={(r['outcome_month'],r['class_id']):r for r in data.to_dict('records')}
        states={}
        for signal in signals.signal_date:
            months=[month_step(signal[:7],i) for i in range(-119,1)]
            monthly=[lookup[(m,k)] for m in months for k in CLASS_IDS]
            table=[{'class_id':k,'payoff':math.fsum(lookup[(m,k)]['class_mean_return']/120 for m in months),
                'status':'available'} for k in CLASS_IDS]
            provenance={'signal_date':signal,'label_observed_through':signal,
                'first_outcome_month':months[0],'last_outcome_month':months[-1],'vector_available':True}
            payload={'table':table,'monthly':monthly,'provenance':provenance}
            digest='sha256:'+hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),
                ensure_ascii=False,allow_nan=False).encode()).hexdigest()
            states[signal]={'provenance':provenance,'fingerprint':digest}
        return states

    def test_all_saved_vector_logical_fingerprints_independently_reproduce(self):
        change=shadow();data,signals,_=means_and_signals(change)
        states=self.baseline_states(data,signals)
        result=verify_preserved_ret_states(data,states)
        self.assertEqual(len(result),48)
        self.assertEqual(result,{s:v['fingerprint'] for s,v in states.items()})

    def test_changed_mean_or_provenance_fails_old_vector_fingerprint(self):
        change=shadow();data,signals,_=means_and_signals(change)
        states=self.baseline_states(data,signals)
        data.loc[0,'class_mean_return']+=.00001
        with self.assertRaisesRegex(CounterfactualError,'logical fingerprint'):
            verify_preserved_ret_states(data,states)
        data,signals,_=means_and_signals(change)
        states=copy.deepcopy(states);states[signals.signal_date.iloc[0]]['provenance']['new_unpinned_claim']=True
        with self.assertRaisesRegex(CounterfactualError,'logical fingerprint'):
            verify_preserved_ret_states(data,states)

    def test_full_48_vectors_and_causal_delta(self):
        change=shadow();data,signals,fingerprints=means_and_signals(change);prior=data.copy(deep=True)
        got=propagate_ret_payoff_impact(data,[change],signals,baseline_vector_fingerprints=fingerprints)
        self.assertEqual(len(got),480)
        self.assertEqual(got.loc[got.signal_date.lt('2021-11-30'),'payoff_delta'].tolist(),[0.]*230)
        after=got[got.signal_date.ge('2021-11-30')]
        self.assertEqual(len(after),250)
        for row in after.itertuples():
            cls=row.class_id
            before=change.monthly_before.set_index('class_id').loc[cls,'class_mean_return']
            shadow_mean=change.monthly_after.set_index('class_id').loc[cls,'class_mean_return']
            self.assertAlmostEqual(row.payoff_delta,(shadow_mean-before)/120,places=15)
        assert_frame_equal(data,prior)
        self.assertTrue(got.baseline_vector_fingerprint.eq('sha256:'+'0'*64).all())
        self.assertEqual(got.iloc[0].first_outcome_month,'2010-01')
        self.assertEqual(got.iloc[-1].first_outcome_month,'2013-12')

    def test_one_empty_cell_withholds_entire_vector_not_partial_classes(self):
        change=shadow();data,signals,fp=means_and_signals(change)
        mask=data.outcome_month.eq('2010-01') & data.class_id.eq(7)
        data.loc[mask,['status','member_count','class_mean_return']]=['empty_realized_class',0,None]
        got=propagate_ret_payoff_impact(data,[change],signals,baseline_vector_fingerprints=fp)
        self.assertFalse(got.iloc[:10].baseline_vector_available.any())
        self.assertTrue(got.iloc[:10].baseline_payoff.isna().all())
        self.assertTrue(got.iloc[10:20].baseline_vector_available.all())

    def test_missing_month_extra_class_or_uncompleted_month_fails(self):
        change=shadow();data,signals,fp=means_and_signals(change)
        for broken in (data.iloc[1:],pd.concat([data,data.iloc[:1]]),data.assign(outcome_end='2023-12-31')):
            with self.subTest(rows=len(broken)),self.assertRaises(CounterfactualError):
                propagate_ret_payoff_impact(broken,[change],signals,baseline_vector_fingerprints=fp)

    def test_baseline_mean_mismatch_and_duplicate_patch_fail(self):
        change=shadow();data,signals,fp=means_and_signals(change)
        data.loc[data.outcome_month.eq('2021-11'),'class_mean_return']+=.1
        with self.assertRaisesRegex(CounterfactualError,'baseline monthly means'):
            propagate_ret_payoff_impact(data,[change],signals,baseline_vector_fingerprints=fp)
        data,signals,fp=means_and_signals(change)
        with self.assertRaisesRegex(CounterfactualError,'Duplicate monthly'):
            propagate_ret_payoff_impact(data,[change,change],signals,baseline_vector_fingerprints=fp)

    def test_grid_clock_and_fingerprint_required(self):
        change=shadow();data,signals,fp=means_and_signals(change)
        for bad_signals,bad_fp in ((signals.iloc[:-1],fp),(signals.iloc[::-1],fp),(signals,{})):
            with self.assertRaises(CounterfactualError):
                propagate_ret_payoff_impact(data,[change],bad_signals,baseline_vector_fingerprints=bad_fp)

    def test_same_month_uncompleted_signal_is_rejected(self):
        change=shadow();data,signals,fp=means_and_signals(change)
        signals=signals.copy();signals.loc[23,'signal_date']='2021-11-29'
        fp={s:'sha256:'+'0'*64 for s in signals.signal_date}
        with self.assertRaisesRegex(CounterfactualError,'Uncompleted outcome'):
            propagate_ret_payoff_impact(data,[change],signals,baseline_vector_fingerprints=fp)


class DependencyTests(unittest.TestCase):
    def test_date_only_train_validation_evaluation_roles(self):
        protocol={'chronology':{'folds':[dict(outer_outcome_year=2021,train_outcome_start='2011-01-01',
            train_outcome_end='2020-12-31',inner_splits=[]),dict(outer_outcome_year=2022,
            train_outcome_start='2012-01-01',train_outcome_end='2021-12-31',inner_splits=[dict(
            split_id='validate2021',train_outcome_start='2012-01-01',train_outcome_end='2020-12-31',
            validation_outcome_start='2021-01-01',validation_outcome_end='2021-12-31')])]}}
        got=annual_dependency_table(['2021-11'],protocol)
        self.assertEqual(got.dependency_role.tolist(),['evaluation_labels_only','supervised_training_labels','inner_validation_labels_only'])
        self.assertFalse(got.fits_or_predictions_performed.any())
        self.assertFalse(got.all_models_need_refit_claim.any())

if __name__=='__main__':unittest.main()
