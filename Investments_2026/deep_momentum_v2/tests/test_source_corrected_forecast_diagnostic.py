"""Synthetic-only checks. No actual forecasts are read during model fits."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import importlib.util
_spec = importlib.util.spec_from_file_location('_corrected_delta_tests', Path(__file__).resolve().parents[1] / 'scripts/compare_source_corrected_forecasts_v2.py')
d = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(d)

def fixture():
    clock={'selection_sessions':pd.bdate_range('2019-12-02','2023-12-29').strftime('%Y-%m-%d').tolist()}
    window=d.windows(clock);rows=[]
    for signal,w in window.items():
        year=int(w['outcome_month'][:4])
        for sid in ('a','b'):
            rows.append({'source_security_id':sid,'signal_date':signal,'intrinio_security_id':'native_'+sid,**w,
                'prediction_eligible':True,'paper_prediction_eligible':True,'identity_eligible':True,'requested':True,
                'identity_status':'resolved','source_identity_status':'needs_review','identity_eligibility_policy':'provisional_unique_resolved_link',
                'identity_review_required':True,'resolution_status':'resolved','fit_year':year,'training_outcome_start':f'{year-10}-01-01',
                'training_outcome_end':f'{year-1}-12-31','training_label_observed_through':{2020:'2019-12-31',2021:'2020-12-31',2022:'2021-12-31',2023:'2022-12-30'}[year]})
    metadata=d.forecast_metadata(pd.DataFrame(rows),window)
    selected=[];monthly=[]
    for signal,w in window.items():
        for side,sid,k in [('long','a',1),('short','b',10)]:
            selected.append({'source_security_id':sid,'signal_date':signal,'intrinio_security_id':'native_'+sid,'outcome_month':w['outcome_month'],
                'execution_date':w['outcome_start'],'side':side,'prediction_eligible':True,'selected':True,'requested_top_k':10,'selected_names_per_side':1,
                'predicted_class':k,'intended_weight':.5 if side=='long' else -.5,'trade_eligible':True,'execution_restricted':False,'retrospective_execution_restriction':True})
        monthly.append({'signal_date':signal,'execution_date':w['outcome_start'],'outcome_month':w['outcome_month'],'original_rows':2,'requested_top_k':10,
            'long_candidates':1,'short_candidates':1,'long_selected':1,'short_selected':1,'selected_names_per_side':1,'effective_per_name_abs_weight':.5})
    return metadata,pd.DataFrame(selected),pd.DataFrame(monthly),window

class DiagnosticChecks(unittest.TestCase):
    def test_identical_forecasts_have_zero_change(self):
        metadata,_,_,_=fixture();p=np.repeat(np.array([[.3]+[.7/9]*9]),len(metadata),axis=0)
        summary,rows=d.probability_changes(p,p.copy(),metadata,'rf_extended')
        self.assertEqual(len(summary),53);self.assertEqual(len(rows),96)
        self.assertTrue(summary.mean_absolute_probability_change.eq(0).all());self.assertTrue(summary.mode_changed_fraction.eq(0).all())

    def test_probability_distances_and_symmetric_mode_changes(self):
        metadata,_,_,_=fixture();p=np.full((len(metadata),10),.05);p[:,0]=.55;q=p.copy();q[:,0]=.05;q[:,9]=.55
        summary,rows=d.probability_changes(p,q,metadata,'rf_extended');allrow=summary.loc[summary.period.eq('all')].iloc[0]
        self.assertAlmostEqual(allrow.mean_absolute_probability_change,.1);self.assertAlmostEqual(allrow.mean_total_variation_distance,.5)
        self.assertEqual(allrow.mode_changed_fraction,1.);self.assertTrue(rows.corrected_mode_class.eq(10).all())

    def test_first_class_wins_exact_mode_tie(self):
        metadata,_,_,_=fixture();p=np.full((len(metadata),10),.1);_,rows=d.probability_changes(p,p,metadata,'rf_extended')
        self.assertTrue(rows.old_mode_class.eq(1).all())

    def test_invalid_probabilities_rejected(self):
        metadata,_,_,_=fixture();p=np.full((len(metadata),10),.1);p[0,0]=1.
        with self.assertRaisesRegex(ValueError,'ten-class'):d.probability_changes(p,p,metadata,'rf_extended')

    def test_within_version_arm_cohort_mismatch_rejected(self):
        metadata,_,_,_=fixture();other=metadata.copy();other.iloc[0,other.columns.get_loc('prediction_eligible')]=False
        with self.assertRaisesRegex(ValueError,'admission'):d.same_cohort(metadata,other)
        with self.assertRaisesRegex(ValueError,'keys changed'):d.same_cohort(metadata,metadata.iloc[:-1])

    def test_cross_version_eligibility_and_route_changes_reported(self):
        metadata,_,_,_=fixture();other=metadata.copy();other.iloc[0,other.columns.get_loc('prediction_eligible')]=False
        other.iloc[0,other.columns.get_loc('paper_prediction_eligible')]=False;other.iloc[1,other.columns.get_loc('intrinio_security_id')]='repaired_native'
        changes=d.cohort_changes(metadata,other,'rf_extended')
        self.assertEqual(changes.loc[changes.field.eq('prediction_eligible'),'changed_rows'].iloc[0],1)
        self.assertEqual(changes.loc[changes.field.eq('intrinio_security_id'),'changed_rows'].iloc[0],1)
        changed_clock=other.copy();changed_clock.iloc[0,changed_clock.columns.get_loc('outcome_end')]='2020-02-03'
        with self.assertRaisesRegex(ValueError,'chronology changed'):d.cohort_changes(metadata,changed_clock,'rf_extended')

    def test_probability_comparison_preserves_nonjoint_rows_without_imputation(self):
        metadata,_,_,_=fixture();old_meta=metadata.copy();new_meta=metadata.copy()
        old_meta.iloc[[2,3],old_meta.columns.get_indexer(['prediction_eligible','paper_prediction_eligible'])]=False
        new_meta.iloc[[1,3],new_meta.columns.get_indexer(['prediction_eligible','paper_prediction_eligible'])]=False
        p=np.full((len(metadata),10),.1);q=p.copy();p[~old_meta.prediction_eligible.to_numpy()]=np.nan;q[~new_meta.prediction_eligible.to_numpy()]=np.nan
        summary,rows=d.probability_changes(p,q,old_meta,'rf_extended',corrected_metadata=new_meta);allrow=summary.loc[summary.period.eq('all')].iloc[0]
        self.assertEqual((allrow.original_rows,allrow.joint_prediction_rows,allrow.old_only_prediction_rows,allrow.corrected_only_prediction_rows,allrow.neither_prediction_rows),(96,93,1,1,1))
        nonjoint=rows.loc[rows.probability_comparison_status.ne('joint')]
        self.assertTrue(nonjoint.mean_absolute_probability_change.isna().all());self.assertTrue(nonjoint.mode_changed.isna().all())
        self.assertTrue(rows.loc[rows.probability_comparison_status.eq('corrected_only'),'old_mode_class'].isna().all())

    def test_zero_joint_periods_keep_grid_and_null_distance_denominators(self):
        metadata,_,_,_=fixture();old_meta=metadata.copy();new_meta=metadata.copy();a=metadata.index.get_level_values('source_security_id')=='a'
        for frame,eligible in ((old_meta,a),(new_meta,~a)):
            frame['prediction_eligible']=eligible;frame['paper_prediction_eligible']=eligible
        p=np.full((len(metadata),10),.1);q=p.copy();p[~a]=np.nan;q[a]=np.nan
        summary,rows=d.probability_changes(p,q,old_meta,'rf_extended',corrected_metadata=new_meta)
        self.assertEqual(len(summary),53);self.assertTrue(summary.joint_prediction_rows.eq(0).all())
        self.assertTrue(summary.mode_changed_fraction.isna().all());self.assertTrue(summary.mean_absolute_probability_change.isna().all())
        self.assertEqual(summary.loc[summary.period.eq('all'),'old_only_prediction_rows'].iloc[0],48)

    def test_nullable_audit_review_flag_preserved_only_off_prediction_population(self):
        metadata,_,_,window=fixture();frame=metadata.reset_index()
        frame.loc[0,['prediction_eligible','paper_prediction_eligible']]=False
        frame['identity_review_required']=frame.identity_review_required.astype(object);frame.loc[0,'identity_review_required']=None
        result=d.forecast_metadata(frame,window);self.assertEqual(result.identity_review_required.isna().sum(),1)
        summary,cases=d.metadata_missingness(result,'synthetic','rf_extended')
        review=summary.loc[summary.field.eq('identity_review_required')].iloc[0]
        self.assertEqual((review.missing_original_rows,review.missing_predicted_rows,review.missing_nonpredicted_rows),(1,0,1))
        self.assertEqual(cases.loc[cases.field.eq('identity_review_required'),'raw_value_state'].tolist(),['null'])

    def test_unknown_review_flag_on_predicted_row_rejected(self):
        metadata,_,_,window=fixture();frame=metadata.reset_index();frame['identity_review_required']=frame.identity_review_required.astype(object)
        frame.loc[0,'identity_review_required']=None
        with self.assertRaisesRegex(ValueError,'unknown identity review'):d.forecast_metadata(frame,window)

    def test_primary_prediction_flag_cannot_be_nullable_or_integer(self):
        for value in (None,1):
            metadata,_,_,window=fixture();frame=metadata.reset_index();frame['prediction_eligible']=frame.prediction_eligible.astype(object)
            frame.loc[0,'prediction_eligible']=value
            with self.assertRaisesRegex(ValueError,'not explicitly Boolean'):d.forecast_metadata(frame,window)

    def test_positive_predicted_identity_request_and_native_are_required(self):
        for field,value,message in [('identity_eligible',False,'positive identity/request'),('requested',False,'positive identity/request'),('intrinio_security_id',None,'valid native identity')]:
            metadata,_,_,window=fixture();frame=metadata.reset_index();frame.loc[0,field]=value
            with self.assertRaisesRegex(ValueError,message):d.forecast_metadata(frame,window)

    def test_nonboolean_nonnull_review_flag_rejected_even_if_not_predicted(self):
        metadata,_,_,window=fixture();frame=metadata.reset_index();frame.loc[0,['prediction_eligible','paper_prediction_eligible']]=False
        frame['identity_review_required']=frame.identity_review_required.astype(object);frame.loc[0,'identity_review_required']='unknown'
        with self.assertRaisesRegex(ValueError,'not explicitly Boolean'):d.forecast_metadata(frame,window)

    def test_selected_overlap_denominators(self):
        _,old,_,window=fixture();new=old.copy();new.loc[new.side.eq('long'),'source_security_id']='replacement'
        overlap=d.selection_overlap(old,new,'rf_extended','class_1_10',window)
        self.assertEqual(len(overlap),96);self.assertTrue(overlap.loc[overlap.side.eq('long'),'jaccard'].eq(0).all())
        self.assertTrue(overlap.loc[overlap.side.eq('short'),'jaccard'].eq(1).all())
        self.assertTrue(overlap.loc[overlap.side.eq('short'),'fraction_old_names_retained'].eq(1).all())

    def test_selection_checks_saved_mode_and_equal_sides(self):
        metadata,selected,monthly,window=fixture();mode=pd.Series(np.where(metadata.index.get_level_values('source_security_id')=='a',1,10),index=metadata.index)
        selected['predicted_class']=selected.predicted_class.astype('Int64')
        d.check_selected(selected,monthly,metadata,mode,window,'class_1_10')
        changed=selected.copy();changed.loc[0,'intended_weight']=.6
        with self.assertRaisesRegex(ValueError,'50/50'):d.check_selected(changed,monthly,metadata,mode,window,'class_1_10')

    def test_pooled_class_basket_rejected(self):
        metadata,selected,monthly,window=fixture();mode=pd.Series(np.where(metadata.index.get_level_values('source_security_id')=='a',1,10),index=metadata.index)
        with self.assertRaisesRegex(ValueError,'pooled or changed'):d.check_selected(selected,monthly,metadata,mode,window,'class_2_9')

    def test_protected_clock_rejected(self):
        with self.assertRaisesRegex(ValueError,'approved forecast horizon'):d.windows({'selection_sessions':['2019-12-31','2024-01-02']})

    def test_early_gate_failure_does_not_load_new_reporter_results(self):
        with tempfile.TemporaryDirectory() as td:
            output=Path(td)/'out'
            with patch.object(d,'reporter',return_value=object()),patch.object(d,'old_gates',side_effect=ValueError('old gate not complete')):
                with self.assertRaisesRegex(ValueError,'old gate not complete'):d.run(td,'no-bindings','0','no-snapshot','0','no-reporter',output)
            self.assertFalse(output.exists())

    def test_read_budget_counts_repeated_key_projections(self):
        class Pins:
            @staticmethod
            def pinned(base,pin):return Path(base)/pin['path']
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'tiny.parquet';pd.DataFrame({'key':['a','b'],'value':[.1,.2]}).to_parquet(path,index=False)
            plan=d.file_plan(td,{'path':path.name},['key','value'],2,Pins,['key'],['key','value'])
            self.assertEqual(plan['projected_compressed_bytes'],plan['metadata_projected_compressed_bytes']+plan['numerical_projected_compressed_bytes'])
            self.assertGreater(plan['projected_compressed_bytes'],plan['numerical_projected_compressed_bytes'])

    def test_model_metadata_uses_actual_arm_descriptor_schema(self):
        class Pins:
            @staticmethod
            def pinned(base,pin):return Path(base)/pin['path']
        with tempfile.TemporaryDirectory() as td:
            run=Path(td);models=run/'models';models.mkdir();audit={'build_identity_sha256':'root','model_identity_sha256':'phase','folds':{}}
            for year in (2020,2021,2022,2023):
                fold=models/str(year);fold.mkdir();value={'status':'COMPLETE','test_year':year,'build_identity_sha256':'root','model_identity_sha256':'phase','arms':{}}
                for arm in d.ARMS:
                    state={'status':'COMPLETE','arm':{'arm_id':arm,'feature_set':'synthetic_fixture','estimator':'synthetic_fixture'},'build_identity_sha256':'root','model_identity_sha256':'phase','metadata':{'synthetic':True}}
                    (fold/(arm+'.json')).write_text(json.dumps(state));value['arms'][arm]={'path':arm+'.json'}
                (fold/'audit.json').write_text(json.dumps(value));audit['folds'][str(year)]={'path':str(year)+'/audit.json'}
            metadata,pins=d.model_metadata({'run':run,'audits':{'models':audit}},Pins)
            self.assertEqual(len(metadata),16);self.assertEqual(len(pins),20)

    def test_fit_parameter_changes_rejected_training_input_changes_allowed(self):
        old={}
        for year in (2020,2021,2022,2023):
            for arm in d.ARMS:
                old[year,arm]={**{name:False for name in ('early_stopping_used','parameter_search_performed','probability_calibration_performed','test_labels_supplied_to_estimator','test_labels_used_for_fitting')},
                    'explicit_parameters':{'random_state':1729},'effective_parameters':{'random_state':1729},'package':'synthetic','package_version':'1',
                    'estimator_kind':arm,'native_class_ids':list(range(10)),'native_to_canonical_mapping':{str(k):k+1 for k in range(10)},
                    'probability_class_ids':list(range(1,11)),'probability_extraction':'synthetic','train_start':f'{year-10}-01-01','train_end':f'{year-1}-12-31',
                    'label_observed_through':f'{year-1}-12-31','matrix_width':30,'test_year':year,'_arm_descriptor':{'arm_id':arm},'fit_count':1,
                    'train_rows':100,'fit_keys_sha256':'old','preprocessor_fingerprint':'old','prepared_prediction_matrix_sha256':'old','prepared_train_matrix_sha256':'old'}
        new=copy.deepcopy(old);new[2020,'rf_extended'].update(train_rows=101,matrix_width=31,fit_keys_sha256='new',preprocessor_fingerprint='new',prepared_train_matrix_sha256='new')
        comparison=d.compare_fit_metadata(old,new);self.assertEqual(len(comparison),16)
        self.assertEqual(comparison.loc[comparison.test_year.eq(2020)&comparison.arm.eq('rf_extended'),'training_rows_change'].iloc[0],1)
        self.assertEqual(comparison.loc[comparison.test_year.eq(2020)&comparison.arm.eq('rf_extended'),'matrix_width_change'].iloc[0],1)
        new[2020,'rf_extended']['explicit_parameters']['random_state']=7
        with self.assertRaisesRegex(ValueError,'parameters/schema'):d.compare_fit_metadata(old,new)

    def test_accounting_rule_change_rejected(self):
        cfg={'backtest':{'cost_bps':[0,5,10,25],'variants':{'unstopped':None,'asset_stop_10pct':.1}},'model_recipe':{'seed':1729}}
        original={'root':{'configuration':cfg},'model':{'configuration_sha256':'cfg','runtime':{},'workers':{}},
            'backtest':{'configuration_sha256':'cfg','clock':{},'basket_policy':{'weights':'50/50'},'runtime':{},'workers':{}}}
        d.compare_rules(original,copy.deepcopy(original));changed=copy.deepcopy(original);changed['backtest']['basket_policy']['weights']='60/40'
        with self.assertRaisesRegex(ValueError,'Accounting changed'):d.compare_rules(original,changed)

if __name__=='__main__':unittest.main(verbosity=2)
