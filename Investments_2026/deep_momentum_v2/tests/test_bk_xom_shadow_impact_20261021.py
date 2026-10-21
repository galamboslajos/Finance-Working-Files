import copy
import importlib.util
import json
import math
from pathlib import Path
import sys
import unittest

from dmv2 import bk_xom_shadow_impact as t


def target(source='A',signal='2019-12-31',month='2020-01',value=.1,klass=1,eligible=True,available=True):
 return dict(source_security_id=source,signal_date=signal,outcome_month=month,outcome_start=month+'-02',outcome_end=month+'-31',
  identity_eligible=True,requested=True,paper_prediction_eligible=eligible,prediction_eligible=eligible,outcome_requested=True,
  observed_return_available=available,return_available=available,target_available=available and eligible,
  target_status='available' if available and eligible else 'missing',return_observation_status='ok' if available else 'missing',
  source_return_status='ok' if available else 'missing',label_return=value if available else None,target_class=klass if available and eligible else None)


def ret_fixture(signal='2019-12-31'):
 rows=[]
 for i in range(120):
  month=t.step_month(signal[:7],-119+i)
  for k in range(1,11):
   rows.append(dict(signal_date=signal,outcome_month=month,outcome_end=month+'-28',class_id=k,
    original_member_count=100,prediction_eligible_count=100,member_count=10,class_mean_return=k/1000,status='available'))
 payoff=[dict(signal_date=signal,class_id=k,payoff=math.fsum(r['class_mean_return']/120 for r in rows if r['class_id']==k),status='available') for k in range(1,11)]
 return rows,payoff


class PureImpactTests(unittest.TestCase):
 def test_module_has_no_numeric_imports_at_top_level(self):
  import ast
  tree=ast.parse(Path(t.__file__).read_text())
  modules=[]
  for node in tree.body:
   if isinstance(node,ast.Import):modules.extend(a.name for a in node.names)
   if isinstance(node,ast.ImportFrom):modules.append(node.module)
  self.assertFalse(any(m.startswith(('numpy','pandas','pyarrow','dmv2','sklearn','xgboost')) for m in modules))

 def test_seal_must_explicitly_disable_compute(self):
  with self.assertRaisesRegex(ValueError,'Unsealed'):
   t.execution_seal({},'targets')
  auth=dict(schema='dmv2-shadow-target-impact-execution/v1',status='SEALED_BEFORE_READ',module_sha256=t.sha(t.__file__),allowed_actions=['targets'])
  with self.assertRaisesRegex(ValueError,'cannot fit'):
   t.execution_seal(auth,'targets')

 def test_targets_retain_all_keys_and_missing_targets(self):
  old=[target(),target('B',available=False)];new=copy.deepcopy(old)
  effects,summary=t.compare_target_rows(old,new)
  self.assertEqual(summary['original_rows'],2);self.assertEqual(summary['target_rows_changed'],0)
  self.assertFalse(effects[1]['old_fit_mask'])

 def test_new_observation_changes_masks_and_class_separately(self):
  old=[target(available=False)];new=[target()]
  effects,summary=t.compare_target_rows(old,new)
  self.assertTrue(effects[0]['fit_mask_changed']);self.assertFalse(effects[0]['prediction_mask_changed'])
  self.assertTrue(effects[0]['target_class_changed'])

 def test_new_return_can_change_peer_class_without_peer_return_change(self):
  old=[target('A',klass=1),target('B',klass=2)];new=copy.deepcopy(old);new[1]['target_class']=1
  effects,_=t.compare_target_rows(old,new)
  self.assertTrue(effects[1]['target_class_changed']);self.assertFalse(effects[1]['label_return_changed'])

 def test_drop_original_member_rejected(self):
  with self.assertRaisesRegex(ValueError,'denominator'):
   t.compare_target_rows([target(),target('B')],[target()])

 def test_changed_window_rejected(self):
  old=target();new=copy.deepcopy(old);new['outcome_start']='2020-01-03'
  with self.assertRaisesRegex(ValueError,'windows'):
   t.compare_target_rows([old],[new])

 def test_unknown_boolean_never_passes(self):
  row=target();row['identity_eligible']=None
  with self.assertRaisesRegex(ValueError,'explicit boolean'):
   t.compare_target_rows([row],[row])

 def test_class_availability_mismatch_rejected(self):
  row=target();row['target_class']=None
  with self.assertRaisesRegex(ValueError,'Canonical class'):
   t.compare_target_rows([row],[row])

 def test_vector_shared_finite_and_missing_transition_counts(self):
  stats,changed=t.compare_vector([1.,2.,None,float('nan'),float('inf')],[1.,3.,4.,None,5.])
  self.assertEqual(changed,[1,2,4]);self.assertEqual(stats['shared_finite_changed'],1);self.assertEqual(stats['missing_to_finite'],2)

 def test_numeric_digest_preserves_missing_mask_and_normalizes_signed_zero(self):
  self.assertEqual(t.value_digest([0.0,None],[0,1]),t.value_digest([-0.0,float('nan')],[0,1]))
  self.assertNotEqual(t.value_digest([0.0,1.],[0,1]),t.value_digest([1.,0.0],[0,1]))

 def test_numeric_digest_rejects_bool_and_text(self):
  for value in (True,'1'):
   with self.assertRaisesRegex(ValueError,'numeric'):
    t.value_digest([value],[0])

 def test_partition_added_feature_missingness_does_not_change_prediction_mask(self):
  effects,_=t.compare_target_rows([target()],[target()]);effects[0]['expanded_X_changed']=True
  result=t.partition_impact(effects,start='2020-01-01',end='2020-12-31',feature_set='expanded',role='prediction')
  self.assertEqual(result['old_selected_rows'],result['new_selected_rows']);self.assertEqual(result['selected_mask_changed_rows'],0)
  self.assertTrue(result['selected_inputs_changed'])

 def test_paper_arm_not_affected_by_expanded_only_value(self):
  effects,_=t.compare_target_rows([target()],[target()]);effects[0]['expanded_X_changed']=True
  result=t.partition_impact(effects,start='2020-01-01',end='2020-12-31',feature_set='paper15',role='training')
  self.assertFalse(result['selected_inputs_changed'])

 def test_evaluation_label_change_does_not_change_prediction_inputs(self):
  effects,_=t.compare_target_rows([target(klass=1)],[target(klass=2)])
  result=t.partition_impact(effects,start='2020-01-01',end='2020-12-31',feature_set='expanded',role='prediction')
  self.assertFalse(result['selected_inputs_changed']);self.assertTrue(result['evaluation_labels_changed'])

 def test_ret_exact_equal_month_grid(self):
  rows,payoff=ret_fixture();t.validate_ret_rows(rows,payoff,expected_signals=1)
  result=t.compare_ret_rows(rows,copy.deepcopy(rows),payoff,copy.deepcopy(payoff),expected_signals=1)
  self.assertEqual(result['summary']['changed_class_payoffs'],0)

 def test_ret_missing_calendar_cell_cannot_compress(self):
  rows,payoff=ret_fixture()
  with self.assertRaisesRegex(ValueError,'120×10'):
   t.validate_ret_rows(rows[:-1],payoff,expected_signals=1)

 def test_ret_future_cell_rejected(self):
  rows,payoff=ret_fixture();rows[0]['outcome_end']='2020-01-02'
  with self.assertRaisesRegex(ValueError,'Future|endpoint'):
   t.validate_ret_rows(rows,payoff,expected_signals=1)

 def test_ret_incomplete_whole_vector_withheld(self):
  rows,payoff=ret_fixture();rows[0].update(class_mean_return=None,status='empty_realized_class',member_count=0)
  with self.assertRaisesRegex(ValueError,'Entire incomplete'):
   t.validate_ret_rows(rows,payoff,expected_signals=1)
  for r in payoff:r.update(payoff=None,status='withheld_incomplete_120_month_class_grid')
  t.validate_ret_rows(rows,payoff,expected_signals=1)

 def test_ret_pooled_or_perturbed_payoff_rejected(self):
  rows,payoff=ret_fixture();payoff[0]['payoff']+=1.e-8
  with self.assertRaisesRegex(ValueError,'equal-calendar'):
   t.validate_ret_rows(rows,payoff,expected_signals=1)

 def test_ret_change_is_distinct_from_classifier_retraining(self):
  old,op=ret_fixture();new,np=copy.deepcopy(old),copy.deepcopy(op);new[0]['class_mean_return']+=.1
  np[0]['payoff']=math.fsum(r['class_mean_return']/120 for r in new if r['class_id']==1)
  result=t.compare_ret_rows(old,new,op,np,expected_signals=1)
  self.assertEqual(result['summary']['changed_class_payoffs'],1);self.assertEqual(result['signals'][0]['effect_on_fitted_classifier'],'none_from_RET_alone')




class FoldTests(unittest.TestCase):
 def fixture(self):
  import calendar
  rows=[]
  for i in range(168):
   month=t.step_month('2010-01',i);prior=t.step_month(month,-1);y,m=map(int,month.split('-'));py,pm=map(int,prior.split('-'))
   row=target(signal=f'{prior}-{calendar.monthrange(py,pm)[1]:02d}',month=month)
   row.update(outcome_start=month+'-01',outcome_end=f'{month}-{calendar.monthrange(y,m)[1]:02d}');rows.append(row)
  effects,_=t.compare_target_rows(rows,copy.deepcopy(rows));folds=[]
  for outer in range(2020,2024):
   start=f'{outer-10}-01-01'
   folds.append(dict(outer_outcome_year=outer,train_outcome_start=start,train_outcome_end=f'{outer-1}-12-31',
    inner_splits=[dict(split_id=f'{outer}_validate_{validation}',train_outcome_start=start,train_outcome_end=f'{validation-1}-12-31',
      validation_outcome_start=f'{validation}-01-01',validation_outcome_end=f'{validation}-12-31') for validation in range(outer-3,outer)]))
  hashes={year:{'raw_input_hashes':{side:{'training':{'paper15':str(year),'expanded':str(year)},
    'prediction':{'paper15':str(year),'expanded':str(year)},'training_y':str(year)} for side in ('old','new')}} for year in range(2010,2024)}
  return effects,{'folds':folds},hashes

 def test_exact_12_inner_4_outer_and_4_baseline_arms(self):
  effects,chronology,hashes=self.fixture();report=t.fold_impacts(effects,chronology,annual_summaries=hashes)
  rf=report['rf_tuning_partitions'];base=report['baseline_outer_partitions']
  self.assertEqual(len(rf),32);self.assertEqual(len(base),32)
  self.assertEqual(sum(r['stage']=='inner' for r in rf),24)
  self.assertEqual({r['arm'] for r in base},{'rf_paper15','rf_expanded','xgboost_paper15','xgboost_expanded'})
  self.assertTrue(all(not r['raw_X_fingerprint_changed'] for r in rf+base))

 def test_2016_expanded_training_change_reaches_all_outer_models(self):
  effects,chronology,hashes=self.fixture()
  for row in effects:
   if row['outcome_month']=='2016-01':row['expanded_X_changed']=True
  hashes[2016]['raw_input_hashes']['new']['training']['expanded']='changed'
  result=t.fold_impacts(effects,chronology,annual_summaries=hashes)
  outer=[r for r in result['rf_tuning_partitions'] if r['stage']=='outer' and r['role']=='training']
  self.assertTrue(all(r['selected_inputs_changed'] and r['raw_X_fingerprint_changed'] for r in outer))
  paper=[r for r in result['baseline_outer_partitions'] if r['feature_set']=='paper15' and r['role']=='training']
  self.assertTrue(all(not r['selected_inputs_changed'] and not r['raw_X_fingerprint_changed'] for r in paper))

 def test_2023_label_only_change_does_not_retrain_past_windows(self):
  effects,chronology,hashes=self.fixture()
  for row in effects:
   if row['outcome_month']=='2023-01':row['target_class_changed']=True
  hashes[2023]['raw_input_hashes']['new']['training_y']='changed'
  result=t.fold_impacts(effects,chronology,annual_summaries=hashes)
  training=[r for r in result['rf_tuning_partitions'] if r['role']=='training']
  self.assertTrue(all(not r['selected_inputs_changed'] and not r['observed_y_fingerprint_changed'] for r in training))
  prediction=[r for r in result['rf_tuning_partitions'] if r['stage']=='outer' and r['outer_year']==2023 and r['role']=='prediction'][0]
  self.assertTrue(prediction['evaluation_labels_changed']);self.assertFalse(prediction['selected_inputs_changed'])

 def test_missing_month_cross_section_fails_closed(self):
  effects,chronology,_=self.fixture();effects=[r for r in effects if r['outcome_month']!='2011-02']
  with self.assertRaisesRegex(ValueError,'complete outcome-month'):
   t.fold_impacts(effects,chronology)

 def test_overlapping_label_signal_clock_rejected(self):
  effects,chronology,_=self.fixture()
  for row in effects:
   if row['outcome_month']=='2020-01':row['signal_date']='2019-12-30'
  with self.assertRaisesRegex(ValueError,'overlap prediction signal'):
   t.fold_impacts(effects,chronology)



class TargetMetadataTests(unittest.TestCase):
 def fixture(self):
  import calendar
  rows=[];windows=[]
  for m in range(1,13):
   month=f'2020-{m:02d}';prior=t.step_month(month,-1);py,pm=map(int,prior.split('-'))
   row=target(signal=f'{prior}-{calendar.monthrange(py,pm)[1]:02d}',month=month)
   row.update(outcome_start=month+'-01',outcome_end=f'{month}-{calendar.monthrange(2020,m)[1]:02d}')
   rows.append({k:v for k,v in row.items() if k not in ('label_return','target_class')})
   windows.append({k:row[k] for k in ('signal_date','outcome_month','outcome_start','outcome_end')})
  return rows,{'provenance':{'label_observed_through':'2020-12-31','month_windows':windows}}

 def test_complete_metadata_before_financial_use(self):
  rows,state=self.fixture();self.assertEqual(len(t.validate_target_metadata(rows,state,2020)),12)

 def test_future_signal_rejected_by_metadata_alone(self):
  rows,state=self.fixture();rows[-1]['signal_date']='2024-01-01'
  with self.assertRaisesRegex(ValueError,'protected'):
   t.validate_target_metadata(rows,state,2020)

 def test_duplicate_key_rejected_by_metadata_alone(self):
  rows,state=self.fixture();rows.append(copy.deepcopy(rows[0]))
  with self.assertRaisesRegex(ValueError,'Duplicate'):
   t.validate_target_metadata(rows,state,2020)

 def test_wrong_endpoint_or_independent_calendar_rejected(self):
  rows,state=self.fixture();rows[0]['outcome_end']='2020-02-01'
  with self.assertRaisesRegex(ValueError,'independent window'):
   t.validate_target_metadata(rows,state,2020)

 def test_incomplete_annual_windows_rejected(self):
  rows,state=self.fixture();state['provenance']['month_windows'].pop()
  with self.assertRaisesRegex(ValueError,'complete outcome month'):
   t.validate_target_metadata(rows,state,2020)

if __name__=='__main__':unittest.main()
