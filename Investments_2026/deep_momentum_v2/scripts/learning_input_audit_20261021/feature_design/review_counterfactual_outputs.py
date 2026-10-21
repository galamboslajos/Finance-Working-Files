"""Independent bounded review of new shadow outputs only, no source-price reads."""
from pathlib import Path
from collections import Counter
import hashlib,json,math,time

S=Path('/private/tmp/dmv2-learning-impact-20261021'); C=S/'counterfactual'; D=C/'result-v1'; O=S/'feature_design'
def need(ok,msg):
 if not ok: raise ValueError(msg)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def pin(path):return dict(path=str(path),size_bytes=path.stat().st_size,sha256=sha(path))
plan=json.loads((C/'numeric_read_plan_v2.json').read_text())
need(sha(C/'numeric_read_plan_v2.json')=='69f3782916305c12dd1fd2d6287fe872d24b50170b19f143403e62d644991983','Frozen numeric plan changed')
summary=json.loads((D/'summary.json').read_text())
need(summary['numeric_read_plan_sha256']==sha(C/'numeric_read_plan_v2.json') and summary['models_fitted']==summary['preprocessors_fitted']==summary['new_inference_calls']==0,'Scope/hash binding')
names=['original_month','conditional_shadow_month','changed_rows','monthly_class_means_before','monthly_class_means_after','causal_ret_vector_deltas','training_and_cv_date_dependencies']
files=[D/(name+'.parquet') for name in names]
amount=sum(4*f.stat().st_size for f in files)
need(amount<4*1024**2,'Bounded saved-output projection only')
reservation={'schema':'dmv2-learning-shadow-independent-output-read/v1','scope':'newly_generated_shadow_outputs_only',
 'reserved_file_upper_bound_bytes':amount,'basis':'four_times_whole_file_for_two_hashes_keys/full_numeric_conversion',
 'financial_production_files_opened':False,'new_source_quote_reads':False,'new_fits':0,'files':[dict(path=str(f),size_bytes=f.stat().st_size) for f in files]}
reserve_path=O/'counterfactual_output_review_reservation.json';need(not reserve_path.exists(),'Preserve prior output review reservation')
reserve_path.write_text(json.dumps(reservation,sort_keys=True,indent=2)+'\n')
original_pins=[pin(p) for p in files]
# Read only the bounded, just-created audit reports after reserving their reads.
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
frames={}
for name,path in zip(names,files):
 pf=pq.ParquetFile(path);need(pf.metadata.num_rows<2000,'Shadow audit output row bound')
 columns=[c for c in ['signal_date','outcome_month','outcome_start','outcome_end','changed_outcome_month'] if c in pf.schema_arrow.names]
 meta=pf.read(columns=columns).to_pandas()
 for col in columns:
  series=meta[col].dropna().astype(str)
  if col=='signal_date':need(series.between('2009-12-01','2023-11-30').all(),'Shadow signal scope')
  else:need(series.str[:4].isin([str(y) for y in range(2010,2024)]).all(),'Shadow outcome scope')
 frames[name]=pf.read().to_pandas()
before,after=frames['original_month'],frames['conditional_shadow_month']
keys=['source_security_id','signal_date'];need(len(before)==len(after) and not before.duplicated(keys).any() and before[keys].equals(after[keys]),'Exact original cohort keys preserved')
need(before.columns.equals(after.columns),'Original target schema preserved')
preserved=keys+['outcome_month','outcome_start','outcome_end','identity_eligible','requested','paper_prediction_eligible','prediction_eligible','outcome_requested']
need(before[preserved].equals(after[preserved]),'Original admission/request/clock states changed')
for frame in (before,after):
 need(frame.outcome_month.eq('2021-11').all() and frame.signal_date.eq('2021-10-29').all(),'Single conditional November month')
 observed=frame.prediction_eligible & frame.return_available
 values=frame.loc[observed,'label_return']
 need(values.map(lambda x:isinstance(x,(float,int)) and math.isfinite(x) and x>=-1).all(),'Finite observed-return rank population')
 # Pandas average ranks independently establish equal-value tie groups; exact
 # integer arithmetic then places each midpoint into the frozen ten bins.
 rank_twice=(values.rank(method='average',ascending=False)*2).astype('int64')
 expected=(1+(10*(rank_twice-1))//(2*len(values))).clip(1,10)
 need(frame.loc[observed,'target_available'].all() and frame.loc[~observed,'target_available'].eq(False).all(),'Observed-only labels')
 need(frame.loc[observed,'target_class'].astype('int64').equals(expected),'Alternative pandas-rank class reproduction')
 need(frame.loc[~observed,'target_class'].isna().all(),'Missing returns stay unlabelled')
changes=frames['changed_rows'];need(len(changes)==6==summary['class_rows_changed'],'Reported six class changes')
def equal(a,b):return pd.isna(a) and pd.isna(b) or not pd.isna(a) and not pd.isna(b) and a==b
changed_return=[i for i in before.index if not equal(before.at[i,'label_return'],after.at[i,'label_return'])]
changed_class=[i for i in before.index if not equal(before.at[i,'target_class'],after.at[i,'target_class'])]
need(len(changed_return)==1==summary['return_rows_changed'] and len(changed_class)==6,'Actual direct/reranked change counts')
need(before.at[changed_return[0],'source_security_id']==plan['original_source_security_id'],'Only original RCM endpoint row changed return')
need(set(changes.source_security_id)==set(before.loc[changed_class,'source_security_id']),'Changed-row ledger exactly matches independent diff')
counts={}
for label,frame,saved in [('before',before,frames['monthly_class_means_before']),('after',after,frames['monthly_class_means_after'])]:
 need(len(saved)==10 and saved.class_id.tolist()==list(range(1,11)),'Complete ten realized class cells')
 assigned=frame.loc[frame.target_available]
 expected_counts=assigned.groupby('target_class').size().to_dict()
 counts[label]={str(int(k)):int(v) for k,v in expected_counts.items()}
 for row in saved.itertuples():
  values=assigned.loc[assigned.target_class.eq(row.class_id),'label_return'].astype(float).tolist()
  need(row.member_count==len(values)>0 and row.status=='available','Independent realized-class support')
  independent=math.fsum(values)/len(values)
  need(math.isclose(row.class_mean_return,independent,rel_tol=1e-14,abs_tol=1e-16),'Independent class-mean arithmetic')
 need(saved.member_count.sum()==frame.target_available.sum(),'Class support sums')
ret=frames['causal_ret_vector_deltas'];need(len(ret)==480 and ret.groupby('signal_date').size().eq(10).all() and ret.signal_date.nunique()==48,'Complete48x10 RET grid')
need(ret.baseline_vector_available.all() and ret.shadow_vector_available.all(),'Both full120month vectors available')
old_fingerprints={state['signal_date']:state['fingerprint'] for state in plan['old_vectors']}
need(all(r.baseline_vector_fingerprint==old_fingerprints[r.signal_date] for r in ret.itertuples()),'Preserved48 logical vector fingerprints')
prior=ret.signal_date.lt('2021-11-30');need(prior.sum()==230 and ret.loc[prior,'payoff_delta'].eq(0).all() and ret.loc[prior,'patched_month_in_window'].eq(False).all(),'Strict pre-Nov30 zero impact')
need(ret.loc[~prior,'patched_month_in_window'].all() and ret.loc[~prior,'signal_date'].min()=='2021-11-30','Earliest completed outcome causal window')
means_before=frames['monthly_class_means_before'].set_index('class_id');means_after=frames['monthly_class_means_after'].set_index('class_id')
max_residual=0.
for row in ret.itertuples():
 expected=0. if row.signal_date<'2021-11-30' else (means_after.at[row.class_id,'class_mean_return']-means_before.at[row.class_id,'class_mean_return'])/120
 residual=abs(row.payoff_delta-expected);max_residual=max(max_residual,residual)
 need(math.isclose(row.shadow_payoff-row.baseline_payoff,row.payoff_delta,rel_tol=1e-13,abs_tol=1e-17),'Saved payoff delta reconciles')
 need(residual<=1e-16,'Conditional monthly mean enters each affected RET vector with1/120 weight')
roles=frames['training_and_cv_date_dependencies']
expected_roles={(2021,'2021_outer_evaluation','evaluation_labels_only'),(2022,'2022_outer_train','supervised_training_labels'),
 (2022,'2022_validate_2021','inner_validation_labels_only'),(2023,'2023_outer_train','supervised_training_labels'),
 (2023,'2023_validate_2021','inner_validation_labels_only'),(2023,'2023_validate_2022','inner_training_labels')}
need(set(zip(roles.outer_year,roles.split_id,roles.dependency_role))==expected_roles and len(roles)==6,'Independent frozen train/CV/evaluation role map')
need(roles.changed_outcome_month.eq('2021-11').all() and roles.fits_or_predictions_performed.eq(False).all(),'Date-only dependency claim')
need(summary['first_affected_signal_date']=='2021-11-30' and summary['first_affected_decision_outcome_month']=='2021-12' and summary['december2021_label_rebuilt'] is False,'Causality and December limitation')
need(summary['baseline_logical_vector_fingerprints_reproduced']==48 and summary['baseline_vector_bytepins_unchanged_before_after'] is True,'Baseline source verification receipt')
need(all(pin(path)==expected for path,expected in zip(files,original_pins)),'New report source files changed during review')
review={'schema':'dmv2-learning-counterfactual-independent-output-review/v1','status':'PASS_NEW_OUTPUT_REPRODUCTION',
 'original_cohort_rows':len(before),'baseline_labelled_rows':int(before.target_available.sum()),'shadow_labelled_rows':int(after.target_available.sum()),
 'return_rows_changed':1,'class_rows_changed':6,'old_existing_names_reclassified':5,'class_supports':counts,
 'ret_vectors':48,'ret_cells':480,'strictly_prior_unchanged_vectors':23,'conditional_affected_vectors':25,
 'first_possible_RET_signal':'2021-11-30','first_possible_RET_decision_outcome':'2021-12','max_payoff_delta_reproduction_absolute_residual':max_residual,
 'dependency_role_counts':dict(Counter(roles.dependency_role)),
 'fitting_or_inference_or_backtests':0,'production_financial_source_reads':0,'reviewed_new_audit_reports_only':True,
 'production_promoted':False,'historical_identity_or_learning_inputs_certified':False,
 'read_reservation':pin(reserve_path),'new_output_pins':original_pins,'summary_pin':pin(D/'summary.json'),
 'checks':['Alternative pandas average descending ranks reproduce every baseline and shadow class exactly','Original keys,request,eligibility and clock fields identical',
 'Onlyone RCM return changes;six classes includingfive previously labelled peers rerank underunchanged bins','All10class supports/means reproduce fromnew auditmonth tables',
 'NoRETdelta in23 earlier vectors;25 later vectors differ only bychangedNovember classmeans/120','Six independent outer/inner train/evaluation dependencies reproduce'],
 'next_steps':['Freeze field-completeOHLCV and dated route policy for fullerRCMhistory before anyfeature rebuild','Separate measuredNov2021label impact frompotentialPaper15/daily/context eligibility ripple',
 'Terminalcash/stock/ADS labels needaseparateeconomicreturnpolicy;remainingmissing-targetqueue staysvisible','If futurecorrected rebuild authorized, assess2022/2023outer models plusaffected inner preprocessing/fits;2021forecasts notretroactivelyrefitforfuturelabelrepair'],
 'limit':'Reproduces bounded newshadowoutputs only;doesnot independentlyrereadproductiontargets,the167originalmonthlymeans orsourceendpoints.'}
path=O/'final_review.json';need(not path.exists(),'Preserve prior final review');path.write_text(json.dumps(review,sort_keys=True,indent=2)+'\n')
print(json.dumps({k:review[k] for k in ('status','original_cohort_rows','baseline_labelled_rows','shadow_labelled_rows','return_rows_changed','class_rows_changed','class_supports','dependency_role_counts','max_payoff_delta_reproduction_absolute_residual')},indent=2))
print(json.dumps(pin(path),indent=2))
