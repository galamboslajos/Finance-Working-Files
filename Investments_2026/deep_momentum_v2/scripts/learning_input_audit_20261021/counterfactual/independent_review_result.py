"""Review only new conditional result artifacts; no production financial reads."""
from pathlib import Path
import hashlib,json,math
import pandas as pd
import pyarrow.parquet as pq

S=Path(__file__).resolve().parent;P=S/'result-v1'
summary=json.loads((P/'summary.json').read_text())
frames={n:pq.read_table(P/(n+'.parquet')).to_pandas() for n in (
    'original_month','conditional_shadow_month','changed_rows','monthly_class_means_before',
    'monthly_class_means_after','causal_ret_vector_deltas','training_and_cv_date_dependencies')}
a=frames['original_month'];b=frames['conditional_shadow_month'];changes=frames['changed_rows']
before=frames['monthly_class_means_before'];after=frames['monthly_class_means_after']
v=frames['causal_ret_vector_deltas'];deps=frames['training_and_cv_date_dependencies'];checks=[]
def check(value,name):
    if not bool(value):raise AssertionError(name)
    checks.append(name)
def same(x,y):return pd.isna(x) and pd.isna(y) or not pd.isna(x) and not pd.isna(y) and x==y
safe=['source_security_id','signal_date','outcome_month','outcome_start','outcome_end','identity_eligible',
      'requested','paper_prediction_eligible','prediction_eligible','outcome_requested']
check(len(a)==len(b)==910,'all910 original rows retained')
check(a[safe].equals(b[safe]),'original keys metadata identity requests and prediction cohort unchanged')
check(int(a.prediction_eligible.sum())==int(b.prediction_eligible.sum())==873,'873 predictioneligible unchanged')
for name,frame,n in [('baseline',a,872),('shadow',b,873)]:
    observed=frame[frame.prediction_eligible & frame.return_available];values=observed.label_return.tolist()
    check(len(observed)==n and observed.target_available.all(),name+' observed and labelled count')
    # Independent count-based equality midrank; no production helper called.
    for row in observed.itertuples():
        higher=sum(x>row.label_return for x in values);tied=sum(x==row.label_return for x in values)
        check(int(row.target_class)==min(10,max(1,1+10*(2*higher+tied)//(2*n))),name+' exact row class')
ri=[i for i,(x,y) in enumerate(zip(a.label_return,b.label_return)) if not same(x,y)]
ci=[i for i,(x,y) in enumerate(zip(a.target_class,b.target_class)) if not same(x,y)]
check(len(ri)==1 and a.iloc[ri[0]].source_security_id==summary['direct_endpoint_row'],'onlyRCM return changed')
check(len(ci)==len(changes)==6==summary['class_rows_changed'],'RCM plusfive other classes changed')
check(set(changes.source_security_id)==set(a.iloc[ci].source_security_id),'changedID ledger exact')
check(b.iloc[ri[0]].label_return==summary['new_endpoint_return'],'endpoint return matches summary')
for name,frame,means in [('baseline',a,before),('shadow',b,after)]:
    for row in means.itertuples():
        group=frame[frame.target_available & frame.target_class.eq(row.class_id)]
        mean=math.fsum(float(x)/len(group) for x in group.label_return) if len(group) else None
        check(row.member_count==len(group) and same(row.class_mean_return,mean),name+' classcount and equalweight mean')
check(len(v)==480 and v.signal_date.nunique()==48 and v.groupby('signal_date').size().eq(10).all(),'48x10 RET grid')
check(v.baseline_vector_available.all() and v.shadow_vector_available.all(),'all48 RET vectors available')
check(v.loc[v.signal_date.lt('2021-11-30'),'payoff_delta'].eq(0).all(),'past RET unchanged')
check(v.loc[v.patched_month_in_window,'signal_date'].min()=='2021-11-30','firstaffected Nov30 signal Decemberdecision')
mu0=before.set_index('class_id').class_mean_return;mu1=after.set_index('class_id').class_mean_return
residual=max(abs(row.payoff_delta-((mu1[row.class_id]-mu0[row.class_id])/120
    if row.signal_date>='2021-11-30' else 0.)) for row in v.itertuples())
check(residual<=2e-17,'all RETdeltas equal classmeanDelta/120 within floatingrounding')
check(not deps.fits_or_predictions_performed.any() and not deps.all_models_need_refit_claim.any(),'dateonly dependencies no refitclaim')
check(summary['models_fitted']==summary['preprocessors_fitted']==summary['new_inference_calls']==0,'zero fits preprocessing inference')
check(summary['production_promoted'] is False and summary['production_inputs_or_prior_runs_modified'] is False,'unpromoted no productionchange')
dec=summary['december_anchor_dependency_receipt']
check(summary['december2021_label_rebuilt'] is False and dec['new_December_return_inferred'] is False,'December anchor no inventedreturn')
check(dec['original_safe_target_metadata'][0]['paper_prediction_eligible'] is False,'originalDecember Paper15 ineligible state retained')
deltas=[{'class_id':k,'baseline_members':int(before.set_index('class_id').loc[k,'member_count']),
    'shadow_members':int(after.set_index('class_id').loc[k,'member_count']),
    'class_mean_delta':float(mu1[k]-mu0[k])} for k in range(1,11)]
stock_classes=changes[['source_security_id','baseline_class','shadow_class','return_changed']].astype(object)
stock_classes=stock_classes.where(pd.notna(stock_classes),None).to_dict('records')
report={'schema':'dmv2-learning-shadow-independent-result-review/v1','status':'PASS',
    'scope':'Only new conditional result artifacts; no original production financial source decoded by reviewer.',
    'production_selection_RET_accounting_called':False,'original_rows':910,'predictioneligible_rows':873,
    'observed_before':872,'observed_after':873,'changed_return_rows':1,'changed_class_rows':6,
    'additional_original_stock_classes_changed':5,'ret_vectors':48,'first_affected_signal':'2021-11-30',
    'first_affected_decision_outcome_month':'2021-12','affected_signal_count':25,
    'max_abs_class_mean_delta':max(abs(x['class_mean_delta']) for x in deltas),
    'max_abs_RET_payoff_delta':float(v.payoff_delta.abs().max()),'max_causal_delta_formula_residual':residual,
    'class_mean_deltas':deltas,'changed_stock_classes':stock_classes,'checked_invariant_count':len(checks),
    'checked_invariants':dict(pd.Series(checks).value_counts()),
    'new_result_file_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in P.iterdir() if f.is_file()},
    'inherited_source_claims':['Original target Arrow fingerprint and48 baseline RET logical fingerprints authenticated/reproduced by frozen central driver.',
        'Original payoff raw bytepins checked before/after by central driver; reviewer did not reopen original financial sources.'],
    'limitations':['Conditional fixedoldcohort November target-only shadow; no later feature/cohort repair measured.',
        'Endpoint evidence inherited from separately reviewed/pinned repair and central driver, not independently decoded here.',
        'No canonical identity, economiccashstock accounting, investability or model/backtest consistency certification.']}
path=S/'independent_result_review.json'
if path.exists():raise SystemExit('Preserve prior review')
path.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False,default=lambda x:int(x))+'\n')
disclosure={'schema':'dmv2-learning-shadow-read-semantics-disclosure/v1',
    'frozen_driver_and_plan_unchanged':True,'numeric_read_plan_sha256':summary['numeric_read_plan_sha256'],
    'summary_field_clarification':'December31_endpoint_evidence_requested_or_read=false means no RCM December31 endpoint was selected, converted to Python, analyzed, used as evidence, or used to construct a December economic return. It is not a claim that all excluded rows escaped transient Arrow block decoding.',
    'whole_selected_annual2021_Arrow_column_chunks_may_decode_mixed_rows':True,
    'financial_values_converted_and_analyzed':'Original fullNovember2021 targetmonth; saved2010Jan–2023Nov classmeans; exactlyRCM October29 and November30 adjustedendpoints.',
    'December_target_financial_values_selected_or_converted_to_Python':False,
    'December31_RCM_endpoint_selected_or_converted_to_Python':False,
    'December31_RCM_endpoint_used_to_infer_December_return':False,
    'protected2024plus_financial_values_decoded':False,
    'execution_source_disclosure':'Complete selected-column compressed chunks were reserved, not only filtered rows. All source date metadata was checked before numerical projection; Arrow filtering preceded financial Python conversion.'}
path=S/'read_semantics_disclosure.json'
if path.exists():raise SystemExit('Preserve prior disclosure')
path.write_text(json.dumps(disclosure,indent=2,sort_keys=True)+'\n')
print(json.dumps({'status':'PASS','checks':len(checks),'changed_stock_classes':stock_classes,
    'max_class_mean_delta':report['max_abs_class_mean_delta'],'max_RET_delta':report['max_abs_RET_payoff_delta'],
    'delta_residual':residual}))
