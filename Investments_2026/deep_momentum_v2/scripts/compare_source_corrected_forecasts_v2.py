"""Compare saved forecast/selection versions after all final gates pass.

This is a passive diagnostic, not another experiment. It never loads models,
features, labels, execution prices or an accounting engine. Old source results
remain obsolete reference evidence. All writes require a new private output.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPORTER_SHA256='a68f6fef0953eac124836b698a21e7b0e38ded225e31bff931be185ddba0b81f'
ARMS=('rf_paper15','xgb_paper15','rf_extended','xgb_extended')
BASKETS=('class_1_10','class_2_9')
KEYS=['source_security_id','signal_date']
PROBABILITIES=[f'probability_{k}' for k in range(1,11)]
LINEAGE=['fit_year','training_outcome_start','training_outcome_end','training_label_observed_through']
META=KEYS+['intrinio_security_id','outcome_month','outcome_start','outcome_end',
    'prediction_eligible','paper_prediction_eligible','identity_eligible','requested',
    'identity_status','source_identity_status','identity_eligibility_policy',
    'identity_review_required','resolution_status']+LINEAGE
SELECT_META=KEYS+['intrinio_security_id','outcome_month','execution_date','side',
    'prediction_eligible','selected','requested_top_k','selected_names_per_side',
    'trade_eligible','execution_restricted','retrospective_execution_restriction']
MONTH_META=['signal_date','execution_date','outcome_month','original_rows','requested_top_k']
MONTH_VALUES=['long_candidates','short_candidates','long_selected','short_selected',
    'selected_names_per_side','effective_per_name_abs_weight']
MAX_FILE_BYTES=16*1024**2
MAX_PROJECTED_BYTES=128*1024**2
MAX_FORECAST_ROWS=60_000

def need(condition,message):
    if not condition:raise ValueError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()

def canonical(value):
    return 'sha256:'+hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()

def js(path):
    need(Path(path).stat().st_size<=MAX_FILE_BYTES,'Metadata exceeds bounded size')
    return json.loads(Path(path).read_text())

def reporter(path):
    need(sha(path)==REPORTER_SHA256,'Frozen passive reporter changed')
    spec=importlib.util.spec_from_file_location('frozen_corrected_reporting',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def old_gates(root,snapshot_path,snapshot_sha256,r):
    need(sha(snapshot_path)==snapshot_sha256.removeprefix('sha256:'),'Old preservation snapshot changed')
    snapshot=js(snapshot_path);need(Path(snapshot['root']).resolve()==root,'Old snapshot belongs to another workspace')
    preservation=r.preserve(snapshot_path)
    run=Path(snapshot['run_directory']).resolve();bt=Path(snapshot['backtest_directory']).resolve()
    need(run.is_relative_to(root) and bt.is_relative_to(run),'Old run leaves declared root')
    paths={'features':run/'audit.json','models':run/'models/audit.json','backtest':bt/'audit.json',
        **{role:run/'verification'/name for role,name in [('feature_verification','features.json'),('model_verification','models.json'),('backtest_verification','backtest.json')]}}
    data={}
    for role,path in paths.items():
        need(sha(path)==snapshot['expected_receipts'][role].removeprefix('sha256:'),'Old final receipt changed: '+role)
        data[role]=js(path)
        need(data[role].get('build_identity_sha256')==snapshot['run_build_identity'],'Old receipt identity mismatch: '+role)
        need(data[role].get('status') in ({'COMPLETE'} if role in ('features','models','backtest') else {'PASS','PASSED'}),'Old phase/verification incomplete: '+role)
    raw={name:sha(paths[name]) for name in ('features','models','backtest')}
    for role in ('feature_verification','model_verification','backtest_verification'):
        need(data[role].get('root_audit_sha256')==raw['features'],'Old independent gate root binding changed')
    fv=data['feature_verification'];mv=data['model_verification'];bv=data['backtest_verification']
    need(fv.get('models_fitted')==0 and fv.get('financial_2024_plus_read') is False,'Old feature verifier scope changed')
    need(mv.get('annual_fits_reconciled')==16 and mv.get('preprocessor_states_reconciled')==8 and mv.get('models_audit_sha256')==raw['models'],'Old16-fit verification incomplete')
    need(mv.get('models_fitted_by_verifier')==0 and mv.get('preprocessors_fitted_by_verifier')==0 and mv.get('financial_2024_plus_decoded') is False,'Old model verifier scope changed')
    need(mv.get('stitched_prediction_months')==48 and data['models'].get('summary',{}).get('fitted_arms')==16 and data['models']['summary'].get('preprocessor_fits')==8,'Old completed model grid changed')
    need(bv.get('backtest_audit_sha256')==raw['backtest'] and bv.get('source_pins_unchanged') is True and bv.get('financial_2024_plus_read') is False,'Old accounting gate changed')
    need(bv.get('scenarios')==64 and bv.get('sessions')==1006 and bv.get('outcome_months')==48,'Old accounting horizon changed')
    need(bv.get('models_fitted')==0 and bv.get('preprocessors_fitted')==0,'Old accounting verifier performed fits')
    return {'run':run,'backtest':bt,'audits':data,'preservation':preservation,'snapshot':snapshot}

def phase_metadata(context,identity,r):
    run=context['run'];bt=context['backtest'];audits=context['audits']
    root_phase=js(run/'build_identity.json');model_phase=js(run/'models/build_identity.json')
    bt_phase=js(r.pinned(bt,audits['backtest']['artifacts']['build_identity.json']))
    need(canonical(root_phase)==identity,'Root configuration identity changed')
    need(canonical(model_phase)==audits['models'].get('model_identity_sha256'),'Model phase identity changed')
    need(canonical(bt_phase)==audits['backtest'].get('phase_identity_sha256'),'Accounting phase identity changed')
    return {'root':root_phase,'model':model_phase,'backtest':bt_phase}

def compare_rules(old,new):
    need(old['root']['configuration']==new['root']['configuration'],'Research configuration changed across source versions')
    for field in ('configuration_sha256','runtime','workers'):
        need(old['model'][field]==new['model'][field],'Model phase changed frozen '+field)
    for field in ('configuration_sha256','clock','basket_policy','runtime','workers'):
        need(old['backtest'][field]==new['backtest'][field],'Accounting changed frozen '+field)
    cfg=old['root']['configuration'];backtest=cfg['backtest']
    need(backtest['cost_bps']==[0,5,10,25] and backtest['variants']=={'unstopped':None,'asset_stop_10pct':.1},'Cost/stop protocol changed')
    return {'status':'IDENTICAL_FROZEN_RULES','configuration_sha256':canonical(cfg),
        'model_recipe_sha256':canonical(cfg['model_recipe']),'same_model_parameters_packages_seed':True,
        'same_accounting_workers_clock_baskets_costs_stops':True,'RET_expanded':False}

def model_metadata(context,r):
    models=context['run']/'models';audit=context['audits']['models'];result={};pins=[]
    for year in (2020,2021,2022,2023):
        fold_path=r.pinned(models,audit['folds'][str(year)]);fold=js(fold_path)
        need(fold.get('status')=='COMPLETE' and fold.get('test_year')==year,'Incomplete fold metadata')
        need(fold.get('build_identity_sha256')==audit['build_identity_sha256'] and fold.get('model_identity_sha256')==audit['model_identity_sha256'],'Stale fold metadata')
        pins.append({'path':str(fold_path),'sha256':sha(fold_path),'size_bytes':fold_path.stat().st_size})
        for arm in ARMS:
            path=r.pinned(fold_path.parent,fold['arms'][arm]);state=js(path)
            need(state.get('status')=='COMPLETE' and isinstance(state.get('arm'),dict) and state['arm'].get('arm_id')==arm,'Incomplete estimator metadata')
            need(state.get('build_identity_sha256')==audit['build_identity_sha256'] and state.get('model_identity_sha256')==audit['model_identity_sha256'],'Stale estimator metadata')
            result[year,arm]={**state['metadata'],'_arm_descriptor':state['arm']};pins.append({'path':str(path),'sha256':sha(path),'size_bytes':path.stat().st_size})
    return result,pins

def compare_fit_metadata(old,new):
    need(set(old)==set(new)==set(itertools.product((2020,2021,2022,2023),ARMS)),'Missing annual estimator metadata')
    same=('explicit_parameters','effective_parameters','package','package_version','estimator_kind',
        'native_class_ids','native_to_canonical_mapping','probability_class_ids','probability_extraction',
        'train_start','train_end','label_observed_through','test_year','_arm_descriptor')
    rows=[]
    for year,arm in sorted(old):
        a=old[year,arm];b=new[year,arm]
        need(all(a.get(key)==b.get(key) for key in same),f'Changed model parameters/schema/chronology: {year}/{arm}')
        need(a.get('fit_count')==b.get('fit_count')==1,'Different per-arm fit count')
        for value in (a,b):
            need(all(value.get(name) is False for name in ('early_stopping_used','parameter_search_performed','probability_calibration_performed','test_labels_supplied_to_estimator','test_labels_used_for_fitting')),'Unapproved fitting policy')
        rows.append({'test_year':year,'arm':arm,'explicit_parameters_sha256':canonical(a['explicit_parameters']),
            'old_training_rows':a['train_rows'],'corrected_training_rows':b['train_rows'],
            'training_rows_change':b['train_rows']-a['train_rows'],'old_matrix_width':a['matrix_width'],
            'corrected_matrix_width':b['matrix_width'],'matrix_width_change':b['matrix_width']-a['matrix_width'],
            'training_keys_same':a['fit_keys_sha256']==b['fit_keys_sha256'],
            'preprocessor_fingerprint_same':a['preprocessor_fingerprint']==b['preprocessor_fingerprint'],
            'prepared_prediction_matrix_same':a['prepared_prediction_matrix_sha256']==b['prepared_prediction_matrix_sha256'],
            'prepared_training_matrix_same':a['prepared_train_matrix_sha256']==b['prepared_train_matrix_sha256'],
            'parameters_packages_seed_and_clock_same':True})
    return pd.DataFrame(rows)

def file_plan(base,pin,columns,max_rows,r,metadata_columns,numerical_columns):
    path=r.pinned(base,pin);need(path.stat().st_size<=MAX_FILE_BYTES,'Saved projection exceeds byte ceiling')
    f=pq.ParquetFile(path);need(f.metadata.num_rows<=max_rows,'Saved projection exceeds row ceiling')
    need(set(columns)<=set(f.schema_arrow.names),'Saved projection lacks required columns')
    def compressed(projection):
        return sum(f.metadata.row_group(g).column(c).total_compressed_size for g in range(f.metadata.num_row_groups)
            for c in range(f.metadata.num_columns) if f.metadata.row_group(g).column(c).path_in_schema in projection)
    need(set(metadata_columns)|set(numerical_columns)==set(columns),'Projection plan omits a declared read')
    metadata_bytes=compressed(metadata_columns);numerical_bytes=compressed(numerical_columns)
    return {'path':str(path),'sha256':sha(path),'size_bytes':path.stat().st_size,
        'rows':f.metadata.num_rows,'columns':columns,'metadata_columns':metadata_columns,'numerical_columns':numerical_columns,
        'metadata_projected_compressed_bytes':metadata_bytes,'numerical_projected_compressed_bytes':numerical_bytes,
        'projected_compressed_bytes':metadata_bytes+numerical_bytes}

def read_projection(plan,columns):
    need(sha(plan['path'])==plan['sha256'],'Projection changed after plan')
    need(set(columns)<=set(plan['columns']),'Projection exceeds frozen column request')
    frame=pq.ParquetFile(plan['path']).read(columns=columns).to_pandas()
    need(len(frame)==plan['rows'],'Projection changed saved row count')
    return frame

def windows(clock):
    sessions=clock['selection_sessions'];need(sessions==sorted(set(sessions)),'Unordered clock')
    need(all(isinstance(x,str) and '2019-12-01'<=x<='2023-12-29' and len(x)==10 for x in sessions),'Clock exceeds approved forecast horizon')
    result={}
    for period in pd.period_range('2020-01','2023-12',freq='M'):
        outcome=str(period);prior=str(period-1)
        prior_days=[x for x in sessions if x.startswith(prior)];current=[x for x in sessions if x.startswith(outcome)]
        need(prior_days and current,'Incomplete independent clock')
        result[prior_days[-1]]={'outcome_month':outcome,'outcome_start':current[0],'outcome_end':current[-1]}
    return result

def forecast_metadata(frame,window):
    need(not frame[KEYS].isna().any().any() and not frame.duplicated(KEYS).any(),'Forecast lost original unique keys')
    need(set(frame.signal_date)==set(window),'Forecast signal horizon changed')
    for field in ('outcome_month','outcome_start','outcome_end'):
        need(frame[field].eq(frame.signal_date.map(lambda key:window[key][field])).all(),'Forecast next-month timing changed')
    for field in ('prediction_eligible','paper_prediction_eligible','identity_eligible','requested'):
        need(frame[field].map(lambda value:isinstance(value,(bool,np.bool_))).all(),'Admission flag not explicitly Boolean')
    need(frame.prediction_eligible.equals(frame.paper_prediction_eligible),'Prediction cohort differs from Paper15')
    review=frame.identity_review_required
    need(review.loc[review.notna()].map(lambda value:isinstance(value,(bool,np.bool_))).all(),'Non-null identity review flag not explicitly Boolean')
    eligible=frame.prediction_eligible
    need(review.loc[eligible].notna().all(),'Predicted row has unknown identity review flag')
    need((frame.loc[eligible,'identity_eligible']&frame.loc[eligible,'requested']).all(),'Predicted row lacks positive identity/request admission')
    need(frame.loc[eligible,'intrinio_security_id'].map(lambda value:isinstance(value,str) and bool(value.strip())).all(),'Predicted row lacks valid native identity')
    need(not frame.loc[eligible].duplicated(['signal_date','intrinio_security_id']).any(),'Predicted native identity collision')
    year=frame.outcome_month.str[:4].astype(int)
    need(frame.fit_year.eq(year).all() and frame.training_outcome_start.eq(year.map(lambda y:f'{y-10}-01-01')).all()
        and frame.training_outcome_end.eq(year.map(lambda y:f'{y-1}-12-31')).all(),'Forecast rolling training clock changed')
    cutoff=year.map({2020:'2019-12-31',2021:'2020-12-31',2022:'2021-12-31',2023:'2022-12-30'})
    need(frame.training_label_observed_through.eq(cutoff).all(),'Training label cutoff changed')
    return frame.set_index(KEYS).sort_index()

def same_cohort(a,b):
    need(a.index.equals(b.index),'Original prediction keys changed')
    need(a.equals(b),'Prediction identity/admission/timing metadata changed')

def metadata_missingness(frame,tag,arm):
    """Preserve unknown audit state; never coerce it into false or eligibility."""
    rows=[];cases=[];eligible=frame.prediction_eligible
    for field in frame.columns:
        missing=frame[field].isna()
        rows.append({'source_version':tag,'arm':arm,'field':field,'original_rows':len(frame),
            'missing_original_rows':int(missing.sum()),'missing_predicted_rows':int((missing&eligible).sum()),
            'missing_nonpredicted_rows':int((missing&~eligible).sum())})
        for (sid,signal),value in eligible.loc[missing].items():
            cases.append({'source_version':tag,'arm':arm,'source_security_id':sid,'signal_date':signal,
                'field':field,'raw_value_state':'null','prediction_eligible':bool(value)})
    return pd.DataFrame(rows),pd.DataFrame(cases,columns=['source_version','arm',*KEYS,'field','raw_value_state','prediction_eligible'])

def cohort_changes(old,new,arm):
    """Original keys/clocks are invariants; admitted state can change lawfully."""
    need(old.index.equals(new.index),'Original prediction keys changed')
    for name in ['outcome_month','outcome_start','outcome_end',*LINEAGE]:
        need(old[name].equals(new[name]),'Original outcome/training chronology changed')
    rows=[]
    for field in old.columns:
        if field in ['outcome_month','outcome_start','outcome_end',*LINEAGE]:continue
        equal=(old[field].eq(new[field]).fillna(False)|(old[field].isna()&new[field].isna()))
        rows.append({'arm':arm,'field':field,'original_rows':len(old),'changed_rows':int((~equal).sum()),
            'changed_fraction':float((~equal).mean())})
    return pd.DataFrame(rows)

def probability_changes(old,new,metadata,arm,*,corrected_metadata=None):
    corrected_metadata=metadata if corrected_metadata is None else corrected_metadata
    need(metadata.index.equals(corrected_metadata.index),'Original probability keys changed')
    old_eligible=metadata.prediction_eligible.to_numpy(dtype=bool)
    new_eligible=corrected_metadata.prediction_eligible.to_numpy(dtype=bool)
    joint=old_eligible&new_eligible
    need(old.shape==new.shape==(len(metadata),10),'Probability shape changed')
    for values,eligible in ((old,old_eligible),(new,new_eligible)):
        need(np.isnan(values[~eligible]).all(),'Ineligible original row has a probability')
        p=values[eligible]
        need(np.isfinite(p).all() and (p>=0).all() and (p<=1).all() and np.allclose(p.sum(1),1,atol=1e-6,rtol=0),'Invalid canonical ten-class probabilities')
    p=old[joint];q=new[joint];difference=np.abs(q-p)
    m=metadata.reset_index();m['old_prediction_eligible']=old_eligible;m['corrected_prediction_eligible']=new_eligible
    m['probability_comparison_status']=np.select([joint,old_eligible&~new_eligible,~old_eligible&new_eligible],['joint','old_only','corrected_only'],default='neither')
    for name,values,eligible in [('old_mode_class',old,old_eligible),('corrected_mode_class',new,new_eligible)]:
        m[name]=pd.Series(pd.NA,index=m.index,dtype='Int64');m.loc[eligible,name]=np.argmax(values[eligible],axis=1)+1
    m['mode_changed']=m.old_mode_class.ne(m.corrected_mode_class).where(joint)
    for name,values in [('mean_absolute_probability_change',difference.mean(1)),('maximum_absolute_probability_change',difference.max(1)),('total_variation_distance',difference.sum(1)/2)]:
        m[name]=np.nan;m.loc[joint,name]=values
    for name,values in [('any_probability_changed_exact',np.any(p!=q,axis=1)),('any_probability_change_gt_1e_12',np.any(difference>1e-12,axis=1))]:
        m[name]=pd.Series(pd.NA,index=m.index,dtype='boolean');m.loc[joint,name]=values
    summaries=[]
    groups=[('all',m)]+[(str(year),g) for year,g in m.groupby('fit_year',sort=True)]+[(month,g) for month,g in m.groupby('outcome_month',sort=True)]
    for period,g in groups:
        j=g.loc[g.probability_comparison_status.eq('joint')]
        summaries.append({'arm':arm,'period':period,'original_rows':len(g),'old_prediction_rows':int(g.old_prediction_eligible.sum()),
            'corrected_prediction_rows':int(g.corrected_prediction_eligible.sum()),'joint_prediction_rows':len(j),
            'old_only_prediction_rows':int(g.probability_comparison_status.eq('old_only').sum()),
            'corrected_only_prediction_rows':int(g.probability_comparison_status.eq('corrected_only').sum()),
            'neither_prediction_rows':int(g.probability_comparison_status.eq('neither').sum()),
            'mode_changed_rows':int(j.mode_changed.sum()),'mode_changed_fraction':float(j.mode_changed.mean()) if len(j) else None,
            'mean_absolute_probability_change':float(j.mean_absolute_probability_change.mean()) if len(j) else None,
            'mean_total_variation_distance':float(j.total_variation_distance.mean()) if len(j) else None,
            'maximum_absolute_probability_change':float(j.maximum_absolute_probability_change.max()) if len(j) else None,
            'any_probability_changed_exact_fraction':float(j.any_probability_changed_exact.mean()) if len(j) else None,
            'any_probability_change_gt_1e_12_fraction':float(j.any_probability_change_gt_1e_12.mean()) if len(j) else None})
    return pd.DataFrame(summaries),m[[*KEYS,'outcome_month','fit_year','old_prediction_eligible','corrected_prediction_eligible',
        'probability_comparison_status','old_mode_class','corrected_mode_class','mode_changed','mean_absolute_probability_change',
        'maximum_absolute_probability_change','total_variation_distance','any_probability_changed_exact','any_probability_change_gt_1e_12']].assign(arm=arm)

def check_selected(frame,monthly,cohort,mode,window,basket):
    need(not frame.duplicated(KEYS).any(),'Selected source key duplicated')
    need(frame.signal_date.isin(window).all() and monthly.signal_date.tolist()==sorted(window),'Selection horizon changed')
    need(frame.side.isin(['long','short']).all() and frame.selected.eq(True).all() and frame.prediction_eligible.eq(True).all(),'Invalid selected metadata')
    need(frame.trade_eligible.eq(True).all() and frame.execution_restricted.eq(False).all(),'Fixed execution restriction violated')
    need(frame.requested_top_k.eq(10).all() and monthly.requested_top_k.eq(10).all(),'Selection cap changed')
    keyed=frame.set_index(KEYS);need(keyed.index.isin(cohort.index).all(),'Selected key absent from original forecasts')
    need(cohort.loc[keyed.index,'prediction_eligible'].all(),'Selected signal-ineligible row')
    need(keyed.intrinio_security_id.equals(cohort.loc[keyed.index,'intrinio_security_id']),'Selected native identity changed')
    need(frame.execution_date.eq(frame.signal_date.map(lambda key:window[key]['outcome_start'])).all(),'Execution is not next-session open')
    need(frame.outcome_month.eq(frame.signal_date.map(lambda key:window[key]['outcome_month'])).all(),'Selected outcome month changed')
    need(keyed.predicted_class.eq(mode.reindex(keyed.index)).all(),'Saved selected class differs from saved forecast mode')
    pair={'class_1_10':{'long':1,'short':10},'class_2_9':{'long':2,'short':9}}[basket]
    need(frame.predicted_class.eq(frame.side.map(pair)).all(),'Class baskets were pooled or changed')
    for row in monthly.itertuples():
        expected=int(min(10,row.long_candidates,row.short_candidates));q=frame.loc[frame.signal_date.eq(row.signal_date)]
        need(row.original_rows==int((cohort.reset_index().signal_date==row.signal_date).sum()),'Original denominator changed')
        need(row.selected_names_per_side==row.long_selected==row.short_selected==expected,'Balanced breadth changed')
        need(len(q)==2*expected and all(q.side.eq(side).sum()==expected for side in ('long','short')),'Saved selected count differs from breadth receipt')
        if expected:
            need(np.isclose(row.effective_per_name_abs_weight,.5/expected,atol=1e-14,rtol=0),'Monthly equal weight policy changed')
            need(np.allclose(q.intended_weight,q.side.map({'long':1,'short':-1})*.5/expected,atol=1e-14,rtol=0),'Selected 50/50 weights changed')
        else:need(pd.isna(row.effective_per_name_abs_weight),'Zero-breadth rebalance invented a weight')

def selection_overlap(old,new,arm,basket,window):
    rows=[]
    for signal in sorted(window):
        for side in ('long','short'):
            a=set(old.loc[old.signal_date.eq(signal)&old.side.eq(side),'source_security_id'])
            b=set(new.loc[new.signal_date.eq(signal)&new.side.eq(side),'source_security_id'])
            same=len(a&b);union=len(a|b)
            rows.append({'arm':arm,'basket':basket,'signal_date':signal,'outcome_month':window[signal]['outcome_month'],'side':side,
                'old_obsolete_selected':len(a),'corrected_selected':len(b),'same_side_overlap':same,'old_removed':len(a-b),'new_added':len(b-a),
                'jaccard':same/union if union else np.nan,'fraction_old_names_retained':same/len(a) if a else np.nan,
                'fraction_corrected_names_from_old':same/len(b) if b else np.nan,'same_selected_set':a==b})
    return pd.DataFrame(rows)

def run(root,bindings_path,bindings_sha256,old_snapshot,old_snapshot_sha256,reporter_path,output):
    root=Path(root).resolve();output=Path(output).resolve();need(not output.exists(),'Preserve every existing diagnostic snapshot')
    r=reporter(reporter_path);old=old_gates(root,old_snapshot,old_snapshot_sha256,r)
    need(sha(bindings_path)==bindings_sha256.removeprefix('sha256:'),'New final bindings changed')
    bindings=js(bindings_path);identity=bindings['build_identity_sha256']
    # Both old model gates pass above, and all final new gates pass inside this
    # exact frozen reader before any probability/selection financial read.
    final=r.read_source_corrected_reports(bindings_path,root=root,expected_bindings_sha256=bindings_sha256,identity_sha256=identity)
    new={'run':Path(final['directory']).parent,'backtest':Path(final['directory']),
        'audits':{'features':js(r.pinned(root,bindings['upstream_audits']['features'])),
            'models':js(r.pinned(root,bindings['upstream_audits']['models'])),'backtest':final['audit']}}
    verification=js(r.pinned(root,bindings['backtest_verification']))
    need(verification.get('source_corrections_and_exact_evidence_rebind_independently_verified') is True,'New corrected evidence rebind has not passed independent accounting review')
    phases={tag:phase_metadata(context,context['audits']['features']['build_identity_sha256'],r) for tag,context in [('old',old),('corrected',new)]}
    rules=compare_rules(phases['old'],phases['corrected']);window=windows(phases['old']['backtest']['clock'])
    fits={};fit_pins={}
    for tag,context in [('old',old),('corrected',new)]:fits[tag],fit_pins[tag]=model_metadata(context,r)
    fit_comparison=compare_fit_metadata(fits['old'],fits['corrected'])
    plans={};metadata={};selected_meta={};monthly_meta={}
    for tag,context in [('old',old),('corrected',new)]:
        plans[tag]={'forecasts':{},'selections':{}};metadata[tag]={};selected_meta[tag]={};monthly_meta[tag]={};reference=None
        for arm in ARMS:
            models=context['run']/'models';arm_path=r.pinned(models,context['audits']['models']['arms'][arm]);state=js(arm_path)
            need(state.get('status')=='COMPLETE' and state.get('model_identity_sha256')==context['audits']['models']['model_identity_sha256'],'Aggregate forecasts stale')
            p=file_plan(arm_path.parent,state['artifacts']['predictions'],META+PROBABILITIES,MAX_FORECAST_ROWS,r,META,KEYS+PROBABILITIES)
            need(Path(p['path'])==arm_path.parent/'predictions.parquet','Forecast pin redirects fixed path')
            plans[tag]['forecasts'][arm]=p;metadata[tag][arm]=forecast_metadata(read_projection(p,META),window)
            if reference is None:reference=metadata[tag][arm]
            else:same_cohort(reference,metadata[tag][arm])
            for basket in BASKETS:
                prefix=f'selections/{arm}/{basket}/';parts={}
                for name,columns,cap in [('selected',SELECT_META+['predicted_class','intended_weight'],960),('monthly',MONTH_META+MONTH_VALUES,48)]:
                    pin=context['audits']['backtest']['artifacts'][prefix+name+'.parquet']
                    parts[name]=file_plan(context['backtest'],pin,columns,cap,r,SELECT_META if name=='selected' else MONTH_META,columns)
                    need(Path(parts[name]['path'])==context['backtest']/prefix/(name+'.parquet'),'Selection pin redirects fixed path')
                key=arm+'/'+basket;plans[tag]['selections'][key]=parts
                selected_meta[tag][key]=read_projection(parts['selected'],SELECT_META)
                monthly_meta[tag][key]=read_projection(parts['monthly'],MONTH_META)
                need(selected_meta[tag][key].signal_date.isin(window).all(),'Selected metadata includes protected signal')
                need(monthly_meta[tag][key].signal_date.tolist()==sorted(window),'Monthly metadata lacks48-signal grid')
        need(set(metadata[tag])==set(ARMS),'Missing estimator forecasts')
    metadata_changes=pd.concat([cohort_changes(metadata['old'][arm],metadata['corrected'][arm],arm) for arm in ARMS],ignore_index=True)
    missing_parts=[metadata_missingness(metadata[tag][arm],tag,arm) for tag in ('old','corrected') for arm in ARMS]
    missing_summary=pd.concat([part[0] for part in missing_parts],ignore_index=True)
    missing_cases=pd.concat([part[1] for part in missing_parts],ignore_index=True)
    observed_metadata_same=bool(metadata_changes.changed_rows.eq(0).all())
    all_plans=[p for value in plans.values() for p in value['forecasts'].values()]+[p for value in plans.values() for pair in value['selections'].values() for p in pair.values()]
    projected=sum(x['projected_compressed_bytes'] for x in all_plans)
    need(projected<=MAX_PROJECTED_BYTES,'Forecast/selection projections exceed frozen128MiB ceiling')
    read_plan={'schema':'dmv2-source-corrected-forecast-selection-read-plan/v1','status':'FROZEN_BEFORE_NUMERICAL_FORECAST_SELECTION_READ',
        'old_source_status':'OBSOLETE_SOURCE_HISTORY_REFERENCE_ONLY','new_root_identity_sha256':identity,
        'rules':rules,'artifact_plans':plans,'annual_metadata_pins':fit_pins,'all_original_prediction_keys_same':True,
        'same_observed_identity_admission_metadata':observed_metadata_same,
        'metadata_change_counts':metadata_changes.to_dict('records'),
        'raw_metadata_missingness':missing_summary.to_dict('records'),
        'maximum_rows_per_forecast':MAX_FORECAST_ROWS,'maximum_file_bytes':MAX_FILE_BYTES,'maximum_projected_bytes':MAX_PROJECTED_BYTES,
        'projected_compressed_bytes':projected,'all_final_old_and_new_model_and_accounting_gates_passed':True,
        'source_prices_features_labels_or_estimators_requested':False,'financial_2024_plus_read':False}
    output.mkdir(parents=True);(output/'read_plan.json').write_text(json.dumps(read_plan,indent=2)+'\n')
    missing_summary.to_csv(output/'forecast_metadata_missingness.csv',index=False)
    missing_cases.to_parquet(output/'forecast_metadata_null_cases.parquet',index=False)
    summaries=[];row_changes=[];overlap=[];modes={tag:{} for tag in plans}
    for arm in ARMS:
        values={}
        for tag in plans:
            p=plans[tag]['forecasts'][arm];raw=read_projection(p,KEYS+PROBABILITIES).set_index(KEYS).reindex(metadata[tag][arm].index)
            values[tag]=raw[PROBABILITIES].to_numpy(dtype=float)
            eligible=metadata[tag][arm].prediction_eligible.to_numpy(dtype=bool)
            modes[tag][arm]=pd.Series(np.argmax(values[tag][eligible],axis=1)+1,index=metadata[tag][arm].index[eligible],name='predicted_class')
        summary,rows=probability_changes(values['old'],values['corrected'],metadata['old'][arm],arm,corrected_metadata=metadata['corrected'][arm]);summaries.append(summary);row_changes.append(rows)
        for basket in BASKETS:
            key=arm+'/'+basket;selected={}
            for tag in plans:
                pair=plans[tag]['selections'][key]
                selected[tag]=read_projection(pair['selected'],SELECT_META+['predicted_class','intended_weight'])
                monthly=read_projection(pair['monthly'],MONTH_META+MONTH_VALUES)
                need(selected[tag][SELECT_META].equals(selected_meta[tag][key]) and monthly[MONTH_META].equals(monthly_meta[tag][key]),'Saved metadata changed after numerical read')
                check_selected(selected[tag],monthly,metadata[tag][arm],modes[tag][arm],window,basket)
            overlap.append(selection_overlap(selected['old'],selected['corrected'],arm,basket,window))
    forecasts=pd.concat(summaries,ignore_index=True);row_frame=pd.concat(row_changes,ignore_index=True);selection=pd.concat(overlap,ignore_index=True)
    need(len(forecasts)==212 and len(selection)==768,'Incomplete comparative diagnostic grid')
    fit_comparison.to_csv(output/'annual_fit_metadata_comparison.csv',index=False)
    metadata_changes.to_csv(output/'prediction_metadata_field_changes.csv',index=False)
    forecasts.to_csv(output/'forecast_changes_by_arm_year_month.csv',index=False)
    row_frame.to_parquet(output/'matched_prediction_changes.parquet',index=False)
    selection.to_csv(output/'selected_overlap_by_month_basket_side.csv',index=False)
    pooled=selection.groupby(['arm','basket','side'],sort=False).agg(months=('signal_date','size'),mean_jaccard=('jaccard','mean'),mean_fraction_old_names_retained=('fraction_old_names_retained','mean'),same_selected_set_months=('same_selected_set','sum'),old_selected=('old_obsolete_selected','sum'),corrected_selected=('corrected_selected','sum'),same_side_overlap=('same_side_overlap','sum')).reset_index()
    pooled['pooled_fraction_old_names_retained']=pooled.same_side_overlap/pooled.old_selected.replace(0,np.nan)
    pooled.to_csv(output/'selected_overlap_summary.csv',index=False)
    after=r.preserve(old_snapshot)
    receipt={'schema':'dmv2-source-corrected-forecast-selection-diagnostic/v1','status':'PASSIVE_COMPARISON_COMPLETE',
        'old_source_status':'OBSOLETE_SOURCE_HISTORY_REFERENCE_ONLY','new_root_identity_sha256':identity,
        'bindings_sha256':sha(bindings_path),'old_snapshot_sha256':sha(old_snapshot),'frozen_reporter_sha256':REPORTER_SHA256,
        'read_plan_sha256':sha(output/'read_plan.json'),'rules':rules,'all_original_prediction_keys_same':True,
        'same_observed_identity_admission_metadata':observed_metadata_same,
        'raw_metadata_null_case_rows':len(missing_cases),'unknown_audit_flags_imputed':False,
        'estimator_metadata_comparisons':16,'forecast_summary_rows':len(forecasts),'selection_overlap_rows':len(selection),
        'old_selected_metadata_preserved_before':old['preservation'],'old_selected_metadata_preserved_after':after,
        'interpretation':'Changes compare actual refitted forecasts and saved selections; they do not isolate or estimate a fill-only causal repair effect.',
        'metrics_caution':'Probability distances average over ten separate classes only on jointly predicted keys. Old-only/corrected-only/neither rows remain explicit with no probability imputation. Mode class uses argmax with first-class tie resolution. Jaccard is intersection/union, and retained-name fractions have explicitly labelled denominators.',
        'new_fits':0,'model_reload':False,'feature_label_or_source_price_read':False,'selection_recomputed':False,'accounting_rerun':False,
        'financial_2024_plus_read':False,'artifacts':{p.name:{'sha256':sha(p),'size_bytes':p.stat().st_size} for p in output.iterdir() if p.is_file()}}
    (output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--bindings',required=True);p.add_argument('--bindings-sha256',required=True)
    p.add_argument('--old-snapshot',required=True);p.add_argument('--old-snapshot-sha256',required=True);p.add_argument('--reporter',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    receipt=run(args.root,args.bindings,args.bindings_sha256,args.old_snapshot,args.old_snapshot_sha256,args.reporter,args.output)
    print(json.dumps({k:receipt[k] for k in ('status','estimator_metadata_comparisons','forecast_summary_rows','selection_overlap_rows','new_fits')}))
