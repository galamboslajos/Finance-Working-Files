#!/usr/bin/env python3
"""Independent fixed extension input, inference, metric and selection checks.

Does not import the producer's monthly rules, fitting or portfolio functions.
Prior source/feature correctness is inherited through exact independent pins;
all new target/RET/portfolio formulas are independently reconstructed here.
"""
from pathlib import Path
import hashlib, itertools, json, math

FEATURES = tuple(p + str(n) for p in ('zMOM', 'meanMOM', 'stdMOM') for n in (1, 3, 6, 9, 12))
PROBS = tuple('probability_' + str(k) for k in range(1, 11))
KEYS = ['source_security_id', 'signal_date']
SOURCE_COLUMNS = [*KEYS, 'intrinio_security_id', 'identity_eligible', 'identity_status',
                  'identity_review_required', 'resolution_status', 'prediction_eligible',
                  'prediction_status', 'outcome_month', 'outcome_start', 'outcome_end',
                  'history_years', 'history_complete', 'expected_history_sessions',
                  'valid_history_sessions', *FEATURES]


def need(value, message):
    if not value: raise ValueError(message)


def pin(path):
    p = Path(path).resolve(); h = hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''): h.update(block)
    return dict(path=str(p), size_bytes=p.stat().st_size, sha256=h.hexdigest())


def check(item):
    need(pin(item['path']) == {k: item[k] for k in ('path', 'size_bytes', 'sha256')}, 'Independent byte binding')
    return Path(item['path'])


def read(path):
    need(Path(path).stat().st_size <= 32 * 1024**2, 'Independent metadata ceiling')
    return json.loads(Path(path).read_text())


def save(path, value):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as stream: json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False); stream.write('\n')
    return pin(p)


def shift(month, offset):
    y, m = map(int, month.split('-')); t = 12*y + m - 1 + offset
    return f'{t//12:04d}-{t%12+1:02d}'



# Extension-specific context and input checks are appended below.
def verify_models(plan_path,guard=lambda:None):
    import joblib,numpy as np,pandas as pd,pyarrow.parquet as pq
    p,cfg,root,prep,vplan=context(plan_path,'models');data=read(root/'verification/data.json')
    need(data['status']=='PASS' and data['prepared_receipt_pin']==pin(root/'prepared/receipt.json'),'Independent prepared source proof')
    receipts={};forecasts={};summaries=[];all_metrics=[]
    for n in (5,10):
        panel=pq.read_table(check(prep['artifacts'][f'panel{n}'])).to_pandas()
        for year,kind in itertools.product((2024,2025,2026),('rf','xgb')):
            guard();directory=root/f'folds/{n}y/{year}/{kind}';done=read(directory/'FIT_COMPLETE.json')
            need(done['plan_pin']==pin(plan_path) and done['status']=='FIT_AND_PREDICT_COMPLETE','Wrong saved model scope')
            mask=panel.prediction_eligible & panel.target_available & panel.outcome_month.between(f'{year-n}-01',f'{year-1}-12') & panel.label_available_at.le(f'{year-1}-12-31')
            test=panel.outcome_month.str.startswith(str(year));qualified=test&panel.prediction_eligible
            X=panel.loc[mask,list(FEATURES)].to_numpy(dtype=np.float32);y=panel.loc[mask,'target_class'].to_numpy(dtype=np.int64);XP=panel.loc[qualified,list(FEATURES)].to_numpy(dtype=np.float32)
            metadata=done['fit_metadata']
            def array_sha(a):return 'sha256:'+hashlib.sha256(memoryview(np.ascontiguousarray(a)).cast('B')).hexdigest()
            encoded=y if kind=='rf' else y-1
            need(array_sha(X)==metadata['prepared_train_matrix_sha256'] and array_sha(XP)==metadata['prepared_prediction_matrix_sha256']
              and array_sha(encoded)==metadata['encoded_training_labels_sha256'],'Independent ordered fit/predict matrices and labels')
            need(metadata['matrix_width']==15 and metadata['fit_count']==1 and metadata['explicit_parameters']==cfg['parameters'][kind],'Feature width/settings drift')
            estimator=joblib.load(check(done['model']));params=estimator.get_params(deep=False)
            need(all(params[k]==v for k,v in cfg['parameters'][kind].items()),'Saved estimator settings differ')
            need(list(estimator.classes_)==(list(range(1,11)) if kind=='rf' else list(range(10))),'Native class mapping')
            if kind=='rf':prob=estimator.predict_proba(XP)
            else:
                z=estimator.predict(XP,output_margin=True).astype(np.float64);z-=z.max(axis=1,keepdims=True);prob=np.exp(z);prob/=prob.sum(axis=1,keepdims=True)
            forecast=pq.read_table(check(done['predictions'])).to_pandas();expected=panel.loc[test]
            need(len(forecast)==len(expected) and list(map(tuple,forecast[['source_security_id','signal_date']].to_numpy()))==list(map(tuple,expected[['source_security_id','signal_date']].to_numpy())),'Original forecast keys dropped/reordered')
            for name in ('prediction_eligible','intrinio_security_id','outcome_month','outcome_start','outcome_end'):
                need(forecast[name].fillna('').tolist()==expected[name].fillna('').tolist(),'Forecast original identity/clock '+name)
            got=forecast.loc[forecast.prediction_eligible,list(PROBS)].to_numpy(dtype=float)
            need(np.allclose(prob,got,rtol=2e-12,atol=2e-12) and np.isfinite(got).all() and (got>=0).all() and np.allclose(got.sum(axis=1),1,rtol=0,atol=1e-8),'Independent saved-model inference')
            need(forecast.loc[~forecast.prediction_eligible,list(PROBS)].isna().all().all(),'Ineligible predictions invented')
            actual_labels=panel.loc[qualified,'target_class'].to_numpy(dtype=float);observed=np.isfinite(actual_labels);months=panel.loc[qualified,'outcome_month'].to_numpy()
            prior=np.bincount(y,minlength=11)[1:]/len(y);saved_summaries=read(check(done['summary']))
            for ref,values in [('model',prob),('uniform',np.full(prob.shape,.1)),('train_prior',np.tile(prior,(len(prob),1)))]:
                s=next(r for r in saved_summaries if r['reference']==ref);lab=actual_labels[observed].astype(int);v=values[observed];mo=months[observed]
                loss=-np.log(np.maximum(v[np.arange(len(lab)),lab-1],1e-15));acc=(v.argmax(axis=1)+1==lab).astype(float);errors=v.copy();errors[np.arange(len(lab)),lab-1]-=1;brier=np.square(errors).sum(axis=1)
                for key,val in [('pooled_log_loss',loss.mean()),('pooled_accuracy',acc.mean()),('pooled_brier_sum10',brier.mean()),
                   ('mean_monthly_log_loss',np.mean([loss[mo==m].mean() for m in sorted(set(mo))])),('mean_monthly_accuracy',np.mean([acc[mo==m].mean() for m in sorted(set(mo))])),
                   ('mean_monthly_brier_sum10',np.mean([brier[mo==m].mean() for m in sorted(set(mo))]))]:
                    need(abs(s[key]-val)<=2e-12,'Independent probability metric '+key)
                need(s['prediction_rows']==len(prob) and s['labelled_rows']==len(lab),'Metric denominators')
                all_metrics.append(s)
                if ref=='model':
                    # Independently reconcile each class's precision/recall and
                    # confusion counts, preserving zero-support classes.
                    cp=pq.read_table(check(done['metrics'][ref+'/mode_class_performance'])).to_pandas();cf=pq.read_table(check(done['metrics'][ref+'/confusion'])).to_pandas();pred=v.argmax(axis=1)+1
                    for k in range(1,11):
                        r=cp.loc[cp.period.eq('all')&cp.class_id.eq(k)].iloc[0];hits=int(((lab==k)&(pred==k)).sum());pn=int((pred==k).sum());an=int((lab==k).sum())
                        need(r.true_positive_count==hits and r.actual_support==an and r.labelled_predicted_support==pn,'Class supports/hits')
                        if pn:need(abs(r.precision-hits/pn)<=2e-12,'Class precision')
                        if an:need(abs(r.recall-hits/an)<=2e-12,'Class recall')
                        for j in range(1,11):need(int(cf.loc[cf.period.eq('all')&cf.actual_class.eq(k)&cf.predicted_class.eq(j),'count'].iloc[0])==int(((lab==k)&(pred==j)).sum()),'Confusion count')
            forecasts[f'{n}/{year}/{kind}']=done['predictions'];receipts[f'{n}/{year}/{kind}']=pin(directory/'FIT_COMPLETE.json');summaries.append({'years':n,'year':year,'kind':kind,'train_rows':len(X),'forecasts':len(prob),'observed_labels':int(observed.sum())})
            del estimator;guard()
    metrics_pin=save(root/'verification/prediction_metrics.json',all_metrics)
    return save(root/'verification/models.json',{'status':'PASS','plan_pin':pin(plan_path),'data_proof_pin':pin(root/'verification/data.json'),'verification_read_plan_pin':vplan,
       'worker_pin':pin(__file__),'verified_fits':12,'fit_receipts':receipts,'forecasts':forecasts,'metrics_pin':metrics_pin,'summary':summaries,'new_fits':0,'financial_scope':'verified monthly panels throughApril2026'})


def context(plan_path,phase):
    import pyarrow.parquet as pq
    p=read(plan_path)
    need(hashlib.sha256(json.dumps({k:v for k,v in p.items() if k!='plan_sha256'},sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()==p['plan_sha256'],'Independent plan digest')
    for item in [p['protocol_pin'],*p['source_binding_pins'],*p['source_inputs'].values(),*p['old_forecasts'].values(),*p['old_selected'].values(),*p['code_pins'].values(),*p['helper_pins'].values()]:check(item)
    need(p['max_fits']==12 and p['max_backtests']==0,'Twelve-fit/no-replay scope')
    cfg=read(check(p['protocol_pin']));root=Path(p['output_directory']);prep=read(root/'prepared/receipt.json')
    need(prep['plan_pin']==pin(plan_path),'Prepared plan binding')
    inventory=dict(prep['artifacts'])
    if phase in ('models','selection'):
        for n,y,k in itertools.product((5,10),(2024,2025,2026),('rf','xgb')):
            directory=root/f'folds/{n}y/{y}/{k}';done=read(directory/'FIT_COMPLETE.json')
            inventory[f'fit/{n}/{y}/{k}']=pin(directory/'FIT_COMPLETE.json')
            for name in ('model','predictions','summary'):inventory[f'{n}/{y}/{k}/{name}']=done[name]
            inventory.update({f'{n}/{y}/{k}/metric/{name}':item for name,item in done['metrics'].items()})
    if phase=='selection':
        audit=read(root/'selection/audit.json');inventory['selection_audit']=pin(root/'selection/audit.json')
        inventory.update({'selection/'+k:v for k,v in audit['artifacts'].items()})
    need(sum(x['size_bytes'] for x in inventory.values())<=cfg['budgets']['derived_read_bytes_per_phase'],'Independent derived read budget')
    for item in inventory.values():check(item)
    vp=save(root/'verification'/('read_plan_'+phase+'.json'),dict(status='SEALED_BEFORE_NUMERIC_READ',plan_pin=pin(plan_path),
        prepared_receipt_pin=pin(root/'prepared/receipt.json'),inventory=inventory,worker_pin=pin(__file__),new_fits=0,new_backtests=0))
    return p,cfg,root,prep,vp


def verify_data(plan_path,guard=lambda:None):
    import numpy as np,pandas as pd,pyarrow.parquet as pq
    p,cfg,root,prep,vp=context(plan_path,'data');proofs={};summary=[]
    cols=[*KEYS,'intrinio_security_id','identity_eligible','identity_status','identity_review_required','resolution_status',
        'prediction_eligible','prediction_status','outcome_month','outcome_start','outcome_end','history_years','history_complete',
        'expected_history_sessions','valid_history_sessions',*FEATURES,'label_return','target_class','target_available','label_available_at','target_entry_date','target_exit_date']
    for n in (5,10):
        guard();parts=[pq.read_table(check(p['source_inputs'][f'{n}/{kind}']),columns=cols).to_pandas() for kind in ('old','new')]
        expected=pd.concat(parts,ignore_index=True);panel=pq.read_table(check(prep['artifacts'][f'panel{n}'])).to_pandas()
        pd.testing.assert_frame_equal(panel,expected,check_exact=True)
        need(len(panel)==177767 and not panel.duplicated(KEYS).any(),'Original full panel denominator')
        need(panel.outcome_month.min()=='2010-01' and panel.outcome_month.max()=='2026-04','Exact financial model scope')
        actual=read(check(prep['artifacts'][f'payoffs{n}']));savedold=read(check(p['source_inputs'][f'{n}/payoffs_old']));savednew=read(check(p['source_inputs'][f'{n}/payoffs_new']))
        need(actual==dict(savedold,**savednew) and len(actual)==76,'Exact inherited payoff state')
        for cutoff,mu in actual.items():
            months=[shift(cutoff[:7],i) for i in range(-12*n+1,1)]
            eligible=panel.loc[panel.outcome_month.isin(months)&panel.label_available_at.le(cutoff)]
            group=eligible.groupby(['outcome_month','target_class']).label_return.mean()
            ordered=group.reindex(pd.MultiIndex.from_product([months,range(1,11)]))
            need(ordered.notna().all(),'Strict RET gate')
            np.testing.assert_allclose(np.asarray(mu),ordered.to_numpy().reshape(12*n,10).mean(axis=0),rtol=1e-13,atol=1e-14)
        for year in (2024,2025,2026):
            train=panel.outcome_month.ge(f'{year-n}-01')&panel.outcome_month.le(f'{year-1}-12')&panel.label_available_at.le(f'{year-1}-12-31')&panel.prediction_eligible&panel.target_available
            test=panel.outcome_month.ge(f'{year}-01')&panel.outcome_month.le(min(f'{year}-12','2026-04'));qualified=test&panel.prediction_eligible
            need(not (train&test).any() and panel.loc[train,'outcome_month'].nunique()==n*12,'Independent disjoint complete training calendar')
            need(panel.loc[train,'label_available_at'].max()<=f'{year-1}-12-31','No incompletely released label')
            X=panel.loc[train,list(FEATURES)].to_numpy(np.float32);y=panel.loc[train,'target_class'].to_numpy(np.int64);XP=panel.loc[qualified,list(FEATURES)].to_numpy(np.float32)
            need(X.shape[1]==XP.shape[1]==15 and np.isfinite(X).all() and np.isfinite(XP).all() and set(y)==set(range(1,11)),'No X/target imputation')
            def sha(a):return 'sha256:'+hashlib.sha256(memoryview(np.ascontiguousarray(a)).cast('B')).hexdigest()
            proofs[f'{n}/{year}']=dict(X=sha(X),y=sha(y),XP=sha(XP),training_rows=len(y),prediction_rows=len(XP),test_original_rows=int(test.sum()),
                train_first=f'{year-n}-01',train_last=f'{year-1}-12',label_cutoff=f'{year-1}-12-31')
        summary.append(dict(history_years=n,original_rows=len(panel),unchanged_old_rows=len(parts[0]),added_rows=len(parts[1]),complete_RET_vectors=len(actual)))
    return save(root/'verification/data.json',dict(status='PASS',plan_pin=pin(plan_path),prepared_receipt_pin=pin(root/'prepared/receipt.json'),
        verification_read_plan_pin=vp,matrix_proofs=proofs,summary=summary,worker_pin=pin(__file__),new_fits=0,new_backtests=0,
        exact_verified_source_reuse=True,causal_training_and_RET=True,source_route_continuation_not_certified=True))


def verify_selection(plan_path,guard=lambda:None):
    import numpy as np,pandas as pd,pyarrow.parquet as pq
    p,cfg,root,prep,vp=context(plan_path,'selection');audit=read(root/'selection/audit.json');mp=pin(root/'verification/models.json');model=read(check(mp))
    need(model['status']=='PASS' and model['verified_fits']==12 and audit['model_proof_pin']==mp,'Verified model selections required')
    metrics=[];annual=[];gaps=[];coverage=[]
    for n,kind in itertools.product((5,10),('rf','xgb')):
        guard();arm=f'{kind}_paper15_{n}y';a=audit['artifacts']
        panel=pq.read_table(check(prep['artifacts'][f'panel{n}'])).to_pandas();payoff=read(check(prep['artifacts'][f'payoffs{n}']))
        forecast=pq.read_table(check(a[arm+'/forecasts'])).to_pandas();ledger=pq.read_table(check(a[arm+'/original_ledger'])).to_pandas();picks=pq.read_table(check(a[arm+'/selected'])).to_pandas()
        need(forecast.outcome_month.nunique()==76 and len(forecast)==len(ledger),'Full76forecast ledger')
        need(not any(str(c).startswith(('target','label','future','realized')) for c in forecast),'No outcomes in selection inputs')
        savedparts=[]
        for year in range(2020,2027):
            item=p['old_forecasts'][f'{n}/{year}/{kind}'] if year<2024 else model['forecasts'][f'{n}/{year}/{kind}']
            savedparts.append(pq.read_table(check(item)).to_pandas()[forecast.columns])
        pd.testing.assert_frame_equal(forecast,pd.concat(savedparts,ignore_index=True),check_exact=True)
        wanted=[];scores={}
        for cutoff,g in forecast.groupby('signal_date',sort=True):
            admitted=g.loc[g.prediction_eligible];mu=payoff[cutoff]
            for row in admitted.itertuples():scores[(row.source_security_id,cutoff)]=math.fsum(float(getattr(row,name))*float(v) for name,v in zip(PROBS,mu))
            candidates=admitted.loc[~admitted.intrinio_security_id.isin(cfg['execution_restrictions'])]
            need(not candidates.intrinio_security_id.duplicated().any(),'Duplicate native selection')
            scored=[(row.source_security_id,scores[(row.source_security_id,cutoff)]) for row in candidates.itertuples()]
            count=min(10,len(scored)//2);long=sorted(scored,key=lambda r:(-r[1],r[0]))[:count];lset={r[0] for r in long}
            short=sorted((r for r in scored if r[0] not in lset),key=lambda r:(r[1],r[0]))[:count]
            for side,sgn,chosen in [('long',1,long),('short',-1,short)]:
                for sid,value in chosen:wanted.append((sid,cutoff,side,sgn*.5/count,value))
        found={(r.source_security_id,r.signal_date):(r.side,r.intended_weight,r.RET) for r in picks.itertuples()}
        need(len(found)==len(wanted) and len(picks)==76*20,'All balanced76monthly baskets')
        for sid,cutoff,side,weight,value in wanted:
            got=found[(sid,cutoff)];need(got[0]==side and abs(got[1]-weight)<1e-15 and abs(got[2]-value)<2e-12,'Independent RET top/bottom ten')
        need(ledger.selected.tolist()==[(sid,d) in found for sid,d in ledger[KEYS].itertuples(index=False,name=None)],'All original selection flags')
        for r in ledger.itertuples():
            key=(r.source_security_id,r.signal_date)
            if key in scores:need(abs(r.RET-scores[key])<2e-12,'Unselected score retained')
            else:need(pd.isna(r.RET),'Blocked probability/RET invented')
        old=pq.read_table(check(p['old_selected'][arm])).to_pandas();fields=[*KEYS,'side','intended_weight','RET']
        pd.testing.assert_frame_equal(picks.loc[picks.outcome_month.le('2023-12'),fields].sort_values(KEYS).reset_index(drop=True),old[fields].sort_values(KEYS).reset_index(drop=True),check_exact=True)
        future=picks.loc[picks.outcome_month.ge('2024-01')].merge(panel[[*KEYS,'target_available']],on=KEYS,how='left',validate='one_to_one')
        bad=future.loc[~future.target_available].copy();bad['arm']=arm;gaps.append(bad[['arm',*KEYS,'outcome_month','intrinio_security_id','side','RET']])
        coverage.append(dict(arm=arm,added_months=28,added_original_rows=int(forecast.outcome_month.ge('2024-01').sum()),
            added_predictions=int((forecast.outcome_month.ge('2024-01')&forecast.prediction_eligible).sum()),added_selected_rows=len(future),
            added_selected_monthly_endpoint_gaps=len(bad),full_forecast_months=76,common_candidate_months=75,old_selected_rows_byte_values_unchanged=True))
        # Aggregate from verified probability vectors and source labels, not means
        # of three unequal-length annual statistics. Every month weighs equally.
        for reference in ('model','uniform','train_prior'):
            rows=[]
            for year in (2024,2025,2026):
                yearpred=forecast.loc[forecast.outcome_month.str.startswith(str(year))&forecast.prediction_eligible].copy()
                labels=yearpred[KEYS+['outcome_month']].merge(panel[[*KEYS,'target_class']],on=KEYS,how='left',validate='one_to_one')
                prob=yearpred[list(PROBS)].to_numpy(float)
                if reference=='uniform':prob=np.full(prob.shape,.1)
                if reference=='train_prior':
                    train=panel.loc[panel.outcome_month.between(f'{year-n}-01',f'{year-1}-12')&panel.label_available_at.le(f'{year-1}-12-31')&panel.prediction_eligible&panel.target_available,'target_class'].to_numpy(int)
                    prior=np.bincount(train,minlength=11)[1:]/len(train);prob=np.tile(prior,(len(prob),1))
                observed=labels.target_class.notna().to_numpy();truth=labels.loc[observed,'target_class'].to_numpy(int);values=prob[observed]
                loss=-np.log(np.maximum(values[np.arange(len(truth)),truth-1],1e-15));acc=(values.argmax(axis=1)+1==truth).astype(float)
                errors=values.copy();errors[np.arange(len(truth)),truth-1]-=1;brier=np.square(errors).sum(axis=1)
                rows.append(pd.DataFrame(dict(month=labels.loc[observed,'outcome_month'].to_numpy(),loss=loss,accuracy=acc,brier=brier)))
                annual.append(dict(arm=arm,year=year,reference=reference,prediction_rows=len(prob),labelled_rows=len(truth),missing_labels=int((~observed).sum()),
                    mean_monthly_log_loss=float(rows[-1].groupby('month').loss.mean().mean()),mean_monthly_accuracy=float(rows[-1].groupby('month').accuracy.mean().mean())))
            full=pd.concat(rows,ignore_index=True);monthly=full.groupby('month')[['loss','accuracy','brier']].mean()
            metrics.append(dict(arm=arm,reference=reference,first_outcome='2024-01',last_outcome='2026-04',months=len(monthly),
                prediction_rows=int((forecast.outcome_month.ge('2024-01')&forecast.prediction_eligible).sum()),labelled_rows=len(full),
                missing_labels=int((forecast.outcome_month.ge('2024-01')&forecast.prediction_eligible).sum())-len(full),
                mean_monthly_log_loss=float(monthly.loss.mean()),mean_monthly_accuracy=float(monthly.accuracy.mean()),mean_monthly_brier_sum10=float(monthly.brier.mean()),
                pooled_log_loss=float(full.loss.mean()),pooled_accuracy=float(full.accuracy.mean())))
    expected_gaps=pd.concat(gaps,ignore_index=True);got_gaps=pq.read_table(check(audit['artifacts']['selected_endpoint_gaps'])).to_pandas()
    pd.testing.assert_frame_equal(expected_gaps,got_gaps,check_exact=True)
    pd.testing.assert_frame_equal(pd.DataFrame(coverage),pq.read_table(check(audit['artifacts']['coverage'])).to_pandas(),check_exact=True)
    report=save(root/'verification/summary.json',dict(status='VERIFIED_MODELS_AND_RET_SELECTIONS_ONLY',metrics=metrics,annual_metrics=annual,selection_coverage=coverage,
        completed_fits=12,new_backtests=0,old_models_and_selections_preserved=True,first_added_outcome='2024-01',last_added_outcome='2026-04',
        common_candidate_months=75,post2023_BK_XOM_routes_unresolved=True,
        limitations=['Data-limited audited identity/history population; BK/XOM interval limits retained','Strict monthly learning labels/RET unchanged; missing outcomes remain unlabelled','No new portfolio returns: action/quote-aware75month realization is next']))
    return save(root/'verification/selection.json',dict(status='PASS',plan_pin=pin(plan_path),model_proof_pin=mp,audit_pin=pin(root/'selection/audit.json'),
        verification_read_plan_pin=vp,summary_pin=report,worker_pin=pin(__file__),verified_monthly_baskets=304,
        additional_monthly_baskets=112,old_baskets_unchanged=True,new_fits=0,new_backtests=0))
