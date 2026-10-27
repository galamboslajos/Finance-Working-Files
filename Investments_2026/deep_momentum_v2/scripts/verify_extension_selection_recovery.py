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
def verify_selection(plan_path,guard=lambda:None):
    import numpy as np,pandas as pd,pyarrow.parquet as pq
    p,cfg,root,prep,vp=context(plan_path,'selection');audit=read(root/'selection/audit.json');mp=p['inherited_model_proof_pin'];model=read(check(mp))
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
        left=picks.loc[picks.outcome_month.le('2023-12'),fields].sort_values(KEYS).reset_index(drop=True)
        right=old[fields].sort_values(KEYS).reset_index(drop=True)
        left['side']=left['side'].astype('string');right['side']=right['side'].astype('string')
        pd.testing.assert_frame_equal(left,right,check_exact=True)
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


def context(plan_path,phase):
    from dmv2.extension_selection_recovery import validate
    recovery=validate(plan_path);p=read(check(recovery['parent_plan_pin']));cfg=read(check(p['protocol_pin']))
    p['inherited_model_proof_pin']=recovery['model_proof_pin'];root=Path(recovery['output_directory']);prep=read(check(recovery['prepared_receipt_pin']))
    audit=read(root/'selection/audit.json');need(audit['plan_pin']==pin(plan_path),'Recovered selection plan')
    need(audit['model_proof_pin']==recovery['model_proof_pin'],'Original model proof binding')
    inventory={**prep['artifacts'],**{'selection/'+k:v for k,v in audit['artifacts'].items()}}
    model=read(check(recovery['model_proof_pin']));inventory.update({'new_forecast/'+k:v for k,v in model['forecasts'].items()})
    inventory.update({'old_forecast/'+k:v for k,v in p['old_forecasts'].items()});inventory.update({'old_selected/'+k:v for k,v in p['old_selected'].items()})
    need(sum(x['size_bytes'] for x in inventory.values())<=cfg['budgets']['derived_read_bytes_per_phase'],'Independent zero-fit read budget')
    for item in inventory.values():check(item)
    vp=save(root/'verification/read_plan_selection.json',dict(status='SEALED_BEFORE_RECOVERED_SELECTION_CHECK',plan_pin=pin(plan_path),
        prepared_receipt_pin=recovery['prepared_receipt_pin'],inventory=inventory,worker_pin=pin(__file__),new_fits=0,new_backtests=0))
    return p,cfg,root,prep,vp
