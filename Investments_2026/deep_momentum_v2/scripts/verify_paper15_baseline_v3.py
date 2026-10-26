#!/usr/bin/env python3
"""Independent source/history, Paper15, decile, RET, inference and cash proofs.

Does not import production baseline calculators, model fitting or accounting.
Saved estimators are loaded only in the explicit model verification phase.
"""
from pathlib import Path
import hashlib,importlib.util,itertools,json,math,sys
from collections import defaultdict

FEATURES=tuple(p+str(h) for p in ('zMOM','meanMOM','stdMOM') for h in (1,3,6,9,12))
PROBS=tuple('probability_'+str(k) for k in range(1,11))
COHORT=('source_security_id','signal_date','intrinio_security_id','identity_eligible','identity_status','identity_review_required','resolution_status','indices','membership_tickers','unified_security_id','source_identity_status','identity_eligibility_policy')
OHLC=('open','high','low','close','adj_open','adj_high','adj_low','adj_close')


def need(v,m):
    if not v:raise ValueError(m)

def pin(path):
    p=Path(path).resolve();h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return dict(path=str(p),size_bytes=p.stat().st_size,sha256=h.hexdigest())

def check(item):
    p=Path(item['path']);need(pin(p)=={k:item[k] for k in ('path','size_bytes','sha256')},'Independent pinned bytes changed');return p

def read(path):
    p=Path(path);need(p.stat().st_size<=32*1024**2,'Independent JSON ceiling');return json.loads(p.read_text())

def save(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
    return pin(p)

def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def shift(month,offset):
    y,m=map(int,month.split('-'));n=y*12+m-1+offset;return f'{n//12:04d}-{n%12+1:02d}'

def context(plan_path,phase):
    p=read(plan_path);body={k:v for k,v in p.items() if k!='plan_sha256'}
    need(hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()==p['plan_sha256'],'Independent plan identity')
    for item in [p['protocol_pin'],*p['source_binding_pins'],*p['code_pins'].values(),*p['helper_pins'].values(),*p['verifier_helper_pins'].values()]:check(item)
    cfg=read(check(p['protocol_pin']));root=Path(p['output_directory']);prep=read(root/'prepared/receipt.json')
    need(prep['plan_pin']==pin(plan_path) and p['max_fits']==16 and p['financial_2024_plus_read'] is False,'Independent scope')
    for item in prep['artifacts'].values():check(item)
    receipt=save(root/'verification'/('read_plan_'+phase+'.json'),{'status':'SEALED_BEFORE_NUMERIC_READ','plan_pin':pin(plan_path),
      'prepared_receipt_pin':pin(root/'prepared/receipt.json'),'worker_pin':pin(__file__),'prepared_artifacts':prep['artifacts'],
      'source_projection_bytes':p['source_projection_bytes'],'maximum_source_projection_bytes':cfg['budgets']['source_projection_bytes_per_pass'],'new_fits':0})
    return p,cfg,root,prep,receipt


def source_projection(spec,natives):
    import pyarrow as pa,pyarrow.compute as pc,pyarrow.parquet as pq
    pf=pq.ParquetFile(check(spec['pin']));cost=0
    for g in spec['row_groups']:
        rg=pf.metadata.row_group(g);cost+=sum(rg.column(c).total_compressed_size for c in range(rg.num_columns) if rg.column(c).path_in_schema in spec['columns'])
    need(cost==spec['projected_compressed_bytes'],'Independent source footer reservation')
    for g in spec['row_groups']:
        keys=pf.read_row_group(g,columns=['intrinio_security_id','trading_date'],use_threads=False)
        need(all(a.null_count==0 for a in keys.columns),'Null original native/date keys')
        m=pc.and_(pc.is_in(keys['intrinio_security_id'],value_set=pa.array(natives)),pc.and_(pc.greater_equal(keys['trading_date'],spec['start']),pc.less_equal(keys['trading_date'],spec['end'])))
        if not pc.any(m).as_py():continue
        selected=keys.filter(m)
        for name in (c for c in spec['columns'] if c not in keys.column_names):
            selected=selected.append_column(name,pf.read_row_group(g,columns=[name],use_threads=False)[name].filter(m))
        yield selected.to_pandas()


def daily_return_oracle(prices,quality,days,ids,positions,native,entry,exit_date,events):
    import numpy as np
    if exit_date>='2024-01-01':return np.nan
    if native not in ids or entry not in positions or exit_date not in positions:return np.nan
    a,b=positions[entry],positions[exit_date];n=ids[native]
    if any(e['predecessor_id']==native and e['effective_date']<=entry for e in events):return np.nan
    if not quality[a,n,0]:return np.nan
    relevant=[e for e in events if entry<e['effective_date']<=exit_date]
    if not any(e['predecessor_id']==native for e in relevant):
        if not quality[a:b+1,n,0].all() or not quality[a:b,n,1].all():return np.nan
        return float(prices[b,n,2]/prices[a,n,2]-1)
    holdings=1./prices[a,n,2];cash=0.
    for i in range(a,b+1):
        if i>a and holdings:
            matching=[e for e in relevant if e['effective_date']==days[i] and e['predecessor_id']==native]
            if len(matching)>1:raise ValueError('Ambiguous same-day target action')
            if matching:
                e=matching[0]
                if not quality[i-1,n,1]:return np.nan
                raw_shares=holdings*(prices[i-1,n,3]/prices[i-1,n,1]);cash=cash+raw_shares*e['cash_per_raw_share']
                if e['stock_ratio']:
                    native=e['successor_id'];n=ids.get(native)
                    if n is None or not quality[i,n,0]:return np.nan
                    holdings=raw_shares*e['stock_ratio']*prices[i,n,0]/prices[i,n,2]
                else:holdings=0.
        for mark in ([0] if i==b else [0,1]):
            if holdings and not quality[i,n,mark]:return np.nan
            wealth=cash+(holdings*prices[i,n,mark+2] if holdings else 0.)
            if not math.isfinite(wealth):return np.nan
    return float(wealth-1)


def verify_data(plan_path,guard=lambda:None):
    import numpy as np,pandas as pd,pyarrow.parquet as pq,exchange_calendars as xc
    p,cfg,root,prep,vplan=context(plan_path,'data');art=prep['artifacts']
    need(p['source_projection_bytes']<=cfg['budgets']['source_projection_bytes_per_pass'],'Independent source read ceiling')
    action_source=next(a for a in p['source_binding_pins'] if a['path'].endswith('full_path_target_audit/2010-2023-20261022-v1/reviewed_plan.json'))
    need(p['target_events'][:29]==read(check(action_source))['events'] and p['target_events'][29:]==cfg['backtest']['additional_reviewed_actions'] and len(p['target_events'])==31,'Independent full legal action terms')
    tail=list(xc.get_calendar('XNYS',start='2024-01-01',end='2024-01-03').sessions.strftime('%Y-%m-%d'))
    need(tail==p['target_exit_calendar_metadata'],'Protected endpoint calendar metadata differs')

    cohorts=pd.concat([pq.read_table(check(x['pin']),columns=list(COHORT)).to_pandas() for x in p['cohorts']],ignore_index=True)
    saved=pq.read_table(check(art['cohorts'])).to_pandas();pd.testing.assert_frame_equal(cohorts,saved,check_exact=True)
    need(len(cohorts)==152307 and not cohorts.duplicated(['source_security_id','signal_date']).any(),'Original historical cohort denominator')
    index=read(check(art['index']));days=list(xc.get_calendar('XNYS',start='2000-01-01',end='2023-12-31').sessions.strftime('%Y-%m-%d'))
    need(days==index['sessions']==p['sessions'],'Independent exchange calendar differs')
    natives=sorted(set(cohorts.loc[cohorts.identity_eligible,'intrinio_security_id'].dropna())|{e[k] for e in p['target_events'] for k in ('predecessor_id','successor_id') if e.get(k)});need(natives==index['natives'],'Native inventory from original admitted cohorts')
    months=sorted(set(d[:7] for d in days));ends={m:max(d for d in days if d.startswith(m)) for m in months};need(months==index['months'],'Monthly endpoint calendar')
    ds=pd.Index(days);ns=pd.Index(natives);end_index={d:i for i,d in enumerate(ends.values())}
    daily=np.full((len(days),len(natives),4),np.nan);quote_validity=np.zeros((len(days),len(natives),2),bool)
    seen=np.zeros((len(days),len(natives)),bool);valid=np.zeros_like(seen);corrected_seen=np.zeros_like(seen);closes=np.full((len(months),len(natives)),np.nan)
    for no,spec in enumerate([*p['raw_prefix'],*p['corrected_daily']]):
        guard()
        for q in source_projection(spec,natives):
            need(q.trading_date.min()>=spec['start'] and q.trading_date.max()<=spec['end']<'2024-01-01','Protected values converted before filter')
            d=ds.get_indexer(q.trading_date);n=ns.get_indexer(q.intrinio_security_id);take=(d>=0)&(n>=0);q=q.loc[take].reset_index(drop=True);d=d[take];n=n[take]
            need(len(set(zip(d,n)))==len(d),'Independent duplicate within price block')
            if no<len(p['raw_prefix']):need(not seen[d,n].any(),'Independent raw duplicate')
            else:
                need(not corrected_seen[d,n].any(),'Independent corrected duplicate');corrected_seen[d,n]=True
            ok=q.frequency.eq('daily')&q.price_quality_status.isin(['accepted','ok'])&q.intraperiod.eq(False)
            numeric=q[list(OHLC)].astype(float)
            complete=ok.to_numpy()&np.isfinite(numeric.to_numpy()).all(axis=1)&numeric.gt(0).all(axis=1).to_numpy()
            complete &= q.raw_ohlc_bounds_valid.eq(True).to_numpy()&q.adjusted_ohlc_bounds_valid.eq(True).to_numpy()
            for prefix in ('','adj_'):
                high=numeric[prefix+'high'];low=numeric[prefix+'low']
                complete &= high.ge(numeric[[prefix+'open',prefix+'close',prefix+'low']].max(axis=1)).to_numpy()
                complete &= low.le(numeric[[prefix+'open',prefix+'close',prefix+'high']].min(axis=1)).to_numpy()
            seen[d,n]=True;valid[d,n]=complete
            daily[d,n]=numeric[['open','close','adj_open','adj_close']].to_numpy()
            accepted=ok.to_numpy()&q.raw_ohlc_bounds_valid.eq(True).to_numpy()&q.adjusted_ohlc_bounds_valid.eq(True).to_numpy()
            for mark,names in [(0,['open','adj_open']),(1,['close','adj_close'])]:
                vv=numeric[names].to_numpy();quote_validity[d,n,mark]=accepted&np.isfinite(vv).all(axis=1)&(vv>0).all(axis=1)
            take=ok.to_numpy()&np.isfinite(numeric.adj_close.to_numpy())&numeric.adj_close.gt(0).to_numpy()&q.trading_date.isin(end_index).to_numpy()
            for i in np.flatnonzero(q.trading_date.isin(end_index).to_numpy()):closes[end_index[q.trading_date.iloc[i]],n[i]]=np.nan
            for i in np.flatnonzero(take):closes[end_index[q.trading_date.iloc[i]],n[i]]=numeric.adj_close.iloc[i]
        if (no+1)%10==0:print(json.dumps({'phase':'independent_source_reconstruction','files':no+1}),flush=True)
    for name,expected in [('valid',valid),('present',seen),('month_closes',closes),('daily_prices',daily),('quote_validity',quote_validity)]:
        actual=np.load(check(art[name]),allow_pickle=False);need(actual.shape==expected.shape and np.array_equal(actual,expected,equal_nan=True),'Independent full source matrix '+name)
    bad={native:np.flatnonzero(~valid[:,j]) for j,native in enumerate(natives)};ni={n:i for i,n in enumerate(natives)};mi={m:i for i,m in enumerate(months)}
    positions={d:i for i,d in enumerate(days)};summaries=[]
    for years in (5,10):
        panel=pq.read_table(check(art[f'panel{years}'])).to_pandas();coverage=pq.read_table(check(art[f'coverage{years}'])).to_pandas()
        need(len(panel)==len(cohorts) and len(coverage)==168,'Original monthly panel length')
        merged=cohorts.merge(panel,on=['source_security_id','signal_date'],suffixes=('_original',''),validate='one_to_one')
        need((merged.intrinio_security_id.fillna('')==merged.intrinio_security_id_original.fillna('')).all()
          and (merged.identity_eligible==merged.identity_eligible_original).all(),'Identity route/admission changed')
        for sig,group in panel.groupby('signal_date',sort=True):
            guard();lower=(pd.Timestamp(sig)-pd.DateOffset(years=years)).strftime('%Y-%m-%d');lo=ds.searchsorted(lower,side='right');hi=ds.searchsorted(sig,side='right')
            raw=[];admit=[];observed=[]
            for r in group.to_dict('records'):
                native=r['intrinio_security_id'];bad_positions=bad.get(native);missing=(int(np.searchsorted(bad_positions,hi)-np.searchsorted(bad_positions,lo)) if bad_positions is not None else hi-lo)
                passed=missing==0;observed.append(hi-lo-missing);admit.append(bool(r['identity_eligible']) and passed)
                need(r['history_complete']==passed and r['expected_history_sessions']==hi-lo and r['valid_history_sessions']==hi-lo-missing
                  and r['history_start_exclusive']==lower and r['history_end']==sig,'Independent signal-time complete-history gate')
                vals=[]
                for h in (1,3,6,9,12):
                    required=[shift(sig[:7],-1),sig[:7]] if h==1 else [shift(sig[:7],z) for z in range(-h,0)]
                    seq=np.array([closes[mi[m],ni[native]] if native in ni else np.nan for m in required])
                    vals.append(float(np.prod(seq[1:]/seq[:-1])-1) if passed and r['identity_eligible'] and np.isfinite(seq).all() and (seq>0).all() else np.nan)
                raw.append(vals)
            raw=np.asarray(raw);common=np.asarray(admit)&np.isfinite(raw).all(axis=1);expected=np.full((len(group),15),np.nan)
            if common.sum()>=2:
                mu=np.mean(raw[common],axis=0);sd=np.std(raw[common],axis=0,ddof=1)
                if np.isfinite(sd).all() and (sd>0).all():expected[common]=np.c_[(raw[common]-mu)/sd,np.tile(mu,(common.sum(),1)),np.tile(sd,(common.sum(),1))]
            need(np.allclose(group[list(FEATURES)].to_numpy(dtype=float),expected,rtol=2e-12,atol=2e-12,equal_nan=True),'Independent15feature formulas/population')
            eligible=np.isfinite(expected).all(axis=1);need(np.array_equal(eligible,group.prediction_eligible.to_numpy()),'Prediction qualification differs')
            future=shift(sig[:7],1);exit_month=shift(future,1)
            entry=next(d for d in days if d.startswith(future));exit_date=next((d for d in [*days,*p['target_exit_calendar_metadata']] if d.startswith(exit_month)),None)
            need(exit_date is not None and group.label_available_at.eq(exit_date).all() and group.target_entry_date.eq(entry).all() and group.target_exit_date.eq(exit_date).all(),'Independent executable completion clock')
            ret=[]
            for j,r in enumerate(group.to_dict('records')):
                ret.append(daily_return_oracle(daily,quote_validity,days,ni,positions,r['intrinio_security_id'],entry,exit_date,p['target_events']) if eligible[j] else np.nan)
            ret=np.array(ret);need(np.allclose(ret,group.label_return.to_numpy(dtype=float),rtol=1e-13,atol=1e-13,equal_nan=True),'Independent executable daily-path outcomes')
            labels=np.full(len(ret),np.nan);good=np.isfinite(ret)
            if good.sum()>=10:
                ranks=pd.Series(group.label_return.to_numpy(dtype=float)[good]).rank(ascending=False,method='average').to_numpy();labels[good]=np.clip(1+np.floor(10*(ranks-.5)/good.sum()),1,10)
            need(np.array_equal(labels,group.target_class.to_numpy(dtype=float),equal_nan=True),'Independent exact-tie class ranks')
            need(np.array_equal(np.isfinite(labels),group.target_available.to_numpy()),'Label states differ')
            cov=coverage.loc[coverage.signal_date.eq(sig)];need(len(cov)==1 and int(cov.iloc[0].prediction_eligible)==int(eligible.sum()) and int(cov.iloc[0].original_members)==len(group),'Coverage original denominator')
        payoffs=read(check(art[f'payoffs{years}']));grid=pq.read_table(check(art[f'payoff_grid{years}'])).to_pandas()
        grid_index={(r['signal_date'],r['outcome_month'],int(r['class_id'])):r for r in grid.to_dict('records')}
        for sig,actual in payoffs.items():
            months_needed=[shift(sig[:7],j) for j in range(-12*years,0)]
            hist=panel.loc[panel.outcome_month.isin(months_needed)&panel.label_available_at.le(sig)&panel.target_available]
            groups={key:part.label_return.to_numpy(dtype=float) for key,part in hist.groupby(['outcome_month','target_class'],sort=False)}
            means=[];missing=False
            for k in range(1,11):
                returns=[]
                for month in months_needed:
                    v=groups.get((month,float(k)),np.array([],dtype=float))
                    cell=grid_index.get((sig,month,k));need(cell is not None and int(cell['count'])==len(v),'Independent every RET month/class count')
                    if len(v):
                        value=float(np.mean(v));need(abs(value-cell['mean_return'])<=2e-12,'Independent class mean');returns.append(value)
                    else:missing=True;need(pd.isna(cell['mean_return']),'Missing class mean imputed')
                means.append(float(np.mean(returns)) if len(returns)==12*years else np.nan)
            need((actual is None)==missing,'RET complete-calendar gate')
            if actual is not None:need(np.allclose(actual,means,rtol=2e-12,atol=2e-12),'Independent RET aggregation')
        summaries.append({'years':years,'original_rows':len(panel),'prediction_eligible':int(panel.prediction_eligible.sum()),'RET_complete_vectors':sum(v is not None for v in payoffs.values()),'RET_forecast_months':len(payoffs)})
    return save(root/'verification/data.json',{'status':'PASS','plan_pin':pin(plan_path),'prepared_receipt_pin':pin(root/'prepared/receipt.json'),
      'verification_read_plan_pin':vplan,'worker_pin':pin(__file__),'original_stock_months':152307,'source_validity_cells_reconstructed':int(valid.size),
      'separate_feature_cohorts_and_deciles_verified':True,'new_fits':0,'financial_2024_plus_read':False,'summary':summaries})


def verify_models(plan_path,guard=lambda:None):
    import joblib,numpy as np,pandas as pd,pyarrow.parquet as pq
    p,cfg,root,prep,vplan=context(plan_path,'models');data=read(root/'verification/data.json')
    need(data['status']=='PASS' and data['prepared_receipt_pin']==pin(root/'prepared/receipt.json'),'Independent prepared source proof')
    receipts={};forecasts={};summaries=[];all_metrics=[]
    for n in (5,10):
        panel=pq.read_table(check(prep['artifacts'][f'panel{n}'])).to_pandas()
        for year,kind in itertools.product((2020,2021,2022,2023),('rf','xgb')):
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
       'worker_pin':pin(__file__),'verified_fits':16,'fit_receipts':receipts,'forecasts':forecasts,'metrics_pin':metrics_pin,'summary':summaries,'new_fits':0,'financial_2024_plus_read':False})

def independent_common_preflight(selected, rebalances, quote_metadata, sessions, events, historical):
    """Enumerate every selected unstopped horizon using dated legal terms only."""
    need(len(events) == 20 and len(historical) == 11, 'Frozen current/prior action scope changed')
    dates = list(sessions); index = {d: i for i, d in enumerate(dates)}
    schedule = [r['execution_date'] for r in rebalances]
    need(len(schedule) == len(set(schedule)) == 47 and all(d in index for d in schedule),
         'Full monthly rebalance grid required')
    exits = {d: schedule[i+1] if i+1 < len(schedule) else dates[-1] for i, d in enumerate(schedule)}
    quotes = {(r['trading_date'], r['intrinio_security_id']): r for r in quote_metadata}
    need(len(quotes) == len(quote_metadata), 'Duplicate preflight quote metadata')
    current = defaultdict(list)
    for event in sorted(events, key=lambda e: (e['effective_date'], e['event_id'])):
        need(event['effective_date'] in index, 'Current action is not an authorized session')
        current[event['effective_date']].append(event)
    retired = sorted([*historical, *events], key=lambda e: (e['effective_date'], e['event_id']))
    required, missing, bad, unfilled, actions = set(), [], [], [], []

    def require_quote(episode, native, d, role):
        required.add((d, native))
        details = {k: episode[k] for k in ('signal_date', 'execution_date', 'source_security_id', 'side')}
        details.update(scheduled_exit_date=exits[episode['execution_date']], intrinio_security_id=native, date=d, role=role)
        quote = quotes.get((d, native))
        if quote is None:
            missing.append({**details, 'reason': 'missing_key_requires_observed_quote_or_verified_legal_action'})
        else:
            if 'price_accepted' in quote:
                accepted = type(quote['price_accepted']) is bool and quote['price_accepted']
            else:
                accepted = quote.get('frequency') == 'daily' and quote.get('intraperiod') is False \
                    and quote.get('price_quality_status') in ('accepted', 'ok')
            accepted = accepted and all(type(quote.get(k)) is bool and quote[k]
                for k in ('raw_ohlc_bounds_valid', 'adjusted_ohlc_bounds_valid') if k in quote)
            if not accepted:
                bad.append({**details, 'reason': 'unaccepted_source_metadata_requires_independent_review'})

    for row in selected:
        native, entry = row['intrinio_security_id'], row['execution_date']
        need(entry in exits, 'Selected episode has no frozen rebalance')
        retirement = next((e for e in reversed(retired)
            if e['predecessor_id'] == native and e['effective_date'] <= entry), None)
        if retirement:
            unfilled.append({k: row[k] for k in ('signal_date', 'execution_date', 'source_security_id', 'intrinio_security_id')}
                | {'event_id': retirement['event_id'], 'status': 'retired_new_order_unfilled_cash_no_entitlement'})
            continue
        for d in dates[index[entry]:index[exits[entry]]+1]:
            for e in current.get(d, []):
                if e['predecessor_id'] == native:
                    need(index[d] > 0, 'Held conversion lacks prior authorized session')
                    require_quote(row, native, dates[index[d]-1], 'action_predecessor_paired_close')
                    actions.append({'signal_date': row['signal_date'], 'source_security_id': row['source_security_id'],
                        'date': d, 'event_id': e['event_id'], 'predecessor_id': native,
                        'successor_id': e['successor_id'], 'status': 'verified_retirement_definition_required_and_present'})
                    native = e['successor_id'] if e['stock_ratio'] > 0 else None
            if native is None: break
            role = 'entry_open' if d == entry else 'scheduled_exit_open' if d == exits[entry] and d != dates[-1] else 'held_open_and_close'
            require_quote(row, native, d, role)
    summary = {'selected_episode_rows': len(selected), 'distinct_maximum_horizon_quote_keys': len(required),
        'missing_key_episode_rows': len(missing), 'quality_flag_episode_rows': len(bad),
        'unique_missing_keys': len({(r['date'], r['intrinio_security_id']) for r in missing}),
        'unique_quality_flag_keys': len({(r['date'], r['intrinio_security_id']) for r in bad}),
        'verified_current_action_definitions': 20, 'historical_retirement_definitions': 11,
        'retired_new_orders': len(unfilled), 'ready_for_accounting': not missing and not bad,
        'source_metadata_only': True, 'prices_or_outcomes_read': False, 'selection_changed': False,
        'unstopped_maximum_horizons_reviewed_for_both_stop_variants': True}
    return {'summary': summary, 'missing_quotes': missing, 'source_quality': bad,
            'retired_orders': unfilled, 'action_requirements': actions}



def record_key(row):
    import pandas as pd
    r=dict(row)
    for k in ('date','signal_date','execution_date','scheduled_exit_date'):
        if k in r:r[k]=pd.Timestamp(r[k]).strftime('%Y-%m-%d')
    return json.dumps(r,sort_keys=True,allow_nan=False)


def bt_projection(spec):
    import pyarrow.parquet as pq
    pf=pq.ParquetFile(check(spec['pin']));cost=sum(pf.metadata.row_group(g).column(c).total_compressed_size for g in range(pf.metadata.num_row_groups)
      for c in range(pf.metadata.row_group(g).num_columns) if pf.metadata.row_group(g).column(c).path_in_schema in spec['columns'])
    need(cost==spec['projected_compressed_bytes'] and pf.metadata.num_rows==spec['rows'],'Independent execution projection budget')
    return pf.read(columns=spec['columns'],use_threads=False)


def verify_backtest(plan_path,guard=lambda:None):
    import numpy as np,pandas as pd,pyarrow as pa,pyarrow.parquet as pq
    p,cfg,root,prep,vplan=context(plan_path,'backtest');out=root/'backtest';audit=read(out/'audit.json');models=read(root/'verification/models.json')
    need(audit['plan_pin']==pin(plan_path) and audit['model_proof_pin']==pin(root/'verification/models.json') and models['status']=='PASS'
       and models['verified_fits']==16 and audit['scenarios']==32,'Verified original models/backtest scope')
    inventory=audit['artifacts'];need(sum(a['size_bytes'] for a in inventory.values())<=cfg['budgets']['derived_verification_bytes'],'Independent derived read ceiling')
    for a in inventory.values():check(a)
    bvplan=save(out/'verification/read_plan.json',{'status':'SEALED_BEFORE_NUMERIC_READ','plan_pin':pin(plan_path),'audit_pin':pin(out/'audit.json'),
      'worker_pin':pin(__file__),'artifacts':inventory,'new_fits':0,'source_projection_bytes':sum(v[k]['projected_compressed_bytes'] for v in [*p['execution_source']['daily'],p['execution_source']['repairs']] for k in ('metadata','values'))})
    equations=module(check(p['verifier_helper_pins']['verify_walkforward_backtest_v2.py']),'_paper15_independent_equations')
    overlay=module(check(p['verifier_helper_pins']['verify_ret_repair_backtest.py']),'_paper15_independent_overlay')
    bt=p['execution_source'];phase=p['execution_phase'];sessions=bt['sessions'];session_set=set(sessions);need(len(sessions)==985 and sessions[0]=='2020-02-03' and sessions[-1]=='2023-12-29','Common47monthclock')
    for a in [*bt['execution_pins'].values(),*bt['execution_evidence_pins']]:check(a)
    source_phase=read(check(bt['execution_pins']['execution_phase_pin']));copy=json.loads(json.dumps(phase));copy['event_registry']['events']=copy['event_registry']['events'][:-2]
    need(copy==source_phase and phase['event_registry']['events'][-2:]==cfg['backtest']['additional_reviewed_actions'],'Execution amendment scope changed')
    original=[]
    for pair in bt['daily']:
        meta=bt_projection(pair['metadata']);rows=meta.to_pylist();need(all('2010-01-01'<=r['trading_date']<'2024-01-01' for r in rows),'Protected execution date')
        positions=[i for i,r in enumerate(rows) if r['trading_date'] in session_set]
        vals=bt_projection(pair['values']);need(vals.select(meta.column_names).equals(meta),'Execution metadata changed')
        original.append(vals.take(pa.array(positions,type=pa.int64())).to_pandas())
    pair=bt['repairs'];meta=bt_projection(pair['metadata']);need(all(r['trading_date'] in session_set for r in meta.to_pylist()),'Overlay scope')
    vals=bt_projection(pair['values']);need(vals.select(meta.column_names).equals(meta),'Overlay keys changed')
    manifest=read(check(bt['execution_pins']['execution_manifest_pin']));hashes={a['sha256'] for a in bt['execution_evidence_pins']}
    def rejection(change):
        item=change['rejection_evidence_pin'];need(item['sha256'] in hashes,'Unpinned rejection')
        check({'path':str((Path(p['data_root'])/item['path']).resolve()),'sha256':item['sha256'],'size_bytes':item.get('size_bytes',item.get('bytes'))})
    prices,changes=overlay.reconstruct_overlay(pd.concat(original,ignore_index=True),vals.to_pandas(),manifest['quote_amendments'],sessions=sessions,authenticate_rejection=rejection)
    need(len(changes)==40,'Original40 execution amendments');del original
    quotes={(r['trading_date'],r['intrinio_security_id']):r for r in prices.to_dict('records')}
    qmeta=[{k:r[k] for k in ('trading_date','intrinio_security_id','frequency','intraperiod','price_quality_status','raw_ohlc_bounds_valid','adjusted_ohlc_bounds_valid')} for r in quotes.values()]
    def saved(n):return pq.read_table(check(inventory[n]))
    comp=saved('comparison').to_pandas();annuals=saved('annual').to_pandas();need(len(comp)==32 and len(annuals)==128,'32scenarios/128annuals')
    results=[]
    for years,kind in itertools.product((5,10),('rf','xgb')):
        arm=f'{kind}_paper15_{years}y';mu=read(check(prep['artifacts'][f'payoffs{years}']));parts=[]
        for y in (2020,2021,2022,2023):parts.append(pq.read_table(check(models['forecasts'][f'{years}/{y}/{kind}'])).to_pandas())
        predictions=pd.concat(parts,ignore_index=True);selected=saved(arm+'/selection/selected').to_pandas();rb=saved(arm+'/selection/rebalances').to_pandas();ledger=saved(arm+'/selection/original_ledger').to_pandas()
        need(len(predictions)==len(ledger) and list(map(tuple,predictions[['source_security_id','signal_date']].to_numpy()))==list(map(tuple,ledger[['source_security_id','signal_date']].to_numpy())),'Original prediction ledger changed')
        for sig,part in predictions.groupby('signal_date',sort=True):
            scores={}
            for r in part.to_dict('records'):
                if r['prediction_eligible'] and r['intrinio_security_id'] not in cfg['backtest']['execution_restrictions'] and mu[sig] is not None:
                    scores[r['source_security_id']]=math.fsum(r['probability_'+str(k)]*mu[sig][k-1] for k in range(1,11))
            n=min(10,len(scores)//2);long=sorted(scores,key=lambda sid:(-scores[sid],sid))[:n];short=sorted((sid for sid in scores if sid not in long),key=lambda sid:(scores[sid],sid))[:n]
            if part.outcome_month.iloc[0]=='2020-01':n=0;long=[];short=[]
            own=selected.loc[selected.signal_date.eq(sig)]
            for side,ids in [('long',long),('short',short)]:
                got=own.loc[own.side.eq(side)];need(set(got.source_security_id)==set(ids) and len(got)==n,'Independent RET ranks changed')
                for r in got.to_dict('records'):
                    equations.number(r['selection_score'],scores[r['source_security_id']],'RET score')
                    equations.number(r['intended_weight'],(.5 if side=='long' else -.5)/n,'Intended50/50weight')
            if part.outcome_month.iloc[0]!='2020-01':need(int(rb.loc[rb.signal_date.eq(sig),'selected_names_per_side'].iloc[0])==n,'Independent monthly basket size')
        pf=independent_common_preflight(selected.to_dict('records'),rb.to_dict('records'),qmeta,sessions,phase['event_registry']['events'],phase['historical_retirements'])
        need(read(check(inventory[arm+'/preflight/summary']))==pf['summary'],'Independent maximum horizon summary')
        for name in ('missing_quotes','source_quality','retired_orders','action_requirements'):
            need(sorted(record_key(r) for r in saved(arm+'/preflight/'+name).to_pylist())==sorted(record_key(r) for r in pf[name]),'Independent preflight '+name)
        for variant,fee in itertools.product(('unstopped','asset_stop_10pct'),(0,5,10,25)):
            guard();base=f'{arm}/{variant}/cost_{fee:02d}bps/'
            tables={n:saved(base+n).to_pandas() for n in ('daily','trades','holdings','failures','stops','action_events','market_trades','orders','identity_handovers')}
            monthly=saved(base+'monthly').to_pandas();summary=comp.loc[comp.arm.eq(arm)&comp.variant.eq(variant)&comp.cost_bps.eq(fee)];need(len(summary)==1,'Exact comparison scenario')
            summary=summary.iloc[0];annual=annuals.loc[annuals.arm.eq(arm)&annuals.variant.eq(variant)&annuals.cost_bps.eq(fee)]
            need(len(monthly)==47 and len(annual)==4 and summary.expected_sessions==985 and summary.outcome_months==47,'Summary common clock')
            kw=dict(events=phase['event_registry']['events'],historical_retirements=phase['historical_retirements'],cost_bps=fee,stop_fraction=None if variant=='unstopped' else .1,
              prepared_quotes=quotes,identity_handovers=phase['identity_handover_registry']['handovers'])
            if bool(summary.complete_proxy_path):
                result=equations.verify_scenario(tables,prices,selected,rb,sessions,**kw)
                equations.verify_metrics(tables['daily'],tables['trades'],monthly,summary,annual)
            else:
                result=equations.verify_failed_scenario(tables,prices,selected,sessions,rebalances=rb,**kw)
                # Arrow all-null columns can be object; make only the public
                # nullable numeric views numeric before the equation checker.
                daily=tables['daily'].copy();mon=monthly.copy()
                for name in ('nav','cash','long_value','short_value','gross_exposure','net_exposure','trading_cost','borrow_cost','turnover_one_way','long_price_pnl','short_price_pnl','long_count','short_count','net_return'):
                    if name in daily:daily[name]=pd.to_numeric(daily[name],errors='raise')
                for name in ('nav','return'):
                    if name in mon:mon[name]=pd.to_numeric(mon[name],errors='raise')
                equations.verify_failed_metrics(daily,tables['trades'],mon,summary,annual)
            results.append({'arm':arm,'variant':variant,'cost_bps':fee,'complete_proxy_path':bool(summary.complete_proxy_path),**result})
            print(json.dumps({'phase':'independent_backtest','verified':len(results),'maximum':32}),flush=True)
    return save(root/'verification/backtest.json',{'status':'PASS','plan_pin':pin(plan_path),'audit_pin':pin(out/'audit.json'),'model_proof_pin':pin(root/'verification/models.json'),
      'verification_read_plan_pin':bvplan,'worker_pin':pin(__file__),'verified_scenarios':32,'scenarios':results,'new_fits':0,'financial_2024_plus_read':False,
      'production_accounting_used_as_expected_answer':False,'source_identity_or_investability_certified':False})
