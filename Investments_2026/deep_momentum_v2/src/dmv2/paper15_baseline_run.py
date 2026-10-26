"""Explicit one-fit Paper15 annual workers and fixed RET daily scenarios."""
from pathlib import Path
import gc,itertools,json,math,signal,time
from .paper15_baseline import FEATURES,PROBS,KEYS,need,select_ret
from .paper15_baseline_data import validate,check,read,save,pin,table_save


def arrays(plan,years,year):
    """Date/state projection before training/prediction financial conversion."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    root=Path(plan['output_directory']);receipt=read(root/'prepared/receipt.json')
    need(receipt['plan_pin']==pin(plan['_path']),'Prepared inputs from wrong plan')
    t=pq.read_table(check(receipt['artifacts'][f'panel{years}']))
    meta=t.select([*KEYS,'outcome_month','outcome_end','label_available_at','prediction_eligible','target_available']).to_pandas()
    need(meta.outcome_end.max()<'2024-01-01','Protected targets')
    cutoff=f'{year-1}-12-31';train=(meta.outcome_month.ge(f'{year-years}-01')&meta.outcome_month.le(f'{year-1}-12')&meta.label_available_at.le(cutoff)&meta.prediction_eligible&meta.target_available).to_numpy(dtype=bool)
    test=meta.outcome_month.str.startswith(str(year)).to_numpy(dtype=bool);qualified=test&meta.prediction_eligible.to_numpy(dtype=bool)
    X=t.select(list(FEATURES)).filter(pa.array(train)).to_pandas().to_numpy(dtype=np.float32)
    y=t.select(['target_class']).filter(pa.array(train)).to_pandas().target_class.to_numpy(dtype=np.int64)
    XP=t.select(list(FEATURES)).filter(pa.array(qualified)).to_pandas().to_numpy(dtype=np.float32)
    # Audit and prediction metadata have no realized outcome values.
    cols=[*KEYS,'intrinio_security_id','identity_eligible','identity_status','identity_review_required','resolution_status','prediction_eligible','prediction_status','outcome_month','outcome_start','outcome_end']
    predictions=t.select(cols).filter(pa.array(test)).to_pandas().reset_index(drop=True)
    return t,meta,train,test,qualified,X,y,XP,predictions


def fit(plan_path,years,year,kind,guard=lambda:None):
    import joblib,numpy as np,pyarrow as pa
    from .pilot_models import fit_predict_pilot_arm
    from .walkforward_metrics import evaluate_probabilities
    p=validate(plan_path);p['_path']=str(Path(plan_path).resolve());cfg=read(check(p['protocol_pin']))
    root=Path(p['output_directory']);verified=read(root/'verification/data.json')
    need(verified['status']=='PASS' and verified['plan_pin']==pin(plan_path) and verified['prepared_receipt_pin']==pin(root/'prepared/receipt.json'),'Independent data check before fitting')
    out=root/f'folds/{years}y/{year}/{kind}';need(not out.exists(),'Preserve consumed estimator attempt');out.mkdir(parents=True)
    save(out/'FIT_STARTED.json',{'plan_pin':pin(plan_path),'years':years,'year':year,'kind':kind,'fit_attempt_consumed':True})
    t,meta,train,test,qualified,X,y,XP,forecast=arrays(p,years,year)
    need(X.shape[1]==XP.shape[1]==15 and np.isfinite(X).all() and np.isfinite(XP).all(),'Exactly15finitepaperinputs;noimputation')
    guard();start=time.monotonic();before=signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Fit time ceiling consumed')))
    signal.alarm(cfg['budgets']['per_fit_seconds'])
    try:result=fit_predict_pilot_arm(kind,X,y,XP,parameters=cfg['parameters'][kind],package_versions=cfg['packages'])
    finally:signal.alarm(0);signal.signal(signal.SIGALRM,before)
    guard();model_path=out/'model.joblib';joblib.dump(result.estimator,model_path,compress=3)
    eligible=forecast.prediction_eligible.to_numpy(dtype=bool)
    for j,n in enumerate(PROBS):forecast[n]=np.nan;forecast.loc[eligible,n]=result.probabilities[:,j]
    artifact=table_save(out/'predictions.parquet',forecast)
    # No future label is supplied to fitting/inference; evaluation follows saved predictions.
    labels=t.select(['target_class']).filter(pa.array(qualified)).to_pandas().target_class
    md=forecast.loc[eligible].reset_index(drop=True);labels=labels.reset_index(drop=True)
    summaries=[];metrics={};prior=np.bincount(y,minlength=11)[1:]/len(y)
    for name,prob in [('model',result.probabilities),('uniform',np.full((len(md),10),.1)),('train_prior',np.tile(prior,(len(md),1)))]:
        evaluation=evaluate_probabilities(prob,labels,class_ids=list(range(1,11)),row_metadata=md,first_month=f'{year}-01',last_month=f'{year}-12')
        summary={**evaluation.summary,'history_years':years,'year':year,'estimator':kind,'reference':name};summaries.append(summary)
        for key,frame in evaluation.tables.items():metrics[name+'/'+key]=table_save(out/'metrics'/name/(key+'.parquet'),frame)
    summary_pin=save(out/'metrics/summary.json',summaries)
    return save(out/'FIT_COMPLETE.json',{'status':'FIT_AND_PREDICT_COMPLETE','plan_pin':pin(plan_path),'years':years,'year':year,'kind':kind,
      'model':pin(model_path),'predictions':artifact,'metrics':metrics,'summary':summary_pin,'fit_metadata':result.metadata,
      'fit_seconds':time.monotonic()-start,'train_outcome_start':f'{year-years}-01','train_outcome_end':f'{year-1}-12',
      'actual_label_cutoff':f'{year-1}-12-31','no_target_or_future_quote_prediction_filter':True,'new_tuning':False})


def backtest(plan_path,guard=lambda:None):
    import pandas as pd,pyarrow.parquet as pq
    from .target_comparison_backtest import _load_prices,QUOTE_META
    from .walkforward_preflight import preflight_selected_horizons
    from .walkforward_accounting_v2 import run_walkforward_accounting_v2
    from .backtest_accounting import DailyAccountingPolicy
    from .daily_backtest import summarize_daily
    from .walkforward_backtest import yearly_breakdown
    p=validate(plan_path);cfg=read(check(p['protocol_pin']));root=Path(p['output_directory'])
    proof=read(root/'verification/models.json');need(proof['status']=='PASS' and proof['verified_fits']==16,'Independent16-model proof required')
    out=root/'backtest';need(not out.exists(),'Preserve existing backtest attempt');out.mkdir(parents=True);save(out/'STARTED.json',{'plan_pin':pin(plan_path),'new_fits':0})
    prep=read(root/'prepared/receipt.json');bt=p['execution_source'];prices,overlay,_=_load_prices(bt,guard);phase=p['execution_phase'];artifacts={'quote_overlay':save(out/'quote_overlay.json',overlay)}
    comparison=[];annuals=[];allselections=[]
    for years,kind in itertools.product((5,10),('rf','xgb')):
        arm=f'{kind}_paper15_{years}y';parts=[]
        for year in (2020,2021,2022,2023):
            done=read(root/f'folds/{years}y/{year}/{kind}/FIT_COMPLETE.json');need(done['predictions']==proof['forecasts'][f'{years}/{year}/{kind}'],'Unverified forecast')
            parts.append(pq.read_table(check(done['predictions'])).to_pandas())
        pred=pd.concat(parts,ignore_index=True);payoffs=read(check(prep['artifacts'][f'payoffs{years}']))
        windows=pred[['signal_date','outcome_month','outcome_start','outcome_end']].drop_duplicates().sort_values('signal_date').to_dict('records');need(len(windows)==48,'48forecastmonths')
        ledger,selected,rb,monthly=select_ret(pred,payoffs,windows,excluded=cfg['backtest']['execution_restrictions'])
        selected=selected.loc[selected.execution_date.ge('2020-02-03')].copy();rb=rb.loc[rb.execution_date.ge('2020-02-03')].copy()
        for name,frame in [('original_ledger',ledger),('selected',selected),('rebalances',rb),('monthly',monthly)]:artifacts[arm+'/selection/'+name]=table_save(out/arm/'selection'/(name+'.parquet'),frame)
        allselections.append((arm,selected,rb))
        preflight=preflight_selected_horizons(selected,rb,prices[list(QUOTE_META)],bt['sessions'],events=phase['event_registry']['events'],historical_retirements=phase['historical_retirements'])
        artifacts[arm+'/preflight/summary']=save(out/arm/'preflight/summary.json',preflight['summary'])
        for n in ('missing_quotes','source_quality','retired_orders','action_requirements'):artifacts[arm+'/preflight/'+n]=table_save(out/arm/'preflight'/(n+'.parquet'),preflight[n])
        for variant,fee in itertools.product(cfg['backtest']['variants'],cfg['backtest']['cost_bps']):
            guard();v=run_walkforward_accounting_v2(selected,prices,bt['sessions'],rebalances=rb,terminal_date='2023-12-29',
              events=phase['event_registry']['events'],historical_retirements=phase['historical_retirements'],identity_handovers=phase['identity_handover_registry']['handovers'],
              policy=DailyAccountingPolicy(cost_bps=fee,borrow_annual_bps=0,required_names_per_side=None,asset_stop_fraction=cfg['backtest']['variants'][variant]))
            row,mon=summarize_daily(v,arm=arm,basket='ret',variant=variant,cost_bps=fee,expected_sessions=985);row.update(outcome_start='2020-02-03',outcome_end='2023-12-29',outcome_months=47)
            annual=yearly_breakdown(v,arm=arm,basket='ret',variant=variant,cost_bps=fee);base=f'{arm}/{variant}/cost_{fee:02d}bps/'
            for n,frame in [*v.tables.items(),('monthly',mon),('annual',annual)]:artifacts[base+n]=table_save(out/(base+n+'.parquet'),frame)
            comparison.append(row);annuals.append(annual)
            save(out/'checkpoints'/(base+'complete.json'),{'status':'COMPLETE_PENDING_VERIFICATION','artifacts':{k:v for k,v in artifacts.items() if k.startswith(base)}})
            print(json.dumps({'phase':'backtest','completed':len(comparison),'maximum':32}),flush=True)
    artifacts['comparison']=table_save(out/'comparison.parquet',pd.DataFrame(comparison));artifacts['annual']=table_save(out/'annual.parquet',pd.concat(annuals,ignore_index=True))
    return save(out/'audit.json',{'status':'COMPLETE_PENDING_INDEPENDENT_CHECK','plan_pin':pin(plan_path),'model_proof_pin':pin(root/'verification/models.json'),
       'artifacts':artifacts,'scenarios':32,'sessions':985,'months':47,'complete_paths':sum(bool(r['complete_proxy_path']) for r in comparison),
       'withheld_paths':sum(not bool(r['complete_proxy_path']) for r in comparison),'new_fits':0,'financial_2024_plus_read':False})
