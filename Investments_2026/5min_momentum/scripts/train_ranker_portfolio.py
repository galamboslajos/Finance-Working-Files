"""Versioned ranker with shared stock selection and portfolio-based stopping.

The pairwise learner, features, historical fit pools, seeds, date splits,
100-member percentile ensembles and 12/3 WFO remain those of ranker v1.
This controller owns only its new run; it never signals or changes RET.
"""
from pathlib import Path
from datetime import datetime,timezone
from collections import deque
import argparse,gc,json,os,shutil,signal,subprocess,sys,time
import numpy as np
import pandas as pd
import duckdb,psutil
import ret_backtest as base
import train_xgb_ret_ensemble as engine
import learner_experiments as previous
from portfolio_stopping import PortfolioStopping,METRIC
from xgb_validation_stopping import train_until_validation_stops

NAME='12_ranker_portfolio100_v1'
REPORT='12_ranker_portfolio'

def utcnow():return datetime.now(timezone.utc).isoformat()

def verify_source(run):
    run=Path(run);config=previous.verify_source(run)
    manifest=json.loads((run/'source_snapshot/SOURCE_MANIFEST.json').read_text())
    for name,digest in manifest.items():
        if base.fingerprint(run/'source_snapshot'/name)!=digest:raise ValueError('Frozen source changed: '+name)
    if base.fingerprint(Path(__file__))!=config['portfolio_trainer_sha256']:raise ValueError('Wrong portfolio trainer.')
    if base.fingerprint(Path(__file__).with_name('portfolio_stopping.py'))!=config['portfolio_metric_sha256']:raise ValueError('Wrong portfolio metric.')
    return config

def read_full_fold(run,fold):
    training,testing=base.read_fold(run,fold)
    con=duckdb.connect();con.execute('SET threads=1')
    full=con.execute('SELECT * FROM read_parquet(?) WHERE trade_date>=CAST(? AS DATE) AND trade_date<CAST(? AS DATE) ORDER BY trade_date,security_id',
        [str(run/'supervised_panel.parquet'),fold['train_start'],fold['test_start']]).df();con.close()
    labeled=full.loc[full.realized_class.notna()]
    if engine.row_key_hash(labeled)!=engine.row_key_hash(training):raise ValueError('Changed historical fit pool.')
    if not full.trade_date.lt(pd.Timestamp(fold['test_start'])).all():raise ValueError('Future validation outcome.')
    return training,full,testing

def checkpoint_complete(run,fid,member):
    if not engine.checkpoint_complete(run,fid,member):return False
    prefix=engine.member_prefix(run,fid,member);meta=json.loads(prefix.with_suffix('.json').read_text())
    for suffix,key in [('.validation_curve.parquet','curve_sha256'),('.validation_daily.parquet','daily_sha256'),('.validation_dates.parquet','validation_dates_sha256')]:
        if not prefix.with_suffix(suffix).exists() or base.fingerprint(prefix.with_suffix(suffix))!=meta[key]:raise ValueError('Missing/corrupt validation history.')
    return True

def prepare(project,run,workers):
    project=Path(project).resolve();run=Path(run).resolve();private=(project/'data').resolve()
    if not run.is_relative_to(private):raise ValueError('Private output required.')
    run.mkdir(parents=True,exist_ok=True);(run/'logs').mkdir(exist_ok=True)
    old=private/'models/11_ret_ranker_parallel_v1/ranker';source=private/'models/10_xgb_ret_rich_path100_v1'
    old_config=json.loads((old/'CONFIG.json').read_text())
    feature=private/'features/10_rich_5min_path_v1/MANIFEST.json';manifest=json.loads(feature.read_text())
    if base.fingerprint(source/'supervised_panel.parquet')!=manifest['output_sha256']['supervised_panel.parquet']:raise ValueError('Input panel changed.')
    hashes={key:base.fingerprint(Path(__file__).with_name(name)) for name,key in [('learner_experiments.py','protocol_code_sha256'),
        ('ret_backtest.py','baseline_code_sha256'),('train_xgb_ret_ensemble.py','code_sha256'),('xgb_validation_stopping.py','stopping_code_sha256'),
        ('portfolio_stopping.py','portfolio_metric_sha256'),('train_ranker_portfolio.py','portfolio_trainer_sha256')]}
    config=dict(old_config,version='ranker_portfolio_stopping_v1',workers=workers,**hashes)
    config.update(stopping_metric=METRIC,maximize_metric=True,early_stopping_patience=30,
        stopping_units='Gross daily return; mean across fixed complete-outcome validation dates of 0.5*(R_N2+R_N3).',
        stopping_date_policy='Predict full validation forecast cohorts. Before fitting, fix dates where every candidate outcome is observed/finite. Other dates never alter this metric mask; retain full ledgers and unknown selected returns.',
        validation_split='Exact ranker-v1 80/20 whole-date split seeds and original labeled-pool index fingerprints; expand held-out dates to all forecast candidates for scoring.',
        selector='Shared select_portfolio_indices: descending score, ascending security ID, first/last N; disjoint equal-weight 50% legs.',
        ranker_v1_config_sha256=base.fingerprint(old/'CONFIG.json'),ranker_v1_reference=str(old),
        concurrency='Two lower-priority one-thread CPU workers; independent controller. Existing ten-worker RET process/settings untouched.',
        priority_nice_increment=10,features_and_outer_cohort_unchanged=True,parameters=previous.ranking_parameters(engine.BASE_SEED))
    if config['parameters']!=old_config['parameters']:raise ValueError('Learner changed.')
    cp=run/'CONFIG.json'
    if cp.exists() and json.loads(cp.read_text())!=config:raise ValueError('Immutable configuration differs; use a new version.')
    if not cp.exists():base.save_json(cp,config)
    for name in ('supervised_panel.parquet','wfo_calendar.parquet','label_coverage.parquet','outcome_status_counts.parquet'):
        path=run/name
        if not path.exists():path.symlink_to(source/name)
        if path.resolve()!=(source/name).resolve():raise ValueError('Input link differs.')
    snapshot=run/'source_snapshot'
    if not snapshot.exists():
        (snapshot/'scripts').mkdir(parents=True);(snapshot/'notebooks').mkdir()
        for path in Path(__file__).parent.glob('*.py'):
            if path.stem.isidentifier():shutil.copy2(path,snapshot/'scripts'/path.name)
        shutil.copy2(project/'notebooks/12_ranker_portfolio.ipynb',snapshot/'notebooks/12_ranker_portfolio.ipynb')
        (snapshot/'data').symlink_to(private,target_is_directory=True)
        files=[*sorted((snapshot/'scripts').glob('*.py')),*sorted((snapshot/'notebooks').glob('*.ipynb'))]
        base.save_json(snapshot/'SOURCE_MANIFEST.json',{str(p.relative_to(snapshot)):base.fingerprint(p) for p in files})
    for name,digest in json.loads((snapshot/'SOURCE_MANIFEST.json').read_text()).items():
        if base.fingerprint(snapshot/name)!=digest:raise ValueError('Frozen source changed.')
    if not (run/'VALIDATION_INPUTS.json').exists():
        columns=['security_id','trade_date','signal_date','outcome_status','future_oc_return','realized_class']
        slim=pd.read_parquet(run/'supervised_panel.parquet',columns=columns)
        known=np.isfinite(slim.future_oc_return.to_numpy(dtype=float))&slim.outcome_status.eq('ok').to_numpy()
        slim['known_outcome']=known
        complete=slim.groupby('trade_date').known_outcome.transform('all')
        slim['stopping_date_complete']=complete
        slim['stopping_exclusion_reason']=np.where(complete,'none','unknown_candidate_outcome_on_date')
        inventory={};summary=[]
        for fold in base.folds():
            folder=run/'folds'/fold['fold_id'];folder.mkdir(parents=True,exist_ok=True)
            frame=slim[slim.trade_date.ge(pd.Timestamp(fold['train_start']))&slim.trade_date.lt(pd.Timestamp(fold['test_start']))]
            row_path=folder/'validation_candidate_ledger.parquet'
            frame.to_parquet(row_path,index=False)
            coverage=frame.groupby('trade_date',as_index=False).agg(forecast_candidates=('security_id','size'),
                labelled_candidates=('realized_class','count'),known_outcomes=('known_outcome','sum'),stopping_date_complete=('stopping_date_complete','first'))
            path=folder/'validation_date_coverage.parquet';coverage.to_parquet(path,index=False)
            for p in (row_path,path):inventory[str(p.relative_to(run))]=base.fingerprint(p)
            summary.append(dict(fold_id=fold['fold_id'],past_candidate_rows=len(frame),past_dates=len(coverage),
                complete_outcome_dates=int(coverage.stopping_date_complete.sum()),incomplete_outcome_dates=int((~coverage.stopping_date_complete).sum())))
        base.save_json(run/'VALIDATION_INPUTS.json',dict(file_sha256=inventory,summary=summary,
            interpretation='Past-date coverage ledger only. Actual held-out dates depend on the unchanged per-member seed; no forecast candidates are removed.'))
        base.save_json(run/'PANEL.json',dict(rows=manifest['summary']['rows'],model_features=config['model_features'],
            panel_sha256=manifest['output_sha256']['supervised_panel.parquet'],feature_manifest_sha256=base.fingerprint(feature),
            source=str(source),features_labels_outcomes_unchanged=True,one_minute_data_used=False))
    return config

def batch(run,fid,members):
    run=Path(run);config=verify_source(run);features=config['model_features']
    fold=next(f for f in base.folds() if f['fold_id']==fid)
    training,full,testing=read_full_fold(run,fold)
    x=training[features].to_numpy(np.float32);xt=testing[features].to_numpy(np.float32)
    engine.validate_matrix(x,features);engine.validate_matrix(xt,features)
    labels=previous.relevance(training)
    import xgboost as xgb
    dtest=xgb.DMatrix(xt,feature_names=features,nthread=1);row_hash=engine.row_key_hash(testing)
    new_cohort=config.get('version')=='session_window_ranker100_v1'
    if new_cohort:
        from session_window_features import FEATURES
        if features!=FEATURES or config.get('historical_split_policy')!='same_seed_method_new_cohort':
            raise ValueError('Undeclared cohort/split change.')
    old=None if new_cohort else Path(config['ranker_v1_reference'])
    for member in members:
        if checkpoint_complete(run,fid,member):continue
        split_seed,booster_seed=engine.member_seeds(fid,member);ti,vi=previous.date_split(training,split_seed)
        valid_dates=training.iloc[vi].trade_date.unique();valid=full[full.trade_date.isin(valid_dates)].copy()
        if set(valid.trade_date)!=set(training.iloc[vi].trade_date):raise ValueError('Changed validation date split.')
        old_meta=None if new_cohort else json.loads(engine.member_prefix(old,fid,member).with_suffix('.json').read_text())
        for key,value in [('split_train_index_sha256',previous.digest(ti)),('split_validation_index_sha256',previous.digest(vi)),('prediction_row_key_sha256',row_hash)]:
            if old_meta is not None and old_meta[key]!=value:raise ValueError('Changed original split or forecast cohort: '+key)
        xv=valid[features].to_numpy(np.float32);engine.validate_matrix(xv,features)
        dt=xgb.DMatrix(x[ti],label=labels[ti],feature_names=features,nthread=1);dt.set_group(previous.group_sizes(training.iloc[ti]))
        # Labels deliberately absent: validation includes unknown future returns.
        dv=xgb.DMatrix(xv,feature_names=features,nthread=1);dv.set_group(previous.group_sizes(valid))
        criterion=PortfolioStopping(valid);parameters=previous.ranking_parameters(booster_seed)
        expected=dict(config['parameters'],seed=booster_seed) if new_cohort else old_meta['parameters']
        if parameters!=expected:raise ValueError('Changed ranker parameters.')
        started=time.monotonic();cpu=time.process_time();at=utcnow()
        model,history=train_until_validation_stops(parameters,dt,dv,30,metric=METRIC,maximize=True,score_function=criterion)
        best=model.best_iteration+1
        if len(criterion.history)!=best+30:raise ValueError('Patience interval differs.')
        prediction=model.predict(dtest,iteration_range=(0,best))
        prefix=engine.member_prefix(run,fid,member);prefix.parent.mkdir(parents=True,exist_ok=True)
        temp=prefix.with_suffix('.building.ubj');trimmed=model[:best];trimmed.save_model(temp);os.replace(temp,prefix.with_suffix('.ubj'))
        tmp=prefix.with_suffix('.npz.building')
        with tmp.open('wb') as stream:np.savez_compressed(stream,raw_scores=prediction,percentile_scores=previous.percentiles(prediction,testing.trade_date))
        os.replace(tmp,prefix.with_suffix('.npz'))
        curve=pd.DataFrame(criterion.history);curve['selected_best']=curve['round'].eq(best)
        curve.to_parquet(prefix.with_suffix('.validation_curve.parquet'),index=False)
        pd.DataFrame(criterion.daily_history).to_parquet(prefix.with_suffix('.validation_daily.parquet'),index=False)
        criterion.coverage.to_parquet(prefix.with_suffix('.validation_dates.parquet'),index=False)
        meta=dict(fold_id=fid,member=member,variant='ranker',split_seed=split_seed,booster_seed=booster_seed,
            split_train_index_sha256=previous.digest(ti),split_validation_index_sha256=previous.digest(vi),
            prediction_row_key_sha256=row_hash,prediction_rows=len(testing),training_pool_rows=len(training),
            inner_train_rows=len(ti),inner_validation_rows=len(vi),full_validation_candidate_rows=len(valid),
            validation_candidate_row_key_sha256=engine.row_key_hash(valid),validation_dates=len(criterion.coverage),
            fixed_stopping_dates=int(criterion.coverage.fixed_stopping_date.sum()),excluded_stopping_dates=int((~criterion.coverage.fixed_stopping_date).sum()),
            best_rounds=best,rounds_examined=len(criterion.history),validation_metric=METRIC,best_validation_score=float(model.best_score),
            parameters=parameters,training_last_outcome=str(training.trade_date.max()),outer_test_start=fold['test_start'],
            fit_started_at_utc=at,fit_completed_at_utc=utcnow(),elapsed_seconds=time.monotonic()-started,cpu_seconds=time.process_time()-cpu,
            model_sha256=base.fingerprint(prefix.with_suffix('.ubj')),prediction_sha256=base.fingerprint(prefix.with_suffix('.npz')),
            curve_sha256=base.fingerprint(prefix.with_suffix('.validation_curve.parquet')),
            daily_sha256=base.fingerprint(prefix.with_suffix('.validation_daily.parquet')),
            validation_dates_sha256=base.fingerprint(prefix.with_suffix('.validation_dates.parquet')),peak_worker_rss_gib=base.peak_rss_gib())
        base.save_json(prefix.with_suffix('.json'),meta)
        print(f'PORTFOLIO {fid} member {member+1}/100: best {best}, examined {len(criterion.history)}, dates {meta["fixed_stopping_dates"]}/{meta["validation_dates"]}',flush=True)
        del model,trimmed,dt,dv,criterion,prediction;gc.collect()

def progress(run,active,started,peak,status='running',reason=None):
    counts=engine.inventory(run);metadata=[json.loads(p.read_text()) for p in run.glob('folds/*/members/member_*.json')]
    state=dict(status=status,controller_pid=os.getpid(),completed_members=sum(counts.values()),target_members=4000,
        completed_by_fold=counts,complete_member_sets=sum(v==100 for v in counts.values()),workers_alive=len(active),
        session_started_at_utc=started,updated_at_utc=utcnow(),peak_combined_rss_gib=peak,
        stopping_metric=METRIC,RET_controller_untouched=True)
    if reason is not None:state['reason']=reason
    base.save_json(run/'RUN_STATUS.json',state);base.save_json(run/'PROGRESS.json',state)
    if len(metadata)>=5:
        workers=json.loads((run/'CONFIG.json').read_text())['workers'];mean=float(np.mean([m['elapsed_seconds'] for m in metadata]))
        base.save_json(run/'RUNTIME_ESTIMATE.json',dict(updated_at_utc=utcnow(),observed_members=len(metadata),mean_fit_seconds=mean,
            estimated_remaining_fit_seconds=(4000-len(metadata))*mean/workers,
            qualification='Observed early folds and lower-priority workers sharing CPU with RET; includes neither preparation/reporting nor host sleep. Observation only, not a cap.'))
    print(f'CORRECTED RANKER: {state["completed_members"]}/4000; workers {len(active)}; {status}',flush=True)
    return state

def run_controller(project,run,workers):
    run=Path(run).resolve();run.mkdir(parents=True,exist_ok=True)
    with engine.controller_lock(run):
        prepare(project,run,workers)
        if (run/'COMPLETE.json').exists():return
        started=utcnow();base.save_json(run/'TRAINING_STARTED.json',dict(at_utc=started,pid=os.getpid()))
        jobs=deque()
        for fold in base.folds():
            missing=[m for m in range(100) if not checkpoint_complete(run,fold['fold_id'],m)]
            for i in range(0,len(missing),5):jobs.append((fold['fold_id'],missing[i:i+5]))
        active=[];last=0.;peak=0.;previous_handler=signal.getsignal(signal.SIGTERM)
        def stop(signum,frame):raise KeyboardInterrupt('Checkpoint stop requested for corrected ranker only.')
        signal.signal(signal.SIGTERM,stop)
        try:
            while active or jobs:
                for task in list(active):
                    code=task['process'].poll()
                    if code is None:continue
                    task['log'].close();active.remove(task)
                    if code:raise RuntimeError('Corrected ranker worker failed: '+task['label'])
                while len(active)<workers and jobs:
                    fid,members=jobs.popleft();label=f'{fid}_{members[0]:03d}'
                    log=(run/'logs'/(label+'.log')).open('a')
                    command=[sys.executable,'-B',str(run/'source_snapshot/scripts/train_ranker_portfolio.py'),
                        '--mode','batch','--run-dir',str(run),'--fold-id',fid,'--members',','.join(map(str,members))]
                    process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    active.append(dict(process=process,log=log,label=label))
                rss=psutil.Process().memory_info().rss
                for task in active:
                    try:rss+=psutil.Process(task['process'].pid).memory_info().rss
                    except psutil.NoSuchProcess:pass
                peak=max(peak,rss/1024**3)
                if time.monotonic()-last>=20:progress(run,active,started,peak);last=time.monotonic()
                time.sleep(.25)
            for fold in base.folds():
                if not all(checkpoint_complete(run,fold['fold_id'],m) for m in range(100)):raise ValueError('Incomplete member or validation history.')
            progress(run,[],started,peak,status='finalizing')
            previous.finalize_variant(run)
            progress(run,[],started,peak,status='reporting')
            frozen=run/'source_snapshot'
            subprocess.run([sys.executable,'-B',str(frozen/'scripts/run_notebook.py'),'12','--project',str(frozen)],check=True)
            state=progress(run,[],started,peak,status='complete');state['completed_at_utc']=utcnow()
            base.save_json(run/'RUN_STATUS.json',state);base.save_json(run/'PROGRESS.json',state)
            base.save_json(run/'COMPLETE.json',dict(completed_at_utc=utcnow(),members=4000,quarters=40,report=REPORT,stopping_metric=METRIC))
        except BaseException as error:
            for task in active:base.terminate_owned(task['process']);task['log'].close()
            active.clear();progress(run,[],started,peak,status='stopped_resumable',reason=str(error));raise
        finally:signal.signal(signal.SIGTERM,previous_handler)

def report(project):
    project=Path(project).resolve();private=(project/'data').resolve();run=private/'models'/NAME
    if not (run/'RESULTS.json').exists() or sum(engine.inventory(run).values())!=4000:
        raise FileNotFoundError('A complete 4,000-member corrected ranker is required. Inspection cannot train or rebuild it.')
    out=private/'reports'/REPORT;out.mkdir(parents=True,exist_ok=True)
    if (out/'SUMMARY.json').exists():return out
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from analyze_volume_overnight_results import extract_importance
    paths={'Historical RET42':private/'models/10_xgb_ret_rich_path100_v1',
        'Original ranker':private/'models/11_ret_ranker_parallel_v1/ranker','Portfolio-stopped ranker':run}
    books=pd.concat([pd.read_parquet(path/'portfolio_daily.parquet').assign(model=label) for label,path in paths.items()],ignore_index=True)
    wide=books.pivot(index='trade_date',columns=['model','n_per_side'],values='gross_return');common=wide.notna().all(axis=1);dates=common[common].index
    common.to_frame('included').to_parquet(out/'common_date_mask.parquet')
    records=[]
    for (label,n),b in books.groupby(['model','n_per_side']):
        b=b.set_index('trade_date').loc[dates]
        for cost in (0,1,5,10):records.append(dict(model=label,n_per_side=int(n),cost_bps_per_side=cost,
            scope='six_way_common_observed_dates_conditional',omitted_dates=int((~common).sum()),**base.performance(b.gross_return-b.trade_notional*cost/10000)))
    table=pd.DataFrame(records);table.to_csv(out/'matched_portfolio_comparison.csv',index=False);table.to_parquet(out/'matched_portfolio_comparison.parquet',index=False)
    checked=0
    for fold in base.folds():
        for member in range(100):
            a=json.loads(engine.member_prefix(run,fold['fold_id'],member).with_suffix('.json').read_text())
            b=json.loads(engine.member_prefix(paths['Original ranker'],fold['fold_id'],member).with_suffix('.json').read_text())
            fields=['parameters','split_seed','booster_seed','split_train_index_sha256','split_validation_index_sha256','prediction_row_key_sha256']
            if not all(a[k]==b[k] for k in fields):raise ValueError('Learner, fit split or forecast cohort changed.')
            checked+=1
    base.save_json(out/'COMPARISON_AUDIT.json',dict(members_checked=checked,learner_seeds_fit_splits_forecast_rows_identical=True,
        changed='Only selection/stopping protocol: shared tie selector; full validation candidate dates; fixed complete-outcome date metric for N2/N3 gross return.'))
    colors={'Historical RET42':'#7c8f9d','Original ranker':'#bc783d','Portfolio-stopped ranker':'#267c77'}
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for ax,cost in zip(axes,(0,1)):
        for (label,n),b in books.groupby(['model','n_per_side']):
            b=b.set_index('trade_date').loc[dates]
            ax.plot(dates,(1+b.gross_return-b.trade_notional*cost/10000).cumprod(),color=colors[label],ls='-' if n==2 else '--',label=f'{label}, {n}/side')
        ax.set_title(f'Conditional growth — {cost} bp/side');ax.legend(fontsize=7);ax.grid(alpha=.15)
    fig.suptitle(f'{len(dates):,} common observed dates; {int((~common).sum())} omitted; same-opening fills optimistic')
    fig.tight_layout();fig.savefig(out/'portfolio_comparison.png',dpi=160);plt.close(fig)
    dest=out/'feature_diagnostics';dest.mkdir(exist_ok=True)
    importance,_=extract_importance(run,dest,allow_constant_models=True)
    fig,ax=plt.subplots(figsize=(10,12));f=importance.sort_values('mean_gain_share')
    ax.barh(f.label,f.mean_gain_share*100,color='#267c77');ax.set_xlabel('Normalized training pairwise gain (%)')
    ax.set_title('Corrected ranker — training importance, not return explained variance');fig.tight_layout();fig.savefig(out/'feature_importance.png',dpi=160);plt.close(fig)
    metas=[json.loads(p.read_text()) for p in run.glob('folds/*/members/member_*.json')]
    pd.DataFrame([{k:m[k] for k in ('fold_id','member','best_rounds','rounds_examined','validation_dates','fixed_stopping_dates','excluded_stopping_dates','best_validation_score')} for m in metas]).to_parquet(out/'member_stopping_diagnostics.parquet',index=False)
    base.save_json(out/'SUMMARY.json',dict(status='complete',models=4000,comparison=records,common_observed_dates=len(dates),
        omitted_dates=int((~common).sum()),median_selected_rounds=float(np.median([m['best_rounds'] for m in metas])),
        mean_fixed_validation_dates=float(np.mean([m['fixed_stopping_dates'] for m in metas])),
        stopping_metric=METRIC,return_EV=None,
        qualifications=['Gross return stopping uses the fixed complete-outcome subset of full-candidate historical validation dates; coverage exclusions retained.',
            'Stock selection is identical in validation and portfolio accounting. All 4000 learner/split/forecast-key comparisons pass.',
            'Original ranker has the audited metric/tie mismatch. Historical RET has row validation; corrected ranker has whole-date validation.',
            'Ranking scores are not calibrated returns, so return EV/R2 is undefined.',
            'Conditional metrics omit unknown selected exits. Same-opening information/fills and reused development history remain optimistic.']))
    return out

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['run','batch','prepare'],required=True)
    parser.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--run-dir',type=Path,required=True);parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--fold-id');parser.add_argument('--members')
    args=parser.parse_args()
    if args.mode=='run':run_controller(args.project,args.run_dir,args.workers)
    elif args.mode=='prepare':prepare(args.project,args.run_dir,args.workers)
    else:batch(args.run_dir,args.fold_id,[int(x) for x in args.members.split(',')])
