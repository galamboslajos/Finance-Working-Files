"""Past-only RET tuning and a fixed pairwise ranker, sharing one CPU scheduler.

This is a new declared protocol. RET member row splits remain identical to
completed path42. HPO and ranker validation use whole dates. Outer rows and
accounting are unchanged. Ranker scores are never called return forecasts.
"""
from __future__ import annotations
from collections import deque
from datetime import datetime,timezone
import gc,hashlib,itertools,json,os,shutil,signal,subprocess,sys,time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

import ret_backtest as base
import train_xgb_ret_ensemble as engine
from build_momentum_features import session_calendar
from rich_5min_features import MODEL_FEATURES
from xgb_validation_stopping import train_until_validation_stops

SUITE_NAME="11_ret_ranker_parallel_v1"
VARIANTS=("ret_tuned","ranker")
TUNING_SPLITS=3


def utcnow():return datetime.now(timezone.utc).isoformat()


def digest(values):
    return hashlib.sha256(np.asarray(values).tobytes()).hexdigest()


def config_grid():
    grid=[dict(max_depth=d,min_child_weight=w,reg_lambda=r,colsample_bytree=c)
        for d,w,r,c in itertools.product((2,3,4),(10,20),(1.,10.),(.8,1.))]
    baseline={k:base.PARAMETERS[k] for k in grid[0]}
    grid.sort(key=lambda p:(p!=baseline,p['max_depth'],p['min_child_weight'],p['reg_lambda'],p['colsample_bytree']))
    assert len(grid)==24 and grid[0]==baseline
    return grid


def date_split(frame,seed):
    """Fresh 80/20 exchange-date split; complete queries stay on one side."""
    dates=np.sort(frame.trade_date.unique())
    if len(dates)<20:raise ValueError("Insufficient past dates.")
    permutation=np.random.default_rng(seed).permutation(len(dates))
    selected=dates[permutation[:int(.8*len(dates))]]
    train=np.flatnonzero(frame.trade_date.isin(selected).to_numpy())
    valid=np.flatnonzero(~frame.trade_date.isin(selected).to_numpy())
    assert not set(frame.trade_date.iloc[train])&set(frame.trade_date.iloc[valid])
    return train,valid


def group_sizes(frame):
    if not frame.trade_date.is_monotonic_increasing:raise ValueError("Sort whole query groups by trading date.")
    return frame.groupby('trade_date',sort=False).size().to_numpy(dtype=np.uint32)


def relevance(frame):
    """Dense ascending return order; exact return ties retain the same grade."""
    if not np.isfinite(frame.future_oc_return).all():raise ValueError("No unknown outcome can be a training label.")
    return (frame.groupby('trade_date').future_oc_return.rank(method='dense')-1).to_numpy(np.float32)


def percentiles(scores,dates):
    scores=np.asarray(scores,float)
    if scores.ndim!=1 or len(scores)!=len(dates) or not np.isfinite(scores).all():raise ValueError("Invalid ranking margins.")
    table=pd.DataFrame({'score':scores,'date':np.asarray(dates)})
    rank=table.groupby('date').score.rank(method='average')
    n=table.groupby('date').score.transform('size')
    return np.where(n>1,(rank-1)/np.maximum(n-1,1),.5).astype(np.float32)


class SymmetricNDCG:
    """Equal-date mean of top-three and bottom-three linear-gain NDCG.

    Labels are within-day ordinal grades. This measures ordering, not P&L.
    An all-tied outcome query has no useful ranking information and scores zero.
    """
    def __init__(self,labels,sizes,k=3):
        self.labels=np.asarray(labels,float);self.k=k
        if sum(sizes)!=len(labels) or not np.isfinite(labels).all():raise ValueError("Bad query labels.")
        self.groups=[];start=0
        for n in sizes:
            n=int(n);g=self.labels[start:start+n]
            weights=1/np.log2(np.arange(min(k,n))+2)
            reverse=g.max()-g
            ideal=np.dot(np.sort(g)[::-1][:k],weights)
            short_ideal=np.dot(np.sort(reverse)[::-1][:k],weights)
            self.groups.append((start,start+n,g,reverse,weights,ideal,short_ideal))
            start+=n

    def __call__(self,scores):
        scores=np.asarray(scores,float)
        if scores.shape!=self.labels.shape or not np.isfinite(scores).all():raise ValueError("Invalid ranking scores.")
        results=[]
        for a,b,g,reverse,w,ideal,short_ideal in self.groups:
            if ideal==0 or short_ideal==0:results.append(0.);continue
            long=np.argsort(-scores[a:b],kind='stable')[:self.k]
            short=np.argsort(scores[a:b],kind='stable')[:self.k]
            results.append(.5*(np.dot(g[long],w)/ideal+np.dot(reverse[short],w)/short_ideal))
        return float(np.mean(results))


def ranking_parameters(seed):
    params={k:v for k,v in base.PARAMETERS.items() if k not in ('objective','num_class','eval_metric')}
    params.update(objective='rank:pairwise',seed=seed,nthread=1,
        disable_default_eval_metric=True,ndcg_exp_gain=False,
        lambdarank_pair_method='mean',lambdarank_num_pair_per_sample=8,
        lambdarank_normalization=True,lambdarank_score_normalization=True)
    return params


def read_tuning(run,fold_id):
    run=Path(run);folder=run/'folds'/fold_id
    selected=json.loads((folder/'TUNING.json').read_text())
    config=json.loads((run/'CONFIG.json').read_text())
    trials=json.loads((folder/'TUNING_PROGRESS.json').read_text())
    if (selected['fold_id']!=fold_id or selected['trials_sha256']!=base.fingerprint(folder/'TUNING_PROGRESS.json')
        or trials['config_sha256']!=base.fingerprint(run/'CONFIG.json')):raise ValueError('Tuning lineage mismatch.')
    unique={(t['candidate'],t['split']):t for t in trials['trials']}
    grid=config['tuning_grid']
    if len(unique)!=len(trials['trials']) or set(unique)!={(i,j) for i in range(len(grid)) for j in range(3)}:
        raise ValueError('Incomplete tuning experiment.')
    scores=[]
    for i,values in enumerate(grid):
        for j in range(3):
            t=unique[(i,j)]
            if t['parameters']!=values or pd.Timestamp(t['training_last_outcome'])>=pd.Timestamp(t['outer_test_start']):raise ValueError('Invalid/future tuning input.')
        scores.append(float(np.mean([unique[(i,j)]['best_validation_logloss'] for j in range(3)])))
    best=int(np.argmin(scores))
    if not np.isfinite(scores).all() or selected['selected_candidate']!=best or selected['selected_parameters']!=grid[best]:
        raise ValueError('Tuning selection differs from the declared loss criterion.')
    return selected


def verify_source(run):
    run=Path(run);config=json.loads((run/'CONFIG.json').read_text())
    for file,key in [('learner_experiments.py','protocol_code_sha256'),('ret_backtest.py','baseline_code_sha256'),
        ('train_xgb_ret_ensemble.py','code_sha256'),('xgb_validation_stopping.py','stopping_code_sha256')]:
        if base.fingerprint(Path(__file__).with_name(file))!=config[key]:raise ValueError('Pinned source differs: '+file)
    return config


def tuning_worker(run,fold_id):
    run=Path(run);config=verify_source(run)
    if config['variant']!='ret_tuned':raise ValueError('Only RET is tuned in this experiment.')
    fold=next(f for f in base.folds() if f['fold_id']==fold_id)
    training,_=base.read_fold(run,fold)
    features=config['model_features'];x=training[features].to_numpy(np.float32)
    engine.validate_matrix(x,features);y=training.realized_class.to_numpy(np.float32)-1
    destination=run/'folds'/fold_id;destination.mkdir(parents=True,exist_ok=True)
    state=destination/'TUNING_PROGRESS.json';finished=destination/'TUNING.json'
    if finished.exists():read_tuning(run,fold_id);return
    saved=json.loads(state.read_text()) if state.exists() else dict(config_sha256=base.fingerprint(run/'CONFIG.json'),trials=[])
    if saved['config_sha256']!=base.fingerprint(run/'CONFIG.json'):raise ValueError('Tuning checkpoint config differs.')
    done={(t['candidate'],t['split']):t for t in saved['trials']}
    grid=config['tuning_grid']
    import xgboost as xgb
    for candidate,values in enumerate(grid):
        for split in range(TUNING_SPLITS):
            if (candidate,split) in done:continue
            seed,booster_seed=engine.member_seeds(fold_id,split)
            ti,vi=date_split(training,seed)
            dtrain=xgb.DMatrix(x[ti],label=y[ti],feature_names=features,nthread=1)
            dvalid=xgb.DMatrix(x[vi],label=y[vi],feature_names=features,nthread=1)
            params=dict(base.PARAMETERS,**values);params.update(seed=booster_seed,nthread=1)
            started=time.monotonic();cpu=time.process_time();at=utcnow()
            model,history=train_until_validation_stops(params,dtrain,dvalid,patience=30)
            trial=dict(candidate=candidate,split=split,parameters=values,split_seed=seed,booster_seed=booster_seed,
                split_train_index_sha256=digest(ti),split_validation_index_sha256=digest(vi),
                train_dates=int(training.iloc[ti].trade_date.nunique()),validation_dates=int(training.iloc[vi].trade_date.nunique()),
                training_last_outcome=str(training.trade_date.max()),outer_test_start=fold['test_start'],
                best_validation_logloss=float(model.best_score),best_rounds=int(model.best_iteration+1),
                rounds_examined=len(history['random20']['mlogloss']),elapsed_seconds=time.monotonic()-started,
                cpu_seconds=time.process_time()-cpu,started_at_utc=at,completed_at_utc=utcnow())
            saved['trials'].append(trial);done[(candidate,split)]=trial
            base.save_json(state,saved)
            print(f'TUNING {fold_id}: {len(done)}/{len(grid)*TUNING_SPLITS} complete fits',flush=True)
            del model,dtrain,dvalid;gc.collect()
    scores=[float(np.mean([done[(i,j)]['best_validation_logloss'] for j in range(TUNING_SPLITS)])) for i in range(len(grid))]
    selected=int(np.argmin(scores))
    base.save_json(finished,dict(fold_id=fold_id,selected_candidate=selected,selected_parameters=grid[selected],
        candidate_mean_logloss=scores,selection='Lowest mean best historical validation mlogloss over three fixed whole-date splits; deterministic first candidate breaks exact ties.',
        trial_fits=len(done),trials_sha256=base.fingerprint(state),training_rows=len(training),
        training_last_outcome=str(training.trade_date.max()),outer_test_start=fold['test_start'],completed_at_utc=utcnow()))


def batch_worker(run,fold_id,members):
    run=Path(run);config=verify_source(run);variant=config['variant']
    fold=next(f for f in base.folds() if f['fold_id']==fold_id)
    training,testing=base.read_fold(run,fold)
    features=config['model_features'];x=training[features].to_numpy(np.float32);test_x=testing[features].to_numpy(np.float32)
    engine.validate_matrix(x,features);engine.validate_matrix(test_x,features)
    y=relevance(training) if variant=='ranker' else training.realized_class.to_numpy(np.float32)-1
    import xgboost as xgb
    dtest=xgb.DMatrix(test_x,feature_names=features,nthread=1)
    key_hash=engine.row_key_hash(testing)
    if variant=='ret_tuned':
        tuning=read_tuning(run,fold_id)
        selected=tuning['selected_parameters']
    for member in members:
        if engine.checkpoint_complete(run,fold_id,member):continue
        split_seed,booster_seed=engine.member_seeds(fold_id,member)
        ti,vi=(date_split(training,split_seed) if variant=='ranker' else engine.split_indices(len(training),split_seed))
        dtrain=xgb.DMatrix(x[ti],label=y[ti],feature_names=features,nthread=1)
        dvalid=xgb.DMatrix(x[vi],label=y[vi],feature_names=features,nthread=1)
        start=time.monotonic();cpu=time.process_time();at=utcnow()
        if variant=='ranker':
            train_groups=group_sizes(training.iloc[ti]);valid_groups=group_sizes(training.iloc[vi])
            dtrain.set_group(train_groups);dvalid.set_group(valid_groups)
            params=ranking_parameters(booster_seed);metric='symmetric_ndcg3'
            model,history=train_until_validation_stops(params,dtrain,dvalid,patience=30,
                metric=metric,maximize=True,score_function=SymmetricNDCG(y[vi],valid_groups))
        else:
            params=dict(base.PARAMETERS,**selected);params.update(seed=booster_seed,nthread=1);metric='mlogloss'
            model,history=train_until_validation_stops(params,dtrain,dvalid,patience=30)
        best=int(model.best_iteration+1);prediction=model.predict(dtest,iteration_range=(0,best))
        prefix=engine.member_prefix(run,fold_id,member);prefix.parent.mkdir(parents=True,exist_ok=True)
        temp=prefix.with_suffix('.building.ubj');trimmed=model[:best];trimmed.save_model(temp);os.replace(temp,prefix.with_suffix('.ubj'))
        temporary=prefix.with_suffix('.npz.building')
        with temporary.open('wb') as stream:
            if variant=='ranker':
                if prediction.shape!=(len(testing),):raise ValueError('Wrong ranking score shape.')
                np.savez_compressed(stream,raw_scores=prediction,percentile_scores=percentiles(prediction,testing.trade_date))
            else:
                if prediction.shape!=(len(testing),10) or not np.isfinite(prediction).all():raise ValueError('Wrong probability matrix.')
                np.savez_compressed(stream,probabilities=prediction)
        os.replace(temporary,prefix.with_suffix('.npz'))
        info=dict(fold_id=fold_id,member=member,variant=variant,split_seed=split_seed,booster_seed=booster_seed,
            split_train_index_sha256=digest(ti),split_validation_index_sha256=digest(vi),
            prediction_row_key_sha256=key_hash,prediction_rows=len(testing),training_pool_rows=len(training),
            inner_train_rows=len(ti),inner_validation_rows=len(vi),best_rounds=best,
            rounds_examined=len(history['random20'][metric]),validation_metric=metric,best_validation_score=float(model.best_score),
            parameters=params,training_last_outcome=str(training.trade_date.max()),outer_test_start=fold['test_start'],
            fit_started_at_utc=at,fit_completed_at_utc=utcnow(),elapsed_seconds=time.monotonic()-start,
            cpu_seconds=time.process_time()-cpu,model_sha256=base.fingerprint(prefix.with_suffix('.ubj')),
            prediction_sha256=base.fingerprint(prefix.with_suffix('.npz')),peak_worker_rss_gib=base.peak_rss_gib())
        base.save_json(prefix.with_suffix('.json'),info)
        print(f'{variant} {fold_id} member {member+1}/100: {best} rounds',flush=True)
        del model,trimmed,dtrain,dvalid,prediction;gc.collect()


def finalize_variant(run):
    run=Path(run);config=verify_source(run);variant=config['variant']
    for fold in base.folds():
        destination=run/'folds'/fold['fold_id']
        if (destination/'FOLD.json').exists():continue
        metadata=[];paths=[]
        for member in range(100):
            if not engine.checkpoint_complete(run,fold['fold_id'],member):raise ValueError('Exactly 100 complete members required.')
            prefix=engine.member_prefix(run,fold['fold_id'],member)
            metadata.append(json.loads(prefix.with_suffix('.json').read_text()));paths.append(prefix.with_suffix('.npz'))
        training,testing=base.read_fold(run,fold);key=engine.row_key_hash(testing)
        if any(m['prediction_row_key_sha256']!=key for m in metadata):raise ValueError('Member forecast rows differ.')
        cols=['security_id','ticker','signal_date','signal_cutoff_utc','trade_date','outcome_status','future_oc_return','realized_class']
        predictions=testing[cols].copy()
        extra={}
        if variant=='ret_tuned':
            probabilities=engine.average_probabilities(paths)
            mu,daily,complete=base.class_payoffs(training,expected_dates=session_calendar(fold['train_start'],
                (pd.Timestamp(fold['test_start'])-pd.Timedelta(days=1)).date().isoformat()).signal_date)
            for k in range(10):predictions[f'p_class_{k+1}']=probabilities[:,k]
            predictions['ret_score']=probabilities@mu
            daily.assign(used_for_calibration=complete).to_parquet(destination/'daily_class_payoffs.parquet')
            extra.update(payoff_vector_class_1_to_10=mu.tolist(),payoff_complete_days=int(complete.sum()),payoff_incomplete_days=int((~complete).sum()))
        else:
            score=np.zeros(len(testing),float)
            for path in paths:
                with np.load(path,allow_pickle=False) as saved:
                    value=saved['percentile_scores']
                    if value.shape!=score.shape or not np.isfinite(value).all() or (value<0).any() or (value>1).any():raise ValueError('Invalid member percentiles.')
                    score+=value
            predictions['ranking_score']=score/100
            predictions['ret_score']=predictions.ranking_score
            extra.update(score_units='Mean within-day percentile rank; not expected returns.',payoff_vector_class_1_to_10=None)
        predictions['fold_id']=fold['fold_id'];predictions['ensemble_members']=100
        predictions.to_parquet(destination/'predictions.parquet',index=False)
        pd.DataFrame(metadata).to_parquet(destination/'members_summary.parquet',index=False)
        base.save_json(destination/'FOLD.json',dict(**fold,ensemble_members=100,training_rows=len(training),
            prediction_rows=len(testing),prediction_row_key_sha256=key,
            mean_best_rounds=float(np.mean([m['best_rounds'] for m in metadata])),**extra))
        print('FINALIZED',variant,fold['fold_id'],flush=True)
    base.summarize(run,score_type='ranking' if variant=='ranker' else 'ret')
    results=json.loads((run/'RESULTS.json').read_text())
    results.update(ensemble_members_per_quarter=100,total_member_models=4000,variant=variant,
        protocol=config['inner_split'],features_unchanged=True)
    results['limitations']=[x for x in results['limitations'] if not x.startswith('Fixed-parameter')]
    results['limitations'] += [
        'Same-opening gap information arrives with the stored bar after the assumed fill. No one-minute data.',
        'RET final members retain original row splits; ranker uses whole-date splits. Learner and validation effects are not isolated.',
        'Previously viewed development history; no fresh final holdout.']
    if variant=='ranker':results['limitations'].append('Uncalibrated rank scores are not return forecasts; return EV/R2 and classification probabilities are not defined.')
    else:results['limitations'].append('Hyperparameters selected in each past 12-month pool using three whole-date splits; those scores are selection statistics, not unbiased validation performance.')
    base.save_json(run/'RESULTS.json',results)
    status=dict(status='complete',completed_members=4000,target_members=4000,complete_member_sets=40,
        workers_alive=0,completed_at_utc=utcnow(),variant=variant)
    base.save_json(run/'RUN_STATUS.json',status);base.save_json(run/'PROGRESS.json',status)


def prepare_suite(project,suite,workers):
    project=Path(project).resolve();suite=Path(suite).resolve();private=(project/'data').resolve()
    if not suite.is_relative_to(private):raise ValueError('Private output required.')
    source=private/'models/10_xgb_ret_rich_path100_v1'
    feature=private/'features/10_rich_5min_path_v1'
    manifest=json.loads((feature/'MANIFEST.json').read_text())
    if base.fingerprint(source/'supervised_panel.parquet')!=manifest['output_sha256']['supervised_panel.parquet']:raise ValueError('Frozen input panel changed.')
    source_hashes={key:base.fingerprint(Path(__file__).with_name(file)) for file,key in [
        ('learner_experiments.py','protocol_code_sha256'),('ret_backtest.py','baseline_code_sha256'),
        ('train_xgb_ret_ensemble.py','code_sha256'),('xgb_validation_stopping.py','stopping_code_sha256')]}
    common=dict(version='parallel_ret_tuning_ranker_v1',members_per_quarter=100,total_quarters=40,total_member_models=4000,
        train_months=12,test_months=3,model_features=MODEL_FEATURES,n_per_side=[2,3],gross_exposure=1.,
        workers=workers,threads_per_worker=1,max_rounds=None,memory_limit_gib=None,
        minimum_available_memory_gib=0.,hard_stop_utc=None,not_before_utc=None,early_stopping_patience=30,
        feature_manifest_sha256=base.fingerprint(feature/'MANIFEST.json'),panel_sha256=manifest['output_sha256']['supervised_panel.parquet'],
        base_seed=engine.BASE_SEED,network_features_added=False,one_minute_data_used=False,**source_hashes)
    old_fold_seconds=[json.loads(p.read_text())['total_member_seconds'] for p in source.glob('folds/*/FOLD.json')]
    if len(old_fold_seconds)!=40:raise ValueError('Expected complete historical RET timing reference.')
    suite_config=dict(**common,variants=list(VARIANTS),tuning_grid=config_grid(),tuning_splits=3,
        tuning_fit_target=24*3*40,final_member_target=8000,parameters=dict(base.PARAMETERS),
        historical_ret_mean_member_seconds=float(sum(old_fold_seconds)/4000),
        concurrency='One shared ten-core CPU pool, round-robin between RET and ranker; all free slots reusable. No fixed RAM, round or time ceilings.')
    path=suite/'CONFIG.json'
    if path.exists() and json.loads(path.read_text())!=suite_config:raise FileExistsError('Different suite configuration; create a new version.')
    if not path.exists():base.save_json(path,suite_config)
    snapshot=suite/'source_snapshot'
    if not snapshot.exists():
        (snapshot/'scripts').mkdir(parents=True)
        for p in Path(__file__).parent.glob('*.py'):
            if p.stem.isidentifier():shutil.copyfile(p,snapshot/'scripts'/p.name)
        (snapshot/'notebooks').mkdir()
        shutil.copyfile(project/'notebooks/11_ret_tuning_and_ranker.ipynb',snapshot/'notebooks/11_ret_tuning_and_ranker.ipynb')
        (snapshot/'data').symlink_to(private,target_is_directory=True)
        files=[*sorted((snapshot/'scripts').glob('*.py')),*sorted((snapshot/'notebooks').glob('*.ipynb'))]
        base.save_json(snapshot/'SOURCE_MANIFEST.json',{str(p.relative_to(snapshot)):base.fingerprint(p) for p in files})
    for relative,digest in json.loads((snapshot/'SOURCE_MANIFEST.json').read_text()).items():
        if base.fingerprint(snapshot/relative)!=digest:raise ValueError('A frozen source file changed: '+relative)
    for variant in VARIANTS:
        run=suite/variant;run.mkdir(exist_ok=True);(run/'logs').mkdir(exist_ok=True)
        config=dict(common,variant=variant,parameters=ranking_parameters(engine.BASE_SEED) if variant=='ranker' else dict(base.PARAMETERS),
            inner_split=('Fresh independent random 80/20 whole trading dates; complete query groups.' if variant=='ranker'
                else 'Unchanged historical path42 independent random 80/20 stock-row member splits.'),
            tuning_grid=config_grid() if variant=='ret_tuned' else [],tuning_splits=3 if variant=='ret_tuned' else 0,
            stopping_metric='symmetric_ndcg3' if variant=='ranker' else 'mlogloss',maximize_metric=variant=='ranker',
            averaging=('Mean of 100 within-date percentile ranks; raw margins are not comparable return units.' if variant=='ranker'
                else 'Mean 100 class probability vectors, then frozen full-training RET payoffs.'),
            target=('Dense within-day future open-to-close return order; exact ties retained.' if variant=='ranker' else 'Unchanged ten future-return classes.'),
            legacy_score_alias=('ret_score equals ranking_score for shared portfolio sorting only; not expected return.' if variant=='ranker' else None))
        cp=run/'CONFIG.json'
        if cp.exists() and json.loads(cp.read_text())!=config:raise FileExistsError('Leaf run has different configuration.')
        if not cp.exists():base.save_json(cp,config)
        for name in ['supervised_panel.parquet','wfo_calendar.parquet','label_coverage.parquet','outcome_status_counts.parquet']:
            destination=run/name
            if not destination.exists():destination.symlink_to(source/name)
        if not (run/'source_snapshot').exists():(run/'source_snapshot').symlink_to(snapshot,target_is_directory=True)
        if not (run/'PANEL.json').exists():base.save_json(run/'PANEL.json',dict(rows=manifest['summary']['rows'],
            model_features=MODEL_FEATURES,panel_sha256=common['panel_sha256'],source=str(source),forecast_cohort_identical=True,
            labels_outcomes_features_unchanged=True,one_minute_data_used=False))
    return suite_config


def tuning_inventory(run):
    records=[];selected=0
    for fold in base.folds():
        folder=run/'folds'/fold['fold_id']
        if (folder/'TUNING_PROGRESS.json').exists():records.extend(json.loads((folder/'TUNING_PROGRESS.json').read_text())['trials'])
        if (folder/'TUNING.json').exists():selected+=1
    return records,selected


def task_queues(suite):
    queues={v:deque() for v in VARIANTS}
    for variant in VARIANTS:
        run=suite/variant
        for fold in base.folds():
            fid=fold['fold_id']
            if variant=='ret_tuned' and not (run/'folds'/fid/'TUNING.json').exists():
                queues[variant].append(dict(variant=variant,kind='tune',fold_id=fid,members=[]));continue
            missing=[m for m in range(100) if not engine.checkpoint_complete(run,fid,m)]
            for k in range(0,len(missing),5):queues[variant].append(dict(variant=variant,kind='variant_batch',fold_id=fid,members=missing[k:k+5]))
    return queues


def start_job(suite,job):
    run=suite/job['variant']
    label=job['kind']+'_'+job.get('fold_id','all')+('_'+str(job['members'][0]) if job['members'] else '')
    path=run/'logs'/(label+'.log');log=path.open('a')
    log.write('\nATTEMPT '+utcnow()+'\n');log.flush()
    command=[sys.executable,'-B',str(run/'source_snapshot/scripts/train_xgb_ret_ensemble.py'),
        '--mode',job['kind'],'--run-dir',str(run),'--fold-id',job.get('fold_id','2017Q1')]
    if job['members']:command += ['--members',','.join(map(str,job['members']))]
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        VECLIB_MAXIMUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
    return dict(job=job,process=process,log=log,label=label)


def publish_progress(suite,active,started,peak):
    leaves={}
    for variant in VARIANTS:
        run=suite/variant;counts=engine.inventory(run);total=sum(counts.values())
        current=json.loads((run/'RUN_STATUS.json').read_text()) if (run/'RUN_STATUS.json').exists() else {}
        phase=('complete' if current.get('status')=='complete' and (run/'RESULTS.json').exists()
            else 'finalizing' if any(x['job']['variant']==variant and x['job']['kind']=='variant_finalize' for x in active)
            else 'running')
        leaf=dict(status=phase,variant=variant,completed_members=total,target_members=4000,
            complete_member_sets=sum(x==100 for x in counts.values()),completed_by_fold=counts,
            workers_alive=sum(x['job']['variant']==variant for x in active),updated_at_utc=utcnow())
        if variant=='ret_tuned':
            trials,selected=tuning_inventory(run);leaf.update(tuning_fits=len(trials),tuning_target=2880,tuned_quarters=selected)
        if phase!='complete':base.save_json(run/'RUN_STATUS.json',leaf);base.save_json(run/'PROGRESS.json',leaf)
        leaves[variant]=leaf
    status=dict(status='running',controller_pid=os.getpid(),completed_members=sum(x['completed_members'] for x in leaves.values()),
        target_members=8000,variants=leaves,workers_alive=len(active),peak_combined_rss_gib=peak,
        session_started_at_utc=started,updated_at_utc=utcnow())
    base.save_json(suite/'RUN_STATUS.json',status);base.save_json(suite/'PROGRESS.json',status)
    # Estimates are observations, never fit or launch limits. Partial candidates
    # and selected settings make the early estimate deliberately approximate.
    trials,_=tuning_inventory(suite/'ret_tuned')
    rank_meta=[json.loads(p.read_text()) for p in (suite/'ranker').glob('folds/*/members/member_*.json')]
    if len(trials)>=3 and len(rank_meta)>=5:
        ret_meta=[json.loads(p.read_text()) for p in (suite/'ret_tuned').glob('folds/*/members/member_*.json')]
        suite_config=json.loads((suite/'CONFIG.json').read_text())
        mean_trial=float(np.mean([t['elapsed_seconds'] for t in trials]))
        mean_ret=float(np.mean([t['elapsed_seconds'] for t in ret_meta])) if ret_meta else suite_config['historical_ret_mean_member_seconds']
        mean_rank=float(np.mean([t['elapsed_seconds'] for t in rank_meta]))
        seconds=((2880-len(trials))*mean_trial+(4000-len(ret_meta))*mean_ret+(4000-len(rank_meta))*mean_rank)/10
        base.save_json(suite/'RUNTIME_ESTIMATE.json',dict(updated_at_utc=utcnow(),remaining_fit_wall_seconds=seconds,
            broad_upper_fit_wall_seconds=seconds*2,observed_trial_fits=len(trials),observed_ret_members=len(ret_meta),observed_rank_members=len(rank_meta),
            mean_trial_seconds=mean_trial,mean_ret_seconds=mean_ret,mean_rank_seconds=mean_rank,
            qualification='Early sampled folds/settings; RET final cost uses historical path42 until selected-setting member fits exist. Assumes ten busy workers; excludes final reporting and host sleep. Not a budget or stopping rule.'))
    print(f"SUITE: RET {leaves['ret_tuned']['completed_members']}/4000; tuning {leaves['ret_tuned']['tuning_fits']}/2880; ranker {leaves['ranker']['completed_members']}/4000; workers {len(active)}",flush=True)
    return status


def run_suite(project,suite,workers=10):
    suite=Path(suite).resolve();suite.mkdir(parents=True,exist_ok=True)
    with engine.controller_lock(suite):
        config=prepare_suite(project,suite,workers)
        if (suite/'COMPLETE.json').exists():print('Suite and report already complete.',flush=True);return
        started=utcnow();base.save_json(suite/'TRAINING_STARTED.json',dict(started_at_utc=started,controller_pid=os.getpid()))
        queues=task_queues(suite);active=[];finalizing=set();peak=0.;last=0.;turn=0
        previous=signal.getsignal(signal.SIGTERM)
        def stop(signum,frame):raise KeyboardInterrupt('User requested checkpoint stop.')
        signal.signal(signal.SIGTERM,stop)
        try:
            while True:
                for task in list(active):
                    code=task['process'].poll()
                    if code is None:continue
                    task['log'].close();active.remove(task)
                    job=task['job']
                    if code:raise RuntimeError(f"{job['variant']} {task['label']} failed; see its private log.")
                    if job['kind']=='tune':
                        run=suite/job['variant'];fid=job['fold_id']
                        missing=[m for m in range(100) if not engine.checkpoint_complete(run,fid,m)]
                        tasks=[dict(variant='ret_tuned',kind='variant_batch',fold_id=fid,members=missing[i:i+5]) for i in range(0,len(missing),5)]
                        queues['ret_tuned'].extendleft(reversed(tasks))
                for variant in VARIANTS:
                    run=suite/variant
                    if variant not in finalizing and not (run/'RESULTS.json').exists() and sum(engine.inventory(run).values())==4000:
                        # Last workers must exit before finalization reads their files.
                        if not any(x['job']['variant']==variant for x in active):
                            queues[variant].appendleft(dict(variant=variant,kind='variant_finalize',fold_id='2017Q1',members=[]));finalizing.add(variant)
                while len(active)<workers and any(queues.values()):
                    choices=[VARIANTS[(turn+i)%2] for i in range(2)]
                    variant=next(v for v in choices if queues[v]);turn=1-VARIANTS.index(variant)
                    active.append(start_job(suite,queues[variant].popleft()))
                rss=psutil.Process().memory_info().rss
                for task in active:
                    try:rss+=psutil.Process(task['process'].pid).memory_info().rss
                    except psutil.NoSuchProcess:pass
                peak=max(peak,rss/1024**3)
                if time.monotonic()-last>=20:
                    publish_progress(suite,active,started,peak);last=time.monotonic()
                if not active and not any(queues.values()):
                    if not all((suite/v/'RESULTS.json').exists() for v in VARIANTS):raise RuntimeError('Tasks drained without two complete experiments.')
                    break
                time.sleep(.25)
            status=publish_progress(suite,[],started,peak);status['status']='reporting';base.save_json(suite/'RUN_STATUS.json',status)
            frozen=suite/'source_snapshot'
            subprocess.run([sys.executable,'-B',str(frozen/'scripts/run_notebook.py'),'11','--project',str(frozen)],check=True)
            status.update(status='complete',completed_at_utc=utcnow(),workers_alive=0)
            base.save_json(suite/'RUN_STATUS.json',status);base.save_json(suite/'PROGRESS.json',status)
            base.save_json(suite/'COMPLETE.json',dict(completed_at_utc=utcnow(),member_models=8000,tuning_fits=2880,report='11_ret_tuning_and_ranker'))
        except BaseException as error:
            for task in active:base.terminate_owned(task['process']);task['log'].close()
            active.clear()
            status=publish_progress(suite,[],started,peak);status.update(status='stopped_resumable',reason=str(error))
            for variant in VARIANTS:
                leaf=status['variants'][variant]
                if leaf['status']!='complete':
                    leaf.update(status='stopped_resumable',reason=str(error),workers_alive=0)
                    base.save_json(suite/variant/'RUN_STATUS.json',leaf);base.save_json(suite/variant/'PROGRESS.json',leaf)
            base.save_json(suite/'RUN_STATUS.json',status);base.save_json(suite/'PROGRESS.json',status)
            raise
        finally:signal.signal(signal.SIGTERM,previous)


def dispatch(args):
    if args.mode=='comparison':run_suite(args.project,args.run_dir,args.workers)
    elif args.mode=='tune':tuning_worker(args.run_dir,args.fold_id)
    elif args.mode=='variant_batch':batch_worker(args.run_dir,args.fold_id,[int(x) for x in args.members.split(',')])
    elif args.mode=='variant_finalize':finalize_variant(args.run_dir)
    else:raise ValueError('Unknown variant operation.')


def report_suite(project):
    """Inspect completed predictions only. Never build panels or fit models."""
    project=Path(project).resolve();private=(project/'data').resolve()
    suite=private/'models'/SUITE_NAME;out=private/'reports/11_ret_tuning_and_ranker';out.mkdir(parents=True,exist_ok=True)
    if not all((suite/v/'RESULTS.json').exists() and json.loads((suite/v/'RUN_STATUS.json').read_text())['status']=='complete' for v in VARIANTS):
        raise FileNotFoundError('Both complete 4,000-member runs are required; inspection will not train missing models.')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from analyze_volume_overnight_results import extract_importance,analyze_forecasts
    historical=private/'models/10_xgb_ret_rich_path100_v1'
    runs={'historical_ret42':historical,'tuned_ret42':suite/'ret_tuned','ranker42':suite/'ranker'}
    books=[]
    for name,run in runs.items():
        b=pd.read_parquet(run/'portfolio_daily.parquet');b['model']=name;books.append(b)
    books=pd.concat(books,ignore_index=True)
    wide=books.pivot(index='trade_date',columns=['model','n_per_side'],values='gross_return')
    common=wide.notna().all(axis=1);dates=common[common].index
    common.to_frame('included').to_parquet(out/'common_date_mask.parquet')
    records=[]
    for (name,n),book in books.groupby(['model','n_per_side']):
        selected=book.set_index('trade_date').loc[dates]
        for cost in (0,1,5,10):records.append(dict(model=name,n_per_side=int(n),cost_bps_per_side=cost,
            calendar_sessions=len(book),omitted_dates=int((~common).sum()),metric_scope='six_way_common_observed_dates_conditional',
            **base.performance(selected.gross_return-selected.trade_notional*cost/10000)))
    comparison=pd.DataFrame(records);comparison.to_parquet(out/'common_date_comparison.parquet',index=False)
    comparison.to_csv(out/'common_date_comparison.csv',index=False)
    colors={'historical_ret42':'#8493a3','tuned_ret42':'#267b82','ranker42':'#b56b3a'}
    fig,axes=plt.subplots(1,2,figsize=(13,4.5))
    for ax,cost in zip(axes,(0,1)):
        for (name,n),book in books.groupby(['model','n_per_side']):
            b=book.set_index('trade_date').loc[dates]
            ax.plot(dates,(1+b.gross_return-b.trade_notional*cost/10000).cumprod(),
                color=colors[name],ls='-' if n==3 else '--',label=f'{name}, N={n}')
        ax.set_title(f'Conditional equity: {cost} bp/side');ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'portfolio_comparison.png',dpi=160);plt.close(fig)
    # RET tuning must retain the historical member splits; ranker uses separate
    # complete-date splits and is explicitly not a matched-split causal comparison.
    fields=['split_seed','booster_seed','split_train_index_sha256','split_validation_index_sha256','prediction_row_key_sha256']
    checked=0
    for fold in base.folds():
        fid=fold['fold_id'];new=suite/'ret_tuned/folds'/fid;old=historical/'folds'/fid
        nf=json.loads((new/'FOLD.json').read_text());of=json.loads((old/'FOLD.json').read_text())
        np.testing.assert_allclose(nf['payoff_vector_class_1_to_10'],of['payoff_vector_class_1_to_10'],rtol=0,atol=1e-12)
        for member in range(100):
            name=f'member_{member:03d}.json'
            a=json.loads((new/'members'/name).read_text());b=json.loads((old/'members'/name).read_text())
            if not all(a[k]==b[k] for k in fields):raise ValueError('Tuned RET changed an authorized member split or forecast key.')
            checked+=1
    base.save_json(out/'COMPARISON_AUDIT.json',dict(ret_members_checked=checked,ret_splits_payoffs_forecast_rows_match=True,
        ranker_split_policy='Whole dates, intentionally different from RET row splits.',features_and_outer_forecast_cohort_unchanged=True))
    tuning=[]
    for fold in base.folds():
        selected=read_tuning(suite/'ret_tuned',fold['fold_id'])
        tuning.append(dict(fold_id=fold['fold_id'],selected_candidate=selected['selected_candidate'],
            mean_validation_logloss=selected['candidate_mean_logloss'][selected['selected_candidate']],**selected['selected_parameters']))
    pd.DataFrame(tuning).to_csv(out/'selected_hyperparameters.csv',index=False)
    # Classification return fit is defined only for the two RET variants.
    ev_out=out/'return_fit';ev_out.mkdir(exist_ok=True)
    ev,annual,_,deciles,ev_info=analyze_forecasts(suite/'ret_tuned',historical,ev_out,
        old_label='historical_ret42',new_label='tuned_ret42')
    fig,axes=plt.subplots(1,2,figsize=(12,4.4))
    for ax,scope in zip(axes,['All observed stock-days','Within-day cross-section']):
        f=ev[ev.scope.eq(scope)]
        ax.bar(f.feature_set,f.explained_variance*100,color=[colors[x] for x in f.feature_set]);ax.axhline(0,color='k',lw=.6)
        ax.set_title(scope);ax.set_ylabel('Return explained variance (%)')
    fig.suptitle('RET return fit; ranker margins are not calibrated returns')
    fig.tight_layout();fig.savefig(out/'ret_explained_variance.png',dpi=160);plt.close(fig)
    importance={}
    importance['historical_ret42']=pd.read_parquet(private/'reports/10_train_xgb_ret_rich_path/feature_diagnostics/feature_importance.parquet')
    for label,key in [('tuned_ret42','ret_tuned'),('ranker42','ranker')]:
        dest=out/(key+'_importance');dest.mkdir(exist_ok=True)
        importance[label],_=extract_importance(suite/key,dest,allow_constant_models=key=='ranker')
    names=importance['tuned_ret42'].sort_values('mean_gain_share',ascending=False).feature.tolist()
    fig,ax=plt.subplots(figsize=(12,13.8));y=np.arange(len(names))
    for shift,label in zip([-.24,0,.24],runs):
        f=importance[label].set_index('feature').reindex(names)
        ax.barh(y+shift,f.mean_gain_share*100,height=.22,label=label,color=colors[label])
    ax.set_yticks(y,names,fontsize=8);ax.invert_yaxis();ax.legend();ax.set_xlabel('Within-model normalized training gain (%)')
    ax.set_title('Training gain by feature; classification and ranking losses differ')
    fig.tight_layout();fig.savefig(out/'feature_importance.png',dpi=150);plt.close(fig)
    base.save_json(out/'SUMMARY.json',dict(status='complete',final_models=8000,ret_tuning_fits=2880,
        common_observed_dates=len(dates),omitted_dates=int((~common).sum()),comparison=comparison.to_dict('records'),
        return_EV=ev.to_dict('records'),ranker_return_EV=None,
        qualifications=['RET row splits unchanged. Ranker whole-date splits differ, so the direct comparison includes a validation-protocol change.',
            'RET hyperparameter selection uses past-only whole-date validation mlogloss; ranker has fixed untuned parameters.',
            'Ranker raw/percentile scores are not returns; no uncalibrated return EV/R2 is reported.',
            'Same-opening information/fills remain optimistic. Unknown selected exits remain unresolved.',
            'All metrics are development evidence; previously viewed history is not a fresh final holdout.']))
    return out
