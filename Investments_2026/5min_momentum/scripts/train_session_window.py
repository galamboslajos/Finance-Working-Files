"""Unchanged RET and portfolio-stopped ranker on audited ten-session inputs.

Preparation, fitting and inspection are separate entry points. All artifacts
are private and versioned. Inspection never starts fits or constructs a panel.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
from pathlib import Path
from datetime import datetime,timezone,timedelta
from collections import deque
import argparse,gc,json,shutil,signal,subprocess,sys,time
import duckdb
import numpy as np
import pandas as pd
import psutil
import ret_backtest as base
import train_xgb_ret_ensemble as engine
import train_ranker_portfolio as ranking
import learner_experiments as protocol
from session_window_features import FEATURES

NAME='16_session_window_ret_ranker100_v1'
REPORT='16_train_session_window'
VARIANTS=('ret','ranker')
ERROR_REASONS=(1,2,7)

def utcnow():return datetime.now(timezone.utc).isoformat()

def eligible_history(ledger):
    """Only known historical data quality enters this gate, never targets."""
    return (ledger.window_price_complete & ledger.window_activity_complete
            & ledger.source_consistent).fillna(False)

def attach_outcomes(con):
    """Same endpoint/status policy as historical RET; no eligibility feedback.

    Relations candidates and anchors must already exist. Unknown selected exits
    remain forecasts. Recompute exact-tie labels within this declared cohort.
    """
    con.execute('''CREATE TEMP TABLE examples AS
      WITH joined AS (SELECT c.*,d.session_open,d.session_close,
        CASE WHEN d.security_id IS NULL THEN 'no_forward_member_day_prices'
          WHEN d.opening_rows!=1 OR d.session_open IS NULL THEN 'missing_or_invalid_open'
          WHEN d.closing_rows!=1 OR d.session_close IS NULL THEN 'missing_or_invalid_close'
          WHEN d.duplicate_rows!=0 THEN 'duplicate_forward_bar'
          WHEN d.source_count!=1 OR d.missing_source_rows!=0 THEN 'ambiguous_forward_source'
          WHEN NOT coalesce(isfinite(d.session_open) AND isfinite(d.session_close)
            AND d.session_open>0 AND d.session_close>0,false) THEN 'invalid_forward_price'
          ELSE 'ok' END outcome_status
        FROM candidates c LEFT JOIN anchors d
          ON c.security_id=d.security_id AND c.trade_date=d.signal_date)
      SELECT *,CASE WHEN outcome_status='ok' THEN session_close/session_open-1 END future_oc_return
      FROM joined''')
    base.build_labels(con)

def source_hashes():
    return {key:base.fingerprint(Path(__file__).with_name(name)) for name,key in (
        ('train_xgb_ret_ensemble.py','code_sha256'),('ret_backtest.py','baseline_code_sha256'),
        ('xgb_validation_stopping.py','stopping_code_sha256'),('learner_experiments.py','protocol_code_sha256'),
        ('portfolio_stopping.py','portfolio_metric_sha256'),('train_ranker_portfolio.py','portfolio_trainer_sha256'),
        ('train_session_window.py','session_trainer_sha256'))}

def verify(suite,*,inputs=False):
    suite=Path(suite);pinned=json.loads((suite/'PINNED.json').read_text())
    frozen=suite/'source_snapshot'
    for name,digest in pinned['sources'].items():
        if base.fingerprint(frozen/name)!=digest:raise ValueError('Frozen source changed: '+name)
    for name,digest in pinned['configs'].items():
        if base.fingerprint(suite/name)!=digest:raise ValueError('Frozen configuration changed: '+name)
    for name,record in pinned['inputs'].items():
        path=Path(name);stat=path.stat()
        if stat.st_size!=record['bytes'] or stat.st_mtime_ns!=record['mtime_ns']:
            raise ValueError('Pinned input stat changed: '+name)
        if inputs and base.fingerprint(path)!=record['sha256']:raise ValueError('Pinned input changed: '+name)
    return json.loads((suite/'CONFIG.json').read_text())

def prepare(project,suite,workers=10):
    project=Path(project).resolve();private=(project/'data').resolve();suite=Path(suite).resolve()
    if not suite.is_relative_to(private):raise ValueError('Licensed outputs must stay in private storage.')
    if (suite/'PINNED.json').exists():verify(suite,inputs=True);return
    if (suite/'PANEL.json').exists():raise ValueError('Unfinished preparation requires inspection; do not overwrite a release.')
    suite.mkdir(parents=True,exist_ok=True)
    release=private/'features/15_session_window_10_v1'
    manifest=json.loads((release/'MANIFEST.json').read_text())
    audit_path=private/'reports/15_build_session_window_features/WINDOW_AUDIT.json'
    audit=json.loads(audit_path.read_text())
    if not manifest['complete'] or manifest['model_features']!=FEATURES or not audit['all_passed']:
        raise ValueError('Audited 267-feature release required.')
    if audit['feature_manifest_sha256']!=base.fingerprint(release/'MANIFEST.json'):
        raise ValueError('Construction audit does not match the input release.')
    for name in ('features.parquet','coverage_ledger.parquet','exchange_calendar.parquet'):
        if base.fingerprint(release/name)!=manifest['outputs_sha256'][name]:raise ValueError('Feature output changed: '+name)
    ledger=pd.read_parquet(release/'coverage_ledger.parquet')
    ledger['history_eligible']=eligible_history(ledger)
    reasons=np.frombuffer(b''.join(ledger.missing_reason_bytes),np.uint8).reshape(len(ledger),len(FEATURES))
    no_errors=~np.isin(reasons,ERROR_REASONS).any(axis=1)
    if not np.array_equal(no_errors,ledger.history_eligible):raise ValueError('Quality gate and per-feature data errors differ.')
    ledger['inside_outcome_scope']=ledger.trade_date.lt(pd.Timestamp(base.EVALUATION_END))
    ledger['forecast_eligible']=ledger.history_eligible & ledger.inside_outcome_scope
    ledger['exclusion_reason']=np.select([~ledger.history_eligible,~ledger.inside_outcome_scope],
        ['historical_price_activity_or_source_error','outside_declared_outcome_scope'],default='included')
    ledger.drop(columns='missing_reason_bytes').to_parquet(suite/'eligibility_ledger.parquet',index=False)
    anchors=private/'features/02_momentum_total_v1/daily_anchors.parquet'
    old=private/'models/10_xgb_ret_rich_path100_v1/supervised_panel.parquet'
    con=duckdb.connect();con.execute('SET threads=2');con.execute("SET TimeZone='UTC'")
    con.read_parquet(str(anchors)).create_view('anchors')
    con.read_parquet(str(release/'features.parquet')).create_view('features')
    con.register('gate',ledger[['security_id','signal_date','trade_date','forecast_eligible','feature_time_upper_bound_utc']])
    con.execute('''CREATE TEMP TABLE candidates AS
      SELECT f.*,coalesce(d.ticker,f.security_id) ticker,g.feature_time_upper_bound_utc signal_cutoff_utc
      FROM features f JOIN gate g USING(security_id,signal_date,trade_date)
      LEFT JOIN anchors d ON f.security_id=d.security_id AND f.signal_date=d.signal_date
      WHERE g.forecast_eligible''')
    attach_outcomes(con)
    con.execute(f'''COPY (SELECT e.*,l.realized_class FROM examples e LEFT JOIN labels l
      USING(security_id,signal_date) ORDER BY trade_date,security_id)
      TO {base.literal(suite/'supervised_panel.parquet')} (FORMAT PARQUET,COMPRESSION ZSTD)''')
    con.read_parquet(str(old)).create_view('legacy')
    # Outcomes must agree on overlapping names; labels intentionally change with the cohort.
    mismatches=con.execute('''SELECT count(*) FROM examples e JOIN legacy p USING(security_id,trade_date)
      WHERE e.outcome_status IS DISTINCT FROM p.outcome_status
        OR e.future_oc_return IS DISTINCT FROM p.future_oc_return''').fetchall()[0][0]
    if mismatches:raise ValueError('Target policy changed on overlapping historical rows.')
    differences=con.execute('''WITH a AS (SELECT security_id,trade_date FROM examples),
      b AS (SELECT security_id,trade_date FROM legacy)
      SELECT count_if(a.security_id IS NOT NULL AND b.security_id IS NOT NULL) common_rows,
        count_if(a.security_id IS NOT NULL AND b.security_id IS NULL) new_only_rows,
        count_if(a.security_id IS NULL AND b.security_id IS NOT NULL) legacy_only_rows
      FROM a FULL OUTER JOIN b USING(security_id,trade_date)''').df().iloc[0].to_dict()
    con.execute(f'''COPY (SELECT coalesce(a.security_id,b.security_id) security_id,
      coalesce(a.trade_date,b.trade_date) trade_date,a.security_id IS NOT NULL new_cohort,
      b.security_id IS NOT NULL legacy_cohort FROM (SELECT security_id,trade_date FROM examples) a
      FULL OUTER JOIN (SELECT security_id,trade_date FROM legacy) b USING(security_id,trade_date))
      TO {base.literal(suite/'cohort_comparison.parquet')} (FORMAT PARQUET,COMPRESSION ZSTD)''')
    coverage=con.execute('''SELECT year(e.trade_date) AS "year",count(*) forecasts,
      count_if(outcome_status='ok') known_outcomes,count(l.realized_class) labels
      FROM examples e LEFT JOIN labels l USING(security_id,signal_date) GROUP BY 1 ORDER BY 1''').df()
    coverage.to_parquet(suite/'label_coverage.parquet',index=False)
    con.execute('SELECT outcome_status,count(*) AS "rows" FROM examples GROUP BY 1').df().to_parquet(suite/'outcome_status_counts.parquet',index=False)
    counts=con.execute('SELECT count(*) AS "rows",count_if(outcome_status!=\'ok\') unknown_outcomes FROM examples').df().iloc[0].to_dict()
    con.close()
    calendar=pd.read_parquet(release/'exchange_calendar.parquet')
    calendar=calendar.loc[calendar.signal_date.ge(base.EVALUATION_START)&calendar.signal_date.lt(base.EVALUATION_END),['signal_date']]
    calendar.rename(columns={'signal_date':'trade_date'}).to_parquet(suite/'wfo_calendar.parquet',index=False)
    info=dict(version=NAME,feature_manifest_sha256=base.fingerprint(release/'MANIFEST.json'),
        model_features=FEATURES,rows=int(counts['rows']),unknown_outcomes=int(counts['unknown_outcomes']),
        history_eligible_rows=int(ledger.history_eligible.sum()),history_ineligible_rows=int((~ledger.history_eligible).sum()),
        eligible_with_definition_NaNs=int((ledger.history_eligible & ledger.missing_features.gt(0)).sum()),
        cohort_comparison=differences,overlap_target_mismatches=int(mismatches),construction_checks=len(audit['checks']),
        feature_release_unchanged=True,current_trade_day_inputs=False,one_minute_data_used=False,
        feature_cutoff='Previous completed regular-session close; exactly ten sessions and nine internal gaps.',
        missing_policy='Exclude historical data/source errors 1/2/7 only. Keep legitimate undefined states as native NaNs; no filling or flags added.',
        labels='Exact-tie deciles recalculated within the new outcome-observed forecast cohort. Unknown outcomes stay forecasts.',
        comparison='Feature and historical-quality cohort replacement, not a feature-only controlled comparison against legacy 42.',
        output_sha256={name:base.fingerprint(suite/name) for name in ('supervised_panel.parquet','eligibility_ledger.parquet','cohort_comparison.parquet','wfo_calendar.parquet')})
    base.save_json(suite/'PANEL.json',info)
    hashes=source_hashes()
    common=dict(members_per_quarter=100,total_quarters=40,total_member_models=4000,train_months=12,test_months=3,
        model_features=FEATURES,n_per_side=[2,3],gross_exposure=1.,workers=workers,threads_per_worker=1,
        max_rounds=None,memory_limit_gib=None,minimum_available_memory_gib=0.,hard_stop_utc=None,not_before_utc=None,
        early_stopping_patience=30,base_seed=engine.BASE_SEED,historical_split_policy='same_seed_method_new_cohort',
        panel_sha256=info['output_sha256']['supervised_panel.parquet'],feature_manifest_sha256=info['feature_manifest_sha256'],
        current_trade_day_inputs=False,one_minute_data_used=False,tuning_grid=[],**hashes)
    for variant in VARIANTS:
        run=suite/variant;run.mkdir();(run/'logs').mkdir()
        config=dict(common,variant=variant,version='session_window_'+variant+'100_v1',experiment='session267',
            parameters=base.PARAMETERS if variant=='ret' else protocol.ranking_parameters(engine.BASE_SEED),
            inner_split='Random 80/20 stock rows' if variant=='ret' else 'Random 80/20 whole dates',
            stopping_metric='mlogloss' if variant=='ret' else ranking.METRIC,
            maximize_metric=variant=='ranker',standard_cost_bps_per_side=10,cost_sensitivities_bps_per_side=[5,10,20])
        base.save_json(run/'CONFIG.json',config)
        for name in ('supervised_panel.parquet','wfo_calendar.parquet','label_coverage.parquet','outcome_status_counts.parquet'):
            (run/name).symlink_to(suite/name)
    base.save_json(suite/'CONFIG.json',dict(version=NAME,variants=VARIANTS,workers=workers,total_member_models=8000,
        members_per_quarter=100,total_quarters=40,early_stopping_patience=30,train_months=12,test_months=3,
        tuning=False,round_RAM_time_caps=None,source_sha256=hashes))
    frozen=suite/'source_snapshot';(frozen/'scripts').mkdir(parents=True);(frozen/'notebooks').mkdir()
    for p in (project/'scripts').glob('*.py'):shutil.copy2(p,frozen/'scripts'/p.name)
    shutil.copy2(project/'notebooks/16_train_session_window.ipynb',frozen/'notebooks/16_train_session_window.ipynb')
    (frozen/'data').symlink_to(private,target_is_directory=True)
    sources={str(p.relative_to(frozen)):base.fingerprint(p) for p in sorted(frozen.rglob('*')) if p.is_file() and 'data' not in p.relative_to(frozen).parts}
    base.save_json(frozen/'SOURCE_MANIFEST.json',sources)
    for variant in VARIANTS:(suite/variant/'source_snapshot').symlink_to(frozen,target_is_directory=True)
    files=[release/'MANIFEST.json',release/'features.parquet',release/'coverage_ledger.parquet',release/'exchange_calendar.parquet',audit_path,anchors,old,suite/'supervised_panel.parquet',suite/'wfo_calendar.parquet',suite/'eligibility_ledger.parquet',suite/'cohort_comparison.parquet']
    pinned=dict(sources=sources,configs={str(p.relative_to(suite)):base.fingerprint(p) for p in [suite/'CONFIG.json',suite/'ret/CONFIG.json',suite/'ranker/CONFIG.json']},
        inputs={str(p):dict(bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns,sha256=base.fingerprint(p)) for p in files})
    base.save_json(suite/'PINNED.json',pinned)
    audit_prepared(suite)
    print(json.dumps(info,indent=2),flush=True)

def audit_prepared(suite):
    suite=Path(suite);verify(suite,inputs=True)
    con=duckdb.connect();con.execute('SET threads=2')
    con.read_parquet(str(suite/'supervised_panel.parquet')).create_view('p')
    checks={}
    pinned=json.loads((suite/'PINNED.json').read_text())
    feature_path=next(Path(p) for p in pinned['inputs'] if p.endswith('/15_session_window_10_v1/features.parquet'))
    con.read_parquet(str(feature_path)).create_view('original_features')
    checks['exact_feature_values_and_masks']=con.execute('SELECT count(*)=0 FROM p JOIN original_features f USING(security_id,signal_date,trade_date) WHERE '+ ' OR '.join(f'p.{c} IS DISTINCT FROM f.{c}' for c in FEATURES)).fetchall()[0][0]
    checks['unique_keys']=con.execute('SELECT count(*)=0 FROM (SELECT security_id,trade_date FROM p GROUP BY ALL HAVING count(*)>1)').fetchall()[0][0]
    checks['historical_feature_clock']=con.execute('SELECT count(*)=0 FROM p WHERE signal_date>=trade_date OR signal_cutoff_utc>=CAST(trade_date AS TIMESTAMPTZ)').fetchall()[0][0]
    checks['no_infinite_inputs']=con.execute('SELECT count(*)=0 FROM p WHERE '+ ' OR '.join(f'isinf({c})' for c in FEATURES)).fetchall()[0][0]
    checks['labels_only_on_known_outcomes']=con.execute("SELECT count(*)=0 FROM p WHERE realized_class IS NOT NULL AND (outcome_status!='ok' OR NOT isfinite(future_oc_return))").fetchall()[0][0]
    checks['unknown_outcomes_retained']=con.execute("SELECT count(*)>0 FROM p WHERE outcome_status!='ok' AND future_oc_return IS NULL AND realized_class IS NULL").fetchall()[0][0]
    checks['correct_open_close_target']=con.execute("SELECT count(*)=0 FROM p WHERE outcome_status='ok' AND future_oc_return IS DISTINCT FROM session_close/session_open-1").fetchall()[0][0]
    con.read_parquet(str(suite/'eligibility_ledger.parquet')).create_view('g')
    checks['no_future_label_eligibility']=con.execute('SELECT count(*)=0 FROM p FULL OUTER JOIN (SELECT * FROM g WHERE forecast_eligible) g USING(security_id,signal_date,trade_date) WHERE p.security_id IS NULL OR g.security_id IS NULL').fetchall()[0][0]
    checks['dense_calendar']=not pd.read_parquet(suite/'wfo_calendar.parquet').trade_date.duplicated().any()
    checks['ten_sessions_nine_gaps']=con.execute('SELECT count(*)=0 FROM g WHERE expected_sessions!=10 OR expected_gaps!=9').fetchall()[0][0]
    # Independently recompute average-rank midpoint classes, not the label SQL.
    slim=con.execute('SELECT trade_date,realized_class,future_oc_return FROM p ORDER BY trade_date,security_id').df()
    known=slim.future_oc_return.notna();s=slim[known].copy();groups=s.groupby('trade_date').future_oc_return
    percentile=(groups.rank(method='average',ascending=False)-.5)/groups.transform('size')
    expected=np.where(groups.transform('size')>=10,np.minimum(10,1+np.floor(10*percentile)),np.nan)
    checks['independent_exact_tie_labels']=bool(np.array_equal(expected,s.realized_class,equal_nan=True))
    con.close()
    for variant in VARIANTS:
        c=json.loads((suite/variant/'CONFIG.json').read_text())
        checks[variant+'_learner_unchanged']=c['parameters']==(base.PARAMETERS if variant=='ret' else protocol.ranking_parameters(engine.BASE_SEED))
        checks[variant+'_uncapped_100_12_3_patience30']=(c['members_per_quarter']==100 and c['train_months']==12 and c['test_months']==3 and c['early_stopping_patience']==30 and c['max_rounds'] is None and c['memory_limit_gib'] is None and c['hard_stop_utc'] is None and c['minimum_available_memory_gib']==0)
    result=dict(all_passed=all(checks.values()),checks=checks,at_utc=utcnow(),model_fits=0)
    base.save_json(suite/'PRETRAIN_AUDIT.json',result)
    if not result['all_passed']:raise ValueError('Pretraining audit failed: '+str(checks))
    return result

def checkpoint(suite,variant,fid,member):
    run=Path(suite)/variant
    return ranking.checkpoint_complete(run,fid,member) if variant=='ranker' else engine.checkpoint_complete(run,fid,member)

def progress(suite,active,status,started,peak,reason=None):
    counts={v:engine.inventory(suite/v) for v in VARIANTS}
    state=dict(status=status,controller_pid=os.getpid(),updated_at_utc=utcnow(),session_started_at_utc=started,
        completed_members={v:sum(c.values()) for v,c in counts.items()},target_members_per_learner=4000,
        completed_by_fold=counts,workers_alive=len(active),peak_combined_rss_gib=peak,reason=reason)
    for variant in VARIANTS:
        metadata=[json.loads(p.read_text()) for p in (suite/variant).glob('folds/*/members/member_*.json')]
        alive=sum(t['variant']==variant for t in active)
        base.save_json(suite/variant/'RUN_STATUS.json',dict(status=status,completed_members=len(metadata),target_members=4000,workers_alive=alive,updated_at_utc=state['updated_at_utc']))
    # Recent completion rate uses current-session events; excludes suspended time.
    now=datetime.now(timezone.utc);begin=datetime.fromisoformat(started)
    recent={v:[] for v in VARIANTS}
    for v in VARIANTS:
        for p in (suite/v).glob('folds/*/members/member_*.json'):
            m=json.loads(p.read_text());at=datetime.fromisoformat(m['fit_completed_at_utc'])
            if at>=begin:recent[v].append(at)
    total=sum(len(a) for a in recent.values());elapsed=(now-begin).total_seconds()
    if total>=10 and elapsed>0:
        remaining=sum(4000-state['completed_members'][v] for v in VARIANTS)
        seconds=remaining*elapsed/total
        state['estimated_finish_utc']=(now+timedelta(seconds=seconds)).isoformat()
        state['estimated_remaining_seconds']=seconds
        state['estimate_qualification']='Current-session aggregate throughput; early folds and learner mix may differ; excludes host sleep and reporting. Estimate only, no cap.'
        base.save_json(suite/'RUNTIME_ESTIMATE.json',state)
    base.save_json(suite/'RUN_STATUS.json',state);base.save_json(suite/'PROGRESS.json',state)
    return state

def run_controller(suite):
    suite=Path(suite).resolve()
    with engine.controller_lock(suite):
        config=verify(suite,inputs=True)
        if not json.loads((suite/'PRETRAIN_AUDIT.json').read_text())['all_passed']:raise ValueError('Pretraining audit missing/failed.')
        if (suite/'COMPLETE.json').exists():return
        started=utcnow();base.save_json(suite/'TRAINING_STARTED.json',dict(at_utc=started,controller_pid=os.getpid()))
        queues={v:deque() for v in VARIANTS}
        for v in VARIANTS:
            for fold in base.folds():
                missing=[m for m in range(100) if not checkpoint(suite,v,fold['fold_id'],m)]
                for i in range(0,len(missing),5):queues[v].append((fold['fold_id'],missing[i:i+5]))
        active=[];peak=0.;last=0.;turn=0
        previous=signal.getsignal(signal.SIGTERM)
        def stop(signum,frame):raise KeyboardInterrupt('Resumable checkpoint stop requested.')
        signal.signal(signal.SIGTERM,stop)
        try:
            while active or any(queues.values()):
                if (suite/'STOP_REQUESTED.json').exists():
                    # Stop launching; current batches commit complete members before exit.
                    for v in VARIANTS:queues[v].clear()
                    if not active:
                        progress(suite,[], 'stopped_resumable',started,peak,'Graceful stop: all active batches checkpointed.')
                        return
                for task in list(active):
                    code=task['process'].poll()
                    if code is None:continue
                    task['log'].close();active.remove(task)
                    if code:raise RuntimeError('Worker failed: '+task['label']+'; inspect '+str(task['log_path']))
                while len(active)<config['workers'] and any(queues.values()) and not (suite/'STOP_REQUESTED.json').exists():
                    options=[v for v in VARIANTS if queues[v]]
                    v=VARIANTS[turn%2] if VARIANTS[turn%2] in options else options[0];turn+=1
                    fid,members=queues[v].popleft();label=f'{v}_{fid}_{members[0]:03d}'
                    path=suite/v/'logs'/(label+'.log');log=path.open('a')
                    command=[sys.executable,'-B',str(suite/'source_snapshot/scripts/train_session_window.py'),
                        '--mode','batch','--suite',str(suite),'--variant',v,'--fold-id',fid,'--members',','.join(map(str,members))]
                    process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    active.append(dict(process=process,log=log,log_path=path,label=label,variant=v))
                rss=psutil.Process().memory_info().rss
                for task in active:
                    try:rss+=psutil.Process(task['process'].pid).memory_info().rss
                    except psutil.NoSuchProcess:pass
                peak=max(peak,rss/1024**3)
                if time.monotonic()-last>=30:
                    state=progress(suite,active,'running',started,peak)
                    print(state['updated_at_utc'],state['completed_members'],flush=True);last=time.monotonic()
                time.sleep(.5)
            if (suite/'STOP_REQUESTED.json').exists():
                progress(suite,[],'stopped_resumable',started,peak,'Graceful stop; complete checkpoints retained.')
                return
            progress(suite,[],'finalizing',started,peak)
            for v in VARIANTS:finalize(suite/v)
            progress(suite,[],'reporting',started,peak)
            subprocess.run([sys.executable,'-B',str(suite/'source_snapshot/scripts/run_notebook.py'),'16','--project',str(suite/'source_snapshot')],check=True)
            progress(suite,[],'complete',started,peak)
            base.save_json(suite/'COMPLETE.json',dict(completed_at_utc=utcnow(),models=8000,report=REPORT))
        except BaseException as error:
            for task in active:base.terminate_owned(task['process']);task['log'].close()
            active.clear();progress(suite,[],'stopped_resumable' if isinstance(error,KeyboardInterrupt) else 'failed_resumable',started,peak,str(error));raise
        finally:signal.signal(signal.SIGTERM,previous)

def finalize(run):
    run=Path(run);variant=run.name
    if (run/'RESULTS.json').exists():return
    for fold in base.folds():
        folder=run/'folds'/fold['fold_id']
        if (folder/'FOLD.json').exists():continue
        if variant=='ret':engine.finalize_fold(run,fold);continue
        training,testing=base.read_fold(run,fold);key=engine.row_key_hash(testing)
        score=np.zeros(len(testing));metadata=[]
        for m in range(100):
            if not ranking.checkpoint_complete(run,fold['fold_id'],m):raise ValueError('Incomplete ranker member.')
            prefix=engine.member_prefix(run,fold['fold_id'],m);meta=json.loads(prefix.with_suffix('.json').read_text())
            if meta['prediction_row_key_sha256']!=key:raise ValueError('Forecast order mismatch.')
            with np.load(prefix.with_suffix('.npz'),allow_pickle=False) as saved:
                value=saved['percentile_scores']
                if value.shape!=score.shape or not np.isfinite(value).all() or (value<0).any() or (value>1).any():raise ValueError('Invalid percentiles.')
                score+=value
            metadata.append(meta)
        cols=['security_id','ticker','signal_date','signal_cutoff_utc','trade_date','outcome_status','future_oc_return','realized_class']
        predictions=testing[cols].copy();predictions['ranking_score']=score/100;predictions['ret_score']=predictions.ranking_score
        predictions['fold_id']=fold['fold_id'];predictions['ensemble_members']=100
        predictions.to_parquet(folder/'predictions.parquet',index=False)
        pd.DataFrame(metadata).to_parquet(folder/'members_summary.parquet',index=False)
        base.save_json(folder/'FOLD.json',dict(**fold,ensemble_members=100,prediction_rows=len(testing),prediction_row_key_sha256=key))
    predictions=pd.concat([pd.read_parquet(run/'folds'/f['fold_id']/'predictions.parquet') for f in base.folds()],ignore_index=True)
    if predictions.duplicated(['security_id','trade_date']).any():raise ValueError('Duplicate WFO forecasts.')
    books=[];positions=[];calendar=pd.read_parquet(run/'wfo_calendar.parquet')
    for n in (2,3):
        b,p=base.portfolio_days(predictions,calendar,n);books.append(b);positions.append(p)
    book=pd.concat(books,ignore_index=True)
    for cost in (0,5,10,20):book[f'net_return_{cost}bps']=book.gross_return-book.trade_notional*cost/10000
    predictions.to_parquet(run/'predictions.parquet',index=False)
    book.to_parquet(run/'portfolio_daily.parquet',index=False)
    pd.concat(positions,ignore_index=True).to_parquet(run/'positions.parquet',index=False)
    base.save_json(run/'RESULTS.json',dict(status='complete',models=4000,features=267,score_type=variant,
        standard_cost_bps_per_side=10,unknown_selected_dates=int(book.gross_return.isna().sum()),
        cohort_changed=True,feature_cutoff='Previous completed close',current_trade_day_inputs=False))

def report(project):
    project=Path(project).resolve();private=(project/'data').resolve();suite=private/'models'/NAME
    if not all((suite/v/'RESULTS.json').exists() for v in VARIANTS):
        raise FileNotFoundError('Both completed 4,000-member releases are required. Inspection cannot train or rebuild.')
    out=private/'reports'/REPORT;out.mkdir(parents=True,exist_ok=True)
    if (out/'SUMMARY.json').exists():return out
    verify(suite,inputs=True)
    from analyze_volume_overnight_results import extract_importance,score_returns
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    paths={'RET 42':private/'models/10_xgb_ret_rich_path100_v1',
        'Ranker 42':private/'models/12_ranker_portfolio100_v1',
        'RET 267':suite/'ret','Ranker 267':suite/'ranker'}
    books=pd.concat([pd.read_parquet(p/'portfolio_daily.parquet').assign(model=name) for name,p in paths.items()],ignore_index=True)
    wide=books.pivot(index='trade_date',columns=['model','n_per_side'],values='gross_return')
    common=wide.notna().all(axis=1);dates=common[common].index
    mask=wide.isna();mask['all_eight_observed']=common;mask.to_parquet(out/'comparison_date_ledger.parquet')
    records=[]
    for (name,n),book in books.groupby(['model','n_per_side']):
        for scope,use in [('eight_way_common_observed_dates',book.set_index('trade_date').loc[dates]),
                          ('own_observed_dates',book[book.gross_return.notna()])]:
            for cost in (0,5,10,20):
                records.append(dict(model=name,n_per_side=int(n),cost_bps_per_side=cost,scope=scope,
                    omitted_dates=len(book)-len(use),**base.performance(use.gross_return-use.trade_notional*cost/10000)))
    table=pd.DataFrame(records);table.to_csv(out/'portfolio_comparison.csv',index=False)
    colors={'RET 42':'#94a3b8','Ranker 42':'#b88d62','RET 267':'#2563eb','Ranker 267':'#16856f'}
    fig,axes=plt.subplots(2,2,figsize=(14,9))
    for ax,cost in zip(axes.flat,(0,5,10,20)):
        for (name,n),book in books.groupby(['model','n_per_side']):
            b=book.set_index('trade_date').loc[dates]
            ax.plot(dates,(1+b.gross_return-b.trade_notional*cost/10000).cumprod(),color=colors[name],ls='-' if n==2 else '--',label=f'{name}, {n}/side')
        ax.set_title(f'{cost} bp per side');ax.grid(alpha=.15);ax.legend(fontsize=7)
    fig.suptitle(f'Conditional growth: {len(dates)} common dates; {(~common).sum()} omitted. Feature and cohort changes.')
    fig.tight_layout();fig.savefig(out/'portfolio_comparison.png',dpi=150);plt.close(fig)
    importance_records=[]
    for variant in VARIANTS:
        dest=out/(variant+'_importance');dest.mkdir(exist_ok=True)
        importance,_=extract_importance(suite/variant,dest,allow_constant_models=True)
        importance['descriptor']=importance.feature.str.slice(6);importance['lag']=importance.feature.str.slice(0,5)
        importance['learner']=variant;importance_records.append(importance)
        top=importance.nlargest(30,'mean_gain_share').sort_values('mean_gain_share')
        fig,ax=plt.subplots(figsize=(10,9));ax.barh(top.feature,top.mean_gain_share*100,color=colors['RET 267' if variant=='ret' else 'Ranker 267'])
        ax.set_title(f'{variant}: top 30 training gain shares, not return EV');ax.set_xlabel('Mean within-model gain share (%)')
        fig.tight_layout();fig.savefig(out/(variant+'_feature_importance.png'),dpi=150);plt.close(fig)
    importance=pd.concat(importance_records,ignore_index=True)
    importance.to_csv(out/'feature_importance.csv',index=False)
    grouped=importance.groupby(['learner','descriptor']).mean_gain_share.sum().unstack(0)
    grouped.to_csv(out/'descriptor_importance.csv')
    fig,axes=plt.subplots(1,2,figsize=(13,8))
    grouped.plot.barh(ax=axes[0],title='Training gain by descriptor');axes[0].set_xlabel('Summed mean gain share')
    importance.groupby(['lag','learner']).mean_gain_share.sum().unstack().plot.bar(ax=axes[1],title='Training gain by historical session')
    fig.tight_layout();fig.savefig(out/'grouped_importance.png',dpi=150);plt.close(fig)
    a=pd.read_parquet(paths['RET 42']/'predictions.parquet',columns=['security_id','trade_date','future_oc_return','ret_score'])
    b=pd.read_parquet(suite/'ret/predictions.parquet',columns=['security_id','trade_date','future_oc_return','ret_score'])
    matched=a.merge(b,on=['security_id','trade_date'],suffixes=('_42','_267'),validate='one_to_one')
    if not np.array_equal(matched.future_oc_return_42,matched.future_oc_return_267,equal_nan=True):raise ValueError('Matched target mismatch.')
    matched=matched[np.isfinite(matched.future_oc_return_42)].copy();ev=[]
    for name,col in [('RET 42','ret_score_42'),('RET 267','ret_score_267')]:
        ev.append(dict(model=name,**score_returns(matched.future_oc_return_42,matched[col])))
    pd.DataFrame(ev).to_csv(out/'matched_return_EV.csv',index=False)
    matched[['security_id','trade_date']].to_parquet(out/'EV_observation_keys.parquet',index=False)
    fig,ax=plt.subplots(figsize=(7,4));e=pd.DataFrame(ev).set_index('model')
    e[['explained_variance','r2']].plot.bar(ax=ax);ax.axhline(0,color='black',lw=.7);ax.set_title(f'Return fit: {len(matched):,} matched observed stock-days')
    fig.tight_layout();fig.savefig(out/'explained_variance.png',dpi=150);plt.close(fig)
    details=[]
    for v in VARIANTS:
        for p in (suite/v).glob('folds/*/members/member_*.json'):
            m=json.loads(p.read_text());details.append(dict(learner=v,**m))
    pd.DataFrame(details).to_parquet(out/'member_diagnostics.parquet',index=False)
    if len(details)!=8000:raise ValueError('Exactly 8,000 models required.')
    base.save_json(out/'SUMMARY.json',dict(status='complete',models=8000,features=267,common_dates=len(dates),
        omitted_dates=int((~common).sum()),return_EV=ev,ranker_return_EV=None,
        cohort=json.loads((suite/'PANEL.json').read_text())['cohort_comparison'],
        qualifications=['Feature representation, historical-quality gate and cohort-conditioned labels/payoffs differ from legacy 42; no isolated feature attribution.',
            'RET keeps random stock-row inner splits; ranker keeps random whole-date splits and mean N2/N3 gross portfolio stopping.',
            'All new predictors end at the previous close. No current-opening gap; opening-fill proxies remain unverified execution.',
            'Gain measures training-loss reduction. Uncalibrated ranking scores have no return EV/R2.',
            'Conditional curves retain cash but omit unknown selected exits. Membership/adjusted vintages and underlying minutes remain provisional.',
            'Previously viewed 2017-2026 development history; not a fresh final holdout. No one-minute work.']))
    return out

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['prepare','audit','run','batch','report'],required=True)
    parser.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--suite',type=Path);parser.add_argument('--workers',type=int,default=10)
    parser.add_argument('--variant',choices=VARIANTS);parser.add_argument('--fold-id');parser.add_argument('--members')
    args=parser.parse_args();suite=args.suite or args.project/'data/models'/NAME
    if args.mode=='prepare':prepare(args.project,suite,args.workers)
    elif args.mode=='audit':print(audit_prepared(suite))
    elif args.mode=='run':run_controller(suite)
    elif args.mode=='report':print(report(args.project))
    else:
        verify(suite);members=[int(x) for x in args.members.split(',')]
        if args.variant=='ret':engine.batch_worker(suite/'ret',args.fold_id,members)
        else:ranking.batch(suite/'ranker',args.fold_id,members)
