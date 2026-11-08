"""100-fit RET ensembles with explicit momentum8 and combined15 input contracts.

Every member is checkpointed atomically. No reduced-member portfolio is ever
published as the ensemble. The resource deadline stops work, not the target count.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
from datetime import datetime,timezone,timedelta
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import duckdb
import numpy as np
import pandas as pd
import psutil

import ret_backtest as base

from build_volume_overnight_features import MODEL_FEATURES as COMBINED_FEATURES
from build_network_paper_features import MODEL_FEATURES as NETWORK_FEATURES
from build_momentum_features import session_calendar
from joint_features import MODEL_FEATURES as JOINT_FEATURES
from xgb_validation_stopping import train_until_validation_stops
EXPERIMENTS = {
    "joint30": dict(features=tuple(JOINT_FEATURES),feature_directory="09_joint_momentum_volume_gap_v1",version="xgb_ret_joint_unique_100_v1"),
    "momentum8": dict(features=tuple(NETWORK_FEATURES), feature_directory="05_network_individual_8_v1", version="xgb_ret_shared_network8_v3"),
    "combined15": dict(features=tuple(COMBINED_FEATURES), feature_directory="06_momentum_volume_overnight_v1", version="xgb_ret_shared_combined15_v2"),
}

def require_start_time(config, now=None):
    if not config.get("not_before_utc"):
        return
    gate=datetime.fromisoformat(config["not_before_utc"])
    if gate.tzinfo is None:
        raise ValueError("Start gate must include a timezone.")
    now=now or datetime.now(timezone.utc)
    if now<gate:
        raise RuntimeError("Model fitting is forbidden before "+gate.isoformat())


def wait_for_start(run, config):
    gate=datetime.fromisoformat(config["not_before_utc"]) if config.get("not_before_utc") else None
    while gate is not None and datetime.now(timezone.utc)<gate:
        base.save_json(run/"RUN_STATUS.json",dict(status="scheduled",completed_members=0,
            target_members=4000,workers_alive=0,not_before_utc=gate.isoformat(),
            controller_pid=os.getpid(),updated_at_utc=datetime.now(timezone.utc).isoformat()))
        time.sleep(min(20,max(0,(gate-datetime.now(timezone.utc)).total_seconds())))
    require_start_time(config)
    base.save_json(run/"TRAINING_STARTED.json",dict(started_at_utc=datetime.now(timezone.utc).isoformat(),
        not_before_utc=config.get("not_before_utc"),controller_pid=os.getpid()))


@contextmanager
def controller_lock(run):
    run=Path(run);run.mkdir(parents=True,exist_ok=True)
    with (run/"CONTROLLER.lock").open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("A controller for this run is already active.") from error
        try:yield
        finally:fcntl.flock(stream.fileno(),fcntl.LOCK_UN)


def validate_matrix(values, features):
    if tuple(features) not in [v["features"] for v in EXPERIMENTS.values()]:
        raise ValueError("Feature names differ from an explicit experiment contract.")
    if values.ndim!=2 or values.shape[1]!=len(features) or np.isinf(values).any():
        raise ValueError("Wrong feature matrix shape or infinite input.")
    for i, name in enumerate(features):
        if name!="overnight_gap" and not np.isfinite(values[:,i]).all():
            raise ValueError("Unexpected missing input: "+name)


MEMBERS=100
BATCH_SIZE=5
WORKERS=10
THREADS=1
HARD_STOP_UTC=None
BASE_SEED=20261105


def member_seeds(fold_id,member):
    if not 0<=member<MEMBERS:
        raise ValueError("Member ID must be 0..99.")
    year,quarter=fold_id.split("Q")
    sequence=np.random.SeedSequence([BASE_SEED,int(year),int(quarter),member])
    a,b=sequence.generate_state(2)
    return int(a),int(b)


def split_indices(n,seed):
    if n<20:
        raise ValueError("Training sample too small for the 80/20 split.")
    permutation=np.random.default_rng(seed).permutation(n)
    stop=int(0.8*n)
    return np.sort(permutation[:stop]),np.sort(permutation[stop:])


def member_prefix(run,fold_id,member):
    return Path(run)/"folds"/fold_id/"members"/f"member_{member:03d}"


def checkpoint_complete(run,fold_id,member):
    prefix=member_prefix(run,fold_id,member)
    metadata=prefix.with_suffix(".json")
    if not metadata.exists():
        return False
    value=json.loads(metadata.read_text())
    if value["member"]!=member or value["fold_id"]!=fold_id:
        raise ValueError("Checkpoint identity mismatch.")
    for suffix,key in [(".ubj","model_sha256"),(".npz","probabilities_sha256")]:
        path=prefix.with_suffix(suffix)
        if not path.exists() or base.fingerprint(path)!=value[key]:
            raise ValueError(f"Incomplete/corrupt checkpoint: {path}")
    return True


def average_probabilities(paths,expected_count=MEMBERS):
    if len(paths)!=expected_count or len(set(map(str,paths)))!=expected_count:
        raise ValueError("An ensemble requires exactly 100 unique members.")
    total=None
    for path in paths:
        with np.load(path,allow_pickle=False) as checkpoint:
            probabilities=checkpoint["probabilities"]
            if probabilities.ndim!=2 or probabilities.shape[1]!=10:
                raise ValueError("Probability checkpoint has the wrong class dimension.")
            if (not np.isfinite(probabilities).all() or (probabilities<0).any()
                or not np.allclose(probabilities.sum(axis=1),1,atol=1e-5)):
                raise ValueError("Invalid probability checkpoint.")
            if total is None:
                total=np.zeros(probabilities.shape,dtype=np.float64)
            if total.shape!=probabilities.shape:
                raise ValueError("Member forecast rows differ.")
            total+=probabilities
    return total/expected_count


def row_key_hash(frame):
    keys=pd.util.hash_pandas_object(frame[["security_id","trade_date"]],index=False)
    return hashlib.sha256(keys.to_numpy().tobytes()).hexdigest()


def batch_worker(run,fold_id,members):
    run=Path(run)
    config=json.loads((run/"CONFIG.json").read_text())
    require_start_time(config)
    if config["code_sha256"]!=base.fingerprint(__file__):
        raise ValueError("Training source changed after this run was pinned.")
    if config["baseline_code_sha256"]!=base.fingerprint(base.__file__):
        raise ValueError("Shared accounting source differs from the pinned run.")
    if config["stopping_code_sha256"]!=base.fingerprint(Path(__file__).with_name("xgb_validation_stopping.py")):
        raise ValueError("Stopping implementation differs from the pinned run.")
    import xgboost as xgb
    features=config["model_features"]
    fold=next(f for f in base.folds() if f["fold_id"]==fold_id)
    training,testing=base.read_fold(run,fold)
    if not training.trade_date.lt(pd.Timestamp(fold["test_start"])).all():
        raise ValueError("Outer-fold future target entered an ensemble fit.")
    x=training[features].to_numpy(dtype=np.float32)
    y=training.realized_class.to_numpy(dtype=np.float32)-1
    test_x=testing[features].to_numpy(dtype=np.float32)
    for matrix in (x,test_x):
        validate_matrix(matrix,features)
    dtest=xgb.DMatrix(test_x,feature_names=features,nthread=THREADS)
    key_hash=row_key_hash(testing)
    hard_stop=datetime.fromisoformat(config["hard_stop_utc"]).timestamp() if config["hard_stop_utc"] else None

    def guard(epoch):
        if hard_stop is not None and time.time()>=hard_stop:
            raise TimeoutError("Explicit ensemble stop reached.")

    for member in members:
        if checkpoint_complete(run,fold_id,member):
            continue
        require_start_time(config)
        if hard_stop is not None and time.time()+15>=hard_stop:
            break
        begin=time.monotonic()
        fit_started_at_utc=datetime.now(timezone.utc).isoformat()
        split_seed,booster_seed=member_seeds(fold_id,member)
        train_idx,valid_idx=split_indices(len(training),split_seed)
        dtrain=xgb.DMatrix(x[train_idx],label=y[train_idx],
                           feature_names=features,nthread=THREADS)
        dvalid=xgb.DMatrix(x[valid_idx],label=y[valid_idx],
                           feature_names=features,nthread=THREADS)
        parameters=dict(base.PARAMETERS,seed=booster_seed,nthread=THREADS)
        log={}
        model,log=train_until_validation_stops(parameters,dtrain,dvalid,
            patience=config["early_stopping_patience"],guard=guard)
        best=model.best_iteration+1
        probabilities=model.predict(dtest,iteration_range=(0,best))
        if probabilities.shape!=(len(testing),10) or not np.isfinite(probabilities).all():
            raise ValueError("Invalid member predictions.")
        prefix=member_prefix(run,fold_id,member)
        prefix.parent.mkdir(parents=True,exist_ok=True)
        model_path=prefix.with_suffix(".ubj")
        temp_model=prefix.with_suffix(".building.ubj")
        # Save exactly the early-stopped model used for probability averaging.
        trimmed=model[:best]
        trimmed.save_model(temp_model)
        os.replace(temp_model,model_path)
        probability_path=prefix.with_suffix(".npz")
        temporary=prefix.with_suffix(".npz.building")
        with temporary.open("wb") as output:
            np.savez_compressed(output,probabilities=probabilities)
        os.replace(temporary,probability_path)
        details=dict(fold_id=fold_id,member=member,split_seed=split_seed,
            fit_started_at_utc=fit_started_at_utc,fit_completed_at_utc=datetime.now(timezone.utc).isoformat(),
            booster_seed=booster_seed,inner_train_rows=len(train_idx),
            inner_validation_rows=len(valid_idx),training_pool_rows=len(training),
            training_first_outcome=training.trade_date.min(),
            training_last_outcome=training.trade_date.max(),
            split_train_index_sha256=hashlib.sha256(train_idx.tobytes()).hexdigest(),
            split_validation_index_sha256=hashlib.sha256(valid_idx.tobytes()).hexdigest(),
            best_rounds=best,rounds_examined=len(log["random20"]["mlogloss"]),
            best_inner_logloss=min(log["random20"]["mlogloss"]),
            prediction_rows=len(testing),prediction_row_key_sha256=key_hash,
            elapsed_seconds=time.monotonic()-begin,peak_worker_rss_gib=base.peak_rss_gib(),
            model_sha256=base.fingerprint(model_path),
            probabilities_sha256=base.fingerprint(probability_path))
        # JSON is written last: it commits a complete, auditable member.
        base.save_json(prefix.with_suffix(".json"),details)
        print(f"{fold_id} member {member+1}/100: {best} rounds, "
              f"{details['elapsed_seconds']:.2f}s",flush=True)
        del model,trimmed,dtrain,dvalid,probabilities
        gc.collect()


def finalize_fold(run,fold):
    destination=run/"folds"/fold["fold_id"]
    details=[]
    paths=[]
    for member in range(MEMBERS):
        if not checkpoint_complete(run,fold["fold_id"],member):
            raise ValueError("Cannot finalize a quarter with fewer than 100 members.")
        prefix=member_prefix(run,fold["fold_id"],member)
        details.append(json.loads(prefix.with_suffix(".json").read_text()))
        paths.append(prefix.with_suffix(".npz"))
    training,testing=base.read_fold(run,fold)
    expected_hash=row_key_hash(testing)
    if any(d["prediction_row_key_sha256"]!=expected_hash for d in details):
        raise ValueError("Ensemble members disagree on forecast row identities/order.")
    probabilities=average_probabilities(paths)
    mu,daily,complete=base.class_payoffs(training, expected_dates=session_calendar(fold["train_start"], (pd.Timestamp(fold["test_start"])-pd.Timedelta(days=1)).date().isoformat()).signal_date)
    columns=["security_id","ticker","signal_date","signal_cutoff_utc","trade_date",
             "outcome_status","future_oc_return","realized_class"]
    predictions=testing[columns].copy()
    for k in range(10):
        predictions[f"p_class_{k+1}"]=probabilities[:,k]
    predictions["ret_score"]=probabilities@mu
    predictions["fold_id"]=fold["fold_id"]
    predictions["ensemble_members"]=MEMBERS
    predictions.to_parquet(destination/"predictions.parquet",index=False)
    daily.assign(used_for_calibration=complete).to_parquet(destination/"daily_class_payoffs.parquet")
    pd.DataFrame(details).to_parquet(destination/"members_summary.parquet",index=False)
    info=dict(**fold,ensemble_members=MEMBERS,
        training_rows=len(training),training_sessions=training.trade_date.nunique(),
        training_first_outcome=training.trade_date.min(),training_last_outcome=training.trade_date.max(),
        prediction_rows=len(testing),payoff_complete_days=int(complete.sum()),
        payoff_incomplete_days=int((~complete).sum()),payoff_vector_class_1_to_10=mu.tolist(),
        mean_best_rounds=float(np.mean([d["best_rounds"] for d in details])),
        min_best_rounds=min(d["best_rounds"] for d in details),
        max_best_rounds=max(d["best_rounds"] for d in details),
        total_member_seconds=sum(d["elapsed_seconds"] for d in details),
        peak_worker_rss_gib=max(d["peak_worker_rss_gib"] for d in details),
        prediction_row_key_sha256=expected_hash)
    base.save_json(destination/"FOLD.json",info)


def inventory(run):
    counts={}
    for fold in base.folds():
        folder=run/"folds"/fold["fold_id"]/"members"
        ids={int(p.stem.split("_")[1]) for p in folder.glob("member_*.json")}
        if not ids.issubset(set(range(MEMBERS))):
            raise ValueError("Unexpected ensemble member ID.")
        counts[fold["fold_id"]]=len(ids)
    return counts


def start_task(run,fold_id,members,env):
    label=f"{fold_id}_{members[0]:03d}_{members[-1]:03d}"
    log=(run/"logs"/f"{label}.log").open("w")
    frozen=Path(run)/"source_snapshot/scripts"/Path(__file__).name
    if not frozen.exists():
        raise ValueError("Worker source snapshot is missing.")
    command=[sys.executable,str(frozen),"--mode","batch",
             "--run-dir",str(run),"--fold-id",fold_id,
             "--members",",".join(map(str,members))]
    process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,
                             env=env,start_new_session=True)
    return dict(process=process,log=log,label=label,fold_id=fold_id,members=members)


def run_tasks(run,tasks,config):
    """Run bounded batches; drain and clean owned workers on every exit path."""
    env=os.environ.copy()
    env.update(OMP_NUM_THREADS=str(THREADS),OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",
               VECLIB_MAXIMUM_THREADS="1",NUMEXPR_NUM_THREADS="1")
    pending=list(tasks)
    active=[]
    peak=0.0
    start=time.monotonic()
    last_report=0
    hard_stop=datetime.fromisoformat(config["hard_stop_utc"]).timestamp() if config["hard_stop_utc"] else None
    try:
        while pending or active:
            if hard_stop is not None and time.time()>=hard_stop-10:
                return dict(stopped_for_deadline=True,wall_seconds=time.monotonic()-start,
                            peak_combined_rss_gib=peak)
            while (pending and len(active)<config["workers"]
                   and psutil.virtual_memory().available>=config["minimum_available_memory_gib"]*1024**3):
                task=pending.pop(0)
                active.append(start_task(run,*task,env))
            rss=psutil.Process().memory_info().rss
            for task in active:
                try:
                    rss+=psutil.Process(task["process"].pid).memory_info().rss
                except psutil.NoSuchProcess:
                    pass
            peak=max(peak,rss/1024**3)
            # No fixed RSS ceiling: available-memory headroom controls new launches.
            finished=[]
            for task in active:
                code=task["process"].poll()
                if code is not None:
                    task["log"].close()
                    if code:
                        raise RuntimeError(f"Worker {task['label']} failed; inspect its private log.")
                    finished.append(task)
            for task in finished:
                active.remove(task)
            if time.monotonic()-last_report>=20:
                counts=inventory(run)
                complete=sum(counts.values())
                status=dict(status="running",completed_members=complete,target_members=4000,
                    complete_member_sets=sum(n==MEMBERS for n in counts.values()),
                    completed_by_fold=counts,workers_alive=len(active),
                    peak_combined_rss_gib=peak,updated_at_utc=datetime.now(timezone.utc).isoformat())
                base.save_json(run/"PROGRESS.json",status)
                print(f"PROGRESS {complete}/4000 members; "
                      f"{status['complete_member_sets']}/40 complete quarters; "
                      f"{len(active)} workers; {rss/1024**3:.2f} GiB RSS",flush=True)
                last_report=time.monotonic()
            time.sleep(0.25)
        return dict(stopped_for_deadline=False,wall_seconds=time.monotonic()-start,
                    peak_combined_rss_gib=peak)
    finally:
        for task in active:
            base.terminate_owned(task["process"])
            task["log"].close()


def prepare_combined_panel(project,run):
    """Pin the completed 15-input panel; keep labels and forecast rows intact."""
    features=project/"data/features/06_momentum_volume_overnight_v1"
    original=project/"data/models/05_xgb_ret_network_features100_uncapped_v1"
    manifest=json.loads((features/"MANIFEST.json").read_text())
    if (manifest["model_features"]!=COMBINED_FEATURES or len(COMBINED_FEATURES)!=15
        or not manifest["summary"]["forecast_cohort_identical"]
        or not manifest["summary"]["original_eight_inputs_labels_outcomes_identical"]):
        raise ValueError("Combined feature release failed its original-cohort checks.")
    source=features/"supervised_panel.parquet"
    source_hash=base.fingerprint(source)
    if source_hash!=manifest["output_sha256"]["supervised_panel.parquet"]:
        raise ValueError("Combined panel differs from the completed feature manifest.")
    if (run/"PANEL.json").exists():
        prior=json.loads((run/"PANEL.json").read_text())
        if (prior["feature_manifest_sha256"]!=base.fingerprint(features/"MANIFEST.json")
            or prior["combined_panel_sha256"]!=source_hash):
            raise ValueError("Run panel uses another combined feature release.")
        return
    shutil.copyfile(source,run/"supervised_panel.parquet")
    for name in ("wfo_calendar.parquet","label_coverage.parquet","outcome_status_counts.parquet"):
        shutil.copyfile(original/name,run/name)
    info=json.loads((original/"PANEL.json").read_text())
    info.update(feature_manifest_sha256=base.fingerprint(features/"MANIFEST.json"),
        combined_panel_sha256=source_hash,rows=manifest["summary"]["panel_rows"],
        model_features=COMBINED_FEATURES,forecast_cohort_identical=True,
        information_cutoff="Fourteen previous-close inputs plus current opening gap; zero-latency same-price entry is optimistic research.",
        missing_policy="Opening gap uses native NaN handling; no row or unknown outcome is removed.",
        one_minute_data_used=False,
        labels_and_outcomes="Copied unchanged from the completed eight-input model panel.")
    base.save_json(run/"PANEL.json",info)
    snapshot_source(run)


def prepare_network_panel(project,run):
    """Replace inputs only; freeze every saved forecast identity, label and outcome."""
    original=project/"data/models/04_xgb_ret_ensemble100_v1"
    features=project/"data/features/05_network_individual_8_v1"
    manifest=json.loads((features/"MANIFEST.json").read_text())
    if not manifest["summary"]["cohort_identical"] or manifest["model_features"]!=NETWORK_FEATURES:
        raise ValueError("The new release does not preserve the original forecast cohort.")
    if (run/"PANEL.json").exists():
        prior=json.loads((run/"PANEL.json").read_text())
        if prior["feature_manifest_sha256"]!=base.fingerprint(features/"MANIFEST.json"):
            raise ValueError("Panel belongs to another feature release.")
        return
    con=duckdb.connect()
    con.execute("SET threads=2")
    con.execute("SET temp_directory=?",[str(run/".duckdb_tmp")])
    con.read_parquet(str(original/"supervised_panel.parquet")).create_view("original")
    con.read_parquet(str(features/"features.parquet")).create_view("features")
    cols=con.execute("DESCRIBE original").df().column_name.tolist()
    original_model=json.loads((original/"CONFIG.json").read_text())["model_features"]
    keep=[x for x in cols if x not in original_model]
    con.execute("CREATE TEMP TABLE new_panel AS SELECT "+",".join("o."+x for x in keep)+
        ","+",".join("f."+x for x in NETWORK_FEATURES)+
        " FROM original o LEFT JOIN features f USING(security_id,signal_date)")
    counts=con.execute("SELECT (SELECT count(*) FROM original),(SELECT count(*) FROM new_panel)").fetchone()
    bad=con.execute("SELECT count(*) FROM new_panel WHERE NOT ("+
        " AND ".join(f"coalesce(isfinite({x}),false)" for x in NETWORK_FEATURES)+")").fetchone()[0]
    if counts[0]!=counts[1] or bad:
        raise ValueError("Feature-only panel changed row identities or introduced undefined inputs.")
    con.execute("COPY (SELECT * FROM new_panel ORDER BY trade_date,security_id) TO ? (FORMAT PARQUET,COMPRESSION ZSTD)",
                [str(run/"supervised_panel.parquet")])
    con.close()
    for name in ("wfo_calendar.parquet","label_coverage.parquet","outcome_status_counts.parquet"):
        shutil.copyfile(original/name,run/name)
    info=json.loads((original/"PANEL.json").read_text())
    info.update(feature_manifest_sha256=base.fingerprint(features/"MANIFEST.json"),
        original_panel_sha256=base.fingerprint(original/"supervised_panel.parquet"),
        rows=counts[0],model_features=NETWORK_FEATURES,forecast_cohort_identical=True,
        labels_and_outcomes="Copied unchanged by identity from the completed original ensemble panel.")
    base.save_json(run/"PANEL.json",info)
    snapshot_source(run)


def prepare_joint_panel(project,run):
    release=project/"data/features/09_joint_momentum_volume_gap_v1"
    manifest=json.loads((release/"MANIFEST.json").read_text())
    source=release/"supervised_panel.parquet"
    if (manifest["model_features"]!=JOINT_FEATURES or len(JOINT_FEATURES)!=30
        or not manifest["summary"]["cohort_labels_outcomes_identical"]
        or manifest["summary"]["exact_duplicate_features_removed"]):
        raise ValueError("Joint feature contract/duplicate audit differs from the pinned experiment.")
    source_hash=base.fingerprint(source)
    if source_hash!=manifest["output_sha256"]:
        raise ValueError("Joint source panel changed.")
    if (run/"PANEL.json").exists():
        old=json.loads((run/"PANEL.json").read_text())
        if old["joint_panel_sha256"]!=source_hash or base.fingerprint(run/"supervised_panel.parquet")!=source_hash:
            raise ValueError("Prepared joint panel differs from its immutable source.")
        return
    shutil.copyfile(source,run/"supervised_panel.parquet")
    original=project/"data/models/07_xgb_ret_volume_overnight100_v1"
    for name in ("wfo_calendar.parquet","label_coverage.parquet","outcome_status_counts.parquet"):
        shutil.copyfile(original/name,run/name)
    info=json.loads((original/"PANEL.json").read_text())
    info.update(feature_manifest_sha256=base.fingerprint(release/"MANIFEST.json"),
        joint_panel_sha256=source_hash,rows=manifest["summary"]["rows"],model_features=JOINT_FEATURES,
        forecast_cohort_identical=True,labels_and_outcomes="Unchanged from completed combined15 experiment.",
        deduplication=manifest["duplication_policy"])
    base.save_json(run/"PANEL.json",info)
    snapshot_source(run)


def snapshot_source(run):
    destination=run/"source_snapshot/scripts"
    destination.mkdir(parents=True,exist_ok=True)
    for path in Path(__file__).parent.glob("*.py"):
        if not path.stem.isidentifier():
            continue
        shutil.copyfile(path,destination/path.name)




def full_run(project,run,hard_stop=HARD_STOP_UTC,workers=WORKERS,experiment="combined15",not_before=None,minimum_available_memory_gib=None):
    spec=EXPERIMENTS[experiment]
    features=list(spec["features"])
    if minimum_available_memory_gib is None:
        minimum_available_memory_gib=0.0 if experiment=="joint30" else 2.0
    if minimum_available_memory_gib<0:
        raise ValueError("Available-memory threshold must be nonnegative.")
    if not_before is not None and datetime.fromisoformat(not_before).tzinfo is None:
        raise ValueError("Start timestamp must include a timezone.")
    if workers<1:
        raise ValueError("Positive worker count required.")
    project=Path(project).resolve()
    run=Path(run).resolve()
    if not run.is_relative_to((project/"data").resolve()):
        raise ValueError("All licensed outputs must remain inside private project data/.")
    run.mkdir(parents=True,exist_ok=True)
    (run/"logs").mkdir(exist_ok=True)
    config=dict(version=spec["version"],experiment=experiment,members_per_quarter=MEMBERS,total_quarters=40,
        total_member_models=4000,train_months=12,test_months=3,
        evaluation_start=base.EVALUATION_START,evaluation_end_exclusive=base.EVALUATION_END,
        inner_split="Fresh independent random 80/20 stock-row split inside the past 12 months for each member; no outer test outcomes admitted.",
        member_prediction="Early-stopped 80% fit at its best iteration; no full-window refit.",
        averaging="Arithmetic mean of 100 aligned ten-class probability vectors; then RET, then rank top/bottom N.",
        parameters=dict(base.PARAMETERS,nthread=THREADS),max_rounds=None,early_stopping_patience=30,
        members_per_batch=BATCH_SIZE,workers=workers,threads_per_worker=THREADS,
        memory_limit_gib=None,minimum_available_memory_gib=minimum_available_memory_gib,hard_stop_utc=hard_stop,not_before_utc=not_before,
        model_features=features,n_per_side=[2,3],gross_exposure=1.0,
        information_cutoff=("Previous completed close" if experiment=="momentum8" else "Current opening event; optimistic same-price fill"),
        missing_policy="Finite historical inputs; native missing handling for opening gap only.",
        payoff_policy="Common full 12-month historical true-class daily means, frozen per outer quarter.",
        feature_manifest_sha256=base.fingerprint(project/"data/features"/spec["feature_directory"]/"MANIFEST.json"),
        stopping_code_sha256=base.fingerprint(Path(__file__).with_name("xgb_validation_stopping.py")),
        code_sha256=base.fingerprint(__file__),baseline_code_sha256=base.fingerprint(base.__file__))
    if (run/"CONFIG.json").exists():
        prior=json.loads((run/"CONFIG.json").read_text())
        # A new resource deadline is allowed for a resumed, otherwise identical run.
        previous=dict(prior);current=dict(config)
        previous.pop("hard_stop_utc");current.pop("hard_stop_utc")
        if previous!=current:
            raise FileExistsError("Different ensemble configuration exists; use a new run directory.")
    base.save_json(run/"CONFIG.json",config)
    if (run/"RESULTS.json").exists():
        print("Reusing completed 100-member ensemble.",flush=True)
        return
    if hard_stop is not None and datetime.fromisoformat(hard_stop).timestamp()<=time.time():
        raise TimeoutError("Choose a future hard-stop time to resume saved checkpoints.")
    started=time.monotonic()
    original_handler=signal.getsignal(signal.SIGTERM)
    def terminate_run(signum,frame):
        raise KeyboardInterrupt("Ensemble termination requested.")
    signal.signal(signal.SIGTERM,terminate_run)
    records=[]
    try:
        {"combined15":prepare_combined_panel,"momentum8":prepare_network_panel,"joint30":prepare_joint_panel}[experiment](project,run)
        wait_for_start(run,config)
        started=time.monotonic()
        if not (run/"BENCHMARK.json").exists():
            # Benchmark fresh members under the actual worker concurrency.
            before=set((run/"folds").glob("*/members/member_*.json"))
            benchmark_tasks=[]
            candidates=base.folds()[::5]+base.folds()
            seen=set()
            for fold in candidates:
                fid=fold["fold_id"]
                if fid in seen:
                    continue
                seen.add(fid)
                missing=[m for m in range(MEMBERS) if not checkpoint_complete(run,fid,m)]
                if len(missing)>=2:
                    benchmark_tasks.append((fid,missing[-2:]))
                if len(benchmark_tasks)==config["workers"]:
                    break
            record=run_tasks(run,benchmark_tasks,config)
            records.append(record)
            new_paths=set((run/"folds").glob("*/members/member_*.json"))-before
            samples=[json.loads(p.read_text()) for p in new_paths]
            if len(samples)<2*len(benchmark_tasks):
                raise TimeoutError("Benchmark stopped at deadline; saved members remain resumable.")
            wall_per_member=record["wall_seconds"]/len(samples)
            estimate=dict(benchmark_members=len(samples),benchmark_wall_seconds=record["wall_seconds"],
                median_member_fit_seconds=float(np.median([s["elapsed_seconds"] for s in samples])),
                estimated_total_wall_seconds=wall_per_member*4000,
                conservative_total_wall_seconds=wall_per_member*4000*1.5,
                measured_benchmark_peak_rss_gib=record["peak_combined_rss_gib"],
                deadline_utc=hard_stop,can_checkpoint_and_resume=True)
            base.save_json(run/"BENCHMARK.json",estimate)
            print("ENSEMBLE_RUNTIME_ESTIMATE",json.dumps(estimate),flush=True)
        else:
            estimate=json.loads((run/"BENCHMARK.json").read_text())
        # Complete each quarter's full 100-member set, before moving to the next.
        tasks=[]
        for fold in base.folds():
            missing=[m for m in range(MEMBERS) if not checkpoint_complete(run,fold["fold_id"],m)]
            for i in range(0,len(missing),BATCH_SIZE):
                tasks.append((fold["fold_id"],missing[i:i+BATCH_SIZE]))
        records.append(run_tasks(run,tasks,config))
        counts=inventory(run)
        complete=sum(counts.values())
        if complete!=4000:
            base.save_json(run/"RUN_STATUS.json",dict(status="checkpointed_at_deadline",
                completed_members=complete,target_members=4000,completed_by_fold=counts,
                workers_alive=0,elapsed_seconds=time.monotonic()-started,
                peak_combined_rss_gib=max(r["peak_combined_rss_gib"] for r in records),
                resume_command="Rerun with the same run directory; --hard-stop-utc is optional and has no default cutoff."))
            print("CHECKPOINTED",(run/"RUN_STATUS.json").read_text(),flush=True)
            return
        for fold in base.folds():
            if not (run/"folds"/fold["fold_id"]/"FOLD.json").exists():
                finalize_fold(run,fold)
            print("FINALIZED",fold["fold_id"],"100/100",flush=True)
        base.summarize(run)
        results=json.loads((run/"RESULTS.json").read_text())
        results["ensemble_members_per_quarter"]=MEMBERS
        results["total_member_models"]=4000
        results["limitations"]=[s for s in results["limitations"] if "Single-seed" not in s]
        results["limitations"].append(("Eight prior-close momentum inputs." if experiment=="momentum8" else "Different momentum normalizations, volume and current opening-gap inputs. Joint30 adds the original fifteen features without duplicate-valued columns. Same-opening-price signal and fill are optimistic research; no one-minute data are used."))
        base.save_json(run/"RESULTS.json",results)
        base.save_json(run/"RUN_STATUS.json",dict(status="complete",completed_members=4000,
            target_members=4000,ensemble_members_per_quarter=100,workers_alive=0,
            elapsed_seconds=time.monotonic()-started,
            peak_combined_rss_gib=max(base.peak_rss_gib(),max(r["peak_combined_rss_gib"] for r in records)),
            completed_at_utc=datetime.now(timezone.utc).isoformat()))
        base.save_json(run/"PROGRESS.json",json.loads((run/"RUN_STATUS.json").read_text()))
        print("ENSEMBLE COMPLETE",(run/"RUN_STATUS.json").read_text(),flush=True)
    except BaseException as error:
        counts=inventory(run)
        base.save_json(run/"RUN_STATUS.json",dict(status="stopped_resumable",reason=str(error),
            completed_members=sum(counts.values()),target_members=4000,
            completed_by_fold=counts,workers_alive=0,elapsed_seconds=time.monotonic()-started))
        raise
    finally:
        signal.signal(signal.SIGTERM,original_handler)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment",choices=sorted(EXPERIMENTS),default="combined15")
    parser.add_argument("--project",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--run-dir",type=Path,required=True)
    parser.add_argument("--mode",choices=["run","batch"],default="run")
    parser.add_argument("--fold-id",default="2017Q1")
    parser.add_argument("--members",default="0,1")
    parser.add_argument("--workers",type=int,default=WORKERS)
    parser.add_argument("--not-before-utc",help="Timezone-aware earliest permitted model-fitting time.")
    parser.add_argument("--minimum-available-memory-gib",type=float,default=None,help="0 disables the memory-headroom launch restriction; this is not a model RAM ceiling.")
    parser.add_argument("--hard-stop-utc",default=HARD_STOP_UTC)
    args=parser.parse_args()
    if args.mode=="batch":
        batch_worker(args.run_dir,args.fold_id,[int(m) for m in args.members.split(",")])
    else:
        if not args.run_dir.resolve().is_relative_to((args.project/"data").resolve()):
            raise ValueError("Run must stay in private project data/.")
        with controller_lock(args.run_dir):
            full_run(args.project,args.run_dir,args.hard_stop_utc,args.workers,args.experiment,
                     args.not_before_utc,args.minimum_available_memory_gib)
            if args.experiment=="joint30" and (args.run_dir/"RESULTS.json").exists():
                subprocess.run([sys.executable,"-B",str(Path(__file__).with_name("run_notebook.py")),
                    "09","--project",str(args.project)],check=True)
