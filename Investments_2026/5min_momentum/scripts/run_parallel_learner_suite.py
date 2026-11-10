"""Resume pinned experiment workers with balanced active CPU allocation.

This controller-only revision changes task scheduling. Frozen feature, model,
split, tuning and accounting sources/configurations remain byte-identical.
"""
from collections import Counter
from pathlib import Path
import argparse,importlib,json,signal,subprocess,sys,time

REVISION='balanced_v2'


def select_variant(queues,active,variants,turn):
    """Balance occupied slots, not dispatched job counts with unequal duration."""
    counts=Counter(task['job']['variant'] for task in active)
    order=[variants[(turn+i)%len(variants)] for i in range(len(variants))]
    choices=[variant for variant in order if queues[variant]]
    if not choices:raise ValueError('No queued tasks.')
    return min(choices,key=lambda variant:counts[variant])


def run(suite,workers):
    suite=Path(suite).resolve()
    frozen=suite/'source_snapshot'
    sys.path.insert(0,str(frozen/'scripts'))
    protocol=importlib.import_module('learner_experiments')
    base,engine=protocol.base,protocol.engine
    with engine.controller_lock(suite):
        # This is an exact immutable-config comparison, including all model seeds,
        # source hashes, grid settings, feature hashes and original worker count.
        protocol.prepare_suite(frozen,suite,workers)
        if (suite/'COMPLETE.json').exists():return
        before=json.loads((suite/'RUN_STATUS.json').read_text())
        if before['status']!='stopped_resumable' or before['workers_alive']!=0:
            raise RuntimeError('Only an explicitly checkpoint-stopped suite may resume.')
        started=protocol.utcnow()
        base.save_json(suite/'controller_revisions'/REVISION/'SESSION_STARTED.json',dict(
            started_at_utc=started,controller_pid=__import__('os').getpid(),previous_status=before,
            change='Balance active workers between queued streams; no research settings changed.'))
        queues=protocol.task_queues(suite)
        active=[];finalizing=set();peak=before.get('peak_combined_rss_gib',0.);last=0.;turn=0
        previous=signal.getsignal(signal.SIGTERM)
        def stop(signum,frame):raise KeyboardInterrupt('Checkpoint stop requested.')
        signal.signal(signal.SIGTERM,stop)
        def progress():
            status=protocol.publish_progress(suite,active,started,peak)
            status.update(controller_revision=REVISION,
                scheduler='Fewest active workers among nonempty queues; ties alternate; free slots reusable.')
            base.save_json(suite/'RUN_STATUS.json',status);base.save_json(suite/'PROGRESS.json',status)
            return status
        try:
            while True:
                for task in list(active):
                    code=task['process'].poll()
                    if code is None:continue
                    task['log'].close();active.remove(task)
                    job=task['job']
                    if code:raise RuntimeError(f"{job['variant']} {task['label']} failed; inspect its log.")
                    if job['kind']=='tune':
                        fid=job['fold_id'];leaf=suite/job['variant']
                        missing=[m for m in range(100) if not engine.checkpoint_complete(leaf,fid,m)]
                        tasks=[dict(variant='ret_tuned',kind='variant_batch',fold_id=fid,members=missing[i:i+5])
                            for i in range(0,len(missing),5)]
                        queues['ret_tuned'].extendleft(reversed(tasks))
                for variant in protocol.VARIANTS:
                    leaf=suite/variant
                    if variant not in finalizing and not (leaf/'RESULTS.json').exists() and sum(engine.inventory(leaf).values())==4000:
                        if not any(t['job']['variant']==variant for t in active):
                            queues[variant].appendleft(dict(variant=variant,kind='variant_finalize',fold_id='2017Q1',members=[]))
                            finalizing.add(variant)
                while len(active)<workers and any(queues.values()):
                    variant=select_variant(queues,active,protocol.VARIANTS,turn)
                    turn=(protocol.VARIANTS.index(variant)+1)%len(protocol.VARIANTS)
                    active.append(protocol.start_job(suite,queues[variant].popleft()))
                rss=protocol.psutil.Process().memory_info().rss
                for task in active:
                    try:rss+=protocol.psutil.Process(task['process'].pid).memory_info().rss
                    except protocol.psutil.NoSuchProcess:pass
                peak=max(peak,rss/1024**3)
                if time.monotonic()-last>=20:
                    progress();last=time.monotonic()
                if not active and not any(queues.values()):
                    if not all((suite/v/'RESULTS.json').exists() for v in protocol.VARIANTS):
                        raise RuntimeError('Tasks drained before both experiments completed.')
                    break
                time.sleep(.25)
            status=progress();status['status']='reporting';base.save_json(suite/'RUN_STATUS.json',status)
            subprocess.run([sys.executable,'-B',str(frozen/'scripts/run_notebook.py'),'11','--project',str(frozen)],check=True)
            status.update(status='complete',completed_at_utc=protocol.utcnow(),workers_alive=0)
            base.save_json(suite/'RUN_STATUS.json',status);base.save_json(suite/'PROGRESS.json',status)
            base.save_json(suite/'COMPLETE.json',dict(completed_at_utc=protocol.utcnow(),member_models=8000,
                tuning_fits=2880,report='11_ret_tuning_and_ranker',controller_revision=REVISION))
        except BaseException as error:
            for task in active:base.terminate_owned(task['process']);task['log'].close()
            active.clear()
            status=progress();status.update(status='stopped_resumable',reason=str(error))
            for variant in protocol.VARIANTS:
                leaf=status['variants'][variant]
                if leaf['status']!='complete':
                    leaf.update(status='stopped_resumable',reason=str(error),workers_alive=0)
                    base.save_json(suite/variant/'RUN_STATUS.json',leaf);base.save_json(suite/variant/'PROGRESS.json',leaf)
            base.save_json(suite/'RUN_STATUS.json',status);base.save_json(suite/'PROGRESS.json',status)
            raise
        finally:signal.signal(signal.SIGTERM,previous)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--workers',type=int,required=True)
    args=parser.parse_args()
    run(args.run_dir,args.workers)
