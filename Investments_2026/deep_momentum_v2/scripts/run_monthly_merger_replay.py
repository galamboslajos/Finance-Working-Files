#!/usr/bin/env python3
"""Explicit zero-fit realized-monthly-return repair controller."""
from pathlib import Path
import argparse,fcntl,importlib.util,json,os,resource,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1]
PHASES=('amendments','verify_amendments','backtest','verify_backtest')


def read(p):return json.loads(Path(p).read_text())


def write(path,value,replace=False):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);data=json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n'
    if replace:
        tmp=p.with_suffix('.writing');tmp.write_text(data);os.replace(tmp,p)
    else:
        with p.open('x') as f:f.write(data)


def worker(a):
    from dmv2.monthly_merger_replay import validate,read,check,prepare_amendments,backtest
    p=validate(a.plan);cfg=read(check(p['protocol_pin']));start=time.monotonic()
    def guard():
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
        if rss>cfg['budgets']['peak_rss_bytes'] or time.monotonic()-start>cfg['budgets']['per_phase_seconds']:raise RuntimeError('Replay phase resource stop')
    if a.phase=='amendments':result=prepare_amendments(a.plan,guard)
    elif a.phase=='backtest':result=backtest(a.plan,guard)
    else:
        spec=importlib.util.spec_from_file_location('_independent_monthly_merger',ROOT/'scripts/verify_monthly_merger_replay.py')
        m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
        result=getattr(m,a.phase)(a.plan,guard)
    guard();write(a.result,dict(status='COMPLETE_PHASE',phase=a.phase,result=result))


def run(plan_path,resume=False):
    from dmv2.monthly_merger_replay import validate,pin,check
    p=validate(plan_path);cfg=read(check(p['protocol_pin']));root=Path(p['output_directory']);root.mkdir(parents=True,exist_ok=True)
    lock=(root/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    lp=root/'ledger.json'
    if lp.exists():
        ledger=read(lp)
        if not resume or ledger['status']!='PAUSED' or ledger['phase'] is not None:raise RuntimeError('Preserve consumed replay; only clean pauses resume')
        if ledger['plan_pin']!=pin(plan_path):raise RuntimeError('Changed replay plan')
        for item in ledger['completed']:check(item['receipt'])
        if (root/'PAUSE.request').exists():(root/'PAUSE.request').rename(root/('PAUSE.ack.'+str(time.time_ns())))
    else:
        if resume:raise RuntimeError('No replay to resume')
        ledger=dict(status='READY',phase=None,plan_pin=pin(plan_path),fits_started=0,fits_completed=0,completed=[],active_seconds=0.,peak_worker_rss_bytes=0)
        write(lp,ledger);write(root/'launch.json',dict(plan_pin=pin(plan_path),maximum_fits=0,existing_models=16,realized_return_only=True,python=sys.executable))
    begin=time.monotonic();base=ledger['active_seconds'];env=dict(os.environ,PYTHONPATH=str(ROOT/'src'),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    def update():ledger['active_seconds']=base+time.monotonic()-begin;write(lp,ledger,replace=True)
    try:
        for i,phase in enumerate(PHASES):
            if i<len(ledger['completed']):
                if ledger['completed'][i]['key']!=phase:raise RuntimeError('Changed replay schedule')
                continue
            if (root/'PAUSE.request').exists():ledger.update(status='PAUSED',phase=None);update();return
            ledger.update(status='RUNNING',phase=phase);update();result=root/'phases'/(phase+'.json')
            if result.exists():raise RuntimeError('Consumed phase preserved')
            log=root/'logs'/(phase+'.log');log.parent.mkdir(parents=True,exist_ok=True)
            with log.open('x') as f:
                child=subprocess.Popen([sys.executable,'-B',__file__,'worker','--plan',str(plan_path),'--phase',phase,'--result',str(result)],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
                started=time.monotonic()
                while child.poll() is None:
                    val=subprocess.run(['/bin/ps','-o','rss=','-p',str(child.pid)],capture_output=True,text=True).stdout.strip();rss=int(val or 0)*1024
                    ledger['peak_worker_rss_bytes']=max(rss,ledger['peak_worker_rss_bytes'])
                    write(root/'resource_checkpoint.json',dict(phase=phase,pid=child.pid,rss_bytes=rss,peak_rss_bytes=ledger['peak_worker_rss_bytes'],new_fits=0),replace=True)
                    if rss>cfg['budgets']['peak_rss_bytes'] or time.monotonic()-started>cfg['budgets']['per_phase_seconds'] or base+time.monotonic()-begin>cfg['budgets']['total_active_seconds'] or sum(x.stat().st_size for x in root.rglob('*') if x.is_file())>cfg['budgets']['output_bytes']:
                        child.terminate()
                        try:child.wait(timeout=5)
                        except subprocess.TimeoutExpired:child.kill();child.wait()
                        raise RuntimeError('Resource stop; preserve consumed phase')
                    time.sleep(.5)
            if child.returncode!=0:raise RuntimeError('Replay/check failed; see '+str(log))
            check(read(result)['result']);ledger['completed'].append(dict(key=phase,receipt=pin(result)));ledger['phase']=None;update()
            print(json.dumps(dict(completed_phase=phase,new_fits=0,active_seconds=ledger['active_seconds'])),flush=True)
        ledger.update(status='COMPLETE_VERIFIED',phase=None);update()
    except BaseException as exc:ledger.update(status='FAILED_PRESERVED',error=str(exc));update();raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('plan','run','pause','resume','worker'));parser.add_argument('--plan',required=True)
    parser.add_argument('--data-root');parser.add_argument('--protocol');parser.add_argument('--output');parser.add_argument('--phase',choices=PHASES);parser.add_argument('--result');a=parser.parse_args()
    if a.action=='plan':
        from dmv2.monthly_merger_replay import seal
        p=seal(ROOT,a.data_root,a.protocol,a.output,a.plan);print(json.dumps(dict(status=p['status'],quote_points=len(p['quote_points']),new_fits=0,cached_file_reservation_bytes=p['cached_file_reservation_bytes'])))
    elif a.action=='worker':worker(a)
    elif a.action=='pause':write(Path(read(a.plan)['output_directory'])/'PAUSE.request',dict(requested_ns=time.time_ns()));print('Pause requested at clean phase boundary.')
    else:run(a.plan,resume=a.action=='resume')


if __name__=='__main__':main()
