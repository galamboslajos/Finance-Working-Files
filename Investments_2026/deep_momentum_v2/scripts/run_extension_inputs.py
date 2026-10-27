#!/usr/bin/env python3
"""Explicit zero-fit extension input audit controller. Notebook import never calls it."""
from pathlib import Path
import argparse,fcntl,json,os,resource,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1]


def read(p):return json.loads(Path(p).read_text())

def write(p,v,replace=False):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);data=json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n'
    if replace:
        t=p.with_suffix('.writing');t.write_text(data);os.replace(t,p)
    else:
        with p.open('x') as f:f.write(data)

def schedule():
    return [('prepare',None,None,None),('verify_data',None,None,None)]

def worker(args):
    from dmv2.extension_inputs import validate,read,check,pin
    from dmv2.paper15_baseline import need
    plan=validate(args.plan);cfg=read(check(plan['protocol_pin']));root=Path(plan['output_directory']);start=time.monotonic()
    def guard():
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
        need(rss<=cfg['budgets']['peak_rss_bytes'] and time.monotonic()-start<=cfg['budgets']['phase_seconds'],'Phase resource ceiling')
    if args.phase=='prepare':
        from dmv2.extension_inputs import prepare
        receipt=prepare(args.plan)
    else:
        import importlib.util
        spec=importlib.util.spec_from_file_location('_independent_extension',ROOT/'scripts/verify_extension_inputs.py')
        m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
        receipt=m.verify_data(args.plan)
    guard();write(args.result,{'status':'COMPLETE_PHASE','result':receipt,'phase':args.phase})

def run(plan_path,resume=False):
    from dmv2.extension_inputs import validate,check,pin
    from dmv2.paper15_baseline import need
    plan=validate(plan_path);root=Path(plan['output_directory']);cfg=read(check(plan['protocol_pin']));root.mkdir(parents=True,exist_ok=True)
    lock=(root/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    lp=root/'ledger.json'
    if lp.exists():
        ledger=read(lp);need(resume and ledger['status']=='PAUSED' and ledger['phase'] is None,'Only clean checkpoint resumes; consumed failures preserved')
        need(ledger['plan_pin']==pin(plan_path),'Changed run plan')
        for c in ledger['completed']:check(c['receipt'])
        marker=root/'PAUSE.request'
        if marker.exists():marker.rename(root/('PAUSE.ack.'+str(time.time_ns())+'.json'))
    else:
        need(not resume,'Cannot resume an absent run');ledger={'status':'READY','phase':None,'plan_pin':pin(plan_path),'fits_started':0,'fits_completed':0,'completed':[],'active_seconds':plan.get('inherited_active_seconds',0),'peak_worker_rss_bytes':0}
        write(lp,ledger);write(root/'launch.json',{'plan_pin':pin(plan_path),'python':sys.executable,'maximum_fits':0,'new_tuning':False})
    started=time.monotonic();base=ledger['active_seconds'];env=dict(os.environ,PYTHONPATH=str(ROOT/'src'),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    def update():ledger['active_seconds']=base+time.monotonic()-started;write(lp,ledger,replace=True)
    try:
        for index,(phase,n,y,k) in enumerate(schedule()):
            key='-'.join(str(x) for x in (phase,n,y,k) if x is not None)
            if index<len(ledger['completed']):need(ledger['completed'][index]['key']==key,'Noncontiguous completed schedule');continue
            if (root/'PAUSE.request').exists():ledger.update(status='PAUSED',phase=None);update();return
            ledger.update(status='RUNNING',phase=key)
            update();receipt=root/'phases'/(key+'.json');need(not receipt.exists(),'Existing consumed phase preserved')
            command=[sys.executable,'-B',__file__,'worker','--plan',str(plan_path),'--phase',phase,'--result',str(receipt)]
            if n is not None:command+=['--years',str(n),'--year',str(y),'--kind',k]
            log=root/'logs'/(key+'.log');log.parent.mkdir(parents=True,exist_ok=True)
            with log.open('x') as stream:
                child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT);begin=time.monotonic()
                while child.poll() is None:
                    rss_text=subprocess.run(['/bin/ps','-o','rss=','-p',str(child.pid)],capture_output=True,text=True).stdout.strip();rss=int(rss_text or 0)*1024
                    ledger['peak_worker_rss_bytes']=max(ledger['peak_worker_rss_bytes'],rss)
                    write(root/'resource_checkpoint.json',{'phase':key,'pid':child.pid,'rss_bytes':rss,'peak_rss_bytes':ledger['peak_worker_rss_bytes'],'fits_completed':ledger['fits_completed'],'active_seconds':base+time.monotonic()-started},replace=True)
                    if rss>cfg['budgets']['peak_rss_bytes'] or time.monotonic()-begin>cfg['budgets']['phase_seconds'] or base+time.monotonic()-started>cfg['budgets']['total_active_seconds'] or sum(p.stat().st_size for p in root.rglob('*') if p.is_file())>cfg['budgets']['output_bytes']:
                        child.terminate()
                        try:child.wait(timeout=5)
                        except subprocess.TimeoutExpired:child.kill();child.wait()
                        raise RuntimeError('Resource stop; incomplete phase consumed and preserved')
                    time.sleep(2)
            need(child.returncode==0 and receipt.exists(),'Phase failed; see '+str(log));result=read(receipt);check(result['result'])
            ledger['completed'].append({'key':key,'receipt':pin(receipt)});ledger['phase']=None
            update();print(json.dumps({'completed_phase':key,'fits_completed':ledger['fits_completed'],'maximum':0,'active_seconds':ledger['active_seconds']}),flush=True)
        ledger.update(status='COMPLETE_VERIFIED_INPUTS',phase=None);update()
    except BaseException as exc:
        ledger.update(status='FAILED_PRESERVED',error=str(exc));update();raise


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('plan','run','resume','pause','worker'));p.add_argument('--plan',required=True)
    p.add_argument('--data-root');p.add_argument('--protocol');p.add_argument('--output');p.add_argument('--phase');p.add_argument('--years',type=int);p.add_argument('--year',type=int);p.add_argument('--kind');p.add_argument('--result');a=p.parse_args()
    if a.action=='plan':
        from dmv2.extension_inputs import seal
        plan=seal(ROOT,a.data_root,a.protocol,a.output,a.plan)
        print(json.dumps({'status':plan['status'],'source_projection_bytes':plan['projected_bytes_per_pass'],'fits':0}));return
    if a.action=='worker':
        if a.phase not in ('prepare','verify_data'):raise ValueError('Input-only phases')
        worker(a)
    elif a.action=='pause':
        root=Path(read(a.plan)['output_directory']);write(root/'PAUSE.request',{'requested_ns':time.time_ns()});print('Pause requested at next completed phase boundary.')
    else:run(a.plan,resume=a.action=='resume')

if __name__=='__main__':main()
