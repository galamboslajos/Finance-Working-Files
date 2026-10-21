#!/usr/bin/env python3
"""Read-only pipeline/resource status. Never opens scores, models or datasets."""
from pathlib import Path
import argparse
import datetime as dt
import json
import re

ROOT=Path(__file__).resolve().parent


def read_status(root, relative):
    p=root/relative
    if not p.exists():return None
    if p.is_symlink() or not p.is_file() or p.stat().st_size>4*1024**2:
        raise ValueError('Expected small regular status JSON')
    return json.loads(p.read_text())


def snapshot(root=ROOT):
    root=Path(root).resolve()
    progress=read_status(root,'run/models/pipeline/progress.json')
    clock=read_status(root,'run/control/active_time.json')
    result={'schema':'dmv2-passive-model-progress/v1',
        'sampled_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
        'pipeline_status':progress['status'] if progress else 'NOT_YET_RECORDED',
        'completed_estimator_fits':progress.get('completed_estimator_fits',0) if progress else 0,
        'maximum_estimator_fits':20,
        'completed_preprocessors':progress.get('completed_preprocessors',0) if progress else 0,
        'maximum_preprocessors':8,
        'completed_phases':progress.get('completed_phases',0) if progress else 0,
        'maximum_phases':50,'stage':progress.get('current_step') if progress else None,
        'completed_active_seconds_by_category':clock.get('consumed_seconds',{}) if clock else {},
        'current_guard_status':'NOT_YET_RECORDED','current_phase_active_seconds':0,
        'current_guard_peak_rss_bytes':0,'overall_guard_peak_rss_bytes':0,
        'memory_pressure_sample':None,'guard_sample_modified_at_utc':None,
        'reads':'pipeline progress, cumulative active_time and referenced guard JSON only; no score/log/model/data reads',
        'timing_note':'Category clock contains completed guards. Current RUNNING time is separate; their estimated sum is a non-atomic status sample.'}
    attempts=progress.get('attempts',[]) if progress else []
    for i,attempt in enumerate(attempts):
        phase=attempt.get('phase')
        if not isinstance(phase,str) or not re.fullmatch(r'pipeline_[A-Za-z0-9_]+',phase):
            raise ValueError('Invalid referenced guard phase')
        relative='run/control/'+phase+'.json';guard=read_status(root,relative)
        if guard is None:continue
        peak=guard.get('peak_process_tree_rss_bytes',0)
        if type(peak) is not int or peak<0:raise ValueError('Invalid guard peak RSS')
        result['overall_guard_peak_rss_bytes']=max(result['overall_guard_peak_rss_bytes'],peak)
        if i==len(attempts)-1:
            result.update(current_guard_status=guard['status'],current_guard_peak_rss_bytes=peak,
                current_phase_active_seconds=guard.get('active_seconds',0),
                memory_pressure_sample=guard.get('macos_memory_pressure',guard.get('resources_before',{}).get('macos_memory_pressure')),
                guard_phase=phase,
                guard_sample_modified_at_utc=dt.datetime.fromtimestamp((root/relative).stat().st_mtime,dt.timezone.utc).isoformat())
    result['estimated_total_active_seconds_by_category']=dict(result['completed_active_seconds_by_category'])
    if attempts and result['current_guard_status']=='RUNNING':
        current=read_status(root,'run/control/'+attempts[-1]['phase']+'.json')
        if current is not None and current.get('status')=='RUNNING':
            category=current['time_category']
            totals=result['estimated_total_active_seconds_by_category']
            totals[category]=totals.get(category,0)+current.get('active_seconds',0)
    for name,maximum in (('completed_estimator_fits',20),('completed_preprocessors',8),('completed_phases',50)):
        if type(result[name]) is not int or not 0<=result[name]<=maximum:raise ValueError('Invalid durable progress count')
    return result


def text_status(z):
    step=z['stage'] or {}
    stage=' / '.join(str(step[k]) for k in ('action','year','arm') if step.get(k) is not None) or 'not recorded'
    totals=', '.join(f'{k}={v/60:.1f}min' for k,v in sorted(z['estimated_total_active_seconds_by_category'].items())) or 'not recorded'
    pressure={1:'normal',2:'warning',4:'critical'}.get(z['memory_pressure_sample'],'not recorded')
    return (f"{z['pipeline_status']}: {z['completed_estimator_fits']}/20 fits, "
        f"{z['completed_preprocessors']}/8 preprocessing states, {z['completed_phases']}/50 phases.\n"
        f"Stage: {stage}; guard: {z['current_guard_status']}.\n"
        f"Active times (status estimate): {totals}.\n"
        f"Guard peak RAM: current {z['current_guard_peak_rss_bytes']/1024**3:.2f}GiB; "
        f"overall {z['overall_guard_peak_rss_bytes']/1024**3:.2f}GiB; sampled pressure {pressure}.\n"
        "Only passive status JSON was read; no scores, models or datasets.")


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--json',action='store_true');a=p.parse_args()
    z=snapshot();print(json.dumps(z,indent=2,sort_keys=True) if a.json else text_status(z))


if __name__=='__main__':main()
