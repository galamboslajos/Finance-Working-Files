"""Single-use bounded saved-mark calibration; explicit plan then execution.

Reads only the already approved Jan/Oct2020 audit and its exact source plan.
No full-history build, estimator fit, inference or portfolio performance run.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import signal
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq

from dmv2.full_path_accounting import prepare_price_index, prepare_action_registry, replay_unit_episode
from dmv2.path_target_audit import load_path_audit_values, verify_path_audit_plan
from dmv2.path_targets import build_path_targets

BASE = Path(__file__).resolve().parents[1]
KEYS = ('source_security_id','signal_date','side','path_variant')
NAMES = ('episodes.parquet','marks.parquet','targets.parquet')
MAX_SOURCE = 10 * 1024**2
MAX_DERIVED = 32 * 1024**2


def pin(path):
    path=Path(path).resolve();h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    return {'path':str(path),'size_bytes':path.stat().st_size,'sha256':h.hexdigest()}


def checked(item):
    assert pin(item['path'])==item, 'Pinned calibration input changed'
    return Path(item['path'])


def write(path, value):
    with Path(path).open('x') as stream:json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')


def meta_fields(name):
    return list(KEYS)+(['date','mark'] if name=='marks.parquet' else
        ['target_family','outcome_month','label_available_at'] if name=='targets.parquet' else [])


def projection(file, columns=None):
    return sum(file.metadata.row_group(g).column(c).total_compressed_size
        for g in range(file.metadata.num_row_groups) for c in range(file.metadata.num_columns)
        if columns is None or file.metadata.schema.column(c).path.split('.')[0] in columns)


def plan(directory, output):
    root=Path(directory).resolve()
    audit=json.loads((root/'audit.json').read_text());review=json.loads((root/'reviewed_plan.json').read_text())
    assert audit['status']=='COMPLETE_BOUNDED_AUDIT'
    verify_path_audit_plan(review)
    saved=[];amount=0
    for name in NAMES:
        item=pin(root/name);declared=audit['artifacts'][name]
        assert item['size_bytes']==declared['size_bytes'] and item['sha256']==declared['sha256'].removeprefix('sha256:')
        f=pq.ParquetFile(root/name);amount+=projection(f)+projection(f,meta_fields(name));saved.append(item)
    code=[pin(Path(__file__)),pin(BASE/'src/dmv2/full_path_accounting.py'),pin(BASE/'src/dmv2/path_targets.py'),
        pin(BASE/'src/dmv2/path_target_audit.py'),pin(BASE/'scripts/run_path_target_audit.py')]
    json_bytes=(root/'audit.json').stat().st_size+(root/'reviewed_plan.json').stat().st_size
    assert amount+json_bytes<=MAX_DERIVED
    source_bytes=review['read_plan']['materialization_projected_compressed_bytes'];assert source_bytes<=MAX_SOURCE
    result={'schema':'dmv2-fast-path-calibration-plan/v2','bounded_directory':str(root),'saved_pins':saved,
        'audit_pin':pin(root/'audit.json'),'reviewed_plan_pin':pin(root/'reviewed_plan.json'),'code_pins':code,
        'derived_parquet_projected_bytes':amount,'derived_JSON_read_bytes':json_bytes,
        'source_projected_bytes':source_bytes,'limits':{'derived_bytes':MAX_DERIVED,'source_bytes':MAX_SOURCE,
        'wall_seconds':120,'peak_rss_bytes':4*1024**3},'fits':0,'inference_calls':0,'portfolio_backtests':0}
    write(output,result);return result


def run(plan_path, output):
    cfg=json.loads(Path(plan_path).read_text());assert cfg['schema']=='dmv2-fast-path-calibration-plan/v2'
    assert cfg['derived_parquet_projected_bytes']+cfg['derived_JSON_read_bytes']<=MAX_DERIVED and cfg['source_projected_bytes']<=MAX_SOURCE
    for item in cfg['code_pins']+cfg['saved_pins']+[cfg['audit_pin'],cfg['reviewed_plan_pin']]:checked(item)
    output=Path(output).resolve();output.mkdir(exist_ok=False)
    write(output/'STARTED.json',{'status':'STARTED','plan_pin':pin(plan_path)})
    began=time.monotonic();peak=0
    def guard():
        nonlocal peak
        measured=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak=max(peak,measured if sys.platform=='darwin' else measured*1024)
        assert peak<=cfg['limits']['peak_rss_bytes'] and time.monotonic()-began<=cfg['limits']['wall_seconds'],'Calibration resource stop'
    def timeout(*_):raise TimeoutError('Calibration120second wall cap')
    previous=signal.signal(signal.SIGALRM,timeout);signal.alarm(120)
    try:
        review=json.loads(checked(cfg['reviewed_plan_pin']).read_text());verify_path_audit_plan(review)
        files={Path(p['path']).name:pq.ParquetFile(checked(p)) for p in cfg['saved_pins']}
        for name,f in files.items():
            rows=f.read(columns=meta_fields(name)).to_pylist()
            assert all(r['signal_date'] in ('2019-12-31','2020-09-30') for r in rows)
            if name=='marks.parquet':assert all('2020-01-01'<=r['date']<='2020-11-02' and r['mark'] in ('open','close') for r in rows)
            if name=='targets.parquet':assert all(r['outcome_month'] in ('2020-01','2020-10') and r['label_available_at'] in ('2020-02-03','2020-11-02') for r in rows)
        # Every saved-table clock passed before financial conversion.
        old_episodes=files['episodes.parquet'].read().to_pylist()
        old_marks=files['marks.parquet'].read().to_pylist()
        old_targets=files['targets.parquet'].read()
        prices,_=load_path_audit_values(review);index=prepare_price_index(prices,review['sessions'])
        registry=prepare_action_registry(review['events'],review['historical_retirements'],sessions=review['sessions']);guard()
        key=lambda r:tuple(r[k] for k in KEYS)
        saved={}
        for row in old_marks:saved.setdefault(key(row),[]).append(row)
        original={(r['source_security_id'],r['signal_date']):r for r in review['original_cohort']}
        windows={r['signal_date']:r for r in review['windows']}
        episodes=[];marks=[];mismatches=[];max_abs=0.;exact=0;total=0;signed_changes=[];scalar_start=time.monotonic()
        for i,row in enumerate(old_episodes):
            if i%100==0:guard()
            k=key(row)
            if not row['requested']:episodes.append(row);continue
            args=dict(source_security_id=k[0],intrinio_security_id=original[k[:2]]['intrinio_security_id'],signal_date=k[1],
                window=windows[k[1]],side=k[2],path_variant=k[3],registry=registry)
            result=replay_unit_episode(index,**args)
            for field in ('requested','status','reason'):
                if result[field]!=row[field]:mismatches.append({'key':k,'field':field,'old':row[field],'new':result[field]})
            episodes.append({**dict(zip(KEYS,k)),**{field:result[field] for field in ('requested','status','reason')}})
            prior=saved.get(k,[])
            assert [(r['date'],r['mark']) for r in prior]==[(r['date'],r['mark']) for r in result['marks']],k
            for old,new in zip(prior,result['marks']):
                max_abs=max(max_abs,abs(old['wealth']-new['wealth']));exact+=old['wealth']==new['wealth'];total+=1
                marks.append({**dict(zip(KEYS,k)),**new})
            signed=replay_unit_episode(index,**args,retain_signed_wealth=True)
            if signed['legacy_capital_gate_would_trigger']:signed_changes.append({'key':k,'legacy_status':result['status'],
                'signed_status':signed['status'],'retained_marks':signed['negative_reference_nav_retained']})
        scalar_seconds=time.monotonic()-scalar_start
        spec=importlib.util.spec_from_file_location('calibration_bounded_cli',BASE/'scripts/run_path_target_audit.py')
        cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
        # Arrow conversion once is essential: list input incurs quadratic schema scans.
        panel=build_path_targets(cli.core_cohort(review),pa.Table.from_pylist(episodes),pa.Table.from_pylist(marks),
            windows=review['windows'],sessions=review['sessions'],observed_through='2020-11-02')
        prior_rows={tuple(r[f] for f in (*KEYS,'target_family')):r for r in old_targets.to_pylist()}
        class_changes=[];numeric_changes=0;max_target=0.
        for row in panel.table.to_pylist():
            k=tuple(row[f] for f in (*KEYS,'target_family'));prior=prior_rows[k]
            if row['target_class']!=prior['target_class']:class_changes.append({'key':k,'old':prior['target_class'],'new':row['target_class']})
            for field in ('cash_return','maximum_drawdown','utility','target_value'):
                a,b=row[field],prior[field]
                if a is None or b is None:assert a is b,(k,field)
                elif a!=b:numeric_changes+=1;max_target=max(max_target,abs(a-b))
        guard()
        for item in cfg['code_pins']+cfg['saved_pins']+[cfg['audit_pin'],cfg['reviewed_plan_pin']]:checked(item)
        accounting=not mismatches and max_abs<=1e-12
        result={'schema':'dmv2-fast-path-bounded-calibration/v2','status':'PASS' if accounting and not class_changes else
            'PASS_ACCOUNTING_CLASS_DIFFERENCES_REQUIRE_REVIEW' if accounting else 'FAIL','plan_pin':pin(plan_path),
            'original_episodes':len(old_episodes),'requested_episodes':sum(r['requested'] for r in old_episodes),
            'marks_compared':total,'exact_mark_equal':exact,'max_absolute_mark_difference':max_abs,
            'status_reason_differences':mismatches,'class_rank_differences':class_changes,
            'changed_numeric_target_cells':numeric_changes,'max_absolute_numeric_target_difference':max_target,
            'signed_domain_changes':signed_changes,'scalar_two_modes_wall_seconds':scalar_seconds,
            'wall_seconds':time.monotonic()-began,'peak_rss_bytes':peak,'fits':0,'inference_calls':0,'portfolio_backtests':0}
        write(output/'result.json',result);return result
    except BaseException as error:
        write(output/'failure.json',{'type':type(error).__name__,'message':str(error),'wall_seconds':time.monotonic()-began})
        raise
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,previous)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['plan','run'])
    parser.add_argument('--bounded-directory',type=Path);parser.add_argument('--plan',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=plan(args.bounded_directory,args.output) if args.action=='plan' else run(args.plan,args.output)
    print(json.dumps(result,sort_keys=True,indent=2))


if __name__=='__main__':main()
