#!/usr/bin/env python3
"""Sequential fixed twenty-fit controller; all numerical work stays in guards.

Import is standard-library only. Metadata planners never fit/load estimators.
A graceful pause finishes the current guarded phase. Resume is explicit and is
allowed only after that pause or a guard's pre-dispatch resource wait. Failed or
interrupted work is preserved and is never automatically retried. SIGKILL of
this controller cannot run cleanup; use the pause file or SIGTERM instead.
"""
from pathlib import Path
import argparse
import contextlib
import datetime
import fcntl
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time

from control import ROOT, check, need, pin, read, replace_json, reserve_read, save_new

YEARS = (2020, 2021, 2022, 2023)
ARMS = ('rf_paper15', 'xgb_paper15', 'rf_extended', 'xgb_extended', 'rf_tuned_extended')
RESUMABLE = {'PAUSED', 'PAUSED_BEFORE_DISPATCH', 'WAITING_FOR_MEMORY',
             'WAITING_FOR_MEMORY_PRESSURE', 'WAITING_FOR_DISK'}


def schedule():
    result = []
    for year in YEARS:
        result.append(dict(action='prepare', year=year, arm=None))
        for arm in ARMS:
            result.extend(dict(action=action, year=year, arm=arm) for action in ('fit', 'predict'))
        result.append(dict(action='verify', year=year, arm=None))
    result.extend(dict(action=action, year=None, arm=None) for action in ('finalize', 'verify_final'))
    return result


def phase_key(step):
    return '_'.join(str(x) for x in (step['action'], step['year'], step['arm']) if x is not None)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configuration(root, protocol_path, expected_sha):
    root, protocol_path = Path(root).resolve(), Path(protocol_path).resolve()
    need(protocol_path.is_relative_to(ROOT) and pin(protocol_path)['sha256'] == expected_sha,
         'Exact sealed protocol pin required')
    p = read(protocol_path)
    need(p['schema'] == 'dmv2-corrected-refit-protocol/v1'
         and p['status'] == 'SEALED_BEFORE_FINANCIAL_READ'
         and p['outcome_years'] == list(YEARS) and p['parameter_search'] is False
         and p['protected_2024_plus'] is True, 'Frozen four-window no-search protocol required')
    need(p['limits']['max_fits'] == 20 and p['limits']['max_preprocessors'] == 8,
         'Twenty fits/eight preprocessing states required')
    need(p['limits']['active_seconds_by_category']['models'] == 14400
         and p['limits']['active_seconds_by_category']['verification'] == 7200,
         'Frozen cumulative model/verification budgets required')
    paths = {'pipeline_pin': Path(__file__), 'worker_pin': ROOT / 'model_worker.py',
             'verification_worker_pin': ROOT / 'verify_models.py',
             'control_pin': ROOT / 'control.py', 'supervisor_pin': ROOT / 'supervise.py'}
    for name, path in paths.items():
        need(p[name] == pin(path), 'Protocol code binding differs: ' + name)
        check(p[name])
    for name in ('catalog_pin', 'input_verification_pin', 'terminal_guard_pin'):
        check(p[name])
    need(read(check(p['input_verification_pin']))['status'] == 'PASS', 'Independent input PASS missing')
    need(read(check(p['terminal_guard_pin']))['status'] == 'PASS', 'Terminal guard PASS missing')
    run = Path(p['run_directory']).resolve()
    need(root.is_dir() and run == ROOT / 'run/models', 'Unexpected exact source/output namespace')
    return root, protocol_path, p, run


def code_checks(p, protocol_path, expected_sha):
    need(pin(protocol_path)['sha256'] == expected_sha, 'Protocol changed during pipeline')
    for name in ('pipeline_pin', 'worker_pin', 'verification_worker_pin', 'control_pin', 'supervisor_pin'):
        check(p[name])


@contextlib.contextmanager
def controller_lock():
    directory = ROOT / 'run/control'
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.pipeline.lock').open('a+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError('One pipeline controller only') from error
        yield


def checkpoint_path(run, step):
    return run / 'pipeline/checkpoints' / ('checkpoint.' + phase_key(step) + '.json')


def result_path(run, step):
    action, year, arm = (step[k] for k in ('action', 'year', 'arm'))
    fold = run / 'folds' / str(year)
    if action == 'prepare': return fold / 'DATA_COMPLETE.json'
    if action == 'fit': return fold / 'arms' / arm / 'FIT_COMPLETE.json'
    if action == 'predict': return fold / 'arms' / arm / 'PREDICT_COMPLETE.json'
    if action == 'verify': return run.parent / 'verification' / f'models_{year}.json'
    if action == 'finalize': return run / 'audit.json'
    if action == 'verify_final': return run.parent / 'verification/models.json'
    raise ValueError('Unknown fixed phase')


def ensure_fresh_action(run, step):
    output = result_path(run, step)
    need(not output.exists(), 'Uncheckpointed completed artifact requires independent manual review')
    action = step['action']
    if action == 'prepare': markers = [output.parent / 'DATA_STARTED.json']
    elif action == 'fit': markers = [output.parent / 'FIT_ATTEMPT.json']
    elif action == 'predict': markers = [output.parent / 'PREDICT_STARTED.json']
    elif action in ('verify', 'verify_final'): markers = [Path(str(output) + '.STARTED.json')]
    else: markers = []
    need(not any(x.exists() for x in markers), 'Incomplete attempt is consumed; no automatic retry')


def validate_result(run, step, source_pin):
    item = pin(result_path(run, step))
    # Forecast/verification receipts contain numerical summaries. Charge their
    # complete bytes before parsing, then inspect only completion/identity fields.
    reserve_read(ROOT, 'pipeline_result_' + phase_key(step), item['size_bytes'],
                 {'plan_pin': item, 'completion_fields_only': True}, 'compressed_source')
    value = read(check(item))
    expected = {'prepare': 'COMPLETE', 'fit': 'FIT_COMPLETE', 'predict': 'COMPLETE',
                'verify': 'PASS', 'finalize': 'COMPLETE_PENDING_INDEPENDENT_VERIFICATION',
                'verify_final': 'PASS'}[step['action']]
    need(value['status'] == expected, 'Phase result is not successfully complete')
    if step['action'] in ('prepare', 'fit', 'predict', 'finalize'):
        wp = run / 'checkpoints' / f"{step['action']}_{step['year']}_{step['arm']}.json"
        c = read(wp)
        need(c['status'] == 'COMPLETE' and c['result_pin'] == item,
             'Worker checkpoint does not bind canonical result')
    elif step['action'] == 'verify':
        need(value['year'] == step['year'] and value['source_plan_pin'] == source_pin,
             'Independent annual proof belongs to another source/fold')
    else:
        need(value['source_plan_pin'] == source_pin and value['verified_estimator_fits'] == 20
             and value['verified_training_only_preprocessing_states'] == 8,
             'Final independent twenty-fit proof differs')
    return item


def counts(completed):
    return {'completed_phases': len(completed), 'completed_estimator_fits': sum(s['action'] == 'fit' for s in completed),
            'completed_preprocessors': 2 * sum(s['action'] == 'prepare' for s in completed),
            'completed_annual_verifications': sum(s['action'] == 'verify' for s in completed),
            'maximum_estimator_fits': 20, 'maximum_preprocessors': 8, 'maximum_phases': 50}


def validate_progress(progress, identity_pin, run, resume):
    need(progress['identity_pin'] == identity_pin and progress['schedule'] == schedule(), 'Pipeline identity/schedule changed')
    need(resume and progress['status'] in RESUMABLE, 'Explicit resume is limited to graceful pause/pre-dispatch wait')
    n = len(progress['completed'])
    need(progress['completed'] == schedule()[:n], 'Completed phases are not the fixed schedule prefix')
    need(set(progress['checkpoint_pins']) == {phase_key(s) for s in progress['completed']},
         'Completed checkpoint pin set differs')
    for step in progress['completed']:
        cp = progress['checkpoint_pins'][phase_key(step)]
        need(Path(cp['path']) == checkpoint_path(run, step), 'Checkpoint path changed')
        c = read(check(cp))
        need(c['step'] == step and c['status'] == 'COMPLETE' and c['identity_pin'] == identity_pin,
             'Phase checkpoint changed')
        check(c['result_pin']); check(c['launch_pin']); check(c['action_plan_pin'])
        guard = read(check(c['guard_pin']))
        need(guard['status'] == 'COMPLETE' and guard['launch_pin'] == c['launch_pin'],
             'Completed guard proof changed')
    return n


def make_phase_plan(worker, verifier, root, protocol_path, run, step, output):
    if step['action'] == 'verify':
        verifier.make_plan(root, protocol_path, run, step['year'], output)
    elif step['action'] == 'verify_final':
        verifier.make_final_plan(protocol_path, run, output)
    else:
        value = worker.action_plan(root, protocol_path, run, step['action'], step['year'], step['arm'], write=False)
        save_new(output, value)
    return pin(output)


def reserve_categories(phase, plan_pin, plan):
    need(set(plan['projected_reads']) == {'compressed_source', 'derived_matrix'}, 'Both read categories must be explicit')
    labels = {}
    for category, amount in plan['projected_reads'].items():
        need(type(amount) is int and amount >= 0, 'Invalid projected read')
        if amount:
            label = phase + '_' + category
            reserve_read(ROOT, label, amount, {'plan_pin': plan_pin, 'phase': phase}, category)
            labels[category] = label
    return labels


def command_vector(p, root, protocol_path, run, step, plan_pin, labels, launch_path):
    args = [p['python_path']]
    if step['action'] in ('verify', 'verify_final'):
        args += [str(ROOT / 'verify_models.py'), 'run' if step['action'] == 'verify' else 'final',
                 '--plan', plan_pin['path'], '--plan-sha256', plan_pin['sha256'],
                 '--output', str(result_path(run, step))]
    else:
        args += [str(ROOT / 'model_worker.py'), 'execute', '--root', str(root),
                 '--protocol', str(protocol_path), '--run', str(run),
                 '--action-plan', plan_pin['path'], '--action-plan-sha256', plan_pin['sha256'],
                 '--launch', str(launch_path)]
    for category, label in sorted(labels.items()): args += ['--reservation', category + '=' + label]
    return args


def make_launch(p, protocol_path, root, run, step, phase, plan_pin, labels, output):
    verification = step['action'] in ('verify', 'verify_final')
    category = 'verification' if verification else 'models'
    maximum = 1800 if verification else min(1200, p['limits']['per_fit_seconds'])
    return save_new(output, {'schema': 'dmv2-corrected-refit-phase-launch/v1',
        'protocol_pin': pin(protocol_path), 'phase': phase, 'time_category': category,
        'maximum_seconds': maximum, 'action_plan_pin': plan_pin,
        'code_pins': [p[k] for k in ('pipeline_pin', 'worker_pin', 'verification_worker_pin', 'control_pin', 'supervisor_pin')],
        'command': command_vector(p, root, protocol_path, run, step, plan_pin, labels, output)})


def terminate_guard(child):
    """SIGTERM gives the supervisor time to kill its separately owned worker group."""
    if child is None or child.poll() is not None: return
    previous = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGHUP)}
    for s in previous: signal.signal(s, signal.SIG_IGN)
    try:
        try: os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError: pass
        try: child.wait(timeout=25)
        except subprocess.TimeoutExpired:
            try: os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError: pass
            child.wait()
    finally:
        for s, handler in previous.items(): signal.signal(s, handler)


def dispatch(p, launch_pin):
    child = None
    try:
        child = subprocess.Popen([p['python_path'], str(ROOT / 'supervise.py'), '--launch',
                                  launch_pin['path'], '--sha256', launch_pin['sha256']],
                                 cwd=ROOT, start_new_session=True)
        return child.wait()
    finally:
        terminate_guard(child)


def pipeline(root, protocol_path, expected_sha, resume=False):
    root, protocol_path, p, run = configuration(root, protocol_path, expected_sha)
    with controller_lock():
        # Imports remain code/metadata only; estimator/array values are child-only.
        worker = load_module(check(p['worker_pin']), 'pipeline_model_worker')
        verifier = load_module(check(p['verification_worker_pin']), 'pipeline_model_verifier')
        identity_path, progress_path = run / 'pipeline/identity.json', run / 'pipeline/progress.json'
        if not identity_path.exists():
            need(not resume and not progress_path.exists() and not run.exists(), 'Fresh initial model namespace required')
            worker.plan(root, protocol_path, run)
            identity = {'schema': 'dmv2-corrected-refit-pipeline-identity/v1', 'protocol_pin': pin(protocol_path),
                'source_plan_pin': pin(run / 'read_plan.json'), 'build_identity_pin': pin(run / 'build_identity.json'),
                'code_pins': {k: p[k] for k in ('pipeline_pin', 'worker_pin', 'verification_worker_pin', 'control_pin', 'supervisor_pin')},
                'schedule': schedule(), 'automatic_failed_retry': False, 'models_only_fixed_parameters': True}
            identity_pin = save_new(identity_path, identity)
            progress = {'schema': 'dmv2-corrected-refit-pipeline-progress/v1', 'status': 'READY',
                'identity_pin': identity_pin, 'schedule': schedule(), 'completed': [], 'checkpoint_pins': {},
                'attempts': [], **counts([])}
            save_new(progress_path, progress)
            start = 0
        else:
            identity_pin = pin(identity_path); identity = read(identity_path)
            need(identity['protocol_pin'] == pin(protocol_path) and identity['code_pins'] ==
                 {k: p[k] for k in identity['code_pins']}, 'Saved pipeline code/protocol identity changed')
            check(identity['source_plan_pin']); check(identity['build_identity_pin'])
            progress = read(progress_path); start = validate_progress(progress, identity_pin, run, resume)
        current_step = None
        def stop(signum, frame):
            raise SystemExit('Controller received signal ' + str(signum))
        previous = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGHUP)}
        for s in previous: signal.signal(s, stop)
        try:
            for step in schedule()[start:]:
                current_step = step; key = phase_key(step)
                code_checks(p, protocol_path, expected_sha)
                if (ROOT / 'PAUSE_REQUESTED.json').exists():
                    progress.update(status='PAUSED', next_step=step, **counts(progress['completed']))
                    replace_json(progress_path, progress); return progress
                ensure_fresh_action(run, step)
                attempts = [a for a in progress['attempts'] if a['step'] == step]
                need(not attempts or attempts[-1]['status'] in RESUMABLE,
                     'Prior failed/incomplete guarded attempt cannot be retried')
                phase = 'pipeline_' + key + '_a' + str(len(attempts) + 1).zfill(3)
                if attempts:
                    plan_pin = attempts[-1]['action_plan_pin']; check(plan_pin)
                    if step['action'] not in ('verify', 'verify_final'):
                        need(read(plan_pin['path']) == worker.action_plan(root, protocol_path, run,
                            step['action'], step['year'], step['arm'], write=False), 'Metadata plan changed on explicit resume')
                else:
                    plan_pin = make_phase_plan(worker, verifier, root, protocol_path, run, step,
                                               run / 'pipeline/plans' / (key + '.json'))
                plan_value = read(check(plan_pin))
                attempt = {'step': step, 'phase': phase, 'status': 'RESERVING', 'action_plan_pin': plan_pin}
                progress['attempts'].append(attempt); progress.update(status='RUNNING', current_step=step)
                replace_json(progress_path, progress)
                labels = reserve_categories(phase, plan_pin, plan_value)
                launch_pin = make_launch(p, protocol_path, root, run, step, phase, plan_pin, labels,
                                         run / 'pipeline/launches' / (phase + '.json'))
                attempt.update(status='DISPATCHING', launch_pin=launch_pin, reservations=labels)
                replace_json(progress_path, progress)
                returncode = dispatch(p, launch_pin)
                guard_pin = pin(ROOT / 'run/control' / (phase + '.json')); guard = read(check(guard_pin))
                attempt.update(guard_pin=guard_pin, returncode=returncode, status=guard['status'])
                if guard['status'] in RESUMABLE:
                    need(returncode == 0, 'Pre-dispatch wait exited abnormally')
                    ensure_fresh_action(run, step)
                    progress.update(status=guard['status'], **counts(progress['completed']))
                    replace_json(progress_path, progress); return progress
                need(returncode == 0 and guard['status'] == 'COMPLETE', 'Guarded phase failed; preserve attempt')
                # A pause can land between guard preflight and worker dispatch.
                if not result_path(run, step).exists() and (ROOT / 'PAUSE_REQUESTED.json').exists():
                    ensure_fresh_action(run, step); attempt['status'] = 'PAUSED'
                    progress.update(status='PAUSED', **counts(progress['completed']))
                    replace_json(progress_path, progress); return progress
                result_pin = validate_result(run, step, identity['source_plan_pin'])
                completed_pin = save_new(checkpoint_path(run, step), {'status': 'COMPLETE', 'step': step,
                    'action_plan_pin': plan_pin, 'launch_pin': launch_pin, 'guard_pin': guard_pin,
                    'result_pin': result_pin, 'identity_pin': identity_pin})
                progress['checkpoint_pins'][key] = completed_pin
                attempt['status'] = 'COMPLETE'; progress['completed'].append(step)
                progress.update(status='BETWEEN_PHASES', **counts(progress['completed']))
                replace_json(progress_path, progress)
            code_checks(p, protocol_path, expected_sha)
            progress.update(status='COMPLETE_VERIFIED', final_verification_pin=pin(result_path(run, schedule()[-1])),
                            **counts(progress['completed']))
            replace_json(progress_path, progress); return progress
        except BaseException as error:
            progress.update(status='FAILED_OR_STOPPED', reason=str(error), current_step=current_step,
                            **counts(progress['completed']))
            replace_json(progress_path, progress)
            raise
        finally:
            for s, handler in previous.items(): signal.signal(s, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True); parser.add_argument('--protocol', required=True)
    parser.add_argument('--protocol-sha256', required=True); parser.add_argument('--resume', action='store_true')
    a = parser.parse_args()
    result = pipeline(a.root, a.protocol, a.protocol_sha256, a.resume)
    print(json.dumps({k: v for k, v in result.items() if k not in ('schedule', 'attempts', 'completed')}, sort_keys=True))


if __name__ == '__main__': main()
