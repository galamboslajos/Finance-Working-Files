"""Explicit sequential comparison with immutable phase tickets and pause checkpoints.

The plan action is metadata-only. Run launches exactly the sealed fits, then
independent verification and the fixed daily replay. Notebook imports never run it.
"""
from pathlib import Path
import argparse
import datetime as dt
import fcntl
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time

STAGE = Path(__file__).resolve().parents[1]
YEARS = (2020, 2021, 2022, 2023)
HEADS = ('endpoint', 'cash_long', 'cash_short', 'utility_long', 'utility_short')


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            value.update(block)
    return value.hexdigest()


def pin(path):
    p = Path(path).resolve()
    return dict(path=str(p), size_bytes=p.stat().st_size, sha256=sha(p))


def check(item):
    p = Path(item['path'])
    need(pin(p) == item, 'Pinned controller input changed: ' + str(p))
    return p


def read(path):
    p = Path(path)
    need(p.stat().st_size <= 16 * 1024**2, 'Controller metadata byte cap')
    return json.loads(p.read_text())


def write(path, value, *, replace=False):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n'
    if replace:
        temp = p.with_name(p.name + '.writing')
        temp.write_text(text)
        os.replace(temp, p)
    else:
        with p.open('x') as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def module(path):
    spec = importlib.util.spec_from_file_location('comparison_independent_verifier', path)
    item = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(item)
    return item


def available_proxy():
    command = subprocess.run(['/usr/bin/vm_stat'], capture_output=True, text=True, check=True)
    page = int(re.search(r'page size of (\d+) bytes', command.stdout).group(1))
    values = {k: int(v) for k, v in re.findall(r'^(Pages [^:]+):\s+(\d+)\.', command.stdout, re.M)}
    return page * sum(values[k] for k in ('Pages free', 'Pages inactive', 'Pages speculative'))


def rss(pid):
    result = subprocess.run(['/bin/ps', '-o', 'rss=', '-p', str(pid)], capture_output=True, text=True)
    return int(result.stdout.strip() or 0) * 1024


def code_pins():
    files = sorted((STAGE / 'scripts').rglob('*.py'))
    files.extend(sorted((STAGE / 'src/dmv2').glob('*.py')))
    return [pin(p) for p in files]


def validate_launch(plan_path, launch):
    need(launch['model_plan_pin'] == pin(plan_path), 'Model plan changed')
    planned = read(plan_path)
    need(launch['protocol_pin'] == planned['protocol_pin'], 'Launch protocol differs from planned protocol')
    check(launch['protocol_pin'])
    need({p['path'] for p in launch['code_pins']} == {p['path'] for p in code_pins()}, 'Complete controller code-pin set required')
    for item in launch['code_pins']:
        check(item)
    protocol = read(launch['protocol_pin']['path'])
    need(launch['python_path'] == str(Path(protocol['project_root']) / '.venv/bin/python'), 'Frozen project Python required')
    return protocol


def active_time(ledger):
    if '_controller_started_monotonic' not in ledger:
        return ledger['active_seconds']
    return ledger['_active_base_seconds'] + time.monotonic() - ledger['_controller_started_monotonic']


def resource_boundary(root, ledger, limits):
    ledger['active_seconds'] = active_time(ledger)
    need(ledger['active_seconds'] < limits['active_seconds'], 'Active time resource stop before dispatch')
    need(sum(p.stat().st_size for p in root.rglob('*') if p.is_file()) <= limits['output_bytes'], 'Output byte resource stop')
    need(rss(os.getpid()) <= limits['rss_stop_bytes'], 'Controller RSS resource stop')


def validate_completed(completed, key, launch_path):
    answer = read(check(completed))
    need(phase_key(answer.get('phase'), answer.get('year'), answer.get('head')) == key
         and 'ticket_pin' in answer, 'Completed phase has another schedule identity')
    ticket_path = check(answer['ticket_pin'])
    ticket = read(ticket_path)
    need(phase_key(answer['phase'], answer.get('year'), answer.get('head')) == key
         and phase_key(ticket['phase'], ticket.get('year'), ticket.get('head')) == key
         and ticket['launch_pin'] == pin(launch_path) and answer['status'] == 'COMPLETE_PHASE',
         'Completed phase has another schedule identity')
    item = answer['worker_result']
    if set(('path', 'size_bytes', 'sha256')) <= set(item):
        check(item)
    if 'receipt_pin' in item:
        check(item['receipt_pin'])
    return item


def require_model_proof(plan_path, proof_pin):
    proof = read(check(proof_pin))
    need(proof.get('status') == 'PASS' and proof.get('phase') == 'models'
         and proof.get('model_plan_pin') == pin(plan_path)
         and proof.get('verified_estimator_fits') == 20 and proof.get('verified_shared_preprocessors') == 4
         and proof.get('new_estimator_fits') == 0 and proof.get('financial_2024_plus_values_read') is False,
         'Exact independent20-fit/four-preprocessor proof required before backtest')
    return proof


def validate_ledger(ledger, plan_path, launch_path):
    need(ledger['model_plan_pin'] == pin(plan_path) and ledger['active_phase'] is None,
         'Checkpoint plan differs or has an unfinished phase')
    schedule = [key for y in YEARS for key in [phase_key('prepare', y), *[phase_key('fit', y, h) for h in HEADS]]]
    schedule += ['models_report', 'verify_models', 'backtest', 'verify_backtest']
    completed = ledger['completed_phases']
    need(set(completed) == set(schedule[:len(completed)]), 'Completed phases are not a contiguous fixed schedule')
    fits = sum(k.startswith('fit-') for k in completed)
    need(ledger['fits_completed'] == ledger['fits_started'] == fits <= 20, 'Checkpoint fit counts differ')
    for key, item in completed.items():
        validate_completed(item, key, launch_path)


def reserve(ledger, category, label, amount, limits):
    need(type(amount) is int and amount >= 0, 'Explicit nonnegative read reservation required')
    need(not any(r['label'] == label for r in ledger['reservations']), 'Read reservation is single-use')
    caps = {'model_source': limits['source_projection_bytes'],
            'backtest_source': limits['backtest_source_projection_bytes'], 'derived': limits['derived_read_bytes']}
    used = sum(r['bytes'] for r in ledger['reservations'] if r['category'] == category)
    need(used + amount <= caps[category], 'Cumulative read allowance exhausted: ' + category)
    ledger['reservations'].append(dict(category=category, label=label, bytes=amount, at=now()))


def phase_key(phase, year=None, head=None):
    return '-'.join(str(v) for v in (phase, year, head) if v is not None)


def result_payload(result):
    if isinstance(result, Path):
        return {'receipt_pin': pin(result)}
    need(isinstance(result, dict), 'Worker must return a JSON receipt')
    return result


def worker(ticket_path, ticket_sha):
    need(sha(ticket_path) == ticket_sha, 'Phase ticket hash mismatch')
    ticket = read(ticket_path)
    launch = read(ticket['launch_pin']['path'])
    check(ticket['launch_pin'])
    plan_path = check(launch['model_plan_pin'])
    protocol = validate_launch(plan_path, launch)
    plan = read(plan_path)
    marker = Path(ticket['started_marker'])
    write(marker, dict(ticket_pin=pin(ticket_path), pid=os.getpid(), started_at=now()))
    started = time.monotonic()
    import resource
    def guard():
        used = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        used = used if sys.platform == 'darwin' else used * 1024
        need(used <= protocol['limits']['rss_stop_bytes'], 'Worker RSS resource stop')
        need(time.monotonic() - started <= ticket['maximum_seconds'], 'Worker time resource stop')
    phase = ticket['phase']
    if phase in ('prepare', 'fit', 'models_report'):
        from dmv2 import target_comparison_models as models
        if phase == 'prepare':
            answer = models.prepare_fold(plan, ticket['year'], resource_check=guard)
        elif phase == 'fit':
            answer = models.fit_head(plan, ticket['year'], ticket['head'], resource_check=guard)
        else:
            answer = models.report_models(plan)
    elif phase in ('verify_models', 'verify_backtest'):
        verifier = module(STAGE / 'scripts/verify_unstopped_target_comparison.py')
        answer = (verifier.verify_models(plan_path, ticket['verification_plan']) if phase == 'verify_models'
                  else verifier.verify_backtest(plan_path, ticket['backtest_plan'], ticket['verification_plan']))
    elif phase == 'backtest':
        require_model_proof(plan_path, ticket['model_verification_pin'])
        from dmv2 import target_comparison_backtest as backtest
        answer = backtest.run_backtest(read(ticket['backtest_plan']), resource_check=guard)
    else:
        raise ValueError('Unrecognized phase')
    guard()
    write(ticket['result_path'], dict(status='COMPLETE_PHASE', phase=phase, year=ticket.get('year'),
        head=ticket.get('head'), ticket_pin=pin(ticket_path), worker_result=result_payload(answer),
        wall_seconds=time.monotonic()-started, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if sys.platform == 'darwin' else 1024), finished_at=now()))


def run_phase(plan_path, launch_path, ledger_path, ledger, phase, year=None, head=None, **extra):
    launch, plan = read(launch_path), read(plan_path)
    protocol = validate_launch(plan_path, launch)
    limits = protocol['limits']
    root = Path(plan['output_directory'])
    key = phase_key(phase, year, head)
    completed = ledger['completed_phases'].get(key)
    if completed:
        return validate_completed(completed, key, launch_path)
    need(ledger.get('active_phase') is None, 'An unfinished phase cannot be automatically retried')
    resource_boundary(root, ledger, limits)
    if (root / 'PAUSE_REQUESTED').exists():
        ledger['status'] = 'PAUSED'
        ledger['active_seconds'] = active_time(ledger)
        ledger['updated_at'] = now()
        write(ledger_path, ledger, replace=True)
        return None
    if phase == 'prepare':
        reserve(ledger, 'model_source', key, int(plan['folds'][str(year)]['source_projection_bytes']), limits)
    if phase.startswith('verify_'):
        measured = read(extra['verification_plan'])
        reserve(ledger, 'derived', key, int(measured['derived_read_bytes']), limits)
    if phase == 'backtest':
        measured = read(extra['backtest_plan'])
        reserve(ledger, 'backtest_source', key, int(measured['source_projection_bytes']), limits)
        reserve(ledger, 'derived', key + '-derived', int(measured['derived_read_bytes']), limits)
    if phase == 'fit':
        need(ledger['fits_started'] < limits['max_fits'], 'Fit-attempt budget exhausted')
        ledger['fits_started'] += 1
    maximum = limits['per_fit_seconds'] if phase == 'fit' else limits['verification_seconds'] if phase.startswith('verify_') else limits['backtest_seconds'] if phase == 'backtest' else 1800
    maximum = min(maximum, limits['active_seconds'] - active_time(ledger))
    need(maximum > 0, 'Active time resource stop before dispatch')
    result_path = root / 'controller/phases' / (key + '.result.json')
    ticket_path = root / 'controller/tickets' / (key + '.json')
    ticket = dict(phase=phase, year=year, head=head, launch_pin=pin(launch_path), maximum_seconds=maximum,
        started_marker=str(root / 'controller/phases' / (key + '.STARTED.json')), result_path=str(result_path), **extra)
    write(ticket_path, ticket)
    ledger['active_phase'] = key
    ledger['status'] = 'RUNNING'
    ledger['updated_at'] = now()
    write(ledger_path, ledger, replace=True)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(STAGE / 'src'),
        OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
    started = time.monotonic()
    log_path = root / 'controller.log'
    with log_path.open('a') as log:
        child = subprocess.Popen([launch['python_path'], '-B', str(Path(__file__).resolve()), '_worker',
            '--ticket', str(ticket_path), '--ticket-sha', sha(ticket_path)], cwd=STAGE, env=env,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            last_size_check = 0.0
            while child.poll() is None:
                elapsed = time.monotonic() - started
                memory = rss(child.pid) + rss(os.getpid())
                ledger['peak_process_tree_rss_bytes'] = max(ledger['peak_process_tree_rss_bytes'], memory)
                need(memory <= limits['rss_stop_bytes'], 'Process-tree memory resource stop')
                need(elapsed <= maximum and active_time(ledger) <= limits['active_seconds'], 'Active time resource stop')
                need(shutil.disk_usage(root).free >= 5*1024**3, 'Insufficient free disk while running')
                if elapsed - last_size_check >= 20:
                    output_bytes = sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
                    need(output_bytes <= limits['output_bytes'], 'Output byte resource stop')
                    last_size_check = elapsed
                ledger['current_phase_seconds'] = elapsed
                ledger['active_seconds'] = active_time(ledger)
                ledger['updated_at'] = now()
                write(ledger_path, ledger, replace=True)
                time.sleep(2)
        except BaseException:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill(); child.wait()
            ledger['active_seconds'] = active_time(ledger)
            raise
    ledger['active_seconds'] = active_time(ledger)
    need(child.returncode == 0 and result_path.is_file(), 'Phase failed; inspect controller.log and preserve all attempts: ' + key)
    answer = read(result_path)
    need(answer['status'] == 'COMPLETE_PHASE' and answer['ticket_pin'] == pin(ticket_path), 'Wrong phase completion receipt')
    if phase.startswith('verify_'):
        need(answer['worker_result'].get('status') == 'PASS', 'Independent verification did not pass')
        if phase == 'verify_models':
            require_model_proof(plan_path, answer['worker_result']['receipt_pin'])
    ledger['peak_process_tree_rss_bytes'] = max(ledger['peak_process_tree_rss_bytes'], answer['peak_rss_bytes'] + rss(os.getpid()))
    need(ledger['peak_process_tree_rss_bytes'] <= limits['rss_stop_bytes'] and time.monotonic() - started <= maximum,
         'Completed phase exceeded memory/time limits')
    resource_boundary(root, ledger, limits)
    ledger['completed_phases'][key] = pin(result_path)
    ledger['active_phase'] = None
    ledger['current_phase_seconds'] = 0
    if phase == 'fit':
        ledger['fits_completed'] += 1
    ledger['updated_at'] = now()
    write(ledger_path, ledger, replace=True)
    print(json.dumps(dict(stage=key, fits_completed=ledger['fits_completed'], fits_maximum=20,
        active_seconds=round(ledger['active_seconds'], 2), peak_rss_bytes=ledger['peak_process_tree_rss_bytes'])), flush=True)
    return answer['worker_result']


def run(plan_path, *, resume=False):
    plan = read(plan_path)
    root = Path(plan['output_directory'])
    launch_path, ledger_path = root / 'launch.json', root / 'ledger.json'
    with (root / '.controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not resume:
            need(not launch_path.exists() and not ledger_path.exists(), 'Use explicit resume for a paused run')
            protocol_path = Path(plan['protocol_pin']['path'])
            check(plan['protocol_pin'])
            protocol = read(protocol_path)
            need(available_proxy() >= protocol['limits']['available_proxy_minimum_bytes'], 'Not enough available/reclaimable memory for launch')
            need(shutil.disk_usage(root).free >= protocol['limits']['free_disk_minimum_bytes'], 'Not enough free disk for launch')
            write(launch_path, dict(model_plan_pin=pin(plan_path), protocol_pin=pin(protocol_path), code_pins=code_pins(),
                python_path=sys.executable, launched_at=now(), maximum_estimator_fits=20, automatic_retry=False))
            ledger = dict(status='RUNNING', model_plan_pin=pin(plan_path), fits_started=0, fits_completed=0,
                active_seconds=0.0, current_phase_seconds=0.0, peak_process_tree_rss_bytes=0,
                active_phase=None, reservations=[], completed_phases={}, updated_at=now())
            reserve(ledger, 'derived', 'model-matrix-dispatch-reserved',
                    int(plan['read_budget']['derived_prepared_read_bytes_total']), protocol['limits'])
            preflight_path = root/'metadata_preflight.json'
            if preflight_path.exists():
                preflight = read(preflight_path)
                need(preflight['status'] == 'PASS_METADATA_ONLY_FOURTEEN_ANNUALS'
                     and preflight['financial_values_read'] is False and preflight['worker_pin'] == plan['worker_pin'],
                     'Prelaunch metadata receipt belongs to another worker')
                reserve(ledger, 'model_source', 'prelaunch-metadata-preflight',
                        preflight['source_projection_bytes'], protocol['limits'])
            write(ledger_path, ledger)
        else:
            ledger = read(ledger_path)
            need(ledger['status'] == 'PAUSED' and ledger['active_phase'] is None, 'Only an orderly paused checkpoint can resume')
            need(not (root / 'PAUSE_REQUESTED').exists(), 'Remove only the pause-request file before explicit resume')
        protocol = validate_launch(plan_path, read(launch_path))
        need(available_proxy() >= protocol['limits']['available_proxy_minimum_bytes'], 'Not enough available/reclaimable memory to run/resume')
        need(shutil.disk_usage(root).free >= protocol['limits']['free_disk_minimum_bytes'], 'Not enough free disk to run/resume')
        validate_ledger(ledger, plan_path, launch_path)
        ledger['_active_base_seconds'] = ledger['active_seconds']
        ledger['_controller_started_monotonic'] = time.monotonic()
        try:
            for year in YEARS:
                if run_phase(plan_path, launch_path, ledger_path, ledger, 'prepare', year) is None:
                    return ledger
                for head in HEADS:
                    if run_phase(plan_path, launch_path, ledger_path, ledger, 'fit', year, head) is None:
                        return ledger
            run_phase(plan_path, launch_path, ledger_path, ledger, 'models_report')
            verifier = module(STAGE / 'scripts/verify_unstopped_target_comparison.py')
            verification_path = root / 'verification/models_read_plan.json'
            if not verification_path.exists():
                verifier.plan_verification(plan_path, None, verification_path)
            # Numerical phases use their own authenticated measured read plans.
            proof = run_phase(plan_path, launch_path, ledger_path, ledger, 'verify_models', verification_plan=str(verification_path))
            if proof is None:
                return ledger
            require_model_proof(plan_path, proof['receipt_pin'])
            from dmv2 import target_comparison_backtest as backtest
            bt_path = root / 'backtest_plan.json'
            if not bt_path.exists():
                value = backtest.plan_backtest(plan['project_root'], plan, protocol, root / 'backtest')
                if not bt_path.exists():
                    write(bt_path, value)
            if run_phase(plan_path, launch_path, ledger_path, ledger, 'backtest', backtest_plan=str(bt_path), model_verification_pin=proof['receipt_pin']) is None:
                return ledger
            bt_verification_path = root / 'verification/backtest_read_plan.json'
            if not bt_verification_path.exists():
                verifier.plan_verification(plan_path, bt_path, bt_verification_path)
            if run_phase(plan_path, launch_path, ledger_path, ledger, 'verify_backtest', backtest_plan=str(bt_path), verification_plan=str(bt_verification_path)) is None:
                return ledger
            need(ledger['fits_completed'] == 20, 'Final fit count differs from protocol')
            ledger['status'] = 'COMPLETE_VERIFIED'
            ledger['updated_at'] = now()
            write(ledger_path, ledger, replace=True)
            return ledger
        except BaseException as error:
            ledger['status'] = 'FAILED_PRESERVED'
            ledger['error'] = dict(type=type(error).__name__, message=str(error))
            ledger['updated_at'] = now()
            write(ledger_path, ledger, replace=True)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'run', 'resume', 'report', 'pause', '_worker'))
    parser.add_argument('--project', type=Path)
    parser.add_argument('--target-audit', type=Path)
    parser.add_argument('--protocol', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--ticket', type=Path)
    parser.add_argument('--ticket-sha')
    args = parser.parse_args()
    if args.action == 'plan':
        from dmv2.target_comparison_models import plan_comparison
        result = plan_comparison(args.project, args.target_audit, args.protocol, args.output)
        print(json.dumps(dict(plan=str(args.output / 'read_plan.json'), source_projection_bytes=result['source_projection_bytes'])))
    elif args.action in ('run', 'resume'):
        print(json.dumps(run(args.plan, resume=args.action == 'resume'), indent=2))
    elif args.action == '_worker':
        worker(args.ticket, args.ticket_sha)
    elif args.action == 'pause':
        root = Path(read(args.plan)['output_directory'])
        write(root / 'PAUSE_REQUESTED', dict(requested_at=now(), boundary='after_current_phase'))
        print('Pause requested after the current phase; completed fits stay preserved.')
    else:
        root = Path(read(args.plan)['output_directory'])
        print(json.dumps(read(root / 'ledger.json'), indent=2))


if __name__ == '__main__':
    main()
