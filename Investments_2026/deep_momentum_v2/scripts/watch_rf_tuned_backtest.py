"""One-second resource supervisor for the separately authorized tuned-RF replay.

Runs only the reviewed replay CLI or independent verifier, never a fit worker.
It preserves failed child logs and never retries a failed phase.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import psutil

GIB = 1024**3
LIMITS = {'prepare': 300, 'preflight': 300, 'backtest': 900, 'verify': 1800}
THREADS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def memory():
    raw = subprocess.check_output(['/usr/bin/vm_stat'], text=True, timeout=5)
    size = int(re.search(r'page size of (\d+) bytes', raw).group(1))
    pages = {name: int(re.search(re.escape(name) + r':\s+(\d+)', raw).group(1))
             for name in ('Pages free', 'Pages inactive', 'Pages speculative')}
    return {'available_proxy_bytes': size * sum(pages.values()), 'page_size': size,
            'pages': pages, 'raw_vm_stat': raw}


def pressure():
    raw = subprocess.check_output(['/usr/sbin/sysctl', '-n',
                                   'kern.memorystatus_vm_pressure_level'], text=True, timeout=5)
    value = int(raw.strip())
    return {'dispatch_value': value, 'normal': value == 1, 'raw': raw.strip()}


def files_size(*roots):
    return sum(p.stat().st_size for root in roots if root.exists()
               for p in root.rglob('*') if p.is_file())


def atomic(path, value):
    temp = path.with_suffix(path.suffix + '.partial')
    with temp.open('w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def stop_tree(child):
    try:
        parent = psutil.Process(child.pid)
        children = parent.children(recursive=True)
        processes = [*reversed(children), parent]
        for proc in processes:
            try:
                proc.terminate()
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(processes, timeout=3)
        for proc in alive:
            try:
                proc.kill()
            except psutil.NoSuchProcess:
                pass
    except psutil.NoSuchProcess:
        pass
    child.wait(timeout=10)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'run', 'control'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--phase', choices=LIMITS, required=True)
    p.add_argument('command', nargs=argparse.REMAINDER)
    a = p.parse_args(argv)
    root, run, control = (getattr(a, name).resolve() for name in ('root', 'run', 'control'))
    assert run.is_relative_to(root / 'outputs/rf_expanded_tuned_backtest')
    assert control.is_relative_to(root / 'outputs/rf_expanded_tuned_backtest_control')
    command = a.command[1:] if a.command[:1] == ['--'] else a.command
    assert len(command) >= 2 and Path(command[0]).absolute() == root / '.venv/bin/python'
    script = 'verify_rf_tuned_backtest.py' if a.phase == 'verify' else 'run_rf_tuned_backtest.py'
    assert Path(command[1]).resolve() == root / 'scripts' / script
    def bound_argument(name):
        assert command.count(name) == 1, 'Require one exact ' + name
        return (root / command[command.index(name) + 1]).resolve()
    assert bound_argument('--root') == root
    if a.phase != 'verify':
        assert command[2] == a.phase
    if a.phase == 'prepare':
        assert bound_argument('--output-directory') == run
    else:
        assert bound_argument('--run') == run
    if a.phase == 'verify':
        assert bound_argument('--backtest') == run / 'backtest'
        assert bound_argument('--output').is_relative_to(run)
    control.mkdir(parents=True, exist_ok=True)
    final = control / (a.phase + '.json')
    checkpoint = control / (a.phase + '_checkpoint.json')
    log = control / (a.phase + '.log')
    assert not final.exists() and not checkpoint.exists() and not log.exists(), 'Preserve previous phase attempt'
    initial = {'memory': memory(), 'pressure': pressure(),
               'free_disk_bytes': shutil.disk_usage(root).free}
    assert initial['memory']['available_proxy_bytes'] >= 7 * GIB, 'Wait for 7 GiB available memory'
    assert initial['pressure']['normal'], 'Wait for normal macOS pressure'
    assert initial['free_disk_bytes'] >= 30 * GIB, 'Wait for 30 GiB free disk'
    env = os.environ.copy()
    env.update({name: '1' for name in THREADS})
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(root / 'src'), PYTHONUNBUFFERED='1')
    started = time.monotonic()
    state = {'schema': 'dmv2-tuned-rf-replay-resource-supervision/v1',
             'phase': a.phase, 'status': 'RUNNING', 'root': str(root), 'run': str(run),
             'control': str(control), 'command': command, 'started_at': now(),
             'limits': {'wall_seconds': LIMITS[a.phase], 'process_tree_rss_stop_bytes': 4 * GIB,
                        'process_tree_rss_warn_bytes': 3 * GIB, 'output_stop_bytes': 256 * 1024**2,
                        'poll_seconds': 1, 'memory_pressure_dispatch_normal': 1},
             'initial_resources': initial, 'peak_process_tree_rss_bytes': 0,
             'new_output_bytes': files_size(run, control), 'failure': None}
    child = None
    try:
        with log.open('xb') as stream:
            child = subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                                     stdout=stream, stderr=subprocess.STDOUT)
            state['child_pid'] = child.pid
            print(json.dumps({'status': 'RUNNING', 'phase': a.phase, 'pid': child.pid,
                              'log': str(log)}), flush=True)
            while True:
                rss = psutil.Process().memory_info().rss
                try:
                    processes = [psutil.Process(child.pid)]
                    processes += processes[0].children(recursive=True)
                    for proc in processes:
                        try:
                            rss += proc.memory_info().rss
                        except psutil.NoSuchProcess:
                            pass
                except psutil.NoSuchProcess:
                    pass
                elapsed = time.monotonic() - started
                current = pressure()
                state.update(elapsed_seconds=elapsed, last_checked_at=now(),
                             process_tree_rss_bytes=rss, pressure=current,
                             new_output_bytes=files_size(run, control))
                state['peak_process_tree_rss_bytes'] = max(state['peak_process_tree_rss_bytes'], rss)
                state['rss_warning'] = rss >= 3 * GIB
                atomic(checkpoint, state)
                if not current['normal']:
                    raise RuntimeError('Non-normal macOS memory pressure')
                if rss >= 4 * GIB:
                    raise RuntimeError('Aggregate process-tree RSS budget exceeded')
                if elapsed > LIMITS[a.phase]:
                    raise RuntimeError('Phase wall-time budget exceeded')
                if state['new_output_bytes'] > 256 * 1024**2:
                    raise RuntimeError('New replay/control output budget exceeded')
                code = child.poll()
                if code is not None:
                    state['exit_code'] = code
                    if code:
                        raise RuntimeError('Reviewed phase exited with code ' + str(code))
                    break
                time.sleep(1)
        state.update(status='PASS_PROCESS_BUDGET', finished_at=now(),
                     elapsed_seconds=time.monotonic() - started)
    except BaseException as error:
        state.update(status='STOPPED_OR_FAILED', finished_at=now(),
                     elapsed_seconds=time.monotonic() - started,
                     failure={'type': type(error).__name__, 'message': str(error)})
        try:
            if child is not None and child.poll() is None:
                try:
                    stop_tree(child)
                except BaseException as termination_error:
                    state['termination_error'] = {'type': type(termination_error).__name__,
                                                  'message': str(termination_error)}
                    # Popen still identifies the owned child even if psutil
                    # cannot enumerate a process. Never signal a guessed PID.
                    try:
                        child.kill()
                        child.wait(timeout=10)
                    except BaseException as fallback_error:
                        state['termination_fallback_error'] = {
                            'type': type(fallback_error).__name__, 'message': str(fallback_error)}
        finally:
            atomic(final, state)
        raise
    atomic(final, state)
    print(json.dumps({'status': state['status'], 'phase': a.phase,
                      'elapsed_seconds': state['elapsed_seconds'],
                      'peak_process_tree_rss_bytes': state['peak_process_tree_rss_bytes'],
                      'receipt': str(final)}), flush=True)


if __name__ == '__main__':
    main()
