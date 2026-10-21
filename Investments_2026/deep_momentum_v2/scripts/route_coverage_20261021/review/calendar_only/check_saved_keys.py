"""Output-only native/date and independently generated calendar metadata check."""
from __future__ import annotations
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '1'
import argparse
from collections import Counter
from datetime import date
import hashlib
from importlib.metadata import distribution, version
import json
from pathlib import Path
import resource
import sys
import time


def need(condition, message):
    if not condition:
        raise ValueError(message)


def pin(path):
    path = Path(path)
    return {'path': str(path.resolve()), 'size_bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def checked(item):
    path = Path(item['path'])
    need(pin(path) == item, 'Frozen metadata/code bytes changed: ' + str(path))
    return path


def run(plan_path):
    began = time.monotonic()
    plan = json.loads(Path(plan_path).read_text())
    need(plan['schema'] == 'dmv2-saved-key-calendar-plan/v1' and plan['status'] == 'FROZEN_OUTPUT_METADATA_ONLY', 'Frozen authority missing')
    need(plan['columns'] == ['intrinio_security_id', 'trading_date'], 'Native/date only')
    need(plan['date_start'] == '2007-01-01' and plan['date_end'] == '2023-12-31', 'Calendar scope widened')
    checked(plan['worker_pin'])
    for item in plan['input_pins'].values():
        checked(item)
    need(version('exchange-calendars') == plan['calendar_version'] == '4.13.2', 'Calendar package changed')
    for item in plan['calendar_code_pins']:
        checked(item)
    out = Path(plan['output'])
    need(not out.exists(), 'Preserve previous review attempt')
    out.mkdir()
    (out / 'reservation.json').write_text(json.dumps({'plan_pin': pin(plan_path), 'no_financial_values': True,
        'new_saved_key_bytes': plan['input_pins']['date_keys']['size_bytes'], 'production_data_objects': 0,
        'purpose': 'Current rule-based calendar reconciliation only; not knowledge-time identity certification'}, sort_keys=True, indent=2) + '\n')
    import pyarrow as pa
    import pyarrow.parquet as pq
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    table = pq.read_table(plan['input_pins']['date_keys']['path'], columns=plan['columns'], use_threads=False)
    need(table.num_rows <= plan['maximum_saved_key_rows'] and table.column_names == plan['columns'], 'Saved metadata scope exceeded')
    need(all(table[c].null_count == 0 for c in table.column_names), 'Null saved keys')
    rows = table.to_pylist()
    ids = {'sec_z5YpGE', 'sec_gA3NYQ', 'sec_zq1J05'}
    keys = [(r['intrinio_security_id'], r['trading_date']) for r in rows]
    need(len(keys) == len(set(keys)) and all(n in ids and plan['date_start'] <= d <= plan['date_end'] and date.fromisoformat(d).isoformat() == d for n, d in keys), 'Unexpected saved key')
    import exchange_calendars as xcals
    cal = xcals.get_calendar('XNYS', start=plan['date_start'], end=plan['date_end'])
    need(type(cal).__module__ == 'exchange_calendars.exchange_calendar_xnys' and cal.name == 'XNYS', 'Calendar implementation changed')
    sessions = sorted(d for d in cal.sessions.strftime('%Y-%m-%d') if plan['date_start'] <= d <= plan['date_end'])
    dates = set(sessions)
    cases = json.loads(Path(plan['input_pins']['cases']['path']).read_text())
    windows = json.loads(Path(plan['input_pins']['source_windows']['path']).read_text())
    anchors = sorted({w['signal_date'] for w in windows} | {w['outcome_end'] for w in windows})
    need(len(windows) == 168 and len(anchors) == 169, 'Original window scope changed')
    checks = []
    for c in cases:
        native = c['native_security_id']
        observed = {d for n, d in keys if n == native}
        counts = dict(sorted(Counter(d[:4] for d in observed).items()))
        need(len(observed) == c['date_key_rows'] and counts == c['year_key_counts'], 'Saved per-native count mismatch')
        need((min(observed) if observed else None) == c['first_date_key'] and (max(observed) if observed else None) == c['last_date_key'], 'Saved boundary mismatch')
        need(sum(d in observed for d in anchors) == c['matched_anchor_dates'] and [d for d in anchors if d not in observed] == c['missing_anchor_dates'], 'Saved anchor mismatch')
        for w, stored in zip(windows, c['month_endpoints'], strict=True):
            expected = {'outcome_month': w['outcome_month'], 'signal_date': w['signal_date'], 'outcome_end': w['outcome_end'], 'start_key_exists': w['signal_date'] in observed, 'end_key_exists': w['outcome_end'] in observed}
            need(stored == expected, 'Saved exact endpoint result mismatch')
        checks.append({'native_security_id': native, 'ticker_label': c['ticker_label'], 'role': c['role'],
            'saved_date_key_rows': len(observed), 'calendar_sessions': len(dates),
            'matched_calendar_sessions': len(observed & dates), 'missing_calendar_sessions': sorted(dates - observed),
            'non_calendar_dates': sorted(observed - dates), 'matched_original_anchors': sum(d in observed for d in anchors),
            'promotion_or_identity_certification': False})
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
    need(rss <= plan['max_rss_bytes'] and time.monotonic() - began <= plan['max_wall_seconds'], 'Review resource bound exceeded')
    result = {'schema': 'dmv2-saved-key-calendar-result/v1', 'status': 'PASS_OUTPUT_AND_CALENDAR_METADATA_ONLY',
        'plan_pin': pin(plan_path), 'calendar_version': version('exchange-calendars'),
        'calendar_scope': [plan['date_start'], plan['date_end']], 'calendar_sessions': len(sessions),
        'calendar_dates_sha256': hashlib.sha256(json.dumps(sessions, separators=(',', ':')).encode()).hexdigest(),
        'cases': checks, 'saved_keys_loaded': len(keys), 'source_data_objects_loaded': 0,
        'financial_values_read': False, 'cloud_reads': False, 'production_mutations': False,
        'peak_rss_bytes': rss, 'elapsed_seconds': time.monotonic() - began,
        'limitations': ['Current installed independent rule-based XNYS session metadata; not contemporaneous knowledge-time certification.',
            'Key completeness does not establish historically correct identity, valid OHLCV, corporate-action units, tradability or fills.',
            'No original source data columns were read; only the new saved date-key artifact was decoded.']}
    for item in plan['input_pins'].values():
        checked(item)
    for item in plan['calendar_code_pins']:
        checked(item)
    text = json.dumps(result, sort_keys=True, indent=2) + '\n'
    need(len(text.encode()) <= plan['max_output_bytes'], 'Output metadata bound exceeded')
    (out / 'result.json').write_text(text)
    print(json.dumps({'status': result['status'], 'calendar_sessions': len(sessions), 'cases': [{k: v for k, v in c.items() if k not in ['missing_calendar_sessions', 'non_calendar_dates']} | {'missing_calendar_sessions': len(c['missing_calendar_sessions']), 'non_calendar_dates': c['non_calendar_dates']} for c in checks]}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    run(parser.parse_args().plan)
