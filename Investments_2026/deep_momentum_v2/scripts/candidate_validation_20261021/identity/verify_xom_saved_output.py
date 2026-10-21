"""Independent bounded QA of a newly saved XOM audit output, never sources."""
from __future__ import annotations
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '1'
import argparse
from datetime import date, datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import time

BASE = Path('/private/tmp/dmv2-candidate-validation-20261021')
SOURCE = 'siblis_isin_b284bb7497616f5afba0'
KEYS = ['security_id', 'trading_date', 'query_symbol', 'adjustment']
NUMERIC = ['open', 'high', 'low', 'close', 'volume']
TOL = .0002


def need(value, message):
    if not value:
        raise ValueError(message)


def pin(path):
    path = Path(path).resolve()
    return {'path': str(path), 'size_bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def checked(item):
    p = Path(item['path'])
    need(p.is_file() and not p.is_symlink() and pin(p) == item, 'Frozen saved metadata/output changed')
    return p


def metadata_guard(rows, sessions):
    known = set(sessions)
    seen = set()
    need(len(rows) <= 10000, 'Saved metadata row cap exceeded')
    for r in rows:
        d = r['trading_date']
        need(isinstance(d, str) and date.fromisoformat(d).isoformat() == d, 'Noncanonical date')
        need(r['security_id'] == SOURCE and r['query_symbol'] == 'XOM' and r['adjustment'] in ('raw', 'all')
             and '2016-01-04' <= d <= '2023-12-31' and d in known, 'Unapproved source/date/mode/session before numeric conversion')
        key = (d, r['adjustment'])
        need(key not in seen, 'Duplicate saved pair key')
        seen.add(key)


def finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def calculate(rows, sessions):
    # Metadata is audited for every supplied row before touching a numeric cell.
    metadata_guard(rows, sessions)
    lookup = {(r['trading_date'], r['adjustment']): r for r in rows}
    nulls = {m: {f: 0 for f in NUMERIC} for m in ('raw', 'all')}
    nonpositive = {m: {f: 0 for f in NUMERIC[:4]} for m in ('raw', 'all')}
    negative_vol = {'raw': 0, 'all': 0}
    zero_vol = {'raw': 0, 'all': 0}
    missing = {m: [] for m in ('raw', 'all')}
    bad_bounds = {m: [] for m in ('raw', 'all')}
    scale_bad = []
    per_date_flags = {}
    all_errors = []
    differing_volume_dates = []
    observed_pairs = 0
    for d in sessions:
        flags = []
        for m in ('raw', 'all'):
            r = lookup.get((d, m))
            if r is None:
                missing[m].append(d)
                flags.append('missing_' + m + '_pair_row')
                continue
            for f in NUMERIC:
                nulls[m][f] += not finite(r[f])
            for f in NUMERIC[:4]:
                nonpositive[m][f] += finite(r[f]) and r[f] <= 0
            negative_vol[m] += finite(r['volume']) and r['volume'] < 0
            zero_vol[m] += finite(r['volume']) and r['volume'] == 0
            good_prices = all(finite(r[f]) and r[f] > 0 for f in NUMERIC[:4])
            if not good_prices:
                flags.append(m + '_OHLC_nonpositive_or_nonfinite')
            else:
                if r['low'] > min(r['open'], r['close']) or r['high'] < max(r['open'], r['close']) or r['low'] > r['high']:
                    bad_bounds[m].append(d)
                    flags.append(m + '_OHLC_bracket_invalid')
            if not finite(r['volume']) or r['volume'] < 0:
                flags.append(m + '_volume_negative_or_nonfinite')
        raw, adj = lookup.get((d, 'raw')), lookup.get((d, 'all'))
        if raw is not None and adj is not None:
            observed_pairs += 1
            if all(finite(r[f]) and r[f] > 0 for r in (raw, adj) for f in NUMERIC[:4]):
                common = adj['close'] / raw['close']
                error = max(abs((adj[f] / raw[f]) / common - 1.) for f in NUMERIC[:4])
                all_errors.append(error)
                if error > TOL:
                    scale_bad.append(d)
                    flags.append('raw_all_OHLC_price_scales_disagree')
            else:
                scale_bad.append(d)
                flags.append('raw_all_OHLC_price_scales_disagree')
            if finite(raw['volume']) and finite(adj['volume']) and raw['volume'] >= 0 and adj['volume'] >= 0 and raw['volume'] != adj['volume']:
                differing_volume_dates.append(d)
        per_date_flags[d] = sorted(flags)
    jumps = []
    def positive_ratio(a, b):
        return a / b if finite(a) and finite(b) and a > 0 and b > 0 else None
    for previous, current in zip(sessions, sessions[1:]):
        def close(d, mode):
            found = lookup.get((d, mode))
            return found['close'] if found else None
        rr = positive_ratio(close(current, 'raw'), close(previous, 'raw'))
        ar = positive_ratio(close(current, 'all'), close(previous, 'all'))
        previous_scale = positive_ratio(close(previous, 'all'), close(previous, 'raw'))
        current_scale = positive_ratio(close(current, 'all'), close(current, 'raw'))
        fc = positive_ratio(current_scale, previous_scale)
        raw_change, all_change, scale_change = [v - 1. if v is not None else None for v in (rr, ar, fc)]
        if any(v is not None and abs(v) > threshold for v, threshold in
               [(raw_change, .5), (all_change, .5), (scale_change, .05)]):
            jumps.append({'previous_session': previous, 'trading_date': current,
                          'raw_close_change': raw_change, 'all_close_change': all_change,
                          'paired_price_factor_change': scale_change})
    return {'rows': len(rows), 'expected_sessions': len(sessions), 'complete_pairs': observed_pairs,
            'missing_dates': missing, 'null_or_nonfinite': nulls, 'nonpositive_OHLC': nonpositive,
            'negative_volume': negative_vol, 'zero_volume': zero_vol, 'OHLC_bracket_dates': bad_bounds,
            'price_scale_failure_dates': scale_bad,
            'maximum_common_scale_relative_residual': max(all_errors) if all_errors else None,
            'differing_volume_dates': differing_volume_dates, 'per_date_flags': per_date_flags,
            'consecutive_session_discontinuities': jumps}


def prepare(receipt_path, plan_path, output):
    receipt_path = Path(receipt_path).resolve()
    need(receipt_path.is_relative_to(BASE / 'xom') and receipt_path.name == 'receipt.json', 'New XOM output root only')
    r = json.loads(receipt_path.read_text())
    need(r['schema'] == 'dmv2-xom-sip-financial-receipt/v1' and r['status'] == 'PASS_FIELD_AUDIT_COMPLETED', 'Successful new XOM receipt required')
    q = checked(r['quotes_pin'])
    need(q.parent == receipt_path.parent and q.name == 'quotes.parquet' and q.stat().st_size < 8 * 1024 * 1024, 'Exact bounded new quote artifact required')
    inputs = {'receipt': pin(receipt_path), 'quotes': r['quotes_pin'], 'summary': r['summary_pin'],
              'row_quality': r['row_quality_pin'], 'sessions': r['sessions_pin']}
    for item in inputs.values():
        checked(item)
    plan = {'schema': 'dmv2-XOM-independent-output-plan/v1', 'status': 'SEALED_BEFORE_SAVED_FINANCIAL_DECODE',
            'frozen_at_utc': datetime.now(timezone.utc).isoformat(), 'worker_pin': pin(__file__), 'inputs': inputs,
            'max_rss_bytes': 2147483648, 'max_wall_seconds': 60, 'max_output_bytes': 1048576,
            'max_saved_rows': 10000, 'price_scale_relative_tolerance': TOL,
            'output': str(Path(output).resolve()), 'original_source_files_read': 0, 'network_calls': 0}
    p = Path(plan_path)
    need(not p.exists(), 'Preserve earlier verification plan')
    p.write_text(json.dumps(plan, sort_keys=True, indent=2) + '\n')
    print(json.dumps({'sealed_plan_pin': pin(p)}))


def run(plan_path):
    began = time.monotonic()
    p = Path(plan_path)
    plan = json.loads(p.read_text())
    need(plan['schema'] == 'dmv2-XOM-independent-output-plan/v1' and plan['status'] == 'SEALED_BEFORE_SAVED_FINANCIAL_DECODE', 'Sealed verification scope missing')
    checked(plan['worker_pin'])
    for item in plan['inputs'].values():
        checked(item)
    r = json.loads(checked(plan['inputs']['receipt']).read_text())
    need(r['status'] == 'PASS_FIELD_AUDIT_COMPLETED' and r['quotes_pin'] == plan['inputs']['quotes']
         and r['summary_pin'] == plan['inputs']['summary'] and r['row_quality_pin'] == plan['inputs']['row_quality']
         and r['sessions_pin'] == plan['inputs']['sessions'], 'Core receipt relationship changed')
    q = Path(plan['inputs']['quotes']['path'])
    need(q.resolve().is_relative_to(BASE / 'xom') and q.name == 'quotes.parquet', 'Output artifact only')
    calendar = json.loads(Path(plan['inputs']['sessions']['path']).read_text())
    need(calendar['package_version'] == '4.13.2' and calendar['calendar_implementation'] == 'exchange_calendars.exchange_calendar_xnys', 'Pinned XNYS calendar required')
    sessions = [d for d in calendar['sessions'] if '2016-01-04' <= d <= '2023-12-31']
    need(sessions == sorted(set(sessions)) and len(sessions) == 2012, 'Exact 2016-2023 independent calendar changed')
    out = Path(plan['output'])
    need(not out.exists(), 'Preserve earlier verification attempt')
    out.mkdir(parents=True)
    (out / 'reservation.json').write_text(json.dumps({'plan_pin': pin(p), 'saved_quote_bytes': q.stat().st_size,
        'columns': KEYS + NUMERIC, 'original_source_objects': 0, 'network_calls': 0}, indent=2) + '\n')
    import pyarrow as pa
    import pyarrow.parquet as pq
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    meta = pq.read_table(q, columns=KEYS, use_threads=False)
    need(meta.num_rows <= 10000, 'Saved metadata cap exceeded')
    metadata_guard(meta.to_pylist(), sessions)
    full = pq.read_table(q, columns=KEYS + NUMERIC, use_threads=False)
    need(full.num_rows == meta.num_rows, 'Saved projection count changed')
    observed = calculate(full.to_pylist(), sessions)
    summary = json.loads(Path(plan['inputs']['summary']['path']).read_text())
    quality = json.loads(Path(plan['inputs']['row_quality']['path']).read_text())
    need(observed['rows'] == r['financial_python_rows'] == r['session_rows_audited'] == summary['input_rows'], 'Saved numerical row counts disagree')
    need(observed['expected_sessions'] == summary['expected_XNYS_sessions'] and observed['complete_pairs'] == summary['complete_raw_all_pairs'], 'Pair coverage differs')
    need(observed['missing_dates']['raw'] == summary['missing_raw_dates'] and observed['missing_dates']['all'] == summary['missing_all_dates'], 'Missing-pair dates differ')
    expected_flags = {v['trading_date']: sorted(v['flags']) for v in quality}
    need(expected_flags == observed['per_date_flags'], 'Per-date quality flags differ')
    need(len(observed['differing_volume_dates']) == summary['volume_difference_pair_dates'], 'Paired volume differences disagree')
    expected_jumps = summary['flagged_consecutive_session_discontinuities']
    need([(r['previous_session'], r['trading_date']) for r in observed['consecutive_session_discontinuities']] ==
         [(r['previous_session'], r['trading_date']) for r in expected_jumps], 'Consecutive-session jump dates differ')
    for observed_jump, expected_jump in zip(observed['consecutive_session_discontinuities'], expected_jumps):
        for field in ('raw_close_change', 'all_close_change', 'paired_price_factor_change'):
            a, b = observed_jump[field], expected_jump[field]
            need(a is None and b is None or a is not None and b is not None and math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-13),
                 'Consecutive-session jump values differ')
    a, b = observed['maximum_common_scale_relative_residual'], summary['max_OHLC_common_scale_relative_residual']
    need(a is None and b is None or a is not None and b is not None and math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-13), 'Common-factor residual differs')
    for item in plan['inputs'].values():
        checked(item)
    checked(plan['worker_pin'])
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
    need(rss <= plan['max_rss_bytes'] and time.monotonic() - began <= plan['max_wall_seconds'], 'Output-only resource cap exceeded')
    result = {'schema': 'dmv2-XOM-independent-saved-output-review/v1', 'status': 'PASS_INDEPENDENT_SAVED_OUTPUT_RECONCILIATION',
        'plan_pin': pin(p), 'inputs': plan['inputs'], 'observed': observed,
        'original_source_financial_reads': 0, 'network_calls': 0, 'fits_predictions_backtests': 0,
        'mapping_promoted': False, 'action_total_return_identity_or_knowledge_time_certified': False,
        'volume_inverse_price_factor_assumed': False, 'peak_rss_bytes': rss, 'elapsed_seconds': time.monotonic() - began,
        'limitations': ['Valid raw/all pairs and coherent fields do not certify complete action histories or economic total returns.',
            'Provider-only rows are prohibited before financial Python conversion; this verifier is scoped to session-only saved output.',
            'No earlier-than-2016 source history is established, and no data is attached to production research cohorts.']}
    text = json.dumps(result, sort_keys=True, indent=2) + '\n'
    need(len(text.encode()) <= plan['max_output_bytes'], 'Output review size cap exceeded')
    (out / 'result.json').write_text(text)
    print(json.dumps({k: result[k] for k in ['status', 'original_source_financial_reads', 'network_calls']} | {'rows': observed['rows'], 'pairs': observed['complete_pairs']}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--prepare-receipt')
    p.add_argument('--plan', required=True)
    p.add_argument('--output')
    a = p.parse_args()
    if a.prepare_receipt:
        need(a.output, 'Preparation requires a fresh output directory')
        prepare(a.prepare_receipt, a.plan, a.output)
    else:
        run(a.plan)
