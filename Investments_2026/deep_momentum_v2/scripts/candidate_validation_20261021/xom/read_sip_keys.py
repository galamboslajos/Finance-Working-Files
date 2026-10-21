#!/usr/bin/env python3
"""Four-column SIP key pass; no price or volume values allowed."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import pyarrow as pa
import pyarrow.compute as pc

from range_source import AuthenticatedRanges, load_group


def need(test, message):
    if not test: raise ValueError(message)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pin(path):
    path = Path(path).resolve()
    return {'path': str(path), 'size_bytes': path.stat().st_size, 'sha256': sha(path)}


def checked(item):
    path = Path(item['path'])
    need(path.stat().st_size == item['size_bytes'] and sha(path) == item['sha256'], 'Pinned metadata changed')
    return path


def write(path, value):
    need(not Path(path).exists(), 'Preserve earlier output')
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def resources(begin):
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss = int(value if sys.platform == 'darwin' else value*1024)
    need(rss <= 2147483648 and time.monotonic()-begin <= 600, 'Key-pass resource budget exceeded')
    return rss


def select_keys(table, plan):
    candidate = table.filter(pc.equal(table['security_id'], plan['security_id']))
    for column in ('trading_date', 'query_symbol', 'adjustment'):
        need(candidate[column].null_count == 0, 'Null key for requested original source ID')
    if candidate.num_rows:
        dates = candidate['trading_date']
        need(pc.all(pc.match_substring_regex(dates, r'^\d{4}-\d{2}-\d{2}$')).as_py() is True,
            'Noncanonical source trading date')
        parsed = pc.strptime(dates, format='%Y-%m-%d', unit='s', error_is_null=False)
        need(pc.all(pc.equal(pc.strftime(parsed, format='%Y-%m-%d'), dates)).as_py() is True,
            'Invalid source trading date')
    mask = pc.and_(pc.equal(table['security_id'], plan['security_id']), pc.and_(
        pc.greater_equal(table['trading_date'], plan['start']),
        pc.less_equal(table['trading_date'], plan['end'])))
    positions = pc.indices_nonzero(mask)
    selected = table.take(positions)
    if selected.num_rows:
        need(pc.all(pc.equal(selected['query_symbol'], plan['query_symbol'])).as_py() is True,
            'Original source ID carries a conflicting query symbol')
        need(pc.all(pc.is_in(selected['adjustment'], value_set=pa.array(['raw', 'all']))).as_py() is True,
            'Unexpected source adjustment label')
    return selected, positions


def run(plan_path, launch_path, output):
    began = time.monotonic()
    plan, launch = (json.loads(Path(p).read_text()) for p in (plan_path, launch_path))
    need(plan['schema'] == 'dmv2-xom-sip-key-plan/v1', 'Wrong key-pass plan')
    need(launch['schema'] == 'dmv2-xom-sip-key-launch/v1' and launch['status'] == 'SEALED_BEFORE_REMOTE_READ', 'Key pass not sealed')
    need(launch['plan_sha256'] == sha(plan_path) and launch['worker_sha256'] == sha(__file__), 'Frozen key worker/plan changed')
    for item in plan['pins']: checked(item)
    need(plan['columns'] == ['security_id', 'trading_date', 'query_symbol', 'adjustment'], 'Forbidden key-pass columns')
    need(plan['projected_compressed_bytes'] <= plan['max_payload_bytes'] <= 134217728, 'Key read reservation exceeded')
    layout = json.loads(checked(plan['layout_pin']).read_text())
    footer = checked(plan['footer_pin']).read_bytes()
    obj = plan['object']; need(obj == layout['object'], 'Key source differs from inspected footer')
    groups = [g for g in layout['row_groups'] if g['index'] in plan['row_groups']]
    need(len(groups) == len(plan['row_groups']), 'Unknown source row group')
    actual_cost = sum(g['column_chunks'][c]['compressed_bytes'] for g in groups for c in plan['columns'])
    need(actual_cost == plan['projected_compressed_bytes'], 'Frozen key projection cost changed')
    out = Path(output).resolve(); need(not out.exists(), 'Preserve prior key attempt'); out.mkdir(parents=True)
    write(out/'reservation.json', {'plan_pin': pin(plan_path), 'launch_pin': pin(launch_path),
        'projected_compressed_bytes': actual_cost, 'financial_columns': []})
    pa.set_cpu_count(1); pa.set_io_thread_count(1)
    transport = AuthenticatedRanges(obj, max_bytes=plan['max_payload_bytes'])
    all_rows, selected_groups = [], []
    for group in groups:
        resources(began)
        table = load_group(footer, obj, group, plan['columns'], transport)
        selected, positions = select_keys(table, plan)
        need(len(all_rows) + selected.num_rows <= 10000, 'Selected-key bound exceeded before conversion')
        rows = selected.to_pylist()
        need(len(all_rows) + len(rows) <= 10000, 'Selected-key bound exceeded')
        if rows:
            selected_groups.append({'index': group['index'], 'positions': positions.to_pylist(), 'keys': rows})
            all_rows.extend(rows)
        del table, selected, positions
    keys = [(r['security_id'], r['trading_date'], r['adjustment']) for r in all_rows]
    need(len(keys) == len(set(keys)), 'Duplicate original source/date/adjustment keys; no deduplication allowed')
    need(all(r['trading_date'] < '2024-01-01' for r in all_rows), 'Protected key output outside scope')
    write(out/'selected_keys.json', sorted(all_rows, key=lambda r: (r['trading_date'], r['adjustment'])))
    write(out/'selected_positions.json', selected_groups)
    write(out/'ranges.json', transport.ranges)
    receipt = {'schema': 'dmv2-xom-sip-key-receipt/v1', 'status': 'PASS_METADATA_KEYS_ONLY',
        'plan_pin': pin(plan_path), 'worker_pin': pin(__file__), 'launch_pin': pin(launch_path),
        'helper_pin': pin(Path(__file__).with_name('range_source.py')),
        'selected_key_pin': pin(out/'selected_keys.json'), 'positions_pin': pin(out/'selected_positions.json'),
        'range_pin': pin(out/'ranges.json'), 'object': obj, 'selected_rows': len(all_rows),
        'selected_row_groups': [g['index'] for g in selected_groups],
        'projected_compressed_bytes': actual_cost, 'actual_http_range_payload_bytes': transport.total,
        'columns_decoded': plan['columns'], 'financial_values_decoded': False,
        'canonical_mapping_promoted': False, 'full_object_downloaded': False,
        'peak_rss_bytes': resources(began), 'elapsed_seconds': time.monotonic()-began}
    write(out/'receipt.json', receipt)
    print(json.dumps({k: receipt[k] for k in ('status', 'selected_rows', 'selected_row_groups', 'actual_http_range_payload_bytes')}))
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', required=True); p.add_argument('--launch', required=True)
    p.add_argument('--output', required=True); args = p.parse_args()
    run(args.plan, args.launch, args.output)


if __name__ == '__main__': main()
