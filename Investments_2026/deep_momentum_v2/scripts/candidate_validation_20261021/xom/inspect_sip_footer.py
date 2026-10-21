#!/usr/bin/env python3
"""Generation-pinned SP1500 footer inspection; no column data reads."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import resource
import selectors
import struct
import subprocess
import sys
import time

import pyarrow.parquet as pq


def need(test, message):
    if not test:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pin(path):
    return {'path': str(Path(path).resolve()), 'size_bytes': Path(path).stat().st_size, 'sha256': sha(path)}


def check(item):
    p = Path(item['path'])
    need(p.stat().st_size == item['size_bytes'] and sha(p) == item['sha256'], 'Pinned input changed')
    return p


def write(path, value):
    need(not Path(path).exists(), 'Preserve earlier output')
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def rss():
    n = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(n if sys.platform == 'darwin' else n * 1024)


def bounded_cli(args, limit, timeout=90):
    """Bound stdout while running, with stderr retained privately only for classification."""
    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    sel = selectors.DefaultSelector()
    sel.register(p.stdout, selectors.EVENT_READ, 'out')
    sel.register(p.stderr, selectors.EVENT_READ, 'err')
    out, err = bytearray(), bytearray()
    deadline = time.monotonic() + timeout
    try:
        while sel.get_map():
            need(time.monotonic() < deadline, 'CLI timeout; no automatic retry')
            for key, _ in sel.select(timeout=min(1, max(0, deadline - time.monotonic()))):
                raw = os.read(key.fileobj.fileno(), 65536)
                if not raw:
                    sel.unregister(key.fileobj)
                elif key.data == 'out':
                    need(len(out) + len(raw) <= limit, 'Range/metadata response exceeded frozen payload bound')
                    out.extend(raw)
                else:
                    err.extend(raw[:max(0, 4096 - len(err))])
        p.wait(timeout=2)
        if p.returncode:
            private = bytes(err).decode(errors='replace').lower()
            kind = ('interactive_reauthentication_required' if any(s in private for s in
                ('reauthentication', 'invalid_grant', 'reauth', 'login required')) else
                'storage_permission_denied' if any(s in private for s in
                ('permission denied', 'accessdenied', 'forbidden', '403')) else 'cli_request_failed')
            raise ValueError(kind + '; credentials and raw stderr not logged')
        return bytes(out)
    finally:
        sel.close()
        if p.poll() is None:
            p.terminate()
            try:
                p.wait(timeout=2)
            except subprocess.TimeoutExpired:
                p.kill(); p.wait()
        p.stdout.close(); p.stderr.close()


def plain(value):
    if isinstance(value, bytes):
        return value.decode('utf-8', errors='strict')
    return str(value) if value is not None else None


def inspect_footer(footer, plan):
    need(len(footer) >= 8 and footer[-4:] == b'PAR1', 'Invalid Parquet footer magic')
    length = struct.unpack('<I', footer[-8:-4])[0]
    need(length + 8 == len(footer), 'Footer length mismatch')
    # This minimal envelope contains only the file magic and footer. Column
    # offsets remain original, and no read_row_group/read_table call occurs.
    pf = pq.ParquetFile(io.BytesIO(b'PAR1' + footer))
    names = pf.schema_arrow.names
    need(names == plan['expected_columns'], 'Original schema differs from pinned manifest')
    need(pf.metadata.num_rows == plan['object']['declared_rows'], 'Original footer row count differs')
    rows, candidates = [], []
    for index in range(pf.metadata.num_row_groups):
        group = pf.metadata.row_group(index)
        fields = {}
        safe_stats = {}
        for j in range(group.num_columns):
            col = group.column(j)
            name = col.path_in_schema
            fields[name] = {'compressed_bytes': col.total_compressed_size,
                'data_page_offset': col.data_page_offset,
                'dictionary_page_offset': col.dictionary_page_offset,
                'num_values': col.num_values}
            # Never touch statistics for any financial field.
            if name in ('security_id', 'trading_date'):
                st = col.statistics
                safe_stats[name] = {'has_min_max': bool(st is not None and st.has_min_max),
                    'min': plain(st.min) if st is not None and st.has_min_max else None,
                    'max': plain(st.max) if st is not None and st.has_min_max else None,
                    'null_count': st.null_count if st is not None else None}
        need(set(fields) == set(names), 'Nested/unexpected physical column layout')
        identity = safe_stats.get('security_id', {})
        dates = safe_stats.get('trading_date', {})
        relevant_identity = (not identity.get('has_min_max') or
            identity['min'] <= plan['security_id'] <= identity['max'])
        relevant_dates = (not dates.get('has_min_max') or not (
            dates['max'] < plan['start'] or dates['min'] > plan['end']))
        candidate = relevant_identity and relevant_dates
        entry = {'index': index, 'rows': group.num_rows, 'safe_metadata_statistics': safe_stats,
            'candidate_for_exact_source_date_metadata_pass': candidate,
            'metadata_projection_bytes': sum(fields[c]['compressed_bytes'] for c in plan['metadata_columns']),
            'ohlcv_projection_bytes_cost_only': sum(fields[c]['compressed_bytes'] for c in plan['financial_cost_columns']),
            'column_chunks': fields}
        rows.append(entry)
        if candidate:
            candidates.append(index)
    return {'schema': 'dmv2-xom-sip-footer-layout/v1', 'object': plan['object'],
        'physical_schema': [[name, str(pf.schema_arrow.field(name).type)] for name in names],
        'rows': pf.metadata.num_rows, 'row_groups': rows, 'candidate_row_groups': candidates,
        'candidate_metadata_projection_bytes': sum(r['metadata_projection_bytes'] for r in rows
            if r['candidate_for_exact_source_date_metadata_pass']),
        'candidate_ohlcv_projection_bytes_cost_only': sum(r['ohlcv_projection_bytes_cost_only'] for r in rows
            if r['candidate_for_exact_source_date_metadata_pass']),
        'financial_min_max_statistics_accessed': False, 'column_values_decoded': False,
        'financial_field_names_and_byte_sizes_are_planning_only': True}


def run(plan_path, launch_path, output):
    began = time.monotonic()
    plan, launch = (json.loads(Path(p).read_text()) for p in (plan_path, launch_path))
    need(plan['schema'] == 'dmv2-xom-sip-footer-plan/v1', 'Wrong footer plan')
    need(launch['schema'] == 'dmv2-xom-sip-footer-launch/v1' and launch['status'] == 'SEALED_BEFORE_REMOTE_READ', 'Footer read not sealed')
    need(launch['plan_sha256'] == sha(plan_path) and launch['worker_sha256'] == sha(__file__), 'Frozen worker/plan changed')
    parent = check(plan['parent_protocol_pin'])
    check(plan['source_manifest_pin'])
    need(launch['parent_protocol_sha256'] == sha(parent), 'Wrong candidate-validation authority')
    need(plan['max_combined_response_bytes'] <= 1024**2 and plan['column_data_reads_allowed'] == 0, 'Unbounded/nonmetadata footer scope')
    obj = plan['object']; need(obj['generation'] == 1783344954247444 and obj['size_bytes'] == 187908505, 'Wrong SIP object')
    out = Path(output).resolve()
    need(not out.exists(), 'Preserve prior attempt')
    out.mkdir(parents=True)
    write(out / 'reservation.json', {'plan_pin': pin(plan_path), 'launch_pin': pin(launch_path),
        'max_combined_response_bytes': plan['max_combined_response_bytes'], 'column_data_reads_allowed': 0})
    cmd = [plan['gcloud_binary'], '--configuration=deep-momentum-research',
        '--account=info@galamboscapital.com', '--project=trusty-agility-492109-v0', '--verbosity=error', 'storage']
    uri = obj['uri'] + '#' + str(obj['generation'])
    meta_raw = bounded_cli(cmd + ['objects', 'describe', uri, '--format=json'], plan['max_describe_bytes'])
    meta = json.loads(meta_raw)
    need(int(meta['generation']) == obj['generation'] and int(meta['size']) == obj['size_bytes'], 'Generation/size metadata mismatch')
    tail = bounded_cli(cmd + ['cat', '--range=' + str(obj['size_bytes'] - 8) + '-' + str(obj['size_bytes'] - 1), uri], 8)
    need(len(tail) == 8 and tail[4:] == b'PAR1', 'Invalid exact tail response')
    length = struct.unpack('<I', tail[:4])[0]
    total = length + 8
    need(0 < total <= plan['max_footer_bytes'] and len(meta_raw) + 8 + total <= plan['max_combined_response_bytes'], 'Footer exceeds frozen response bound')
    start = obj['size_bytes'] - total
    footer = bounded_cli(cmd + ['cat', '--range=' + str(start) + '-' + str(obj['size_bytes'] - 1), uri], total)
    need(len(footer) == total and footer[-8:] == tail, 'Footer response length/tail mismatch')
    result = inspect_footer(footer, plan)
    need(rss() <= plan['max_rss_bytes'] and time.monotonic() - began <= plan['max_wall_seconds'], 'Footer inspection resource limit exceeded')
    (out / 'footer.bin').write_bytes(footer)
    write(out / 'layout.json', result)
    receipt = {'schema': 'dmv2-xom-sip-footer-receipt/v1', 'status': 'PASS_FOOTER_ONLY',
        'plan_pin': pin(plan_path), 'worker_pin': pin(__file__), 'launch_pin': pin(launch_path),
        'object': obj, 'generation_and_size_verified': True, 'full_object_hash_repeated': False,
        'actual_cli_stdout_bytes': len(meta_raw) + len(tail) + len(footer),
        'ranges': [{'start': obj['size_bytes'] - 8, 'bytes': 8, 'sha256': hashlib.sha256(tail).hexdigest()},
            {'start': start, 'bytes': total, 'sha256': hashlib.sha256(footer).hexdigest()}],
        'layout_pin': pin(out / 'layout.json'), 'footer_pin': pin(out / 'footer.bin'),
        'column_data_reads': 0, 'financial_values_decoded': False,
        'financial_min_max_statistics_accessed': False, 'canonical_mapping_promoted': False,
        'candidate_row_groups': result['candidate_row_groups'], 'peak_rss_bytes': rss(),
        'elapsed_seconds': time.monotonic() - began}
    write(out / 'receipt.json', receipt)
    print(json.dumps({k: receipt[k] for k in ('status', 'actual_cli_stdout_bytes', 'candidate_row_groups')}))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True); parser.add_argument('--launch', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(); run(args.plan, args.launch, args.output)


if __name__ == '__main__':
    main()
