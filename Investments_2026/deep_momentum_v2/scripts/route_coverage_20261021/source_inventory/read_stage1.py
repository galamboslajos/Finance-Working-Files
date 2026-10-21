#!/usr/bin/env python3
"""Read frozen local identity/acquisition metadata; produce unpromoted candidates."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import resource
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq


FORBIDDEN = {'open', 'high', 'low', 'close', 'volume', 'adj_open', 'adj_high', 'adj_low',
    'adj_close', 'adj_volume', 'factor', 'dividend', 'split_ratio', 'provider_adjusted_return',
    'label_return', 'target_class', 'change', 'percent_change', 'raw_high_gap', 'raw_low_gap'}
TABLES = {'security_master_siblis', 'security_identifier_history', 'security_identity_bridge',
    'intrinio_identity_links', 'intrinio_security_master', 'identifier_resolution_evidence'}


def need(test, message):
    if not test:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def pin(path):
    path = Path(path).resolve()
    return {'path': str(path), 'size_bytes': path.stat().st_size, 'sha256': sha(path)}


def checked(item):
    path = Path(item['path']).resolve()
    need(path.is_file() and path.stat().st_size == item['size_bytes'], 'Metadata source size changed')
    need(sha(path) == item['sha256'], 'Metadata source bytes changed')
    return path


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def base_ticker(value):
    return re.split(r'[:.\-\s]', (value or '').upper().strip())[0]


def aliases(value):
    if value in (None, ''):
        return set()
    result = json.loads(value)
    need(isinstance(result, list) and all(isinstance(x, str) for x in result), 'Malformed source alias list')
    return {x.upper().strip() for x in result if x.strip()}


def candidates(tables, cases):
    outputs = []
    # A direct-ID short circuit must not turn malformed source alias evidence
    # into a silently accepted candidate record.
    for row in tables['intrinio_identity_links']:
        aliases(row.get('observed_tickers_json'))
        aliases(row.get('observed_isins_json'))
    for case in cases:
        source_id, label, original = case['source_security_id'], case['ticker_label'], case['native_security_id']
        source = [r for r in tables['security_master_siblis'] if r['security_id'] == source_id]
        history = [r for r in tables['security_identifier_history'] if r['security_id'] == source_id]
        isins = {r['isin'].upper() for r in source if r.get('isin')}
        isins.update(r['identifier_value'].upper() for r in history if
            (r.get('identifier_type') or '').lower() == 'isin' and r.get('identifier_value'))
        exact_bridge = [r for r in tables['security_identity_bridge']
            if r['source_product_id'] == 'sp_composite1500' and r['source_security_id'] == source_id]
        bridge = [r for r in tables['security_identity_bridge'] if r in exact_bridge
            or (r.get('isin') or '').upper() in isins or base_ticker(r.get('primary_ticker')) == label]
        uids = {r['unified_security_id'] for r in bridge}
        links = [r for r in tables['intrinio_identity_links'] if
            r['unified_security_id'] in uids or r.get('intrinio_security_id') == original
            or (r.get('query_identifier') or '').upper() in isins
            or base_ticker(r.get('query_identifier')) == label
            or any(base_ticker(t) == label for t in aliases(r.get('observed_tickers_json')))
            or bool(aliases(r.get('observed_isins_json')) & isins)]
        native_ids = {original} | {r['intrinio_security_id'] for r in links if r.get('intrinio_security_id')}
        direct_master = [r for r in tables['intrinio_security_master'] if
            r['intrinio_security_id'] in native_ids or base_ticker(r.get('ticker')) == label
            or base_ticker(r.get('composite_ticker')) == label
            or (r.get('requested_identifier') or '').upper() in isins]
        company_ids = {r['company_id'] for r in direct_master if r.get('company_id')}
        figis = {r[k] for r in direct_master for k in ('figi', 'composite_figi', 'share_class_figi') if r.get(k)}
        master = [r for r in tables['intrinio_security_master'] if r in direct_master
            or (r.get('company_id') and r['company_id'] in company_ids)
            or any(r.get(k) in figis for k in ('figi', 'composite_figi', 'share_class_figi'))]
        native_ids.update(r['intrinio_security_id'] for r in master)
        need(len(native_ids) <= 200 and len(master) <= 200 and len(links) <= 200, 'Unbounded identity candidate expansion')
        evidence = [r for r in tables['identifier_resolution_evidence'] if
            r.get('intrinio_security_id') in native_ids or r.get('requested_identifier') in native_ids]
        summaries = []
        for native in sorted(native_ids):
            provider = [r for r in master if r['intrinio_security_id'] == native]
            records = [r for r in evidence if r.get('intrinio_security_id') == native
                or r.get('requested_identifier') == native]
            summaries.append({'intrinio_security_id': native, 'is_current_published_route': native == original,
                'provider_master_rows': len(provider),
                'provider_tickers': sorted({r['ticker'] for r in provider if r.get('ticker')}),
                'reported_acquisition_rows': [r.get('price_row_count') for r in records],
                'reported_http_statuses': [r.get('http_status') for r in records],
                'reported_acquisition_statuses': [r.get('status') for r in records],
                'qualification': 'CANDIDATE_ONLY_NOT_PROMOTED',
                'exact_date_history_coverage_verified': False,
                'independent_source_isin_provider_identity_verified': False})
        outputs.append({**case, 'source_master_rows': source, 'source_identifier_history': history,
            'exact_sp1500_bridge_rows': exact_bridge, 'related_source_alias_bridge_rows': bridge,
            'related_published_identity_links': links, 'provider_master_candidate_rows': master,
            'acquisition_candidate_rows': evidence, 'candidate_native_ids': sorted(native_ids),
            'candidate_summaries': summaries, 'canonical_mapping_promoted': False,
            'source_alias_agreement_is_independent_confirmation': False,
            'price_history_absence_proven': False,
            'no_original_lookup_response_payload_opened': True})
    return outputs


def peak_rss():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == 'darwin' else value * 1024)


def run(plan_path, config_path, output):
    begin = time.monotonic()
    plan, cfg = json.loads(Path(plan_path).read_text()), json.loads(Path(config_path).read_text())
    need(plan['schema'] == 'dmv2-xom-bk-source-stage1-plan/v1', 'Wrong stage1 plan')
    need(cfg['schema'] == 'dmv2-xom-bk-source-stage1-config/v1', 'Wrong stage1 config')
    need(cfg['worker_sha256'] == sha(__file__) and cfg['plan_sha256'] == sha(plan_path), 'Frozen worker/plan changed')
    protocol_path = checked(plan['protocol_pin']); protocol = json.loads(protocol_path.read_text())
    need(cfg['protocol_sha256'] == plan['protocol_pin']['sha256'], 'Wrong master metadata authority')
    need(protocol['status'] == 'FROZEN_METADATA_ONLY', 'Unfrozen master scope')
    limits = protocol['budget']
    need(plan['cases'] == protocol['cases'], 'Case scope differs from master protocol')
    need(plan['projected_compressed_bytes'] <= limits['max_projected_column_bytes']
        and plan['physical_rows'] <= limits['max_candidate_rows']
        and plan['returned_metadata_rows_upper_bound'] <= limits['max_returned_metadata_rows']
        and plan['data_files'] <= limits['max_data_files']
        and plan['footer_files'] <= limits['max_footer_files'], 'Unapproved metadata read bound')
    need({t['table'] for t in plan['tables']} == TABLES and len(plan['tables']) == 6, 'Need exact six metadata-only tables')
    output = Path(output).resolve()
    need(not output.exists(), 'Preserve previous stage attempt')
    output.mkdir(parents=True)
    write(output / 'read_reservation.json', {'schema': 'dmv2-xom-bk-stage1-read-reservation/v1',
        'plan_pin': pin(plan_path), 'config_pin': pin(config_path),
        'projected_compressed_bytes': plan['projected_compressed_bytes'],
        'candidate_rows_upper_bound': plan['physical_rows'], 'data_files': plan['data_files'],
        'financial_columns': [], 'tables': plan['tables']})
    checked(plan['planner_pin']); checked(plan['footer_helper_pin']); checked(plan['source_config_pin'])
    for item in plan['manifest_pins']:
        checked(item)
    for table in plan['tables']:
        need(not set(table['columns']) & FORBIDDEN, 'Financial column requested as metadata')
        for obj in table['objects']:
            path = checked(obj)
            with path.open('rb') as handle:
                handle.seek(obj['footer_offset']); raw = handle.read(obj['footer_size_bytes'])
            need(hashlib.sha256(raw).hexdigest() == obj['footer_sha256'], 'Pinned physical footer changed')
            pf = pq.ParquetFile(path)
            need(pf.metadata.num_rows == obj['rows'], 'Source metadata footer rows changed')
            need([[name, str(pf.schema_arrow.field(name).type)] for name in table['columns']] == obj['schema'], 'Projection schema changed')
    pa.set_cpu_count(1); pa.set_io_thread_count(1)
    tables = {}
    returned = 0
    for table in plan['tables']:
        need(time.monotonic() - begin <= limits['max_wall_seconds'] and peak_rss() <= limits['max_rss_bytes'], 'Stage1 resource bound exceeded')
        rows = []
        for obj in table['objects']:
            rows.extend(pq.ParquetFile(obj['path']).read(columns=table['columns'], use_threads=False).to_pylist())
        need(len(rows) == table['physical_rows'], 'Metadata conversion changed planned row count')
        returned += len(rows)
        need(returned <= limits['max_returned_metadata_rows'] and peak_rss() <= limits['max_rss_bytes'], 'Returned metadata/RSS limit exceeded')
        tables[table['table']] = rows
    result_cases = candidates(tables, plan['cases'])
    write(output / 'cases.json', result_cases)
    receipt = {'schema': 'dmv2-xom-bk-source-stage1-result/v1', 'status': 'PASS_METADATA_CANDIDATE_INVENTORY',
        'protocol_pin': plan['protocol_pin'], 'worker_pin': pin(__file__), 'plan_pin': pin(plan_path),
        'configuration_pin': pin(config_path), 'cases_pin': pin(output / 'cases.json'),
        'candidate_rows_considered': plan['physical_rows'], 'metadata_rows_converted': returned,
        'projected_compressed_bytes': plan['projected_compressed_bytes'],
        'source_data_files': plan['data_files'], 'source_footers': plan['footer_files'],
        'peak_rss_bytes': peak_rss(), 'elapsed_seconds': time.monotonic() - begin,
        'financial_columns_decoded': [], 'price_key_inventory_executed': False,
        'cloud_reads': False, 'canonical_mappings_promoted': False,
        'models_rebuilt_or_loaded': False, 'source_or_old_run_mutations': False,
        'case_summary': [{'ticker_label': c['ticker_label'], 'candidate_native_ids': c['candidate_native_ids'],
            'candidates': c['candidate_summaries']} for c in result_cases],
        'limitations': protocol['claim_limits']}
    need(receipt['elapsed_seconds'] <= limits['max_wall_seconds'] and receipt['peak_rss_bytes'] <= limits['max_rss_bytes'], 'Final metadata resource limit exceeded')
    need(sum(p.stat().st_size for p in output.iterdir()) <= limits['max_output_bytes'], 'Output metadata budget exceeded')
    write(output / 'receipt.json', receipt)
    print(json.dumps(receipt['case_summary']))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True); parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(args.plan, args.config, args.output)


if __name__ == '__main__':
    main()
