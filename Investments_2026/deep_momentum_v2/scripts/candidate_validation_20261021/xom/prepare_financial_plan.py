#!/usr/bin/env python3
"""Prepare metadata-only exact XOM financial projection after the sealed key pass."""
import argparse
import json
from pathlib import Path
from read_sip_keys import checked, pin, write, need

FINANCIAL = ['open', 'high', 'low', 'close', 'volume']
PROVENANCE = ['timestamp', 'feed', 'timeframe', 'asof', 'source_run_id',
              'source_raw_run_root', 'source_task_id', 'source_page_hash', 'source_request_id']


def prepare(key_receipt, qa_path, sessions_path, output):
    key_pin = pin(key_receipt); key = json.loads(checked(key_pin).read_text())
    need(key['status'] == 'PASS_METADATA_KEYS_ONLY' and not key['financial_values_decoded'], 'Key pass must succeed first')
    key_plan = json.loads(checked(key['plan_pin']).read_text())
    layout = json.loads(checked(key_plan['layout_pin']).read_text())
    positions = json.loads(checked(key['positions_pin']).read_text())
    checked(key['selected_key_pin']); checked(key['range_pin']); checked(key['helper_pin'])
    need([g['index'] for g in positions] == key['selected_row_groups'], 'Selected row-group receipt differs')
    need(sum(len(g['positions']) for g in positions) == key['selected_rows'] <= 10000, 'Selected positions exceed scope')
    sessions = json.loads(checked(pin(sessions_path)).read_text())
    need(sessions['sessions'] and sessions['sessions'] == sorted(set(sessions['sessions'])), 'Independent session calendar invalid')
    session_set = {d for d in sessions['sessions'] if key_plan['start'] <= d <= key_plan['end']}
    financial_groups = [g['index'] for g in positions if any(k['trading_date'] in session_set for k in g['keys'])]
    financial_rows = sum(k['trading_date'] in session_set for g in positions for k in g['keys'])
    groups = [g for g in layout['row_groups'] if g['index'] in financial_groups]
    need(groups and financial_rows, 'No exact source session row groups')
    columns = PROVENANCE + FINANCIAL
    cost = sum(g['column_chunks'][c]['compressed_bytes'] for g in groups for c in columns)
    footer_receipt = json.loads(checked(key_plan['pins'][1]).read_text())
    prior = footer_receipt['actual_cli_stdout_bytes'] + key['actual_http_range_payload_bytes']
    need(prior + cost <= 134217728, 'Cumulative XOM read cap exceeded; no automatic scope expansion')
    need(key['object'] == layout['object'] == footer_receipt['object'], 'Pinned source objects disagree')
    plan = {'schema': 'dmv2-xom-sip-financial-plan/v1', 'status': 'READY_FOR_ROOT_SEAL_FINANCIAL_READ',
        'object': key['object'], 'security_id': key_plan['security_id'], 'query_symbol': key_plan['query_symbol'],
        'start': key_plan['start'], 'end': key_plan['end'], 'key_receipt_pin': key_pin,
        'positions_pin': key['positions_pin'], 'selected_key_pin': key['selected_key_pin'],
        'footer_pin': key_plan['footer_pin'], 'layout_pin': key_plan['layout_pin'],
        'helper_pin': key['helper_pin'], 'key_worker_pin': key['worker_pin'],
        'qa_pin': pin(qa_path), 'sessions_pin': pin(sessions_path),
        'protocol_pin': key_plan['pins'][0], 'footer_receipt_pin': key_plan['pins'][1],
        'key_row_groups': key['selected_row_groups'], 'row_groups': financial_groups,
        'selected_rows': key['selected_rows'], 'financial_session_rows': financial_rows,
        'provenance_columns': PROVENANCE, 'financial_columns': FINANCIAL,
        'projected_compressed_bytes': cost, 'prior_case_payload_bytes': prior,
        'reserved_cumulative_case_payload_bytes': prior+cost,
        'max_case_payload_bytes': 134217728, 'max_output_bytes': 67108864,
        'max_rss_bytes': 2147483648, 'max_wall_seconds': 600, 'threads': 1,
        'excluded_available_fields': ['vwap', 'trade_count'],
        'unavailable_action_quality_fields': ['split_ratio', 'dividend', 'provider_adjustment_factor', 'provider_price_quality'],
        'transient_mixed_block_decode_disclosure': 'Only pinned matching row groups/columns fetched. Arrow transiently decodes other source/date cells within these compressed column blocks, possibly 2024+, before immutable key-position take. Excluded cells never convert to Python, are never returned or analyzed. No per-cell physical isolation is claimed.',
        'retained_raw_source_chunks': False, 'full_object_download_allowed': False,
        'mapping_promotion_allowed': False, 'rebuild_fit_backtest_allowed': False}
    write(output, plan)
    print(json.dumps({'plan_pin': pin(output), 'selected_row_groups': plan['row_groups'],
        'selected_rows': plan['selected_rows'], 'cumulative_payload_bytes': prior+cost}))
    return plan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--key-receipt', required=True); p.add_argument('--qa', required=True)
    p.add_argument('--sessions', required=True); p.add_argument('--output', required=True)
    a = p.parse_args(); prepare(a.key_receipt, a.qa, a.sessions, a.output)

if __name__ == '__main__': main()
