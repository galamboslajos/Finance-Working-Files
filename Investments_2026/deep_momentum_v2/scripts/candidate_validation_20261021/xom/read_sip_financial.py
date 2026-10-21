#!/usr/bin/env python3
"""Sealed bounded source OHLCV validation; no learning or execution inputs changed."""
import argparse
import importlib.util
import json
from pathlib import Path
import time

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from range_source import AuthenticatedRanges, load_group
from read_sip_keys import checked, pin, sha, write, need, resources
from prepare_financial_plan import FINANCIAL, PROVENANCE

KEYS = ['security_id', 'trading_date', 'query_symbol', 'adjustment']


def selected_values(table, selection):
    positions = selection['positions']
    need(positions == sorted(set(positions)) and all(isinstance(i, int) and not isinstance(i, bool)
        and 0 <= i < table.num_rows for i in positions), 'Invalid immutable row positions')
    need(len(positions) == len(selection['keys']), 'Position/key length mismatch')
    # This is the only transition toward financial Python values; take FIRST.
    return table.take(pa.array(positions, type=pa.int64()))


def load_qa(item):
    path = checked(item)
    spec = importlib.util.spec_from_file_location('frozen_xom_paired_qa', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def run(plan_path, launch_path, output):
    began = time.monotonic()
    initial_plan_pin, initial_launch_pin, initial_worker_pin = pin(plan_path), pin(launch_path), pin(__file__)
    plan = json.loads(Path(plan_path).read_text()); launch = json.loads(Path(launch_path).read_text())
    need(plan['schema'] == 'dmv2-xom-sip-financial-plan/v1', 'Wrong financial plan')
    need(launch['schema'] == 'dmv2-xom-sip-financial-launch/v1' and launch['status'] == 'SEALED_BEFORE_FINANCIAL_READ', 'Financial read not sealed')
    need(launch['plan_sha256'] == sha(plan_path) and launch['worker_sha256'] == sha(__file__), 'Frozen financial worker/plan changed')
    for name in ('key_receipt_pin','positions_pin','selected_key_pin','footer_pin','layout_pin','helper_pin',
                 'key_worker_pin','qa_pin','sessions_pin','protocol_pin','footer_receipt_pin'):
        checked(plan[name])
    need(launch['planner_sha256'] == sha(Path(__file__).with_name('prepare_financial_plan.py')), 'Frozen planner/constants changed')
    need(plan['financial_columns'] == FINANCIAL and plan['provenance_columns'] == PROVENANCE, 'Financial/provenance projection differs')
    need(plan['max_case_payload_bytes'] == 134217728 and plan['max_output_bytes'] == 67108864
         and plan['max_rss_bytes'] == 2147483648 and plan['max_wall_seconds'] == 600 and plan['threads'] == 1,
         'Frozen resource settings changed')
    key = json.loads(checked(plan['key_receipt_pin']).read_text())
    need(key['status'] == 'PASS_METADATA_KEYS_ONLY' and key['positions_pin'] == plan['positions_pin']
         and key['selected_key_pin'] == plan['selected_key_pin'] and key['selected_rows'] == plan['selected_rows'] <= 10000,
         'Financial selection differs from authenticated key pass')
    positions = json.loads(checked(plan['positions_pin']).read_text())
    saved_keys = json.loads(checked(plan['selected_key_pin']).read_text())
    layout = json.loads(checked(plan['layout_pin']).read_text()); footer = checked(plan['footer_pin']).read_bytes()
    need(key['object'] == layout['object'] == plan['object'], 'Source object identity changed')
    need(plan['key_row_groups'] == key['selected_row_groups'] == [s['index'] for s in positions], 'Selected key row groups changed')
    groups = [g for g in layout['row_groups'] if g['index'] in plan['row_groups']]
    cost = sum(g['column_chunks'][c]['compressed_bytes'] for g in groups for c in PROVENANCE+FINANCIAL)
    need(cost == plan['projected_compressed_bytes'] and cost+plan['prior_case_payload_bytes'] ==
         plan['reserved_cumulative_case_payload_bytes'] <= plan['max_case_payload_bytes'], 'Cumulative source range reservation differs')
    footer_receipt = json.loads(checked(plan['footer_receipt_pin']).read_text())
    need(plan['prior_case_payload_bytes'] == footer_receipt['actual_cli_stdout_bytes']+
        key['actual_http_range_payload_bytes'], 'Prior source payload not fully reserved')
    calendar = json.loads(checked(plan['sessions_pin']).read_text()); all_sessions = calendar['sessions']
    need(all_sessions and all_sessions == sorted(set(all_sessions)), 'Invalid independent calendar')
    sessions = [d for d in all_sessions if plan['start'] <= d <= plan['end']]
    need(sessions, 'Empty scoped independent calendar')
    key_tuples = lambda rows: sorted((r['security_id'],r['trading_date'],r['query_symbol'],r['adjustment']) for r in rows)
    all_keys = [k for s in positions for k in s['keys']]
    need(len(all_keys) == len(saved_keys) == plan['selected_rows'] and key_tuples(all_keys) == key_tuples(saved_keys), 'Key-position manifest disagrees')
    need(len(set(key_tuples(all_keys))) == len(all_keys) and all(r['security_id'] == plan['security_id']
        and r['query_symbol'] == plan['query_symbol'] and r['adjustment'] in ('raw','all')
        and plan['start'] <= r['trading_date'] <= plan['end'] < '2024-01-01' for r in all_keys), 'Unauthorized financial row before source decode')
    known_sessions = set(sessions)
    session_selections, extra_keys = [], []
    for selection in positions:
        admitted = [(i,k) for i,k in zip(selection['positions'],selection['keys']) if k['trading_date'] in known_sessions]
        extra_keys.extend(k for k in selection['keys'] if k['trading_date'] not in known_sessions)
        if admitted:
            session_selections.append({'index':selection['index'], 'positions':[i for i,k in admitted],
                                       'keys':[k for i,k in admitted]})
    need(plan['row_groups'] == [s['index'] for s in session_selections] and
        sum(len(s['keys']) for s in session_selections) == plan['financial_session_rows'],
        'Independent session row mask differs before financial read')
    qa = load_qa(plan['qa_pin'])
    out = Path(output).resolve(); need(not out.exists(), 'Preserve earlier financial attempt'); out.mkdir(parents=True)
    write(out/'reservation.json', {'plan_pin': pin(plan_path), 'launch_pin': pin(launch_path),
        'projected_compressed_bytes': cost, 'reserved_cumulative_case_payload_bytes': plan['reserved_cumulative_case_payload_bytes'],
        'financial_columns': FINANCIAL, 'provenance_columns': PROVENANCE,
        'selected_rows': len(all_keys), 'transient_mixed_block_decode_disclosure': plan['transient_mixed_block_decode_disclosure']})
    pa.set_cpu_count(1); pa.set_io_thread_count(1)
    transport = AuthenticatedRanges(plan['object'], max_bytes=cost)
    filtered = []; selected_lookup = {s['index']:s for s in session_selections}
    for group in groups:
        resources(began)
        selection = selected_lookup[group['index']]
        # Provenance is checked on the admitted positions before price conversion.
        provenance_all = load_group(footer, plan['object'], group, PROVENANCE, transport)
        provenance = selected_values(provenance_all, selection); del provenance_all
        for col, expected in (('feed','sip'),('timeframe','1Day')):
            need(provenance[col].null_count == 0 and pc.all(pc.equal(provenance[col], expected)).as_py() is True,
                 'Conflicting selected provider feed/timeframe')
        financial_all = load_group(footer, plan['object'], group, FINANCIAL, transport)
        financial = selected_values(financial_all, selection); del financial_all
        keys = pa.Table.from_pylist(selection['keys']).select(KEYS)
        combined = keys
        for column in PROVENANCE: combined = combined.append_column(column, provenance[column])
        for column in FINANCIAL: combined = combined.append_column(column, financial[column])
        need(combined.num_rows == len(selection['keys']), 'Filtered financial/key count changed')
        filtered.append(combined)
        del financial, provenance, combined
    need(transport.total == cost, 'Actual financial range payload differs from reservation')
    table = pa.concat_tables(filtered); del filtered
    need(table.num_rows == plan['financial_session_rows'] <= 10000, 'Filtered output row count differs')
    # Numeric Python conversion happens only on the exact original source/date mask.
    rows = table.to_pylist()
    summary, quality = qa.audit_rows(rows, sessions)
    summary['extra_non_XNYS_keys_metadata_only'] = extra_keys
    summary['original_scoped_source_keys_including_extras'] = plan['selected_rows']
    summary['excluded_available_fields'] = plan['excluded_available_fields']
    summary['unavailable_action_quality_fields'] = plan['unavailable_action_quality_fields']
    summary['transient_mixed_block_decode_disclosure'] = plan['transient_mixed_block_decode_disclosure']
    pq.write_table(table, out/'quotes.parquet', compression='zstd')
    write(out/'field_summary.json', summary); write(out/'row_quality.json', quality)
    write(out/'ranges.json', transport.ranges)
    output_size = sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
    need(output_size + 32768 <= plan['max_output_bytes'], 'Retained output cap exceeded')
    for item in (initial_plan_pin, initial_launch_pin, initial_worker_pin,
                 *[plan[name] for name in ('key_receipt_pin','positions_pin','selected_key_pin','footer_pin',
                    'layout_pin','helper_pin','key_worker_pin','qa_pin','sessions_pin','protocol_pin','footer_receipt_pin')]):
        checked(item)
    need(launch['planner_sha256'] == sha(Path(__file__).with_name('prepare_financial_plan.py')), 'Frozen planner changed during read')
    receipt = {'schema':'dmv2-xom-sip-financial-receipt/v1','status':'PASS_FIELD_AUDIT_COMPLETED',
        'plan_pin':pin(plan_path),'launch_pin':pin(launch_path),'worker_pin':pin(__file__),
        'helper_pin':plan['helper_pin'],'qa_pin':plan['qa_pin'],'sessions_pin':plan['sessions_pin'],
        'object':plan['object'],'quotes_pin':pin(out/'quotes.parquet'),'summary_pin':pin(out/'field_summary.json'),
        'row_quality_pin':pin(out/'row_quality.json'),'range_pin':pin(out/'ranges.json'),
        'financial_python_rows':len(rows),'session_rows_audited':len(rows),'extra_non_session_keys_metadata_only':len(extra_keys),
        'actual_financial_range_payload_bytes':transport.total,
        'cumulative_case_payload_bytes':transport.total+plan['prior_case_payload_bytes'],
        'retained_output_bytes_before_receipt':output_size,'peak_rss_bytes':resources(began),
        'elapsed_seconds':time.monotonic()-began,'canonical_mapping_promoted':False,
        'features_labels_models_backtests_built':False,'full_object_downloaded':False,
        'original_source_chunks_retained':False,'financial_2024_plus_python_rows':0,
        'physical_per_cell_isolation_claimed':False,
        'source_full_object_sha256_verification':'Manifest SHA recorded; exact generation/ranges and each payload hash checked; full object not downloaded or rehashed.',
        'transient_mixed_block_decode_disclosure':plan['transient_mixed_block_decode_disclosure']}
    write(out/'receipt.json',receipt)
    print(json.dumps({k:receipt[k] for k in ('status','financial_python_rows','actual_financial_range_payload_bytes','cumulative_case_payload_bytes')}))
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',required=True);p.add_argument('--launch',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.plan,a.launch,a.output)

if __name__=='__main__':main()
