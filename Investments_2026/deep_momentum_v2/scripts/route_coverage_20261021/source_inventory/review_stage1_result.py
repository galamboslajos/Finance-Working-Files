#!/usr/bin/env python3
"""Review saved stage1 metadata only; never reopen source Parquet tables."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def pin(path):
    return {'path': str(path.resolve()), 'size_bytes': path.stat().st_size,
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def need(test, message):
    if not test:
        raise ValueError(message)


def main():
    directory = HERE / 'result-v1'
    receipt, cases, plan = (read(p) for p in (directory / 'receipt.json', directory / 'cases.json', HERE / 'plan_stage1.json'))
    cfg_path = Path(receipt['configuration_pin']['path'])
    config = read(cfg_path)
    need(receipt['status'] == 'PASS_METADATA_CANDIDATE_INVENTORY', 'Stage1 incomplete')
    need(receipt['cases_pin'] == pin(directory / 'cases.json'), 'Saved case bytes changed')
    need(receipt['plan_pin'] == pin(HERE / 'plan_stage1.json'), 'Saved plan changed')
    need(receipt['worker_pin'] == pin(HERE / 'read_stage1.py'), 'Worker changed after result')
    need(config['worker_sha256'] == receipt['worker_pin']['sha256'], 'Wrong frozen worker')
    need(config['plan_sha256'] == receipt['plan_pin']['sha256'], 'Wrong frozen plan')
    need(receipt['metadata_rows_converted'] == plan['physical_rows'] == 85045, 'Metadata row ledger differs')
    need(receipt['projected_compressed_bytes'] == plan['projected_compressed_bytes'] == 6495846, 'Read reservation differs')
    need(receipt['source_data_files'] == 6 and receipt['financial_columns_decoded'] == [], 'Forbidden source read')
    need(receipt['cloud_reads'] is False and receipt['canonical_mappings_promoted'] is False, 'Unexpected mutation/acquisition')
    need({c['ticker_label'] for c in cases} == {'XOM', 'BK'}, 'Wrong case scope')
    by_case = {c['ticker_label']: c for c in cases}
    xom, bk = by_case['XOM'], by_case['BK']
    x_source = xom['source_master_rows'][0]
    x_bridge = xom['exact_sp1500_bridge_rows'][0]
    x_link = next(r for r in xom['related_published_identity_links'] if r['unified_security_id'] == x_bridge['unified_security_id'])
    x_provider = next(r for r in xom['provider_master_candidate_rows'] if r['intrinio_security_id'] == xom['native_security_id'])
    x_acq = next(r for r in xom['acquisition_candidate_rows'] if r['intrinio_security_id'] == xom['native_security_id'])
    b_bridge = bk['exact_sp1500_bridge_rows'][0]
    b_link = next(r for r in bk['related_published_identity_links'] if r['unified_security_id'] == b_bridge['unified_security_id'])
    b_provider = next(r for r in bk['provider_master_candidate_rows'] if r['intrinio_security_id'] == bk['native_security_id'])
    b_acq = next(r for r in bk['acquisition_candidate_rows'] if r['intrinio_security_id'] == bk['native_security_id'])
    alternate = next(r for r in bk['provider_master_candidate_rows'] if r['intrinio_security_id'] == 'sec_zq1J05')
    alt_link = next(r for r in bk['related_published_identity_links'] if r['intrinio_security_id'] == 'sec_zq1J05')
    alt_acq = next(r for r in bk['acquisition_candidate_rows'] if r['intrinio_security_id'] == 'sec_zq1J05')
    need(x_source['isin'] == x_bridge['isin'] == 'US30231G1022', 'Unexpected source XOM literal')
    need(x_link['query_identifier_type'] == 'isin' and x_link['query_identifier'] == 'XOM', 'XOM query/type inconsistency changed')
    need(x_link['source_response_hash'] == x_provider['source_response_hash'], 'XOM capture metadata hashes differ')
    need(x_provider['first_stock_price'] == '2026-07-06' and x_acq['price_row_count'] == 15, 'XOM metadata coverage differs')
    need(b_provider['code'] == 'PRF' and b_acq['price_row_count'] == 0, 'BK selected security-class evidence differs')
    need(b_link['source_response_hash'] == b_provider['source_response_hash'], 'BK capture metadata hashes differ')
    need(alt_link['query_identifier'] == 'US0640581007' and alt_link['query_identifier_type'] == 'isin', 'BK candidate literal-query evidence differs')
    need(alternate['code'] == 'EQS' and alternate['type'] == 'Ordinary Shares'
        and alt_acq['price_row_count'] == 13430, 'BK candidate ordinary-share capture differs')
    findings = {'schema': 'dmv2-xom-bk-stage1-independent-review/v1',
        'status': 'PASS_SAVED_METADATA_REVIEW',
        'reviewer_code_pin': pin(Path(__file__)),
        'input_pins': [pin(p) for p in (directory / 'receipt.json', directory / 'cases.json',
            HERE / 'plan_stage1.json', HERE / 'read_stage1.py', cfg_path)],
        'metadata_only': True, 'new_source_parquet_reads': 0,
        'financial_values_decoded': False, 'canonical_mapping_promoted': False,
        'xom': {'source_security_id': xom['source_security_id'], 'current_native': xom['native_security_id'],
            'source_isin': x_source['isin'], 'unified_id': x_bridge['unified_security_id'],
            'declared_query_type': x_link['query_identifier_type'], 'literal_query': x_link['query_identifier'],
            'literal_query_equals_source_isin': False,
            'provider_code': x_provider['code'], 'provider_type': x_provider['type'],
            'provider_first_stock_price': x_provider['first_stock_price'],
            'provider_last_stock_price': x_provider['last_stock_price'],
            'reported_whole_capture_rows': x_acq['price_row_count'],
            'captured_native_history_outside_research_period': True,
            'alternative_native_observed_in_stage1': False,
            'interpretation': 'Published route leads to a captured native with only 2026 reported history; the source ISIN alias does not establish a literal ISIN lookup. This does not prove that Exxon historical prices are absent.'},
        'bk': {'source_security_id': bk['source_security_id'], 'current_native': bk['native_security_id'],
            'unified_id': b_bridge['unified_security_id'], 'literal_query': b_link['query_identifier'],
            'selected_provider_code': b_provider['code'], 'selected_provider_type': b_provider['type'],
            'selected_primary_security': b_provider['primary_security'],
            'selected_primary_listing': b_provider['primary_listing'],
            'selected_reported_capture_rows': b_acq['price_row_count'],
            'ordinary_share_candidate': {'intrinio_security_id': alternate['intrinio_security_id'],
                'query_identifier': alt_link['query_identifier'], 'query_identifier_type': alt_link['query_identifier_type'],
                'provider_ticker': alternate['ticker'], 'provider_code': alternate['code'], 'provider_type': alternate['type'],
                'provider_figi': alternate['figi'], 'provider_share_class_figi': alternate['share_class_figi'],
                'provider_first_stock_price': alternate['first_stock_price'], 'provider_last_stock_price': alternate['last_stock_price'],
                'reported_whole_capture_rows': alt_acq['price_row_count'],
                'status': 'CANDIDATE_ONLY_NOT_PROMOTED', 'exact_2010_2023_price_keys_checked': False},
            'interpretation': 'Selected published BK lookup is a Series I preferred security. A separate provider ordinary-share candidate exists. Source BK membership lacks an ISIN; dated common-share linkage and exact history still need verification.'},
        'claim_limits': ['Reported acquisition counts do not independently verify exact-date price coverage.',
            'Shared source-response hashes link published captured metadata; original identity-response payloads were not opened.',
            'Source aliases and declared query types are not independent historical identity proof.',
            'No mapping, eligibility, feature, label, RET payoff, model, source or backtest artifact changed.']}
    path = HERE / 'independent_result_review.json'
    if path.exists():
        raise FileExistsError('Preserve earlier independent review')
    path.write_text(json.dumps(findings, indent=2, sort_keys=True) + '\n')
    print(json.dumps(pin(path)))


if __name__ == '__main__':
    main()
