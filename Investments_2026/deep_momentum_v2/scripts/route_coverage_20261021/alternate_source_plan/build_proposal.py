#!/usr/bin/env python3
"""Build a locator proposal from existing local JSON; no new table/cloud reads."""
import hashlib
import json
from pathlib import Path

R = Path('/Users/lajosgalambos/Desktop/Finance-Working-Files/Investments_2026/deep_momentum_v2')
S = Path(__file__).resolve().parent


def load(path):
    return json.loads(path.read_text())


def pin(path):
    return {'path': str(path), 'size_bytes': path.stat().st_size,
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    prior_path = R / 'outputs/identity_audits/supporting/remote_manifest_metadata/dmv2-prior-intrinio-manifest.json'
    retained_path = R / 'outputs/identity_audits/supporting/remote_manifest_metadata/dmv2-retained-intrinio-manifest.json'
    current_path = R / 'data/raw/equity_us_intrinio_market_daily_v1/professional-reference-20260807-r1-pdrs-v2-r3/MANIFEST.json'
    unified_path = R / 'data/derived/daily_action_inputs/equity_us_market_daily_unified_MANIFEST.json'
    sp_path = R / 'data/derived/daily_action_inputs/sp_composite1500_market_daily_MANIFEST.json'
    cfg_path = R / 'configs/data_sources.local.json'
    cfg = load(cfg_path)
    identity_path = R / next(s['manifest'] for s in cfg['sources'] if s['key'] == 'identity')
    old, cur, unified, sp, identity = (load(p) for p in (prior_path, current_path, unified_path, sp_path, identity_path))
    old_obj = {o['relative_path']: o for o in old['objects']}
    cur_obj = {o['relative_path']: o for o in cur['objects']}
    unified_obj = {o['relative_path']: o for o in unified['objects']}
    identity_obj = {o['relative_path']: o for o in identity['objects']}
    duplicate = []
    for table in ('intrinio_security_master', 'intrinio_identity_links', 'identifier_resolution_evidence'):
        rel = f'tables/{table}/part-00000.parquet'
        a, b = old_obj[rel], cur_obj[rel]
        identical = all(a[k] == b[k] for k in ('uri', 'generation', 'sha256', 'size_bytes'))
        if not identical:
            raise ValueError('Older metadata objects differ; revise proposal')
        duplicate.append({'table': table, 'object': a, 'identical_to_configured_local_object': True,
            'recommended_new_remote_reads': 0,
            'reason': 'Exact same immutable URI, generation, SHA and size already inspected in stage1.'})
    bridge_rel = 'tables/security_identity_bridge/part-00000.parquet'
    a, b = unified_obj[bridge_rel], identity_obj[bridge_rel]
    same_bridge = a['sha256'] == b['sha256'] and a['size_bytes'] == b['size_bytes']
    if not same_bridge:
        raise ValueError('Unified market bridge differs; inspect separately before claiming reuse')
    remote = []
    for table in ('symbol_normalization_evidence', 'missing_symbol_evidence'):
        obj = next(o for o in sp['objects'] if o['relative_path'] == f'tables/{table}/part-00000.parquet')
        schema = next(t for t in sp['schema']['tables'] if t['name'] == table)
        remote.append({'table': table, 'object': obj,
            'declared_rows': schema['row_count'], 'allowed_columns': [c['name'] for c in schema['columns']],
            'metadata_only_schema': True,
            'preconversion_filter': "query_symbol in {'XOM','BK'} or raw_membership_ticker in {'XOM','BK'} where that column exists",
            'output_row_limit': 1000, 'financial_columns': [],
            'purpose': 'Inspect published literal symbol normalization and request/response row-count inventory; not provider-native identity promotion.'})
    plan = {'schema': 'dmv2-xom-bk-alternate-source-locator-proposal/v1',
        'status': 'PROPOSAL_ONLY_NOT_EXECUTED', 'root': str(R),
        'cases': [{'ticker_label': 'XOM', 'source_security_id': 'siblis_isin_b284bb7497616f5afba0',
            'source_isin': 'US30231G1022', 'unified_security_id': 'equity_us_bf463bc8d3ea387578b7bafc'},
            {'ticker_label': 'BK', 'source_security_id': 'siblis_ticker_company_1c21a7b293fd137c59c9'}],
        'local_manifest_pins': [pin(p) for p in (prior_path, retained_path, current_path, identity_path, unified_path, sp_path)],
        'retained_intrinio_manifest_locators': [
            {'uri': old['release_root'] + '/MANIFEST.json', 'generation': 1786102843388189,
                'size_bytes': 74655, 'sha256': pin(prior_path)['sha256'], 'already_cached': True},
            {'uri': 'gs://quant-research-trusty-agility-492109-v0/dataset-platform/curated/equity_us/intrinio_market_daily/v1/stable/MANIFEST.json',
                'generation': 1785655514257064, 'size_bytes': 46576,
                'sha256': pin(retained_path)['sha256'], 'already_cached': True}],
        'duplicate_intrinio_metadata_no_new_read_needed': duplicate,
        'rcm_legacy_origin': {'source': 'same configured Intrinio master already inspected',
            'related_metadata_receipt_pin': pin(R / 'data/derived/ret_repairs/20261020-v1/related_provider_identity_metadata.json'),
            'legacy_native': 'sec_yKbEk4',
            'not_an_older_or_separately_acquired_master': True},
        'alternative_unified_market_product': {
            'release_root': unified['release_root'], 'declared_coverage': unified['coverage'],
            'source_ref_manifest_locator': unified['source_refs'][0],
            'security_identity_bridge_object': a,
            'bridge_byte_identical_to_configured_local_identity_bridge': True,
            'local_full_bridge_path': str(identity_path.parent / bridge_rel),
            'already_in_stage1_plan': True, 'additional_bridge_read_needed': False,
            'new_intrinio_native_mapping_supplied_by_bridge': False,
            'daily_object_not_to_download': unified_obj['tables/daily_bars/part-00000.parquet'],
            'financial_daily_object_reads_in_this_proposal': 0},
        'small_alternative_source_metadata_proposal': {
            'release_root': sp['release_root'], 'declared_coverage': sp['coverage'],
            'source_ref_manifest_locator': sp['source_refs'][0], 'objects': remote,
            'maximum_download_bytes': sum(x['object']['size_bytes'] for x in remote),
            'physical_metadata_row_upper_bound': sum(x['declared_rows'] for x in remote),
            'maximum_converted_output_rows_per_object': 1000,
            'maximum_data_objects': 2, 'maximum_footer_files': 2,
            'requires_explicit_additive_metadata_protocol_before_any_new_read': True,
            'generation_pinned_reads_only': True,
            'stop_if_size_generation_schema_or_sha_differs': True,
            'raw_price_JSON_or_daily_bar_downloads_allowed': False,
            'not_yet_cached_in_searched_prior_evidence': True},
        'cloud_calls_performed': 0, 'new_source_data_decodes': 0, 'mapping_promotions': 0,
        'claim_limits': ['Cached earlier Intrinio publications reference exactly the same metadata objects as the configured release; they cannot supply an older XOM capture.',
            'The alternative unified bridge is source-alias evidence already in stage1, not independent provider-returned identity confirmation.',
            'Alternative SIP market products declare coverage from 2016-01-04; they cannot establish 2010–2015 history.',
            'Symbol normalization and request row counts may locate an alternative response history, but do not verify share-class identity, OHLCV correctness or exact trading-date coverage.',
            'A provider-native replacement for XOM has not been discovered in the inspected local metadata.',
            'No cloud read, price download, new mapping, rebuild, model load or backtest was performed.']}
    out = S / 'alternate_source_metadata_proposal.json'
    if out.exists():
        raise FileExistsError('Preserve prior proposal')
    out.write_text(json.dumps(plan, indent=2, sort_keys=True) + '\n')
    print(json.dumps(pin(out)))


if __name__ == '__main__':
    main()
