#!/usr/bin/env python3
"""Read only pinned local manifests and footers; never decode table values."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path('/Users/lajosgalambos/Desktop/Finance-Working-Files/Investments_2026/deep_momentum_v2')
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from dmv2.catalog import SourceCatalog
import pyarrow.parquet as pq


PROJECTIONS = {
 ('siblis', 'security_master_siblis'): ['security_id', 'primary_ticker', 'company_name', 'isin',
     'provider_security_key', 'identity_confidence', 'source_namespace'],
 ('siblis', 'security_identifier_history'): ['security_id', 'identifier_type', 'identifier_value',
     'valid_from', 'valid_to', 'source_namespace', 'identity_confidence'],
 ('identity', 'security_identity_bridge'): ['unified_security_id', 'identity_key', 'bridge_method',
     'bridge_confidence', 'bridge_version', 'source_product_id', 'source_security_id', 'primary_ticker',
     'normalized_ticker', 'company_name', 'normalized_company_name', 'isin', 'provider_security_key',
     'source_namespace', 'source_identity_confidence', 'first_membership_date', 'last_membership_date'],
 ('intrinio', 'intrinio_identity_links'): ['unified_security_id', 'intrinio_security_id',
     'query_identifier', 'query_identifier_type', 'observed_tickers_json', 'observed_isins_json',
     'source_product_ids_json', 'identity_review_required', 'resolution_status', 'http_status',
     'source_response_hash', 'source_relative_path', 'source_identity_run_id', 'source_identity_manifest_hash'],
 ('intrinio', 'intrinio_security_master'): ['intrinio_security_id', 'company_id', 'requested_identifier',
     'ticker', 'composite_ticker', 'figi', 'composite_figi', 'share_class_figi', 'primary_security',
     'primary_listing', 'etf', 'type', 'code', 'exchange_mic', 'first_stock_price', 'last_stock_price',
     'delisted', 'retrieved_at_utc', 'source_run_id', 'source_response_hash', 'source_relative_path'],
 ('intrinio', 'identifier_resolution_evidence'): ['requested_identifier', 'status', 'http_status',
     'intrinio_security_id', 'ticker', 'composite_ticker', 'first_stock_price', 'last_stock_price',
     'delisted', 'price_row_count', 'price_page_count', 'message', 'source_run_id'],
 ('intrinio', 'raw_page_inventory'): ['requested_identifier', 'intrinio_security_id', 'page_kind',
     'page_index', 'http_status', 'row_count', 'raw_json_sha256', 'raw_relative_path', 'hash_verified'],
}


def raw_pin(path):
    return {'path': str(path), 'size_bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def footer_plan(obj, columns):
    obj.check()
    pf = pq.ParquetFile(obj.path)
    missing = set(columns) - set(pf.schema_arrow.names)
    if missing:
        raise ValueError(f'Missing metadata fields: {missing}')
    groups = []
    for i in range(pf.metadata.num_row_groups):
        group = pf.metadata.row_group(i)
        projected = sum(group.column(j).total_compressed_size for j in range(group.num_columns)
            if group.column(j).path_in_schema.split('.')[0] in columns)
        groups.append({'index': i, 'rows': group.num_rows, 'projected_compressed_bytes': projected})
    return {'path': str(obj.path), 'relative_path': obj.relative_path,
        'size_bytes': obj.size_bytes, 'declared_object_sha256': obj.sha256,
        'rows': pf.metadata.num_rows, 'columns': columns,
        'projected_compressed_bytes': sum(g['projected_compressed_bytes'] for g in groups),
        'row_groups': groups, 'schema': [[name, str(pf.schema_arrow.field(name).type)] for name in columns]}


def main():
    catalog = SourceCatalog(ROOT)
    tables = []
    for (source, table), columns in PROJECTIONS.items():
        schema = catalog.table_schema(source, table)
        objects = [footer_plan(o, columns) for o in catalog.table_objects(source, table)]
        tables.append({'source': source, 'table': table, 'columns': columns,
            'physical_rows': sum(o['rows'] for o in objects), 'objects': objects,
            'projected_compressed_bytes': sum(o['projected_compressed_bytes'] for o in objects)})
    price_columns = ['intrinio_security_id', 'trading_date']
    price_objects = [footer_plan(o, price_columns) for o in catalog.table_objects('intrinio', 'intrinio_daily_prices')]
    plan = {'schema': 'dmv2-xom-bk-source-metadata-plan/v1',
        'root': str(ROOT), 'source_config_pin': raw_pin(ROOT / 'configs/data_sources.local.json'),
        'manifest_pins': [{**raw_pin(s.base / 'MANIFEST.json'),
            'source_key': s.key, 'canonical_sha256': s.manifest_sha256} for s in catalog],
        'targets': [{'ticker_label': 'XOM', 'source_security_id': 'siblis_isin_b284bb7497616f5afba0', 'current_native': 'sec_z5YpGE'},
            {'ticker_label': 'BK', 'source_security_id': 'siblis_ticker_company_1c21a7b293fd137c59c9', 'current_native': 'sec_gA3NYQ'}],
        'stage1_tables': tables,
        'stage1_projected_compressed_bytes': sum(t['projected_compressed_bytes'] for t in tables),
        'stage1_physical_rows': sum(t['physical_rows'] for t in tables),
        'stage1_object_count': sum(len(t['objects']) for t in tables),
        'stage1_candidate_logic': ['Keep exact two Siblis source rows and identifier histories.',
            'Keep matching source-ID/product bridge routes, same-literal ISIN or ticker aliases as candidates only.',
            'Keep exact native IDs, related unified links and literal-query/ticker candidates.',
            'Provider-master same ticker/FIGI/company IDs identify candidate alternatives, never accepted mappings.',
            'Whole-history acquisition/page row counts are retrieval inventory, not exact-date price coverage.'],
        'stage2_planning_only': {'table': 'intrinio_daily_prices', 'columns': price_columns,
            'date_start': '2007-01-01', 'date_end': '2023-12-31',
            'native_ids': ['sec_z5YpGE', 'sec_gA3NYQ'],
            'objects': price_objects, 'metadata_footer_count': len(price_objects),
            'unpruned_all_object_projected_upper_bound_bytes': sum(o['projected_compressed_bytes'] for o in price_objects),
            'candidate_rows_upper_bound': sum(o['rows'] for o in price_objects),
            'requires_frozen_specific_candidate_plan_before_read': True},
        'financial_columns': [], 'table_data_decoded': False, 'cloud_reads': False,
        'canonical_mappings_promoted': False,
        'limitations': ['Manifest/object hashes are declared pins; byte hashes of source Parquet payloads not repeated.',
            'This plan inspects physical schemas and footers only; it establishes no observed provider/security row values.',
            'Source aliases are not independent provider-returned identity evidence.',
            'First/last-price dates and reported acquisition row counts are metadata, not dated identity validity.']}
    path = OUT / 'source_inventory_plan.json'
    if path.exists():
        raise FileExistsError('Preserve earlier plan')
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'path': str(path), 'stage1_projected_compressed_bytes': plan['stage1_projected_compressed_bytes'],
        'stage1_physical_rows': plan['stage1_physical_rows'], 'stage1_object_count': plan['stage1_object_count'],
        'stage2_unpruned_projected_bytes': plan['stage2_planning_only']['unpruned_all_object_projected_upper_bound_bytes']}))


if __name__ == '__main__':
    main()
