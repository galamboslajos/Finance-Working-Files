#!/usr/bin/env python3
"""Seal six metadata-only source tables after manifest/footer/byte checks."""
import hashlib
import json
from pathlib import Path
import struct
import plan_source_inventory as p


def footer_pin(path):
    with path.open('rb') as handle:
        handle.seek(-8, 2)
        tail = handle.read(8)
        if tail[4:] != b'PAR1':
            raise ValueError('Not a Parquet footer')
        length = struct.unpack('<I', tail[:4])[0]
        handle.seek(-(length + 8), 2)
        raw = handle.read(length + 8)
    return {'footer_sha256': hashlib.sha256(raw).hexdigest(),
        'footer_size_bytes': len(raw), 'footer_offset': path.stat().st_size - len(raw)}


def main():
    catalog = p.SourceCatalog(p.ROOT)
    tables = []
    for (source, table), columns in p.PROJECTIONS.items():
        if table == 'raw_page_inventory':
            continue
        catalog.table_schema(source, table)
        objects = []
        for obj in catalog.table_objects(source, table):
            item = p.footer_plan(obj, columns)
            digest = p.raw_pin(obj.path)['sha256']
            if obj.sha256 != 'sha256:' + digest:
                raise ValueError('Metadata object bytes differ from published manifest')
            objects.append({**item, 'sha256': digest, **footer_pin(obj.path)})
        tables.append({'source': source, 'table': table, 'columns': columns,
            'physical_rows': sum(o['rows'] for o in objects), 'objects': objects,
            'projected_compressed_bytes': sum(o['projected_compressed_bytes'] for o in objects)})
    protocol = p.OUT.parent / 'protocol.json'
    plan = {'schema': 'dmv2-xom-bk-source-stage1-plan/v1', 'root': str(p.ROOT),
        'protocol_pin': p.raw_pin(protocol), 'planner_pin': p.raw_pin(Path(__file__)),
        'footer_helper_pin': p.raw_pin(p.OUT / 'plan_source_inventory.py'),
        'source_config_pin': p.raw_pin(p.ROOT / 'configs/data_sources.local.json'),
        'manifest_pins': [{**p.raw_pin(s.base / 'MANIFEST.json'),
            'source_key': s.key, 'canonical_sha256': s.manifest_sha256} for s in catalog],
        'cases': json.loads(protocol.read_text())['cases'], 'tables': tables,
        'projected_compressed_bytes': sum(t['projected_compressed_bytes'] for t in tables),
        'physical_rows': sum(t['physical_rows'] for t in tables),
        'data_files': sum(len(t['objects']) for t in tables),
        'footer_files': sum(len(t['objects']) for t in tables),
        'integrity_metadata_object_bytes_separate': sum(o['size_bytes'] for t in tables for o in t['objects']),
        'returned_metadata_rows_upper_bound': sum(t['physical_rows'] for t in tables),
        'table_values_decoded_during_planning': False, 'financial_columns': [],
        'price_key_inventory_executed': False, 'cloud_reads': False,
        'canonical_identity_promoted': False}
    path = p.OUT / 'plan_stage1.json'
    if path.exists():
        raise FileExistsError('Preserve earlier sealed plan')
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + '\n')
    print(json.dumps({k: plan[k] for k in ('projected_compressed_bytes', 'physical_rows', 'data_files',
        'footer_files', 'integrity_metadata_object_bytes_separate')}))


if __name__ == '__main__':
    main()
