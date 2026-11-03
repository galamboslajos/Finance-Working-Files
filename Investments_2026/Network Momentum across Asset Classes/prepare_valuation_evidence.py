#!/usr/bin/env python3
"""Materialize price evidence from pinned sources; never modify source prices.

Run audit_wfo_availability.py first. Download the separately pinned independent
SIP object into this output directory before running the valuation repair.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds

from build_momentum_features import write_json
from prepare_sp500_daily import DEFAULT_SOURCE, Release, file_hash, require

ROOT = Path(__file__).resolve().parent


def prepare(dataset, availability, output, triage_output, source_root):
    require(not output.exists() and not triage_output.exists(), 'Evidence output exists; choose fresh directories')
    dm = json.loads((dataset / 'manifest.json').read_text())
    missing = pd.read_parquet(availability / 'missing_next_session_labels.parquet')
    require(len(missing) == 90 and not missing.duplicated(['date', 'source_security_id']).any(), 'Availability scope changed')
    paths = []
    for item in dm['output_files']:
        if item['path'].startswith('price_history/') and 2015 <= int(item['path'].split('/')[1][5:]) <= 2026:
            path = dataset / item['path']
            require(path.stat().st_size == item['size_bytes'] and file_hash(path) == item['sha256'], 'Prepared price source changed')
            paths.append(path)
    context = ds.dataset(paths).to_table(filter=ds.field('intrinio_security_id').isin(set(missing.intrinio_security_id))).to_pandas()
    context = context.loc[context.date.between('2015-12-01', '2026-01-02')].sort_values(['intrinio_security_id', 'date']).reset_index(drop=True)
    require(not context.duplicated(['intrinio_security_id', 'date']).any(), 'Duplicate native price observation')
    next_rows = context.rename(columns={'date':'label_date'})
    triage = missing.merge(next_rows[['intrinio_security_id', 'label_date', 'quote_status', 'price_quality_status', 'close', 'adj_close']],
        on=['intrinio_security_id', 'label_date'], how='left', validate='many_to_one')
    triage = triage.rename(columns={'quote_status':'next_quote_status', 'price_quality_status':'next_price_quality_status'})
    triage['next_observation_state'] = triage.next_quote_status.fillna('absent_native_bar')
    release = Release(source_root, 'prices')
    master = release.table('intrinio_security_master').to_table().to_pandas()
    links = release.table('intrinio_identity_links').to_table().to_pandas()
    events = json.loads((ROOT / 'configs/valuation_events_v1.json').read_text())['events']
    native = set(missing.intrinio_security_id) | {e['native'] for e in events if 'native' in e}
    dates = sorted({str(d.date()) for d in missing.date} | {str(d.date()) for d in missing.label_date})
    prices = release.table('intrinio_daily_prices').to_table(filter=ds.field('intrinio_security_id').isin(native)
        & ds.field('trading_date').isin(dates)).to_pandas()
    output.mkdir(parents=True); triage_output.mkdir(parents=True)
    for name, table in [('original_native_context', context), ('event_date_source_prices', prices),
                        ('security_master', master), ('identity_links', links)]:
        table.to_parquet(output / (name + '.parquet'), index=False)
    triage.to_csv(triage_output / 'missing_label_source_triage.csv', index=False)
    write_json(output / 'source_provenance.json', release.provenance())
    write_json(output / 'evidence_manifest.json', {'status':'complete', 'script_sha256':file_hash(__file__),
        'prepared_dataset_manifest_sha256':file_hash(dataset / 'manifest.json'),
        'availability_keys_sha256':file_hash(availability / 'missing_next_session_labels.parquet'),
        'source_provenance_sha256':file_hash(output / 'source_provenance.json'),
        'outputs':[{'path':str(p), 'sha256':file_hash(p)} for p in [*output.glob('*.parquet'), triage_output / 'missing_label_source_triage.csv']]})
    print(json.dumps({'status':'complete', 'native_context_rows':len(context), 'event_date_price_rows':len(prices), 'missing_keys':len(missing)}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,default=ROOT / 'data/sp500_daily')
    p.add_argument('--availability',type=Path,default=ROOT / 'data/wfo_availability/v1')
    p.add_argument('--output',type=Path,default=ROOT / 'data/performance_gap_audit/v1')
    p.add_argument('--triage-output',type=Path,default=ROOT / 'data/wfo_availability/source_triage_v1')
    p.add_argument('--source-root',type=Path,default=DEFAULT_SOURCE)
    a=p.parse_args(); prepare(a.dataset.resolve(),a.availability.resolve(),a.output.resolve(),a.triage_output.resolve(),a.source_root.resolve())
