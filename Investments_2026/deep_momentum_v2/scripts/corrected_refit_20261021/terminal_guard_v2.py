"""Metadata-only29-event monthly-target guard; no returns/classes/price reads."""
import argparse
from collections import Counter
from datetime import date
import json
import os
from pathlib import Path
import sys
from control import ROOT, check, digest, need, pin, read, require_reservation, save_new

R = Path('/Users/lajosgalambos/Desktop/Finance-Working-Files/Investments_2026/deep_momentum_v2')
BASE = R / 'outputs/bk_xom_shadow/2010-2023-20261021-v1/installed_catalog.json'
OLD = R / 'outputs/learning_input_impact/20261021-v1/metadata/result-v2'
OLD_REGISTRY_SHA = 'c75122e8dea69acbfc9efc6fe337b6f155c69b12be4b25c7744bc83e78bd89c1'
OLD_AUDIT_SHA = '84445cd26ae9fae1538950d6d3278b46073ad1809abdbfea53760c10248716ed'
KEYS = ('source_security_id', 'signal_date')
ORIGINAL = (*KEYS, 'intrinio_security_id')
TARGET = (*KEYS, 'outcome_month', 'prediction_eligible', 'target_available',
          'source_return_status', 'return_observation_status', 'target_status')
FAMILIES = {'current_reviewed': 18, 'historical_retirement': 11}


def iso(value):
    need(isinstance(value, str) and date.fromisoformat(value).isoformat() == value, 'Canonical metadata date required')
    return value


def lookup(rows):
    result = {}
    for row in rows:
        source, signal = (row[k] for k in KEYS)
        need(isinstance(source, str) and source.strip() == source and bool(source), 'Missing source identity')
        iso(signal); key = (source, signal)
        need(key not in result, 'Duplicate original metadata key')
        result[key] = row
    return result


def registry_checks(events, audit):
    need(audit['status'] == 'PASS_METADATA_AUDIT' and audit['known_legal_event_count'] == 29
         and audit['all_original_keys_preserved'] is True and audit['new_fits'] == 0,
         'Authenticated original metadata audit required')
    need(len(events) == len({e['event_id'] for e in events}) == 29
         and dict(Counter(e['family'] for e in events)) == FAMILIES,
         'Exactly29 unique event families18current/11historical required')
    for e in events:
        iso(e['effective_date'])
        need(len(e['overlap_rows']) == e['event_outcome_overlap_rows'] == 1
             and e['missing_target_event_overlap_rows'] == 1
             and e['post_effective_original_cohort_rows'] == 0 and not e['post_effective_rows'],
             'Old event cohort/withheld baseline changed')
        r = e['overlap_rows'][0]
        need(type(r['prediction_eligible']) is bool and r['prediction_eligible'] is True
             and r['target_available'] is False and r['return_status'] == 'missing_end',
             'Original terminal overlap must be eligible/missing, never a synthetic label')
    return events


def validate_target_rows(rows, year):
    found = lookup(rows)
    need({r['outcome_month'] for r in rows} == {f'{year}-{m:02d}' for m in range(1, 13)}, 'Missing/unknown annual target month')
    for r in rows:
        d = date.fromisoformat(r['signal_date']); following = f'{d.year+(d.month==12):04d}-{d.month%12+1:02d}'
        need(r['outcome_month'] == following and f'{year}-01' <= following <= f'{year}-12'
             and following <= '2023-12', 'Target metadata exceeds approved next-month scope')
        need(type(r['prediction_eligible']) is bool and type(r['target_available']) is bool
             and (not r['target_available'] or r['prediction_eligible']), 'Unknown target/prediction metadata flags')
        need(isinstance(r['target_status'], str) and bool(r['target_status']), 'Explicit target status required')
        for field in ('source_return_status', 'return_observation_status'):
            need(r[field] is None or isinstance(r[field], str) and bool(r[field]), 'Invalid nullable source status')
            need(r[field] is not None or r['target_available'] is False,
                 'Available target cannot be certified by unknown source status')
    return found


def audit_events(events, base_original, candidate_original, base_targets, candidate_targets):
    old, new = lookup(base_original), lookup(candidate_original)
    need(set(old) == set(new), 'Original constituent-month key set changed')
    need(all(old[k]['intrinio_security_id'] == new[k]['intrinio_security_id'] for k in old),
         'Price-only input revision changed original membership routes')
    bt, nt = lookup(base_targets), lookup(candidate_targets)
    need(set(bt) == set(nt) == set(old), 'Target ledger changed original cohort denominator')
    reports = []
    for e in events:
        declared = e['overlap_rows'][0]; key = tuple(declared[k] for k in KEYS)
        need(key in nt, 'Known terminal event source/signal row disappeared')
        a, b = bt[key], nt[key]
        need(old[key]['intrinio_security_id'] == new[key]['intrinio_security_id'] == e['predecessor_id'],
             'Terminal overlap is not the declared predecessor identity')
        need(a['outcome_month'] == b['outcome_month'] == declared['outcome_month'], 'Terminal event month changed')
        need(a['prediction_eligible'] == b['prediction_eligible'] == declared['prediction_eligible'],
             'Terminal knowledge changed a signal-time prediction cohort')
        need(a['target_available'] is False and b['target_available'] is False,
             'Previously missing terminal target became available without an economic-return policy')
        need(a['source_return_status'] == b['source_return_status'] == declared['return_status'],
             'Terminal endpoint status changed')
        need(all(isinstance(r[k], str) and bool(r[k]) for r in (a, b)
             for k in ('source_return_status','return_observation_status','target_status')),
             'Known29 terminal rows require explicit missingness evidence, not unknown status')
        need(a['return_observation_status'] == b['return_observation_status']
             and a['target_status'] == b['target_status'], 'Terminal missingness status changed')
        predecessor = e['predecessor_id']
        old_rows = [r for r in base_original if r['intrinio_security_id'] == predecessor]
        new_rows = [r for r in candidate_original if r['intrinio_security_id'] == predecessor]
        need(len(old_rows) == len(new_rows) == e['original_predecessor_cohort_rows'], 'Predecessor membership family changed')
        after = [r for r in new_rows if r['signal_date'] >= e['effective_date']]
        need(not after and not [r for r in old_rows if r['signal_date'] >= e['effective_date']],
             'Post-effective original predecessor membership appeared')
        reports.append(dict(event_id=e['event_id'], family=e['family'], source_security_id=key[0],
            signal_date=key[1], outcome_month=b['outcome_month'], prediction_eligible=b['prediction_eligible'],
            target_available=False, source_return_status=b['source_return_status'],
            return_observation_status=b['return_observation_status'], target_status=b['target_status'],
            predecessor_original_rows=len(new_rows), post_effective_original_rows=0))
    return reports


def project(item, columns):
    import pyarrow.parquet as pq
    f = pq.ParquetFile(check(item)); found = set()
    need(set(columns) <= set(f.schema_arrow.names), 'Metadata projection/schema mismatch')
    size = 0
    for g in range(f.metadata.num_row_groups):
        for c in range(f.metadata.row_group(g).num_columns):
            leaf = f.metadata.row_group(g).column(c); name = leaf.path_in_schema.split('.')[0]
            if name in columns: size += leaf.total_compressed_size; found.add(name)
    need(set(columns) <= found, 'Missing metadata physical column chunks')
    return dict(pin=item, columns=list(columns), rows=f.metadata.num_rows, projected_compressed_bytes=size)


def make_plan(candidate_path, output, base_path=BASE, registry_path=OLD/'known_event_overlaps.json', audit_path=OLD/'audit.json'):
    need(digest(registry_path) == OLD_REGISTRY_SHA and digest(audit_path) == OLD_AUDIT_SHA,
         'Original29-event registry/audit bytes changed')
    events, old_audit = read(registry_path), read(audit_path); registry_checks(events, old_audit)
    base, candidate = read(base_path), read(candidate_path)
    for catalog in (base, candidate):
        need(len(catalog['annuals']) == 14 and all(type(r['year']) is int for r in catalog['annuals'])
             and len({r['year'] for r in catalog['annuals']}) == 14,
             'Exactly14 unique integer annual records required')
    old = {r['year']: r for r in base['annuals']}; new = {r['year']: r for r in candidate['annuals']}
    need(set(old) == set(new) == set(range(2010, 2024)), 'Fourteen complete annual catalogs required')
    projections = {}; annuals = {}
    for year in range(2010, 2024):
        annuals[str(year)] = {}
        for version, record in (('base', old[year]), ('candidate', new[year])):
            for name, columns in (('original_cohort', ORIGINAL), ('targets', TARGET)):
                item = record['artifacts'][name]
                key = item['path'] + '|' + ','.join(columns)
                if key not in projections: projections[key] = project(item, columns)
                else:
                    need(all(item[k] == projections[key]['pin'][k] for k in ('path','size_bytes','sha256')),
                         'Same metadata path/projection carries conflicting raw pins')
                annuals[str(year)][version+'_'+name] = key
    json_pins = dict(base_catalog=pin(base_path), candidate_catalog=pin(candidate_path),
        old_registry=pin(registry_path), old_metadata_audit=pin(audit_path))
    result = dict(schema='dmv2-terminal-metadata-plan/v1', status='SEALED_BEFORE_METADATA_TABLE_READ',
        worker_pin=pin(__file__), control_pin=pin(ROOT/'control.py'), json_pins=json_pins,
        annuals=annuals, projections=projections, projected_compressed_bytes=
            sum(s['projected_compressed_bytes'] for s in projections.values()) + sum(x['size_bytes'] for x in json_pins.values()),
        target_columns=list(TARGET), original_columns=list(ORIGINAL),
        selected_financial_columns=[], class_return_price_model_values_read=False,
        target_definition_policy='unchanged_provider_adjusted_endpoints_no_terminal_wealth_substitution')
    return save_new(output, result)


def run(plan_path, expected_sha, reservation, output):
    need(digest(plan_path) == expected_sha, 'Terminal metadata plan changed')
    plan_pin = pin(plan_path)
    z = read(plan_path); need(z['schema'] == 'dmv2-terminal-metadata-plan/v1'
        and z['status'] == 'SEALED_BEFORE_METADATA_TABLE_READ'
        and z['worker_pin'] == pin(__file__), 'Unsealed terminal metadata worker/plan')
    check(z['control_pin'])
    require_reservation(ROOT, reservation, z['projected_compressed_bytes'], pin(plan_path), 'compressed_source')
    guard_path = os.environ.get('DMV2_GUARD_LAUNCH_PATH')
    need(guard_path and digest(guard_path) == os.environ.get('DMV2_GUARD_LAUNCH_SHA256'), 'Authenticated external resource guard required')
    launch = read(guard_path)
    need(launch['command'][1:] == sys.argv and any(p == pin(__file__) for p in launch['code_pins']),
         'External launch command/worker differs')
    need(not Path(output).exists(), 'Preserve existing terminal guard attempt')
    save_new(str(output)+'.STARTED.json', dict(status='ATTEMPT_CONSUMED', plan_pin=pin(plan_path), financial_columns=[]))
    metadata = {name: read(check(p)) for name, p in z['json_pins'].items()}
    events = registry_checks(metadata['old_registry'], metadata['old_metadata_audit'])
    import pyarrow.parquet as pq
    tables = {}
    for key, spec in z['projections'].items():
        need(spec['columns'] in (list(TARGET), list(ORIGINAL)), 'Unapproved metadata column projection')
        t = pq.ParquetFile(check(spec['pin'])).read(columns=spec['columns'])
        need(t.num_rows == spec['rows'], 'Metadata source rowcount changed')
        tables[key] = t.to_pylist()
    book = {name: [] for name in ('base_original_cohort','candidate_original_cohort','base_targets','candidate_targets')}
    for year, refs in z['annuals'].items():
        for version in ('base','candidate'):
            original = tables[refs[version+'_original_cohort']]
            targets = tables[refs[version+'_targets']]
            need(set(lookup(original)) == set(validate_target_rows(targets,int(year))), 'Annual target/original keys differ')
        for name in book: book[name].extend(tables[refs[name]])
    need(len(book['base_original_cohort']) == len(book['candidate_original_cohort']) == 152307,
         'Original152307 historical constituent-month rows required')
    reports = audit_events(events, book['base_original_cohort'], book['candidate_original_cohort'],
        book['base_targets'], book['candidate_targets'])
    for p in z['json_pins'].values(): check(p)
    for spec in z['projections'].values(): check(spec['pin'])
    check(z['worker_pin']); check(z['control_pin']); check(plan_pin)
    result = dict(schema='dmv2-terminal-metadata-guard/v1', status='PASS', events_reviewed=29,
        event_family_counts=FAMILIES, target_definition_unchanged=True, missing_terminal_targets_imputed=False,
        plan_pin=pin(plan_path), worker_pin=pin(__file__), candidate_catalog_pin=z['json_pins']['candidate_catalog'],
        base_catalog_pin=z['json_pins']['base_catalog'], old_registry_pin=z['json_pins']['old_registry'],
        old_metadata_audit_pin=z['json_pins']['old_metadata_audit'], projections=z['projections'],
        original_keys_unchanged=True, original_rows=152307, terminal_overlap_rows_checked=29,
        missing_terminal_overlap_rows=29, post_effective_original_membership_rows=0,
        events=reports, financial_columns_read=[], target_return_or_class_values_read=False,
        nullable_source_status_counts={version:{field:sum(r[field] is None for r in book[version+'_targets'])
            for field in ('source_return_status','return_observation_status')} for version in ('base','candidate')},
        nullable_status_policy='Preserved as unknown only on unavailable targets; never accepted as evidence for the29terminal overlaps',
        model_fits=0, preprocessor_fits=0, inference_calls=0, backtests=0,
        limits='Metadata-only preservation of known endpoint-target policy; does not resolve terminal economic returns or certify all issuer identities.')
    return save_new(output, result)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['plan','run'])
    p.add_argument('--candidate-catalog');p.add_argument('--base-catalog',default=str(BASE))
    p.add_argument('--output',required=True);p.add_argument('--plan');p.add_argument('--sha256');p.add_argument('--reservation')
    args=p.parse_args()
    result=make_plan(args.candidate_catalog,args.output,args.base_catalog) if args.action=='plan' else run(args.plan,args.sha256,args.reservation,args.output)
    print(json.dumps(result,sort_keys=True))
