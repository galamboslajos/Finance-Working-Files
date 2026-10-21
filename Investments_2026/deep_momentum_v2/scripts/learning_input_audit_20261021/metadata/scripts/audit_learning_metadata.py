#!/usr/bin/env python3
"""Versioned metadata-only audit of the exact corrected learning cohort.

Planning inspects JSON and Parquet footers only. Execution projects an explicit
allowlist before conversion, never decodes classes, returns, prices or feature
values, and reconciles four independently saved original-key tables per year.
Known legal-event overlaps are candidate dependencies, not invented outcomes.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

PLAN_SCHEMA = 'dmv2-learning-metadata-plan/v1'
CONFIG_SCHEMA = 'dmv2-learning-metadata-config/v1'
RESULT_SCHEMA = 'dmv2-learning-metadata-audit/v1'
YEARS = tuple(range(2010, 2024))
KEYS = ('source_security_id', 'signal_date')
PROJECTIONS = {
 'targets': [*KEYS, 'outcome_month', 'outcome_start', 'outcome_end',
  'identity_eligible', 'requested', 'paper_prediction_eligible',
  'prediction_eligible', 'outcome_requested', 'return_available',
  'observed_return_available', 'target_available', 'target_status',
  'return_observation_status', 'source_return_status'],
 'monthly_returns': [*KEYS, 'intrinio_security_id', 'identity_status',
  'source_identity_status', 'universe', 'identity_review_required',
  'resolution_status', 'identity_eligible', 'identity_eligibility_policy',
  'identity_eligibility_reason', 'outcome_month', 'outcome_start', 'outcome_end',
  'return_status', 'start_frequency', 'end_frequency', 'start_intraperiod',
  'end_intraperiod', 'start_price_quality_status', 'end_price_quality_status',
  'observed_daily_session_count', 'expected_daily_session_count',
  'daily_coverage_status', 'missing_daily_dates', 'invalid_daily_dates',
  'interior_gap_dates', 'endpoint_failure_reasons', 'daily_quote_path_complete',
  'daily_backtest_ready', 'knowledge_time_verified', 'accounting_certified'],
 'original_cohort': [*KEYS, 'indices', 'membership_tickers',
  'unified_security_id', 'intrinio_security_id', 'identity_status',
  'identity_review_required', 'resolution_status', 'source_identity_status',
  'identity_eligible', 'identity_eligibility_policy', 'identity_eligibility_reason'],
 'feature_coverage': [*KEYS, 'intrinio_security_id', 'identity_status',
  'source_identity_status', 'identity_eligible', 'identity_eligibility_policy',
  'identity_eligibility_reason', 'identity_review_required', 'resolution_status',
  'paper_prediction_eligible', 'requested', 'knowledge_time_verified',
  'accounting_certified', 'available_feature_count', 'registered_feature_count',
  'feature_status', 'model_ready'],
}
FORBIDDEN = {'target_class', 'label_return', 'provider_adjusted_return',
 'open', 'high', 'low', 'close', 'adj_close', 'volume', 'adj_volume', 'factor',
 'start_adj_close', 'end_adj_close', 'start_close', 'end_close'}


class ContractError(RuntimeError):
    pass


def need(test, message):
    if not test:
        raise ContractError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def pin(path, root):
    path = Path(path).resolve()
    return {'path': str(path.relative_to(Path(root).resolve())),
            'sha256': sha(path), 'size_bytes': path.stat().st_size}


def within(root, raw):
    root = Path(root).resolve()
    path = (root / raw).resolve() if not Path(raw).is_absolute() else Path(raw).resolve()
    need(path.is_relative_to(root), 'Source path escapes approved project root')
    return path


def check_pin(root, item):
    path = within(root, item['path'])
    need(path.is_file() and path.stat().st_size == item['size_bytes'], 'Pinned source size changed')
    need(sha(path) == item['sha256'], 'Pinned source bytes changed')
    return path


def read_json(path):
    need(Path(path).stat().st_size <= 4 * 1024**2, 'Unbounded JSON metadata')
    return json.loads(Path(path).read_text())


def artifact_path(root, directory, item):
    raw = item['path']
    direct = within(root, raw)
    relative = within(root, str(Path(directory) / raw))
    return direct if direct.exists() else relative


def schema_plan(path, kind):
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    columns = list(PROJECTIONS[kind])
    need(not set(columns) & FORBIDDEN, 'Forbidden financial projection')
    need(set(columns) <= set(pf.schema_arrow.names), f'{kind} lacks required metadata columns')
    need(pf.metadata.num_rows <= 15000, 'Annual metadata table row bound exceeded')
    amount = 0
    for i in range(pf.metadata.num_row_groups):
        group = pf.metadata.row_group(i)
        for j in range(group.num_columns):
            column = group.column(j)
            if column.path_in_schema.split('.')[0] in columns:
                amount += column.total_compressed_size
    return {'kind': kind, 'columns': columns, 'rows': pf.metadata.num_rows,
            'row_groups': pf.metadata.num_row_groups,
            'projected_compressed_bytes': amount,
            'schema': [[c.name, str(c.type)] for c in pf.schema_arrow if c.name in columns]}


def make_plan(root, parent_ret, repair_run):
    root = Path(root).resolve()
    original_path = within(root, str(Path(parent_ret) / 'build_identity.json'))
    phase_path = within(root, str(Path(repair_run) / 'preflight/build_identity.json'))
    original, phase = read_json(original_path), read_json(phase_path)
    need(original.get('schema') == 'dmv2-ret-backtest-root/v1', 'Wrong corrected annual source chain')
    need(phase.get('schema') == 'dmv2-ret-repair-preflight/v1', 'Wrong reviewed event registry source')
    sources = original['annual_sources']
    need([r['year'] for r in sources] == list(YEARS), 'Need all fourteen exact annual sources')
    years = []
    for source in sources:
        audit_path = check_pin(root, source['audit_pin'])
        audit = read_json(audit_path)
        need(audit['status'] in ('COMPLETE', 'PASS', 'SEALED'), 'Annual source is not completed')
        need(audit['year'] == source['year'] and audit['build_identity_sha256'] == source['build_identity_sha256'], 'Annual identity mismatch')
        tables = []
        for kind in PROJECTIONS:
            entry = audit['artifacts'][kind]
            path = artifact_path(root, source['directory'], entry)
            item = {**entry, 'path': str(path.relative_to(root))}
            check_pin(root, item)
            if kind == 'targets':
                need(item['sha256'] == source['targets_pin']['sha256'], 'Targets do not match original RET source chain')
            tables.append({**schema_plan(path, kind), 'pin': item})
        need(len({t['rows'] for t in tables}) == 1, 'Annual original table row counts disagree')
        state_path = check_pin(root, source['target_state_pin'])
        windows = read_json(state_path)['provenance']['month_windows']
        need(len(windows) == 12 and [w['outcome_month'] for w in windows] ==
             [f"{source['year']}-{m:02d}" for m in range(1, 13)], 'Pinned monthly source clocks are incomplete')
        years.append({'year': source['year'], 'directory': source['directory'],
            'build_identity_sha256': source['build_identity_sha256'],
            'audit_pin': source['audit_pin'], 'target_state_pin': source['target_state_pin'],
            'month_windows': windows, 'tables': tables})
    events = []
    for family, records in [('current_reviewed', phase['event_registry']['events']),
                            ('historical_retirement', phase['historical_retirements'])]:
        for event in records:
            need(event.get('verified') is True, 'Unreviewed event cannot be treated as known overlap')
            events.append({k: event.get(k) for k in ('event_id', 'predecessor_id',
                'successor_id', 'effective_date', 'legal_effective_date')} | {'family': family})
    need(len(events) == 29 and len({e['event_id'] for e in events}) == 29, 'Expected eighteen reviewed plus eleven prior legal events')
    return {'schema': PLAN_SCHEMA, 'project_root': str(root),
        'parent_ret_identity_pin': pin(original_path, root), 'repair_phase_pin': pin(phase_path, root),
        'years': years, 'known_events': events,
        'projected_compressed_bytes': sum(t['projected_compressed_bytes'] for y in years for t in y['tables']),
        'integrity_hashed_parquet_bytes': sum(t['pin']['size_bytes'] for y in years for t in y['tables']),
        'financial_columns_decoded': [], 'footer_planning_only': True,
        'canonical_identity_promoted': False,
        'coverage_scope': 'all_original_historical_sp500_sp400_monthly_cohorts_2010_2023'}


def keyed(rows):
    result = {}
    for row in rows:
        key = tuple(row[k] for k in KEYS)
        need(key not in result, 'Duplicate original source/signal key')
        result[key] = row
    return result


def merge_year(year, frames, windows=None):
    indexed = {kind: keyed(rows) for kind, rows in frames.items()}
    original = indexed['original_cohort']
    need(all(set(other) == set(original) for other in indexed.values()), 'Original cohort keys lost or added by a saved table')
    result = []
    for key, cohort in original.items():
        monthly, target, coverage = (indexed[k][key] for k in ('monthly_returns', 'targets', 'feature_coverage'))
        month = monthly['outcome_month']
        need(month.startswith(str(year) + '-') and 1 <= int(month[5:]) <= 12, 'Outcome month outside approved annual boundary')
        need(target['outcome_month'] == month and target['outcome_start'] == monthly['outcome_start']
             and target['outcome_end'] == monthly['outcome_end'], 'Independent saved outcome clocks disagree')
        need(monthly['signal_date'] < monthly['outcome_start'] <= monthly['outcome_end'] < '2024-01-01', 'Future or overlapping outcome clock')
        need(month == monthly['outcome_start'][:7] == monthly['outcome_end'][:7], 'Outcome endpoints leave their declared month')
        if windows is not None:
            window = next((w for w in windows if w['outcome_month'] == month), None)
            need(window is not None and all(monthly[k] == window[k] for k in
                ('signal_date', 'outcome_month', 'outcome_start', 'outcome_end')), 'Rows disagree with pinned source clock window')
        need(cohort['indices'] and set(cohort['indices']) <= {'sp500', 'sp400'}, 'Unexpected research index membership')
        for field in ('intrinio_security_id', 'identity_eligible'):
            need(cohort[field] == monthly[field] == coverage[field], 'Dated original route metadata mismatch')
        need(target['identity_eligible'] == monthly['identity_eligible'], 'Target identity eligibility differs')
        need(target['paper_prediction_eligible'] == coverage['paper_prediction_eligible'], 'Signal population differs between targets and features')
        need(target['prediction_eligible'] == (target['identity_eligible'] and target['paper_prediction_eligible']), 'Prediction mask violates identity/Paper15 contract')
        for field in ('identity_eligible', 'prediction_eligible', 'target_available', 'return_available', 'observed_return_available'):
            need(type(target[field]) is bool, 'Target availability metadata must be explicit boolean')
        need(not target['target_available'] or target['prediction_eligible'], 'Noneligible target was marked available')
        result.append({'year': year, 'source_security_id': key[0], 'signal_date': key[1],
            'intrinio_security_id': cohort['intrinio_security_id'], 'outcome_month': month,
            'outcome_start': monthly['outcome_start'], 'outcome_end': monthly['outcome_end'],
            'membership_tickers': cohort['membership_tickers'], 'indices': cohort['indices'],
            'identity_eligible': monthly['identity_eligible'],
            'identity_status': monthly['identity_status'], 'resolution_status': monthly['resolution_status'],
            'identity_review_required': monthly['identity_review_required'],
            'identity_eligibility_reason': monthly['identity_eligibility_reason'],
            'prediction_eligible': target['prediction_eligible'], 'paper_prediction_eligible': target['paper_prediction_eligible'],
            'target_available': target['target_available'], 'return_available': target['return_available'],
            'observed_return_available': target['observed_return_available'],
            'target_status': target['target_status'], 'return_status': monthly['return_status'],
            'return_observation_status': target['return_observation_status'],
            'endpoint_failure_reasons': monthly['endpoint_failure_reasons'] or [],
            'daily_coverage_status': monthly['daily_coverage_status'],
            'daily_quote_path_complete': monthly['daily_quote_path_complete'],
            'missing_daily_dates': monthly['missing_daily_dates'], 'invalid_daily_dates': monthly['invalid_daily_dates'],
            'interior_gap_dates': monthly['interior_gap_dates'],
            'feature_status': coverage['feature_status'],
            'available_feature_count': coverage['available_feature_count'],
            'registered_feature_count': coverage['registered_feature_count']})
    months = sorted({r['outcome_month'] for r in result})
    need(months == [f'{year}-{m:02d}' for m in range(1, 13)], 'Annual metadata must preserve all twelve outcome months')
    return result


def summary(rows):
    fields = ('identity_status', 'resolution_status', 'identity_eligibility_reason',
        'target_status', 'return_status', 'return_observation_status',
        'daily_coverage_status', 'feature_status')
    boolean_fields = ('identity_eligible', 'prediction_eligible', 'paper_prediction_eligible',
        'observed_return_available', 'return_available', 'target_available', 'identity_review_required',
        'daily_quote_path_complete')
    return {'original_rows': len(rows),
        'identity_eligible_rows': sum(r['identity_eligible'] is True for r in rows),
        'prediction_eligible_rows': sum(r['prediction_eligible'] is True for r in rows),
        'paper_prediction_eligible_rows': sum(r['paper_prediction_eligible'] is True for r in rows),
        'observed_return_available_rows': sum(r['observed_return_available'] is True for r in rows),
        'target_available_rows': sum(r['target_available'] is True for r in rows),
        'eligible_missing_target_rows': sum(r['prediction_eligible'] and not r['target_available'] for r in rows),
        'all_original_missing_target_rows': sum(not r['target_available'] for r in rows),
        'no_native_route_rows': sum(not r['intrinio_security_id'] for r in rows),
        'nonadmitted_identity_rows': sum(not r['identity_eligible'] for r in rows),
        'review_required_rows': sum(r['identity_review_required'] is True for r in rows),
        'review_required_false_rows': sum(r['identity_review_required'] is False for r in rows),
        'review_required_unknown_rows': sum(r['identity_review_required'] is None for r in rows),
        'boolean_flag_counts': {field: {'true': sum(r[field] is True for r in rows),
            'false': sum(r[field] is False for r in rows),
            'unknown': sum(r[field] is None for r in rows)} for field in boolean_fields},
        'complete_daily_path_rows': sum(r['daily_quote_path_complete'] is True for r in rows),
        'incomplete_daily_path_rows': sum(r['daily_coverage_status'] == 'incomplete' for r in rows),
        'missing_start_endpoint_rows': sum('missing_start' in r['endpoint_failure_reasons'] for r in rows),
        'missing_end_endpoint_rows': sum('missing_end' in r['endpoint_failure_reasons'] for r in rows),
        'eligible_missing_end_endpoint_rows': sum(r['prediction_eligible'] and 'missing_end' in r['endpoint_failure_reasons'] for r in rows),
        'endpoint_failure_reasons': dict(sorted(Counter(x for r in rows for x in r['endpoint_failure_reasons']).items())),
        'missing_daily_date_occurrences': sum(len(r['missing_daily_dates'] or []) for r in rows),
        'invalid_daily_date_occurrences': sum(len(r['invalid_daily_dates'] or []) for r in rows),
        'interior_gap_date_occurrences': sum(len(r['interior_gap_dates'] or []) for r in rows),
        **{field + '_counts': dict(sorted(Counter(str(r[field]) for r in rows).items())) for field in fields}}


def route_diagnostics(rows):
    collisions = defaultdict(list)
    routes = defaultdict(list)
    for row in rows:
        routes[(row['source_security_id'], row['intrinio_security_id'], row['year'])].append(row)
        if row['intrinio_security_id']:
            collisions[(row['signal_date'], row['intrinio_security_id'])].append(row)
    collision_keys = set()
    collision_rows = []
    for (signal, native), group in sorted(collisions.items()):
        sources = sorted({r['source_security_id'] for r in group})
        if len(sources) > 1:
            collision_keys.update((r['source_security_id'], signal) for r in group)
            collision_rows.append({'signal_date': signal, 'intrinio_security_id': native,
                'source_security_ids': sources, 'all_original_rows_preserved': True,
                'identity_promoted': False})
    detailed = []
    for (source, native, year), group in sorted(routes.items(), key=lambda x: (x[0][2], x[0][0], x[0][1] or '')):
        detailed.append({'year': year, 'source_security_id': source,
            'intrinio_security_id': native,
            'membership_tickers': sorted({t for r in group for t in (r['membership_tickers'] or [])}),
            'first_signal_date': min(r['signal_date'] for r in group),
            'last_signal_date': max(r['signal_date'] for r in group),
            'signal_route_collision_rows': sum((r['source_security_id'], r['signal_date']) in collision_keys for r in group),
            **summary(group)})
    return detailed, collision_rows


def event_overlaps(rows, events):
    by_native = defaultdict(list)
    for row in rows:
        by_native[row['intrinio_security_id']].append(row)
    outputs = []
    for event in events:
        candidates = by_native.get(event['predecessor_id'], [])
        effective = event['effective_date']
        overlaps = [r for r in candidates if r['signal_date'] < effective <= r['outcome_end']]
        post = [r for r in candidates if r['signal_date'] >= effective]
        outputs.append({**event, 'original_predecessor_cohort_rows': len(candidates),
            'event_outcome_overlap_rows': len(overlaps),
            'eligible_event_outcome_overlap_rows': sum(r['prediction_eligible'] for r in overlaps),
            'missing_target_event_overlap_rows': sum(not r['target_available'] for r in overlaps),
            'eligible_missing_target_event_overlap_rows': sum(r['prediction_eligible'] and not r['target_available'] for r in overlaps),
            'post_effective_original_cohort_rows': len(post),
            'post_effective_prediction_eligible_rows': sum(r['prediction_eligible'] for r in post),
            'overlap_rows': [{k: r[k] for k in (*KEYS, 'outcome_month', 'intrinio_security_id',
                'membership_tickers', 'prediction_eligible', 'target_available',
                'return_status', 'endpoint_failure_reasons')} for r in overlaps],
            'post_effective_rows': [{k: r[k] for k in (*KEYS, 'outcome_month', 'intrinio_security_id',
                'membership_tickers', 'prediction_eligible', 'target_available',
                'return_status', 'endpoint_failure_reasons')} for r in post],
            'candidate_dependency_only': True, 'new_terminal_classification_performed': False})
    return outputs


def gap_review_queue(rows, events):
    """Retain every eligible missing target; a dated event match is not a cause."""
    by_native = defaultdict(list)
    for event in events:
        by_native[event['predecessor_id']].append(event)
    matched = {}
    for row in rows:
        matched[(row['source_security_id'], row['signal_date'])] = sorted(
            e['event_id'] for e in by_native.get(row['intrinio_security_id'], [])
            if row['signal_date'] < e['effective_date'] <= row['outcome_end'])
    queue = []
    for row in rows:
        if row['prediction_eligible'] and not row['target_available']:
            ids = matched[(row['source_security_id'], row['signal_date'])]
            queue.append({k: row[k] for k in ('year', *KEYS, 'intrinio_security_id',
                'outcome_month', 'outcome_start', 'outcome_end', 'membership_tickers',
                'indices', 'identity_eligible', 'identity_status', 'resolution_status',
                'identity_review_required', 'prediction_eligible', 'return_available',
                'observed_return_available', 'target_available', 'target_status',
                'return_status', 'return_observation_status', 'endpoint_failure_reasons',
                'daily_coverage_status', 'missing_daily_dates', 'invalid_daily_dates',
                'interior_gap_dates')} | {'known_dated_event_overlap_ids': ids,
                'review_state': 'known_event_overlap_causation_not_certified' if ids else
                    'unresolved_no_pinned_event_overlap',
                'terminal_cause_inferred': False})
    def endpoint_counts(candidates):
        out = {}
        for label in ('missing_start', 'missing_end'):
            subset = [r for r in candidates if label in r['endpoint_failure_reasons']]
            out[label + '_rows'] = len(subset)
            out[label + '_known_event_overlap_rows'] = sum(bool(matched[(r['source_security_id'], r['signal_date'])]) for r in subset)
            out[label + '_no_known_event_overlap_rows'] = sum(not matched[(r['source_security_id'], r['signal_date'])] for r in subset)
        return out
    return queue, {
        'eligible_missing_target_rows': len(queue),
        'eligible_missing_target_observed_return_available_rows': sum(r['observed_return_available'] for r in queue),
        'eligible_missing_target_known_event_overlap_rows': sum(bool(r['known_dated_event_overlap_ids']) for r in queue),
        'eligible_missing_target_no_known_event_overlap_rows': sum(not r['known_dated_event_overlap_ids'] for r in queue),
        'eligible_missing_target_endpoint_counts': endpoint_counts(queue),
        'all_original_endpoint_counts': endpoint_counts(rows),
        'known_event_overlap_is_not_a_terminal_cause_certification': True}


def rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == 'darwin' else value * 1024)


def run(root, plan_path, config_path, output):
    start = time.monotonic()
    plan, config = read_json(plan_path), read_json(config_path)
    need(plan['schema'] == PLAN_SCHEMA and config['schema'] == CONFIG_SCHEMA, 'Wrong frozen audit schemas')
    need(config['plan_sha256'] == sha(plan_path), 'Plan changed after freeze')
    need(config['worker_sha256'] == sha(__file__), 'Worker changed after freeze')
    need(config['parent_ret_identity_sha256'] == plan['parent_ret_identity_pin']['sha256']
         and config['repair_phase_sha256'] == plan['repair_phase_pin']['sha256'], 'Source authorities changed after freeze')
    need(0 < config['projected_read_limit_bytes'] <= 128 * 1024**2 and
         plan['projected_compressed_bytes'] <= config['projected_read_limit_bytes'], 'Metadata projection budget exceeded')
    need(0 < config['rss_limit_bytes'] <= 2 * 1024**3 and 0 < config['wall_seconds'] <= 600, 'Unapproved execution resource bounds')
    root, output = Path(root).resolve(), Path(output).resolve()
    need(str(root) == plan['project_root'], 'Root changed after planning')
    need(not output.exists(), 'Preserve prior attempt: output already exists')
    output.mkdir(parents=True)
    # Persist the complete read reservation before any projected metadata conversion.
    write_json(output / 'read_reservation.json', {'schema': 'dmv2-learning-metadata-read-reservation/v1',
        'plan_pin': {'path': str(Path(plan_path).resolve()), 'sha256': sha(plan_path)},
        'config_pin': {'path': str(Path(config_path).resolve()), 'sha256': sha(config_path)},
        'projected_compressed_bytes': plan['projected_compressed_bytes'],
        'byte_integrity_hashed_parquet_bytes_separate': plan['integrity_hashed_parquet_bytes'],
        'financial_columns_decoded': [], 'tables': [t for y in plan['years'] for t in y['tables']]})
    check_pin(root, plan['parent_ret_identity_pin'])
    check_pin(root, plan['repair_phase_pin'])
    # Check every source and footer again before the first conversion, not adaptively.
    for annual in plan['years']:
        check_pin(root, annual['audit_pin'])
        state_path = check_pin(root, annual['target_state_pin'])
        need(read_json(state_path)['provenance']['month_windows'] == annual['month_windows'], 'Pinned source clock state changed')
        for table in annual['tables']:
            path = check_pin(root, table['pin'])
            need(schema_plan(path, table['kind']) == {k: table[k] for k in
                ('kind', 'columns', 'rows', 'row_groups', 'projected_compressed_bytes', 'schema')}, 'Parquet footer/projection changed')
    import pyarrow.parquet as pq
    all_rows = []
    for annual in plan['years']:
        need(time.monotonic() - start <= config['wall_seconds'] and rss_bytes() <= config['rss_limit_bytes'], 'Audit resource bound exceeded')
        frames = {}
        for table in annual['tables']:
            frames[table['kind']] = pq.ParquetFile(within(root, table['pin']['path'])).read(columns=table['columns'], use_threads=False).to_pylist()
            need(rss_bytes() <= config['rss_limit_bytes'], 'Projected metadata conversion RSS exceeded')
        all_rows.extend(merge_year(annual['year'], frames, annual['month_windows']))
        del frames
    need(len({r['outcome_month'] for r in all_rows}) == 168, 'All original168 months must be retained')
    by_month, by_year = defaultdict(list), defaultdict(list)
    for row in all_rows:
        by_month[row['outcome_month']].append(row)
        by_year[row['year']].append(row)
    routes, collisions = route_diagnostics(all_rows)
    overlaps = event_overlaps(all_rows, plan['known_events'])
    gap_queue, gap_counts = gap_review_queue(all_rows, plan['known_events'])
    result = {'schema': RESULT_SCHEMA, 'status': 'PASS_METADATA_AUDIT',
        'plan_sha256': sha(plan_path), 'config_sha256': sha(config_path),
        'worker_sha256': sha(__file__), 'months': 168, 'years': list(YEARS),
        'source_table_count': 56, 'projected_compressed_bytes': plan['projected_compressed_bytes'],
        'byte_integrity_hashed_parquet_bytes_separate': plan['integrity_hashed_parquet_bytes'],
        'summary': summary(all_rows),
        'monthly': [{'outcome_month': m, 'signal_dates': sorted({r['signal_date'] for r in g}),
            **summary(g)} for m, g in sorted(by_month.items())],
        'annual': [{'year': y, **summary(g)} for y, g in sorted(by_year.items())],
        'route_collision_signal_native_count': len(collisions),
        'route_source_native_year_groups': len(routes),
        'known_legal_event_count': len(overlaps),
        'known_legal_events_with_original_overlap': sum(e['event_outcome_overlap_rows'] > 0 for e in overlaps),
        'gap_review_queue_counts': gap_counts,
        'elapsed_seconds': time.monotonic() - start, 'peak_rss_bytes': rss_bytes(),
        'all_original_keys_preserved': True, 'financial_columns_decoded': [],
        'financial_2024_plus_decoded': False, 'canonical_identity_promoted': False,
        'models_or_inference_loaded': False, 'new_fits': 0,
        'new_terminal_classification_performed': False,
        'limitations': ['Saved metadata/status audit, not numerical return correctness or economic terminal-return certification.',
            'Known legal-event overlaps are dated-route candidate dependencies; absence of overlap is not source completeness.',
            'Unexplained endpoint gaps remain unexplained, never relabelled as delistings by this audit.',
            'Original rows, missing outcomes and provisional identity flags are retained.',
            'No daily OHLCV, feature values, class IDs, model probabilities or financial returns decoded.']}
    need(result['peak_rss_bytes'] <= config['rss_limit_bytes'] and result['elapsed_seconds'] <= config['wall_seconds'], 'Final audit resource bound exceeded')
    write_json(output / 'audit.json', result)
    write_json(output / 'route_collisions.json', collisions)
    write_json(output / 'known_event_overlaps.json', overlaps)
    write_json(output / 'source_month_windows.json', [w for annual in plan['years'] for w in annual['month_windows']])
    with (output / 'source_native_year_diagnostics.jsonl').open('w') as handle:
        for item in routes:
            handle.write(canonical(item).decode() + '\n')
    with (output / 'eligible_missing_target_review_queue.jsonl').open('w') as handle:
        for item in sorted(gap_queue, key=lambda r: (r['outcome_month'], r['source_security_id'])):
            handle.write(canonical(item).decode() + '\n')
    render_summary(output / 'README.md', result)
    return result


def write_json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')


def render_summary(path, result):
    lines = ['# Full original-cohort learning-input metadata audit', '',
        '2010–2023; 168 outcome months. This is a metadata audit, not a rebuild or numerical return certification.', '',
        '| Year | Original | Identity eligible | Prediction eligible | Labels available | Eligible missing labels | Missing start | Missing end | Incomplete daily paths |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in result['annual']:
        lines.append('| ' + ' | '.join(str(r[k]) for k in ('year', 'original_rows', 'identity_eligible_rows',
            'prediction_eligible_rows', 'target_available_rows', 'eligible_missing_target_rows',
            'missing_start_endpoint_rows', 'missing_end_endpoint_rows', 'incomplete_daily_path_rows')) + ' |')
    lines.extend(['', f"Known legal events: {result['known_legal_event_count']}; original-cohort event-month overlaps for {result['known_legal_events_with_original_overlap']} events.",
        f"Signal/native route collisions: {result['route_collision_signal_native_count']}. Rows and review flags are preserved.", '',
        'Missing endpoints are not automatically terminal events. See per-month counters, source/native/year diagnostics and explicit known-event overlap records.', '',
        *['- ' + item for item in result['limitations']], ''])
    Path(path).write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('plan')
    p.add_argument('--root', required=True); p.add_argument('--parent-ret', required=True)
    p.add_argument('--repair-run', required=True); p.add_argument('--output', required=True)
    p = sub.add_parser('run')
    p.add_argument('--root', required=True); p.add_argument('--plan', required=True)
    p.add_argument('--config', required=True); p.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'plan':
        need(not Path(args.output).exists(), 'Preserve existing plan')
        result = make_plan(args.root, args.parent_ret, args.repair_run)
        write_json(args.output, result)
        print(json.dumps({k: result[k] for k in ('schema', 'projected_compressed_bytes', 'integrity_hashed_parquet_bytes')}))
    else:
        result = run(args.root, args.plan, args.config, args.output)
        print(json.dumps({k: result[k] for k in ('status', 'months', 'summary', 'peak_rss_bytes')}))


if __name__ == '__main__':
    main()
