#!/usr/bin/env python3
"""Independent consistency review of saved metadata-only audit outputs.

No Parquet reads, model loading, numerical return work or production mutation.
"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parents[1]
RESULT = HERE / 'result-v2'


def read(path):
    return json.loads(path.read_text())


def pin(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024**2), b''):
            h.update(block)
    return {'path': str(path), 'size_bytes': path.stat().st_size, 'sha256': h.hexdigest()}


def need(test, message):
    if not test:
        raise RuntimeError(message)


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def main():
    audit = read(RESULT / 'audit.json')
    events = read(RESULT / 'known_event_overlaps.json')
    queue = [json.loads(line) for line in (RESULT / 'eligible_missing_target_review_queue.jsonl').read_text().splitlines()]
    collisions = read(RESULT / 'route_collisions.json')
    clock = read(RESULT / 'source_month_windows.json')
    plan, cfg = read(HERE / 'plan.json'), read(HERE / 'config-v2.json')
    worker = HERE / 'scripts/audit_learning_metadata.py'
    need(cfg['worker_sha256'] == pin(worker)['sha256'], 'Frozen worker changed')
    need(cfg['plan_sha256'] == pin(HERE / 'plan.json')['sha256'], 'Frozen plan changed')
    need(audit['config_sha256'] == pin(HERE / 'config-v2.json')['sha256'], 'Result config changed')
    need(audit['worker_sha256'] == cfg['worker_sha256'], 'Result worker differs')
    need(audit['plan_sha256'] == cfg['plan_sha256'], 'Result plan differs')
    need(audit['status'] == 'PASS_METADATA_AUDIT' and audit['months'] == len(clock) == 168, 'Incomplete metadata audit')
    need(audit['financial_columns_decoded'] == [] and audit['financial_2024_plus_decoded'] is False, 'Forbidden decoded-value claim')
    need(audit['peak_rss_bytes'] <= cfg['rss_limit_bytes'], 'RSS budget exceeded')
    need(audit['elapsed_seconds'] <= cfg['wall_seconds'], 'Wall budget exceeded')
    need(audit['projected_compressed_bytes'] <= cfg['projected_read_limit_bytes'], 'Read budget exceeded')
    count_keys = ('original_rows', 'identity_eligible_rows', 'prediction_eligible_rows',
        'observed_return_available_rows', 'target_available_rows', 'eligible_missing_target_rows',
        'missing_start_endpoint_rows', 'missing_end_endpoint_rows', 'review_required_rows',
        'review_required_false_rows', 'review_required_unknown_rows', 'incomplete_daily_path_rows')
    for key in count_keys:
        need(sum(row[key] for row in audit['annual']) == audit['summary'][key], f'Annual total differs: {key}')
        need(sum(row[key] for row in audit['monthly']) == audit['summary'][key], f'Monthly total differs: {key}')
    need(len(queue) == audit['summary']['eligible_missing_target_rows'] == 293, 'Missing-target queue incomplete')
    need(len({(r['source_security_id'], r['signal_date']) for r in queue}) == len(queue), 'Duplicate queued key')
    for row in queue:
        need(row['identity_eligible'] is True and row['prediction_eligible'] is True
             and row['target_available'] is False, 'Queued row violates retained cohort')
        expected = sorted(e['event_id'] for e in plan['known_events']
            if e['predecessor_id'] == row['intrinio_security_id']
            and row['signal_date'] < e['effective_date'] <= row['outcome_end'])
        need(row['known_dated_event_overlap_ids'] == expected, 'Dated event mapping differs')
        need(row['terminal_cause_inferred'] is False, 'Invented terminal cause')
    by_status = Counter(r['return_status'] for r in queue)
    need(by_status == {'missing_end': 277, 'source_quality_flag': 16}, 'Gap status partition changed')
    matched = [r for r in queue if r['known_dated_event_overlap_ids']]
    need(len(matched) == 29 and all(r['return_status'] == 'missing_end' for r in matched), 'Known-event gap partition changed')
    need(len(events) == 29 and all(e['event_outcome_overlap_rows'] == 1 for e in events), 'Known-event original overlaps differ')
    need(all(e['post_effective_original_cohort_rows'] == e['post_effective_prediction_eligible_rows'] == 0 for e in events), 'Unexpected post-event cohort rows')
    need(len(collisions) == audit['route_collision_signal_native_count'] == 622, 'Route collision partition differs')
    need(sum(len(c['source_security_ids']) for c in collisions) == 1244, 'Collision original-row count differs')
    source = 'siblis_ticker_company_8500ae2d7e99a52f8713'
    rcm, route_totals = [], defaultdict(lambda: {'missing_start': 0, 'missing_end': 0, 'original': 0,
        'eligible': 0, 'years': [], 'tickers': set()})
    with (RESULT / 'source_native_year_diagnostics.jsonl').open() as handle:
        for line in handle:
            row = json.loads(line)
            if row['source_security_id'] == source:
                rcm.append({k: row[k] for k in ('year', 'source_security_id', 'intrinio_security_id',
                    'first_signal_date', 'last_signal_date', 'original_rows', 'prediction_eligible_rows',
                    'target_available_rows', 'eligible_missing_target_rows', 'missing_start_endpoint_rows',
                    'missing_end_endpoint_rows', 'incomplete_daily_path_rows', 'target_status_counts', 'return_status_counts')})
            group = route_totals[(row['source_security_id'], row['intrinio_security_id'])]
            group['missing_start'] += row['missing_start_endpoint_rows']
            group['missing_end'] += row['missing_end_endpoint_rows']
            group['original'] += row['original_rows']
            group['eligible'] += row['prediction_eligible_rows']
            group['years'].append(row['year']); group['tickers'].update(row['membership_tickers'])
    need([r['year'] for r in rcm] == [2021, 2022, 2023], 'RCM saved cohort interval changed')
    rcm_queue = [r for r in queue if r['source_security_id'] == source]
    need(len(rcm_queue) == 1 and rcm_queue[0]['outcome_month'] == '2021-11', 'RCM actionable queued month changed')
    unknown_review = audit['summary']['boolean_flag_counts']['identity_review_required']
    need(unknown_review == {'true': 133729, 'false': 17465, 'unknown': 1113}, 'Unknown review state lost')
    prior_path = Path('/private/tmp/dmv2-ret-repair-20261020/rcm_local/after_control_metadata.json')
    prior = read(prior_path)
    need(prior['financial_columns'] == [], 'Prior after-control receipt was not metadata-only')
    dates = prior['dates']
    broad = {'evidence_pin': pin(prior_path),
        'evidence_status': 'existing_published_source_date_inventory_not_new_acquisition',
        'requested_start': prior['provenance']['request']['start'],
        'requested_end': prior['provenance']['request']['end'],
        'old_source_native': 'sec_yKbEk4', 'old_source_metadata_dates': len(dates.get('sec_yKbEk4', [])),
        'target_native': 'sec_zMM1vd', 'target_metadata_dates': len(dates.get('sec_zMM1vd', [])),
        'complete_raw_adjusted_ohlcv_or_identity_certified': False,
        'note': 'The 22 execution bars end2021-12-01; this separately recorded interval has138old-route dates and0target-route dates. Full dated route, volume and warmup history remain to audit.'}
    tops = []
    for (source_id, native), group in sorted(route_totals.items(),
        key=lambda item: -(item[1]['missing_start'] + item[1]['missing_end']))[:12]:
        tops.append({'source_security_id': source_id, 'intrinio_security_id': native,
            **{k: group[k] for k in ('missing_start', 'missing_end', 'original', 'eligible')},
            'years': sorted(group['years']), 'tickers': sorted(group['tickers']),
            'terminal_cause_inferred': False})
    findings = {'schema': 'dmv2-learning-gap-queue-findings/v1',
        'status': 'PASS_SAVED_METADATA_CONSISTENCY_REVIEW', 'original_rows': 152307,
        'identity_eligible_rows': 149824, 'prediction_eligible_rows': 140680,
        'target_available_rows': 140387, 'eligible_missing_target_rows': len(queue),
        'eligible_gap_statuses': dict(by_status), 'known_dated_event_overlap_rows': 29,
        'missing_end_no_known_event_overlap_rows': 248, 'quality_flag_no_known_event_overlap_rows': 16,
        'eligible_missing_target_observed_return_available_rows': 0,
        'known_reviewed_current_events': 18, 'known_historical_retirement_events': 11,
        'known_events_with_one_original_outcome_overlap': 29,
        'post_effective_original_cohort_rows_for_known_events': 0,
        'identity_review_required_counts': unknown_review,
        'blocked_collision_original_rows': 1244, 'collision_signal_native_groups': 622,
        'queue_unique_source_ids': len({r['source_security_id'] for r in queue}),
        'queue_unique_native_ids': len({r['intrinio_security_id'] for r in queue}),
        'missing_end_unique_source_ids': len({r['source_security_id'] for r in queue if r['return_status'] == 'missing_end'}),
        'largest_queue_months': [{'outcome_month': month, 'rows': n} for month, n in Counter(r['outcome_month'] for r in queue).most_common(8)],
        'quality_flag_rows': [{k: r[k] for k in ('source_security_id', 'intrinio_security_id', 'membership_tickers', 'outcome_month', 'endpoint_failure_reasons')} for r in queue if r['return_status'] == 'source_quality_flag'],
        'rcm_source_security_id': source, 'rcm_saved_original_cohort_years': rcm,
        'rcm_eligible_missing_target_rows': [{k: r[k] for k in ('source_security_id', 'intrinio_security_id', 'signal_date', 'outcome_month', 'outcome_end', 'endpoint_failure_reasons', 'missing_daily_dates')} for r in rcm_queue],
        'rcm_broader_source_date_inventory': broad,
        'top_original_cohort_endpoint_gap_routes': tops,
        'limitations': ['These293records cover existing prediction-eligible missing targets, not all lost feature/warmup coverage.',
            'RCM2022has12originalmonths but0prediction-eligible rows;2023has12observed endpoint returns butonly6prediction-eligible rows.',
            'Known legal-event date overlaps are candidate dependencies, not certifications of corrected economic labels.',
            '248unmatched missing ends and16quality flags remain unresolved; no newterminal cause inferred.',
            'No new financial data, class values, return values, OHLCV, model inference or fits in this review.']}
    write(HERE / 'gap_queue_findings.json', findings)
    inputs = [HERE / n for n in ('plan.json', 'config-v2.json', 'scripts/audit_learning_metadata.py')]
    inputs += [RESULT / n for n in ('audit.json', 'eligible_missing_target_review_queue.jsonl',
        'known_event_overlaps.json', 'route_collisions.json', 'source_month_windows.json',
        'source_native_year_diagnostics.jsonl', 'read_reservation.json')]
    receipt = {'schema': 'dmv2-learning-metadata-independent-review/v1',
        'status': 'PASS_SAVED_METADATA_CONSISTENCY_REVIEW',
        'reviewer_code_pin': pin(Path(__file__)), 'input_pins': [pin(p) for p in inputs],
        'findings_pin': pin(HERE / 'gap_queue_findings.json'),
        'original_rows_retained': 152307, 'outcome_months': 168,
        'eligible_missing_target_queue_rows': 293, 'all29known_event_intervals_checked': True,
        'parquet_tables_redecoded': 0, 'financial_values_decoded': False,
        'identity_or_corporate_action_completeness_certified': False,
        'source_numerical_value_correctness_independently_certified': False}
    write(HERE / 'independent_review.json', receipt)
    print(json.dumps({'status': receipt['status'], 'findings_pin': receipt['findings_pin']}))


if __name__ == '__main__':
    main()
