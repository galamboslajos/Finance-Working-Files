import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/audit_learning_metadata.py'
spec = importlib.util.spec_from_file_location('learning_metadata_audit', SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def sample_row(year=2020, month=1, source='source_a', native='sec_a'):
    prev = f'{year-1}-12' if month == 1 else f'{year}-{month-1:02d}'
    return {'year': year, 'source_security_id': source, 'signal_date': prev + '-28',
        'intrinio_security_id': native, 'outcome_month': f'{year}-{month:02d}',
        'outcome_start': f'{year}-{month:02d}-01', 'outcome_end': f'{year}-{month:02d}-28',
        'membership_tickers': ['OLD'], 'indices': ['sp500'],
        'identity_eligible': True, 'identity_status': 'resolved_review',
        'resolution_status': 'unique', 'identity_review_required': True,
        'identity_eligibility_reason': 'provisionally_admitted',
        'prediction_eligible': True, 'paper_prediction_eligible': True,
        'target_available': True, 'return_available': True, 'observed_return_available': True,
        'target_status': 'available', 'return_status': 'ok',
        'return_observation_status': 'available', 'endpoint_failure_reasons': [],
        'daily_coverage_status': 'complete', 'daily_quote_path_complete': True,
        'missing_daily_dates': [], 'invalid_daily_dates': [], 'interior_gap_dates': [],
        'feature_status': 'available', 'available_feature_count': 715,
        'registered_feature_count': 728}


def frames(rows):
    result = {k: [] for k in m.PROJECTIONS}
    for row in rows:
        common = {'requested': True, 'outcome_requested': True,
            'source_return_status': row['return_status'], 'source_identity_status': row['identity_status'],
            'identity_eligibility_policy': 'provisional', 'unified_security_id': 'unified_a',
            'universe': 'combined', 'start_frequency': 'daily', 'end_frequency': 'daily',
            'start_intraperiod': False, 'end_intraperiod': False,
            'start_price_quality_status': 'accepted', 'end_price_quality_status': 'accepted',
            'observed_daily_session_count': 20, 'expected_daily_session_count': 20,
            'daily_backtest_ready': False, 'knowledge_time_verified': False,
            'accounting_certified': False, 'model_ready': False, **row}
        for kind, fields in m.PROJECTIONS.items():
            result[kind].append({k: copy.deepcopy(common[k]) for k in fields})
    return result


def event(native='sec_a', effective='2020-01-15', event_id='event_a'):
    return {'event_id': event_id, 'predecessor_id': native, 'successor_id': None,
        'effective_date': effective, 'legal_effective_date': effective, 'family': 'current_reviewed'}


class PureMetadataTests(unittest.TestCase):
    def test_projection_excludes_prices_returns_labels_and_features(self):
        self.assertFalse(set().union(*(set(p) for p in m.PROJECTIONS.values())) & m.FORBIDDEN)
        self.assertNotIn('target_class', m.PROJECTIONS['targets'])
        self.assertNotIn('provider_adjusted_return', m.PROJECTIONS['monthly_returns'])

    def test_all_twelve_original_months_and_review_flags_retained(self):
        rows = [sample_row(month=i) for i in range(1, 13)]
        merged = m.merge_year(2020, frames(rows))
        self.assertEqual(len(merged), 12)
        self.assertTrue(all(r['identity_review_required'] for r in merged))

    def test_original_key_loss_is_rejected(self):
        f = frames([sample_row(month=i) for i in range(1, 13)])
        f['targets'].pop()
        with self.assertRaisesRegex(m.ContractError, 'keys lost'):
            m.merge_year(2020, f)

    def test_duplicated_source_signal_is_rejected(self):
        row = sample_row()
        with self.assertRaisesRegex(m.ContractError, 'Duplicate'):
            m.keyed([row, row])

    def test_protected_future_clock_rejected(self):
        f = frames([sample_row(month=i) for i in range(1, 13)])
        for kind in ('targets', 'monthly_returns'):
            f[kind][0]['outcome_end'] = '2024-01-02'
        with self.assertRaisesRegex(m.ContractError, 'Future'):
            m.merge_year(2020, f)

    def test_dated_route_disagreement_rejected(self):
        f = frames([sample_row(month=i) for i in range(1, 13)])
        f['feature_coverage'][0]['intrinio_security_id'] = 'other'
        with self.assertRaisesRegex(m.ContractError, 'route metadata mismatch'):
            m.merge_year(2020, f)

    def test_pinned_clock_disagreement_rejected(self):
        rows = [sample_row(month=i) for i in range(1, 13)]
        windows = [{k: r[k] for k in ('signal_date', 'outcome_month', 'outcome_start', 'outcome_end')} for r in rows]
        windows[0]['signal_date'] = '2019-12-31'
        with self.assertRaisesRegex(m.ContractError, 'pinned source clock'):
            m.merge_year(2020, frames(rows), windows)

    def test_identity_and_paper_cohort_combination(self):
        rows = [sample_row(month=i) for i in range(1, 13)]
        rows[0].update(identity_eligible=False, prediction_eligible=False,
            target_available=False, intrinio_security_id=None)
        merged = m.merge_year(2020, frames(rows))
        self.assertEqual(m.summary(merged)['nonadmitted_identity_rows'], 1)
        self.assertTrue(merged[0]['paper_prediction_eligible'])

    def test_collisions_are_diagnosed_not_dropped(self):
        a = sample_row(source='a')
        b = sample_row(source='b')
        detail, collisions = m.route_diagnostics([a, b])
        self.assertEqual(len(detail), 2)
        self.assertEqual(len(collisions), 1)
        self.assertEqual(collisions[0]['source_security_ids'], ['a', 'b'])
        self.assertTrue(collisions[0]['all_original_rows_preserved'])

    def test_native_change_across_dates_is_not_signal_collision(self):
        a = sample_row(native='native_old')
        b = sample_row(month=2, native='native_new')
        detail, collisions = m.route_diagnostics([a, b])
        self.assertEqual(len(detail), 2)
        self.assertFalse(collisions)

    def test_missing_end_not_automatically_terminal(self):
        r = sample_row()
        r.update(target_available=False, observed_return_available=False,
            return_available=False, endpoint_failure_reasons=['missing_end'],
            return_status='missing_end')
        queue, counts = m.gap_review_queue([r], [])
        self.assertEqual(queue[0]['review_state'], 'unresolved_no_pinned_event_overlap')
        self.assertFalse(queue[0]['terminal_cause_inferred'])
        self.assertEqual(counts['eligible_missing_target_endpoint_counts']['missing_end_no_known_event_overlap_rows'], 1)

    def test_nullable_review_flag_is_unknown_never_passed(self):
        rows = [sample_row(source=str(i)) for i in range(3)]
        for row, flag in zip(rows, (True, False, None)):
            row['identity_review_required'] = flag
        result = m.summary(rows)
        self.assertEqual(result['review_required_rows'], 1)
        self.assertEqual(result['review_required_false_rows'], 1)
        self.assertEqual(result['review_required_unknown_rows'], 1)
        self.assertEqual(result['boolean_flag_counts']['identity_review_required'],
            {'true': 1, 'false': 1, 'unknown': 1})

    def test_nullable_identity_eligibility_remains_contract_error(self):
        rows = [sample_row(month=i) for i in range(1, 13)]
        rows[0].update(identity_eligible=None, prediction_eligible=False, target_available=False)
        with self.assertRaises(m.ContractError):
            m.merge_year(2020, frames(rows))

    def test_eligible_observed_return_missing_target_still_queued(self):
        r = sample_row()
        r['target_available'] = False
        queue, counts = m.gap_review_queue([r], [])
        self.assertEqual(len(queue), 1)
        self.assertEqual(counts['eligible_missing_target_observed_return_available_rows'], 1)

    def test_events_require_exact_native_and_dated_interval(self):
        r = sample_row()
        good = event()
        future = event(effective='2020-02-15', event_id='future')
        wrong = event(native='wrong', event_id='wrong')
        queue_row = {**r, 'target_available': False}
        queue, counts = m.gap_review_queue([queue_row], [good, future, wrong])
        self.assertEqual(queue[0]['known_dated_event_overlap_ids'], ['event_a'])
        overlaps = m.event_overlaps([r], [good, future, wrong])
        self.assertEqual([e['event_outcome_overlap_rows'] for e in overlaps], [1, 0, 0])

    def test_post_event_membership_is_retained_separately(self):
        row = sample_row(month=2)
        overlaps = m.event_overlaps([row], [event()])[0]
        self.assertEqual(overlaps['event_outcome_overlap_rows'], 0)
        self.assertEqual(overlaps['post_effective_original_cohort_rows'], 1)

    def test_path_escape_pin_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(m.ContractError, 'escapes'):
                m.within(d, '../not_allowed')

    def test_mutated_pin_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); p = root / 'x.json'; p.write_text('{}')
            pin = m.pin(p, root)
            p.write_text('[]')
            with self.assertRaisesRegex(m.ContractError, 'bytes changed'):
                m.check_pin(root, pin)

    def test_plan_schema_survives_json_roundtrip(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'targets.parquet'
            data = {k: [frames([sample_row()])['targets'][0][k]] for k in m.PROJECTIONS['targets']}
            data['label_return'] = [999.0]
            pq.write_table(pa.Table.from_pydict(data), p)
            plan = m.schema_plan(p, 'targets')
            self.assertEqual(plan, json.loads(json.dumps(plan)))
            self.assertNotIn('label_return', plan['columns'])

    def test_full_fourteen_year_synthetic_plan_freeze_and_run(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as d:
            top = Path(d); root = top / 'research'; root.mkdir()
            sources = []
            for year in m.YEARS:
                annual = root / 'years' / str(year); annual.mkdir(parents=True)
                rows = [sample_row(year=year, month=i) for i in range(1, 13)]
                rows[0].update(target_available=False, return_status='missing_end',
                    return_available=False, observed_return_available=False,
                    endpoint_failure_reasons=['missing_end'], daily_quote_path_complete=False,
                    daily_coverage_status='incomplete')
                rows[1]['identity_review_required'] = None
                grouped = frames(rows)
                artifacts = {}
                for kind, records in grouped.items():
                    path = annual / (kind + '.parquet')
                    table = pa.Table.from_pylist(records)
                    if kind == 'targets':
                        table = table.append_column('label_return', pa.array([12345.] * 12))
                    pq.write_table(table, path)
                    artifacts[kind] = m.pin(path, root)
                windows = [{k: row[k] for k in ('signal_date', 'outcome_month', 'outcome_start', 'outcome_end')} for row in rows]
                state = annual / 'target_state.json'
                m.write_json(state, {'provenance': {'month_windows': windows}})
                audit = annual / 'audit.json'
                m.write_json(audit, {'year': year, 'status': 'COMPLETE',
                    'build_identity_sha256': 'synthetic', 'artifacts': artifacts})
                sources.append({'year': year, 'directory': str(annual.relative_to(root)),
                    'audit_pin': m.pin(audit, root), 'build_identity_sha256': 'synthetic',
                    'targets_pin': artifacts['targets'], 'target_state_pin': m.pin(state, root)})
            parent = root / 'ret'; parent.mkdir()
            m.write_json(parent / 'build_identity.json', {
                'schema': 'dmv2-ret-backtest-root/v1', 'annual_sources': sources})
            repaired = root / 'repaired' / 'preflight'; repaired.mkdir(parents=True)
            events = [{**event(event_id=f'e{i}'), 'verified': True} for i in range(29)]
            m.write_json(repaired / 'build_identity.json', {
                'schema': 'dmv2-ret-repair-preflight/v1',
                'event_registry': {'events': events[:18]}, 'historical_retirements': events[18:]})
            plan = m.make_plan(root, 'ret', 'repaired')
            plan_path = top / 'plan.json'; m.write_json(plan_path, plan)
            config_path = top / 'config.json'; m.write_json(config_path, {
                'schema': m.CONFIG_SCHEMA, 'plan_sha256': m.sha(plan_path),
                'worker_sha256': m.sha(SCRIPT),
                'parent_ret_identity_sha256': plan['parent_ret_identity_pin']['sha256'],
                'repair_phase_sha256': plan['repair_phase_pin']['sha256'],
                'projected_read_limit_bytes': 128 * 1024**2,
                'rss_limit_bytes': 2 * 1024**3, 'wall_seconds': 600})
            result = m.run(root, plan_path, config_path, top / 'output')
            self.assertEqual(result['months'], 168)
            self.assertEqual(result['summary']['original_rows'], 168)
            self.assertEqual(result['summary']['eligible_missing_target_rows'], 14)
            self.assertEqual(result['summary']['review_required_unknown_rows'], 14)
            self.assertEqual(result['summary']['review_required_rows'], 154)
            self.assertEqual(result['financial_columns_decoded'], [])
            queue = (top / 'output/eligible_missing_target_review_queue.jsonl').read_text().splitlines()
            self.assertEqual(len(queue), 14)
            self.assertNotIn('label_return', json.dumps(result))
            self.assertEqual(len(json.loads((top / 'output/source_month_windows.json').read_text())), 168)


if __name__ == '__main__':
    unittest.main()
