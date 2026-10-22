"""Metadata-first, monthly-streamed return/drawdown label audit.

Only an explicitly sealed plan dispatches numerical reads. No price, feature,
model, cloud or backtest access is implemented. Existing source files remain
immutable and failed attempts are preserved without automatic retries.
"""
from pathlib import Path
import datetime as dt
import hashlib
import json
import math
import os
import resource
import time

SCHEMA = 'dmv2_return_dd_audit_v1'
SOURCE_AUDIT_SHA256 = '8f09fec5ed2b0404f9518ef4b6be58e50a4a1a7751e9768ed7e49fe320124547'
NUMERIC = ('cash_return', 'maximum_drawdown', 'utility', 'target_value', 'target_class')
VALUE_COLUMNS = ('cash_return', 'maximum_drawdown', 'target_class')
SIDES = ('long', 'short')
FLOOR = .01
CUTOFF = '2023-12-29'


def need(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return 'sha256:' + hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def pin(path):
    path = Path(path).resolve()
    need(path.is_file(), 'Pinned regular file required')
    return {'path': str(path), 'size_bytes': path.stat().st_size, 'sha256': sha256(path)}


def check(item):
    need(set(item) == {'path', 'size_bytes', 'sha256'}, 'Closed pin schema required')
    path = Path(item['path'])
    need(path.is_absolute() and path.is_file()
         and path.stat().st_size == item['size_bytes'] and sha256(path) == item['sha256'],
         'Pinned source bytes changed: ' + str(path))
    return path


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    return pin(path)


def relative_pin(path, root):
    result = pin(path); result['path'] = str(Path(path).resolve().relative_to(Path(root).resolve()))
    return result


def month_range():
    return [f'{year}-{month:02d}' for year in range(2010, 2024) for month in range(1, 13)]


def projection_bytes(pf, columns):
    need(len(columns) == len(set(columns)) and set(columns) <= set(pf.schema_arrow.names),
         'Invalid projected source columns')
    return sum(pf.metadata.row_group(g).column(c).total_compressed_size
        for g in range(pf.metadata.num_row_groups)
        for c in range(pf.metadata.row_group(g).num_columns)
        if pf.metadata.row_group(g).column(c).path_in_schema.split('.')[0] in columns)


def _limits(source_projection_bytes):
    need(type(source_projection_bytes) is int and source_projection_bytes in (64 * 1024**2, 128 * 1024**2),
         'Source projection ceiling must be 64MiB or reviewed 128MiB')
    return {'source_projection_bytes': source_projection_bytes,
            'derived_verification_read_bytes': 128 * 1024**2,
            'output_bytes': 128 * 1024**2, 'rss_stop_bytes': 4 * 1024**3,
            'active_seconds': 600, 'estimator_fits': 0, 'trade_replays': 0}


def _protocol(path, limits):
    need(path is not None, 'Frozen label-engineering protocol required')
    value = read(path)
    need(value.get('schema') == 'dmv2_return_dd_target_protocol_v1'
         and value.get('status') == 'FROZEN_LABEL_ENGINEERING_AND_AUDIT_ONLY'
         and value.get('protocol_id') == 'signed_executable_return_over_drawdown_20261022_v1'
         and value.get('target_family') == 'return_over_drawdown'
         and value.get('formula') == 'cash_return / max(maximum_drawdown, denominator_floor)'
         and value.get('denominator_floor') == FLOOR and value.get('observed_through') == CUTOFF
         and value.get('history') == ['2010-01', '2023-12'], 'Frozen formula/floor/scope changed')
    need(value.get('path', {}).get('learning_variant') == 'unstopped'
         and value['path'].get('sides') == list(SIDES)
         and value.get('ranking', {}).get('classes') == 10
         and value['ranking'].get('best_class') == 1 and value['ranking'].get('minimum_observed') == 10
         and value.get('payoffs', {}).get('calendar_months') == 120
         and value['payoffs'].get('required_month_class_cells') == 1200
         and value.get('training_audit', {}).get('years') == list(range(2020, 2024)),
         'Frozen completion/rank/payoff/training scope changed')
    need(all(value.get('limits', {}).get(key) == expected for key, expected in limits.items())
         and value['limits'].get('processes') == 1
         and value['limits'].get('independent_source_projection_bytes') == limits['source_projection_bytes'],
         'Frozen audit-only compute/read ceilings changed')
    return value


def plan_audit(source_audit, output_directory, plan_path, *, protocol_path=None,
               source_projection_bytes=64 * 1024**2):
    """Authenticate JSON metadata and Parquet footers without numeric decoding."""
    import pyarrow.parquet as pq
    source, output, destination = (Path(x).resolve() for x in (source_audit, output_directory, plan_path))
    limits = _limits(source_projection_bytes)
    _protocol(protocol_path, limits)
    need(not output.exists() and not destination.exists(), 'Fresh output directory and plan required')
    audit_path = source / 'audit.json'; proof_path = source / 'independent_full_verification.json'
    source_plan_path = source / 'reviewed_plan.json'
    need(sha256(audit_path) == SOURCE_AUDIT_SHA256, 'Only independently verified current full-path audit allowed')
    audit, proof, parent_plan = read(audit_path), read(proof_path), read(source_plan_path)
    need(proof.get('status') == 'PASS_FULL_FORMULAS_RANKS_CLOCK_AND_ORIGINAL_DENOMINATORS'
         and proof.get('audit_sha256') == SOURCE_AUDIT_SHA256
         and proof.get('financial_2024_plus_read') is False
         and proof.get('months_checked') == 168 and proof.get('original_rows') == 152307,
         'Matching complete independent source proof required')
    need(audit['plan_sha256'] == parent_plan['plan_sha256'] == proof['plan_sha256'],
         'Source plan/audit/proof lineage mismatch')
    need(parent_plan['plan_sha256'] == canonical({k: v for k, v in parent_plan.items() if k != 'plan_sha256'}),
         'Source reviewed-plan fingerprint mismatch')
    need(len(parent_plan['annuals']) == 14, 'Full fourteen-year source history required')
    windows = {w['outcome_month']: {**w, 'label_available_at': w['exit_date']}
               for annual in parent_plan['annuals'] for w in annual['windows']}
    need(sorted(windows) == month_range(), 'Original full monthly completion clocks required')
    annual_receipts = {}
    annual_pins = {}
    for year in range(2010, 2024):
        annual_path = source / 'years' / str(year) / 'audit.json'
        actual_pin = pin(annual_path); expected = audit['year_audit_pins'][str(year)]
        need(actual_pin['sha256'] == expected['sha256'] and actual_pin['size_bytes'] == expected['size_bytes'],
             'Independent full audit annual receipt changed')
        annual = read(annual_path)
        need(annual['status'] == 'COMPLETE_ANNUAL_PATH_AUDIT'
             and annual['plan_sha256'] == parent_plan['plan_sha256']
             and annual['months'] == [f'{year}-{m:02d}' for m in range(1, 13)],
             'Original annual source audit changed')
        annual_receipts[year] = annual; annual_pins[str(year)] = actual_pin
    months = []
    for month in month_range():
        cp_path = source / 'years' / month[:4] / 'months' / month / 'checkpoint.json'
        cp_pin = pin(cp_path)
        cp_saved = annual_receipts[int(month[:4])]['artifacts'][f'months/{month}/checkpoint.json']
        need(cp_pin['sha256'] == cp_saved['sha256'] and cp_pin['size_bytes'] == cp_saved['size_bytes'],
             'Annual source checkpoint chain changed')
        cp = read(cp_path)
        attempt = cp_path.parent / cp['attempt']
        need(attempt.resolve().parent == cp_path.parent.resolve(), 'Unsafe selected source attempt')
        monthly_audit_path = attempt / 'audit.json'
        need(sha256(monthly_audit_path) == cp['audit_sha256']
             and cp['plan_sha256'] == parent_plan['plan_sha256'], 'Monthly checkpoint mismatch')
        monthly = read(monthly_audit_path)
        need(monthly['status'] == 'COMPLETE_MONTHLY_PATH_AUDIT'
             and monthly['plan_sha256'] == parent_plan['plan_sha256']
             and monthly['outcome_month'] == month, 'Source monthly audit incomplete or redirected')
        target = attempt / 'targets.parquet'; target_pin = pin(target)
        saved = monthly['artifacts']['targets.parquet']
        need(target_pin['sha256'] == saved['sha256'] and target_pin['size_bytes'] == saved['size_bytes'],
             'Source target artifact changed')
        expected_month = next(m for a in parent_plan['annuals'] for m in a['monthly']
                              if m['outcome_month'] == month)
        need(all(monthly[key] == expected_month[key] for key in ('original_rows', 'prediction_eligible_rows',
             'original_key_sha256', 'eligible_key_sha256'))
             and monthly['signal_date'] == windows[month]['signal_date']
             and monthly['exit_date'] == windows[month]['exit_date'],
             'Reviewed original monthly denominator, keys or clock changed')
        pf = pq.ParquetFile(target)
        need(pf.metadata.num_rows == 8 * monthly['original_rows'], 'Source two-family two-variant two-side dimensions changed')
        names = pf.schema_arrow.names
        need(set(NUMERIC) <= set(names), 'Source target numerical schema incomplete')
        metadata = [name for name in names if name not in NUMERIC]
        values = list(VALUE_COLUMNS) if windows[month]['exit_date'] <= CUTOFF else []
        need(month == '2023-12' or values, 'Unexpected protected source window')
        months.append({'outcome_month': month, 'source_target_pin': target_pin,
            'source_checkpoint_pin': pin(cp_path), 'source_audit_pin': pin(monthly_audit_path),
            'metadata_columns': metadata, 'value_columns': values,
            'projected_metadata_bytes': projection_bytes(pf, metadata),
            'projected_value_bytes': projection_bytes(pf, values),
            'source_rows': pf.metadata.num_rows, 'original_rows': monthly['original_rows'],
            'eligible_rows': monthly['prediction_eligible_rows'],
            'source_window': windows[month], 'original_key_sha256': monthly['original_key_sha256'],
            'eligible_key_sha256': monthly['eligible_key_sha256']})
    cost = sum(m['projected_metadata_bytes'] + m['projected_value_bytes'] for m in months)
    need(cost <= limits['source_projection_bytes'], f'Metadata-measured projected source bytes {cost} exceed ceiling {limits["source_projection_bytes"]}; review budget before financial reads')
    need(sum(m['original_rows'] for m in months) == 152307, 'Original source denominator changed')
    sessions = parent_plan['sessions']
    folds = [{'year': year, 'train_start_month': f'{year-10}-01',
              'train_end_month': f'{year-1}-12',
              'as_of': max(day for day in sessions if day <= f'{year-1}-12-31')}
             for year in range(2020, 2024)]
    body = {'schema': SCHEMA, 'status': 'SEALED_BEFORE_FINANCIAL_READ',
        'source_audit_directory': str(source), 'output_directory': str(output),
        'source_full_audit_pin': pin(audit_path), 'source_proof_pin': pin(proof_path),
        'source_plan_pin': pin(source_plan_path), 'source_annual_pins': annual_pins, 'months': months, 'folds': folds,
        'source_filter': {'target_family': 'executable_return', 'path_variant': 'unstopped'},
        'denominator_floor': FLOOR, 'observed_through': CUTOFF,
        'target_family': 'return_over_drawdown', 'source_projection_bytes': cost,
        'limits': limits, 'code_pins': {'controller': pin(__file__),
            'targets': pin(Path(__file__).with_name('return_dd_targets.py')),
            'cli': pin(Path(__file__).parents[2] / 'scripts/run_return_dd_audit.py')},
        'financial_2024_plus_allowed': False, 'models_fitted': False,
        'portfolio_backtest_run': False, 'source_prices_read': False, 'feature_values_read': False,
        'integrity_hashing_separate_from_financial_decode': True,
        'block_decode_disclosure': 'Only declared columns are decoded; filters may decode full compressed column blocks. December2023 has metadata only, with no numeric column requested.'}
    if protocol_path is not None:
        body['protocol_pin'] = pin(protocol_path)
    body['plan_sha256'] = canonical(body)
    save(destination, body)
    return body


def verify_plan(plan):
    need(plan.get('schema') == SCHEMA and plan.get('status') == 'SEALED_BEFORE_FINANCIAL_READ'
         and plan.get('plan_sha256') == canonical({k: v for k, v in plan.items() if k != 'plan_sha256'}),
         'Changed or unsealed return/DD plan')
    need(plan['denominator_floor'] == FLOOR and plan['observed_through'] == CUTOFF
         and plan['target_family'] == 'return_over_drawdown'
         and plan['source_filter'] == {'target_family': 'executable_return', 'path_variant': 'unstopped'}
         and plan['financial_2024_plus_allowed'] is False, 'Frozen target scope changed')
    need([x['outcome_month'] for x in plan['months']] == month_range(), 'All original168 months required')
    need(sum(m['original_rows'] for m in plan['months']) == 152307, 'Frozen original denominator changed')
    need(plan['limits'] == _limits(plan['limits']['source_projection_bytes']), 'Compute limits changed')
    need(plan['source_projection_bytes'] == sum(m['projected_metadata_bytes'] + m['projected_value_bytes'] for m in plan['months'])
         <= plan['limits']['source_projection_bytes'], 'Source-read budget mismatch')
    for key in ('source_full_audit_pin', 'source_proof_pin', 'source_plan_pin'):
        check(plan[key])
    need('protocol_pin' in plan, 'Frozen protocol pin required')
    _protocol(check(plan['protocol_pin']), plan['limits'])
    need(plan['code_pins']['controller'] == pin(__file__)
         and plan['code_pins']['targets'] == pin(Path(__file__).with_name('return_dd_targets.py')),
         'Controller or target worker changed after planning')
    for item in [*plan['code_pins'].values(), *plan['source_annual_pins'].values()]:
        check(item)
    for month in plan['months']:
        need(not set(month['metadata_columns']).intersection(NUMERIC), 'Numeric source field disguised as metadata')
        need(month['value_columns'] == ([] if month['outcome_month'] == '2023-12' else list(VALUE_COLUMNS)),
             'Protected or unexpected numeric projection')
        for key in ('source_target_pin', 'source_checkpoint_pin', 'source_audit_pin'):
            check(month[key])
    return plan


def _rss_bytes():
    # macOS ru_maxrss is bytes; Linux reports KiB.
    import sys
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == 'darwin' else value * 1024


def _guard(started, limits):
    need(time.monotonic() - started <= limits['active_seconds'], 'Active time ceiling exceeded; attempt preserved')
    need(_rss_bytes() <= limits['rss_stop_bytes'], 'RAM ceiling exceeded; attempt preserved')


def _source_month(month, reserved):
    import pyarrow as pa
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(check(month['source_target_pin']))
    need(projection_bytes(pf, month['metadata_columns']) == month['projected_metadata_bytes']
         and projection_bytes(pf, month['value_columns']) == month['projected_value_bytes'], 'Source projection changed')
    columns = month['metadata_columns'] + month['value_columns']
    # Entire column blocks are charged before decoding, including filtered-out families.
    reserved['source_projection_bytes'] += month['projected_metadata_bytes'] + month['projected_value_bytes']
    table = pq.read_table(month['source_target_pin']['path'], columns=columns,
        filters=[('target_family', '=', 'executable_return'), ('path_variant', '=', 'unstopped')],
        use_threads=False)
    need(table.num_rows == 2 * month['original_rows'], 'Original long/short denominator changed')
    rows = table.select(['side', 'prediction_eligible', 'signal_date', 'entry_date', 'exit_date',
                         'exit_mark', 'label_available_at']).to_pylist()
    for side in SIDES:
        side_rows = [row for row in rows if row['side'] == side]
        need(len(side_rows) == month['original_rows']
             and sum(row['prediction_eligible'] for row in side_rows) == month['eligible_rows'],
             'Original directional admission changed')
    need(all(all(row[name] == month['source_window'][name] for name in ('signal_date', 'entry_date',
         'exit_date', 'exit_mark', 'label_available_at')) for row in rows), 'Original monthly clock changed')
    if not month['value_columns']:
        for name, dtype in [('cash_return', pa.float64()), ('maximum_drawdown', pa.float64()), ('target_class', pa.int8())]:
            table = table.append_column(name, pa.nulls(table.num_rows, type=dtype))
    # The authenticated source family defines executable target_value == cash_return.
    # Reconstruct this alias without decoding the redundant source financial column.
    return table.append_column('target_value', table['cash_return'])


def _training_counts(metadata, fold, side):
    selected = [row for row in metadata if row['side'] == side
                and fold['train_start_month'] <= row['outcome_month'] <= fold['train_end_month']]
    eligible = [row for row in selected if row['prediction_eligible']]
    targets = [row for row in eligible if row['target_available']]
    fits = [row for row in targets if row['label_available_at'] <= fold['as_of']]
    return {**fold, 'side': side, 'original_rows': len(selected),
        'prediction_eligible_rows': len(eligible), 'target_available_rows': len(targets),
        'fit_rows': len(fits), 'incomplete_or_unavailable_rows': len(eligible) - len(fits),
        'uncompleted_boundary_rows': sum(row['label_available_at'] > fold['as_of'] for row in eligible),
        'prediction_eligibility_unchanged': True, 'fit_performed': False}


def run_audit(plan_path):
    """Materialize a fresh audit once; never retry, resume, fit or replay trades."""
    import pyarrow.parquet as pq
    from .return_dd_targets import build_return_dd_targets, monthly_class_means, payoffs_from_monthly_means
    started = time.monotonic(); plan_path = Path(plan_path).resolve()
    plan = verify_plan(read(plan_path)); out = Path(plan['output_directory'])
    need(not out.exists(), 'Existing audit attempt preserved; use fresh reviewed plan')
    out.mkdir(parents=True)
    reserved = {'source_projection_bytes': 0, 'derived_read_bytes': 0}
    save(out / 'launch.json', {'schema': SCHEMA, 'status': 'RUNNING',
        'plan_pin': pin(plan_path), 'worker_pins': plan['code_pins'], 'reserved_reads': reserved.copy(),
        'estimator_fits': 0, 'trade_replays': 0})
    stats, coverage, changes, metadata, artifacts = [], [], [], [], {}
    monthly_fingerprints = {}
    try:
        for month in plan['months']:
            _guard(started, plan['limits'])
            source = _source_month(month, reserved)
            panel = build_return_dd_targets(source, observed_through=plan['observed_through'],
                source_provenance={'source_target_pin': month['source_target_pin'],
                                   'source_audit_pin': month['source_audit_pin']})
            need(panel.table.num_rows == 2 * month['original_rows'], 'New target panel dropped original rows')
            path = out / 'months' / month['outcome_month']
            path.mkdir(parents=True)
            target_path = path / 'targets.parquet'
            pq.write_table(panel.table, target_path, compression='zstd')
            diagnostic_path = path / 'diagnostics.json'
            save(diagnostic_path, {'outcome_month': month['outcome_month'],
                'diagnostics': panel.diagnostics.to_pylist(), 'provenance': panel.provenance,
                'fingerprint': panel.fingerprint})
            for file in (target_path, diagnostic_path):
                artifacts[str(file.relative_to(out))] = relative_pin(file, out)
            stats.extend(monthly_class_means(panel))
            coverage.extend(panel.diagnostics.to_pylist())
            comparison = panel.table.select(['side', 'target_available', 'target_class',
                                             'source_cash_target_class']).to_pylist()
            for side in SIDES:
                compared = [row for row in comparison if row['side'] == side
                            and row['target_available'] and row['source_cash_target_class'] is not None]
                changed = sum(row['target_class'] != row['source_cash_target_class'] for row in compared)
                changes.append({'outcome_month': month['outcome_month'], 'side': side,
                    'class_comparison_rows': len(compared), 'class_changed_rows': changed,
                    'class_unchanged_rows': len(compared) - changed})
            metadata.extend(panel.table.select(['source_security_id', 'signal_date', 'outcome_month',
                'side', 'prediction_eligible', 'target_available', 'label_available_at']).to_pylist())
            monthly_fingerprints[month['outcome_month']] = panel.fingerprint
            checkpoint_path = path / 'checkpoint.json'
            save(checkpoint_path, {'status': 'COMPLETE_MONTH', 'outcome_month': month['outcome_month'],
                'target_pin': relative_pin(target_path, out), 'diagnostics_pin': relative_pin(diagnostic_path, out),
                'source_reserved_bytes': reserved['source_projection_bytes'],
                'active_seconds': time.monotonic() - started, 'peak_rss_bytes': _rss_bytes()})
            artifacts[str(checkpoint_path.relative_to(out))] = relative_pin(checkpoint_path, out)
            _guard(started, plan['limits'])
            del source, panel
        need(len(stats) == 3360 and len(coverage) == 336 and len(metadata) == 304614,
             'Full original monthly audit dimensions changed')
        windows = [month['source_window'] for month in plan['months']]
        payoffs, readiness = [], []
        for window in windows:
            if window['outcome_month'] < '2020-01':
                continue
            for side in SIDES:
                state = payoffs_from_monthly_means(stats, windows, signal_date=window['signal_date'],
                    side=side, observed_through=plan['observed_through'],
                    source_policy_fingerprint=canonical(monthly_fingerprints))
                payoffs.append({'signal_date': window['signal_date'], 'side': side,
                    'table': state.table.to_pylist(), 'provenance': state.provenance,
                    'fingerprint': state.fingerprint})
                available = sum(row['status'] == 'available' for row in state.monthly.to_pylist())
                readiness.append({'signal_date': window['signal_date'], 'side': side,
                    **{key: state.provenance[key] for key in ('first_outcome_month', 'last_outcome_month',
                        'vector_available', 'required_calendar_months', 'required_month_class_cells')},
                    'available_month_class_cells': available, 'missing_month_class_cells': 1200 - available})
        training = [_training_counts(metadata, fold, side) for fold in plan['folds'] for side in SIDES]
        aggregate = {'monthly_class_means.json': stats, 'coverage.json': coverage,
                     'payoffs.json': payoffs, 'payoff_readiness.json': readiness,
                     'training_completion.json': training, 'class_changes.json': changes}
        for name, value in aggregate.items():
            save(out / name, value); artifacts[name] = relative_pin(out / name, out)
        _guard(started, plan['limits'])
        output_bytes = sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
        need(output_bytes <= plan['limits']['output_bytes'], 'Audit output ceiling exceeded')
        report = {'schema': SCHEMA, 'status': 'COMPLETE_PENDING_INDEPENDENT_VERIFICATION',
            'reviewed_plan_pin': pin(plan_path), 'plan_sha256': plan['plan_sha256'],
            'source_full_audit_pin': plan['source_full_audit_pin'],
            'source_proof_pin': plan['source_proof_pin'], 'artifacts': artifacts,
            'months': 168, 'original_rows': 152307, 'direction_rows': 304614,
            'monthly_class_mean_rows': len(stats), 'coverage_rows': len(coverage),
            'class_change_rows': sum(row['class_changed_rows'] for row in changes),
            'class_comparison_rows': sum(row['class_comparison_rows'] for row in changes),
            'payoff_states': len(payoffs), 'training_folds': len(training),
            'denominator_floor': FLOOR, 'observed_through': CUTOFF,
            'reserved_reads': reserved, 'output_bytes': output_bytes,
            'active_seconds': time.monotonic() - started, 'peak_rss_bytes': _rss_bytes(),
            'financial_2024_plus_read': False, 'source_prices_read': False,
            'feature_values_read': False, 'models_fitted': False, 'portfolio_backtest_run': False,
            'new_estimator_fits': 0, 'new_backtest_replays': 0,
            'source_identity_and_action_certification': False,
            'prediction_eligibility_unchanged': True, 'negative_loss_ordering_is_unresolved': True,
            'training_count_definitions': {
                'incomplete_or_unavailable_rows': 'prediction_eligible_rows minus fit_rows',
                'uncompleted_boundary_rows': 'prediction-eligible original rows with label_available_at after annual as_of; includes already unavailable labels'}}
        save(out / 'audit.json', report)
        return report
    except BaseException as exc:
        save(out / 'failure.json', {'status': 'FAILED_PRESERVED', 'error_type': type(exc).__name__,
            'error': str(exc), 'reserved_reads': reserved,
            'active_seconds': time.monotonic() - started, 'peak_rss_bytes': _rss_bytes(),
            'models_fitted': False, 'portfolio_backtest_run': False})
        raise
