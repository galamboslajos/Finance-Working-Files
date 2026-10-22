"""Independent return/DD label audit; never imports the production target worker.

The planning entry point uses JSON, pins and Parquet footers only. The verify
entry point requires that saved plan before decoding the bounded projections.
No prices, features, model objects, protected financial values or trade replay.
"""
from collections import Counter, defaultdict
import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import time

SCHEMA = 'dmv2-return-dd-independent-verification/v1'
KEYS = ('source_security_id', 'signal_date', 'side')
SIDES = ('long', 'short')
CLASSES = tuple(range(1, 11))
FLOOR = .01
CUTOFF = '2023-12-29'
SOURCE_AUDIT_SHA = '8f09fec5ed2b0404f9518ef4b6be58e50a4a1a7751e9768ed7e49fe320124547'
NUMERIC = ('cash_return', 'maximum_drawdown', 'target_value', 'target_class')


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(part)
    return digest.hexdigest()


def pin(path):
    p = Path(path).resolve()
    return {'path': str(p), 'size_bytes': p.stat().st_size, 'sha256': sha(p)}


def checked_pin(value):
    p = Path(value['path']).resolve()
    need(p.is_file() and p.stat().st_size == value['size_bytes'] and sha(p) == value['sha256'],
         'Pinned artifact changed: ' + str(p))
    return p


def read_json(path, cap=16 * 1024**2):
    p = Path(path)
    need(p.stat().st_size <= cap, 'Metadata byte ceiling')
    return json.loads(p.read_text())


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    p = Path(path)
    need(not p.exists(), 'Preserve previous receipt')
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')


def peak_rss():
    value=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform=='darwin' else value*1024


def guard(started):
    need(time.monotonic()-started<=600, 'Independent verification time ceiling')
    need(peak_rss()<=4*1024**3, 'Independent verification memory ceiling')


def validate_protocol(plan):
    protocol=read_json(checked_pin(plan['protocol_pin']))
    need(protocol['schema']=='dmv2_return_dd_target_protocol_v1'
         and protocol['status']=='FROZEN_LABEL_ENGINEERING_AND_AUDIT_ONLY'
         and protocol['target_family']=='return_over_drawdown'
         and protocol['denominator_floor']==FLOOR and protocol['observed_through']==CUTOFF
         and protocol['history']==['2010-01','2023-12']
         and protocol['path']['learning_variant']=='unstopped'
         and protocol['path']['sides']==list(SIDES)
         and protocol['ranking']['classes']==10 and protocol['ranking']['best_class']==1
         and protocol['ranking']['minimum_observed']==10
         and protocol['payoffs']['calendar_months']==120
         and protocol['payoffs']['required_month_class_cells']==1200,
         'Frozen engineering protocol changed')
    limits=protocol['limits']
    need(limits['source_projection_bytes']==plan['limits']['source_projection_bytes']
         and limits['independent_source_projection_bytes']==plan['limits']['source_projection_bytes']
         and limits['derived_verification_read_bytes']==128*1024**2
         and limits['rss_stop_bytes']==4*1024**3 and limits['active_seconds']==600
         and limits['estimator_fits']==limits['trade_replays']==0,
         'Frozen verification compute or read budget changed')


def shift(month, offset):
    need(type(month) is str and len(month) == 7 and dt.date.fromisoformat(month + '-01').strftime('%Y-%m') == month,
         'Canonical month required')
    y, m = map(int, month.split('-')); n = y * 12 + m - 1 + offset
    return f'{n // 12:04d}-{n % 12 + 1:02d}'


def day(value):
    need(type(value) is str and dt.date.fromisoformat(value).isoformat() == value, 'Canonical date required')
    return value


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def eq_numeric(actual, expected, name):
    if expected is None:
        need(actual is None, 'Unavailable ' + name + ' was filled')
    else:
        need(number(actual) and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12),
             'Independent ' + name + ' mismatch')


def same(actual, expected, name='aggregate'):
    if isinstance(expected, dict):
        need(isinstance(actual, dict) and set(actual) == set(expected), 'Independent keys differ: ' + name)
        for key in expected:
            same(actual[key], expected[key], name + '.' + str(key))
    elif isinstance(expected, list):
        need(isinstance(actual, list) and len(actual) == len(expected), 'Independent list length differs: ' + name)
        for i, value in enumerate(expected):
            same(actual[i], value, name + '[' + str(i) + ']')
    elif type(expected) is float:
        eq_numeric(actual, expected, name)
    else:
        need(type(actual) is type(expected) and actual == expected, 'Independent value differs: ' + name)


def exact_classes(rows):
    """Descending unsplit exact ties, independently using unique-value counts."""
    need(len(rows) >= 10, 'At least ten observed paths required')
    counts = Counter(r['target_value'] for r in rows)
    result = {}; greater = 0
    for value in sorted(counts, reverse=True):
        count = counts[value]
        result[value] = min(10, 1 + (10 * (2 * greater + count)) // (2 * len(rows)))
        greater += count
    return {tuple(r[k] for k in KEYS): result[r['target_value']] for r in rows}


def expected_month(source, window):
    """Calculate expected labels from two projected original source-side rows."""
    need(window['outcome_month'] < '2024-01' and window['exit_mark'] == 'open', 'Next-open target clock required')
    signal, month = day(window['signal_date']), window['outcome_month']
    need(shift(signal[:7], 1) == month and window['entry_date'][:7] == month
         and window['exit_date'][:7] == shift(month, 1), 'Next-month open clock changed')
    protected = window['exit_date'] >= '2024-01-01'
    original = {}; expected = []; seen = set()
    for row in source:
        key = tuple(row[k] for k in KEYS)
        need(key not in seen and row['side'] in SIDES and row['path_variant'] == 'unstopped'
             and row['target_family'] == 'executable_return', 'Duplicate or foreign source target')
        seen.add(key)
        need(row['signal_date'] == signal and row['outcome_month'] == month,
             'Original month changed')
        need(all(row[k] == window[k] for k in ('entry_date', 'exit_date', 'exit_mark'))
             and row['label_available_at'] == window['exit_date'], 'Source completion clock changed')
        need(type(row['prediction_eligible']) is bool and type(row['path_available']) is bool,
             'Explicit source admission flags required')
        base = (row['source_security_id'], signal)
        need(type(base[0]) is str and bool(base[0]) and base[0].strip() == base[0], 'Canonical source ID required')
        identity = {k: row[k] for k in row if k not in ('side', 'cash_return', 'maximum_drawdown',
            'target_value', 'target_class', 'target_available', 'target_status', 'path_available', 'path_status',
            'path_requested', 'nonpositive_wealth_observed', 'drawdown_status')}
        if base in original:
            need(original[base] == identity, 'Directional original cohort metadata differ')
        else:
            original[base] = identity
        input_status = ('prediction_ineligible' if not row['prediction_eligible'] else
            'scheduled_endpoint_not_observed' if window['exit_date'] > CUTOFF else
            'protected_next_open_endpoint_unavailable' if protected else
            'path_request_not_exhaustive' if row['path_requested'] is not True else
            'source_path_unavailable' if not row['path_available'] else 'available')
        out = dict(row)
        out.update(target_family='return_over_drawdown', source_target_family=row['target_family'],
            source_target_available=row['target_available'], source_target_status=row['target_status'],
            source_cash_target_class=None, source_cash_target_value=None,
            target_value=None, target_class=None, target_available=False, target_status=input_status,
            ratio_input_status=input_status, ratio_floor_hit=None, ratio_negative_return=None)
        if row['path_available']:
            need(not protected and row['prediction_eligible'] and row['path_requested'] is True,
                 'Future or ineligible observed path')
            cash, dd = row['cash_return'], row['maximum_drawdown']
            need(number(cash) and number(dd) and dd >= 0, 'Invalid source path statistics')
            need(number(row.get('target_value')) and row['target_value'] == cash,
                 'Source cash-return target differs from executable return')
            out['target_value'] = cash / max(dd, FLOOR)
            out.update(source_cash_target_class=row.get('target_class'), source_cash_target_value=cash,
                       ratio_floor_hit=dd < FLOOR, ratio_negative_return=cash < 0)
        else:
            need(all(row.get(k) is None for k in NUMERIC), 'Unavailable source path carries financial values')
        expected.append(out)
    need(len(seen) == 2 * len(original) and {r['side'] for r in expected} == set(SIDES),
         'Original directional population changed')
    summaries = []
    for side in SIDES:
        group = [r for r in expected if r['side'] == side]
        need({(r['source_security_id'], r['signal_date']) for r in group} == set(original),
             'Future availability filtered original cohort')
        eligible = [r for r in group if r['prediction_eligible']]
        valid = [r for r in eligible if r['path_available']]
        exhaustive = all(r['path_requested'] is True for r in eligible)
        status = ('non_exhaustive_original_path_requests' if not exhaustive else
                  'scheduled_endpoint_not_observed' if protected else
                  'insufficient_observed_population' if len(valid) < 10 else 'available')
        classes = exact_classes(valid) if status == 'available' else {}
        for row in valid:
            row['target_status'] = status
            if status == 'available':
                row.update(target_class=classes[tuple(row[k] for k in KEYS)], target_available=True)
        count = Counter(r['target_class'] for r in group if r['target_available'])
        tie_sizes = Counter(r['target_value'] for r in valid)
        summaries.append({'signal_date': signal, 'outcome_month': month, 'side': side,
            'path_variant': 'unstopped', 'target_family': 'return_over_drawdown',
            'original_members': len(group), 'prediction_eligible_members': len(eligible),
            'observed_source_paths': sum(r['path_available'] for r in eligible),
            'valid_ratio_values': len(valid), 'labelled_members': sum(count.values()),
            'floor_hit_count': sum(r['ratio_floor_hit'] is True for r in valid),
            'zero_drawdown_count': sum(r['maximum_drawdown'] == 0 for r in valid),
            'negative_return_count': sum(r['ratio_negative_return'] is True for r in valid),
            'negative_return_above_floor_count': sum(r['ratio_negative_return'] is True and r['maximum_drawdown'] >= FLOOR for r in valid),
            'drawdown_above_one_count': sum(r['maximum_drawdown'] > 1 for r in valid),
            'largest_exact_tie_count': max(tie_sizes.values(), default=0),
            'exact_tie_group_count': sum(n > 1 for n in tie_sizes.values()),
            'minus_one_ratio_count': sum(r['target_value'] == -1. for r in valid),
            'input_status_counts': dict(sorted(Counter(r['ratio_input_status'] for r in eligible).items())),
            'empty_classes': [c for c in CLASSES if not count[c]],
            'class_member_counts': [count[c] for c in CLASSES], 'rank_status': status})
    return expected, summaries


def verify_month(source, actual, window):
    expected, summaries = expected_month(source, window)
    index = {tuple(r[k] for k in KEYS): r for r in actual}
    need(len(index) == len(actual) == len(expected), 'Duplicate, missing or extra output rows')
    for row in expected:
        key = tuple(row[k] for k in KEYS)
        need(key in index, 'Original cohort row was dropped')
        observed = index[key]
        for field, value in row.items():
            if field in NUMERIC or field == 'source_cash_target_value':
                eq_numeric(observed.get(field), value, field)
                if field == 'target_class':
                    need(observed.get(field) is None or type(observed[field]) is int, 'Canonical class integer required')
            else:
                need(field in observed and observed[field] == value,
                     'Original metadata or independent target changed: ' + field)
                if type(value) is bool:
                    need(type(observed[field]) is bool, 'Boolean metadata type changed: ' + field)
                if field == 'source_cash_target_class':
                    need(observed[field] is None or type(observed[field]) is int, 'Canonical source class integer required')
        if 'drawdown_denominator' in observed:
            eq_numeric(observed['drawdown_denominator'],
                       max(row['maximum_drawdown'], FLOOR) if row['path_available'] else None,
                       'drawdown denominator')
        if 'floor_active' in observed:
            need(observed['floor_active'] == (row['maximum_drawdown'] < FLOOR if row['path_available'] else None),
                 'Floor-active flag changed')
    return expected, summaries


def parquet_projection(path, columns):
    import pyarrow.parquet as pq
    file = pq.ParquetFile(path); need(len(columns) == len(set(columns)), 'Duplicate projected columns')
    need(set(columns) <= set(file.schema_arrow.names), 'Missing projected columns')
    size = 0
    for group in range(file.metadata.num_row_groups):
        meta = file.metadata.row_group(group)
        for i in range(meta.num_columns):
            col = meta.column(i)
            if col.path_in_schema.split('.')[0] in columns:
                size += col.total_compressed_size
    return {'rows': file.metadata.num_rows, 'compressed_bytes': size, 'columns': list(columns)}


def plan_verification(plan_path, run_directory, output):
    """Create a separate metadata-first verification reservation; no value reads."""
    plan_path, run, output = map(lambda p: Path(p).resolve(), (plan_path, run_directory, output))
    p = read_json(plan_path)
    need(p['schema'] == 'dmv2_return_dd_audit_v1' and p['denominator_floor'] == FLOOR
         and p['target_family'] == 'return_over_drawdown' and p['observed_through']==CUTOFF,
         'Unfrozen return/DD policy')
    need(p['plan_sha256'] == 'sha256:' + canonical({k:value for k,value in p.items() if k!='plan_sha256'})
         and p['status'] == 'SEALED_BEFORE_FINANCIAL_READ', 'Unsealed reviewed plan')
    need(p['source_full_audit_pin']['sha256']==SOURCE_AUDIT_SHA
         and set(p['source_annual_pins'])=={str(y) for y in range(2010,2024)}, 'Approved original source lineage required')
    for item in p['code_pins'].values():checked_pin(item)
    validate_protocol(p)
    audit_path = run / 'audit.json'; audit = read_json(audit_path)
    need(audit['status']=='COMPLETE_PENDING_INDEPENDENT_VERIFICATION'
         and audit['reviewed_plan_pin']==pin(plan_path) and audit['plan_sha256']==p['plan_sha256']
         and audit['direction_rows']==304614 and audit['months']==168
         and audit['models_fitted'] is False and audit['portfolio_backtest_run'] is False
         and audit['financial_2024_plus_read'] is False, 'Producer audit scope or completion changed')
    for name,item in audit['artifacts'].items():
        path=(run/item['path']).resolve()
        need(path.is_relative_to(run) and item['path']==name
             and path.stat().st_size==item['size_bytes'] and sha(path)==item['sha256'],
             'Producer artifact pin changed: '+name)
    months = []; total_source = total_derived = 0
    for month in p['months']:
        target = run / 'months' / month['outcome_month'] / 'targets.parquet'
        source = checked_pin(month['source_target_pin'])
        import pyarrow.parquet as pq
        expected_metadata=[name for name in pq.ParquetFile(source).schema_arrow.names
                           if name not in (*NUMERIC,'utility')]
        need(month['metadata_columns']==expected_metadata
             and month['value_columns']==([] if month['outcome_month']=='2023-12'
                                         else ['cash_return','maximum_drawdown','target_class']),
             'Unauthorized source numerical projection')
        cols = list(dict.fromkeys(month['metadata_columns'] + month['value_columns']))
        source_projection = parquet_projection(source, month['metadata_columns'] if month['source_window']['exit_date'] >= '2024-01-01' else cols)
        output_projection = parquet_projection(target, pq.ParquetFile(target).schema_arrow.names)
        need(source_projection['rows'] == month['source_rows'] and output_projection['rows'] == month['original_rows'] * 2,
             'Planned original output denominator changed')
        total_source += source_projection['compressed_bytes']; total_derived += output_projection['compressed_bytes']
        diagnostic = run / 'months' / month['outcome_month'] / 'diagnostics.json'
        need(diagnostic.stat().st_size <= 256 * 1024, 'Monthly diagnostic byte ceiling')
        total_derived += diagnostic.stat().st_size
        months.append({'outcome_month': month['outcome_month'], 'source_projection': source_projection,
            'output_pin': pin(target), 'output_projection': output_projection, 'diagnostics_pin': pin(diagnostic)})
    need(len(months) == 168 and [m['outcome_month'] for m in months] == [shift('2010-01', i) for i in range(168)],
         'Complete ordered 2010–2023 history required')
    artifacts = {}
    for name in ('coverage.json', 'class_changes.json', 'payoff_readiness.json', 'training_completion.json', 'monthly_class_means.json', 'payoffs.json', 'audit.json'):
        path = run / name; need(path.stat().st_size <= 16 * 1024**2, 'Aggregate metadata ceiling')
        artifacts[name] = pin(path); total_derived += path.stat().st_size
    need(p['limits']['source_projection_bytes'] in (64*1024**2,128*1024**2)
         and total_source == p['source_projection_bytes'] <= p['limits']['source_projection_bytes']
         and total_derived <= 128 * 1024**2,
         'Independent read reservation ceiling')
    v = {'schema': SCHEMA, 'mode': 'metadata_only_read_plan', 'plan_pin': pin(plan_path),
        'source_full_audit_pin': p['source_full_audit_pin'], 'source_proof_pin': p['source_proof_pin'],
        'source_plan_pin': p['source_plan_pin'], 'months': months, 'artifacts': artifacts,
        'source_projection_bytes': total_source, 'derived_read_bytes': total_derived,
        'worker_pin': pin(__file__), 'run_directory': str(run), 'floor': FLOOR,
        'models_fitted': 0, 'portfolio_backtest_run': False, 'financial_2024_plus_read': False}
    v['verification_plan_sha256']='sha256:'+canonical(v)
    write_json(output, v)
    return v


def verify(plan_path, run_directory, verification_plan_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    started = time.monotonic()
    plan_path, run, vp = map(lambda p: Path(p).resolve(), (plan_path, run_directory, verification_plan_path))
    v = read_json(vp); need(v['schema'] == SCHEMA and v['worker_pin'] == pin(__file__)
        and v['verification_plan_sha256']=='sha256:'+canonical({k:value for k,value in v.items() if k!='verification_plan_sha256'}),
        'Verification worker or sealed plan changed')
    need(checked_pin(v['plan_pin']) == plan_path and v['run_directory'] == str(run), 'Verification scope changed')
    p = read_json(plan_path)
    need(Path(p['output_directory']).resolve()==run and p['denominator_floor']==FLOOR
         and p['plan_sha256']=='sha256:'+canonical({k:value for k,value in p.items() if k!='plan_sha256'}),
         'Reviewed producer plan changed')
    for item in p['code_pins'].values():checked_pin(item)
    validate_protocol(p)
    for key in ('source_full_audit_pin', 'source_proof_pin', 'source_plan_pin'):
        checked_pin(v[key]); need(v[key] == p[key], 'Original source proof changed')
    proof = read_json(checked_pin(v['source_proof_pin']))
    need(proof['status'] == 'PASS_FULL_FORMULAS_RANKS_CLOCK_AND_ORIGINAL_DENOMINATORS'
         and proof['months_checked'] == 168 and proof['audit_sha256'] == v['source_full_audit_pin']['sha256'],
         'Original full path audit was not independently verified')
    for artifact in v['artifacts'].values():
        checked_pin(artifact)
    original_plan = read_json(checked_pin(v['source_plan_pin']))
    source_audit=read_json(checked_pin(v['source_full_audit_pin']))
    need(original_plan['plan_sha256']=='sha256:'+canonical({k:value for k,value in original_plan.items() if k!='plan_sha256'})
         and source_audit['plan_sha256']==proof['plan_sha256']==original_plan['plan_sha256'],
         'Original full audit reviewed-plan lineage changed')
    original_windows = {w['outcome_month']: w for annual in original_plan['annuals'] for w in annual['windows']}
    source_annuals={}
    for year,item in p['source_annual_pins'].items():
        path=checked_pin(item);expected=source_audit['year_audit_pins'][year]
        need(item['sha256']==expected['sha256'] and item['size_bytes']==expected['size_bytes'], 'Original annual audit lineage changed')
        source_annuals[year]=read_json(path)
    all_expected = []; all_summaries = []; output_rows = source_bytes = 0
    for item, expected_plan in zip(p['months'], v['months'], strict=True):
        guard(started)
        need(item['outcome_month'] == expected_plan['outcome_month'], 'Verification month changed')
        for key in ('source_checkpoint_pin', 'source_audit_pin', 'source_target_pin'):
            checked_pin(item[key])
        annual=source_annuals[item['outcome_month'][:4]]
        cp_saved=annual['artifacts']['months/'+item['outcome_month']+'/checkpoint.json']
        cp=read_json(item['source_checkpoint_pin']['path']);source_month_audit=read_json(item['source_audit_pin']['path'])
        need(item['source_checkpoint_pin']['sha256']==cp_saved['sha256']
             and item['source_checkpoint_pin']['size_bytes']==cp_saved['size_bytes']
             and cp['audit_sha256']==item['source_audit_pin']['sha256']
             and cp['plan_sha256']==original_plan['plan_sha256'], 'Original checkpoint chain changed')
        saved_target=source_month_audit['artifacts']['targets.parquet']
        need(item['source_target_pin']['sha256']==saved_target['sha256']
             and item['source_target_pin']['size_bytes']==saved_target['size_bytes'], 'Original target receipt chain changed')
        for key in ('signal_date', 'outcome_month', 'entry_date', 'exit_date', 'exit_mark'):
            need(item['source_window'][key] == original_windows[item['outcome_month']][key],
                 'Planned completion clock changed original session calendar')
        source_path = Path(item['source_target_pin']['path'])
        columns = expected_plan['source_projection']['columns']
        need(parquet_projection(source_path, columns) == expected_plan['source_projection'], 'Source footer projection changed')
        table = pq.read_table(source_path, columns=columns)
        # Filter on metadata before projecting any decoded numerical values.
        selected = [i for i, row in enumerate(table.select(['path_variant', 'target_family']).to_pylist())
                    if row['path_variant'] == 'unstopped' and row['target_family'] == 'executable_return']
        rows = table.take(pa.array(selected, type=pa.int64())).to_pylist()
        for row in rows:
            row['target_value'] = row.get('cash_return')
        if item['source_window']['exit_date'] >= '2024-01-01':
            need(not any(k in columns for k in NUMERIC), 'Protected numerical source columns decoded')
            for row in rows:
                row.update({k: None for k in NUMERIC})
        actual_path = checked_pin(expected_plan['output_pin'])
        need(parquet_projection(actual_path, expected_plan['output_projection']['columns']) == expected_plan['output_projection'],
             'Output projection changed')
        actual = pq.read_table(actual_path, columns=expected_plan['output_projection']['columns']).to_pylist()
        expected, summaries = verify_month(rows, actual, item['source_window'])
        diagnostic = read_json(checked_pin(expected_plan['diagnostics_pin']))
        need(diagnostic['outcome_month'] == item['outcome_month'], 'Monthly diagnostic scope changed')
        same(diagnostic['diagnostics'], summaries, 'monthly diagnostics')
        provenance = diagnostic['provenance']
        need(diagnostic['fingerprint'] == 'sha256:' + canonical(provenance)
             and provenance['denominator_floor'] == FLOOR and provenance['endpoint_policy'] == 'next_open'
             and provenance['path_variant'] == 'unstopped' and provenance['target_cost_bps'] == 0
             and provenance['negative_returns'] == 'literal_signed_ratio_no_piecewise_transform_no_clipping'
             and provenance['class_one_is_highest'] is True
             and provenance['estimator_fits'] == 0 and provenance['trade_replays'] == 0,
             'Frozen label provenance changed')
        need(len(rows) == item['original_rows'] * 2
             and sum(r['prediction_eligible'] for r in rows) == item['eligible_rows'] * 2, 'Original planned denominators changed')
        slim_columns = (*KEYS, 'outcome_month', 'entry_date', 'exit_date', 'label_available_at',
                        'prediction_eligible', 'target_value', 'target_class', 'target_available', 'cash_return', 'source_cash_target_class')
        all_expected.extend({k: row[k] for k in slim_columns} for row in expected)
        all_summaries.extend(summaries); output_rows += len(actual)
        source_bytes += expected_plan['source_projection']['compressed_bytes']
        guard(started)
    aggregate_checks = verify_aggregates(p, run, all_expected, all_summaries)
    guard(started)
    receipt = {'schema': SCHEMA, 'status': 'PASS', 'plan_pin': pin(plan_path),
        'verification_plan_pin': pin(vp), 'worker_pin': pin(__file__), 'months_checked': 168,
        'directional_rows_checked': output_rows, 'source_projection_bytes': source_bytes,
        'derived_read_bytes': v['derived_read_bytes'], 'formula': 'R/max(DD,0.01)',
        'class_order': '1_best_10_worst_exact_ties_unsplit_midrank',
        'models_fitted': 0, 'portfolio_backtest_run': False, 'source_prices_read': False,
        'feature_values_read': False, 'financial_2024_plus_read': False,
        'source_identity_and_action_certification': False, 'aggregate_checks': aggregate_checks,
        'wall_seconds': time.monotonic() - started,'peak_rss_bytes':peak_rss()}
    write_json(vp.parent / 'independent_verification.json', receipt)
    return receipt


def verify_aggregates(plan, run, rows, summaries):
    """Reconstruct all payoff grids and annual completion counts independently."""
    windows = {m['outcome_month']: m['source_window'] for m in plan['months']}
    same(read_json(run / 'coverage.json'), summaries, 'full diagnostic coverage')
    groups = defaultdict(list)
    for row in rows:
        groups[(row['outcome_month'], row['side'])].append(row)
    class_changes=[]
    for month in sorted(windows):
        for side in SIDES:
            compared=[r for r in groups[(month,side)] if r['target_available'] and r['source_cash_target_class'] is not None]
            changed=sum(r['target_class']!=r['source_cash_target_class'] for r in compared)
            class_changes.append({'outcome_month':month,'side':side,'class_comparison_rows':len(compared),
                'class_changed_rows':changed,'class_unchanged_rows':len(compared)-changed})
    same(read_json(run/'class_changes.json'),class_changes,'class assignment changes')
    statistics = []
    for month, window in sorted(windows.items()):
        for side in SIDES:
            original = groups[(month, side)]
            for class_id in CLASSES:
                members = [r for r in original if r['target_available'] and r['prediction_eligible'] and r['target_class'] == class_id]
                count = len(members)
                statistics.append({'signal_date': window['signal_date'], 'outcome_month': month,
                    'side': side, 'class_id': class_id, 'label_available_at': window['label_available_at'],
                    'original_members': len(original), 'prediction_eligible_members': sum(r['prediction_eligible'] for r in original),
                    'member_count': count,
                    'mean_ratio': math.fsum(r['target_value'] / count for r in members) if count else None,
                    'mean_cash_return': math.fsum(r['cash_return'] / count for r in members) if count else None,
                    'status': 'available' if count else ('scheduled_endpoint_not_observed'
                        if window['label_available_at'] > CUTOFF else 'empty_realized_class')})
    same(read_json(run / 'monthly_class_means.json'), statistics, 'monthly class means')
    lookup = {(r['outcome_month'],r['side'],r['class_id']): r for r in statistics}
    payloads = read_json(run / 'payoffs.json')
    need(type(payloads) is list and len(payloads) == 96, 'All 48 original forecast months need two payoff states')
    state_index = {(r['signal_date'],r['side']):r for r in payloads}
    need(len(state_index)==96, 'Duplicate or omitted payoff state')
    readiness = []
    for outcome in sorted(m for m in windows if '2020-01' <= m <= '2023-12'):
        signal = windows[outcome]['signal_date']; last = shift(signal[:7],-1)
        months = [shift(last, offset) for offset in range(-119,1)]
        for side in SIDES:
            expected_monthly = []
            for month in months:
                window = windows.get(month)
                for class_id in CLASSES:
                    cell = lookup.get((month,side,class_id))
                    status = ('missing_calendar_month' if window is None else
                        'scheduled_endpoint_not_observed' if window['label_available_at'] > signal else
                        'missing_month_class_cell' if cell is None else cell['status'])
                    available = status == 'available'
                    expected_monthly.append({'outcome_month':month,'class_id':class_id,
                        'original_member_count':cell['original_members'] if cell else 0,
                        'prediction_eligible_count':cell['prediction_eligible_members'] if cell else 0,
                        'member_count':cell['member_count'] if available else 0,
                        'class_mean_ratio':cell['mean_ratio'] if available else None,
                        'class_mean_cash_return':cell['mean_cash_return'] if available else None,
                        'status':status})
            available_cells = sum(r['status']=='available' for r in expected_monthly)
            available = available_cells == 1200
            expected_table = [{'class_id':c,
                'class_mean_ratio':math.fsum(r['class_mean_ratio']/120 for r in expected_monthly if r['class_id']==c) if available else None,
                'class_mean_cash_return':math.fsum(r['class_mean_cash_return']/120 for r in expected_monthly if r['class_id']==c) if available else None,
                'status':'available' if available else 'withheld_incomplete_120_month_class_grid'} for c in CLASSES]
            need((signal,side) in state_index, 'Original forecast payoff state was dropped')
            state = state_index[(signal,side)]; same(state['table'],expected_table,'causal payoff vector')
            provenance = state['provenance']
            expected_provenance = {'target_family':'return_over_drawdown','side':side,'path_variant':'unstopped',
                'signal_date':signal,'label_observed_through':signal,'first_outcome_month':months[0],
                'last_outcome_month':last,'required_calendar_months':120,'required_month_class_cells':1200,
                'vector_available':available,'time_weighting':'equal_calendar_month_means',
                'source_monthly_statistics_fingerprint':'sha256:'+canonical([r for r in statistics
                    if r['side']==side and r['outcome_month'] in set(months) and r['label_available_at']<=signal]),
                'source_monthly_metadata_fingerprint':'sha256:'+canonical([{k:value for k,value in r.items()
                    if k not in ('mean_ratio','mean_cash_return')} for r in statistics]),
                'statistics_hash_scope':'requested_side_required_120_months_actual_completion_le_signal',
                'inspected_monthly_class_cells':sum(r['side']==side and r['outcome_month'] in set(months)
                    and r['label_available_at']<=signal for r in statistics),
                'ratio_score_semantics':'expected_signed_return_over_drawdown_not_cash_return',
                'cash_score_semantics':'expected_executable_cash_return_conditional_on_ratio_classes'}
            for key,value in expected_provenance.items():same(provenance[key],value,'payoff provenance.'+key)
            need(state['fingerprint']=='sha256:'+canonical({'provenance':provenance,'table':state['table'],'monthly':expected_monthly}),
                 'Payoff fingerprint does not bind independently reconstructed causal grid')
            readiness.append({'signal_date':signal,'side':side,'first_outcome_month':months[0],
                'last_outcome_month':last,'vector_available':available,'required_calendar_months':120,
                'required_month_class_cells':1200,'available_month_class_cells':available_cells,
                'missing_month_class_cells':1200-available_cells})
    same(read_json(run / 'payoff_readiness.json'), readiness, 'payoff readiness')
    training = []
    for year in range(2020,2024):
        first,last=f'{year-10}-01',f'{year-1}-12'; cutoff=windows[f'{year}-01']['signal_date']
        for side in SIDES:
            cohort = [r for r in rows if first<=r['outcome_month']<=last and r['side']==side]
            eligible = [r for r in cohort if r['prediction_eligible']]
            available = [r for r in eligible if r['target_available']]
            fit = [r for r in available if r['label_available_at']<=cutoff]
            training.append({'year':year,'side':side,'train_start_month':first,'train_end_month':last,
                'as_of':cutoff,'original_rows':len(cohort),'prediction_eligible_rows':len(eligible),
                'target_available_rows':len(available),'fit_rows':len(fit),
                'incomplete_or_unavailable_rows':len(eligible)-len(fit),
                'uncompleted_boundary_rows':sum(r['label_available_at']>cutoff for r in eligible),
                'prediction_eligibility_unchanged':True,'fit_performed':False})
    same(read_json(run/'training_completion.json'),training,'ten-year completion purge')
    return {'monthly_diagnostics':len(summaries),'monthly_class_mean_cells':len(statistics),
        'payoff_states':len(readiness),'training_completion_rows':len(training),
        'complete_payoff_vectors':sum(r['vector_available'] for r in readiness),'class_change_rows':len(class_changes)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('plan', 'verify'))
    parser.add_argument('--plan', required=True); parser.add_argument('--run', required=True)
    parser.add_argument('--verification-plan', required=True)
    args = parser.parse_args()
    result = (plan_verification(args.plan, args.run, args.verification_plan) if args.mode == 'plan'
              else verify(args.plan, args.run, args.verification_plan))
    print(json.dumps(result, sort_keys=True))
