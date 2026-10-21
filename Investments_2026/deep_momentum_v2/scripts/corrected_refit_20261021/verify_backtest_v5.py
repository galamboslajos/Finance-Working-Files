#!/usr/bin/env python3
"""Independent verification of 15 new selections and all 120 saved daily paths.

No estimator, preprocessor, feature panel or target panel is opened.  Fresh
probabilities and the existing corrected RET vectors are checked independently;
the preserved independent cash/units equations check saved accounting ledgers.
There is no production selection, overlay or accounting expected-answer call.
Planning JSON is reserved before parsing financial receipts, then a second
exact reservation covers every numerical projection and repeated JSON parse.
"""
from pathlib import Path
import argparse
import collections
import copy
from datetime import date, datetime
import hashlib
import importlib.util
import itertools
import json
import os
import sys
import time

from control import ROOT, check, pin, read, need, require_reservation, save_new
import backtest_verification_math_v2 as independent

SCHEMA = 'dmv2-independent-corrected-backtest-verification/v1'
BASE = ROOT / 'run/backtest'
MAX_READ = 2 * 1024**3
MAX_JSON = 16 * 1024**2
HELPERS = {
    'cash_units': ('verify_walkforward_backtest_v2.py', '16c3651289258bed24a83d500f6b8589fac3bcb4d358e516fe981acec527a794'),
    'ret_selection': ('verify_ret_backtest.py', 'cd315f90848349b3f64045923e5410ee1adea92bec0eaafcd72f974f03c7c89e'),
    'quote_overlay': ('verify_ret_repair_backtest.py', 'af1c2cc26f4fb6433f247808a5eaff40ebb8a49299ac8bc646ae37f72df70f89'),
}
NUMERIC_ENV = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
               'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')
DATE_FIELDS = ('date', 'trading_date', 'signal_date', 'held_signal_date', 'execution_date',
    'scheduled_exit_date', 'outcome_start', 'outcome_end', 'period_start_date', 'entry_date',
    'exit_date', 'effective_date')
SAFE_FIELDS = (*DATE_FIELDS, 'outcome_month', 'year', 'source_security_id', 'intrinio_security_id',
    'side', 'mark', 'arm', 'basket', 'variant', 'cost_bps', 'status', 'reason', 'role', 'event_id',
    'frequency', 'intraperiod', 'price_quality_status', 'raw_ohlc_bounds_valid',
    'adjusted_ohlc_bounds_valid', 'complete_proxy_path', 'prediction_eligible',
    'paper_prediction_eligible', 'identity_eligible', 'prediction_status', 'trade_eligible',
    'execution_restricted', 'selection_status', 'selected', 'ret_status')
FORECAST_META = ('source_security_id', 'signal_date', 'intrinio_security_id', 'outcome_month',
    'outcome_start', 'outcome_end', 'identity_eligible', 'paper_prediction_eligible',
    'prediction_eligible', 'prediction_status', 'fit_year', 'training_outcome_start',
    'training_outcome_end', 'training_label_observed_through')
QUOTE_META = ('trading_date', 'intrinio_security_id', 'frequency', 'intraperiod',
    'price_quality_status', 'raw_ohlc_bounds_valid', 'adjusted_ohlc_bounds_valid')
QUOTE_VALUES = ('trading_date', 'intrinio_security_id', 'adj_open', 'adj_high', 'adj_low', 'adj_close',
    'frequency', 'intraperiod', 'price_quality_status', 'adjusted_ohlc_bounds_valid',
    'raw_ohlc_bounds_valid', 'factor', 'split_ratio', 'dividend', 'open', 'high', 'low', 'close')
PAYOFF_META = ('class_id', 'status')
MONTHLY_META = ('outcome_month', 'outcome_end', 'class_id', 'original_member_count',
    'prediction_eligible_count', 'member_count', 'status')
LEDGERS = ('daily', 'trades', 'holdings', 'failures', 'stops', 'action_events',
           'market_trades', 'orders', 'identity_handovers')


def clean_pin(item):
    return {k: item[k] for k in ('path', 'sha256', 'size_bytes')}


def collect_pins(value):
    result = {}
    def visit(v):
        if isinstance(v, dict):
            if {'path', 'sha256', 'size_bytes'} <= set(v):
                p = clean_pin(v); old = result.setdefault(p['path'], p)
                need(old == p, 'Conflicting immutable pin for one source path')
            else:
                for item in v.values(): visit(item)
        elif isinstance(v, list):
            for item in v: visit(item)
    visit(value)
    return list(result.values())


def exact_paths(paths):
    result = {}
    for name, item in paths.items():
        p = clean_pin(item); need(p['size_bytes'] <= MAX_JSON, 'Financial JSON exceeds bounded size')
        check(p)
        if p['path'] in result:
            need(result[p['path']] == p, 'Duplicate JSON path has different byte authority')
        result[p['path']] = p
    return list(result.values())


def plan_json(root, source_path, protocol_path, output):
    """Only an upstream paths/projections source plan is parsed at this step."""
    root, source_path, protocol_path, output = map(lambda p: Path(p).resolve(),
        (root, source_path, protocol_path, output))
    need(root.name == 'deep_momentum_v2' and source_path.is_relative_to(ROOT)
         and protocol_path.is_relative_to(ROOT) and output.is_relative_to(ROOT / 'plans'),
         'Exact source/workspace planning namespaces required')
    source_pin = pin(source_path); source = read(source_path)
    need(source['schema'] == 'dmv2-corrected-forecasts-backtest-source-plan/v1'
         and source['scenario_count'] == 120, 'New forecast source-plan identity required')
    named = {'source': source_pin, 'protocol': pin(protocol_path)}
    for name in ('catalog_pin', 'input_verification_pin', 'terminal_guard_pin', 'model_protocol_pin',
        'final_model_verification_pin', 'base_configuration_pin', 'execution_phase_pin',
        'execution_manifest_pin', 'execution_verification_pin', 'execution_backtest_audit_pin'):
        named[name] = source[name]
    named.update({'model_' + str(y): item for y, item in source['model_verification_pins'].items()})
    named.update({'prediction_' + str(f['year']) + '_' + f['arm']: f['prediction_receipt_pin']
                  for f in source['forecasts']})
    named.update({'state_' + p['signal_date']: p['state_pin'] for p in source['payoffs']})
    for phase in ('selection', 'preflight', 'replay'):
        named['audit_' + phase] = pin(BASE / phase / 'audit.json')
    unique = exact_paths(named)
    helpers = {}
    for key, (name, expected) in HELPERS.items():
        item = pin(root / 'scripts' / name)
        need(item['sha256'] == expected, 'Established independent equations changed: ' + name)
        helpers[key] = item
    return save_new(output, {'schema': 'dmv2-corrected-backtest-verification-JSON-plan/v1',
        'status': 'SEALED_BEFORE_FINANCIAL_JSON_PARSE', 'project_root': str(root),
        'source_plan_pin': source_pin, 'backtest_protocol_pin': pin(protocol_path),
        'worker_pin': pin(__file__), 'math_pin': pin(independent.__file__),
        'control_pin': pin(ROOT / 'control.py'), 'independent_helper_pins': helpers,
        'named_JSON_pins': named, 'unique_JSON_pins': unique,
        'projected_reads': {'compressed_source': sum(p['size_bytes'] for p in unique), 'derived_matrix': 0}})


def parsed_json(named):
    cache = {}; values = {}
    for name, item in named.items():
        key = item['path']
        if key not in cache: cache[key] = read(check(item))
        values[name] = cache[key]
    return values


def validate_json_budget(z, *, total=None):
    """A sealed plan must still describe the exact distinct JSON dispatches."""
    expected = {p['path']: p for p in exact_paths(z['named_JSON_pins'])}
    actual = z['unique_JSON_pins']
    need(len(actual) == len(expected) and {p['path']: clean_pin(p) for p in actual} == expected,
         'Planning JSON distinct pin set differs from named reads')
    amount = sum(p['size_bytes'] for p in expected.values())
    reads = z['projected_reads']
    need(set(reads) == {'compressed_source', 'derived_matrix'} and type(reads['compressed_source']) is int
         and type(reads['derived_matrix']) is int and reads['derived_matrix'] == 0
         and reads['compressed_source'] == (amount if total is None else total),
         'Planning JSON/read projection exact byte sum differs')
    return amount


def projected_cost(v):
    if isinstance(v, dict):
        if 'projected_compressed_bytes' in v:
            value = v['projected_compressed_bytes']
            need(type(value) is int and value >= 0, 'Invalid independent projection byte count')
            return value
        return sum(projected_cost(x) for x in v.values())
    return 0


def projection(item, columns, maximum_rows):
    import pyarrow.parquet as pq
    f = pq.ParquetFile(check(item)); names = f.schema_arrow.names
    need(len(names) == len(set(names)) and set(columns) <= set(names)
         and f.metadata.num_rows <= maximum_rows, 'Independent schema/row envelope changed')
    groups = []
    for i in range(f.metadata.num_row_groups):
        rg = f.metadata.row_group(i); sizes = collections.defaultdict(int)
        for j in range(rg.num_columns):
            c = rg.column(j)
            need(c.total_compressed_size >= 0, 'Negative Parquet projection size')
            sizes[c.path_in_schema.split('.')[0]] += c.total_compressed_size
        need(set(columns) <= set(sizes) or not columns, 'Projected root missing from footer leaves')
        groups.append(sum(sizes[c] for c in columns))
    return {'pin': clean_pin(item), 'columns': list(columns), 'rows': f.metadata.num_rows,
        'schema_fields': {c: str(f.schema_arrow.field(c).type) for c in columns},
        'maximum_rows': maximum_rows, 'row_group_compressed_bytes': groups,
        'projected_compressed_bytes': sum(groups)}


def table_pair(item, maximum_rows):
    import pyarrow.parquet as pq
    names = pq.ParquetFile(check(item)).schema_arrow.names
    meta = [c for c in names if c in SAFE_FIELDS]
    need(meta or not pq.ParquetFile(item['path']).metadata.num_rows, 'Nonempty saved table has no safe metadata')
    return {'metadata': projection(item, meta, maximum_rows), 'values': projection(item, names, maximum_rows)}


def validate_authority(z, data):
    """Financial JSON parsing has already been reserved before this is called."""
    source, protocol = data['source'], data['protocol']
    need(protocol['schema'] == 'dmv2-corrected-forecasts-backtest-protocol/v1'
         and protocol['status'] == 'SEALED_BEFORE_FINANCIAL_READ'
         and protocol['source_plan_pin'] == z['source_plan_pin'] and protocol['scenarios'] == 120
         and protocol['allow_withheld_paths'] is True and protocol['financial_2024_plus_authorized'] is False,
         'Frozen 120-case protocol is absent or changed')
    model = data['final_model_verification_pin']; mp = data['model_protocol_pin']
    need(model['schema'] == 'dmv2-independent-corrected-model-final-verification/v1'
         and model['status'] == 'PASS' and model['verified_estimator_fits'] == 20
         and model['verified_training_only_preprocessing_states'] == 8
         and model['outcome_months'] == 48 and model['same_cohort_all_arms'] is True
         and model['all_stitched_forecasts_exact_to_independent_annual_proofs'] is True
         and model['catalog_pin'] == source['catalog_pin']
         and model['protocol_pin'] == source['model_protocol_pin']
         and model['worker_pin'] == mp['verification_worker_pin']
         and model['annual_verification_pins'] == source['model_verification_pins'],
         'Final 20-fit/eight-state independently verified model source required')
    need(data['input_verification_pin']['status'] == 'PASS'
         and data['terminal_guard_pin']['status'] == 'PASS'
         and mp['input_verification_pin'] == source['input_verification_pin']
         and mp['terminal_guard_pin'] == source['terminal_guard_pin'], 'Input and terminal proof changed')
    catalog = data['catalog_pin']
    need([a['year'] for a in catalog['annuals']] == list(range(2010, 2024))
         and len(catalog['RET_states']) == 48
         and catalog['independent_verification'] == source['input_verification_pin'],
         'Fourteen corrected annuals/48RET states detached from model inputs')
    need({(f['year'], f['arm']) for f in source['forecasts']}
         == set(itertools.product(independent.YEARS, independent.ARMS)) and len(source['forecasts']) == 20,
         'Exactly twenty newly verified forecasts required')
    for f in source['forecasts']:
        q = data['model_' + str(f['year'])]
        receipt = data['prediction_' + str(f['year']) + '_' + f['arm']]
        need(q['status'] == 'PASS' and q['year'] == f['year']
             and q['catalog_pin'] == source['catalog_pin'] and q['protocol_pin'] == source['model_protocol_pin']
             and q['worker_pin'] == mp['verification_worker_pin']
             and q['arms'][f['arm']]['prediction_receipt_pin'] == f['prediction_receipt_pin']
             and receipt['status'] == 'COMPLETE' and receipt['year'] == f['year'] and receipt['arm'] == f['arm']
             and clean_pin(receipt['artifacts']['predictions']) == clean_pin(f['predictions_pin']),
             'New forecast artifact is not the independently verified annual prediction')
    annuals = {a['year']: a for a in catalog['annuals']}
    need({a['year'] for a in source['daily']} == set(independent.YEARS) and len(source['daily']) == 4,
         'Four source quote years required')
    for a in source['daily']:
        need(clean_pin(a['values']['pin']) == clean_pin(annuals[a['year']]['artifacts']['daily_prices_raw']),
             'Daily execution source differs from corrected input catalog')
    states = {p['signal_date']: p['artifacts'] for p in catalog['RET_states']}
    need(len(states) == 48 and len(source['payoffs']) == 48, 'Exactly 48 unique catalog RET states required')
    for p in source['payoffs']:
        a = states[p['signal_date']]
        need(clean_pin(p['state_pin']) == clean_pin(a['payoff_state.json'])
             and clean_pin(p['values']['pin']) == clean_pin(a['payoffs.parquet'])
             and clean_pin(p['monthly']['pin']) == clean_pin(a['monthly.parquet']),
             'RET vector differs from the independently verified corrected catalog')
    old, phase, manifest = (data[k] for k in ('execution_verification_pin', 'execution_phase_pin', 'execution_manifest_pin'))
    need(old['schema'] == 'dmv2-ret-repair-independent-verification/v1' and old['status'] == 'PASS'
         and old['scenarios'] == 40 and old['sessions'] == 1006 and old['outcome_months'] == 48
         and old['script_sha256'] == HELPERS['quote_overlay'][1]
         and old['source_pins_unchanged'] is True
         and old['prior_retirements_and_PENN_exception_preserved'] is True
         and old['backtest_phase_identity_sha256'] == independent.canonical(phase)
         and old['backtest_audit_sha256'] == source['execution_backtest_audit_pin']['sha256']
         and old['repair_manifest_pin']['sha256'] == source['execution_manifest_pin']['sha256'],
         'Established independent execution-repair/unit evidence PASS detached')
    need(len(phase['event_registry']['events']) == 18 and len(phase['historical_retirements']) == 11
         and len(manifest['quote_amendments']) == 40
         and phase['repair_manifest_pin']['sha256'] == source['execution_manifest_pin']['sha256'],
         'Frozen 18+11 actions /40execution-only amendments changed')
    handovers = phase['identity_handover_registry']['handovers']
    need(len(handovers) == 1 and phase['identity_handover_registry']['source_identity_promoted'] is False
         and handovers[0]['verified'] is False and handovers[0]['provisionally_admitted'] is True,
         'Exact provisional PENN execution waiver changed')
    authority = source['execution_authority']
    for key in ('event_registry', 'historical_retirements', 'identity_handover_registry', 'identity_handover_approval'):
        need(authority[key + '_sha256'] == independent.canonical(phase[key]), 'Execution authority hash changed: ' + key)
    cfg = data['base_configuration_pin']; bt = cfg['backtest']
    need(bt['cost_bps'] == list(independent.FEES) and bt['variants'] == {'unstopped': None, 'asset_stop_10pct': .1}
         and bt['top_k_per_side'] == 10 and bt['long_weight'] == .5 and bt['short_weight'] == -.5
         and set(cfg['execution_restrictions']['native_ids']) == independent.EXCLUDED
         and cfg['accounting']['continuous_2020_2023_ledger'] is True
         and cfg['accounting']['annual_fit_forced_liquidation'] is False
         and cfg['accounting']['terminal_session'] == '2023-12-29'
         and cfg['accounting']['borrow_annual_bps'] == cfg['accounting']['funding_annual_bps'] == 0,
         'Costs, stop, breadth, fixed restrictions, funding or annual liquidation changed')
    for name in ('selection', 'preflight', 'replay'):
        audit = data['audit_' + name]
        need(audit['schema'] == 'dmv2-corrected-backtest-' + name + '-audit/v1'
             and audit['status'] == 'COMPLETE' and audit['source_plan_pin'] == z['source_plan_pin']
             and audit['protocol_pin'] == z['backtest_protocol_pin']
             and audit['new_estimator_fits'] == audit['new_inference_calls'] == 0
             and audit['original_source_pins_unchanged'] is True
             and audit['financial_2024_plus_values_converted'] is False
             and audit['execution_authority'] == authority, 'Completed backtest phase identity changed: ' + name)
    s, p, r = (data['audit_' + n] for n in ('selection', 'preflight', 'replay'))
    need(s['selection_sets'] == p['selection_sets'] == 15 and s['old_selection_reuse'] is False
         and s['selection_used_quote_or_future_target_availability'] is False
         and p['selection_audit_pin'] == z['named_JSON_pins']['audit_selection']
         and p['selection_changed_using_quote_coverage'] is False and p['performance_computed'] is False
         and p['all_maximum_unstopped_horizons_checked'] is True
         and r['selection_audit_pin'] == z['named_JSON_pins']['audit_selection']
         and r['preflight_audit_pin'] == z['named_JSON_pins']['audit_preflight']
         and r['scenarios'] == 120 and r['sessions'] == 1006 and r['outcome_months'] == 48
         and r['missing_selected_positions_dropped'] is False,
         'Fresh selection, coverage preflight or all120-case replay chain changed')
    return True


def make_plan(json_plan_path, expected_sha, reservation, output):
    jp = pin(json_plan_path); need(jp['sha256'] == expected_sha, 'Independent planning JSON manifest changed')
    z = read(check(jp))
    need(z['schema'] == 'dmv2-corrected-backtest-verification-JSON-plan/v1'
         and z['worker_pin'] == pin(__file__) and z['math_pin'] == pin(independent.__file__),
         'Independent verifier planning code changed')
    validate_json_budget(z)
    require_reservation(ROOT, reservation, z['projected_reads']['compressed_source'], jp, 'compressed_source')
    save_new(str(output) + '.STARTED.json', {'status': 'ATTEMPT_CONSUMED', 'JSON_plan_pin': jp})
    data = parsed_json(z['named_JSON_pins']); validate_authority(z, data)
    source = data['source']; specs = {'forecasts': {}, 'payoffs': {}, 'daily': {}, 'repairs': {},
        'selection': {}, 'preflight': {}, 'replay': {}}
    for f in source['forecasts']:
        key = str(f['year']) + '/' + f['arm']; item = f['predictions_pin']
        specs['forecasts'][key] = {'year': f['year'], 'arm': f['arm'],
            'metadata': projection(item, FORECAST_META, 12000),
            'values': projection(item, [*independent.KEYS, *independent.PROBABILITIES], 12000)}
    for p in source['payoffs']:
        specs['payoffs'][p['signal_date']] = {'state_key': 'state_' + p['signal_date'],
            'metadata': projection(p['monthly']['pin'], MONTHLY_META, 1200),
            'table_metadata': projection(p['values']['pin'], PAYOFF_META, 10),
            'values': projection(p['monthly']['pin'], [*MONTHLY_META[:-1], 'class_mean_return', 'status'], 1200),
            'table_values': projection(p['values']['pin'], ('class_id', 'payoff', 'status'), 10)}
    for a in source['daily']:
        specs['daily'][str(a['year'])] = {'metadata': projection(a['values']['pin'], QUOTE_META, 300000),
            'values': projection(a['values']['pin'], QUOTE_VALUES, 300000)}
    item = source['execution_repairs']['values']['pin']
    specs['repairs'] = {'metadata': projection(item, QUOTE_META, 40), 'values': projection(item, QUOTE_VALUES, 40)}
    for phase in ('selection', 'preflight', 'replay'):
        for key, item in data['audit_' + phase]['artifacts'].items():
            need(Path(item['path']).resolve() == BASE / phase / key and Path(key).suffix == '.parquet'
                 and '..' not in Path(key).parts, 'Backtest audit redirects a canonical artifact')
            specs[phase][key] = table_pair(item, 150000 if phase == 'selection' else 250000)
    selection_keys = {f'{a}/{b}/{n}.parquet' for a, b, n in itertools.product(
        independent.ARMS, independent.BASKETS, ('original_ledger', 'selected', 'rebalances', 'monthly'))}
    replay_keys = {f'arms/{a}/{b}/{v}/cost_{f:02d}bps/{n}.parquet' for a, b, v, f, n in itertools.product(
        independent.ARMS, independent.BASKETS, independent.VARIANTS, independent.FEES, (*LEDGERS, 'monthly'))}
    replay_keys |= {n + '.parquet' for n in ('comparison', 'yearly', 'selection_coverage')}
    preflight_keys = {f'coverage/{a}/{b}/{n}.parquet' for a, b, n in itertools.product(
        independent.ARMS, independent.BASKETS, ('missing_quotes', 'source_quality', 'retired_orders', 'action_requirements'))}
    preflight_keys.add('summary.parquet')
    need(set(specs['selection']) == selection_keys and set(specs['preflight']) == preflight_keys
         and set(specs['replay']) == replay_keys, 'Canonical60 selection/61preflight/1203replay table set changed')
    amount = projected_cost(specs) + sum(p['size_bytes'] for p in z['unique_JSON_pins'])
    need(0 < amount <= MAX_READ, 'Independent120-case verification exceeds frozen2GiB read cap')
    pins = collect_pins([source, data['audit_selection'], data['audit_preflight'], data['audit_replay'],
        z['named_JSON_pins'], z['independent_helper_pins']])
    for item in pins: check(item)
    for item in z['unique_JSON_pins']: check(item)
    return save_new(output, {'schema': 'dmv2-corrected-backtest-independent-read-plan/v1',
        'status': 'SEALED_BEFORE_FINANCIAL_TABLE_READ', 'JSON_plan_pin': jp,
        'planning_JSON_reservation': reservation, 'project_root': z['project_root'],
        'source_plan_pin': z['source_plan_pin'], 'backtest_protocol_pin': z['backtest_protocol_pin'],
        'named_JSON_pins': z['named_JSON_pins'], 'unique_JSON_pins': z['unique_JSON_pins'],
        'independent_helper_pins': z['independent_helper_pins'], 'source_and_artifact_pins': pins,
        'worker_pin': pin(__file__), 'math_pin': pin(independent.__file__), 'control_pin': pin(ROOT / 'control.py'),
        'supervisor_pin': pin(ROOT / 'supervise.py'), 'tables': specs,
        'projected_reads': {'compressed_source': amount, 'derived_matrix': 0},
        'new_estimator_fits': 0, 'new_inference_calls': 0, 'scenarios': 120,
        'disclosure': 'Whole approved Parquet column chunks may physically decode before Arrow row clipping; future/non-session values never enter financial Python. Integrity hash bytes are separate.'})


def load_helpers(items):
    modules = {}
    for name, item in items.items():
        need(item['sha256'] == HELPERS[name][1], 'Established independent helper changed')
        spec = importlib.util.spec_from_file_location('_corrected_independent_' + name, check(item))
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); modules[name] = module
    return modules


def read_projection(spec):
    import pyarrow.parquet as pq
    need(projection(spec['pin'], spec['columns'], spec['maximum_rows']) == spec,
         'Independent frozen footer projection changed')
    return pq.ParquetFile(spec['pin']['path']).read(columns=spec['columns'], use_threads=False)


def read_values(pair, metadata):
    table = read_projection(pair['values'])
    actual = table.select(pair['metadata']['columns'])
    need((metadata_fingerprint(actual) == metadata if type(metadata) is str else actual.to_pylist() == metadata),
         'Saved table metadata/order changed between guarded passes')
    return table.to_pandas()


def read_preflight_records(pair, metadata, *, name):
    """Keep Arrow null IDs as None in the four nonfinancial diagnostics.

    Use exactly the same reserved projection and metadata identity as the
    DataFrame reader.  Avoid pandas' nullable-ID conversion; this grants no
    missing-value equivalence and true NaN/inf still reach the strict comparer.
    """
    fields = {
        'missing_quotes': {'signal_date', 'execution_date', 'scheduled_exit_date',
            'source_security_id', 'intrinio_security_id', 'side', 'date', 'role', 'reason'},
        'source_quality': {'signal_date', 'execution_date', 'scheduled_exit_date',
            'source_security_id', 'intrinio_security_id', 'side', 'date', 'role', 'reason'},
        'retired_orders': {'signal_date', 'execution_date', 'source_security_id',
            'intrinio_security_id', 'event_id', 'status'},
        'action_requirements': {'signal_date', 'source_security_id', 'date', 'event_id',
            'predecessor_id', 'successor_id', 'status'},
    }
    need(name in fields and set(pair['values']['columns']) <= fields[name],
         'Arrow record reader is restricted to registered preflight metadata fields')
    table = read_projection(pair['values'])
    need(set(table.column_names) == set(pair['values']['columns'])
         and (table.num_rows == 0 or set(table.column_names) == fields[name]),
         'Nonempty preflight diagnostic requires its exact registered metadata schema')
    actual = table.select(pair['metadata']['columns'])
    need((metadata_fingerprint(actual) == metadata if type(metadata) is str else actual.to_pylist() == metadata),
         'Saved preflight metadata/order changed between guarded passes')
    return table.to_pylist()


def metadata_fingerprint(table):
    """Compact exact Arrow metadata identity; never cache millions of dicts."""
    import pyarrow as pa
    table = table.combine_chunks().replace_schema_metadata(None)
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as stream: stream.write_table(table)
    return hashlib.sha256(sink.getvalue()).hexdigest()


def guarded_metadata(pair, *, context, run_scope):
    """Artifact-specific metadata guards; never broadly waive row clocks."""
    try:
        table = read_projection(pair['metadata'])
        if context == ('preflight', 'summary.parquet'):
            independent.preflight_summary_metadata(table.to_pylist(), table.column_names,
                pair['values']['schema_fields'], run_scope=run_scope)
        elif table.num_rows:
            for batch in table.to_batches(max_chunksize=4096):
                independent.generic_table_metadata(batch.to_pylist(), table.column_names, allow_empty=True)
        else:
            independent.generic_table_metadata([], table.column_names, allow_empty=True)
        return metadata_fingerprint(table)
    except ValueError as error:
        raise ValueError('Metadata guard rejected ' + '/'.join(context) + ': ' + str(error)) from error


def require_guard(plan_pin, z):
    path, sha = (os.environ.get(k) for k in ('DMV2_GUARD_LAUNCH_PATH', 'DMV2_GUARD_LAUNCH_SHA256'))
    need(path and sha and pin(path)['sha256'] == sha, 'Exact external verification resource guard required')
    launch = read(path)
    need(launch['schema'] == 'dmv2-corrected-refit-phase-launch/v1'
         and launch['protocol_pin'] == z['backtest_protocol_pin'] and launch['action_plan_pin'] == plan_pin
         and launch['time_category'] in ('backtest', 'verification')
         and launch['command'] == [sys.executable, *sys.argv], 'Verifier command/phase differs from sealed launch')
    for item in (z['worker_pin'], z['math_pin'], z['control_pin'], z['supervisor_pin'], *z['independent_helper_pins'].values()):
        need(item in launch['code_pins'], 'Independent verifier/helper code omitted from resource guard')
    need(all(os.environ.get(k) == '1' for k in NUMERIC_ENV), 'Single numeric-thread environment required')


def rows_equal(actual, expected, label):
    need(len(actual) == len(expected), label + ': row count changed')
    def canon(r):
        need(type(r) is dict and all(type(k) is str for k in r), label + ': metadata record must use string fields')
        normalized = {}
        for field, value in r.items():
            if value is not None and field in DATE_FIELDS:
                if isinstance(value, datetime):
                    need(value.tzinfo is None and value.hour == value.minute == value.second == value.microsecond == 0
                         and getattr(value, 'nanosecond', 0) == 0, label + ': timezone/intraday metadata date')
                    value = value.date().isoformat()
                elif isinstance(value, date): value = value.isoformat()
                need(type(value) is str, label + ': arbitrary object in metadata date')
                independent.day(value, lower='2019-12-01' if field in ('signal_date', 'held_signal_date') else '2020-01-01')
            elif value is not None:
                need(type(value) in (str, bool, int, float), label + ': arbitrary object outside date fields')
            normalized[field] = value
        return json.dumps(normalized, sort_keys=True, ensure_ascii=False, allow_nan=False)
    need(sorted(map(canon, actual)) == sorted(map(canon, expected)), label + ': logical rows changed')


def reconstruct_coverage(all_selections):
    """Mirror only the producer's registered union serialization, not decisions.

    Saved-selection plan keys are JSON-sorted. Thus its first class frame sets
    the union column order and each other verified schema supplies null padding.
    Exact frame comparison still checks every registered column, value and dtype.
    """
    import pandas as pd
    return pd.concat([tables['monthly'].assign(arm=arm, basket=basket)
        for (arm, basket), tables in sorted(all_selections.items())], ignore_index=True)


def class_breadth_guards(tables, windows, basket):
    monthly, rebalances, selected, ledger = (tables[k] for k in ('monthly', 'rebalances', 'selected', 'original_ledger'))
    need(monthly.signal_date.tolist() == windows.signal_date.tolist()
         and rebalances.signal_date.tolist() == windows.signal_date.tolist(), 'A class basket lost a monthly rebalance')
    for window in windows.to_dict('records'):
        m = monthly.loc[monthly.signal_date.eq(window['signal_date'])].iloc[0]
        r = rebalances.loc[rebalances.signal_date.eq(window['signal_date'])].iloc[0]
        n = min(10, int(m.long_candidates), int(m.short_candidates))
        need(m.status == r.selection_status == ('available' if n else 'withheld_zero_mode_candidates')
             and m.execution_date == r.execution_date == window['entry_date'],
             'Zero-sided class basket became synthetic cash or changed execution')
        need(int(m.long_selected) == int(m.short_selected) == int(m.selected_names_per_side) == n
             and int(r.selected_names_per_side) == n, 'Class basket actual balanced breadth changed')
        s = selected.loc[selected.signal_date.eq(window['signal_date'])]
        need(len(s) == 2*n and set(s.side) <= {'long', 'short'}, 'Extra/unknown class basket side')
        original = ledger.loc[ledger.signal_date.eq(window['signal_date'])]
        if not n: need(not original.selected.any(), 'Zero-sided class basket selected a holding')
    return True


def begin_verification_attempt(output, plan_pin):
    """Consume a V5 attempt without altering any prior-version marker/proof."""
    output = Path(output).resolve()
    need(output == ROOT / 'run/verification/backtest.json' and not output.exists(),
         'Fresh canonical independent receipt required')
    return save_new(str(output) + '.v5.STARTED.json',
        {'status': 'ATTEMPT_CONSUMED', 'plan_pin': plan_pin, 'verifier_revision': 5})


def verify(plan_path, expected_sha, reservation, output):
    started = time.monotonic(); pp = pin(plan_path)
    need(pp['sha256'] == expected_sha, 'Independent read plan changed')
    z = read(check(pp))
    need(z['schema'] == 'dmv2-corrected-backtest-independent-read-plan/v1'
         and z['worker_pin'] == pin(__file__) and z['math_pin'] == pin(independent.__file__)
         and z['scenarios'] == 120 and z['projected_reads']['derived_matrix'] == 0
         and 0 < z['projected_reads']['compressed_source'] <= MAX_READ, 'Independent verifier identity/budget changed')
    validate_json_budget(z, total=projected_cost(z['tables']) + sum(p['size_bytes'] for p in z['unique_JSON_pins']))
    require_reservation(ROOT, reservation, z['projected_reads']['compressed_source'], pp, 'compressed_source')
    require_guard(pp, z)
    attempt = begin_verification_attempt(output, pp)
    for item in z['source_and_artifact_pins']: check(item)
    for item in (z['control_pin'], z['supervisor_pin'], z['JSON_plan_pin']): check(item)
    data = parsed_json(z['named_JSON_pins']); validate_authority(z, data)
    modules = load_helpers(z['independent_helper_pins'])
    equations, ret, overlay = (modules[k] for k in ('cash_units', 'ret_selection', 'quote_overlay'))
    sessions, windows = equations.independent_clock(); windows_rows = windows.to_dict('records')
    need(len(sessions) == 1006 and len(windows) == 48, 'Independent four-year trading clock changed')
    specs = z['tables']; forecast_meta = {}; common = {}; payoff_meta = {}; selected_meta = {}; quote_meta = {}
    # Every forecast, RET and source quote clock precedes ALL numerical Python.
    for key, spec in specs['forecasts'].items():
        rows = read_projection(spec['metadata']).to_pylist()
        cohort = independent.forecast_metadata(rows, spec['year'], windows_rows)
        if spec['year'] in common: need(cohort == common[spec['year']], 'Five model cohorts/order differ')
        else: common[spec['year']] = cohort
        forecast_meta[key] = rows
    for signal, spec in specs['payoffs'].items():
        monthly = read_projection(spec['metadata']).to_pylist()
        table = read_projection(spec['table_metadata']).to_pylist()
        independent.payoff_metadata(monthly, table, data[spec['state_key']], signal)
        payoff_meta[signal] = (monthly, table)
    need(sorted(payoff_meta) == windows.signal_date.tolist(), 'Corrected RET states lost an authorized signal')
    for year, spec in specs['daily'].items():
        table = read_projection(spec['metadata'])
        positions, excluded = independent.quote_positions(table.select(['trading_date', 'intrinio_security_id']).to_pylist(), int(year), sessions)
        quote_meta[year] = (metadata_fingerprint(table), positions, len(excluded))
    repair_meta = read_projection(specs['repairs']['metadata']).to_pylist()
    independent.generic_table_metadata(repair_meta, specs['repairs']['metadata']['columns'])
    need(len(repair_meta) == 40 and all(r['trading_date'] in sessions for r in repair_meta),
         'The exact forty execution repairs left the authorized clock')
    for phase in ('selection', 'preflight', 'replay'):
        selected_meta[phase] = {}
        run_scope = {'sessions': data['audit_replay']['sessions'],
            'outcome_months': data['audit_replay']['outcome_months'],
            'selection_sets': data['audit_preflight']['selection_sets'],
            'source_plan_pin': z['source_plan_pin'],
            'preflight_source_plan_pin': data['audit_preflight']['source_plan_pin'],
            'replay_source_plan_pin': data['audit_replay']['source_plan_pin']}
        for key, pair in specs[phase].items():
            selected_meta[phase][key] = guarded_metadata(pair, context=(phase, key), run_scope=run_scope)
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    forecasts = {arm: [] for arm in independent.ARMS}
    for key, spec in specs['forecasts'].items():
        values = read_projection(spec['values']); meta = forecast_meta[key]
        need(values.select(independent.KEYS).to_pylist() == [{k: r[k] for k in independent.KEYS} for r in meta],
             'Probability row keys/order changed')
        frame = pd.DataFrame(meta)
        p = np.column_stack([values[c].to_numpy(zero_copy_only=False) for c in independent.PROBABILITIES])
        eligible = frame.prediction_eligible.to_numpy(dtype=bool)
        need(np.isfinite(p[eligible]).all() and (p[eligible] >= 0).all()
             and (np.abs(p[eligible].sum(axis=1)-1) <= 1e-8).all() and np.isnan(p[~eligible]).all(),
             'New saved probability simplex/withheld nulls changed')
        for j, col in enumerate(independent.PROBABILITIES): frame[col] = p[:, j]
        forecasts[spec['arm']].append((spec['year'], frame))
    forecasts = {a: pd.concat([f for _, f in sorted(parts, key=lambda p: p[0])], ignore_index=True)
                 for a, parts in forecasts.items()}
    payoffs, payoff_checks = {}, []
    for signal, spec in specs['payoffs'].items():
        monthly = read_projection(spec['values']); table = read_projection(spec['table_values'])
        need(monthly.select(spec['metadata']['columns']).to_pylist() == payoff_meta[signal][0]
             and table.select(spec['table_metadata']['columns']).to_pylist() == payoff_meta[signal][1],
             'Saved RET grid changed between guarded passes')
        value = independent.payoff_values(monthly.to_pylist(), table.to_pylist(), data[spec['state_key']], signal)
        if value['mu'] is not None: value['mu'] = np.asarray(value['mu'], dtype=float)
        payoffs[signal] = value
        payoff_checks.append({'signal_date': signal, 'calendar_months': 120,
            'vector_available': value['mu'] is not None, 'saved_equal_month_means_rechecked': True,
            'learning_labels_or_future_returns_read': False})
    all_selections, selection_checks = {}, []
    for arm, basket in itertools.product(independent.ARMS, independent.BASKETS):
        tables = {n: read_values(specs['selection'][f'{arm}/{basket}/{n}.parquet'],
            selected_meta['selection'][f'{arm}/{basket}/{n}.parquet'])
            for n in ('original_ledger', 'selected', 'monthly', 'rebalances')}
        if basket == 'ret':
            report = ret.verify_selection(forecasts[arm], tables, windows, payoffs)
        else:
            report = equations.verify_selection(forecasts[arm], tables['original_ledger'], tables['selected'],
                tables['monthly'], windows, basket=basket)
            class_breadth_guards(tables, windows, basket)
        # The full original ledger is needed only for independent selection
        # verification. Keep the three small execution frames afterward.
        all_selections[(arm, basket)] = {k: value for k, value in tables.items() if k != 'original_ledger'}
        selection_checks.append({'arm': arm, 'basket': basket, **report})
    del forecasts, forecast_meta, common, payoff_meta, payoffs, tables
    pieces, outside_counts = [], {}
    for year, spec in specs['daily'].items():
        table = read_projection(spec['values']); fingerprint, positions, outside = quote_meta[year]
        need(metadata_fingerprint(table.select(spec['metadata']['columns'])) == fingerprint,
             'Original daily quote metadata changed')
        # The whole approved Arrow block may decode, but only independently
        # calendar/year-filtered rows enter financial Python.
        filtered = table.take(pa.array(positions, type=pa.int64()))
        pieces.append(filtered.to_pandas()); outside_counts[year] = outside
    original = pd.concat(pieces, ignore_index=True); del pieces
    del quote_meta, table, filtered, positions
    repairs = read_values(specs['repairs'], repair_meta)
    phase = data['execution_phase_pin']; manifest = data['execution_manifest_pin']
    source = data['source']; evidence = {p['sha256'] for p in source['execution_evidence_pins']}
    def rejection(change):
        item = change['rejection_evidence_pin']; need(item['sha256'] in evidence, 'Accepted-bar rejection evidence detached')
        path = (Path(z['project_root']) / item['path']).resolve()
        check({**clean_pin(item), 'path': str(path)})
    prices, quote_changes = overlay.reconstruct_overlay(original, repairs, manifest['quote_amendments'],
        sessions=sessions, authenticate_rejection=rejection)
    del original, repairs
    need(len(quote_changes) == 40 and not prices.duplicated(['trading_date', 'intrinio_security_id']).any(),
         'Execution-only quote overlay changed or collided with new routes')
    metadata = prices[list(QUOTE_META)].to_dict('records')
    preflight_summary = read_values(specs['preflight']['summary.parquet'], selected_meta['preflight']['summary.parquet'])
    need(len(preflight_summary) == 15 and not preflight_summary.duplicated(['arm', 'basket']).any(),
         'Preflight summary lost an arm/basket')
    preflight_checks = []
    for (arm, basket), tables in all_selections.items():
        expected = independent.independent_preflight(tables['selected'].to_dict('records'),
            tables['rebalances'].to_dict('records'), metadata, sessions,
            phase['event_registry']['events'], phase['historical_retirements'])
        for name in ('missing_quotes', 'source_quality', 'retired_orders', 'action_requirements'):
            key = f'coverage/{arm}/{basket}/{name}.parquet'
            actual = read_preflight_records(specs['preflight'][key], selected_meta['preflight'][key], name=name)
            rows_equal(actual, expected[name], 'Independent maximum-horizon ' + name)
        row = preflight_summary.loc[preflight_summary.arm.eq(arm) & preflight_summary.basket.eq(basket)]
        need(len(row) == 1, 'Missing preflight summary arm/basket')
        for k, v in expected['summary'].items(): need(row.iloc[0][k] == v, 'Independent preflight summary changed: ' + k)
        need(row.iloc[0].withheld_selection_months == int(tables['monthly'].status.ne('available').sum()),
             'Zero-sided selection month was dropped from quote preflight')
        preflight_checks.append({'arm': arm, 'basket': basket, **expected['summary']})
    del metadata, expected, preflight_summary
    # Construct one price lookup only after the separate nonfinancial
    # maximum-horizon metadata copy has been released.
    quote_index = {(r['trading_date'], r['intrinio_security_id']): r for r in prices.to_dict('records')}
    comparison = read_values(specs['replay']['comparison.parquet'], selected_meta['replay']['comparison.parquet'])
    yearly = read_values(specs['replay']['yearly.parquet'], selected_meta['replay']['yearly.parquet'])
    coverage = read_values(specs['replay']['selection_coverage.parquet'], selected_meta['replay']['selection_coverage.parquet'])
    expected_cases = set(itertools.product(independent.ARMS, independent.BASKETS, independent.VARIANTS, independent.FEES))
    need(len(comparison) == 120 and not comparison.duplicated(['arm', 'basket', 'variant', 'cost_bps']).any()
         and set(map(tuple, comparison[['arm', 'basket', 'variant', 'cost_bps']].to_numpy())) == expected_cases,
         'All120comparison scenarios must remain, including withheld cases')
    need(len(yearly) == 480 and not yearly.duplicated(['arm', 'basket', 'variant', 'cost_bps', 'year']).any()
         and set(yearly.year) == set(independent.YEARS), 'Full480scenario/year grid required')
    need(len(coverage) == 15*48 and not coverage.duplicated(['arm', 'basket', 'signal_date']).any(),
         'All720arm/basket/month coverage rows required')
    ret.exact_frame(coverage, reconstruct_coverage(all_selections),
        'Replay coverage differs from exact sorted saved-monthly union/order/null-padding/dtypes')
    scenario_checks = []
    for arm, basket, variant, fee in itertools.product(independent.ARMS, independent.BASKETS, independent.VARIANTS, independent.FEES):
        base = f'arms/{arm}/{basket}/{variant}/cost_{fee:02d}bps/'
        tables = {name: read_values(specs['replay'][base + name + '.parquet'],
            selected_meta['replay'][base + name + '.parquet']) for name in LEDGERS}
        monthly = read_values(specs['replay'][base + 'monthly.parquet'], selected_meta['replay'][base + 'monthly.parquet'])
        chosen = all_selections[(arm, basket)]
        summary = comparison.loc[comparison.arm.eq(arm) & comparison.basket.eq(basket)
            & comparison.variant.eq(variant) & comparison.cost_bps.eq(fee)].iloc[0]
        annual = yearly.loc[yearly.arm.eq(arm) & yearly.basket.eq(basket)
            & yearly.variant.eq(variant) & yearly.cost_bps.eq(fee)]
        kwargs = {'events': phase['event_registry']['events'], 'historical_retirements': phase['historical_retirements'],
            'cost_bps': fee, 'stop_fraction': None if variant == 'unstopped' else .1,
            'prepared_quotes': quote_index, 'identity_handovers': phase['identity_handover_registry']['handovers']}
        need(type(summary.complete_proxy_path) in (bool, np.bool_), 'Unknown path completeness')
        if summary.complete_proxy_path:
            checked = equations.verify_scenario(tables, prices, chosen['selected'], chosen['rebalances'], sessions, **kwargs)
            equations.verify_metrics(tables['daily'], tables['trades'], monthly, summary, annual)
            checked.update(path_status='complete_proxy', economic_performance_available=True)
        else:
            checked = equations.verify_failed_scenario(tables, prices, chosen['selected'], sessions,
                rebalances=chosen['rebalances'], **kwargs)
            # Only this legacy np.allclose boundary needs a float null view.
            # The preceding cash/units verification consumes the original ledgers.
            metric_daily, metric_monthly = failed_metrics_float_views(tables['daily'], monthly)
            equations.verify_failed_metrics(metric_daily, tables['trades'], metric_monthly, summary, annual)
            del metric_daily, metric_monthly
            checked.update(path_status='failed_closed', economic_performance_available=False)
        scenario_checks.append({'arm': arm, 'basket': basket, 'variant': variant, 'cost_bps': fee, **checked})
        print(json.dumps({'phase': 'independent_corrected_backtest', 'verified_scenarios': len(scenario_checks),
                          'expected_scenarios': 120}), flush=True)
        del tables, monthly, annual, checked
    replay = data['audit_replay']; complete = int(comparison.complete_proxy_path.sum())
    need(replay['complete_proxy_paths'] == complete and replay['withheld_paths'] == 120-complete,
         'Saved completed/withheld audit counts differ')
    for item in z['source_and_artifact_pins']: check(item)
    for item in (pp, z['worker_pin'], z['math_pin'], z['control_pin'], z['supervisor_pin'], z['JSON_plan_pin']): check(item)
    receipt = {'schema': SCHEMA, 'status': 'PASS', 'plan_pin': pp, 'started_pin': attempt,
        'worker_pin': pin(__file__), 'math_pin': pin(independent.__file__),
        'source_plan_pin': z['source_plan_pin'], 'backtest_protocol_pin': z['backtest_protocol_pin'],
        'selection_audit_pin': z['named_JSON_pins']['audit_selection'],
        'preflight_audit_pin': z['named_JSON_pins']['audit_preflight'],
        'replay_audit_pin': z['named_JSON_pins']['audit_replay'],
        'final_model_verification_pin': source['final_model_verification_pin'],
        'independent_helper_pins': z['independent_helper_pins'],
        'scenarios': 120, 'sessions': 1006, 'outcome_months': 48, 'selection_sets': 15,
        'complete_proxy_paths': complete, 'withheld_paths': 120-complete,
        'all_scenarios_retained': True, 'original_source_pins_unchanged': True,
        'missing_selected_positions_dropped': False, 'new_estimator_fits': 0, 'new_inference_calls': 0,
        'production_selection_overlay_or_accounting_used_as_expected_answers': False,
        'old_execution_units_inherited_from_exact_independent_PASS': True,
        'rejected_learning_patches_are_separate': True, 'future_target_availability_used_for_selection': False,
        'typed_XOM_route_preserved': True, 'source_precision_tolerance_overridden': False,
        'historical_identity_knowledge_time_survivorship_or_investability_certified': False,
        'RET_learning_labels_recomputed': False, 'RET_source_month_class_means_recomputed': False,
        'saved_RET_vectors_and_causal_equal_month_formula_rechecked': True,
        'quote_overlay_checks': quote_changes, 'source_quote_rows_after_amendment': len(prices),
        'excluded_source_quote_rows_metadata_only': outside_counts,
        'payoff_checks': payoff_checks, 'selection_checks': selection_checks,
        'maximum_horizon_checks': preflight_checks, 'scenario_checks': scenario_checks,
        'reserved_compressed_source_bytes': z['projected_reads']['compressed_source'],
        'integrity_hash_io_is_separate_from_projected_financial_column_budget': True,
        'financial_2024_plus_values_converted': False, 'execution_seconds': time.monotonic()-started}
    return save_new(output, receipt)


def failed_metrics_float_views(daily, monthly):
    """Checked float64 copies for the legacy failed-metric comparison only.

    Arrow null columns can arrive as object/None instead of float/NaN.  This
    changes that representation without imputing any value or touching saved
    ledgers; every finite real value and every missing position is preserved.
    """
    import numbers
    import numpy as np
    import pandas as pd

    def view(frame, columns):
        result = frame.copy(deep=True)
        for column in columns:
            need(column in frame and list(frame.columns).count(column) == 1,
                 'Failed-metric float view requires one registered column: ' + column)
            converted, missing = [], []
            for value in frame[column]:
                if value is None or value is pd.NA:
                    converted.append(np.nan); missing.append(True); continue
                need(not isinstance(value, (bool, np.bool_)) and isinstance(value, numbers.Real),
                     'Failed-metric float view rejects non-real scalar: ' + column)
                if value != value:
                    converted.append(np.nan); missing.append(True); continue
                try:
                    number = float(value)
                except (OverflowError, ValueError, TypeError):
                    need(False, 'Failed-metric float view rejects nonrepresentable real: ' + column)
                need(np.isfinite(number), 'Failed-metric float view rejects infinity: ' + column)
                need(number == value,
                     'Failed-metric float view would change a finite value: ' + column)
                if isinstance(value, numbers.Integral):
                    need(int(number) == int(value),
                         'Failed-metric float view would round an integer: ' + column)
                converted.append(number); missing.append(False)
            values = np.asarray(converted, dtype=np.float64)
            need(np.array_equal(np.isnan(values), np.asarray(missing, dtype=bool)),
                 'Failed-metric float view changed the missing mask: ' + column)
            result[column] = values
        return result

    return view(daily, ('nav',)), view(monthly, ('nav', 'return'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('plan-json', 'plan', 'run'))
    for name in ('root', 'source-plan', 'protocol', 'json-plan', 'json-plan-sha256', 'plan', 'plan-sha256', 'reservation', 'output'):
        p.add_argument('--' + name)
    a = p.parse_args()
    if a.command == 'plan-json': result = plan_json(a.root, a.source_plan, a.protocol, a.output)
    elif a.command == 'plan': result = make_plan(a.json_plan, a.json_plan_sha256, a.reservation, a.output)
    else: result = verify(a.plan, a.plan_sha256, a.reservation, a.output)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__': main()
