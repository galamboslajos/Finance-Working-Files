"""Explicit, additive RET replay of five authenticated saved forecast arms.

This module never fits, loads an estimator, performs inference, discovers new
data or changes an existing research artifact. Payoffs use corrected targets;
portfolio selection is completed before quote-coverage inspection.
"""
from __future__ import annotations

import base64
from collections import defaultdict
import datetime as dt
import itertools
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

from .action_loading import ACTION_PRICE_COLUMNS
from .backtest_accounting import DailyAccountingPolicy
from .backtest_loading import _pin, _projected_bytes, _within
from .backtest_selection import PROBABILITIES, SAFE_COLUMNS
from .catalog import canonical_sha256
from .daily_backtest import _save_table, summarize_daily
from .monthly import _step_month
from .ret import (CLASS_IDS, MONTHLY_PAYOFF_SCHEMA, PAYOFF_SCHEMA, RET_POLICY,
                  RET_POLICY_DEFINITION, RetPayoffPanel, _fingerprint,
                  verify_ret_payoffs)
from .ret_selection import select_ret_walkforward_basket
from .rf_tuned_backtest import (COHORT_COLUMNS, NUMERIC_ENV, YEARS, authenticated,
    daily_footer_plan, execution_context, exclusive, need, read_json,
    require_frozen_pin, resources, runtime, validate_forecast_metadata, write_json)
from .targets import MonthlyTargetPanel, SAFE_METADATA, target_month_fingerprint, verify_target_panel
from .walkforward_accounting_v2 import IMPLEMENTATION_REVISION, run_walkforward_accounting_v2
from .walkforward_backtest import _configuration, _price_plan, yearly_breakdown
from .walkforward_loading import resolve_walkforward_year
from .walkforward_preflight import preflight_selected_horizons

PHASE = 'ret_continuous_backtest_v1'
ARMS = ('rf_paper15', 'xgb_paper15', 'rf_extended', 'xgb_extended', 'rf_extended_tuned')
BASKET = 'ret'
SCENARIOS = tuple(itertools.product(ARMS, ('unstopped', 'asset_stop_10pct'), (0, 5, 10, 25)))
LIMITS = {'prepare_wall_seconds': 300, 'preflight_wall_seconds': 900,
    'backtest_wall_seconds': 1200, 'verification_wall_seconds': 3600,
    'rss_stop_bytes': 4 * 1024**3, 'rss_warning_bytes': 3 * 1024**3,
    'minimum_available_memory_bytes': 7 * 1024**3, 'normal_pressure_dispatch': 1,
    'cumulative_compressed_read_bytes': 1024**3, 'output_bytes': 512 * 1024**2,
    'workers': 1, 'numeric_threads': 1}
NEW_FILES = ('src/dmv2/ret_backtest.py', 'src/dmv2/ret_selection.py',
    'scripts/run_ret_backtest.py', 'scripts/verify_ret_backtest.py',
    'scripts/watch_ret_backtest.py')


def configuration(root, path):
    cfg = read_json(path)
    need(cfg.get('schema') == 'dmv2-ret-backtest-execution/v1'
         and cfg.get('status') == 'FROZEN_EXECUTION_AUTHORIZED'
         and cfg.get('execution_authorized') is True, 'Explicit separate RET authority required')
    need(cfg.get('arm_ids') == list(ARMS) and cfg.get('scenarios') == 40,
         'Only the five saved arms and forty frozen RET scenarios are allowed')
    need(all(type(cfg.get(k)) is int and cfg[k] == 0 for k in
         ('new_estimator_fits', 'new_preprocessor_fits', 'new_inference_calls')),
         'RET must perform zero fits and zero inference')
    need(cfg.get('financial_2024_plus_authorized') is False and cfg.get('cloud_operations') is False,
         'Protected financial or cloud scope broadened')
    need(cfg.get('allow_withheld_paths') is True and cfg.get('resource_limits') == LIMITS,
         'Frozen resource or fail-closed coverage policy changed')
    selection = cfg.get('selection', {})
    required_selection = {'all_equal_scores': 'retain_existing_stable_id_tie_policy_report_zero_spread',
        'basket': BASKET, 'long_ties': 'RET_desc_source_id_asc',
        'names_per_side': 'min(10,floor(trade_eligible_count/2))',
        'no_future_price_or_label_availability_filter': True,
        'ranking': 'all_trade_eligible_predictions_by_full_probability_weighted_payoff_NO_mode_class_filter',
        'short_ties': 'RET_asc_source_id_asc_excluding_reserved_longs', 'sign_gate': False,
        'trade_restrictions': 'unchanged_DF_RDC'}
    need(all(selection.get(k) == v and type(selection.get(k)) is type(v) for k, v in required_selection.items()),
         'RET score ranking, ties, breadth or trading restrictions changed')
    payoff = cfg.get('payoffs', {})
    required_payoff = {'completed_calendar_months': 120, 'incomplete_grid': 'withhold_entire_vector',
        'monthly_class_means': 'equal_weight_observed_true_class_returns',
        'old_full_target_RET_states_reused': False, 'policy': RET_POLICY,
        'sources': 'exact_corrected_targets_used_by_saved_models', 'time_weighting': 'equal_months',
        'union_outcomes': ['2010-01', '2023-11']}
    need(payoff == required_payoff, 'Corrected complete-calendar RET payoff policy changed')
    expected_execution = {'annual_liquidation': False, 'asset_stop_choices': [None, .1],
        'borrow_and_funding_bps': 0, 'cost_bps': [0, 5, 10, 25], 'long_weight': .5,
        'missing_held_marks': 'retain_exposure_withhold_full_path', 'months': 48,
        'outcome_years': list(YEARS), 'same_original_actions_retirements_PENN_exception': True,
        'sessions': 1006, 'short_weight': -.5, 'stop_cash': 'cash_until_next_monthly_rebalance_no_replacement',
        'weights': 'equal_within_side_at_monthly_next_open'}
    need(cfg.get('execution') == expected_execution, 'Frozen daily execution, stop, cost or exposure rules changed')
    return cfg


def reserve_reads(run, phase, projected_bytes):
    need(type(projected_bytes) is int and projected_bytes >= 0, 'Invalid compressed reservation')
    path = Path(run) / 'read_usage.json'; state = read_json(path)
    need(state['schema'] == 'dmv2-ret-read-usage/v1' and state['limit_bytes'] == LIMITS['cumulative_compressed_read_bytes'],
         'RET read ledger changed')
    need(not any(row['phase'] == phase for row in state['reservations']), 'No automatic phase retries')
    need(state['total_projected_compressed_bytes'] == sum(row['projected_compressed_bytes'] for row in state['reservations']),
         'RET read ledger does not reconcile')
    total = state['total_projected_compressed_bytes'] + projected_bytes
    need(total <= state['limit_bytes'], 'RET read budget exceeded before decoding')
    state['reservations'].append({'phase': phase, 'projected_compressed_bytes': projected_bytes,
        'reserved_at': dt.datetime.now(dt.timezone.utc).isoformat()})
    state['total_projected_compressed_bytes'] = total
    write_json(path, state, replace=True)
    return state


def _absolute_pin(root, parent, item):
    path = authenticated(parent, item)
    return _pin(root, path)


def _source_bindings(root, cfg, tuned):
    source_path = authenticated(root, cfg['source_corrected_identity_pin'])
    source = source_path.parent; identity = read_json(source_path)
    need(canonical_sha256(identity) == tuned['execution_reference_root_identity_sha256']
         and str(source.relative_to(root)) == tuned['source_reference_directory'],
         'Corrected target source identity differs from verified execution source')
    pins = [cfg['source_corrected_identity_pin']]; annuals = []
    for year in range(2010, 2024):
        resolved = resolve_walkforward_year(root, source, identity, year)
        directory, audit = Path(resolved['directory']), resolved['audit']
        ap = _pin(root, directory / 'audit.json')
        if year >= 2020:
            require_frozen_pin(root, ap, tuned['source_pins'])
        else:
            reference = identity['annual_sources'][str(year)]
            for key in ('parent_identity_pin', 'parent_audit_pin'):
                authenticated(root, reference[key]); pins.append(reference[key])
        binding = {'year': year, 'directory': str(directory.relative_to(root)),
            'audit_pin': ap, 'build_identity_sha256': resolved['build_identity_sha256']}
        pins.append(ap)
        for name in ('targets', 'target_state', 'target_diagnostics'):
            item = _absolute_pin(root, directory, audit['artifacts'][name])
            # The original sealed annual audit authenticates these exact
            # children, including targets not consumed by the prior daily
            # backtest. They need not also appear as duplicated direct pins.
            binding[name + '_pin'] = item; pins.append(item)
        annuals.append(binding)
    return annuals, pins


def _forecast_bindings(root, tuned):
    old_cfg = tuned['configuration']; source = root / tuned['source_reference_directory']
    pins = []
    model_pin = _pin(root, source / 'models/audit.json')
    require_frozen_pin(root, model_pin, tuned['source_pins'])
    audit = read_json(authenticated(root, model_pin)); pins.append(model_pin)
    vp = old_cfg['source_model_verification_pin']; verified = read_json(authenticated(root, vp)); pins.append(vp)
    need(verified['status'] == 'PASSED' and verified['models_audit_sha256'] == model_pin['sha256']
         and verified['model_identity_sha256'] == audit['model_identity_sha256']
         and verified['build_identity_sha256'] == tuned['execution_reference_root_identity_sha256']
         and verified['annual_fits_reconciled'] == 16 and verified['stitched_prediction_months'] == 48,
         'The original four saved forecast arms lack their exact independent PASS')
    entries = []
    for arm in ARMS[:-1]:
        arm_pin = _absolute_pin(root, source / 'models', audit['arms'][arm])
        aa = read_json(authenticated(root, arm_pin)); pins.append(arm_pin)
        need(aa['status'] == 'COMPLETE' and aa['model_identity_sha256'] == audit['model_identity_sha256'],
             'Stitched original arm differs from verified models')
        pp = _absolute_pin(root, Path(root / arm_pin['path']).parent, aa['artifacts']['predictions'])
        pins.append(pp)
        entries.append({'arm': arm, 'files': [{'outer_year': None, 'predictions_pin': pp,
            'arm_audit_pin': arm_pin}], 'model_identity_sha256': audit['model_identity_sha256']})
    entries.append({'arm': ARMS[-1], 'files': tuned['forecast_entries']})
    for entry in tuned['forecast_entries']:
        pp = entry['predictions_pin']; authenticated(root, pp); pins.append(pp)
    return entries, pins


def prepare(root, *, configuration_path, output_directory):
    root = Path(root).resolve(); output = _within(root / 'outputs/ret_backtest', Path(output_directory).resolve())
    need(output != root / 'outputs/ret_backtest' and not output.exists(), 'Fresh private RET directory required')
    cfg = configuration(root, configuration_path); started = time.monotonic()
    tuned = read_json(authenticated(root, cfg['tuned_backtest_identity_pin']))
    ta = read_json(authenticated(root, cfg['tuned_backtest_audit_pin']))
    tv = read_json(authenticated(root, cfg['tuned_backtest_verification_pin']))
    tba = read_json(authenticated(root, ta['backtest_audit_pin']))
    need(tuned['schema'] == 'dmv2-rf-tuned-backtest-root/v1'
         and ta['build_identity_sha256'] == canonical_sha256(tuned)
         and tv['status'] == 'PASS' and tv['build_identity_sha256'] == canonical_sha256(tuned)
         and tv['root_audit_sha256'] == cfg['tuned_backtest_audit_pin']['sha256']
         and tv['backtest_audit_sha256'] == ta['backtest_audit_pin']['sha256']
         and tv['backtest_phase_identity_sha256'] == tba['phase_identity_sha256']
         and tv['scenarios'] == 16 and tv['sessions'] == 1006,
         'Verified tuned replay dependency is absent or changed')
    for item in (*tuned['code_pins'], *tuned['source_pins']): authenticated(root, item)
    base_pin = tuned['configuration']['base_configuration_pin']
    base = read_json(authenticated(root, base_pin)); _configuration(base)
    annuals, source_pins = _source_bindings(root, cfg, tuned)
    forecasts, forecast_pins = _forecast_bindings(root, tuned)
    source_pins.extend([*tuned['source_pins'], *forecast_pins, ta['backtest_audit_pin'], cfg['tuned_backtest_identity_pin'],
        cfg['tuned_backtest_audit_pin'], cfg['tuned_backtest_verification_pin'], base_pin])
    code_pins = list(tuned['code_pins']) + [_pin(root, root / name) for name in NEW_FILES]
    by_path = {}
    for item in code_pins:
        path = str((root / item['path']).resolve())
        need(path not in by_path or by_path[path]['sha256'] == item['sha256'], 'A frozen source worker changed')
        by_path[path] = item
    identity = {'schema': 'dmv2-ret-backtest-root/v1', 'project_root': str(root), 'run_root': str(output),
        'configuration': cfg, 'configuration_pin': _pin(root, configuration_path),
        'base_configuration_pin': base_pin, 'base_configuration': base,
        'tuned_backtest_identity_pin': cfg['tuned_backtest_identity_pin'],
        'source_reference_directory': tuned['source_reference_directory'],
        'execution_reference_root_identity_sha256': tuned['execution_reference_root_identity_sha256'],
        'execution_reference_audit_pin': tuned['execution_reference_audit_pin'],
        'execution_reference_phase_pin': tuned['execution_reference_phase_pin'],
        'annual_sources': annuals, 'forecast_entries': forecasts,
        'source_pins': list({(str((root / p['path']).resolve()), p['sha256']): p for p in source_pins}.values()),
        'code_pins': list(by_path.values()), 'runtime': runtime(),
        'new_estimator_fits': 0, 'new_preprocessor_fits': 0, 'new_inference_calls': 0,
        'financial_2024_plus_read': False, 'old_runs_and_pointers_changed': False,
        'created_at': dt.datetime.now(dt.timezone.utc).isoformat()}
    resources(output, cfg, started, LIMITS['prepare_wall_seconds'], launch=True)
    output.mkdir(parents=True)
    write_json(output / 'build_identity.json', identity)
    write_json(output / 'read_usage.json', {'schema': 'dmv2-ret-read-usage/v1',
        'limit_bytes': LIMITS['cumulative_compressed_read_bytes'], 'total_projected_compressed_bytes': 0, 'reservations': []})
    write_json(output / 'audit.json', {'status': 'PREPARED', 'phase': PHASE,
        'build_identity_sha256': canonical_sha256(identity), 'financial_values_decoded': False,
        'models_fitted': 0, 'preprocessors_fitted': 0, 'new_inference_calls': 0})
    return output


def load_run(root, run):
    root = Path(root).resolve(); run = _within(root / 'outputs/ret_backtest', Path(run).resolve())
    identity, audit = read_json(run / 'build_identity.json'), read_json(run / 'audit.json')
    need(identity['schema'] == 'dmv2-ret-backtest-root/v1' and identity['project_root'] == str(root)
         and identity['run_root'] == str(run) and canonical_sha256(identity) == audit['build_identity_sha256'],
         'RET prepared identity changed')
    cfg = configuration(root, authenticated(root, identity['configuration_pin']))
    need(cfg == identity['configuration'] and identity['runtime'] == runtime(), 'RET frozen runtime/config changed')
    for p in (*identity['code_pins'], *identity['source_pins']): authenticated(root, p)
    need(all(os.environ.get(name) == '1' for name in NUMERIC_ENV), 'RET requires one numeric thread')
    base = read_json(authenticated(root, identity['base_configuration_pin']))
    need(base == identity['base_configuration'], 'Frozen base execution rules changed')
    return identity, audit, cfg, base


def target_footer_plan(root, annuals):
    plan, total = [], 0
    for annual in annuals:
        files = {}
        for name in ('targets', 'target_diagnostics'):
            path = authenticated(root, annual[name + '_pin']); pf = pq.ParquetFile(path)
            need(pf.metadata.num_rows <= (12000 if name == 'targets' else 12), 'Unbounded annual target file')
            amount = _projected_bytes(path, pf.schema_arrow.names)
            files[name] = {'path': str(path), 'columns': pf.schema_arrow.names, 'projected_compressed_bytes': amount}
            total += amount
        plan.append({**annual, 'files': files})
    need([x['year'] for x in plan] == list(range(2010, 2024)), 'Corrected target history must cover fourteen years')
    return plan, total


def load_monthly_class_means(root, plan):
    """Authenticate annual schemas and monthly hashes before numerical means.

    Full annual Arrow blocks may decode while reading Parquet; no December2023
    financial row is converted to Python, fingerprinted or used by a payoff.
    No 2024+ source file is admitted at all.
    """
    means, windows, target_pins, diagnostic_pins = {}, {}, {}, {}
    source_policy = None
    for entry in plan:
        year = entry['year']; need(2010 <= year <= 2023, 'Protected annual target source')
        state = read_json(authenticated(root, entry['target_state_pin']))
        need(canonical_sha256(state['provenance']) == state['artifact_sha256'], 'Annual target state fingerprint changed')
        schemas = {k: pa.ipc.read_schema(pa.BufferReader(base64.b64decode(v, validate=True))) for k, v in state['schemas'].items()}
        tables = {name: pq.ParquetFile(entry['files'][name]['path']).read(columns=entry['files'][name]['columns']).cast(schemas[name])
                  for name in ('targets', 'target_diagnostics')}
        cutoff = min(state['provenance']['label_observed_through'], '2023-11-30')
        verify_target_panel(MonthlyTargetPanel(tables['targets'], tables['target_diagnostics'],
            state['provenance'], state['artifact_sha256']), through=cutoff, start_month=f'{year}-01')
        need(source_policy in (None, state['provenance']['policy_sha256']), 'Target policy changes across annual sources')
        source_policy = state['provenance']['policy_sha256']
        safe = tables['targets'].select(list(SAFE_METADATA)).to_pylist()
        need(all(row['outcome_month'].startswith(str(year) + '-') for row in safe), 'Annual target dates leave source year')
        source_windows = state['provenance']['month_windows']
        for window in source_windows:
            month = window['outcome_month']
            if month > '2023-11': continue
            need(month not in means and f'{year}-01' <= month <= f'{year}-12'
                 and window['outcome_end'] <= cutoff, 'Target month not completed or duplicate')
            selected = tables['targets'].filter(pc.equal(tables['targets']['outcome_month'], month))
            diagnostic = tables['target_diagnostics'].filter(pc.equal(tables['target_diagnostics']['outcome_month'], month))
            need(target_month_fingerprint(selected) == state['provenance']['month_target_fingerprints'][month]
                 and target_month_fingerprint(diagnostic) == state['provenance']['month_diagnostic_fingerprints'][month],
                 'Original corrected month fingerprint changed')
            groups = {k: [] for k in CLASS_IDS}
            for row in selected.select(['target_available', 'target_class', 'label_return']).to_pylist():
                if row['target_available']:
                    k, value = row['target_class'], row['label_return']
                    need(type(k) is int and k in CLASS_IDS and isinstance(value, (float, int))
                         and not isinstance(value, bool) and math.isfinite(value) and value >= -1,
                         'Invalid released corrected class/return')
                    groups[k].append(float(value))
            original = selected.num_rows
            eligible = int(pc.sum(pc.cast(selected['prediction_eligible'], pa.int64())).as_py())
            means[month] = [{'outcome_month': month, 'outcome_end': window['outcome_end'], 'class_id': k,
                'original_member_count': original, 'prediction_eligible_count': eligible, 'member_count': len(groups[k]),
                'class_mean_return': math.fsum(x / len(groups[k]) for x in groups[k]) if groups[k] else None,
                'status': 'available' if groups[k] else 'empty_realized_class'} for k in CLASS_IDS]
            windows[month] = window
            target_pins[month] = state['provenance']['month_target_fingerprints'][month]
            diagnostic_pins[month] = state['provenance']['month_diagnostic_fingerprints'][month]
    expected = [_step_month('2010-01', i) for i in range(167)]
    need(sorted(means) == expected, 'RET requires the exact167 completed source months January2010–November2023')
    return {'monthly': means, 'windows': windows, 'target_fingerprints': target_pins,
        'diagnostic_fingerprints': diagnostic_pins, 'source_target_policy_sha256': source_policy}


def build_shared_payoffs(class_means, signals):
    signals = list(signals)
    need(len(signals) == len(set(signals)) == 48, 'RET must retain all48 unique original signal states')
    result = {}
    for signal in signals:
        last = signal[:7]; first = _step_month(last, -119)
        months = [_step_month(first, i) for i in range(120)]
        need(all(m in class_means['monthly'] for m in months), 'Payoff window lacks an actual calendar month')
        rows = [dict(row) for month in months for row in class_means['monthly'][month]]
        need(all(row['outcome_end'] <= signal for row in rows), 'Future outcome entered a RET vector')
        available = all(row['status'] == 'available' for row in rows)
        table = pa.Table.from_pylist([{'class_id': k,
            'payoff': math.fsum(row['class_mean_return'] / 120 for row in rows if row['class_id'] == k) if available else None,
            'status': 'available' if available else 'withheld_incomplete_120_month_class_grid'} for k in CLASS_IDS], schema=PAYOFF_SCHEMA)
        monthly = pa.Table.from_pylist(rows, schema=MONTHLY_PAYOFF_SCHEMA)
        provenance = {'ret_policy': RET_POLICY, 'policy_sha256': canonical_sha256(RET_POLICY_DEFINITION),
            'source_target_policy_sha256': class_means['source_target_policy_sha256'], 'signal_date': signal,
            'label_observed_through': signal, 'first_outcome_month': first, 'last_outcome_month': last,
            'expected_calendar_months': 120, 'expected_classes': list(CLASS_IDS), 'expected_month_class_cells': 1200,
            'available_month_class_cells': sum(r['status'] == 'available' for r in rows), 'vector_available': available,
            'monthly_class_mean': 'equal_weight_observed_realized_class_returns', 'payoff_time_weighting': 'equal_calendar_month_weights',
            'missing_month_or_class_policy': 'withhold_entire_payoff_vector', 'uses_predicted_classes': False,
            'financial_rows_after_cutoff_consumed': False,
            'source_month_target_fingerprints': {m: class_means['target_fingerprints'][m] for m in months},
            'source_completion_windows': [class_means['windows'][m] for m in months],
            'return_basis': 'provisional_provider_adjusted_endpoint_ratio', 'economic_return_certified': False,
            'knowledge_time_certified': False, 'same_close_operational_availability_certified': False,
            'model_ready': False, 'estimator_fit_performed': False}
        payoff = RetPayoffPanel(table, monthly, provenance, _fingerprint(table, monthly, provenance))
        verify_ret_payoffs(payoff); result[signal] = payoff
    need(len(result) == 48, 'RET must retain all48 original signal states')
    return result


def forecast_footer_plan(root, entries):
    plan, total = [], 0
    for entry in entries:
        files = []
        for item in entry['files']:
            path = authenticated(root, item['predictions_pin']); pf = pq.ParquetFile(path)
            columns = [c for c in SAFE_COLUMNS if c in pf.schema_arrow.names]
            need(set(COHORT_COLUMNS) <= set(columns) and set(PROBABILITIES) <= set(pf.schema_arrow.names)
                 and pf.metadata.num_rows <= (12000 if item['outer_year'] is not None else 45000),
                 'Saved forecast schema/count changed')
            amount = _projected_bytes(path, columns) + _projected_bytes(path, PROBABILITIES)
            files.append({**item, 'metadata_columns': columns, 'path': str(path), 'projected_compressed_bytes': amount})
            total += amount
        plan.append({'arm': entry['arm'], 'files': files})
    need(tuple(x['arm'] for x in plan) == ARMS, 'Saved forecast arm order changed')
    return plan, total


def load_forecasts(plan, windows):
    result, common = {}, None
    for entry in plan:
        frames = []
        for file in entry['files']:
            pf = pq.ParquetFile(file['path'])
            meta = pf.read(columns=file['metadata_columns']).to_pandas()
            need(meta.signal_date.map(lambda x: isinstance(x, str)).all(), 'Forecast dates must retain canonical strings')
            for year in YEARS:
                part = meta.loc[meta.outcome_month.str.startswith(str(year) + '-')].reset_index(drop=True)
                if not part.empty: validate_forecast_metadata(part, year, windows)
            need(meta.outcome_month.str.slice(0, 4).isin([str(y) for y in YEARS]).all(), 'Forecast financial dates leave2020–23')
            probabilities = pf.read(columns=list(PROBABILITIES)).to_pandas()
            need(len(probabilities) == len(meta), 'Probability rows changed after metadata guard')
            for name in PROBABILITIES: meta[name] = probabilities[name].to_numpy()
            frames.append(meta)
        frame = pd.concat(frames, ignore_index=True)
        need(len(frame) == 43655 and int(frame.prediction_eligible.sum()) == 42145
             and frame.outcome_month.nunique() == 48 and not frame.duplicated(['source_security_id', 'signal_date']).any(),
             'Original forecast cohort/count changed')
        cohort = frame.loc[:, list(COHORT_COLUMNS)]
        if common is None: common = cohort
        else: pd.testing.assert_frame_equal(cohort, common, check_dtype=False, check_exact=True)
        result[entry['arm']] = frame
    return result


def _context(root, identity, base):
    tuned = read_json(authenticated(root, identity['tuned_backtest_identity_pin']))
    return execution_context(root, tuned, base)


def _save_payoffs(directory, payoffs, artifacts):
    for signal, payoff in payoffs.items():
        rel = 'payoffs/' + signal
        for name, table in (('payoffs', payoff.table), ('monthly', payoff.monthly)):
            path = directory / rel / (name + '.parquet'); path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, path); artifacts[rel + '/' + name + '.parquet'] = _pin(directory, path)
        path = directory / rel / 'payoff_state.json'
        write_json(path, {'provenance': payoff.provenance, 'fingerprint': payoff.fingerprint})
        artifacts[rel + '/payoff_state.json'] = _pin(directory, path)


def preflight(root, run_directory):
    root, run = Path(root).resolve(), Path(run_directory).resolve()
    identity, root_audit, cfg, base = load_run(root, run); directory = run / 'preflight'
    need(not directory.exists() and root_audit['status'] == 'PREPARED', 'Preserve existing/interrupted RET preflight')
    started = time.monotonic()
    with exclusive(run), threadpool_limits(limits=1):
        resources(run, cfg, started, LIMITS['preflight_wall_seconds'], launch=True); directory.mkdir()
        write_json(directory / 'audit.json', {'status': 'STARTED', 'phase': PHASE, 'parent_identity_sha256': canonical_sha256(identity)})
        try:
            context = _context(root, identity, base)
            tp, target_bytes = target_footer_plan(root, identity['annual_sources'])
            fp, forecast_bytes = forecast_footer_plan(root, identity['forecast_entries'])
            dp, metadata_bytes, financial_bytes = daily_footer_plan(root, identity)
            reserve_reads(run, 'preflight', target_bytes + forecast_bytes + metadata_bytes)
            means = load_monthly_class_means(root, tp)
            payoffs = build_shared_payoffs(means, list(context['windows'].signal_date))
            forecasts = load_forecasts(fp, context['windows'])
            _, metadata, old_plan = _price_plan(root, root / identity['source_reference_directory'],
                identity['execution_reference_root_identity_sha256'], context['source_pins'])
            artifacts, summaries, validations = {}, [], []
            _save_payoffs(directory, payoffs, artifacts)
            _save_table(directory, 'source_month_class_means.parquet', pd.DataFrame([r for m in sorted(means['monthly']) for r in means['monthly'][m]]), artifacts)
            for arm in ARMS:
                result = select_ret_walkforward_basket(forecasts[arm], payoffs=payoffs,
                    sessions=context['clock']['selection_sessions'], configuration=base, outcome_years=YEARS)
                need(len(result.monthly) == 48 and len(result.ledger) == 43655, 'RET selection lost original rows/months')
                # Recompute from the same signal-only inputs to detect accidental mutation before persistence.
                expected = select_ret_walkforward_basket(forecasts[arm], payoffs=payoffs,
                    sessions=context['clock']['selection_sessions'], configuration=base, outcome_years=YEARS)
                for name in ('ledger', 'selections', 'rebalances', 'monthly'):
                    pd.testing.assert_frame_equal(getattr(result, name), getattr(expected, name), check_exact=True)
                gap = preflight_selected_horizons(result.selections, result.rebalances, metadata,
                    context['sessions'], events=context['event_registry']['events'], historical_retirements=context['historical_retirements'])
                for name, frame in (('original_ledger', result.ledger), ('selected', result.selections),
                    ('rebalances', result.rebalances), ('monthly', result.monthly)):
                    _save_table(directory, f'selections/{arm}/{name}.parquet', frame, artifacts)
                for name, frame in gap.items():
                    if name != 'summary': _save_table(directory, f'coverage/{arm}/{name}.parquet', frame, artifacts)
                withheld = int(result.monthly.status.ne('available').sum())
                summaries.append({'arm': arm, 'basket': BASKET, **gap['summary'], 'withheld_selection_months': withheld})
                validations.append({'arm': arm, 'original_rows': len(result.ledger), 'signal_months': 48,
                    'selected_rows': len(result.selections), 'shared_payoff_vectors': 48, 'signal_only_exact_replay': True})
                resources(run, cfg, started, LIMITS['preflight_wall_seconds'])
            phase = {**{k: context[k] for k in ('clock', 'event_registry', 'historical_retirements', 'identity_handover_registry',
                'identity_handover_registry_pin', 'identity_handover_approval', 'source_pins')},
                'schema': 'dmv2-ret-backtest-preflight/v1', 'phase': PHASE,
                'parent_identity_sha256': canonical_sha256(identity),
                'execution_reference_root_identity_sha256': identity['execution_reference_root_identity_sha256'],
                'configuration_sha256': canonical_sha256(cfg), 'source_reference_directory': identity['source_reference_directory'],
                'annual_sources': identity['annual_sources'], 'forecast_entries': identity['forecast_entries'],
                'target_plan': tp, 'forecast_plan': fp,
                'read_plan': {'annuals': dp, 'target_bytes': target_bytes, 'forecast_bytes': forecast_bytes,
                    'daily_metadata_bytes': metadata_bytes, 'daily_financial_bytes': financial_bytes,
                    'preflight_reserved_bytes': target_bytes + forecast_bytes + metadata_bytes, 'candidate_rows': 947517},
                'basket': BASKET, 'shared_payoff_vectors': 48, 'payoff_month_class_cells': 57600,
                'source_outcome_months': 167, 'new_estimator_fits': 0, 'new_preprocessor_fits': 0,
                'new_inference_calls': 0, 'performance_computed': False, 'financial_2024_plus_read': False,
                'implementation_revision': IMPLEMENTATION_REVISION}
            write_json(directory / 'build_identity.json', phase)
            artifacts['build_identity.json'] = _pin(directory, directory / 'build_identity.json')
            _save_table(directory, 'summary.parquet', pd.DataFrame(summaries), artifacts)
            _save_table(directory, 'selection_validation.parquet', pd.DataFrame(validations), artifacts)
            write_json(directory / 'audit.json', {'status': 'COMPLETE', 'phase': PHASE,
                'build_identity_sha256': canonical_sha256(identity), 'phase_identity_sha256': canonical_sha256(phase),
                'artifacts': artifacts, 'summary': summaries, 'performance_computed': False,
                'source_pins_unchanged': True, 'execution_seconds': time.monotonic() - started}, replace=True)
            return directory
        except BaseException as error:
            write_json(directory / 'failure.json', {'status': 'FAILED_PRESERVED_NO_RETRY',
                'error_type': type(error).__name__, 'message': str(error)}); raise


def _saved_table_plan(directory, artifacts):
    paths, amount = {}, 0
    for arm in ARMS:
        for name in ('original_ledger', 'selected', 'rebalances', 'monthly'):
            rel = f'selections/{arm}/{name}.parquet'; path = authenticated(directory, artifacts[rel])
            amount += _projected_bytes(path, pq.ParquetFile(path).schema_arrow.names); paths[(arm, name)] = path
    return paths, amount


def backtest(root, run_directory, *, allow_withheld_paths=False):
    root, run = Path(root).resolve(), Path(run_directory).resolve()
    identity, root_audit, cfg, base = load_run(root, run); directory, pre = run / 'backtest', run / 'preflight'
    need(type(allow_withheld_paths) is bool and not directory.exists(), 'Preserve existing RET numerical attempts')
    audit, phase = read_json(pre / 'audit.json'), read_json(pre / 'build_identity.json')
    need(audit['status'] == 'COMPLETE' and audit['build_identity_sha256'] == canonical_sha256(identity)
         and audit['phase_identity_sha256'] == canonical_sha256(phase), 'Exact completed RET preflight required')
    for p in audit['artifacts'].values(): authenticated(pre, p)
    unresolved = any(not row['ready_for_accounting'] or row['withheld_selection_months'] for row in audit['summary'])
    need(not unresolved or (allow_withheld_paths and cfg['allow_withheld_paths']), 'Explicit fail-closed coverage review required')
    started, peak = time.monotonic(), 0
    with exclusive(run), threadpool_limits(limits=1):
        resources(run, cfg, started, LIMITS['backtest_wall_seconds'], launch=True); directory.mkdir()
        write_json(directory / 'audit.json', {'status': 'STARTED', 'phase': PHASE, 'build_identity_sha256': canonical_sha256(identity)})
        try:
            paths, selection_bytes = _saved_table_plan(pre, audit['artifacts'])
            dp, metadata_bytes, financial_bytes = daily_footer_plan(root, identity)
            reserve_reads(run, 'backtest', metadata_bytes + financial_bytes + selection_bytes)
            context = _context(root, identity, base)
            _, metadata, _ = _price_plan(root, root / identity['source_reference_directory'],
                identity['execution_reference_root_identity_sha256'], context['source_pins'])
            prices = pd.concat([pq.ParquetFile(p['path']).read(columns=list(ACTION_PRICE_COLUMNS)).to_pandas() for p in dp], ignore_index=True)
            need(prices[['trading_date', 'intrinio_security_id']].equals(metadata[['trading_date', 'intrinio_security_id']]),
                 'Daily quote keys changed after guarded metadata')
            run_phase = {**phase, 'schema': 'dmv2-ret-backtest-phase/v1',
                'preflight_audit_pin': _pin(root, pre / 'audit.json'), 'preflight_before_performance': True,
                'allow_withheld_paths_after_explicit_review': allow_withheld_paths, 'performance_computed': True,
                'code_pins': identity['code_pins'], 'runtime': identity['runtime'],
                'selection_artifacts': {k: _pin(root, authenticated(pre, v)) for k, v in audit['artifacts'].items() if k.startswith('selections/')}}
            write_json(directory / 'build_identity.json', run_phase)
            artifacts = {'build_identity.json': _pin(directory, directory / 'build_identity.json')}
            comparison, annual, failures, actions, orders, coverage = [], [], [], [], [], []
            for arm in ARMS:
                tables = {name: pq.read_table(paths[(arm, name)]).to_pandas() for name in ('original_ledger', 'selected', 'rebalances', 'monthly')}
                coverage.append(tables['monthly'].assign(arm=arm, basket=BASKET))
                for variant, stop in base['backtest']['variants'].items():
                    for cost in base['backtest']['cost_bps']:
                        observed = resources(run, cfg, started, LIMITS['backtest_wall_seconds']); peak = max(peak, observed['process_tree_rss_bytes'])
                        result = run_walkforward_accounting_v2(tables['selected'], prices, context['sessions'],
                            rebalances=tables['rebalances'], terminal_date='2023-12-29', events=context['event_registry']['events'],
                            historical_retirements=context['historical_retirements'], identity_handovers=context['identity_handovers'],
                            policy=DailyAccountingPolicy(cost_bps=cost, borrow_annual_bps=0, required_names_per_side=None, asset_stop_fraction=stop))
                        row, monthly = summarize_daily(result, arm=arm, basket=BASKET, variant=variant, cost_bps=cost, expected_sessions=1006)
                        need(len(monthly) == 48, 'RET replay lost a monthly outcome')
                        row.update(outcome_start='2020-01-01', outcome_end='2023-12-31', outcome_months=48,
                            terminal_date='2023-12-29', annual_liquidation=False, annual_model_fits=4)
                        comparison.append(row); annual.append(yearly_breakdown(result, arm=arm, basket=BASKET, variant=variant, cost_bps=cost))
                        rel = f'arms/{arm}/{BASKET}/{variant}/cost_{cost:02d}bps'
                        for name, frame in result.tables.items(): _save_table(directory, rel + '/' + name + '.parquet', frame, artifacts)
                        _save_table(directory, rel + '/monthly.parquet', monthly, artifacts)
                        if not result.failures.empty: failures.append(result.failures.assign(arm=arm, basket=BASKET, variant=variant, cost_bps=cost))
                        if not result.action_events.empty: actions.append(result.action_events.assign(arm=arm, basket=BASKET, variant=variant, cost_bps=cost))
                        if not result.orders.empty: orders.append(result.orders.assign(arm=arm, basket=BASKET, variant=variant, cost_bps=cost))
                        observed = resources(run, cfg, started, LIMITS['backtest_wall_seconds']); peak = max(peak, observed['process_tree_rss_bytes'])
                        print(json.dumps({'phase': PHASE, 'completed_scenarios': len(comparison), 'expected_scenarios': 40,
                            'arm': arm, 'variant': variant, 'cost_bps': cost, 'status': result.status}), flush=True)
                        del result
            need(len(comparison) == 40, 'Incomplete forty-scenario RET grid')
            reports = {'comparison': pd.DataFrame(comparison), 'yearly': pd.concat(annual, ignore_index=True),
                'selection_coverage': pd.concat(coverage, ignore_index=True),
                'failures': pd.concat(failures, ignore_index=True) if failures else pd.DataFrame(columns=['date', 'reason', 'arm', 'basket', 'variant', 'cost_bps']),
                'action_coverage': pd.concat(actions, ignore_index=True) if actions else pd.DataFrame(columns=['event_id', 'arm', 'basket', 'variant', 'cost_bps']),
                'order_coverage': pd.concat(orders, ignore_index=True) if orders else pd.DataFrame(columns=['status', 'arm', 'basket', 'variant', 'cost_bps'])}
            for name, frame in reports.items(): _save_table(directory, name + '.parquet', frame, artifacts)
            for p in identity['source_pins']: authenticated(root, p)
            write_json(directory / 'audit.json', {'status': 'COMPLETE', 'phase': PHASE, 'implementation_revision': IMPLEMENTATION_REVISION,
                'build_identity_sha256': canonical_sha256(identity), 'phase_identity_sha256': canonical_sha256(run_phase),
                'artifacts': artifacts, 'scenarios': 40, 'sessions': 1006, 'outcome_months': 48,
                'complete_proxy_paths': sum(x['complete_proxy_path'] for x in comparison),
                'failed_closed_paths': sum(not x['complete_proxy_path'] for x in comparison),
                'models_fitted': False, 'preprocessing_fitted': False, 'new_inference_calls': 0, 'financial_2024_plus_read': False,
                'source_identity_promoted': False, 'source_pins_unchanged': True, 'old_runs_and_pointers_changed': False,
                'preflight_before_performance': True, 'peak_rss_bytes': peak, 'execution_seconds': time.monotonic() - started,
                'RET': 'five_saved_arms_shared_corrected_120_month_payoffs', 'annual_liquidation': False}, replace=True)
            write_json(run / 'audit.json', {**root_audit, 'status': 'COMPLETE_PENDING_INDEPENDENT_VERIFICATION',
                'backtest_audit_pin': _pin(root, directory / 'audit.json'), 'financial_values_decoded': True}, replace=True)
            return directory
        except BaseException as error:
            write_json(directory / 'failure.json', {'status': 'FAILED_PRESERVED_NO_RETRY',
                'error_type': type(error).__name__, 'message': str(error)}); raise
