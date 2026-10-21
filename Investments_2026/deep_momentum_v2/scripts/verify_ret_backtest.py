#!/usr/bin/env python3
"""Independent, zero-fit RET and daily-accounting audit.

Source class means, rolling payoffs, probability-weighted scores and selections
are reconstructed here without importing production target/RET/selection code.
Only the previously reviewed independent cash/units equations are reused.
Every Parquet conversion follows immutable pins, a full-file read reservation,
and metadata/date checks. No estimator or serialized model is opened.
"""
from __future__ import annotations

import argparse
import base64
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import itertools
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path
import platform
import resource
import time

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

BASE_SHA = '16c3651289258bed24a83d500f6b8589fac3bcb4d358e516fe981acec527a794'
SOURCE_IDENTITY = 'sha256:559ba2f7fbe187f192b40af2ab8fbe0a4fd70a1b843e2cbd0503f7e35729fcae'
ARMS = ('rf_paper15', 'xgb_paper15', 'rf_extended', 'xgb_extended', 'rf_extended_tuned')
BASKET = 'ret'
VARIANTS = ('unstopped', 'asset_stop_10pct')
FEES = (0, 5, 10, 25)
EXCLUDED = {'sec_gjP4xM', 'sec_zmYNnz'}
KEYS = ['source_security_id', 'signal_date']
PROBABILITIES = [f'probability_{k}' for k in range(1, 11)]
FORECAST_META = [*KEYS, 'intrinio_security_id', 'prediction_eligible',
                 'paper_prediction_eligible', 'identity_eligible', 'requested',
                 'target_available', 'outcome_month', 'outcome_start', 'outcome_end']
TARGET_META = [*KEYS, 'outcome_month', 'outcome_start', 'outcome_end',
               'identity_eligible', 'requested', 'paper_prediction_eligible',
               'prediction_eligible', 'outcome_requested', 'target_available']
LEDGERS = ('daily', 'trades', 'holdings', 'failures', 'stops', 'action_events',
           'market_trades', 'orders', 'identity_handovers')
MONTHLY_FIELDS = ('outcome_month', 'outcome_end', 'class_id', 'original_member_count',
                  'prediction_eligible_count', 'member_count', 'class_mean_return', 'status')
READ_LIMIT = 1024**3


class VerificationError(RuntimeError):
    pass


def need(condition, message):
    if not bool(condition):
        raise VerificationError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            value.update(block)
    return value.hexdigest()


def canonical(value):
    return 'sha256:' + hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def json_file(path):
    path = Path(path)
    need(path.is_file() and path.stat().st_size <= 16 * 1024**2, 'Missing or oversized JSON')
    return json.loads(path.read_text())


def local(root, path):
    root = Path(root).resolve()
    result = (root / path).resolve()
    need(result.is_relative_to(root), 'Pinned path leaves its declared root')
    return result


def pinned(root, item):
    need(isinstance(item, dict) and {'path', 'sha256', 'size_bytes'} <= set(item), 'Complete immutable pin required')
    path = local(root, item['path'])
    need(path.is_file() and path.stat().st_size == item['size_bytes']
         and digest(path) == item['sha256'], 'Pinned artifact changed: ' + str(path))
    return path


def number(actual, expected, message, *, tolerance=3e-13):
    if expected is None or pd.isna(expected):
        need(actual is None or pd.isna(actual), message + ': expected missing')
    else:
        need(isinstance(actual, Real) and not isinstance(actual, (bool, np.bool_))
             and math.isfinite(float(actual))
             and abs(float(actual) - float(expected)) <= tolerance * max(1., abs(float(expected))), message)


def month_step(month, offset):
    need(isinstance(month, str) and len(month) == 7 and month[4] == '-', 'Invalid calendar month')
    year, index = map(int, month.split('-'))
    need(1 <= index <= 12, 'Invalid calendar month')
    value = year * 12 + index - 1 + offset
    return f'{value // 12:04d}-{value % 12 + 1:02d}'


def trailing_months(signal):
    need(isinstance(signal, str) and '2019-12-01' <= signal <= '2023-11-30', 'RET signal outside 48 approved months')
    last = signal[:7]
    return [month_step(last, offset) for offset in range(-119, 1)]


def exact_frame(actual, expected, message):
    try:
        pd.testing.assert_frame_equal(actual, expected, check_exact=True,
            check_dtype=True, check_names=True, check_like=False)
    except (AssertionError, TypeError, ValueError) as error:
        raise VerificationError(message) from error


def load_equations(root):
    path = Path(root) / 'scripts/verify_walkforward_backtest_v2.py'
    need(digest(path) == BASE_SHA, 'Previously reviewed independent cash/units equations changed')
    spec = importlib.util.spec_from_file_location('_ret_independent_cash_units', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ProjectionBudget:
    """A full compressed file is charged for every projection, before reading."""

    def __init__(self, *, prior=0, reserved=0, limit=READ_LIMIT):
        need(type(prior) is int and type(reserved) is int and 0 <= prior <= prior + reserved <= limit,
             'Invalid independent decode reservation')
        self.prior, self.reserved, self.limit, self.used = prior, reserved, limit, 0

    def arrow(self, path, columns=None):
        path = Path(path)
        charge = path.stat().st_size
        need(self.used + charge <= self.reserved
             and self.prior + self.used + charge <= self.limit,
             'Independent decode exceeded its prior cumulative reservation')
        self.used += charge
        return pq.ParquetFile(path).read(columns=columns)

    def read(self, path, columns=None):
        return self.arrow(path, columns).to_pandas()


def reserve_verification(run, amount):
    path = Path(run) / 'read_usage.json'
    usage = json_file(path)
    entries = usage.get('reservations')
    need(type(amount) is int and amount >= 0 and isinstance(entries, list)
         and all(isinstance(row, dict) for row in entries), 'Invalid cumulative read ledger')
    need(not any(row.get('phase') == 'independent_verification' for row in entries),
         'Preserve previous verification attempt; no automatic second decode')
    need(all(type(row.get('projected_compressed_bytes')) is int
             and row['projected_compressed_bytes'] >= 0 for row in entries), 'Invalid earlier read reservation')
    prior = sum(row['projected_compressed_bytes'] for row in entries)
    need(usage.get('limit_bytes') == READ_LIMIT
         and usage.get('total_projected_compressed_bytes') == prior
         and prior + amount <= READ_LIMIT, 'Cumulative independent read limit exceeded')
    entries.append({'phase': 'independent_verification', 'projected_compressed_bytes': amount,
                    'reserved_at': datetime.now(timezone.utc).isoformat()})
    usage['total_projected_compressed_bytes'] = prior + amount
    temporary = path.with_suffix('.independent.partial')
    with temporary.open('x') as stream:
        json.dump(usage, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)
    return prior


def guarded_table(v, budget, path, *, maximum_rows=2_000_000):
    """Validate every date key before converting a saved financial table."""
    pf = pq.ParquetFile(path)
    need(pf.metadata.num_rows <= maximum_rows, 'Saved financial table exceeds row limit')
    dates = [field for field in ('date', 'trading_date', 'signal_date', 'held_signal_date',
                                'execution_date') if field in pf.schema_arrow.names]
    need(dates or pf.metadata.num_rows == 0, 'Saved financial table lacks date keys')
    meta = budget.read(path, dates)
    for field in dates:
        low = '2019-12-01' if field in ('signal_date', 'held_signal_date') else '2020-01-01'
        need(meta[field].dropna().map(lambda value: low <= v.iso(value) <= '2023-12-31').all(),
             'Protected financial period rejected before numeric conversion')
    return budget.read(path)


def source_month_fingerprint(table):
    """Independent exact Arrow logical fingerprint, matching the sealed state."""
    schema = table.schema.remove_metadata()
    rebuilt = pa.Table.from_pylist(table.to_pylist(), schema=schema)
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, schema) as stream:
        stream.write_table(rebuilt)
    return 'sha256:' + hashlib.sha256(sink.getvalue()).hexdigest()


def class_means(table, window):
    """Compute means from observed true classes; retain original denominators."""
    need(set((*TARGET_META, 'target_class', 'label_return')) <= set(table.column_names),
         'Source target class/return metadata missing')
    rows = table.to_pylist()
    need(bool(rows), 'Empty observed source month')
    need(len({(row['source_security_id'], row['signal_date']) for row in rows}) == len(rows),
         'Duplicate source target key')
    groups = {k: [] for k in range(1, 11)}
    eligible = 0
    for row in rows:
        need(row['signal_date'] == window['signal_date']
             and all(row[field] == window[field] for field in ('outcome_month', 'outcome_start', 'outcome_end')),
             'Target row differs from independent completion window')
        for field in ('identity_eligible', 'requested', 'paper_prediction_eligible',
                      'prediction_eligible', 'target_available'):
            need(isinstance(row[field], (bool, np.bool_)), 'Source admission or availability is not Boolean')
        need(row['prediction_eligible'] == (row['identity_eligible'] and row['paper_prediction_eligible'])
             and (not row['prediction_eligible'] or row['identity_eligible'] and row['requested'])
             and (not row['target_available'] or row['prediction_eligible']),
             'Observed outcomes changed signal-time target population')
        eligible += int(row['prediction_eligible'])
        if row['target_available']:
            cls, value = row['target_class'], row['label_return']
            need(isinstance(cls, Integral) and not isinstance(cls, (bool, np.bool_)) and cls in groups
                 and isinstance(value, Real) and not isinstance(value, (bool, np.bool_))
                 and math.isfinite(value) and value >= -1., 'Invalid observed true-class payoff')
            groups[int(cls)].append(float(value))
    output = []
    for cls, values in groups.items():
        output.append({'outcome_month': window['outcome_month'], 'outcome_end': window['outcome_end'],
            'class_id': cls, 'original_member_count': len(rows), 'prediction_eligible_count': eligible,
            'member_count': len(values),
            'class_mean_return': math.fsum(value / len(values) for value in values) if values else None,
            'status': 'available' if values else 'empty_realized_class'})
    return output


def independent_payoff(months, signal, *, cutoff=None):
    """Exactly 120 completed months with every class; never pooled stock means."""
    cutoff = signal if cutoff is None else cutoff
    need(cutoff <= signal and cutoff[:7] >= '2010-01', 'RET cutoff exceeds its signal')
    expected = trailing_months(signal)
    grid = []
    for month in expected:
        for cls in range(1, 11):
            row = months.get((month, cls))
            need(row is not None and row['outcome_month'] == month and row['class_id'] == cls,
                 'Missing true-class calendar cell')
            need(row['status'] == 'available' and row['member_count'] > 0
                 and row['outcome_end'] <= cutoff
                 and isinstance(row['class_mean_return'], Real)
                 and math.isfinite(row['class_mean_return']), 'Incomplete or uncompleted true-class payoff grid')
            grid.append(row)
    payoff = np.asarray([math.fsum(months[(month, cls)]['class_mean_return'] / 120.
                                  for month in expected) for cls in range(1, 11)], dtype=float)
    return payoff, grid


def independent_scores(probabilities, payoffs):
    p = np.asarray(probabilities, dtype=float)
    mu = np.asarray(payoffs, dtype=float)
    need(p.ndim == 2 and p.shape[1] == 10 and mu.shape == (10,), 'Ten canonical classes required')
    need(np.isfinite(p).all() and (p >= 0).all() and np.isfinite(mu).all()
         and np.all(np.abs(p.sum(axis=1) - 1.) <= 1e-8), 'Invalid canonical probability simplex')
    # Deliberately use explicit scalar sums, independent of the production BLAS
    # matrix multiply. Agreement tolerance covers last-bit summation order only.
    return np.asarray([math.fsum(float(prob) * float(payoff) for prob, payoff in zip(row, mu, strict=True))
                       for row in p], dtype=float)


def independent_rank(group, scores):
    """Stable disjoint global RET ranks, with no mode/sign/future-outcome gate."""
    need(len(group) == len(scores) and not group.duplicated(KEYS).any(), 'RET rank key/count mismatch')
    candidate = group.loc[group.prediction_eligible & ~group.intrinio_security_id.isin(EXCLUDED),
                          ['source_security_id', 'intrinio_security_id']].copy()
    score = pd.Series(scores, index=group.index)
    need(np.isfinite(score.loc[candidate.index].to_numpy()).all(), 'Eligible RET score missing')
    candidate['ret_score'] = score.loc[candidate.index]
    n = min(10, len(candidate) // 2)
    longs = candidate.sort_values(['ret_score', 'source_security_id'], ascending=[False, True], kind='stable').head(n)
    shorts = candidate.loc[~candidate.source_security_id.isin(longs.source_security_id)].sort_values(
        ['ret_score', 'source_security_id'], ascending=[True, True], kind='stable').head(n)
    need(len(longs) == len(shorts) == n and not set(longs.source_security_id) & set(shorts.source_security_id),
         'RET global sides overlap or differ in breadth')
    return longs, shorts, n


def verify_forecast_metadata(frame, windows, *, year=None):
    need(set(FORECAST_META) <= set(frame) and not frame.duplicated(KEYS).any(), 'Original forecast metadata missing/duplicate')
    expected = {row.signal_date: row for row in windows.itertuples() if year is None or row.outcome_month[:4] == str(year)}
    need(set(frame.signal_date) == set(expected), 'Forecast month grid escaped its approved bounds')
    for field in ('prediction_eligible', 'paper_prediction_eligible', 'identity_eligible', 'requested', 'target_available'):
        need(frame[field].map(lambda value: isinstance(value, (bool, np.bool_))).all(), 'Forecast admission is not strict Boolean')
    need(frame.prediction_eligible.equals(frame.paper_prediction_eligible)
         and (~frame.prediction_eligible | (frame.identity_eligible & frame.requested)).all(),
         'Expanded features changed shared Paper15 signal-time admission')
    for row in frame.itertuples(index=False):
        window = expected[row.signal_date]
        need(isinstance(row.source_security_id, str) and bool(row.source_security_id.strip())
             and row.outcome_month == window.outcome_month and row.outcome_start == window.outcome_start
             and row.outcome_end == window.outcome_end and row.signal_date < row.outcome_start <= row.outcome_end,
             'Forecast metadata has invalid identity or next-month chronology')
        if row.prediction_eligible:
            need(isinstance(row.intrinio_security_id, str) and bool(row.intrinio_security_id.strip()),
                 'Eligible forecast lacks native identity')
    admitted = frame.loc[frame.prediction_eligible]
    need(not admitted.duplicated(['signal_date', 'intrinio_security_id']).any(), 'Eligible same-signal identity collision')


# The remaining persistence adapter is kept below the independent arithmetic so
# its code-only tests can exercise leakage/tie/budget cases without local data.

LIMITS = {'prepare_wall_seconds': 300, 'preflight_wall_seconds': 900,
    'backtest_wall_seconds': 1200, 'verification_wall_seconds': 3600,
    'rss_stop_bytes': 4*1024**3, 'rss_warning_bytes': 3*1024**3,
    'minimum_available_memory_bytes': 7*1024**3, 'normal_pressure_dispatch': 1,
    'cumulative_compressed_read_bytes': READ_LIMIT, 'output_bytes': 512*1024**2,
    'workers': 1, 'numeric_threads': 1}
RET_DEFINITION = {'policy_id': 'realized_class_equal_month_120_v1', 'class_ids': list(range(1, 11)),
    'completed_calendar_months': 120, 'window_end': 'signal_calendar_month',
    'monthly_class_mean': 'equal_weight_observed_realized_class_returns',
    'time_weighting': 'equal_calendar_month_weights',
    'incomplete_month_class_grid': 'withhold_entire_payoff_vector',
    'probability_sum_tolerance': 1e-8, 'probability_renormalization': False,
    'absent_class_zero_padding': False}
PAYOFF_SCHEMA = pa.schema([('class_id', pa.int8()), ('payoff', pa.float64()), ('status', pa.string())])
GRID_SCHEMA = pa.schema([('outcome_month', pa.string()), ('outcome_end', pa.string()),
    ('class_id', pa.int8()), ('original_member_count', pa.int64()),
    ('prediction_eligible_count', pa.int64()), ('member_count', pa.int64()),
    ('class_mean_return', pa.float64()), ('status', pa.string())])


def authenticate(root, run, v):
    """Bind every new input to the independently verified original authorities."""
    from importlib.metadata import version
    identity, audit = json_file(run/'build_identity.json'), json_file(run/'audit.json')
    identity_sha = canonical(identity)
    need(identity.get('schema') == 'dmv2-ret-backtest-root/v1'
         and identity.get('project_root') == str(root) and identity.get('run_root') == str(run)
         and audit.get('status') == 'COMPLETE_PENDING_INDEPENDENT_VERIFICATION'
         and audit.get('build_identity_sha256') == identity_sha, 'Completed fresh RET identity missing')
    cfg = json_file(pinned(root, identity['configuration_pin']))
    need(cfg == identity['configuration'] and cfg.get('schema') == 'dmv2-ret-backtest-execution/v1'
         and cfg.get('protocol_id') == 'dmv2-ret-selection-backtest-20261020-v1'
         and cfg.get('status') == 'FROZEN_EXECUTION_AUTHORIZED' and cfg.get('execution_authorized') is True
         and cfg.get('arm_ids') == list(ARMS) and cfg.get('scenarios') == 40
         and cfg.get('allow_withheld_paths') is True and cfg.get('resource_limits') == LIMITS
         and cfg.get('financial_2024_plus_authorized') is False and cfg.get('cloud_operations') is False
         and all(type(cfg.get(field)) is int and cfg[field] == 0 for field in
                 ('new_estimator_fits', 'new_preprocessor_fits', 'new_inference_calls')),
         'Separate forty-scenario zero-fit authority changed')
    need(cfg['selection'] == {'all_equal_scores': 'retain_existing_stable_id_tie_policy_report_zero_spread',
        'basket': 'ret', 'long_ties': 'RET_desc_source_id_asc',
        'names_per_side': 'min(10,floor(trade_eligible_count/2))',
        'no_future_price_or_label_availability_filter': True,
        'paper_breadth_difference': 'User selected ten per side; paper uses top/bottom ten percent.',
        'ranking': 'all_trade_eligible_predictions_by_full_probability_weighted_payoff_NO_mode_class_filter',
        'short_ties': 'RET_asc_source_id_asc_excluding_reserved_longs', 'sign_gate': False,
        'trade_restrictions': 'unchanged_DF_RDC'}, 'RET ranking/sign/tie policy changed')
    need(cfg['payoffs'] == {'completed_calendar_months': 120, 'incomplete_grid': 'withhold_entire_vector',
        'monthly_class_means': 'equal_weight_observed_true_class_returns', 'old_full_target_RET_states_reused': False,
        'policy': 'realized_class_equal_month_120_v1', 'sources': 'exact_corrected_targets_used_by_saved_models',
        'time_weighting': 'equal_months', 'union_outcomes': ['2010-01', '2023-11']},
        'Corrected causal RET payoff policy changed')
    need(cfg['execution'] == {'annual_liquidation': False, 'asset_stop_choices': [None, .1],
        'borrow_and_funding_bps': 0, 'cost_bps': list(FEES), 'long_weight': .5,
        'missing_held_marks': 'retain_exposure_withhold_full_path', 'months': 48,
        'outcome_years': [2020, 2021, 2022, 2023], 'same_original_actions_retirements_PENN_exception': True,
        'sessions': 1006, 'short_weight': -.5, 'stop_cash': 'cash_until_next_monthly_rebalance_no_replacement',
        'weights': 'equal_within_side_at_monthly_next_open'}, 'Frozen RET execution policy changed')
    expected_runtime = {'python': platform.python_version(), **{name: version(name) for name in
        ('numpy', 'pandas', 'pyarrow', 'exchange-calendars', 'psutil', 'threadpoolctl')}}
    need(identity['runtime'] == expected_runtime, 'Frozen numerical runtime changed')
    need(all(identity.get(field) == 0 for field in ('new_estimator_fits', 'new_preprocessor_fits', 'new_inference_calls'))
         and identity.get('financial_2024_plus_read') is False
         and identity.get('old_runs_and_pointers_changed') is False, 'Root compute/period scope changed')
    for item in [*identity['code_pins'], *identity['source_pins']]: pinned(root, item)
    need(any(pinned(root, item) == Path(__file__).resolve() for item in identity['code_pins']),
         'Independent verifier is not part of the sealed execution code')
    tuned = json_file(pinned(root, cfg['tuned_backtest_identity_pin']))
    tuned_audit = json_file(pinned(root, cfg['tuned_backtest_audit_pin']))
    tuned_receipt = json_file(pinned(root, cfg['tuned_backtest_verification_pin']))
    tuned_bt_audit_path = pinned(root, tuned_audit['backtest_audit_pin'])
    tuned_bt_audit = json_file(tuned_bt_audit_path)
    tuned_phase = json_file(tuned_bt_audit_path.parent/'build_identity.json')
    need(tuned.get('schema') == 'dmv2-rf-tuned-backtest-root/v1'
         and tuned_receipt.get('schema') == 'dmv2-rf-tuned-backtest-independent-verification/v2'
         and tuned_receipt.get('status') == 'PASS' and tuned_receipt.get('scenarios') == 16
         and tuned_receipt.get('sessions') == 1006 and tuned_receipt.get('outcome_months') == 48
         and tuned_audit['build_identity_sha256'] == tuned_receipt['build_identity_sha256'] == canonical(tuned)
         and tuned_receipt['root_audit_sha256'] == cfg['tuned_backtest_audit_pin']['sha256']
         and tuned_receipt['backtest_audit_sha256'] == digest(tuned_bt_audit_path)
         and tuned_receipt['backtest_phase_identity_sha256'] == tuned_bt_audit['phase_identity_sha256'] == canonical(tuned_phase),
         'Exact independently verified tuned forecast/execution reference missing')
    for item in [*tuned['code_pins'], *tuned['source_pins']]: pinned(root, item)
    need(identity['tuned_backtest_identity_pin'] == cfg['tuned_backtest_identity_pin']
         and identity['source_reference_directory'] == tuned['source_reference_directory']
         and identity['execution_reference_root_identity_sha256'] == tuned['execution_reference_root_identity_sha256'] == SOURCE_IDENTITY
         and identity['execution_reference_phase_pin'] == tuned['execution_reference_phase_pin']
         and identity['execution_reference_audit_pin'] == tuned['execution_reference_audit_pin'],
         'New root detached from preserved corrected execution')
    reference_run = local(root, identity['source_reference_directory'])
    source_identity_path = pinned(root, cfg['source_corrected_identity_pin'])
    source_identity = json_file(source_identity_path)
    need(source_identity_path.parent == reference_run and canonical(source_identity) == SOURCE_IDENTITY,
         'Corrected target/model parent identity changed')
    old_phase_path = pinned(root, identity['execution_reference_phase_pin'])
    old_phase = json_file(old_phase_path)
    old_audit = json_file(pinned(root, identity['execution_reference_audit_pin']))
    need(old_audit['status'] == 'COMPLETE' and old_audit['scenarios'] == 64
         and old_audit['phase_identity_sha256'] == canonical(old_phase)
         and old_phase['parent_identity_sha256'] == old_audit['build_identity_sha256'] == SOURCE_IDENTITY,
         'Original independently checked continuous accounting reference changed')
    old_receipt = json_file(pinned(root, tuned['configuration']['source_reference_verification_pin']))
    need(old_receipt['status'] == 'PASS'
         and old_receipt['backtest_audit_sha256'] == identity['execution_reference_audit_pin']['sha256']
         and old_receipt['backtest_phase_identity_sha256'] == canonical(old_phase)
         and old_receipt['build_identity_sha256'] == SOURCE_IDENTITY,
         'Original corrected daily-accounting PASS detached')
    base = json_file(pinned(root, identity['base_configuration_pin']))
    need(base == identity['base_configuration'] and canonical(base) == old_phase['configuration_sha256']
         and base['backtest']['cost_bps'] == list(FEES)
         and base['backtest']['variants'] == {'unstopped': None, 'asset_stop_10pct': .1}
         and base['backtest']['top_k_per_side'] == 10
         and base['backtest']['long_weight'] == .5 and base['backtest']['short_weight'] == -.5
         and set(base['execution_restrictions']['native_ids']) == EXCLUDED
         and base['accounting']['continuous_2020_2023_ledger'] is True
         and base['accounting']['annual_fit_forced_liquidation'] is False
         and base['accounting']['terminal_session'] == '2023-12-29'
         and base['accounting']['borrow_annual_bps'] == base['accounting']['funding_annual_bps'] == 0,
         'Frozen stops/costs/actions/continuous ledger rules changed')
    model_receipt_pin = tuned['configuration']['source_model_verification_pin']
    model_receipt = json_file(pinned(root, model_receipt_pin))
    model_audit_path = reference_run/'models/audit.json'
    model_audit = json_file(model_audit_path)
    need(model_receipt['status'] == 'PASSED' and model_receipt['annual_fits_reconciled'] == 16
         and model_receipt['stitched_prediction_months'] == 48
         and model_receipt['build_identity_sha256'] == SOURCE_IDENTITY
         and model_receipt['models_audit_sha256'] == digest(model_audit_path)
         and model_receipt['model_identity_sha256'] == model_audit['model_identity_sha256'],
         'Original four-arm forecast verification detached')
    need([row['arm'] for row in identity['forecast_entries']] == list(ARMS), 'Exact five forecast arms required')
    for entry in identity['forecast_entries'][:-1]:
        arm_path = pinned(model_audit_path.parent, model_audit['arms'][entry['arm']])
        arm_audit = json_file(arm_path)
        expected = pinned(arm_path.parent, arm_audit['artifacts']['predictions'])
        need(len(entry['files']) == 1 and entry['files'][0]['outer_year'] is None
             and pinned(root, entry['files'][0]['predictions_pin']) == expected
             and entry['files'][0]['predictions_pin']['sha256'] == arm_audit['artifacts']['predictions']['sha256'],
             'RET substituted an unverified original forecast ledger')
    need(identity['forecast_entries'][-1]['files'] == tuned['forecast_entries'], 'RET substituted selected tuned forecasts')
    for entry in identity['forecast_entries']:
        for item in entry['files']: pinned(root, item['predictions_pin'])
    annuals = identity['annual_sources']
    need([row['year'] for row in annuals] == list(range(2010, 2024)), 'Exact fourteen corrected annual sources required')
    new_pins = {local(root, item['path']): item for item in identity['source_pins']}
    old_pins = {local(root, item['path']): item for item in tuned['source_pins']}
    for entry in annuals:
        year = entry['year']; ref = source_identity['annual_sources'][str(year)]
        directory = local(reference_run if ref.get('parent_reference') == 'THIS_BUILD' else root, ref['directory'])
        need(directory == local(root, entry['directory']), 'Corrected annual map redirected')
        ap = pinned(root, entry['audit_pin']); aa = json_file(ap)
        need(ap.parent == directory and aa['status'] == 'COMPLETE' and aa['year'] == year
             and aa['build_identity_sha256'] == entry['build_identity_sha256'], 'Corrected annual audit changed')
        if year < 2020:
            need(entry['audit_pin'] == ref['audit_pin'], 'Historical corrected annual audit detached')
            pi = json_file(pinned(root, ref['parent_identity_pin']))
            pac = json_file(pinned(root, ref['parent_audit_pin']))
            need(canonical(pi) == ref['build_identity_sha256'] == aa['build_identity_sha256']
                 and pac['status'] == 'COMPLETE' and pac['build_identity_sha256'] == ref['build_identity_sha256'],
                 'Historical corrected annual parent changed')
        else:
            need(ap in old_pins and old_pins[ap]['sha256'] == entry['audit_pin']['sha256'],
                 'Current annual audit absent from verified source authority')
        for field in ('targets', 'target_state', 'target_diagnostics'):
            original = pinned(directory, aa['artifacts'][field]); new = pinned(root, entry[field+'_pin'])
            need(new == original and entry[field+'_pin']['sha256'] == aa['artifacts'][field]['sha256'],
                 'Corrected annual target bytes differ from saved-model inputs')
    pre, backtest = run/'preflight', run/'backtest'
    pre_phase, pre_audit = json_file(pre/'build_identity.json'), json_file(pre/'audit.json')
    phase, bt_audit = json_file(backtest/'build_identity.json'), json_file(backtest/'audit.json')
    need(pre_phase['schema'] == 'dmv2-ret-backtest-preflight/v1'
         and pre_audit['status'] == 'COMPLETE' and pre_audit['build_identity_sha256'] == identity_sha
         and pre_audit['phase_identity_sha256'] == canonical(pre_phase)
         and pre_phase['parent_identity_sha256'] == identity_sha
         and pre_audit.get('performance_computed') is False and pre_phase.get('performance_computed') is False,
         'Selection/coverage preflight did not precede performance')
    need(phase['schema'] == 'dmv2-ret-backtest-phase/v1' and bt_audit['status'] == 'COMPLETE'
         and bt_audit['phase'] == phase['phase'] == 'ret_continuous_backtest_v1'
         and bt_audit['build_identity_sha256'] == phase['parent_identity_sha256'] == identity_sha
         and bt_audit['phase_identity_sha256'] == canonical(phase)
         and pinned(root, audit['backtest_audit_pin']) == backtest/'audit.json'
         and pinned(root, phase['preflight_audit_pin']) == pre/'audit.json'
         and phase.get('preflight_before_performance') is True
         and bt_audit.get('preflight_before_performance') is True
         and bt_audit['scenarios'] == 40 and bt_audit['sessions'] == 1006 and bt_audit['outcome_months'] == 48
         and bt_audit.get('models_fitted') is False and bt_audit.get('preprocessing_fitted') is False
         and bt_audit.get('new_inference_calls') == 0 and bt_audit.get('financial_2024_plus_read') is False
         and bt_audit.get('annual_liquidation') is False and bt_audit.get('source_identity_promoted') is False,
         'Forty-scenario continuous backtest identity/scope changed')
    need(pre_phase['annual_sources'] == phase['annual_sources'] == annuals
         and pre_phase['forecast_entries'] == phase['forecast_entries'] == identity['forecast_entries']
         and phase['configuration_sha256'] == pre_phase['configuration_sha256'] == canonical(cfg)
         and phase['basket'] == pre_phase['basket'] == BASKET
         and pre_phase['shared_payoff_vectors'] == 48 and pre_phase['payoff_month_class_cells'] == 57600
         and pre_phase['source_outcome_months'] == 167, 'RET preflight inputs/breadth changed before replay')
    for name in ('clock', 'event_registry', 'historical_retirements', 'identity_handover_registry',
                 'identity_handover_registry_pin', 'identity_handover_approval'):
        need(phase[name] == pre_phase[name] == old_phase[name], 'Frozen economic rule/evidence changed: '+name)
    for item in old_phase['source_pins']:
        path = pinned(root, item)
        need(path in new_pins and new_pins[path]['sha256'] == item['sha256'], 'Original execution source omitted/replaced')
    for item in phase['source_pins']: pinned(root, item)
    for item in pre_audit['artifacts'].values(): pinned(pre, item)
    for item in bt_audit['artifacts'].values(): pinned(backtest, item)
    for arm in ARMS:
        for name in ('original_ledger', 'selected', 'monthly', 'rebalances'):
            rel = f'selections/{arm}/{name}.parquet'
            old = pinned(pre, pre_audit['artifacts'][rel])
            new = pinned(root, phase['selection_artifacts'][rel])
            need(old == new, 'Backtest must reference exact immutable preflight selection frames')
    evidence = deepcopy(phase); evidence['parent_identity_sha256'] = SOURCE_IDENTITY
    handovers = v.review_handover_registry(root, evidence)
    need(len(handovers) == 1 and len(phase['event_registry']['events']) == 9
         and len(phase['historical_retirements']) == 11, 'Unchanged PENN/current/prior event scope required')
    dates, windows = v.independent_clock()
    return identity, pre_phase, pre_audit, phase, bt_audit, dates, windows, reference_run, handovers


def verification_read_plan(root, run, identity, phase, pre_audit, bt_audit):
    files = []
    def add(path, count):
        path = Path(path)
        need(path.is_file() and type(count) is int and count > 0, 'Missing bounded verification input')
        files.append({'path': str(path), 'passes': count, 'size_bytes': path.stat().st_size,
                      'reserved_compressed_bytes': path.stat().st_size * count})
    for annual in identity['annual_sources']:
        for name in ('targets', 'target_diagnostics'): add(pinned(root, annual[name+'_pin']), 2)
    for arm in identity['forecast_entries']:
        for file in arm['files']: add(pinned(root, file['predictions_pin']), 2)
    source = local(root, identity['source_reference_directory'])
    for year in range(2020, 2024): add(source/'years'/str(year)/'daily_prices_raw.parquet', 2)
    for item in phase['event_registry'].get('supplemental_quote_sources', []): add(pinned(root, item['pin']), 2)
    pre, backtest = run/'preflight', run/'backtest'
    add(pre/'source_month_class_means.parquet', 2)
    signals = [str(x) for x in pd.period_range('2019-12', '2023-11', freq='M')]
    available = sorted(path for path in (pre/'payoffs').iterdir() if path.is_dir())
    need(len(available) == 48 and [path.name[:7] for path in available] == signals, 'Exactly 48 dated payoff directories required')
    for directory in available:
        add(directory/'payoffs.parquet', 1); add(directory/'monthly.parquet', 2)
    for arm in ARMS:
        for name in ('original_ledger', 'selected', 'monthly', 'rebalances'):
            add(pre/f'selections/{arm}/{name}.parquet', 2)
        for variant, fee in itertools.product(VARIANTS, FEES):
            directory = backtest/f'arms/{arm}/{BASKET}/{variant}/cost_{fee:02d}bps'
            for name in (*LEDGERS, 'monthly'): add(directory/(name+'.parquet'), 2)
    for name in ('comparison', 'yearly', 'selection_coverage'): add(backtest/(name+'.parquet'), 2)
    return files, sum(row['reserved_compressed_bytes'] for row in files)


def read_source_means(root, identity, budget):
    cells, fingerprints, windows, source_policy = {}, {}, {}, None
    for annual in identity['annual_sources']:
        year = annual['year']; state = json_file(pinned(root, annual['target_state_pin']))
        provenance = state['provenance']
        need(canonical(provenance) == state['artifact_sha256'], 'Saved target provenance fingerprint changed')
        need(provenance['return_basis'] == 'provider_adjusted_close_ratio'
             and provenance['policy']['class_1'] == 'highest_return'
             and provenance['policy']['ranking_population'] == 'signal_time_identity_and_paper15_prediction_eligible'
             and provenance['policy']['minimum_return'] == -1
             and provenance['label_observed_through'] <= f'{year}-12-31', 'Source target policy/cutoff changed')
        need(source_policy in (None, provenance['policy_sha256']), 'Source target policies differ across years')
        source_policy = provenance['policy_sha256']
        declared = {row['outcome_month']: row for row in provenance['month_windows']}
        need(sorted(declared) == [f'{year}-{month:02}' for month in range(1, 13)], 'Annual completion windows changed')
        schemas = {name: pa.ipc.read_schema(pa.BufferReader(base64.b64decode(value, validate=True)))
                   for name, value in state['schemas'].items()}
        target_path = pinned(root, annual['targets_pin']); diagnostic_path = pinned(root, annual['target_diagnostics_pin'])
        need(pq.ParquetFile(target_path).metadata.num_rows <= 12000
             and pq.ParquetFile(diagnostic_path).metadata.num_rows == 12, 'Annual target bound exceeded')
        meta = budget.read(target_path, TARGET_META)
        need(not meta.duplicated(KEYS).any() and meta.outcome_month.isin(declared).all(), 'Original annual target keys changed')
        for row in meta.to_dict('records'):
            window = declared[row['outcome_month']]
            need(row['signal_date'] == window['signal_date'] and row['signal_date'] < row['outcome_start'] <= row['outcome_end']
                 and all(row[field] == window[field] for field in ('outcome_start', 'outcome_end'))
                 and row['outcome_month'] == month_step(row['signal_date'][:7], 1),
                 'Annual source timing rejected before financial target read')
        diagnostic_meta = budget.read(diagnostic_path, ['outcome_month', 'outcome_end'])
        need(diagnostic_meta.outcome_month.tolist() == list(declared)
             and all(row.outcome_end == declared[row.outcome_month]['outcome_end'] for row in diagnostic_meta.itertuples()),
             'Source diagnostic completion window changed')
        # Reading the mixed final annual Arrow block can decode Dec2023 in C++;
        # it is excluded before any financial Python conversion or fingerprint.
        raw = budget.arrow(target_path).cast(schemas['targets'])
        raw_diagnostic = budget.arrow(diagnostic_path).cast(schemas['target_diagnostics'])
        selected = raw.filter(pc.less_equal(raw['outcome_month'], '2023-11'))
        selected_diagnostic = raw_diagnostic.filter(pc.less_equal(raw_diagnostic['outcome_month'], '2023-11'))
        for month, window in declared.items():
            if month > '2023-11':
                continue
            need(window['outcome_end'] <= min(provenance['label_observed_through'], '2023-11-30'),
                 'Uncompleted final source month entered payoff history')
            target = selected.filter(pc.equal(selected['outcome_month'], month))
            diagnostic = selected_diagnostic.filter(pc.equal(selected_diagnostic['outcome_month'], month))
            need(source_month_fingerprint(target) == provenance['month_target_fingerprints'][month]
                 and source_month_fingerprint(diagnostic) == provenance['month_diagnostic_fingerprints'][month],
                 'Corrected true-class monthly Arrow fingerprint changed')
            need(diagnostic.num_rows == 1, 'Monthly rank-diagnostic cardinality changed')
            for row in class_means(target, window):
                key = (month, row['class_id']); need(key not in cells, 'Repeated true-class source month')
                cells[key] = row
            fingerprints[month] = provenance['month_target_fingerprints'][month]
            windows[month] = window
    expected = [month_step('2010-01', index) for index in range(167)]
    need(sorted(windows) == expected and len(cells) == 1670, 'Exactly 167 corrected completed source months required')
    return cells, fingerprints, windows, source_policy


def verify_payoffs(run, windows, source, budget):
    cells, fingerprints, completion, source_policy = source
    pre = run/'preflight'
    means_path = pre/'source_month_class_means.parquet'
    metadata = budget.read(means_path, ['outcome_month', 'outcome_end', 'class_id'])
    expected_keys = [(month, cls) for month in sorted(completion) for cls in range(1, 11)]
    need(list(map(tuple, metadata[['outcome_month', 'class_id']].to_numpy())) == expected_keys
         and all(row.outcome_end == completion[row.outcome_month]['outcome_end'] for row in metadata.itertuples()),
         'Saved monthly true-class source grid escaped independent metadata')
    means = budget.read(means_path)
    expected_means = pd.DataFrame([cells[key] for key in expected_keys], columns=list(MONTHLY_FIELDS))
    need(list(means.columns) == list(expected_means.columns), 'Saved monthly means schema changed')
    for actual, expected in zip(means.to_dict('records'), expected_means.to_dict('records'), strict=True):
        for field in MONTHLY_FIELDS:
            if field == 'class_mean_return': number(actual[field], expected[field], 'Independent source class mean')
            else: need(actual[field] == expected[field], 'Independent source class denominator/status changed')
    result, reports = {}, []
    for signal in windows.signal_date:
        directory = pre/'payoffs'/signal
        state = json_file(directory/'payoff_state.json'); provenance = state['provenance']
        first, last = trailing_months(signal)[0], signal[:7]
        need(provenance['signal_date'] == provenance['label_observed_through'] == signal
             and provenance['first_outcome_month'] == first and provenance['last_outcome_month'] == last
             and provenance['ret_policy'] == RET_DEFINITION['policy_id']
             and provenance['policy_sha256'] == canonical(RET_DEFINITION)
             and provenance['source_target_policy_sha256'] == source_policy
             and provenance['uses_predicted_classes'] is False
             and provenance['financial_rows_after_cutoff_consumed'] is False,
             'Shared payoff changed its causal scope/policy')
        need(provenance['expected_calendar_months'] == 120
             and provenance['expected_classes'] == list(range(1, 11))
             and provenance['expected_month_class_cells'] == 1200
             and provenance['monthly_class_mean'] == 'equal_weight_observed_realized_class_returns'
             and provenance['payoff_time_weighting'] == 'equal_calendar_month_weights'
             and provenance['missing_month_or_class_policy'] == 'withhold_entire_payoff_vector'
             and provenance['return_basis'] == 'provisional_provider_adjusted_endpoint_ratio'
             and all(provenance[field] is False for field in ('economic_return_certified', 'knowledge_time_certified',
                 'same_close_operational_availability_certified', 'model_ready', 'estimator_fit_performed')),
             'Shared payoff weighting or certification metadata changed')
        requested = trailing_months(signal)
        grid = [cells[(month, cls)] for month in requested for cls in range(1, 11)]
        need(all(row['outcome_end'] <= signal for row in grid), 'Future class return entered payoff vector')
        available = all(row['status'] == 'available' for row in grid)
        mu = independent_payoff(cells, signal)[0] if available else None
        need(provenance['vector_available'] is available
             and provenance['available_month_class_cells'] == sum(row['status'] == 'available' for row in grid)
             and provenance['source_month_target_fingerprints'] == {month: fingerprints[month] for month in requested}
             and provenance['source_completion_windows'] == [completion[month] for month in requested],
             'Payoff fingerprint sources or complete/withheld vector changed')
        gp = directory/'monthly.parquet'; pp = directory/'payoffs.parquet'
        need(pq.ParquetFile(gp).metadata.num_rows == 1200
             and pq.ParquetFile(pp).metadata.num_rows == 10, 'Payoff grid/vector row count changed')
        keys = budget.read(gp, ['outcome_month', 'outcome_end', 'class_id'])
        need(list(map(tuple, keys[['outcome_month', 'class_id']].to_numpy())) == [(month, cls) for month in requested for cls in range(1, 11)]
             and keys.outcome_end.le(signal).all(), 'Future or nonconsecutive monthly class row before payoff decode')
        grid_arrow = budget.arrow(gp); payoff_arrow = budget.arrow(pp)
        need(grid_arrow.schema.remove_metadata() == GRID_SCHEMA
             and payoff_arrow.schema.remove_metadata() == PAYOFF_SCHEMA,
             'Shared payoff numeric schema changed')
        for actual, expected in zip(grid_arrow.to_pylist(), grid, strict=True):
            for field in MONTHLY_FIELDS:
                if field == 'class_mean_return': number(actual[field], expected[field], 'Shared 120-month class mean')
                else: need(actual[field] == expected[field], 'Shared 120-month class grid changed')
        payoff_rows = payoff_arrow.to_pylist()
        need([row['class_id'] for row in payoff_rows] == list(range(1, 11)), 'Canonical payoff class order changed')
        for index, row in enumerate(payoff_rows):
            need(row['status'] == ('available' if available else 'withheld_incomplete_120_month_class_grid'),
                 'Partially released or incorrectly withheld RET vector')
            number(row['payoff'], None if mu is None else mu[index], 'Independent equal-month 120-month class payoff')
        need(state['fingerprint'] == canonical({'table': payoff_rows, 'monthly': grid_arrow.to_pylist(), 'provenance': provenance}),
             'Saved shared payoff logical fingerprint changed')
        result[signal] = {'mu': mu, 'fingerprint': state['fingerprint']}
        reports.append({'signal_date': signal, 'first_outcome_month': first,
            'last_outcome_month': last, 'month_class_cells': 1200, 'vector_available': available,
            'source_class_means_independently_recomputed': True})
    return result, reports


def read_forecasts(root, identity, windows, budget):
    pending, common, reports = {}, None, []
    # Complete metadata/cohort verification for all five arms precedes every
    # probability projection; target availability is audit-only, never a filter.
    for entry in identity['forecast_entries']:
        parts = []
        for file in entry['files']:
            path = pinned(root, file['predictions_pin']); pf = pq.ParquetFile(path)
            need(pf.metadata.num_rows <= (12000 if file['outer_year'] is not None else 45000)
                 and set((*FORECAST_META, *PROBABILITIES)) <= set(pf.schema_arrow.names), 'Forecast schema/row bound changed')
            meta = budget.read(path, FORECAST_META)
            verify_forecast_metadata(meta, windows, year=file['outer_year'])
            parts.append((file, meta))
        combined = pd.concat([meta for _, meta in parts], ignore_index=True)
        need(len(combined) == 43655 and int(combined.prediction_eligible.sum()) == 42145
             and int((combined.prediction_eligible & ~combined.target_available).sum()) == 65,
             'Original forecast rows, admissions or missing-outcome predictions lost')
        if common is None: common = combined
        else: exact_frame(combined, common, 'Five arms changed original ordered keys/admissions/outcome availability')
        pending[entry['arm']] = parts
    forecasts = {}
    for arm, parts in pending.items():
        output = []
        for file, meta in parts:
            probability = budget.read(pinned(root, file['predictions_pin']), PROBABILITIES)
            need(len(probability) == len(meta), 'Probability rows differ from checked forecast metadata')
            p = probability.loc[meta.prediction_eligible, PROBABILITIES].to_numpy(dtype=float)
            independent_scores(p, np.zeros(10))
            need(probability.loc[~meta.prediction_eligible].isna().all().all(), 'Ineligible original row received probabilities')
            output.append(pd.concat([meta.reset_index(drop=True), probability.reset_index(drop=True)], axis=1))
        forecasts[arm] = pd.concat(output, ignore_index=True)
        reports.append({'arm': arm, 'original_rows': 43655, 'prediction_rows': 42145,
            'missing_outcome_predictions_retained': 65, 'months': 48, 'same_ordered_cohort': True})
    return forecasts, reports


def verify_selection(forecasts, tables, windows, payoffs):
    ledger, chosen = tables['original_ledger'], tables['selected']
    monthly, rebalances = tables['monthly'], tables['rebalances']
    need(len(ledger) == len(forecasts) and not chosen.duplicated(KEYS).any(), 'RET selector lost original keys or duplicated holdings')
    for field in (*KEYS, 'intrinio_security_id', 'prediction_eligible', 'paper_prediction_eligible', 'identity_eligible',
                  'outcome_month', 'outcome_start', 'outcome_end'):
        exact_frame(ledger[[field]], forecasts[[field]], 'RET original ledger changed ' + field)
    restricted = forecasts.intrinio_security_id.isin(EXCLUDED)
    trading = forecasts.prediction_eligible & ~restricted
    need(ledger.execution_restricted.equals(restricted) and ledger.trade_eligible.equals(trading),
         'RET imposed a new hindsight restriction or changed prediction admission')
    for field in ('execution_restricted', 'trade_eligible', 'selected', 'ret_score_flat'):
        need(ledger[field].map(lambda value: isinstance(value, (bool, np.bool_))).all(),
             'RET selector decision/diagnostic flag is not strict Boolean')
    admitted = forecasts.prediction_eligible
    modes = forecasts.loc[admitted, PROBABILITIES].to_numpy(dtype=float).argmax(axis=1) + 1
    need(np.array_equal(ledger.loc[admitted, 'predicted_class'].to_numpy(dtype=int), modes), 'Diagnostic class mode changed')
    need(monthly.signal_date.tolist() == windows.signal_date.tolist()
         and rebalances.signal_date.tolist() == windows.signal_date.tolist(), 'RET omitted a monthly signal/rebalance')
    selected_keys, details = [], []
    for window in windows.to_dict('records'):
        signal = window['signal_date']; mask = forecasts.signal_date.eq(signal)
        group = forecasts.loc[mask]; original = ledger.loc[mask]
        eligible = group.prediction_eligible; trade = eligible & ~group.intrinio_security_id.isin(EXCLUDED)
        payoff = payoffs[signal]; mu = payoff['mu']
        available_n = min(10, int(trade.sum()) // 2)
        reference_scores = pd.Series(np.nan, index=group.index, dtype=float)
        if mu is not None:
            # The production selector multiplies trade-eligible rows as one
            # float64 matrix. Reproduce that precise floating arithmetic only
            # after independent scalar summation verifies the mathematical RET.
            canonical_trade = np.ascontiguousarray(group.loc[trade, PROBABILITIES].to_numpy(dtype=float)).copy()
            reference_scores.loc[trade] = canonical_trade @ mu
            independent = independent_scores(group.loc[eligible, PROBABILITIES].to_numpy(dtype=float), mu)
            canonical_all = np.ascontiguousarray(group.loc[eligible, PROBABILITIES].to_numpy(dtype=float)).copy()
            all_product = canonical_all @ mu
            need(np.allclose(independent, all_product, rtol=0., atol=3e-13), 'Matrix RET differs from independent scalar sum')
            reference_scores.loc[eligible & ~trade] = all_product[(~trade.loc[eligible]).to_numpy()]
            need(np.array_equal(original.selection_score.to_numpy(dtype=float), reference_scores.to_numpy(), equal_nan=True),
                 'Stored RET score differs from exact canonical probability-weighted payoff')
            longs, shorts, n = independent_rank(group, reference_scores.to_numpy())
        else:
            need(original.selection_score.isna().all(), 'Withheld RET vector produced numeric scores')
            longs, shorts, n = group.iloc[:0], group.iloc[:0], 0
        need(original.payoff_fingerprint.eq(payoff['fingerprint']).all(), 'An arm used a different shared payoff vector')
        need(original.loc[eligible, 'ret_status'].eq('available' if mu is not None else
             'withheld_incomplete_120_month_class_grid').all()
             and original.loc[~eligible, 'ret_status'].eq('withheld_signal_time_baseline').all(),
             'RET original cohort score availability changed')
        row = monthly.loc[monthly.signal_date.eq(signal)].iloc[0]
        rebalance = rebalances.loc[rebalances.signal_date.eq(signal)].iloc[0]
        need(row.execution_date == rebalance.execution_date == window['entry_date'], 'RET entry moved from next session open')
        need(row.status == rebalance.selection_status
             and (row.status == 'available') == (mu is not None and n > 0), 'Withheld RET rebalance became a synthetic cash portfolio')
        for field, value in {'original_rows': len(group), 'prediction_eligible': int(eligible.sum()),
            'trade_eligible': int(trade.sum()), 'long_candidates': int(trade.sum()),
            'short_candidates': int(trade.sum())-available_n, 'selected_names_per_side': n,
            'long_selected': n, 'short_selected': n}.items():
            need(int(row[field]) == value, 'RET breadth/denominator changed: ' + field)
        flat = bool(mu is not None and trade.any() and reference_scores.loc[trade].eq(reference_scores.loc[trade].iloc[0]).all())
        need(bool(row.ret_score_flat) == flat and original.ret_score_flat.eq(flat).all(), 'Flat RET score diagnostic changed')
        if mu is not None and trade.any():
            candidates = group.loc[trade, ['source_security_id']].assign(selection_score=reference_scores.loc[trade])
            ranks = candidates.sort_values(['selection_score', 'source_security_id'], ascending=[False, True], kind='stable')
            low_ranks = ranks.iloc[available_n:].sort_values(['selection_score', 'source_security_id'], ascending=[True, True], kind='stable')
            for field, ranked in (('ret_long_rank', ranks), ('ret_short_rank_after_long_reservation', low_ranks)):
                expected_rank = {index: rank for rank, index in enumerate(ranked.index, 1)}
                for index, item in original[field].items():
                    if index in expected_rank: need(item == expected_rank[index], 'Full RET rank diagnostic changed')
                    else: need(pd.isna(item), 'Noncandidate received a RET rank')
            number(row.ret_score_min, reference_scores.loc[trade].min(), 'RET minimum score')
            number(row.ret_score_max, reference_scores.loc[trade].max(), 'RET maximum score')
        expected_indices = set()
        for side, expected in (('long', longs), ('short', shorts)):
            actual = chosen.loc[chosen.signal_date.eq(signal) & chosen.side.eq(side)]
            need(sorted(actual.source_security_id.tolist()) == sorted(expected.source_security_id.tolist()),
                 'RET full-distribution rank, disjointness or stable-source tie selection changed')
            need(actual.execution_date.eq(window['entry_date']).all(), 'Selected RET execution clock changed')
            for item in actual.to_dict('records'):
                source = group.loc[group.source_security_id.eq(item['source_security_id'])].iloc[0]
                need(item['intrinio_security_id'] == source.intrinio_security_id, 'Selected native identity changed')
                number(item['selection_score'], reference_scores.loc[source.name], 'Selected RET score')
                number(item['intended_weight'], (.5 if side == 'long' else -.5) / n, 'RET equal within-side weight')
                selected_keys.append((item['source_security_id'], signal)); expected_indices.add(source.name)
            actual_flags = original.loc[original.side.eq(side) & original.selected, 'source_security_id'].tolist()
            need(sorted(actual_flags) == sorted(expected.source_security_id.tolist()), 'Original ledger side/selection flag changed')
        need(set(original.index[original.selected]) == expected_indices
             and original.loc[~original.selected, 'side'].isna().all(), 'Unselected RET row contains an executed side')
        need(original.loc[restricted.loc[mask] & admitted.loc[mask], 'selection_status'].eq(
             'execution_excluded_retrospective_data_quality').all(), 'Fixed restriction reason changed')
        need(original.loc[~eligible, 'selection_status'].eq('withheld_signal_time_baseline').all(),
             'Unqualified forecast row was admitted to RET selection')
        if mu is not None and n:
            need(original.loc[original.selected, 'selection_status'].eq('selected').all()
                 and original.loc[trade & ~original.selected, 'selection_status'].eq('not_selected').all(),
                 'RET selected/not-selected original status changed')
        details.append({'signal_date': signal, 'trade_eligible': int(trade.sum()),
                        'names_per_side': n, 'flat_scores': flat, 'vector_available': mu is not None})
    need(len(chosen) == len(selected_keys), 'Extra selected rows outside canonical monthly grid')
    return {'original_rows': len(ledger), 'prediction_rows': int(admitted.sum()),
            'signal_months': len(windows), 'selected_slots': len(selected_keys),
            'future_label_or_quote_filter_used': False, 'months': details}


def read_quotes(v, root, phase, reference_run, sessions, budget):
    """All annual and supplemental key guards precede every quote value read."""
    pins = {local(root, item['path']): item for item in phase['source_pins']}
    paths, original_ids, total_rows = [], set(), 0
    for year in range(2020, 2024):
        directory = reference_run/'years'/str(year)
        ap, path = directory/'audit.json', directory/'daily_prices_raw.parquet'
        need(ap in pins and path in pins, 'Independent source quote/audit pin missing')
        audit = json_file(pinned(root, pins[ap])); pinned(root, pins[path])
        need(audit['status'] == 'COMPLETE' and audit['year'] == year
             and audit['build_identity_sha256'] == SOURCE_IDENTITY, 'Execution annual source year/identity changed')
        pf = pq.ParquetFile(path)
        need(pf.metadata.num_rows <= 300000 and set(v.PRICE_FIELDS) <= set(pf.schema_arrow.names),
             'Execution quote schema/row ceiling changed')
        total_rows += pf.metadata.num_rows
        keys = budget.read(path, ['trading_date', 'intrinio_security_id'])
        need(keys.trading_date.map(lambda day: isinstance(day, str) and f'{year}-01-01' <= day <= f'{year}-12-31').all()
             and keys.trading_date.isin(sessions).all()
             and keys.intrinio_security_id.map(lambda value: isinstance(value, str) and bool(value.strip())).all()
             and not keys.duplicated(['trading_date', 'intrinio_security_id']).any(),
             'Execution quote metadata rejected before numeric decoding')
        original_ids.update(keys.intrinio_security_id)
        original_ids.update(audit['feature_provenance']['requested_ids'])
        paths.append(path)
    need(total_rows == 947517, 'Preserved four-year execution quote population changed')
    successors = {event['successor_id'] for event in phase['event_registry']['events'] if event.get('successor_id')}
    for source in phase['event_registry'].get('supplemental_quote_sources', []):
        path = pinned(root, source['pin']); allowed = set(source['allowed_native_ids'])
        need(path.is_relative_to(root/'data/derived') and path.suffix == '.parquet'
             and allowed and allowed <= successors and not allowed & original_ids
             and '2020-01-01' <= source['start'] <= source['end'] <= '2023-12-31',
             'Supplemental quote source changed approved identity or period')
        keys = budget.read(path, ['trading_date', 'intrinio_security_id'])
        need(set(keys.intrinio_security_id) <= allowed and keys.trading_date.between(source['start'], source['end']).all()
             and keys.trading_date.isin(sessions).all(), 'Supplemental quote keys escaped source approval')
        paths.append(path)
    prices = pd.concat([budget.read(path, v.PRICE_FIELDS) for path in paths], ignore_index=True)
    need(not prices.duplicated(['trading_date', 'intrinio_security_id']).any(), 'Execution source quote overwrite')
    return prices


def read_reports(run, budget):
    backtest = run/'backtest'
    bounds = budget.read(backtest/'comparison.parquet', ['outcome_start', 'outcome_end'])
    need(len(bounds) == 40 and bounds.outcome_start.eq('2020-01-01').all()
         and bounds.outcome_end.eq('2023-12-31').all(), 'Comparison financial period rejected before numeric read')
    years = budget.read(backtest/'yearly.parquet', ['year'])
    need(len(years) == 160 and set(years.year) == {2020, 2021, 2022, 2023}, 'Annual metric chronology rejected')
    return budget.read(backtest/'comparison.parquet'), budget.read(backtest/'yearly.parquet')


def verify_grid(comparison, yearly, coverage):
    columns = ['arm', 'basket', 'variant', 'cost_bps']
    expected = set(itertools.product(ARMS, (BASKET,), VARIANTS, FEES))
    need(len(comparison) == 40 and not comparison.duplicated(columns).any()
         and set(map(tuple, comparison[columns].to_numpy())) == expected, 'Exact forty RET scenarios required')
    need(len(yearly) == 160 and not yearly.duplicated([*columns, 'year']).any()
         and set(map(tuple, yearly[[*columns, 'year']].to_numpy())) ==
             {(*item, year) for item in expected for year in range(2020, 2024)}, 'Exact four calendar-year slices per RET path required')
    need(len(coverage) == 240 and set(coverage.arm) == set(ARMS) and set(coverage.basket) == {BASKET}
         and not coverage.duplicated(['arm', 'signal_date']).any(), 'Exactly five 48-month RET coverage records required')
    need(comparison.complete_proxy_path.map(lambda value: isinstance(value, (bool, np.bool_))).all(),
         'Every RET scenario needs explicit complete/withheld state')


def monthly_table(budget, path):
    pf = pq.ParquetFile(path)
    need(pf.metadata.num_rows == 48 and 'outcome_month' in pf.schema_arrow.names, 'Exactly 48 saved outcome months required')
    keys = budget.read(path, ['outcome_month'])
    need(keys.outcome_month.tolist() == [str(month) for month in pd.period_range('2020-01', '2023-12', freq='M')],
         'Monthly financial report escaped frozen dates')
    return budget.read(path)


def verify_saved_run(root, run, *, output):
    started = time.monotonic()
    root, run, output = [Path(path).resolve() for path in (root, run, output)]
    need(run.is_relative_to(root/'outputs/ret_backtest') and run != root/'outputs/ret_backtest'
         and output.is_relative_to(run/'verification') and not output.exists(), 'Fresh private RET verification output required')
    v = load_equations(root)
    identity, pre_phase, pre_audit, phase, bt_audit, dates, windows, reference_run, handovers = authenticate(root, run, v)
    root_hash, pre_hash, bt_hash = digest(run/'audit.json'), digest(run/'preflight/audit.json'), digest(run/'backtest/audit.json')
    read_plan, amount = verification_read_plan(root, run, identity, phase, pre_audit, bt_audit)
    prior = reserve_verification(run, amount)
    budget = ProjectionBudget(prior=prior, reserved=amount)

    def check_resources():
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if platform.system() != 'Darwin': peak *= 1024
        need(time.monotonic() - started <= LIMITS['verification_wall_seconds']
             and peak <= LIMITS['rss_stop_bytes'] and budget.used <= amount,
             'Independent RET verifier resource ceiling exceeded')
        return peak

    source = read_source_means(root, identity, budget)
    payoffs, payoff_reports = verify_payoffs(run, windows, source, budget)
    check_resources()
    forecasts, forecast_reports = read_forecasts(root, identity, windows, budget)
    comparison, yearly = read_reports(run, budget)
    coverage = guarded_table(v, budget, run/'backtest/selection_coverage.parquet')
    verify_grid(comparison, yearly, coverage)
    need(bt_audit['complete_proxy_paths'] == int(comparison.complete_proxy_path.sum())
         and bt_audit['failed_closed_paths'] == int((~comparison.complete_proxy_path).sum()),
         'Saved RET complete/withheld scenario counts disagree')
    prices = read_quotes(v, root, phase, reference_run, dates, budget)
    quote_index = {(row['trading_date'], row['intrinio_security_id']): row for row in prices.to_dict('records')}
    selections, scenario_checks, semantic = [], [], []
    for arm in ARMS:
        check_resources()
        tables = {name: guarded_table(v, budget, run/f'preflight/selections/{arm}/{name}.parquet', maximum_rows=60000)
                  for name in ('original_ledger', 'selected', 'monthly', 'rebalances')}
        # The replay uses these identical immutable paths. Reconstructing each
        # frame below authenticates its logical contents, rather than relying
        # on Parquet byte equality as evidence for selection correctness.
        for name in tables:
            rel = f'selections/{arm}/{name}.parquet'
            need(pinned(root, phase['selection_artifacts'][rel]) ==
                 pinned(run/'preflight', pre_audit['artifacts'][rel]), 'Replay detached its immutable selection table')
            semantic.append({'arm': arm, 'table': name, 'same_immutable_preflight_frame_used': True,
                             'logical_contents_independently_checked': True})
        selections.append({'arm': arm, **verify_selection(forecasts[arm], tables, windows, payoffs)})
        joined = coverage.loc[coverage.arm.eq(arm)].drop(columns=['arm']).reset_index(drop=True)
        # Every individual monthly table already carries its canonical basket.
        exact_frame(joined, tables['monthly'].reset_index(drop=True), 'Aggregate RET coverage differs from its original monthly frame')
        for variant, fee in itertools.product(VARIANTS, FEES):
            check_resources()
            directory = run/f'backtest/arms/{arm}/{BASKET}/{variant}/cost_{fee:02d}bps'
            ledgers = {name: guarded_table(v, budget, directory/(name+'.parquet')) for name in LEDGERS}
            monthly = monthly_table(budget, directory/'monthly.parquet')
            mask = comparison.arm.eq(arm) & comparison.basket.eq(BASKET) & comparison.variant.eq(variant) & comparison.cost_bps.eq(fee)
            summary = comparison.loc[mask].iloc[0]
            annual = yearly.loc[yearly.arm.eq(arm) & yearly.basket.eq(BASKET) & yearly.variant.eq(variant) & yearly.cost_bps.eq(fee)]
            stop = None if variant == 'unstopped' else .1
            if summary.complete_proxy_path:
                checked = v.verify_scenario(ledgers, prices, tables['selected'], tables['rebalances'], dates,
                    events=phase['event_registry']['events'], historical_retirements=phase['historical_retirements'],
                    cost_bps=fee, stop_fraction=stop, prepared_quotes=quote_index, identity_handovers=handovers)
                v.verify_metrics(ledgers['daily'], ledgers['trades'], monthly, summary, annual)
                checked.update(path_status='complete_proxy', economic_performance_available=True)
            else:
                checked = v.verify_failed_scenario(ledgers, prices, tables['selected'], dates,
                    events=phase['event_registry']['events'], historical_retirements=phase['historical_retirements'],
                    cost_bps=fee, stop_fraction=stop, prepared_quotes=quote_index,
                    rebalances=tables['rebalances'], identity_handovers=handovers)
                v.verify_failed_metrics(ledgers['daily'], ledgers['trades'], monthly, summary, annual)
                checked['path_status'] = 'failed_closed'
            scenario_checks.append({'arm': arm, 'basket': BASKET, 'variant': variant, 'cost_bps': fee, **checked})
            print(json.dumps({'phase': 'independent_ret_backtest_verification',
                'verified_scenarios': len(scenario_checks), 'expected_scenarios': 40}), flush=True)
    for item in [*identity['code_pins'], *identity['source_pins'], *phase['source_pins']]: pinned(root, item)
    for item in pre_audit['artifacts'].values(): pinned(run/'preflight', item)
    for item in bt_audit['artifacts'].values(): pinned(run/'backtest', item)
    need(digest(run/'audit.json') == root_hash and digest(run/'preflight/audit.json') == pre_hash
         and digest(run/'backtest/audit.json') == bt_hash, 'Audited RET identity/result changed during verification')
    result = {'schema': 'dmv2-ret-backtest-independent-verification/v1', 'status': 'PASS',
        'phase': 'independent_ret_backtest_verification', 'build_identity_sha256': canonical(identity),
        'root_audit_sha256': root_hash, 'preflight_audit_sha256': pre_hash,
        'backtest_audit_sha256': bt_hash, 'backtest_phase_identity_sha256': canonical(phase),
        'script_sha256': digest(__file__), 'independent_equations_script_sha256': BASE_SHA,
        'models_fitted': 0, 'preprocessors_fitted': 0, 'new_inference_calls': 0,
        'production_RET_or_selection_called': False, 'production_accounting_called': False,
        'source_pins_unchanged': True, 'old_runs_and_pointers_changed': False,
        'source_outcome_months': 167, 'source_month_class_cells': 1670,
        'source_class_means_independently_recomputed': True,
        'shared_payoff_vectors': 48, 'payoff_month_class_cells': 57600,
        'all_original_forecast_keys_and_admissions_preserved': True,
        'all_five_arms_share_identical_payoff_vectors': True,
        'missing_label_predictions_retained_per_arm': 65,
        'financial_2024_plus_read': False, 'financial_outcomes_after_2023_11_used_for_RET': False,
        'mixed_2023_annual_Arrow_block_decode_before_filter_possible': True,
        'no_excluded_future_financial_rows_converted_to_Python': True,
        'sessions': 1006, 'outcome_months': 48, 'scenarios': 40,
        'complete_proxy_paths': int(comparison.complete_proxy_path.sum()),
        'withheld_paths': int((~comparison.complete_proxy_path).sum()),
        'payoffs': payoff_reports, 'forecasts': forecast_reports,
        'selections': selections, 'preflight_semantic_checks': semantic, 'scenario_checks': scenario_checks,
        'knowledge_time_identity_survivorship_or_investability_certified': False,
        'withheld_economic_performance_certified': False,
        'prior_phase_projected_compressed_bytes': prior,
        'verification_reserved_compressed_file_upper_bound_bytes': amount,
        'verification_actual_compressed_file_upper_bound_bytes': budget.used,
        'verification_read_plan': read_plan,
        'execution_seconds': time.monotonic() - started, 'peak_rss_bytes': check_resources()}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False); stream.write('\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'run', 'backtest', 'output'):
        parser.add_argument('--'+name, required=True, type=Path)
    args = parser.parse_args()
    need(args.backtest.resolve() == args.run.resolve()/'backtest', 'Exact fresh RET backtest directory required')
    result = verify_saved_run(args.root, args.run, output=args.output)
    print(json.dumps({key: value for key, value in result.items()
        if key not in ('payoffs', 'forecasts', 'selections', 'scenario_checks', 'verification_read_plan')}))


if __name__ == '__main__':
    main()
