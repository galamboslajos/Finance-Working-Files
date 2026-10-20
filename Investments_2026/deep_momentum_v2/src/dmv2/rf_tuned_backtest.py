"""Additive, zero-fit replay of the verified selected RF annual forecasts.

Original source identities and execution approvals stay intact.  The new run
pins those dependencies; it does not promote identity or inherit fitting power.
The explicit CLI is the only execution entry point; no notebook imports it.
"""
from __future__ import annotations

from contextlib import contextmanager
import datetime as dt
import fcntl
from importlib.metadata import version
import itertools
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import pandas as pd
import psutil
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

from .backtest_accounting import DailyAccountingPolicy
from .backtest_loading import _hash, _pin, _projected_bytes, _verify_pin, _within
from .backtest_selection import BasketSelection, SAFE_COLUMNS, PROBABILITIES
from .catalog import canonical_sha256
from .daily_backtest import _save_table, summarize_daily
from .source_corrected_backtest import _reference_context
from .universe import DataContractError
from .walkforward_accounting_v2 import IMPLEMENTATION_REVISION, run_walkforward_accounting_v2
from .walkforward_backtest import _configuration, _historical_registry, _price_plan, yearly_breakdown, WORKERS as OLD_WORKERS
from .walkforward_backtest_v2 import _identity_registry
from .walkforward_clock import walkforward_clock
from .walkforward_preflight import QUOTE_METADATA_COLUMNS, preflight_selected_horizons, validate_current_events
from .walkforward_selection import BALANCED_BREADTH_POLICY, select_walkforward_basket, validate_walkforward_basket
from .action_loading import ACTION_PRICE_COLUMNS

PHASE = 'rf_tuned_continuous_backtest_v1'
ARM = 'rf_extended_tuned'
YEARS = (2020, 2021, 2022, 2023)
SCENARIOS = tuple(itertools.product(('class_1_10', 'class_2_9'),
                                  ('unstopped', 'asset_stop_10pct'), (0, 5, 10, 25)))
NUMERIC_ENV = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
               'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')
COHORT_COLUMNS = ('source_security_id', 'signal_date', 'intrinio_security_id',
                  'identity_eligible', 'paper_prediction_eligible', 'prediction_eligible',
                  'outcome_month', 'outcome_start', 'outcome_end')
LINEAGE_COLUMNS = ('fit_year', 'training_outcome_start', 'training_outcome_end',
                   'training_label_observed_through')
NEW_FILES = ('src/dmv2/rf_tuned_backtest.py', 'scripts/run_rf_tuned_backtest.py',
             'scripts/verify_rf_tuned_backtest.py', 'scripts/watch_rf_tuned_backtest.py')


def need(value, message):
    if not value:
        raise DataContractError(message)


def read_json(path):
    path = Path(path)
    need(path.is_file() and path.stat().st_size <= 16 * 1024**2, 'Unbounded/missing JSON metadata')
    return json.loads(path.read_text())


def write_json(path, value, *, replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    need(replace or not path.exists(), 'Preserve an existing checkpoint: ' + str(path))
    temporary = path.with_name(path.name + '.partial')
    need(not temporary.exists(), 'Preserve an interrupted checkpoint: ' + str(temporary))
    with temporary.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def absolute_pin(root, item):
    return {**item, 'path': str(_within(root, Path(root) / item['path']).resolve())}


def authenticated(root, item):
    return _verify_pin(Path(root), item)


def require_frozen_pin(root, actual, frozen):
    """Compare a live child claim with the old, already sealed parent pin."""
    path = str((Path(root) / actual['path']).resolve())
    matches = [pin for pin in frozen if str((Path(root) / pin['path']).resolve()) == path]
    need(matches and all(pin['sha256'] == actual['sha256'] and pin['size_bytes'] == actual['size_bytes']
                         for pin in matches), 'Child source differs from completed parent phase pin: ' + path)


def runtime():
    return {'python': platform.python_version(), **{name: version(name) for name in
            ('numpy', 'pandas', 'pyarrow', 'exchange-calendars', 'psutil', 'threadpoolctl')}}


def configuration(root, path):
    cfg = read_json(path)
    need(cfg.get('schema') == 'dmv2-rf-tuned-backtest-execution/v1'
         and cfg.get('status') == 'FROZEN_EXECUTION_AUTHORIZED'
         and cfg.get('execution_authorized') is True, 'Separate explicit backtest authority absent')
    need(cfg.get('arm') == ARM and cfg.get('scenarios') == 16
         and cfg.get('new_estimator_fits') == cfg.get('new_preprocessor_fits') == cfg.get('new_inference_calls') == 0,
         'Only the sixteen zero-fit/inference RF replay scenarios are allowed')
    b = cfg['resource_limits']
    expected = {'rss_stop_bytes': 4 * 1024**3, 'rss_warning_bytes': 3 * 1024**3,
                'minimum_available_memory_bytes': 7 * 1024**3, 'normal_pressure_dispatch': 1,
                'cumulative_compressed_read_bytes': 512 * 1024**2, 'output_bytes': 256 * 1024**2,
                'backtest_wall_seconds': 900, 'preflight_wall_seconds': 300,
                'verification_wall_seconds': 1800, 'workers': 1, 'numeric_threads': 1}
    need(b == expected, 'Frozen resource limits changed')
    need(cfg.get('financial_2024_plus_authorized') is False and cfg.get('cloud_operations') is False,
         'Protected or cloud scope was broadened')
    base = read_json(authenticated(root, cfg['base_configuration_pin']))
    _configuration(base)
    return cfg, base


def resources(run, cfg, started, seconds, *, launch=False):
    process = psutil.Process()
    try:
        rss = sum(p.memory_info().rss for p in (process, *process.children(recursive=True)))
    except psutil.Error as error:
        raise DataContractError('Cannot measure process-tree RSS') from error
    limits = cfg['resource_limits']
    need(rss < limits['rss_stop_bytes'], 'RF replay process-tree RSS stop')
    need(time.monotonic() - started <= seconds, 'RF replay phase wall-time stop')
    output_bytes = sum(p.stat().st_size for p in Path(run).rglob('*') if p.is_file())
    need(output_bytes < limits['output_bytes'], 'RF replay new-output limit')
    pressure = subprocess.run(['/usr/sbin/sysctl', '-n', 'kern.memorystatus_vm_pressure_level'],
                              check=True, text=True, capture_output=True).stdout.strip()
    need(pressure == '1', 'RF replay requires normal known macOS memory pressure')
    if launch:
        raw = subprocess.run(['/usr/bin/vm_stat'], check=True, text=True, capture_output=True).stdout
        import re
        size = int(re.search(r'page size of (\d+) bytes', raw).group(1))
        pages = {name: int(re.search(re.escape(name) + r':\s*(\d+)', raw).group(1))
                 for name in ('Pages free', 'Pages inactive', 'Pages speculative')}
        need(size * sum(pages.values()) >= limits['minimum_available_memory_bytes'],
             'RF replay launch needs seven GiB available/reclaimable memory')
    return {'process_tree_rss_bytes': rss, 'output_bytes': output_bytes,
            'pressure_dispatch': 1, 'rss_warning': rss >= limits['rss_warning_bytes']}


@contextmanager
def exclusive(run):
    path = Path(run) / 'controller.lock'
    with path.open('a') as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise DataContractError('Another replay/verification phase is active') from error
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def reserve_reads(run, phase, projected_bytes, *, limit=512 * 1024**2):
    """Charge exact compressed projections before any table conversion.

    The independent verifier uses the same simple ledger schema but its own
    implementation. Failed/interrupted reservations remain charged.
    """
    need(type(projected_bytes) is int and projected_bytes >= 0, 'Invalid compressed read reservation')
    path = Path(run) / 'read_usage.json'
    state = read_json(path)
    need(state['schema'] == 'dmv2-rf-tuned-read-usage/v1' and state['limit_bytes'] == limit,
         'Read-budget ledger changed')
    need(not any(row['phase'] == phase for row in state['reservations']), 'No automatic phase retries')
    need(state['total_projected_compressed_bytes'] == sum(row['projected_compressed_bytes'] for row in state['reservations']),
         'Compressed read ledger does not reconcile')
    total = state['total_projected_compressed_bytes'] + projected_bytes
    need(total <= limit, 'Cumulative compressed read budget exceeded before conversion')
    state['reservations'].append({'phase': phase, 'projected_compressed_bytes': projected_bytes,
                                  'reserved_at': dt.datetime.now(dt.timezone.utc).isoformat()})
    state['total_projected_compressed_bytes'] = total
    write_json(path, state, replace=True)
    return state


def _base_reference(root, cfg):
    directory = _within(root, Path(root) / cfg['source_reference_backtest'])
    audit_pin, phase_pin = cfg['source_reference_audit_pin'], cfg['source_reference_phase_pin']
    authenticated(root, audit_pin); authenticated(root, phase_pin)
    audit = read_json(directory / 'audit.json')
    context, phase, old_identity = _reference_context(root, directory,
        audit_sha256=audit_pin['sha256'], phase_sha256=audit['phase_identity_sha256'])
    # The old completed phase is a pre-existing frozen authority. Never admit a
    # daily file merely because a freshly edited annual audit hashes it now.
    for item in phase['source_pins']:
        authenticated(root, item)
    context['immutable_source_pins'] = list({(p['path'], p['sha256']): p
        for p in (*context['immutable_source_pins'], *phase['source_pins'])}.values())
    old_verifier = read_json(authenticated(root, cfg['source_reference_verification_pin']))
    need(old_verifier['status'] == 'PASS'
         and old_verifier['backtest_audit_sha256'] == audit_pin['sha256']
         and old_verifier['backtest_phase_identity_sha256'] == audit['phase_identity_sha256']
         and old_verifier['build_identity_sha256'] == canonical_sha256(old_identity),
         'Original corrected daily accounting lacks its exact independent PASS')
    context['immutable_source_pins'].append(cfg['source_reference_verification_pin'])
    return directory, context, phase, old_identity


def _forecast_entries(root, cfg, tuning_identity, ledger, verifier):
    entries, pins = [], []
    need(ledger['status'] == 'COMPLETE_PENDING_INDEPENDENT_VERIFICATION'
         and ledger['initiated_estimator_fits'] == 100 and ledger['initiated_preprocessor_fits'] == 16
         and len(ledger['tasks']) == 120 and all(t['status'] == 'COMPLETE' for t in ledger['tasks']),
         'Tuning search is not a complete fixed-grid run')
    need(verifier['status'] == 'PASS_COMPLETE' and verifier['complete_search_verified'] is True
         and verifier['runtime_verified'] is True
         and verifier['build_identity_sha256'] == tuning_identity['identity_sha256']
         and verifier['run_root'] == tuning_identity['run_root']
         and verifier['verified_fit_manifests'] == 100 and len(verifier['verified_prepared_splits']) == 16
         and verifier['verified_inner_probability_ledgers'] == 96
         and verifier['verified_selected_outer_model_reproductions'] == 4,
         'A complete independent tuning verification is required')
    need(verifier['new_estimator_fits'] == verifier['new_preprocessor_fits']
         == verifier['new_source_financial_decodes'] == 0, 'Unexpected verification side effects')
    source_root = Path(root) / cfg['source_reference_root']
    model_audit_pin = next(x for x in tuning_identity['reference_pins'] if Path(x['path']) == source_root / 'models/audit.json')
    model_audit = read_json(authenticated(root, model_audit_pin)); pins.append(model_audit_pin)
    model_verify_pin = cfg['source_model_verification_pin']
    model_verify = read_json(authenticated(root, model_verify_pin)); pins.append(model_verify_pin)
    need(model_verify['status'] == 'PASSED' and model_verify['models_audit_sha256'] == model_audit_pin['sha256']
         and model_verify['build_identity_sha256'] == model_audit['build_identity_sha256']
         and model_verify['model_identity_sha256'] == model_audit['model_identity_sha256']
         and model_verify['annual_fits_reconciled'] == 16 and model_verify['stitched_prediction_months'] == 48,
         'Original four-arm forecasts lack exact independent verification')
    for year in YEARS:
        task = next(t for t in ledger['tasks'] if t['action'] == 'predict_outer' and t['outer_year'] == year)
        receipt = read_json(authenticated(root, task['receipt_pin']))
        manifest = read_json(authenticated(root, receipt['prediction_manifest']))
        winner_pin = ledger['winners'][str(year)]
        winner = read_json(authenticated(root, winner_pin))
        selected = next(x for x in verifier['selected_winners'] if x['outer_year'] == year)
        need(manifest['outer_year'] == year and manifest['split_id'] == f'{year}_outer_train'
             and manifest['status'] == 'COMPLETE' and manifest['winner_path'] == winner_pin
             and manifest['candidate_id'] == winner['candidate_id'] == selected['candidate_id'],
             'Selected outer forecast/winner binding changed')
        prepared = read_json(authenticated(root, manifest['prepare_manifest']))
        fit = read_json(authenticated(root, manifest['fit_manifest']))
        need(prepared['stage'] == fit['stage'] == 'outer' and prepared['outer_year'] == fit['outer_year'] == year
             and fit['prepare_manifest'] == manifest['prepare_manifest']
             and fit['candidate_id'] == manifest['candidate_id'], 'Wrong outer prepared/fit provenance')
        need(fit['probability_class_ids'] == list(range(1, 11))
             and manifest['prediction_provenance']['class_ids'] == list(range(1, 11)),
             'Selected forecast probability class order is not canonical 1 through 10')
        train = prepared['train_provenance']
        need(train['train_start'] == f'{year-10}-01-01' and train['train_end'] == f'{year-1}-12-31',
             'Forecast does not use exactly the preceding ten completed years')
        cutoff = {2020: '2019-12-31', 2021: '2020-12-31', 2022: '2021-12-31', 2023: '2022-12-30'}[year]
        need(train.get('label_observed_through') == cutoff, 'Annual completed-label cutoff changed')
        fold_pin = absolute_pin(root, {**model_audit['folds'][str(year)],
            'path': str(source_root / 'models' / model_audit['folds'][str(year)]['path'])})
        fold = read_json(authenticated(root, fold_pin))
        arm_pin = absolute_pin(root, {**fold['arms']['rf_extended'],
            'path': str(Path(fold_pin['path']).parent / fold['arms']['rf_extended']['path'])})
        arm = read_json(authenticated(root, arm_pin))
        old_predictions = absolute_pin(root, {**arm['artifacts']['predictions'],
            'path': str(Path(arm_pin['path']).parent / arm['artifacts']['predictions']['path'])})
        entry = {'outer_year': year, 'candidate_id': manifest['candidate_id'],
            'prediction_manifest_pin': receipt['prediction_manifest'],
            'predictions_pin': manifest['artifacts']['predictions'],
            'prepare_manifest_pin': manifest['prepare_manifest'], 'fit_manifest_pin': manifest['fit_manifest'],
            'winner_pin': winner_pin, 'original_rf_predictions_pin': old_predictions,
            'training_outcome_start': train['train_start'], 'training_outcome_end': train['train_end'],
            'training_label_observed_through': cutoff}
        entries.append(entry)
        pins.extend([task['receipt_pin'], receipt['prediction_manifest'], manifest['prepare_manifest'],
            manifest['fit_manifest'], winner_pin, manifest['artifacts']['predictions'], fold_pin, arm_pin, old_predictions])
    return entries, pins


def prepare(root, *, configuration_path, output_directory):
    root = Path(root).resolve(); output = _within(root / 'outputs/rf_expanded_tuned_backtest', Path(output_directory).resolve())
    need(output != root / 'outputs/rf_expanded_tuned_backtest' and not output.exists(), 'Use a fresh private replay directory')
    cfg, base = configuration(root, configuration_path)
    tuning_identity = read_json(authenticated(root, cfg['tuning_identity_pin']))
    ledger = read_json(authenticated(root, cfg['tuning_ledger_pin']))
    verifier = read_json(authenticated(root, cfg['tuning_verification_pin']))
    need(absolute_pin(root, ledger['build_identity_pin']) == absolute_pin(root, cfg['tuning_identity_pin']),
         'Tuning final ledger identity differs')
    identity_payload = {k: v for k, v in tuning_identity.items() if k not in ('identity_sha256', 'run_root', 'created_at')}
    need(canonical_sha256(identity_payload) == tuning_identity['identity_sha256'], 'Tuning identity canonical hash differs')
    for item in (*tuning_identity['code_pins'], *tuning_identity['reference_pins']):
        authenticated(root, item)
    reference, context, old_phase, old_identity = _base_reference(root, cfg)
    source_identity = canonical_sha256(old_identity)
    need(source_identity == cfg['source_reference_root_identity_sha256'], 'Original corrected source identity changed')
    entries, pins = _forecast_entries(root, cfg, tuning_identity, ledger, verifier)
    old_config = old_identity['configuration']
    need(old_config == base, 'Existing reference policy differs from pinned base policy')
    pins.extend(context['immutable_source_pins'])
    for year in YEARS:
        directory = root / cfg['source_reference_root'] / 'years' / str(year)
        audit_pin = _pin(root, directory / 'audit.json'); audit = read_json(authenticated(root, audit_pin))
        require_frozen_pin(root, audit_pin, old_phase['source_pins'])
        need(audit['status'] == 'COMPLETE' and audit['year'] == year
             and audit['build_identity_sha256'] == source_identity, 'Wrong original corrected daily annual identity')
        daily_pin = absolute_pin(root, {**audit['artifacts']['daily_prices_raw'],
            'path': str(directory / audit['artifacts']['daily_prices_raw']['path'])})
        require_frozen_pin(root, daily_pin, old_phase['source_pins'])
        authenticated(root, daily_pin); pins.extend([audit_pin, daily_pin])
    code = [root / 'src/dmv2' / name for name in (*OLD_WORKERS, 'walkforward_backtest_v2.py',
           'walkforward_accounting_v2.py', 'source_corrected_backtest.py')]
    code.extend(root / name for name in NEW_FILES)
    code_pins = [_pin(root, p) for p in dict.fromkeys(code)]
    # Keep the reviewed complete production dependency tree, including indirect
    # action/selection helpers, bound between prepare and every later phase.
    by_path = {str((root / p['path']).resolve()): p for p in tuning_identity['code_pins']}
    for item in code_pins:
        path = str((root / item['path']).resolve())
        if path in by_path:
            need(by_path[path]['sha256'] == item['sha256'], 'Existing production worker changed')
        by_path[path] = item
    code_pins = list(by_path.values())
    identity = {'schema': 'dmv2-rf-tuned-backtest-root/v1', 'project_root': str(root),
        'run_root': str(output), 'configuration': cfg, 'configuration_pin': _pin(root, configuration_path),
        'base_configuration': base, 'source_reference_directory': cfg['source_reference_root'],
        'execution_reference_root_identity_sha256': source_identity,
        'execution_reference_phase_pin': cfg['source_reference_phase_pin'],
        'execution_reference_audit_pin': cfg['source_reference_audit_pin'],
        'tuning_identity_pin': cfg['tuning_identity_pin'], 'tuning_ledger_pin': cfg['tuning_ledger_pin'],
        'tuning_verification_pin': cfg['tuning_verification_pin'], 'forecast_entries': entries,
        'source_pins': list({(p['path'], p['sha256']): p for p in pins}.values()),
        'code_pins': code_pins, 'runtime': runtime(), 'new_estimator_fits': 0,
        'new_preprocessor_fits': 0, 'new_inference_calls': 0, 'financial_2024_plus_read': False,
        'models_or_comparators_changed': False,
        'created_at': dt.datetime.now(dt.timezone.utc).isoformat()}
    output.mkdir(parents=True)
    write_json(output / 'build_identity.json', identity)
    write_json(output / 'read_usage.json', {'schema': 'dmv2-rf-tuned-read-usage/v1',
        'limit_bytes': cfg['resource_limits']['cumulative_compressed_read_bytes'],
        'total_projected_compressed_bytes': 0, 'reservations': []})
    write_json(output / 'audit.json', {'status': 'PREPARED', 'phase': PHASE,
        'build_identity_sha256': canonical_sha256(identity), 'financial_values_decoded': False,
        'models_fitted': 0, 'preprocessors_fitted': 0, 'new_inference_calls': 0})
    return output


def load_run(root, run):
    root, run = Path(root).resolve(), _within(Path(root).resolve() / 'outputs/rf_expanded_tuned_backtest', Path(run).resolve())
    identity = read_json(run / 'build_identity.json'); audit = read_json(run / 'audit.json')
    need(identity['run_root'] == str(run) and identity['project_root'] == str(root)
         and canonical_sha256(identity) == audit['build_identity_sha256'], 'Fresh replay identity changed')
    cfg, base = configuration(root, authenticated(root, identity['configuration_pin']))
    need(cfg == identity['configuration'] and base == identity['base_configuration'], 'Frozen policy changed')
    need(identity['runtime'] == runtime(), 'Replay runtime changed')
    for item in (identity['tuning_identity_pin'], identity['tuning_ledger_pin'], identity['tuning_verification_pin'],
                 *identity['code_pins'], *identity['source_pins']):
        authenticated(root, item)
    need(all(os.environ.get(name) == '1' for name in NUMERIC_ENV), 'Use one numeric-library thread')
    return identity, audit, cfg, base


def validate_forecast_metadata(frame, year, windows):
    need(set(COHORT_COLUMNS) <= set(frame) and not frame.duplicated(['source_security_id', 'signal_date']).any(),
         'Original forecast keys/eligibility are missing or duplicated')
    need(0 < len(frame) <= 12000 and not frame[['source_security_id', 'signal_date']].isna().any().any(),
         'Unbounded/empty/missing forecast keys')
    need(frame.source_security_id.map(lambda x: isinstance(x, str) and bool(x.strip())).all(),
         'Original source IDs must be nonempty stable strings')
    clock = windows.set_index('signal_date')
    expected = clock.loc[clock.outcome_month.str.startswith(str(year) + '-')]
    need(set(frame.signal_date) == set(expected.index), 'Forecast does not cover twelve exact independent signals')
    for column in ('outcome_month', 'outcome_start', 'outcome_end'):
        need(frame[column].equals(frame.signal_date.map(expected[column])), 'Forecast chronology differs from XNYS clock')
    for column in ('prediction_eligible', 'paper_prediction_eligible', 'identity_eligible'):
        need(frame[column].map(lambda x: isinstance(x, (bool, np.bool_))).all(), 'Nonboolean forecast eligibility')
    need(frame.prediction_eligible.equals(frame.paper_prediction_eligible), 'Expanded predictors changed paper prediction cohort')
    need(frame.loc[frame.prediction_eligible, 'intrinio_security_id'].map(lambda x: isinstance(x, str) and bool(x.strip())).all(),
         'Eligible forecast lacks native identity')
    need(not frame.loc[frame.prediction_eligible].duplicated(['intrinio_security_id', 'signal_date']).any(),
         'Eligible source IDs collide on the same dated native route')


def forecast_footer_plan(identity):
    plan, total = [], 0
    for entry in identity['forecast_entries']:
        path = entry['predictions_pin']['path']; old = entry['original_rf_predictions_pin']['path']
        pf, opf = pq.ParquetFile(path), pq.ParquetFile(old)
        fields = set(pf.schema_arrow.names)
        columns = [c for c in SAFE_COLUMNS if c in fields]
        need(set(COHORT_COLUMNS) <= set(columns) and set(PROBABILITIES) <= fields,
             'Selected forecasts lack safe metadata/probability schema')
        need(pf.metadata.num_rows <= 12000 and opf.metadata.num_rows == pf.metadata.num_rows,
             'Forecast original-row counts differ')
        projected = (_projected_bytes(path, columns) + _projected_bytes(path, PROBABILITIES)
                     + _projected_bytes(old, COHORT_COLUMNS))
        plan.append({**entry, 'metadata_columns': columns, 'projected_compressed_bytes': projected})
        total += projected
    return plan, total


def load_forecasts(plan, windows):
    frames = []
    for entry in plan:
        year, path = entry['outer_year'], entry['predictions_pin']['path']
        metadata = pq.ParquetFile(path).read(columns=entry['metadata_columns']).to_pandas()
        validate_forecast_metadata(metadata, year, windows)
        old = pq.ParquetFile(entry['original_rf_predictions_pin']['path']).read(columns=list(COHORT_COLUMNS)).to_pandas()
        validate_forecast_metadata(old, year, windows)
        pd.testing.assert_frame_equal(metadata.loc[:, list(COHORT_COLUMNS)], old.loc[:, list(COHORT_COLUMNS)], check_dtype=False)
        probability = pq.ParquetFile(path).read(columns=list(PROBABILITIES)).to_pandas()
        need(len(probability) == len(metadata), 'Probability rows do not match guarded metadata')
        for name in PROBABILITIES:
            metadata[name] = probability[name].to_numpy()
        # No target, realized return, daily inventory or fitted matrix is read.
        metadata['fit_year'] = year
        for name in LINEAGE_COLUMNS[1:]: metadata[name] = entry[name]
        need(metadata.training_label_observed_through.le(metadata.signal_date).all(), 'Training target spills beyond signal')
        frames.append(metadata)
    result = pd.concat(frames, ignore_index=True)
    need(len(result) == 43655 and result.outcome_month.nunique() == 48, 'Full original forecast cohort changed')
    need(not result.duplicated(['source_security_id', 'signal_date']).any(), 'Annual forecasts overlap')
    return result


def daily_footer_plan(root, identity):
    source = Path(root) / identity['source_reference_directory']
    plans, metadata_bytes, financial_bytes, rows = [], 0, 0, 0
    for year in YEARS:
        directory = source / 'years' / str(year)
        audit = read_json(directory / 'audit.json')
        path = authenticated(directory, audit['artifacts']['daily_prices_raw'])
        pf = pq.ParquetFile(path)
        names = set(pf.schema_arrow.names)
        need(set(ACTION_PRICE_COLUMNS) <= names, 'Daily paired execution columns missing')
        metadata = [x for x in QUOTE_METADATA_COLUMNS if x in names and x in ACTION_PRICE_COLUMNS]
        # _price_plan reads initial keys, then metadata. Financial fields are a
        # separate later materialization and are not charged during preflight.
        meta_bytes = _projected_bytes(path, ('trading_date', 'intrinio_security_id')) + _projected_bytes(path, metadata)
        value_bytes = _projected_bytes(path, ACTION_PRICE_COLUMNS)
        rows += pf.metadata.num_rows; metadata_bytes += meta_bytes; financial_bytes += value_bytes
        plans.append({'year': year, 'path': str(path), 'rows': pf.metadata.num_rows,
            'metadata_columns': metadata, 'metadata_projected_compressed_bytes': meta_bytes,
            'financial_projected_compressed_bytes': value_bytes})
    need(rows == 947517 and rows <= 1000000, 'Reviewed daily annual row scope changed')
    return plans, metadata_bytes, financial_bytes


def execution_context(root, identity, base):
    cfg = identity['configuration']
    directory, context, old_phase, old_identity = _base_reference(root, cfg)
    sessions, windows, clock = walkforward_clock()
    need(len(sessions) == 1006 and len(windows) == 48, 'Frozen XNYS execution clock changed')
    pins = list(identity['source_pins'])
    registry = old_phase['event_registry']
    need(registry['build_identity_sha256'] == identity['execution_reference_root_identity_sha256']
         and len(registry['events']) == 9 and registry['supplemental_quote_sources'] == [],
         'Original event or supplement scope changed')
    validate_current_events(registry['events'], sessions)
    historical = _historical_registry(root, base, pins)
    need(historical == old_phase['historical_retirements'] and len(historical) == 11, 'Preperiod retirements changed')
    handover, handover_pin, records, approval = _identity_registry(root,
        Path(root) / old_phase['identity_handover_registry_pin']['path'],
        identity['execution_reference_root_identity_sha256'], sessions,
        dict(zip(pd.to_datetime(windows.execution_date), pd.to_datetime(windows.signal_date))), pins,
        decision_path=Path(root) / old_phase['identity_handover_approval']['decision_pin']['path'])
    need(handover == old_phase['identity_handover_registry'] and approval == old_phase['identity_handover_approval'],
         'Original PENN evidence/decision changed')
    return {'sessions': sessions, 'windows': windows, 'clock': clock, 'event_registry': registry,
            'historical_retirements': historical, 'identity_handover_registry': handover,
            'identity_handover_registry_pin': handover_pin, 'identity_handovers': records,
            'identity_handover_approval': approval, 'source_pins': pins}


def preflight(root, run_directory):
    root, run = Path(root).resolve(), Path(run_directory).resolve()
    identity, root_audit, cfg, base = load_run(root, run)
    directory = run / 'preflight'
    need(not directory.exists() and root_audit['status'] == 'PREPARED', 'Preserve completed or interrupted preflight')
    started = time.monotonic()
    with exclusive(run), threadpool_limits(limits=1):
        resources(run, cfg, started, 300, launch=True)
        directory.mkdir()
        write_json(directory / 'audit.json', {'status': 'STARTED', 'phase': PHASE,
            'parent_identity_sha256': canonical_sha256(identity)})
        try:
            context = execution_context(root, identity, base)
            forecasts_plan, forecast_bytes = forecast_footer_plan(identity)
            daily_plan, metadata_bytes, financial_bytes = daily_footer_plan(root, identity)
            usage = reserve_reads(run, 'preflight', forecast_bytes + metadata_bytes)
            forecasts = load_forecasts(forecasts_plan, context['windows'])
            _, quote_metadata, old_plan = _price_plan(root, root / identity['source_reference_directory'],
                identity['execution_reference_root_identity_sha256'], context['source_pins'])
            artifacts, summaries, validations = {}, [], []
            for basket in ('class_1_10', 'class_2_9'):
                result = select_walkforward_basket(forecasts, basket=basket,
                    sessions=context['clock']['selection_sessions'], configuration=base)
                check = validate_walkforward_basket(result, forecasts, basket=basket,
                    sessions=context['clock']['selection_sessions'], configuration=base)
                need(len(result.monthly) == 48, 'Selection lost an original prediction month')
                gap = preflight_selected_horizons(result.selections, result.rebalances, quote_metadata,
                    context['sessions'], events=context['event_registry']['events'],
                    historical_retirements=context['historical_retirements'])
                for name, frame in (('original_ledger', result.ledger), ('selected', result.selections),
                                    ('rebalances', result.rebalances), ('monthly', result.monthly)):
                    _save_table(directory, f'selections/{ARM}/{basket}/{name}.parquet', frame, artifacts)
                for name, frame in gap.items():
                    if name != 'summary': _save_table(directory, f'coverage/{basket}/{name}.parquet', frame, artifacts)
                summaries.append({'arm': ARM, 'basket': basket, **gap['summary'],
                    'zero_breadth_months': int(result.monthly.selected_names_per_side.eq(0).sum())})
                validations.append({'arm': ARM, 'basket': basket, **{k: v for k, v in check.items() if k != 'policy'}})
                resources(run, cfg, started, 300)
            phase = {**{k: context[k] for k in ('clock', 'event_registry', 'historical_retirements',
                'identity_handover_registry', 'identity_handover_registry_pin', 'identity_handover_approval', 'source_pins')},
                'schema': 'dmv2-rf-tuned-preflight/v1', 'phase': PHASE,
                'parent_identity_sha256': canonical_sha256(identity),
                'execution_reference_root_identity_sha256': identity['execution_reference_root_identity_sha256'],
                'source_reference_directory': identity['source_reference_directory'],
                'execution_reference_phase_pin': identity['execution_reference_phase_pin'],
                'configuration_sha256': canonical_sha256(cfg), 'basket_policy': BALANCED_BREADTH_POLICY,
                'forecast_entries': identity['forecast_entries'], 'forecast_plan': forecasts_plan,
                'read_plan': {'annuals': daily_plan, 'daily_metadata_bytes': metadata_bytes,
                    'daily_financial_bytes': financial_bytes, 'forecast_bytes': forecast_bytes,
                    'preflight_reserved_bytes': forecast_bytes + metadata_bytes, 'candidate_rows': 947517,
                    'supplemental_execution_quotes': old_plan['supplemental_execution_quotes'] if 'supplemental_execution_quotes' in old_plan else {'sources': 0, 'candidate_rows': 0}},
                'models_fitted': False, 'preprocessing_fitted': False, 'new_inference_calls': 0,
                'performance_computed': False, 'financial_2024_plus_read': False,
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
                'error_type': type(error).__name__, 'message': str(error)})
            raise


def _saved_table_plan(directory, artifacts):
    paths, projected = {}, 0
    for basket in ('class_1_10', 'class_2_9'):
        for name in ('original_ledger', 'selected', 'rebalances', 'monthly'):
            rel = f'selections/{ARM}/{basket}/{name}.parquet'
            path = authenticated(directory, artifacts[rel])
            projected += _projected_bytes(path, pq.ParquetFile(path).schema_arrow.names)
            paths[(basket, name)] = path
    validation = authenticated(directory, artifacts['selection_validation.parquet'])
    projected += _projected_bytes(validation, pq.ParquetFile(validation).schema_arrow.names)
    paths[('report', 'selection_validation')] = validation
    return paths, projected


def backtest(root, run_directory, *, allow_withheld_paths=False):
    need(type(allow_withheld_paths) is bool, 'Explicit withheld-path scope required')
    root, run = Path(root).resolve(), Path(run_directory).resolve()
    identity, root_audit, cfg, base = load_run(root, run)
    directory, pre = run / 'backtest', run / 'preflight'
    need(not directory.exists(), 'Preserve all existing numerical attempts; no automatic retry')
    audit = read_json(pre / 'audit.json'); phase = read_json(pre / 'build_identity.json')
    need(audit['status'] == 'COMPLETE' and audit['build_identity_sha256'] == canonical_sha256(identity)
         and audit['phase_identity_sha256'] == canonical_sha256(phase), 'Exact completed preflight required')
    for item in audit['artifacts'].values(): authenticated(pre, item)
    unresolved = any(not row['ready_for_accounting'] or row['zero_breadth_months'] for row in audit['summary'])
    need(not unresolved or allow_withheld_paths, 'Coverage review required; preserve selections and use explicitly reviewed fail-closed paths')
    started = time.monotonic(); peak = 0
    with exclusive(run), threadpool_limits(limits=1):
        resources(run, cfg, started, 900, launch=True)
        directory.mkdir()
        write_json(directory / 'audit.json', {'status': 'STARTED', 'phase': PHASE,
            'build_identity_sha256': canonical_sha256(identity)})
        try:
            paths, selection_bytes = _saved_table_plan(pre, audit['artifacts'])
            daily_plan, metadata_bytes, financial_bytes = daily_footer_plan(root, identity)
            reserve_reads(run, 'backtest', metadata_bytes + financial_bytes + selection_bytes)
            context = execution_context(root, identity, base)
            _, metadata, _ = _price_plan(root, root / identity['source_reference_directory'],
                identity['execution_reference_root_identity_sha256'], context['source_pins'])
            prices = pd.concat([pq.ParquetFile(p['path']).read(columns=list(ACTION_PRICE_COLUMNS)).to_pandas()
                                for p in daily_plan], ignore_index=True)
            need(prices[['trading_date', 'intrinio_security_id']].equals(metadata[['trading_date', 'intrinio_security_id']]),
                 'Quote keys changed between guarded metadata and financial materialization')
            run_phase = {**phase, 'schema': 'dmv2-rf-tuned-backtest-phase/v1',
                'preflight_audit_pin': _pin(root, pre / 'audit.json'),
                'allow_withheld_paths_after_explicit_review': allow_withheld_paths,
                'preflight_before_performance': True, 'performance_computed': True,
                'tuning_verification_pin': identity['tuning_verification_pin'],
                'code_pins': identity['code_pins'], 'runtime': identity['runtime']}
            write_json(directory / 'build_identity.json', run_phase)
            artifacts = {'build_identity.json': _pin(directory, directory / 'build_identity.json')}
            comparison, annual, failures, actions, orders, coverage = [], [], [], [], [], []
            for basket in ('class_1_10', 'class_2_9'):
                tables = {name: pq.read_table(paths[(basket, name)]).to_pandas()
                          for name in ('original_ledger', 'selected', 'rebalances', 'monthly')}
                coverage.append(tables['monthly'].assign(arm=ARM, basket=basket))
                for name, frame in tables.items(): _save_table(directory, f'selections/{ARM}/{basket}/{name}.parquet', frame, artifacts)
                for variant, stop in base['backtest']['variants'].items():
                    for cost in base['backtest']['cost_bps']:
                        observed = resources(run, cfg, started, 900); peak = max(peak, observed['process_tree_rss_bytes'])
                        result = run_walkforward_accounting_v2(tables['selected'], prices, context['sessions'],
                            rebalances=tables['rebalances'], terminal_date='2023-12-29',
                            events=context['event_registry']['events'], historical_retirements=context['historical_retirements'],
                            identity_handovers=context['identity_handovers'],
                            policy=DailyAccountingPolicy(cost_bps=cost, borrow_annual_bps=0,
                                required_names_per_side=None, asset_stop_fraction=stop))
                        row, monthly = summarize_daily(result, arm=ARM, basket=basket, variant=variant,
                            cost_bps=cost, expected_sessions=1006)
                        need(len(monthly) == 48, 'Daily replay lost an outcome month')
                        row.update(outcome_start='2020-01-01', outcome_end='2023-12-31', outcome_months=48,
                            terminal_date='2023-12-29', annual_liquidation=False, annual_model_fits=4)
                        comparison.append(row)
                        annual.append(yearly_breakdown(result, arm=ARM, basket=basket, variant=variant, cost_bps=cost))
                        rel = f'arms/{ARM}/{basket}/{variant}/cost_{cost:02d}bps'
                        for name, frame in result.tables.items(): _save_table(directory, rel + '/' + name + '.parquet', frame, artifacts)
                        _save_table(directory, rel + '/monthly.parquet', monthly, artifacts)
                        if not result.failures.empty: failures.append(result.failures.assign(arm=ARM, basket=basket, variant=variant, cost_bps=cost))
                        if not result.action_events.empty: actions.append(result.action_events.assign(arm=ARM, basket=basket, variant=variant, cost_bps=cost))
                        if not result.orders.empty: orders.append(result.orders.assign(arm=ARM, basket=basket, variant=variant, cost_bps=cost))
                        observed = resources(run, cfg, started, 900); peak = max(peak, observed['process_tree_rss_bytes'])
                        print(json.dumps({'phase': PHASE, 'completed_scenarios': len(comparison),
                            'expected_scenarios': 16, 'basket': basket, 'variant': variant,
                            'cost_bps': cost, 'status': result.status}), flush=True)
                        del result
            need(len(comparison) == 16, 'Incomplete sixteen-scenario grid')
            reports = {'comparison': pd.DataFrame(comparison), 'yearly': pd.concat(annual, ignore_index=True),
                'selection_coverage': pd.concat(coverage, ignore_index=True),
                'selection_validation': pq.read_table(paths[('report', 'selection_validation')]).to_pandas(),
                'failures': pd.concat(failures, ignore_index=True) if failures else pd.DataFrame(columns=['date', 'reason', 'arm', 'basket', 'variant', 'cost_bps']),
                'action_coverage': pd.concat(actions, ignore_index=True) if actions else pd.DataFrame(columns=['event_id', 'arm', 'basket', 'variant', 'cost_bps']),
                'order_coverage': pd.concat(orders, ignore_index=True) if orders else pd.DataFrame(columns=['status', 'arm', 'basket', 'variant', 'cost_bps'])}
            for name, frame in reports.items(): _save_table(directory, name + '.parquet', frame, artifacts)
            for pin in identity['source_pins']: authenticated(root, pin)
            write_json(directory / 'audit.json', {'status': 'COMPLETE', 'phase': PHASE,
                'implementation_revision': IMPLEMENTATION_REVISION, 'build_identity_sha256': canonical_sha256(identity),
                'phase_identity_sha256': canonical_sha256(run_phase), 'artifacts': artifacts,
                'scenarios': 16, 'sessions': 1006, 'outcome_months': 48,
                'complete_proxy_paths': sum(x['complete_proxy_path'] for x in comparison),
                'failed_closed_paths': sum(not x['complete_proxy_path'] for x in comparison),
                'models_fitted': False, 'preprocessing_fitted': False, 'new_inference_calls': 0,
                'financial_2024_plus_read': False, 'source_identity_promoted': False,
                'source_pins_unchanged': True, 'old_runs_and_pointers_changed': False,
                'preflight_before_performance': True, 'peak_rss_bytes': peak,
                'execution_seconds': time.monotonic() - started, 'RET': 'not_evaluated',
                'annual_liquidation': False}, replace=True)
            write_json(run / 'audit.json', {**root_audit, 'status': 'COMPLETE_PENDING_INDEPENDENT_VERIFICATION',
                'backtest_audit_pin': _pin(root, directory / 'audit.json'), 'financial_values_decoded': True}, replace=True)
            return directory
        except BaseException as error:
            write_json(directory / 'failure.json', {'status': 'FAILED_PRESERVED_NO_RETRY',
                'error_type': type(error).__name__, 'message': str(error)})
            raise
