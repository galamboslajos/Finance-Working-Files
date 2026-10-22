#!/usr/bin/env python3
"""Independent zero-fit checks for the fixed unstopped-target comparison.

Planning hashes files and reads Parquet footers only. Verification consumes an
explicit frozen derived-read plan. No production selector, fit or preprocessor
is used as an expected answer. Existing independently implemented cash/units
verification is reused with an exact helper pin. Source identity, quote quality
and entitlement certification are outside this proof.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import ast
import struct
import time
import datetime as dt
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

HEADS = ('endpoint', 'cash_long', 'cash_short', 'utility_long', 'utility_short')
YEARS = (2020, 2021, 2022, 2023)
CUTOFFS = {2020: '2019-12-31', 2021: '2020-12-31', 2022: '2021-12-31', 2023: '2022-12-30'}
WINNERS = {2020: (12, 20), 2021: (8, 20), 2022: (12, 100), 2023: (12, 100)}
CLASSES = tuple(range(1, 11))
KEYS = ('source_security_id', 'signal_date')
EXCLUDED = {'sec_gjP4xM', 'sec_zmYNnz'}
MAX_DERIVED_BYTES = 16 * 1024 ** 3
SCHEMA = 'dmv2-unstopped-target-independent-verification/v4'


def need(ok, message):
    if not bool(ok):
        raise ValueError(message)


def canonical(value):
    return 'sha256:' + hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(',', ':'), allow_nan=False, ensure_ascii=False).encode()).hexdigest()


def canonical_hex(value):
    """Backtest V1 logical hashes are strict plain hex; model hashes stay prefixed."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        allow_nan=False, ensure_ascii=False).encode()).hexdigest()


def pin(path):
    path = Path(path).resolve()
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 ** 2), b''):
            h.update(b)
    return {'path': str(path), 'size_bytes': path.stat().st_size, 'sha256': h.hexdigest()}


def check(item):
    need(pin(item['path']) == {k: item[k] for k in ('path', 'size_bytes', 'sha256')},
         'Changed pinned verification input: ' + item['path'])
    return Path(item['path'])


def load_json(path):
    return json.loads(Path(path).read_text())


def save_new(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False); f.write('\n')
    return pin(path)


def day(value, *, protected=True):
    need(type(value) is str and dt.date.fromisoformat(value).isoformat() == value,
         'Canonical date required')
    need(not protected or value < '2024-01-01', 'Protected financial date')
    return value


def shift(month, offset):
    need(type(month) is str and len(month) == 7 and dt.date.fromisoformat(month + '-01').strftime('%Y-%m') == month,
         'Canonical outcome month required')
    y, m = map(int, month.split('-')); n = y * 12 + m - 1 + offset
    return f'{n // 12:04d}-{n % 12 + 1:02d}'


def same(actual, expected, message, tolerance=3e-12):
    if isinstance(expected, dict):
        need(isinstance(actual, dict) and set(actual) == set(expected), message + ' keys')
        for k, v in expected.items(): same(actual[k], v, message + '/' + str(k), tolerance)
    elif isinstance(expected, (list, tuple)):
        need(isinstance(actual, (list, tuple)) and len(actual) == len(expected), message + ' length')
        for i, v in enumerate(expected): same(actual[i], v, message + '/' + str(i), tolerance)
    elif isinstance(expected, float):
        need(type(actual) in (float, int) and not isinstance(actual, bool) and math.isfinite(actual)
             and abs(actual - expected) <= tolerance * max(1., abs(expected)), message)
    else:
        need(actual == expected and (not isinstance(expected, bool) or type(actual) is bool), message)


def key_rows(rows):
    keys = []
    for row in rows:
        sid, signal = (row[k] for k in KEYS)
        need(type(sid) is str and sid and sid.strip() == sid, 'Original stable source ID required')
        day(signal); keys.append((sid, signal))
    need(len(set(keys)) == len(keys), 'Duplicate original source/signal key')
    return keys


def verify_common_mask(rows, year):
    """Derive the shared mask from all original training keys and five labels.

    The complete ledger is checked before any matrix/label numerical conversion.
    Actual source-label equivalence is a separate caller check against pinned
    corrected endpoint/path target files, rather than an assumption here.
    """
    need(year in YEARS and rows, 'One bounded nonempty annual training ledger required')
    keys = key_rows(rows); mask = []
    for row in rows:
        month = row['outcome_month']; shift(month, 0)
        need(f'{year-10}-01' <= month <= f'{year-1}-12', 'Training outcome escaped rolling ten years')
        need(shift(row['signal_date'][:7], 1) == month, 'Training signal/outcome misalignment')
        need(type(row['prediction_eligible']) is bool and type(row['common_fit_mask']) is bool,
             'Explicit original admission and common supervised mask required')
        valid = row['prediction_eligible']
        for head in HEADS:
            available = row[head + ('__' if head + '__target_available' in row else '_') + 'target_available']; cls = row[head + ('__' if head + '__target_class' in row else '_') + 'target_class']
            completion = row[head + ('__' if head + '__label_available_at' in row else '_') + 'label_available_at']; day(completion, protected=False)
            need(type(available) is bool, 'Explicit per-head label availability required')
            if available:
                need(type(cls) in (int, float) and not isinstance(cls, bool) and math.isfinite(cls)
                     and int(cls) == cls and int(cls) in CLASSES, 'Noncanonical target class')
            else:
                need(cls is None or isinstance(cls, float) and math.isnan(cls), 'Missing target was imputed')
            need(completion > row['signal_date'], 'Target completed before its prediction signal')
            valid = valid and available and completion <= CUTOFFS[year]
        need(row['common_fit_mask'] is valid, 'Common supervised mask differs from actual completion gate')
        mask.append(valid)
    need(any(mask), 'Empty common supervised mask')
    return {'keys': keys, 'mask': mask, 'fit_rows': sum(mask),
        'fit_keys_sha256': canonical([list(k) for k, take in zip(keys, mask) if take]),
        'cutoff': CUTOFFS[year], 'train_start': f'{year-10}-01-01', 'train_end': f'{year-1}-12-31'}


def verify_prediction_metadata(rows, year):
    keys = key_rows(rows)
    months = set()
    for row in rows:
        month = row['outcome_month']; shift(month, 0)
        need(month.startswith(str(year) + '-') and shift(row['signal_date'][:7], 1) == month,
             'Forecast escaped its fixed annual window')
        need(type(row['prediction_eligible']) is bool, 'Forecast admission must be signal-only Boolean')
        if 'paper_prediction_eligible' in row:
            need(row['prediction_eligible'] is row['paper_prediction_eligible'], 'Future labels changed forecast cohort')
        months.add(month)
    need(months == {f'{year}-{m:02d}' for m in range(1, 13)}, 'Original forecast monthly grid changed')
    return keys


def independent_payoff(target_rows, signal, *, value_field, short_endpoint=False):
    """Reconstruct exact equal-calendar-month means; common path completion lag."""
    day(signal)
    last = shift(signal[:7], -1); months = [shift(last, d) for d in range(-119, 1)]
    groups = {(m, c): [] for m in months for c in CLASSES}
    keys = set()
    for row in target_rows:
        month = row['outcome_month']
        if month not in months:
            continue
        key = row['source_security_id'], row['signal_date']
        need(key not in keys, 'Duplicate original key in one payoff head'); keys.add(key)
        need(shift(row['signal_date'][:7], 1) == month, 'Payoff target signal/month mismatch')
        completion = row['label_available_at']; day(completion, protected=False)
        need(type(row['target_available']) is bool and type(row['prediction_eligible']) is bool,
             'Explicit payoff availability/admission required')
        if row['target_available']:
            need(row['prediction_eligible'] and completion <= signal, 'Uncompleted or ineligible payoff consumed')
            cls, value = row['target_class'], row[value_field]
            need(type(cls) is int and cls in CLASSES and type(value) in (int, float)
                 and not isinstance(value, bool) and math.isfinite(value), 'Invalid realized payoff')
            groups[(month, cls)].append(-float(value) if short_endpoint else float(value))
    monthly = [{'outcome_month': m, 'class_id': c, 'member_count': len(groups[(m, c)]),
        'mean_payoff': math.fsum(v / len(groups[(m, c)]) for v in groups[(m, c)]) if groups[(m, c)] else None,
        'status': 'available' if groups[(m, c)] else 'empty_or_unavailable_calendar_class'}
        for m in months for c in CLASSES]
    available = all(row['member_count'] for row in monthly)
    payoffs = [math.fsum(row['mean_payoff'] / 120 for row in monthly if row['class_id'] == c)
        for c in CLASSES] if available else None
    return {'signal_date': signal, 'first_outcome_month': months[0], 'last_outcome_month': last,
        'monthly': monthly, 'available': bool(available), 'payoffs': payoffs}


def probability_vector(row, prefix='probability_'):
    values = [row[prefix + str(c)] for c in CLASSES]
    need(all(type(v) in (float, int) and not isinstance(v, bool) and math.isfinite(v) and v >= 0 for v in values)
         and abs(math.fsum(values) - 1.) <= 1e-8, 'Invalid saved probability simplex; no normalization allowed')
    return [float(v) for v in values]


def independent_selection(rows, *, arm, basket, long_payoffs=None, short_payoffs=None):
    """Pure signal-only choices, including directional overlap and stable ties.

    Input rows expose two aligned probability vectors named long_probability_N
    and short_probability_N. Endpoint's short vector is the same original vector;
    its best/second short classes are10/9. New short-model classes are1/2.
    """
    arm = 'endpoint' if arm == 'endpoint_matched_clock' else arm
    basket = {'best_mode': 'class_1_10', 'second_mode': 'class_2_9'}.get(basket, basket)
    need(arm in ('endpoint', 'cash', 'utility') and basket in ('score', 'class_1_10', 'class_2_9'),
         'Unknown frozen target arm or separate basket')
    keys = key_rows(rows)
    need(len({k[1] for k in keys}) == 1, 'Selection must be one original monthly cohort')
    eligible = []
    for row in rows:
        need(type(row['prediction_eligible']) is bool, 'Unknown forecast admission')
        if not row['prediction_eligible']:
            continue
        lp, sp = probability_vector(row, 'long_probability_'), probability_vector(row, 'short_probability_')
        if row['intrinio_security_id'] in EXCLUDED:
            continue
        sid = row['source_security_id']
        mode_l = 1 + max(range(10), key=lambda i: lp[i]); mode_s = 1 + max(range(10), key=lambda i: sp[i])
        eligible.append((sid, lp, sp, mode_l, mode_s))
    if basket == 'score':
        if long_payoffs is None or short_payoffs is None:
            return {'long': [], 'short': [], 'n': 0, 'status': 'withheld_incomplete_120_month_class_grid'}
        need(len(long_payoffs) == len(short_payoffs) == 10
             and all(type(x) in (float, int) and math.isfinite(x) for x in [*long_payoffs, *short_payoffs]),
             'Complete finite causal payoff vectors required')
        long_rank = sorted(eligible, key=lambda r: (-math.fsum(p * m for p, m in zip(r[1], long_payoffs)), r[0]))
        n = min(10, len(eligible) // 2)
        chosen_long = [r[0] for r in long_rank[:n]]
        short_rank = sorted((r for r in eligible if r[0] not in chosen_long),
            key=lambda r: (-math.fsum(p * m for p, m in zip(r[2], short_payoffs)), r[0]))
    else:
        lc = 1 if basket == 'class_1_10' else 2
        sc = (11 - lc) if arm == 'endpoint' else lc
        long_rank = sorted((r for r in eligible if r[3] == lc), key=lambda r: (-r[1][lc - 1], r[0]))
        short_candidates = [r for r in eligible if r[4] == sc]
        n = 0; chosen_long = []
        for candidate_n in range(min(10, len(long_rank), len(short_candidates)), 0, -1):
            attempt = [r[0] for r in long_rank[:candidate_n]]
            if sum(r[0] not in attempt for r in short_candidates) >= candidate_n:
                n, chosen_long = candidate_n, attempt; break
        short_rank = sorted((r for r in short_candidates if r[0] not in chosen_long), key=lambda r: (-r[2][sc - 1], r[0]))
    chosen_short = [r[0] for r in short_rank[:n]]
    need(len(chosen_long) == len(chosen_short) == n and not set(chosen_long) & set(chosen_short),
         'Unbalanced or overlapping independent baskets')
    return {'long': chosen_long, 'short': chosen_short, 'n': n,
        'status': 'available' if n else 'withheld_zero_distinct_candidates',
        'long_weight': .5 / n if n else None, 'short_weight': -.5 / n if n else None}


def load_independent(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def footer(item):
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(check(item)); names = pf.schema_arrow.names
    cost = sum(pf.metadata.row_group(g).column(c).total_compressed_size
        for g in range(pf.metadata.num_row_groups) for c in range(pf.metadata.row_group(g).num_columns))
    return {'pin': item, 'columns': names, 'rows': pf.metadata.num_rows,
        'projected_compressed_bytes': cost}


def make_plan(run, phase, output):
    """Inventory only immutable completed generated artifacts; no value decode."""
    root = Path(run).resolve(); need(phase in ('models', 'backtest'), 'Explicit verifier phase required')
    need(root.is_dir(), 'Completed run directory required')
    subtree = root / ('folds' if phase == 'models' else 'backtest')
    need(subtree.is_dir(), 'Completed requested phase directory missing')
    paths = sorted(p for p in subtree.rglob('*') if p.is_file() and p.suffix in ('.json', '.parquet', '.npy', '.joblib'))
    need(paths, 'No immutable generated artifacts')
    artifacts = []; cost = 0
    for path in paths:
        item = pin(path)
        if path.suffix == '.parquet':
            spec = footer(item); cost += 2 * spec['projected_compressed_bytes']
        elif path.suffix in ('.npy', '.joblib'):
            spec = {'pin': item, 'format': path.suffix[1:]}; cost += item['size_bytes']
        else:
            spec = {'pin': item, 'format': 'json'}; cost += item['size_bytes']
        artifacts.append(spec)
    need(cost <= MAX_DERIVED_BYTES, 'Independent derived-read reservation exceeds16GiB')
    body = {'schema': SCHEMA, 'status': 'SEALED_BEFORE_FINANCIAL_READ', 'phase': phase,
        'run_directory': str(root), 'worker_pin': pin(__file__), 'artifacts': artifacts,
        'derived_read_reservation_bytes': cost, 'derived_read_limit_bytes': MAX_DERIVED_BYTES,
        'new_estimator_fits': 0, 'source_prices_or_feature_formulas_recomputed': False,
        'financial_2024_plus_authorized': False, 'automatic_retry': False}
    return save_new(output, body)


# Numerical adapters are below. They authenticate every planned artifact before
# decoding and retain separate proof scope for source inputs and saved matrices.

def clean_pin(item):
    return {k: item[k] for k in ('path', 'size_bytes', 'sha256')}


def array_payload(item):
    """Hash raw NPY payload independently, without loading financial arrays."""
    path = check(item)
    with path.open('rb') as f:
        need(f.read(6) == b'\x93NUMPY', 'Invalid saved matrix magic')
        version = tuple(f.read(2)); need(version in ((1, 0), (2, 0), (3, 0)), 'Unsupported NPY version')
        n = struct.unpack('<H' if version == (1, 0) else '<I', f.read(2 if version == (1, 0) else 4))[0]
        header = ast.literal_eval(f.read(n).decode('latin1' if version != (3, 0) else 'utf8'))
        need(set(header) == {'descr', 'fortran_order', 'shape'} and header['fortran_order'] is False,
             'Saved matrix must be a plain non-object contiguous array')
        need(type(header['descr']) is str and header['descr'][1:] in ('f4', 'f8', 'i8'), 'Unexpected NPY numeric dtype')
        h = hashlib.sha256(); amount = 0
        for block in iter(lambda: f.read(1024 ** 2), b''):
            amount += len(block); h.update(block)
    expected = math.prod(header['shape']) * int(header['descr'][2:])
    need(amount == expected, 'Saved array payload length changed')
    return {'shape': list(header['shape']), 'dtype': header['descr'],
        'array_sha256': 'sha256:' + h.hexdigest(), 'payload_bytes': amount}


def array_sha(array):
    import numpy as np
    a = np.ascontiguousarray(array)
    return 'sha256:' + hashlib.sha256(memoryview(a).cast('B') if a.size else b'').hexdigest()


def _all_specs(plan):
    for annual in plan['annuals'].values():
        for k in ('endpoint_metadata', 'endpoint_values'):
            yield annual[k]
        for month in annual['path_months']:
            yield month['metadata']; yield month['values']


def plan_verification(model_plan_path, backtest_plan_path_or_None, output):
    """Seal exact generated/source-target reads after completion, without values."""
    mp = pin(model_plan_path); model = load_json(check(mp))
    need(model['status'] == 'SEALED_BEFORE_FINANCIAL_READ' and model['heads'] == list(HEADS)
         and model['expected_fits'] == 20 and model['expected_preprocessors'] == 4
         and model['financial_2024_plus_allowed'] is False, 'Fixed20-fit comparison plan required')
    need(model['plan_sha256'] == canonical({k: v for k, v in model.items() if k != 'plan_sha256'}), 'Model plan fingerprint changed')
    run = Path(model['output_directory']).resolve(); phase = 'models' if backtest_plan_path_or_None is None else 'backtest'
    backtest = None if phase == 'models' else load_json(backtest_plan_path_or_None)
    subtree = run / 'folds' if phase == 'models' else Path(backtest['output_directory']).resolve()
    need(subtree.is_dir(), 'Completed phase output missing')
    helpers = {'model_math': pin(Path(__file__).parent / 'corrected_refit_20261021/verify_models.py'),
        'accounting_math': pin(Path(__file__).parent / 'verify_walkforward_backtest_v2.py'),
        'overlay_math': pin(Path(__file__).parent / 'verify_ret_repair_backtest.py'),
        'preflight_math': pin(Path(__file__).parent / 'corrected_refit_20261021/backtest_verification_math_v2.py')}
    paths = sorted(p for p in subtree.rglob('*') if p.is_file() and p.suffix in ('.parquet', '.json', '.npy', '.joblib'))
    need(paths, 'Completed generated phase artifacts missing')
    inventory = {}; cost = 0
    for path in paths:
        item = pin(path)
        if path.suffix == '.parquet':
            spec = footer(item); cost += 2 * spec['projected_compressed_bytes']
        else:
            spec = {'pin': item, 'format': path.suffix[1:]}; cost += item['size_bytes']
        inventory[str(path)] = spec
    source_specs = []
    # Source classes/completion are derived target tables, not prices/features.
    if phase == 'models':
        pairs = None
        for spec in _all_specs(model):
            check(spec['pin']); source_specs.append(spec)
            cost += spec['projected_compressed_bytes']
    else:
        need(backtest['model_plan_sha256'] == model['plan_sha256'] and backtest['scenarios'] == 72
             and backtest['economic_months'] == 47, 'Frozen backtest/model scope mismatch')
        forecast_plan = load_json(subtree / 'forecast_read_plan.json')
        pairs = {'endpoint': backtest['endpoint'], 'path_targets': backtest['path_targets'],
            'daily': backtest['daily'], 'repairs': [backtest['repairs']], 'forecasts': forecast_plan['forecasts']}
        for group in pairs.values():
            for pair in group:
                for name in ('metadata', 'values'):
                    spec = pair[name]; check(spec['pin']); source_specs.append(spec)
                    cost += spec['projected_compressed_bytes']
    need(cost <= MAX_DERIVED_BYTES, 'Independent phase requires over16GiB derived reads')
    body = {'schema': SCHEMA, 'status': 'SEALED_BEFORE_FINANCIAL_READ', 'phase': phase,
        'receipt_directory': str(Path(output).resolve().parent),
        'derived_read_bytes': cost, 'model_plan_pin': mp, 'backtest_plan_pin': None if phase == 'models' else pin(backtest_plan_path_or_None),
        'run_directory': str(run), 'worker_pin': pin(__file__), 'helper_pins': helpers,
        'artifacts': inventory, 'source_target_specs': source_specs, 'source_pair_specs': pairs,
        'model_verification_pin': None if phase == 'models' else pin(run / 'verification/models.json'),
        'derived_read_reservation_bytes': cost, 'derived_read_limit_bytes': MAX_DERIVED_BYTES,
        'new_estimator_fits': 0, 'source_prices_or_feature_formulas_recomputed': False,
        'financial_2024_plus_authorized': False, 'automatic_retry': False}
    return save_new(output, body)


def _context(model_plan_path, verification_plan, phase):
    need(verification_plan is not None, 'Sealed independent verification plan required')
    vp = pin(verification_plan); v = load_json(check(vp)); mp = pin(model_plan_path)
    need(v['schema'] == SCHEMA and v['phase'] == phase and v['status'] == 'SEALED_BEFORE_FINANCIAL_READ'
         and v['model_plan_pin'] == mp and v['worker_pin'] == pin(__file__)
         and v['derived_read_reservation_bytes'] <= MAX_DERIVED_BYTES
         and v['financial_2024_plus_authorized'] is False and v['new_estimator_fits'] == 0,
         'Independent plan/code/scope mismatch')
    model = load_json(check(mp)); run = Path(v['run_directory'])
    from importlib.metadata import version
    for package, expected in model['runtime'].items():
        need(version(package) == expected, 'Independent verification runtime changed: ' + package)
    need(run.resolve() == Path(model['output_directory']).resolve(), 'Independent output redirected')
    for item in v['artifacts'].values(): check(item['pin'])
    for item in v['helper_pins'].values(): check(item)
    for item in (model['protocol_pin'], model['catalog_pin'], model['input_verification_pin'],
                 model['target_audit_pin'], model['target_plan_pin'], model['target_verification_pin']): check(item)
    need(load_json(check(model['input_verification_pin']))['status'] == 'PASS'
         and load_json(check(model['target_verification_pin']))['status'] == 'PASS_FULL_FORMULAS_RANKS_CLOCK_AND_ORIGINAL_DENOMINATORS',
         'Inherited corrected-input or numerical-target proof missing')
    return model, run, v, vp


def _projected(spec):
    import pyarrow.parquet as pq
    path = check(spec['pin']); pf = pq.ParquetFile(path)
    names = list(spec['columns'])
    need(set(names) <= set(pf.schema_arrow.names), 'Frozen target projection columns changed')
    cost = sum(pf.metadata.row_group(g).column(c).total_compressed_size
        for g in range(pf.metadata.num_row_groups) for c in range(pf.metadata.row_group(g).num_columns)
        if pf.metadata.row_group(g).column(c).path_in_schema.split('.')[0] in names)
    need(cost == spec['projected_compressed_bytes'], 'Frozen target compressed projection changed')
    t = pf.read(columns=names, use_threads=False)
    need(t.num_rows == spec['rows'], 'Frozen target projection row count changed')
    return t


def _source_targets(model):
    """Independent original-denominator and source-class mapping for all14years."""
    sources = {}
    specs = {'cash_long': ('executable_return', 'long'), 'cash_short': ('executable_return', 'short'),
        'utility_long': ('return_minus_drawdown', 'long'), 'utility_short': ('return_minus_drawdown', 'short')}
    for year in range(2010, 2024):
        a = model['annuals'][str(year)]; windows = {w['signal_date']: w for w in a['windows']}
        meta = _projected(a['endpoint_metadata']).to_pylist(); keys = key_rows(meta)
        need(len(meta) == a['original_rows'] and len(windows) == 12, 'Original annual target denominator changed')
        lookup = {}
        for row in meta:
            w = windows[row['signal_date']]
            need(row['outcome_month'] == w['outcome_month'] and row['outcome_start'] == w['outcome_start']
                 and row['outcome_end'] == w['outcome_end'] and row['outcome_end'] < '2024-01-01', 'Endpoint metadata clock changed')
            need(type(row['prediction_eligible']) is bool and row['prediction_eligible'] is row['paper_prediction_eligible']
                 and type(row['target_available']) is bool, 'Original target admission changed')
            lookup[tuple(row[k] for k in KEYS)] = row
        need(sum(r['prediction_eligible'] for r in meta) == a['prediction_eligible_rows'], 'Original eligible denominator changed')
        # Metadata admission precedes numerical class conversion.
        old = _projected(a['endpoint_values']).to_pylist()
        need(set(key_rows(old)) == set(keys), 'Endpoint classes lost original keys')
        mapped = {h: {} for h in HEADS}
        for row in old:
            k = tuple(row[n] for n in KEYS); m = lookup[k]
            mapped['endpoint'][k] = {'target_class': row['target_class'], 'target_available': m['target_available'],
                'label_available_at': m['outcome_end'], 'prediction_eligible': m['prediction_eligible'], 'outcome_month': m['outcome_month']}
        for monthly in a['path_months']:
            m = _projected(monthly['metadata']).to_pylist(); by_key = {}
            for row in m:
                k = tuple(row[n] for n in KEYS)
                need(k in lookup and row['outcome_month'] == monthly['outcome_month'], 'Path escaped original cohort/month')
                w = windows[row['signal_date']]
                need(row['label_available_at'] == row['exit_date'] == w['exit_date']
                     and row['entry_date'] == w['entry_date'] and row['exit_mark'] == 'open'
                     and row['prediction_eligible'] is lookup[k]['prediction_eligible'], 'Path completion/admission changed')
                need(row['exit_date'] < '2024-01-01' or row['outcome_month'] == '2023-12' and row['target_available'] is False,
                     'Protected endpoint released')
                group_key = (*k, row['target_family'], row['side'], row['path_variant'])
                need(group_key not in by_key, 'Duplicate target directional metadata'); by_key[group_key] = row
            values = _projected(monthly['values']).to_pylist()
            need(len(values) == len(m) == monthly['original_rows'] * 8, 'Path original group population changed')
            for row in values:
                if row['path_variant'] != 'unstopped': continue
                k = tuple(row[n] for n in KEYS)
                group_key = (*k, row['target_family'], row['side'], row['path_variant'])
                need(group_key in by_key, 'Path class key lacks admitted metadata'); md = by_key[group_key]
                for h, pair in specs.items():
                    if (row['target_family'], row['side']) == pair:
                        need(k not in mapped[h], 'Duplicate path class key')
                        mapped[h][k] = {'target_class': row['target_class'], 'target_available': md['target_available'],
                            'label_available_at': md['label_available_at'], 'prediction_eligible': md['prediction_eligible'], 'outcome_month': md['outcome_month']}
            del m, values, by_key
        need(all(set(mapped[h]) == set(keys) for h in HEADS), 'Missing original class head population')
        sources[year] = mapped
    return sources


def _parquet(path, inventory, *, metadata=False):
    import pyarrow.parquet as pq
    path = str(Path(path).resolve()); need(path in inventory, 'Unplanned generated financial read')
    entry = inventory[path]; check(entry['pin'])
    pf = pq.ParquetFile(path); names = entry['columns']
    if metadata:
        names = [n for n in names if n not in ('target_class',) and not n.endswith('__target_class')
            and not n.startswith('probability_') and n not in ('selection_score', 'expected_score_long', 'expected_score_short')]
    return pf.read(columns=names, use_threads=False)


def _source_equivalence(rows, source, *, head=None):
    for r in rows:
        k = tuple(r[n] for n in KEYS)
        need(k in source['endpoint'], 'Copied target key absent from original source')
        need(r['prediction_eligible'] is source['endpoint'][k]['prediction_eligible'], 'Copied future label changed prediction admission')
        heads = (head,) if head is not None else HEADS
        for h in heads:
            m = source[h][k]; prefix = '' if head is not None else h + '__'
            for field in ('target_class', 'target_available', 'label_available_at'):
                actual, expected = r[prefix + field], m[field]
                if actual is None or isinstance(actual, float) and math.isnan(actual):
                    need(expected is None, 'Source target missingness changed')
                else:
                    need(actual == expected, 'Copied target differs from independently pinned source ' + h + '/' + field)


def verify_class_reports(probabilities, labels, months, confusion, mode_rows):
    """Reconstruct full confusion and precision/recall/support by month and class."""
    import numpy as np
    p = np.asarray(probabilities); y = np.asarray(labels, dtype=float)
    mode = np.argmax(p, axis=1) + 1; observed = np.isfinite(y)
    all_periods = [('all', np.ones(len(y), dtype=bool))] + [(m, np.array([v == m for v in months])) for m in sorted(set(months))]
    expected_confusion, expected_modes = [], []
    for period, take in all_periods:
        for actual in CLASSES:
            for predicted in CLASSES:
                expected_confusion.append({'period': period, 'actual_class': actual, 'predicted_class': predicted,
                    'count': int(np.sum(take & observed & (y == actual) & (mode == predicted)))})
        for c in CLASSES:
            predicted = take & (mode == c); observed_predicted = predicted & observed; actual = take & observed & (y == c)
            hits = int(np.sum(observed_predicted & actual)); predicted_n = int(predicted.sum())
            observed_n, actual_n = int(observed_predicted.sum()), int(actual.sum())
            expected_modes.append({'period': period, 'class_id': c, 'prediction_rows': int(take.sum()),
                'predicted_support': predicted_n, 'labelled_predicted_support': observed_n,
                'missing_label_predicted_support': predicted_n - observed_n, 'actual_support': actual_n,
                'true_positive_count': hits, 'precision': hits / observed_n if observed_n else None,
                'recall': hits / actual_n if actual_n else None,
                'status': 'available' if observed_n and actual_n else 'missing_predicted_or_actual_support'})
    same(confusion, expected_confusion, 'Independent full confusion counts')
    same(mode_rows, expected_modes, 'Independent class precision/recall/support')
    return {'confusion_rows': len(expected_confusion), 'class_metric_rows': len(expected_modes)}


def verify_models(model_plan_path, verification_plan=None):
    """Verify20saved RFs and four paired supervised/preprocessing states;0fits."""
    import gc
    import joblib
    import numpy as np
    from threadpoolctl import threadpool_limits
    started = time.monotonic(); model, run, v, vp = _context(model_plan_path, verification_plan, 'models')
    destination = run / 'verification/models.json'; need(not destination.exists(), 'Preserve prior model verification')
    math_worker = load_independent(check(v['helper_pins']['model_math']), '_independent_target_model_math')
    need(len(list(run.glob('folds/*/heads/*/FIT_ATTEMPT.json'))) == 20, 'Exactly20consumed estimator attempts required')
    need(len(list(run.glob('folds/*/prepared/receipt.json'))) == 4, 'Exactlyfour shared preprocessors required')
    sources = _source_targets(model); results = []
    for year in YEARS:
        directory = run / 'folds' / str(year) / 'prepared'; receipt_path = directory / 'receipt.json'
        receipt = load_json(receipt_path); need(receipt['status'] == 'PREPARED' and receipt['year'] == year
            and receipt['plan_sha256'] == model['plan_sha256'] and receipt['preprocessor_fits'] == 1, 'Prepared state identity changed')
        for item in receipt['artifacts'].values(): check(item)
        metadata = _parquet(directory / 'train_ledger.parquet', v['artifacts'], metadata=True).to_pylist()
        need(len(metadata) == sum(model['annuals'][str(y)]['original_rows'] for y in range(year-10, year)), 'Original training denominator changed')
        train = _parquet(directory / 'train_ledger.parquet', v['artifacts']).to_pylist()
        merged_source = {h: {k: r for y in range(year-10, year) for k, r in sources[y][h].items()} for h in HEADS}
        _source_equivalence(train, merged_source)
        mask = verify_common_mask(train, year); fit_keys = [k for k, take in zip(mask['keys'], mask['mask']) if take]
        need(mask['fit_keys_sha256'] == receipt['fit_keys_sha256'] and mask['fit_rows'] == receipt['train_fit_rows'], 'Prepared common key identity changed')
        artifact = receipt['artifacts']; state = load_json(check(artifact['preprocessor']))
        raw_train = np.load(check(artifact['raw_train_X']), mmap_mode='r', allow_pickle=False)
        need(raw_train.dtype == np.float64 and raw_train.shape == (len(fit_keys), 715), 'Exact common-row raw training projection required')
        expected_state = math_worker.derive_preprocessor(raw_train, model['predictor_names'], np.ones(len(fit_keys), dtype=bool),
            fit_keys, model['paper_names'], mask['train_start'], mask['train_end'], mask['cutoff'])
        same(state, expected_state, 'Independent training-only preprocessor')
        need(state['fingerprint'] == receipt['preprocessor_fingerprint'], 'Shared preprocessing fingerprint changed')
        X = math_worker.prepare_matrix(raw_train, model['predictor_names'], state, np.ones(len(fit_keys), dtype=bool))
        payload = array_payload(artifact['train_X'])
        need(payload['shape'] == list(X.shape) and payload['dtype'].endswith('f4')
             and payload['array_sha256'] == array_sha(X) == artifact['train_X']['array_sha256'], 'Prepared training matrix differs from raw train-only reconstruction')
        train_hash = array_sha(X); del raw_train, X
        pred_ledgers = {}
        for head in HEADS:
            path = check(artifact['prediction_ledger_' + head]); meta = _parquet(path, v['artifacts'], metadata=True).to_pylist()
            verify_prediction_metadata(meta, year)
            rows = _parquet(path, v['artifacts']).to_pylist(); _source_equivalence(rows, sources[year], head=head)
            need(set(key_rows(rows)) == set(sources[year]['endpoint']), 'Prediction original population differs from source')
            pred_ledgers[head] = rows
        rows = pred_ledgers['endpoint']; pred_keys = key_rows(rows); selected = np.array([r['prediction_eligible'] for r in rows], dtype=bool)
        need(canonical([list(k) for k, take in zip(pred_keys, selected) if take]) == receipt['prediction_keys_sha256']
             and canonical([list(k) for k in pred_keys]) == receipt['raw_prediction_keys_sha256'], 'Ordered original prediction identity changed')
        raw_prediction = np.load(check(artifact['raw_prediction_X']), mmap_mode='r', allow_pickle=False)
        need(raw_prediction.dtype == np.float64 and raw_prediction.shape == (len(rows), 715), 'Original raw forecast matrix shape changed')
        paper = np.array([model['predictor_names'].index(n) for n in model['paper_names']])
        need(np.array_equal(np.isfinite(raw_prediction[:, paper]).all(axis=1), selected), 'Signal-time Paper15 eligibility differs from raw features')
        prediction_X = math_worker.prepare_matrix(raw_prediction, model['predictor_names'], state, selected)
        payload = array_payload(artifact['prediction_X'])
        need(payload['shape'] == list(prediction_X.shape) and payload['dtype'].endswith('f4')
             and payload['array_sha256'] == array_sha(prediction_X) == artifact['prediction_X']['array_sha256'], 'Prepared forecast matrix differs from shared training-only transform')
        del raw_prediction
        for head in HEADS:
            out = run / 'folds' / str(year) / 'heads' / head
            fit = load_json(out / 'FIT_COMPLETE.json'); pred = load_json(out / 'PREDICT_COMPLETE.json')
            attempt = load_json(out / 'FIT_ATTEMPT.json')
            need(attempt['status'] == 'FIT_CONSUMED' and attempt['estimator_fits'] == 1
                 and attempt['year'] == year and attempt['head'] == head
                 and attempt['plan_sha256'] == model['plan_sha256']
                 and clean_pin(attempt['prepared_pin']) == pin(receipt_path), 'Consumed estimator attempt lineage changed')
            need(fit['status'] == 'FIT_COMPLETE' and fit['estimator_fits'] == 1 and pred['status'] == 'PREDICT_COMPLETE'
                 and pred['estimator_fits'] == 0 and fit['year'] == pred['year'] == year and fit['head'] == pred['head'] == head
                 and fit['plan_sha256'] == pred['plan_sha256'] == model['plan_sha256']
                 and clean_pin(fit['prepared_pin']) == pin(receipt_path) == clean_pin(pred['prepared_pin'])
                 and clean_pin(pred['fit_pin']) == pin(out / 'FIT_COMPLETE.json'), 'Saved fit/prediction lineage changed')
            params = fit['explicit_parameters']; depth, leaf = WINNERS[year]
            need(params == model['folds'][str(year)]['fixed_rf_parameters'], 'Fitted RF parameters differ from sealed annual plan')
            need(params['n_estimators'] == 500 and params['max_features'] == 'sqrt' and params['random_state'] == 1729
                 and params['n_jobs'] == 4 and params['max_depth'] == depth and params['min_samples_leaf'] == leaf
                 and params['class_weight'] is None and params['bootstrap'] is True
                 and fit['parameter_search'] is False and fit['calibration'] is False
                 and fit['validation_labels_supplied_to_estimator'] is False, 'Fixed annual RF contract changed')
            need(fit['fit_effective_parameters'] == params and fit['saved_inference_effective_parameters'] == {**params, 'n_jobs': 1}
                 and fit['native_class_ids'] == fit['probability_class_ids'] == list(CLASSES), 'Effective RF parameters/class map changed')
            y = np.array([r[head + '__target_class'] for r, take in zip(train, mask['mask']) if take], dtype=np.int64)
            yp = array_payload(artifact['train_y_' + head])
            need(yp['array_sha256'] == array_sha(y) == fit['train_labels_sha256'] and fit['train_matrix_sha256'] == train_hash
                 and set(y) == set(CLASSES), 'Fitted target labels/matrix differ from common source classes')
            estimator = joblib.load(check(fit['model']))
            from sklearn.ensemble import RandomForestClassifier
            need(type(estimator) is RandomForestClassifier and len(estimator.estimators_) == 500, 'Saved model is not the fixed500-tree random forest')
            need(estimator.n_features_in_ == prediction_X.shape[1], 'Reloaded RF feature width changed')
            with threadpool_limits(limits=1):
                probabilities = math_worker.infer_probabilities(estimator, 'rf', prediction_X, {**params, 'n_jobs': 1})
            for item in pred['artifacts'].values(): check(item)
            pp = array_payload(pred['artifacts']['probabilities'])
            need(pp['array_sha256'] == array_sha(probabilities), 'Saved RF probabilities differ from exact reload inference')
            forecast_meta = _parquet(out / 'predictions.parquet', v['artifacts'], metadata=True).to_pylist()
            verify_prediction_metadata(forecast_meta, year)
            forecast = _parquet(out / 'predictions.parquet', v['artifacts']).to_pylist()
            need(key_rows(forecast) == key_rows(pred_ledgers[head]), 'Forecast original keys/order changed')
            p_position = 0
            for actual, original in zip(forecast, pred_ledgers[head]):
                for name in original:
                    a, b = actual[name], original[name]
                    need(a == b or a is None and b is None, 'Forecast original ledger metadata/targets changed')
                if original['prediction_eligible']:
                    p = probability_vector(actual)
                    need(np.array_equal(np.array(p), probabilities[p_position]), 'Forecast table probability differs from exact saved inference')
                    p_position += 1
                else:
                    need(all(actual['probability_' + str(c)] is None for c in CLASSES), 'Ineligible forecast acquired model probabilities')
            eligible_rows = [r for r in pred_ledgers[head] if r['prediction_eligible']]
            labels = [r['target_class'] if r['target_available'] else float('nan') for r in eligible_rows]
            summary, monthly = math_worker.independent_metrics(probabilities, labels, [r['outcome_month'] for r in eligible_rows])
            same(pred['summary'], summary, 'Independent prediction metrics')
            same(load_json(check(pred['artifacts']['summary'])), summary, 'Saved summary metrics')
            actual_monthly = _parquet(check(pred['artifacts']['monthly']), v['artifacts']).to_pylist()
            same(actual_monthly, monthly, 'Independent monthly metrics')
            verify_class_reports(probabilities, labels, [r['outcome_month'] for r in eligible_rows],
                _parquet(check(pred['artifacts']['confusion']), v['artifacts']).to_pylist(),
                _parquet(check(pred['artifacts']['mode_class_performance']), v['artifacts']).to_pylist())
            results.append({'year': year, 'head': head, 'common_fit_rows': len(fit_keys), 'original_predictions': len(rows),
                'eligible_predictions': len(eligible_rows), 'saved_model_probability_reload_exact': True,
                'mean_monthly_log_loss': summary['mean_monthly_log_loss']})
            del estimator, probabilities, forecast, forecast_meta, y
            gc.collect()
        del prediction_X, train, metadata, pred_ledgers, merged_source
        print(json.dumps({'phase': 'independent_target_models', 'verified_fits': len(results), 'maximum_fits': 20}), flush=True)
        gc.collect()
    receipt = {'schema': SCHEMA, 'status': 'PASS', 'phase': 'models', 'verification_plan_pin': vp,
        'model_plan_pin': pin(model_plan_path), 'worker_pin': pin(__file__), 'helper_pins': v['helper_pins'],
        'verified_estimator_fits': len(results), 'verified_shared_preprocessors': 4,
        'new_estimator_fits': 0, 'source_target_class_completion_and_original_denominators_rechecked': True,
        'raw_saved_training_and_prediction_preprocessing_reconstructed': True, 'source_feature_formulas_recomputed': False,
        'source_feature_value_copy_equivalence_inherited_from_pinned_producer_and_prior_input_proof': True,
        'source_identity_action_completeness_or_investability_certified': False,
        'financial_2024_plus_values_read': False, 'reserved_derived_read_bytes': v['derived_read_reservation_bytes'],
        'results': results, 'seconds': time.monotonic() - started}
    receipt['receipt_pin'] = save_new(destination, receipt)
    return receipt


def preflight_record_key(row):
    """Only canonicalize declared calendar dates; keep the full exact record."""
    result = dict(row)
    for name in ('date', 'signal_date', 'execution_date', 'scheduled_exit_date'):
        if name not in result:
            continue
        value = result[name]
        if type(value) is dt.datetime:
            need(value.tzinfo is None and value.time() == dt.time(), 'Noncanonical preflight timestamp')
            value = value.date().isoformat()
        elif type(value) is dt.date:
            value = value.isoformat()
        result[name] = day(value)
    return json.dumps(result, sort_keys=True, allow_nan=False)


def independent_common_preflight(selected, rebalances, quote_metadata, sessions, events, historical):
    """Enumerate every selected unstopped horizon using dated legal terms only."""
    need(len(events) == 18 and len(historical) == 11, 'Frozen current/prior action scope changed')
    dates = list(sessions); index = {d: i for i, d in enumerate(dates)}
    schedule = [r['execution_date'] for r in rebalances]
    need(len(schedule) == len(set(schedule)) == 47 and all(d in index for d in schedule),
         'Full monthly rebalance grid required')
    exits = {d: schedule[i+1] if i+1 < len(schedule) else dates[-1] for i, d in enumerate(schedule)}
    quotes = {(r['trading_date'], r['intrinio_security_id']): r for r in quote_metadata}
    need(len(quotes) == len(quote_metadata), 'Duplicate preflight quote metadata')
    current = defaultdict(list)
    for event in sorted(events, key=lambda e: (e['effective_date'], e['event_id'])):
        need(event['effective_date'] in index, 'Current action is not an authorized session')
        current[event['effective_date']].append(event)
    retired = sorted([*historical, *events], key=lambda e: (e['effective_date'], e['event_id']))
    required, missing, bad, unfilled, actions = set(), [], [], [], []

    def require_quote(episode, native, d, role):
        required.add((d, native))
        details = {k: episode[k] for k in ('signal_date', 'execution_date', 'source_security_id', 'side')}
        details.update(scheduled_exit_date=exits[episode['execution_date']], intrinio_security_id=native, date=d, role=role)
        quote = quotes.get((d, native))
        if quote is None:
            missing.append({**details, 'reason': 'missing_key_requires_observed_quote_or_verified_legal_action'})
        else:
            if 'price_accepted' in quote:
                accepted = type(quote['price_accepted']) is bool and quote['price_accepted']
            else:
                accepted = quote.get('frequency') == 'daily' and quote.get('intraperiod') is False \
                    and quote.get('price_quality_status') in ('accepted', 'ok')
            accepted = accepted and all(type(quote.get(k)) is bool and quote[k]
                for k in ('raw_ohlc_bounds_valid', 'adjusted_ohlc_bounds_valid') if k in quote)
            if not accepted:
                bad.append({**details, 'reason': 'unaccepted_source_metadata_requires_independent_review'})

    for row in selected:
        native, entry = row['intrinio_security_id'], row['execution_date']
        need(entry in exits, 'Selected episode has no frozen rebalance')
        retirement = next((e for e in reversed(retired)
            if e['predecessor_id'] == native and e['effective_date'] <= entry), None)
        if retirement:
            unfilled.append({k: row[k] for k in ('signal_date', 'execution_date', 'source_security_id', 'intrinio_security_id')}
                | {'event_id': retirement['event_id'], 'status': 'retired_new_order_unfilled_cash_no_entitlement'})
            continue
        for d in dates[index[entry]:index[exits[entry]]+1]:
            for e in current.get(d, []):
                if e['predecessor_id'] == native:
                    need(index[d] > 0, 'Held conversion lacks prior authorized session')
                    require_quote(row, native, dates[index[d]-1], 'action_predecessor_paired_close')
                    actions.append({'signal_date': row['signal_date'], 'source_security_id': row['source_security_id'],
                        'date': d, 'event_id': e['event_id'], 'predecessor_id': native,
                        'successor_id': e['successor_id'], 'status': 'verified_retirement_definition_required_and_present'})
                    native = e['successor_id'] if e['stock_ratio'] > 0 else None
            if native is None: break
            role = 'entry_open' if d == entry else 'scheduled_exit_open' if d == exits[entry] and d != dates[-1] else 'held_open_and_close'
            require_quote(row, native, d, role)
    summary = {'selected_episode_rows': len(selected), 'distinct_maximum_horizon_quote_keys': len(required),
        'missing_key_episode_rows': len(missing), 'quality_flag_episode_rows': len(bad),
        'unique_missing_keys': len({(r['date'], r['intrinio_security_id']) for r in missing}),
        'unique_quality_flag_keys': len({(r['date'], r['intrinio_security_id']) for r in bad}),
        'verified_current_action_definitions': 18, 'historical_retirement_definitions': 11,
        'retired_new_orders': len(unfilled), 'ready_for_accounting': not missing and not bad,
        'source_metadata_only': True, 'prices_or_outcomes_read': False, 'selection_changed': False,
        'unstopped_maximum_horizons_reviewed_for_both_stop_variants': True}
    return {'summary': summary, 'missing_quotes': missing, 'source_quality': bad,
            'retired_orders': unfilled, 'action_requirements': actions}


def _pair(pair, validate):
    """Key/clock checks before converting any paired financial columns."""
    meta = _projected(pair['metadata']); validate(meta.to_pylist())
    values = _projected(pair['values'])
    need(values.select(meta.column_names).equals(meta), 'Metadata changed before financial conversion')
    return values


def full_payoff_groups(rows, windows):
    """Aggregate preserved original cohorts once, independently of the producer."""
    grouped = {}; counts = {}; seen = set()
    for r in rows:
        key = tuple(r[n] for n in KEYS); need(key not in seen, 'Duplicate source key in one payoff head'); seen.add(key)
        month = r['outcome_month']; shift(month, 0)
        need(month in windows and shift(r['signal_date'][:7], 1) == month, 'Payoff target clock differs from original month')
        need(type(r['prediction_eligible']) is bool and type(r['target_available']) is bool,
             'Unknown payoff original admission')
        counts.setdefault(month, [0, 0]); counts[month][0] += 1; counts[month][1] += int(r['prediction_eligible'])
        if r['target_available']:
            cls, value = r['target_class'], r['payoff_value']
            need(r['prediction_eligible'] and type(cls) is int and cls in CLASSES and type(value) in (int, float)
                 and not isinstance(value, bool) and math.isfinite(value), 'Invalid observed own-class payoff')
            grouped.setdefault((month, cls), []).append(float(value))
    return grouped, counts


def payoff_from_groups(grouped, counts, windows, signal):
    last = shift(signal[:7], -1); months = [shift(last, n) for n in range(-119, 1)]
    monthly = []
    for month in months:
        window = windows.get(month); completion = None if window is None else window['label_available_at']
        if completion is not None: day(completion, protected=False)
        completed = completion is not None and completion <= signal
        original, eligible = counts.get(month, (0, 0))
        for c in CLASSES:
            values = grouped.get((month, c), []) if completed else []
            status = ('available' if values else 'scheduled_endpoint_not_observed' if window and not completed
                else 'missing_calendar_month' if not window or not original else 'empty_realized_class')
            monthly.append({'outcome_month': month, 'class_id': c, 'label_available_at': completion,
                'original_member_count': original, 'prediction_eligible_count': eligible, 'member_count': len(values),
                'mean_payoff': math.fsum(v / len(values) for v in values) if values else None, 'status': status})
    available = all(r['status'] == 'available' for r in monthly)
    table = [{'class_id': c, 'payoff': math.fsum(r['mean_payoff'] / 120 for r in monthly if r['class_id'] == c) if available else None,
        'status': 'available' if available else 'withheld_incomplete_120_month_class_grid'} for c in CLASSES]
    return monthly, table, available


def _bt_rows(path, inventory, sessions=None):
    """Validate generated ledger clock fields before values become Python."""
    import pyarrow.parquet as pq
    key = str(Path(path).resolve()); need(key in inventory, 'Unplanned derived backtest read')
    item = inventory[key]; check(item['pin']); pf = pq.ParquetFile(key)
    dates = [c for c in pf.schema_arrow.names if c in ('date', 'trading_date', 'signal_date', 'held_signal_date',
        'execution_date', 'scheduled_exit_date', 'outcome_start', 'outcome_end', 'entry_date', 'exit_date', 'effective_date')]
    metadata = pf.read(columns=dates, use_threads=False)
    for row in metadata.to_pylist():
        for field, value in row.items():
            if value is None: continue
            if isinstance(value, dt.datetime): value = value.date().isoformat()
            elif isinstance(value, dt.date): value = value.isoformat()
            day(value)
            if sessions is not None and field == 'date':
                need(value in sessions, 'Derived economic date outside common47month session clock')
    t = pf.read(columns=item['columns'], use_threads=False)
    need(t.select(dates).equals(metadata), 'Generated backtest metadata changed before values')
    return t


def _failed_float_views(daily, monthly):
    """Exact scalar-preserving null representation for the legacy metric API."""
    import numbers
    import numpy as np
    import pandas as pd
    def view(frame, columns):
        result = frame.copy(deep=True)
        for c in columns:
            values = []
            for value in frame[c]:
                if value is None or value is pd.NA:
                    values.append(np.nan); continue
                need(isinstance(value, numbers.Real) and not isinstance(value, (bool, np.bool_)), 'Nonreal failed metric value')
                if value != value:
                    values.append(np.nan); continue
                number = float(value); need(math.isfinite(number) and number == value, 'Failed metric scalar rounded or infinite')
                values.append(number)
            result[c] = np.asarray(values, dtype=np.float64)
        return result
    return view(daily, ('nav',)), view(monthly, ('nav', 'return'))


def selection_identity_metadata(table, expected_keys):
    """Preserve optional Arrow IDs exactly; never equate numerical NaN to None."""
    rows = table.select([*KEYS, 'intrinio_security_id', 'prediction_eligible']).to_pylist()
    result = {}
    for row in rows:
        key = tuple(row[n] for n in KEYS)
        need(key not in result, 'Duplicate original identity metadata key')
        native = row['intrinio_security_id']
        need(native is None or type(native) is str and bool(native.strip()), 'Invalid native identity metadata')
        need(type(row['prediction_eligible']) is bool and (not row['prediction_eligible'] or native is not None),
             'Eligible identity metadata has no native route')
        result[key] = row
    need(set(result) == expected_keys, 'Original identity metadata population changed')
    return result


def require_original_identity(metadata, left, right):
    need(all(tuple(r[n] for n in KEYS) == tuple(metadata[n] for n in KEYS) for r in (left, right))
         and metadata['prediction_eligible'] == left['prediction_eligible'] == right['prediction_eligible']
         and metadata['intrinio_security_id'] == left['intrinio_security_id'] == right['intrinio_security_id'],
         'Selection admission/native route changed original forecasts')


def verify_backtest(model_plan_path, backtest_plan_path, verification_plan=None):
    """Independent causal payoffs, baskets,72cash/units paths and summaries."""
    import gc
    import itertools
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    started = time.monotonic(); model, run, v, vp = _context(model_plan_path, verification_plan, 'backtest')
    bp = pin(backtest_plan_path); need(v['backtest_plan_pin'] == bp, 'Backtest independent plan redirected')
    bt = load_json(check(bp)); directory = Path(bt['output_directory']); inventory = v['artifacts']
    destination = Path(v['receipt_directory']) / 'independent_verification.json'
    need(destination.parent.resolve() == Path(verification_plan).resolve().parent, 'New proof must accompany its separate sealed read plan')
    need(not destination.exists(), 'Preserve old backtest proof')
    model_proof = load_json(check(v['model_verification_pin']))
    need(model_proof['status'] == 'PASS' and model_proof['phase'] == 'models'
         and model_proof['model_plan_pin'] == pin(model_plan_path) and model_proof['verified_estimator_fits'] == 20,
         'Independent20-model PASS required before backtest proof')
    need(bt['plan_sha256'] == canonical_hex({k: value for k, value in bt.items() if k != 'plan_sha256'}), 'Backtest plan fingerprint changed')
    need(bt['model_plan_sha256'] == model['plan_sha256'] and bt['scenarios'] == 72 and bt['economic_months'] == 47
         and bt['financial_2024_plus_read'] is False, 'Backtest scope differs from fixed comparison')
    need(bt['sessions'] == sorted(set(bt['sessions'])) and bt['sessions'][0] == '2020-02-03'
         and bt['sessions'][-1] == '2023-12-29', 'Common47month economic period changed')
    sessions = set(bt['sessions']); pairs = v['source_pair_specs']
    full = load_json(check(model['target_plan_pin']))
    windows = {w['outcome_month']: {**w, 'label_available_at': w['exit_date']} for a in full['annuals'] for w in a['windows']}
    groups = {('endpoint_matched_clock', 'long'): [], ('endpoint_matched_clock', 'short'): [],
        ('cash', 'long'): [], ('cash', 'short'): [], ('utility', 'long'): [], ('utility', 'short'): []}
    def targets_metadata(rows):
        for r in rows:
            day(r['signal_date']); need('2010-01' <= r['outcome_month'] <= '2023-12', 'Source payoff month outside frozenhistory')
            need(type(r['prediction_eligible']) is bool and type(r['target_available']) is bool,
                 'Source payoff original admission unknown')
            if 'label_available_at' in r:
                day(r['label_available_at'], protected=False)
                need(r['label_available_at'] < '2024-01-01' or r['target_available'] is False,
                     'Protected source payoff released')
    for pair in pairs['endpoint']:
        rows = _pair(pair, targets_metadata).to_pylist()
        for r in rows:
            row = {**r, 'payoff_value': r['label_return']}
            groups[('endpoint_matched_clock', 'long')].append(row)
        del rows
    groups[('endpoint_matched_clock', 'short')] = groups[('endpoint_matched_clock', 'long')]
    for pair in pairs['path_targets']:
        rows = _pair(pair, targets_metadata).to_pylist()
        for r in rows:
            if r['path_variant'] != 'unstopped': continue
            arm = 'cash' if r['target_family'] == 'executable_return' else 'utility'
            field = 'cash_return' if arm == 'cash' else 'utility'
            groups[(arm, r['side'])].append({**r, 'payoff_value': r[field]})
        del rows
    states = {}; payoff_checks = []
    for (arm, side), rows in groups.items():
        aggregate, counts = full_payoff_groups(rows, windows)
        for w in bt['windows']:
            signal = w['signal_date']; monthly, table, available = payoff_from_groups(aggregate, counts, windows, signal)
            base = directory / 'payoffs' / arm / signal / side
            actual_monthly = _bt_rows(base / 'monthly.parquet', inventory).to_pylist()
            actual_table = _bt_rows(base / 'payoffs.parquet', inventory).to_pylist()
            saved = load_json(base / 'state.json'); p = saved['provenance']
            same(actual_monthly, monthly, 'Independent realized own-class monthly means')
            same(actual_table, table, 'Independent120month payoff vector')
            need(p['arm'] == arm and p['side'] == side and p['signal_date'] == p['label_observed_through'] == signal
                 and p['last_outcome_month'] == shift(signal[:7], -1) and p['first_outcome_month'] == monthly[0]['outcome_month']
                 and p['completed_calendar_months'] == 120 and p['vector_available'] is available,
                 'Causal score provenance/window changed')
            need(saved['fingerprint'] == canonical_hex({'table': actual_table, 'monthly': actual_monthly, 'provenance': p}), 'Payoff logical fingerprint changed')
            vector = [r['payoff'] for r in table] if available else None
            if vector is not None and arm == 'endpoint_matched_clock' and side == 'short': vector = [-x for x in vector]
            states[(arm, side, signal)] = vector
            payoff_checks.append({'arm': arm, 'side': side, 'signal_date': signal, 'available': available})
        gc.collect()
    del groups
    forecasts = {h: {} for h in HEADS}
    for pair in pairs['forecasts']:
        def forecast_metadata(rows): verify_prediction_metadata(rows, pair['year'])
        rows = _pair(pair, forecast_metadata).to_pylist()
        for r in rows:
            k = tuple(r[n] for n in KEYS); need(k not in forecasts[pair['head']], 'Duplicate saved forecast originalkey')
            forecasts[pair['head']][k] = r
    commonkeys = set(forecasts['endpoint'])
    need(all(set(x) == commonkeys for x in forecasts.values()), 'Fiveheads changed original forecast population')
    all_selections = {}; selection_checks = []
    for arm, basket in itertools.product(('endpoint_matched_clock', 'cash', 'utility'), ('score', 'best_mode', 'second_mode')):
        base = directory / 'selection' / arm / basket
        arrow_tables = {n: _bt_rows(base / (n + '.parquet'), inventory)
            for n in ('original_ledger', 'selected', 'rebalances', 'monthly')}
        safe_identity = selection_identity_metadata(arrow_tables['original_ledger'], commonkeys)
        tables = {n: table.to_pandas() for n, table in arrow_tables.items()}
        ledger = tables['original_ledger'].to_dict('records'); need(set(key_rows(ledger)) == commonkeys, 'Selection ledger dropped originalkeys')
        lh, sh = ('endpoint', 'endpoint') if arm == 'endpoint_matched_clock' else (arm + '_long', arm + '_short')
        for w in bt['windows']:
            signal = w['signal_date']; original = []
            actual = [r for r in ledger if r['signal_date'] == signal]
            for r in actual:
                k = tuple(r[n] for n in KEYS); left, right = forecasts[lh][k], forecasts[sh][k]
                require_original_identity(safe_identity[k], left, right)
                x = {n: left[n] for n in ('source_security_id', 'signal_date', 'intrinio_security_id', 'prediction_eligible')}
                for side, source in (('long', left), ('short', right)):
                    for c in CLASSES:
                        x[side + '_probability_' + str(c)] = source['probability_' + str(c)]
                        a, b = r['probability_' + side + '_' + str(c)], source['probability_' + str(c)]
                        need(a == b or pd.isna(a) and b is None, 'Selection probability differs from saved source forecast')
                    if left['prediction_eligible']:
                        pvec = probability_vector(source)
                        mode = 1 + max(range(10), key=lambda j: pvec[j])
                        need(r['predicted_class_' + side] == mode, 'Directional predicted mode mapping changed')
                        mu = states[(arm, side, signal)]
                        expected_score = None if mu is None else math.fsum(p * m for p, m in zip(pvec, mu))
                        val = r['expected_score_' + side]
                        need(pd.isna(val) if expected_score is None else val == expected_score,
                             'Directional expected score differs from causal own-class probabilities/payoffs')
                original.append(x)
            choice = independent_selection(original, arm=arm, basket=basket,
                long_payoffs=states[(arm, 'long', signal)], short_payoffs=states[(arm, 'short', signal)])
            if w['outcome_month'] == '2020-01':
                choice = {'long': [], 'short': [], 'n': 0, 'status': 'withheld_incomplete_120_month_class_grid'
                    if basket == 'score' else 'outside_common_comparison_period'}
            if choice['status'] == 'withheld_zero_distinct_candidates': choice['status'] = 'withheld_zero_distinct_directional_candidates'
            monthly = tables['monthly'].loc[tables['monthly'].signal_date.eq(signal)]
            rebalances = tables['rebalances'].loc[tables['rebalances'].signal_date.eq(signal)]
            need(len(monthly) == len(rebalances) == 1, 'Missing original monthly rebalance')
            need(monthly.iloc[0]['status'] == rebalances.iloc[0]['selection_status'] == choice['status']
                 and int(monthly.iloc[0]['selected_names_per_side']) == int(rebalances.iloc[0]['selected_names_per_side']) == choice['n'],
                 'Monthly balanced breadth/status differs from independent signals')
            for r in actual:
                sid = r['source_security_id']; expected_side = 'long' if sid in choice['long'] else 'short' if sid in choice['short'] else None
                need(bool(r['selected']) is (expected_side is not None)
                     and (r['side'] == expected_side or expected_side is None and pd.isna(r['side'])), 'Selected holding differs from independent disjoint ranks')
            chosen = tables['selected'].loc[tables['selected'].signal_date.eq(signal)]
            for side in ('long', 'short'):
                part = chosen.loc[chosen.side.eq(side)]
                need(set(part.source_security_id) == set(choice[side]) and len(part) == choice['n'], 'Saved selected originals differ from independentbasket')
                if choice['n']:
                    need(np.all(part.intended_weight.to_numpy() == (.5 if side == 'long' else -.5) / choice['n']), 'Balanced selectedweights changed')
            selection_checks.append({'arm': arm, 'basket': basket, 'signal_date': signal, 'n': choice['n'], 'status': choice['status']})
        all_selections[(arm, basket)] = tables
    del forecasts
    # Independent quote-overlay reconstruction, never the production overlay.
    overlay = load_independent(check(v['helper_pins']['overlay_math']), '_independent_target_overlay')
    equations = load_independent(check(v['helper_pins']['accounting_math']), '_independent_target_cash_units')
    original = []
    for pair in pairs['daily']:
        meta = _projected(pair['metadata']); positions = []
        for i, r in enumerate(meta.to_pylist()):
            d = day(r['trading_date']);
            if d in sessions: positions.append(i)
        values = _projected(pair['values']); need(values.select(meta.column_names).equals(meta), 'Execution metadata changedbeforeprices')
        original.append(values.take(pa.array(positions, type=pa.int64())).to_pandas())
    original = pd.concat(original, ignore_index=True)
    def repair_meta(rows):
        for r in rows: need(day(r['trading_date']) in sessions, 'Execution-only overlay outside commonperiod')
    repairs = _pair(pairs['repairs'][0], repair_meta).to_pandas()
    phase = load_json(check(bt['execution_pins']['execution_phase_pin'])); manifest = load_json(check(bt['execution_pins']['execution_manifest_pin']))
    evidence = {i['sha256'] for i in bt['execution_evidence_pins']}
    def rejection(change):
        item = change['rejection_evidence_pin']; need(item['sha256'] in evidence, 'Quote rejection not source-pinned')
        path = (Path(bt['project_root']) / item['path']).resolve(); check({**clean_pin(item), 'path': str(path)})
    prices, changes = overlay.reconstruct_overlay(original, repairs, manifest['quote_amendments'], sessions=bt['sessions'], authenticate_rejection=rejection)
    need(len(changes) == 40, 'Execution-only quote amendment count changed'); del original, repairs
    quote_index = {(r['trading_date'], r['intrinio_security_id']): r for r in prices.to_dict('records')}
    preflight_checks = []
    quote_meta = prices[[n for n in ('trading_date', 'intrinio_security_id', 'frequency', 'intraperiod',
        'price_quality_status', 'raw_ohlc_bounds_valid', 'adjusted_ohlc_bounds_valid') if n in prices]].to_dict('records')
    for (arm, basket), selected_tables in all_selections.items():
        chosen = selected_tables['selected'].loc[selected_tables['selected'].execution_date.ge('2020-02-03')]
        rebalances = selected_tables['rebalances'].loc[selected_tables['rebalances'].execution_date.ge('2020-02-03')]
        expected = independent_common_preflight(chosen.to_dict('records'), rebalances.to_dict('records'), quote_meta,
            bt['sessions'], phase['event_registry']['events'], phase['historical_retirements'])
        base = directory / 'preflight' / arm / basket
        for name in ('missing_quotes', 'source_quality', 'retired_orders', 'action_requirements'):
            actual = _bt_rows(base / (name + '.parquet'), inventory).to_pylist()
            need(sorted(preflight_record_key(r) for r in actual)
                 == sorted(preflight_record_key(r) for r in expected[name]),
                 'Independent selected-horizon ' + name)
        saved = load_json(base / 'summary.json')
        for name, value in expected['summary'].items(): same(saved[name], value, 'Independent preflight summary/' + name)
        preflight_checks.append({'arm': arm, 'basket': basket, **expected['summary']})
    del quote_meta
    comparison = _bt_rows(directory / 'comparison.parquet', inventory).to_pandas()
    yearly = _bt_rows(directory / 'yearly.parquet', inventory).to_pandas()
    cases = set(itertools.product(('endpoint_matched_clock', 'cash', 'utility'), ('score', 'best_mode', 'second_mode'), ('unstopped', 'asset_stop_10pct'), (0, 5, 10, 25)))
    need(len(comparison) == 72 and set(map(tuple, comparison[['arm', 'basket', 'variant', 'cost_bps']].to_numpy())) == cases,
         'Full72scenario comparison grid changed')
    need(len(yearly) == 288, 'Full72by4annual comparison grid changed')
    checked = []
    names = ('daily', 'trades', 'holdings', 'failures', 'stops', 'action_events', 'market_trades', 'orders', 'identity_handovers')
    for arm, basket, variant, fee in sorted(cases):
        base = directory / 'replay' / arm / basket / variant / f'cost_{fee:02d}bps'
        tables = {n: _bt_rows(base / (n + '.parquet'), inventory, sessions).to_pandas() for n in names}
        monthly = _bt_rows(base / 'monthly.parquet', inventory).to_pandas()
        selection = all_selections[(arm, basket)]
        selected = selection['selected'].loc[selection['selected'].execution_date.ge('2020-02-03')].copy()
        rebalances = selection['rebalances'].loc[selection['rebalances'].execution_date.ge('2020-02-03')].copy()
        summary = comparison.loc[comparison.arm.eq(arm) & comparison.basket.eq(basket)
            & comparison.variant.eq(variant) & comparison.cost_bps.eq(fee)].iloc[0]
        annual = yearly.loc[yearly.arm.eq(arm) & yearly.basket.eq(basket) & yearly.variant.eq(variant) & yearly.cost_bps.eq(fee)]
        own_annual = _bt_rows(base / 'annual.parquet', inventory).to_pandas()
        pd.testing.assert_frame_equal(own_annual.reset_index(drop=True), annual.reset_index(drop=True), check_exact=True)
        need(len(monthly) == 47 and summary.outcome_months == 47 and summary.outcome_start == '2020-02-03'
             and summary.outcome_end == '2023-12-29', 'Common economic summary interval changed')
        kwargs = {'events': phase['event_registry']['events'], 'historical_retirements': phase['historical_retirements'],
            'cost_bps': fee, 'stop_fraction': None if variant == 'unstopped' else .1, 'prepared_quotes': quote_index,
            'identity_handovers': phase['identity_handover_registry']['handovers']}
        if bool(summary.complete_proxy_path):
            outcome = equations.verify_scenario(tables, prices, selected, rebalances, bt['sessions'], **kwargs)
            equations.verify_metrics(tables['daily'], tables['trades'], monthly, summary, annual)
        else:
            outcome = equations.verify_failed_scenario(tables, prices, selected, bt['sessions'], rebalances=rebalances, **kwargs)
            dd, mm = _failed_float_views(tables['daily'], monthly)
            equations.verify_failed_metrics(dd, tables['trades'], mm, summary, annual)
        checked.append({'arm': arm, 'basket': basket, 'variant': variant, 'cost_bps': fee,
            'complete_proxy_path': bool(summary.complete_proxy_path), **outcome})
        print(json.dumps({'phase': 'independent_target_backtest', 'verified_scenarios': len(checked), 'maximum_scenarios': 72}), flush=True)
        del tables, monthly
        gc.collect()
    receipt = {'schema': SCHEMA, 'status': 'PASS', 'phase': 'backtest', 'verification_plan_pin': vp,
        'model_plan_pin': pin(model_plan_path), 'backtest_plan_pin': bp, 'model_verification_pin': v['model_verification_pin'],
        'worker_pin': pin(__file__), 'helper_pins': v['helper_pins'], 'verified_scenarios': len(checked),
        'producer_backtest_audit_pin': pin(directory / 'audit.json'),
        'hash_contract_correction': 'Strict plain-hex backtest plan/payoff fingerprints; prefixed model/key hashes unchanged; no numerical rule change',
        'identity_metadata_correction': 'Exact closed Arrow original-key/native/eligibility projection before Pandas conversion; no NaN equality or numerical rule waiver',
        'preflight_date_correction': 'Strict ISO dates and naive midnight Arrow dates compared in complete record multisets; no clock, numerical or row-set waiver',
        'verified_payoff_states': len(payoff_checks), 'verified_original_selection_months': len(selection_checks),
        'new_estimator_fits': 0, 'production_selection_or_accounting_used_as_expected_answer': False,
        'source_identity_action_completeness_or_investability_certified': False,
        'financial_2024_plus_values_read': False, 'reserved_derived_read_bytes': v['derived_read_bytes'],
        'payoff_checks': payoff_checks, 'selection_checks': selection_checks, 'preflight_checks': preflight_checks, 'scenario_checks': checked,
        'execution_overlay_rows_independently_reconstructed': len(changes), 'seconds': time.monotonic() - started}
    receipt['receipt_pin'] = save_new(destination, receipt)
    return receipt


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='action', required=True)
    plan = sub.add_parser('plan'); plan.add_argument('--model-plan', required=True)
    plan.add_argument('--backtest-plan'); plan.add_argument('--output', required=True)
    for name in ('models', 'backtest'):
        a = sub.add_parser(name); a.add_argument('--model-plan', required=True); a.add_argument('--verification-plan', required=True)
        if name == 'backtest': a.add_argument('--backtest-plan', required=True)
    args = p.parse_args(argv)
    if args.action == 'plan': result = plan_verification(args.model_plan, args.backtest_plan, args.output)
    elif args.action == 'models': result = verify_models(args.model_plan, args.verification_plan)
    else: result = verify_backtest(args.model_plan, args.backtest_plan, args.verification_plan)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == '__main__': main()
