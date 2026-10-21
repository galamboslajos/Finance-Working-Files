#!/usr/bin/env python3
"""Independent zero-fit checks for the corrected frozen-setting model rerun.

Import is standard-library only. Source/model execution requires a separately
sealed footer plan and an exact precharged shared read reservation. No production
preprocessor, training adapter, fit function, or metric function supplies an
expected answer.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import sys

SCHEMA = 'dmv2-independent-corrected-model-verification/v1'
KEYS = ('source_security_id', 'signal_date')
ADMISSION = ('identity_eligible', 'requested', 'paper_prediction_eligible')
CLASSES = tuple(range(1, 11))
FLOOR = 1e-15
HASH_IO_BYTES = 0


def need(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False).encode()


def logical(value):
    return 'sha256:' + hashlib.sha256(canonical(value)).hexdigest()


def pin(path):
    global HASH_IO_BYTES
    path = Path(path).resolve()
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
            HASH_IO_BYTES += len(block)
    return {'path': str(path), 'sha256': h.hexdigest(), 'size_bytes': path.stat().st_size}


def authenticated(value):
    need(Path(value['path']).is_absolute() and not Path(value['path']).is_symlink(), 'Absolute regular pin required')
    actual = pin(value['path'])
    need(actual == {k: value[k] for k in ('path', 'sha256', 'size_bytes')}, 'Byte pin changed: ' + str(value['path']))
    return Path(value['path'])


def date(value):
    need(isinstance(value, str), 'Dates must be canonical strings')
    try:
        d = dt.date.fromisoformat(value)
    except (ValueError, TypeError) as e:
        raise ValueError('Invalid date') from e
    need(d.isoformat() == value, 'Noncanonical date')
    return value


def next_month(signal):
    d = dt.date.fromisoformat(signal)
    return f'{d.year + (d.month == 12):04d}-{d.month % 12 + 1:02d}'


def validate_metadata(features, coverage, targets, original, windows, outcome_year):
    """Only safe keys/dates/admission values are accepted by this guard."""
    need(isinstance(outcome_year, int) and not isinstance(outcome_year, bool)
         and 2010 <= outcome_year <= 2023, 'Unsupported annual outcome scope')
    need(len(windows) == 12, 'Exactly twelve independently pinned monthly windows required')
    expected = {}
    for signal, w in windows.items():
        date(signal); date(w['outcome_start']); date(w['outcome_end'])
        month = next_month(signal)
        need(w['outcome_month'] == month and month.startswith(str(outcome_year))
             and signal < w['outcome_start'] <= w['outcome_end'] < '2024-01-01'
             and w['outcome_start'][:7] == w['outcome_end'][:7] == month,
             'Window chronology differs from complete following outcome month')
        expected[signal] = {k: w[k] for k in ('outcome_month', 'outcome_start', 'outcome_end')}
    mappings = []
    for label, rows in [('features', features), ('coverage', coverage),
                        ('targets', targets), ('original', original)]:
        m = {}
        for r in rows:
            sid, signal = (r[k] for k in KEYS)
            need(isinstance(sid, str) and sid.strip() and sid == sid.strip(), 'Invalid original source key')
            date(signal)
            need(signal in expected, 'Unknown/future signal metadata before financial decode')
            key = (sid, signal)
            need(key not in m, 'Duplicate ' + label + ' source/signal key')
            if label == 'original' and r.get('as_of') is not None:
                as_of = r['as_of'].isoformat() if type(r['as_of']) is dt.date else r['as_of']
                need(date(as_of) == signal, 'Original membership as_of differs from signal cutoff')
            if label in ('coverage', 'targets'):
                need(all(type(r.get(k)) is bool for k in ADMISSION), 'Unknown admission booleans')
                need(not r['paper_prediction_eligible'] or (r['identity_eligible'] and r['requested']),
                     'Paper cohort overrides identity/request restriction')
            if label == 'targets':
                need(all(r[k] == v for k, v in expected[signal].items()), 'Target clocks disagree with pinned window')
                need(type(r.get('target_available')) is bool, 'Unknown target availability')
                need(not r['target_available'] or r['paper_prediction_eligible'], 'Outcome selected an ineligible row')
            m[key] = r
        need(set(r['signal_date'] for r in rows) == set(expected), 'Missing original monthly cohort')
        mappings.append(m)
    fm, cm, tm, om = mappings
    need(set(fm) == set(cm) == set(tm) == set(om), 'Original denominator/key set changed')
    for key in fm:
        need(all(cm[key][name] == tm[key][name] for name in ADMISSION), 'Feature/target cohort mismatch')
    return [tuple(r[k] for k in KEYS) for r in features]


def array_sha(a):
    import numpy as np
    a = np.ascontiguousarray(a)
    return 'sha256:' + hashlib.sha256(memoryview(a).cast('B') if a.size else b'').hexdigest()


def numeric_matrix(values):
    import numpy as np
    a = np.asarray(values)
    need(a.ndim == 2 and a.dtype.kind in 'fiu', 'Predictors must be a real numeric matrix, never bool/text')
    a = a.astype(np.float64)
    a[~np.isfinite(a)] = np.nan
    return a


def derive_preprocessor(raw, names, selected, keys, paper_names, train_start, train_end, observed_through=None):
    """Derive medians/drop/schema from exactly the supervised training mask."""
    import numpy as np
    a = numeric_matrix(raw)
    selected = np.asarray(selected)
    need(selected.dtype.kind == 'b' and selected.shape == (len(a),) and selected.any(), 'Explicit nonempty Boolean fit mask required')
    need(len(names) == a.shape[1] and len(set(names)) == len(names), 'Duplicate/misaligned predictor schema')
    need(set(paper_names) <= set(names), 'Paper15 missing from predictor schema')
    need(np.isfinite(a[selected][:, [names.index(n) for n in paper_names]]).all(), 'Selected training baseline is missing')
    fitted = a[selected]
    kept, dropped, medians = [], {}, {}
    for j, name in enumerate(names):
        finite = sorted(float(x) for x in fitted[:, j] if math.isfinite(x))
        if not finite:
            dropped[name] = 'all_missing_in_training_partition'
        elif len(set(finite)) == 1:
            dropped[name] = 'constant_finite_values_in_training_partition'
        else:
            kept.append(name)
            n = len(finite)
            medians[name] = finite[n // 2] if n % 2 else finite[n // 2 - 1] / 2 + finite[n // 2] / 2
    need(kept, 'No nonconstant training predictors')
    manifest = {
        'schema_version': 1, 'preprocessing_policy': 'train_partition_median_missing_indicator_v1',
        'input_columns': list(names), 'retained_columns': kept,
        'output_columns': [v for n in kept for v in (n, n + '__missing')],
        'medians': medians, 'dropped_columns': dropped,
        'baseline_required_for_population': list(paper_names), 'fit_rows': int(selected.sum()),
        'fit_keys_sha256': logical([list(k) for k, take in zip(keys, selected) if take]),
        'train_start': date(train_start), 'train_end': date(train_end),
        'label_observed_through': date(observed_through or train_end),
        'missing_indicators': 'one for every retained predictor, fixed even when training is complete',
        'nonfinite_inputs': 'missing; paper baseline rows withheld rather than imputed',
        'imputation': 'training-partition finite-value median; labels never imputed',
        'scaling': None, 'clipping': None, 'winsorization': None,
        'training_only_schema_selection': True, 'estimator_fitted': False,
        'knowledge_time_verified': False, 'accounting_certified': False, 'model_ready': False}
    return {'manifest': manifest, 'fingerprint': logical(manifest)}


def prepare_matrix(raw, names, state, selected):
    import numpy as np
    a = numeric_matrix(raw)
    mask = np.asarray(selected)
    need(mask.dtype.kind == 'b' and mask.shape == (len(a),), 'Explicit matrix row mask required')
    manifest = state['manifest']
    need(logical(manifest) == state['fingerprint'], 'Preprocessing state fingerprint changed')
    need(list(names) == manifest['input_columns'], 'Input column order changed')
    out = np.empty((int(mask.sum()), len(manifest['output_columns'])), dtype=np.float64)
    for i, name in enumerate(manifest['retained_columns']):
        col = a[mask, names.index(name)]
        missing = ~np.isfinite(col)
        out[:, 2 * i] = np.where(missing, manifest['medians'][name], col)
        out[:, 2 * i + 1] = missing
    need(np.isfinite(out).all(), 'Prepared matrix is not finite')
    bound = float(np.finfo(np.float32).max)
    need(not out.size or np.max(np.abs(out)) <= bound, 'Matrix overflows float32')
    return np.ascontiguousarray(out, dtype=np.float32)


def canonical_labels(values, available):
    import numpy as np
    available = np.asarray(available)
    need(available.dtype.kind == 'b' and available.shape == (len(values),), 'Explicit target mask required')
    result = np.full(len(values), np.nan)
    for i, value in enumerate(values):
        if not available[i]:
            need(value is None or (isinstance(value, (int, float)) and not isinstance(value, bool) and math.isnan(value)),
                 'Unavailable target contains a numerical class')
            continue
        need(not isinstance(value, (bool, str)) and isinstance(value, (int, float))
             and math.isfinite(value) and int(value) == value and int(value) in CLASSES,
             'Target is not a canonical integer class')
        result[i] = int(value)
    return result


def infer_probabilities(estimator, kind, matrix, explicit_parameters):
    """Independent exact reload inference; never calls fit/production helpers."""
    import numpy as np
    params = estimator.get_params(deep=False)
    need(params.get('n_jobs') == 1 and explicit_parameters.get('n_jobs') == 1, 'Reload inference requires one model worker')
    for k, v in explicit_parameters.items():
        need(params.get(k) == v, 'Reloaded frozen parameter mismatch: ' + k)
    native = list(estimator.classes_)
    expected = list(CLASSES) if kind == 'rf' else list(range(10))
    need(len(native) == 10 and all(not isinstance(k, (bool, np.bool_)) and isinstance(k, (int, np.integer)) for k in native)
         and set(native) == set(expected), 'Missing/invalid model class map')
    if kind == 'rf':
        values = np.asarray(estimator.predict_proba(matrix), dtype=np.float64)
        canonical_ids = native
    elif kind == 'xgb':
        margin = np.asarray(estimator.predict(matrix, output_margin=True))
        need(margin.shape == (len(matrix), 10) and np.isfinite(margin).all(), 'Invalid XGBoost native margins')
        shifted = margin.astype(np.float64) - margin.max(axis=1, keepdims=True).astype(np.float64)
        exp = np.exp(shifted)
        values = exp / exp.sum(axis=1, keepdims=True)
        canonical_ids = [int(k) + 1 for k in native]
    else:
        raise ValueError('Unsupported estimator kind')
    need(values.shape == (len(matrix), 10) and np.isfinite(values).all() and (values >= 0).all(), 'Invalid probabilities')
    values = values[:, [canonical_ids.index(k) for k in CLASSES]]
    need(np.all(np.abs(values.sum(axis=1) - 1) <= 1e-8), 'Probability sums do not equal one')
    return values


def independent_metrics(probabilities, labels, months):
    import numpy as np
    p = np.asarray(probabilities)
    y = np.asarray(labels, dtype=float)
    need(p.shape == (len(y), 10) and len(months) == len(y), 'Metric row mismatch')
    need(np.isfinite(p).all() and (p >= 0).all() and np.all(np.abs(p.sum(axis=1) - 1) <= 1e-8), 'Metric probability validation failed')
    valid = np.isfinite(y)
    need(np.isin(y[valid], CLASSES).all(), 'Metric target class invalid')
    modes = np.argmax(p, axis=1) + 1
    loss, brier, hit = np.full(len(y), np.nan), np.full(len(y), np.nan), np.full(len(y), np.nan)
    for i in np.flatnonzero(valid):
        c = int(y[i]) - 1
        loss[i] = -math.log(max(float(p[i, c]), FLOOR))
        brier[i] = math.fsum((float(p[i, j]) - (j == c)) ** 2 for j in range(10))
        hit[i] = modes[i] == y[i]
    monthly = []
    for month in sorted(set(months)):
        indices = [i for i, m in enumerate(months) if m == month]
        seen = [i for i in indices if valid[i]]
        monthly.append({'outcome_month': month, 'prediction_rows': len(indices), 'labelled_rows': len(seen),
            'missing_label_rows': len(indices) - len(seen), 'label_coverage': len(seen) / len(indices),
            'zero_true_class_probability_rows': int(sum(p[i, int(y[i]) - 1] == 0 for i in seen)),
            'floored_true_class_probability_rows': int(sum(p[i, int(y[i]) - 1] < FLOOR for i in seen)),
            'status': 'available' if seen else 'no_observed_labels',
            'log_loss': math.fsum(loss[i] for i in seen) / len(seen) if seen else None,
            'brier_sum10': math.fsum(brier[i] for i in seen) / len(seen) if seen else None,
            'accuracy': math.fsum(hit[i] for i in seen) / len(seen) if seen else None})
    summary = {'prediction_rows': len(y), 'labelled_rows': int(valid.sum()), 'missing_label_rows': int((~valid).sum())}
    for name, values in [('log_loss', loss), ('brier_sum10', brier), ('accuracy', hit)]:
        summary['pooled_' + name] = math.fsum(values[valid]) / int(valid.sum()) if valid.any() else None
        found = [m[name] for m in monthly if m[name] is not None]
        summary['mean_monthly_' + name] = math.fsum(found) / len(found) if found else None
    seen = np.flatnonzero(valid)
    def ece(class_id=None):
        bins = [[] for _ in range(10)]
        for i in seen:
            confidence = float(p[i].max()) if class_id is None else float(p[i, class_id - 1])
            hit_value = int(modes[i] == y[i]) if class_id is None else int(y[i] == class_id)
            bins[min(int(confidence * 10), 9)].append((confidence, hit_value))
        return math.fsum(abs(math.fsum(c for c, h in b) - sum(h for c, h in b)) for b in bins) / len(seen) if len(seen) else None
    summary.update({
        'label_coverage': int(valid.sum()) / len(y) if len(y) else None,
        'outcome_months': len(monthly), 'observed_label_months': sum(m['labelled_rows'] > 0 for m in monthly),
        'first_outcome_month': monthly[0]['outcome_month'] if monthly else None,
        'last_outcome_month': monthly[-1]['outcome_month'] if monthly else None,
        'pooled_top_label_ece': ece(),
        'pooled_classwise_ece_mean': math.fsum(ece(k) for k in CLASSES) / 10 if len(seen) else None,
        'zero_true_class_probability_rows': int(sum(p[i, int(y[i]) - 1] == 0 for i in seen)),
        'floored_true_class_probability_rows': int(sum(p[i, int(y[i]) - 1] < FLOOR for i in seen)),
        'log_loss_evaluation_floor': FLOOR, 'log_loss_floor_changes_stored_probabilities': False,
        'probability_clipping_or_renormalization': False,
        'brier_definition': 'sum_of_ten_squared_probability_errors',
        'primary_aggregation': 'equal_weight_mean_of_observed_monthly_means',
        'hard_mode_probability_tie_rule': 'lowest_canonical_class_id',
        'calibration_fixed_equal_width_bins': 10, 'label_imputation_performed': False,
        'RET_or_portfolio_evaluation_performed': False})
    return summary, monthly


# Source/footer planning and sealed execution are appended only after the producer
# artifact contract is frozen. These helpers have no research read side effects.

def footer_projection(artifact_pin, columns):
    """Plan whole flat annual columns from footer lengths only."""
    import pyarrow.parquet as pq
    p = authenticated(artifact_pin)
    columns = list(columns)
    need(columns and len(set(columns)) == len(columns), 'Distinct exact columns required')
    pf = pq.ParquetFile(p)
    need(len(set(pf.schema_arrow.names)) == len(pf.schema_arrow.names)
         and set(columns) <= set(pf.schema_arrow.names), 'Projected schema mismatch')
    sizes = []
    for g in range(pf.metadata.num_row_groups):
        rg = pf.metadata.row_group(g)
        found = {}
        for j in range(rg.num_columns):
            leaf = rg.column(j)
            root = leaf.path_in_schema.split('.')[0]
            need(type(leaf.total_compressed_size) is int and leaf.total_compressed_size >= 0, 'Invalid compressed leaf size')
            found[root] = found.get(root, 0) + leaf.total_compressed_size
        need(set(columns) <= set(found), 'Projection root missing from Parquet leaf paths')
        sizes.append(sum(found[c] for c in columns))
    return {'pin': {k: artifact_pin[k] for k in ('path', 'sha256', 'size_bytes')},
            'columns': columns, 'rows': pf.metadata.num_rows,
            'row_group_compressed_bytes': sizes, 'projected_compressed_bytes': sum(sizes),
            'schema_fields': {c: str(pf.schema_arrow.field(c).type) for c in columns}}


def read_projected(entry):
    import pyarrow.parquet as pq
    authenticated(entry['pin'])
    pf = pq.ParquetFile(entry['pin']['path'])
    need(pf.metadata.num_rows == entry['rows'], 'Source footer row count changed')
    need({c: str(pf.schema_arrow.field(c).type) for c in entry['columns']} == entry['schema_fields'],
         'Source footer schema changed')
    table = pf.read(columns=entry['columns'], use_threads=False)
    need(table.num_rows == entry['rows'], 'Source projection truncated')
    return table


def guarded_numeric(table, columns):
    import pyarrow as pa
    import numpy as np
    need(set(columns) <= set(table.column_names), 'Missing numeric projection')
    for c in columns:
        t = table.schema.field(c).type
        need(pa.types.is_integer(t) or pa.types.is_floating(t) or pa.types.is_null(t),
             'Feature/class field must be numerical, not metadata or text')
    frame = table.select(columns).to_pandas()
    return np.asarray(frame, dtype=np.float64)


def assert_close(actual, expected, label, tolerance=2e-12):
    """Independent summation may differ at final floating-point bits only."""
    if isinstance(expected, dict):
        need(isinstance(actual, dict) and set(expected) <= set(actual), label + ' metric keys differ')
        for k, v in expected.items():
            assert_close(actual[k], v, label + '.' + k, tolerance)
    elif isinstance(expected, (list, tuple)):
        need(isinstance(actual, (list, tuple)) and len(actual) == len(expected), label + ' metric rows differ')
        for i, (a, e) in enumerate(zip(actual, expected)):
            assert_close(a, e, label + '.' + str(i), tolerance)
    elif expected is None or isinstance(expected, (str, bool)):
        need(type(actual) is type(expected) and actual == expected, label + ' differs')
    elif isinstance(expected, int):
        need(not isinstance(actual, bool) and actual == expected, label + ' count differs')
    else:
        need(isinstance(actual, (int, float)) and math.isfinite(actual)
             and math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance), label + ' value differs')

YEARS = (2020, 2021, 2022, 2023)
ARMS = {'rf_paper15': ('rf', 'paper15'), 'xgb_paper15': ('xgb', 'paper15'),
        'rf_extended': ('rf', 'expanded'), 'xgb_extended': ('xgb', 'expanded'),
        'rf_tuned_extended': ('rf', 'expanded')}
COVERAGE_FIELDS = (*KEYS, 'intrinio_security_id', 'identity_status', 'source_identity_status',
    *ADMISSION[:1], 'identity_eligibility_policy', 'identity_eligibility_reason',
    'identity_review_required', 'resolution_status', 'paper_prediction_eligible',
    'requested', 'knowledge_time_verified', 'accounting_certified',
    'available_feature_count', 'registered_feature_count', 'feature_status', 'model_ready')
ORIGINAL_METADATA = (*KEYS, 'as_of', 'indices', 'membership_tickers', 'unified_security_id',
    'intrinio_security_id', 'identity_status', 'bridge_method', 'bridge_confidence',
    'identity_review_required', 'resolution_status', 'source_identity_status',
    'identity_eligible', 'identity_eligibility_policy', 'identity_eligibility_reason',
    'original_intrinio_security_id', 'original_identity_status', 'original_resolution_status',
    'original_identity_eligible', 'reviewed_security_isin', 'identity_amendment_id',
    'identity_amendment_sha256', 'replacement_native_status', 'el_paso_route_amendment_id',
    'el_paso_route_amendment_sha256', 'el_paso_route_original_native_id',
    'el_paso_route_replacement_native_id', 'el_paso_route_reviewed_cusip',
    'el_paso_route_lookup_identifier', 'el_paso_route_historical_identity_certified',
    'shadow_original_intrinio_security_id', 'shadow_original_identity_status',
    'shadow_original_identity_review_required', 'shadow_original_resolution_status',
    'shadow_original_identity_eligible', 'shadow_route_policy', 'computational_price_key',
    'price_route_provider', 'price_route_source_key', 'provider_intrinio_security_id',
    'price_route_status', 'shadow_canonical_route_promoted')
TARGET_META = (*KEYS, 'outcome_month', 'outcome_start', 'outcome_end', *ADMISSION,
    'prediction_eligible', 'outcome_requested', 'return_available', 'observed_return_available',
    'target_available', 'target_status', 'return_observation_status', 'source_return_status')
PROBABILITY_NAMES = tuple(f'probability_{k}' for k in CLASSES)


def read_json(path):
    return json.loads(Path(path).read_text())


def save_new(path, value):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False); f.write('\n')
    return pin(p)


def published_hash(value):
    return 'sha256:' + hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def fixed_file(run, year, relative):
    p = Path(run).resolve() / 'folds' / str(year) / relative
    need(p.is_file() and not p.is_symlink(), 'Missing exact new annual artifact: ' + str(p))
    return pin(p)


def make_plan(root, protocol_path, run, year, output):
    """Inspect authenticated metadata and footer lengths; no source values/models."""
    need(year in YEARS and type(year) is int, 'Unapproved verification year')
    root, run = Path(root).resolve(), Path(run).resolve()
    p = read_json(protocol_path)
    need(p['schema'] == 'dmv2-corrected-refit-protocol/v1'
         and p['status'] == 'SEALED_BEFORE_FINANCIAL_READ'
         and p['outcome_years'] == list(YEARS) and p['parameter_search'] is False
         and p['protected_2024_plus'] is True and p['limits']['max_fits'] == 20,
         'Frozen no-search four-window protocol required')
    need(run == Path(p['run_directory']).resolve(), 'Verifier run differs from exact protocol')
    catalog = read_json(authenticated(p['catalog_pin']))
    need(catalog['schema'] == p['catalog_schema'] and catalog['status'] == p['catalog_status'], 'Catalog version changed')
    need(catalog['independent_verification'] == p['input_verification_pin'], 'Catalog independent proof differs from protocol')
    proof = read_json(authenticated(p['input_verification_pin']))
    need(proof['status'] == 'PASS', 'Independent input PASS required')
    terminal = read_json(authenticated(p['terminal_guard_pin']))
    need(terminal['status'] == 'PASS' and terminal['events_reviewed'] == 29
         and terminal['target_definition_unchanged'] is True
         and terminal['missing_terminal_targets_imputed'] is False, 'Unapproved terminal target semantics')
    source_pin = pin(run / 'read_plan.json'); source = read_json(source_pin['path'])
    need(source['protocol_pin'] == pin(protocol_path) and source['catalog_pin'] == p['catalog_pin'], 'Producer source identity mismatch')
    need(source['worker_pin'] == p['worker_pin'], 'Producer source code changed')
    names, paper = source['predictor_names'], source['paper_names']
    registered = source['registered_specs']
    need(len(registered) == 728 and len(names) == len(set(names)) == 715
         and len(paper) == len(set(paper)) == 15 and set(paper) <= set(names), 'Frozen728/715/Paper15 schema changed')
    need(names == [x['name'] for x in registered if x.get('model_eligible', True)], 'Predictor allowlist/order changed')
    annual_records = {r['year']: r for r in catalog['annuals']}
    need(set(annual_records) == set(range(2010, 2024)) and len(catalog['annuals']) == 14, 'Catalog must retain14 original annuals')
    annuals = {}
    cost = 0
    for y in range(year - 10, year + 1):
        a = annual_records[y]['artifacts']; state_pin = a['target_state']
        state = read_json(authenticated(state_pin))
        windows = {r['signal_date']: {k: r[k] for k in ('outcome_month', 'outcome_start', 'outcome_end')}
                   for r in state['provenance']['month_windows']}
        need(len(windows) == 12 and windows == source['annuals'][str(y)]['windows'], 'Independent catalog windows differ from producer plan')
        observed = date(state['provenance']['label_observed_through'])
        need(observed <= f'{y}-12-31', 'Annual label observation exceeds allowed scope')
        meta = {'features': footer_projection(a['features'], KEYS),
                'coverage': footer_projection(a['feature_coverage'], COVERAGE_FIELDS),
                'targets': footer_projection(a['targets'], TARGET_META),
                'original': footer_projection(a['original_cohort'], [n for n in ORIGINAL_METADATA if n in __import__('pyarrow.parquet',fromlist=['ParquetFile']).ParquetFile(authenticated(a['original_cohort'])).schema_arrow.names])}
        numeric = {'features': footer_projection(a['features'], [*KEYS, *names]),
                   'classes': footer_projection(a['targets'], [*KEYS, 'target_class'])}
        annuals[str(y)] = {'year': y, 'windows': windows, 'observed_through': observed,
                          'target_state_pin': state_pin, 'metadata': meta, 'numeric': numeric}
        cost += sum(x['projected_compressed_bytes'] for x in [*meta.values(), *numeric.values()])
    prepared_pin = fixed_file(run, year, 'DATA_COMPLETE.json')
    prepared = read_json(prepared_pin['path'])
    need(prepared['status'] == 'COMPLETE' and prepared['year'] == year
         and prepared['source_plan_sha256'] == published_hash(source), 'Prepared source/fold identity changed')
    derived, financial_json, generated = {}, {}, {}
    # Read exact new ledger metadata and class separately; never infer admission
    # from stored probabilities or from whether a future label exists.
    for split in ('train', 'test'):
        item = fixed_file(run, year, f'data/{split}_ledger.parquet')
        extra = ('fit_mask',) if split == 'train' else ()
        source_years = range(year-10,year) if split == 'train' else (year,)
        original_fields = list(dict.fromkeys(n for y in source_years for n in annuals[str(y)]['metadata']['original']['columns']))
        cols = list(dict.fromkeys([*TARGET_META, *COVERAGE_FIELDS, *original_fields,
            'feature_complete', 'fit_year', 'training_outcome_start', 'training_outcome_end',
            'training_label_observed_through', *extra]))
        meta = footer_projection(item, cols)
        values = footer_projection(item, [*KEYS, 'target_class'])
        generated[split + '_ledger_metadata'] = meta
        generated[split + '_ledger_classes'] = values
        cost += meta['projected_compressed_bytes'] + values['projected_compressed_bytes']
    derived['train_y'] = fixed_file(run, year, 'data/train_y.npy')
    for bank in ('paper15', 'expanded'):
        financial_json[bank + '_manifest'] = fixed_file(run, year, f'preprocessors/{bank}/manifest.json')
        for split in ('train', 'test'):
            derived[bank + '_' + split] = fixed_file(run, year, f'preprocessors/{bank}/{split}_X.npy')
    arms = {}
    for arm, (kind, bank) in ARMS.items():
        base = 'arms/' + arm + '/'
        fit_pin = fixed_file(run, year, base + 'FIT_COMPLETE.json')
        fitted = read_json(fit_pin['path'])
        need(fitted['status'] == 'FIT_COMPLETE' and fitted['fit_count'] == 1
             and fitted['expected_metadata']['prepared_pin'] == prepared_pin,
             'Model fit does not belong to this source/preprocessing identity')
        model_pin = fixed_file(run, year, base + ('model.joblib' if kind == 'rf' else 'model.ubj'))
        need(fitted['model'] == model_pin, 'Model source pin/path differs')
        derived[arm + '_model'] = model_pin
        derived[arm + '_probabilities'] = fixed_file(run, year, base + 'probabilities.npy')
        receipt_pin = fixed_file(run, year, base + 'PREDICT_COMPLETE.json')
        summary_pin = fixed_file(run, year, base + 'reports/summary.json')
        # These JSONs contain financial diagnostics: charge before parsing below.
        financial_json[arm + '_receipt'] = receipt_pin
        financial_json[arm + '_summary'] = summary_pin
        forecast_pin = fixed_file(run, year, base + 'predictions.parquet')
        metadata_cols = [*generated['test_ledger_metadata']['columns'], 'prediction_status']
        fm = footer_projection(forecast_pin, metadata_cols)
        fv = footer_projection(forecast_pin, [*KEYS, 'target_class', *PROBABILITY_NAMES])
        monthly = footer_projection(fixed_file(run, year, base + 'reports/monthly.parquet'),
            ['outcome_month', 'prediction_rows', 'labelled_rows', 'missing_label_rows',
             'label_coverage', 'log_loss', 'brier_sum10', 'accuracy',
             'zero_true_class_probability_rows', 'floored_true_class_probability_rows', 'status'])
        cost += fm['projected_compressed_bytes'] + fv['projected_compressed_bytes'] + monthly['projected_compressed_bytes']
        arms[arm] = {'kind': kind, 'bank': bank, 'fit_pin': fit_pin, 'fit_attempt_pin': fixed_file(run, year, base + 'FIT_ATTEMPT.json'), 'model_pin': model_pin,
                     'receipt_pin': receipt_pin, 'summary_pin': summary_pin,
                     'forecast_metadata': fm, 'forecast_values': fv, 'monthly_metrics': monthly}
    cost += sum(x['size_bytes'] for x in financial_json.values())
    exact_prepared = {'data/train_y.npy': derived['train_y']}
    for split in ('train','test'): exact_prepared[f'data/{split}_ledger.parquet'] = generated[split + '_ledger_metadata']['pin']
    for bank in ('paper15','expanded'):
        exact_prepared[f'preprocessors/{bank}/manifest.json'] = financial_json[bank + '_manifest']
        for split in ('train','test'): exact_prepared[f'preprocessors/{bank}/{split}_X.npy'] = derived[bank + '_' + split]
    need(set(prepared['artifacts']) == set(exact_prepared), 'Prepared artifact set changed')
    for name,item in exact_prepared.items():
        need({k:prepared['artifacts'][name][k] for k in ('path','sha256','size_bytes')} == item, 'Prepared artifact pin redirects exact path: ' + name)
    body = {'schema': 'dmv2-independent-corrected-model-plan/v1', 'status': 'SEALED_BEFORE_NUMERICAL_READ',
        'root': str(root), 'run': str(run), 'year': year, 'protocol_pin': pin(protocol_path),
        'catalog_pin': p['catalog_pin'], 'input_verification_pin': p['input_verification_pin'],
        'terminal_guard_pin': p['terminal_guard_pin'], 'worker_pin': pin(__file__),
        'control_pin': pin(Path(__file__).parent / 'control.py'), 'source_plan_pin': source_pin,
        'prepared_pin': prepared_pin, 'annuals': annuals, 'predictor_names': names, 'paper_names': paper,
        'derived': derived, 'financial_json': financial_json, 'generated': generated, 'arms': arms,
        'projected_reads': {'compressed_source': cost, 'derived_matrix': sum(x['size_bytes'] for x in derived.values())},
        'model_fits': 0, 'preprocessor_fits': 0, 'inference_only_models': 5,
        'source_value_scope': 'Eleven complete approved annual original-key projections, after all annual timing and admission metadata guards.',
        'block_decode_disclosure': 'Selected annual column chunks can decode whole Arrow blocks; no excluded future financial values are converted to Python. Derived memory-map scans are charged once per explicit load by logical file size, not claimed physical I/O.'}
    return save_new(output, body)

def same_scalar(a, b):
    if isinstance(a,float) and math.isnan(a): a = None
    if isinstance(b,float) and math.isnan(b): b = None
    if a is None or b is None:
        return a is None and b is None
    return type(a) is type(b) and a == b


def check_rows(actual, expected, label):
    need(len(actual) == len(expected), label + ' row count changed')
    for i, (a, e) in enumerate(zip(actual, expected)):
        need(set(e) <= set(a), label + ' fields missing')
        for k, value in e.items():
            need(same_scalar(a[k], value), f'{label} row{i} field{k} changed')


def verify_array(item, expected):
    import numpy as np
    p = authenticated(item)
    a = np.load(p, mmap_mode='r', allow_pickle=False)
    need(a.shape == expected.shape and a.dtype == expected.dtype
         and np.array_equal(a, expected, equal_nan=True), 'Prepared numerical array differs from independent source reconstruction')
    return a


def load_frozen_estimator(model_pin, kind):
    path = authenticated(model_pin)
    if kind == 'rf':
        import joblib
        from sklearn.ensemble import RandomForestClassifier
        estimator = joblib.load(path)
        need(type(estimator) is RandomForestClassifier, 'New pinned model is not expected RF type')
    else:
        from xgboost import XGBClassifier
        estimator = XGBClassifier(n_jobs=1, device='cpu')
        estimator.load_model(path)
        estimator.set_params(n_jobs=1, device='cpu')
    return estimator


def validate_fit_metadata(fitted, arm, year, prepared_pin, source, state, X, y, protocol, recipe, tuned):
    import numpy as np
    kind, bank = ARMS[arm]
    need(fitted['status'] == 'FIT_COMPLETE' and fitted['fit_count'] == 1, 'Model has incomplete/multiple-fit receipt')
    expected_parameters = dict(tuned['explicit_parameters']) if arm == 'rf_tuned_extended' else dict(recipe['estimators'][kind])
    if arm == 'rf_tuned_extended':
        fixed = {2020: (12, 20), 2021: (8, 20), 2022: (12, 100), 2023: (12, 100)}
        need(tuned['status'] == 'COMPLETE' and tuned['stage'] == 'outer' and tuned['outer_year'] == year
             and (expected_parameters['max_depth'], expected_parameters['min_samples_leaf']) == fixed[year]
             and expected_parameters['n_estimators'] == 500 and expected_parameters['n_jobs'] == 4,
             'Prior frozen tuned setting changed; no new search allowed')
    else:
        need(expected_parameters['n_jobs'] == 1, 'Baseline frozen fit parallelism changed')
    m, e = fitted['metadata'], fitted['expected_metadata']
    need(e['year'] == year and e['arm'] == arm and e['prepared_pin'] == prepared_pin
         and e['source_plan_sha256'] == published_hash(source)
         and e['preprocessor_fingerprint'] == state['fingerprint'], 'Model input chronology/preprocessing binding changed')
    need(m['explicit_parameters'] == expected_parameters, 'Frozen statistical parameters differ')
    need(all(m['effective_parameters'].get(k) == value for k, value in expected_parameters.items()), 'Effective fit parameters differ from frozen recipe')
    inference = {**m['effective_parameters'], 'n_jobs': 1}
    need(m['saved_inference_effective_parameters'] == inference, 'Saved inference parameters changed beyond worker count')
    need(m['native_class_ids'] == (list(CLASSES) if kind == 'rf' else list(range(10)))
         and m['probability_class_ids'] == list(CLASSES), 'Persisted class order changed')
    encoded = np.ascontiguousarray(y if kind == 'rf' else y - 1)
    need(m['train_rows'] == len(X) and m['matrix_width'] == X.shape[1]
         and m['prepared_train_matrix_sha256'] == array_sha(X)
         and m['encoded_training_labels_sha256'] == array_sha(encoded), 'Model training matrices/classes differ')
    for flag in ('test_labels_supplied_to_estimator', 'early_stopping_used',
                 'parameter_search_performed', 'probability_calibration_performed'):
        need(m[flag] is False, 'Unsupported fitting objective/selection: ' + flag)
    return {**expected_parameters, 'n_jobs': 1}


def verify_model_native(estimator, kind, expected_parameters, width):
    need(estimator.get_params(deep=False).get('n_jobs') == 1, 'Model inference n_jobs must equal1')
    need(int(estimator.n_features_in_) == width, 'Native model feature count changed')
    if kind == 'rf':
        actual = estimator.get_params(deep=False)
        need(all(actual.get(k) == value for k, value in expected_parameters.items()), 'RF serialized statistical parameters differ')
        need(len(estimator.estimators_) == expected_parameters['n_estimators'], 'RF serialized tree count differs')
    else:
        booster = estimator.get_booster()
        config = json.loads(booster.save_config())
        learner = config['learner']
        need(learner['objective']['name'] == 'multi:softprob'
             and int(learner['learner_model_param']['num_class']) == 10
             and int(learner['learner_model_param']['num_feature']) == width,
             'XGBoost native objective/class/feature configuration differs')
        need(booster.num_boosted_rounds() == expected_parameters['n_estimators'], 'XGBoost serialized round count differs')
        # UBJ is a model format, not a guarantee that all sklearn training-only
        # get_params fields survive reload. Exact frozen fitting parameters are
        # checked against the pinned recipe, fit receipt and producer code.


def run_verification(plan_path, expected_sha, labels, output):
    import gc
    import os
    need(pin(plan_path)['sha256'] == expected_sha, 'Independent numerical plan changed')
    plan_pin = pin(plan_path); plan = read_json(plan_path)
    need(plan['schema'] == 'dmv2-independent-corrected-model-plan/v1'
         and plan['status'] == 'SEALED_BEFORE_NUMERICAL_READ'
         and plan['worker_pin'] == pin(__file__), 'Independent verifier code/plan changed')
    authenticated(plan['control_pin'])
    from control import ROOT, require_reservation
    need(Path(plan['run']).resolve().is_relative_to(ROOT / 'run'), 'Unexpected shared run/read-ledger namespace')
    expected_categories = {k for k, n in plan['projected_reads'].items() if n}
    need(set(labels) == expected_categories, 'Exactly one precharged reservation per nonzero category required')
    for category in expected_categories:
        require_reservation(ROOT, labels[category], plan['projected_reads'][category], plan_pin, category)
    require_external_guard(plan_pin, plan)
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        need(os.environ.get(name) == '1', 'External one-numeric-thread guard required: ' + name)
    need(not Path(output).exists(), 'Preserve prior independent verification attempt')
    started_pin = save_new(str(output) + '.STARTED.json', {'status': 'ATTEMPT_CONSUMED', 'plan_pin': plan_pin,
                              'model_fits': 0, 'preprocessor_fits': 0})
    protocol = read_json(authenticated(plan['protocol_pin']))
    need(protocol['catalog_pin'] == plan['catalog_pin']
         and protocol['input_verification_pin'] == plan['input_verification_pin'], 'Input proof/protocol changed')
    authenticated(plan['catalog_pin']); authenticated(plan['input_verification_pin']); authenticated(plan['terminal_guard_pin'])
    source = read_json(authenticated(plan['source_plan_pin']))
    prepared = read_json(authenticated(plan['prepared_pin']))
    recipe = read_json(authenticated(protocol['baseline_recipe_pin']))
    tuned = read_json(authenticated(protocol['tuned_fit_manifest_pins'][str(plan['year'])]))
    authenticated(protocol['worker_pin'])
    for item in protocol['helper_pins'].values(): authenticated(item)
    for package, expected in protocol['runtime'].items():
        from importlib.metadata import version
        need(version(package) == expected, 'Frozen runtime changed: ' + package)
    year = plan['year']; annual_metadata = {}; all_meta = []; source_keys = {}
    # Validate the ENTIRE eleven-year supplied scope before numeric source rows.
    for y, spec in plan['annuals'].items():
        rows = {k: read_projected(e).to_pylist() for k, e in spec['metadata'].items()}
        ordered = validate_metadata(rows['features'], rows['coverage'], rows['targets'],
                                    rows['original'], spec['windows'], int(y))
        coverage = {tuple(r[k] for k in KEYS): r for r in rows['coverage']}
        targets = {tuple(r[k] for k in KEYS): r for r in rows['targets']}
        original = {tuple(r[k] for k in KEYS): r for r in rows['original']}
        merged = [{**original[k], **coverage[k], **targets[k]} for k in ordered]
        annual_metadata[y] = {'rows': rows, 'ordered': ordered, 'merged': merged}
        source_keys[y] = ordered
    train_years = [str(y) for y in range(year - 10, year)]
    train_meta = [r for y in train_years for r in annual_metadata[y]['merged']]
    test_meta = annual_metadata[str(year)]['merged']
    need(len(train_meta) and len(test_meta) and max(r['outcome_end'] for r in train_meta) < min(r['outcome_start'] for r in test_meta),
         'Training outcomes overlap prediction outcomes')
    need(all(f'{year-10}-01-01' <= r['outcome_start'] <= r['outcome_end'] <= f'{year-1}-12-31' for r in train_meta), 'Training window widened')
    observed = max(plan['annuals'][y]['observed_through'] for y in train_years)
    expected_generated = {}
    for split, meta in [('train', train_meta), ('test', test_meta)]:
        rows = read_projected(plan['generated'][split + '_ledger_metadata']).to_pylist()
        need(len(rows) == len(meta), 'Generated ledger changed original denominator')
        for r, e in zip(rows, meta):
            for field in plan['generated'][split + '_ledger_metadata']['columns']:
                if field in ORIGINAL_METADATA and field not in e: e[field] = None
            for k, value in e.items(): need(same_scalar(r[k], value), f'Generated {split} ledger changed original metadata {k}')
            need(type(r['feature_complete']) is bool and r['fit_year'] == year
                 and r['training_outcome_start'] == f'{year-10}-01-01'
                 and r['training_outcome_end'] == f'{year-1}-12-31'
                 and r['training_label_observed_through'] == observed,
                 'Generated ledger has future/wrong fold metadata')
            if split == 'train':
                need(type(r['fit_mask']) is bool and r['fit_mask'] == (e['paper_prediction_eligible'] and e['target_available']),
                     'Stored supervised fit mask differs from independent cohort')
        expected_generated[split] = rows
    forecast_metadata_rows = {}
    for arm, spec in plan['arms'].items():
        rows = read_projected(spec['forecast_metadata']).to_pylist()
        forecast_metadata_rows[arm] = rows
        need(len(rows) == len(test_meta), 'Forecast ledger removed original keys')
        for r, e in zip(rows, expected_generated['test']):
            for k, value in e.items():
                need(same_scalar(r[k], value), 'Forecast metadata/cohort changed before probabilities')
            expected_status = 'predicted' if e['paper_prediction_eligible'] else 'withheld_signal_time_baseline'
            need(r['prediction_status'] == expected_status, 'Future label/missingness changed prediction status')
    import numpy as np
    from threadpoolctl import threadpool_limits
    names, paper = plan['predictor_names'], plan['paper_names']
    Xs, ys = {}, {}
    for y, spec in plan['annuals'].items():
        fm = annual_metadata[y]['rows']['features']; tm = annual_metadata[y]['rows']['targets']
        features = read_projected(spec['numeric']['features'])
        check_rows(features.select(KEYS).to_pylist(), fm, 'Feature numerical row keys')
        values = guarded_numeric(features, names)
        classes = read_projected(spec['numeric']['classes'])
        check_rows(classes.select(KEYS).to_pylist(), [{k: r[k] for k in KEYS} for r in tm], 'Target numerical row keys')
        target_values = classes['target_class'].to_pylist()
        available = np.array([r['target_available'] for r in tm], dtype=bool)
        vector = canonical_labels(target_values, available)
        class_map = {tuple(r[k] for k in KEYS): v for r, v in zip(tm, vector)}
        ordered = annual_metadata[y]['ordered']
        labels = np.array([class_map[k] for k in ordered], dtype=np.float64)
        baseline = np.isfinite(values[:, [names.index(n) for n in paper]]).all(axis=1)
        need(np.array_equal(baseline, [r['paper_prediction_eligible'] for r in annual_metadata[y]['merged']]), 'Paper15 finite input cohort changed')
        Xs[y], ys[y] = values, labels
        del features, classes
    train_X = np.concatenate([Xs[y] for y in train_years]); test_X = Xs[str(year)]
    train_y_all = np.concatenate([ys[y] for y in train_years]); test_y_all = ys[str(year)]
    train_keys = [tuple(r[k] for k in KEYS) for r in train_meta]
    test_keys = [tuple(r[k] for k in KEYS) for r in test_meta]
    fit_mask = np.array([r['paper_prediction_eligible'] and r['target_available'] for r in train_meta], dtype=bool)
    prediction_mask = np.array([r['paper_prediction_eligible'] for r in test_meta], dtype=bool)
    need(np.array_equal(np.isfinite(train_y_all), fit_mask), 'Training class mask changed')
    need(set(train_y_all[fit_mask]) == set(CLASSES), 'Training classes missing; zero padding forbidden')
    for split, raw, labels_all, keys in [('train', train_X, train_y_all, train_keys), ('test', test_X, test_y_all, test_keys)]:
        numeric = read_projected(plan['generated'][split + '_ledger_classes'])
        check_rows(numeric.select(KEYS).to_pylist(), [dict(zip(KEYS,k)) for k in keys], 'Prepared ledger class keys')
        value = canonical_labels(numeric['target_class'].to_pylist(), np.isfinite(labels_all))
        need(np.array_equal(value, labels_all, equal_nan=True), 'Prepared ledger classes changed')
        need(np.array_equal(np.isfinite(raw).all(axis=1), [r['feature_complete'] for r in expected_generated[split]]),
             'Stored full-predictor finite mask differs')
    train_y = np.ascontiguousarray(train_y_all[fit_mask], dtype=np.int64)
    verify_array(plan['derived']['train_y'], train_y)
    need(prepared['train_original_rows'] == len(train_meta) and prepared['test_original_rows'] == len(test_meta)
         and prepared['train_fit_rows'] == int(fit_mask.sum()) and prepared['test_prediction_rows'] == int(prediction_mask.sum())
         and prepared['test_labelled_rows'] == int(np.isfinite(test_y_all).sum())
         and prepared['fit_keys_sha256'] == logical([list(k) for k, keep in zip(train_keys,fit_mask) if keep])
         and prepared['prediction_keys_sha256'] == logical([list(k) for k, keep in zip(test_keys,prediction_mask) if keep]),
         'Prepared fold counters/ordered keys do not reproduce')
    del Xs, ys; gc.collect()
    verified_arms = {}; state_pins = {}; prediction_labels = test_y_all[prediction_mask]
    months = [r['outcome_month'] for r, keep in zip(test_meta,prediction_mask) if keep]
    for bank in ('paper15', 'expanded'):
        chosen = paper if bank == 'paper15' else names
        positions = [names.index(n) for n in chosen]
        independent_state = derive_preprocessor(train_X[:, positions], chosen, fit_mask, train_keys, paper,
            f'{year-10}-01-01', f'{year-1}-12-31', observed)
        saved_state = read_json(authenticated(plan['financial_json'][bank + '_manifest']))
        need(saved_state == independent_state, 'Training-only preprocessing manifest/median/drop/schema differs')
        need(prepared['schemas'][bank]['fingerprint'] == independent_state['fingerprint'], 'Prepared schema fingerprint differs')
        tr = prepare_matrix(train_X[:, positions], chosen, independent_state, fit_mask)
        te = prepare_matrix(test_X[:, positions], chosen, independent_state, prediction_mask)
        verify_array(plan['derived'][bank + '_train'], tr); verify_array(plan['derived'][bank + '_test'], te)
        state_pins[bank] = {'fingerprint': independent_state['fingerprint'], 'width': tr.shape[1],
                           'train_matrix_sha256': array_sha(tr), 'prediction_matrix_sha256': array_sha(te)}
        for arm, spec in plan['arms'].items():
            if spec['bank'] != bank: continue
            fitted = read_json(authenticated(spec['fit_pin']))
            attempt = read_json(authenticated(spec['fit_attempt_pin']))
            need(attempt['status'] == 'ATTEMPT_CONSUMED' and attempt['estimator_fits'] == 1
                 and attempt['expected_metadata'] == fitted['expected_metadata'], 'Model attempt/fit completion binding changed')
            expected_parameters = validate_fit_metadata(fitted, arm, year, plan['prepared_pin'], source,
                independent_state, tr, train_y, protocol, recipe, tuned)
            estimator = load_frozen_estimator(spec['model_pin'], spec['kind'])
            verify_model_native(estimator, spec['kind'], expected_parameters, te.shape[1])
            with threadpool_limits(limits=1):
                # RF parameters persist completely; UBJ training-only parameters
                # are validated separately through native model and fit lineage.
                inference_params = expected_parameters if spec['kind'] == 'rf' else {'n_jobs': 1}
                inferred = infer_probabilities(estimator, spec['kind'], te, inference_params)
            verify_array(plan['derived'][arm + '_probabilities'], inferred)
            forecasts = read_projected(spec['forecast_values'])
            check_rows(forecasts.select(KEYS).to_pylist(), [dict(zip(KEYS,k)) for k in test_keys], 'Forecast numerical keys')
            pvalues = guarded_numeric(forecasts, PROBABILITY_NAMES)
            need(np.array_equal(pvalues[prediction_mask], inferred)
                 and np.isnan(pvalues[~prediction_mask]).all(), 'Forecast values changed or ineligible forecast was invented')
            label_values = canonical_labels(forecasts['target_class'].to_pylist(), np.isfinite(test_y_all))
            need(np.array_equal(label_values, test_y_all, equal_nan=True), 'Forecast labels changed')
            receipt = read_json(authenticated(spec['receipt_pin']))
            need(receipt['status'] == 'COMPLETE' and receipt['year'] == year and receipt['arm'] == arm
                 and receipt['fit_receipt_pin'] == spec['fit_pin'] and receipt['prepared_pin'] == plan['prepared_pin']
                 and receipt['estimator_fits'] == receipt['preprocessing_fits'] == 0, 'Prediction completion identity changed')
            for item in receipt['artifacts'].values(): authenticated(item)
            need(receipt['artifacts']['predictions'] == spec['forecast_values']['pin'], 'Saved forecast artifact binding changed')
            need({k: receipt['artifacts']['probabilities'][k] for k in ('path','sha256','size_bytes')} == plan['derived'][arm + '_probabilities'],
                 'Saved probability array binding changed')
            expected_summary, expected_monthly = independent_metrics(inferred, prediction_labels, months)
            summary = read_json(authenticated(spec['summary_pin']))
            assert_close(summary, expected_summary, arm + ' summary')
            assert_close(receipt['summary'], expected_summary, arm + ' receipt summary')
            monthly_table = read_projected(spec['monthly_metrics'])
            need(monthly_table.select(['outcome_month']).to_pylist() == [{'outcome_month':f'{year}-{m:02d}'} for m in range(1,13)],
                 'Monthly metric scope changed before financial conversion')
            assert_close(monthly_table.to_pylist(), expected_monthly, arm + ' monthly metrics')
            verified_arms[arm] = {'kind':spec['kind'], 'feature_set':bank, 'fit_receipt_pin':spec['fit_pin'],
                'prediction_receipt_pin':spec['receipt_pin'], 'model_pin':spec['model_pin'],
                'probabilities_sha256':array_sha(inferred), 'prediction_rows':len(inferred),
                'labelled_rows':int(np.isfinite(prediction_labels).sum()), 'metric_summary':expected_summary,
                'forecast_metadata_sha256': frame_fingerprint(forecast_metadata_rows[arm], spec['forecast_metadata']['columns']),
                'forecast_values_sha256': frame_fingerprint(forecasts.to_pylist(), spec['forecast_values']['columns']),
                'forecast_metadata_columns':spec['forecast_metadata']['columns'],
                'forecast_value_columns':spec['forecast_values']['columns'],
                'reload_inference_exact':True, 'new_estimator_fits':0}
            del estimator, inferred, forecasts, pvalues; gc.collect()
        del tr, te; gc.collect()
    # Every byte source, model, prepared matrix and receipt is preserved.
    for spec in plan['annuals'].values():
        authenticated(spec['target_state_pin'])
        for group in ('metadata','numeric'):
            for entry in spec[group].values():authenticated(entry['pin'])
    for item in [*plan['derived'].values(), *plan['financial_json'].values()]:authenticated(item)
    for name in ('protocol_pin','catalog_pin','input_verification_pin','terminal_guard_pin','source_plan_pin','prepared_pin','worker_pin','control_pin'):
        authenticated(plan[name])
    authenticated(protocol['worker_pin'])
    for item in protocol['helper_pins'].values(): authenticated(item)
    authenticated(protocol['baseline_recipe_pin'])
    authenticated(protocol['tuned_fit_manifest_pins'][str(year)])
    for spec in plan['arms'].values():
        for name in ('fit_pin', 'fit_attempt_pin', 'receipt_pin', 'summary_pin'): authenticated(spec[name])
    need(set(verified_arms) == set(ARMS), 'All five model arms must pass')
    result = {'schema':SCHEMA, 'status':'PASS', 'year':year, 'plan_pin':plan_pin,
        'worker_pin':pin(__file__), 'started_pin':started_pin, 'protocol_pin':plan['protocol_pin'],
        'catalog_pin':plan['catalog_pin'], 'prepared_pin':plan['prepared_pin'],
        'source_plan_pin':plan['source_plan_pin'], 'train_start':f'{year-10}-01-01',
        'train_end':f'{year-1}-12-31', 'training_label_observed_through':observed,
        'train_original_rows':len(train_meta), 'train_fit_rows':int(fit_mask.sum()),
        'test_original_rows':len(test_meta), 'test_prediction_rows':int(prediction_mask.sum()),
        'test_labelled_rows':int(np.isfinite(test_y_all).sum()), 'preprocessing_states':state_pins,
        'arms':verified_arms, 'original_keys_preserved':True, 'all_new_source_and_artifact_pins_preserved':True,
        'model_fits':0, 'preprocessor_fits':0, 'inference_calls':5, 'parameter_search':False,
        'reserved_reads':plan['projected_reads'], 'integrity_hash_io_bytes':HASH_IO_BYTES, 'protected_2024_financial_values_converted':False,
        'scope_limits':['Arithmetic verification of catalog-bound supplied features/classes, not independent recomputation of every daily/context feature formula.',
            'RF serialized parameters and XGB native objective/width/classes/rounds plus frozen fit-code/receipt lineage verified; UBJ does not promise serialization of every sklearn training-only parameter.',
            'Conditional provider-adjusted endpoint labels and provisional identity/adjustment-vintage/knowledge-time/survivorship limitations retained.',
            'No independent estimator refit; exact source matrices and saved-model probability reloads do not prove investable performance.']}
    return save_new(output,result)


def frame_fingerprint(rows, columns):
    def typed(value):
        if isinstance(value, dt.date): return {'date_value':value.isoformat()}
        if isinstance(value,float) and not math.isfinite(value): return {'nonfinite_float':repr(value)}
        if isinstance(value,list): return [typed(x) for x in value]
        if isinstance(value,dict): return {k:typed(x) for k,x in value.items()}
        return value
    return logical({'columns':list(columns),'rows':[[typed(r[c]) for c in columns] for r in rows]})


def make_final_plan(protocol_path, run, output):
    run=Path(run).resolve();p=read_json(protocol_path)
    need(run==Path(p['run_directory']).resolve(), 'Final verification run differs from protocol')
    audit_pin=pin(run/'audit.json');audit=read_json(audit_pin['path'])
    need(audit['schema']=='dmv2-corrected-refit-model-audit/v1'
         and audit['status']=='COMPLETE_PENDING_INDEPENDENT_VERIFICATION'
         and audit['estimator_fits']==20 and audit['preprocessor_fits']==8
         and audit['outer_years']==list(YEARS) and audit['parameter_search'] is False,
         'Final model compilation is incomplete or unapproved')
    need(audit['source_plan_pin'] == pin(run/'read_plan.json'), 'Final model compilation source plan changed')
    receipts={str(y):pin(run.parent/'verification'/f'models_{y}.json') for y in YEARS}
    entries={};cost=sum(x['size_bytes'] for x in receipts.values())
    for arm in ARMS:
        item=pin(run/'stitched'/f'{arm}.parquet')
        need(audit['forecast_artifacts'][arm]==item,'Stitched forecast path/pin differs')
        import pyarrow.parquet as pq
        safe=list(dict.fromkeys([*TARGET_META,*COVERAGE_FIELDS,*ORIGINAL_METADATA,'feature_complete',
            'fit_year','training_outcome_start','training_outcome_end','training_label_observed_through','prediction_status']))
        present=pq.ParquetFile(authenticated(item)).schema_arrow.names
        need(set(present)<=set([*safe,'target_class',*PROBABILITY_NAMES]),'Unknown stitched financial/metadata field')
        meta=footer_projection(item,[n for n in safe if n in present])
        values=footer_projection(item,[*KEYS,'target_class',*PROBABILITY_NAMES])
        entries[arm]={'metadata':meta,'values':values}
        cost+=meta['projected_compressed_bytes']+values['projected_compressed_bytes']
    return save_new(output,{'schema':'dmv2-independent-corrected-model-final-plan/v1',
        'status':'SEALED_BEFORE_NUMERICAL_READ','run':str(run),'protocol_pin':pin(protocol_path),
        'worker_pin':pin(__file__),'control_pin':pin(Path(__file__).parent/'control.py'),
        'source_plan_pin':pin(run/'read_plan.json'),'audit_pin':audit_pin,
        'catalog_pin':p['catalog_pin'],'annual_verification_pins':receipts,'arms':entries,
        'projected_reads':{'compressed_source':cost,'derived_matrix':0},'model_fits':0,'inference_calls':0})


def run_final(plan_path,expected_sha,labels,output):
    import os
    need(pin(plan_path)['sha256']==expected_sha,'Final independent plan changed')
    pp=pin(plan_path);z=read_json(plan_path)
    need(z['schema']=='dmv2-independent-corrected-model-final-plan/v1'
         and z['worker_pin']==pin(__file__),'Final verifier/code identity changed')
    authenticated(z['control_pin'])
    from control import ROOT,require_reservation
    need(Path(z['run']).resolve().is_relative_to(ROOT/'run'),'Final run outside shared namespace')
    need(set(labels)=={'compressed_source'},'One exact final compressed-source reservation required')
    require_reservation(ROOT,labels['compressed_source'],z['projected_reads']['compressed_source'],pp,'compressed_source')
    require_external_guard(pp,z)
    need(all(os.environ.get(n)=='1' for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')),
         'Single numeric thread environment required')
    need(not Path(output).exists(),'Preserve final verification attempt')
    started=save_new(str(output)+'.STARTED.json',{'status':'ATTEMPT_CONSUMED','plan_pin':pp,'model_fits':0,'inference_calls':0})
    protocol=read_json(authenticated(z['protocol_pin']));source=read_json(authenticated(z['source_plan_pin']))
    audit=read_json(authenticated(z['audit_pin']));authenticated(z['catalog_pin'])
    authenticated(protocol['worker_pin'])
    for item in protocol['helper_pins'].values(): authenticated(item)
    annuals={}
    for year,item in z['annual_verification_pins'].items():
        q=read_json(authenticated(item))
        need(q['schema']==SCHEMA and q['status']=='PASS' and q['year']==int(year)
             and q['protocol_pin']==z['protocol_pin'] and q['catalog_pin']==z['catalog_pin']
             and q['source_plan_pin']==z['source_plan_pin'] and q['worker_pin']==z['worker_pin']
             and set(q['arms'])==set(ARMS),
             'Annual independent proof belongs to a different input/code/fold')
        annuals[year]=q
    run=Path(z['run']);need(len(list(run.glob('folds/*/arms/*/FIT_ATTEMPT.json')))==20
        and len(list(run.glob('folds/*/arms/*/FIT_COMPLETE.json')))==20
        and len(list(run.glob('folds/*/DATA_COMPLETE.json')))==4,'Twenty attempts/completions and four prepared folds required')
    metadata={}
    for arm,spec in z['arms'].items():
        rows=read_projected(spec['metadata']).to_pylist()
        keys=[tuple(r[k] for k in KEYS) for r in rows]
        need(len(keys)==len(set(keys)),'Duplicate stitched original keys')
        need({r['outcome_month'] for r in rows}=={f'{y}-{m:02d}' for y in YEARS for m in range(1,13)},'Stitched48-month scope changed')
        for r in rows:
            date(r['signal_date']);date(r['outcome_start']);date(r['outcome_end'])
            year=r['outcome_end'][:4]
            need(year in annuals and r['signal_date'] in source['annuals'][year]['windows'],'Unknown future stitched row')
            w=source['annuals'][year]['windows'][r['signal_date']]
            need(all(r[k]==w[k] for k in ('outcome_month','outcome_start','outcome_end'))
                 and r['signal_date']<r['outcome_start']<=r['outcome_end']<'2024-01-01'
                 and type(r['prediction_eligible']) is bool and type(r['target_available']) is bool
                 and (not r['target_available'] or r['prediction_eligible']), 'Stitched timing/cohort differs before values')
        for year,q in annuals.items():
            subset=[r for r in rows if r['outcome_month'].startswith(year)]
            proof=q['arms'][arm]
            need(all(r[c] is None for r in subset for c in spec['metadata']['columns'] if c not in proof['forecast_metadata_columns']), 'Stitch invented previously absent source metadata')
            need(frame_fingerprint(subset,proof['forecast_metadata_columns'])==proof['forecast_metadata_sha256'],
                 'Stitched annual metadata/order not equal independently verified forecast')
        metadata[arm]=rows
    import numpy as np
    summaries={}
    for arm,spec in z['arms'].items():
        table=read_projected(spec['values']);rows=metadata[arm]
        check_rows(table.select(KEYS).to_pylist(),[{k:r[k] for k in KEYS} for r in rows],'Stitched financial keys')
        values=table.to_pylist()
        for year,q in annuals.items():
            chosen=[r for r,m in zip(values,rows) if m['outcome_month'].startswith(year)]
            need(frame_fingerprint(chosen,spec['values']['columns'])==q['arms'][arm]['forecast_values_sha256'],
                 'Stitched annual numerical probabilities/classes differ from independently reloaded model')
        mask=np.array([r['prediction_eligible'] for r in rows],dtype=bool)
        observed=np.array([r['target_available'] for r in rows],dtype=bool)
        y=canonical_labels(table['target_class'].to_pylist(),observed)
        probs=guarded_numeric(table,PROBABILITY_NAMES)
        need(np.isnan(probs[~mask]).all(),'Ineligible stitched forecast invented')
        summaries[arm]=independent_metrics(probs[mask],y[mask],[r['outcome_month'] for r,take in zip(rows,mask) if take])[0]
    for q in annuals.values():
        authenticated(q['prepared_pin'])
        for a in q['arms'].values():
            for name in ('fit_receipt_pin','prediction_receipt_pin','model_pin'):authenticated(a[name])
    for item in z['annual_verification_pins'].values():authenticated(item)
    for spec in z['arms'].values():authenticated(spec['values']['pin'])
    for annual in source['annuals'].values():
        for group in ('metadata','numeric'):
            for item in annual[group].values(): authenticated(item['pin'])
        authenticated(annual['target_state_pin'])
    authenticated(protocol['worker_pin'])
    for item in protocol['helper_pins'].values(): authenticated(item)
    for name in ('protocol_pin','source_plan_pin','audit_pin','catalog_pin','worker_pin','control_pin'):authenticated(z[name])
    return save_new(output,{'schema':'dmv2-independent-corrected-model-final-verification/v1','status':'PASS',
        'plan_pin':pp,'worker_pin':pin(__file__),'started_pin':started,'protocol_pin':z['protocol_pin'],
        'catalog_pin':z['catalog_pin'],'model_audit_pin':z['audit_pin'],'source_plan_pin':z['source_plan_pin'],
        'annual_verification_pins':z['annual_verification_pins'],'verified_estimator_fits':20,
        'verified_training_only_preprocessing_states':8,'outcome_years':list(YEARS),'outcome_months':48,
        'all_stitched_forecasts_exact_to_independent_annual_proofs':True,'same_cohort_all_arms':True,
        'comparative_prediction_metrics':summaries,'model_fits':0,'preprocessor_fits':0,'inference_calls':0,
        'parameter_search':False,'backtests':0,'reserved_reads':z['projected_reads'],'integrity_hash_io_bytes':HASH_IO_BYTES,
        'scope_limits':['Combines annual independent source/preprocessing/reload proofs and exact saved forecast concatenation, not new model fitting.',
            'Conditional endpoint labels and provisional source identity/adjustment-vintage/survivorship assumptions remain.',
            'All feature formulas were not independently reimplemented; this is not proof of investable performance.']})


def require_external_guard(plan_pin, plan):
    import os
    path = os.environ.get('DMV2_GUARD_LAUNCH_PATH')
    expected = os.environ.get('DMV2_GUARD_LAUNCH_SHA256')
    need(path and expected and pin(path)['sha256'] == expected, 'Authenticated external resource guard required')
    launch = read_json(path)
    need(launch['schema'] == 'dmv2-corrected-refit-phase-launch/v1'
         and launch['protocol_pin'] == plan['protocol_pin']
         and launch['action_plan_pin'] == plan_pin, 'Guard launch differs from exact independent plan')
    command = launch['command']
    need(len(command) >= 2 and Path(command[0]).resolve() == Path(sys.executable).resolve()
         and Path(command[1]).resolve() == Path(__file__).resolve()
         and command[2:] == sys.argv[1:], 'Independent verifier was not dispatched by the sealed command')
    need(plan['worker_pin'] in launch['code_pins'] and plan['control_pin'] in launch['code_pins'],
         'Guard did not pin independent verifier/control code')


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=['plan','run','plan-final','final'])
    ap.add_argument('--root');ap.add_argument('--protocol');ap.add_argument('--run');ap.add_argument('--year',type=int)
    ap.add_argument('--output',required=True);ap.add_argument('--plan');ap.add_argument('--plan-sha256')
    ap.add_argument('--reservation',action='append',default=[])
    a=ap.parse_args()
    if a.command=='plan': result=make_plan(a.root,a.protocol,a.run,a.year,a.output)
    elif a.command=='plan-final': result=make_final_plan(a.protocol,a.run,a.output)
    else:
        labels=dict(x.split('=',1) for x in a.reservation)
        need(len(labels)==len(a.reservation),'Duplicate reservation category')
        result=(run_verification if a.command=='run' else run_final)(a.plan,a.plan_sha256,labels,a.output)
    print(json.dumps(result,sort_keys=True))

if __name__=='__main__':main()
