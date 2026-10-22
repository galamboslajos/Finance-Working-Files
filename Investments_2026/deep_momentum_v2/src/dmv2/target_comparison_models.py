"""Pinned, completion-aware, fixed-RF target comparison; no import-time I/O.

Planning reads only metadata and Parquet footers. Numerical actions are explicit,
retain original prediction cohorts, share one common supervised mask and never
retry an incomplete fit. Financial 2024+ observations are prohibited.
"""
from pathlib import Path
from dataclasses import replace
import hashlib
import importlib.metadata
import json
import math
import os
import signal
import time

YEARS = (2020, 2021, 2022, 2023)
HEADS = ('endpoint', 'cash_long', 'cash_short', 'utility_long', 'utility_short')
HEAD_SPEC = {'endpoint': ('endpoint', 'long'), 'cash_long': ('executable_return', 'long'),
 'cash_short': ('executable_return', 'short'), 'utility_long': ('return_minus_drawdown', 'long'),
 'utility_short': ('return_minus_drawdown', 'short')}
WINNERS = {2020: (12, 20), 2021: (8, 20), 2022: (12, 100), 2023: (12, 100)}
KEYS = ('source_security_id', 'signal_date')
CATALOG_SHA = '7ebc434ae394410ff5f5377a538b0e43eb290d6685f7f9659ef098d8378c7394'
SCHEMA = 'dmv2_unstopped_target_comparison_models_v1'
COVERAGE = (*KEYS, 'intrinio_security_id', 'identity_status', 'source_identity_status',
 'identity_eligible', 'identity_eligibility_policy', 'identity_eligibility_reason',
 'identity_review_required', 'resolution_status', 'paper_prediction_eligible', 'requested',
 'knowledge_time_verified', 'accounting_certified')
OLD_META = (*KEYS, 'outcome_month', 'outcome_start', 'outcome_end', 'identity_eligible',
 'requested', 'paper_prediction_eligible', 'prediction_eligible', 'target_available')
PATH_META = (*KEYS, 'outcome_month', 'entry_date', 'exit_date', 'exit_mark', 'label_available_at',
 'side', 'path_variant', 'target_family', 'prediction_eligible', 'target_available', 'target_status')
HELPERS = ('preprocessing.py', 'features.py', 'paper_features.py', 'pilot_models.py',
 'classifier_pilot.py', 'ret.py', 'walkforward_metrics.py', 'full_path_audit.py')


def need(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return 'sha256:' + hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()


def pin(path):
    p = Path(path).resolve()
    need(p.is_file(), 'Pinned regular file required')
    return {'path': str(p), 'size_bytes': p.stat().st_size, 'sha256': file_hash(p)}


def check(item):
    p = Path(item['path'])
    need(p.is_absolute() and p.is_file() and p.stat().st_size == item['size_bytes']
         and file_hash(p) == item['sha256'], 'Changed pinned bytes: ' + str(p))
    return p


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False); f.write('\n')
        f.flush(); os.fsync(f.fileno())
    return pin(p)


def projected(item, columns):
    import pyarrow.parquet as pq
    p = check(item); pf = pq.ParquetFile(p)
    need(len(columns) == len(set(columns)) and set(columns) <= set(pf.schema_arrow.names),
         'Invalid or missing projected columns')
    cost = sum(pf.metadata.row_group(g).column(c).total_compressed_size
        for g in range(pf.metadata.num_row_groups) for c in range(pf.metadata.row_group(g).num_columns)
        if pf.metadata.row_group(g).column(c).path_in_schema.split('.')[0] in columns)
    return {'pin': item, 'columns': list(columns), 'rows': pf.metadata.num_rows,
            'projected_compressed_bytes': cost}


def _protocol(path):
    value = read(path)
    need(value.get('schema') == 'dmv2_unstopped_target_comparison_v1'
         and value.get('status') == 'SEALED_BEFORE_FINANCIAL_READ', 'Frozen comparison protocol required')
    need(value.get('inner_validation', {}).get('new_search') is False
         and value['inner_validation']['new_inner_fits'] == 0, 'No parameter search permitted')
    need(value['outer_years'] == list(YEARS) and value['heads'] == list(HEADS)
         and value['path_targets']['variant'] == 'unstopped'
         and value['base_predictors'] == 715 and value['corrected_catalog_sha256'] == CATALOG_SHA,
         'Frozen feature/target/year/catalog scope changed')
    expected_limits = dict(source_projection_bytes=4 * 1024**3, derived_read_bytes=32 * 1024**3,
        output_bytes=24 * 1024**3, rss_stop_bytes=16 * 1024**3, active_seconds=10800,
        per_fit_seconds=1200, max_fits=20, max_preprocessors=4, processes=1, fit_threads=4,
        inference_threads=1)
    need(all(value['limits'].get(k) == v for k, v in expected_limits.items()), 'Frozen compute ceilings changed')
    need(value['runtime'] == _runtime(), 'Frozen package runtime changed')
    need(value['fixed_rf']['base']['n_estimators'] == 500 and value['fixed_rf']['base']['max_features'] == 'sqrt'
         and value['fixed_rf']['base']['random_state'] == 1729 and value['fixed_rf']['base']['n_jobs'] == 4
         and value['fixed_rf']['base']['class_weight'] is None, 'Frozen base RF settings changed')
    for year in YEARS:
        need(value['fixed_rf']['by_year'][str(year)] == dict(zip(('max_depth', 'min_samples_leaf'), WINNERS[year])),
             'Inherited annual RF settings changed')
    return value


def _runtime():
    return {name: importlib.metadata.version(name) for name in ('numpy', 'pandas', 'pyarrow',
        'scikit-learn', 'scipy', 'joblib', 'threadpoolctl', 'exchange-calendars')}


def fixed_parameters(year):
    need(type(year) is int and year in YEARS, 'Only approved four outer years')
    depth, leaf = WINNERS[year]
    return dict(bootstrap=True, ccp_alpha=0.0, class_weight=None, criterion='gini',
        max_depth=depth, max_features='sqrt', max_leaf_nodes=None, max_samples=None,
        min_impurity_decrease=0.0, min_samples_leaf=leaf, min_samples_split=2,
        min_weight_fraction_leaf=0.0, monotonic_cst=None, n_estimators=500, n_jobs=4,
        oob_score=False, random_state=1729, verbose=0, warm_start=False)


def _month_paths(audit_root, year):
    import pyarrow.parquet as pq
    annual = read(audit_root / 'years' / str(year) / 'audit.json')
    result = []
    for month in annual['months']:
        checkpoint_path = audit_root / 'years' / str(year) / 'months' / month / 'checkpoint.json'
        cp = read(checkpoint_path); attempt = checkpoint_path.parent / cp['attempt']
        need(attempt.resolve().parent == checkpoint_path.parent.resolve(), 'Unsafe target attempt')
        audit_path = attempt / 'audit.json'; need(file_hash(audit_path) == cp['audit_sha256'], 'Monthly target audit changed')
        report = read(audit_path); item = report['artifacts']['targets.parquet']
        target = attempt / 'targets.parquet'
        need(target.stat().st_size == item['size_bytes'] and file_hash(target) == item['sha256'], 'Monthly target bytes changed')
        tp = pin(target)
        result.append({'outcome_month': month, 'monthly_audit_pin': pin(audit_path),
            'checkpoint_pin': pin(checkpoint_path), 'original_rows': report['original_rows'],
            'prediction_eligible_rows': report['prediction_eligible_rows'],
            'original_key_sha256': report['original_key_sha256'], 'eligible_key_sha256': report['eligible_key_sha256'],
            'metadata': projected(tp, PATH_META), 'values': projected(tp, (*KEYS, 'side', 'path_variant', 'target_family', 'target_class'))})
    return result


def plan_comparison(project_root, target_audit, protocol_path, output):
    """Metadata/footer-only plan. ``output`` is a fresh private run directory."""
    from .features import registered_feature_specs
    from .paper_features import PAPER_FEATURE_NAMES
    from .full_path_audit import read_full_path_audit
    project, audit_root, out = map(lambda p: Path(p).resolve(), (project_root, target_audit, output))
    need(not out.exists(), 'Preserve existing attempt; use a fresh run directory')
    protocol = _protocol(protocol_path)
    audit = read_full_path_audit(audit_root)
    proof_path = audit_root / 'independent_full_verification.json'; proof = read(proof_path)
    need(proof.get('status') == 'PASS_FULL_FORMULAS_RANKS_CLOCK_AND_ORIGINAL_DENOMINATORS', 'Independent target proof required')
    need(proof.get('audit_sha256') == file_hash(audit_root / 'audit.json'), 'Target proof belongs to another audit')
    catalog_path = project / 'outputs/corrected_refit/2020-2023-20261021-v1/input_catalog.json'
    need(file_hash(catalog_path) == CATALOG_SHA, 'Only latest corrected learning catalog allowed')
    catalog = read(catalog_path); need(catalog['status'] == 'VERIFIED_PROVISIONAL_FROZEN', 'Corrected source freeze missing')
    need(read(check(catalog['independent_verification']))['status'] == 'PASS', 'Independent corrected-input proof missing')
    reviewed = read(audit_root / 'reviewed_plan.json')
    specs = registered_feature_specs(); predictors = [x['name'] for x in specs if x.get('model_eligible', True)]
    need(len(predictors) == 715 and len(set(predictors)) == 715, 'Frozen 715 predictors required')
    annuals = {}
    for row in catalog['annuals']:
        y = row['year']; need(type(y) is int and 2010 <= y <= 2023, 'Source annual outside study')
        src = row['artifacts']; source_year = next(a for a in reviewed['annuals'] if a['year'] == y)
        import pyarrow.parquet as pq
        original_columns = pq.ParquetFile(check(src['original_cohort'])).schema_arrow.names
        annuals[str(y)] = {'year': y, 'windows': source_year['windows'],
            'original_rows': source_year['original_rows'], 'prediction_eligible_rows': source_year['prediction_eligible_rows'],
            'feature_keys': projected(src['features'], KEYS),
            'feature_values': projected(src['features'], (*KEYS, *predictors)),
            'coverage': projected(src['feature_coverage'], COVERAGE),
            'original': projected(src['original_cohort'], original_columns),
            'endpoint_metadata': projected(src['targets'], OLD_META),
            'endpoint_values': projected(src['targets'], (*KEYS, 'target_class')),
            'path_months': _month_paths(audit_root, y)}
    need(set(annuals) == {str(y) for y in range(2010, 2024)}, 'Fourteen source annuals required')
    folds = {}
    sessions = reviewed['sessions']
    for year in YEARS:
        ys = list(range(year - 10, year + 1))
        cutoff = max(s for s in sessions if s <= f'{year-1}-12-31')
        cost = sum(annual_cost(annuals[str(y)]) for y in ys)
        folds[str(year)] = {'year': year, 'annual_years': ys, 'train_start': f'{year-10}-01-01',
            'train_end': f'{year-1}-12-31', 'as_of': cutoff,
            'prediction_start': f'{year}-01-01', 'prediction_end': f'{year}-12-31',
            'source_projection_bytes': cost, 'fixed_rf_parameters': fixed_parameters(year)}
    source_cost = sum(x['source_projection_bytes'] for x in folds.values())
    need(source_cost <= 4 * 1024 ** 3, 'Planned source projection exceeds 4GiB')
    body = {'schema': SCHEMA, 'status': 'SEALED_BEFORE_FINANCIAL_READ', 'project_root': str(project),
        'target_audit_directory': str(audit_root), 'output_directory': str(out),
        'protocol_pin': pin(protocol_path), 'catalog_pin': pin(catalog_path),
        'input_verification_pin': catalog['independent_verification'],
        'target_audit_pin': pin(audit_root / 'audit.json'), 'target_plan_pin': pin(audit_root / 'reviewed_plan.json'),
        'target_verification_pin': pin(proof_path), 'worker_pin': pin(__file__),
        'helper_pins': {name: pin(Path(__file__).parent / name) for name in HELPERS},
        'runtime': _runtime(), 'annuals': annuals, 'folds': folds, 'predictor_names': predictors,
        'paper_names': list(PAPER_FEATURE_NAMES), 'heads': list(HEADS), 'expected_fits': 20,
        'expected_preprocessors': 4, 'source_projection_bytes': source_cost,
        'financial_2024_plus_allowed': False, 'parameter_search': False,
        'primary_economic_period': ['2020-02', '2023-12'],
        'read_budget': comparison_read_budget(annuals, folds),
        'block_decode_disclosure': 'Selected columns may decode full Arrow blocks; only validated pre2024 annual rows reach Python financial values.'}
    body['plan_sha256'] = canonical(body)
    out.mkdir(parents=True); save(out / 'read_plan.json', body)
    return body


def comparison_read_budget(annuals, folds):
    """Metadata-derived matrix bounds; actual byte receipts stay separate."""
    by_year = {}
    for key, fold in folds.items():
        train_n = sum(annuals[str(y)]['prediction_eligible_rows'] for y in fold['annual_years'][:-1])
        pred_n = annuals[key]['prediction_eligible_rows']
        # At most715 retained features plus715 fixed missing indicators.
        # NPY header ceiling4096 perfile; ledgers are charged conservatively
        # from metadata/source projections. No model reload is needed for an
        # uninterrupted fit+predict phase; explicit inference resume charges its
        # actual saved model bytes separately before dispatch.
        train = train_n * 1430 * 4 + 4096
        prediction = pred_n * 1430 * 4 + 4096
        labels = train_n * 8 + 4096
        model_reads = 5 * (train + prediction + labels)
        metadata_bytes = sum(sum(annuals[str(y)][n]['projected_compressed_bytes'] for n in
            ('feature_keys', 'coverage', 'original', 'endpoint_metadata')) +
            sum(m['metadata']['projected_compressed_bytes'] for m in annuals[str(y)]['path_months'])
            for y in fold['annual_years'])
        by_year[key] = dict(source_projection_bytes=fold['source_projection_bytes'],
            metadata_projection_bytes=metadata_bytes, derived_prepared_read_bytes=model_reads,
            prepared_training_matrix_upper_bytes=train, prepared_prediction_matrix_upper_bytes=prediction)
    result = dict(source_projection_bytes_total=sum(x['source_projection_bytes'] for x in by_year.values()),
        derived_prepared_read_bytes_total=sum(x['derived_prepared_read_bytes'] for x in by_year.values()),
        metadata_projection_bytes=sum(x['metadata_projection_bytes'] for x in by_year.values()),
        independent_verification_additional_reserved_bytes=12 * 1024**3,
        by_year=by_year, repeated_rolling_source_reads_counted=True,
        integrity_hashing_separate_from_financial_decode=True)
    need(result['derived_prepared_read_bytes_total'] + result['independent_verification_additional_reserved_bytes'] <= 32 * 1024**3, 'Derived model and verification read reservations exceed32GiB')
    return result


def annual_cost(annual):
    return sum(annual[k]['projected_compressed_bytes'] for k in ('feature_keys', 'feature_values', 'coverage',
        'original', 'endpoint_metadata', 'endpoint_values')) + sum(m['metadata']['projected_compressed_bytes']
        + m['values']['projected_compressed_bytes'] for m in annual['path_months'])


def verify_plan(plan):
    need(plan.get('schema') == SCHEMA and plan.get('plan_sha256') == canonical({k: v for k, v in plan.items() if k != 'plan_sha256'}), 'Changed comparison plan')
    need(plan['heads'] == list(HEADS) and plan['expected_fits'] == 20 and plan['expected_preprocessors'] == 4,
         'Frozen five-head/four-year scope changed')
    need(plan['runtime'] == _runtime(), 'Frozen runtime changed')
    for item in [plan[k] for k in ('protocol_pin', 'catalog_pin', 'input_verification_pin', 'target_audit_pin',
            'target_plan_pin', 'target_verification_pin', 'worker_pin')]: check(item)
    need(plan['worker_pin'] == pin(__file__), 'Worker redirect or code change')
    need(plan['catalog_pin']['sha256'] == CATALOG_SHA and plan['financial_2024_plus_allowed'] is False, 'Protected/catalog gate')
    for name, item in plan['helper_pins'].items(): need(item == pin(Path(__file__).parent / name), 'Helper code changed')
    _protocol(check(plan['protocol_pin']))
    return plan


def _read_projection(spec):
    import pyarrow.parquet as pq
    table = pq.ParquetFile(check(spec['pin'])).read(columns=spec['columns'])
    need(table.num_rows == spec['rows'], 'Pinned table rowcount changed')
    return table


def _index(frame):
    need(not frame.duplicated(list(KEYS)).any() and not frame[list(KEYS)].isna().any().any(), 'Original keys missing/duplicated')
    return frame.set_index(list(KEYS))


def _keys_digest(index):
    return canonical([list(k) for k in sorted(index)])


def validate_annual_metadata(annual, feature_keys, coverage, original, endpoint, paths):
    """All metadata clocks/admission validated before financial conversion."""
    import pandas as pd
    y = annual['year']; windows = {w['signal_date']: w for w in annual['windows']}
    need(len(windows) == 12 and {w['outcome_month'] for w in windows.values()} == {f'{y}-{m:02d}' for m in range(1, 13)}, 'Annual monthly window changed')
    tables = [_index(x) for x in (feature_keys, coverage, original, endpoint)]
    keys = set(tables[0].index)
    need(len(keys) == annual['original_rows'] and all(set(x.index) == keys for x in tables), 'Original annual cohort mismatch')
    cov, old = tables[1], tables[3]
    for sid, signal in keys:
        need(isinstance(sid, str) and bool(sid.strip()) and signal in windows and signal < '2024-01-01', 'Source key outside explicit annual window')
    for col in ('identity_eligible', 'requested', 'paper_prediction_eligible'):
        need(cov[col].map(lambda x: type(x) is bool).all() and cov[col].equals(old[col]), 'Admission differs or is not boolean')
    need(old.prediction_eligible.equals(cov.paper_prediction_eligible)
         and old.target_available.map(lambda x: type(x) is bool).all(), 'Endpoint prediction cohort/availability changed')
    for row in old.reset_index().to_dict('records'):
        w = windows[row['signal_date']]
        need(all(row[k] == w[k] for k in ('outcome_month', 'outcome_start', 'outcome_end')), 'Endpoint target clock changed')
    need(int(cov.paper_prediction_eligible.sum()) == annual['prediction_eligible_rows'], 'Eligible count changed')
    for month, meta in paths:
        need(meta.side.isin(('long', 'short')).all() and meta.path_variant.isin(('unstopped', 'asset_stop_10pct')).all()
            and meta.target_family.isin(('executable_return', 'return_minus_drawdown')).all(), 'Path head selector changed')
        need(not meta.duplicated([*KEYS, 'side', 'path_variant', 'target_family']).any(), 'Duplicate path group keys')
        need(len(meta) == month['original_rows'] * 8, 'Path original denominator changed')
        for (side, variant, family), group in meta.groupby(['side', 'path_variant', 'target_family'], sort=False):
            g = _index(group); signal = month['outcome_month']; w = next(w for w in windows.values() if w['outcome_month'] == signal)
            expected = cov.loc[cov.index.get_level_values('signal_date') == w['signal_date']]
            need(set(g.index) == set(expected.index), 'Path group original cohort changed')
            g = g.reindex(expected.index)
            need(g.prediction_eligible.equals(expected.paper_prediction_eligible), 'Future availability changed prediction cohort')
            need(_keys_digest(g.index) == month['original_key_sha256']
                and _keys_digest(g.index[g.prediction_eligible]) == month['eligible_key_sha256'], 'Path group original key identity changed')
            for row in g.reset_index().to_dict('records'):
                need(row['outcome_month'] == w['outcome_month'] and row['entry_date'] == w['entry_date']
                    and row['exit_date'] == row['label_available_at'] == w['exit_date'] and row['exit_mark'] == 'open', 'Actual path completion clock changed')
                need(type(row['target_available']) is bool and (not row['target_available'] or row['prediction_eligible']), 'Invalid target admission')
                need(row['exit_date'] < '2024-01-01' or (row['outcome_month'] == '2023-12' and not row['target_available']), 'Protected target availability')
    return cov, old


def _annual_values(annual, predictors):
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.compute as pc
    metadata = {k: _read_projection(annual[k]).to_pandas() for k in ('feature_keys', 'coverage', 'original', 'endpoint_metadata')}
    paths = [(m, _read_projection(m['metadata']).to_pandas()) for m in annual['path_months']]
    cov, old = validate_annual_metadata(annual, metadata['feature_keys'], metadata['coverage'], metadata['original'], metadata['endpoint_metadata'], paths)
    # All annual keys, calendar bounds and actual completion dates admitted first.
    feature_table = _read_projection(annual['feature_values'])
    allowed = pa.array(list({w['signal_date'] for w in annual['windows']}))
    feature_table = feature_table.filter(pc.is_in(feature_table['signal_date'], value_set=allowed))
    need(feature_table.num_rows == annual['original_rows'], 'Unadmitted feature financial rows')
    X = _index(feature_table.to_pandas()).reindex(cov.index)[predictors]
    need(X.shape == (len(cov), 715), 'Frozen715 feature projection changed')
    for col in X:
        need(not X[col].map(lambda v: isinstance(v, (str, bool, np.bool_))).any(), 'Feature must be numeric or missing')
    numeric = X.astype(float)
    from .paper_features import PAPER_FEATURE_NAMES
    need(pd.Series(np.isfinite(numeric[list(PAPER_FEATURE_NAMES)]).all(axis=1), index=X.index).equals(cov.paper_prediction_eligible), 'Paper15 finite features disagree with signal cohort')
    old_values = _index(_read_projection(annual['endpoint_values']).to_pandas()).reindex(X.index)
    labels = {'endpoint': old_values.target_class.astype(float)}
    ledgers = {}
    original = _index(metadata['original']).reindex(X.index)
    base = original.join(cov[[c for c in cov if c not in original]])
    base = base.join(old[[c for c in ('outcome_month', 'outcome_start', 'outcome_end') if c not in base]])
    base['prediction_eligible'] = cov.paper_prediction_eligible
    base['paper_prediction_eligible'] = cov.paper_prediction_eligible
    endpoint = base.copy(); endpoint['target_available'] = old.target_available
    endpoint['label_available_at'] = endpoint.outcome_end
    endpoint['side'] = 'long'; endpoint['target_family'] = 'endpoint'; endpoint['path_variant'] = 'endpoint_control'
    endpoint['target_class'] = labels['endpoint']; ledgers['endpoint'] = endpoint
    chunks = {head: [] for head in HEADS[1:]}
    for monthly, meta in paths:
        t = _read_projection(monthly['values'])
        for head in HEADS[1:]:
            family, side = HEAD_SPEC[head]
            mask = pc.and_(pc.equal(t['path_variant'], 'unstopped'), pc.and_(pc.equal(t['side'], side), pc.equal(t['target_family'], family)))
            values = _index(t.filter(mask).select([*KEYS, 'target_class']).to_pandas())
            selected = _index(meta[(meta.path_variant == 'unstopped') & (meta.side == side) & (meta.target_family == family)])
            need(set(values.index) == set(selected.index), 'Path value keys differ from admitted metadata')
            selected['target_class'] = values.reindex(selected.index).target_class.astype(float)
            chunks[head].append(selected)
    for head in HEADS[1:]:
        path = pd.concat(chunks[head]).reindex(X.index)
        frame = base.copy()
        for col in ('side', 'target_family', 'path_variant', 'entry_date', 'exit_date', 'exit_mark', 'label_available_at', 'target_available', 'target_class', 'target_status'):
            frame[col] = path[col]
        labels[head] = frame.target_class.astype(float); ledgers[head] = frame
    for head in HEADS:
        available = ledgers[head].target_available.astype(bool)
        need(labels[head].notna().equals(available) and labels[head][available].isin(range(1, 11)).all(), 'Available labels must be canonical classes')
    return numeric, labels, ledgers


def common_fit_mask(ledgers, *, train_start, train_end, as_of):
    """Intersection affects training only; no prediction-cohort selection."""
    import numpy as np
    import pandas as pd
    need(set(ledgers) == set(HEADS) and train_start <= train_end <= '2023-12-31' and as_of <= train_end, 'Invalid explicit training bounds')
    index = ledgers['endpoint'].index
    need(all(t.index.equals(index) for t in ledgers.values()), 'Five target ledgers must align')
    result = pd.Series(True, index=index)
    for head in HEADS:
        frame = ledgers[head]
        need(frame.prediction_eligible.map(lambda v: isinstance(v, (bool, np.bool_))).all()
             and frame.target_available.map(lambda v: isinstance(v, (bool, np.bool_))).all(),
             'Training admission flags must be explicit booleans')
        need(not frame.target_class.map(lambda v: isinstance(v, (str, bool, np.bool_))).any(),
             'Training classes cannot be text or boolean values')
        prediction = frame.prediction_eligible.astype(bool)
        available = frame.target_available.astype(bool)
        complete = (frame.outcome_start >= train_start) & (frame.outcome_end <= train_end) & (frame.label_available_at <= as_of)
        values = pd.to_numeric(frame.target_class, errors='coerce')
        valid = available & np.isfinite(values) & values.isin(range(1, 11))
        need(not (available & ~prediction).any(), 'Future target admission exceeds signal cohort')
        result &= prediction & valid & complete
    return result


def array_hash(array):
    import numpy as np
    a = np.ascontiguousarray(array)
    return 'sha256:' + hashlib.sha256(memoryview(a).cast('B') if a.size else b'').hexdigest()


def save_array(path, array):
    import numpy as np
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('xb') as f:
        np.save(f, array, allow_pickle=False); f.flush(); os.fsync(f.fileno())
    return {**pin(p), 'shape': list(array.shape), 'dtype': str(array.dtype), 'array_sha256': array_hash(array)}


def save_frame(path, frame):
    import pyarrow as pa
    import pyarrow.parquet as pq
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('xb') as f:
        pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), f, compression='zstd')
    return pin(p)


def _resource(resource_check):
    if resource_check:
        resource_check()


def _prepared(plan, year):
    directory = Path(plan['output_directory']) / 'folds' / str(year) / 'prepared'
    receipt = read(directory / 'receipt.json')
    need(receipt['status'] == 'PREPARED' and receipt['plan_sha256'] == plan['plan_sha256'] and receipt['year'] == year, 'Prepared fold identity changed')
    for item in receipt['artifacts'].values(): check(item)
    return directory, receipt


def prepare_fold(plan, year, resource_check=None):
    import gc
    import numpy as np
    import pandas as pd
    from .features import TrainingSlice
    from .preprocessing import fit_preprocessor
    from .classifier_pilot import _finite32
    verify_plan(plan); need(type(year) is int and year in YEARS, 'Unapproved fold')
    fold = plan['folds'][str(year)]; out = Path(plan['output_directory']) / 'folds' / str(year) / 'prepared'
    need(not out.exists(), 'Preserve existing preparation attempt')
    save(out / 'STARTED.json', {'status': 'PREPARATION_CONSUMED', 'year': year, 'plan_sha256': plan['plan_sha256'], 'preprocessor_fits': 1})
    parts = {}
    for y in fold['annual_years']:
        _resource(resource_check); parts[y] = _annual_values(plan['annuals'][str(y)], plan['predictor_names'])
    train_years = fold['annual_years'][:-1]
    train_X = pd.concat([parts[y][0] for y in train_years]); test_X = parts[year][0]
    train_ledgers = {head: pd.concat([parts[y][2][head] for y in train_years]) for head in HEADS}
    test_ledgers = parts[year][2]
    need(train_X.index.is_unique and test_X.index.is_unique and not (set(train_X.index) & set(test_X.index)), 'Training and prediction key overlap')
    mask = common_fit_mask(train_ledgers, train_start=fold['train_start'], train_end=fold['train_end'], as_of=fold['as_of'])
    need(mask.any(), 'No common completed supervised rows')
    merged = train_ledgers['endpoint'].copy()
    for head in HEADS:
        f = train_ledgers[head]
        for name in ('target_class', 'target_available', 'label_available_at'):
            merged[head + '__' + name] = f[name]
    merged['common_fit_mask'] = mask
    # Calendar outcome_end remains true; actual next-open completion was checked
    # independently in common_fit_mask and remains explicit per head in ledger.
    y = train_ledgers['endpoint'].target_class.rename('endpoint_class')
    provenance = {'train_start': fold['train_start'], 'train_end': fold['train_end'], 'label_observed_through': fold['as_of']}
    training = TrainingSlice(train_X, y, train_ledgers['endpoint'], provenance)
    _resource(resource_check)
    fitted = fit_preprocessor(training, fit_mask=mask)
    a = fitted.transform(train_X, prediction_mask=mask)
    predicted_mask = test_ledgers['endpoint'].prediction_eligible.astype(bool)
    b = fitted.transform(test_X, prediction_mask=predicted_mask)
    need(a.ledger.transformed.equals(mask) and b.ledger.transformed.equals(predicted_mask), 'Preprocessing changed declared cohorts')
    artifacts = {'train_X': save_array(out / 'train_X.npy', _finite32(a.X.loc[mask])),
        'prediction_X': save_array(out / 'prediction_X.npy', _finite32(b.X.loc[predicted_mask])),
        'raw_train_X': save_array(out / 'raw_train_X.npy', train_X.loc[mask].to_numpy(dtype=np.float64)),
        'raw_prediction_X': save_array(out / 'raw_prediction_X.npy', test_X.to_numpy(dtype=np.float64)),
        'train_ledger': save_frame(out / 'train_ledger.parquet', merged.reset_index()),
        'preprocessor': save(out / 'preprocessor.json', {'manifest': fitted.manifest, 'fingerprint': fitted.fingerprint})}
    heads = {}
    for head in HEADS:
        train_y = train_ledgers[head].loc[mask].target_class.to_numpy(dtype=np.int64)
        need(set(train_y) == set(range(1, 11)), 'Every fitted head requires all ten observed classes')
        artifacts['train_y_' + head] = save_array(out / ('train_y_' + head + '.npy'), train_y)
        test_ledger = test_ledgers[head].copy(); test_ledger['head'] = head; test_ledger['fit_year'] = year
        test_ledger['training_outcome_start'] = fold['train_start']; test_ledger['training_outcome_end'] = fold['train_end']
        test_ledger['training_label_observed_through'] = fold['as_of']
        artifacts['prediction_ledger_' + head] = save_frame(out / ('prediction_ledger_' + head + '.parquet'), test_ledger.reset_index())
        original_fit = train_ledgers[head].prediction_eligible & train_ledgers[head].target_available
        heads[head] = {'available_before_completion_gate': int(original_fit.sum()),
            'own_completed_rows': int((original_fit & (train_ledgers[head].label_available_at <= fold['as_of'])).sum()),
            'common_fit_rows': int(mask.sum()), 'completion_purged_rows': int((original_fit & (train_ledgers[head].label_available_at > fold['as_of'])).sum())}
    result = {'schema': SCHEMA, 'status': 'PREPARED', 'plan_sha256': plan['plan_sha256'], 'year': year,
        'artifacts': artifacts, 'heads': heads, 'preprocessor_fits': 1, 'estimator_fits': 0,
        'train_original_rows': len(train_X), 'train_fit_rows': int(mask.sum()), 'prediction_original_rows': len(test_X),
        'prediction_rows': int(predicted_mask.sum()), 'matrix_width': len(fitted.feature_names),
        'fit_keys_sha256': canonical([list(k) for k in train_X.index[mask]]),
        'prediction_keys_sha256': canonical([list(k) for k in test_X.index[predicted_mask]]),
        'primary_economic_period': ['2020-02', '2023-12'],
        'raw_prediction_keys_sha256': canonical([list(k) for k in test_X.index]),
        'preprocessor_fingerprint': fitted.fingerprint, 'source_projection_bytes': fold['source_projection_bytes'], 'resource_phase': 'prepare',
        'derived_reads_for_fit_heads_bytes': 5 * artifacts['train_X']['size_bytes']
            + sum(artifacts['train_y_' + h]['size_bytes'] for h in HEADS)
            + 5 * artifacts['prediction_X']['size_bytes'] + sum(artifacts['prediction_ledger_' + h]['size_bytes'] for h in HEADS)}
    _resource(resource_check)
    return save(out / 'receipt.json', result)


def fit_head(plan, year, head, resource_check=None):
    """One fit then saved-model prediction; complete fit may resume inference only."""
    import numpy as np
    import pandas as pd
    from joblib import dump, load, parallel_backend
    from sklearn.ensemble import RandomForestClassifier
    from threadpoolctl import threadpool_limits
    from .pilot_models import _json_parameters
    from .ret import align_class_probabilities
    from .classifier_pilot import _predictions
    from .walkforward_metrics import evaluate_probabilities
    verify_plan(plan); need(type(year) is int and year in YEARS and head in HEADS, 'Unapproved model head')
    directory, prepared = _prepared(plan, year)
    out = Path(plan['output_directory']) / 'folds' / str(year) / 'heads' / head
    need(not (out / 'PREDICT_STARTED.json').exists(), 'Preserve consumed incomplete/completed prediction attempt')
    _resource(resource_check)
    started = time.monotonic()
    if (out / 'FIT_COMPLETE.json').exists():
        fitted = read(out / 'FIT_COMPLETE.json')
        need(fitted['prepared_pin'] == pin(directory / 'receipt.json') and fitted['head'] == head and fitted['year'] == year, 'Completed fit identity changed')
        estimator = load(check(fitted['model']))
    else:
        need(not (out / 'FIT_ATTEMPT.json').exists(), 'Incomplete fit is consumed; no automatic retry')
        need(len(list(Path(plan['output_directory']).glob('folds/*/heads/*/FIT_ATTEMPT.json'))) < 20, 'Twenty estimator attempts exhausted')
        X = np.load(check(prepared['artifacts']['train_X']), mmap_mode='r', allow_pickle=False)
        y = np.load(check(prepared['artifacts']['train_y_' + head]), mmap_mode='r', allow_pickle=False)
        need(X.dtype == np.float32 and X.shape == (prepared['train_fit_rows'], prepared['matrix_width']) and np.isfinite(X).all(), 'Prepared train matrix changed')
        need(y.dtype == np.int64 and y.shape == (len(X),) and set(y) == set(range(1, 11)), 'Canonical complete training labels required')
        params = fixed_parameters(year)
        save(out / 'FIT_ATTEMPT.json', {'status': 'FIT_CONSUMED', 'year': year, 'head': head,
            'plan_sha256': plan['plan_sha256'], 'prepared_pin': pin(directory / 'receipt.json'), 'estimator_fits': 1})
        estimator = RandomForestClassifier(**params)
        previous = signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Frozen per-fit1200second budget exhausted')))
        signal.setitimer(signal.ITIMER_REAL, 1200)
        try:
            with threadpool_limits(limits=1), parallel_backend('threading', n_jobs=4):
                estimator.fit(X, y)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0); signal.signal(signal.SIGALRM, previous)
        native = [int(k) for k in estimator.classes_]
        need(native == list(range(1, 11)), 'Canonical ten-class map changed')
        effective = _json_parameters(estimator.get_params(deep=False)); estimator.set_params(n_jobs=1)
        model = out / 'model.joblib'
        with model.open('xb') as f:
            dump(estimator, f, compress=0); f.flush(); os.fsync(f.fileno())
        fitted = {'schema': SCHEMA, 'status': 'FIT_COMPLETE', 'plan_sha256': plan['plan_sha256'], 'year': year, 'head': head,
            'prepared_pin': pin(directory / 'receipt.json'), 'model': pin(model), 'estimator_fits': 1,
            'preprocessor_fingerprint': prepared['preprocessor_fingerprint'], 'explicit_parameters': params,
            'fit_effective_parameters': effective, 'saved_inference_effective_parameters': _json_parameters(estimator.get_params(deep=False)),
            'native_class_ids': native, 'probability_class_ids': list(range(1, 11)),
            'train_matrix_sha256': array_hash(X), 'train_labels_sha256': array_hash(y),
            'fit_rows': len(y), 'matrix_width': X.shape[1], 'fit_seconds': time.monotonic() - started,
            'parameter_search': False, 'calibration': False, 'validation_labels_supplied_to_estimator': False}
        save(out / 'FIT_COMPLETE.json', fitted)
    _resource(resource_check)
    save(out / 'PREDICT_STARTED.json', {'status': 'PREDICT_CONSUMED', 'fit_pin': pin(out / 'FIT_COMPLETE.json'), 'estimator_fits': 0})
    need(estimator.get_params(deep=False)['n_jobs'] == 1 and [int(k) for k in estimator.classes_] == list(range(1, 11)), 'Saved inference convention changed')
    matrix = np.load(check(prepared['artifacts']['prediction_X']), mmap_mode='r', allow_pickle=False)
    ledger = pd.read_parquet(check(prepared['artifacts']['prediction_ledger_' + head])).set_index(list(KEYS))
    expected = ledger.index[ledger.prediction_eligible]
    need(canonical([list(k) for k in expected]) == prepared['prediction_keys_sha256'] and matrix.shape == (len(expected), prepared['matrix_width']) and np.isfinite(matrix).all(), 'Prediction population/matrix mismatch')
    with threadpool_limits(limits=1):
        probabilities = align_class_probabilities(estimator.predict_proba(matrix), class_ids=estimator.classes_)
    frame = _predictions(ledger, probabilities, expected)
    evaluation = evaluate_probabilities(probabilities, ledger.loc[expected].target_class.reset_index(drop=True), class_ids=tuple(range(1, 11)),
        row_metadata=ledger.loc[expected].reset_index(), first_month=f'{year}-01', last_month=f'{year}-12')
    artifacts = {'predictions': save_frame(out / 'predictions.parquet', frame),
        'probabilities': save_array(out / 'probabilities.npy', probabilities),
        'summary': save(out / 'reports/summary.json', evaluation.summary)}
    for name, table in evaluation.tables.items(): artifacts[name] = save_frame(out / 'reports' / (name + '.parquet'), table.reset_index(drop=True))
    _resource(resource_check)
    return save(out / 'PREDICT_COMPLETE.json', {'schema': SCHEMA, 'status': 'PREDICT_COMPLETE', 'plan_sha256': plan['plan_sha256'],
        'year': year, 'head': head, 'fit_pin': pin(out / 'FIT_COMPLETE.json'), 'prepared_pin': pin(directory / 'receipt.json'),
        'artifacts': artifacts, 'summary': evaluation.summary, 'estimator_fits': 0, 'prediction_rows': len(expected), 'resource_phase': 'fit_predict',
        'derived_prepared_read_bytes': prepared['artifacts']['train_X']['size_bytes']
            + prepared['artifacts']['train_y_' + head]['size_bytes']
            + prepared['artifacts']['prediction_X']['size_bytes']
            + prepared['artifacts']['prediction_ledger_' + head]['size_bytes']})


def report_models(plan):
    """Small saved metric/receipt reader; never loads models/matrices/panels."""
    verify_plan(plan)
    result = []
    for year in YEARS:
        for head in HEADS:
            path = Path(plan['output_directory']) / 'folds' / str(year) / 'heads' / head / 'PREDICT_COMPLETE.json'
            item = read(path)
            need(item['status'] == 'PREDICT_COMPLETE' and item['plan_sha256'] == plan['plan_sha256']
                and item['year'] == year and item['head'] == head, 'Incomplete/wrong-model metric receipt')
            result.append({'year': year, 'head': head, **item['summary']})
    return {'status': 'MODELS_COMPLETE_PENDING_INDEPENDENT_VERIFICATION', 'estimator_fits': 20,
        'preprocessor_fits': 4, 'rows': result, 'parameter_search': False, 'financial_2024_plus_read': False}
