"""Model-only impurity importance for the verified four annual endpoint forests.

No import-time I/O or numerical imports. Planning authenticates JSON and bytes,
never deserializes a model. Extraction explicitly loads four pinned local models
sequentially; it never opens feature/price panels or calls fit/predict. This is
training impurity importance, not causal or out-of-sample importance.
"""
from pathlib import Path
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import resource
import sys
import time

SCHEMA = 'dmv2-model-only-rf-importance/v1'
YEARS = (2020, 2021, 2022, 2023)
MAX_JSON = 8 * 1024 ** 2
LIMITS = {'model_read_bytes': 512 * 1024 ** 2, 'rss_bytes': 4 * 1024 ** 3,
          'active_seconds': 300, 'model_loads': 4, 'new_fits': 0,
          'prediction_calls': 0, 'source_panel_reads': 0}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return 'sha256:' + hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()


def pin(path):
    p = Path(path).resolve()
    need(p.is_file(), 'Regular pinned file required')
    return dict(path=str(p), size_bytes=p.stat().st_size, sha256=file_hash(p))


def check(item):
    need(isinstance(item, dict) and set(item) == {'path', 'size_bytes', 'sha256'}, 'Closed file pin required')
    p = Path(item['path'])
    need(p.is_absolute() and p.resolve() == p and p.is_file()
         and type(item['size_bytes']) is int and p.stat().st_size == item['size_bytes']
         and file_hash(p) == item['sha256'], 'Pinned bytes changed: ' + str(p))
    return p


def _pairs(pairs):
    value = {}
    for key, item in pairs:
        need(key not in value, 'Duplicate JSON key')
        value[key] = item
    return value


def read_json(path):
    p = Path(path)
    need(p.is_file() and p.stat().st_size <= MAX_JSON, 'JSON metadata exceeds bound')
    def reject(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    return json.loads(p.read_text(encoding='utf-8'), object_pairs_hook=_pairs, parse_constant=reject)


def save(path, value):
    p = Path(path)
    encoded = (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                          allow_nan=False) + '\n').encode('utf-8')
    need(len(encoded) <= MAX_JSON, 'Output JSON exceeds bound')
    with p.open('xb') as f:
        f.write(encoded); f.flush(); os.fsync(f.fileno())
    return pin(p)


def _runtime():
    return {n: importlib.metadata.version(n) for n in ('numpy', 'scikit-learn', 'scipy', 'joblib')}


def _logical(plan):
    return canonical({k: v for k, v in plan.items() if k != 'plan_sha256'})


def validate_manifest(preprocessor, predictors):
    """Strictly preserve the training-only input/retained/output column map."""
    need(set(preprocessor) == {'manifest', 'fingerprint'}, 'Closed preprocessor payload required')
    manifest = preprocessor['manifest']
    need(preprocessor['fingerprint'] == canonical(manifest), 'Preprocessor canonical fingerprint changed')
    need(manifest['input_columns'] == predictors and len(predictors) == len(set(predictors))
         and all(isinstance(n, str) and n and not n.endswith('__missing') for n in predictors),
         'Input predictor order/name changed')
    retained = manifest['retained_columns']; dropped = manifest['dropped_columns']
    need(isinstance(retained, list) and isinstance(dropped, dict)
         and retained == [n for n in predictors if n not in dropped]
         and set(retained).isdisjoint(dropped) and set(retained) | set(dropped) == set(predictors),
         'Retained/dropped partition changed')
    need(manifest['output_columns'] == [c for n in retained for c in (n, n + '__missing')],
         'Interleaved value/missing-indicator mapping changed')
    need(manifest['training_only_schema_selection'] is True
         and manifest['imputation'] == 'training-partition finite-value median; labels never imputed'
         and manifest['preprocessing_policy'] == 'train_partition_median_missing_indicator_v1',
         'Training-only preprocessor policy changed')
    return retained


def _inventory_pin(inventory, path):
    p = Path(path).resolve(); item = inventory.get(str(p))
    need(isinstance(item, dict) and 'pin' in item, 'Artifact absent from independent model inventory')
    need(item['pin'] == pin(p), 'Artifact differs from independently checked bytes')
    return item['pin']


def plan_rf_importance(run_directory, output_plan):
    """Seal a fresh plan before any model deserialization; metadata/hash reads only."""
    run, output = Path(run_directory).resolve(), Path(output_plan).resolve()
    need(not output.exists(), 'Preserve existing plan; use a fresh path')
    need(not output.is_relative_to(run), 'Analysis plan must be outside the preserved source run')
    model_plan_path = run / 'read_plan.json'
    proof_path = run / 'verification/models.json'
    verification_path = run / 'verification/models_read_plan.json'
    model_plan, proof, original = map(read_json, (model_plan_path, proof_path, verification_path))
    need(model_plan.get('schema') == 'dmv2_unstopped_target_comparison_models_v1'
         and model_plan.get('plan_sha256') == _logical(model_plan), 'Original model plan invalid')
    need(model_plan['status'] == 'SEALED_BEFORE_FINANCIAL_READ'
         and model_plan['expected_fits'] == 20 and model_plan['expected_preprocessors'] == 4,
         'Original fixed model scope changed')
    need(proof.get('status') == 'PASS' and proof.get('phase') == 'models'
         and proof.get('verified_estimator_fits') == 20 and proof.get('verified_shared_preprocessors') == 4
         and proof.get('new_estimator_fits') == 0 and proof.get('financial_2024_plus_values_read') is False,
         'Independent model-only PASS for all20 fits/four preprocessors required')
    need(proof['model_plan_pin'] == pin(model_plan_path)
         and proof['verification_plan_pin'] == pin(verification_path)
         and original['model_plan_pin'] == pin(model_plan_path)
         and original.get('phase') == 'models'
         and original.get('status') == 'SEALED_BEFORE_FINANCIAL_READ', 'Independent proof lineage changed')
    predictors = model_plan['predictor_names']
    need(len(predictors) == 715, 'Frozen715 predictor names required')
    need(all(model_plan['runtime'][k] == v for k, v in _runtime().items()), 'Saved model/runtime mismatch')
    helper = model_plan['helper_pins']['preprocessing.py']; check(helper)
    inventory = original['artifacts']; folds = []
    for year in YEARS:
        fold = run / 'folds' / str(year)
        fit_path = fold / 'heads/endpoint/FIT_COMPLETE.json'
        prepared_path = fold / 'prepared/receipt.json'
        preprocessor_path = fold / 'prepared/preprocessor.json'
        fit_pin, prepared_pin, preprocessor_pin = [_inventory_pin(inventory, p) for p in
                                                   (fit_path, prepared_path, preprocessor_path)]
        fit, prepared, pp = map(read_json, (fit_path, prepared_path, preprocessor_path))
        model_pin = _inventory_pin(inventory, fold / 'heads/endpoint/model.joblib')
        need(fit['status'] == 'FIT_COMPLETE' and fit['year'] == year and fit['head'] == 'endpoint'
             and fit['estimator_fits'] == 1 and fit['prepared_pin'] == prepared_pin
             and fit['model'] == model_pin and fit['plan_sha256'] == model_plan['plan_sha256'],
             'Endpoint model receipt lineage changed')
        need(prepared['status'] == 'PREPARED' and prepared['year'] == year
             and prepared['preprocessor_fits'] == 1 and prepared['estimator_fits'] == 0
             and prepared['plan_sha256'] == model_plan['plan_sha256']
             and prepared['artifacts']['preprocessor'] == preprocessor_pin,
             'Prepared receipt lineage changed')
        retained = validate_manifest(pp, predictors)
        fingerprint = pp['fingerprint']
        params = dict(model_plan['folds'][str(year)]['fixed_rf_parameters']); params['n_jobs'] = 1
        need(fit['saved_inference_effective_parameters'] == params
             and params['n_estimators'] == 500 and params['n_jobs'] == 1
             and fit['native_class_ids'] == fit['probability_class_ids'] == list(range(1, 11))
             and fit['preprocessor_fingerprint'] == prepared['preprocessor_fingerprint'] == fingerprint
             and fit['matrix_width'] == prepared['matrix_width'] == 2 * len(retained),
             'Saved model classes/parameters/feature width changed')
        folds.append(dict(year=year, head='endpoint', model_pin=model_pin, fit_pin=fit_pin,
            prepared_pin=prepared_pin, preprocessor_pin=preprocessor_pin,
            retained_names=retained, preprocessor_fingerprint=fingerprint,
            expected_parameters=params, matrix_width=2 * len(retained)))
    model_bytes = sum(f['model_pin']['size_bytes'] for f in folds)
    need(model_bytes <= LIMITS['model_read_bytes'], 'Four-model byte budget exceeded')
    plan = dict(schema=SCHEMA, status='SEALED_BEFORE_MODEL_DESERIALIZATION', run_directory=str(run),
        model_plan_pin=pin(model_plan_path), independent_models_proof_pin=pin(proof_path),
        independent_models_read_plan_pin=pin(verification_path), preprocessing_helper_pin=helper,
        worker_pin=pin(__file__), runtime=_runtime(), predictor_names=predictors, folds=folds,
        years=list(YEARS), head='endpoint', model_read_bytes=model_bytes, limits=dict(LIMITS),
        source_panel_reads=0, new_estimator_fits=0, prediction_calls=0, backtest_replays=0,
        method='mean decrease in impurity (MDI), raw global normalization; value and missing components separated',
        interpretation='Training split usage; correlated features share importance. No sign, causality or test efficacy is implied.',
        failed_backtest_ledger_ignored='Independent20-fit model PASS is the gate; portfolio failures do not invalidate saved model bytes.')
    plan['plan_sha256'] = _logical(plan)
    output.parent.mkdir(parents=True, exist_ok=True); save(output, plan)
    return plan


def importance_rows(predictors, retained, importances, year):
    """Map only the manifest's interleaved vector; dropped predictors are null."""
    values = [float(x) for x in importances]
    need(len(values) == 2 * len(retained) and len(retained) == len(set(retained))
         and retained == [n for n in predictors if n in retained], 'Importance vector/column order mismatch')
    need(all(math.isfinite(v) and v >= 0 for v in values)
         and math.isclose(math.fsum(values), 1., rel_tol=1e-9, abs_tol=1e-12),
         'Importance vector must be finite, nonnegative and globally normalized')
    index = {name: i for i, name in enumerate(retained)}
    rows = []
    for name in predictors:
        i = index.get(name)
        value, missing = (None, None) if i is None else (values[2*i], values[2*i+1])
        rows.append(dict(feature_name=name, year=year, retained=i is not None,
            value_importance=value, missing_importance=missing,
            total_importance=None if i is None else value + missing))
    return rows


def peak_rss_bytes():
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(raw if sys.platform == 'darwin' else raw * 1024)


def extract_rf_importance(plan_file, output_directory):
    """Explicit bounded deserialization of independently checked local endpoint models."""
    started = time.monotonic(); plan_path = Path(plan_file).resolve()
    plan = read_json(plan_path)
    need(plan.get('schema') == SCHEMA and plan.get('status') == 'SEALED_BEFORE_MODEL_DESERIALIZATION'
         and plan.get('plan_sha256') == _logical(plan), 'Analysis plan invalid')
    need(plan['limits'] == LIMITS and plan['years'] == list(YEARS) and plan['head'] == 'endpoint'
         and len(plan['folds']) == 4 and plan['runtime'] == _runtime()
         and plan['model_read_bytes'] <= LIMITS['model_read_bytes']
         and all(plan[k] == 0 for k in ('source_panel_reads', 'new_estimator_fits',
                                       'prediction_calls', 'backtest_replays')), 'Closed analysis scope changed')
    need(plan['worker_pin'] == pin(__file__), 'Extractor code changed after plan')
    for name in ('model_plan_pin', 'independent_models_proof_pin', 'independent_models_read_plan_pin',
                 'preprocessing_helper_pin'):
        check(plan[name])
    # Revalidate the proof/manifest lineage from the pinned metadata.
    # Every trusted model pin must remain from the independently checked local inventory.
    original = read_json(check(plan['independent_models_read_plan_pin']))
    source = read_json(check(plan['model_plan_pin']))
    proof = read_json(check(plan['independent_models_proof_pin']))
    need(proof['status'] == 'PASS' and proof['phase'] == 'models'
         and proof['verified_estimator_fits'] == 20 and proof['verified_shared_preprocessors'] == 4
         and proof['model_plan_pin'] == plan['model_plan_pin']
         and proof['verification_plan_pin'] == plan['independent_models_read_plan_pin']
         and source['predictor_names'] == plan['predictor_names'], 'Model PASS/input map changed')
    out = Path(output_directory).resolve()
    need(not out.exists(), 'Preserve old extraction; fresh output directory required')
    need(not out.is_relative_to(Path(plan['run_directory'])),
         'Analysis output must be outside the preserved source run')
    out.mkdir(parents=True)
    save(out / 'STARTED.json', dict(schema=SCHEMA, status='MODEL_ONLY_EXTRACTION_STARTED',
        analysis_plan_pin=pin(plan_path), limits=dict(LIMITS), new_estimator_fits=0,
        prediction_calls=0, source_panel_reads=0, backtest_replays=0))
    loaded = 0; decoded = 0; rows = []; completed_years = []
    def guard():
        need(time.monotonic() - started <= LIMITS['active_seconds'], 'Active-time ceiling exceeded')
        need(peak_rss_bytes() <= LIMITS['rss_bytes'], 'Peak process RSS ceiling exceeded')
    try:
        import joblib
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier
        guard()
        for expected_year, fold in zip(YEARS, plan['folds']):
            need(fold['year'] == expected_year and fold['head'] == 'endpoint', 'Annual model load order changed')
            for name in ('fit_pin', 'prepared_pin', 'preprocessor_pin', 'model_pin'):
                item = fold[name]; check(item)
                need(original['artifacts'][item['path']]['pin'] == item, 'Artifact outside authenticated inventory')
            run = Path(plan['run_directory']); expected_model = run / 'folds' / str(expected_year) / 'heads/endpoint/model.joblib'
            need(fold['model_pin']['path'] == str(expected_model), 'Untrusted model redirect')
            pp = read_json(check(fold['preprocessor_pin']))
            need(validate_manifest(pp, plan['predictor_names']) == fold['retained_names']
                 and pp['fingerprint'] == fold['preprocessor_fingerprint'], 'Retained feature map changed')
            fit = read_json(check(fold['fit_pin'])); prepared = read_json(check(fold['prepared_pin']))
            need(fit['model'] == fold['model_pin'] and fit['prepared_pin'] == fold['prepared_pin']
                 and fit['saved_inference_effective_parameters'] == fold['expected_parameters']
                 and fit['preprocessor_fingerprint'] == prepared['preprocessor_fingerprint'] == pp['fingerprint']
                 and prepared['artifacts']['preprocessor'] == fold['preprocessor_pin'], 'Receipt lineage changed')
            decoded += fold['model_pin']['size_bytes']
            need(decoded <= LIMITS['model_read_bytes'] and loaded < LIMITS['model_loads'], 'Model decode ceiling exceeded')
            guard()
            # joblib may execute Python: only the independently verified, byte-pinned
            # local path above is admitted. No model supplied by an external caller.
            model = joblib.load(check(fold['model_pin'])); loaded += 1
            guard()
            need(type(model) is RandomForestClassifier and model.get_params(deep=False) == fold['expected_parameters']
                 and model.n_estimators == 500 and len(model.estimators_) == 500
                 and model.n_features_in_ == fold['matrix_width'] == 2 * len(fold['retained_names'])
                 and np.array_equal(model.classes_, np.arange(1, 11)), 'Unexpected saved RF type/schema/parameters')
            vector = model.feature_importances_
            rows.extend(importance_rows(plan['predictor_names'], fold['retained_names'], vector, expected_year))
            del vector, model; gc.collect(); completed_years.append(expected_year); guard()
        need(decoded == plan['model_read_bytes'] and loaded == 4 and len(rows) == 2860,
             'Incomplete model-only extraction')
        raw_pin = save(out / 'feature_importances.json', rows)
        receipt = dict(schema=SCHEMA, status='COMPLETE_MODEL_ONLY_IMPORTANCE',
            analysis_plan_pin=pin(plan_path), worker_pin=pin(__file__), artifact_pin=raw_pin,
            years=completed_years, head='endpoint', feature_rows=len(rows), base_predictors=715,
            method=plan['method'], interpretation=plan['interpretation'],
            model_loads=loaded, model_read_bytes=decoded, peak_process_rss_bytes=peak_rss_bytes(),
            active_seconds=time.monotonic()-started, limits=dict(LIMITS),
            new_estimator_fits=0, prediction_calls=0, source_panel_reads=0, backtest_replays=0)
        guard(); save(out / 'receipt.json', receipt)
        return receipt
    except BaseException as exc:
        save(out / 'failure.json', dict(schema=SCHEMA, status='FAILED_PRESERVED', error_type=type(exc).__name__,
            error=str(exc), analysis_plan_pin=pin(plan_path), completed_years=completed_years,
            model_loads=loaded, model_read_bytes=decoded, active_seconds=time.monotonic()-started,
            peak_process_rss_bytes=peak_rss_bytes(), new_estimator_fits=0, prediction_calls=0,
            source_panel_reads=0, backtest_replays=0))
        raise


def read_rf_importance(directory):
    """Read only bounded importance JSON after its completed receipt authenticates it."""
    out = Path(directory).resolve(); receipt = read_json(out / 'receipt.json')
    need(receipt.get('schema') == SCHEMA and receipt.get('status') == 'COMPLETE_MODEL_ONLY_IMPORTANCE'
         and receipt.get('model_loads') == 4 and receipt.get('new_estimator_fits') == 0
         and receipt.get('source_panel_reads') == 0, 'Complete model-only importance receipt required')
    need(receipt['artifact_pin']['path'] == str(out / 'feature_importances.json'), 'Importance output redirect')
    rows = read_json(check(receipt['artifact_pin']))
    need(len(rows) == receipt['feature_rows'] == 2860, 'Importance rowcount changed')
    return {'receipt': receipt, 'rows': rows}
