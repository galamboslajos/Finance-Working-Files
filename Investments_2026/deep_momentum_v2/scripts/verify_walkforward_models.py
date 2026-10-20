"""Independent saved walk-forward verification. No fit or production worker imports.

Only explicit2010--2023 financial projections are decoded; each fold is released
before the next. Stored models are used solely for inference reconciliation.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import platform
import resource
import time
from importlib.metadata import version

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ARMS = ("rf_paper15","xgb_paper15","rf_extended","xgb_extended")
REFERENCES = ("uniform","train_prior")
YEARS = (2020,2021,2022,2023)
KEYS = ["source_security_id","signal_date"]
PROBS = [f"probability_{i}" for i in range(1,11)]
EPS = 1e-15
CUTOFFS = {2020:"2019-12-31",2021:"2020-12-31",2022:"2021-12-31",2023:"2022-12-30"}
PAPER15 = [f"{prefix}{h}" for prefix in ("zMOM","meanMOM","stdMOM") for h in (1,3,6,9,12)]

def require(condition, message):
    if not condition:
        raise AssertionError(message)

def digest(path):
    state = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            state.update(block)
    return state.hexdigest()

def canonical(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return "sha256:" + hashlib.sha256(raw.encode()).hexdigest()

def json_file(path):
    return json.loads(path.read_text())

def local(base, relative):
    path = (base / relative).resolve()
    require(path.is_relative_to(base.resolve()), f"Artifact leaves approved directory: {relative}")
    return path

def verify_artifacts(base, artifacts):
    count = 0
    if isinstance(artifacts, dict):
        records = [{"path": key, **value} for key, value in artifacts.items()]
    else:
        records = artifacts
    require(isinstance(records, list), "Artifact manifest must be a dict/list")
    for item in records:
        path = local(base, item["path"])
        require(path.is_file(), f"Missing artifact: {path}")
        require(path.stat().st_size == item["size_bytes"], f"Wrong artifact size: {path}")
        require(digest(path) == item["sha256"].removeprefix("sha256:"), f"Changed artifact: {path}")
        count += 1
    return count

def read_table(path, *, first, last):
    """Read safe date/key metadata first, then the bounded financial columns."""
    require('2010-01' <= first <= last <= '2023-12', 'Verifier financial bounds leave2010--2023')
    schema = pq.ParquetFile(path).schema_arrow.names
    require(set(KEYS + ["outcome_month", "outcome_start", "outcome_end"]) <= set(schema), f"Missing dated keys: {path}")
    metadata = pq.read_table(path, columns=KEYS + ["outcome_month", "outcome_start", "outcome_end"]).to_pandas()
    require(not metadata.duplicated(KEYS).any(), f"Duplicate original keys: {path}")
    require(metadata.outcome_month.between(first, last).all(), f"Out-of-pilot financial rows: {path}")
    for sid, signal, month, start, end in metadata.itertuples(index=False, name=None):
        require(isinstance(sid, str) and sid.strip(), "Invalid source ID")
        require(str(pd.Period(signal[:7], freq="M") + 1) == month, "Wrong prediction month")
        require(signal < start <= end and start[:7] == month == end[:7], "Invalid outcome endpoints")
    return pq.read_table(path).to_pandas()

def independent_scores(frame):
    """Direct one-hot scores and equal-month aggregation, independent of workers."""
    values = frame[PROBS].to_numpy(dtype=float)
    require(np.isfinite(values).all(), "Prediction probabilities must be finite")
    require((values >= 0).all(), "Negative probability")
    require(np.all(np.abs(values.sum(axis=1) - 1) <= 1e-8), "Probability rows do not sum to one")
    y = frame.target_class.to_numpy(dtype=float, na_value=np.nan)
    observed = np.isfinite(y)
    require(np.isin(y[observed], np.arange(1, 11)).all(), "Bad canonical class")
    if "target_available" in frame:
        require(np.array_equal(observed, frame.target_available.to_numpy(dtype=bool)), "Target availability changed")
    hard = np.argmax(values, axis=1) + 1
    scored = frame[KEYS + ["outcome_month"]].copy()
    scored["log_loss"] = np.nan
    scored["brier_sum10"] = np.nan
    scored["accuracy"] = np.nan
    scored["observed"] = observed
    chosen = y[observed].astype(int) - 1
    true_probability = values[observed, chosen]
    onehot = np.eye(10)[chosen]
    scored.loc[observed, "log_loss"] = -np.log(np.maximum(true_probability, EPS))
    scored.loc[observed, "brier_sum10"] = np.square(values[observed] - onehot).sum(axis=1)
    scored.loc[observed, "accuracy"] = (hard[observed] == chosen + 1).astype(float)
    monthly = scored.groupby("outcome_month", sort=True).agg(
        prediction_rows=("observed", "size"), labelled_rows=("observed", "sum"),
        log_loss=("log_loss", "mean"), brier_sum10=("brier_sum10", "mean"), accuracy=("accuracy", "mean"))
    summary = {"prediction_rows": len(frame), "labelled_rows": int(observed.sum()),
        "missing_label_rows": int((~observed).sum()), "outcome_months": len(monthly),
        "mean_monthly_log_loss": float(monthly.log_loss.mean()),
        "mean_monthly_brier_sum10": float(monthly.brier_sum10.mean()),
        "mean_monthly_accuracy": float(monthly.accuracy.mean()),
        "pooled_log_loss": float(scored.log_loss.mean()),
        "pooled_brier_sum10": float(scored.brier_sum10.mean()),
        "pooled_accuracy": float(scored.accuracy.mean()),
        "zero_true_class_probability_rows": int(np.sum(true_probability == 0)),
        "floored_true_class_probability_rows": int(np.sum(true_probability < EPS))}
    return summary, monthly, scored

def equal_scalar(actual, expected, name):
    if expected is None or (isinstance(expected, float) and np.isnan(expected)):
        require(actual is None or pd.isna(actual), f"Unexpected value {name}")
    elif isinstance(expected, (int, np.integer)):
        require(actual == expected, f"Different count {name}: {actual} != {expected}")
    else:
        require(np.isclose(actual, expected, rtol=1e-12, atol=2e-14), f"Different score {name}: {actual} != {expected}")

def verify_metrics(directory, frame, state):
    summary, monthly, scored = independent_scores(frame)
    persisted = state["summary"]
    for name, value in summary.items():
        equal_scalar(persisted[name], value, name)
    saved_monthly = pq.read_table(directory / "reports/monthly.parquet").to_pandas().set_index("outcome_month")
    require(list(saved_monthly.index) == list(monthly.index), "Monthly report dates changed")
    for name in ("prediction_rows", "labelled_rows", "log_loss", "brier_sum10", "accuracy"):
        require(np.allclose(saved_monthly[name].astype(float), monthly[name].astype(float), rtol=1e-12, atol=2e-14, equal_nan=True), f"Monthly {name} differs")
    saved_rows = pq.read_table(directory / "reports/row_scores.parquet").to_pandas().set_index(KEYS)
    expected_rows = scored.set_index(KEYS)
    require(set(saved_rows.index) == set(expected_rows.index), "Row-score keys differ")
    saved_rows = saved_rows.reindex(expected_rows.index)
    for expected, actual in (("log_loss", "log_loss"), ("brier_sum10", "brier_sum10"), ("accuracy", "correct")):
        require(np.allclose(saved_rows[actual].astype(float), expected_rows[expected].astype(float), rtol=1e-12, atol=2e-14, equal_nan=True), f"Row {expected} differs")
    probability = frame[PROBS].to_numpy(dtype=float)
    y = frame.target_class.to_numpy(dtype=float, na_value=np.nan)
    labelled = np.isfinite(y)
    hard = np.argmax(probability, axis=1) + 1
    ece_parts = {}
    for kind in ("toplabel", "classwise"):
        table = pq.read_table(directory / f"reports/{kind}_calibration.parquet").to_pandas()
        require(len(table) == (10 if kind == "toplabel" else 100), "Calibration bins changed")
        weighted_gaps = 0.0
        for class_id in ([None] if kind == "toplabel" else range(1, 11)):
            confidence = probability[labelled].max(axis=1) if class_id is None else probability[labelled, class_id - 1]
            hit = hard[labelled] == y[labelled] if class_id is None else y[labelled] == class_id
            bins = np.searchsorted(np.arange(1, 10) / 10, confidence, side="right")
            for bucket in range(10):
                row = table.loc[table.bin_id == bucket]
                if class_id is not None:
                    row = row.loc[row.class_id == class_id]
                require(len(row) == 1, "Calibration cell duplicated or absent")
                row = row.iloc[0]
                selected = bins == bucket
                count = int(selected.sum())
                equal_scalar(row.labelled_count, count, "calibration labelled_count")
                mean = float(confidence[selected].mean()) if count else None
                frequency = float(hit[selected].mean()) if count else None
                gap = abs(mean - frequency) if count else None
                equal_scalar(row.mean_probability, mean, "calibration mean_probability")
                equal_scalar(row.observed_frequency, frequency, "calibration observed_frequency")
                equal_scalar(row.absolute_gap, gap, "calibration absolute_gap")
                weighted_gaps += count * (gap or 0)
        ece_parts[kind] = weighted_gaps / labelled.sum() / (10 if kind == "classwise" else 1)
    equal_scalar(persisted["pooled_top_label_ece"], ece_parts["toplabel"], "top-label ECE")
    equal_scalar(persisted["pooled_classwise_ece_mean"], ece_parts["classwise"], "classwise ECE")
    tail = pq.read_table(directory / "reports/extreme_precision.parquet").to_pandas()
    require(len(tail) == 2 * (1 + frame.outcome_month.nunique()), "Tail-precision calendar changed")
    for period in ["all", *sorted(frame.outcome_month.unique())]:
        period_mask = np.ones(len(frame), dtype=bool) if period == "all" else (frame.outcome_month == period).to_numpy()
        for class_id in (1, 10):
            row = tail.loc[(tail.period == period) & (tail.class_id == class_id)]
            require(len(row) == 1, "Tail-precision cell duplicated or absent")
            row = row.iloc[0]
            selected = period_mask & (hard == class_id)
            observed = selected & labelled
            support, observed_count = int(selected.sum()), int(observed.sum())
            hits = int((observed & (y == class_id)).sum())
            for name, value in (("prediction_rows", int(period_mask.sum())), ("hard_mode_prediction_count", support), ("labelled_hard_mode_count", observed_count), ("missing_label_hard_mode_count", support - observed_count), ("true_positive_count", hits)):
                equal_scalar(row[name], value, f"tail {name}")
            equal_scalar(row.precision, hits / observed_count if observed_count else None, "tail precision")
    return summary

def verify_saved_model(directory, kind, prepared, expected):
    """Reload hash-verified local models and predict; this never invokes fit."""
    from threadpoolctl import threadpool_limits
    if kind == "rf":
        import joblib
        estimator = joblib.load(directory / "model.joblib")
        require(list(estimator.classes_) == list(range(1, 11)), "Reloaded forest classes changed")
        with threadpool_limits(limits=1):
            actual = estimator.predict_proba(prepared)
    else:
        from scipy.special import softmax
        from xgboost import XGBClassifier
        estimator = XGBClassifier(n_jobs=1, device="cpu")
        estimator.load_model(directory / "model.ubj")
        estimator.set_params(n_jobs=1, device="cpu")
        require(list(estimator.classes_) == list(range(10)), "Reloaded XGBoost classes changed")
        with threadpool_limits(limits=1):
            margins = estimator.predict(prepared, output_margin=True)
        actual = softmax(np.asarray(margins, dtype=np.float64), axis=1)
    difference = float(np.max(np.abs(actual - expected)))
    require(np.allclose(actual, expected, rtol=1e-12, atol=2e-14), "Reloaded model probabilities differ from saved predictions")
    return difference


class VerificationBudget:
    def __init__(self, execution):
        self.seconds = execution['verification_maximum_wall_seconds']
        self.memory = execution['verification_maximum_peak_rss_bytes']
        require(type(self.seconds) is int and 0 < self.seconds <= 1800, 'Verifier requires its30-minute ceiling')
        require(type(self.memory) is int and 0 < self.memory <= 8 * 1024**3, 'Verifier requires its8GiB ceiling')
        require(execution['verification_model_fits'] == 0, 'Verifier can never fit a model')
        self.started = time.monotonic()
        self.peak = 0

    def check(self):
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self.peak = max(self.peak, int(peak if platform.system() == 'Darwin' else peak * 1024))
        require(self.peak <= self.memory, 'Verification exceeded its8GiB memory ceiling')
        require(time.monotonic() - self.started <= self.seconds, 'Verification exceeded its30-minute ceiling')


def array_sha(array):
    return 'sha256:' + hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast('B')).hexdigest()


def equal_column(left, right, name):
    left = left.reset_index(drop=True).astype(object).where(left.reset_index(drop=True).notna(), None)
    right = right.reset_index(drop=True).astype(object).where(right.reset_index(drop=True).notna(), None)
    require(left.equals(right), 'Ledger/source column differs: ' + name)


def check_dates(frame, first, last, *, year=None):
    require(not frame.duplicated(KEYS).any(), 'Duplicate original source/signal key')
    require(frame.outcome_month.between(first, last).all(), 'Financial table would exceed the declared outcome window')
    require(set(frame.outcome_month) == set(str(p) for p in pd.period_range(first, last, freq='M')),
        'Original cohort lost a calendar month')
    for signal, month, start, end in frame[['signal_date','outcome_month','outcome_start','outcome_end']].itertuples(index=False, name=None):
        require(str(pd.Period(signal[:7], freq='M') + 1) == month and signal < start <= end
            and start[:7] == month == end[:7], 'Predictor/outcome clocks overlap')
    if year is not None:
        expected = {'fit_year':year, 'training_outcome_start':f'{year-10}-01-01',
            'training_outcome_end':f'{year-1}-12-31', 'training_label_observed_through':CUTOFFS[year]}
        for name, value in expected.items():
            require(name in frame and frame[name].eq(value).all(), 'Incorrect annual lineage: ' + name)


def _pin_path(base, item):
    verify_artifacts(base, [item])
    return local(base, item['path'])


def annual_directory(root, run, identity, root_audit, year):
    """Resolve independently from the root manifest; never use model workers."""
    require(2010 <= year <= 2023, 'Protected annual source cannot be resolved')
    entry = identity['annual_sources'][str(year)]
    if entry.get('parent_reference') == 'THIS_BUILD':
        require(year in YEARS and entry['directory_scope'] == 'run'
            and entry['directory'] == f'years/{year}', 'New annual source path changed')
        directory = local(run, entry['directory'])
        audit_path = _pin_path(run, root_audit['years'][str(year)])
        require(audit_path == directory / 'audit.json', 'New annual source audit redirect')
        expected_identity = canonical(identity)
    else:
        require(year < 2020 and entry['directory_scope'] == 'root', 'Historical source scope changed')
        directory = local(root, entry['directory'])
        audit_path = _pin_path(root, entry['audit_pin'])
        require(audit_path == directory / 'audit.json', 'Historical annual audit redirect')
        parent_path = _pin_path(root, entry['parent_identity_pin'])
        parent_audit_path = _pin_path(root, entry['parent_audit_pin'])
        parent = json_file(parent_audit_path)
        expected_identity = entry['build_identity_sha256']
        require(canonical(json_file(parent_path)) == expected_identity
            and parent['status'] == 'COMPLETE' and parent['build_identity_sha256'] == expected_identity,
            'Historical parent identity changed')
        require(_pin_path(parent_audit_path.parent, parent['years'][str(year)]) == audit_path,
            'Historical parent does not bind its annual audit')
    audit = json_file(audit_path)
    require(audit['status'] == 'COMPLETE' and audit['year'] == year
        and audit['build_identity_sha256'] == expected_identity, 'Wrong annual source identity/year')
    require({'features','feature_coverage','original_cohort','targets','target_state'} <= set(audit['artifacts']),
        'Incomplete annual model inputs')
    return directory, audit


def annual_metadata(directory, audit, year):
    paths = {name:_pin_path(directory,audit['artifacts'][name])
        for name in ('features','feature_coverage','original_cohort','targets','target_state')}
    for name,path in paths.items():
        require(path == directory / (name + ('.json' if name == 'target_state' else '.parquet')),
            'Annual artifact redirects a fixed model input')
    meta = pq.read_table(paths['targets'], columns=KEYS+['outcome_month','outcome_start','outcome_end']).to_pandas()
    check_dates(meta, f'{year}-01', f'{year}-12')
    require(len(meta) <= 60_000, 'Annual original cohort exceeds the bounded row ceiling')
    keys = pd.MultiIndex.from_frame(meta[KEYS])
    for name in ('features','feature_coverage','original_cohort'):
        other = pq.read_table(paths[name], columns=KEYS).to_pandas()
        require(not other.duplicated(KEYS).any() and set(pd.MultiIndex.from_frame(other)) == set(keys),
            'Annual original cohort denominator changed: ' + name)
        require(other.signal_date.between(f'{year-1}-12-01', f'{year}-11-30').all(),
            'Annual predictor keys exceed their approved outcome year')
    state = json_file(paths['target_state'])
    require(state['artifact_sha256'] == audit['target_artifact_sha256']
        and state['provenance']['label_observed_through'] <= f'{year}-12-31', 'Annual target observation cutoff changed')
    return paths, meta


def source_values(root, run, identity, root_audit, ledger, years, columns, *, training, budget):
    """Reconcile original rows and load just selected feature rows, year by year."""
    parts = []
    indexed = ledger.set_index(KEYS)
    for year in years:
        budget.check()
        directory,audit = annual_directory(root,run,identity,root_audit,year)
        paths,metadata = annual_metadata(directory,audit,year)
        selected = indexed.loc[indexed.outcome_month.str.startswith(str(year)+'-')]
        require(set(selected.index) == set(pd.MultiIndex.from_frame(metadata[KEYS])),
            'Prepared ledger loses original historical membership rows')
        target = read_table(paths['targets'],first=f'{year}-01',last=f'{year}-12').set_index(KEYS).reindex(selected.index)
        coverage_names = ['identity_eligible','requested','paper_prediction_eligible','intrinio_security_id']
        coverage = pq.read_table(paths['feature_coverage'],columns=KEYS+coverage_names).to_pandas().set_index(KEYS).reindex(selected.index)
        for name in ('outcome_month','outcome_start','outcome_end','identity_eligible','requested',
                'paper_prediction_eligible','prediction_eligible','target_available','target_class'):
            equal_column(selected[name],target[name],f'{year}/{name}')
        for name in coverage_names:
            equal_column(selected[name],coverage[name],f'{year}/coverage/{name}')
        require(selected.prediction_eligible.equals(selected.paper_prediction_eligible), 'Shared Paper15 admission changed')
        available = selected.target_class.notna()
        require(np.array_equal(available, selected.target_available), 'Missing target was imputed or availability changed')
        if training:
            require(np.array_equal(selected.fit_mask, selected.prediction_eligible & selected.target_available),
                'Training mask excludes rows based on expanded-feature completeness')
            mask = selected.fit_mask
        else:
            mask = selected.prediction_eligible
        features = pq.read_table(paths['features'],columns=KEYS+columns).to_pandas().set_index(KEYS)
        require(list(features.columns) == columns, 'Registered source feature schema/order changed')
        admitted = features.reindex(selected.index[mask]).replace([np.inf,-np.inf],np.nan)
        require(admitted[PAPER15].notna().all().all(), 'Eligible row lacks a finite Paper15 input')
        parts.append(admitted)
        del target,coverage,features,admitted
        gc.collect()
    combined = pd.concat(parts).reindex(indexed.index[indexed.fit_mask if training else indexed.prediction_eligible])
    require(combined.index.is_unique and not combined.empty, 'Feature source assembly lost selected keys')
    return combined


def verify_preprocessing(fold, train, test, train_values, test_values, year, *, expanded_names):
    """Recompute statistics directly; no fitted state or worker is constructed."""
    fit = train.loc[train.fit_mask]
    result = {}
    for bank,names in (('paper15',PAPER15),('expanded',expanded_names)):
        directory = fold/'preprocessors'/bank
        state = json_file(directory/'manifest.json'); manifest = state['manifest']
        require(canonical(manifest) == state['fingerprint'], 'Preprocessor fingerprint changed')
        for key,value in (('input_columns',names),('train_start',f'{year-10}-01-01'),
                ('train_end',f'{year-1}-12-31'),('label_observed_through',CUTOFFS[year]),
                ('fit_rows',len(fit)),('fit_keys_sha256',canonical(fit[KEYS].values.tolist())),
                ('baseline_required_for_population',PAPER15),('training_only_schema_selection',True)):
            require(manifest[key] == value, 'Preprocessing contract differs: '+bank+'/'+key)
        require(all(manifest.get(key) is None for key in ('scaling','clipping','winsorization')),
            'Unexpected learned normalization')
        retained,dropped = [],{}
        for name in names:
            values = train_values[name].dropna().to_numpy(dtype=float)
            if len(values) == 0: dropped[name] = 'all_missing_in_training_partition'
            elif np.all(values == values[0]): dropped[name] = 'constant_finite_values_in_training_partition'
            else:
                retained.append(name)
                equal_scalar(manifest['medians'][name],float(np.median(values)),'training-only median '+name)
        require(manifest['retained_columns'] == retained and manifest['dropped_columns'] == dropped,
            'Feature retention uses information outside its supervised training partition')
        require(set(manifest['medians']) == set(retained), 'Median state includes an unretained feature')
        outputs = [part for name in retained for part in (name,name+'__missing')]
        require(manifest['output_columns'] == outputs, 'Missing flags are not fixed for every retained feature')
        shapes = {}
        for filename,source in (('train_X.npy',train_values),('test_X.npy',test_values)):
            matrix = np.load(directory/filename,mmap_mode='r',allow_pickle=False)
            require(matrix.dtype == np.float32 and matrix.shape == (len(source),len(outputs)),
                'Prepared matrix shape/dtype differs: '+bank+'/'+filename)
            for offset,name in enumerate(retained):
                values = source[name].to_numpy(dtype=float)
                missing = ~np.isfinite(values)
                transformed = np.where(missing,manifest['medians'][name],values).astype(np.float32)
                require(np.array_equal(matrix[:,2*offset],transformed), 'Prepared feature differs: '+bank+'/'+name)
                require(np.array_equal(matrix[:,2*offset+1],missing.astype(np.float32)), 'Prepared missing flag differs: '+bank+'/'+name)
            require(np.isfinite(matrix).all(), 'Prepared matrix contains a nonfinite value')
            shapes[filename] = {'shape':list(matrix.shape),'array_sha256':array_sha(matrix)}
        result[bank] = {'fingerprint':state['fingerprint'],'retained_columns':len(retained),
            'dropped_columns':len(dropped),'input_columns':len(names),'matrices':shapes}
    return result


def verify_mode_metrics(directory, frame):
    """Independent ten-class precision/recall and actual-by-predicted counts."""
    p = frame[PROBS].to_numpy(dtype=float)
    y = frame.target_class.to_numpy(dtype=float,na_value=np.nan)
    hard,observed = np.argmax(p,axis=1)+1,np.isfinite(y)
    mode = pq.read_table(directory/'reports/mode_class_performance.parquet').to_pandas()
    confusion = pq.read_table(directory/'reports/confusion.parquet').to_pandas()
    periods = ['all',*sorted(frame.outcome_month.unique())]
    require(len(mode)==len(periods)*10 and len(confusion)==len(periods)*100,'Mode/confusion calendar differs')
    require(not mode.duplicated(['period','class_id']).any() and not confusion.duplicated(['period','actual_class','predicted_class']).any(),
        'Duplicate mode/confusion cell')
    for period in periods:
        selected = np.ones(len(frame),dtype=bool) if period=='all' else frame.outcome_month.eq(period).to_numpy()
        for k in range(1,11):
            predicted = selected & (hard==k)
            labelled = predicted & observed
            actual = selected & observed & (y==k)
            hits = int((actual & predicted).sum())
            row = mode.loc[(mode.period==period)&(mode.class_id==k)]
            require(len(row)==1,'Missing mode class cell'); row=row.iloc[0]
            for name,value in (('prediction_rows',int(selected.sum())),('predicted_support',int(predicted.sum())),
                    ('labelled_predicted_support',int(labelled.sum())),('missing_label_predicted_support',int((predicted&~observed).sum())),
                    ('actual_support',int(actual.sum())),('true_positive_count',hits),
                    ('precision',hits/labelled.sum() if labelled.any() else None),
                    ('recall',hits/actual.sum() if actual.any() else None)):
                equal_scalar(row[name],value,'mode/'+name)
            for guessed in range(1,11):
                count = confusion.loc[(confusion.period==period)&(confusion.actual_class==k)&(confusion.predicted_class==guessed),'count']
                require(len(count)==1 and count.iloc[0]==int((actual&(hard==guessed)).sum()),'Confusion count differs')


def verify_fit(directory, kind, bank, year, train, test, preprocessor, prepared, recipe, identity, model_identity):
    complete = json_file(directory/'FIT_COMPLETE.json'); attempt = json_file(directory/'FIT_ATTEMPT.json')
    require(complete['status']=='FIT_COMPLETE' and complete['fit_count']==1 and attempt['attempt']==1,
        'An arm does not have exactly one completed fit')
    for marker in (complete,attempt):
        require(marker['build_identity_sha256']==identity and marker['model_identity_sha256']==model_identity,
            'Fit checkpoint changed identity')
        expected = marker['expected_metadata']
        require(expected['test_year']==year and expected['preprocessor_fingerprint']==preprocessor['fingerprint']
            and expected['fit_keys_sha256']==canonical(train.loc[train.fit_mask,KEYS].values.tolist())
            and expected['train_start']==f'{year-10}-01-01' and expected['train_end']==f'{year-1}-12-31'
            and expected['label_observed_through']==CUTOFFS[year], 'Checkpoint training lineage differs')
    require(attempt['expected_metadata']==complete['expected_metadata'] and attempt['estimator_kind']==kind
        and attempt['maximum_total_fits']==16, 'Fit attempt differs from its completion')
    _pin_path(directory,complete['model'])
    require(complete['model']['path']==('model.joblib' if kind=='rf' else 'model.ubj'),'Unexpected saved model path')
    metadata = complete['metadata']; labels = np.load(directory.parents[1]/'data/train_y.npy',allow_pickle=False)
    native = list(range(1,11)) if kind=='rf' else list(range(10))
    encoded = labels if kind=='rf' else labels-1
    for name,value in (('native_class_ids',native),('probability_class_ids',list(range(1,11))),
            ('train_rows',int(train.fit_mask.sum())),('matrix_width',preprocessor['matrices']['train_X.npy']['shape'][1]),
            ('explicit_parameters',recipe['estimators'][kind]),
            ('prepared_train_matrix_sha256',preprocessor['matrices']['train_X.npy']['array_sha256']),
            ('encoded_training_labels_sha256',array_sha(encoded))):
        require(metadata[name]==value,'Fitted metadata differs: '+name)
    require(metadata['native_to_canonical_mapping']==[{'native_class_id':n,'canonical_class_id':n if kind=='rf' else n+1} for n in native],
        'Native/canonical class map differs')
    for name in ('test_labels_supplied_to_estimator','early_stopping_used','parameter_search_performed','probability_calibration_performed'):
        require(metadata[name] is False,'Unapproved model operation '+name)
    expected = complete['expected_metadata']
    for key,filename in (('train_matrix',f'preprocessors/{bank}/train_X.npy'),('training_labels','data/train_y.npy')):
        require(_pin_path(directory.parents[1],expected[key])==directory.parents[1]/filename,'Fit checkpoint prepared-input redirect')
    require(prepared['schemas'][bank]['fingerprint']==preprocessor['fingerprint'],'Prepared schema state differs')
    return complete


def verify_fold(root,run,identity,root_audit,models,model_identity,year,budget):
    fold = models/'folds'/str(year); audit=json_file(fold/'audit.json')
    require(audit['status']=='COMPLETE' and audit['test_year']==year and audit['phase']=='walkforward_models_v1'
        and audit['build_identity_sha256']==canonical(identity) and audit['model_identity_sha256']==model_identity,
        'Wrong completed annual model phase')
    verify_artifacts(fold,audit['artifacts']); verify_artifacts(fold,audit['arms'])
    plan=json_file(fold/'read_plan.json'); prepared=json_file(fold/'DATA_COMPLETE.json')
    require(plan['test_year']==year and set(plan['years'])=={str(y) for y in range(year-10,year+1)}
        and plan['train_start']==f'{year-10}-01-01' and plan['train_end']==f'{year-1}-12-31'
        and plan['label_observed_through']==CUTOFFS[year] and plan['financial_2024_plus_decoded'] is False,
        'Fold plan does not contain ten completed outcome years plus one prediction year')
    require(canonical(plan)==prepared['plan_fingerprint'],'Prepared inputs changed their source plan')
    verify_artifacts(fold,prepared['artifacts'])
    for item in plan['years'].values(): verify_artifacts(root,item['source_pins'])
    train=read_table(fold/'data/train_ledger.parquet',first=f'{year-10}-01',last=f'{year-1}-12')
    test=read_table(fold/'data/test_ledger.parquet',first=f'{year}-01',last=f'{year}-12')
    check_dates(train,f'{year-10}-01',f'{year-1}-12',year=year); check_dates(test,f'{year}-01',f'{year}-12',year=year)
    require(train.outcome_end.max()==CUTOFFS[year] and train.outcome_end.max()<test.outcome_start.min()
        and not set(map(tuple,train[KEYS].values))&set(map(tuple,test[KEYS].values)), 'Fold clocks/keys overlap')
    require(np.array_equal(train.fit_mask,train.prediction_eligible & train.target_available),'Shared training fit mask differs')
    y=np.load(fold/'data/train_y.npy',allow_pickle=False)
    require(y.dtype==np.int64 and np.array_equal(y,train.loc[train.fit_mask,'target_class'].to_numpy(dtype=np.int64)),
        'Prepared training labels differ or contain imputation')
    counts={str(k):int((y==k).sum()) for k in range(1,11)}
    require(all(counts.values()) and prepared['train_class_counts']==counts,'Training class counts differ')
    for key,value in (('train_original_rows',len(train)),('train_fit_rows',len(y)),('test_original_rows',len(test)),
            ('test_prediction_rows',int(test.prediction_eligible.sum())),('test_labelled_rows',int(test.target_available.sum())),
            ('fit_keys_sha256',canonical(train.loc[train.fit_mask,KEYS].values.tolist())),
            ('prediction_keys_sha256',canonical(test.loc[test.prediction_eligible,KEYS].values.tolist()))):
        require(prepared[key]==value,'Prepared checkpoint differs: '+key)
    expanded=json_file(fold/'preprocessors/expanded/manifest.json')['manifest']['input_columns']
    dictionary=json_file(run/'feature_dictionary.json')
    names=[spec['name'] for spec in dictionary if spec.get('model_eligible',True)]
    require(expanded==names and len(names)==715 and canonical(dictionary)==plan['feature_dictionary_sha256'],
        'Registered feature dictionary/order differs')
    train_values=source_values(root,run,identity,root_audit,train,range(year-10,year),names,training=True,budget=budget)
    test_values=source_values(root,run,identity,root_audit,test,[year],names,training=False,budget=budget)
    preprocessing=verify_preprocessing(fold,train,test,train_values,test_values,year,expanded_names=names)
    del train_values,test_values; gc.collect(); pa.default_memory_pool().release_unused(); budget.check()
    prior=np.array([counts[str(k)]/len(y) for k in range(1,11)])
    require(np.allclose(prepared['training_prior_vector'],prior,rtol=0,atol=1e-15),'Training prior differs')
    arm_results={}; frames={}
    for name in (*ARMS,*REFERENCES):
        budget.check(); directory=fold/('arms' if name in ARMS else 'refs')/name
        state=json_file(directory/'audit.json'); verify_artifacts(directory,state['artifacts'])
        require(state['status']=='COMPLETE' and state['build_identity_sha256']==canonical(identity)
            and state['model_identity_sha256']==model_identity,'Arm/reference identity changed')
        predictions=read_table(directory/'predictions.parquet',first=f'{year}-01',last=f'{year}-12')
        for column in test:
            require(column in predictions,'Prediction lost original ledger metadata '+column)
            equal_column(predictions[column],test[column],name+'/'+column)
        mask=predictions.prediction_eligible.to_numpy(dtype=bool)
        require(predictions.loc[~mask,PROBS].isna().all().all(),'Ineligible rows received forecasts')
        eligible=predictions.loc[mask].reset_index(drop=True)
        verify_metrics(directory,eligible,state); verify_mode_metrics(directory,eligible)
        max_difference=0.
        if name in ARMS:
            kind='rf' if name.startswith('rf_') else 'xgb'; bank='paper15' if name.endswith('paper15') else 'expanded'
            complete=verify_fit(directory,kind,bank,year,train,test,preprocessing[bank],prepared,
                identity['configuration']['model_recipe'],canonical(identity),model_identity)
            for key,value in complete['metadata'].items(): require(state['metadata'][key]==value,'Arm audit differs from fitted checkpoint '+key)
            require(state['metadata']['prediction_keys_sha256']==prepared['prediction_keys_sha256']
                and state['metadata']['prepared_prediction_matrix_sha256']==preprocessing[bank]['matrices']['test_X.npy']['array_sha256']
                and state['metadata']['prediction_rows']==int(mask.sum()) and state['metadata']['test_labels_used_for_fitting'] is False,
                'Inference metadata differs from shared prepared rows')
            matrix=np.load(fold/'preprocessors'/bank/'test_X.npy',mmap_mode='r',allow_pickle=False)
            max_difference=verify_saved_model(directory,kind,matrix,eligible[PROBS].to_numpy(dtype=float))
            del matrix
        else:
            vector=np.full(10,.1) if name=='uniform' else prior
            require(state['metadata']['fit_count']==0 and state['metadata']['test_year']==year
                and np.allclose(eligible[PROBS].to_numpy(dtype=float),vector[None,:],rtol=0,atol=1e-15),
                'Reference uses information outside supervised training counts')
        arm_results[name]={'prediction_rows':int(mask.sum()),'labelled_rows':int(eligible.target_class.notna().sum()),
            'model_reload_max_probability_difference':max_difference,'metrics_reconciled':True}
        frames[name]=predictions
    return {'test_year':year,'train_original_rows':len(train),'train_fit_rows':len(y),
        'prediction_original_rows':len(test),'preprocessors':preprocessing,'arms':arm_results},frames


def verify_walkforward_models(root,run,*,metadata_only=False):
    root,run=Path(root).resolve(),Path(run).resolve()
    require(run.is_relative_to(root/'outputs/walkforward'),'Run leaves the approved private output namespace')
    identity=json_file(run/'build_identity.json'); root_audit=json_file(run/'audit.json')
    pin=canonical(identity); cfg=identity['configuration']; models=run/'models'
    require(root_audit['status']=='COMPLETE' and root_audit['build_identity_sha256']==pin,
        'Feature/target build is incomplete or changed')
    require(json_file(root/'configs/walkforward_2020_2023.json')==cfg,'Current configuration differs from frozen root')
    require(cfg['protocol_id']=='dmv2-walkforward-2020-2023-v1' and cfg['test_years']==list(YEARS)
        and cfg['execution']['maximum_new_model_fits']==16 and cfg['execution']['maximum_new_preprocessor_fits']==8
        and cfg['execution']['financial_2024_plus_analysis'] is False,'Frozen16-fit outer scope changed')
    budget=VerificationBudget(cfg['execution']); budget.check()
    phase=json_file(models/'build_identity.json'); model_pin=canonical(phase); audit=json_file(models/'audit.json')
    require(audit['status']=='COMPLETE' and audit['phase']=='walkforward_models_v1'
        and audit['build_identity_sha256']==pin and audit['model_identity_sha256']==model_pin
        and phase['parent_identity_sha256']==pin and phase['configuration_sha256']==canonical(cfg),'Model phase changed')
    require(audit['summary']['fitted_arms']==16 and audit['summary']['preprocessor_fits']==8
        and audit['summary']['months']==48 and audit['summary']['outer_years']==list(YEARS),'Incomplete16-fit/48-month model phase')
    require(set(audit['arms'])=={*ARMS,*REFERENCES} and set(audit['folds'])=={str(y) for y in YEARS},'Incomplete arm/fold grid')
    hashes_before={str(p.relative_to(run)):digest(p) for p in (run/'audit.json',run/'build_identity.json',models/'audit.json',models/'build_identity.json')}
    artifact_count=verify_artifacts(run,root_audit['years'])
    for mapping in (audit['artifacts'],audit['arms'],audit['folds']): artifact_count+=verify_artifacts(models,mapping)
    verify_artifacts(root,phase['annual_source_audit_pins'])
    for worker,sha in phase['workers'].items(): require(digest(root/'src/dmv2'/worker)==sha.removeprefix('sha256:'),'Frozen model worker changed '+worker)
    for package,expected in cfg['model_recipe']['packages'].items(): require(version(package)==expected,'Pinned inference package differs '+package)
    require(digest(root/'configs/classifier_pilot.json')==cfg['model_recipe_source']['sha256'].removeprefix('sha256:'),
        'Original estimator recipe changed')
    source=json_file(run/'source_preservation_manifest.json')
    preserved=verify_artifacts(root,source.get('artifacts',source))
    markers=list(models.glob('folds/*/arms/*/FIT_COMPLETE.json'))
    require(len(markers)==16 and len(list(models.glob('folds/*/arms/*/FIT_ATTEMPT.json')))==16,
        'The original fit-attempt/completion count is not16')
    for year in range(2010,2024):
        directory,annual=annual_directory(root,run,identity,root_audit,year)
        annual_metadata(directory,annual,year); budget.check()
    result={'status':'METADATA_PASSED' if metadata_only else 'PASSED','run_directory':str(run),
        'build_identity_sha256':pin,'model_identity_sha256':model_pin,'root_audit_sha256':hashes_before['audit.json'],
        'models_audit_sha256':hashes_before['models/audit.json'],'verifier_sha256':digest(Path(__file__).resolve()),
        'independent_verifier':'direct_training_statistics_saved_model_inference_no_production_workers_no_fits',
        'artifact_files_verified':artifact_count,'preserved_source_files':preserved,'original_fit_markers':len(markers),
        'models_fitted_by_verifier':0,'preprocessors_fitted_by_verifier':0,'financial_2024_plus_decoded':False,
        'financial_values_decoded':not metadata_only,'resource_ceiling_bytes':budget.memory,'wall_ceiling_seconds':budget.seconds}
    if not metadata_only:
        from threadpoolctl import threadpool_limits
        fold_results={}; stitched={name:[] for name in (*ARMS,*REFERENCES)}
        with threadpool_limits(limits=1):
            for year in YEARS:
                detail,frames=verify_fold(root,run,identity,root_audit,models,model_pin,year,budget)
                fold_results[str(year)]=detail
                for name,frame in frames.items(): stitched[name].append(frame)
                del frames,detail; gc.collect(); pa.default_memory_pool().release_unused(); budget.check()
            for name,frames in stitched.items():
                directory=models/('arms' if name in ARMS else 'refs')/name
                state=json_file(directory/'audit.json'); verify_artifacts(directory,state['artifacts'])
                actual=read_table(directory/'predictions.parquet',first='2020-01',last='2023-12')
                expected=pd.concat(frames,ignore_index=True)
                pd.testing.assert_frame_equal(actual,expected,check_dtype=False,check_exact=True)
                check_dates(actual,'2020-01','2023-12')
                selected=actual.loc[actual.prediction_eligible].reset_index(drop=True)
                verify_metrics(directory,selected,state); verify_mode_metrics(directory,selected)
                budget.check()
        result.update(folds=fold_results,annual_fits_reconciled=16,preprocessor_states_reconciled=8,
            stitched_prediction_months=48,aggregate_arms_and_references_reconciled=6,maximum_financial_outcome_decoded='2023-12-31')
    for path,sha in hashes_before.items(): require(digest(run/path)==sha,'Audit/identity changed during verification '+path)
    require(len(list(models.glob('folds/*/arms/*/FIT_ATTEMPT.json')))==16,'Unexpected fit marker created during verification')
    budget.check(); result.update(execution_seconds=time.monotonic()-budget.started,peak_rss_bytes=budget.peak)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True); parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--metadata-only',action='store_true',help='Hash inputs and inspect date/key metadata only; no financial values or model deserialization')
    args=parser.parse_args(argv)
    output=args.output.resolve(); run=args.run.resolve(); root=args.root.resolve()
    require(not output.exists(),'Preserve an existing verification receipt')
    require(not output.is_relative_to(run) or output.is_relative_to(run/'verification'),
        'A new in-run verification receipt must stay in the separate verification directory')
    # An alarm interrupts even a long native inference/hash; memory is checked
    # between bounded annual reads and model calls. No estimate changes budget.
    import signal
    def timeout(signum,frame): raise AssertionError('Verification exceeded its30-minute ceiling')
    signal.signal(signal.SIGALRM,timeout); signal.alarm(1800)
    try:
        result=verify_walkforward_models(root,run,metadata_only=args.metadata_only)
        require(not output.exists(),'Preserve an existing verification receipt')
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n')
    finally:
        signal.alarm(0)
    print(json.dumps({key:result[key] for key in ('status','artifact_files_verified','preserved_source_files','models_fitted_by_verifier','financial_values_decoded')},sort_keys=True))


if __name__=='__main__':
    main()

