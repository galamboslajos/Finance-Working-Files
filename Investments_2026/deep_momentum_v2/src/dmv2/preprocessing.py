"""Training-partition-only missingness for the forest research interface.

No estimator is fitted here. The caller chooses an explicit supervised-fit mask
on an existing TrainingSlice. No target availability is used at prediction time.
"""
from dataclasses import dataclass
import json
from numbers import Real

import numpy as np
import pandas as pd

from .catalog import canonical_sha256
from .paper_features import PAPER_FEATURE_NAMES
from .universe import DataContractError, iso_date

PREPROCESSING_POLICY = 'train_partition_median_missing_indicator_v1'
KEYS = ('source_security_id', 'signal_date')


@dataclass(frozen=True)
class PreparedPrediction:
    X: pd.DataFrame
    ledger: pd.DataFrame
    provenance: dict


def _frame(X):
    if not isinstance(X, pd.DataFrame) or X.columns.duplicated().any():
        raise DataContractError('Predictors must be a DataFrame with distinct named columns')
    if not isinstance(X.index, pd.MultiIndex) or tuple(X.index.names) != KEYS or not X.index.is_unique:
        raise DataContractError('Predictors need unique source_security_id/signal_date index keys')
    if not set(PAPER_FEATURE_NAMES) <= set(X):
        raise DataContractError('All fifteen paper inputs are required to define the prediction population')
    from .features import registered_feature_specs
    allowed = {s['name'] for s in registered_feature_specs() if s.get('model_eligible', True)}
    if not set(X) <= allowed:
        raise DataContractError('Predictors must use the registered model-eligible feature allowlist')
    for sid, signal in X.index:
        if not isinstance(sid, str) or not sid.strip():
            raise DataContractError('Invalid predictor source_security_id')
        if iso_date(signal, 'signal_date') >= '2024-01-01':
            raise DataContractError('Protected 2024+ predictors cannot enter preprocessing')
    if any(X[name].map(lambda v: isinstance(v, str)).any() for name in X):
        raise DataContractError('Numerical text is not a registered numerical predictor value')
    try:
        numeric = X.astype(float)
    except (TypeError, ValueError) as error:
        raise DataContractError('Predictors must contain numerical values or missing data') from error
    if any(X[name].map(lambda v: isinstance(v, (bool, np.bool_))).any() for name in X):
        raise DataContractError('Boolean audit flags are not numerical predictors')
    return numeric.replace([np.inf, -np.inf], np.nan)


def _mask(mask, index, label):
    if not isinstance(mask, pd.Series) or not mask.index.equals(index):
        raise DataContractError(f'{label} must be an explicitly aligned Series')
    if mask.isna().any() or not mask.map(lambda v: isinstance(v, (bool, np.bool_))).all():
        raise DataContractError(f'{label} must contain nonmissing booleans')
    return mask.astype(bool)


@dataclass(frozen=True)
class FittedPreprocessor:
    """Immutable serialized fitted state; transform never learns from its rows."""
    _manifest_json: str
    fingerprint: str

    @property
    def manifest(self):
        # Return a detached copy so callers cannot mutate fitted state.
        return json.loads(self._manifest_json)

    @property
    def feature_names(self):
        return tuple(self.manifest['output_columns'])

    def transform(self, X, *, prediction_mask):
        manifest = self.manifest
        if canonical_sha256(manifest) != self.fingerprint:
            raise DataContractError('Fitted preprocessing manifest fingerprint changed')
        numeric = _frame(X)
        if list(numeric) != manifest['input_columns']:
            raise DataContractError('Prediction input columns/order differ from the frozen training schema')
        requested = _mask(prediction_mask, numeric.index, 'prediction_mask')
        baseline = numeric[list(PAPER_FEATURE_NAMES)].notna().all(axis=1)
        admitted = requested & baseline
        prepared = pd.DataFrame(np.nan, index=numeric.index, columns=manifest['output_columns'])
        if admitted.any():
            for name in manifest['retained_columns']:
                values = numeric.loc[admitted, name]
                prepared.loc[admitted, name] = values.fillna(manifest['medians'][name])
                prepared.loc[admitted, name + '__missing'] = values.isna().astype(float)
        ledger = pd.DataFrame({'prediction_requested': requested,
                               'paper_prediction_eligible': baseline,
                               'transformed': admitted}, index=numeric.index)
        ledger['preprocessing_status'] = np.where(~requested, 'not_requested',
            np.where(~baseline, 'incomplete_paper_baseline', 'transformed'))
        return PreparedPrediction(prepared, ledger, {
            'preprocessing_policy': PREPROCESSING_POLICY,
            'fitted_manifest_sha256': self.fingerprint,
            'rows_retained': len(numeric), 'transformed_rows': int(admitted.sum()),
            'targets_consumed': False, 'fit_performed': False,
            'scaling_clipping_winsorization': False, 'model_ready': False})


def fit_preprocessor(training_slice, *, fit_mask):
    """Fit medians/schema only on caller-selected valid supervised train rows.

    Repeat separately inside every inner-training split and outer-training
    window. Full-period coverage reports must never choose this state.
    Invalid explicitly selected rows raise instead of being silently dropped.
    """
    from .features import TrainingSlice
    if not isinstance(training_slice, TrainingSlice):
        raise DataContractError('fit_preprocessor requires an existing TrainingSlice')
    X, y, ledger, provenance = (training_slice.X, training_slice.y,
                              training_slice.ledger, training_slice.provenance)
    numeric = _frame(X)
    selected = _mask(fit_mask, numeric.index, 'fit_mask')
    if not selected.any():
        raise DataContractError('At least one supervised training row must be explicitly selected')
    if not isinstance(y, pd.Series) or not y.index.equals(numeric.index):
        raise DataContractError('Training labels must be aligned with every original predictor row')
    if not isinstance(ledger, pd.DataFrame) or not ledger.index.equals(numeric.index):
        raise DataContractError('Training outcome ledger must retain the original aligned rows')
    required = {'target_available', 'outcome_start', 'outcome_end'}
    if not required <= set(ledger):
        raise DataContractError('Training ledger lacks target availability or completed-outcome timing')
    try:
        first, last, observed = [iso_date(provenance[name], name) for name in
                                ('train_start', 'train_end', 'label_observed_through')]
    except KeyError as error:
        raise DataContractError('Training provenance lacks explicit train/label cutoffs') from error
    if first > last or observed > last or last >= '2024-01-01':
        raise DataContractError('Training requires ordered completed pre-holdout outcome cutoffs')
    baseline = numeric[list(PAPER_FEATURE_NAMES)].notna().all(axis=1)
    if y.map(lambda v: isinstance(v, str)).any():
        raise DataContractError('Numerical text cannot supply a training label')
    labels = pd.to_numeric(y, errors='coerce')
    finite_y = np.isfinite(labels) & ~y.map(lambda v: isinstance(v, (bool, np.bool_)))
    available = ledger.target_available.map(lambda v: isinstance(v, (bool, np.bool_)) and bool(v))
    valid = baseline & finite_y & available
    for key in numeric.index[selected]:
        start = iso_date(ledger.loc[key, 'outcome_start'], 'outcome_start')
        end = iso_date(ledger.loc[key, 'outcome_end'], 'outcome_end')
        signal = key[1]
        following = str(pd.Period(signal[:7], 'M') + 1)
        if not (signal < start <= end and first <= start and end <= min(last, observed)
                and start[:7] == end[:7] == following and end < '2024-01-01'):
            raise DataContractError('Selected fit row has an incomplete, future or out-of-window outcome')
    if not valid[selected].all():
        raise DataContractError('Selected fit rows require all15 finite baseline inputs, finite labels and available targets')
    fitted = numeric.loc[selected]
    retained, dropped, medians = [], {}, {}
    for name in fitted:
        finite = fitted[name].dropna()
        if finite.empty:
            dropped[name] = 'all_missing_in_training_partition'
        elif finite.nunique(dropna=True) < 2:
            dropped[name] = 'constant_finite_values_in_training_partition'
        else:
            retained.append(name)
            ordered = finite.sort_values().to_numpy(dtype=float)
            middle = len(ordered) // 2
            medians[name] = float(ordered[middle]) if len(ordered) % 2 else float(ordered[middle-1]/2 + ordered[middle]/2)
    if not retained:
        raise DataContractError('No nonconstant observed predictors remain in the selected training partition')
    columns = [name for feature in retained for name in (feature, feature + '__missing')]
    manifest = {'schema_version': 1, 'preprocessing_policy': PREPROCESSING_POLICY,
        'input_columns': list(numeric), 'retained_columns': retained,
        'output_columns': columns, 'medians': medians, 'dropped_columns': dropped,
        'baseline_required_for_population': list(PAPER_FEATURE_NAMES),
        'fit_rows': int(selected.sum()),
        'fit_keys_sha256': canonical_sha256([list(key) for key in numeric.index[selected]]),
        'train_start': first, 'train_end': last, 'label_observed_through': observed,
        'missing_indicators': 'one for every retained predictor, fixed even when training is complete',
        'nonfinite_inputs': 'missing; paper baseline rows withheld rather than imputed',
        'imputation': 'training-partition finite-value median; labels never imputed',
        'scaling': None, 'clipping': None, 'winsorization': None,
        'training_only_schema_selection': True, 'estimator_fitted': False,
        'knowledge_time_verified': False, 'accounting_certified': False, 'model_ready': False}
    return FittedPreprocessor(json.dumps(manifest, sort_keys=True, separators=(',', ':'), allow_nan=False),
                              canonical_sha256(manifest))
