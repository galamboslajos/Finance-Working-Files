"""Signal-only feature assembly and explicit feature/target interfaces.

The feature worker never accepts the outcome ledger as a feature source.
Provider history remains historical reconstruction, not knowledge-time proof.
"""
from dataclasses import dataclass
from collections import Counter

import numpy as np
import pandas as pd
import pyarrow as pa

from .catalog import canonical_sha256
from .data import session_dates
from .universe import DataContractError, KnowledgeTimeUnavailable, iso_date
from .identity_policy import (STRICT_IDENTITY_POLICY, identity_policy_provenance,
                              validate_identity_policy)


KEYS = ("source_security_id", "signal_date")


@dataclass(frozen=True)
class FeaturePanel:
    table: pa.Table
    coverage: pa.Table
    specs: tuple
    diagnostics: dict
    summary: dict
    provenance: dict


def registered_feature_specs():
    from .daily_features import daily_feature_specs
    from .paper_features import paper_feature_specs
    from .context_features import context_feature_specs
    specs = (*daily_feature_specs(), *paper_feature_specs(), *context_feature_specs())
    names = [s['name'] for s in specs]
    if len(names) != len(set(names)) or len(names) != 728:
        raise DataContractError("The registered bank must have 728 distinct definitions")
    return tuple(specs)


def build_feature_panel(daily_prices, calendar, signal_cohorts, *, requested_security_ids,
                        daily_cohorts=None, mode='historical_reconstruction',
                        identity_policy=STRICT_IDENTITY_POLICY,normalization_policy='complete_cohort',
                        context_policy='complete_cohort'):
    """Retain every original signal member; compute only registered past inputs."""
    if mode == 'knowledge_time':
        raise KnowledgeTimeUnavailable('Current membership/identity/adjustment evidence cannot certify knowledge time')
    if mode != 'historical_reconstruction':
        raise DataContractError('Unknown feature reconstruction mode')
    validate_identity_policy(identity_policy)
    from .daily_features import compute_daily_features
    from .paper_features import compute_paper_features
    from .context_features import compute_context_features

    snapshots = tuple(signal_cohorts)
    if not snapshots:
        raise DataContractError('Explicit historical signal cohorts are required')
    signals = [iso_date(s.as_of, 'signal date') for s in snapshots]
    if len(signals) != len(set(signals)) or signals != sorted(signals):
        raise DataContractError('Signal cohorts must be unique and chronological')
    if max(signals) >= '2024-01-01':
        raise DataContractError('Protected 2024+ signals cannot enter features')
    ids = tuple(requested_security_ids)
    if not ids or len(ids) != len(set(ids)) or any(not isinstance(v, str) or not v.strip() for v in ids):
        raise DataContractError('Requested IDs must be explicit, unique and nonempty')
    sessions = session_dates(calendar)
    daily = compute_daily_features(daily_prices, sessions, signal_dates=signals, requested_security_ids=ids)
    paper = compute_paper_features(daily_prices, sessions, signal_cohorts=snapshots,
                                   requested_security_ids=ids, normalization_policy=normalization_policy,
                                   identity_policy=identity_policy)
    context = compute_context_features(daily, daily_prices, sessions, signal_cohorts=snapshots,
                                       requested_security_ids=ids, daily_cohorts=daily_cohorts,
                                       context_policy=context_policy, identity_policy=identity_policy)
    specs = (*daily.specs,*paper.specs,*context.specs)
    names = [s['name'] for s in specs]
    ledger = []
    # The paper worker defensively qualifies checked metadata routes under the
    # explicit policy. Review status is retained even when provisionally eligible.
    identity_fields = ['identity_status', 'source_identity_status', 'identity_eligible',
                       'identity_eligibility_policy', 'identity_eligibility_reason']
    identity_fields += [name for name in ('identity_review_required', 'resolution_status')
                       if name in paper.table]
    identity_fields += [name for name in ('paper_common_population','paper_prediction_eligible') if name in paper.table]
    for row in paper.table[['source_security_id', 'signal_date', 'intrinio_security_id',
                            *identity_fields]].to_dict('records'):
            ledger.append({**row,
                           'requested': row['intrinio_security_id'] in ids,
                           'knowledge_time_verified': False, 'accounting_certified': False})
    cohort = pd.DataFrame(ledger)
    original_keys = [(r['source_security_id'], snapshot.as_of) for snapshot in snapshots
                     for r in snapshot.table.select(['source_security_id']).to_pylist()]
    if len(original_keys) != len(set(original_keys)) or set(original_keys) != set(map(tuple, cohort[list(KEYS)].values)):
        raise DataContractError('Feature worker changed the original historical signal-cohort denominator')
    if cohort.duplicated(list(KEYS)).any():
        raise DataContractError('Duplicate source-ID/signal cohort key')
    frame = cohort[list(KEYS)].copy()
    attached = cohort[['source_security_id', 'signal_date', 'intrinio_security_id', 'identity_eligible']].copy()
    attached.loc[~attached['identity_eligible'], 'intrinio_security_id'] = None
    daily_names = [s['name'] for s in daily.specs]
    d = attached.merge(daily.table[['intrinio_security_id', 'signal_date', *daily_names]],
                       on=['intrinio_security_id', 'signal_date'], how='left', validate='many_to_one')
    frame = frame.merge(d[[*KEYS, *daily_names]], on=list(KEYS), how='left', validate='one_to_one')
    for result in (paper, context):
        cols = [s['name'] for s in result.specs]
        part = result.table[[*KEYS, *cols]]
        if part.duplicated(list(KEYS)).any() or set(map(tuple, part[list(KEYS)].values)) != set(map(tuple, frame[list(KEYS)].values)):
            raise DataContractError('Worker changed original signal-cohort denominator')
        frame = frame.merge(part, on=list(KEYS), how='left', validate='one_to_one')
    frame = frame[[*KEYS, *names]].copy()
    frame[names] = frame[names].astype(float).replace([np.inf, -np.inf], np.nan)
    finite_counts = frame[names].notna().sum(axis=1)
    ledger_frame = cohort.copy()
    ledger_frame['available_feature_count'] = finite_counts.to_numpy()
    ledger_frame['registered_feature_count'] = len(names)
    ledger_frame['feature_status'] = np.where(~cohort.identity_eligible, 'identity_unresolved',
        np.where(~cohort.requested, 'not_requested', np.where(finite_counts > 0, 'partial_features', 'no_usable_features')))
    ledger_frame['model_ready'] = False
    summary = {'rows': len(frame), 'signals': len(signals), 'registered_features': len(names),
               'roles': dict(Counter(s['role'] for s in specs)),
               'candidate_model_columns': sum(s.get('model_eligible', True) for s in specs),
               'feature_status_counts': dict(Counter(ledger_frame['feature_status'])),
               'finite_feature_values': int(finite_counts.sum()), 'model_ready': False,
               'knowledge_time_verified': False, 'accounting_certified': False,
               'identity_eligibility_policy': identity_policy,
               'identity_eligible_rows': int(cohort.identity_eligible.sum()),
               'provisional_identity_eligible_rows': int((cohort.identity_eligible &
                                                          cohort.identity_status.ne('resolved')).sum()),
               'original_identity_status_counts': dict(Counter(cohort.source_identity_status)),
               'full_daily_context_loaded': daily_cohorts is not None}
    groups = {}
    for day in sessions:
        groups.setdefault(day[:7], []).append(day)
    target_windows = {}
    for signal in signals:
        following = str(pd.Period(signal[:7], 'M') + 1)
        evidence = str(pd.Period(signal[:7], 'M') + 2)
        if following in groups and evidence in groups:
            target_windows[signal] = {'outcome_month': following,
                'outcome_start': groups[following][0], 'outcome_end': groups[following][-1]}
    provenance = {'mode': mode, 'signal_dates': signals, 'requested_ids': list(ids),
                  'identity_eligibility_policy': identity_policy,
                  'feature_dictionary_sha256': canonical_sha256(specs),
                  'technical': daily.provenance, 'paper': paper.provenance, 'context': context.provenance,
                  'outcomes_consumed': False, 'feature_clock': 'Through closed signal session, inclusive',
                  'target_clock': 'Strictly after signal close through the next month-end; joined separately',
                  'knowledge_time_verified': False, 'cohort_denominator_preserved': True,
                  'source_limitations': 'Retrieval-time adjusted history and identity routes lack historical revision/validity certification.'}
    provenance['target_windows'] = target_windows
    provenance['calendar_sessions_sha256'] = canonical_sha256(sessions)
    provenance.update(identity_policy_provenance(identity_policy))
    return FeaturePanel(pa.Table.from_pandas(frame, preserve_index=False),
                        pa.Table.from_pandas(ledger_frame, preserve_index=False), specs,
                        {'daily': daily.coverage, 'paper': paper.coverage,
                         'context': context.coverage.loc[context.coverage.feature_name.ne('benchmark_daily_ew_return_1s')].reset_index(drop=True),
                         'daily_benchmark': context.coverage.loc[context.coverage.feature_name.eq('benchmark_daily_ew_return_1s')].reset_index(drop=True),
                         'paper_raw_momentum': paper.table[[*KEYS, *(c for c in paper.table if c.startswith('MOM') and ('_raw' in c))]].copy()},
                        summary, provenance)


def predictor_frame(panel, *, columns=None):
    """Explicit name allowlist; metadata, targets and audit flags cannot enter X."""
    registered = {s['name']: s for s in registered_feature_specs()}
    if len(panel.specs) != len(registered) or {s['name'] for s in panel.specs} != set(registered):
        raise DataContractError('Panel specs must match the frozen registered feature dictionary')
    allowed = [s['name'] for s in panel.specs if registered[s['name']].get('model_eligible', True)]
    chosen = allowed if columns is None else list(columns)
    if not chosen or len(chosen) != len(set(chosen)) or not set(chosen) <= set(allowed):
        raise DataContractError('Predictors must be distinct eligible names from the frozen feature dictionary')
    if set(panel.table.column_names) != set([*KEYS, *(s['name'] for s in panel.specs)]):
        raise DataContractError('Feature table contains unregistered inputs or target/metadata fields')
    selected = panel.table.select([*KEYS, *chosen]).to_pandas()
    if selected[list(KEYS)].isna().any().any() or selected.duplicated(list(KEYS)).any():
        raise DataContractError('Predictor source-ID/signal keys must be nonmissing and unique')
    for row in selected[list(KEYS)].to_dict('records'):
        if not isinstance(row['source_security_id'], str) or not row['source_security_id'].strip():
            raise DataContractError('Invalid predictor source ID')
        iso_date(row['signal_date'], 'predictor signal date')
        if row['signal_date'] >= '2024-01-01':
            raise DataContractError('Protected 2024+ predictor rows cannot enter the model interface')
    return selected.set_index(list(KEYS))


@dataclass(frozen=True)
class TrainingSlice:
    X: pd.DataFrame
    y: pd.Series
    ledger: pd.DataFrame
    provenance: dict


def training_slice(panel, outcomes, *, train_start, train_end, label_observed_through, columns=None):
    """Align by original keys and completed outcome dates, retaining missing y.

    No dropna, scaling, feature selection, label classes or model fit occurs.
    label_observed_through is an explicit conservative date-only research
    cutoff; actual publication/revision and fit/entry timestamps remain open.
    """
    first, last, observed = (iso_date(v, n) for v, n in
                            [(train_start, 'train_start'), (train_end, 'train_end'),
                             (label_observed_through, 'label_observed_through')])
    if first > last or observed > last or last >= '2024-01-01':
        raise DataContractError('Training requires ordered pre-holdout completed-outcome cutoffs')
    table = outcomes.table if hasattr(outcomes, 'table') else outcomes
    required = [*KEYS, 'outcome_month', 'outcome_start', 'outcome_end', 'return_status', 'provider_adjusted_return']
    if isinstance(table, pa.Table):
        if not set(required) <= set(table.column_names):
            raise DataContractError('Outcome ledger is missing required timing keys')
        dates = table.select([*KEYS, 'outcome_month', 'outcome_start', 'outcome_end']).to_pandas()
    elif isinstance(table, pd.DataFrame):
        if not set(required) <= set(table):
            raise DataContractError('Outcome ledger is missing required timing keys')
        dates = table[[*KEYS, 'outcome_month', 'outcome_start', 'outcome_end']].copy().reset_index(drop=True)
    else:
        raise DataContractError('Outcomes require an explicit Arrow table or DataFrame')
    for row in dates.to_dict('records'):
        s, a, e = [iso_date(row[k], k) for k in ['signal_date', 'outcome_start', 'outcome_end']]
        sm = pd.Period(s[:7], 'M')
        if not s < a <= e or a[:7] != e[:7] or e[:7] != str(sm + 1) or row['outcome_month'] != e[:7]:
            raise DataContractError('Target must occupy the next calendar month strictly after its signal')
        if e >= '2024-01-01':
            raise DataContractError('Protected financial target rows cannot enter training assembly')
        expected = panel.provenance.get('target_windows', {}).get(s)
        if expected is None or any(row[k] != expected[k] for k in ('outcome_month', 'outcome_start', 'outcome_end')):
            raise DataContractError('Target endpoints must match the independently pinned complete next-month calendar window')
    if dates.duplicated(list(KEYS)).any():
        raise DataContractError('Duplicate feature/target key')
    eligible = (dates.outcome_start >= first) & (dates.outcome_end <= min(last, observed))
    indices = dates.index[eligible]
    selected = table.take(pa.array(indices.tolist())).select(required).to_pandas() if isinstance(table, pa.Table) else table.iloc[indices][required].copy()
    X_all = predictor_frame(panel, columns=columns)
    target = selected.set_index(list(KEYS))
    expected_signals = {s for s, w in panel.provenance.get('target_windows', {}).items()
                        if w['outcome_start'] >= first and w['outcome_end'] <= min(last, observed)}
    expected_keys = X_all.index[X_all.index.get_level_values('signal_date').isin(expected_signals)]
    if set(expected_keys) != set(target.index):
        raise DataContractError('Target ledger changed the original signal-cohort denominator; missing members must remain explicit')
    if not target.index.isin(X_all.index).all():
        raise DataContractError('Training targets lack an original signal-cohort feature row')
    X = X_all.reindex(target.index).copy()
    values = pd.to_numeric(target.provider_adjusted_return, errors='coerce')
    y = values.where((target.return_status == 'ok') & np.isfinite(values)).rename('next_month_provider_adjusted_return')
    ledger = target.drop(columns=['provider_adjusted_return']).copy()
    ledger['feature_complete'] = X.notna().all(axis=1)
    ledger['target_available'] = y.notna()
    ledger['model_ready'] = False
    return TrainingSlice(X, y, ledger, {'train_start': first, 'train_end': last, 'label_observed_through': observed,
        'target_interval': '(signal_close,next_month_end_close]', 'feature_interval': 'ends_at_signal_close',
        'rows_retained': len(target), 'target_statuses_do_not_select_features': True, 'model_ready': False,
        'knowledge_time_verified': False, 'no_preprocessing_fit_performed': True})
