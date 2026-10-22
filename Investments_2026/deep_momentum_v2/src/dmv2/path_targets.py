"""Pure, versioned directional path targets; no source I/O, models or replay.

The caller supplies independently dated sessions and normalized standalone
economic wealth ledgers. This module does not certify their prices/entitlements.
It never substitutes a terminal close for the next-open endpoint.
"""
from dataclasses import dataclass
from collections import Counter
import datetime as dt
import hashlib
import json
import math
from numbers import Real

import pyarrow as pa
import pyarrow.compute as pc

POLICY = 'directional_economic_path_targets_v1'
FAMILIES = ('executable_return', 'return_minus_drawdown')
SIDES = ('long', 'short')
VARIANTS = ('unstopped', 'asset_stop_10pct')
KEYS = ('source_security_id', 'signal_date')
EPISODE_KEYS = (*KEYS, 'side', 'path_variant')
CLASSES = tuple(range(1, 11))
MARK_ORDER = {'open': 0, 'close': 1}
META = (*EPISODE_KEYS, 'target_family', 'outcome_month', 'entry_date', 'exit_date',
        'exit_mark', 'label_available_at', 'prediction_eligible', 'path_requested',
        'path_available', 'target_available', 'target_status')


@dataclass(frozen=True)
class PathTargetPanel:
    table: pa.Table
    diagnostics: pa.Table
    provenance: dict
    fingerprint: str


@dataclass(frozen=True)
class PathTrainingSlice:
    ledger: pa.Table
    fit_table: pa.Table
    provenance: dict


@dataclass(frozen=True)
class PathPayoffState:
    table: pa.Table
    monthly: pa.Table
    provenance: dict
    fingerprint: str


def _need(value, message):
    if not value:
        raise ValueError(message)


def _day(value, *, protected=True):
    _need(type(value) is str and dt.date.fromisoformat(value).isoformat() == value,
          'Canonical date required')
    _need(not protected or value < '2024-01-01', 'Protected financial date')
    return value


def _month(value):
    _need(type(value) is str and len(value) == 7 and value[4] == '-', 'Canonical month required')
    _need(dt.date.fromisoformat(value + '-01').strftime('%Y-%m') == value, 'Canonical month required')
    return value


def _shift(month, offset):
    _month(month); y, m = map(int, month.split('-')); n = y * 12 + m - 1 + offset
    return f'{n // 12:04d}-{n % 12 + 1:02d}'


def _hash(value):
    return 'sha256:' + hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
        separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def _table_hash(table):
    canonical = pa.Table.from_pylist(table.to_pylist(), schema=table.schema.remove_metadata())
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, canonical.schema) as stream:
        stream.write_table(canonical)
    return 'sha256:' + hashlib.sha256(sink.getvalue()).hexdigest()


def _columns(value):
    if isinstance(value, pa.Table):
        names = value.column_names
    elif hasattr(value, 'columns'):
        names = list(value.columns)
    else:
        _need(isinstance(value, (list, tuple)), 'Arrow, DataFrame or record sequence required')
        names = list(dict.fromkeys(k for row in value for k in row))
    _need(len(names) == len(set(names)), 'Duplicate table columns')
    return names


def _rows(value, columns, indices=None):
    _need(set(columns) <= set(_columns(value)), 'Missing required input columns')
    if isinstance(value, pa.Table):
        table = value.select(list(columns))
        if indices is not None:
            table = table.take(pa.array(indices, type=pa.int64()))
        return table.to_pylist()
    if hasattr(value, 'iloc'):
        frame = value.loc[:, list(columns)]
        if indices is not None:
            frame = frame.iloc[indices]
        # Preserve nullable original metadata consistently across pandas
        # versions. Select the authorized columns/rows before conversion.
        return pa.Table.from_pandas(frame, preserve_index=False).to_pylist()
    source = value if indices is None else [value[i] for i in indices]
    return [{k: row.get(k) for k in columns} for row in source]


def _identifier(value):
    _need(type(value) is str and value and value.strip() == value, 'Canonical source ID required')
    return value


def _key(row):
    return _identifier(row['source_security_id']), _day(row['signal_date'])


def _episode_key(row):
    _need(row['side'] in SIDES and row['path_variant'] in VARIANTS, 'Unknown side or stop variant')
    return (*_key(row), row['side'], row['path_variant'])


def _calendar(windows, sessions, endpoint_policy):
    _need(endpoint_policy in ('next_open', 'terminal_close'), 'Explicit endpoint policy required')
    dates = list(sessions)
    _need(dates and dates == sorted(set(dates)), 'Unique sorted independent sessions required')
    for date in dates:
        _day(date, protected=False)
    months = {}
    for date in dates:
        months.setdefault(date[:7], []).append(date)
    records = _rows(windows, ('signal_date', 'outcome_month', 'entry_date', 'exit_date', 'exit_mark'))
    result = {}
    for row in records:
        signal = _day(row['signal_date']); month = _month(row['outcome_month'])
        _need('2010-01' <= month <= '2023-12' and _shift(signal[:7], 1) == month,
              'Path outcome outside next-month 2010–2023 scope')
        prior, current = months.get(_shift(month, -1), []), months.get(month, [])
        _need(prior and current and signal == prior[-1] and row['entry_date'] == current[0],
              'Signal/entry differs from independent session calendar')
        if endpoint_policy == 'next_open':
            following = months.get(_shift(month, 1), [])
            _need(following and row['exit_date'] == following[0] and row['exit_mark'] == 'open',
                  'Next-open endpoint cannot be replaced by a terminal close')
            grid = [(date, mark) for date in current for mark in ('open', 'close')]
            grid.append((following[0], 'open'))
        else:
            _need(row['exit_date'] == current[-1] and row['exit_mark'] == 'close',
                  'Explicit terminal-close endpoint differs from calendar')
            grid = [(date, mark) for date in current for mark in ('open', 'close')]
        _need(signal not in result, 'Duplicate monthly window')
        result[signal] = {**row, 'label_available_at': row['exit_date'], 'required_grid': grid}
    _need(result and len({w['outcome_month'] for w in result.values()}) == len(result), 'Unique nonempty windows required')
    return result, _hash(dates)


def _wealth_statistics(values, penalty):
    peak, drawdown = 1.0, 0.0
    for value in values:
        if value is None:
            return None
        _need(isinstance(value, Real) and not isinstance(value, bool), 'Wealth must be numeric, never coerced')
        value = float(value)
        if not math.isfinite(value):
            return None
        peak = max(peak, value)
        drawdown = max(drawdown, (peak - value) / peak)
    _need(values and abs(float(values[0]) - 1.0) <= 1e-12,
          'Zero-cost normalized wealth must start at one within 1e-12; no rescaling')
    cash_return = float(values[-1]) - 1.0
    utility = cash_return - penalty * drawdown
    if not all(map(math.isfinite, (cash_return, drawdown, utility))):
        return None
    return cash_return, drawdown, utility, any(value <= 0 for value in values)


def _typed_table(records):
    table = pa.Table.from_pylist(records)
    types = {'cash_return': pa.float64(), 'maximum_drawdown': pa.float64(), 'utility': pa.float64(),
             'target_value': pa.float64(), 'target_class': pa.int8(), 'path_requested': pa.bool_()}
    for name, dtype in types.items():
        table = table.set_column(table.schema.get_field_index(name), name, pa.array([r[name] for r in records], type=dtype))
    return table


def build_path_targets(original_cohort, episodes, marks, *, windows, sessions,
                       observed_through, risk_penalty=1.0, endpoint_policy='next_open'):
    """Retain all original keys × two directions × stops × target families.

    ``marks.wealth`` is standalone economic capital from an initial 1, including
    signed action cash/shares. Costs, borrow and funding are zero in this version.
    Future/unavailable path wealth is never requested from the supplied table.
    """
    cutoff = _day(observed_through)
    _need(isinstance(risk_penalty, Real) and not isinstance(risk_penalty, bool)
          and math.isfinite(risk_penalty) and risk_penalty >= 0, 'Finite nonnegative risk penalty required')
    clock, calendar_hash = _calendar(windows, sessions, endpoint_policy)
    cohort = _rows(original_cohort, _columns(original_cohort))
    _need(cohort and all(type(r.get('prediction_eligible')) is bool for r in cohort), 'Explicit original admission required')
    reserved = {'outcome_month', 'entry_date', 'exit_date', 'exit_mark', 'label_available_at',
                'side', 'path_variant', 'target_family', 'cash_return', 'maximum_drawdown', 'utility',
                'target_value', 'target_class', 'target_available', 'path_available', 'path_requested'}
    _need(not reserved.intersection(_columns(original_cohort)), 'Original cohort contains target fields')
    original = {_key(r): r for r in cohort}
    _need(len(original) == len(cohort) and {k[1] for k in original} == set(clock), 'Duplicate or unknown original keys/windows')
    ep_rows = _rows(episodes, (*EPISODE_KEYS, 'requested', 'status', 'reason')) if len(episodes) else []
    lookup = {}
    for row in ep_rows:
        key = _episode_key(row)
        _need(key[:2] in original and key not in lookup, 'Extra or duplicate episode key')
        _need(row['requested'] is None or type(row['requested']) is bool, 'Explicit nullable path request state required')
        _need(row['status'] in ('complete', 'unavailable', 'not_requested')
              and type(row['reason']) is str, 'Explicit path status/reason required')
        lookup[key] = row
    mark_meta = _rows(marks, (*EPISODE_KEYS, 'date', 'mark')) if len(marks) else []
    groups = {}; seen = set()
    for index, row in enumerate(mark_meta):
        key = _episode_key(row); date = _day(row['date']); mark = row['mark']
        _need(key[:2] in original and mark in MARK_ORDER, 'Unknown path mark metadata')
        tick = (date, mark)
        _need(tick in clock[key[1]]['required_grid'], 'Mark outside independent required grid')
        _need((*key, *tick) not in seen, 'Duplicate path mark')
        seen.add((*key, *tick)); groups.setdefault(key, []).append((tick, index))
    rows = []
    for member in cohort:
        key = _key(member); window = clock[key[1]]
        for side in SIDES:
            for variant in VARIANTS:
                ep_key = (*key, side, variant); episode = lookup.get(ep_key)
                requested = episode['requested'] if episode else None
                statistics = None; status = 'missing_episode' if episode is None else episode['reason']
                if not member['prediction_eligible']:
                    status = 'prediction_ineligible'
                elif window['exit_date'] >= '2024-01-01':
                    status = 'protected_next_open_endpoint_unavailable'
                elif window['exit_date'] > cutoff:
                    status = 'scheduled_endpoint_not_observed'
                elif requested is not True:
                    status = 'path_request_not_exhaustive'
                elif episode['status'] != 'complete':
                    status = 'source_path_unavailable:' + episode['reason']
                else:
                    positions = sorted(groups.get(ep_key, []), key=lambda x: (x[0][0], MARK_ORDER[x[0][1]]))
                    if [tick for tick, _ in positions] != window['required_grid']:
                        status = 'incomplete_required_path_marks'
                    else:
                        values = [r['wealth'] for r in _rows(marks, ('wealth',), [i for _, i in positions])]
                        statistics = _wealth_statistics(values, float(risk_penalty))
                        status = 'available' if statistics is not None else 'nonfinite_or_missing_path_wealth'
                for family in FAMILIES:
                    cash, drawdown, utility, nonpositive = statistics or (None, None, None, None)
                    rows.append({**member, **{k: window[k] for k in ('outcome_month', 'entry_date', 'exit_date', 'exit_mark', 'label_available_at')},
                        'side': side, 'path_variant': variant, 'target_family': family,
                        'path_requested': requested, 'path_available': statistics is not None,
                        'path_status': status, 'cash_return': cash, 'maximum_drawdown': drawdown,
                        'utility': utility, 'nonpositive_wealth_observed': nonpositive,
                        'drawdown_status': None if statistics is None else ('defined_initial_positive_highwater_nonpositive_wealth' if nonpositive else 'defined_positive_wealth'),
                        'target_value': cash if family == 'executable_return' else utility,
                        'target_class': None, 'target_available': False, 'target_status': status})
    diagnostics = []
    for signal in sorted(clock):
        for side in SIDES:
            for variant in VARIANTS:
                for family in FAMILIES:
                    group = [r for r in rows if (r['signal_date'], r['side'], r['path_variant'], r['target_family']) == (signal, side, variant, family)]
                    eligible = [r for r in group if r['prediction_eligible']]
                    valid = [r for r in eligible if r['path_available']]
                    if any(r['path_requested'] is not True for r in eligible):
                        status = 'non_exhaustive_original_path_requests'
                    elif clock[signal]['exit_date'] > cutoff:
                        status = 'scheduled_endpoint_not_observed'
                    elif len(valid) < 10:
                        status = 'insufficient_observed_population'
                    else:
                        status = 'available'; ordered = sorted(valid, key=lambda r: r['target_value'], reverse=True)
                        lower = 0
                        while lower < len(ordered):
                            upper = lower + 1
                            while upper < len(ordered) and ordered[upper]['target_value'] == ordered[lower]['target_value']:
                                upper += 1
                            class_id = min(10, 1 + (10 * (lower + upper)) // (2 * len(ordered)))
                            for row in ordered[lower:upper]:
                                row.update(target_class=class_id, target_available=True, target_status='available')
                            lower = upper
                    if status != 'available':
                        for row in valid:
                            row['target_status'] = status
                    counts = Counter(r['target_class'] for r in valid if r['target_available'])
                    diagnostics.append({'signal_date': signal, 'outcome_month': clock[signal]['outcome_month'],
                        'side': side, 'path_variant': variant, 'target_family': family,
                        'original_members': len(group), 'prediction_eligible_members': len(eligible),
                        'observed_paths': len(valid), 'labelled_members': sum(counts.values()),
                        'exhaustive_requests': all(r['path_requested'] is True for r in eligible),
                        'nonpositive_wealth_paths': sum(r['nonpositive_wealth_observed'] is True for r in valid),
                        'empty_classes': [c for c in CLASSES if c not in counts], 'rank_status': status})
    table = _typed_table(rows); diagnostic = pa.Table.from_pylist(diagnostics)
    month_pins = {m: _table_hash(table.filter(pc.equal(table['outcome_month'], m)))
                  for m in sorted({r['outcome_month'] for r in rows})}
    provenance = {'policy_id': POLICY, 'endpoint_policy': endpoint_policy, 'risk_penalty': float(risk_penalty),
        'risk_penalty_status': 'engineering_candidate_not_empirically_selected', 'target_cost_bps': 0,
        'initial_wealth_tolerance': 1e-12,
        'borrow_and_funding_bps': 0, 'observed_through': cutoff, 'calendar_sessions_sha256': calendar_hash,
        'calendar_contract': 'caller_supplied_independent_XNYS_sessions_not_quote_inventory',
        'windows': list(clock.values()), 'original_key_sha256': _hash([list(_key(r)) for r in cohort]),
        'metadata_sha256': _table_hash(table.select(list(META))), 'month_fingerprints': month_pins,
        'drawdown': 'positive_peak_to_trough_fraction_of_running_highwater_including_initial_1',
        'nonpositive_wealth': 'retain_finite_cash_return_and_DD_above_one_with_explicit_audit_flag',
        'original_endpoint_control_rebuilt': False, 'no_future_availability_prediction_filter': True,
        'accounting_certified': False, 'knowledge_time_certified': False, 'estimator_fits': 0}
    return PathTargetPanel(table, diagnostic, provenance, _hash(provenance))


def _verify_metadata(panel):
    _need(isinstance(panel, PathTargetPanel) and _hash(panel.provenance) == panel.fingerprint, 'Path target provenance changed')
    _need(_table_hash(panel.table.select(list(META))) == panel.provenance['metadata_sha256'], 'Path target metadata changed')


def completed_training_slice(panel, *, train_start_month, train_end_month, as_of,
                             target_family, side, path_variant):
    """Return all original slice metadata plus only completed supervised values."""
    _verify_metadata(panel); cutoff = _day(as_of)
    first, last = _month(train_start_month), _month(train_end_month)
    _need(first <= last and target_family in FAMILIES and side in SIDES and path_variant in VARIANTS, 'Invalid training slice')
    metadata = panel.table.select(list(META)).to_pylist(); positions = []; ledger = []
    for i, row in enumerate(metadata):
        if first <= row['outcome_month'] <= last and (row['target_family'], row['side'], row['path_variant']) == (target_family, side, path_variant):
            fit = row['prediction_eligible'] and row['target_available'] and row['label_available_at'] <= cutoff
            ledger.append({**row, 'fit_eligible': fit, 'as_of': cutoff})
            if fit:
                positions.append(i)
    selected = panel.table.take(pa.array(positions, type=pa.int64()))
    for month in {row['outcome_month'] for row in ledger if row['fit_eligible']}:
        _need(_table_hash(panel.table.filter(pc.equal(panel.table['outcome_month'], month)))
              == panel.provenance['month_fingerprints'][month], 'Completed training month values changed')
    return PathTrainingSlice(pa.Table.from_pylist(ledger), selected,
        {'as_of': cutoff, 'train_start_month': first, 'train_end_month': last,
         'prediction_eligibility_unchanged': True, 'fit_rows': len(positions), 'fit_performed': False})


def build_path_payoffs(panel, *, signal_date, target_family, side, path_variant, payoff_field):
    """Exactly 120 consecutive calendar outcome months, availability-aware.

    Next-open labels exclude the signal calendar month. Missing month/class cells
    withhold the whole vector; neither time gaps nor empty classes are compressed.
    signal_date is the caller's independently established monthly forecast cutoff.
    Availability here is date-only end-of-day, not same-open operating certification.
    """
    _verify_metadata(panel); signal = _day(signal_date)
    _need(signal <= panel.provenance['observed_through'] and target_family in FAMILIES
          and side in SIDES and path_variant in VARIANTS and payoff_field in ('cash_return', 'utility'), 'Invalid payoff scope')
    last = _shift(signal[:7], -1) if panel.provenance['endpoint_policy'] == 'next_open' else signal[:7]
    months = [_shift(last, i) for i in range(-119, 1)]
    window_by_month = {w['outcome_month']: w for w in panel.provenance['windows']}
    metadata = panel.table.select(list(META)).to_pylist(); monthly = []
    for month in months:
        window = window_by_month.get(month); groups = {c: [] for c in CLASSES}
        indices = [i for i, r in enumerate(metadata) if r['outcome_month'] == month
                   and (r['target_family'], r['side'], r['path_variant']) == (target_family, side, path_variant)]
        status = 'missing_calendar_month' if window is None or not indices else None
        if window is not None and window['label_available_at'] > signal:
            status = 'scheduled_endpoint_not_observed'
        if status is None:
            selected = panel.table.filter(pc.equal(panel.table['outcome_month'], month))
            _need(_table_hash(selected) == panel.provenance['month_fingerprints'][month], 'Completed month values changed')
            for row in panel.table.take(pa.array(indices, type=pa.int64())).select(['target_available', 'target_class', payoff_field]).to_pylist():
                if row['target_available']:
                    _need(type(row['target_class']) is int and row['target_class'] in CLASSES
                          and isinstance(row[payoff_field], Real) and not isinstance(row[payoff_field], bool)
                          and math.isfinite(row[payoff_field]), 'Invalid completed class payoff')
                    groups[row['target_class']].append(row[payoff_field])
        for class_id in CLASSES:
            values = groups[class_id]
            monthly.append({'outcome_month': month, 'class_id': class_id,
                'original_member_count': len(indices),
                'prediction_eligible_count': sum(metadata[i]['prediction_eligible'] for i in indices),
                'member_count': len(values),
                'mean_payoff': math.fsum(x / len(values) for x in values) if values else None,
                'status': 'available' if values else (status or 'empty_realized_class')})
    available = all(r['status'] == 'available' for r in monthly)
    table = pa.Table.from_pylist([{'class_id': c, 'payoff': math.fsum(r['mean_payoff'] / 120 for r in monthly if r['class_id'] == c) if available else None,
        'status': 'available' if available else 'withheld_incomplete_120_month_class_grid'} for c in CLASSES])
    provenance = {'policy_id': POLICY, 'source_policy_fingerprint': panel.fingerprint, 'signal_date': signal,
        'label_observed_through': signal,
        'target_family': target_family, 'side': side, 'path_variant': path_variant, 'payoff_field': payoff_field,
        'score_semantics': 'expected_executable_cash_return' if payoff_field == 'cash_return' else 'expected_path_utility_not_original_RET',
        'first_outcome_month': months[0], 'last_outcome_month': last, 'completed_calendar_months': 120,
        'vector_available': available, 'time_weighting': 'equal_calendar_month_means',
        'target_cost_bps': 0, 'risk_penalty': panel.provenance['risk_penalty']}
    monthly_table = pa.Table.from_pylist(monthly)
    fingerprint = _hash({'provenance': provenance, 'table': table.to_pylist(), 'monthly': monthly})
    return PathPayoffState(table, monthly_table, provenance, fingerprint)


def score_path_probabilities(probabilities, payoffs, *, class_ids, signal_date, as_of):
    _need(isinstance(payoffs, PathPayoffState) and payoffs.fingerprint == _hash({
        'provenance': payoffs.provenance, 'table': payoffs.table.to_pylist(), 'monthly': payoffs.monthly.to_pylist()}), 'Payoff state changed')
    signal, cutoff = _day(signal_date), _day(as_of)
    _need(signal == cutoff == payoffs.provenance['signal_date']
          == payoffs.provenance['label_observed_through'], 'Score must use the exact causal monthly payoff cutoff')
    ids, values = list(class_ids), list(probabilities)
    _need(len(ids) == len(values) == 10 and all(type(c) is int for c in ids) and set(ids) == set(CLASSES), 'Explicit complete class alignment required')
    _need(all(isinstance(v, Real) and not isinstance(v, bool) and math.isfinite(v) and v >= 0 for v in values)
          and abs(math.fsum(values) - 1) <= 1e-8, 'Finite probability simplex required; no renormalization')
    score = None
    if payoffs.provenance['vector_available']:
        lookup = dict(zip(ids, values))
        score = math.fsum(lookup[r['class_id']] * r['payoff'] for r in payoffs.table.to_pylist())
    return {'score': score, 'score_semantics': payoffs.provenance['score_semantics'],
        'side': payoffs.provenance['side'], 'path_variant': payoffs.provenance['path_variant'],
        'status': 'available' if score is not None else 'withheld_incomplete_120_month_class_grid'}
