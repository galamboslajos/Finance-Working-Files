"""Pure signed executable-return / drawdown targets. No I/O, prices or models.

This version deliberately preserves the literal signed ratio, including its
unfavourable negative-return ordering. It is an engineering candidate, not a
validated utility measure. The one-percent denominator floor is frozen, not fit.
Inputs must be the projected, independently verified *unstopped executable
return* target rows. This module does not reconstruct or certify their paths.
"""
from collections import Counter
from dataclasses import dataclass
import datetime as dt
import hashlib
import json
import math
from numbers import Real

import pyarrow as pa
import pyarrow.compute as pc

POLICY = 'signed_executable_return_over_drawdown_v1'
FAMILY = 'return_over_drawdown'
DENOMINATOR_FLOOR = 0.01
SIDES = ('long', 'short')
CLASSES = tuple(range(1, 11))
KEYS = ('source_security_id', 'signal_date', 'side', 'path_variant')
SOURCE_META = (*KEYS, 'target_family', 'outcome_month', 'entry_date', 'exit_date',
               'exit_mark', 'label_available_at', 'prediction_eligible',
               'path_requested', 'path_available', 'target_available', 'target_status')
META = (*SOURCE_META, 'source_target_family', 'source_target_available',
        'source_target_status', 'ratio_input_status')
NUMERIC_SOURCE = ('cash_return', 'maximum_drawdown')


@dataclass(frozen=True)
class ReturnDDPanel:
    table: pa.Table
    diagnostics: pa.Table
    provenance: dict
    fingerprint: str


@dataclass(frozen=True)
class ReturnDDTrainingSlice:
    ledger: pa.Table
    fit_table: pa.Table
    provenance: dict


@dataclass(frozen=True)
class ReturnDDPayoffState:
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
    _need(type(value) is str and len(value) == 7
          and dt.date.fromisoformat(value + '-01').strftime('%Y-%m') == value,
          'Canonical month required')
    return value


def _shift(month, offset):
    _month(month)
    year, mon = map(int, month.split('-')); n = year * 12 + mon - 1 + offset
    return f'{n // 12:04d}-{n % 12 + 1:02d}'


def _hash(value):
    return 'sha256:' + hashlib.sha256(json.dumps(value, sort_keys=True,
        ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def _table_hash(table):
    # Canonical Arrow batches avoid chunking-dependent fingerprints.
    canonical = pa.Table.from_pylist(table.to_pylist(), schema=table.schema.remove_metadata())
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, canonical.schema) as stream:
        stream.write_table(canonical)
    return 'sha256:' + hashlib.sha256(sink.getvalue()).hexdigest()


def _columns(value):
    if isinstance(value, pa.Table):
        names = value.column_names
    else:
        _need(isinstance(value, (list, tuple)) and value,
              'Nonempty Arrow table or Python record sequence required')
        names = list(dict.fromkeys(k for row in value for k in row))
    _need(len(names) == len(set(names)), 'Duplicate input columns')
    return names


def _rows(value, columns, indices=None):
    _need(set(columns) <= set(_columns(value)), 'Missing required input columns')
    if isinstance(value, pa.Table):
        table = value.select(list(columns))
        if indices is not None:
            table = table.take(pa.array(indices, type=pa.int64()))
        return table.to_pylist()
    records = value if indices is None else [value[i] for i in indices]
    return [{column: row.get(column) for column in columns} for row in records]


def _key(row):
    source = row['source_security_id']
    _need(type(source) is str and source and source.strip() == source,
          'Canonical source security ID required')
    _need(row['side'] in SIDES and row['path_variant'] == 'unstopped',
          'Only independent long/short unstopped paths are allowed')
    return source, _day(row['signal_date']), row['side'], row['path_variant']


def signed_ratio(cash_return, maximum_drawdown):
    """Return (value, status, floor-hit, negative-return) without coercion.

    None stays unavailable. Finite negative drawdown and nonfinite observations
    are invalid, never repaired. DD above one and finite losses below -100% are
    retained because standalone short wealth can become nonpositive.
    """
    if cash_return is None or maximum_drawdown is None:
        return None, 'missing_return_or_drawdown', None, None
    if any(not isinstance(x, Real) or isinstance(x, bool)
           for x in (cash_return, maximum_drawdown)):
        return None, 'invalid_non_numeric_input', None, None
    ret, dd = float(cash_return), float(maximum_drawdown)
    if not math.isfinite(ret) or not math.isfinite(dd):
        return None, 'invalid_nonfinite_input', None, None
    if dd < 0:
        return None, 'invalid_negative_drawdown', None, None
    value = ret / max(dd, DENOMINATOR_FLOOR)
    if not math.isfinite(value):
        return None, 'invalid_nonfinite_ratio', dd < DENOMINATOR_FLOOR, ret < 0
    return value, 'available', dd < DENOMINATOR_FLOOR, ret < 0


def _typed_table(rows):
    table = pa.Table.from_pylist(rows)
    types = {'cash_return': pa.float64(), 'maximum_drawdown': pa.float64(),
             'target_value': pa.float64(), 'target_class': pa.int8(),
             'source_cash_target_value': pa.float64(), 'source_cash_target_class': pa.int8(),
             'ratio_floor_hit': pa.bool_(), 'ratio_negative_return': pa.bool_(),
             'path_requested': pa.bool_()}
    for name, dtype in types.items():
        table = table.set_column(table.schema.get_field_index(name), name,
                                pa.array([row[name] for row in rows], type=dtype))
    return table


def build_return_dd_targets(source, *, observed_through, source_provenance=None):
    """Build a new label family without changing original admission or clocks.

    Metadata is selected first. Numeric source values are inspected only for
    originally eligible, requested, available paths whose next-open endpoint
    has completed by observed_through. No source target-class availability is
    used as a future prediction filter. All original source metadata columns
    are retained; existing numeric target fields are not copied into audit data.
    """
    cutoff = _day(observed_through)
    names = _columns(source)
    _need(set((*SOURCE_META, *NUMERIC_SOURCE)) <= set(names), 'Incomplete executable target projection')
    forbidden = {'source_target_family', 'source_target_available', 'source_target_status',
                 'ratio_input_status', 'ratio_floor_hit', 'ratio_negative_return'}
    _need(not forbidden.intersection(names), 'Ratio output columns already present')
    financial = {'cash_return', 'maximum_drawdown', 'utility', 'target_value', 'target_class'}
    metadata_names = [name for name in names if name not in financial]
    metadata = _rows(source, metadata_names)
    _need(metadata, 'Nonempty source target projection required')
    rows, windows, seen, numeric_indices = [], {}, set(), []
    for i, row in enumerate(metadata):
        key = _key(row)
        _need(key not in seen, 'Duplicate original episode key'); seen.add(key)
        _need(row['target_family'] == 'executable_return', 'Source must be executable_return only')
        _need(type(row['prediction_eligible']) is bool and type(row['path_available']) is bool
              and type(row['target_available']) is bool
              and (row['path_requested'] is None or type(row['path_requested']) is bool),
              'Explicit unchanged source admission/path flags required')
        month = _month(row['outcome_month'])
        _need('2010-01' <= month <= '2023-12' and _shift(row['signal_date'][:7], 1) == month,
              'Next-month outcome scope must remain 2010–2023')
        entry = _day(row['entry_date']); exit_date = _day(row['exit_date'], protected=False)
        available = _day(row['label_available_at'], protected=False)
        _need(row['signal_date'] < entry <= exit_date <= available
              and entry[:7] == month and exit_date[:7] == _shift(month, 1)
              and row['exit_mark'] == 'open', 'Exact next-open path completion clock required')
        clock = {name: row[name] for name in ('signal_date', 'outcome_month', 'entry_date',
                  'exit_date', 'exit_mark', 'label_available_at')}
        _need(month not in windows or windows[month] == clock, 'Inconsistent original monthly completion clock')
        windows[month] = clock
        status = ('prediction_ineligible' if not row['prediction_eligible'] else
                  'scheduled_endpoint_not_observed' if available > cutoff else
                  'protected_next_open_endpoint_unavailable' if exit_date >= '2024-01-01' else
                  'path_request_not_exhaustive' if row['path_requested'] is not True else
                  'source_path_unavailable' if not row['path_available'] else 'pending_numeric_check')
        result = {**row, 'source_target_family': row['target_family'],
            'source_target_available': row['target_available'],
            'source_target_status': row['target_status'],
            'target_family': FAMILY, 'cash_return': None, 'maximum_drawdown': None,
            'target_value': None, 'target_class': None, 'target_available': False,
            'source_cash_target_value': None, 'source_cash_target_class': None,
            'target_status': status, 'ratio_input_status': status,
            'ratio_floor_hit': None, 'ratio_negative_return': None}
        rows.append(result)
        if status == 'pending_numeric_check':
            numeric_indices.append(i)
    numeric_columns = (*NUMERIC_SOURCE, *[name for name in ('target_value', 'target_class') if name in names])
    values = _rows(source, numeric_columns, numeric_indices)
    for i, value in zip(numeric_indices, values):
        ratio, status, floor, negative = signed_ratio(value['cash_return'], value['maximum_drawdown'])
        old_class, old_value = value.get('target_class'), value.get('target_value')
        _need(old_class is None or type(old_class) is int and old_class in CLASSES,
              'Invalid original executable-return class')
        _need(old_value is None or isinstance(old_value, Real) and not isinstance(old_value, bool)
              and math.isfinite(old_value), 'Invalid original executable-return target value')
        # Invalid/nonfinite source numbers are never serialized into new panels.
        if status == 'available':
            rows[i].update(cash_return=float(value['cash_return']),
                           maximum_drawdown=float(value['maximum_drawdown']),
                           source_cash_target_value=value.get('target_value'),
                           source_cash_target_class=value.get('target_class'))
        rows[i].update(target_value=ratio, target_status=status, ratio_input_status=status,
                       ratio_floor_hit=floor, ratio_negative_return=negative)
    diagnostics = []
    for month in sorted(windows):
        for side in SIDES:
            group = [row for row in rows if row['outcome_month'] == month and row['side'] == side]
            if not group:
                continue
            eligible = [row for row in group if row['prediction_eligible']]
            valid = [row for row in eligible if row['ratio_input_status'] == 'available']
            status = ('non_exhaustive_original_path_requests'
                      if any(row['path_requested'] is not True for row in eligible) else
                      'scheduled_endpoint_not_observed' if windows[month]['label_available_at'] > cutoff else
                      'insufficient_observed_population' if len(valid) < 10 else 'available')
            if status == 'available':
                ordered = sorted(valid, key=lambda row: row['target_value'], reverse=True)
                lo = 0
                while lo < len(ordered):
                    hi = lo + 1
                    while hi < len(ordered) and ordered[hi]['target_value'] == ordered[lo]['target_value']:
                        hi += 1
                    class_id = min(10, 1 + (10 * (lo + hi)) // (2 * len(ordered)))
                    for row in ordered[lo:hi]:
                        row.update(target_class=class_id, target_available=True, target_status='available')
                    lo = hi
            else:
                for row in valid:
                    row['target_status'] = status
            counts = Counter(row['target_class'] for row in valid if row['target_available'])
            ties = Counter(row['target_value'] for row in valid)
            diagnostics.append({'signal_date': windows[month]['signal_date'], 'outcome_month': month,
                'side': side, 'path_variant': 'unstopped', 'target_family': FAMILY,
                'original_members': len(group), 'prediction_eligible_members': len(eligible),
                'observed_source_paths': sum(row['path_available'] for row in eligible),
                'valid_ratio_values': len(valid), 'labelled_members': sum(counts.values()),
                'floor_hit_count': sum(row['ratio_floor_hit'] is True for row in valid),
                'zero_drawdown_count': sum(row['maximum_drawdown'] == 0 for row in valid),
                'negative_return_count': sum(row['ratio_negative_return'] is True for row in valid),
                'negative_return_above_floor_count': sum(row['ratio_negative_return'] is True
                    and row['maximum_drawdown'] >= DENOMINATOR_FLOOR for row in valid),
                'drawdown_above_one_count': sum(row['maximum_drawdown'] > 1 for row in valid),
                'largest_exact_tie_count': max(ties.values(), default=0),
                'exact_tie_group_count': sum(n > 1 for n in ties.values()),
                'minus_one_ratio_count': ties.get(-1.0, 0),
                'input_status_counts': dict(sorted(Counter(row['ratio_input_status'] for row in eligible).items())),
                'empty_classes': [class_id for class_id in CLASSES if not counts[class_id]],
                'class_member_counts': [counts[class_id] for class_id in CLASSES],
                'rank_status': status})
    table = _typed_table(rows)
    metadata_columns = list(dict.fromkeys([*metadata_names, *META]))
    provenance = {'policy_id': POLICY, 'target_family': FAMILY,
        'formula': 'cash_return / max(maximum_drawdown, 0.01)',
        'denominator_floor': DENOMINATOR_FLOOR, 'floor_status': 'fixed_engineering_candidate_not_tuned',
        'negative_returns': 'literal_signed_ratio_no_piecewise_transform_no_clipping',
        'negative_ratio_caveat': 'larger_drawdown_can_improve_negative_return_ratio',
        'endpoint_policy': 'next_open', 'path_variant': 'unstopped', 'target_cost_bps': 0,
        'observed_through': cutoff, 'windows': [windows[month] for month in sorted(windows)],
        'source_provenance': source_provenance, 'metadata_columns': metadata_columns,
        'metadata_sha256': _table_hash(table.select(metadata_columns)),
        'month_fingerprints': {month: _table_hash(table.filter(pc.equal(table['outcome_month'], month)))
                               for month in sorted(windows)},
        'prediction_eligibility_unchanged': True, 'class_one_is_highest': True,
        'class_ties': 'descending_exact_equal_midrank_no_jitter',
        'minimum_rank_population': 10, 'no_future_availability_prediction_filter': True,
        'diagnostics_are_features': False, 'accounting_certified_here': False,
        'estimator_fits': 0, 'trade_replays': 0}
    return ReturnDDPanel(table, pa.Table.from_pylist(diagnostics), provenance, _hash(provenance))


def _verify_metadata(panel):
    _need(isinstance(panel, ReturnDDPanel) and panel.fingerprint == _hash(panel.provenance),
          'Ratio target provenance changed')
    _need(_table_hash(panel.table.select(panel.provenance['metadata_columns'])) == panel.provenance['metadata_sha256'],
          'Ratio target metadata changed')


def _verify_month(panel, month):
    selected = panel.table.filter(pc.equal(panel.table['outcome_month'], month))
    _need(_table_hash(selected) == panel.provenance['month_fingerprints'][month],
          'Completed ratio month values changed')


def completed_training_slice(panel, *, train_start_month, train_end_month, as_of, side):
    """Keep all original metadata in the training ledger; supervise completed rows only."""
    _verify_metadata(panel)
    first, last, cutoff = _month(train_start_month), _month(train_end_month), _day(as_of)
    _need(first <= last and side in SIDES and cutoff <= panel.provenance['observed_through'],
          'Invalid completed training slice')
    metadata = panel.table.select(panel.provenance['metadata_columns']).to_pylist(); ledger, positions = [], []
    for i, row in enumerate(metadata):
        if first <= row['outcome_month'] <= last and row['side'] == side:
            fit = row['prediction_eligible'] and row['target_available'] and row['label_available_at'] <= cutoff
            ledger.append({**row, 'fit_eligible': fit, 'as_of': cutoff})
            if fit:
                positions.append(i)
    for month in {row['outcome_month'] for row in ledger if row['fit_eligible']}:
        _verify_month(panel, month)
    return ReturnDDTrainingSlice(pa.Table.from_pylist(ledger),
        panel.table.take(pa.array(positions, type=pa.int64())),
        {'policy_id': POLICY, 'as_of': cutoff, 'train_start_month': first,
         'train_end_month': last, 'side': side, 'fit_rows': len(positions),
         'prediction_eligibility_unchanged': True, 'fit_performed': False})


def monthly_class_means(panel):
    """Small, exact class sufficient statistics for streaming monthly audits.

    The caller may release the full monthly panel after saving these statistics.
    They are outcomes/payoff audit data and must never be joined into X.
    """
    _verify_metadata(panel)
    metadata = panel.table.select(list(META)).to_pylist()
    result = []
    for month, window in sorted((w['outcome_month'], w) for w in panel.provenance['windows']):
        _verify_month(panel, month)
        for side in SIDES:
            indices = [i for i, row in enumerate(metadata)
                       if row['outcome_month'] == month and row['side'] == side]
            if not indices:
                continue
            groups = {class_id: [] for class_id in CLASSES}
            complete = [i for i in indices if metadata[i]['target_available']
                        and metadata[i]['prediction_eligible']]
            for row in panel.table.take(pa.array(complete, type=pa.int64())).select(
                    ['target_class', 'target_value', 'cash_return']).to_pylist():
                _need(type(row['target_class']) is int and row['target_class'] in CLASSES
                      and all(isinstance(row[name], Real) and not isinstance(row[name], bool)
                      and math.isfinite(row[name]) for name in ('target_value', 'cash_return')),
                      'Invalid ratio-class sufficient statistic')
                groups[row['target_class']].append(row)
            for class_id in CLASSES:
                values = groups[class_id]; n = len(values)
                result.append({'signal_date': window['signal_date'], 'outcome_month': month,
                    'side': side, 'class_id': class_id,
                    'label_available_at': window['label_available_at'],
                    'original_members': len(indices),
                    'prediction_eligible_members': sum(metadata[i]['prediction_eligible'] for i in indices),
                    'member_count': n,
                    'mean_ratio': math.fsum(row['target_value'] / n for row in values) if n else None,
                    'mean_cash_return': math.fsum(row['cash_return'] / n for row in values) if n else None,
                    'status': 'available' if n else ('scheduled_endpoint_not_observed'
                        if window['label_available_at'] > panel.provenance['observed_through']
                        else 'empty_realized_class')})
    return result


def payoffs_from_monthly_means(stats, windows, *, signal_date, side,
                               observed_through, source_policy_fingerprint=None):
    """Build strict causal dual payoff vectors from small streaming statistics.

    All 120 consecutive calendar months and all ten classes are mandatory.
    Actual next-open completion must be <= signal_date; a calendar position
    alone never establishes availability. Missing cells withhold both vectors.
    """
    signal, cutoff = _day(signal_date), _day(observed_through)
    _need(side in SIDES and signal <= cutoff, 'Invalid payoff cutoff or side')
    # Inspect only metadata until the exact historical months, direction and
    # actual completion clock are known. Future class means must not even enter
    # a validation expression or fingerprint for an earlier signal.
    records = _rows(stats, ('signal_date', 'outcome_month', 'side', 'class_id',
        'label_available_at', 'original_members', 'prediction_eligible_members',
        'member_count', 'status'))
    _need({'mean_ratio', 'mean_cash_return'} <= set(_columns(stats)),
          'Distinct ratio and cash-return monthly mean columns required')
    metadata_fingerprint = _hash(records)
    clocks = _rows(windows, ('signal_date', 'outcome_month', 'entry_date', 'exit_date',
                            'exit_mark', 'label_available_at'))
    window_by_month = {}
    for window in clocks:
        month = _month(window['outcome_month'])
        _need(month not in window_by_month, 'Duplicate monthly completion clock')
        _need(_shift(_day(window['signal_date'])[:7], 1) == month
              and _day(window['entry_date'])[:7] == month
              and _day(window['exit_date'], protected=False)[:7] == _shift(month, 1)
              and window['signal_date'] < window['entry_date'] <= window['exit_date']
              <= _day(window['label_available_at'], protected=False)
              and window['exit_mark'] == 'open', 'Invalid next-open completion clock')
        window_by_month[month] = window
    lookup = {}
    for row in records:
        month = _month(row['outcome_month']); key = month, row['side'], row['class_id']
        _need(row['side'] in SIDES and type(row['class_id']) is int
              and row['class_id'] in CLASSES and key not in lookup, 'Duplicate or invalid monthly class cell')
        window = window_by_month.get(month)
        _need(window is not None and row['signal_date'] == window['signal_date']
              and row['label_available_at'] == window['label_available_at'],
              'Monthly class statistic completion clock differs')
        counts = [row[name] for name in ('original_members', 'prediction_eligible_members', 'member_count')]
        _need(all(type(n) is int and n >= 0 for n in counts) and counts[2] <= counts[1] <= counts[0],
              'Invalid monthly class membership counts')
        _need(row['status'] in ('available', 'empty_realized_class', 'scheduled_endpoint_not_observed'),
              'Unknown monthly class status')
        if row['status'] == 'available':
            _need(row['member_count'] > 0 and row['label_available_at'] <= cutoff,
                  'Invalid available monthly class metadata')
        else:
            _need(row['member_count'] == 0, 'Unavailable class membership must remain zero')
        lookup[key] = row
    last = _shift(signal[:7], -1); months = [_shift(last, i) for i in range(-119, 1)]
    month_set = set(months)
    inspected_indices = [i for i, row in enumerate(records) if row['side'] == side
        and row['outcome_month'] in month_set and row['label_available_at'] <= signal]
    inspected_records = []
    for index, values in zip(inspected_indices,
            _rows(stats, ('mean_ratio', 'mean_cash_return'), inspected_indices)):
        row = records[index]
        if row['status'] == 'available':
            _need(all(isinstance(values[name], Real) and not isinstance(values[name], bool)
                      and math.isfinite(values[name]) for name in ('mean_ratio', 'mean_cash_return')),
                  'Invalid available dual monthly class mean')
        else:
            _need(values['mean_ratio'] is None and values['mean_cash_return'] is None,
                  'Unavailable class means must remain null')
        row.update(values)
        inspected_records.append(row)
    monthly = []
    for month in months:
        window = window_by_month.get(month)
        for class_id in CLASSES:
            row = lookup.get((month, side, class_id))
            status = ('missing_calendar_month' if window is None else
                      'scheduled_endpoint_not_observed' if window['label_available_at'] > signal else
                      'missing_month_class_cell' if row is None else row['status'])
            available = status == 'available'
            monthly.append({'outcome_month': month, 'class_id': class_id,
                'original_member_count': row['original_members'] if row else 0,
                'prediction_eligible_count': row['prediction_eligible_members'] if row else 0,
                'member_count': row['member_count'] if available else 0,
                'class_mean_ratio': row['mean_ratio'] if available else None,
                'class_mean_cash_return': row['mean_cash_return'] if available else None,
                'status': status})
    available = all(row['status'] == 'available' for row in monthly)
    aggregate = [{'class_id': class_id,
        'class_mean_ratio': math.fsum(row['class_mean_ratio'] / 120 for row in monthly
                            if row['class_id'] == class_id) if available else None,
        'class_mean_cash_return': math.fsum(row['class_mean_cash_return'] / 120 for row in monthly
                            if row['class_id'] == class_id) if available else None,
        'status': 'available' if available else 'withheld_incomplete_120_month_class_grid'}
        for class_id in CLASSES]
    provenance = {'policy_id': POLICY, 'source_policy_fingerprint': source_policy_fingerprint,
        'source_monthly_statistics_fingerprint': _hash(inspected_records),
        'source_monthly_metadata_fingerprint': metadata_fingerprint,
        'statistics_hash_scope': 'requested_side_required_120_months_actual_completion_le_signal',
        'inspected_monthly_class_cells': len(inspected_records),
        'target_family': FAMILY, 'side': side, 'path_variant': 'unstopped',
        'signal_date': signal, 'label_observed_through': signal,
        'first_outcome_month': months[0], 'last_outcome_month': last,
        'required_calendar_months': 120, 'required_month_class_cells': 1200,
        'vector_available': available, 'time_weighting': 'equal_calendar_month_means',
        'ratio_score_semantics': 'expected_signed_return_over_drawdown_not_cash_return',
        'cash_score_semantics': 'expected_executable_cash_return_conditional_on_ratio_classes'}
    table, monthly_table = pa.Table.from_pylist(aggregate), pa.Table.from_pylist(monthly)
    fingerprint = _hash({'provenance': provenance, 'table': aggregate, 'monthly': monthly})
    return ReturnDDPayoffState(table, monthly_table, provenance, fingerprint)


def build_return_dd_payoffs(panel, *, signal_date, side):
    """Release both ratio and cash-return class means after a full causal 120 months.

    The window always ends in signal calendar month minus one. Every one of the
    120x10 cells must have completed observations. Means are equal-weighted by
    calendar month, not pooled across a changing number of securities.
    """
    _verify_metadata(panel); signal = _day(signal_date)
    _need(side in SIDES and signal <= panel.provenance['observed_through'], 'Invalid payoff cutoff or side')
    last = _shift(signal[:7], -1); months = [_shift(last, i) for i in range(-119, 1)]
    windows = {window['outcome_month']: window for window in panel.provenance['windows']}
    metadata = panel.table.select(list(META)).to_pylist()
    monthly = []
    for month in months:
        window = windows.get(month)
        indices = [i for i, row in enumerate(metadata) if row['outcome_month'] == month and row['side'] == side]
        groups = {class_id: [] for class_id in CLASSES}
        status = 'missing_calendar_month' if window is None or not indices else None
        if window is not None and window['label_available_at'] > signal:
            status = 'scheduled_endpoint_not_observed'
        if status is None:
            _verify_month(panel, month)
            completed = [i for i in indices if metadata[i]['target_available']
                         and metadata[i]['prediction_eligible'] and metadata[i]['label_available_at'] <= signal]
            for row in panel.table.take(pa.array(completed, type=pa.int64())).select(
                    ['target_class', 'target_value', 'cash_return']).to_pylist():
                _need(type(row['target_class']) is int and row['target_class'] in CLASSES
                      and all(isinstance(row[name], Real) and not isinstance(row[name], bool)
                              and math.isfinite(row[name]) for name in ('target_value', 'cash_return')),
                      'Invalid completed ratio-class payoff')
                groups[row['target_class']].append(row)
        for class_id in CLASSES:
            records = groups[class_id]; n = len(records)
            monthly.append({'outcome_month': month, 'class_id': class_id,
                'original_member_count': len(indices),
                'prediction_eligible_count': sum(metadata[i]['prediction_eligible'] for i in indices),
                'member_count': n,
                'class_mean_ratio': math.fsum(row['target_value'] / n for row in records) if n else None,
                'class_mean_cash_return': math.fsum(row['cash_return'] / n for row in records) if n else None,
                'status': 'available' if n else (status or 'empty_realized_class')})
    available = all(row['status'] == 'available' for row in monthly)
    records = [{'class_id': class_id,
        'class_mean_ratio': math.fsum(row['class_mean_ratio'] / 120 for row in monthly
                            if row['class_id'] == class_id) if available else None,
        'class_mean_cash_return': math.fsum(row['class_mean_cash_return'] / 120 for row in monthly
                            if row['class_id'] == class_id) if available else None,
        'status': 'available' if available else 'withheld_incomplete_120_month_class_grid'} for class_id in CLASSES]
    provenance = {'policy_id': POLICY, 'source_policy_fingerprint': panel.fingerprint,
        'target_family': FAMILY, 'side': side, 'path_variant': 'unstopped',
        'signal_date': signal, 'label_observed_through': signal,
        'first_outcome_month': months[0], 'last_outcome_month': last,
        'required_calendar_months': 120, 'required_month_class_cells': 1200,
        'vector_available': available, 'time_weighting': 'equal_calendar_month_means',
        'ratio_score_semantics': 'expected_signed_return_over_drawdown_not_cash_return',
        'cash_score_semantics': 'expected_executable_cash_return_conditional_on_ratio_classes'}
    table, monthly_table = pa.Table.from_pylist(records), pa.Table.from_pylist(monthly)
    fingerprint = _hash({'provenance': provenance, 'table': records, 'monthly': monthly})
    return ReturnDDPayoffState(table, monthly_table, provenance, fingerprint)


def score_return_dd_probabilities(probabilities, state, *, class_ids, signal_date, as_of):
    """Return distinct expected-ratio and expected-cash scores, never conflate them."""
    _need(isinstance(state, ReturnDDPayoffState) and state.fingerprint == _hash({
        'provenance': state.provenance, 'table': state.table.to_pylist(),
        'monthly': state.monthly.to_pylist()}), 'Ratio payoff state changed')
    signal, cutoff = _day(signal_date), _day(as_of)
    _need(signal == cutoff == state.provenance['signal_date'] == state.provenance['label_observed_through'],
          'Exact causal monthly payoff cutoff required')
    ids, values = list(class_ids), list(probabilities)
    _need(len(ids) == 10 and set(ids) == set(CLASSES) and all(type(i) is int for i in ids)
          and len(values) == 10 and all(isinstance(v, Real) and not isinstance(v, bool)
          and math.isfinite(v) and 0 <= v <= 1 for v in values)
          and math.isclose(math.fsum(values), 1.0, rel_tol=0, abs_tol=1e-10),
          'Ten explicitly identified normalized class probabilities required')
    scores = {'expected_return_over_drawdown': None, 'expected_executable_cash_return': None}
    if state.provenance['vector_available']:
        lookup = dict(zip(ids, values))
        for result, column in (('expected_return_over_drawdown', 'class_mean_ratio'),
                               ('expected_executable_cash_return', 'class_mean_cash_return')):
            scores[result] = math.fsum(lookup[row['class_id']] * row[column] for row in state.table.to_pylist())
    return {**scores, 'side': state.provenance['side'], 'signal_date': signal,
            'payoff_fingerprint': state.fingerprint, 'vector_available': state.provenance['vector_available']}
