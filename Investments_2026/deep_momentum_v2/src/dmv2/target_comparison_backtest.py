"""Frozen three-target, signal-only selection and unchanged daily accounting.

Import performs no reads. Planning opens only source JSON/Parquet metadata;
``run_backtest`` is the explicit, separately guarded numerical entry point.
Original endpoint returns and new cash/utility targets are never substituted
for one another, and execution-only quote repairs never enter learning payoffs.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date
import hashlib
import itertools
import json
import math
from pathlib import Path
import time

ARMS = ('endpoint_matched_clock', 'cash', 'utility')
BASKETS = ('score', 'best_mode', 'second_mode')
HEADS = ('endpoint', 'cash_long', 'cash_short', 'utility_long', 'utility_short')
VARIANTS = {'unstopped': None, 'asset_stop_10pct': .1}
FEES = (0, 5, 10, 25)
YEARS = (2020, 2021, 2022, 2023)
PROBS = tuple('probability_' + str(i) for i in range(1, 11))
KEYS = ('source_security_id', 'signal_date')
EXCLUDED = ('sec_gjP4xM', 'sec_zmYNnz')
QUOTE_META = ('trading_date', 'intrinio_security_id', 'frequency', 'intraperiod',
              'price_quality_status', 'raw_ohlc_bounds_valid', 'adjusted_ohlc_bounds_valid')
QUOTE_VALUES = (*QUOTE_META, 'adj_open', 'adj_high', 'adj_low', 'adj_close',
                'factor', 'split_ratio', 'dividend', 'open', 'high', 'low', 'close')
TARGET_META = (*KEYS, 'outcome_month', 'prediction_eligible', 'target_available')
PATH_META = (*TARGET_META, 'label_available_at', 'side', 'path_variant', 'target_family')
FORECAST_META = (*KEYS, 'intrinio_security_id', 'outcome_month', 'outcome_start',
    'outcome_end', 'prediction_eligible', 'prediction_status')
MAX_PHASE_BYTES = 1024 ** 3
SCHEMA = 'dmv2_unstopped_target_comparison_backtest_v1'


def need(value, message):
    if not bool(value):
        raise ValueError(message)


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def pin(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return {'path': str(path), 'sha256': digest.hexdigest(), 'size_bytes': path.stat().st_size}


def check(item):
    need(pin(item['path']) == {k: item[k] for k in ('path', 'sha256', 'size_bytes')},
         'Pinned source or artifact changed: ' + item['path'])
    return Path(item['path'])


def read_json(value):
    if isinstance(value, dict):
        return value
    return json.loads(Path(value).read_text())


def write_new(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    return pin(path)


def shift(month, offset):
    y, m = map(int, month.split('-')); need(1 <= m <= 12, 'Invalid calendar month')
    n = 12 * y + m - 1 + offset
    return f'{n // 12:04d}-{n % 12 + 1:02d}'


def _day(value, *, financial=True):
    need(type(value) is str and date.fromisoformat(value).isoformat() == value,
         'Canonical date required')
    need(not financial or '2010-01-01' <= value <= '2023-12-31',
         'Protected financial date')
    return value


def _records(value):
    if hasattr(value, 'to_pylist'):
        return value.to_pylist()
    if hasattr(value, 'to_dict'):
        return value.to_dict('records')
    return value


def build_payoffs(targets, signals, windows, *, arm, side):
    """Causal 120-month own-class payoffs; input is a projected derived panel.

    Endpoint rows carry original ``target_class``/``label_return``. Path rows
    carry independently audited classes and cash_return/utility; no reranking.
    Every ten-class cell in every completed calendar month is mandatory.
    """
    need(arm in ARMS and side in ('long', 'short'), 'Unknown payoff arm or side')
    by_window = {w['outcome_month']: {**w, 'label_available_at':
        w.get('label_available_at', w.get('exit_date'))} for w in _records(windows)}
    grouped = defaultdict(list); seen = set()
    original = defaultdict(int); eligible = defaultdict(int)
    for r in _records(targets):
        month = r['outcome_month']; _day(r['signal_date'], financial=False)
        need(type(r['prediction_eligible']) is bool and type(r['target_available']) is bool,
             'Explicit target admission and availability required')
        if arm != 'endpoint_matched_clock':
            if (r['side'], r['path_variant'], r['target_family']) != (
                    side, 'unstopped', 'executable_return' if arm == 'cash' else 'return_minus_drawdown'):
                continue
        key = (r['source_security_id'], r['signal_date'])
        need(key not in seen and type(key[0]) is str and bool(key[0].strip()), 'Duplicate or blank original target key')
        seen.add(key)
        original[month] += 1; eligible[month] += int(r['prediction_eligible'])
        if r['target_available']:
            need(r['prediction_eligible'] and type(r['target_class']) is int
                 and r['target_class'] in range(1, 11), 'Available target has no preserved class')
            field = 'label_return' if arm == 'endpoint_matched_clock' else ('cash_return' if arm == 'cash' else 'utility')
            value = r[field]
            need(isinstance(value, (int, float)) and not isinstance(value, bool)
                 and math.isfinite(value), 'Observed class payoff must be finite')
            grouped[(month, r['target_class'])].append(float(value))
    states = {}
    for signal in signals:
        _day(signal); last = shift(signal[:7], -1)
        months = [shift(last, n) for n in range(-119, 1)]
        rows = []
        for month in months:
            window = by_window.get(month)
            completed = bool(window and window['label_available_at'] <= signal)
            for c in range(1, 11):
                values = grouped[(month, c)] if completed else []
                status = ('available' if values else 'scheduled_endpoint_not_observed'
                    if window and not completed else 'missing_calendar_month'
                    if not window or not original[month] else 'empty_realized_class')
                rows.append({'outcome_month': month, 'class_id': c,
                    'label_available_at': window['label_available_at'] if window else None,
                    'original_member_count': original[month], 'prediction_eligible_count': eligible[month],
                    'member_count': len(values),
                    'mean_payoff': math.fsum(v / len(values) for v in values) if values else None,
                    'status': status})
        available = all(r['status'] == 'available' for r in rows)
        table = [{'class_id': c, 'payoff': math.fsum(r['mean_payoff'] / 120 for r in rows
            if r['class_id'] == c) if available else None,
            'status': 'available' if available else 'withheld_incomplete_120_month_class_grid'}
            for c in range(1, 11)]
        provenance = {'arm': arm, 'side': side, 'signal_date': signal,
            'label_observed_through': signal, 'first_outcome_month': months[0],
            'last_outcome_month': last, 'completed_calendar_months': 120,
            'vector_available': available, 'time_weighting': 'equal_month_class_means',
            'path_variant': 'unstopped', 'target_cost_bps': 0,
            'score_semantics': ('matched_clock_endpoint_expected_return_control_not_original_RET'
                if arm == 'endpoint_matched_clock' else 'expected_executable_cash_return'
                if arm == 'cash' else 'expected_path_utility_not_RET'),
            'endpoint_short_negates_long_score': arm == 'endpoint_matched_clock' and side == 'short'}
        state = {'table': table, 'monthly': rows, 'provenance': provenance}
        state['fingerprint'] = canonical(state)
        states[signal] = state
    return states


def score(probabilities, state, *, signal_date, arm, side):
    need(state['fingerprint'] == canonical({k: state[k] for k in ('table', 'monthly', 'provenance')}),
         'Payoff state fingerprint changed')
    p = state['provenance']
    need(p['signal_date'] == p['label_observed_through'] == signal_date
         and p['arm'] == arm and p['side'] == side, 'Wrong directional causal payoff state')
    months = [shift(shift(signal_date[:7], -1), n) for n in range(-119, 1)]
    need(p['completed_calendar_months'] == 120 and p['first_outcome_month'] == months[0]
         and p['last_outcome_month'] == months[-1]
         and [(r['outcome_month'], r['class_id']) for r in state['monthly']]
            == [(m, c) for m in months for c in range(1, 11)]
         and [r['class_id'] for r in state['table']] == list(range(1, 11)), 'Exact 120-by-ten payoff calendar required')
    ready = all(r['status'] == 'available' for r in state['monthly'])
    need(p['vector_available'] == ready, 'Partial payoff grid released')
    for row in state['monthly']:
        need(0 <= row['member_count'] <= row['prediction_eligible_count'] <= row['original_member_count'],
             'Payoff count exceeds original denominator')
        if row['status'] == 'available':
            need(row['member_count'] > 0 and row['label_available_at'] <= signal_date
                 and math.isfinite(row['mean_payoff']), 'Uncompleted or missing observed class payoff')
    for row in state['table']:
        need(row['status'] == ('available' if ready else 'withheld_incomplete_120_month_class_grid'),
             'Partially released payoff vector')
        expected = math.fsum(r['mean_payoff'] / 120 for r in state['monthly']
            if r['class_id'] == row['class_id']) if ready else None
        need(row['payoff'] == expected, 'Equal-calendar-month payoff formula changed')
    values = list(probabilities)
    need(len(values) == 10 and all(isinstance(v, (int, float)) and not isinstance(v, bool)
         and math.isfinite(v) and v >= 0 for v in values)
         and abs(math.fsum(values) - 1) <= 1e-8, 'Invalid explicit ten-class simplex')
    if not p['vector_available']:
        return None
    result = math.fsum(values[r['class_id'] - 1] * r['payoff'] for r in state['table'])
    return -result if arm == 'endpoint_matched_clock' and side == 'short' else result


def select_baskets(forecasts, payoffs, windows, sessions):
    """All original forecast rows retained; disjoint direction-specific ranks.

    No financial quote, future target field or future-membership table is read.
    ``forecasts`` is head -> dataframe, ``payoffs`` is (arm,side) -> signal state.
    """
    import numpy as np
    import pandas as pd
    from .walkforward_selection import BasketSelection, SAFE_COLUMNS
    need(set(forecasts) == set(HEADS), 'Exactly five forecast heads required')
    windows = _records(windows); signals = [w['signal_date'] for w in windows]
    need(len(signals) == 48 and len(set(signals)) == 48, 'All 48 independent signals required')
    calendar = tuple(sessions); need(tuple(sorted(set(calendar))) == calendar, 'Ordered independent sessions required')
    common = None; prepared = {}
    for head in HEADS:
        value = forecasts[head]
        need(value.columns.is_unique and set((*FORECAST_META, *PROBS)) <= set(value),
             'Forecast metadata/probability schema incomplete')
        # Project before conversion: future-label columns cannot choose rows.
        safe = [c for c in SAFE_COLUMNS if c in value]
        frame = value.loc[:, list(dict.fromkeys((*safe, *FORECAST_META, *PROBS)))].copy()
        need(not frame.duplicated(list(KEYS)).any() and set(frame.signal_date) == set(signals),
             'Original forecast keys missing or duplicated')
        need(frame.source_security_id.map(lambda x: type(x) is str and bool(x.strip())).all(),
             'Nonblank original stable source required')
        frame = frame.sort_values(list(KEYS), kind='stable').reset_index(drop=True)
        meta = frame[list(dict.fromkeys((*FORECAST_META, *safe)))]
        for r in meta.to_dict('records'):
            _day(r['signal_date']); need(type(r['prediction_eligible']) is bool, 'Explicit signal admission required')
            need(not r['prediction_eligible'] or type(r['intrinio_security_id']) is str
                 and bool(r['intrinio_security_id'].strip()), 'Eligible forecast has no native route')
            if 'paper_prediction_eligible' in r:
                need(r['paper_prediction_eligible'] == r['prediction_eligible'], 'Prediction admission changed paper cohort')
            if 'identity_eligible' in r:
                need(not r['prediction_eligible'] or r['identity_eligible'] is True, 'Ineligible identity reached predictions')
            w = next(w for w in windows if w['signal_date'] == r['signal_date'])
            need(all(r[k] == w[k] for k in ('outcome_month', 'outcome_start', 'outcome_end')),
                 'Forecast clock differs from independent calendar')
        if common is None:
            common = meta
        else:
            pd.testing.assert_frame_equal(common, meta, check_exact=True)
        eligible = frame.prediction_eligible.to_numpy(dtype=bool)
        need(not frame.loc[eligible].duplicated(['signal_date', 'intrinio_security_id']).any(),
             'Eligible native identity collision')
        for values in frame.loc[eligible, list(PROBS)].to_numpy(dtype=float):
            need(np.isfinite(values).all() and (values >= 0).all()
                 and abs(math.fsum(map(float, values)) - 1) <= 1e-8, 'Invalid original probability simplex')
        prepared[head] = frame
    output = {}
    for arm, basket in itertools.product(ARMS, BASKETS):
        lh, sh = ('endpoint', 'endpoint') if arm == 'endpoint_matched_clock' else (arm + '_long', arm + '_short')
        left, right = prepared[lh], prepared[sh]
        ledger = left[[c for c in left if c not in PROBS]].copy(deep=True)
        for side, frame in (('long', left), ('short', right)):
            for i in range(1, 11):
                ledger[f'probability_{side}_{i}'] = frame[f'probability_{i}']
            ledger[f'predicted_class_{side}'] = pd.array([None] * len(ledger), dtype='Int64')
            eligible = ledger.prediction_eligible.to_numpy(dtype=bool)
            ledger.loc[eligible, f'predicted_class_{side}'] = np.argmax(frame.loc[eligible, list(PROBS)].to_numpy(), axis=1) + 1
            ledger[f'expected_score_{side}'] = np.nan
            ledger[f'rank_{side}'] = pd.array([None] * len(ledger), dtype='Int64')
        ledger['execution_restricted'] = ledger.intrinio_security_id.isin(EXCLUDED)
        ledger['trade_eligible'] = ledger.prediction_eligible & ~ledger.execution_restricted
        ledger['selected'] = False; ledger['side'] = None; ledger['selection_score'] = np.nan
        ledger['selection_status'] = np.where(ledger.prediction_eligible, 'not_selected', 'withheld_signal_time_baseline')
        ledger['selected_names_per_side'] = 0; ledger['requested_top_k'] = 10
        ledger['effective_per_name_abs_weight'] = np.nan
        monthly, rebalances = [], []
        for w in windows:
            signal = w['signal_date']; idx = ledger.index[ledger.signal_date.eq(signal)]
            admitted = idx[ledger.loc[idx, 'prediction_eligible'].to_numpy()]
            trading = idx[ledger.loc[idx, 'trade_eligible'].to_numpy()]
            states = {side: payoffs[(arm, side)].get(signal) for side in ('long', 'short')}
            ready = all(state is not None and state['provenance']['vector_available'] for state in states.values())
            for side, frame in (('long', left), ('short', right)):
                if states[side] is not None:
                    # Authenticate the 1,200-cell state once per signal/side,
                    # then use the already validated original probability rows.
                    score([.1] * 10, states[side], signal_date=signal, arm=arm, side=side)
                    if states[side]['provenance']['vector_available']:
                        vector = states[side]['table']
                        sign = -1 if arm == 'endpoint_matched_clock' and side == 'short' else 1
                        probabilities = frame.loc[admitted, list(PROBS)].to_numpy(dtype=float)
                        ledger.loc[admitted, f'expected_score_{side}'] = [sign * math.fsum(
                            float(values[r['class_id'] - 1]) * r['payoff'] for r in vector)
                            for values in probabilities]
            if basket == 'score':
                candidates = {side: list(trading) for side in ('long', 'short')}
                rank_values = {side: ledger[f'expected_score_{side}'] for side in ('long', 'short')}
                n = min(10, len(trading) // 2)
            else:
                wanted = 1 if basket == 'best_mode' else 2
                wanted_short = 11 - wanted if arm == 'endpoint_matched_clock' else wanted
                candidates = {'long': [i for i in trading if ledger.at[i, 'predicted_class_long'] == wanted],
                    'short': [i for i in trading if ledger.at[i, 'predicted_class_short'] == wanted_short]}
                rank_values = {'long': ledger[f'probability_long_{wanted}'],
                    'short': ledger[f'probability_short_{wanted_short}']}
                # Own-direction modes can overlap. Recompute N until the fixed
                # long-first reservation leaves enough distinct short names.
                n = min(10, len(candidates['long']), len(candidates['short']))
            rankings = {side: sorted(candidates[side], key=lambda i: (-float(rank_values[side].at[i]),
                ledger.at[i, 'source_security_id'])) if (basket != 'score' or ready) else [] for side in ('long', 'short')}
            while n and len([i for i in rankings['short'] if i not in set(rankings['long'][:n])]) < n:
                n -= 1
            reserved = set(rankings['long'][:n]); shorts = [i for i in rankings['short'] if i not in reserved]
            if w['outcome_month'] == '2020-01':
                status = ('withheld_incomplete_120_month_class_grid' if basket == 'score'
                    else 'outside_common_comparison_period')
            elif basket == 'score' and not ready:
                status = 'withheld_incomplete_120_month_class_grid'
            else:
                status = 'available' if n else 'withheld_zero_distinct_directional_candidates'
            actual = n if status == 'available' else 0
            for side, ranked in (('long', rankings['long']), ('short', shorts)):
                ledger.loc[ranked, f'rank_{side}'] = range(1, len(ranked) + 1)
                chosen = ranked[:actual]
                ledger.loc[chosen, 'selected'] = True; ledger.loc[chosen, 'side'] = side
                ledger.loc[chosen, 'selection_score'] = rank_values[side].loc[chosen]
                ledger.loc[chosen, 'selection_status'] = 'selected'
            if not actual:
                ledger.loc[trading, 'selection_status'] = status
            restricted = idx[(ledger.loc[idx, 'prediction_eligible'] & ledger.loc[idx, 'execution_restricted']).to_numpy()]
            ledger.loc[restricted, 'selection_status'] = 'execution_excluded_retrospective_data_quality'
            ledger.loc[idx, 'selected_names_per_side'] = actual
            ledger.loc[idx, 'effective_per_name_abs_weight'] = .5 / actual if actual else np.nan
            rebalances.append({'signal_date': signal, 'execution_date': w['outcome_start'],
                'selection_status': status, 'selected_names_per_side': actual})
            monthly.append({'signal_date': signal, 'execution_date': w['outcome_start'], 'outcome_month': w['outcome_month'],
                'arm': arm, 'basket': basket, 'original_rows': len(idx), 'prediction_eligible': len(admitted),
                'trade_eligible': len(trading), 'long_candidates': len(candidates['long']),
                'short_candidates': len(candidates['short']), 'long_selected': actual, 'short_selected': actual,
                'selected_names_per_side': actual, 'requested_top_k': 10, 'status': status,
                'payoff_long_available': bool(states['long'] and states['long']['provenance']['vector_available']),
                'payoff_short_available': bool(states['short'] and states['short']['provenance']['vector_available']),
                'overlapping_directional_candidates': len(set(candidates['long']) & set(candidates['short'])),
                **{f'{side}_rank_score_flat': bool(ranked and all(float(rank_values[side].at[i])
                    == float(rank_values[side].at[ranked[0]]) for i in ranked))
                    for side, ranked in (('long', rankings['long']), ('short', shorts))},
                **{f'{side}_cutoff_score': float(rank_values[side].at[ranked[actual - 1]])
                    if actual else None for side, ranked in (('long', rankings['long']), ('short', shorts))},
                'future_quote_or_target_availability_used': False})
        rebalances = pd.DataFrame(rebalances)
        selected = ledger.loc[ledger.selected].merge(rebalances[['signal_date', 'execution_date']], on='signal_date', validate='many_to_one')
        selected['intended_weight'] = np.where(selected.side.eq('long'), selected.effective_per_name_abs_weight,
            -selected.effective_per_name_abs_weight)
        selected = selected.sort_values(['signal_date', 'side', 'source_security_id'], kind='stable').reset_index(drop=True)
        output[(arm, basket)] = BasketSelection(ledger, selected, rebalances, pd.DataFrame(monthly))
    return output


def projection(item, columns, *, maximum_rows):
    """Parquet footer lengths/schema only; no column statistics or values."""
    import pyarrow.parquet as pq
    source = pq.ParquetFile(check(item)); names = source.schema_arrow.names
    need(len(names) == len(set(names)) and set(columns) <= set(names)
         and source.metadata.num_rows <= maximum_rows, 'Projected source schema/row ceiling changed')
    sizes = []
    for i in range(source.metadata.num_row_groups):
        row = source.metadata.row_group(i); chunks = {}
        for j in range(row.num_columns):
            field = row.column(j)
            if field.path_in_schema in columns:
                need(field.path_in_schema not in chunks, 'Duplicate projected Parquet leaf')
                chunks[field.path_in_schema] = field.total_compressed_size
        need(set(chunks) == set(columns), 'Projected physical fields incomplete')
        sizes.append(sum(chunks.values()))
    return {'pin': item, 'columns': list(columns), 'rows': source.metadata.num_rows,
        'row_group_compressed_bytes': sizes, 'projected_compressed_bytes': sum(sizes)}


def _projection_pair(item, metadata, values, maximum_rows):
    return {'metadata': projection(item, metadata, maximum_rows=maximum_rows),
        'values': projection(item, tuple(dict.fromkeys((*metadata, *values))), maximum_rows=maximum_rows)}


def _protocol(path):
    protocol = read_json(path)
    need(protocol['schema'] == 'dmv2_unstopped_target_comparison_v1'
         and protocol['protocol_id'] == 'dmv2-rf-expanded-unstopped-three-targets-20261022-v2'
         and protocol['status'] == 'SEALED_BEFORE_FINANCIAL_READ'
         and protocol['backtest']['arms'] == list(ARMS)
         and protocol['backtest']['baskets'] == list(BASKETS)
         and protocol['backtest']['scenarios'] == 72
         and protocol['backtest']['start'] == '2020-02-03'
         and protocol['backtest']['terminal'] == '2023-12-29'
         and protocol['backtest']['cost_bps'] == list(FEES)
         and protocol['backtest']['asset_stop_fractions'] == [None, .1]
         and protocol['payoffs']['window_end'] == 'signal_calendar_month_minus_one'
         and protocol['interpretation']['financial_2024_plus_read'] is False,
         'Frozen common-clock three-target execution authority required')
    return protocol


def plan_backtest(project, model_plan, protocol, output):
    """Seal exact existing source projections and bounded future forecast reads.

    May run before estimator completion. Future files are not opened or used;
    their footer projections are sealed in a separate receipt after completion,
    before any numerical forecast read. Their reserved ceiling is charged here.
    """
    project = Path(project).resolve(); model = read_json(model_plan); policy = _protocol(protocol)
    protocol_pin = model['protocol_pin'] if isinstance(protocol, dict) else pin(protocol)
    need(read_json(check(protocol_pin)) == policy, 'Protocol dictionary differs from pinned frozen source')
    need(project == Path(policy['project_root']).resolve() and model['project_root'] == str(project),
         'Model/backtest project mismatch')
    need(model['heads'] == list(HEADS) and model['expected_fits'] == 20
         and model['protocol_pin'] == protocol_pin, 'Exactly twenty frozen fits required')
    for key in ('catalog_pin', 'target_audit_pin', 'target_plan_pin', 'target_verification_pin'):
        check(model[key])
    need(model['catalog_pin']['sha256'] == policy['corrected_catalog_sha256']
         and model['target_audit_pin']['sha256'] == policy['full_audit_sha256'],
         'Only independently audited corrected catalog/target package allowed')
    old_source_path = project / 'outputs/corrected_refit/2020-2023-20261021-v1/backtest_source_plan.json'
    old = read_json(old_source_path)
    phase = read_json(check(old['execution_phase_pin']))
    manifest = read_json(check(old['execution_manifest_pin']))
    verification = read_json(check(old['execution_verification_pin']))
    need(verification['status'] == 'PASS' and len(phase['event_registry']['events']) == 18
         and len(phase['historical_retirements']) == 11 and len(manifest['quote_amendments']) == 40
         and len(phase['identity_handover_registry']['handovers']) == 1,
         'Preserved execution repair/action/identity evidence required')
    for event in phase['event_registry']['events']:
        need('2020-02-03' <= _day(event['effective_date']) <= '2023-12-29',
             'Current event outside common clock; do not silently reclassify it')
    from .walkforward_clock import walkforward_clock
    sessions, windows_frame, clock = walkforward_clock()
    windows = windows_frame.to_dict('records')
    reduced = [s for s in sessions if policy['backtest']['start'] <= s <= policy['backtest']['terminal']]
    endpoint, path_sources = [], []
    for year, annual in model['annuals'].items():
        need(str(annual['year']) == year and 2010 <= int(year) <= 2023, 'Target annual outside scope')
        item = annual['endpoint_metadata']['pin']
        endpoint.append({'year': int(year), **_projection_pair(item, TARGET_META,
            ('target_class', 'label_return'), 12000)})
        for month in annual['path_months']:
            item = month['metadata']['pin']
            pair = _projection_pair(item, PATH_META, ('target_class', 'cash_return', 'utility'), 16000)
            path_sources.append({'outcome_month': month['outcome_month'], **pair})
    need(len(endpoint) == 14 and len(path_sources) == 168, 'All target months required')
    daily = []
    for item in old['daily']:
        daily.append({'year': item['year'], **_projection_pair(item['metadata']['pin'],
            QUOTE_META, QUOTE_VALUES, 300000)})
    need({a['year'] for a in daily} == set(YEARS), 'Four execution-price annuals required')
    repairs = _projection_pair(old['execution_repairs']['metadata']['pin'], QUOTE_META, QUOTE_VALUES, 40)
    forecast_limit = 128 * 1024 ** 2
    target_cost = 2 * sum(sum(x[k]['projected_compressed_bytes'] for k in ('metadata', 'values')) for x in endpoint)
    target_cost += 4 * sum(sum(x[k]['projected_compressed_bytes'] for k in ('metadata', 'values')) for x in path_sources)
    price_cost = sum(sum(x[k]['projected_compressed_bytes'] for k in ('metadata', 'values')) for x in daily)
    price_cost += sum(repairs[k]['projected_compressed_bytes'] for k in ('metadata', 'values'))
    need(target_cost + price_cost + forecast_limit <= MAX_PHASE_BYTES, 'Backtest source read reservation exceeds 1GiB')
    root = Path(model['output_directory']).resolve()
    model_plan_path = root / 'read_plan.json'
    need(read_json(model_plan_path) == model, 'Model-plan dictionary differs from its immutable saved source')
    forecast_paths = [{'year': year, 'head': head,
        'path': str(root / 'folds' / str(year) / 'heads' / head / 'predictions.parquet'),
        'receipt_path': str(root / 'folds' / str(year) / 'heads' / head / 'PREDICT_COMPLETE.json')}
        for year, head in itertools.product(YEARS, HEADS)]
    out = Path(output).resolve()
    if out.suffix == '.json':
        plan_path, run_directory = out, out.parent / 'backtest'
    else:
        plan_path, run_directory = out.parent / 'backtest_plan.json', out
    body = {'schema': SCHEMA, 'status': 'SEALED_BEFORE_FINANCIAL_READ',
        'project_root': str(project), 'output_directory': str(run_directory),
        'model_output_directory': str(root), 'model_plan_sha256': model['plan_sha256'],
        'model_plan_pin': pin(model_plan_path),
        'protocol_pin': protocol_pin, 'worker_pin': pin(__file__),
        'model_source_pins': {key: model[key] for key in ('catalog_pin', 'target_audit_pin', 'target_plan_pin', 'target_verification_pin')},
        'old_source_plan_pin': pin(old_source_path),
        'execution_pins': {k: old[k] for k in ('execution_phase_pin', 'execution_manifest_pin', 'execution_verification_pin')},
        'execution_evidence_pins': old['execution_evidence_pins'],
        'endpoint': endpoint, 'path_targets': path_sources, 'daily': daily, 'repairs': repairs,
        'forecasts': forecast_paths, 'forecast_reserved_bytes': forecast_limit,
        'source_projected_bytes': target_cost + price_cost,
        'source_projection_bytes': price_cost,
        'derived_read_bytes': target_cost + forecast_limit,
        'projected_read_bytes': target_cost + price_cost + forecast_limit,
        'target_read_passes': {'endpoint': 2, 'path': 4},
        'windows': windows, 'selection_sessions': clock['selection_sessions'], 'sessions': reduced,
        'expected_sessions': len(reduced), 'forecast_months': 48, 'economic_months': 47,
        'scenarios': 72, 'new_fits': 0, 'financial_2024_plus_read': False,
        'limits': {'rss_stop_bytes': 8 * 1024 ** 3, 'active_seconds': 3600,
            'projection_bytes': MAX_PHASE_BYTES},
        'helper_pins': {name: pin(Path(__file__).parent / name) for name in (
            'walkforward_accounting_v2.py', 'walkforward_accounting.py', 'backtest_accounting.py',
            'walkforward_preflight.py', 'ret_repair_backtest.py', 'daily_backtest.py',
            'walkforward_backtest.py', 'walkforward_clock.py', 'action_loading.py', 'backtest_loading.py')}}
    body['plan_sha256'] = canonical(body)
    write_new(plan_path, body)
    return body


def verify_plan(plan):
    need(plan['schema'] == SCHEMA and plan['status'] == 'SEALED_BEFORE_FINANCIAL_READ'
         and plan['plan_sha256'] == canonical({k: v for k, v in plan.items() if k != 'plan_sha256'}),
         'Backtest plan identity changed')
    need(plan['scenarios'] == 72 and plan['economic_months'] == 47 and plan['forecast_months'] == 48
         and plan['projected_read_bytes'] <= MAX_PHASE_BYTES and plan['new_fits'] == 0
         and plan['financial_2024_plus_read'] is False, 'Backtest scope changed')
    need(plan['worker_pin'] == pin(__file__), 'Backtest worker changed after freeze')
    for item in (plan['protocol_pin'], plan['old_source_plan_pin'], *plan['model_source_pins'].values(),
                 *plan['execution_pins'].values(), *plan['execution_evidence_pins'], *plan['helper_pins'].values()):
        check(item)
    if plan['model_plan_pin'] is not None:
        check(plan['model_plan_pin'])
    _protocol(check(plan['protocol_pin']))
    return plan


def _read_projection(spec):
    import pyarrow.parquet as pq
    table = pq.ParquetFile(check(spec['pin'])).read(columns=spec['columns'])
    need(table.num_rows == spec['rows'], 'Projected table row count changed')
    return table


def _target_rows(plan, *, arm, side, resource_check):
    import pyarrow.compute as pc
    for pair in plan['endpoint' if arm == 'endpoint_matched_clock' else 'path_targets']:
        resource_check()
        metadata = _read_projection(pair['metadata'])
        rows = metadata.to_pylist()
        for r in rows:
            _day(r['signal_date'], financial=False)
            need('2010-01' <= r['outcome_month'] <= '2023-12', 'Payoff metadata outside historical years')
            need(type(r['prediction_eligible']) is bool and type(r['target_available']) is bool,
                 'Missing source target state')
            if arm != 'endpoint_matched_clock':
                need(r['path_variant'] in VARIANTS and r['side'] in ('long', 'short')
                     and r['target_family'] in ('executable_return', 'return_minus_drawdown'), 'Unknown path metadata')
                _day(r['label_available_at'], financial=False)
                need(r['label_available_at'] < '2024-01-01' or not r['target_available'],
                     'Protected next-open label cannot be observed')
        values = _read_projection(pair['values'])
        need(values.select(metadata.column_names).equals(metadata), 'Target metadata changed before numeric conversion')
        if arm != 'endpoint_matched_clock':
            values = values.filter(pc.and_(pc.and_(pc.equal(values['side'], side),
                pc.equal(values['path_variant'], 'unstopped')), pc.equal(values['target_family'],
                    'executable_return' if arm == 'cash' else 'return_minus_drawdown')))
        yield from values.to_pylist()


def _save_table(directory, name, frame, artifacts):
    import pyarrow as pa
    import pyarrow.parquet as pq
    path = directory / name; need(not path.exists(), 'Preserve every prior numerical artifact')
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(frame, list):
        table = pa.Table.from_pylist(frame)
    elif isinstance(frame, pa.Table):
        table = frame
    else:
        table = pa.Table.from_pandas(frame, preserve_index=False)
    pq.write_table(table, path, compression='zstd')
    artifacts[name] = pin(path)


def _forecast_plan(plan, directory):
    import pyarrow.parquet as pq
    from .walkforward_selection import SAFE_COLUMNS
    result = []; total = 0
    for entry in plan['forecasts']:
        receipt_pin = pin(entry['receipt_path']); receipt = read_json(check(receipt_pin))
        need(receipt['status'] == 'PREDICT_COMPLETE' and receipt['plan_sha256'] == plan['model_plan_sha256']
             and receipt['year'] == entry['year'] and receipt['head'] == entry['head'], 'All twenty matching fits/forecasts must finish first')
        item = pin(entry['path'])
        need(receipt['artifacts']['predictions'] == item, 'Forecast differs from completed model receipt')
        names = pq.ParquetFile(check(item)).schema_arrow.names
        metadata = tuple(dict.fromkeys((*FORECAST_META, *(n for n in SAFE_COLUMNS if n in names))))
        pair = _projection_pair(item, metadata, PROBS, 12000)
        total += sum(pair[k]['projected_compressed_bytes'] for k in ('metadata', 'values'))
        result.append({**entry, **pair, 'receipt_pin': receipt_pin})
    need(len(result) == 20 and total <= plan['forecast_reserved_bytes'], 'Future forecast reservation exceeded')
    write_new(directory / 'forecast_read_plan.json', {'status': 'SEALED_BEFORE_FINANCIAL_READ',
        'parent_plan_sha256': plan['plan_sha256'], 'projected_bytes': total, 'forecasts': result})
    return result


def _load_forecasts(entries, plan, resource_check):
    import pandas as pd
    parts = defaultdict(list); by_signal = {w['signal_date']: w for w in plan['windows']}
    for pair in entries:
        resource_check(); metadata = _read_projection(pair['metadata'])
        for row in metadata.to_pylist():
            _day(row['signal_date']); w = by_signal.get(row['signal_date'])
            need(w is not None and w['outcome_month'].startswith(str(pair['year']) + '-')
                 and all(row[k] == w[k] for k in ('outcome_month', 'outcome_start', 'outcome_end'))
                 and type(row['prediction_eligible']) is bool, 'Forecast metadata outside sealed clock')
        values = _read_projection(pair['values'])
        need(values.select(metadata.column_names).equals(metadata), 'Forecast metadata changed before probabilities')
        parts[pair['head']].append(values.to_pandas())
    return {head: pd.concat(parts[head], ignore_index=True) for head in HEADS}


def _load_prices(plan, resource_check):
    import pandas as pd
    import pyarrow as pa
    from .ret_repair_backtest import overlay_quotes
    sessions = set(plan['sessions']); original = []; excluded_counts = {}
    for pair in plan['daily']:
        resource_check(); metadata = _read_projection(pair['metadata'])
        rows = metadata.to_pylist(); positions = []; seen = set()
        for i, row in enumerate(rows):
            d = _day(row['trading_date'])
            native = row['intrinio_security_id']
            need(type(native) is str and bool(native.strip()) and (d, native) not in seen,
                 'Duplicate or blank execution quote metadata key')
            seen.add((d, native))
            if d.startswith(str(pair['year']) + '-') and d in sessions:
                positions.append(i)
        excluded_counts[str(pair['year'])] = len(rows) - len(positions)
        values = _read_projection(pair['values'])
        need(values.select(metadata.column_names).equals(metadata), 'Execution quote metadata changed')
        # Whole approved Arrow blocks may decode, but the January prefix and
        # annual anchor rows never become Python financial observations.
        original.append(values.take(pa.array(positions, type=pa.int64())).to_pandas())
    repairs = plan['repairs']; meta = _read_projection(repairs['metadata'])
    for row in meta.to_pylist():
        need(_day(row['trading_date']) in sessions, 'Execution overlay outside common financial clock')
    values = _read_projection(repairs['values'])
    need(values.select(meta.column_names).equals(meta), 'Execution overlay metadata changed')
    phase = read_json(check(plan['execution_pins']['execution_phase_pin']))
    manifest = read_json(check(plan['execution_pins']['execution_manifest_pin']))
    source = pd.concat(original, ignore_index=True)
    prices = overlay_quotes(source, values.to_pandas(), manifest['quote_amendments'])
    receipt = {'execution_only': True, 'quote_amendments': len(manifest['quote_amendments']),
        'original_rows': len(source), 'amended_rows': len(prices),
        'excluded_source_rows_metadata_only': excluded_counts,
        'manifest_pin': plan['execution_pins']['execution_manifest_pin'],
        'new_learning_rows': 0}
    return prices, receipt, phase


def run_backtest(plan, resource_check=None):
    """Explicit 72-path replay. No models, estimators or learning bars are read."""
    import pandas as pd
    import pyarrow as pa
    from .walkforward_accounting_v2 import run_walkforward_accounting_v2
    from .backtest_accounting import DailyAccountingPolicy
    from .walkforward_preflight import preflight_selected_horizons
    from .daily_backtest import summarize_daily
    from .walkforward_backtest import yearly_breakdown
    import resource
    import sys
    plan = read_json(plan); verify_plan(plan)
    proof_path = Path(plan['model_output_directory']) / 'verification/models.json'
    proof_pin = pin(proof_path); proof = read_json(check(proof_pin))
    need(proof['status'] == 'PASS' and proof['phase'] == 'models'
         and proof['verified_estimator_fits'] == 20 and proof['verified_shared_preprocessors'] == 4
         and proof['new_estimator_fits'] == 0 and proof['financial_2024_plus_values_read'] is False
         and proof['model_plan_pin'] == plan['model_plan_pin'],
         'Independent twenty-fit/four-preprocessor PASS required before backtesting')
    directory = Path(plan['output_directory']); need(not directory.exists(), 'Preserve prior backtest attempt/checkpoints')
    directory.mkdir(parents=True)
    write_new(directory / 'STARTED.json', {'plan_sha256': plan['plan_sha256'], 'status': 'ATTEMPT_CONSUMED', 'fits': 0})
    start = time.monotonic(); peak = 0
    def resources():
        nonlocal peak
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak = max(peak, int(value if sys.platform == 'darwin' else value * 1024))
        need(peak <= plan['limits']['rss_stop_bytes'] and time.monotonic() - start <= plan['limits']['active_seconds'],
             'Backtest resource ceiling exceeded; preserve consumed attempt')
        if resource_check is not None:
            resource_check()
    artifacts = {}; resources(); forecasts_plan = _forecast_plan(plan, directory)
    full = read_json(check(plan['model_source_pins']['target_plan_pin']))
    target_windows = [w for a in full['annuals'] for w in a['windows']]
    payoffs = {}; signals = [w['signal_date'] for w in plan['windows']]
    for arm, side in itertools.product(ARMS, ('long', 'short')):
        state = build_payoffs(_target_rows(plan, arm=arm, side=side, resource_check=resources),
            signals, target_windows, arm=arm, side=side)
        payoffs[(arm, side)] = state
        for signal, entry in state.items():
            base = f'payoffs/{arm}/{signal}/{side}/'
            _save_table(directory, base + 'monthly.parquet', entry['monthly'], artifacts)
            _save_table(directory, base + 'payoffs.parquet', entry['table'], artifacts)
            artifacts[base + 'state.json'] = write_new(directory / (base + 'state.json'),
                {'provenance': entry['provenance'], 'fingerprint': entry['fingerprint']})
        resources()
    forecasts = _load_forecasts(forecasts_plan, plan, resources)
    selections = select_baskets(forecasts, payoffs, plan['windows'], plan['selection_sessions'])
    del forecasts, payoffs
    for (arm, basket), selected in selections.items():
        for name, frame in (('original_ledger', selected.ledger), ('selected', selected.selections),
                ('rebalances', selected.rebalances), ('monthly', selected.monthly)):
            _save_table(directory, f'selection/{arm}/{basket}/{name}.parquet', frame, artifacts)
    prices, quote_receipt, phase = _load_prices(plan, resources)
    artifacts['quote_overlay.json'] = write_new(directory / 'quote_overlay.json', quote_receipt)
    comparison, annuals, coverage = [], [], []
    for (arm, basket), selected in selections.items():
        resources(); cut = selected.rebalances.execution_date.ge('2020-02-03')
        rebalances = selected.rebalances.loc[cut].copy()
        chosen = selected.selections.loc[selected.selections.execution_date.ge('2020-02-03')].copy()
        coverage.append(selected.monthly)
        preflight = preflight_selected_horizons(chosen, rebalances, prices[list(QUOTE_META)], plan['sessions'],
            events=phase['event_registry']['events'], historical_retirements=phase['historical_retirements'])
        artifacts[f'preflight/{arm}/{basket}/summary.json'] = write_new(directory / f'preflight/{arm}/{basket}/summary.json', preflight['summary'])
        for name in ('missing_quotes', 'source_quality', 'retired_orders', 'action_requirements'):
            _save_table(directory, f'preflight/{arm}/{basket}/{name}.parquet', preflight[name], artifacts)
        for variant, fee in itertools.product(VARIANTS, FEES):
            resources()
            value = run_walkforward_accounting_v2(chosen, prices, plan['sessions'], rebalances=rebalances,
                terminal_date='2023-12-29', events=phase['event_registry']['events'],
                historical_retirements=phase['historical_retirements'],
                identity_handovers=phase['identity_handover_registry']['handovers'],
                policy=DailyAccountingPolicy(cost_bps=fee, borrow_annual_bps=0,
                    required_names_per_side=None, asset_stop_fraction=VARIANTS[variant]))
            row, monthly = summarize_daily(value, arm=arm, basket=basket, variant=variant,
                cost_bps=fee, expected_sessions=plan['expected_sessions'])
            row.update(outcome_start='2020-02-03', outcome_end='2023-12-29', outcome_months=47)
            annual = yearly_breakdown(value, arm=arm, basket=basket, variant=variant, cost_bps=fee)
            base = f'replay/{arm}/{basket}/{variant}/cost_{fee:02d}bps/'
            for name, frame in value.tables.items():
                _save_table(directory, base + name + '.parquet', frame, artifacts)
            _save_table(directory, base + 'monthly.parquet', monthly, artifacts)
            _save_table(directory, base + 'annual.parquet', annual, artifacts)
            comparison.append(row); annuals.append(annual)
            write_new(directory / f'checkpoints/{arm}/{basket}/{variant}/cost_{fee:02d}bps.json',
                {'status': 'REPLAY_COMPLETE_PENDING_INDEPENDENT_VERIFICATION', 'plan_sha256': plan['plan_sha256'],
                    'scenario': [arm, basket, variant, fee], 'artifacts': {k: v for k, v in artifacts.items() if k.startswith(base)}})
            print(json.dumps({'phase': 'target_backtest', 'completed_scenarios': len(comparison), 'expected_scenarios': 72}), flush=True)
    _save_table(directory, 'comparison.parquet', pd.DataFrame(comparison), artifacts)
    _save_table(directory, 'yearly.parquet', pd.concat(annuals, ignore_index=True), artifacts)
    _save_table(directory, 'selection_coverage.parquet', pd.concat(coverage, ignore_index=True), artifacts)
    resources(); verify_plan(plan); check(proof_pin)
    result = {'schema': SCHEMA, 'status': 'COMPLETE_PENDING_INDEPENDENT_VERIFICATION',
        'plan_sha256': plan['plan_sha256'], 'artifacts': artifacts, 'scenarios': 72,
        'model_verification_pin': proof_pin,
        'sessions': plan['expected_sessions'], 'economic_months': 47, 'forecast_months': 48,
        'complete_proxy_paths': sum(bool(r['complete_proxy_path']) for r in comparison),
        'withheld_paths': sum(not bool(r['complete_proxy_path']) for r in comparison),
        'peak_rss_bytes': peak, 'active_seconds': time.monotonic() - start,
        'projected_read_bytes_reserved': plan['projected_read_bytes'],
        'fits': 0, 'new_learning_rows': 0, 'financial_2024_plus_read': False,
        'source_identity_or_investability_certified': False}
    write_new(directory / 'audit.json', result)
    return result


def report(plan):
    """Small hash-checked saved reports only; no model or source-price load."""
    plan = read_json(plan); verify_plan(plan)
    directory = Path(plan['output_directory']); result = read_json(directory / 'audit.json')
    need(result['plan_sha256'] == plan['plan_sha256'] and result['scenarios'] == 72
         and result['economic_months'] == 47, 'Saved backtest report scope changed')
    for name in ('comparison.parquet', 'yearly.parquet', 'selection_coverage.parquet'):
        check(result['artifacts'][name])
    return {k: result[k] for k in ('status', 'scenarios', 'sessions', 'economic_months',
        'forecast_months', 'complete_proxy_paths', 'withheld_paths', 'peak_rss_bytes', 'active_seconds')}
