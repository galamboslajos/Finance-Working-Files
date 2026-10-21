"""Independent, import-safe checks for the corrected-input backtest.

This module imports no production research code and reads no files.  Financial
values are accepted only by an explicitly guarded caller after metadata checks.
The independently reviewed cash/units equations live in a separately pinned
historical verification script, rather than in the production backtester.
"""
from collections import defaultdict
from datetime import date, datetime
import hashlib
import json
import math

YEARS = (2020, 2021, 2022, 2023)
ARMS = ('rf_paper15', 'xgb_paper15', 'rf_extended', 'xgb_extended', 'rf_tuned_extended')
BASKETS = ('ret', 'class_1_10', 'class_2_9')
VARIANTS = ('unstopped', 'asset_stop_10pct')
FEES = (0, 5, 10, 25)
KEYS = ('source_security_id', 'signal_date')
PROBABILITIES = tuple('probability_' + str(i) for i in range(1, 11))
EXCLUDED = {'sec_gjP4xM', 'sec_zmYNnz'}
BK_SOURCE = 'siblis_ticker_company_1c21a7b293fd137c59c9'
BK_NATIVE = 'sec_zq1J05'
XOM_SOURCE = 'siblis_isin_b284bb7497616f5afba0'
XOM_NATIVE = 'dmv2_sip:siblis_isin_b284bb7497616f5afba0:all'
ISSUE_COLUMNS = ('signal_date', 'execution_date', 'scheduled_exit_date', 'source_security_id',
                 'intrinio_security_id', 'side', 'date', 'role', 'reason')
PREFLIGHT_SUMMARY_COUNTS = (
    'selected_episode_rows', 'distinct_maximum_horizon_quote_keys', 'missing_key_episode_rows',
    'quality_flag_episode_rows', 'unique_missing_keys', 'unique_quality_flag_keys',
    'verified_current_action_definitions', 'historical_retirement_definitions',
    'retired_new_orders', 'withheld_selection_months')
PREFLIGHT_SUMMARY_FLAGS = (
    'ready_for_accounting', 'source_metadata_only', 'prices_or_outcomes_read',
    'selection_changed', 'unstopped_maximum_horizons_reviewed_for_both_stop_variants')
PREFLIGHT_SUMMARY_SCHEMA = {'arm': 'string', 'basket': 'string',
    **{k: 'int64' for k in PREFLIGHT_SUMMARY_COUNTS},
    **{k: 'bool' for k in PREFLIGHT_SUMMARY_FLAGS}}


def need(value, message):
    if not bool(value):
        raise ValueError(message)


def canonical(value):
    return 'sha256:' + hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def day(value, *, lower='2010-01-01', upper='2023-12-31'):
    need(type(value) is str and date.fromisoformat(value).isoformat() == value
         and lower <= value <= upper, 'Noncanonical or protected financial date')
    return value


def month_step(month, offset):
    need(type(month) is str and len(month) == 7 and month[4] == '-', 'Canonical month required')
    y, m = map(int, month.split('-'))
    need(1 <= m <= 12 and f'{y:04d}-{m:02d}' == month, 'Canonical month required')
    value = y * 12 + m - 1 + offset
    return f'{value // 12:04d}-{value % 12 + 1:02d}'


def trailing_months(signal):
    day(signal, lower='2019-12-01', upper='2023-11-30')
    return [month_step(signal[:7], i) for i in range(-119, 1)]


def close(actual, expected, message, *, tolerance=3e-13):
    if expected is None:
        need(actual is None, message + ': missingness differs')
    else:
        need(type(actual) in (int, float) and type(expected) in (int, float)
             and math.isfinite(actual) and math.isfinite(expected)
             and abs(actual - expected) <= tolerance * max(1., abs(expected)), message)


def forecast_metadata(rows, year, windows):
    """No outcome availability or future quote field is accepted as admission."""
    need(year in YEARS and len(rows) <= 12000, 'Approved annual forecast envelope required')
    expected = {r['signal_date']: r for r in windows if r['outcome_month'].startswith(str(year) + '-')}
    need(len(expected) == 12, 'Independent twelve-month clock required')
    keys, natives = set(), set()
    cutoff = {2020: '2019-12-31', 2021: '2020-12-31', 2022: '2021-12-31', 2023: '2022-12-30'}[year]
    for r in rows:
        signal = day(r['signal_date'], lower='2019-12-01', upper='2023-11-30')
        sid = r['source_security_id']
        need(type(sid) is str and sid and sid.strip() == sid, 'Nonblank original stable source required')
        key = sid, signal
        need(key not in keys and signal in expected, 'Duplicate or unknown forecast key')
        keys.add(key)
        need(all(r[k] == expected[signal][k] for k in ('outcome_month', 'outcome_start', 'outcome_end')),
             'Forecast outcome clock differs from independent sessions')
        need(all(type(r[k]) is bool for k in ('identity_eligible', 'paper_prediction_eligible', 'prediction_eligible')),
             'Explicit original admission booleans required')
        eligible = r['prediction_eligible']
        need(eligible == r['paper_prediction_eligible'] and (not eligible or r['identity_eligible']),
             'Future outcome or identity changed prediction admission')
        need(r['prediction_status'] == ('predicted' if eligible else 'withheld_signal_time_baseline'),
             'Prediction status changed original admission')
        need(r['fit_year'] == year and r['training_outcome_start'] == f'{year-10}-01-01'
             and r['training_outcome_end'] == f'{year-1}-12-31'
             and r['training_label_observed_through'] == cutoff, 'Rolling ten-year lineage changed')
        if eligible:
            native = r['intrinio_security_id']
            need(type(native) is str and native and native.strip() == native
                 and (signal, native) not in natives, 'Missing or colliding dated execution key')
            natives.add((signal, native))
            if sid == BK_SOURCE: need(native == BK_NATIVE, 'BK route changed')
            if sid == XOM_SOURCE: need(native == XOM_NATIVE, 'Typed SIP XOM key changed or was promoted')
    need({key[1] for key in keys} == set(expected), 'Missing an original forecast month')
    return [(r['source_security_id'], r['signal_date'], r['intrinio_security_id'],
             r['identity_eligible'], r['paper_prediction_eligible'], r['prediction_eligible']) for r in rows]


def payoff_metadata(rows, table_rows, state, signal):
    """Validate all saved RET clocks/grid/provenance before numerical Python."""
    expected_months = trailing_months(signal)
    need([(r['outcome_month'], r['class_id']) for r in rows]
         == [(m, c) for m in expected_months for c in range(1, 11)],
         'RET must retain the exact ordered 120-calendar-month by ten-class grid')
    need([r['class_id'] for r in table_rows] == list(range(1, 11)), 'RET payoff class order changed')
    p = state['provenance']
    need(p['signal_date'] == p['label_observed_through'] == signal
         and p['first_outcome_month'] == expected_months[0] and p['last_outcome_month'] == expected_months[-1]
         and p['expected_calendar_months'] == 120 and p['expected_classes'] == list(range(1, 11))
         and p['expected_month_class_cells'] == 1200, 'RET payoff causal scope changed')
    need(p['monthly_class_mean'] == 'equal_weight_observed_realized_class_returns'
         and p['payoff_time_weighting'] == 'equal_calendar_month_weights'
         and p['missing_month_or_class_policy'] == 'withhold_entire_payoff_vector'
         and p['uses_predicted_classes'] is False and p['financial_rows_after_cutoff_consumed'] is False,
         'RET uses a different payoff history or weighting')
    need(p['return_basis'] == 'provisional_provider_adjusted_endpoint_ratio'
         and all(p[k] is False for k in ('economic_return_certified', 'knowledge_time_certified',
             'same_close_operational_availability_certified', 'model_ready', 'estimator_fit_performed')),
         'RET certification/provisional contract changed')
    completion = p['source_completion_windows']
    need(len(completion) == 120 and [r['outcome_month'] for r in completion] == expected_months,
         'RET source completion windows changed')
    for i, month in enumerate(expected_months):
        window = completion[i]
        day(window['outcome_end'])
        need(window['outcome_end'][:7] == month and window['outcome_end'] <= signal,
             'Uncompleted source return entered a RET vector')
        for r in rows[i*10:(i+1)*10]:
            need(r['outcome_end'] == window['outcome_end'], 'RET class row completion clock changed')
            for k in ('original_member_count', 'prediction_eligible_count', 'member_count'):
                need(type(r[k]) is int and r[k] >= 0, 'RET coverage count is unknown/negative')
            need(r['member_count'] <= r['prediction_eligible_count'] <= r['original_member_count'],
                 'RET coverage exceeds original historical denominator')
    available = all(r['status'] == 'available' for r in rows)
    need(p['vector_available'] is available
         and p['available_month_class_cells'] == sum(r['status'] == 'available' for r in rows),
         'RET partially released or silently imputed a missing cell')
    expected_status = 'available' if available else 'withheld_incomplete_120_month_class_grid'
    need(all(r['status'] == expected_status for r in table_rows), 'RET whole-vector gate changed')
    return available


def payoff_values(rows, table_rows, state, signal):
    """Recheck saved equal-month payoffs, without reading any learning labels."""
    available = payoff_metadata(rows, table_rows, state, signal)
    for r in rows:
        value = r['class_mean_return']
        if r['status'] == 'available':
            need(type(value) is float and math.isfinite(value) and r['member_count'] > 0,
                 'Observed RET class mean is invalid')
        else:
            need(value is None, 'Unavailable RET class mean was imputed')
    mu = [] if available else None
    for cls, actual in enumerate(table_rows, 1):
        expected = math.fsum(r['class_mean_return'] / 120 for r in rows if r['class_id'] == cls) if available else None
        # This exact saved pipeline uses math.fsum in this specified order.
        need(actual['payoff'] == expected, 'Saved payoff differs from exact equal-month realized-class means')
        if available: mu.append(expected)
    need(state['fingerprint'] == canonical({'table': table_rows, 'monthly': rows, 'provenance': state['provenance']}),
         'Shared saved RET logical fingerprint changed')
    return {'mu': mu, 'fingerprint': state['fingerprint']}


def generic_table_metadata(rows, names, *, allow_empty=False):
    """Bound date/key fields before any saved financial table is converted."""
    date_names = {'date', 'trading_date', 'signal_date', 'held_signal_date', 'execution_date',
                  'scheduled_exit_date', 'outcome_start', 'outcome_end', 'period_start_date',
                  'entry_date', 'exit_date', 'effective_date'}
    need(bool(date_names & set(names)) or 'outcome_month' in names or 'year' in names
         or (allow_empty and not rows), 'Saved financial table lacks a scoped clock')
    for r in rows:
        for k in date_names & set(r):
            if r[k] is not None:
                value = r[k]
                if isinstance(value, datetime):
                    need(value.tzinfo is None and value.time() == datetime.min.time(),
                         'Intraday/timezone saved financial date')
                    value = value.date().isoformat()
                elif isinstance(value, date): value = value.isoformat()
                day(value, lower='2019-12-01' if k in ('signal_date', 'held_signal_date') else '2020-01-01')
        if 'outcome_month' in r and r['outcome_month'] is not None:
            m = r['outcome_month']; month_step(m, 0)
            need('2020-01' <= m <= '2023-12', 'Protected saved metric month')
        if 'year' in r:
            need(type(r['year']) is int and r['year'] in YEARS, 'Protected saved yearly metric')
        for k in ('intrinio_security_id', 'source_security_id'):
            if k in r and r[k] is not None:
                need(type(r[k]) is str and bool(r[k].strip()), 'Blank saved identity key')
    return True


def preflight_summary_metadata(rows, names, full_schema, *, run_scope):
    """The sole no-row-clock exception is the exact nonfinancial 15-set summary.

    The caller authenticates the source/phase chain first. This summary inherits
    that run's fixed calendar; every numerical price/return/holding table keeps
    the unchanged generic per-row clock guard.
    """
    normalized = dict(full_schema)
    for key in ('arm', 'basket'):
        if normalized.get(key) in ('string', 'large_string'): normalized[key] = 'string'
    need(list(names) == ['arm', 'basket'] and normalized == PREFLIGHT_SUMMARY_SCHEMA,
         'Preflight summary requires the exact 17-field coverage-only schema')
    need(run_scope['sessions'] == 1006 and run_scope['outcome_months'] == 48
         and run_scope['selection_sets'] == 15
         and run_scope['source_plan_pin'] == run_scope['preflight_source_plan_pin']
         == run_scope['replay_source_plan_pin'], 'Preflight summary detached from authenticated 48-month/1006-session run')
    expected = {(a, b) for a in ARMS for b in BASKETS}
    need(len(rows) == 15 and all(set(r) == {'arm', 'basket'} for r in rows)
         and {(r['arm'], r['basket']) for r in rows} == expected,
         'Preflight summary must contain exactly one row for each of fifteen arm/basket sets')
    return True


def quote_positions(rows, year, sessions):
    """No holding or future availability is an input to source-year clipping."""
    need(year in YEARS and len(sessions) == 1006, 'Four-year independent quote clock required')
    seen, positions, outside = set(), [], []
    calendar = set(sessions)
    for i, r in enumerate(rows):
        native, d = r['intrinio_security_id'], r['trading_date']
        need(type(native) is str and native and native.strip() == native, 'Blank original quote identity')
        need(type(d) is str and date.fromisoformat(d).isoformat() == d, 'Noncanonical original quote date')
        key = d, native
        need(key not in seen, 'Duplicate source quote key')
        seen.add(key)
        if d[:4] == str(year) and d in calendar: positions.append(i)
        else: outside.append(key)
    return positions, outside


def independent_preflight(selected, rebalances, quote_metadata, sessions, events, historical):
    """Enumerate every selected unstopped horizon using dated legal terms only."""
    need(len(events) == 18 and len(historical) == 11, 'Frozen current/prior action scope changed')
    dates = list(sessions); index = {d: i for i, d in enumerate(dates)}
    schedule = [r['execution_date'] for r in rebalances]
    need(len(schedule) == len(set(schedule)) == 48 and all(d in index for d in schedule),
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
