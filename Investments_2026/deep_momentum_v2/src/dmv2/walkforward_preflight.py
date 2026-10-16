"""Inspect every selected maximum holding horizon before computing performance.

Only date/native keys and nonfinancial source-quality flags are inputs. Missing
observations are repair leads, never admission filters or inferred cash payouts.
The unstopped maximum horizon is reviewed even for a possible stopped variant.
"""
from collections import defaultdict

import numpy as np
import pandas as pd

from .walkforward_accounting import _dates, _historical_retirements, _prepare_actions
from .universe import DataContractError

QUOTE_METADATA_COLUMNS = ('trading_date', 'intrinio_security_id', 'frequency', 'intraperiod',
    'price_quality_status', 'raw_ohlc_bounds_valid', 'adjusted_ohlc_bounds_valid', 'price_accepted')
ISSUE_COLUMNS = ['signal_date', 'execution_date', 'scheduled_exit_date', 'source_security_id',
    'intrinio_security_id', 'side', 'date', 'role', 'reason']


def validate_current_events(events, sessions):
    records = events.to_dict('records') if isinstance(events, pd.DataFrame) else list(events)
    allowed = {'cash_merger', 'stock_merger', 'mixed_merger', 'verified_retirement'}
    for event in records:
        if not isinstance(event, dict) or event.get('action_type', 'verified_retirement') not in allowed:
            raise DataContractError('Unsupported corporate action needs a separately reviewed economic ledger')
    return _prepare_actions(records, _dates(list(sessions), 'preflight sessions'))


def preflight_selected_horizons(selections, rebalances, quote_metadata, sessions, *,
        events=(), historical_retirements=()):
    """Return all missing/quality keys for frozen selections without any PnL.

    Every episode runs through the next scheduled monthly opening, or the final
    authorized close. Cash/stock conversions use only independently verified
    event definitions, including predecessor prior-close and successor-open keys.
    No price, outcome or later membership may be supplied in quote_metadata.
    """
    if (not isinstance(quote_metadata, pd.DataFrame)
            or not {'trading_date', 'intrinio_security_id'} <= set(quote_metadata)
            or not set(quote_metadata) <= set(QUOTE_METADATA_COLUMNS)):
        raise DataContractError('Selected-horizon preflight accepts only exact date/identity/quality metadata')
    sessions = _dates(list(sessions), 'preflight sessions')
    if (sessions.empty or sessions.has_duplicates or not sessions.is_monotonic_increasing
            or sessions.min() < pd.Timestamp('2020-01-01') or sessions.max() > pd.Timestamp('2023-12-31')):
        raise DataContractError('Preflight sessions leave the authorized 2020–2023 clock')
    metadata = quote_metadata.copy(deep=True)
    metadata['trading_date'] = _dates(metadata.trading_date, 'preflight quote dates')
    if (not metadata.trading_date.isin(sessions).all()
            or metadata.duplicated(['trading_date', 'intrinio_security_id']).any()
            or metadata.intrinio_security_id.isna().any()):
        raise DataContractError('Preflight quote keys are duplicate, null or out of scope')
    if not {'signal_date', 'execution_date', 'source_security_id', 'intrinio_security_id', 'side'} <= set(selections):
        raise DataContractError('Preflight requires the untouched frozen selection keys')
    selection = selections.copy(deep=True)
    selection['execution_date'] = _dates(selection.execution_date, 'selected execution dates')
    schedule = rebalances.copy(deep=True)
    schedule['execution_date'] = _dates(schedule.execution_date, 'preflight rebalance dates')
    if (schedule.execution_date.duplicated().any() or not schedule.execution_date.isin(sessions).all()
            or not selection.execution_date.isin(schedule.execution_date).all()):
        raise DataContractError('Preflight selection does not match its explicit schedule')
    definitions = validate_current_events(events, sessions)
    history = _historical_retirements(historical_retirements, definitions)
    retired = sorted([*history, *definitions], key=lambda item: (item['effective_date'], item['event_id']))
    current = defaultdict(list)
    for event in definitions:
        current[event['effective_date']].append(event)
    quote_rows = {(row['trading_date'], row['intrinio_security_id']): row for row in metadata.to_dict('records')}
    openings = sorted(schedule.execution_date)
    exits = {day: openings[i+1] if i+1 < len(openings) else sessions[-1] for i, day in enumerate(openings)}
    gaps, quality, unfilled, needed_events = [], [], [], []
    required_keys = set()

    def needed(row, native, date, role):
        required_keys.add((date, native))
        original = quote_rows.get((date, native))
        details = {name: getattr(row, name) for name in ('signal_date', 'execution_date',
            'source_security_id', 'side')}
        details.update(scheduled_exit_date=exits[row.execution_date], intrinio_security_id=native,
            date=date, role=role)
        if original is None:
            gaps.append({**details, 'reason': 'missing_key_requires_observed_quote_or_verified_legal_action'})
            return
        if 'price_accepted' in original:
            accepted = isinstance(original['price_accepted'], (bool, np.bool_)) and bool(original['price_accepted'])
        else:
            accepted = (original.get('frequency') == 'daily'
                and isinstance(original.get('intraperiod'), (bool, np.bool_)) and not original['intraperiod']
                and original.get('price_quality_status') in {'accepted', 'ok'})
        for name in ('raw_ohlc_bounds_valid', 'adjusted_ohlc_bounds_valid'):
            if name in original:
                accepted &= isinstance(original[name], (bool, np.bool_)) and bool(original[name])
        if not accepted:
            quality.append({**details, 'reason': 'unaccepted_source_metadata_requires_independent_review'})

    for row in selection.itertuples(index=False):
        native = row.intrinio_security_id
        retirement = next((event for event in reversed(retired)
            if event['predecessor_id'] == native and event['effective_date'] <= row.execution_date), None)
        if retirement is not None:
            unfilled.append({'signal_date': row.signal_date, 'execution_date': row.execution_date,
                'source_security_id': row.source_security_id, 'intrinio_security_id': native,
                'event_id': retirement['event_id'], 'status': 'retired_new_order_unfilled_cash_no_entitlement'})
            continue
        for day in sessions[(sessions >= row.execution_date) & (sessions <= exits[row.execution_date])]:
            for event in current.get(day, []):
                if event['predecessor_id'] != native:
                    continue
                previous = sessions.get_loc(day) - 1
                if previous < 0:
                    raise DataContractError('Held conversion lacks its prior authorized session')
                needed(row, native, sessions[previous], 'action_predecessor_paired_close')
                needed_events.append({'signal_date': row.signal_date, 'source_security_id': row.source_security_id,
                    'date': day, 'event_id': event['event_id'], 'predecessor_id': native,
                    'successor_id': event['successor_id'], 'status': 'verified_retirement_definition_required_and_present'})
                native = event['successor_id'] if event['stock_ratio'] > 0 else None
            if native is None:
                break
            needed(row, native, day, 'entry_open' if day == row.execution_date else
                'scheduled_exit_open' if day == exits[row.execution_date] and day != sessions[-1] else 'held_open_and_close')
    gap_table, quality_table = pd.DataFrame(gaps, columns=ISSUE_COLUMNS), pd.DataFrame(quality, columns=ISSUE_COLUMNS)
    summary = {'selected_episode_rows': len(selection), 'distinct_maximum_horizon_quote_keys': len(required_keys),
        'missing_key_episode_rows': len(gap_table), 'quality_flag_episode_rows': len(quality_table),
        'unique_missing_keys': len(gap_table[['date', 'intrinio_security_id']].drop_duplicates()),
        'unique_quality_flag_keys': len(quality_table[['date', 'intrinio_security_id']].drop_duplicates()),
        'verified_current_action_definitions': len(definitions), 'historical_retirement_definitions': len(history),
        'retired_new_orders': len(unfilled), 'ready_for_accounting': gap_table.empty and quality_table.empty,
        'source_metadata_only': True, 'prices_or_outcomes_read': False,
        'selection_changed': False, 'unstopped_maximum_horizons_reviewed_for_both_stop_variants': True}
    return {'summary': summary, 'missing_quotes': gap_table, 'source_quality': quality_table,
        'retired_orders': pd.DataFrame(unfilled), 'action_requirements': pd.DataFrame(needed_events)}
