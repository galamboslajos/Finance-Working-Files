"""Independent frozen XNYS clock for 48 monthly 2020–2023 outcomes."""
from importlib.metadata import version

import pandas as pd

from .backtest_loading import EXCHANGE_CALENDARS_VERSION, _package_code_hash
from .catalog import canonical_sha256
from .universe import DataContractError

YEARS = (2020, 2021, 2022, 2023)
ANNUAL_SESSIONS = {2020: 253, 2021: 252, 2022: 251, 2023: 250}
FIRST_SESSION, LAST_SESSION = '2020-01-02', '2023-12-29'


def walkforward_clock():
    if version('exchange-calendars') != EXCHANGE_CALENDARS_VERSION:
        raise DataContractError('Walk-forward clock requires the existing pinned calendar package')
    import exchange_calendars as xcals
    calendar = xcals.get_calendar('XNYS', start='2019-12-01', end='2023-12-31')
    if calendar.name != 'XNYS' or type(calendar).__module__ != 'exchange_calendars.exchange_calendar_xnys':
        raise DataContractError('Unexpected XNYS calendar implementation')
    dates = tuple(calendar.sessions.strftime('%Y-%m-%d'))
    sessions = tuple(day for day in dates if '2020-01-01' <= day <= '2023-12-31')
    if (len(sessions) != 1006 or sessions[0] != FIRST_SESSION or sessions[-1] != LAST_SESSION
            or {year: sum(day.startswith(str(year)) for day in sessions) for year in YEARS} != ANNUAL_SESSIONS):
        raise DataContractError('Walk-forward clock lost its exact 1,006 authorized sessions')
    windows = []
    months = pd.period_range('2020-01', '2023-12', freq='M')
    for index, month in enumerate(months):
        previous = [day for day in dates if day[:7] == str(month - 1)]
        current = [day for day in sessions if day[:7] == str(month)]
        following = [day for day in sessions if day[:7] == str(month + 1)]
        windows.append({'signal_date': previous[-1], 'outcome_month': str(month),
            'outcome_start': current[0], 'outcome_end': current[-1],
            'entry_date': current[0], 'execution_date': current[0],
            'exit_date': following[0] if index < 47 else current[-1],
            'exit_price_field': 'adj_open' if index < 47 else 'adj_close'})
    _, code_hash = _package_code_hash(xcals)
    return sessions, pd.DataFrame(windows), {'calendar': 'XNYS',
        'package_version': EXCHANGE_CALENDARS_VERSION, 'calendar_code_sha256': code_hash,
        'calendar_start': '2019-12-01', 'calendar_end': '2023-12-31',
        'sessions_sha256': canonical_sha256(list(sessions)),
        'selection_sessions': list(dates),
        'selection_sessions_sha256': canonical_sha256(list(dates)),
        'calendar_source': 'independent_rule_based_calendar_not_quote_inventory',
        'annual_session_counts': ANNUAL_SESSIONS, 'outcome_months': 48,
        'annual_liquidation': False, 'terminal_date': LAST_SESSION,
        'financial_2024_plus_read': False}
