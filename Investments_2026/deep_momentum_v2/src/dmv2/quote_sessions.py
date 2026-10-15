"""Partition source quotes by an independent calendar before financial analysis.

Provider rows on non-sessions remain source evidence; they do not create exchange
sessions. This helper reads only keys in Python and filters values inside Arrow.
"""
import pyarrow as pa
import pyarrow.compute as pc
from .catalog import canonical_sha256
from .universe import DataContractError,iso_date


def exact_session_quotes(table, sessions, *, start, end):
    first,last=iso_date(start,'price start'),iso_date(end,'price end')
    if first>last or last>='2024-01-01':
        raise DataContractError('Quote session partition requires ordered pre-holdout bounds')
    if not isinstance(table,pa.Table) or not {'intrinio_security_id','trading_date'}<=set(table.column_names):
        raise DataContractError('Quote session partition requires Arrow source keys')
    if isinstance(sessions,(str,bytes)):
        raise DataContractError('Quote sessions must be an explicit ordered calendar')
    dates=tuple(iso_date(d,'calendar session') for d in sessions)
    if not dates or dates!=tuple(sorted(set(dates))) or first not in dates or last not in dates:
        raise DataContractError('Quote boundaries must be exact independent calendar sessions')
    known=set(dates);excluded=[];seen=set()
    for row in table.select(['intrinio_security_id','trading_date']).to_pylist():
        sid=row['intrinio_security_id'];day=iso_date(row['trading_date'],'quote date')
        if not isinstance(sid,str) or not sid.strip() or (sid,day) in seen:
            raise DataContractError('Invalid/duplicate source quote key')
        seen.add((sid,day))
        if not first<=day<=last:
            raise DataContractError('Source quote outside declared bounds; do not hide it as a non-session')
        if day not in known:excluded.append({**row,'reason':'not_independent_exchange_session'})
    # Preserve exact object identity for ordinary valid source blocks.
    selected=table if not excluded else table.filter(pc.is_in(table['trading_date'],value_set=pa.array(dates)))
    return selected,{'policy':'independent_exchange_sessions_only_v1','source_rows_read':table.num_rows,
        'analysis_rows':selected.num_rows,'excluded_non_session_rows':len(excluded),
        'excluded_quote_keys':excluded,'calendar_sessions_sha256':canonical_sha256(dates),
        'original_source_rows_unchanged':True,'financial_values_converted_to_python':False}
