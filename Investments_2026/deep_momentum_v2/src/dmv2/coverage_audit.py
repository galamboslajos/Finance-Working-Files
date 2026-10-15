"""Bounded historical membership/identity coverage, without financial reads.

Counts are an upper bound on a qualified normalization population. Identity
routes are historical reconstructions; this audit does not certify quote,
feature-history, publication-time or original-vintage availability.
"""
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import tempfile

import pyarrow as pa
import pyarrow.compute as pc

from .catalog import canonical_sha256
from .data import BRIDGE_COLUMNS, CALENDAR_COLUMNS, LINK_COLUMNS, MEMBERSHIP_COLUMNS, session_dates
from .exchange_sessions import reconcile_exchange_calendar
from .loading import BoundedTableReader, BudgetExceeded, ReadBudget
from .universe import DataContractError, build_snapshot, iso_date


DEFAULT_DATES = ('2000-01-31', '2005-01-31', '2009-01-30')
AUDIT_BUDGET = ReadBudget(max_metadata_files=128, max_data_files=12,
                         max_scanned_rows=36_000_000, max_output_rows=100_000,
                         max_read_bytes=134_217_728)
STAT_FIELDS = {'max_metadata_files': 'metadata_files', 'max_data_files': 'data_files',
               'max_scanned_rows': 'scanned_rows', 'max_output_rows': 'returned_rows',
               'max_read_bytes': 'projected_compressed_bytes'}


def _qualified_rows(snapshot):
    fields = ('source_security_id', 'intrinio_security_id', 'identity_status', 'indices')
    if not set(fields) <= set(snapshot.table.column_names):
        raise DataContractError('Coverage snapshot missing required identity/cohort fields')
    optional = [c for c in ('identity_review_required', 'resolution_status', 'as_of')
                if c in snapshot.table.column_names]
    rows = snapshot.table.select([*fields, *optional]).to_pylist()
    seen, routes = set(), {}
    for row in rows:
        sid = row['source_security_id']
        if not isinstance(sid, str) or not sid.strip() or sid in seen:
            raise DataContractError('Coverage requires unique nonmissing original source IDs')
        seen.add(sid)
        if 'as_of' in row and row['as_of'] != snapshot.as_of:
            raise DataContractError('Coverage cohort row clock differs from snapshot signal')
        indices = row['indices']
        if not isinstance(indices, list) or not indices or not set(indices) <= {'sp500', 'sp400'}:
            raise DataContractError('Coverage snapshot requires an explicit SP500/SP400 sleeve')
        status = row['identity_status']
        if not isinstance(status, str) or not status:
            raise DataContractError('Missing coverage identity status')
        row['source_identity_status'] = status
        pid = row['intrinio_security_id']
        if pid is not None:
            if not isinstance(pid, str) or not pid.strip():
                raise DataContractError('Invalid provider identity route')
            routes.setdefault(pid, []).append(row)
        if status == 'resolved':
            if pid is None:
                raise DataContractError('Resolved coverage identity has no provider ID')
            if row.get('identity_review_required') is not False or row.get('resolution_status') != 'resolved':
                row['identity_status'] = 'needs_review'
    for linked in routes.values():
        if len(linked) > 1:
            for row in linked:
                row['identity_status'] = 'target_collision'
    return rows


def summarize_coverage_snapshot(snapshot, *, historical_price_ids=None):
    """Summarize original populations; optional IDs are inventory only.

    historical_price_ids is not dated price evidence and cannot establish
    observed quote/history coverage. It never selects members or statistics.
    """
    day = iso_date(snapshot.as_of, 'coverage as_of')
    if day >= '2024-01-01':
        raise DataContractError('Protected 2024+ is outside this development coverage audit')
    if snapshot.universe not in ('sp500', 'sp400', 'combined'):
        raise DataContractError('Unknown coverage universe')
    if snapshot.provenance.get('mode', 'historical_reconstruction') != 'historical_reconstruction':
        raise DataContractError('Coverage requires disclosed historical reconstruction')
    if historical_price_ids is not None:
        if isinstance(historical_price_ids, (str, bytes)):
            raise DataContractError('historical_price_ids must be an explicit sequence')
        inventory = tuple(historical_price_ids)
        if len(inventory) != len(set(inventory)) or any(not isinstance(v, str) or not v.strip() for v in inventory):
            raise DataContractError('Historical ID inventory must be unique and nonmissing')
        inventory = set(inventory)
    else:
        inventory = None
    rows = _qualified_rows(snapshot)
    groups = {}
    for group in ('universe', 'sp500', 'sp400'):
        members = rows if group == 'universe' else [r for r in rows if group in r['indices']]
        qualified = [r for r in members if r['identity_status'] == 'resolved']
        groups[group] = {'original_members': len(members),
            'identity_status_counts': dict(sorted(Counter(r['identity_status'] for r in members).items())),
            'qualified_identity_members': len(qualified),
            'qualified_identity_fraction': len(qualified)/len(members) if members else None,
            'candidate_signal_qualified_population_upper_bound': len(qualified),
            'active_complete_cohort_possible_from_identity': bool(members) and len(qualified) == len(members),
            'active_complete_cohort_normalized_rows_upper_bound': len(members) if members and len(qualified) == len(members) else 0,
            'five_momentum_population_verified': False,
            'inventory_intersection_members': sum(r['intrinio_security_id'] in inventory for r in qualified) if inventory is not None else None}
    return {'as_of': day, 'universe': snapshot.universe, 'groups': groups,
            'original_members': len(rows), 'membership_exclusions': snapshot.excluded.num_rows,
            'knowledge_time_verified': False, 'financial_values_read': False,
            'price_history_coverage_verified': False, 'normalization_adequacy_verified': False,
            'inventory_semantics': 'Provider-ID inventory intersection only, not dated financial observations',
            'status': 'METADATA_ONLY_IDENTITY_COVERAGE', 'constituent_ledger': rows,
            'membership_exclusion_ledger': snapshot.excluded.to_pylist(),
            'source_provenance': snapshot.provenance}


def _remaining(ceiling, stages):
    left = {}
    for limit, statistic in STAT_FIELDS.items():
        left[limit] = getattr(ceiling, limit)-sum(s[statistic] for s in stages.values())
        if left[limit] <= 0:
            raise BudgetExceeded(f'Coverage audit cumulative {limit} exhausted; no partial success')
    return ReadBudget(**left)


def build_coverage_audit(catalog, *, dates=DEFAULT_DATES, universe='combined', budget=None):
    """Three first-training-window anchor dates; metadata projections only.

    This is a diagnostic, not an all-month audit or permission to calculate
    financial features for all members. No return, target or price table read.
    """
    if isinstance(dates, (str, bytes)):
        raise DataContractError('Coverage dates require an explicit sequence')
    dates = tuple(iso_date(day, 'coverage date') for day in dates)
    if not 1 <= len(dates) <= 3 or tuple(sorted(set(dates))) != dates:
        raise DataContractError('Coverage dates must be one to three unique chronological dates')
    if any(not '2000-01-01' <= day <= '2009-12-31' for day in dates):
        raise DataContractError('This bounded calibration audit is confined to the initial 2000-2009 training years; 2024+ is protected')
    if universe not in ('sp500', 'sp400', 'combined'):
        raise DataContractError('Unknown coverage universe')
    if budget is not None and not isinstance(budget, ReadBudget):
        raise DataContractError('Coverage budget must be an explicit ReadBudget')
    ceiling = budget or AUDIT_BUDGET
    for field in STAT_FIELDS:
        if getattr(ceiling, field) > getattr(AUDIT_BUDGET, field):
            raise DataContractError('Coverage budget cannot exceed this bounded audit ceiling')
    reader = BoundedTableReader(catalog, ceiling)
    stats, sources = {}, {}

    def capture(name, loaded):
        stats[name], sources[name] = loaded.stats, loaded.provenance
        return loaded.table

    source_calendar = capture('calendar', reader.read_metadata(
        'siblis', 'trading_calendar_sessions', columns=CALENDAR_COLUMNS,
        allowed_columns=CALENDAR_COLUMNS, max_rows=20_000, key_columns=('trading_date',)))
    calendars = {}
    for day in dates:
        r = reconcile_exchange_calendar(source_calendar, start_month=day[:7], end_month=day[:7])
        listed = session_dates(r.table)
        month_days = [d for d in listed if d[:7] == day[:7]]
        if not month_days or day != month_days[-1]:
            raise DataContractError('Coverage anchors must be independent exchange month-ends')
        calendars[day] = r.provenance
    reader.budget = _remaining(ceiling, stats)
    bridge = capture('bridge', reader.read_metadata(
        'identity', 'security_identity_bridge', columns=BRIDGE_COLUMNS,
        allowed_columns=BRIDGE_COLUMNS, max_rows=30_000,
        key_columns=('source_product_id', 'source_security_id')))
    reader.budget = _remaining(ceiling, stats)
    links = capture('links', reader.read_metadata(
        'intrinio', 'intrinio_identity_links', columns=LINK_COLUMNS,
        allowed_columns=LINK_COLUMNS, max_rows=20_000, key_columns=('unified_security_id',)))
    summaries = []
    indices = ('sp500','sp400') if universe == 'combined' else (universe,)
    for day in dates:
        reader.budget = _remaining(ceiling, stats)
        members = capture('membership_'+day, reader.read(
            'siblis', 'membership_by_index_by_date', start=day, end=day,
            columns=MEMBERSHIP_COLUMNS, allowed_columns=MEMBERSHIP_COLUMNS,
            key_columns=('index_id', 'security_id', 'trading_date'),
            filters={'index_id':indices}, financial=False))
        if not members.num_rows:
            raise DataContractError(f'No original membership at audit anchor {day}')
        snapshot = build_snapshot(members, bridge, links, as_of=day, universe=universe,
                                  mode='historical_reconstruction')
        summaries.append(summarize_coverage_snapshot(snapshot))
    totals = {statistic:sum(s[statistic] for s in stats.values()) for statistic in STAT_FIELDS.values()}
    for limit, statistic in STAT_FIELDS.items():
        if totals[statistic] > getattr(ceiling,limit):
            raise BudgetExceeded(f'Coverage audit exceeded cumulative {limit}; no partial success')
    workers = ('coverage_audit.py','catalog.py','data.py','loading.py','universe.py','exchange_sessions.py')
    root = Path(__file__).resolve().parent
    return {'schema_version':1, 'audit_role':'INITIAL_TRAINING_ANCHOR_INPUT_METADATA',
        'dates':list(dates),'universe':universe,'snapshots':summaries,
        'source_pins':catalog.provenance(), 'loading_provenance':sources,
        'calendar_reconciliation':calendars, 'loading_stats':{'stages':stats,'total':totals},
        'budget':asdict(ceiling), 'code_sha256':{n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in workers},
        'runtime':{'python':platform.python_version(),'pyarrow':pa.__version__},
        'financial_values_read':False,'targets_read':False,'fits_performed':0,
        'claim_limits':'Three metadata anchors are not full historical quote/momentum coverage, statistical representativeness or knowledge-time certification.'}


def save_coverage_audit(root, audit):
    root = Path(root).resolve()
    pins = {key:audit[key] for key in ('dates','universe','source_pins','budget','code_sha256','runtime')}
    parent = (root/'outputs/coverage_audits').resolve()
    if not parent.is_relative_to(root):
        raise DataContractError('Coverage output leaves the project')
    parent.mkdir(parents=True,exist_ok=True)
    target = parent/(canonical_sha256(pins).split(':')[1][:16]+'.json')
    if not target.resolve().is_relative_to(root):
        raise DataContractError('Coverage output leaves the project')
    text = json.dumps(audit,indent=2,sort_keys=True,allow_nan=False)+'\n'
    if target.exists():
        if target.read_text()!=text:
            raise DataContractError('Coverage evidence changed under identical pins; preserve and investigate')
        return target
    with tempfile.NamedTemporaryFile('w',dir=parent,delete=False,encoding='utf-8') as handle:
        handle.write(text); temporary=Path(handle.name)
    temporary.replace(target)
    return target
