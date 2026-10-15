"""Original-cohort identity evidence ledger; never an eligibility repair.

Source aliases and declared provider queries are separate observations. The
current contract lacks dated provider links and original publication evidence,
so neither observation certifies historical identity or knowledge-time PIT.
No prices, outcomes, model inputs or current delisting flags are inspected.
"""
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import platform
import re
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq

from .catalog import canonical_sha256
from .coverage_audit import DEFAULT_DATES, STAT_FIELDS, _remaining
from .data import CALENDAR_COLUMNS, MEMBERSHIP_COLUMNS, session_dates
from .exchange_sessions import reconcile_exchange_calendar
from .loading import BoundedTableReader, FINANCIAL_COLUMNS, ReadBudget
from .universe import DataContractError, build_snapshot, iso_date


BRIDGE = ('source_product_id', 'source_security_id', 'unified_security_id',
          'bridge_method', 'bridge_confidence', 'isin', 'primary_ticker')
LINKS = ('unified_security_id', 'intrinio_security_id', 'query_identifier',
         'query_identifier_type', 'observed_isins_json', 'observed_tickers_json',
         'identity_review_required', 'resolution_status', 'http_status',
         'source_response_hash', 'source_relative_path', 'source_identity_run_id',
         'source_identity_manifest_hash')
HISTORY = ('security_id', 'identifier_type', 'identifier_value', 'valid_from', 'valid_to')
ACQUISITION = ('requested_identifier', 'intrinio_security_id', 'status',
               'http_status', 'price_row_count', 'price_page_count')
MEMBERS = (*MEMBERSHIP_COLUMNS, 'isin')
IDENTITY_BUDGET = ReadBudget(max_metadata_files=128, max_data_files=12,
    max_scanned_rows=20_000_000, max_output_rows=100_000, max_read_bytes=134_217_728)


@dataclass(frozen=True)
class IdentityAuditResult:
    table: pa.Table
    review_queue: pa.Table
    summary: dict
    provenance: dict
    interval_profile: dict


def _rows(table, required, label, optional=()):
    if not isinstance(table, pa.Table) or set(required) - set(table.column_names):
        raise DataContractError(f'{label} requires an Arrow table with {required}')
    if (set(table.column_names) & (FINANCIAL_COLUMNS | {'target', 'label', 'return', 'next_month_return'})
        or any(c.startswith(('target_', 'outcome_', 'future_', 'label_')) for c in table.column_names)):
        raise DataContractError('Identity audit accepts metadata only, not financial values or targets')
    # Only reviewed fields reach Python, even when an in-memory caller supplies
    # extra columns. Unknown fields cannot become evidence accidentally.
    return table.select([*required, *[c for c in optional if c in table.column_names]]).to_pylist()


def valid_isin(value):
    """ISO-shaped uppercase ISIN plus expanded Luhn check; no issuer inference."""
    if not isinstance(value, str) or not re.fullmatch(r'[A-Z]{2}[A-Z0-9]{9}[0-9]', value):
        return False
    digits = ''.join(str(ord(c)-55) if c.isalpha() else c for c in value)
    total = 0
    for i, c in enumerate(reversed(digits)):
        number = int(c)*(2 if i % 2 else 1)
        total += number//10 + number % 10
    return total % 10 == 0


def _aliases(value):
    if value in (None, ''):
        return set()
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError) as error:
        raise DataContractError('Malformed published identity alias array') from error
    if not isinstance(parsed, list) or any(not isinstance(v, str) or not v.strip() for v in parsed):
        raise DataContractError('Identity aliases must be explicit nonblank strings')
    return set(parsed)


def _interval(row):
    start, end = row.get('valid_from'), row.get('valid_to')
    if start in (None, '') or end in (None, ''):
        return None, 'missing_boundary'
    try:
        start, end = iso_date(start, 'valid_from'), iso_date(end, 'valid_to')
    except DataContractError:
        return None, 'invalid_boundary'
    if start >= end:
        return None, 'empty_or_reversed_interval'
    return (start, end), 'complete_interval'


def _table_hash(table):
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return 'sha256:'+hashlib.sha256(sink.getvalue().to_pybytes()).hexdigest()


def audit_identity_links(cohorts, bridge, links, identifier_history, *,
                         signal_membership, acquisition_evidence=None):
    """Classify one to three pre-holdout original cohorts without promotion.

    Complete source identifier intervals are tested as half-open [start,end).
    This is a conservative diagnostic convention, not a published source date
    contract or dated provider link. Missing ends never become infinite ends.
    Global aliases are diagnostics only; collisions are recomputed per signal.
    """
    if isinstance(cohorts, (str, bytes)):
        raise DataContractError('Identity cohorts require an explicit sequence')
    cohorts = tuple(cohorts)
    dates = tuple(iso_date(s.as_of, 'identity signal') for s in cohorts)
    if not 1 <= len(cohorts) <= 3 or tuple(sorted(set(dates))) != dates:
        raise DataContractError('Identity audit requires one to three unique chronological cohorts')
    if any(day >= '2024-01-01' for day in dates) or len({s.universe for s in cohorts}) != 1:
        raise DataContractError('Identity cohorts require one universe and pre-2024 signals')
    bridge_rows = _rows(bridge, ('source_product_id','source_security_id','unified_security_id'), 'bridge',
                        ('isin','primary_ticker','bridge_method','bridge_confidence'))
    link_rows = _rows(links, ('unified_security_id','intrinio_security_id','query_identifier',
        'query_identifier_type','observed_isins_json','observed_tickers_json',
        'identity_review_required','resolution_status','http_status'), 'links',
        ('source_response_hash','source_relative_path','source_identity_run_id','source_identity_manifest_hash'))
    history_rows = _rows(identifier_history, HISTORY, 'identifier history')
    membership_rows = _rows(signal_membership, ('security_id','trading_date','index_id',
        'isin','membership_ticker','is_member','membership_effective_date','membership_end_date',
        'membership_availability_date'), 'signal membership')
    acquisition_rows = [] if acquisition_evidence is None else _rows(
        acquisition_evidence, ACQUISITION, 'acquisition metadata')
    routes, link_map, history, members = (defaultdict(list) for _ in range(4))
    global_aliases, bridge_isins, acquisition = (defaultdict(set) for _ in range(3))
    for r in bridge_rows:
        if r['source_product_id'] == 'sp_composite1500':
            routes[r['source_security_id']].append(r)
        if r.get('isin'):
            bridge_isins[r['unified_security_id']].add(r['isin'])
    for r in link_rows:
        link_map[r['unified_security_id']].append(r)
        if r['intrinio_security_id']:
            global_aliases[r['intrinio_security_id']].add(r['unified_security_id'])
        _aliases(r['observed_isins_json']); _aliases(r['observed_tickers_json'])
    interval_profile = Counter()
    for r in history_rows:
        history[r['security_id']].append(r)
        interval_profile[_interval(r)[1]] += 1
    for r in membership_rows:
        if r['trading_date'] in dates:
            members[(r['trading_date'],r['security_id'])].append(r)
    for r in acquisition_rows:
        if (r['status'] == 'complete' and r['http_status'] == 200
                and type(r['price_row_count']) is int and r['price_row_count'] > 0):
            acquisition[r['intrinio_security_id']].add(r['requested_identifier'])
    output = []
    for snapshot in cohorts:
        if snapshot.provenance.get('mode') != 'historical_reconstruction':
            raise DataContractError('Identity audit requires disclosed historical reconstruction')
        original = _rows(snapshot.table, ('source_security_id','as_of','indices',
            'intrinio_security_id','identity_status'), 'original cohort',
            ('unified_security_id','identity_review_required','resolution_status'))
        if not original or len({r['source_security_id'] for r in original}) != len(original):
            raise DataContractError('Original identity cohort must be nonempty with unique source IDs')
        dated_rows = []
        for r in original:
            sid, day = r['source_security_id'], snapshot.as_of
            if not isinstance(sid,str) or not sid.strip() or r['as_of'] != day:
                raise DataContractError('Invalid original cohort key or clock')
            if not r['indices'] or not set(r['indices']) <= {'sp500','sp400'}:
                raise DataContractError('Original identity cohort needs explicit index sleeves')
            observed = [m for m in members[(day,sid)] if m['index_id'] in r['indices']]
            if len(observed) != len(r['indices']) or {m['index_id'] for m in observed} != set(r['indices']):
                raise DataContractError('Missing or duplicate signal membership; no constituent may disappear')
            for m in observed:
                effective=iso_date(m['membership_effective_date'],'identity membership effective')
                available=iso_date(m['membership_availability_date'],'identity membership availability')
                end=None if m['membership_end_date'] in (None,'') else iso_date(
                    m['membership_end_date'],'identity membership end')
                if (m['is_member'] is not True or effective>day or available>day
                    or (end is not None and (end<day or end<effective))):
                    raise DataContractError('Signal identity evidence fails original membership clock gates')
            raw_isins = {m['isin'] for m in observed if m['isin'] not in (None,'')}
            source_isin = next(iter(raw_isins)) if len(raw_isins)==1 else None
            isin_status = ('MISSING' if not raw_isins else 'CONFLICTING' if len(raw_isins)>1
                           else 'VALID' if valid_isin(source_isin) else 'INVALID')
            candidates = routes[sid]
            uids = {b['unified_security_id'] for b in candidates}
            uid = next(iter(uids)) if len(uids)==1 and None not in uids and '' not in uids else None
            linked = link_map[uid] if uid else []
            link = linked[0] if len(linked)==1 else None
            pid = link['intrinio_security_id'] if link else None
            pid = pid if isinstance(pid,str) and pid.strip() else None
            structure = ('AMBIGUOUS' if (candidates and not uid) or len(linked)>1
                         else 'UNLINKED' if not pid else 'UNAMBIGUOUS')
            alias_agrees = bool(source_isin and link and source_isin in _aliases(link['observed_isins_json']))
            exact_query = bool(isin_status=='VALID' and link and pid and
                link['query_identifier_type']=='isin' and link['query_identifier']==source_isin and
                link['resolution_status']=='resolved' and link['http_status']==200)
            ticker_agrees = bool(link and {m['membership_ticker'] for m in observed
                if m['membership_ticker']} & _aliases(link['observed_tickers_json']))
            route_differs=(pid!=r['intrinio_security_id'] or
                ('unified_security_id' in r and uid!=r['unified_security_id']))
            route_agreement=False if route_differs else True if uid and pid else None
            dated_support = False
            for h in history[sid]:
                interval, _ = _interval(h)
                if (isin_status=='VALID' and h['identifier_type']=='isin' and
                    h['identifier_value']==source_isin and interval and interval[0] <= day < interval[1]):
                    dated_support = True
            dated_rows.append(dict(source_security_id=sid, signal_date=day, indices=r['indices'],
                original_identity_status=r['identity_status'],
                original_provider_review_required=r.get('identity_review_required'),
                original_link_resolution_status=r.get('resolution_status'),
                original_unified_security_id=r.get('unified_security_id'),
                original_intrinio_security_id=r['intrinio_security_id'], unified_security_id=uid,
                intrinio_security_id=pid, link_structure=structure,
                route_agrees_with_snapshot=route_agreement,
                provider_review_required=link['identity_review_required'] if link else None,
                link_resolution_status=link['resolution_status'] if link else None,
                source_isin_status=isin_status, source_alias_isin_agreement=alias_agrees,
                source_isins=sorted(raw_isins),
                source_membership_tickers=sorted({m['membership_ticker'] for m in observed if m['membership_ticker']}),
                declared_query_identifier=link['query_identifier'] if link else None,
                declared_query_identifier_type=link['query_identifier_type'] if link else None,
                published_observed_isins_json=link['observed_isins_json'] if link else None,
                published_observed_tickers_json=link['observed_tickers_json'] if link else None,
                exact_provider_isin_query=exact_query, source_ticker_alias_agreement=ticker_agrees,
                same_date_provider_collision=False, global_provider_alias_count=len(global_aliases[pid]) if pid else 0,
                dated_identifier_support=dated_support, temporal_identity_verified=False,
                knowledge_time_verified=False, qualification='REVIEW_REQUIRED_NOT_PROMOTED',
                independent_provider_isin_verified=False, raw_response_content_verified=False,
                declared_source_response_hash=link.get('source_response_hash') if link else None,
                declared_source_relative_path=link.get('source_relative_path') if link else None,
                acquisition_reported_positive_rows=bool(pid and acquisition[pid]),
                exact_query_acquisition_record_matched=bool(link and pid and
                    link['query_identifier'] in acquisition[pid]),
                price_history_coverage_verified=False))
        collisions = Counter(r['intrinio_security_id'] for r in dated_rows if r['intrinio_security_id'])
        for r in dated_rows:
            if r['intrinio_security_id'] and collisions[r['intrinio_security_id']] > 1:
                r['same_date_provider_collision']=True; r['link_structure']='COLLISION'
            r['review_priority'] = ('01_STRUCTURAL_CONFLICT' if r['route_agrees_with_snapshot'] is False
                or r['link_structure'] in ('COLLISION','AMBIGUOUS')
                or r['source_isin_status']=='CONFLICTING' else '02_MISSING_ROUTE' if r['link_structure']=='UNLINKED'
                else '03_DECLARED_ISIN_QUERY_NEEDS_PROVENANCE' if r['exact_provider_isin_query']
                else '04_WEAK_OR_MISMATCHED_IDENTIFIER')
        output.extend(dated_rows)
    summaries = []
    for day in dates:
        rows = [r for r in output if r['signal_date']==day]
        summaries.append(dict(signal_date=day, original_members=len(rows),
            original_identity_status_counts=dict(sorted(Counter(r['original_identity_status'] for r in rows).items())),
            link_structure_counts=dict(sorted(Counter(r['link_structure'] for r in rows).items())),
            source_isin_status_counts=dict(sorted(Counter(r['source_isin_status'] for r in rows).items())),
            review_priority_counts=dict(sorted(Counter(r['review_priority'] for r in rows).items())),
            exact_declared_isin_queries=sum(r['exact_provider_isin_query'] for r in rows),
            source_alias_isin_agreements=sum(r['source_alias_isin_agreement'] for r in rows),
            provider_review_flag_true=sum(r['provider_review_required'] is True for r in rows),
            dated_identifier_support_rows=sum(r['dated_identifier_support'] for r in rows),
            acquisition_reported_positive_rows=sum(r['acquisition_reported_positive_rows'] for r in rows),
            original_resolved_members=sum(r['original_identity_status']=='resolved' for r in rows),
            recomputed_route_mismatches=sum(r['route_agrees_with_snapshot'] is False for r in rows),
            promoted_identities=0, temporal_identity_verified_rows=0, knowledge_time_verified_rows=0))
    provenance = dict(audit_role='NON_PROMOTING_IDENTITY_EVIDENCE_REVIEW',
        original_cohort_selection='Supplied dated membership, before identity/price evidence',
        alias_semantics='Published source/platform alias agreement; independence unverified',
        query_semantics='Valid literal source ISIN and declared successful query, not raw HTTP request verification',
        interval_semantics='Conservative complete source [start,end) diagnostic; provider validity and knowledge times absent',
        global_alias_semantics='Inventory diagnostic only; never historical eligibility',
        acquisition_semantics='Whole acquisition reported count only; not signal-date or warmup coverage',
        eligibility_changed=False, financial_values_read=False, targets_read=False, fits_performed=0,
        source_alias_inventory=dict(link_rows=len(link_rows), observed_isins_equal_bridge_set=sum(
            _aliases(r['observed_isins_json'])==bridge_isins[r['unified_security_id']] for r in link_rows)),
        original_cohort_sha256={s.as_of:_table_hash(s.table) for s in cohorts},
        membership_exclusions={s.as_of:s.excluded.to_pylist() for s in cohorts},
        input_metadata_sha256={name:_table_hash(table) for name,table in
            (('bridge',bridge),('links',links),('identifier_history',identifier_history),('signal_membership',signal_membership))})
    if acquisition_evidence is not None:
        provenance['input_metadata_sha256']['acquisition_evidence']=_table_hash(acquisition_evidence)
    table = pa.Table.from_pylist(output)
    queue = pa.Table.from_pylist(sorted(output,key=lambda r:(r['review_priority'],r['signal_date'],r['source_security_id'])),schema=table.schema)
    return IdentityAuditResult(table,queue,dict(schema_version=1, status='REVIEW_REQUIRED_NOT_PROMOTED',
        universe=cohorts[0].universe, dates=list(dates), original_rows=len(output), snapshots=summaries,
        promoted_identities=0, eligibility_changed=False, financial_values_read=False,
        targets_read=False, model_ready=False),provenance,dict(sorted(interval_profile.items())))


def build_identity_audit(catalog, *, dates=DEFAULT_DATES, universe='combined', budget=None):
    """Read exactly three bounded initial-training anchors and metadata only."""
    if isinstance(dates,(str,bytes)):
        raise DataContractError('Identity dates require an explicit sequence')
    dates=tuple(iso_date(d,'identity date') for d in dates)
    if not 1<=len(dates)<=3 or tuple(sorted(set(dates)))!=dates or any(
        not '2000-01-01'<=d<='2009-12-31' for d in dates):
        raise DataContractError('Bounded identity dates must be one to three ordered 2000-2009 signals')
    if universe not in ('sp500','sp400','combined'):
        raise DataContractError('Unknown identity universe')
    if budget is not None and not isinstance(budget,ReadBudget):
        raise DataContractError('Identity audit budget must be an explicit ReadBudget')
    ceiling=budget or IDENTITY_BUDGET
    if not isinstance(ceiling,ReadBudget) or any(getattr(ceiling,k)>getattr(IDENTITY_BUDGET,k) for k in STAT_FIELDS):
        raise DataContractError('Identity audit budget exceeds its bounded ceiling')
    reader=BoundedTableReader(catalog,ceiling); stats={}; pins={}
    def read(name,source,table,columns,limit,keys=()):
        reader.budget=_remaining(ceiling,stats)
        loaded=reader.read_metadata(source,table,columns=columns,allowed_columns=columns,
                                    max_rows=limit,key_columns=keys)
        stats[name]=loaded.stats; pins[name]=loaded.provenance
        return loaded.table
    calendar=read('calendar','siblis','trading_calendar_sessions',CALENDAR_COLUMNS,20_000,('trading_date',))
    calendar_pins={}
    for day in dates:
        reconciled=reconcile_exchange_calendar(calendar,start_month=day[:7],end_month=day[:7])
        month=[d for d in session_dates(reconciled.table) if d[:7]==day[:7]]
        if not month or day!=month[-1]:
            raise DataContractError('Identity anchors must be independent exchange month-ends')
        calendar_pins[day]=reconciled.provenance
    bridge=read('bridge','identity','security_identity_bridge',BRIDGE,30_000,('source_product_id','source_security_id'))
    links=read('links','intrinio','intrinio_identity_links',LINKS,20_000,('unified_security_id',))
    history=read('identifier_history','siblis','security_identifier_history',HISTORY,30_000)
    acquisition=read('acquisition','intrinio','identifier_resolution_evidence',ACQUISITION,15_000,('requested_identifier',))
    reader.budget=_remaining(ceiling,stats)
    indices=('sp500','sp400') if universe=='combined' else (universe,)
    loaded=reader.read('siblis','membership_by_index_by_date',start=dates[0],end=dates[-1],
        columns=MEMBERS,allowed_columns=MEMBERS,key_columns=('index_id','security_id','trading_date'),
        filters={'index_id':indices,'trading_date':dates},financial=False)
    stats['membership']=loaded.stats; pins['membership']=loaded.provenance
    totals={v:sum(s[v] for s in stats.values()) for v in STAT_FIELDS.values()}
    if any(totals[v]>getattr(ceiling,k) for k,v in STAT_FIELDS.items()):
        from .loading import BudgetExceeded
        raise BudgetExceeded('Identity audit cumulative budget exceeded; no partial success')
    cohorts=tuple(build_snapshot(loaded.table,bridge,links,as_of=day,universe=universe,
                                 mode='historical_reconstruction') for day in dates)
    result=audit_identity_links(cohorts,bridge,links,history,signal_membership=loaded.table,
                                acquisition_evidence=acquisition)
    workers=('identity_audit.py','catalog.py','loading.py','universe.py','exchange_sessions.py','coverage_audit.py','data.py')
    directory=Path(__file__).resolve().parent
    provenance={**result.provenance,'source_pins':catalog.provenance(),
        'loading_provenance':pins,'calendar_reconciliation':calendar_pins,'budget':asdict(ceiling),
        'loading_stats':{'stages':stats,'total':totals},
        'runtime':{'python':platform.python_version(),'pyarrow':pa.__version__},
        'code_sha256':{n:hashlib.sha256((directory/n).read_bytes()).hexdigest() for n in workers}}
    return IdentityAuditResult(result.table,result.review_queue,result.summary,provenance,result.interval_profile)


def save_identity_audit(root,result):
    """Persist private original ledger, sorted review queue and exact audit pins."""
    root=Path(root).resolve()
    digest=canonical_sha256({'summary':result.summary,'provenance':result.provenance,
        'interval_profile':result.interval_profile,'ledger':_table_hash(result.table)}).split(':')[1][:16]
    directory=(root/'outputs/identity_audits'/digest).resolve()
    if not directory.is_relative_to(root):
        raise DataContractError('Identity output leaves project root')
    directory.mkdir(parents=True,exist_ok=True)
    report={**result.summary,'provenance':result.provenance,'interval_profile':result.interval_profile,
        'ledger_sha256':_table_hash(result.table),'review_queue_sha256':_table_hash(result.review_queue)}
    text=json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n'
    target=directory/'summary.json'
    if not target.resolve().is_relative_to(root):
        raise DataContractError('Identity summary leaves project root')
    if target.exists() and target.read_text()!=text:
        raise DataContractError('Identity evidence drift under identical pins')
    for name,table in (('constituent_ledger.parquet',result.table),('review_queue.parquet',result.review_queue)):
        destination=directory/name
        if not destination.resolve().is_relative_to(root):
            raise DataContractError('Identity output symlink leaves project root')
        if destination.exists():
            if not pq.read_table(destination).equals(table):
                raise DataContractError('Saved identity ledger changed under identical pins')
        else:
            with tempfile.NamedTemporaryFile(dir=directory,delete=False) as handle:
                temporary=Path(handle.name)
            pq.write_table(table,temporary); temporary.replace(destination)
    if not target.exists():
        with tempfile.NamedTemporaryFile('w',dir=directory,delete=False,encoding='utf-8') as handle:
            handle.write(text); temporary=Path(handle.name)
        temporary.replace(target)
    return target
