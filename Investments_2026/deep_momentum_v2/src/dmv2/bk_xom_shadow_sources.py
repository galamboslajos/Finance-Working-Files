"""Pure, cache-only BK/XOM shadow source views; this module performs no I/O.

The frozen feature workers call their join key ``intrinio_security_id``. In
these shadow views it is explicitly a computational key, not a provider claim.
Original native IDs, all source flags and typed provider identities are separate
audit fields. Actual source pin checks, reservations and reads belong to caller.
"""
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from datetime import date

import pyarrow as pa
import pyarrow.compute as pc

from dmv2.catalog import canonical_sha256
from dmv2.identity_policy import PROVISIONAL_IDENTITY_POLICY, qualify_identity_records
from dmv2.universe import DataContractError

BK_SOURCE = 'siblis_ticker_company_1c21a7b293fd137c59c9'
BK_UNIFIED = 'equity_us_a2ca9feacdef53087ce87b15'
BK_OLD, BK_KEY = 'sec_gA3NYQ', 'sec_zq1J05'
XOM_SOURCE = 'siblis_isin_b284bb7497616f5afba0'
XOM_UNIFIED = 'equity_us_bf463bc8d3ea387578b7bafc'
XOM_OLD = 'sec_z5YpGE'
XOM_KEY = 'dmv2_sip:siblis_isin_b284bb7497616f5afba0:all'
POLICY = 'dmv2-BK-XOM-cache-only-shadow-routes-20261021-v1'
KEYS = ('source_security_id', 'signal_date')
PRICE_KEYS = ('intrinio_security_id', 'trading_date')
PRICE_COLUMNS = (*PRICE_KEYS, 'frequency', 'intraperiod', 'open', 'high', 'low', 'close',
    'volume', 'adj_open', 'adj_high', 'adj_low', 'adj_close', 'adj_volume', 'factor',
    'split_ratio', 'dividend', 'raw_ohlc_bounds_valid', 'adjusted_ohlc_bounds_valid',
    'price_quality_status')
PRICE_SCHEMA = pa.schema([(name, pa.string() if name in (*PRICE_KEYS, 'frequency', 'price_quality_status')
    else pa.bool_() if name in ('intraperiod', 'raw_ohlc_bounds_valid', 'adjusted_ohlc_bounds_valid')
    else pa.float64()) for name in PRICE_COLUMNS])
WARNING = 'raw_all_OHLC_price_scales_disagree'
QUALITY_POLICY = 'retain_precision_warned_published_all'
ROUTES = {
 BK_SOURCE: (BK_UNIFIED, BK_OLD, BK_KEY, 'intrinio', '2007-07-02', '2023-12-29'),
 XOM_SOURCE: (XOM_UNIFIED, XOM_OLD, XOM_KEY, 'Alpaca/SIP', '2016-01-04', '2023-12-29'),
}


def need(value, message):
    if not bool(value):
        raise DataContractError(message)


def day(value, *, financial=True):
    need(isinstance(value, str), 'Canonical date string required')
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise DataContractError('Invalid date') from error
    need(parsed.isoformat() == value and (not financial or value < '2024-01-01'),
         'Noncanonical or protected financial/cohort date')
    return value


def ordered_keys(table):
    need(isinstance(table, pa.Table) and set(KEYS) <= set(table.column_names), 'Original Arrow cohort keys required')
    pairs = tuple((r[KEYS[0]], day(r[KEYS[1]])) for r in table.select(list(KEYS)).to_pylist())
    need(len(pairs) == len(set(pairs)), 'Duplicate original member/date')
    return pairs


def _table(rows, original):
    """Retain source column types/order; append separate audit fields."""
    names = list(original.column_names)
    names.extend(n for r in rows for n in r if n not in names)
    fields = {f.name: f for f in original.schema}
    arrays = [pa.array([r.get(n) for r in rows], type=fields[n].type if n in fields else None)
              for n in names]
    return pa.Table.from_arrays(arrays, names=names)


def _route_one(table, expected_keys):
    need(ordered_keys(table) == tuple(expected_keys), 'Complete original ordered membership required before routes/prices')
    required = {'source_security_id', 'signal_date', 'unified_security_id', 'intrinio_security_id',
                'identity_status', 'identity_review_required', 'resolution_status', 'indices'}
    need(required <= set(table.column_names), 'Complete source identity/sleeve metadata required')
    forbidden = {'open', 'high', 'low', 'close', 'volume', 'adj_open', 'adj_high', 'adj_low',
                 'adj_close', 'adj_volume', 'factor', 'split_ratio', 'dividend',
                 'label_return', 'target_class', 'target', 'future_return'}
    need(not forbidden & set(table.column_names), 'Routing accepts metadata only')
    originals = table.to_pylist()
    rows = []
    for original in originals:
        r = deepcopy(original)
        need(not any(n.startswith('shadow_') for n in r), 'Shadow routes cannot be automatically reapplied')
        d = day(r['signal_date']); sid = r['source_security_id']; old = r['intrinio_security_id']
        need(isinstance(r['indices'], list) and r['indices'] and set(r['indices']) <= {'sp500', 'sp400'},
             'Original historical index sleeves required')
        for clock in ('as_of', 'trading_date'):
            if clock in r:
                need(r[clock] == d, 'Conflicting original cohort clocks')
        for field in ('intrinio_security_id', 'identity_status', 'identity_review_required',
                      'resolution_status', 'identity_eligible', 'requested',
                      'paper_prediction_eligible', 'prediction_eligible'):
            if field in r:
                r['shadow_original_' + field] = r[field]
        for field in ('requested', 'paper_prediction_eligible', 'prediction_eligible'):
            r.pop(field, None)
        # A saved collision was set only after unique structural routes were
        # assembled. Reconstruct that pre-collision state, then count all rows
        # afresh. Missing, ambiguous and quarantined routes stay blocked.
        if r['identity_status'] == 'target_collision':
            r['identity_status'] = ('resolved' if r['identity_review_required'] is False
                and r['resolution_status'] == 'resolved' else 'needs_review')
        provider = 'intrinio'; source_key = old
        status = 'unchanged_corrected_annual_cache_route'
        if sid in ROUTES:
            unified, prior, key, candidate_provider, start, end = ROUTES[sid]
            need(r['unified_security_id'] == unified and old == prior,
                 'Reviewed source/unified/original native route changed')
            if start <= d <= end:
                r['intrinio_security_id'] = key
                provider, source_key = candidate_provider, key if sid == BK_SOURCE else XOM_SOURCE
                status = 'conditional_reviewed_shadow_route'
            else:
                status = 'original_route_no_candidate_interval_or_history_fill'
        r.update(shadow_route_policy=POLICY, computational_price_key=r['intrinio_security_id'],
            price_route_provider=provider, price_route_source_key=source_key,
            provider_intrinio_security_id=r['intrinio_security_id'] if provider == 'intrinio' else None,
            price_route_status=status, shadow_canonical_route_promoted=False)
        rows.append(r)
    # Typed source conflicts and duplicate unified instruments related to an
    # amendment count ALL original members, even those already blocked.
    by_day = defaultdict(list)
    for i, r in enumerate(rows):
        by_day[r['signal_date']].append(i)
    conflicts = set()
    for indexes in by_day.values():
        for i in indexes:
            r = rows[i]
            if r['source_security_id'] in ROUTES and r['price_route_status'] == 'conditional_reviewed_shadow_route':
                for j in indexes:
                    other = rows[j]
                    if i != j and (other['unified_security_id'] == r['unified_security_id'] or
                        (other['price_route_provider'], other['price_route_source_key']) ==
                        (r['price_route_provider'], r['price_route_source_key'])):
                        conflicts.update((i, j))
    qualified = qualify_identity_records(rows, identity_policy=PROVISIONAL_IDENTITY_POLICY)
    for i in conflicts:
        qualified[i].update(identity_status='target_collision', identity_eligible=False,
            identity_eligibility_reason='blocked_shadow_typed_route_or_unified_collision')
    output = _table(qualified, table)
    need(ordered_keys(output) == tuple(expected_keys), 'Route amendment changed original denominators/order')
    ledger = [{k: r.get(k) for k in (*KEYS, 'unified_security_id', 'shadow_original_intrinio_security_id',
        'intrinio_security_id', 'computational_price_key', 'price_route_provider', 'price_route_source_key',
        'provider_intrinio_security_id', 'price_route_status', 'shadow_original_identity_status',
        'identity_status', 'identity_eligible', 'identity_eligibility_reason', 'identity_review_required',
        'resolution_status')} for r in qualified]
    return output, ledger


@dataclass(frozen=True)
class RoutingResult:
    monthly_cohort: pa.Table
    daily_cohorts: pa.Table
    requested_ids: tuple
    monthly_ledger: tuple
    daily_ledger: tuple
    provenance: dict


def route_cohorts(monthly, daily, *, expected_monthly_keys, expected_daily_keys):
    """Requalify complete original dated populations before any price access."""
    m, ml = _route_one(monthly, expected_monthly_keys)
    d, dl = _route_one(daily, expected_daily_keys)
    requested = tuple(sorted({r['intrinio_security_id'] for r in (*ml, *dl) if r['identity_eligible']}))
    return RoutingResult(m, d, requested, tuple(ml), tuple(dl), {
        'policy': POLICY, 'all_original_members_before_price_joins': True,
        'original_monthly_keys_sha256': canonical_sha256(expected_monthly_keys),
        'original_daily_keys_sha256': canonical_sha256(expected_daily_keys),
        'monthly_original_rows': len(ml), 'daily_original_rows': len(dl),
        'requested_ids': list(requested), 'quote_availability_used': False,
        'original_review_flags_preserved': True, 'canonical_route_promoted': False,
        'legacy_intrinio_column_is_explicit_computational_join_key': True,
        'RCM_DISCA_execution_only_repairs_applied': False})


def _quote_keys(table, *, start, end, allowed_ids=None, sessions=None):
    need(isinstance(table, pa.Table) and set(PRICE_KEYS) <= set(table.column_names), 'Arrow daily source keys required')
    keys = []
    for r in table.select(list(PRICE_KEYS)).to_pylist():
        sid, d = r['intrinio_security_id'], day(r['trading_date'])
        need(isinstance(sid, str) and sid and start <= d <= end, 'Quote key/date outside explicit pre-holdout scope')
        need(allowed_ids is None or sid in allowed_ids, 'Unrequested source key')
        need(sessions is None or d in sessions, 'Non-XNYS quote before numerical use')
        keys.append((sid, d))
    need(len(keys) == len(set(keys)), 'Duplicate daily source keys; no silent deduplication')
    return keys


def _core(table):
    need(set(PRICE_COLUMNS) <= set(table.column_names), 'Canonical source fields missing')
    return table.select(list(PRICE_COLUMNS)).cast(PRICE_SCHEMA, safe=True)


def normalize_candidates(bk_quotes, sip_pairs, *, sessions, xom_quality_metadata,
                         expected_xom_warning_dates):
    """Preserve published pairs exactly; no factor/dividend application or fills.

    Caller has authenticated these bounded candidate outputs before passing
    them. Source metadata is checked for ALL rows before column casts/assembly.
    All 69 real warnings are passed by the frozen caller, never recalibrated.
    """
    known = {day(d, financial=False) for d in sessions}
    _quote_keys(bk_quotes, start='2007-07-02', end='2023-12-29', allowed_ids={BK_KEY}, sessions=known)
    names = {'security_id', 'trading_date', 'query_symbol', 'adjustment', 'open', 'high', 'low', 'close', 'volume'}
    need(isinstance(sip_pairs, pa.Table) and names <= set(sip_pairs.column_names), 'Exact raw/all SIP fields required')
    seen = set()
    for r in sip_pairs.select(['security_id', 'trading_date', 'query_symbol', 'adjustment']).to_pylist():
        d = day(r['trading_date']); k = (d, r['adjustment'])
        need(r['security_id'] == XOM_SOURCE and r['query_symbol'] == 'XOM' and r['adjustment'] in {'raw', 'all'}
             and '2016-01-04' <= d <= '2023-12-29' and d in known and k not in seen,
             'SIP source/date/mode/session/uniqueness guard before financial casts')
        seen.add(k)
    raw = sip_pairs.filter(pc.equal(sip_pairs['adjustment'], 'raw')).sort_by('trading_date')
    adjusted = sip_pairs.filter(pc.equal(sip_pairs['adjustment'], 'all')).sort_by('trading_date')
    dates = raw['trading_date'].to_pylist()
    need(dates == adjusted['trading_date'].to_pylist(), 'Missing raw/all pair cannot be fabricated')
    quality = list(xom_quality_metadata)
    need(all(set(r) == {'trading_date', 'flags'} and isinstance(r['flags'], list)
             and set(r['flags']) <= {WARNING} for r in quality), 'Separate exact warning-only metadata required')
    need([r['trading_date'] for r in quality] == dates, 'Complete ordered original XOM quality ledger required')
    warnings = tuple(r['trading_date'] for r in quality if r['flags'])
    need(warnings == tuple(expected_xom_warning_dates), 'Frozen XOM 2 bp warnings changed or erased')
    n = len(dates)
    values = {'intrinio_security_id': pa.array([XOM_KEY]*n), 'trading_date': raw['trading_date'],
        'frequency': pa.array(['daily']*n), 'intraperiod': pa.array([False]*n),
        'price_quality_status': pa.array(['accepted']*n),
        'raw_ohlc_bounds_valid': pa.array([True]*n), 'adjusted_ohlc_bounds_valid': pa.array([True]*n)}
    for field in ('open', 'high', 'low', 'close', 'volume'):
        values[field] = raw[field]; values['adj_' + field] = adjusted[field]
    for field in ('factor', 'split_ratio', 'dividend'):
        values[field] = pa.nulls(n, type=pa.float64())
    sip = pa.Table.from_arrays([pc.cast(values[f.name], f.type, safe=True) for f in PRICE_SCHEMA], schema=PRICE_SCHEMA)
    bk = _core(bk_quotes)
    ledger = [{'intrinio_security_id': BK_KEY, 'trading_date': d, 'provider': 'intrinio',
        'provider_intrinio_security_id': BK_KEY, 'source_key': BK_KEY, 'adjustment': 'published_adjusted',
        'provider_quality_available': True, 'retained_precision_warning': False}
        for d in bk['trading_date'].to_pylist()]
    ledger.extend({'intrinio_security_id': XOM_KEY, 'trading_date': d, 'provider': 'Alpaca/SIP',
        'provider_intrinio_security_id': None, 'source_key': XOM_SOURCE, 'adjustment': 'raw/all_pair',
        'provider_quality_available': False, 'retained_precision_warning': d in set(warnings),
        'computational_base_quality_policy': QUALITY_POLICY,
        'factor_dividend_split_unavailable': True} for d in dates)
    return pa.concat_tables([bk, sip]), tuple(ledger)


@dataclass(frozen=True)
class ShadowPriceViews:
    feature_prices: pa.Table
    outcome_prices: pa.Table
    feature_history_raw: pa.Table
    outcome_history_raw: pa.Table
    provenance: dict


def compose_price_views(base_feature_history, base_outcome_history, candidates, routing, *,
                        raw_start, feature_end, raw_end, sessions, outcome_dates,
                        original_requested_ids, max_rows=2_100_000):
    """Append only reviewed candidate histories to corrected original caches.

    Separate raw views retain original cache rows, including blocked identities;
    the builder views contain only preplanned computational IDs and dates. No
    candidate is selected by future coverage. Calendar-only 2024 evidence is
    allowed in sessions, never in a price row or financial result.
    """
    need(isinstance(routing, RoutingResult), 'Metadata-qualified complete cohorts required first')
    need(day(raw_start) <= day(feature_end) < day(raw_end), 'Explicit annual feature/outcome boundaries required')
    known = {day(d, financial=False) for d in sessions}
    permitted = {day(d) for d in outcome_dates}
    need(permitted and min(permitted) >= raw_start and max(permitted) == raw_end and permitted <= known,
         'Exact independent outcome/anchor dates required')
    original_ids = set(original_requested_ids)
    _quote_keys(base_feature_history, start=raw_start, end=feature_end, allowed_ids=original_ids, sessions=known)
    _quote_keys(base_outcome_history, start=raw_start, end=raw_end, allowed_ids=original_ids, sessions=known)
    # The authenticated normalized candidate bundle spans the study. The annual
    # calendar covers just that year's warmup. First clip in Arrow, then require
    # its returned annual rows to be exact members of the annual session grid.
    _quote_keys(candidates, start='2007-07-02', end='2023-12-29', allowed_ids={BK_KEY, XOM_KEY})
    bases = [base_feature_history, base_outcome_history]
    outputs = []
    added_counts = []
    # No old-source price-level splice is possible even if an unexpectedly
    # populated original XOM/BK route appears in a future cache version.
    for base, end in zip(bases, (feature_end, raw_end), strict=True):
        keys = _quote_keys(base, start=raw_start, end=end, allowed_ids=original_ids, sessions=known)
        need(not ({sid for sid, _ in keys} & {BK_OLD, XOM_OLD}),
             'Original empty route unexpectedly has bars; explicit overlap/units review required')
        extra = candidates.filter(pc.and_(pc.greater_equal(candidates['trading_date'], raw_start),
                                          pc.less_equal(candidates['trading_date'], end)))
        need(not set(keys) & set(_quote_keys(extra, start=raw_start, end=end, sessions=known)),
             'Reviewed candidate overlaps accepted cache key; no automatic replacement')
        combined = pa.concat_tables([_core(base), extra])
        need(combined.num_rows <= max_rows, 'Annual quote row cap exceeded; never truncate')
        outputs.append(combined); added_counts.append(extra.num_rows)
    feature_raw, outcome_raw = outputs
    ids = pa.array(routing.requested_ids, type=pa.string())
    feature = feature_raw.filter(pc.is_in(feature_raw['intrinio_security_id'], value_set=ids))
    outcome = outcome_raw.filter(pc.and_(pc.is_in(outcome_raw['intrinio_security_id'], value_set=ids),
        pc.is_in(outcome_raw['trading_date'], value_set=pa.array(sorted(permitted)))))
    return ShadowPriceViews(feature, outcome, feature_raw, outcome_raw, {
        **routing.provenance, 'base_source': 'authenticated_corrected_annual_caches_only',
        'candidate_appended_rows': dict(zip(('feature', 'outcome'), added_counts)),
        'base_values_replaced': 0, 'raw_source_fallback': False,
        'feature_end': feature_end, 'outcome_end': raw_end,
        'separate_daily_feature_and_outcome_views': True,
        'XOM_precision_warning_policy': QUALITY_POLICY,
        'XOM_original_warning_ledger_required_separately': True,
        'provider_quality_fabricated': False, 'computational_quality_policy_explicit': True,
        'source_factor_or_dividend_applied_again': False, 'missing_values_filled': False,
        'financial_2024_plus_returned': False, 'fits_preprocessing_inference_backtests': 0})
