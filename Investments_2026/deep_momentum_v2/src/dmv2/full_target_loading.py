"""Cache-only annual monthly targets, retaining the original signal cohorts.

Planning reads cohort/calendar metadata and Parquet footers, never quote values.
Materialization projects six quote columns, then removes warmup and nonexchange
sessions in Arrow before the existing exact-endpoint monthly worker sees values.
No cache miss can trigger a source read, request replacement or survivor filter.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .catalog import canonical_sha256
from .full_panel import _verify_artifacts, _within
from .identity_policy import PROVISIONAL_IDENTITY_POLICY, qualify_identity_records
from .monthly import REQUIRED_PRICE_COLUMNS, MonthlyReturnPanel, build_month_windows, build_monthly_returns
from .targets import COHORT_FIELDS, MonthlyTargetPanel, build_monthly_targets, target_month_fingerprint
from .universe import DataContractError, EXCLUDED_SCHEMA, UniverseSnapshot, iso_date

KEYS = ('source_security_id', 'signal_date')
META_ARTIFACTS = ('original_cohort', 'feature_coverage', 'exchange_calendar')
USED_ARTIFACTS = (*META_ARTIFACTS, 'daily_history_raw')
IDENTITY_FIELDS = ('intrinio_security_id', 'identity_status', 'source_identity_status',
                   'identity_eligible', 'identity_eligibility_policy', 'identity_eligibility_reason',
                   'identity_review_required', 'resolution_status')


@dataclass(frozen=True)
class CachedTargetYearPlan:
    directory: Path
    year: int
    audit: dict
    calendar: pa.Table
    windows: tuple
    original_cohort: pa.Table
    feature_coverage: pa.Table
    snapshots: tuple
    requested_ids: tuple[str, ...]
    read_plan: dict
    provenance: dict
    fingerprint: str


@dataclass(frozen=True)
class CachedTargetYear:
    monthly: MonthlyReturnPanel
    targets: MonthlyTargetPanel
    plan: CachedTargetYearPlan
    loading_stats: dict
    provenance: dict


def _year(value):
    if type(value) is not int or not 2010 <= value <= 2023:
        raise DataContractError('Cached target outcome year must be 2010–2023')
    return value


def _required(table, columns, name):
    if len(table.column_names) != len(set(table.column_names)) or not set(columns) <= set(table.column_names):
        raise DataContractError(name + ' is missing required or has duplicate columns')


def _path(directory, artifact):
    path = _within(directory, directory / artifact['path'])
    if not path.is_file():
        raise DataContractError('Required cached target input is absent: ' + artifact['path'])
    if path.stat().st_size != artifact['size_bytes']:
        raise DataContractError('Full-panel artifact changed: ' + artifact['path'])
    return path


def _reference_year(reference_run, year):
    if not isinstance(reference_run, dict) or not {'directory', 'audit'} <= set(reference_run):
        raise DataContractError('Supply an explicitly verified feature run')
    parent = Path(reference_run['directory']).resolve()
    root_audit = reference_run['audit']
    if root_audit.get('status') != 'COMPLETE' or str(year) not in root_audit.get('years', {}):
        raise DataContractError('Verified feature run has no complete requested year')
    item = root_audit['years'][str(year)]
    _verify_artifacts(parent, {str(year): item})
    path = _path(parent, item)
    directory = path.parent
    audit = json.loads(path.read_text())
    if audit.get('year') != year or audit.get('build_identity_sha256') != root_audit.get('build_identity_sha256'):
        raise DataContractError('Annual feature audit changed its year or recipe')
    if not set(USED_ARTIFACTS) <= set(audit.get('artifacts', {})):
        raise DataContractError('Annual cache lacks a required target input')
    return directory, audit, item['sha256']


def _plan_fingerprint(plan):
    return canonical_sha256({
        'year': plan.year, 'directory': str(plan.directory),
        'audit': plan.audit, 'requested_ids': list(plan.requested_ids),
        'calendar': target_month_fingerprint(plan.calendar),
        'original_cohort': target_month_fingerprint(plan.original_cohort),
        'feature_coverage': target_month_fingerprint(plan.feature_coverage),
        'windows': [{'signal_date': w.signal_date, 'outcome_month': w.outcome_month,
                     'outcome_start': w.outcome_start, 'outcome_end': w.outcome_end,
                     'expected_outcome_sessions': list(w.expected_outcome_sessions)} for w in plan.windows],
        'snapshots': [{'as_of': s.as_of, 'universe': s.universe,
                       'table': target_month_fingerprint(s.table), 'provenance': s.provenance}
                      for s in plan.snapshots],
        'read_plan': plan.read_plan, 'provenance': plan.provenance})


def plan_cached_target_year(reference_run, year) -> CachedTargetYearPlan:
    """Plan exactly twelve outcomes from immutable cached metadata/footers.

    Every identity-admitted original member must be in the pre-existing ID
    request, even when its cache contains no quotes. Quote inventory is never
    consulted to form cohorts, qualify identities or construct requests.
    """
    year = _year(year)
    directory, audit, annual_pin = _reference_year(reference_run, year)
    artifacts = audit['artifacts']
    _verify_artifacts(directory, {n: artifacts[n] for n in META_ARTIFACTS})
    original = pq.read_table(_path(directory, artifacts['original_cohort']))
    # Only signal-time metadata is needed; never read the feature matrix.
    coverage_columns = tuple(dict.fromkeys((*COHORT_FIELDS, *IDENTITY_FIELDS)))
    coverage = pq.read_table(_path(directory, artifacts['feature_coverage']), columns=list(coverage_columns))
    calendar = pq.read_table(_path(directory, artifacts['exchange_calendar']))
    _required(original, (*KEYS, 'as_of', 'intrinio_security_id', 'identity_status'), 'Original cohort')
    _required(coverage, coverage_columns, 'Feature coverage')
    windows = build_month_windows(calendar, start_month=f'{year}-01', end_month=f'{year}-12')
    signals = tuple(w.signal_date for w in windows)
    provenance = audit.get('feature_provenance', {})
    if (provenance.get('mode') != 'historical_reconstruction'
            or provenance.get('identity_policy') != PROVISIONAL_IDENTITY_POLICY
            or provenance.get('signal_dates') != list(signals)):
        raise DataContractError('Annual cache must preserve the declared provisional signal-time contract')
    declared_windows = {w.signal_date: {'outcome_month': w.outcome_month,
        'outcome_start': w.outcome_start, 'outcome_end': w.outcome_end} for w in windows}
    if provenance.get('target_windows') != declared_windows:
        raise DataContractError('Cached target bounds differ from the independent exchange calendar')
    raw_requests = provenance.get('requested_ids')
    if (not isinstance(raw_requests, list) or not raw_requests
            or any(not isinstance(v, str) or not v.strip() or v != v.strip() for v in raw_requests)
            or len(raw_requests) != len(set(raw_requests))):
        raise DataContractError('Annual cache requires a nonempty unique pre-price request declaration')
    requests = tuple(raw_requests)
    loading = audit.get('loading_provenance', {})
    reference_loading = loading.get('reference_loading_provenance', loading)
    executed = reference_loading.get('stages', {}).get('raw_daily_prices', {}).get('request', {})
    if (executed.get('filters', {}).get('intrinio_security_id') != raw_requests
            or executed.get('end') != windows[-1].outcome_end
            or not isinstance(executed.get('start'), str)
            or executed['start'] > windows[0].signal_date):
        raise DataContractError('Forward outcome cache does not match its actual executed ID/date request')
    original_rows = original.to_pylist()
    coverage_rows = coverage.to_pylist()
    original_keys = [(r['source_security_id'], r['signal_date']) for r in original_rows]
    coverage_keys = [(r['source_security_id'], r['signal_date']) for r in coverage_rows]
    if (not original_keys or len(set(original_keys)) != len(original_keys)
            or original_keys != coverage_keys or {k[1] for k in original_keys} != set(signals)):
        raise DataContractError('Cached original and feature ledgers must preserve every ordered signal key')
    qualified = qualify_identity_records(original_rows, identity_policy=PROVISIONAL_IDENTITY_POLICY)
    for original_row, admitted, feature in zip(original_rows, qualified, coverage_rows, strict=True):
        if original_row['as_of'] != original_row['signal_date']:
            raise DataContractError('Original membership cutoff differs from the signal date')
        for field in IDENTITY_FIELDS:
            if feature[field] != admitted.get(field):
                raise DataContractError('Feature ledger changed original identity metadata: ' + field)
        for field in ('identity_eligible', 'requested', 'paper_prediction_eligible'):
            if type(feature[field]) is not bool:
                raise DataContractError('Feature ledger requires explicit signal-time booleans')
        if feature['requested'] != (admitted.get('intrinio_security_id') in requests):
            # Existing feature ledgers mark unresolved routes unrequested even
            # when that native ID was requested for another historical date.
            if feature['identity_eligible'] or feature['requested']:
                raise DataContractError('Past request flag differs from the pre-price ID declaration')
        if admitted['identity_eligible'] and admitted['intrinio_security_id'] not in requests:
            raise DataContractError('An admitted original member was omitted from the pre-price request')
    snapshots = []
    for signal in signals:
        table = original.filter(pc.equal(original['signal_date'], signal))
        snapshots.append(UniverseSnapshot(table, pa.Table.from_pylist([], schema=EXCLUDED_SCHEMA),
            signal, 'combined', {'mode': 'historical_reconstruction', 'knowledge_time_verified': False,
                'original_signal_cohort_preserved': True, 'annual_feature_audit_sha256': annual_pin},
            {'constituents': table.num_rows}))
    price_path = _path(directory, artifacts['daily_history_raw'])
    footer = pq.ParquetFile(price_path).metadata
    schema = pq.read_schema(price_path)
    if not set(REQUIRED_PRICE_COLUMNS) <= set(schema.names):
        raise DataContractError('Cached prices lack required monthly quote columns')
    projected_bytes = 0
    metadata_projection_bytes = 0
    for group_number in range(footer.num_row_groups):
        group = footer.row_group(group_number)
        for column_number in range(group.num_columns):
            column = group.column(column_number)
            if column.path_in_schema in REQUIRED_PRICE_COLUMNS:
                projected_bytes += column.total_compressed_size
            if column.path_in_schema in ('intrinio_security_id', 'trading_date'):
                metadata_projection_bytes += column.total_compressed_size
    read_plan = {'metadata_files': 4, 'data_files': 1, 'price_metadata_passes': 1,
        'scanned_rows': footer.num_rows, 'returned_rows_upper_bound': footer.num_rows,
        'selected_row_groups': footer.num_row_groups,
        'projected_compressed_bytes': projected_bytes + metadata_projection_bytes,
        'price_projection_compressed_bytes': projected_bytes,
        'price_metadata_projection_compressed_bytes': metadata_projection_bytes,
        'price_columns': list(REQUIRED_PRICE_COLUMNS), 'outcome_months': 12,
        'original_rows': original.num_rows, 'paper_qualified_rows': sum(r['paper_prediction_eligible'] for r in coverage_rows),
        'identity_eligible_rows': sum(r['identity_eligible'] for r in coverage_rows),
        'signal_dates': list(signals), 'raw_cache_rows': footer.num_rows, 'daily_cache_rows': footer.num_rows,
        'metadata_count': 4, 'estimated_peak_memory_bytes': footer.num_rows * 2048,
        'financial_values_read': False,
        'semantics': 'Conservative whole annual column-chunk bytes including the repeated date/ID metadata pass; date filtering happens in Arrow before Python conversion.'}
    detached = {'year': year, 'annual_feature_audit_sha256': annual_pin,
        'feature_build_identity_sha256': audit['build_identity_sha256'],
        'used_artifacts': {name: artifacts[name] for name in USED_ARTIFACTS},
        'identity_policy': PROVISIONAL_IDENTITY_POLICY, 'universe': 'combined',
        'request_selection': 'pre_price_annual_feature_request_declaration',
        'original_keys_sha256': canonical_sha256([list(key) for key in original_keys]),
        'raw_source_reads': False, 'financial_values_read': False}
    plan = CachedTargetYearPlan(directory, year, audit, calendar, windows, original,
        coverage, tuple(snapshots), requests, read_plan, detached, '')
    return CachedTargetYearPlan(**{**plan.__dict__, 'fingerprint': _plan_fingerprint(plan)})


def build_cached_target_year(reference_run, year, *, plan=None) -> CachedTargetYear:
    """Materialize one authorized pre-holdout year using only the local cache."""
    year = _year(year)
    plan = plan_cached_target_year(reference_run, year) if plan is None else plan
    if not isinstance(plan, CachedTargetYearPlan) or plan.year != year or _plan_fingerprint(plan) != plan.fingerprint:
        raise DataContractError('Cached target plan changed after metadata planning')
    directory, audit, annual_pin = _reference_year(reference_run, year)
    if directory != plan.directory or annual_pin != plan.provenance['annual_feature_audit_sha256']:
        raise DataContractError('Cached target plan belongs to a different annual reference')
    used = {name: audit['artifacts'][name] for name in USED_ARTIFACTS}
    _verify_artifacts(directory, used)
    path = _path(directory, used['daily_history_raw'])
    # Validate full date/ID metadata before decoding the adjusted-close column.
    metadata = pq.read_table(path, columns=['intrinio_security_id', 'trading_date'])
    ids = set(metadata['intrinio_security_id'].to_pylist())
    if not ids <= set(plan.requested_ids):
        raise DataContractError('Cached quote IDs fall outside the pre-price request')
    dates = metadata['trading_date'].to_pylist()
    for day in set(dates):
        if iso_date(day, 'cached price date') >= '2024-01-01':
            raise DataContractError('Protected 2024+ quote metadata cannot enter target arithmetic')
    prices = pq.read_table(path, columns=list(REQUIRED_PRICE_COLUMNS))
    if not prices.select(['intrinio_security_id', 'trading_date']).equals(metadata):
        raise DataContractError('Cached quote metadata changed between projection passes')
    permitted = sorted({day for window in plan.windows
                        for day in (window.signal_date, *window.expected_outcome_sessions)})
    # Source holiday/weekend quotes are retained in immutable raw artifacts,
    # but cannot become arithmetic endpoints or expected-session evidence.
    in_request_bounds = pc.and_(pc.greater_equal(prices['trading_date'], permitted[0]),
                               pc.less_equal(prices['trading_date'], permitted[-1]))
    within_bounds = prices.filter(in_request_bounds)
    valid_session = pc.is_in(within_bounds['trading_date'], value_set=pa.array(permitted))
    selected = within_bounds.filter(valid_session)
    off_session_dates = sorted(set(within_bounds.filter(pc.invert(valid_session))['trading_date'].to_pylist()))
    monthly = build_monthly_returns(selected, plan.calendar, plan.snapshots,
        start_month=f'{year}-01', end_month=f'{year}-12',
        requested_security_ids=plan.requested_ids, mode='historical_reconstruction',
        identity_policy=PROVISIONAL_IDENTITY_POLICY)
    if not monthly.table.select(list(KEYS)).equals(plan.feature_coverage.select(list(KEYS)), check_metadata=False):
        # Arrow string and large-string storage differ across existing workers.
        if monthly.table.select(list(KEYS)).to_pylist() != plan.feature_coverage.select(list(KEYS)).to_pylist():
            raise DataContractError('Monthly return worker changed the original signal key order')
    targets = build_monthly_targets(plan.feature_coverage, monthly,
        label_observed_through=plan.windows[-1].outcome_end)
    if targets.table.select(list(KEYS)).to_pylist() != plan.original_cohort.select(list(KEYS)).to_pylist():
        raise DataContractError('Target worker changed original signal keys')
    _verify_artifacts(directory, used)
    stats = {**plan.read_plan, 'financial_values_read': True,
        'returned_rows': selected.num_rows,
        'discarded_warmup_rows': prices.num_rows - within_bounds.num_rows,
        'discarded_offsession_rows': within_bounds.num_rows - selected.num_rows,
        'discarded_offsession_dates': off_session_dates,
        'price_metadata_rows': metadata.num_rows,
        'monthly_return_rows': monthly.table.num_rows, 'target_rows': targets.table.num_rows}
    provenance = {**plan.provenance, 'plan_fingerprint': plan.fingerprint,
        'financial_values_read': True, 'cache_artifacts_verified_before_and_after': True,
        'financial_columns_projected': list(REQUIRED_PRICE_COLUMNS),
        'raw_daily_artifacts_modified': False, 'features_recalculated': False,
        'model_fitted': False, 'knowledge_time_verified': False, 'accounting_certified': False,
        'monthly_return_status_counts': dict(Counter(monthly.table['return_status'].to_pylist()))}
    return CachedTargetYear(monthly, targets, plan, stats, provenance)
