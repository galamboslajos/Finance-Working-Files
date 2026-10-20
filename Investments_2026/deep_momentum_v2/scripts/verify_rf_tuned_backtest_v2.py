#!/usr/bin/env python3
"""Independent zero-fit audit of the additive tuned-RF daily replay.

Only the previously reviewed independent cash/units equations are reused. This
script never imports production selection/accounting, an estimator, or a model
serializer. The existing forecast/model verification remains its own authority;
this audit authenticates that receipt and rechecks forecast-to-execution flow.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
import itertools
import json
import os
from pathlib import Path
import platform
import resource
import time

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

BASE_SHA = '16c3651289258bed24a83d500f6b8589fac3bcb4d358e516fe981acec527a794'
ORIGINAL_VERIFIER_SHA = 'b0b8f6ad39c82b33a7546210baa3cc696509c2ce44144a93d81a0fff262a48d0'
SOURCE_IDENTITY = 'sha256:559ba2f7fbe187f192b40af2ab8fbe0a4fd70a1b843e2cbd0503f7e35729fcae'
ARM = 'rf_extended_tuned'
BASKETS = ('class_1_10', 'class_2_9')
VARIANTS = ('unstopped', 'asset_stop_10pct')
FEES = (0, 5, 10, 25)
KEYS = ['source_security_id', 'signal_date']
PROBABILITIES = [f'probability_{k}' for k in range(1, 11)]
FORECAST_FIELDS = [*KEYS, 'intrinio_security_id', 'prediction_eligible',
                   'paper_prediction_eligible', *PROBABILITIES]
FORECAST_METADATA = [*KEYS, 'intrinio_security_id', 'prediction_eligible',
                     'paper_prediction_eligible', 'identity_eligible', 'target_available',
                     'outcome_month', 'outcome_start', 'outcome_end']
LEDGERS = ('daily', 'trades', 'holdings', 'failures', 'stops', 'action_events',
           'market_trades', 'orders', 'identity_handovers')


def load_equations(root):
    from hashlib import sha256
    path = Path(root) / 'scripts/verify_walkforward_backtest_v2.py'
    if sha256(path.read_bytes()).hexdigest() != BASE_SHA:
        raise RuntimeError('Previously reviewed independent equation script changed')
    spec = importlib.util.spec_from_file_location('_tuned_rf_independent_equations', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def same_pin(v, root, left, right):
    return (v.local(root,left['path']) == v.local(root,right['path'])
            and left['size_bytes'] == right['size_bytes'] and left['sha256'] == right['sha256'])


def verify_grid(v, comparison, yearly, coverage):
    expected = set(itertools.product((ARM,), BASKETS, VARIANTS, FEES))
    columns = ['arm', 'basket', 'variant', 'cost_bps']
    v.require(len(comparison) == 16 and not comparison.duplicated(columns).any()
              and set(map(tuple, comparison[columns].to_numpy())) == expected,
              'Exact sixteen tuned-RF scenarios required; no comparator replay')
    v.require(len(yearly) == 64 and not yearly.duplicated([*columns, 'year']).any()
              and set(map(tuple, yearly[[*columns, 'year']].to_numpy())) ==
              {(*item, year) for item in expected for year in range(2020, 2024)},
              'Exact continuous four-year scenario slices required')
    v.require(len(coverage) == 96 and set(coverage.arm) == {ARM}
              and set(coverage.basket) == set(BASKETS)
              and not coverage.duplicated(['arm', 'basket', 'signal_date']).any(),
              'Exactly two 48-month coverage records required')
    v.require(comparison.complete_proxy_path.map(
        lambda item: isinstance(item, (bool, np.bool_))).all(),
        'Explicit complete/withheld states required for every scenario')


def verify_forecast_metadata(v, keys, year, windows):
    """Run before a probability projection, including strict admission types."""
    required = set(KEYS) | {'intrinio_security_id', 'prediction_eligible',
                            'paper_prediction_eligible', 'outcome_month',
                            'outcome_start', 'outcome_end'}
    v.require(required <= set(keys), 'Forecast causal metadata missing')
    v.require(not keys.duplicated(KEYS).any(), 'Duplicate original forecast key')
    chosen_windows = windows.loc[windows.outcome_month.str[:4].eq(str(year))]
    expected = {row.signal_date: row for row in chosen_windows.itertuples()}
    v.require(len(expected) == 12 and set(keys.signal_date) == set(expected),
              'Forecast lost a month or entered another outer year')
    for field in ('prediction_eligible', 'paper_prediction_eligible'):
        v.require(keys[field].map(lambda x: isinstance(x, (bool, np.bool_))).all(),
                  'Admission mask is not strict Boolean')
    v.require(keys.prediction_eligible.equals(keys.paper_prediction_eligible),
              'Expanded missingness changed the original Paper15 prediction cohort')
    for row in keys.itertuples(index=False):
        window = expected[row.signal_date]
        v.require(isinstance(row.source_security_id, str) and bool(row.source_security_id.strip()),
                  'Missing original stable source key')
        v.require(row.outcome_month == window.outcome_month
                  and row.outcome_start == window.outcome_start
                  and row.outcome_end == window.outcome_end
                  and row.signal_date < row.outcome_start <= row.outcome_end,
                  'Forecast timing differs from independent monthly execution clock')
        if row.prediction_eligible:
            v.require(isinstance(row.intrinio_security_id, str)
                      and bool(row.intrinio_security_id.strip()),
                      'Eligible forecast has no unique native identity')
    eligible = keys.loc[keys.prediction_eligible]
    v.require(not eligible.duplicated(['signal_date', 'intrinio_security_id']).any(),
              'Same-signal eligible source/provider collision')
    return {'original_rows': len(keys), 'prediction_rows': int(keys.prediction_eligible.sum()),
            'months': 12, 'target_numeric_values_decoded': False}


def read_forecast(v, path, year, windows, read, *, verified_metadata=None):
    pf = pq.ParquetFile(path)
    names = set(pf.schema_arrow.names)
    v.require(pf.metadata.num_rows <= 12_000 and set(FORECAST_FIELDS) <= names,
              'Bounded original forecast schema required')
    v.require(set(FORECAST_METADATA) <= names, 'Forecast metadata projection missing')
    keys = read(path, FORECAST_METADATA) if verified_metadata is None else verified_metadata
    report = verify_forecast_metadata(v, keys, year, windows)
    # Deliberately omit target_class, returns, observed-label masks and quality
    # outcomes. Their values can never affect independently expected baskets.
    values = read(path, FORECAST_FIELDS)
    admitted = values.prediction_eligible
    p = values.loc[admitted, PROBABILITIES].to_numpy(dtype=float)
    v.require(np.isfinite(p).all() and (p >= 0).all()
              and np.allclose(p.sum(axis=1), 1, atol=1e-9, rtol=1e-9),
              'Native forecast simplex changed')
    v.require(values.loc[~admitted, PROBABILITIES].isna().all().all(),
              'Signal-ineligible rows unexpectedly contain probabilities')
    return values, report


def verify_training_manifest(v, prediction, prepare, fit, winner, year, windows):
    """Recheck ten-year cutoffs from pinned metadata without matrix/model reads."""
    first = windows.loc[windows.outcome_month.eq(f'{year}-01')].iloc[0]
    v.require(prediction.get('status') == fit.get('status') == prepare.get('status') == 'COMPLETE'
              and prediction.get('outer_year') == fit.get('outer_year') == prepare.get('outer_year') == year
              and prediction.get('candidate_id') == fit.get('candidate_id') == winner.get('candidate_id')
              and fit.get('stage') == prepare.get('stage') == 'outer',
              'Annual selected forecast/fit/preparation/winner metadata binding changed')
    provenance = prepare.get('train_provenance', {})
    v.require(provenance.get('train_start') == f'{year-10}-01-01'
              and provenance.get('train_end') == f'{year-1}-12-31'
              and provenance.get('label_observed_through') == first.signal_date
              and first.signal_date < first.outcome_start
              and provenance.get('future_financial_rows_converted') is False
              and prediction.get('estimator_fits') == prediction.get('preprocessing_fits') == 0,
              'Annual forecast has changed ten-year training or causal label cutoff')
    v.require(fit.get('probability_class_ids') == list(range(1, 11))
              and prediction.get('prediction_provenance', {}).get('class_ids') == list(range(1, 11)),
              'Canonical best-to-worst probability columns changed')


class ProjectionBudget:
    """Conservative compressed file bytes charged before each Parquet decode.

    Counts full file size for every projection rather than claiming row-group
    compression precision; this upper bound also includes date-key scans.
    """
    def __init__(self, v, prior=0, limit=512 * 1024**2):
        self.v, self.prior, self.limit = v, int(prior), int(limit)
        self.used = 0

    def read(self, path, columns=None):
        path = Path(path)
        charge = path.stat().st_size
        self.v.require(self.prior + self.used + charge <= self.limit,
                       'Cumulative projected compressed verification read budget exceeded')
        self.used += charge
        return pq.ParquetFile(path).read(columns=columns).to_pandas()


def reserve_verification(v, run, amount):
    """Reserve the entire independent phase before its first table decode."""
    from datetime import datetime, timezone
    path = run / 'read_usage.json'
    usage = v.json_file(path)
    entries = usage.get('reservations')
    v.require(isinstance(entries, list) and all(isinstance(item, dict) for item in entries),
              'Cumulative read ledger missing')
    v.require(all(item.get('phase') != 'independent_verification_v2' for item in entries),
              'Preserve prior verification attempt; no automatic repeated decode')
    prior = sum(item['projected_compressed_bytes'] for item in entries)
    v.require(prior == usage.get('total_projected_compressed_bytes')
              and usage.get('limit_bytes') == 512 * 1024**2
              and isinstance(amount, int) and amount >= 0
              and prior + amount <= usage['limit_bytes'],
              'Cumulative decode reservation would exceed frozen limit')
    entries.append({'phase': 'independent_verification_v2',
                    'projected_compressed_bytes': amount,
                    'reserved_at': datetime.now(timezone.utc).isoformat()})
    usage['total_projected_compressed_bytes'] = prior + amount
    temporary = path.with_suffix('.json.independent.partial')
    with temporary.open('x') as stream:
        json.dump(usage, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)
    return prior


def verification_read_upper_bound(v, root, identity, backtest, old_backtest):
    """Full compressed file sizes, including repeated key/number projections."""
    entries = identity['forecast_entries']
    files = []
    for item in entries:
        files += [(v.local(root, item['predictions_pin']['path']), 2)]
        files += [(v.local(root, item['original_rf_predictions_pin']['path']), 1)]
        # The original first-year forecast and reference integration controls
        # are included even if the guarded equality test is ultimately false.
        if item['outer_year'] == 2020:
            files += [(v.local(root, item['original_rf_predictions_pin']['path']), 1)]
    reference_run = v.local(root, identity['source_reference_directory'])
    for year in range(2020, 2024):
        files += [(reference_run / 'years' / str(year) / 'daily_prices_raw.parquet', 2)]
    phase = v.json_file(backtest / 'build_identity.json')
    files += [(v.local(root, item['pin']['path']), 2)
              for item in phase['event_registry'].get('supplemental_quote_sources', [])]
    files += [(backtest / name, 2) for name in ('comparison.parquet', 'yearly.parquet', 'selection_coverage.parquet')]
    for basket in BASKETS:
        directory = backtest / 'selections' / ARM / basket
        files += [(directory / (name + '.parquet'), 2)
                  for name in ('original_ledger', 'selected', 'monthly', 'rebalances')]
        # Additional exact preflight/replay frame checks are charged separately
        # from the later source-based selection/accounting verification.
        files += [(directory / (name + '.parquet'), 2)
                  for name in ('original_ledger', 'selected', 'monthly', 'rebalances')]
        files += [(backtest.parent/'preflight/selections'/ARM/basket/(name+'.parquet'),2)
                  for name in ('original_ledger', 'selected', 'monthly', 'rebalances')]
        files += [(old_backtest / 'selections/rf_extended' / basket / 'selected.parquet', 2)]
        for variant, fee in itertools.product(VARIANTS, FEES):
            directory = backtest / 'arms' / ARM / basket / variant / f'cost_{fee:02d}bps'
            files += [(directory / (name + '.parquet'), 2) for name in (*LEDGERS, 'monthly')]
            original = old_backtest / 'arms/rf_extended' / basket / variant / f'cost_{fee:02d}bps'
            files += [(original / (name + '.parquet'), 2) for name in ('daily', 'trades', 'stops')]
    v.require(all(path.is_file() for path, _ in files), 'Missing exact scenario artifact before decode plan')
    return sum(path.stat().st_size * count for path, count in files)


def checked_table(v, budget, path, *, kind, columns=None, maximum_rows=2_000_000):
    """Check all time keys first, then decode only the explicit approved table."""
    pf = pq.ParquetFile(path)
    v.require(pf.metadata.num_rows <= maximum_rows, 'Bounded saved ledger rows exceeded')
    names = set(pf.schema_arrow.names)
    timing = [key for key in ('date', 'trading_date', 'signal_date', 'held_signal_date',
                              'execution_date') if key in names]
    v.require(timing or pf.metadata.num_rows == 0, 'Saved financial ledger lacks time keys: ' + kind)
    keys = budget.read(path, timing)
    for field in timing:
        low = '2019-12-01' if field in ('signal_date', 'held_signal_date') else '2020-01-01'
        v.require(keys[field].dropna().map(lambda x: low <= v.iso(x) <= '2023-12-31').all(),
                  'Protected-period key rejected before numeric ledger read: ' + kind)
    return budget.read(path, columns)


def monthly_table(v, budget, path):
    pf = pq.ParquetFile(path)
    v.require(pf.metadata.num_rows == 48 and 'outcome_month' in pf.schema_arrow.names,
              'Exactly 48 monthly report rows required')
    keys = budget.read(path, ['outcome_month'])
    v.require(keys.outcome_month.tolist() == [str(x) for x in pd.period_range('2020-01', '2023-12', freq='M')],
              'Monthly report escaped frozen outcome period')
    return budget.read(path)


def read_reports(v, budget, backtest):
    comparison_path=backtest/'comparison.parquet'
    bounds=budget.read(comparison_path,['outcome_start','outcome_end'])
    v.require(len(bounds)==16 and bounds.outcome_start.eq('2020-01-01').all()
              and bounds.outcome_end.eq('2023-12-31').all(),
              'Comparison report period rejected before numeric metric decode')
    yearly_path=backtest/'yearly.parquet'
    years=budget.read(yearly_path,['year'])
    v.require(len(years)==64 and set(years.year)=={2020,2021,2022,2023},
              'Annual metric report period rejected before numeric decode')
    return budget.read(comparison_path),budget.read(yearly_path)


def read_quotes(v, root, phase, reference_run, reference_identity, sessions, budget):
    """All annual/supplement key checks precede any price projection."""
    pins = {Path(item['path']).resolve() if Path(item['path']).is_absolute()
            else v.local(root, item['path']): item for item in phase['source_pins']}
    paths, original_ids = [], set()
    for year in range(2020, 2024):
        directory = reference_run / 'years' / str(year)
        audit_path = directory / 'audit.json'
        v.require(audit_path in pins, 'Reference execution annual audit unpinned')
        audit = v.json_file(v.pinned(root, pins[audit_path]))
        v.require(audit.get('status') == 'COMPLETE' and audit.get('year') == year
                  and audit.get('build_identity_sha256') == reference_identity,
                  'Reference execution annual identity/year changed')
        path = directory / 'daily_prices_raw.parquet'
        v.require(path in pins, 'Reference execution annual quotes unpinned')
        v.pinned(root, pins[path])
        v.require(pq.ParquetFile(path).metadata.num_rows <= 300_000, 'Bounded annual quote row limit')
        keys = budget.read(path, ['trading_date', 'intrinio_security_id'])
        v.require(keys.trading_date.map(lambda day: isinstance(day, str)
                    and f'{year}-01-01' <= day <= f'{year}-12-31').all()
                  and keys.trading_date.isin(sessions).all(),
                  'Annual date keys rejected before numeric quote decode')
        v.require(keys.intrinio_security_id.map(lambda value: isinstance(value, str) and bool(value.strip())).all()
                  and not keys.duplicated(['trading_date', 'intrinio_security_id']).any(),
                  'Null/duplicate native annual quote key rejected before numeric read')
        original_ids.update(audit['feature_provenance']['requested_ids'])
        original_ids.update(keys.intrinio_security_id)
        paths.append(path)
    registry = phase['event_registry']
    successors = {event['successor_id'] for event in registry['events'] if event.get('successor_id')}
    for source in registry.get('supplemental_quote_sources', []):
        path = v.pinned(root, source['pin'])
        allowed = set(source['allowed_native_ids'])
        v.require(path.is_relative_to(root / 'data/derived') and path.suffix == '.parquet'
                  and allowed and allowed <= successors and not allowed & original_ids
                  and '2020-01-01' <= source['start'] <= source['end'] <= '2023-12-31',
                  'Supplement changed native identity or approved period')
        keys = budget.read(path, ['trading_date', 'intrinio_security_id'])
        v.require(set(keys.intrinio_security_id) <= allowed
                  and keys.trading_date.between(source['start'], source['end']).all()
                  and keys.trading_date.isin(sessions).all(),
                  'Supplement keys rejected before numeric quote decode')
        paths.append(path)
    prices = pd.concat([budget.read(path, v.PRICE_FIELDS) for path in paths], ignore_index=True)
    v.require(not prices.duplicated(['trading_date', 'intrinio_security_id']).any(),
              'Annual/supplement quote overwrite')
    return prices


def verify_correction(v, root, run, path, expected_sha):
    """Separate immutable authority for one verification-only encoding repair."""
    path=v.local(run,path)
    v.require(path.is_file() and v.digest(path)==expected_sha,
              'Verification-only correction authority changed')
    correction=v.json_file(path)
    root_identity=v.json_file(run/'build_identity.json')
    v.require(correction.get('schema')=='dmv2-rf-tuned-backtest-verification-correction/v2'
              and correction.get('status')=='FROZEN_VERIFICATION_ONLY_AUTHORIZED'
              and correction.get('run_root')==str(run)
              and correction.get('root_identity_sha256')==v.canonical(root_identity)
              and correction.get('allowed_physical_change')=={'column':'side','from':'string','to':'large_string'}
              and correction.get('new_estimator_fits')==correction.get('new_preprocessor_fits')==correction.get('new_inference_calls')==0
              and correction.get('backtest_replays')==0
              and correction.get('verification_seconds_remaining')==1737
              and correction.get('diagnostic_projected_compressed_bytes')==2_973_088,
              'Correction exceeds the exact preserved verification-only scope')
    old=v.pinned(root,correction['old_verifier_pin'])
    new=v.pinned(root,correction['new_verifier_pin'])
    v.require(old==root/'scripts/verify_rf_tuned_backtest.py' and v.digest(old)==ORIGINAL_VERIFIER_SHA
              and new==Path(__file__).resolve(), 'Original or additive verifier code identity changed')
    pins=correction.get('failed_verification_pins')
    v.require(isinstance(pins,list) and len(pins)>=2, 'Preserved failed verification log/control receipts required')
    found=False
    for pin in pins:
        payload=v.pinned(root,pin)
        if payload.suffix=='.log':
            v.require(payload.stat().st_size<=1024**2, 'Unbounded failed-verification log')
            found=found or 'Replay changed a frozen preflight selection artifact' in payload.read_text()
    v.require(found,'Correction is not bound to the exact observed pre-numeric verifier failure')
    diagnostic=v.json_file(v.pinned(root,correction['diagnostic_pin']))
    v.require(diagnostic.get('compressed_file_read_upper_bound_bytes')==2_973_088
              and len(diagnostic.get('tables',[]))==8
              and all(item.get('pandas_exact')=='EXACT_EQUAL' and item.get('dtypes_equal') is True for item in diagnostic['tables']),
              'Bounded original diagnostic does not establish exact frame equality')
    usage=v.json_file(run/'read_usage.json')
    diagnostics=[item for item in usage['reservations'] if item.get('phase')=='verification_selection_serialization_diagnosis']
    v.require(len(diagnostics)==1 and diagnostics[0]['projected_compressed_bytes']==2_973_088
              and not any(item.get('phase')=='independent_verification' for item in usage['reservations']),
              'Diagnostic bytes must be charged exactly once; V1 failed before a decode reservation')
    return correction,{'path':str(path.relative_to(root)),'size_bytes':path.stat().st_size,'sha256':expected_sha}


def verify_preflight_encoding_metadata(v, first, second):
    """Admit only the observed side string-to-large_string serialization change."""
    a,b=pq.ParquetFile(first),pq.ParquetFile(second)
    v.require(a.metadata.num_rows==b.metadata.num_rows and a.metadata.num_rows<=60_000
              and a.schema_arrow.names==b.schema_arrow.names,
              'Preflight/replay row count or column order changed')
    changed=[]
    for x,y in zip(a.schema_arrow,b.schema_arrow):
        v.require(x.name==y.name and x.nullable==y.nullable and x.metadata==y.metadata,
                  'Preflight/replay field metadata or nullability changed')
        if x.type!=y.type:
            v.require(x.name=='side' and str(x.type)=='string' and str(y.type)=='large_string',
                      'Physical schema change exceeds the exact side string-width correction')
            changed.append(x.name)
    equal=v.digest(first)==v.digest(second)
    v.require(equal or changed==['side'],
              'Nonidentical serialization without the exact reviewed side-width change')
    return {'raw_bytes_equal':equal,'physical_fields_changed':changed,'rows':a.metadata.num_rows}


def verify_preflight_semantics(v,budget,run):
    results=[]
    for basket in BASKETS:
        for name in ('original_ledger','selected','monthly','rebalances'):
            rel=f'selections/{ARM}/{basket}/{name}.parquet'
            first,second=run/'preflight'/rel,run/'backtest'/rel
            metadata=verify_preflight_encoding_metadata(v,first,second)
            a=checked_table(v,budget,first,kind='exact preflight '+name,maximum_rows=60_000)
            b=checked_table(v,budget,second,kind='exact replay '+name,maximum_rows=60_000)
            # No sort, coercion, normalization, null filling, or tolerance.
            pd.testing.assert_frame_equal(a,b,check_exact=True,check_dtype=True,
                                          check_index_type=True,check_column_type=True,
                                          check_names=True,check_like=False)
            results.append({'basket':basket,'table':name,'exact_values_order_nulls_and_dtypes_equal':True,**metadata})
    return results


def authenticated_metadata(v, root, run):
    """Authenticate every authority before a single financial table read."""
    from importlib.metadata import version
    identity = v.json_file(run / 'build_identity.json')
    root_audit = v.json_file(run / 'audit.json')
    identity_hash = v.canonical(identity)
    v.require(identity.get('schema') == 'dmv2-rf-tuned-backtest-root/v1'
              and identity.get('run_root') == str(run) and identity.get('project_root') == str(root)
              and root_audit.get('status') == 'COMPLETE_PENDING_INDEPENDENT_VERIFICATION'
              and root_audit.get('build_identity_sha256') == identity_hash,
              'Completed fresh tuned-backtest root identity missing')
    cfg = v.json_file(v.pinned(root, identity['configuration_pin']))
    v.require(cfg == identity['configuration'] and cfg.get('schema') == 'dmv2-rf-tuned-backtest-execution/v1'
              and cfg.get('status') == 'FROZEN_EXECUTION_AUTHORIZED' and cfg.get('execution_authorized') is True
              and cfg.get('arm') == ARM and cfg.get('scenarios') == 16
              and cfg.get('new_estimator_fits') == cfg.get('new_preprocessor_fits') == cfg.get('new_inference_calls') == 0
              and cfg.get('financial_2024_plus_authorized') is False and cfg.get('cloud_operations') is False,
              'Separate fixed sixteen-scenario zero-fit authority changed')
    limits = {'rss_stop_bytes':4*1024**3, 'rss_warning_bytes':3*1024**3,
              'minimum_available_memory_bytes':7*1024**3, 'normal_pressure_dispatch':1,
              'cumulative_compressed_read_bytes':512*1024**2, 'output_bytes':256*1024**2,
              'backtest_wall_seconds':900, 'preflight_wall_seconds':300,
              'verification_wall_seconds':1800, 'workers':1, 'numeric_threads':1}
    v.require(cfg.get('resource_limits') == limits, 'Separate local resource protocol changed')
    base = v.json_file(v.pinned(root, cfg['base_configuration_pin']))
    v.require(base == identity['base_configuration']
              and base['backtest']['baskets'] == list(BASKETS)
              and base['backtest']['cost_bps'] == list(FEES)
              and base['backtest']['variants'] == {'unstopped':None,'asset_stop_10pct':.1}
              and base['backtest']['top_k_per_side'] == 10
              and base['backtest']['long_weight'] == .5 and base['backtest']['short_weight'] == -.5
              and set(base['execution_restrictions']['native_ids']) == v.EXCLUDED
              and base['accounting']['continuous_2020_2023_ledger'] is True
              and base['accounting']['annual_fit_forced_liquidation'] is False
              and base['accounting']['terminal_session'] == '2023-12-29'
              and base['accounting']['borrow_annual_bps'] == base['accounting']['funding_annual_bps'] == 0,
              'Frozen baskets/stops/costs/continuous cash protocol changed')
    expected_runtime = {'python':platform.python_version(), **{name:version(name) for name in
                        ('numpy','pandas','pyarrow','exchange-calendars','psutil','threadpoolctl')}}
    v.require(identity['runtime'] == expected_runtime, 'Recorded numeric runtime changed')
    for item in [*identity['code_pins'], *identity['source_pins'], identity['tuning_identity_pin'],
                 identity['tuning_ledger_pin'], identity['tuning_verification_pin']]: v.pinned(root,item)
    v.require(any(v.pinned(root,item) == root/'scripts/verify_rf_tuned_backtest.py'
                  and item['sha256']==ORIGINAL_VERIFIER_SHA for item in identity['code_pins']),
              'Original sealed independent verifier changed')
    tuning = v.json_file(v.pinned(root,identity['tuning_identity_pin']))
    payload = {key:value for key,value in tuning.items() if key not in ('identity_sha256','run_root','created_at')}
    receipt = v.json_file(v.pinned(root,identity['tuning_verification_pin']))
    ledger = v.json_file(v.pinned(root,identity['tuning_ledger_pin']))
    v.require(v.canonical(payload) == tuning['identity_sha256'] == receipt['build_identity_sha256']
              and receipt.get('status') == 'PASS_COMPLETE' and receipt.get('complete_search_verified') is True
              and receipt.get('runtime_verified') is True and receipt['run_root'] == tuning['run_root']
              and receipt.get('verified_fit_manifests') == 100
              and receipt.get('verified_inner_probability_ledgers') == 96
              and receipt.get('verified_selected_outer_model_reproductions') == 4
              and len(receipt.get('verified_prepared_splits',[])) == 16
              and same_pin(v,root,ledger.get('build_identity_pin',{}),identity['tuning_identity_pin'])
              and ledger.get('status') == 'COMPLETE_PENDING_INDEPENDENT_VERIFICATION'
              and ledger.get('initiated_estimator_fits') == 100 and ledger.get('initiated_preprocessor_fits') == 16
              and len(ledger.get('tasks',[])) == 120 and all(t['status']=='COMPLETE' for t in ledger['tasks']),
              'Exact complete-grid independent tuning authority missing')
    reference_run = v.local(root,identity['source_reference_directory'])
    old_phase_path = v.pinned(root,identity['execution_reference_phase_pin'])
    old_backtest = old_phase_path.parent
    v.require(old_backtest.parent == reference_run and not run.is_relative_to(reference_run)
              and not reference_run.is_relative_to(run), 'Fresh output overlaps the preserved reference')
    old_phase = v.json_file(old_phase_path)
    old_audit = v.json_file(v.pinned(root,identity['execution_reference_audit_pin']))
    old_identity = identity['execution_reference_root_identity_sha256']
    v.require(old_audit['status'] == 'COMPLETE' and old_audit['scenarios'] == 64
              and old_audit['phase_identity_sha256'] == v.canonical(old_phase)
              and old_audit['build_identity_sha256'] == old_phase['parent_identity_sha256'] == old_identity
              and old_identity == SOURCE_IDENTITY == cfg['source_reference_root_identity_sha256']
              and old_identity != identity_hash and old_phase['configuration_sha256'] == v.canonical(base),
              'Completed corrected comparator execution identity changed')
    for item in old_audit['artifacts'].values(): v.pinned(old_backtest,item)
    for name, expected in old_phase['workers'].items():
        v.require(v.digest(root/'src/dmv2'/name) == expected, 'Frozen original numerical worker changed: '+name)
    backtest = run/'backtest'
    phase = v.json_file(backtest/'build_identity.json'); audit = v.json_file(backtest/'audit.json')
    v.require(v.pinned(root,root_audit['backtest_audit_pin']) == backtest/'audit.json',
              'Fresh root completion pointer differs from the audited daily phase')
    v.require(phase.get('schema') == 'dmv2-rf-tuned-backtest-phase/v1'
              and phase.get('phase') == audit.get('phase') == 'rf_tuned_continuous_backtest_v1'
              and phase.get('implementation_revision') == audit.get('implementation_revision') == 'identity_route_failure_v2'
              and audit.get('status') == 'COMPLETE' and audit.get('scenarios') == 16
              and audit.get('sessions') == 1006 and audit.get('outcome_months') == 48
              and phase['parent_identity_sha256'] == audit['build_identity_sha256'] == identity_hash
              and audit['phase_identity_sha256'] == v.canonical(phase)
              and phase['configuration_sha256'] == v.canonical(cfg)
              and phase['forecast_entries'] == identity['forecast_entries']
              and phase['execution_reference_root_identity_sha256'] == old_identity
              and phase['source_reference_directory'] == identity['source_reference_directory']
              and phase['execution_reference_phase_pin'] == identity['execution_reference_phase_pin'],
              'Daily replay lost its sealed tuned-forecast/execution identity')
    v.require(phase.get('models_fitted') is False and phase.get('preprocessing_fitted') is False
              and audit.get('models_fitted') is False and audit.get('preprocessing_fitted') is False
              and phase.get('new_inference_calls') == audit.get('new_inference_calls') == 0
              and phase.get('financial_2024_plus_read') is audit.get('financial_2024_plus_read') is False
              and phase.get('preflight_before_performance') is audit.get('preflight_before_performance') is True
              and audit.get('annual_liquidation') is False
              and audit.get('source_identity_promoted') is False,
              'Backtest broadened fitting, clock, identity, or protected financial scope')
    for key in ('event_registry','historical_retirements','identity_handover_registry',
                'identity_handover_registry_pin','identity_handover_approval','clock','basket_policy'):
        v.require(phase[key] == old_phase[key], 'Original economic evidence/rule changed: '+key)
    new_pins={v.local(root,item['path']):item for item in [*identity['source_pins'],*phase['source_pins']]}
    for item in old_phase['source_pins']:
        path=v.local(root,item['path'])
        v.require(path in new_pins and same_pin(v,root,item,new_pins[path]),
                  'Original independently verified execution source pin was replaced or omitted')
        v.pinned(root,item)
    evidence_context = deepcopy(phase); evidence_context['parent_identity_sha256'] = old_identity
    handovers = v.review_handover_registry(root,evidence_context)
    v.require(len(handovers) == 1 and len(phase['event_registry']['events']) == 9
              and len(phase['historical_retirements']) == 11,
              'Unchanged single PENN handover, nine current and eleven prior events required')
    for item in phase['source_pins']: v.pinned(root,item)
    for item in audit['artifacts'].values(): v.pinned(backtest,item)
    pre = run/'preflight'; pre_audit = v.json_file(v.pinned(root,phase['preflight_audit_pin']))
    pre_phase = v.json_file(pre/'build_identity.json')
    v.require(pre_audit['status'] == 'COMPLETE' and pre_audit['build_identity_sha256'] == identity_hash
              and pre_audit['phase_identity_sha256'] == v.canonical(pre_phase)
              and pre_phase['forecast_entries'] == identity['forecast_entries']
              and pre_audit.get('performance_computed') is False,
              'Completed selection/coverage preflight missing before replay')
    for item in pre_audit['artifacts'].values(): v.pinned(pre,item)
    for basket in BASKETS:
        for name in ('original_ledger','selected','monthly','rebalances'):
            rel=f'selections/{ARM}/{basket}/{name}.parquet'
            verify_preflight_encoding_metadata(v,pre/rel,backtest/rel)
    dates, windows = v.independent_clock()
    entries = identity['forecast_entries']
    v.require([item['outer_year'] for item in entries] == list(range(2020,2024)),
              'Exactly four selected annual forecast ledgers required')
    source_pins={v.local(root,item['path']):item for item in identity['source_pins']}
    original_model_audit_path=reference_run/'models/audit.json'
    v.require(original_model_audit_path in source_pins, 'Original model audit is not preserved')
    original_model_audit=v.json_file(v.pinned(root,source_pins[original_model_audit_path]))
    for entry in entries:
        year = entry['outer_year']
        prediction = v.json_file(v.pinned(root,entry['prediction_manifest_pin']))
        prepare = v.json_file(v.pinned(root,entry['prepare_manifest_pin']))
        fit = v.json_file(v.pinned(root,entry['fit_manifest_pin']))
        winner = v.json_file(v.pinned(root,entry['winner_pin']))
        task=next(task for task in ledger['tasks'] if task['action']=='predict_outer' and task['outer_year']==year)
        task_receipt=v.json_file(v.pinned(root,task['receipt_pin']))
        selected=next(item for item in receipt['selected_winners'] if item['outer_year']==year)
        v.require(task_receipt['prediction_manifest'] == entry['prediction_manifest_pin']
                  and prediction['artifacts']['predictions'] == entry['predictions_pin']
                  and prediction['prepare_manifest'] == fit['prepare_manifest'] == entry['prepare_manifest_pin']
                  and prediction['fit_manifest'] == entry['fit_manifest_pin']
                  and prediction['winner_path'] == entry['winner_pin'] == ledger['winners'][str(year)]
                  and prediction['candidate_id'] == selected['candidate_id'],
                  'Forecast pins are not the verified selected outer checkpoint')
        for key in ('predictions_pin','original_rf_predictions_pin'): v.pinned(root,entry[key])
        fold_path=v.pinned(original_model_audit_path.parent,original_model_audit['folds'][str(year)])
        fold=v.json_file(fold_path)
        arm_path=v.pinned(fold_path.parent,fold['arms']['rf_extended'])
        arm=v.json_file(arm_path)
        original_prediction_path=v.pinned(arm_path.parent,arm['artifacts']['predictions'])
        v.require(original_prediction_path == v.local(root,entry['original_rf_predictions_pin']['path'])
                  and arm['artifacts']['predictions']['sha256'] == entry['original_rf_predictions_pin']['sha256'],
                  'Original RF comparison cohort detached from its verified annual model audit')
        verify_training_manifest(v,prediction,prepare,fit,winner,year,windows)
    return identity, phase, audit, dates, windows, old_backtest, reference_run, handovers


def verify_saved_run(root, run, *, output, correction, correction_sha256):
    started = time.monotonic()
    root,run,output = map(lambda path:Path(path).resolve(), (root,run,output))
    v = load_equations(root)
    v.require(run.is_relative_to(root/'outputs/rf_expanded_tuned_backtest')
              and output.is_relative_to(run) and not output.exists(),
              'Fresh independent output under the separate tuned run required')
    amendment,amendment_pin=verify_correction(v,root,run,correction,correction_sha256)
    identity,phase,audit,dates,windows,old_backtest,reference_run,handovers = authenticated_metadata(v,root,run)
    backtest=run/'backtest'; root_audit_hash=v.digest(run/'audit.json'); bt_audit_hash=v.digest(backtest/'audit.json')
    amount=verification_read_upper_bound(v,root,identity,backtest,old_backtest)
    prior=reserve_verification(v,run,amount); budget=ProjectionBudget(v,prior)
    def check_resources():
        peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if platform.system()!='Darwin': peak*=1024
        v.require(time.monotonic()-started <= amendment['verification_seconds_remaining'] and peak <= 4*1024**3,
                  'Independent verifier local resource ceiling exceeded')
        v.require(budget.used <= amount, 'Independent decoder exceeded its prior full-file reservation')
        return peak
    forecast_parts=[]; metadata_parts={}; reports=[]; identical_2020=False
    # All four own/original metadata pairs precede ANY numeric forecast projection.
    for entry in identity['forecast_entries']:
        year=entry['outer_year']; tuned_path=v.pinned(root,entry['predictions_pin'])
        original_path=v.pinned(root,entry['original_rf_predictions_pin'])
        tuned=budget.read(tuned_path,FORECAST_METADATA); original=budget.read(original_path,FORECAST_METADATA)
        verify_forecast_metadata(v,tuned,year,windows); verify_forecast_metadata(v,original,year,windows)
        v.require(tuned.fillna('<NA>').equals(original.fillna('<NA>')),
                  'Original cohort/source/native/admissions/order/missing labels changed in tuned forecasts')
        metadata_parts[year]=tuned
    for entry in identity['forecast_entries']:
        year=entry['outer_year']; path=v.pinned(root,entry['predictions_pin'])
        forecasts,report=read_forecast(v,path,year,windows,budget.read,verified_metadata=metadata_parts[year])
        forecast_parts.append(forecasts); reports.append({'outer_year':year,**report})
        if year==2020:
            original=budget.read(v.pinned(root,entry['original_rf_predictions_pin']),FORECAST_FIELDS)
            identical_2020=np.array_equal(forecasts[PROBABILITIES].to_numpy(),original[PROBABILITIES].to_numpy(),equal_nan=True)
    forecasts=pd.concat(forecast_parts,ignore_index=True)
    all_metadata=pd.concat(metadata_parts.values(),ignore_index=True)
    v.require(len(forecasts)==43_655 and int(forecasts.prediction_eligible.sum())==42_145
              and int((all_metadata.prediction_eligible & ~all_metadata.target_available).sum())==65,
              'Original full cohort or sixty-five missing-outcome predictions lost')
    exact_preflight_checks=verify_preflight_semantics(v,budget,run)
    comparison,yearly=read_reports(v,budget,backtest)
    coverage=checked_table(v,budget,backtest/'selection_coverage.parquet',kind='selection coverage')
    verify_grid(v,comparison,yearly,coverage)
    v.require(audit['complete_proxy_paths']==int(comparison.complete_proxy_path.sum())
              and audit['failed_closed_paths']==int((~comparison.complete_proxy_path).sum()),
              'Saved complete/withheld counts disagree')
    prices=read_quotes(v,root,phase,reference_run,identity['execution_reference_root_identity_sha256'],dates,budget)
    quote_index={(row['trading_date'],row['intrinio_security_id']):row for row in prices.to_dict('records')}
    check_resources(); selection_reports=[]; scenario_reports=[]; first_year_controls=[]
    for basket in BASKETS:
        directory=backtest/'selections'/ARM/basket
        tables={name:checked_table(v,budget,directory/(name+'.parquet'),kind='selection '+name,maximum_rows=60_000)
                for name in ('original_ledger','selected','monthly','rebalances')}
        chosen,rebalance=tables['selected'],tables['rebalances']
        joined=coverage.loc[coverage.arm.eq(ARM)&coverage.basket.eq(basket)].drop(columns=['arm']).reset_index(drop=True)
        v.require(v.normalize(joined).equals(v.normalize(tables['monthly']).reset_index(drop=True)),
                  'Aggregate and individual monthly breadth differ')
        selection_reports.append({'arm':ARM,'basket':basket,**v.verify_selection(
            forecasts,tables['original_ledger'],chosen,tables['monthly'],windows,basket=basket)})
        if identical_2020:
            original=checked_table(v,budget,old_backtest/'selections/rf_extended'/basket/'selected.parquet',kind='original selected',maximum_rows=60_000)
            fields=['source_security_id','signal_date','intrinio_security_id','execution_date','side','selection_score','intended_weight']
            a=v.normalize(chosen); b=v.normalize(original)
            a=a.loc[a.execution_date.str[:4].eq('2020'),fields].reset_index(drop=True)
            b=b.loc[b.execution_date.str[:4].eq('2020'),fields].reset_index(drop=True)
            pd.testing.assert_frame_equal(a,b,check_dtype=False,check_exact=True)
        for variant,fee in itertools.product(VARIANTS,FEES):
            check_resources(); stop=None if variant=='unstopped' else .1
            path=backtest/'arms'/ARM/basket/variant/f'cost_{fee:02d}bps'
            ledgers={name:checked_table(v,budget,path/(name+'.parquet'),kind=name) for name in LEDGERS}
            monthly=monthly_table(v,budget,path/'monthly.parquet')
            mask=comparison.arm.eq(ARM)&comparison.basket.eq(basket)&comparison.variant.eq(variant)&comparison.cost_bps.eq(fee)
            summary=comparison.loc[mask].iloc[0]
            annual=yearly.loc[yearly.arm.eq(ARM)&yearly.basket.eq(basket)&yearly.variant.eq(variant)&yearly.cost_bps.eq(fee)]
            if summary.complete_proxy_path:
                checked=v.verify_scenario(ledgers,prices,chosen,rebalance,dates,events=phase['event_registry']['events'],
                    historical_retirements=phase['historical_retirements'],cost_bps=fee,stop_fraction=stop,
                    prepared_quotes=quote_index,identity_handovers=handovers)
                v.verify_metrics(ledgers['daily'],ledgers['trades'],monthly,summary,annual)
                checked.update(path_status='complete_proxy',economic_performance_available=True)
            else:
                checked=v.verify_failed_scenario(ledgers,prices,chosen,dates,events=phase['event_registry']['events'],
                    historical_retirements=phase['historical_retirements'],cost_bps=fee,stop_fraction=stop,
                    prepared_quotes=quote_index,rebalances=rebalance,identity_handovers=handovers)
                v.verify_failed_metrics(ledgers['daily'],ledgers['trades'],monthly,summary,annual)
                checked['path_status']='failed_closed'
            if identical_2020:
                for name in ('daily','trades','stops'):
                    original=checked_table(v,budget,old_backtest/'arms/rf_extended'/basket/variant/f'cost_{fee:02d}bps'/(name+'.parquet'),kind='original '+name)
                    a=v.normalize(ledgers[name]);b=v.normalize(original)
                    a=a.loc[a.date.str[:4].eq('2020')].reset_index(drop=True)
                    b=b.loc[b.date.str[:4].eq('2020')].reset_index(drop=True)
                    pd.testing.assert_frame_equal(a,b,check_dtype=False,check_exact=True)
                first_year_controls.append({'basket':basket,'variant':variant,'cost_bps':fee,
                                            '2020_selected_daily_trades_stops_exact':True})
            scenario_reports.append({'arm':ARM,'basket':basket,'variant':variant,'cost_bps':fee,**checked})
            print(json.dumps({'phase':'independent_tuned_rf_backtest_verification','verified_scenarios':len(scenario_reports),'expected_scenarios':16}),flush=True)
    for item in [*identity['code_pins'],*identity['source_pins'],*phase['source_pins']]:v.pinned(root,item)
    for item in audit['artifacts'].values():v.pinned(backtest,item)
    v.require(v.digest(run/'audit.json')==root_audit_hash and v.digest(backtest/'audit.json')==bt_audit_hash,
              'An audited root/result changed during independent verification')
    for item in [amendment['old_verifier_pin'],amendment['new_verifier_pin'],
                 amendment['diagnostic_pin'],*amendment['failed_verification_pins']]:v.pinned(root,item)
    v.pinned(root,amendment_pin)
    result={'schema':'dmv2-rf-tuned-backtest-independent-verification/v2',
        'status':'PASS','phase':'independent_tuned_rf_backtest_verification_v2',
        'build_identity_sha256':v.canonical(identity),'backtest_phase_identity_sha256':v.canonical(phase),
        'backtest_audit_sha256':bt_audit_hash,'root_audit_sha256':root_audit_hash,
        'script_sha256':v.digest(__file__),'independent_equations_script_sha256':BASE_SHA,
        'models_fitted':0,'preprocessors_fitted':0,'new_inference_calls':0,'production_accounting_called':False,
        'source_pins_unchanged':True,'original_actions_and_exact_PENN_handover_unchanged':True,
        'execution_reference_root_identity_sha256':identity['execution_reference_root_identity_sha256'],
        'execution_reference_audit_pin':identity['execution_reference_audit_pin'],
        'execution_reference_phase_pin':identity['execution_reference_phase_pin'],
        'financial_2024_plus_read':False,'sessions':1006,'outcome_months':48,'scenarios':16,
        'complete_proxy_paths':int(comparison.complete_proxy_path.sum()),'withheld_paths':int((~comparison.complete_proxy_path).sum()),
        'all_original_forecast_keys_and_admissions_preserved':True,'missing_label_predictions_retained':65,
        'forecasts':reports,'selections':selection_reports,'scenario_checks':scenario_reports,
        'verification_correction_pin':amendment_pin,
        'original_failed_verification_preserved':True,'preflight_exact_semantic_checks':exact_preflight_checks,
        'first_year_original_forecasts_bitwise_equal':bool(identical_2020),'first_year_integration_controls':first_year_controls,
        'verification_scope':'pinned_verified_forecasts_to_selection_and_independent_cash_units_cost_actions_stops_and_metrics;failed_path_retained_exposure',
        'knowledge_time_identity_survivorship_or_investability_certified':False,'withheld_economic_performance_certified':False,
        'prior_phase_projected_compressed_bytes':prior,'verification_reserved_compressed_file_upper_bound_bytes':amount,
        'verification_actual_compressed_file_upper_bound_bytes':budget.used,
        'execution_seconds':time.monotonic()-started,'peak_rss_bytes':check_resources()}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream:json.dump(result,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'run', 'backtest', 'output','correction'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--correction-sha256',required=True)
    args = parser.parse_args()
    if args.backtest.resolve() != (args.run.resolve()/'backtest'):
        raise RuntimeError('Exact sealed run/backtest output required')
    result = verify_saved_run(args.root, args.run, output=args.output,
                              correction=args.correction,correction_sha256=args.correction_sha256)
    print(json.dumps({k: value for k, value in result.items()
                      if k not in ('selections', 'scenario_checks')}, sort_keys=True))


if __name__ == '__main__':
    main()
