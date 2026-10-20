"""Explicit continuous 48-month walk-forward backtest, with pre-PnL coverage.

No fitting, feature calculation, quote fetching or mutation of prior runs occurs.
Dates and native keys are checked before every financial Parquet projection.
All selections and maximum-horizon gaps are saved before performance is computed.
"""
from importlib.metadata import version
from importlib.util import find_spec
import json
from pathlib import Path
import platform
import re
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .action_loading import ACTION_PRICE_COLUMNS
from .backtest_accounting import DailyAccountingPolicy
from .backtest_loading import ARMS, _hash, _json, _pin, _projected_bytes, _verify_pin, _within
from .backtest_selection import PROBABILITIES, SAFE_COLUMNS
from .catalog import canonical_sha256
from .daily_backtest import _rss, _save_table, _write_json, summarize_daily
from .universe import DataContractError
from .walkforward_accounting import run_walkforward_accounting
from .walkforward_clock import YEARS, walkforward_clock
from .walkforward_preflight import QUOTE_METADATA_COLUMNS, preflight_selected_horizons, validate_current_events
from .walkforward_selection import (BALANCED_BREADTH_POLICY, EXCLUDED_NATIVE_IDS,
    _selection_configuration, select_walkforward_basket, validate_walkforward_basket)

PHASE = 'walkforward_continuous_backtest_v1'
WORKERS = ('walkforward_clock.py', 'walkforward_selection.py', 'walkforward_accounting.py',
    'walkforward_preflight.py', 'walkforward_backtest.py', 'walkforward_models.py', 'backtest_accounting.py',
    'backtest_selection.py', 'backtest_loading.py', 'daily_backtest.py', 'catalog.py', 'universe.py')
REPORTS = ('comparison', 'yearly', 'selection_coverage', 'selection_validation', 'failures',
    'action_coverage', 'order_coverage')
MAX_READ_BYTES = 512 * 1024**2
MAX_PRICE_ROWS = 2_000_000


def _configuration(cfg):
    _selection_configuration(cfg)
    bt, execution, accounting = cfg['backtest'], cfg.get('execution', {}), cfg.get('accounting', {})
    if (bt.get('arms') != list(ARMS) or bt.get('cost_bps') != [0, 5, 10, 25]
            or bt.get('variants') != {'unstopped': None, 'asset_stop_10pct': .1}
            or bt.get('RET') != 'not_evaluated_in_this_class_basket_experiment'
            or execution.get('financial_2020_2023_analysis_authorized') is not True
            or execution.get('financial_2024_plus_analysis') is not False
            or execution.get('cloud_mutations') is not False
            or accounting.get('continuous_2020_2023_ledger') is not True
            or accounting.get('annual_fit_forced_liquidation') is not False
            or accounting.get('terminal_session') != '2023-12-29'
            or accounting.get('new_postselection_hindsight_exclusions') is not False
            or accounting.get('borrow_annual_bps') != 0 or accounting.get('funding_annual_bps') != 0):
        raise DataContractError('Walk-forward backtest differs from its frozen continuous period/scenario/accounting contract')
    for name, ceiling in (('backtest_maximum_wall_seconds', 900),
            ('backtest_maximum_peak_rss_bytes', 2 * 1024**3), ('maximum_output_bytes', 12 * 1024**3)):
        value = execution.get(name)
        if type(value) is not int or not 0 < value <= ceiling:
            raise DataContractError('Explicit bounded backtest resource required: ' + name)
    breadth = bt.get('basket_breadth', {})
    if (breadth.get('policy') != 'balanced_available_capped_10'
            or breadth.get('names_per_side') != 'min(10,eligible_mode_long_count,eligible_mode_short_count)'
            or breadth.get('equal_side_name_counts') is not True
            or breadth.get('long_total_weight') != .5 or breadth.get('short_total_weight') != -.5
            or breadth.get('zero_candidate_side') != 'withhold_full_path'
            or breadth.get('candidate_information') != 'signal_time_forecasts_and_fixed_execution_restrictions_only'
            or breadth.get('rank_and_ties') != 'unchanged_class_probability_descending_then_source_id_ascending'):
        raise DataContractError('Walk-forward selection must retain the exact declared balanced contemporaneous policy')
    return cfg


def _historical_registry(root, cfg, pins):
    declared = cfg.get('historical_retirement_registry', {})
    expected_directory = str(Path(cfg['corrected_validation_directory']) / 'backtest_completed')
    if (not isinstance(declared, dict)
            or set(declared) != {'directory','audit_sha256','build_identity_sha256'}
            or declared.get('directory') != expected_directory
            or any(not isinstance(declared.get(name),str) or not re.fullmatch(r'[0-9a-f]{64}',declared[name])
                for name in ('audit_sha256','build_identity_sha256'))):
        raise DataContractError('Historical retirement registry must use its exact pre-financial frozen file pins')
    directory = _within(root, root / declared['directory'])
    audit_path, identity_path = directory / 'audit.json', directory / 'build_identity.json'
    # Authenticate the two old files against the newly frozen protocol BEFORE
    # parsing their event/financial JSON. Runtime pinning alone is insufficient.
    if _hash(audit_path) != declared['audit_sha256'] or _hash(identity_path) != declared['build_identity_sha256']:
        raise DataContractError('Pre-period retirement bytes changed after the outer protocol was frozen')
    audit, phase = _json(audit_path), _json(identity_path)
    if (audit.get('status') != 'COMPLETE' or audit.get('phase') != 'independent_quote_action_completion_v1'
            or audit.get('build_identity_sha256') != cfg['corrected_validation_build_identity_sha256']
            or canonical_sha256(phase) != audit.get('phase_identity_sha256') or len(phase.get('events', [])) != 11):
        raise DataContractError('Pre-period retirements require the pinned completed eleven-event development run')
    pins.extend([_pin(root, audit_path), _pin(root, identity_path)])
    return phase['events']


def _prediction_plan(root, run, identity, windows, pins):
    from .walkforward_models import read_walkforward_models
    saved = read_walkforward_models(run / 'models', identity_sha256=identity, root=root)
    audit = saved['audit']
    if (audit.get('phase') != 'walkforward_models_v1' or audit.get('status') != 'COMPLETE'
            or audit.get('summary', {}).get('fitted_arms') != 16
            or audit.get('summary', {}).get('outer_years') != list(YEARS)
            or audit.get('summary', {}).get('months') != 48):
        raise DataContractError('Walk-forward predictions require exactly sixteen completed annual fits and 48 months')
    pins.extend([_pin(root, run / 'models/audit.json'), _pin(root, run / 'models/build_identity.json')])
    paths, cohort, total_bytes = {}, None, 0
    windows = windows.set_index('signal_date')
    for arm in ARMS:
        arm_path = _verify_pin(run / 'models', audit['arms'][arm])
        part = _json(arm_path)
        if (part.get('status') != 'COMPLETE' or part.get('build_identity_sha256') != identity
                or part.get('model_identity_sha256') != audit.get('model_identity_sha256')):
            raise DataContractError('Walk-forward aggregate arm is stale or incomplete')
        path = _verify_pin(arm_path.parent, part['artifacts']['predictions'])
        fields = set(pq.ParquetFile(path).schema_arrow.names)
        lineage = ('fit_year', 'training_outcome_start', 'training_outcome_end', 'training_label_observed_through')
        columns = [name for name in (*SAFE_COLUMNS, *lineage) if name in fields]
        required = {'source_security_id', 'signal_date', 'intrinio_security_id', 'prediction_eligible',
            'paper_prediction_eligible', *lineage}
        if not required <= set(columns) or not set(PROBABILITIES) <= fields:
            raise DataContractError('Saved walk-forward forecasts lack original keys/admissions/probabilities')
        total_bytes += _projected_bytes(path, columns) + _projected_bytes(path, PROBABILITIES)
        frame = pq.ParquetFile(path).read(columns=columns).to_pandas()
        if (len(frame) > 60_000 or frame.duplicated(['source_security_id', 'signal_date']).any()
                or set(frame.signal_date) != set(windows.index)
                or not frame.prediction_eligible.map(lambda value: isinstance(value, (bool, np.bool_))).all()
                or not frame.prediction_eligible.equals(frame.paper_prediction_eligible)):
            raise DataContractError('Saved forecast keys changed the complete 48-month original prediction cohort')
        for field in ('outcome_month', 'outcome_start', 'outcome_end'):
            if field in frame and not frame[field].eq(frame.signal_date.map(windows[field])).all():
                raise DataContractError('Saved forecast outcome metadata differs from the independent XNYS clock')
        fit_year = frame.signal_date.map(windows.outcome_month).str[:4].astype(int)
        cutoffs = {2020:'2019-12-31',2021:'2020-12-31',2022:'2021-12-31',2023:'2022-12-30'}
        if (not frame.fit_year.eq(fit_year).all()
                or not frame.training_outcome_start.eq(fit_year.map(lambda year:f'{year-10}-01-01')).all()
                or not frame.training_outcome_end.eq(fit_year.map(lambda year:f'{year-1}-12-31')).all()
                or not frame.training_label_observed_through.eq(fit_year.map(cutoffs)).all()):
            raise DataContractError('Forecast lineage does not use exactly the preceding ten completed years and annual XNYS cutoff')
        checked = frame[[name for name in ('source_security_id', 'signal_date', 'intrinio_security_id',
            'identity_eligible', 'paper_prediction_eligible', 'prediction_eligible', *lineage) if name in frame]]
        if cohort is None: cohort = checked
        elif not checked.equals(cohort):
            raise DataContractError('Walk-forward model arms have different original cohorts/admissions/order')
        paths[arm] = (path, columns)
        pins.extend([_pin(root, arm_path), _pin(root, path)])
    if total_bytes > MAX_READ_BYTES:
        raise DataContractError('Walk-forward forecast projections exceed the explicit read budget')
    return paths, total_bytes


def _price_plan(root, run, identity, pins):
    plans, metadata, rows, total_bytes, requested_ids = {}, [], 0, 0, set()
    for year in YEARS:
        directory = run / 'years' / str(year)
        audit_path = directory / 'audit.json'; audit = _json(audit_path)
        if (audit.get('status') != 'COMPLETE' or audit.get('year') != year
                or audit.get('build_identity_sha256') != identity):
            raise DataContractError('Walk-forward execution annual cache has the wrong completed identity/year')
        path = _verify_pin(directory, audit['artifacts']['daily_prices_raw'])
        pf = pq.ParquetFile(path)
        columns = [name for name in QUOTE_METADATA_COLUMNS if name in pf.schema_arrow.names and name in ACTION_PRICE_COLUMNS]
        # Date/native keys are always decoded before financial column planning.
        keys = pf.read(columns=['trading_date', 'intrinio_security_id']).to_pandas()
        if (keys.isna().any().any() or keys.duplicated(list(keys)).any()
                or not keys.trading_date.map(lambda day: isinstance(day, str)
                    and f'{year}-01-01' <= day <= f'{year}-12-31').all()):
            raise DataContractError('Annual quote keys escape the explicit outcome year before financial decoding')
        declared_ids = audit.get('feature_provenance', {}).get('requested_ids')
        if not isinstance(declared_ids,list) or not declared_ids or any(not isinstance(native,str) or not native.strip() for native in declared_ids):
            raise DataContractError('Execution-only supplements require the original annual feature request IDs')
        requested_ids.update(declared_ids); requested_ids.update(keys.intrinio_security_id)
        if not set(ACTION_PRICE_COLUMNS) <= set(pf.schema_arrow.names):
            raise DataContractError('Annual daily cache lacks paired raw/adjusted execution fields')
        rows += len(keys)
        total_bytes += _projected_bytes(path, columns) + _projected_bytes(path, ACTION_PRICE_COLUMNS)
        if rows > MAX_PRICE_ROWS or total_bytes > MAX_READ_BYTES:
            raise DataContractError('Walk-forward annual caches exceed explicit projection/row limits')
        meta = pf.read(columns=columns).to_pandas()
        if not meta[['trading_date', 'intrinio_security_id']].equals(keys):
            raise DataContractError('Quote keys changed between guarded metadata projections')
        metadata.append(meta); plans[year] = path
        pins.extend([_pin(root, audit_path), _pin(root, path)])
    result = pd.concat(metadata, ignore_index=True)
    if result.duplicated(['trading_date', 'intrinio_security_id']).any():
        raise DataContractError('Disjoint walk-forward execution years overlap')
    return plans, result, {'candidate_rows': rows, 'projected_compressed_bytes': total_bytes,
        'original_requested_native_ids': sorted(requested_ids),
        'metadata_guards_before_financial_decode': True, 'financial_2024_plus_read': False}


def _supplement_plan(root, registry, original_metadata, original_requested_ids, sessions, pins):
    """Validate append-only execution quotes for NEW, evidenced successors.

    No existing feature-requested native may be extended after fitting. Such a
    repair needs its feature/target impact reviewed in the input build instead.
    The bounded exact native/date range is checked using metadata before any
    financial projection is planned or decoded. This function never reads OHLC.
    """
    sources = registry.get('supplemental_quote_sources', [])
    if not isinstance(sources,list) or len(sources)>16:
        raise DataContractError('Bounded explicit supplemental quote source list required')
    plans, parts, seen, total_bytes, rows = [], [], set(), 0, 0
    original_ids = set(original_requested_ids) | set(original_metadata.intrinio_security_id)
    successors = {event.get('successor_id') for event in registry['events'] if event.get('successor_id')}
    required={'pin','allowed_native_ids','start','end','evidence_reference','identity_evidence_reference'}
    for source in sources:
        if not isinstance(source,dict) or not required<=set(source) or not set(source)<=required|{'receipt_pin'}:
            raise DataContractError('Supplemental quotes need exact reviewed identity/date/source declarations')
        ids=source['allowed_native_ids']; first,last=source['start'],source['end']
        if (not isinstance(ids,list) or not ids or len(ids)>8 or len(set(ids))!=len(ids)
                or any(not isinstance(native,str) or not native.strip() for native in ids)
                or not set(ids)<=successors or set(ids)&original_ids
                or not isinstance(first,str) or not isinstance(last,str)
                or not re.fullmatch(r'\d{4}-\d{2}-\d{2}',first) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}',last)
                or not '2020-01-01'<=first<=last<='2023-12-31'
                or any(not isinstance(source[name],str) or not source[name].startswith('https://')
                    for name in ('evidence_reference','identity_evidence_reference'))):
            raise DataContractError('Execution supplements must identify new in-period successor natives, never repair existing feature requests')
        try:
            pd.Timestamp(first); pd.Timestamp(last)
        except (TypeError,ValueError) as exc: raise DataContractError('Malformed supplement bounds') from exc
        path=_verify_pin(root,source['pin'])
        if path.suffix!='.parquet' or not path.is_relative_to(root/'data/derived') or path.stat().st_size>16*1024**2:
            raise DataContractError('Supplemental execution quotes must be small pinned private derived Parquet')
        pf=pq.ParquetFile(path); fields=set(pf.schema_arrow.names)
        if not set(ACTION_PRICE_COLUMNS)<=fields or pf.metadata.num_rows>10_000:
            raise DataContractError('Supplemental quotes require bounded complete paired raw/adjusted daily fields')
        columns=[name for name in QUOTE_METADATA_COLUMNS if name in fields and name in ACTION_PRICE_COLUMNS]
        # Safe metadata projection precedes all financial planning.
        keys=pf.read(columns=['trading_date','intrinio_security_id']).to_pandas()
        if (keys.empty or keys.isna().any().any() or keys.duplicated(list(keys)).any()
                or not set(keys.intrinio_security_id)<=set(ids)
                or not keys.trading_date.map(lambda day:isinstance(day,str) and first<=day<=last).all()
                or not keys.trading_date.isin(sessions).all()):
            raise DataContractError('Supplemental quote metadata escapes exact approved dates/natives/sessions')
        actual=set(keys.itertuples(index=False,name=None))
        if actual&seen: raise DataContractError('Supplemental sources overlap; no overwrite is permitted')
        seen.update(actual)
        total_bytes+=_projected_bytes(path,columns)+_projected_bytes(path,ACTION_PRICE_COLUMNS); rows+=len(keys)
        if total_bytes>64*1024**2 or rows>10_000:
            raise DataContractError('Execution-only supplement aggregate budget exceeded')
        metadata=pf.read(columns=columns).to_pandas()
        if not metadata[['trading_date','intrinio_security_id']].equals(keys):
            raise DataContractError('Supplemental keys changed between metadata projections')
        parts.append(metadata); plans.append(path); pins.append(_pin(root,path))
        if 'receipt_pin' in source:
            _verify_pin(root,source['receipt_pin']); pins.append(source['receipt_pin'])
    metadata=pd.concat(parts,ignore_index=True) if parts else original_metadata.iloc[:0].copy()
    return plans,metadata,{'sources':len(plans),'candidate_rows':rows,
        'projected_compressed_bytes':total_bytes,'execution_only':True,
        'existing_requested_native_ids_extended':False,'existing_quote_keys_overwritten':False,
        'metadata_before_financial_decode':True}


def yearly_breakdown(result, *, arm, basket, variant, cost_bps):
    """Derive annual figures from one continuous NAV; never reset capital."""
    daily = result.daily.drop_duplicates('date', keep='last').copy()
    daily['year'] = pd.to_datetime(daily.date).dt.year
    complete = result.status == 'complete_proxy' and daily.nav.notna().all()
    previous = float(result.summary['initial_nav']); rows = []
    for year, group in daily.groupby('year', sort=True):
        row = dict(arm=arm, basket=basket, variant=variant, cost_bps=cost_bps, year=int(year),
            expected_sessions=len(group), complete_full_period_proxy=bool(complete), initial_nav=None,
            final_nav=None, total_return=None, annualized_volatility=None, sharpe_zero_cash_rate=None,
            maximum_drawdown=None, trading_cost_paid=None)
        if complete:
            nav = group.nav.to_numpy(dtype=float); returns = nav / np.r_[previous, nav[:-1]] - 1
            sd = float(np.std(returns, ddof=1)); peak = np.maximum.accumulate(np.r_[previous, nav])[1:]
            row.update(initial_nav=previous, final_nav=float(nav[-1]), total_return=float(nav[-1]/previous-1),
                annualized_volatility=sd*np.sqrt(252), sharpe_zero_cash_rate=None if sd == 0 else float(returns.mean()/sd*np.sqrt(252)),
                maximum_drawdown=float(np.min(nav/peak-1)),
                trading_cost_paid=float(result.trades.loc[pd.to_datetime(result.trades.date).dt.year.eq(year), 'cost'].sum()))
            previous = float(nav[-1])
        rows.append(row)
    return pd.DataFrame(rows)


def run_walkforward_backtest(root, run_directory, configuration, identity_sha256, *,
        event_registry=None, output_directory=None, preflight_only=False, allow_withheld_paths=False):
    """Prepare frozen selections/coverage first, then explicitly run 64 paths.

    Missing key/source-quality/zero-breadth evidence blocks performance by default.
    An explicitly reviewed ``allow_withheld_paths=True`` retains those exposures
    and lets the engine record failed paths; it never changes their selections.
    A changed action registry produces a new phase identity and must use a new
    output_directory if an older numerical attempt already exists.
    """
    if type(preflight_only) is not bool or type(allow_withheld_paths) is not bool:
        raise DataContractError('Explicit boolean preflight/withheld-path scope required')
    root = Path(root).resolve(); run = _within(root, Path(run_directory).resolve())
    cfg = _configuration(configuration); started = time.monotonic(); peak = _rss()
    def check_resources():
        nonlocal peak
        peak=max(peak,_rss())
        if (time.monotonic()-started>cfg['execution']['backtest_maximum_wall_seconds']
                or peak>cfg['execution']['backtest_maximum_peak_rss_bytes']):
            raise DataContractError('Walk-forward backtest exceeds its frozen resource ceiling')
    main_identity = _json(run / 'build_identity.json')
    if canonical_sha256(main_identity) != identity_sha256 or main_identity.get('configuration') != cfg:
        raise DataContractError('Backtest configuration differs from its exact parent build identity')
    pins = [_pin(root, run / 'build_identity.json')]
    cfg_path = root / 'configs/walkforward_2020_2023.json'
    if _json(cfg_path) != cfg: raise DataContractError('Saved walk-forward configuration changed')
    pins.append(_pin(root, cfg_path))
    sessions, windows, clock = walkforward_clock()
    historical = _historical_registry(root, cfg, pins)
    registry = {'schema': 'dmv2-walkforward-actions/v1', 'events': []} if event_registry is None else event_registry
    if isinstance(registry, (str, Path)):
        path = _within(root, Path(registry).resolve()); registry = _json(path); pins.append(_pin(root, path))
    if not isinstance(registry, dict) or registry.get('schema') != 'dmv2-walkforward-actions/v1' or not isinstance(registry.get('events'), list):
        raise DataContractError('Explicit independently reviewed action-registry manifest required')
    for item in registry.get('source_pins', []):
        _verify_pin(root,item); pins.append(item)
    events = registry['events']; validate_current_events(events, sessions)
    prediction_plan, prediction_bytes = _prediction_plan(root, run, identity_sha256, windows, pins)
    annual_plan, metadata, read_plan = _price_plan(root, run, identity_sha256, pins)
    supplement_paths,supplement_metadata,supplement_stats=_supplement_plan(root,registry,metadata,
        read_plan['original_requested_native_ids'],sessions,pins)
    metadata=pd.concat([metadata,supplement_metadata],ignore_index=True)
    read_plan['supplemental_execution_quotes']=supplement_stats
    if read_plan['projected_compressed_bytes'] + prediction_bytes + supplement_stats['projected_compressed_bytes'] > MAX_READ_BYTES:
        raise DataContractError('Combined execution/forecast projections exceed the explicit read budget')
    if not metadata.trading_date.isin(sessions).all():
        raise DataContractError('Execution metadata contains non-XNYS dates')
    check_resources()
    selections, preflights, validations = {}, {}, []
    for arm, (path, columns) in prediction_plan.items():
        forecasts = pq.ParquetFile(path).read(columns=list(dict.fromkeys([*columns, *PROBABILITIES]))).to_pandas()
        for basket in cfg['backtest']['baskets']:
            selected = select_walkforward_basket(forecasts, basket=basket, sessions=clock['selection_sessions'], configuration=cfg)
            check = validate_walkforward_basket(selected, forecasts, basket=basket, sessions=clock['selection_sessions'], configuration=cfg)
            if len(selected.monthly) != 48: raise DataContractError('Selection lost a walk-forward prediction month')
            selections[(arm, basket)] = selected
            validations.append({'arm': arm, 'basket': basket, **{k:v for k,v in check.items() if k != 'policy'}})
            preflights[(arm, basket)] = preflight_selected_horizons(selected.selections, selected.rebalances, metadata,
                sessions, events=events, historical_retirements=historical)
            check_resources()
    phase = {'schema_version': 1, 'phase': PHASE, 'parent_identity_sha256': identity_sha256,
        'configuration_sha256': canonical_sha256(cfg), 'source_pins': pins, 'clock': clock,
        'event_registry': registry, 'historical_retirements': historical,
        'models_fitted': False, 'preprocessing_fitted': False,
        'basket_policy': BALANCED_BREADTH_POLICY, 'read_plan': read_plan,
        'allow_withheld_paths_after_explicit_review': allow_withheld_paths,
        'workers': {name: _hash(Path(find_spec('dmv2.'+name[:-3]).origin)) for name in WORKERS},
        'runtime': {'python': platform.python_version(), **{name:version(name) for name in ('numpy','pandas','pyarrow','exchange-calendars')}}}
    phase_pin = canonical_sha256(phase)
    preflight_dir = run / 'preflight' / phase_pin[7:23]
    if not preflight_dir.exists():
        preflight_dir.mkdir(parents=True); artifacts = {}
        _write_json(preflight_dir / 'build_identity.json', phase)
        artifacts['build_identity.json'] = _pin(preflight_dir, preflight_dir / 'build_identity.json')
        summaries = []
        for (arm, basket), item in preflights.items():
            selected = selections[(arm, basket)]
            for name, frame in item.items():
                if name != 'summary': _save_table(preflight_dir, f'{arm}/{basket}/{name}.parquet', frame, artifacts)
            for name, frame in (('original_ledger', selected.ledger), ('selected', selected.selections),
                    ('rebalances', selected.rebalances), ('monthly', selected.monthly)):
                _save_table(preflight_dir, f'{arm}/{basket}/{name}.parquet', frame, artifacts)
            summaries.append({'arm':arm,'basket':basket,**item['summary'],
                'zero_breadth_months': int(selected.monthly.selected_names_per_side.eq(0).sum())})
        _save_table(preflight_dir, 'summary.parquet', pd.DataFrame(summaries), artifacts)
        _write_json(preflight_dir / 'audit.json', {'status':'COMPLETE','phase':'walkforward_selected_horizon_preflight_v1',
            'build_identity_sha256':identity_sha256,'phase_identity_sha256':phase_pin,'artifacts':artifacts,
            'selection_changed':False,'performance_computed':False,'maximum_horizons_reviewed':True})
    else:
        saved = _json(preflight_dir / 'audit.json')
        if saved.get('phase_identity_sha256') != phase_pin or saved.get('status') != 'COMPLETE':
            raise DataContractError('Preserve the incomplete or stale preflight attempt')
        for item in saved['artifacts'].values(): _verify_pin(preflight_dir, item)
    if preflight_only: return preflight_dir
    unresolved = any(not item['summary']['ready_for_accounting'] for item in preflights.values())
    zero_breadth = any(selection.monthly.selected_names_per_side.eq(0).any() for selection in selections.values())
    if (unresolved or zero_breadth) and not allow_withheld_paths:
        raise DataContractError('Pre-performance coverage requires repair/review; frozen selections and all gaps saved at '+str(preflight_dir))
    directory = _within(run, Path(output_directory).resolve()) if output_directory is not None else run / 'backtest'
    if directory.exists():
        return read_walkforward_backtest(directory, identity_sha256=identity_sha256, phase_identity_sha256=phase_pin, root=root)['directory']
    directory.mkdir(parents=True); artifacts = {}
    _write_json(directory / 'build_identity.json', phase)
    artifacts['build_identity.json'] = _pin(directory, directory / 'build_identity.json')
    _write_json(directory / 'audit.json', {'status':'STARTED','phase':PHASE,'build_identity_sha256':identity_sha256,'phase_identity_sha256':phase_pin})
    prices = pd.concat([pq.ParquetFile(path).read(columns=list(ACTION_PRICE_COLUMNS)).to_pandas()
        for path in [*annual_plan.values(),*supplement_paths]], ignore_index=True)
    if not prices[['trading_date','intrinio_security_id']].equals(metadata[['trading_date','intrinio_security_id']]):
        raise DataContractError('Execution keys changed between metadata and financial projections')
    comparison, yearly, failures, actions, orders = [], [], [], [], []
    try:
        for (arm, basket), selected in selections.items():
            for name, frame in (('original_ledger',selected.ledger),('selected',selected.selections),
                    ('rebalances',selected.rebalances),('monthly',selected.monthly)):
                _save_table(directory, f'selections/{arm}/{basket}/{name}.parquet', frame, artifacts)
            for variant, stop in cfg['backtest']['variants'].items():
                for cost in cfg['backtest']['cost_bps']:
                    check_resources()
                    result = run_walkforward_accounting(selected.selections, prices, sessions,
                        rebalances=selected.rebalances, terminal_date='2023-12-29', events=events,
                        historical_retirements=historical, policy=DailyAccountingPolicy(cost_bps=cost,
                            borrow_annual_bps=0, required_names_per_side=None, asset_stop_fraction=stop))
                    row, monthly = summarize_daily(result, arm=arm, basket=basket, variant=variant,
                        cost_bps=cost, expected_sessions=1006)
                    if len(monthly)!=48: raise DataContractError('Continuous accounting lost a monthly return')
                    row.update(outcome_start='2020-01-01',outcome_end='2023-12-31',outcome_months=48,
                        terminal_date='2023-12-29',annual_liquidation=False,annual_model_fits=4)
                    comparison.append(row)
                    yearly.append(yearly_breakdown(result,arm=arm,basket=basket,variant=variant,cost_bps=cost))
                    relative=f'arms/{arm}/{basket}/{variant}/cost_{cost:02d}bps'
                    for name, table in result.tables.items(): _save_table(directory,relative+'/'+name+'.parquet',table,artifacts)
                    _save_table(directory,relative+'/monthly.parquet',monthly,artifacts)
                    if not result.failures.empty: failures.append(result.failures.assign(arm=arm,basket=basket,variant=variant,cost_bps=cost))
                    if not result.action_events.empty: actions.append(result.action_events.assign(arm=arm,basket=basket,variant=variant,cost_bps=cost))
                    if not result.orders.empty: orders.append(result.orders.assign(arm=arm,basket=basket,variant=variant,cost_bps=cost))
                    print(json.dumps({'phase':PHASE,'arm':arm,'basket':basket,'variant':variant,'cost_bps':cost,
                        'status':result.status,'completed_scenarios':len(comparison)}),flush=True)
                    if sum(path.stat().st_size for path in run.rglob('*') if path.is_file()) > cfg['execution']['maximum_output_bytes']:
                        raise DataContractError('Walk-forward output budget exceeded')
        reports = {'comparison':pd.DataFrame(comparison),'yearly':pd.concat(yearly,ignore_index=True),
            'selection_coverage':pd.concat([value.monthly.assign(arm=arm,basket=basket)
                for (arm,basket),value in selections.items()],ignore_index=True),
            'selection_validation':pd.DataFrame(validations),
            'failures':pd.concat(failures,ignore_index=True) if failures else pd.DataFrame(columns=['date','reason','arm','basket','variant','cost_bps']),
            'action_coverage':pd.concat(actions,ignore_index=True) if actions else pd.DataFrame(columns=['event_id','arm','basket','variant','cost_bps']),
            'order_coverage':pd.concat(orders,ignore_index=True) if orders else pd.DataFrame(columns=['status','arm','basket','variant','cost_bps'])}
        for name,frame in reports.items(): _save_table(directory,name+'.parquet',frame,artifacts)
        for item in pins: _verify_pin(root,item)
        check_resources()
        _write_json(directory/'audit.json',{'status':'COMPLETE','phase':PHASE,'build_identity_sha256':identity_sha256,
            'phase_identity_sha256':phase_pin,'artifacts':artifacts,'scenarios':64,'sessions':1006,'outcome_months':48,
            'complete_proxy_paths':sum(item['complete_proxy_path'] for item in comparison),
            'failed_closed_paths':sum(not item['complete_proxy_path'] for item in comparison),
            'models_fitted':False,'preprocessing_fitted':False,'financial_2024_plus_read':False,
            'source_pins_unchanged':True,'original_pointers_changed':False,'annual_liquidation':False,
            'RET':'not_evaluated','execution_seconds':time.monotonic()-started,'peak_rss_bytes':max(peak,_rss()),
            'preflight_directory':str(preflight_dir.relative_to(run)), 'preflight_before_performance':True})
    except Exception as error:
        _write_json(directory/'failure.json',{'exception':type(error).__name__,'message':str(error)}); raise
    return directory


def read_walkforward_backtest(directory, *, identity_sha256=None, phase_identity_sha256=None, root=None):
    """Verify completed saved artifacts; no numerical execution or model loads."""
    directory=Path(directory).resolve(); audit=_json(directory/'audit.json')
    if (audit.get('status')!='COMPLETE' or audit.get('phase')!=PHASE or audit.get('scenarios')!=64
            or audit.get('sessions')!=1006 or audit.get('outcome_months')!=48
            or audit.get('models_fitted') is not False or audit.get('annual_liquidation') is not False
            or audit.get('financial_2024_plus_read') is not False
            or identity_sha256 is not None and audit.get('build_identity_sha256')!=identity_sha256
            or phase_identity_sha256 is not None and audit.get('phase_identity_sha256')!=phase_identity_sha256):
        raise DataContractError('Saved continuous walk-forward backtest changed identity or scope')
    if not {'build_identity.json',*[name+'.parquet' for name in REPORTS]}<=set(audit.get('artifacts',{})):
        raise DataContractError('Saved continuous backtest omits required reports')
    for item in audit['artifacts'].values(): _verify_pin(directory,item)
    phase=_json(directory/'build_identity.json')
    if canonical_sha256(phase)!=audit.get('phase_identity_sha256') or phase.get('parent_identity_sha256')!=audit.get('build_identity_sha256'):
        raise DataContractError('Saved continuous backtest phase changed')
    if root is not None:
        for item in phase['source_pins']: _verify_pin(Path(root).resolve(),item)
    comparison=pd.read_parquet(directory/'comparison.parquet')
    coverage=pd.read_parquet(directory/'selection_coverage.parquet')
    if (len(comparison)!=64 or comparison.duplicated(['arm','basket','variant','cost_bps']).any()
            or set(comparison.arm)!=set(ARMS) or set(comparison.basket)!={'class_1_10','class_2_9'}
            or set(comparison.variant)!={'unstopped','asset_stop_10pct'} or set(comparison.cost_bps)!={0,5,10,25}
            or not comparison.expected_sessions.eq(1006).all() or len(coverage)!=384
            or coverage.duplicated(['arm','basket','signal_date']).any()):
        raise DataContractError('Saved continuous backtest lost its complete scenario/month grid')
    return {'directory':directory,'audit':audit,'comparison':comparison,'selection_coverage':coverage}
