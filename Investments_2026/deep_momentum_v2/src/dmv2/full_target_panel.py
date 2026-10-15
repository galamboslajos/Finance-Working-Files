"""Explicit cache-only target build and saved class/RET coverage reader."""
from __future__ import annotations
import argparse
import base64
import gc
from importlib.metadata import version
import json
from pathlib import Path
import platform
import tempfile
import time

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .catalog import SourceCatalog, canonical_sha256
from .context_panel_roles import read_context_panel_roles
from .full_panel import (_artifact, _hash, _json, _limits, _progress,
                         _verify_artifacts, _within, _write_table)
from .full_target_loading import plan_cached_target_year, build_cached_target_year
from .target_coverage import aggregate_target_coverage, audit_ret_coverage, combine_target_panels
from .target_workflow import load_target_ret_protocol
from .targets import MonthlyTargetPanel, verify_target_panel
from .universe import DataContractError

YEARS = tuple(range(2010, 2024))
WORKERS = ('full_target_panel.py', 'full_target_loading.py', 'target_coverage.py',
    'targets.py', 'ret.py', 'classification.py', 'target_workflow.py', 'monthly.py',
    'identity_policy.py', 'universe.py', 'data.py', 'catalog.py', 'loading.py',
    'context_panel_roles.py', 'full_panel.py', 'chronology.py')
SMALL_REPORTS = ('target_coverage_month', 'target_coverage_year', 'target_coverage_period',
    'target_class_year', 'rolling_training_coverage', 'rolling_training_classes',
    'ret_coverage_signal', 'ret_coverage_year')


def configuration(catalog):
    value = json.loads((catalog.root / 'configs/full_target_build.json').read_text())
    if (type(value.get('schema_version')) is not int or value['schema_version'] != 1
            or type(value.get('start_year')) is not int or value['start_year'] != 2010
            or type(value.get('end_year')) is not int or value['end_year'] != 2023
            or value.get('universe') != 'combined'
            or value.get('source') != 'verified_saved_daily_history_raw_and_original_feature_cohorts'
            or value.get('identity_policy') != 'provisional_unique_resolved_link'):
        raise DataContractError('Full target build differs from its approved scope')
    execution = value.get('execution', {})
    if (type(execution.get('workers')) is not int or execution['workers'] != 1
            or execution.get('backend') != 'local_verified_cache'
            or any(execution.get(k) is not False for k in ('model_fitting', 'imputation_fitting',
                'backtesting', 'cloud_mutations', 'raw_source_fallback'))):
        raise DataContractError('Full target building cannot fit or fall back to raw/cloud reads')
    for key in ('maximum_wall_seconds', 'maximum_peak_rss_bytes', 'maximum_output_bytes',
                'maximum_projected_read_bytes'):
        if type(execution.get(key)) is not int or execution[key] <= 0:
            raise DataContractError('Full target building requires positive explicit resource ceilings')
    if value.get('ret_audit') != {'all_feature_signals': 168,
        'canonical_outcome_years': [2020, 2021, 2022, 2023], 'expected_canonical_signals': 48,
        'calendar_months': 120, 'all_ten_classes_every_month': True,
        'missing_grid_policy': 'withhold_entire_payoff_vector'}:
        raise DataContractError('Full RET audit cannot change the calendar or model-test bounds')
    return value


def build_identity(catalog):
    cfg = configuration(catalog)
    catalog.verify(hashes=False)
    return {'schema_version': 1, 'configuration': cfg,
        'target_ret_protocol': load_target_ret_protocol(catalog.root),
        'feature_roles': json.loads((catalog.root / 'configs/context_panel_roles.json').read_text()),
        'source_pins': catalog.provenance(),
        'workers': {name: _hash(Path(__file__).parent / name) for name in WORKERS},
        'runtime': {'python': platform.python_version(), **{n: version(n)
                    for n in ('numpy', 'pandas', 'pyarrow', 'exchange-calendars')}}}


def plan_full_targets(catalog):
    identity = build_identity(catalog)
    roles = read_context_panel_roles(catalog)
    plans = {year: plan_cached_target_year(roles['primary'], year) for year in YEARS}
    signals = [w.signal_date for plan in plans.values() for w in plan.windows]
    totals = {'outcome_years': list(YEARS), 'signals': len(signals),
        'original_rows': sum(p.read_plan['original_rows'] for p in plans.values()),
        'paper_qualified_rows': sum(p.read_plan['paper_qualified_rows'] for p in plans.values()),
        'projected_read_bytes': sum(p.read_plan['projected_compressed_bytes'] for p in plans.values()),
        'estimated_peak_memory_bytes': max(p.read_plan['estimated_peak_memory_bytes'] for p in plans.values()),
        'daily_cache_rows_with_annual_overlap': sum(p.read_plan['daily_cache_rows'] for p in plans.values()),
        'signal_dates': signals, 'financial_values_read': False, 'raw_source_reads': False}
    if (totals['signals'] != 168 or len(set(signals)) != 168
            or totals['original_rows'] != 152307 or totals['paper_qualified_rows'] != 140777):
        raise DataContractError('Full target planning changed the verified original cohort scope')
    limits = identity['configuration']['execution']
    if (totals['projected_read_bytes'] > limits['maximum_projected_read_bytes']
            or totals['estimated_peak_memory_bytes'] > limits['maximum_peak_rss_bytes']):
        raise DataContractError('Full target plan exceeds the recorded read/memory limits')
    return identity, roles, plans, totals


def _preserved_manifest(root, roles):
    entries = {}
    for role in ('primary', 'sensitivity'):
        run = roles[role]
        directory = run['directory']
        for path in (directory / 'audit.json',):
            entries[str(path.relative_to(root))] = {'path': str(path.relative_to(root)),
                'sha256': _hash(path), 'size_bytes': path.stat().st_size}
        for item in [*run['audit']['reports'].values(), *run['audit']['years'].values()]:
            path = _within(directory, directory / item['path'])
            entries[str(path.relative_to(root))] = {**item, 'path': str(path.relative_to(root))}
        for year in YEARS:
            annual = directory / 'years' / str(year)
            audit = json.loads((annual / 'audit.json').read_text())
            for item in audit['artifacts'].values():
                path = _within(annual, annual / item['path'])
                entries[str(path.relative_to(root))] = {**item, 'path': str(path.relative_to(root))}
    for name in ('outputs/full_feature_panel/LATEST.json', 'outputs/context_sensitivity/LATEST_80.json',
                 'configs/full_panel_build.json', 'configs/normalization_protocol.json',
                 'configs/research_workflow.json', 'configs/context_panel_roles.json'):
        path = root / name
        entries[name] = {'path': name, 'sha256': _hash(path), 'size_bytes': path.stat().st_size}
    return {'artifacts': list(entries.values()), 'scope': 'Both immutable feature runs and their daily artifacts'}


def verify_preserved_inputs(root, manifest):
    items = manifest.get('artifacts', [])
    if (not items or len({item['path'] for item in items}) != len(items)):
        raise DataContractError('Source preservation manifest is empty or duplicates paths')
    _verify_artifacts(root, {item['path']: item for item in manifest['artifacts']})
    return len(manifest['artifacts'])


def _save_target_state(directory, targets, artifacts):
    _write_table(directory, 'targets', targets.table, artifacts)
    _write_table(directory, 'target_diagnostics', targets.diagnostics, artifacts)
    _json(directory / 'target_state.json', {'provenance': targets.provenance,
        'artifact_sha256': targets.artifact_sha256,
        'schemas': {name: base64.b64encode(table.schema.serialize()).decode('ascii')
                    for name, table in (('targets', targets.table),
                                        ('target_diagnostics', targets.diagnostics))}})
    artifacts['target_state'] = _artifact(directory, directory / 'target_state.json')


def _load_target_state(directory):
    state = json.loads((directory / 'target_state.json').read_text())
    # Parquet standardizes nested-list child names. Restore the exact original
    # Arrow schemas before checking the already-pinned logical values/schema.
    schemas = {name: pa.ipc.read_schema(pa.BufferReader(base64.b64decode(value, validate=True)))
               for name, value in state['schemas'].items()}
    result = MonthlyTargetPanel(
        pq.read_table(directory / 'targets.parquet').cast(schemas['targets']),
        pq.read_table(directory / 'target_diagnostics.parquet').cast(schemas['target_diagnostics']),
        state['provenance'], state['artifact_sha256'])
    verify_target_panel(result)
    return result


def _year_audit(directory, pin):
    audit = json.loads((directory / 'audit.json').read_text())
    if (audit.get('build_identity_sha256') != pin or audit.get('status') != 'COMPLETE'
            or type(audit.get('year')) is not int or str(audit['year']) != directory.name
            or not {'monthly_returns', 'targets', 'target_diagnostics', 'target_state',
                    'monthly_provenance'} <= set(audit.get('artifacts', {}))):
        raise DataContractError('Target year checkpoint is stale or incomplete')
    _verify_artifacts(directory, audit['artifacts'])
    return audit


def build_full_targets(catalog, *, years=None):
    started = time.monotonic()
    identity, roles, plans, totals = plan_full_targets(catalog)
    pin = canonical_sha256(identity)
    _progress('target_read_plan', **{k: v for k, v in totals.items() if k != 'signal_dates'})
    parent = _within(catalog.root, catalog.root / 'outputs/full_target_panel')
    destination = _within(parent, parent / ('2010-2023-' + pin[7:23]))
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'years').mkdir(exist_ok=True)
    if (destination / 'build_identity.json').exists():
        if json.loads((destination / 'build_identity.json').read_text()) != identity:
            raise DataContractError('Target run identity changed')
    else:
        _json(destination / 'build_identity.json', identity)
        _json(destination / 'source_plan.json', {'summary': totals,
            'years': {str(y): {'read_plan': p.read_plan, 'provenance': p.provenance} for y, p in plans.items()}})
        _json(destination / 'preserved_source_manifest.json', _preserved_manifest(catalog.root, roles))
    selected = YEARS if years is None else tuple(years)
    if (not selected or selected != tuple(sorted(set(selected))) or not set(selected) <= set(YEARS)):
        raise DataContractError('Target years must be ordered distinct 2010–2023 values')
    for year in selected:
        directory = destination / 'years' / str(year)
        if (directory / 'audit.json').exists():
            _year_audit(directory, pin)
            _progress('target_year_reused', year=year)
            continue
        if directory.exists():
            raise DataContractError('Preserve and investigate incomplete target year: ' + str(directory))
        bundle = build_cached_target_year(roles['primary'], year, plan=plans[year])
        temporary = Path(tempfile.mkdtemp(prefix=f'.{year}-', dir=destination / 'years'))
        artifacts = {}
        _write_table(temporary, 'monthly_returns', bundle.monthly.table, artifacts)
        _save_target_state(temporary, bundle.targets, artifacts)
        _json(temporary / 'monthly_provenance.json', bundle.monthly.provenance)
        artifacts['monthly_provenance'] = _artifact(temporary, temporary / 'monthly_provenance.json')
        _json(temporary / 'audit.json', {'status': 'COMPLETE', 'year': year,
            'build_identity_sha256': pin, 'artifacts': artifacts,
            'target_artifact_sha256': bundle.targets.artifact_sha256,
            'loading_stats': bundle.loading_stats, 'provenance': bundle.provenance})
        temporary.rename(directory)
        _progress('target_year_completed', year=year, original_rows=bundle.targets.table.num_rows,
            labelled_rows=sum(bundle.targets.table['target_available'].to_pylist()))
        del bundle
        gc.collect()
        _limits(identity['configuration'], started, destination, totals['projected_read_bytes'])
    if not all((destination / 'years' / str(y) / 'audit.json').exists() for y in YEARS):
        return destination
    if (destination / 'audit.json').exists():
        read_full_target_run(catalog, directory=destination)
        return destination
    yearly, monthly, year_pins, year_artifacts = [], [], {}, {}
    for year in YEARS:
        directory = destination / 'years' / str(year)
        audit = _year_audit(directory, pin)
        yearly.append(_load_target_state(directory))
        monthly.append(pq.read_table(directory / 'monthly_returns.parquet'))
        year_artifacts[str(year)] = _artifact(destination, directory / 'audit.json')
        year_pins[str(year)] = {'audit_sha256': _hash(directory / 'audit.json'),
                               'target_artifact_sha256': audit['target_artifact_sha256']}
    targets = combine_target_panels(yearly, source_year_pins=year_pins)
    artifacts = {}
    _write_table(destination, 'monthly_returns', pa.concat_tables(monthly), artifacts)
    _save_target_state(destination, targets, artifacts)
    coverage = aggregate_target_coverage(targets)
    _progress('target_coverage_completed', **coverage['summary'])
    _progress('ret_coverage_started', signals=168, canonical_nominal_test_signals=48,
              calendar_months=120, classes=10)
    ret = audit_ret_coverage(targets, totals['signal_dates'])
    reports = destination / 'reports'
    reports.mkdir(exist_ok=True)
    for name, table in {**coverage['tables'], **ret['tables']}.items():
        path = reports / (name + '.parquet')
        pq.write_table(table, path, compression='zstd')
        artifacts[name] = _artifact(destination, path, rows=table.num_rows)
    if len(ret['canonical_payoffs']) != 48:
        raise DataContractError('RET audit did not verify all nominal model-test signals')
    for payoff in ret['canonical_payoffs']:
        signal = payoff.provenance['signal_date']
        directory = destination / 'ret_states' / signal
        # Resume only within this same pinned run, whose annual inputs were verified.
        directory.mkdir(parents=True, exist_ok=True)
        _write_table(directory, 'payoffs', payoff.table, {})
        _write_table(directory, 'monthly', payoff.monthly, {})
        _json(directory / 'state.json', {'provenance': payoff.provenance, 'fingerprint': payoff.fingerprint})
        for name in ('payoffs.parquet', 'monthly.parquet', 'state.json'):
            artifacts[f'ret/{signal}/{name}'] = _artifact(destination, directory / name)
    for name in ('build_identity.json', 'source_plan.json', 'preserved_source_manifest.json'):
        artifacts[name] = _artifact(destination, destination / name)
    manifest = json.loads((destination / 'preserved_source_manifest.json').read_text())
    preserved = verify_preserved_inputs(catalog.root, manifest)
    summary = {'targets': coverage['summary'], 'ret': ret['summary'],
        'preserved_source_files_verified': preserved, 'daily_data_unchanged': True,
        'feature_data_unchanged': True, 'models_fitted': False, 'imputation_fitted': False,
        'knowledge_time_verified': False, 'accounting_certified': False, 'model_ready': False,
        'elapsed_seconds': time.monotonic() - started}
    _limits(identity['configuration'], started, destination, totals['projected_read_bytes'])
    _json(destination / 'audit.json', {'status': 'COMPLETE', 'build_identity_sha256': pin,
        'summary': summary, 'years': year_artifacts, 'artifacts': artifacts})
    _json(parent / 'LATEST.json', {'path': destination.name, 'build_identity_sha256': pin,
                                 'audit_sha256': _hash(destination / 'audit.json')})
    _progress('full_target_build_completed', directory=str(destination), **summary)
    return destination


def read_full_target_run(catalog, *, directory=None):
    identity = build_identity(catalog)
    pin = canonical_sha256(identity)
    parent = _within(catalog.root, catalog.root / 'outputs/full_target_panel')
    if directory is None:
        pointer = json.loads((parent / 'LATEST.json').read_text())
        directory = _within(parent, parent / pointer['path'])
        if pointer.get('build_identity_sha256') != pin or _hash(directory / 'audit.json') != pointer['audit_sha256']:
            raise DataContractError('Full target pointer changed or is stale')
    directory = _within(parent, directory)
    audit = json.loads((directory / 'audit.json').read_text())
    if (audit.get('status') != 'COMPLETE' or audit.get('build_identity_sha256') != pin
            or set(audit.get('years', {})) != {str(y) for y in YEARS}):
        raise DataContractError('Full target run is stale or incomplete')
    _verify_artifacts(directory, audit['artifacts'])
    _verify_artifacts(directory, audit['years'])
    required = {'monthly_returns', 'targets', 'target_diagnostics', 'target_state',
                'build_identity.json', 'source_plan.json', 'preserved_source_manifest.json',
                'ret_month_class_grid', 'ret_class_payoffs', *SMALL_REPORTS}
    if (not required <= set(audit['artifacts'])
            or json.loads((directory / 'build_identity.json').read_text()) != identity):
        raise DataContractError('Full target run is missing required artifacts or its current identity')
    for year in YEARS:
        _year_audit(directory / 'years' / str(year), pin)
    verify_preserved_inputs(catalog.root, json.loads((directory / 'preserved_source_manifest.json').read_text()))
    return {'directory': directory, 'audit': audit, 'tables': {
        name: pd.read_parquet(directory / audit['artifacts'][name]['path']) for name in SMALL_REPORTS}}


def read_full_targets(run):
    """Explicitly load the saved labels, separately from lazy feature/daily data."""
    _verify_artifacts(run['directory'], run['audit']['artifacts'])
    return _load_target_state(run['directory'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'build', 'verify'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--years', type=int, nargs='+')
    args = parser.parse_args()
    catalog = SourceCatalog(args.root)
    if args.action == 'plan':
        _, _, _, totals = plan_full_targets(catalog)
        _progress('target_read_plan', **totals)
    elif args.action == 'build':
        build_full_targets(catalog, years=args.years)
    else:
        run = read_full_target_run(catalog)
        _progress('full_targets_verified', directory=str(run['directory']), **run['audit']['summary'])


if __name__ == '__main__':
    main()
