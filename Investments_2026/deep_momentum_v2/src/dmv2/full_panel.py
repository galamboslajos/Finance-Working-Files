"""Explicit annual full-panel build and verified, lightweight notebook access.

Monthly signals and original historical denominators are preserved. This module
does not fit preprocessing/models or run portfolios. Daily source rows live in
separate artifacts; numeric features never receive post-signal future outcomes.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import tempfile
import time

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from .catalog import SourceCatalog, canonical_sha256
from .features import FeaturePanel, registered_feature_specs
from .context_features import PRIMARY_CONTEXT_POLICY
from .identity_policy import PROVISIONAL_IDENTITY_POLICY
from .paper_features import COMMON_MOMENTUM_POLICY, PAPER_FEATURE_NAMES
from .universe import DataContractError


VERSION = 1
KEYS = ('source_security_id', 'signal_date')
WORKERS = ('catalog.py', 'loading.py', 'universe.py', 'identity_policy.py', 'data.py',
    'exchange_sessions.py', 'quote_sessions.py', 'monthly.py', 'monthly_loading.py',
    'features.py', 'daily_features.py', 'paper_features.py', 'context_features.py',
    'full_loading.py', 'full_features.py', 'panel_coverage.py', 'full_panel.py')


def _specs():
    from .daily_features import daily_feature_specs
    from .paper_features import paper_feature_specs
    from .context_features import context_feature_specs
    return (*daily_feature_specs(), *paper_feature_specs(COMMON_MOMENTUM_POLICY),
            *context_feature_specs())


def _hash(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def _json(path, value):
    path = Path(path)
    text = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n'
    with tempfile.NamedTemporaryFile('w', dir=path.parent, prefix='.' + path.name, delete=False) as stream:
        stream.write(text)
        temporary = Path(stream.name)
    os.replace(temporary, path)


def _within(root, path):
    root, path = Path(root).resolve(), Path(path).resolve()
    if not path.is_relative_to(root):
        raise DataContractError('Full-panel artifact leaves the project directory')
    return path


def _configuration(catalog):
    settings = json.loads((catalog.root / 'configs/full_panel_build.json').read_text())
    normalization = json.loads((catalog.root / 'configs/normalization_protocol.json').read_text())
    workflow = json.loads((catalog.root / 'configs/research_workflow.json').read_text())
    if (settings.get('schema_version') != VERSION or settings.get('start_year') != 2010
            or settings.get('end_year') != 2023 or settings.get('universe') != 'combined'
            or settings.get('history_sessions') != 503
            or settings.get('identity_policy') != PROVISIONAL_IDENTITY_POLICY
            or settings.get('normalization_policy') != COMMON_MOMENTUM_POLICY
            or settings.get('context_policy') != PRIMARY_CONTEXT_POLICY
            or settings.get('include_outcome_daily') is not True):
        raise DataContractError('Full-panel configuration differs from the approved scope')
    execution = settings['execution']
    if (execution.get('workers') != 1 or execution.get('backend') != 'local_downloaded_sources'
            or any(execution.get(key) is not False for key in
                   ('model_fitting', 'imputation_fitting', 'backtesting', 'cloud_mutations'))):
        raise DataContractError('Full-panel build cannot fit, backtest or mutate cloud resources')
    for key in ('maximum_wall_seconds', 'maximum_peak_rss_bytes', 'maximum_output_bytes',
                'maximum_projected_read_bytes'):
        if type(execution.get(key)) is not int or execution[key] <= 0:
            raise DataContractError('Explicit positive resource ceilings are required')
    if (workflow['research_start'] != '2010-01-01'
            or workflow['training'] != {'mode': 'rolling', 'train_years': 10,
                                       'test_years': 1, 'last_test_year': 2023}
            or normalization['paper_normalization']['policy'] != COMMON_MOMENTUM_POLICY
            or normalization['chronology']['initial_outcome_year'] != 2010
            or normalization['context_policy']['active_worker_policy'] != settings['context_policy']
            or normalization['context_policy'].get('minimum_coverage_fraction') != .8
            or normalization['context_policy'].get('sensitivity_worker_policy') != 'available_cohort_90pct'
            or normalization['context_policy'].get('sensitivity_coverage_fraction') != .9
            or normalization['context_policy'].get('exhaustive_request_required') is not True
            or workflow['features'].get('context_policy', 'complete_cohort') != settings['context_policy']):
        raise DataContractError('Canonical chronology and normalization must agree')
    return settings, normalization, workflow


def build_identity(catalog):
    settings, normalization, workflow = _configuration(catalog)
    catalog.verify(hashes=False)
    directory = Path(__file__).parent
    return {'version': VERSION, 'configuration': settings,
        'normalization': normalization, 'workflow': workflow,
        'source_pins': catalog.provenance(),
        'workers': {name: _hash(directory / name) for name in WORKERS},
        'runtime': {'python': platform.python_version(), **{name: version(name)
                    for name in ('numpy', 'pandas', 'pyarrow', 'exchange-calendars')}},
        'feature_dictionary_sha256': canonical_sha256(_specs())}


def _artifact(directory, path, **extra):
    directory = Path(directory).resolve()
    path = _within(directory, path)
    return {'path': str(path.relative_to(directory)), 'sha256': _hash(path),
            'size_bytes': path.stat().st_size, **extra}


def _verify_artifacts(directory, artifacts):
    for name, item in artifacts.items():
        path = _within(directory, Path(directory) / item['path'])
        if (not path.is_file() or path.stat().st_size != item['size_bytes']
                or _hash(path) != item['sha256']):
            raise DataContractError(f'Full-panel artifact changed: {name}')


def _write_table(directory, name, value, artifacts):
    if isinstance(value, pd.DataFrame):
        value = pa.Table.from_pandas(value, preserve_index=False)
    if not isinstance(value, pa.Table):
        raise TypeError('Audit table must be Arrow or pandas')
    path = directory / (name + '.parquet')
    pq.write_table(value, path, compression='zstd', row_group_size=65536)
    artifacts[name] = _artifact(directory, path, rows=value.num_rows,
                                 columns=value.column_names)


def _original(snapshots):
    parts = []
    for snapshot in snapshots:
        part = snapshot.table
        if 'signal_date' not in part.column_names:
            part = part.append_column('signal_date', pa.array([snapshot.as_of] * part.num_rows))
        parts.append(part)
    return pa.concat_tables(parts, promote_options='default')


def _daily_storage_views(sample, year):
    """Keep complete warmup separately from disjoint outcome-year daily rows.

    A newly admitted December member's signal quote may be absent from the
    preceding year's requested-ID set. Never discard its annual raw history.
    Only the disjoint view can be concatenated across years without overlap.
    """
    raw = sample.raw_daily_prices.table
    feature_history = sample.feature_daily_prices.table
    if feature_history.num_rows and pc.max(feature_history['trading_date']).as_py() > sample.signal_dates[-1]:
        raise DataContractError('Feature-history artifact exceeds its final signal')
    core = raw if year == 2010 else raw.filter(pc.greater_equal(raw['trading_date'], f'{year}-01-01'))
    return {'daily_prices_raw': core, 'daily_history_raw': raw,
            'feature_history_daily': feature_history}


def _verify_year(directory, identity_sha256):
    path = directory / 'audit.json'
    audit = json.loads(path.read_text())
    if audit['build_identity_sha256'] != identity_sha256:
        raise DataContractError('Year artifact has stale source, code, runtime or policy pins')
    _verify_artifacts(directory, audit['artifacts'])
    return audit


def _rss_bytes():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(rss if platform.system() == 'Darwin' else rss * 1024)


def _limits(configuration, started, output, projected_read_bytes):
    execution = configuration['execution']
    current = {'maximum_wall_seconds': time.monotonic() - started,
        'maximum_peak_rss_bytes': _rss_bytes(),
        'maximum_output_bytes': sum(p.stat().st_size for p in output.rglob('*') if p.is_file()),
        'maximum_projected_read_bytes': projected_read_bytes}
    for key, actual in current.items():
        if actual > execution[key]:
            raise DataContractError(f'Full-panel resource ceiling exceeded: {key}={actual}')
    return current


def _progress(event, **values):
    print(json.dumps({'event': event, **values}, sort_keys=True, allow_nan=False), flush=True)


def build_full_panel(catalog, *, years=None):
    from .full_loading import plan_full_feature_sources, load_feature_year
    from .full_features import build_full_feature_panel
    identity = build_identity(catalog)
    identity_hash = canonical_sha256(identity)
    configuration = identity['configuration']
    parent = _within(catalog.root, catalog.root / 'outputs/full_feature_panel')
    parent.mkdir(parents=True, exist_ok=True)
    destination = _within(catalog.root, parent / ('2010-2023-' + identity_hash.split(':')[1][:16]))
    destination.mkdir(exist_ok=True)
    identity_path = destination / 'build_identity.json'
    if identity_path.exists():
        if json.loads(identity_path.read_text()) != identity:
            raise DataContractError('Existing run identity differs; preserve and investigate')
    else:
        _json(identity_path, identity)
    started = time.monotonic()
    plan = plan_full_feature_sources(catalog, start_year=2010, end_year=2023,
        universe='combined', history_sessions=503, identity_policy=PROVISIONAL_IDENTITY_POLICY)
    _json(destination / 'source_plan.json', plan.provenance)
    selected = tuple(range(2010, 2024)) if years is None else tuple(years)
    if not selected or selected != tuple(sorted(set(selected))) or any(y not in range(2010, 2024) for y in selected):
        raise DataContractError('Requested build years must be distinct ordered 2010-2023 years')
    (destination / 'years').mkdir(exist_ok=True)
    dictionary_path = destination / 'feature_dictionary.json'
    _json(dictionary_path, list(_specs()))
    conservative_bytes = sum(request['price_estimate']['projected_compressed_bytes'] +
        request['membership_estimate']['projected_compressed_bytes'] for request in plan.year_requests.values())
    if conservative_bytes > configuration['execution']['maximum_projected_read_bytes']:
        raise DataContractError('Metadata-only full-run read estimate exceeds its resource plan')
    read_bytes = plan.loading_stats['total']['projected_compressed_bytes']
    for year in selected:
        final = destination / 'years' / str(year)
        if final.exists():
            audit = _verify_year(final, identity_hash)
            read_bytes += audit['loading_stats']['total']['projected_compressed_bytes']
            _progress('year_reused', year=year, rows=audit['summary']['rows'])
            continue
        _progress('year_started', year=year, destination=str(destination))
        sample = load_feature_year(plan, year, include_outcome_daily=True)
        read_bytes += sample.loading_stats['total']['projected_compressed_bytes']
        _progress('year_loaded', year=year, price_rows=sample.raw_daily_prices.table.num_rows,
                  feature_price_rows=sample.feature_daily_prices.table.num_rows,
                  requested_ids=len(sample.requested_ids))
        panel = build_full_feature_panel(sample.feature_daily_prices.table, sample.calendar,
            sample.snapshots, requested_security_ids=sample.requested_ids,
            daily_cohorts=sample.daily_cohorts, identity_policy=PROVISIONAL_IDENTITY_POLICY,
            normalization_policy=COMMON_MOMENTUM_POLICY,
            context_policy=configuration['context_policy'])
        original = _original(sample.snapshots)
        if panel.table.num_rows != original.num_rows:
            raise DataContractError('Original annual cohort denominator changed')
        if canonical_sha256(panel.specs) != identity['feature_dictionary_sha256']:
            raise DataContractError('Full-panel worker changed the active feature dictionary')
        daily_views = _daily_storage_views(sample, year)
        raw = daily_views['daily_prices_raw']
        temp = Path(tempfile.mkdtemp(dir=destination / 'years', prefix=f'.{year}-'))
        artifacts = {}
        _write_table(temp, 'features', panel.table, artifacts)
        _write_table(temp, 'feature_coverage', panel.coverage, artifacts)
        _write_table(temp, 'original_cohort', original, artifacts)
        for name, table in daily_views.items():
            _write_table(temp, name, table, artifacts)
        _write_table(temp, 'daily_cohorts', sample.daily_cohorts, artifacts)
        _write_table(temp, 'exchange_calendar', sample.calendar, artifacts)
        diagnostic_json = {}
        for key, value in panel.diagnostics.items():
            if isinstance(value, (pd.DataFrame, pa.Table)):
                _write_table(temp, key + '_diagnostics', value, artifacts)
            else:
                diagnostic_json[key] = value
        _json(temp / 'diagnostic_metadata.json', diagnostic_json)
        artifacts['diagnostic_metadata'] = _artifact(temp, temp / 'diagnostic_metadata.json')
        audit = {'build_identity_sha256': identity_hash, 'year': year,
            'summary': panel.summary, 'feature_provenance': panel.provenance,
            'loading_provenance': sample.loading_provenance,
            'loading_stats': sample.loading_stats, 'artifacts': artifacts,
            'raw_daily_storage': {'start': pc.min(raw['trading_date']).as_py() if raw.num_rows else None,
                'end': pc.max(raw['trading_date']).as_py() if raw.num_rows else None,
                'rows': raw.num_rows, 'separate_from_predictors': True,
                'complete_annual_warmup_preserved': True,
                'overlapping_history_artifact': 'daily_history_raw.parquet',
                'past_only_feature_history_artifact': 'feature_history_daily.parquet'},
            'completed_at': datetime.now(timezone.utc).isoformat()}
        _json(temp / 'audit.json', audit)
        temp.rename(final)
        del sample, panel, original, raw, daily_views, table
        gc.collect()
        pa.default_memory_pool().release_unused()
        _progress('year_completed', year=year, rows=audit['summary']['rows'], peak_rss_bytes=_rss_bytes())
        _limits(configuration, started, destination, read_bytes)
    if all((destination / 'years' / str(year) / 'audit.json').is_file() for year in range(2010, 2024)):
        finalize_full_panel(catalog, destination, started=started, projected_read_bytes=read_bytes)
    return destination


def finalize_full_panel(catalog, destination, *, started=None, projected_read_bytes=None):
    from .panel_coverage import StreamingPanelCoverage
    identity = build_identity(catalog)
    identity_hash = canonical_sha256(identity)
    destination = _within(catalog.root, destination)
    settings = identity['configuration']
    started = time.monotonic() if started is None else started
    audits, year_manifest = [], {}
    all_signals = []
    source_plan = json.loads((destination / 'source_plan.json').read_text())
    expected_signals = source_plan['signal_dates']
    if len(expected_signals) != 168 or source_plan['monthly_original_rows'] != 152307:
        raise DataContractError('Source plan differs from the independently audited original scope')
    for year in range(2010, 2024):
        directory = destination / 'years' / str(year)
        audit = _verify_year(directory, identity_hash)
        audits.append(audit)
        original = pq.read_table(directory / 'original_cohort.parquet')
        all_signals.extend(sorted(set(original['signal_date'].to_pylist())))
        year_manifest[str(year)] = _artifact(destination, directory / 'audit.json')
    if all_signals != expected_signals:
        raise DataContractError('Annual artifacts omit or change declared independent signal dates')
    aggregator = StreamingPanelCoverage(_specs(), expected_signals=expected_signals,
                                       start_year=2010, end_year=2023)
    for year, audit in zip(range(2010, 2024), audits):
        directory = destination / 'years' / str(year)
        diagnostics = json.loads((directory / 'diagnostic_metadata.json').read_text())
        for name, artifact in audit['artifacts'].items():
            if name.endswith('_diagnostics'):
                diagnostics[name[:-12]] = pq.read_table(directory / artifact['path']).to_pandas()
        panel = FeaturePanel(pq.read_table(directory / 'features.parquet'),
            pq.read_table(directory / 'feature_coverage.parquet'), _specs(),
            diagnostics, audit['summary'], audit['feature_provenance'])
        aggregator.add_panel(panel, original_membership=pq.read_table(directory / 'original_cohort.parquet'))
    reports = aggregator.finish()
    artifacts, summary = {}, reports['summary']
    if (summary['original_stock_months'] != 152307 or summary['paper_qualified_stock_months'] != 140777):
        raise DataContractError('Full-panel paper coverage differs from the independent endpoint audit')
    for name, value in reports['tables'].items():
        path = destination / (name + '.csv')
        value.to_csv(path, index=False)
        artifacts[name] = _artifact(destination, path, rows=len(value))
    for name in ('source_plan.json', 'build_identity.json', 'feature_dictionary.json'):
        artifacts[name] = _artifact(destination, destination / name)
    _json(destination / 'coverage_summary.json', summary)
    artifacts['coverage_summary'] = _artifact(destination, destination / 'coverage_summary.json')
    if projected_read_bytes is None:
        projected_read_bytes = sum(a['loading_stats']['total']['projected_compressed_bytes'] for a in audits)
    resources = _limits(settings, started, destination, projected_read_bytes)
    report = {'version': VERSION, 'status': 'COMPLETE', 'build_identity_sha256': identity_hash,
        'configuration': settings, 'years': year_manifest, 'reports': artifacts,
        'summary': summary, 'resources': resources,
        'knowledge_time_verified': False, 'accounting_certified': False,
        'model_ready': False, 'daily_backtest_ready': False,
        'models_fitted': False, 'imputation_fitted': False,
        'completed_at': datetime.now(timezone.utc).isoformat()}
    _json(destination / 'audit.json', report)
    _json(destination.parent / 'LATEST.json', {'path': destination.name,
        'build_identity_sha256': identity_hash, 'audit_sha256': _hash(destination / 'audit.json')})
    _progress('build_completed', destination=str(destination), summary=summary)
    return report


def read_full_feature_run(catalog, *, directory=None, verify_year_files=True):
    """Verify completed artifacts and return metadata/reports without price reads."""
    identity = build_identity(catalog)
    identity_hash = canonical_sha256(identity)
    parent = _within(catalog.root, catalog.root / 'outputs/full_feature_panel')
    pointer = None
    if directory is None:
        pointer = json.loads((parent / 'LATEST.json').read_text())
        directory = parent / pointer['path']
    directory = _within(parent, directory)
    if pointer is not None and (pointer['build_identity_sha256'] != identity_hash or
            _hash(directory / 'audit.json') != pointer['audit_sha256']):
        raise DataContractError('Full-panel stable pointer has stale or changed evidence')
    report = json.loads((directory / 'audit.json').read_text())
    if report['status'] != 'COMPLETE' or report['build_identity_sha256'] != identity_hash:
        raise DataContractError('Full-panel audit has stale source/code/policy/runtime pins')
    _verify_artifacts(directory, report['reports'])
    _verify_artifacts(directory, report['years'])
    if set(report['years']) != {str(y) for y in range(2010, 2024)}:
        raise DataContractError('Full-panel run is missing expected annual partitions')
    if verify_year_files:
        for year in range(2010, 2024):
            _verify_year(directory / 'years' / str(year), identity_hash)
    return {'directory': directory, 'audit': report,
        'tables': {name: pd.read_csv(directory / item['path']) for name, item in report['reports'].items()
                   if item['path'].endswith('.csv')}}


def read_pinned_feature_run(catalog, *, directory, expected_audit_sha256,
                            expected_identity_sha256, worker_snapshot_directory,
                            verify_year_files=True):
    """Read an explicitly pinned historical run without claiming current code.

    Both pins must come from the independently recorded reference contract, not
    from the archive being opened. Every historical worker is checked against a
    retained source snapshot. Current source, runtime, chronology and policy
    contracts still have to agree; only current worker/dictionary-code equality
    is replaced by verification of that explicit historical code snapshot.
    This does not relax the default current-run reader or make a stale run the
    active run, and it never executes the archived code.
    """
    if (not isinstance(expected_audit_sha256, str)
            or len(expected_audit_sha256) != 64
            or any(c not in '0123456789abcdef' for c in expected_audit_sha256)
            or not isinstance(expected_identity_sha256, str)
            or not expected_identity_sha256.startswith('sha256:')
            or len(expected_identity_sha256) != 71
            or any(c not in '0123456789abcdef' for c in expected_identity_sha256[7:])):
        raise DataContractError('Historical reference requires explicit SHA-256 audit and identity pins')
    parent = _within(catalog.root, catalog.root / 'outputs/full_feature_panel')
    directory = _within(parent, directory)
    audit_path = directory / 'audit.json'
    if not audit_path.is_file() or _hash(audit_path) != expected_audit_sha256:
        raise DataContractError('Historical reference audit differs from its explicit pin')
    report = json.loads(audit_path.read_text())
    identity_path = directory / 'build_identity.json'
    if not identity_path.is_file():
        raise DataContractError('Historical reference is missing its pinned build identity')
    identity = json.loads(identity_path.read_text())
    if canonical_sha256(identity) != expected_identity_sha256:
        raise DataContractError('Historical reference identity differs from its explicit pin')
    active_identity = build_identity(catalog)
    for key in ('version', 'configuration', 'normalization', 'workflow', 'source_pins', 'runtime'):
        if identity.get(key) != active_identity[key]:
            raise DataContractError(f'Historical reference differs from the active {key} contract')
    snapshot = Path(worker_snapshot_directory).resolve()
    if set(identity.get('workers', {})) != set(WORKERS):
        raise DataContractError('Historical reference does not pin every required worker')
    for name in WORKERS:
        path = _within(snapshot, snapshot / name)
        if not path.is_file() or _hash(path) != identity['workers'][name]:
            raise DataContractError(f'Historical reference worker snapshot changed: {name}')
    if (report.get('version') != VERSION or report.get('status') != 'COMPLETE'
            or report.get('build_identity_sha256') != expected_identity_sha256
            or report.get('configuration') != identity['configuration']
            or any(report.get(key) is not False for key in
                   ('models_fitted', 'imputation_fitted', 'model_ready', 'daily_backtest_ready'))):
        raise DataContractError('Historical reference is not a completed diagnostic-only run')
    required_reports = {'build_identity.json', 'source_plan.json', 'feature_dictionary.json',
                        'coverage_summary'}
    if not required_reports <= set(report.get('reports', {})):
        raise DataContractError('Historical reference is missing required report artifacts')
    _verify_artifacts(directory, report['reports'])
    _verify_artifacts(directory, report['years'])
    if set(report['years']) != {str(y) for y in range(2010, 2024)}:
        raise DataContractError('Historical reference is missing expected annual partitions')
    dictionary = json.loads((directory / 'feature_dictionary.json').read_text())
    if canonical_sha256(dictionary) != identity.get('feature_dictionary_sha256'):
        raise DataContractError('Historical reference feature dictionary differs from its identity pin')
    source_plan = json.loads((directory / 'source_plan.json').read_text())
    summary = report['summary']
    if (source_plan.get('source_pins') != identity['source_pins']
            or source_plan.get('start_year') != 2010 or source_plan.get('end_year') != 2023
            or source_plan.get('universe') != 'combined' or source_plan.get('history_sessions') != 503
            or source_plan.get('monthly_original_rows') != 152307
            or len(source_plan.get('signal_dates', ())) != 168
            or len(set(source_plan['signal_dates'])) != 168
            or summary.get('original_stock_months') != 152307
            or summary.get('paper_qualified_stock_months') != 140777
            or summary.get('completed_signals') != 168
            or summary.get('feature_dictionary_sha256') != identity['feature_dictionary_sha256']):
        raise DataContractError('Historical reference differs from the original 2010-2023 scope')
    if verify_year_files:
        for year in range(2010, 2024):
            audit = _verify_year(directory / 'years' / str(year), expected_identity_sha256)
            if audit.get('year') != year:
                raise DataContractError('Historical reference annual partition is mislabelled')
    return {'directory': directory, 'audit': report,
        'verification_mode': 'pinned_historical_reference',
        'reference_pins': {'audit_sha256': expected_audit_sha256,
                           'build_identity_sha256': expected_identity_sha256,
                           'worker_snapshot_directory': str(snapshot)},
        'tables': {name: pd.read_csv(directory / item['path']) for name, item in report['reports'].items()
                   if item['path'].endswith('.csv')}}


def read_feature_partition(run, *, signal_date, columns=None):
    """Read one exact monthly signal only; never silently load the full panel."""
    from .universe import iso_date
    from .monthly import _step_month
    signal_date = iso_date(signal_date, 'signal_date')
    outcome_year = int(_step_month(signal_date[:7], 1)[:4])
    if outcome_year not in range(2010, 2024):
        raise DataContractError('Requested signal is outside the full-panel scope')
    registered = {s['name'] for s in registered_feature_specs()}
    selected = [*KEYS, *(columns if columns is not None else sorted(registered))]
    if len(set(selected)) != len(selected) or not set(selected) <= registered | set(KEYS):
        raise DataContractError('Partition read contains unregistered columns')
    path = Path(run['directory']) / 'years' / str(outcome_year) / 'features.parquet'
    table = ds.dataset(path, format='parquet').to_table(columns=selected,
        filter=ds.field('signal_date') == signal_date)
    if not table.num_rows:
        raise DataContractError('Requested signal is not an exact materialized month end')
    return table.to_pandas()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'build', 'verify'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--years', type=int, nargs='+')
    args = parser.parse_args()
    catalog = SourceCatalog(args.root)
    if args.action == 'plan':
        from .full_loading import plan_full_feature_sources
        plan = plan_full_feature_sources(catalog)
        _progress('plan', monthly_original_rows=plan.provenance['monthly_original_rows'],
            signal_dates=plan.provenance['signal_dates'], year_requests=plan.year_requests,
            financial_values_read=False)
    elif args.action == 'build':
        build_full_panel(catalog, years=args.years)
    else:
        value = read_full_feature_run(catalog)
        _progress('verified', directory=str(value['directory']), summary=value['audit']['summary'])


if __name__ == '__main__':
    main()
