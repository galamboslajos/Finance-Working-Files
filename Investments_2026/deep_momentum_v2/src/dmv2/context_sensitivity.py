"""Explicit cached-input context sensitivity; preserve the pinned 90% reference.

Only past daily feature histories and completed signal-only feature panels are
decoded. Raw daily backtest artifacts, paper inputs and technical values are
copied unchanged. This does not fit preprocessing, inspect targets or backtest.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import shutil
import tempfile
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .catalog import SourceCatalog, canonical_sha256
from .context_features import SENSITIVITY_CONTEXT_POLICY, balanced_cross_sectional_bases
from .data import session_dates
from .features import FeaturePanel
from .full_features import compute_compact_context_features
from .full_panel import (_artifact, _hash, _json, _limits, _progress, _verify_artifacts,
                         _verify_year, _within, _write_table, build_identity,
                         read_pinned_feature_run)
from .identity_policy import PROVISIONAL_IDENTITY_POLICY
from .panel_coverage import StreamingPanelCoverage
from .universe import DataContractError

KEYS = ['source_security_id', 'signal_date']
REPLACED = {'features', 'feature_coverage', 'context_diagnostics', 'daily_benchmark_diagnostics'}
REFERENCE_POLICY = 'available_cohort_90pct'


def configuration(catalog):
    value = json.loads((catalog.root / 'configs/context_sensitivity_80.json').read_text())
    if (value.get('schema_version') != 1 or value.get('context_policy') != SENSITIVITY_CONTEXT_POLICY
            or value.get('start_year') != 2010 or value.get('end_year') != 2023
            or value.get('reference_policy') != REFERENCE_POLICY
            or value.get('primary_policy_changed') is not False
            or value.get('universe') != 'combined' or value.get('history_sessions') != 503
            or value.get('identity_policy') != PROVISIONAL_IDENTITY_POLICY
            or value.get('normalization_policy') != 'signal_qualified_common_momentum_v1'):
        raise DataContractError('Sensitivity must preserve the approved 90% reference and 2010-2023 scope')
    active = build_identity(catalog)
    if active['configuration']['context_policy'] != REFERENCE_POLICY:
        raise DataContractError('The primary 90% configuration must remain active')
    if any(value['execution'].get(k) is not False for k in
           ('model_fitting', 'imputation_fitting', 'backtesting', 'cloud_mutations')):
        raise DataContractError('Sensitivity cannot fit or run portfolios/cloud mutations')
    if value['execution'].get('workers') != 1:
        raise DataContractError('Sensitivity is bounded to one local worker')
    for name in ('maximum_wall_seconds', 'maximum_peak_rss_bytes', 'maximum_output_bytes',
                 'maximum_projected_read_bytes'):
        if type(value['execution'].get(name)) is not int or value['execution'][name] <= 0:
            raise DataContractError('Sensitivity requires explicit positive resource ceilings')
    return value, active


def read_reference_run(catalog, *, verify_year_files=True):
    value = json.loads((catalog.root / 'configs/context_sensitivity_80.json').read_text())
    ref = value['reference']
    return read_pinned_feature_run(catalog,
        directory=_within(catalog.root, catalog.root / ref['directory']),
        expected_audit_sha256=ref['audit_sha256'],
        expected_identity_sha256=ref['build_identity_sha256'],
        worker_snapshot_directory=_within(catalog.root, catalog.root / ref['worker_snapshot_directory']),
        verify_year_files=verify_year_files)


def sensitivity_identity(catalog):
    value, active = configuration(catalog)
    files = ('context_sensitivity.py', 'context_comparison.py')
    active['sensitivity_workers'] = {n: _hash(Path(__file__).parent / n) for n in files}
    return {'schema_version': 1, 'configuration': value, 'active_reference_contract': active,
            'source_pins': catalog.provenance(), 'context_policy': SENSITIVITY_CONTEXT_POLICY}


def rebuild_cached_year(reference, *, context_policy=SENSITIVITY_CONTEXT_POLICY):
    """Recalculate context only; retain every other feature and ledger field."""
    if context_policy != SENSITIVITY_CONTEXT_POLICY:
        raise DataContractError('This sensitivity is explicitly the 80% case')
    audit = json.loads((reference / 'audit.json').read_text())
    _verify_artifacts(reference, audit['artifacts'])
    specs = json.loads((reference.parents[1] / 'feature_dictionary.json').read_text())
    feature = pq.read_table(reference / 'features.parquet')
    ledger = pq.read_table(reference / 'feature_coverage.parquet')
    original = pq.read_table(reference / 'original_cohort.parquet')
    if not feature.select(KEYS).equals(ledger.select(KEYS)):
        raise DataContractError('Cached feature/ledger key order differs')
    cov = ledger.to_pandas()
    bases = list(balanced_cross_sectional_bases())
    past_values = feature.select([*KEYS, *bases]).to_pandas().merge(
        cov[[*KEYS, 'intrinio_security_id', 'identity_eligible', 'requested']],
        on=KEYS, validate='one_to_one')
    # Collision/unresolved routes never enter the native-key indicator relation.
    past_values = past_values.loc[past_values.identity_eligible & past_values.requested,
                                  ['intrinio_security_id', 'signal_date', *bases]]
    if past_values.duplicated(['intrinio_security_id', 'signal_date']).any():
        raise DataContractError('Cached admitted native indicator keys collide')
    history = pq.read_table(reference / 'feature_history_daily.parquet')
    calendar = pq.read_table(reference / 'exchange_calendar.parquet')
    daily_cohorts = pq.read_table(reference / 'daily_cohorts.parquet')
    context = compute_compact_context_features(SimpleNamespace(table=past_values), history,
        session_dates(calendar), signal_cohorts=original.to_pandas(),
        requested_security_ids=audit['feature_provenance']['requested_ids'],
        daily_cohorts=daily_cohorts, identity_policy=PROVISIONAL_IDENTITY_POLICY,
        context_policy=context_policy)
    expected = feature.select(KEYS).to_pandas()
    if (context.table.duplicated(KEYS).any() or
            set(map(tuple, context.table[KEYS].to_numpy())) != set(map(tuple, expected.to_numpy()))):
        raise DataContractError('Sensitivity context changed original signal keys')
    aligned = context.table.set_index(KEYS).reindex(pd.MultiIndex.from_frame(expected)).reset_index()
    pd.testing.assert_frame_equal(expected, aligned[KEYS], check_exact=True)
    for spec in context.specs:
        name = spec['name']
        index = feature.schema.get_field_index(name)
        feature = feature.set_column(index, feature.schema.field(index),
            pa.array(aligned[name], type=pa.float64(), from_pandas=True))
    finite = np.isfinite(feature.drop(KEYS).to_pandas().to_numpy(dtype=float)).sum(axis=1)
    status = np.where(~cov.identity_eligible, 'identity_unresolved',
        np.where(~cov.requested, 'not_requested', np.where(finite > 0, 'partial_features', 'no_usable_features')))
    for name, values in (('available_feature_count', finite), ('feature_status', status)):
        i = ledger.schema.get_field_index(name)
        ledger = ledger.set_column(i, ledger.schema.field(i),
                                   pa.array(values, type=ledger.schema.field(i).type))
    summary = deepcopy(audit['summary'])
    summary.update(finite_feature_values=int(finite.sum()), feature_status_counts=dict(Counter(status)))
    provenance = deepcopy(audit['feature_provenance'])
    provenance.update(context=context.provenance, context_rebuilt_from_verified_cache=True,
        paper_and_technical_recalculated=False, outcomes_consumed=False)
    benchmark = context.coverage.feature_name.eq('benchmark_daily_ew_return_1s')
    diagnostics = {'context': context.coverage.loc[~benchmark].reset_index(drop=True),
                   'daily_benchmark': context.coverage.loc[benchmark].reset_index(drop=True)}
    return FeaturePanel(feature, ledger, tuple(specs), diagnostics, summary, provenance), audit


def _stored_panel(directory, specs, audit):
    diagnostics = json.loads((directory / 'diagnostic_metadata.json').read_text())
    for name, artifact in audit['artifacts'].items():
        if name.endswith('_diagnostics'):
            diagnostics[name[:-12]] = pq.read_table(directory / artifact['path']).to_pandas()
    return FeaturePanel(pq.read_table(directory / 'features.parquet'),
        pq.read_table(directory / 'feature_coverage.parquet'), tuple(specs), diagnostics,
        audit['summary'], audit['feature_provenance'])


def build_sensitivity(catalog, *, years=None):
    identity = sensitivity_identity(catalog)
    pin = canonical_sha256(identity)
    cfg = identity['configuration']
    reference = read_reference_run(catalog)
    pointer_path = catalog.root / 'outputs/full_feature_panel/LATEST.json'
    primary_pointer = pointer_path.read_bytes()
    destination = _within(catalog.root, catalog.root / 'outputs/context_sensitivity' /
                           ('80pct-2010-2023-' + pin.split(':')[1][:16]))
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / 'build_identity.json').exists():
        if json.loads((destination / 'build_identity.json').read_text()) != identity:
            raise DataContractError('Sensitivity run identity changed')
    else:
        _json(destination / 'build_identity.json', identity)
    (destination / 'years').mkdir(exist_ok=True)
    specs = json.loads((reference['directory'] / 'feature_dictionary.json').read_text())
    _json(destination / 'feature_dictionary.json', specs)
    started, read_bytes = time.monotonic(), 0
    selected = tuple(range(2010, 2024)) if years is None else tuple(years)
    if not selected or selected != tuple(sorted(set(selected))) or any(y not in range(2010, 2024) for y in selected):
        raise DataContractError('Explicit ordered 2010-2023 sensitivity years required')
    for year in selected:
        target = destination / 'years' / str(year)
        source = reference['directory'] / 'years' / str(year)
        if target.exists():
            _verify_year(target, pin)
            _progress('sensitivity_year_reused', year=year)
            continue
        _progress('sensitivity_year_started', year=year, destination=str(destination))
        panel, original_audit = rebuild_cached_year(source)
        read_bytes += sum(item['size_bytes'] for item in original_audit['artifacts'].values())
        temporary = Path(tempfile.mkdtemp(dir=destination / 'years', prefix=f'.{year}-'))
        artifacts = {}
        for name, item in original_audit['artifacts'].items():
            if name in REPLACED:
                continue
            path = temporary / item['path']
            shutil.copyfile(source / item['path'], path)
            artifacts[name] = deepcopy(item)
            if _hash(path) != item['sha256']:
                raise DataContractError('Preserved cached artifact changed during copy')
        for name, value in (('features', panel.table), ('feature_coverage', panel.coverage),
                            ('context_diagnostics', panel.diagnostics['context']),
                            ('daily_benchmark_diagnostics', panel.diagnostics['daily_benchmark'])):
            _write_table(temporary, name, value, artifacts)
        # Check sources again after use so an ordinary concurrent mutation fails.
        _verify_artifacts(source, original_audit['artifacts'])
        audit = {'build_identity_sha256': pin, 'year': year, 'artifacts': artifacts,
            'summary': panel.summary, 'feature_provenance': panel.provenance,
            'reference_year_audit_sha256': _hash(source / 'audit.json'),
            'raw_daily_storage': original_audit['raw_daily_storage'],
            'loading_provenance': {'mode': 'verified_cached_past_inputs', 'raw_source_reads': False,
                                  'reference_loading_provenance': original_audit['loading_provenance']},
            'loading_stats': {'cumulative_logical_cached_artifact_bytes': read_bytes,
                             'projected_read_accounting': 'Five artifact passes budgeted conservatively; not measured physical I/O'},
            'completed_at': datetime.now(timezone.utc).isoformat()}
        _json(temporary / 'audit.json', audit)
        temporary.rename(target)
        del panel
        gc.collect()
        pa.default_memory_pool().release_unused()
        _progress('sensitivity_year_completed', year=year, rows=audit['summary']['rows'])
        _limits(cfg, started, destination, 5 * read_bytes)
    if all((destination / 'years' / str(y) / 'audit.json').exists() for y in range(2010, 2024)):
        signals = json.loads((reference['directory'] / 'source_plan.json').read_text())['signal_dates']
        aggregation = StreamingPanelCoverage(specs, expected_signals=signals, start_year=2010, end_year=2023)
        year_artifacts = {}
        for year in range(2010, 2024):
            directory = destination / 'years' / str(year)
            audit = _verify_year(directory, pin)
            aggregation.add_panel(_stored_panel(directory, specs, audit),
                                   original_membership=pq.read_table(directory / 'original_cohort.parquet'))
            year_artifacts[str(year)] = _artifact(destination, directory / 'audit.json')
        reports = aggregation.finish()
        if reports['summary']['original_stock_months'] != 152307 or reports['summary']['paper_qualified_stock_months'] != 140777:
            raise DataContractError('Sensitivity changed original or paper-qualified rows')
        report_artifacts = {}
        for name, frame in reports['tables'].items():
            path = destination / (name + '.csv')
            frame.to_csv(path, index=False)
            report_artifacts[name] = _artifact(destination, path)
        _json(destination / 'coverage_summary.json', reports['summary'])
        for name in ('build_identity.json', 'feature_dictionary.json', 'coverage_summary.json'):
            report_artifacts[name] = _artifact(destination, destination / name)
        resources = _limits(cfg, started, destination, 5 * read_bytes)
        result = {'version': 1, 'status': 'COMPLETE', 'build_identity_sha256': pin,
            'configuration': cfg, 'summary': reports['summary'], 'years': year_artifacts,
            'reports': report_artifacts, 'resources': resources, 'context_policy': SENSITIVITY_CONTEXT_POLICY,
            'reference_directory': str(reference['directory']), 'knowledge_time_verified': False,
            'accounting_certified': False, 'model_ready': False, 'daily_backtest_ready': False,
            'models_fitted': False, 'imputation_fitted': False, 'primary_90pct_pointer_changed': False}
        _json(destination / 'audit.json', result)
        if pointer_path.read_bytes() != primary_pointer:
            raise DataContractError('Primary 90% pointer changed during sensitivity')
        _json(destination.parent / 'LATEST_80.json', {'path': destination.name,
            'build_identity_sha256': pin, 'audit_sha256': _hash(destination / 'audit.json')})
        _progress('sensitivity_build_completed', destination=str(destination), summary=reports['summary'])
    return destination


def read_sensitivity_run(catalog):
    identity = sensitivity_identity(catalog)
    pin = canonical_sha256(identity)
    parent = _within(catalog.root, catalog.root / 'outputs/context_sensitivity')
    pointer = json.loads((parent / 'LATEST_80.json').read_text())
    directory = _within(parent, parent / pointer['path'])
    audit = json.loads((directory / 'audit.json').read_text())
    if (pointer['build_identity_sha256'] != pin or audit['build_identity_sha256'] != pin
            or pointer['audit_sha256'] != _hash(directory / 'audit.json') or audit['status'] != 'COMPLETE'
            or set(audit['years']) != {str(y) for y in range(2010, 2024)}):
        raise DataContractError('Sensitivity pointer is stale or incomplete')
    _verify_artifacts(directory, audit['reports'])
    _verify_artifacts(directory, audit['years'])
    for year in range(2010, 2024):
        _verify_year(directory / 'years' / str(year), pin)
    return {'directory': directory, 'audit': audit,
            'tables': {n: pd.read_csv(directory / v['path']) for n, v in audit['reports'].items()
                       if v['path'].endswith('.csv')}}


def read_comparison_run(catalog):
    """Verify both panels and the separately pinned descriptive reports."""
    sensitivity = read_sensitivity_run(catalog)
    reference = read_reference_run(catalog)
    pointer = json.loads((catalog.root / 'outputs/context_sensitivity/LATEST_80.json').read_text())
    directory = sensitivity['directory'] / 'comparison'
    path = directory / 'comparison_manifest.json'
    if not path.exists() or 'comparison_manifest_sha256' not in pointer:
        raise FileNotFoundError('No completed pinned sensitivity comparison')
    if _hash(path) != pointer['comparison_manifest_sha256']:
        raise DataContractError('Sensitivity comparison manifest changed')
    manifest = json.loads(path.read_text())
    for name, run in (('reference_audit', reference), ('sensitivity_audit', sensitivity)):
        if manifest[name]['sha256'] != _hash(run['directory'] / 'audit.json'):
            raise DataContractError('Comparison uses a different completed panel')
    for name, item in manifest['reports'].items():
        artifact = _within(directory, directory / name)
        if artifact.stat().st_size != item['bytes'] or _hash(artifact) != item['sha256']:
            raise DataContractError('Comparison report changed: ' + name)
    summary = json.loads((directory / 'comparison_summary.json').read_text())
    if summary['status'] != 'CONTEXT_80_VS_90_COMPARISON_VERIFIED':
        raise DataContractError('Sensitivity comparison is incomplete')
    return {'reference': reference, 'sensitivity': sensitivity, 'directory': directory,
            'summary': summary, 'tables': {name: pd.read_csv(directory / filename)
                                          for name, filename in summary['reports'].items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('build', 'verify', 'compare'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--years', type=int, nargs='+')
    args = parser.parse_args()
    catalog = SourceCatalog(args.root)
    if args.action == 'build':
        build_sensitivity(catalog, years=args.years)
    else:
        sensitivity, reference = read_sensitivity_run(catalog), read_reference_run(catalog)
        if args.action == 'compare':
            from .context_comparison import compare_context_runs
            result = compare_context_runs(reference['directory'], sensitivity['directory'],
                                           sensitivity['directory'] / 'comparison')
            pointer_path = catalog.root / 'outputs/context_sensitivity/LATEST_80.json'
            pointer = json.loads(pointer_path.read_text())
            if pointer['path'] != sensitivity['directory'].name:
                raise DataContractError('Sensitivity pointer changed during comparison')
            pointer['comparison_manifest_sha256'] = _hash(
                sensitivity['directory'] / 'comparison/comparison_manifest.json')
            _json(pointer_path, pointer)
            _progress('comparison_completed', **{k: result[k] for k in (
                'gained_context_values_original', 'lost_context_values_original',
                'changed_shared_context_values_original', 'sensitivity_ever_finite_predictors')})
        else:
            _progress('sensitivity_verified', directory=str(sensitivity['directory']))


if __name__ == '__main__':
    main()
