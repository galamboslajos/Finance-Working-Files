"""Bounded notebook feature stage; data/code/dictionary pins saved separately."""
from dataclasses import dataclass
import errno
import hashlib
import json
from pathlib import Path
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq

from .catalog import canonical_sha256
from .feature_loading import load_feature_sample
from .features import build_feature_panel
from .loading import ReadBudget


@dataclass(frozen=True)
class FeatureRun:
    sample: object
    panel: object
    report: dict


def run_feature_sample(catalog, settings):
    from .workflow import validate_workflow, worker_code_provenance,workflow_identity_policy
    validate_workflow(settings)
    controls = settings['features']
    if not controls['enabled']:
        return None
    months = settings['monthly_returns']
    sample = load_feature_sample(catalog, start_month=months['start_month'], end_month=months['end_month'],
        universe=settings['universe']['engineering_sample'], sample_size=months['sample_size'],
        history_sessions=controls['history_sessions'], budget=ReadBudget(**controls['read_budget']),
        identity_policy=workflow_identity_policy(settings))
    panel = build_feature_panel(sample.daily_prices.table, sample.calendar, sample.snapshots,
        requested_security_ids=sample.requested_ids,identity_policy=workflow_identity_policy(settings),
        normalization_policy=controls['cohort_policy'])
    report = {'schema_version': 1, 'workflow_sha256': canonical_sha256(settings),
        'source_pins': catalog.provenance(), 'code_provenance': worker_code_provenance(),
        'summary': panel.summary, 'feature_provenance': panel.provenance,
        'loading_stats': sample.loading_stats, 'loading_provenance': sample.loading_provenance,
        'feature_dictionary': list(panel.specs),
        'signal_cohorts': [{'as_of': s.as_of, 'summary': s.summary, 'pit': s.provenance,
                           'constituent_ledger': s.table.to_pylist(),
                           'membership_exclusions': s.excluded.to_pylist()} for s in sample.snapshots],
        'artifact_roles': {'feature_history_daily.parquet': 'Independent past-only OHLCV history ending at latest signal',
            'features.parquet': 'Signal-date keys and registered feature values only; no target columns',
            'feature_coverage.parquet': 'Original-cohort identity/request/availability audit; never a predictor source',
            'feature_dictionary.json': 'Frozen formulas, dependencies, windows and model-column exclusions',
            'exchange_calendar.parquet': 'Independent exchange-session metadata with boundary evidence',
            'source_calendar.parquet': 'Original vendor calendar retained',
            'audit.json': 'Code/runtime/source/request/coverage provenance; no model fit'}}
    return FeatureRun(sample, panel, report)


def save_feature_sample(root, run):
    root = Path(root).resolve()
    report = run.report
    pins = {field: report[field] for field in ('workflow_sha256', 'source_pins', 'code_provenance')}
    pins['feature_dictionary_sha256'] = canonical_sha256(run.panel.specs)
    identifier = canonical_sha256(pins).split(':')[1][:16]
    parent = (root/'outputs/notebook_audits').resolve()
    destination = parent/f'features-{run.sample.signal_dates[0]}-{run.sample.signal_dates[-1]}-{identifier}'
    if not parent.is_relative_to(root) or not destination.resolve().is_relative_to(root):
        raise ValueError('Feature output directory leaves the project')
    parent.mkdir(parents=True, exist_ok=True)
    buffers, artifacts = {}, {}
    tables = {'feature_history_daily.parquet': run.sample.daily_prices.table,
              'features.parquet': run.panel.table, 'feature_coverage.parquet': run.panel.coverage,
              'exchange_calendar.parquet': run.sample.calendar, 'source_calendar.parquet': run.sample.source_calendar}
    for name, frame in run.panel.diagnostics.items():
        tables[f'{name}_diagnostics.parquet'] = pa.Table.from_pandas(frame, preserve_index=False)
    for name, table in tables.items():
        sink = pa.BufferOutputStream()
        pq.write_table(table, sink, compression='zstd')
        buffers[name] = sink.getvalue().to_pybytes()
        artifacts[name] = {'sha256': hashlib.sha256(buffers[name]).hexdigest(), 'size_bytes': len(buffers[name]),
                           'rows': table.num_rows, 'columns': table.column_names}
    buffers['feature_dictionary.json'] = (json.dumps(list(run.panel.specs), indent=2, sort_keys=True, allow_nan=False)+'\n').encode()
    artifacts['feature_dictionary.json'] = {'sha256': hashlib.sha256(buffers['feature_dictionary.json']).hexdigest(),
                                           'size_bytes': len(buffers['feature_dictionary.json'])}
    buffers['audit.json'] = (json.dumps({**report, 'artifacts': artifacts}, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()

    def verify_existing():
        if not destination.is_dir():
            raise ValueError('Feature artifact is not a directory; preserve and investigate')
        for name, value in buffers.items():
            path = destination/name
            if not path.resolve().is_relative_to(root) or not path.is_file() or path.read_bytes() != value:
                raise ValueError('Feature evidence changed under unchanged pins; preserve and investigate')
        return destination

    if destination.exists():
        return verify_existing()
    with tempfile.TemporaryDirectory(dir=parent, prefix='.features-tmp-') as temp:
        directory = Path(temp)
        for name, value in buffers.items():
            (directory/name).write_bytes(value)
        try:
            directory.rename(destination)
        except OSError as error:
            if error.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                raise
            return verify_existing()
    return destination
