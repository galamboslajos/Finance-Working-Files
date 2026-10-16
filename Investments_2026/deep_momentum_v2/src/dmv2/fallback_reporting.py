"""Passive, bounded saved reports for the separately versioned fallback run.

The default verifies small identity/audit/summary receipts only. It never plans
sources, compiles quotes, decodes feature/price/target panels, deserializes models
or starts numerical work. Optional deep verification hashes source bytes only.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from .catalog import canonical_sha256
from .full_panel import _hash, _verify_artifacts, _within
from .universe import DataContractError


YEARS = tuple(range(2010, 2020))
ARMS = ('rf_paper15', 'xgb_paper15', 'rf_extended', 'xgb_extended')
MAX_JSON_BYTES = 16 * 1024**2
MAX_COMPARISON_BYTES = 1024**2


def _json(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size > MAX_JSON_BYTES:
        raise DataContractError('Saved fallback JSON exceeds its passive-read bound')
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise DataContractError('Saved fallback JSON must be a mapping')
    return value


def _pinned_json(base, item, *, expected_path=None):
    if not isinstance(item, dict) or not {'path', 'sha256', 'size_bytes'} <= set(item):
        raise DataContractError('Complete small report pin required')
    path = _within(base, Path(base) / item['path'])
    if (path.suffix != '.json' or item['size_bytes'] > MAX_JSON_BYTES
            or expected_path is not None and path != Path(expected_path).resolve()):
        raise DataContractError('Passive report pin redirects or exceeds its JSON scope')
    _verify_artifacts(base, {item['path']: item})
    return _json(path)


def _receipt(path, expected, label):
    if expected is not None and expected != 'sha256:' + _hash(path):
        raise DataContractError('Saved ' + label + ' audit differs from its explicit receipt')


def _deep_sources(root, identity):
    for name, pin in identity.get('workers', {}).items():
        if Path(name).name != name or _hash(Path(__file__).with_name(name)) != pin:
            raise DataContractError('Frozen fallback worker changed: ' + name)
    config = root / 'configs/data_limited_fallback.json'
    if identity.get('configuration_raw_sha256') != 'sha256:' + _hash(config):
        raise DataContractError('Frozen fallback configuration bytes changed')
    seen = set()
    for year, part in identity.get('years', {}).items():
        source = part['source_provenance']
        reference = _within(root, root / source['reference_primary']['directory'])
        root_audit = reference / 'audit.json'
        if _hash(root_audit) != source['reference_primary']['audit_sha256']:
            raise DataContractError('Original fallback source root audit changed')
        annual = reference / 'years' / str(year)
        if _hash(annual / 'audit.json') != source['annual_audit_sha256']:
            raise DataContractError('Original fallback source annual audit changed')
        for item in source['used_artifacts'].values():
            path = _within(annual, annual / item['path'])
            if path not in seen:
                _verify_artifacts(annual, {item['path']: item})
                seen.add(path)


def _model_report(run, parent_pin, expected_receipt, verify_sources):
    directory = run / 'models'
    if not (directory / 'audit.json').exists():
        if expected_receipt is not None:
            raise DataContractError('Explicitly pinned saved model audit is missing')
        return {'status': 'PENDING'}, pd.DataFrame()
    path = directory / 'audit.json'
    _receipt(path, expected_receipt, 'model')
    audit = _json(path)
    if audit.get('status') != 'COMPLETE':
        raise DataContractError('Saved fallback model audit has an unexpected incomplete status')
    if (audit.get('build_identity_sha256') != parent_pin
            or audit.get('summary', {}).get('fitted_arms') != 4
            or set(audit.get('arms', {})) != {*ARMS, 'uniform', 'train_prior'}):
        raise DataContractError('Saved fallback model audit changed its identity or four-arm scope')
    phase = _pinned_json(directory, audit['artifacts']['build_identity.json'],
        expected_path=directory / 'build_identity.json')
    if (canonical_sha256(phase) != audit.get('model_identity_sha256')
            or phase.get('parent_identity_sha256') != parent_pin):
        raise DataContractError('Saved fallback model phase identity changed')
    rows = []
    for name, pin in audit['arms'].items():
        expected = directory / ('arms' if name in ARMS else 'refs') / name / 'audit.json'
        arm = _pinned_json(directory, pin, expected_path=expected)
        if (arm.get('status') != 'COMPLETE' or arm.get('build_identity_sha256') != parent_pin
                or arm.get('model_identity_sha256') != audit['model_identity_sha256']):
            raise DataContractError('Saved fallback arm summary identity changed')
        if name in ARMS and arm.get('arm', {}).get('arm_id') != name:
            raise DataContractError('Saved fallback arm summary is mislabelled')
        rows.append({'arm_id': name, **arm['summary']})
    if verify_sources:
        from .fallback_models import read_fallback_models
        read_fallback_models(directory, identity_sha256=parent_pin,
            model_identity_sha256=audit['model_identity_sha256'])
        for name, pin in phase.get('workers', {}).items():
            if Path(name).name != name or _hash(Path(__file__).with_name(name)) != pin:
                raise DataContractError('Frozen fallback model worker changed')
        for annual in phase['data_plan']['years'].values():
            _verify_artifacts(run.parent.parent.parent, {
                item['path']: item for item in annual['source_pins']})
    return {'status': 'COMPLETE', **audit['summary']}, pd.DataFrame(rows)


def _backtest_report(run, parent_pin, expected_receipt, verify_sources):
    directory = run / 'backtest'
    if not (directory / 'audit.json').exists():
        if expected_receipt is not None:
            raise DataContractError('Explicitly pinned saved backtest audit is missing')
        return {'status': 'PENDING'}, pd.DataFrame()
    path = directory / 'audit.json'
    _receipt(path, expected_receipt, 'backtest')
    audit = _json(path)
    if audit.get('status') == 'STARTED':
        if audit.get('build_identity_sha256') != parent_pin:
            raise DataContractError('Started fallback backtest has a different parent identity')
        return {'status': 'STARTED'}, pd.DataFrame()
    if (audit.get('status') != 'COMPLETE' or audit.get('build_identity_sha256') != parent_pin
            or audit.get('scenarios') != 64
            or audit.get('research_status') != 'DATA_LIMITED_RESTRICTED_UNIVERSE'):
        raise DataContractError('Saved fallback backtest identity/scope changed')
    phase = _pinned_json(directory, audit['artifacts']['build_identity.json'],
        expected_path=directory / 'build_identity.json')
    if (canonical_sha256(phase) != audit.get('phase_identity_sha256')
            or phase.get('parent_identity_sha256') != parent_pin):
        raise DataContractError('Saved fallback backtest phase identity changed')
    item = audit['artifacts']['comparison.parquet']
    report = _within(directory, directory / item['path'])
    if report != directory / 'comparison.parquet' or item['size_bytes'] > MAX_COMPARISON_BYTES:
        raise DataContractError('Saved fallback comparison exceeds its small report scope')
    _verify_artifacts(directory, {item['path']: item})
    footer = pq.ParquetFile(report)
    if footer.metadata.num_rows != 64:
        raise DataContractError('Saved fallback comparison lost its 64 scenarios')
    comparison = footer.read().to_pandas()
    if (not {'arm', 'basket', 'variant', 'cost_bps'} <= set(comparison)
            or comparison.duplicated(['arm', 'basket', 'variant', 'cost_bps']).any()
            or set(comparison.arm) != set(ARMS)
            or set(comparison.basket) != {'class_1_10', 'class_2_9'}
            or set(comparison.variant) != {'unstopped', 'asset_stop_10pct'}
            or set(comparison.cost_bps) != {0, 5, 10, 25}):
        raise DataContractError('Saved fallback comparison grid changed')
    if verify_sources:
        for item in audit['artifacts'].values():
            _verify_artifacts(directory, {item['path']: item})
        _verify_artifacts(run.parent.parent.parent, {item['path']: item for item in phase['source_pins']})
        for name, pin in phase.get('workers', {}).items():
            if Path(name).name != name or _hash(Path(__file__).with_name(name)) != pin:
                raise DataContractError('Frozen fallback backtest worker changed')
    return {name: audit[name] for name in ('status', 'scenarios',
        'complete_proxy_paths', 'failed_closed_paths', 'execution_seconds')}, comparison


def read_data_limited_run(root, directory, *, identity_sha256,
                          expected_receipts=None, verify_sources=False):
    """Read saved reports only, optionally checking explicit final audit SHAs.

    A building run can be PARTIAL or have no root audit yet. Missing files are
    not swallowed: the caller may display pending only for FileNotFoundError.
    Integrity/schema errors always raise. Default checks do not certify full
    source bytes; ``verify_sources=True`` adds those expensive integrity hashes.
    """
    if type(verify_sources) is not bool or not isinstance(identity_sha256, str):
        raise DataContractError('Explicit saved-run identity and boolean source-check flag required')
    root = Path(root).resolve()
    parent = root / 'outputs/data_limited_backtest'
    run = _within(parent, Path(directory).resolve())
    if run.name != '2010-2019-' + identity_sha256[7:23]:
        raise DataContractError('Saved fallback run directory differs from its explicit identity')
    if run.exists() and any(run.iterdir()) and not (run / 'build_identity.json').exists():
        raise DataContractError('Existing fallback run lost its build identity')
    identity = _json(run / 'build_identity.json')
    if (canonical_sha256(identity) != identity_sha256
            or identity.get('phase') != 'data_limited_feature_target_rebuild_v1'
            or identity.get('configuration', {}).get('protocol_id') != 'dmv2-data-limited-restricted-universe-2018-2019-v1'
            or set(identity.get('years', {})) != {str(year) for year in YEARS}):
        raise DataContractError('Saved fallback feature identity changed')
    receipts = {} if expected_receipts is None else expected_receipts
    if not isinstance(receipts, dict) or not set(receipts) <= {'features', 'models', 'backtest'}:
        raise DataContractError('Only explicit feature/model/backtest audit receipts are accepted')
    if verify_sources:
        _deep_sources(root, identity)
        preserved = _json(run / 'source_preservation_manifest.json')
        _verify_artifacts(root, preserved)
    root_audit = run / 'audit.json'
    rows = []
    if root_audit.exists():
        _receipt(root_audit, receipts.get('features'), 'feature')
        audit = _json(root_audit)
        if (audit.get('status') not in {'PARTIAL', 'COMPLETE'}
                or audit.get('build_identity_sha256') != identity_sha256
                or audit.get('research_status') != 'DATA_LIMITED_RESTRICTED_UNIVERSE'
                or not set(audit.get('years', {})) <= {str(year) for year in YEARS}):
            raise DataContractError('Saved fallback annual checkpoint scope changed')
        for year, item in audit['years'].items():
            annual_dir = run / 'years' / year
            annual = _pinned_json(run, item, expected_path=annual_dir / 'audit.json')
            if (annual.get('status') != 'COMPLETE' or annual.get('year') != int(year)
                    or annual.get('build_identity_sha256') != identity_sha256):
                raise DataContractError('Saved fallback annual audit identity changed')
            for name in ('feature_summary', 'monthly_summary', 'feature_provenance',
                         'monthly_provenance', 'loading_provenance', 'target_state'):
                _pinned_json(annual_dir, annual['artifacts'][name],
                    expected_path=annual_dir / (name + '.json'))
            if verify_sources:
                _verify_artifacts(annual_dir, annual['artifacts'])
            rows.append({'year': int(year), **annual['summary']})
        if (audit['status'] == 'COMPLETE') != (set(audit['years']) == {str(year) for year in YEARS}):
            raise DataContractError('Saved fallback completion status disagrees with annual receipts')
        status = audit['status']
    else:
        if receipts.get('features') is not None:
            raise FileNotFoundError(root_audit)
        status = 'BUILDING_NO_SAVED_AUDIT'
    model_summary, model_comparison = _model_report(run, identity_sha256, receipts.get('models'), verify_sources)
    backtest_summary, backtest_comparison = _backtest_report(run, identity_sha256, receipts.get('backtest'), verify_sources)
    return {'directory': run, 'identity_sha256': identity_sha256,
        'summary': {'feature_status': status, 'completed_outcome_years': sorted(row['year'] for row in rows),
            'model_status': model_summary['status'], 'backtest_status': backtest_summary['status'],
            'research_status': 'DATA_LIMITED_RESTRICTED_UNIVERSE',
            'financial_2020_plus_analysis': False,
            'verification_mode': 'small_saved_receipts_and_reports_only' if not verify_sources else 'full_integrity_hashes_plus_saved_reports',
            'full_source_bytes_verified': verify_sources,
            'quotes_compiled': False, 'models_deserialized': False, 'numerical_work_started': False},
        'year_summaries': pd.DataFrame(rows).sort_values('year') if rows else pd.DataFrame(),
        'model_summary': model_summary, 'model_comparison': model_comparison,
        'backtest_summary': backtest_summary, 'backtest_comparison': backtest_comparison}
