"""Promote explicitly pinned context panels without rebuilding or relabelling them.

The artifact identities retain their original build roles. This reader verifies
both archived recipes without executing them, then permits only the declared
80%-primary/90%-sensitivity change in the active project contract. The ordinary
current-run and historical-run readers retain their stricter equality checks.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pandas as pd

from .catalog import canonical_sha256
from .full_panel import (WORKERS, VERSION, _hash, _verify_artifacts, _verify_year,
                         _within, build_identity)
from .universe import DataContractError

PRIMARY_POLICY = 'available_cohort_80pct'
SENSITIVITY_POLICY = 'available_cohort_90pct'
COMPARISON_ORIENTATION = '90pct_reference_to_80pct_candidate'
FULL_PROTOCOL = 'dmv2-full-feature-panel-2010-2023-context80-primary-v3'
FULL_AUTHORITY = ('User promoted the verified 80% original-cohort context panel to primary and retained 90% as sensitivity on 2026-10-15; paper15, PIT, daily inputs and exact lookbacks unchanged.')
NORMALIZATION_PROTOCOL = 'dmv2-normalization-missingness-v4-2010-context80-primary'
NORMALIZATION_AUTHORITY = ('User promoted the verified 80% context panel to primary and retained 90% as sensitivity on 2026-10-15; provisional identity and 2010 initialization remain unchanged.')
DAILY_POLICY = ('Apply 80% and exhaustive admitted-ID requests separately on EVERY historical daily union/sleeve; retain 90% sensitivity and never project month-end membership backward.')


def _load(path):
    return json.loads(Path(path).read_text())


def _valid_pin(value, *, identity=False):
    if identity:
        if not isinstance(value, str) or not value.startswith('sha256:'):
            return False
        value = value[7:]
    return (isinstance(value, str) and len(value) == 64
            and all(c in '0123456789abcdef' for c in value))


def _role_configuration(catalog):
    value = _load(catalog.root / 'configs/context_panel_roles.json')
    if (value.get('schema_version') != 1
            or value.get('primary_policy') != PRIMARY_POLICY
            or value.get('sensitivity_policy') != SENSITIVITY_POLICY
            or value.get('historical_comparison_orientation') != COMPARISON_ORIENTATION):
        raise DataContractError('Context roles must explicitly select 80% primary and 90% sensitivity')
    for role, policy in (('primary', PRIMARY_POLICY), ('sensitivity', SENSITIVITY_POLICY)):
        pin = value.get(role, {})
        if (pin.get('context_policy') != policy or not _valid_pin(pin.get('audit_sha256'))
                or not _valid_pin(pin.get('build_identity_sha256'), identity=True)):
            raise DataContractError('Context role requires explicit policy, audit and identity pins: ' + role)
    if not _valid_pin(value.get('comparison_manifest_sha256')):
        raise DataContractError('Context roles require an explicit comparison manifest pin')
    return value


def _promoted_contract(reference):
    """The complete, narrow configuration delta authorized by this promotion."""
    expected = deepcopy(reference)
    settings = expected['configuration']
    settings['protocol_id'] = FULL_PROTOCOL
    settings['authorization'] = FULL_AUTHORITY
    settings['context_policy'] = PRIMARY_POLICY
    normalization = expected['normalization']
    normalization.update(protocol_id=NORMALIZATION_PROTOCOL, version='4.0.0',
                         authority=NORMALIZATION_AUTHORITY)
    normalization['context_policy'].update(active_worker_policy=PRIMARY_POLICY,
        minimum_coverage_fraction=.8, sensitivity_worker_policy=SENSITIVITY_POLICY,
        sensitivity_coverage_fraction=.9, daily_benchmarks=DAILY_POLICY)
    expected['workflow']['features']['context_policy'] = PRIMARY_POLICY
    return expected


def _verify_active_contract(catalog, historical):
    """Reject every semantic change beyond the explicit primary-role promotion."""
    active = build_identity(catalog)
    expected = _promoted_contract(historical)
    for key in ('version', 'configuration', 'normalization', 'workflow', 'source_pins',
                'runtime', 'feature_dictionary_sha256'):
        if active.get(key) != expected.get(key):
            raise DataContractError('Context promotion differs from its approved active ' + key + ' contract')
    return active


def _verify_snapshot(directory, hashes, required):
    if set(hashes) != set(required):
        raise DataContractError('Historical recipe does not pin every required worker')
    for name in required:
        path = _within(directory, directory / name)
        if not path.is_file() or _hash(path) != hashes[name]:
            raise DataContractError('Historical recipe worker snapshot changed: ' + name)


def _read_run(catalog, pin):
    directory = _within(catalog.root / 'outputs', catalog.root / pin['directory'])
    audit_path = directory / 'audit.json'
    if not audit_path.is_file() or _hash(audit_path) != pin['audit_sha256']:
        raise DataContractError('Context role audit differs from its explicit pin')
    report = _load(audit_path)
    identity = _load(directory / 'build_identity.json')
    identity_hash = pin['build_identity_sha256']
    if canonical_sha256(identity) != identity_hash:
        raise DataContractError('Context role identity differs from its explicit pin')
    if (report.get('status') != 'COMPLETE' or report.get('build_identity_sha256') != identity_hash
            or report.get('configuration') != identity.get('configuration')
            or identity.get('configuration', {}).get('context_policy') != pin['context_policy']
            or any(report.get(key) is not False for key in
                   ('models_fitted', 'imputation_fitted', 'model_ready', 'daily_backtest_ready'))
            or set(report.get('years', {})) != {str(y) for y in range(2010, 2024)}):
        raise DataContractError('Context role is not a complete diagnostic-only 2010-2023 run')
    required_reports = {'build_identity.json', 'feature_dictionary.json'}
    if not required_reports <= set(report.get('reports', {})):
        raise DataContractError('Context role is missing its recipe/dictionary report pins')
    _verify_artifacts(directory, report['reports'])
    _verify_artifacts(directory, report['years'])
    for year in range(2010, 2024):
        if _verify_year(directory / 'years' / str(year), identity_hash).get('year') != year:
            raise DataContractError('Context role annual partition is mislabelled')
    contract = identity.get('active_reference_contract', identity)
    dictionary_hash = canonical_sha256(_load(directory / 'feature_dictionary.json'))
    summary = report['summary']
    if (dictionary_hash != contract.get('feature_dictionary_sha256')
            or summary.get('feature_dictionary_sha256') != dictionary_hash
            or summary.get('original_stock_months') != 152307
            or summary.get('paper_qualified_stock_months') != 140777
            or summary.get('completed_signals') != 168):
        raise DataContractError('Context role differs from the verified original cohort/dictionary scope')
    return ({'directory': directory, 'audit': report,
             'verification_mode': 'pinned_immutable_context_role',
             'reference_pins': deepcopy(pin),
             'tables': {name: pd.read_csv(directory / item['path'])
                        for name, item in report['reports'].items() if item['path'].endswith('.csv')}}, identity)


def _verify_recipe(catalog, roles, primary_identity, sensitivity_identity):
    recipe = _within(catalog.root / 'outputs/reference_code',
                     catalog.root / roles['archived_recipe_directory'])
    reference = primary_identity.get('active_reference_contract', {})
    if (primary_identity.get('schema_version') != 1
            or primary_identity.get('context_policy') != PRIMARY_POLICY
            or primary_identity.get('source_pins') != reference.get('source_pins')):
        raise DataContractError('Primary role does not contain the pinned historical 80% recipe')
    config_files = {'configuration': 'full_panel_build.json',
                    'normalization': 'normalization_protocol.json', 'workflow': 'research_workflow.json'}
    for key, name in config_files.items():
        saved = _load(_within(recipe, recipe / 'configs' / name))
        if reference.get(key) != saved or sensitivity_identity.get(key) != saved:
            raise DataContractError('Archived context recipes disagree with configuration snapshot: ' + name)
    if primary_identity['configuration'] != _load(recipe / 'configs/context_sensitivity_80.json'):
        raise DataContractError('Archived 80% calculation config differs from its original snapshot')
    original_reference = primary_identity['configuration'].get('reference', {})
    if original_reference != {key: value for key, value in roles['sensitivity'].items()
                              if key != 'context_policy'}:
        raise DataContractError('Primary artifact does not refer to the pinned 90% sensitivity recipe')
    for key in ('version', 'source_pins', 'runtime'):
        if sensitivity_identity.get(key) != reference.get(key):
            raise DataContractError('Historical context panels differ in their ' + key + ' contract')
    if reference.get('version') != VERSION:
        raise DataContractError('Unsupported historical context recipe version')
    _verify_snapshot(_within(recipe, recipe / 'src/dmv2'), reference.get('workers', {}), WORKERS)
    _verify_snapshot(_within(recipe, recipe / 'src/dmv2'), reference.get('sensitivity_workers', {}),
                     ('context_sensitivity.py', 'context_comparison.py'))
    snapshot90 = _within(catalog.root / 'outputs/reference_code',
        catalog.root / roles['sensitivity']['worker_snapshot_directory'])
    _verify_snapshot(snapshot90, sensitivity_identity.get('workers', {}), WORKERS)
    _verify_active_contract(catalog, reference)
    return reference


def _read_comparison(roles, primary, sensitivity):
    directory = primary['directory'] / 'comparison'
    path = directory / 'comparison_manifest.json'
    if not path.is_file() or _hash(path) != roles['comparison_manifest_sha256']:
        raise DataContractError('Context comparison differs from its explicit manifest pin')
    manifest = _load(path)
    for name, run in (('reference_audit', sensitivity), ('sensitivity_audit', primary)):
        if manifest[name]['sha256'] != _hash(run['directory'] / 'audit.json'):
            raise DataContractError('Context comparison uses a different completed panel')
    for name, item in manifest['reports'].items():
        artifact = _within(directory, directory / name)
        if (not artifact.is_file() or artifact.stat().st_size != item['bytes']
                or _hash(artifact) != item['sha256']):
            raise DataContractError('Context comparison report changed: ' + name)
    if 'comparison_summary.json' not in manifest['reports']:
        raise DataContractError('Context comparison summary lacks its hash pin')
    summary = _load(directory / 'comparison_summary.json')
    if summary.get('status') != 'CONTEXT_80_VS_90_COMPARISON_VERIFIED':
        raise DataContractError('Context comparison is incomplete')
    if not set(summary['reports'].values()) <= set(manifest['reports']):
        raise DataContractError('Context comparison table lacks a hash pin')
    return {'reference': sensitivity, 'sensitivity': primary,
            'orientation': COMPARISON_ORIENTATION, 'directory': directory, 'summary': summary,
            'tables': {name: pd.read_csv(_within(directory, directory / filename))
                       for name, filename in summary['reports'].items()}}


def read_context_panel_roles(catalog):
    """Read immutable 80%-primary/90%-sensitivity artifacts plus the pinned comparison.

    This performs read-only hash checks and reads reports. It does not calculate
    features, fit a model, execute archived code or consume protected outcomes.
    """
    roles = _role_configuration(catalog)
    primary, primary_identity = _read_run(catalog, roles['primary'])
    sensitivity, sensitivity_identity = _read_run(catalog, roles['sensitivity'])
    _verify_recipe(catalog, roles, primary_identity, sensitivity_identity)
    comparison = _read_comparison(roles, primary, sensitivity)
    return {'primary': primary, 'sensitivity': sensitivity, 'comparison': comparison, 'roles': roles}
