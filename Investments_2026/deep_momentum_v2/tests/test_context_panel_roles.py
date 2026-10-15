"""Immutable role promotion tests use synthetic metadata and no financial data."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

from dmv2.catalog import canonical_sha256
from dmv2.context_panel_roles import (_promoted_contract, _read_comparison, _read_run,
    _role_configuration, _verify_active_contract, _verify_recipe, read_context_panel_roles)
from dmv2.full_panel import (_artifact, _hash, _json, _specs, build_identity, WORKERS)
from dmv2.universe import DataContractError

ROOT = Path(__file__).resolve().parents[1]


class ContextPanelRoleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        shutil.copytree(ROOT / 'configs', self.root / 'configs', ignore=shutil.ignore_patterns('*.local.json'))
        self.catalog = SimpleNamespace(root=self.root, verify=lambda hashes: (),
            provenance=lambda: {'sources': [{'source_key': 'synthetic_source'}]})

    def reference_contract(self):
        value = build_identity(self.catalog)
        value['configuration'].update(protocol_id='historical90', authorization='historical authority',
                                      context_policy='available_cohort_90pct')
        n = value['normalization']
        n.update(protocol_id='historical90', version='3.0.0', authority='historical authority')
        n['context_policy'].update(active_worker_policy='available_cohort_90pct',
            minimum_coverage_fraction=.9, daily_benchmarks='historical90 daily benchmark statement')
        for key in ('sensitivity_worker_policy', 'sensitivity_coverage_fraction'):
            n['context_policy'].pop(key)
        value['workflow']['features']['context_policy'] = 'available_cohort_90pct'
        return value

    def run_fixture(self, identity, directory):
        directory.mkdir(parents=True)
        identity_hash = canonical_sha256(identity)
        _json(directory / 'build_identity.json', identity)
        _json(directory / 'feature_dictionary.json', list(_specs()))
        contract = identity.get('active_reference_contract', identity)
        summary = {'original_stock_months': 152307, 'paper_qualified_stock_months': 140777,
            'completed_signals': 168, 'feature_dictionary_sha256': contract['feature_dictionary_sha256']}
        reports = {name: _artifact(directory, directory / name)
                   for name in ('build_identity.json', 'feature_dictionary.json')}
        years = {}
        for year in range(2010, 2024):
            annual = directory / 'years' / str(year)
            annual.mkdir(parents=True)
            _json(annual / 'sample.json', {'synthetic_year': year})
            _json(annual / 'audit.json', {'year': year, 'build_identity_sha256': identity_hash,
                'artifacts': {'synthetic': _artifact(annual, annual / 'sample.json')}})
            years[str(year)] = _artifact(directory, annual / 'audit.json')
        _json(directory / 'audit.json', {'status': 'COMPLETE', 'build_identity_sha256': identity_hash,
            'configuration': identity['configuration'], 'reports': reports, 'years': years,
            'summary': summary, 'models_fitted': False, 'imputation_fitted': False,
            'model_ready': False, 'daily_backtest_ready': False})
        return {'context_policy': identity['configuration']['context_policy'],
            'directory': str(directory.relative_to(self.root)),
            'audit_sha256': _hash(directory / 'audit.json'), 'build_identity_sha256': identity_hash}

    def roles_fixture(self):
        reference = self.reference_contract()
        recipe = self.root / 'outputs/reference_code/context80'
        workers = recipe / 'src/dmv2'
        workers.mkdir(parents=True)
        for name in (*WORKERS, 'context_sensitivity.py', 'context_comparison.py'):
            shutil.copyfile(ROOT / 'src/dmv2' / name, workers / name)
        reference['workers'] = {n: _hash(workers / n) for n in WORKERS}
        reference['sensitivity_workers'] = {n: _hash(workers / n)
            for n in ('context_sensitivity.py', 'context_comparison.py')}
        snapshot90 = self.root / 'outputs/reference_code/context90'
        snapshot90.mkdir(parents=True)
        for name in WORKERS:
            shutil.copyfile(workers / name, snapshot90 / name)
        old90 = deepcopy(reference)
        old90.pop('sensitivity_workers')
        p90 = self.run_fixture(old90, self.root / 'outputs/full_feature_panel/synthetic90')
        p90['worker_snapshot_directory'] = str(snapshot90.relative_to(self.root))
        sensitivity_cfg = json.loads((ROOT / 'configs/context_sensitivity_80.json').read_text())
        sensitivity_cfg['reference'] = {k:v for k,v in p90.items() if k != 'context_policy'}
        recipe_configs = recipe / 'configs'
        recipe_configs.mkdir()
        for key, filename in (('configuration','full_panel_build.json'),
            ('normalization','normalization_protocol.json'),('workflow','research_workflow.json')):
            _json(recipe_configs / filename, reference[key])
        _json(recipe_configs / 'context_sensitivity_80.json', sensitivity_cfg)
        primary_identity = {'schema_version': 1, 'configuration': sensitivity_cfg,
            'active_reference_contract': reference, 'source_pins': reference['source_pins'],
            'context_policy': 'available_cohort_80pct'}
        p80 = self.run_fixture(primary_identity, self.root / 'outputs/context_sensitivity/synthetic80')
        comparison = self.root / p80['directory'] / 'comparison'
        comparison.mkdir()
        pd.DataFrame({'synthetic': [1]}).to_csv(comparison / 'coverage.csv', index=False)
        _json(comparison / 'comparison_summary.json', {'status': 'CONTEXT_80_VS_90_COMPARISON_VERIFIED',
            'reports': {'synthetic_coverage': 'coverage.csv'}})
        files = ('coverage.csv', 'comparison_summary.json')
        _json(comparison / 'comparison_manifest.json', {
            'reference_audit': {'sha256': p90['audit_sha256']},
            'sensitivity_audit': {'sha256': p80['audit_sha256']},
            'reports': {n: {'bytes': (comparison/n).stat().st_size,
                            'sha256': _hash(comparison/n)} for n in files}})
        roles = {'schema_version': 1, 'primary_policy': 'available_cohort_80pct',
            'sensitivity_policy': 'available_cohort_90pct',
            'historical_comparison_orientation': '90pct_reference_to_80pct_candidate',
            'archived_recipe_directory': str(recipe.relative_to(self.root)),
            'primary': p80, 'sensitivity': p90,
            'comparison_manifest_sha256': _hash(comparison / 'comparison_manifest.json')}
        _json(self.root / 'configs/context_panel_roles.json', roles)
        return roles, primary_identity, old90

    def test_active80_contract_only_permits_the_exact_role_promotion(self):
        reference = self.reference_contract()
        _verify_active_contract(self.catalog, reference)
        for key, subkey in (('paper_normalization','standard_deviation_ddof'),
                           ('missingness','scaling'),('chronology','protected_financial_start'),
                           ('universe','identity_policy')):
            altered = deepcopy(reference)
            altered['normalization'][key][subkey] = 'unapproved_change'
            with self.subTest(key=key), self.assertRaisesRegex(DataContractError,'normalization'):
                _verify_active_contract(self.catalog, altered)
        changed = deepcopy(reference)
        changed['source_pins'] = {'sources': [{'source_key': 'different_source'}]}
        with self.assertRaisesRegex(DataContractError,'source_pins'):
            _verify_active_contract(self.catalog, changed)

    def test_role_reader_preserves_immutable_identities_and_comparison_direction(self):
        roles, p80identity, p90identity = self.roles_fixture()
        before = {role: (self.root / roles[role]['directory'] / 'audit.json').read_bytes()
                  for role in ('primary','sensitivity')}
        result = read_context_panel_roles(self.catalog)
        self.assertEqual(result['primary']['audit']['configuration']['context_policy'], 'available_cohort_80pct')
        self.assertEqual(result['sensitivity']['audit']['configuration']['context_policy'], 'available_cohort_90pct')
        self.assertIs(result['comparison']['reference'], result['sensitivity'])
        self.assertIs(result['comparison']['sensitivity'], result['primary'])
        self.assertEqual(result['comparison']['orientation'], '90pct_reference_to_80pct_candidate')
        for role in ('primary','sensitivity'):
            self.assertEqual(before[role], (self.root / roles[role]['directory'] / 'audit.json').read_bytes())

    def test_root_audit_year_and_recipe_tampering_are_rejected(self):
        roles, primary_identity, sensitivity_identity = self.roles_fixture()
        paths = [self.root / roles['primary']['directory'] / 'audit.json',
                 self.root / roles['sensitivity']['directory'] / 'years/2010/sample.json',
                 self.root / roles['archived_recipe_directory'] / 'src/dmv2/context_sensitivity.py',
                 self.root / roles['archived_recipe_directory'] / 'configs/normalization_protocol.json',
                 self.root / roles['sensitivity']['worker_snapshot_directory'] / 'paper_features.py',
                 self.root / roles['primary']['directory'] / 'comparison/coverage.csv']
        for path in paths:
            original = path.read_bytes()
            path.write_bytes(original + b'\n#tampered\n')
            with self.subTest(path=path), self.assertRaises((DataContractError,json.JSONDecodeError)):
                read_context_panel_roles(self.catalog)
            path.write_bytes(original)

    def test_changed_identity_pin_source_and_active_workflow_are_rejected(self):
        roles, _, _ = self.roles_fixture()
        original = deepcopy(roles)
        roles['primary']['build_identity_sha256'] = 'sha256:' + '0'*64
        _json(self.root / 'configs/context_panel_roles.json', roles)
        with self.assertRaisesRegex(DataContractError,'identity'):
            read_context_panel_roles(self.catalog)
        _json(self.root / 'configs/context_panel_roles.json', original)
        self.catalog.provenance = lambda: {'sources': [{'source_key': 'different_source'}]}
        with self.assertRaisesRegex(DataContractError,'source_pins'):
            read_context_panel_roles(self.catalog)

    def test_archived80_reference_must_match_the_selected90_panel(self):
        roles, primary_identity, sensitivity_identity = self.roles_fixture()
        changed = deepcopy(roles)
        changed['sensitivity']['audit_sha256'] = '0'*64
        with self.assertRaisesRegex(DataContractError, 'does not refer'):
            _verify_recipe(self.catalog, changed, primary_identity, sensitivity_identity)

    def test_role_policy_swaps_and_unpinned_comparison_are_rejected(self):
        roles, _, _ = self.roles_fixture()
        for changed in ({**roles, 'primary_policy':'available_cohort_90pct'},
                        {**roles, 'comparison_manifest_sha256':None}):
            _json(self.root / 'configs/context_panel_roles.json', changed)
            with self.assertRaises(DataContractError):
                _role_configuration(self.catalog)


if __name__ == '__main__':
    unittest.main()
