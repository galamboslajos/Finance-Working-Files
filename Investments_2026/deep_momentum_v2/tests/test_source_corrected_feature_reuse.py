"""Standard-library synthetic pin/identity/proof guards; no financial libraries."""
from pathlib import Path
from copy import deepcopy
import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1] / 'scripts'
spec = importlib.util.spec_from_file_location('reuse', HERE / 'verify_source_corrected_feature_reuse.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
REAL_HELPER = HERE / 'verify_walkforward_features_v2.py'


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(obj, sort_keys=True))


def fixture(root):
    cfg = {'context_policy': 'available_cohort_80pct', 'execution': {'financial_2024_plus_analysis': False}}
    save(root / 'configs/walkforward_2020_2023.json', cfg)
    old = root / 'outputs/walkforward/2020-2023-aacd7579fcf21e6a'
    old_identity = {'configuration': cfg, 'synthetic': 'old-feature-root'}
    old_id = m.canonical(old_identity); save(old / 'build_identity.json', old_identity)
    parent_by_group = {}
    for group in ('ep', 'baseline', 'opch'):
        parent = root / 'sources' / group
        identity = {'synthetic_source_group': group}; identity_sha = m.canonical(identity)
        save(parent / 'build_identity.json', identity)
        save(parent / 'audit.json', {'status': 'COMPLETE', 'build_identity_sha256': identity_sha})
        parent_by_group[group] = (parent, identity_sha)
    roles = ('features', 'monthly_returns', 'targets', 'daily_history_raw', 'daily_prices_raw',
             'original_cohort', 'feature_provenance', 'exchange_calendar')
    entries, audits = [], {}
    for year in range(2010, 2024):
        group = 'ep' if year <= 2013 else 'baseline' if year < 2020 else 'opch'
        parent, source_id = parent_by_group[group]
        if year in (2020, 2021): parent, source_id = old, old_id
        directory = parent / 'years' / str(year)
        artifacts = {}
        for role in roles:
            file = directory / (role + '.parquet'); file.parent.mkdir(parents=True, exist_ok=True)
            file.write_bytes(f'SYNTHETIC_UNDECODED_BYTES:{year}:{role}'.encode())
            p = m.raw_pin(root, file); p['path'] = file.name; artifacts[role] = p
        audit = {'status': 'COMPLETE', 'year': year, 'build_identity_sha256': source_id, 'artifacts': artifacts}
        save(directory / 'audit.json', audit); audits[year] = audit
        entries.append({'year': year, 'directory': str(directory.relative_to(root)),
            'audit_sha256': m.digest(directory / 'audit.json'), 'build_identity_sha256': source_id,
            'corrected': year in m.REPAIR_YEARS, 'original_rows': 1})
    quality = root / 'outputs/source_quality/reviewed'; reviewdir = quality / 'input_independent_review'
    reviews = {}
    for item in entries:
        y = item['year']
        if y not in m.REPAIR_YEARS: continue
        proof = {'status': 'PASS_BOUNDED_INDEPENDENT_INPUT_RECOMPUTATION', 'year': y,
            'build_identity_sha256': item['build_identity_sha256'], 'annual_audit_sha256': item['audit_sha256'],
            'helper_sha256': m.OLD_FEATURE_HELPER_SHA, 'paper15': {'rows': 1, 'values_checked': 15},
            'classes': {'original_rows': 1}, 'no2024_numeric_rows': 'PASS', 'monthly_returns': {'ok': 1},
            'input_artifact_sha256': {k: p['sha256'] for k, p in audits[y]['artifacts'].items()}}
        path = reviewdir / f'{y}_review.json'; save(path, proof)
        reviews[str(y)] = {'path': f'/synthetic/original-location/{y}_review.json', 'sha256': m.digest(path)}
    aggregate = {'status': 'PASS_BOUNDED_SIX_ANNUAL_INPUT_RECOMPUTATION', 'years_checked': list(m.REPAIR_YEARS),
        'model_fits': 0, 'backtest_runs': 0, 'annual_review_files': reviews,
        'checks': {name: 'PASS' for name in ('all15_paper_formulas_and_common_population_normalization',
            'exact_monthly_endpoints_and_missing_return_policy', 'ten_class_exact_tie_rank_and_missing_label_policy',
            'no2024_financial_rows_returned')}}
    review_path = reviewdir / 'review_receipt.json'; save(review_path, aggregate)
    manifest_path = quality / 'corrected_annual_sources.json'
    save(manifest_path, {'status': 'VERIFIED_INPUT_REFERENCES_NOT_MODEL_RESULTS', 'years': entries,
        'context_policy': 'available_cohort_80pct', 'financial_2024_plus_analysis': False})
    handoff_artifacts = [m.raw_pin(root, manifest_path), m.raw_pin(root, review_path)]
    handoff_artifacts.extend(m.raw_pin(root, reviewdir / f'{y}_review.json') for y in m.REPAIR_YEARS)
    for i in range(412 - len(handoff_artifacts)):
        file = quality / 'evidence' / f'pin{i:03d}.bin'; file.parent.mkdir(exist_ok=True)
        file.write_bytes(f'SYNTHETIC_{i}'.encode()); handoff_artifacts.append(m.raw_pin(root, file))
    handoff_path = quality / 'handoff.json'
    save(handoff_path, {'status': 'REPAIRED_INPUTS_READY_MODELS_STALE', 'existing_runs_modified': False,
        'financial_2024_plus_analysis': False, 'artifacts': handoff_artifacts})
    old_years = {}
    for y in range(2020, 2024):
        item = next(i for i in entries if i['year'] == y)
        path = root / item['directory'] / 'audit.json'
        pin = m.raw_pin(root, path)
        pin['path'] = f'years/{y}/audit.json'
        old_years[str(y)] = pin
    save(old / 'audit.json', {'status': 'COMPLETE', 'build_identity_sha256': old_id, 'years': old_years})
    old_proof = {'status': 'PASS', 'models_fitted': 0, 'financial_2024_plus_read': False,
        'root_audit_sha256': m.digest(old / 'audit.json'), 'build_identity_sha256': old_id,
        'verifier_contract_revision': 'v2-explicit-shared-market-quarantine', 'months': 48,
        'verifier_source_sha256': m.OLD_FEATURE_HELPER_SHA, 'verifier_sha256': m.OLD_FEATURE_HELPER_SHA,
        'years': {str(y): {'original_rows': 1, 'paper15': {'values_checked': 15},
                         'no2024_price_rows_returned': True, 'feature_and_outcome_daily_views_separate': True}
                  for y in range(2020, 2024)}}
    old_proof_path = old / 'verification/features.json'; save(old_proof_path, old_proof)
    helper = root / 'scripts/verify_walkforward_features_v2.py'; helper.parent.mkdir()
    shutil.copyfile(REAL_HELPER, helper)
    worker = root / 'src/dmv2/paper_features.py'; worker.parent.mkdir(parents=True); worker.write_text('# synthetic unused worker\n')
    refs, routes = {}, {}
    for item in entries:
        y = item['year']; source = root / item['directory']
        ref = {'directory': item['directory'], 'audit_pin': m.raw_pin(root, source / 'audit.json'),
            'build_identity_sha256': item['build_identity_sha256'], 'corrected': item['corrected'], 'original_rows': 1}
        refs[str(y)] = ref
        if y < 2020:
            parent = source.parent.parent
            routes[str(y)] = {'directory': item['directory'], 'directory_scope': 'root',
                'audit_pin': ref['audit_pin'], 'build_identity_sha256': item['build_identity_sha256'],
                'reused_without_numeric_rebuild': True,
                'parent_identity_pin': m.raw_pin(root, parent / 'build_identity.json'),
                'parent_audit_pin': m.raw_pin(root, parent / 'audit.json')}
        else: routes[str(y)] = {'directory': f'years/{y}', 'directory_scope': 'run', 'parent_reference': 'THIS_BUILD', 'audit_pin': None}
    identity = {'phase': 'source_corrected_walkforward_2020_2023_v1', 'configuration': cfg,
        'configuration_sha256': m.canonical(cfg), 'reviewed_source_annuals': refs, 'annual_sources': routes,
        'preserved_reference_pins': {str(p.relative_to(root)): m.raw_pin(root, p) for p in (old / 'audit.json', old / 'build_identity.json')},
        'source_correction_manifest_pin': m.raw_pin(root, manifest_path),
        'source_correction_handoff_pin': m.raw_pin(root, handoff_path),
        'independent_corrected_input_review_pin': m.raw_pin(root, review_path),
        'workers': {'paper_features.py': m.digest(worker)}}
    identity_sha = m.canonical(identity); run = root / 'outputs/walkforward_source_corrected/synthetic'
    save(run / 'build_identity.json', identity)
    year_pins, copied_bytes = {}, 0
    for item in entries:
        y = item['year']
        if y < 2020: continue
        source = root / item['directory']; target = run / 'years' / str(y)
        shutil.copytree(source, target)
        shutil.copyfile(source / 'audit.json', target / 'reference_annual_audit.json')
        a = deepcopy(audits[y]); a['build_identity_sha256'] = identity_sha
        rp = m.raw_pin(root, target / 'reference_annual_audit.json'); rp['path'] = 'reference_annual_audit.json'
        a['artifacts']['reference_annual_audit'] = rp
        a['source_corrected_replay_lineage'] = {**refs[str(y)], 'numerical_artifact_bytes_unchanged': True,
            'new_feature_calculations': 0, 'original_audit_preserved_as': 'reference_annual_audit.json'}
        save(target / 'audit.json', a)
        p = m.raw_pin(root, target / 'audit.json'); p['path'] = f'years/{y}/audit.json'; year_pins[str(y)] = p
        copied_bytes += sum(p['size_bytes'] for p in audits[y]['artifacts'].values())
    save(run / 'audit.json', {'status': 'COMPLETE', 'phase': 'reviewed_annual_input_reuse_only',
        'build_identity_sha256': identity_sha, 'financial_2024_plus_analysis': False,
        'original_sources_and_runs_preserved': True, 'model_fits': 0, 'backtest_runs': 0,
        'years': year_pins, 'copied_input_bytes': copied_bytes})
    return run, identity_sha, old_proof_path


class ReuseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.run, self.identity, self.old_proof = fixture(self.root)
        # Synthetic fixture has its own pre-frozen receipt hash, not a real receipt.
        self.guard = patch.object(m, 'OLD_FEATURE_RECEIPT_SHA', m.digest(self.old_proof))
        self.guard.start(); self.addCleanup(self.guard.stop)

    def prepare(self): return m.prepare(self.root, self.run, self.identity)

    def test_completed_reuse_authenticates_all_scopes_without_financial_imports(self):
        with patch.dict(sys.modules, {'pyarrow': None, 'pandas': None, 'numpy': None, 'sklearn': None, 'xgboost': None}):
            result = m.verify(self.root, self.run, self.identity, self.root / 'private_check')
        self.assertEqual(result['status'], 'PASS'); self.assertEqual(result['annual_reference_count'], 14)
        self.assertEqual(result['detached_annual_count'], 4); self.assertEqual(result['original_handoff_artifacts_preserved'], 412)
        self.assertEqual(result['verification_scope'], 'new_root_input_reference_reuse_and_prior_formula_evidence')
        self.assertEqual(result['new_feature_formula_calculations'], 0); self.assertFalse(result['financial_columns_decoded'])
        self.assertEqual(result['models_fitted'], 0); self.assertEqual(result['model_files_loaded'], 0)
        self.assertEqual(result['root_audit_sha256'], m.digest(self.run / 'audit.json'))

    def test_old_receipt_raw_pin_checked_before_its_json_is_trusted(self):
        self.old_proof.write_bytes(self.old_proof.read_bytes() + b'\n')
        original = m.read
        def guard(path):
            if Path(path) == self.old_proof: raise AssertionError('Old JSON trusted before original raw hash')
            return original(path)
        with patch.object(m, 'read', guard), self.assertRaisesRegex(AssertionError, 'original feature-verification raw'):
            self.prepare()

    def test_root_identity_change_refused(self):
        with self.assertRaisesRegex(AssertionError, 'root identity changed'):
            m.prepare(self.root, self.run, 'sha256:' + '0' * 64)

    def test_root_reuse_audit_cannot_claim_new_calculation_or_fit(self):
        path = self.run / 'audit.json'; a = m.read(path); a['model_fits'] = 1; save(path, a)
        with self.assertRaisesRegex(AssertionError, 'reuse-only input audit'): self.prepare()

    def test_actual_held_input_bytes_are_hashed_not_just_declared(self):
        plan = self.prepare()
        pin = next(p for p in plan['input_pins'] if p['path'].endswith('/features.parquet'))
        path = self.root / pin['path']; raw = path.read_bytes(); path.write_bytes(b'X' + raw[1:])
        with self.assertRaisesRegex(AssertionError, 'Actual input bytes changed'):
            m.verify(self.root, self.run, self.identity, self.root / 'mutated_check')

    def test_preserved_reference_audit_alias_is_refused(self):
        path = self.run / 'years/2022/audit.json'; a = m.read(path)
        a['artifacts']['reference_annual_audit']['path'] = './reference_annual_audit.json'; save(path, a)
        # Keep the root annual raw pin current to test the nested path rule itself.
        audit = m.read(self.run / 'audit.json'); audit['years']['2022'].update(sha256=m.digest(path), size_bytes=path.stat().st_size)
        save(self.run / 'audit.json', audit)
        with self.assertRaisesRegex(AssertionError, 'source-audit copy redirected'): self.prepare()

    def test_detached_same_byte_feature_path_redirect_is_refused(self):
        path = self.run / 'years/2022/audit.json'; a = m.read(path)
        original = path.parent / a['artifacts']['features']['path']; alias = path.parent / 'alias.parquet'
        shutil.copyfile(original, alias); a['artifacts']['features']['path'] = alias.name; save(path, a)
        audit = m.read(self.run / 'audit.json'); audit['years']['2022'].update(sha256=m.digest(path), size_bytes=path.stat().st_size)
        save(self.run / 'audit.json', audit)
        with self.assertRaisesRegex(AssertionError, 'full artifact pin differs'): self.prepare()

    def test_handoff_artifact_payload_change_is_detected(self):
        plan = self.prepare(); pin = next(p for p in plan['original_handoff_artifacts'] if p['path'].endswith('.bin'))
        path = self.root / pin['path']; raw = path.read_bytes(); path.write_bytes(b'X' + raw[1:])
        with self.assertRaisesRegex(AssertionError, 'Actual input bytes changed'):
            m.verify(self.root, self.run, self.identity, self.root / 'handoff_changed')

    def test_explicit_frozen_byte_budget_is_enforced(self):
        with patch.object(m, 'MAX_UNIQUE_FILES', 1), self.assertRaisesRegex(AssertionError, 'budget exceeded'):
            self.prepare()

    def test_existing_receipt_directory_is_never_overwritten(self):
        output = self.root / 'existing'; output.mkdir()
        with self.assertRaisesRegex(AssertionError, 'Preserve every existing'):
            m.verify(self.root, self.run, self.identity, output)

    def test_source_correction_proof_does_not_expand_formula_claims(self):
        plan = self.prepare(); scopes = plan['upstream_mathematical_proof_scope_by_year']
        self.assertEqual(set(scopes), set(map(str, range(2010, 2024))))
        for y in m.REPAIR_YEARS: self.assertEqual(scopes[str(y)]['kind'], 'UPSTREAM_CORRECTED_BOUNDED_RECOMPUTATION')
        for y in (2020, 2021): self.assertEqual(scopes[str(y)]['kind'], 'UPSTREAM_ORIGINAL_UNCHANGED_RECOMPUTATION')
        for y in range(2014, 2020): self.assertFalse(scopes[str(y)]['independent_formula_certification_extended_by_this_receipt'])


class FreshProcessReuseTests(unittest.TestCase):
    def test_all_standard_library_guards_in_a_fresh_process(self):
        # The production verifier caps process-lifetime peak RSS at 256 MiB.
        # Other suites import model libraries first, so keep this no-financial-
        # import contract in a separate interpreter without changing that cap.
        import subprocess
        result = subprocess.run(
            [sys.executable, '-B', str(Path(__file__).resolve()), 'ReuseTests'],
            capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('Ran 11 tests', result.stderr)


def load_tests(loader, standard_tests, pattern):
    return loader.loadTestsFromTestCase(FreshProcessReuseTests)


if __name__ == '__main__': unittest.main()
