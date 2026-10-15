"""Full-run persistence, source policy and small reader contracts; no real prices."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from dmv2.full_panel import (_artifact, _configuration, _json, _specs, _verify_artifacts,
    _verify_year, _within, _daily_storage_views, build_identity, read_feature_partition,
    read_full_feature_run, read_pinned_feature_run, WORKERS)
from dmv2.catalog import canonical_sha256
from dmv2.universe import DataContractError

ROOT = Path(__file__).resolve().parents[1]


class FullPanelPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / 'configs').mkdir()
        for name in ('full_panel_build.json', 'normalization_protocol.json', 'research_workflow.json'):
            shutil.copyfile(ROOT / 'configs' / name, self.root / 'configs' / name)
        self.catalog = SimpleNamespace(root=self.root, verify=lambda hashes: (),
            provenance=lambda: {'sources': [{'source_key': 'structural_fixture'}]})

    def test_active_dictionary_preserves_fifteen_policy_and_frozen_allowlist(self):
        specs = _specs()
        self.assertEqual(len(specs), 728)
        self.assertEqual(sum(s.get('model_eligible', True) for s in specs), 715)
        paper = [s for s in specs if s['name'].startswith(('zMOM', 'meanMOM', 'stdMOM'))]
        self.assertEqual(len(paper), 15)
        self.assertEqual({s['normalization_policy'] for s in paper}, {'signal_qualified_common_momentum_v1'})

    def test_source_and_canonical_policy_changes_alter_run_identity(self):
        before = canonical_sha256(build_identity(self.catalog))
        self.catalog.provenance = lambda: {'sources': [{'source_key': 'different_pin'}]}
        self.assertNotEqual(canonical_sha256(build_identity(self.catalog)), before)
        cfg = json.loads((self.root / 'configs/research_workflow.json').read_text())
        cfg['research_start'] = '2000-01-01'
        _json(self.root / 'configs/research_workflow.json', cfg)
        with self.assertRaises(DataContractError):
            _configuration(self.catalog)

    def test_models_and_unapproved_scope_rejected_before_source_reads(self):
        path = self.root / 'configs/full_panel_build.json'
        initial = json.loads(path.read_text())
        for key, value in (('end_year', 2024), ('history_sessions', 252), ('universe', 'sp1500')):
            cfg = {**initial, key: value}
            _json(path, cfg)
            with self.subTest(key=key), self.assertRaises(DataContractError):
                _configuration(self.catalog)
        cfg = initial
        cfg['execution']['model_fitting'] = True
        _json(path, cfg)
        with self.assertRaises(DataContractError):
            _configuration(self.catalog)

    def test_active_context_policy_must_agree_across_three_configurations(self):
        settings, normalization, workflow = _configuration(self.catalog)
        self.assertEqual(settings["context_policy"], "available_cohort_80pct")
        self.assertEqual(normalization["context_policy"]["active_worker_policy"], settings["context_policy"])
        self.assertEqual(workflow["features"]["context_policy"], settings["context_policy"])
        before = build_identity(self.catalog)
        self.assertEqual(before["configuration"]["context_policy"], "available_cohort_80pct")
        for name, section, key, value in (
            ("full_panel_build.json", None, "context_policy", "complete_cohort"),
            ("normalization_protocol.json", "context_policy", "active_worker_policy", "complete_cohort"),
            ("normalization_protocol.json", "context_policy", "minimum_coverage_fraction", .9),
            ("normalization_protocol.json", "context_policy", "sensitivity_worker_policy", "available_cohort_80pct"),
            ("normalization_protocol.json", "context_policy", "sensitivity_coverage_fraction", .8),
            ("normalization_protocol.json", "context_policy", "exhaustive_request_required", False),
            ("research_workflow.json", "features", "context_policy", "complete_cohort"),
        ):
            path = self.root / "configs" / name
            original = path.read_text()
            changed = json.loads(original)
            (changed if section is None else changed[section])[key] = value
            _json(path, changed)
            with self.subTest(name=name, key=key), self.assertRaises(DataContractError):
                _configuration(self.catalog)
            path.write_text(original)

    def test_checkpoint_hash_and_size_and_source_pin_are_verified(self):
        year = self.root / '2010'
        year.mkdir()
        path = year / 'features.parquet'
        pq.write_table(pa.table({'source_security_id': ['fixture'], 'signal_date': ['2009-12-31']}), path)
        artifacts = {'features': _artifact(year, path)}
        _json(year / 'audit.json', {'build_identity_sha256': 'pin', 'artifacts': artifacts})
        self.assertEqual(_verify_year(year, 'pin')['build_identity_sha256'], 'pin')
        with self.assertRaises(DataContractError):
            _verify_year(year, 'changed_source_pin')
        path.write_bytes(path.read_bytes()[:-1] + b'X')
        with self.assertRaises(DataContractError):
            _verify_artifacts(year, artifacts)

    def _pinned_reference(self):
        directory = self.root / 'outputs/full_feature_panel/historical-reference'
        directory.mkdir(parents=True)
        snapshot = self.root / 'reference-code'
        snapshot.mkdir()
        for name in WORKERS:
            shutil.copyfile(ROOT / 'src/dmv2' / name, snapshot / name)
        # Historical code differs from current code, but its exact bytes have
        # independent identity/audit pins and must never be executed by reading.
        path = snapshot / 'full_panel.py'
        path.write_text(path.read_text() + '\n# Historical structural fixture.\n')
        identity = build_identity(self.catalog)
        identity['workers'] = {name: hashlib.sha256((snapshot / name).read_bytes()).hexdigest()
                               for name in WORKERS}
        identity_hash = canonical_sha256(identity)
        _json(directory / 'build_identity.json', identity)
        _json(directory / 'feature_dictionary.json', list(_specs()))
        _json(directory / 'source_plan.json', {'start_year': 2010, 'end_year': 2023,
            'universe': 'combined', 'history_sessions': 503, 'monthly_original_rows': 152307,
            'signal_dates': [f'{year}-{month:02d}-28' for year in range(2010, 2024)
                             for month in range(1, 13)], 'source_pins': identity['source_pins']})
        summary = {'original_stock_months': 152307, 'paper_qualified_stock_months': 140777,
            'completed_signals': 168, 'feature_dictionary_sha256': identity['feature_dictionary_sha256']}
        _json(directory / 'coverage_summary.json', summary)
        reports = {name: _artifact(directory, directory / name) for name in
                   ('build_identity.json', 'feature_dictionary.json', 'source_plan.json')}
        reports['coverage_summary'] = _artifact(directory, directory / 'coverage_summary.json')
        years = {}
        for year in range(2010, 2024):
            path = directory / 'years' / str(year)
            path.mkdir(parents=True)
            _json(path / 'structural-artifact.json', {'year': year})
            artifacts = {'structural': _artifact(path, path / 'structural-artifact.json')}
            _json(path / 'audit.json', {'build_identity_sha256': identity_hash,
                                       'year': year, 'artifacts': artifacts})
            years[str(year)] = _artifact(directory, path / 'audit.json')
        _json(directory / 'audit.json', {'version': 1, 'status': 'COMPLETE',
            'build_identity_sha256': identity_hash, 'configuration': identity['configuration'],
            'reports': reports, 'years': years, 'summary': summary,
            'models_fitted': False, 'imputation_fitted': False,
            'model_ready': False, 'daily_backtest_ready': False})
        return {'directory': directory,
            'expected_audit_sha256': hashlib.sha256((directory / 'audit.json').read_bytes()).hexdigest(),
            'expected_identity_sha256': identity_hash, 'worker_snapshot_directory': snapshot}

    def test_explicit_historical_reader_preserves_strict_current_reader(self):
        pins = self._pinned_reference()
        run = read_pinned_feature_run(self.catalog, **pins)
        self.assertEqual(run['verification_mode'], 'pinned_historical_reference')
        self.assertEqual(run['reference_pins']['build_identity_sha256'], pins['expected_identity_sha256'])
        with self.assertRaises(DataContractError):
            read_full_feature_run(self.catalog, directory=pins['directory'])

    def test_historical_reader_requires_independent_audit_and_identity_pins(self):
        pins = self._pinned_reference()
        for key, value in (('expected_audit_sha256', '0' * 64),
                           ('expected_identity_sha256', 'sha256:' + '0' * 64),
                           ('expected_audit_sha256', None)):
            with self.subTest(key=key), self.assertRaises(DataContractError):
                read_pinned_feature_run(self.catalog, **{**pins, key: value})

    def test_historical_reader_rejects_changed_source_and_policy_contracts(self):
        pins = self._pinned_reference()
        original_provenance = self.catalog.provenance
        self.catalog.provenance = lambda: {'sources': [{'source_key': 'changed'}]}
        with self.assertRaises(DataContractError):
            read_pinned_feature_run(self.catalog, **pins)
        self.catalog.provenance = original_provenance
        path = self.root / 'configs/normalization_protocol.json'
        contract = json.loads(path.read_text())
        contract['context_policy']['minimum_coverage_fraction'] = .9
        _json(path, contract)
        with self.assertRaises(DataContractError):
            read_pinned_feature_run(self.catalog, **pins)

    def test_historical_reader_rejects_missing_or_modified_worker_snapshot(self):
        pins = self._pinned_reference()
        path = pins['worker_snapshot_directory'] / 'context_features.py'
        original = path.read_bytes()
        path.write_bytes(original + b'\n# altered\n')
        with self.assertRaises(DataContractError):
            read_pinned_feature_run(self.catalog, **pins)
        path.unlink()
        with self.assertRaises(DataContractError):
            read_pinned_feature_run(self.catalog, **pins)

    def test_historical_reader_verifies_report_and_year_artifacts(self):
        pins = self._pinned_reference()
        report = pins['directory'] / 'coverage_summary.json'
        original = report.read_bytes()
        report.write_bytes(original + b'\n')
        with self.assertRaises(DataContractError):
            read_pinned_feature_run(self.catalog, **pins)
        report.write_bytes(original)
        artifact = pins['directory'] / 'years/2010/structural-artifact.json'
        artifact.write_bytes(artifact.read_bytes() + b'\n')
        with self.assertRaises(DataContractError):
            read_pinned_feature_run(self.catalog, **pins)

    def test_artifact_and_pointer_symlinks_cannot_leave_project(self):
        with tempfile.TemporaryDirectory() as external:
            (self.root / 'escape').symlink_to(external, target_is_directory=True)
            with self.assertRaises(DataContractError):
                _within(self.root, self.root / 'escape' / 'features.parquet')

    def test_one_signal_reader_rejects_targets_and_protected_outcomes(self):
        run = {'directory': self.root}
        with self.assertRaises(DataContractError):
            read_feature_partition(run, signal_date='2023-12-29')
        with self.assertRaises(DataContractError):
            read_feature_partition(run, signal_date='2009-12-31', columns=['provider_adjusted_return'])
        path = self.root / 'years/2010/features.parquet'
        path.parent.mkdir(parents=True)
        pq.write_table(pa.table({'source_security_id': ['A', 'A'],
            'signal_date': ['2009-12-31', '2010-01-29'], 'zMOM1': [None, None]}), path)
        frame = read_feature_partition(run, signal_date='2009-12-31', columns=['zMOM1'])
        self.assertEqual(frame.signal_date.tolist(), ['2009-12-31'])
        with self.assertRaises(DataContractError):
            read_feature_partition(run, signal_date='2009-12-30', columns=['zMOM1'])

    def test_new_december_entrant_anchor_and_daily_warmup_are_not_discarded(self):
        raw = pa.table({'intrinio_security_id': ['NEW'] * 3,
            'trading_date': ['2019-12-31', '2020-01-02', '2020-12-31']})
        feature = raw.slice(0, 2)
        sample = SimpleNamespace(raw_daily_prices=SimpleNamespace(table=raw),
            feature_daily_prices=SimpleNamespace(table=feature), signal_dates=('2020-11-30',))
        views = _daily_storage_views(sample, 2020)
        self.assertEqual(views['daily_prices_raw']['trading_date'].to_pylist(),
                         ['2020-01-02', '2020-12-31'])
        self.assertIs(views['daily_history_raw'], raw)
        self.assertIs(views['feature_history_daily'], feature)
        self.assertIn('2019-12-31', views['daily_history_raw']['trading_date'].to_pylist())
        sample.feature_daily_prices.table = raw
        with self.assertRaises(DataContractError):
            _daily_storage_views(sample, 2020)


if __name__ == '__main__':
    unittest.main()
