"""Synthetic passive reports: no panel/quote/model loading or numerical work."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from dmv2.catalog import canonical_sha256
from dmv2.full_panel import _artifact, _hash, _json
import dmv2.fallback_reporting as reporting
from dmv2.universe import DataContractError


def fixture(root):
    (root / 'configs').mkdir()
    config = root / 'configs/data_limited_fallback.json'
    _json(config, {'synthetic': True})
    reference = root / 'outputs/reference'
    reference.mkdir(parents=True)
    _json(reference / 'audit.json', {'synthetic': True})
    years = {}
    for year in reporting.YEARS:
        annual = reference / 'years' / str(year)
        annual.mkdir(parents=True)
        _json(annual / 'audit.json', {'year': year})
        (annual / 'raw.bin').write_bytes(b'synthetic source bytes; never decode')
        years[str(year)] = {'source_provenance': {
            'reference_primary': {'directory': 'outputs/reference',
                'audit_sha256': _hash(reference / 'audit.json')},
            'annual_audit_sha256': _hash(annual / 'audit.json'),
            'used_artifacts': {'raw': _artifact(annual, annual / 'raw.bin')}}}
    identity = {'phase': 'data_limited_feature_target_rebuild_v1',
        'configuration': {'protocol_id': 'dmv2-data-limited-restricted-universe-2018-2019-v1'},
        'configuration_raw_sha256': 'sha256:' + _hash(config), 'workers': {}, 'years': years}
    pin = canonical_sha256(identity)
    run = root / 'outputs/data_limited_backtest' / ('2010-2019-' + pin[7:23])
    run.mkdir(parents=True)
    _json(run / 'build_identity.json', identity)
    _json(run / 'source_preservation_manifest.json', {'configuration': _artifact(root, config)})
    annual = run / 'years/2010'
    annual.mkdir(parents=True)
    artifacts = {}
    for name in ('feature_summary', 'monthly_summary', 'feature_provenance',
                 'monthly_provenance', 'loading_provenance', 'target_state'):
        path = annual / (name + '.json')
        _json(path, {'synthetic': True})
        artifacts[name] = _artifact(annual, path)
    _json(annual / 'audit.json', {'status': 'COMPLETE', 'year': 2010,
        'build_identity_sha256': pin, 'summary': {'original_rows': 120,
        'prediction_eligible_rows': 119, 'labelled_rows': 118}, 'artifacts': artifacts})
    _json(run / 'audit.json', {'status': 'PARTIAL', 'build_identity_sha256': pin,
        'research_status': 'DATA_LIMITED_RESTRICTED_UNIVERSE',
        'years': {'2010': _artifact(run, annual / 'audit.json')}})
    return run, pin


class FallbackReportingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.run, self.pin = fixture(self.root)

    def read(self, **kwargs):
        return reporting.read_data_limited_run(self.root, self.run,
            identity_sha256=self.pin, **kwargs)

    def test_partial_reader_reads_only_small_json_receipts(self):
        with patch.object(reporting.pq, 'ParquetFile', side_effect=AssertionError('Decoded panel')):
            saved = self.read()
        self.assertEqual(saved['summary']['feature_status'], 'PARTIAL')
        self.assertEqual(saved['summary']['completed_outcome_years'], [2010])
        self.assertFalse(saved['summary']['full_source_bytes_verified'])
        self.assertFalse(saved['summary']['numerical_work_started'])
        self.assertFalse(saved['summary']['models_deserialized'])
        self.assertEqual(saved['model_summary']['status'], 'PENDING')
        self.assertEqual(saved['backtest_summary']['status'], 'PENDING')

    def test_changed_annual_summary_fails_without_any_panel_decode(self):
        (self.run / 'years/2010/feature_summary.json').write_text('{}')
        with patch.object(reporting.pq, 'ParquetFile', side_effect=AssertionError('Decoded panel')), \
             self.assertRaises(DataContractError):
            self.read()

    def test_changed_annual_audit_fails_its_parent_pin(self):
        path = self.run / 'years/2010/audit.json'
        path.write_text(path.read_text() + '\n')
        with self.assertRaises(DataContractError):
            self.read()

    def test_explicit_final_receipt_detects_root_audit_edit(self):
        receipt = 'sha256:' + _hash(self.run / 'audit.json')
        self.read(expected_receipts={'features': receipt})
        (self.run / 'audit.json').write_text((self.run / 'audit.json').read_text() + '\n')
        with self.assertRaises(DataContractError):
            self.read(expected_receipts={'features': receipt})

    def test_changed_original_source_requires_explicit_deep_check(self):
        self.read(verify_sources=True)
        (self.root / 'outputs/reference/years/2010/raw.bin').write_bytes(b'changed')
        self.assertFalse(self.read()['summary']['full_source_bytes_verified'])
        with self.assertRaises(DataContractError):
            self.read(verify_sources=True)

    def test_missing_pinned_summary_is_integrity_error_not_pending(self):
        (self.run / 'years/2010/target_state.json').unlink()
        with self.assertRaises(DataContractError):
            self.read()

    def test_building_without_root_audit_is_pending(self):
        (self.run / 'audit.json').unlink()
        self.assertEqual(self.read()['summary']['feature_status'], 'BUILDING_NO_SAVED_AUDIT')

    def test_lost_existing_build_identity_is_not_treated_as_not_started(self):
        (self.run / 'build_identity.json').unlink()
        with self.assertRaises(DataContractError):
            self.read()

    def test_wrong_run_identity_or_path_fails(self):
        with self.assertRaises(DataContractError):
            reporting.read_data_limited_run(self.root, self.run, identity_sha256='sha256:' + 'a' * 64)

    def test_pinned_missing_final_model_audit_is_not_pending(self):
        with self.assertRaises(DataContractError):
            self.read(expected_receipts={'models': 'sha256:' + 'b' * 64})

    def test_completed_model_reports_do_not_read_matrices_or_estimators(self):
        directory = self.run / 'models'
        directory.mkdir()
        phase = {'parent_identity_sha256': self.pin, 'synthetic': True}
        model_pin = canonical_sha256(phase)
        _json(directory / 'build_identity.json', phase)
        arms = {}
        for name in (*reporting.ARMS, 'uniform', 'train_prior'):
            arm_dir = directory / ('arms' if name in reporting.ARMS else 'refs') / name
            arm_dir.mkdir(parents=True)
            _json(arm_dir / 'audit.json', {'status': 'COMPLETE',
                'build_identity_sha256': self.pin, 'model_identity_sha256': model_pin,
                'arm': {'arm_id': name}, 'summary': {'outcome_months': 24,
                    'mean_monthly_log_loss': 2.3}})
            arms[name] = _artifact(directory, arm_dir / 'audit.json')
        _json(directory / 'audit.json', {'status': 'COMPLETE',
            'build_identity_sha256': self.pin, 'model_identity_sha256': model_pin,
            'summary': {'fitted_arms': 4}, 'arms': arms,
            'artifacts': {'build_identity.json': _artifact(directory, directory / 'build_identity.json')}})
        with patch.object(reporting.pq, 'ParquetFile', side_effect=AssertionError('Decoded panel')), \
             patch('dmv2.fallback_models.read_fallback_models', side_effect=AssertionError('Deep model hash check')):
            saved = self.read()
        self.assertEqual(saved['model_summary']['status'], 'COMPLETE')
        self.assertEqual(len(saved['model_comparison']), 6)
        self.assertEqual(saved['model_comparison'].outcome_months.tolist(), [24] * 6)

    def test_completed_backtest_decodes_only_its_small_64_row_comparison(self):
        directory = self.run / 'backtest'
        directory.mkdir()
        phase = {'parent_identity_sha256': self.pin, 'synthetic': True}
        _json(directory / 'build_identity.json', phase)
        comparison = pd.DataFrame([{'arm': arm, 'basket': basket, 'variant': variant,
            'cost_bps': cost, 'complete_proxy_path': True}
            for arm in reporting.ARMS for basket in ('class_1_10', 'class_2_9')
            for variant in ('unstopped', 'asset_stop_10pct') for cost in (0, 5, 10, 25)])
        comparison.to_parquet(directory / 'comparison.parquet', index=False)
        _json(directory / 'audit.json', {'status': 'COMPLETE',
            'build_identity_sha256': self.pin, 'phase_identity_sha256': canonical_sha256(phase),
            'research_status': 'DATA_LIMITED_RESTRICTED_UNIVERSE',
            'scenarios': 64, 'complete_proxy_paths': 64, 'failed_closed_paths': 0,
            'execution_seconds': 1, 'artifacts': {name: _artifact(directory, directory / name)
                for name in ('build_identity.json', 'comparison.parquet')}})
        original = reporting.pq.ParquetFile
        def report_only(path):
            self.assertEqual(Path(path), directory / 'comparison.parquet')
            return original(path)
        with patch.object(reporting.pq, 'ParquetFile', side_effect=report_only):
            saved = self.read()
        self.assertEqual(saved['backtest_summary']['status'], 'COMPLETE')
        self.assertEqual(len(saved['backtest_comparison']), 64)

    def test_started_backtest_is_pending_without_decoding_its_ledger(self):
        directory = self.run / 'backtest'
        directory.mkdir()
        _json(directory / 'audit.json', {'status': 'STARTED', 'build_identity_sha256': self.pin})
        with patch.object(reporting.pq, 'ParquetFile', side_effect=AssertionError('Decoded panel')):
            self.assertEqual(self.read()['backtest_summary']['status'], 'STARTED')

    def test_notebook_section_is_passive_and_clears_stale_views(self):
        notebook = json.loads((Path(__file__).resolve().parents[1] /
            'notebooks/00_research_master.ipynb').read_text())
        cells = [cell for cell in notebook['cells'] if
            'fallback_saved_report' in ''.join(cell.get('source', [])) and cell['cell_type'] == 'code']
        self.assertEqual(len(cells), 1)
        source = ''.join(cells[0]['source'])
        self.assertIn('read_data_limited_run', source)
        self.assertIn('verify_sources=False', source)
        self.assertIn('fallback_saved_report = None', source)
        self.assertIn('except FileNotFoundError', source)
        for forbidden in ('run_fallback_models(', 'run_fallback_backtest(',
                          'plan_rebuild(', '_quotes(', 'load_fallback_year(',
                          'pickle.load(', 'joblib.load('):
            self.assertNotIn(forbidden, source)
        self.assertEqual(cells[0].get('outputs'), [])
        self.assertIsNone(cells[0].get('execution_count'))


if __name__ == '__main__':
    unittest.main()
