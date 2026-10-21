"""Fabricated metadata trees only; no actual catalogs/proofs/models/R reads."""
from pathlib import Path
from unittest.mock import patch
import contextlib
import importlib.util
import json
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('scope_under_test', Path(__file__).with_name('build_install_scope.py'))
g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)


class Fixture:
    def __init__(self, temporary):
        self.root = Path(temporary); self.source = self.root/'source'; self.repo = self.root/'repo'
        self.control = self.root/'control'; self.dest = self.repo/'outputs/corrected_refit/2020-2023-20261021-v1'
        self.source.mkdir(); self.repo.mkdir(); self.control.mkdir()
        self.original = self.repo/'unchanged_original.bin'; self.original.write_bytes(b'opaque original input')
        self.original_pin = g.pin(self.original)
        # Proof/report files are deliberately invalid JSON. Accidental parsing fails.
        for relative in set(g.ARTIFACTS.values()) | set(g.GATES.values()) | set(g.BACKTEST.values()) | {'build_install_scope.py'}:
            p = self.source/relative; p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b'opaque: never parse a financial proof, report, table or model\xff')
        self.models = g.pin(self.source/g.GATES['models'])
        self.catalog = {'schema': 'fabricated-catalog/v1', 'status': 'VERIFIED_PROVISIONAL_FROZEN',
            'annuals': [{'year': y, 'artifacts': {'opaque': self.original_pin}} for y in range(2010, 2024)],
            'RET_states': [{'signal_date': f'{y}-{m:02d}-28', 'artifacts': {'opaque': self.original_pin}}
                           for y in range(2020, 2024) for m in range(1, 13)]}
        self.refresh_metadata()
        p = self.source/'run/control/finished.json'; p.parent.mkdir(parents=True, exist_ok=True)
        self.write(p, {'status': 'COMPLETE'})

    def write(self, path, value):
        Path(path).write_text(json.dumps(value)); return g.pin(path)

    def refresh_metadata(self):
        catalog = self.write(self.source/'input_catalog.json', self.catalog)
        protocol = self.write(self.source/'model_protocol.json', {
            'schema': 'dmv2-corrected-refit-protocol/v1', 'catalog_pin': catalog,
            'catalog_schema': self.catalog['schema'], 'catalog_status': self.catalog['status'],
            'outcome_years': [2020, 2021, 2022, 2023], 'parameter_search': False,
            'helper_pins': {'opaque': self.original_pin}})
        self.write(self.source/'run/models/read_plan.json', {
            'schema': 'dmv2-corrected-refit-source-plan/v1', 'catalog_pin': catalog,
            'protocol_pin': protocol, 'repeated_original': self.original_pin})
        backtest = self.write(self.source/'backtest_source_plan.json', {
            'schema': 'dmv2-corrected-forecasts-backtest-source-plan/v1',
            'catalog_pin': catalog, 'model_protocol_pin': protocol,
            'scenario_count': 120, 'selection_sets': 15,
            'execution_evidence_pins': [self.original_pin]})
        self.write(self.source/'backtest_protocol.json', {
            'schema': 'dmv2-corrected-forecasts-backtest-protocol/v1',
            'source_plan_pin': backtest, 'model_protocol_pin': protocol,
            'scenarios': 120, 'selection_sets': 15, 'sessions': 1006, 'outcome_months': 48,
            'new_estimator_fits': 0, 'new_inference_calls': 0,
            'changed_portfolio_rules': False, 'changed_stop_or_cost_rules': False})
        self.write(self.source/'run/models/pipeline/progress.json', {
            'status': 'COMPLETE_VERIFIED', 'completed_estimator_fits': 20,
            'completed_preprocessors': 8, 'completed_phases': 50,
            'completed_annual_verifications': 4, 'final_verification_pin': self.models})

    @contextlib.contextmanager
    def patches(self):
        with patch.object(g, 'SOURCE', self.source), patch.object(g, 'REPO', self.repo), \
             patch.object(g, 'CONTROL', self.control), patch.object(g, 'DESTINATION', self.dest):
            yield

    def build(self, name='scope.json'):
        return g.build(self.control/name, all_writers_stopped=True)


class Tests(unittest.TestCase):
    def test_exact_scope_deduplicates_nonempty_originals_and_pins_fixture_reports_opaquely(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            f = Fixture(d)
            with f.patches(): result = f.build()
            body = json.loads(Path(result['path']).read_text())
            self.assertEqual(body['original_input_pins'], [f.original_pin])
            self.assertEqual(set(body['gates']), set(g.GATES))
            self.assertEqual(set(body['backtest_expected_pins']), set(g.BACKTEST))
            for name in ('fixture_verification', 'backtest_report_receipt', 'backtest_report_json',
                         'backtest_report_markdown', 'final_readme', 'public_document'):
                self.assertEqual(g.pin(body['artifacts'][name]['path']), body['artifacts'][name])
            self.assertFalse(body['proof_and_report_bodies_parsed'])
            self.assertEqual(body['reservations_created'], 0)

    def test_only_closed_metadata_paths_and_safe_guard_are_parsed(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            f = Fixture(d); original = g.read_metadata; seen = []
            allowed = {f.source/relative for relative in g.METADATA.values()} | {f.source/'run/control/finished.json'}
            def reader(path):
                self.assertIn(Path(path), allowed); seen.append(Path(path)); return original(path)
            with f.patches(), patch.object(g, 'read_metadata', side_effect=reader): f.build()
            self.assertEqual(set(seen), allowed)

    def test_duplicate_pin_conflict_rejected_even_outside_repository(self):
        item = {'path': '/private/tmp/fabricated-opaque-reference', 'size_bytes': 1, 'sha256': '0'*64}
        with self.assertRaisesRegex(ValueError, 'Conflicting pins'):
            g.embedded_pins([item, {**item, 'sha256': '1'*64}])

    def test_empty_originals_changed_source_or_wrong_annual_grid_fail_closed(self):
        for case in ('empty', 'changed', 'years'):
            with self.subTest(case=case), tempfile.TemporaryDirectory(dir='/private/tmp') as d:
                f = Fixture(d)
                if case == 'empty':
                    substitute = f.source/'opaque_input.bin'; substitute.write_bytes(b'not original repo')
                    f.original_pin = g.pin(substitute)
                    for annual in f.catalog['annuals']: annual['artifacts']['opaque'] = f.original_pin
                    for state in f.catalog['RET_states']: state['artifacts']['opaque'] = f.original_pin
                    f.refresh_metadata()
                elif case == 'changed': f.original.write_bytes(b'mutation')
                else:
                    f.catalog['annuals'][-1]['year'] = 2022; f.refresh_metadata()
                with f.patches(), self.assertRaises(ValueError): f.build()
                self.assertFalse((f.control/'scope.json').exists())

    def test_terminal_ack_and_running_guards_block_before_scope_output(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            f = Fixture(d)
            with f.patches(), patch.object(g, 'read_metadata', side_effect=AssertionError('must not read')):
                with self.assertRaisesRegex(ValueError, 'explicitly acknowledge'):
                    g.build(f.control/'scope.json')
            f.write(f.source/'run/control/finished.json', {'status': 'RUNNING'})
            with f.patches(), self.assertRaisesRegex(ValueError, 'still RUNNING'): f.build()

    def test_missing_final_locator_and_existing_output_never_succeed(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            f = Fixture(d); (f.source/'run/reports/backtest_v1/report.md').unlink()
            with f.patches(), self.assertRaisesRegex(ValueError, 'regular file'): f.build()
            (f.source/'run/reports/backtest_v1/report.md').write_bytes(b'opaque final doc')
            (f.control/'scope.json').write_bytes(b'preserve')
            with f.patches(), self.assertRaisesRegex(ValueError, 'Fresh external'): f.build()
            self.assertEqual((f.control/'scope.json').read_bytes(), b'preserve')

    def test_metadata_binding_drift_and_symlink_output_escape_rejected(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            f = Fixture(d)
            protocol_path = f.source/'backtest_protocol.json'
            data = json.loads(protocol_path.read_text()); data['sessions'] = 1005; f.write(protocol_path, data)
            with f.patches(), self.assertRaisesRegex(ValueError, 'protocol detached'): f.build()
            f.refresh_metadata(); outside = f.root/'outside'; outside.mkdir()
            (f.control/'escape').symlink_to(outside, target_is_directory=True)
            with f.patches(), self.assertRaisesRegex(ValueError, 'Fresh external'):
                g.build(f.control/'escape/scope.json', all_writers_stopped=True)


if __name__ == '__main__':
    unittest.main()
