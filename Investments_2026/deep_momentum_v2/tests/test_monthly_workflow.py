"""Stage controls and independent-table evidence; synthetic fixtures only."""
from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import pyarrow as pa
import pyarrow.parquet as pq

from dmv2.loading import LoadedTable
from dmv2.workflow import MonthlyRun, run_monthly_sample, save_monthly_sample, validate_workflow


def configuration():
    return json.loads((Path(__file__).resolve().parents[1] / 'configs/research_workflow.json').read_text())


@dataclass
class Window:
    outcome_month: str = '2000-01'
    signal_date: str = '1999-12-31'
    outcome_start: str = '2000-01-03'
    outcome_end: str = '2000-01-31'


def fixture_run():
    # Preserve a nonfinite source observation exactly in Parquet. The monthly
    # derived return is unavailable; strict JSON contains audit metadata only.
    daily = pa.table({'intrinio_security_id': ['fixture-a'], 'trading_date': ['1999-12-31'],
                      'adj_close': [float('nan')]})
    monthly = pa.table({'source_security_id': ['member-a'], 'outcome_month': ['2000-01'],
                        'provider_adjusted_return': pa.array([None], type=pa.float64()),
                        'return_status': ['invalid_endpoint']})
    sample = SimpleNamespace(daily_prices=LoadedTable(daily, {}, {}),
                             panel=SimpleNamespace(table=monthly, windows=(Window(),)))
    sample.source_calendar = pa.table({'trading_date': ['1999-12-31']})
    sample.calendar = pa.table({'trading_date': ['1999-12-31']})
    report = {'workflow_sha256': 'sha256:fixture-settings', 'source_pins': {'release': 'fixture-r1'},
              'code_provenance': {'worker_sha256': {'monthly.py': 'fixture-code'}, 'python_version': 'fixture'},
              'summary': {'rows': 1}}
    return MonthlyRun(sample, report)


class MonthlyControlsTests(unittest.TestCase):
    def test_disabled_stage_touches_no_sources(self):
        settings = configuration()
        settings['monthly_returns']['enabled'] = False
        with patch('dmv2.workflow.load_monthly_sample') as loader:
            self.assertIsNone(run_monthly_sample(Mock(), settings))
            loader.assert_not_called()

    def test_invalid_controls_fail_before_any_reads(self):
        cases = [('start_month', '1999-12'), ('end_month', '2024-01'),
                 ('end_month', '1999-12'), ('end_month', '2000-04'),
                 ('start_month', '2000-13'), ('start_month', '200001'),
                 ('start_month', None), ('enabled', 1), ('persist', 'true'),
                 ('sample_size', 0), ('sample_size', 11), ('sample_size', True),
                 ('return_basis', 'certified_total_return'), ('read_budget', []),
                 ('calendar_basis', 'price_availability_intersection'),
                 ('read_budget', {'max_scanned_rows': 0}), ('read_budget', {'unknown': 1})]
        with patch('dmv2.workflow.load_monthly_sample') as loader:
            for key, value in cases:
                settings = configuration()
                settings['monthly_returns'][key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    run_monthly_sample(Mock(), settings)
            loader.assert_not_called()

    def test_missing_monthly_settings_fail_cleanly(self):
        for value in (None, []):
            settings = configuration()
            settings['monthly_returns'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_workflow(settings)
        settings = configuration()
        del settings['monthly_returns']['persist']
        with self.assertRaises(ValueError):
            validate_workflow(settings)

    def test_configured_stage_keeps_daily_table_and_cohort_evidence(self):
        result = fixture_run().sample
        result.panel.summary = {'rows': 1}
        result.panel.provenance = {'knowledge_time_verified': False}
        result.requested_ids = ('fixture-a',)
        result.snapshots = ()
        result.loading_stats = {'total': {'returned_rows': 1}}
        result.loading_provenance = {'sample_selection': 'first signal only'}
        catalog = Mock()
        catalog.provenance.return_value = {'release': 'fixture-r1'}
        with patch('dmv2.workflow.load_monthly_sample', return_value=result) as loader:
            run = run_monthly_sample(catalog, configuration())
        self.assertIs(run.sample.daily_prices.table, result.daily_prices.table)
        self.assertEqual(run.report['requested_ids'], ['fixture-a'])
        self.assertFalse(run.report['return_provenance']['knowledge_time_verified'])
        self.assertEqual(loader.call_args.kwargs['start_month'], '2000-01')
        self.assertEqual(loader.call_args.kwargs['end_month'], '2000-02')
        self.assertEqual(loader.call_args.kwargs['sample_size'], 2)
        self.assertEqual(loader.call_args.kwargs['mode'], 'historical_reconstruction')


class IndependentEvidenceTests(unittest.TestCase):
    def test_daily_and_monthly_artifacts_are_separate_idempotent_and_hashed(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = fixture_run()
            path = save_monthly_sample(temporary, run)
            original = {name: (path/name).read_bytes() for name in ('daily.parquet', 'monthly.parquet', 'audit.json')}
            self.assertEqual(path, save_monthly_sample(temporary, run))
            self.assertEqual(original, {name: (path/name).read_bytes() for name in original})
            audit = json.loads((path/'audit.json').read_text())
            self.assertEqual(audit['artifacts']['daily.parquet']['rows'], 1)
            self.assertEqual(audit['artifacts']['monthly.parquet']['rows'], 1)
            self.assertNotEqual(original['daily.parquet'], original['monthly.parquet'])
            self.assertEqual(pq.read_table(path/'monthly.parquet')['return_status'].to_pylist(), ['invalid_endpoint'])
            self.assertNotIn('NaN', (path/'audit.json').read_text())

    def test_changed_table_under_same_pins_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = fixture_run()
            path = save_monthly_sample(temporary, run)
            original = (path/'monthly.parquet').read_bytes()
            run.sample.panel.table = run.sample.panel.table.set_column(
                3, 'return_status', pa.array(['missing_end']))
            with self.assertRaisesRegex(ValueError, 'changed'):
                save_monthly_sample(temporary, run)
            self.assertEqual((path/'monthly.parquet').read_bytes(), original)

    def test_changed_code_provenance_creates_new_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = fixture_run()
            first = save_monthly_sample(temporary, run)
            run.report['code_provenance']['worker_sha256']['monthly.py'] = 'changed-code'
            second = save_monthly_sample(temporary, run)
            self.assertNotEqual(first, second)

    def test_modified_saved_table_fails_without_repairing_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = fixture_run()
            path = save_monthly_sample(temporary, run)
            (path/'daily.parquet').write_bytes(b'changed-evidence')
            with self.assertRaisesRegex(ValueError, 'changed'):
                save_monthly_sample(temporary, run)
            self.assertEqual((path/'daily.parquet').read_bytes(), b'changed-evidence')

    def test_output_symlink_cannot_escape_project(self):
        with tempfile.TemporaryDirectory() as temporary, tempfile.TemporaryDirectory() as other:
            root = Path(temporary)
            (root/'outputs').symlink_to(other, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, 'leaves'):
                save_monthly_sample(root, fixture_run())
            self.assertEqual(list(Path(other).iterdir()), [])


if __name__ == '__main__':
    unittest.main()
