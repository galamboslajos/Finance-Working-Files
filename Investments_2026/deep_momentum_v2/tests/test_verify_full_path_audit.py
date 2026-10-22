"""Independent generated-ledger verifier checks on fabricated monthly inputs."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from dmv2.full_path_accounting import prepare_action_registry, prepare_price_index
from test_full_path_audit import fixtures, annual_values, cli


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('full_independent_formula_check', ROOT / 'scripts/verify_full_path_audit.py')
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class FullIndependentVerifierTests(unittest.TestCase):
    def generated(self, root):
        plan = fixtures()
        annual = plan['annuals'][0]
        window = annual['windows'][0]
        cohort, prices, legacy, overlay = annual_values(plan, 2010)
        cohort = cohort.loc[cohort.signal_date == window['signal_date']].copy()
        legacy = legacy.loc[legacy.signal_date == window['signal_date']].copy()
        index = prepare_price_index(prices, plan['sessions'])
        registry = prepare_action_registry([], sessions=plan['sessions'])
        report = cli.run_month(plan, annual, window, cohort, index, legacy, overlay, registry, root, lambda: None)
        directory = root / 'years/2010/months/2010-01/attempt-0001'
        return directory, report, window

    def repin(self, directory, report, name):
        report['artifacts'][name] = dict(path=name, size_bytes=(directory/name).stat().st_size, sha256=checker.sha(directory/name))

    def test_actual_synthetic_worker_and_independent_formula_grid_agree(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory, report, window = self.generated(Path(temporary))
            result = checker.verify_month(directory, report, window)
            self.assertEqual(result['original_rows'], 12)
            self.assertEqual(result['eligible_rows'], 11)
            self.assertEqual(result['target_rows'], 96)
            self.assertEqual(result['available_values'], 80)
            self.assertEqual(sum(result['grid'].values()), 80)

    def test_repinned_wrong_utility_is_rejected_by_independent_arithmetic(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory, report, window = self.generated(Path(temporary))
            path = directory/'targets.parquet'
            rows = pq.read_table(path).to_pylist()
            row = next(r for r in rows if r['path_available'])
            row['utility'] = .1
            pq.write_table(pa.Table.from_pylist(rows), path)
            self.repin(directory, report, 'targets.parquet')
            with self.assertRaisesRegex(ValueError, 'Independent utility differs'):
                checker.verify_month(directory, report, window)

    def test_repinned_missing_daily_mark_cannot_claim_complete_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory, report, window = self.generated(Path(temporary))
            path = directory/'marks.parquet'
            table = pq.read_table(path)
            pq.write_table(table.slice(1), path)
            self.repin(directory, report, 'marks.parquet')
            with self.assertRaises(ValueError):
                checker.verify_month(directory, report, window)

    def test_wrong_class_cells_cannot_hide_behind_unchanged_total(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory, report, window = self.generated(Path(temporary))
            first = next(r for r in report['class_grid'] if r['members'])
            other = next(r for r in report['class_grid'] if r is not first)
            first['members'] -= 1
            other['members'] += 1
            with self.assertRaisesRegex(ValueError, 'Individual class coverage cells differ'):
                checker.verify_month(directory, report, window)

    def test_same_population_count_with_wrong_key_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory, report, window = self.generated(Path(temporary))
            report['original_key_sha256'] = 'sha256:' + '0' * 64
            with self.assertRaisesRegex(ValueError, 'population key identity changed'):
                checker.verify_month(directory, report, window)

    def test_repinned_blanket_class_withholding_cannot_shrink_ranking_pool(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory, report, window = self.generated(Path(temporary))
            path = directory/'targets.parquet'
            rows = pq.read_table(path).to_pylist()
            for row in rows:
                row['target_available'] = False
                row['target_class'] = None
            pq.write_table(pa.Table.from_pylist(rows), path)
            self.repin(directory, report, 'targets.parquet')
            for row in report['class_grid']:
                row['members'] = 0
            with self.assertRaisesRegex(ValueError, 'Class availability differs'):
                checker.verify_month(directory, report, window)

    def test_sub_tolerance_value_change_cannot_split_independent_exact_tie(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory, report, window = self.generated(Path(temporary))
            path = directory/'targets.parquet'
            rows = pq.read_table(path).to_pylist()
            row = next(r for r in rows if r['target_available'] and r['target_family'] == 'executable_return')
            row['target_value'] += 5e-13
            row['target_class'] = 1
            pq.write_table(pa.Table.from_pylist(rows), path)
            self.repin(directory, report, 'targets.parquet')
            with self.assertRaisesRegex(ValueError, 'exact-midrank class differs'):
                checker.verify_month(directory, report, window)


if __name__ == '__main__':
    unittest.main()
