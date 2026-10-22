"""Independent verifier fixtures contain invented capital paths only."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('independent_path_check', ROOT / 'scripts/verify_path_target_audit.py')
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


class IndependentFormulaTests(unittest.TestCase):
    def fixture(self, directory, *, wrong_value=False, missing_mark=False):
        directory = Path(directory)
        keys = dict(source_security_id='invented', signal_date='2019-12-31')
        grid = [dict(date='2020-01-02', mark='open'), dict(date='2020-02-03', mark='open')]
        marks, targets, episodes = [], [], []
        for side in ('long', 'short'):
            for variant in ('unstopped', 'asset_stop_10pct'):
                episode = dict(**keys, side=side, path_variant=variant)
                episodes.append(dict(**episode, status='complete'))
                marks.extend(dict(**episode, **tick, wealth=value)
                             for tick, value in zip(grid, (1., .8)))
                for family in ('executable_return', 'return_minus_drawdown'):
                    value = -.2 if family == 'executable_return' else -.4
                    targets.append(dict(**episode, outcome_month='2020-01', label_available_at='2020-02-03',
                        target_family=family, path_available=True, target_available=False,
                        cash_return=-.2, maximum_drawdown=.2, utility=-.4,
                        target_value=value + (.01 if wrong_value else 0), target_class=None))
        if missing_mark:
            marks.pop()
        artifacts = {}
        for name, records in [('marks', marks), ('targets', targets), ('episodes', episodes)]:
            path = directory / (name + '.parquet')
            pq.write_table(pa.Table.from_pylist(records), path)
            artifacts[path.name] = dict(path=path.name, size_bytes=path.stat().st_size, sha256=verifier.sha(path))
        (directory / 'reviewed_plan.json').write_text(json.dumps(dict(plan_sha256='invented',
            windows=[dict(signal_date='2019-12-31', required_marks=grid)])))
        (directory / 'audit.json').write_text(json.dumps(dict(status='COMPLETE_BOUNDED_AUDIT',
            models_fitted=False, financial_2024_plus_read=False, plan_sha256='invented',
            original_rows=1, complete_episode_count=4, artifacts=artifacts)))

    def test_independent_hand_calculated_return_dd_utility_and_denominators(self):
        with tempfile.TemporaryDirectory() as directory:
            self.fixture(directory)
            result = verifier.verify(directory)
            self.assertEqual(result['target_rows_checked'], 8)
            self.assertEqual(result['available_target_values_checked'], 8)
            self.assertFalse(result['full_training_history_certified'])
            with self.assertRaisesRegex(ValueError, 'Preserve previous'):
                verifier.verify(directory)

    def test_changed_utility_is_not_accepted_by_repinning_derived_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            self.fixture(directory, wrong_value=True)
            with self.assertRaisesRegex(ValueError, 'Wrong target family value'):
                verifier.verify(directory)

    def test_missing_mark_cannot_claim_a_complete_episode(self):
        with tempfile.TemporaryDirectory() as directory:
            self.fixture(directory, missing_mark=True)
            with self.assertRaisesRegex(ValueError, 'Completed episode grid'):
                verifier.verify(directory)


if __name__ == '__main__':
    unittest.main()
