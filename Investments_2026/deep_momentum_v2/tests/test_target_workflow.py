"""Bounded notebook orchestration on structural supplied tables only."""
from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

import pyarrow as pa

from dmv2.features import FeaturePanel
from dmv2.monthly import MonthlyReturnPanel
from dmv2.target_workflow import (load_target_ret_protocol, run_shared_target_sample,
                                  validate_target_ret_protocol)
from dmv2.universe import DataContractError
from test_targets import JAN, fixture

ROOT = Path(__file__).resolve().parents[1]


class TargetWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.protocol = load_target_ret_protocol(ROOT)
        cohort, outcomes = fixture()
        coverage = pa.Table.from_pandas(cohort, preserve_index=False)
        self.features = FeaturePanel(coverage.select(['source_security_id', 'signal_date']),
                                     coverage, (), {}, {}, {})
        self.returns = MonthlyReturnPanel(pa.Table.from_pandas(outcomes, preserve_index=False),
                                          {}, {'universe': 'combined'}, (JAN,))

    def test_existing_sample_retains_keys_and_withholds_missing_ten_years(self):
        result = run_shared_target_sample(self.features, self.returns, self.protocol)
        self.assertEqual(result.summary['original_rows_retained'], 20)
        self.assertEqual(result.summary['class_available_rows'], 20)
        self.assertFalse(result.summary['ret_vector_available'])
        self.assertEqual(result.summary['ret_available_month_class_cells'], 10)
        self.assertFalse(result.summary['raw_source_reads_performed'])
        self.assertFalse(result.summary['model_fitted'])
        self.assertEqual(self.returns.table.num_rows, 20)
        self.assertEqual(self.features.table.to_pylist(), result.targets.table.select(self.features.table.column_names).to_pylist())

    def test_partial_request_withholds_ranks_without_changing_prediction_eligibility(self):
        coverage = self.features.coverage.set_column(3, 'requested', pa.array([False] + [True] * 19))
        features = FeaturePanel(self.features.table, coverage, (), {}, {}, {})
        result = run_shared_target_sample(features, self.returns, self.protocol)
        self.assertEqual(result.summary['prediction_eligible_rows'], 20)
        self.assertEqual(result.summary['class_available_rows'], 0)
        self.assertEqual(result.summary['rank_status_counts'], {'non_exhaustive_original_requests': 1})

    def test_limits_fail_before_target_worker_and_never_enable_full_work(self):
        oversized = MonthlyReturnPanel(self.returns.table, {}, {'universe': 'combined'}, (JAN,) * 4)
        with patch('dmv2.target_workflow.build_monthly_targets') as worker:
            with self.assertRaises(DataContractError):
                run_shared_target_sample(self.features, oversized, self.protocol)
            worker.assert_not_called()
        for key, value in [('model_fitting_enabled', True), ('full_history_build_enabled', True),
                           ('maximum_sample_original_rows', 3001)]:
            altered = deepcopy(self.protocol)
            altered['execution'][key] = value
            with self.subTest(key=key), self.assertRaises(DataContractError):
                validate_target_ret_protocol(altered)

    def test_context_fraction_and_class_definitions_cannot_silently_change(self):
        cases = []
        altered = deepcopy(self.protocol)
        altered['coverage']['target_coverage_fraction_gate'] = .8
        cases.append(altered)
        altered = deepcopy(self.protocol)
        altered['targets']['classes'][0] = True
        cases.append(altered)
        altered = deepcopy(self.protocol)
        altered['ret']['completed_calendar_months'] = 119
        cases.append(altered)
        for altered in cases:
            with self.subTest(altered=altered), self.assertRaises(DataContractError):
                validate_target_ret_protocol(altered)

    def test_wrong_universe_and_changed_original_ledger_are_rejected(self):
        other = MonthlyReturnPanel(self.returns.table, {}, {'universe': 'sp500'}, (JAN,))
        with self.assertRaises(DataContractError):
            run_shared_target_sample(self.features, other, self.protocol)
        changed = FeaturePanel(self.features.table.slice(1), self.features.coverage, (), {}, {}, {})
        with self.assertRaises(DataContractError):
            run_shared_target_sample(changed, self.returns, self.protocol)


if __name__ == '__main__':
    unittest.main()
