"""Cached reconstruction boundaries use structural fixtures, never real data."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from dmv2.context_features import context_feature_specs
from dmv2.context_sensitivity import rebuild_cached_year
from dmv2.features import registered_feature_specs
from dmv2.full_panel import _artifact
from dmv2.universe import DataContractError


class CachedSensitivityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.year = self.root / 'years/2010'
        self.year.mkdir(parents=True)
        self.specs = registered_feature_specs()
        (self.root / 'feature_dictionary.json').write_text(json.dumps(self.specs))
        names = [s['name'] for s in self.specs]
        self.features = pa.table({'source_security_id': ['A', 'B'],
            'signal_date': ['2009-12-31'] * 2, **{n: [1., 2.] for n in names}})
        self.ledger = pa.table({'source_security_id': ['A', 'B'], 'signal_date': ['2009-12-31'] * 2,
            'intrinio_security_id': ['P1', 'P2'], 'identity_eligible': [True, True],
            'requested': [True, True], 'paper_prediction_eligible': [True, True],
            'available_feature_count': [728, 728], 'feature_status': ['partial_features'] * 2})
        artifacts = {}
        tables = {'features': self.features, 'feature_coverage': self.ledger,
            'original_cohort': self.features.select(['source_security_id', 'signal_date']),
            'feature_history_daily': pa.table({'trading_date': ['2009-12-31']}),
            'daily_cohorts': self.features.select(['source_security_id', 'signal_date']),
            'exchange_calendar': pa.table({'trading_date': ['2009-12-31'], 'calendar_name': ['fixture'],
                                           'is_trading_session': [True]})}
        for name, table in tables.items():
            p = self.year / (name + '.parquet')
            pq.write_table(table, p)
            artifacts[name] = _artifact(self.year, p)
        audit = {'artifacts': artifacts, 'summary': {},
                 'feature_provenance': {'requested_ids': ['P1', 'P2']}}
        (self.year / 'audit.json').write_text(json.dumps(audit))

    def context(self):
        table = self.features.select(['source_security_id', 'signal_date']).to_pandas().iloc[::-1].copy()
        table = pd.concat([table, pd.DataFrame({s['name']: [8., np.nan]
            for s in context_feature_specs()}, index=table.index)], axis=1)
        return SimpleNamespace(table=table, specs=context_feature_specs(),
            coverage=pd.DataFrame({'feature_name': ['fixture', 'benchmark_daily_ew_return_1s']}),
            provenance={'context_policy': 'available_cohort_80pct'})

    def test_alignment_and_exact_preservation_of_all313_other_inputs(self):
        with patch('dmv2.context_sensitivity.compute_compact_context_features', return_value=self.context()) as worker:
            panel, _ = rebuild_cached_year(self.year)
        preserved = [s['name'] for s in self.specs if s['role'] in ('paper', 'technical')]
        self.assertEqual(len(preserved), 313)
        self.assertTrue(panel.table.select(preserved).equals(self.features.select(preserved)))
        # Reversed context rows must align by original source/date keys.
        self.assertEqual(panel.table['market_ew_return_1s'].to_pylist(), [None, 8.])
        self.assertEqual(panel.coverage['paper_prediction_eligible'].to_pylist(), [True, True])
        self.assertFalse(panel.provenance['paper_and_technical_recalculated'])
        self.assertEqual(worker.call_args.kwargs['context_policy'], 'available_cohort_80pct')
        self.assertEqual(panel.diagnostics['daily_benchmark'].feature_name.tolist(), ['benchmark_daily_ew_return_1s'])

    def test_cache_tampering_and_wrong_policy_fail_before_context(self):
        with patch('dmv2.context_sensitivity.compute_compact_context_features') as worker:
            with self.assertRaises(DataContractError):
                rebuild_cached_year(self.year, context_policy='available_cohort_90pct')
            p = self.year / 'features.parquet'
            p.write_bytes(p.read_bytes()[:-1] + b'X')
            with self.assertRaises(DataContractError):
                rebuild_cached_year(self.year)
            worker.assert_not_called()

    def test_changed_original_context_keys_are_rejected(self):
        bad = self.context()
        bad.table.loc[bad.table.index[0], 'source_security_id'] = 'foreign'
        with patch('dmv2.context_sensitivity.compute_compact_context_features', return_value=bad):
            with self.assertRaises(DataContractError):
                rebuild_cached_year(self.year)

    def test_cached_ledger_order_cannot_silently_assign_to_other_stocks(self):
        path = self.year / 'feature_coverage.parquet'
        pq.write_table(self.ledger.take(pa.array([1, 0])), path)
        audit_path = self.year / 'audit.json'
        audit = json.loads(audit_path.read_text())
        audit['artifacts']['feature_coverage'] = _artifact(self.year, path)
        audit_path.write_text(json.dumps(audit))
        with patch('dmv2.context_sensitivity.compute_compact_context_features') as worker:
            with self.assertRaisesRegex(DataContractError, 'key order'):
                rebuild_cached_year(self.year)
            worker.assert_not_called()


if __name__ == '__main__':
    unittest.main()
