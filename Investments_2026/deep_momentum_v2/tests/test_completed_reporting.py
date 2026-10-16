"""Bounded final-report reader: preserved nine-action run plus additive phase."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from dmv2.action_loading import EVENT_TERMS
from dmv2.catalog import canonical_sha256
from dmv2.full_panel import _artifact, _hash, _json
from dmv2.sni_action import SNI_EVENT
from dmv2.universe import DataContractError
import dmv2.completed_reporting as reporting
import dmv2.fallback_reporting as passive
from test_independent_reporting import fixture as independent_fixture, receipts as independent_receipts


from dmv2.vvc_action import VVC_EVENT as _VVC_EVENT
from dmv2.tss_action import TSS_EVENT as _TSS_EVENT
VVC_EVENT = deepcopy(_VVC_EVENT)
TSS_EVENT = deepcopy(_TSS_EVENT)
SIGNALS = ['2017-12-29', '2018-01-31', '2018-02-28', '2018-03-29', '2018-04-30',
    '2018-05-31', '2018-06-29', '2018-07-31', '2018-08-31', '2018-09-28', '2018-10-31',
    '2018-11-30', '2018-12-31', '2019-01-31', '2019-02-28', '2019-03-29', '2019-04-30',
    '2019-05-31', '2019-06-28', '2019-07-31', '2019-08-30', '2019-09-30', '2019-10-31', '2019-11-29']


def fixture(root):
    run, pin = independent_fixture(root)
    identity = json.loads((run / 'build_identity.json').read_text())
    parent_cfg = identity['configuration']
    (root / 'configs').mkdir(exist_ok=True)
    _json(root / 'configs/independent_backtest.json', parent_cfg)
    cfg = deepcopy(parent_cfg)
    cfg.update(protocol_id='dmv2-independent-quote-action-completion-2018-2019-v1',
        parent_configuration_sha256=canonical_sha256(parent_cfg))
    cfg['backtest']['events'] = 'original_nine_unchanged_plus_verified_VVC_and_TSS'
    _json(root / 'configs/independent_backtest_completion.json', cfg)
    old_directory = run / 'backtest'
    old_phase = json.loads((old_directory / 'build_identity.json').read_text())
    old_phase['events'] = [dict(event_id=ticker+'-'+terms[1], predecessor_id=terms[0],
        effective_date=terms[2], cash_per_raw_share=terms[3], stock_ratio=terms[4],
        preserved_audit={'original': ticker}) for ticker, terms in EVENT_TERMS.items()] + [deepcopy(SNI_EVENT)]
    _json(old_directory / 'build_identity.json', old_phase)
    old_audit = json.loads((old_directory / 'audit.json').read_text())
    old_audit['phase_identity_sha256'] = canonical_sha256(old_phase)
    old_audit['artifacts']['build_identity.json'] = _artifact(old_directory, old_directory / 'build_identity.json')
    old_audit.update(complete_proxy_paths=56, failed_closed_paths=8)
    _json(old_directory / 'audit.json', old_audit)
    directory = run / 'backtest_completed'; directory.mkdir()
    phase = {'phase': 'independent_quote_action_completion_v1', 'parent_identity_sha256': pin,
        'configuration_sha256': canonical_sha256(cfg),
        'parent_configuration_sha256': canonical_sha256(parent_cfg),
        'models_fitted': False, 'preprocessing_fitted': False,
        'events': old_phase['events'] + [deepcopy(VVC_EVENT), deepcopy(TSS_EVENT)],
        'source_pins': [_artifact(root, path) for path in (old_directory / 'audit.json',
            root / 'configs/independent_backtest.json', root / 'configs/independent_backtest_completion.json')]}
    _json(directory / 'build_identity.json', phase)
    comparison = pd.read_parquet(old_directory / 'comparison.parquet')
    comparison['expected_sessions'] = 503
    comparison['observed_sessions'] = 503
    comparison.to_parquet(directory / 'comparison.parquet', index=False)
    coverage = pd.DataFrame([dict(arm=arm, basket=basket, signal_date=signal,
        long_candidates=9 if i==0 else 12, short_candidates=37,
        selected_names_per_side=9 if i==0 else 10, requested_top_k=10,
        top_k_per_side=9 if i==0 else 10, long_selected=9 if i==0 else 10,
        short_selected=9 if i==0 else 10, long_per_name_weight=.5/(9 if i==0 else 10),
        short_per_name_weight=-.5/(9 if i==0 else 10),
        effective_per_name_abs_weight=.5/(9 if i==0 else 10), status='available')
        for arm in sorted(reporting.ARMS) for basket in ('class_1_10', 'class_2_9')
        for i, signal in enumerate(SIGNALS)])
    coverage.to_parquet(directory / 'selection_coverage.parquet', index=False)
    audit = {'status': 'COMPLETE', 'phase': 'independent_quote_action_completion_v1',
        'build_identity_sha256': pin, 'phase_identity_sha256': canonical_sha256(phase),
        'scenarios': 64, 'models_fitted': False, 'financial_2020_plus_read': False,
        'complete_proxy_paths': 64, 'failed_closed_paths': 0, 'execution_seconds': .01,
        'artifacts': {name: _artifact(directory, directory / name) for name in (
            'build_identity.json', 'comparison.parquet', 'selection_coverage.parquet')}}
    _json(directory / 'audit.json', audit)
    return run, pin


def receipts(run):
    original = independent_receipts(run)
    return {'features': original['features'], 'models': original['models'],
        'original_backtest': original['backtest'],
        'backtest': 'sha256:' + _hash(run / 'backtest_completed/audit.json')}


def repin_phase(run, phase):
    directory = run / 'backtest_completed'
    _json(directory / 'build_identity.json', phase)
    audit = json.loads((directory / 'audit.json').read_text())
    audit['phase_identity_sha256'] = canonical_sha256(phase)
    audit['artifacts']['build_identity.json'] = _artifact(directory, directory / 'build_identity.json')
    _json(directory / 'audit.json', audit)


class CompletedReportingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.run, self.pin = fixture(self.root)

    def read(self, **updates):
        kwargs = dict(identity_sha256=self.pin, expected_receipts=receipts(self.run), verify_sources=False)
        kwargs.update(updates)
        return reporting.read_completed_run(self.root, self.run, **kwargs)

    def test_passive_reader_verifies_small_receipts_and_only_three_report_files(self):
        original = reporting.pq.ParquetFile
        allowed = {self.run / 'backtest/comparison.parquet',
            self.run / 'backtest_completed/comparison.parquet',
            self.run / 'backtest_completed/selection_coverage.parquet'}
        def reports_only(path, *args, **kwargs):
            self.assertIn(Path(path), allowed)
            return original(path, *args, **kwargs)
        with patch.object(reporting.pq, 'ParquetFile', side_effect=reports_only), \
             patch('pickle.load', side_effect=AssertionError('Model deserialization')):
            saved = self.read()
        self.assertEqual(len(saved['backtest_comparison']), 64)
        self.assertEqual(len(saved['selection_coverage']), 192)
        self.assertEqual(saved['summary']['completed_action_count'], 11)
        self.assertEqual(saved['original_backtest_summary']['complete_proxy_paths'], 56)
        self.assertEqual(saved['backtest_summary']['complete_proxy_paths'], 64)
        self.assertFalse(saved['summary']['models_deserialized'])
        self.assertFalse(saved['summary']['numerical_work_started'])

    def test_four_final_receipts_required_and_each_must_be_full_sha256(self):
        for label in ('features', 'models', 'backtest', 'original_backtest'):
            for invalid in (None, '', 'sha256:bad'):
                with self.subTest(label=label, invalid=invalid):
                    expected = receipts(self.run); expected[label] = invalid
                    with self.assertRaises(DataContractError): self.read(expected_receipts=expected)

    def test_changed_original_receipt_and_changed_completed_receipt_fail_closed(self):
        for relative in ('backtest/audit.json', 'backtest_completed/audit.json'):
            with self.subTest(relative=relative):
                expected = receipts(self.run); path = self.run / relative; before = path.read_bytes()
                path.write_bytes(before + b'\n')
                with self.assertRaises(DataContractError): self.read(expected_receipts=expected)
                path.write_bytes(before)

    def test_original_nine_cannot_change_even_with_self_consistent_new_phase_hash(self):
        path = self.run / 'backtest_completed/build_identity.json'; phase = json.loads(path.read_text())
        phase['events'][0]['cash_per_raw_share'] += 1.
        repin_phase(self.run, phase)
        with self.assertRaises(DataContractError): self.read()

    def test_new_vvc_or_tss_terms_cannot_change_with_self_consistent_receipts(self):
        path = self.run / 'backtest_completed/build_identity.json'; original = json.loads(path.read_text())
        for index in (9, 10):
            with self.subTest(index=index):
                phase = deepcopy(original); phase['events'][index]['cash_per_raw_share'] += 1.
                repin_phase(self.run, phase)
                with self.assertRaises(DataContractError): self.read()
                repin_phase(self.run, original)

    def test_original_backtest_source_pin_is_required(self):
        path = self.run / 'backtest_completed/build_identity.json'; phase = json.loads(path.read_text())
        phase['source_pins'] = [item for item in phase['source_pins'] if not item['path'].endswith('backtest/audit.json')]
        repin_phase(self.run, phase)
        with self.assertRaises(DataContractError): self.read()

    def test_configuration_byte_drift_rejected_by_passive_reader(self):
        path = self.root / 'configs/independent_backtest_completion.json'
        path.write_bytes(path.read_bytes() + b'\n')
        with self.assertRaises(DataContractError): self.read()

    def test_wrong_completed_scope_or_new_fitting_claim_rejected(self):
        path = self.run / 'backtest_completed/audit.json'; original = json.loads(path.read_text())
        for field, value in (('models_fitted', True), ('financial_2020_plus_read', True), ('scenarios', 65)):
            with self.subTest(field=field):
                audit = deepcopy(original); audit[field] = value; _json(path, audit)
                with self.assertRaises(DataContractError): self.read()
                _json(path, original)

    def test_duplicate_month_or_missing_model_basket_in_coverage_is_not_a_full_grid(self):
        directory = self.run / 'backtest_completed'; path = directory / 'selection_coverage.parquet'
        frame = pd.read_parquet(path); frame.iloc[1] = frame.iloc[0]
        frame.to_parquet(path, index=False)
        audit = json.loads((directory / 'audit.json').read_text())
        audit['artifacts'][path.name] = _artifact(directory, path); _json(directory / 'audit.json', audit)
        with self.assertRaises(DataContractError): self.read()

    def test_less_than_full_503_sessions_or_missing_comparison_row_rejected(self):
        directory = self.run / 'backtest_completed'; path = directory / 'comparison.parquet'
        frame = pd.read_parquet(path); frame.loc[0, 'expected_sessions'] = 502
        frame.to_parquet(path, index=False)
        audit = json.loads((directory / 'audit.json').read_text())
        audit['artifacts'][path.name] = _artifact(directory, path); _json(directory / 'audit.json', audit)
        with self.assertRaises(DataContractError): self.read()


if __name__ == '__main__': unittest.main()
