import json
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import tempfile
import unittest
from unittest.mock import patch

from dmv2.catalog import canonical_sha256
from dmv2.data_limited_fallback import (_configuration, _directory, _save_year,
                                        _verify_year, plan_rebuild)
from dmv2.full_panel import _hash
from dmv2.universe import DataContractError


class FallbackOrchestrationTests(unittest.TestCase):
    def test_ten_year_plan_releases_each_full_membership_table(self):
        import weakref
        cfg = _configuration(Path(__file__).parents[1] / 'configs/data_limited_fallback.json')
        previous = []
        def yearly(root, year, **kwargs):
            if previous:
                self.assertIsNone(previous[-1]())
            class Annual:
                pass
            part = Annual()
            part.fingerprint = 'sha256:' + str(year)
            part.read_plan = {'projected_compressed_bytes': 10, 'original_monthly_rows': 12}
            part.provenance = {'year': year}
            previous.append(weakref.ref(part))
            return part
        with patch('dmv2.data_limited_fallback.load_identity_amendment'), patch(
                'dmv2.data_limited_fallback.plan_fallback_year', side_effect=yearly):
            plans, identity, pin = plan_rebuild(Path('/private/tmp'), cfg,
                SimpleNamespace(fingerprint='sha256:quote', provenance={}))
        self.assertEqual(identity['original_stock_months'], 120)
        self.assertEqual(len(plans), 10)
        self.assertTrue(all(ref() is None for ref in previous))

    def test_frozen_decision_is_specific_and_unrepaired(self):
        cfg = _configuration(Path(__file__).parents[1] / 'configs/data_limited_fallback.json')
        assert cfg['feature_outcome_years'] == list(range(2010, 2020))
        assert cfg['execution_restrictions']['valid_historical_training_rows_retained'] is True
        assert cfg['missing_route_fallback']['replacement_native_id'] is None
        assert set(cfg['execution_restrictions']['native_ids']) == {'sec_gjP4xM', 'sec_zmYNnz'}
        assert cfg['interpretation']['selection_and_survivorship_bias_removed'] is False


    def test_configuration_mutation_cannot_silently_change_restrictions(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        tmp_path = Path(temporary.name)
        cfg = json.loads((Path(__file__).parents[1] / 'configs/data_limited_fallback.json').read_text())
        cfg['execution_restrictions']['native_ids']['extra'] = 'hindsight'
        path = tmp_path / 'changed.json'
        path.write_text(json.dumps(cfg))
        with self.assertRaisesRegex(DataContractError, 'frozen'):
            _configuration(path)


    def test_new_output_namespace_cannot_overwrite_reference(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        tmp_path = Path(temporary.name)
        pin = canonical_sha256({'phase': 'new'})
        assert _directory(tmp_path, pin) == tmp_path.resolve() / 'outputs/data_limited_backtest' / ('2010-2019-' + pin[7:23])


    def test_complete_annual_checkpoint_authenticates_saved_separate_daily_view(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        tmp_path = Path(temporary.name)
        import dmv2.data_limited_fallback as mod
        original = mod._save_target_state
        mod._save_target_state = lambda directory, targets, artifacts: None
        self.addCleanup(setattr, mod, '_save_target_state', original)
        keys = pa.table({'source_security_id': ['old_bbby', 'valid'],
                         'signal_date': ['2009-12-31', '2009-12-31']})
        coverage = keys.append_column('paper_prediction_eligible', pa.array([False, True]))
        target_table = keys.append_column('target_available', pa.array([False, True]))
        daily = pa.table({'intrinio_security_id': ['valid'], 'trading_date': ['2010-01-04'], 'open': [10.]})
        result = SimpleNamespace(
            loaded=SimpleNamespace(plan=SimpleNamespace(year=2010, original_cohort=keys,
                calendar=pa.table({'date': ['2010-01-04']}),
                read_plan={'monthly_wrong_route_rows': 1, 'daily_wrong_route_rows': 20}),
                feature_prices=daily, outcome_prices=daily),
            features=SimpleNamespace(table=keys, coverage=coverage, provenance={}, summary={}),
            monthly=SimpleNamespace(table=keys, provenance={}, summary={}),
            targets=SimpleNamespace(table=target_table, artifact_sha256='sha256:target'),
            provenance={})
        identity = {'configuration': {'research_status': 'DATA_LIMITED_RESTRICTED_UNIVERSE'},
                    'quote_amendments_fingerprint': 'sha256:quotes'}
        pin = canonical_sha256(identity)
        audit = _save_year(tmp_path, result, identity, pin, 1.)
        assert audit['summary']['original_rows'] == 2
        assert audit['summary']['prediction_eligible_rows'] == 1
        assert audit['summary']['labelled_rows'] == 1
        assert audit['quote_amendments_fingerprint'] == 'sha256:quotes'
        assert 'daily_prices_raw' in audit['artifacts']
        assert _verify_year(tmp_path, pin, 2010) == audit
        daily_path = tmp_path / 'daily_prices_raw.parquet'
        before = _hash(daily_path)
        daily_path.write_bytes(daily_path.read_bytes() + b'changed')
        assert before != _hash(daily_path)
        with self.assertRaisesRegex(DataContractError, 'artifact changed'):
            _verify_year(tmp_path, pin, 2010)
