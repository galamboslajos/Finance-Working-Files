"""Synthetic new-run metadata guards and exactly-four-fit orchestration."""
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from dmv2.catalog import canonical_sha256
from dmv2.full_panel import _artifact, _json
import dmv2.fallback_models as models
from dmv2.pilot_loading import COVERAGE_METADATA, TARGET_METADATA, load_pilot_data
from dmv2.pilot_models import PilotArmResult
from dmv2.universe import DataContractError
from test_pilot_loading import fixture
from test_ret import target_fixture


PARENT = 'sha256:' + 'a' * 64


def fallback_fixture(root):
    """Copy synthetic financial bytes into a separate complete NEW annual run."""
    # Twenty names in alternate months ensure a single missing outcome still
    # leaves enough observed members for all ten class bins in that month.
    with patch('test_pilot_loading.target_fixture', side_effect=lambda **kwargs:
            target_fixture(variable_members=True, **kwargs)):
        roles, targets = fixture(root)
    run = root / 'outputs/fallback'
    for year in models.YEARS:
        old_features = roles['primary']['directory'] / 'years' / str(year)
        old_targets = targets['directory'] / 'years' / str(year)
        annual = run / 'years' / str(year)
        annual.mkdir(parents=True)
        fa = json.loads((old_features / 'audit.json').read_text())
        ta = json.loads((old_targets / 'audit.json').read_text())
        artifacts = {}
        for source in (old_features, old_targets):
            for name in json.loads((source / 'audit.json').read_text())['artifacts']:
                original = source / json.loads((source / 'audit.json').read_text())['artifacts'][name]['path']
                shutil.copyfile(original, annual / original.name)
                artifacts[name] = _artifact(annual, annual / original.name)
        _json(annual / 'audit.json', {'status': 'COMPLETE', 'year': year,
            'build_identity_sha256': PARENT, 'feature_provenance': fa['feature_provenance'],
            'target_artifact_sha256': ta['target_artifact_sha256'], 'artifacts': artifacts})
    source = Path(__file__).resolve().parents[1]
    (root / 'configs').mkdir()
    shutil.copyfile(source / 'configs/classifier_pilot.json', root / 'configs/classifier_pilot.json')
    configuration = json.loads((source / 'configs/data_limited_fallback.json').read_text())
    return run, configuration


class FakeEstimator:
    def save_model(self, path):
        Path(path).write_bytes(b'synthetic model; not a research fit')

    def get_booster(self):
        return self

    def save_config(self):
        return '{}'


class FallbackModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name).resolve()
        cls.run_directory, cls.configuration = fallback_fixture(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def plan(self):
        return models.plan_fallback_model_inputs(self.root, self.run_directory, PARENT)

    def repin(self, year, name, table):
        directory = self.run_directory / 'years' / str(year)
        path = directory / (name + '.parquet')
        audit_path = directory / 'audit.json'
        old = {path: path.read_bytes(), audit_path: audit_path.read_bytes()}
        self.addCleanup(lambda: [item.write_bytes(content) for item, content in old.items()])
        pq.write_table(table, path)
        audit = json.loads(audit_path.read_text())
        audit['artifacts'][name] = _artifact(directory, path)
        _json(audit_path, audit)

    def test_plan_never_decodes_financial_columns_or_old_pointers(self):
        original = pq.ParquetFile.read
        reads = []
        def guarded(parquet, columns=None, *args, **kwargs):
            self.assertIsNotNone(columns)
            self.assertLessEqual(set(columns), set(COVERAGE_METADATA) | set(TARGET_METADATA))
            self.assertNotIn('target_class', columns)
            self.assertNotIn('label_return', columns)
            reads.append(tuple(columns))
            return original(parquet, columns=columns, *args, **kwargs)
        with patch.object(pq.ParquetFile, 'read', guarded), \
             patch.object(models, 'load_pilot_data', side_effect=AssertionError('Financial decode')):
            plan = self.plan()
        self.assertTrue(reads)
        self.assertEqual(plan.read_plan['rows_original_train'], 1440)
        self.assertEqual(plan.read_plan['rows_original_validation'], 360)
        self.assertEqual(plan.read_plan['rows_labelled_validation'], 359)
        self.assertFalse(plan.read_plan['original_pointers_followed'])
        for annual in plan.years.values():
            self.assertEqual(annual.feature_directory, self.run_directory / 'years' / str(annual.year))
            self.assertEqual(annual.target_directory, annual.feature_directory)

    def test_new_classes_and_all_original_rows_use_shared_adapter(self):
        bundle = load_pilot_data(SimpleNamespace(root=self.root), plan=self.plan())
        self.assertEqual(len(bundle.train.X), 1440)
        self.assertEqual(len(bundle.validation.X), 360)
        self.assertEqual(bundle.validation.y.isna().sum(), 1)
        self.assertEqual(len(bundle.train.X.columns), 715)
        self.assertLess(bundle.train.ledger.outcome_end.max(), bundle.validation.ledger.outcome_start.min())

    def test_future_signal_key_fails_during_metadata_plan(self):
        path = self.run_directory / 'years/2019/features.parquet'
        table = pq.read_table(path)
        dates = table['signal_date'].to_pylist()
        dates[0] = '2020-01-31'
        table = table.set_column(table.schema.get_field_index('signal_date'), 'signal_date', pa.array(dates))
        self.repin(2019, 'features', table)
        with patch.object(models, 'load_pilot_data', side_effect=AssertionError('Financial decode')), \
             self.assertRaises(DataContractError):
            self.plan()

    def test_changed_value_bytes_fail_before_materialization(self):
        plan = self.plan()
        path = self.run_directory / 'years/2018/features.parquet'
        old = path.read_bytes()
        self.addCleanup(lambda: path.write_bytes(old))
        path.write_bytes(old + b'changed')
        with self.assertRaises(DataContractError):
            load_pilot_data(SimpleNamespace(root=self.root), plan=plan)

    def test_admission_disagreement_fails_before_numerical_decode(self):
        path = self.run_directory / 'years/2018/feature_coverage.parquet'
        table = pq.read_table(path)
        flags = table['paper_prediction_eligible'].to_pylist()
        flags[0] = False
        table = table.set_column(table.schema.get_field_index('paper_prediction_eligible'),
            'paper_prediction_eligible', pa.array(flags))
        self.repin(2018, 'feature_coverage', table)
        with self.assertRaises(DataContractError):
            self.plan()

    def test_parent_identity_mismatch_fails(self):
        with self.assertRaises(DataContractError):
            models.plan_fallback_model_inputs(self.root, self.run_directory, 'sha256:' + 'b' * 64)

    def test_known_wrong_bbby_route_cannot_be_readmitted(self):
        from dmv2.identity_amendment import SOURCE_ROUTES
        annual = SimpleNamespace(coverage=pa.Table.from_pylist([{
            'source_security_id': next(iter(SOURCE_ROUTES)),
            'identity_eligible': True, 'paper_prediction_eligible': True,
            'intrinio_security_id': 'sec_ydbQnj'}]))
        with patch.object(models, '_year_plan', return_value=annual), self.assertRaisesRegex(DataContractError, 'BBBY'):
            self.plan()

    def test_recipe_cannot_silently_change_fit_settings(self):
        cfg = json.loads(json.dumps(self.configuration))
        cfg['model_recipe']['estimators']['rf']['n_estimators'] = 2
        with self.assertRaises(DataContractError):
            models._recipe(self.root, cfg)

    def test_model_recipe_requires_new_explicit_fallback_authority(self):
        cfg = json.loads(json.dumps(self.configuration))
        cfg['missing_route_fallback']['approved'] = False
        with self.assertRaises(DataContractError):
            models._recipe(self.root, cfg)

    def test_incomplete_attempt_cannot_implicitly_fit_again(self):
        directory = self.run_directory / 'models'
        directory.mkdir()
        self.addCleanup(lambda: shutil.rmtree(directory))
        (directory / 'ATTEMPT.json').write_text('{}')
        with patch.object(models, '_phase_identity', return_value={'synthetic': True}), \
             patch.object(models, 'fit_predict_pilot_arm', side_effect=AssertionError('Fit repeated')), \
             self.assertRaisesRegex(DataContractError, 'automatic refitting'):
            models.run_fallback_models(self.root, self.run_directory, self.configuration, PARENT)

    def test_orchestration_calls_exactly_four_mocked_fits_then_reuses_without_fitting(self):
        directory = self.run_directory / 'models'
        self.addCleanup(lambda: shutil.rmtree(directory) if directory.exists() else None)
        calls = []
        def fake_fit(kind, train_X, train_y, predict_X, *, parameters, package_versions):
            self.assertEqual(len(train_X), len(train_y))
            self.assertEqual(set(train_y.astype(int)), set(range(1, 11)))
            self.assertEqual(parameters, self.configuration['model_recipe']['estimators'][kind])
            calls.append((kind, len(train_X), len(predict_X)))
            return PilotArmResult(np.full((len(predict_X), 10), .1),
                {'fit_count': 1, 'validation_labels_supplied_to_estimator': False}, FakeEstimator())
        phase = {'parent_identity_sha256': PARENT, 'synthetic': True}
        with patch.object(models, '_phase_identity', return_value=phase), \
             patch.object(models, 'fit_predict_pilot_arm', side_effect=fake_fit):
            result = models.run_fallback_models(self.root, self.run_directory, self.configuration, PARENT)
        self.assertEqual([item[0] for item in calls], ['rf', 'xgb', 'rf', 'xgb'])
        audit = models.read_fallback_models(result, identity_sha256=PARENT,
            model_identity_sha256=canonical_sha256(phase))['audit']
        self.assertEqual(audit['summary']['fitted_arms'], 4)
        self.assertEqual(audit['summary']['validation_original_rows'], 360)
        self.assertEqual(audit['summary']['validation_prediction_rows'], 360)
        self.assertEqual(audit['summary']['validation_missing_label_prediction_rows'], 1)
        for arm in models.ARMS:
            frame = pq.read_table(result / 'arms' / arm['arm_id'] / 'predictions.parquet').to_pandas()
            self.assertEqual(len(frame), 360)
            self.assertEqual(frame.target_class.isna().sum(), 1)
        with patch.object(models, '_phase_identity', return_value=phase), \
             patch.object(models, 'fit_predict_pilot_arm', side_effect=AssertionError('Repeated fit')):
            self.assertEqual(models.run_fallback_models(self.root, self.run_directory, self.configuration, PARENT), result)


if __name__ == '__main__':
    unittest.main()
