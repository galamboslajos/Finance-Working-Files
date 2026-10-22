"""Invented metadata/vectors only: no saved research models or panels opened."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dmv2 import rf_importance as m


def pp(names, dropped=None):
    dropped = {} if dropped is None else dropped
    retained = [n for n in names if n not in dropped]
    manifest = dict(input_columns=names, retained_columns=retained, dropped_columns=dropped,
        output_columns=[c for n in retained for c in (n, n+'__missing')],
        training_only_schema_selection=True,
        imputation='training-partition finite-value median; labels never imputed',
        preprocessing_policy='train_partition_median_missing_indicator_v1')
    return dict(manifest=manifest, fingerprint=m.canonical(manifest))


class MetadataFixture:
    def __init__(self, root):
        self.root = root; self.run = root/'invented-run'; self.run.mkdir()
        self.names = ['fake_'+str(i) for i in range(715)]
        helper = root/'invented_preprocessing.py'; helper.write_text('# invented\n')
        params = dict(n_estimators=500, n_jobs=4)
        self.source = dict(schema='dmv2_unstopped_target_comparison_models_v1',
            status='SEALED_BEFORE_FINANCIAL_READ', expected_fits=20, expected_preprocessors=4,
            predictor_names=self.names, runtime=m._runtime(), helper_pins={'preprocessing.py':m.pin(helper)},
            folds={str(y):{'fixed_rf_parameters':params} for y in m.YEARS})
        self.source['plan_sha256'] = m._logical(self.source)
        m.save(self.run/'read_plan.json', self.source)
        inventory = {}
        for y in m.YEARS:
            fold=self.run/'folds'/str(y); (fold/'prepared').mkdir(parents=True)
            (fold/'heads/endpoint').mkdir(parents=True)
            processor = pp(self.names)
            pp_pin=m.save(fold/'prepared/preprocessor.json', processor)
            prepared=dict(status='PREPARED', year=y, preprocessor_fits=1, estimator_fits=0,
                plan_sha256=self.source['plan_sha256'], artifacts={'preprocessor':pp_pin},
                preprocessor_fingerprint=processor['fingerprint'], matrix_width=1430)
            prepared_pin=m.save(fold/'prepared/receipt.json', prepared)
            # Deliberately not a pickle; tests only plan and authenticate its bytes.
            model=fold/'heads/endpoint/model.joblib'; model.write_bytes(b'not-a-model')
            fit=dict(status='FIT_COMPLETE', year=y, head='endpoint', estimator_fits=1,
                prepared_pin=prepared_pin, model=m.pin(model), plan_sha256=self.source['plan_sha256'],
                saved_inference_effective_parameters={**params, 'n_jobs':1},
                native_class_ids=list(range(1,11)), probability_class_ids=list(range(1,11)),
                preprocessor_fingerprint=processor['fingerprint'], matrix_width=1430)
            fit_pin=m.save(fold/'heads/endpoint/FIT_COMPLETE.json', fit)
            for item in [pp_pin,prepared_pin,fit_pin,m.pin(model)]:
                inventory[item['path']]={'pin':item, 'format':'json' if item['path'].endswith('.json') else 'joblib'}
        (self.run/'verification').mkdir()
        self.vplan=dict(status='SEALED_BEFORE_FINANCIAL_READ', phase='models',
            model_plan_pin=m.pin(self.run/'read_plan.json'), artifacts=inventory)
        m.save(self.run/'verification/models_read_plan.json', self.vplan)
        self.proof=dict(status='PASS',phase='models',verified_estimator_fits=20,
            verified_shared_preprocessors=4,new_estimator_fits=0,financial_2024_plus_values_read=False,
            model_plan_pin=m.pin(self.run/'read_plan.json'),
            verification_plan_pin=m.pin(self.run/'verification/models_read_plan.json'))
        m.save(self.run/'verification/models.json',self.proof)

    def replace(self, relative, value):
        p=self.run/relative; p.write_text(json.dumps(value))


class ManifestAndImportanceTests(unittest.TestCase):
    def test_value_missing_components_preserve_global_mass(self):
        rows=m.importance_rows(['a','b'],['a','b'],[.2,.1,.4,.3],2020)
        self.assertEqual(rows[0]['value_importance'],.2)
        self.assertEqual(rows[0]['missing_importance'],.1)
        self.assertAlmostEqual(sum(x['total_importance'] for x in rows),1.)

    def test_dropped_predictor_is_null_not_zero(self):
        rows=m.importance_rows(['a','b','c'],['a','c'],[.2,.1,.4,.3],2020)
        self.assertFalse(rows[1]['retained']); self.assertIsNone(rows[1]['total_importance'])
        self.assertIsNone(rows[1]['value_importance']); self.assertIsNone(rows[1]['missing_importance'])

    def test_no_normalization_of_components_separately(self):
        rows=m.importance_rows(['a','b'],['a','b'],[.1,.0,.4,.5],2020)
        self.assertEqual(sum(x['value_importance'] for x in rows),.5)
        self.assertEqual(sum(x['missing_importance'] for x in rows),.5)

    def test_invalid_vectors_fail_closed(self):
        for vector in ([1.], [.2,.2], [-.1,1.1], [float('nan'),1.], [float('inf'),0.]):
            with self.subTest(vector=vector), self.assertRaises(ValueError):
                m.importance_rows(['a'],['a'],vector,2020)

    def test_reordered_retained_fails(self):
        with self.assertRaises(ValueError): m.importance_rows(['a','b'],['b','a'],[.25]*4,2020)

    def test_valid_manifest_with_dropped_feature(self):
        self.assertEqual(m.validate_manifest(pp(['a','b'],{'b':'no finite training values'}),['a','b']),['a'])

    def test_manifest_fingerprint_uses_utf8(self):
        x=pp(['é']); self.assertEqual(m.validate_manifest(x,['é']),['é'])
        wrong=copy.deepcopy(x); wrong['fingerprint']=x['fingerprint'][7:]
        with self.assertRaises(ValueError): m.validate_manifest(wrong,['é'])

    def test_output_must_be_interleaved(self):
        x=pp(['a','b']); x['manifest']['output_columns']=['a','b','a__missing','b__missing']
        x['fingerprint']=m.canonical(x['manifest'])
        with self.assertRaises(ValueError): m.validate_manifest(x,['a','b'])

    def test_dropped_partition_must_be_exact(self):
        x=pp(['a','b']); x['manifest']['dropped_columns']={'other':'missing'}
        x['fingerprint']=m.canonical(x['manifest'])
        with self.assertRaises(ValueError): m.validate_manifest(x,['a','b'])

    def test_schema_policy_must_be_training_only(self):
        x=pp(['a']); x['manifest']['training_only_schema_selection']=False
        x['fingerprint']=m.canonical(x['manifest'])
        with self.assertRaises(ValueError): m.validate_manifest(x,['a'])


class ClosedMetadataTests(unittest.TestCase):
    def test_duplicate_and_nonfinite_json_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'x.json'
            for text in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}'):
                path.write_text(text)
                with self.assertRaises(ValueError): m.read_json(path)

    def test_json_bound(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'x'; p.write_bytes(b' '*100)
            with patch.object(m,'MAX_JSON',99), self.assertRaises(ValueError): m.read_json(p)

    def test_pinned_bytes_changed(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'x'; p.write_bytes(b'a'); item=m.pin(p); p.write_bytes(b'b')
            with self.assertRaises(ValueError): m.check(item)

    def test_save_preserves_existing_files(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'x'; m.save(p,{'a':1})
            with self.assertRaises(FileExistsError): m.save(p,{'a':2})
            self.assertEqual(m.read_json(p),{'a':1})


class PlanningTests(unittest.TestCase):
    def setUp(self):
        self.td=tempfile.TemporaryDirectory(); self.root=Path(self.td.name)
        self.fixture=MetadataFixture(self.root)
    def tearDown(self): self.td.cleanup()
    def plan(self): return m.plan_rf_importance(self.fixture.run,self.root/'analysis/read_plan.json')

    def test_model_pass_is_gate_without_backtest_ledger(self):
        self.assertFalse((self.fixture.run/'ledger.json').exists())
        plan=self.plan()
        self.assertEqual(plan['model_read_bytes'],44)
        self.assertEqual(plan['source_panel_reads'],0)
        self.assertEqual(plan['new_estimator_fits'],0)
        self.assertEqual(plan['years'],[2020,2021,2022,2023])
        self.assertEqual(plan['plan_sha256'],m._logical(plan))

    def test_incomplete_model_proof_fails(self):
        proof=copy.deepcopy(self.fixture.proof); proof['verified_estimator_fits']=19
        self.fixture.replace('verification/models.json',proof)
        with self.assertRaises(ValueError): self.plan()

    def test_failed_model_proof_fails(self):
        proof=copy.deepcopy(self.fixture.proof); proof['status']='FAILED_PRESERVED'
        self.fixture.replace('verification/models.json',proof)
        with self.assertRaises(ValueError): self.plan()

    def test_changed_model_bytes_fail_before_deserialization(self):
        (self.fixture.run/'folds/2020/heads/endpoint/model.joblib').write_bytes(b'changed')
        with self.assertRaises(ValueError): self.plan()

    def test_changed_preprocessor_fails_before_deserialization(self):
        p=self.fixture.run/'folds/2020/prepared/preprocessor.json'; p.write_text('{}')
        with self.assertRaises(ValueError): self.plan()

    def test_model_byte_budget_checked_at_plan(self):
        with patch.object(m,'LIMITS',{**m.LIMITS,'model_read_bytes':43}), self.assertRaises(ValueError): self.plan()

    def test_original_verification_plan_tamper_fails(self):
        vp=copy.deepcopy(self.fixture.vplan); vp['status']='NOT_SEALED'
        self.fixture.replace('verification/models_read_plan.json',vp)
        with self.assertRaises(ValueError): self.plan()

    def test_plan_file_preserved(self):
        self.plan()
        with self.assertRaises(ValueError): self.plan()

    def test_source_run_is_not_mutated_by_analysis_plan(self):
        with self.assertRaises(ValueError):
            m.plan_rf_importance(self.fixture.run,self.fixture.run/'new_analysis.json')
        self.assertFalse((self.fixture.run/'new_analysis.json').exists())

    def test_changed_analysis_plan_rejected_before_output(self):
        self.plan(); path=self.root/'analysis/read_plan.json'; value=m.read_json(path)
        value['head']='cash_long'; path.write_text(json.dumps(value))
        with self.assertRaises(ValueError): m.extract_rf_importance(path,self.root/'output')
        self.assertFalse((self.root/'output').exists())


if __name__ == '__main__': unittest.main()
