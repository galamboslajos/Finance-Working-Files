"""Fabricated archive trees only; no real source inventory/model reads/R writes."""
from pathlib import Path
from unittest.mock import patch
import contextlib
import importlib.util
import json
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('archive_under_test',Path(__file__).with_name('install_corrected_refit.py'))
g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)


class Fixture:
    def __init__(self,root,withheld=120):
        self.root=Path(root);self.source=self.root/'source';self.repo=self.root/'repo';self.control=self.root/'control'
        self.dest=self.repo/'outputs/corrected_refit/2020-2023-20261021-v1'
        self.source.mkdir();self.repo.mkdir();self.control.mkdir();self.calls=[]
        def save(relative,value):return g.save_new(self.source/relative,value)
        save('control.py',{'fake':'not executable'})
        protocol=save('model_protocol.json',{'status':'SEALED_BEFORE_FINANCIAL_READ'})
        model=save('run/verification/models.json',{'schema':'dmv2-independent-corrected-model-final-verification/v1',
            'status':'PASS','verified_estimator_fits':20,'verified_training_only_preprocessing_states':8,
            'outcome_years':[2020,2021,2022,2023],'outcome_months':48,'same_cohort_all_arms':True,
            'protocol_pin':protocol,'unused_financial_summary':{'invented_loss':2.2}})
        progress=save('run/models/pipeline/progress.json',{'status':'COMPLETE_VERIFIED',
            'completed_estimator_fits':20,'completed_preprocessors':8,'completed_phases':50,
            'completed_annual_verifications':4,'final_verification_pin':model})
        expected={name:save('run/backtest/'+name+'.json',{'opaque_provenance':name}) for name in g.BACKTEST_PINS}
        backtest=save('run/verification/backtest.json',{'schema':'dmv2-independent-corrected-backtest-verification/v1',
            'status':'PASS','scenarios':120,'sessions':1006,'outcome_months':48,'selection_sets':15,
            'complete_proxy_paths':120-withheld,'withheld_paths':withheld,'all_scenarios_retained':True,
            'original_source_pins_unchanged':True,'missing_selected_positions_dropped':False,
            'new_estimator_fits':0,'new_inference_calls':0,**expected})
        report=save('run/reports/models_v1/receipt.json',{'status':'COMPLETE','final_verification_pin':model,
             'model_fits':0,'inference_calls':0})
        save('run/control/fabricated_guard.json',{'status':'COMPLETE'})
        self.opaque=self.source/'run/models/opaque_model.bin';self.opaque.write_bytes(b'never deserialize this model\x00\xff')
        save('attempts/failed/metadata.json',{'status':'FAILED_METADATA','original_path':str(self.opaque)})
        self.doc=self.source/'README.md';self.doc.write_text('Original research notes.\n')
        (self.source/'empty_preserved_directory').mkdir()
        external=self.repo/'original_input.bin';external.write_bytes(b'original input unchanged')
        self.scope={'schema':'dmv2-corrected-refit-archive-scope/v1','status':'SEALED_AFTER_NUMERICAL_COMPLETION_BEFORE_ARCHIVE',
            'source_root':str(self.source),'destination':str(self.dest),'all_numerical_stages_terminal':True,
            'gates':{'progress':progress,'models':model,'backtest':backtest,'model_report':report},
            'model_protocol_pin':protocol,'backtest_expected_pins':expected,
            'artifacts':{'model_proof':model,'backtest_proof':backtest,'model_report':report,'opaque_model':g.pin(self.opaque)},
            'original_input_pins':[g.pin(external)]}
        self.scope_path=self.control/'scope.json';g.save_new(self.scope_path,self.scope)

    @contextlib.contextmanager
    def patches(self):
        with patch.object(g,'SOURCE',self.source),patch.object(g,'REPO',self.repo), \
             patch.object(g,'CONTROL',self.control),patch.object(g,'DESTINATION',self.dest), \
             patch.object(g,'require_gate_reservation',side_effect=lambda *args:self.calls.append(args)):
            yield

    def prepare(self):
        gp=g.make_gate_plan(self.scope_path,self.control/'gate_plan.json')
        self.gate_plan=gp
        self.install_plan=g.prepare(gp['path'],gp['sha256'],'fresh-gate-reservation',self.control/'install_plan.json')
        return self.install_plan


class Tests(unittest.TestCase):
    def test_all_withheld_verified_paths_can_archive_and_every_byte_is_preserved(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            f=Fixture(d,withheld=120)
            with f.patches():
                pp=f.prepare();before=g.inventory(f.source);receipt=g.execute(pp['path'],pp['sha256'])
                self.assertEqual(g.inventory(f.source),before)
                actual=g.read(receipt['path']);self.assertEqual(actual['status'],'COMPLETE_INSTALLED')
                self.assertEqual(actual['gate_summary']['complete_proxy_paths'],0)
                self.assertEqual(actual['gate_summary']['withheld_paths'],120)
                catalog=g.read(f.dest/'installed_catalog.json')
                for relative,item in before['files'].items():
                    self.assertEqual((f.dest/relative).read_bytes(),Path(item['path']).read_bytes())
                    self.assertEqual(catalog['files'][relative]['path'],str(f.dest/relative))
                self.assertEqual(catalog['artifacts']['opaque_model']['path'],str(f.dest/'run/models/opaque_model.bin'))
                legacy=(f.source/'attempts/failed/metadata.json').read_bytes()
                self.assertEqual((f.dest/'attempts/failed/metadata.json').read_bytes(),legacy)
                self.assertTrue((f.dest/'empty_preserved_directory').is_dir())
                self.assertFalse((f.repo/'docs').exists())
                self.assertEqual(len(f.calls),1)

    def test_gate_plan_never_parses_metric_bearing_proof_json(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            f=Fixture(d);original=g.read
            def metadata_only(path):
                self.assertEqual(Path(path),f.scope_path)
                return original(path)
            with f.patches(),patch.object(g,'read',side_effect=metadata_only):
                g.make_gate_plan(f.scope_path,f.control/'gate_plan.json')

    def test_failed_or_incomplete_proofs_fail_before_inventory(self):
        for field in ('models','backtest','progress'):
            with self.subTest(field=field),tempfile.TemporaryDirectory(dir='/private/tmp') as d:
                f=Fixture(d);proofs={name:g.read(item['path']) for name,item in f.scope['gates'].items()}
                proofs[field]['status']='RUNNING'
                with self.assertRaises(ValueError):g.gate_summary(f.scope,proofs)

    def test_missing_scenarios_and_dropped_positions_are_not_accepted(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            f=Fixture(d);proofs={name:g.read(item['path']) for name,item in f.scope['gates'].items()}
            proofs['backtest']['withheld_paths']=119
            with self.assertRaisesRegex(ValueError,'disappeared'):g.gate_summary(f.scope,proofs)
            proofs['backtest']['withheld_paths']=120;proofs['backtest']['missing_selected_positions_dropped']=True
            with self.assertRaisesRegex(ValueError,'backtest PASS'):g.gate_summary(f.scope,proofs)

    def test_source_mutation_or_new_files_after_seal_fail_before_destination_creation(self):
        for mutation in ('change','add'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory(dir='/private/tmp') as d:
                f=Fixture(d)
                with f.patches():
                    pp=f.prepare()
                    (f.opaque if mutation=='change' else f.source/'new_unsealed.txt').write_bytes(b'changed')
                    with self.assertRaisesRegex(ValueError,'after inventory seal'):g.execute(pp['path'],pp['sha256'])
                    self.assertFalse(f.dest.exists())

    def test_existing_destination_is_never_overwritten(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            f=Fixture(d)
            with f.patches():
                pp=f.prepare();f.dest.mkdir(parents=True);sentinel=f.dest/'keep';sentinel.write_bytes(b'keep')
                with self.assertRaisesRegex(ValueError,'existing archive'):g.execute(pp['path'],pp['sha256'])
                self.assertEqual(sentinel.read_bytes(),b'keep')

    def test_symlink_source_entry_and_control_escape_rejected(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            f=Fixture(d)
            with f.patches():
                (f.source/'bad_link').symlink_to(f.opaque)
                with self.assertRaisesRegex(ValueError,'symlink'):g.inventory(f.source)
                with self.assertRaisesRegex(ValueError,'external archive-control'):g.external_output(f.control/'..'/'source'/'bad_plan.json')

    def test_gate_reservation_cannot_be_consumed_twice_with_new_output_name(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            f=Fixture(d)
            with f.patches():
                f.prepare()
                with self.assertRaises(FileExistsError):g.prepare(f.gate_plan['path'],f.gate_plan['sha256'],
                    'fresh-gate-reservation',f.control/'other_install_plan.json')

    def test_original_external_input_mutation_blocks_install(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            f=Fixture(d)
            with f.patches():
                pp=f.prepare();Path(f.scope['original_input_pins'][0]['path']).write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'Pinned source bytes'):g.execute(pp['path'],pp['sha256'])
                self.assertFalse(f.dest.exists())

    def test_generated_metadata_is_inside_twenty_gib_limit(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            f=Fixture(d)
            with f.patches():
                small=g.inventory(f.source)['source_bytes']+100
                with patch.object(g,'MAX_BYTES',small):
                    pp=f.prepare()
                    with self.assertRaisesRegex(ValueError,'generated metadata'):g.execute(pp['path'],pp['sha256'])
                self.assertFalse(f.dest.exists())

    def test_failed_partial_copy_is_preserved_without_success_receipt(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            f=Fixture(d)
            with f.patches():
                pp=f.prepare();original=g.copy_new;calls=[]
                def fail_second(*args):
                    calls.append(args)
                    if len(calls)==2:raise OSError('fabricated disk failure')
                    return original(*args)
                with patch.object(g,'copy_new',side_effect=fail_second),self.assertRaises(OSError):g.execute(pp['path'],pp['sha256'])
                self.assertTrue(f.dest.exists());self.assertFalse((f.dest/'archive_receipt.json').exists())
                with self.assertRaisesRegex(ValueError,'existing archive'):g.execute(pp['path'],pp['sha256'])

    def test_optional_public_doc_is_separate_explicit_and_new_only(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            f=Fixture(d)
            with f.patches():
                pp=f.prepare();g.execute(pp['path'],pp['sha256'])
                target=f.repo/'docs/CORRECTED_REFIT_20261021.md'
                item=g.publish_doc(pp['path'],pp['sha256'],'README.md',target)
                self.assertEqual(Path(item['path']).read_bytes(),f.doc.read_bytes())
                with self.assertRaisesRegex(ValueError,'fresh repository'):g.publish_doc(pp['path'],pp['sha256'],'README.md',target)
                with self.assertRaisesRegex(ValueError,'Document not'):g.publish_doc(pp['path'],pp['sha256'],'run/models/opaque_model.bin',f.repo/'docs/BINARY.md')

    def test_optional_doc_rejects_symlinked_docs_directory(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            f=Fixture(d)
            with f.patches():
                pp=f.prepare();g.execute(pp['path'],pp['sha256'])
                outside=f.root/'outside';outside.mkdir();(f.repo/'docs').symlink_to(outside,target_is_directory=True)
                with self.assertRaisesRegex(ValueError,'Symlinked document'):g.publish_doc(pp['path'],pp['sha256'],'README.md',f.repo/'docs/ESCAPE.md')
                self.assertFalse((outside/'ESCAPE.md').exists())


if __name__=='__main__':unittest.main()
