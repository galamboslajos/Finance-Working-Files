"""Focused dictionary-contract regressions; no actual research data or fits."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

BASE=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('independent_models_v2',BASE/'scripts/verify_walkforward_models_v2.py')
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n');return path
def pin(base,path):return {'path':str(path.relative_to(base)),'size_bytes':path.stat().st_size,'sha256':v.digest(path)}


def fixture(root):
    run=root/'outputs/walkforward/synthetic';run.mkdir(parents=True)
    dictionary=[{'name':name,'model_eligible':True} for name in [*v.PAPER15,*(f'extra_{index}' for index in range(700))]]
    dictionary.extend({'name':f'diagnostic_{index}','model_eligible':False} for index in range(13))
    canonical=v.canonical(dictionary);reference=root/'outputs/context_sensitivity/synthetic'
    path=save(reference/'feature_dictionary.json',dictionary)
    source_identity={'active_reference_contract':{'feature_dictionary_sha256':canonical}}
    identity_path=save(reference/'build_identity.json',source_identity);source_pin=v.canonical(source_identity)
    audit_path=save(reference/'audit.json',{'status':'COMPLETE','context_policy':'available_cohort_80pct',
        'build_identity_sha256':source_pin,'reports':{'build_identity.json':pin(reference,identity_path),
            'feature_dictionary.json':pin(reference,path)}})
    entry={'directory':str(reference.relative_to(root)),'audit_sha256':v.digest(audit_path),
        'build_identity_sha256':source_pin,'context_policy':'available_cohort_80pct'}
    identity={'feature_dictionary_sha256':canonical,'years':{str(year):{'source_provenance':{'reference_primary':dict(entry)}} for year in v.YEARS}}
    annuals={};schema=pa.schema([(name,pa.string()) for name in v.KEYS]+[(spec['name'],pa.float32()) for spec in dictionary])
    for year in range(2010,2024):
        directory=run/'years'/str(year);directory.mkdir(parents=True)
        file=directory/'features.parquet';pq.write_table(pa.Table.from_batches([],schema=schema),file)
        audit={'feature_provenance':{'feature_dictionary_sha256':canonical},'artifacts':{'features':pin(directory,file)}}
        save(directory/'audit.json',audit);annuals[year]=(directory,audit)
    return run,identity,{},dictionary,reference,annuals


class DictionaryContractV2Tests(unittest.TestCase):
    def test_missing_run_root_dictionary_uses_exact_frozen_reference_no_numeric_read(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();run,identity,root_audit,dictionary,reference,annuals=fixture(root)
            self.assertFalse((run/'feature_dictionary.json').exists())
            with patch.object(v,'annual_directory',side_effect=lambda root,run,identity,audit,year:annuals[year]), \
                    patch.object(v.pq,'read_table',side_effect=AssertionError('Dictionary resolver decoded values')):
                actual,proof=v.resolve_feature_dictionary(root,run,identity,root_audit)
            self.assertEqual(actual,dictionary);self.assertEqual(proof['registered_features'],728)
            self.assertEqual(proof['model_eligible_features'],715);self.assertEqual(len(proof['annual_schema_checks']),14)
            self.assertFalse(proof['feature_values_decoded']);self.assertFalse(proof['production_worker_imports'])
            self.assertEqual(proof['dictionary_path'],str((reference/'feature_dictionary.json').relative_to(root)))

    def test_unpinned_run_root_dictionary_is_ignored(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();run,identity,audit,dictionary,reference,annuals=fixture(root)
            save(run/'feature_dictionary.json',{'WRONG_UNPINNED_ROOT_FILE':True})
            with patch.object(v,'annual_directory',side_effect=lambda root,run,identity,audit,year:annuals[year]):
                actual,proof=v.resolve_feature_dictionary(root,run,identity,audit)
            self.assertEqual(actual,dictionary)

    def test_primary_reference_raw_pin_and_dictionary_canonical_tampering_fail(self):
        for kind in ('reference_audit','dictionary_raw','dictionary_canonical','redirect'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary).resolve();run,identity,audit,dictionary,reference,annuals=fixture(root)
                file=reference/'feature_dictionary.json';audit_file=reference/'audit.json'
                if kind=='reference_audit':audit_file.write_text(audit_file.read_text()+'\n')
                elif kind=='dictionary_raw':file.write_text(file.read_text()+'\n')
                else:
                    state=json.loads(audit_file.read_text())
                    if kind=='dictionary_canonical':
                        dictionary[0]['name']='WRONG';save(file,dictionary)
                        state['reports']['feature_dictionary.json']=pin(reference,file)
                    else:
                        other=save(reference/'wrong.json',dictionary)
                        state['reports']['feature_dictionary.json']=pin(reference,other)
                    save(audit_file,state)
                    for year in v.YEARS:identity['years'][str(year)]['source_provenance']['reference_primary']['audit_sha256']=v.digest(audit_file)
                with patch.object(v,'annual_directory',side_effect=lambda root,run,identity,audit,year:annuals[year]):
                    with self.assertRaises(AssertionError):v.resolve_feature_dictionary(root,run,identity,audit)

    def test_cross_year_reference_provenance_and_schema_fail_closed(self):
        for kind in ('reference','annual_hash','schema'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary).resolve();run,identity,audit,dictionary,reference,annuals=fixture(root)
                if kind=='reference':identity['years']['2021']['source_provenance']['reference_primary']['directory']='wrong'
                elif kind=='annual_hash':annuals[2014][1]['feature_provenance']['feature_dictionary_sha256']='sha256:'+'0'*64
                else:
                    directory,annual=annuals[2014];file=directory/'features.parquet'
                    pq.write_table(pa.table({'source_security_id':pa.array([],pa.string()),'signal_date':pa.array([],pa.string())}),file)
                    annual['artifacts']['features']=pin(directory,file)
                with patch.object(v,'annual_directory',side_effect=lambda root,run,identity,audit,year:annuals[year]):
                    with self.assertRaises(AssertionError):v.resolve_feature_dictionary(root,run,identity,audit)

    def test_original_training_statistics_and_no_fit_guards_remain(self):
        # Reuse the preserved V1 synthetic numeric tests against the V2 module;
        # the saved-data end-to-end fixture's invented root file is excluded.
        source=BASE/'v1_preserved/tests/test_verify_walkforward_models.py'
        if not source.exists():source=BASE/'tests/test_verify_walkforward_models.py'
        spec=importlib.util.spec_from_file_location('_v1_numeric_regressions',source)
        previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
        previous.verifier=v
        for name in ('test_direct_training_statistics_and_matrix_flags','test_new_test_value_cannot_change_training_median',
                'test_missing_flag_or_schema_tamper_rejected','test_future_window_refused_before_any_parquet_read',
                'test_resource_budget_and_no_fit_authority','test_cli_receipt_writes_once_in_separate_verification_directory'):
            with self.subTest(original_regression=name):previous.IndependentVerifierTests(name).debug()


if __name__=='__main__':unittest.main()
