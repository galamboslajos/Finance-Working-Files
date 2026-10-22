from pathlib import Path
import importlib.util
import unittest

import pyarrow as pa

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('unstopped_verifier_v3',ROOT/'scripts/verify_unstopped_target_comparison_v3.py')
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)


class ExactArrowIdentityTests(unittest.TestCase):
    def row(self,native=None,eligible=False):
        return {'source_security_id':'original','signal_date':'2020-04-30',
                'intrinio_security_id':native,'prediction_eligible':eligible}

    def metadata(self,rows):
        return v.selection_identity_metadata(pa.Table.from_pylist(rows),{('original','2020-04-30')})

    def test_optional_arrow_null_is_preserved_without_numeric_nan_equality(self):
        row=self.row();table=pa.Table.from_pylist([row],schema=pa.schema([
            ('source_security_id',pa.string()),('signal_date',pa.string()),('intrinio_security_id',pa.string()),
            ('prediction_eligible',pa.bool_())]))
        saved=table.to_pandas().to_dict('records')[0]
        saved['intrinio_security_id']=float('nan')
        safe=v.selection_identity_metadata(table,{('original','2020-04-30')})[('original','2020-04-30')]
        self.assertIsNone(safe['intrinio_security_id'])
        v.require_original_identity(safe,row,row)
        with self.assertRaises(ValueError):v.require_original_identity(saved,row,row)

    def test_changed_nonnull_route_and_one_sided_null_are_rejected(self):
        row=self.row('native',True);safe=self.metadata([row])[('original','2020-04-30')]
        for other in [self.row('wrong',True),self.row(None,True)]:
            with self.assertRaises(ValueError):v.require_original_identity(safe,row,other)

    def test_eligible_missing_id_is_rejected_before_comparison(self):
        with self.assertRaisesRegex(ValueError,'no native route'):self.metadata([self.row(None,True)])

    def test_duplicate_or_missing_original_key_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'Duplicate'):self.metadata([self.row(),self.row()])
        table=pa.Table.from_pylist([self.row()])
        with self.assertRaisesRegex(ValueError,'population'):v.selection_identity_metadata(table,{('other','2020-04-30')})

    def test_invalid_numerical_native_id_and_changed_admission_still_fail(self):
        with self.assertRaisesRegex(ValueError,'Invalid native'):self.metadata([self.row(float('nan'))])
        safe=self.metadata([self.row('native',False)])[('original','2020-04-30')]
        with self.assertRaises(ValueError):v.require_original_identity(safe,self.row('native',True),self.row('native',True))


def load_tests(loader,tests,pattern):
    import test_verify_unstopped_target_comparison as old
    import test_verify_unstopped_target_comparison_v2 as digest
    old.v=v;digest.v2=v
    tests.addTests(loader.loadTestsFromModule(old))
    tests.addTests(loader.loadTestsFromTestCase(digest.ExactDigestContracts))
    return tests


if __name__=='__main__':unittest.main()
