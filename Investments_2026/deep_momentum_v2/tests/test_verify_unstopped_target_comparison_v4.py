from pathlib import Path
import importlib.util
import unittest
import datetime as dt

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('unstopped_verifier_v4',ROOT/'scripts/verify_unstopped_target_comparison_v4.py')
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)

class StrictPreflightDates(unittest.TestCase):
    def row(self,date='2020-02-03'):
        return {'date':date,'source_security_id':'original','reason':'missing','value':1.23456789012345}
    def test_equivalent_midnight_dates_have_exact_same_record(self):
        reference=v.preflight_record_key(self.row())
        for value in [dt.date(2020,2,3),dt.datetime(2020,2,3)]:
            self.assertEqual(reference,v.preflight_record_key(self.row(value)))
    def test_nonmidnight_timezone_missing_invalid_and_protected_dates_fail(self):
        for value in [dt.datetime(2020,2,3,0,0,0,1),dt.datetime(2020,2,3,tzinfo=dt.timezone.utc),None,'2020-2-3','2024-01-01']:
            with self.assertRaises((ValueError,TypeError)):v.preflight_record_key(self.row(value))
    def test_changed_date_fields_values_and_multiplicity_still_differ(self):
        reference=v.preflight_record_key(self.row())
        for change in [{'date':'2020-02-04'},{'source_security_id':'wrong'},{'value':1.23456789012346},{'extra':False}]:
            self.assertNotEqual(reference,v.preflight_record_key(self.row()|change))
        self.assertNotEqual(sorted([reference]),sorted([reference,reference]))
    def test_nan_and_unknown_datetime_fields_not_silently_coerced(self):
        for change in [{'value':float('nan')},{'other_date':dt.datetime(2020,2,3)}]:
            with self.assertRaises((ValueError,TypeError)):v.preflight_record_key(self.row()|change)

def load_tests(loader,tests,pattern):
    import test_verify_unstopped_target_comparison_v3 as previous
    previous.v=v
    tests.addTests(loader.loadTestsFromModule(previous))
    return tests

if __name__=='__main__':unittest.main()
