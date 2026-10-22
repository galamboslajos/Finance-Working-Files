from pathlib import Path
import importlib.util
import unittest
import pyarrow as pa

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('unstopped_verifier_v5',ROOT/'scripts/verify_unstopped_target_comparison_v5.py')
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)

class ExactAnnualRecords(unittest.TestCase):
    def test_null_column_and_typed_null_column_preserve_same_records(self):
        left=pa.Table.from_pylist([{'year':2020,'initial_nav':None}])
        whole=pa.Table.from_pylist([{'year':2020,'initial_nav':None},{'year':2021,'initial_nav':1.0}])
        self.assertNotEqual(left.to_pandas().initial_nav.dtype,whole.to_pandas().initial_nav.dtype)
        v.require_exact_annual_records(left.to_pylist(),whole.slice(0,1).to_pylist())
    def test_changed_null_numeric_precision_and_scalar_type_are_rejected(self):
        reference=[{'value':1.23456789012345,'missing':None}]
        for row in [{'value':1.23456789012346,'missing':None},{'value':1.23456789012345,'missing':0.0},
                    {'value':1.23456789012345,'missing':float('nan')},{'value':1,'missing':None}]:
            with self.assertRaises(ValueError):v.require_exact_annual_records(reference,[row])
        with self.assertRaises(ValueError):v.require_exact_annual_records([{'value':1}],[{'value':1.0}])
    def test_reordered_dropped_duplicated_rows_or_columns_rejected(self):
        rows=[{'year':2020,'value':None},{'year':2021,'value':2.0}]
        for changed in [list(reversed(rows)),rows[:1],rows+rows[:1],[r|{'extra':None} for r in rows]]:
            with self.assertRaises(ValueError):v.require_exact_annual_records(rows,changed)
    def test_nonfinite_values_rejected_even_when_same(self):
        for value in [float('nan'),float('inf'),-float('inf')]:
            with self.assertRaises(ValueError):v.require_exact_annual_records([{'x':value}],[{'x':value}])

def load_tests(loader,tests,pattern):
    import test_verify_unstopped_target_comparison_v4 as previous
    previous.v=v
    tests.addTests(loader.loadTestsFromModule(previous))
    return tests

if __name__=='__main__':unittest.main()
