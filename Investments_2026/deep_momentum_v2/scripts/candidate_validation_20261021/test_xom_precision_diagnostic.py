from pathlib import Path
import importlib.util
import unittest
p=Path(__file__).with_name('xom_precision_diagnostic.py')
s=importlib.util.spec_from_file_location('precision',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

class PrecisionTests(unittest.TestCase):
    def test_exact_common_factor(self):
        r=m.common_interval([10.,12.,9.,11.],[5.,6.,4.5,5.5],raw_half_width=0.,adjusted_half_width=.005)
        self.assertTrue(r['compatible']);self.assertLessEqual(r['factor_lower'],.5);self.assertGreaterEqual(r['factor_upper'],.5)
    def test_rounding_compatible(self):
        raw=[33.13,33.27,32.74,33.01];adj=[round(x*.713421,2) for x in raw]
        self.assertTrue(m.common_interval(raw,adj,raw_half_width=0.,adjusted_half_width=.005)['compatible'])
    def test_inconsistent_price_not_excused(self):
        self.assertFalse(m.common_interval([10.,12.,9.,11.],[5.,6.,4.5,5.7],raw_half_width=.005,adjusted_half_width=.005)['compatible'])
    def test_grid_is_observed_property(self):
        self.assertTrue(m.on_cent_grid([33.13,5.5]));self.assertFalse(m.on_cent_grid([33.135]))
    def test_invalid_inputs_fail(self):
        with self.assertRaises(AssertionError):m.common_interval([0.,2.,3.,4.],[1.,2.,3.,4.],raw_half_width=.005,adjusted_half_width=.005)

if __name__=='__main__':unittest.main()
