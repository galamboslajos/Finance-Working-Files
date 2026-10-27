import unittest
import numpy as np
import pandas as pd
from dmv2.extension_selection_recovery import assert_old_selection_equal

class RecoveryTests(unittest.TestCase):
    def test_side_representation_only_is_accepted(self):
        a=pd.DataFrame(dict(side=pd.Series(['long','short'],dtype=object),RET=[.01,-.02],intended_weight=[.05,-.05]))
        b=a.copy();b['side']=b.side.astype('str')
        assert_old_selection_equal(a,b)
    def test_one_ulp_return_difference_is_not_relaxed(self):
        a=pd.DataFrame(dict(side=['long'],RET=[.01],intended_weight=[.05]));b=a.copy();b.loc[0,'RET']=np.nextafter(.01,1.)
        with self.assertRaises(AssertionError):assert_old_selection_equal(a,b)
    def test_weight_or_side_change_is_rejected(self):
        a=pd.DataFrame(dict(side=['long'],RET=[.01],intended_weight=[.05]))
        for column,value in [('side','short'),('intended_weight',.06)]:
            b=a.copy();b.loc[0,column]=value
            with self.assertRaises(AssertionError):assert_old_selection_equal(a,b)
