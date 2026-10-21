"""Pure invented metadata, no source/model/accounting execution."""
from calendar import monthrange
import copy
import unittest
import backtest_integration as b
import backtest_runner as r


def ret_meta():
    signal='2019-12-31';first='2010-01'
    state={'provenance':{'signal_date':signal,'label_observed_through':signal,
        'first_outcome_month':first,'last_outcome_month':'2019-12','vector_available':True}}
    rows=[]
    for i in range(120):
        month=b.shift_month(first,i);year,number=map(int,month.split('-'))
        end=f'{month}-{monthrange(year,number)[1]:02d}'
        for c in range(1,11):rows.append(dict(outcome_month=month,outcome_end=end,class_id=c,
            original_member_count=20,prediction_eligible_count=20,member_count=2,status='available'))
    return rows,state,signal


class MetadataRuntime(unittest.TestCase):
    def test_exact_ret_metadata_grid(self):b.validate_ret_metadata(*ret_meta())
    def test_ret_class_order_change_fails_before_values(self):
        rows,state,signal=ret_meta();rows[0],rows[1]=rows[1],rows[0]
        with self.assertRaises(ValueError):b.validate_ret_metadata(rows,state,signal)
    def test_ret_wrong_endpoint_month(self):
        rows,state,signal=ret_meta();rows[0]['outcome_end']='2010-02-01'
        with self.assertRaises(ValueError):b.validate_ret_metadata(rows,state,signal)
    def test_ret_future_endpoint(self):
        rows,state,signal=ret_meta();rows[-1]['outcome_end']='2024-01-01'
        with self.assertRaises(ValueError):b.validate_ret_metadata(rows,state,signal)
    def test_ret_future_provenance(self):
        rows,state,signal=ret_meta();state['provenance']['label_observed_through']='2020-01-01'
        with self.assertRaises(ValueError):b.validate_ret_metadata(rows,state,signal)
    def test_ret_incomplete_grid_no_shortening(self):
        rows,state,signal=ret_meta()
        with self.assertRaises(ValueError):b.validate_ret_metadata(rows[:-1],state,signal)
    def test_ret_boolean_member_count_rejected(self):
        rows,state,signal=ret_meta();rows[0]['member_count']=True
        with self.assertRaises(ValueError):b.validate_ret_metadata(rows,state,signal)
    def test_only_three_new_phases_and_1GiB_output(self):
        self.assertEqual(r.PHASES,('selection','preflight','replay'))
        self.assertEqual(r.MAX_BACKTEST_OUTPUT,1024**3)
        self.assertEqual(r.BASE.name,'backtest')

if __name__=='__main__':unittest.main()
