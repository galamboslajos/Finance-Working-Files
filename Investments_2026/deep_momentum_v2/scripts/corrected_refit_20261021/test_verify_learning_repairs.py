import copy
import unittest
import learning_repairs as production
import verify_learning_repairs as v
from test_learning_repairs import rcm_fixture,disca_fixture


def fixture(reject_rcm=False,reject_disca=False):
    rcm,inv=rcm_fixture();disca,controls=disca_fixture()
    if reject_rcm:rcm[0]['volume']*=1.001
    if reject_disca:controls[0]['adj_volume']*=1.001
    rd,rq=production.rcm_case(rcm,inv);dd,dq=production.disca_case(disca,controls)
    quotes=rq+dq
    manifest={'status':'PASS','verification_status':'VERIFIED_PROVISIONAL','cases':{'RCM':rd,'DISCA':dd},
      'row_count':len(quotes),'columns':list(v.COLS),'append_only':True,'replace_existing_rows':False,
      'canonical_identity_promoted':False,'terminal_target_policy_changed':False}
    return rcm,disca,controls,quotes,manifest,inv


class Independent(unittest.TestCase):
    def test_nonfinancial_missing_action_nan_preserved_exactly(self):
        self.assertTrue(v.same_bar({'factor':float('nan')},{'factor':float('nan')}))
        self.assertFalse(v.same_bar({'factor':float('nan')},{'factor':1.}))
    def test_boolean_control_zero_never_numeric(self):
        left={d:{k:0. for k in v.FIELDS} for d in ['2022-04-01']}
        right=copy.deepcopy(left);left['2022-04-01']['volume']=False
        ok,records=v.controls(left,right,['2022-04-01'])
        self.assertFalse(ok);self.assertIsNone([r for r in records if r['column']=='volume'][0]['relative_residual'])
    def test_both_accepted_independent_composition(self):
        result=v.verify_rows(*fixture());self.assertEqual(result['admitted_rows'],165)
    def test_rcm_reject_is_normal_and_disca_still_accepted(self):
        result=v.verify_rows(*fixture(True,False));self.assertEqual(result['admitted_rows'],5)
    def test_disca_reject_normal(self):
        result=v.verify_rows(*fixture(False,True));self.assertEqual(result['admitted_rows'],160)
    def test_both_rejected_zero_rows(self):
        result=v.verify_rows(*fixture(True,True));self.assertEqual(result['admitted_rows'],0)
    def test_changed_copied_price_fails(self):
        args=list(fixture());args[3][0]['close']+=.01
        with self.assertRaises(ValueError):v.verify_rows(*args)
    def test_invented_factor_fails(self):
        args=list(fixture());args[3][-1]['factor']=1.
        with self.assertRaises(ValueError):v.verify_rows(*args)
    def test_rejected_case_cannot_admit_partial(self):
        args=list(fixture(True,False));args[3].append({**args[0][2],'intrinio_security_id':v.RCM_TARGET})
        with self.assertRaises(ValueError):v.verify_rows(*args)
    def test_fake_quality_status_fails(self):
        args=list(fixture());args[4]['cases']['RCM']['status']='REJECTED_SOURCE_QUALITY'
        with self.assertRaises(ValueError):v.verify_rows(*args)
    def test_changed_control_residual_fails(self):
        args=list(fixture());args[4]['cases']['DISCA']['control_comparisons'][0]['relative_residual']=.1
        with self.assertRaises(ValueError):v.verify_rows(*args)
    def test_metadata_guard_exact_scope(self):
        rcm,disca,controls,quotes,m,inv=fixture()
        metadata={'rcm':[{k:r[k] for k in ('intrinio_security_id','trading_date')} for r in rcm],
            'disca':[{k:r[k] for k in v.META} for r in disca],
            'controls':[{k:r[k] for k in ('intrinio_security_id','trading_date')} for r in controls],
            'quotes':[{k:r[k] for k in ('intrinio_security_id','trading_date')} for r in quotes]}
        v.validate_metadata(metadata,inv)
        metadata['quotes'][0]['intrinio_security_id']='wrong'
        with self.assertRaises(ValueError):v.validate_metadata(metadata,inv)

if __name__=='__main__':unittest.main()
