import copy
import math
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import verify_models as v

class FakeRF:
    classes_=np.arange(1,11)
    def __init__(self, n_jobs=1): self.n_jobs=n_jobs
    def get_params(self,deep=False):return {'n_jobs':self.n_jobs,'n_estimators':2}
    def fit(self,*args,**kwargs):raise AssertionError('No fit allowed')
    def predict_proba(self,X):return np.full((len(X),10),0.1)

class FakeXGB(FakeRF):
    classes_=np.arange(10)
    def predict(self,X,output_margin=False):
        assert output_margin
        return np.tile(np.arange(10,dtype=np.float32),(len(X),1))

class TestIndependentModels(unittest.TestCase):
    def setUp(self):
        self.names=['p','q','constant','empty','sparse']
        self.paper=['p','q']
        self.raw=np.array([[1,2,4,np.nan,10],[2,3,4,np.nan,np.nan],[9,8,100,99,999]],float)
        self.mask=np.array([True,True,False])
        self.keys=[('a','2010-01-29'),('b','2010-01-29'),('future','2019-12-31')]
    def state(self):
        return v.derive_preprocessor(self.raw,self.names,self.mask,self.keys,self.paper,'2010-01-01','2019-12-31')
    def test_train_only_median_and_drop(self):
        s=self.state()['manifest']
        self.assertEqual(s['medians'],{'p':1.5,'q':2.5})
        self.assertEqual(s['dropped_columns'],{'constant':'constant_finite_values_in_training_partition','empty':'all_missing_in_training_partition','sparse':'constant_finite_values_in_training_partition'})
        self.assertEqual(s['output_columns'],['p','p__missing','q','q__missing'])
    def test_all_missing_predictor_drops_even_if_future_observed(self):
        s=self.state()['manifest'];self.assertNotIn('empty',s['retained_columns'])
    def test_fixed_missing_flags_and_float32(self):
        s=self.state();x=self.raw.copy();x[2,0]=np.inf
        a=v.prepare_matrix(x,self.names,s,np.array([False,False,True]))
        self.assertEqual(a.dtype,np.float32);self.assertEqual(a.tolist(),[[1.5,1,8,0]])
    def test_schema_order_change_rejected(self):
        with self.assertRaises(ValueError):v.prepare_matrix(self.raw,self.names[::-1],self.state(),self.mask)
    def test_state_tampering_rejected(self):
        s=self.state();s['manifest']['medians']['p']=999
        with self.assertRaises(ValueError):v.prepare_matrix(self.raw,self.names,s,self.mask)
    def test_overflow_rejected(self):
        x=self.raw.copy();x[2,0]=1e100
        with self.assertRaises(ValueError):v.prepare_matrix(x,self.names,self.state(),np.array([False,False,True]))
    def test_boolean_and_text_matrix_rejected(self):
        for a in [np.ones((3,5),bool),np.full((3,5),'1')]:
            with self.assertRaises(ValueError):v.numeric_matrix(a)
    def test_missing_training_baseline_rejected(self):
        x=self.raw.copy();x[0,0]=np.nan
        with self.assertRaises(ValueError):v.derive_preprocessor(x,self.names,self.mask,self.keys,self.paper,'2010-01-01','2019-12-31')
    def test_classes_require_explicit_availability(self):
        np.testing.assert_equal(v.canonical_labels([1,None,10],[True,False,True]),[1,np.nan,10])
        with self.assertRaises(ValueError):v.canonical_labels([1,2],[True,False])
        with self.assertRaises(ValueError):v.canonical_labels([True],[True])
    def test_reload_inference_no_fit_and_worker_guard(self):
        X=np.ones((2,2),np.float32)
        p=v.infer_probabilities(FakeRF(),'rf',X,{'n_jobs':1,'n_estimators':2})
        np.testing.assert_array_equal(p,np.full((2,10),.1))
        with self.assertRaises(ValueError):v.infer_probabilities(FakeRF(2),'rf',X,{'n_jobs':1})
    def test_native_xgb_margin_softmax(self):
        p=v.infer_probabilities(FakeXGB(),'xgb',np.ones((1,2),np.float32),{'n_jobs':1})
        expected=np.exp(np.arange(10,dtype=float)-9);expected/=expected.sum()
        np.testing.assert_array_equal(p[0],expected)
    def test_missing_model_class_rejected(self):
        m=FakeRF();m.classes_=np.arange(1,10)
        with self.assertRaises(ValueError):v.infer_probabilities(m,'rf',np.ones((1,2)),{'n_jobs':1})
    def test_metrics_monthly_weighting_not_pooled(self):
        p=np.full((3,10),.1);p[0]=[1]+[0]*9
        s,m=v.independent_metrics(p,[1,2,np.nan],['2020-01','2020-02','2020-02'])
        self.assertAlmostEqual(s['mean_monthly_log_loss'],math.log(10)/2)
        self.assertEqual(s['missing_label_rows'],1)
        self.assertEqual(s['mean_monthly_accuracy'],.5)
    def test_floor_does_not_change_stored_probabilities(self):
        p=np.array([[1]+[0]*9],float);before=p.copy()
        s,_=v.independent_metrics(p,[2],['2020-01'])
        self.assertEqual(s['pooled_log_loss'],-math.log(1e-15));np.testing.assert_array_equal(p,before)

class TestMetadata(unittest.TestCase):
    def fixture(self):
        windows={};fs=[];cs=[];ts=[];orig=[]
        import calendar,datetime
        for m in range(1,13):
            prior=datetime.date(2019 if m==1 else 2020,12 if m==1 else m-1,1)
            signal=prior.replace(day=calendar.monthrange(prior.year,prior.month)[1]).isoformat()
            w={'outcome_month':f'2020-{m:02d}','outcome_start':f'2020-{m:02d}-01','outcome_end':f'2020-{m:02d}-{calendar.monthrange(2020,m)[1]}'}
            windows[signal]=w;k={'source_security_id':'a','signal_date':signal}
            a={**k,'identity_eligible':True,'requested':True,'paper_prediction_eligible':True}
            fs.append(k);orig.append(k);cs.append(a);ts.append({**a,**w,'target_available':True})
        return [fs,cs,ts,orig,windows,2020]
    def test_all_original_keys_guarded(self):
        args=self.fixture();self.assertEqual(len(v.validate_metadata(*args)),12)
        args[3].pop()
        with self.assertRaises(ValueError):v.validate_metadata(*args)
    def test_wrong_future_clock_before_values(self):
        args=self.fixture();args[2][0]['outcome_end']='2024-01-31'
        with self.assertRaises(ValueError):v.validate_metadata(*args)
    def test_unknown_admission_and_target_filter_rejected(self):
        args=self.fixture();args[1][0]['requested']=None
        with self.assertRaises(ValueError):v.validate_metadata(*args)
        args=self.fixture();args[2][0]['paper_prediction_eligible']=False
        with self.assertRaises(ValueError):v.validate_metadata(*args)
    def test_nested_list_root_projection_counts_every_leaf(self):
        import tempfile
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'invented.parquet'
            pq.write_table(pa.table({'indices':[['sp500'],['sp400']],'unrequested':[1.0,2.0]}),p)
            projection=v.footer_projection(v.pin(p),['indices'])
            pf=pq.ParquetFile(p)
            expected=sum(pf.metadata.row_group(0).column(j).total_compressed_size for j in range(pf.metadata.row_group(0).num_columns)
                         if pf.metadata.row_group(0).column(j).path_in_schema.startswith('indices.'))
            self.assertEqual(projection['projected_compressed_bytes'],expected)
            self.assertGreater(expected,0)
    def test_duplicate_key_rejected(self):
        args=self.fixture();args[0].append(args[0][0])
        with self.assertRaises(ValueError):v.validate_metadata(*args)

if __name__=='__main__':unittest.main()
