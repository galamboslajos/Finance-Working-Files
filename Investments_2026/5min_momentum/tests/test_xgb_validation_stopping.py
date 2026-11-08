"""Prove numerical parity with native early stopping, and absence of a budget."""
import sys
from pathlib import Path
import unittest
import numpy as np
import xgboost as xgb

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from xgb_validation_stopping import train_until_validation_stops


class ValidationStoppingTests(unittest.TestCase):
    def test_native_fit_probabilities_and_stopping_iterations_match(self):
        rng=np.random.default_rng(41)
        x=rng.normal(size=(1000,8)).astype(np.float32)
        y=rng.integers(0,10,1000)
        dtrain=xgb.DMatrix(x[:800],label=y[:800],nthread=1)
        dvalid=xgb.DMatrix(x[800:],label=y[800:],nthread=1)
        params=dict(objective="multi:softprob",num_class=10,eval_metric="mlogloss",
            max_depth=3,eta=.05,min_child_weight=20,subsample=.8,max_bin=64,
            colsample_bytree=1.,reg_lambda=1.,seed=20261105,nthread=1,tree_method="hist")
        reference_history={}
        reference=xgb.train(params,dtrain,num_boost_round=1000,
            evals=[(dvalid,"random20")],early_stopping_rounds=30,
            evals_result=reference_history,verbose_eval=False)
        observed,history=train_until_validation_stops(params,dtrain,dvalid,30)
        self.assertEqual(reference.best_iteration,observed.best_iteration)
        self.assertEqual(reference_history,history)
        best=observed.best_iteration+1
        np.testing.assert_array_equal(reference.predict(dvalid,iteration_range=(0,best)),
                                      observed.predict(dvalid,iteration_range=(0,best)))

    def test_resource_guard_interrupts_without_an_iteration_budget(self):
        data=xgb.DMatrix(np.arange(100,dtype=np.float32).reshape(50,2),label=np.arange(50)%10,nthread=1)
        seen=[]
        def guard(iteration):
            seen.append(iteration)
            if iteration==3:
                raise InterruptedError("User/resource interruption")
        with self.assertRaises(InterruptedError):
            train_until_validation_stops(dict(objective="multi:softprob",num_class=10,
                eval_metric="mlogloss",nthread=1),data,data,patience=30,guard=guard)
        self.assertEqual(seen,[0,1,2,3])

    def test_improving_fit_can_pass_round_400(self):
        x=np.linspace(-3,3,300,dtype=np.float32).reshape(-1,1)
        y=np.minimum(9,np.arange(300)//30)
        data=xgb.DMatrix(x,label=y,nthread=1)
        seen=[]
        def guard(iteration):
            seen.append(iteration)
            if iteration==405:
                raise InterruptedError("Verified passage beyond the former cap")
        with self.assertRaises(InterruptedError):
            train_until_validation_stops(dict(objective="multi:softprob",num_class=10,
                eval_metric="mlogloss",eta=.001,max_depth=1,nthread=1),data,data,
                patience=30,guard=guard)
        self.assertEqual(max(seen),405)


if __name__=="__main__":
    unittest.main()
