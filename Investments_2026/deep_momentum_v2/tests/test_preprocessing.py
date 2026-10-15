"""Training-only preprocessing: fixtures contain no research observations."""
import unittest

import numpy as np
import pandas as pd

from dmv2.features import TrainingSlice
from dmv2.paper_features import PAPER_FEATURE_NAMES
from dmv2.preprocessing import fit_preprocessor
from dmv2.universe import DataContractError

EXTRA = 'px_simple_return_1s'
CONSTANT = 'px_log_return_1s'
ALL_MISSING = 'px_simple_return_5s'


def training():
    index = pd.MultiIndex.from_tuples([(sid,'2009-11-30') for sid in 'abcd'], names=['source_security_id','signal_date'])
    X = pd.DataFrame({name: np.arange(4)*.1 + j*.01 for j,name in enumerate(PAPER_FEATURE_NAMES)}, index=index)
    X[EXTRA] = [1,np.nan,3,100]
    X[CONSTANT] = [7,7,7,1000]
    X[ALL_MISSING] = [np.nan,np.nan,np.nan,1000]
    y = pd.Series([.1,.2,.3,np.nan], index=index)
    ledger = pd.DataFrame({'outcome_start':'2009-12-01','outcome_end':'2009-12-31',
                           'target_available':[True,True,True,False]}, index=index)
    return TrainingSlice(X,y,ledger,{'train_start':'2000-01-01','train_end':'2009-12-31',
                                    'label_observed_through':'2009-12-31'})


def fitmask(t): return pd.Series([True,True,True,False], index=t.X.index)


class PreprocessingTests(unittest.TestCase):
    def test_training_medians_schema_drop_reasons_and_stable_missing_indicators(self):
        t = training()
        fitted = fit_preprocessor(t,fit_mask=fitmask(t))
        self.assertEqual(fitted.manifest['medians'][EXTRA], 2)
        self.assertEqual(fitted.manifest['dropped_columns'][CONSTANT], 'constant_finite_values_in_training_partition')
        self.assertEqual(fitted.manifest['dropped_columns'][ALL_MISSING], 'all_missing_in_training_partition')
        self.assertIn('zMOM1__missing',fitted.feature_names)
        result = fitted.transform(t.X, prediction_mask=pd.Series(True,index=t.X.index))
        self.assertEqual(result.X.iloc[1][EXTRA],2)
        self.assertEqual(result.X.iloc[1][EXTRA+'__missing'],1)
        self.assertEqual(result.X.iloc[3][EXTRA],100)
        self.assertTrue(result.ledger.transformed.all())  # Missing y never selects prediction rows.
        self.assertFalse(result.provenance['targets_consumed'])
        self.assertEqual(fitted.manifest['fit_rows'],3)

    def test_heldout_extreme_values_cannot_change_fitted_median_or_schema(self):
        t = training()
        first = fit_preprocessor(t, fit_mask=fitmask(t))
        t.X.loc[t.X.index[-1],EXTRA] = 1e99
        t.X.loc[t.X.index[-1],ALL_MISSING] = -1e99
        second = fit_preprocessor(t,fit_mask=fitmask(t))
        self.assertEqual(first.fingerprint,second.fingerprint)
        detached = first.manifest
        detached['medians'][EXTRA] = -1000
        self.assertEqual(first.manifest['medians'][EXTRA],2)
        before = first.fingerprint
        first.transform(t.X,prediction_mask=pd.Series(True,index=t.X.index))
        self.assertEqual(first.fingerprint,before)

    def test_no_baseline_imputation_and_original_rows_retained(self):
        t = training()
        fitted = fit_preprocessor(t,fit_mask=fitmask(t))
        X = t.X.copy()
        X.loc[X.index[-1],'zMOM1'] = np.inf
        mask = pd.Series([True,False,True,True],index=X.index)
        result = fitted.transform(X,prediction_mask=mask)
        self.assertEqual(len(result.X),4)
        self.assertTrue(result.X.iloc[1].isna().all())
        self.assertTrue(result.X.iloc[-1].isna().all())
        self.assertEqual(result.ledger.iloc[-1].preprocessing_status,'incomplete_paper_baseline')
        self.assertEqual(result.ledger.iloc[1].preprocessing_status,'not_requested')

    def test_fit_mask_must_be_explicit_boolean_and_exactly_aligned(self):
        t = training()
        for mask in (None,[True]*4,pd.Series(True,index=t.X.index[::-1]),
                     pd.Series([1,1,1,0],index=t.X.index),pd.Series([True,True,None,False],index=t.X.index)):
            with self.subTest(mask=str(type(mask))),self.assertRaises(DataContractError):
                fit_preprocessor(t,fit_mask=mask)

    def test_fit_rejects_selected_missing_baseline_label_or_target_unavailable(self):
        for change in ('baseline','label','available','late','wrong_month'):
            t = training()
            if change=='baseline': t.X.loc[t.X.index[0],'zMOM1']=np.nan
            elif change=='label': t.y.iloc[0]=np.inf
            elif change=='available': t.ledger.iloc[0,t.ledger.columns.get_loc('target_available')]=False
            elif change=='late': t.ledger.loc[t.X.index[0],'outcome_end']='2010-01-29'
            else: t.ledger.loc[t.X.index[0],'outcome_start']='2009-11-01'
            with self.subTest(change=change),self.assertRaises(DataContractError):
                fit_preprocessor(t,fit_mask=fitmask(t))

    def test_schema_target_fields_holdout_rows_and_missing_baseline_rejected(self):
        t = training()
        fitted = fit_preprocessor(t,fit_mask=fitmask(t))
        for change in ('extra','reorder','holdout','baseline'):
            X=t.X.copy()
            if change=='extra': X['outcome_end']=20091231
            elif change=='reorder': X=X[list(X)[::-1]]
            elif change=='holdout': X.index=pd.MultiIndex.from_tuples([(sid,'2024-01-31') for sid in 'abcd'],names=X.index.names)
            else: X=X.drop(columns=['zMOM1'])
            with self.subTest(change=change),self.assertRaises(DataContractError):
                fitted.transform(X,prediction_mask=pd.Series(True,index=X.index))

    def test_numerical_text_in_predictors_or_labels_is_rejected(self):
        for location in ('X','y'):
            t=training()
            if location=='X':
                X=t.X.astype({EXTRA:object})
                X.loc[X.index[0],EXTRA]='1.0'
                t=TrainingSlice(X,t.y,t.ledger,t.provenance)
            else:
                y=t.y.astype(object)
                y.iloc[0]='0.1'
                t=TrainingSlice(t.X,y,t.ledger,t.provenance)
            with self.subTest(location=location),self.assertRaises(DataContractError):
                fit_preprocessor(t,fit_mask=fitmask(t))

    def test_repeat_fit_on_inner_training_partition_drops_features_using_only_inner_rows(self):
        t = training()
        outer = fit_preprocessor(t,fit_mask=fitmask(t))
        inner_mask=pd.Series([True,True,False,False],index=t.X.index)
        inner = fit_preprocessor(t,fit_mask=inner_mask)
        self.assertIn(EXTRA,outer.manifest['retained_columns'])
        self.assertNotIn(EXTRA,inner.manifest['retained_columns'])
        self.assertNotEqual(inner.fingerprint,outer.fingerprint)


if __name__=='__main__':unittest.main()
