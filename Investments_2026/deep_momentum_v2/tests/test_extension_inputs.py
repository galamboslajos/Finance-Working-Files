"""Structural, adversarial tests only; no financial source data or model fit."""
import copy
import unittest
import numpy as np
import pandas as pd
from dmv2.extension_inputs import build_panel, qualify, payoff_grid, scope
from dmv2.paper15_baseline import FEATURES


def fixture():
    days=pd.bdate_range('2000-01-03','2024-01-31').strftime('%Y-%m-%d').tolist()
    months=sorted({d[:7] for d in days});natives=['id'+str(i) for i in range(12)]
    closes=100+np.outer(np.arange(1,len(months)+1),np.linspace(1,3,12))**1.2
    arrays=dict(valid=np.ones((len(days),12),bool),month_closes=closes)
    index=dict(sessions=days,months=months,natives=natives)
    rows=[dict(source_security_id='source'+str(i),signal_date='2023-12-29',intrinio_security_id=n,
        unified_security_id='u'+str(i),identity_status='needs_review',identity_review_required=True,
        resolution_status='resolved',indices=['sp500'],membership_tickers=['T'+str(i)]) for i,n in enumerate(natives)]
    return pd.DataFrame(qualify(rows)),arrays,index


class ExtensionInputsTests(unittest.TestCase):
    def test_forward_prices_change_labels_but_never_eligibility_or_features(self):
        c,a,i=fixture();before,_=build_panel(c,a,i,5)
        changed=copy.deepcopy(a);changed['month_closes'][-1]=np.nan;changed['valid'][-20:]=False
        after,_=build_panel(c,changed,i,5)
        np.testing.assert_array_equal(before.prediction_eligible,after.prediction_eligible)
        np.testing.assert_allclose(before[list(FEATURES)],after[list(FEATURES)],equal_nan=True,rtol=0,atol=0)
        self.assertTrue(after.prediction_eligible.all());self.assertFalse(after.target_available.any())

    def test_past_missingness_only_affects_matching_window(self):
        c,a,i=fixture();a['valid'][i['sessions'].index('2015-01-05'),0]=False
        five,_=build_panel(c,a,i,5);ten,_=build_panel(c,a,i,10)
        self.assertTrue(five.iloc[0].history_complete);self.assertFalse(ten.iloc[0].history_complete)
        self.assertFalse(ten.iloc[0].prediction_eligible)
        self.assertEqual(len(five),len(ten));self.assertEqual(len(ten),12)

    def test_price_collision_blocks_both_original_rows(self):
        c,a,i=fixture();rows=c.to_dict('records');rows[1]['intrinio_security_id']=rows[0]['intrinio_security_id']
        out=qualify(rows);self.assertFalse(out[0]['identity_eligible']);self.assertFalse(out[1]['identity_eligible'])
        self.assertEqual(len(out),12)

    def test_route_interval_is_visible_not_silently_reverted(self):
        rows=[dict(source_security_id='siblis_isin_b284bb7497616f5afba0',signal_date='2024-01-31',
            intrinio_security_id='sec_z5YpGE',unified_security_id='u',identity_status='resolved',
            identity_review_required=False,resolution_status='resolved')]
        out=qualify(rows)[0];self.assertFalse(out['identity_eligible']);self.assertEqual(out['identity_status'],'reviewed_route_interval_ended')

    def test_guard_rejects_unapproved_future_and_missing_flag(self):
        c,a,i=fixture();rows=c.to_dict('records');rows[0]['identity_review_required']=None
        self.assertFalse(qualify(rows)[0]['identity_eligible'])
        rows[0]['signal_date']='2026-04-30'
        with self.assertRaises(ValueError):qualify(rows)

    def test_payoff_gate_never_shortens_and_future_outcomes_ignored(self):
        c,a,i=fixture();panel,_=build_panel(c,a,i,5)
        mu,cells=payoff_grid(panel,'2023-12-29',5)
        self.assertIsNone(mu);self.assertEqual(len(cells),600)
        self.assertFalse(any(x['outcome_month']=='2024-01' for x in cells))

    def test_normalization_shared_cohort_ddof1_and_exact15(self):
        c,a,i=fixture();p,_=build_panel(c,a,i,5)
        np.testing.assert_allclose(p[list(FEATURES)[:5]].mean(),0,atol=1e-13)
        np.testing.assert_allclose(p[list(FEATURES)[:5]].std(ddof=1),1,atol=1e-13)
        self.assertTrue(all(p[name].nunique()==1 for name in FEATURES[5:]))

    def test_input_contract_cannot_enable_fitting_or_source_upgrade(self):
        import json
        from pathlib import Path
        cfg=json.loads((Path(__file__).resolve().parents[1]/'configs/walkforward_extension_inputs_20261027.json').read_text())
        scope(cfg);cfg['maximum_fits']=1
        with self.assertRaises(ValueError):scope(cfg)


if __name__=='__main__':unittest.main()
