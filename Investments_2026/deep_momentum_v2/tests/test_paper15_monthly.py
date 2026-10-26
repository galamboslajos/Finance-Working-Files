"""Invented prices/probabilities only; no empirical fits or private data reads."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from dmv2.paper15_baseline import FEATURES, PROBS, shift
from dmv2.paper15_monthly import monthly_panel, monthly_payoffs, select_monthly, monthly_account, monthly_statistics


class MonthlyRules(unittest.TestCase):
    def source(self):
        rows = [dict(source_security_id=f'S{j:02}', signal_date='2019-12-31', intrinio_security_id=f'N{j:02}',
                     outcome_month='2020-01', outcome_start='2020-01-02', outcome_end='2020-01-31',
                     prediction_eligible=True, **{c:float(j) for c in FEATURES}) for j in range(20)]
        closes = np.array([[100.]*20, [100+j for j in range(20)]])
        return pd.DataFrame(rows), closes, ['2019-12','2020-01'], [f'N{j:02}' for j in range(20)], ['2019-12-31','2020-01-02','2020-01-31']

    def test_exact_monthly_closes_and_end_completion(self):
        args = self.source(); out = monthly_panel(*args)
        np.testing.assert_array_equal(out.label_return, args[1][1]/args[1][0]-1)
        pd.testing.assert_frame_equal(out[list(FEATURES)], args[0][list(FEATURES)])
        self.assertTrue(out.label_available_at.eq('2020-01-31').all())
        self.assertTrue(out.target_entry_date.eq('2019-12-31').all())
        self.assertEqual(out.target_class.iloc[-1],1); self.assertEqual(out.target_class.iloc[0],10)

    def test_missing_future_return_never_changes_eligibility_or_features(self):
        args=list(self.source());args[1][1,0]=np.nan;out=monthly_panel(*args)
        self.assertEqual(len(out),20);self.assertTrue(out.prediction_eligible.all());self.assertTrue(pd.isna(out.label_return.iloc[0]))
        pd.testing.assert_frame_equal(out[list(FEATURES)],args[0][list(FEATURES)])

    def test_protected_outcome_and_stale_endpoint_dates_rejected(self):
        args=list(self.source());args[0].loc[0,'outcome_month']='2024-01'
        with self.assertRaises(ValueError):monthly_panel(*args)
        args=list(self.source());args[0].loc[0,'signal_date']='2019-12-30'
        with self.assertRaises(ValueError):monthly_panel(*args)

    def test_RET_includes_completed_signal_month_not_future(self):
        rows=[dict(outcome_month=shift('2019-12',j),label_available_at=shift('2019-12',j)+'-28',target_class=k,label_return=k/100)
              for j in range(-59,1) for k in range(1,11)]
        panel=pd.DataFrame(rows);mu,grid=monthly_payoffs(panel,'2019-12-31',5)
        np.testing.assert_allclose(mu,np.arange(1,11)/100);self.assertEqual(max(x['outcome_month'] for x in grid),'2019-12')
        future=pd.DataFrame([dict(outcome_month='2020-01',label_available_at='2020-01-31',target_class=1,label_return=1e8)])
        self.assertEqual(mu,monthly_payoffs(pd.concat([panel,future]),'2019-12-31',5)[0])
        self.assertIsNone(monthly_payoffs(panel.iloc[1:],'2019-12-31',5)[0]);self.assertIsNone(monthly_payoffs(panel,'2019-12-31',10)[0])

    def test_signal_selection_retains_january_and_rejects_outcomes(self):
        p=self.source()[0].copy()
        for k,name in enumerate(PROBS):p[name]=[float(j%10==k) for j in range(20)]
        ledger,chosen=select_monthly(p,{'2019-12-31':list(range(10))})
        self.assertEqual(len(chosen),20);self.assertEqual(int((chosen.side=='long').sum()),10)
        self.assertAlmostEqual(chosen.intended_weight.sum(),0);self.assertTrue(ledger.selected.all())
        p['label_return']=0
        with self.assertRaises(ValueError):select_monthly(p,{'2019-12-31':list(range(10))})

    def basket(self):
        selected=pd.DataFrame([dict(source_security_id=sid, signal_date=signal, outcome_month=month,
             intrinio_security_id=sid,side=side,intended_weight=w) for signal,month in [('2019-12-31','2020-01'),('2020-01-31','2020-02')]
             for sid,side,w in [('A','long',.5),('B','short',-.5)]])
        outcome=selected[['source_security_id','signal_date']].copy();outcome['label_return']=[.2,-.1,0.,0.]
        return selected,outcome

    def test_signed_returns_and_drift_adjusted_turnover(self):
        selected,outcome=self.basket();m,pos,fail=monthly_account(selected,outcome,['2020-01','2020-02'],0)
        self.assertTrue(fail.empty);self.assertAlmostEqual(m.nav_end.iloc[0],1.15)
        self.assertAlmostEqual(m.one_way_turnover.iloc[0],1)
        self.assertAlmostEqual(m.one_way_turnover.iloc[1],.15/1.15)
        self.assertAlmostEqual(m.long_contribution.iloc[0],.1);self.assertAlmostEqual(m.short_contribution.iloc[0],.05)
        self.assertAlmostEqual(pos.loc[(pos.outcome_month=='2020-01')&(pos.source_security_id=='B'),'end_notional'].iloc[0],-.45)

    def test_cost_deducted_from_cash_with_pretrade_NAV_targets(self):
        selected,outcome=self.basket();m,_,fail=monthly_account(selected,outcome,['2020-01','2020-02'],10)
        self.assertTrue(fail.empty);self.assertAlmostEqual(m.nav_end.iloc[0],1.149)
        self.assertAlmostEqual(m.trading_cost.iloc[0],.001);self.assertAlmostEqual(m.trading_cost.iloc[1],.00015)
        self.assertAlmostEqual(m.nav_end.iloc[1],1.14885)

    def test_missing_selected_monthly_return_withholds_instead_of_zero_or_drop(self):
        selected,outcome=self.basket();outcome.loc[0,'label_return']=np.nan
        m,p,f=monthly_account(selected,outcome,['2020-01','2020-02'],10)
        self.assertTrue(m.empty);self.assertEqual(f.reason.iloc[0],'missing_selected_monthly_return')
        self.assertEqual(len(selected),4);self.assertTrue(all(v is None for v in monthly_statistics(m,False).values()))

    def test_monthly_risk_statistics_and_initial_NAV_drawdown(self):
        f=pd.DataFrame(dict(net_return=[-.1,.1],nav_end=[.9,.99],one_way_turnover=[1.,2.],trading_cost=[0.,0.]))
        s=monthly_statistics(f,True);self.assertAlmostEqual(s['total_return'],-.01)
        self.assertAlmostEqual(s['monthly_maximum_drawdown'],-.1)
        self.assertAlmostEqual(s['annualized_volatility'],np.std([-.1,.1],ddof=1)*np.sqrt(12))

    def test_independent_rank_oracle_preserves_exact_ties(self):
        path=Path(__file__).parents[1]/'scripts/verify_paper15_monthly.py'
        spec=importlib.util.spec_from_file_location('_monthly_independent_test',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        self.assertTrue((module.rank_oracle([1.]*20)==6).all())
        self.assertTrue(np.isnan(module.rank_oracle([1.,2.])).all())
        expected=module.rank_oracle(np.arange(20,dtype=float));self.assertEqual(expected[0],10);self.assertEqual(expected[-1],1)


if __name__=='__main__':unittest.main()
