"""Timing, true-class payoff weighting, concentrated portfolios and missing P&L."""
from pathlib import Path
import sys
import tempfile
import unittest

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from build_momentum_features import MODEL_FEATURES
from ret_backtest import (build_labels,class_payoffs,folds,
                               performance,portfolio_days,read_fold)


class WfoTests(unittest.TestCase):
    def test_rolling_calendar_windows_and_no_outer_overlap(self):
        plan=folds()
        self.assertEqual(len(plan),40)
        self.assertEqual(plan[0]['train_start'],'2016-01-01')
        self.assertEqual(plan[0]['test_start'],'2017-01-01')
        self.assertEqual(plan[1]['train_start'],'2016-04-01')
        self.assertEqual(plan[-1]['test_end'],'2026-10-31')
        self.assertEqual(plan[-1]['planned_test_end'],'2027-01-01')
        self.assertTrue(plan[-1]['partial_block'])
        self.assertEqual(sum(not p['partial_block'] for p in plan),39)
        for i,f in enumerate(plan):
            start=pd.Timestamp(f['test_start'])
            self.assertEqual(start-pd.DateOffset(months=12),pd.Timestamp(f['train_start']))
            self.assertEqual(start+pd.DateOffset(months=3),pd.Timestamp(f['planned_test_end']))
            if not f['partial_block']:
                self.assertEqual(f['test_end'],f['planned_test_end'])
            if i:
                self.assertEqual(plan[i-1]['test_end'],f['test_start'])

    def test_exact_ties_and_small_cohorts_preserve_class_semantics(self):
        rows=[]
        for d,returns in enumerate(([.1]*10+[-.1]*10,[0.0]*20,list(range(9)))):
            for i,r in enumerate(returns):
                rows.append(dict(security_id=str(i),signal_date=pd.Timestamp('2021-01-01')+
                    pd.Timedelta(days=d),trade_date=pd.Timestamp('2021-01-02')+
                    pd.Timedelta(days=d),future_oc_return=r,outcome_status='ok'))
        con=duckdb.connect()
        con.register('examples',pd.DataFrame(rows).sample(frac=1,random_state=8))
        build_labels(con)
        frame=con.execute('SELECT * FROM labels').df()
        first=frame[frame.signal_date.eq(pd.Timestamp('2021-01-01'))]
        self.assertTrue(first[first.security_id.astype(int).lt(10)].realized_class.eq(3).all())
        self.assertTrue(first[first.security_id.astype(int).ge(10)].realized_class.eq(8).all())
        self.assertTrue(frame[frame.signal_date.eq(pd.Timestamp('2021-01-02'))].realized_class.eq(6).all())
        self.assertTrue(frame[frame.signal_date.eq(pd.Timestamp('2021-01-03'))].realized_class.isna().all())
        con.close()

    def test_payoffs_average_days_not_pooled_stock_rows(self):
        rows=[]
        for d in range(101):
            for k in range(1,11):
                for i in range(100 if d==0 else 1):
                    rows.append(dict(trade_date=pd.Timestamp('2020-01-01')+pd.Timedelta(days=d),
                        realized_class=k,future_oc_return=.2 if d==0 else 0.0))
        mu,daily,complete=class_payoffs(pd.DataFrame(rows))
        np.testing.assert_allclose(mu,np.full(10,.2/101))
        self.assertEqual(complete.sum(),101)

    def predictions(self):
        return pd.DataFrame(dict(trade_date=[pd.Timestamp('2022-01-03')]*6,
            security_id=[str(i) for i in range(6)],ticker=[f'S{i}' for i in range(6)],
            ret_score=[6,5,4,3,2,1],future_oc_return=[.1,.2,.3,0,-.1,-.2]))

    def test_equal_leg_weights_and_long_short_pnl(self):
        predictions=self.predictions()
        calendar=pd.DataFrame({'trade_date':[pd.Timestamp('2022-01-03')]})
        days,positions=portfolio_days(predictions,calendar,2)
        self.assertAlmostEqual(days.gross_return.iloc[0],.15)
        self.assertAlmostEqual(days.trade_notional.iloc[0],2.0)
        self.assertAlmostEqual(positions.weight.sum(),0)
        self.assertAlmostEqual(positions.weight.abs().sum(),1)
        self.assertEqual(set(positions.security_id),{'0','1','4','5'})
        self.assertTrue(positions.weight.abs().eq(.25).all())

    def test_missing_selected_outcome_is_not_replaced_or_zeroed(self):
        predictions=self.predictions()
        predictions.loc[0,'future_oc_return']=np.nan
        calendar=pd.DataFrame({'trade_date':[pd.Timestamp('2022-01-03')]})
        days,positions=portfolio_days(predictions,calendar,2)
        self.assertTrue(np.isnan(days.gross_return.iloc[0]))
        self.assertEqual(days.missing_positions.iloc[0],1)
        self.assertIn('0',set(positions.security_id))
        self.assertNotIn('2',set(positions.security_id))

    def test_explicit_no_forecast_day_is_cash(self):
        calendar=pd.DataFrame({'trade_date':pd.to_datetime(['2022-01-03','2022-01-04'])})
        days,_=portfolio_days(self.predictions(),calendar,3)
        row=days.iloc[1]
        self.assertEqual(row.pnl_status,'cash_insufficient_forecasts')
        self.assertEqual(row.gross_return,0)
        self.assertEqual(row.selected_positions,0)

    def test_training_cutoff_and_missing_test_labels_do_not_filter_predictions(self):
        records=[]
        for date in pd.bdate_range('2021-01-01','2022-03-31'):
            rec=dict(security_id='1',signal_date=date-pd.Timedelta(days=1),
                trade_date=date,realized_class=1 if date.year==2021 else np.nan,
                future_oc_return=.01)
            rec.update({f:0.1 for f in MODEL_FEATURES})
            records.append(rec)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            pd.DataFrame(records).to_parquet(root/'supervised_panel.parquet',index=False)
            train,test=read_fold(root,folds('2022-01-01','2024-01-01')[0])
            self.assertTrue(train.trade_date.lt(pd.Timestamp('2022-01-01')).all())
            self.assertTrue(test.realized_class.isna().all())
            self.assertEqual(len(test),len(pd.bdate_range('2022-01-01','2022-03-31')))

    def test_metrics_refuse_missing_returns_and_measure_initial_drawdown(self):
        with self.assertRaises(ValueError):
            performance([0.1,np.nan])
        result=performance([-.1,.1])
        self.assertAlmostEqual(result['max_drawdown'],.1)
        self.assertAlmostEqual(result['total_return'],-.01)


if __name__=='__main__':
    unittest.main()
