import unittest

import numpy as np
import pandas as pd

from evaluate_walk_forward import paper_proxy_returns, performance


class WFOEvaluationTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({'date':pd.to_datetime(['2018-12-28','2018-12-28','2018-12-31','2018-12-31']),
                             'label_date':pd.to_datetime(['2018-12-31','2018-12-31','2019-01-02','2019-01-02']),
                             'forward_return':[.01,-.005,.02,-.01], 'daily_volatility':[.02]*4})

    def test_missing_label_keeps_nodes_and_withholds_portfolio_date(self):
        frame=self.frame();frame.loc[1,'forward_return']=np.nan
        result=paper_proxy_returns(frame,np.ones(4))
        self.assertEqual(result.nodes.tolist(),[2,2])
        self.assertEqual(result.available_labels.tolist(),[1,2])
        self.assertTrue(np.isnan(result['return'].iloc[0]))
        self.assertFalse(result.complete.iloc[0])

    def test_future_label_values_cannot_change_validation_returns(self):
        frame=self.frame()
        expected=paper_proxy_returns(frame,np.ones(4),cutoff='2018-12-31')
        frame.loc[2:,'forward_return']=[100.,-1000.]
        changed=paper_proxy_returns(frame,np.ones(4),cutoff='2018-12-31')
        pd.testing.assert_frame_equal(expected,changed)
        self.assertFalse(expected.complete.iloc[-1])

    def test_asset_risk_scaling_and_all_node_denominator(self):
        result=paper_proxy_returns(self.frame(),np.array([1.,-1.,1.,-1.]))
        expected=.15*(.01+.005)/(2*np.sqrt(252)*.02)
        self.assertAlmostEqual(result['return'].iloc[0],expected,places=14)

    def test_gaps_prevent_continuous_wealth_metrics(self):
        stats=performance(np.r_[np.linspace(-.01,.02,40),np.nan])
        self.assertEqual(stats['missing_dates'],1)
        self.assertIsNone(stats['cagr'])
        self.assertIsNone(stats['maximum_drawdown'])
        self.assertIsNotNone(stats['sharpe_complete_dates'])
        full=performance(np.linspace(-.01,.02,40))
        self.assertIsNotNone(full['cagr'])
        self.assertLessEqual(full['maximum_drawdown'],0)


if __name__=='__main__':
    unittest.main()
