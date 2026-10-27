import unittest
import pandas as pd
from dmv2.fixed_forecast_reporting import check_monthly_curve


class CurveReportingTests(unittest.TestCase):
    def test_altered_monthly_wealth_or_return_rejected(self):
        d=pd.DataFrame(dict(date=pd.bdate_range('2020-02-03',periods=985),mark='close',status='ok',rebalance=False,nav=1.,nav_before_cost=1.))
        m=d.groupby(d.date.dt.strftime('%Y-%m'),sort=True).tail(1)[['date']].copy()
        m['wealth']=1.;m['outcome_month']=m.date.dt.strftime('%Y-%m');m['return']=0.
        check_monthly_curve(m,d,'next_open')
        for col in ('wealth','return'):
            changed=m.copy();changed.loc[changed.index[0],col]+=.001
            with self.assertRaises(AssertionError):check_monthly_curve(changed,d,'next_open')
        with self.assertRaises(ValueError):check_monthly_curve(m,d.iloc[:-1],'next_open')


if __name__=='__main__':unittest.main()
