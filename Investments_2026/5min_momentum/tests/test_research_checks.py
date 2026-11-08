"""Adversarial leakage, timestamp, normalization and cohort guard tests."""
from pathlib import Path
import sys
import unittest
import importlib
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from research_checks import (require_dense_sessions, require_past_training,
    execution_timing_violations, require_execution_information_timing)
from volume_features import make_volume_features, VOLUME_FEATURES
from network_paper_features import individual_features
from build_network_paper_features import PARAMETERS
import ret_backtest as backtest
import train_xgb_ret_ensemble as ensemble


class ResearchChecks(unittest.TestCase):
    def test_missing_day_cannot_be_compressed_into_a_rolling_window(self):
        expected=pd.date_range("2024-01-02",periods=10,freq="B")
        require_dense_sessions(expected,expected)
        for bad in [expected.delete(4),expected[::-1],expected.insert(3,expected[3])]:
            with self.assertRaises(ValueError):
                require_dense_sessions(bad,expected)

    def frame(self,date):
        date=pd.Timestamp(date)
        return pd.DataFrame(dict(security_id=["A"],trade_date=[date],
            signal_date=[date-pd.Timedelta(days=1)],realized_class=[1],future_oc_return=[.01]))

    def test_held_out_outcome_poisoning_cannot_enter_training(self):
        past,future=self.frame("2023-12-29"),self.frame("2024-01-02")
        require_past_training(past,future,"2024-01-01")
        with self.assertRaises(ValueError):
            require_past_training(pd.concat([past,future]),future,"2024-01-01")

    def test_undefined_training_outcome_and_duplicate_forecasts_are_rejected(self):
        past,future=self.frame("2023-12-29"),self.frame("2024-01-02")
        bad=past.copy();bad["future_oc_return"]=np.nan
        with self.assertRaises(ValueError): require_past_training(bad,future,"2024-01-01")
        with self.assertRaises(ValueError): require_past_training(past,pd.concat([future,future]),"2024-01-01")

    def test_five_minute_gap_cannot_pass_as_available_at_the_open(self):
        frame=pd.DataFrame(dict(overnight_gap=[.01,np.nan],
            signal_cutoff_utc=pd.to_datetime(["2024-01-02T14:30Z"]*2),
            opening_bar_available_at_utc=pd.to_datetime(["2024-01-02T14:35Z"]*2)))
        self.assertEqual(execution_timing_violations(frame).tolist(),[True,False])
        with self.assertRaises(ValueError): require_execution_information_timing(frame)
        frame["signal_cutoff_utc"]=pd.to_datetime(["2024-01-02T14:35Z"]*2)
        require_execution_information_timing(frame)

    def test_only_the_explicit_opening_gap_can_use_native_nan(self):
        names=list(ensemble.EXPERIMENTS["combined15"]["features"])
        values=np.ones((3,15),dtype=np.float32);values[:,-1]=np.nan
        ensemble.validate_matrix(values,names)
        values[0,8]=np.nan
        with self.assertRaises(ValueError):ensemble.validate_matrix(values,names)
        with self.assertRaises(ValueError):ensemble.validate_matrix(np.ones((3,15)),names[:-1]+["future_oc_return"])

    def test_experiment_import_does_not_mutate_the_shared_feature_list(self):
        self.assertFalse(hasattr(backtest,"MODEL_FEATURES"))
        original=ensemble.EXPERIMENTS["momentum8"]["features"]
        importlib.reload(ensemble)
        self.assertFalse(hasattr(backtest,"MODEL_FEATURES"))
        self.assertEqual(ensemble.EXPERIMENTS["momentum8"]["features"],original)
        self.assertIsInstance(original,tuple)

    def test_all_six_volume_features_are_invariant_to_stock_activity_units(self):
        i=pd.bdate_range("2024-01-02",periods=30)
        n=len(i);d=np.linspace(1000,2000,n)
        frame=pd.DataFrame(dict(dollar_volume=d,last_hour_dollar_volume=.15*d,
            signed_dollar_volume=.3*d,share_volume=d/100,closing_price_raw=101.,
            session_minutes=390.,activity_valid=True,signed_valid=True),index=i)
        scaled=frame.copy()
        for c in ["dollar_volume","last_hour_dollar_volume","signed_dollar_volume","share_volume"]:
            scaled[c]*=1_000_000
        np.testing.assert_allclose(make_volume_features(frame)[VOLUME_FEATURES],
            make_volume_features(scaled)[VOLUME_FEATURES],rtol=1e-12,atol=1e-12,equal_nan=True)

    def test_actual_compressed_parameters_are_prefix_causal_with_missing_slots(self):
        t=np.arange(2400);p=pd.Series(100*np.exp(.001*np.sin(t/23)+.00001*t))
        p.iloc[1000]=np.nan
        horizons=(12,78,234,390,780)
        def calculate(x):
            returns=pd.DataFrame({str(h):x/x.shift(h)-1 for h in horizons})
            lengths=pd.DataFrame({str(h):float(h) for h in horizons},index=x.index)
            return individual_features(x,x/x.shift(1)-1,returns,lengths,PARAMETERS)
        a=calculate(p);changed=p.copy();changed.iloc[1800:]*=1000
        pd.testing.assert_frame_equal(a.iloc[:1800],calculate(changed).iloc[:1800])

    def test_payoff_coverage_counts_dates_with_no_supervised_rows(self):
        dates=pd.bdate_range("2023-01-02",periods=102)
        records=[dict(trade_date=date,realized_class=k,future_oc_return=.001*k)
            for date in dates[:-1] for k in range(1,11)]
        mu,daily,complete=backtest.class_payoffs(pd.DataFrame(records),expected_dates=dates)
        self.assertEqual(len(daily),102)
        self.assertEqual(int((~complete).sum()),1)
        np.testing.assert_allclose(mu,np.arange(1,11)*.001)

    def test_nan_scores_and_duplicate_stock_rows_cannot_change_selection_silently(self):
        frame=pd.concat([self.frame("2024-01-02").assign(security_id=str(i),ret_score=float(i)) for i in range(6)])
        cal=pd.DataFrame(dict(trade_date=[pd.Timestamp("2024-01-02")]))
        with self.assertRaises(ValueError):backtest.portfolio_days(pd.concat([frame,frame.iloc[:1]]),cal,2)
        frame.loc[frame.security_id.eq("5"),"ret_score"]=np.nan
        with self.assertRaises(ValueError):backtest.portfolio_days(frame,cal,2)


if __name__=="__main__":unittest.main()
