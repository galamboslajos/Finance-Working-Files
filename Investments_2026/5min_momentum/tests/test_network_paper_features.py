"""Independent formula, dimensional consistency and causality checks."""
import sys
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from network_paper_features import (FeatureParameters, historical_clip,
                                    individual_features, normalized_macds,
                                    weighted_volatility)


class NetworkFeatureTests(unittest.TestCase):
    def test_first_clip_observation_has_zero_population_dispersion(self):
        first=pd.DataFrame({"x":[np.nan,3.,4.]})
        result=historical_clip(first,10)
        self.assertTrue(np.isnan(result.x.iloc[0]))
        self.assertEqual(result.x.iloc[1],3.)

    def test_population_ewm_matches_explicit_appendix_weights(self):
        r = pd.Series([.01, -.02, .03, -.015, .007])
        observed = weighted_volatility(r, 4).iloc[-1]
        weights = (1-2/5)**np.arange(len(r)-1, -1, -1)
        mean = np.average(r, weights=weights)
        expected = np.sqrt(np.average((r-mean)**2, weights=weights))
        self.assertAlmostEqual(observed, expected, places=14)

    def fixture(self):
        t = np.arange(1500)
        p = pd.Series(100*np.exp(.0002*t+.02*np.sin(t/23)+.005*np.cos(t/7)))
        r = p/p.shift(1)-1
        returns = pd.DataFrame({str(h):p/p.shift(h)-1 for h in (2,5,10,20,40)})
        lengths = pd.DataFrame({str(h):float(h) for h in (2,5,10,20,40)}, index=p.index)
        return p,r,returns,lengths

    def test_macd_uses_alpha_one_over_scale_and_both_normalizations(self):
        p,_,_,_ = self.fixture()
        cfg = FeatureParameters()
        fast,slow = p.iloc[0],p.iloc[0]
        raw = []
        for value in p:
            fast = value/8+(1-1/8)*fast
            slow = value/24+(1-1/24)*slow
            raw.append(fast-slow)
        p_sd = np.std(p.iloc[-63:], ddof=0)
        normalized = []
        for i in range(len(p)-252, len(p)):
            normalized.append(raw[i]/np.std(p.iloc[i-62:i+1], ddof=0))
        expected = raw[-1]/p_sd/np.std(normalized, ddof=0)
        observed = normalized_macds(p,cfg).nm_macd_1.iloc[-1]
        self.assertAlmostEqual(observed,expected,places=9)

    def test_eight_features_and_price_rescaling_invariance(self):
        p,r,ret,length = self.fixture()
        a=individual_features(p,r,ret,length,FeatureParameters())
        b=individual_features(p*19,r,ret,length,FeatureParameters())
        self.assertEqual(a.shape[1],8)
        np.testing.assert_allclose(a,b,atol=1e-8,rtol=1e-8,equal_nan=True)
        sigma=weighted_volatility(r,60).iloc[-1]
        uncut=individual_features(p,r,ret,length,FeatureParameters(),apply_clipping=False)
        self.assertAlmostEqual(uncut.nm_scaled_40.iloc[-1],ret['40'].iloc[-1]/sigma/np.sqrt(40))

    def test_future_perturbations_do_not_change_completed_features(self):
        p,r,ret,length=self.fixture()
        baseline=individual_features(p,r,ret,length,FeatureParameters())
        cutoff=1100
        changed=p.copy()
        changed.iloc[cutoff+1:]*=10
        future_r=changed/changed.shift(1)-1
        future_ret=pd.DataFrame({str(h):changed/changed.shift(h)-1 for h in (2,5,10,20,40)})
        alternate=individual_features(changed,future_r,future_ret,length,FeatureParameters())
        pd.testing.assert_frame_equal(baseline.loc[:cutoff],alternate.loc[:cutoff])

    def test_no_gap_compression_and_zero_denominators_remain_undefined(self):
        p,_,_,_=self.fixture()
        p.iloc[1000]=np.nan
        macds=normalized_macds(p,FeatureParameters())
        self.assertTrue(macds.loc[1000].isna().all())
        # The missing scheduled observation invalidates the full price-std window.
        self.assertTrue(macds.loc[1001:1062].isna().all().all())
        constant=pd.Series(np.repeat(100.,1500))
        self.assertTrue(normalized_macds(constant,FeatureParameters()).isna().all().all())
        with self.assertRaises(ValueError):
            normalized_macds(p,FeatureParameters(macd_scales=((.5,2),(2,4),(4,8))))


if __name__=='__main__':
    unittest.main()
