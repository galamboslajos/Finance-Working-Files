"""Meaningful accounting checks for the standard per-execution cost."""
from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
from trading_costs import STANDARD_COST_BPS,round_trip_notional,net_after_costs

class Costs(unittest.TestCase):
    def test_two_legs_and_N_do_not_double_charge_or_change_total_cost(self):
        self.assertEqual(STANDARD_COST_BPS,10)
        for n in (2,3):
            weights=np.r_[np.full(n,.5/n),np.full(n,-.5/n)]
            notional=round_trip_notional(weights,np.zeros(2*n)).sum()
            self.assertAlmostEqual(notional,2)
            self.assertAlmostEqual(float(net_after_costs(0,notional)),-.002)

    def test_exit_notional_changes_with_actual_stop_fill_for_both_directions(self):
        weights=np.array([.25,-.25]);raw=np.array([-.08,.08])
        notional=round_trip_notional(weights,raw)
        np.testing.assert_allclose(notional,[.48,.52])
        net=net_after_costs(weights*raw,notional)
        np.testing.assert_allclose(net,[-.02048,-.02052])

    def test_stopped_position_pays_entry_and_exit_even_when_held_one_bar(self):
        self.assertAlmostEqual(float(net_after_costs(-.0025,round_trip_notional(.25,-.01))),-.0029975)

    def test_cash_zero_and_unknown_outcomes_stay_unknown(self):
        result=net_after_costs([0,np.nan,0],[0,2,np.nan])
        self.assertEqual(result[0],0)
        self.assertTrue(np.isnan(result[1:]).all())
        self.assertTrue(np.isnan(round_trip_notional(.25,np.nan)))

    def test_cost_difference_is_linear_in_executed_notional(self):
        gross=np.array([.02,-.01,0]);notional=np.array([2.1,1.9,0])
        np.testing.assert_allclose(net_after_costs(gross,notional,1)-net_after_costs(gross,notional),.0009*notional)

    def test_invalid_units_or_impossible_inputs_rejected(self):
        for cost in (-1,np.inf,np.nan):
            with self.assertRaises(ValueError):net_after_costs(.01,2,cost)
        with self.assertRaises(ValueError):net_after_costs(.01,-2)
        with self.assertRaises(ValueError):round_trip_notional(.25,-1.2)
        with self.assertRaises(ValueError):net_after_costs(np.inf,2)

if __name__=='__main__':unittest.main()
