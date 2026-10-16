"""Balanced contemporaneous counts, unchanged ordering and ledger safety."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from dmv2.action_accounting import run_action_accounting
from dmv2.backtest_accounting import DailyAccountingPolicy
from dmv2.backtest_selection import PROBABILITIES
from dmv2.universe import DataContractError

try:
    from dmv2 import balanced_baskets as worker
except ImportError:
    spec = importlib.util.spec_from_file_location("dmv2.balanced_baskets", Path(__file__).with_name("balanced_baskets.py"))
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)

ROOT = Path(__file__).resolve().parents[1]
SESSIONS = ("2017-12-29", "2018-01-02", "2018-01-31")


def configuration():
    return json.loads((ROOT / "configs/data_limited_fallback.json").read_text())


def predictions(nlong=9, nshort=37, classes=(1, 10), tie=False):
    rows = []
    for side, count, mode in (("long", nlong, classes[0]), ("short", nshort, classes[1])):
        for i in range(count):
            p = .8 if tie else .8 - i / 1000
            values = {f"probability_{k}": p if k == mode else (1 - p) / 9 for k in range(1, 11)}
            rows.append({"source_security_id": f"{side}-{i:03d}", "intrinio_security_id": f"native-{side}-{i:03d}",
                "signal_date": "2017-12-29", "prediction_eligible": True, "paper_prediction_eligible": True,
                "prediction_status": "predicted", "identity_eligible": True,
                "identity_review_required": True, **values})
    return pd.DataFrame(rows)


class BalancedBasketsTests(unittest.TestCase):
    def select(self, source=None, basket="class_1_10", **kwargs):
        return worker.select_balanced_monthly_basket(predictions() if source is None else source,
            basket=basket, sessions=SESSIONS, configuration=kwargs.get("configuration", configuration()))

    def validate(self, result, source, basket="class_1_10"):
        return worker.validate_balanced_monthly_basket(result, source, basket=basket,
            sessions=SESSIONS, configuration=configuration())

    def test_nine_and_thirtyseven_become_nine_and_nine_with_half_capital_each(self):
        source = predictions()
        result = self.select(source)
        self.assertEqual(result.selections.groupby("side").size().to_dict(), {"long": 9, "short": 9})
        month = result.monthly.iloc[0]
        self.assertEqual((month.long_candidates, month.short_candidates), (9, 37))
        self.assertEqual(month.original_rows, 46)
        self.assertEqual(month.requested_top_k, 10)
        self.assertEqual(month.selected_names_per_side, 9)
        self.assertAlmostEqual(month.effective_per_name_abs_weight, .5 / 9.)
        self.assertEqual(self.validate(result, source)["adapted_months"], 1)

    def test_ten_or_more_keeps_the_old_top_ten(self):
        source = predictions(12, 37)
        result = self.select(source)
        self.assertEqual(result.selections.groupby("side").size().to_dict(), {"long": 10, "short": 10})
        self.assertEqual(set(result.selections.source_security_id),
            {f"{side}-{i:03d}" for side in ("long", "short") for i in range(10)})
        self.assertEqual(result.monthly.iloc[0].breadth_adaptation_reason, "unchanged_full_ten_per_side")

    def test_zero_leg_retains_withheld_schedule_and_original_denominator(self):
        source = predictions(0, 37)
        result = self.select(source)
        self.assertTrue(result.selections.empty)
        self.assertEqual(len(result.ledger), 37)
        self.assertEqual(result.rebalances.iloc[0].selection_status, "withheld_zero_mode_candidates")
        self.assertEqual(result.monthly.iloc[0].original_rows, 37)
        self.assertTrue(np.isnan(result.monthly.iloc[0].effective_per_name_abs_weight))
        self.assertEqual(self.validate(result, source)["withheld_zero_breadth_months"], 1)

    def test_probability_ties_use_source_id_and_do_not_respond_to_row_order(self):
        source = predictions(9, 37, tie=True).sample(frac=1., random_state=17).reset_index(drop=True)
        result = self.select(source)
        self.assertEqual(set(result.selections.loc[result.selections.side.eq("short"), "source_security_id"]),
            {f"short-{i:03d}" for i in range(9)})
        self.assertEqual(result.ledger.source_security_id.tolist(), source.source_security_id.tolist())

    def test_second_pair_does_not_pool_extremes(self):
        source = predictions(9, 37, classes=(2, 9))
        result = self.select(source, basket="class_2_9")
        self.assertEqual(set(result.selections.predicted_class), {2, 9})
        self.assertEqual(set(result.selections.loc[result.selections.side.eq("long"), "predicted_class"]), {2})
        other = self.select(source, basket="class_1_10")
        self.assertTrue(other.selections.empty)
        self.assertEqual(self.validate(result, source, basket="class_2_9")["adapted_months"], 1)

    def test_fixed_trade_exclusions_remain_prediction_rows_before_balancing(self):
        source = predictions(10, 38)
        source.loc[0, "intrinio_security_id"] = "sec_gjP4xM"
        source.loc[10, "intrinio_security_id"] = "sec_zmYNnz"
        result = self.select(source)
        self.assertEqual(len(result.ledger), 48)
        self.assertTrue(result.ledger.prediction_eligible.all())
        self.assertEqual(result.ledger.trade_eligible.sum(), 46)
        excluded = result.ledger.loc[result.ledger.execution_restricted]
        self.assertEqual(len(excluded), 2)
        self.assertFalse(excluded.selected.any())
        self.assertTrue(excluded.selection_status.eq("execution_excluded_retrospective_data_quality").all())
        self.assertEqual(result.selections.groupby("side").size().to_dict(), {"long": 9, "short": 9})
        self.assertEqual(result.monthly.iloc[0].prediction_eligible, 48)
        self.assertEqual(result.monthly.iloc[0].execution_restricted_prediction_rows, 2)

    def test_ineligible_bbby_and_other_original_rows_are_not_promoted_or_dropped(self):
        source = predictions(10, 37)
        source.loc[0, "source_security_id"] = "siblis_isin_c297a5f81a23c9f90503"
        source.loc[0, "prediction_eligible"] = False
        source.loc[0, "paper_prediction_eligible"] = False
        source.loc[0, "identity_eligible"] = False
        source.loc[0, "intrinio_security_id"] = None
        source.loc[0, list(PROBABILITIES)] = np.nan
        before = source.copy(deep=True)
        result = self.select(source)
        pd.testing.assert_frame_equal(source, before)
        self.assertEqual(len(result.ledger), len(source))
        self.assertFalse(result.ledger.loc[0, "selected"])
        self.assertFalse(result.ledger.loc[0, "trade_eligible"])
        self.assertEqual(result.monthly.iloc[0].selected_names_per_side, 9)

    def test_targets_future_membership_and_quotes_do_not_change_any_selection_table(self):
        source = predictions()
        expected = self.select(source)
        source["target_class"] = 10
        source["target_return"] = np.nan
        source["daily_quote_path_complete"] = False
        source["future_membership"] = False
        actual = self.select(source)
        for name in ("ledger", "selections", "rebalances", "monthly"):
            pd.testing.assert_frame_equal(getattr(expected, name), getattr(actual, name))

    def test_validator_rejects_extra_exclusion_wrong_rank_weight_or_dropped_original(self):
        source = predictions()
        for alteration in ("extra_exclusion", "rank", "weight", "denominator"):
            with self.subTest(alteration=alteration):
                result = deepcopy(self.select(source))
                if alteration == "extra_exclusion":
                    result.ledger.loc[0, "trade_eligible"] = False
                elif alteration == "rank":
                    result.selections.loc[0, "source_security_id"] = "short-036"
                elif alteration == "weight":
                    result.selections.loc[0, "intended_weight"] = .05
                else:
                    result.ledger.drop(index=0, inplace=True)
                with self.assertRaises(DataContractError):
                    self.validate(result, source)

    def test_accounting_equal_weights_actual_nine_per_side_without_fixed_ten_gate(self):
        source = predictions()
        result = self.select(source)
        self.validate(result, source)
        selected = result.selections
        quote_rows = [{"trading_date": day, "intrinio_security_id": native,
            "adj_open": 100., "adj_close": 100., "frequency": "daily", "intraperiod": False,
            "price_quality_status": "accepted"} for day in SESSIONS[1:]
            for native in selected.intrinio_security_id]
        accounted = run_action_accounting(selected, pd.DataFrame(quote_rows), SESSIONS[1:],
            rebalances=result.rebalances, terminal_date=SESSIONS[-1],
            policy=DailyAccountingPolicy(required_names_per_side=None))
        self.assertEqual(accounted.status, "complete_proxy")
        self.assertTrue(np.allclose(accounted.orders.intended_weight.abs(), .5 / 9.))
        self.assertAlmostEqual(accounted.orders.loc[accounted.orders.side.eq("long"), "intended_weight"].sum(), .5)
        self.assertAlmostEqual(accounted.orders.loc[accounted.orders.side.eq("short"), "intended_weight"].sum(), -.5)
        with_ten = run_action_accounting(selected, pd.DataFrame(quote_rows), SESSIONS[1:],
            rebalances=result.rebalances, terminal_date=SESSIONS[-1],
            policy=DailyAccountingPolicy(required_names_per_side=10))
        self.assertNotEqual(with_ten.status, "complete_proxy")

    def test_ret_cannot_bypass_unchanged_history_gate(self):
        with self.assertRaises(DataContractError):
            self.select(basket="ret")


if __name__ == "__main__":
    unittest.main()
