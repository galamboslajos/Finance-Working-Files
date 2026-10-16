import unittest
import numpy as np
import pandas as pd

from dmv2.backtest_selection import select_monthly_basket
from dmv2.universe import DataContractError


def fixture():
    rows = []
    for k in (1, 2, 9, 10):
        for i in range(12):
            p = np.full(10, .02)
            p[k - 1] = .82
            rows.append({"source_security_id": f"source{k:02d}-{i:02d}",
                "intrinio_security_id": f"native{k:02d}-{i:02d}",
                "signal_date": "2017-12-29", "outcome_month": "2018-01",
                "prediction_eligible": True, "target_available": i % 2 == 0,
                "target_class": (i % 10) + 1,
                **{f"probability_{j+1}": v for j, v in enumerate(p)}})
    rows.append({**rows[-1], "source_security_id": "blocked", "prediction_eligible": False})
    return pd.DataFrame(rows)


SESSIONS = ("2017-12-28", "2017-12-29", "2018-01-02", "2018-01-03")


class SelectionTests(unittest.TestCase):
    def select(self, frame=None, **kwargs):
        return select_monthly_basket(fixture() if frame is None else frame,
            basket=kwargs.pop("basket", "class_1_10"), sessions=SESSIONS, **kwargs)

    def test_fixed_ten_per_direction_without_observed_label_filter(self):
        r = self.select()
        self.assertEqual(len(r.ledger), 49)
        self.assertEqual(r.selections.groupby("side").size().to_dict(), {"long": 10, "short": 10})
        self.assertEqual(r.rebalances.execution_date.tolist(), ["2018-01-02"])
        self.assertNotIn("target_class", r.ledger)
        self.assertNotIn("target_available", r.ledger)
        self.assertEqual(r.monthly.long_candidates.tolist(), [12])
        self.assertEqual(r.monthly.long_cutoff_equal_candidates.tolist(), [12])

    def test_separate_second_tail_classes(self):
        r = self.select(basket="class_2_9")
        self.assertEqual(set(r.selections.predicted_class), {2, 9})

    def test_tied_selection_uses_stable_id_not_input_order_or_targets(self):
        frame = fixture()
        expected = self.select(frame).selections.source_security_id.tolist()
        frame = frame.sample(frac=1, random_state=7)
        frame["target_available"] = True
        frame["target_class"] = 10
        self.assertEqual(self.select(frame).selections.source_security_id.tolist(), expected)

    def test_ranks_confidence_inside_mode_class_not_global_probability(self):
        frame = fixture()
        idx = frame.index[frame.source_security_id == "source01-11"][0]
        frame.loc[idx, "probability_1"] = .91
        frame.loc[idx, "probability_2"] = .01
        for j in range(3, 11):
            frame.loc[idx, f"probability_{j}"] = .01
        r = self.select(frame)
        self.assertIn("source01-11", r.selections.source_security_id.tolist())

    def test_shortage_withholds_both_sides_without_other_class_fallback(self):
        frame = fixture()
        frame = frame.loc[~frame.source_security_id.isin(["source01-09", "source01-10", "source01-11"])]
        r = self.select(frame)
        self.assertTrue(r.selections.empty)
        self.assertEqual(r.monthly.status.tolist(), ["withheld_insufficient_mode_candidates"])
        self.assertEqual(len(r.rebalances), 1)

    def test_ret_remains_withheld_without_verified_complete_payoffs(self):
        r = self.select(basket="ret")
        self.assertTrue(r.selections.empty)
        self.assertEqual(r.monthly.status.tolist(), ["withheld_incomplete_120_month_class_grid"])

    def test_no_price_or_outcome_api_argument(self):
        with self.assertRaises(TypeError):
            self.select(future_prices=pd.DataFrame())

    def test_duplicate_native_identity_blocks_selection(self):
        frame = fixture()
        frame.loc[1, "intrinio_security_id"] = frame.loc[0, "intrinio_security_id"]
        with self.assertRaisesRegex(DataContractError, "collision"):
            self.select(frame)

    def test_invalid_probabilities_fail_without_repair(self):
        frame = fixture()
        frame.loc[0, "probability_1"] = .81
        with self.assertRaisesRegex(DataContractError, "sum to one"):
            self.select(frame)

    def test_2020_request_rejected_before_probability_conversion(self):
        frame = fixture()
        frame["signal_date"] = "2019-12-31"
        frame["outcome_month"] = "2020-01"
        with self.assertRaises(DataContractError):
            select_monthly_basket(frame, basket="class_1_10", sessions=(*SESSIONS, "2020-01-02"))


if __name__ == "__main__":
    unittest.main()
