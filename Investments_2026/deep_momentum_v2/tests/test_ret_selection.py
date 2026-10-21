"""Synthetic RET ranking contracts; no financial tables, fits or backtests."""
from __future__ import annotations

from calendar import monthrange
from copy import deepcopy
from dataclasses import replace
from functools import lru_cache
import math
import unittest

import numpy as np
import pandas as pd
import pyarrow as pa

from dmv2.catalog import canonical_sha256
from dmv2.monthly import _step_month
from dmv2.ret import (CLASS_IDS, MONTHLY_PAYOFF_SCHEMA, PAYOFF_SCHEMA,
    RET_POLICY, RET_POLICY_DEFINITION, RetPayoffPanel, _fingerprint,
    verify_ret_payoffs)
from dmv2.ret_selection import select_ret_walkforward_basket, validate_ret_walkforward_basket
from dmv2.universe import DataContractError
from dmv2.walkforward_clock import walkforward_clock
from dmv2.walkforward_selection import EXCLUDED_NATIVE_IDS, PROBABILITIES


@lru_cache(maxsize=1)
def clock_fixture():
    return walkforward_clock()


def config_fixture():
    # Minimal synthetic contract accepted by the preserved configuration guard.
    return {"protocol_id": "dmv2-walkforward-2020-2023-v1",
        "execution_restrictions": {"native_ids": {value: "fixed restriction" for value in EXCLUDED_NATIVE_IDS},
            "period": "all_2020_2023_signals_before_selection",
            "valid_historical_training_rows_retained": True,
            "prediction_rows_and_probabilities_retained": True},
        "backtest": {"start": "2020-01-01", "end": "2023-12-31", "top_k_per_side": 10,
            "baskets": ["class_1_10", "class_2_9"], "long_weight": .5, "short_weight": -.5}}


@lru_cache(maxsize=256)
def payoff_fixture(signal, shift=0.0, flat=False, missing=False):
    first = _step_month(signal[:7], -119)
    monthly = []
    values = [shift + value for value in (.20, .10, .08, .05, .03, .01, -.01, -.03, -.10, -.20)]
    if flat:
        values = [shift + .01] * 10
    for offset in range(120):
        month = _step_month(first, offset)
        year, number = map(int, month.split("-"))
        end = min(f"{month}-{monthrange(year, number)[1]:02d}", signal)
        for class_id in CLASS_IDS:
            absent = missing and offset == 3 and class_id == 7
            monthly.append({"outcome_month": month, "outcome_end": end,
                "class_id": class_id, "original_member_count": 10,
                "prediction_eligible_count": 10, "member_count": 0 if absent else 1,
                "class_mean_return": None if absent else values[class_id - 1],
                "status": "empty_realized_class" if absent else "available"})
    rows = [{"class_id": class_id,
        "payoff": None if missing else math.fsum([values[class_id - 1] / 120] * 120),
        "status": "withheld_incomplete_120_month_class_grid" if missing else "available"}
        for class_id in CLASS_IDS]
    table, months = pa.Table.from_pylist(rows, schema=PAYOFF_SCHEMA), pa.Table.from_pylist(monthly, schema=MONTHLY_PAYOFF_SCHEMA)
    provenance = {"ret_policy": RET_POLICY, "policy_sha256": canonical_sha256(RET_POLICY_DEFINITION),
        "signal_date": signal, "label_observed_through": signal,
        "first_outcome_month": first, "last_outcome_month": signal[:7],
        "vector_available": not missing}
    result = RetPayoffPanel(table, months, provenance, _fingerprint(table, months, provenance))
    assert verify_ret_payoffs(result)
    return result


def predictions_fixture(count=25, years=(2020,), shift=0.0, flat=False):
    _, windows, _ = clock_fixture()
    selected = windows.loc[windows.outcome_month.str[:4].astype(int).isin(years)]
    rows, payoffs = [], {}
    for window in selected.itertuples():
        signal = window.signal_date
        payoffs[signal] = payoff_fixture(signal, shift=shift, flat=flat)
        for i in range(count):
            # Every forecast has mode class 5. RET still ranks all candidates.
            p = np.full(10, .05)
            p[4] = .55
            p[0] += .04 * (2 * i / max(1, count - 1) - 1)
            p[9] -= .04 * (2 * i / max(1, count - 1) - 1)
            rows.append({"source_security_id": f"id{i:03d}", "intrinio_security_id": f"native{i:03d}",
                "signal_date": signal, "outcome_month": window.outcome_month,
                "outcome_start": window.outcome_start, "outcome_end": window.outcome_end,
                "prediction_eligible": True, "paper_prediction_eligible": True,
                "identity_status": "provisionally_admitted", "needs_review": True,
                **dict(zip(PROBABILITIES, p))})
    return pd.DataFrame(rows), payoffs


def select(predictions, payoffs, *, years=(2020,), sessions=None, configuration=None):
    return select_ret_walkforward_basket(predictions, payoffs=payoffs,
        sessions=clock_fixture()[2]["selection_sessions"] if sessions is None else sessions,
        configuration=config_fixture() if configuration is None else configuration,
        outcome_years=years)


class RetSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictions, cls.payoffs = predictions_fixture()
        cls.result = select(cls.predictions, cls.payoffs)

    def test_full_probability_RET_ranks_without_mode_gate(self):
        result = self.result
        self.assertTrue(result.ledger.predicted_class.eq(5).all())
        self.assertTrue(result.monthly.status.eq("available").all())
        first = result.selections.loc[result.selections.signal_date.eq("2019-12-31")]
        self.assertEqual(set(first.loc[first.side.eq("long"), "source_security_id"]), {f"id{i:03d}" for i in range(15, 25)})
        self.assertEqual(set(first.loc[first.side.eq("short"), "source_security_id"]), {f"id{i:03d}" for i in range(10)})
        mu = np.asarray(self.payoffs["2019-12-31"].table["payoff"].to_pylist())
        p = self.predictions.loc[self.predictions.signal_date.eq("2019-12-31"), list(PROBABILITIES)].to_numpy()
        manual = np.array([math.fsum(float(a) * float(b) for a, b in zip(row, mu)) for row in p])
        np.testing.assert_allclose(result.ledger.loc[result.ledger.signal_date.eq("2019-12-31"), "selection_score"], manual, rtol=0, atol=2e-16)

    def test_all_original_rows_flags_and_order_retained(self):
        ledger = self.result.ledger
        for field in ("source_security_id", "signal_date", "prediction_eligible", "paper_prediction_eligible", "needs_review", "identity_status"):
            pd.testing.assert_series_equal(ledger[field], self.predictions[field], check_names=True)
        self.assertEqual(len(ledger), len(self.predictions))
        self.assertNotIn("probability_1", ledger)

    def test_weights_balance_and_no_overlap(self):
        for group in self.result.selections.groupby("signal_date"):
            frame = group[1]
            self.assertEqual(len(frame), 20)
            self.assertFalse(frame.source_security_id.duplicated().any())
            self.assertAlmostEqual(frame.loc[frame.side.eq("long"), "intended_weight"].sum(), .5)
            self.assertAlmostEqual(frame.loc[frame.side.eq("short"), "intended_weight"].sum(), -.5)

    def test_odd_small_breadth_uses_floor_half(self):
        frame, payoffs = predictions_fixture(count=5)
        result = select(frame, payoffs)
        self.assertTrue(result.monthly.selected_names_per_side.eq(2).all())
        self.assertEqual(len(result.selections), 48)
        self.assertTrue(result.selections.effective_per_name_abs_weight.eq(.25).all())

    def test_single_candidate_withholds_instead_of_cash_or_overlap(self):
        frame, payoffs = predictions_fixture(count=1)
        result = select(frame, payoffs)
        self.assertTrue(result.monthly.status.eq("withheld_zero_distinct_RET_breadth").all())
        self.assertTrue(result.rebalances.selection_status.eq("withheld_zero_distinct_RET_breadth").all())
        self.assertTrue(result.selections.empty)
        self.assertEqual(len(result.ledger), 12)

    def test_zero_trade_eligible_withholds(self):
        frame, payoffs = predictions_fixture(count=2)
        frame["prediction_eligible"] = False
        frame[list(PROBABILITIES)] = np.nan
        frame["intrinio_security_id"] = None
        result = select(frame, payoffs)
        self.assertTrue(result.monthly.trade_eligible.eq(0).all())
        self.assertTrue(result.monthly.status.eq("withheld_zero_distinct_RET_breadth").all())
        self.assertTrue(result.ledger.selection_status.eq("withheld_signal_time_baseline").all())

    def test_missing_payoff_signal_keeps_rows_and_withholds(self):
        payoffs = dict(self.payoffs)
        del payoffs["2019-12-31"]
        result = select(self.predictions, payoffs)
        failed = result.monthly.iloc[0]
        self.assertEqual(failed.status, "withheld_incomplete_120_month_class_grid")
        self.assertEqual(failed.selected_names_per_side, 0)
        self.assertEqual(len(result.ledger), len(self.predictions))
        self.assertTrue(result.ledger.loc[result.ledger.signal_date.eq("2019-12-31"), "selection_score"].isna().all())

    def test_one_missing_cell_withholds_whole_vector_no_reduced_history(self):
        payoffs = dict(self.payoffs)
        payoffs["2019-12-31"] = payoff_fixture("2019-12-31", missing=True)
        result = select(self.predictions, payoffs)
        self.assertEqual(result.monthly.iloc[0].status, "withheld_incomplete_120_month_class_grid")
        self.assertEqual(len(result.selections), 11 * 20)

    def test_future_payoff_cutoff_and_different_signal_rejected(self):
        original = self.payoffs["2019-12-31"]
        for field, value in (("label_observed_through", "2020-01-01"), ("signal_date", "2020-01-31")):
            provenance = deepcopy(original.provenance)
            provenance[field] = value
            changed = replace(original, provenance=provenance,
                fingerprint=_fingerprint(original.table, original.monthly, provenance))
            payoffs = dict(self.payoffs)
            payoffs["2019-12-31"] = changed
            with self.subTest(field=field), self.assertRaises(DataContractError):
                select(self.predictions, payoffs)

    def test_modified_payoff_fingerprint_rejected(self):
        payoffs = dict(self.payoffs)
        original = payoffs["2019-12-31"]
        payoffs["2019-12-31"] = replace(original, fingerprint="changed")
        with self.assertRaises(DataContractError):
            select(self.predictions, payoffs)

    def test_all_positive_and_all_negative_scores_keep_ranked_book(self):
        for shift in (1.0, -.5):
            frame, payoffs = predictions_fixture(count=5, shift=shift)
            result = select(frame, payoffs)
            with self.subTest(shift=shift):
                self.assertTrue(result.monthly.status.eq("available").all())
                self.assertTrue((result.ledger.selection_score > 0).all() if shift > 0 else (result.ledger.selection_score < 0).all())
                self.assertEqual(len(result.selections), 48)

    def test_flat_scores_have_stable_disjoint_ties_and_diagnostic(self):
        frame, payoffs = predictions_fixture(count=5, flat=True)
        # Identical distributions guarantee exact floating-point RET ties.
        frame[list(PROBABILITIES)] = .1
        result = select(frame, payoffs)
        first = result.selections.loc[result.selections.signal_date.eq("2019-12-31")]
        self.assertEqual(list(first.loc[first.side.eq("long"), "source_security_id"]), ["id000", "id001"])
        self.assertEqual(list(first.loc[first.side.eq("short"), "source_security_id"]), ["id002", "id003"])
        self.assertTrue(result.monthly.ret_score_flat.all())
        self.assertTrue(result.monthly.status.eq("available").all())
        self.assertTrue(result.monthly.long_cutoff_equal_candidates.eq(5).all())
        self.assertTrue(result.monthly.short_cutoff_equal_candidates.eq(3).all())

    def test_permuted_rows_and_probability_columns_have_same_selections(self):
        permuted = self.predictions.sample(frac=1, random_state=47).reset_index(drop=True)
        columns = list(reversed(permuted.columns))
        result = select(permuted[columns], self.payoffs)
        fields = ["signal_date", "source_security_id", "side", "selection_score", "intended_weight"]
        pd.testing.assert_frame_equal(result.selections[fields], self.result.selections[fields], check_exact=True)
        self.assertEqual(list(result.ledger.source_security_id), list(permuted.source_security_id))

    def test_fixed_trade_restrictions_retain_admissions_modes_and_scores(self):
        frame, payoffs = predictions_fixture(count=5)
        frame.loc[frame.source_security_id.eq("id004"), "intrinio_security_id"] = EXCLUDED_NATIVE_IDS[0]
        frame.loc[frame.source_security_id.eq("id000"), "intrinio_security_id"] = EXCLUDED_NATIVE_IDS[1]
        result = select(frame, payoffs)
        restricted = result.ledger.execution_restricted
        self.assertTrue(result.ledger.prediction_eligible.all())
        self.assertTrue(result.ledger.loc[restricted, "selection_score"].notna().all())
        self.assertTrue(result.ledger.loc[restricted, "predicted_class"].eq(5).all())
        self.assertTrue(result.ledger.loc[restricted, "selection_status"].eq("execution_excluded_retrospective_data_quality").all())
        self.assertFalse(result.selections.intrinio_security_id.isin(EXCLUDED_NATIVE_IDS).any())
        self.assertTrue(result.monthly.trade_eligible.eq(3).all())
        self.assertTrue(result.monthly.selected_names_per_side.eq(1).all())

    def test_bad_restricted_original_probability_cannot_hide_behind_exclusion(self):
        frame = self.predictions.copy()
        frame.loc[0, "intrinio_security_id"] = EXCLUDED_NATIVE_IDS[0]
        frame.loc[0, "probability_1"] = np.nan
        with self.assertRaises(DataContractError):
            select(frame, self.payoffs)

    def test_targets_quote_flags_and_supplied_mode_are_ignored(self):
        frame = self.predictions.copy()
        frame["target_available"] = False
        frame["label_return"] = "NEVER DECODE THIS"
        frame["target_class"] = object()
        frame["future_quote_available"] = False
        frame["predicted_class"] = 10
        result = select(frame, self.payoffs)
        for field in ("ledger", "selections", "monthly", "rebalances"):
            pd.testing.assert_frame_equal(getattr(result, field), getattr(self.result, field), check_exact=True)

    def test_eligibility_is_original_signal_state_not_outcome_availability(self):
        frame = self.predictions.copy()
        frame["target_available"] = False
        frame.loc[frame.source_security_id.eq("id024"), "prediction_eligible"] = False
        result = select(frame, self.payoffs)
        self.assertEqual(len(result.ledger), len(frame))
        self.assertFalse(result.selections.source_security_id.eq("id024").any())
        self.assertTrue(result.monthly.trade_eligible.eq(24).all())

    def test_invalid_probability_class_columns_or_boolean_values_rejected(self):
        invalid = []
        for value in (np.nan, np.inf, -.1, True, "0.1"):
            frame = self.predictions.copy()
            frame["probability_1"] = frame.probability_1.astype(object)
            frame.loc[0, "probability_1"] = value
            invalid.append(frame)
        invalid.append(self.predictions.drop(columns="probability_10"))
        invalid.append(self.predictions.assign(probability_1=.9))
        for frame in invalid:
            with self.subTest(columns=len(frame.columns)), self.assertRaises(DataContractError):
                select(frame, self.payoffs)

    def test_nonboolean_admission_and_duplicate_keys_or_native_ids_rejected(self):
        altered = []
        frame = self.predictions.copy(); frame["prediction_eligible"] = 1; altered.append(frame)
        frame = self.predictions.copy(); frame.loc[1, "source_security_id"] = frame.loc[0, "source_security_id"]; altered.append(frame)
        frame = self.predictions.copy(); frame.loc[1, "intrinio_security_id"] = frame.loc[0, "intrinio_security_id"]; altered.append(frame)
        frame = self.predictions.copy(); frame.loc[0, "intrinio_security_id"] = None; altered.append(frame)
        for frame in altered:
            with self.assertRaises(DataContractError):
                select(frame, self.payoffs)

    def test_missing_or_noncanonical_signal_and_metadata_rejected(self):
        for field, value in (("signal_date", "2019-12-30"), ("outcome_start", "2020-01-03"), ("outcome_month", "2020-02")):
            frame = self.predictions.copy()
            frame.loc[0, field] = value
            with self.subTest(field=field), self.assertRaises(DataContractError):
                select(frame, self.payoffs)
        frame = self.predictions.loc[self.predictions.signal_date.ne("2019-12-31")]
        with self.assertRaises(DataContractError):
            select(frame, self.payoffs)

    def test_noncanonical_sessions_and_payoff_key_rejected(self):
        sessions = clock_fixture()[2]["selection_sessions"]
        with self.assertRaises(DataContractError):
            select(self.predictions, self.payoffs, sessions=sessions[:-1])
        payoffs = dict(self.payoffs)
        payoffs["2024-01-31"] = next(iter(payoffs.values()))
        with self.assertRaises(DataContractError):
            select(self.predictions, payoffs)

    def test_only_requested_canonical_12_or_48_month_scope_accepted(self):
        for years in ((), (2019,), (2024,), (2020, 2021), (2020.0,), (True,)):
            with self.subTest(years=years), self.assertRaises(DataContractError):
                select(self.predictions, self.payoffs, years=years)
        with self.assertRaises(DataContractError):
            select(self.predictions, self.payoffs, years=(2020, 2021, 2022, 2023))
        frame, payoffs = predictions_fixture(count=2, years=(2020, 2021, 2022, 2023))
        result = select(frame, payoffs, years=(2020, 2021, 2022, 2023))
        self.assertEqual(len(result.monthly), 48)
        self.assertEqual(len(result.selections), 96)
        self.assertEqual(result.rebalances.iloc[-1].execution_date, "2023-12-01")

    def test_frozen_restrictions_and_weights_cannot_be_changed(self):
        for field, value in (("long_weight", 1.0), ("top_k_per_side", 11)):
            cfg = config_fixture()
            cfg["backtest"][field] = value
            with self.assertRaises(DataContractError):
                select(self.predictions, self.payoffs, configuration=cfg)
        cfg = config_fixture()
        cfg["execution_restrictions"]["native_ids"]["new_exclusion"] = "forbidden"
        with self.assertRaises(DataContractError):
            select(self.predictions, self.payoffs, configuration=cfg)

    def test_validator_passes_expected_and_rejects_changed_weights_or_order(self):
        report = validate_ret_walkforward_basket(self.result, self.predictions,
            payoffs=self.payoffs, sessions=clock_fixture()[2]["selection_sessions"],
            configuration=config_fixture(), outcome_years=(2020,))
        self.assertEqual(report["monthly_rebalances"], 12)
        self.assertFalse(report["outcomes_or_future_quote_coverage_used"])
        for field in ("ledger", "selections", "rebalances", "monthly"):
            changed = getattr(self.result, field).iloc[::-1].reset_index(drop=True)
            broken = replace(self.result, **{field: changed})
            with self.subTest(field=field), self.assertRaises(DataContractError):
                validate_ret_walkforward_basket(broken, self.predictions,
                    payoffs=self.payoffs, sessions=clock_fixture()[2]["selection_sessions"],
                    configuration=config_fixture(), outcome_years=(2020,))


if __name__ == "__main__":
    unittest.main()
