"""Cross-worker regressions use artificial observations, never research data."""
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from dmv2.classification import classification_training_slice
from dmv2.features import FeaturePanel, predictor_frame, registered_feature_specs
from dmv2.paper_features import PAPER_FEATURE_NAMES
from dmv2.preprocessing import fit_preprocessor
from dmv2.ret import build_ret_payoffs, score_ret
from dmv2.targets import build_monthly_targets
from dmv2.universe import DataContractError


def _month_fixture(months=("2010-01", "2010-02"), *, missing=(), future_shift=0,
                   outcome_unrequested=()):
    """Twelve prediction names plus two retained original exclusions per month."""
    specs = registered_feature_specs()
    cohort, outcomes, values, windows = [], [], [], {}
    for number, month in enumerate(months):
        period = pd.Period(month, "M")
        signal = (period - 1).end_time.date().isoformat()
        window = {"outcome_month": month, "outcome_start": period.start_time.date().isoformat(),
                  "outcome_end": period.end_time.date().isoformat()}
        windows[signal] = window
        for i in range(14):
            key = {"source_security_id": f"s{i:02d}", "signal_date": signal}
            admission = {"identity_eligible": i != 12, "requested": i != 12,
                         "paper_prediction_eligible": i < 12}
            cohort.append({**key, **admission, "universe": "combined"})
            value = .12 - .01 * i
            if number:
                value += future_shift
            unavailable = (month, i) in missing
            forward_requested = i != 12 and (month, i) not in outcome_unrequested
            outcomes.append({**key, **window,
                             "requested": forward_requested,
                             "provider_adjusted_return": None if unavailable or not forward_requested else value,
                             "return_status": ("not_requested" if not forward_requested else
                                               "missing_end" if unavailable else "ok")})
            values.append({**key, **{spec["name"]:
                (float(i + 1 + number * 1000) if i < 12 else None) for spec in specs}})
    panel = FeaturePanel(pa.Table.from_pylist(values), pa.Table.from_pylist(cohort), specs, {}, {},
                         {"target_windows": windows})
    target = build_monthly_targets(pa.Table.from_pylist(cohort), pa.Table.from_pylist(outcomes),
                                  label_observed_through=max(w["outcome_end"] for w in windows.values()),
                                  month_windows=windows)
    return panel, target


def _slice(panel, targets, **changes):
    params = {"train_start": "2010-01-01", "train_end": "2010-02-28",
              "label_observed_through": "2010-01-31", "columns": list(PAPER_FEATURE_NAMES)}
    params.update(changes)
    return classification_training_slice(panel, targets, **params)


class SharedClassificationIntegrationTests(unittest.TestCase):
    def test_adapter_uses_canonical_classes_and_preserves_unlabelled_original_rows(self):
        panel, target = _month_fixture()
        result = _slice(panel, target)
        self.assertEqual(len(result.X), 14)
        self.assertEqual(result.y.name, "next_month_return_class")
        self.assertEqual(set(result.y.dropna()), set(range(1, 11)))
        self.assertEqual(result.y.iloc[0], 1)
        self.assertNotAlmostEqual(result.y.iloc[0], .12)
        self.assertEqual(result.ledger.target_available.tolist(), result.y.notna().tolist())
        self.assertEqual(result.ledger.paper_prediction_eligible.sum(), 12)
        self.assertIn("return_available", result.ledger)
        self.assertFalse(result.y.iloc[12:].notna().any())
        self.assertTrue(result.provenance["target_statuses_do_not_select_features"])
        self.assertFalse(result.provenance["knowledge_time_verified"])

    def test_extra_feature_missingness_and_missing_return_do_not_select_prediction_rows(self):
        panel, target = _month_fixture()
        baseline = _slice(panel, target)
        missing_panel, missing_target = _month_fixture(missing=(("2010-01", 0),))
        result = _slice(missing_panel, missing_target)
        pd.testing.assert_frame_equal(baseline.X, result.X)
        pd.testing.assert_series_equal(baseline.ledger.paper_prediction_eligible,
                                       result.ledger.paper_prediction_eligible)
        self.assertTrue(pd.isna(result.y.iloc[0]))
        self.assertEqual(len(result.X), 14)
        rows = panel.table.to_pylist()
        rows[1]["px_simple_return_1s"] = None
        expanded = replace(panel, table=pa.Table.from_pylist(rows))
        extra = _slice(expanded, target, columns=[*PAPER_FEATURE_NAMES, "px_simple_return_1s"])
        self.assertEqual(len(extra.X), len(baseline.X))
        self.assertTrue(extra.ledger.paper_prediction_eligible.iloc[1])
        pd.testing.assert_series_equal(extra.y, baseline.y)

    def test_future_rows_are_removed_before_feature_conversion_and_do_not_change_training(self):
        panel, target = _month_fixture()
        baseline = _slice(panel, target)
        _, changed_targets = _month_fixture(future_shift=1e10)
        seen = []

        def inspect(selected, **kwargs):
            seen.extend(selected.table["signal_date"].to_pylist())
            return predictor_frame(selected, **kwargs)

        with patch("dmv2.classification.predictor_frame", side_effect=inspect):
            changed = _slice(panel, changed_targets)
        self.assertEqual(set(seen), {"2009-12-31"})
        pd.testing.assert_frame_equal(baseline.X, changed.X)
        pd.testing.assert_series_equal(baseline.y, changed.y)
        pd.testing.assert_frame_equal(baseline.ledger, changed.ledger)

    def test_shared_class_slice_fits_existing_preprocessor_only_on_explicit_label_mask(self):
        panel, target = _month_fixture(missing=(("2010-01", 0),))
        result = _slice(panel, target)
        fitted = fit_preprocessor(result, fit_mask=result.ledger.target_available)
        self.assertEqual(fitted.manifest["fit_rows"], 11)
        prepared = fitted.transform(result.X, prediction_mask=result.ledger.paper_prediction_eligible)
        self.assertEqual(prepared.ledger.transformed.sum(), 12)
        self.assertTrue(prepared.ledger.transformed.iloc[0])  # Missing future label is still predictable.

    def test_observed_return_is_not_a_class_when_month_has_too_few_outcomes(self):
        panel, target = _month_fixture(missing=tuple(("2010-01", i) for i in range(4)))
        result = _slice(panel, target)
        self.assertEqual(result.ledger.return_available.sum(), 8)
        self.assertEqual(result.ledger.target_available.sum(), 0)
        self.assertTrue(result.y.isna().all())
        self.assertEqual(result.ledger.paper_prediction_eligible.sum(), 12)
        with self.assertRaises(DataContractError):
            fit_preprocessor(result, fit_mask=result.ledger.return_available)

    def test_forward_outcome_requests_are_distinct_from_past_feature_requests(self):
        panel, baseline_target = _month_fixture()
        baseline = _slice(panel, baseline_target)
        for member in (0, 13):  # Include an original member outside the Paper15 cohort.
            changed_panel, partial_target = _month_fixture(outcome_unrequested=(("2010-01", member),))
            result = _slice(changed_panel, partial_target)
            with self.subTest(member=member):
                pd.testing.assert_frame_equal(baseline.X, result.X)
                pd.testing.assert_series_equal(baseline.ledger.paper_prediction_eligible,
                                               result.ledger.paper_prediction_eligible)
                self.assertTrue(result.ledger.requested.iloc[member])  # Past feature request remains true.
                self.assertFalse(result.ledger.outcome_requested.iloc[member])
                self.assertTrue(result.y.isna().all())
                self.assertEqual(result.ledger.target_available.sum(), 0)

    def test_unused_future_feature_and_class_values_are_not_converted_or_verified(self):
        panel, target = _month_fixture()
        baseline = _slice(panel, target)
        future_feature = pc.greater(panel.table["signal_date"], "2009-12-31")
        changed_feature_table = panel.table.set_column(2, panel.table.column_names[2],
            pc.if_else(future_feature, float("inf"), panel.table.column(2)))
        future_target = pc.greater(target.table["outcome_month"], "2010-01")
        changed_target_table = target.table.set_column(
            target.table.schema.get_field_index("target_class"), "target_class",
            pc.if_else(future_target, pa.scalar(0, pa.int8()), target.table["target_class"]))
        changed = _slice(replace(panel, table=changed_feature_table),
                         replace(target, table=changed_target_table))
        pd.testing.assert_frame_equal(baseline.X, changed.X)
        pd.testing.assert_series_equal(baseline.y, changed.y)
        self.assertEqual(baseline.provenance, changed.provenance)

    def test_missing_members_extra_members_and_admission_mismatch_fail_before_join(self):
        panel, targets = _month_fixture()
        for change in ("missing_feature", "missing_coverage", "changed_key", "changed_admission"):
            if change == "missing_feature":
                changed = replace(panel, table=panel.table.slice(1))
            elif change == "missing_coverage":
                changed = replace(panel, coverage=panel.coverage.slice(1))
            elif change == "changed_key":
                rows = panel.table.to_pylist()
                rows[0]["source_security_id"] = "new_member"
                changed = replace(panel, table=pa.Table.from_pylist(rows))
            else:
                rows = panel.coverage.to_pylist()
                rows[0]["paper_prediction_eligible"] = False
                changed = replace(panel, coverage=pa.Table.from_pylist(rows))
            with self.subTest(change=change), self.assertRaises(DataContractError):
                _slice(changed, targets)

    def test_paper_eligibility_must_agree_with_finite_past_inputs_not_extra_complete_cases(self):
        panel, targets = _month_fixture()
        rows = panel.table.to_pylist()
        rows[0][PAPER_FEATURE_NAMES[0]] = None
        with self.assertRaises(DataContractError):
            _slice(replace(panel, table=pa.Table.from_pylist(rows)), targets)
        with self.assertRaises(DataContractError):
            _slice(panel, targets, columns=["px_simple_return_1s"])


def _payoff_fixture(*, omit_month=None):
    """First 60 months have ten stocks; later 60 have twenty stocks."""
    cohort, outcomes, windows = [], [], {}
    for offset in range(120):
        period = pd.Period("2010-01", "M") + offset
        month = str(period)
        if month == omit_month:
            continue
        signal = (period - 1).end_time.date().isoformat()
        window = {"outcome_month": month, "outcome_start": period.start_time.date().isoformat(),
                  "outcome_end": period.end_time.date().isoformat()}
        windows[signal] = window
        n = 10 if offset < 60 else 20
        for i in range(n):
            key = {"source_security_id": f"s{i:02d}", "signal_date": signal}
            cohort.append({**key, "identity_eligible": True, "requested": True,
                           "paper_prediction_eligible": True, "universe": "combined"})
            # Repeated pairs share an exact tie in the larger cross-section.
            # Each period still has ten nonempty realized classes.
            class_id = i + 1 if n == 10 else i // 2 + 1
            value = .1 - .01 * class_id + (.2 if offset >= 60 else 0)
            outcomes.append({**key, **window, "requested": True,
                             "return_status": "ok", "provider_adjusted_return": value})
    return build_monthly_targets(pa.Table.from_pylist(cohort), pa.Table.from_pylist(outcomes),
                                 label_observed_through="2019-12-31", month_windows=windows)


class SharedRETIntegrationTests(unittest.TestCase):
    def test_realized_classes_use_equal_month_weights_and_explicit_probability_mapping(self):
        targets = _payoff_fixture()
        payoff = build_ret_payoffs(targets, signal_date="2019-12-31",
                                   label_observed_through="2019-12-31")
        self.assertTrue(payoff.provenance["vector_available"])
        actual = np.asarray(payoff.table["payoff"].to_pylist())
        np.testing.assert_allclose(actual, [.2 - .01 * k for k in range(1, 11)], atol=1e-14)
        # Pooling stock rows would overweight the larger, later cross-sections.
        self.assertNotAlmostEqual(actual[0], (.09 + 2 * .29) / 3)
        classes = [10, 8, 6, 4, 2, 9, 7, 5, 3, 1]
        index = pd.MultiIndex.from_tuples([("missing_label_name", "2019-12-31"),
                                          ("another_name", "2019-12-31")],
                                         names=["source_security_id", "signal_date"])
        probs = np.zeros((2, 10))
        probs[0, classes.index(1)] = 1
        probs[1, classes.index(10)] = 1
        result = score_ret(pd.DataFrame(probs, index=index), payoff, class_ids=classes,
                           signal_date="2019-12-31", label_observed_through="2019-12-31")
        self.assertTrue(result.index.equals(index))
        np.testing.assert_allclose(result.ret_score, [actual[0], actual[-1]], atol=1e-14)

    def test_missing_month_withholds_every_payoff_and_score_without_dropping_prediction_rows(self):
        payoff = build_ret_payoffs(_payoff_fixture(omit_month="2011-05"),
                                   signal_date="2019-12-31", label_observed_through="2019-12-31")
        self.assertFalse(payoff.provenance["vector_available"])
        self.assertEqual(payoff.monthly.num_rows, 1200)
        self.assertEqual(payoff.table["payoff"].null_count, 10)
        result = score_ret(pd.DataFrame(np.full((3, 10), .1), index=["a", "b", "c"]),
                           payoff, class_ids=list(range(1, 11)), signal_date="2019-12-31",
                           label_observed_through="2019-12-31")
        self.assertEqual(list(result.index), ["a", "b", "c"])
        self.assertTrue(result.ret_score.isna().all())


if __name__ == "__main__":
    unittest.main()
