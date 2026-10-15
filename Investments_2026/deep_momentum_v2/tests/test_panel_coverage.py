"""Streaming audits retain historical denominators without consulting labels."""
from types import SimpleNamespace
import copy
import unittest

import numpy as np
import pandas as pd
import pyarrow as pa

from dmv2.features import registered_feature_specs
from dmv2.panel_coverage import MISSING_REASONS, StreamingPanelCoverage
from dmv2.paper_features import COMMON_MOMENTUM_POLICY, PAPER_FEATURE_NAMES
from dmv2.universe import DataContractError


def fixture(signals=("2009-12-31", "2019-12-31")):
    specs = registered_feature_specs()
    names = [s["name"] for s in specs]
    rows, ledgers, members, statuses = [], [], [], []
    technical = [s["name"] for s in specs if s["role"] == "technical"]
    for signal in signals:
        for i in range(3):
            identity, requested = i != 2, i != 2
            key = dict(source_security_id=f"source-{i}", signal_date=signal)
            row = {**key, **{name: float(i + 1) for name in names}}
            if i == 1:
                row[technical[0]] = np.nan
            if i == 2:
                for name in names:
                    row[name] = np.nan
            rows.append(row)
            ledgers.append({**key, "intrinio_security_id": f"native-{i}",
                "identity_eligible": identity, "requested": requested,
                "paper_prediction_eligible": i != 2})
            members.append({**key, "indices": ["sp500" if i != 1 else "sp400"]})
            statuses.append(dict(intrinio_security_id=f"native-{i}", signal_date=signal,
                **{name: 1 if i == 1 and name == technical[0] else 0 for name in technical}))
    panel = SimpleNamespace(table=pa.Table.from_pandas(pd.DataFrame(rows), preserve_index=False),
        coverage=pa.Table.from_pylist(ledgers), specs=specs, provenance={"outcomes_consumed": False},
        diagnostics={"daily_status_codes": pd.DataFrame(statuses).astype({name: "uint8" for name in technical}),
            "daily_status_legend": {0: "ok", 1: "missing_session"}})
    return panel, pd.DataFrame(members)


class PanelCoverageTests(unittest.TestCase):
    def test_outcome_year_denominators_qualified_subset_and_reason_partition(self):
        panel, members = fixture()
        audit = StreamingPanelCoverage(panel.specs, ("2009-12-31", "2019-12-31"))
        report = audit.add_panel(panel, original_membership=members).finish()
        self.assertEqual(report["summary"]["original_stock_months"], 6)
        self.assertEqual(report["summary"]["paper_qualified_stock_months"], 4)
        self.assertEqual(report["summary"]["candidate_predictors"], 715)
        yearly = report["tables"]["by_feature_year"]
        self.assertEqual(set(yearly.outcome_year), {2010, 2020})
        first = next(s["name"] for s in panel.specs if s["role"] == "technical")
        selected = yearly[(yearly.outcome_year == 2010) & (yearly.feature_name == first) & (yearly.universe == "combined")]
        original = selected[selected.population == "original"].iloc[0]
        qualified = selected[selected.population == "paper_qualified"].iloc[0]
        self.assertEqual((original.rows, original.finite_rows, original.missing_rows), (3, 1, 2))
        self.assertEqual((qualified.rows, qualified.finite_rows, qualified.missing_rows), (2, 1, 1))
        self.assertEqual((original.missing_identity_blocked, original.missing_missing_exact_session), (1, 1))
        self.assertTrue(np.array_equal(yearly.rows, yearly.finite_rows + yearly[["missing_" + n for n in MISSING_REASONS]].sum(axis=1)))
        family = report["tables"]["by_family_year"]
        combined = family[(family.universe == "combined") & (family.population == "original")]
        self.assertEqual(combined.feature_value_slots.sum(), 6 * 728)
        self.assertFalse(report["summary"]["outcomes_consumed"])
        self.assertEqual(report["summary"]["fits_performed"], 0)

    def test_streamed_batches_match_annual_batch_and_do_not_fit(self):
        panel, members = fixture()
        together = StreamingPanelCoverage(panel.specs, ("2009-12-31", "2019-12-31"))
        together.add_panel(panel, original_membership=members)
        separate = StreamingPanelCoverage(panel.specs, ("2009-12-31", "2019-12-31"))
        for signal in ("2019-12-31", "2009-12-31"):
            one, original = fixture((signal,))
            separate.add_signal(one, original_membership=original)
        first, second = together.finish(), separate.finish()
        self.assertEqual(first["summary"], second["summary"])
        pd.testing.assert_frame_equal(first["tables"]["by_feature_year"], second["tables"]["by_feature_year"])

    def test_missing_signal_partial_checkpoint_and_duplicate_refusal(self):
        panel, members = fixture(("2009-12-31",))
        audit = StreamingPanelCoverage(panel.specs, ("2009-12-31", "2010-01-29"))
        audit.add_panel(panel, original_membership=members)
        with self.assertRaisesRegex(DataContractError, "incomplete"):
            audit.finish()
        self.assertEqual(audit.finish(require_complete=False)["summary"]["status"], "PARTIAL_FEATURE_COVERAGE_CHECKPOINT")
        with self.assertRaisesRegex(DataContractError, "repeated"):
            audit.add_panel(panel, original_membership=members)
        self.assertEqual(audit.finish(require_complete=False)["summary"]["original_stock_months"], 3)

    def test_source_members_cannot_disappear_and_sleeves_remain_explicit(self):
        panel, members = fixture(("2009-12-31",))
        for bad in (members.iloc[:2], members.assign(indices=[[], ["sp400"], ["sp500"]])):
            with self.subTest(bad=bad), self.assertRaises(DataContractError):
                StreamingPanelCoverage(panel.specs, ("2009-12-31",)).add_panel(panel, original_membership=bad)

    def test_future_target_metadata_and_changed_dictionary_are_refused(self):
        panel, members = fixture(("2009-12-31",))
        bad = copy.deepcopy(panel)
        bad.table = bad.table.append_column("target_return", pa.array([1., 2., 3.]))
        with self.assertRaisesRegex(DataContractError, "target/audit"):
            StreamingPanelCoverage(panel.specs, ("2009-12-31",)).add_panel(bad, original_membership=members)
        bad = copy.deepcopy(panel)
        bad.provenance["outcomes_consumed"] = True
        with self.assertRaisesRegex(DataContractError, "outcome-free"):
            StreamingPanelCoverage(panel.specs, ("2009-12-31",)).add_panel(bad, original_membership=members)
        bad = copy.deepcopy(panel)
        bad.specs = tuple({**s, "formula": "changed"} if i == 0 else s for i, s in enumerate(bad.specs))
        with self.assertRaisesRegex(DataContractError, "dictionary changed"):
            StreamingPanelCoverage(panel.specs, ("2009-12-31",)).add_panel(bad, original_membership=members)
        with self.assertRaisesRegex(DataContractError, "outcome years"):
            StreamingPanelCoverage(panel.specs, ("2023-12-29",))

    def test_paper_eligibility_cannot_look_at_targets_or_disagree_with_values(self):
        panel, members = fixture(("2009-12-31",))
        coverage = panel.coverage.to_pandas()
        coverage.loc[0, "paper_prediction_eligible"] = False
        panel.coverage = pa.Table.from_pandas(coverage, preserve_index=False)
        with self.assertRaisesRegex(DataContractError, "all-fifteen"):
            StreamingPanelCoverage(panel.specs, ("2009-12-31",)).add_panel(panel, original_membership=members)

    def test_common_five_policy_cannot_admit_partly_normalized_paper_inputs(self):
        panel, members = fixture(("2009-12-31",))
        panel.specs = tuple({**s, "normalization_policy": COMMON_MOMENTUM_POLICY} if s["role"] == "paper" else s for s in panel.specs)
        frame = panel.table.to_pandas()
        frame.loc[0, PAPER_FEATURE_NAMES[0]] = np.nan
        panel.table = pa.Table.from_pandas(frame, preserve_index=False)
        with self.assertRaisesRegex(DataContractError, "all fifteen"):
            StreamingPanelCoverage(panel.specs, ("2009-12-31",)).add_panel(panel, original_membership=members)

    def test_audit_does_not_accept_changed_formulas_or_financial_membership(self):
        panel, members = fixture(("2009-12-31",))
        changed = tuple({**s, "formula": "target_return"} if i == 0 else s for i, s in enumerate(panel.specs))
        with self.assertRaisesRegex(DataContractError, "frozen feature definition"):
            StreamingPanelCoverage(changed, ("2009-12-31",))
        with self.assertRaisesRegex(DataContractError, "outcome/target"):
            StreamingPanelCoverage(panel.specs, ("2009-12-31",)).add_panel(panel, original_membership=members.assign(outcome_month="2010-01"))

    def test_context_missingness_stays_a_broad_category(self):
        panel, members = fixture(("2009-12-31",))
        name = next(s["name"] for s in panel.specs if s["role"] == "relative_context")
        frame = panel.table.to_pandas()
        frame.loc[0, name] = np.nan
        panel.table = pa.Table.from_pandas(frame, preserve_index=False)
        panel.diagnostics["context"] = pd.DataFrame([dict(signal_date="2009-12-31", feature_name=name,
            status="incomplete_calendar_pairs;ok", latest_input_date="2009-12-31")])
        result = StreamingPanelCoverage(panel.specs, ("2009-12-31",)).add_panel(panel, original_membership=members).finish()
        part = result["tables"]["by_feature_year"]
        original = part[(part.feature_name == name) & (part.population == "original") & (part.universe == "combined")].iloc[0]
        self.assertEqual(original.missing_context_incomplete_or_undefined, 1)
        states = result["tables"]["worker_status_counts"]
        self.assertEqual(states.iloc[0].status, "incomplete_calendar_pairs;ok")
        self.assertEqual(states.iloc[0].diagnostic_records, 1)

    def test_future_diagnostic_clock_refused_before_any_counts_accumulate(self):
        panel, members = fixture(("2009-12-31",))
        panel.diagnostics["context"] = pd.DataFrame([dict(signal_date="2009-12-31", status="ok", latest_input_date="2010-01-04")])
        audit = StreamingPanelCoverage(panel.specs, ("2009-12-31",))
        with self.assertRaisesRegex(DataContractError, "clock exceeds"):
            audit.add_panel(panel, original_membership=members)
        self.assertEqual(audit.finish(require_complete=False)["summary"]["completed_signals"], 0)

    def test_unknown_codes_or_value_status_disagreement_fail_closed(self):
        for value in (9, 0):
            panel, members = fixture(("2009-12-31",))
            first = next(s["name"] for s in panel.specs if s["role"] == "technical")
            panel.diagnostics["daily_status_codes"].loc[1, first] = value
            with self.subTest(code=value), self.assertRaises(DataContractError):
                StreamingPanelCoverage(panel.specs, ("2009-12-31",)).add_panel(panel, original_membership=members)

    def test_original_membership_row_order_does_not_realign_features_incorrectly(self):
        panel, members = fixture(("2009-12-31",))
        normal = StreamingPanelCoverage(panel.specs, ("2009-12-31",)).add_panel(panel, original_membership=members).finish()
        panel.coverage = panel.coverage.take(pa.array([2, 0, 1]))
        shuffled = StreamingPanelCoverage(panel.specs, ("2009-12-31",)).add_panel(panel, original_membership=members.iloc[::-1]).finish()
        pd.testing.assert_frame_equal(normal["tables"]["by_feature_year"], shuffled["tables"]["by_feature_year"])


if __name__ == "__main__":
    unittest.main()
