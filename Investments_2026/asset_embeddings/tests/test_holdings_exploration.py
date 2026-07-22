import json
import unittest
from pathlib import Path

import pandas as pd

from src.holdings_exploration import (
    DATASET_SPECS,
    duplicate_profile,
    filing_conflict_profile,
    iterative_bipartite_filter,
    matrix_readiness,
    schema_profile,
    timing_profile,
    value_profile,
)


class HoldingsExplorationTests(unittest.TestCase):
    def test_committed_notebook_contains_no_private_outputs(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        notebook_path = project_root / "notebooks" / "01_explore_13f_nport.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        self.assertTrue(all(cell.get("execution_count") is None for cell in code_cells))
        self.assertTrue(all(not cell.get("outputs") for cell in code_cells))

    def test_schema_profile_preserves_missing_state(self) -> None:
        frame = pd.DataFrame({"asset": ["A", None, "B"], "value": [1.0, 2.0, None]})
        profile = schema_profile(frame).set_index("column")
        self.assertEqual(profile.loc["asset", "null_rows"], 1)
        self.assertAlmostEqual(profile.loc["value", "null_pct"], 100 / 3, places=3)

    def test_timing_profile_flags_availability_and_chronology_violations(self) -> None:
        spec = DATASET_SPECS["13f"]
        frame = pd.DataFrame(
            {
                spec.report_date_column: ["2024-03-31", "2024-06-30"],
                spec.filed_at_column: ["2024-05-15T12:00:00Z", "2024-05-15T12:00:00Z"],
                spec.availability_column: ["2024-05-15T12:00:00Z", "2024-05-14T12:00:00Z"],
            }
        )
        profile = timing_profile(frame, spec)
        pit_row = profile.loc[profile["role"] == "PIT check"].iloc[0]
        chronology_row = profile.loc[profile["role"] == "chronology check"].iloc[0]
        self.assertEqual(pit_row["violation_rows"], 1)
        self.assertEqual(chronology_row["violation_rows"], 1)

    def test_duplicate_profile_uses_filing_and_ordinal(self) -> None:
        spec = DATASET_SPECS["nport"]
        frame = pd.DataFrame(
            {
                spec.holding_id_column: ["h1", "h1", "h2"],
                spec.filing_column: ["f1", "f1", "f1"],
                spec.holding_ordinal_column: [1, 1, 2],
            }
        )
        profile = duplicate_profile(frame, spec).set_index("key")
        self.assertEqual(profile.loc["holding_id", "duplicate_rows"], 2)
        self.assertEqual(profile.loc["filing_and_ordinal", "duplicate_rows"], 2)

    def test_value_profile_keeps_negative_zero_and_missing_distinct(self) -> None:
        spec = DATASET_SPECS["nport"]
        frame = pd.DataFrame({spec.value_column: [-2.0, 0.0, 3.0, None]})
        profile = value_profile(frame, spec).iloc[0]
        self.assertEqual(profile["negative"], 1)
        self.assertEqual(profile["zero"], 1)
        self.assertEqual(profile["positive"], 1)
        self.assertEqual(profile["missing"], 1)

    def test_iterative_filter_converges_on_both_sides(self) -> None:
        pairs = pd.DataFrame(
            {
                "investor": ["i1", "i1", "i2", "i2", "i3"],
                "asset": ["a1", "a2", "a1", "a2", "a3"],
            }
        )
        filtered, log = iterative_bipartite_filter(
            pairs,
            investor_column="investor",
            asset_column="asset",
            minimum_assets_per_investor=2,
            minimum_investors_per_asset=2,
        )
        self.assertEqual(set(filtered["investor"]), {"i1", "i2"})
        self.assertEqual(set(filtered["asset"]), {"a1", "a2"})
        self.assertEqual(log.iloc[-1]["pairs_before"], log.iloc[-1]["pairs_after"])

    def test_matrix_readiness_reports_every_filter_stage(self) -> None:
        spec = DATASET_SPECS["nport"]
        frame = pd.DataFrame(
            {
                spec.report_date_column: ["2024-03-31"] * 5,
                spec.investor_column: ["i1", "i1", "i2", "i2", "i3"],
                spec.filing_column: ["f1", "f1", "f2", "f2", "f3"],
                "cusip": ["a1", "a2", "a1", "a2", "a3"],
                spec.value_column: [1.0, 2.0, 1.0, -2.0, 5.0],
                spec.category_column: ["EC", "EC", "EC", "EC", "DB"],
            }
        )
        summary, audit, _ = matrix_readiness(
            frame,
            spec,
            asset_column="cusip",
            positive_values_only=True,
            category_filter=("asset_category", ("EC",)),
            minimum_assets_per_investor=1,
            minimum_investors_per_asset=1,
        )
        self.assertEqual(summary.iloc[0]["pairs"], 3)
        self.assertEqual(
            list(audit["stage"]),
            [
                "input",
                "valid report/investor/asset",
                "one filing per investor-report",
                "positive holding value",
                "asset_category in ('EC',)",
                "aggregated investor-asset pairs",
            ],
        )

    def test_matrix_readiness_rejects_blank_investor_ids(self) -> None:
        spec = DATASET_SPECS["nport"]
        frame = pd.DataFrame(
            {
                spec.report_date_column: ["2024-03-31", "2024-03-31"],
                spec.investor_column: ["series-1", ""],
                spec.filing_column: ["f1", "f2"],
                "cusip": ["asset-1", "asset-2"],
                spec.value_column: [1.0, 2.0],
            }
        )
        summary, audit, _ = matrix_readiness(
            frame,
            spec,
            asset_column="cusip",
            positive_values_only=False,
            minimum_assets_per_investor=1,
            minimum_investors_per_asset=1,
        )
        self.assertEqual(summary.iloc[0]["pairs"], 1)
        self.assertEqual(audit.iloc[1]["rows"], 1)

    def test_matrix_readiness_refuses_unresolved_filing_conflicts(self) -> None:
        spec = DATASET_SPECS["13f"]
        frame = pd.DataFrame(
            {
                spec.report_date_column: ["2024-03-31", "2024-03-31"],
                spec.investor_column: ["manager-1", "manager-1"],
                spec.filing_column: ["original", "amendment"],
                "cusip": ["asset-1", "asset-1"],
                spec.value_column: [1.0, 2.0],
            }
        )
        conflicts = filing_conflict_profile(frame, spec)
        self.assertEqual(conflicts.iloc[0]["filings"], 2)
        with self.assertRaisesRegex(ValueError, "Resolve amendments or versions explicitly"):
            matrix_readiness(
                frame,
                spec,
                asset_column="cusip",
                positive_values_only=False,
                minimum_assets_per_investor=1,
                minimum_investors_per_asset=1,
            )


if __name__ == "__main__":
    unittest.main()
