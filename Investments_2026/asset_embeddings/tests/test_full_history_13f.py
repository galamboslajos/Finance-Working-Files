import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.full_history_13f import (
    bounded_parquet_sample,
    manager_period_multiplicity,
    parquet_schema,
    parquet_table_inventory,
    prepare_13f_filings,
    quarterly_filing_summary,
)


class FullHistory13FTests(unittest.TestCase):
    def _filings_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "accession_no": ["a1", "a2", "a3"],
                "manager_cik": ["m1", "m1", "m2"],
                "form_type": ["13F-HR", "13F-HR/A", "13F-HR"],
                "period_of_report": ["2024-03-31", "2024-03-31", "2024-05-07"],
                "filed_at": [
                    "2024-05-15T12:00:00Z",
                    "2024-05-20T12:00:00Z",
                    "2024-05-01T12:00:00Z",
                ],
                "availability_timestamp_utc": [
                    "2024-05-15T12:00:00Z",
                    "2024-05-20T12:01:00Z",
                    "2024-04-30T12:00:00Z",
                ],
                "is_amendment": [False, True, False],
                "holdings_count": [10, 11, 12],
            }
        )

    def test_prepare_filings_preserves_and_flags_timing_states(self) -> None:
        prepared = prepare_13f_filings(self._filings_frame())
        self.assertEqual(len(prepared), 3)
        self.assertEqual(prepared["is_standard_quarter_end"].tolist(), [True, True, False])
        self.assertEqual(prepared["report_not_after_filing"].tolist(), [True, True, False])
        self.assertEqual(prepared["available_not_before_filing"].tolist(), [True, True, False])
        self.assertEqual(prepared["is_amendment_flag"].tolist(), [0, 1, 0])
        self.assertAlmostEqual(prepared.loc[1, "availability_delay_minutes"], 1.0)

    def test_quarterly_summary_does_not_silently_resolve_amendments(self) -> None:
        prepared = prepare_13f_filings(self._filings_frame())
        summary = quarterly_filing_summary(prepared)
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary.iloc[0]["filings"], 2)
        self.assertEqual(summary.iloc[0]["managers"], 1)
        self.assertEqual(summary.iloc[0]["amendments"], 1)
        self.assertEqual(summary.iloc[0]["holding_lines_reported"], 21)

    def test_manager_period_multiplicity_reports_conflicts_without_ids(self) -> None:
        prepared = prepare_13f_filings(self._filings_frame())
        distribution = manager_period_multiplicity(prepared).set_index(
            "filings_per_manager_period"
        )
        self.assertEqual(distribution.loc[2, "manager_periods"], 1)
        self.assertNotIn("manager_cik", distribution.columns)

    def test_parquet_inventory_and_schema_use_footers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            table_dir = root / "tables" / "holdings"
            table_dir.mkdir(parents=True)
            pq.write_table(pa.table({"asset": ["a", "b"], "value": [1.0, 2.0]}), table_dir / "p1.parquet")
            pq.write_table(pa.table({"asset": ["c"], "value": [3.0]}), table_dir / "p2.parquet")

            inventory = parquet_table_inventory(root).set_index("table")
            self.assertEqual(inventory.loc["holdings", "parquet_files"], 2)
            self.assertEqual(inventory.loc["holdings", "rows"], 3)
            self.assertEqual(inventory.loc["holdings", "schema_variants"], 1)

            schema = parquet_schema(table_dir).set_index("column")
            self.assertTrue(schema.loc["asset", "all_files"])
            self.assertEqual(schema.loc["value", "files_present"], 2)

    def test_bounded_sample_is_deterministic_and_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            table_dir = Path(temporary)
            for index in range(4):
                pq.write_table(
                    pa.table(
                        {
                            "partition": [index] * 5,
                            "row": list(range(5)),
                        }
                    ),
                    table_dir / f"p{index}.parquet",
                )

            sample, design = bounded_parquet_sample(
                table_dir,
                ["partition", "row"],
                max_files=3,
                rows_per_file=2,
            )
            self.assertEqual(len(sample), 6)
            self.assertEqual(set(sample["partition"]), {0, 2, 3})
            self.assertEqual(design.iloc[0]["sampled_partitions"], 3)

    def test_committed_notebook_is_output_free_and_contains_no_cloud_uri(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        notebook_path = project_root / "notebooks" / "02_explore_13f_full_history.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        self.assertTrue(all(cell.get("execution_count") is None for cell in code_cells))
        self.assertTrue(all(not cell.get("outputs") for cell in code_cells))
        notebook_text = notebook_path.read_text(encoding="utf-8")
        self.assertNotIn("gs://", notebook_text)
        self.assertNotIn("@", notebook_text)


if __name__ == "__main__":
    unittest.main()
