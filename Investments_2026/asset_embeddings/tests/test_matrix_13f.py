import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.matrix_13f import (
    CUSIP_POLICY,
    MATRIX_CONTRACT_VERSION,
    MatrixParameters,
    build_quarter_matrix,
    materialize_quarter_matrix,
)


class Matrix13FTests(unittest.TestCase):
    def _write_table(
        self,
        root: Path,
        table: str,
        rows: list[dict[str, object]],
        filename: str,
    ) -> None:
        directory = root / "tables" / table
        directory.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows), directory / filename)

    def _product(self, root: Path) -> Path:
        product = root / "product"
        common = {
            "accession_no": "a1",
            "manager_cik": "0001",
            "form_type": "13F-HR",
            "period_of_report": "2024-03-31",
            "filed_at": "2024-05-10T12:00:00Z",
            "availability_timestamp_utc": "2024-05-10T12:00:00Z",
            "is_amendment": False,
        }
        filings = [
            {
                **common,
                "filed_month": "2024-05",
                "holdings_count": 2,
            },
            {
                **common,
                "accession_no": "a2",
                "form_type": "13F-HR/A",
                "filed_at": "2024-05-20T12:00:00Z",
                "availability_timestamp_utc": "2024-05-20T12:00:00Z",
                "is_amendment": True,
                "filed_month": "2024-05",
                "holdings_count": 1,
            },
        ]
        covers = [
            {
                **common,
                "amendment_info_json": None,
                "report_type": "13F HOLDINGS REPORT",
            },
            {
                **common,
                "accession_no": "a2",
                "form_type": "13F-HR/A",
                "filed_at": "2024-05-20T12:00:00Z",
                "availability_timestamp_utc": "2024-05-20T12:00:00Z",
                "is_amendment": True,
                "amendment_info_json": json.dumps(
                    {"amendmentNo": 1, "amendmentType": "RESTATEMENT"}
                ),
                "report_type": "13F HOLDINGS REPORT",
            },
        ]

        def holding(
            holding_id: str,
            accession: str,
            cusip: str,
            value: float,
        ) -> dict[str, object]:
            return {
                "holding_id": holding_id,
                "accession_no": accession,
                "manager_cik": "0001",
                "period_of_report": "2024-03-31",
                "filed_at": "2024-05-10T12:00:00Z",
                "availability_timestamp_utc": "2024-05-10T12:00:00Z",
                "cusip": cusip,
                "title_of_class": "COM",
                "put_call": None,
                "shares_or_principal_type": "SH",
                "value_usd": value,
                "shares_or_principal_amount": value,
                "reported_issuer_cik": f"issuer-{cusip}",
            }

        holdings = [
            holding("h1", "a1", "000000001", 60.0),
            holding("h2", "a1", "000000002", 40.0),
            {
                **holding("h3", "a2", "000000003", 100.0),
                "filed_at": "2024-05-20T12:00:00Z",
                "availability_timestamp_utc": "2024-05-20T12:00:00Z",
            },
        ]
        self._write_table(product, "filings", filings, "part-202405.parquet")
        self._write_table(product, "cover_pages", covers, "part-202405.parquet")
        self._write_table(product, "holdings", holdings, "part-202405.parquet")
        return product

    def test_quarter_matrix_uses_only_events_visible_at_cutoff(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            product = self._product(Path(temporary))
            result = build_quarter_matrix(
                product,
                report_date="2024-03-31",
                decision_at="2024-05-15T23:59:59Z",
                parameters=MatrixParameters(
                    maximum_position_weight=1.0,
                    minimum_assets_per_manager=1,
                    minimum_managers_per_asset=1,
                    batch_size=1,
                ),
            )
            self.assertEqual(len(result.pairs), 2)
            self.assertTrue(result.pairs["binary_owned"].eq(1).all())
            self.assertTrue(
                result.pairs["last_source_event_available_at_utc"].le(
                    result.decision_at_utc
                ).all()
            )
            self.assertTrue(result.checks["violations"].eq(0).all())
            event_audit = result.event_audit.set_index("event_status")
            self.assertEqual(int(event_audit.loc["future_event", "events"]), 1)

    def test_materialization_is_fingerprinted_and_resumable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            product = self._product(root)
            output = root / "model_inputs"
            parameters = MatrixParameters(
                maximum_position_weight=1.0,
                minimum_assets_per_manager=1,
                minimum_managers_per_asset=1,
                batch_size=2,
            )
            first = materialize_quarter_matrix(
                product,
                output,
                report_date="2024-03-31",
                decision_at="2024-05-15T23:59:59Z",
                input_fingerprint="input-v1",
                source_fingerprint="source-v1",
                parameters=parameters,
            )
            second = materialize_quarter_matrix(
                product,
                output,
                report_date="2024-03-31",
                decision_at="2024-05-15T23:59:59Z",
                input_fingerprint="input-v1",
                source_fingerprint="source-v1",
                parameters=parameters,
            )
            self.assertFalse(first.cache_hit)
            self.assertTrue(second.cache_hit)
            self.assertEqual(first.pair_file_sha256, second.pair_file_sha256)

            manifest_path = (
                output / "report_quarter=2024Q1" / "manifest.json"
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["contract_version"], MATRIX_CONTRACT_VERSION)
            self.assertEqual(manifest["cusip_policy"], CUSIP_POLICY)
            self.assertEqual(manifest["pair_rows"], 2)
            self.assertNotIn("manager_cik", manifest)
            self.assertNotIn("typed_security_id", manifest)

    def test_committed_matrix_notebook_is_output_free_and_public_safe(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        notebook_path = project_root / "notebooks" / "06_build_13f_pit_matrices.ipynb"
        notebook_text = notebook_path.read_text(encoding="utf-8")
        notebook = json.loads(notebook_text)
        code_cells = [
            cell for cell in notebook["cells"] if cell["cell_type"] == "code"
        ]
        self.assertTrue(all(cell.get("execution_count") is None for cell in code_cells))
        self.assertTrue(all(not cell.get("outputs") for cell in code_cells))
        self.assertNotIn("gs://", notebook_text)
        self.assertNotIn("galamboscapital", notebook_text.lower())


if __name__ == "__main__":
    unittest.main()
