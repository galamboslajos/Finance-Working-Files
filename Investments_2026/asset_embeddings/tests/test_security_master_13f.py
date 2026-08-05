import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.security_master_13f import (
    build_cash_share_matrix_candidate,
    classify_security_rows,
    full_history_security_identity_audit,
    mapping_cutoff_summary,
    normalize_cusip,
    security_conflict_audit,
)


class SecurityMaster13FTests(unittest.TestCase):
    def _row(
        self,
        *,
        manager: str,
        cusip: str,
        value: float,
        put_call: str | None = None,
        state_eligible: bool = True,
        accession: str = "a1",
    ) -> dict[str, object]:
        return {
            "holding_id": f"{manager}-{cusip}-{put_call or 'cash'}",
            "accession_no": accession,
            "manager_cik": manager,
            "period_of_report": "2024-03-31",
            "cusip": cusip,
            "reported_issuer_cik": f"issuer-{cusip}",
            "title_of_class": "COM",
            "put_call": put_call,
            "shares_or_principal_type": "SH",
            "value_usd": value,
            "state_model_eligible": state_eligible,
        }

    def test_cusip_normalization_is_narrow_and_format_preserving(self) -> None:
        values = pd.Series([" 12345-ab6 ", "ab@12#cd3", None], dtype="string")
        self.assertEqual(
            normalize_cusip(values).tolist(),
            ["12345AB6", "AB@12#CD3", pd.NA],
        )

    def test_security_rows_keep_typed_channels_and_explicit_reasons(self) -> None:
        holdings = pd.DataFrame.from_records(
            [
                self._row(manager="m1", cusip="000000001", value=100),
                self._row(
                    manager="m1",
                    cusip="000000001",
                    value=20,
                    put_call="CALL",
                ),
                self._row(manager="m2", cusip="BAD", value=50),
                self._row(
                    manager="m3",
                    cusip="000000003",
                    value=10,
                    state_eligible=False,
                ),
            ]
        )
        classified = classify_security_rows(holdings)
        self.assertEqual(
            classified["matrix_eligibility_reason"].tolist(),
            [
                "eligible_reported_cash_share_security",
                "separate_instrument_channel",
                "nonstandard_reported_cusip",
                "ineligible_manager_period_state",
            ],
        )
        self.assertNotEqual(
            classified.iloc[0]["typed_security_id"],
            classified.iloc[1]["typed_security_id"],
        )

    def test_conflict_audit_reports_counts_without_raw_identifiers(self) -> None:
        holdings = pd.DataFrame.from_records(
            [
                self._row(manager="m1", cusip="000000001", value=100),
                self._row(
                    manager="m2",
                    cusip="000000001",
                    value=10,
                    put_call="PUT",
                ),
                {
                    **self._row(manager="m3", cusip="000000001", value=20),
                    "reported_issuer_cik": "different-issuer",
                },
            ]
        )
        classified = classify_security_rows(holdings)
        audit = security_conflict_audit(classified).set_index("metric")
        self.assertEqual(
            audit.loc["CUSIPs spanning multiple instrument channels", "count"],
            1,
        )
        self.assertEqual(
            audit.loc["CUSIPs linked to multiple reported issuer CIKs", "count"],
            1,
        )
        self.assertEqual(list(audit.columns), ["count"])

    def test_date_only_mapping_snapshot_uses_conservative_cutoff(self) -> None:
        mapping = pd.DataFrame(
            {
                "mapping_snapshot_date": ["2024-05-14", "2024-05-15", "2024-05-16", None],
                "rows": [1, 2, 3, 4],
                "resolved_cik_rows": [1, 2, 3, 0],
            }
        )
        summary = mapping_cutoff_summary(
            mapping,
            decision_at="2024-05-15T20:00:00Z",
        ).set_index("mapping_cutoff_status")
        self.assertEqual(summary.loc["usable_before_decision", "rows"], 1)
        self.assertEqual(summary.loc["same_day_time_unknown", "rows"], 2)
        self.assertEqual(summary.loc["future_snapshot", "rows"], 3)
        self.assertEqual(summary.loc["missing_snapshot_date", "rows"], 4)

    def test_matrix_candidate_applies_concentration_and_iterative_degrees(self) -> None:
        rows = [
            self._row(manager="m1", cusip="000000001", value=50),
            self._row(manager="m1", cusip="000000002", value=50),
            self._row(manager="m2", cusip="000000001", value=60),
            self._row(manager="m2", cusip="000000002", value=40),
            self._row(manager="m3", cusip="000000001", value=90),
            self._row(manager="m3", cusip="000000003", value=10),
        ]
        classified = classify_security_rows(pd.DataFrame.from_records(rows))
        candidate = build_cash_share_matrix_candidate(
            classified,
            minimum_assets_per_manager=2,
            minimum_managers_per_asset=2,
        )
        self.assertEqual(len(candidate.positions), 6)
        self.assertEqual(len(candidate.filtered_positions), 4)
        self.assertEqual(candidate.filtered_positions["manager_cik"].nunique(), 2)
        self.assertEqual(candidate.filtered_positions["typed_security_id"].nunique(), 2)
        self.assertTrue(candidate.checks["violations"].eq(0).all())

    def test_full_history_identity_audit_scans_all_rows_and_reuses_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            table = root / "holdings"
            cache = root / "cache"
            table.mkdir()
            rows = [
                {
                    "period_of_report": "2024-03-31",
                    "cusip": "000000001",
                    "reported_issuer_cik": "issuer-1",
                    "title_of_class": "COM",
                    "put_call": None,
                    "shares_or_principal_type": "SH",
                    "value_usd": 100.0,
                },
                {
                    "period_of_report": "2024-03-31",
                    "cusip": "000000001",
                    "reported_issuer_cik": "issuer-2",
                    "title_of_class": "COM",
                    "put_call": "CALL",
                    "shares_or_principal_type": "SH",
                    "value_usd": 20.0,
                },
            ]
            pq.write_table(
                pa.Table.from_pylist(rows),
                table / "part-1.parquet",
            )
            first = full_history_security_identity_audit(
                table,
                cache_directory=cache,
                batch_size=1,
            )
            second = full_history_security_identity_audit(
                table,
                cache_directory=cache,
                batch_size=1,
            )
            self.assertEqual(first.metadata["rows_scanned"], 2)
            self.assertFalse(first.metadata["cache_hit"])
            self.assertTrue(second.metadata["cache_hit"])
            self.assertEqual(int(first.coverage["rows"].sum()), 2)
            conflicts = first.conflicts.set_index("metric")
            self.assertEqual(
                conflicts.loc[
                    "CUSIPs linked to multiple reported issuer CIKs across history",
                    "count",
                ],
                1,
            )

    def test_committed_security_notebook_is_output_free_and_public_safe(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        notebook_path = project_root / "notebooks" / "05_audit_13f_security_master.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        self.assertTrue(all(cell.get("execution_count") is None for cell in code_cells))
        self.assertTrue(all(not cell.get("outputs") for cell in code_cells))
        notebook_text = notebook_path.read_text(encoding="utf-8")
        self.assertNotIn("gs://", notebook_text)
        self.assertNotIn("@", notebook_text)


if __name__ == "__main__":
    unittest.main()
