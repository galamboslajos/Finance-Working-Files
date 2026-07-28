import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.data_spine_13f import (
    amendment_quality_summary,
    amendment_sequence_summary,
    classify_instrument_rows,
    full_history_holdings_audit,
    manager_identity_audit,
    prepare_amendment_events,
    quarter_coverage_audit,
    standard_13f_deadline,
)


class DataSpine13FTests(unittest.TestCase):
    def _cover_pages(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "accession_no": ["a1", "a2", "a3"],
                "manager_cik": ["0001", "0001", "0001"],
                "manager_name": ["Manager A", "Manager A", "Manager A"],
                "form_type": ["13F-HR", "13F-HR/A", "13F-HR/A"],
                "period_of_report": ["2024-03-31"] * 3,
                "filed_at": [
                    "2024-05-10T12:00:00Z",
                    "2024-05-20T12:00:00Z",
                    "2024-06-01T12:00:00Z",
                ],
                "availability_timestamp_utc": [
                    "2024-05-10T12:00:00Z",
                    "2024-05-20T12:00:00Z",
                    "2024-06-01T12:00:00Z",
                ],
                "is_amendment": [False, True, True],
                "amendment_info_json": [
                    None,
                    json.dumps(
                        {"amendmentNo": 1, "amendmentType": "NEW HOLDINGS"}
                    ),
                    json.dumps(
                        {"amendmentNo": 2, "amendmentType": "RESTATEMENT"}
                    ),
                ],
                "report_type": ["13F HOLDINGS REPORT"] * 3,
                "crd_number": ["100"] * 3,
                "sec_file_number": ["801-1"] * 3,
                "form_13f_file_number": ["028-1"] * 3,
            }
        )

    def test_amendment_events_preserve_supplement_and_restatement_roles(self) -> None:
        events = prepare_amendment_events(self._cover_pages())
        self.assertEqual(
            events["event_role"].tolist(),
            ["ORIGINAL", "NEW_HOLDINGS", "RESTATEMENT"],
        )
        self.assertEqual(events["amendment_number"].tolist(), [pd.NA, 1, 2])
        quality = amendment_quality_summary(events).set_index("metric")
        self.assertEqual(quality.loc["canonical amendment rows", "rows"], 2)
        self.assertEqual(quality.loc["amendments missing a known type", "rows"], 0)

    def test_amendment_sequences_are_aggregate_and_ordered_by_availability(self) -> None:
        events = prepare_amendment_events(self._cover_pages())
        summary = amendment_sequence_summary(events)
        self.assertEqual(
            summary.iloc[0]["sequence"],
            "ORIGINAL > NEW_HOLDINGS > RESTATEMENT",
        )
        self.assertEqual(summary.iloc[0]["manager_periods"], 1)
        self.assertNotIn("manager_cik", summary.columns)

    def test_standard_deadline_rolls_weekends_and_federal_holidays(self) -> None:
        self.assertEqual(
            standard_13f_deadline("2026-03-31"),
            pd.Timestamp("2026-05-15"),
        )
        self.assertEqual(
            standard_13f_deadline("2026-06-30"),
            pd.Timestamp("2026-08-14"),
        )
        self.assertEqual(
            standard_13f_deadline("2026-12-31"),
            pd.Timestamp("2027-02-16"),
        )

    def test_quarter_coverage_requires_observation_before_quarter_end_and_deadline(self) -> None:
        filings = pd.DataFrame(
            {
                "accession_no": ["a1", "a2", "a3"],
                "manager_cik": ["m1", "m1", "m1"],
                "period_of_report": [
                    "2013-03-31",
                    "2013-06-30",
                    "2026-06-30",
                ],
                "availability_timestamp_utc": [
                    "2013-05-20T00:00:00Z",
                    "2013-08-14T00:00:00Z",
                    "2026-07-01T00:00:00Z",
                ],
                "is_amendment": [False, False, False],
            }
        )
        coverage = quarter_coverage_audit(
            filings,
            observation_start="2013-05-20T00:00:00Z",
            data_freeze="2026-07-28T00:00:00Z",
        ).set_index("report_date_utc")
        self.assertFalse(
            coverage.loc[
                pd.Timestamp("2013-03-31", tz="UTC"), "eligible_complete_window"
            ]
        )
        self.assertTrue(
            coverage.loc[
                pd.Timestamp("2013-06-30", tz="UTC"), "eligible_complete_window"
            ]
        )
        self.assertFalse(
            coverage.loc[
                pd.Timestamp("2026-06-30", tz="UTC"), "eligible_complete_window"
            ]
        )

    def test_manager_identity_audit_returns_no_raw_identifiers(self) -> None:
        pages = self._cover_pages()
        pages.loc[2, "manager_name"] = "Manager A Holdings"
        audit = manager_identity_audit(pages)
        self.assertEqual(
            audit.set_index("metric").loc["CIKs with multiple normalized names", "rows"],
            1,
        )
        self.assertEqual(list(audit.columns), ["metric", "rows"])

    def test_instrument_classification_uses_explicit_typed_channels(self) -> None:
        holdings = pd.DataFrame(
            {
                "title_of_class": [
                    "COM",
                    "COM",
                    "COM",
                    "CONVERTIBLE NOTE",
                    "PREFERRED",
                    "ETF",
                    "COMMON",
                    None,
                ],
                "put_call": [None, "PUT", "CALL", None, None, None, "", None],
                "shares_or_principal_type": [
                    "SH",
                    "SH",
                    "SH",
                    "PRN",
                    "SH",
                    "SH",
                    "SH",
                    None,
                ],
            }
        )
        classified = classify_instrument_rows(holdings)
        self.assertEqual(
            classified["instrument_class"].tolist(),
            [
                "cash_share_candidate",
                "put_option",
                "call_option",
                "convertible_or_debt",
                "preferred_equity",
                "fund_or_etf_explicit",
                "cash_share_candidate",
                "unknown",
            ],
        )

    def test_full_holdings_audit_scans_every_row_and_reuses_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            table_dir = root / "holdings"
            cache_dir = root / "cache"
            table_dir.mkdir()
            base = {
                "period_of_report": ["2024-03-31", "2024-03-31"],
                "filed_at": [
                    "2024-05-10T00:00:00Z",
                    "2024-05-10T00:00:00Z",
                ],
                "availability_timestamp_utc": [
                    "2024-05-10T00:00:00Z",
                    "2024-05-10T00:00:00Z",
                ],
                "title_of_class": ["COM", "COM"],
                "put_call": [None, "PUT"],
                "shares_or_principal_type": ["SH", "SH"],
                "value_usd": [100.0, 25.0],
                "reported_issuer_cik": ["1", None],
                "resolved_issuer_cik": ["1", "2"],
                "issuer_provider_entity_id": ["p1", "p2"],
                "issuer_identity_confidence": ["high", "medium"],
            }
            pq.write_table(pa.Table.from_pydict(base), table_dir / "part-1.parquet")
            pq.write_table(pa.Table.from_pydict(base), table_dir / "part-2.parquet")

            first = full_history_holdings_audit(
                table_dir,
                cache_directory=cache_dir,
                batch_size=1,
            )
            second = full_history_holdings_audit(
                table_dir,
                cache_directory=cache_dir,
                batch_size=1,
            )
            self.assertEqual(first.metadata["rows_scanned"], 4)
            self.assertFalse(first.metadata["cache_hit"])
            self.assertTrue(second.metadata["cache_hit"])
            self.assertEqual(int(first.instrument["rows"].sum()), 4)
            self.assertEqual(int(first.issuer_identity["rows"].sum()), 4)
            self.assertEqual(int(first.timing["rows"].sum()), 4)

    def test_committed_audit_notebook_is_output_free_and_public_safe(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        notebook_path = project_root / "notebooks" / "03_audit_13f_data_spine.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        self.assertTrue(all(cell.get("execution_count") is None for cell in code_cells))
        self.assertTrue(all(not cell.get("outputs") for cell in code_cells))
        notebook_text = notebook_path.read_text(encoding="utf-8")
        self.assertNotIn("gs://", notebook_text)
        self.assertNotIn("@", notebook_text)


if __name__ == "__main__":
    unittest.main()
