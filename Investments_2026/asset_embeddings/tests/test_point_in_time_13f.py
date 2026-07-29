import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.point_in_time_13f import (
    build_final_13f_panel,
    build_point_in_time_13f_panel,
    eligible_holdings,
    load_holdings_for_events,
    prepare_point_in_time_events,
)


class PointInTime13FTests(unittest.TestCase):
    def _source_frames(
        self,
        event_specs: list[dict[str, object]],
        holding_specs: list[dict[str, object]],
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        filings: list[dict[str, object]] = []
        covers: list[dict[str, object]] = []
        for spec in event_specs:
            role = spec.get("role", "ORIGINAL")
            is_amendment = role != "ORIGINAL"
            form_type = "13F-HR/A" if is_amendment else "13F-HR"
            available_at = str(spec["available_at"])
            accession = str(spec["accession_no"])
            report_type = str(spec.get("report_type", "13F HOLDINGS REPORT"))
            holding_count = int(spec.get("holding_count", 0))
            amendment_info = None
            if is_amendment:
                amendment_type = str(spec.get("amendment_type", role)).replace(
                    "_",
                    " ",
                )
                amendment_info = json.dumps(
                    {
                        "amendmentNo": int(spec.get("amendment_number", 1)),
                        "amendmentType": amendment_type,
                    }
                )
            common = {
                "accession_no": accession,
                "manager_cik": str(spec.get("manager_cik", "0001")),
                "form_type": form_type,
                "period_of_report": str(spec.get("period_of_report", "2024-03-31")),
                "filed_at": available_at,
                "availability_timestamp_utc": available_at,
                "is_amendment": is_amendment,
            }
            filings.append(
                {
                    **common,
                    "filed_month": pd.Timestamp(available_at).strftime("%Y-%m"),
                    "holdings_count": holding_count,
                }
            )
            covers.append(
                {
                    **common,
                    "amendment_info_json": amendment_info,
                    "report_type": report_type,
                }
            )

        holdings = pd.DataFrame.from_records(holding_specs)
        if holdings.empty:
            holdings = pd.DataFrame(
                columns=(
                    "holding_id",
                    "accession_no",
                    "manager_cik",
                    "period_of_report",
                    "asset",
                    "value_usd",
                )
            )
        return (
            pd.DataFrame.from_records(filings),
            pd.DataFrame.from_records(covers),
            holdings,
        )

    def _three_event_history(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return self._source_frames(
            [
                {
                    "accession_no": "a1",
                    "available_at": "2024-05-10T12:00:00Z",
                    "holding_count": 1,
                },
                {
                    "accession_no": "a2",
                    "available_at": "2024-05-20T12:00:00Z",
                    "role": "NEW_HOLDINGS",
                    "amendment_number": 1,
                    "holding_count": 1,
                },
                {
                    "accession_no": "a3",
                    "available_at": "2024-06-01T12:00:00Z",
                    "role": "RESTATEMENT",
                    "amendment_number": 2,
                    "holding_count": 1,
                },
            ],
            [
                {
                    "holding_id": "h1",
                    "accession_no": "a1",
                    "manager_cik": "0001",
                    "period_of_report": "2024-03-31",
                    "asset": "first",
                    "value_usd": 100.0,
                },
                {
                    "holding_id": "h2",
                    "accession_no": "a2",
                    "manager_cik": "0001",
                    "period_of_report": "2024-03-31",
                    "asset": "supplement",
                    "value_usd": 25.0,
                },
                {
                    "holding_id": "h3",
                    "accession_no": "a3",
                    "manager_cik": "0001",
                    "period_of_report": "2024-03-31",
                    "asset": "restated",
                    "value_usd": 110.0,
                },
            ],
        )

    def test_future_amendments_do_not_change_an_earlier_snapshot(self) -> None:
        filings, covers, holdings = self._three_event_history()
        full_history = build_point_in_time_13f_panel(
            filings,
            covers,
            holdings,
            decision_at="2024-05-15T23:59:59Z",
        )
        truncated_history = build_point_in_time_13f_panel(
            filings.iloc[:1],
            covers.iloc[:1],
            holdings[holdings["accession_no"].eq("a1")],
            decision_at="2024-05-15T23:59:59Z",
        )
        self.assertEqual(full_history.holdings_asof["holding_id"].tolist(), ["h1"])
        self.assertEqual(
            full_history.holdings_asof["holding_id"].tolist(),
            truncated_history.holdings_asof["holding_id"].tolist(),
        )
        self.assertEqual(
            full_history.manager_periods.iloc[0]["state_rows"],
            truncated_history.manager_periods.iloc[0]["state_rows"],
        )
        self.assertEqual(
            full_history.event_ledger["event_status"].tolist(),
            ["applied", "future_event", "future_event"],
        )

    def test_new_holdings_supplements_only_after_availability(self) -> None:
        filings, covers, holdings = self._three_event_history()
        panel = build_point_in_time_13f_panel(
            filings,
            covers,
            holdings,
            decision_at="2024-05-21T00:00:00Z",
        )
        self.assertEqual(
            panel.holdings_asof["holding_id"].tolist(),
            ["h1", "h2"],
        )
        applied = panel.event_ledger[panel.event_ledger["applied"]]
        self.assertEqual(
            applied["transition"].tolist(),
            ["initialize_from_original", "supplement_with_new_holdings"],
        )
        self.assertTrue(panel.manager_periods.iloc[0]["model_eligible"])

    def test_restatement_replaces_the_observable_state(self) -> None:
        filings, covers, holdings = self._three_event_history()
        panel = build_point_in_time_13f_panel(
            filings,
            covers,
            holdings,
            decision_at="2024-06-02T00:00:00Z",
        )
        self.assertEqual(panel.holdings_asof["holding_id"].tolist(), ["h3"])
        self.assertEqual(
            panel.event_ledger.iloc[-1]["transition"],
            "replace_with_restatement",
        )
        self.assertEqual(
            eligible_holdings(panel)["holding_id"].tolist(),
            ["h3"],
        )

    def test_final_state_is_explicitly_separate_from_historical_asof(self) -> None:
        filings, covers, holdings = self._three_event_history()
        historical = build_point_in_time_13f_panel(
            filings,
            covers,
            holdings,
            decision_at="2024-05-15T23:59:59Z",
        )
        final = build_final_13f_panel(filings, covers, holdings)
        self.assertEqual(historical.holdings_asof["holding_id"].tolist(), ["h1"])
        self.assertEqual(final.holdings_asof["holding_id"].tolist(), ["h3"])

    def test_amendment_without_an_observable_base_is_incomplete(self) -> None:
        filings, covers, holdings = self._source_frames(
            [
                {
                    "accession_no": "a2",
                    "available_at": "2024-05-20T12:00:00Z",
                    "role": "NEW_HOLDINGS",
                    "holding_count": 1,
                }
            ],
            [
                {
                    "holding_id": "h2",
                    "accession_no": "a2",
                    "manager_cik": "0001",
                    "period_of_report": "2024-03-31",
                    "asset": "supplement",
                    "value_usd": 25.0,
                }
            ],
        )
        panel = build_point_in_time_13f_panel(
            filings,
            covers,
            holdings,
            decision_at="2024-05-21T00:00:00Z",
        )
        self.assertEqual(panel.holdings_asof["holding_id"].tolist(), ["h2"])
        self.assertFalse(panel.manager_periods.iloc[0]["state_complete"])
        self.assertFalse(panel.manager_periods.iloc[0]["model_eligible"])
        self.assertTrue(eligible_holdings(panel).empty)

    def test_ambiguous_amendment_is_quarantined_and_blocks_model_use(self) -> None:
        filings, covers, holdings = self._source_frames(
            [
                {
                    "accession_no": "a1",
                    "available_at": "2024-05-10T12:00:00Z",
                    "holding_count": 1,
                },
                {
                    "accession_no": "a2",
                    "available_at": "2024-05-20T12:00:00Z",
                    "role": "NEW_HOLDINGS",
                    "amendment_type": "UNSPECIFIED",
                    "holding_count": 1,
                },
            ],
            [
                {
                    "holding_id": "h1",
                    "accession_no": "a1",
                    "manager_cik": "0001",
                    "period_of_report": "2024-03-31",
                    "asset": "first",
                    "value_usd": 100.0,
                },
                {
                    "holding_id": "h2",
                    "accession_no": "a2",
                    "manager_cik": "0001",
                    "period_of_report": "2024-03-31",
                    "asset": "unknown",
                    "value_usd": 25.0,
                },
            ],
        )
        panel = build_point_in_time_13f_panel(
            filings,
            covers,
            holdings,
            decision_at="2024-05-21T00:00:00Z",
        )
        self.assertEqual(panel.holdings_asof["holding_id"].tolist(), ["h1"])
        self.assertEqual(
            panel.event_ledger.iloc[1]["event_status"],
            "quarantined_unknown_amendment_type",
        )
        self.assertFalse(panel.manager_periods.iloc[0]["model_eligible"])

    def test_notices_are_retained_in_the_ledger_but_not_as_zero_portfolios(self) -> None:
        filings, covers, holdings = self._source_frames(
            [
                {
                    "accession_no": "n1",
                    "available_at": "2024-05-10T12:00:00Z",
                    "report_type": "13F NOTICE",
                    "holding_count": 0,
                }
            ],
            [],
        )
        panel = build_point_in_time_13f_panel(
            filings,
            covers,
            holdings,
            decision_at="2024-05-15T00:00:00Z",
        )
        self.assertTrue(panel.holdings_asof.empty)
        self.assertEqual(
            panel.event_ledger.iloc[0]["event_status"],
            "excluded_delegated_notice",
        )
        self.assertEqual(
            panel.manager_periods.iloc[0]["state_status"],
            "no_applied_state",
        )

    def test_holding_count_mismatch_is_quarantined(self) -> None:
        filings, covers, holdings = self._source_frames(
            [
                {
                    "accession_no": "a1",
                    "available_at": "2024-05-10T12:00:00Z",
                    "holding_count": 1,
                }
            ],
            [],
        )
        panel = build_point_in_time_13f_panel(
            filings,
            covers,
            holdings,
            decision_at="2024-05-15T00:00:00Z",
        )
        self.assertEqual(
            panel.event_ledger.iloc[0]["event_status"],
            "quarantined_holding_count_mismatch",
        )
        self.assertFalse(panel.manager_periods.iloc[0]["model_eligible"])

    def test_holding_metadata_conflict_is_quarantined(self) -> None:
        filings, covers, holdings = self._source_frames(
            [
                {
                    "accession_no": "a1",
                    "available_at": "2024-05-10T12:00:00Z",
                    "holding_count": 1,
                }
            ],
            [
                {
                    "holding_id": "h1",
                    "accession_no": "a1",
                    "manager_cik": "different-manager",
                    "period_of_report": "2024-03-31",
                    "asset": "first",
                    "value_usd": 100.0,
                }
            ],
        )
        panel = build_point_in_time_13f_panel(
            filings,
            covers,
            holdings,
            decision_at="2024-05-15T00:00:00Z",
        )
        self.assertEqual(
            panel.event_ledger.iloc[0]["event_status"],
            "quarantined_holding_metadata_conflict",
        )

    def test_duplicate_original_is_quarantined(self) -> None:
        filings, covers, holdings = self._source_frames(
            [
                {
                    "accession_no": "a1",
                    "available_at": "2024-05-10T12:00:00Z",
                    "holding_count": 1,
                },
                {
                    "accession_no": "a2",
                    "available_at": "2024-05-11T12:00:00Z",
                    "holding_count": 1,
                },
            ],
            [
                {
                    "holding_id": "h1",
                    "accession_no": "a1",
                    "manager_cik": "0001",
                    "period_of_report": "2024-03-31",
                    "asset": "first",
                    "value_usd": 100.0,
                },
                {
                    "holding_id": "h2",
                    "accession_no": "a2",
                    "manager_cik": "0001",
                    "period_of_report": "2024-03-31",
                    "asset": "second",
                    "value_usd": 200.0,
                },
            ],
        )
        panel = build_point_in_time_13f_panel(
            filings,
            covers,
            holdings,
            decision_at="2024-05-15T00:00:00Z",
        )
        self.assertEqual(panel.holdings_asof["holding_id"].tolist(), ["h1"])
        self.assertEqual(
            panel.event_ledger.iloc[1]["event_status"],
            "quarantined_duplicate_original",
        )
        self.assertFalse(panel.manager_periods.iloc[0]["model_eligible"])

    def test_decision_timestamp_must_be_timezone_aware(self) -> None:
        filings, covers, _ = self._three_event_history()
        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            prepare_point_in_time_events(
                filings,
                covers,
                decision_at="2024-05-15",
            )

    def test_parquet_loader_reads_only_visible_candidate_event_months(self) -> None:
        filings, covers, _ = self._three_event_history()
        events = prepare_point_in_time_events(
            filings,
            covers,
            decision_at="2024-05-15T23:59:59Z",
        )
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            may_rows = pa.Table.from_pylist(
                [
                    {
                        "holding_id": "h1",
                        "accession_no": "a1",
                        "manager_cik": "0001",
                        "period_of_report": "2024-03-31",
                    },
                    {
                        "holding_id": "unrelated",
                        "accession_no": "other",
                        "manager_cik": "0002",
                        "period_of_report": "2024-03-31",
                    },
                ]
            )
            june_rows = pa.Table.from_pylist(
                [
                    {
                        "holding_id": "h3",
                        "accession_no": "a3",
                        "manager_cik": "0001",
                        "period_of_report": "2024-03-31",
                    }
                ]
            )
            pq.write_table(may_rows, directory / "part-202405.parquet")
            pq.write_table(june_rows, directory / "part-202406.parquet")

            loaded = load_holdings_for_events(directory, events, batch_size=1)
            self.assertEqual(loaded["holding_id"].tolist(), ["h1"])

    def test_committed_validation_notebook_is_output_free_and_public_safe(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        notebook_path = (
            project_root
            / "notebooks"
            / "04_validate_13f_point_in_time_panel.ipynb"
        )
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        self.assertTrue(all(cell.get("execution_count") is None for cell in code_cells))
        self.assertTrue(all(not cell.get("outputs") for cell in code_cells))
        notebook_text = notebook_path.read_text(encoding="utf-8")
        self.assertNotIn("gs://", notebook_text)
        self.assertNotIn("@", notebook_text)


if __name__ == "__main__":
    unittest.main()
