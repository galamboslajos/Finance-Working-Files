"""Synthetic full calendar, immutable resume, boundaries and metadata guards."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import pyarrow as pa

import dmv2.full_path_audit as m
from dmv2.full_path_accounting import prepare_price_index, prepare_action_registry
from dmv2.path_targets import build_path_targets

SCRIPT = Path(m.__file__).resolve().parents[2] / "scripts/run_full_path_audit.py"
spec = importlib.util.spec_from_file_location("full_path_audit_cli_test", SCRIPT)
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


def fixtures():
    sessions, windows, calendar = m.full_clock()
    annuals = []
    for year in m.YEARS:
        chosen = [w for w in windows if w["outcome_month"].startswith(str(year))]
        annuals.append({"year": year, "windows": chosen, "declared_cache_requested_ids": [f"native{i}" for i in range(11)],
            "materialization_compressed_bytes": 100, "original_rows": 144, "prediction_eligible_rows": 132,
            "monthly": [{"outcome_month": w["outcome_month"], "original_rows": 12, "prediction_eligible_rows": 11,
                "original_key_sha256": m.canonical_sha256(sorted([[f"source{i}", w["signal_date"]] for i in range(12)])),
                "eligible_key_sha256": m.canonical_sha256(sorted([[f"source{i}", w["signal_date"]] for i in range(11)]))} for w in chosen]})
    plan = {"schema": m.SCHEMA, "protocol": m.PROTOCOL, "protocol_id": "full-path-target-audit-2010-2023-20261022-v1",
        "source_pins": [{"sha256": m.CATALOG_SHA256}], "worker_sha256": m._workers(),
        "semantics": {"model_fits": 0, "portfolio_backtests": 0}, "sessions": list(sessions), "annuals": annuals,
        "events": [], "historical_retirements": [], "calendar": calendar,
        "limits": {"active_wall_seconds": 2700, "peak_rss_bytes": 4 * 1024**3,
            "output_bytes_per_year": 512 * 1024**2, "output_bytes_total": 8 * 1024**3},
        "read_plan": {"original_months": 168, "original_years": 14, "planning_metadata_projected_compressed_bytes": 1, "maximum_projected_compressed_bytes": 10000}}
    plan["plan_sha256"] = m.canonical_sha256(plan)
    return plan


def annual_values(plan, year):
    annual = next(a for a in plan["annuals"] if a["year"] == year)
    rows = []
    for window in annual["windows"]:
        for index in range(12):
            rows.append({"source_security_id": f"source{index}", "intrinio_security_id": f"native{index}" if index < 11 else None,
                "signal_date": window["signal_date"], "outcome_month": window["outcome_month"],
                "prediction_eligible": index < 11, "paper_prediction_eligible": index < 11,
                "target_available": index < 11, "identity_review_required": True})
    cohort = pd.DataFrame(rows)
    days = sorted({r["date"] for w in annual["windows"] for r in w["required_marks"] if r["date"] < "2024-01-01"})
    prices = pd.DataFrame([{"trading_date": day, "intrinio_security_id": f"native{index}",
        "open": 100., "close": 100., "adj_open": 100., "adj_high": 100., "adj_low": 100., "adj_close": 100.,
        "frequency": "daily", "intraperiod": False, "price_quality_status": "accepted",
        "raw_ohlc_bounds_valid": True, "adjusted_ohlc_bounds_valid": True} for day in days for index in range(10)])
    legacy = cohort[["source_security_id", "signal_date"]].copy()
    legacy["label_return"] = 0.
    legacy["target_class"] = 5
    return cohort, prices, legacy, prices.iloc[0:0].copy()


class FullPathAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = fixtures()

    def test_full_independent_clock_has168_months_and_calendar_only_protected_exit(self):
        windows = [w for a in self.plan["annuals"] for w in a["windows"]]
        self.assertEqual(len(windows), 168)
        self.assertEqual(windows[0]["signal_date"], "2009-12-31")
        self.assertEqual(windows[-1]["exit_date"], "2024-01-02")

    def test_wrong_corrected_catalog_refuses_before_source_table_reads(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "backtest_source_plan.json").write_text(json.dumps({"catalog_pin": {"sha256": "wrong"}}))
            with patch.object(m, "metadata", side_effect=AssertionError("must not decode")):
                with self.assertRaisesRegex(Exception, "exact reviewed corrected catalog"):
                    m.plan_full_path_audit(root, archive_directory=root)

    def test_guard_target_clock_rejects_future_before_financial_conversion(self):
        windows = self.plan["annuals"][-1]["windows"]
        rows = [{k: w[k] for k in ("signal_date", "outcome_month", "outcome_start", "outcome_end")} for w in windows]
        m.guard_target_clock(pa.Table.from_pylist(rows), 2023)
        rows[-1]["outcome_end"] = "2024-01-01"
        with self.assertRaises(Exception):
            m.guard_target_clock(pa.Table.from_pylist(rows), 2023)

    def test_json_byte_cap(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "large.json"
            path.write_text("[] " * 10)
            with self.assertRaises(Exception):
                m.read_json(path, cap=4)

    def test_core_cohort_preserves_null_metadata_and_renames_legacy(self):
        cohort, _, _, _ = annual_values(self.plan, 2010)
        table = m.core_cohort(cohort)
        self.assertNotIn("outcome_month", table.column_names)
        self.assertIn("legacy_target_available", table.column_names)
        self.assertIsNone(table.to_pylist()[-1]["intrinio_security_id"])

    def test_all_original_rows_and_requested_boundary_gap_survive(self):
        cohort, prices, _, _ = annual_values(self.plan, 2010)
        window = self.plan["annuals"][0]["windows"][-1]
        cohort = cohort.loc[cohort.signal_date == window["signal_date"]]
        prices = prices.loc[~((prices.intrinio_security_id == "native0") & (prices.trading_date == window["exit_date"]))]
        plan = deepcopy(self.plan)
        plan["annuals"][1]["declared_cache_requested_ids"].remove("native0")
        index = prepare_price_index(prices, plan["sessions"])
        episodes, marks, details = m.build_month_paths(plan, cohort, index, window)
        self.assertEqual(len(episodes), 48)
        missing = episodes.to_pandas().query("source_security_id == 'source0'")
        self.assertTrue(missing.reason.str.contains("not_requested_boundary").all())
        self.assertTrue(missing.requested.isna().all())
        self.assertTrue(missing.audit_requested.all())
        self.assertEqual(int(episodes.to_pandas().status.eq("not_requested").sum()), 4)
        panel = build_path_targets(m.core_cohort(cohort), episodes, marks, windows=[window], sessions=plan["sessions"], observed_through="2023-12-31")
        self.assertEqual(len(panel.table), 96)
        self.assertEqual(set(panel.diagnostics["rank_status"].to_pylist()), {"non_exhaustive_original_path_requests"})

    def test_unsupported_successor_identity_is_unknown_not_an_incomplete_engineering_request(self):
        plan = deepcopy(self.plan)
        plan["events"] = [{"event_id": "unsupported", "predecessor_id": "native0", "successor_id": "unquoted_successor:CUSIP:UNKNOWN",
            "effective_date": "2010-01-06", "cash_per_raw_share": 0., "stock_ratio": 1., "verified": True, "evidence_reference": "synthetic-evidence"}]
        cohort, prices, _, _ = annual_values(plan, 2010)
        window = plan["annuals"][0]["windows"][0]
        cohort = cohort.loc[cohort.signal_date == window["signal_date"]]
        episodes, marks, _ = m.build_month_paths(plan, cohort, prepare_price_index(prices, plan["sessions"]), window)
        rows = episodes.to_pandas().query("source_security_id == 'source0'")
        self.assertTrue(rows.requested.all())
        self.assertTrue(rows.reason.str.contains("unsupported_successor_identity").all())
        panel = build_path_targets(m.core_cohort(cohort), episodes, marks, windows=[window], sessions=plan["sessions"], observed_through="2023-12-31")
        self.assertNotIn("non_exhaustive_original_path_requests", panel.diagnostics["rank_status"].to_pylist())

    def test_saved_plan_rejects_another_consistently_self_hashed_catalog(self):
        plan = deepcopy(self.plan)
        m.validate_saved_plan_contract(plan)
        plan["source_pins"] = [{"sha256": "0" * 64}]
        plan["plan_sha256"] = m.canonical_sha256({k: v for k, v in plan.items() if k != "plan_sha256"})
        with self.assertRaisesRegex(Exception, "exact corrected catalog"):
            m.validate_saved_plan_contract(plan)

    def test_final_month_never_calls_financial_replay_even_if_stopped(self):
        cohort, prices, _, _ = annual_values(self.plan, 2023)
        window = self.plan["annuals"][-1]["windows"][-1]
        cohort = cohort.loc[cohort.signal_date == window["signal_date"]]
        with patch("dmv2.full_path_accounting.replay_unit_episode", side_effect=AssertionError("protected")):
            episodes, marks, _ = m.build_month_paths(self.plan, cohort, None, window)
        self.assertEqual(len(marks), 0)
        self.assertEqual(int(episodes.to_pandas().reason.eq("protected_scheduled_endpoint").sum()), 44)

    def test_signed_short_losses_are_retained_in_full_audit_mode(self):
        cohort, prices, _, _ = annual_values(self.plan, 2010)
        window = self.plan["annuals"][0]["windows"][0]
        cohort = cohort.loc[cohort.signal_date == window["signal_date"]]
        at = prices.trading_date.eq(window["exit_date"]) & prices.intrinio_security_id.eq("native0")
        for name in ("open", "close", "adj_open", "adj_high", "adj_low", "adj_close"):
            prices.loc[at, name] = 350.
        episodes, marks, details = m.build_month_paths(self.plan, cohort, prepare_price_index(prices, self.plan["sessions"]), window)
        final = marks.to_pandas().query("source_security_id == 'source0' and side == 'short' and path_variant == 'unstopped'").iloc[-1]
        self.assertAlmostEqual(final.wealth, -1.5)
        self.assertEqual(episodes.to_pandas().query("source_security_id == 'source0' and side == 'short' and path_variant == 'unstopped'").iloc[0].status, "complete")

    def test_supplemental_overlay_is_isolated_and_recomputes_full_rank_cohort(self):
        cohort, prices, _, _ = annual_values(self.plan, 2010)
        window = self.plan["annuals"][0]["windows"][0]
        cohort = cohort.loc[cohort.signal_date == window["signal_date"]]
        index = prepare_price_index(prices, self.plan["sessions"])
        registry = prepare_action_registry([], sessions=self.plan["sessions"])
        episodes, marks, _ = m.build_month_paths(self.plan, cohort, index, window, registry=registry)
        overlay = prices.loc[(prices.trading_date == window["exit_date"]) & prices.intrinio_security_id.eq("native0")].copy()
        for name in ("open", "close", "adj_open", "adj_high", "adj_low", "adj_close"):
            overlay[name] = 110.
        panel, _, summary = cli.supplemental_overlay(self.plan, cohort, index, window, overlay, episodes, marks, registry)
        self.assertEqual(summary["candidate_original_members"], 1)
        self.assertEqual(len(panel.table), 96)
        self.assertEqual(index.quotes[(window["exit_date"], "native0")]["adj_open"], 100.)

    def test_payoff_clock_excludes_uncompleted_signal_month_and_never_shortens120(self):
        reports = [{"outcome_month": "2020-01", "signal_date": "2019-12-31", "class_grid": []}]
        result = cli.payoff_readiness(reports)
        self.assertEqual(len(result), 8)
        self.assertEqual(result[0]["window_end"], "2019-11")
        self.assertEqual(result[0]["window_start"], "2009-12")
        self.assertEqual(result[0]["missing_class_cells"], 1200)

    def test_reader_missing_is_explicit(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(FileNotFoundError):
                m.read_full_path_audit(temporary)

    def test_complete_full_synthetic_run_resume_and_artifact_tamper(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            with patch.object(cli, "verify_plan", return_value=True), patch.object(m, "verify_plan", return_value=True), \
                patch.object(cli, "load_annual_values", side_effect=lambda plan, year: annual_values(plan, year)), \
                patch("builtins.print"):
                paused = cli.run(self.plan, output, through_year=2010)
                self.assertEqual(paused["status"], "PAUSED_AT_ANNUAL_CHECKPOINT")
                old = m.hash_file(output / "years/2010/audit.json")
                completed = cli.run(self.plan, output, resume=True)
                self.assertEqual(completed["status"], "COMPLETE_FULL_PATH_AUDIT")
                self.assertEqual(completed["original_rows"], 12 * 168)
                self.assertEqual(completed["class_grid_cells"], 168 * 8 * 10)
                self.assertEqual(old, m.hash_file(output / "years/2010/audit.json"))
                self.assertFalse(completed["models_fitted"])
                self.assertTrue(completed["training_availability"][0]["purged_not_yet_available_label_rows"] > 0)
                self.assertEqual(completed["supplemental_overlay_months"], [])
                with patch.object(m.pq, "ParquetFile", side_effect=AssertionError("passive reader must not decode")):
                    read_only = m.read_full_path_audit(output)
                self.assertEqual(read_only["original_months"], 168)
                year_audit = output / "years/2010/audit.json"
                original_receipt = year_audit.read_text()
                malicious = json.loads(original_receipt)
                malicious["artifacts"]["../../outside.json"] = {"sha256": "unused", "size_bytes": 0}
                year_audit.write_text(json.dumps(malicious))
                with self.assertRaisesRegex(Exception, "Annual audit receipt changed"):
                    m.read_full_path_audit(output)
                year_audit.write_text(original_receipt)
                with self.assertRaises(Exception):
                    cli.run(self.plan, output, resume=True)
            artifact = output / "years/2010/months/2010-01/attempt-0001/targets.parquet"
            artifact.write_bytes(artifact.read_bytes() + b"tampered")
            with self.assertRaisesRegex(Exception, "Monthly artifact changed"):
                m.read_full_path_audit(output)

    def test_hard_interruption_cannot_reset_active_budget(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            cli.write_json(output / "reviewed_plan.json", self.plan)
            (output / "run_attempts").mkdir()
            cli.write_json(output / "run_attempts/run-0001.STARTED.json", {"active_seconds_before_attempt": 0})
            with patch.object(cli, "verify_plan", return_value=True):
                with self.assertRaisesRegex(Exception, "unmeasured active time"):
                    cli.run(self.plan, output, resume=True)


if __name__ == "__main__":
    unittest.main()
