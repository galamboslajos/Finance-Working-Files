"""Synthetic clocks/guard fixtures only; no research outcomes enter tests."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import dmv2.backtest_loading as loading
from dmv2.catalog import canonical_sha256
from dmv2.identity_policy import qualify_identity_records, PROVISIONAL_IDENTITY_POLICY
from dmv2.universe import DataContractError


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))


def write_table(directory, name, table):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (name + ".parquet")
    pq.write_table(table, path)
    return {**loading._pin(directory, path), "rows": table.num_rows, "columns": table.column_names}


def fixture(root, *, future_daily=False, missing_quote=False, blocked=False):
    sessions, windows, calendar = loading._independent_clock()
    pilot = root / "outputs/model_pilot/synthetic"
    annual_pins, metadata_rows = {}, []
    for year in loading.YEARS:
        directory = root / f"outputs/primary80/years/{year}"
        originals, coverage = [], []
        for row in windows[windows.outcome_month.str.startswith(str(year))].to_dict("records"):
            original = {"source_security_id": "source-a", "signal_date": row["signal_date"],
                "as_of": row["signal_date"], "unified_security_id": "unified-a",
                "intrinio_security_id": "native-a", "identity_status": "needs_review",
                "identity_review_required": True, "resolution_status": "resolved"}
            admitted = qualify_identity_records([original], identity_policy=PROVISIONAL_IDENTITY_POLICY)[0]
            eligible = not (blocked and row["outcome_month"] == "2018-02")
            source = {**{name: admitted[name] for name in (*loading.KEYS, *loading.IDENTITY_FIELDS)},
                "requested": True, "paper_prediction_eligible": eligible}
            prediction = {**source, **{name: row[name] for name in ("outcome_month", "outcome_start", "outcome_end")},
                "prediction_eligible": eligible, "outcome_requested": True,
                "return_available": row["outcome_month"] != "2018-01",
                "observed_return_available": row["outcome_month"] != "2018-01",
                "target_available": row["outcome_month"] != "2018-01",
                "target_status": "missing_return" if row["outcome_month"] == "2018-01" else "ok",
                "return_observation_status": "synthetic", "source_return_status": "synthetic",
                "feature_complete": False, "model_ready": False, "knowledge_time_verified": False,
                "accounting_certified": False, "available_feature_count": 15,
                "registered_feature_count": 728, "feature_status": "synthetic",
                "prediction_status": "ok" if eligible else "not_prediction_eligible"}
            originals.append(original); coverage.append(source); metadata_rows.append(prediction)
        bars = []
        for day in sessions:
            if day[:4] != str(year) or (missing_quote and day == "2018-01-03"):
                continue
            bars.append({"trading_date": day, "intrinio_security_id": "native-a",
                "adj_open": 100.0, "adj_high": 102.0, "adj_low": 98.0, "adj_close": 101.0,
                "frequency": "daily", "intraperiod": False, "price_quality_status": "accepted",
                "raw_ohlc_bounds_valid": True, "adjusted_ohlc_bounds_valid": True,
                "factor": 1.0, "split_ratio": 1.0, "dividend": 0.0})
        if future_daily and year == 2019:
            bars[-1]["trading_date"] = "2020-01-02"
        artifacts = {name: write_table(directory, name, pa.Table.from_pylist(rows))
            for name, rows in (("original_cohort", originals), ("feature_coverage", coverage),
                               ("daily_prices_raw", bars))}
        year_windows = {row["signal_date"]: {name: row[name] for name in ("outcome_month", "outcome_start", "outcome_end")}
            for row in windows[windows.outcome_month.str.startswith(str(year))].to_dict("records")}
        write_json(directory / "audit.json", {"year": year, "artifacts": artifacts,
            "feature_provenance": {"identity_policy": PROVISIONAL_IDENTITY_POLICY,
                "mode": "historical_reconstruction", "requested_ids": ["native-a"], "target_windows": year_windows}})
        annual_pins[str(year)] = {"feature_directory": directory.relative_to(root).as_posix(),
            "source_pins": [loading._pin(root, directory / "audit.json")]}
    identity = {"configuration": {"universe": "combined", "identity_policy": PROVISIONAL_IDENTITY_POLICY,
        "context_policy": "available_cohort_80pct", "train_outcome_start": "2010-01-01",
        "train_outcome_end": "2017-12-31", "validation_outcome_start": "2018-01-01",
        "validation_outcome_end": "2019-12-31"}, "data_plan": {"years": annual_pins}}
    write_json(pilot / "build_identity.json", identity)
    build_hash = canonical_sha256(identity)
    arms = {}
    for arm in loading.ARMS:
        directory = pilot / "arms" / arm
        rows = [{**row, **{name: 0.1 if row["prediction_eligible"] else None for name in loading.PROBABILITIES},
                 "target_class": 1} for row in metadata_rows]
        artifact = write_table(directory, "predictions", pa.Table.from_pylist(rows))
        write_json(directory / "audit.json", {"status": "COMPLETE", "arm": {"arm_id": arm},
            "build_identity_sha256": build_hash, "artifacts": {"predictions": artifact}})
        arms[arm] = loading._pin(pilot, directory / "audit.json")
    write_json(pilot / "audit.json", {"status": "COMPLETE", "build_identity_sha256": build_hash,
        "arms": arms, "artifacts": {"build_identity.json": loading._pin(pilot, pilot / "build_identity.json")}})
    write_json(pilot.parent / "LATEST.json", {"path": "synthetic", "build_identity_sha256": build_hash,
        "audit_sha256": loading._hash(pilot / "audit.json")})
    return pilot


class BacktestLoadingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.pilot = fixture(self.root)

    def rewrite_predictions(self, arm, change):
        directory = self.pilot / "arms" / arm
        path = directory / "predictions.parquet"
        rows = pq.read_table(path).to_pylist()
        change(rows)
        artifact = write_table(directory, "predictions", pa.Table.from_pylist(rows))
        audit_path = directory / "audit.json"
        audit = json.loads(audit_path.read_text())
        audit["artifacts"]["predictions"] = artifact
        write_json(audit_path, audit)
        root_path = self.pilot / "audit.json"
        root_audit = json.loads(root_path.read_text())
        root_audit["arms"][arm] = loading._pin(self.pilot, audit_path)
        write_json(root_path, root_audit)
        pointer_path = self.pilot.parent / "LATEST.json"
        pointer = json.loads(pointer_path.read_text())
        pointer["audit_sha256"] = loading._hash(root_path)
        write_json(pointer_path, pointer)

    def test_future_prediction_clock_fails_before_probability_read(self):
        self.rewrite_predictions("rf_paper15", lambda rows: rows[0].update(signal_date="2020-01-31"))
        read = pq.ParquetFile.read
        def metadata_only(parquet, columns=None, *args, **kwargs):
            self.assertFalse(set(columns or ()) & set(loading.PROBABILITIES))
            return read(parquet, columns=columns, *args, **kwargs)
        with patch.object(pq.ParquetFile, "read", metadata_only), \
                self.assertRaisesRegex(DataContractError, "approved 2018"):
            loading.plan_daily_backtest(self.root)

    def test_invalid_probability_sum_is_rejected_without_repair(self):
        self.rewrite_predictions("rf_paper15", lambda rows: rows[0].update(probability_1=0.2))
        plan = loading.plan_daily_backtest(self.root)
        with self.assertRaisesRegex(DataContractError, "ten-class contract"):
            loading.load_daily_backtest(self.root, plan=plan)

    def test_different_arm_cohort_status_is_rejected_before_probabilities(self):
        self.rewrite_predictions("xgb_paper15", lambda rows: rows[0].update(target_available=True))
        with self.assertRaisesRegex(DataContractError, "cohorts or metadata flags"):
            loading.plan_daily_backtest(self.root)

    def test_plan_only_reads_explicit_metadata_projections(self):
        read = pq.ParquetFile.read
        calls = []
        def guarded(parquet, columns=None, *args, **kwargs):
            self.assertIsNotNone(columns)
            self.assertFalse(set(columns) & set(loading.PROBABILITIES))
            self.assertNotIn("target_class", columns)
            self.assertFalse(set(columns) & {"adj_open", "adj_high", "adj_low", "adj_close"})
            calls.append(tuple(columns))
            return read(parquet, columns=columns, *args, **kwargs)
        with patch.object(pq.ParquetFile, "read", guarded):
            plan = loading.plan_daily_backtest(self.root)
        self.assertEqual(len(calls), 10)
        self.assertEqual(plan.read_plan["original_rows_per_arm"], 24)
        self.assertEqual(plan.read_plan["independent_sessions"], 503)
        self.assertEqual(plan.requested_ids, ("native-a",))
        self.assertFalse(plan.read_plan["financial_values_read"])
        self.assertFalse(plan.read_plan["financial_2020_plus_read"])
        self.assertEqual(plan.windows.iloc[-1].exit_date, "2019-12-31")
        self.assertEqual(plan.windows.iloc[-1].exit_price_field, "adj_close")
        self.assertEqual(plan.windows.iloc[0].signal_date, "2017-12-29")
        self.assertEqual(plan.windows.iloc[0].execution_date, "2018-01-02")
        self.assertIn("2017-12-29", plan.provenance["calendar"]["selection_sessions"])
        self.assertEqual(plan.provenance["calendar"]["selection_sessions"][-1], "2019-12-31")
        self.assertNotIn("2017-12-29", plan.sessions)
        self.assertEqual(plan.provenance["calendar"]["selection_sessions_sha256"],
            canonical_sha256(plan.provenance["calendar"]["selection_sessions"]))

    def test_loading_keeps_missing_future_label_rows_and_all_original_flags(self):
        plan = loading.plan_daily_backtest(self.root)
        data = loading.load_daily_backtest(self.root, plan=plan)
        self.assertEqual(len(data.daily_prices), 503)
        self.assertLess(data.daily_prices.trading_date.max(), "2020-01-01")
        for frame in data.predictions.values():
            self.assertEqual(len(frame), 24)
            self.assertEqual(frame.prediction_eligible.sum(), 24)
            self.assertFalse(frame.iloc[0].target_available)
            self.assertTrue(frame.identity_review_required.all())
            self.assertTrue((frame.identity_status == "needs_review").all())
            self.assertNotIn("target_class", frame)
        self.assertTrue(data.provenance["source_files_unchanged"])
        self.assertFalse(data.provenance["financial_2020_plus_read"])

    def test_missing_daily_quote_is_not_filled_or_removed_from_predictions(self):
        fixture(self.root, missing_quote=True)
        data = loading.load_daily_backtest(self.root)
        self.assertEqual(len(data.daily_prices), 502)
        self.assertNotIn("2018-01-03", set(data.daily_prices.trading_date))
        self.assertEqual(len(data.sessions), 503)
        self.assertTrue(all(len(frame) == 24 for frame in data.predictions.values()))

    def test_prediction_ineligible_rows_retain_nan_probabilities(self):
        fixture(self.root, blocked=True)
        data = loading.load_daily_backtest(self.root)
        for frame in data.predictions.values():
            self.assertEqual(len(frame), 24)
            self.assertEqual(frame.prediction_eligible.sum(), 23)
            self.assertTrue(frame.loc[~frame.prediction_eligible, list(loading.PROBABILITIES)].isna().all().all())

    def test_future_daily_cache_fails_before_financial_read(self):
        fixture(self.root, future_daily=True)
        read = pq.ParquetFile.read
        def metadata_only(parquet, columns=None, *args, **kwargs):
            self.assertFalse(set(columns or ()) & set(loading.PROBABILITIES))
            self.assertNotIn("adj_open", columns or ())
            return read(parquet, columns=columns, *args, **kwargs)
        with patch.object(pq.ParquetFile, "read", metadata_only), \
                self.assertRaisesRegex(DataContractError, "outside its approved outcome year"):
            loading.plan_daily_backtest(self.root)

    def test_hash_tampering_rejected_before_metadata_or_financial_read(self):
        path = self.pilot / "arms/rf_paper15/predictions.parquet"
        path.write_bytes(path.read_bytes() + b"changed")
        with patch.object(pq.ParquetFile, "read", side_effect=AssertionError("decoded corrupt file")), \
                self.assertRaisesRegex(DataContractError, "hash/size"):
            loading.plan_daily_backtest(self.root)

    def test_changed_input_after_plan_rejected_before_financial_read(self):
        plan = loading.plan_daily_backtest(self.root)
        path = plan.daily_paths[2018]
        path.write_bytes(path.read_bytes() + b"changed")
        with patch.object(pq.ParquetFile, "read", side_effect=AssertionError("decoded changed input")), \
                self.assertRaisesRegex(DataContractError, "hash/size"):
            loading.load_daily_backtest(self.root, plan=plan)

    def test_changed_plan_path_rejected_before_financial_read(self):
        plan = loading.plan_daily_backtest(self.root)
        plan.daily_paths[2018] = self.root / "outputs/2020.parquet"
        with patch.object(pq.ParquetFile, "read", side_effect=AssertionError("decoded changed path")), \
                self.assertRaisesRegex(DataContractError, "plan changed"):
            loading.load_daily_backtest(self.root, plan=plan)

    def test_changed_plan_metadata_rejected_before_financial_read(self):
        plan = loading.plan_daily_backtest(self.root)
        plan.windows.loc[0, "outcome_end"] = "2020-01-31"
        with patch.object(pq.ParquetFile, "read", side_effect=AssertionError("decoded changed metadata")), \
                self.assertRaisesRegex(DataContractError, "plan changed"):
            loading.load_daily_backtest(self.root, plan=plan)

    def test_explicit_read_budgets_reject_whole_request(self):
        with self.assertRaisesRegex(DataContractError, "row/byte ceiling"):
            loading.plan_daily_backtest(self.root, max_rows=1)
        with self.assertRaisesRegex(DataContractError, "row/byte ceiling"):
            loading.plan_daily_backtest(self.root, max_projected_bytes=1)
        for value in (0, -1, True):
            with self.assertRaisesRegex(DataContractError, "positive integers"):
                loading.plan_daily_backtest(self.root, max_rows=value)

    def test_pointer_path_cannot_escape_outputs(self):
        write_json(self.pilot.parent / "LATEST.json", {"path": "../../secrets"})
        with self.assertRaisesRegex(DataContractError, "leaves its declared root"):
            loading.plan_daily_backtest(self.root)

    def test_wrong_root_rejected(self):
        plan = loading.plan_daily_backtest(self.root)
        with self.assertRaisesRegex(DataContractError, "another root"):
            loading.load_daily_backtest(self.root / "elsewhere", plan=plan)

    def test_independent_sessions_do_not_consult_quote_inventory(self):
        sessions, windows, provenance = loading._independent_clock()
        self.assertNotIn("2018-12-05", sessions)  # Bush funeral special closure.
        self.assertNotIn("2019-04-19", sessions)  # Good Friday.
        self.assertIn("2019-12-31", sessions)
        self.assertEqual(len(windows), 24)
        self.assertIn("not_quote_inventory", provenance["calendar_source"])

    def test_calendar_runtime_pin_enforced(self):
        with patch.object(loading, "version", return_value="different"), \
                self.assertRaisesRegex(DataContractError, "reviewed"):
            loading._independent_clock()


if __name__ == "__main__":
    unittest.main()
