"""Guard and lineage checks for the bounded additive action input adapter."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import dmv2.action_loading as action
import dmv2.backtest_loading as base
from dmv2.catalog import canonical_sha256
from dmv2.universe import DataContractError
from test_backtest_loading import fixture, write_json, write_table


def action_fixture(root):
    pilot = fixture(root, blocked=True)
    identity_path = pilot / "build_identity.json"
    identity = json.loads(identity_path.read_text())
    for year in base.YEARS:
        directory = root / identity["data_plan"]["years"][str(year)]["feature_directory"]
        path = directory / "daily_prices_raw.parquet"
        rows = pq.read_table(path).to_pylist()
        for row in rows:
            for field in ("open", "high", "low", "close"):
                row[field] = row["adj_" + field] * 2
        extra = [{**row, "intrinio_security_id": native}
            for native in action.KNOWN_SUCCESSORS.values() for row in rows]
        artifact = write_table(directory, "daily_prices_raw", pa.Table.from_pylist(rows + extra))
        audit_path = directory / "audit.json"
        audit = json.loads(audit_path.read_text())
        audit["artifacts"]["daily_prices_raw"] = artifact
        write_json(audit_path, audit)
        identity["data_plan"]["years"][str(year)]["source_pins"] = [base._pin(root, audit_path)]
    write_json(identity_path, identity)
    build_hash = canonical_sha256(identity)
    audit_path = pilot / "audit.json"
    audit = json.loads(audit_path.read_text())
    audit["build_identity_sha256"] = build_hash
    audit["artifacts"]["build_identity.json"] = base._pin(pilot, identity_path)
    for arm in base.ARMS:
        path = pilot / "arms" / arm / "audit.json"
        arm_audit = json.loads(path.read_text())
        arm_audit["build_identity_sha256"] = build_hash
        write_json(path, arm_audit)
        audit["arms"][arm] = base._pin(pilot, path)
    write_json(audit_path, audit)
    write_json(pilot.parent / "LATEST.json", {"path": "synthetic",
        "audit_sha256": base._hash(audit_path), "build_identity_sha256": build_hash})
    evidence_dir = root / "data/derived/daily_action_inputs"
    records = []
    for ticker, (native, legal, session, cash, ratio) in action.EVENT_TERMS.items():
        records.append({"old_ticker": ticker, "old_provider_id": native,
            "effective_date": legal, "application_session": session,
            "cash_per_old_raw_share": cash, "successor_raw_shares": ratio,
            "source_url": "https://www.sec.gov/Archives/action-guard-fixture"})
    write_json(evidence_dir / "action_evidence.json", {"events": records})
    return evidence_dir


class ActionLoadingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.evidence = action_fixture(self.root)

    def supplement(self, *, future=False, wrong_id=False, missing=False):
        instrument = "sec_Xa9e96" if wrong_id else "vendor:reviewed:ensco-rowan-2019"
        binding = {"event_id": "RDC-2019-04-11", "instrument_id": instrument,
            "identifier_type": "CUSIP", "identifier": "G3166L100",
            "evidence_reference": "https://www.sec.gov/Archives/ensco-completion-fixture"}
        original = base.plan_daily_backtest(self.root)
        rows = pq.ParquetFile(original.daily_paths[2019]).read(columns=list(action.ACTION_PRICE_COLUMNS)).to_pylist()
        rows = [{**row, "intrinio_security_id": instrument} for row in rows
            if row["intrinio_security_id"] == "native-a" and "2019-04-11" <= row["trading_date"] <= "2019-05-01"]
        if future:
            rows[0]["trading_date"] = "2020-01-02"
        if missing:
            rows.pop(0)
        path = self.evidence / "ensco.parquet"
        pq.write_table(pa.Table.from_pylist(rows), path)
        item = {**base._pin(self.evidence, path), "instrument_id": instrument,
            "evidence_reference": "https://www.sec.gov/Archives/ensco-completion-fixture"}
        write_json(self.evidence / "supplemental_inputs.json", {
            "schema": "dmv2-action-supplement/v1", "successor_bindings": [binding], "quote_files": [item]})
        return path

    def test_plan_is_metadata_only_and_counts_extra_projection(self):
        read = pq.ParquetFile.read
        def metadata_only(parquet, columns=None, *args, **kwargs):
            self.assertIsNotNone(columns)
            self.assertFalse(set(columns) & {"open", "adj_open", "target_class", *base.PROBABILITIES})
            return read(parquet, columns=columns, *args, **kwargs)
        with patch.object(pq.ParquetFile, "read", metadata_only):
            plan = action.plan_action_inputs(self.root)
        self.assertEqual(plan.read_plan["daily_candidate_rows"], 503 * 4)
        self.assertEqual(plan.read_plan["original_requested_stable_ids"], 1)
        self.assertEqual(len(plan.requested_ids), 4)
        self.assertGreater(plan.read_plan["additional_projected_compressed_bytes"], 0)
        self.assertEqual(plan.read_plan["projected_compressed_bytes"],
            plan.base_plan.read_plan["projected_compressed_bytes"] + plan.read_plan["additional_projected_compressed_bytes"])
        self.assertFalse(plan.ready)
        self.assertEqual(plan.readiness_failures[0]["identifier"], "CUSIP:G3166L100")

    def test_unresolved_rdc_refuses_default_load_before_prices(self):
        plan = action.plan_action_inputs(self.root)
        with patch.object(base, "load_daily_backtest", side_effect=AssertionError("financial decode")), \
                self.assertRaisesRegex(DataContractError, "not ready"):
            action.load_action_inputs(self.root, plan=plan)
        rdc = next(event for event in plan.events if event["predecessor_id"] == "sec_zmYNnz")
        self.assertIsNone(rdc["successor_id"])
        self.assertEqual(rdc["successor_identity_status"], "unresolved")
        self.assertEqual(rdc["stock_ratio"], 0.6875)

    def test_audit_materialization_preserves_flags_raw_prices_and_successors(self):
        plan = action.plan_action_inputs(self.root)
        pins = {pin["path"]: base._hash(self.root / pin["path"]) for pin in plan.base_plan.source_pins}
        data = action.load_action_inputs(self.root, plan=plan, require_ready=False)
        self.assertFalse(data.ready)
        self.assertTrue(data.provenance["audit_only_unresolved_inputs"])
        self.assertFalse(data.provenance["identity_flags_promoted"])
        self.assertEqual(len(data.daily_prices), 503 * 4)
        self.assertTrue((data.daily_prices.open == data.daily_prices.adj_open * 2).all())
        for frame in data.predictions.values():
            self.assertEqual(len(frame), 24)
            self.assertEqual(frame.prediction_eligible.sum(), 23)
            self.assertTrue(frame.identity_review_required.all())
            self.assertTrue((frame.identity_status == "needs_review").all())
        self.assertEqual(pins, {pin["path"]: base._hash(self.root / pin["path"]) for pin in plan.base_plan.source_pins})

    def test_legal_date_and_application_session_remain_distinct(self):
        plan = action.plan_action_inputs(self.root)
        by_id = {event["predecessor_id"]: event for event in plan.events}
        self.assertEqual(by_id["sec_gedWvo"]["effective_date"], "2018-03-08")
        self.assertEqual(by_id["sec_zrJv8z"]["legal_effective_date"], "2018-12-20")
        self.assertEqual(by_id["sec_zrJv8z"]["effective_date"], "2018-12-21")
        self.assertTrue(all(not event["settlement_verified"] for event in plan.events))

    def test_2019_successor_file_with_evidence_allows_readiness(self):
        self.supplement()
        plan = action.plan_action_inputs(self.root)
        self.assertTrue(plan.ready)
        data = action.load_action_inputs(self.root, plan=plan)
        self.assertTrue(data.ready)
        rdc = data.events[data.events.predecessor_id.eq("sec_zmYNnz")].iloc[0]
        self.assertEqual(rdc.successor_id, "vendor:reviewed:ensco-rowan-2019")
        self.assertEqual(rdc.successor_identity_status, "provisional_candidate")
        self.assertFalse(data.provenance["financial_2020_plus_read"])

    def test_missing_successor_dates_are_reported_without_hindsight_filter(self):
        self.supplement(missing=True)
        plan = action.plan_action_inputs(self.root)
        self.assertFalse(plan.ready)
        self.assertEqual(plan.readiness_failures[0]["missing_dates"], ["2019-04-11"])
        self.assertEqual(len(plan.base_plan.prediction_metadata["rf_paper15"]), 24)

    def test_post2021_val_is_rejected_before_quote_values(self):
        self.supplement(wrong_id=True)
        with self.assertRaisesRegex(DataContractError, "actual 2019"):
            action.plan_action_inputs(self.root)

    def test_future_supplement_fails_before_numeric_read(self):
        self.supplement(future=True)
        read = pq.ParquetFile.read
        def metadata_only(parquet, columns=None, *args, **kwargs):
            self.assertNotIn("open", columns)
            self.assertNotIn("adj_open", columns)
            return read(parquet, columns=columns, *args, **kwargs)
        with patch.object(pq.ParquetFile, "read", metadata_only), \
                self.assertRaisesRegex(DataContractError, "outside 2018"):
            action.plan_action_inputs(self.root)

    def test_postplan_evidence_mutation_is_rejected_before_values(self):
        plan = action.plan_action_inputs(self.root)
        path = self.evidence / "action_evidence.json"
        path.write_bytes(path.read_bytes() + b" ")
        with patch.object(base, "load_daily_backtest", side_effect=AssertionError("financial decode")), \
                self.assertRaisesRegex(DataContractError, "hash/size"):
            action.load_action_inputs(self.root, plan=plan, require_ready=False)

    def test_changed_event_terms_rejected(self):
        path = self.evidence / "action_evidence.json"
        value = json.loads(path.read_text())
        value["events"][0]["cash_per_old_raw_share"] = 104.0
        write_json(path, value)
        with self.assertRaisesRegex(DataContractError, "terms changed"):
            action.plan_action_inputs(self.root)

    def test_extra_projection_budget_is_enforced(self):
        measured = action.plan_action_inputs(self.root).read_plan["projected_compressed_bytes"]
        with self.assertRaisesRegex(DataContractError, "action projections"):
            action.plan_action_inputs(self.root, max_projected_bytes=measured - 1)

    def test_mutated_plan_refuses_before_values(self):
        plan = action.plan_action_inputs(self.root)
        plan.read_plan["daily_selected_rows"] -= 1
        with self.assertRaisesRegex(DataContractError, "plan changed"):
            action.load_action_inputs(self.root, plan=plan, require_ready=False)

    def test_explicit_withhold_policy_keeps_unquoted_asset_and_failure(self):
        default = action.plan_action_inputs(self.root)
        plan = action.plan_action_inputs(self.root, unquoted_successor_policy="withhold_affected_paths")
        self.assertNotEqual(default.fingerprint, plan.fingerprint)
        self.assertFalse(plan.ready)
        self.assertTrue(plan.accounting_may_run)
        self.assertEqual(plan.readiness_failures[0]["identifier"], "CUSIP:G3166L100")
        data = action.load_action_inputs(self.root, plan=plan)
        rdc = data.events[data.events.predecessor_id.eq("sec_zmYNnz")].iloc[0]
        self.assertEqual(rdc.successor_id, action.RDC_UNQUOTED_ASSET)
        self.assertFalse(rdc.successor_is_provider_mapping)
        self.assertFalse(rdc.successor_quoted)
        self.assertNotIn(action.RDC_UNQUOTED_ASSET, plan.requested_ids)
        self.assertNotIn(action.RDC_UNQUOTED_ASSET, set(data.daily_prices.intrinio_security_id))
        self.assertTrue(data.provenance["fail_closed_accounting_only"])
        self.assertFalse(data.provenance["audit_only_unresolved_inputs"])
        self.assertFalse(data.provenance["fully_priced_inputs_ready"])

    def test_explicit_withhold_policy_blocks_only_path_holding_unquoted_entitlement(self):
        from dmv2.action_accounting import run_action_accounting
        data = action.load_action_inputs(self.root, unquoted_successor_policy="withhold_affected_paths")
        sessions = ["2019-04-10", "2019-04-11"]
        prices = data.daily_prices[data.daily_prices.trading_date.isin(sessions)].copy()
        predecessor = prices[prices.intrinio_security_id.eq("native-a")].copy()
        predecessor["intrinio_security_id"] = "sec_zmYNnz"
        predecessor = predecessor[predecessor.trading_date.eq("2019-04-10")]
        prices = pd.concat([prices, predecessor], ignore_index=True)
        rebalances = pd.DataFrame([{"signal_date": "2019-03-29", "execution_date": sessions[0]}])
        def selected(native):
            return pd.DataFrame([
                {"signal_date": "2019-03-29", "execution_date": sessions[0],
                 "source_security_id": "original-long", "intrinio_security_id": native, "side": "long"},
                {"signal_date": "2019-03-29", "execution_date": sessions[0],
                 "source_security_id": "original-short", "intrinio_security_id": "sec_Qz8jlg", "side": "short"}])
        event = data.events[data.events.predecessor_id.eq("sec_zmYNnz")]
        affected = run_action_accounting(selected("sec_zmYNnz"), prices, sessions,
            rebalances=rebalances, terminal_date=sessions[-1], events=event)
        unaffected = run_action_accounting(selected("native-a"), prices, sessions,
            rebalances=rebalances, terminal_date=sessions[-1], events=event)
        self.assertEqual(affected.status, "failed_closed")
        self.assertIn(action.RDC_UNQUOTED_ASSET, set(affected.failures.intrinio_security_id))
        self.assertEqual(unaffected.status, "complete_proxy")

    def test_policy_mismatch_and_unknown_policy_are_rejected(self):
        with self.assertRaisesRegex(DataContractError, "policy required"):
            action.plan_action_inputs(self.root, unquoted_successor_policy="invent_value")
        plan = action.plan_action_inputs(self.root, unquoted_successor_policy="withhold_affected_paths")
        with self.assertRaisesRegex(DataContractError, "scope differs"):
            action.load_action_inputs(self.root, plan=plan, unquoted_successor_policy="reject")

    def test_successor_binding_cannot_reuse_an_unrelated_cached_stock(self):
        write_json(self.evidence / "supplemental_inputs.json", {
            "schema": "dmv2-action-supplement/v1", "quote_files": [],
            "successor_bindings": [{"event_id": "RDC-2019-04-11", "instrument_id": "native-a",
                "identifier_type": "CUSIP", "identifier": "G3166L100",
                "evidence_reference": "https://www.sec.gov/Archives/ensco-completion-fixture"}]})
        with self.assertRaisesRegex(DataContractError, "cannot alias"):
            action.plan_action_inputs(self.root)


if __name__ == "__main__":
    unittest.main()
