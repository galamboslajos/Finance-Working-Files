"""Identity metadata fixtures are corroboration diagnostics, never approvals."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import pyarrow as pa
import pyarrow.parquet as pq

from dmv2.universe import DataContractError, EXCLUDED_SCHEMA, UNIVERSE_SCHEMA, UniverseSnapshot


APPLE_ISIN = "US0378331005"
MICROSOFT_ISIN = "US5949181045"
SIGNAL = "2000-01-31"


def identity_fixture():
    members = [dict(security_id=source, trading_date=SIGNAL, index_id=index,
        membership_ticker=ticker, isin=isin, is_member=True,
        membership_effective_date="1999-01-01", membership_end_date=None,
        membership_availability_date="1999-01-01") for source, index, ticker, isin in
        (("source-a", "sp500", "HISTORICAL_A", APPLE_ISIN),
         ("source-b", "sp400", "HISTORICAL_B", MICROSOFT_ISIN))]
    rows = [dict(source_security_id=member["security_id"], as_of=SIGNAL,
        intrinio_security_id="provider-"+"ab"[i], unified_security_id="unified-"+"ab"[i],
        indices=[member["index_id"]], membership_tickers=[member["membership_ticker"]],
        membership_records_json=json.dumps([member]),
        identity_status="needs_review" if i == 0 else "resolved",
        identity_review_required=i == 0, resolution_status="resolved") for i, member in enumerate(members)]
    cohort = UniverseSnapshot(pa.Table.from_pylist(rows, schema=UNIVERSE_SCHEMA),
        pa.Table.from_pylist([], schema=EXCLUDED_SCHEMA), SIGNAL, "combined",
        {"mode": "historical_reconstruction", "knowledge_time_verified": False}, {})
    bridge = [dict(source_product_id="sp_composite1500", source_security_id=member["security_id"],
        unified_security_id="unified-"+"ab"[i], isin=member["isin"], bridge_method="isin")
        for i, member in enumerate(members)]
    links = [dict(unified_security_id="unified-"+"ab"[i], intrinio_security_id="provider-"+"ab"[i],
        query_identifier=member["isin"], query_identifier_type="isin",
        observed_isins_json=json.dumps([member["isin"]]),
        observed_tickers_json=json.dumps([member["membership_ticker"]]),
        identity_review_required=i == 0, resolution_status="resolved", http_status=200,
        source_response_hash="STRUCTURAL_RESPONSE_HASH") for i, member in enumerate(members)]
    history = [dict(security_id=member["security_id"], identifier_type="isin",
        identifier_value=member["isin"], valid_from=None, valid_to=None)
        for member in members]
    return dict(cohorts=(cohort,), bridge=bridge, links=links,
                identifier_history=history, signal_membership=members)


def arrow_args(fixture):
    return {key: pa.Table.from_pylist(value) if isinstance(value, list) else value
            for key, value in fixture.items()}


class IdentityAuditTests(unittest.TestCase):
    def audit(self, fixture=None):
        from dmv2.identity_audit import audit_identity_links
        return audit_identity_links(**arrow_args(fixture or identity_fixture()))

    def rows(self, result):
        return {row["source_security_id"]: row for row in result.table.to_pylist()}

    def test_source_alias_and_exact_query_support_do_not_promote_any_identity(self):
        result = self.audit()
        rows = self.rows(result)
        self.assertEqual(set(rows), {"source-a", "source-b"})
        for row in rows.values():
            self.assertTrue(row["source_alias_isin_agreement"])
            self.assertTrue(row["exact_provider_isin_query"])
            self.assertTrue(row["source_ticker_alias_agreement"])
            self.assertFalse(row["dated_identifier_support"])
            self.assertFalse(row["temporal_identity_verified"])
            self.assertFalse(row["knowledge_time_verified"])
            self.assertFalse(row["independent_provider_isin_verified"])
            self.assertFalse(row["raw_response_content_verified"])
            self.assertEqual(row["qualification"], "REVIEW_REQUIRED_NOT_PROMOTED")
        self.assertTrue(rows["source-a"]["provider_review_required"])
        self.assertFalse(rows["source-b"]["provider_review_required"])
        self.assertEqual(rows["source-a"]["original_identity_status"], "needs_review")

    def test_wrong_checksum_and_mislabeled_query_type_do_not_count_as_strong_query(self):
        valid_source_status = self.rows(self.audit())["source-a"]["source_isin_status"]
        for defect in ("checksum", "ticker_as_isin"):
            with self.subTest(defect=defect):
                fixture = identity_fixture()
                bad = "US0378331004" if defect == "checksum" else "HISTORICAL_A"
                fixture["links"][0]["query_identifier"] = bad
                fixture["links"][0]["observed_isins_json"] = json.dumps([bad])
                if defect == "checksum":
                    fixture["signal_membership"][0]["isin"] = bad
                    fixture["bridge"][0]["isin"] = bad
                row = self.rows(self.audit(fixture))["source-a"]
                self.assertFalse(row["exact_provider_isin_query"])
                if defect == "checksum":
                    self.assertNotEqual(row["source_isin_status"], valid_source_status)
                else:
                    self.assertFalse(row["source_alias_isin_agreement"])
                self.assertFalse(row["temporal_identity_verified"])

    def test_current_query_ticker_is_not_the_historical_source_ticker(self):
        fixture = identity_fixture()
        fixture["links"][0].update(query_identifier="NEW_A", query_identifier_type="ticker",
            observed_tickers_json=json.dumps(["NEW_A"]))
        row = self.rows(self.audit(fixture))["source-a"]
        self.assertFalse(row["source_ticker_alias_agreement"])
        self.assertFalse(row["exact_provider_isin_query"])
        self.assertEqual(row["qualification"], "REVIEW_REQUIRED_NOT_PROMOTED")

    def test_global_ever_aliases_cannot_become_same_date_collision_gate(self):
        fixture = identity_fixture()
        original = self.rows(self.audit(fixture))["source-a"]
        fixture["links"].append({**fixture["links"][0], "unified_security_id": "future-unified"})
        changed = self.rows(self.audit(fixture))["source-a"]
        self.assertGreater(changed["global_provider_alias_count"], original["global_provider_alias_count"])
        self.assertFalse(changed["same_date_provider_collision"])
        self.assertEqual(changed["link_structure"], original["link_structure"])
        self.assertEqual(changed["qualification"], original["qualification"])

    def test_actual_same_date_provider_collision_marks_both_original_names(self):
        fixture = identity_fixture()
        fixture["links"][1]["intrinio_security_id"] = "provider-a"
        # Caller metadata still preserves the original pre-audit status/route.
        result = self.audit(fixture)
        rows = self.rows(result)
        self.assertEqual(result.table.num_rows, 2)
        self.assertTrue(all(row["same_date_provider_collision"] for row in rows.values()))
        self.assertTrue(all(row["link_structure"] == "COLLISION" for row in rows.values()))
        self.assertTrue(all(not row["temporal_identity_verified"] for row in rows.values()))

    def test_dated_identifier_support_uses_half_open_interval_not_blank_open_end(self):
        cases = [
            ("1999-01-01", "2001-01-01", True),
            (SIGNAL, "2001-01-01", True),
            ("1999-01-01", SIGNAL, False),
            (None, "2001-01-01", False),
            ("1999-01-01", None, False),
            ("2001-01-01", "2002-01-01", False),
            ("2001-01-01", "1999-01-01", False),
            ("1999-01-01T00:00:00Z", "2001-01-01", False),
        ]
        for start, end, expected in cases:
            with self.subTest(start=start, end=end):
                fixture = identity_fixture()
                fixture["identifier_history"][0].update(valid_from=start, valid_to=end)
                row = self.rows(self.audit(fixture))["source-a"]
                self.assertEqual(row["dated_identifier_support"], expected)
                self.assertFalse(row["knowledge_time_verified"])
                self.assertFalse(row["temporal_identity_verified"])
                self.assertEqual(row["qualification"], "REVIEW_REQUIRED_NOT_PROMOTED")

    def test_source_effective_dates_and_availability_do_not_certify_provider_link_vintage(self):
        fixture = identity_fixture()
        fixture["identifier_history"][0].update(valid_from="1999-01-01", valid_to="2001-01-01",
                                                available_at_utc="2000-01-31T23:59:59Z")
        row = self.rows(self.audit(fixture))["source-a"]
        self.assertFalse(row["knowledge_time_verified"])
        self.assertFalse(row["temporal_identity_verified"])
        self.assertEqual(row["qualification"], "REVIEW_REQUIRED_NOT_PROMOTED")

    def test_acquisition_record_count_is_not_signal_date_quote_or_history_coverage(self):
        fixture = identity_fixture()
        fixture["acquisition_evidence"] = [dict(requested_identifier=APPLE_ISIN,
            intrinio_security_id="provider-a", status="complete", http_status=200,
            price_row_count=321, price_page_count=2)]
        result = self.audit(fixture)
        row = self.rows(result)["source-a"]
        self.assertTrue(row["acquisition_reported_positive_rows"])
        self.assertFalse(row["temporal_identity_verified"])
        self.assertFalse(row["knowledge_time_verified"])
        self.assertEqual(row["qualification"], "REVIEW_REQUIRED_NOT_PROMOTED")

    def test_original_unlinked_member_remains_in_result_and_review_queue(self):
        fixture = identity_fixture()
        member = dict(security_id="source-c", trading_date=SIGNAL, index_id="sp500",
            membership_ticker="LOST_C", isin=None, is_member=True,
            membership_effective_date="1999-01-01", membership_end_date=None,
            membership_availability_date="1999-01-01")
        fixture["signal_membership"].append(member)
        old = fixture["cohorts"][0]
        rows = old.table.to_pylist()+[dict(source_security_id="source-c", as_of=SIGNAL,
            intrinio_security_id=None, unified_security_id=None, indices=["sp500"],
            membership_tickers=["LOST_C"], membership_records_json=json.dumps([member]),
            identity_status="unmapped_bridge", identity_review_required=None,
            resolution_status=None)]
        fixture["cohorts"] = (UniverseSnapshot(pa.Table.from_pylist(rows, schema=UNIVERSE_SCHEMA),
            old.excluded, old.as_of, old.universe, old.provenance, old.summary),)
        result = self.audit(fixture)
        self.assertEqual(result.table.num_rows, 3)
        rows = self.rows(result)
        self.assertEqual(rows["source-c"]["link_structure"], "UNLINKED")
        self.assertEqual(rows["source-c"]["original_identity_status"], "unmapped_bridge")
        self.assertFalse(rows["source-c"]["exact_provider_isin_query"])
        self.assertIsNone(rows["source-c"]["route_agrees_with_snapshot"])
        self.assertIn("source-c", result.review_queue.column("source_security_id").to_pylist())

    def test_identical_bridge_uid_routes_are_distinct_from_conflicting_routes(self):
        fixture = identity_fixture()
        original = self.rows(self.audit(fixture))["source-a"]
        fixture["bridge"].append(fixture["bridge"][0].copy())
        repeated = self.rows(self.audit(fixture))["source-a"]
        self.assertEqual(repeated["link_structure"], original["link_structure"])
        fixture["bridge"][-1]["unified_security_id"] = "different-unified"
        ambiguous = self.rows(self.audit(fixture))["source-a"]
        self.assertEqual(ambiguous["link_structure"], "AMBIGUOUS")
        self.assertFalse(ambiguous["exact_provider_isin_query"])

    def test_duplicate_provider_link_records_do_not_become_unique_by_same_id(self):
        fixture = identity_fixture()
        fixture["links"].append(fixture["links"][0].copy())
        row = self.rows(self.audit(fixture))["source-a"]
        self.assertEqual(row["link_structure"], "AMBIGUOUS")
        self.assertFalse(row["exact_provider_isin_query"])
        self.assertEqual(row["qualification"], "REVIEW_REQUIRED_NOT_PROMOTED")

    def test_recomputed_route_change_is_a_structural_review_without_replacing_original(self):
        fixture = identity_fixture()
        fixture["links"][0]["intrinio_security_id"] = "different-provider"
        row = self.rows(self.audit(fixture))["source-a"]
        self.assertEqual(row["original_intrinio_security_id"], "provider-a")
        self.assertEqual(row["intrinio_security_id"], "different-provider")
        self.assertFalse(row["route_agrees_with_snapshot"])
        self.assertEqual(row["review_priority"], "01_STRUCTURAL_CONFLICT")
        self.assertEqual(row["qualification"], "REVIEW_REQUIRED_NOT_PROMOTED")

    def test_protected_duplicate_unordered_cohorts_and_malformed_aliases_fail_closed(self):
        for defect in ("protected", "duplicate_date", "malformed_alias", "nonlist_alias", "blank_alias", "duplicate_source"):
            with self.subTest(defect=defect):
                fixture = identity_fixture()
                if defect == "protected":
                    old = fixture["cohorts"][0]
                    fixture["cohorts"] = (UniverseSnapshot(old.table, old.excluded, "2024-01-31",
                        old.universe, old.provenance, old.summary),)
                elif defect == "duplicate_date":
                    fixture["cohorts"] *= 2
                elif defect == "malformed_alias":
                    fixture["links"][0]["observed_isins_json"] = "not JSON"
                elif defect == "nonlist_alias":
                    fixture["links"][0]["observed_isins_json"] = json.dumps({"isin": APPLE_ISIN})
                elif defect == "blank_alias":
                    fixture["links"][0]["observed_isins_json"] = json.dumps([""])
                else:
                    old = fixture["cohorts"][0]
                    fixture["cohorts"] = (UniverseSnapshot(pa.concat_tables([old.table, old.table.slice(0, 1)]),
                        old.excluded, old.as_of, old.universe, old.provenance, old.summary),)
                with self.assertRaises(DataContractError):
                    self.audit(fixture)

    def test_financial_or_target_columns_cannot_enter_metadata_audit(self):
        for input_name, column in (("bridge", "adj_close"), ("links", "target_return"),
                                   ("identifier_history", "outcome_end"), ("signal_membership", "future_return")):
            with self.subTest(input=input_name, column=column):
                fixture = identity_fixture()
                for row in fixture[input_name]:
                    row[column] = 123.
                with self.assertRaises(DataContractError):
                    self.audit(fixture)

    def test_ineligible_source_membership_cannot_be_treated_as_signal_known_evidence(self):
        cases = [("is_member", False), ("membership_effective_date", "2001-01-01"),
                 ("membership_effective_date", None), ("membership_availability_date", "2001-01-01"),
                 ("membership_availability_date", None), ("membership_end_date", "1999-12-31")]
        for field, value in cases:
            with self.subTest(field=field, value=value):
                fixture = identity_fixture()
                fixture["signal_membership"][0][field] = value
                with self.assertRaises(DataContractError):
                    self.audit(fixture)

    def test_builder_preflight_dates_universe_and_budget_before_source_access(self):
        from dmv2.identity_audit import IDENTITY_BUDGET, build_identity_audit
        from dataclasses import replace
        cases = [dict(dates="2000-01-31"), dict(dates=()),
                 dict(dates=("2000-01-31", "2000-01-31")),
                 dict(dates=("2005-01-31", "2000-01-31")), dict(dates=("2024-01-31",)),
                 dict(dates=("2000-02-30",)), dict(dates=("1999-12-31",)), dict(universe="sp600"),
                 dict(budget=replace(IDENTITY_BUDGET, max_scanned_rows=IDENTITY_BUDGET.max_scanned_rows+1)),
                 dict(budget={})]
        for changes in cases:
            with self.subTest(changes=changes):
                catalog = Mock()
                with patch("dmv2.identity_audit.BoundedTableReader") as reader:
                    with self.assertRaises(DataContractError):
                        build_identity_audit(catalog, **changes)
                    reader.assert_not_called()
                self.assertEqual(catalog.mock_calls, [])


class IdentityAuditPersistenceTests(unittest.TestCase):
    def setUp(self):
        from dmv2.identity_audit import audit_identity_links
        self.result = audit_identity_links(**arrow_args(identity_fixture()))
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def save(self):
        from dmv2.identity_audit import save_identity_audit
        return save_identity_audit(self.root, self.result)

    def test_same_pins_persist_identically_without_overwrite(self):
        first = self.save()
        files = [first, first.parent/"constituent_ledger.parquet", first.parent/"review_queue.parquet"]
        original = {path.name: path.read_bytes() for path in files}
        self.assertEqual(self.save(), first)
        self.assertEqual({path.name: path.read_bytes() for path in files}, original)
        report = json.loads(first.read_text())
        self.assertEqual(report["promoted_identities"], 0)
        self.assertFalse(report["model_ready"])

    def test_tampered_ledger_is_preserved_and_rejected(self):
        first = self.save()
        ledger = first.parent/"constituent_ledger.parquet"
        frame = pq.read_table(ledger)
        replacement = pa.array(["FORGED_APPROVAL"]*frame.num_rows)
        forged = frame.set_column(frame.schema.get_field_index("qualification"), "qualification", replacement)
        pq.write_table(forged, ledger)
        altered_bytes = ledger.read_bytes()
        with self.assertRaisesRegex(DataContractError, "ledger changed"):
            self.save()
        self.assertEqual(ledger.read_bytes(), altered_bytes)

    def test_tampered_summary_is_preserved_and_rejected(self):
        first = self.save()
        first.write_text(first.read_text()+"\n")
        altered_bytes = first.read_bytes()
        with self.assertRaisesRegex(DataContractError, "evidence drift"):
            self.save()
        self.assertEqual(first.read_bytes(), altered_bytes)

    def test_output_directory_symlink_cannot_leave_project_root(self):
        with tempfile.TemporaryDirectory() as outside:
            (self.root/"outputs").mkdir()
            (self.root/"outputs"/"identity_audits").symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(DataContractError, "leaves project root"):
                self.save()
            self.assertEqual(list(Path(outside).iterdir()), [])

    def test_summary_symlink_is_rejected_before_external_content_read(self):
        first = self.save()
        with tempfile.TemporaryDirectory() as outside:
            external = Path(outside)/"metadata.json"
            external.write_text("EXTERNAL_CONTENT_MUST_NOT_BE_READ")
            first.unlink()
            first.symlink_to(external)
            original_read = Path.read_text
            def guarded_read(path, *args, **kwargs):
                if path.resolve() == external.resolve():
                    raise AssertionError("External summary content read before containment guard")
                return original_read(path, *args, **kwargs)
            with patch.object(Path, "read_text", guarded_read):
                with self.assertRaisesRegex(DataContractError, "leaves project root"):
                    self.save()


if __name__ == "__main__":
    unittest.main()
