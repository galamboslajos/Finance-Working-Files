"""Structural fixtures only; no historical financial observations."""
import calendar
import copy
import unittest

from dmv2.start_coverage import evaluate_start_coverage
from dmv2.universe import DataContractError


def fixture():
    months = [(2008 + offset // 12, offset % 12 + 1) for offset in range(26)]
    ends = {f"{year:04d}-{month:02d}": f"{year:04d}-{month:02d}-{calendar.monthrange(year, month)[1]}"
            for year, month in months}
    signal = ends["2009-12"]
    def member(source, native, status="resolved", review=False):
        return {"source_security_id": source, "as_of": signal, "signal_date": signal,
                "outcome_month": "2010-01", "intrinio_security_id": native,
                "unified_security_id": "u-" + source, "identity_status": status,
                "identity_review_required": review, "resolution_status": "resolved",
                "identity_eligible": True}
    ledger = [member("a", "id-a"), member("b", "id-b", "needs_review", True),
              member("blocked", None, "unmapped_intrinio", None)]
    def quote(native, day, value):
        return {"intrinio_security_id": native, "trading_date": day, "frequency": "daily",
                "intraperiod": False, "price_quality_status": "accepted", "adj_close": value}
    prices = {}
    outcomes = {}
    for native, growth in (("id-a", 1.01), ("id-b", 1.02)):
        for offset, (month, day) in enumerate(sorted(ends.items())):
            if month <= "2009-12":
                prices[(native, day)] = quote(native, day, 10 * growth**offset)
            elif month == "2010-01":
                outcomes[(native, day)] = quote(native, day, 10 * growth**offset)
    return ledger, ends, prices, outcomes


class StartCoverageTests(unittest.TestCase):
    def run_fixture(self, modify=None):
        args = fixture()
        if modify:
            modify(*args)
        return evaluate_start_coverage(*args[:3], outcome_quotes=args[3])

    def test_every_original_member_is_retained_and_policy_is_recomputed(self):
        rows, summary = self.run_fixture()
        self.assertEqual(len(rows), 3)
        self.assertEqual({row["source_security_id"] for row in rows}, {"a", "b", "blocked"})
        self.assertEqual(summary["original_stock_months"], 3)
        self.assertEqual(summary["complete_five_momenta"], 2)
        by_id = {row["source_security_id"]: row for row in rows}
        self.assertTrue(by_id["b"]["identity_review_required"])
        self.assertTrue(by_id["b"]["identity_eligible"])
        self.assertFalse(by_id["blocked"]["identity_eligible"])
        self.assertEqual(by_id["blocked"]["coverage_category"], "identity_blocked")

    def test_missing_or_changed_future_outcome_does_not_select_features(self):
        before, _ = self.run_fixture()
        def delete_outcomes(ledger, ends, prices, outcomes):
            outcomes.clear()
        after, _ = self.run_fixture(delete_outcomes)
        for old, new in zip(before, after, strict=True):
            for key in old:
                if key not in ("outcome_endpoint_status", "paired_outcome_endpoints_available"):
                    self.assertEqual(old[key], new[key], key)
        self.assertEqual(sum(row["paired_outcome_endpoints_available"] for row in before), 2)
        self.assertFalse(any(row["paired_outcome_endpoints_available"] for row in after))

    def test_monthly_paper_indexing_has_eleven_returns_for_mom12(self):
        rows, _ = self.run_fixture()
        by_id = {row["source_security_id"]: row for row in rows}
        self.assertAlmostEqual(by_id["a"]["MOM1_raw"], .01)
        self.assertAlmostEqual(by_id["a"]["MOM12_raw"], 1.01**11 - 1)

    def test_missing_intermediate_endpoint_is_not_telescoped(self):
        def remove(ledger, ends, prices, outcomes):
            prices.pop(("id-a", ends["2009-06"]))
        rows, summary = self.run_fixture(remove)
        a = next(row for row in rows if row["source_security_id"] == "a")
        self.assertEqual(a["MOM12_raw_status"], "missing_exact_endpoint")
        self.assertEqual(a["coverage_category"], "valid_signal_quote_incomplete_momentum_history")
        self.assertEqual(summary["complete_five_momenta"], 1)
        self.assertFalse(any(row["paper_normalization_diagnostic_ready"] for row in rows))

    def test_missing_signal_quote_is_separate_from_identity_block(self):
        def remove(ledger, ends, prices, outcomes):
            prices.pop(("id-b", ends["2009-12"]))
        rows, summary = self.run_fixture(remove)
        self.assertEqual(summary["exclusive_categories"]["no_exact_signal_quote"], 1)
        self.assertEqual(summary["exclusive_categories"]["identity_blocked"], 1)
        self.assertEqual(len(rows), 3)

    def test_same_date_native_collision_cannot_be_claimed_eligible(self):
        def collision(ledger, ends, prices, outcomes):
            ledger[1]["intrinio_security_id"] = "id-a"
        rows, summary = self.run_fixture(collision)
        self.assertFalse(any(row["identity_eligible"] for row in rows))
        self.assertEqual(summary["exclusive_categories"]["identity_blocked"], 3)

    def test_protected_or_post_signal_feature_values_fail_closed(self):
        for day in ("2010-01-31", "2024-01-02"):
            args = fixture()
            quote = dict(next(iter(args[2].values())), trading_date=day)
            args[2][(quote["intrinio_security_id"], day)] = quote
            with self.subTest(day=day), self.assertRaises(DataContractError):
                evaluate_start_coverage(*args[:3], outcome_quotes=args[3])

    def test_duplicate_original_members_and_incorrect_outcome_month_fail(self):
        for change in ("duplicate", "outcome"):
            args = fixture()
            if change == "duplicate":
                args[0].append(dict(args[0][0]))
            else:
                args[0][0]["outcome_month"] = "2010-02"
            with self.subTest(change=change), self.assertRaises(DataContractError):
                evaluate_start_coverage(*args[:3], outcome_quotes=args[3])

    def test_input_data_are_not_mutated(self):
        args = fixture()
        original = copy.deepcopy(args)
        evaluate_start_coverage(*args[:3], outcome_quotes=args[3])
        self.assertEqual(args, original)

    def test_all_five_momenta_use_one_common_population(self):
        def remove(ledger, ends, prices, outcomes):
            prices.pop(("id-a", ends["2009-06"]))
        _, summary = self.run_fixture(remove)
        self.assertEqual(summary["signals"][0]["common_momentum_population_count"], 1)
        self.assertEqual(summary["signals"][0]["normalization_diagnostic_status"], "insufficient_common_population")
        self.assertFalse(summary["normalization_diagnostic"]["outcome_availability_used_for_population"])


class SavedStartCoverageTests(unittest.TestCase):
    """Cache checks use structural artifacts, never historical prices."""
    def setUp(self):
        import json
        import tempfile
        from contextlib import ExitStack
        from pathlib import Path
        from unittest.mock import patch
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq
        from dmv2 import start_coverage as worker
        from dmv2.catalog import canonical_sha256

        self.worker = worker
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.out = Path(self.temporary.name) / "audit"
        self.out.mkdir()
        root = Path(self.temporary.name) / "source"
        root.mkdir()
        source = root / "fixture.source"
        source.write_bytes(b"structural source pin")
        class Object:
            relative_path = "fixture.source"
            size_bytes = source.stat().st_size
            sha256 = worker._hash_file(source)
            def check(self):
                if source.stat().st_size != self.size_bytes:
                    raise DataContractError("Structural source size drift")
        obj = Object()
        class Catalog:
            def __init__(self):
                self.root = root
            def provenance(self):
                return {"sources": [{"source_key": "fixture", "manifest_sha256": "fixture-hash"}]}
            def table_objects(self, source_key, table):
                return (obj,)
        self.catalog = Catalog()
        inputs = fixture()
        rows, summary = evaluate_start_coverage(*inputs[:3], outcome_quotes=inputs[3])
        dates = tuple(inputs[1].values())
        signals = (inputs[1]["2009-12"],)
        endpoints = tuple(day for month, day in inputs[1].items() if "2008-12" <= month <= "2010-01")
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(worker, "_code_pins", return_value={"fixture": "code-pin"}))
        self.stack.enter_context(patch.object(worker, "_runtime_pins", return_value={"fixture": "runtime-pin"}))
        self.stack.enter_context(patch.object(worker, "_calendar_parameters", return_value=(dates, inputs[1], signals, endpoints)))
        self.stack.enter_context(patch.object(worker, "_package_code_hash", return_value=({}, "calendar-pin")))
        pq.write_table(pa.Table.from_pylist(rows), self.out / "original_ledger.parquet")
        pd.DataFrame([{key: value for key, value in row.items() if not isinstance(value, dict)} for row in summary["yearly"]]).to_csv(self.out / "yearly.csv", index=False)
        provenance = {"source_key": "fixture", "table": "fixture",
                      "objects": [{"relative_path": obj.relative_path, "size_bytes": obj.size_bytes, "sha256": obj.sha256}]}
        summary.update({"parameters": worker._parameters(2010, 2010), "source_project_root": str(root),
            "source_pins": self.catalog.provenance(), "code_sha256": worker._code_pins(), "runtime": worker._runtime_pins(),
            "metadata_loading_provenance": {"fixture": provenance}, "price_loading_provenance": provenance,
            "calendar_provenance": {"code_sha256": "calendar-pin", "month_ends_sha256": canonical_sha256(inputs[1])},
            "artifacts": {name: {"size_bytes": (self.out/name).stat().st_size, "sha256": worker._hash_file(self.out/name)}
                          for name in ("original_ledger.parquet", "yearly.csv")}})
        requested = tuple(sorted({row["intrinio_security_id"] for row in rows if row["identity_eligible"]}))
        summary["request_sha256"] = worker._request_pin(summary, signals, endpoints, requested)
        self.evidence = summary
        self.write_evidence()

    def write_evidence(self):
        import json
        (self.out / "evidence.json").write_text(json.dumps(self.evidence, sort_keys=True))

    def read(self):
        return self.worker.read_saved_start_coverage(self.catalog, self.out, start_year=2010, end_year=2010)

    def test_pinned_cache_reads_without_raw_price_scan(self):
        self.assertEqual(self.read(), self.evidence)

    def test_changed_code_or_runtime_rejects_cache(self):
        from unittest.mock import patch
        for field in ("_code_pins", "_runtime_pins"):
            with self.subTest(field=field), patch.object(self.worker, field, return_value={"drift": True}), self.assertRaises(DataContractError):
                self.read()

    def test_changed_artifact_bytes_reject_cache(self):
        with (self.out / "yearly.csv").open("a") as stream:
            stream.write("changed\n")
        with self.assertRaisesRegex(DataContractError, "artifact integrity"):
            self.read()

    def test_changed_denominator_or_year_rejects_cache(self):
        for key in ("original_stock_months", "yearly"):
            original = copy.deepcopy(self.evidence)
            if key == "yearly":
                self.evidence[key][0]["outcome_year"] = 2011
            else:
                self.evidence[key] += 1
            self.write_evidence()
            with self.subTest(key=key), self.assertRaises(DataContractError):
                self.read()
            self.evidence = original

    def test_changed_source_root_rejects_cache(self):
        self.evidence["source_project_root"] += "-other"
        self.write_evidence()
        with self.assertRaisesRegex(DataContractError, "source root"):
            self.read()

    def test_existing_output_is_immutable(self):
        with self.assertRaisesRegex(DataContractError, "already exists"):
            self.worker.run_start_coverage(self.catalog, self.out, start_year=2010, end_year=2010)


if __name__ == "__main__":
    unittest.main()
