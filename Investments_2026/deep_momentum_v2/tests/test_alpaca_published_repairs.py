"""Published-projection contracts using bounded synthetic artifacts only."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from dmv2.alpaca_repairs import prepare_published_alpaca_repair_declarations
from dmv2.ohlc_repairs import OhlcRepairPolicy, apply_reviewed_ohlc_repairs
from dmv2.universe import DataContractError


class PublishedAlpacaRepairTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.table_uri = "gs://synthetic-only/published/daily_bars.parquet"
        self.path = self.root / "selected.parquet"
        self.receipt_path = self.root / "range_receipt.json"
        self.receipt_path.write_text(json.dumps({"synthetic_fixture_only": True}))
        self.rows = [{"security_id": "siblis_synthetic", "trading_date": "2018-08-02",
            "query_symbol": "SYNTHETIC", "adjustment": "raw", "feed": "sip", "timeframe": "1Day",
            "source_task_id": "task", "source_request_id": "request", "source_page_hash": "1" * 64,
            "open": 100., "high": 110., "low": 90., "close": 105.}]
        self.prices = pd.DataFrame([{"intrinio_security_id": "synthetic_native", "trading_date": "2018-08-02",
            "open": 80., "high": 110., "low": 90., "close": 105., "adj_open": 64., "adj_high": 88.,
            "adj_low": 72., "adj_close": 84., "factor": 1., "price_quality_status": "provider_ohlc_anomaly",
            "raw_ohlc_bounds_valid": False, "adjusted_ohlc_bounds_valid": False}])
        self.provenance = {
            "publication_manifest_uri": "gs://synthetic-only/MANIFEST.json",
            "publication_manifest_generation": "123", "publication_manifest_canonical_sha256": "sha256:" + "2" * 64,
            "publication_manifest_canonical_pin_verified": True, "source_table_uri": self.table_uri,
            "source_table_generation": "456", "source_table_size_bytes": 10000000,
            "source_table_sha256_declared": "sha256:" + "3" * 64, "whole_source_table_sha256_reread": False,
            "raw_page_hash_verified": False, "range_receipt": self.pin(self.receipt_path),
            "projection_policy": "date_and_id_filter_in_Arrow_before_Python_conversion",
            "mixed_row_group_numeric_decoding_possible": True, "retained_manifest_pin_verified": False,
            "projected_compressed_bytes": 40000, "maximum_projected_compressed_bytes": 64 * 1024 * 1024}

    def pin(self, path):
        return {"path": path.name, "size_bytes": path.stat().st_size,
                "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(), "source_uri": self.table_uri}

    def write_projection(self):
        pq.write_table(pa.Table.from_pylist(self.rows), self.path)
        return self.pin(self.path)

    def run_case(self, **changes):
        args = dict(prices=self.prices, evidence_root=self.root,
                    projection_evidence=self.write_projection(), source_provenance=self.provenance,
                    native_mapping={"siblis_synthetic": "synthetic_native"},
                    review_notes="Reviewed synthetic-only exact original source ID and raw USD share units")
        args.update(changes)
        return prepare_published_alpaca_repair_declarations(**args)

    def test_projection_pin_is_verified_and_raw_lineage_not_promoted(self):
        result = self.run_case()
        self.assertEqual(len(result.repairs), 1)
        self.assertEqual(result.repairs[0]["provider"], "Alpaca SIP published projection")
        ledger = result.identity_and_control_ledger[0]
        self.assertFalse(ledger["raw_page_hash_verified"])
        self.assertFalse(ledger["whole_source_table_sha256_reread"])
        self.assertFalse(ledger["retained_manifest_pin_verified"])
        self.assertEqual(ledger["source_page_hash_retained_lineage"], "1" * 64)
        self.assertEqual(ledger["raw_price_differences"]["open"], 20.)
        self.assertNotEqual(ledger["source_page_hash_retained_lineage"], ledger["projection_evidence_sha256"])

    def test_end_to_end_published_overlay_preserves_original_rejection(self):
        result = self.run_case()
        overlay = apply_reviewed_ohlc_repairs(self.prices, result.repairs, evidence_root=self.root,
            policy=OhlcRepairPolicy(0., 0., .0001, "synthetic-frozen-policy"))
        self.assertEqual(overlay.prices.iloc[0].open, 100.)
        self.assertEqual(overlay.prices.iloc[0].adj_open, 80.)
        self.assertEqual(overlay.prices.iloc[0].original_open, 80.)
        self.assertEqual(overlay.prices.iloc[0].original_price_quality_status, "provider_ohlc_anomaly")

    def test_2020_artifact_rejected_before_any_ohlc_read(self):
        self.rows[0]["trading_date"] = "2020-01-02"
        real_class = pq.ParquetFile
        read_columns = []
        class WatchedFile:
            def __init__(self, path):
                self.inner = real_class(path)
                self.metadata = self.inner.metadata; self.schema_arrow = self.inner.schema_arrow
            def read(self, *, columns):
                read_columns.append(columns)
                if "open" in columns:
                    raise AssertionError("Financial projection must not be reached")
                return self.inner.read(columns=columns)
        with patch("pyarrow.parquet.ParquetFile", WatchedFile):
            with self.assertRaisesRegex(DataContractError, "2018–2019"):
                self.run_case()
        self.assertEqual(len(read_columns), 1)

    def test_raw_feed_and_timeframe_mismatch_fail_before_ohlc(self):
        for field, value in (("adjustment", "split"), ("feed", "iex"), ("timeframe", "1Hour")):
            with self.subTest(field=field):
                original = self.rows[0][field]; self.rows[0][field] = value
                with self.assertRaisesRegex(DataContractError, "raw SIP"):
                    self.run_case()
                self.rows[0][field] = original

    def test_unadmitted_identity_and_missing_key_remain_separate_states(self):
        self.rows[0]["security_id"] = "unadmitted_source"
        result = self.run_case()
        self.assertEqual(result.repairs, ())
        self.assertEqual(result.identity_and_control_ledger[0]["status"], "NO_ORIGINAL_ADMITTED_SOURCE_ID")
        self.rows[0]["security_id"] = "siblis_synthetic"
        result = self.run_case(prices=self.prices.iloc[:0])
        self.assertEqual(result.repairs, ())
        self.assertEqual(result.identity_and_control_ledger[0]["status"], "NEW_KEY_SOURCE_ADMISSION_REQUIRED")

    def test_overlap_control_does_not_change_accepted_original(self):
        self.prices.loc[0, "price_quality_status"] = "accepted"
        result = self.run_case()
        self.assertEqual(result.repairs, ())
        self.assertEqual(result.identity_and_control_ledger[0]["status"], "CONTROL_EXISTING_ACCEPTED_BAR")

    def test_source_generation_pin_verification_and_range_bound_are_explicit(self):
        for field, value in (("publication_manifest_canonical_pin_verified", False),
                ("raw_page_hash_verified", True), ("whole_source_table_sha256_reread", True),
                ("retained_manifest_pin_verified", True), ("source_table_generation", ""),
                ("mixed_row_group_numeric_decoding_possible", False),
                ("projected_compressed_bytes", 65 * 1024 * 1024)):
            with self.subTest(field=field):
                source = copy.deepcopy(self.provenance); source[field] = value
                with self.assertRaises(DataContractError):
                    self.run_case(source_provenance=source)

    def test_actual_projection_and_receipt_bytes_must_match_pins(self):
        pin = self.write_projection(); pin["sha256"] = "sha256:" + "0" * 64
        with self.assertRaisesRegex(DataContractError, "hash changed"):
            self.run_case(projection_evidence=pin)
        self.receipt_path.write_text("mutated receipt")
        with self.assertRaises(DataContractError):
            self.run_case()

    def test_selected_duplicate_keys_and_invalid_bars_are_not_cleaned(self):
        self.rows *= 2
        with self.assertRaisesRegex(DataContractError, "Duplicate selected"):
            self.run_case()
        self.rows = self.rows[:1]; self.rows[0]["open"] = 111.
        with self.assertRaisesRegex(DataContractError, "bounds fail"):
            self.run_case()

    def test_published_source_uri_cannot_change_from_projection(self):
        source = copy.deepcopy(self.provenance); source["source_table_uri"] = "gs://another/source.parquet"
        with self.assertRaisesRegex(DataContractError, "differs from declared"):
            self.run_case(source_provenance=source)


if __name__ == "__main__":
    unittest.main()
