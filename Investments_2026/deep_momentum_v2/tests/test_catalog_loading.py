"""Tiny structural fixtures only; these are not research market observations."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from dmv2.catalog import CatalogError, IntegrityError, SourceCatalog, canonical_sha256
from dmv2.loading import (BoundedTableReader, BudgetExceeded, PriceLoader,
                          ProtectedDataError, ReadBudget, ReadError)


def fixture(root: Path, rows=None, *, table="intrinio_daily_prices", groups=2):
    rows = rows if rows is not None else {
        "intrinio_security_id": ["fixture_A", "fixture_A", "fixture_B", "fixture_A"],
        "trading_date": ["2023-12-28", "2023-12-29", "2023-12-29", "2024-01-02"],
        "close": [1.0, 2.0, 3.0, 999_999.0],
        "volume": [10.0, 20.0, 30.0, 999_999.0],
    }
    base = root / "data/raw/test_only/release"
    path = base / f"tables/{table}/part-00000.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    arrow = pa.table(rows)
    pq.write_table(arrow, path, row_group_size=groups)
    dtypes = {"string": "string", "double": "double", "int64": "int64", "bool": "bool"}
    manifest = {
        "release_id": "structural-test-only", "dataset_id": "not-research-data",
        "schema": {"tables": [{"name": table, "row_count": arrow.num_rows,
                                "columns": [{"name": field.name, "dtype": dtypes[str(field.type)]}
                                            for field in arrow.schema]}]},
        "objects": [{"relative_path": path.relative_to(base).as_posix(),
                     "size_bytes": path.stat().st_size,
                     "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()}],
    }
    manifest_path = base / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest))
    config = {"sources": [{"key": "intrinio", "label": "Test-only input",
                           "manifest": manifest_path.relative_to(root).as_posix(),
                           "manifest_sha256": canonical_sha256(manifest)}]}
    (root / "configs").mkdir(exist_ok=True)
    (root / "configs/data_sources.local.json").write_text(json.dumps(config))
    return manifest_path, path


class CatalogLoadingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.manifest_path, self.data_path = fixture(self.root)

    def test_manifest_pin_and_deterministic_provenance(self):
        first, second = SourceCatalog(self.root), SourceCatalog(self.root)
        self.assertEqual(first.provenance(), second.provenance())
        self.assertEqual(first.verify(hashes=True)[0].object_count, 1)
        manifest = json.loads(self.manifest_path.read_text())
        manifest["release_id"] = "changed"
        self.manifest_path.write_text(json.dumps(manifest))
        with self.assertRaises(IntegrityError):
            SourceCatalog(self.root)

    def test_manifest_cannot_escape_release_or_duplicate_objects(self):
        manifest = json.loads(self.manifest_path.read_text())
        manifest["objects"][0]["relative_path"] = "../../outside.parquet"
        self._repin(manifest)
        with self.assertRaises(CatalogError):
            SourceCatalog(self.root)

    def _repin(self, manifest):
        self.manifest_path.write_text(json.dumps(manifest))
        path = self.root / "configs/data_sources.local.json"
        config = json.loads(path.read_text())
        config["sources"][0]["manifest_sha256"] = canonical_sha256(manifest)
        path.write_text(json.dumps(config))

    def test_extra_directory_file_is_not_part_of_release(self):
        unrelated = self.data_path.parent / "part-unlisted.parquet"
        unrelated.write_bytes(b"not a release input")
        catalog = SourceCatalog(self.root)
        self.assertEqual(len(catalog.table_objects("intrinio", "intrinio_daily_prices")), 1)

    def test_source_symlink_escape_is_rejected(self):
        outside = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: outside.rmdir())
        (self.root / "escaping").symlink_to(outside, target_is_directory=True)
        path = self.root / "configs/data_sources.local.json"
        config = json.loads(path.read_text())
        config["sources"][0]["manifest"] = "escaping/MANIFEST.json"
        path.write_text(json.dumps(config))
        with self.assertRaises(CatalogError):
            SourceCatalog(self.root)

    def test_pre_holdout_date_and_id_projection_with_mixed_rowgroup(self):
        # Group2 contains a protected sentinel, which must not leave Arrow filtering.
        loader = PriceLoader(SourceCatalog(self.root))
        result = loader.load(["fixture_A"], "2023-12-28", "2023-12-29", ["close", "volume"])
        self.assertEqual(result.table["close"].to_pylist(), [1.0, 2.0])
        self.assertEqual(result.table["volume"].to_pylist(), [10.0, 20.0])
        self.assertNotIn("999999", json.dumps(result.provenance))
        self.assertIn("transiently decode", result.provenance["holdout_policy"])
        repeated = loader.load(["fixture_A"], "2023-12-28", "2023-12-29", ["close", "volume"])
        self.assertEqual(result.provenance, repeated.provenance)

    def test_protected_request_rejected_before_data_read(self):
        with self.assertRaises(ProtectedDataError):
            PriceLoader(SourceCatalog(self.root)).load(["fixture_A"], "2023-12-29", "2024-01-01", ["close"])

    def test_generic_caller_cannot_omit_financial_flag_to_bypass_holdout(self):
        reader = BoundedTableReader(SourceCatalog(self.root))
        with self.assertRaises(ProtectedDataError):
            reader.read("intrinio", "intrinio_daily_prices", start="2023-12-29", end="2024-01-02",
                        columns=["close"], allowed_columns=["trading_date", "close"], financial=False)
        result = reader.read("intrinio", "intrinio_daily_prices", start="2023-12-28", end="2023-12-29",
                             columns=["close"], allowed_columns=["trading_date", "close"], financial=False)
        self.assertNotIn(999_999.0, result.table["close"].to_pylist())

    def test_financial_field_cannot_be_converted_as_filter_or_key(self):
        reader = BoundedTableReader(SourceCatalog(self.root))
        with self.assertRaises(ReadError):
            reader.read("intrinio", "intrinio_daily_prices", start="2023-12-28", end="2023-12-29",
                        columns=["close"], allowed_columns=["trading_date", "close"],
                        key_columns=["close"], financial=True)
        with self.assertRaises(ReadError):
            reader.read("intrinio", "intrinio_daily_prices", start="2023-12-28", end="2023-12-29",
                        columns=["close"], allowed_columns=["trading_date", "close"], filters={"close": [1.0]})
        with self.assertRaises(ReadError):
            reader.read("intrinio", "intrinio_daily_prices", start="2023-12-28", end="2023-12-29",
                        columns=["intrinio_security_id"],
                        allowed_columns=["trading_date", "intrinio_security_id", "close"],
                        key_columns=["close"], financial=False)

    def test_anomaly_financial_gaps_cannot_bypass_guard_as_metadata(self):
        fields = ["raw_high_gap", "raw_low_gap", "adjusted_high_gap", "adjusted_low_gap"]
        fixture(self.root, {"trading_date": ["2023-12-29", "2024-01-02"],
                            **{field: [1.0, 999_999.0] for field in fields}},
                table="price_anomaly_evidence")
        reader = BoundedTableReader(SourceCatalog(self.root))
        for field in fields:
            with self.subTest(field=field):
                with self.assertRaises(ReadError):
                    reader.read_metadata("intrinio", "price_anomaly_evidence", columns=[field],
                                         allowed_columns=[field])
                with self.assertRaises(ProtectedDataError):
                    reader.read("intrinio", "price_anomaly_evidence", start="2023-12-29", end="2024-01-02",
                                columns=[field], allowed_columns=["trading_date", field], financial=False)

    def test_projection_and_date_requests_are_strict(self):
        loader = PriceLoader(SourceCatalog(self.root))
        for start, end, columns in [("2023-12-30", "2023-12-29", ["close"]),
                                    ("2023-13-01", "2023-12-29", ["close"]),
                                    ("2023-12-28", "2023-12-29", ["delisted"]),
                                    ("2023-12-28", "2023-12-29", "close")]:
            with self.subTest(columns=columns, start=start):
                with self.assertRaises(ReadError):
                    loader.load(["fixture_A"], start, end, columns)

    def test_budget_failure_does_not_return_truncated_rows(self):
        catalog = SourceCatalog(self.root)
        for budget in [ReadBudget(max_scanned_rows=1), ReadBudget(max_output_rows=1),
                       ReadBudget(max_read_bytes=1)]:
            with self.subTest(budget=budget):
                with self.assertRaises(BudgetExceeded):
                    PriceLoader(catalog, budget).load(["fixture_A"], "2023-12-28", "2023-12-29", ["close"])

    def test_duplicate_and_null_keys_fail_without_cleaning(self):
        for ids, dates in [(["fixture_A", "fixture_A"], ["2023-12-29", "2023-12-29"]),
                           ([None, "fixture_A"], ["2023-12-28", "2023-12-29"]),
                           (["fixture_A", "fixture_A"], [None, "2023-12-29"])]:
            fixture(self.root, {"intrinio_security_id": ids, "trading_date": dates,
                                "close": [1.0, 2.0]})
            with self.subTest(ids=ids, dates=dates):
                with self.assertRaises(ReadError):
                    PriceLoader(SourceCatalog(self.root)).load(["fixture_A"], "2023-12-28", "2023-12-29", ["close"])

    def test_missing_security_is_retained_as_empty_source_result(self):
        result = PriceLoader(SourceCatalog(self.root)).load(["fixture_absent"], "2023-12-28", "2023-12-29", ["close"])
        self.assertEqual(result.table.num_rows, 0)
        self.assertEqual(result.table.column_names, ["trading_date", "intrinio_security_id", "close"])

    def test_small_metadata_read_checks_manifest_count_and_projection(self):
        fixture(self.root, {"identity": ["fixture_A", "fixture_B"], "known": [True, False]}, table="test_identity")
        reader = BoundedTableReader(SourceCatalog(self.root))
        with self.assertRaises(BudgetExceeded):
            reader.read_metadata("intrinio", "test_identity", columns=["identity"], allowed_columns=["identity"], max_rows=1)
        result = reader.read_metadata("intrinio", "test_identity", columns=["identity"], allowed_columns=["identity"], key_columns=["identity"])
        self.assertEqual(result.table.column_names, ["identity"])
        self.assertEqual(result.table.num_rows, 2)

    def test_schema_drift_is_an_error(self):
        manifest = json.loads(self.manifest_path.read_text())
        manifest["schema"]["tables"][0]["columns"][2]["dtype"] = "int64"
        self._repin(manifest)
        with self.assertRaises(CatalogError):
            SourceCatalog(self.root).table_schema("intrinio", "intrinio_daily_prices")

    def test_manifest_footer_row_count_drift_is_an_error(self):
        manifest = json.loads(self.manifest_path.read_text())
        manifest["schema"]["tables"][0]["row_count"] = 3
        self._repin(manifest)
        with self.assertRaises(CatalogError):
            SourceCatalog(self.root).table_schema("intrinio", "intrinio_daily_prices")
        with self.assertRaises(CatalogError):
            PriceLoader(SourceCatalog(self.root)).load(["fixture_A"], "2023-12-28", "2023-12-29", ["close"])


if __name__ == "__main__":
    unittest.main()
