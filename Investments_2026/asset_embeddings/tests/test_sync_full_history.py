import json
from pathlib import Path, PurePosixPath
import tempfile
import unittest

from scripts.sync_full_history import (
    build_product_plan,
    gcs_media_url,
    missing_objects,
    safe_relative_path,
)


class SyncFullHistoryTests(unittest.TestCase):
    def manifest(self):
        return {
            "dataset_version": "v1",
            "table_files": [
                {
                    "gcs_uri": "gs://private-bucket/product/tables/a.parquet",
                    "relative_path": "tables/a.parquet",
                    "bytes": 4,
                    "table": "holdings",
                },
                {
                    "gcs_uri": "gs://private-bucket/product/tables/b.parquet",
                    "relative_path": "tables/b.parquet",
                    "bytes": 3,
                    "table": "holdings",
                },
            ],
        }

    def test_safe_relative_path_rejects_traversal_and_absolute_paths(self):
        for value in ("../secret", "tables/../../secret", "/tmp/secret", ".", ""):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    safe_relative_path(value)

    def test_plan_preserves_manifest_structure_under_dataset_version(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.manifest()
            payload = json.dumps(manifest).encode()
            plan = build_product_plan(
                "13f",
                "gs://private-bucket/product/MANIFEST.json",
                payload,
                manifest,
                root,
            )

            self.assertEqual(plan.expected_bytes, 7)
            self.assertEqual(len(plan.objects), 2)
            self.assertEqual(
                plan.objects[0].destination,
                root / "13f" / "v1" / "tables" / "a.parquet",
            )
            self.assertEqual(
                plan.objects[0].relative_path,
                PurePosixPath("tables/a.parquet"),
            )

    def test_missing_objects_skips_only_exact_size_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.manifest()
            plan = build_product_plan(
                "nport",
                "gs://private-bucket/product/MANIFEST.json",
                json.dumps(manifest).encode(),
                manifest,
                root,
            )
            first, second = plan.objects
            first.destination.parent.mkdir(parents=True)
            first.destination.write_bytes(b"1234")
            second.destination.write_bytes(b"xx")

            self.assertEqual(missing_objects(plan.objects), [second])

    def test_plan_rejects_duplicate_destinations(self):
        manifest = self.manifest()
        manifest["table_files"][1]["relative_path"] = "tables/a.parquet"
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "repeats"):
                build_product_plan(
                    "13f",
                    "gs://private-bucket/product/MANIFEST.json",
                    json.dumps(manifest).encode(),
                    manifest,
                    Path(directory),
                )

    def test_media_url_encodes_object_name(self):
        url = gcs_media_url("gs://private-bucket/folder name/a+b.parquet")
        self.assertEqual(
            url,
            "https://storage.googleapis.com/download/storage/v1/b/private-bucket/"
            "o/folder%20name%2Fa%2Bb.parquet?alt=media",
        )


if __name__ == "__main__":
    unittest.main()
