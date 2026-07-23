#!/usr/bin/env python3
"""Resumably mirror private manifest-listed Parquet products from GCS.

The script never lists a bucket. It reads exact manifest URIs from ignored local
configuration, validates every relative destination, and downloads only missing
or size-mismatched objects.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import threading
import time
from typing import Iterable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DESTINATION = PROJECT_ROOT / "data" / "full_history"
DEFAULT_GCLOUD = PROJECT_ROOT / "scripts" / "gcloud"
ENV_FILE = PROJECT_ROOT / ".env"
PRODUCT_ENV = {
    "13f": "ASSET_EMBEDDINGS_13F_MANIFEST_URI",
    "nport": "ASSET_EMBEDDINGS_NPORT_MANIFEST_URI",
}
GIB = 1024**3


@dataclasses.dataclass(frozen=True)
class ObjectSpec:
    dataset: str
    uri: str
    relative_path: PurePosixPath
    destination: Path
    expected_bytes: int


@dataclasses.dataclass(frozen=True)
class ProductPlan:
    dataset: str
    manifest_uri: str
    manifest_bytes: bytes
    manifest: Mapping[str, object]
    objects: tuple[ObjectSpec, ...]

    @property
    def expected_bytes(self) -> int:
        return sum(item.expected_bytes for item in self.objects)


def load_local_env(path: Path = ENV_FILE) -> None:
    """Load simple KEY=VALUE entries without overriding exported variables."""
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip("'\"")
        os.environ.setdefault(key.strip(), value)


def require_gs_uri(value: str, variable: str) -> str:
    if not value.startswith("gs://") or value.count("/") < 3:
        raise ValueError(f"{variable} must contain an exact gs:// object URI")
    return value


def fetch_manifest(manifest_uri: str, gcloud: Path = DEFAULT_GCLOUD) -> tuple[bytes, dict]:
    completed = subprocess.run(
        [str(gcloud), "storage", "cat", manifest_uri],
        check=True,
        capture_output=True,
    )
    payload = completed.stdout
    manifest = json.loads(payload)
    if not isinstance(manifest, dict):
        raise ValueError("Manifest root must be a JSON object")
    return payload, manifest


def safe_relative_path(value: object) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise ValueError("Manifest relative_path must be a non-empty string")
    path = PurePosixPath(value)
    if path == PurePosixPath(".") or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe manifest relative_path: {value!r}")
    return path


def build_product_plan(
    dataset: str,
    manifest_uri: str,
    manifest_bytes: bytes,
    manifest: Mapping[str, object],
    destination_root: Path,
) -> ProductPlan:
    table_files = manifest.get("table_files")
    if not isinstance(table_files, list) or not table_files:
        raise ValueError(f"{dataset} manifest has no table_files")

    dataset_version = manifest.get("dataset_version")
    if not isinstance(dataset_version, str) or not dataset_version:
        raise ValueError(f"{dataset} manifest has no dataset_version")

    product_root = destination_root / dataset / dataset_version
    objects: list[ObjectSpec] = []
    seen_paths: set[PurePosixPath] = set()
    for entry in table_files:
        if not isinstance(entry, dict):
            raise ValueError(f"{dataset} manifest contains a non-object table entry")
        relative_path = safe_relative_path(entry.get("relative_path"))
        if relative_path in seen_paths:
            raise ValueError(f"{dataset} manifest repeats {relative_path}")
        seen_paths.add(relative_path)

        uri = entry.get("gcs_uri")
        expected_bytes = entry.get("bytes")
        if not isinstance(uri, str) or not uri.startswith("gs://"):
            raise ValueError(f"{dataset} manifest has an invalid object URI")
        if not isinstance(expected_bytes, int) or expected_bytes < 0:
            raise ValueError(f"{dataset} manifest has invalid byte metadata for {relative_path}")

        objects.append(
            ObjectSpec(
                dataset=dataset,
                uri=uri,
                relative_path=relative_path,
                destination=product_root.joinpath(*relative_path.parts),
                expected_bytes=expected_bytes,
            )
        )

    return ProductPlan(
        dataset=dataset,
        manifest_uri=manifest_uri,
        manifest_bytes=manifest_bytes,
        manifest=manifest,
        objects=tuple(objects),
    )


def missing_objects(objects: Iterable[ObjectSpec]) -> list[ObjectSpec]:
    return [
        item
        for item in objects
        if not item.destination.is_file()
        or item.destination.stat().st_size != item.expected_bytes
    ]


def nearest_existing_parent(path: Path) -> Path:
    candidate = path
    while not candidate.exists():
        candidate = candidate.parent
    return candidate


def assert_disk_capacity(destination: Path, required_bytes: int, reserve_gib: float) -> None:
    free_bytes = shutil.disk_usage(nearest_existing_parent(destination)).free
    reserve_bytes = int(reserve_gib * GIB)
    if required_bytes + reserve_bytes > free_bytes:
        raise RuntimeError(
            "Insufficient disk space: "
            f"{required_bytes / GIB:.1f} GiB required plus {reserve_gib:.1f} GiB reserve, "
            f"but only {free_bytes / GIB:.1f} GiB is free"
        )


class TokenProvider:
    def __init__(self, gcloud: Path = DEFAULT_GCLOUD, refresh_after_seconds: int = 2400):
        self.gcloud = gcloud
        self.refresh_after_seconds = refresh_after_seconds
        self._token: str | None = None
        self._created_at = 0.0
        self._lock = threading.Lock()

    def invalidate(self) -> None:
        with self._lock:
            self._token = None
            self._created_at = 0.0

    def get(self) -> str:
        with self._lock:
            age = time.monotonic() - self._created_at
            if self._token is None or age >= self.refresh_after_seconds:
                completed = subprocess.run(
                    [str(self.gcloud), "auth", "print-access-token"],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                self._token = completed.stdout.strip()
                self._created_at = time.monotonic()
            return self._token


def gcs_media_url(uri: str) -> str:
    without_scheme = uri.removeprefix("gs://")
    bucket, separator, object_name = without_scheme.partition("/")
    if not separator or not bucket or not object_name:
        raise ValueError("Expected a complete gs://bucket/object URI")
    return (
        "https://storage.googleapis.com/download/storage/v1/b/"
        f"{quote(bucket, safe='')}/o/{quote(object_name, safe='')}?alt=media"
    )


def download_object(
    item: ObjectSpec,
    token_provider: TokenProvider,
    *,
    retries: int = 4,
    chunk_bytes: int = 8 * 1024 * 1024,
) -> ObjectSpec:
    if item.destination.is_file() and item.destination.stat().st_size == item.expected_bytes:
        return item

    item.destination.parent.mkdir(parents=True, exist_ok=True)
    partial = item.destination.with_name(item.destination.name + ".partial")
    if partial.exists() and partial.stat().st_size > item.expected_bytes:
        partial.unlink()

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"Authorization": f"Bearer {token_provider.get()}"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = Request(gcs_media_url(item.uri), headers=headers)

        try:
            with urlopen(request, timeout=120) as response:
                status = getattr(response, "status", response.getcode())
                append = offset > 0 and status == 206
                mode = "ab" if append else "wb"
                with partial.open(mode) as destination:
                    while True:
                        chunk = response.read(chunk_bytes)
                        if not chunk:
                            break
                        destination.write(chunk)

            actual_bytes = partial.stat().st_size
            if actual_bytes != item.expected_bytes:
                raise IOError(
                    f"Size mismatch for {item.relative_path}: "
                    f"expected {item.expected_bytes}, received {actual_bytes}"
                )
            os.replace(partial, item.destination)
            return item
        except HTTPError as exc:
            last_error = exc
            if exc.code in {401, 403}:
                token_provider.invalidate()
        except (URLError, TimeoutError, OSError) as exc:
            last_error = exc

        if attempt < retries:
            time.sleep(min(2**attempt, 15))

    raise RuntimeError(f"Failed to download {item.relative_path}") from last_error


def save_manifest(plan: ProductPlan, destination_root: Path) -> None:
    version = str(plan.manifest["dataset_version"])
    destination = destination_root / plan.dataset / version / "MANIFEST.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    temporary.write_bytes(plan.manifest_bytes)
    os.replace(temporary, destination)


def format_gib(value: int) -> str:
    return f"{value / GIB:.2f} GiB"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=["13f", "nport", "all"],
        default="all",
        help="Product to synchronize",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=DEFAULT_DESTINATION,
        help="Ignored local mirror root",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--minimum-free-gib", type=float, default=30.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.minimum_free_gib < 0:
        raise ValueError("--minimum-free-gib cannot be negative")

    load_local_env()
    datasets = list(PRODUCT_ENV) if args.dataset == "all" else [args.dataset]
    destination = args.destination.expanduser().resolve()

    plans: list[ProductPlan] = []
    for dataset in datasets:
        variable = PRODUCT_ENV[dataset]
        manifest_uri = require_gs_uri(os.environ.get(variable, ""), variable)
        manifest_bytes, manifest = fetch_manifest(manifest_uri)
        plans.append(
            build_product_plan(
                dataset,
                manifest_uri,
                manifest_bytes,
                manifest,
                destination,
            )
        )

    all_objects = [item for plan in plans for item in plan.objects]
    missing = missing_objects(all_objects)
    expected_bytes = sum(item.expected_bytes for item in all_objects)
    missing_bytes = sum(item.expected_bytes for item in missing)

    print(
        f"Plan: {len(all_objects):,} files, {format_gib(expected_bytes)} total; "
        f"{len(missing):,} files, {format_gib(missing_bytes)} missing"
    )
    for plan in plans:
        plan_missing = missing_objects(plan.objects)
        print(
            f"  {plan.dataset}: {len(plan.objects):,} files, "
            f"{format_gib(plan.expected_bytes)}; {len(plan_missing):,} missing"
        )

    if args.verify_only:
        return 0 if not missing else 1
    if args.dry_run:
        assert_disk_capacity(destination, missing_bytes, args.minimum_free_gib)
        print("Dry run passed disk-capacity and manifest validation.")
        return 0

    assert_disk_capacity(destination, missing_bytes, args.minimum_free_gib)
    for plan in plans:
        save_manifest(plan, destination)

    if not missing:
        print("Local mirror already matches the manifests by file size.")
        return 0

    token_provider = TokenProvider()
    completed_files = 0
    completed_bytes = 0
    progress_lock = threading.Lock()

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download_object, item, token_provider): item for item in missing
        }
        for future in concurrent.futures.as_completed(futures):
            item = futures[future]
            try:
                future.result()
            except Exception:
                for pending in futures:
                    pending.cancel()
                raise
            with progress_lock:
                completed_files += 1
                completed_bytes += item.expected_bytes
                if completed_files == 1 or completed_files % 25 == 0 or completed_files == len(missing):
                    print(
                        f"Downloaded {completed_files:,}/{len(missing):,} files "
                        f"({format_gib(completed_bytes)} of {format_gib(missing_bytes)})",
                        flush=True,
                    )

    remaining = missing_objects(all_objects)
    if remaining:
        raise RuntimeError(f"Verification failed: {len(remaining)} files remain incomplete")
    print("Full-history mirror verified against manifest file sizes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
