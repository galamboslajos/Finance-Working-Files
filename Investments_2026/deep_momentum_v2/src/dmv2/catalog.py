"""Pinned, local-only source inventory. A directory listing is never a data source."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, Iterator


class CatalogError(ValueError):
    """A configuration or source contract is invalid."""


class IntegrityError(CatalogError):
    """An input does not match its pinned release."""


_HASH = re.compile(r"sha256:[0-9a-f]{64}\Z")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CatalogError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    except (OSError, json.JSONDecodeError) as error:
        raise CatalogError(f"Cannot read JSON configuration/manifest: {path}") from error
    if not isinstance(value, dict):
        raise CatalogError("Configuration/manifest must be a JSON object")
    return value


def canonical_sha256(value: Any) -> str:
    """Match the existing release pin: sorted UTF-8 JSON, no whitespace."""
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                     allow_nan=False).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _hash_pin(value: Any) -> str:
    if not isinstance(value, str) or not _HASH.fullmatch(value):
        raise CatalogError("SHA-256 pins must be sha256: followed by 64 lowercase hex digits")
    return value


def _relative_path(base: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise CatalogError("Input paths must be nonempty relative POSIX paths")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or ":" in pure.parts[0]:
        raise CatalogError("Input path leaves its approved directory")
    path = (base / relative).resolve()
    if not path.is_relative_to(base):
        raise CatalogError("Input path leaves its approved directory (including symlinks)")
    return path


@dataclass(frozen=True)
class CatalogObject:
    source_key: str
    relative_path: str
    path: Path
    size_bytes: int
    sha256: str

    def check(self, hashes: bool = False) -> None:
        if not self.path.is_file() or self.path.stat().st_size != self.size_bytes:
            raise IntegrityError(f"Missing or wrong-size input: {self.relative_path}")
        if hashes:
            digest = hashlib.sha256()
            with self.path.open("rb") as stream:
                for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                    digest.update(block)
            if "sha256:" + digest.hexdigest() != self.sha256:
                raise IntegrityError(f"Input hash mismatch: {self.relative_path}")


@dataclass(frozen=True)
class SourceRelease:
    key: str
    config: dict[str, Any]
    base: Path
    manifest: dict[str, Any]
    manifest_sha256: str
    objects: tuple[CatalogObject, ...]


@dataclass(frozen=True)
class IntegrityReport:
    source_key: str
    label: str
    release_id: str
    object_count: int
    size_bytes: int
    hashes_checked: bool


class SourceCatalog:
    """Resolve exclusively pinned objects inside the local project/release roots.

    Construction verifies canonical manifest hashes and path/object contracts.
    Object sizes are checked at read time; verify(hashes=True) explicitly reads
    raw bytes for complete integrity checking and does not inspect price values.
    """

    def __init__(self, root: str | Path, config_path: str | Path | None = None):
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise CatalogError("Project root does not exist")
        if config_path is None:
            path = self.root / "configs/data_sources.local.json"
        else:
            candidate = Path(config_path)
            path = candidate.resolve() if candidate.is_absolute() else _relative_path(self.root, str(candidate))
        if not path.is_relative_to(self.root):
            raise CatalogError("Source configuration leaves the project directory")
        config = _read_json(path)
        entries = config.get("sources")
        if not isinstance(entries, list) or not entries:
            raise CatalogError("Source configuration requires a nonempty sources list")
        releases: list[SourceRelease] = []
        seen: set[str] = set()
        for entry in entries:
            if not isinstance(entry, dict):
                raise CatalogError("Each source must be a JSON object")
            key = entry.get("key")
            if not isinstance(key, str) or not key or key in seen:
                raise CatalogError("Source keys must be nonempty and unique")
            seen.add(key)
            manifest_path = _relative_path(self.root, entry.get("manifest"))
            manifest = _read_json(manifest_path)
            pin = _hash_pin(entry.get("manifest_sha256"))
            if canonical_sha256(manifest) != pin:
                raise IntegrityError(f"Manifest pin mismatch: {key}")
            if not isinstance(manifest.get("release_id"), str) or not manifest["release_id"]:
                raise CatalogError(f"Source requires release_id: {key}")
            raw_objects = manifest.get("objects")
            if not isinstance(raw_objects, list) or not raw_objects:
                raise CatalogError(f"Source requires pinned objects: {key}")
            objects: list[CatalogObject] = []
            names: set[str] = set()
            base = manifest_path.parent
            for obj in raw_objects:
                if not isinstance(obj, dict):
                    raise CatalogError("Manifest objects must be JSON objects")
                relative = obj.get("relative_path")
                object_path = _relative_path(base, relative)
                canonical_relative = object_path.relative_to(base).as_posix()
                if canonical_relative in names:
                    raise CatalogError(f"Duplicate manifest object path: {relative}")
                names.add(canonical_relative)
                size = obj.get("size_bytes")
                if type(size) is not int or size < 0:
                    raise CatalogError(f"Invalid manifest object size: {relative}")
                objects.append(CatalogObject(key, relative, object_path, size, _hash_pin(obj.get("sha256"))))
            tables = manifest.get("schema", {}).get("tables")
            if not isinstance(tables, list):
                raise CatalogError(f"Source requires declared table schemas: {key}")
            table_names: set[str] = set()
            for table in tables:
                if not isinstance(table, dict) or not isinstance(table.get("name"), str):
                    raise CatalogError("Invalid declared table")
                name = table["name"]
                if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) or name in table_names:
                    raise CatalogError("Table names must be safe and unique")
                table_names.add(name)
                columns = table.get("columns")
                if not isinstance(columns, list) or not columns:
                    raise CatalogError(f"Table requires column contract: {name}")
                names_list = [column.get("name") for column in columns if isinstance(column, dict)]
                if (len(names_list) != len(columns) or
                        any(not isinstance(name, str) or not name for name in names_list) or
                        len(set(names_list)) != len(names_list)):
                    raise CatalogError(f"Table columns must be named and unique: {name}")
            releases.append(SourceRelease(key, dict(entry), base, manifest, pin,
                                          tuple(sorted(objects, key=lambda obj: obj.relative_path))))
        self.sources = tuple(sorted(releases, key=lambda source: source.key))

    def __iter__(self) -> Iterator[SourceRelease]:
        return iter(self.sources)

    def source(self, key: str) -> SourceRelease:
        for source in self.sources:
            if source.key == key:
                return source
        raise CatalogError(f"Unconfigured source: {key}")

    def table_definition(self, source_key: str, table: str) -> dict[str, Any]:
        source = self.source(source_key)
        for item in source.manifest["schema"]["tables"]:
            if item["name"] == table:
                return item
        raise CatalogError(f"Undeclared table: {source_key}/{table}")

    def table_objects(self, source_key: str, table: str) -> tuple[CatalogObject, ...]:
        self.table_definition(source_key, table)
        prefix = f"tables/{table}/"
        objects = tuple(obj for obj in self.source(source_key).objects
                        if obj.relative_path.startswith(prefix) and obj.relative_path.endswith(".parquet"))
        if not objects:
            raise CatalogError(f"No manifest-listed Parquet objects: {source_key}/{table}")
        return objects

    def table_schema(self, source_key: str, table: str):
        """Inspect every pinned table footer; never decode data columns."""
        import pyarrow.parquet as pq
        definition = self.table_definition(source_key, table)
        schema = None
        rows = 0
        for obj in self.table_objects(source_key, table):
            obj.check()
            with pq.ParquetFile(obj.path) as parquet:
                current = parquet.schema_arrow.remove_metadata()
                rows += parquet.metadata.num_rows
            self.validate_schema(source_key, table, current)
            if schema is not None and current != schema:
                raise CatalogError(f"Inconsistent table schemas: {source_key}/{table}")
            schema = current
        expected_rows = definition.get("row_count")
        if type(expected_rows) is not int or expected_rows < 0 or rows != expected_rows:
            raise CatalogError(f"Manifest/footer row counts differ: {source_key}/{table}")
        return schema

    def validate_schema(self, source_key: str, table: str, schema) -> None:
        """Check the declared logical field types, allowing Arrow string widths."""
        import pyarrow as pa
        columns = self.table_definition(source_key, table)["columns"]
        if schema.names != [column["name"] for column in columns]:
            raise CatalogError(f"Manifest/physical columns differ: {source_key}/{table}")
        type_checks = {
            "str": lambda value: pa.types.is_string(value) or pa.types.is_large_string(value),
            "string": lambda value: pa.types.is_string(value) or pa.types.is_large_string(value),
            "double": pa.types.is_float64,
            "float64": pa.types.is_float64,
            "int64": pa.types.is_int64,
            "bool": pa.types.is_boolean,
            "date32": pa.types.is_date32,
            "date64": pa.types.is_date64,
        }
        for column in columns:
            dtype = column.get("dtype")
            if dtype not in type_checks or not type_checks[dtype](schema.field(column["name"]).type):
                raise CatalogError(f"Manifest/physical dtype differs: {source_key}/{table}/{column['name']}")

    def verify(self, hashes: bool = False) -> tuple[IntegrityReport, ...]:
        reports = []
        for source in self.sources:
            for obj in source.objects:
                obj.check(hashes=hashes)
            reports.append(IntegrityReport(source.key, source.config.get("label", source.key),
                                           source.manifest["release_id"], len(source.objects),
                                           sum(obj.size_bytes for obj in source.objects), hashes))
        return tuple(reports)

    def provenance(self) -> dict[str, Any]:
        """Deterministic source pins; no absolute local paths or credentials."""
        return {"sources": [{"source_key": source.key,
                             "dataset_id": source.manifest.get("dataset_id"),
                             "release_id": source.manifest["release_id"],
                             "manifest_sha256": source.manifest_sha256}
                            for source in self.sources]}
