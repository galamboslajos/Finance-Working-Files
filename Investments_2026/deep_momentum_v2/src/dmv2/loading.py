"""Bounded projected reads, with an output guard around the protected holdout.

Parquet decodes compressed row groups. A selected group can contain excluded
dates, so a read may transiently decode protected values. The date/ID mask is
applied in Arrow before excluded financial values reach Python, summaries,
joins, or returned tables. This is an output/analysis guard, not physical
separation of the source file into development and holdout partitions.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
import re
from typing import Any, Mapping, Sequence

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .catalog import CatalogError, CatalogObject, SourceCatalog, canonical_sha256


HOLDOUT_START = date(2024, 1, 1)
FINANCIAL_COLUMNS = frozenset({
    "open", "high", "low", "close", "volume", "adj_open", "adj_high", "adj_low",
    "adj_close", "adj_volume", "factor", "split_ratio", "dividend", "change",
    "percent_change", "fifty_two_week_high", "fifty_two_week_low",
    "raw_high_gap", "raw_low_gap", "adjusted_high_gap", "adjusted_low_gap",
})


class ReadError(ValueError):
    """A requested read violates the source or bounded-read contract."""


class BudgetExceeded(ReadError):
    """The whole request is rejected instead of returning a truncated sample."""


class ProtectedDataError(ReadError):
    """A financial-value request enters the protected holdout."""


@dataclass(frozen=True)
class ReadBudget:
    max_metadata_files: int = 128
    max_data_files: int = 8
    max_scanned_rows: int = 1_000_000
    max_output_rows: int = 100_000
    max_read_bytes: int = 128 * 1024 * 1024

    def __post_init__(self):
        for name, value in asdict(self).items():
            if type(value) is not int or value <= 0:
                raise ReadError(f"Budget {name} must be a positive integer")


@dataclass(frozen=True)
class LoadedTable:
    table: pa.Table
    provenance: dict[str, Any]
    stats: dict[str, int]


def parse_date(value: str | date, name: str = "date") -> date:
    if type(value) is date:
        return value
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ReadError(f"{name} requires an ISO calendar date YYYY-MM-DD")
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise ReadError(f"Invalid {name}: {value}") from error


def _columns(value: Sequence[str], name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise ReadError(f"{name} must be an explicit sequence of field names")
    result = tuple(value)
    if any(not isinstance(item, str) or not item for item in result):
        raise ReadError(f"{name} contains an invalid field name")
    if len(set(result)) != len(result):
        raise ReadError(f"{name} contains duplicate field names")
    return result


def _check_keys(table: pa.Table, keys: tuple[str, ...]) -> None:
    if not keys:
        return
    # Only metadata keys are converted. Financial columns are never inspected here.
    rows = table.select(keys).to_pylist()
    seen = set()
    for row in rows:
        key = tuple(row[name] for name in keys)
        if any(value is None or (isinstance(value, str) and not value.strip()) for value in key):
            raise ReadError("Null or blank required key; no rows have been silently dropped")
        if key in seen:
            raise ReadError("Duplicate required key; no rows have been silently deduplicated")
        seen.add(key)


def _stats_bounds(group, field: str):
    for index in range(group.num_columns):
        column = group.column(index)
        if column.path_in_schema == field:
            return column.statistics
    return None


def _compressed_bytes(group, columns: Sequence[str]) -> int:
    names = set(columns)
    return sum(group.column(index).total_compressed_size
               for index in range(group.num_columns)
               if group.column(index).path_in_schema in names)


class BoundedTableReader:
    """A local manifest-backed reader; row/file/byte budgets apply to the whole request.

    Caller-reviewed allowed_columns controls projection and filter fields. Use
    financial=True whenever columns contain financial observations. Missing
    rows remain absent in the source read: the caller must join to its historical
    eligible cohort/grid and retain missing-price statuses and denominators.
    """

    def __init__(self, catalog: SourceCatalog, budget: ReadBudget | None = None):
        self.catalog = catalog
        self.budget = budget or ReadBudget()

    def _footers(self, source: str, table: str):
        objects = self.catalog.table_objects(source, table)
        if len(objects) > self.budget.max_metadata_files:
            raise BudgetExceeded(f"Metadata file budget exceeded: required {len(objects)}, "
                                 f"limit {self.budget.max_metadata_files}")
        schema = None
        results = []
        for obj in objects:
            obj.check()
            with pq.ParquetFile(obj.path) as parquet:
                current = parquet.schema_arrow.remove_metadata()
                self.catalog.validate_schema(source, table, current)
                if schema is not None and schema != current:
                    raise CatalogError(f"Inconsistent schemas: {source}/{table}")
                schema = current
                results.append((obj, parquet.metadata))
        declared_rows = self.catalog.table_definition(source, table).get("row_count")
        actual_rows = sum(metadata.num_rows for _, metadata in results)
        if type(declared_rows) is not int or declared_rows < 0 or actual_rows != declared_rows:
            raise CatalogError(f"Manifest/footer row counts differ: {source}/{table}")
        return schema, results

    def _request(self, schema, columns, allowed_columns, extra=()):
        columns = _columns(columns, "columns")
        allowed = _columns(allowed_columns, "allowed_columns")
        if not columns:
            raise ReadError("Read requires at least one explicitly projected column")
        requested = tuple(dict.fromkeys((*extra, *columns)))
        if set(requested) - set(allowed):
            raise ReadError("Projection/filter fields exceed the reviewed allowed-column list")
        if set(requested) - set(schema.names):
            raise ReadError("Requested fields are absent from the pinned source schema")
        return requested

    def _budget(self, groups, projected_columns):
        files = len({obj.relative_path for obj, _, _ in groups})
        rows = sum(group.num_rows for _, _, group in groups)
        size = sum(_compressed_bytes(group, projected_columns) for _, _, group in groups)
        if files > self.budget.max_data_files:
            raise BudgetExceeded(f"Data file budget exceeded: required {files}, limit {self.budget.max_data_files}; "
                                 f"scanned_rows={rows}, projected_compressed_bytes={size}")
        if rows > self.budget.max_scanned_rows:
            raise BudgetExceeded(f"Scanned row budget exceeded: required {rows}, limit {self.budget.max_scanned_rows}; "
                                 f"data_files={files}, projected_compressed_bytes={size}; no partial results returned")
        if size > self.budget.max_read_bytes:
            raise BudgetExceeded(f"Projected compressed-byte budget exceeded: required {size}, "
                                 f"limit {self.budget.max_read_bytes}; data_files={files}, scanned_rows={rows}")
        return files, rows, size

    def _provenance(self, source, table, columns, objects, request, financial):
        release = self.catalog.source(source)
        pins = {obj.relative_path: obj for obj in objects}
        return {
            "source_key": source,
            "dataset_id": release.manifest.get("dataset_id"),
            "release_id": release.manifest["release_id"],
            "manifest_sha256": release.manifest_sha256,
            "table": table,
            "columns": list(columns),
            "request": request,
            "budget": asdict(self.budget),
            "budget_semantics": (
                "Scanned-row limits include date/key scans of candidate row groups. Byte limits are "
                "an upper bound on projected compressed column chunks, not total disk I/O or RAM. "
                "Footer reads are counted separately by metadata-file limit."),
            "objects": [{"relative_path": obj.relative_path, "size_bytes": obj.size_bytes,
                         "sha256": obj.sha256}
                        for _, obj in sorted(pins.items())],
            "integrity": "canonical manifest pin and object sizes; object hashes not reread",
            "holdout_policy": (
                "Requests end before 2024-01-01. Date/ID masks applied in Arrow before output or analysis. "
                "Selected Parquet groups may transiently decode excluded financial values."
                if financial else "Projected metadata only; not a financial-value read."),
        }

    def read(self, source: str, table: str, *, start: str | date, end: str | date,
             columns: Sequence[str], allowed_columns: Sequence[str],
             date_column: str = "trading_date", key_columns: Sequence[str] = (),
             filters: Mapping[str, Sequence[Any]] | None = None,
             financial: bool = False) -> LoadedTable:
        """Inclusive date bounds, explicit projection and optional metadata-key filters.

        Date/key columns are read first. Only candidate row groups containing
        requested rows have financial projections read; filtering occurs in
        Arrow before those columns are returned. Strict null-date/key and
        duplicate-key failures preserve evidence instead of cleaning it away.
        """
        columns = _columns(columns, "columns")
        lower, upper = parse_date(start, "start"), parse_date(end, "end")
        if lower > upper:
            raise ReadError("start is later than end")
        # Public generic callers cannot bypass the holdout guard by omitting a flag.
        financial = bool(financial or set(columns) & FINANCIAL_COLUMNS)
        if financial and upper >= HOLDOUT_START:
            raise ProtectedDataError("Financial reads must end before protected 2024-01-01")
        if not isinstance(date_column, str) or not date_column:
            raise ReadError("An explicit date column is required")
        keys = _columns(key_columns, "key_columns")
        filter_values = {}
        for name, values in (filters or {}).items():
            if not isinstance(name, str) or not name or isinstance(values, (str, bytes)):
                raise ReadError("Filters require field names and explicit value sequences")
            values = tuple(values)
            if not values or any(value is None for value in values):
                raise ReadError("Filter values must be nonempty and non-null")
            filter_values[name] = values
        schema, footers = self._footers(source, table)
        metadata_columns = tuple(dict.fromkeys((date_column, *keys, *sorted(filter_values))))
        if set(metadata_columns) & FINANCIAL_COLUMNS:
            raise ReadError("Financial fields cannot serve as first-pass dates, metadata keys or filters")
        if financial and (source != "intrinio" or table != "intrinio_daily_prices" or
                          date_column != "trading_date" or
                          set(metadata_columns) - {"trading_date", "intrinio_security_id"}):
            raise ReadError("Financial reads require the reviewed price/date/stable-ID contract")
        projected = self._request(schema, columns, allowed_columns, metadata_columns)
        groups = []
        for obj, metadata in footers:
            for index in range(metadata.num_row_groups):
                group = metadata.row_group(index)
                bounds = _stats_bounds(group, date_column)
                if bounds is not None and bounds.null_count:
                    raise ReadError("Null dates in candidate source; date eligibility cannot be inferred")
                if bounds is not None and bounds.has_min_max:
                    minimum = parse_date(bounds.min, "source date minimum")
                    maximum = parse_date(bounds.max, "source date maximum")
                    if maximum < lower or minimum > upper:
                        continue
                skip = False
                for name, values in filter_values.items():
                    stats = _stats_bounds(group, name)
                    if stats is not None and stats.has_min_max:
                        try:
                            skip = all(value < stats.min or value > stats.max for value in values)
                        except TypeError as error:
                            raise ReadError(f"Filter value dtype differs: {name}") from error
                        if skip:
                            break
                if not skip:
                    groups.append((obj, index, group))
        files, scanned_rows, projected_bytes = self._budget(groups, projected)
        results = []
        returned = 0
        read_objects: list[CatalogObject] = []
        for obj, index, group in groups:
            obj.check()
            with pq.ParquetFile(obj.path) as parquet:
                metadata_table = parquet.read_row_group(index, columns=list(metadata_columns))
                values = metadata_table[date_column].to_pylist()
                dates = [parse_date(value, "source date") for value in values]
                for key in keys:
                    key_values = metadata_table[key].to_pylist()
                    if any(value is None or (isinstance(value, str) and not value.strip())
                           for value in key_values):
                        raise ReadError("Null or blank required key; no rows silently dropped")
                mask = pa.array([lower <= value <= upper for value in dates], type=pa.bool_())
                for name, values in filter_values.items():
                    try:
                        allowed = pa.array(values, type=schema.field(name).type)
                        mask = pc.and_(mask, pc.fill_null(pc.is_in(metadata_table[name], value_set=allowed), False))
                    except (pa.ArrowInvalid, pa.ArrowTypeError) as error:
                        raise ReadError(f"Filter value dtype differs: {name}") from error
                count = pc.sum(pc.cast(mask, pa.int64())).as_py() or 0
                returned += count
                if returned > self.budget.max_output_rows:
                    raise BudgetExceeded(f"Output row budget exceeded: at least {returned}, "
                                         f"limit {self.budget.max_output_rows}; no truncated success")
                read_objects.append(obj)
                if count:
                    extra_columns = [name for name in projected if name not in metadata_columns]
                    if extra_columns:
                        # Excluded financial values never undergo Python conversion or analysis.
                        extra = parquet.read_row_group(index, columns=extra_columns).filter(mask)
                        selected = metadata_table.filter(mask)
                        for name in extra_columns:
                            selected = selected.append_column(name, extra[name])
                    else:
                        selected = metadata_table.filter(mask)
                    results.append(selected.select(projected))
            obj.check()
        result = pa.concat_tables(results) if results else pa.Table.from_batches([], schema=pa.schema(
            [schema.field(name) for name in projected]))
        _check_keys(result, keys)
        request = {"start": lower.isoformat(), "end": upper.isoformat(), "date_column": date_column,
                   "key_columns": list(keys),
                   "filters": {name: sorted(set(values)) for name, values in sorted(filter_values.items())}}
        provenance = self._provenance(source, table, projected, read_objects, request, financial)
        provenance["request_sha256"] = canonical_sha256(provenance)
        return LoadedTable(result, provenance, {"metadata_files": len(footers), "data_files": files,
                                               "scanned_rows": scanned_rows, "returned_rows": result.num_rows,
                                               "projected_compressed_bytes": projected_bytes})

    def read_metadata(self, source: str, table: str, *, columns: Sequence[str],
                      allowed_columns: Sequence[str], max_rows: int = 30_000,
                      key_columns: Sequence[str] = ()) -> LoadedTable:
        """Read a small reviewed metadata table with upfront manifest/footer limits.

        This is for identity/link/calendar metadata, never price or financial
        observations. It does not establish historical availability of a field.
        """
        if type(max_rows) is not int or max_rows <= 0:
            raise ReadError("max_rows must be a positive integer")
        columns = _columns(columns, "columns")
        keys = _columns(key_columns, "key_columns")
        if (set(columns) | set(keys)) & FINANCIAL_COLUMNS:
            raise ReadError("Financial observations are not eligible for metadata-table reads")
        definition = self.catalog.table_definition(source, table)
        declared_rows = definition.get("row_count")
        if type(declared_rows) is not int or declared_rows < 0:
            raise ReadError("Metadata read requires an explicit manifest row_count")
        if declared_rows > max_rows or declared_rows > self.budget.max_output_rows:
            raise BudgetExceeded(f"Manifest metadata row budget exceeded: required {declared_rows}, "
                                 f"limit {min(max_rows, self.budget.max_output_rows)}")
        if source == "intrinio" and table == "intrinio_daily_prices":
            raise ReadError("Financial price table is not eligible for metadata-table reads")
        schema, footers = self._footers(source, table)
        projected = self._request(schema, columns, allowed_columns, keys)
        groups = [(obj, index, metadata.row_group(index)) for obj, metadata in footers
                  for index in range(metadata.num_row_groups)]
        files, rows, size = self._budget(groups, projected)
        if rows != declared_rows:
            raise CatalogError("Manifest/footer metadata row counts differ")
        results = []
        for obj, index, _ in groups:
            with pq.ParquetFile(obj.path) as parquet:
                results.append(parquet.read_row_group(index, columns=list(projected)))
            obj.check()
        result = pa.concat_tables(results) if results else pa.Table.from_batches(
            [], schema=pa.schema([schema.field(name) for name in projected]))
        _check_keys(result, keys)
        request = {"metadata_only": True, "max_rows": max_rows, "key_columns": list(keys)}
        provenance = self._provenance(source, table, projected, [obj for obj, _, _ in groups], request, False)
        provenance["request_sha256"] = canonical_sha256(provenance)
        return LoadedTable(result, provenance, {"metadata_files": len(footers), "data_files": files,
                                               "scanned_rows": rows, "returned_rows": rows,
                                               "projected_compressed_bytes": size})


PRICE_COLUMNS = (
    "intrinio_security_id", "trading_date", "open", "high", "low", "close", "volume",
    "adj_open", "adj_high", "adj_low", "adj_close", "adj_volume", "factor", "split_ratio",
    "dividend", "raw_ohlc_bounds_valid", "adjusted_ohlc_bounds_valid", "price_quality_status",
    "frequency", "intraperiod",
)


class PriceLoader:
    """Load explicit historical security IDs without choosing survivors or outcomes."""

    def __init__(self, catalog: SourceCatalog, budget: ReadBudget | None = None):
        self.reader = BoundedTableReader(catalog, budget)

    def load(self, security_ids: Sequence[str], start: str | date, end: str | date,
             columns: Sequence[str]) -> LoadedTable:
        if isinstance(security_ids, (str, bytes)):
            raise ReadError("security_ids must be an explicit nonempty sequence")
        ids = tuple(security_ids)
        if not ids or any(not isinstance(value, str) or not value.strip() for value in ids):
            raise ReadError("security_ids must contain nonempty stable provider IDs")
        if len(set(ids)) != len(ids):
            raise ReadError("security_ids contains duplicate requested IDs")
        return self.reader.read("intrinio", "intrinio_daily_prices", start=start, end=end,
                                columns=columns, allowed_columns=PRICE_COLUMNS,
                                key_columns=("intrinio_security_id", "trading_date"),
                                filters={"intrinio_security_id": ids}, financial=True)
