"""Memory-conscious diagnostics for the local full-history 13F product.

The functions in this module inspect ignored local Parquet files. They return
aggregate diagnostics and bounded samples; they do not define production
cleaning, amendment, identifier, or universe rules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import pyarrow.parquet as pq

from src.holdings_exploration import require_columns


REQUIRED_FILING_COLUMNS = (
    "accession_no",
    "manager_cik",
    "form_type",
    "period_of_report",
    "filed_at",
    "availability_timestamp_utc",
    "is_amendment",
    "holdings_count",
)


def _parquet_files(table_directory: str | Path) -> list[Path]:
    directory = Path(table_directory).expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Parquet table directory does not exist: {directory}")
    files = sorted(directory.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet files found below: {directory}")
    return files


def parquet_table_inventory(product_root: str | Path) -> pd.DataFrame:
    """Read Parquet footers and summarize every table without loading payloads."""

    root = Path(product_root).expanduser().resolve()
    tables_root = root / "tables"
    if not tables_root.is_dir():
        raise FileNotFoundError(f"Expected a tables directory below: {root}")

    records: list[dict[str, object]] = []
    for table_directory in sorted(path for path in tables_root.iterdir() if path.is_dir()):
        files = _parquet_files(table_directory)
        rows = 0
        row_groups = 0
        storage_bytes = 0
        schema_signatures: set[tuple[tuple[str, str], ...]] = set()
        column_counts: set[int] = set()

        for path in files:
            parquet_file = pq.ParquetFile(path)
            rows += parquet_file.metadata.num_rows
            row_groups += parquet_file.metadata.num_row_groups
            storage_bytes += path.stat().st_size
            signature = tuple(
                (field.name, str(field.type)) for field in parquet_file.schema_arrow
            )
            schema_signatures.add(signature)
            column_counts.add(len(signature))

        records.append(
            {
                "table": table_directory.name,
                "parquet_files": len(files),
                "rows": rows,
                "columns_min": min(column_counts),
                "columns_max": max(column_counts),
                "schema_variants": len(schema_signatures),
                "row_groups": row_groups,
                "storage_gib": storage_bytes / 1024**3,
            }
        )

    return pd.DataFrame.from_records(records).sort_values("table").reset_index(drop=True)


def parquet_schema(table_directory: str | Path) -> pd.DataFrame:
    """Return column types and cross-file presence for a Parquet table."""

    files = _parquet_files(table_directory)
    ordered_columns: list[str] = []
    field_types: dict[str, set[str]] = {}
    files_present: dict[str, int] = {}

    for path in files:
        schema = pq.ParquetFile(path).schema_arrow
        present_in_file: set[str] = set()
        for field in schema:
            if field.name not in field_types:
                ordered_columns.append(field.name)
                field_types[field.name] = set()
                files_present[field.name] = 0
            field_types[field.name].add(str(field.type))
            present_in_file.add(field.name)
        for column in present_in_file:
            files_present[column] += 1

    return pd.DataFrame.from_records(
        [
            {
                "column": column,
                "arrow_types": " | ".join(sorted(field_types[column])),
                "files_present": files_present[column],
                "all_files": files_present[column] == len(files),
            }
            for column in ordered_columns
        ]
    )


def read_parquet_columns(
    table_directory: str | Path,
    columns: Sequence[str],
) -> pd.DataFrame:
    """Read selected columns from all partitions of a reasonably sized table."""

    files = _parquet_files(table_directory)
    requested = list(dict.fromkeys(columns))
    if not requested:
        raise ValueError("At least one column must be requested")

    frames: list[pd.DataFrame] = []
    for path in files:
        available = set(pq.ParquetFile(path).schema_arrow.names)
        missing = sorted(set(requested).difference(available))
        if missing:
            raise KeyError(
                f"A Parquet partition is missing requested columns: {', '.join(missing)}"
            )
        frames.append(pq.ParquetFile(path).read(columns=requested).to_pandas())
    return pd.concat(frames, ignore_index=True)


def prepare_13f_filings(filings: pd.DataFrame) -> pd.DataFrame:
    """Parse 13F timing fields and expose violations without dropping rows."""

    require_columns(filings, REQUIRED_FILING_COLUMNS, context="full-history 13F filings")
    prepared = filings.copy()
    prepared["report_date_utc"] = pd.to_datetime(
        prepared["period_of_report"], errors="coerce", utc=True
    )
    prepared["filed_at_utc"] = pd.to_datetime(
        prepared["filed_at"], errors="coerce", utc=True
    )
    prepared["available_at_utc"] = pd.to_datetime(
        prepared["availability_timestamp_utc"], errors="coerce", utc=True
    )
    prepared["holdings_count_numeric"] = pd.to_numeric(
        prepared["holdings_count"], errors="coerce"
    )
    prepared["is_amendment_flag"] = (
        prepared["is_amendment"].astype("boolean").fillna(False).astype("int64")
    )

    report_dates = prepared["report_date_utc"]
    prepared["is_standard_quarter_end"] = (
        report_dates.dt.is_quarter_end & report_dates.dt.month.isin((3, 6, 9, 12))
    )
    prepared["filing_lag_days"] = (
        prepared["filed_at_utc"] - prepared["report_date_utc"]
    ).dt.total_seconds() / 86_400
    prepared["availability_delay_minutes"] = (
        prepared["available_at_utc"] - prepared["filed_at_utc"]
    ).dt.total_seconds() / 60
    prepared["report_not_after_filing"] = (
        prepared["report_date_utc"].notna()
        & prepared["filed_at_utc"].notna()
        & (prepared["report_date_utc"] <= prepared["filed_at_utc"])
    )
    prepared["available_not_before_filing"] = (
        prepared["available_at_utc"].notna()
        & prepared["filed_at_utc"].notna()
        & (prepared["available_at_utc"] >= prepared["filed_at_utc"])
    )
    return prepared


def timing_field_summary(filings: pd.DataFrame) -> pd.DataFrame:
    """Summarize economic, filed, and available timestamps side by side."""

    require_columns(
        filings,
        ("report_date_utc", "filed_at_utc", "available_at_utc"),
        context="prepared 13F timing summary",
    )
    records = []
    for role, column in (
        ("economic portfolio date", "report_date_utc"),
        ("published filing time", "filed_at_utc"),
        ("research availability time", "available_at_utc"),
    ):
        values = filings[column]
        records.append(
            {
                "role": role,
                "column": column,
                "minimum": values.min(),
                "maximum": values.max(),
                "missing_rows": int(values.isna().sum()),
                "distinct_values": int(values.nunique(dropna=True)),
            }
        )
    return pd.DataFrame.from_records(records)


def quarterly_filing_summary(
    filings: pd.DataFrame,
    *,
    standard_quarter_ends_only: bool = True,
) -> pd.DataFrame:
    """Aggregate filings by economic report date without resolving amendments."""

    required = (
        "report_date_utc",
        "accession_no",
        "manager_cik",
        "is_amendment_flag",
        "holdings_count_numeric",
        "filing_lag_days",
        "filed_at_utc",
        "available_at_utc",
        "is_standard_quarter_end",
    )
    require_columns(filings, required, context="prepared 13F quarterly summary")
    work = filings.copy()
    if standard_quarter_ends_only:
        work = work[work["is_standard_quarter_end"]]
    work = work.dropna(subset=["report_date_utc"])

    summary = (
        work.groupby("report_date_utc", dropna=False)
        .agg(
            filings=("accession_no", "nunique"),
            managers=("manager_cik", "nunique"),
            holding_lines_reported=("holdings_count_numeric", "sum"),
            amendments=("is_amendment_flag", "sum"),
            filing_lag_median_days=("filing_lag_days", "median"),
            filing_lag_p90_days=("filing_lag_days", lambda values: values.quantile(0.90)),
            first_filed_at=("filed_at_utc", "min"),
            last_filed_at=("filed_at_utc", "max"),
            last_available_at=("available_at_utc", "max"),
        )
        .reset_index()
        .sort_values("report_date_utc")
        .reset_index(drop=True)
    )
    return summary


def filing_lag_summary(filings: pd.DataFrame) -> pd.DataFrame:
    """Describe report-to-filing lag separately for every filing form."""

    require_columns(
        filings,
        ("form_type", "accession_no", "report_date_utc", "filing_lag_days"),
        context="prepared 13F filing lag summary",
    )
    return (
        filings.groupby("form_type", dropna=False)
        .agg(
            filings=("accession_no", "nunique"),
            report_dates=("report_date_utc", "nunique"),
            lag_p10_days=("filing_lag_days", lambda values: values.quantile(0.10)),
            lag_median_days=("filing_lag_days", "median"),
            lag_p90_days=("filing_lag_days", lambda values: values.quantile(0.90)),
            lag_min_days=("filing_lag_days", "min"),
            lag_max_days=("filing_lag_days", "max"),
        )
        .reset_index()
        .sort_values("form_type")
        .reset_index(drop=True)
    )


def manager_period_multiplicity(filings: pd.DataFrame) -> pd.DataFrame:
    """Count how often a manager-report date is represented by multiple filings."""

    require_columns(
        filings,
        ("manager_cik", "report_date_utc", "accession_no"),
        context="prepared 13F manager-period multiplicity",
    )
    usable = filings.dropna(subset=["manager_cik", "report_date_utc", "accession_no"])
    counts = usable.groupby(["manager_cik", "report_date_utc"])["accession_no"].nunique()
    distribution = counts.value_counts().sort_index()
    return pd.DataFrame(
        {
            "filings_per_manager_period": distribution.index.astype(int),
            "manager_periods": distribution.to_numpy(dtype=int),
        }
    )


def bounded_parquet_sample(
    table_directory: str | Path,
    columns: Iterable[str],
    *,
    max_files: int = 5,
    rows_per_file: int = 20_000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read a deterministic first batch from evenly spaced partitions."""

    if max_files < 1:
        raise ValueError("max_files must be at least 1")
    if rows_per_file < 1:
        raise ValueError("rows_per_file must be at least 1")
    files = _parquet_files(table_directory)
    requested = list(dict.fromkeys(columns))
    if not requested:
        raise ValueError("At least one column must be requested")

    sample_count = min(max_files, len(files))
    if sample_count == 1:
        indices = [0]
    else:
        indices = sorted(
            {
                round(position * (len(files) - 1) / (sample_count - 1))
                for position in range(sample_count)
            }
        )

    frames: list[pd.DataFrame] = []
    for index in indices:
        parquet_file = pq.ParquetFile(files[index])
        missing = sorted(set(requested).difference(parquet_file.schema_arrow.names))
        if missing:
            raise KeyError(
                f"A sampled Parquet partition is missing columns: {', '.join(missing)}"
            )
        batch = next(
            parquet_file.iter_batches(
                columns=requested,
                batch_size=rows_per_file,
            ),
            None,
        )
        if batch is not None:
            frames.append(batch.to_pandas())

    sample = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=requested)
    design = pd.DataFrame(
        [
            {
                "available_partitions": len(files),
                "sampled_partitions": len(indices),
                "maximum_rows_per_partition": rows_per_file,
                "sample_rows": len(sample),
                "sampling_rule": "first batch from evenly spaced filing-month partitions",
            }
        ]
    )
    return sample, design
