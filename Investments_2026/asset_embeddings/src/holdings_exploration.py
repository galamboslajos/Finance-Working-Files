"""Small, explicit helpers for exploring holdings data.

The functions in this module are designed for bounded samples. They do not
encode production cleaning rules and never silently impute missing values.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Iterable, Mapping, Sequence

import pandas as pd


@dataclass(frozen=True)
class HoldingsSpec:
    """Column contract for one holdings product."""

    name: str
    investor_column: str
    report_date_column: str
    filed_at_column: str
    availability_column: str
    filing_column: str
    holding_id_column: str
    holding_ordinal_column: str
    value_column: str
    asset_identifier_columns: tuple[str, ...]
    supplemental_identifier_columns: tuple[str, ...] = ()
    category_column: str | None = None
    weight_column: str | None = None


DATASET_SPECS: Mapping[str, HoldingsSpec] = {
    "13f": HoldingsSpec(
        name="13F",
        investor_column="manager_cik",
        report_date_column="period_of_report",
        filed_at_column="filed_at",
        availability_column="availability_timestamp_utc",
        filing_column="accession_no",
        holding_id_column="holding_id",
        holding_ordinal_column="holding_ordinal",
        value_column="value_usd",
        asset_identifier_columns=("resolved_issuer_cik", "cusip", "ticker"),
        supplemental_identifier_columns=("reported_issuer_cik", "issuer_provider_entity_id"),
    ),
    "nport": HoldingsSpec(
        name="N-PORT",
        investor_column="series_id",
        report_date_column="period_of_report_end",
        filed_at_column="filed_at",
        availability_column="availability_timestamp_utc",
        filing_column="accession_no",
        holding_id_column="holding_id",
        holding_ordinal_column="holding_ordinal",
        value_column="value_usd",
        asset_identifier_columns=("cusip", "isin", "ticker"),
        supplemental_identifier_columns=("registrant_cik", "issuer_lei", "other_identifier"),
        category_column="asset_category",
        weight_column="pct_value",
    ),
}


def require_columns(frame: pd.DataFrame, columns: Iterable[str], *, context: str) -> None:
    """Raise a readable error when required columns are absent."""

    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{context} is missing required columns: {', '.join(missing)}")


def read_parquet_sample(path: str | Path) -> pd.DataFrame:
    """Read a local Parquet sample and reject empty inputs."""

    sample_path = Path(path).expanduser().resolve()
    if not sample_path.is_file():
        raise FileNotFoundError(f"Sample does not exist: {sample_path}")
    frame = pd.read_parquet(sample_path)
    if frame.empty:
        raise ValueError(f"Sample contains no rows: {sample_path}")
    return frame


def fetch_private_object(
    uri: str,
    destination: str | Path,
    *,
    gcloud_executable: str | Path,
) -> Path:
    """Download one explicitly supplied private object without shell expansion."""

    if not uri.startswith("gs://"):
        raise ValueError("The private object URI must start with gs://")
    destination_path = Path(destination).expanduser().resolve()
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [str(gcloud_executable), "storage", "cp", uri, str(destination_path)],
        check=True,
    )
    return destination_path


def schema_profile(frame: pd.DataFrame) -> pd.DataFrame:
    """Return column types, completeness, and sample cardinality."""

    rows = len(frame)
    records = []
    for column in frame.columns:
        series = frame[column]
        non_null = int(series.notna().sum())
        records.append(
            {
                "column": column,
                "dtype": str(series.dtype),
                "non_null_rows": non_null,
                "null_rows": rows - non_null,
                "null_pct": round(100 * (rows - non_null) / rows, 3),
                "sample_distinct": int(series.nunique(dropna=True)),
            }
        )
    return pd.DataFrame.from_records(records)


def timing_profile(frame: pd.DataFrame, spec: HoldingsSpec) -> pd.DataFrame:
    """Summarize economic, filing, and availability dates without conflating them."""

    columns = [
        spec.report_date_column,
        spec.filed_at_column,
        spec.availability_column,
    ]
    require_columns(frame, columns, context=f"{spec.name} timing profile")
    parsed = {column: pd.to_datetime(frame[column], errors="coerce", utc=True) for column in columns}
    records = []
    for role, column in zip(("economic", "filed", "available"), columns):
        values = parsed[column]
        records.append(
            {
                "role": role,
                "column": column,
                "minimum": values.min(),
                "maximum": values.max(),
                "missing_rows": int(values.isna().sum()),
                "distinct_values": int(values.nunique(dropna=True)),
                "violation_rows": pd.NA,
            }
        )

    economic = parsed[spec.report_date_column]
    filed = parsed[spec.filed_at_column]
    available = parsed[spec.availability_column]
    records.append(
        {
            "role": "PIT check",
            "column": f"{spec.availability_column} >= {spec.filed_at_column}",
            "minimum": pd.NaT,
            "maximum": pd.NaT,
            "missing_rows": int((available.isna() | filed.isna()).sum()),
            "distinct_values": pd.NA,
            "violation_rows": int((available < filed).fillna(False).sum()),
        }
    )
    records.append(
        {
            "role": "chronology check",
            "column": f"{spec.report_date_column} <= {spec.filed_at_column}",
            "minimum": pd.NaT,
            "maximum": pd.NaT,
            "missing_rows": int((economic.isna() | filed.isna()).sum()),
            "distinct_values": pd.NA,
            "violation_rows": int((economic > filed).fillna(False).sum()),
        }
    )
    return pd.DataFrame.from_records(records)


def identifier_profile(frame: pd.DataFrame, spec: HoldingsSpec) -> pd.DataFrame:
    """Measure coverage of investor, filing, holding, and asset identifiers."""

    candidates = [
        spec.investor_column,
        spec.filing_column,
        spec.holding_id_column,
        *spec.asset_identifier_columns,
        *spec.supplemental_identifier_columns,
    ]
    available = [column for column in candidates if column in frame.columns]
    rows = len(frame)
    records = []
    for column in available:
        values = frame[column]
        populated = values.notna() & values.astype("string").str.strip().ne("")
        records.append(
            {
                "column": column,
                "populated_rows": int(populated.sum()),
                "coverage_pct": round(100 * populated.mean(), 3),
                "sample_distinct": int(values[populated].nunique()),
                "rows": rows,
            }
        )
    return pd.DataFrame.from_records(records)


def duplicate_profile(frame: pd.DataFrame, spec: HoldingsSpec) -> pd.DataFrame:
    """Count exact IDs and filing-ordinal natural-key duplicates."""

    checks = {
        "holding_id": [spec.holding_id_column],
        "filing_and_ordinal": [spec.filing_column, spec.holding_ordinal_column],
    }
    records = []
    for name, columns in checks.items():
        require_columns(frame, columns, context=f"{spec.name} duplicate check")
        usable = frame.dropna(subset=columns)
        duplicate_rows = int(usable.duplicated(columns, keep=False).sum())
        records.append(
            {
                "key": name,
                "columns": " + ".join(columns),
                "usable_rows": len(usable),
                "duplicate_rows": duplicate_rows,
                "duplicate_pct": round(100 * duplicate_rows / len(usable), 3)
                if len(usable)
                else 0.0,
            }
        )
    return pd.DataFrame.from_records(records)


def filing_conflict_profile(frame: pd.DataFrame, spec: HoldingsSpec) -> pd.DataFrame:
    """Find investor-report groups represented by more than one filing."""

    columns = [spec.report_date_column, spec.investor_column, spec.filing_column]
    require_columns(frame, columns, context=f"{spec.name} filing conflict profile")
    work = frame.dropna(subset=columns).copy()
    for column in columns:
        work = work[work[column].astype("string").str.strip().ne("")]
    counts = (
        work.groupby([spec.report_date_column, spec.investor_column], dropna=False)[spec.filing_column]
        .nunique()
        .rename("filings")
        .reset_index()
    )
    conflicts = counts[counts["filings"] > 1].copy()
    return conflicts.sort_values("filings", ascending=False).reset_index(drop=True)


def value_profile(frame: pd.DataFrame, spec: HoldingsSpec) -> pd.DataFrame:
    """Describe signed holding values, including zero and negative states."""

    require_columns(frame, [spec.value_column], context=f"{spec.name} value profile")
    values = pd.to_numeric(frame[spec.value_column], errors="coerce")
    valid = values.dropna()
    quantiles = valid.quantile([0.0, 0.01, 0.5, 0.99, 1.0]) if len(valid) else pd.Series(dtype=float)
    return pd.DataFrame(
        [
            {
                "rows": len(values),
                "missing": int(values.isna().sum()),
                "negative": int((values < 0).sum()),
                "zero": int((values == 0).sum()),
                "positive": int((values > 0).sum()),
                "minimum": quantiles.get(0.0),
                "p01": quantiles.get(0.01),
                "median": quantiles.get(0.5),
                "p99": quantiles.get(0.99),
                "maximum": quantiles.get(1.0),
            }
        ]
    )


def portfolio_profile(frame: pd.DataFrame, spec: HoldingsSpec) -> pd.DataFrame:
    """Describe filing-level portfolios without aggregating amendments together."""

    keys = [spec.report_date_column, spec.investor_column, spec.filing_column]
    require_columns(frame, [*keys, spec.value_column], context=f"{spec.name} portfolio profile")
    work = frame.dropna(subset=keys).copy()
    for key in keys:
        work = work[work[key].astype("string").str.strip().ne("")]
    work[spec.value_column] = pd.to_numeric(work[spec.value_column], errors="coerce")
    grouped = work.groupby(keys, dropna=False, sort=False)
    portfolio = grouped.agg(
        holdings=(spec.holding_id_column, "size"),
        reported_value=(spec.value_column, "sum"),
        negative_value_rows=(spec.value_column, lambda values: int((values < 0).sum())),
    )
    if portfolio.empty:
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "portfolios": len(portfolio),
                "holdings_p10": portfolio["holdings"].quantile(0.10),
                "holdings_median": portfolio["holdings"].median(),
                "holdings_p90": portfolio["holdings"].quantile(0.90),
                "reported_value_median": portfolio["reported_value"].median(),
                "portfolios_with_negative_rows": int((portfolio["negative_value_rows"] > 0).sum()),
            }
        ]
    )


def category_profile(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    top_n: int = 12,
) -> pd.DataFrame:
    """Return top values for explicitly selected categorical columns."""

    records = []
    for column in columns:
        if column not in frame.columns:
            continue
        counts = frame[column].astype("string").fillna("<MISSING>").value_counts(dropna=False).head(top_n)
        for value, count in counts.items():
            records.append(
                {
                    "column": column,
                    "value": value,
                    "rows": int(count),
                    "row_pct": round(100 * count / len(frame), 3),
                }
            )
    return pd.DataFrame.from_records(records)


def iterative_bipartite_filter(
    pairs: pd.DataFrame,
    *,
    investor_column: str,
    asset_column: str,
    minimum_assets_per_investor: int,
    minimum_investors_per_asset: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Iteratively enforce both sides of a holdings-network coverage rule."""

    require_columns(
        pairs,
        [investor_column, asset_column],
        context="iterative holdings filter",
    )
    current = pairs.dropna(subset=[investor_column, asset_column]).copy()
    log = []
    iteration = 0
    while True:
        iteration += 1
        before = len(current)
        investor_counts = current.groupby(investor_column)[asset_column].nunique()
        keep_investors = investor_counts[investor_counts >= minimum_assets_per_investor].index
        current = current[current[investor_column].isin(keep_investors)]

        asset_counts = current.groupby(asset_column)[investor_column].nunique()
        keep_assets = asset_counts[asset_counts >= minimum_investors_per_asset].index
        current = current[current[asset_column].isin(keep_assets)]
        current = current.reset_index(drop=True)
        after = len(current)
        log.append(
            {
                "iteration": iteration,
                "pairs_before": before,
                "pairs_after": after,
                "investors_after": current[investor_column].nunique(),
                "assets_after": current[asset_column].nunique(),
            }
        )
        if after == before:
            break
    return current, pd.DataFrame.from_records(log)


def matrix_readiness(
    frame: pd.DataFrame,
    spec: HoldingsSpec,
    *,
    asset_column: str,
    positive_values_only: bool,
    category_filter: tuple[str, Sequence[str]] | None = None,
    minimum_assets_per_investor: int = 20,
    minimum_investors_per_asset: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build an explicitly filtered sample holdings matrix and report its density."""

    required = [
        spec.report_date_column,
        spec.investor_column,
        spec.filing_column,
        asset_column,
        spec.value_column,
    ]
    if category_filter:
        required.append(category_filter[0])
    require_columns(frame, required, context=f"{spec.name} matrix readiness")

    work = frame[required].copy()
    audit = [{"stage": "input", "rows": len(work)}]
    identity_columns = [
        spec.report_date_column,
        spec.investor_column,
        spec.filing_column,
        asset_column,
    ]
    work = work.dropna(subset=identity_columns)
    for column in identity_columns:
        work = work[work[column].astype("string").str.strip().ne("")]
    audit.append({"stage": "valid report/investor/asset", "rows": len(work)})

    filing_counts = work.groupby(
        [spec.report_date_column, spec.investor_column],
        dropna=False,
    )[spec.filing_column].nunique()
    conflict_count = int((filing_counts > 1).sum())
    if conflict_count:
        raise ValueError(
            f"{spec.name} has {conflict_count} investor-report groups with multiple filings. "
            "Resolve amendments or versions explicitly before forming a matrix."
        )
    audit.append({"stage": "one filing per investor-report", "rows": len(work)})

    work[spec.value_column] = pd.to_numeric(work[spec.value_column], errors="coerce")
    if positive_values_only:
        work = work[work[spec.value_column] > 0]
        audit.append({"stage": "positive holding value", "rows": len(work)})
    if category_filter:
        category_column, allowed_values = category_filter
        work = work[work[category_column].isin(list(allowed_values))]
        audit.append(
            {
                "stage": f"{category_column} in {tuple(allowed_values)}",
                "rows": len(work),
            }
        )

    pair_columns = [spec.report_date_column, spec.investor_column, asset_column]
    pairs = (
        work.groupby(pair_columns, as_index=False, dropna=False)[spec.value_column]
        .sum(min_count=1)
        .rename(columns={spec.value_column: "holding_value"})
    )
    audit.append({"stage": "aggregated investor-asset pairs", "rows": len(pairs)})

    summaries = []
    filter_logs = []
    filtered_parts = []
    for report_date, period_pairs in pairs.groupby(spec.report_date_column, dropna=False):
        filtered, log = iterative_bipartite_filter(
            period_pairs,
            investor_column=spec.investor_column,
            asset_column=asset_column,
            minimum_assets_per_investor=minimum_assets_per_investor,
            minimum_investors_per_asset=minimum_investors_per_asset,
        )
        filtered_parts.append(filtered)
        log.insert(0, spec.report_date_column, report_date)
        filter_logs.append(log)

        investors = int(period_pairs[spec.investor_column].nunique())
        assets = int(period_pairs[asset_column].nunique())
        density = len(period_pairs) / (investors * assets) if investors and assets else 0.0
        summaries.append(
            {
                spec.report_date_column: report_date,
                "investors": investors,
                "assets": assets,
                "pairs": len(period_pairs),
                "density_pct": round(100 * density, 6),
                "median_assets_per_investor": period_pairs.groupby(spec.investor_column)[asset_column]
                .nunique()
                .median(),
                "median_investors_per_asset": period_pairs.groupby(asset_column)[spec.investor_column]
                .nunique()
                .median(),
                "filtered_investors": int(filtered[spec.investor_column].nunique()),
                "filtered_assets": int(filtered[asset_column].nunique()),
                "filtered_pairs": len(filtered),
            }
        )

    filtered_pairs = pd.concat(filtered_parts, ignore_index=True) if filtered_parts else pairs.iloc[0:0]
    filter_log = pd.concat(filter_logs, ignore_index=True) if filter_logs else pd.DataFrame()
    audit_frame = pd.DataFrame.from_records(audit)
    audit_frame["rows_removed_from_prior_stage"] = audit_frame["rows"].shift(1) - audit_frame["rows"]
    return pd.DataFrame.from_records(summaries), audit_frame, filter_log
