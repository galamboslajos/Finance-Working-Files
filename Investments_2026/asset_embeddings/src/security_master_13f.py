"""Point-in-time 13F security identity and matrix-candidate helpers.

The source-reported CUSIP is available with the filing.  It can therefore be
used as a security-level identifier without applying a later issuer mapping.
This module preserves every instrument channel, treats the cash-share channel
as a representation baseline rather than verified common equity, and reports
every exclusion before a manager-security matrix is formed.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re

import pandas as pd
import pyarrow.parquet as pq

from src.data_spine_13f import (
    classify_instrument_rows,
    mapping_snapshot_audit,
    parquet_input_fingerprint,
)
from src.holdings_exploration import iterative_bipartite_filter, require_columns


SECURITY_MASTER_CONTRACT_VERSION = "13f-security-master-v1"
CASH_SHARE_CHANNEL = "cash_share_candidate"
CUSIP_PATTERN = re.compile(r"^[0-9A-Z*@#]{9}$")


@dataclass(frozen=True)
class FullHistorySecurityAudit:
    """Aggregate, public-safe diagnostics from every holdings partition."""

    coverage: pd.DataFrame
    conflicts: pd.DataFrame
    metadata: dict[str, object]


@dataclass(frozen=True)
class MatrixCandidate:
    """Local pair table and aggregate diagnostics for one or more periods."""

    positions: pd.DataFrame
    filtered_positions: pd.DataFrame
    audit: pd.DataFrame
    iterative_log: pd.DataFrame
    summary: pd.DataFrame
    checks: pd.DataFrame


def _clean_string(values: pd.Series) -> pd.Series:
    cleaned = values.astype("string").str.strip()
    return cleaned.mask(cleaned.eq(""))


def normalize_cusip(values: pd.Series) -> pd.Series:
    """Normalize only case and filing-format separators in reported CUSIPs."""

    cleaned = _clean_string(values).str.upper()
    return cleaned.str.replace(r"[\s-]+", "", regex=True).mask(cleaned.isna())


def classify_security_rows(
    holdings: pd.DataFrame,
    *,
    require_state_eligibility: bool = True,
) -> pd.DataFrame:
    """Attach typed security identity and explicit baseline eligibility.

    `typed_security_id` contains a raw source identifier and must remain in
    ignored local data or memory.  The returned frame is suitable for private
    Data Wrangler inspection, but not for display in a committed notebook.
    """

    required = (
        "cusip",
        "title_of_class",
        "put_call",
        "shares_or_principal_type",
        "value_usd",
    )
    require_columns(holdings, required, context="13F security classification")
    if require_state_eligibility:
        require_columns(
            holdings,
            ("state_model_eligible",),
            context="13F point-in-time security classification",
        )

    classified = classify_instrument_rows(holdings)
    classified["cusip_normalized"] = normalize_cusip(classified["cusip"])
    cusip = classified["cusip_normalized"]
    cusip_status = pd.Series(
        "nonstandard_format",
        index=classified.index,
        dtype="string",
    )
    cusip_status.loc[cusip.isna()] = "missing"
    cusip_status.loc[cusip.str.fullmatch(CUSIP_PATTERN, na=False)] = "usable_reported"
    classified["cusip_status"] = cusip_status

    if "reported_issuer_cik" in classified.columns:
        reported_cik = _clean_string(classified["reported_issuer_cik"])
    else:
        reported_cik = pd.Series(pd.NA, index=classified.index, dtype="string")
    classified["reported_issuer_cik_clean"] = reported_cik
    classified["reported_issuer_cik_status"] = reported_cik.notna().map(
        {True: "reported_with_filing", False: "missing"}
    ).astype("string")

    classified["typed_security_id"] = pd.Series(
        pd.NA,
        index=classified.index,
        dtype="string",
    )
    usable_cusip = cusip_status.eq("usable_reported")
    classified.loc[usable_cusip, "typed_security_id"] = (
        classified.loc[usable_cusip, "instrument_class"]
        + ":"
        + cusip.loc[usable_cusip]
    )

    values = pd.to_numeric(classified["value_usd"], errors="coerce")
    classified["value_usd_numeric"] = values
    value_status = pd.Series("positive", index=classified.index, dtype="string")
    value_status.loc[values.isna()] = "missing"
    value_status.loc[values.eq(0)] = "zero"
    value_status.loc[values.lt(0)] = "negative"
    classified["value_status"] = value_status

    if "state_model_eligible" in classified.columns:
        state_eligible = (
            classified["state_model_eligible"].astype("boolean").fillna(False)
        )
    else:
        state_eligible = pd.Series(True, index=classified.index, dtype="boolean")

    reason = pd.Series(
        "eligible_reported_cash_share_security",
        index=classified.index,
        dtype="string",
    )
    reason.loc[~state_eligible] = "ineligible_manager_period_state"
    reason.loc[state_eligible & cusip_status.eq("missing")] = "missing_reported_cusip"
    reason.loc[
        state_eligible & cusip_status.eq("nonstandard_format")
    ] = "nonstandard_reported_cusip"
    identity_usable = state_eligible & cusip_status.eq("usable_reported")
    reason.loc[
        identity_usable & ~classified["instrument_class"].eq(CASH_SHARE_CHANNEL)
    ] = "separate_instrument_channel"
    baseline_identity = identity_usable & classified["instrument_class"].eq(
        CASH_SHARE_CHANNEL
    )
    reason.loc[baseline_identity & values.isna()] = "missing_reported_value"
    reason.loc[baseline_identity & values.eq(0)] = "zero_reported_value"
    reason.loc[baseline_identity & values.lt(0)] = "negative_reported_value"

    classified["matrix_eligibility_reason"] = reason
    classified["matrix_eligible"] = reason.eq(
        "eligible_reported_cash_share_security"
    )
    return classified


def security_row_audit(classified: pd.DataFrame) -> pd.DataFrame:
    """Aggregate classified rows without returning raw security identifiers."""

    required = (
        "instrument_class",
        "cusip_status",
        "reported_issuer_cik_status",
        "matrix_eligibility_reason",
        "value_usd_numeric",
    )
    require_columns(classified, required, context="classified 13F security rows")
    work = classified.copy()
    work["_rows"] = 1
    work["_positive_value_rows"] = work["value_usd_numeric"].gt(0).astype("int64")
    work["_reported_value_usd"] = work["value_usd_numeric"].fillna(0.0)
    return (
        work.groupby(
            [
                "instrument_class",
                "cusip_status",
                "reported_issuer_cik_status",
                "matrix_eligibility_reason",
            ],
            dropna=False,
        )
        .agg(
            holding_rows=("_rows", "sum"),
            positive_value_rows=("_positive_value_rows", "sum"),
            reported_value_usd=("_reported_value_usd", "sum"),
        )
        .reset_index()
        .sort_values(
            ["matrix_eligibility_reason", "instrument_class", "cusip_status"]
        )
        .reset_index(drop=True)
    )


def security_conflict_audit(classified: pd.DataFrame) -> pd.DataFrame:
    """Count identifier collisions without exposing the identifiers themselves."""

    required = (
        "cusip_normalized",
        "cusip_status",
        "instrument_class",
        "reported_issuer_cik_clean",
        "matrix_eligible",
    )
    require_columns(classified, required, context="classified 13F security rows")
    usable = classified.loc[classified["cusip_status"].eq("usable_reported")].copy()
    if usable.empty:
        return pd.DataFrame(
            {
                "metric": [
                    "usable reported CUSIPs",
                    "CUSIPs spanning multiple instrument channels",
                    "CUSIPs linked to multiple reported issuer CIKs",
                    "eligible cash-share CUSIPs",
                ],
                "count": [0, 0, 0, 0],
            }
        )

    channels = usable.groupby("cusip_normalized")["instrument_class"].nunique()
    cik_pairs = usable.dropna(
        subset=["reported_issuer_cik_clean"]
    ).drop_duplicates(["cusip_normalized", "reported_issuer_cik_clean"])
    ciks = cik_pairs.groupby("cusip_normalized")["reported_issuer_cik_clean"].nunique()
    eligible_assets = usable.loc[usable["matrix_eligible"], "cusip_normalized"].nunique()
    return pd.DataFrame(
        {
            "metric": [
                "usable reported CUSIPs",
                "CUSIPs spanning multiple instrument channels",
                "CUSIPs linked to multiple reported issuer CIKs",
                "eligible cash-share CUSIPs",
            ],
            "count": [
                int(channels.size),
                int(channels.gt(1).sum()),
                int(ciks.gt(1).sum()),
                int(eligible_assets),
            ],
        }
    )


def mapping_cutoff_summary(
    mapping_summary: pd.DataFrame,
    *,
    decision_at: object,
) -> pd.DataFrame:
    """Classify mapping snapshots conservatively relative to a decision time.

    A date-only mapping snapshot is usable only when its date is strictly
    earlier than the decision date.  A same-day snapshot has unknown intraday
    timing and is therefore not silently admitted.
    """

    require_columns(
        mapping_summary,
        ("mapping_snapshot_date", "rows"),
        context="13F mapping snapshot summary",
    )
    decision = pd.Timestamp(decision_at)
    if decision.tzinfo is None:
        raise ValueError("decision_at must be timezone-aware")
    decision_date = decision.tz_convert("UTC").date()

    work = mapping_summary.copy()
    snapshot = pd.to_datetime(work["mapping_snapshot_date"], errors="coerce", utc=True)
    status = pd.Series("future_snapshot", index=work.index, dtype="string")
    status.loc[snapshot.isna()] = "missing_snapshot_date"
    status.loc[snapshot.dt.date < decision_date] = "usable_before_decision"
    status.loc[snapshot.dt.date == decision_date] = "same_day_time_unknown"
    work["mapping_cutoff_status"] = status

    numeric = [
        column
        for column in ("rows", "resolved_cik_rows", "provider_entity_rows")
        if column in work.columns
    ]
    return (
        work.groupby("mapping_cutoff_status", dropna=False)[numeric]
        .sum()
        .reset_index()
        .sort_values("mapping_cutoff_status")
        .reset_index(drop=True)
    )


def _parquet_files(table_directory: str | Path) -> list[Path]:
    directory = Path(table_directory).expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Parquet table directory does not exist: {directory}")
    files = sorted(directory.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet files found below: {directory}")
    return files


def _scan_full_history_security_identity(
    table_directory: str | Path,
    *,
    batch_size: int,
) -> FullHistorySecurityAudit:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    files = _parquet_files(table_directory)
    columns = (
        "period_of_report",
        "cusip",
        "reported_issuer_cik",
        "title_of_class",
        "put_call",
        "shares_or_principal_type",
        "value_usd",
    )
    aggregate_parts: list[pd.DataFrame] = []
    cusip_channels: dict[str, set[str]] = {}
    cusip_reported_ciks: dict[str, set[str]] = {}
    rows_scanned = 0
    batches_scanned = 0

    for path in files:
        parquet_file = pq.ParquetFile(path)
        missing = sorted(set(columns).difference(parquet_file.schema_arrow.names))
        if missing:
            raise KeyError(
                f"A holdings partition is missing security-audit columns: {', '.join(missing)}"
            )
        for batch in parquet_file.iter_batches(columns=list(columns), batch_size=batch_size):
            frame = batch.to_pandas()
            rows_scanned += len(frame)
            batches_scanned += 1
            classified = classify_security_rows(
                frame,
                require_state_eligibility=False,
            )
            report_date = pd.to_datetime(
                classified["period_of_report"], errors="coerce", utc=True
            )
            classified["report_quarter"] = (
                report_date.dt.tz_localize(None).dt.to_period("Q").astype("string")
            )
            classified["_rows"] = 1
            classified["_positive_value_rows"] = (
                classified["value_usd_numeric"].gt(0).astype("int64")
            )
            classified["_reported_value_usd"] = classified[
                "value_usd_numeric"
            ].fillna(0.0)
            aggregate_parts.append(
                classified.groupby(
                    [
                        "report_quarter",
                        "instrument_class",
                        "cusip_status",
                        "reported_issuer_cik_status",
                    ],
                    dropna=False,
                )
                .agg(
                    rows=("_rows", "sum"),
                    positive_value_rows=("_positive_value_rows", "sum"),
                    reported_value_usd=("_reported_value_usd", "sum"),
                )
                .reset_index()
            )

            usable = classified.loc[
                classified["cusip_status"].eq("usable_reported"),
                [
                    "cusip_normalized",
                    "instrument_class",
                    "reported_issuer_cik_clean",
                ],
            ]
            for cusip, channel in usable[
                ["cusip_normalized", "instrument_class"]
            ].drop_duplicates().itertuples(index=False, name=None):
                cusip_channels.setdefault(str(cusip), set()).add(str(channel))
            for cusip, cik in usable.dropna(
                subset=["reported_issuer_cik_clean"]
            )[["cusip_normalized", "reported_issuer_cik_clean"]].drop_duplicates().itertuples(
                index=False,
                name=None,
            ):
                cusip_reported_ciks.setdefault(str(cusip), set()).add(str(cik))

    combined = pd.concat(aggregate_parts, ignore_index=True)
    coverage = (
        combined.groupby(
            [
                "report_quarter",
                "instrument_class",
                "cusip_status",
                "reported_issuer_cik_status",
            ],
            dropna=False,
        )[["rows", "positive_value_rows", "reported_value_usd"]]
        .sum()
        .reset_index()
        .sort_values(
            ["report_quarter", "instrument_class", "cusip_status"]
        )
        .reset_index(drop=True)
    )
    conflicts = pd.DataFrame(
        {
            "metric": [
                "usable reported CUSIPs",
                "CUSIPs spanning multiple instrument channels",
                "CUSIPs linked to multiple reported issuer CIKs across history",
            ],
            "count": [
                len(cusip_channels),
                sum(len(channels) > 1 for channels in cusip_channels.values()),
                sum(len(ciks) > 1 for ciks in cusip_reported_ciks.values()),
            ],
        }
    )
    return FullHistorySecurityAudit(
        coverage=coverage,
        conflicts=conflicts,
        metadata={
            "security_master_contract_version": SECURITY_MASTER_CONTRACT_VERSION,
            "cache_hit": False,
            "input_fingerprint": parquet_input_fingerprint(table_directory),
            "partitions_scanned": len(files),
            "batches_scanned": batches_scanned,
            "rows_scanned": rows_scanned,
            "batch_size": batch_size,
        },
    )


def full_history_security_identity_audit(
    table_directory: str | Path,
    *,
    cache_directory: str | Path | None = None,
    batch_size: int = 250_000,
) -> FullHistorySecurityAudit:
    """Scan all reported security identities, with aggregate cache reuse."""

    fingerprint = parquet_input_fingerprint(table_directory)
    if cache_directory is None:
        return _scan_full_history_security_identity(
            table_directory,
            batch_size=batch_size,
        )

    cache_root = Path(cache_directory).expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_key = f"{SECURITY_MASTER_CONTRACT_VERSION}-{fingerprint}"
    paths = {
        "coverage": cache_root / f"{cache_key}-coverage.parquet",
        "conflicts": cache_root / f"{cache_key}-conflicts.parquet",
        "metadata": cache_root / f"{cache_key}-metadata.json",
    }
    if all(path.is_file() for path in paths.values()):
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        metadata["cache_hit"] = True
        return FullHistorySecurityAudit(
            coverage=pd.read_parquet(paths["coverage"]),
            conflicts=pd.read_parquet(paths["conflicts"]),
            metadata=metadata,
        )

    result = _scan_full_history_security_identity(
        table_directory,
        batch_size=batch_size,
    )
    result.coverage.to_parquet(paths["coverage"], index=False)
    result.conflicts.to_parquet(paths["conflicts"], index=False)
    paths["metadata"].write_text(
        json.dumps(result.metadata, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return result


def cached_mapping_snapshot_audit(
    table_directory: str | Path,
    *,
    cache_directory: str | Path,
    batch_size: int = 250_000,
) -> pd.DataFrame:
    """Cache the aggregate mapping-snapshot audit outside Git."""

    fingerprint = parquet_input_fingerprint(table_directory)
    cache_root = Path(cache_directory).expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    path = cache_root / f"mapping-snapshot-{fingerprint}.parquet"
    if path.is_file():
        return pd.read_parquet(path)
    prior_cache = sorted(
        cache_root.rglob(f"*{fingerprint}*mapping-snapshot.parquet")
    )
    if prior_cache:
        return pd.read_parquet(prior_cache[0])
    result = mapping_snapshot_audit(table_directory, batch_size=batch_size)
    result.to_parquet(path, index=False)
    return result


def build_cash_share_matrix_candidate(
    classified: pd.DataFrame,
    *,
    manager_column: str = "manager_cik",
    report_column: str = "period_of_report",
    accession_column: str = "accession_no",
    maximum_position_weight: float = 0.75,
    minimum_assets_per_manager: int = 20,
    minimum_managers_per_asset: int = 20,
) -> MatrixCandidate:
    """Build an audited cash-share pair table without constructing a dense matrix."""

    if not 0 < maximum_position_weight <= 1:
        raise ValueError("maximum_position_weight must be in (0, 1]")
    if minimum_assets_per_manager < 1 or minimum_managers_per_asset < 1:
        raise ValueError("minimum matrix degrees must be positive")
    required = (
        manager_column,
        report_column,
        "typed_security_id",
        "matrix_eligible",
        "value_usd_numeric",
    )
    require_columns(classified, required, context="13F cash-share matrix candidate")

    work = classified.loc[classified["matrix_eligible"]].copy()
    work[manager_column] = _clean_string(work[manager_column])
    work[report_column] = pd.to_datetime(work[report_column], errors="coerce").dt.normalize()
    work = work.dropna(subset=[manager_column, report_column, "typed_security_id"])

    group_columns = [report_column, manager_column, "typed_security_id"]
    aggregations: dict[str, tuple[str, object]] = {
        "position_value_usd": ("value_usd_numeric", lambda values: values.sum(min_count=1)),
        "holding_lines": ("typed_security_id", "size"),
    }
    if accession_column in work.columns:
        aggregations["source_filings"] = (accession_column, "nunique")
    positions = work.groupby(group_columns, as_index=False).agg(**aggregations)
    positions = positions.loc[positions["position_value_usd"].gt(0)].copy()

    manager_keys = [report_column, manager_column]
    positions["candidate_portfolio_value_usd"] = positions.groupby(manager_keys)[
        "position_value_usd"
    ].transform("sum")
    positions["candidate_position_weight"] = (
        positions["position_value_usd"] / positions["candidate_portfolio_value_usd"]
    )
    max_weights = positions.groupby(manager_keys)["candidate_position_weight"].transform(
        "max"
    )
    concentration_eligible = max_weights.le(maximum_position_weight)
    concentrated = positions.loc[concentration_eligible].copy()

    filtered_parts: list[pd.DataFrame] = []
    logs: list[pd.DataFrame] = []
    for report_date, period in concentrated.groupby(report_column, dropna=False):
        filtered, log = iterative_bipartite_filter(
            period,
            investor_column=manager_column,
            asset_column="typed_security_id",
            minimum_assets_per_investor=minimum_assets_per_manager,
            minimum_investors_per_asset=minimum_managers_per_asset,
        )
        filtered_parts.append(filtered)
        log.insert(0, report_column, report_date)
        logs.append(log)
    filtered_positions = (
        pd.concat(filtered_parts, ignore_index=True)
        if filtered_parts
        else positions.iloc[0:0].copy()
    )
    iterative_log = pd.concat(logs, ignore_index=True) if logs else pd.DataFrame()

    if not filtered_positions.empty:
        filtered_positions["retained_portfolio_value_usd"] = filtered_positions.groupby(
            manager_keys
        )["position_value_usd"].transform("sum")
        filtered_positions["matrix_weight"] = (
            filtered_positions["position_value_usd"]
            / filtered_positions["retained_portfolio_value_usd"]
        )
    else:
        filtered_positions["retained_portfolio_value_usd"] = pd.Series(dtype="float64")
        filtered_positions["matrix_weight"] = pd.Series(dtype="float64")

    def stage_record(stage: str, frame: pd.DataFrame) -> dict[str, object]:
        return {
            "stage": stage,
            "pairs": len(frame),
            "managers": int(frame[manager_column].nunique()) if len(frame) else 0,
            "assets": int(frame["typed_security_id"].nunique()) if len(frame) else 0,
            "reported_value_usd": float(frame["position_value_usd"].sum()) if len(frame) else 0.0,
        }

    audit = pd.DataFrame.from_records(
        [
            stage_record("aggregated eligible cash-share positions", positions),
            stage_record("after 75 percent concentration rule", concentrated),
            stage_record("after iterative manager-asset degree rules", filtered_positions),
        ]
    )

    summaries: list[dict[str, object]] = []
    for report_date, frame in filtered_positions.groupby(report_column, dropna=False):
        managers = int(frame[manager_column].nunique())
        assets = int(frame["typed_security_id"].nunique())
        pairs = len(frame)
        summaries.append(
            {
                report_column: report_date,
                "managers": managers,
                "assets": assets,
                "pairs": pairs,
                "density_pct": 100 * pairs / (managers * assets)
                if managers and assets
                else 0.0,
            }
        )
    summary = pd.DataFrame.from_records(summaries)

    duplicate_pairs = int(
        filtered_positions.duplicated(group_columns).sum()
        if len(filtered_positions)
        else 0
    )
    if len(filtered_positions):
        weight_sums = filtered_positions.groupby(manager_keys)["matrix_weight"].sum()
        maximum_weight_error = float((weight_sums - 1.0).abs().max())
    else:
        maximum_weight_error = 0.0
    checks = pd.DataFrame(
        {
            "check": [
                "manager-security pairs are unique",
                "retained manager weights reconcile to one",
            ],
            "violations": [
                duplicate_pairs,
                int(maximum_weight_error > 1e-10),
            ],
            "maximum_absolute_error": [0.0, maximum_weight_error],
        }
    )
    return MatrixCandidate(
        positions=positions.reset_index(drop=True),
        filtered_positions=filtered_positions.reset_index(drop=True),
        audit=audit,
        iterative_log=iterative_log,
        summary=summary,
        checks=checks,
    )
