"""Point-in-time 13F data-spine audits.

This module turns the project's provisional data-spine decisions into explicit,
testable diagnostics. It does not build a production investor-asset matrix.
All raw identifiers remain in ignored local inputs; returned audit tables are
aggregate summaries unless a function explicitly documents otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Iterable
from zoneinfo import ZoneInfo

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
import pyarrow.parquet as pq

from src.holdings_exploration import require_columns


AUDIT_CONTRACT_VERSION = "13f-data-spine-v1"
AMENDMENT_TYPES = frozenset({"RESTATEMENT", "NEW_HOLDINGS"})

COVER_PAGE_AUDIT_COLUMNS = (
    "accession_no",
    "manager_cik",
    "manager_name",
    "form_type",
    "period_of_report",
    "filed_at",
    "availability_timestamp_utc",
    "is_amendment",
    "amendment_info_json",
    "report_type",
    "crd_number",
    "sec_file_number",
    "form_13f_file_number",
)

HOLDINGS_AUDIT_COLUMNS = (
    "period_of_report",
    "filed_at",
    "availability_timestamp_utc",
    "title_of_class",
    "put_call",
    "shares_or_principal_type",
    "value_usd",
    "reported_issuer_cik",
    "resolved_issuer_cik",
    "issuer_provider_entity_id",
    "issuer_identity_confidence",
)


@dataclass(frozen=True)
class HoldingsAuditResult:
    """Memory-bounded aggregate results from every holdings partition."""

    instrument: pd.DataFrame
    issuer_identity: pd.DataFrame
    timing: pd.DataFrame
    metadata: dict[str, object]


def _nonblank(values: pd.Series) -> pd.Series:
    strings = values.astype("string")
    return strings.notna() & strings.str.strip().ne("")


def _parquet_files(table_directory: str | Path) -> list[Path]:
    directory = Path(table_directory).expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Parquet table directory does not exist: {directory}")
    files = sorted(directory.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet files found below: {directory}")
    return files


def parquet_input_fingerprint(table_directory: str | Path) -> str:
    """Fingerprint local Parquet inputs for aggregate-cache invalidation."""

    directory = Path(table_directory).expanduser().resolve()
    digest = sha256()
    for path in _parquet_files(directory):
        stat = path.stat()
        digest.update(str(path.relative_to(directory)).encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
    return digest.hexdigest()


def _parse_amendment_json(raw: object) -> tuple[object, object, bool]:
    if raw is None or raw is pd.NA or (isinstance(raw, float) and pd.isna(raw)):
        return pd.NA, pd.NA, False
    try:
        parsed = json.loads(str(raw))
    except (TypeError, ValueError, json.JSONDecodeError):
        return pd.NA, pd.NA, True
    if not isinstance(parsed, dict):
        return pd.NA, pd.NA, True
    amendment_number = parsed.get("amendmentNo", pd.NA)
    amendment_type = parsed.get("amendmentType", pd.NA)
    if amendment_type is not pd.NA and amendment_type is not None:
        amendment_type = str(amendment_type).strip().upper().replace(" ", "_")
    return amendment_number, amendment_type, False


def prepare_amendment_events(cover_pages: pd.DataFrame) -> pd.DataFrame:
    """Parse and reconcile every cover-page amendment signal without dropping rows."""

    required = (
        "accession_no",
        "manager_cik",
        "form_type",
        "period_of_report",
        "filed_at",
        "availability_timestamp_utc",
        "is_amendment",
        "amendment_info_json",
        "report_type",
    )
    require_columns(cover_pages, required, context="13F cover-page amendment audit")
    prepared = cover_pages.copy()

    parsed = prepared["amendment_info_json"].map(_parse_amendment_json)
    parsed_frame = pd.DataFrame(
        parsed.tolist(),
        columns=("amendment_number_raw", "amendment_type_raw", "amendment_json_invalid"),
        index=prepared.index,
    )
    prepared = pd.concat([prepared, parsed_frame], axis=1)
    prepared["amendment_number"] = pd.to_numeric(
        prepared["amendment_number_raw"], errors="coerce"
    ).astype("Int64")
    prepared["amendment_type"] = prepared["amendment_type_raw"].where(
        prepared["amendment_type_raw"].isin(AMENDMENT_TYPES),
        "UNKNOWN",
    )

    form = prepared["form_type"].astype("string").str.strip().str.upper()
    prepared["form_marks_amendment"] = form.str.endswith("/A", na=False)
    prepared["flag_marks_amendment"] = (
        prepared["is_amendment"].astype("boolean").fillna(False)
    )
    prepared["json_marks_amendment"] = (
        prepared["amendment_number"].notna()
        | prepared["amendment_type_raw"].notna()
    )
    prepared["canonical_is_amendment"] = (
        prepared[
            ["form_marks_amendment", "flag_marks_amendment", "json_marks_amendment"]
        ]
        .any(axis=1)
        .astype(bool)
    )
    prepared["amendment_signal_conflict"] = (
        prepared[
            ["form_marks_amendment", "flag_marks_amendment", "json_marks_amendment"]
        ]
        .nunique(axis=1)
        .gt(1)
    )
    prepared["amendment_type_missing"] = (
        prepared["canonical_is_amendment"]
        & prepared["amendment_type"].eq("UNKNOWN")
    )
    prepared["event_role"] = "ORIGINAL"
    amendment_rows = prepared["canonical_is_amendment"]
    prepared.loc[amendment_rows, "event_role"] = prepared.loc[
        amendment_rows, "amendment_type"
    ]

    prepared["report_date_utc"] = pd.to_datetime(
        prepared["period_of_report"], errors="coerce", utc=True
    )
    prepared["filed_at_utc"] = pd.to_datetime(
        prepared["filed_at"], errors="coerce", utc=True
    )
    prepared["available_at_utc"] = pd.to_datetime(
        prepared["availability_timestamp_utc"], errors="coerce", utc=True
    )
    return prepared


def amendment_quality_summary(events: pd.DataFrame) -> pd.DataFrame:
    """Return aggregate amendment diagnostics without raw manager identifiers."""

    required = (
        "canonical_is_amendment",
        "amendment_signal_conflict",
        "amendment_type_missing",
        "amendment_json_invalid",
        "available_at_utc",
        "report_date_utc",
        "accession_no",
    )
    require_columns(events, required, context="prepared amendment events")
    amendment_rows = events["canonical_is_amendment"]
    records = [
        ("all cover-page rows", len(events)),
        ("canonical amendment rows", int(amendment_rows.sum())),
        ("signal-conflict rows", int(events["amendment_signal_conflict"].sum())),
        ("amendments missing a known type", int(events["amendment_type_missing"].sum())),
        ("invalid amendment JSON rows", int(events["amendment_json_invalid"].sum())),
        (
            "amendments missing availability",
            int((amendment_rows & events["available_at_utc"].isna()).sum()),
        ),
        (
            "rows missing report date",
            int(events["report_date_utc"].isna().sum()),
        ),
        (
            "duplicate accession rows",
            int(events.duplicated("accession_no", keep=False).sum()),
        ),
    ]
    return pd.DataFrame(records, columns=("metric", "rows"))


def amendment_type_summary(events: pd.DataFrame) -> pd.DataFrame:
    """Count form, report, and amendment roles across the complete event table."""

    require_columns(
        events,
        ("form_type", "report_type", "event_role", "accession_no"),
        context="prepared amendment type summary",
    )
    return (
        events.groupby(["form_type", "report_type", "event_role"], dropna=False)
        .agg(filings=("accession_no", "nunique"))
        .reset_index()
        .sort_values(["form_type", "report_type", "event_role"])
        .reset_index(drop=True)
    )


def amendment_sequence_summary(events: pd.DataFrame) -> pd.DataFrame:
    """Summarize manager-quarter event sequences without exposing managers."""

    required = (
        "manager_cik",
        "report_date_utc",
        "available_at_utc",
        "accession_no",
        "event_role",
        "amendment_number",
    )
    require_columns(events, required, context="prepared amendment sequences")
    usable = events.dropna(
        subset=["manager_cik", "report_date_utc", "accession_no", "available_at_utc"]
    ).copy()
    usable = usable.sort_values(
        ["manager_cik", "report_date_utc", "available_at_utc", "accession_no"]
    )

    records: list[dict[str, object]] = []
    for _, group in usable.groupby(["manager_cik", "report_date_utc"], sort=False):
        if len(group) < 2:
            continue
        amendment_numbers = group.loc[
            group["event_role"].ne("ORIGINAL"), "amendment_number"
        ].dropna()
        records.append(
            {
                "sequence": " > ".join(group["event_role"].astype(str)),
                "filings": int(group["accession_no"].nunique()),
                "availability_tie": bool(group["available_at_utc"].duplicated().any()),
                "amendment_number_decrease": bool(
                    len(amendment_numbers) > 1
                    and amendment_numbers.astype(int).diff().dropna().lt(0).any()
                ),
                "starts_without_original": bool(group.iloc[0]["event_role"] != "ORIGINAL"),
            }
        )
    if not records:
        return pd.DataFrame(
            columns=(
                "sequence",
                "manager_periods",
                "filings",
                "availability_tie_groups",
                "amendment_number_decrease_groups",
                "starts_without_original_groups",
            )
        )

    sequences = pd.DataFrame.from_records(records)
    return (
        sequences.groupby("sequence", dropna=False)
        .agg(
            manager_periods=("sequence", "size"),
            filings=("filings", "sum"),
            availability_tie_groups=("availability_tie", "sum"),
            amendment_number_decrease_groups=("amendment_number_decrease", "sum"),
            starts_without_original_groups=("starts_without_original", "sum"),
        )
        .reset_index()
        .sort_values(["manager_periods", "sequence"], ascending=[False, True])
        .reset_index(drop=True)
    )


def filing_cover_join_audit(
    filings: pd.DataFrame,
    cover_pages: pd.DataFrame,
) -> pd.DataFrame:
    """Audit accession coverage before joining cover-page decisions to filings."""

    require_columns(filings, ("accession_no",), context="filings join audit")
    require_columns(cover_pages, ("accession_no",), context="cover-pages join audit")
    filing_keys = filings["accession_no"].dropna().astype("string")
    cover_keys = cover_pages["accession_no"].dropna().astype("string")
    filing_set = set(filing_keys)
    cover_set = set(cover_keys)
    records = [
        ("filing rows", len(filings)),
        ("cover-page rows", len(cover_pages)),
        ("distinct filing accessions", len(filing_set)),
        ("distinct cover-page accessions", len(cover_set)),
        ("filing accessions without cover page", len(filing_set - cover_set)),
        ("cover-page accessions without filing", len(cover_set - filing_set)),
        ("duplicate filing accession rows", int(filing_keys.duplicated(keep=False).sum())),
        ("duplicate cover-page accession rows", int(cover_keys.duplicated(keep=False).sum())),
    ]
    return pd.DataFrame(records, columns=("metric", "rows"))


_BUSINESS_DAY = CustomBusinessDay(calendar=USFederalHolidayCalendar())


def standard_13f_deadline(report_date: object) -> pd.Timestamp:
    """Return the normal SEC due date: 45 days, rolled to a US business day."""

    quarter_end = pd.Timestamp(report_date)
    if quarter_end.tzinfo is not None:
        quarter_end = quarter_end.tz_localize(None)
    quarter_end = quarter_end.normalize()
    due = quarter_end + pd.Timedelta(days=45)
    if len(pd.date_range(due, due, freq=_BUSINESS_DAY)) == 0:
        due = due + _BUSINESS_DAY
    return pd.Timestamp(due).normalize()


def _deadline_cutoff_utc(report_date: object) -> pd.Timestamp:
    deadline = standard_13f_deadline(report_date)
    local = deadline.tz_localize(ZoneInfo("America/New_York")) + pd.Timedelta(
        hours=23, minutes=59, seconds=59
    )
    return local.tz_convert("UTC")


def quarter_coverage_audit(
    filings: pd.DataFrame,
    *,
    observation_start: object | None = None,
    data_freeze: object | None = None,
) -> pd.DataFrame:
    """Identify quarters whose complete normal filing window was observable."""

    required = (
        "accession_no",
        "manager_cik",
        "period_of_report",
        "availability_timestamp_utc",
        "is_amendment",
    )
    require_columns(filings, required, context="13F quarter coverage audit")
    work = filings.copy()
    work["report_date_utc"] = pd.to_datetime(
        work["period_of_report"], errors="coerce", utc=True
    )
    work["available_at_utc"] = pd.to_datetime(
        work["availability_timestamp_utc"], errors="coerce", utc=True
    )
    work["is_standard_quarter_end"] = (
        work["report_date_utc"].dt.is_quarter_end
        & work["report_date_utc"].dt.month.isin((3, 6, 9, 12))
    )
    work = work[work["is_standard_quarter_end"] & work["report_date_utc"].notna()]

    observed_from = pd.to_datetime(
        observation_start if observation_start is not None else work["available_at_utc"].min(),
        errors="coerce",
        utc=True,
    )
    frozen_at = pd.to_datetime(
        data_freeze if data_freeze is not None else work["available_at_utc"].max(),
        errors="coerce",
        utc=True,
    )
    if pd.isna(observed_from) or pd.isna(frozen_at):
        raise ValueError("Coverage audit requires non-missing observation and freeze timestamps")

    summary = (
        work.groupby("report_date_utc", dropna=False)
        .agg(
            filings=("accession_no", "nunique"),
            managers=("manager_cik", "nunique"),
            amendments=("is_amendment", lambda values: int(values.fillna(False).sum())),
            first_available_at=("available_at_utc", "min"),
            last_available_at=("available_at_utc", "max"),
        )
        .reset_index()
        .sort_values("report_date_utc")
        .reset_index(drop=True)
    )
    summary["normal_deadline"] = summary["report_date_utc"].map(
        lambda value: standard_13f_deadline(value.tz_localize(None))
    )
    summary["information_cutoff_utc"] = summary["report_date_utc"].map(
        lambda value: _deadline_cutoff_utc(value.tz_localize(None))
    )
    summary["observation_started_by_quarter_end"] = (
        observed_from <= summary["report_date_utc"]
    )
    summary["normal_deadline_observed"] = (
        summary["information_cutoff_utc"] <= frozen_at
    )
    summary["eligible_complete_window"] = (
        summary["observation_started_by_quarter_end"]
        & summary["normal_deadline_observed"]
    )
    summary["observation_start_utc"] = observed_from
    summary["data_freeze_utc"] = frozen_at
    return summary


def manager_identity_audit(cover_pages: pd.DataFrame) -> pd.DataFrame:
    """Aggregate manager identifier stability without printing names or CIKs."""

    columns = (
        "manager_cik",
        "manager_name",
        "crd_number",
        "sec_file_number",
        "form_13f_file_number",
    )
    require_columns(cover_pages, columns, context="13F manager identity audit")
    work = cover_pages.loc[:, columns].copy()
    for column in columns:
        work[column] = work[column].astype("string").str.strip()
        work.loc[work[column].eq(""), column] = pd.NA

    normalized_name = (
        work["manager_name"]
        .str.upper()
        .str.replace(r"[^A-Z0-9]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    named = work.assign(normalized_manager_name=normalized_name).dropna(
        subset=["normalized_manager_name", "manager_cik"]
    )
    names_per_cik = named.groupby("manager_cik")["normalized_manager_name"].nunique()
    ciks_per_name = named.groupby("normalized_manager_name")["manager_cik"].nunique()

    records: list[tuple[str, int]] = [
        ("cover-page rows", len(work)),
        ("distinct manager CIKs", int(work["manager_cik"].nunique(dropna=True))),
        ("rows missing manager CIK", int(work["manager_cik"].isna().sum())),
        ("CIKs with multiple normalized names", int(names_per_cik.gt(1).sum())),
        ("normalized names linked to multiple CIKs", int(ciks_per_name.gt(1).sum())),
    ]
    for column in ("crd_number", "sec_file_number", "form_13f_file_number"):
        usable = work.dropna(subset=["manager_cik"])
        values_per_cik = usable.groupby("manager_cik")[column].nunique(dropna=True)
        records.extend(
            [
                (f"rows missing {column}", int(work[column].isna().sum())),
                (f"CIKs with multiple {column} values", int(values_per_cik.gt(1).sum())),
            ]
        )
    return pd.DataFrame(records, columns=("metric", "rows"))


def classify_instrument_rows(holdings: pd.DataFrame) -> pd.DataFrame:
    """Assign a conservative, provisional instrument class to holding rows."""

    required = ("title_of_class", "put_call", "shares_or_principal_type")
    require_columns(holdings, required, context="13F instrument classification")
    classified = holdings.copy()
    title = (
        classified["title_of_class"]
        .astype("string")
        .str.upper()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    put_call = classified["put_call"].astype("string").str.upper().str.strip()
    put_call = put_call.mask(put_call.eq(""))
    amount_type = (
        classified["shares_or_principal_type"].astype("string").str.upper().str.strip()
    )
    amount_type = amount_type.mask(amount_type.eq(""))

    instrument_class = pd.Series(
        "unknown",
        index=classified.index,
        dtype="string",
    )
    reason = pd.Series(
        "insufficient explicit instrument evidence",
        index=classified.index,
        dtype="string",
    )

    rules: list[tuple[pd.Series, str, str]] = [
        (put_call.eq("PUT"), "put_option", "put_call explicitly reports PUT"),
        (put_call.eq("CALL"), "call_option", "put_call explicitly reports CALL"),
        (
            amount_type.eq("PRN")
            | title.str.contains(
                r"\b(?:CONV|CONVERT|DEB|DEBT|NOTE|BOND)\b", na=False
            ),
            "convertible_or_debt",
            "principal amount or explicit debt/convertible class",
        ),
        (
            title.str.contains(r"\b(?:PFD|PREF|PREFERRED)\b", na=False),
            "preferred_equity",
            "explicit preferred class",
        ),
        (
            title.str.contains(r"\b(?:WARRANT|WT|RIGHT|RT)\b", na=False),
            "warrant_or_right",
            "explicit warrant or right class",
        ),
        (
            title.str.contains(r"\b(?:UNIT|UNT)\b", na=False),
            "unit",
            "explicit unit class",
        ),
        (
            title.str.contains(r"\b(?:ETF|ETN|FUND)\b", na=False),
            "fund_or_etf_explicit",
            "explicit fund, ETF, or ETN class",
        ),
        (
            amount_type.eq("SH") & put_call.isna(),
            "cash_share_candidate",
            "share amount with no reported put/call flag",
        ),
    ]
    assigned = pd.Series(False, index=classified.index)
    for mask, label, explanation in rules:
        use = mask.fillna(False) & ~assigned
        instrument_class.loc[use] = label
        reason.loc[use] = explanation
        assigned |= use

    classified["instrument_class"] = instrument_class
    classified["classification_reason"] = reason
    return classified


def _sum_audit_parts(parts: list[pd.DataFrame], keys: Iterable[str]) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame()
    combined = pd.concat(parts, ignore_index=True)
    value_columns = [column for column in combined.columns if column not in keys]
    return (
        combined.groupby(list(keys), dropna=False)[value_columns]
        .sum()
        .reset_index()
        .sort_values(list(keys))
        .reset_index(drop=True)
    )


def _scan_holdings(table_directory: str | Path, *, batch_size: int) -> HoldingsAuditResult:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    files = _parquet_files(table_directory)
    instrument_parts: list[pd.DataFrame] = []
    identity_parts: list[pd.DataFrame] = []
    timing_parts: list[pd.DataFrame] = []
    rows_scanned = 0
    batches_scanned = 0

    for path in files:
        parquet_file = pq.ParquetFile(path)
        missing = sorted(set(HOLDINGS_AUDIT_COLUMNS).difference(parquet_file.schema_arrow.names))
        if missing:
            raise KeyError(
                f"A holdings partition is missing audit columns: {', '.join(missing)}"
            )
        for batch in parquet_file.iter_batches(
            columns=list(HOLDINGS_AUDIT_COLUMNS),
            batch_size=batch_size,
        ):
            frame = batch.to_pandas()
            rows_scanned += len(frame)
            batches_scanned += 1
            report_date = pd.to_datetime(
                frame["period_of_report"], errors="coerce", utc=True
            )
            frame["report_quarter"] = report_date.dt.tz_localize(None).dt.to_period("Q").astype(
                "string"
            )
            values = pd.to_numeric(frame["value_usd"], errors="coerce")
            classified = classify_instrument_rows(frame)
            classified["_rows"] = 1
            classified["_positive_value_rows"] = values.gt(0).astype("int64")
            classified["_missing_value_rows"] = values.isna().astype("int64")
            classified["_reported_value_usd"] = values.fillna(0.0)
            classified["_resolved_issuer_rows"] = _nonblank(
                classified["resolved_issuer_cik"]
            ).astype("int64")
            instrument_parts.append(
                classified.groupby(
                    ["report_quarter", "instrument_class"], dropna=False
                )
                .agg(
                    rows=("_rows", "sum"),
                    positive_value_rows=("_positive_value_rows", "sum"),
                    missing_value_rows=("_missing_value_rows", "sum"),
                    reported_value_usd=("_reported_value_usd", "sum"),
                    resolved_issuer_rows=("_resolved_issuer_rows", "sum"),
                )
                .reset_index()
            )

            confidence = (
                frame["issuer_identity_confidence"]
                .astype("string")
                .str.strip()
                .fillna("<missing>")
                .replace("", "<missing>")
            )
            identity = pd.DataFrame(
                {
                    "report_quarter": frame["report_quarter"],
                    "identity_confidence": confidence,
                    "rows": 1,
                    "reported_cik_rows": _nonblank(frame["reported_issuer_cik"]).astype(
                        "int64"
                    ),
                    "resolved_cik_rows": _nonblank(frame["resolved_issuer_cik"]).astype(
                        "int64"
                    ),
                    "provider_entity_rows": _nonblank(
                        frame["issuer_provider_entity_id"]
                    ).astype("int64"),
                }
            )
            identity_parts.append(
                identity.groupby(
                    ["report_quarter", "identity_confidence"], dropna=False
                )
                .sum(numeric_only=True)
                .reset_index()
            )

            filed_at = pd.to_datetime(frame["filed_at"], errors="coerce", utc=True)
            available_at = pd.to_datetime(
                frame["availability_timestamp_utc"], errors="coerce", utc=True
            )
            timing = pd.DataFrame(
                {
                    "report_quarter": frame["report_quarter"],
                    "rows": 1,
                    "missing_report_date_rows": report_date.isna().astype("int64"),
                    "missing_filed_at_rows": filed_at.isna().astype("int64"),
                    "missing_available_at_rows": available_at.isna().astype("int64"),
                    "report_after_filing_rows": (
                        report_date.notna() & filed_at.notna() & report_date.gt(filed_at)
                    ).astype("int64"),
                    "available_before_filing_rows": (
                        available_at.notna()
                        & filed_at.notna()
                        & available_at.lt(filed_at)
                    ).astype("int64"),
                }
            )
            timing_parts.append(
                timing.groupby("report_quarter", dropna=False)
                .sum(numeric_only=True)
                .reset_index()
            )

    return HoldingsAuditResult(
        instrument=_sum_audit_parts(
            instrument_parts,
            ("report_quarter", "instrument_class"),
        ),
        issuer_identity=_sum_audit_parts(
            identity_parts,
            ("report_quarter", "identity_confidence"),
        ),
        timing=_sum_audit_parts(timing_parts, ("report_quarter",)),
        metadata={
            "audit_contract_version": AUDIT_CONTRACT_VERSION,
            "cache_hit": False,
            "input_fingerprint": parquet_input_fingerprint(table_directory),
            "partitions_scanned": len(files),
            "batches_scanned": batches_scanned,
            "rows_scanned": rows_scanned,
            "batch_size": batch_size,
        },
    )


def full_history_holdings_audit(
    table_directory: str | Path,
    *,
    cache_directory: str | Path | None = None,
    batch_size: int = 250_000,
) -> HoldingsAuditResult:
    """Scan every holdings row, optionally reusing fingerprinted aggregate cache."""

    fingerprint = parquet_input_fingerprint(table_directory)
    if cache_directory is None:
        return _scan_holdings(table_directory, batch_size=batch_size)

    cache_root = Path(cache_directory).expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_key = f"{AUDIT_CONTRACT_VERSION}-{fingerprint}"
    paths = {
        "instrument": cache_root / f"{cache_key}-instrument.parquet",
        "issuer_identity": cache_root / f"{cache_key}-issuer-identity.parquet",
        "timing": cache_root / f"{cache_key}-timing.parquet",
        "metadata": cache_root / f"{cache_key}-metadata.json",
    }
    if all(path.is_file() for path in paths.values()):
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        metadata["cache_hit"] = True
        return HoldingsAuditResult(
            instrument=pd.read_parquet(paths["instrument"]),
            issuer_identity=pd.read_parquet(paths["issuer_identity"]),
            timing=pd.read_parquet(paths["timing"]),
            metadata=metadata,
        )

    result = _scan_holdings(table_directory, batch_size=batch_size)
    result.instrument.to_parquet(paths["instrument"], index=False)
    result.issuer_identity.to_parquet(paths["issuer_identity"], index=False)
    result.timing.to_parquet(paths["timing"], index=False)
    paths["metadata"].write_text(
        json.dumps(result.metadata, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return result


def mapping_snapshot_audit(
    table_directory: str | Path,
    *,
    batch_size: int = 250_000,
) -> pd.DataFrame:
    """Aggregate every issuer-mapping snapshot without exposing identifiers."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    parts: list[pd.DataFrame] = []
    for path in _parquet_files(table_directory):
        parquet_file = pq.ParquetFile(path)
        columns = (
            "mapping_snapshot_date",
            "identity_confidence",
            "resolved_issuer_cik",
            "issuer_provider_entity_id",
        )
        missing = sorted(set(columns).difference(parquet_file.schema_arrow.names))
        if missing:
            raise KeyError(
                f"An issuer mapping partition is missing audit columns: {', '.join(missing)}"
            )
        for batch in parquet_file.iter_batches(columns=list(columns), batch_size=batch_size):
            frame = batch.to_pandas()
            snapshot_date = pd.to_datetime(
                frame["mapping_snapshot_date"], errors="coerce", utc=True
            )
            confidence = (
                frame["identity_confidence"]
                .astype("string")
                .str.strip()
                .fillna("<missing>")
                .replace("", "<missing>")
            )
            aggregate = pd.DataFrame(
                {
                    "mapping_snapshot_date": snapshot_date.dt.date.astype("string"),
                    "identity_confidence": confidence,
                    "rows": 1,
                    "resolved_cik_rows": _nonblank(frame["resolved_issuer_cik"]).astype(
                        "int64"
                    ),
                    "provider_entity_rows": _nonblank(
                        frame["issuer_provider_entity_id"]
                    ).astype("int64"),
                }
            )
            parts.append(
                aggregate.groupby(
                    ["mapping_snapshot_date", "identity_confidence"], dropna=False
                )
                .sum(numeric_only=True)
                .reset_index()
            )
    return _sum_audit_parts(
        parts,
        ("mapping_snapshot_date", "identity_confidence"),
    )
