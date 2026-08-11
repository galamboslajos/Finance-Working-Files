"""Resumable point-in-time 13F manager-security matrix materialization.

The committed code is public, while the generated pair tables remain below an
ignored local directory because they contain manager CIKs and reported CUSIPs.
Every quarter is rebuilt from filing events visible at its standardized cutoff;
later amendments never rewrite an earlier matrix.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Callable

import pandas as pd
import pyarrow.parquet as pq

from src.data_spine_13f import parquet_input_fingerprint, quarter_coverage_audit
from src.full_history_13f import read_parquet_columns
from src.point_in_time_13f import (
    FILING_EVENT_COLUMNS,
    build_local_quarter_panel,
)
from src.security_master_13f import (
    CASH_SHARE_CHANNEL,
    SECURITY_MASTER_CONTRACT_VERSION,
    build_cash_share_matrix_candidate,
    classify_security_rows,
    security_row_audit,
)


MATRIX_CONTRACT_VERSION = "13f-pit-matrix-v1"
CUSIP_POLICY = "strict_reported_nine_character_no_repair"
DEFAULT_HOLDING_COLUMNS = (
    "holding_id",
    "accession_no",
    "manager_cik",
    "period_of_report",
    "filed_at",
    "availability_timestamp_utc",
    "cusip",
    "title_of_class",
    "put_call",
    "shares_or_principal_type",
    "value_usd",
    "shares_or_principal_amount",
    "reported_issuer_cik",
)


@dataclass(frozen=True)
class MatrixParameters:
    """Frozen baseline representation filters."""

    maximum_position_weight: float = 0.75
    minimum_assets_per_manager: int = 20
    minimum_managers_per_asset: int = 20
    batch_size: int = 250_000


@dataclass(frozen=True)
class QuarterMatrix13F:
    """One private pair table plus public-safe aggregate diagnostics."""

    pairs: pd.DataFrame
    row_audit: pd.DataFrame
    event_audit: pd.DataFrame
    manager_state_audit: pd.DataFrame
    stage_audit: pd.DataFrame
    iterative_log: pd.DataFrame
    checks: pd.DataFrame
    summary: pd.DataFrame
    report_date_utc: pd.Timestamp
    decision_at_utc: pd.Timestamp


@dataclass(frozen=True)
class MaterializedQuarter:
    """Small record returned after building or reusing a local artifact."""

    report_date: str
    decision_at_utc: str
    quarter_directory: str
    pair_rows: int
    managers: int
    assets: int
    input_fingerprint: str
    source_fingerprint: str
    pair_file_sha256: str
    cache_hit: bool


@dataclass(frozen=True)
class FullHistoryValidation:
    """Aggregate gates plus one public-safe status row per expected quarter."""

    checks: pd.DataFrame
    quarter_status: pd.DataFrame


def _utc_timestamp(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError("A point-in-time cutoff must be timezone-aware")
    return timestamp.tz_convert("UTC")


def _report_timestamp(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_localize(None)
    return timestamp.normalize()


def _quarter_label(report_date: object) -> str:
    return str(_report_timestamp(report_date).to_period("Q"))


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_code_fingerprint(project_root: str | Path) -> str:
    """Fingerprint only the modules that define a quarterly matrix."""

    root = Path(project_root).expanduser().resolve()
    relative_paths = (
        Path("src/data_spine_13f.py"),
        Path("src/point_in_time_13f.py"),
        Path("src/security_master_13f.py"),
        Path("src/matrix_13f.py"),
    )
    digest = sha256()
    for relative_path in relative_paths:
        path = root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Matrix source file is missing: {path}")
        digest.update(relative_path.as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def product_input_fingerprint(product_root: str | Path) -> str:
    """Fingerprint the three immutable source tables used by the PIT builder."""

    root = Path(product_root).expanduser().resolve()
    digest = sha256()
    for table in ("filings", "cover_pages", "holdings"):
        digest.update(table.encode("utf-8"))
        digest.update(
            parquet_input_fingerprint(root / "tables" / table).encode("ascii")
        )
    return digest.hexdigest()


def decision_complete_quarters(product_root: str | Path) -> pd.DataFrame:
    """Derive every usable report quarter and cutoff from the local product."""

    root = Path(product_root).expanduser().resolve()
    filings = read_parquet_columns(root / "tables" / "filings", FILING_EVENT_COLUMNS)
    coverage = quarter_coverage_audit(filings)
    eligible = coverage.loc[coverage["eligible_complete_window"]].copy()
    if eligible.empty:
        raise ValueError("No decision-complete 13F quarter is available")
    return eligible.sort_values("report_date_utc").reset_index(drop=True)


def _audit_counts(frame: pd.DataFrame, column: str, count_name: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=(column, count_name))
    return (
        frame.groupby(column, dropna=False)
        .size()
        .rename(count_name)
        .reset_index()
        .sort_values(column)
        .reset_index(drop=True)
    )


def build_quarter_matrix(
    product_root: str | Path,
    *,
    report_date: object,
    decision_at: object,
    parameters: MatrixParameters = MatrixParameters(),
) -> QuarterMatrix13F:
    """Build and validate one sparse manager-security pair table in memory."""

    report_date_value = _report_timestamp(report_date)
    decision_at_utc = _utc_timestamp(decision_at)
    panel = build_local_quarter_panel(
        product_root,
        report_date=report_date_value,
        decision_at=decision_at_utc,
        holdings_columns=DEFAULT_HOLDING_COLUMNS,
        batch_size=parameters.batch_size,
    )
    classified = classify_security_rows(panel.holdings_asof)
    candidate = build_cash_share_matrix_candidate(
        classified,
        maximum_position_weight=parameters.maximum_position_weight,
        minimum_assets_per_manager=parameters.minimum_assets_per_manager,
        minimum_managers_per_asset=parameters.minimum_managers_per_asset,
    )

    group_columns = ["period_of_report", "manager_cik", "typed_security_id"]
    eligible = classified.loc[classified["matrix_eligible"]].copy()
    eligible["period_of_report"] = pd.to_datetime(
        eligible["period_of_report"], errors="coerce"
    ).dt.normalize()
    provenance = (
        eligible.groupby(group_columns, as_index=False)
        .agg(
            first_source_event_available_at_utc=(
                "source_event_available_at_utc",
                "min",
            ),
            last_source_event_available_at_utc=(
                "source_event_available_at_utc",
                "max",
            ),
            state_last_event_at_utc=("state_last_event_at_utc", "max"),
            snapshot_decision_at_utc=("snapshot_decision_at_utc", "max"),
        )
    )
    pairs = candidate.filtered_positions.merge(
        provenance,
        on=group_columns,
        how="left",
        validate="one_to_one",
    )
    pairs["report_date_utc"] = pd.to_datetime(
        pairs["period_of_report"], errors="coerce", utc=True
    )
    pairs["decision_at_utc"] = decision_at_utc
    pairs["binary_owned"] = 1
    pairs = pairs.sort_values(
        ["report_date_utc", "manager_cik", "typed_security_id"]
    ).reset_index(drop=True)

    event_audit = _audit_counts(panel.event_ledger, "event_status", "events")
    manager_state_audit = _audit_counts(
        panel.manager_periods,
        "state_status",
        "manager_periods",
    )

    prefix = f"{CASH_SHARE_CHANNEL}:"
    manager_degrees = pairs.groupby("manager_cik")["typed_security_id"].nunique()
    asset_degrees = pairs.groupby("typed_security_id")["manager_cik"].nunique()
    future_source_events = int(
        pairs["last_source_event_available_at_utc"].gt(decision_at_utc).sum()
    )
    wrong_snapshot_cutoff = int(
        pairs["snapshot_decision_at_utc"].ne(decision_at_utc).sum()
    )
    wrong_report_date = int(
        pairs["report_date_utc"].ne(report_date_value.tz_localize("UTC")).sum()
    )
    wrong_channel = int(
        (~pairs["typed_security_id"].astype("string").str.startswith(prefix)).sum()
    )
    concentration_violations = int(
        pairs.groupby("manager_cik")["matrix_weight"]
        .max()
        .gt(parameters.maximum_position_weight + 1e-12)
        .sum()
    )
    degree_manager_violations = int(
        manager_degrees.lt(parameters.minimum_assets_per_manager).sum()
    )
    degree_asset_violations = int(
        asset_degrees.lt(parameters.minimum_managers_per_asset).sum()
    )
    local_checks = pd.DataFrame(
        {
            "check": [
                "source events are available by decision",
                "snapshot cutoff matches the quarter cutoff",
                "pair report date matches the requested quarter",
                "pairs use only the typed cash-share channel",
                "retained weights respect the concentration cap",
                "retained managers satisfy minimum asset degree",
                "retained assets satisfy minimum manager degree",
            ],
            "violations": [
                future_source_events,
                wrong_snapshot_cutoff,
                wrong_report_date,
                wrong_channel,
                concentration_violations,
                degree_manager_violations,
                degree_asset_violations,
            ],
            "maximum_absolute_error": [0.0] * 7,
        }
    )
    checks = pd.concat([candidate.checks, local_checks], ignore_index=True)

    managers = int(pairs["manager_cik"].nunique()) if len(pairs) else 0
    assets = int(pairs["typed_security_id"].nunique()) if len(pairs) else 0
    summary = pd.DataFrame.from_records(
        [
            {
                "report_date_utc": report_date_value.tz_localize("UTC"),
                "decision_at_utc": decision_at_utc,
                "pairs": len(pairs),
                "managers": managers,
                "assets": assets,
                "density_pct": 100 * len(pairs) / (managers * assets)
                if managers and assets
                else 0.0,
                "reported_value_usd": float(pairs["position_value_usd"].sum()),
                "blocking_violations": int(checks["violations"].sum()),
            }
        ]
    )
    return QuarterMatrix13F(
        pairs=pairs,
        row_audit=security_row_audit(classified),
        event_audit=event_audit,
        manager_state_audit=manager_state_audit,
        stage_audit=candidate.audit,
        iterative_log=candidate.iterative_log,
        checks=checks,
        summary=summary,
        report_date_utc=report_date_value.tz_localize("UTC"),
        decision_at_utc=decision_at_utc,
    )


def _write_json_atomic(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def _materialization_from_manifest(
    manifest: dict[str, object],
    *,
    quarter_directory: Path,
    cache_hit: bool,
) -> MaterializedQuarter:
    return MaterializedQuarter(
        report_date=str(manifest["report_date"]),
        decision_at_utc=str(manifest["decision_at_utc"]),
        quarter_directory=str(quarter_directory),
        pair_rows=int(manifest["pair_rows"]),
        managers=int(manifest["managers"]),
        assets=int(manifest["assets"]),
        input_fingerprint=str(manifest["input_fingerprint"]),
        source_fingerprint=str(manifest["source_fingerprint"]),
        pair_file_sha256=str(manifest["pair_file_sha256"]),
        cache_hit=cache_hit,
    )


def materialize_quarter_matrix(
    product_root: str | Path,
    output_root: str | Path,
    *,
    report_date: object,
    decision_at: object,
    input_fingerprint: str,
    source_fingerprint: str,
    parameters: MatrixParameters = MatrixParameters(),
    force: bool = False,
) -> MaterializedQuarter:
    """Build or reuse one fingerprinted local quarterly artifact."""

    output = Path(output_root).expanduser().resolve()
    quarter_directory = output / f"report_quarter={_quarter_label(report_date)}"
    quarter_directory.mkdir(parents=True, exist_ok=True)
    pair_path = quarter_directory / "pairs.parquet"
    audit_path = quarter_directory / "audit.json"
    manifest_path = quarter_directory / "manifest.json"
    parameter_payload = asdict(parameters)

    if not force and pair_path.is_file() and manifest_path.is_file():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        reusable = (
            prior.get("contract_version") == MATRIX_CONTRACT_VERSION
            and prior.get("security_master_contract_version")
            == SECURITY_MASTER_CONTRACT_VERSION
            and prior.get("cusip_policy") == CUSIP_POLICY
            and prior.get("input_fingerprint") == input_fingerprint
            and prior.get("source_fingerprint") == source_fingerprint
            and prior.get("parameters") == parameter_payload
            and prior.get("pair_file_sha256") == _sha256_file(pair_path)
        )
        if reusable:
            return _materialization_from_manifest(
                prior,
                quarter_directory=quarter_directory,
                cache_hit=True,
            )

    quarter = build_quarter_matrix(
        product_root,
        report_date=report_date,
        decision_at=decision_at,
        parameters=parameters,
    )
    if not quarter.checks["violations"].eq(0).all():
        failed = quarter.checks.loc[quarter.checks["violations"].ne(0)]
        raise AssertionError(f"Quarter matrix checks failed:\n{failed.to_string(index=False)}")

    temporary_pair = pair_path.with_suffix(".parquet.tmp")
    quarter.pairs.to_parquet(temporary_pair, index=False)
    temporary_pair.replace(pair_path)
    pair_file_sha256 = _sha256_file(pair_path)
    summary = quarter.summary.iloc[0]
    audit_payload = {
        "row_audit": quarter.row_audit.to_dict(orient="records"),
        "event_audit": quarter.event_audit.to_dict(orient="records"),
        "manager_state_audit": quarter.manager_state_audit.to_dict(orient="records"),
        "stage_audit": quarter.stage_audit.to_dict(orient="records"),
        "iterative_log": quarter.iterative_log.to_dict(orient="records"),
        "checks": quarter.checks.to_dict(orient="records"),
        "summary": quarter.summary.to_dict(orient="records"),
    }
    _write_json_atomic(audit_path, audit_payload)
    manifest = {
        "contract_version": MATRIX_CONTRACT_VERSION,
        "security_master_contract_version": SECURITY_MASTER_CONTRACT_VERSION,
        "cusip_policy": CUSIP_POLICY,
        "report_date": quarter.report_date_utc.date().isoformat(),
        "decision_at_utc": quarter.decision_at_utc.isoformat(),
        "input_fingerprint": input_fingerprint,
        "source_fingerprint": source_fingerprint,
        "parameters": parameter_payload,
        "pair_rows": int(summary["pairs"]),
        "managers": int(summary["managers"]),
        "assets": int(summary["assets"]),
        "pair_file_sha256": pair_file_sha256,
    }
    _write_json_atomic(manifest_path, manifest)
    return _materialization_from_manifest(
        manifest,
        quarter_directory=quarter_directory,
        cache_hit=False,
    )


def materialize_full_history(
    product_root: str | Path,
    output_root: str | Path,
    *,
    project_root: str | Path,
    parameters: MatrixParameters = MatrixParameters(),
    force: bool = False,
    progress: Callable[[str], None] | None = print,
) -> pd.DataFrame:
    """Materialize all decision-complete quarters with safe resume semantics."""

    quarters = decision_complete_quarters(product_root)
    input_fingerprint = product_input_fingerprint(product_root)
    source_fingerprint = source_code_fingerprint(project_root)
    records: list[dict[str, object]] = []
    for row in quarters.itertuples(index=False):
        if progress is not None:
            progress(f"Building or validating {_quarter_label(row.report_date_utc)}")
        artifact = materialize_quarter_matrix(
            product_root,
            output_root,
            report_date=row.report_date_utc,
            decision_at=row.information_cutoff_utc,
            input_fingerprint=input_fingerprint,
            source_fingerprint=source_fingerprint,
            parameters=parameters,
            force=force,
        )
        records.append(asdict(artifact))

    run_summary = pd.DataFrame.from_records(records)
    if len(run_summary) != len(quarters):
        raise AssertionError("Not every decision-complete quarter was materialized")
    if run_summary["report_date"].duplicated().any():
        raise AssertionError("The full-history run contains duplicate report quarters")

    output = Path(output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    run_manifest = {
        "contract_version": MATRIX_CONTRACT_VERSION,
        "security_master_contract_version": SECURITY_MASTER_CONTRACT_VERSION,
        "cusip_policy": CUSIP_POLICY,
        "input_fingerprint": input_fingerprint,
        "source_fingerprint": source_fingerprint,
        "parameters": asdict(parameters),
        "decision_complete_quarters": len(quarters),
        "materialized_quarters": records,
    }
    _write_json_atomic(output / "run_manifest.json", run_manifest)
    return run_summary


def validate_materialized_history(
    product_root: str | Path,
    output_root: str | Path,
    *,
    project_root: str | Path,
    parameters: MatrixParameters = MatrixParameters(),
) -> FullHistoryValidation:
    """Validate all local artifacts without loading the complete pair history."""

    quarters = decision_complete_quarters(product_root)
    output = Path(output_root).expanduser().resolve()
    input_fingerprint = product_input_fingerprint(product_root)
    source_fingerprint = source_code_fingerprint(project_root)
    expected_parameters = asdict(parameters)
    records: list[dict[str, object]] = []

    for row in quarters.itertuples(index=False):
        label = _quarter_label(row.report_date_utc)
        directory = output / f"report_quarter={label}"
        pair_path = directory / "pairs.parquet"
        audit_path = directory / "audit.json"
        manifest_path = directory / "manifest.json"
        files_present = all(
            path.is_file() for path in (pair_path, audit_path, manifest_path)
        )
        record: dict[str, object] = {
            "report_quarter": label,
            "files_present": files_present,
            "manifest_current": False,
            "pair_hash_valid": False,
            "row_count_valid": False,
            "quarter_checks_pass": False,
            "pair_rows": 0,
            "managers": 0,
            "assets": 0,
        }
        if files_present:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            record["manifest_current"] = bool(
                manifest.get("contract_version") == MATRIX_CONTRACT_VERSION
                and manifest.get("security_master_contract_version")
                == SECURITY_MASTER_CONTRACT_VERSION
                and manifest.get("cusip_policy") == CUSIP_POLICY
                and manifest.get("input_fingerprint") == input_fingerprint
                and manifest.get("source_fingerprint") == source_fingerprint
                and manifest.get("parameters") == expected_parameters
                and manifest.get("report_date")
                == pd.Timestamp(row.report_date_utc).date().isoformat()
                and manifest.get("decision_at_utc")
                == _utc_timestamp(row.information_cutoff_utc).isoformat()
            )
            actual_hash = _sha256_file(pair_path)
            record["pair_hash_valid"] = bool(
                actual_hash == manifest.get("pair_file_sha256")
            )
            parquet_rows = pq.ParquetFile(pair_path).metadata.num_rows
            record["row_count_valid"] = bool(
                parquet_rows == int(manifest.get("pair_rows", -1))
            )
            checks = pd.DataFrame.from_records(audit.get("checks", []))
            record["quarter_checks_pass"] = bool(
                not checks.empty and checks["violations"].eq(0).all()
            )
            record["pair_rows"] = int(manifest.get("pair_rows", 0))
            record["managers"] = int(manifest.get("managers", 0))
            record["assets"] = int(manifest.get("assets", 0))
        records.append(record)

    quarter_status = pd.DataFrame.from_records(records)
    expected_labels = set(quarter_status["report_quarter"])
    actual_labels = {
        path.name.removeprefix("report_quarter=")
        for path in output.glob("report_quarter=*")
        if path.is_dir()
    }
    checks = pd.DataFrame(
        {
            "check": [
                "all decision-complete quarters are present",
                "no unexpected quarter partitions are present",
                "all manifests match current data, code, and policy",
                "all pair-file hashes match their manifests",
                "all Parquet row counts match their manifests",
                "all quarter-level PIT and matrix checks pass",
            ],
            "violations": [
                int((~quarter_status["files_present"]).sum()),
                len(actual_labels.difference(expected_labels)),
                int((~quarter_status["manifest_current"]).sum()),
                int((~quarter_status["pair_hash_valid"]).sum()),
                int((~quarter_status["row_count_valid"]).sum()),
                int((~quarter_status["quarter_checks_pass"]).sum()),
            ],
        }
    )
    return FullHistoryValidation(checks=checks, quarter_status=quarter_status)
