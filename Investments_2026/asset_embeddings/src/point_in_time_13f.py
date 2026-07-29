"""Canonical point-in-time state transitions for 13F holdings.

The immutable source grain is manager CIK x filing accession x holding line.
This module applies filing events in availability order and never rewrites an
earlier snapshot with a later amendment. It deliberately preserves raw
security identifiers; issuer mapping and common-equity eligibility are later,
separately tested stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd
import pyarrow.parquet as pq

from src.data_spine_13f import prepare_amendment_events
from src.full_history_13f import read_parquet_columns
from src.holdings_exploration import require_columns


POINT_IN_TIME_CONTRACT_VERSION = "13f-point-in-time-v1"
SELF_REPORTED_REPORT_TYPE = "13F HOLDINGS REPORT"

FILING_EVENT_COLUMNS = (
    "accession_no",
    "manager_cik",
    "form_type",
    "period_of_report",
    "filed_at",
    "filed_month",
    "availability_timestamp_utc",
    "is_amendment",
    "holdings_count",
)

COVER_EVENT_COLUMNS = (
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

HOLDING_STATE_COLUMNS = (
    "holding_id",
    "accession_no",
    "manager_cik",
    "period_of_report",
)


@dataclass(frozen=True)
class PointInTime13FPanel:
    """A decision-time snapshot plus its complete event and state audit trail."""

    holdings_asof: pd.DataFrame
    manager_periods: pd.DataFrame
    event_ledger: pd.DataFrame
    decision_at_utc: pd.Timestamp
    contract_version: str = POINT_IN_TIME_CONTRACT_VERSION


def _utc_decision_timestamp(value: object) -> pd.Timestamp:
    decision_at = pd.Timestamp(value)
    if decision_at.tzinfo is None:
        raise ValueError("decision_at must be timezone-aware")
    return decision_at.tz_convert("UTC")


def _clean_string(values: pd.Series) -> pd.Series:
    strings = values.astype("string").str.strip()
    return strings.mask(strings.eq(""))


def _require_unique_nonmissing_key(
    frame: pd.DataFrame,
    column: str,
    *,
    context: str,
) -> None:
    keys = _clean_string(frame[column])
    if keys.isna().any():
        raise ValueError(f"{context} contains a missing {column}")
    duplicates = keys.duplicated(keep=False)
    if duplicates.any():
        raise ValueError(f"{context} contains duplicate {column} values")


def _coverage_state(report_type: pd.Series) -> pd.Series:
    normalized = report_type.astype("string").str.upper().str.strip()
    states = pd.Series(
        "unknown_or_conflicting",
        index=report_type.index,
        dtype="string",
    )
    states.loc[normalized.eq("13F HOLDINGS REPORT").fillna(False)] = (
        "self_reported_holdings"
    )
    states.loc[normalized.eq("13F COMBINATION REPORT").fillna(False)] = (
        "partial_combination"
    )
    states.loc[normalized.eq("13F NOTICE").fillna(False)] = "delegated_notice"
    return states


def _different_when_present(left: pd.Series, right: pd.Series) -> pd.Series:
    return left.notna() & right.notna() & left.ne(right)


def prepare_point_in_time_events(
    filings: pd.DataFrame,
    cover_pages: pd.DataFrame,
    *,
    decision_at: object,
) -> pd.DataFrame:
    """Join filing and cover-page metadata into a decision-specific event ledger.

    The filing table is the complete event inventory. Missing cover pages,
    conflicting metadata, notices, combination reports, ambiguous amendments,
    and events unavailable at the cutoff remain explicit ledger states.
    """

    require_columns(filings, FILING_EVENT_COLUMNS, context="13F filing events")
    require_columns(cover_pages, COVER_EVENT_COLUMNS, context="13F cover-page events")
    _require_unique_nonmissing_key(
        filings,
        "accession_no",
        context="13F filing events",
    )
    _require_unique_nonmissing_key(
        cover_pages,
        "accession_no",
        context="13F cover-page events",
    )
    decision_at_utc = _utc_decision_timestamp(decision_at)

    filing_events = filings.copy()
    filing_events["accession_no"] = _clean_string(filing_events["accession_no"])
    filing_events["manager_cik"] = _clean_string(filing_events["manager_cik"])
    filing_events["report_date_utc"] = pd.to_datetime(
        filing_events["period_of_report"],
        errors="coerce",
        utc=True,
    )
    filing_events["filed_at_utc"] = pd.to_datetime(
        filing_events["filed_at"],
        errors="coerce",
        utc=True,
    )
    filing_events["available_at_utc"] = pd.to_datetime(
        filing_events["availability_timestamp_utc"],
        errors="coerce",
        utc=True,
    )
    filing_events["expected_holding_rows"] = pd.to_numeric(
        filing_events["holdings_count"],
        errors="coerce",
    ).astype("Int64")

    prepared_covers = prepare_amendment_events(cover_pages)
    prepared_covers["accession_no"] = _clean_string(
        prepared_covers["accession_no"]
    )
    prepared_covers["manager_cik"] = _clean_string(prepared_covers["manager_cik"])
    cover_fields = prepared_covers.loc[
        :,
        [
            "accession_no",
            "manager_cik",
            "form_type",
            "report_date_utc",
            "filed_at_utc",
            "available_at_utc",
            "is_amendment",
            "report_type",
            "event_role",
            "amendment_number",
            "amendment_signal_conflict",
            "amendment_type_missing",
            "amendment_json_invalid",
        ],
    ].rename(
        columns={
            "manager_cik": "cover_manager_cik",
            "form_type": "cover_form_type",
            "report_date_utc": "cover_report_date_utc",
            "filed_at_utc": "cover_filed_at_utc",
            "available_at_utc": "cover_available_at_utc",
            "is_amendment": "cover_is_amendment",
        }
    )
    cover_fields["cover_page_present"] = True

    events = filing_events.merge(
        cover_fields,
        on="accession_no",
        how="left",
        validate="one_to_one",
    )
    events["cover_page_present"] = (
        events["cover_page_present"].astype("boolean").fillna(False)
    )
    events["event_role"] = events["event_role"].astype("string").fillna("UNKNOWN")
    events["coverage_state"] = _coverage_state(events["report_type"])

    manager_conflict = _different_when_present(
        events["manager_cik"],
        events["cover_manager_cik"],
    )
    form_conflict = _different_when_present(
        _clean_string(events["form_type"]).str.upper(),
        _clean_string(events["cover_form_type"]).str.upper(),
    )
    report_date_conflict = _different_when_present(
        events["report_date_utc"],
        events["cover_report_date_utc"],
    )
    filed_at_conflict = _different_when_present(
        events["filed_at_utc"],
        events["cover_filed_at_utc"],
    )
    availability_conflict = _different_when_present(
        events["available_at_utc"],
        events["cover_available_at_utc"],
    )
    filing_amendment_flag = events["is_amendment"].astype("boolean")
    cover_amendment_flag = events["cover_is_amendment"].astype("boolean")
    amendment_flag_conflict = _different_when_present(
        filing_amendment_flag,
        cover_amendment_flag,
    )
    events["source_field_conflict"] = (
        manager_conflict
        | form_conflict
        | report_date_conflict
        | filed_at_conflict
        | availability_conflict
        | amendment_flag_conflict
    )

    events["decision_at_utc"] = decision_at_utc
    events["visible_by_decision"] = (
        events["available_at_utc"].notna()
        & events["available_at_utc"].le(decision_at_utc)
    )
    events["event_status"] = "pending"

    future = events["available_at_utc"].gt(decision_at_utc)
    events.loc[future.fillna(False), "event_status"] = "future_event"

    def quarantine_pending(mask: pd.Series, status: str) -> None:
        use = events["event_status"].eq("pending") & mask.fillna(False)
        events.loc[use, "event_status"] = status

    quarantine_pending(
        events["available_at_utc"].isna()
        | events["filed_at_utc"].isna()
        | events["manager_cik"].isna()
        | events["report_date_utc"].isna(),
        "quarantined_missing_key_or_availability",
    )
    quarantine_pending(
        ~events["cover_page_present"],
        "quarantined_missing_cover_page",
    )
    quarantine_pending(
        events["source_field_conflict"],
        "quarantined_source_field_conflict",
    )

    pending_notice = (
        events["event_status"].eq("pending")
        & events["coverage_state"].eq("delegated_notice")
    )
    events.loc[pending_notice, "event_status"] = "excluded_delegated_notice"
    pending_combination = (
        events["event_status"].eq("pending")
        & events["coverage_state"].eq("partial_combination")
    )
    events.loc[pending_combination, "event_status"] = "excluded_partial_combination"
    quarantine_pending(
        events["coverage_state"].eq("unknown_or_conflicting"),
        "quarantined_unknown_report_type",
    )
    quarantine_pending(
        events["amendment_signal_conflict"].astype("boolean").fillna(False),
        "quarantined_amendment_signal_conflict",
    )
    quarantine_pending(
        events["event_role"].eq("UNKNOWN")
        | events["amendment_type_missing"].astype("boolean").fillna(False)
        | events["amendment_json_invalid"].astype("boolean").fillna(False),
        "quarantined_unknown_amendment_type",
    )
    quarantine_pending(
        events["expected_holding_rows"].isna()
        | events["expected_holding_rows"].lt(0),
        "quarantined_invalid_expected_holding_count",
    )

    events["holding_rows_loaded"] = pd.Series(pd.NA, index=events.index, dtype="Int64")
    events["applied"] = False
    events["transition"] = pd.Series(pd.NA, index=events.index, dtype="string")
    events["state_complete_after_event"] = pd.Series(
        pd.NA,
        index=events.index,
        dtype="boolean",
    )
    events["visible_quarantine_after_event"] = pd.Series(
        pd.NA,
        index=events.index,
        dtype="boolean",
    )
    events["state_rows_after_event"] = pd.Series(
        pd.NA,
        index=events.index,
        dtype="Int64",
    )
    return events.sort_values(
        [
            "manager_cik",
            "report_date_utc",
            "available_at_utc",
            "filed_at_utc",
            "accession_no",
        ],
        na_position="last",
    ).reset_index(drop=True)


def _audit_loaded_holding_rows(
    events: pd.DataFrame,
    holdings: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    require_columns(holdings, HOLDING_STATE_COLUMNS, context="13F holding-state rows")
    audited_events = events.copy()
    holding_rows = holdings.copy()
    holding_rows["holding_id"] = _clean_string(holding_rows["holding_id"])
    holding_rows["accession_no"] = _clean_string(holding_rows["accession_no"])
    holding_rows["manager_cik"] = _clean_string(holding_rows["manager_cik"])
    holding_rows["_report_date_utc"] = pd.to_datetime(
        holding_rows["period_of_report"],
        errors="coerce",
        utc=True,
    )

    if holding_rows["holding_id"].isna().any():
        raise ValueError("13F holding-state rows contain a missing holding_id")
    if holding_rows["holding_id"].duplicated(keep=False).any():
        raise ValueError("13F holding-state rows contain duplicate holding_id values")

    loaded_counts = holding_rows.groupby("accession_no", dropna=False).size()
    audited_events["holding_rows_loaded"] = (
        audited_events["accession_no"].map(loaded_counts).fillna(0).astype("Int64")
    )
    pending = audited_events["event_status"].eq("pending")
    count_mismatch = audited_events["holding_rows_loaded"].ne(
        audited_events["expected_holding_rows"]
    )
    audited_events.loc[
        pending & count_mismatch,
        "event_status",
    ] = "quarantined_holding_count_mismatch"

    relevant = holding_rows.merge(
        audited_events.loc[
            :,
            ["accession_no", "manager_cik", "report_date_utc"],
        ].rename(
            columns={
                "manager_cik": "_event_manager_cik",
                "report_date_utc": "_event_report_date_utc",
            }
        ),
        on="accession_no",
        how="inner",
        validate="many_to_one",
    )
    if not relevant.empty:
        relevant["_metadata_conflict"] = (
            relevant["manager_cik"].isna()
            | relevant["_report_date_utc"].isna()
            | relevant["manager_cik"].ne(relevant["_event_manager_cik"]).fillna(True)
            | relevant["_report_date_utc"]
            .ne(relevant["_event_report_date_utc"])
            .fillna(True)
        )
        conflict_by_accession = relevant.groupby("accession_no")[
            "_metadata_conflict"
        ].any()
        metadata_conflict = (
            audited_events["accession_no"]
            .map(conflict_by_accession)
            .fillna(False)
            .astype(bool)
        )
        audited_events.loc[
            audited_events["event_status"].eq("pending") & metadata_conflict,
            "event_status",
        ] = "quarantined_holding_metadata_conflict"

    return audited_events, holding_rows


def build_point_in_time_13f_panel(
    filings: pd.DataFrame,
    cover_pages: pd.DataFrame,
    holdings: pd.DataFrame,
    *,
    decision_at: object,
) -> PointInTime13FPanel:
    """Apply all visible filing events and return the canonical as-of state.

    `holdings_asof` includes complete and incomplete audit states. Downstream
    modeling must use the explicit `state_model_eligible` flag rather than
    silently assuming every returned row belongs to a complete portfolio.
    """

    decision_at_utc = _utc_decision_timestamp(decision_at)
    prepared_events = prepare_point_in_time_events(
        filings,
        cover_pages,
        decision_at=decision_at_utc,
    )
    events, holding_rows = _audit_loaded_holding_rows(prepared_events, holdings)
    holdings_by_accession = {
        accession: list(indexes)
        for accession, indexes in holding_rows.groupby(
            "accession_no",
            sort=False,
        ).groups.items()
    }

    state_records: list[dict[str, object]] = []
    final_state_indexes: dict[tuple[object, object], list[int]] = {}
    grouped = events.dropna(subset=["manager_cik", "report_date_utc"]).groupby(
        ["manager_cik", "report_date_utc"],
        sort=False,
    )
    for (manager_cik, report_date_utc), group in grouped:
        state_indexes: list[int] = []
        base_observed = False
        state_complete = False
        visible_quarantine = False
        last_applied_accession: object = pd.NA
        last_applied_at: object = pd.NaT
        applied_events = 0

        for event_index in group.index:
            status = str(events.at[event_index, "event_status"])
            if status.startswith("quarantined_"):
                visible_quarantine = True
                events.at[event_index, "visible_quarantine_after_event"] = True
                continue
            if status != "pending":
                continue

            accession = events.at[event_index, "accession_no"]
            role = str(events.at[event_index, "event_role"])
            event_indexes = holdings_by_accession.get(accession, [])
            transition: str

            if role == "ORIGINAL":
                if base_observed:
                    events.at[event_index, "event_status"] = (
                        "quarantined_duplicate_original"
                    )
                    visible_quarantine = True
                    events.at[
                        event_index,
                        "visible_quarantine_after_event",
                    ] = True
                    continue
                state_indexes = list(event_indexes)
                base_observed = True
                state_complete = True
                transition = "initialize_from_original"
            elif role == "NEW_HOLDINGS":
                state_indexes.extend(event_indexes)
                if base_observed:
                    transition = "supplement_with_new_holdings"
                else:
                    state_complete = False
                    transition = "supplement_without_observable_base"
            elif role == "RESTATEMENT":
                state_indexes = list(event_indexes)
                if base_observed:
                    state_complete = True
                    transition = "replace_with_restatement"
                else:
                    state_complete = False
                    transition = "restatement_without_observable_base"
            else:
                events.at[event_index, "event_status"] = (
                    "quarantined_unknown_transition"
                )
                visible_quarantine = True
                events.at[event_index, "visible_quarantine_after_event"] = True
                continue

            events.at[event_index, "event_status"] = (
                "applied" if state_complete else "applied_incomplete"
            )
            events.at[event_index, "applied"] = True
            events.at[event_index, "transition"] = transition
            events.at[event_index, "state_complete_after_event"] = state_complete
            events.at[
                event_index,
                "visible_quarantine_after_event",
            ] = visible_quarantine
            events.at[event_index, "state_rows_after_event"] = len(state_indexes)
            last_applied_accession = accession
            last_applied_at = events.at[event_index, "available_at_utc"]
            applied_events += 1

        model_eligible = bool(
            applied_events > 0 and state_complete and not visible_quarantine
        )
        if applied_events == 0:
            state_status = "no_applied_state"
        elif not state_complete:
            state_status = "incomplete"
        elif visible_quarantine:
            state_status = "complete_with_quarantine"
        else:
            state_status = "complete"

        visible_group = events.loc[group.index]
        visible_group = visible_group[visible_group["visible_by_decision"]]
        state_records.append(
            {
                "manager_cik": manager_cik,
                "report_date_utc": report_date_utc,
                "decision_at_utc": decision_at_utc,
                "visible_events": len(visible_group),
                "applied_events": applied_events,
                "quarantined_events": int(
                    visible_group["event_status"]
                    .astype("string")
                    .str.startswith("quarantined_")
                    .sum()
                ),
                "excluded_events": int(
                    visible_group["event_status"]
                    .astype("string")
                    .str.startswith("excluded_")
                    .sum()
                ),
                "state_rows": len(state_indexes),
                "state_complete": state_complete,
                "has_visible_quarantine": visible_quarantine,
                "model_eligible": model_eligible,
                "state_status": state_status,
                "last_applied_accession_no": last_applied_accession,
                "last_applied_at_utc": last_applied_at,
            }
        )
        if applied_events > 0:
            final_state_indexes[(manager_cik, report_date_utc)] = state_indexes

    manager_periods = pd.DataFrame.from_records(state_records)
    if manager_periods.empty:
        manager_periods = pd.DataFrame(
            columns=(
                "manager_cik",
                "report_date_utc",
                "decision_at_utc",
                "visible_events",
                "applied_events",
                "quarantined_events",
                "excluded_events",
                "state_rows",
                "state_complete",
                "has_visible_quarantine",
                "model_eligible",
                "state_status",
                "last_applied_accession_no",
                "last_applied_at_utc",
            )
        )

    state_frames: list[pd.DataFrame] = []
    source_event_metadata = events.set_index("accession_no")[
        ["event_role", "available_at_utc"]
    ]
    manager_state_metadata = manager_periods.set_index(
        ["manager_cik", "report_date_utc"]
    )
    for key, indexes in final_state_indexes.items():
        if not indexes:
            continue
        state = holding_rows.loc[indexes].copy()
        state["source_event_role"] = state["accession_no"].map(
            source_event_metadata["event_role"]
        )
        state["source_event_available_at_utc"] = state["accession_no"].map(
            source_event_metadata["available_at_utc"]
        )
        state["snapshot_decision_at_utc"] = decision_at_utc
        state["state_last_event_at_utc"] = manager_state_metadata.loc[
            key,
            "last_applied_at_utc",
        ]
        state["state_complete"] = manager_state_metadata.loc[key, "state_complete"]
        state["state_has_visible_quarantine"] = manager_state_metadata.loc[
            key,
            "has_visible_quarantine",
        ]
        state["state_model_eligible"] = manager_state_metadata.loc[
            key,
            "model_eligible",
        ]
        state_frames.append(state.drop(columns=["_report_date_utc"]))

    if state_frames:
        holdings_asof = pd.concat(state_frames, ignore_index=True)
    else:
        output_columns = [
            column
            for column in holding_rows.columns
            if column != "_report_date_utc"
        ] + [
            "source_event_role",
            "source_event_available_at_utc",
            "snapshot_decision_at_utc",
            "state_last_event_at_utc",
            "state_complete",
            "state_has_visible_quarantine",
            "state_model_eligible",
        ]
        holdings_asof = pd.DataFrame(columns=output_columns)

    return PointInTime13FPanel(
        holdings_asof=holdings_asof,
        manager_periods=manager_periods.sort_values(
            ["report_date_utc", "manager_cik"]
        ).reset_index(drop=True),
        event_ledger=events.sort_values(
            [
                "report_date_utc",
                "manager_cik",
                "available_at_utc",
                "filed_at_utc",
                "accession_no",
            ],
            na_position="last",
        ).reset_index(drop=True),
        decision_at_utc=decision_at_utc,
    )


def build_final_13f_panel(
    filings: pd.DataFrame,
    cover_pages: pd.DataFrame,
    holdings: pd.DataFrame,
) -> PointInTime13FPanel:
    """Build the latest corrected state for ex-post audit only.

    This view must never replace a historical `build_point_in_time_13f_panel`
    result in a model or backtest.
    """

    require_columns(
        filings,
        ("availability_timestamp_utc",),
        context="13F final-state filings",
    )
    latest_availability = pd.to_datetime(
        filings["availability_timestamp_utc"],
        errors="coerce",
        utc=True,
    ).max()
    if pd.isna(latest_availability):
        raise ValueError("Final-state construction requires an availability timestamp")
    return build_point_in_time_13f_panel(
        filings,
        cover_pages,
        holdings,
        decision_at=latest_availability,
    )


def load_holdings_for_events(
    holdings_directory: str | Path,
    events: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    batch_size: int = 250_000,
) -> pd.DataFrame:
    """Stream only filed-month partitions needed by visible candidate events."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    require_columns(
        events,
        ("accession_no", "filed_at_utc", "event_status"),
        context="prepared 13F event ledger",
    )
    selected = events[events["event_status"].eq("pending")].copy()
    accessions = set(_clean_string(selected["accession_no"]).dropna())
    if not accessions:
        return pd.DataFrame(columns=list(columns or HOLDING_STATE_COLUMNS))

    filed_months = set(
        selected["filed_at_utc"].dropna().dt.strftime("%Y%m").astype(str)
    )
    directory = Path(holdings_directory).expanduser().resolve()
    paths = sorted(
        path
        for month in filed_months
        for path in directory.glob(f"part-{month}.parquet")
    )
    if not paths:
        raise FileNotFoundError(
            "No holdings partitions matched the visible event filed months"
        )

    requested = list(dict.fromkeys(columns or pq.ParquetFile(paths[0]).schema_arrow.names))
    for required in HOLDING_STATE_COLUMNS:
        if required not in requested:
            requested.append(required)

    parts: list[pd.DataFrame] = []
    for path in paths:
        parquet_file = pq.ParquetFile(path)
        missing = sorted(set(requested).difference(parquet_file.schema_arrow.names))
        if missing:
            raise KeyError(
                f"A holdings partition is missing requested columns: {', '.join(missing)}"
            )
        for batch in parquet_file.iter_batches(
            columns=requested,
            batch_size=batch_size,
        ):
            frame = batch.to_pandas()
            use = _clean_string(frame["accession_no"]).isin(accessions)
            if use.any():
                parts.append(frame.loc[use].copy())

    if not parts:
        return pd.DataFrame(columns=requested)
    return pd.concat(parts, ignore_index=True)


def build_local_quarter_panel(
    product_root: str | Path,
    *,
    report_date: object,
    decision_at: object,
    holdings_columns: Sequence[str] | None = None,
    batch_size: int = 250_000,
) -> PointInTime13FPanel:
    """Build one reporting-quarter snapshot from the ignored local product."""

    root = Path(product_root).expanduser().resolve()
    filings = read_parquet_columns(root / "tables" / "filings", FILING_EVENT_COLUMNS)
    covers = read_parquet_columns(
        root / "tables" / "cover_pages",
        COVER_EVENT_COLUMNS,
    )
    target_report_date = pd.Timestamp(report_date)
    if target_report_date.tzinfo is not None:
        target_report_date = target_report_date.tz_localize(None)
    target_report_date = target_report_date.normalize()
    filing_dates = pd.to_datetime(filings["period_of_report"], errors="coerce")
    cover_dates = pd.to_datetime(covers["period_of_report"], errors="coerce")
    filings = filings.loc[filing_dates.eq(target_report_date)].copy()
    covers = covers.loc[cover_dates.eq(target_report_date)].copy()
    if filings.empty:
        raise ValueError("No filings match the requested report_date")

    events = prepare_point_in_time_events(
        filings,
        covers,
        decision_at=decision_at,
    )
    holdings = load_holdings_for_events(
        root / "tables" / "holdings",
        events,
        columns=holdings_columns,
        batch_size=batch_size,
    )
    return build_point_in_time_13f_panel(
        filings,
        covers,
        holdings,
        decision_at=decision_at,
    )


def eligible_holdings(panel: PointInTime13FPanel) -> pd.DataFrame:
    """Return only complete, unquarantined manager-period states."""

    if "state_model_eligible" not in panel.holdings_asof.columns:
        return panel.holdings_asof.iloc[0:0].copy()
    return panel.holdings_asof[
        panel.holdings_asof["state_model_eligible"].astype("boolean").fillna(False)
    ].copy()
