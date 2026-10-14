"""Historical constituent reconstruction with visible identity and quote failures.

The current sources do not establish knowledge-time identity/revision history.
This module never infers eligibility from current security-master flags or prices.
"""
from dataclasses import dataclass
from datetime import date
import json
import math

import pyarrow as pa


class DataContractError(ValueError):
    """The input cannot support an unambiguous, auditable result."""


class KnowledgeTimeUnavailable(DataContractError):
    """Historical knowledge-time evidence is absent from the source contract."""


def iso_date(value, name):
    if not isinstance(value, str):
        raise DataContractError(f"{name} must be an ISO date string")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise DataContractError(f"Invalid {name}: {value!r}") from error
    if parsed.isoformat() != value:
        raise DataContractError(f"{name} must use YYYY-MM-DD")
    return value


def _columns(table, required, name):
    missing = sorted(set(required) - set(table.column_names))
    if missing:
        raise DataContractError(f"{name} missing columns: {', '.join(missing)}")


def _id(value, name):
    if not isinstance(value, str) or not value.strip():
        raise DataContractError(f"Missing or invalid {name}")
    return value


def _optional_date(value, name):
    return None if value in (None, "") else iso_date(value, name)


UNIVERSE_SCHEMA = pa.schema([
    ("source_security_id", pa.string()), ("as_of", pa.string()),
    ("indices", pa.list_(pa.string())),
    ("membership_tickers", pa.list_(pa.string())),
    ("membership_records_json", pa.string()),
    ("unified_security_id", pa.string()), ("intrinio_security_id", pa.string()),
    ("identity_status", pa.string()), ("bridge_method", pa.string()),
    ("bridge_confidence", pa.string()), ("identity_review_required", pa.bool_()),
    ("resolution_status", pa.string()),
])
EXCLUDED_SCHEMA = pa.schema([
    ("source_security_id", pa.string()), ("index_id", pa.string()),
    ("trading_date", pa.string()), ("reason", pa.string()),
])


@dataclass(frozen=True)
class UniverseSnapshot:
    table: pa.Table
    excluded: pa.Table
    as_of: str
    universe: str
    provenance: dict
    summary: dict


@dataclass(frozen=True)
class QuoteSnapshot:
    table: pa.Table
    summary: dict


def build_snapshot(membership, bridge, links, *, as_of, universe,
                   mode="historical_reconstruction"):
    """Build a dated source-ID cohort; retain every unresolved constituent.

    End dates are inclusive under the audited Siblis contract. Future or absent
    availability never qualifies a record. A resolved identity is a unique
    retrieval-time route, not a certification of historical identity knowledge.
    """
    iso_date(as_of, "as_of")
    choices = {"sp500": {"sp500"}, "sp400": {"sp400"},
               "combined": {"sp500", "sp400"}}
    if universe not in choices:
        raise DataContractError("universe must be sp500, sp400 or combined")
    if mode == "knowledge_time":
        raise KnowledgeTimeUnavailable(
            "Current sources lack original membership publication timestamps, "
            "dated identity validity and revision-as-of price history. "
            "Only explicitly disclosed historical_reconstruction is supported.")
    if mode != "historical_reconstruction":
        raise DataContractError("Unknown PIT mode")
    _columns(membership, ["index_id", "trading_date", "security_id", "is_member",
             "membership_effective_date", "membership_end_date",
             "membership_availability_date"], "membership")
    _columns(bridge, ["source_product_id", "source_security_id",
                     "unified_security_id"], "bridge")
    _columns(links, ["unified_security_id", "intrinio_security_id",
                    "identity_review_required", "resolution_status"], "links")

    cohort, excluded, seen = {}, [], set()
    input_rows = 0
    for row in membership.to_pylist():
        day = iso_date(row["trading_date"], "trading_date")
        if row["index_id"] not in choices[universe] or day != as_of:
            continue
        input_rows += 1
        sid = _id(row["security_id"], "security_id")
        key = (row["index_id"], sid)
        if key in seen:
            raise DataContractError(f"Duplicate dated membership key: {key}")
        seen.add(key)
        effective = _optional_date(row["membership_effective_date"], "effective_date")
        end = _optional_date(row["membership_end_date"], "end_date")
        available = _optional_date(row["membership_availability_date"], "availability_date")
        if effective and end and end < effective:
            raise DataContractError("Membership end precedes effective date")
        reason = None
        if row["is_member"] is not True:
            reason = "not_member" if row["is_member"] is False else "membership_unknown"
        elif effective is None:
            reason = "effective_date_missing"
        elif effective > as_of:
            reason = "future_effective"
        elif end is not None and end < as_of:
            reason = "membership_ended"
        elif available is None:
            reason = "availability_missing"
        elif available > as_of:
            reason = "future_availability"
        if reason:
            excluded.append({"source_security_id": sid, "index_id": row["index_id"],
                             "trading_date": day, "reason": reason})
        else:
            cohort.setdefault(sid, []).append(row)

    routes, link_map = {}, {}
    for row in bridge.to_pylist():
        if row["source_product_id"] == "sp_composite1500":
            sid = _id(row["source_security_id"], "source_security_id")
            routes.setdefault(sid, []).append(row)
    for row in links.to_pylist():
        uid = _id(row["unified_security_id"], "unified_security_id")
        link_map.setdefault(uid, []).append(row)

    output = []
    for sid, records in sorted(cohort.items()):
        result = {"source_security_id": sid, "as_of": as_of,
                  "indices": sorted({r["index_id"] for r in records}),
                  "membership_tickers": sorted({r["membership_ticker"] for r in records
                                                if r.get("membership_ticker")}),
                  "membership_records_json": json.dumps(records, sort_keys=True),
                  "unified_security_id": None, "intrinio_security_id": None,
                  "identity_status": "unmapped_bridge",
                  "bridge_method": None, "bridge_confidence": None,
                  "identity_review_required": None, "resolution_status": None}
        candidates = routes.get(sid, [])
        uids = set()
        malformed_route = False
        for candidate in candidates:
            value = candidate["unified_security_id"]
            if value in (None, ""):
                malformed_route = True
            else:
                uids.add(_id(value, "bridge unified_security_id"))
        if malformed_route or len(uids) > 1 or (candidates and len(uids) != 1):
            result["identity_status"] = "ambiguous_bridge"
        elif uids:
            uid = next(iter(uids))
            result["unified_security_id"] = uid
            methods = {r.get("bridge_method") for r in candidates}
            confidence = {r.get("bridge_confidence") for r in candidates}
            result["bridge_method"] = next(iter(methods)) if len(methods) == 1 else "mixed"
            result["bridge_confidence"] = next(iter(confidence)) if len(confidence) == 1 else "mixed"
            mapped = link_map.get(uid, [])
            result["identity_status"] = "unmapped_intrinio"
            ids = {_id(r["intrinio_security_id"], "intrinio_security_id")
                   for r in mapped if r["intrinio_security_id"] not in (None, "")}
            if len(ids) > 1 or len(mapped) > 1:
                result["identity_status"] = "ambiguous_intrinio"
            elif len(ids) == 1:
                result["intrinio_security_id"] = next(iter(ids))
                result["identity_review_required"] = mapped[0]["identity_review_required"]
                result["resolution_status"] = mapped[0]["resolution_status"]
                result["identity_status"] = "resolved" if (
                    mapped[0]["identity_review_required"] is False
                    and mapped[0]["resolution_status"] == "resolved") else "needs_review"
        output.append(result)

    targets = {}
    for row in output:
        if row["intrinio_security_id"]:
            targets.setdefault(row["intrinio_security_id"], []).append(row)
    for records in targets.values():
        if len(records) > 1:
            for row in records:
                row["identity_status"] = "target_collision"
    statuses = {}
    for row in output:
        status = row["identity_status"]
        statuses[status] = statuses.get(status, 0) + 1
    provenance = {
        "mode": mode, "membership_end_date": "inclusive",
        "membership_availability": "supplied_reconstructed_date_not_original_publication_evidence",
        "identity_route": "sp_composite1500_source_id_to_unified_id_to_intrinio_id",
        "identity_temporal_evidence": "retrieval_time_mapping_without_validity_intervals",
        "knowledge_time_verified": False,
        "current_security_master_filter_used": False,
        "price_availability_filter_used": False,
    }
    return UniverseSnapshot(
        pa.Table.from_pylist(output, schema=UNIVERSE_SCHEMA),
        pa.Table.from_pylist(excluded, schema=EXCLUDED_SCHEMA), as_of, universe,
        provenance, {"source_rows_at_date": input_rows,
                     "excluded_membership_rows": len(excluded),
                     "constituents": len(output), "identity_status_counts": statuses})


def attach_quotes(snapshot, prices, *, requested_security_ids=None):
    """Left-attach exact-date quotes, preserving unresolved/missing constituents.

    No forward filling, price-based membership selection or return fabrication.
    The caller must load approved data only and retain loading provenance.
    """
    if snapshot.as_of >= "2024-01-01":
        raise DataContractError("Protected 2024+ financial values cannot be attached")
    _columns(prices, ["intrinio_security_id", "trading_date", "close"], "prices")
    requested = None
    if requested_security_ids is not None:
        if isinstance(requested_security_ids, (str, bytes)):
            raise DataContractError("requested_security_ids must be an explicit sequence")
        requested = {_id(value, "requested_security_id") for value in requested_security_ids}
    # Validate date and key metadata before any financial Python conversion.
    seen = set()
    for key in prices.select(["intrinio_security_id", "trading_date"]).to_pylist():
        sid = _id(key["intrinio_security_id"], "intrinio_security_id")
        if iso_date(key["trading_date"], "trading_date") != snapshot.as_of:
            raise DataContractError("Quotes must be dated exactly at the snapshot cutoff")
        if sid in seen:
            raise DataContractError(f"Duplicate quote key: {sid}, {snapshot.as_of}")
        if requested is not None and sid not in requested:
            raise DataContractError("Quote ID falls outside the declared price request")
        seen.add(sid)
    quote_columns = [c for c in ["close", "volume", "adj_close", "adj_volume",
                                "price_quality_status"] if c in prices.column_names]
    quotes = {}
    for row in prices.select(["intrinio_security_id", "trading_date", *quote_columns]).to_pylist():
        sid = _id(row["intrinio_security_id"], "intrinio_security_id")
        if iso_date(row["trading_date"], "trading_date") != snapshot.as_of:
            raise DataContractError("Quotes must be dated exactly at the snapshot cutoff")
        if sid in quotes:
            raise DataContractError(f"Duplicate quote key: {sid}, {snapshot.as_of}")
        quotes[sid] = row
    output, counts = [], {}
    for source in snapshot.table.to_pylist():
        row = dict(source)
        row.update({f"quote_{c}": None for c in quote_columns})
        status = "identity_unresolved"
        if source["identity_status"] == "resolved":
            quote = quotes.get(source["intrinio_security_id"])
            status = "not_requested" if (
                requested is not None and source["intrinio_security_id"] not in requested
            ) else "missing_price"
            if quote is not None:
                row.update({f"quote_{c}": quote[c] for c in quote_columns})
                close = quote["close"]
                valid = isinstance(close, (int, float)) and math.isfinite(close) and close > 0
                status = "ok" if valid else "invalid_price"
                quality = quote.get("price_quality_status")
                if valid and quality not in (None, "", "accepted", "valid", "ok"):
                    status = "source_quality_flag"
        row["quote_status"] = status
        counts[status] = counts.get(status, 0) + 1
        output.append(row)
    extra_schema = [
        pa.field(f"quote_{c}", prices.schema.field(c).type) for c in quote_columns
    ] + [pa.field("quote_status", pa.string())]
    schema = pa.schema([*UNIVERSE_SCHEMA, *extra_schema])
    return QuoteSnapshot(pa.Table.from_pylist(output, schema=schema),
                         {"constituents": len(output), "quote_status_counts": counts})
