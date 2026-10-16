"""Selective raw Alpaca SIP evidence, with no later-year numeric conversion.

JSON number tokens are initially retained as strings. Only an exact requested
symbol/trading-date match in 2018–2019 may reach numeric conversion. This
reader never downloads data, changes selection, or admits a price overlay.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime
import hashlib
import json
from math import isfinite
import re
from zoneinfo import ZoneInfo

from .universe import DataContractError

OHLC_FIELDS = {"open": "o", "high": "h", "low": "l", "close": "c"}
OPTIONAL_FINANCIAL_FIELDS = {"volume": "v", "trade_count": "n", "vwap": "vw"}
NUMBER_TOKEN = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?\Z")
NEW_YORK = ZoneInfo("America/New_York")


def _need(condition, message):
    if not condition:
        raise DataContractError(message)


def _text(value, name):
    _need(isinstance(value, str) and bool(value.strip()), name + " must be explicit")
    return value


def _sha(value, name):
    _need(isinstance(value, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", value),
          name + " requires a canonical SHA256 pin")
    return value


def _requested_day(value):
    _need(isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value),
          "Requested trading date must be ISO YYYY-MM-DD")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise DataContractError("Invalid requested trading date") from exc
    _need(date(2018, 1, 1) <= parsed <= date(2019, 12, 31),
          "Alpaca evidence requests are limited to 2018–2019")
    return value


def _trading_day(timestamp):
    _need(isinstance(timestamp, str), "Alpaca daily timestamp must be a string")
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as exc:
        raise DataContractError("Malformed Alpaca daily timestamp") from exc
    _need(parsed.tzinfo is not None and parsed.utcoffset() is not None,
          "Alpaca daily timestamp requires an explicit timezone")
    local = parsed.astimezone(NEW_YORK)
    _need((local.hour, local.minute, local.second, local.microsecond) == (0, 0, 0, 0),
          "Alpaca 1Day bar must begin at New York midnight")
    return local.date().isoformat()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        _need(key not in result, "Duplicate JSON key in Alpaca source page")
        result[key] = value
    return result


def _reject_constant(value):
    raise DataContractError("Nonstandard JSON numeric token in Alpaca source page")


def _selected_number(token, field, *, allow_zero=False):
    # This function is called only after exact requested-key/date membership.
    _need(isinstance(token, str) and NUMBER_TOKEN.fullmatch(token),
          "Selected Alpaca " + field + " is not a JSON numeric token")
    value = float(token)
    _need(isfinite(value) and value >= 0 and (allow_zero or value > 0),
          "Selected Alpaca " + field + " is nonfinite or invalid")
    return value


@dataclass(frozen=True)
class AlpacaPageSelection:
    records: tuple
    coverage: tuple
    provenance: dict
    summary: dict


@dataclass(frozen=True)
class AlpacaRepairDeclarations:
    repairs: tuple
    identity_and_control_ledger: tuple


def parse_alpaca_raw_page(raw_bytes, requested_keys, *, expected_sha256, provenance,
                          maximum_page_bytes=16 * 1024 * 1024,
                          maximum_selected_records=512, maximum_page_bars=25000):
    """Parse one authenticated uncompressed JSON response without broad numbers.

    `requested_keys` is an explicit tuple/list of `(query_symbol, YYYY-MM-DD)`.
    `provenance` must carry separately reviewed SIP/raw/1Day declarations and
    exact original source pins. It does not establish historical identities.
    Uncompressed response bytes must match the original source-page hash.
    A compressed object's transport pin belongs in the caller's read receipt.
    Supported response bodies are Alpaca's single-symbol bars list plus symbol,
    and its multi-symbol bars dictionary. Wrappers require explicit adaptation.
    """
    _need(isinstance(raw_bytes, bytes), "Actual source response bytes required")
    for value, ceiling, name in ((maximum_page_bytes, 16 * 1024 * 1024, "page bytes"),
            (maximum_selected_records, 512, "selected records"), (maximum_page_bars, 100000, "page bars")):
        _need(type(value) is int and 0 < value <= ceiling, "Explicit bounded " + name + " required")
    _need(0 < len(raw_bytes) <= maximum_page_bytes, "Alpaca page exceeds the declared byte bound")
    _sha(expected_sha256, "Expected raw page")
    actual = "sha256:" + hashlib.sha256(raw_bytes).hexdigest()
    _need(actual == expected_sha256, "Alpaca raw page bytes differ from the original source pin")
    fields = {"feed", "adjustment", "timeframe", "source_uri", "source_generation",
              "source_page_sha256", "source_task_id", "source_request_id", "query_symbol"}
    _need(isinstance(provenance, dict) and set(provenance) == fields,
          "Closed original Alpaca request/source provenance required")
    _need(provenance["feed"] == "sip" and provenance["adjustment"] == "raw"
          and provenance["timeframe"] == "1Day", "Original raw SIP 1Day request required")
    _need(provenance["source_page_sha256"] == actual, "Alpaca provenance page pin differs")
    _need(_text(provenance["source_uri"], "Source URI").startswith("gs://"),
          "Original owned GCS raw source required")
    generation = provenance["source_generation"]
    _need(isinstance(generation, str) and generation.isdigit() and int(generation) > 0,
          "Original raw-object generation required")
    for field in ("source_task_id", "source_request_id", "query_symbol"):
        _text(provenance[field], field)
    _need(isinstance(requested_keys, (tuple, list)) and 0 < len(requested_keys) <= maximum_selected_records,
          "Explicit finite requested symbol/date grid required")
    wanted = set()
    for key in requested_keys:
        _need(isinstance(key, (tuple, list)) and len(key) == 2, "Requested key must be symbol/date")
        symbol, day = key; _text(symbol, "Requested query symbol"); _requested_day(day)
        _need((symbol, day) not in wanted, "Duplicate requested Alpaca key")
        wanted.add((symbol, day))
    try:
        payload = json.loads(raw_bytes, parse_float=str, parse_int=str,
                             parse_constant=_reject_constant, object_pairs_hook=_unique_object)
    except DataContractError:
        raise
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise DataContractError("Invalid Alpaca source JSON") from exc
    _need(isinstance(payload, dict) and "bars" in payload, "Unknown Alpaca response envelope")
    bars = payload["bars"]
    streams = []
    if isinstance(bars, list):
        symbol = payload.get("symbol", provenance["query_symbol"])
        _need(symbol == provenance["query_symbol"], "Response symbol differs from original requested symbol")
        streams.append((symbol, bars))
    elif isinstance(bars, dict):
        _need(len(bars) <= 256, "Alpaca page symbol bound exceeded")
        for symbol, items in bars.items():
            _text(symbol, "Response query symbol")
            _need(isinstance(items, list), "Alpaca symbol bars must be a list")
            streams.append((symbol, items))
    else:
        raise DataContractError("Unknown Alpaca bars representation")
    _need(sum(len(items) for _, items in streams) <= maximum_page_bars,
          "Alpaca page metadata row bound exceeded")
    selected, counts, seen = [], Counter(), set()
    for symbol, items in streams:
        for bar in items:
            counts["bar_metadata_rows"] += 1
            _need(isinstance(bar, dict), "Alpaca bar must be an object")
            day = _trading_day(bar.get("t"))
            if not "2018-01-01" <= day <= "2019-12-31":
                counts["outside_2018_2019_metadata_rows_skipped"] += 1
                continue
            if (symbol, day) not in wanted:
                counts["unrequested_metadata_rows_skipped"] += 1
                continue
            _need(symbol == provenance["query_symbol"], "Selected symbol lacks its original task provenance")
            _need((symbol, day) not in seen, "Duplicate selected Alpaca source key")
            seen.add((symbol, day))
            raw_strings = {field: bar.get(source_field) for field, source_field in
                           {**OHLC_FIELDS, **OPTIONAL_FINANCIAL_FIELDS}.items()}
            raw = {field: _selected_number(raw_strings[field], field) for field in OHLC_FIELDS}
            _need(raw["low"] <= min(raw["open"], raw["close"])
                  <= max(raw["open"], raw["close"]) <= raw["high"], "Selected Alpaca raw OHLC bounds fail")
            optional = {}
            for field in OPTIONAL_FINANCIAL_FIELDS:
                value = raw_strings[field]
                if value is not None:
                    optional[field] = _selected_number(value, field, allow_zero=field != "vwap")
            counts["numeric_records_converted"] += 1
            _need(len(selected) < maximum_selected_records, "Selected Alpaca numerical bound exceeded")
            selected.append({"query_symbol": symbol, "trading_date": day, "source_timestamp": bar["t"],
                "raw_ohlc": raw, "optional_values": optional, "original_source_value_strings": raw_strings,
                "source_page_sha256": actual, "source_uri": provenance["source_uri"],
                "source_generation": generation, "source_task_id": provenance["source_task_id"],
                "source_request_id": provenance["source_request_id"], "feed": "sip", "adjustment": "raw",
                "timeframe": "1Day", "source_admitted": False, "historical_identity_certified": False})
    coverage = tuple({"query_symbol": symbol, "trading_date": day,
                      "status": "SOURCE_ROW_FOUND" if (symbol, day) in seen else "NOT_IN_THIS_RAW_PAGE"}
                     for symbol, day in sorted(wanted))
    summary = {**dict(counts), "requested_records": len(wanted), "selected_records": len(selected),
        "json_number_decode": "all_number_tokens_as_strings_before_requested_key_filter",
        "numeric_2020_plus_conversions": 0, "source_admitted": False,
        "historical_identity_certified": False, "source_page_bytes": len(raw_bytes), "source_page_sha256": actual}
    return AlpacaPageSelection(tuple(sorted(selected, key=lambda row: (row["query_symbol"], row["trading_date"]))),
                               coverage, dict(provenance), summary)


def prepare_alpaca_repair_declarations(prices, records, *, source_identity_rows, native_mapping,
                                       evidence_by_page_hash, identity_evidence, review_notes):
    """Prepare existing-rejected-bar declarations using exact historical keys.

    Source rows must carry the same Siblis ID as the previously admitted native
    mapping, on the actual requested date. This remains provisional identity
    evidence. Accepted original rows are controls; absent original rows require
    a separate source-admission path and are never inserted here. The caller
    persists this complete identity/control ledger, and passes `repairs` through
    `apply_reviewed_ohlc_repairs` under its explicitly frozen numerical policy.
    """
    from .ohlc_repairs import original_bar_sha256
    import pandas as pd

    _text(review_notes, "Explicit source/identity/unit review")
    _need(isinstance(native_mapping, dict) and bool(native_mapping), "Original admitted source/native mapping required")
    _need(isinstance(records, (list, tuple)) and len(records) <= 512,
          "Bounded selected actual Alpaca records required")
    _need(isinstance(source_identity_rows, (list, tuple)) and len(source_identity_rows) <= 2048,
          "Bounded selected curated source-identity metadata required")
    _need(isinstance(evidence_by_page_hash, dict) and isinstance(identity_evidence, dict),
          "Explicit raw-page and identity evidence pins required")
    _need({"intrinio_security_id", "trading_date", "price_quality_status"} <= set(prices.columns),
          "Original price keys and rejection states required")
    _need(not prices.duplicated(["intrinio_security_id", "trading_date"]).any() and prices.index.is_unique,
          "Unique original price keys required")
    for day in prices.trading_date.unique():
        _requested_day(day)
    repairs, ledger, seen = [], [], set()
    for record in records:
        _need(isinstance(record, dict), "Actual selected Alpaca record required")
        symbol, day = record.get("query_symbol"), record.get("trading_date")
        _text(symbol, "Selected symbol"); _requested_day(day)
        _need((symbol, day) not in seen, "Duplicate actual selected Alpaca key")
        seen.add((symbol, day))
        _need(record.get("feed") == "sip" and record.get("adjustment") == "raw"
              and record.get("timeframe") == "1Day", "Raw SIP daily record required")
        page_hash = _sha(record.get("source_page_sha256"), "Selected source page")
        candidates = [row for row in source_identity_rows if isinstance(row, dict)
            and row.get("query_symbol") == symbol and row.get("trading_date") == day
            and row.get("security_id") in native_mapping]
        # Curation can duplicate a symbol/date across unresolved source identities.
        # Only exactly one originally admitted source route can own this repair.
        _need(len(candidates) == 1, "Alpaca source/date has no unique admitted original Siblis identity")
        identity = candidates[0]
        metadata_hash = identity.get("source_page_hash", "")
        if isinstance(metadata_hash, str) and not metadata_hash.startswith("sha256:"):
            metadata_hash = "sha256:" + metadata_hash
        _need(metadata_hash == page_hash and identity.get("adjustment") == "raw"
              and identity.get("source_task_id") == record.get("source_task_id")
              and identity.get("source_request_id") == record.get("source_request_id"),
              "Alpaca raw page differs from curated date/source provenance")
        source_id = identity["security_id"]
        native_id = _text(native_mapping[source_id], "Original admitted native ID")
        match = prices.loc[(prices.intrinio_security_id == native_id) & (prices.trading_date == day)]
        _need(len(match) <= 1, "Duplicate existing original source key")
        status = ("NEW_KEY_SOURCE_ADMISSION_REQUIRED" if match.empty
                  else "CONTROL_EXISTING_ACCEPTED_BAR" if match.iloc[0].price_quality_status == "accepted"
                  else "PREPARED_REJECTED_OHLC_BAR" if match.iloc[0].price_quality_status
                       in ("provider_ohlc_anomaly", "provider_incomplete_ohlc")
                  else "OTHER_ORIGINAL_SOURCE_STATE_REQUIRES_REVIEW")
        def original_ohlc():
            if match.empty:
                return None
            result = {}
            for field in OHLC_FIELDS:
                value = match.iloc[0].get(field)
                result[field] = None if pd.isna(value) else float(value)
            return result

        ledger_row = {"query_symbol": symbol, "trading_date": day, "source_security_id": source_id,
            "alpaca_security_id": source_id, "intrinio_security_id": native_id, "status": status,
            "source_page_sha256": page_hash, "source_task_id": record.get("source_task_id"),
            "source_request_id": record.get("source_request_id"), "historical_identity_certified": False,
            "source_admitted": False, "original_ohlc": original_ohlc(),
            "alpaca_raw_ohlc": dict(record["raw_ohlc"])}
        ledger.append(ledger_row)
        if status != "PREPARED_REJECTED_OHLC_BAR":
            continue
        _need(page_hash in evidence_by_page_hash, "Actual replacement page evidence pin missing")
        evidence = evidence_by_page_hash[page_hash]
        _need(isinstance(evidence, dict) and evidence.get("sha256") == page_hash,
              "Actual replacement artifact differs from selected page")
        original = match.iloc[0]
        repairs.append({"repair_id": "alpaca-sip-raw-" + native_id + "-" + day,
            "intrinio_security_id": native_id, "trading_date": day,
            "original_bar_sha256": original_bar_sha256(original), "raw_ohlc": dict(record["raw_ohlc"]),
            "source_kind": "alternate_provider_raw_bar", "provider": "Alpaca SIP",
            "evidence": dict(evidence), "identity_review": {
                "status": "REVIEWED_PROVISIONAL_SAME_SECURITY", "source_identifier": source_id + ":" + symbol,
                "native_security_id": native_id, "valid_from": day, "valid_to": day, "currency": "USD",
                "price_basis": "unadjusted_USD_per_original_share", "review_notes": review_notes,
                "evidence": dict(identity_evidence)}})
    return AlpacaRepairDeclarations(tuple(repairs), tuple(ledger))


def prepare_published_alpaca_repair_declarations(prices, *, evidence_root, projection_evidence,
        source_provenance, native_mapping, review_notes):
    """Prepare repairs from a pinned, physically bounded published projection.

    This is a different evidence route from raw JSON. The projection's actual
    bytes and range receipt are authenticated; retained raw-page lineage hashes
    are not claimed verified. Original table generation/size and the publication
    manifest pin are explicit source declarations. Full table SHA verification
    is not inferred from partial range reads. Any transient mixed-row-group
    decoding performed by the source exporter remains disclosed separately.
    """
    from pathlib import Path
    import pandas as pd
    import pyarrow.parquet as pq
    from .ohlc_repairs import _verify_evidence, original_bar_sha256

    _text(review_notes, "Explicit published-source/identity/unit review")
    _need(isinstance(native_mapping, dict) and bool(native_mapping), "Original admitted source/native mapping required")
    _need(isinstance(source_provenance, dict), "Published projection source provenance required")
    expected = {"publication_manifest_uri", "publication_manifest_generation", "publication_manifest_canonical_sha256",
        "publication_manifest_canonical_pin_verified", "source_table_uri", "source_table_generation",
        "source_table_size_bytes", "source_table_sha256_declared", "whole_source_table_sha256_reread",
        "raw_page_hash_verified", "range_receipt", "projection_policy", "mixed_row_group_numeric_decoding_possible",
        "retained_manifest_pin_verified", "projected_compressed_bytes", "maximum_projected_compressed_bytes"}
    _need(set(source_provenance) == expected, "Closed published projection provenance required")
    _need(source_provenance["publication_manifest_canonical_pin_verified"] is True,
          "Verified publication manifest canonical pin required")
    _need(source_provenance["whole_source_table_sha256_reread"] is False
          and source_provenance["raw_page_hash_verified"] is False
          and source_provenance["retained_manifest_pin_verified"] is False,
          "Do not promote partial range reads, retained manifest or raw-page lineage to hash verification")
    _need(source_provenance["projection_policy"] == "date_and_id_filter_in_Arrow_before_Python_conversion"
          and source_provenance["mixed_row_group_numeric_decoding_possible"] is True,
          "Explicit bounded Arrow output/analysis guard disclosure required")
    for field in ("publication_manifest_uri", "source_table_uri"):
        _need(_text(source_provenance[field], field).startswith("gs://"), "Owned published GCS source required")
    for field in ("publication_manifest_generation", "source_table_generation"):
        value = source_provenance[field]
        _need(isinstance(value, str) and value.isdigit() and int(value) > 0, "Pinned original source generation required")
    for field in ("publication_manifest_canonical_sha256", "source_table_sha256_declared"):
        _sha(source_provenance[field], field)
    size = source_provenance["source_table_size_bytes"]
    _need(type(size) is int and size > 0, "Pinned original source table size required")
    actual_bytes = source_provenance["projected_compressed_bytes"]
    maximum_bytes = source_provenance["maximum_projected_compressed_bytes"]
    _need(type(actual_bytes) is int and type(maximum_bytes) is int
          and 0 < actual_bytes <= maximum_bytes <= 64 * 1024 * 1024,
          "Published source projection exceeds the declared 64MiB ceiling")
    root = Path(evidence_root).resolve()
    projection_pin = _verify_evidence(root, projection_evidence)
    receipt_pin = _verify_evidence(root, source_provenance["range_receipt"])
    _need(projection_pin["source_uri"] == source_provenance["source_table_uri"],
          "Selected projection differs from declared published source table")
    file = pq.ParquetFile(root / projection_pin["path"])
    metadata_columns = ["security_id", "trading_date", "query_symbol", "adjustment", "feed", "timeframe",
                        "source_task_id", "source_page_hash", "source_request_id"]
    required = {*metadata_columns, *OHLC_FIELDS}
    _need(required <= set(file.schema_arrow.names) and 0 < file.metadata.num_rows <= 512,
          "Finite selected published quote artifact with full identity/OHLC fields required")
    # Physical projection artifact must itself contain only authorized dates.
    # No OHLC chunk is decoded until this independent metadata check passes.
    metadata = file.read(columns=metadata_columns).to_pylist()
    original_keys, selected_keys = set(), set()
    for row in metadata:
        _requested_day(row["trading_date"])
        _text(row["security_id"], "Published original source ID")
        _text(row["query_symbol"], "Published query symbol")
        _need(row["adjustment"] == "raw" and row["feed"] == "sip" and row["timeframe"] == "1Day",
              "Published quote requires original raw SIP daily declarations")
        key = row["security_id"], row["trading_date"]
        _need(key not in selected_keys, "Duplicate selected published original source/date key")
        selected_keys.add(key)
    _need({"intrinio_security_id", "trading_date", "price_quality_status"} <= set(prices.columns)
          and not prices.duplicated(["intrinio_security_id", "trading_date"]).any() and prices.index.is_unique,
          "Unique original price keys/states required")
    for day in prices.trading_date.unique():
        _requested_day(day)
    values = file.read(columns=list(OHLC_FIELDS)).to_pylist()
    repairs, ledger = [], []
    for row, raw in zip(metadata, values, strict=True):
        _need(all(isinstance(value, (float, int)) and not isinstance(value, bool)
                  and isfinite(value) and value > 0 for value in raw.values()),
              "Published replacement raw OHLC must be finite and positive")
        raw = {field: float(raw[field]) for field in OHLC_FIELDS}
        _need(raw["low"] <= min(raw["open"], raw["close"])
              <= max(raw["open"], raw["close"]) <= raw["high"], "Published replacement raw OHLC bounds fail")
        source_id, day = row["security_id"], row["trading_date"]
        # Unadmitted alternative source identities are retained as unresolved
        # metadata; they never override the original provisional mapping.
        native_id = native_mapping.get(source_id)
        original = prices.iloc[:0] if native_id is None else prices.loc[
            (prices.intrinio_security_id == native_id) & (prices.trading_date == day)]
        status = ("NO_ORIGINAL_ADMITTED_SOURCE_ID" if native_id is None
            else "NEW_KEY_SOURCE_ADMISSION_REQUIRED" if original.empty
            else "CONTROL_EXISTING_ACCEPTED_BAR" if original.iloc[0].price_quality_status == "accepted"
            else "PREPARED_REJECTED_OHLC_BAR" if original.iloc[0].price_quality_status
                in ("provider_ohlc_anomaly", "provider_incomplete_ohlc")
            else "OTHER_ORIGINAL_SOURCE_STATE_REQUIRES_REVIEW")
        before, differences = {}, {}
        for field in OHLC_FIELDS:
            value = None if original.empty else original.iloc[0].get(field)
            value = None if value is None or pd.isna(value) else float(value)
            before[field] = value
            differences[field] = None if value is None else raw[field] - value
        ledger.append({"query_symbol": row["query_symbol"], "trading_date": day,
            "source_security_id": source_id, "alpaca_security_id": source_id, "intrinio_security_id": native_id,
            "status": status, "original_ohlc": before, "alpaca_raw_ohlc": raw,
            "raw_price_differences": differences, "source_page_hash_retained_lineage": row["source_page_hash"],
            "source_task_id": row["source_task_id"], "source_request_id": row["source_request_id"],
            "projection_evidence_sha256": projection_pin["sha256"], "range_receipt_sha256": receipt_pin["sha256"],
            "source_table_generation": source_provenance["source_table_generation"],
            "raw_page_hash_verified": False, "whole_source_table_sha256_reread": False,
            "retained_manifest_pin_verified": False, "historical_identity_certified": False, "source_admitted": False})
        if status != "PREPARED_REJECTED_OHLC_BAR":
            continue
        native_id = _text(native_id, "Original admitted native ID")
        key = native_id, day
        _need(key not in original_keys, "Published rows collide on original native source/date")
        original_keys.add(key)
        repairs.append({"repair_id": "published-alpaca-sip-raw-" + native_id + "-" + day,
            "intrinio_security_id": native_id, "trading_date": day,
            "original_bar_sha256": original_bar_sha256(original.iloc[0]), "raw_ohlc": dict(raw),
            "source_kind": "alternate_provider_raw_bar", "provider": "Alpaca SIP published projection",
            "evidence": dict(projection_pin), "identity_review": {
                "status": "REVIEWED_PROVISIONAL_SAME_SECURITY", "source_identifier": source_id + ":" + row["query_symbol"],
                "native_security_id": native_id, "valid_from": day, "valid_to": day, "currency": "USD",
                "price_basis": "unadjusted_USD_per_original_share", "review_notes": review_notes,
                "evidence": dict(projection_pin)}})
    return AlpacaRepairDeclarations(tuple(repairs), tuple(ledger))
