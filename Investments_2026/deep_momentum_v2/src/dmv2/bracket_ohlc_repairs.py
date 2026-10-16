"""Separate three-key full-SIP-bar amendment; no frozen close-gate changes.

The entire original target bar was already rejected. Replace it only with the
actual pinned same-source SIP daily bar, in units derived from a PRECEDING valid
raw/adjusted pair. A following valid overlap bar checks units without supplying
the multiplier. This is retrieval-time historical reconstruction, not certified
knowledge-time data, executions, or a complete corporate-action feed.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from math import isfinite
from pathlib import Path

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .ohlc_repairs import OHLC, ORIGINAL_FIELDS, original_bar_sha256, _verify_evidence, _number
from .universe import DataContractError

POLICY_ID = "dmv2-three-rejected-20190812-full-sip-bar-preceding-units-v1"
MAX_CONTROL_CLOSE_ABSOLUTE_DIFFERENCE = .05
MAX_CONTROL_CLOSE_RELATIVE_DIFFERENCE = .005
MAX_ADJUSTED_PAIR_RESIDUAL = .0002
TARGET_DATE = "2019-08-12"
PRECEDING_DATE = "2019-08-09"
FOLLOWING_DATE = "2019-08-13"
CASES = (
    ("ACC", "siblis_ticker_company_49bd327980365e61c508", "sec_nzJbNz"),
    ("K", "siblis_isin_937a64ed8a9900b47f25", "sec_ge8b0y"),
    ("SO", "siblis_isin_435ee7b1063381caec4f", "sec_zqkWkg"),
)
SOURCE_URI = "gs://quant-research-trusty-agility-492109-v0/dataset-platform/curated/equity_us/sp_composite1500_market_daily/v1/tables/daily_bars/part-00000.parquet"
PUBLICATION_URI = "gs://quant-research-trusty-agility-492109-v0/dataset-platform/curated/equity_us/sp_composite1500_market_daily/v1/_releases/professional-reference-20260807-r1/MANIFEST.json"
PUBLICATION_GENERATION = "1786102788412943"
PUBLICATION_SHA256 = "sha256:67119e4939e031cbe2c4aad752b8f99a7a72e9bc1ca43595e3c1efc5b40d7f03"
TABLE_GENERATION = "1783344954247444"
TABLE_SIZE = 187908505
TABLE_DECLARED_SHA256 = "sha256:4a4919d48128d5c573ddbca64649d8cc78783f7bbb818219a3ad5f34728e38a0"


@dataclass(frozen=True)
class BracketedOhlcRepairResult:
    prices: pd.DataFrame
    ledger: pd.DataFrame
    evidence_pins: tuple


def _need(value, message):
    if not value:
        raise DataContractError(message)


def _false_boolean(value):
    return isinstance(value, (bool, np.bool_)) and not value


def _raw_valid(raw):
    values = {field: _number(raw[field], "SIP raw " + field) for field in OHLC}
    _need(values["low"] <= min(values["open"], values["close"])
          <= max(values["open"], values["close"]) <= values["high"], "SIP raw OHLC bounds fail")
    return values


def _valid_source_control(row, expected_hash, *, following):
    _need(original_bar_sha256(row) == expected_hash, "Original valid control differs from reviewed pin")
    _need(row["price_quality_status"] == "accepted" and row["frequency"] == "daily"
          and _false_boolean(row["intraperiod"]) and row["raw_ohlc_bounds_valid"] == True
          and row["adjusted_ohlc_bounds_valid"] == True, "Original valid control quality/frequency fails")
    _raw_valid(row)
    _raw_valid({field: row["adj_" + field] for field in OHLC})
    _need(_number(row["split_ratio"], "Control split ratio") == 1., "Nonunit control split requires separate action treatment")
    dividend = _number(row["dividend"], "Control dividend", nonnegative=True)
    # A preceding ex-dividend session is already reflected in that valid paired
    # quote. A following action could change units over the bracket and refuses.
    _need(not following or dividend == 0., "Following action changes the checked unit interval")
    return dividend


def _selected_sip(root, manifest):
    evidence = _verify_evidence(root, manifest["projection_evidence"])
    provenance = manifest["source_provenance"]
    _need(provenance["source_table_uri"] == SOURCE_URI
          and provenance["source_table_generation"] == TABLE_GENERATION
          and provenance["source_table_size_bytes"] == TABLE_SIZE
          and provenance["source_table_sha256_declared"] == TABLE_DECLARED_SHA256
          and provenance["publication_manifest_uri"] == PUBLICATION_URI
          and provenance["publication_manifest_generation"] == PUBLICATION_GENERATION
          and provenance["publication_manifest_canonical_sha256"] == PUBLICATION_SHA256
          and provenance["publication_manifest_canonical_pin_verified"] is True,
          "The separately reviewed publication/source pins differ")
    _need(evidence["source_uri"] == SOURCE_URI
          and provenance["whole_source_table_sha256_reread"] is False
          and provenance["raw_page_hash_verified"] is False
          and provenance["retained_manifest_pin_verified"] is False
          and provenance["mixed_row_group_numeric_decoding_possible"] is True
          and provenance["projection_policy"] == "date_and_id_filter_in_Arrow_before_Python_conversion",
          "Partial source/projection limitations must stay explicit")
    actual_bytes, maximum_bytes = provenance["projected_compressed_bytes"], provenance["maximum_projected_compressed_bytes"]
    _need(type(actual_bytes) is int and type(maximum_bytes) is int
          and 0 < actual_bytes <= maximum_bytes <= 64 * 1024 * 1024, "Original source projection bound differs")
    receipt_pin = _verify_evidence(root, provenance["range_receipt"])
    receipt = json.loads((root / receipt_pin["path"]).read_text())
    obj = receipt["object"]
    _need(obj["uri"] == SOURCE_URI and str(obj["generation"]) == TABLE_GENERATION
          and obj["size_bytes"] == TABLE_SIZE and obj["sha256"] == TABLE_DECLARED_SHA256
          and receipt["source_publication_manifest_sha256"] == PUBLICATION_SHA256
          and receipt["selected_artifact_sha256"] == evidence["sha256"].removeprefix("sha256:")
          and receipt["only_2018_2019_financial_rows_returned_or_used"] is True
          and receipt["excluded_financial_rows_converted_to_python_or_returned"] is False
          and receipt["whole_source_table_sha256_reread"] is False
          and receipt["raw_page_hashes_verified"] is False,
          "Original source receipt does not bind the selected projection and exclusions")
    file = pq.ParquetFile(root / evidence["path"])
    fields = ["security_id", "trading_date", "query_symbol", "adjustment", "feed", "timeframe",
              "source_task_id", "source_page_hash", "source_request_id"]
    _need(set((*fields, *OHLC)) <= set(file.schema_arrow.names) and 0 < file.metadata.num_rows <= 512,
          "Small published projection with original metadata and raw OHLC required")
    table = file.read(columns=fields)
    # Only identifiers are inspected before the date guard. No financial chunk
    # is read if even one projected date enters the reserved later period.
    _need(all(isinstance(day, str) and "2018-01-01" <= day <= "2019-12-31"
              for day in table["trading_date"].to_pylist()), "Published projection leaves authorized 2018–2019 dates")
    wanted = {(source, day) for _, source, _ in CASES
              for day in (PRECEDING_DATE, TARGET_DATE, FOLLOWING_DATE)}
    joined = pc.binary_join_element_wise(pc.cast(table["security_id"], pa.string()),
                                        pc.cast(table["trading_date"], pa.string()), "|")
    key_values = pa.array([source + "|" + day for source, day in sorted(wanted)])
    indices = pc.indices_nonzero(pc.is_in(joined, value_set=key_values))
    selected = table.take(indices)
    metadata = selected.to_pylist()
    _need(len(metadata) == 9 and len({(row["security_id"], row["trading_date"]) for row in metadata}) == 9,
          "Every original source identity/date needs one complete nine-key SIP bracket")
    raw_values = file.read(columns=list(OHLC)).take(indices).to_pylist()
    rows = {}
    symbol_by_source = {source: symbol for symbol, source, _ in CASES}
    for meta, raw in zip(metadata, raw_values, strict=True):
        _need(meta["query_symbol"] == symbol_by_source[meta["security_id"]]
              and meta["feed"] == "sip" and meta["adjustment"] == "raw" and meta["timeframe"] == "1Day",
              "SIP query symbol, feed, timeframe or raw-adjustment evidence differs")
        for field in ("source_task_id", "source_page_hash", "source_request_id"):
            _need(isinstance(meta[field], str) and bool(meta[field]), "Original retained lineage metadata required")
        rows[meta["security_id"], meta["trading_date"]] = {**meta, "raw_ohlc": _raw_valid(raw)}
    return rows, (evidence, receipt_pin)


def apply_bracketed_sip_ohlc_repairs(prices, *, evidence_root, manifest):
    """Apply only the separately reviewed ACC/K/SO rejected keys.

    Old rejected closes are preserved as evidence; they are deliberately not
    accepted or used to derive corrected adjustment units. The old close-match
    worker/configuration is unchanged. Original selections, rows, targets,
    predictors and fitted models are outside this worker.
    Source volume and adjusted volume are deliberately left unchanged: this
    amendment admits full OHLC only, and does not reconcile trade-volume feeds.
    """
    _need(isinstance(prices, pd.DataFrame) and prices.index.is_unique, "Unique original price frame required")
    required = {"intrinio_security_id", "trading_date", "frequency", "intraperiod", "split_ratio",
                "dividend", *ORIGINAL_FIELDS}
    _need(required <= set(prices.columns) and not prices.duplicated(["intrinio_security_id", "trading_date"]).any(),
          "Original full raw/adjusted price/action fields and unique keys required")
    _need(all(isinstance(day, str) and "2018-01-01" <= day <= "2019-12-31"
              for day in prices.trading_date.unique()), "Original prices leave authorized 2018–2019 dates")
    _need(isinstance(manifest, dict)
          and set(manifest) == {"schema", "policy_id", "projection_evidence", "source_provenance", "cases", "review_notes"}
          and manifest["schema"] == "dmv2-three-bracketed-sip-ohlc/v1" and manifest["policy_id"] == POLICY_ID
          and isinstance(manifest["review_notes"], str) and bool(manifest["review_notes"].strip()),
          "Closed separate three-key amendment manifest required")
    _need(isinstance(manifest["cases"], list) and len(manifest["cases"]) == 3,
          "Exactly the three reviewed original rejected keys required")
    case_fields = {"query_symbol", "source_security_id", "intrinio_security_id", "trading_date",
                   "preceding_date", "following_date", "original_target_bar_sha256",
                   "preceding_control_bar_sha256", "following_control_bar_sha256"}
    specs = {}
    for case in manifest["cases"]:
        _need(isinstance(case, dict) and set(case) == case_fields, "Closed reviewed case pins required")
        signature = (case["query_symbol"], case["source_security_id"], case["intrinio_security_id"])
        _need(signature in CASES and case["trading_date"] == TARGET_DATE
              and case["preceding_date"] == PRECEDING_DATE and case["following_date"] == FOLLOWING_DATE,
              "Amendment cannot expand to other stocks or dates")
        _need(signature not in specs, "Duplicate amendment case")
        specs[signature] = case
    _need(set(specs) == set(CASES), "All three exact source/native identities required")
    root = Path(evidence_root).resolve()
    sip, pins = _selected_sip(root, manifest)
    output = prices.copy(deep=True)
    for field in ORIGINAL_FIELDS:
        if "original_" + field not in output.columns:
            output["original_" + field] = output[field]
    for name in ("bracket_ohlc_repair_policy_id", "bracket_ohlc_repair_source_sha256"):
        if name not in output.columns:
            output[name] = None
    ledger = []
    for signature in CASES:
        symbol, source_id, sid = signature
        case = specs[signature]
        original_rows = {}
        for day in (PRECEDING_DATE, TARGET_DATE, FOLLOWING_DATE):
            matched = prices.loc[(prices.intrinio_security_id == sid) & (prices.trading_date == day)]
            _need(len(matched) == 1, "Original target/control key is absent")
            original_rows[day] = matched.iloc[0]
        target = original_rows[TARGET_DATE]
        _need(original_bar_sha256(target) == case["original_target_bar_sha256"],
              "Original rejected target differs from reviewed source pin")
        _need(target["price_quality_status"] == "provider_ohlc_anomaly"
              and target["frequency"] == "daily" and _false_boolean(target["intraperiod"])
              and target["raw_ohlc_bounds_valid"] == False and target["adjusted_ohlc_bounds_valid"] == False,
              "This amendment can replace only its already-rejected complete OHLC anomaly")
        _need(_number(target["split_ratio"], "Target split ratio") == 1.
              and _number(target["dividend"], "Target dividend", nonnegative=True) == 0.,
              "Target action requires a separately reviewed unit/event treatment")
        preceding, following = original_rows[PRECEDING_DATE], original_rows[FOLLOWING_DATE]
        previous_dividend = _valid_source_control(preceding, case["preceding_control_bar_sha256"], following=False)
        next_dividend = _valid_source_control(following, case["following_control_bar_sha256"], following=True)
        # This is the sole source of corrected denomination. Following values
        # appear only in pass/fail QA and the provenance ledger.
        multiplier = _number(preceding["adj_close"], "Preceding adjusted close") / _number(preceding["close"], "Preceding raw close")
        residuals, control_close_differences = [], []
        for day, original in ((PRECEDING_DATE, preceding), (FOLLOWING_DATE, following)):
            reference = sip[source_id, day]["raw_ohlc"]
            old_close = _number(original["close"], "Valid control raw close")
            difference = abs(reference["close"] - old_close)
            _need(difference <= MAX_CONTROL_CLOSE_ABSOLUTE_DIFFERENCE
                  and difference / old_close <= MAX_CONTROL_CLOSE_RELATIVE_DIFFERENCE,
                  "Independent valid-control close conflicts under the unchanged control gate")
            control_close_differences.append(difference)
            for field in OHLC:
                raw_value = _number(original[field], "Valid control raw " + field)
                adjusted_value = _number(original["adj_" + field], "Valid control adjusted " + field)
                residuals.append(abs(adjusted_value - raw_value * multiplier))
                # A complete independent raw daily control must reproduce all
                # valid raw fields, not merely share a possibly ambiguous close.
                _need(reference[field] == raw_value, "Independent valid-control full raw OHLC disagrees")
        _need(max(residuals) <= MAX_ADJUSTED_PAIR_RESIDUAL,
              "Preceding adjustment units fail the following valid-control check")
        replacement = sip[source_id, TARGET_DATE]["raw_ohlc"]
        old_target_close = _number(target["close"], "Rejected target raw close")
        old_target_multiplier = _number(target["adj_close"], "Rejected target adjusted close") / old_target_close
        old_target_residual = max(abs(_number(target[field], "Rejected target raw " + field) * old_target_multiplier
                                     - _number(target["adj_" + field], "Rejected target adjusted " + field))
                                  for field in OHLC)
        adjusted = {"adj_" + field: replacement[field] * multiplier for field in OHLC}
        _need(all(isfinite(value) and value > 0 for value in adjusted.values()), "Corrected adjusted bar is invalid")
        target_index = target.name
        for field, value in {**replacement, **adjusted}.items():
            output.at[target_index, field] = value
        output.at[target_index, "price_quality_status"] = "accepted"
        output.at[target_index, "raw_ohlc_bounds_valid"] = True
        output.at[target_index, "adjusted_ohlc_bounds_valid"] = True
        if "price_accepted" in output.columns:
            output.at[target_index, "price_accepted"] = True
        output.at[target_index, "bracket_ohlc_repair_policy_id"] = POLICY_ID
        output.at[target_index, "bracket_ohlc_repair_source_sha256"] = pins[0]["sha256"]
        ledger.append({"policy_id": POLICY_ID, "query_symbol": symbol, "source_security_id": source_id,
            "intrinio_security_id": sid, "trading_date": TARGET_DATE,
            "status": "REPLACED_ALREADY_REJECTED_FULL_BAR_WITH_ACTUAL_SIP_RAW",
            "original_target_bar_sha256": case["original_target_bar_sha256"],
            "preceding_control_bar_sha256": case["preceding_control_bar_sha256"],
            "following_control_bar_sha256": case["following_control_bar_sha256"],
            "original_raw_ohlc": {field: float(target[field]) for field in OHLC},
            "original_adjusted_ohlc": {field: float(target["adj_" + field]) for field in OHLC},
            "replacement_raw_ohlc": dict(replacement), "replacement_adjusted_ohlc": adjusted,
            "rejected_target_original_pair_multiplier": old_target_multiplier,
            "rejected_target_maximum_internal_paired_residual": old_target_residual,
            "target_raw_close_absolute_difference": abs(replacement["close"] - old_target_close),
            "target_raw_close_relative_difference": abs(replacement["close"] - old_target_close) / old_target_close,
            "adjustment_multiplier": multiplier, "multiplier_source_date": PRECEDING_DATE,
            "following_control_date_for_qa_only": FOLLOWING_DATE,
            "maximum_adjusted_pair_residual": max(residuals),
            "maximum_independent_control_close_difference": max(control_close_differences),
            "preceding_dividend_already_in_paired_units": previous_dividend,
            "target_dividend": 0., "following_dividend": next_dividend,
            "source_projection_sha256": pins[0]["sha256"], "range_receipt_sha256": pins[1]["sha256"],
            "source_table_generation": TABLE_GENERATION, "publication_manifest_sha256": PUBLICATION_SHA256,
            "retained_target_source_page_hash": sip[source_id,TARGET_DATE]["source_page_hash"],
            "raw_page_hash_verified": False, "whole_source_table_sha256_reread": False,
            "historical_reconstruction": True, "knowledge_time_certified": False,
            "complete_corporate_action_feed_certified": False,
            "old_close_compatibility_policy_modified": False, "raw_prices_interpolated": False,
            "volume_fields_reconciled": False, "original_volume_and_adjusted_volume_retained": True})
    _need(len(output) == len(prices), "Full-bar replacement changed the original population")
    return BracketedOhlcRepairResult(output, pd.DataFrame(ledger), pins)

