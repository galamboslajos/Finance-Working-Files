"""Append only four reviewed real raw bars into provisional adjusted units.

Prices are actual published Alpaca OHLC. Only their denomination is transported
using an earlier accepted Intrinio raw/adjusted pair and a following independent
control. This is explicit retrieval-time unit evidence, not historical PIT or a
complete corporate-action-feed certification. No raw price is interpolated.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import pandas as pd

from .alpaca_repairs import prepare_published_alpaca_repair_declarations
from .ohlc_repairs import OHLC, ORIGINAL_FIELDS, _number, original_bar_sha256
from .universe import DataContractError

POLICY_ID = "dmv2-missing-four-actual-alpaca-raw-bracketed-unit-transport-v1"
MAXIMUM_CLOSE_ABSOLUTE_DIFFERENCE = .05
MAXIMUM_CLOSE_RELATIVE_DIFFERENCE = .005
MAXIMUM_ADJUSTED_PAIR_RESIDUAL = .0002
MISSING_SPECS = (
    ("siblis_ticker_company_51d90c64fe316317136e", "sec_XLvxMP", "2018-09-07", "2018-09-06", "2018-09-11"),
    ("siblis_ticker_company_51d90c64fe316317136e", "sec_XLvxMP", "2018-09-10", "2018-09-06", "2018-09-11"),
    ("siblis_ticker_company_bd1a073c13197ea7e381", "sec_geo2bz", "2018-09-07", "2018-09-06", "2018-09-10"),
    ("siblis_ticker_company_35ec09e133685ccfefae", "sec_zqdekX", "2018-09-07", "2018-09-06", "2018-09-10"),
)


@dataclass(frozen=True)
class SupplementalBarResult:
    prices: pd.DataFrame
    ledger: pd.DataFrame
    evidence_pins: tuple


def _require(value, message):
    if not value:
        raise DataContractError(message)


def _source_row(prices, sid, day):
    rows = prices.loc[(prices.intrinio_security_id == sid) & (prices.trading_date == day)]
    _require(len(rows) == 1, "Required original bracket source bar is absent or duplicated")
    row = rows.iloc[0]
    _require(row.price_quality_status == "accepted" and row.get("frequency") == "daily"
             and not bool(row.get("intraperiod", True)), "Original bracket source quality/frequency fails")
    for flag in ("raw_ohlc_bounds_valid", "adjusted_ohlc_bounds_valid"):
        _require(row.get(flag) == True, "Original bracket source OHLC rejection remains")
    _require(_number(row.get("split_ratio"), "Bracket split ratio") == 1.
             and _number(row.get("dividend"), "Bracket dividend", nonnegative=True) == 0.,
             "Bracket source records an action or lacks explicit zero/unit fields")
    return row


def admit_reviewed_missing_bars(prices, *, evidence_root, projection_evidence,
        source_provenance, native_mapping, review_notes):
    """Append eligible exact missing keys; retain refusals and all original rows.

    Scope is the four fixed FRT/FIVE/WCG keys. Both bracket raw closes must agree
    with actual same-Siblis-ID Alpaca controls under the frozen .05/.005 limits.
    Every original bracket OHLC/adjusted pair must share the earlier close-derived
    multiplier within .0002 adjusted USD. Nonunit splits, nonzero dividends,
    missing controls, or adjustment discontinuities refuse the corresponding
    append. No accepted existing source bar can be replaced by this adapter.
    """
    _require(isinstance(prices, pd.DataFrame) and set((*ORIGINAL_FIELDS,
        "intrinio_security_id", "trading_date", "frequency", "intraperiod", "split_ratio", "dividend"))
        <= set(prices.columns), "Complete original raw/adjusted price and action fields required")
    _require(prices.index.is_unique and not prices.duplicated(["intrinio_security_id", "trading_date"]).any(),
             "Unique original price keys required")
    prepared = prepare_published_alpaca_repair_declarations(prices, evidence_root=evidence_root,
        projection_evidence=projection_evidence, source_provenance=source_provenance,
        native_mapping=native_mapping, review_notes=review_notes)
    source_rows = {(row["source_security_id"], row["trading_date"]): row
                   for row in prepared.identity_and_control_ledger}
    records, appended = [], []
    for source_id, sid, day, left_day, right_day in MISSING_SPECS:
        entry = {"policy_id": POLICY_ID, "source_security_id": source_id, "intrinio_security_id": sid,
            "trading_date": day, "preceding_control_date": left_day, "following_control_date": right_day,
            "status": "REJECTED_UNCHANGED", "reason": None,
            "raw_price_interpolation": False, "original_rows_replaced": 0,
            "unit_transport": "preceding_accepted_adj_close_over_raw_close_validated_by_following_control",
            "no_action_basis": "explicit_zero_dividend_unit_split_on_brackets_and_constant_paired_price_basis",
            "complete_corporate_action_feed_verified": False, "historical_identity_certified": False,
            "knowledge_time_certified": False, "raw_page_hash_verified": False}
        try:
            _require(native_mapping.get(source_id) == sid, "Frozen original source/native mapping differs")
            existing = prices.loc[(prices.intrinio_security_id == sid) & (prices.trading_date == day)]
            _require(existing.empty, "Original key already exists; append-only adapter cannot replace it")
            selected = source_rows.get((source_id, day))
            _require(selected is not None and selected["status"] == "NEW_KEY_SOURCE_ADMISSION_REQUIRED",
                     "Actual published bar is absent or lacks its same-source identity evidence")
            left, right = _source_row(prices, sid, left_day), _source_row(prices, sid, right_day)
            controls = []
            for original, control_day in ((left, left_day), (right, right_day)):
                control = source_rows.get((source_id, control_day))
                _require(control is not None and control["status"] == "CONTROL_EXISTING_ACCEPTED_BAR",
                         "Actual same-source published overlap control is missing")
                raw_close = _number(original["close"], "Original control raw close")
                difference = abs(control["alpaca_raw_ohlc"]["close"] - raw_close)
                _require(difference <= MAXIMUM_CLOSE_ABSOLUTE_DIFFERENCE
                         and difference / raw_close <= MAXIMUM_CLOSE_RELATIVE_DIFFERENCE,
                         "Published bracket control raw close conflicts with original source")
                controls.append(control)
            multiplier = _number(left["adj_close"], "Preceding adjusted close") / _number(left["close"], "Preceding raw close")
            residuals = []
            for original in (left, right):
                for field in OHLC:
                    residuals.append(abs(_number(original[field], "Bracket raw " + field) * multiplier
                                         - _number(original["adj_" + field], "Bracket adjusted " + field)))
            _require(max(residuals) <= MAXIMUM_ADJUSTED_PAIR_RESIDUAL,
                     "Bracketing adjustment basis is discontinuous or exceeds rounding tolerance")
            raw = selected["alpaca_raw_ohlc"]
            row = {field: None for field in prices.columns}
            row.update({"intrinio_security_id": sid, "trading_date": day, "frequency": "daily",
                "intraperiod": False, **raw,
                **{"adj_" + field: raw[field] * multiplier for field in OHLC},
                "raw_ohlc_bounds_valid": True, "adjusted_ohlc_bounds_valid": True,
                "price_quality_status": "accepted",
                # These original-provider action columns were not observed on a
                # missing day. Do not manufacture a supplier dividend or split.
                "factor": None, "split_ratio": None, "dividend": None,
                "supplemental_bar_policy_id": POLICY_ID,
                "supplemental_source_kind": "alternate_provider_actual_raw_bar_provisional_adjustment_unit_transport",
                "supplemental_projection_sha256": projection_evidence["sha256"],
                "supplemental_original_source_status": "missing_quote",
                "supplemental_adjustment_multiplier": multiplier,
                "supplemental_adjustment_transport_provisional": True,
                "source_security_id": source_id})
            # Preserve the caller's existing acceptance contract. Introducing
            # this optional column would fill all original rows with NaN and
            # switch the accounting engine away from their valid source flags.
            if "price_accepted" in prices.columns:
                row["price_accepted"] = True
            for field in ORIGINAL_FIELDS:
                if "original_" + field in prices.columns:
                    row["original_" + field] = "missing_quote" if field == "price_quality_status" else None
            _require(all(isfinite(row["adj_" + field]) and row["adj_" + field] > 0 for field in OHLC),
                     "Transported adjusted OHLC is invalid")
            appended.append(row)
            entry.update({"status": "APPENDED_ACTUAL_RAW_BAR_PROVISIONAL_UNITS", "reason": None,
                "raw_ohlc": dict(raw), "adjusted_ohlc": {field: row["adj_" + field] for field in OHLC},
                "preceding_raw_to_adjusted_multiplier": multiplier,
                "following_raw_to_adjusted_multiplier": float(right["adj_close"]) / float(right["close"]),
                "maximum_bracket_adjusted_residual": max(residuals),
                "preceding_original_bar_sha256": original_bar_sha256(left),
                "following_original_bar_sha256": original_bar_sha256(right),
                "projection_evidence_sha256": projection_evidence["sha256"],
                "range_receipt_sha256": source_provenance["range_receipt"]["sha256"]})
        except DataContractError as exc:
            entry["reason"] = str(exc)
        records.append(entry)
    # Concatenation appends source keys only. It does not sort or rewrite any
    # original row; retained original population/values stay directly testable.
    output = pd.concat([prices.copy(deep=True), pd.DataFrame(appended)], ignore_index=True) if appended else prices.copy(deep=True)
    _require(len(output) == len(prices) + len(appended)
             and not output.duplicated(["intrinio_security_id", "trading_date"]).any(),
             "Supplemental append changed original population or introduced duplicate source keys")
    return SupplementalBarResult(output, pd.DataFrame(records),
                                 (dict(projection_evidence), dict(source_provenance["range_receipt"])))
