"""Explicit, provenance-pinned OHLC overlays; original source rows stay intact.

This module fetches nothing and discovers no replacements. A caller must supply
reviewed actual raw-share bars, authenticated evidence files, dated instrument
matching, and numerical compatibility limits frozen before portfolio results.
An alternative vendor is labelled as such, never called an Intrinio correction.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib
import json
from math import isfinite
from numbers import Real
from pathlib import Path, PurePosixPath
import re

import pandas as pd

from .catalog import canonical_sha256
from .universe import DataContractError

OHLC = ("open", "high", "low", "close")
ADJUSTED = tuple("adj_" + field for field in OHLC)
ORIGINAL_FIELDS = (*OHLC, *ADJUSTED, "factor", "price_quality_status",
                   "raw_ohlc_bounds_valid", "adjusted_ohlc_bounds_valid")


def _number(value, name, *, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise DataContractError(name + " must be finite numeric data")
    value = float(value)
    if value < 0 or (not nonnegative and value == 0):
        raise DataContractError(name + " has invalid sign")
    return value


def _day(value):
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise DataContractError("Repair dates must be ISO calendar-date strings")
    try:
        result = date.fromisoformat(value)
    except ValueError as exc:
        raise DataContractError("Invalid repair calendar date") from exc
    if not date(2018, 1, 1) <= result <= date(2019, 12, 31):
        raise DataContractError("Repair leaves the authorized 2018–2019 period")
    return result


def _hash(value, name):
    if not isinstance(value, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", value):
        raise DataContractError(name + " requires a canonical SHA256 pin")
    return value


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise DataContractError(name + " must be explicit")
    return value


def original_bar_sha256(row):
    """Pin exact original values and rejection flags, including missing fields."""
    values = {}
    for field in ORIGINAL_FIELDS:
        value = row.get(field)
        if value is None or pd.isna(value):
            value = None
        elif hasattr(value, "item"):
            value = value.item()
        values[field] = value
    return canonical_sha256(values)


@dataclass(frozen=True)
class OhlcRepairPolicy:
    """No empirical tolerances are silently supplied by this worker.

    Both absolute AND relative close differences must pass. The paired-price
    residual limit allows only the provider's documented decimal rounding when
    transporting raw-share OHLC into the existing adjusted-price proxy basis.
    Intrinio's `factor` is retained as evidence, never used as that multiplier.
    """

    maximum_raw_close_absolute_difference: float
    maximum_raw_close_relative_difference: float
    maximum_original_adjusted_rounding_residual: float
    policy_id: str

    def __post_init__(self):
        _text(self.policy_id, "Repair policy identity")
        for field in ("maximum_raw_close_absolute_difference",
                      "maximum_raw_close_relative_difference",
                      "maximum_original_adjusted_rounding_residual"):
            _number(getattr(self, field), field, nonnegative=True)


@dataclass(frozen=True)
class OhlcRepairResult:
    prices: pd.DataFrame
    ledger: pd.DataFrame
    evidence_pins: tuple


def _verify_evidence(root, evidence):
    if not isinstance(evidence, dict) or set(evidence) != {"path", "size_bytes", "sha256", "source_uri"}:
        raise DataContractError("Closed evidence artifact pin required")
    relative = evidence["path"]
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise DataContractError("Evidence requires a relative POSIX path")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or any(part in (".", "..") for part in relative.split("/")):
        raise DataContractError("Evidence path leaves approved evidence root")
    path = root.joinpath(*pure.parts)
    if not path.resolve().is_relative_to(root) or any(p.is_symlink() for p in (path, *path.parents)):
        raise DataContractError("Evidence cannot use escaped or linked paths")
    size = evidence["size_bytes"]
    if type(size) is not int or not 0 < size <= 16 * 1024 * 1024:
        raise DataContractError("Evidence artifact must have a finite small byte bound")
    if not path.is_file() or path.stat().st_size != size:
        raise DataContractError("Evidence artifact missing or wrong size")
    expected = _hash(evidence["sha256"], "Evidence artifact")
    if "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise DataContractError("Evidence artifact hash changed")
    _text(evidence["source_uri"], "Evidence source URI")
    return dict(evidence)


def apply_reviewed_ohlc_repairs(prices, repairs, *, evidence_root, policy):
    """Return an independent overlay and ledger, never mutate source caches.

    Every repair must match one existing source row. Missing entire bars need a
    separately reviewed source-admission path; this worker will not manufacture
    a source key. The original close anchors adjustment units only after both
    close-compatibility limits pass. Identity/date evidence is an explicit
    reviewed declaration with actual source pins, not independent certification.
    """
    if not isinstance(prices, pd.DataFrame) or not isinstance(policy, OhlcRepairPolicy):
        raise DataContractError("Explicit price frame and frozen OHLC policy required")
    keys = ["intrinio_security_id", "trading_date"]
    if not set((*keys, *ORIGINAL_FIELDS)) <= set(prices.columns):
        raise DataContractError("Repair needs complete original raw/adjusted fields and flags")
    if prices[keys].isna().any().any() or prices.duplicated(keys).any() or not prices.index.is_unique:
        raise DataContractError("Original price keys must be unique and nonmissing")
    for day in prices.trading_date.unique():
        _day(day)
    if not isinstance(repairs, (list, tuple)) or len(repairs) > 256:
        raise DataContractError("Explicit bounded repair declarations required")
    root = Path(evidence_root).resolve()
    if not root.is_dir():
        raise DataContractError("Evidence root is missing")
    result = prices.copy(deep=True)
    if any("original_" + field in result for field in ORIGINAL_FIELDS):
        raise DataContractError("Do not stack or overwrite an existing OHLC overlay")
    for field in ORIGINAL_FIELDS:
        result["original_" + field] = result[field]
    result["ohlc_repair_id"] = None
    result["ohlc_repair_source_kind"] = None
    result["ohlc_repair_policy_id"] = None
    result["ohlc_repair_evidence_sha256"] = None
    seen, ledger, pins = set(), [], []
    fields = {"repair_id", "intrinio_security_id", "trading_date", "original_bar_sha256",
              "raw_ohlc", "source_kind", "provider", "evidence", "identity_review"}
    for repair in repairs:
        if not isinstance(repair, dict) or set(repair) != fields:
            raise DataContractError("Closed reviewed repair schema required")
        repair_id = _text(repair["repair_id"], "Repair ID")
        sid = _text(repair["intrinio_security_id"], "Native security ID")
        day = repair["trading_date"]; _day(day)
        if (sid, day) in seen:
            raise DataContractError("Duplicate repair key")
        seen.add((sid, day))
        matched = result.index[(result.intrinio_security_id == sid) & (result.trading_date == day)]
        if len(matched) != 1:
            raise DataContractError("Repair does not match one existing original source bar")
        index = matched[0]; original = prices.loc[index]
        if original_bar_sha256(original) != _hash(repair["original_bar_sha256"], "Original bar"):
            raise DataContractError("Original bar differs from reviewed source evidence")
        if original["price_quality_status"] not in ("provider_ohlc_anomaly", "provider_incomplete_ohlc"):
            raise DataContractError("This overlay is limited to previously rejected OHLC bars")
        if repair["source_kind"] not in ("provider_corrected_raw_bar", "alternate_provider_raw_bar"):
            raise DataContractError("Actual corrected or alternative-provider raw bar required")
        provider = _text(repair["provider"], "Replacement provider")
        identity = repair["identity_review"]
        if not isinstance(identity, dict) or set(identity) != {"status", "source_identifier", "native_security_id",
                "valid_from", "valid_to", "currency", "price_basis", "review_notes", "evidence"}:
            raise DataContractError("Dated instrument and raw-share basis review required")
        if (identity["status"] != "REVIEWED_PROVISIONAL_SAME_SECURITY" or identity["native_security_id"] != sid
                or identity["currency"] != "USD" or identity["price_basis"] != "unadjusted_USD_per_original_share"):
            raise DataContractError("Reviewed identity/currency/raw-share basis mismatch")
        try:
            valid_from = date.fromisoformat(identity["valid_from"])
            valid_to = date.fromisoformat(identity["valid_to"])
        except (ValueError, TypeError) as exc:
            raise DataContractError("Invalid instrument validity dates") from exc
        if not valid_from <= date.fromisoformat(day) <= valid_to:
            raise DataContractError("Replacement date outside reviewed instrument interval")
        _text(identity["source_identifier"], "Source instrument identifier")
        _text(identity["review_notes"], "Source/instrument/units review notes")
        evidence = _verify_evidence(root, repair["evidence"])
        identity_evidence = _verify_evidence(root, identity["evidence"])
        raw = repair["raw_ohlc"]
        if not isinstance(raw, dict) or set(raw) != set(OHLC):
            raise DataContractError("Actual complete raw OHLC required")
        raw = {field: _number(raw[field], "Replacement " + field) for field in OHLC}
        if not raw["low"] <= min(raw["open"], raw["close"]) <= max(raw["open"], raw["close"]) <= raw["high"]:
            raise DataContractError("Replacement OHLC bounds fail")
        original_close = _number(original["close"], "Original raw close")
        adjusted_close = _number(original["adj_close"], "Original adjusted close")
        close_difference = abs(raw["close"] - original_close)
        close_relative = close_difference / original_close
        if (close_difference > policy.maximum_raw_close_absolute_difference
                or close_relative > policy.maximum_raw_close_relative_difference):
            raise DataContractError("Alternate close conflict needs separate source/unit review")
        multiplier = adjusted_close / original_close
        residuals = []
        for field in OHLC:
            a, b = original[field], original["adj_" + field]
            if pd.isna(a) or pd.isna(b):
                continue
            residual = abs(_number(a, "Original " + field) * multiplier - _number(b, "Original adjusted " + field))
            residuals.append(residual)
            if residual > policy.maximum_original_adjusted_rounding_residual:
                raise DataContractError("Original raw/adjusted pairs conflict with close-derived basis")
        adjusted = {"adj_" + field: raw[field] * multiplier for field in OHLC}
        if not all(isfinite(value) and value > 0 for value in adjusted.values()):
            raise DataContractError("Repair adjustment conversion is nonfinite")
        for field, value in {**raw, **adjusted}.items():
            result.at[index, field] = value
        result.at[index, "price_quality_status"] = "accepted"
        result.at[index, "raw_ohlc_bounds_valid"] = True
        result.at[index, "adjusted_ohlc_bounds_valid"] = True
        result.at[index, "ohlc_repair_id"] = repair_id
        result.at[index, "ohlc_repair_source_kind"] = repair["source_kind"]
        result.at[index, "ohlc_repair_policy_id"] = policy.policy_id
        result.at[index, "ohlc_repair_evidence_sha256"] = evidence["sha256"]
        ledger.append({"repair_id": repair_id, "intrinio_security_id": sid, "trading_date": day,
            "source_kind": repair["source_kind"], "provider": provider,
            "original_bar_sha256": repair["original_bar_sha256"], "source_evidence_sha256": evidence["sha256"],
            "identity_evidence_sha256": identity_evidence["sha256"], "policy_id": policy.policy_id,
            "raw_to_existing_adjusted_multiplier": multiplier, "original_factor_retained_unused": original["factor"],
            "raw_close_absolute_difference": close_difference, "raw_close_relative_difference": close_relative,
            "maximum_original_pair_residual": max(residuals, default=0),
            "original_rejection_status": original["price_quality_status"],
            "historical_identity_certified": False, "knowledge_time_certified": False})
        pins.extend((evidence, identity_evidence))
    return OhlcRepairResult(result, pd.DataFrame(ledger), tuple(pins))
