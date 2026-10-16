"""Reviewed old-BBBY identity amendment without mutating reference inputs.

The saved source queries the current BBBY ticker rather than the old issuer's
ISIN. Its historical native price route is invalidated in a separate metadata
view. No replacement native ID is invented. A complete repair still needs a
verified old-issuer provider response and correctly identified price history.
"""
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path

import pyarrow as pa

from .catalog import canonical_sha256
from .identity_policy import PROVISIONAL_IDENTITY_POLICY, qualify_identity_records
from .universe import DataContractError, UniverseSnapshot, iso_date

AMENDMENT_ID = "old-bbby-native-conflict-20261016-v1"
OLD_ISIN = "US0758961009"
WRONG_NATIVE = "sec_ydbQnj"
SOURCE_ROUTES = {
    "siblis_ticker_company_00326c7d9ad13903e5e3": "equity_us_883f1d58c4dcf36a3092bd1f",
    "siblis_isin_c297a5f81a23c9f90503": "equity_us_aacab20bc16f70ab265fd1e1",
}
DEFINITION = {
    "schema_version": 1, "amendment_id": AMENDMENT_ID,
    "source_routes": SOURCE_ROUTES, "reviewed_issuer": "Historical Bed Bath & Beyond Inc",
    "reviewed_isin": OLD_ISIN, "reviewed_cusip": "075896100",
    "invalid_native_id": WRONG_NATIVE, "replacement_native_id": None,
    "effect": "invalidate_known_wrong_native_route_retain_original_cohort",
    "repair_status": "WRONG_ROUTE_DISABLED_REPLACEMENT_NATIVE_UNRESOLVED",
    "default_analysis_through": "2019-12-31",
    "historical_identifier_validity_intervals_verified": False,
    "historical_knowledge_time_certified": False,
    "price_values_used_to_select_identity": False,
}

_ALLOWED = frozenset((
    "source_security_id", "as_of", "signal_date", "trading_date", "index_id", "indices",
    "source_product_id", "membership_tickers", "membership_records_json", "membership_ticker",
    "unified_security_id", "intrinio_security_id", "identity_status", "source_identity_status",
    "bridge_method", "bridge_confidence", "identity_review_required", "resolution_status",
    "identity_eligible", "identity_eligibility_policy", "identity_eligibility_reason",
    "requested", "paper_prediction_eligible", "prediction_eligible", "outcome_month",
))


@dataclass(frozen=True)
class IdentityAmendment:
    definition: dict
    evidence_pins: tuple
    fingerprint: str


@dataclass(frozen=True)
class AmendedIdentity:
    records: tuple
    ledger: tuple
    summary: dict
    amendment_sha256: str


def _need(condition, message):
    if not condition:
        raise DataContractError(message)


def _digest(path):
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _evidence(root, pin):
    _need(set(pin) == {"path", "size_bytes", "sha256"}, "Closed metadata-evidence pin required")
    _need(isinstance(pin["path"], str), "Evidence path required")
    base = Path(root).resolve();path = (base / pin["path"]).resolve()
    _need(path.is_relative_to(base) and not Path(pin["path"]).is_absolute(), "Evidence must remain inside its package")
    _need(type(pin["size_bytes"]) is int and 0 < pin["size_bytes"] <= 256 * 1024, "Metadata evidence bound exceeded")
    _need(path.is_file() and path.stat().st_size == pin["size_bytes"], "Metadata evidence size changed")
    _need(_digest(path) == pin["sha256"], "Metadata evidence hash changed")
    return json.loads(path.read_text())


def load_identity_amendment(package_root):
    """Authenticate the separate correction and its decisive metadata proof."""
    base = Path(package_root).resolve();path = base / "amendment.json"
    _need(path.is_file() and path.stat().st_size <= 32 * 1024, "Bounded amendment manifest required")
    raw = json.loads(path.read_text())
    _need(set(raw) == {"definition", "evidence"} and raw["definition"] == DEFINITION,
          "Unknown identity amendment; use a separate reviewed version")
    pins = raw["evidence"]
    _need(isinstance(pins, list) and len(pins) == 1, "One pinned local lookup proof required")
    proof = _evidence(base, pins[0]);conclusions = proof.get("conclusions", {})
    _need(conclusions.get("old_bed_bath_isin", {}).get("value") == OLD_ISIN
          and conclusions["old_bed_bath_isin"].get("verified_in_pinned_siblis_identity_metadata") is True
          and conclusions.get("historical_overstock_isin") == "US6903701018"
          and conclusions.get("conflicting_modern_native_id") == WRONG_NATIVE
          and conclusions.get("correct_old_bed_bath_intrinio_native_id") is None,
          "Metadata evidence does not establish this known unresolved native conflict")
    projection = proof.get("metadata_projection", {})
    links = {r["unified_security_id"]: r for r in projection.get("intrinio_identity_links", [])}
    old = links.get(SOURCE_ROUTES["siblis_isin_c297a5f81a23c9f90503"], {})
    ticker = links.get(SOURCE_ROUTES["siblis_ticker_company_00326c7d9ad13903e5e3"], {})
    modern = links.get("equity_us_78d1c5654bcd67cc2e83df43", {})
    _need(old.get("query_identifier_type") == "isin" and old.get("query_identifier") == "BBBY"
          and "query_identifier=BBBY-" in old.get("source_relative_path", "")
          and modern.get("query_identifier") == "US6903701018"
          and all(r.get("intrinio_security_id") == WRONG_NATIVE for r in (old,ticker,modern))
          and isinstance(old.get("source_response_hash"), str)
          and len(old["source_response_hash"]) == 64
          and old["source_response_hash"] == ticker.get("source_response_hash") == modern.get("source_response_hash"),
          "Exact query-literal/response-hash conflict evidence required")
    data = {"definition": deepcopy(raw["definition"]), "evidence": deepcopy(pins)}
    return IdentityAmendment(data["definition"], tuple(data["evidence"]), canonical_sha256(data))


def _verify(amendment):
    _need(isinstance(amendment, IdentityAmendment) and amendment.definition == DEFINITION,
          "Reviewed identity amendment required")
    _need(canonical_sha256({"definition": amendment.definition, "evidence": list(amendment.evidence_pins)})
          == amendment.fingerprint, "Identity amendment changed after review")


def apply_identity_amendment(records, amendment, *, date_field="signal_date",
                             analysis_through="2019-12-31", identity_policy=PROVISIONAL_IDENTITY_POLICY):
    """Return a dated metadata view with known wrong native exposure disabled.

    Every original source/date row remains. Original route/status/flags are
    retained in the audit columns. Price columns are rejected before reading
    values. Passing 2023 as the bound authorizes metadata inspection only.
    """
    _verify(amendment)
    iso_date(analysis_through, "analysis_through")
    _need(analysis_through < "2024-01-01", "Protected 2024+ amendment scope")
    _need(date_field in {"signal_date", "as_of", "trading_date"}, "Explicit cohort date field required")
    _need(isinstance(records, (list, tuple)), "Supply explicit metadata records")
    # Check names/dates for every row before touching any route or claimed flags.
    for row in records:
        _need(isinstance(row, dict) and set(row) <= _ALLOWED, "Identity amendment accepts metadata columns only")
        day = iso_date(row.get(date_field), "cohort date")
        _need(day <= analysis_through, "Cohort exceeds explicit amendment analysis bound")
        _need(day < "2024-01-01", "Protected 2024+ amendment cohort")
        for alias in ("as_of", "signal_date", "trading_date"):
            if alias in row:
                _need(iso_date(row[alias], "cohort date alias") == day,
                      "Conflicting cohort date aliases; qualification and amendment must use the same date")
    amended=[];ledger=[]
    for original in records:
        row=deepcopy(original);sid=row.get("source_security_id");day=row[date_field]
        # Old feature/request flags describe the reference panel. Identity
        # correction alone cannot certify new predictors or exhaustive requests.
        for name in ("paper_prediction_eligible", "prediction_eligible", "requested"):
            if name in row:row["original_"+name]=row.pop(name)
        if sid in SOURCE_ROUTES:
            _need(row.get("unified_security_id") == SOURCE_ROUTES[sid], "Known BBBY source/unified route changed; review its new evidence")
            _need(row.get("intrinio_security_id") == WRONG_NATIVE,
                  "Known BBBY native route changed; do not overwrite a possible new correction")
            for name in ("intrinio_security_id", "identity_status", "resolution_status", "identity_eligible"):
                row["original_" + name]=row.get(name)
            row.setdefault("source_identity_status", row["identity_status"])
            row.update(intrinio_security_id=None, identity_status="historical_identity_conflict",
                resolution_status="invalidated_known_wrong_historical_security",
                reviewed_security_isin=OLD_ISIN, identity_amendment_id=AMENDMENT_ID,
                identity_amendment_sha256=amendment.fingerprint,
                replacement_native_status="unresolved_requires_provider_evidence")
            ledger.append({"source_security_id":sid,"date":day,"original_intrinio_security_id":WRONG_NATIVE,
                "replacement_intrinio_security_id":None,"reviewed_security_isin":OLD_ISIN,
                "status":"WRONG_ROUTE_DISABLED_REPLACEMENT_NATIVE_UNRESOLVED"})
        # The common qualifier must see its own date field to recompute collisions.
        if date_field == "trading_date":row["signal_date"]=day
        amended.append(row)
    output=qualify_identity_records(amended, identity_policy=identity_policy)
    _need(len(output)==len(records), "Identity amendment must retain every original row")
    return AmendedIdentity(tuple(output),tuple(ledger),{
        "original_rows":len(records),"retained_rows":len(output),"invalidated_routes":len(ledger),
        "replacement_native_id_resolved":False,"ready_for_corrected_rebuild":False,
        "analysis_through":analysis_through,"financial_values_read":False,
        "identity_status_counts":dict(Counter(r["identity_status"] for r in output)),
        "historical_knowledge_time_certified":False},amendment.fingerprint)


def amend_universe_snapshot(snapshot, amendment, *, identity_policy=PROVISIONAL_IDENTITY_POLICY):
    """Amend a detached historical snapshot; its original object is immutable."""
    _need(isinstance(snapshot, UniverseSnapshot), "UniverseSnapshot required")
    result=apply_identity_amendment(snapshot.table.to_pylist(),amendment,date_field="as_of",
                                   identity_policy=identity_policy)
    names=sorted({name for row in result.records for name in row})
    normalized=[{name:row.get(name) for name in names} for row in result.records]
    return replace(snapshot,table=pa.Table.from_pylist(normalized),
        provenance={**snapshot.provenance,"identity_amendment_id":AMENDMENT_ID,
            "identity_amendment_sha256":amendment.fingerprint,"reference_provenance_retained":True,
            "replacement_native_id_resolved":False},
        summary={**snapshot.summary,"reference_identity_status_counts":snapshot.summary.get("identity_status_counts"),
            "identity_status_counts":result.summary["identity_status_counts"],"identity_amendment":result.summary})


def require_corrected_price_route(amended):
    """Never let a quarantine be mistaken for repaired, model-ready prices."""
    summary=amended.summary if isinstance(amended,AmendedIdentity) else amended.summary.get("identity_amendment",{})
    _need(summary.get("replacement_native_id_resolved") is True,
          "Old BBBY replacement native ID and historical prices remain unresolved; corrected rebuild is not ready")


def read_saved_identity_review(package_root):
    """Verify a saved amendment and small summary without decoding any panel."""
    base=Path(package_root).resolve()
    if not (base/"audit.json").is_file():
        raise FileNotFoundError("No saved BBBY identity correction review")
    amendment=load_identity_amendment(base)
    path=base/"audit.json"
    _need(path.stat().st_size<=256*1024,"Saved correction audit exceeds its metadata bound")
    audit=json.loads(path.read_text())
    _need(audit.get("schema")=="dmv2-bbby-identity-review/v1"
          and audit.get("amendment_sha256")==amendment.fingerprint
          and audit.get("replacement_native_id_resolved") is False
          and audit.get("corrected_features_rebuilt") is False
          and audit.get("models_refitted") is False,
          "Saved correction review identity/status mismatch")
    artifacts=audit.get("artifacts")
    _need(isinstance(artifacts,list) and 0<len(artifacts)<=100,"Bounded saved artifact pins required")
    for pin in artifacts:
        _need(set(pin)=={"path","size_bytes","sha256"} and isinstance(pin["path"],str),
              "Closed saved artifact pin required")
        target=(base/pin["path"]).resolve()
        _need(target.is_relative_to(base) and not Path(pin["path"]).is_absolute(),
              "Saved correction artifact escaped its package")
        _need(target.is_file() and target.stat().st_size==pin["size_bytes"]
              and _digest(target)==pin["sha256"],"Saved correction artifact hash/size changed")
    return {"amendment":amendment,"audit":audit,"summary":audit["summary"],"path":base}
