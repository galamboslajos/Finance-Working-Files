"""Add the evidenced SNI/Discovery entitlement without changing old events.

SNI ceased Nasdaq trading effective March 7, 2018. An existing position receives
the default non-election mixed package, not an invented post-delisting SNI quote.
The frozen engine retains theoretical fractional DISCK units; this is explicitly
distinct from the issuer's actual cash-in-lieu procedure and unknown settlement.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .action_loading import EVENT_TERMS
from .universe import DataContractError


SNI_PREDECESSOR_ID = "sec_zobYxX"
SNI_SUCCESSOR_ID = "sec_gA9OJg"
SNI_EVENT = {
    "event_id": "SNI-2018-03-06",
    "predecessor_id": SNI_PREDECESSOR_ID,
    "effective_date": "2018-03-07",
    "legal_effective_date": "2018-03-06",
    "successor_id": SNI_SUCCESSOR_ID,
    "cash_per_raw_share": 65.82,
    "stock_ratio": 1.0584,
    "verified": True,
    "evidence_reference": "https://www.sec.gov/Archives/edgar/data/1437107/000143710718000042/form8-kxclosing20180306.htm",
    "timing_evidence_reference": "https://www.nasdaqtrader.com/TraderNews.aspx?id=ECA2018-37",
    "settlement_verified": False,
    "knowledge_time_verified": False,
    "consideration_policy": "default_non_election_mixed_consideration",
    "successor_share_class": "Discovery Series C common stock",
    "successor_ticker": "DISCK",
    "successor_isin": "US25470F3029",
    "predecessor_cusip": "811065101",
    "successor_identity_status": "provisional_provider_mapping_supported_by_exact_class_and_isin",
    "fractional_share_policy": "theoretical_fractional_units_retained_no_cash_in_lieu_invented",
    "actual_fractional_share_terms": "issuer_pays_cash_in_lieu_actual_amount_and_timing_unverified",
}


def _hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return "sha256:" + value.hexdigest()


def _pin(path, root):
    return {"path": str(path.relative_to(root)), "size_bytes": path.stat().st_size,
            "sha256": _hash(path)}


def _verify_pin(directory, item):
    if not isinstance(item, dict) or set(item) != {"path", "size_bytes", "sha256"}:
        raise DataContractError("SNI evidence requires exact local file pins")
    path = (directory / item["path"]).resolve()
    if not path.is_relative_to(directory) or not path.is_file():
        raise DataContractError("SNI evidence path is missing or escapes its package")
    if path.stat().st_size != item["size_bytes"] or _hash(path) != item["sha256"]:
        raise DataContractError("SNI evidence bytes changed after review")
    return path


def load_sni_action(evidence_directory):
    """Verify a small private receipt and return fixed documented mixed terms."""
    directory = Path(evidence_directory).resolve()
    with (directory / "sni_action_evidence.json").open() as stream:
        receipt = json.load(stream)
    if (receipt.get("schema") != "dmv2-sni-action-evidence/v1"
            or receipt.get("event") != SNI_EVENT
            or receipt.get("no_new_quote_created") is not True
            or receipt.get("successor_class_verified") is not True
            or receipt.get("settlement_or_fractional_cash_invented") is not False):
        raise DataContractError("SNI event evidence/terms changed")
    expected = {"successor_identity.json", "predecessor_successor_controls.parquet"}
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != expected:
        raise DataContractError("SNI event receipt omits its bounded identity/price controls")
    paths = {name: _verify_pin(directory, item) for name, item in artifacts.items()}
    with paths["successor_identity.json"].open() as stream:
        identity = json.load(stream)
    master = identity.get("master")
    links = identity.get("links", [])
    if (not isinstance(master, dict)
            or master.get("intrinio_security_id") != SNI_SUCCESSOR_ID
            or master.get("ticker") != "DISCK"
            or master.get("type") != "Ordinary Shares - Series C"
            or not any(link.get("intrinio_security_id") == SNI_SUCCESSOR_ID
                and link.get("query_identifier_type") == "isin"
                and link.get("query_identifier") == "US25470F3029"
                and link.get("resolution_status") == "resolved" for link in links)):
        raise DataContractError("SNI successor must identify Discovery Series C, not Series A/B")
    validate_sni_action_prices(pd.read_parquet(paths["predecessor_successor_controls.parquet"]))
    return deepcopy(SNI_EVENT)


def validate_sni_action_prices(prices):
    """Require actual adjacent paired-price bases; never interpolate an exit."""
    if not isinstance(prices, pd.DataFrame):
        raise DataContractError("SNI action requires an observed price DataFrame")
    date_field = "trading_date" if "trading_date" in prices else "date"
    required = {date_field, "intrinio_security_id", "open", "close", "adj_open", "adj_close",
                "frequency", "intraperiod", "price_quality_status"}
    if not required <= set(prices):
        raise DataContractError("SNI action requires original paired raw/adjusted price fields")
    # Date/identity metadata is inspected before monetary fields.
    try:
        dates = pd.DatetimeIndex(pd.to_datetime(prices[date_field], errors="raise"))
    except (ValueError, TypeError, OverflowError) as exc:
        raise DataContractError("Malformed SNI action price dates") from exc
    if (dates.tz is not None or dates.isna().any()
            or not dates.equals(dates.normalize())
            or dates.min() < pd.Timestamp("2018-01-01")
            or dates.max() > pd.Timestamp("2019-12-31")):
        raise DataContractError("SNI price evidence must remain within 2018–2019")
    for native, day, fields in ((SNI_PREDECESSOR_ID, "2018-03-06", ("close", "adj_close")),
                                (SNI_SUCCESSOR_ID, "2018-03-07", ("open", "adj_open"))):
        mask = (prices.intrinio_security_id.eq(native).to_numpy()
                & (dates == pd.Timestamp(day)))
        selected = prices.loc[mask]
        if len(selected) != 1:
            raise DataContractError("SNI action lacks a unique observed predecessor/successor basis")
        row = selected.iloc[0]
        if (row.frequency != "daily" or not isinstance(row.intraperiod, (bool, np.bool_))
                or bool(row.intraperiod) or row.price_quality_status != "accepted"):
            raise DataContractError("SNI conversion quote is not an accepted completed daily observation")
        for field in fields:
            value = row[field]
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)) or not np.isfinite(value) or value <= 0:
                raise DataContractError("SNI action requires positive finite paired quote units")
    # A retired SNI quote must not be introduced as the supposed solution.
    if ((prices.intrinio_security_id.eq(SNI_PREDECESSOR_ID).to_numpy())
            & (dates >= pd.Timestamp("2018-03-07"))).any():
        raise DataContractError("SNI cannot be quoted as tradable after its verified suspension")


def append_sni_action(events, evidence_directory, *, prices=None):
    """Return all eight original events verbatim plus the ninth SNI event."""
    original = events.to_dict("records") if isinstance(events, pd.DataFrame) else list(events)
    if not all(isinstance(item, dict) for item in original):
        raise DataContractError("Original event mappings must be supplied unchanged")
    if (len(original) != 8
            or {row.get("event_id") for row in original}
                != {ticker + "-" + terms[1] for ticker, terms in EVENT_TERMS.items()}):
        raise DataContractError("The additive SNI action requires all eight original events")
    # Verify economic/date terms without rewriting original records or keys.
    for row in original:
        ticker = row["event_id"].split("-", 1)[0]
        terms = EVENT_TERMS[ticker]
        if (row.get("predecessor_id"), row.get("effective_date"),
                row.get("cash_per_raw_share"), row.get("stock_ratio")) != (
                terms[0], terms[2], terms[3], terms[4]):
            raise DataContractError("Original action terms changed while adding SNI")
    action = load_sni_action(evidence_directory)
    if prices is not None:
        validate_sni_action_prices(prices)
    return deepcopy(original) + [action]
