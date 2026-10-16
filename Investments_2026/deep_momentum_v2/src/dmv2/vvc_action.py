"""Verified Vectren cash retirement, including its unrepresented stub dividend.

The February 1, 2019 opening suspension is independently documented by the
issuer. January raw and adjusted observations have equal units and no dividend
adjustment; the separate closing dividend is therefore added exactly once.
Existing action accounting treats the total signed amount as an economic
receivable/liability, without claiming broker settlement at that opening.
"""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .action_loading import EVENT_TERMS
from .sni_action import SNI_EVENT, _verify_pin
from .universe import DataContractError


VVC_PREDECESSOR_ID = "sec_gkmmPz"
VVC_EVENT = {
    "event_id": "VVC-2019-02-01",
    "predecessor_id": VVC_PREDECESSOR_ID,
    "effective_date": "2019-02-01",
    "legal_effective_date": "2019-02-01",
    "successor_id": None,
    "cash_per_raw_share": 72.41145,
    "stock_ratio": 0.0,
    "merger_cash_per_raw_share": 72.0,
    "stub_dividend_per_raw_share": 0.41145,
    "stub_dividend_previous_adjusted_price_recognition": False,
    "stub_dividend_payment_deadline": "2019-02-08",
    "verified": True,
    "evidence_reference": "https://www.sec.gov/Archives/edgar/data/1130310/000119312519025980/d682591d8k.htm",
    "timing_evidence_reference": "https://www.sec.gov/Archives/edgar/data/1096385/000119312519025987/d699426d8k.htm",
    "dividend_evidence_reference": "https://www.sec.gov/Archives/edgar/data/1096385/000119312519025987/d699426d8k.htm",
    "suspension_timing": "before_NYSE_open_on_application_session",
    "settlement_verified": False,
    "knowledge_time_verified": False,
    "new_order_after_retirement": "unfilled_cash_no_merger_or_dividend_entitlement",
    "cash_recognition_policy": "signed_economic_merger_plus_unrepresented_stub_dividend_at_effective_open_zero_rate",
}


def validate_vvc_action_prices(prices):
    if not isinstance(prices, pd.DataFrame):
        raise DataContractError("VVC action requires original observed daily prices")
    date_field = "trading_date" if "trading_date" in prices else "date"
    required = {date_field, "intrinio_security_id", "open", "high", "low", "close",
        "adj_open", "adj_high", "adj_low", "adj_close", "dividend", "split_ratio",
        "frequency", "intraperiod", "price_quality_status"}
    if not required <= set(prices):
        raise DataContractError("VVC action requires paired raw/adjusted and dividend controls")
    try:
        dates = pd.DatetimeIndex(pd.to_datetime(prices[date_field], errors="raise"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise DataContractError("Malformed VVC evidence dates") from exc
    if (dates.tz is not None or dates.isna().any() or not dates.equals(dates.normalize())
            or dates.min() < pd.Timestamp("2018-01-01") or dates.max() > pd.Timestamp("2019-12-31")):
        raise DataContractError("VVC financial evidence must remain within 2018–2019")
    native = prices.intrinio_security_id.eq(VVC_PREDECESSOR_ID).to_numpy()
    if (native & (dates >= pd.Timestamp("2019-02-01"))).any():
        raise DataContractError("Cannot fabricate tradable VVC quotes after the verified suspension")
    january = prices.loc[native & (dates >= pd.Timestamp("2019-01-01"))].copy()
    january_dates = dates[native & (dates >= pd.Timestamp("2019-01-01"))]
    if (len(january) != 21 or january_dates.has_duplicates
            or january_dates.min() != pd.Timestamp("2019-01-02")
            or january_dates.max() != pd.Timestamp("2019-01-31")):
        raise DataContractError("All 21 observed January VVC controls including January 31 are required")
    if (not january.frequency.eq("daily").all()
            or not january.intraperiod.map(lambda value: isinstance(value, (bool, np.bool_)) and not value).all()
            or not january.price_quality_status.eq("accepted").all()):
        raise DataContractError("VVC basis controls must be accepted completed daily observations")
    for field in ("open", "high", "low", "close"):
        raw = pd.to_numeric(january[field], errors="coerce").to_numpy(dtype=float)
        adjusted = pd.to_numeric(january["adj_" + field], errors="coerce").to_numpy(dtype=float)
        if (not np.isfinite(raw).all() or not (raw > 0).all()
                or not np.isfinite(adjusted).all()
                or not np.allclose(raw, adjusted, rtol=0., atol=1e-12)):
            raise DataContractError("VVC dividend add-on requires unchanged January raw/adjusted units")
    if (not january.dividend.eq(0.).all() or not january.split_ratio.eq(1.).all()):
        raise DataContractError("VVC stub may already be represented by a January action; review before adding")


def load_vvc_action(evidence_directory):
    directory = Path(evidence_directory).resolve()
    with (directory / "vvc_action_evidence.json").open() as stream:
        receipt = json.load(stream)
    if (receipt.get("schema") != "dmv2-vvc-action-evidence/v1"
            or receipt.get("event") != VVC_EVENT
            or receipt.get("no_new_quote_created") is not True
            or receipt.get("stub_dividend_not_previously_represented") is not True
            or set(receipt.get("artifacts", {})) != {"predecessor_controls.parquet"}):
        raise DataContractError("VVC reviewed cash and dividend evidence changed")
    path = _verify_pin(directory, receipt["artifacts"]["predecessor_controls.parquet"])
    validate_vvc_action_prices(pd.read_parquet(path))
    return deepcopy(VVC_EVENT)


def append_vvc_action(events, evidence_directory, *, prices=None):
    """Preserve nine reviewed event dictionaries and append the tenth event."""
    original = events.to_dict("records") if isinstance(events, pd.DataFrame) else list(events)
    expected_ids = {ticker + "-" + terms[1] for ticker, terms in EVENT_TERMS.items()} | {SNI_EVENT["event_id"]}
    if (len(original) != 9 or not all(isinstance(row, dict) for row in original)
            or {row.get("event_id") for row in original} != expected_ids):
        raise DataContractError("VVC amendment requires the unchanged original nine actions")
    for row in original:
        if row["event_id"] == SNI_EVENT["event_id"]:
            if row != SNI_EVENT:
                raise DataContractError("Adding VVC cannot alter the reviewed SNI action")
        else:
            ticker = row["event_id"].split("-", 1)[0]
            terms = EVENT_TERMS[ticker]
            if (row.get("predecessor_id"), row.get("effective_date"), row.get("cash_per_raw_share"), row.get("stock_ratio")) != (
                    terms[0], terms[2], terms[3], terms[4]):
                raise DataContractError("Adding VVC cannot alter the original eight event terms")
    action = load_vvc_action(evidence_directory)
    if prices is not None:
        validate_vvc_action_prices(prices)
    return deepcopy(original) + [action]
