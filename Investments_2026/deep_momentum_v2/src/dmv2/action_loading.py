"""Bounded raw/adjusted inputs for the additive 2018–2019 action ledger.

Original pilot/cache loaders and source files remain immutable. Planning reads
only date/identity metadata and hashes; unresolved successor evidence blocks
the default financial materialization. Audit-only loading is explicitly marked
unready. An external quote file confers neither historical identity certification
nor permission to overwrite an original bar.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from . import backtest_loading as base
from .catalog import canonical_sha256
from .universe import DataContractError, iso_date

ACTION_PRICE_COLUMNS = (*base.PRICE_COLUMNS, "open", "high", "low", "close")
KNOWN_SUCCESSORS = {"ESRX": "sec_Qz8jlg", "GXP": "sec_Xn1EW4", "LLL": "sec_gYrYW2"}
UNQUOTED_SUCCESSOR_POLICIES = ("reject", "withhold_affected_paths")
RDC_UNQUOTED_ASSET = "unquoted_successor:CUSIP:G3166L100"
# Completion terms reviewed independently from the provider quote inventory.
EVENT_TERMS = {
    "BIVV": ("sec_gedWvo", "2018-03-08", "2018-03-08", 105.0, 0.0),
    "ESRX": ("sec_zrJv8z", "2018-12-20", "2018-12-21", 48.75, 0.2434),
    "IDTI": ("sec_gVbw5g", "2019-03-29", "2019-04-01", 49.0, 0.0),
    "GXP": ("sec_XWOwJX", "2018-06-04", "2018-06-05", 0.0, 0.5981),
    "RDC": ("sec_zmYNnz", "2019-04-11", "2019-04-11", 0.0, 0.6875),
    "ESL": ("sec_XnlJ4y", "2019-03-14", "2019-03-14", 122.5, 0.0),
    "BID": ("sec_Bgk1Bz", "2019-10-03", "2019-10-03", 57.0, 0.0),
    "LLL": ("sec_zEMdWz", "2019-06-29", "2019-07-01", 0.0, 1.3),
}


@dataclass(frozen=True)
class ActionInputPlan:
    root: Path
    evidence_dir: Path
    base_plan: base.BacktestDataPlan
    events: tuple
    evidence_pins: tuple
    supplemental_paths: tuple
    supplemental_metadata: tuple
    successor_bindings: tuple
    requested_ids: tuple
    unquoted_successor_policy: str
    readiness_failures: tuple
    read_plan: dict
    fingerprint: str

    @property
    def ready(self):
        return not self.readiness_failures

    @property
    def summary(self):
        return self.read_plan

    @property
    def accounting_may_run(self):
        """An allowed attempt can still withhold its first unvalued holding."""
        return self.ready or (self.unquoted_successor_policy == "withhold_affected_paths"
            and all(item["reason"] in {"unresolved_successor_identity", "missing_successor_quote_keys"}
                    for item in self.readiness_failures))


@dataclass(frozen=True)
class ActionInputBundle:
    predictions: dict
    daily_prices: pd.DataFrame
    events: pd.DataFrame
    sessions: tuple
    windows: pd.DataFrame
    plan: ActionInputPlan
    provenance: dict

    @property
    def ready(self):
        return self.plan.ready


def _source_url(value):
    parsed = urlparse(value if isinstance(value, str) else "")
    if parsed.scheme != "https" or parsed.hostname not in {
            "www.sec.gov", "sec.gov", "www.renesas.com", "www.nasdaqtrader.com"}:
        raise DataContractError("Action requires a reviewed primary-source evidence URL")
    return value


def _keys(table, sessions):
    frame = table.to_pandas()
    if frame.isna().any().any() or frame.duplicated(list(base.PRICE_KEYS)).any():
        raise DataContractError("Action quote metadata has null or duplicate identity/date keys")
    for day in frame.trading_date.unique():
        if not "2018-01-01" <= iso_date(day, "trading_date") <= "2019-12-31":
            raise DataContractError("Action quotes would decode financial values outside 2018–2019")
    if not frame.intrinio_security_id.map(lambda x: isinstance(x, str) and bool(x.strip())).all():
        raise DataContractError("Action quote metadata has an invalid instrument identity")
    if not frame.trading_date.isin(sessions).all():
        raise DataContractError("Action quotes include a non-exchange session")
    return frame


def _snapshot(plan):
    return {"root": str(plan.root), "evidence_dir": str(plan.evidence_dir),
        "base_fingerprint": plan.base_plan.fingerprint, "events": list(plan.events),
        "evidence_pins": list(plan.evidence_pins),
        "supplemental_paths": [str(path) for path in plan.supplemental_paths],
        "supplemental_metadata": [base._table_hash(table) for table in plan.supplemental_metadata],
        "successor_bindings": list(plan.successor_bindings), "requested_ids": list(plan.requested_ids),
        "unquoted_successor_policy": plan.unquoted_successor_policy,
        "readiness_failures": list(plan.readiness_failures), "read_plan": plan.read_plan}


def _verify(plan):
    if not isinstance(plan, ActionInputPlan) or canonical_sha256(_snapshot(plan)) != plan.fingerprint:
        raise DataContractError("Action input plan changed after validation")
    base.verify_backtest_inputs(plan.base_plan)
    for item in plan.evidence_pins:
        base._verify_pin(plan.evidence_dir, item)


def plan_action_inputs(root, *, evidence_dir=None, max_rows=base.DEFAULT_MAX_ROWS,
        max_projected_bytes=base.DEFAULT_MAX_PROJECTED_BYTES, unquoted_successor_policy="reject"):
    """Plan two pinned annual caches plus optional exact pinned successor quotes.

    ``evidence_dir/action_evidence.json`` is required. Optional
    ``supplemental_inputs.json`` uses schema ``dmv2-action-supplement/v1`` with
    ``successor_bindings`` and ``quote_files`` lists. A binding has event_id,
    instrument_id, identifier_type, identifier and evidence_reference. Only the
    unresolved RDC CUSIP G3166L100 may be bound this way; a binding alone is not
    a quote. Each quote file has path/size_bytes/sha256/instrument_id and
    evidence_reference, and must supply the full reviewed raw/adjusted columns.
    Files are append-only missing observations; overlap is rejected, never an
    implicit OHLC correction. Corrections use the separately reviewed overlay.

    The explicit ``withhold_affected_paths`` policy permits fail-closed ledger
    attempts with an unquoted public entitlement asset. It is not an Intrinio
    identifier, a quote, or a complete comparison. Default ``reject`` retains
    global refusal while a successor is unresolved.
    """
    if unquoted_successor_policy not in UNQUOTED_SUCCESSOR_POLICIES:
        raise DataContractError("Explicit reviewed unquoted-successor policy required")
    root = Path(root if isinstance(root, (str, Path)) else root.root).resolve()
    evidence_dir = Path(evidence_dir or root / "data/derived/daily_action_inputs").resolve()
    original = base.plan_daily_backtest(root, max_rows=max_rows,
        max_projected_bytes=max_projected_bytes)
    evidence_path = evidence_dir / "action_evidence.json"
    evidence = base._json(evidence_path)
    pins = [base._pin(evidence_dir, evidence_path)]
    records = evidence.get("events")
    if (not isinstance(records, list) or len(records) != 8
            or {item.get("old_ticker") for item in records if isinstance(item, dict)} != set(EVENT_TERMS)):
        raise DataContractError("Exactly the eight reviewed corporate actions are required")
    # Every event date is checked before processing its monetary terms.
    for item in records:
        for name in ("effective_date", "application_session"):
            if not "2018-01-01" <= iso_date(item.get(name), name) <= "2019-12-31":
                raise DataContractError("Action evidence lies outside 2018–2019")
        if item["application_session"] not in original.sessions:
            raise DataContractError("Action application date is not an approved exchange session")
    manifest_path = evidence_dir / "supplemental_inputs.json"
    supplemental = {"schema": "dmv2-action-supplement/v1", "successor_bindings": [], "quote_files": []}
    if manifest_path.exists():
        supplemental = base._json(manifest_path)
        pins.append(base._pin(evidence_dir, manifest_path))
    if (set(supplemental) != {"schema", "successor_bindings", "quote_files"}
            or supplemental["schema"] != "dmv2-action-supplement/v1"
            or not isinstance(supplemental["successor_bindings"], list)
            or not isinstance(supplemental["quote_files"], list)
            or len(supplemental["successor_bindings"]) > 1
            or len(supplemental["quote_files"]) > 8):
        raise DataContractError("Closed bounded supplemental action manifest required")
    bindings = supplemental["successor_bindings"]
    for binding in bindings:
        if (not isinstance(binding, dict) or set(binding) != {
                "event_id", "instrument_id", "identifier_type", "identifier", "evidence_reference"}
                or binding["event_id"] != "RDC-2019-04-11"
                or binding["identifier_type"] != "CUSIP" or binding["identifier"] != "G3166L100"
                or not isinstance(binding["instrument_id"], str) or not binding["instrument_id"].strip()
                or binding["instrument_id"].startswith(("unresolved:", "unquoted_successor:"))
                or binding["instrument_id"] in {"sec_gk5pAW", "sec_Xa9e96", "sec_gx2bWz"}):
            raise DataContractError("Supplement must identify the actual 2019 Ensco Rowan security")
        _source_url(binding["evidence_reference"])
    events = []
    for item in records:
        ticker = item["old_ticker"]
        actual = (item.get("old_provider_id"), item["effective_date"], item["application_session"],
            item.get("cash_per_old_raw_share"), item.get("successor_raw_shares"))
        if actual != EVENT_TERMS[ticker] or any(type(value) is bool for value in actual[-2:]):
            raise DataContractError("Reviewed corporate-action terms changed: " + ticker)
        successor = KNOWN_SUCCESSORS.get(ticker)
        if ticker == "RDC" and bindings:
            successor = bindings[0]["instrument_id"]
        unquoted = ticker == "RDC" and not bindings
        if unquoted and unquoted_successor_policy == "withhold_affected_paths":
            successor = RDC_UNQUOTED_ASSET
        events.append({"event_id": ticker + "-" + item["effective_date"],
            "predecessor_id": item["old_provider_id"], "effective_date": item["application_session"],
            "legal_effective_date": item["effective_date"], "successor_id": successor,
            "cash_per_raw_share": item["cash_per_old_raw_share"], "stock_ratio": item["successor_raw_shares"],
            "verified": True, "evidence_reference": _source_url(item.get("source_url")),
            "timing_evidence_reference": _source_url(item.get("timing_source_url", item["source_url"])),
            "settlement_verified": False, "knowledge_time_verified": False,
            "successor_identity_status": "public_entitlement_unquoted_not_provider_mapping" if
                successor == RDC_UNQUOTED_ASSET else "provisional_candidate" if successor else
                "unresolved" if item["successor_raw_shares"] else "not_applicable",
            "successor_is_provider_mapping": bool(successor and successor != RDC_UNQUOTED_ASSET),
            "successor_quoted": bool(successor and successor != RDC_UNQUOTED_ASSET),
            "unresolved_successor_identifier": "CUSIP:G3166L100" if unquoted else None})
    requested_ids = tuple(sorted(set(original.requested_ids) | set(KNOWN_SUCCESSORS.values()) |
        {binding["instrument_id"] for binding in bindings}))
    all_keys, selected_count, candidate_count, projection = set(), 0, 0, 0
    for year in base.YEARS:
        frame = _keys(original.daily_metadata[year], original.sessions)
        all_keys.update(zip(frame.trading_date, frame.intrinio_security_id))
        selected_count += int(frame.intrinio_security_id.isin(requested_ids).sum())
        candidate_count += len(frame)
        projection += base._projected_bytes(original.daily_paths[year], ACTION_PRICE_COLUMNS)
    supplemental_paths, supplemental_metadata = [], []
    bound_ids = {binding["instrument_id"] for binding in bindings}
    if bound_ids & {native for _, native in all_keys}:
        raise DataContractError("Unresolved successor cannot alias an original cached security")
    for item in supplemental["quote_files"]:
        if (not isinstance(item, dict) or set(item) != {"path", "size_bytes", "sha256", "instrument_id", "evidence_reference"}
                or item["instrument_id"] not in bound_ids):
            raise DataContractError("Supplemental quotes require a reviewed exact instrument binding")
        _source_url(item["evidence_reference"])
        path = base._verify_pin(evidence_dir, item)
        if path.suffix != ".parquet" or path.stat().st_size > 4 * 1024 * 1024:
            raise DataContractError("Supplemental quotes require a small pinned Parquet file")
        pins.append(base._pin(evidence_dir, path))
        table = base._metadata(path, base.PRICE_KEYS)
        frame = _keys(table, original.sessions)
        if not len(frame) or len(frame) > 2000 or set(frame.intrinio_security_id) != {item["instrument_id"]}:
            raise DataContractError("Supplemental quote file differs from its exact instrument or row bound")
        keys = set(zip(frame.trading_date, frame.intrinio_security_id))
        if keys & all_keys:
            raise DataContractError("Supplemental quotes cannot overwrite an existing original quote")
        all_keys.update(keys)
        candidate_count += len(frame)
        selected_count += len(frame)
        # The supplemental key guard is an additional real projection; count
        # it separately from the later full quote projection.
        projection += (base._projected_bytes(path, base.PRICE_KEYS)
            + base._projected_bytes(path, ACTION_PRICE_COLUMNS))
        supplemental_paths.append(path); supplemental_metadata.append(table)
    total_projection = original.read_plan["projected_compressed_bytes"] + projection
    if total_projection > max_projected_bytes or original.read_plan["candidate_rows"] + sum(
            len(table) for table in supplemental_metadata) > max_rows:
        raise DataContractError("Measured action projections exceed explicit row/byte ceiling")
    failures = []
    for event in events:
        if event["stock_ratio"] and event["successor_id"] in (None, RDC_UNQUOTED_ASSET):
            failures.append({"event_id": event["event_id"], "reason": "unresolved_successor_identity",
                "identifier": event["unresolved_successor_identifier"],
                "unquoted_asset_key": event["successor_id"]})
        elif event["stock_ratio"]:
            # Required successor observations through the next rebalance open;
            # price quality is deliberately not a cohort-admission criterion.
            window = original.windows[original.windows.outcome_start.le(event["effective_date"]) &
                original.windows.outcome_end.ge(event["effective_date"])]
            end = window.iloc[0].exit_date
            needed = [day for day in original.sessions if event["effective_date"] <= day <= end]
            missing = [day for day in needed if (day, event["successor_id"]) not in all_keys]
            if missing:
                failures.append({"event_id": event["event_id"], "reason": "missing_successor_quote_keys",
                    "instrument_id": event["successor_id"], "missing_dates": missing})
    read_plan = {"outcome_years": [2018, 2019], "base_read_plan": original.read_plan,
        "daily_candidate_rows": candidate_count, "daily_selected_rows": selected_count,
        "price_projection": list(ACTION_PRICE_COLUMNS), "original_requested_stable_ids": len(original.requested_ids),
        "requested_ids_including_successors": len(requested_ids),
        "additional_projected_compressed_bytes": projection, "projected_compressed_bytes": total_projection,
        "max_rows": max_rows, "max_projected_bytes": max_projected_bytes,
        "financial_values_read": False, "financial_2020_plus_read": False,
        "all_physical_date_keys_checked_before_values": True,
        "base_price_projection_repeated_explicitly_counted": True,
        "unquoted_successor_policy": unquoted_successor_policy,
        "fail_closed_accounting_only": bool(failures) and unquoted_successor_policy == "withhold_affected_paths",
        "unquoted_assets_are_provider_mappings": False,
        "complete_paths_are_not_a_complete_comparison": True,
        "readiness_failures": failures, "supplemental_quote_files": len(supplemental_paths)}
    plan = ActionInputPlan(root, evidence_dir, original, tuple(events), tuple(pins),
        tuple(supplemental_paths), tuple(supplemental_metadata), tuple(bindings), requested_ids, unquoted_successor_policy,
        tuple(failures), read_plan, "")
    object.__setattr__(plan, "fingerprint", canonical_sha256(_snapshot(plan)))
    return plan


def load_action_inputs(root, *, evidence_dir=None, plan=None, require_ready=True,
        unquoted_successor_policy=None):
    """Materialize measured inputs; default refusal precedes any financial read.

    ``require_ready=False`` is only for inspecting unresolved evidence. Its
    bundle is explicitly unready and must not be passed to actual accounting.
    """
    root = Path(root if isinstance(root, (str, Path)) else root.root).resolve()
    if unquoted_successor_policy is not None and unquoted_successor_policy not in UNQUOTED_SUCCESSOR_POLICIES:
        raise DataContractError("Explicit reviewed unquoted-successor policy required")
    plan = plan_action_inputs(root, evidence_dir=evidence_dir,
        unquoted_successor_policy="reject" if unquoted_successor_policy is None else unquoted_successor_policy) if plan is None else plan
    if (not isinstance(plan, ActionInputPlan) or plan.root != root or type(require_ready) is not bool
            or evidence_dir is not None and Path(evidence_dir).resolve() != plan.evidence_dir
            or unquoted_successor_policy is not None and unquoted_successor_policy != plan.unquoted_successor_policy):
        raise DataContractError("Action input plan/root/evidence scope differs")
    _verify(plan)
    if require_ready and not plan.accounting_may_run:
        raise DataContractError("Action inputs are not ready: " + json.dumps(list(plan.readiness_failures), sort_keys=True))
    original = base.load_daily_backtest(root, plan=plan.base_plan)
    tables = []
    paths = list(plan.base_plan.daily_paths.values()) + list(plan.supplemental_paths)
    metadata = list(plan.base_plan.daily_metadata.values()) + list(plan.supplemental_metadata)
    for path, expected in zip(paths, metadata, strict=True):
        table = pq.ParquetFile(path).read(columns=list(ACTION_PRICE_COLUMNS))
        if not table.select(list(base.PRICE_KEYS)).equals(expected):
            raise DataContractError("Action quote keys changed after metadata planning")
        mask = pc.is_in(table["intrinio_security_id"], value_set=pa.array(plan.requested_ids))
        tables.append(table.filter(mask).to_pandas())
    prices = pd.concat(tables, ignore_index=True).sort_values(list(base.PRICE_KEYS), ignore_index=True)
    if len(prices) != plan.read_plan["daily_selected_rows"] or prices.duplicated(list(base.PRICE_KEYS)).any():
        raise DataContractError("Action materialization differs from the measured key grid")
    _verify(plan)
    provenance = {**original.provenance, "action_input_fingerprint": plan.fingerprint,
        "read_plan": {**plan.read_plan, "financial_values_read": True},
        "ready_for_action_accounting": plan.accounting_may_run,
        # This loader checks successor evidence and date/key support. Only
        # the complete held-position ledger can establish full price coverage.
        "fully_priced_inputs_ready": False,
        "successor_evidence_ready": plan.ready,
        "fail_closed_accounting_only": not plan.ready and plan.accounting_may_run,
        "audit_only_unresolved_inputs": not plan.ready and not plan.accounting_may_run,
        "unquoted_successor_policy": plan.unquoted_successor_policy,
        "complete_paths_are_not_a_complete_comparison": True,
        "reviewed_event_terms": len(plan.events), "source_files_unchanged": True,
        "identity_flags_promoted": False, "knowledge_time_verified": False,
        "accounting_certified": False, "financial_2020_plus_read": False,
        "selection_rules_changed": False, "original_source_bars_overwritten": False}
    return ActionInputBundle(original.predictions, prices, pd.DataFrame(plan.events), original.sessions,
        original.windows.copy(deep=True), plan, provenance)
