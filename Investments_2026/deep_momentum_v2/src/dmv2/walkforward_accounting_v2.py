"""Additive V2: retain unresolved runtime identity routes as withheld paths.

All V1 structural validation, quote/action/stop/trade arithmetic and artifact
contracts remain unchanged. Only two runtime identity guards become explicit
per-path failures before held/selected quote marks and trades. Mandatory legal
actions retain the original event-before-rebalance ordering. No handover or
identity equivalence is inferred, and no selection is changed.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from math import isfinite
from typing import Iterable
import numpy as np
import pandas as pd
from .walkforward_accounting import (
    ActionAccountingResult, DailyAccountingPolicy, DataContractError,
    _prepare, _prepare_actions, _historical_retirements, _positive_number,
    _true, _false, _dates, _DAILY_COLUMNS, _TRADE_COLUMNS, _HOLDING_COLUMNS,
    _FAILURE_COLUMNS, _STOP_COLUMNS,
)

IMPLEMENTATION_REVISION = 'identity_route_failure_v2'
HANDOVER_POLICY = 'exact_dated_execution_handover_under_active_identity_waiver'
_HANDOVER_COLUMNS = ['date','mark','handover_id','evidence_registry_record_id',
    'old_source_security_id','new_source_security_id','intrinio_security_id',
    'signal_date','held_signal_date','units_before','units_after','application_status',
    'execution_phase','evidence_reference','source_identity_verified',
    'source_identity_promoted','provisionally_admitted']


@dataclass
class IdentityAccountingResult(ActionAccountingResult):
    identity_handovers: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def tables(self):
        return {**super().tables, 'identity_handovers':self.identity_handovers}


def _prepare_handovers(records, sessions, schedule):
    records = records.to_dict('records') if isinstance(records,pd.DataFrame) else list(records)
    if len(records)>1:
        raise DataContractError('Execution handover registry exceeds its explicit one-record bound')
    result=[]
    for original in records:
        if not isinstance(original,dict):
            raise DataContractError('Execution handovers require explicit mapping records')
        row=dict(original)
        for name in ('handover_id','old_source_security_id','new_source_security_id',
                'intrinio_security_id','evidence_reference'):
            if not isinstance(row.get(name),str) or not row[name].strip():
                raise DataContractError('Execution handover requires a nonempty '+name)
        for name in ('effective_date','signal_date','held_signal_date'):
            row[name]=_dates([row.get(name)],'execution handover '+name)[0]
        if (row.get('verified') is not False or row.get('provisionally_admitted') is not True
                or row.get('execution_evidence_reviewed') is not True
                or row.get('admission_policy')!=HANDOVER_POLICY
                or row['old_source_security_id']==row['new_source_security_id']
                or row['effective_date'] not in sessions or row['effective_date'] not in schedule
                or row['signal_date']!=schedule[row['effective_date']]
                or not row['held_signal_date']<row['signal_date']<row['effective_date']):
            raise DataContractError('Execution handover must be exact dated, provisional and evidence reviewed')
        result.append(row)
    if (len({x['handover_id'] for x in result})!=len(result)
            or len({(x['effective_date'],x['old_source_security_id']) for x in result})!=len(result)
            or len({(x['effective_date'],x['new_source_security_id']) for x in result})!=len(result)
            or len({(x['effective_date'],x['intrinio_security_id']) for x in result})!=len(result)):
        raise DataContractError('Execution handover records must be unique and unambiguous')
    return result


def _admitted_handover(records, *, date, mark, old_key, held, selected_row, selected):
    if mark!='open' or held.get('action_predecessor_id'):
        return None
    old_sources={row['source_security_id'] for row in selected}
    native=selected_row['intrinio_security_id']
    if old_key in old_sources or sum(row['intrinio_security_id']==native for row in selected)!=1:
        return None
    for row in records:
        if (row['effective_date']==date and row['old_source_security_id']==old_key
                and row['new_source_security_id']==selected_row['source_security_id']
                and row['intrinio_security_id']==native==held['intrinio_security_id']
                and row['signal_date']==selected_row['signal_date']
                and row['held_signal_date']==held['signal_date']):
            return row
    return None

def run_walkforward_accounting_v2(
    selections: pd.DataFrame,
    prices: pd.DataFrame,
    sessions: Iterable,
    *,
    rebalances: pd.DataFrame,
    terminal_date,
    policy: DailyAccountingPolicy | None = None,
    events: pd.DataFrame | Iterable[dict] = (),
    historical_retirements: Iterable[dict] = (),
    identity_handovers: pd.DataFrame | Iterable[dict] = (),
) -> IdentityAccountingResult:
    """Account fixed selections with a verified, dated corporate-action ledger.

    Selections require signal_date, execution_date, source_security_id,
    intrinio_security_id and side. The independent rebalance schedule includes
    EVERY month, including empty selections. Prices require date/trading_date,
    provider ID and adjusted open/close; the stop variant also needs adjusted
    high/low. Raw completion/quality flags, or an explicit adapter acceptance
    flag, are checked before using each required quote.

    ``events`` supplies evidenced retirement terms per raw predecessor share:
    event_id, effective_date, predecessor_id, successor_id (None for cash-only),
    cash_per_raw_share, stock_ratio, verified=True and evidence_reference.
    Paired adjusted/raw prior-close and successor-open prices convert units;
    the provider's unrelated ``factor`` field is never used. Theoretical
    fractional successor shares are retained, without an invented cash-in-lieu
    payment. Same-session action chains require a separate ordered protocol.

    Each session has an open mark after scheduled trades/gap stops and a close
    mark after intraday stops. Final close additionally liquidates all survivors.
    The optional per-asset stop is fixed relative to each month's entry open:
    long entry*(1-fraction), short entry*(1+fraction). A gap beyond the threshold
    exits at open; an adverse daily high/low crossing assumes an idealized fill
    at the threshold. This OHLC simulation does not guarantee an attainable fill
    or cap a portfolio loss. Stopped names remain cash until monthly rebalance.
    """
    policy = policy or DailyAccountingPolicy()
    if not isinstance(policy, DailyAccountingPolicy):
        raise DataContractError("policy must be DailyAccountingPolicy")
    selections, prices, sessions, schedule, terminal = _prepare(
        selections, prices, sessions, rebalances, terminal_date)
    handover_definitions = _prepare_handovers(identity_handovers, sessions, schedule)
    action_definitions = _prepare_actions(events, sessions)
    retirements = _historical_retirements(historical_retirements, action_definitions)
    retirement_definitions = sorted([*retirements, *action_definitions], key=lambda event: (event["effective_date"], event["event_id"]))
    if policy.asset_stop_fraction is not None and not {"adj_high", "adj_low"}.issubset(prices.columns):
        raise DataContractError("Per-asset OHLC stops require adjusted high and low")
    # This is a computational projection after the all-row date guard. It never
    # changes a dated selection: only IDs in the already frozen selection ledger
    # can become held. Do not materialize unrelated OHLCV/action columns.
    quote_columns = [col for col in (
        "date", "intrinio_security_id", "adj_open", "adj_close", "adj_high", "adj_low",
        "frequency", "intraperiod", "price_quality_status", "price_accepted",
        "adjusted_ohlc_bounds_valid", "raw_ohlc_bounds_valid",
        "open", "close", "raw_open", "raw_close",
    ) if col in prices.columns]
    required_ids = set(selections.intrinio_security_id)
    required_ids.update(row["successor_id"] for row in action_definitions
                        if row["successor_id"] is not None)
    required_ids.update(row["predecessor_id"] for row in action_definitions)
    requested_prices = prices.loc[prices.intrinio_security_id.isin(required_ids), quote_columns]
    quote_rows = {(row["date"], row["intrinio_security_id"]): row for row in requested_prices.to_dict("records")}
    selections_by_date = {date: group.to_dict("records") for date, group in selections.groupby("execution_date", sort=False)}
    extras = [col for col in selections.columns if col not in {
        "signal_date", "execution_date", "source_security_id", "intrinio_security_id", "side",
        *_HOLDING_COLUMNS, *_TRADE_COLUMNS,
    }]
    daily, trades, holdings, failures, stops = [], [], [], [], []
    applied_actions, market_trades, orders = [], [], []
    applied_handovers = []
    actions_by_date = {}
    for event in action_definitions:
        actions_by_date.setdefault(event["effective_date"], []).append(event)
    positions: dict[str, dict] = {}
    last_marks: dict[str, float] = {}
    cash, previous_nav = float(policy.initial_nav), float(policy.initial_nav)
    previous_date, previous_mark = None, None
    failed = False

    def fail(date, mark, reason, position=None, field=None, role=None):
        nonlocal failed
        failed = True
        failures.append({"date": date, "mark": mark, "reason": reason,
                         "source_security_id": None if position is None else position["source_security_id"],
                         "intrinio_security_id": None if position is None else position["intrinio_security_id"],
                         "price_field": field, "role": role})

    def quote(date, mark, position, role, field=None):
        field = field or ("adj_open" if mark == "open" else "adj_close")
        row = quote_rows.get((date, position["intrinio_security_id"]))
        if row is None:
            fail(date, mark, "missing_quote", position, field, role)
            return None
        if "price_accepted" in row:
            accepted = _true(row["price_accepted"])
        else:
            frequency, quality = row.get("frequency"), row.get("price_quality_status")
            accepted = (isinstance(frequency, str) and frequency == "daily"
                        and _false(row.get("intraperiod"))
                        and isinstance(quality, str) and quality in {"accepted", "ok"})
        for flag in ("adjusted_ohlc_bounds_valid", "raw_ohlc_bounds_valid"):
            if flag in row:
                accepted = accepted and _true(row[flag])
        if not accepted:
            fail(date, mark, "source_quality_flag", position, field, role)
            return None
        value = row[field]
        if not _positive_number(value):
            fail(date, mark, "invalid_price", position, field, role)
            return None
        return float(value)

    def record_holdings(date, mark, status, marks=None, old_positions=None, stop_keys=()):
        source = dict(old_positions or {})
        source.update(positions)
        for key in sorted(source):
            position = source[key]
            units = positions[key]["units"] if key in positions else 0.0
            price = None if marks is None else marks.get(key)
            row = {"date": date, "mark": mark,
                   "status": "stopped_cash" if key in stop_keys else status,
                   "source_security_id": key, "intrinio_security_id": position["intrinio_security_id"],
                   "signal_date": position["signal_date"], "side": position["side"],
                   "units": units, "valuation_price": price,
                   "market_value": None if price is None else units * price,
                   "entry_reference": position.get("entry_reference"),
                   "stop_price": position.get("stop_price")}
            row.update({col: position.get(col) for col in extras})
            holdings.append(row)

    def blocked_row(date, mark):
        row = {col: None for col in _DAILY_COLUMNS}
        row.update(date=date, mark=mark, period_start_date=previous_date,
                   period_start_mark=previous_mark, status="failed_closed",
                   long_count=sum(p["units"] > 0 for p in positions.values()),
                   short_count=sum(p["units"] < 0 for p in positions.values()),
                   rebalance=(mark == "open" and date in schedule),
                   signal_date=schedule.get(date) if mark == "open" else None)
        daily.append(row)
        record_holdings(date, mark, "retained_unvalued_after_failure")

    marks_schedule = [(date, mark) for date in sessions
              for mark in ("open", "terminal_close" if date == terminal else "close")]
    for date, mark in marks_schedule:
        if failed:
            blocked_row(date, mark)
            previous_date, previous_mark = date, mark
            continue
        before_actions = {key: dict(position) for key, position in positions.items()}
        prior_values = {key: position["units"] * last_marks[key]
                        for key, position in before_actions.items()}
        action_cash = {"long": 0.0, "short": 0.0}
        if mark == "open":
            # Each event is a dated legal entitlement, never an inference from
            # a missing quote. Successors may lie outside the index universe.
            for event in actions_by_date.get(date, []):
                for key in sorted(list(positions)):
                    position = positions[key]
                    if position["intrinio_security_id"] != event["predecessor_id"]:
                        continue
                    previous_session_index = sessions.get_loc(date) - 1
                    if previous_session_index < 0:
                        fail(date, mark, "missing_event_predecessor_basis", position,
                             "raw_close/adj_close", "action")
                        break
                    previous_session = sessions[previous_session_index]
                    basis_row = quote_rows.get((previous_session, event["predecessor_id"]))
                    raw_close = None if basis_row is None else basis_row.get("raw_close", basis_row.get("close"))
                    adjusted_close = None if basis_row is None else basis_row.get("adj_close")
                    if not _positive_number(raw_close) or not _positive_number(adjusted_close):
                        fail(date, mark, "missing_event_predecessor_basis", position,
                             "raw_close/adj_close", "action")
                        break
                    if previous_date != previous_session or previous_mark != "close":
                        fail(date, mark, "nonconsecutive_event_predecessor_mark", position,
                             "adj_close", "action")
                        break
                    factor_before = float(adjusted_close) / float(raw_close)
                    old_units = position["units"]
                    raw_entitlement = old_units * factor_before
                    signed_cash = raw_entitlement * event["cash_per_raw_share"]
                    old_provider = position["intrinio_security_id"]
                    old_stop, old_reference = position.get("stop_price"), position.get("entry_reference")
                    new_units, successor_factor, successor_open = 0.0, None, None
                    if event["stock_ratio"] > 0:
                        successor = dict(position, intrinio_security_id=event["successor_id"])
                        successor_open = quote(date, mark, successor, "action_successor", "adj_open")
                        successor_row = quote_rows.get((date, event["successor_id"]))
                        raw_open = None if successor_row is None else successor_row.get("raw_open", successor_row.get("open"))
                        if not _positive_number(raw_open):
                            fail(date, mark, "missing_event_successor_basis", successor,
                                 "raw_open/adj_open", "action")
                        if failed:
                            break
                        successor_factor = successor_open / float(raw_open)
                        new_units = raw_entitlement * event["stock_ratio"] / successor_factor
                    # All basis checks precede mutation of this entitlement.
                    cash += signed_cash
                    action_cash[position["side"]] += signed_cash
                    if new_units:
                        position["intrinio_security_id"] = event["successor_id"]
                        position["units"] = new_units
                        position.setdefault("original_entry_reference", old_reference)
                        position.setdefault("original_stop_price", old_stop)
                        position["entry_reference"] = (old_units * old_reference - signed_cash) / new_units
                        position["stop_price"] = None if old_stop is None else (old_units * old_stop - signed_cash) / new_units
                        position["action_predecessor_id"] = old_provider
                        # A nonpositive long threshold is unreachable after the
                        # mandatory cash component; retain the value and disable
                        # only the impossible threshold test, not the holding.
                    else:
                        del positions[key]
                    applied_actions.append({
                        "date": date, "mark": mark, "event_id": event["event_id"],
                        "source_security_id": key, "predecessor_id": old_provider,
                        "successor_id": event["successor_id"], "side": position["side"],
                        "old_adjusted_units": old_units, "predecessor_factor": factor_before,
                        "signed_raw_entitlement": raw_entitlement,
                        "cash_per_raw_share": event["cash_per_raw_share"],
                        "signed_cash": signed_cash, "stock_ratio": event["stock_ratio"],
                        "successor_factor": successor_factor, "new_adjusted_units": new_units,
                        "successor_open": successor_open, "old_stop_price": old_stop,
                        "new_stop_price": position.get("stop_price") if new_units else None,
                        "old_entry_reference": old_reference,
                        "new_entry_reference": position.get("entry_reference") if new_units else None,
                        "evidence_reference": event.get("evidence_reference"),
                        "settlement_convention": "economic_entitlement_at_effective_open_cash_zero_rate",
                    })
                if failed:
                    break
        selected = selections_by_date.get(date, []) if mark == "open" and date in schedule else None
        if selected is not None:
            if {row["side"] for row in selected} != {"long", "short"}:
                fail(date, mark, "empty_long_or_short_leg", role="selection")
            elif policy.required_names_per_side is not None and any(
                    sum(row["side"] == side for row in selected) != policy.required_names_per_side
                    for side in ("long", "short")):
                fail(date, mark, "incorrect_names_per_side", role="selection")
        executable_selected = [] if selected is not None else None
        current_orders = []
        for row in selected or []:
            retired = next((event for event in reversed(retirement_definitions)
                            if event["predecessor_id"] == row["intrinio_security_id"]
                            and event["effective_date"] <= date), None)
            order = {"date": date, "signal_date": row["signal_date"],
                     "source_security_id": row["source_security_id"],
                     "intrinio_security_id": row["intrinio_security_id"], "side": row["side"],
                     "status": "unfilled_retired_security" if retired else "pending_validation",
                     "retirement_event_id": None if retired is None else retired["event_id"],
                     "intended_weight": (.5 if row["side"] == "long" else -.5) / sum(
                         member["side"] == row["side"] for member in selected),
                     "target_notional": None, "new_adjusted_units": 0.0}
            orders.append(order)
            current_orders.append(order)
            if retired is None:
                executable_selected.append(row)
        required = {key: dict(position, role="held") for key, position in positions.items()}
        identity_route_failed = False
        pending_handovers = []
        for row in executable_selected or []:
            key = row["source_security_id"]
            if key in required and required[key]["intrinio_security_id"] != row["intrinio_security_id"]:
                fail(date, mark, "held_identity_route_changed", row, role="selected")
                identity_route_failed = True
                break
            collisions = [(held_key,held) for held_key,held in positions.items()
                if held_key != key and held["intrinio_security_id"] == row["intrinio_security_id"]
                and not held.get("action_predecessor_id")]
            for held_key,held in collisions:
                admitted = _admitted_handover(handover_definitions,date=date,mark=mark,
                    old_key=held_key,held=held,selected_row=row,selected=selected)
                if admitted is None:
                    fail(date, mark, "selected_provider_held_source_collision", row, role="selected")
                    identity_route_failed = True
                    break
                pending_handovers.append((admitted,held["units"]))
            if identity_route_failed:
                break
            required.setdefault(key, dict(row, role="selected"))
        if identity_route_failed:
            # Preserve the rejected route, intended orders and actual held lots.
            # Never mark, stop, sell, transfer or replace these lots to complete a path.
            blocked_row(date, mark)
            previous_date, previous_mark = date, mark
            continue
        marks = {key: quote(date, mark, position, position["role"]) for key, position in required.items()}
        stop_fills = {}
        if not failed and policy.asset_stop_fraction is not None:
            for key, position in positions.items():
                threshold = position["stop_price"]
                if threshold is None:
                    continue
                # At scheduled monthly open, old stops are cancelled before the
                # new episode starts; all old units still mark at actual open.
                if mark == "open" and selected is None:
                    crossed = marks[key] <= threshold if position["side"] == "long" else marks[key] >= threshold
                    if crossed:
                        stop_fills[key] = "gap_open"
                elif mark != "open":
                    low = quote(date, mark, position, "held", "adj_low")
                    high = quote(date, mark, position, "held", "adj_high")
                    opening = quote(date, mark, position, "held", "adj_open")
                    if low is not None and high is not None and opening is not None and marks[key] is not None:
                        if not low <= min(opening, marks[key]) <= max(opening, marks[key]) <= high:
                            fail(date, mark, "invalid_adjusted_ohlc_bounds", position, "adj_low/adj_high", "held")
                        crossed = low <= threshold if position["side"] == "long" else high >= threshold
                        if crossed:
                            stop_fills[key] = "idealized_intraday_threshold"
                            marks[key] = threshold
        if failed:
            blocked_row(date, mark)
            previous_date, previous_mark = date, mark
            continue
        long_pnl = (sum(p["units"] * marks[key] for key, p in positions.items() if p["side"] == "long")
                    - sum(value for key, value in prior_values.items() if before_actions[key]["side"] == "long")
                    + action_cash["long"])
        short_pnl = (sum(p["units"] * marks[key] for key, p in positions.items() if p["side"] == "short")
                     - sum(value for key, value in prior_values.items() if before_actions[key]["side"] == "short")
                     + action_cash["short"])
        days = 0 if previous_date is None else (date - previous_date).days
        previous_short_value = -sum(value for key, value in prior_values.items()
                                    if before_actions[key]["side"] == "short")
        borrow_cost = previous_short_value * float(policy.borrow_annual_bps) / 10000 * days / 365
        cash -= borrow_cost
        pretrade_nav = cash + sum(p["units"] * marks[key] for key, p in positions.items())
        if not isfinite(pretrade_nav) or pretrade_nav <= 0:
            fail(date, mark, "nonpositive_or_nonfinite_nav", role="portfolio")
            blocked_row(date, mark)
            previous_date, previous_mark = date, mark
            continue
        old_positions = {key: dict(value) for key, value in before_actions.items()}
        old_positions.update({key: dict(value) for key, value in positions.items()})
        target, phase = None, None
        if selected is not None:
            nlong = sum(row["side"] == "long" for row in selected)
            nshort = len(selected) - nlong
            target = {}
            for row in executable_selected:
                key = row["source_security_id"]
                weight = .5 / nlong if row["side"] == "long" else -.5 / nshort
                reference = marks[key]
                threshold = None if policy.asset_stop_fraction is None else reference + (
                    -reference * policy.asset_stop_fraction if row["side"] == "long"
                    else reference * policy.asset_stop_fraction)
                target[key] = dict(row, units=weight * pretrade_nav / reference,
                                   entry_reference=reference, stop_price=threshold)
            for order in current_orders:
                order["target_notional"] = order["intended_weight"] * pretrade_nav
                if order["status"] != "unfilled_retired_security":
                    order["status"] = "executed"
                    order["new_adjusted_units"] = target[order["source_security_id"]]["units"]
            phase = "rebalance"
        elif stop_fills:
            target = {key: dict(position) for key, position in positions.items() if key not in stop_fills}
            phase = "stop_gap" if mark == "open" else "stop_intraday"
        if mark == "terminal_close":
            target, phase = {}, "terminal_liquidation"
        trading_cost, traded_notional = 0.0, 0.0
        if target is not None:
            instructions = []
            for key in sorted(set(positions) | set(target)):
                old_units = positions.get(key, {}).get("units", 0.0)
                new_units = target.get(key, {}).get("units", 0.0)
                delta = new_units - old_units
                if delta == 0:
                    continue
                position = target.get(key, positions.get(key))
                instructions.append((key, position, old_units, new_units, delta, delta * marks[key]))
            # Mandatory successors can coexist with a selected source lot on
            # the same provider. Net physical orders before charging costs.
            groups = {}
            for instruction in instructions:
                source_key, position = instruction[0], instruction[1]
                # Intraday stop fills need not be contemporaneous with another
                # lot's threshold fill or terminal close. Never net those
                # executions merely because they share a date/provider.
                execution_bucket = source_key if source_key in stop_fills and mark != "open" else "simultaneous_mark"
                group_key = (position["intrinio_security_id"], float(marks[source_key]), execution_bucket)
                groups.setdefault(group_key, []).append(instruction)
            for (provider, execution_price, execution_bucket), group in sorted(groups.items()):
                signed_order = sum(item[5] for item in group)
                physical_absolute = abs(signed_order)
                gross_instructions = sum(abs(item[5]) for item in group)
                cost = physical_absolute * float(policy.cost_bps) / 10000
                cash -= signed_order + cost
                trading_cost += cost
                traded_notional += physical_absolute
                physical_phase = (("stop_gap" if mark == "open" else "stop_intraday")
                                  if all(item[0] in stop_fills for item in group) else phase)
                market_trades.append({"date": date, "mark": mark, "phase": physical_phase,
                                      "intrinio_security_id": provider, "price": execution_price,
                                      "execution_bucket": execution_bucket,
                                      "signed_notional": signed_order,
                                      "absolute_notional": physical_absolute, "cost": cost,
                                      "lot_instruction_notional": gross_instructions,
                                      "internally_netted_notional": gross_instructions - physical_absolute,
                                      "source_lot_count": len(group)})
                for key, position, old_units, new_units, delta, signed in group:
                    allocated_cost = cost * abs(signed) / gross_instructions if gross_instructions else 0.0
                    trade_phase = ("stop_gap" if mark == "open" else "stop_intraday") if key in stop_fills else phase
                    row = {"date": date, "mark": mark, "phase": trade_phase,
                           "signal_date": schedule.get(date) if phase == "rebalance" else position["signal_date"],
                           "source_security_id": key, "intrinio_security_id": position["intrinio_security_id"],
                           "units_before": old_units, "units_after": new_units, "units_delta": delta,
                           "price": marks[key], "signed_notional": signed,
                           "absolute_notional": abs(signed), "cost": allocated_cost,
                           "direction": "buy" if delta > 0 else "sell",
                           "entry_reference": position.get("entry_reference"), "stop_price": position.get("stop_price"),
                           "stop_fill_assumption": stop_fills.get(key)}
                    row.update({col: position.get(col) for col in extras})
                    trades.append(row)
                    if key in stop_fills:
                        stops.append({"date": date, "mark": mark, "source_security_id": key,
                                      "intrinio_security_id": position["intrinio_security_id"],
                                      "side": position["side"], "entry_reference": position["entry_reference"],
                                      "stop_price": position["stop_price"], "fill_price": marks[key],
                                      "fill_type": stop_fills[key], "units_before": old_units, "cost": allocated_cost})
            positions = target
            for handover,old_units in pending_handovers:
                applied_handovers.append({
                    'date':date,'mark':mark,'handover_id':handover['handover_id'],
                    'evidence_registry_record_id':handover['handover_id'],
                    'old_source_security_id':handover['old_source_security_id'],
                    'new_source_security_id':handover['new_source_security_id'],
                    'intrinio_security_id':handover['intrinio_security_id'],
                    'signal_date':handover['signal_date'],'held_signal_date':handover['held_signal_date'],
                    'units_before':old_units,'units_after':target[handover['new_source_security_id']]['units'],
                    'application_status':'applied_at_monthly_open','execution_phase':phase,
                    'evidence_reference':handover['evidence_reference'],
                    'source_identity_verified':False,'source_identity_promoted':False,
                    'provisionally_admitted':True})
        long_value = sum(p["units"] * marks[key] for key, p in positions.items() if p["units"] > 0)
        short_value = sum(p["units"] * marks[key] for key, p in positions.items() if p["units"] < 0)
        nav = cash + long_value + short_value
        if not isfinite(nav) or nav <= 0:
            fail(date, mark, "nonpositive_or_nonfinite_nav_after_cost", role="portfolio")
            blocked_row(date, mark)
            previous_date, previous_mark = date, mark
            continue
        net_pnl = long_pnl + short_pnl - trading_cost - borrow_cost
        if not np.isclose(nav - previous_nav, net_pnl, rtol=1e-9, atol=1e-12 * max(1, abs(previous_nav))):
            raise DataContractError("Cash/units NAV does not reconcile to PnL attribution")
        daily.append({"date": date, "mark": mark, "period_start_date": previous_date,
                      "period_start_mark": previous_mark, "status": "ok",
                      "nav_before_cost": pretrade_nav, "nav": nav, "return": nav / previous_nav - 1,
                      "cash": cash, "long_market_value": long_value, "short_market_value": short_value,
                      "gross_exposure": (long_value - short_value) / nav,
                      "net_exposure": (long_value + short_value) / nav,
                      "long_price_pnl": long_pnl, "short_price_pnl": short_pnl,
                      "price_pnl": long_pnl + short_pnl, "trading_cost": trading_cost,
                      "borrow_cost": borrow_cost, "net_pnl": net_pnl,
                      "traded_notional": traded_notional, "turnover_one_way": traded_notional / pretrade_nav,
                      "long_count": sum(p["units"] > 0 for p in positions.values()),
                      "short_count": sum(p["units"] < 0 for p in positions.values()),
                      "rebalance": phase == "rebalance",
                      "signal_date": schedule.get(date) if phase == "rebalance" else None})
        record_holdings(date, mark, "liquidated" if mark == "terminal_close" else "held", marks,
                        old_positions=old_positions if target is not None else None, stop_keys=stop_fills)
        last_marks = {key: marks[key] for key in positions}
        previous_nav, previous_date, previous_mark = nav, date, mark

    status = "failed_closed" if failed else "complete_proxy"
    daily_frame = pd.DataFrame(daily, columns=_DAILY_COLUMNS)
    trade_frame = pd.DataFrame(trades, columns=_TRADE_COLUMNS + extras)
    holding_frame = pd.DataFrame(holdings, columns=_HOLDING_COLUMNS + extras)
    failure_frame = pd.DataFrame(failures, columns=_FAILURE_COLUMNS)
    stop_frame = pd.DataFrame(stops, columns=_STOP_COLUMNS)
    good = daily_frame.status.eq("ok")
    final_nav = float(daily_frame.iloc[-1].nav) if not failed else None
    summary = {
        "status": status, "accounting_basis": "provider_adjusted_proxy_with_verified_raw_entitlements",
        "initial_nav": float(policy.initial_nav), "final_nav": final_nav,
        "total_return": None if failed else final_nav / float(policy.initial_nav) - 1,
        "first_failure_date": None if not failed else failure_frame.iloc[0].date.date().isoformat(),
        "valid_mark_count": int(good.sum()), "expected_mark_count": len(daily_frame),
        "failure_count": len(failure_frame), "trade_count": len(trade_frame), "stop_count": len(stop_frame),
        "valid_prefix_trading_cost": float(daily_frame.loc[good, "trading_cost"].sum()),
        "valid_prefix_borrow_cost": float(daily_frame.loc[good, "borrow_cost"].sum()),
        "executed_trade_cost": float(trade_frame["cost"].sum()),
        "cost_bps": float(policy.cost_bps), "borrow_annual_bps": float(policy.borrow_annual_bps),
        "long_target_weight": .5, "short_target_weight": -.5,
        "required_names_per_side": policy.required_names_per_side,
        "asset_stop_fraction": policy.asset_stop_fraction,
        "stop_reference": "monthly_entry_adjusted_open_with_equivalent_package_threshold_after_actions",
        "stop_fill_model": "gap_at_open_else_idealized_adverse_daily_high_low_threshold",
        "stopped_name_policy": "cash_until_next_monthly_rebalance_no_replacement",
        "monthly_open_stop_priority": "cancel_old_episode_stop_then_rebalance_at_open",
        "target_capital_basis": "pre_cost_pre_trade_nav",
        "borrow_accrual": "previous_short_mark_actual_calendar_days_over_365_no_extra_intraday_day",
        "cash_interest_rate": 0.0, "funding_rate": 0.0,
        "knowledge_time_certified": False, "economic_accounting_certified": False,
        "borrow_availability_verified": False, "complete_economic_performance_available": False,
        "terminal_convention": "last_authorized_session_adjusted_close_liquidation",
        "action_count": len(applied_actions),
        "corporate_action_policy": "dated_signed_raw_entitlement_successor_hold_until_scheduled_exit",
        "action_settlement_convention": "face_value_economic_receivable_or_liability_effective_open_zero_funding",
        "action_stop_policy": "carry_original_package_threshold_net_of_cash_scaled_to_successor_units",
        "market_order_netting": "same_provider_same_mark_net_notional_costs_with_original_lots_retained",
        "source_dividends_and_splits_added_again": False,
        "fractional_successor_share_policy": "theoretical_fractional_units_retained_no_cash_in_lieu_invented",
        "borrow_netting_policy": "inherited_gross_short_lot_value_no_certified_stock_loan_offsets",
        "retired_selected_order_policy": "retain_intended_slot_unfilled_in_cash_no_replacement_no_reweighting",
        "unfilled_retired_order_count": sum(row["status"] == "unfilled_retired_security" for row in orders),
    }
    summary.update(identity_handover_count=len(applied_handovers),
        execution_identity_handover_policy=HANDOVER_POLICY,
        canonical_source_identity_promoted=False)
    return IdentityAccountingResult(daily_frame, trade_frame, holding_frame, failure_frame, status, summary,
                                  stop_frame, pd.DataFrame(applied_actions), pd.DataFrame(market_trades), pd.DataFrame(orders),
                                  pd.DataFrame(applied_handovers,columns=_HANDOVER_COLUMNS))

