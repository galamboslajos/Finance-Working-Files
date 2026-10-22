"""Fast, isolated stock-path accounting; no source IO or portfolio selection.

The half-capital asset plus flat opposite cash reference is retained as the
arithmetic convention of the bounded audit. Returned wealth is 1+2*(NAV-1).
No rounding, clipping, replacement prices or missing-path imputation occurs.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from math import isfinite
from numbers import Real

from .universe import DataContractError

ARITHMETIC_VERSION = "scalar_half_nav_cash_reference_v1"
CASH_LONG = "__dmv2_path_cash_reference_long__"
CASH_SHORT = "__dmv2_path_cash_reference_short__"
_CASH = {CASH_LONG, CASH_SHORT}
_FIELDS = ("intrinio_security_id", "trading_date", "adj_open", "adj_close", "adj_high", "adj_low",
    "open", "close", "raw_open", "raw_close", "frequency", "intraperiod", "price_quality_status",
    "price_accepted", "adjusted_ohlc_bounds_valid", "raw_ohlc_bounds_valid")


def _need(value, message):
    if not value:
        raise DataContractError(message)


def _day(value, financial=False):
    _need(type(value) is str and date.fromisoformat(value).isoformat() == value, "Canonical path date required")
    if financial:
        _need("2010-01-01" <= value < "2024-01-01", "Unauthorized financial path date")
    return value


def _true(value):
    return type(value).__name__ in {"bool", "bool_"} and bool(value)


def _false(value):
    return type(value).__name__ in {"bool", "bool_"} and not bool(value)


def _positive(value):
    return not _is_bool(value) and isinstance(value, Real) and isfinite(value) and value > 0


def _is_bool(value):
    return type(value).__name__ in {"bool", "bool_"}


def _records(value, columns):
    if hasattr(value, "column_names"):
        return value.select(list(columns)).to_pylist()
    if hasattr(value, "iloc"):
        import pyarrow as pa
        return pa.Table.from_pandas(value.loc[:, list(columns)], preserve_index=False).to_pylist()
    return [{k: row.get(k) for k in columns} for row in value]


@dataclass(frozen=True)
class PriceIndex:
    quotes: dict
    sessions: tuple
    positions: dict
    fields: frozenset


@dataclass(frozen=True)
class ActionRegistry:
    by_date: dict
    retirements: tuple


def prepare_price_index(prices, sessions):
    """Validate ALL date/ID keys before converting any financial field.

    Caller owns bounded Arrow filtering and independent-session provenance.
    Calendar-only 2009/2024 dates are allowed; price rows are 2010--2023 only.
    """
    sessions = tuple(_day(d) for d in sessions)
    _need(sessions and tuple(sorted(set(sessions))) == sessions, "Independent sessions must be sorted and unique")
    names = set(prices.column_names if hasattr(prices, "column_names") else
        prices.columns if hasattr(prices, "columns") else (k for row in prices for k in row))
    date_name = "trading_date" if "trading_date" in names else "date"
    _need({date_name, "intrinio_security_id", "adj_open", "adj_close"} <= names, "Missing required path-price fields")
    keys = _records(prices, (date_name, "intrinio_security_id"))
    unique = set()
    allowed = set(sessions)
    for row in keys:
        day, native = _day(row[date_name], True), row["intrinio_security_id"]
        _need(type(native) is str and native.strip() == native and native and native not in _CASH,
            "Invalid real price route or cash-control collision")
        _need(day in allowed, "Price row outside independently authorized sessions")
        key = day, native
        _need(key not in unique, "Duplicate provider/date price key")
        unique.add(key)
    columns = [field for field in _FIELDS if field in names]
    if date_name not in columns:
        columns.append(date_name)
    rows = _records(prices, columns)
    quotes = {}
    for row in rows:
        row["trading_date"] = row[date_name]
        quotes[row["trading_date"], row["intrinio_security_id"]] = row
    return PriceIndex(quotes, sessions, {day: i for i, day in enumerate(sessions)}, frozenset(names))


def prepare_action_registry(events=(), historical_retirements=(), *, sessions):
    """Freeze exact evidenced legal actions, with generic 2010--2023 clocks."""
    events = events.to_dict("records") if hasattr(events, "iloc") else list(events)
    historical = list(historical_retirements)
    allowed = set(sessions)
    current, retired = [], []
    for original in events:
        row = dict(original)
        for field in ("event_id", "predecessor_id", "evidence_reference"):
            _need(type(row.get(field)) is str and row[field].strip(), "Action requires explicit identity/evidence")
        day = _day(row.get("effective_date"))
        _need(day < "2024-01-01" and day in allowed and _true(row.get("verified")), "Action outside evidenced session scope")
        _need(row["predecessor_id"] not in _CASH and row.get("successor_id") not in _CASH,
            "Research actions cannot apply to synthetic cash")
        current.append(row)
    _need(len({r["event_id"] for r in current}) == len(current), "Duplicate corporate-action identity")
    _need(len({(r["effective_date"], r["predecessor_id"]) for r in current}) == len(current), "Conflicting corporate-action date")
    for row in current:
        _need(not any(other["effective_date"] == row["effective_date"] and
            other["predecessor_id"] == row.get("successor_id") for other in current), "Same-session action chain requires explicit ordering")
    # All event dates precede monetary conversion.
    for row in current:
        for field in ("cash_per_raw_share", "stock_ratio"):
            value = row.get(field)
            _need(not _is_bool(value) and isinstance(value, Real) and isfinite(value) and value >= 0,
                "Action financial terms must be finite nonnegative")
            row[field] = float(value)
        _need(row["cash_per_raw_share"] > 0 or row["stock_ratio"] > 0, "Invented zero action payoff")
        successor = row.get("successor_id")
        if row["stock_ratio"]:
            _need(type(successor) is str and successor.strip() and successor != row["predecessor_id"], "Stock action requires distinct successor")
        else:
            _need(successor is None, "Cash-only action cannot carry successor")
    for original in historical:
        row = dict(original)
        _need(row.get("verified") is True and all(type(row.get(k)) is str and row[k].strip()
            for k in ("event_id", "predecessor_id", "evidence_reference")), "Unverified historical retirement")
        _need("1962-01-01" <= _day(row.get("effective_date")) < "2024-01-01", "Historical retirement date outside scope")
        _need(row["predecessor_id"] not in _CASH, "Cash control cannot be retired")
        retired.append(row)
    _need(len({r["event_id"] for r in retired}) == len(retired) and
        len({r["predecessor_id"] for r in retired}) == len(retired), "Duplicate historical retirement")
    _need(not {r["predecessor_id"] for r in retired} & {r["predecessor_id"] for r in current}, "Retired native reused in action registry")
    by_date = {}
    for row in sorted(current, key=lambda r: (r["effective_date"], r["event_id"])):
        by_date.setdefault(row["effective_date"], []).append(row)
    return ActionRegistry(by_date, tuple(sorted(current + retired, key=lambda r: (r["effective_date"], r["event_id"]))))


class _PathFailure(Exception):
    pass


def replay_unit_episode(index, *, source_security_id, intrinio_security_id, signal_date,
        window, side, path_variant, events=(), historical_retirements=(), registry=None,
        retain_signed_wealth=False):
    """O(sessions) signed one-stock path, matching frozen zero-cost conventions."""
    _need(isinstance(index, PriceIndex), "Prepared immutable price index required")
    _need(type(retain_signed_wealth) is bool, "Explicit signed-unit label domain required")
    _need(side in {"long", "short"} and path_variant in {"unstopped", "asset_stop_10pct"}, "Unknown directional path variant")
    _need(all(type(k) is str and k.strip() and k not in _CASH for k in (source_security_id, intrinio_security_id)), "Invalid original path identity")
    signal, entry, exit_day = _day(signal_date), _day(window["entry_date"]), _day(window["exit_date"])
    _need(signal == window["signal_date"] and signal < entry < exit_day, "Path window/signal mismatch")
    _need("2010-01-01" <= entry < "2024-01-01", "Entry outside 2010--2023")
    if exit_day >= "2024-01-01":
        return _unavailable("protected_scheduled_endpoint", retain_signed_wealth)
    required = window["required_marks"]
    _need(required and required[0] == {"date": entry, "mark": "open"} and
        required[-1] == {"date": exit_day, "mark": "open"}, "Exact scheduled next-open marks required")
    _need(entry in index.positions and exit_day in index.positions, "Window absent from independent calendar")
    days = index.sessions[index.positions[entry]:index.positions[exit_day] + 1]
    expected = [{"date": d, "mark": m} for d in days[:-1] for m in ("open", "close")] + [{"date": exit_day, "mark": "open"}]
    _need(required == expected, "Missing or extra required economic mark")
    registry = registry or prepare_action_registry(events, historical_retirements, sessions=index.sessions)
    _need(isinstance(registry, ActionRegistry), "Prepared legal action registry required")
    retired = next((r for r in reversed(registry.retirements) if r["predecessor_id"] == intrinio_security_id and r["effective_date"] <= entry), None)
    stopped = path_variant == "asset_stop_10pct"
    _need(not stopped or {"adj_high", "adj_low"} <= index.fields, "Per-asset OHLC stops require adjusted high and low")
    dummy = CASH_SHORT if side == "long" else CASH_LONG
    sign = 1. if side == "long" else -1.
    cash, unit, dummy_unit, native, threshold, reference = 1., 0., 0., intrinio_security_id, None, None
    held, marks, failures, stops, actions = False, [], [], [], []
    failed, capital = False, False
    would_trigger, negative_retained = False, 0

    def fail(day, mark, reason, field=None, role=None, native_override=None):
        nonlocal capital
        capital |= "nav" in reason or "capital" in reason
        failures.append({"date": day, "mark": mark, "reason": reason,
            "source_security_id": source_security_id if role != "portfolio" else None,
            "intrinio_security_id": (native_override or native) if role != "portfolio" else None,
            "price_field": field, "role": role})
        raise _PathFailure()

    def quote(day, mark, field, role="held", provider=None, soft=False):
        provider = provider or native
        def rejected(reason):
            try:
                fail(day, mark, reason, field, role, provider)
            except _PathFailure:
                if not soft:
                    raise
                return None
        row = index.quotes.get((day, provider))
        if row is None:
            return rejected("missing_quote")
        accepted = _true(row["price_accepted"]) if "price_accepted" in row else (
            isinstance(row.get("frequency"), str) and row["frequency"] == "daily" and _false(row.get("intraperiod")) and
            isinstance(row.get("price_quality_status"), str) and row["price_quality_status"] in {"accepted", "ok"})
        for flag in ("adjusted_ohlc_bounds_valid", "raw_ohlc_bounds_valid"):
            if flag in row:
                accepted = accepted and _true(row[flag])
        if not accepted:
            return rejected("source_quality_flag")
        value = row.get(field)
        if not _positive(value):
            return rejected("invalid_price")
        return float(value)

    def nav(price):
        # Exact legacy addition order: cash + long market value + short value.
        return cash + (unit * price if unit > 0 else max(dummy_unit, 0.)) + (unit * price if unit < 0 else min(dummy_unit, 0.))

    for item in required:
        day, mark = item["date"], item["mark"]
        try:
            if mark == "open" and held:
                for event in registry.by_date.get(day, ()):
                    if event["predecessor_id"] != native:
                        continue
                    prior = index.sessions[index.positions[day] - 1]
                    basis = index.quotes.get((prior, native))
                    raw = None if basis is None else basis.get("raw_close", basis.get("close"))
                    adjusted = None if basis is None else basis.get("adj_close")
                    if not _positive(raw) or not _positive(adjusted):
                        fail(day, mark, "missing_event_predecessor_basis", "raw_close/adj_close", "action")
                    if not marks or marks[-1]["date"] != prior or marks[-1]["mark"] != "close":
                        fail(day, mark, "nonconsecutive_event_predecessor_mark", "adj_close", "action")
                    factor = float(adjusted) / float(raw)
                    old_units, old_native, old_reference, old_threshold = unit, native, reference, threshold
                    entitlement = old_units * factor
                    signed_cash = entitlement * event["cash_per_raw_share"]
                    new_units = 0.
                    if event["stock_ratio"] > 0:
                        successor = event["successor_id"]
                        opening = quote(day, mark, "adj_open", "action_successor", successor, soft=True)
                        row = index.quotes.get((day, successor))
                        raw_open = None if row is None else row.get("raw_open", row.get("open"))
                        if not _positive(raw_open):
                            fail(day, mark, "missing_event_successor_basis", "raw_open/adj_open", "action", successor)
                        if opening is None:
                            raise _PathFailure()
                        new_units = entitlement * event["stock_ratio"] / (opening / float(raw_open))
                    cash += signed_cash
                    unit = new_units
                    if new_units:
                        native = event["successor_id"]
                        reference = (old_units * old_reference - signed_cash) / new_units
                        threshold = None if old_threshold is None else (old_units * old_threshold - signed_cash) / new_units
                    else:
                        held = False
                    actions.append({"date": day, "mark": mark, "event_id": event["event_id"],
                        "predecessor_id": old_native, "successor_id": event.get("successor_id"), "signed_cash": signed_cash})
            price = 1.
            if day == entry and mark == "open":
                if retired is None:
                    price = quote(day, mark, "adj_open", "selected")
                unit = 0. if retired else sign * .5 / price
                dummy_unit = -sign * .5
                # Synthetic provider sorts before all admitted real routes.
                for _, amount in sorted(((dummy, dummy_unit), (intrinio_security_id, unit * price))):
                    cash -= amount
                held = retired is None
                reference = price
                threshold = None if not stopped else reference + (-reference * .1 if side == "long" else reference * .1)
            elif held:
                price = quote(day, mark, "adj_open" if mark == "open" else "adj_close")
                fill_type = None
                if stopped and threshold is not None and day != exit_day:
                    if mark == "open":
                        crossed = price <= threshold if side == "long" else price >= threshold
                        if crossed:
                            fill_type = "gap_open"
                    else:
                        low = quote(day, mark, "adj_low", soft=True)
                        high = quote(day, mark, "adj_high", soft=True)
                        opening = quote(day, mark, "adj_open", soft=True)
                        if low is None or high is None or opening is None:
                            raise _PathFailure()
                        if not low <= min(opening, price) <= max(opening, price) <= high:
                            fail(day, mark, "invalid_adjusted_ohlc_bounds", "adj_low/adj_high", "held")
                        crossed = low <= threshold if side == "long" else high >= threshold
                        if crossed:
                            price, fill_type = threshold, "idealized_intraday_threshold"
                if fill_type:
                    stops.append({"date": day, "mark": mark, "fill_type": fill_type, "fill_price": price,
                        "stop_price": threshold, "intrinio_security_id": native})
            # The frozen engine uses position insertion order (real, dummy)
            # for pretrade capital and long/short order for its output mark.
            # Preserve both operation orders: a one-ULP exit difference can
            # otherwise change exact midpoint ties despite economic equality.
            value = cash + sum((unit * price, dummy_unit) if held else (dummy_unit,))
            pretrade_nonpositive = value <= 0
            if not isfinite(value) or (value <= 0 and not retain_signed_wealth):
                fail(day, mark, "nonpositive_or_nonfinite_nav", role="portfolio")
            if value <= 0:
                would_trigger = True
                negative_retained += 1
            if day != entry or mark != "open":
                if held and stops and stops[-1]["date"] == day and stops[-1]["mark"] == mark:
                    cash -= -unit * price
                    unit, held = 0., False
                    value = nav(1.)
                if day == exit_day:
                    # Preserve the existing cash-only replacement-open operation
                    # order; no real exit-close value is read.
                    # Finite signed-unit losses have no positive-capital gate.
                    # When the compatibility reference is nonpositive, close
                    # existing lots to flat cash, never reverse dummy weights.
                    targets = {} if retain_signed_wealth and value <= 0 else {
                        CASH_LONG: .5 * value, CASH_SHORT: -.5 * value}
                    old = {dummy: dummy_unit}
                    if held:
                        old[source_security_id] = unit
                    grouped = []
                    for key in sorted(set(old) | set(targets)):
                        provider = native if key == source_security_id else key
                        execution_price = price if key == source_security_id else 1.
                        delta = targets.get(key, 0.) - old.get(key, 0.)
                        if delta:
                            grouped.append((provider, delta * execution_price))
                    for _, amount in sorted(grouped):
                        cash -= amount
                    unit, held = 0., False
                    dummy_unit = 0.
                    value = cash + targets.get(CASH_LONG, 0.) + targets.get(CASH_SHORT, 0.)
                else:
                    value = nav(price if held else 1.)
            else:
                value = nav(price)
            if not isfinite(value) or (value <= 0 and not retain_signed_wealth):
                fail(day, mark, "nonpositive_or_nonfinite_nav_after_cost", role="portfolio")
            if value <= 0 and not pretrade_nonpositive:
                would_trigger = True
                negative_retained += 1
            marks.append({"date": day, "mark": mark, "wealth": 1. + 2. * (value - 1.)})
        except _PathFailure:
            failed = True
            break
    reason = ";".join(sorted({r["reason"] for r in failures}))
    if retired:
        reason = "retired_before_entry_unfilled_not_zero_payoff"
    available = not failed and retired is None and len(marks) == len(required)
    return {"requested": True, "status": "complete" if available else "unavailable", "reason": reason,
        "marks": marks, "stop_count": len(stops), "action_count": len(actions), "unfilled_entry": retired is not None,
        "engine_status": "failed_closed" if failed else "complete_proxy", "failures": failures,
        "engine_capital_gate_triggered": capital, "stops": stops, "actions": actions,
        "legacy_capital_gate_would_trigger": bool(would_trigger or capital),
        "negative_reference_nav_retained": negative_retained,
        "arithmetic_version": ARITHMETIC_VERSION + ("_signed_unit_domain" if retain_signed_wealth else "_legacy_positive_reference_domain")}


def _unavailable(reason, retain_signed_wealth=False):
    return {"requested": True, "status": "unavailable", "reason": reason, "marks": [], "stop_count": 0,
        "action_count": 0, "unfilled_entry": False, "engine_status": "not_run_unavailable_endpoint", "failures": [],
        "engine_capital_gate_triggered": False, "stops": [], "actions": [],
        "legacy_capital_gate_would_trigger": False, "negative_reference_nav_retained": 0,
        "arithmetic_version": ARITHMETIC_VERSION + ("_signed_unit_domain" if retain_signed_wealth else "_legacy_positive_reference_domain")}
