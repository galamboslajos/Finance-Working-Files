"""Fail-closed adjusted-price proxy accounting for the bounded development test.

This is a cash/units ledger, not certified executable total-return accounting.
``adj_open`` is a provider-adjusted index level used as a proxy execution and
valuation price. Units stay fixed between monthly rebalances or explicit stop
exits; dividends/splits are NOT added again. Every session has separate open and
close marks; final close liquidates survivors inside the authorized period.

No availability check changes the selection list. One unavailable selected or
held quote invalidates the continuous strategy path from that point onward.
The engine retains positions and expected marks and never resumes the path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from numbers import Real
from typing import Iterable

import numpy as np
import pandas as pd

from .universe import DataContractError


@dataclass(frozen=True)
class DailyAccountingPolicy:
    """Declared development conventions, independent of forecast performance.

    Both sides receive half of the pre-cost, pre-trade NAV. Trade costs are a
    flat number of basis points per one-way absolute traded notional, including
    initial entry and final liquidation. Cash interest and funding rates are
    zero. Borrow uses the preceding mark's absolute short value, actual calendar
    days / 365 between marks; same-date open-to-close accrues no extra full day.
    Zero borrow is a diagnostic assumption, not evidence of available borrowing.
    """

    cost_bps: float = 0.0
    borrow_annual_bps: float = 0.0
    initial_nav: float = 1.0
    required_names_per_side: int | None = None
    asset_stop_fraction: float | None = None

    def __post_init__(self) -> None:
        for name in ("cost_bps", "borrow_annual_bps", "initial_nav"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
                raise DataContractError(f"{name} must be a finite number")
            if value < 0 or (name == "initial_nav" and value == 0):
                raise DataContractError(f"Invalid {name}")
        if (self.required_names_per_side is not None and
                (isinstance(self.required_names_per_side, bool)
                 or not isinstance(self.required_names_per_side, int)
                 or self.required_names_per_side < 1)):
            raise DataContractError("required_names_per_side must be a positive integer or None")
        if self.asset_stop_fraction is not None and (
                isinstance(self.asset_stop_fraction, bool)
                or not isinstance(self.asset_stop_fraction, Real)
                or not isfinite(self.asset_stop_fraction)
                or not 0 < self.asset_stop_fraction < 1):
            raise DataContractError("asset_stop_fraction must be finite, strictly between zero and one, or None")


@dataclass
class DailyAccountingResult:
    daily: pd.DataFrame
    trades: pd.DataFrame
    holdings: pd.DataFrame
    failures: pd.DataFrame
    status: str
    summary: dict
    stops: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def tables(self) -> dict[str, pd.DataFrame]:
        return {name: getattr(self, name) for name in ("daily", "trades", "holdings", "failures", "stops")}


_DAILY_COLUMNS = [
    "date", "mark", "period_start_date", "period_start_mark", "status",
    "nav_before_cost", "nav", "return", "cash", "long_market_value",
    "short_market_value", "gross_exposure", "net_exposure", "long_price_pnl",
    "short_price_pnl", "price_pnl", "trading_cost", "borrow_cost", "net_pnl",
    "traded_notional", "turnover_one_way", "long_count", "short_count",
    "rebalance", "signal_date",
]
_TRADE_COLUMNS = [
    "date", "mark", "phase", "signal_date", "source_security_id",
    "intrinio_security_id", "units_before", "units_after", "units_delta",
    "price", "signed_notional", "absolute_notional", "cost", "direction",
    "entry_reference", "stop_price", "stop_fill_assumption",
]
_HOLDING_COLUMNS = [
    "date", "mark", "status", "source_security_id", "intrinio_security_id",
    "signal_date", "side", "units", "valuation_price", "market_value",
    "entry_reference", "stop_price",
]
_FAILURE_COLUMNS = [
    "date", "mark", "reason", "source_security_id", "intrinio_security_id",
    "price_field", "role",
]
_STOP_COLUMNS = ["date", "mark", "source_security_id", "intrinio_security_id",
                 "side", "entry_reference", "stop_price", "fill_price", "fill_type",
                 "units_before", "cost"]


def _dates(values, name: str) -> pd.DatetimeIndex:
    try:
        result = pd.DatetimeIndex(pd.to_datetime(values, errors="raise"))
    except (ValueError, TypeError, OverflowError) as exc:
        raise DataContractError(f"Invalid {name}") from exc
    if result.tz is not None or result.isna().any() or not result.equals(result.normalize()):
        raise DataContractError(f"{name} must contain complete, timezone-naive dates")
    return result


def _true(value) -> bool:
    return isinstance(value, (bool, np.bool_)) and bool(value)


def _false(value) -> bool:
    return isinstance(value, (bool, np.bool_)) and not bool(value)


def _positive_number(value) -> bool:
    return (not isinstance(value, (bool, np.bool_)) and isinstance(value, Real)
            and isfinite(value) and value > 0)


def _prepare(selections, prices, sessions, rebalances, terminal_date):
    """Validate all dates before converting any numerical price value."""
    if not isinstance(selections, pd.DataFrame) or not isinstance(prices, pd.DataFrame):
        raise DataContractError("Selections and prices must be DataFrames")
    if not isinstance(rebalances, pd.DataFrame):
        raise DataContractError("Explicit rebalance schedule must be a DataFrame")
    sessions = _dates(list(sessions), "sessions")
    if len(sessions) == 0 or sessions.has_duplicates or not sessions.is_monotonic_increasing:
        raise DataContractError("Sessions must be nonempty, unique and increasing")
    terminal = _dates([terminal_date], "terminal date")[0]
    if terminal != sessions[-1]:
        raise DataContractError("Terminal date must be the final authorized session")
    if sessions.min() < pd.Timestamp("2018-01-01") or sessions.max() > pd.Timestamp("2019-12-31"):
        raise DataContractError("Development accounting is restricted to 2018–2019")
    needed_selection = {"signal_date", "execution_date", "source_security_id", "intrinio_security_id", "side"}
    if not needed_selection.issubset(selections.columns):
        raise DataContractError(f"Missing selection columns: {sorted(needed_selection - set(selections.columns))}")
    if not {"signal_date", "execution_date"}.issubset(rebalances.columns):
        raise DataContractError("Schedule requires signal_date and execution_date")
    price_date = "date" if "date" in prices.columns else "trading_date"
    needed_price = {price_date, "intrinio_security_id", "adj_open", "adj_close"}
    if not needed_price.issubset(prices.columns):
        raise DataContractError(f"Missing price columns: {sorted(needed_price - set(prices.columns))}")
    selections, prices, rebalances = selections.copy(deep=True), prices.copy(deep=True), rebalances.copy(deep=True)
    selections["execution_date"] = _dates(selections.execution_date, "selection execution dates")
    selections["signal_date"] = _dates(selections.signal_date, "selection signal dates")
    rebalances["execution_date"] = _dates(rebalances.execution_date, "rebalance execution dates")
    rebalances["signal_date"] = _dates(rebalances.signal_date, "rebalance signal dates")
    prices["date"] = _dates(prices[price_date], "price dates")
    # Guard before price conversion, even if a caller supplies out-of-scope rows
    # that happen not to be held by the selected strategy.
    if not prices["date"].isin(sessions).all():
        raise DataContractError("Price rows lie outside authorized 2018–2019 sessions")
    first_sessions = pd.Series(sessions, index=sessions.to_period("M")).groupby(level=0).first().tolist()
    if (rebalances.execution_date.duplicated().any()
            or sorted(rebalances.execution_date.tolist()) != first_sessions):
        raise DataContractError("Schedule must contain exactly each month's first authorized session")
    if not selections.execution_date.isin(rebalances.execution_date).all():
        raise DataContractError("Selection execution date absent from rebalance schedule")
    for frame in (selections, rebalances):
        if ((frame.signal_date >= frame.execution_date)
                | (frame.signal_date.dt.to_period("M") != frame.execution_date.dt.to_period("M") - 1)).any():
            raise DataContractError("Signals must precede execution in the prior calendar month")
    schedule = rebalances.set_index("execution_date").signal_date.to_dict()
    if any(row.signal_date != schedule[row.execution_date] for row in selections.itertuples()):
        raise DataContractError("Selection signal differs from the explicit schedule")
    for frame in (selections, prices):
        id_columns = ["intrinio_security_id"] + (["source_security_id"] if frame is selections else [])
        for column in id_columns:
            if frame[column].isna().any() or not frame[column].map(lambda x: isinstance(x, str) and bool(x.strip())).all():
                raise DataContractError(f"{column} must be nonempty strings")
    if selections.duplicated(["execution_date", "source_security_id"]).any():
        raise DataContractError("Duplicate selected source identity")
    if selections.duplicated(["execution_date", "intrinio_security_id"]).any():
        raise DataContractError("Selected provider identity collision")
    if prices.duplicated(["date", "intrinio_security_id"]).any():
        raise DataContractError("Duplicate provider/date price key")
    if not selections.side.isin(["long", "short"]).all():
        raise DataContractError("Selection side must be long or short")
    # A source may not change provider route inside this continuous experiment.
    if (selections.groupby("source_security_id").intrinio_security_id.nunique() > 1).any():
        raise DataContractError("Provider route changes within the continuous strategy")
    return selections, prices, sessions, schedule, terminal


def run_daily_accounting(
    selections: pd.DataFrame,
    prices: pd.DataFrame,
    sessions: Iterable,
    *,
    rebalances: pd.DataFrame,
    terminal_date,
    policy: DailyAccountingPolicy | None = None,
) -> DailyAccountingResult:
    """Account one strategy/cost/stop scenario from fixed signal-time selections.

    Selections require signal_date, execution_date, source_security_id,
    intrinio_security_id and side. The independent rebalance schedule includes
    EVERY month, including empty selections. Prices require date/trading_date,
    provider ID and adjusted open/close; the stop variant also needs adjusted
    high/low. Raw completion/quality flags, or an explicit adapter acceptance
    flag, are checked before using each required quote.

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
    if policy.asset_stop_fraction is not None and not {"adj_high", "adj_low"}.issubset(prices.columns):
        raise DataContractError("Per-asset OHLC stops require adjusted high and low")
    # This is a computational projection after the all-row date guard. It never
    # changes a dated selection: only IDs in the already frozen selection ledger
    # can become held. Do not materialize unrelated OHLCV/action columns.
    quote_columns = [col for col in (
        "date", "intrinio_security_id", "adj_open", "adj_close", "adj_high", "adj_low",
        "frequency", "intraperiod", "price_quality_status", "price_accepted",
        "adjusted_ohlc_bounds_valid", "raw_ohlc_bounds_valid",
    ) if col in prices.columns]
    requested_prices = prices.loc[prices.intrinio_security_id.isin(set(selections.intrinio_security_id)), quote_columns]
    quote_rows = {(row["date"], row["intrinio_security_id"]): row for row in requested_prices.to_dict("records")}
    selections_by_date = {date: group.to_dict("records") for date, group in selections.groupby("execution_date", sort=False)}
    extras = [col for col in selections.columns if col not in {
        "signal_date", "execution_date", "source_security_id", "intrinio_security_id", "side",
        *_HOLDING_COLUMNS, *_TRADE_COLUMNS,
    }]
    daily, trades, holdings, failures, stops = [], [], [], [], []
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

    events = [(date, mark) for date in sessions
              for mark in ("open", "terminal_close" if date == terminal else "close")]
    for date, mark in events:
        if failed:
            blocked_row(date, mark)
            previous_date, previous_mark = date, mark
            continue
        selected = selections_by_date.get(date, []) if mark == "open" and date in schedule else None
        if selected is not None:
            if {row["side"] for row in selected} != {"long", "short"}:
                fail(date, mark, "empty_long_or_short_leg", role="selection")
            elif policy.required_names_per_side is not None and any(
                    sum(row["side"] == side for row in selected) != policy.required_names_per_side
                    for side in ("long", "short")):
                fail(date, mark, "incorrect_names_per_side", role="selection")
        required = {key: dict(position, role="held") for key, position in positions.items()}
        for row in selected or []:
            key = row["source_security_id"]
            if key in required and required[key]["intrinio_security_id"] != row["intrinio_security_id"]:
                raise DataContractError("Held identity route changed")
            if any(held_key != key and held["intrinio_security_id"] == row["intrinio_security_id"]
                   for held_key, held in positions.items()):
                raise DataContractError("Selected provider collides with a different held source identity")
            required.setdefault(key, dict(row, role="selected"))
        marks = {key: quote(date, mark, position, position["role"]) for key, position in required.items()}
        stop_fills = {}
        if not failed and policy.asset_stop_fraction is not None:
            for key, position in positions.items():
                threshold = position["stop_price"]
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
        long_pnl = sum(p["units"] * (marks[key] - last_marks[key]) for key, p in positions.items() if p["units"] > 0)
        short_pnl = sum(p["units"] * (marks[key] - last_marks[key]) for key, p in positions.items() if p["units"] < 0)
        days = 0 if previous_date is None else (date - previous_date).days
        previous_short_value = -sum(p["units"] * last_marks[key] for key, p in positions.items() if p["units"] < 0)
        borrow_cost = previous_short_value * float(policy.borrow_annual_bps) / 10000 * days / 365
        cash -= borrow_cost
        pretrade_nav = cash + sum(p["units"] * marks[key] for key, p in positions.items())
        if not isfinite(pretrade_nav) or pretrade_nav <= 0:
            fail(date, mark, "nonpositive_or_nonfinite_nav", role="portfolio")
            blocked_row(date, mark)
            previous_date, previous_mark = date, mark
            continue
        old_positions = {key: dict(value) for key, value in positions.items()}
        target, phase = None, None
        if selected is not None:
            nlong = sum(row["side"] == "long" for row in selected)
            nshort = len(selected) - nlong
            target = {}
            for row in selected:
                key = row["source_security_id"]
                weight = .5 / nlong if row["side"] == "long" else -.5 / nshort
                reference = marks[key]
                threshold = None if policy.asset_stop_fraction is None else reference + (
                    -reference * policy.asset_stop_fraction if row["side"] == "long"
                    else reference * policy.asset_stop_fraction)
                target[key] = dict(row, units=weight * pretrade_nav / reference,
                                   entry_reference=reference, stop_price=threshold)
            phase = "rebalance"
        elif stop_fills:
            target = {key: dict(position) for key, position in positions.items() if key not in stop_fills}
            phase = "stop_gap" if mark == "open" else "stop_intraday"
        if mark == "terminal_close":
            target, phase = {}, "terminal_liquidation"
        trading_cost, traded_notional = 0.0, 0.0
        if target is not None:
            for key in sorted(set(positions) | set(target)):
                old_units = positions.get(key, {}).get("units", 0.0)
                new_units = target.get(key, {}).get("units", 0.0)
                delta = new_units - old_units
                if delta == 0:
                    continue
                position = target.get(key, positions.get(key))
                signed = delta * marks[key]
                absolute = abs(signed)
                cost = absolute * float(policy.cost_bps) / 10000
                cash -= signed + cost
                trading_cost += cost
                traded_notional += absolute
                trade_phase = ("stop_gap" if mark == "open" else "stop_intraday") if key in stop_fills else phase
                row = {"date": date, "mark": mark, "phase": trade_phase,
                       "signal_date": schedule.get(date) if phase == "rebalance" else position["signal_date"],
                       "source_security_id": key, "intrinio_security_id": position["intrinio_security_id"],
                       "units_before": old_units, "units_after": new_units, "units_delta": delta,
                       "price": marks[key], "signed_notional": signed,
                       "absolute_notional": absolute, "cost": cost,
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
                                  "fill_type": stop_fills[key], "units_before": old_units, "cost": cost})
            positions = target
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
        "status": status, "accounting_basis": "provider_adjusted_price_proxy_units",
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
        "stop_reference": "monthly_entry_adjusted_open_reset_even_retained_names",
        "stop_fill_model": "gap_at_open_else_idealized_adverse_daily_high_low_threshold",
        "stopped_name_policy": "cash_until_next_monthly_rebalance_no_replacement",
        "monthly_open_stop_priority": "cancel_old_episode_stop_then_rebalance_at_open",
        "target_capital_basis": "pre_cost_pre_trade_nav",
        "borrow_accrual": "previous_short_mark_actual_calendar_days_over_365_no_extra_intraday_day",
        "cash_interest_rate": 0.0, "funding_rate": 0.0,
        "knowledge_time_certified": False, "economic_accounting_certified": False,
        "borrow_availability_verified": False, "complete_economic_performance_available": False,
        "terminal_convention": "last_authorized_session_adjusted_close_liquidation",
    }
    return DailyAccountingResult(daily_frame, trade_frame, holding_frame, failure_frame, status, summary, stop_frame)
