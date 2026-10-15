"""Causal daily technical definitions, snapshotted at explicit signal sessions.

The 298 names follow the pinned Sadig e9a7a018 catalogue. V2 deliberately
requires complete exact-calendar dependency windows instead of its 80% gates.
No labels, outcome availability, membership, fills or model admission occur here.
Provider-adjusted historical values remain retrieval-time reconstructions;
causal arithmetic does not certify their historical publication availability.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import math
from typing import Sequence

import numpy as np
import pandas as pd
import pyarrow as pa

from .universe import DataContractError, iso_date


SOURCE_COMMIT = "e9a7a01881a46374694f60fc34eb7d7e4a26c2b8"
HOLDOUT_START = "2024-01-01"
ENDPOINTS = (1, 5, 10, 20, 21, 42, 63, 126, 252)
WINDOWS = (5, 10, 20, 42, 63, 126, 252)
TREND_WINDOWS = (5, 10, 20, 42, 63, 100, 126, 200, 252)
OSC_WINDOWS = (5, 14, 20, 28, 42, 63)
SKIPS = ((21, 1), (42, 5), (63, 5), (63, 21), (126, 21), (252, 21), (252, 63))
META = ("intrinio_security_id", "trading_date", "frequency", "intraperiod", "price_quality_status")
PRICE_FIELDS = ("adj_close", "adj_open", "adj_high", "adj_low", "close", "high", "low",
                "volume", "factor", "dividend", "split_ratio")
FORBIDDEN_EXACT = frozenset({"provider_adjusted_return", "return_status", "outcome_month",
                            "outcome_start", "outcome_end", "target", "label", "y",
                            "delisted", "last_price_date", "daily_backtest_ready"})


@dataclass(frozen=True)
class DailyFeaturesResult:
    table: pd.DataFrame
    coverage: pd.DataFrame
    specs: tuple[dict, ...]
    provenance: dict


def _definitions():
    """Return formulas and exact inclusive lag spans of native inputs.

    A span (field, first_lag, last_lag) requires that field at every exchange
    session in that range, not every observed stock row. Every native dependency
    also requires an accepted, completed daily quote metadata record.
    """
    entries = []
    def add(name, family, formula, spans):
        spans = tuple(spans)
        entries.append({"name": name, "family": family, "role": "technical",
                        "required_inputs": tuple(dict.fromkeys(s[0] for s in spans)),
                        "lookback_sessions": max(s[2] for s in spans) + 1,
                        "formula": formula, "_spans": spans})
    def close(n): return (("adj_close", 0, n - 1),)
    for h in ENDPOINTS:
        add(f"px_simple_return_{h}s", "security_return", f"adj_close[t]/adj_close[t-{h}]-1", close(h + 1))
        add(f"px_log_return_{h}s", "security_return", f"log(adj_close[t]/adj_close[t-{h}])", close(h + 1))
    for far, near in SKIPS:
        add(f"px_skip_return_{far}_{near}s", "security_momentum",
            f"adj_close[t-{near}]/adj_close[t-{far}]-1", (("adj_close", near, far),))
    distribution = {
        "mean_log_return": "mean(log daily return)",
        "realized_vol": "sample_std(log daily return,ddof=1)*sqrt(252)",
        "downside_vol": "sqrt(252*mean(min(log daily return,0)^2))",
        "upside_vol": "sqrt(252*mean(max(log daily return,0)^2))",
        "down_up_vol_ratio": "downside_vol/upside_vol; undefined if upside_vol=0",
        "skew": "bias-corrected Fisher-Pearson sample skewness",
        "excess_kurtosis": "bias-corrected Fisher excess sample kurtosis",
        "positive_return_share": "mean(log daily return>0)",
        "return_autocorr_1": "Pearson correlation of daily log return and its exact one-session lag",
        "max_daily_loss": "minimum daily log return (signed, not clipped)",
        "max_daily_gain": "maximum daily log return (signed, not clipped)",
    }
    for w in WINDOWS:
        for metric, formula in distribution.items():
            add(f"risk_{metric}_{w}s", "security_return_distribution", f"{formula}; trailing {w} returns",
                close(w + 2 if metric == "return_autocorr_1" else w + 1))
    trend = {
        "ma_gap": "current close / mean(close)-1",
        "ema_gap": "current close / finite weighted mean(close, weight=exp(2*j/(w+1)))-1; not recursive EMA",
        "high_distance": "current close/max(adjusted high)-1",
        "low_distance": "current close/min(adjusted low)-1",
        "drawdown": "current close/max(close)-1; distance from rolling peak, not maximum drawdown",
        "trend_slope": "OLS slope of log(close) on exact session index",
        "trend_r2": "squared Pearson correlation of log(close) and session index; undefined for constant close",
        "path_efficiency": "abs(close[t]-close[t-w])/sum(abs(daily price changes))",
    }
    for w in TREND_WINDOWS:
        for metric, formula in trend.items():
            spans = close(w + 1 if metric == "path_efficiency" else w)
            if metric in ("high_distance", "low_distance"):
                spans = (("adj_close", 0, 0), ("adj_high" if metric == "high_distance" else "adj_low", 0, w - 1))
            add(f"trend_{metric}_{w}s", "security_trend", f"{formula}; trailing {w} sessions", spans)
    for w in OSC_WINDOWS:
        add(f"osc_rsi_{w}s", "security_oscillator", f"100-100/(1+mean(gain)/mean(loss)); {w} simple averages; flat undefined", close(w + 1))
        add(f"osc_stochastic_{w}s", "security_oscillator", f"(close-min(low))/(max(high)-min(low)); {w} sessions; range 0..1",
            close(w + 1) + (("adj_high", 0, w - 1), ("adj_low", 0, w - 1)))
    ohlc = {
        "overnight_return": ("adj_open[t]/adj_close[t-1]-1", (("adj_open", 0, 0), ("adj_close", 1, 1))),
        "intraday_return": ("adj_close/adj_open-1", (("adj_open", 0, 0), ("adj_close", 0, 0))),
        "high_low_range": ("log(adj_high/adj_low)", (("adj_high", 0, 0), ("adj_low", 0, 0))),
        "true_range": ("max(high-low,abs(high-prior_close),abs(low-prior_close))/prior_close",
                       (("adj_high", 0, 0), ("adj_low", 0, 0), ("adj_close", 1, 1))),
        "close_location": ("(close-low)/(high-low)", (("adj_close", 0, 0), ("adj_high", 0, 0), ("adj_low", 0, 0))),
        "body_fraction": ("abs(close-open)/(high-low)", (("adj_close", 0, 0), ("adj_open", 0, 0), ("adj_high", 0, 0), ("adj_low", 0, 0))),
        "upper_shadow_fraction": ("(high-max(open,close))/(high-low)", (("adj_close", 0, 0), ("adj_open", 0, 0), ("adj_high", 0, 0), ("adj_low", 0, 0))),
        "lower_shadow_fraction": ("(min(open,close)-low)/(high-low)", (("adj_close", 0, 0), ("adj_open", 0, 0), ("adj_high", 0, 0), ("adj_low", 0, 0))),
        "gap_abs": ("abs(adj_open[t]/adj_close[t-1]-1)", (("adj_open", 0, 0), ("adj_close", 1, 1))),
    }
    for metric, (formula, spans) in ohlc.items():
        add(f"ohlc_{metric}_1s", "security_ohlc", formula, spans)
        for w in (5, 20, 63, 126):
            add(f"ohlc_{metric}_mean_{w}s", "security_ohlc", f"mean({formula}); trailing {w} sessions",
                tuple((field, lo, hi + w - 1) for field, lo, hi in spans))
    liquidity = {
        "log_volume": ("log(1+volume)", (("volume", 0, 0),)),
        "log_dollar_volume": ("log(1+raw_close*raw_volume)", (("close", 0, 0), ("volume", 0, 0))),
        "volume_change": ("log((1+volume[t])/(1+volume[t-1]))", (("volume", 0, 1),)),
        "dollar_volume_change": ("log((1+close[t]*volume[t])/(1+close[t-1]*volume[t-1]))", (("close", 0, 1), ("volume", 0, 1))),
        "amihud": ("abs(simple daily adjusted return)/(raw_close*raw_volume); volume>0",
                   (("adj_close", 0, 1), ("close", 0, 0), ("volume", 0, 0))),
        "abnormal_volume_ratio": ("volume/mean(volume over 252 complete sessions); not share turnover", (("volume", 0, 251),)),
        "zero_volume_share": ("indicator(volume=0)", (("volume", 0, 0),)),
        "high_low_spread_proxy": ("log(raw_high/raw_low)^2", (("high", 0, 0), ("low", 0, 0))),
        "price_impact_proxy": ("abs(simple daily adjusted return)/raw_volume; volume>0", (("adj_close", 0, 1), ("volume", 0, 0))),
    }
    for w in WINDOWS:
        for metric, (formula, spans) in liquidity.items():
            add(f"liq_{metric}_{w}s", "security_liquidity", f"mean({formula}); trailing {w} sessions",
                tuple((field, lo, hi + w - 1) for field, lo, hi in spans))
    add("event_dividend_yield_proxy", "security_event_and_level", "single-row dividend/raw_close; not annual dividend yield",
        (("dividend", 0, 0), ("close", 0, 0)))
    add("event_dividend_event", "security_event_and_level", "indicator(dividend != 0); null stays null", (("dividend", 0, 0),))
    add("event_split_event", "security_event_and_level", "indicator(split_ratio not in (0,1)); null stays null", (("split_ratio", 0, 0),))
    add("event_adjustment_factor_change", "security_event_and_level", "log(factor[t]/factor[t-1])", (("factor", 0, 1),))
    assert len(entries) == 298 and len({s["name"] for s in entries}) == 298
    return tuple(entries)


def daily_feature_specs() -> tuple[dict, ...]:
    """Frozen names, formulas and dependencies; no observed-data feature choice."""
    return tuple({key: value for key, value in spec.items() if not key.startswith("_")}
                 for spec in _definitions())


def _sequence(values, label):
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise DataContractError(f"{label} must be an explicit sequence")
    result = tuple(values)
    try:
        unique = len(set(result)) == len(result)
    except TypeError as error:
        raise DataContractError(f"{label} contains a non-scalar value") from error
    if not result or not unique:
        raise DataContractError(f"{label} must be nonempty and unique")
    return result


def _validate_input(daily_prices, sessions, signals, requested):
    names = daily_prices.column_names if hasattr(daily_prices, "column_names") else tuple(daily_prices.columns)
    if any(not isinstance(name, str) for name in names):
        raise DataContractError("Daily prices require string column names")
    if len(set(names)) != len(names):
        raise DataContractError("Daily prices contain duplicate column names")
    forbidden = [name for name in names if name.lower() in FORBIDDEN_EXACT or
                 name.lower().startswith(("target", "label", "outcome", "future", "next_", "forward", "prediction", "realized_", "class_label"))]
    if forbidden:
        raise DataContractError(f"Feature worker rejects target/outcome columns: {', '.join(forbidden)}")
    missing = set(META) - set(names)
    if missing:
        raise DataContractError(f"Daily prices missing metadata columns: {', '.join(sorted(missing))}")
    # Read only date/ID metadata before rejecting future/protected financial rows.
    if hasattr(daily_prices, "select"):
        keys = daily_prices.select(["intrinio_security_id", "trading_date"]).to_pylist()
    elif isinstance(daily_prices, pd.DataFrame):
        keys = daily_prices.loc[:, ["intrinio_security_id", "trading_date"]].to_dict("records")
    else:
        raise DataContractError("Daily prices must be an Arrow table or pandas DataFrame")
    seen = set()
    session_set, requested_set = set(sessions), set(requested)
    for row in keys:
        sid, day = row["intrinio_security_id"], iso_date(row["trading_date"], "price trading_date")
        if not isinstance(sid, str) or not sid.strip() or sid not in requested_set:
            raise DataContractError("Daily prices include an invalid or unrequested security ID")
        if day >= HOLDOUT_START or day > max(signals):
            raise DataContractError("Daily feature input contains future or protected financial rows")
        if day not in session_set:
            raise DataContractError(f"Daily quote is outside the exact supplied sessions: {day}")
        if (sid, day) in seen:
            raise DataContractError("Duplicate daily price security/session keys")
        seen.add((sid, day))
    columns = list(META) + [name for name in PRICE_FIELDS if name in names]
    if "adjusted_ohlc_bounds_valid" in names:
        columns.append("adjusted_ohlc_bounds_valid")
    # Financial conversion happens only after the preceding metadata guards.
    frame = daily_prices.select(columns).to_pandas() if hasattr(daily_prices, "select") else daily_prices.loc[:, columns].copy(deep=True)
    frame["trading_date"] = [iso_date(value, "price trading_date") for value in frame["trading_date"]]
    return frame


def _numeric(series, *, positive=False, nonnegative=False):
    # Booleans and coercible text are not financial observations.
    values = [float(value) if isinstance(value, (int, float, np.integer, np.floating))
              and not isinstance(value, (bool, np.bool_)) else np.nan for value in series]
    result = pd.Series(values, index=series.index, dtype=float)
    valid = np.isfinite(result)
    if positive: valid &= result > 0
    if nonnegative: valid &= result >= 0
    return result.where(valid)


def _strict_mean(series, w):
    return series.rolling(w, min_periods=w).mean()


def _technical_series(native):
    """All operations are backward-looking on the uncompressed session grid."""
    result = {}
    c, o, h, l = (native[name] for name in ("adj_close", "adj_open", "adj_high", "adj_low"))
    v, raw_c = native["volume"], native["close"]
    previous = c.shift(1)
    simple, log_r, change = c / previous - 1, np.log(c / previous), c - previous
    def divide(numerator, denominator): return numerator / denominator.where(denominator != 0)
    for lag in ENDPOINTS:
        complete = c.rolling(lag + 1, min_periods=lag + 1).count() == lag + 1
        result[f"px_simple_return_{lag}s"] = (c / c.shift(lag) - 1).where(complete)
        result[f"px_log_return_{lag}s"] = np.log(c / c.shift(lag)).where(complete)
    for far, near in SKIPS:
        complete = c.shift(near).rolling(far - near + 1, min_periods=far - near + 1).count() == far - near + 1
        result[f"px_skip_return_{far}_{near}s"] = (c.shift(near) / c.shift(far) - 1).where(complete)
    for w in WINDOWS:
        rolling = log_r.rolling(w, min_periods=w)
        nonconstant = rolling.max() > rolling.min()
        down = np.sqrt(252 * _strict_mean(log_r.clip(upper=0) ** 2, w))
        up = np.sqrt(252 * _strict_mean(log_r.clip(lower=0) ** 2, w))
        metrics = {
            "mean_log_return": rolling.mean(), "realized_vol": rolling.std(ddof=1) * math.sqrt(252),
            "downside_vol": down, "upside_vol": up, "down_up_vol_ratio": divide(down, up),
            "skew": rolling.skew().where(nonconstant), "excess_kurtosis": rolling.kurt().where(nonconstant),
            "positive_return_share": _strict_mean((log_r > 0).astype(float).where(log_r.notna()), w),
            "return_autocorr_1": rolling.corr(log_r.shift(1)),
            "max_daily_loss": rolling.min(), "max_daily_gain": rolling.max(),
        }
        for metric, value in metrics.items(): result[f"risk_{metric}_{w}s"] = value
    for w in TREND_WINDOWS:
        rolling = c.rolling(w, min_periods=w)
        weights = np.exp((np.arange(w) - (w - 1)) * 2.0 / (w + 1))
        finite_ema = rolling.apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
        x = np.arange(w, dtype=float)
        centered = x - x.mean()
        denom = float(np.dot(centered, centered))
        lc = np.log(c)
        slope = lc.rolling(w, min_periods=w).apply(lambda y: np.dot(centered, y) / denom, raw=True)
        def r2(y):
            centered_y = y - y.mean()
            y_ss = np.dot(centered_y, centered_y)
            return np.dot(centered, centered_y) ** 2 / (denom * y_ss) if y_ss > 0 else np.nan
        metrics = {
            "ma_gap": c / rolling.mean() - 1, "ema_gap": c / finite_ema - 1,
            "high_distance": c / h.rolling(w, min_periods=w).max() - 1,
            "low_distance": c / l.rolling(w, min_periods=w).min() - 1,
            "drawdown": c / rolling.max() - 1, "trend_slope": slope,
            "trend_r2": lc.rolling(w, min_periods=w).apply(r2, raw=True),
            "path_efficiency": divide((c - c.shift(w)).abs(), change.abs().rolling(w, min_periods=w).sum()),
        }
        for metric, value in metrics.items(): result[f"trend_{metric}_{w}s"] = value
    for w in OSC_WINDOWS:
        gain, loss = _strict_mean(change.clip(lower=0), w), _strict_mean(-change.clip(upper=0), w)
        rsi = 100 - 100 / (1 + divide(gain, loss))
        rsi = rsi.where(~((loss == 0) & (gain > 0)), 100.)
        result[f"osc_rsi_{w}s"] = rsi
        high, low = h.rolling(w, min_periods=w).max(), l.rolling(w, min_periods=w).min()
        result[f"osc_stochastic_{w}s"] = divide(c - low, high - low).where(change.rolling(w, min_periods=w).count() == w)
    spread = h - l
    ohlc = {
        "overnight_return": o / previous - 1, "intraday_return": c / o - 1,
        "high_low_range": np.log(h / l),
        "true_range": pd.concat((h - l, (h - previous).abs(), (l - previous).abs()), axis=1).max(axis=1, skipna=False) / previous,
        "close_location": divide(c - l, spread), "body_fraction": divide((c - o).abs(), spread),
        "upper_shadow_fraction": divide(h - pd.concat((o, c), axis=1).max(axis=1, skipna=False), spread),
        "lower_shadow_fraction": divide(pd.concat((o, c), axis=1).min(axis=1, skipna=False) - l, spread),
        "gap_abs": (o / previous - 1).abs(),
    }
    for metric, value in ohlc.items():
        result[f"ohlc_{metric}_1s"] = value
        for w in (5, 20, 63, 126): result[f"ohlc_{metric}_mean_{w}s"] = _strict_mean(value, w)
    notional = raw_c * v
    liq = {
        "log_volume": np.log1p(v), "log_dollar_volume": np.log1p(notional),
        "volume_change": np.log((1 + v) / (1 + v.shift(1))),
        "dollar_volume_change": np.log((1 + notional) / (1 + notional.shift(1))),
        "amihud": divide(simple.abs(), notional.where(v > 0)),
        "abnormal_volume_ratio": divide(v, _strict_mean(v, 252)),
        "zero_volume_share": (v == 0).astype(float).where(v.notna()),
        "high_low_spread_proxy": np.log(native["high"] / native["low"]) ** 2,
        "price_impact_proxy": divide(simple.abs(), v.where(v > 0)),
    }
    for w in WINDOWS:
        for metric, value in liq.items(): result[f"liq_{metric}_{w}s"] = _strict_mean(value, w)
    d, split, factor = native["dividend"], native["split_ratio"], native["factor"]
    result["event_dividend_yield_proxy"] = d / raw_c
    result["event_dividend_event"] = (d != 0).astype(float).where(d.notna())
    result["event_split_event"] = (~split.isin((0, 1))).astype(float).where(split.notna())
    result["event_adjustment_factor_change"] = np.log(factor / factor.shift(1))
    return result


def compute_daily_features(daily_prices, session_dates, *, signal_dates, requested_security_ids) -> DailyFeaturesResult:
    """Compute daily histories first, then emit every requested ID/signal pair.

    Input rows after the latest signal are rejected before financial conversion.
    Earlier snapshots are unaffected by later rows in this same bounded input.
    Missing securities and gaps survive as diagnostic records; no future quote
    determines eligibility. The caller supplies an independently pinned calendar
    and signal-time identity/cohort rules, which this arithmetic worker cannot
    independently certify. Signals are close-of-session information timestamps.
    """
    sessions = tuple(iso_date(day, "session date") for day in _sequence(session_dates, "session_dates"))
    if sessions != tuple(sorted(sessions)):
        raise DataContractError("session_dates must be in increasing exact-session order")
    signals = tuple(iso_date(day, "signal date") for day in _sequence(signal_dates, "signal_dates"))
    if any(day >= HOLDOUT_START or day not in set(sessions) for day in signals):
        raise DataContractError("Signals must be supplied pre-holdout exchange sessions")
    requested = _sequence(requested_security_ids, "requested_security_ids")
    if any(not isinstance(sid, str) or not sid.strip() for sid in requested):
        raise DataContractError("Invalid requested security ID")
    frame = _validate_input(daily_prices, sessions, signals, requested)
    grid = tuple(day for day in sessions if day <= max(signals))
    positions = {day: index for index, day in enumerate(grid)}
    definitions = _definitions()
    rows, coverage = [], []
    for sid in requested:
        stock = frame.loc[frame["intrinio_security_id"] == sid].set_index("trading_date").reindex(grid)
        present = stock["intrinio_security_id"].notna().to_numpy()
        quality = ((stock["frequency"] == "daily") & stock["intraperiod"].map(lambda v: isinstance(v, (bool, np.bool_)) and not v)
                   & stock["price_quality_status"].isin(("accepted", "ok"))).to_numpy(dtype=bool)
        native = {}
        for name in PRICE_FIELDS:
            values = stock[name] if name in stock else pd.Series(np.nan, index=stock.index)
            native[name] = _numeric(values, positive=name not in ("volume", "dividend", "split_ratio"),
                                    nonnegative=name in ("volume", "split_ratio")).where(quality)
        # An invalid OHLC range blocks features which consume the offending range,
        # never a separately valid close-only momentum/return observation.
        for prefix in ("adj_", ""):
            high_name, low_name = prefix + "high", prefix + "low"
            high, low = native[high_name], native[low_name]
            bounds = high.notna() & low.notna() & (high < low)
            if prefix:
                bad_open = pd.Series(False, index=stock.index)
                for field in ("adj_open", "adj_close"):
                    value = native[field]
                    bounds |= high.notna() & value.notna() & (high < value)
                    bounds |= low.notna() & value.notna() & (low > value)
                    if field == "adj_open":
                        bad_open |= high.notna() & value.notna() & (high < value)
                        bad_open |= low.notna() & value.notna() & (low > value)
                if "adjusted_ohlc_bounds_valid" in stock:
                    flagged = stock["adjusted_ohlc_bounds_valid"].map(lambda x: isinstance(x, (bool, np.bool_)) and not x)
                    bounds |= flagged
                    bad_open |= flagged
                native["adj_open"] = native["adj_open"].mask(bad_open)
            native[high_name] = high.mask(bounds)
            native[low_name] = low.mask(bounds)
        values = _technical_series(native)
        valid_fields = {name: value.notna().to_numpy() for name, value in native.items()}
        for signal in signals:
            idx = positions[signal]
            output = {"intrinio_security_id": sid, "signal_date": signal}
            for spec in definitions:
                required = {}
                for field, lo, hi in spec["_spans"]:
                    for lag in range(lo, hi + 1): required.setdefault(idx - lag, set()).add(field)
                available_positions = [pos for pos in required if pos >= 0]
                observed = sum(present[pos] and quality[pos] and all(valid_fields[field][pos] for field in required[pos])
                               for pos in available_positions)
                value = float(values[spec["name"]].iloc[idx])
                complete = observed == len(required)
                if min(required) < 0:
                    status = "insufficient_calendar_warmup"
                elif any(not present[pos] for pos in required):
                    status = "missing_session"
                elif any(not quality[pos] for pos in required):
                    status = "invalid_quote_metadata"
                elif not complete:
                    status = "missing_or_invalid_input"
                elif not math.isfinite(value):
                    status = "undefined_formula"
                else:
                    status = "ok"
                output[spec["name"]] = value if status == "ok" else np.nan
                coverage.append({"intrinio_security_id": sid, "signal_date": signal,
                                 "feature_name": spec["name"], "status": status,
                                 "expected_sessions": len(required), "observed_sessions": int(observed),
                                 "required_start_date": grid[min(required)] if min(required) >= 0 else None,
                                 "latest_input_date": grid[max(required)] if max(required) >= 0 else None})
            rows.append(output)
    specs = daily_feature_specs()
    return DailyFeaturesResult(pd.DataFrame(rows, columns=["intrinio_security_id", "signal_date"] + [s["name"] for s in specs]),
                               pd.DataFrame(coverage), specs,
                               {"source_commit": SOURCE_COMMIT, "feature_count": len(specs),
                                "window_policy": "STRICT_COMPLETE_EXACT_SESSION_DEPENDENCIES_V1",
                                "signal_timing": "AFTER_SESSION_CLOSE; execution convention remains separate",
                                "signal_dates": list(signals), "input_through": max(signals),
                                "requested_security_ids": list(requested), "calendar_sessions": len(grid),
                                "future_rows_rejected": True, "target_columns_rejected": True,
                                "historical_reconstruction": True, "knowledge_time_verified": False,
                                "adjustment_vintage_verified": False, "model_ready": False})
