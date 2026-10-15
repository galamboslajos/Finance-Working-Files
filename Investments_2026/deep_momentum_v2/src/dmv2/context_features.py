"""Past-only context candidates with original historical cohort denominators.

Definitions follow the pinned Sadig monthly catalogue. V2 deliberately requires
complete calendar windows and complete source cohorts, instead of silently
normalizing a requested subset. Missing daily membership is never reconstructed
from month-end membership or future quote availability. Numerical calculation
does not certify retrieval-time identity, price revisions or terminal accounting.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
from typing import Sequence

import numpy as np
import pandas as pd

from .universe import DataContractError, UniverseSnapshot
from .identity_policy import (STRICT_IDENTITY_POLICY, identity_policy_provenance,
                              qualify_identity_records, validate_identity_policy)


SOURCE_COMMIT = "e9a7a01881a46374694f60fc34eb7d7e4a26c2b8"
MARKET_HORIZONS = (1, 5, 20, 63, 126, 252)
RELATIVE_WINDOWS = (20, 42, 63, 126, 252)
SLEEVES = ("sp500", "sp400")
MARKET_METRICS = ("ew_return", "median_return", "positive_breadth", "return_dispersion",
                  "return_iqr", "return_skew", "new_high_breadth", "new_low_breadth",
                  "volume_up_breadth", "market_concentration")
STRUCTURAL_CONSTANTS = frozenset(("market_new_high_breadth_1s",
                                 "market_new_low_breadth_1s", "market_volume_up_breadth_1s"))
KEYS = ("source_security_id", "signal_date")


@dataclass(frozen=True)
class ContextFeaturesResult:
    table: pd.DataFrame
    coverage: pd.DataFrame
    specs: tuple[dict, ...]
    provenance: dict


def balanced_cross_sectional_bases() -> tuple[str, ...]:
    names = [f"px_simple_return_{h}s" for h in (1, 5, 10, 20, 21, 42, 63, 126, 252)]
    names += [f"px_skip_return_{far}_{near}s" for far, near in
              ((21, 1), (42, 5), (63, 5), (63, 21), (126, 21), (252, 21), (252, 63))]
    for metric in ("mean_log_return", "realized_vol", "downside_vol", "skew"):
        names += [f"risk_{metric}_{h}s" for h in (5, 20, 63, 126, 252)]
    for metric in ("ma_gap", "trend_slope", "drawdown"):
        names += [f"trend_{metric}_{h}s" for h in (5, 10, 20, 42, 63, 100, 126, 252)]
    for metric in ("log_dollar_volume", "abnormal_volume_ratio", "amihud", "dollar_volume_change"):
        names += [f"liq_{metric}_{h}s" for h in (5, 20, 63, 126, 252)]
    assert len(names) == len(set(names)) == 80
    return tuple(names)


def _base_lookback(name):
    if name.startswith("px_skip_return_"):
        return int(name.split("_")[-2]) + 1
    n = int(name.rsplit("_", 1)[-1][:-1])
    if name.startswith("liq_abnormal_volume_ratio_"):
        return n + 251
    if name.startswith(("px_", "risk_", "liq_amihud_", "liq_dollar_volume_change_")):
        return n + 1
    return n


def context_feature_specs() -> tuple[dict, ...]:
    result = []
    for base in balanced_cross_sectional_bases():
        for scope in ("universe", "index_sleeve"):
            for transform in ("rank_pct", "robust_zscore"):
                formula = ("(average_tie_rank-1)/(N-1); undefined if N<2" if transform == "rank_pct"
                           else "(x-median(x))/(1.4826*median(abs(x-median(x)))); undefined if MAD=0")
                alias = (f"xs_{scope}_rank_pct__px_simple_return_{base.rsplit('_', 1)[-1]}"
                         if transform == "rank_pct" and base.startswith("risk_mean_log_return_") else None)
                result.append(dict(name=f"xs_{scope}_{transform}__{base}", family="cross_sectional",
                    role="cross_sectional", required_inputs=(base, "historical_signal_cohort"),
                    lookback_sessions=_base_lookback(base), formula=formula,
                    model_eligible=alias is None, alias_of=alias,
                    alias_condition="complete equal-length log-return windows" if alias else None,
                    source_commit=SOURCE_COMMIT))
    formulas = {
        "ew_return": "mean(adj_close[t]/adj_close[t-h]-1) over the historical cohort at t",
        "median_return": "median(adj_close[t]/adj_close[t-h]-1) over the historical cohort at t",
        "positive_breadth": "mean(stock h-session return>0)",
        "return_dispersion": "sample standard deviation of stock h-session returns",
        "return_iqr": "linear-interpolated Q75-Q25 of stock h-session returns",
        "return_skew": "adjusted Fisher-Pearson sample skewness of stock h-session returns",
        "new_high_breadth": "mean(adj_close[t]=max(last h session closes))",
        "new_low_breadth": "mean(adj_close[t]=min(last h session closes))",
        "volume_up_breadth": "mean(volume[t]/mean(last h session raw volumes)-1>0)",
        "market_concentration": "top ceil(N/10) mean-dollar-volume sum / all-member sum",
    }
    for h in MARKET_HORIZONS:
        for metric in MARKET_METRICS:
            name = f"market_{metric}_{h}s"
            inputs = (("close", "volume") if metric == "market_concentration" else
                      ("volume",) if metric == "volume_up_breadth" else ("adj_close",))
            result.append(dict(name=name, family="market_state", role="market_state",
                required_inputs=(*inputs, "historical_daily_cohort"),
                lookback_sessions=h if metric in ("new_high_breadth", "new_low_breadth",
                                                 "volume_up_breadth", "market_concentration") else h + 1,
                formula=formulas[metric], structural_constant=name in STRUCTURAL_CONSTANTS,
                model_eligible=name not in STRUCTURAL_CONSTANTS, source_commit=SOURCE_COMMIT))
    for h in MARKET_HORIZONS[1:]:
        result.append(dict(name=f"market_sleeve_spread_sp400_minus_sp500_{h}s",
            family="market_state", role="market_state",
            required_inputs=("adj_close", "historical_daily_cohort"), lookback_sessions=h + 1,
            formula="mean(current SP400 members' h-session returns)-mean(current SP500 members' returns)",
            model_eligible=True, source_commit=SOURCE_COMMIT))
    for n in RELATIVE_WINDOWS:
        for benchmark in ("market", "sleeve"):
            for stem, formula in ((f"relative_{benchmark}_beta", "sample covariance(stock,benchmark)/sample variance(benchmark)"),
                                  (f"relative_relative_return_{benchmark}", "sum(stock daily simple return-benchmark daily simple return)")):
                result.append(dict(name=f"{stem}_{n}s", family="relative_context", role="relative_context",
                    required_inputs=("adj_close", "historical_daily_cohort"), lookback_sessions=n + 1,
                    formula=formula + "; identical complete calendar pairs; sleeve changes reset its window",
                    model_eligible=True, source_commit=SOURCE_COMMIT))
            if benchmark == "market":
                for stem, formula in (("relative_idio_vol_market", "sqrt(252*(sample variance(stock)-covariance^2/variance(benchmark)))"),
                                      ("relative_residual_return_market", "sum(stock daily simple returns)-beta*sum(benchmark daily simple returns)")):
                    result.append(dict(name=f"{stem}_{n}s", family="relative_context", role="relative_context",
                        required_inputs=("adj_close", "historical_daily_cohort"), lookback_sessions=n + 1,
                        formula=formula + "; identical complete calendar pairs",
                        model_eligible=True, source_commit=SOURCE_COMMIT))
    assert len(result) == len({r["name"] for r in result}) == 415
    return tuple(result)


def _frame(value, label):
    if isinstance(value, pd.DataFrame):
        result = value.copy()
    elif hasattr(value, "to_pandas"):
        result = value.to_pandas()
    else:
        raise DataContractError(f"{label} must be a DataFrame or Arrow table")
    if result.columns.duplicated().any():
        raise DataContractError(f"{label} has duplicate columns")
    _reject_target_columns(result, label)
    return result


def _reject_target_columns(frame, label):
    _reject_target_names(frame.columns, label)


def _reject_target_names(names, label):
    forbidden = [name for name in names if any(
        str(name).lower().startswith(prefix) for prefix in
        ("target", "label", "outcome", "future", "next_", "forward", "realized", "class_label", "prediction"))
        or name in ("provider_adjusted_return", "return_status", "daily_backtest_ready", "delisted", "last_price_date")]
    if forbidden:
        raise DataContractError(f"{label} contains prohibited outcome/target fields: {forbidden}")


def _dates(values, label):
    result = []
    for value in values:
        if isinstance(value, (date, pd.Timestamp)):
            value = value.date().isoformat() if isinstance(value, pd.Timestamp) else value.isoformat()
        if not isinstance(value, str):
            raise DataContractError(f"Invalid {label}")
        try:
            parsed = date.fromisoformat(value)
        except ValueError as error:
            raise DataContractError(f"Invalid {label}") from error
        if parsed.isoformat() != value:
            raise DataContractError(f"{label} must use YYYY-MM-DD")
        if value >= "2024-01-01":
            raise DataContractError("Protected 2024+ context input dates are prohibited")
        result.append(value)
    return result


def _cohorts(value, label, *, identity_policy=STRICT_IDENTITY_POLICY):
    """Validate explicit dated rows while retaining unresolved source members."""
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, pd.DataFrame)):
        frames = []
        for snapshot in value:
            if not isinstance(snapshot, UniverseSnapshot):
                raise DataContractError(f"{label} sequence must contain UniverseSnapshot objects")
            frame = _frame(snapshot.table, label)
            frame["signal_date"] = snapshot.as_of
            if "as_of" in frame and not (frame["as_of"] == snapshot.as_of).all():
                raise DataContractError(f"{label} row dates differ from snapshot date")
            frames.append(frame)
        result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    else:
        result = _frame(value, label)
        if "signal_date" not in result:
            for alternative in ("as_of", "trading_date"):
                if alternative in result:
                    result = result.rename(columns={alternative: "signal_date"})
                    break
    required = {*KEYS, "intrinio_security_id", "identity_status"}
    if missing := required - set(result.columns):
        raise DataContractError(f"{label} missing columns: {sorted(missing)}")
    if "source_identity_status" not in result:
        result["source_identity_status"] = result["identity_status"]
    result["signal_date"] = _dates(result["signal_date"], label)
    for clock in ("as_of", "trading_date"):
        if clock in result and _dates(result[clock], clock) != result["signal_date"].tolist():
            raise DataContractError(f"{label} dated row clock differs from its declared signal")
    if result.duplicated(list(KEYS)).any():
        raise DataContractError(f"Duplicate {label} source/date key")
    if not result["source_security_id"].map(lambda x: isinstance(x, str) and bool(x.strip())).all():
        raise DataContractError(f"Invalid {label} source ID")
    if "indices" in result:
        def sleeve(indices):
            if not isinstance(indices, (list, tuple, np.ndarray)):
                raise DataContractError(f"{label} indices must be explicit lists")
            if not set(indices) <= set(SLEEVES):
                raise DataContractError("Context scope supports SP500/SP400 only")
            return indices[0] if len(indices) == 1 else None
        result["index_id"] = result["indices"].map(sleeve)
    elif "index_id" not in result or not result["index_id"].isin(SLEEVES).all():
        raise DataContractError(f"{label} requires a unique SP500/SP400 index sleeve")
    result.loc[result["index_id"].isna(), "identity_status"] = "index_ambiguous"
    if "identity_review_required" in result:
        result.loc[(result["identity_status"] == "resolved") &
                   ~result["identity_review_required"].eq(False), "identity_status"] = "needs_review"
    if "resolution_status" in result:
        result.loc[(result["identity_status"] == "resolved") &
                   ~result["resolution_status"].eq("resolved"), "identity_status"] = "needs_review"
    provider_valid = result["intrinio_security_id"].map(lambda x: isinstance(x, str) and bool(x.strip()))
    if ((result["identity_status"] == "resolved") & ~provider_valid).any():
        raise DataContractError(f"Resolved {label} identity has no provider ID")
    duplicates = result[provider_valid].duplicated(["signal_date", "intrinio_security_id"], keep=False)
    result.loc[result[provider_valid].index[duplicates], "identity_status"] = "target_collision"
    # Explicit source membership clocks, when supplied, cannot cross the signal.
    for row in result.to_dict("records"):
        evidence = json.loads(row["membership_records_json"]) if row.get("membership_records_json") else [row]
        for record in evidence:
            if "trading_date" in record and _dates([record["trading_date"]], "membership trading_date")[0] != row["signal_date"]:
                raise DataContractError(f"{label} membership evidence date differs from its signal")
            for key in ("membership_effective_date", "membership_availability_date"):
                if label == "daily_cohorts" and key not in record:
                    raise DataContractError("Daily context requires explicit effective/availability evidence")
                if key in record:
                    val = record[key]
                    if val in (None, "") or _dates([val], key)[0] > row["signal_date"]:
                        raise DataContractError(f"{label} contains missing or future membership clock")
            end = record.get("membership_end_date")
            if end not in (None, ""):
                # Later interval endpoints are audit metadata, never predictors.
                try:
                    end_date = date.fromisoformat(end).isoformat()
                except (ValueError, TypeError) as error:
                    raise DataContractError("Invalid membership end date") from error
                if end_date < row["signal_date"]:
                    raise DataContractError(f"{label} contains expired membership")
    # Qualification applies to the checked routes, never a caller's stored
    # eligibility flag. A provisional admission retains its review status.
    qualified = qualify_identity_records(result.to_dict("records"),
                                          identity_policy=identity_policy)
    if qualified:
        result = pd.DataFrame(qualified)
    return result.sort_values(list(KEYS)).reset_index(drop=True)


def _finite(values):
    def native(value):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
            return np.nan
        return float(value)
    numeric = values.map(native).astype(float)
    return numeric.where(np.isfinite(numeric))


def _coverage_row(day, name, group, count, observed, status):
    return dict(signal_date=day, feature_name=name, context_group=group,
                cohort_size=int(count), observed_members=int(observed), status=status,
                normalization_policy="complete_original_cohort", latest_input_date=day)


def _xs(table, cohorts, indicators, requested, coverage):
    attached = cohorts.merge(indicators, on=["intrinio_security_id", "signal_date"], how="left", validate="many_to_one")
    identity_ok = attached["identity_eligible"] & attached["intrinio_security_id"].isin(requested)
    for day, universe in attached.groupby("signal_date", sort=True):
        for base in balanced_cross_sectional_bases():
            for scope in ("universe", "index_sleeve"):
                groups = [("universe", universe)] if scope == "universe" else list(universe.groupby("index_id", dropna=False))
                for group, rows in groups:
                    values = _finite(rows[base])
                    valid = values.notna() & identity_ok.loc[rows.index]
                    complete = bool(valid.all()) and len(rows) > 0
                    rank_name = f"xs_{scope}_rank_pct__{base}"
                    z_name = f"xs_{scope}_robust_zscore__{base}"
                    rank_status = "incomplete_cohort"
                    z_status = "incomplete_cohort"
                    if complete:
                        if len(rows) > 1:
                            ranked = (values.rank(method="average") - 1) / (len(rows) - 1)
                            _assign(table, day, rows["source_security_id"], rank_name, ranked)
                            rank_status = "ok"
                        else:
                            rank_status = "undefined_single_member"
                        center = values.median()
                        mad = (values - center).abs().median()
                        if mad > 0:
                            _assign(table, day, rows["source_security_id"], z_name, (values-center)/(1.4826*mad))
                            z_status = "ok"
                        else:
                            z_status = "undefined_zero_dispersion"
                    for name, status in ((rank_name, rank_status), (z_name, z_status)):
                        coverage.append(_coverage_row(day, name, str(group), len(rows), valid.sum(), status))


def _assign(table, day, source_ids, name, values):
    lookup = dict(zip(source_ids, values, strict=True))
    mask = (table["signal_date"] == day) & table["source_security_id"].isin(lookup)
    table.loc[mask, name] = table.loc[mask, "source_security_id"].map(lookup).to_numpy()


def _validated_price_arrays(daily_prices, sessions, requested, through):
    if isinstance(daily_prices, pd.DataFrame):
        names = list(daily_prices.columns)
    elif hasattr(daily_prices, "column_names"):
        names = list(daily_prices.column_names)
    else:
        raise DataContractError("daily_prices must be a DataFrame or Arrow table")
    if len(names) != len(set(names)) or any(not isinstance(name, str) for name in names):
        raise DataContractError("daily_prices requires unique string columns")
    _reject_target_names(names, "daily_prices")
    required = {"intrinio_security_id", "trading_date", "adj_close", "close", "volume",
                "frequency", "intraperiod", "price_quality_status"}
    if missing := required - set(names):
        raise DataContractError(f"daily_prices missing columns: {sorted(missing)}")
    metadata_names = ["intrinio_security_id", "trading_date", "frequency", "intraperiod", "price_quality_status"]
    # Arrow financial columns are not converted until every request key passes.
    frame = (daily_prices[metadata_names].copy().reset_index(drop=True) if isinstance(daily_prices, pd.DataFrame)
             else daily_prices.select(metadata_names).to_pandas())
    frame["trading_date"] = _dates(frame["trading_date"], "price date")
    if not frame["trading_date"].isin(sessions).all():
        raise DataContractError("Price date is outside the supplied session grid")
    if not frame["intrinio_security_id"].isin(requested).all():
        raise DataContractError("Price ID is outside the declared request")
    if frame.duplicated(["intrinio_security_id", "trading_date"]).any():
        raise DataContractError("Duplicate daily price key")
    # Refuse an unbounded request by key metadata before financial conversion.
    # A multi-signal request can contain dates after an earlier signal, but none
    # after its final signal. Individual calculations still slice at each signal.
    if (frame["trading_date"] > through).any():
        raise DataContractError("Price rows exceed the final declared context signal")
    numeric_names = ["adj_close", "close", "volume"]
    financial = (daily_prices[numeric_names].copy().reset_index(drop=True) if isinstance(daily_prices, pd.DataFrame)
                 else daily_prices.select(numeric_names).to_pandas())
    frame = pd.concat([frame, financial], axis=1)
    exact_closed = frame["intraperiod"].map(lambda x: isinstance(x, (bool, np.bool_)) and not x)
    valid_quality = ((frame["frequency"] == "daily") & exact_closed
                     & frame["price_quality_status"].isin(("accepted", "ok")))
    arrays = {}
    for sid, rows in frame.groupby("intrinio_security_id", sort=True):
        arrays[sid] = {}
        for field in ("adj_close", "close", "volume"):
            numeric = _finite(rows[field])
            numeric = numeric.where(valid_quality.loc[rows.index])
            numeric = numeric.where(numeric >= 0 if field == "volume" else numeric > 0)
            arrays[sid][field] = pd.Series(numeric.to_numpy(), index=rows["trading_date"]).reindex(sessions).to_numpy()
    return arrays


def _validate_signal_context_inventory(signal_cohorts, historical):
    """A present daily slice cannot be a silently reduced signal cohort."""
    groups = dict(tuple(historical.groupby("signal_date", sort=True)))
    fields = ["source_security_id", "intrinio_security_id", "index_id", "identity_status",
              "identity_eligible", "identity_eligibility_policy"]
    for day, rows in signal_cohorts.groupby("signal_date", sort=True):
        observed = groups.get(day)
        if observed is None:
            # The numerical builders withhold absent dates; they never fill them.
            continue
        expected = rows[fields].sort_values("source_security_id").reset_index(drop=True)
        actual = observed[fields].sort_values("source_security_id").reset_index(drop=True)
        if not expected.fillna("__missing__").equals(actual.fillna("__missing__")):
            raise DataContractError("Daily context changed the original signal-cohort denominator or identity/sleeve routes")


def _price_metrics(arrays, sid, i, h):
    nan = float("nan")
    result = dict(stock_return=nan, high=nan, low=nan, volume_up=nan, dollar_volume=nan)
    if sid not in arrays:
        return result
    fields = arrays[sid]
    if i >= h:
        prices = fields["adj_close"][i-h:i+1]
        if np.isfinite(prices).all():
            result["stock_return"] = prices[-1]/prices[0]-1
    if i >= h-1:
        prices = fields["adj_close"][i-h+1:i+1]
        if np.isfinite(prices).all():
            result["high"] = float(prices[-1] == prices.max())
            result["low"] = float(prices[-1] == prices.min())
        volume = fields["volume"][i-h+1:i+1]
        if np.isfinite(volume).all() and volume.mean() > 0:
            result["volume_up"] = float(volume[-1]/volume.mean()-1 > 0)
        dollars = fields["close"][i-h+1:i+1]*volume
        if np.isfinite(dollars).all():
            result["dollar_volume"] = dollars.mean()
    return result


def _daily_context(daily_cohorts, arrays, sessions, requested):
    """Produce complete daily benchmarks from that day's historical members."""
    positions = {day: i for i, day in enumerate(sessions)}
    market, sleeve, members = {}, {}, {}
    for day, rows in daily_cohorts.groupby("signal_date", sort=True):
        i = positions[day]
        current = []
        for row in rows.to_dict("records"):
            admitted = row["identity_eligible"] and row["intrinio_security_id"] in requested
            value = _price_metrics(arrays, row["intrinio_security_id"], i, 1)["stock_return"] if admitted else np.nan
            current.append((row, value))
            members[(row["source_security_id"], day)] = dict(row, stock_return=value)
        returns = np.asarray([value for _, value in current], dtype=float)
        market[day] = returns.mean() if len(returns) and np.isfinite(returns).all() else np.nan
        for index in SLEEVES:
            values = np.asarray([value for row, value in current if row["index_id"] == index], dtype=float)
            sleeve[(day, index)] = values.mean() if len(values) and np.isfinite(values).all() else np.nan
    return market, sleeve, members


def _market(table, signal_cohorts, daily_cohorts, arrays, sessions, requested, coverage):
    groups = dict(tuple(daily_cohorts.groupby("signal_date", sort=True)))
    for day in sorted(signal_cohorts["signal_date"].unique()):
        rows = groups.get(day)
        for h in MARKET_HORIZONS:
            counts = 0 if rows is None else len(rows)
            evaluated = []
            if rows is not None:
                i = sessions.index(day)
                for row in rows.to_dict("records"):
                    admitted = row["identity_eligible"] and row["intrinio_security_id"] in requested
                    values = _price_metrics(arrays, row["intrinio_security_id"], i, h) if admitted else {
                        field: np.nan for field in ("stock_return", "high", "low", "volume_up", "dollar_volume")}
                    evaluated.append(dict(row, **values))
            frame = pd.DataFrame(evaluated)
            for metric in MARKET_METRICS:
                name = f"market_{metric}_{h}s"
                field = ("high" if metric == "new_high_breadth" else "low" if metric == "new_low_breadth"
                         else "volume_up" if metric == "volume_up_breadth" else "dollar_volume"
                         if metric == "market_concentration" else "stock_return")
                values = frame[field].to_numpy(dtype=float) if counts else np.array([])
                observed = int(np.isfinite(values).sum())
                value, status = np.nan, "context_missing_date" if rows is None else "incomplete_cohort"
                if counts and observed == counts:
                    value, status = _market_aggregate(metric, values)
                    if not np.isfinite(value) and status == "ok":
                        status = "nonfinite_calculation"
                if np.isfinite(value):
                    table.loc[table["signal_date"] == day, name] = value
                coverage.append(_coverage_row(day, name, "universe", counts, observed, status))
            if h == 1:
                continue
            name = f"market_sleeve_spread_sp400_minus_sp500_{h}s"
            means, observed = {}, 0
            for index in SLEEVES:
                values = frame.loc[frame["index_id"] == index, "stock_return"].to_numpy(dtype=float) if counts else np.array([])
                observed += int(np.isfinite(values).sum())
                means[index] = values.mean() if len(values) and np.isfinite(values).all() else np.nan
            value = means["sp400"]-means["sp500"]
            status = "ok" if np.isfinite(value) else "context_missing_date" if rows is None else "incomplete_cohort"
            if np.isfinite(value):
                table.loc[table["signal_date"] == day, name] = value
            coverage.append(_coverage_row(day, name, "sp400_minus_sp500", counts, observed, status))


def _market_aggregate(metric, values):
    if metric in ("ew_return", "new_high_breadth", "new_low_breadth", "volume_up_breadth"):
        return values.mean(), "ok"
    if metric == "median_return":
        return np.median(values), "ok"
    if metric == "positive_breadth":
        return np.mean(values > 0), "ok"
    if metric == "return_dispersion":
        return (np.std(values, ddof=1), "ok") if len(values) > 1 else (np.nan, "undefined_single_member")
    if metric == "return_iqr":
        return np.quantile(values, .75)-np.quantile(values, .25), "ok"
    if metric == "return_skew":
        variance = np.mean((values-values.mean())**2)
        if len(values) < 3 or variance == 0:
            return np.nan, "undefined_moment"
        n = len(values)
        skew = np.sqrt(n*(n-1))/(n-2)*np.mean((values-values.mean())**3)/variance**1.5
        return skew, "ok"
    if values.sum() <= 0:
        return np.nan, "undefined_zero_volume"
    top = int(np.ceil(len(values)/10))
    return np.sort(values)[-top:].sum()/values.sum(), "ok"


def _relative(table, signal_cohorts, market, sleeve, members, sessions, coverage):
    stats = {}
    for row in signal_cohorts.to_dict("records"):
        day, sid = row["signal_date"], row["source_security_id"]
        i = sessions.index(day)
        for n in RELATIVE_WINDOWS:
            dates = sessions[max(0, i-n+1):i+1]
            for benchmark in ("market", "sleeve"):
                names = [f"relative_{benchmark}_beta_{n}s", f"relative_relative_return_{benchmark}_{n}s"]
                if benchmark == "market":
                    names += [f"relative_idio_vol_market_{n}s", f"relative_residual_return_market_{n}s"]
                stock, reference, complete = [], [], len(dates) == n and row["identity_eligible"]
                current = members.get((sid, day))
                if current is None or current["intrinio_security_id"] != row["intrinio_security_id"] or current["index_id"] != row["index_id"]:
                    complete = False
                for previous in dates:
                    historical = members.get((sid, previous))
                    valid_member = historical is not None and historical["identity_eligible"]
                    # A source identity must route consistently through its window.
                    if valid_member and historical["intrinio_security_id"] != row["intrinio_security_id"]:
                        valid_member = False
                    if benchmark == "sleeve" and valid_member and historical["index_id"] != row["index_id"]:
                        valid_member = False
                    stock.append(historical["stock_return"] if valid_member else np.nan)
                    reference.append(market.get(previous, np.nan) if benchmark == "market"
                                     else sleeve.get((previous, row["index_id"]), np.nan))
                stock, reference = np.asarray(stock), np.asarray(reference)
                paired = np.isfinite(stock) & np.isfinite(reference)
                complete &= bool(paired.all())
                values = {name: np.nan for name in names}
                status = {name: "incomplete_calendar_pairs" for name in names}
                if complete:
                    simple_name = f"relative_relative_return_{benchmark}_{n}s"
                    values[simple_name], status[simple_name] = np.sum(stock-reference), "ok"
                    variance = np.var(reference, ddof=1)
                    if variance > 0:
                        covariance = np.cov(stock, reference, ddof=1)[0, 1]
                        beta = covariance/variance
                        beta_name = f"relative_{benchmark}_beta_{n}s"
                        values[beta_name], status[beta_name] = beta, "ok"
                        if benchmark == "market":
                            residual = np.var(stock, ddof=1)-covariance**2/variance
                            tolerance = 1e-12*max(abs(np.var(stock, ddof=1)), 1e-30)
                            vol_name = f"relative_idio_vol_market_{n}s"
                            if residual >= -tolerance:
                                values[vol_name], status[vol_name] = np.sqrt(252*max(residual, 0)), "ok"
                            else:
                                status[vol_name] = "invalid_residual_variance"
                            return_name = f"relative_residual_return_market_{n}s"
                            values[return_name], status[return_name] = stock.sum()-beta*reference.sum(), "ok"
                    else:
                        for name in names:
                            if name != simple_name:
                                status[name] = "undefined_zero_benchmark_variance"
                for name in names:
                    value = values[name]
                    if np.isfinite(value):
                        _assign(table, day, [sid], name, [value])
                    record = stats.setdefault((day, name), dict(total=0, valid=0, states=set()))
                    record["total"] += 1
                    record["valid"] += int(np.isfinite(value))
                    record["states"].add(status[name])
    for (day, name), summary in sorted(stats.items()):
        status = "ok" if summary["states"] == {"ok"} else ";".join(sorted(summary["states"]))
        coverage.append(_coverage_row(day, name, "source_members", summary["total"], summary["valid"], status))


def compute_context_features(daily_result, daily_prices, session_dates, *, signal_cohorts,
                             requested_security_ids, daily_cohorts=None,
                             context_policy="complete_cohort",
                             identity_policy=STRICT_IDENTITY_POLICY) -> ContextFeaturesResult:
    """Compute snapshots, never admitting constituents through target availability.

The caller owns bounded source reads. Full daily cohorts are an explicit input,
not inferred from signal-date lists. ``coverage`` records original denominators
and withheld definitions. Numeric model input selection must exclude the three
declared constants and retain the separate identity/coverage ledger.
"""
    if context_policy != "complete_cohort":
        raise DataContractError("Only the complete_cohort context policy is implemented")
    validate_identity_policy(identity_policy)
    if isinstance(requested_security_ids, (str, bytes)):
        raise DataContractError("requested_security_ids must be an explicit sequence")
    requested = tuple(requested_security_ids)
    if len(set(requested)) != len(requested) or not all(isinstance(s, str) and s.strip() for s in requested):
        raise DataContractError("Invalid or duplicate requested security IDs")
    requested = set(requested)
    # Calendar metadata may contain future boundary days. Financial input guards
    # remain separate; no context calculation sees dates after its own signal.
    sessions = []
    for value in session_dates:
        if isinstance(value, (date, pd.Timestamp)):
            value = value.date().isoformat() if isinstance(value, pd.Timestamp) else value.isoformat()
        try:
            if date.fromisoformat(value).isoformat() != value:
                raise ValueError
        except (TypeError, ValueError) as error:
            raise DataContractError("Invalid session calendar") from error
        sessions.append(value)
    if len(sessions) != len(set(sessions)) or sessions != sorted(sessions):
        raise DataContractError("Session calendar must be sorted and unique")
    cohorts = _cohorts(signal_cohorts, "signal_cohorts", identity_policy=identity_policy)
    if cohorts.empty or not cohorts["signal_date"].isin(sessions).all():
        raise DataContractError("Signal cohorts must use supplied calendar dates")
    indicators = _frame(daily_result.table, "daily_result")
    required = {"intrinio_security_id", "signal_date", *balanced_cross_sectional_bases()}
    if missing := required - set(indicators.columns):
        raise DataContractError(f"Daily-result feature bases missing: {sorted(missing)}")
    indicators["signal_date"] = _dates(indicators["signal_date"], "indicator date")
    if indicators.duplicated(["intrinio_security_id", "signal_date"]).any():
        raise DataContractError("Duplicate daily-result feature key")
    if not indicators["signal_date"].isin(sessions).all() or not indicators["intrinio_security_id"].isin(requested).all():
        raise DataContractError("Daily-result key is outside the calendar/request")
    specs = context_feature_specs()
    table = pd.concat([cohorts[list(KEYS)].reset_index(drop=True),
                       pd.DataFrame(np.nan, index=range(len(cohorts)), columns=[r["name"] for r in specs])], axis=1)
    coverage = []
    _xs(table, cohorts, indicators, requested, coverage)
    arrays = _validated_price_arrays(daily_prices, sessions, requested, cohorts["signal_date"].max())
    if daily_cohorts is None:
        for day, rows in cohorts.groupby("signal_date", sort=True):
            for spec in specs:
                if spec["role"] != "cross_sectional":
                    coverage.append(_coverage_row(day, spec["name"], "not_loaded", len(rows), 0, "context_not_loaded"))
    else:
        historical = _cohorts(daily_cohorts, "daily_cohorts", identity_policy=identity_policy)
        if historical.empty or not historical["signal_date"].isin(sessions).all():
            raise DataContractError("Daily cohorts must use supplied calendar dates")
        if (historical["signal_date"] > cohorts["signal_date"].max()).any():
            raise DataContractError("Daily cohort rows exceed the final declared context signal")
        _validate_signal_context_inventory(cohorts, historical)
        _market(table, cohorts, historical, arrays, sessions, requested, coverage)
        market, sleeve, members = _daily_context(historical, arrays, sessions, requested)
        _relative(table, cohorts, market, sleeve, members, sessions, coverage)
    provenance = dict(source_commit=SOURCE_COMMIT, registered_columns=415,
        cross_sectional_bases=80, context_scope="historical_sp500_sp400_union",
        context_policy=context_policy, identity_eligibility_policy=identity_policy,
        complete_calendar_windows_required=True,
        signal_identity_eligible_rows=int(cohorts["identity_eligible"].sum()),
        signal_provisional_identity_eligible_rows=int((cohorts["identity_eligible"] &
                                                       cohorts["identity_status"].ne("resolved")).sum()),
        daily_membership_loaded=daily_cohorts is not None,
        feature_timestamp="signal close; all inputs dated at or before signal",
        structural_constants=sorted(STRUCTURAL_CONSTANTS),
        knowledge_time_verified=False, identity_continuity_certified=False,
        current_security_master_filter_used=False, future_quote_eligibility_used=False,
        target_fields_used=False, no_fill_no_zero_imputation=True,
        sleeve_switch_policy="complete same-index history required within relative window",
        window_policy="V2 complete sessions/cohorts; stricter than source 80-percent windows")
    provenance.update(identity_policy_provenance(identity_policy))
    return ContextFeaturesResult(table, pd.DataFrame(coverage), specs, provenance)
