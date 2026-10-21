"""Dated learning-only OHLCV repairs; import-safe and no I/O in pure adapters.

Structural contract violations stop. Observed quality/control incompatibilities
produce a REJECTED_SOURCE_QUALITY decision and zero admitted rows.
"""
import datetime
import hashlib
import json
import math

PRICE_COLUMNS = ("intrinio_security_id", "trading_date", "frequency", "intraperiod",
    "open", "high", "low", "close", "volume", "adj_open", "adj_high", "adj_low",
    "adj_close", "adj_volume", "factor", "split_ratio", "dividend",
    "raw_ohlc_bounds_valid", "adjusted_ohlc_bounds_valid", "price_quality_status")
OHLCV = ("open", "high", "low", "close", "volume", "adj_open", "adj_high",
         "adj_low", "adj_close", "adj_volume")
OHLC = ("open", "high", "low", "close")
TOLERANCE = 2e-4
RCM_OLD = "sec_yKbEk4"
RCM_TARGET = "sec_zMM1vd"
DISCA_TARGET = "sec_Xnq2jn"
DISCA_UNIFIED = "equity_us_4f745ae6337bd9ef8d4bb45b"
DISCA_ROUTES = {
    "2022-04-01": ("DISCA", "siblis_isin_context_f53b6add4120f42b1c53"),
    **{f"2022-04-{d:02}": ("DISCA", "siblis_isin_context_f53b6add4120f42b1c53") for d in range(4, 9)},
    "2022-04-11": ("WBD", "siblis_isin_context_bd09bf2365326584bc2c")}


def need(value, message):
    if not value:
        raise ValueError(message)


def day(value):
    need(type(value) is str and datetime.date.fromisoformat(value).isoformat() == value,
         "Canonical ISO date required")
    need(value < "2024-01-01", "Protected future date")
    return value


def key(row):
    return row["intrinio_security_id"], day(row["trading_date"])


def unique(rows):
    result = {}
    for row in rows:
        k = key(row)
        need(k not in result, "Duplicate quote key")
        result[k] = row
    return result


def relative(left, right):
    if not isinstance(left, (int, float)) or isinstance(left, bool) or not math.isfinite(left):
        return None
    if not isinstance(right, (int, float)) or isinstance(right, bool) or not math.isfinite(right):
        return None
    if right == 0:
        return 0.0 if left == 0 else None
    return abs(left - right) / abs(right)


def quality(row, provider_flags=True):
    """Return empirical failures; never invent missing factors/actions/volumes."""
    failures = []
    if row.get("frequency") != "daily" or row.get("intraperiod") is not False:
        failures.append("not_final_daily")
    for prefix in ("", "adj_"):
        values = [row.get(prefix + column) for column in OHLC]
        if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in values):
            failures.append(prefix + "nonpositive_or_nonfinite_OHLC")
        elif not (values[2] <= min(values[0], values[3]) <= max(values[0], values[3]) <= values[1]):
            failures.append(prefix + "OHLC_bounds")
        volume = row.get(prefix + "volume")
        if type(volume) not in (int, float) or not math.isfinite(volume) or volume < 0:
            failures.append(prefix + "invalid_volume")
    if not any("OHLC" in item for item in failures):
        ratio = row["adj_close"] / row["close"]
        if any(abs(row["adj_" + c] / row[c] / ratio - 1) > TOLERANCE for c in OHLC):
            failures.append("incompatible_paired_OHLC_units")
    if provider_flags and (row.get("raw_ohlc_bounds_valid") is not True
            or row.get("adjusted_ohlc_bounds_valid") is not True
            or row.get("price_quality_status") != "accepted"):
        failures.append("provider_quality_not_accepted")
    return failures


def compare_controls(source, target, dates):
    comparisons = []
    failures = []
    for date in dates:
        left, right = source[date], target[date]
        for column in OHLCV:
            residual = relative(left.get(column), right.get(column))
            accepted = residual is not None and residual <= TOLERANCE
            comparisons.append({"trading_date": date, "column": column,
                                "relative_residual": residual, "accepted": accepted})
            if not accepted:
                failures.append("control_" + column + ":" + date)
    return comparisons, failures


def result(case, rows, failures, comparisons, policy, source_count):
    accepted = not failures
    return ({"schema": "dmv2-learning-OHLCV-case/v1", "case": case,
        "status": "ACCEPTED_PROVISIONAL_DATED_APPEND_ONLY" if accepted else "REJECTED_SOURCE_QUALITY",
        "quality_reasons": sorted(set(failures)), "control_relative_tolerance": TOLERANCE,
        "control_comparisons": comparisons, "source_rows": source_count,
        "candidate_repair_rows": len(rows), "admitted_repair_rows": len(rows) if accepted else 0,
        "policy": policy, "canonical_identity_promoted": False}, rows if accepted else [])


def rcm_case(rows, inventory):
    indexed = unique(rows)
    old_dates = inventory["routes"][RCM_OLD]
    target_dates = inventory["routes"][RCM_TARGET]
    need(len(old_dates) == 162 and len(target_dates) == 4, "Frozen RCM inventory count differs")
    expected = {(RCM_OLD, day(d)) for d in old_dates} | {(RCM_TARGET, day(d)) for d in target_dates}
    need(set(indexed) == expected and len(rows) == 166, "RCM source key set differs")
    controls = ["2021-10-28", "2021-10-29"]
    need(all((RCM_OLD, d) in indexed and (RCM_TARGET, d) in indexed for d in controls), "Missing RCM paired controls")
    need(target_dates == ["2021-10-28", "2021-10-29", "2022-06-22", "2022-06-23"], "RCM target controls changed")
    gaps = [d for d in old_dates if "2021-11-01" <= d <= "2022-06-21"]
    need(len(gaps) == 160 and set(old_dates) == set(controls + gaps), "RCM dated gap scope differs")
    failures = []
    for row in rows:
        need(set(row) == set(PRICE_COLUMNS), "RCM source schema changed")
        failures += [f"{key(row)}:{reason}" for reason in quality(row)]
    comparisons, bad = compare_controls(
        {d: indexed[(RCM_OLD, d)] for d in controls},
        {d: indexed[(RCM_TARGET, d)] for d in controls}, controls)
    failures += bad
    repairs = [{**indexed[(RCM_OLD, d)], "intrinio_security_id": RCM_TARGET} for d in gaps]
    return result("RCM", repairs, failures, comparisons, {
        "route_policy": "two_pre_gap_paired_controls_and_primary_dated_one_for_one_reorganization",
        "repair_start": "2021-11-01", "repair_end": "2022-06-21",
        "after_same_date_overlap_available": False,
        "after_target_rows": ["2022-06-22", "2022-06-23"],
        "after_rows_are_equality_controls": False,
        "source_route": RCM_OLD, "learning_native": RCM_TARGET,
        "source_adjustments_actions_preserved": True}, len(rows))


def disca_case(literal_rows, target_controls):
    expected_dates = sorted(DISCA_ROUTES)
    pairs = {}
    allowed = {"unified_security_id", "trading_date", "adjustment", "query_symbol", "feed",
        "timeframe", "selected_source_product_id", "selected_source_security_id", *OHLC, "volume"}
    for row in literal_rows:
        need(set(row) == allowed, "DISCA literal source schema differs")
        date = day(row["trading_date"])
        need(date in DISCA_ROUTES, "Out-of-scope DISCA date")
        symbol, route = DISCA_ROUTES[date]
        need(row["unified_security_id"] == DISCA_UNIFIED and row["query_symbol"] == symbol
            and row["selected_source_product_id"] == "russell_1000_2000_3000"
            and row["selected_source_security_id"] == route
            and row["feed"] == "sip" and row["timeframe"] == "1Day", "DISCA dated source route mismatch")
        need(row["adjustment"] in ("raw", "all"), "Unapproved adjustment")
        pairkey = date, row["adjustment"]
        need(pairkey not in pairs, "Duplicate DISCA raw/all key")
        pairs[pairkey] = row
    need(set(pairs) == {(d, a) for d in expected_dates for a in ("raw", "all")}, "DISCA paired source dates incomplete")
    normalized = {}
    for date in expected_dates:
        raw, adj = pairs[(date, "raw")], pairs[(date, "all")]
        row = {"intrinio_security_id": DISCA_TARGET, "trading_date": date,
            "frequency": "daily", "intraperiod": False,
            **{c: raw[c] for c in (*OHLC, "volume")},
            **{"adj_" + c: adj[c] for c in (*OHLC, "volume")},
            "factor": None, "split_ratio": None, "dividend": None,
            "raw_ohlc_bounds_valid": True, "adjusted_ohlc_bounds_valid": True,
            "price_quality_status": "accepted"}
        normalized[date] = row
    indexed = unique(target_controls)
    controls = ["2022-04-01", "2022-04-11"]
    need(set(indexed) == {(DISCA_TARGET, d) for d in controls}, "DISCA target controls differ")
    failures = []
    for date, row in normalized.items():
        failures += [date + ":" + reason for reason in quality(row, provider_flags=False)]
    for row in target_controls:
        need(set(row) == set(PRICE_COLUMNS), "DISCA target control schema changed")
        failures += [str(key(row)) + ":" + reason for reason in quality(row)]
    comparisons, bad = compare_controls(normalized, {d: indexed[(DISCA_TARGET, d)] for d in controls}, controls)
    failures += bad
    gaps = [f"2022-04-{d:02}" for d in range(4, 9)]
    return result("DISCA", [normalized[d] for d in gaps], failures, comparisons, {
        "route_policy": "dated_series_A_DISCA_to_WBD_regular_way_one_for_one",
        "repair_start": gaps[0], "repair_end": gaps[-1], "learning_native": DISCA_TARGET,
        "volume_controls_independently_required": True,
        "missing_source_factor_actions": "preserved_as_missing_no_imputation",
        "derived_quality_flags": "validated_paired_daily_OHLCV_not_provider_certification"}, len(literal_rows))


def append_only(base_rows, repair_rows, start=None, end=None):
    """Add only absent exact dated keys; return original rows unchanged then additions."""
    base = unique(base_rows)
    patch = unique(repair_rows)
    selected = [r for k, r in sorted(patch.items())
                if (start is None or k[1] >= start) and (end is None or k[1] <= end)]
    need(not any(key(row) in base for row in selected), "Append-only repair conflicts with existing key")
    return list(base_rows) + selected
