"""Metadata-only, additive feature-to-horizon assignments.

Every original predictor receives exactly one group. These are *declared
lookback windows*, not shifted copies of a feature and not causal importance.
Calendar-month paper inputs, ordinary session windows and composite windows
stay separate. The registered raw dependency span is retained independently.
The module imports no numerical libraries and never opens files or estimators.
"""
from __future__ import annotations

import re
from collections.abc import Sequence


SCHEMA = "dmv2-feature-importance-horizons/v1"
_PAPER = re.compile(r"(?:zMOM|meanMOM|stdMOM)(1|3|6|9|12)\Z")
_SESSION = re.compile(r"_(\d+)s\Z")
_SKIP = re.compile(r"px_skip_return_(\d+)_(\d+)s\Z")
_ABNORMAL = re.compile(r"liq_abnormal_volume_ratio_(\d+)s\Z")
_FAMILIES = frozenset(("security_return", "security_momentum",
    "security_return_distribution", "security_trend", "security_oscillator",
    "security_ohlc", "security_liquidity", "security_event_and_level",
    "paper_momentum_state", "cross_sectional", "market_state",
    "relative_context"))


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _positive_int(value, message):
    _need(isinstance(value, int) and not isinstance(value, bool) and value > 0,
          message)
    return value


def _history(spec, base_spec=None):
    """Keep inclusive raw lag spans, rather than rounding a span to months."""
    n = _positive_int(spec.get("lookback_sessions"),
                      "A session feature needs its registered raw-history span")
    native = base_spec or spec
    spans = native.get("_spans")
    result = {"units": "exchange_sessions", "span_sessions": n,
              "maximum_input_lag_sessions": n - 1,
              "native_input_spans": None,
              "definition": "inclusive registered lag span relative to signal; exact required observations retained below when native spans are available"}
    if spans is not None:
        _need(isinstance(spans, (list, tuple)) and len(spans) > 0,
              "Invalid native dependency spans")
        output = []
        for field, first, last in spans:
            _need(isinstance(field, str) and field and isinstance(first, int)
                  and not isinstance(first, bool) and isinstance(last, int)
                  and not isinstance(last, bool) and 0 <= first <= last,
                  "Invalid native dependency span")
            output.append({"field": field, "minimum_lag_sessions": first,
                           "maximum_lag_sessions": last,
                           "required_observation_count": last - first + 1})
        _need(max(s["maximum_lag_sessions"] for s in output) + 1 == n,
              "Registered context span differs from native base")
        result["native_input_spans"] = output
    return result


def build_horizon_mapping(specs, predictor_names):
    """Map an explicit ordered allowlist using registered definition metadata.

    ``specs`` contains the registered728 definitions, or invented test metadata.
    Optional ``_spans`` from daily_features._definitions() preserve exact native
    requirements. Cross-sectional transforms inherit their native base window;
    transformations create no additional temporal lag. Unknown names or formats
    fail closed; there is no guessed suffix-based fallback for unknown families.
    """
    _need(isinstance(predictor_names, Sequence) and not isinstance(predictor_names, (str, bytes)),
          "Predictors must be an ordered sequence")
    names = list(predictor_names)
    _need(bool(names) and all(isinstance(x, str) and x for x in names)
          and len(names) == len(set(names)), "Predictor names must be unique and nonempty")
    _need(isinstance(specs, Sequence) and not isinstance(specs, (str, bytes)),
          "Definitions must be an explicit sequence")
    by_name = {}
    for item in specs:
        _need(isinstance(item, dict) and isinstance(item.get("name"), str),
              "Invalid feature definition")
        _need(item["name"] not in by_name, "Duplicate feature definition")
        by_name[item["name"]] = item
    result = []
    for ordinal, name in enumerate(names):
        _need(name in by_name, "Unregistered predictor: " + name)
        spec = by_name[name]
        family = spec.get("family")
        _need(family in _FAMILIES, "Unsupported feature family: " + str(family))
        _need(spec.get("model_eligible", True) is True,
              "Excluded dictionary definition entered predictors: " + name)
        base_name = name
        base_spec = None
        if family == "cross_sectional":
            _need(name.count("__") == 1 and name.startswith((
                "xs_universe_rank_pct__", "xs_universe_robust_zscore__",
                "xs_index_sleeve_rank_pct__", "xs_index_sleeve_robust_zscore__")),
                "Unrecognized cross-sectional transform")
            base_name = name.split("__", 1)[1]
            _need(base_name in by_name, "Missing cross-sectional native definition")
            base_spec = by_name[base_name]
            _need(base_spec.get("family") not in ("cross_sectional", "paper_momentum_state"),
                  "Unsupported nested cross-sectional base")
            _need(spec.get("lookback_sessions") == base_spec.get("lookback_sessions"),
                  "Cross-sectional/native lookback disagreement")
        row = {"name": name, "feature_ordinal": ordinal, "family": family,
               "role": spec.get("role"), "base_name": base_name,
               "required_inputs": list(spec.get("required_inputs", ())),
               "registered_formula": spec.get("formula"), "multiscale": False,
               "mapping_schema": SCHEMA}
        if family == "paper_momentum_state":
            match = _PAPER.fullmatch(name)
            _need(match is not None, "Unrecognized paper feature")
            h = int(match[1])
            _need(spec.get("lookback_months") == h, "Paper declared-horizon mismatch")
            lags = [1, 0] if h == 1 else list(range(h, 0, -1))
            row.update(group=f"paper_months_{h}", group_label=f"Paper nominal {h} month" + ("s" if h != 1 else ""),
                group_type="paper_months", group_sort_key=[3, h, 0], units="calendar_months",
                nominal_horizon=h, declared_horizon={"nominal_calendar_months": h,
                    "included_month_return_count": 1 if h == 1 else h - 1,
                    "most_recent_return_month_lag": 0 if h == 1 else 1},
                actual_required_history={"units": "exact_calendar_month_ends",
                    "maximum_month_end_lag": h, "required_month_end_lags": lags,
                    "required_endpoint_count": len(lags),
                    "calendar_to_session_conversion": None,
                    "cohort_admission_gate": "active common-five momentum population requires all five paper windows; distinct from this predictor's nominal horizon"})
        else:
            row["actual_required_history"] = _history(spec, base_spec)
            skip = _SKIP.fullmatch(base_name)
            abnormal = _ABNORMAL.fullmatch(base_name)
            if skip:
                far, near = map(int, skip.groups())
                _need(0 < near < far and spec["lookback_sessions"] == far + 1,
                      "Skip-return horizon/span disagreement")
                row.update(group=f"skip_sessions_{far}_{near}",
                    group_label=f"Skip {far}/{near} sessions", group_type="skip_sessions",
                    group_sort_key=[1, far, near], units="exchange_sessions",
                    nominal_horizon=far, multiscale=True,
                    declared_horizon={"far_lag_sessions": far, "near_lag_sessions": near,
                        "return_interval_sessions": far - near},
                    composite_definition="close[t-near]/close[t-far]-1; complete close window from near through far")
            elif abnormal:
                w = int(abnormal[1])
                _need(spec["lookback_sessions"] == w + 251,
                      "Abnormal-volume rolling/baseline span disagreement")
                row.update(group=f"abnormal_volume_sessions_{w}_252",
                    group_label=f"Abnormal volume {w} / 252 sessions",
                    group_type="abnormal_volume_sessions", group_sort_key=[2, w, 252],
                    units="exchange_sessions", nominal_horizon=w, multiscale=True,
                    declared_horizon={"outer_average_sessions": w, "volume_baseline_sessions": 252},
                    composite_definition="mean over w sessions of volume divided by its own trailing252-session mean; w+251 exact volume observations")
            elif family == "security_event_and_level":
                _need(base_name in ("event_dividend_yield_proxy", "event_dividend_event",
                    "event_split_event", "event_adjustment_factor_change"),
                    "Unrecognized current/event predictor")
                row.update(group="current_event", group_label="Current / event",
                    group_type="current_event", group_sort_key=[4, 0, 0],
                    units="exchange_sessions", nominal_horizon=None,
                    declared_horizon={"current_session": True,
                        "previous_session_required": base_name == "event_adjustment_factor_change"})
            else:
                match = _SESSION.search(base_name)
                _need(match is not None, "Unrecognized ordinary session horizon")
                h = int(match[1]); _positive_int(h, "Invalid ordinary horizon")
                row.update(group=f"sessions_{h}", group_label=f"{h} session" + ("s" if h != 1 else ""),
                    group_type="sessions", group_sort_key=[0, h, 0], units="exchange_sessions",
                    nominal_horizon=h, declared_horizon={"window_sessions": h})
        row.update(feature_name=name, group_id=row["group"],
            group_kind={"sessions": "ordinary_sessions", "paper_months": "paper_months",
                        "skip_sessions": "multiscale", "abnormal_volume_sessions": "multiscale",
                        "current_event": "current_event"}[row["group_type"]],
            sort_key=row["group_sort_key"])
        result.append(row)
    _need(len(result) == len(names) and {r["name"] for r in result} == set(names),
          "Feature-to-horizon mapping lost predictors")
    return result
