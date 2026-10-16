import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import unittest
import tempfile
from functools import wraps

def raises(kind, match=None):
    case = unittest.TestCase()
    return case.assertRaisesRegex(kind, match) if match is not None else case.assertRaises(kind)


def parametrize(name, values):
    def decorate(fn):
        @wraps(fn)
        def run():
            for value in values:
                try:
                    fn(value)
                except Exception as exc:
                    raise AssertionError(f"{name}={value}: {exc}") from exc
        return run
    return decorate


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_walkforward_features_v2.py"
if not SCRIPT.exists():
    SCRIPT = Path(__file__).resolve().parents[1] / "verify_walkforward_features_v2.py"
spec = importlib.util.spec_from_file_location("independent_features", SCRIPT)
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)


def metadata():
    return pd.DataFrame({"intrinio_security_id": ["native"], "trading_date": ["2020-01-02"]})


def test_price_metadata_guard_rejects_protected_before_numeric_projection():
    bad = metadata(); bad.loc[0, "trading_date"] = "2024-01-02"
    with raises(AssertionError, match="Protected"):
        v.metadata_guard(bad, start="2018-01-01", end="2023-12-29", requested={"native"})


@parametrize("mutation", ["unrequested", "duplicate", "not_session", "numeric_column"])
def test_key_metadata_rejects_unbounded_or_extra_columns(mutation):
    frame = metadata(); allowed = {"2020-01-02"}
    if mutation == "unrequested": frame.loc[0, "intrinio_security_id"] = "other"
    if mutation == "duplicate": frame = pd.concat([frame, frame], ignore_index=True)
    if mutation == "not_session": allowed = {"2020-01-03"}
    if mutation == "numeric_column": frame["adj_close"] = object()
    with raises(AssertionError):
        v.metadata_guard(frame, start="2018-01-01", end="2023-12-29", requested={"native"}, allowed_dates=allowed)


def test_original_order_and_missing_members_retained():
    rows = pd.DataFrame({"source_security_id": ["a", "b"], "signal_date": ["2019-12-31"] * 2})
    v.same_keys(rows, rows.copy())
    with raises(AssertionError): v.same_keys(rows, rows.iloc[:1])
    with raises(AssertionError): v.same_keys(rows, rows.iloc[::-1])
    with raises(AssertionError): v.same_keys(rows, pd.concat([rows, rows.iloc[:1]]))


def test_standalone_budget_rejects_time_rss_or_fit_allowance():
    cfg = {"verification_maximum_wall_seconds": 1800, "verification_maximum_peak_rss_bytes": 8 * 1024**3, "verification_model_fits": 0}
    clock = [0.]; peak = [10]
    budget = v.Budget(cfg, clock=lambda: clock[0], peak=lambda: peak[0])
    clock[0] = 1801.
    with raises(AssertionError, match="time budget"): budget.check("synthetic")
    clock[0] = 0.; peak[0] = 8 * 1024**3 + 1
    with raises(AssertionError, match="RSS budget"): budget.check("synthetic")
    with raises(AssertionError): v.Budget({**cfg, "verification_model_fits": 1})


def test_identity_policy_admits_review_flag_and_blocks_same_date_collisions():
    rows = pd.DataFrame({"source_security_id": ["a", "b", "c", "d"], "signal_date": ["2019-12-31"] * 4,
                         "intrinio_security_id": ["unique", "shared", "shared", None],
                         "unified_security_id": ["ua", "ub", "uc", "ud"],
                         "identity_status": ["needs_review", "resolved", "target_collision", "unmapped_intrinio"],
                         "identity_review_required": [True, False, False, False], "resolution_status": ["resolved"] * 4,
                         "identity_eligible": [True, False, False, False]})
    assert v.identity_admission(rows) == 1
    rows.loc[1, "identity_eligible"] = True
    with raises(AssertionError): v.identity_admission(rows)


def class_fixture():
    n = 12
    keys = {"source_security_id": [f"s{i:02d}" for i in range(n)], "signal_date": ["2019-12-31"] * n}
    eligible = [True] * 11 + [False]
    cohort = pd.DataFrame({**keys, "identity_eligible": eligible, "requested": eligible,
                           "paper_prediction_eligible": eligible})
    observed = [True] * 10 + [False] * 2
    values = [.2, .2, .15, .1, .05, 0., -.05, -.1, -.2, -.5, np.nan, .9]
    returns = pd.DataFrame({**keys, "return_status": ["ok"] * 10 + ["missing_end", "identity_unresolved"],
                            "provider_adjusted_return": values})
    target = pd.DataFrame({**cohort.to_dict("list"), "prediction_eligible": eligible,
        "outcome_month": ["2020-01"] * n, "outcome_end": ["2020-01-31"] * n,
        "outcome_requested": eligible, "return_available": observed, "observed_return_available": observed,
        "source_return_status": returns.return_status, "label_return": values[:10] + [np.nan] * 2,
        "target_class": [2, 2, 3, 4, 5, 6, 7, 8, 9, 10, np.nan, np.nan], "target_available": observed})
    counts = {str(k): 0 if k == 1 else 2 if k == 2 else 1 for k in range(1, 11)}
    diagnostics = pd.DataFrame({"outcome_month": ["2020-01"], "exhaustive_requests": [True],
        "class_counts_json": [json.dumps(counts)], "original_members": [12], "prediction_eligible_members": [11],
        "observed_return_members": [10], "labelled_members": [10], "tied_return_groups": [1], "maximum_tie_size": [2]})
    return cohort, target, returns, diagnostics


def test_exact_ties_and_missing_targets_use_original_denominator():
    result = v.independent_classes(*class_fixture(), cutoff="2020-01-31")
    assert result == {"original_rows": 12, "eligible_rows": 11, "observed_rows": 10, "classes_recomputed": 10, "tie_groups_checked": 1}


@parametrize("problem", ["class_tie", "imputed_target", "future_cohort_filter", "early_cutoff", "unrequested"])
def test_shared_target_verifier_rejects_leakage_or_changed_bins(problem):
    cohort, target, returns, diagnostics = class_fixture()
    cutoff = "2020-01-31"
    if problem == "class_tie": target.loc[0, "target_class"] = 1
    if problem == "imputed_target": target.loc[10, "label_return"] = 0.
    if problem == "future_cohort_filter": target.loc[10, "prediction_eligible"] = False
    if problem == "early_cutoff": cutoff = "2020-01-30"
    if problem == "unrequested": target.loc[10, "outcome_requested"] = False
    with raises(AssertionError): v.independent_classes(cohort, target, returns, diagnostics, cutoff)


def test_monthly_return_uses_signal_close_not_first_outcome_open():
    cohort = pd.DataFrame({"source_security_id": ["a", "b"], "signal_date": ["2019-12-31"] * 2, "identity_eligible": [True, True]})
    returns = pd.DataFrame({**cohort[v.KEYS].to_dict("list"), "intrinio_security_id": ["a", "b"],
        "outcome_end": ["2020-01-31"] * 2, "return_status": ["ok", "missing_end"], "provider_adjusted_return": [.2, np.nan]})
    quotes = pd.DataFrame({"intrinio_security_id": ["a", "a", "a", "b"],
        "trading_date": ["2019-12-31", "2020-01-02", "2020-01-31", "2019-12-31"],
        "frequency": ["daily"] * 4, "intraperiod": [False] * 4, "price_quality_status": ["ok"] * 4,
        "adj_close": [100., 110., 120., 10.]})
    assert v.independent_monthly_returns(cohort, returns, quotes, {"a", "b"}) == {"ok": 1, "missing_end": 1}
    returns.loc[1, "provider_adjusted_return"] = 0.
    with raises(AssertionError): v.independent_monthly_returns(cohort, returns, quotes, {"a", "b"})


def paper_fixture():
    calendar = pd.bdate_range("2018-01-01", "2020-02-28").strftime("%Y-%m-%d").tolist()
    ends = {}
    for d in calendar: ends[d[:7]] = d
    signal = "2019-12-31"
    cohort = pd.DataFrame({"source_security_id": ["a", "b", "c"], "signal_date": [signal] * 3,
                          "intrinio_security_id": ["a", "b", "c"], "identity_eligible": [True] * 3,
                          "requested": [True] * 3, "paper_prediction_eligible": [True] * 3})
    quotes = pd.DataFrame([{"intrinio_security_id": s, "trading_date": d, "frequency": "daily", "intraperiod": False,
                           "price_quality_status": "ok", "adj_close": 100 * (1 + growth) ** i}
                          for s, growth in [("a", .01), ("b", .02), ("c", .03)] for i, d in enumerate(ends.values())])
    features = cohort[v.KEYS].copy()
    # With constant per-month gross returns, each H>1 has H-1 included returns.
    for h in v.HORIZONS:
        raw = np.array([(1 + g) ** (1 if h == 1 else h - 1) - 1 for g in (.01, .02, .03)])
        features[f"zMOM{h}"] = (raw - raw.mean()) / raw.std(ddof=1)
        features[f"meanMOM{h}"] = raw.mean()
        features[f"stdMOM{h}"] = raw.std(ddof=1)
    return cohort, features, quotes, calendar


def test_all15_formulas_and_common_population():
    cohort, features, quotes, calendar = paper_fixture()
    result = v.independent_paper15(cohort, features, quotes, calendar, {"a", "b", "c"})
    assert result["eligible_rows"] == 3 and result["values_checked"] == 45


def test_paper_missing_intermediate_month_never_telescopes_over_gap():
    cohort, features, quotes, calendar = paper_fixture()
    quotes = quotes.loc[~((quotes.intrinio_security_id == "a") & quotes.trading_date.eq("2019-06-28"))]
    with raises(AssertionError, match="eligibility"):
        v.independent_paper15(cohort, features, quotes, calendar, {"a", "b", "c"})


def test_paper_outcome_quotes_cannot_affect_past_feature():
    cohort, features, quotes, calendar = paper_fixture()
    quotes.loc[quotes.trading_date.ge("2020-01-01"), "adj_close"] = 1e100
    assert v.independent_paper15(cohort, features, quotes, calendar, {"a", "b", "c"})["eligible_rows"] == 3


def test_quarantine_retains_original_rows_and_execution_restrictions_are_not_feature_filters():
    original = pd.DataFrame({"source_security_id": [sorted(v.BBBY)[0], "df", "rdc"],
                            "signal_date": ["2019-12-31"] * 3,
                            "intrinio_security_id": ["sec_ydbQnj", "sec_gjP4xM", "sec_zmYNnz"]})
    cohort = original.copy()
    for name in ("identity_eligible", "requested", "paper_prediction_eligible"): cohort[name] = [False, True, True]
    features = original[v.KEYS].copy(); features["a"] = [np.nan, 1., 2.]
    targets = pd.DataFrame({"target_available": [False, True, True]})
    result = v.quarantine_check(original, cohort, features, targets)
    assert result["BBBY_original_rows_retained_unadmitted"] == 1
    assert result["execution_only_restrictions"]["sec_gjP4xM"]["observed_labels"] == 1
    cohort.loc[1, "intrinio_security_id"] = None
    with raises(AssertionError, match="restriction"): v.quarantine_check(original, cohort, features, targets)


def test_deterministic_daily_samples_use_exact_past21_sessions():
    sessions = pd.bdate_range("2019-12-01", periods=21).strftime("%Y-%m-%d").tolist()
    signal = sessions[-1]
    cohort = pd.DataFrame({"source_security_id": ["a"], "signal_date": [signal], "intrinio_security_id": ["a"], "identity_eligible": [True], "requested": [True]})
    quotes = pd.DataFrame({"intrinio_security_id": ["a"] * 21, "trading_date": sessions,
        "frequency": ["daily"] * 21, "intraperiod": [False] * 21, "price_quality_status": ["ok"] * 21,
        "adj_close": [100 * math.exp(.01 * i) for i in range(21)]})
    values = quotes.adj_close.to_numpy()
    features = pd.DataFrame({"px_simple_return_1s": [math.expm1(.01)], "px_simple_return_20s": [math.expm1(.2)],
        "px_log_return_1s": [.01], "px_log_return_20s": [.2], "trend_ma_gap_20s": [values[-1] / values[-20:].mean() - 1], "risk_realized_vol_20s": [0.]})
    assert v.daily_samples(cohort, features, quotes, sessions)["values_checked"] == 6
    with raises(AssertionError): v.daily_samples(cohort, features, quotes.iloc[1:], sessions)


def quarantine_broadcast_fixture():
    original = pd.DataFrame({"source_security_id": [sorted(v.BBBY)[0], "valid"],
                            "signal_date": ["2019-12-31"] * 2,
                            "intrinio_security_id": ["wrong_route", "valid_native"]})
    cohort = original.copy()
    for name in ("identity_eligible", "requested", "paper_prediction_eligible"):
        cohort[name] = [False, True]
    features = original[v.KEYS].copy()
    features["px_simple_return_1s"] = [np.nan, .1]
    for name in v.SHARED_MARKET_STATE:
        features[name] = [.2, .2]
    features[v.SHARED_MARKET_STATE[-1]] = [np.nan, np.nan]
    targets = pd.DataFrame({"target_available": [False, True]})
    return original, cohort, features, targets


def test_quarantine_allows_only_explicit_common_market_broadcast():
    result = v.quarantine_check(*quarantine_broadcast_fixture())
    assert len(result["explicit_shared_market_columns"]) == 65
    assert result["security_specific_columns_checked"] == 1
    assert result["quarantined_security_specific_features_all_null"] is True
    assert result["shared_market_same_signal_broadcast_verified"] is True


@parametrize("field", ["px_simple_return_1s", "zMOM1", "meanMOM1", "stdMOM1", "xs_universe_rank_pct__px_simple_return_1s", "relative_market_beta_20s", "market_unregistered_prefix_exception"])
def test_quarantine_still_rejects_any_own_or_unregistered_populated_feature(field):
    original, cohort, features, targets = quarantine_broadcast_fixture()
    features.loc[0, field] = .7
    with raises(AssertionError, match="security-specific"):
        v.quarantine_check(original, cohort, features, targets)


@parametrize("mutation", ["different_bad_value", "different_valid_value", "missing_bad_value", "missing_valid_value"])
def test_quarantine_rejects_noncommon_market_values_or_missingness(mutation):
    original, cohort, features, targets = quarantine_broadcast_fixture()
    column = v.SHARED_MARKET_STATE[0]
    row = 0 if "bad" in mutation else 1
    features.loc[row, column] = np.nan if mutation.startswith("missing") else .9
    with raises(AssertionError, match="Shared market state differs"):
        v.quarantine_check(original, cohort, features, targets)


@parametrize("flag", ["identity_eligible", "requested", "paper_prediction_eligible", "target_available"])
def test_shared_market_values_cannot_restore_quarantined_admission_or_label(flag):
    original, cohort, features, targets = quarantine_broadcast_fixture()
    (targets if flag == "target_available" else cohort).loc[0, flag] = True
    with raises(AssertionError):
        v.quarantine_check(original, cohort, features, targets)


def test_quarantine_shared_market_contract_requires_same_original_key_order():
    original, cohort, features, targets = quarantine_broadcast_fixture()
    with raises(AssertionError):
        v.quarantine_check(original, cohort, features.iloc[::-1], targets)


def test_shared_market_allowlist_is_exactly65_distinct_frozen_names():
    assert len(v.SHARED_MARKET_STATE) == len(set(v.SHARED_MARKET_STATE)) == 65
    assert "market_new_high_breadth_1s" in v.SHARED_MARKET_STATE
    assert "market_sleeve_spread_sp400_minus_sp500_252s" in v.SHARED_MARKET_STATE
    assert not any(name in v.SHARED_MARKET_STATE for name in ("meanMOM1", "stdMOM1", "market_fake", "relative_market_beta_20s"))


def _check_path_pins(tmp_path):
    source = tmp_path / "saved.json"; source.write_text("fixed\n")
    pin = {"path": source.name, "sha256": v.digest(source), "size_bytes": source.stat().st_size}
    assert v.checked(tmp_path, pin) == source.resolve()
    source.write_text("tampered\n")
    with raises(AssertionError): v.checked(tmp_path, pin)
    with raises(AssertionError): v.inside(tmp_path, "../elsewhere")


def test_timing_guard_requires_next_month_and_no2024_labels():
    sessions = pd.bdate_range("2019-12-01", "2021-02-01").strftime("%Y-%m-%d").tolist()
    groups = {}
    for day in sessions: groups.setdefault(day[:7], []).append(day)
    rows = [{"source_security_id": "a", "signal_date": groups[str(pd.Period(f"2020-{m:02d}") - 1)][-1],
             "outcome_month": f"2020-{m:02d}", "outcome_start": groups[f"2020-{m:02d}"][0], "outcome_end": groups[f"2020-{m:02d}"][-1]} for m in range(1, 13)]
    frame = pd.DataFrame(rows); diag = frame[v.WINDOW].copy()
    assert len(v.timing_guard(frame, frame.copy(), diag, sessions, 2020, diag.to_dict("records"))) == 12
    poisoned = frame.copy(); poisoned.loc[11, "outcome_end"] = "2024-01-02"
    with raises(AssertionError): v.timing_guard(poisoned, poisoned.copy(), poisoned[v.WINDOW], sessions, 2020, poisoned[v.WINDOW].to_dict("records"))


def test_path_pins_reject_escape_and_tampering():
    with tempfile.TemporaryDirectory() as path:
        _check_path_pins(Path(path))


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(value) for name, value in sorted(globals().items()) if name.startswith("test_") and callable(value))


if __name__ == "__main__":
    unittest.main()
