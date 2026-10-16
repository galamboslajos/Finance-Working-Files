"""Independent, no-fit audit of saved 2020--2023 feature/target annuals.

No dmv2 workers are imported. All saved price-table keys are inspected before
any financial projection. Exact monthly returns, all Paper15 values and shared
target deciles are recomputed, plus deterministic small daily-feature samples.
This is reconstruction verification, not identity/knowledge-time certification.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date
import gc
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

YEARS = (2020, 2021, 2022, 2023)
KEYS = ["source_security_id", "signal_date"]
PRICE_KEYS = ["intrinio_security_id", "trading_date"]
WINDOW = ["signal_date", "outcome_month", "outcome_start", "outcome_end"]
HORIZONS = (1, 3, 6, 9, 12)
PAPER15 = [f"{prefix}{h}" for prefix in ("zMOM", "meanMOM", "stdMOM") for h in HORIZONS]
BBBY = {"siblis_ticker_company_00326c7d9ad13903e5e3", "siblis_isin_c297a5f81a23c9f90503"}
RESTRICTED = {"sec_gjP4xM", "sec_zmYNnz"}
EXPECTED_ROWS = {2020: 10916, 2021: 10920, 2022: 10911, 2023: 10908}
QUALITY = [*PRICE_KEYS, "frequency", "intraperiod", "price_quality_status", "adj_close"]


def need(ok, message):
    if not ok:
        raise AssertionError(message)


class Budget:
    """Bound the standalone audit without importing any production worker."""
    def __init__(self, execution, *, clock=time.monotonic, peak=None):
        self.clock = clock
        self.started = clock()
        self.peak = peak or (lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024))
        self.seconds = execution["verification_maximum_wall_seconds"]
        self.bytes = execution["verification_maximum_peak_rss_bytes"]
        need(type(self.seconds) is int and 0 < self.seconds <= 1800
             and type(self.bytes) is int and 0 < self.bytes <= 8 * 1024**3
             and execution["verification_model_fits"] == 0, "Frozen independent verification budget changed")
        self.check("initialization")

    def check(self, label):
        need(self.clock() - self.started <= self.seconds, "Verification time budget exceeded at " + label)
        need(self.peak() <= self.bytes, "Verification RSS budget exceeded at " + label)

    def frame(self, path, **kwargs):
        self.check("before reading " + Path(path).name)
        frame = pq.read_table(path, use_threads=False, **kwargs).to_pandas()
        self.check("after reading " + Path(path).name)
        return frame

    def compute(self, label, fn, *args, **kwargs):
        self.check("before " + label)
        result = fn(*args, **kwargs)
        self.check("after " + label)
        return result


def digest(path, budget=None):
    value = hashlib.sha256()
    if budget: budget.check("before hashing " + Path(path).name)
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
            if budget: budget.check("hashing " + Path(path).name)
    if budget: budget.check("after hashing " + Path(path).name)
    return value.hexdigest()


def canonical(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def inside(base, relative):
    need(isinstance(relative, str) and not Path(relative).is_absolute(), "Relative artifact path required")
    base = Path(base).resolve()
    path = (base / relative).resolve()
    need(path.is_relative_to(base), "Artifact path escaped declared base")
    return path


def checked(base, pin, budget=None):
    path = inside(base, pin["path"])
    need(path.is_file() and path.stat().st_size == pin["size_bytes"]
         and digest(path, budget) == pin["sha256"].removeprefix("sha256:"), "Artifact pin changed: " + str(path))
    return path


def iso(value):
    need(isinstance(value, str) and date.fromisoformat(value).isoformat() == value, "Noncanonical date")
    return value


def same_keys(left, right, keys=KEYS):
    need(not left.duplicated(keys).any() and not right.duplicated(keys).any(), "Duplicate original keys")
    need(left[keys].to_dict("records") == right[keys].to_dict("records"), "Original ordered denominator changed")


def near(actual, expected, label):
    actual, expected = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    need(actual.shape == expected.shape and np.array_equal(np.isnan(actual), np.isnan(expected)), label + ": missingness differs")
    need(np.allclose(actual, expected, rtol=2e-11, atol=2e-12, equal_nan=True), label + ": values differ")


def identity_admission(frame):
    """Recompute provisional structural admission from same-date metadata only."""
    present = lambda values: values.map(lambda x: isinstance(x, str) and bool(x.strip()))
    native = present(frame.intrinio_security_id)
    unique = ~frame.loc[native].duplicated(["signal_date", "intrinio_security_id"], keep=False).reindex(frame.index, fill_value=False)
    explicit_flag = frame.identity_review_required.map(lambda x: isinstance(x, (bool, np.bool_)))
    status = frame.identity_status
    expected = (native & unique & present(frame.unified_security_id) & explicit_flag
                & frame.resolution_status.eq("resolved") & status.isin(["resolved", "needs_review"])
                & (status.eq("resolved") | frame.identity_review_required.eq(True)))
    expected &= ~frame.source_security_id.isin(BBBY)
    need(np.array_equal(frame.identity_eligible.to_numpy(dtype=bool), expected.to_numpy(dtype=bool)),
         "Signal-time provisional identity admission differs from dated structural evidence")
    return int(expected.sum())


def metadata_guard(frame, *, start, end, requested, allowed_dates=None):
    """This function receives keys only; reject scope before numeric conversion."""
    need(set(frame) == set(PRICE_KEYS), "Price metadata guard must receive keys only")
    need(not frame.duplicated(PRICE_KEYS).any(), "Duplicate daily quote key")
    need(set(frame.intrinio_security_id) <= set(requested), "Unrequested native in saved daily view")
    for day in frame.trading_date.unique():
        need(start <= iso(day) <= end < "2024-01-01", "Protected or out-of-window daily price metadata")
        need(allowed_dates is None or day in allowed_dates, "Daily quote is outside exact allowed sessions")
    return len(frame)


def timing_guard(target_meta, returns_meta, diagnostic_meta, calendar, year, declared):
    need(year in YEARS and len(diagnostic_meta) == 12, "Annual scope changed")
    same_keys(target_meta, returns_meta)
    pd.testing.assert_frame_equal(target_meta, returns_meta, check_dtype=False)
    windows = target_meta[WINDOW].drop_duplicates().sort_values("signal_date").reset_index(drop=True)
    pd.testing.assert_frame_equal(windows, diagnostic_meta[WINDOW].sort_values("signal_date").reset_index(drop=True), check_dtype=False)
    pd.testing.assert_frame_equal(windows, pd.DataFrame(declared)[WINDOW].sort_values("signal_date").reset_index(drop=True), check_dtype=False)
    need(set(windows.outcome_month) == {f"{year}-{m:02d}" for m in range(1, 13)}, "Incomplete annual target months")
    groups = {}
    for day in calendar:
        groups.setdefault(iso(day)[:7], []).append(day)
    for r in windows.itertuples(index=False):
        period = pd.Period(r.outcome_month, "M")
        need(r.signal_date[:7] == str(period - 1) and iso(r.signal_date) < iso(r.outcome_start) <= iso(r.outcome_end) < "2024-01-01", "Feature/label chronology overlap")
        need(r.signal_date == groups[str(period - 1)][-1] and r.outcome_start == groups[r.outcome_month][0]
             and r.outcome_end == groups[r.outcome_month][-1] and str(period + 1) in groups, "Window is not exact completed exchange month")
    return windows


def valid_quote(row):
    if row is None:
        return None, "missing_exact_endpoint"
    if row.frequency != "daily" or not isinstance(row.intraperiod, (bool, np.bool_)) or row.intraperiod or row.price_quality_status not in ("ok", "accepted"):
        return None, "source_quality_flag"
    value = row.adj_close
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (float, int, np.number)) or not math.isfinite(value) or value <= 0:
        return None, "invalid_endpoint"
    return float(value), "ok"


def quote_map(frame):
    need(not frame.duplicated(PRICE_KEYS).any(), "Duplicate endpoint quote")
    return {(r.intrinio_security_id, r.trading_date): r for r in frame.itertuples(index=False)}


def independent_monthly_returns(cohort, returns, quotes, requested):
    same_keys(cohort, returns)
    q = quote_map(quotes)
    statuses = Counter()
    for c, r in zip(cohort.itertuples(index=False), returns.itertuples(index=False), strict=True):
        first, last = q.get((r.intrinio_security_id, r.signal_date)), q.get((r.intrinio_security_id, r.outcome_end))
        a, sa = valid_quote(first)
        b, sb = valid_quote(last)
        value = np.nan
        if not c.identity_eligible:
            status = "identity_unresolved"
        elif r.intrinio_security_id not in requested:
            status = "not_requested"
        elif first is None:
            status = "missing_start"
        elif last is None:
            status = "missing_end"
        elif any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (float, int, np.number)) or not math.isfinite(v) or v <= 0 for v in (first.adj_close, last.adj_close)):
            status = "invalid_endpoint"
        elif sa != "ok" or sb != "ok":
            status = "source_quality_flag"
        else:
            value = b / a - 1
            status = "ok" if math.isfinite(value) else "invalid_endpoint"
        need(r.return_status == status, "Exact monthly return status differs")
        near([r.provider_adjusted_return], [value if status == "ok" else np.nan], "Exact monthly endpoint return")
        statuses[status] += 1
    return dict(statuses)


def independent_paper15(cohort, features, quotes, calendar, requested):
    same_keys(cohort, features)
    q = quote_map(quotes)
    month_ends = {}
    for day in calendar:
        month_ends[day[:7]] = day
    checked_rows = qualified = 0
    statuses = Counter()
    for signal, rows in cohort.groupby("signal_date", sort=True):
        raw = np.full((len(rows), 5), np.nan)
        for i, r in enumerate(rows.itertuples(index=False)):
            if not r.identity_eligible or r.intrinio_security_id not in requested:
                continue
            p = pd.Period(signal[:7], "M")
            for j, h in enumerate(HORIZONS):
                months = [str(p - 1), str(p)] if h == 1 else [str(p + offset) for offset in range(-h, 0)]
                if any(m not in month_ends for m in months):
                    continue
                values = [valid_quote(q.get((r.intrinio_security_id, month_ends[m])))[0] for m in months]
                if any(v is None for v in values):
                    continue
                gross = [b / a for a, b in zip(values, values[1:])]
                product = math.prod(gross)
                if math.isfinite(product) and product > 0:
                    raw[i, j] = product - 1
        common = rows.identity_eligible.to_numpy(dtype=bool) & np.isfinite(raw).all(axis=1)
        count = int(common.sum())
        exhaustive = bool(rows.loc[rows.identity_eligible, "requested"].all())
        expected = np.full((len(rows), 15), np.nan)
        state = "incomplete_request" if not exhaustive else "insufficient_common_population" if count < 2 else "qualified_common_population"
        if state == "qualified_common_population":
            means = np.array([math.fsum(raw[common, j]) / count for j in range(5)])
            sds = np.array([math.sqrt(math.fsum((v - means[j]) ** 2 for v in raw[common, j]) / (count - 1)) for j in range(5)])
            if not np.isfinite(np.r_[means, sds]).all() or (sds <= 0).any():
                state = "degenerate_dispersion"
            else:
                expected[common] = np.column_stack([(raw[common] - means) / sds, np.broadcast_to(means, (count, 5)), np.broadcast_to(sds, (count, 5))])
        eligible = common & (state == "qualified_common_population")
        need(np.array_equal(rows.paper_prediction_eligible.to_numpy(dtype=bool), eligible), "Independent Paper15 eligibility differs")
        near(features.loc[rows.index, PAPER15], expected, "Independent all15 paper formulas at " + signal)
        checked_rows += len(rows)
        qualified += int(eligible.sum())
        statuses[state] += 1
    return {"rows": checked_rows, "eligible_rows": qualified, "values_checked": checked_rows * 15, "normalization_months": dict(statuses)}


def independent_classes(cohort, target, returns, diagnostics, cutoff):
    same_keys(cohort, target)
    same_keys(cohort, returns)
    for field in ("identity_eligible", "requested", "paper_prediction_eligible"):
        need(target[field].equals(cohort[field]), "Target admission differs from past feature cohort")
    eligible = cohort.identity_eligible & cohort.paper_prediction_eligible
    need(np.array_equal(target.prediction_eligible, eligible), "Prediction eligibility used future outcomes")
    observed = eligible & target.outcome_requested.fillna(False).astype(bool) & returns.return_status.eq("ok") & np.isfinite(returns.provider_adjusted_return) & returns.provider_adjusted_return.ge(-1) & target.outcome_end.le(cutoff)
    need(np.array_equal(target.return_available, observed) and np.array_equal(target.observed_return_available, observed), "Return availability changed/missing return imputed")
    near(target.label_return, returns.provider_adjusted_return.where(observed), "Observed labels")
    by_month = diagnostics.set_index("outcome_month")
    total = ties = 0
    for month, rows in target.groupby("outcome_month", sort=True):
        valid = rows.loc[rows.return_available]
        admitted = rows.loc[rows.identity_eligible]
        exhaustive = bool(admitted.requested.all() and admitted.outcome_requested.eq(True).all()
                          and not rows.loc[rows.identity_eligible, "source_return_status"].eq("not_requested").any())
        completed = bool(rows.outcome_end.le(cutoff).all())
        expected = pd.Series(np.nan, index=rows.index)
        if exhaustive and completed and len(valid) >= 10:
            rank2 = (valid.label_return.rank(method="average", ascending=False) * 2).astype("int64")
            expected.loc[valid.index] = (1 + 10 * (rank2 - 1) // (2 * len(valid))).clip(1, 10)
            total += len(valid)
        near(rows.target_class, expected, "Independent exact-tie classes " + month)
        need(np.array_equal(rows.target_available, expected.notna()), "Target availability differs")
        d = by_month.loc[month]
        need(bool(d.exhaustive_requests) == exhaustive, "Request-completeness diagnostic differs")
        counts = rows.target_class.value_counts()
        need(json.loads(d.class_counts_json) == {str(k): int(counts.get(k, 0)) for k in range(1, 11)}, "Class-count audit differs")
        sizes = valid.label_return.value_counts()
        tied = sizes[sizes > 1]
        need(d.original_members == len(rows) and d.prediction_eligible_members == rows.prediction_eligible.sum()
             and d.observed_return_members == len(valid) and d.labelled_members == expected.notna().sum(), "Original coverage denominator differs")
        need(d.tied_return_groups == len(tied) and d.maximum_tie_size == (int(sizes.max()) if len(sizes) else 0), "Tie diagnostics differ")
        ties += len(tied)
    return {"original_rows": len(target), "eligible_rows": int(eligible.sum()), "observed_rows": int(observed.sum()), "classes_recomputed": total, "tie_groups_checked": ties}


def quarantine_check(original, cohort, features, targets):
    same_keys(original, cohort)
    bad = original.source_security_id.isin(BBBY)
    need(not cohort.loc[bad, ["identity_eligible", "requested", "paper_prediction_eligible"]].any().any(), "Known wrong BBBY route was admitted")
    need(not targets.loc[bad, "target_available"].any(), "Known wrong BBBY route acquired label")
    need(features.loc[bad, features.columns.difference(KEYS)].isna().all().all(), "Known wrong BBBY route acquired features")
    counts = {}
    for native in sorted(RESTRICTED):
        mask = original.intrinio_security_id.eq(native)
        need(cohort.loc[mask, "intrinio_security_id"].equals(original.loc[mask, "intrinio_security_id"]), "Execution-only restriction censored source identity")
        counts[native] = {"original_rows_retained": int(mask.sum()), "identity_eligible_rows": int(cohort.loc[mask, "identity_eligible"].sum()),
                          "feature_eligible_rows": int(cohort.loc[mask, "paper_prediction_eligible"].sum()), "observed_labels": int(targets.loc[mask, "target_available"].sum())}
    return {"BBBY_original_rows_retained_unadmitted": int(bad.sum()), "execution_only_restrictions": counts}


def daily_samples(cohort, features, quotes, sessions):
    q = quote_map(quotes)
    names = ["px_simple_return_1s", "px_simple_return_20s", "px_log_return_1s", "px_log_return_20s", "trend_ma_gap_20s", "risk_realized_vol_20s"]
    checked_rows = 0
    for signal, rows in cohort.groupby("signal_date", sort=True):
        sample = rows.loc[rows.identity_eligible & rows.requested].sort_values(["intrinio_security_id", "source_security_id"]).drop_duplicates("intrinio_security_id").head(2)
        pos = sessions.index(signal)
        dates = sessions[pos - 20:pos + 1]
        need(len(dates) == 21 and dates[-1] == signal and max(dates) <= signal, "Daily sample consumed a future observation")
        for idx, row in sample.iterrows():
            values = [valid_quote(q.get((row.intrinio_security_id, day)))[0] for day in dates]
            expected = {}
            for h in (1, 20):
                valid = all(v is not None for v in values[-h - 1:])
                expected[f"px_simple_return_{h}s"] = values[-1] / values[-h - 1] - 1 if valid else np.nan
                expected[f"px_log_return_{h}s"] = math.log(values[-1] / values[-h - 1]) if valid else np.nan
            expected["trend_ma_gap_20s"] = values[-1] / np.mean(values[-20:]) - 1 if all(v is not None for v in values[-20:]) else np.nan
            expected["risk_realized_vol_20s"] = float(np.std(np.diff(np.log(values)), ddof=1) * math.sqrt(252)) if all(v is not None for v in values) else np.nan
            near(features.loc[idx, names], [expected[n] for n in names], "Past-only deterministic daily sample")
            checked_rows += 1
    return {"rows_checked": checked_rows, "values_checked": checked_rows * len(names), "features": names,
            "selection": "first two distinct native IDs per signal from past identity/request metadata, lexicographic", "through_signal_inclusive": True}


def preservation(root, run, cfg, budget=None):
    records = json.loads((run / "source_preservation_manifest.json").read_text())
    need(bool(records), "Missing source preservation audit")
    for pin in records.values():
        checked(root, pin, budget)
    required = set()
    for prefix in (cfg["parent_training_directory"], cfg["corrected_validation_directory"],
                   "outputs/model_pilot/2010-2017__2018-2019-a2cd7669fde3f582",
                   "outputs/daily_backtest/2018-2019-3018cad95c61c1da", "outputs/action_backtest/2018-2019-b088d2328a8aea03"):
        directory = inside(root, prefix)
        files = [p for p in directory.rglob("*") if p.is_file()]
        need(bool(files), "Expected preserved original run is missing")
        required.update(str(p.relative_to(root)) for p in files)
    need(required <= set(records), "Source preservation omitted original run files")
    return {"files_hashed": len(records), "required_original_run_files": len(required), "all_recorded_bytes_unchanged": True}


@threadpool_limits.wrap(limits=1)
def run_verification(root, run, expected_identity):
    root = Path(root).resolve(); run = Path(run).resolve()
    need(run.is_relative_to(root / "outputs/walkforward"), "Walkforward output path required")
    identity = json.loads((run / "build_identity.json").read_text())
    need(canonical(identity) == expected_identity, "Explicit feature recipe identity changed")
    cfg = identity["configuration"]
    budget = Budget(cfg["execution"])
    need(cfg["feature_outcome_years"] == list(YEARS) and cfg["context_policy"] == "available_cohort_80pct"
         and cfg["execution"]["financial_2024_plus_analysis"] is False, "Frozen feature scope changed")
    audit = json.loads((run / "audit.json").read_text())
    need(audit["status"] == "COMPLETE" and audit["build_identity_sha256"] == expected_identity
         and set(audit["years"]) == set(map(str, YEARS)) and audit["financial_2024_plus_analysis"] is False, "Feature phase is incomplete or out of scope")
    result = {"status": "PASS", "build_identity_sha256": expected_identity,
              "root_audit_sha256": digest(run / "audit.json", budget), "models_fitted": 0, "model_fits": 0,
              "financial_2024_plus_read": False, "years": {},
              "source_preservation_before": preservation(root, run, cfg, budget), "limitations": ["Historical reconstruction only; identity and knowledge-time remain uncertified.", "Recomputations use saved reviewed daily views, not a new independent vendor acquisition.", "This check does not certify all715 expanded formulas; six daily formulas are deterministically sampled.", "Parquet projections may decode excluded values inside mixed Arrow blocks; no2024 price row is returned or converted to Python."]}
    total_rows = 0
    for year in YEARS:
        budget.check("before annual verification " + str(year))
        directory = checked(run, audit["years"][str(year)], budget).parent
        annual = json.loads((directory / "audit.json").read_text())
        need(annual["status"] == "COMPLETE" and annual["year"] == year and annual["build_identity_sha256"] == expected_identity, "Annual recipe changed")
        paths = {name: checked(directory, pin, budget) for name, pin in annual["artifacts"].items()}
        state = json.loads(paths["target_state"].read_text()); prov = annual["feature_provenance"]
        need(canonical(state["provenance"]) == state["artifact_sha256"] == annual["target_artifact_sha256"], "Target provenance hash changed")
        need(prov["feature_dictionary_sha256"] == identity["feature_dictionary_sha256"] and prov["outcomes_consumed"] is False
             and prov["technical"]["input_through"] == max(prov["signal_dates"]), "Feature dictionary/clock claims differ")
        need(prov["context"]["context_policy"] == "available_cohort_80pct"
             and prov["context"]["minimum_original_cohort_coverage"] == .8
             and prov["context"]["target_fields_used"] is False
             and prov["context"]["future_quote_eligibility_used"] is False, "Context gate or declared information set changed")
        original = budget.frame(paths["original_cohort"])
        cohort = budget.frame(paths["feature_coverage"])
        need(len(original) == EXPECTED_ROWS[year], "Frozen original annual denominator changed")
        same_keys(original, cohort)
        calendar_frame = budget.frame(paths["exchange_calendar"], columns=["trading_date", "calendar_name", "is_trading_session"])
        need(calendar_frame.calendar_name.nunique() == 1 and calendar_frame.is_trading_session.isin([True, False]).all(), "Calendar state changed")
        sessions = sorted(calendar_frame.loc[calendar_frame.is_trading_session, "trading_date"].tolist())
        need(len(sessions) == len(set(sessions)), "Duplicate exchange sessions")
        # All timing and source-key guards precede every numeric table read.
        tmeta = budget.frame(paths["targets"], columns=[*KEYS, *WINDOW[1:]])
        rmeta = budget.frame(paths["monthly_returns"], columns=[*KEYS, *WINDOW[1:]])
        dmeta = budget.frame(paths["target_diagnostics"], columns=WINDOW)
        windows = timing_guard(tmeta, rmeta, dmeta, sessions, year, state["provenance"]["month_windows"])
        same_keys(original, tmeta)
        need(state["provenance"]["label_observed_through"] == windows.outcome_end.max(), "Label cutoff differs")
        need(original.as_of.equals(original.signal_date), "Historical membership clock differs")
        plan = identity["years"][str(year)]
        requested = prov["requested_ids"]
        source_prov = plan["source_provenance"]
        source_dir = inside(root, source_prov["reference_primary"]["directory"]) / "years" / str(year)
        need(digest(source_dir / "audit.json", budget) == source_prov["annual_audit_sha256"].removeprefix("sha256:"), "Source annual metadata audit changed")
        source_audit = json.loads((source_dir / "audit.json").read_text())
        source_original = budget.frame(checked(source_dir, source_prov["used_artifacts"]["original_cohort"], budget),
                                        columns=[*KEYS, "intrinio_security_id", "identity_review_required", "resolution_status"])
        same_keys(source_original, original)
        unchanged = ~original.source_security_id.isin(BBBY)
        for field in ("intrinio_security_id", "identity_review_required", "resolution_status"):
            need(source_original.loc[unchanged, field].equals(original.loc[unchanged, field]), "Unrelated original source identity/admission changed")
        need(original.identity_eligible.equals(cohort.identity_eligible), "Feature worker changed frozen amended identity admission")
        identity_admission(original)
        daily = budget.frame(paths["daily_cohorts"])
        source_daily = budget.frame(checked(source_dir, source_prov["used_artifacts"]["daily_cohorts"], budget), columns=KEYS)
        same_keys(source_daily, daily)
        first_signal = windows.signal_date.min(); last_signal = windows.signal_date.max()
        context_start = sessions[sessions.index(first_signal) - 251]
        raw_start = sessions[sessions.index(first_signal) - 502]
        need(set(daily.signal_date) == {d for d in sessions if context_start <= d <= last_signal}, "Daily membership history replaced with frozen month-end cohort")
        if "as_of" in daily:
            need(daily.as_of.equals(daily.signal_date), "Daily membership clock differs")
        need((daily.membership_effective_date <= daily.signal_date).all()
             and (daily.membership_availability_date <= daily.signal_date).all()
             and (daily.membership_end_date.isna() | (daily.membership_end_date >= daily.signal_date)).all(),
             "Historical daily membership used an unavailable or ineffective record")
        identity_admission(daily)
        expected_requests = set(original.loc[original.identity_eligible, "intrinio_security_id"]) | set(daily.loc[daily.identity_eligible, "intrinio_security_id"])
        need(len(requested) == len(set(requested)) and set(requested) == expected_requests,
             "Pre-price exhaustive original requests were selected by quote availability")
        need(np.array_equal(cohort.requested, cohort.intrinio_security_id.isin(requested)), "Past feature request flags differ from actual request manifest")
        raw_metas = {}
        permitted_outcome = {d for d in sessions if first_signal <= d <= windows.outcome_end.max()}
        for name, start, end, allowed in [("feature_history_daily", raw_start, last_signal, set(sessions)),
                                        ("daily_history_raw", first_signal, windows.outcome_end.max(), permitted_outcome),
                                        ("daily_prices_raw", f"{year}-01-01", windows.outcome_end.max(), permitted_outcome)]:
            metadata = budget.frame(paths[name], columns=PRICE_KEYS)
            metadata_guard(metadata, start=start, end=end, requested=requested, allowed_dates=allowed)
            raw_metas[name] = metadata
        expected_daily = raw_metas["daily_history_raw"].loc[raw_metas["daily_history_raw"].trading_date.ge(f"{year}-01-01")].reset_index(drop=True)
        same_keys(expected_daily, raw_metas["daily_prices_raw"], PRICE_KEYS)
        fnames = pq.ParquetFile(paths["features"]).schema_arrow.names
        same_keys(original, budget.frame(paths["features"], columns=KEYS))
        source_feature_path = checked(source_dir, source_audit["artifacts"]["features"], budget)
        need(fnames == pq.ParquetFile(source_feature_path).schema_arrow.names, "Feature names differ from pinned unchanged dictionary")
        need(len(fnames) == 730 and fnames[:2] == KEYS and set(PAPER15) <= set(fnames), "Feature-only730-column schema changed")
        need(not any(n.lower().startswith(("target", "label", "outcome", "future", "next_", "forward", "prediction", "realized_")) for n in fnames[2:]), "Outcome or audit leaked into predictor schema")
        # Numeric decoding starts only after complete key/date validation.
        features = budget.frame(paths["features"]); same_keys(original, features)
        targets = budget.frame(paths["targets"])
        returns = budget.frame(paths["monthly_returns"])
        diagnostics = budget.frame(paths["target_diagnostics"])
        for field in ("intrinio_security_id", "identity_eligible"):
            pd.testing.assert_series_equal(cohort[field], returns[field], check_dtype=False)
        outcome_request = json.loads(paths["monthly_provenance"].read_text())["requested_security_ids"]
        need(set(outcome_request) == set(requested) and len(outcome_request) == len(requested), "Past-feature and outcome request scopes differ")
        need(np.array_equal(targets.outcome_requested, returns.intrinio_security_id.isin(outcome_request)),
             "Target outcome-request flags used successful or future-return availability")
        finite = np.isfinite(features.iloc[:, 2:].to_numpy()).sum(axis=1)
        need(np.array_equal(cohort.available_feature_count, finite) and cohort.registered_feature_count.eq(728).all(), "Saved feature availability counts differ")
        # Price projections: all exchange month-ends for Paper15,24 endpoints for labels,
        # and21 sessions for two metadata-selected native IDs at each signal.
        months = {}
        for day in sessions:
            if day <= last_signal: months[day[:7]] = day
        pquotes = budget.frame(paths["feature_history_daily"], columns=QUALITY, filters=[("trading_date", "in", list(months.values()))])
        endpoints = sorted(set(windows.signal_date) | set(windows.outcome_end))
        oquotes = budget.frame(paths["daily_history_raw"], columns=QUALITY, filters=[("trading_date", "in", endpoints)])
        sample_ids, sample_dates = set(), set()
        for signal, rows in cohort.groupby("signal_date", sort=True):
            sample = rows.loc[rows.identity_eligible & rows.requested].sort_values(["intrinio_security_id", "source_security_id"]).drop_duplicates("intrinio_security_id").head(2)
            sample_ids.update(sample.intrinio_security_id)
            slot = sessions.index(signal); sample_dates.update(sessions[slot-20:slot+1])
        dquotes = budget.frame(paths["feature_history_daily"], columns=QUALITY, filters=[("intrinio_security_id", "in", sorted(sample_ids)), ("trading_date", "in", sorted(sample_dates))])
        report = {"original_rows": len(original), "feature_rows": len(features), "signals": 12,
                  "historical_daily_cohort_rows_retained": len(daily), "503_session_raw_warmup_verified": True,
                  "252_observation_dated_context_verified": True, "feature_and_outcome_daily_views_separate": True,
                  "no2024_price_rows_returned": True, "quarantine_and_restrictions": budget.compute("quarantine", quarantine_check, original, cohort, features, targets),
                  "monthly_return_statuses_independently_recomputed": budget.compute("monthly endpoint recomputation", independent_monthly_returns, cohort, returns, oquotes, set(requested)),
                  "paper15": budget.compute("Paper15 recomputation", independent_paper15, cohort, features, pquotes, sessions, set(requested)),
                  "shared_classes": budget.compute("target class recomputation", independent_classes, cohort, targets, returns, diagnostics, state["provenance"]["label_observed_through"]),
                  "daily_formula_sample": budget.compute("daily formula samples", daily_samples, cohort, features, dquotes, sessions),
                  "feature_available_values": int(finite.sum()),
                  "feature_finite_counts": {name: int(np.isfinite(features[name].to_numpy()).sum()) for name in fnames[2:]},
                  "financial_projections_rows": {"past_month_ends": len(pquotes), "label_endpoints": len(oquotes), "daily_sample": len(dquotes)}}
        result["years"][str(year)] = report; total_rows += len(original)
        del features, targets, returns, diagnostics, pquotes, oquotes, dquotes, raw_metas, daily, original, cohort
        gc.collect()
        budget.check("after annual verification " + str(year))
    need(total_rows == 43655, "Four-year original denominator changed")
    result["original_rows"] = total_rows; result["months"] = 48
    result["source_preservation_after"] = preservation(root, run, cfg, budget)
    result["source_preservation_count"] = result["source_preservation_after"]["files_hashed"]
    result["execution_budget"] = {"maximum_wall_seconds": budget.seconds, "maximum_peak_rss_bytes": budget.bytes,
                                  "model_fits": 0, "threads": 1, "checks_between_hash_chunks_reads_and_recomputations": True}
    budget.check("verification completed")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--expected-identity-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(); started = time.monotonic()
    need(not args.output.exists(), "Preserve an existing independent verification receipt")
    run = args.run if args.run.is_absolute() else args.root / args.run
    result = run_verification(args.root, run, args.expected_identity_sha256)
    result["elapsed_seconds"] = time.monotonic() - started
    result["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
    need(result["elapsed_seconds"] <= 1800 and result["peak_rss_bytes"] <= 8 * 1024**3, "Independent verification resource budget exceeded")
    result["verifier_sha256"] = digest(__file__)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "original_rows": result["original_rows"], "months": 48, "model_fits": 0, "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
