"""Independent saved-output audit; never imports the production target workers.

Run explicitly after the authorized build. Financial inspection is confined to
the completed 2010--2023 targets/returns and three predetermined endpoint samples.

Example (from the configured V2 workspace):
  .venv/bin/python scripts/verify_full_target_panel.py --root . \
      --run outputs/full_target_panel/2010-2023-<identity> --output /tmp/verification.json

The selected run must match LATEST or supply --expected-audit-sha256 explicitly.
All timing metadata and file pins are checked before financial conversion.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


KEYS = ["source_security_id", "signal_date"]
CLASSES = list(range(1, 11))
SAMPLES = ("2009-12-31", "2014-12-31", "2023-11-30")


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def canonical_digest(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                     allow_nan=False).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def contained(base, path):
    base, path = Path(base).resolve(), Path(path).resolve()
    require(path.is_relative_to(base), "Audited resource escapes its declared directory")
    return path


def check_entries(base, entries):
    require(isinstance(entries, dict) and entries, "An audit is missing its artifact manifest")
    checked = {}
    for item in entries.values():
        require(isinstance(item, dict) and isinstance(item.get("path"), str), "Invalid artifact entry")
        path = contained(base, Path(base) / item["path"])
        expected = item["sha256"].removeprefix("sha256:")
        require(path.is_file() and digest(path) == expected, "Saved artifact hash changed: " + str(path))
        require(path.stat().st_size == item["size_bytes"], "Saved artifact size changed: " + str(path))
        checked[str(path)] = expected
    return checked


def verified_run(root, run, expected_audit):
    path = contained(root / "outputs/full_target_panel", run / "audit.json")
    if expected_audit is None:
        pointer = json.loads((root / "outputs/full_target_panel/LATEST.json").read_text())
        require(contained(root / "outputs/full_target_panel", path.parent.parent / pointer["path"]) == run,
                "An older run requires an explicit --expected-audit-sha256")
        expected_audit = pointer["audit_sha256"]
    require(digest(path) == expected_audit.removeprefix("sha256:"), "Root target audit differs from its explicit pin")
    audit = json.loads(path.read_text())
    require(audit.get("status") == "COMPLETE" and set(audit.get("years", {})) == {str(y) for y in range(2010, 2024)},
            "The target run is incomplete or has a different year scope")
    checked = {str(path): digest(path), **check_entries(run, audit["artifacts"]),
               **check_entries(run, audit["years"])}
    identity = json.loads((run / "build_identity.json").read_text())
    require(canonical_digest(identity) == audit["build_identity_sha256"], "Saved recipe identity differs from its canonical hash")
    cfg = identity["configuration"]
    require(cfg["start_year"] == 2010 and cfg["end_year"] == 2023 and cfg["universe"] == "combined"
            and cfg["execution"]["raw_source_fallback"] is False
            and cfg["execution"]["model_fitting"] is False, "Saved target recipe changed the authorized scope")
    for year in range(2010, 2024):
        directory = run / "years" / str(year)
        annual = json.loads((directory / "audit.json").read_text())
        require(annual.get("status") == "COMPLETE" and annual.get("year") == year
                and annual.get("build_identity_sha256") == audit["build_identity_sha256"],
                "An annual target checkpoint is incomplete or changed recipe/year")
        checked.update(check_entries(directory, annual["artifacts"]))
    state = json.loads((run / "target_state.json").read_text())
    require(canonical_digest(state["provenance"]) == state["artifact_sha256"]
            == audit["summary"]["targets"]["target_panel_sha256"], "Target-state fingerprint changed")
    roles = json.loads((root / "configs/context_panel_roles.json").read_text())
    require(identity["feature_roles"] == roles, "Saved build and active feature-role pins differ")
    return audit, state, checked


def iso(value, name):
    require(isinstance(value, str), name + " is not an explicit date")
    require(date.fromisoformat(value).isoformat() == value, name + " is not canonical ISO")
    return value


def validate_timing_metadata(root, run, state):
    """Only identifiers, calendar dates and flags are decoded in this stage."""
    fields = [*KEYS, "outcome_month", "outcome_start", "outcome_end"]
    metadata = {}
    for name in ("targets", "monthly_returns", "target_diagnostics"):
        columns = fields if name != "target_diagnostics" else fields[1:]
        frame = pq.read_table(resource(run, name), columns=columns).to_pandas()
        require(len(frame) > 0, "Empty timing metadata: " + name)
        for row in frame.itertuples(index=False):
            signal = iso(row.signal_date, "signal_date")
            start, end = iso(row.outcome_start, "outcome_start"), iso(row.outcome_end, "outcome_end")
            month = pd.Period(row.outcome_month, freq="M")
            require(str(month) == row.outcome_month and "2010-01" <= row.outcome_month <= "2023-12"
                    and start[:7] == end[:7] == row.outcome_month
                    and signal[:7] == str(month - 1) and signal < start <= end < "2024-01-01",
                    "Target timing metadata crosses signal, month or protected boundaries")
        metadata[name] = frame
    target = metadata["targets"].set_index(KEYS).sort_index()
    returns = metadata["monthly_returns"].set_index(KEYS).sort_index()
    require(target.index.is_unique and target.index.equals(returns.index), "Return/target metadata changed original keys")
    pd.testing.assert_frame_equal(target, returns, check_dtype=False, check_exact=True)
    windows = metadata["target_diagnostics"].sort_values("signal_date").reset_index(drop=True)
    unique = metadata["targets"].drop(columns="source_security_id").drop_duplicates().sort_values("signal_date").reset_index(drop=True)
    pd.testing.assert_frame_equal(unique, windows, check_dtype=False, check_exact=True)
    require(len(windows) == 168, "Timing metadata lacks all 168 independent outcome months")
    declared = pd.DataFrame(state["provenance"]["month_windows"])[list(windows)].sort_values("signal_date").reset_index(drop=True)
    pd.testing.assert_frame_equal(declared, windows, check_dtype=False, check_exact=True)
    cutoff = iso(state["provenance"]["label_observed_through"], "label observation cutoff")
    require(cutoff == windows.outcome_end.max() < "2024-01-01", "Target state exceeds completed pre-holdout outcomes")
    roles = json.loads((root / "configs/context_panel_roles.json").read_text())
    for year in range(2010, 2024):
        year_dir = contained(root, root / roles["primary"]["directory"] / "years" / str(year))
        calendar = pq.read_table(year_dir / "exchange_calendar.parquet", columns=["trading_date", "is_trading_session"]).to_pandas()
        calendar = calendar[calendar.is_trading_session].trading_date.tolist()
        require(len(calendar) == len(set(calendar)), "Duplicate independent exchange-calendar dates")
        for row in windows[windows.outcome_month.str.startswith(str(year))].itertuples(index=False):
            prior = str(pd.Period(row.outcome_month, freq="M") - 1)
            following = str(pd.Period(row.outcome_month, freq="M") + 1)
            previous_dates = [d for d in calendar if d[:7] == prior]
            current_dates = [d for d in calendar if d[:7] == row.outcome_month]
            require(previous_dates and current_dates and any(d[:7] == following for d in calendar),
                    "Independent adjacent-month completion evidence is absent")
            require((row.signal_date, row.outcome_start, row.outcome_end) ==
                    (max(previous_dates), min(current_dates), max(current_dates)),
                    "Target endpoints differ from pinned independent exchange sessions")
    signals = set(windows.signal_date)
    ret_signal = pq.read_table(resource(run, "ret_coverage_signal"), columns=["signal_date", "label_observed_through"]).to_pandas()
    require(len(ret_signal) == 168 and set(ret_signal.signal_date) == signals
            and ret_signal.signal_date.equals(ret_signal.label_observed_through), "RET signal metadata changed cutoffs")
    grid = pq.read_table(resource(run, "ret_month_class_grid"), columns=["signal_date", "outcome_month", "outcome_end"]).to_pandas()
    for signal, rows in grid.groupby("signal_date"):
        iso(signal, "RET signal")
        expected = set(pd.period_range(pd.Period(signal[:7], freq="M") - 119,
                                      pd.Period(signal[:7], freq="M"), freq="M").astype(str))
        require(signal in signals and len(rows) == 1200 and set(rows.outcome_month) == expected,
                "RET timing grid is not a complete trailing 120-calendar-month window")
        for end in rows.outcome_end.dropna().unique():
            require(iso(end, "RET outcome end") <= signal < "2024-01-01", "RET would decode an outcome after its signal")
    for signal in sorted(signals):
        outcome_month = str(pd.Period(signal[:7], freq="M") + 1)
        if "2020-01" <= outcome_month <= "2023-12":
            directory = run / "ret_states" / signal
            value = json.loads((directory / "state.json").read_text())["provenance"]
            require(value["signal_date"] == value["label_observed_through"] == signal,
                    "Canonical payoff state changed cutoff")
            dates = pq.read_table(directory / "monthly.parquet", columns=["outcome_end"]).column(0).to_pylist()
            require(all(end is None or iso(end, "Canonical payoff month end") <= signal for end in dates),
                    "Canonical state contains a financial month after its cutoff")
    return {"original_metadata_rows_checked": len(target), "months_checked": len(windows),
            "independent_calendar_endpoints_verified": True, "all_ret_metadata_checked_before_financial_conversion": True}


def resource(directory, name):
    paths = [directory / (name + ".parquet"), directory / "reports" / (name + ".parquet")]
    matches = [path for path in paths if path.is_file()]
    require(len(matches) == 1, "Expected one saved table: " + name)
    return contained(directory, matches[0])


def read_frame(directory, name):
    return pq.read_table(resource(directory, name)).to_pandas()


def compare_series(actual, expected, name):
    pd.testing.assert_series_equal(actual, expected, check_names=False, check_dtype=False,
                                   check_exact=True, obj=name)


def assert_close(actual, expected, name):
    require(np.allclose(actual, expected, rtol=1e-13, atol=1e-15, equal_nan=True), name)


def checked_source_manifest(root, run, explicit):
    path = Path(explicit) if explicit else run / "preserved_source_manifest.json"
    require(path.resolve() == (run / "preserved_source_manifest.json").resolve(),
            "Use the preservation manifest pinned by the completed target audit")
    value = json.loads(path.read_text())
    entries = value.get("artifacts", value.get("files", value)) if isinstance(value, dict) else value
    if isinstance(entries, dict):
        entries = [{"path": name, **item} for name, item in entries.items()]
    require(isinstance(entries, list) and entries, "Missing preservation manifest entries")
    result = {}
    for entry in entries:
        target = Path(entry["path"])
        if not target.is_absolute():
            target = root / target
        target = target.resolve()
        require(target.is_relative_to(root.resolve()), "Preserved source escapes project")
        expected = entry["sha256"].removeprefix("sha256:")
        require(target.is_file() and digest(target) == expected, "Preserved source changed: " + str(target))
        if "size_bytes" in entry:
            require(target.stat().st_size == entry["size_bytes"], "Preserved source size changed")
        result[str(target)] = expected
    return result


def source_cohorts(root):
    roles = json.loads((root / "configs/context_panel_roles.json").read_text())
    primary, sensitivity, originals = [], [], []
    for year in range(2010, 2024):
        parts = []
        for role in ("primary", "sensitivity"):
            year_dir = root / roles[role]["directory"] / "years" / str(year)
            parts.append(pq.read_table(year_dir / "feature_coverage.parquet", columns=[
                *KEYS, "intrinio_security_id", "identity_eligible", "requested", "paper_prediction_eligible"
            ]).to_pandas().set_index(KEYS).sort_index())
        pd.testing.assert_frame_equal(parts[0], parts[1], check_dtype=False, check_exact=True)
        primary.append(parts[0])
        sensitivity.append(parts[1])
        original_dir = root / roles["primary"]["directory"] / "years" / str(year)
        originals.append(pq.read_table(original_dir / "original_cohort.parquet", columns=KEYS)
                         .to_pandas().set_index(KEYS).sort_index())
    original = pd.concat(originals).sort_index()
    cohort = pd.concat(primary).sort_index()
    require(cohort.index.equals(original.index) and cohort.index.is_unique,
            "Feature ledgers changed original membership keys")
    return roles, cohort


def verify_classes(target, returns, diagnostics, cohort):
    require(len(target) == 152307 and len(diagnostics) == 168,
            "Unexpected full-panel stock-month/month denominator")
    target = target.set_index(KEYS).sort_index()
    returns = returns.set_index(KEYS).sort_index()
    require(target.index.is_unique and target.index.equals(cohort.index), "Targets changed original keys")
    require(returns.index.is_unique and returns.index.equals(cohort.index), "Monthly returns changed original keys")
    for field in ("identity_eligible", "requested", "paper_prediction_eligible"):
        compare_series(target[field], cohort[field], field)
    eligible = cohort.identity_eligible & cohort.paper_prediction_eligible
    compare_series(target.prediction_eligible, eligible, "Signal-time prediction cohort")
    require(int(eligible.sum()) == 140777, "Paper15 prediction cohort changed")
    require(set(target.outcome_month) == set(pd.period_range("2010-01", "2023-12", freq="M").astype(str)),
            "Target history changed its consecutive calendar months")
    require((target.outcome_end < "2024-01-01").all() and
            (target.index.get_level_values("signal_date") < target.outcome_start).all(),
            "Target timing crosses signal/holdout boundaries")
    require((returns.outcome_month == target.outcome_month).all() and
            (returns.outcome_end == target.outcome_end).all(), "Return/target windows changed")
    observed = (eligible & target.outcome_requested.fillna(False) &
                returns.return_status.eq("ok") & np.isfinite(returns.provider_adjusted_return) &
                returns.provider_adjusted_return.ge(-1))
    compare_series(target.return_available, observed, "Observed return availability")
    compare_series(target.observed_return_available, observed, "Observed-return audit flag")
    compare_series(target.label_return, returns.provider_adjusted_return.where(observed), "Label returns")
    by_month = diagnostics.set_index("outcome_month")
    classes_checked = ties = 0
    for month, rows in target.groupby("outcome_month", sort=True):
        valid = rows[rows.return_available]
        admitted = rows[rows.identity_eligible]
        complete_request = admitted.requested.all() and admitted.outcome_requested.eq(True).all()
        d = by_month.loc[month]
        require(bool(d.exhaustive_requests) == bool(complete_request), "Request completeness differs")
        if complete_request and len(valid) >= 10:
            # Pandas independently forms exact tie groups. Twice-average-rank is
            # integral, avoiding floating boundaries when assigning decile bins.
            ranks2 = (valid.label_return.rank(method="average", ascending=False) * 2).astype("int64")
            expected = (1 + (10 * (ranks2 - 1)) // (2 * len(valid))).clip(1, 10)
            compare_series(valid.target_class, expected, "Independent midrank classes " + month)
            require(rows.target_available.equals(rows.return_available), "Released classes lost observed members")
            require(d.rank_status == "ok", "A class-ready month was withheld")
            classes_checked += len(valid)
        else:
            require(not rows.target_available.any() and rows.target_class.isna().all(),
                    "An incomplete-request/undersized month released classes")
        require(rows.loc[~rows.target_available, "target_class"].isna().all(), "Unavailable class was filled")
        counts = rows.target_class.value_counts().to_dict()
        require(json.loads(d.class_counts_json) == {str(k): int(counts.get(k, 0)) for k in CLASSES},
                "Saved class counts differ")
        sizes = valid.label_return.value_counts()
        tied = sizes[sizes > 1]
        require(int(d.tied_return_groups) == len(tied), "Tie-group count differs")
        require(int(d.maximum_tie_size) == (int(sizes.max()) if len(sizes) else 0), "Maximum tie size differs")
        require(int(d.original_members) == len(rows) and int(d.prediction_eligible_members) == rows.prediction_eligible.sum()
                and int(d.observed_return_members) == rows.return_available.sum()
                and int(d.labelled_members) == rows.target_available.sum(), "Monthly coverage counts differ")
        ties += len(tied)
    return target, returns, {"original_rows": len(target), "calendar_months": len(by_month),
        "prediction_eligible_rows": int(eligible.sum()), "observed_return_rows": int(observed.sum()),
        "classes_independently_recomputed": classes_checked, "tied_return_groups_checked": ties,
        "primary_sensitivity_original_keys_and_admission_exact_equal": True}


def verify_ret(run, target, diagnostics):
    coverage = read_frame(run, "ret_coverage_signal").sort_values("signal_date")
    grid = read_frame(run, "ret_month_class_grid")
    payoffs = read_frame(run, "ret_class_payoffs")
    require(len(coverage) == 168 and len(grid) == 168 * 1200 and len(payoffs) == 168 * 10,
            "RET audit omitted signal/calendar/class cells")
    require(set(coverage.signal_date) == set(target.index.get_level_values("signal_date")),
            "RET audit changed forecast signals")
    ends = diagnostics.set_index("outcome_month").outcome_end.to_dict()
    values = target[target.target_available].reset_index().groupby(["outcome_month", "target_class"]).label_return
    means = values.mean().to_dict()
    counts = values.size().to_dict()
    originals = target.groupby("outcome_month").size().to_dict()
    eligible = target.groupby("outcome_month").prediction_eligible.sum().to_dict()
    grid = grid.set_index(["signal_date", "outcome_month", "class_id"]).sort_index()
    payoffs = payoffs.set_index(["signal_date", "class_id"]).sort_index()
    require(grid.index.is_unique and payoffs.index.is_unique, "Duplicate RET state keys")
    nominal_checked = available_count = 0
    for row in coverage.itertuples(index=False):
        signal = row.signal_date
        last = pd.Period(signal[:7], freq="M")
        months = pd.period_range(last - 119, last, freq="M").astype(str).tolist()
        require(row.first_outcome_month == months[0] and row.last_outcome_month == months[-1]
                and row.label_observed_through == signal, "RET window/cutoff changed")
        expected_keys = pd.MultiIndex.from_product([[signal], months, CLASSES], names=grid.index.names)
        current = grid.loc[signal]
        require(current.index.equals(expected_keys.droplevel(0)), "RET calendar grid was shortened/compressed")
        all_available = True
        independent_means = {k: [] for k in CLASSES}
        status_counts = Counter()
        for month in months:
            for k in CLASSES:
                actual = grid.loc[(signal, month, k)]
                if month not in ends:
                    status = "missing_month_completion_evidence"
                elif ends[month] > signal:
                    status = "month_not_observed_by_cutoff"
                elif not originals.get(month, 0):
                    status = "missing_month"
                elif not counts.get((month, k), 0):
                    status = "empty_realized_class"
                else:
                    status = "available"
                require(actual.status == status, "RET monthly class status differs")
                require(int(actual.original_member_count) == originals.get(month, 0) and
                        int(actual.prediction_eligible_count) == eligible.get(month, 0), "RET denominator differs")
                status_counts[status] += 1
                if status == "available":
                    require(int(actual.member_count) == counts[(month, k)], "RET class member count differs")
                    expected_mean = means[(month, k)]
                    assert_close(actual.class_mean_return, expected_mean, "Independent monthly class mean differs")
                    independent_means[k].append(expected_mean)
                else:
                    all_available = False
                    require(pd.isna(actual.class_mean_return), "Unavailable class mean was filled")
        require(bool(row.vector_available) == all_available and
                int(row.available_month_class_cells) == status_counts["available"], "RET vector availability differs")
        require(json.loads(row.missing_month_class_status_counts_json) == dict(status_counts), "RET status counts differ")
        require(bool(row.nominal_test_signal) == ("2020-01" <= str(last + 1) <= "2023-12"),
                "RET nominal test chronology differs")
        for k in CLASSES:
            actual = payoffs.loc[(signal, k)]
            if all_available:
                require(actual.status == "available", "Complete RET vector was withheld")
                assert_close(actual.payoff, float(np.mean(independent_means[k])), "Independent equal-month payoff differs")
            else:
                require(actual.status == "withheld_incomplete_120_month_class_grid" and pd.isna(actual.payoff),
                        "An incomplete RET vector was released or imputed")
        if row.nominal_test_signal:
            state_dir = run / "ret_states" / signal
            state = json.loads((state_dir / "state.json").read_text())
            require(state["provenance"]["signal_date"] == signal and
                    state["provenance"]["label_observed_through"] == signal,
                    "Persisted canonical RET state changed cutoff")
            saved_payoffs = pq.read_table(state_dir / "payoffs.parquet").to_pandas().set_index("class_id").sort_index()
            saved_monthly = pq.read_table(state_dir / "monthly.parquet").to_pandas().set_index(["outcome_month", "class_id"]).sort_index()
            pd.testing.assert_frame_equal(saved_payoffs, payoffs.loc[signal], check_dtype=False,
                                          check_index_type=False, check_exact=True)
            pd.testing.assert_frame_equal(saved_monthly, current, check_dtype=False,
                                          check_index_type=False, check_exact=True)
        available_count += int(all_available)
        nominal_checked += int(row.nominal_test_signal)
    require(nominal_checked == 48, "Nominal 2020--2023 signals changed")
    return {"signals_independently_verified": len(coverage), "month_class_cells_verified": len(grid),
            "class_payoffs_verified": len(payoffs), "nominal_test_vectors_verified": nominal_checked,
            "available_vectors": available_count, "class_means_independently_recomputed": len(means),
            "equal_month_not_pooled_weighting_verified": True}


def verify_endpoints(root, roles, returns):
    sample_rows = observations = 0
    statuses = Counter()
    for signal in SAMPLES:
        monthly = returns.xs(signal, level="signal_date")
        require(len(monthly) > 0, "Predetermined endpoint sample absent")
        year = monthly.outcome_month.iloc[0][:4]
        source_dir = root / roles["primary"]["directory"] / "years" / year
        audit = json.loads((source_dir / "audit.json").read_text())
        raw = source_dir / audit["artifacts"]["daily_history_raw"]["path"]
        before = digest(raw)
        require(before == audit["artifacts"]["daily_history_raw"]["sha256"], "Endpoint source hash changed")
        source_loading = audit["loading_provenance"].get("reference_loading_provenance", audit["loading_provenance"])
        request = source_loading["stages"]["raw_daily_prices"]["request"]
        requested = set(request["filters"]["intrinio_security_id"])
        require(set(audit["feature_provenance"]["requested_ids"]) == requested,
                "Past/forward actual request manifests differ")
        end = monthly.outcome_end.iloc[0]
        require(request["start"] <= signal < end <= request["end"] < "2024-01-01", "Endpoint sample outside request bounds")
        # Check the entire cache's nonfinancial keys before decoding adjusted
        # closes, even though only two predetermined dates will be returned.
        raw_metadata = pq.read_table(raw, columns=["intrinio_security_id", "trading_date"]).to_pandas()
        require(not raw_metadata.duplicated(["intrinio_security_id", "trading_date"]).any(),
                "Duplicate source quote metadata key")
        require(set(raw_metadata.intrinio_security_id) <= requested, "Raw cache contains an unrequested native ID")
        for day in raw_metadata.trading_date.unique():
            require(request["start"] <= iso(day, "Raw cache quote date") <= request["end"] < "2024-01-01",
                    "Raw cache metadata exceeds its executed request or protected boundary")
        table = pq.read_table(raw, columns=["intrinio_security_id", "trading_date", "adj_close",
            "frequency", "intraperiod", "price_quality_status"], filters=[("trading_date", "in", [signal, end])])
        quotes = table.to_pandas().set_index(["intrinio_security_id", "trading_date"])
        require(quotes.index.is_unique, "Duplicate endpoint source key")
        observations += len(quotes)
        for row in monthly.itertuples():
            native = row.intrinio_security_id
            start = quotes.loc[(native, signal)] if (native, signal) in quotes.index else None
            finish = quotes.loc[(native, end)] if (native, end) in quotes.index else None
            if not row.identity_eligible:
                status = "identity_unresolved"
            elif native not in requested:
                status = "not_requested"
            elif start is None:
                status = "missing_start"
            elif finish is None:
                status = "missing_end"
            elif not (math.isfinite(start.adj_close) and start.adj_close > 0 and
                      math.isfinite(finish.adj_close) and finish.adj_close > 0):
                status = "invalid_endpoint"
            elif not all(q.frequency == "daily" and q.intraperiod == False and
                         q.price_quality_status in ("accepted", "ok") for q in (start, finish)):
                status = "source_quality_flag"
            else:
                status = "ok"
                ratio = finish.adj_close / start.adj_close - 1
                if not math.isfinite(ratio):
                    status = "invalid_endpoint"
            require(row.return_status == status, "Independent exact-endpoint return status differs")
            if status == "ok":
                require(row.provider_adjusted_return == ratio, "Independent exact-endpoint ratio differs")
            else:
                require(pd.isna(row.provider_adjusted_return), "Rejected endpoint ratio was filled")
            sample_rows += 1
            statuses[status] += 1
        require(digest(raw) == before, "Endpoint verification modified raw source")
    return {"fixed_signal_dates": list(SAMPLES), "original_rows_checked": sample_rows,
            "projected_endpoint_quotes_checked": observations, "independent_status_counts": dict(statuses),
            "source_request_ids_used_instead_of_future_quote_inventory": True,
            "raw_metadata_validated_before_financial_projection": True,
            "raw_financial_columns_read": ["adj_close"], "source_years_read": [2010, 2015, 2023]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--source-manifest")
    parser.add_argument("--expected-audit-sha256", help="Explicit immutable root-audit pin for an older saved run")
    parser.add_argument("--output")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    run = Path(args.run)
    run = (run if run.is_absolute() else root / run).resolve()
    require(run.is_relative_to(root / "outputs"), "Target verification requires a private project output run")
    audit, state, run_files = verified_run(root, run, args.expected_audit_sha256)
    preserved = checked_source_manifest(root, run, args.source_manifest)
    timing = validate_timing_metadata(root, run, state)
    roles, cohort = source_cohorts(root)
    targets, returns, diagnostics = [read_frame(run, name) for name in
                                    ("targets", "monthly_returns", "target_diagnostics")]
    targets, returns, classes = verify_classes(targets, returns, diagnostics, cohort)
    ret = verify_ret(run, targets, diagnostics)
    endpoints = verify_endpoints(root, roles, returns)
    for path, expected in preserved.items():
        require(digest(path) == expected, "Preserved source changed during independent verification")
    for path, expected in run_files.items():
        require(digest(path) == expected, "Target artifact changed during independent verification")
    report = {"status": "PASS", "independent_of_production_workers": True,
              "run_directory": str(run), "run_path_relative_to_project": str(run.relative_to(root)),
              "build_identity_sha256": audit["build_identity_sha256"],
              "root_audit_sha256": digest(run / "audit.json"),
              "target_panel_sha256": state["artifact_sha256"],
              "verifier_sha256": digest(Path(__file__)),
              "target_artifacts_verified_before_and_after": len(run_files),
              "timing_metadata": timing, "classes": classes, "ret": ret, "endpoints": endpoints,
              "preserved_source_files_verified_before_and_after": len(preserved),
              "daily_or_feature_data_modified": False, "models_fitted": False,
              "protected_2024_financial_values_read": False}
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        output = Path(args.output).resolve()
        require(str(output) not in {*preserved, *run_files}, "Cannot overwrite an audited input with a receipt")
        output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
