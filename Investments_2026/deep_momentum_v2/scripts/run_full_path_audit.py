"""Explicit immutable annual/monthly full path-label audit; never fits models."""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import resource
import sys
import time

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from dmv2.full_path_audit import (PROTOCOL, YEARS, SIDES, VARIANTS, need, pin, verify_pin,
    read_json, hash_file, plan_full_path_audit, verify_plan, load_annual_values,
    core_cohort, build_month_paths, read_full_path_audit)
from dmv2.path_targets import build_path_targets
from dmv2.catalog import canonical_sha256


def write_json(path, value):
    path = Path(path)
    need(not path.exists(), "Never overwrite an immutable audit/checkpoint")
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False, default=str) + "\n")


def relative_pin(path, directory):
    value = pin(path)
    value["path"] = str(Path(path).relative_to(directory))
    return value


def artifact_receipt(directory, plan_hash):
    report = read_json(Path(directory) / "audit.json")
    need(report["plan_sha256"] == plan_hash and report.get("protocol") == PROTOCOL
        and report["status"] in {"COMPLETE_MONTHLY_PATH_AUDIT", "COMPLETE_ANNUAL_PATH_AUDIT"}, "Checkpoint plan/status changed")
    for name, value in report.get("artifacts", {}).items():
        path = (Path(directory) / name).resolve()
        need(path.is_relative_to(Path(directory).resolve()), "Unsafe checkpoint artifact")
        verify_pin({**value, "path": str(path)})
    return report


def checkpoint(root, year, month, plan_hash):
    directory = root / "years" / str(year) / "months" / month
    control = directory / "checkpoint.json"
    if not control.exists():
        return None
    item = read_json(control)
    need(item["plan_sha256"] == plan_hash, "Monthly checkpoint belongs to another plan")
    target = (directory / item["attempt"]).resolve()
    need(target.parent == directory.resolve(), "Unsafe attempt path")
    need(hash_file(target / "audit.json") == item["audit_sha256"], "Monthly audit changed")
    report = artifact_receipt(target, plan_hash)
    need(report["status"] == "COMPLETE_MONTHLY_PATH_AUDIT" and report["year"] == year and report["outcome_month"] == month,
        "Copied wrong-month checkpoint cannot be resumed")
    return report


def resources(plan, output, started, prior_seconds):
    elapsed = prior_seconds + time.monotonic() - started
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss = int(rss if sys.platform == "darwin" else rss * 1024)
    need(elapsed <= plan["limits"]["active_wall_seconds"], "Full audit active wall-time ceiling reached; preserve checkpoints")
    need(rss <= plan["limits"]["peak_rss_bytes"], "Full audit RSS ceiling exceeded")
    size = sum(p.stat().st_size for p in Path(output).rglob("*") if p.is_file())
    need(size <= plan["limits"]["output_bytes_total"], "Full audit total output ceiling exceeded")
    for annual in (Path(output) / "years").glob("*"):
        need(sum(p.stat().st_size for p in annual.rglob("*") if p.is_file()) <= plan["limits"]["output_bytes_per_year"],
            "Full audit annual output ceiling exceeded")
    return elapsed, rss


def monthly_tables(table, legacy):
    frame = table.to_pandas()
    controls = legacy.rename(columns={"label_return": "legacy_label_return", "target_class": "legacy_target_class"})
    comparison = frame[["source_security_id", "signal_date", "side", "path_variant", "target_family",
        "prediction_eligible", "path_available", "target_available", "cash_return", "maximum_drawdown", "utility", "target_value", "target_class"]].merge(
            controls, on=["source_security_id", "signal_date"], how="left", validate="many_to_one")
    comparison["difference_from_legacy_endpoint_return"] = comparison.target_value - comparison.legacy_label_return
    summaries, grid, ties = [], [], []
    for group, rows in comparison.groupby(["side", "path_variant", "target_family"], sort=True):
        keys = dict(zip(("side", "path_variant", "target_family"), group))
        available = rows.loc[rows.target_available]
        overlap = rows.target_class.notna() & rows.legacy_target_class.notna()
        both = rows.path_available & rows.legacy_label_return.notna()
        summaries.append({**keys, "original_rows": len(rows), "prediction_eligible_rows": int(rows.prediction_eligible.sum()),
            "path_available_rows": int(rows.path_available.sum()), "target_available_rows": len(available),
            "class_comparison_rows": int(overlap.sum()),
            "class_changed_rows": int((rows.loc[overlap, "target_class"] != rows.loc[overlap, "legacy_target_class"]).sum()),
            "legacy_only_rows": int((rows.legacy_label_return.notna() & ~rows.path_available).sum()),
            "executable_only_rows": int((rows.path_available & rows.legacy_label_return.isna()).sum()),
            "mean_absolute_difference_from_endpoint_return": None if not both.any() else float(rows.loc[both, "difference_from_legacy_endpoint_return"].abs().mean()),
            "short_vs_legacy_long_class_is_diagnostic_not_same_target": group[0] == "short"})
        for class_id in range(1, 11):
            count = int((available.target_class == class_id).sum())
            grid.append({**keys, "class_id": class_id, "members": count, "status": "available" if count else "empty_realized_class"})
        sizes = available.target_value.value_counts(dropna=False).value_counts().sort_index()
        ties.extend({**keys, "equal_value_group_size": int(size), "number_of_groups": int(count)} for size, count in sizes.items())
    return comparison, summaries, grid, ties


def supplemental_overlay(plan, cohort, primary_index, window, overlay, primary_episodes, primary_marks, registry):
    """Replay only potentially affected original episodes; never promote quotes.

    The candidate superset is based on each original route plus its reviewed
    successor chain, independently of observed future quote availability.
    """
    from dmv2.full_path_accounting import prepare_price_index
    if overlay.empty or window["exit_date"] >= "2024-01-01":
        return None
    patch = overlay.loc[overlay.trading_date.between(window["entry_date"], window["exit_date"])]
    if patch.empty:
        return None
    routes = set(patch.intrinio_security_id)
    changed = True
    while changed:
        before = len(routes)
        routes.update(e["predecessor_id"] for e in plan["events"] if e.get("successor_id") in routes
            and window["entry_date"] <= e["effective_date"] <= window["exit_date"])
        changed = before != len(routes)
    selected = cohort.loc[cohort.prediction_eligible & cohort.intrinio_security_id.isin(routes)]
    if selected.empty:
        return None
    # A cheap fresh dictionary replaces exact quoted keys only. The pure index
    # adapter exposes its constructor contract; no source/learning files mutate.
    patch_index = prepare_price_index(patch, plan["sessions"])
    diagnostic_index = replace(primary_index, quotes={**primary_index.quotes, **patch_index.quotes})
    ep, marks, details = build_month_paths(plan, selected, diagnostic_index, window, registry=registry)
    selected_keys = set(selected.source_security_id)
    primary_ep = primary_episodes.to_pandas()
    primary_mark = primary_marks.to_pandas()
    ep_all = pd.concat([primary_ep.loc[~primary_ep.source_security_id.isin(selected_keys)], ep.to_pandas()], ignore_index=True)
    marks_all = pd.concat([primary_mark.loc[~primary_mark.source_security_id.isin(selected_keys)], marks.to_pandas()], ignore_index=True)
    panel = build_path_targets(core_cohort(cohort), ep_all, marks_all, windows=[window], sessions=plan["sessions"],
        observed_through="2023-12-31", risk_penalty=1.0, endpoint_policy="next_open")
    return panel, details, {"candidate_original_members": len(selected), "replayed_episode_count": len(ep),
        "not_learning_admitted": True, "full_cohort_class_ranks_recomputed": True}


def run_month(plan, annual, window, cohort, index, legacy, overlay, registry, output, check):
    year, month = annual["year"], window["outcome_month"]
    parent = output / "years" / str(year) / "months" / month
    parent.mkdir(parents=True, exist_ok=True)
    number = 1
    while (parent / f"attempt-{number:04d}").exists():
        number += 1
    directory = parent / f"attempt-{number:04d}"
    directory.mkdir()
    started = time.monotonic()
    write_json(directory / "STARTED.json", {"plan_sha256": plan["plan_sha256"], "month": month})
    try:
        episodes, marks, details = build_month_paths(plan, cohort, index, window, resource_check=check, registry=registry)
        panel = build_path_targets(core_cohort(cohort), episodes, marks, windows=[window], sessions=plan["sessions"],
            observed_through="2023-12-31", risk_penalty=1.0, endpoint_policy="next_open")
        comparison, summaries, grid, ties = monthly_tables(panel.table, legacy)
        artifacts = {}
        for name, table in (("episodes", episodes), ("marks", marks), ("episode_details", details), ("targets", panel.table),
            ("diagnostics", panel.diagnostics), ("legacy_alignment", pa.Table.from_pandas(comparison, preserve_index=False))):
            path = directory / (name + ".parquet")
            pq.write_table(table, path)
            artifacts[path.name] = relative_pin(path, directory)
        diagnostic = supplemental_overlay(plan, cohort, index, window, overlay, episodes, marks, registry)
        supplemental = None
        if diagnostic is not None:
            supplemental_panel, supplemental_details, supplemental = diagnostic
            primary = panel.table.select(["source_security_id", "signal_date", "side", "path_variant", "target_family", "path_available", "target_class", "target_value"]).to_pandas()
            other = supplemental_panel.table.select(primary.columns.tolist()).to_pandas()
            joined = primary.merge(other, on=["source_security_id", "signal_date", "side", "path_variant", "target_family"], suffixes=("_primary", "_diagnostic"), validate="one_to_one")
            changed_class = joined.target_class_primary.ne(joined.target_class_diagnostic) & ~(joined.target_class_primary.isna() & joined.target_class_diagnostic.isna())
            shared = joined.target_value_primary.notna() & joined.target_value_diagnostic.notna()
            supplemental.update({"target_rows": len(joined), "path_availability_changed_rows": int(joined.path_available_primary.ne(joined.path_available_diagnostic).sum()),
                "class_changed_rows": int(changed_class.sum()), "shared_numeric_value_changed_rows": int(joined.loc[shared, "target_value_primary"].ne(joined.loc[shared, "target_value_diagnostic"]).sum())})
            for name, table in (("supplemental_targets", supplemental_panel.table), ("supplemental_episode_details", supplemental_details)):
                path = directory / (name + ".parquet")
                pq.write_table(table, path)
                artifacts[path.name] = relative_pin(path, directory)
        episode_frame = episodes.to_pandas()
        report = {"status": "COMPLETE_MONTHLY_PATH_AUDIT", "protocol": PROTOCOL, "plan_sha256": plan["plan_sha256"],
            "year": year, "outcome_month": month, "signal_date": window["signal_date"], "exit_date": window["exit_date"],
            "original_rows": len(cohort), "prediction_eligible_rows": int(cohort.prediction_eligible.sum()),
            "original_key_sha256": canonical_sha256(sorted([list(key) for key in cohort.loc[:, ["source_security_id", "signal_date"]].itertuples(index=False, name=None)])),
            "eligible_key_sha256": canonical_sha256(sorted([list(key) for key in cohort.loc[cohort.prediction_eligible, ["source_security_id", "signal_date"]].itertuples(index=False, name=None)])),
            "episode_status_counts": dict(Counter(episode_frame.status)), "episode_reason_counts": dict(Counter(episode_frame.reason)),
            "legacy_capital_gate_would_trigger_episodes": sum(r.get("legacy_capital_gate_would_trigger") is True for r in details.to_pylist()),
            "negative_reference_nav_retained_marks": sum(r.get("negative_reference_nav_retained") or 0 for r in details.to_pylist()),
            "target_diagnostics": panel.diagnostics.to_pylist(), "class_grid": grid, "exact_value_tie_histogram": ties,
            "legacy_alignment": summaries, "supplemental_execution_overlay": supplemental,
            "endpoint_target_control_rebuilt": False, "models_fitted": False, "portfolio_backtest_run": False,
            "financial_2024_plus_read": False, "wall_seconds": time.monotonic() - started, "artifacts": artifacts}
        write_json(directory / "audit.json", report)
        check()
        write_json(parent / "checkpoint.json", {"plan_sha256": plan["plan_sha256"], "attempt": directory.name,
            "audit_sha256": hash_file(directory / "audit.json")})
        return report
    except BaseException as error:
        write_json(directory / "failure.json", {"exception": type(error).__name__, "message": str(error), "models_fitted": False})
        raise


def payoff_readiness(month_reports):
    """Coverage-only exact120-calendar-month ten-class grids, no payoffs/RET."""
    grid = {}
    for report in month_reports:
        for row in report["class_grid"]:
            grid[(report["outcome_month"], row["side"], row["path_variant"], row["target_family"], row["class_id"])] = row["members"]
    result = []
    for report in month_reports:
        signal = report["signal_date"]
        last = pd.Period(signal[:7], "M") - 1
        months = [str(last + offset) for offset in range(-119, 1)]
        for side in SIDES:
            for variant in VARIANTS:
                for family in ("executable_return", "return_minus_drawdown"):
                    missing = sum(not grid.get((month, side, variant, family, class_id), 0) for month in months for class_id in range(1, 11))
                    result.append({"signal_date": signal, "side": side, "path_variant": variant, "target_family": family,
                        "window_start": months[0], "window_end": months[-1], "required_calendar_months": 120,
                        "required_class_cells": 1200, "missing_class_cells": missing,
                        "status": "available" if missing == 0 else "withheld_incomplete_120_month_class_grid",
                        "early_stop_does_not_advance_label_availability": True})
    return result


def training_completion(month_reports):
    """Actual labelled rows purged by next-open availability in four fixed folds."""
    result = []
    for test_year in range(2020, 2024):
        cutoff = max(r["signal_date"] for r in month_reports if r["outcome_month"] == f"{test_year}-01")
        training = [r for r in month_reports if f"{test_year-10}-01" <= r["outcome_month"] <= f"{test_year-1}-12"]
        for side in SIDES:
            for variant in VARIANTS:
                for family in ("executable_return", "return_minus_drawdown"):
                    summaries = [(month, next(s for s in month["legacy_alignment"] if (s["side"], s["path_variant"], s["target_family"]) == (side, variant, family))) for month in training]
                    result.append({"test_year": test_year, "as_of": cutoff, "side": side, "path_variant": variant, "target_family": family,
                        "train_start_month": f"{test_year-10}-01", "train_end_month": f"{test_year-1}-12",
                        "original_training_rows": sum(s["original_rows"] for _, s in summaries),
                        "signal_eligible_training_rows": sum(s["prediction_eligible_rows"] for _, s in summaries),
                        "completed_supervised_rows": sum(s["target_available_rows"] for m, s in summaries if m["exit_date"] <= cutoff),
                        "purged_not_yet_available_label_rows": sum(s["target_available_rows"] for m, s in summaries if m["exit_date"] > cutoff),
                        "missing_path_or_rank_label_rows": sum(s["prediction_eligible_rows"] - s["target_available_rows"] for _, s in summaries),
                        "models_fitted": False})
    return result


def aggregate_coverage(months):
    result = []
    for side in SIDES:
        for variant in VARIANTS:
            for family in ("executable_return", "return_minus_drawdown"):
                selected = [r for month in months for r in month["legacy_alignment"]
                    if (r["side"], r["path_variant"], r["target_family"]) == (side, variant, family)]
                result.append({"side": side, "path_variant": variant, "target_family": family,
                    **{field: sum(row[field] for row in selected) for field in
                        ("original_rows", "prediction_eligible_rows", "path_available_rows", "target_available_rows", "class_comparison_rows", "class_changed_rows", "legacy_only_rows", "executable_only_rows")}})
    return result


def run(plan, output, *, resume=False, through_year=2023):
    from dmv2.full_path_accounting import prepare_price_index, prepare_action_registry
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    verify_plan(plan)
    need(through_year in YEARS, "Bounded checkpoint year must be2010–2023")
    output = Path(output).resolve()
    if output.exists():
        need(resume, "Existing immutable audit requires explicit resume")
        reviewed = read_json(output / "reviewed_plan.json")
        need(reviewed == plan, "Resume cannot alter the reviewed plan")
        need(not (output / "audit.json").exists(), "Completed full audit is immutable")
    else:
        need(not resume, "Resume requires an existing checkpoint root")
        output.mkdir(parents=True)
        write_json(output / "reviewed_plan.json", plan)
    run_directory = output / "run_attempts"
    run_directory.mkdir(exist_ok=True)
    starts = list(run_directory.glob("*.STARTED.json"))
    need(all((run_directory / (path.name.removesuffix(".STARTED.json") + ".json")).exists() for path in starts),
        "Interrupted run has unmeasured active time; preserve evidence and require a separately reviewed new resource plan")
    prior_runs = [read_json(path) for path in run_directory.glob("*.json") if not path.name.endswith(".STARTED.json")]
    prior_seconds = sum(item["active_wall_seconds"] for item in prior_runs)
    number = len(prior_runs) + 1
    started = time.monotonic()
    write_json(run_directory / f"run-{number:04d}.STARTED.json", {"plan_sha256": plan["plan_sha256"],
        "active_seconds_before_attempt": prior_seconds, "hard_kill_resume_policy": "fail_closed_unmeasured_active_time"})
    check = lambda: resources(plan, output, started, prior_seconds)
    months = []
    registry = prepare_action_registry(plan["events"], sessions=plan["sessions"])
    read_directory = output / "read_reservations"
    read_directory.mkdir(exist_ok=True)
    try:
        for annual in plan["annuals"]:
            year = annual["year"]
            if year > through_year:
                break
            pending = [w for w in annual["windows"] if checkpoint(output, year, w["outcome_month"], plan["plan_sha256"]) is None]
            if pending:
                check()
                reservations = list(read_directory.glob("*.json"))
                reserved = sum(read_json(p)["projected_compressed_bytes"] for p in reservations)
                proposed = annual["materialization_compressed_bytes"]
                need(plan["read_plan"]["planning_metadata_projected_compressed_bytes"] + reserved + proposed
                    <= plan["read_plan"]["maximum_projected_compressed_bytes"], "Cumulative source read reservation exceeded, including retries")
                write_json(read_directory / f"read-{len(reservations)+1:04d}.json", {"plan_sha256": plan["plan_sha256"],
                    "year": year, "projected_compressed_bytes": proposed, "financial_values_may_be_read": True})
                cohort, prices, legacy, overlay = load_annual_values(plan, year)
                index = prepare_price_index(prices, plan["sessions"])
            year_reports = []
            for window in annual["windows"]:
                month = window["outcome_month"]
                report = checkpoint(output, year, month, plan["plan_sha256"])
                if report is None:
                    selected = cohort.loc[cohort.signal_date == window["signal_date"]].copy()
                    old = legacy.loc[legacy.signal_date == window["signal_date"]].copy()
                    report = run_month(plan, annual, window, selected, index, old, overlay, registry, output, check)
                year_reports.append(report)
            year_dir = output / "years" / str(year)
            year_audit = year_dir / "audit.json"
            if not year_audit.exists():
                verify_plan(plan, year=year)
                write_json(year_audit, {"status": "COMPLETE_ANNUAL_PATH_AUDIT", "protocol": PROTOCOL,
                    "plan_sha256": plan["plan_sha256"], "year": year,
                    "original_rows": sum(m["original_rows"] for m in year_reports),
                    "prediction_eligible_rows": sum(m["prediction_eligible_rows"] for m in year_reports),
                    "months": [m["outcome_month"] for m in year_reports], "artifacts": {
                        str(path.relative_to(year_dir)): relative_pin(path, year_dir) for path in year_dir.rglob("checkpoint.json")}})
            else:
                artifact_receipt(year_dir, plan["plan_sha256"])
            months.extend(year_reports)
            print(json.dumps({"stage": "annual_checkpoint_complete", "year": year,
                "original_rows": sum(m["original_rows"] for m in year_reports)}), flush=True)
            if pending:
                del cohort, prices, legacy, overlay, index
        elapsed, peak = check()
        if through_year == 2023:
            need(len(months) == 168, "Full audit cannot truncate monthly coverage")
            readiness = payoff_readiness(months)
            pq.write_table(pa.Table.from_pylist(readiness), output / "payoff_readiness.parquet")
            grid = [{"outcome_month": m["outcome_month"], **row} for m in months for row in m["class_grid"]]
            pq.write_table(pa.Table.from_pylist(grid), output / "class_grid.parquet")
            ties = [{"outcome_month": m["outcome_month"], **row} for m in months for row in m["exact_value_tie_histogram"]]
            pq.write_table(pa.Table.from_pylist(ties), output / "tie_histogram.parquet")
            training = training_completion(months)
            pq.write_table(pa.Table.from_pylist(training), output / "training_completion.parquet")
            summary = {"status": "COMPLETE_FULL_PATH_AUDIT", "protocol": PROTOCOL, "plan_sha256": plan["plan_sha256"],
                "years": list(YEARS), "original_months": len(months), "original_rows": sum(m["original_rows"] for m in months),
                "prediction_eligible_rows": sum(m["prediction_eligible_rows"] for m in months),
                "monthly_episode_status": [{"outcome_month": m["outcome_month"], **m["episode_status_counts"]} for m in months],
                "target_coverage": aggregate_coverage(months),
                "largest_exact_equal_value_group": max((row["equal_value_group_size"] for row in ties), default=0),
                "legacy_capital_gate_would_trigger_episodes": sum(m["legacy_capital_gate_would_trigger_episodes"] for m in months),
                "negative_reference_nav_retained_marks": sum(m["negative_reference_nav_retained_marks"] for m in months),
                "annual_boundary_request_metadata": {str(a["year"]): a.get("boundary_request_audit") for a in plan["annuals"]},
                "class_grid_cells": len(grid), "empty_class_cells": sum(r["members"] == 0 for r in grid),
                "payoff_readiness_rows": len(readiness), "complete_120_month_vectors": sum(r["status"] == "available" for r in readiness),
                "supplemental_overlay_months": [m["outcome_month"] for m in months if m["supplemental_execution_overlay"] is not None],
                "training_history_path_targets_ready": True, "training_history_readiness_means_audit_complete_not_certified_label_coverage": True,
                "training_availability": training, "fitting_authorized": False,
                "models_fitted": False, "portfolio_backtest_run": False, "financial_2024_plus_read": False,
                "endpoint_target_control_rebuilt": False, "execution_overlay_learning_admitted": False,
                "active_wall_seconds": elapsed, "peak_rss_bytes": peak, "read_plan": plan["read_plan"],
                "reserved_materialization_compressed_bytes_including_retries": sum(read_json(p)["projected_compressed_bytes"] for p in read_directory.glob("*.json")),
                "year_audit_pins": {str(year): relative_pin(output / "years" / str(year) / "audit.json", output) for year in YEARS},
                "artifacts": {name: relative_pin(output / name, output) for name in
                    ("payoff_readiness.parquet", "class_grid.parquet", "tie_histogram.parquet", "training_completion.parquet")}}
            write_json(output / "audit.json", summary)
        result = read_full_path_audit(output) if through_year == 2023 else {"status": "PAUSED_AT_ANNUAL_CHECKPOINT", "through_year": through_year}
        check()
        write_json(run_directory / f"run-{number:04d}.json", {"status": "complete" if through_year == 2023 else "paused_at_year_checkpoint",
            "active_wall_seconds": time.monotonic() - started, "through_year": through_year})
        return result
    except BaseException as error:
        write_json(run_directory / f"run-{number:04d}.json", {"status": "failed", "exception": type(error).__name__,
            "message": str(error), "active_wall_seconds": time.monotonic() - started})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "run", "resume", "report"))
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--through-year", type=int, default=2023)
    args = parser.parse_args(argv)
    if args.action == "plan":
        plan = plan_full_path_audit(args.project, archive_directory=args.archive)
        write_json(args.output, plan)
        print(json.dumps(plan["read_plan"], indent=2))
    elif args.action == "report":
        print(json.dumps(read_full_path_audit(args.output), indent=2))
    else:
        need(args.plan is not None, "Explicit reviewed plan required")
        print(json.dumps(run(read_json(args.plan), args.output, resume=args.action == "resume", through_year=args.through_year), indent=2))


if __name__ == "__main__":
    main()
