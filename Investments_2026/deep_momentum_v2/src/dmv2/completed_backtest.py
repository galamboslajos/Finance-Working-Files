"""Complete the independently repaired backtest with the verified VVC entitlement.

The existing accounting engine, annual caches, forecasts and selections remain
unchanged. This additive phase appends the verified VVC and TSS retirement
packages as actions ten/eleven, preserving the preceding 56-complete/eight-withheld run.
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import version
import json
from pathlib import Path
import platform
import time

import pandas as pd
import pyarrow.parquet as pq

from .action_accounting import run_action_accounting
from .backtest_accounting import DailyAccountingPolicy
from .backtest_loading import (ARMS, PREDICTION_METADATA, _hash, _independent_clock,
    _json, _pin, _projected_bytes, _validate_prediction_metadata, _verify_pin, _within)
from .backtest_selection import PROBABILITIES, SAFE_COLUMNS, select_monthly_basket
from .balanced_baskets import (BALANCED_BREADTH_POLICY, select_balanced_monthly_basket,
    validate_balanced_monthly_basket)
from .catalog import canonical_sha256
from .daily_backtest import _save_table, _write_json, summarize_daily, _rss
from .fallback_backtest import _annual_prices, _configuration as _original_configuration, _reviewed_events
from .sni_action import append_sni_action
from .vvc_action import append_vvc_action
from .tss_action import append_tss_action
from .independent_backtest import _configuration as _parent_configuration
from .universe import DataContractError

PHASE = "independent_quote_action_completion_v1"
EXCLUDED_NATIVE_IDS = ("sec_gjP4xM", "sec_zmYNnz")
REPORTS = ("comparison", "selection_coverage", "ret_readiness", "failures",
    "action_coverage", "order_coverage", "selection_validation")
WORKERS = ("completed_backtest.py", "independent_backtest.py", "balanced_baskets.py", "sni_action.py", "vvc_action.py", "tss_action.py",
    "validation_reprediction.py", "fallback_backtest.py", "backtest_selection.py",
    "action_accounting.py", "backtest_accounting.py", "action_loading.py",
    "daily_backtest.py", "backtest_loading.py", "ret.py", "catalog.py",
    "universe.py", "exchange_sessions.py", "identity_policy.py", "full_panel.py")


def _configuration(configuration):
    if not isinstance(configuration, dict):
        raise DataContractError("Independent quote backtest requires its explicit frozen configuration")
    restriction = configuration.get("execution_restrictions", {})
    backtest = configuration.get("backtest", {})
    execution = configuration.get("execution", {})
    if (configuration.get("protocol_id") != "dmv2-independent-quote-action-completion-2018-2019-v1"
            or configuration.get("research_status") != "DATA_LIMITED_RESTRICTED_UNIVERSE"
            or tuple(sorted(restriction.get("native_ids", {}))) != EXCLUDED_NATIVE_IDS
            or restriction.get("valid_historical_training_rows_retained") is not True
            or restriction.get("prediction_rows_and_probabilities_retained") is not True
            or restriction.get("period") != "all_2018_2019_signals_before_selection"
            or execution.get("maximum_new_model_fits") != 0
            or execution.get("maximum_new_preprocessor_fits") != 0
            or execution.get("financial_2020_plus_analysis") is not False
            or execution.get("cloud_mutations") is not False
            or backtest.get("start") != "2018-01-01" or backtest.get("end") != "2019-12-31"
            or backtest.get("arms") != list(ARMS)
            or backtest.get("baskets") != ["class_1_10", "class_2_9"]
            or backtest.get("cost_bps") != [0, 5, 10, 25]
            or backtest.get("top_k_per_side") != 10
            or backtest.get("long_weight") != .5 or backtest.get("short_weight") != -.5
            or backtest.get("variants") != {"unstopped": None, "asset_stop_10pct": .1}):
        raise DataContractError("Independent quote backtest changed its fixed period, restrictions, zero-fit rule or scenario grid")
    breadth = backtest.get("basket_breadth", {})
    if (breadth.get("policy") != "balanced_available_capped_10"
            or breadth.get("names_per_side") != "min(10,eligible_mode_long_count,eligible_mode_short_count)"
            or breadth.get("equal_side_name_counts") is not True
            or breadth.get("long_total_weight") != .5 or breadth.get("short_total_weight") != -.5
            or breadth.get("zero_candidate_side") != "withhold_full_path"
            or breadth.get("candidate_information") != "signal_time_forecasts_and_fixed_execution_restrictions_only"
            or breadth.get("rank_and_ties") != "unchanged_class_probability_descending_then_source_id_ascending"):
        raise DataContractError("Independent quote backtest requires the declared balanced contemporaneous breadth policy")
    for name, maximum in (("backtest_maximum_wall_seconds", 900),
            ("backtest_maximum_peak_rss_bytes", 2 * 1024**3),
            ("maximum_output_bytes", 6 * 1024**3)):
        value = execution.get(name)
        if type(value) is not int or not 0 < value <= maximum:
            raise DataContractError("Independent quote backtest requires a bounded explicit local ceiling: " + name)
    return configuration


def _completion_parent_configuration(root, cfg, main_identity):
    """Require every empirical rule except the two extra actions to remain frozen."""
    path = Path(root) / "configs/independent_backtest.json"
    parent_cfg = _parent_configuration(_json(path))
    parent_pin = canonical_sha256(parent_cfg)
    if (cfg.get("parent_configuration_sha256") != parent_pin
            or main_identity.get("configuration") != parent_cfg
            or main_identity.get("configuration_sha256") != parent_pin):
        raise DataContractError("Completion must reuse the exact prior independent configuration and feature identity")
    mutable_metadata = {"protocol_id", "authority", "status", "backtest"}
    for name in set(parent_cfg) - mutable_metadata:
        if cfg.get(name) != parent_cfg[name]:
            raise DataContractError("Completion changed a frozen prior research rule: " + name)
    before = {name: value for name, value in parent_cfg["backtest"].items() if name != "events"}
    after = {name: value for name, value in cfg["backtest"].items() if name != "events"}
    if before != after:
        raise DataContractError("Completion may add VVC/TSS entitlements only; baskets, costs and all other rules stay frozen")
    return parent_cfg, path


def _prior_backtest_pins(root, run, identity):
    """Hash the original 56/8 run without decoding any performance table."""
    directory = Path(run) / "backtest"
    audit = _json(directory / "audit.json")
    phase = _json(directory / "build_identity.json")
    if (audit.get("status") != "COMPLETE" or audit.get("phase") != "independent_quote_backtest_v1"
            or audit.get("build_identity_sha256") != identity or audit.get("scenarios") != 64
            or audit.get("complete_proxy_paths") != 56 or audit.get("failed_closed_paths") != 8
            or canonical_sha256(phase) != audit.get("phase_identity_sha256")
            or phase.get("parent_identity_sha256") != identity or len(phase.get("events", [])) != 9):
        raise DataContractError("Completion requires the immutable preceding 56-complete/eight-withheld independent backtest")
    paths = [directory / "audit.json", directory / "build_identity.json"]
    paths.extend(_verify_pin(directory, item) for item in audit["artifacts"].values())
    return [_pin(root, path) for path in dict.fromkeys(paths)]


def _predictions(models, identity, windows, pins, root, *, read_stats=None):
    from .validation_reprediction import read_validation_reprediction
    saved = read_validation_reprediction(models,identity_sha256=identity,root=root)
    audit = saved["audit"]
    pins.extend([_pin(root,models/"audit.json"),_pin(root,models/"build_identity.json")])
    paths, metadata, key_order, common_cohort = {}, {}, None, None
    projected=0
    for arm in ARMS:
        arm_audit_path = _verify_pin(models,audit["arms"][arm])
        arm_audit = _json(arm_audit_path)
        if arm_audit.get("build_identity_sha256") != identity:
            raise DataContractError("New model arm belongs to a different parent reconstruction")
        path = _verify_pin(arm_audit_path.parent,arm_audit["artifacts"]["predictions"])
        pins.extend([_pin(root,arm_audit_path),_pin(root,path)])
        fields = set(pq.ParquetFile(path).schema_arrow.names)
        columns = [field for field in PREDICTION_METADATA if field in fields]
        meta = pq.ParquetFile(path).read(columns=columns)
        projected+=_projected_bytes(path,columns)
        keys = _validate_prediction_metadata(meta,windows)
        cohort=meta.select(["source_security_id","signal_date","intrinio_security_id",
            "identity_eligible","requested","paper_prediction_eligible","prediction_eligible"])
        if key_order is None:
            key_order = keys
        elif keys != key_order:
            raise DataContractError("New model arms have different original validation key order")
        if common_cohort is None:
            common_cohort=cohort
        elif not cohort.equals(common_cohort):
            raise DataContractError("New model arms have different signal-time prediction cohorts")
        if len(meta)>25000:
            raise DataContractError("New validation prediction ledger exceeds fixed row bound")
        metadata[arm],paths[arm] = meta,path
        financial_columns=list(dict.fromkeys([*[name for name in SAFE_COLUMNS if name in meta.column_names],*PROBABILITIES]))
        projected+=_projected_bytes(path,financial_columns)
    if projected>128*1024**2:
        raise DataContractError("Prediction projections exceed the explicit backtest read budget")
    if read_stats is not None:
        read_stats.update(candidate_rows=sum(len(table) for table in metadata.values()),
            projected_compressed_bytes=projected,metadata_and_probability_projections_counted=True,
            future_labels_decoded=False)
    # All timing, cohort and hash guards passed before any probability decode.
    result = {}
    for arm,path in paths.items():
        columns = list(dict.fromkeys([*[name for name in SAFE_COLUMNS if name in metadata[arm].column_names],*PROBABILITIES]))
        result[arm] = pq.ParquetFile(path).read(columns=columns).to_pandas()
    return result


@dataclass(frozen=True)
class SavedCompletedBacktest:
    path: Path
    audit: dict
    comparison: pd.DataFrame
    selection_coverage: pd.DataFrame
    ret_readiness: pd.DataFrame
    failures: pd.DataFrame
    action_coverage: pd.DataFrame
    order_coverage: pd.DataFrame
    selection_validation: pd.DataFrame


def read_completed_backtest(directory, *, identity_sha256=None, phase_identity_sha256=None, root=None):
    """Verify saved small reports, never invoke predictions or accounting."""
    directory = Path(directory).resolve()
    audit = _json(directory / "audit.json")
    if (audit.get("status") != "COMPLETE" or audit.get("phase") != PHASE
            or audit.get("scenarios") != 64
            or audit.get("research_status") != "DATA_LIMITED_RESTRICTED_UNIVERSE"
            or audit.get("models_fitted") is not False
            or audit.get("financial_2020_plus_read") is not False
            or identity_sha256 is not None and audit.get("build_identity_sha256") != identity_sha256
            or phase_identity_sha256 is not None and audit.get("phase_identity_sha256") != phase_identity_sha256):
        raise DataContractError("Saved independent quote backtest is stale or incomplete")
    required = {"build_identity.json", *[name + ".parquet" for name in REPORTS]}
    if not required <= set(audit.get("artifacts", {})):
        raise DataContractError("Saved independent quote backtest omits pinned reports")
    for item in audit["artifacts"].values():
        _verify_pin(directory, item)
    phase = _json(directory / "build_identity.json")
    if (canonical_sha256(phase) != audit.get("phase_identity_sha256")
            or phase.get("phase") != PHASE
            or phase.get("parent_identity_sha256") != audit.get("build_identity_sha256")
            or phase.get("models_fitted") is not False
            or len(phase.get("events", [])) != 11):
        raise DataContractError("Saved independent quote backtest phase identity changed")
    if root is not None:
        root = Path(root).resolve()
        _within(root, directory)
        for pin in phase["source_pins"]:
            _verify_pin(root, pin)
    reports = {name: pd.read_parquet(directory / (name + ".parquet")) for name in REPORTS}
    comparison = reports["comparison"]
    if (len(comparison) != 64 or comparison.duplicated(["arm", "basket", "variant", "cost_bps"]).any()
            or set(comparison.arm) != set(ARMS) or set(comparison.basket) != {"class_1_10", "class_2_9"}
            or set(comparison.variant) != {"unstopped", "asset_stop_10pct"}
            or set(comparison.cost_bps) != {0, 5, 10, 25}
            or len(reports["selection_validation"]) != 8):
        raise DataContractError("Saved independent backtest lost its complete grid or eight basket validation receipts")
    return SavedCompletedBacktest(directory, audit, **reports)


def run_completed_backtest(root, run_directory, configuration, identity_sha256, *,
        quote_amendments, sni_evidence_directory, vvc_evidence_directory, tss_evidence_directory):
    """Explicit 64-scenario evaluation with reused fits and independent repairs."""
    root = Path(root).resolve()
    run = _within(root, Path(run_directory).resolve())
    cfg = _configuration(configuration)
    started = time.monotonic()
    quote_amendments.verify(root)
    main_identity = _json(run / "build_identity.json")
    parent_cfg, parent_cfg_path = _completion_parent_configuration(root, cfg, main_identity)
    if (canonical_sha256(main_identity) != identity_sha256
            or main_identity.get("quote_amendments_fingerprint") != quote_amendments.fingerprint):
        raise DataContractError("Independent backtest differs from the frozen parent build or amended quote bundle")
    sessions, windows, clock = _independent_clock()
    if len(sessions) != 503 or sessions[0] != "2018-01-02" or sessions[-1] != "2019-12-31":
        raise DataContractError("Independent quote backtest requires all 503 authorized 2018--2019 sessions")
    pins = [_pin(root, run / "build_identity.json")]
    original_path = root / "configs/data_limited_fallback.json"
    original_cfg = _original_configuration(_json(original_path))
    pins.append(_pin(root, original_path))
    pins.append(_pin(root, parent_cfg_path))
    pins.extend(_prior_backtest_pins(root, run, identity_sha256))
    new_config_path = root / "configs/independent_backtest_completion.json"
    if _json(new_config_path) != cfg:
        raise DataContractError("Independent backtest caller differs from the saved frozen configuration")
    pins.append(_pin(root, new_config_path))
    peak, prediction_read = _rss(), {}
    predictions = _predictions(run / "models", identity_sha256, windows, pins, root,
        read_stats=prediction_read)
    prices, read_plan = _annual_prices(root, run, identity_sha256, quote_amendments.fingerprint, pins,
        max_projected_bytes=128 * 1024**2 - prediction_read.get("projected_compressed_bytes", 0),
        max_rows=1000000 - prediction_read.get("candidate_rows", 0))
    read_plan = {**read_plan, "prediction_projections": prediction_read,
        "total_projected_compressed_bytes": read_plan.get("projected_compressed_bytes", 0)
            + prediction_read.get("projected_compressed_bytes", 0),
        "total_candidate_rows": read_plan.get("candidate_rows", 0) + prediction_read.get("candidate_rows", 0),
        "quote_amendments_loaded_from_precompiled_bundle": True}
    if not prices.trading_date.isin(sessions).all():
        raise DataContractError("Independent execution cache includes non-XNYS or out-of-period sessions")
    old_events, event_path = _reviewed_events(root, sessions)
    pins.append(_pin(root, event_path))
    evidence = _within(root, Path(sni_evidence_directory).resolve())
    event_records = append_sni_action(old_events, evidence, prices=prices)
    if len(event_records) != 9 or event_records[:8] != old_events:
        raise DataContractError("Independent repair must preserve all original eight actions and append exactly SNI")
    for filename in ("sni_action_evidence.json", "successor_identity.json", "predecessor_successor_controls.parquet"):
        pins.append(_pin(root, evidence / filename))
    vvc_evidence = _within(root, Path(vvc_evidence_directory).resolve())
    old_nine_events = event_records
    event_records = append_vvc_action(old_nine_events, vvc_evidence, prices=prices)
    if len(event_records) != 10 or event_records[:9] != old_nine_events:
        raise DataContractError("Completion must preserve all original nine actions and append exactly VVC")
    for filename in ("vvc_action_evidence.json", "predecessor_controls.parquet"):
        pins.append(_pin(root, vvc_evidence / filename))
    tss_evidence = _within(root, Path(tss_evidence_directory).resolve())
    old_ten_events = event_records
    event_records = append_tss_action(old_ten_events, tss_evidence, prices=prices)
    if len(event_records) != 11 or event_records[:10] != old_ten_events:
        raise DataContractError("Completion must preserve all original ten actions and append exactly TSS")
    for filename in ("tss_action_evidence.json", "successor_identity.json", "predecessor_successor_controls.parquet"):
        pins.append(_pin(root, tss_evidence / filename))
    events = pd.DataFrame(event_records)
    phase_identity = {"schema_version": 1, "phase": PHASE,
        "parent_identity_sha256": identity_sha256, "configuration_sha256": canonical_sha256(cfg),
        "parent_configuration_sha256": canonical_sha256(parent_cfg),
        "quote_amendments_fingerprint": quote_amendments.fingerprint, "source_pins": pins,
        "models_fitted": False, "preprocessing_fitted": False,
        "basket_policy": BALANCED_BREADTH_POLICY,
        "clock": clock, "read_plan": read_plan, "events": event_records,
        "workers": {name: _hash(Path(__file__).parent / name) for name in WORKERS},
        "runtime": {"python": platform.python_version(), **{name: version(name)
            for name in ("numpy", "pandas", "pyarrow", "exchange-calendars")}}}
    phase_hash = canonical_sha256(phase_identity)
    directory = run / "backtest_completed"
    if directory.exists():
        return read_completed_backtest(directory, identity_sha256=identity_sha256,
            phase_identity_sha256=phase_hash, root=root).path
    directory.mkdir(parents=True)
    artifacts = {}
    _write_json(directory / "build_identity.json", phase_identity)
    artifacts["build_identity.json"] = _pin(directory, directory / "build_identity.json")
    _write_json(directory / "audit.json", {"status": "STARTED", "phase": PHASE,
        "build_identity_sha256": identity_sha256, "phase_identity_sha256": phase_hash})
    comparison, failures, coverage, ret_ready = [], [], [], []
    action_coverage, order_coverage, selection_validation = [], [], []

    def check_resources():
        nonlocal peak
        peak = max(peak, _rss())
        limits = cfg["execution"]
        if (time.monotonic() - started > limits["backtest_maximum_wall_seconds"]
                or peak > limits["backtest_maximum_peak_rss_bytes"]
                or sum(path.stat().st_size for path in run.rglob("*") if path.is_file()) > limits["maximum_output_bytes"]):
            raise DataContractError("Independent quote backtest exceeds its explicit local resource ceiling")
    try:
        for arm in ARMS:
            prediction = predictions[arm]
            ret = select_monthly_basket(prediction, basket="ret", sessions=clock["selection_sessions"],
                top_k=10, payoffs=None)
            ret_ready.append(ret.monthly.assign(arm=arm))
            if not ret.selections.empty or not ret.monthly.status.eq("withheld_incomplete_120_month_class_grid").all():
                raise DataContractError("Independent backtest must preserve the unavailable strict RET gate")
            for basket in cfg["backtest"]["baskets"]:
                selection = select_balanced_monthly_basket(prediction, basket=basket,
                    sessions=clock["selection_sessions"], configuration=original_cfg)
                validated = validate_balanced_monthly_basket(selection, prediction, basket=basket,
                    sessions=clock["selection_sessions"], configuration=original_cfg)
                selection_validation.append({"arm": arm, "basket": basket, **validated})
                coverage.append(selection.monthly.assign(arm=arm))
                for name, frame in (("original_ledger", selection.ledger), ("selected", selection.selections),
                        ("rebalances", selection.rebalances)):
                    relative_selection = f"selections/{arm}/{basket}/{name}.parquet"
                    _save_table(directory, relative_selection, frame, artifacts)
                    if _hash(directory / relative_selection) != _hash(run / "backtest" / relative_selection):
                        raise DataContractError("Corporate-action completion changed a frozen prior selection artifact")
                for variant, stop in cfg["backtest"]["variants"].items():
                    for fee in cfg["backtest"]["cost_bps"]:
                        check_resources()
                        result = run_action_accounting(selection.selections, prices, sessions,
                            rebalances=selection.rebalances, terminal_date="2019-12-31", events=events,
                            policy=DailyAccountingPolicy(cost_bps=fee, borrow_annual_bps=0,
                                required_names_per_side=None, asset_stop_fraction=stop))
                        relative = f"scenarios/{arm}/{basket}/{variant}/{fee}bps"
                        for name, frame in result.tables.items():
                            _save_table(directory, relative + "/" + name + ".parquet", frame, artifacts)
                        summary, monthly = summarize_daily(result, arm=arm, basket=basket, variant=variant,
                            cost_bps=fee, expected_sessions=len(sessions))
                        summary.update(research_status="DATA_LIMITED_RESTRICTED_UNIVERSE",
                            retrospective_execution_restriction=True, full_historical_universe_performance_claim=False,
                            all_original_prediction_rows_retained=True, valid_DF_RDC_training_history_retained=True,
                            basket_policy=BALANCED_BREADTH_POLICY, required_names_per_side=None,
                            action_count=result.summary["action_count"],
                            unfilled_retired_order_count=result.summary["unfilled_retired_order_count"])
                        if result.status == "complete_proxy":
                            summary["trading_cost_paid"] = float(result.market_trades.cost.sum())
                        comparison.append(summary)
                        _save_table(directory, relative + "/monthly.parquet", monthly, artifacts)
                        _write_json(directory / relative / "summary.json", {**result.summary,
                            "research_status": "DATA_LIMITED_RESTRICTED_UNIVERSE", "basket_policy": BALANCED_BREADTH_POLICY})
                        artifacts[relative + "/summary.json"] = _pin(directory, directory / relative / "summary.json")
                        if not result.failures.empty:
                            failures.append(result.failures.assign(arm=arm, basket=basket, variant=variant, cost_bps=fee))
                        if not result.action_events.empty:
                            action_coverage.extend(dict(arm=arm, basket=basket, variant=variant, cost_bps=fee,
                                event_id=event, side=side, applied_lot_count=int(count)) for (event, side), count
                                in result.action_events.groupby(["event_id", "side"], dropna=False).size().items())
                        if not result.orders.empty:
                            order_coverage.extend(dict(arm=arm, basket=basket, variant=variant, cost_bps=fee,
                                status=status, original_order_count=int(count)) for status, count
                                in result.orders.groupby("status", dropna=False).size().items())
                        print(json.dumps({"phase": PHASE, "arm": arm, "basket": basket,
                            "variant": variant, "cost_bps": fee, "status": result.status,
                            "first_failure_date": summary["first_failure_date"]}), flush=True)
        reports = {"comparison": pd.DataFrame(comparison),
            "selection_coverage": pd.concat(coverage, ignore_index=True),
            "ret_readiness": pd.concat(ret_ready, ignore_index=True),
            "selection_validation": pd.DataFrame(selection_validation),
            "failures": pd.concat(failures, ignore_index=True) if failures else pd.DataFrame(columns=[
                "date", "mark", "reason", "source_security_id", "intrinio_security_id", "price_field", "role",
                "arm", "basket", "variant", "cost_bps"]),
            "action_coverage": pd.DataFrame(action_coverage, columns=["arm", "basket", "variant", "cost_bps",
                "event_id", "side", "applied_lot_count"]),
            "order_coverage": pd.DataFrame(order_coverage, columns=["arm", "basket", "variant", "cost_bps",
                "status", "original_order_count"])}
        for name, frame in reports.items():
            _save_table(directory, name + ".parquet", frame, artifacts)
        for pin in pins:
            _verify_pin(root, pin)
        quote_amendments.verify(root)
        check_resources()
        audit = {"status": "COMPLETE", "phase": PHASE, "research_status": "DATA_LIMITED_RESTRICTED_UNIVERSE",
            "build_identity_sha256": identity_sha256, "phase_identity_sha256": phase_hash, "artifacts": artifacts,
            "scenarios": len(comparison), "complete_proxy_paths": sum(row["complete_proxy_path"] for row in comparison),
            "failed_closed_paths": sum(not row["complete_proxy_path"] for row in comparison),
            "execution_seconds": time.monotonic() - started, "peak_rss_bytes": peak,
            "read_plan": read_plan, "source_pins_unchanged": True, "models_fitted": False,
            "financial_2020_plus_read": False, "original_pointers_changed": False,
            "RET": "withheld_all_24_signals_per_arm", "execution_excluded_native_ids": list(EXCLUDED_NATIVE_IDS),
            "excluded_instruments_retained_in_training_and_predictions": True,
            "retrospective_data_quality_restriction": True, "historical_universe_performance_claim": False,
            "selection_and_survivorship_bias_removed": False, "knowledge_time_certified": False,
            "complete_economic_accounting_certified": False, "basket_policy": BALANCED_BREADTH_POLICY,
            "monthly_selection_validator_applied": True, "independent_sni_action_added": True,
            "independent_vvc_action_added": True, "independent_tss_action_added": True,
            "prior_56_complete_eight_withheld_backtest_bytes_preserved": True,
            "required_names_per_side": None}
        _write_json(directory / "audit.json", audit)
    except Exception as error:
        _write_json(directory / "failure.json", {"exception": type(error).__name__, "message": str(error)})
        raise
    return directory
