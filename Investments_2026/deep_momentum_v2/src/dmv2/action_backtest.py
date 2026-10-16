"""Explicit additive corporate-action rerun; saved reports never run accounting.

The eight original selected/rebalance artifacts are copied byte for byte. This
module never selects stocks, fits models, or opens financial values after 2019.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from importlib.metadata import version
import json
from pathlib import Path
import platform
import shutil
import time

import pandas as pd
import pyarrow.parquet as pq

from .action_accounting import run_action_accounting
from .action_loading import plan_action_inputs, load_action_inputs, _verify as verify_action_inputs
from .alpaca_repairs import prepare_published_alpaca_repair_declarations
from .backtest_accounting import DailyAccountingPolicy
from .backtest_loading import ARMS, _hash, _json, _pin, _verify_pin, _within
from .catalog import canonical_sha256
from .daily_backtest import (configuration as original_configuration, read_daily_backtest,
    summarize_daily, _write_json, _save_table, _rss, _check_resources)
from .ohlc_repairs import OhlcRepairPolicy, apply_reviewed_ohlc_repairs, ORIGINAL_FIELDS
from .universe import DataContractError
from .supplemental_bars import admit_reviewed_missing_bars

CONFIGURATION_SHA256 = "sha256:9df913c65520a571205ad17cd60cd1d9a36aa5c97047ff257b232984aca9b20b"
WORKERS = ("action_backtest.py", "action_accounting.py", "action_loading.py", "alpaca_repairs.py", "ohlc_repairs.py", "supplemental_bars.py")
REPORTS = ("comparison", "selection_coverage", "ret_readiness", "failures", "action_coverage", "order_coverage", "ohlc_repairs", "quote_source_review", "supplemental_bars")


def _root(root):
    return Path(root if isinstance(root, (str, Path)) else root.root).resolve()


def configuration(root):
    cfg = _json(_root(root) / "configs/action_backtest.json")
    if canonical_sha256(cfg) != CONFIGURATION_SHA256:
        raise DataContractError("Frozen action backtest configuration changed; define a separate protocol")
    if (cfg["arms"] != list(ARMS) or cfg["baskets"] != ["class_1_10", "class_2_9"]
            or cfg["start"] != "2018-01-01" or cfg["end"] != "2019-12-31"
            or cfg["trading_cost_bps"] != [0, 5, 10, 25]
            or cfg["required_names_per_side"] != 10
            or cfg["long_weight"] != .5 or cfg["short_weight"] != -.5):
        raise DataContractError("Action rerun must retain its original bounded 64-scenario recipe")
    return cfg


def _worker_hashes():
    return {name: _hash(Path(__file__).parent / name) for name in WORKERS}


def _reviewed_manifest(root, evidence_dir, name="reviewed_ohlc_inputs.json"):
    path = evidence_dir / name
    if not path.is_file():
        raise DataContractError("Reviewed actual OHLC source inputs are missing; the original three bars are not fixed")
    if path.stat().st_size > 128 * 1024:
        raise DataContractError("Reviewed OHLC manifest exceeds its 128KiB bound")
    manifest = _json(path)
    if (set(manifest) != {"schema", "projection_evidence", "source_provenance", "native_mapping", "review_notes"}
            or manifest["schema"] != "dmv2-reviewed-published-ohlc/v1"):
        raise DataContractError("Closed reviewed published OHLC inputs required")
    pins = [_pin(root, path)]
    for item in (manifest["projection_evidence"], manifest["source_provenance"].get("range_receipt", {})):
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise DataContractError("Reviewed OHLC projection and range-receipt pins required")
        candidate = _within(evidence_dir, evidence_dir / item["path"])
        pins.append(_pin(root, candidate))
    return manifest, pins


def _original_inputs(root, cfg):
    """Verify saved source, then inspect only selection date/identity metadata."""
    original = read_daily_backtest(root)
    if canonical_sha256(original_configuration(root)) != cfg["original_protocol_sha256"]:
        raise DataContractError("Original daily backtest recipe changed")
    pins = [_pin(root, root / "outputs/daily_backtest/LATEST.json"),
            _pin(root, original.path / "audit.json"), _pin(root, original.path / "build_identity.json")]
    paths, selected_hashes, rebalance_hashes = {}, {}, {}
    for arm in ARMS:
        for basket in cfg["baskets"]:
            key = f"{arm}/{basket}"
            prefix = f"selections/{key}"
            for name in ("original_ledger", "selected", "rebalances"):
                relative = f"{prefix}/{name}.parquet"
                path = _verify_pin(original.path, original.audit["artifacts"][relative])
                pins.append(_pin(root, path))
                paths[relative] = path
                if name == "selected":
                    selected_hashes[key] = _hash(path)
                if name == "rebalances":
                    rebalance_hashes[key] = _hash(path)
            selected = pq.ParquetFile(paths[prefix + "/selected.parquet"]).read(columns=[
                "signal_date", "execution_date", "source_security_id", "intrinio_security_id", "side"]).to_pandas()
            _check_selection_metadata(selected)
    if len(selected_hashes) != 8 or len(rebalance_hashes) != 8:
        raise DataContractError("The original eight selected/rebalance artifacts must all be retained")
    return original, paths, pins, selected_hashes, rebalance_hashes


def _check_selection_metadata(selected):
    required = {"signal_date", "execution_date", "source_security_id", "intrinio_security_id", "side"}
    if not required <= set(selected) or selected[list(required)].isna().any().any():
        raise DataContractError("Original selection metadata is incomplete")
    execution = pd.to_datetime(selected.execution_date, errors="raise")
    signal = pd.to_datetime(selected.signal_date, errors="raise")
    if (not execution.between("2018-01-01", "2019-12-31").all()
            or not signal.between("2017-12-01", "2019-11-30").all()
            or (signal >= execution).any()
            or (signal.dt.to_period("M") != execution.dt.to_period("M") - 1).any()):
        raise DataContractError("Original selection dates leave the authorized development chronology")
    if selected.duplicated(["execution_date", "source_security_id"]).any():
        raise DataContractError("Duplicate original selection identity")
    counts = selected.groupby(["execution_date", "side"]).size()
    if (not counts.eq(10).all() or set(selected.side) != {"long", "short"}
            or not selected.groupby("execution_date").side.nunique().eq(2).all()):
        raise DataContractError("Frozen selections must retain ten original names per direction")


def _all_pins(root, *groups):
    pins = {}
    for group in groups:
        for pin in group:
            _verify_pin(root, pin)
            previous = pins.setdefault(pin["path"], pin)
            if previous != pin:
                raise DataContractError("Conflicting source pins")
    return [pins[key] for key in sorted(pins)]


def plan_action_backtest(root, *, evidence_dir=None):
    """Metadata, file hashes and finite read plan only; no price preparation."""
    root = _root(root)
    cfg = configuration(root)
    evidence_dir = Path(evidence_dir or root / "data/derived/daily_action_inputs").resolve()
    _within(root, evidence_dir)
    original, paths, original_pins, selected_hashes, rebalance_hashes = _original_inputs(root, cfg)
    manifest, repair_pins = _reviewed_manifest(root, evidence_dir)
    additional_manifests = []
    extra_name = cfg["additional_reviewed_sources"]["optional_manifest"]
    if (evidence_dir / extra_name).exists():
        extra, extra_pins = _reviewed_manifest(root, evidence_dir, extra_name)
        scope = cfg["additional_reviewed_sources"]
        if (not set(extra["native_mapping"]) <= set(scope["allowed_source_ids"])
                or set(extra["native_mapping"].values()) != {scope["allowed_native_id"]}):
            raise DataContractError("Additional reviewed source is limited to the original PCG native/source identity")
        additional_manifests.append(extra)
        repair_pins.extend(extra_pins)
    plan = plan_action_inputs(root, evidence_dir=evidence_dir,
        max_rows=cfg["execution"]["maximum_candidate_rows"],
        max_projected_bytes=cfg["execution"]["maximum_projected_read_bytes"],
        unquoted_successor_policy=cfg["corporate_actions"]["unquoted_successor_policy"])
    action_pins = [_pin(root, _verify_pin(evidence_dir, pin)) for pin in plan.evidence_pins]
    pins = _all_pins(root, original_pins, repair_pins, action_pins, plan.base_plan.source_pins)
    identity = {"schema_version": 1, "configuration": cfg, "workers": _worker_hashes(),
        "source_pins": pins, "action_input_fingerprint": plan.fingerprint,
        "original_run": original.path.relative_to(root).as_posix(),
        "original_selected_sha256": selected_hashes, "original_rebalances_sha256": rebalance_hashes,
        "read_plan": plan.read_plan,
        "runtime": {"python": platform.python_version(), **{name: version(name) for name in ("pandas", "numpy", "pyarrow")}}}
    return {"root": root, "configuration": cfg, "evidence_dir": evidence_dir, "original": original,
        "original_paths": paths, "reviewed_manifest": manifest, "additional_reviewed_manifests": additional_manifests,
        "additional_manifests": additional_manifests,
        "input_plan": plan, "identity": identity,
        "build_identity_sha256": canonical_sha256(identity)}


def _prepare_prices(bundle, manifest, cfg, evidence_dir, *, additional_manifests=()):
    all_prepared = []
    for source in (manifest, *additional_manifests):
        all_prepared.append(prepare_published_alpaca_repair_declarations(bundle.daily_prices,
            evidence_root=evidence_dir, projection_evidence=source["projection_evidence"],
            source_provenance=source["source_provenance"], native_mapping=source["native_mapping"],
            review_notes=source["review_notes"]))
    declarations = [repair for prepared in all_prepared for repair in prepared.repairs]
    keys = [(repair["intrinio_security_id"], repair["trading_date"]) for repair in declarations]
    if len(set(keys)) != len(keys):
        raise DataContractError("Independent reviewed sources propose duplicate repairs; explicit source conflict review required")
    settings = cfg["ohlc_repairs"]
    policy = OhlcRepairPolicy(**{name: settings[name] for name in (
        "maximum_raw_close_absolute_difference", "maximum_raw_close_relative_difference",
        "maximum_original_adjusted_rounding_residual", "policy_id")})
    # Check each proposed overlay against its exact original row. A source
    # compatibility conflict on a nonmandatory bar retains that rejected bar
    # for per-path failure; integrity/identity/evidence errors remain fatal.
    original = bundle.daily_prices
    accepted, passing_declarations, rejection_by_key = [], [], {}
    compatibility_errors = {
        "Alternate close conflict needs separate source/unit review",
        "Original raw/adjusted pairs conflict with close-derived basis",
    }
    for repair in declarations:
        key = repair["intrinio_security_id"], repair["trading_date"]
        subset = original.loc[original.intrinio_security_id.eq(key[0]) & original.trading_date.eq(key[1])]
        try:
            checked = apply_reviewed_ohlc_repairs(subset, [repair], evidence_root=evidence_dir, policy=policy)
        except DataContractError as error:
            if str(error) not in compatibility_errors:
                raise
            rejection_by_key[key] = str(error)
            continue
        accepted.extend(checked.ledger.to_dict("records"))
        passing_declarations.append(repair)
    repair_ledger = pd.DataFrame(accepted)
    mandatory = {tuple(item) for item in settings["mandatory_original_keys"]}
    actual = set(zip(repair_ledger.get("intrinio_security_id", []), repair_ledger.get("trading_date", [])))
    if not mandatory <= actual:
        raise DataContractError("The three original OHLC issues are not all repaired from actual compatible source bars: "
                                + json.dumps(sorted(mandatory - actual)))
    # One final overlay onto the immutable full original frame. No stacking of
    # already-repaired rows is permitted by the shared repair worker.
    repaired = apply_reviewed_ohlc_repairs(original, passing_declarations, evidence_root=evidence_dir, policy=policy)
    prices, repair_ledger = repaired.prices, repaired.ledger
    review = []
    for record in (row for prepared in all_prepared for row in prepared.identity_and_control_ledger):
        row = dict(record)
        key = row.get("intrinio_security_id"), row.get("trading_date")
        row["overlay_applied"] = key in actual
        row["overlay_compatibility_rejection"] = rejection_by_key.get(key)
        review.append(row)
    return prices, repair_ledger, pd.DataFrame(review)



def _report_frame(records, columns):
    return pd.DataFrame(records) if records else pd.DataFrame(columns=columns)


@dataclass(frozen=True)
class SavedActionBacktest:
    path: Path
    audit: dict
    comparison: pd.DataFrame
    selection_coverage: pd.DataFrame
    ret_readiness: pd.DataFrame
    failures: pd.DataFrame
    action_coverage: pd.DataFrame
    order_coverage: pd.DataFrame
    ohlc_repairs: pd.DataFrame
    quote_source_review: pd.DataFrame
    supplemental_bars: pd.DataFrame


def read_action_backtest(root):
    """Verify hashes and small reports; do not plan, load prices or run engines."""
    root = _root(root)
    parent = root / "outputs/action_backtest"
    pointer_path = parent / "LATEST.json"
    if not pointer_path.exists():
        raise FileNotFoundError("No saved corporate-action development backtest")
    pointer = _json(pointer_path)
    directory = _within(parent, parent / pointer.get("path", ""))
    audit_path = directory / "audit.json"
    if not audit_path.is_file() or _hash(audit_path) != pointer.get("audit_sha256"):
        raise DataContractError("Saved action backtest pointer/audit changed")
    audit = _json(audit_path)
    if audit.get("status") != "COMPLETE" or audit.get("build_identity_sha256") != pointer.get("build_identity_sha256"):
        raise DataContractError("Action backtest execution incomplete or identity changed")
    identity = _json(_verify_pin(directory, audit["artifacts"]["build_identity.json"]))
    if (canonical_sha256(identity) != pointer["build_identity_sha256"]
            or identity["configuration"] != configuration(root) or identity["workers"] != _worker_hashes()):
        raise DataContractError("Action backtest recipe or worker identity changed")
    for pin in identity["source_pins"]:
        _verify_pin(root, pin)
    for pin in audit["artifacts"].values():
        _verify_pin(directory, pin)
    reports = {name: pd.read_parquet(directory / (name + ".parquet")) for name in REPORTS}
    comparison = reports["comparison"]
    if len(comparison) != 64 or len(reports["ret_readiness"]) != 24:
        raise DataContractError("Action report lost scenario/RET coverage")
    expected_keys = {(arm,basket,variant,fee) for arm in ARMS for basket in ("class_1_10","class_2_9")
                     for variant in ("unstopped","asset_stop_10pct") for fee in (0,5,10,25)}
    actual_keys = set(comparison[["arm","basket","variant","cost_bps"]].itertuples(index=False,name=None))
    if actual_keys != expected_keys:
        raise DataContractError("Action comparison lost a fixed scenario")
    for key, expected in identity["original_selected_sha256"].items():
        if _hash(directory / f"selections/{key}/selected.parquet") != expected:
            raise DataContractError("Original selected bytes changed in action rerun")
    for key, expected in identity["original_rebalances_sha256"].items():
        if _hash(directory / f"selections/{key}/rebalances.parquet") != expected:
            raise DataContractError("Original rebalance bytes changed in action rerun")
    return SavedActionBacktest(directory, audit, **reports)


def run_action_backtest(root, *, evidence_dir=None):
    """Explicit fixed trial: 64 scenarios, immutable selections and source files."""
    started = time.monotonic()
    planned = plan_action_backtest(root, evidence_dir=evidence_dir)
    root, cfg = planned["root"], planned["configuration"]
    identity, identity_hash = planned["identity"], planned["build_identity_sha256"]
    directory = root / "outputs/action_backtest" / ("2018-2019-" + identity_hash.removeprefix("sha256:")[:16])
    if directory.exists():
        if (directory / "audit.json").exists() and _json(directory / "audit.json").get("status") == "COMPLETE":
            _write_json(directory.parent / "LATEST.json", {"path": directory.name,
                "build_identity_sha256": identity_hash, "audit_sha256": _hash(directory / "audit.json")})
            return read_action_backtest(root)
        raise DataContractError("An incomplete identical action run exists; inspect before retrying")
    # Refuse missing/incompatible actual repair evidence before publishing any
    # new run or touching either output pointer.
    bundle = load_action_inputs(root, plan=planned["input_plan"],
        unquoted_successor_policy=cfg["corporate_actions"]["unquoted_successor_policy"])
    prices, repair_ledger, quote_review = _prepare_prices(bundle, planned["reviewed_manifest"], cfg, planned["evidence_dir"],
        additional_manifests=planned.get("additional_reviewed_manifests", ()))
    manifest = planned["reviewed_manifest"]
    supplemental = admit_reviewed_missing_bars(prices, evidence_root=planned["evidence_dir"],
        projection_evidence=manifest["projection_evidence"], source_provenance=manifest["source_provenance"],
        native_mapping=manifest["native_mapping"], review_notes=manifest["review_notes"])
    prices = supplemental.prices
    directory.mkdir(parents=True)
    artifacts = {}
    _write_json(directory / "build_identity.json", identity)
    artifacts["build_identity.json"] = _pin(directory, directory / "build_identity.json")
    _write_json(directory / "audit.json", {"status": "STARTED", "build_identity_sha256": identity_hash})
    original_pointer = next(pin for pin in identity["source_pins"] if pin["path"] == "outputs/daily_backtest/LATEST.json")
    try:
        for relative, source in planned["original_paths"].items():
            target = directory / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            if _hash(target) != _hash(source):
                raise DataContractError("Original selection artifact byte copy failed")
            artifacts[relative] = _pin(directory, target)
        original = planned["original"]
        _save_table(directory, "selection_coverage.parquet", original.selection_coverage, artifacts)
        _save_table(directory, "ret_readiness.parquet", original.ret_readiness, artifacts)
        _save_table(directory, "ohlc_repairs.parquet", repair_ledger, artifacts)
        _save_table(directory, "quote_source_review.parquet", quote_review, artifacts)
        _save_table(directory, "supplemental_bars.parquet", supplemental.ledger, artifacts)
        comparison, failures, action_coverage, order_coverage = [], [], [], []
        for arm in ARMS:
            for basket in cfg["baskets"]:
                prefix = f"selections/{arm}/{basket}"
                selected = pd.read_parquet(directory / (prefix + "/selected.parquet"))
                rebalances = pd.read_parquet(directory / (prefix + "/rebalances.parquet"))
                for variant, settings in cfg["variants"].items():
                    for fee in cfg["trading_cost_bps"]:
                        _check_resources(cfg, started, directory)
                        result = run_action_accounting(selected, prices, bundle.sessions,
                            rebalances=rebalances, terminal_date="2019-12-31", events=bundle.events,
                            policy=DailyAccountingPolicy(cost_bps=fee, borrow_annual_bps=0,
                                required_names_per_side=10, asset_stop_fraction=settings["asset_stop_fraction"]))
                        relative = f"scenarios/{arm}/{basket}/{variant}/{fee}bps"
                        for name, table in result.tables.items():
                            _save_table(directory, relative + "/" + name + ".parquet", table, artifacts)
                        summary, monthly = summarize_daily(result, arm=arm, basket=basket,
                            variant=variant, cost_bps=fee, expected_sessions=len(bundle.sessions))
                        summary.update(action_count=result.summary["action_count"],
                            unfilled_retired_order_count=result.summary["unfilled_retired_order_count"],
                            source_identity_certified=False, knowledge_time_certified=False,
                            same_fixed_original_selection=True, conditional_complete_path_comparison_only=False)
                        if result.status == "complete_proxy":
                            summary["trading_cost_paid"] = float(result.market_trades.cost.sum())
                        _save_table(directory, relative + "/monthly.parquet", monthly, artifacts)
                        _write_json(directory / relative / "summary.json", result.summary)
                        artifacts[relative + "/summary.json"] = _pin(directory, directory / relative / "summary.json")
                        comparison.append(summary)
                        if not result.failures.empty:
                            failures.append(result.failures.assign(arm=arm,basket=basket,variant=variant,cost_bps=fee))
                        if not result.action_events.empty:
                            counts = result.action_events.groupby(["event_id","side"], dropna=False).size()
                            action_coverage.extend(dict(arm=arm,basket=basket,variant=variant,cost_bps=fee,event_id=event,
                                side=side,applied_lot_count=int(count)) for (event,side),count in counts.items())
                        counts = result.orders.groupby("status",dropna=False).size() if not result.orders.empty else []
                        order_coverage.extend(dict(arm=arm,basket=basket,variant=variant,cost_bps=fee,status=status,
                            original_order_count=int(count)) for status,count in getattr(counts,"items",lambda:[])())
                        print(json.dumps({"arm":arm,"basket":basket,"variant":variant,"cost_bps":fee,
                            "status":result.status,"first_failure_date":summary["first_failure_date"]}),flush=True)
        reports = {"comparison":pd.DataFrame(comparison),
            "failures":pd.concat(failures,ignore_index=True) if failures else pd.DataFrame(columns=["date","mark","reason","source_security_id","intrinio_security_id","price_field","role","arm","basket","variant","cost_bps"]),
            "action_coverage":_report_frame(action_coverage,["arm","basket","variant","cost_bps","event_id","side","applied_lot_count"]),
            "order_coverage":_report_frame(order_coverage,["arm","basket","variant","cost_bps","status","original_order_count"])}
        for name, frame in reports.items():
            _save_table(directory, name + ".parquet", frame, artifacts)
        verify_action_inputs(planned["input_plan"])
        for pin in identity["source_pins"]:
            _verify_pin(root,pin)
        _verify_pin(root,original_pointer)
        _check_resources(cfg,started,directory)
        audit = {"status":"COMPLETE","build_identity_sha256":identity_hash,"artifacts":artifacts,
            "execution_seconds":time.monotonic()-started,"peak_rss_bytes":_rss(),"scenarios":len(comparison),
            "original_selection_bytes_unchanged":True,"original_rebalance_bytes_unchanged":True,
            "original_backtest_pointer_unchanged":True,"all_pinned_sources_unchanged":True,
            "financial_2020_plus_read":False,
            "source_projection_mixed_row_group_numeric_decoding_possible":True,
            "excluded_financial_values_returned_or_used":False,
            "models_fitted":False,"RET":"withheld_all_24_signals",
            "mandatory_original_OHLC_issues_repaired":3,"applied_OHLC_repair_count":len(repair_ledger),
            "supplemental_bar_coverage": supplemental.ledger.to_dict("records"),
            "complete_proxy_paths":sum(row["complete_proxy_path"] for row in comparison),
            "failed_closed_paths":sum(not row["complete_proxy_path"] for row in comparison),
            "source_plan":bundle.provenance,"unresolved_action_inputs":list(planned["input_plan"].readiness_failures),
            "historical_identity_certified":False,"knowledge_time_certified":False,
            "economic_accounting_certified":False,"complete_path_only_model_leaderboard":False,
            "configuration_sha256":canonical_sha256(cfg)}
        _write_json(directory / "audit.json",audit)
        _write_json(directory.parent / "LATEST.json",{"path":directory.name,"build_identity_sha256":identity_hash,
            "audit_sha256":_hash(directory / "audit.json")})
    except Exception as error:
        _write_json(directory / "failure.json",{"exception":type(error).__name__,"message":str(error)})
        raise
    return read_action_backtest(root)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("plan","run","report"))
    parser.add_argument("--project",type=Path,default=Path.cwd())
    parser.add_argument("--evidence-dir",type=Path)
    args=parser.parse_args(argv)
    if args.action=="plan":
        planned=plan_action_backtest(args.project,evidence_dir=args.evidence_dir)
        print(json.dumps({"build_identity_sha256":planned["build_identity_sha256"],
            "read_plan":planned["input_plan"].read_plan,"original_selected_sha256":planned["identity"]["original_selected_sha256"]},indent=2))
    else:
        saved=run_action_backtest(args.project,evidence_dir=args.evidence_dir) if args.action=="run" else read_action_backtest(args.project)
        print(saved.comparison.to_string(index=False))


if __name__=="__main__":
    main()
