"""Explicit bounded daily development backtest; lightweight saved-report reader.

Run All reads verified reports only. The ``run`` CLI separately reuses saved
2018–2019 probabilities and two outcome-only daily caches, without model fits.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from importlib.metadata import version
import json
from pathlib import Path
import platform
import resource
import sys
import time

import numpy as np
import pandas as pd

from .backtest_accounting import DailyAccountingPolicy, run_daily_accounting
from .backtest_loading import (ARMS, plan_daily_backtest, load_daily_backtest,
    verify_backtest_inputs, _hash, _json, _pin, _verify_pin, _within)
from .backtest_selection import select_monthly_basket
from .catalog import canonical_sha256
from .universe import DataContractError

WORKERS = ("daily_backtest.py", "backtest_loading.py", "backtest_selection.py", "backtest_accounting.py")
CONFIGURATION_SHA256 = "sha256:6545b81b4d658e3676dbe59c7afe4cbdf7878eff7853554c9bbf0af89a7b76ca"


def configuration(root):
    cfg = _json(Path(root) / "configs/daily_backtest.json")
    if canonical_sha256(cfg) != CONFIGURATION_SHA256:
        raise DataContractError("Frozen daily backtest configuration changed; define a separate protocol")
    if (cfg.get("schema_version") != 1
            or cfg.get("protocol_id") != "dmv2-daily-development-2018-2019-fixed10-assetstop10-v1"
            or cfg.get("status") != "FROZEN_BEFORE_FIRST_BACKTEST"
            or cfg.get("start") != "2018-01-01" or cfg.get("end") != "2019-12-31"
            or cfg.get("arms") != list(ARMS)
            or cfg.get("baskets") != ["class_1_10", "class_2_9"]
            or cfg.get("trading_cost_bps") != [0, 5, 10, 25]):
        raise DataContractError("Daily backtest must preserve its bounded four-arm development recipe")
    selection = cfg.get("selection", {})
    if (selection.get("top_k_per_direction") != 10
            or selection.get("candidates") != "corresponding_argmax_class_only"
            or selection.get("ranking") != "descending_corresponding_class_probability"
            or selection.get("long_weight") != .5 or selection.get("short_weight") != -.5
            or selection.get("future_prices_or_labels_select_members") is not False):
        raise DataContractError("Fixed10 within-mode signal-only selection changed")
    variants = cfg.get("variants", {})
    if (set(variants) != {"unstopped", "asset_stop_10pct"}
            or variants["unstopped"].get("asset_stop_fraction") is not None
            or variants["asset_stop_10pct"].get("asset_stop_fraction") != .1):
        raise DataContractError("Daily backtest requires unchanged baseline and separate 10% asset stop")
    if any(cfg["execution"].get(k) is not False for k in ("model_fitting", "raw_source_fallback",
            "financial_2020_plus_read", "cloud_mutations", "automatic_notebook_backtesting")):
        raise DataContractError("Daily backtest cannot fit, read later values or mutate cloud resources")
    for key in ("maximum_candidate_rows", "maximum_projected_read_bytes", "maximum_peak_rss_bytes",
                "maximum_wall_seconds", "maximum_output_bytes"):
        if type(cfg["execution"].get(key)) is not int or cfg["execution"][key] <= 0:
            raise DataContractError("Explicit positive development resource ceilings required")
    if (cfg.get("accounting", {}).get("borrow_annual_bps") != 0
            or cfg["accounting"].get("cash_interest_annual_rate") != 0
            or cfg["accounting"].get("funding_annual_rate") != 0):
        raise DataContractError("This first diagnostic freezes zero borrowing/funding/cash rates")
    return cfg


def _worker_hashes():
    return {name: _hash(Path(__file__).parent / name) for name in WORKERS}


def _identity(root, cfg, plan):
    frozen = _json(plan.pilot_directory / "build_identity.json")
    frozen_workers = frozen["workers"]
    for name, expected in frozen_workers.items():
        if _hash(Path(root) / "src/dmv2" / name) != expected:
            raise DataContractError("Frozen classifier pilot worker changed: " + name)
    return {"schema_version": 1, "configuration": cfg, "workers": _worker_hashes(),
        "source_plan_fingerprint": plan.fingerprint, "source_plan": plan.read_plan,
        "source_pins": list(plan.source_pins), "frozen_pilot_workers": frozen_workers,
        "runtime": {"python": platform.python_version(), **{p: version(p)
            for p in ("pandas", "numpy", "pyarrow", "exchange-calendars")}}}


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _save_table(root, relative, frame, artifacts):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    artifacts[relative] = _pin(root, path)


def _rss():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def _check_resources(cfg, started, directory):
    limits = cfg["execution"]
    if time.monotonic() - started > limits["maximum_wall_seconds"] or _rss() > limits["maximum_peak_rss_bytes"]:
        raise DataContractError("Daily development run exceeded frozen time/RSS ceiling")
    if sum(p.stat().st_size for p in directory.rglob("*") if p.is_file()) > limits["maximum_output_bytes"]:
        raise DataContractError("Daily development run exceeded frozen output ceiling")


def summarize_daily(result, *, arm, basket, variant, cost_bps, expected_sessions=503):
    """Full-period metrics only for complete daily proxy paths, no prefix Sharpe."""
    closes = result.daily.drop_duplicates("date", keep="last").copy()
    if len(closes) != expected_sessions:
        raise DataContractError("Accounting ledger lost expected daily sessions")
    complete = result.status == "complete_proxy" and closes.nav.notna().all()
    row = {"arm": arm, "basket": basket, "variant": variant, "cost_bps": cost_bps,
        "status": result.status, "expected_sessions": expected_sessions,
        "valid_close_sessions": int(closes.nav.notna().sum()),
        "first_failure_date": result.summary["first_failure_date"],
        "stop_count": result.summary["stop_count"], "trade_count": result.summary["trade_count"],
        "initial_nav": 1.0, "final_nav": None, "total_return": None,
        "annualized_arithmetic_mean": None, "cagr": None, "annualized_volatility": None,
        "sharpe_zero_cash_rate": None, "maximum_drawdown": None,
        "total_one_way_turnover": None, "trading_cost_paid": None,
        "complete_proxy_path": bool(complete), "economic_accounting_certified": False}
    monthly = pd.DataFrame({"outcome_month": pd.to_datetime(closes.date).dt.strftime("%Y-%m"),
                            "nav": closes.nav.to_numpy()}).groupby("outcome_month", as_index=False).last()
    # pandas groupby.last skips nulls; explicitly mark an incomplete month.
    available = closes.assign(month=pd.to_datetime(closes.date).dt.strftime("%Y-%m")).groupby("month").nav.apply(lambda x: x.notna().all())
    monthly.loc[~monthly.outcome_month.map(available), "nav"] = np.nan
    monthly["return"] = monthly.nav.div(monthly.nav.shift(1, fill_value=1.0)).sub(1)
    if complete:
        nav = closes.nav.to_numpy(dtype=float)
        returns = nav / np.r_[1., nav[:-1]] - 1.
        sd = float(np.std(returns, ddof=1))
        peak = np.maximum.accumulate(np.r_[1., nav])[1:]
        row.update({"final_nav": float(nav[-1]), "total_return": float(nav[-1] - 1),
            "annualized_arithmetic_mean": float(np.mean(returns) * 252),
            "cagr": float(nav[-1] ** (252 / len(nav)) - 1),
            "annualized_volatility": sd * np.sqrt(252),
            "sharpe_zero_cash_rate": None if sd == 0 else float(np.mean(returns) / sd * np.sqrt(252)),
            "maximum_drawdown": float(np.min(nav / peak - 1)),
            "total_one_way_turnover": float(result.daily.turnover_one_way.sum()),
            "trading_cost_paid": float(result.trades.cost.sum())})
    return row, monthly


@dataclass(frozen=True)
class SavedDailyBacktest:
    path: Path
    audit: dict
    comparison: pd.DataFrame
    selection_coverage: pd.DataFrame
    ret_readiness: pd.DataFrame
    failures: pd.DataFrame


def read_daily_backtest(root):
    """Verify saved hashes and small reports; never execute accounting or fitting."""
    root = Path(root if isinstance(root, (str, Path)) else root.root).resolve()
    parent = root / "outputs/daily_backtest"
    pointer_path = parent / "LATEST.json"
    if not pointer_path.exists():
        raise FileNotFoundError("No saved daily development backtest")
    pointer = _json(pointer_path)
    directory = _within(parent, parent / pointer.get("path", ""))
    audit_path = directory / "audit.json"
    if not audit_path.is_file() or _hash(audit_path) != pointer.get("audit_sha256"):
        raise DataContractError("Saved daily backtest pointer/audit changed")
    audit = _json(audit_path)
    if audit.get("status") != "COMPLETE" or audit.get("build_identity_sha256") != pointer.get("build_identity_sha256"):
        raise DataContractError("Daily backtest execution incomplete or identity changed")
    identity = _json(_verify_pin(directory, audit["artifacts"]["build_identity.json"]))
    if (canonical_sha256(identity) != pointer["build_identity_sha256"]
            or identity["configuration"] != configuration(root) or identity["workers"] != _worker_hashes()):
        raise DataContractError("Daily backtest recipe or worker identity changed")
    for name, expected in identity["frozen_pilot_workers"].items():
        if _hash(root / "src/dmv2" / name) != expected:
            raise DataContractError("Original classifier pilot worker changed")
    for pin in identity["source_pins"]:
        _verify_pin(root, pin)
    for pin in audit["artifacts"].values():
        _verify_pin(directory, pin)
    reports = {name: pd.read_parquet(directory / f"{name}.parquet")
        for name in ("comparison", "selection_coverage", "ret_readiness", "failures")}
    if len(reports["comparison"]) != 64 or len(reports["ret_readiness"]) != 24:
        raise DataContractError("Daily report lost fixed scenarios or monthly RET coverage")
    return SavedDailyBacktest(directory, audit, reports["comparison"], reports["selection_coverage"],
                              reports["ret_readiness"], reports["failures"])


def run_daily_backtest(root):
    """Run the frozen saved-probability development experiment explicitly once."""
    root = Path(root if isinstance(root, (str, Path)) else root.root).resolve()
    cfg = configuration(root)
    started = time.monotonic()
    plan = plan_daily_backtest(root, max_rows=cfg["execution"]["maximum_candidate_rows"],
        max_projected_bytes=cfg["execution"]["maximum_projected_read_bytes"])
    identity = _identity(root, cfg, plan)
    identity_hash = canonical_sha256(identity)
    directory = root / "outputs/daily_backtest" / ("2018-2019-" + identity_hash.removeprefix("sha256:")[:16])
    if directory.exists():
        if (directory / "audit.json").exists() and _json(directory / "audit.json").get("status") == "COMPLETE":
            pointer = {"path": directory.name, "build_identity_sha256": identity_hash,
                       "audit_sha256": _hash(directory / "audit.json")}
            _write_json(directory.parent / "LATEST.json", pointer)
            return read_daily_backtest(root)
        raise DataContractError("An incomplete identical daily run exists; inspect it before a new attempt")
    directory.mkdir(parents=True)
    artifacts = {}
    _write_json(directory / "build_identity.json", identity)
    artifacts["build_identity.json"] = _pin(directory, directory / "build_identity.json")
    _write_json(directory / "audit.json", {"status": "STARTED", "build_identity_sha256": identity_hash})
    try:
        bundle = load_daily_backtest(root, plan=plan)
        selection_sessions = bundle.provenance["calendar"]["selection_sessions"]
        all_coverage, comparison, failures = [], [], []
        ret = select_monthly_basket(next(iter(bundle.predictions.values())), basket="ret",
            sessions=selection_sessions, top_k=10).monthly
        _save_table(directory, "ret_readiness.parquet", ret, artifacts)
        # RET is structurally withheld: no pre-2010 payoff history is admitted.
        # Do not load target returns merely to rediscover the missing months.
        for arm in ARMS:
            for basket in cfg["baskets"]:
                selected = select_monthly_basket(bundle.predictions[arm], basket=basket,
                    sessions=selection_sessions, top_k=10)
                prefix = f"selections/{arm}/{basket}"
                _save_table(directory, prefix + "/original_ledger.parquet", selected.ledger, artifacts)
                _save_table(directory, prefix + "/selected.parquet", selected.selections, artifacts)
                _save_table(directory, prefix + "/rebalances.parquet", selected.rebalances, artifacts)
                coverage = selected.monthly.assign(arm=arm)
                all_coverage.append(coverage)
                # The security list is frozen before the quote join. Restricting
                # the projection to those IDs cannot repair future missingness.
                ids = set(selected.selections.intrinio_security_id)
                prices = bundle.daily_prices.loc[bundle.daily_prices.intrinio_security_id.isin(ids)].copy()
                for variant, settings in cfg["variants"].items():
                    for fee in cfg["trading_cost_bps"]:
                        _check_resources(cfg, started, directory)
                        result = run_daily_accounting(selected.selections, prices, bundle.sessions,
                            rebalances=selected.rebalances, terminal_date="2019-12-31",
                            policy=DailyAccountingPolicy(cost_bps=fee, required_names_per_side=10,
                                asset_stop_fraction=settings["asset_stop_fraction"]))
                        relative = f"scenarios/{arm}/{basket}/{variant}/{fee}bps"
                        for name, table in result.tables.items():
                            _save_table(directory, relative + "/" + name + ".parquet", table, artifacts)
                        summary, monthly = summarize_daily(result, arm=arm, basket=basket,
                            variant=variant, cost_bps=fee, expected_sessions=len(bundle.sessions))
                        _save_table(directory, relative + "/monthly.parquet", monthly, artifacts)
                        comparison.append(summary)
                        if not result.failures.empty:
                            failures.append(result.failures.assign(arm=arm, basket=basket, variant=variant, cost_bps=fee))
                        print(json.dumps({"arm": arm, "basket": basket, "variant": variant,
                            "cost_bps": fee, "status": result.status,
                            "first_failure_date": summary["first_failure_date"]}), flush=True)
        coverage = pd.concat(all_coverage, ignore_index=True)
        failures_frame = pd.concat(failures, ignore_index=True) if failures else pd.DataFrame(
            columns=["date", "mark", "reason", "source_security_id", "intrinio_security_id", "price_field", "role", "arm", "basket", "variant", "cost_bps"])
        for name, frame in (("comparison", pd.DataFrame(comparison)), ("selection_coverage", coverage), ("failures", failures_frame)):
            _save_table(directory, name + ".parquet", frame, artifacts)
        verify_backtest_inputs(plan)
        _check_resources(cfg, started, directory)
        audit = {"status": "COMPLETE", "build_identity_sha256": identity_hash, "artifacts": artifacts,
            "execution_seconds": time.monotonic() - started, "peak_rss_bytes": _rss(),
            "source_files_unchanged": True, "source_file_count": len(plan.source_pins),
            "financial_2020_plus_read": False, "models_fitted": False, "RET": "withheld_all_24_signals",
            "scenarios": len(comparison), "economic_accounting_certified": False,
            "source_plan": bundle.provenance, "configuration_sha256": canonical_sha256(cfg)}
        _write_json(directory / "audit.json", audit)
        _write_json(directory.parent / "LATEST.json", {"path": directory.name,
            "build_identity_sha256": identity_hash, "audit_sha256": _hash(directory / "audit.json")})
    except Exception as error:
        _write_json(directory / "failure.json", {"exception": type(error).__name__, "message": str(error)})
        raise
    return read_daily_backtest(root)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "run", "report"))
    parser.add_argument("--project", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    if args.action == "plan":
        print(json.dumps(plan_daily_backtest(args.project).read_plan, indent=2))
    else:
        saved = run_daily_backtest(args.project) if args.action == "run" else read_daily_backtest(args.project)
        print(saved.comparison.to_string(index=False))


if __name__ == "__main__":
    main()
