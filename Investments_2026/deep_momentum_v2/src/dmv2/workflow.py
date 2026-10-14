"""Notebook orchestration; settings are public, observations remain local.

Run All performs explicit bounded foundation checks. This module does not
silently turn a research date range into a full scan, feature build or model fit.
"""
from dataclasses import asdict, dataclass
import errno
import json
from pathlib import Path
import hashlib
from importlib.metadata import version
import platform
import re
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq

from .catalog import SourceCatalog, canonical_sha256
from .chronology import build_annual_folds
from .data import load_universe
from .loading import PriceLoader, ReadBudget
from .monthly_loading import load_monthly_sample
from .universe import attach_quotes, iso_date


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate workflow setting: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"Invalid JSON constant: {value}")


def load_workflow(root):
    """Load the one workflow configuration and validate before any source read."""
    root = Path(root).resolve()
    path = root / "configs/research_workflow.json"
    settings = json.loads(path.read_text(), object_pairs_hook=_object,
                          parse_constant=_reject_constant)
    validate_workflow(settings)
    return settings


def validate_workflow(settings):
    if (not isinstance(settings, dict) or type(settings.get("schema_version")) is not int
            or settings.get("schema_version") != 2):
        raise ValueError("Workflow requires schema_version 2")
    required = {"schema_version", "research_start", "training", "warmup",
                "universe", "inspection", "monthly_returns", "execution"}
    if set(settings) != required:
        raise ValueError("Workflow settings have missing or unknown top-level fields")
    fields = {
        "training": {"mode", "train_years", "test_years", "last_test_year"},
        "warmup": {"policy", "raw_start"},
        "universe": {"primary", "engineering_sample"},
        "inspection": {"as_of", "run_universe_audit", "run_price_sample",
                       "price_sample_size", "persist_audit", "read_budget"},
        "monthly_returns": {"enabled", "start_month", "end_month", "sample_size",
                            "return_basis", "calendar_basis", "persist", "read_budget"},
        "execution": {"scope", "full_panel_build_enabled", "model_execution_enabled"},
    }
    for name, expected in fields.items():
        if not isinstance(settings[name], dict) or set(settings[name]) != expected:
            raise ValueError(f"{name} has missing or unknown settings")
    training = settings["training"]
    if training.get("test_years") != 1 or type(training.get("test_years")) is not int:
        raise ValueError("This workflow supports one-year outcome tests")
    build_annual_folds(settings["research_start"], training["last_test_year"],
                       training["train_years"], mode=training["mode"])
    warmup = settings["warmup"]
    if warmup != {"policy": "pending_feature_and_return_audit", "raw_start": None}:
        raise ValueError("Feature warmup is not yet defined; do not freeze it through a notebook override")
    universe = settings["universe"]
    if universe.get("primary") not in (None, "sp500", "sp400", "combined"):
        raise ValueError("Unknown primary universe")
    if universe.get("engineering_sample") not in ("sp500", "sp400", "combined"):
        raise ValueError("Unknown engineering-sample universe")
    inspection = settings["inspection"]
    day = iso_date(inspection["as_of"], "inspection as_of")
    if day < settings["research_start"] or day > f"{training['last_test_year']:04d}-12-31":
        raise ValueError("Engineering date must be within the selected pre-holdout research period")
    for field in ("run_universe_audit", "run_price_sample", "persist_audit"):
        if type(inspection.get(field)) is not bool:
            raise ValueError(f"{field} requires an explicit boolean")
    if inspection["run_price_sample"] and not inspection["run_universe_audit"]:
        raise ValueError("A price sample requires the historical constituent audit first")
    size = inspection.get("price_sample_size")
    if type(size) is not int or not 1 <= size <= 10:
        raise ValueError("Engineering price_sample_size must be in 1..10")
    budget = inspection["read_budget"]
    if not isinstance(budget, dict):
        raise ValueError("read_budget must be an explicit object")
    try:
        ReadBudget(**budget)
    except TypeError as error:
        raise ValueError("Unknown read_budget setting") from error
    monthly = settings["monthly_returns"]
    for field in ("enabled", "persist"):
        if type(monthly[field]) is not bool:
            raise ValueError(f"monthly_returns.{field} requires an explicit boolean")
    for field in ("start_month", "end_month"):
        value = monthly[field]
        if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}", value):
            raise ValueError(f"monthly_returns.{field} requires YYYY-MM")
        iso_date(value + "-01", field)
    first, last = monthly["start_month"], monthly["end_month"]
    if first > last or first < settings["research_start"][:7] or last > f"{training['last_test_year']:04d}-12":
        raise ValueError("Monthly outcomes must be within the selected pre-holdout research period")
    length = 12 * (int(last[:4]) - int(first[:4])) + int(last[5:]) - int(first[5:]) + 1
    if length > 3:
        raise ValueError("The local monthly engineering stage is capped at three outcome months")
    if type(monthly["sample_size"]) is not int or not 1 <= monthly["sample_size"] <= 10:
        raise ValueError("Monthly engineering sample_size must be in 1..10")
    if monthly["return_basis"] != "provider_adjusted_close_ratio":
        raise ValueError("Only the documented provider-adjusted close research ratio is implemented")
    if monthly["calendar_basis"] != "exchange_calendars_XNYS":
        raise ValueError("Monthly dates require the independent XNYS calendar and source reconciliation")
    if not isinstance(monthly["read_budget"], dict):
        raise ValueError("Monthly read_budget must be an explicit object")
    try:
        ReadBudget(**monthly["read_budget"])
    except TypeError as error:
        raise ValueError("Unknown monthly read_budget setting") from error
    if settings["execution"] != {
        "scope": "bounded_foundations_and_monthly_sample", "full_panel_build_enabled": False,
        "model_execution_enabled": False,
    }:
        raise ValueError("Full panels and models are not implemented in this foundation workflow")


@dataclass(frozen=True)
class InspectionRun:
    snapshot: object
    prices: object | None
    quote_coverage: object | None
    report: dict


def worker_code_provenance():
    """Fingerprint the workers actually imported, independent of an unrelated Git HEAD."""
    directory = Path(__file__).resolve().parent
    names = ("__init__.py", "catalog.py", "chronology.py", "data.py",
             "exchange_sessions.py", "loading.py", "monthly.py", "monthly_loading.py", "universe.py", "workflow.py")
    return {
        "worker_sha256": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                          for name in names},
        "python_version": platform.python_version(), "pyarrow_version": pa.__version__,
        "exchange_calendars_version": version("exchange-calendars"),
        "pandas_version": version("pandas"),
    }


def run_inspection(catalog, settings):
    """Run the configured one-date audit; never select a modelling subset."""
    validate_workflow(settings)
    controls = settings["inspection"]
    if not controls["run_universe_audit"]:
        return None
    loaded = load_universe(
        catalog, as_of=controls["as_of"],
        universe=settings["universe"]["engineering_sample"],
        mode="historical_reconstruction", budget=ReadBudget(**controls["read_budget"]),
    )
    snapshot = loaded.snapshot
    report = {
        "schema_version": 1, "workflow_sha256": canonical_sha256(settings),
        "research_start": settings["research_start"],
        "engineering_date": snapshot.as_of, "universe": snapshot.universe,
        "code_provenance": worker_code_provenance(),
        "source_pins": catalog.provenance(), "summary": snapshot.summary,
        "pit": snapshot.provenance, "loading_stats": loaded.loading_stats,
        "loading_provenance": loaded.loading_provenance,
        "constituent_ledger": snapshot.table.to_pylist(),
        "membership_exclusions": snapshot.excluded.to_pylist(),
        "price_sampling": {"enabled": controls["run_price_sample"]},
    }
    prices = coverage = None
    if controls["run_price_sample"]:
        ids = sorted({row["intrinio_security_id"] for row in snapshot.table.to_pylist()
                      if row["identity_status"] == "resolved"})[:controls["price_sample_size"]]
        if not ids:
            raise ValueError("No uniquely mapped engineering IDs; source identity review is required")
        prices = PriceLoader(catalog, ReadBudget(**controls["read_budget"])).load(
            ids, snapshot.as_of, snapshot.as_of,
            ["close", "volume", "adj_close", "adj_volume", "price_quality_status"],
        )
        coverage = attach_quotes(snapshot, prices.table, requested_security_ids=ids)
        report["price_sampling"].update({
            "requested_ids": ids, "stats": prices.stats, "provenance": prices.provenance,
            "rows": prices.table.to_pylist(), "coverage_summary": coverage.summary,
            "coverage_ledger": coverage.table.select(
                ["source_security_id", "identity_status", "quote_status"]).to_pylist(),
        })
    return InspectionRun(snapshot, prices, coverage, report)


@dataclass(frozen=True)
class MonthlyRun:
    sample: object
    report: dict


def run_monthly_sample(catalog, settings):
    """Run a capped engineering sample; retain daily prices and the original cohort."""
    validate_workflow(settings)
    controls = settings["monthly_returns"]
    if not controls["enabled"]:
        return None
    sample = load_monthly_sample(
        catalog, start_month=controls["start_month"], end_month=controls["end_month"],
        universe=settings["universe"]["engineering_sample"],
        sample_size=controls["sample_size"], budget=ReadBudget(**controls["read_budget"]),
        mode="historical_reconstruction",
    )
    report = {
        "schema_version": 1, "workflow_sha256": canonical_sha256(settings),
        "source_pins": catalog.provenance(), "code_provenance": worker_code_provenance(),
        "research_start": settings["research_start"], "summary": sample.panel.summary,
        "return_provenance": sample.panel.provenance,
        "windows": [asdict(window) for window in sample.panel.windows],
        "requested_ids": list(sample.requested_ids),
        "loading_stats": sample.loading_stats, "loading_provenance": sample.loading_provenance,
        "signal_cohorts": [
            {"as_of": snapshot.as_of, "summary": snapshot.summary, "pit": snapshot.provenance,
             "constituent_ledger": snapshot.table.to_pylist(),
             "membership_exclusions": snapshot.excluded.to_pylist()}
            for snapshot in sample.snapshots],
        "artifact_roles": {
            "daily.parquet": "Unchanged source daily OHLCV/action/quality observations for the declared sample",
            "monthly.parquet": "Original-cohort monthly research returns and explicit coverage states",
            "source_calendar.parquet": "Original pinned vendor calendar retained without repair",
            "exchange_calendar.parquet": "Independent XNYS sessions used for the bounded monthly windows",
        },
    }
    return MonthlyRun(sample, report)


def save_monthly_sample(root, monthly):
    """Atomically save independent daily/monthly sample tables and their audit.

    Never overwrite evidence under unchanged settings/source/code/runtime pins.
    Nonfinite source observations stay unchanged in Parquet, not nonstandard JSON.
    """
    root = Path(root).resolve()
    report = monthly.report
    pins = {field: report[field] for field in ("workflow_sha256", "source_pins", "code_provenance")}
    identifier = canonical_sha256(pins).split(":")[1][:16]
    first, last = monthly.sample.panel.windows[0], monthly.sample.panel.windows[-1]
    parent = (root / "outputs/notebook_audits").resolve()
    destination = parent / f"monthly-{first.outcome_month}-{last.outcome_month}-{identifier}"
    if not parent.is_relative_to(root) or not destination.resolve().is_relative_to(root):
        raise ValueError("Monthly output directory leaves the project")
    parent.mkdir(parents=True, exist_ok=True)
    buffers, artifacts = {}, {}
    for name, table in (("daily.parquet", monthly.sample.daily_prices.table),
                        ("monthly.parquet", monthly.sample.panel.table),
                        ("source_calendar.parquet", monthly.sample.source_calendar),
                        ("exchange_calendar.parquet", monthly.sample.calendar)):
        sink = pa.BufferOutputStream()
        pq.write_table(table, sink, compression="zstd")
        buffers[name] = sink.getvalue().to_pybytes()
        artifacts[name] = {"sha256": hashlib.sha256(buffers[name]).hexdigest(),
                           "size_bytes": len(buffers[name]), "rows": table.num_rows,
                           "columns": [{"name": f.name, "type": str(f.type)} for f in table.schema]}
    text = json.dumps({**report, "artifacts": artifacts}, indent=2, sort_keys=True, allow_nan=False) + "\n"

    def verify_existing():
        if not destination.is_dir():
            raise ValueError("Monthly artifact is not a directory; preserve and investigate")
        audit = destination / "audit.json"
        if not audit.resolve().is_relative_to(root) or not audit.is_file() or audit.read_text() != text:
            raise ValueError("Monthly evidence changed under unchanged provenance; preserve and investigate")
        for name, data in buffers.items():
            path = destination / name
            if not path.resolve().is_relative_to(root) or not path.is_file() or path.read_bytes() != data:
                raise ValueError("Monthly table evidence changed; preserve and investigate")
        return destination

    if destination.exists():
        return verify_existing()
    with tempfile.TemporaryDirectory(dir=parent, prefix=".monthly-tmp-") as temporary:
        directory = Path(temporary)
        for name, data in buffers.items():
            (directory / name).write_bytes(data)
        (directory / "audit.json").write_text(text, encoding="utf-8")
        try:
            directory.rename(destination)
        except OSError as error:
            if error.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                raise
            return verify_existing()
    return destination


def save_inspection(root, inspection):
    """Idempotently save a private audit, isolated from the versioned notebook.

    The same settings/source/code/runtime pins identify the same artifact. A changed result
    under that identity is rejected rather than silently overwriting evidence.
    """
    root = Path(root).resolve()
    pins = {"workflow_sha256": inspection.report["workflow_sha256"],
            "source_pins": inspection.report["source_pins"],
            "code_provenance": inspection.report["code_provenance"]}
    identifier = canonical_sha256(pins).split(":")[1][:16]
    directory = (root / "outputs/notebook_audits").resolve()
    if not directory.is_relative_to(root):
        raise ValueError("Notebook output directory leaves the project")
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"inspection-{inspection.snapshot.as_of}-{identifier}.json"
    if not destination.resolve().is_relative_to(root):
        raise ValueError("Notebook output file leaves the project")
    text = json.dumps(inspection.report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if destination.exists():
        if destination.read_text() != text:
            raise ValueError("Audit evidence changed under the same settings/source/code/runtime pins; preserve and investigate")
        return destination
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=directory,
                                         suffix=".tmp", delete=False) as stream:
            stream.write(text)
            temporary = Path(stream.name)
        temporary.replace(destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return destination
