"""Passive, source-controlled aggregate comparison; never opens private runs.

Authentication here means exact published report/fixture agreement. It is not
an independent numerical audit of private prices, models or backtest ledgers.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from pathlib import Path
import re


REPORT_PATH = "docs/CORRECTED_BACKTEST_COMPARISON_20261021.md"
FIXTURE_PATH = "configs/corrected_comparison_20261021.json"
REPORT_SHA256 = "f5aa7e07ce9e1ae9d812eccc8e1aa22d9bb577484ea57ac0ed18148f44ca32cd"
INPUT_VERSION = "corrected_refit/2020-2023-20261021-v1"
MODELS = (
    "RF · Paper15", "XGBoost · Paper15", "RF · expanded baseline",
    "XGBoost · expanded", "RF · expanded, prior CV settings",
)
BASKETS = ("ret", "class_1_10", "class_2_9")
VARIANTS = ("asset_stop_10pct", "unstopped")
METRICS = ("gross_total_return", "net_total_return", "net_cagr",
           "net_annualized_volatility", "net_sharpe", "net_maximum_drawdown")
PERIOD = {"start": "2020-01-02", "end": "2023-12-29", "outcome_months": 48,
          "trading_sessions": 1006, "continuous_nav": True}
POLICY = {"net_one_way_cost_bps": 10, "gross_one_way_cost_bps": 0,
          "asset_stop_fraction": 0.1, "borrow_cost": 0, "funding_cost": 0,
          "sharpe_cash_rate": 0, "annualization_sessions": 252,
          "already_examined_development_period": True}
COUNTS = {"scenario_count": 120, "complete_paths": 56, "withheld_paths": 64,
          "metric_rows": 30, "cost_rows": 30, "annual_rows": 15}


class ComparisonContractError(ValueError):
    """Published aggregate metadata or values disagree with their contract."""


def _require(condition, message):
    if not condition:
        raise ComparisonContractError(message)


def _same_json(left, right):
    # JSON's booleans must not compare equal to financial zero or count one.
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


def _public_path(root, relative):
    path = Path(root).resolve() / relative
    _require(path.resolve() == path, "Public aggregate/protocol path cannot redirect to private evidence")
    return path


def _load_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, "Duplicate JSON key: " + key)
            result[key] = value
        return result
    try:
        return json.loads(raw, object_pairs_hook=pairs,
                          parse_constant=lambda value: (_ for _ in ()).throw(
                              ComparisonContractError("Nonfinite JSON value: " + value)))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ComparisonContractError("Invalid aggregate JSON") from exc


def _number(value, *, percent=True):
    if value == "—":
        return None
    pattern = r"[+-]?\d+\.\d{2}%" if percent else r"[+-]?\d+\.\d{2}"
    _require(re.fullmatch(pattern, value) is not None, "Unexpected displayed number: " + value)
    try:
        result = Decimal(value[:-1] if percent else value)
    except InvalidOperation as exc:
        raise ComparisonContractError("Invalid displayed number") from exc
    return float(result / 100 if percent else result)


def _table(text, heading, columns):
    _require(text.count("## " + heading + "\n") == 1, "Missing/duplicate report section: " + heading)
    section = text.split("## " + heading + "\n", 1)[1].split("\n## ", 1)[0]
    lines = [line for line in section.splitlines() if line.startswith("|")]
    _require(len(lines) >= 2, "Missing report table")
    rows = [[part.strip() for part in line.strip().strip("|").split("|")] for line in lines]
    _require(rows[0] == columns, "Published report table schema changed")
    _require(all(re.fullmatch(r":?-+:?", part) for part in rows[1]), "Invalid Markdown table separator")
    _require(all(len(row) == len(columns) for row in rows[2:]), "Ragged published report table")
    return rows[2:]


def published_report_tables(text):
    """Parse only the exact public aggregate tables, retaining all withheld rows."""
    _require("56 complete / 64 withheld" in text and "1,006 trading sessions" in text
             and "48 monthly outcomes" in text, "Published period/path-count statement changed")
    metric_header = ["Model", "Basket", "Gross total", "Net10bp total", "Net CAGR",
                     "Net vol", "Net Sharpe", "Net max DD", "Gross / net status"]
    metrics = []
    expected_keys = [(model, basket) for model in MODELS for basket in BASKETS]
    for heading, variant in (("10% per-asset stop: primary comparison", VARIANTS[0]),
                             ("Unstopped comparator", VARIANTS[1])):
        rows = _table(text, heading, metric_header)
        _require([(row[0], row[1]) for row in rows] == expected_keys, "Missing/reordered model or basket rows")
        for row in rows:
            _require(row[8] in ("complete / complete", "withheld / withheld"), "Invalid gross/net availability")
            status = row[8].split(" / ")[0]
            values = [_number(value, percent=index != 4) for index, value in enumerate(row[2:8])]
            _require(all(value is None for value in values) if status == "withheld"
                     else all(value is not None for value in values), "Withheld performance must be null, never zero")
            if status == "complete":
                gross, net, cagr, vol, sharpe, drawdown = values
                _require(gross >= net > -1 and cagr > -1 and vol >= 0 and -1 <= drawdown <= 0,
                         "Invalid return/volatility/drawdown semantics")
                _require(abs((1 + net) ** (252 / 1006) - 1 - cagr) < 0.00015,
                         "CAGR does not describe the full 1,006-session path")
            metrics.append({"model": row[0], "basket": row[1], "variant": variant,
                            "status": status, **dict(zip(METRICS, values))})
    costs = []
    rows = _table(text, "Trading-cost sensitivity", ["Model", "Basket", "Stop", "0bp", "5bp", "10bp", "25bp"])
    expected_cost_keys = [(model, basket, stop) for model in MODELS for basket in BASKETS for stop in ("10%", "none")]
    _require([(row[0], row[1], row[2]) for row in rows] == expected_cost_keys, "Cost comparison grid changed")
    lookup = {(row["model"], row["basket"], row["variant"]): row for row in metrics}
    for row in rows:
        variant = VARIANTS[0] if row[2] == "10%" else VARIANTS[1]
        record = {"model": row[0], "basket": row[1], "variant": variant, "returns": []}
        metric = lookup[row[0], row[1], variant]
        for fee, value in zip((0, 5, 10, 25), row[3:]):
            match = re.fullmatch(r"(.+) \((complete|withheld)\)", value)
            _require(match is not None, "Invalid cost availability cell")
            result, status = _number(match[1]), match[2]
            _require(status == metric["status"] and ((result is None) == (status == "withheld")),
                     "Cost status/null mismatch")
            if fee in (0, 10):
                _require(result == metric["gross_total_return" if fee == 0 else "net_total_return"],
                         "Gross/net cost table does not match metric table")
            record["returns"].append({"cost_bps": fee, "total_return": result, "status": status})
        finite = [item["total_return"] for item in record["returns"] if item["total_return"] is not None]
        _require(finite == sorted(finite, reverse=True), "Increasing trading costs improve the displayed return")
        costs.append(record)
    annual = []
    rows = _table(text, "Annual returns: 10% stop, 10bp", ["Model", "Basket", "2020", "2021", "2022", "2023", "Status"])
    _require([(row[0], row[1]) for row in rows] == expected_keys, "Annual comparison grid changed")
    for row in rows:
        metric = lookup[row[0], row[1], VARIANTS[0]]
        values = [_number(value) for value in row[2:6]]
        _require(row[6] == metric["status"] and all((v is None) == (row[6] == "withheld") for v in values),
                 "Withheld full paths cannot expose annual prefix performance")
        if row[6] == "complete":
            _require(abs(math.prod(1 + value for value in values) - 1 - metric["net_total_return"]) < 0.0005,
                     "Annual slices do not compound to the full-period return")
        annual.append({"model": row[0], "basket": row[1], "status": row[6],
                       "returns": dict(zip(("2020", "2021", "2022", "2023"), values))})
    complete = sum(item["status"] == "complete" for row in costs for item in row["returns"])
    _require(len(metrics) == 30 and len(costs) == 30 and len(annual) == 15
             and complete == 56 and 120 - complete == 64, "Incomplete 120-path aggregate comparison")
    return {"rows": metrics, "cost_sensitivity": costs, "annual_returns": annual}


def read_current_comparison(root):
    """Authenticate bounded public report/fixture only; return plain records."""
    root = Path(root)
    report = _public_path(root, REPORT_PATH)
    fixture = _public_path(root, FIXTURE_PATH)
    _require(report.stat().st_size <= 64 * 1024 and fixture.stat().st_size <= 128 * 1024,
             "Published aggregate byte bound exceeded")
    raw = report.read_bytes()
    _require(hashlib.sha256(raw).hexdigest() == REPORT_SHA256, "Published comparison report hash changed")
    data = _load_json(fixture.read_bytes())
    _require(isinstance(data, dict), "Aggregate fixture must be an object")
    expected_fields = {"schema", "input_version", "source_report", "period", "policy", "counts",
                       "verification_scope", "rows", "cost_sensitivity", "annual_returns"}
    _require(set(data) == expected_fields, "Aggregate fixture fields changed")
    _require(data["schema"] == "dmv2_published_comparison_v1" and data["input_version"] == INPUT_VERSION,
             "Wrong current input version")
    _require(_same_json(data["source_report"], {"path": REPORT_PATH, "sha256": REPORT_SHA256}), "Wrong source report pin")
    _require(_same_json(data["period"], PERIOD) and _same_json(data["policy"], POLICY) and _same_json(data["counts"], COUNTS),
             "Published period, policy or full path counts changed")
    _require(_same_json(data["verification_scope"], {"published_aggregate_only": True,
             "private_independent_receipts_checked": False}), "Public aggregate read must not claim private verification")
    tables = published_report_tables(raw.decode("utf-8"))
    for name, expected in tables.items():
        _require(_same_json(data[name], expected), "Aggregate fixture/report mismatch: " + name)
    return {**data, "status": "PUBLISHED_AGGREGATES_AUTHENTICATED",
            "ret_with_stops": [row.copy() for row in data["rows"]
                               if row["basket"] == "ret" and row["variant"] == VARIANTS[0]],
            "private_verification": "Documented in the report; private independent receipts were not opened or verified by this reader."}


def read_path_target_readiness(root):
    """Read an optional engineering protocol; never read a saved target panel."""
    path = _public_path(root, "configs/path_target_protocol_20261022.json")
    if not path.exists():
        return {"configured": False, "status": "PROTOCOL_NOT_AVAILABLE",
                "target_build_verified": False, "computation_started": False}
    _require(path.stat().st_size <= 64 * 1024, "Target protocol metadata byte bound exceeded")
    raw = path.read_bytes()
    data = _load_json(raw)
    _require(data.get("schema") == "dmv2_path_target_protocol_v1"
             and data.get("protocol_id") == "path-target-engineering-20261022-v1"
             and data.get("status") == "ENGINEERING_ONLY_NOT_FIT_READY", "Wrong path-target engineering protocol")
    execution = data.get("execution", {})
    _require(all(execution.get(key) is False for key in ("model_fitting_enabled", "full_history_build_enabled", "notebook_compute_enabled"))
             and type(execution.get("max_estimator_fits")) is int and execution["max_estimator_fits"] == 0,
             "Notebook target readiness cannot authorize fitting or full builds")
    chronology = data.get("chronology", {})
    expected_clock = {"research_start": "2010-01-01", "last_outcome_year": 2023,
                      "rolling_train_years": 10, "protected_from": "2024-01-01"}
    _require(_same_json({key: chronology.get(key) for key in expected_clock}, expected_clock),
             "Path-target chronology or protected boundary changed")
    _require(data.get("target_families") == ["original_endpoint_control", "executable_return", "return_minus_drawdown"],
             "Target-family comparison changed")
    utility = data.get("candidate_utility", {})
    _require(utility.get("formula") == "cash_return - lambda * maximum_drawdown"
             and type(utility.get("lambda")) in (int, float) and utility["lambda"] == 1.0
             and utility.get("selection_status") == "engineering_candidate_not_empirically_selected",
             "Candidate utility must remain an unselected engineering definition")
    return {"configured": True, "status": data["status"], "protocol_id": data["protocol_id"],
            "protocol_sha256": hashlib.sha256(raw).hexdigest(), "chronology": chronology,
            "target_families": data["target_families"], "candidate_utility": data.get("candidate_utility"),
            "execution": execution, "target_build_verified": False, "computation_started": False,
            "evidence_scope": "Protocol metadata only. No target panel, model, audit receipt or private financial artifact was opened."}
