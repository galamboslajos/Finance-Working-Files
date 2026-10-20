"""Cache-only 2020--2023 walk-forward inputs with explicit BBBY quarantine.

This additive path never claims the unresolved old-BBBY route is repaired. It
retains the original monthly and daily membership rows, invalidates only the
reviewed wrong route, and plans the admitted ID request before opening prices.
Only new 2020--2023 annual caches are planned; metadata guards protect2024+.
No cache miss can fetch data, and loading requires explicit frozen authority.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .catalog import canonical_sha256
from .data import CALENDAR_COLUMNS, session_dates
from .identity_amendment import (AMENDMENT_ID, SOURCE_ROUTES, _ALLOWED,
    IdentityAmendment, apply_identity_amendment)
from .identity_policy import PROVISIONAL_IDENTITY_POLICY
from .monthly import build_month_windows
from .universe import DataContractError, EXCLUDED_SCHEMA, UniverseSnapshot, iso_date


FALLBACK_POLICY = "BBBY_KNOWN_WRONG_ROUTE_WALKFORWARD_FALLBACK_V1"
SCOPE_STATUS = "DATA_LIMITED_RESTRICTED_UNIVERSE"
META_ARTIFACTS = ("original_cohort", "daily_cohorts", "exchange_calendar")
PRICE_ARTIFACTS = ("feature_history_daily", "daily_history_raw")
USED_ARTIFACTS = (*META_ARTIFACTS, *PRICE_ARTIFACTS)
KEYS = ("source_security_id", "signal_date")
PRICE_KEYS = ("intrinio_security_id", "trading_date")
PRICE_COLUMNS = (*PRICE_KEYS, "frequency", "intraperiod", "open", "high", "low", "close",
    "volume", "adj_open", "adj_high", "adj_low", "adj_close", "adj_volume", "factor",
    "split_ratio", "dividend", "raw_ohlc_bounds_valid", "adjusted_ohlc_bounds_valid",
    "price_quality_status")
NUMERICAL_WORKERS = ("catalog.py", "context_features.py", "daily_features.py", "data.py",
    "exchange_sessions.py", "features.py", "full_features.py", "full_loading.py", "full_panel.py",
    "identity_policy.py", "loading.py", "monthly.py", "monthly_loading.py", "panel_coverage.py",
    "paper_features.py", "quote_sessions.py", "universe.py")
# The previously merged role promotion adds public policy aliases and changes
# active full-panel configuration validation. The archived arithmetic is still
# authenticated below; these are the only permitted current-byte differences.
PROMOTED_WORKERS = {
    "context_features.py": "c27af10bb6b0f39637d49ce02abc2e526847916b83a20bc511a617db296087f4",
    "full_panel.py": "e1284ee2f95e498020030eb07757f25b2e51becdabf5fc091f6b542a78794bb0",
}


@dataclass(frozen=True)
class WalkforwardReadLimits:
    max_price_cache_rows: int = 2_000_000
    max_daily_cohort_rows: int = 1_000_000
    max_monthly_cohort_rows: int = 30_000
    max_projected_compressed_bytes: int = 512 * 1024 * 1024
    max_json_bytes: int = 16 * 1024 * 1024


@dataclass(frozen=True)
class WalkforwardYearPlan:
    root: Path
    directory: Path
    year: int
    audit: dict
    calendar: pa.Table
    windows: tuple
    original_cohort: pa.Table
    daily_cohorts: pa.Table
    snapshots: tuple
    requested_ids: tuple[str, ...]
    original_requested_ids: tuple[str, ...]
    raw_start: str
    feature_end: str
    raw_end: str
    read_plan: dict
    provenance: dict
    limits: WalkforwardReadLimits
    fingerprint: str


@dataclass(frozen=True)
class LoadedWalkforwardYear:
    feature_prices: pa.Table
    outcome_prices: pa.Table
    plan: WalkforwardYearPlan
    loading_stats: dict
    provenance: dict


def _need(condition, message):
    if not condition:
        raise DataContractError(message)


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path, limit):
    _need(Path(path).is_file() and Path(path).stat().st_size <= limit, "Missing or oversized cache metadata")
    return json.loads(Path(path).read_text())


def _inside(base, relative):
    base = Path(base).resolve()
    _need(isinstance(relative, str) and not Path(relative).is_absolute(), "Relative pinned artifact path required")
    path = (base / relative).resolve()
    _need(path.is_relative_to(base), "Pinned cache path escapes its root")
    return path


def _verify_artifact(base, item):
    _need(isinstance(item, dict) and {"path", "sha256", "size_bytes"} <= set(item), "Complete artifact pin required")
    path = _inside(base, item["path"])
    _need(path.is_file() and path.stat().st_size == item["size_bytes"]
          and _hash(path) == item["sha256"], "Pinned fallback input changed: " + item["path"])
    return path


def _table_sha(table):
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema.remove_metadata()) as writer:
        writer.write_table(table.replace_schema_metadata(None).combine_chunks())
    return "sha256:" + hashlib.sha256(sink.getvalue()).hexdigest()


def _verify_reference(root, limits):
    roles = _json(root / "configs/context_panel_roles.json", limits.max_json_bytes)
    pin = roles.get("primary", {})
    _need(roles.get("primary_policy") == "available_cohort_80pct"
          and pin.get("context_policy") == "available_cohort_80pct", "Pinned primary 80% reference required")
    directory = _inside(root, pin.get("directory"))
    _need(directory.is_relative_to(root / "outputs"), "Fallback reference must be an existing output")
    audit_path = directory / "audit.json"
    _need(audit_path.is_file() and _hash(audit_path) == pin.get("audit_sha256"), "Primary root audit pin changed")
    audit = _json(audit_path, limits.max_json_bytes)
    identity = _json(directory / "build_identity.json", limits.max_json_bytes)
    _need(canonical_sha256(identity) == pin.get("build_identity_sha256")
          and audit.get("build_identity_sha256") == pin.get("build_identity_sha256")
          and audit.get("status") == "COMPLETE", "Pinned complete primary recipe required")
    contract = identity.get("active_reference_contract", {})
    _need(identity.get("context_policy") == "available_cohort_80pct"
          and identity.get("source_pins") == contract.get("source_pins")
          and bool(contract.get("source_pins")), "Primary recipe source pins disagree")
    worker_pins = contract.get("workers", {})
    archived = _inside(root, roles.get("archived_recipe_directory")) / "src/dmv2"
    for name in NUMERICAL_WORKERS:
        source = Path(__file__).with_name(name)
        _need((archived / name).is_file() and worker_pins.get(name) == _hash(archived / name),
              "Archived numerical recipe changed: " + name)
        _need(source.is_file() and _hash(source) in {worker_pins.get(name), PROMOTED_WORKERS.get(name)},
              "Frozen numerical worker changed: " + name)
    return directory, audit, identity, pin


def _amend_table(table, amendment):
    """Qualify one complete dated cohort at a time, preserving extra clocks."""
    _need(set((*KEYS, "intrinio_security_id", "identity_status", "indices")) <= set(table.column_names),
          "Original complete identity and sleeve metadata required")
    date_values = table["signal_date"].to_pylist()
    _need(date_values == sorted(date_values), "Cohort cache must preserve chronological original keys")
    chunks, ledger, counts = [], [], 0
    for day in sorted(set(date_values)):
        iso_date(day, "cohort date")
        _need(day <= "2023-12-31", "2024+ cohort is outside the financial fallback")
        dated = table.filter(pc.equal(table["signal_date"], day))
        original = dated.to_pylist()
        for row in original:
            _need(isinstance(row.get("indices"), list) and bool(row["indices"])
                  and set(row["indices"]) <= {"sp500", "sp400"}, "Complete valid historical sleeve evidence required")
        projected = [{k: v for k, v in row.items() if k in _ALLOWED} for row in original]
        result = apply_identity_amendment(projected, amendment,
            analysis_through="2023-12-31", identity_policy=PROVISIONAL_IDENTITY_POLICY)
        stale_admission = {"requested", "paper_prediction_eligible", "prediction_eligible"}
        merged = [{**{k: v for k, v in old.items() if k not in stale_admission}, **new}
                  for old, new in zip(original, result.records, strict=True)]
        names = list(dict.fromkeys(name for row in merged for name in row))
        chunks.append(pa.Table.from_pylist([{n: row.get(n) for n in names} for row in merged]))
        ledger.extend(result.ledger);counts += len(result.ledger)
    _need(chunks, "Original membership cache is empty")
    output = pa.concat_tables(chunks, promote_options="default")
    _need(output.select(list(KEYS)).to_pylist() == table.select(list(KEYS)).to_pylist(),
          "Identity amendment changed ordered original membership keys")
    return output, tuple(ledger), counts


def _projection(path, columns):
    footer = pq.ParquetFile(path)
    _need(set(columns) <= set(footer.schema_arrow.names), "Raw annual cache lacks frozen quote columns")
    size = sum(footer.metadata.row_group(g).column(c).total_compressed_size
               for g in range(footer.metadata.num_row_groups)
               for c in range(footer.metadata.row_group(g).num_columns)
               if footer.metadata.row_group(g).column(c).path_in_schema in columns)
    return {"rows": footer.metadata.num_rows, "row_groups": footer.metadata.num_row_groups,
            "columns": list(columns), "projected_compressed_bytes": size}


def _fingerprint(plan):
    return canonical_sha256({"year": plan.year, "root": str(plan.root), "directory": str(plan.directory),
        "audit": plan.audit, "calendar": _table_sha(plan.calendar),
        "original_cohort": _table_sha(plan.original_cohort), "daily_cohorts": _table_sha(plan.daily_cohorts),
        "requested_ids": list(plan.requested_ids), "original_requested_ids": list(plan.original_requested_ids),
        "bounds": [plan.raw_start, plan.feature_end, plan.raw_end], "read_plan": plan.read_plan,
        "provenance": plan.provenance, "limits": asdict(plan.limits),
        "windows": [asdict(w) for w in plan.windows],
        "snapshots": [{"as_of": s.as_of, "universe": s.universe, "table": _table_sha(s.table),
                       "provenance": s.provenance} for s in plan.snapshots]})


def plan_walkforward_year(root, year, *, amendment: IdentityAmendment,
                       approved_missing_route_fallback=False, limits=None):
    """Plan one 2020--2023 outcome year using dates/IDs and footers only.

    Whole-file hashes authenticate existing cache bytes without decoding their
    financial columns. Daily qualification includes the historical warmup;
    month-end membership is never projected backward onto that warmup.
    """
    _need(approved_missing_route_fallback is True, "Explicit data-limited missing-route fallback approval required")
    _need(type(year) is int and 2020 <= year <= 2023, "Walk-forward outcome year must be2020--2023; protected2024+ remains untouched")
    root = Path(root).resolve();limits = WalkforwardReadLimits() if limits is None else limits
    _need(isinstance(limits, WalkforwardReadLimits) and all(type(v) is int and v > 0 for v in asdict(limits).values()),
          "Positive bounded fallback read limits required")
    directory, root_audit, identity, reference_pin = _verify_reference(root, limits)
    item = root_audit.get("years", {}).get(str(year))
    annual_path = _verify_artifact(directory, item)
    annual_dir = annual_path.parent;audit = _json(annual_path, limits.max_json_bytes)
    _need(audit.get("year") == year and audit.get("build_identity_sha256") == reference_pin["build_identity_sha256"],
          "Annual cache belongs to a different pinned recipe/year")
    artifacts = audit.get("artifacts", {})
    _need(set(USED_ARTIFACTS) <= set(artifacts), "Annual cache lacks a required fallback input")
    paths = {name: _verify_artifact(annual_dir, artifacts[name]) for name in USED_ARTIFACTS}
    for name, ceiling in (("original_cohort", limits.max_monthly_cohort_rows),
                          ("daily_cohorts", limits.max_daily_cohort_rows)):
        _need(pq.ParquetFile(paths[name]).metadata.num_rows <= ceiling, "Cohort metadata exceeds frozen row ceiling")
    original = pq.read_table(paths["original_cohort"])
    daily_original = pq.read_table(paths["daily_cohorts"])
    calendar = pq.read_table(paths["exchange_calendar"], columns=list(CALENDAR_COLUMNS))
    windows = build_month_windows(calendar, start_month=f"{year}-01", end_month=f"{year}-12")
    signals = tuple(w.signal_date for w in windows)
    _need(len(windows) == 12 and set(original["signal_date"].to_pylist()) == set(signals),
          "Annual source must retain exactly the original twelve outcome cohorts")
    _need(original["as_of"].to_pylist() == original["signal_date"].to_pylist(), "Membership clocks differ from signal close")
    feature_provenance = audit.get("feature_provenance", {})
    loading = audit.get("loading_provenance", {})
    loading = loading.get("reference_loading_provenance", loading)
    request = loading.get("request", {})
    executed = loading.get("stages", {}).get("raw_daily_prices", {}).get("request", {})
    original_requests = feature_provenance.get("requested_ids")
    _need(isinstance(original_requests, list) and bool(original_requests)
          and all(isinstance(v, str) and bool(v.strip()) and v == v.strip() for v in original_requests)
          and len(original_requests) == len(set(original_requests)), "Original explicit pre-price requests required")
    raw_start = iso_date(request.get("raw_start"), "raw start")
    feature_end = iso_date(request.get("feature_end"), "feature cutoff")
    raw_end = iso_date(request.get("raw_end"), "outcome cutoff")
    _need(raw_start <= signals[0] <= feature_end == signals[-1] < raw_end <= "2023-12-31"
          and raw_end == windows[-1].outcome_end and request.get("history_sessions") == 503
          and request.get("signal_dates") == list(signals)
          and request.get("requested_ids") == original_requests
          and executed.get("filters", {}).get("intrinio_security_id") == original_requests
          and executed.get("start") == raw_start and executed.get("end") == raw_end,
          "Cache bounds/actual executed request disagree with frozen annual request")
    _need(loading.get("source_pins") == identity["source_pins"]
          and feature_provenance.get("mode") == "historical_reconstruction"
          and feature_provenance.get("identity_policy") == PROVISIONAL_IDENTITY_POLICY
          and feature_provenance.get("signal_dates") == list(signals)
          and feature_provenance.get("feature_dictionary_sha256") == identity["active_reference_contract"].get("feature_dictionary_sha256"),
          "Annual source/identity/dictionary recipe pins differ")
    _need(feature_provenance.get("target_windows") == {
        w.signal_date: {"outcome_month": w.outcome_month, "outcome_start": w.outcome_start,
                        "outcome_end": w.outcome_end} for w in windows},
          "Original signal/outcome windows differ from the independent calendar")
    sessions_ordered = session_dates(calendar)
    _need(signals[0] in sessions_ordered and sessions_ordered.index(signals[0]) >= 502
          and sessions_ordered[sessions_ordered.index(signals[0])-502] == raw_start
          and sessions_ordered[sessions_ordered.index(signals[0])-251] == request.get("daily_context_start"),
          "Original exact 503-session feature and 252-observation daily context warmup required")
    amended, monthly_ledger, monthly_count = _amend_table(original, amendment)
    daily, daily_ledger, daily_count = _amend_table(daily_original, amendment)
    daily_dates = set(daily["signal_date"].to_pylist());sessions = set(session_dates(calendar))
    _need(daily_dates <= sessions and max(daily_dates) <= feature_end
          and min(daily_dates) == request.get("daily_context_start"), "Daily warmup membership differs from its original dated request")
    requested = tuple(sorted({r["intrinio_security_id"]
        for table in (amended, daily)
        for r in table.select(["intrinio_security_id", "identity_eligible"]).to_pylist()
        if r["identity_eligible"]}))
    _need(bool(requested) and set(requested) <= set(original_requests), "Amended pre-price request escapes the immutable annual cache request")
    snapshots = tuple(UniverseSnapshot(amended.filter(pc.equal(amended["signal_date"], day)),
        pa.Table.from_pylist([], schema=EXCLUDED_SCHEMA), day, "combined",
        {"mode": "historical_reconstruction", "knowledge_time_verified": False,
         "original_signal_cohort_preserved": True, "fallback_policy": FALLBACK_POLICY,
         "identity_amendment_sha256": amendment.fingerprint},
        {"constituents": sum(value == day for value in amended["signal_date"].to_pylist())}) for day in signals)
    projections = {name: _projection(paths[name], PRICE_COLUMNS) for name in PRICE_ARTIFACTS}
    metadata_projections = {name: _projection(paths[name], PRICE_KEYS) for name in PRICE_ARTIFACTS}
    projected_bytes = sum(v["projected_compressed_bytes"] for v in (*projections.values(), *metadata_projections.values()))
    _need(all(v["rows"] <= limits.max_price_cache_rows for v in projections.values())
          and projected_bytes <= limits.max_projected_compressed_bytes, "Annual fallback quote projection exceeds frozen budget")
    read_plan = {"financial_values_read": False, "price_projections": projections,
        "price_metadata_projections": metadata_projections, "projected_compressed_bytes": projected_bytes,
        "original_monthly_rows": original.num_rows, "original_daily_rows": daily_original.num_rows,
        "monthly_wrong_route_rows": monthly_count, "daily_wrong_route_rows": daily_count,
        "admitted_request_ids": len(requested), "original_request_ids": len(original_requests),
        "source_cache_files": 2, "raw_source_reads": False,
        "semantics": "Whole authenticated annual cache column chunks; separate metadata/date guards before financial decoding."}
    provenance = {"policy": FALLBACK_POLICY, "scope_status": SCOPE_STATUS,
        "approved_missing_route_fallback": True, "fully_identity_repaired": False,
        "replacement_native_unresolved": True, "identity_amendment_id": AMENDMENT_ID,
        "identity_amendment_sha256": amendment.fingerprint, "reference_primary": reference_pin,
        "annual_audit_sha256": item["sha256"], "source_pins": identity["source_pins"],
        "frozen_numerical_workers": identity["active_reference_contract"]["workers"],
        "active_numerical_workers": {n: _hash(Path(__file__).with_name(n)) for n in NUMERICAL_WORKERS},
        "used_artifacts": {name: artifacts[name] for name in USED_ARTIFACTS},
        "original_monthly_keys_sha256": _table_sha(original.select(list(KEYS))),
        "original_daily_keys_sha256": _table_sha(daily_original.select(list(KEYS))),
        "monthly_amendment_ledger": list(monthly_ledger),
        "daily_amendment_ledger_sha256": canonical_sha256(list(daily_ledger)),
        "original_rows_and_denominators_retained": True,
        "membership_requalified_per_historical_day": True, "financial_values_read": False,
        "financial_read_through": "2023-12-31", "financial_2024_plus_read": False,
        "raw_source_files_modified": False, "no_quote_availability_selection": True,
        "df_and_rdc_historical_feature_target_rows_retained": True,
        "historical_knowledge_time_certified": False}
    plan = WalkforwardYearPlan(root, annual_dir, year, audit, calendar, windows, amended,
        daily, snapshots, requested, tuple(original_requests), raw_start, feature_end, raw_end,
        read_plan, provenance, limits, "")
    quality_projection = _projection(paths['daily_history_raw'], [*PRICE_KEYS,'price_quality_status','raw_ohlc_bounds_valid','adjusted_ohlc_bounds_valid'])
    quality = preflight_walkforward_quote_metadata(plan)
    read_plan = {**read_plan, 'quality_metadata_projection': quality_projection,
        'projected_compressed_bytes': projected_bytes + quality_projection['projected_compressed_bytes'],
        'quote_quality_preflight': quality}
    _need(read_plan['projected_compressed_bytes'] <= limits.max_projected_compressed_bytes,
          'Quality and price projections exceed the annual read budget')
    plan = replace(plan, read_plan=read_plan)
    return replace(plan, fingerprint=_fingerprint(plan))


def _read_prices(plan, name):
    pin = plan.provenance["used_artifacts"][name]
    path = _verify_artifact(plan.directory, pin)
    metadata = pq.read_table(path, columns=list(PRICE_KEYS))
    days = metadata["trading_date"].to_pylist()
    for day in set(days):
        _need(plan.raw_start <= iso_date(day, "cache price date") <=
              (plan.feature_end if name == "feature_history_daily" else plan.raw_end),
              "Cached quote metadata exceeds the authorized pre-2024 financial bounds")
    _need(set(metadata["intrinio_security_id"].to_pylist()) <= set(plan.original_requested_ids),
          "Quote cache native ID escapes the original pre-price request")
    _need(metadata.num_rows <= plan.limits.max_price_cache_rows, "Quote row budget exceeded")
    prices = pq.read_table(path, columns=list(PRICE_COLUMNS))
    _need(prices.select(list(PRICE_KEYS)).equals(metadata), "Price metadata changed between guarded projections")
    return prices


def load_walkforward_year(root, *, plan, quote_amendments=None, approved_financial_read=False):
    """Read the two independent bounded cache views; keep daily data separate.

    A compiled, reviewed quote overlay is caller supplied and applied to each
    original view once. Source files are never edited. The overlay fingerprint
    belongs in the caller's frozen run identity before any feature computation.
    """
    _need(approved_financial_read is True, "Explicit frozen-protocol approval required before 2020--2023 price decoding")
    _need(isinstance(plan, WalkforwardYearPlan) and Path(root).resolve() == plan.root
          and _fingerprint(plan) == plan.fingerprint, "Fallback source plan changed after metadata planning")
    directory, root_audit, identity, reference_pin = _verify_reference(plan.root, plan.limits)
    item = root_audit["years"][str(plan.year)]
    _need(item["sha256"] == plan.provenance["annual_audit_sha256"]
          and _verify_artifact(directory, item).parent == plan.directory, "Fallback plan reference changed")
    feature = _read_prices(plan, "feature_history_daily")
    outcome = _read_prices(plan, "daily_history_raw")
    original_feature_rows, original_outcome_rows = feature.num_rows, outcome.num_rows
    quote_fingerprint = None
    if quote_amendments is not None:
        _need(callable(getattr(quote_amendments, "apply", None))
              and callable(getattr(quote_amendments, "verify", None))
              and isinstance(getattr(quote_amendments, "fingerprint", None), str),
              "Reviewed compiled quote amendments required")
        quote_amendments.verify(plan.root)
        quote_fingerprint = quote_amendments.fingerprint
        feature = quote_amendments.apply(feature, start=plan.raw_start, end=plan.feature_end)
        outcome = quote_amendments.apply(outcome, start=plan.raw_start, end=plan.raw_end)
        _need(quote_amendments.fingerprint == quote_fingerprint, "Quote amendment changed during application")
        quote_amendments.verify(plan.root)
    permitted_feature = sorted(day for day in session_dates(plan.calendar)
                               if plan.raw_start <= day <= plan.feature_end)
    permitted_outcome = sorted({day for w in plan.windows for day in (w.signal_date, *w.expected_outcome_sessions)})
    def select(table, permitted, cutoff):
        _need(isinstance(table, pa.Table) and set(PRICE_COLUMNS) <= set(table.column_names), "Quote overlay changed required schema")
        for day in set(table["trading_date"].to_pylist()):
            _need(plan.raw_start <= iso_date(day, "overlaid quote date") <= cutoff <= "2023-12-31", "Quote overlay escaped date bounds")
        _need(set(table["intrinio_security_id"].to_pylist()) <= set(plan.original_requested_ids), "Quote overlay introduced unrequested native ID")
        _need(table.num_rows <= plan.limits.max_price_cache_rows, "Quote amendment exceeds row ceiling")
        filtered = table.filter(pc.and_(pc.is_in(table["intrinio_security_id"], value_set=pa.array(plan.requested_ids)),
                                       pc.is_in(table["trading_date"], value_set=pa.array(permitted))))
        return filtered
    feature = select(feature, permitted_feature, plan.feature_end)
    outcome = select(outcome, permitted_outcome, plan.raw_end)
    for name in USED_ARTIFACTS:
        _verify_artifact(plan.directory, plan.provenance["used_artifacts"][name])
    stats = {**plan.read_plan, "financial_values_read": True,
        "feature_source_rows": original_feature_rows, "outcome_source_rows": original_outcome_rows,
        "feature_returned_rows": feature.num_rows, "outcome_returned_rows": outcome.num_rows}
    provenance = {**plan.provenance, "financial_values_read": True,
        "plan_fingerprint": plan.fingerprint, "quote_amendment_sha256": quote_fingerprint,
        "quote_amendments_fingerprint": quote_fingerprint,
        "cache_artifacts_verified_before_and_after": True,
        "feature_financial_cutoff": plan.feature_end, "outcome_financial_cutoff": plan.raw_end,
        "feature_and_outcome_views_separate": True, "feature_uses_outcome_rows": False,
        "features_calculated": False, "models_fitted": False,
        "annual_cache_2024_plus_numeric_decoding_possible": False,
        "source_projection_mixed_row_group_numeric_decoding_possible": quote_amendments is not None,
        "excluded_financial_values_returned_or_used": False}
    return LoadedWalkforwardYear(feature, outcome, plan, stats, provenance)


@dataclass(frozen=True)
class WalkforwardHistoricalQuoteAdapter:
    """Apply reviewed <=2019 patches inside later warmup without date disguises.

    Metadata partitions the Arrow table. Original order and all post2019 values
    are preserved; actual reviewed append rows are appended after original rows.
    Protected2024 metadata stops before any legacy quote worker is called.
    """
    historical: object
    fingerprint: str

    @property
    def provenance(self):
        return {'policy_id':'walkforward_pre2020_quote_patch_passthrough_v1',
            'fingerprint':self.fingerprint,'historical':self.historical.provenance,
            'only_pre2020_rows_passed_to_historical_worker':True,
            'post2019_rows_changed':False,'original_row_order_preserved':True,
            'source_files_changed':False,'financial_2024_plus_read':False}

    def verify(self,root):
        self.historical.verify(root)
        _need(self.fingerprint==canonical_sha256({'policy_id':'walkforward_pre2020_quote_patch_passthrough_v1',
            'historical_fingerprint':self.historical.fingerprint}),'Historical quote adapter changed')

    def apply(self,table,*,start=None,end=None):
        _need(isinstance(table,pa.Table) and set(PRICE_KEYS)<=set(table.column_names),'Arrow native/date keys required')
        dates=table['trading_date'].to_pylist();natives=table['intrinio_security_id'].to_pylist()
        _need(all(isinstance(d,str) and d<'2024-01-01' for d in dates),'Protected2024 quote metadata refused')
        lower=start or min(dates,default='2020-01-01');upper=end or max(dates,default='2023-12-31')
        _need(lower<=upper<'2024-01-01' and all(lower<=d<=upper for d in dates),'Explicit quote adapter scope mismatch')
        keys=list(zip(natives,dates,strict=True));_need(len(keys)==len(set(keys)),'Duplicate original quote keys')
        positions=[i for i,d in enumerate(dates) if d<='2019-12-31']
        if not positions:return table
        prior=table.take(pa.array(positions,type=pa.int64()))
        corrected=self.historical.apply(prior,start=lower,end=min(upper,'2019-12-31'))
        _need(corrected.schema==table.schema and len(corrected)>=len(prior),'Historical quote adapter changed schema or dropped rows')
        _need(corrected.slice(0,len(prior)).select(list(PRICE_KEYS)).equals(prior.select(list(PRICE_KEYS))),
              'Historical quote adapter changed original key order')
        for field in ('volume','adj_volume','factor','split_ratio','dividend'):
            if field in table.column_names:
                _need(corrected[field].slice(0,len(prior)).equals(prior[field]),'Historical replacement changed original volume/action field')
        source_position={position:i for i,position in enumerate(positions)}
        # An Arrow gather restores original interleaved native/date row order.
        # Later observations are selected directly from the original columns.
        lookup=pa.array([len(table)+source_position[i] if i in source_position else i for i in range(len(table))],type=pa.int64())
        combined=pa.concat_tables([table,corrected.slice(0,len(prior))])
        output=combined.take(lookup)
        if len(corrected)>len(prior):
            appended=corrected.slice(len(prior));append_keys=appended.select(list(PRICE_KEYS)).to_pylist()
            _need(all(row['intrinio_security_id'] in set(natives) and lower<=row['trading_date']<=min(upper,'2019-12-31') for row in append_keys),
                  'Historical append escapes original IDs or date scope')
            final_keys=keys+[(row['intrinio_security_id'],row['trading_date']) for row in append_keys]
            _need(len(final_keys)==len(set(final_keys)),'Historical append collides with original keys')
            output=pa.concat_tables([output,appended])
        _need(output.schema==table.schema and len(output)==len(table)+len(corrected)-len(prior),'Historical warmup patch changed output contract')
        return output

    def apply_table(self,table,*,start=None,end=None):return self.apply(table,start=start,end=end)


def wrap_walkforward_quotes(historical):
    if isinstance(historical,WalkforwardHistoricalQuoteAdapter):return historical
    # The completed bundle already applies old and recent amendments in order;
    # partitioning it again would accidentally suppress its recent layer.
    try:
        from .walkforward_quotes import CompleteWalkforwardQuotes
    except ImportError:
        CompleteWalkforwardQuotes = ()
    if isinstance(historical,CompleteWalkforwardQuotes):return historical
    _need(callable(getattr(historical,'apply',None)) and callable(getattr(historical,'verify',None))
          and isinstance(getattr(historical,'fingerprint',None),str),'Reviewed historical quote bundle required')
    fp=canonical_sha256({'policy_id':'walkforward_pre2020_quote_patch_passthrough_v1','historical_fingerprint':historical.fingerprint})
    return WalkforwardHistoricalQuoteAdapter(historical,fp)


def preflight_walkforward_quote_metadata(plan):
    """Inspect quality states and interior date holes without opening OHLC.

    Complete original requests are reviewed before new forecasts. Findings are
    evidence only and never establish nontradability or alter eligibility.
    """
    _need(isinstance(plan,WalkforwardYearPlan),'Explicit walk-forward plan required')
    path=_verify_artifact(plan.directory,plan.provenance['used_artifacts']['daily_history_raw'])
    columns=[*PRICE_KEYS,'price_quality_status','raw_ohlc_bounds_valid','adjusted_ohlc_bounds_valid']
    footer=pq.ParquetFile(path);_need(set(columns)<=set(footer.schema_arrow.names),'Quality metadata columns absent')
    table=footer.read(columns=columns,use_threads=False)
    dates=table['trading_date'].to_pylist()
    _need(all(plan.raw_start<=d<=plan.raw_end<'2024-01-01' for d in dates),'Quality metadata leaves scoped cache')
    table=table.filter(pc.and_(pc.greater_equal(table['trading_date'],f'{plan.year}-01-01'),pc.less_equal(table['trading_date'],f'{plan.year}-12-31')))
    rows=table.to_pylist();invalid=[];observed={}
    for row in rows:
        native,day=row['intrinio_security_id'],row['trading_date']
        _need(native in plan.original_requested_ids,'Quality cache native outside original request')
        observed.setdefault(native,set()).add(day)
        if row['price_quality_status']!='accepted' or row['raw_ohlc_bounds_valid'] is not True or row['adjusted_ohlc_bounds_valid'] is not True:
            invalid.append(row)
    sessions=[day for day in session_dates(plan.calendar) if f'{plan.year}-01-01'<=day<=f'{plan.year}-12-31']
    missing=[];extents=[]
    for native,seen in sorted(observed.items()):
        first,last=min(seen),max(seen);extents.append({'intrinio_security_id':native,'first_observed_date':first,'last_observed_date':last,'observed_session_keys':len(seen)})
        missing.extend({'intrinio_security_id':native,'trading_date':day} for day in sessions if first<=day<=last and day not in seen)
    invalid.sort(key=lambda x:(x['intrinio_security_id'],x['trading_date']))
    return {'scope':'entire_original_annual_requested_cache_not_selected_forecasts',
        'year':plan.year,'metadata_columns':columns,'financial_columns_read':False,
        'invalid_quote_keys':invalid,'invalid_quote_key_count':len(invalid),
        'interior_missing_keys':missing,'interior_missing_key_count':len(missing),
        'observed_native_date_extents':extents,'original_requested_ids':list(plan.original_requested_ids),
        'only_existing_first_last_interval_checked':True,'first_last_dates_are_audit_only':True,
        'missing_keys_establish_nontradability':False,'eligibility_changed':False,
        'financial_2024_plus_read':False,'source_artifact_pin':plan.provenance['used_artifacts']['daily_history_raw']}


def resolve_walkforward_year(root,run,identity,year):
    """Return an authenticated old or newly built annual reference, metadata only.

    Reused years carry external audit and parent identity pins. New years use
    THIS_BUILD in the immutable plan; the completed phase audit pins their new
    annual audits, avoiding a circular identity hash.
    """
    root,run=Path(root).resolve(),Path(run).resolve()
    _need(run.is_relative_to(root) and type(year)is int and 2010<=year<=2023,'Bounded walk-forward annual reference required')
    pin=canonical_sha256(identity)
    _need(_json(run/'build_identity.json',16*1024*1024)==identity,'Walk-forward root identity changed')
    reference=identity.get('annual_sources',{}).get(str(year));_need(isinstance(reference,dict),'Explicit annual source reference absent')
    directory=_inside(run if reference.get('parent_reference')=='THIS_BUILD' else root,reference['directory'])
    if reference.get('parent_reference')=='THIS_BUILD':
        _need(2020<=year<=2023 and directory==run/'years'/str(year),'New annual reference leaves declared destination')
        complete=_json(run/'audit.json',16*1024*1024)
        _need(complete.get('status')=='COMPLETE' and complete.get('build_identity_sha256')==pin,'Walk-forward feature phase incomplete or changed')
        annual_path=_verify_artifact(run,complete['years'][str(year)])
        expected_parent=pin
        audit_pin={'path':str(annual_path.relative_to(root)),'size_bytes':annual_path.stat().st_size,'sha256':_hash(annual_path)}
    else:
        audit_pin=reference['audit_pin'];annual_path=_verify_artifact(root,audit_pin)
        expected_parent=reference['build_identity_sha256']
        parent_path=_verify_artifact(root,reference['parent_identity_pin'])
        _need(canonical_sha256(_json(parent_path,16*1024*1024))==expected_parent,'Reused annual parent identity changed')
        parent_audit_path=_verify_artifact(root,reference['parent_audit_pin'])
        parent_audit=_json(parent_audit_path,16*1024*1024)
        _need(parent_audit.get('status')=='COMPLETE' and parent_audit.get('build_identity_sha256')==expected_parent,
              'Reused parent completion audit changed')
    _need(annual_path.parent==directory,'Annual audit points outside declared directory')
    audit=_json(annual_path,16*1024*1024)
    _need(audit.get('status')=='COMPLETE' and audit.get('year')==year and audit.get('build_identity_sha256')==expected_parent,'Annual source year/recipe changed')
    return {'directory':directory,'audit':audit,'build_identity_sha256':expected_parent,'audit_pin':audit_pin}
