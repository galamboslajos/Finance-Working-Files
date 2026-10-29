#!/usr/bin/env python3
"""Assemble daily historical S&P 500 constituents and prices from Deep Momentum.

Run with --help. Defaults to the full *intersection* of the local membership
and price releases. Sources are read only. The output is an end-of-day
historical reconstruction, not a certified knowledge-time or total-return feed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq


PROJECT = Path(__file__).resolve().parent
DEFAULT_SOURCE = Path("/Users/lajosgalambos/Desktop/Finance-Working-Files/"
                      "Investments_2026/deep_momentum_v2")
RELEASES = {
    "prices": (
        "equity_us_intrinio_market_daily_v1/"
        "professional-reference-20260807-r1-pdrs-v2-r3",
        "467c97c83ee6b326d87bcb9ade3bc8e49f44efdbf6ed0afc872c748005d309db"),
    "membership": (
        "equity_us_sp_composite1500_membership_daily_v1/"
        "professional-reference-20260807-r1",
        "cad6a0bb88eb1ce267547bd16e46ca171f227a410169df808e0268ab32a60df7"),
    "identity": (
        "equity_us_index_membership_daily_unified_v1/"
        "professional-reference-20260807-r1",
        "533e7379a43d7886f8307b12552198b803fbf715aac102efb2b46e9244521d08"),
}
# Confirmed wrong routes in the prior project's reviewed audit. Quarantine the
# route, retaining every member/date. Do not guess a replacement or extend a
# bounded quote amendment to the full history.
KNOWN_WRONG_ROUTES = {
    "siblis_ticker_company_00326c7d9ad13903e5e3": "old_bbby_wrong_issuer",
    "siblis_isin_c297a5f81a23c9f90503": "old_bbby_wrong_issuer",
    "siblis_ticker_company_953e9b44770b9402bccb": "el_paso_wrong_issuer",
}
OPCH_BAD_DATES = {
    "2022-01-28", "2022-01-31", "2022-02-02", "2022-02-03", "2022-02-08",
    "2022-02-09", "2022-02-11", "2022-02-14", "2022-02-18", "2022-02-22",
    "2022-02-25", "2022-03-02", "2022-03-04", "2022-03-08", "2022-03-10",
}
AUDIT_REFERENCES = (
    "docs/SOURCE_PIT_AUDIT.md", "docs/BBBY_IDENTITY_AMENDMENT.md",
    "docs/SOURCE_QUALITY_REPAIR_20261016.md",
)
PRICE_COLUMNS = [
    "intrinio_security_id", "trading_date", "ticker", "frequency", "intraperiod",
    "open", "high", "low", "close", "volume", "adj_open", "adj_high",
    "adj_low", "adj_close", "adj_volume", "factor", "split_ratio", "dividend",
    "raw_ohlc_bounds_valid", "adjusted_ohlc_bounds_valid", "price_quality_status",
    "retrieved_at_utc", "source_run_id", "source_page_hash", "source_relative_path",
]
MEMBERSHIP_COLUMNS = [
    "index_id", "index_name", "trading_date", "security_id", "membership_ticker",
    "company_name", "isin", "is_member", "membership_effective_date",
    "membership_end_date", "membership_availability_date", "identity_confidence",
    "pit_policy", "source_ref_id", "source_workbook", "source_sheet", "source_row",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical_hash(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iso_date(value):
    require(isinstance(value, str) and date.fromisoformat(value).isoformat() == value,
            f"Expected YYYY-MM-DD: {value!r}")
    return value


def parse_dates(series, *, optional=False):
    values = series.replace("", None)
    if not optional:
        require(values.notna().all(), f"Missing required date in {series.name}")
    require(values.dropna().str.fullmatch(r"\d{4}-\d{2}-\d{2}").all(),
            f"Noncanonical date in {series.name}")
    return pd.to_datetime(values, format="%Y-%m-%d", errors="raise")


def unique_keys(frame, columns, label):
    require(frame[columns].notna().all().all(), f"Null key in {label}")
    for column in columns:
        require(frame[column].astype(str).str.strip().ne("").all(), f"Blank key in {label}")
    require(not frame.duplicated(columns).any(), f"Duplicate {columns} in {label}; no deduplication performed")


class Release:
    """Only manifest-listed local objects; verify pins, sizes, schemas and hashes."""

    def __init__(self, source_root, key):
        relative, pin = RELEASES[key]
        self.base = (source_root / "data/raw" / relative).resolve()
        self.manifest = json.loads((self.base / "MANIFEST.json").read_text())
        require(canonical_hash(self.manifest) == pin, f"Changed manifest: {key}")
        self.key, self.pin, self.used = key, pin, {}

    def table(self, name):
        definitions = [t for t in self.manifest["schema"]["tables"] if t["name"] == name]
        require(len(definitions) == 1, f"Undeclared table: {name}")
        objects = [o for o in self.manifest["objects"]
                   if o["relative_path"].startswith(f"tables/{name}/")
                   and o["relative_path"].endswith(".parquet")]
        require(bool(objects), f"No pinned objects for {name}")
        paths, total_rows = [], 0
        for obj in objects:
            path = (self.base / obj["relative_path"]).resolve()
            require(path.is_relative_to(self.base), "Manifest path escapes release")
            require(path.is_file() and path.stat().st_size == obj["size_bytes"],
                    f"Missing or wrong-size source: {path}")
            require("sha256:" + file_hash(path) == obj["sha256"], f"Changed source: {path}")
            with pq.ParquetFile(path) as parquet:
                require(parquet.schema_arrow.names == [c["name"] for c in definitions[0]["columns"]],
                        f"Manifest/schema column mismatch: {path}")
                total_rows += parquet.metadata.num_rows
            paths.append(str(path))
        require(total_rows == definitions[0]["row_count"], f"Wrong row total: {name}")
        self.used[name] = objects
        return ds.dataset(paths, format="parquet")

    def provenance(self):
        return {"release_id": self.manifest["release_id"],
                "dataset_id": self.manifest["dataset_id"],
                "manifest_sha256": "sha256:" + self.pin,
                "coverage": self.manifest["coverage"],
                "used_objects": self.used,
                "integrity": "canonical manifest, whole-object SHA256, size, columns, footer row counts"}


def identity_mapping(bridge, links, policy):
    """One row per source security; ambiguity is a visible status, never a guess."""
    bridge = bridge.loc[bridge.source_product_id.eq("sp_composite1500")]
    grouped_links = {uid: part for uid, part in links.groupby("unified_security_id")}
    rows = []
    for sid, part in bridge.groupby("source_security_id", dropna=False):
        require(pd.notna(sid) and bool(str(sid).strip()), "Blank bridge source ID")
        row = {"source_security_id": sid, "unified_security_id": None,
               "intrinio_security_id": None, "bridge_method": None,
               "bridge_confidence": None, "identity_review_required": None,
               "resolution_status": None, "query_identifier_type": None,
               "identity_status": "ambiguous_bridge", "identity_eligible": False,
               "known_identity_issue": KNOWN_WRONG_ROUTES.get(sid)}
        if len(part) == 1 and pd.notna(part.iloc[0].unified_security_id):
            b = part.iloc[0]
            row.update({c: b[c] for c in ("unified_security_id", "bridge_method", "bridge_confidence")})
            candidates = grouped_links.get(b.unified_security_id)
            row["identity_status"] = "unmapped_intrinio"
            if candidates is not None:
                row["identity_status"] = "ambiguous_intrinio"
                if len(candidates) == 1:
                    link = candidates.iloc[0]
                    row.update({c: link[c] for c in (
                        "intrinio_security_id", "identity_review_required",
                        "resolution_status", "query_identifier_type")})
                    native = row["intrinio_security_id"]
                    valid = (pd.notna(native) and bool(str(native).strip())
                             and link.resolution_status == "resolved")
                    review = pd.isna(link.identity_review_required) or bool(link.identity_review_required)
                    row["identity_status"] = ("provisional_review" if review else "resolved") if valid else "unresolved_intrinio"
                    row["identity_eligible"] = bool(valid and (policy == "provisional" or not review))
        if row["known_identity_issue"]:
            row.update(identity_status="known_wrong_issuer", identity_eligible=False)
        rows.append(row)
    return pd.DataFrame(rows)


def qualify_membership(frame, sessions):
    frame = frame.rename(columns={"security_id": "source_security_id", "trading_date": "date"}).copy()
    for col in ("date", "membership_effective_date", "membership_end_date", "membership_availability_date"):
        frame[col] = parse_dates(frame[col], optional=col != "date")
    start, end, available = (frame[c] for c in (
        "membership_effective_date", "membership_end_date", "membership_availability_date"))
    require(not (start.notna() & end.notna() & end.lt(start)).any(), "Membership end precedes start")
    frame["membership_status"] = np.select([
        ~frame.date.isin(sessions), ~frame.is_member.fillna(False).astype(bool),
        start.isna(), start.gt(frame.date), end.lt(frame.date),
        available.isna(), available.gt(frame.date),
    ], ["non_session", "not_member", "effective_date_missing", "future_effective",
        "membership_ended", "availability_missing", "future_availability"], default="eligible")
    return frame


def attach_identity(members, mapping):
    out = members.merge(mapping, on="source_security_id", how="left", validate="many_to_one")
    out["identity_status"] = out.identity_status.fillna("unmapped_bridge")
    out["identity_eligible"] = out.identity_eligible.eq(True)
    # Aliases at disjoint dates may share a native ID. Same-date aliases may not.
    admitted = out.identity_eligible
    collision = out.loc[admitted].duplicated(["date", "intrinio_security_id"], keep=False)
    indices = collision.index[collision]
    out.loc[indices, "identity_status"] = "same_date_target_collision"
    out.loc[indices, "identity_eligible"] = False
    require(len(out) == len(members), "Identity join changed original cohort size")
    return out


def prepare_prices(prices, sessions):
    """Past-only endpoint ratios; gaps and bad endpoints never become zero returns."""
    out = prices.rename(columns={"trading_date": "date", "ticker": "price_ticker"}).copy()
    out["date"] = parse_dates(out.date)
    unique_keys(out, ["intrinio_security_id", "date"], "price history")
    valid_ohlc = pd.Series(True, index=out.index)
    for prefix in ("", "adj_"):
        values = out[[prefix + c for c in ("open", "high", "low", "close")]]
        valid_ohlc &= np.isfinite(values).all(axis=1) & values.gt(0).all(axis=1)
        valid_ohlc &= (out[prefix + "low"].le(values.min(axis=1))
                       & out[prefix + "high"].ge(values.max(axis=1)))
    valid_volume = ((out.volume.isna() | (np.isfinite(out.volume) & out.volume.ge(0)))
                    & (out.adj_volume.isna() | (np.isfinite(out.adj_volume) & out.adj_volume.ge(0))))
    known_bad = (out.intrinio_security_id.eq("sec_zmox6E")
                 & out.date.dt.strftime("%Y-%m-%d").isin(OPCH_BAD_DATES))
    out["quote_status"] = np.select([
        ~out.date.isin(sessions), known_bad,
        ~out.frequency.eq("daily") | ~out.intraperiod.eq(False),
        ~valid_ohlc | ~valid_volume,
        ~out.raw_ohlc_bounds_valid.eq(True) | ~out.adjusted_ohlc_bounds_valid.eq(True)
        | ~out.price_quality_status.eq("accepted"),
    ], ["non_session", "known_bad_quote", "wrong_frequency", "invalid_ohlcv",
        "source_quality_flag"], default="ok")
    # Source observations on exchange holidays remain in the history/audit view,
    # but cannot interrupt or supply the previous trading-session endpoint.
    out = out.sort_values(["intrinio_security_id", "date"]).reset_index(drop=True)
    trading = out.loc[out.date.isin(sessions)].copy()
    group = trading.groupby("intrinio_security_id", sort=False)
    prev_date = group.date.shift()
    previous_session = pd.Series(sessions[:-1], index=sessions[1:])
    expected = trading.date.map(previous_session)
    previous_ok = group.quote_status.shift().eq("ok")
    ready = trading.quote_status.eq("ok") & previous_ok & prev_date.eq(expected)
    ratio = trading.adj_close / group.adj_close.shift() - 1
    trading["provider_adjusted_return_1d"] = ratio.where(ready)
    trading["log_return_1d"] = np.log1p(trading.provider_adjusted_return_1d)
    trading["return_status"] = np.select([
        ~trading.quote_status.eq("ok"), ~prev_date.eq(expected), ~previous_ok,
    ], ["invalid_current_quote", "missing_previous_session", "invalid_previous_quote"], default="ok")
    out = out.join(trading[["provider_adjusted_return_1d", "log_return_1d", "return_status"]])
    out["return_status"] = out.return_status.fillna("non_session")
    return out


def join_panel(members, prices):
    unique_keys(members, ["source_security_id", "date"], "membership")
    unique_keys(prices, ["intrinio_security_id", "date"], "prepared prices")
    left = members.copy()
    # Do not attach a known wrong issuer's prices, even as an incidental merge.
    left["_price_join_id"] = left.intrinio_security_id.where(left.identity_eligible)
    right = prices.rename(columns={"intrinio_security_id": "_price_join_id"})
    out = left.merge(right, on=["_price_join_id", "date"], how="left", validate="many_to_one")
    out = out.drop(columns="_price_join_id")
    out["quote_status"] = out.quote_status.fillna("missing_price")
    out.loc[~out.identity_eligible, "quote_status"] = "identity_unresolved"
    out["return_status"] = out.return_status.fillna("missing_current_quote")
    out.loc[~out.identity_eligible, "return_status"] = "identity_unresolved"
    out["analysis_eligible"] = out.identity_eligible & out.quote_status.eq("ok")
    # All diagnostics, membership interval ends and identities remain auditable.
    # No present-day survival, complete-future-path or liquidity screen is used.
    require(len(out) == len(members), "Price join changed original cohort size")
    return out.sort_values(["date", "source_security_id"]).reset_index(drop=True)


def write_parquet(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Explicit string typing keeps all-null metadata columns consistent by year.
    table = pa.Table.from_pandas(frame, preserve_index=False)
    for i, field in enumerate(table.schema):
        if pa.types.is_null(field.type):
            table = table.set_column(i, field.name, table[field.name].cast(pa.string()))
    pq.write_table(table, path, compression="zstd", row_group_size=100_000)


def stage_dataset(dataset, columns, predicate, stage, *, date_column, label):
    count, ids, dates = 0, set(), set()
    key = "security_id" if label == "membership" else "intrinio_security_id"
    for i, batch in enumerate(dataset.scanner(columns=columns, filter=predicate,
                                             batch_size=100_000, use_threads=False).to_batches()):
        frame = batch.to_pandas()
        if frame.empty:
            continue
        parsed = parse_dates(frame[date_column])
        ids.update(frame[key].dropna().unique())
        dates.update(parsed.unique())
        for year, part in frame.groupby(parsed.dt.year):
            write_parquet(part, stage / f"year={year}" / f"part-{i:05d}.parquet")
        count += len(frame)
        if i % 20 == 0:
            print(f"  {label}: {count:,} rows staged", flush=True)
    return count, ids, pd.DatetimeIndex(sorted(dates))


def read_year(stage, year):
    folder = stage / f"year={year}"
    if not folder.exists():
        return None
    # Read the files directly; an inferred Hive year must not enter the schema.
    return ds.dataset([str(p) for p in sorted(folder.glob("*.parquet"))], format="parquet").to_table().to_pandas()


def build(args):
    source_root, output = args.source_root.resolve(), args.output.resolve()
    require(not output.exists(), f"Output already exists: {output}; choose a new --output")
    require(not output.is_relative_to(source_root), "Write the new dataset outside the source project")
    releases = {key: Release(source_root, key) for key in RELEASES}
    coverage = [r.manifest["coverage"] for r in releases.values()]
    start = iso_date(args.start) if args.start else max(c["start"] for c in coverage)
    end = iso_date(args.end) if args.end else min(c["end"] for c in coverage)
    require(start <= end, "Start is after end")
    require(start >= max(c["start"] for c in coverage) and end <= min(c["end"] for c in coverage),
            "Requested dates exceed the common pinned source coverage")
    history_start = releases["prices"].manifest["coverage"]["start"]
    calendar = xcals.get_calendar("XNYS", start=history_start, end=end)
    sessions = calendar.sessions.tz_localize(None)
    print(f"Full-source range: {start} through {end}; price warmup from {history_start}", flush=True)
    print("Verifying pinned source files...", flush=True)
    membership_source = releases["membership"].table("membership_by_index_by_date")
    bridge = releases["identity"].table("security_identity_bridge").to_table(columns=[
        "source_product_id", "source_security_id", "unified_security_id", "bridge_method", "bridge_confidence"]).to_pandas()
    links = releases["prices"].table("intrinio_identity_links").to_table(columns=[
        "unified_security_id", "intrinio_security_id", "identity_review_required",
        "resolution_status", "query_identifier_type"]).to_pandas()
    mapping = identity_mapping(bridge, links, args.identity_policy)
    price_source = releases["prices"].table("intrinio_daily_prices")
    if args.preflight:
        print(json.dumps({"start": start, "end": end, "history_start": history_start,
                          "price_rows_in_release": price_source.count_rows(),
                          "identity_policy": args.identity_policy,
                          "output": str(output)}, indent=2))
        return
    output.mkdir(parents=True)
    summary = {"membership_rows": 0, "excluded_membership_rows": 0, "panel_rows": 0,
               "price_history_rows": 0, "valid_quote_rows": 0, "valid_return_rows": 0}
    coverage_parts, exclusions = [], Counter()
    all_members, original_dates = set(), set()
    with tempfile.TemporaryDirectory(prefix=".sp500-stage-", dir=output) as temp:
        stage = Path(temp)
        predicate = (ds.field("index_id") == "sp500") & (ds.field("trading_date") >= start) & (ds.field("trading_date") <= end)
        n, source_ids, membership_dates = stage_dataset(
            membership_source, MEMBERSHIP_COLUMNS, predicate, stage / "members",
            date_column="trading_date", label="membership")
        require(n > 0, "No S&P 500 membership in requested range")
        summary["membership_rows"] = n
        mapping = mapping.loc[mapping.source_security_id.isin(source_ids)].copy()
        write_parquet(mapping, output / "identity_mapping.parquet")
        native_ids = sorted(mapping.loc[mapping.identity_eligible, "intrinio_security_id"].dropna().unique())
        require(bool(native_ids), "No admitted price routes")
        predicate = (ds.field("intrinio_security_id").isin(native_ids)
                     & (ds.field("trading_date") >= history_start) & (ds.field("trading_date") <= end))
        stage_dataset(price_source, PRICE_COLUMNS, predicate, stage / "prices",
                      date_column="trading_date", label="prices")
        tail = None
        # Earlier price history is saved for warmup; it never establishes members.
        for year in range(int(history_start[:4]), int(end[:4]) + 1):
            raw = read_year(stage / "prices", year)
            if raw is not None:
                current = raw if tail is None else pd.concat([tail, raw], ignore_index=True)
                prepared = prepare_prices(current, sessions)
                # Carry the exact last XNYS session only. Earlier stale observations
                # cannot bridge a missing year-end session into the next year's return.
                year_sessions = sessions[sessions.year == year]
                tail = raw.loc[raw.trading_date.eq(year_sessions[-1].strftime("%Y-%m-%d"))].copy() if len(year_sessions) else None
                prepared = prepared.loc[prepared.date.dt.year.eq(year)].copy()
                write_parquet(prepared, output / "price_history" / f"year={year}" / "part-00000.parquet")
                summary["price_history_rows"] += len(prepared)
            else:
                prepared, tail = None, None
            members = read_year(stage / "members", year)
            if members is None:
                continue
            unique_keys(members, ["security_id", "trading_date"], "source membership")
            members = qualify_membership(members, sessions)
            excluded = members.loc[~members.membership_status.eq("eligible")]
            if len(excluded):
                write_parquet(excluded, output / "membership_exclusions" / f"year={year}" / "part-00000.parquet")
                exclusions.update(excluded.membership_status)
            members = attach_identity(members.loc[members.membership_status.eq("eligible")], mapping)
            summary["excluded_membership_rows"] += len(excluded)
            if members.empty:
                continue
            if prepared is None:
                # Construct a typed empty quote table without inventing a bar.
                empty = price_source.head(0, columns=PRICE_COLUMNS).to_pandas()
                prepared = prepare_prices(empty, sessions)
            panel = join_panel(members, prepared)
            write_parquet(panel, output / "panel" / f"year={year}" / "part-00000.parquet")
            all_members.update(panel.source_security_id.unique())
            original_dates.update(panel.date.unique())
            daily = panel.groupby("date").agg(
                members=("source_security_id", "size"),
                admitted_identities=("identity_eligible", "sum"),
                valid_quotes=("analysis_eligible", "sum"),
                valid_returns=("provider_adjusted_return_1d", "count"))
            daily["quote_coverage"] = daily.valid_quotes / daily.members
            daily["return_coverage"] = daily.valid_returns / daily.members
            coverage_parts.append(daily)
            summary["panel_rows"] += len(panel)
            summary["valid_quote_rows"] += int(panel.analysis_eligible.sum())
            summary["valid_return_rows"] += int(panel.provider_adjusted_return_1d.notna().sum())
            summary.setdefault("quote_status_counts", Counter()).update(panel.quote_status)
            summary.setdefault("identity_status_counts", Counter()).update(panel.identity_status)
            print(f"  {year}: {len(panel):,} member-days, {panel.analysis_eligible.mean():.1%} usable quotes", flush=True)
    require(summary["panel_rows"] + summary["excluded_membership_rows"] == summary["membership_rows"],
            "Membership reconciliation failed")
    require(bool(coverage_parts), "No eligible member-date observations")
    daily = pd.concat(coverage_parts).sort_index()
    daily = daily.reindex(sessions[(sessions >= start) & (sessions <= end)]).rename_axis("date")
    daily["membership_observed"] = daily.members.notna()
    daily.reset_index().to_csv(output / "coverage_daily.csv", index=False)
    annual = daily.groupby(daily.index.year).agg(
        sessions=("membership_observed", "size"), observed_membership_sessions=("membership_observed", "sum"),
        member_days=("members", "sum"), valid_quote_days=("valid_quotes", "sum"), valid_return_days=("valid_returns", "sum"))
    annual["quote_coverage"] = annual.valid_quote_days / annual.member_days
    annual["return_coverage"] = annual.valid_return_days / annual.member_days
    annual.to_csv(output / "coverage_annual.csv", index_label="year")
    summary.update(unique_source_securities=len(all_members),
                   first_member_date=pd.Timestamp(min(original_dates)).date().isoformat(),
                   last_member_date=pd.Timestamp(max(original_dates)).date().isoformat(),
                   sessions_without_membership=int((~daily.membership_observed).sum()),
                   membership_exclusion_counts=dict(exclusions))
    audit_pins = {name: file_hash(source_root / name) for name in AUDIT_REFERENCES}
    manifest = {
        "schema_version": 1, "status": "complete", "index": "sp500",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "requested_start": start, "requested_end": end, "price_history_start": history_start,
        "source_root": str(source_root), "identity_policy": args.identity_policy,
        "calendar": {"name": "XNYS", "package": "exchange_calendars", "version": xcals.__version__},
        "summary": summary, "sources": {k: r.provenance() for k, r in releases.items()},
        "prior_audit_sha256": audit_pins, "preparation_script_sha256": file_hash(__file__),
        "output_files": [{"path": p.relative_to(output).as_posix(), "rows": pq.ParquetFile(p).metadata.num_rows,
                          "size_bytes": p.stat().st_size, "sha256": file_hash(p)}
                         for p in sorted(output.rglob("*.parquet"))],
        "conventions": {
            "grain": ["source_security_id", "date"], "signal_timing": "end_of_session",
            "membership_end": "inclusive", "availability": "must be <= observation date",
            "join": "source_product_id=sp_composite1500 -> source ID -> unified ID -> Intrinio ID/date",
            "returns": "adj_close[t]/adj_close[previous XNYS session]-1; computed before membership join; no fills",
            "price_history_population": "ever-member mapped native IDs, including pre/post membership history; use panel for dated membership",
            "dividends": "retained source field; never added again to provider-adjusted ratios",
        },
        "limitations": [
            "Historical-effective reconstruction; membership availability is reconstructed and identities lack dated validity.",
            "Adjusted history reflects provider retrieval vintage, not historical revision-as-of data.",
            "Full history was explicitly requested for this separate project, including available 2024+ data; no models/backtests are run.",
            "PIT membership avoids selecting only current survivors; missing quotes, identity conflicts and delisting cash flows still limit unbiased returns.",
            "Provisional resolved routes retain producer review flags; they are not independently certified historical identities.",
            "Known wrong BBBY/El Paso routes are quarantined; bounded prior quote repairs are not extrapolated or silently applied.",
            "Full daily member cohorts are preserved. An analysis_eligible subset is conditional on available admissible data and can be biased.",
            "A membership-only daily panel does not value holdings after index removal; use price_history and separate terminal-event accounting.",
            "Calendar reconciliation is against the recorded exchange_calendars vintage, not independent certification of every historical closure.",
            "Raw/adjusted OHLCV units are retained as supplied; dollar volume and market capitalization are not inferred.",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Dataset ready: {output}")


def load_panel(root=PROJECT / "data/sp500_daily", *, start=None, end=None, columns=None):
    """Read an explicit date slice; retain missing/ambiguous constituents by default."""
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    require(manifest.get("status") == "complete", "Dataset build is incomplete")
    start = iso_date(start) if start else manifest["requested_start"]
    end = iso_date(end) if end else manifest["requested_end"]
    require(start <= end, "Start is after end")
    paths = [str(root / obj["path"]) for obj in manifest["output_files"]
             if obj["path"].startswith("panel/") and int(start[:4]) <= int(obj["path"].split("/")[1][5:]) <= int(end[:4])]
    require(bool(paths), "No panel partitions in requested range")
    source = ds.dataset(paths, format="parquet")
    predicate = (ds.field("date") >= pa.scalar(pd.Timestamp(start))) & (ds.field("date") <= pa.scalar(pd.Timestamp(end)))
    return source.to_table(columns=columns, filter=predicate).to_pandas()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE,
                        help="Deep Momentum V2 project containing the pinned data/raw releases")
    parser.add_argument("--output", type=Path, default=PROJECT / "data/sp500_daily")
    parser.add_argument("--start", help="YYYY-MM-DD; default earliest common source coverage")
    parser.add_argument("--end", help="YYYY-MM-DD; default latest common source coverage")
    parser.add_argument("--identity-policy", choices=["provisional", "strict"], default="provisional",
                        help="Provisional admits unique resolved review-flagged routes, keeping flags")
    parser.add_argument("--preflight", action="store_true", help="Verify source objects and print bounds; create no dataset")
    build(parser.parse_args())


if __name__ == "__main__":
    main()
