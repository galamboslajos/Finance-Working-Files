"""Select published five-minute S&P 500 bars and preserve the full member list.

No aggregation, price filling, or cloud access is performed by this script.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import duckdb
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/alpaca_5min_20261103"
OUT = ROOT / "data/sp500_5min_20261103"
PIT = ROOT / "data/sp500_pit_20261103"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def save_json(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str) + "\n")
    os.replace(tmp, path)


def prepare_membership(c):
    support = SOURCE / "membership_support"
    v3 = json.loads((SOURCE / "MEMBERSHIP_V3_MANIFEST.json").read_text())
    minute = json.loads((SOURCE / "SOURCE_1MIN_MANIFEST.json").read_text())
    reference = next(x for x in minute["source_refs"] if x["source_id"] == "equity_us_market_daily_unified_v3")
    assert digest(SOURCE / "MEMBERSHIP_V3_MANIFEST.json") == reference["source_manifest_sha256"]
    for name, filename in [("membership_matches", "MEMBERSHIP_MATCHES.parquet"),
                           ("membership_price_gaps", "MEMBERSHIP_PRICE_GAPS.parquet")]:
        obj = next(x for x in v3["objects"] if x["relative_path"] == f"tables/{name}/part-00000.parquet")
        path = support / filename
        assert path.stat().st_size == obj["size_bytes"] and digest(path) == obj["sha256"]
        c.from_parquet(str(path)).create_view(name)
    c.from_parquet(str(SOURCE / "tables/security_days/part-00000.parquet")).create_view("security_days")
    c.execute("""CREATE TEMP TABLE sp500_days AS
      SELECT * FROM security_days WHERE list_contains(string_split(index_ids, ','), 'sp500')""")
    # The source excludes conflicting membership series from priced rows, but
    # preserves them in its gap table. Apply the same assignment when enriching.
    c.execute("""CREATE TEMP TABLE priced AS
      SELECT d.*, m.membership_series_id, m.membership_ticker, m.company_name,
             m.isin, m.match_method, m.valid_from, m.valid_to
      FROM sp500_days d JOIN membership_matches m
        ON m.index_id='sp500' AND d.security_id=m.security_id
       AND d.trading_date BETWEEN m.valid_from AND m.valid_to
       AND NOT EXISTS (SELECT 1 FROM membership_price_gaps g
          WHERE g.index_id='sp500' AND g.membership_series_id=m.membership_series_id
            AND g.trading_date=d.trading_date)""")
    priced_count = c.execute("SELECT count(*) FROM priced").fetchone()[0]
    assert priced_count == c.execute("SELECT count(*) FROM sp500_days").fetchone()[0]
    c.execute("""CREATE TEMP TABLE members AS
      SELECT 'sp500' AS index_id, security_id, trading_date, membership_series_id,
             membership_ticker, ticker AS price_ticker, company_name, isin, match_method,
             valid_from AS match_valid_from, valid_to AS match_valid_to,
             true AS is_member, true AS has_daily_price, status AS intraday_status,
             raw_bars_5min, all_bars_5min, NULL::VARCHAR AS price_gap_reason
      FROM priced
      UNION ALL
      SELECT g.index_id, nullif(g.security_id,''), g.trading_date, g.membership_series_id,
             g.membership_ticker, NULL, g.company_name, g.isin, g.match_method,
             m.valid_from, m.valid_to, true, false, 'no_daily_price', 0, 0, g.reason
      FROM membership_price_gaps g JOIN membership_matches m
        ON g.index_id=m.index_id AND g.membership_series_id=m.membership_series_id
       AND g.membership_ticker=m.membership_ticker
       AND g.trading_date BETWEEN m.valid_from AND m.valid_to
      WHERE g.index_id='sp500'""")
    count = c.execute("SELECT count(*) FROM members").fetchone()[0]
    expected = c.execute("SELECT sum(member_dates) FROM membership_matches WHERE index_id='sp500'").fetchone()[0]
    assert count == expected
    assert c.execute("""SELECT count(*) FROM (SELECT membership_series_id,trading_date
        FROM members GROUP BY ALL HAVING count(*)!=1)""").fetchone()[0] == 0
    mismatches = c.execute("""SELECT count(*) FROM membership_matches m
      LEFT JOIN (SELECT membership_series_id,membership_ticker,match_valid_from,match_valid_to,
                        count(*) n FROM members GROUP BY ALL) d
        ON m.membership_series_id=d.membership_series_id
       AND m.membership_ticker=d.membership_ticker
       AND m.valid_from=d.match_valid_from AND m.valid_to=d.match_valid_to
      WHERE m.index_id='sp500' AND coalesce(d.n,0)!=m.member_dates""").fetchone()[0]
    assert mismatches == 0
    PIT.mkdir(parents=True, exist_ok=True)
    daily = PIT / "membership_daily.parquet"
    c.execute(f"""COPY (SELECT * EXCLUDE(trading_date), CAST(trading_date AS DATE) trading_date,
        CASE WHEN starts_with(membership_series_id,'sharadar_sp500:')
          THEN 'sharadar_sp500_list_and_changes' ELSE 'upstream_unified_membership' END
          AS membership_evidence_type
        FROM members ORDER BY trading_date,membership_series_id)
        TO {literal(daily)} (FORMAT PARQUET, COMPRESSION ZSTD)""")
    by_security = PIT / "membership_by_security_date.parquet"
    c.execute(f"""COPY (SELECT 'sp500' index_id,security_id,CAST(trading_date AS DATE) trading_date,
        true is_member, bool_or(has_daily_price) has_daily_price,
        max(raw_bars_5min) raw_bars_5min,max(all_bars_5min) all_bars_5min,
        count(*)>1 has_conflicting_membership_series,
        list(DISTINCT membership_series_id ORDER BY membership_series_id) membership_series_ids,
        list(DISTINCT membership_ticker ORDER BY membership_ticker) membership_tickers,
        list(DISTINCT price_gap_reason ORDER BY price_gap_reason)
          FILTER (WHERE price_gap_reason IS NOT NULL) price_gap_reasons
        FROM members WHERE security_id IS NOT NULL GROUP BY security_id,trading_date
        ORDER BY trading_date,security_id)
        TO {literal(by_security)} (FORMAT PARQUET, COMPRESSION ZSTD)""")
    summary = c.execute("""SELECT count(*) rows, count(DISTINCT security_id) matched_securities,
       count(DISTINCT membership_series_id) membership_series, min(trading_date) start,
       max(trading_date) finish, count_if(NOT has_daily_price) unpriced_member_rows,
       count_if(security_id IS NULL OR security_id='') unresolved_identity_rows FROM members""").fetchdf().to_dict("records")[0]
    for name in ('rows', 'matched_securities', 'membership_series', 'unpriced_member_rows', 'unresolved_identity_rows'):
        summary[name] = int(summary[name])
    caveats = [
        "Retrospective membership snapshot with date-level availability; no observed historical intraday announcement timestamps.",
        "S&P 500 membership after 2026-03-31 uses Sharadar lists and changes; change dates may precede the effective session by up to one week.",
        "The membership list retains rows without price data and identity conflicts; use membership_series_id plus trading_date as its row key.",
        "Use membership_by_security_date.parquet for joins to bars; it has one row per resolved security/date. Unresolved identities remain in membership_daily.parquet.",
        "security_id uses Sharadar permatickers and is compatible with this published bar release; older local Siblis IDs are different.",
    ]
    save_json(PIT / "MANIFEST.json", {"dataset": "sp500_membership_for_published_alpaca_5min",
       "summary": summary, "source_membership_manifest": str(SOURCE / "MEMBERSHIP_V3_MANIFEST.json"),
       "source_security_days": str(SOURCE / "tables/security_days/part-00000.parquet"),
       "validation": {"all_source_membership_intervals_reconciled": True,
                      "unique_membership_series_date": True},
       "known_issues": caveats, "usage_rights": "internal_research_only",
       "objects": [{"path": str(p), "bytes": p.stat().st_size, "sha256": digest(p)}
                   for p in (daily,by_security)]})
    c.execute("""CREATE TEMP TABLE bar_keys AS
      SELECT security_id,trading_date,substring(trading_date,1,7) AS month
      FROM sp500_days""")
    return summary, caveats


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    c = duckdb.connect()
    c.execute("SET threads=4")
    c.execute("SET memory_limit='2GB'")
    summary, caveats = prepare_membership(c)
    source_manifest = json.loads((SOURCE / "MANIFEST.json").read_text())
    assert digest(SOURCE / "SOURCE_1MIN_MANIFEST.json") == source_manifest["source_refs"][0]["source_manifest_sha256"]
    objects = [o for o in source_manifest["objects"] if o["relative_path"].startswith("tables/bars_5min/")]
    complete = {}
    started = time.monotonic()
    while len(complete) < len(objects):
        for obj in objects:
            rel = obj["relative_path"]
            src = SOURCE / rel
            if rel in complete or not src.exists():
                continue
            parts = Path(rel).parts
            adjustment = parts[2].split("=", 1)[1]
            month = parts[3].split("=", 1)[1]
            assert adjustment in ("raw", "all")
            dst = OUT / Path(*parts[2:])
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_suffix(".parquet.building")
            query = f"""SELECT b.* FROM read_parquet({literal(src)},hive_partitioning=false) b
              SEMI JOIN (SELECT security_id,trading_date FROM bar_keys
                WHERE month={literal(month)}) m USING(security_id,trading_date)"""
            c.execute(f"COPY ({query}) TO {literal(tmp)} (FORMAT PARQUET, COMPRESSION ZSTD)")
            expected = c.execute(f"SELECT sum({adjustment}_bars_5min) FROM sp500_days WHERE starts_with(trading_date,?)", [month]).fetchone()[0]
            rows = pq.ParquetFile(tmp).metadata.num_rows
            assert rows == expected, (adjustment, month, rows, expected)
            os.replace(tmp, dst)
            complete[rel] = {"path": str(dst), "rows": rows, "size_bytes": dst.stat().st_size,
                             "sha256": digest(dst), "source_sha256": obj["sha256"]}
            report = {"complete": False, "completed_partitions": len(complete),
                      "expected_partitions": len(objects), "membership": summary,
                      "bars": sum(x["rows"] for x in complete.values()),
                      "bytes": sum(x["size_bytes"] for x in complete.values()),
                      "elapsed_seconds": round(time.monotonic()-started,1), "objects": list(complete.values())}
            save_json(OUT / "MANIFEST.json", report)
            if len(complete) % 10 == 0:
                print(json.dumps({k:v for k,v in report.items() if k!='objects'}), flush=True)
        if len(complete) < len(objects):
            time.sleep(5)
    download = json.loads((SOURCE / "DOWNLOAD_REPORT.json").read_text())
    assert download["complete"] and download["completed_objects"] == len(source_manifest["objects"])
    source_rows = sum(pq.ParquetFile(SOURCE / o["relative_path"]).metadata.num_rows for o in objects)
    expected_source_rows = next(t["row_count"] for t in source_manifest["schema"]["tables"] if t["name"] == "bars_5min")
    assert source_rows == expected_source_rows
    report.update({"complete": True, "dataset": "sp500_published_alpaca_5min",
       "timeframe": "5Min", "price_adjustments": ["raw", "all"],
       "selection": "Date-specific S&P 500 members from the published security_days table; prices copied without aggregation or filling.",
       "source_manifest": str(SOURCE / "MANIFEST.json"),
       "source_manifest_sha256": digest(SOURCE / "MANIFEST.json"),
       "source_rows_verified": source_rows, "source_files_sha256_verified": True,
       "membership_manifest": str(PIT / "MANIFEST.json"),
       "usage_rights": "internal_research_only",
       "known_issues": caveats + [
         "Published bars cover index-member days only, without pre-entry or post-exit stock history.",
         "Adjusted price histories mix July and November 2026 retrieval vintages and may step when the source changes.",
         "Regular-session flags use 09:30-16:00 New York by the clock; apply an exchange calendar to handle early closes.",
         "Pre-market and post-market bars are retained; minute_bars may be less than five and empty buckets have no row.",
       ], "completed_at_utc": datetime.now(timezone.utc).isoformat()})
    save_json(OUT / "MANIFEST.json", report)
    print(json.dumps({k:v for k,v in report.items() if k!='objects'}), flush=True)


if __name__ == "__main__":
    main()
