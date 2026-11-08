"""Build causal daily momentum features from local published five-minute bars.

No labels, model fitting, downloads, price filling or portfolio selection occur.
All vendor-derived outputs belong in the private, Git-ignored data directory.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import duckdb
import exchange_calendars as xcals
import pandas as pd


RETURN_NAMES = ("1h", "1d", "3d", "5d", "10d")
MODEL_FEATURES = [f"{kind}_ret_{h}" for h in RETURN_NAMES
                  for kind in ("z", "mean", "std")]
POLICY_VERSION = "total_return_paper15_v1"


def literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def session_calendar(start, end):
    """Include the next session's date, without reading its prices/membership."""
    end = pd.Timestamp(end)
    calendar = xcals.get_calendar("XNYS", start=start,
                                 end=(end + pd.Timedelta(days=14)).date().isoformat())
    schedule = calendar.schedule
    result = pd.DataFrame({
        "signal_date": pd.to_datetime(schedule.index).tz_localize(None),
        "market_open_utc": schedule["open"].to_numpy(),
        "market_close_utc": schedule["close"].to_numpy(),
    })
    result["next_session"] = result["signal_date"].shift(-1)
    result = result[result.signal_date.le(end)].copy()
    result["session_num"] = range(len(result))
    result["expected_slots"] = ((result.market_close_utc - result.market_open_utc)
                                .dt.total_seconds() / 300).astype(int)
    return result


def scan_daily(con, paths, progress=print):
    """Aggregate exact calendar anchors and coverage in one monthly scan."""
    started = time.monotonic()
    for number, path in enumerate(paths, 1):
        con.from_parquet(str(path), hive_partitioning=False).create_view(
            "month_bars", replace=True)
        query = """
        WITH typed AS (
          SELECT b.security_id, CAST(b.trading_date AS DATE) signal_date, b.ticker,
            try_cast(b.bar_timestamp_utc AS TIMESTAMPTZ) ts,
            try_cast(b.available_at_utc AS TIMESTAMPTZ) available_at,
            b.open,b.high,b.low,b.close,b.minute_bars,b.source
          FROM month_bars b
        ), regular AS (
          SELECT b.*,s.market_open_utc,s.market_close_utc,
            epoch(ts)::BIGINT % 300=0 aligned,
            coalesce(isfinite(open) AND isfinite(high) AND isfinite(low)
              AND isfinite(close) AND low>0 AND high>=low
              AND open BETWEEN low AND high AND close BETWEEN low AND high
              AND available_at=ts+INTERVAL '5 minutes'
              AND minute_bars BETWEEN 1 AND 5,false) valid_row
          FROM typed b JOIN sessions s USING(signal_date)
          WHERE ts>=s.market_open_utc AND ts<s.market_close_utc
        )
        SELECT security_id,signal_date,min(ticker) ticker,
          count(*) rth_rows,
          count(DISTINCT CASE WHEN aligned THEN ts END) observed_slots,
          count(*)-count(DISTINCT ts) duplicate_rows,
          count_if(NOT aligned OR NOT valid_row) bad_rows,
          count_if(minute_bars<5) short_minute_buckets,
          count(DISTINCT source) source_count,
          count_if(source IS NULL OR source='') missing_source_rows,
          min(source) source,
          count_if(ts=market_open_utc) opening_rows,
          count_if(ts=market_close_utc-INTERVAL '5 minutes') closing_rows,
          max(open) FILTER (WHERE ts=market_open_utc AND valid_row) session_open,
          max(close) FILTER (WHERE ts=market_close_utc-INTERVAL '5 minutes'
            AND valid_row) session_close,
          -- A bar labelled close-65m ENDS at close-60m. Using the bar
          -- labelled close-60m would accidentally produce a 55-minute return.
          max(close) FILTER (WHERE ts=market_close_utc-INTERVAL '65 minutes'
            AND valid_row) hour_start_close,
          max(available_at) max_available_at
        FROM regular GROUP BY security_id,signal_date
        """
        statement = "CREATE TEMP TABLE daily AS " if number == 1 else "INSERT INTO daily "
        con.execute(statement + query)
        if number == 1 or number % 12 == 0 or number == len(paths):
            progress(f"Scanned {number}/{len(paths)} months in "
                     f"{time.monotonic()-started:.1f}s", flush=True)
    duplicate_keys = con.execute("""SELECT count(*) FROM (
        SELECT security_id,signal_date FROM daily GROUP BY ALL HAVING count(*)>1)
        """).fetchall()[0][0]
    if duplicate_keys:
        raise ValueError("Overlapping daily aggregates; expected one price partition per month.")


def assemble_features(con):
    """Create features for every resolved signal-day member, including exclusions.

    Required relations: sessions, daily and members. A dense security/calendar
    grid prevents missing days or membership breaks from shortening lookbacks.
    """
    for relation in ("sessions", "members"):
        key = "signal_date" if relation == "sessions" else "security_id,signal_date"
        if con.execute(f"SELECT count(*) FROM (SELECT {key} FROM {relation} "
                       "GROUP BY ALL HAVING count(*)>1)").fetchall()[0][0]:
            raise ValueError(f"Duplicate keys in {relation}.")
    lag_columns = ",\n".join(
        f"lag(session_close,{h}) OVER w close_lag_{h}" for h in (1, 3, 5, 10))
    return_expressions = ["session_close/hour_start_close-1"] + [
        f"session_close/close_lag_{h}-1" for h in (1, 3, 5, 10)]
    returns = ",\n".join(
        f"CASE WHEN price_eligible THEN {expr} END ret_{name}"
        for name, expr in zip(RETURN_NAMES, return_expressions))
    stats = ",\n".join(
        f"avg(ret_{name}) OVER x mean_ret_{name}, "
        f"stddev_pop(ret_{name}) OVER x std_ret_{name}"
        for name in RETURN_NAMES)
    dispersion = " AND ".join(f"std_ret_{name}>0" for name in RETURN_NAMES)
    model_columns = ",\n".join(
        f"CASE WHEN prediction_eligible THEN {expr} END {kind}_ret_{name}"
        for name in RETURN_NAMES
        for kind, expr in (
            ("z", f"(ret_{name}-mean_ret_{name})/nullif(std_ret_{name},0)"),
            ("mean", f"mean_ret_{name}"), ("std", f"std_ret_{name}")))
    con.execute(f"""CREATE OR REPLACE TEMP TABLE features AS
    WITH grid AS (
      SELECT ids.security_id,s.*,
        m.security_id IS NOT NULL is_signal_member,
        coalesce(m.has_conflicting_membership_series,false) membership_conflict,
        m.membership_tickers,
        d.ticker,d.session_open,d.session_close,d.hour_start_close,d.source,
        coalesce(d.observed_slots,0) observed_slots,
        coalesce(d.short_minute_buckets,0) short_minute_buckets,
        coalesce(d.bad_rows,0) bad_rows,
        coalesce(d.duplicate_rows,0) duplicate_rows,
        coalesce(d.source_count=1 AND d.missing_source_rows=0,false) source_ok,
        coalesce(m.security_id IS NOT NULL AND NOT m.has_conflicting_membership_series
          AND d.closing_rows=1 AND d.session_close>0 AND d.bad_rows=0
          AND d.duplicate_rows=0 AND d.source_count=1
          AND d.missing_source_rows=0,false) close_ok,
        coalesce(m.security_id IS NOT NULL
          AND NOT m.has_conflicting_membership_series
          AND d.observed_slots=s.expected_slots AND d.rth_rows=s.expected_slots
          AND d.bad_rows=0 AND d.duplicate_rows=0
          AND d.opening_rows=1 AND d.closing_rows=1
          AND d.source_count=1 AND d.missing_source_rows=0,false) session_ok,
        d.max_available_at
      FROM (SELECT DISTINCT security_id FROM members) ids CROSS JOIN sessions s
      LEFT JOIN members m ON m.security_id=ids.security_id AND m.signal_date=s.signal_date
      LEFT JOIN daily d ON d.security_id=ids.security_id AND d.signal_date=s.signal_date
    ), lags AS (
      SELECT *,{lag_columns},lag(close_ok,10) OVER w close_lag_10_ok,
        CASE WHEN source IS NOT NULL AND lag(source) OVER w IS NOT NULL
          AND source!=lag(source) OVER w THEN 1 ELSE 0 END source_transition
      FROM grid WINDOW w AS (PARTITION BY security_id ORDER BY session_num)
    ), history AS (
      SELECT *, count(*) OVER w history_sessions,
        sum(CASE WHEN session_ok THEN 1 ELSE 0 END) OVER w complete_sessions_10d,
        sum(expected_slots-observed_slots) OVER w missing_slots_10d,
        sum(source_transition) OVER w source_changes_10d,
        sum(short_minute_buckets) OVER w short_minute_buckets_10d,
        max(max_available_at) OVER w feature_max_available_at
      FROM lags WINDOW w AS (PARTITION BY security_id ORDER BY session_num
        ROWS BETWEEN 9 PRECEDING AND CURRENT ROW)
    ), qualified AS (
      SELECT *,coalesce(is_signal_member AND NOT membership_conflict
        AND history_sessions=10 AND complete_sessions_10d=10
        AND close_lag_10_ok AND hour_start_close>0
        AND source_changes_10d=0,false) price_eligible
      FROM history WHERE is_signal_member
    ), raw_returns AS (
      SELECT *,{returns} FROM qualified
    ), cross_section AS (
      SELECT *,count(ret_1h) OVER x cross_section_n,{stats}
      FROM raw_returns WINDOW x AS (PARTITION BY signal_date)
    ), final_flags AS (
      SELECT *,coalesce(price_eligible AND cross_section_n>=2
        AND {dispersion},false) prediction_eligible
      FROM cross_section
    )
    SELECT security_id,CAST(signal_date AS DATE) signal_date,
      market_close_utc signal_cutoff_utc,CAST(next_session AS DATE) next_session,
      ticker,membership_tickers,membership_conflict,source,
      expected_slots,observed_slots,history_sessions,complete_sessions_10d,
      missing_slots_10d,source_changes_10d,short_minute_buckets_10d,
      feature_max_available_at,price_eligible,prediction_eligible,cross_section_n,
      list_filter([
        CASE WHEN membership_conflict THEN 'conflicting_signal_membership' END,
        CASE WHEN history_sessions<10 OR close_lag_10 IS NULL
          THEN 'missing_10_session_anchor_or_history' END,
        CASE WHEN complete_sessions_10d<10 THEN 'incomplete_or_invalid_history' END,
        CASE WHEN NOT coalesce(close_lag_10_ok,false) THEN 'invalid_10_session_anchor' END,
        CASE WHEN hour_start_close IS NULL THEN 'missing_hour_anchor' END,
        CASE WHEN source_changes_10d>0 THEN 'source_change_in_lookback' END,
        CASE WHEN price_eligible AND NOT prediction_eligible
          THEN 'undefined_cross_section' END
      ], lambda reason: reason IS NOT NULL) exclusion_reasons,
      {','.join('ret_'+name for name in RETURN_NAMES)},
      {model_columns}
    FROM final_flags
    """)
    if con.execute("""SELECT count(*) FROM features WHERE
        (prediction_eligible AND feature_max_available_at>signal_cutoff_utc)
        OR (prediction_eligible AND array_length(exclusion_reasons)>0)
        OR (NOT prediction_eligible AND array_length(exclusion_reasons)=0)
        """).fetchall()[0][0]:
        raise ValueError("Feature timing or exclusion ledger failed validation.")


def build(project, output=None, end_date=None, progress=print):
    project = Path(project).resolve()
    prices = project / "data/sp500_5min_20261103"
    pit = project / "data/sp500_pit_20261103"
    output = Path(output or project / "data/features/02_momentum_total_v1").resolve()
    private = (project / "data").resolve()
    if not output.is_relative_to(private):
        raise ValueError("Vendor-derived output must remain inside this project's private data directory.")
    if (output / "MANIFEST.json").exists():
        raise FileExistsError(f"Completed output already exists: {output}. Choose a new output directory.")
    manifest = json.loads((prices / "MANIFEST.json").read_text())
    if not manifest["complete"]:
        raise ValueError("Source preparation is incomplete.")
    start = pd.Timestamp(manifest["membership"]["start"]).date().isoformat()
    release_end = pd.Timestamp(manifest["membership"]["finish"]).date().isoformat()
    end = end_date or release_end
    if not start <= end <= release_end:
        raise ValueError("End date must be within the pinned price release.")
    paths = sorted(p for p in (prices / "adjustment=all").glob("month=*/*.parquet")
                   if p.parent.name.split("=",1)[1] <= end[:7])
    if not paths:
        raise FileNotFoundError("No adjusted monthly price files found.")
    output.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET threads=4")
    con.execute("SET memory_limit='2GB'")
    con.execute("SET TimeZone='UTC'")
    spill = output / ".duckdb_tmp"
    spill.mkdir(exist_ok=True)
    con.execute(f"SET temp_directory={literal(spill)}")
    sessions = session_calendar(start,end)
    con.register("sessions",sessions)
    members_path = pit / "membership_by_security_date.parquet"
    con.execute(f"""CREATE TEMP VIEW members AS SELECT security_id,
      trading_date signal_date,has_conflicting_membership_series,membership_tickers
      FROM read_parquet({literal(members_path)})
      WHERE is_member AND trading_date BETWEEN DATE {literal(start)} AND DATE {literal(end)}""")
    scan_daily(con,paths,progress)
    assemble_features(con)
    con.execute(f"COPY (SELECT * FROM features ORDER BY signal_date,security_id) "
                f"TO {literal(output/'features.parquet')} (FORMAT PARQUET,COMPRESSION ZSTD)")
    con.execute(f"COPY (SELECT * FROM daily ORDER BY signal_date,security_id) "
                f"TO {literal(output/'daily_anchors.parquet')} (FORMAT PARQUET,COMPRESSION ZSTD)")
    con.execute(f"""COPY (SELECT * FROM read_parquet({literal(pit/'membership_daily.parquet')})
      WHERE (security_id IS NULL OR security_id='')
        AND trading_date BETWEEN DATE {literal(start)} AND DATE {literal(end)})
      TO {literal(output/'unresolved_members.parquet')} (FORMAT PARQUET,COMPRESSION ZSTD)""")
    year = con.execute("""SELECT year(signal_date) AS "year",count(*) resolved_member_days,
      count_if(prediction_eligible) eligible_member_days,
      count(DISTINCT signal_date) sessions,
      round(100.0*count_if(prediction_eligible)/count(*),2) eligible_pct
      FROM features GROUP BY 1 ORDER BY 1""").df()
    year.to_parquet(output / "coverage_by_year.parquet",index=False)
    reasons = con.execute("""SELECT reason,count(*) member_days FROM
      (SELECT unnest(exclusion_reasons) reason FROM features)
      GROUP BY reason ORDER BY member_days DESC,reason""").df()
    reasons.to_parquet(output / "exclusion_counts.parquet",index=False)
    summary = con.execute("""SELECT count(*) resolved_member_days,
      count_if(prediction_eligible) eligible_member_days,
      count(DISTINCT CASE WHEN prediction_eligible THEN security_id END) eligible_securities,
      min(CASE WHEN prediction_eligible THEN signal_date END) first_eligible_signal,
      max(signal_date) last_signal
      FROM features""").df().iloc[0].to_dict()
    summary["unresolved_member_days"] = con.execute(
        "SELECT count(*) FROM read_parquet(?)",[str(output/'unresolved_members.parquet')]).fetchall()[0][0]
    result = {
        "policy_version":POLICY_VERSION,"created_at_utc":datetime.now(timezone.utc).isoformat(),
        "scope_start":start,"scope_end":end,"adjustment":"all","calendar":"XNYS",
        "information_cutoff":"Completed previous-session regular close; timestamps are modeled, not archived receipt times.",
        "target_design_only":"Next-session open-to-close; no labels or forward prices loaded into feature assembly.",
        "model_features":MODEL_FEATURES,"raw_return_columns":["ret_"+h for h in RETURN_NAMES],
        "return_definition":"Simple end-to-end returns; 1/3/5/10 session close-to-close include overnight gaps. Last hour is intraday.",
        "cross_section":"Common price-eligible signal-day membership cohort; population standard deviation (ddof=0), no clipping or winsorization.",
        "eligibility":"All expected five-minute slots valid in the latest 10 sessions, valid close 10 sessions ago, resolved nonconflicting signal membership, no source-tag change across the lookback. Present buckets with 1-4 source minutes are admitted.",
        "signal_universe":"Membership at signal close, without consulting next-session membership or outcomes.",
        "summary":summary,"coverage_by_year":year.to_dict("records"),
        "exclusion_counts":reasons.to_dict("records"),
        "source_manifests":{
            "prices_sha256":sha256(prices/'MANIFEST.json'),
            "membership_sha256":sha256(pit/'MANIFEST.json')},
        "code_sha256":sha256(__file__),
        "known_issues":[
            "Historical membership is retrospective and has no archived announcement times; post-March-2026 change dates may precede effective sessions.",
            "Only index-member days have prices; pre-entry and post-exit histories are unavailable in this copy.",
            "Adjusted prices are a retrospective vendor snapshot with mixed retrieval vintages. Source-change exclusions are a conservative proxy, not proof of point-in-time corporate-action accuracy.",
            "Strict slot coverage can exclude legitimate halts or inactive buckets, creating a conditional research universe. Missing future outcomes must not select forecasts or erase positions.",
            "The last five-minute close and first five-minute open are bar-price proxies, not verified closing/opening auction fills.",
            "The model-ready column list excludes raw returns and metadata; no market-cap size feature is supplied."
        ],"usage_rights":"internal_research_only",
    }
    # Convert numpy and date scalars without changing the summary's financial values.
    def json_value(value):
        if hasattr(value,"item"):
            return value.item()
        return str(value)
    (output / "MANIFEST.json").write_text(json.dumps(result,indent=2,default=json_value)+"\n")
    con.close()
    progress(json.dumps(summary,default=json_value),flush=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output",type=Path)
    parser.add_argument("--end-date",help="Optional ISO date for a bounded build; includes all earlier source history.")
    args = parser.parse_args()
    build(args.project,args.output,args.end_date)
