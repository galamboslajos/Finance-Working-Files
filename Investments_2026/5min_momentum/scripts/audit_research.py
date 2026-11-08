"""Independently audit frozen features, timing, cohorts and selected positions.

Reads local derived five-minute artifacts only. Never fits, fills, downloads or
rewrites a release. Numerical failures and unresolved research assumptions are
separate outputs; passing arithmetic does not certify execution or historical PIT.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import duckdb
import numpy as np
import pandas as pd

from build_momentum_features import session_calendar
from research_checks import execution_timing_violations, feature_quality


def audit(project):
    started = time.monotonic()
    project = Path(project).resolve()
    private = (project/"data").resolve()
    run = private/"models/07_xgb_ret_volume_overnight100_v1"
    feature_dir = private/"features/06_momentum_volume_overnight_v1"
    out = private/"reports/08_research_review"
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads((run/"CONFIG.json").read_text())
    manifest = json.loads((feature_dir/"MANIFEST.json").read_text())
    features = config["model_features"]
    checks, findings = [], []
    def check(name, ok, **details):
        checks.append(dict(check=name, passed=bool(ok), **details))
        print(("PASS " if ok else "FAIL ")+name, flush=True)
    def finding(code, severity, description, **details):
        findings.append(dict(code=code, severity=severity, description=description, **details))
    con = duckdb.connect()
    con.execute("SET threads=6")
    con.execute("SET TimeZone='UTC'")
    con.execute("SET temp_directory=?", [str(out/".duckdb_tmp")])
    for alias, path in {
        "panel": run/"supervised_panel.parquet", "pred": run/"predictions.parquet",
        "pos": run/"positions.parquet", "book": run/"portfolio_daily.parquet",
        "old": private/"features/02_momentum_total_v1/features.parquet",
        "anchors": private/"features/02_momentum_total_v1/daily_anchors.parquet",
        "volume": feature_dir/"daily_volume_anchors.parquet",
        "members": private/"sp500_pit_20261103/membership_by_security_date.parquet",
    }.items():
        con.read_parquet(str(path)).create_view(alias)
    parent_manifest = json.loads((private/"features/05_network_individual_8_v1/MANIFEST.json").read_text())
    sessions = session_calendar(parent_manifest["scope_start"], parent_manifest["scope_end"])
    for h in (1, 3, 5, 10):
        sessions[f"lag_{h}"] = sessions.signal_date.shift(h)
    con.register("sessions", sessions)

    for alias, key in [("panel", "security_id,trade_date"), ("pred", "security_id,trade_date"),
                       ("members", "security_id,trading_date"), ("volume", "security_id,signal_date")]:
        n = con.execute(f"SELECT count(*) FROM (SELECT {key} FROM {alias} GROUP BY ALL HAVING count(*)>1)").fetchone()[0]
        check(alias+" unique row identities", n == 0, duplicates=n)
    timing = con.execute("""SELECT count_if(p.signal_date>=p.trade_date),
        count_if(p.trade_date IS DISTINCT FROM s.next_session),
        count_if(p.history_cutoff_utc IS DISTINCT FROM s.market_close_utc),
        count_if(p.signal_cutoff_utc IS DISTINCT FROM t.market_open_utc)
        FROM panel p LEFT JOIN sessions s USING(signal_date)
        LEFT JOIN sessions t ON p.trade_date=t.signal_date""").fetchone()
    check("signal, target and information clocks match exchange sessions", sum(timing)==0, violations=list(timing))
    membership = con.execute("""SELECT count(*) FROM panel p LEFT JOIN members m
        ON p.security_id=m.security_id AND p.signal_date=m.trading_date
        WHERE m.security_id IS NULL OR NOT m.is_member OR m.has_conflicting_membership_series""").fetchone()[0]
    check("forecast cohort uses resolved signal-day membership", membership==0, violations=membership)

    joins = " ".join(f"LEFT JOIN anchors a{h} ON a{h}.security_id=f.security_id AND a{h}.signal_date=s.lag_{h}" for h in (1,3,5,10))
    errors = ["abs(f.ret_1h-(a.session_close/a.hour_start_close-1))"] + [
        f"abs(f.ret_{h}d-(a.session_close/a{h}.session_close-1))" for h in (1,3,5,10)]
    r = con.execute(f"SELECT {','.join('max('+x+')' for x in errors)},count_if(a10.session_close IS NULL) FROM old f JOIN sessions s USING(signal_date) LEFT JOIN anchors a USING(security_id,signal_date) {joins} WHERE f.prediction_eligible").fetchone()
    check("return horizons reconstructed from exact calendar endpoints", max(r[:5])<1e-12 and r[5]==0, maximum_errors=list(r[:5]), missing_long_anchors=r[5])

    # Independent SQL reconstruction of all six volume fields on a dense calendar.
    con.execute("""CREATE TEMP TABLE volume_reconstruction AS
      WITH grid AS (
        SELECT ids.security_id,s.signal_date,v.* EXCLUDE(security_id,signal_date)
        FROM (SELECT DISTINCT security_id FROM panel) ids CROSS JOIN sessions s
        LEFT JOIN volume v ON ids.security_id=v.security_id AND s.signal_date=v.signal_date
      ), values_ AS (
        SELECT *,CASE WHEN activity_valid THEN dollar_volume END d,
          CASE WHEN activity_valid THEN dollar_volume/session_minutes END rate,
          CASE WHEN activity_valid THEN last_hour_dollar_volume END hour_activity,
          CASE WHEN activity_valid AND signed_valid THEN signed_dollar_volume END signed_activity,
          CASE WHEN activity_valid AND signed_valid THEN dollar_volume END signed_den,
          CASE WHEN activity_valid AND share_volume>0 THEN dollar_volume/share_volume END vw
        FROM grid
      ), rolling AS (
        SELECT *,median(rate) OVER prior10 ref_rate,median(hour_activity) OVER prior10 ref_hour,
          count(rate) OVER prior10 ref_days,count(hour_activity) OVER prior10 hour_ref_days,
          avg(rate) OVER last3 recent, count(rate) OVER last3 n3,
          avg(rate) OVER prior7 preceding,count(rate) OVER prior7 n7,
          sum(signed_activity) OVER last5 signed5,sum(signed_den) OVER last5 den5,
          count(signed_activity) OVER last5 n5
        FROM values_ WINDOW
          prior10 AS (PARTITION BY security_id ORDER BY signal_date ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING),
          last3 AS (PARTITION BY security_id ORDER BY signal_date ROWS BETWEEN 2 PRECEDING AND CURRENT ROW),
          prior7 AS (PARTITION BY security_id ORDER BY signal_date ROWS BETWEEN 9 PRECEDING AND 3 PRECEDING),
          last5 AS (PARTITION BY security_id ORDER BY signal_date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW)
      ) SELECT security_id,signal_date,ref_days,hour_ref_days,
        CASE WHEN ref_rate>0 THEN rate/ref_rate END vol_relative_activity_10d,
        CASE WHEN ref_hour>0 THEN hour_activity/ref_hour END vol_relative_last_hour_10d,
        CASE WHEN n3=3 AND n7=7 AND preceding>0 THEN recent/preceding END vol_activity_trend_3v7,
        CASE WHEN d>0 THEN signed_activity/d END vol_signed_balance_1d,
        CASE WHEN n5=5 AND den5>0 THEN signed5/den5 END vol_signed_balance_5d,
        CASE WHEN closing_price_raw>0 AND vw>0 THEN closing_price_raw/vw-1 END vol_close_vwap_gap
      FROM rolling""")
    reconstruction = []
    for name in features[8:14]:
        result = con.execute(f"""SELECT count(*),count_if((p.{name} IS NULL)!=(v.{name} IS NULL)),
          max(abs(p.{name}-v.{name})) FROM panel p
          LEFT JOIN volume_reconstruction v USING(security_id,signal_date)""").fetchone()
        reconstruction.append(dict(feature=name, rows=result[0], missing_mismatches=result[1], maximum_absolute_error=result[2]))
        check("independent volume formula: "+name, result[1]==0 and result[2]<1e-9, **reconstruction[-1])
    pd.DataFrame(reconstruction).to_parquet(out/"volume_reconstruction.parquet", index=False)

    labels = con.execute("""WITH labelled AS (
        SELECT *,rank() OVER (PARTITION BY trade_date ORDER BY future_oc_return DESC) r,
          count(*) OVER (PARTITION BY trade_date,future_oc_return) ties,
          count(*) OVER (PARTITION BY trade_date) n
        FROM panel WHERE outcome_status='ok' AND isfinite(future_oc_return)
      ) SELECT count(*) FROM labelled WHERE realized_class IS DISTINCT FROM
        CASE WHEN n>=10 THEN least(10,1+floor(10.0*(r-0.5+(ties-1)/2.0)/n)) END""").fetchone()[0]
    check("labels independently reconstructed, including exact ties", labels==0, mismatches=labels)
    outcome = con.execute("""SELECT max(abs(p.future_oc_return-(a.session_close/a.session_open-1))),
        count_if(a.security_id IS NULL) FROM panel p LEFT JOIN anchors a
        ON p.security_id=a.security_id AND p.trade_date=a.signal_date WHERE p.outcome_status='ok'""").fetchone()
    check("targets are exact next-session open-to-close proxies", outcome[0]<1e-12 and outcome[1]==0, maximum_error=outcome[0], missing=outcome[1])
    # Selection is reconstructed before inspecting outcomes, including NaN exits.
    con.execute("""CREATE TEMP TABLE ranked AS SELECT *,
        row_number() OVER (PARTITION BY trade_date ORDER BY ret_score DESC,security_id) r,
        count(*) OVER (PARTITION BY trade_date) population FROM pred""")
    bad = con.execute("""SELECT count(*) FROM pos p LEFT JOIN ranked r USING(security_id,trade_date)
        WHERE r.security_id IS NULL OR
          (p.side='long' AND r.r>p.n_per_side) OR
          (p.side='short' AND r.r<=r.population-p.n_per_side) OR
          abs(abs(p.weight)-0.5/p.n_per_side)>1e-12 OR
          p.future_oc_return IS DISTINCT FROM r.future_oc_return""").fetchone()[0]
    check("top/bottom selection, fixed weights and unknown exits retained", bad==0, violations=bad)
    pnl = con.execute("""WITH z AS (SELECT trade_date,n_per_side,
          sum(weight*future_oc_return) pnl,count_if(NOT isfinite(future_oc_return) OR future_oc_return IS NULL) missing,
          sum(abs(weight)*(2+future_oc_return)) notional
        FROM pos GROUP BY 1,2)
        SELECT count(*) FROM book b JOIN z USING(trade_date,n_per_side)
        WHERE (missing>0 AND b.gross_return IS NOT NULL)
          OR (missing=0 AND (b.gross_return IS NULL OR abs(b.gross_return-z.pnl)>1e-12 OR abs(b.trade_notional-z.notional)>1e-12))""").fetchone()[0]
    check("portfolio P&L and two-sided traded notional", pnl==0, violations=pnl)

    sample = pd.read_parquet(run/"supervised_panel.parquet", columns=features+[
        "security_id","signal_date","trade_date","signal_cutoff_utc","opening_bar_available_at_utc","volume_reference_days","last_hour_reference_days"])
    quality = feature_quality(sample,features)
    quality.to_parquet(out/"feature_distribution.parquet",index=False)
    quality.to_csv(out/"feature_distribution.csv",index=False)
    check("feature domains: no infinite inputs; only gap may be missing", quality.infinite.sum()==0 and quality.loc[quality.feature.ne("overnight_gap"),"missing"].sum()==0)
    domains = con.execute("""SELECT count(*) FROM panel WHERE
        vol_relative_activity_10d<0 OR vol_relative_last_hour_10d<0 OR vol_activity_trend_3v7<0
        OR abs(vol_signed_balance_1d)>1+1e-12 OR abs(vol_signed_balance_5d)>1+1e-12
        OR vol_close_vwap_gap<=-1 OR overnight_gap<=-1""").fetchone()[0]
    check("dimensionless relative-volume and return domains", domains==0, violations=domains)
    reference_short = int(((sample.volume_reference_days<10)|(sample.last_hour_reference_days<10)).sum())
    sample.loc[(sample.volume_reference_days<10)|(sample.last_hour_reference_days<10),
        ["security_id","signal_date","trade_date","volume_reference_days","last_hour_reference_days"]].to_parquet(out/"short_volume_reference_rows.parquet",index=False)
    finding("volume_reference_completeness", "research_design", "The frozen ratio uses available observations within the prior ten scheduled sessions. A full-ten-reference requirement is a separate stricter cohort policy; its affected rows are exported rather than silently removed.", short_reference_rows=reference_short)
    gap_bad = int(execution_timing_violations(sample).sum())
    finding("opening_delivery", "execution_blocker", "The gap is delivered after the assumed same-open decision/fill. The benchmark remains research-only; delayed execution is deferred.", affected_panel_rows=gap_bad)
    finding("retrospective_membership", "unresolved_source_bias", "Membership has no archived announcement timestamps; post-March-2026 Sharadar change dates may precede effective sessions. This is not certified historical as-known PIT.")
    finding("adjustment_vintages", "unresolved_source_bias", "Retrospectively adjusted prices mix retrieval vintages. Same-source checks do not prove historical corporate-action availability.")
    excluded = con.execute("SELECT count(*),count_if(NOT prediction_eligible) FROM old").fetchone()
    cash = con.execute("SELECT n_per_side,count_if(selected_positions=0),count_if(gross_return IS NULL) FROM book GROUP BY 1 ORDER BY 1").fetchall()
    finding("eligibility_selection", "research_design", "Ten fully observed sessions exclude legitimate halts and induce a conditional universe. Do not silently relax or strengthen this cohort in a comparison.", member_days=excluded[0], excluded_member_days=excluded[1], cash_and_unresolved_by_n=cash)
    finding("random_inner_validation", "research_design", "Random 80/20 stock-row validation shares dates and overlapping historical features. Outer folds are past-only, but inner validation is not an independent chronological estimate. A date-grouped sensitivity remains unrun.")
    finding("development_reuse", "research_design", "2017–2026 has already been viewed across several feature/portfolio choices. Walk-forward forecasts are not a fresh final holdout.")
    finding("nonbinding_clipping", "research_design", "Current-inclusive five-sigma clipping with ten-session EWM half-life is mathematically nonbinding. Volume and gap have no extra clipping. Fixing transforms needs a newly versioned release, not editing old results.")
    finding("warmup_and_vol_clock", "research_design", "MACD normalization permits two observations within fixed 195/780-slot windows; this is less strict than full-window warm-up. Five-minute volatility treats overnight first-bar increments as single observations. Both need explicit sensitivity choices before a new comparison.")
    finding("bar_prices_and_shorts", "execution_blocker", "Bar endpoints are not verified auction fills. Quotes, slippage and short borrow are unmodeled. Missing selected outcomes remain unresolved; conditional annual metrics do not certify a complete path.")

    member_checks, payoff_dates, ret_errors, payoff_errors, probability_errors = [], [], [], [], []
    baseline = private/"models/05_xgb_ret_network_features100_uncapped_v1"
    for fold in sorted((run/"folds").iterdir()):
        if not fold.is_dir(): continue
        info = json.loads((fold/"FOLD.json").read_text())
        paths = sorted((fold/"members").glob("member_*.json"))
        mismatch = 0
        for path in paths:
            a, b = json.loads(path.read_text()), json.loads((baseline/"folds"/fold.name/"members"/path.name).read_text())
            fields = ["split_seed","booster_seed","split_train_index_sha256","split_validation_index_sha256","prediction_row_key_sha256"]
            mismatch += int(any(a[k]!=b[k] for k in fields))
        member_checks.append(dict(fold_id=fold.name,members=len(paths),mismatches=mismatch))
        start, finish = pd.Timestamp(info["train_start"]), pd.Timestamp(info["test_start"])
        expected = sessions.loc[sessions.signal_date.ge(start)&sessions.signal_date.lt(finish),"signal_date"]
        dates = con.execute("SELECT trade_date,count(DISTINCT realized_class) k FROM panel WHERE trade_date>=? AND trade_date<? AND realized_class IS NOT NULL GROUP BY 1",[start,finish]).df()
        actual_complete = int(dates.k.eq(10).sum())
        mu = con.execute("""WITH daily AS (
            SELECT trade_date,realized_class,avg(future_oc_return) r FROM panel
            WHERE trade_date>=? AND trade_date<? AND realized_class IS NOT NULL
            GROUP BY 1,2), complete AS (
            SELECT trade_date FROM daily GROUP BY 1 HAVING count(*)=10)
            SELECT realized_class,avg(r) FROM daily JOIN complete USING(trade_date)
            GROUP BY 1 ORDER BY 1""",[start,finish]).fetchall()
        assert len(mu)==10
        payoff_errors.append(float(np.max(np.abs(np.array([x[1] for x in mu])-np.array(info["payoff_vector_class_1_to_10"])))))
        payoff_dates.append(dict(fold_id=fold.name,expected_training_sessions=len(expected),
            complete_payoff_sessions=actual_complete,missing_or_incomplete_sessions=len(expected)-actual_complete,
            old_reported_incomplete_sessions=info["payoff_incomplete_days"]))
        preds = pd.read_parquet(fold/"predictions.parquet",columns=["ret_score"]+[f"p_class_{k}" for k in range(1,11)])
        probs = preds[[f"p_class_{k}" for k in range(1,11)]].to_numpy()
        assert np.isfinite(probs).all() and np.all(probs>=0)
        probability_errors.append(float(np.max(np.abs(probs.sum(axis=1)-1))))
        ret_errors.append(float(np.max(np.abs(preds.ret_score.to_numpy()-probs@np.array(info["payoff_vector_class_1_to_10"])))))
    member_frame = pd.DataFrame(member_checks)
    check("40 complete 100-member folds with unchanged split/row fingerprints", len(member_frame)==40 and member_frame.members.eq(100).all() and member_frame.mismatches.eq(0).all())
    check("RET and payoffs independently reconstructed from historical outcomes", max(ret_errors)<1e-12 and max(payoff_errors)<1e-12 and max(probability_errors)<1e-5,
        maximum_ret_error=max(ret_errors),maximum_payoff_error=max(payoff_errors),maximum_probability_sum_error=max(probability_errors))
    check("approved uncapped validation-stopping configuration", config["max_rounds"] is None and config["early_stopping_patience"]==30 and config["memory_limit_gib"] is None)
    payoff_frame = pd.DataFrame(payoff_dates)
    payoff_frame.to_parquet(out/"training_payoff_date_coverage.parquet",index=False)
    undercount = int((payoff_frame.missing_or_incomplete_sessions-payoff_frame.old_reported_incomplete_sessions).sum())
    finding("payoff_date_reporting", "corrected_reporting", "Old incomplete-payoff counters omitted dates with no supervised rows. The shared payoff helper now accepts the complete exchange calendar. Historical payoff vectors and returns are unchanged.", undercounted_fold_dates=undercount)
    result = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        audit_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        invariant_checks=checks, all_invariants_passed=all(x["passed"] for x in checks),
        findings=findings, execution_ready=False, historical_as_known_pit_certified=False,
        feature_formulas_changed=False, new_model_fitted=False, one_minute_data_used=False,
        frozen_model_configuration_sha256=hashlib.sha256((run/"CONFIG.json").read_bytes()).hexdigest(),
        elapsed_seconds=time.monotonic()-started)
    (out/"AUDIT.json").write_text(json.dumps(result,indent=2,default=str,allow_nan=False)+"\n")
    con.close()
    print(json.dumps({"checks":len(checks),"all_invariants_passed":result["all_invariants_passed"],"open_findings":len(findings),"report":str(out)},indent=2),flush=True)
    if not result["all_invariants_passed"]:
        raise ValueError("Frozen research failed an invariant; inspect AUDIT.json.")
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project",type=Path,default=Path(__file__).resolve().parents[1])
    audit(parser.parse_args().project)
