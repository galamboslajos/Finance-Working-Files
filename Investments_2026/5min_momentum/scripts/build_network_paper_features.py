"""Build eight short-scale individual Network Momentum variables, no graph.

Preserve the original forecast cohort and exact five end-to-end return horizons.
Use older causal smoothing state as explicitly selected by the user. All outputs
stay private. Calendar slots are not compressed or prices filled across gaps.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import time

import duckdb
import numpy as np
import pandas as pd

from network_paper_features import FeatureParameters, historical_clip, normalized_macds, weighted_volatility

HORIZONS = ("1h", "1d", "3d", "5d", "10d")
MODEL_FEATURES = ["nm_scaled_"+h for h in HORIZONS]+["nm_macd_1","nm_macd_2","nm_macd_3"]
RATIO = 780/252
PARAMETERS = FeatureParameters(volatility_span=60*RATIO,
    macd_scales=tuple((s*RATIO,l*RATIO) for s,l in ((8,24),(16,48),(32,96))),
    price_std_window=195,macd_std_window=780,clipping_halflife=10.,
    rolling_min_periods=2)


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value,indent=2,default=str,allow_nan=False)+"\n")


def build(project, batch_size=16):
    project=Path(project).resolve()
    private=(project/"data").resolve()
    output=private/"features/05_network_individual_8_v1"
    original=private/"features/02_momentum_total_v1"
    if (output/"MANIFEST.json").exists():
        m=json.loads((output/"MANIFEST.json").read_text())
        if m["builder_sha256"]!=fingerprint(__file__) or m["math_sha256"]!=fingerprint(Path(__file__).with_name("network_paper_features.py")):
            raise FileExistsError("Existing feature release uses different code.")
        print("Reusing completed eight-feature release.",flush=True)
        return output
    output.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    import build_momentum_features as old
    manifest=json.loads((original/"MANIFEST.json").read_text())
    sessions=old.session_calendar(manifest["scope_start"],manifest["scope_end"])
    day_index=pd.DatetimeIndex(sessions.signal_date)
    counts=sessions.expected_slots.to_numpy(dtype=int)
    closes=np.cumsum(counts)-1
    ticks=pd.DatetimeIndex(np.concatenate([pd.date_range(s.market_open_utc,
        s.market_close_utc,freq="5min",inclusive="left").to_numpy()
        for s in sessions.itertuples(index=False)]))
    if ticks.tz is None:
        ticks=ticks.tz_localize("UTC")
    lengths=pd.DataFrame(index=day_index)
    lengths["1h"]=12.
    total=pd.Series(np.cumsum(counts),index=day_index,dtype=float)
    for h in (1,3,5,10):
        lengths[f"{h}d"]=total-total.shift(h)
    con=duckdb.connect()
    con.execute("SET threads=2")
    con.execute("SET memory_limit='512MB'")
    con.execute("SET TimeZone='UTC'")
    con.execute("SET temp_directory=?",[str(output/".duckdb_tmp")])
    con.read_parquet(str(original/"features.parquet")).create_view("old_features")
    ledger=con.execute("""SELECT security_id,signal_date,prediction_eligible,
      ret_1h,ret_1d,ret_3d,ret_5d,ret_10d FROM old_features
      ORDER BY security_id,signal_date""").df()
    anchors=pd.read_parquet(original/"daily_anchors.parquet")
    ids=sorted(ledger.loc[ledger.prediction_eligible,"security_id"].unique())
    paths=sorted((private/"sp500_5min_20261103/adjustment=all").glob("month=*/*.parquet"))
    con.read_parquet([str(p) for p in paths],hive_partitioning=False).create_view("bars")
    con.register("sessions",sessions)
    pieces=[]
    for start in range(0,len(ids),batch_size):
        selected=ids[start:start+batch_size]
        con.register("batch_ids",pd.DataFrame({"security_id":selected}))
        frame=con.execute("""WITH r AS (
          SELECT b.security_id,try_cast(b.bar_timestamp_utc AS TIMESTAMPTZ) ts,
            b.close,b.source,
            coalesce(isfinite(b.open) AND isfinite(b.high) AND isfinite(b.low)
              AND isfinite(b.close) AND b.low>0 AND b.high>=b.low
              AND b.open BETWEEN b.low AND b.high AND b.close BETWEEN b.low AND b.high
              AND try_cast(b.available_at_utc AS TIMESTAMPTZ)=ts+INTERVAL '5 minutes'
              AND b.minute_bars BETWEEN 1 AND 5 AND epoch(ts)::BIGINT%300=0,false) valid
          FROM bars b JOIN batch_ids USING(security_id)
          JOIN sessions s ON CAST(b.trading_date AS DATE)=s.signal_date
          WHERE ts>=s.market_open_utc AND ts<s.market_close_utc)
          SELECT security_id,ts,
            CASE WHEN count(*)=1 AND bool_and(valid) THEN max(close) END close,
            CASE WHEN count(*)=1 THEN max(source) END source
          FROM r GROUP BY 1,2 ORDER BY 1,2""").df()
        for security, rows in frame.groupby("security_id",sort=False):
            observed=rows.set_index("ts").reindex(ticks)
            price=observed.close
            # Forward propagation of a provenance tag only; no price is filled.
            tag=observed.source.ffill()
            changed=tag.notna() & (tag.ne(tag.shift()).fillna(False) | tag.shift().isna())
            segment=changed.cumsum()
            bar_stats=pd.DataFrame(index=ticks,columns=["sigma"]+MODEL_FEATURES[5:],dtype=float)
            for _, block in observed.groupby(segment,sort=False):
                if block.close.notna().sum()<2:
                    continue
                p=block.close
                one=p/p.shift(1)-1
                bar_stats.loc[block.index,"sigma"]=weighted_volatility(one,PARAMETERS.volatility_span)
                bar_stats.loc[block.index,MODEL_FEATURES[5:]]=normalized_macds(p,PARAMETERS)
            daily=bar_stats.iloc[closes].copy()
            daily.index=day_index
            a=anchors[anchors.security_id.eq(security)].set_index("signal_date").reindex(day_index)
            returns=pd.DataFrame(index=day_index)
            returns["1h"]=a.session_close/a.hour_start_close-1
            for h in (1,3,5,10):
                returns[f"{h}d"]=a.session_close/a.session_close.shift(h)-1
            raw=returns.div(np.sqrt(lengths).mul(daily.sigma.where(daily.sigma.gt(0)),axis=0))
            raw.columns=MODEL_FEATURES[:5]
            raw=pd.concat([raw,daily[MODEL_FEATURES[5:]]],axis=1)
            # Clipping is evaluated on daily signals, hence half-life 10 sessions.
            day_segments=segment.iloc[closes].copy()
            day_segments.index=day_index
            clipped=pd.DataFrame(index=day_index,columns=MODEL_FEATURES,dtype=float)
            for _, block in raw.groupby(day_segments,sort=False):
                clipped.loc[block.index]=historical_clip(block,PARAMETERS.clipping_halflife)
            own=ledger[ledger.security_id.eq(security)].set_index("signal_date")
            valid=own.index[own.prediction_eligible]
            for h in HORIZONS:
                if not np.allclose(returns.loc[valid,h],own.loc[valid,"ret_"+h],atol=1e-12,rtol=1e-12):
                    raise ValueError(f"Changed raw return horizon: {security}, {h}.")
            if not np.isfinite(clipped.loc[valid,MODEL_FEATURES].to_numpy()).all():
                failed=clipped.loc[valid].loc[lambda x:~np.isfinite(x).all(axis=1)]
                failed.to_parquet(output/f"undefined_{security}.parquet")
                raise ValueError(f"Undefined features on original eligible cohort: {security}; no rows silently dropped.")
            result=clipped.loc[valid].rename_axis("signal_date").reset_index()
            result.insert(0,"security_id",security)
            pieces.append(result)
        print(f"FEATURES {min(start+batch_size,len(ids))}/{len(ids)} securities; "
              f"{time.monotonic()-started:.1f}s",flush=True)
    calculated=pd.concat(pieces,ignore_index=True)
    if calculated.duplicated(["security_id","signal_date"]).any():
        raise ValueError("Duplicate feature keys.")
    con.register("calculated",calculated)
    columns=con.execute("DESCRIBE old_features").df().column_name.tolist()
    keep=[name for name in columns if name not in old.MODEL_FEATURES]
    con.execute("""CREATE TEMP TABLE new_features AS SELECT """+
        ",".join("o."+x for x in keep)+","+",".join("n."+x for x in MODEL_FEATURES)+
        " FROM old_features o LEFT JOIN calculated n USING(security_id,signal_date)")
    wrong=con.execute("""SELECT count(*) FROM new_features WHERE prediction_eligible
        AND NOT ("""+" AND ".join(f"coalesce(isfinite({x}),false)" for x in MODEL_FEATURES)+")").fetchone()[0]
    if wrong or len(calculated)!=int(ledger.prediction_eligible.sum()):
        raise ValueError("The feature change altered the eligible cohort.")
    con.execute("COPY (SELECT * FROM new_features ORDER BY signal_date,security_id) TO ? (FORMAT PARQUET,COMPRESSION ZSTD)",
                [str(output/"features.parquet")])
    summary=dict(rows=len(ledger),eligible_rows=len(calculated),securities=len(ids),
        original_eligible_rows=int(ledger.prediction_eligible.sum()),cohort_identical=True,
        elapsed_seconds=time.monotonic()-started,
        peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**3)
    write_json(output/"MANIFEST.json",dict(version="network_individual_8_shortscale_v1",
        created_at_utc=datetime.now(timezone.utc).isoformat(),model_features=MODEL_FEATURES,
        scope_start=manifest["scope_start"],scope_end=manifest["scope_end"],parameters=asdict(PARAMETERS),
        paper_period_to_five_minute_period_ratio=RATIO,summary=summary,
        original_manifest_sha256=fingerprint(original/"MANIFEST.json"),
        original_features_sha256=fingerprint(original/"features.parquet"),
        builder_sha256=fingerprint(__file__),math_sha256=fingerprint(Path(__file__).with_name("network_paper_features.py")),
        clock="Dense XNYS regular-session five-minute close observations; early closes respected. Overnight close-to-first-close move is one bar-clock return observation.",
        preprocessing="Raw horizon endpoints unchanged. Per-security provenance segments reset smoothing and clipping. Rolling dispersion uses available observations (min 2) within fixed 195/780-slot windows, without filling or compressing missing slots. No new numerical coverage threshold; original strict eligibility retained exactly.",
        clipping="Per-feature per-security daily EWM mean +/- 5 population standard deviations, half-life ten scheduled sessions, through current completed signal. No realized-return clipping.",
        usage_rights="internal_research_only",known_issues=manifest["known_issues"]+[
            "This is an intraday adaptation of individual features, not the graph or OLS strategy.",
            "Older causal EMA state is permitted; the two-week cap concerns return horizons, not every dependency.",
            "Five-minute volatility scaling treats heterogeneous overnight and intraday increments as bar-clock observations.",
            "Original overly restrictive eligibility is intentionally retained for the feature-only comparison."]))
    con.close()
    print(json.dumps(summary),flush=True)
    return output


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--batch-size",type=int,default=16)
    args=parser.parse_args()
    build(args.project,args.batch_size)
