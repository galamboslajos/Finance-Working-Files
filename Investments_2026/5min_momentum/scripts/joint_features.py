"""Join immutable momentum blocks without recomputation, filling or row filtering."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd

from build_momentum_features import MODEL_FEATURES as ORIGINAL_FEATURES
from build_volume_overnight_features import MODEL_FEATURES as RECENT_FEATURES

MODEL_FEATURES=list(dict.fromkeys(ORIGINAL_FEATURES+RECENT_FEATURES))
DIRECTORY="09_joint_momentum_volume_gap_v1"


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b""):h.update(chunk)
    return h.hexdigest()


def exact_duplicate_columns(frame,names,dtype=np.float32):
    """First occurrence wins; NaN masks and signed zero are normalized for hashing.

    Correlated columns and different mathematical normalizations are retained.
    No outcomes, fitted correlations or held-out performance choose features.
    """
    retained,duplicates,buckets=[],[],{}
    for name in names:
        values=frame[name].to_numpy(dtype=dtype).copy()
        if np.isinf(values).any():raise ValueError("Infinite input: "+name)
        values[np.isnan(values)]=np.nan
        values[values==0]=0
        digest=hashlib.sha256(values.tobytes()).hexdigest()
        match=None
        for previous in buckets.get(digest,[]):
            if np.array_equal(values,frame[previous].to_numpy(dtype=dtype),equal_nan=True):
                match=previous;break
        if match is None:
            retained.append(name);buckets.setdefault(digest,[]).append(name)
        else:duplicates.append(dict(removed=name,retained=match,definition="identical across all rows at model float32 precision"))
    return retained,duplicates


def build(project):
    project=Path(project).resolve();private=(project/"data").resolve()
    out=private/"features"/DIRECTORY
    old=private/"models/04_xgb_ret_ensemble100_v1"
    recent=private/"models/07_xgb_ret_volume_overnight100_v1"
    hashes={"original_panel_sha256":sha(old/"supervised_panel.parquet"),
            "recent_panel_sha256":sha(recent/"supervised_panel.parquet"),
            "builder_sha256":sha(__file__)}
    if (out/"MANIFEST.json").exists():
        saved=json.loads((out/"MANIFEST.json").read_text())
        if saved["input_hashes"]!=hashes:raise FileExistsError("Different inputs/code require a new joint release.")
        if sha(out/"supervised_panel.parquet")!=saved["output_sha256"]:raise ValueError("Joint panel changed.")
        return out
    out.mkdir(parents=True,exist_ok=True)
    con=duckdb.connect();con.execute("SET threads=6");con.execute("SET TimeZone='UTC'")
    con.read_parquet(str(old/"supervised_panel.parquet")).create_view("original")
    con.read_parquet(str(recent/"supervised_panel.parquet")).create_view("recent")
    for relation in ["original","recent"]:
        n=con.execute(f"SELECT count(*) FROM (SELECT security_id,signal_date,trade_date FROM {relation} GROUP BY ALL HAVING count(*)!=1)").fetchall()[0][0]
        if n:raise ValueError("Duplicate panel keys: "+relation)
    keys="security_id,signal_date,trade_date"
    stable="security_id,signal_date,trade_date,ticker,session_open,session_close,outcome_status,future_oc_return,realized_class"
    difference=con.execute(f"WITH d AS ((SELECT {stable} FROM original EXCEPT ALL SELECT {stable} FROM recent) UNION ALL (SELECT {stable} FROM recent EXCEPT ALL SELECT {stable} FROM original)) SELECT count(*) FROM d").fetchall()[0][0]
    if difference:raise ValueError("Cohort, labels or target outcomes differ between source blocks.")
    late=con.execute(f"SELECT count(*) FROM original o JOIN recent r USING({keys}) WHERE o.signal_cutoff_utc IS DISTINCT FROM r.history_cutoff_utc").fetchall()[0][0]
    if late:raise ValueError("Original features have a different historical information cutoff.")
    original_columns=json.loads((old/"CONFIG.json").read_text())["model_features"]
    if original_columns!=ORIGINAL_FEATURES:raise ValueError("Unexpected original feature contract.")
    recent_columns=json.loads((recent/"CONFIG.json").read_text())["model_features"]
    if recent_columns!=RECENT_FEATURES:raise ValueError("Unexpected recent feature contract.")
    append=[name for name in ORIGINAL_FEATURES if name not in RECENT_FEATURES]
    frame=con.execute(f"SELECT r.*,{','.join('o.'+x for x in append)} FROM recent r JOIN original o USING({keys}) ORDER BY r.trade_date,r.security_id").df()
    if len(frame)!=con.execute("SELECT count(*) FROM recent").fetchall()[0][0]:raise ValueError("Join changed row count.")
    retained,duplicates=exact_duplicate_columns(frame,MODEL_FEATURES)
    for name in retained:
        if name!="overnight_gap" and not np.isfinite(frame[name]).all():raise ValueError("Missing original/historical feature: "+name)
    # Both sources are copied, not recomputed using a changed cohort or transform.
    for relation,names in [("original",ORIGINAL_FEATURES),("recent",RECENT_FEATURES)]:
        con.register("joint",frame)
        fields=keys+","+",".join(names)
        count=con.execute(f"SELECT count(*) FROM ((SELECT {fields} FROM joint EXCEPT ALL SELECT {fields} FROM {relation}) UNION ALL (SELECT {fields} FROM {relation} EXCEPT ALL SELECT {fields} FROM joint))").fetchall()[0][0]
        if count:raise ValueError("A source feature value changed.")
    metadata=[name for name in frame if name not in MODEL_FEATURES]
    panel=frame[metadata+retained]
    path=out/"supervised_panel.parquet";panel.to_parquet(path,index=False,compression="zstd")
    result=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),model_features=retained,
        candidate_features=MODEL_FEATURES,input_hashes=hashes,output_sha256=sha(path),
        summary=dict(rows=len(frame),features=len(retained),candidate_features=len(MODEL_FEATURES),
            original_features=len(ORIGINAL_FEATURES),network_features=8,volume_features=6,opening_gap_features=1,
            exact_duplicate_features_removed=duplicates,cohort_labels_outcomes_identical=True,
            source_features_bitwise_preserved=True),
        duplication_policy="Remove duplicate names and columns identical at float32 model precision. Retain distinct transformations; no correlation-based or outcome-based selection.",
        information_timing="29 previous-close inputs plus one current opening-gap input; optimistic same-open information and fill.",
        inherited_limits="Same eligibility, native missing gap, partial historical volume references, nonbinding clipping, retrospective membership/vintages and reused development history as completed source experiments.",
        transformations_changed=False,one_minute_data_used=False,usage_rights="internal_research_only")
    (out/"MANIFEST.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    con.close();print(json.dumps(result["summary"],indent=2),flush=True)
    return out


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project",type=Path,default=Path(__file__).resolve().parents[1])
    build(parser.parse_args().project)
