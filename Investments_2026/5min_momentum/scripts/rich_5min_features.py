"""Causal regular-session path features, added to the immutable joint30 panel.

No download, fitting, price filling, eligibility change or one-minute access.
Undefined ratios stay NaN. References use scheduled sessions, never observed rows.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import warnings

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from build_momentum_features import session_calendar
from joint_features import MODEL_FEATURES as JOINT_FEATURES, sha, exact_duplicate_columns

PATH_FEATURES = [
    "path_rv_relative_10d", "path_downside_variance_share",
    "path_signed_efficiency", "path_late_variance_concentration",
    "path_late_vs_early_return", "path_max_drawdown_scaled",
    "path_vwap_above_fraction", "path_return_autocorr_1",
    "path_extrema_time_balance", "path_return_volume_alignment",
    "path_late_relative_activity", "path_signed_late_relative_activity",
]
MODEL_FEATURES = JOINT_FEATURES + PATH_FEATURES
DIRECTORY = "10_rich_5min_path_v1"
MAX_SLOTS = 78
REFERENCE_DAYS = 10


def ratio(a, b):
    a, b = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float))
    return np.divide(a, b, out=np.full(a.shape, np.nan), where=np.isfinite(b) & (b > 0))


def correlation(a, b, mask):
    """All scheduled pairs required; zero dispersion is undefined, not zero."""
    count = mask.sum(axis=1)
    x = np.where(mask, a, 0.0)
    y = np.where(mask, b, 0.0)
    mx, my = ratio(x.sum(axis=1), count), ratio(y.sum(axis=1), count)
    dx, dy = np.where(mask, a-mx[:, None], 0), np.where(mask, b-my[:, None], 0)
    result = ratio((dx*dy).sum(axis=1), np.sqrt((dx*dx).sum(axis=1)*(dy*dy).sum(axis=1)))
    return np.clip(result, -1, 1)


def calculate_grid(grid, expected_slots):
    """One security on a dense session × 78-slot grid, including missing days.

    Returns features and explicit availability diagnostics for every grid day.
    Input source codes are nonzero only when a source tag is actually present.
    Structural after-close cells are excluded; scheduled NaNs are never skipped.
    """
    n = np.asarray(expected_slots, int)
    if (n < 24).any() or (n > MAX_SLOTS).any():
        raise ValueError("Unsupported exchange-session length.")
    days = len(n)
    slot = np.arange(MAX_SLOTS)[None, :]
    scheduled = slot < n[:, None]
    late = scheduled & (slot >= n[:, None]-12)
    early = scheduled & (slot < 12)
    o, h, l, c, v, w = [np.asarray(grid[k], float) for k in ("open", "high", "low", "close", "volume", "vwap")]
    src, count = grid["source"], grid["count"]
    valid_price = (np.isfinite(o) & np.isfinite(h) & np.isfinite(l) & np.isfinite(c)
        & (l > 0) & (h >= l) & (o >= l) & (o <= h) & (c >= l) & (c <= h)
        & grid["time_valid"] & (count == 1) & (src > 0))
    homogeneous = ((src == src[:, :1]) | ~scheduled).all(axis=1) & (src[:, 0] > 0)
    price_ok = (valid_price | ~scheduled).all(axis=1) & homogeneous
    quantity_ok = price_ok & ((np.isfinite(v) & (v >= 0) & np.isfinite(w) & (w > 0)
        & (w >= l) & (w <= h)) | ~scheduled).all(axis=1)
    # First increment excludes the overnight move; subsequent increments use
    # adjacent scheduled closes. A missing slot cannot shorten this sequence.
    previous = np.concatenate([o[:, :1], c[:, :-1]], axis=1)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        r = np.log(c/previous)
    r = np.where(scheduled, r, 0)
    rv2 = (r*r).sum(axis=1)
    rv = np.sqrt(rv2)
    rms = np.where(price_ok, np.sqrt(rv2/n), np.nan)
    out = {}
    out["path_downside_variance_share"] = ratio((np.minimum(r, 0)**2).sum(axis=1), rv2)
    out["path_signed_efficiency"] = ratio(r.sum(axis=1), np.abs(r).sum(axis=1))
    out["path_late_variance_concentration"] = ratio(np.where(late, r*r, 0).sum(axis=1), rv2)*n/12
    out["path_late_vs_early_return"] = ratio(np.where(late, r, 0).sum(axis=1)-np.where(early, r, 0).sum(axis=1), rv)
    peaks = np.maximum.accumulate(np.concatenate([o[:, :1], c], axis=1), axis=1)[:, 1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        drawdown = np.log(peaks/c)
    out["path_max_drawdown_scaled"] = ratio(np.where(scheduled, drawdown, 0).max(axis=1), rv)
    out["path_return_autocorr_1"] = correlation(r[:, :-1], r[:, 1:], scheduled[:, 1:])
    max_h = np.where(scheduled, h, -np.inf).max(axis=1)
    min_l = np.where(scheduled, l, np.inf).min(axis=1)
    at_h, at_l = scheduled & (h == max_h[:, None]), scheduled & (l == min_l[:, None])
    out["path_extrema_time_balance"] = (ratio((at_h*slot).sum(axis=1), at_h.sum(axis=1))
        - ratio((at_l*slot).sum(axis=1), at_l.sum(axis=1)))/(n-1)
    dollars = np.where(scheduled & quantity_ok[:, None], w*v, np.nan)
    cum_shares = np.cumsum(np.where(scheduled, v, 0), axis=1)
    cum_dollars = np.cumsum(np.where(scheduled, dollars, 0), axis=1)
    cumulative_vwap = ratio(cum_dollars, cum_shares)
    prefix_ok = ((np.isfinite(cumulative_vwap) | ~scheduled).all(axis=1)) & quantity_ok
    out["path_vwap_above_fraction"] = np.where(prefix_ok,
        ((c > cumulative_vwap) & scheduled).sum(axis=1)/n, np.nan)

    # Shape: current day × slot × prior ten calendar days. The current day is
    # outside its own window. Structural early-close slots have no observation.
    q = np.full_like(dollars, np.nan)
    rms_ref = np.full(days, np.nan)
    profile_ok = np.zeros(days, bool)
    if days > REFERENCE_DAYS:
        from numpy.lib.stride_tricks import sliding_window_view
        old_d = sliding_window_view(dollars, REFERENCE_DAYS, axis=0)[:-1]
        old_src = sliding_window_view(np.where(scheduled & quantity_ok[:, None], src, 0), REFERENCE_DAYS, axis=0)[:-1]
        old_schedule = sliding_window_view(scheduled, REFERENCE_DAYS, axis=0)[:-1]
        expected = old_schedule.sum(axis=2)
        observed = np.isfinite(old_d).sum(axis=2)
        same_source = ((old_src == src[REFERENCE_DAYS:, :, None]) | ~old_schedule).all(axis=2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(old_d, axis=2)
        good = (expected > 0) & (observed == expected) & same_source
        q[REFERENCE_DAYS:] = np.where(good, ratio(dollars[REFERENCE_DAYS:], med), np.nan)
        profile_ok[REFERENCE_DAYS:] = ((np.isfinite(q[REFERENCE_DAYS:]) | ~scheduled[REFERENCE_DAYS:]).all(axis=1)
            & quantity_ok[REFERENCE_DAYS:])
        old_rms = sliding_window_view(rms, REFERENCE_DAYS)[:-1]
        old_day_src = sliding_window_view(np.where(price_ok, src[:, 0], 0), REFERENCE_DAYS)[:-1]
        complete = np.isfinite(old_rms).all(axis=1) & (old_day_src == src[REFERENCE_DAYS:, :1]).all(axis=1)
        rms_ref[REFERENCE_DAYS:] = np.where(complete, np.median(old_rms, axis=1), np.nan)
    out["path_rv_relative_10d"] = ratio(rms, rms_ref)
    qsum = np.where(scheduled, q, 0).sum(axis=1)
    qlate = np.where(late, q, 0).sum(axis=1)
    out["path_late_relative_activity"] = np.where(profile_ok, ratio(qlate, qsum)*n/12, np.nan)
    out["path_signed_late_relative_activity"] = np.where(profile_ok,
        ratio(np.where(late, np.sign(r)*q, 0).sum(axis=1), qlate), np.nan)
    out["path_return_volume_alignment"] = np.where(profile_ok, correlation(r, np.log1p(q), scheduled), np.nan)
    for name in PATH_FEATURES:
        out[name] = np.where(price_ok, out[name], np.nan)
        if np.isinf(out[name]).any():
            raise ValueError("Infinite constructed input: "+name)
    out.update(path_price_grid_valid=price_ok, path_quantity_grid_valid=quantity_ok,
        path_profile_reference_valid=profile_ok, path_rms_reference_valid=np.isfinite(rms_ref) & (rms_ref > 0),
        path_scheduled_missing_slots=((count == 0) & scheduled).sum(axis=1),
        path_duplicate_slots=((count > 1) & scheduled).sum(axis=1),
        path_invalid_slots=((~valid_price) & (count > 0) & scheduled).sum(axis=1),
        path_partial_minute_buckets=((grid["minute_bars"] < 5) & (count == 1) & scheduled).sum(axis=1))
    return pd.DataFrame(out)


def security_grid(frame, calendar):
    """Map timestamps to dense cells without filling absent observations."""
    shape = (len(calendar), MAX_SLOTS)
    grid = {k: np.full(shape, np.nan) for k in ("open", "high", "low", "close", "volume", "vwap", "minute_bars")}
    grid.update(source=np.zeros(shape, int), count=np.zeros(shape, int), time_valid=np.zeros(shape, bool))
    if not frame.empty:
        row = pd.Index(calendar.session_num).get_indexer(frame.session_num)
        if (row < 0).any():
            raise ValueError("Bars outside the supplied dense calendar.")
        col = frame.slot.to_numpy(int)
        if (col < 0).any() or (col >= MAX_SLOTS).any():
            raise ValueError("Invalid slot index.")
        np.add.at(grid["count"], (row, col), 1)
        for name in grid:
            if name != "count":
                grid[name][row, col] = frame[name].to_numpy()
    return grid


def read_month(con, path):
    con.from_parquet(str(path), hive_partitioning=False).create_view("bars", replace=True)
    return con.execute("""
        WITH typed AS (SELECT security_id,try_cast(trading_date AS DATE) signal_date,
          try_cast(bar_timestamp_utc AS TIMESTAMPTZ) ts,
          try_cast(available_at_utc AS TIMESTAMPTZ) available_at,
          open,high,low,close,volume,vwap,minute_bars,source FROM bars)
        SELECT b.security_id,s.session_num,
          floor(epoch(ts-s.market_open_utc)/300)::INTEGER slot,
          b.open,b.high,b.low,b.close,b.volume,b.vwap,b.minute_bars,b.source,
          coalesce(epoch(ts-s.market_open_utc)::BIGINT % 300=0
            AND ts=date_trunc('second',ts)
            AND available_at=ts+INTERVAL '5 minutes'
            AND minute_bars BETWEEN 1 AND 5,false) time_valid
        FROM typed b JOIN sessions s USING(signal_date)
        WHERE ts>=s.market_open_utc AND ts<s.market_close_utc
        ORDER BY security_id,s.session_num,slot
    """).df()


def build(project):
    project = Path(project).resolve()
    private = (project/"data").resolve()
    out = private/"features"/DIRECTORY
    baseline = private/"features/09_joint_momentum_volume_gap_v1"
    raw = private/"sp500_5min_20261103"
    inputs = dict(builder_sha256=sha(__file__), calendar_code_sha256=sha(Path(__file__).with_name("build_momentum_features.py")),
        baseline_manifest_sha256=sha(baseline/"MANIFEST.json"), baseline_panel_sha256=sha(baseline/"supervised_panel.parquet"),
        raw_manifest_sha256=sha(raw/"MANIFEST.json"))
    if (out/"MANIFEST.json").exists():
        saved = json.loads((out/"MANIFEST.json").read_text())
        if inputs != saved["input_hashes"]:
            raise FileExistsError("New code/inputs require a new feature version.")
        for name, digest in saved["output_sha256"].items():
            if sha(out/name) != digest:
                raise ValueError("Completed path release changed: "+name)
        return out
    if out.exists() and list(out.iterdir()):
        raise FileExistsError("Unfinished release exists; inspect it before retrying.")
    out.mkdir(parents=True, exist_ok=True)
    original = pd.read_parquet(baseline/"supervised_panel.parquet", columns=["security_id", "signal_date", "history_cutoff_utc"])
    if original.duplicated(["security_id", "signal_date"]).any():
        raise ValueError("Duplicate baseline rows.")
    original["month"] = original.signal_date.dt.strftime("%Y-%m")
    calendar = session_calendar("2016-01-04", original.signal_date.max())
    calendar = calendar.set_index("session_num", drop=False)
    calendar.to_parquet(out/"session_calendar.parquet", index=False)
    manifest = json.loads((raw/"MANIFEST.json").read_text())
    files = [x for x in manifest["objects"] if "/adjustment=raw/" in x["path"]]
    if not manifest["complete"] or len(files) != 130:
        raise ValueError("Expected the completed 130-month raw five-minute release.")
    files = sorted(files, key=lambda x: x["path"])
    con = duckdb.connect(); con.execute("SET threads=4"); con.execute("SET TimeZone='UTC'")
    con.register("sessions", calendar)
    carry = pd.DataFrame()
    writer = None
    partial = out/"daily_path_features.building.parquet"
    source_codes = {}
    inventory = []
    try:
        for number, obj in enumerate(files, 1):
            path = Path(obj["path"])
            if not path.is_relative_to(raw) or sha(path) != obj["sha256"].removeprefix("sha256:"):
                raise ValueError("Five-minute input fingerprint mismatch: "+str(path))
            month = path.parent.name.removeprefix("month=")
            cal = calendar[calendar.signal_date.dt.strftime("%Y-%m") == month]
            if cal.empty:
                continue
            current = read_month(con, path)
            for source in current.source.dropna().unique():
                if source and source not in source_codes:
                    source_codes[source] = len(source_codes)+1
            current["source"] = current.source.map(source_codes).fillna(0).astype(int)
            bars = pd.concat([carry, current], ignore_index=True) if not carry.empty else current
            lo, hi = int(cal.session_num.min()), int(cal.session_num.max())
            dense = calendar.loc[max(0, lo-REFERENCE_DAYS):hi]
            # Missing whole days remain represented in dense, including securities
            # not observed in this month. Older-than-ten-day rows never substitute.
            wanted = original[original.month == month]
            groups = {str(s): f for s, f in bars.groupby("security_id", sort=False)}
            records = []
            date_to_row = pd.Index(dense.signal_date)
            for security, target in wanted.groupby("security_id", sort=False):
                frame = groups.get(str(security), bars.iloc[:0])
                values = calculate_grid(security_grid(frame, dense), dense.expected_slots.to_numpy())
                indices = date_to_row.get_indexer(target.signal_date)
                if (indices < 0).any():
                    raise ValueError("Baseline signal date outside exchange calendar.")
                selected = values.iloc[indices].reset_index(drop=True)
                selected.insert(0, "signal_date", target.signal_date.to_numpy())
                selected.insert(0, "security_id", target.security_id.to_numpy())
                selected["path_available_at_utc"] = pd.to_datetime(dense.iloc[indices].market_close_utc.to_numpy(), utc=True).astype("datetime64[us, UTC]")
                records.append(selected)
            if records:
                table = pa.Table.from_pandas(pd.concat(records, ignore_index=True), preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(partial, table.schema, compression="zstd")
                writer.write_table(table)
            carry = current[current.session_num > hi-REFERENCE_DAYS].copy()
            inventory.append(dict(path=str(path), sha256=obj["sha256"], rows_regular=len(current)))
            if number % 10 == 0 or number == 1:
                print(f"PATH FEATURES {number}/130 months", flush=True)
    finally:
        if writer is not None:
            writer.close()
        con.close()
    partial.rename(out/"daily_path_features.parquet")
    # Exact preservation test uses every baseline column, including labels,
    # unknown outcomes, opening clocks, and all thirty historical inputs.
    con = duckdb.connect(); con.execute("SET threads=4")
    con.read_parquet(str(baseline/"supervised_panel.parquet")).create_view("old")
    con.read_parquet(str(out/"daily_path_features.parquet")).create_view("paths")
    con.execute("CREATE TEMP TABLE panel AS SELECT o.*,p.* EXCLUDE(security_id,signal_date) FROM old o LEFT JOIN paths p USING(security_id,signal_date)")
    oldcols = pq.read_schema(baseline/"supervised_panel.parquet").names
    names = ",".join(oldcols)
    differences = con.execute(f"SELECT count(*) FROM ((SELECT {names} FROM old EXCEPT ALL SELECT {names} FROM panel) UNION ALL (SELECT {names} FROM panel EXCEPT ALL SELECT {names} FROM old))").fetchone()[0]
    bad = con.execute("SELECT count(*) FROM panel WHERE path_available_at_utc IS NULL OR CAST(path_available_at_utc AS TIMESTAMPTZ) IS DISTINCT FROM history_cutoff_utc").fetchone()[0]
    if differences or bad or con.execute("SELECT count(*) FROM paths").fetchone()[0] != len(original):
        raise ValueError("Cohort/label/source preservation or causal timestamp check failed.")
    features = con.execute("SELECT "+",".join(MODEL_FEATURES)+" FROM panel").df()
    retained, duplicates = exact_duplicate_columns(features, MODEL_FEATURES)
    if duplicates or retained != MODEL_FEATURES:
        raise ValueError("Exact duplicate candidates need an explicit new experiment contract.")
    distributions = []
    for name in MODEL_FEATURES:
        x = features[name].to_numpy(float); good = np.isfinite(x)
        if np.isinf(x).any() or not good.any() or np.unique(x[good]).size < 2:
            raise ValueError("Invalid/constant model input: "+name)
        if name in JOINT_FEATURES and name != "overnight_gap" and not good.all():
            raise ValueError("Historical input became missing: "+name)
        distributions.append(dict(feature=name, finite=int(good.sum()), missing=int((~good).sum()),
            minimum=float(x[good].min()), median=float(np.median(x[good])), maximum=float(x[good].max())))
    pd.DataFrame(distributions).to_csv(out/"feature_distribution.csv", index=False)
    con.execute("COPY (SELECT * FROM panel ORDER BY trade_date,security_id) TO ? (FORMAT PARQUET,COMPRESSION ZSTD)", [str(out/"supervised_panel.parquet")])
    quality = con.execute("SELECT count(*) AS panel_rows,count_if(NOT path_price_grid_valid) invalid_price_days,count_if(NOT path_quantity_grid_valid) invalid_quantity_days,count_if(NOT path_profile_reference_valid) incomplete_profile_days,count_if(NOT path_rms_reference_valid) incomplete_rms_reference_days,count_if(path_partial_minute_buckets>0) days_with_partial_buckets FROM panel").df().iloc[0].to_dict()
    con.close()
    result = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), model_features=MODEL_FEATURES,
        new_features=PATH_FEATURES, input_hashes=inputs, verified_raw_objects=inventory, source_codes=source_codes,
        output_sha256={name:sha(out/name) for name in ("supervised_panel.parquet", "daily_path_features.parquet", "session_calendar.parquet", "feature_distribution.csv")},
        summary=dict(rows=len(original), features=len(MODEL_FEATURES), new_features=len(PATH_FEATURES),
            cohort_labels_outcomes_identical=True, source_features_bitwise_preserved=True,
            exact_duplicate_features_removed=duplicates, quality={k:int(v) for k,v in quality.items()}),
        reference_policy="Previous ten scheduled sessions, exclude current; source-matched. Afternoon slots absent on legitimate early closes are structural, not missing observations.",
        missing_policy="No new eligibility gate or price filling. Undefined new inputs remain native NaN. Exact per-row availability flags and NaN masks retained.",
        units="All added inputs dimensionless. Raw shares enter only VWAP and stock/slot-relative dollar activity; uniform price/share rescaling cancels.",
        information_timing="All twelve additions available by previous completed close. Existing optimistic same-opening gap/fill unchanged.",
        current_increment="log(first close/first open), then log(close/previous scheduled close); excludes overnight move.",
        historical_comparison="New frozen feature release; joint30 remains unchanged; fixed forecast cohort, labels, outcomes and original thirty inputs.",
        one_minute_data_used=False, usage_rights="internal_research_only")
    (out/"MANIFEST.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(result["summary"], indent=2), flush=True)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    build(parser.parse_args().project)
