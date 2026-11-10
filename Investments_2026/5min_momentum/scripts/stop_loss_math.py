"""Fixed protective stops and exact past-only, one-dimensional calibration.

No estimator fitting, forecasts, price filling or broker orders. Distances are
in price units; returns are decimal fractions. The calibration objective is
the equally weighted N=2/N=3 mean daily net return, not classification accuracy.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

FAMILIES = ("atr", "gap", "hybrid")


def prior_context(daily, calendar, lookback=10):
    """Dense exchange sessions; ATR is mean prior ten valid true ranges.

    A true range requires that session and its immediate predecessor, on the
    same published adjustment/source basis. Current H/L/C and path validity
    never enter current ATR or gap. Full-session validity is for evaluation.
    """
    if daily.duplicated(["security_id", "trade_date"]).any():
        raise ValueError("Duplicate security/day context")
    dates = pd.DatetimeIndex(calendar.trade_date)
    if not dates.is_unique or not dates.is_monotonic_increasing:
        raise ValueError("Calendar must be dense, unique and ordered")
    pieces = []
    for sid, frame in daily.groupby("security_id", sort=False):
        g = frame.set_index("trade_date").reindex(dates).copy()
        valid = g.path_valid.fillna(False).astype(bool)
        same = g.source.notna() & g.source.eq(g.source.shift(1))
        previous = g.session_close.shift(1)
        tr = pd.concat([g.session_high-g.session_low,
                        (g.session_high-previous).abs(),
                        (g.session_low-previous).abs()], axis=1).max(axis=1, skipna=False)
        tr = tr.where(valid & valid.shift(1, fill_value=False) & same)
        g["atr_prior"] = tr.shift(1).rolling(lookback, min_periods=lookback).mean()
        # Check every historical source against the source known at the open,
        # without consulting later bars on the trade day.
        codes = pd.Series(pd.factorize(g.source)[0]+1, index=dates).where(valid, 0)
        lo = codes.shift(1).rolling(lookback+1, min_periods=lookback+1).min()
        hi = codes.shift(1).rolling(lookback+1, min_periods=lookback+1).max()
        opening_codes = g.opening_source.map({s:i+1 for i,s in enumerate(pd.factorize(g.source)[1])})
        history_ok = lo.gt(0) & lo.eq(hi) & hi.eq(opening_codes)
        g["atr_prior"] = g.atr_prior.where(history_ok)
        g["gap_price"] = (g.session_open-previous).where(
            valid.shift(1, fill_value=False) & g.opening_source.eq(g.source.shift(1)))
        g["previous_close"] = previous
        g["atr_reference_end"] = pd.Series(dates, index=dates).shift(1)
        g["security_id"] = sid
        pieces.append(g.rename_axis("trade_date").reset_index())
    return pd.concat(pieces, ignore_index=True)


def scale_price(family, atr, gap):
    if family == "atr": return float(atr)
    if family == "gap": return abs(float(gap))
    if family == "hybrid": return max(float(atr), abs(float(gap)))
    raise ValueError("Unknown stop family")


def replay(path, direction, distance):
    """First touch, active from opening entry. Worse bar open on a gap-through.

    Caller must fix a complete-path evaluation cohort independently of stops.
    A stop is not a guaranteed fill; touch prices assume no extra slippage.
    """
    o,h,l,c = [np.asarray(path[k], float) for k in ("open","high","low","close")]
    if direction not in (-1, 1) or not len(o) or not np.isfinite(np.r_[o,h,l,c]).all():
        raise ValueError("Invalid path/direction")
    if not (np.isinf(distance) or (np.isfinite(distance) and distance > 0)):
        raise ValueError("A fixed stop distance must be positive")
    entry = o[0]
    barrier = entry-direction*distance
    adverse = (entry-l)/entry if direction == 1 else (h-entry)/entry
    # Tolerance only reconciles algebraically identical price/ratio arithmetic;
    # it is far below the precision of the published prices.
    hit = adverse >= distance/entry-1e-14
    slots = np.flatnonzero(hit)
    if len(slots):
        slot = int(slots[0])
        fill = min(o[slot], barrier) if direction == 1 else max(o[slot], barrier)
        stopped = True
    else:
        slot, fill, stopped = len(o)-1, c[-1], False
    raw_return = fill/entry-1
    return dict(raw_return=float(raw_return), directional_return=float(direction*raw_return),
                exit_slot=slot, stopped=stopped, fill_price=float(fill),
                gap_through=bool(stopped and direction*(fill-barrier)<0))


def net_contribution(raw_return, weight, cost_bps=1):
    return weight*raw_return-abs(weight)*(2+raw_return)*cost_bps/10000


def response_segments(path, direction, scale_fraction, weight, cost_bps=1):
    """Piecewise-affine net P&L in multiplier k, with intervals (left,right].

    Record adverse extremes determine first-touch intervals. Splitting each
    interval at its bar open handles gap-through prices exactly. The last
    interval is the unchanged closing exit. No arbitrary threshold grid.
    """
    if not np.isfinite(scale_fraction) or scale_fraction <= 0:
        raise ValueError("Undefined stop scale")
    if np.sign(weight) != direction:
        raise ValueError("Weight and direction disagree")
    o,h,l,c = [np.asarray(path[k], float) for k in ("open","high","low","close")]
    p = o[0]; adverse = (p-l)/p if direction == 1 else (h-p)/p
    opens = -direction*(o/p-1)
    fee = cost_bps/10000; factor = abs(weight)*(-1+fee*direction)
    base = -2*fee*abs(weight)
    boundaries, slopes, intercepts = [], [], []
    old = 0.
    for b, opening in zip(adverse, opens):
        if b <= old: continue
        if opening > old:
            boundaries.append(min(opening,b)/scale_fraction)
            slopes.append(0.); intercepts.append(base+factor*opening)
        if b > max(old,opening):
            boundaries.append(b/scale_fraction)
            slopes.append(factor*scale_fraction); intercepts.append(base)
        old = b
    terminal = net_contribution(c[-1]/p-1, weight, cost_bps)
    return np.asarray(boundaries), np.asarray(slopes+[0.]), np.asarray(intercepts+[terminal])


def calibrate(segments):
    """Maximize summed net returns at every positive empirical breakpoint.

    Each supplied position weight already includes its N-mixture and equal-day
    normalization. No-stop is an explicit candidate and wins numerical ties.
    Return k=inf when no protective stop improves the past calibration book.
    """
    if not segments: raise ValueError("Empty calibration sample")
    starts_m = sum(s[1][0] for s in segments)
    starts_b = sum(s[2][0] for s in segments)
    baseline = sum(s[2][-1] for s in segments)
    xs, dm, db = [], [], []
    for boundaries, m, b in segments:
        if len(boundaries):
            xs.extend(boundaries); dm.extend(np.diff(m)); db.extend(np.diff(b))
    if not xs: return dict(multiplier=np.inf, score=baseline, baseline=baseline, breakpoints=0)
    order = np.argsort(xs, kind="stable")
    x = np.asarray(xs)[order]; dm = np.asarray(dm)[order]; db = np.asarray(db)[order]
    unique, first = np.unique(x, return_index=True)
    delta_m, delta_b = np.add.reduceat(dm,first), np.add.reduceat(db,first)
    after_m = starts_m+np.cumsum(delta_m)
    after_b = starts_b+np.cumsum(delta_b)
    before_m = np.r_[starts_m,after_m[:-1]]
    before_b = np.r_[starts_b,after_b[:-1]]
    # Positive observed distances only; no zero-width stops or machine-epsilon
    # thresholds designed to sit just beyond a historical adverse extreme.
    candidates = unique
    scores = before_m*unique+before_b
    best_score = float(scores.max())
    tolerance = 1e-12
    if best_score <= baseline+tolerance:
        return dict(multiplier=np.inf, score=baseline, baseline=baseline, breakpoints=len(unique))
    eligible = scores >= best_score-tolerance
    k = float(candidates[eligible].max())
    # Return the score of the chosen wider-distance tie, not another candidate.
    score = float(scores[np.flatnonzero(candidates==k)[-1]])
    return dict(multiplier=k, score=score, baseline=baseline, breakpoints=len(unique))


def select_past(frame, quarter_start):
    start = pd.Timestamp(quarter_start)
    return frame[frame.trade_date.ge(start-pd.DateOffset(months=12)) & frame.trade_date.lt(start)]
