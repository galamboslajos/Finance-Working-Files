"""Strict, reusable input/time checks; do not fit or alter frozen feature releases."""
from __future__ import annotations
import numpy as np
import pandas as pd


def require_dense_sessions(index, expected_sessions):
    """Reject a missing, shuffled or duplicated scheduled day before rolling."""
    index = pd.DatetimeIndex(index)
    expected = pd.DatetimeIndex(expected_sessions)
    if not index.is_unique or not index.is_monotonic_increasing or not index.equals(expected):
        raise ValueError("Use the exact dense exchange-session index; do not compress missing dates.")


def require_past_training(training, testing, test_start):
    boundary = pd.Timestamp(test_start)
    if training.empty or not training.trade_date.lt(boundary).all():
        raise ValueError("Training outcomes must precede the outer forecast block.")
    if not testing.trade_date.ge(boundary).all():
        raise ValueError("Outer test dates precede their declared block.")
    if not training.realized_class.isin(range(1,11)).all() or not np.isfinite(training.future_oc_return).all():
        raise ValueError("Supervised training requires observed finite labels and outcomes.")
    for frame in (training, testing):
        if frame.duplicated(["security_id", "trade_date"]).any():
            raise ValueError("Duplicated stock/date rows.")
        if not frame.signal_date.lt(frame.trade_date).all():
            raise ValueError("Signal-date history must precede the target session.")


def execution_timing_violations(frame):
    """An observed gap cannot be traded before its stored delivery timestamp.

    The research benchmark may deliberately violate this condition. Calling it
    execution-ready must fail rather than silently relabel that assumption.
    """
    used = frame.overnight_gap.notna()
    available = pd.to_datetime(frame.opening_bar_available_at_utc, utc=True)
    decision = pd.to_datetime(frame.signal_cutoff_utc, utc=True)
    return used & (available.isna() | decision.isna() | available.gt(decision))


def require_execution_information_timing(frame):
    """Necessary delivery-time check only; does not certify fills, prices or borrow."""
    if execution_timing_violations(frame).any():
        raise ValueError("Opening gap is delivered after the assumed decision/fill; research-only timing.")


def feature_quality(frame, features):
    """Report tails without deleting/winsorizing observations after seeing targets."""
    rows = []
    for name in features:
        x = frame[name].astype(float)
        finite = x[np.isfinite(x)]
        q = finite.quantile([0, .001, .01, .5, .99, .999, 1])
        rows.append(dict(feature=name, rows=len(x), missing=int(x.isna().sum()),
            infinite=int(np.isinf(x).sum()), minimum=q.loc[0], p001=q.loc[.001], p01=q.loc[.01],
            median=q.loc[.5], p99=q.loc[.99], p999=q.loc[.999], maximum=q.loc[1]))
    return pd.DataFrame(rows)
