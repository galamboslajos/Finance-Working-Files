"""Six causal stock-level volume features on a dense exchange-session clock."""
from __future__ import annotations

import numpy as np
import pandas as pd
from research_checks import require_dense_sessions

VOLUME_FEATURES = [
    "vol_relative_activity_10d",
    "vol_relative_last_hour_10d",
    "vol_activity_trend_3v7",
    "vol_signed_balance_1d",
    "vol_signed_balance_5d",
    "vol_close_vwap_gap",
]


def divide(numerator, denominator):
    """A missing/zero reference stays undefined, never replaced with zero."""
    return numerator.where(np.isfinite(numerator)).div(
        denominator.where(np.isfinite(denominator) & denominator.gt(0)))


def make_volume_features(daily, *, expected_sessions=None):
    """One security, one row per scheduled session, sorted without gaps removed.

    Expected fields: dollar_volume, last_hour_dollar_volume, signed_dollar_volume,
    share_volume, closing_price_raw, session_minutes, activity_valid, signed_valid.
    All refer exclusively to completed regular-session five-minute bars. Dollar
    volume is SUM(raw VWAP * raw shares), a bar-derived notional proxy.
    """
    if not daily.index.is_unique or not daily.index.is_monotonic_increasing:
        raise ValueError("Use a sorted unique dense exchange-session clock.")
    if expected_sessions is not None:
        require_dense_sessions(daily.index,expected_sessions)
    valid=daily.activity_valid.fillna(False).astype(bool)
    signed_valid=valid & daily.signed_valid.fillna(False).astype(bool)
    dollars=daily.dollar_volume.where(valid)
    hour=daily.last_hour_dollar_volume.where(valid)
    shares=daily.share_volume.where(valid)
    signed=daily.signed_dollar_volume.where(signed_valid)
    rate=divide(dollars,daily.session_minutes)
    # The fixed ten-session reference excludes the current observation. Missing
    # historical days are omitted within that window, not filled or replaced
    # with older dates. Its actual count is an audit field, not another input.
    prior_rate=rate.shift(1).rolling(10,min_periods=1)
    prior_hour=hour.shift(1).rolling(10,min_periods=1)
    result=pd.DataFrame(index=daily.index)
    result[VOLUME_FEATURES[0]]=divide(rate,prior_rate.median())
    result[VOLUME_FEATURES[1]]=divide(hour,prior_hour.median())
    recent=rate.rolling(3,min_periods=3).mean()
    preceding=rate.shift(3).rolling(7,min_periods=7).mean()
    result[VOLUME_FEATURES[2]]=divide(recent,preceding)
    result[VOLUME_FEATURES[3]]=divide(signed,dollars)
    result[VOLUME_FEATURES[4]]=divide(signed.rolling(5,min_periods=5).sum(),
        dollars.where(signed_valid).rolling(5,min_periods=5).sum())
    session_vwap=divide(dollars,shares)
    last=daily.closing_price_raw.where(np.isfinite(daily.closing_price_raw)
                                     & daily.closing_price_raw.gt(0))
    result[VOLUME_FEATURES[5]]=divide(last,session_vwap)-1
    result=result.where(np.isfinite(result))
    result["volume_reference_days"]=prior_rate.count()
    result["last_hour_reference_days"]=prior_hour.count()
    return result


def opening_gap(previous_close, opening_price, source_matched, unique_open=True):
    """Opening-price-only input. No opening-bar close/high/low/volume is used.

    Same-source adjusted endpoints prevent mixing retrieval adjustment levels.
    Unknown/incomparable gaps remain NaN; this does not remove a forecast row.
    The historical bar's actual delivery timestamp remains a separate audit field.
    """
    previous=previous_close.where(np.isfinite(previous_close) & previous_close.gt(0))
    current=opening_price.where(np.isfinite(opening_price) & opening_price.gt(0))
    return (divide(current,previous)-1).where(source_matched & unique_open)
