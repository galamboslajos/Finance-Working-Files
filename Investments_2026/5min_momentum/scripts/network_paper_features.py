"""Eight individual momentum variables from Pu et al., section 2.2.

This module contains feature mathematics only. It does not choose a clock,
membership/eligibility rule, return horizon, label, estimator or portfolio.
Window units must match the supplied series; the caller must state adaptations.
No missing price is filled and no forward observation is used.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureParameters:
    """Defaults are the paper's daily units, not a proposed intraday mapping."""

    volatility_span: float = 60.0
    macd_scales: tuple[tuple[float, float], ...] = ((8., 24.), (16., 48.), (32., 96.))
    price_std_window: int = 63
    macd_std_window: int = 252
    clipping_halflife: float = 252.0
    rolling_ddof: int = 0  # Paper does not specify a rolling-std divisor.
    rolling_min_periods: int | None = None  # None requires the entire window.

    def validate(self):
        if self.volatility_span < 1 or self.clipping_halflife <= 0:
            raise ValueError("Invalid exponential decay parameter.")
        if len(self.macd_scales) != 3 or any(not 1 <= s < l for s, l in self.macd_scales):
            raise ValueError("Exactly three MACD pairs with 1 <= S < L are required.")
        if min(self.price_std_window, self.macd_std_window) < 2:
            raise ValueError("Rolling normalizers require at least two periods.")
        if self.rolling_ddof not in (0, 1):
            raise ValueError("Explicitly choose population or sample rolling dispersion.")
        if self.rolling_min_periods is not None and not (
                2 <= self.rolling_min_periods <= min(self.price_std_window,self.macd_std_window)):
            raise ValueError("Rolling min_periods must fit both windows.")


def weighted_volatility(unit_returns: pd.Series, span: float) -> pd.Series:
    """Appendix A.2: normalized exponential weights, population variance.

    pandas' default bias=False would add a correction absent from that equation.
    At least two finite observations are required; missing input stays missing.
    """
    if span < 1:
        raise ValueError("The EWM span must be at least one.")
    clean = unit_returns.where(np.isfinite(unit_returns))
    return clean.ewm(span=span, adjust=True, ignore_na=False,
                     min_periods=2).std(bias=True).where(clean.notna())


def normalized_macds(price: pd.Series, parameters: FeatureParameters) -> pd.DataFrame:
    """Equations 1--3, including both price and MACD dispersion divisors.

    EMA alpha is 1/J, which differs from pandas ewm(span=J)'s 2/(J+1).
    Full scheduled windows are the default. A caller may explicitly permit
    observed-value dispersion inside that fixed window; calendar gaps never
    shorten the window. The caller supplies a dense clock and eligibility ledger.
    """
    parameters.validate()
    clean = price.where(np.isfinite(price) & price.gt(0))
    price_sd = clean.rolling(parameters.price_std_window,
        min_periods=parameters.rolling_min_periods or parameters.price_std_window).std(ddof=parameters.rolling_ddof)
    result = {}
    for k, (short, long) in enumerate(parameters.macd_scales, 1):
        fast = clean.ewm(alpha=1/short, adjust=False, ignore_na=False).mean()
        slow = clean.ewm(alpha=1/long, adjust=False, ignore_na=False).mean()
        macd_norm = ((fast-slow)/price_sd.where(price_sd.gt(0))).where(clean.notna())
        macd_sd = macd_norm.rolling(parameters.macd_std_window,
            min_periods=parameters.rolling_min_periods or parameters.macd_std_window).std(ddof=parameters.rolling_ddof)
        result[f"nm_macd_{k}"] = (macd_norm/macd_sd.where(macd_sd.gt(0))).where(clean.notna())
    return pd.DataFrame(result, index=price.index)


def historical_clip(values: pd.DataFrame, halflife: float) -> pd.DataFrame:
    """Per-security, per-feature EWM mean +/- five EWM standard deviations.

    Bounds include the current completed observation, never a future value.
    Section 2.2 does not state the bias/adjust flags or a prior-only bound;
    normalized population weights are explicit implementation conventions here.
    Clipping changes feature values only, never portfolio realized returns.
    """
    if halflife <= 0:
        raise ValueError("Clipping half-life must be positive.")
    clean = values.where(np.isfinite(values))
    # With one observation the population variance is exactly zero, so the
    # bounds both equal that observation. It remains a valid unclipped feature.
    weights = clean.ewm(halflife=halflife, adjust=True, ignore_na=False, min_periods=1)
    mean, sd = weights.mean(), weights.std(bias=True)
    return clean.clip(lower=mean-5*sd, upper=mean+5*sd).where(mean.notna() & sd.notna())


def individual_features(price: pd.Series, unit_returns: pd.Series,
                        cumulative_returns: pd.DataFrame,
                        horizon_lengths: pd.DataFrame,
                        parameters: FeatureParameters,
                        *, apply_clipping: bool = True) -> pd.DataFrame:
    """Five R(h)/(sigma*sqrt(h)) values and three normalized MACDs.

    R(h) may be exact exchange-session endpoint returns supplied by the caller.
    h must use the same clock units as unit_returns; a minute sigma cannot be
    divided into a return with an unconverted day horizon. Each index must be
    sorted, unique and identical. Each call must concern one security only.
    """
    parameters.validate()
    for obj in (unit_returns, cumulative_returns, horizon_lengths):
        if not obj.index.equals(price.index):
            raise ValueError("All inputs must use identical observation identities/order.")
    if not price.index.is_unique or not price.index.is_monotonic_increasing:
        raise ValueError("A dense sorted, unique observation clock is required.")
    if cumulative_returns.shape[1] != 5 or not cumulative_returns.columns.equals(horizon_lengths.columns):
        raise ValueError("Exactly five return horizons with matching lengths are required.")
    sigma = weighted_volatility(unit_returns, parameters.volatility_span)
    lengths = horizon_lengths.where(np.isfinite(horizon_lengths) & horizon_lengths.gt(0))
    scaled = cumulative_returns.where(np.isfinite(cumulative_returns)).div(
        np.sqrt(lengths).mul(sigma.where(sigma.gt(0)), axis=0))
    scaled.columns = [f"nm_scaled_{name}" for name in scaled.columns]
    result = pd.concat([scaled, normalized_macds(price, parameters)], axis=1)
    result = result.where(np.isfinite(price) & price.gt(0), axis=0)
    return historical_clip(result, parameters.clipping_halflife) if apply_clipping else result
