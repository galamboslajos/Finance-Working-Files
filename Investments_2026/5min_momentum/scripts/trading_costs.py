"""Standard research cost: 10 bp per executed entry/exit dollar notional.

This is a declared aggregate cost assumption, not measured spread/impact/borrow.
NaN outcomes remain unknown. Original completed releases keep their assumptions.
"""
from __future__ import annotations
import numpy as np

STANDARD_COST_BPS = 10


def round_trip_notional(weight, raw_return):
    """Entry |w| plus actual exit |w| × (exit price / entry price)."""
    weight, raw_return = np.broadcast_arrays(np.asarray(weight,float),np.asarray(raw_return,float))
    if np.isinf(weight).any() or np.isinf(raw_return).any() or (raw_return < -1).any():
        raise ValueError('Invalid position return or weight')
    return np.abs(weight)*(2+raw_return)


def net_after_costs(gross_return, trade_notional, cost_bps=STANDARD_COST_BPS):
    """Charge each executed side; a flat round trip at 100% gross costs 20 bp."""
    if not np.isscalar(cost_bps) or not np.isfinite(cost_bps) or cost_bps < 0:
        raise ValueError('Cost must be a finite nonnegative number of bp per side')
    gross, notional = np.broadcast_arrays(np.asarray(gross_return,float),np.asarray(trade_notional,float))
    if np.isinf(gross).any() or np.isinf(notional).any() or (notional < 0).any():
        raise ValueError('Invalid gross return or executed notional')
    return gross-notional*cost_bps/10000
