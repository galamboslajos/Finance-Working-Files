"""Pure independent Decimal check of an assumed rounding envelope; no I/O."""
from decimal import Decimal, localcontext


def factor_intersection(raw, adjusted, half_width='0.005'):
    if len(raw) != 4 or len(adjusted) != 4:
        raise ValueError('Exactly four OHLC pairs required')
    with localcontext() as ctx:
        ctx.prec = 50
        h = Decimal(half_width)
        if not h.is_finite() or h <= 0:
            raise ValueError('Positive fixed half-width required')
        rr = [Decimal(str(x)) for x in raw]
        aa = [Decimal(str(x)) for x in adjusted]
        if any(not x.is_finite() or x <= h for x in rr + aa):
            return {'status': 'UNAVAILABLE_NONPOSITIVE_OR_ZERO_CROSSING_INTERVAL',
                    'common_factor_possible_under_assumption': None}
        lows = [(a - h) / (r + h) for r, a in zip(rr, aa)]
        highs = [(a + h) / (r - h) for r, a in zip(rr, aa)]
        lower, upper = max(lows), min(highs)
        return {'status': 'MATHEMATICAL_DIAGNOSTIC_ONLY', 'lower': str(lower), 'upper': str(upper),
                'half_width': str(h), 'common_factor_possible_under_assumption': lower <= upper,
                'source_rounding_or_economic_returns_certified': False}
