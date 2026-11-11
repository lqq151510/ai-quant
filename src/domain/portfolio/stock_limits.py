"""Position limits shared by the portfolio backtest and plan generator."""

import numpy as np


MAX_STOCK_NAV_WEIGHT = 0.05
MAX_STOCK_HOLDINGS_SHARE = 0.50


def limit_stock_weights(weights: np.ndarray, stock_mask: np.ndarray) -> np.ndarray:
    """Cut individual stock targets only. Weights are fractions of total NAV."""
    limited = np.array(weights, dtype=float, copy=True)
    mask = np.asarray(stock_mask)
    if limited.ndim != 1 or mask.shape != limited.shape or mask.dtype != np.bool_:
        raise ValueError("stock_mask must be a boolean vector aligned to weights")
    if not np.isfinite(limited).all() or (limited < 0).any():
        raise ValueError("target weights must be finite and non-negative")

    stocks = np.flatnonzero(mask)
    if stocks.size == 0:
        return limited

    limited[stocks] = np.minimum(limited[stocks], MAX_STOCK_NAV_WEIGHT)
    largest = stocks[np.argmax(limited[stocks])]
    others = limited[:largest].sum() + limited[largest + 1:].sum()
    # At most one asset can exceed half of invested value. Reducing it to the
    # sum of the others gives both sides half, with the released weight in cash.
    limited[largest] = min(limited[largest], others)
    return limited
