"""Portfolio weight limits and whole-lot feasibility checks."""

from decimal import Decimal
from numbers import Integral

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


def apply_lot_constraints(
    weights: np.ndarray, prices: np.ndarray, nav: float, lot: int = 100,
) -> tuple[np.ndarray, list[str | None]]:
    """Round each target down at its reference price. Released weight remains cash."""
    targets = np.asarray(weights, dtype=float)
    marks = np.asarray(prices, dtype=float)
    if targets.ndim != 1 or marks.shape != targets.shape:
        raise ValueError("prices must be a vector aligned to weights")
    if not np.isfinite(nav) or nav <= 0:
        raise ValueError("nav must be finite and positive")
    if isinstance(lot, bool) or not isinstance(lot, Integral) or lot <= 0:
        raise ValueError("lot must be a positive integer")
    if not np.isfinite(targets).all() or (targets < 0).any():
        raise ValueError("target weights must be finite and non-negative")
    exact_targets = [Decimal(str(weight)) for weight in targets]
    if sum(exact_targets) > Decimal(1) + Decimal('1e-15'):
        raise ValueError("target weights cannot exceed total NAV")

    equity = Decimal(str(nav))
    executable = np.zeros_like(targets)
    reasons: list[str | None] = [None] * len(targets)
    for index, (weight, price) in enumerate(zip(exact_targets, marks)):
        if not weight:
            continue
        if not np.isfinite(price) or price <= 0:
            reasons[index] = "reference_price_unavailable"
            continue
        lot_notional = Decimal(str(price)) * int(lot)
        lots = (weight * equity) // lot_notional
        if not lots:
            reasons[index] = "target_below_one_lot"
            continue
        executable[index] = float(lots * lot_notional / equity)
    return executable, reasons
