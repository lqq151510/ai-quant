"""Optional, fixed-rule portfolio overlays for the HRP strategy study."""

from __future__ import annotations

from collections.abc import Mapping
from math import fsum, isfinite

import numpy as np
import pandas as pd

from .allocation import _CAPS
from .protocol import ETF_SYMBOLS


REBALANCE_BAND = 0.02
MIN_POSITIVE_ETFS = 2
DEFENSIVE_EXPOSURE_SCALE = 0.50
OPTIMIZED_STRATEGIES = {
    "allocation_hrp_band": (True, False),
    "allocation_hrp_breadth": (False, True),
    "allocation_hrp_combined": (True, True),
}
_TOLERANCE = 1e-12


def market_exposure_scale(momentum: pd.DataFrame) -> pd.Series:
    """Scale exposure using each session's four observed ETF momentum scores."""
    scores = momentum.reindex(columns=ETF_SYMBOLS).to_numpy(dtype=float, na_value=np.nan)
    scale = np.where((scores > 0).sum(axis=1) >= MIN_POSITIVE_ETFS, 1.0, DEFENSIVE_EXPOSURE_SCALE)
    scale[~np.isfinite(scores).all(axis=1)] = np.nan
    return pd.Series(scale, index=momentum.index, name="market_exposure_scale")


def rebalance_target(
    target: Mapping[str, float], current: Mapping[str, float], kind: str, *,
    use_band: bool, exposure_scale: float = 1.0,
) -> dict[str, float] | None:
    """Return the next target, or None to leave actual positions untouched.

    The band measures half the absolute weight change including residual cash.
    Entering or exiting names, repairing cap violations and reducing current
    gross exposure in a defensive regime always bypass the band.
    """
    cap = _CAPS[kind]
    if not isfinite(exposure_scale) or not 0 <= exposure_scale <= 1:
        raise ValueError("exposure scale must be finite and between zero and one")
    if any(not isfinite(weight) or weight < 0 or weight > cap + _TOLERANCE for weight in target.values()):
        raise ValueError("target weights must be finite, long-only and within the allocation cap")
    scaled = {symbol: weight * exposure_scale for symbol, weight in target.items() if weight * exposure_scale > 0}
    target_gross = fsum(scaled.values())
    if target_gross > 1 + _TOLERANCE:
        raise ValueError("target weights cannot use leverage")
    if not use_band or any(
        not isfinite(weight) or weight < 0 or weight > cap + _TOLERANCE for weight in current.values()
    ):
        return scaled

    held = {symbol: weight for symbol, weight in current.items() if weight > 0}
    current_gross = fsum(held.values())
    if (held.keys() != scaled.keys() or current_gross > 1 + _TOLERANCE
            or (exposure_scale < 1 and current_gross > target_gross + _TOLERANCE)):
        return scaled

    turnover = (fsum(abs(held[symbol] - weight) for symbol, weight in scaled.items())
                + abs(current_gross - target_gross)) / 2
    return None if turnover <= REBALANCE_BAND + _TOLERANCE else scaled
