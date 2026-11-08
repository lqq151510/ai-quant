"""Compare three ways to allocate the same long-only research portfolio."""

from __future__ import annotations

from functools import cache

import numpy as np
import pandas as pd


GROSS_BUDGET_FRACTION = 0.80
RETURN_LOOKBACK = 60
_VOLATILITY_WINDOW = 20
_CAPS = {"stock": 0.10, "etf": 0.35}
_METHODS = ("allocation_equal", "allocation_inverse_vol", "allocation_hrp")


@cache
def _allocation_dependencies():
    try:
        from scipy.cluster.hierarchy import leaves_list, linkage
        from scipy.spatial.distance import squareform
        from sklearn.covariance import LedoitWolf
    except ModuleNotFoundError as exc:
        raise ValueError(
            "Allocation comparison requires optional dependencies. "
            "Run `uv sync --extra allocation` from research_v1."
        ) from exc
    return leaves_list, linkage, squareform, LedoitWolf


def require_allocation_dependencies() -> None:
    """Check optional dependencies before starting an allocation comparison."""
    _allocation_dependencies()


def _cluster_variance(covariance: np.ndarray, members: np.ndarray) -> float:
    sub_covariance = covariance[np.ix_(members, members)]
    variances = sub_covariance.diagonal()
    weights = variances.min() / variances
    weights /= weights.sum()
    return float(weights @ sub_covariance @ weights)


def _hrp_weights(returns: np.ndarray) -> np.ndarray:
    leaves_list, linkage, squareform, LedoitWolf = _allocation_dependencies()
    # Shrink the covariance toward a scaled identity, then cluster by correlation.
    covariance = LedoitWolf().fit(returns).covariance_
    volatility = np.sqrt(covariance.diagonal())
    correlation = covariance / np.outer(volatility, volatility)
    distance = np.sqrt(np.clip(0.5 * (1 - correlation), 0, 1))
    np.fill_diagonal(distance, 0)
    order = leaves_list(linkage(squareform(distance, checks=False), method="single"))

    # Split the ordered leaves by the risk of each cluster's inverse-variance portfolio.
    weights = np.ones(len(order))
    variance_floor = np.finfo(float).eps * covariance.diagonal().max()
    clusters = [order]
    while clusters:
        cluster = clusters.pop()
        if len(cluster) > 1:
            middle = len(cluster) // 2
            left, right = cluster[:middle], cluster[middle:]
            left_risk = max(_cluster_variance(covariance, left), variance_floor)
            right_risk = max(_cluster_variance(covariance, right), variance_floor)
            weights[left] *= right_risk / (left_risk + right_risk)
            weights[right] *= left_risk / (left_risk + right_risk)
            clusters.extend((left, right))
    return weights


def _cap_weights(raw: np.ndarray, gross: float, cap: float) -> np.ndarray:
    """Keep the common gross by reallocating excess among uncapped candidates."""
    weights = np.zeros_like(raw)
    active = np.arange(len(raw))
    while active.size:
        proportional = raw[active] / raw[active].sum() * gross
        capped = proportional > cap
        if not capped.any():
            weights[active] = proportional
            break
        weights[active[capped]] = cap
        gross -= cap * capped.sum()
        active = active[~capped]
    return weights


def allocation_targets(
    scores: pd.Series, history: pd.DataFrame, kind: str
) -> dict[str, dict[str, float]]:
    """Allocate from positive momentum and chronological adjusted daily returns.

    The caller supplies rows through the signal close. Eligibility requires 20
    consecutive finite returns above -100% with nonzero volatility. Stock
    candidates are limited to the ten highest scores, breaking ties by code.
    HRP uses the contiguous suffix of up to 60 returns shared by all candidates.
    The fixed budget leaves cash and room for the methods to differ under the caps.
    """
    cap = _CAPS[kind]
    targets: dict[str, dict[str, float]] = {method: {} for method in _METHODS}
    if len(history) < _VOLATILITY_WINDOW:
        return targets

    positive = scores[scores.gt(0) & np.isfinite(scores)]
    names = sorted(positive.index, key=lambda symbol: (-positive.loc[symbol], symbol))
    returns = history.tail(RETURN_LOOKBACK).reindex(columns=names).to_numpy(dtype=float, na_value=np.nan)
    valid = np.isfinite(returns) & (returns > -1)
    complete = np.flatnonzero(valid[-_VOLATILITY_WINDOW:].all(axis=0))
    recent = returns[-_VOLATILITY_WINDOW:, complete]
    varying = np.ptp(recent, axis=0) > 0
    volatility = recent.std(axis=0)
    eligible = varying & np.isfinite(volatility) & (volatility > 0)
    selected = complete[eligible]
    volatility = volatility[eligible]
    if kind == "stock":
        selected, volatility = selected[:10], volatility[:10]
    if not selected.size:
        return targets

    gross = GROSS_BUDGET_FRACTION * min(1.0, selected.size * cap)
    equal = np.ones(selected.size)
    inverse_vol = volatility.min() / volatility
    hrp = equal
    if selected.size > 1:
        invalid_rows = np.flatnonzero(~valid[:, selected].all(axis=1))
        start = int(invalid_rows[-1]) + 1 if invalid_rows.size else 0
        hrp = _hrp_weights(returns[start:, selected])
    for method, raw in zip(_METHODS, (equal, inverse_vol, hrp), strict=True):
        weights = _cap_weights(raw, gross, cap)
        targets[method] = {names[index]: float(weight) for index, weight in zip(selected, weights, strict=True)}
    return targets
