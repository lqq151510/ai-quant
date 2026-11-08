"""Daily bar proxies and causal research transforms for controlled comparisons."""

from __future__ import annotations

from collections.abc import Iterable
from numbers import Integral

import numpy as np
import pandas as pd

from .timeline import sessions as exchange_sessions


def daily_bar_features(bars: pd.DataFrame, *, sessions: Iterable[object] | None = None) -> pd.DataFrame:
    """Compute close-location value and an up/down squared-return ratio at the close.

    The ratio sums squared positive log(close/open) returns and divides by
    the sum of squared negative returns across the last 20 exchange sessions,
    including the current one. It requires a valid open and close on every
    session and nonzero downside. Supply the exchange calendar to catch dates
    absent from all symbols. Otherwise dates are inferred from the bar panel.
    """
    keys = bars[["date", "symbol"]].copy()
    days = pd.DatetimeIndex(pd.to_datetime(keys["date"], errors="raise")).normalize()
    if days.hasnans or days.tz is not None or keys["symbol"].isna().any():
        raise ValueError("daily bars require timezone-naive dates and symbols")
    work = pd.DataFrame({"date": days, "symbol": keys["symbol"].to_numpy()})
    if work.duplicated(["date", "symbol"]).any():
        raise ValueError("duplicate daily bars")
    calendar = days.unique().sort_values() if sessions is None else exchange_sessions(sessions)
    work["session"] = calendar.get_indexer(days)
    if work["session"].lt(0).any():
        raise ValueError("daily bar outside the exchange calendar")

    ohlc = bars[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
    open_, high, low, close = ohlc.to_numpy(dtype=float, na_value=np.nan).T
    valid_range = np.isfinite(high) & np.isfinite(low) & (high > low) & (close >= low) & (close <= high)
    clv = np.full(len(work), np.nan)
    clv[valid_range] = 2 * ((close[valid_range] - low[valid_range]) / (high[valid_range] - low[valid_range])) - 1

    valid_return = np.isfinite(open_) & np.isfinite(close) & (open_ > 0) & (close > 0)
    log_return = np.full(len(work), np.nan)
    log_return[valid_return] = np.log(close[valid_return]) - np.log(open_[valid_return])
    work["log_return"] = log_return
    ratio = np.full(len(work), np.nan)
    for _, asset in work.sort_values("session").groupby("symbol", sort=False):
        returns = asset["log_return"]
        upside = returns.clip(lower=0).pow(2).rolling(20).sum()
        downside = returns.clip(upper=0).pow(2).rolling(20).sum()
        consecutive = asset["session"].diff(19).eq(19)
        values = upside / downside.where(downside.gt(0) & consecutive)
        ratio[asset.index] = values.where(np.isfinite(values)).to_numpy()
    return keys.assign(clv=clv, up_down_variance_ratio=ratio)


class LowdinTransform:
    """Symmetric whitening with a pseudoinverse square root of training covariance.

    The centered sample covariance uses the n-1 denominator. Eigenvalues at
    or below rcond times the largest eigenvalue are discarded. Transformed
    training covariance is identity at full rank and the orthogonal projector
    onto retained directions otherwise. The number of columns stays fixed.
    """

    def __init__(self, *, rcond: float = 1e-12) -> None:
        if not np.isfinite(rcond) or not 0 < rcond < 1:
            raise ValueError("rcond must be finite and between zero and one")
        self.rcond = rcond

    def fit(self, features: np.ndarray) -> LowdinTransform:
        matrix = np.asarray(features, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] == 0 or not np.isfinite(matrix).all():
            raise ValueError("Löwdin fit requires at least two finite rows and one feature")
        mean = matrix.mean(axis=0)
        centered = matrix - mean
        covariance = (centered.T @ centered) / (len(centered) - 1)
        if not np.isfinite(covariance).all():
            raise ValueError("Löwdin training covariance must be finite")
        eigenvalues, directions = np.linalg.eigh(covariance)
        cutoff = eigenvalues[-1] * self.rcond
        retained = eigenvalues > cutoff
        basis = directions[:, retained]
        self.mean_ = mean
        self.whitener_ = (basis / np.sqrt(eigenvalues[retained])) @ basis.T
        self.effective_rank_ = int(retained.sum())
        self.discarded_rank_ = matrix.shape[1] - self.effective_rank_
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        if not hasattr(self, "whitener_"):
            raise ValueError("fit the Löwdin transform first")
        matrix = np.asarray(features, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.mean_) or not np.isfinite(matrix).all():
            raise ValueError("Löwdin transform requires finite rows with the fitted feature count")
        result = (matrix - self.mean_) @ self.whitener_
        if not np.isfinite(result).all():
            raise ValueError("Löwdin transformed features must be finite")
        return result


def wasserstein_drift(
    returns: pd.DataFrame, *, reference_sessions: int = 60, recent_sessions: int = 10,
) -> pd.Series:
    """Compare pooled daily returns in adjacent, disjoint exchange-session windows.

    Rows must represent the full ascending session calendar. The recent window
    ends at the current close and the reference ends before the recent window
    starts. Both windows need at least 80% finite cells and two observations.
    Only finite returns participate, with equal weight per observation.
    """
    windows = (reference_sessions, recent_sessions)
    if any(not isinstance(size, Integral) or isinstance(size, bool) or size <= 0 for size in windows):
        raise ValueError("Wasserstein window lengths must be positive integers")
    if not returns.index.is_unique or not returns.index.is_monotonic_increasing:
        raise ValueError("Wasserstein returns require unique ascending session rows")
    try:
        from scipy.stats import wasserstein_distance
    except ModuleNotFoundError as exc:
        raise ValueError("Wasserstein drift requires `uv sync --extra allocation` from research_v1.") from exc

    matrix = returns.to_numpy(dtype=float, na_value=np.nan)
    output = np.full(len(returns), np.nan)
    width = reference_sessions + recent_sessions
    for end in range(width, len(matrix) + 1):
        split = end - recent_sessions
        reference, recent = matrix[end - width:split], matrix[split:end]
        clean_reference = reference[np.isfinite(reference)]
        clean_recent = recent[np.isfinite(recent)]
        if len(clean_reference) >= max(2, 0.8 * reference.size) and len(clean_recent) >= max(2, 0.8 * recent.size):
            output[end - 1] = wasserstein_distance(clean_reference, clean_recent)
    return pd.Series(output, index=returns.index, name="wasserstein_drift")


def time_decay_weights(session_ages: Iterable[int], *, half_life: float = 120) -> np.ndarray:
    """Return mean-one exponential weights from nonnegative exchange-session ages.

    Age zero denotes the latest training session. Rows sharing the same
    session age receive equal weights regardless of the number of rows there.
    """
    ages = np.asarray(list(session_ages), dtype=float)
    if ages.ndim != 1 or not ages.size or not np.isfinite(ages).all() or (ages < 0).any() or (ages % 1 != 0).any():
        raise ValueError("expected a nonempty sequence of nonnegative integer session ages")
    if not np.isfinite(half_life) or half_life <= 0:
        raise ValueError("half-life must be positive and finite")
    weights = np.exp2(-(ages - ages.min()) / half_life)
    return weights / weights.mean()
