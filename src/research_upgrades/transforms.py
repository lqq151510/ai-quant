"""Training-only symmetric whitening, decay weights, and causal distribution monitoring."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import wasserstein_distance
from sklearn.mixture import GaussianMixture


class LowdinTransform:
    """Symmetric inverse square root, fitted on training rows only.

    Rank-deficient directions are projected out, rather than pretending that
    duplicate or constant inputs can be made independent.
    """

    def __init__(self, relative_tolerance: float = 1e-8):
        if not 0 < relative_tolerance < 1:
            raise ValueError("relative_tolerance must be between 0 and 1")
        self.relative_tolerance = relative_tolerance
        self.mean = None
        self.matrix = None
        self.rank = 0
        self.projection = None

    def fit(self, x: np.ndarray) -> "LowdinTransform":
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] == 0 or not np.isfinite(x).all():
            raise ValueError("x must be a finite, nonempty 2D matrix with at least two rows")
        self.mean = x.mean(axis=0)
        centered = x - self.mean
        covariance = centered.T @ centered / len(centered)
        eigvals, eigvecs = np.linalg.eigh(covariance)
        cutoff = max(float(eigvals[-1]) * self.relative_tolerance, np.finfo(float).eps)
        keep = eigvals > cutoff
        self.rank = int(keep.sum())
        inverse_roots = np.zeros_like(eigvals)
        inverse_roots[keep] = 1.0 / np.sqrt(eigvals[keep])
        self.matrix = (eigvecs * inverse_roots) @ eigvecs.T
        self.projection = (eigvecs * keep.astype(float)) @ eigvecs.T
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.matrix is None:
            raise ValueError("Fit the transform before using it")
        x = np.asarray(x, dtype=np.float64)
        if x.shape[-1] != len(self.mean) or not np.isfinite(x).all():
            raise ValueError("Input features must match the fitted finite feature dimensions")
        return (x - self.mean) @ self.matrix


def exponential_time_weights(dates, half_life: float = 120.0) -> np.ndarray:
    """Weight each ordered eligible observation date; 120 means 120 sampled dates."""
    index = pd.DatetimeIndex(pd.to_datetime(dates))
    if not np.isfinite(half_life) or half_life <= 0:
        raise ValueError("half_life must be positive and finite")
    if len(index) == 0 or index.hasnans or not index.is_monotonic_increasing or index.has_duplicates:
        raise ValueError("dates must be nonempty, unique, and strictly increasing")
    ages = np.arange(len(index) - 1, -1, -1, dtype=np.float64)
    weights = np.exp2(-ages / half_life)
    return weights / weights.mean()


def macro_distribution_observations(
    bars: pd.DataFrame, reference_days: int = 120, recent_days: int = 20,
) -> pd.DataFrame:
    """Compare pooled returns in two nonoverlapping windows ending at t.

    The result describes the supplied universe, not automatically the whole market.
    """
    if reference_days < 2 or recent_days < 2:
        raise ValueError("Both comparison windows must contain at least two dates")
    df = bars.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values(["symbol", "trade_date"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    if df.duplicated(["symbol", "trade_date"]).any() or df["trade_date"].isna().any():
        raise ValueError("Macro bars require unique symbol/date keys and valid dates")
    if not np.isfinite(df["close"]).all() or (df["close"] <= 0).any():
        raise ValueError("Macro close prices must be positive and finite")
    market_dates = pd.Index(sorted(df["trade_date"].unique()))
    df["date_index"] = market_dates.get_indexer(df["trade_date"])
    df["ret"] = df.groupby("symbol")["close"].pct_change(fill_method=None)
    # A suspension or a missing source record must not be treated as a one-day return.
    gap = df.groupby("symbol")["date_index"].diff()
    df.loc[gap != 1, "ret"] = np.nan
    daily = df.groupby("trade_date").agg(
        mkt_ret=("ret", "mean"), dispersion=("ret", "std"),
        returns=("ret", lambda x: x.dropna().to_numpy(dtype=float)),
    ).sort_index()
    daily = daily[daily["returns"].map(len) > 1].copy()
    daily["trend"] = daily["mkt_ret"].rolling(5, min_periods=5).mean()
    daily["volatility"] = daily["mkt_ret"].rolling(10, min_periods=10).std()
    values = daily["returns"].tolist()
    distances = np.full(len(daily), np.nan)
    for end in range(reference_days + recent_days - 1, len(daily)):
        start = end + 1 - recent_days
        reference = np.concatenate(values[start - reference_days:start])
        recent = np.concatenate(values[start:end + 1])
        distances[end] = wasserstein_distance(reference, recent)
    daily["wasserstein"] = distances
    return daily.drop(columns="returns").dropna().reset_index()


@dataclass
class RegimeObservation:
    trade_date: pd.Timestamp
    probabilities: np.ndarray
    distribution_distance: float
    distribution_alert: bool


class WassersteinRegimeMonitor:
    """Research GMM emissions + sticky forward filter, with W1 as fifth input.

    The threshold is a training-period quantile, not a calibrated crash probability.
    """

    COLUMNS = ["mkt_ret", "trend", "volatility", "dispersion", "wasserstein"]

    def __init__(self, self_transition: float = 0.82, quantile: float = 0.95):
        if not 0 < self_transition < 1 or not 0 < quantile < 1:
            raise ValueError("Transition probability and quantile must lie between 0 and 1")
        self.self_transition = self_transition
        self.quantile = quantile
        self.gmm = GaussianMixture(n_components=3, reg_covar=1e-4, random_state=42)
        self.mean = None
        self.std = None
        self.permutation = None
        self.threshold = None

    def fit(self, training: pd.DataFrame) -> "WassersteinRegimeMonitor":
        x = training[self.COLUMNS].to_numpy(dtype=float)
        if len(x) < 10 or not np.isfinite(x).all():
            raise ValueError("At least ten finite training macro observations are required")
        self.mean = x.mean(axis=0)
        self.std = np.maximum(x.std(axis=0), 1e-8)
        self.gmm.fit((x - self.mean) / self.std)
        # Semantic order follows the prototype: higher return/lower volatility first.
        means = self.gmm.means_
        self.permutation = np.argsort(means[:, 0] + means[:, 1] - 0.8 * means[:, 2])[::-1]
        self.threshold = float(np.quantile(x[:, 4], self.quantile))
        return self

    def filter_sequence(self, observations: pd.DataFrame) -> list[RegimeObservation]:
        if self.threshold is None:
            raise ValueError("Fit the monitor before filtering")
        dates = pd.DatetimeIndex(pd.to_datetime(observations["trade_date"]))
        if dates.has_duplicates or not dates.is_monotonic_increasing:
            raise ValueError("Macro observations must be sorted and unique")
        x = observations[self.COLUMNS].to_numpy(dtype=float)
        if not np.isfinite(x).all():
            raise ValueError("Macro observations must be finite")
        # Use the public score_samples and predict_proba APIs to recover component log likelihoods.
        normalized = (x - self.mean) / self.std
        log_joint = np.log(np.maximum(self.gmm.predict_proba(normalized), 1e-300))
        log_joint += self.gmm.score_samples(normalized)[:, None]
        log_emission = log_joint - np.log(self.gmm.weights_)[None, :]
        log_emission = log_emission[:, self.permutation]
        transition = np.full((3, 3), (1.0 - self.self_transition) / 2.0)
        np.fill_diagonal(transition, self.self_transition)
        alpha = np.array([0.30, 0.50, 0.20])
        results = []
        for date, row, emission in zip(dates, x, log_emission):
            log_post = np.log(alpha @ transition) + emission / 3.5
            alpha = np.exp(log_post - logsumexp(log_post))
            results.append(RegimeObservation(date, alpha.copy(), float(row[4]), bool(row[4] > self.threshold)))
        return results
