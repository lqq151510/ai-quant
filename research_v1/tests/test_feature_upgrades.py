import numpy as np
import pandas as pd
import pytest

from quant_research.feature_upgrades import LowdinTransform, daily_bar_features, time_decay_weights, wasserstein_drift


def test_close_location_value_and_bar_keys_are_preserved():
    bars = pd.DataFrame({
        "date": pd.bdate_range("2024-01-02", periods=6), "symbol": "A", "open": 10.0,
        "low": [8.0, 8.0, 8.0, 10.0, 8.0, 8.0], "high": [12.0, 12.0, 12.0, 10.0, np.nan, 12.0],
        "close": [8.0, 10.0, 12.0, 10.0, 10.0, 13.0],
    }, index=[5, 1, 1, 4, 3, 2])
    original = bars.copy()
    features = daily_bar_features(bars)
    pd.testing.assert_frame_equal(features[["date", "symbol"]], bars[["date", "symbol"]])
    np.testing.assert_allclose(features["clv"], [-1, 0, 1, np.nan, np.nan, np.nan])
    assert features["up_down_variance_ratio"].isna().all()
    pd.testing.assert_frame_equal(bars, original)


def test_daily_variance_ratio_uses_consecutive_sessions_and_stays_causal():
    days = pd.bdate_range("2024-01-02", periods=42)
    bar_panel = pd.DataFrame({
        "date": days, "symbol": "A", "open": 100.0, "low": 80.0, "high": 160.0,
        "close": 100 * np.exp(np.where(np.arange(len(days)) % 2 == 0, 0.2, -0.1)),
    })
    missing_session = bar_panel[~bar_panel["date"].eq(days[20])]
    features = daily_bar_features(missing_session, sessions=days).set_index("date")["up_down_variance_ratio"]
    assert features.loc[:days[18]].isna().all()
    assert features.loc[days[19]] == pytest.approx(4)
    assert features.loc[days[21]:days[39]].isna().all()
    assert features.loc[days[40]] == pytest.approx(4)

    always_up = bar_panel.assign(symbol="B", close=110.0)
    missing_open = bar_panel.assign(symbol="C")
    missing_open.loc[19, "open"] = np.nan
    panel = pd.concat([bar_panel, always_up, missing_open]).sample(frac=1, random_state=8)
    original = daily_bar_features(panel, sessions=days)
    assert original.loc[original["symbol"].eq("B"), "up_down_variance_ratio"].isna().all()
    invalid = original["symbol"].eq("C") & original["date"].between(days[19], days[38])
    assert original.loc[invalid, "up_down_variance_ratio"].isna().all()
    changed = panel.copy()
    changed.loc[changed["date"].eq(days[35]), "close"] = 150.0
    altered = daily_bar_features(changed, sessions=days)
    past = panel["date"].lt(days[35])
    pd.testing.assert_frame_equal(original.loc[past], altered.loc[past])


def test_lowdin_training_covariance_is_identity_and_holdout_reuses_fit():
    train = np.array([[1, 1], [3, 1], [2, 4], [5, 7]], dtype=float)
    transform = LowdinTransform().fit(train)
    whitened = transform.transform(train)
    np.testing.assert_allclose(whitened.mean(axis=0), [0, 0], atol=1e-12)
    np.testing.assert_allclose(np.cov(whitened, rowvar=False), np.eye(2), atol=1e-12)
    assert (transform.effective_rank_, transform.discarded_rank_) == (2, 0)
    holdout = np.array([[10, -5], [1e6, -1e6]])
    projected = transform.transform(holdout)
    np.testing.assert_allclose(projected[:1], transform.transform(holdout[:1]))
    np.testing.assert_array_equal(transform.mean_, train.mean(axis=0))


def test_lowdin_rank_deficiency_preserves_columns_and_reports_projector():
    trend = np.arange(6.0)
    train = np.column_stack([trend, 2 * trend, np.full_like(trend, 5)])
    transform = LowdinTransform().fit(train)
    result = transform.transform(train)
    assert result.shape == train.shape
    assert (transform.effective_rank_, transform.discarded_rank_) == (1, 2)
    np.testing.assert_allclose(np.cov(result, rowvar=False), [[0.2, 0.4, 0], [0.4, 0.8, 0], [0, 0, 0]], atol=1e-12)
    assert np.all(LowdinTransform().fit(np.ones((3, 2))).transform(train[:, :2]) == 0)
    with pytest.raises(ValueError, match="fit the Löwdin"):
        LowdinTransform().transform(train)
    with pytest.raises(ValueError, match="finite rows"):
        transform.transform([[0, np.inf, 0]])


def test_wasserstein_uses_disjoint_windows_and_current_close_only():
    pytest.importorskip("scipy")
    days = pd.bdate_range("2024-01-02", periods=7)
    returns = pd.DataFrame({"A": [0, 0, 0, 4, 4, 9, 15], "B": [0, 0, 0, 4, 4, 9, 15]}, index=days)
    result = wasserstein_drift(returns, reference_sessions=3, recent_sessions=2)
    assert result.index.equals(days)
    assert result.iloc[:4].isna().all()
    assert result.iloc[4] == pytest.approx(4)
    changed = returns.copy()
    changed.iloc[-1] = 100
    updated = wasserstein_drift(changed, reference_sessions=3, recent_sessions=2)
    pd.testing.assert_series_equal(result.iloc[:-1], updated.iloc[:-1])
    assert updated.iloc[-1] != result.iloc[-1]


def test_wasserstein_finite_coverage_and_valid_zero_drift():
    pytest.importorskip("scipy")
    returns = pd.DataFrame({"A": [0.0] * 70})
    returns.iloc[-1] = np.inf
    returns.iloc[-2] = np.nan
    result = wasserstein_drift(returns)
    assert result.iloc[:-1].isna().all()
    assert result.iloc[-1] == 0
    returns.iloc[-3] = -np.inf
    assert np.isnan(wasserstein_drift(returns).iloc[-1])
    with pytest.raises(ValueError, match="unique ascending"):
        wasserstein_drift(returns.iloc[::-1])


def test_time_decay_uses_session_offsets_and_equal_weights_for_same_day():
    weights = time_decay_weights([240, 120, 0, 0])
    np.testing.assert_allclose(weights / weights[-1], [0.25, 0.5, 1, 1])
    assert weights.mean() == pytest.approx(1)
    assert time_decay_weights([10, 0], half_life=10)[0] / time_decay_weights([10, 0], half_life=10)[1] == 0.5
    for ages in ([], [-1, 0], [0.5, 0], [np.nan, 0]):
        with pytest.raises(ValueError, match="session ages"):
            time_decay_weights(ages)
    with pytest.raises(ValueError, match="half-life"):
        time_decay_weights([0], half_life=0)
