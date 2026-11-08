import numpy as np
import pandas as pd
import pytest

from quant_research.protocol import ETF_SYMBOLS
from quant_research.strategy_optimization import market_exposure_scale, rebalance_target


def test_breadth_requires_four_finite_scores_from_each_session():
    momentum = pd.DataFrame(
        [[1.0, 0.0, -0.1, -0.2], [0.1, 0.2, -0.1, -0.2], [0.1, 0.2, np.inf, 0.3]],
        columns=ETF_SYMBOLS,
        index=pd.date_range("2024-01-01", periods=3),
    )
    expected = pd.Series([0.5, 1.0, np.nan], index=momentum.index, name="market_exposure_scale")
    pd.testing.assert_series_equal(market_exposure_scale(momentum), expected)
    assert market_exposure_scale(momentum.iloc[:, :3]).isna().all()
    pd.testing.assert_series_equal(market_exposure_scale(momentum.iloc[:2]), expected.iloc[:2])


def test_band_accounts_for_both_asset_and_cash_weight_changes():
    target = {"A": 0.20, "B": 0.20}
    # Total weight transferred from cash is exactly the band.
    assert rebalance_target(target, {"A": 0.19, "B": 0.19}, "etf", use_band=True) is None
    assert rebalance_target(target, {"A": 0.188, "B": 0.19}, "etf", use_band=True) == target
    assert rebalance_target(target, target, "etf", use_band=False) == target


@pytest.mark.parametrize(
    ("target", "current"),
    [({"A": 0.08}, {"A": 0.08, "B": 0.001}), ({"A": 0.08, "B": 0.001}, {"A": 0.08}),
     ({"A": 0.10}, {"A": 0.101}), ({"A": 0.08}, {"A": float("nan")}), ({}, {"A": 0.001})],
)
def test_band_does_not_suppress_entries_exits_or_invalid_holdings(target, current):
    assert rebalance_target(target, current, "stock", use_band=True) == target


def test_defensive_cut_bypasses_band_when_current_exposure_is_above_target():
    target = {"A": 0.20, "B": 0.20}
    scaled = {"A": 0.10, "B": 0.10}
    assert rebalance_target(target, {"A": 0.105, "B": 0.105}, "etf", use_band=True, exposure_scale=0.5) == scaled
    assert rebalance_target(target, scaled, "etf", use_band=True, exposure_scale=0.5) is None
    assert rebalance_target(target, {}, "etf", use_band=False, exposure_scale=0.5) == scaled
