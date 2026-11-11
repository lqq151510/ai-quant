"""主入口标签必须从下一开盘起算，回测不能把首日回撤和持仓漂移算丢."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.application.trainer import backtest_top_k, path_metrics
from src.domain.features.pipeline import (
    SpatioTemporalDataset,
    attach_executable_open_returns,
    build_spatio_temporal_dataset,
    chronological_split,
)


class ExecutableOpenReturnTests(unittest.TestCase):
    def test_overnight_gap_is_not_a_tradable_return(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=4)
        bars = pd.DataFrame({
            "symbol": ["510300"] * 4,
            "trade_date": dates,
            "open": [10.0, 12.0, 12.0, 12.0],
            "close": [10.0, 12.0, 12.0, 12.0],
        })
        labeled = attach_executable_open_returns(bars, pred_horizon=1)
        first = labeled.iloc[0]
        self.assertAlmostEqual(float(first["next_1d_ret"]), 0.0)
        self.assertAlmostEqual(float(first["future_ret_h"]), 0.0)
        leaked = bars["close"].shift(-1).iloc[0] / bars["close"].iloc[0] - 1.0
        self.assertAlmostEqual(float(leaked), 0.2)
        self.assertNotAlmostEqual(float(first["next_1d_ret"]), float(leaked))

    def test_dataset_settlement_uses_next_open(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=40)
        rows = []
        for symbol, gap in (("510300", True), ("510500", False)):
            close = np.full(len(dates), 10.0)
            open_ = np.full(len(dates), 10.0)
            if gap:
                open_[26:] = 12.0
                close[26:] = 12.0
            rows.append(pd.DataFrame({
                "symbol": symbol,
                "trade_date": dates,
                "open": open_,
                "high": np.maximum(open_, close),
                "low": np.minimum(open_, close),
                "close": close,
                "volume": np.full(len(dates), 100_000.0),
            }))
        dataset = build_spatio_temporal_dataset(pd.concat(rows), seq_len=5, pred_horizon=1)
        signal = pd.Timestamp(dates[25]).normalize()
        position = next(
            i for i, day in enumerate(dataset.dates) if pd.Timestamp(day).normalize() == signal
        )
        asset = dataset.symbols.index("510300")
        self.assertAlmostEqual(float(dataset.y_raw_ret[position, asset]), 0.0, places=6)

    def test_holding_weights_drift_and_first_loss_counts_as_drawdown(self) -> None:
        preds = np.zeros((3, 2), dtype=np.float32)
        returns = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=np.float32)
        ann_ret, _, _, _ = backtest_top_k(preds, returns, top_k=2, tx_cost=0.0)
        drifted = path_metrics(np.array([0.5, 1.0 / 3.0, 0.0]), np.zeros(3))[0]
        rebalanced = path_metrics(np.array([0.5, 0.5, 0.0]), np.zeros(3))[0]
        self.assertAlmostEqual(ann_ret, drifted)
        self.assertGreater(abs(ann_ret - rebalanced), 1.0)

        loss_ann, _, _, max_drawdown = backtest_top_k(
            np.ones((1, 1), dtype=np.float32),
            np.array([[-0.1]], dtype=np.float32),
            top_k=1,
            tx_cost=0.0,
        )
        self.assertAlmostEqual(max_drawdown, 0.1)
        self.assertLess(loss_ann, 0.0)

    def test_purge_gap_must_cover_open_label(self) -> None:
        n_days = 30
        dataset = SpatioTemporalDataset(
            dates=list(pd.bdate_range("2024-01-02", periods=n_days)),
            symbols=["510300"],
            x_seq=np.zeros((n_days, 1, 2, 1), dtype=np.float32),
            y_alpha=np.zeros((n_days, 1), dtype=np.float32),
            y_raw_ret=np.zeros((n_days, 1), dtype=np.float32),
        )
        with self.assertRaises(ValueError):
            chronological_split(dataset, purge_gap=5, label_horizon=5)
        train_ds, _, _ = chronological_split(dataset, purge_gap=6, label_horizon=5)
        self.assertGreater(len(train_ds.dates), 0)


if __name__ == "__main__":
    unittest.main()
