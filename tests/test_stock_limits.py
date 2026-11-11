"""Regressions for individual stock limits in portfolio targets and plan output."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from src.application import trading_plan_generator as plans
from src.application.walk_forward_engine import _stock_mask_from_bars, apply_turnover_deadband
from src.domain.portfolio.stock_limits import limit_stock_weights
from src.domain.regime.market_hmm import MarketRegimeSnapshot
from src.infrastructure.market_data import CORE_UNIVERSE, AssetMeta


class TestStockLimits(unittest.TestCase):
    def test_five_percent_nav_cap_leaves_released_weight_in_cash(self):
        weights = np.array([0.25, 0.06, 0.40])
        mask = np.array([True, True, False])
        result = limit_stock_weights(weights, mask)
        np.testing.assert_allclose(result, [0.05, 0.05, 0.40])
        np.testing.assert_array_equal(weights, [0.25, 0.06, 0.40])
        np.testing.assert_array_equal(mask, [True, True, False])
        self.assertFalse(np.shares_memory(result, weights))

    def test_concentration_uses_post_cap_holdings_excluding_cash(self):
        weights = np.array([0.25, 0.02, 0.01])
        result = limit_stock_weights(weights, np.array([True, True, False]))
        np.testing.assert_allclose(result, [0.03, 0.02, 0.01])
        self.assertAlmostEqual(result[0] / result.sum(), 0.5)

    def test_single_stock_cannot_invest_but_single_etf_can(self):
        np.testing.assert_array_equal(limit_stock_weights(np.array([0.6]), np.array([True])), [0])
        np.testing.assert_array_equal(limit_stock_weights(np.array([0.6]), np.array([False])), [0.6])

    def test_first_entry_and_inertia_do_not_exceed_the_limits(self):
        mask = np.array([True, False])
        proposed = np.array([0.5, 0.5])
        target, trade = apply_turnover_deadband(proposed, np.zeros(2), stock_mask=mask, max_exposure=0.6)
        self.assertTrue(trade)
        np.testing.assert_allclose(target, [0.05, 0.30])
        current = np.array([0.05, 0.10])
        target, trade = apply_turnover_deadband(proposed, current, stock_mask=mask, max_exposure=0.6)
        self.assertTrue(trade)
        self.assertLessEqual(target[0], 0.05)
        self.assertLessEqual(target.sum(), 0.6)

    def test_existing_stock_breach_bypasses_deadband(self):
        mask = np.array([True, False])
        for previous, expected in (([0.051, 0.1], [0.05, 0.1]), ([0.04, 0.02], [0.02, 0.02])):
            with self.subTest(previous=previous):
                weights = np.array(previous)
                target, trade = apply_turnover_deadband(weights, weights, stock_mask=mask)
                self.assertTrue(trade)
                np.testing.assert_allclose(target, expected)
                np.testing.assert_array_equal(weights, previous)
        previous = np.array([0.03, 0.04])
        target, trade = apply_turnover_deadband(np.array([0.031, 0.04]), previous, stock_mask=mask)
        self.assertFalse(trade)
        np.testing.assert_array_equal(target, previous)

    def test_tiny_removed_etf_cannot_count_towards_concentration_denominator(self):
        target, trade = apply_turnover_deadband(
            np.array([0.05, 0.04995, 0.00006]), np.zeros(3), stock_mask=np.array([True, True, False]),
        )
        self.assertTrue(trade)
        np.testing.assert_allclose(target, [0.04995, 0.04995, 0.0])

    def test_inertia_cannot_keep_a_new_single_stock_candidate_alive(self):
        for previous in (np.zeros(2), np.array([0.0, 0.2])):
            target, trade = apply_turnover_deadband(np.array([0.5, 0.0]), previous, stock_mask=np.array([True, False]))
            np.testing.assert_array_equal(target, [0.0, 0.0])
            self.assertEqual(trade, bool(np.any(previous)))

    def test_kind_is_explicit_and_fails_for_ambiguous_or_missing_rows(self):
        bars = pd.DataFrame({"symbol": ["600000", "600000", "510300"], "kind": ["etf", "etf", "stock"]})
        np.testing.assert_array_equal(_stock_mask_from_bars(bars, ["510300", "600000"]), [True, False])
        for invalid_kind in ("stock", None, "fund"):
            with self.subTest(invalid_kind=invalid_kind):
                invalid = bars.copy()
                invalid.loc[0, "kind"] = invalid_kind
                with self.assertRaisesRegex(ValueError, "bars.kind"):
                    _stock_mask_from_bars(invalid, ["510300", "600000"])
        with self.assertRaisesRegex(ValueError, "bars.kind"):
            _stock_mask_from_bars(bars.drop(columns="kind"), ["510300", "600000"])
        with self.assertRaisesRegex(ValueError, "bars.kind"):
            _stock_mask_from_bars(bars, ["000001", "600000"])

    def test_core_universe_preserves_stock_default_and_explicit_etfs(self):
        self.assertEqual(AssetMeta("000001", "Stock", "Sector", True).kind, "stock")
        self.assertEqual({asset.symbol for asset in CORE_UNIVERSE if asset.kind == "etf"},
                         {"512480", "515880", "515790", "510880", "510300", "518880", "513100"})


class TestStockLimitsInPlan(unittest.TestCase):
    def generate_plan(self, *, hrp: bool, stock_eligible=True, etf_eligible=True):
        day = pd.Timestamp("2024-12-24")
        symbols = ["002594", "510300"]
        snapshot = pd.DataFrame({
            "symbol": symbols, "close": [10.0, 10.0], "atr_14": [0.5, 0.5], "ret_1d_raw": [0.0, 0.0],
            "ma_20": [9.0 if stock_eligible else 11.0, 9.0 if etf_eligible else 11.0],
        })
        model_returns = np.array([0.8, 0.2])
        if not stock_eligible:
            model_returns = np.array([0.0, 1.0])
        if not etf_eligible:
            model_returns = np.array([1.0, 0.0])
        regime = MarketRegimeSnapshot(day, 0, "测试状态", np.array([1, 0, 0]), 0.4)
        with TemporaryDirectory() as directory:
            output = Path(directory) / "plan.md"
            with (
                patch.object(plans, "infer_latest_live_alpha", return_value=(day, symbols, np.array([0.2, 0.1]), np.zeros(2))),
                patch.object(plans, "compute_latest_atr_and_snapshot", return_value=snapshot),
                patch.object(plans.HierarchicalRiskParityOptimizer, "allocate_hrp_weights", return_value=model_returns) as allocator,
                patch.object(plans, "build_alpha_risk_prior", return_value=torch.log(torch.tensor([[8.0, 2.0, 1.0]]))) as fallback,
            ):
                report = plans.generate_semi_auto_trading_plan(
                    pd.DataFrame(), torch.nn.Identity(), {}, pd.DataFrame(), 0.0,
                    latest_regime=regime, hist_raw_rets=np.zeros((20, 2)) if hrp else None, output_path=str(output),
                )
            self.assertEqual(output.read_text(encoding="utf-8"), report)
            self.assertEqual(allocator.called, hrp)
            self.assertEqual(fallback.called, not hrp)
        return report

    def test_both_allocators_apply_gross_cap_and_stock_cap_before_cash_reporting(self):
        for hrp in (True, False):
            with self.subTest(hrp=hrp):
                report = self.generate_plan(hrp=hrp)
                self.assertIn("002594 32.00%→5.00%", report)
                self.assertIn("建议账户总仓位**: **13.0%", report)
                self.assertIn("建议现金防守仓位**: **87.0%", report)
                self.assertIn("总净值 5%、目标持仓市值 50%", report)
                stock_row = next(line for line in report.splitlines() if "`002594`" in line)
                etf_row = next(line for line in report.splitlines() if "`510300`" in line)
                self.assertIn("**5.0%**", stock_row)
                self.assertIn("**8.0%**", etf_row)

    def test_only_stock_has_zero_weight_and_no_buy_zone_in_both_allocators(self):
        for hrp in (True, False):
            with self.subTest(hrp=hrp):
                report = self.generate_plan(hrp=hrp, etf_eligible=False)
                self.assertIn("建议现金防守仓位**: **100.0%", report)
                stock_row = next(line for line in report.splitlines() if "`002594`" in line)
                self.assertIn("**0.0%** | ⏸️ 暂不配置 | `--`", stock_row)

    def test_only_etf_is_eligible_up_to_the_gross_cap(self):
        report = self.generate_plan(hrp=True, stock_eligible=False)
        self.assertIn("建议账户总仓位**: **40.0%", report)
        self.assertIn("无个股权重截断", report)


if __name__ == "__main__":
    unittest.main()
