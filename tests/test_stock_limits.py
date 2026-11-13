"""Regressions for individual stock limits in portfolio targets and plan output."""

import unittest

import numpy as np
import pandas as pd

from src.application.walk_forward_engine import _stock_mask_from_bars, apply_turnover_deadband
from src.domain.portfolio.stock_limits import apply_lot_constraints, limit_stock_weights
from src.infrastructure.market_data import CORE_UNIVERSE, AssetMeta


class TestStockLimits(unittest.TestCase):
    def test_lot_constraints_keep_unallocated_weight_in_cash(self):
        weights = np.array([0.087, 0.10, 0.10, 0.02])
        executable, reasons = apply_lot_constraints(weights, np.array([110.90, 7.00, 5.00, np.nan]), 14000)
        np.testing.assert_allclose(executable, [0, 0.10, 1000 / 14000, 0])
        self.assertEqual(reasons, ["target_below_one_lot", None, None, "reference_price_unavailable"])
        np.testing.assert_array_equal(weights, [0.087, 0.10, 0.10, 0.02])

    def test_lot_constraints_respect_custom_lot_and_validate_targets(self):
        executable, reasons = apply_lot_constraints([0.3], [0.1], 1000, lot=1000)
        np.testing.assert_array_equal(executable, [0.3])
        self.assertEqual(reasons, [None])
        with self.assertRaisesRegex(ValueError, "total NAV"):
            apply_lot_constraints([0.8, 0.4], [1, 1], 1000)

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


if __name__ == "__main__":
    unittest.main()
