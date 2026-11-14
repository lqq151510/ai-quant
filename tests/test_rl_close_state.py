"""The walk-forward RL environment reveals only returns known by each close."""

import unittest

import numpy as np

from src.domain.rl_portfolio.a_share_env import ASharePortfolioEnv


class CloseTransitionTests(unittest.TestCase):
    def test_blocked_opening_purchase_stays_in_cash_and_pays_no_fee(self) -> None:
        alpha = np.zeros((1, 1))
        enabled = np.ones_like(alpha, dtype=bool)
        env = ASharePortfolioEnv(
            alpha, alpha, np.ones_like(alpha), overnight_rets=alpha,
            buy_allowed=~enabled, sell_allowed=enabled,
            buy_cost=0.01,
        )
        env.reset()
        _, _, _, _, info = env.step(np.array([5.0, -5.0]))

        self.assertAlmostEqual(info["blocked_buy_turnover"], 0.95)
        self.assertEqual(info["buy_turnover"], 0.0)
        self.assertEqual(info["turnover"], 0.0)
        self.assertEqual(info["nav"], 1.0)
        self.assertEqual(info["cash_weight"], 1.0)

    def test_blocked_sale_keeps_position_and_does_not_fund_buys(self) -> None:
        alpha = np.zeros((2, 2))
        allowed = np.ones_like(alpha, dtype=bool)
        sell_mask = allowed.copy()
        sell_mask[1, 0] = False
        intraday = np.array([[0.0, 0.0], [-0.20, 0.0]])
        buy_cost = 0.01

        def run(sell_allowed: np.ndarray):
            env = ASharePortfolioEnv(
                alpha, alpha, intraday, overnight_rets=alpha,
                buy_allowed=allowed, sell_allowed=sell_allowed,
                buy_cost=buy_cost, sell_cost=0.02, rebalance_interval=1,
            )
            start, _ = env.reset()
            before, _, _, _, first = env.step(np.array([4.0, -4.0, -5.0]))
            weights_before = env.weights.copy()
            _, _, _, _, second = env.step(np.array([-4.0, 4.0, -5.0]))
            return start, before, first, weights_before, second, env.weights

        start, before, first, opening_w, second, ending_w = run(sell_mask)
        control_start, control_before, _, _, control_second, _ = run(allowed)
        np.testing.assert_array_equal(start, control_start)
        np.testing.assert_array_equal(before, control_before)
        self.assertGreater(second["blocked_sell_turnover"], 0.2)
        self.assertGreater(second["cash_limited_buy_turnover"], 0.2)
        self.assertEqual(second["sell_turnover"], 0.0)
        self.assertGreater(control_second["sell_turnover"], 0.2)
        buy_fill = first["cash_weight"] / (1.0 + buy_cost)
        self.assertAlmostEqual(second["buy_turnover"], buy_fill)
        self.assertAlmostEqual(second["cash_weight"], 0.0)
        expected_growth = 1.0 - buy_cost * buy_fill - 0.20 * opening_w[0]
        self.assertAlmostEqual(second["nav"], first["nav"] * expected_growth)
        self.assertAlmostEqual(ending_w[0] * second["nav"], first["nav"] * 0.8 * opening_w[0])
        self.assertAlmostEqual(ending_w.sum() + second["cash_weight"], 1.0)

    def test_opening_execution_requires_two_aligned_boolean_masks_and_overnight_returns(self) -> None:
        alpha = np.zeros((2, 1))
        allowed = np.ones_like(alpha, dtype=bool)
        with self.assertRaisesRegex(ValueError, "supplied together"):
            ASharePortfolioEnv(alpha, alpha, alpha, overnight_rets=alpha, buy_allowed=allowed)
        with self.assertRaisesRegex(ValueError, "require overnight_rets"):
            ASharePortfolioEnv(alpha, alpha, alpha, buy_allowed=allowed, sell_allowed=allowed)
        for invalid in (allowed[:1], alpha):
            with self.assertRaisesRegex(ValueError, "boolean and match alpha shape"):
                ASharePortfolioEnv(
                    alpha, alpha, alpha, overnight_rets=alpha,
                    buy_allowed=allowed, sell_allowed=invalid,
                )

    def test_old_holdings_earn_overnight_and_no_trade_keeps_drifted_weights(self) -> None:
        alpha = np.zeros((2, 1))
        env = ASharePortfolioEnv(
            alpha, alpha, np.full_like(alpha, -0.5), overnight_rets=np.ones_like(alpha),
            buy_cost=0.01, sell_cost=0.01,
        )
        action = np.array([5.0, -5.0])
        env.reset()
        obs, _, done, _, first = env.step(action)
        self.assertFalse(done)
        self.assertAlmostEqual(first["turnover"], 0.95)
        self.assertAlmostEqual(first["nav"], (1 - 0.0095) * (0.95 * 0.5 + 0.05))
        self.assertAlmostEqual(first["bench_ret"], 0.0)
        self.assertAlmostEqual(obs[2], 0.475 / 0.525)
        first_nav = first["nav"]

        _, _, done, _, second = env.step(action)
        self.assertTrue(done)
        self.assertEqual(second["turnover"], 0.0)
        self.assertAlmostEqual(second["nav"], first_nav)
        self.assertAlmostEqual(second["bench_ret"], 0.0)

    def test_rebalance_turnover_is_measured_from_opening_weights(self) -> None:
        alpha = np.zeros((4, 2))
        overnight = alpha.copy()
        overnight[-1, 0] = 1.0
        env = ASharePortfolioEnv(alpha, alpha, alpha, overnight_rets=overnight, buy_cost=0, sell_cost=0)
        old_action = np.array([-2.0, 2.0, -5.0])
        new_action = np.array([2.0, -2.0, -5.0])
        env.reset()
        for _ in range(3):
            env.step(old_action)

        opening_values = env.weights * (1.0 + overnight[-1])
        opening_growth = opening_values.sum() + env.cash_weight
        opening_weights = opening_values / opening_growth
        new_tilt = 1.0 + 0.30 * np.tanh(new_action[:2])
        target = 0.95 * new_tilt / new_tilt.sum()
        _, _, _, _, info = env.step(new_action)
        self.assertAlmostEqual(info["turnover"], np.abs(target - opening_weights).sum())
        self.assertAlmostEqual(info["net_ret"], opening_growth - 1.0)

    def test_future_return_legs_cannot_change_preceding_observation(self) -> None:
        alpha = np.zeros((3, 1))
        action = np.array([5.0, -5.0])
        def observations(second_overnight: float, second_intraday: float) -> list[np.ndarray]:
            env = ASharePortfolioEnv(
                alpha, alpha, np.array([[0.01], [second_intraday], [0.03]]),
                overnight_rets=np.array([[0.02], [second_overnight], [-0.04]]),
            )
            initial, _ = env.reset()
            return [initial, env.step(action)[0], env.step(action)[0]]

        first = observations(0.03, 0.01)
        changed = observations(0.80, -0.55)
        np.testing.assert_array_equal(first[0], changed[0])
        np.testing.assert_array_equal(first[1], changed[1])
        self.assertFalse(np.array_equal(first[2], changed[2]))

    def test_causal_mode_requires_aligned_finite_return_legs(self) -> None:
        alpha = np.zeros((2, 1))
        for invalid in (np.zeros((1, 1)), np.full_like(alpha, np.nan), np.full_like(alpha, -1)):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "returns must match"):
                ASharePortfolioEnv(alpha, alpha, alpha, overnight_rets=invalid)
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "returns must match"):
                ASharePortfolioEnv(alpha, alpha, invalid, overnight_rets=alpha)
