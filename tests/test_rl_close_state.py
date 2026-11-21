"""The walk-forward RL environment reveals only returns known by each close."""

import unittest

import numpy as np
import pandas as pd

from research_v1.quant_research.ledger import Costs, simulate
from src.application.walk_forward_engine import (
    _opening_execution_masks,
    _weights_from_policy_logits,
    apply_turnover_deadband,
    policy_rebalance_thresholds,
    select_buffered_top_k,
)
from src.domain.rl_portfolio.a_share_env import ASharePortfolioEnv
from src.domain.rl_portfolio.ppo_agent import (
    CASH_LOGIT_NORMAL,
    DRAWDOWN_CASH_COEF,
    DRAWDOWN_LOGIT_COEF,
    DRAWDOWN_LOGIT_START,
    VOL_LOGIT_COEF,
    excess_cash_from_policy_logit,
)


def _softmax_cash(logits: np.ndarray) -> float:
    clipped = np.clip(np.asarray(logits, dtype=np.float64), -10.0, 10.0)
    shifted = np.exp(clipped - np.max(clipped))
    return float(shifted[-1] / np.sum(shifted))


def _ledger_base_cash(logits: np.ndarray, current_dd: float) -> float:
    clipped = np.clip(np.asarray(logits, dtype=np.float64), -10.0, 10.0)
    excess_cash = excess_cash_from_policy_logit(float(clipped[-1]), current_dd)
    return float(np.clip(
        _softmax_cash(clipped) + DRAWDOWN_CASH_COEF * max(0.0, current_dd) + excess_cash,
        0.05,
        0.40,
    ))


def _rebalance_cash(logits: np.ndarray, current_dd: float) -> float:
    """零收益、零费用下，结算后的现金权重等于这次调仓的目标现金。"""
    alpha = np.zeros((1, len(logits) - 1), dtype=np.float64)
    env = ASharePortfolioEnv(
        alpha, alpha, np.zeros_like(alpha), buy_cost=0.0, sell_cost=0.0, rebalance_interval=1,
    )
    env.reset()
    env.peak_nav = 1.0
    env.nav = 1.0 - current_dd
    _, _, _, _, info = env.step(np.asarray(logits, dtype=np.float64))
    return float(info["cash_weight"])


class CloseTransitionTests(unittest.TestCase):
    def test_share_ledger_schedule_waits_after_a_rejected_purchase(self) -> None:
        calendar = pd.bdate_range("2022-01-04", periods=5)
        symbol = "510300"
        bars = pd.DataFrame({
            "date": calendar, "symbol": symbol, "kind": "etf",
            "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
            "volume": 1_000_000.0, "amount": 1_000_000.0,
            "tradable": [True, False, True, True, True],
            "upper_limit": 1.1, "lower_limit": 0.9,
        })
        cost = Costs(commission_rate=0.0, minimum_commission=0.0, slippage_bps=0.0)
        interval = 3
        scheduled_closes = {day.date() for day in calendar[:-1:interval]}
        ledger = simulate(
            bars, calendar, {}, initial_cash=100_000.0, cost=cost,
            close_policy=lambda state: {symbol: 0.9} if state.date in scheduled_closes else None,
        )
        self.assertEqual(ledger.orders["status"].tolist(), ["rejected", "filled"])
        self.assertEqual(ledger.orders.iloc[0]["reason"], "suspended")
        filled_dates = ledger.orders.loc[ledger.orders["filled_qty"] > 0, "date"].tolist()
        self.assertEqual(filled_dates, [calendar[-1].date()])

        execution_dates = calendar[1:]
        masks = _opening_execution_masks(bars, execution_dates, [symbol], cost)
        zeros = np.zeros((len(execution_dates), 1))
        action = np.array([5.0, -5.0])
        for ledger_mode in (True, False):
            with self.subTest(ledger_mode=ledger_mode):
                ledger_options = (
                    {"hmm_exposure": np.ones(len(execution_dates)), "stock_mask": np.zeros(1, dtype=bool)}
                    if ledger_mode else {}
                )
                env = ASharePortfolioEnv(
                    zeros, zeros, zeros, overnight_rets=zeros,
                    buy_cost=0.0, sell_cost=0.0, rebalance_interval=interval, top_k=1,
                    **masks, **ledger_options,
                )
                env.reset()
                transitions = [env.step(action)[-1] for _ in execution_dates]
                self.assertGreater(transitions[0]["blocked_buy_turnover"], 0.0)
                self.assertEqual(transitions[0]["cash_weight"], 1.0)
                env_fill_dates = [
                    day.date() for day, info in zip(execution_dates, transitions)
                    if info["buy_turnover"] > 0
                ]
                self.assertEqual(env_fill_dates, filled_dates if ledger_mode else [calendar[2].date()])

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
        # 同一持仓经账本死区混合后单边换手约 0.19，卖出仍被挡住，买入仍受现金约束。
        self.assertGreater(second["blocked_sell_turnover"], 0.1)
        self.assertGreater(second["cash_limited_buy_turnover"], 0.1)
        self.assertEqual(second["sell_turnover"], 0.0)
        self.assertGreater(control_second["sell_turnover"], 0.1)
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

        close_weights = np.asarray(env.weights, dtype=np.float64)
        opening_values = close_weights * (1.0 + overnight[-1])
        opening_growth = opening_values.sum() + env.cash_weight
        opening_weights = opening_values / opening_growth
        new_tilt = 1.0 + 0.30 * np.tanh(new_action[:2])
        proposed = 0.95 * new_tilt / new_tilt.sum()
        l1_deadband, inertia_blend, _, _ = policy_rebalance_thresholds(env.n_assets)
        adjusted, should_trade = apply_turnover_deadband(
            proposed, close_weights, l1_deadband=l1_deadband, inertia_blend=inertia_blend,
        )
        self.assertTrue(should_trade)
        _, _, _, _, info = env.step(new_action)
        self.assertAlmostEqual(info["turnover"], np.abs(adjusted - opening_weights).sum())
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

    def test_rebalance_cash_matches_policy_excess_cash(self) -> None:
        asset_logits = np.array([1.0, 0.2, -0.4])
        drawdown = 0.15
        normal_logit = CASH_LOGIT_NORMAL
        hot_logit = CASH_LOGIT_NORMAL + VOL_LOGIT_COEF * 2.0
        prior_logit = CASH_LOGIT_NORMAL + DRAWDOWN_LOGIT_COEF * (drawdown - DRAWDOWN_LOGIT_START)
        normal = np.concatenate([asset_logits, [normal_logit]])
        hot = np.concatenate([asset_logits, [hot_logit]])
        prior = np.concatenate([asset_logits, [prior_logit]])

        for logits, current_dd in (
            (normal, 0.0),
            (hot, 0.0),
            (normal, drawdown),
            (hot, drawdown),
            (prior, drawdown),
        ):
            with self.subTest(cash_logit=float(logits[-1]), drawdown=current_dd):
                cash = _rebalance_cash(logits, current_dd)
                self.assertGreater(cash, 0.05)
                self.assertLess(cash, 0.40)
                self.assertAlmostEqual(cash, _ledger_base_cash(logits, current_dd), places=5)

        calm = _rebalance_cash(normal, 0.0)
        stressed = _rebalance_cash(hot, 0.0)
        self.assertAlmostEqual(excess_cash_from_policy_logit(normal_logit, 0.0), 0.0, places=12)
        self.assertGreater(excess_cash_from_policy_logit(hot_logit, 0.0), 0.0)
        self.assertGreater(stressed, calm)
        self.assertGreater(_rebalance_cash(hot, drawdown), _rebalance_cash(normal, drawdown))

        prior_cash = _rebalance_cash(prior, drawdown)
        raw_cash = _softmax_cash(prior)
        single_count = float(np.clip(raw_cash + DRAWDOWN_CASH_COEF * drawdown, 0.05, 0.40))
        repeated_drawdown = max(0.0, prior_logit - CASH_LOGIT_NORMAL) * (
            DRAWDOWN_CASH_COEF / DRAWDOWN_LOGIT_COEF
        )
        counted_twice = float(np.clip(
            raw_cash + DRAWDOWN_CASH_COEF * drawdown + repeated_drawdown, 0.05, 0.40,
        ))
        self.assertAlmostEqual(excess_cash_from_policy_logit(prior_logit, drawdown), 0.0, places=12)
        self.assertGreater(repeated_drawdown, 0.0)
        self.assertAlmostEqual(prior_cash, single_count, places=5)
        self.assertLess(prior_cash, counted_twice)

    def test_rebalance_selection_and_trade_match_ledger_operators(self) -> None:
        self.assertEqual(policy_rebalance_thresholds(4), (0.16, 0.35, 0.55, 1))
        self.assertEqual(policy_rebalance_thresholds(28), (0.22, 0.45, 1.15, 2))

        # 同一持仓和分数下，0.55 的换仓优势会留下默认 0.45 会换掉的旧仓。
        scores = np.array([1.00, 0.98, 0.96, 0.50])
        held = np.array([0.0, 0.0, 1.0, 1.0])
        alpha = ((scores - 0.35 * held) / 0.60).reshape(1, 4)
        logits = np.array([0.0, 0.0, 0.0, 0.0, -5.0])
        prev = np.array([0.0, 0.0, 0.50, 0.50])
        env = ASharePortfolioEnv(
            alpha, np.zeros_like(alpha), np.zeros_like(alpha),
            buy_cost=0.0, sell_cost=0.0, rebalance_interval=1, top_k=2, smooth_alpha=False,
        )
        env.reset()
        env.weights = prev.copy()
        env.cash_weight = 0.0
        composite = (
            0.60 * env.alpha_preds[0]
            + 0.40 * np.tanh(logits[:4])
            + 0.35 * (prev > 1e-3)
        )
        l1_deadband, inertia_blend, swap_margin, max_drop = policy_rebalance_thresholds(env.n_assets)
        expected = select_buffered_top_k(
            composite, prev, env.top_k, swap_margin=swap_margin, max_drop=max_drop,
        )
        default_selected = select_buffered_top_k(composite, prev, env.top_k, swap_margin=0.45)
        self.assertEqual(expected.tolist(), [2, 3])
        self.assertNotEqual(default_selected.tolist(), expected.tolist())
        equity = 1.0 - _ledger_base_cash(logits, 0.0)
        tilt = 1.0 + 0.30 * np.tanh(logits[expected])
        proposed = np.zeros(4, dtype=np.float64)
        proposed[expected] = equity * tilt / np.sum(tilt)
        _, expected_trade = apply_turnover_deadband(
            proposed, prev, l1_deadband=l1_deadband, inertia_blend=inertia_blend,
        )
        _, _, _, _, info = env.step(logits)
        self.assertEqual(info["selected_assets"], tuple(expected.tolist()))
        self.assertEqual(info["traded"], expected_trade)
        self.assertFalse(expected_trade)

        # 宽池死区 0.22 会压住这笔调仓，函数默认 0.16 会成交。
        resize_logits = np.array([2.0, -1.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0])
        resize_alpha = np.zeros((1, 8))
        resize_equity = 1.0 - _ledger_base_cash(resize_logits, 0.0)
        resize_tilt = 1.0 + 0.30 * np.tanh(resize_logits[:2])
        resize_proposed = np.zeros(8, dtype=np.float64)
        resize_proposed[:2] = resize_equity * resize_tilt / np.sum(resize_tilt)
        resize_prev = resize_proposed.copy()
        resize_prev[0] -= 0.15
        resize_prev[1] += 0.15
        self.assertGreater(float(resize_prev[0]), 0.02)
        resize_env = ASharePortfolioEnv(
            resize_alpha, resize_alpha, resize_alpha,
            buy_cost=0.0, sell_cost=0.0, rebalance_interval=1, top_k=2, smooth_alpha=False,
        )
        resize_env.reset()
        resize_env.weights = resize_prev.copy()
        resize_env.cash_weight = float(1.0 - resize_prev.sum())
        wide_deadband, wide_blend, wide_swap, wide_drop = policy_rebalance_thresholds(resize_env.n_assets)
        resize_scores = (
            0.60 * resize_env.alpha_preds[0]
            + 0.40 * np.tanh(resize_logits[:8])
            + 0.35 * (resize_prev > 1e-3)
        )
        resize_selected = select_buffered_top_k(
            resize_scores, resize_prev, resize_env.top_k, swap_margin=wide_swap, max_drop=wide_drop,
        )
        self.assertEqual(resize_selected.tolist(), [0, 1])
        _, resize_trade = apply_turnover_deadband(
            resize_proposed, resize_prev, l1_deadband=wide_deadband, inertia_blend=wide_blend,
        )
        _, default_trade = apply_turnover_deadband(
            resize_proposed, resize_prev, l1_deadband=0.16, inertia_blend=0.40,
        )
        self.assertFalse(resize_trade)
        self.assertTrue(default_trade)
        self.assertGreater(float(np.abs(resize_proposed - resize_prev).sum()), 0.12)
        _, _, _, _, resize_info = resize_env.step(resize_logits)
        self.assertEqual(resize_info["selected_assets"], tuple(resize_selected.tolist()))
        self.assertEqual(resize_info["traded"], resize_trade)

    def test_training_target_matches_share_ledger_hmm_and_stock_caps(self) -> None:
        alpha = np.array([[1.0, 0.2, -0.4, -1.0]])
        logits = np.array([1.2, 0.4, -1.0, -1.5, -2.0])
        hmm = np.array([0.70])
        no_stocks = np.zeros(4, dtype=bool)

        def settled_weights(env: ASharePortfolioEnv) -> tuple[np.ndarray, float]:
            env.reset()
            _, _, _, _, info = env.step(logits)
            self.assertTrue(info["traded"])
            return np.asarray(env.weights, dtype=np.float64), float(env.cash_weight)

        constrained = ASharePortfolioEnv(
            alpha, np.zeros_like(alpha), np.zeros_like(alpha),
            buy_cost=0.0, sell_cost=0.0, rebalance_interval=1, top_k=2, smooth_alpha=False,
            hmm_exposure=hmm, stock_mask=no_stocks,
        )
        plain = ASharePortfolioEnv(
            alpha, np.zeros_like(alpha), np.zeros_like(alpha),
            buy_cost=0.0, sell_cost=0.0, rebalance_interval=1, top_k=2, smooth_alpha=False,
        )
        capped_weights, capped_cash = settled_weights(constrained)
        plain_weights, _ = settled_weights(plain)
        l1_deadband, inertia_blend, swap_margin, max_drop = policy_rebalance_thresholds(4)
        proposed = _weights_from_policy_logits(
            logits, alpha[0], np.zeros(4), 0.0, 0.70, top_k=2,
            stock_mask=no_stocks, swap_margin=swap_margin, max_drop=max_drop,
        )
        expected, trade = apply_turnover_deadband(
            proposed, np.zeros(4), l1_deadband=l1_deadband, inertia_blend=inertia_blend,
            max_exposure=0.70,
        )
        self.assertTrue(trade)
        np.testing.assert_allclose(capped_weights, expected, atol=1e-6)
        self.assertAlmostEqual(capped_cash, 1.0 - float(expected.sum()), places=5)
        self.assertLessEqual(float(capped_weights.sum()), 0.64 + 1e-6)
        self.assertGreater(float(plain_weights.sum()), float(capped_weights.sum()))

        broad_alpha = np.zeros((1, 8))
        broad_alpha[0, :3] = (2.0, 1.5, 1.0)
        broad_logits = np.array([3.0, 2.5, 2.0, -1.0, -1.0, -1.0, 0.2, -2.0, -4.0])
        stock_mask = np.array([True, True, True, True, True, True, False, False])
        broad = ASharePortfolioEnv(
            broad_alpha, np.zeros_like(broad_alpha), np.zeros_like(broad_alpha),
            buy_cost=0.0, sell_cost=0.0, rebalance_interval=1, top_k=3, smooth_alpha=False,
            hmm_exposure=np.array([1.0]), stock_mask=stock_mask,
        )
        broad.reset()
        _, _, _, _, info = broad.step(broad_logits)
        self.assertTrue(info["traded"])
        weights = np.asarray(broad.weights, dtype=np.float64)
        wide_l1, wide_blend, wide_swap, wide_drop = policy_rebalance_thresholds(8)
        broad_proposed = _weights_from_policy_logits(
            broad_logits, broad_alpha[0], np.zeros(8), 0.0, 1.0, top_k=3,
            stock_mask=stock_mask, swap_margin=wide_swap, max_drop=wide_drop,
        )
        broad_expected, broad_trade = apply_turnover_deadband(
            broad_proposed, np.zeros(8), l1_deadband=wide_l1, inertia_blend=wide_blend,
            max_exposure=1.0, stock_mask=stock_mask,
        )
        self.assertTrue(broad_trade)
        np.testing.assert_allclose(weights, broad_expected, atol=1e-6)
        self.assertTrue((weights[stock_mask] <= 0.05 + 1e-6).all())
        self.assertAlmostEqual(float(weights.sum()) + float(broad.cash_weight), 1.0, places=5)

        with self.assertRaisesRegex(ValueError, "supplied together"):
            ASharePortfolioEnv(alpha, alpha, np.zeros_like(alpha), hmm_exposure=hmm)
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            ASharePortfolioEnv(
                alpha, alpha, np.zeros_like(alpha),
                hmm_exposure=np.array([1.2]), stock_mask=no_stocks,
            )

    def test_session_cost_rates_replace_the_scalar_buy_cost(self) -> None:
        alpha = np.zeros((1, 1))
        action = np.array([5.0, -5.0])
        rated = ASharePortfolioEnv(
            alpha, alpha, np.zeros_like(alpha),
            buy_cost=0.0003, sell_cost=0.0008,
            buy_cost_rates=np.array([[0.0008]]), sell_cost_rates=np.array([[0.0008]]),
            rebalance_interval=1, top_k=1, smooth_alpha=False,
        )
        scalar = ASharePortfolioEnv(
            alpha, alpha, np.zeros_like(alpha),
            buy_cost=0.0003, sell_cost=0.0008,
            rebalance_interval=1, top_k=1, smooth_alpha=False,
        )
        rated.reset()
        scalar.reset()
        _, _, _, _, rated_info = rated.step(action)
        _, _, _, _, scalar_info = scalar.step(action)
        self.assertAlmostEqual(rated_info["turnover"], 0.95)
        self.assertAlmostEqual(rated_info["nav"], 1.0 - 0.95 * 0.0008)
        self.assertAlmostEqual(scalar_info["nav"], 1.0 - 0.95 * 0.0003)
        self.assertGreater(scalar_info["nav"], rated_info["nav"])
