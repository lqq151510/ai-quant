"""领域层 Stage 3：符合 A 股微观结构 (T+1 锁仓 / 印花税 / 现金空仓维度) 的强化学习组合优化环境."""

from typing import Dict, Tuple
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from src.domain.rl_portfolio.ppo_agent import DRAWDOWN_CASH_COEF, excess_cash_from_policy_logit


def _ledger_rebalance_operators():
    """账本算子在应用层，应用层又会导入本环境。延迟导入只为避开循环依赖。"""
    from src.application.walk_forward_engine import (
        apply_turnover_deadband,
        policy_rebalance_thresholds,
        select_buffered_top_k,
    )

    return select_buffered_top_k, apply_turnover_deadband, policy_rebalance_thresholds


class ASharePortfolioEnv(gym.Env):
    """A股动态仓位管理与风险控制 Gym 环境.

    核心实战特性：
    1. 包含显式「现金仓位 (Cash)」维度：允许智能体在高波动或退潮期主动降仓/空仓防守。
    2. 严格执行双边交易成本：买入佣金 + 卖出印花税与滑点惩罚。
    3. 非对称回撤惩罚奖励函数：在追求超额收益的同时严控最大回撤 (Max Drawdown)。
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        alpha_preds: np.ndarray,
        vol_features: np.ndarray,
        real_next_rets: np.ndarray,
        buy_cost: float = 0.0003,
        sell_cost: float = 0.0008,
        dd_penalty_weight: float = 2.5,
        dd_threshold: float = 0.03,
        *,
        overnight_rets: np.ndarray | None = None,
        rebalance_interval: int = 3,
        top_k: int = 3,
        buy_allowed: np.ndarray | None = None,
        sell_allowed: np.ndarray | None = None,
        smooth_alpha: bool = True,
        hmm_exposure: np.ndarray | None = None,
        stock_mask: np.ndarray | None = None,
        buy_cost_rates: np.ndarray | None = None,
        sell_cost_rates: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        if isinstance(rebalance_interval, bool) or not isinstance(rebalance_interval, int) or rebalance_interval < 1:
            raise ValueError("rebalance_interval must be a positive integer")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        self.rebalance_interval = rebalance_interval
        self.top_k = top_k
        if overnight_rets is None:
            assert len(alpha_preds) == len(real_next_rets)
        else:
            # 第 t 行包含信号日收盘至下一交易日收盘的两个结算阶段。
            for leg in (overnight_rets, real_next_rets):
                if leg.shape != alpha_preds.shape or not np.all(np.isfinite(leg) & (leg > -1.0)):
                    raise ValueError("overnight and intraday returns must match alpha shape and be finite and greater than -1")
        if (buy_allowed is None) != (sell_allowed is None):
            raise ValueError("buy_allowed and sell_allowed must be supplied together")
        if buy_allowed is not None and sell_allowed is not None:
            if overnight_rets is None:
                raise ValueError("opening execution masks require overnight_rets")
            for mask in (buy_allowed, sell_allowed):
                if mask.shape != alpha_preds.shape or mask.dtype != np.bool_:
                    raise ValueError("opening execution masks must be boolean and match alpha shape")
            if not np.isfinite(buy_cost) or buy_cost < 0 or not 0 <= sell_cost < 1:
                raise ValueError("opening execution requires nonnegative costs and sell_cost below one")
        self.buy_allowed = None if buy_allowed is None else buy_allowed.copy()
        self.sell_allowed = None if sell_allowed is None else sell_allowed.copy()
        if (hmm_exposure is None) != (stock_mask is None):
            raise ValueError("hmm_exposure and stock_mask must be supplied together")
        self.hmm_exposure = None if hmm_exposure is None else np.asarray(hmm_exposure, dtype=np.float64).copy()
        self.stock_mask = None if stock_mask is None else np.asarray(stock_mask).copy()
        if (buy_cost_rates is None) != (sell_cost_rates is None):
            raise ValueError("buy_cost_rates and sell_cost_rates must be supplied together")
        self.buy_cost_rates = None if buy_cost_rates is None else np.asarray(buy_cost_rates, dtype=np.float64).copy()
        self.sell_cost_rates = None if sell_cost_rates is None else np.asarray(sell_cost_rates, dtype=np.float64).copy()
        self.overnight_rets = None if overnight_rets is None else overnight_rets.astype(np.float64)
        # 调用方已经做因果平滑时必须关掉，避免训练观测和样本外决策用两套 Alpha。
        if smooth_alpha:
            smoothed = np.zeros_like(alpha_preds, dtype=np.float32)
            for t in range(len(alpha_preds)):
                smoothed[t] = 0.5 * smoothed[t - 1] + 0.5 * alpha_preds[t] if t > 0 else alpha_preds[t]
            self.alpha_preds = smoothed
        else:
            self.alpha_preds = np.asarray(alpha_preds, dtype=np.float32)
        self.vol_features = vol_features.astype(np.float32)
        self.real_next_rets = real_next_rets.astype(np.float32 if overnight_rets is None else np.float64)

        self.n_steps, self.n_assets = self.alpha_preds.shape
        if self.hmm_exposure is not None and (
            self.hmm_exposure.shape != (self.n_steps,)
            or not np.isfinite(self.hmm_exposure).all()
            or (self.hmm_exposure < 0).any()
            or (self.hmm_exposure > 1).any()
        ):
            raise ValueError("hmm_exposure must be a finite per-session cap between 0 and 1")
        if self.stock_mask is not None and (
            self.stock_mask.shape != (self.n_assets,) or self.stock_mask.dtype != np.bool_
        ):
            raise ValueError("stock_mask must be a boolean vector aligned to assets")
        if self.buy_cost_rates is not None and (
            self.buy_cost_rates.shape != (self.n_steps, self.n_assets)
            or self.sell_cost_rates.shape != (self.n_steps, self.n_assets)
            or not np.isfinite(self.buy_cost_rates).all()
            or not np.isfinite(self.sell_cost_rates).all()
            or (self.buy_cost_rates < 0).any()
            or (self.sell_cost_rates < 0).any()
            or (self.sell_cost_rates >= 1).any()
        ):
            raise ValueError("turnover cost rates must be finite, nonnegative, below one on sells, and match the session-asset shape")
        self.buy_cost = buy_cost
        self.sell_cost = sell_cost
        self.dd_penalty_weight = dd_penalty_weight
        self.dd_threshold = dd_threshold

        # 状态维度: Alpha预测(N) + 波动率特征(N) + 当前持仓权重(N) + 现金权重(1) + 当前回撤(1)
        obs_dim = self.n_assets * 3 + 2
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32
        )
        # 动作维度: N 只标的 + 1 个现金仓位 (经过 Softmax 映射为目标权重)
        self.action_space = spaces.Box(
            low=-5.0, high=5.0, shape=(self.n_assets + 1,), dtype=np.float32
        )

        self.step_idx = 0
        self.weights = np.zeros(self.n_assets, dtype=np.float64 if self.buy_allowed is not None else np.float32)
        self.cash_weight = 1.0
        self.nav = 1.0
        self.peak_nav = 1.0

    def _get_obs(self) -> np.ndarray:
        idx = min(self.step_idx, self.n_steps - 1)
        current_dd = (self.peak_nav - self.nav) / max(self.peak_nav, 1e-8)
        return np.concatenate([
            self.alpha_preds[idx],
            self.vol_features[idx],
            self.weights.astype(np.float32, copy=False),
            np.array([self.cash_weight, current_dd], dtype=np.float32),
        ])

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self.step_idx = 0
        self.weights = np.zeros(self.n_assets, dtype=np.float64 if self.buy_allowed is not None else np.float32)
        self.cash_weight = 1.0
        self.nav = 1.0
        self.peak_nav = 1.0
        return self._get_obs(), {}

    def _turnover_rates(self) -> tuple[np.ndarray, np.ndarray]:
        if self.buy_cost_rates is None or self.sell_cost_rates is None:
            return (
                np.full(self.n_assets, self.buy_cost, dtype=np.float64),
                np.full(self.n_assets, self.sell_cost, dtype=np.float64),
            )
        return self.buy_cost_rates[self.step_idx], self.sell_cost_rates[self.step_idx]

    def _execute_opening(
        self, opening_w: np.ndarray, opening_cash: float, target_w: np.ndarray,
        buy_allowed: np.ndarray, sell_allowed: np.ndarray,
    ) -> tuple[np.ndarray, float, dict[str, float]]:
        """Execute fractionally in units of opening NAV and debit fees from actual cash."""
        buy_rates, sell_rates = self._turnover_rates()
        delta = target_w - opening_w
        desired_buys = np.maximum(delta, 0.0)
        desired_sells = np.maximum(-delta, 0.0)
        sells = np.where(sell_allowed, desired_sells, 0.0)
        buys = np.where(buy_allowed, desired_buys, 0.0)
        sell_turnover = float(sells.sum())
        cash = opening_cash + float(np.sum(sells * (1.0 - sell_rates)))
        eligible_buy_turnover = float(buys.sum())
        buy_gross = float(np.sum(buys * (1.0 + buy_rates)))
        if buy_gross > cash and buy_gross > 0.0:
            buys *= cash / buy_gross
        buy_turnover = float(buys.sum())
        cash = max(0.0, cash - float(np.sum(buys * (1.0 + buy_rates))))
        return opening_w - sells + buys, cash, {
            "buy_turnover": buy_turnover,
            "sell_turnover": sell_turnover,
            "blocked_buy_turnover": float(desired_buys[~buy_allowed].sum()),
            "blocked_sell_turnover": float(desired_sells[~sell_allowed].sum()),
            "cash_limited_buy_turnover": max(0.0, eligible_buy_turnover - buy_turnover),
        }

    def _share_ledger_target(self, logits: np.ndarray) -> tuple[np.ndarray, float, bool, tuple[int, ...]]:
        """调仓目标与份额账本同一套：HMM 上限、个股约束，再走免交易死区。"""
        from src.application.walk_forward_engine import _weights_from_policy_logits

        _, apply_turnover_deadband, policy_rebalance_thresholds = _ledger_rebalance_operators()
        l1_deadband, inertia_blend, swap_margin, max_drop = policy_rebalance_thresholds(self.n_assets)
        prev_w = np.asarray(self.weights, dtype=np.float64)
        current_dd = (self.peak_nav - self.nav) / max(self.peak_nav, 1e-8)
        hmm_cap = float(self.hmm_exposure[self.step_idx])
        proposed = _weights_from_policy_logits(
            logits,
            np.asarray(self.alpha_preds[self.step_idx], dtype=np.float64),
            prev_w,
            current_dd,
            hmm_cap,
            min(self.top_k, self.n_assets),
            stock_mask=self.stock_mask,
            swap_margin=swap_margin,
            max_drop=max_drop,
        )
        limit_stocks = bool(np.any(self.stock_mask))
        adjusted, should_trade = apply_turnover_deadband(
            proposed,
            prev_w,
            l1_deadband=l1_deadband,
            inertia_blend=inertia_blend,
            max_exposure=hmm_cap,
            stock_mask=self.stock_mask if limit_stocks else None,
        )
        selected = tuple(int(idx) for idx in np.flatnonzero(proposed > 1e-4))
        if not should_trade:
            return self.weights.copy(), float(self.cash_weight), False, selected
        target_w = np.asarray(adjusted, dtype=np.float64)
        return target_w, max(0.0, 1.0 - float(target_w.sum())), True, selected

    def _unconstrained_rebalance_target(
        self, logits: np.ndarray, raw_target_cash: float,
    ) -> tuple[np.ndarray, float, bool, tuple[int, ...]]:
        """没有传入账本约束时，只对齐选股、现金和免交易，不套 HMM 与个股上限。"""
        select_buffered_top_k, apply_turnover_deadband, policy_rebalance_thresholds = _ledger_rebalance_operators()
        l1_deadband, inertia_blend, swap_margin, max_drop = policy_rebalance_thresholds(self.n_assets)
        k_pick = min(max(1, self.top_k), self.n_assets)
        asset_logits = logits[:self.n_assets]
        prev_w = np.asarray(self.weights, dtype=np.float64)
        composite_score = (
            0.60 * self.alpha_preds[self.step_idx]
            + 0.40 * np.tanh(asset_logits)
            + 0.35 * (prev_w > 1e-3)
        )
        top_idx = select_buffered_top_k(
            composite_score, prev_w, k_pick, swap_margin=swap_margin, max_drop=max_drop,
        )
        top_mask = np.zeros(self.n_assets, dtype=np.float64)
        tilt = 1.0 + 0.30 * np.tanh(asset_logits[top_idx])
        top_mask[top_idx] = tilt / np.sum(tilt)
        current_dd = (self.peak_nav - self.nav) / max(self.peak_nav, 1e-8)
        cash_logit = float(logits[-1])
        excess_cash = excess_cash_from_policy_logit(cash_logit, current_dd)
        base_cash = float(np.clip(
            raw_target_cash + DRAWDOWN_CASH_COEF * max(0.0, current_dd) + excess_cash,
            0.05,
            0.40,
        ))
        proposed_w = top_mask * (1.0 - base_cash)
        adjusted_w, should_trade = apply_turnover_deadband(
            proposed_w, prev_w, l1_deadband=l1_deadband, inertia_blend=inertia_blend,
        )
        selected = tuple(int(idx) for idx in top_idx)
        if not should_trade:
            return self.weights.copy(), float(self.cash_weight), False, selected
        return np.asarray(adjusted_w, dtype=np.float64), base_cash, True, selected

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        # 将动作 logits 转换为合法的组合权重 (含现金维度)
        logits = np.clip(action, -10.0, 10.0)
        exp_a = np.exp(logits - np.max(logits))
        raw_alloc = exp_a / np.sum(exp_a)

        raw_target_cash = float(raw_alloc[-1])

        # 传入 HMM 与个股掩码时，目标权重与份额账本同一套；否则只对齐选股、现金和免交易。
        selected_assets: tuple[int, ...] = ()
        rebalance = False
        ledger_mode = self.hmm_exposure is not None
        scheduled_rebalance = self.step_idx % self.rebalance_interval == 0
        # 份额账本在拒单后等到下一次预定调仓，即使持仓仍为零。
        if scheduled_rebalance or (not ledger_mode and np.sum(self.weights) < 1e-5):
            if ledger_mode:
                target_w, target_cash, rebalance, selected_assets = self._share_ledger_target(logits)
            else:
                target_w, target_cash, rebalance, selected_assets = self._unconstrained_rebalance_target(
                    logits, raw_target_cash,
                )
        else:
            target_w = self.weights.copy()
            target_cash = self.cash_weight

        asset_rets = self.real_next_rets[self.step_idx]
        opening_w = self.weights
        opening_growth = 1.0
        overnight = None if self.overnight_rets is None else self.overnight_rets[self.step_idx]
        if overnight is not None:
            # 旧仓承担隔夜收益，收盘决策在下一开盘才生效。
            opening_values = self.weights * (1.0 + overnight)
            opening_growth = float(np.sum(opening_values)) + self.cash_weight
            opening_w = opening_values / opening_growth
            if not rebalance:
                target_w = opening_w
                target_cash = self.cash_weight / opening_growth

        execution_info = {}
        gated = self.buy_allowed is not None and self.sell_allowed is not None
        if gated:
            target_w, target_cash, execution_info = self._execute_opening(
                opening_w, self.cash_weight / opening_growth, target_w,
                self.buy_allowed[self.step_idx], self.sell_allowed[self.step_idx],
            )
            buy_turnover = execution_info["buy_turnover"]
            sell_turnover = execution_info["sell_turnover"]
            # The cash balance already includes the fees for the trades that filled.
            tx_cost = 0.0
        else:
            delta_w = target_w - opening_w
            buy_rates, sell_rates = self._turnover_rates()
            buys = np.maximum(delta_w, 0.0)
            sells = np.maximum(-delta_w, 0.0)
            buy_turnover = float(buys.sum())
            sell_turnover = float(sells.sum())
            tx_cost = float(np.sum(buys * buy_rates + sells * sell_rates))

        post_values = target_w * (1.0 + asset_rets)
        total_val = float(np.sum(post_values)) + target_cash
        if overnight is None:
            # 兼容原有单段收益环境。
            net_ret = float(np.dot(target_w, asset_rets)) - tx_cost
            bench_ret = float(np.mean(asset_rets))
        else:
            net_ret = opening_growth * (1.0 - tx_cost) * total_val - 1.0
            bench_ret = float(np.mean((1.0 + overnight) * (1.0 + asset_rets) - 1.0))

        self.nav *= opening_growth * total_val if gated else (1.0 + net_ret)
        self.peak_nav = max(self.peak_nav, self.nav)
        drawdown = (self.peak_nav - self.nav) / self.peak_nav
        close_value = total_val if gated else max(total_val, 1e-8)
        self.weights = (post_values / close_value).astype(self.weights.dtype)
        self.cash_weight = float(target_cash / close_value)

        # 奖励塑形: 净超额收益 - 强化换手惩罚 - 超阈值回撤二次方惩罚
        excess_ret = net_ret - 0.7 * bench_ret
        dd_excess = max(0.0, drawdown - self.dd_threshold)
        reward = float(
            excess_ret * 100.0
            - 0.45 * (buy_turnover + sell_turnover)
            - self.dd_penalty_weight * (dd_excess ** 2) * 100.0
        )

        self.step_idx += 1
        terminated = self.step_idx >= self.n_steps
        info = {
            "net_ret": net_ret,
            "bench_ret": bench_ret,
            "turnover": buy_turnover + sell_turnover,
            "cash_weight": self.cash_weight,
            "nav": self.nav,
            "drawdown": drawdown,
            "selected_assets": selected_assets,
            "traded": rebalance,
            **execution_info,
        }
        return self._get_obs(), reward, terminated, False, info
