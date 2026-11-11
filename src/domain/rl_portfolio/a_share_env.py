"""领域层 Stage 3：符合 A 股微观结构 (T+1 锁仓 / 印花税 / 现金空仓维度) 的强化学习组合优化环境."""

from typing import Dict, Tuple
import gymnasium as gym
from gymnasium import spaces
import numpy as np


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
    ) -> None:
        super().__init__()
        if isinstance(rebalance_interval, bool) or not isinstance(rebalance_interval, int) or rebalance_interval < 1:
            raise ValueError("rebalance_interval must be a positive integer")
        self.rebalance_interval = rebalance_interval
        if overnight_rets is None:
            assert len(alpha_preds) == len(real_next_rets)
        else:
            # 第 t 行包含信号日收盘至下一交易日收盘的两个结算阶段。
            for leg in (overnight_rets, real_next_rets):
                if leg.shape != alpha_preds.shape or not np.all(np.isfinite(leg) & (leg > -1.0)):
                    raise ValueError("overnight and intraday returns must match alpha shape and be finite and greater than -1")
        self.overnight_rets = None if overnight_rets is None else overnight_rets.astype(np.float64)
        # 对齐 3 日预测周期做 EMA 信号平滑，消除单日高频噪音导致的无效换手
        smoothed = np.zeros_like(alpha_preds, dtype=np.float32)
        for t in range(len(alpha_preds)):
            smoothed[t] = 0.5 * smoothed[t - 1] + 0.5 * alpha_preds[t] if t > 0 else alpha_preds[t]
        self.alpha_preds = smoothed
        self.vol_features = vol_features.astype(np.float32)
        self.real_next_rets = real_next_rets.astype(np.float32 if overnight_rets is None else np.float64)

        self.n_steps, self.n_assets = self.alpha_preds.shape
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
        self.weights = np.zeros(self.n_assets, dtype=np.float32)
        self.cash_weight = 1.0
        self.nav = 1.0
        self.peak_nav = 1.0

    def _get_obs(self) -> np.ndarray:
        idx = min(self.step_idx, self.n_steps - 1)
        current_dd = (self.peak_nav - self.nav) / max(self.peak_nav, 1e-8)
        return np.concatenate([
            self.alpha_preds[idx],
            self.vol_features[idx],
            self.weights,
            np.array([self.cash_weight, current_dd], dtype=np.float32),
        ])

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self.step_idx = 0
        self.weights = np.zeros(self.n_assets, dtype=np.float32)
        self.cash_weight = 1.0
        self.nav = 1.0
        self.peak_nav = 1.0
        return self._get_obs(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        # 将动作 logits 转换为合法的组合权重 (含现金维度)
        logits = np.clip(action, -10.0, 10.0)
        exp_a = np.exp(logits - np.max(logits))
        raw_alloc = exp_a / np.sum(exp_a)

        raw_target_w = raw_alloc[:self.n_assets]
        raw_target_cash = float(raw_alloc[-1])

        # 对齐波段调仓周期：结合 Alpha 预测、RL 策略动作 Logits 与当前持仓惯性缓冲选股
        rebalance = False
        if self.step_idx % self.rebalance_interval == 0 or np.sum(self.weights) < 1e-5:
            k_pick = min(3, self.n_assets)
            composite_score = (
                0.60 * self.alpha_preds[self.step_idx]
                + 0.40 * np.tanh(logits[:self.n_assets])
                + 0.35 * (self.weights > 1e-3).astype(np.float32)
            )
            top_idx = np.argsort(composite_score)[-k_pick:]
            top_mask = np.zeros(self.n_assets, dtype=np.float32)
            tilt = 1.0 + 0.30 * np.tanh(logits[top_idx])
            top_mask[top_idx] = tilt / np.sum(tilt)
            # PPO 动态控制总权益仓位敞口与现金防守比例
            current_dd = (self.peak_nav - self.nav) / max(self.peak_nav, 1e-8)
            base_cash = float(np.clip(raw_target_cash + 0.50 * current_dd, 0.05, 0.40))
            equity_exposure = 1.0 - base_cash
            proposed_w = top_mask * equity_exposure
            # 若与当前持仓 L1 偏差处于免交易死区内，则保持原持仓消除摩擦损耗
            if np.sum(self.weights) > 1e-5 and float(np.sum(np.abs(proposed_w - self.weights))) < 0.12:
                target_w = self.weights.copy()
                target_cash = self.cash_weight
            else:
                target_w = proposed_w
                target_cash = base_cash
                rebalance = True
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

        delta_w = target_w - opening_w
        buy_turnover = float(np.sum(np.maximum(delta_w, 0.0)))
        sell_turnover = float(np.sum(np.maximum(-delta_w, 0.0)))
        tx_cost = buy_turnover * self.buy_cost + sell_turnover * self.sell_cost

        post_values = target_w * (1.0 + asset_rets)
        total_val = float(np.sum(post_values)) + target_cash
        if overnight is None:
            # 兼容原有单段收益环境。
            net_ret = float(np.dot(target_w, asset_rets)) - tx_cost
            bench_ret = float(np.mean(asset_rets))
        else:
            # 摩擦按开盘权益计提，剩余资金按目标权重分配后承担日内收益。
            net_ret = opening_growth * (1.0 - tx_cost) * total_val - 1.0
            bench_ret = float(np.mean((1.0 + overnight) * (1.0 + asset_rets) - 1.0))

        self.nav *= (1.0 + net_ret)
        self.peak_nav = max(self.peak_nav, self.nav)
        drawdown = (self.peak_nav - self.nav) / self.peak_nav
        self.weights = (post_values / max(total_val, 1e-8)).astype(np.float32)
        self.cash_weight = float(target_cash / max(total_val, 1e-8))

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
        }
        return self._get_obs(), reward, terminated, False, info
