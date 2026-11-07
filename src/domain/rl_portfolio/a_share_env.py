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
    ) -> None:
        super().__init__()
        assert len(alpha_preds) == len(real_next_rets)
        self.alpha_preds = alpha_preds.astype(np.float32)
        self.vol_features = vol_features.astype(np.float32)
        self.real_next_rets = real_next_rets.astype(np.float32)

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

        # A股 T+1 与波段持仓惯性约束 (Inertia = 0.65 降低无效日内高频换手摩擦)
        if np.sum(self.weights) > 1e-5:
            inertia = 0.65
            target_w = inertia * self.weights + (1.0 - inertia) * raw_target_w
            target_cash = max(0.0, 1.0 - float(np.sum(target_w)))
        else:
            target_w = raw_target_w
            target_cash = raw_target_cash

        # 计算换手与 A 股非对称买卖摩擦成本
        delta_w = target_w - self.weights
        buy_turnover = float(np.sum(np.maximum(delta_w, 0.0)))
        sell_turnover = float(np.sum(np.maximum(-delta_w, 0.0)))
        tx_cost = buy_turnover * self.buy_cost + sell_turnover * self.sell_cost

        # 结算当日真实组合收益
        asset_rets = self.real_next_rets[self.step_idx]
        gross_ret = float(np.dot(target_w, asset_rets))
        net_ret = gross_ret - tx_cost

        # 更新净值与回撤
        self.nav *= (1.0 + net_ret)
        self.peak_nav = max(self.peak_nav, self.nav)
        drawdown = (self.peak_nav - self.nav) / self.peak_nav

        # 价格变动后的自然漂移权重更新
        post_values = target_w * (1.0 + asset_rets)
        total_val = float(np.sum(post_values)) + target_cash
        self.weights = (post_values / max(total_val, 1e-8)).astype(np.float32)
        self.cash_weight = float(target_cash / max(total_val, 1e-8))

        # 奖励塑形: 净超额收益 - 换手惩罚 - 超阈值回撤二次方惩罚
        bench_ret = float(np.mean(asset_rets))
        excess_ret = net_ret - 0.7 * bench_ret
        dd_excess = max(0.0, drawdown - self.dd_threshold)
        reward = float(
            excess_ret * 100.0
            - 0.15 * (buy_turnover + sell_turnover)
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
