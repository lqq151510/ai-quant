"""领域层新增算法 5：离线强化学习 IQL (Implicit Q-Learning) + HRP 层次风险平价与 HMM 状态门控组合决策."""

from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn

from src.domain.portfolio.hrp import HierarchicalRiskParityOptimizer
from src.domain.rl_portfolio.a_share_env import ASharePortfolioEnv
from src.domain.rl_portfolio.ppo_agent import build_alpha_risk_prior


class IQLNetworks(nn.Module):
    """Implicit Q-Learning (Kostrikov et al., ICLR 2022) 的 Q-Network, V-Network 与 Policy Network."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.v_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.q_net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.policy_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, act_dim),
        )


def expectile_loss(diff: torch.Tensor, expectile: float = 0.75) -> torch.Tensor:
    """IQL 核心非对称期望回归损失: L_2^tau(u) = |tau - 1(u < 0)| * u^2."""
    weight = torch.where(diff >= 0, expectile, 1.0 - expectile)
    return (weight * (diff ** 2)).mean()


def train_offline_iql_agent(
    train_env: ASharePortfolioEnv,
    n_behavior_episodes: int = 18,
    iql_epochs: int = 40,
    expectile_tau: float = 0.75,
    awr_temp: float = 3.0,
) -> IQLNetworks:
    """从多风险偏好行为策略采集的离线数据集上，通过 Expectile Regression + AWR 训练 Offline IQL."""
    obs_dim = train_env.observation_space.shape[0]
    act_dim = train_env.action_space.shape[0]
    n_assets = train_env.n_assets

    # 1. 构建包含多种现金防守水位与噪声水平的离线历史数据集 D = {(s, a, r, s', done)}
    s_buf, a_buf, r_buf, sp_buf, d_buf, prior_buf = [], [], [], [], [], []
    cash_biases = [-0.4, 0.0, 0.4, 0.9, 1.4, 2.0]

    for ep in range(n_behavior_episodes):
        obs, _ = train_env.reset()
        done = False
        c_bias = cash_biases[ep % len(cash_biases)]
        noise_scale = 0.25 if ep % 2 == 0 else 0.60

        while not done:
            prior = build_alpha_risk_prior(obs, n_assets).squeeze(0).numpy()
            prior[-1] += c_bias
            act = prior + np.random.normal(0.0, noise_scale, size=act_dim).astype(np.float32)
            next_obs, reward, done, _, _ = train_env.step(act)

            s_buf.append(obs)
            a_buf.append(act - prior)  # 学习相对于先验的残差修正动作
            r_buf.append(reward)
            sp_buf.append(next_obs)
            d_buf.append(float(done))
            prior_buf.append(prior)
            obs = next_obs

    s_t = torch.tensor(np.array(s_buf), dtype=torch.float32)
    a_t = torch.tensor(np.array(a_buf), dtype=torch.float32)
    r_t = torch.tensor(np.array(r_buf), dtype=torch.float32).unsqueeze(-1)
    sp_t = torch.tensor(np.array(sp_buf), dtype=torch.float32)
    d_t = torch.tensor(np.array(d_buf), dtype=torch.float32).unsqueeze(-1)

    r_t = (r_t - r_t.mean()) / (r_t.std() + 1e-6)

    iql = IQLNetworks(obs_dim, act_dim)
    v_opt = torch.optim.AdamW(iql.v_net.parameters(), lr=2e-3, weight_decay=1e-4)
    q_opt = torch.optim.AdamW(iql.q_net.parameters(), lr=2e-3, weight_decay=1e-4)
    pi_opt = torch.optim.AdamW(iql.policy_net.parameters(), lr=2e-3, weight_decay=1e-4)

    gamma = 0.97
    for _ in range(iql_epochs):
        # Step A: Expectile Regression 更新价值上界函数 V(s)
        with torch.no_grad():
            q_val = iql.q_net(torch.cat([s_t, a_t], dim=-1))
        v_pred = iql.v_net(s_t)
        v_loss = expectile_loss(q_val - v_pred, expectile=expectile_tau)
        v_opt.zero_grad()
        v_loss.backward()
        v_opt.step()

        # Step B: 用目标 V(s') 更新动作价值函数 Q(s, a)，完全规避 OOD 动作查询
        with torch.no_grad():
            next_v = iql.v_net(sp_t)
            target_q = r_t + gamma * (1.0 - d_t) * next_v
        q_pred = iql.q_net(torch.cat([s_t, a_t], dim=-1))
        q_loss = nn.SmoothL1Loss()(q_pred, target_q)
        q_opt.zero_grad()
        q_loss.backward()
        q_opt.step()

        # Step C: Advantage-Weighted Regression (AWR) 提取保守策略网络 pi(s)
        with torch.no_grad():
            adv = iql.q_net(torch.cat([s_t, a_t], dim=-1)) - iql.v_net(s_t)
            weights = torch.clamp(torch.exp(awr_temp * adv), max=20.0)
        pi_pred = iql.policy_net(s_t)
        pi_loss = (weights * ((pi_pred - a_t) ** 2)).mean()
        pi_opt.zero_grad()
        pi_loss.backward()
        pi_opt.step()

    return iql


def evaluate_hrp_hmm_iql_portfolio(
    iql_model: IQLNetworks,
    test_alpha: np.ndarray,
    uncertainty_width: np.ndarray,
    train_raw_rets: np.ndarray,
    test_raw_rets: np.ndarray,
    test_vol_features: np.ndarray,
    hmm_max_exposures: List[float],
    top_k: int = 3,
    buy_cost: float = 0.0003,
    sell_cost: float = 0.0008,
) -> Dict[str, np.ndarray | float]:
    """融合 5 大进阶算法的样本外测试集组合回测:
    [Tree+GNN Alpha] + [Quantile Uncertainty 排雷] + [HRP 层次风险平价] + [HMM 状态上限] + [Offline IQL 仓位控制].
    """
    iql_model.eval()
    n_days, n_assets = test_alpha.shape

    # 3日 EMA 平滑 Alpha 与不确定性惩罚
    smoothed_alpha = np.zeros_like(test_alpha, dtype=np.float32)
    for t in range(n_days):
        adj_a = test_alpha[t] - 0.05 * uncertainty_width[t]
        smoothed_alpha[t] = 0.5 * smoothed_alpha[t - 1] + 0.5 * adj_a if t > 0 else adj_a

    all_hist_rets = np.vstack([train_raw_rets[-60:], test_raw_rets])
    weights = np.zeros(n_assets, dtype=np.float32)
    cash_weight = 1.0
    nav = 1.0
    peak_nav = 1.0

    net_rets, bench_rets, cash_weights = [], [], []
    last_target_w = np.zeros(n_assets, dtype=np.float32)

    for t in range(n_days):
        current_dd = (peak_nav - nav) / max(peak_nav, 1e-8)
        obs = np.concatenate([
            smoothed_alpha[t],
            test_vol_features[t],
            weights,
            np.array([cash_weight, current_dd], dtype=np.float32),
        ])

        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            prior = build_alpha_risk_prior(obs, n_assets).squeeze(0).numpy()
            delta_a = iql_model.policy_net(obs_t).squeeze(0).numpy()
            logits = np.clip(prior + 0.30 * delta_a, -10.0, 10.0)
            exp_a = np.exp(logits - np.max(logits))
            raw_alloc = exp_a / np.sum(exp_a)

        if t % 3 == 0 or np.sum(weights) < 1e-5:
            top_idx = np.argsort(smoothed_alpha[t])[-top_k:].tolist()
            # 仅使用截至当日 t 的过去 45 天真实历史收益率计算 HRP 层次风险平价基底权重
            hist_window = all_hist_rets[t: t + 60]
            hrp_base_w = HierarchicalRiskParityOptimizer.allocate_hrp_weights(
                hist_returns=hist_window,
                candidate_indices=top_idx,
                alpha_tilts=smoothed_alpha[t],
            )
            # 结合 HMM 市场状态建议敞口上限与 Offline IQL 现金防守决策
            hmm_cap = hmm_max_exposures[min(t, len(hmm_max_exposures) - 1)]
            iql_cash = float(raw_alloc[-1]) + 0.60 * current_dd
            equity_exposure = float(np.clip(min(1.0 - iql_cash, hmm_cap), 0.45, 0.90))
            target_w = hrp_base_w * equity_exposure
            target_cash = 1.0 - equity_exposure
            last_target_w = target_w.copy()
        else:
            target_w = weights.copy()
            target_cash = cash_weight

        delta_w = target_w - weights
        buy_turn = float(np.sum(np.maximum(delta_w, 0.0)))
        sell_turn = float(np.sum(np.maximum(-delta_w, 0.0)))
        tx_cost = buy_turn * buy_cost + sell_turn * sell_cost

        asset_r = test_raw_rets[t]
        net_r = float(np.dot(target_w, asset_r)) - tx_cost
        bench_r = float(np.mean(asset_r))

        nav *= (1.0 + net_r)
        peak_nav = max(peak_nav, nav)

        post_val = target_w * (1.0 + asset_r)
        tot_v = float(np.sum(post_val)) + target_cash
        weights = (post_val / max(tot_v, 1e-8)).astype(np.float32)
        cash_weight = float(target_cash / max(tot_v, 1e-8))

        net_rets.append(net_r)
        bench_rets.append(bench_r)
        cash_weights.append(cash_weight)

    return {
        "net_rets": np.array(net_rets),
        "bench_rets": np.array(bench_rets),
        "avg_cash_weight": float(np.mean(cash_weights)),
        "latest_hrp_weights": last_target_w,
        "latest_cash": cash_weight,
    }
