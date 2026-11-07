"""领域层 Stage 3 进阶：基于 PPO (Proximal Policy Optimization) + GAE 的 A股动态仓位管理智能体."""

from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn

from src.domain.rl_portfolio.a_share_env import ASharePortfolioEnv


def build_alpha_risk_prior(obs: np.ndarray, n_assets: int, top_k: int = 3) -> torch.Tensor:
    """结合截面 Alpha 排名、波动率与当前回撤构建风险感知先验偏置."""
    alphas = obs[:n_assets].copy()
    vols = obs[n_assets:2 * n_assets]
    current_dd = float(obs[-1])

    # 聚焦截面 Top-K 高置信度标的，抑制尾部噪音标的
    threshold = np.sort(alphas)[-top_k]
    masked_alpha = np.where(alphas >= threshold, alphas * 2.8 + 1.8, -5.0)

    # 动态现金防守水位：当处于回撤期 (>5%) 或市场截面波动率飙升时提高现金先验
    mean_vol = float(np.mean(vols))
    cash_logit = 0.5 + 10.0 * max(0.0, current_dd - 0.04) + 0.5 * max(0.0, mean_vol)
    prior = np.concatenate([masked_alpha, np.array([cash_logit], dtype=np.float32)])
    return torch.tensor(prior, dtype=torch.float32).unsqueeze(0)


class PPOPortfolioActorCritic(nn.Module):
    """PPO 策略与价值网络 (Actor-Critic Dual Network with Orthogonal Init)."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.actor_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.critic_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.log_std = nn.Parameter(torch.full((1, action_dim), -0.8))

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.actor_net(obs), self.critic_net(obs)


def compute_gae(
    rewards: List[float],
    values: List[float],
    gamma: float = 0.97,
    gae_lambda: float = 0.92,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """计算广义优势估计 GAE(gamma, lambda) 与目标回报 Returns."""
    advantages = []
    gae = 0.0
    next_val = 0.0
    for r, v in zip(reversed(rewards), reversed(values)):
        delta = r + gamma * next_val - v
        gae = delta + gamma * gae_lambda * gae
        advantages.insert(0, gae)
        next_val = v
    adv_t = torch.tensor(advantages, dtype=torch.float32)
    val_t = torch.tensor(values, dtype=torch.float32)
    ret_t = adv_t + val_t
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-6)
    return adv_t, ret_t


def train_ppo_portfolio_agent(
    train_env: ASharePortfolioEnv,
    rollout_episodes: int = 30,
    ppo_epochs: int = 4,
    clip_eps: float = 0.20,
    lr: float = 1.5e-3,
    entropy_coef: float = 0.01,
) -> PPOPortfolioActorCritic:
    """使用 PPO Clipped Surrogate Objective 训练 A股动态仓位控制智能体."""
    obs_dim = train_env.observation_space.shape[0]
    act_dim = train_env.action_space.shape[0]
    n_assets = train_env.n_assets

    model = PPOPortfolioActorCritic(obs_dim, act_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    for _ in range(rollout_episodes):
        obs, _ = train_env.reset()
        obs_list, act_list, old_logp_list, rew_list, val_list, prior_list = [], [], [], [], [], []
        done = False

        # 1. 采集完整交易周期轨迹
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            prior_t = build_alpha_risk_prior(obs, n_assets)
            with torch.no_grad():
                mean_delta, val = model(obs_t)
                mean = mean_delta * 0.35 + prior_t
                std = torch.exp(model.log_std)
                dist = torch.distributions.Normal(mean, std)
                act = dist.sample()
                logp = dist.log_prob(act).sum(dim=-1)

            next_obs, reward, done, _, _ = train_env.step(act.squeeze(0).numpy())
            obs_list.append(obs_t)
            act_list.append(act)
            old_logp_list.append(logp)
            rew_list.append(reward)
            val_list.append(float(val.item()))
            prior_list.append(prior_t)
            obs = next_obs

        # 2. 计算 GAE 优势函数
        adv_t, ret_t = compute_gae(rew_list, val_list)
        batch_obs = torch.cat(obs_list, dim=0)
        batch_act = torch.cat(act_list, dim=0)
        batch_old_logp = torch.cat(old_logp_list, dim=0)
        batch_priors = torch.cat(prior_list, dim=0)

        # 3. 多轮 PPO Clipped Surrogate 更新
        for _ in range(ppo_epochs):
            mean_delta, values_pred = model(batch_obs)
            mean = mean_delta * 0.35 + batch_priors
            std = torch.exp(model.log_std)
            dist = torch.distributions.Normal(mean, std)

            new_logp = dist.log_prob(batch_act).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1).mean()

            ratio = torch.exp(new_logp - batch_old_logp)
            surr1 = ratio * adv_t
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_t
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.SmoothL1Loss()(values_pred.squeeze(-1), ret_t)

            loss = policy_loss + 0.5 * value_loss - entropy_coef * entropy
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    return model


def evaluate_ppo_agent(
    model: PPOPortfolioActorCritic,
    test_env: ASharePortfolioEnv,
) -> Dict[str, np.ndarray | float]:
    """在样本外测试集上评估 PPO 智能体，并返回最后一天的实盘目标仓位分配."""
    model.eval()
    obs, _ = test_env.reset()
    done = False
    net_rets, bench_rets, cash_weights = [], [], []
    last_alloc = np.zeros(test_env.n_assets + 1, dtype=np.float32)

    while not done:
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            prior_t = build_alpha_risk_prior(obs, test_env.n_assets)
            mean_delta, _ = model(obs_t)
            act = (mean_delta * 0.35 + prior_t).squeeze(0).numpy()

            # 计算对应 softmax 权重供实盘计划单展示
            logits = np.clip(act, -10.0, 10.0)
            exp_a = np.exp(logits - np.max(logits))
            last_alloc = exp_a / np.sum(exp_a)

        obs, _, done, _, info = test_env.step(act)
        net_rets.append(info["net_ret"])
        bench_rets.append(info["bench_ret"])
        cash_weights.append(info["cash_weight"])

    return {
        "net_rets": np.array(net_rets),
        "bench_rets": np.array(bench_rets),
        "avg_cash_weight": float(np.mean(cash_weights)),
        "final_weights": test_env.weights.copy(),
        "final_cash": float(test_env.cash_weight),
        "latest_target_alloc": last_alloc,
    }
