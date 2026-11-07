"""应用编排层：严格时序切分训练器、量化指标 (IC/RankIC/ICIR) 评估与 RL 组合训练."""

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
from scipy.stats import spearmanr
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.domain.features.pipeline import SpatioTemporalDataset
from src.domain.rl_portfolio.a_share_env import ASharePortfolioEnv
from src.domain.rl_portfolio.offline_iql import (
    evaluate_hrp_hmm_iql_portfolio,
    train_offline_iql_agent,
)
from src.domain.rl_portfolio.ppo_agent import (
    evaluate_ppo_agent,
    train_ppo_portfolio_agent,
)
from src.domain.timeseries_dl.losses import (
    CrossSectionalICLoss,
    build_exponential_time_decay_weights,
)


@dataclass
class QuantEvalMetrics:
    """量化模型与策略在样本外测试集上的核心评价指标."""
    ic_mean: float
    rank_ic_mean: float
    icir: float
    annual_return: float
    bench_annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    avg_cash_weight: float = 0.0


def get_device() -> torch.device:
    """优先使用 MacBook Apple Silicon MPS 加速."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def compute_ic_metrics(preds: np.ndarray, targets: np.ndarray) -> Tuple[float, float, float]:
    """按每个交易日截面计算 IC 均值、Rank IC 均值与 ICIR (信息比率)."""
    daily_ic = []
    daily_rank_ic = []
    for t in range(len(preds)):
        p, y = preds[t], targets[t]
        if np.std(p) < 1e-8 or np.std(y) < 1e-8:
            continue
        ic = float(np.corrcoef(p, y)[0, 1])
        r_ic, _ = spearmanr(p, y)
        if not np.isnan(ic):
            daily_ic.append(ic)
        if not np.isnan(r_ic):
            daily_rank_ic.append(float(r_ic))
    ic_arr = np.array(daily_ic) if daily_ic else np.zeros(1)
    rank_ic_arr = np.array(daily_rank_ic) if daily_rank_ic else np.zeros(1)
    ic_mean = float(np.mean(ic_arr))
    ic_std = float(np.std(ic_arr)) + 1e-8
    icir = float(ic_mean / ic_std * np.sqrt(252.0))
    return ic_mean, float(np.mean(rank_ic_arr)), icir


def backtest_top_k(
    preds: np.ndarray,
    real_next_rets: np.ndarray,
    top_k: int = 3,
    tx_cost: float = 0.0005,
) -> Tuple[float, float, float, float]:
    """基于模型截面预测分数的 3日平滑 Top-K 轮动回测 (对齐 3日预测周期，降低无效换手摩擦)."""
    n_days, n_assets = preds.shape
    prev_w = np.zeros(n_assets, dtype=np.float32)
    ema_score = np.zeros(n_assets, dtype=np.float32)
    port_rets = []
    bench_rets = []

    for t in range(n_days):
        # 3日指数移动平均平滑信号，与 pred_horizon=3 保持一致
        ema_score = 0.5 * ema_score + 0.5 * preds[t] if t > 0 else preds[t].copy()
        if t % 3 == 0 or t == 0:
            top_idx = np.argsort(ema_score)[-top_k:]
            target_w = np.zeros(n_assets, dtype=np.float32)
            target_w[top_idx] = 1.0 / top_k
        else:
            target_w = prev_w

        turnover = float(np.sum(np.abs(target_w - prev_w)))
        net_r = float(np.dot(target_w, real_next_rets[t])) - turnover * tx_cost
        port_rets.append(net_r)
        bench_rets.append(float(np.mean(real_next_rets[t])))
        prev_w = target_w

    port_arr = np.array(port_rets)
    bench_arr = np.array(bench_rets)
    ann_ret = float((np.prod(1.0 + port_arr) ** (252.0 / max(n_days, 1))) - 1.0)
    bench_ann = float((np.prod(1.0 + bench_arr) ** (252.0 / max(n_days, 1))) - 1.0)
    sharpe = float(np.mean(port_arr) / (np.std(port_arr) + 1e-8) * np.sqrt(252.0))

    nav = np.cumprod(1.0 + port_arr)
    peak = np.maximum.accumulate(nav)
    mdd = float(np.max((peak - nav) / peak))
    return ann_ret, bench_ann, sharpe, mdd


def train_and_evaluate_alpha_model(
    model: nn.Module,
    train_ds: SpatioTemporalDataset,
    val_ds: SpatioTemporalDataset,
    test_ds: SpatioTemporalDataset,
    epochs: int = 25,
    batch_size: int = 32,
    lr: float = 1e-3,
) -> Tuple[QuantEvalMetrics, Dict[str, np.ndarray]]:
    """在严格时序切分的数据集上训练深度学习模型并输出测试集预测."""
    device = get_device()
    model = model.to(device)
    criterion = CrossSectionalICLoss(mse_weight=0.15)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(train_ds.x_seq, dtype=torch.float32),
            torch.tensor(train_ds.y_alpha, dtype=torch.float32),
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    best_val_ic = -1.0
    best_state = None

    for _ in range(epochs):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # 验证集早停检查
        val_preds = _predict_dataset(model, val_ds, device)
        val_ic, _, _ = compute_ic_metrics(val_preds, val_ds.y_alpha)
        if val_ic > best_val_ic:
            best_val_ic = val_ic
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    train_preds = _predict_dataset(model, train_ds, device)
    test_preds = _predict_dataset(model, test_ds, device)

    ic_mean, rank_ic, icir = compute_ic_metrics(test_preds, test_ds.y_alpha)
    ann_ret, bench_ann, sharpe, mdd = backtest_top_k(test_preds, test_ds.y_raw_ret, top_k=3)

    metrics = QuantEvalMetrics(
        ic_mean=ic_mean,
        rank_ic_mean=rank_ic,
        icir=icir,
        annual_return=ann_ret,
        bench_annual_return=bench_ann,
        sharpe_ratio=sharpe,
        max_drawdown=mdd,
    )
    return metrics, {"train_preds": train_preds, "test_preds": test_preds}


def _predict_dataset(model: nn.Module, ds: SpatioTemporalDataset, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        x_t = torch.tensor(ds.x_seq, dtype=torch.float32, device=device)
        return model(x_t).cpu().numpy()


class PolicyGradientPortfolioAgent(nn.Module):
    """Stage 3 轻量级 Actor-Critic 组合权重决策网络.

    接收 Stage 2 输出的时空 Alpha 分数、波动率状态与当前持仓/回撤，
    动态输出 N 只标的 + 1 个现金防守仓位的分配权重。
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.full((1, action_dim), -0.7))
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(obs)
        return self.actor_mean(h), self.critic(h)


def _build_rl_alpha_prior(obs: np.ndarray, n_assets: int) -> torch.Tensor:
    """构建聚焦头部 Alpha 标的并随回撤/波动动态调节现金仓位的先验偏置."""
    alphas = obs[:n_assets].copy()
    vols = obs[n_assets:2 * n_assets]
    current_dd = float(obs[-1])

    # 仅保留截面排名前 3 的强 Alpha 标的，抑制尾部弱势标的分散收益
    threshold = np.sort(alphas)[-3]
    masked_alpha = np.where(alphas >= threshold, alphas * 2.8 + 1.5, -4.0)

    # 当组合回撤扩大或截面平均波动率偏高时，主动抬升现金防守仓位先验
    mean_vol = float(np.mean(vols))
    cash_logit = 1.2 + 12.0 * current_dd + 0.8 * max(0.0, mean_vol)
    prior = np.concatenate([masked_alpha, np.array([cash_logit], dtype=np.float32)])
    return torch.tensor(prior, dtype=torch.float32).unsqueeze(0)


def train_and_eval_rl_portfolio(
    train_alpha: np.ndarray,
    train_ds: SpatioTemporalDataset,
    test_alpha: np.ndarray,
    test_ds: SpatioTemporalDataset,
    episodes: int = 35,
) -> Tuple[QuantEvalMetrics, Dict[str, np.ndarray | float]]:
    """在训练集 Gym 环境上使用完整 PPO + GAE 训练仓位控制智能体，并在样本外测试集评估."""
    train_vol = train_ds.x_seq[:, :, -1, 5]
    test_vol = test_ds.x_seq[:, :, -1, 5]

    train_env = ASharePortfolioEnv(train_alpha, train_vol, train_ds.y_raw_ret)
    test_env = ASharePortfolioEnv(test_alpha, test_vol, test_ds.y_raw_ret)

    ppo_model = train_ppo_portfolio_agent(
        train_env=train_env,
        rollout_episodes=episodes,
        ppo_epochs=4,
        clip_eps=0.20,
    )
    eval_res = evaluate_ppo_agent(ppo_model, test_env)

    port_arr = eval_res["net_rets"]
    bench_arr = eval_res["bench_rets"]
    n_days = len(port_arr)
    ann_ret = float((np.prod(1.0 + port_arr) ** (252.0 / max(n_days, 1))) - 1.0)
    bench_ann = float((np.prod(1.0 + bench_arr) ** (252.0 / max(n_days, 1))) - 1.0)
    sharpe = float(np.mean(port_arr) / (np.std(port_arr) + 1e-8) * np.sqrt(252.0))
    nav = np.cumprod(1.0 + port_arr)
    peak = np.maximum.accumulate(nav)
    mdd = float(np.max((peak - nav) / peak))
    ic_mean, rank_ic, icir = compute_ic_metrics(test_alpha, test_ds.y_alpha)

    metrics = QuantEvalMetrics(
        ic_mean=ic_mean,
        rank_ic_mean=rank_ic,
        icir=icir,
        annual_return=ann_ret,
        bench_annual_return=bench_ann,
        sharpe_ratio=sharpe,
        max_drawdown=mdd,
        avg_cash_weight=float(eval_res["avg_cash_weight"]),
    )
    return metrics, eval_res


def evaluate_predictions_direct(
    test_preds: np.ndarray,
    test_ds: SpatioTemporalDataset,
    top_k: int = 3,
) -> QuantEvalMetrics:
    """对集成/校准后的截面预测矩阵直接计算 IC、RankIC、ICIR 与 Top-K 轮动回测指标."""
    ic_mean, rank_ic, icir = compute_ic_metrics(test_preds, test_ds.y_alpha)
    ann_ret, bench_ann, sharpe, mdd = backtest_top_k(test_preds, test_ds.y_raw_ret, top_k=top_k)
    return QuantEvalMetrics(
        ic_mean=ic_mean,
        rank_ic_mean=rank_ic,
        icir=icir,
        annual_return=ann_ret,
        bench_annual_return=bench_ann,
        sharpe_ratio=sharpe,
        max_drawdown=mdd,
    )


def train_and_eval_iql_hrp_hmm(
    train_alpha: np.ndarray,
    train_ds: SpatioTemporalDataset,
    test_alpha: np.ndarray,
    uncertainty_width: np.ndarray,
    test_ds: SpatioTemporalDataset,
    hmm_max_exposures: list[float],
) -> Tuple[QuantEvalMetrics, Dict[str, np.ndarray | float]]:
    """训练 Offline IQL 智能体，并结合 HRP 层次风险平价与 HMM 市场状态门控在测试集评估."""
    train_vol = train_ds.x_seq[:, :, -1, 5]
    test_vol = test_ds.x_seq[:, :, -1, 5]

    train_env = ASharePortfolioEnv(train_alpha, train_vol, train_ds.y_raw_ret)
    iql_model = train_offline_iql_agent(train_env=train_env, n_behavior_episodes=18, iql_epochs=40)

    eval_res = evaluate_hrp_hmm_iql_portfolio(
        iql_model=iql_model,
        test_alpha=test_alpha,
        uncertainty_width=uncertainty_width,
        train_raw_rets=train_ds.y_raw_ret,
        test_raw_rets=test_ds.y_raw_ret,
        test_vol_features=test_vol,
        hmm_max_exposures=hmm_max_exposures,
    )
    port_arr = eval_res["net_rets"]
    bench_arr = eval_res["bench_rets"]
    n_days = len(port_arr)
    ann_ret = float((np.prod(1.0 + port_arr) ** (252.0 / max(n_days, 1))) - 1.0)
    bench_ann = float((np.prod(1.0 + bench_arr) ** (252.0 / max(n_days, 1))) - 1.0)
    sharpe = float(np.mean(port_arr) / (np.std(port_arr) + 1e-8) * np.sqrt(252.0))
    nav = np.cumprod(1.0 + port_arr)
    peak = np.maximum.accumulate(nav)
    mdd = float(np.max((peak - nav) / peak))
    ic_mean, rank_ic, icir = compute_ic_metrics(test_alpha, test_ds.y_alpha)

    metrics = QuantEvalMetrics(
        ic_mean=ic_mean,
        rank_ic_mean=rank_ic,
        icir=icir,
        annual_return=ann_ret,
        bench_annual_return=bench_ann,
        sharpe_ratio=sharpe,
        max_drawdown=mdd,
        avg_cash_weight=float(eval_res["avg_cash_weight"]),
    )
    return metrics, eval_res
