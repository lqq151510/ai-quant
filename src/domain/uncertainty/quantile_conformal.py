"""领域层新增算法 2：分位数不确定性量化 [q10, q50, q90] 与分裂共形预测校准 (Conformal Quantile Regression)."""

from dataclasses import dataclass
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn


@dataclass
class UncertaintyPrediction:
    """带共形预测校准区间的截面不确定性输出."""
    q10_lower: np.ndarray      # 校准后 10% 悲观下界, shape = (T, N)
    q50_median: np.ndarray     # 50% 中位数预测, shape = (T, N)
    q90_upper: np.ndarray      # 校准后 90% 乐观上界, shape = (T, N)
    interval_width: np.ndarray # 不确定性宽度 (q90 - q10), shape = (T, N)
    robust_alpha: np.ndarray   # 下界风险调整后稳健 Alpha 分数, shape = (T, N)
    conformal_q_hat: float     # 验证集标定的共形校准半径


class PinballQuantileLoss(nn.Module):
    """多分位数弹球损失函数 (Multi-Quantile Pinball Loss for tau in [0.10, 0.50, 0.90])."""

    def __init__(self, quantiles: Tuple[float, ...] = (0.10, 0.50, 0.90)) -> None:
        super().__init__()
        self.register_buffer("quantiles", torch.tensor(quantiles, dtype=torch.float32).view(1, 1, -1))

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算三通道分位数损失与单调性惩罚.

        Args:
            preds: shape = (Batch_days, N_assets, 3) -> [q10, q50, q90]
            target: shape = (Batch_days, N_assets)
        """
        err = target.unsqueeze(-1) - preds
        pinball = torch.maximum(self.quantiles * err, (self.quantiles - 1.0) * err).mean()
        # 强制非交叉约束: q10 <= q50 <= q90
        cross_penalty = (
            torch.relu(preds[..., 0] - preds[..., 1]).mean()
            + torch.relu(preds[..., 1] - preds[..., 2]).mean()
        )
        return pinball + 0.5 * cross_penalty


class QuantileUncertaintyModel(nn.Module):
    """在时空表征之上输出 [q10, q50, q90] 三个分位数的认知不确定性网络."""

    def __init__(self, in_features: int = 8, seq_len: int = 20, d_model: int = 48) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(seq_len * in_features, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )
        # 输出基础中位数 m 以及非负下偏移 d_low, 上偏移 d_high 保证天然不交叉
        self.median_head = nn.Linear(d_model, 1)
        self.spread_head = nn.Sequential(
            nn.Linear(d_model, 2),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """输入 (B, N, L, F) -> 输出 (B, N, 3) 对应 [q10, q50, q90]."""
        b, n, l, f = x.shape
        h = self.proj(x.reshape(b * n, l * f)).view(b, n, -1)
        q50 = self.median_head(h)
        spreads = self.spread_head(h)
        q10 = q50 - spreads[..., 0:1]
        q90 = q50 + spreads[..., 1:2]
        return torch.cat([q10, q50, q90], dim=-1)


def fit_and_calibrate_conformal_quantile(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    eval_x: np.ndarray,
    alpha_miscoverage: float = 0.20,
    epochs: int = 20,
    device: torch.device = torch.device("cpu"),
) -> UncertaintyPrediction:
    """训练分位数回归模型并在验证集上执行 Split Conformal Prediction 校准 (Roman et al., CQR).

    模型拟合与区间校准只使用各自分区中有真实标签的标的。
    最终对 eval_x 中每个标的生成上下界与不确定性宽度。
    """
    train_observed = np.isfinite(train_y)
    val_observed = np.isfinite(val_y)
    if not train_observed.any():
        raise ValueError("CQR requires observed training labels")
    if not val_observed.any():
        raise ValueError("CQR requires observed validation labels")

    _, _, seq_len, n_feats = train_x.shape
    model = QuantileUncertaintyModel(in_features=n_feats, seq_len=seq_len).to(device)
    criterion = PinballQuantileLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)

    # Each asset is encoded independently, so keep only observed asset sequences for fitting.
    tx = torch.tensor(train_x[train_observed][:, None], dtype=torch.float32, device=device)
    ty = torch.tensor(train_y[train_observed][:, None], dtype=torch.float32, device=device)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(tx)
        loss = criterion(out, ty)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_out = model(torch.tensor(val_x[val_observed][:, None], dtype=torch.float32, device=device))
        val_out = val_out[:, 0].cpu().numpy()
        eval_out = model(torch.tensor(eval_x, dtype=torch.float32, device=device)).cpu().numpy()

    # Conformalized Quantile Regression (CQR) 非一致性得分: E = max(q10 - y, y - q90)
    observed_val_y = val_y[val_observed]
    val_scores = np.maximum(val_out[:, 0] - observed_val_y, observed_val_y - val_out[:, 2])
    n_cal = len(val_scores)
    q_level = np.clip(np.ceil((n_cal + 1) * (1.0 - alpha_miscoverage)) / n_cal, 0.5, 0.95)
    q_hat = float(np.quantile(val_scores, q_level))

    q10_cal = eval_out[..., 0] - q_hat
    q50_med = eval_out[..., 1]
    q90_cal = eval_out[..., 2] + q_hat
    width = np.maximum(q90_cal - q10_cal, 1e-4)

    # 下界保护型稳健 Alpha: 重视中位数与悲观下界，惩罚认知不确定性过宽的妖股/闪崩股
    robust_alpha = 0.65 * q50_med + 0.35 * q10_cal - 0.12 * width

    return UncertaintyPrediction(
        q10_lower=q10_cal,
        q50_median=q50_med,
        q90_upper=q90_cal,
        interval_width=width,
        robust_alpha=robust_alpha,
        conformal_q_hat=q_hat,
    )
