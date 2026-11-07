"""领域层 Stage 1B：金融量化专属截面排序相关性损失函数 (Differentiable IC Loss + 概念漂移指数时间衰减)."""

from typing import Optional
import numpy as np
import torch
import torch.nn as nn


def build_exponential_time_decay_weights(n_days: int, half_life: float = 200.0) -> np.ndarray:
    """算法升级 C：构造概念漂移自适应的指数时间衰减样本权重.

    数学公式：
    w_t = exp( - ln(2) * (T_train - 1 - t) / tau_{1/2} )
    并归一化使 mean(w) = 1.0，使模型自动降低远古失效风格周期的权重，紧贴近期市场主线。
    """
    age = np.arange(n_days - 1, -1, -1, dtype=np.float32)
    decay = np.exp(-np.log(2.0) * age / max(half_life, 1.0))
    return (decay / (np.mean(decay) + 1e-8)).astype(np.float32)


class CrossSectionalICLoss(nn.Module):
    """混合截面皮尔逊相关系数损失 (1 - IC)、概念漂移指数衰减加权与轻量级 SmoothL1 正则.

    为什么不用纯 MSE？
    金融收益率信噪比极低，纯 MSE 会导致神经网络输出全部坍缩为 0 附近常数。
    直接优化每个交易日截面上的皮尔逊相关系数 (IC)，强迫模型拉开截面多空相对分差。
    """

    def __init__(self, mse_weight: float = 0.15, eps: float = 1e-6) -> None:
        super().__init__()
        self.mse_weight = mse_weight
        self.eps = eps
        self.mse = nn.SmoothL1Loss(reduction="none")

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """计算批次内每个交易日截面的加权 IC Loss.

        Args:
            pred: 预测截面分数, shape = (Batch_days, N_assets)
            target: 真实截面标准化超额收益, shape = (Batch_days, N_assets)
            sample_weights: 可选时间衰减权重, shape = (Batch_days,)
        """
        pred_centered = pred - pred.mean(dim=-1, keepdim=True)
        target_centered = target - target.mean(dim=-1, keepdim=True)

        cov = (pred_centered * target_centered).sum(dim=-1)
        std_pred = torch.sqrt((pred_centered ** 2).sum(dim=-1) + self.eps)
        std_target = torch.sqrt((target_centered ** 2).sum(dim=-1) + self.eps)

        daily_ic = cov / (std_pred * std_target + self.eps)
        daily_reg = self.mse(pred, target).mean(dim=-1)
        daily_total = (1.0 - daily_ic) + self.mse_weight * daily_reg

        if sample_weights is not None:
            w = sample_weights / (sample_weights.sum() + self.eps)
            return (daily_total * w).sum()
        return daily_total.mean()
