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
            target: 真实截面标准化超额收益, shape = (Batch_days, N_assets)，缺失值排除
            sample_weights: 可选时间衰减权重, shape = (Batch_days,)
        """
        observed = torch.isfinite(target)
        count = observed.sum(dim=-1, keepdim=True)
        count_safe = count.clamp_min(1)
        pred_observed = torch.where(observed, pred, 0.0)
        target_observed = torch.where(observed, target, 0.0)

        pred_mean = pred_observed.sum(dim=-1, keepdim=True) / count_safe
        target_mean = target_observed.sum(dim=-1, keepdim=True) / count_safe
        pred_centered = torch.where(observed, pred_observed - pred_mean, 0.0)
        target_centered = torch.where(observed, target_observed - target_mean, 0.0)
        cov = (pred_centered * target_centered).sum(dim=-1)
        pred_ss = pred_centered.square().sum(dim=-1)
        target_ss = target_centered.square().sum(dim=-1)
        daily_ic = cov / (torch.sqrt(pred_ss + self.eps) * torch.sqrt(target_ss + self.eps) + self.eps)
        daily_reg = self.mse(pred_observed, target_observed).sum(dim=-1) / count_safe.squeeze(-1)

        has_observation = count.squeeze(-1) > 0
        # Compare the observed ranges to detect constant sections even with roundoff in the mean.
        pred_varies = pred.masked_fill(~observed, -torch.inf).amax(dim=-1) > pred.masked_fill(~observed, torch.inf).amin(dim=-1)
        target_varies = (
            target.masked_fill(~observed, -torch.inf).amax(dim=-1)
            > target.masked_fill(~observed, torch.inf).amin(dim=-1)
        )
        has_ic = (count.squeeze(-1) > 1) & pred_varies & target_varies
        weights = torch.ones_like(daily_reg) if sample_weights is None else sample_weights
        ic_weights = weights * has_ic
        reg_weights = weights * has_observation
        ic_loss = (torch.where(has_ic, 1.0 - daily_ic, 0.0) * ic_weights).sum()
        reg_loss = (daily_reg * reg_weights).sum()
        if sample_weights is None:
            return ic_loss / ic_weights.sum().clamp_min(1) + self.mse_weight * reg_loss / reg_weights.sum().clamp_min(1)
        return ic_loss / (ic_weights.sum() + self.eps) + self.mse_weight * reg_loss / (reg_weights.sum() + self.eps)
